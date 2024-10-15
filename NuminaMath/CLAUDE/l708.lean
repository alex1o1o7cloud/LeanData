import Mathlib

namespace NUMINAMATH_CALUDE_evergreen_marching_band_max_size_l708_70885

theorem evergreen_marching_band_max_size :
  ∃ (n : ℕ),
    (∀ k : ℕ, 15 * k < 800 → 15 * k ≤ 15 * n) ∧
    (15 * n < 800) ∧
    (15 * n % 19 = 2) ∧
    (15 * n = 750) := by
  sorry

end NUMINAMATH_CALUDE_evergreen_marching_band_max_size_l708_70885


namespace NUMINAMATH_CALUDE_largest_prime_2010_digits_divisibility_l708_70875

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_2010_digits (n : ℕ) : Prop := 10^2009 ≤ n ∧ n < 10^2010

def largest_prime_with_2010_digits (p : ℕ) : Prop :=
  is_prime p ∧ has_2010_digits p ∧ ∀ q : ℕ, is_prime q → has_2010_digits q → q ≤ p

theorem largest_prime_2010_digits_divisibility (p : ℕ) 
  (h : largest_prime_with_2010_digits p) : 
  12 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_2010_digits_divisibility_l708_70875


namespace NUMINAMATH_CALUDE_group_size_calculation_l708_70879

theorem group_size_calculation (n : ℕ) 
  (h1 : n * (40 - 3) = n * 40 - 40 + 10) : n = 10 := by
  sorry

#check group_size_calculation

end NUMINAMATH_CALUDE_group_size_calculation_l708_70879


namespace NUMINAMATH_CALUDE_perimeter_difference_l708_70887

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of a modified rectangle with a vertical shift --/
def modifiedRectanglePerimeter (length width shift : ℕ) : ℕ :=
  2 * length + 2 * width + 2 * shift

/-- The positive difference between the perimeter of a 6x1 rectangle with a vertical shift
    and the perimeter of a 4x1 rectangle is 6 units --/
theorem perimeter_difference : 
  modifiedRectanglePerimeter 6 1 1 - rectanglePerimeter 4 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l708_70887


namespace NUMINAMATH_CALUDE_book_env_intersection_l708_70819

/-- The number of people participating in both Book Club and Environmental Theme Painting --/
def intersection_book_env (total participants : ℕ) 
  (book_club fun_sports env_painting : ℕ) 
  (book_fun fun_env : ℕ) : ℕ :=
  book_club + fun_sports + env_painting - total - book_fun - fun_env

/-- Theorem stating the number of people participating in both Book Club and Environmental Theme Painting --/
theorem book_env_intersection : 
  ∀ (total participants : ℕ) 
    (book_club fun_sports env_painting : ℕ) 
    (book_fun fun_env : ℕ),
  total = 120 →
  book_club = 80 →
  fun_sports = 50 →
  env_painting = 40 →
  book_fun = 20 →
  fun_env = 10 →
  intersection_book_env total participants book_club fun_sports env_painting book_fun fun_env = 20 := by
  sorry

#check book_env_intersection

end NUMINAMATH_CALUDE_book_env_intersection_l708_70819


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l708_70807

theorem quadratic_roots_sum_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (8 * x₁^2 + 2 * k * x₁ + k - 1 = 0) ∧ 
    (8 * x₂^2 + 2 * k * x₂ + k - 1 = 0) ∧ 
    (x₁^2 + x₂^2 = 1) ∧
    (4 * k^2 - 32 * (k - 1) ≥ 0)) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l708_70807


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_and_cubes_l708_70854

theorem consecutive_integers_sum_of_squares_and_cubes :
  ∀ n : ℤ,
  (n - 1)^2 + n^2 + (n + 1)^2 = 8450 →
  n = 53 ∧ (n - 1)^3 + n^3 + (n + 1)^3 = 446949 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_and_cubes_l708_70854


namespace NUMINAMATH_CALUDE_total_lemon_heads_eaten_l708_70843

/-- The number of Lemon Heads in each package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- Theorem: Given the conditions, Louis ate 54 Lemon Heads in total -/
theorem total_lemon_heads_eaten :
  lemon_heads_per_package * boxes_finished = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_lemon_heads_eaten_l708_70843


namespace NUMINAMATH_CALUDE_exists_unsolvable_chessboard_l708_70806

/-- Represents a 12x12 chessboard where each square can be black or white -/
def Chessboard := Fin 12 → Fin 12 → Bool

/-- Represents a row or column flip operation -/
inductive FlipOperation
| row (i : Fin 12)
| col (j : Fin 12)

/-- Applies a flip operation to a chessboard -/
def applyFlip (board : Chessboard) (op : FlipOperation) : Chessboard :=
  match op with
  | FlipOperation.row i => fun x y => if x = i then !board x y else board x y
  | FlipOperation.col j => fun x y => if y = j then !board x y else board x y

/-- Checks if all squares on the board are black -/
def allBlack (board : Chessboard) : Prop :=
  ∀ i j, board i j = true

/-- Theorem: There exists an initial chessboard configuration that cannot be made all black -/
theorem exists_unsolvable_chessboard : 
  ∃ (initial : Chessboard), ¬∃ (ops : List FlipOperation), allBlack (ops.foldl applyFlip initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unsolvable_chessboard_l708_70806


namespace NUMINAMATH_CALUDE_stock_price_change_l708_70834

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let day1 := initial_price * (1 - 0.25)
  let day2 := day1 * (1 + 0.40)
  let day3 := day2 * (1 - 0.10)
  (day3 - initial_price) / initial_price = -0.055 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l708_70834


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l708_70842

theorem fraction_to_decimal : 19 / (2^2 * 5^3) = 0.095 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l708_70842


namespace NUMINAMATH_CALUDE_fraction_c_simplest_form_l708_70894

def is_simplest_form (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ≠ 0 → k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

theorem fraction_c_simplest_form (x y : ℤ) (hx : x ≠ 0) :
  is_simplest_form (x + y) (2 * x) :=
sorry

end NUMINAMATH_CALUDE_fraction_c_simplest_form_l708_70894


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l708_70828

/-- Represents the coin distribution between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  -- Paul's final coin count
  let paul_coins := x
  -- Pete's coin count using the sum formula
  let pete_coins := x * (x + 1) / 2
  -- The condition that Pete has 5 times as many coins as Paul
  pete_coins = 5 * paul_coins

/-- The total number of coins distributed -/
def total_coins (x : ℕ) : ℕ := 6 * x

theorem coin_distribution_theorem :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l708_70828


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l708_70866

-- System 1
theorem system_one_solution : 
  ∃ (x y : ℝ), y = x - 4 ∧ x + y = 6 ∧ x = 5 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℝ), 2*x + y = 1 ∧ 4*x - y = 5 ∧ x = 1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l708_70866


namespace NUMINAMATH_CALUDE_positive_test_probability_l708_70862

/-- Probability of a positive test result given the disease prevalence and test characteristics -/
theorem positive_test_probability
  (P_A : ℝ)
  (P_B_given_A : ℝ)
  (P_B_given_not_A : ℝ)
  (h1 : P_A = 0.01)
  (h2 : P_B_given_A = 0.99)
  (h3 : P_B_given_not_A = 0.1)
  (h4 : ∀ (P_A P_B_given_A P_B_given_not_A : ℝ),
    P_A ≥ 0 ∧ P_A ≤ 1 →
    P_B_given_A ≥ 0 ∧ P_B_given_A ≤ 1 →
    P_B_given_not_A ≥ 0 ∧ P_B_given_not_A ≤ 1 →
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≥ 0 ∧
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≤ 1) :
  P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) = 0.1089 := by
  sorry


end NUMINAMATH_CALUDE_positive_test_probability_l708_70862


namespace NUMINAMATH_CALUDE_sqrt_2_pow_12_l708_70851

theorem sqrt_2_pow_12 : Real.sqrt (2^12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_pow_12_l708_70851


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l708_70825

/-- A quadratic function f(x) = ax² + bx satisfying specific conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ (x : ℝ), f a b x = x ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  (∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
  (Set.Icc 0 3).image (f a b) = Set.Icc (-3/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l708_70825


namespace NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l708_70805

theorem polynomial_expansion_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
            a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a₈ = 45 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_coefficient_l708_70805


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l708_70839

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l708_70839


namespace NUMINAMATH_CALUDE_prime_triplets_l708_70893

def is_prime_triplet (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  Nat.Prime (a - b - 8) ∧ Nat.Prime (b - c - 8)

theorem prime_triplets :
  ∀ a b c : ℕ, is_prime_triplet a b c ↔ (a = 23 ∧ b = 13 ∧ (c = 2 ∨ c = 3)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplets_l708_70893


namespace NUMINAMATH_CALUDE_least_possible_radios_l708_70860

theorem least_possible_radios (n d : ℕ) (h1 : d > 0) : 
  (d + 8 * n - 16 - d = 72) → (∃ (m : ℕ), m ≥ n ∧ m ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_least_possible_radios_l708_70860


namespace NUMINAMATH_CALUDE_kennel_long_furred_dogs_l708_70855

/-- Represents the number of dogs with a certain property in a kennel -/
structure DogCount where
  total : ℕ
  brown : ℕ
  neither_long_nor_brown : ℕ

/-- Calculates the number of long-furred dogs in the kennel -/
def long_furred_dogs (d : DogCount) : ℕ :=
  d.total - d.neither_long_nor_brown - d.brown

/-- Theorem stating that in a kennel with the given properties, there are 10 long-furred dogs -/
theorem kennel_long_furred_dogs :
  let d : DogCount := ⟨45, 27, 8⟩
  long_furred_dogs d = 10 := by
  sorry

#eval long_furred_dogs ⟨45, 27, 8⟩

end NUMINAMATH_CALUDE_kennel_long_furred_dogs_l708_70855


namespace NUMINAMATH_CALUDE_distributions_five_balls_four_boxes_l708_70876

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def totalDistributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with the condition that one specific box must contain at least one ball -/
def distributionsWithConstraint (n k : ℕ) : ℕ :=
  totalDistributions n k - totalDistributions n (k - 1)

theorem distributions_five_balls_four_boxes :
  distributionsWithConstraint 5 4 = 781 := by
  sorry

end NUMINAMATH_CALUDE_distributions_five_balls_four_boxes_l708_70876


namespace NUMINAMATH_CALUDE_reading_activity_results_l708_70823

def characters_per_day : ℕ := 850
def days_per_week : ℕ := 7
def total_weeks : ℕ := 20

def characters_per_week : ℕ := characters_per_day * days_per_week
def total_characters : ℕ := characters_per_week * total_weeks

def approximate_ten_thousands (n : ℕ) : ℕ :=
  (n + 5000) / 10000

theorem reading_activity_results :
  characters_per_week = 5950 ∧
  total_characters = 119000 ∧
  approximate_ten_thousands total_characters = 12 :=
by sorry

end NUMINAMATH_CALUDE_reading_activity_results_l708_70823


namespace NUMINAMATH_CALUDE_gcd_1729_867_l708_70890

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l708_70890


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l708_70874

theorem absolute_value_inequality (x : ℝ) :
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l708_70874


namespace NUMINAMATH_CALUDE_cos_165_degrees_l708_70867

theorem cos_165_degrees : Real.cos (165 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_165_degrees_l708_70867


namespace NUMINAMATH_CALUDE_stock_price_calculation_l708_70824

/-- Calculates the price of a stock given investment details -/
theorem stock_price_calculation 
  (investment : ℝ) 
  (dividend_rate : ℝ) 
  (annual_income : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800)
  (h2 : dividend_rate = 0.20)
  (h3 : annual_income = 1000)
  (h4 : face_value = 100) : 
  (investment / (annual_income / dividend_rate)) * face_value = 136 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l708_70824


namespace NUMINAMATH_CALUDE_blue_balls_count_l708_70844

theorem blue_balls_count (total : ℕ) (red : ℕ) (orange : ℕ) (pink : ℕ) 
  (h1 : total = 50)
  (h2 : red = 20)
  (h3 : orange = 5)
  (h4 : pink = 3 * orange)
  : total - (red + orange + pink) = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l708_70844


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l708_70835

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l708_70835


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l708_70852

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (5 * a^3 - 2019 * a + 4029 = 0) →
  (5 * b^3 - 2019 * b + 4029 = 0) →
  (5 * c^3 - 2019 * c + 4029 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l708_70852


namespace NUMINAMATH_CALUDE_sequence_properties_l708_70873

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_properties :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l708_70873


namespace NUMINAMATH_CALUDE_weekend_newspaper_delivery_l708_70814

/-- The total number of newspapers delivered on the weekend -/
def total_newspapers (saturday_papers sunday_papers : ℕ) : ℕ :=
  saturday_papers + sunday_papers

/-- Theorem: The total number of newspapers delivered on the weekend is 110 -/
theorem weekend_newspaper_delivery : total_newspapers 45 65 = 110 := by
  sorry

end NUMINAMATH_CALUDE_weekend_newspaper_delivery_l708_70814


namespace NUMINAMATH_CALUDE_sum_of_digits_of_b_is_nine_l708_70845

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1995 digits -/
def has1995Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_b_is_nine (N : ℕ) 
  (h1 : has1995Digits N) 
  (h2 : N % 9 = 0) : 
  let a := sumOfDigits N
  let b := sumOfDigits a
  sumOfDigits b = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_b_is_nine_l708_70845


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l708_70853

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section represented by the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 64y^2 - 12x + 16y + 36 = 0 represents a hyperbola --/
theorem equation_represents_hyperbola :
  determineConicSection 1 (-64) 0 (-12) 16 36 = ConicSection.Hyperbola :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l708_70853


namespace NUMINAMATH_CALUDE_inequality_solution_set_l708_70837

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (3 - x) ≤ 1 ↔ x > 3 ∨ x ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l708_70837


namespace NUMINAMATH_CALUDE_curve_self_intersection_l708_70884

-- Define the parametric equations
def x (t : ℝ) : ℝ := t^2 + 3
def y (t : ℝ) : ℝ := t^3 - 6*t + 4

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 9 ∧ y a = 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l708_70884


namespace NUMINAMATH_CALUDE_evaluate_expression_l708_70817

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l708_70817


namespace NUMINAMATH_CALUDE_a_range_l708_70878

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f is increasing on [1, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

theorem a_range (a : ℝ) (h : f_increasing a) : a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l708_70878


namespace NUMINAMATH_CALUDE_equation_positive_root_l708_70849

theorem equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l708_70849


namespace NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l708_70865

theorem sqrt_twelve_equals_two_sqrt_three :
  Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l708_70865


namespace NUMINAMATH_CALUDE_regular_time_limit_proof_l708_70859

/-- Represents the regular time limit in hours -/
def regular_time_limit : ℕ := 40

/-- Regular pay rate in dollars per hour -/
def regular_pay_rate : ℕ := 3

/-- Overtime pay rate in dollars per hour -/
def overtime_pay_rate : ℕ := 2 * regular_pay_rate

/-- Total pay received in dollars -/
def total_pay : ℕ := 192

/-- Overtime hours worked -/
def overtime_hours : ℕ := 12

theorem regular_time_limit_proof :
  regular_time_limit * regular_pay_rate + overtime_hours * overtime_pay_rate = total_pay :=
by sorry

end NUMINAMATH_CALUDE_regular_time_limit_proof_l708_70859


namespace NUMINAMATH_CALUDE_first_round_score_l708_70827

def card_values : List ℕ := [2, 4, 7, 13]

theorem first_round_score (total_score : ℕ) (last_round_score : ℕ) 
  (h1 : total_score = 16)
  (h2 : last_round_score = 2)
  (h3 : card_values.sum = 26)
  (h4 : ∃ (n : ℕ), n * card_values.sum = 16 + 17 + 21 + 24)
  : ∃ (first_round_score : ℕ), 
    first_round_score ∈ card_values ∧ 
    ∃ (second_round_score : ℕ), 
      second_round_score ∈ card_values ∧ 
      first_round_score + second_round_score + last_round_score = total_score ∧
      first_round_score = 7 :=
by
  sorry

#check first_round_score

end NUMINAMATH_CALUDE_first_round_score_l708_70827


namespace NUMINAMATH_CALUDE_rd_funding_exceeds_2_million_l708_70869

/-- R&D funding function -/
def rd_funding (x : ℕ) : ℝ := 1.3 * (1 + 0.12)^x

/-- Year when funding exceeds 2 million -/
def exceed_year : ℕ := 4

theorem rd_funding_exceeds_2_million : 
  rd_funding exceed_year > 2 ∧ 
  ∀ y : ℕ, y < exceed_year → rd_funding y ≤ 2 := by
  sorry

#eval exceed_year + 2015

end NUMINAMATH_CALUDE_rd_funding_exceeds_2_million_l708_70869


namespace NUMINAMATH_CALUDE_maximize_negative_products_l708_70801

theorem maximize_negative_products (n : ℕ) (h : n > 0) :
  let f : ℕ → ℕ := λ k => k * (n - k)
  let max_k : ℕ := if n % 2 = 0 then n / 2 else (n - 1) / 2
  ∀ k, k ≤ n → f k ≤ f max_k ∧
    (n % 2 ≠ 0 → f k ≤ f ((n + 1) / 2)) :=
by sorry


end NUMINAMATH_CALUDE_maximize_negative_products_l708_70801


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l708_70868

def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

theorem magnitude_of_vector_sum :
  ‖a + 3 • b‖ = Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l708_70868


namespace NUMINAMATH_CALUDE_max_integers_satisfying_inequalities_l708_70881

theorem max_integers_satisfying_inequalities :
  (∃ x : ℕ, x = 7 ∧ 50 * x < 360 ∧ ∀ y : ℕ, 50 * y < 360 → y ≤ x) ∧
  (∃ y : ℕ, y = 4 ∧ 80 * y < 352 ∧ ∀ z : ℕ, 80 * z < 352 → z ≤ y) ∧
  (∃ z : ℕ, z = 6 ∧ 70 * z < 424 ∧ ∀ w : ℕ, 70 * w < 424 → w ≤ z) ∧
  (∃ w : ℕ, w = 4 ∧ 60 * w < 245 ∧ ∀ v : ℕ, 60 * v < 245 → v ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_satisfying_inequalities_l708_70881


namespace NUMINAMATH_CALUDE_money_problem_l708_70872

theorem money_problem (c d : ℝ) (h1 : 7 * c + d > 84) (h2 : 5 * c - d = 35) :
  c > 9.92 ∧ d > 14.58 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l708_70872


namespace NUMINAMATH_CALUDE_solution_set_l708_70838

theorem solution_set (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | a^(2*x - 7) > a^(4*x - 1)} = {x : ℝ | x > -3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l708_70838


namespace NUMINAMATH_CALUDE_sallys_score_l708_70858

/-- Calculates the score for a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Proves that Sally's score in the math contest is 12.5 -/
theorem sallys_score :
  let correct := 15
  let incorrect := 10
  let unanswered := 5
  calculate_score correct incorrect unanswered = 12.5 := by
  sorry

#eval calculate_score 15 10 5

end NUMINAMATH_CALUDE_sallys_score_l708_70858


namespace NUMINAMATH_CALUDE_arithmetic_mean_4_16_l708_70800

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_4_16_l708_70800


namespace NUMINAMATH_CALUDE_system_solutions_l708_70816

/-- The system of equations -/
def system (p x y : ℝ) : Prop :=
  p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1

/-- The system has at least three different real solutions -/
def has_three_solutions (p : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    system p x₁ y₁ ∧ 
    system p x₂ y₂ ∧ 
    system p x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

/-- The main theorem -/
theorem system_solutions :
  ∀ p : ℝ, has_three_solutions p ↔ p = 1 ∨ p = -1 :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l708_70816


namespace NUMINAMATH_CALUDE_correct_statements_l708_70870

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| GeneralToGeneral
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific
| SpecificToGeneral

-- Define a function to describe the correct direction for each reasoning type
def correct_direction (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Define the statements
def statement (n : Nat) : ReasoningType × ReasoningDirection :=
  match n with
  | 1 => (ReasoningType.Inductive, ReasoningDirection.GeneralToGeneral)
  | 2 => (ReasoningType.Inductive, ReasoningDirection.PartToWhole)
  | 3 => (ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific)
  | 4 => (ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific)
  | 5 => (ReasoningType.Analogical, ReasoningDirection.SpecificToGeneral)
  | _ => (ReasoningType.Inductive, ReasoningDirection.PartToWhole) -- Default case

-- Define a function to check if a statement is correct
def is_correct (n : Nat) : Prop :=
  let (rt, rd) := statement n
  rd = correct_direction rt

-- Theorem stating that statements 2, 3, and 4 are the correct ones
theorem correct_statements :
  (is_correct 2 ∧ is_correct 3 ∧ is_correct 4) ∧
  (∀ n, n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 → ¬is_correct n) :=
sorry


end NUMINAMATH_CALUDE_correct_statements_l708_70870


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l708_70883

theorem chocolate_bars_count (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 21) 
  (h2 : bars_per_box = 25) : 
  small_boxes * bars_per_box = 525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l708_70883


namespace NUMINAMATH_CALUDE_sequence_general_term_l708_70802

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val ^ 2 + 8 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℚ := 6 * n.val + 5

/-- Theorem stating that the given general term formula is correct for the sequence -/
theorem sequence_general_term (n : ℕ+) : a n = S n - S (n - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l708_70802


namespace NUMINAMATH_CALUDE_rectangular_field_area_l708_70833

theorem rectangular_field_area (length width area : ℝ) : 
  length = width + 10 →
  length = 19.13 →
  area = length * width →
  area = 174.6359 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l708_70833


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_empty_solution_l708_70880

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Theorem for the range of a when f(x) < a has no solutions
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, f x ≥ a} = {a : ℝ | a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_empty_solution_l708_70880


namespace NUMINAMATH_CALUDE_largest_operation_l708_70848

theorem largest_operation : ∀ a b c d e : ℝ,
  a = 15432 + 1 / 3241 →
  b = 15432 - 1 / 3241 →
  c = 15432 * (1 / 3241) →
  d = 15432 / (1 / 3241) →
  e = 15432.3241 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_operation_l708_70848


namespace NUMINAMATH_CALUDE_left_seats_count_l708_70811

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 11 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 92 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_left_seats_count_l708_70811


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l708_70861

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean_2_6 : (a 2 + a 6) / 2 = 5)
  (h_mean_3_7 : (a 3 + a 7) / 2 = 7) :
  ∃ f : ℕ → ℝ, (∀ n, a n = f n) ∧ (∀ n, f n = 2 * n - 3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l708_70861


namespace NUMINAMATH_CALUDE_equation_solutions_l708_70813

theorem equation_solutions :
  (∃ x : ℝ, x * (x + 10) = -9 ↔ x = -9 ∨ x = -1) ∧
  (∃ x : ℝ, x * (2 * x + 3) = 8 * x + 12 ↔ x = -3/2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l708_70813


namespace NUMINAMATH_CALUDE_complex_roots_distance_l708_70898

/-- Given three complex numbers z₁, z₂, z₃ with |zⱼ| ≤ 1 for j = 1, 2, 3, 
    and w₁, w₂ being the roots of the equation 
    (z - z₁)(z - z₂) + (z - z₂)(z - z₃) + (z - z₃)(z - z₁) = 0,
    then for j = 1, 2, 3, min{|zⱼ - w₁|, |zⱼ - w₂|} ≤ 1. -/
theorem complex_roots_distance (z₁ z₂ z₃ w₁ w₂ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 1)
  (h₃ : Complex.abs z₃ ≤ 1)
  (hw : (w₁ - z₁) * (w₁ - z₂) + (w₁ - z₂) * (w₁ - z₃) + (w₁ - z₃) * (w₁ - z₁) = 0 ∧
        (w₂ - z₁) * (w₂ - z₂) + (w₂ - z₂) * (w₂ - z₃) + (w₂ - z₃) * (w₂ - z₁) = 0) :
  (min (Complex.abs (z₁ - w₁)) (Complex.abs (z₁ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₂ - w₁)) (Complex.abs (z₂ - w₂)) ≤ 1) ∧
  (min (Complex.abs (z₃ - w₁)) (Complex.abs (z₃ - w₂)) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_distance_l708_70898


namespace NUMINAMATH_CALUDE_big_bonsai_cost_l708_70847

/-- Represents the cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- Represents the number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- Represents the number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 190

/-- Proves that the cost of a big bonsai is $20 -/
theorem big_bonsai_cost : 
  ∃ (big_bonsai_cost : ℕ), 
    small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings ∧ 
    big_bonsai_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_big_bonsai_cost_l708_70847


namespace NUMINAMATH_CALUDE_steak_per_member_l708_70822

theorem steak_per_member (family_members : ℕ) (steak_size : ℕ) (steaks_needed : ℕ) :
  family_members = 5 →
  steak_size = 20 →
  steaks_needed = 4 →
  (steaks_needed * steak_size) / family_members = 16 := by
sorry

end NUMINAMATH_CALUDE_steak_per_member_l708_70822


namespace NUMINAMATH_CALUDE_square_of_five_times_sqrt_three_l708_70803

theorem square_of_five_times_sqrt_three : (5 * Real.sqrt 3) ^ 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_square_of_five_times_sqrt_three_l708_70803


namespace NUMINAMATH_CALUDE_rugby_league_matches_l708_70857

/-- The number of matches played in a rugby league -/
def total_matches (n : ℕ) (k : ℕ) : ℕ :=
  k * (n.choose 2)

/-- Theorem: In a league with 10 teams, where each team plays against every other team exactly 4 times, the total number of matches played is 180. -/
theorem rugby_league_matches :
  total_matches 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_rugby_league_matches_l708_70857


namespace NUMINAMATH_CALUDE_betty_picked_15_oranges_l708_70840

def orange_problem (betty_oranges : ℕ) : Prop :=
  let bill_oranges : ℕ := 12
  let frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)
  let seeds_planted : ℕ := 2 * frank_oranges
  let trees_grown : ℕ := seeds_planted
  let oranges_per_tree : ℕ := 5
  let total_oranges : ℕ := trees_grown * oranges_per_tree
  total_oranges = 810

theorem betty_picked_15_oranges :
  ∃ (betty_oranges : ℕ), orange_problem betty_oranges ∧ betty_oranges = 15 :=
by sorry

end NUMINAMATH_CALUDE_betty_picked_15_oranges_l708_70840


namespace NUMINAMATH_CALUDE_selection_theorem_l708_70850

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 3

theorem selection_theorem :
  (choose total_people num_representatives = 35) ∧
  (choose num_girls 1 * choose num_boys 2 +
   choose num_girls 2 * choose num_boys 1 +
   choose num_girls 3 = 31) ∧
  (choose total_people num_representatives -
   choose num_boys num_representatives -
   choose num_girls num_representatives = 30) := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l708_70850


namespace NUMINAMATH_CALUDE_oviparous_produces_significant_differences_l708_70877

-- Define the modes of reproduction
inductive ReproductionMode
  | Vegetative
  | Oviparous
  | Fission
  | Budding

-- Define the reproduction categories
inductive ReproductionCategory
  | Sexual
  | Asexual

-- Define a function to categorize reproduction modes
def categorizeReproduction : ReproductionMode → ReproductionCategory
  | ReproductionMode.Vegetative => ReproductionCategory.Asexual
  | ReproductionMode.Oviparous => ReproductionCategory.Sexual
  | ReproductionMode.Fission => ReproductionCategory.Asexual
  | ReproductionMode.Budding => ReproductionCategory.Asexual

-- Define a property for producing offspring with significant differences
def produceSignificantDifferences (mode : ReproductionMode) : Prop :=
  categorizeReproduction mode = ReproductionCategory.Sexual

theorem oviparous_produces_significant_differences :
  ∀ (mode : ReproductionMode),
    produceSignificantDifferences mode ↔ mode = ReproductionMode.Oviparous :=
by sorry

end NUMINAMATH_CALUDE_oviparous_produces_significant_differences_l708_70877


namespace NUMINAMATH_CALUDE_yellow_flowers_count_l708_70818

/-- Represents the number of flowers of each color in the garden -/
structure FlowerCounts where
  total : Nat
  green : Nat
  red : Nat
  blue : Nat
  yellow : Nat

/-- The conditions of the flower garden problem -/
def gardenConditions : FlowerCounts → Prop := fun c =>
  c.total = 96 ∧
  c.green = 9 ∧
  c.red = 3 * c.green ∧
  c.blue = c.total / 2 ∧
  c.yellow = c.total - c.green - c.red - c.blue

/-- Theorem stating that under the given conditions, there are 12 yellow flowers -/
theorem yellow_flowers_count (c : FlowerCounts) : 
  gardenConditions c → c.yellow = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_flowers_count_l708_70818


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l708_70895

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  lastInningsScore : ℕ
  averageIncrease : ℚ
  notOutCount : ℕ

/-- Calculates the average score after a given number of innings -/
def averageAfterInnings (performance : BatsmanPerformance) : ℚ :=
  sorry

theorem batsman_average_after_12th_innings 
  (performance : BatsmanPerformance)
  (h_innings : performance.innings = 12)
  (h_lastScore : performance.lastInningsScore = 60)
  (h_avgIncrease : performance.averageIncrease = 2)
  (h_notOut : performance.notOutCount = 0) :
  averageAfterInnings performance = 38 :=
sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l708_70895


namespace NUMINAMATH_CALUDE_combined_job_time_l708_70812

def job_time_A : ℝ := 8
def job_time_B : ℝ := 12

theorem combined_job_time : 
  let rate_A := 1 / job_time_A
  let rate_B := 1 / job_time_B
  let combined_rate := rate_A + rate_B
  1 / combined_rate = 4.8 := by sorry

end NUMINAMATH_CALUDE_combined_job_time_l708_70812


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_roots_and_point_l708_70826

/-- Given a quadratic polynomial q(x) with roots at x = -2 and x = 3, and q(1) = -10,
    prove that q(x) = 5/3x^2 - 5/3x - 10 -/
theorem quadratic_polynomial_from_roots_and_point (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = -2 ∨ x = 3) →  -- roots at x = -2 and x = 3
  (∃ a b c, ∀ x, q x = a * x^2 + b * x + c) →  -- q is a quadratic polynomial
  q 1 = -10 →  -- q(1) = -10
  ∀ x, q x = 5/3 * x^2 - 5/3 * x - 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_roots_and_point_l708_70826


namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l708_70832

/-- Calculates the cost per use of a heating pad. -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * num_weeks)

/-- Theorem stating that a $30 heating pad used 3 times a week for 2 weeks costs $5 per use. -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l708_70832


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l708_70841

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem bouncing_ball_distance :
  totalDistance 200 (2/3) 4 = 4200 :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l708_70841


namespace NUMINAMATH_CALUDE_solution_l708_70829

def complex_number_problem (z : ℂ) : Prop :=
  (∃ (r : ℝ), z - 3 * Complex.I = r) ∧
  (∃ (t : ℝ), (z - 5 * Complex.I) / (2 - Complex.I) = t * Complex.I)

theorem solution (z : ℂ) (h : complex_number_problem z) :
  z = -1 + 3 * Complex.I ∧ Complex.abs (z / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_l708_70829


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l708_70856

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 1| - 5

-- Theorem for part 1
theorem solve_inequality (a : ℝ) (ha : a ≠ 0) (h2 : f 2 a = 0) :
  (a = 4 → ∀ x, f x a ≤ 10 ↔ -10/3 ≤ x ∧ x ≤ 20/3) ∧
  (a = -4 → ∀ x, f x a ≤ 10 ↔ -6 ≤ x ∧ x ≤ 4) :=
sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) (ha : a < 0) 
  (h_triangle : ∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ a = 0 ∧ f x₂ a < 0 ∧ f x₃ a = 0) :
  -3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l708_70856


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l708_70804

/-- In a quadrilateral EFGH where ∠E = 3∠F = 4∠G = 6∠H, the measure of ∠E is 360 * (4/7) degrees. -/
theorem angle_measure_in_special_quadrilateral :
  ∀ (E F G H : ℝ),
  E + F + G + H = 360 →
  E = 3 * F →
  E = 4 * G →
  E = 6 * H →
  E = 360 * (4/7) := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l708_70804


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l708_70896

/-- A quadratic function passing through (-1,0) and (3,0) with a minimum value of 28 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  passes_through_three : a * 3^2 + b * 3 + c = 0
  min_value : ∃ (x : ℝ), ∀ (y : ℝ), a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 28

/-- The sum of coefficients of the quadratic function is 28 -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l708_70896


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l708_70809

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  a₃ * a₅ * a₇ * a₉ * a₁₁ = 243 → a₁₀^2 / a₁₃ = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l708_70809


namespace NUMINAMATH_CALUDE_complexity_power_of_two_no_complexity_less_than_n_l708_70830

-- Define complexity of an integer
def complexity (n : ℕ) : ℕ := sorry

-- Theorem for part (a)
theorem complexity_power_of_two (k : ℕ) :
  ∀ m : ℕ, 2^k ≤ m → m < 2^(k+1) → complexity m ≤ k := by sorry

-- Theorem for part (b)
theorem no_complexity_less_than_n :
  ∀ n : ℕ, n > 1 → ∃ m : ℕ, n ≤ m → m < 2*n → complexity m ≥ complexity n := by sorry

end NUMINAMATH_CALUDE_complexity_power_of_two_no_complexity_less_than_n_l708_70830


namespace NUMINAMATH_CALUDE_tinas_trip_distance_l708_70821

/-- Tina's trip consists of three parts: highway, city, and rural roads. 
    This theorem proves that the total distance of her trip is 120 miles. -/
theorem tinas_trip_distance : ℝ → Prop :=
  fun total_distance =>
    (total_distance / 2 + 30 + total_distance / 4 = total_distance) →
    total_distance = 120

/-- Proof of the theorem -/
lemma prove_tinas_trip_distance : tinas_trip_distance 120 := by
  sorry

end NUMINAMATH_CALUDE_tinas_trip_distance_l708_70821


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l708_70886

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 50
  let y : ℝ := 12 + Real.sqrt 50
  x + y = 24 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l708_70886


namespace NUMINAMATH_CALUDE_digit_equation_solution_l708_70815

theorem digit_equation_solution : ∃! (Θ : ℕ), Θ > 0 ∧ Θ < 10 ∧ (476 : ℚ) / Θ = 50 + 4 * Θ :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l708_70815


namespace NUMINAMATH_CALUDE_base6_addition_problem_l708_70846

/-- Represents a digit in base 6 -/
def Base6Digit := Fin 6

/-- Checks if three Base6Digits are distinct -/
def are_distinct (s h e : Base6Digit) : Prop :=
  s ≠ h ∧ s ≠ e ∧ h ≠ e

/-- Converts a natural number to its base 6 representation -/
def to_base6 (n : ℕ) : ℕ :=
  sorry

/-- Adds two base 6 numbers -/
def base6_add (a b : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem base6_addition_problem :
  ∃ (s h e : Base6Digit),
    are_distinct s h e ∧
    0 < s.val ∧ 0 < h.val ∧ 0 < e.val ∧
    base6_add (s.val * 36 + h.val * 6 + e.val) (e.val * 36 + s.val * 6 + h.val) = s.val * 36 + h.val * 6 + s.val ∧
    s.val = 4 ∧ h.val = 2 ∧ e.val = 3 ∧
    to_base6 (s.val + h.val + e.val) = 13 :=
  sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l708_70846


namespace NUMINAMATH_CALUDE_election_majority_l708_70882

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 160 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l708_70882


namespace NUMINAMATH_CALUDE_unique_grid_solution_l708_70889

/-- Represents a 3x3 grid with some fixed values and variables A, B, C, D -/
structure Grid :=
  (A B C D : ℕ)

/-- Checks if two numbers are adjacent in the grid -/
def adjacent (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 4) ∨ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 5) ∨
  (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 5) ∨ (x = 4 ∧ y = 7) ∨ (x = 5 ∧ y = 6) ∨
  (x = 5 ∧ y = 8) ∨ (x = 6 ∧ y = 9) ∨ (x = 7 ∧ y = 8) ∨ (x = 8 ∧ y = 9) ∨
  (y = 1 ∧ x = 2) ∨ (y = 1 ∧ x = 4) ∨ (y = 2 ∧ x = 3) ∨ (y = 2 ∧ x = 5) ∨
  (y = 3 ∧ x = 6) ∨ (y = 4 ∧ x = 5) ∨ (y = 4 ∧ x = 7) ∨ (y = 5 ∧ x = 6) ∨
  (y = 5 ∧ x = 8) ∨ (y = 6 ∧ x = 9) ∨ (y = 7 ∧ x = 8) ∨ (y = 8 ∧ x = 9)

/-- The main theorem to prove -/
theorem unique_grid_solution :
  ∀ (g : Grid),
    (g.A ≠ 1 ∧ g.A ≠ 3 ∧ g.A ≠ 5 ∧ g.A ≠ 7 ∧ g.A ≠ 9) →
    (g.B ≠ 1 ∧ g.B ≠ 3 ∧ g.B ≠ 5 ∧ g.B ≠ 7 ∧ g.B ≠ 9) →
    (g.C ≠ 1 ∧ g.C ≠ 3 ∧ g.C ≠ 5 ∧ g.C ≠ 7 ∧ g.C ≠ 9) →
    (g.D ≠ 1 ∧ g.D ≠ 3 ∧ g.D ≠ 5 ∧ g.D ≠ 7 ∧ g.D ≠ 9) →
    (∀ (x y : ℕ), adjacent x y → x + y < 12) →
    (g.A = 8 ∧ g.B = 6 ∧ g.C = 4 ∧ g.D = 2) :=
by sorry


end NUMINAMATH_CALUDE_unique_grid_solution_l708_70889


namespace NUMINAMATH_CALUDE_quadratic_root_property_l708_70892

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + m = 0) → 
  (x₂^2 + 2*x₂ + m = 0) → 
  (x₁ + x₂ = x₁*x₂ - 1) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l708_70892


namespace NUMINAMATH_CALUDE_tank_capacity_is_90_l708_70836

/-- Represents a gasoline tank with a certain capacity -/
structure GasolineTank where
  capacity : ℚ
  initialFraction : ℚ
  finalFraction : ℚ
  usedAmount : ℚ

/-- Theorem stating that the tank capacity is 90 gallons given the conditions -/
theorem tank_capacity_is_90 (tank : GasolineTank)
  (h1 : tank.initialFraction = 5/6)
  (h2 : tank.finalFraction = 2/3)
  (h3 : tank.usedAmount = 15)
  (h4 : tank.initialFraction * tank.capacity - tank.finalFraction * tank.capacity = tank.usedAmount) :
  tank.capacity = 90 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_90_l708_70836


namespace NUMINAMATH_CALUDE_inequality_solution_set_l708_70808

theorem inequality_solution_set (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) 1 ∪ Set.Icc 2 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l708_70808


namespace NUMINAMATH_CALUDE_min_value_and_angle_l708_70863

theorem min_value_and_angle (A : Real) : 
  let f := fun A => 2 * Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min_value : Real) (min_angle : Real),
    (∀ A, f A ≥ min_value) ∧
    (f min_angle = min_value) ∧
    (min_value = -1) ∧
    (min_angle = 270 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_angle_l708_70863


namespace NUMINAMATH_CALUDE_appliance_sales_prediction_l708_70871

/-- Represents the sales and cost data for an appliance -/
structure ApplianceData where
  sales : ℕ
  cost : ℕ

/-- Checks if two ApplianceData are inversely proportional -/
def inversely_proportional (a b : ApplianceData) : Prop :=
  a.sales * a.cost = b.sales * b.cost

theorem appliance_sales_prediction
  (blender_initial blender_final microwave_initial microwave_final : ApplianceData)
  (h1 : inversely_proportional blender_initial blender_final)
  (h2 : inversely_proportional microwave_initial microwave_final)
  (h3 : blender_initial.sales = 15)
  (h4 : blender_initial.cost = 300)
  (h5 : blender_final.cost = 450)
  (h6 : microwave_initial.sales = 25)
  (h7 : microwave_initial.cost = 400)
  (h8 : microwave_final.cost = 500) :
  blender_final.sales = 10 ∧ microwave_final.sales = 20 := by
  sorry

end NUMINAMATH_CALUDE_appliance_sales_prediction_l708_70871


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l708_70888

theorem fraction_sum_proof (x A B : ℚ) : 
  (5*x - 11) / (2*x^2 + x - 6) = A / (x + 2) + B / (2*x - 3) → 
  A = 3 ∧ B = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l708_70888


namespace NUMINAMATH_CALUDE_odd_periodic_function_value_l708_70891

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_value : f (-1) = 2) : 
  f 13 = -2 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_value_l708_70891


namespace NUMINAMATH_CALUDE_minimal_extensive_h_21_l708_70864

/-- An extensive function is a function from positive integers to integers
    such that f(x) + f(y) ≥ x² + y² for all positive integers x and y. -/
def Extensive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ (x.val ^ 2 : ℤ) + (y.val ^ 2 : ℤ)

/-- The sum of the first 30 values of an extensive function -/
def SumFirst30 (f : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f ⟨i + 1, Nat.succ_pos i⟩)

/-- An extensive function with minimal sum of first 30 values -/
def MinimalExtensive (h : ℕ+ → ℤ) : Prop :=
  Extensive h ∧ ∀ g : ℕ+ → ℤ, Extensive g → SumFirst30 h ≤ SumFirst30 g

theorem minimal_extensive_h_21 (h : ℕ+ → ℤ) (hmin : MinimalExtensive h) :
    h ⟨21, by norm_num⟩ ≥ 301 := by
  sorry

end NUMINAMATH_CALUDE_minimal_extensive_h_21_l708_70864


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l708_70897

/-- The equation of the trajectory of point M -/
def trajectory_equation (x y : ℝ) : Prop :=
  10 * Real.sqrt (x^2 + y^2) = |3*x + 4*y - 12|

/-- The trajectory of point M is an ellipse -/
theorem trajectory_is_ellipse :
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), trajectory_equation x y ↔ 
    ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l708_70897


namespace NUMINAMATH_CALUDE_choir_members_count_l708_70831

theorem choir_members_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 200) ∧
    (∀ n ∈ s, (n + 3) % 7 = 0 ∧ (n + 5) % 8 = 0) ∧
    s.card = 2 ∧
    123 ∈ s ∧ 179 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_choir_members_count_l708_70831


namespace NUMINAMATH_CALUDE_clock_hands_alignment_l708_70820

/-- The number of whole seconds remaining in an hour when the clock hands make equal angles with the vertical -/
def remaining_seconds : ℕ := by sorry

/-- The angle (in degrees) that the hour hand and minute hand make with the vertical when they align -/
def alignment_angle : ℚ := by sorry

theorem clock_hands_alignment :
  (alignment_angle * 120 : ℚ) = (360 - alignment_angle) * 10 ∧
  remaining_seconds = 3600 - Int.floor (alignment_angle * 120) := by sorry

end NUMINAMATH_CALUDE_clock_hands_alignment_l708_70820


namespace NUMINAMATH_CALUDE_nonagon_intersection_points_l708_70810

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of distinct interior intersection points of diagonals in a regular nonagon -/
def intersectionPoints (n : RegularNonagon) : ℕ := sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def chooseFromNine : ℕ := Nat.choose 9 4

/-- Theorem stating that the number of intersection points in a regular nonagon
    is equal to the number of ways to choose 4 vertices from 9 -/
theorem nonagon_intersection_points (n : RegularNonagon) :
  intersectionPoints n = chooseFromNine := by sorry

end NUMINAMATH_CALUDE_nonagon_intersection_points_l708_70810


namespace NUMINAMATH_CALUDE_well_cared_fish_lifespan_l708_70899

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog relative to a hamster -/
def dog_lifespan_factor : ℝ := 4

/-- The additional lifespan of a well-cared fish compared to a dog, in years -/
def fish_extra_lifespan : ℝ := 2

/-- The number of months in a year -/
def months_per_year : ℝ := 12

/-- Theorem: A well-cared fish can live 144 months -/
theorem well_cared_fish_lifespan :
  hamster_lifespan * dog_lifespan_factor * months_per_year + fish_extra_lifespan * months_per_year = 144 :=
by sorry

end NUMINAMATH_CALUDE_well_cared_fish_lifespan_l708_70899
