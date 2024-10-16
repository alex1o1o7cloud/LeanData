import Mathlib

namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l356_35646

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

/-- The theorem statement -/
theorem binomial_probability_theorem (ξ η : BinomialRV) 
  (h_ξ : ξ.n = 2) (h_η : η.n = 4) (h_p : ξ.p = η.p) 
  (h_prob : prob_ge ξ 1 = 5/9) : 
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l356_35646


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l356_35608

theorem factorial_equation_solution :
  ∃! N : ℕ, (6 : ℕ).factorial * (11 : ℕ).factorial = 20 * N.factorial :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l356_35608


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l356_35684

/-- The time it takes for three workers to complete a job together and individually -/
def JobCompletion (t_together t_a t_b t_c : ℝ) : Prop :=
  (1 / t_together) = (1 / t_a) + (1 / t_b) + (1 / t_c)

/-- Theorem stating that given the conditions, worker B completes the job in 6 days -/
theorem worker_b_completion_time :
  ∀ (t_together : ℝ),
  t_together = 3.428571428571429 →
  JobCompletion t_together 24 6 12 :=
by sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l356_35684


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l356_35658

theorem arithmetic_calculations :
  (72 * 54 + 28 * 54 = 5400) ∧
  (60 * 25 * 8 = 12000) ∧
  (2790 / (250 * 12 - 2910) = 31) ∧
  ((100 - 1456 / 26) * 78 = 3432) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l356_35658


namespace NUMINAMATH_CALUDE_projection_equals_negative_two_l356_35692

def a : Fin 2 → ℝ
| 0 => 4
| 1 => -7

def b : Fin 2 → ℝ
| 0 => 3
| 1 => -4

theorem projection_equals_negative_two :
  let proj := (((a - 2 • b) • b) / (b • b)) • b
  proj = (-2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_projection_equals_negative_two_l356_35692


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l356_35610

theorem sum_of_specific_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l356_35610


namespace NUMINAMATH_CALUDE_number_equality_l356_35690

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l356_35690


namespace NUMINAMATH_CALUDE_time_sum_after_increment_l356_35615

-- Define a type for time on a 12-hour digital clock
structure Time12Hour where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ
  is_pm : Bool

-- Function to add hours, minutes, and seconds to a given time
def addTime (start : Time12Hour) (hours minutes seconds : ℕ) : Time12Hour :=
  sorry

-- Function to calculate A + B + C for a given time
def sumTime (t : Time12Hour) : ℕ :=
  t.hours + t.minutes + t.seconds

-- Theorem statement
theorem time_sum_after_increment :
  let start_time := Time12Hour.mk 3 0 0 true
  let end_time := addTime start_time 190 45 30
  sumTime end_time = 76 := by sorry

end NUMINAMATH_CALUDE_time_sum_after_increment_l356_35615


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l356_35642

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x^2

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f a (-2 + h) ≤ f a (-2)) →
  a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l356_35642


namespace NUMINAMATH_CALUDE_octal_sum_equality_l356_35616

/-- Represents a number in base 8 --/
def OctalNumber : Type := List Nat

/-- Converts an OctalNumber to a natural number --/
def octal_to_nat (n : OctalNumber) : Nat :=
  n.foldl (fun acc d => 8 * acc + d) 0

/-- Adds two OctalNumbers in base 8 --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_sum_equality : 
  octal_add [1, 4, 6, 3] [2, 7, 5] = [1, 7, 5, 0] :=
sorry

end NUMINAMATH_CALUDE_octal_sum_equality_l356_35616


namespace NUMINAMATH_CALUDE_f_properties_l356_35679

/-- A function f that is constant for all x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a + 1|

theorem f_properties (a : ℝ) (h_a : a > 0) (h_const : ∀ x y, f a x = f a y) :
  (∀ x, f a x ≥ 1) ∧
  (f a 3 < 11/2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l356_35679


namespace NUMINAMATH_CALUDE_unfollows_calculation_correct_l356_35629

/-- Calculates the number of unfollows for an Instagram influencer over a year -/
def calculate_unfollows (initial_followers : ℕ) (daily_new_followers : ℕ) (final_followers : ℕ) : ℕ :=
  let potential_followers := initial_followers + daily_new_followers * 365
  potential_followers - final_followers

/-- Theorem: The number of unfollows is correct given the problem conditions -/
theorem unfollows_calculation_correct :
  calculate_unfollows 100000 1000 445000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_unfollows_calculation_correct_l356_35629


namespace NUMINAMATH_CALUDE_hayden_evening_snack_l356_35673

/-- Calculates the amount of nuts in one serving given the bag cost, weight, coupon value, and cost per serving after coupon. -/
def nuts_per_serving (bag_cost : ℚ) (bag_weight : ℚ) (coupon : ℚ) (serving_cost : ℚ) : ℚ :=
  let cost_after_coupon := bag_cost - coupon
  let num_servings := cost_after_coupon / serving_cost
  bag_weight / num_servings

/-- Theorem stating that under the given conditions, the amount of nuts in one serving is 1 oz. -/
theorem hayden_evening_snack :
  nuts_per_serving 25 40 5 (1/2) = 1 := by sorry

end NUMINAMATH_CALUDE_hayden_evening_snack_l356_35673


namespace NUMINAMATH_CALUDE_smallest_sum_of_leftmost_three_digits_l356_35638

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_zero (n : ℕ) : Prop := ∃ (a c : ℕ), n = 100 * a + c ∧ a < 10 ∧ c < 100

def all_digits_different (x y : ℕ) : Prop :=
  ∀ (d : ℕ), d < 10 → (
    (∃ (i : ℕ), i < 3 ∧ (x / 10^i) % 10 = d) ↔
    ¬(∃ (j : ℕ), j < 3 ∧ (y / 10^j) % 10 = d)
  )

def sum_of_leftmost_three_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10)

theorem smallest_sum_of_leftmost_three_digits
  (x y : ℕ)
  (hx : is_three_digit x)
  (hy : is_three_digit y)
  (hx0 : contains_zero x)
  (hdiff : all_digits_different x y)
  (hsum : 1000 ≤ x + y ∧ x + y ≤ 9999) :
  ∀ (z : ℕ), is_three_digit z → contains_zero z → all_digits_different z (x + y - z) →
    sum_of_leftmost_three_digits (x + y) ≤ sum_of_leftmost_three_digits (z + (x + y - z)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_leftmost_three_digits_l356_35638


namespace NUMINAMATH_CALUDE_smallest_bench_arrangement_l356_35664

theorem smallest_bench_arrangement (n : ℕ) : 
  (∃ k : ℕ, 8 * n = 10 * k) ∧ 
  (n % 8 = 0) ∧ (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, 8 * m = 10 * k) ∧ (m % 8 = 0) ∧ (m % 10 = 0))) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_bench_arrangement_l356_35664


namespace NUMINAMATH_CALUDE_odd_power_decomposition_l356_35628

theorem odd_power_decomposition (m : ℤ) : 
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_decomposition_l356_35628


namespace NUMINAMATH_CALUDE_arctan_sum_tan_l356_35614

theorem arctan_sum_tan (x y : Real) :
  x = 45 * π / 180 →
  y = 30 * π / 180 →
  Real.arctan (Real.tan x + 2 * Real.tan y) = 75 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_tan_l356_35614


namespace NUMINAMATH_CALUDE_opposite_number_l356_35644

theorem opposite_number (a : ℝ) : -a = -2023 → a = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l356_35644


namespace NUMINAMATH_CALUDE_a_10_value_l356_35695

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l356_35695


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l356_35696

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l356_35696


namespace NUMINAMATH_CALUDE_youtube_views_calculation_youtube_views_proof_l356_35670

theorem youtube_views_calculation (initial_views : ℕ) 
  (increase_factor : ℕ) (additional_views : ℕ) : ℕ :=
  let views_after_four_days := initial_views + initial_views * increase_factor
  let total_views := views_after_four_days + additional_views
  total_views

theorem youtube_views_proof :
  youtube_views_calculation 4000 10 50000 = 94000 := by
  sorry

end NUMINAMATH_CALUDE_youtube_views_calculation_youtube_views_proof_l356_35670


namespace NUMINAMATH_CALUDE_sine_matrix_determinant_zero_l356_35652

theorem sine_matrix_determinant_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 3
    | 0, 1 => Real.sin 4
    | 0, 2 => Real.sin 5
    | 1, 0 => Real.sin 6
    | 1, 1 => Real.sin 7
    | 1, 2 => Real.sin 8
    | 2, 0 => Real.sin 9
    | 2, 1 => Real.sin 10
    | 2, 2 => Real.sin 11
  Matrix.det A = 0 := by
  sorry

-- Sine angle addition formula
axiom sine_angle_addition (x y : ℝ) :
  Real.sin (x + y) = Real.sin x * Real.cos y + Real.cos x * Real.sin y

end NUMINAMATH_CALUDE_sine_matrix_determinant_zero_l356_35652


namespace NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l356_35682

/-- A cubic sequence is a sequence of integers given by a_n = n^3 + bn^2 + cn + d,
    where b, c, and d are integer constants and n ranges over all integers. -/
def CubicSequence (b c d : ℤ) : ℤ → ℤ := fun n ↦ n^3 + b*n^2 + c*n + d

/-- A number is a perfect square if there exists an integer whose square equals the number. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

theorem cubic_sequence_with_two_squares_exists : ∃ b c d : ℤ,
  let a := CubicSequence b c d
  IsPerfectSquare (a 2015) ∧
  IsPerfectSquare (a 2016) ∧
  (∀ n : ℤ, n ≠ 2015 ∧ n ≠ 2016 → ¬ IsPerfectSquare (a n)) ∧
  a 2015 * a 2016 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_sequence_with_two_squares_exists_l356_35682


namespace NUMINAMATH_CALUDE_laptop_repairs_count_l356_35648

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def phone_repairs : ℕ := 5
def computer_repairs : ℕ := 2
def total_earnings : ℕ := 121

theorem laptop_repairs_count :
  ∃ (laptop_repairs : ℕ),
    phone_repair_cost * phone_repairs +
    laptop_repair_cost * laptop_repairs +
    computer_repair_cost * computer_repairs = total_earnings ∧
    laptop_repairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_laptop_repairs_count_l356_35648


namespace NUMINAMATH_CALUDE_houses_painted_in_three_hours_l356_35647

/-- The number of houses that can be painted in a given time -/
def houses_painted (minutes_per_house : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_house

/-- Theorem: Given it takes 20 minutes to paint a house, 
    the number of houses that can be painted in 3 hours is 9 -/
theorem houses_painted_in_three_hours :
  houses_painted 20 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_houses_painted_in_three_hours_l356_35647


namespace NUMINAMATH_CALUDE_division_problem_l356_35678

theorem division_problem : 12 / (2 / (5 - 3)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l356_35678


namespace NUMINAMATH_CALUDE_set_equality_l356_35689

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l356_35689


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l356_35626

def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_8 : ℚ := 8 / 9

theorem product_of_repeating_decimals : 
  repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l356_35626


namespace NUMINAMATH_CALUDE_pushup_sequence_sum_l356_35691

theorem pushup_sequence_sum (a : ℕ → ℕ) :
  (a 0 = 10) →
  (∀ n : ℕ, a (n + 1) = a n + 5) →
  (a 0 + a 1 + a 2 = 45) := by
  sorry

end NUMINAMATH_CALUDE_pushup_sequence_sum_l356_35691


namespace NUMINAMATH_CALUDE_eighth_power_fraction_l356_35655

theorem eighth_power_fraction (x : ℝ) (h : x > 0) :
  (x^(1/2)) / (x^(1/4)) = x^(1/4) :=
by sorry

end NUMINAMATH_CALUDE_eighth_power_fraction_l356_35655


namespace NUMINAMATH_CALUDE_english_score_calculation_l356_35641

theorem english_score_calculation (average_before : ℝ) (average_after : ℝ) : 
  average_before = 92 →
  average_after = 94 →
  (3 * average_before + english_score) / 4 = average_after →
  english_score = 100 :=
by
  sorry

#check english_score_calculation

end NUMINAMATH_CALUDE_english_score_calculation_l356_35641


namespace NUMINAMATH_CALUDE_remainder_sum_theorem_l356_35627

theorem remainder_sum_theorem (n : ℕ) : 
  (∃ a b c : ℕ, 
    0 < a ∧ a < 29 ∧
    0 < b ∧ b < 41 ∧
    0 < c ∧ c < 59 ∧
    n % 29 = a ∧
    n % 41 = b ∧
    n % 59 = c ∧
    a + b + c = n) → 
  (n = 79 ∨ n = 114) :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_theorem_l356_35627


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l356_35687

theorem sum_of_x_solutions (x y : ℝ) : 
  y = 8 → x^2 + y^2 = 144 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l356_35687


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l356_35671

/-- The function f(x) = 2x^2 - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x

theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_extreme_point_of_f_l356_35671


namespace NUMINAMATH_CALUDE_f_2014_equals_zero_l356_35685

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the periodicity property of f
def HasPeriodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x + f 2

-- Theorem statement
theorem f_2014_equals_zero 
  (h_even : IsEven f) 
  (h_periodicity : HasPeriodicity f) : 
  f 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2014_equals_zero_l356_35685


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l356_35639

def is_valid_assignment (K O A L V D : ℕ) : Prop :=
  K ≠ O ∧ K ≠ A ∧ K ≠ L ∧ K ≠ V ∧ K ≠ D ∧
  O ≠ A ∧ O ≠ L ∧ O ≠ V ∧ O ≠ D ∧
  A ≠ L ∧ A ≠ V ∧ A ≠ D ∧
  L ≠ V ∧ L ≠ D ∧
  V ≠ D ∧
  K < 10 ∧ O < 10 ∧ A < 10 ∧ L < 10 ∧ V < 10 ∧ D < 10

def satisfies_equation (K O A L V D : ℕ) : Prop :=
  1000 * K + 100 * O + 10 * K + A +
  1000 * K + 100 * O + 10 * L + A =
  1000 * V + 100 * O + 10 * D + A

theorem alphametic_puzzle_solution :
  ∃! (K O A L V D : ℕ), 
    is_valid_assignment K O A L V D ∧
    satisfies_equation K O A L V D ∧
    K = 3 ∧ O = 9 ∧ A = 0 ∧ L = 8 ∧ V = 7 ∧ D = 1 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l356_35639


namespace NUMINAMATH_CALUDE_expression_simplification_l356_35602

theorem expression_simplification (x : ℝ) (h : x = 2 + Real.sqrt 3) :
  (x + 1) / (x^2 - 4) * ((1 / (x + 1)) + 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l356_35602


namespace NUMINAMATH_CALUDE_exists_k_l356_35605

/-- A game configuration with two players and blank squares. -/
structure GameConfig where
  s₁ : ℕ  -- Steps for player 1
  s₂ : ℕ  -- Steps for player 2
  board_size : ℕ  -- Total number of squares on the board

/-- Winning probability for a player given a game configuration and number of blank squares. -/
def winning_probability (config : GameConfig) (player : ℕ) (num_blanks : ℕ) : ℝ :=
  sorry

/-- The statement that proves the existence of k satisfying the given conditions. -/
theorem exists_k (config : GameConfig) : ∃ k : ℕ,
  (∀ n < k, winning_probability config 1 n > 1/2) ∧
  (∃ board_config : List ℕ, 
    board_config.length = k ∧ 
    winning_probability config 2 k > 1/2) :=
by
  -- Assume s₁ = 3 and s₂ = 2
  have h1 : config.s₁ = 3 := by sorry
  have h2 : config.s₂ = 2 := by sorry

  -- Prove that k = 3 satisfies the conditions
  use 3
  sorry


end NUMINAMATH_CALUDE_exists_k_l356_35605


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l356_35662

-- Define the parabola E: y^2 = 4x
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the directrix l: x = -1
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define a function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Main theorem
theorem fixed_point_theorem (A B : ℝ × ℝ) (h_A : A ∈ E) (h_B : B ∈ E) 
  (h_line : ∃ k : ℝ, A.2 - F.2 = k * (A.1 - F.1) ∧ B.2 - F.2 = k * (B.1 - F.1)) :
  ∃ t : ℝ, reflect_x A + t • (B - reflect_x A) = (-1, 0) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l356_35662


namespace NUMINAMATH_CALUDE_count_squares_below_line_l356_35622

/-- The number of 1x1 squares in the first quadrant with interiors lying entirely below the line 7x + 268y = 1876 -/
def squares_below_line : ℕ := 801

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 7 * x + 268 * y = 1876

theorem count_squares_below_line :
  squares_below_line = 801 :=
sorry

end NUMINAMATH_CALUDE_count_squares_below_line_l356_35622


namespace NUMINAMATH_CALUDE_sum_ratio_equals_55_49_l356_35661

theorem sum_ratio_equals_55_49 : 
  let sum_n (n : ℕ) := n * (n + 1) / 2
  let sum_squares (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6
  let sum_cubes (n : ℕ) := (sum_n n) ^ 2
  (sum_n 10 * sum_cubes 10) / (sum_squares 10) ^ 2 = 55 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_55_49_l356_35661


namespace NUMINAMATH_CALUDE_problem_2017_l356_35611

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + d * (n - 1)

/-- The problem statement -/
theorem problem_2017 : arithmeticSequence 4 3 672 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_2017_l356_35611


namespace NUMINAMATH_CALUDE_det_value_for_quadratic_root_l356_35654

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_value_for_quadratic_root (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det (x + 1) (3*x) (x - 2) (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_value_for_quadratic_root_l356_35654


namespace NUMINAMATH_CALUDE_unique_function_characterization_l356_35603

theorem unique_function_characterization :
  ∀ f : ℕ+ → ℕ+,
  (∀ x y : ℕ+, x < y → f x < f y) →
  (∀ x y : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ x : ℕ+, f x = x^2 := by
sorry

end NUMINAMATH_CALUDE_unique_function_characterization_l356_35603


namespace NUMINAMATH_CALUDE_regular_octagon_side_length_l356_35607

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  sideLength : ℝ
  perimeter : ℝ

/-- The perimeter of a regular octagon is 8 times the length of one side. -/
def RegularOctagon.perimeterFormula (o : RegularOctagon) : ℝ :=
  8 * o.sideLength

theorem regular_octagon_side_length (o : RegularOctagon) 
    (h : o.perimeter = 23.6) : o.sideLength = 2.95 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_side_length_l356_35607


namespace NUMINAMATH_CALUDE_father_current_age_l356_35601

/-- The age of the daughter now -/
def daughter_age : ℕ := 10

/-- The age of the father now -/
def father_age : ℕ := 4 * daughter_age

/-- In 20 years, the father will be twice as old as the daughter -/
axiom future_relation : father_age + 20 = 2 * (daughter_age + 20)

theorem father_current_age : father_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_father_current_age_l356_35601


namespace NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l356_35623

/-- An isosceles triangle with given perimeter and base -/
structure IsoscelesTriangle where
  perimeter : ℝ
  base : ℝ
  legs_equal : ℝ
  perimeter_eq : perimeter = 2 * legs_equal + base

/-- Theorem: In an isosceles triangle with perimeter 26 cm and base 11 cm, each leg is 7.5 cm -/
theorem isosceles_triangle_leg_length 
  (triangle : IsoscelesTriangle) 
  (h_perimeter : triangle.perimeter = 26) 
  (h_base : triangle.base = 11) : 
  triangle.legs_equal = 7.5 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_leg_length_l356_35623


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l356_35674

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l356_35674


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l356_35665

-- Define the position function
def s (t : ℝ) : ℝ := t^2 - 2*t + 5

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 6 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l356_35665


namespace NUMINAMATH_CALUDE_jessica_watermelons_l356_35659

/-- Given that Jessica grew some watermelons and 30 carrots,
    rabbits ate 27 watermelons, and Jessica has 8 watermelons left,
    prove that Jessica originally grew 35 watermelons. -/
theorem jessica_watermelons :
  ∀ (original_watermelons : ℕ) (carrots : ℕ),
    carrots = 30 →
    original_watermelons - 27 = 8 →
    original_watermelons = 35 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l356_35659


namespace NUMINAMATH_CALUDE_correct_ages_l356_35680

/-- Represents the ages of a family -/
structure FamilyAges where
  kareem : ℕ
  son : ℕ
  daughter : ℕ
  wife : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.kareem = 3 * ages.son ∧
  ages.daughter = ages.son / 2 ∧
  ages.kareem + 10 + ages.son + 10 + ages.daughter + 10 = 120 ∧
  ages.wife = ages.kareem - 8

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem correct_ages : 
  let ages : FamilyAges := ⟨60, 20, 10, 52⟩
  satisfiesConditions ages :=
by sorry

end NUMINAMATH_CALUDE_correct_ages_l356_35680


namespace NUMINAMATH_CALUDE_equal_numbers_l356_35618

theorem equal_numbers (a b c : ℝ) (h : |a - b| = 2*|b - c| ∧ |a - b| = 3*|c - a|) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_l356_35618


namespace NUMINAMATH_CALUDE_quadratic_function_range_l356_35683

/-- A quadratic function with specific properties -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b^2 - b + 1

/-- The theorem statement -/
theorem quadratic_function_range (a b : ℝ) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∀ x ∈ Set.Icc (-1) 1, f a b x > 0) →
  b < -1 ∨ b > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l356_35683


namespace NUMINAMATH_CALUDE_arithmetic_problem_l356_35649

theorem arithmetic_problem : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l356_35649


namespace NUMINAMATH_CALUDE_sum_of_first_10_odd_numbers_l356_35645

def sum_of_odd_numbers (n : ℕ) : ℕ := n^2

theorem sum_of_first_10_odd_numbers : sum_of_odd_numbers 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_odd_numbers_l356_35645


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l356_35686

/-- The function f(x) = a^(x-2016) + 1 has a fixed point at (2016, 2) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = a^(x - 2016) + 1) ∧ f 2016 = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l356_35686


namespace NUMINAMATH_CALUDE_no_three_numbers_exist_l356_35676

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  ((∃ k : ℕ, a^2 - 1 = b * k ∨ a^2 - 1 = c * k) ∧
   (∃ l : ℕ, b^2 - 1 = a * l ∨ b^2 - 1 = c * l) ∧
   (∃ m : ℕ, c^2 - 1 = a * m ∨ c^2 - 1 = b * m)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_numbers_exist_l356_35676


namespace NUMINAMATH_CALUDE_system_solution_unique_l356_35619

theorem system_solution_unique (x y : ℝ) : 
  x + 3 * y = -1 ∧ 2 * x + y = 3 ↔ x = 2 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l356_35619


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l356_35688

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

theorem sum_of_special_primes_is_prime :
  ∀ A B : ℕ,
    is_prime A →
    is_prime B →
    is_prime (A - B) →
    is_prime (A + B) →
    A > B →
    B = 2 →
    is_odd A →
    is_odd (A - B) →
    is_odd (A + B) →
    (∃ k : ℕ, A = (A - B) + 2*k ∧ (A + B) = A + 2*k) →
    is_prime (A + B + (A - B) + B) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l356_35688


namespace NUMINAMATH_CALUDE_distance_A_to_B_l356_35617

/-- The distance between points A(1, 0) and B(0, -1) is √2. -/
theorem distance_A_to_B : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l356_35617


namespace NUMINAMATH_CALUDE_square_fence_perimeter_is_77_and_third_l356_35621

/-- The outer perimeter of a square fence with given specifications -/
def squareFencePerimeter (totalPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := totalPosts / 4 + 1
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := gapsPerSide * gapWidth + postsPerSide * postWidth
  4 * sideLength

/-- Theorem stating the perimeter of the square fence with given specifications -/
theorem square_fence_perimeter_is_77_and_third :
  squareFencePerimeter 16 (1/3) 6 = 77 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_is_77_and_third_l356_35621


namespace NUMINAMATH_CALUDE_abc_inequality_l356_35651

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l356_35651


namespace NUMINAMATH_CALUDE_seed_germination_problem_l356_35620

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.30 * x + 0.50 * 200) / (x + 200) = 0.35714285714285715 → 
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l356_35620


namespace NUMINAMATH_CALUDE_expression_value_l356_35634

theorem expression_value : 3^(0^(2^2)) + ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l356_35634


namespace NUMINAMATH_CALUDE_max_squares_is_seven_l356_35668

/-- A shape formed by unit-length sticks on a plane -/
structure StickShape where
  sticks : ℕ
  squares : ℕ
  rows : ℕ
  first_row_squares : ℕ

/-- Predicate to check if a shape is valid according to the problem constraints -/
def is_valid_shape (s : StickShape) : Prop :=
  s.sticks = 20 ∧
  s.rows ≥ 1 ∧
  s.first_row_squares ≥ 1 ∧
  s.first_row_squares ≤ s.squares ∧
  (s.squares - s.first_row_squares) % (s.rows - 1) = 0

/-- The maximum number of squares that can be formed -/
def max_squares : ℕ := 7

/-- Theorem stating that the maximum number of squares is 7 -/
theorem max_squares_is_seven :
  ∀ s : StickShape, is_valid_shape s → s.squares ≤ max_squares :=
sorry

end NUMINAMATH_CALUDE_max_squares_is_seven_l356_35668


namespace NUMINAMATH_CALUDE_equation_one_solutions_l356_35625

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 - 4 = 0 ↔ x = -1 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l356_35625


namespace NUMINAMATH_CALUDE_milk_price_calculation_l356_35637

/-- Proves that given the initial volume of milk, volume of water added, and final price of the mixture,
    the original price of milk per litre can be calculated. -/
theorem milk_price_calculation (initial_milk_volume : ℝ) (water_added : ℝ) (final_mixture_price : ℝ) :
  initial_milk_volume = 60 →
  water_added = 15 →
  final_mixture_price = 32 / 3 →
  ∃ (original_milk_price : ℝ), original_milk_price = 800 / 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_calculation_l356_35637


namespace NUMINAMATH_CALUDE_star_example_l356_35667

/-- Custom binary operation ※ -/
def star (a b : ℕ) : ℕ := a * b + a + b

/-- Theorem: (3※4)※1 = 39 -/
theorem star_example : star (star 3 4) 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l356_35667


namespace NUMINAMATH_CALUDE_min_value_abc_l356_35630

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l356_35630


namespace NUMINAMATH_CALUDE_point_relationship_l356_35669

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the points
def point1 : ℝ × ℝ := (-4, f (-4))
def point2 : ℝ × ℝ := (-1, f (-1))
def point3 : ℝ × ℝ := (2, f 2)

-- Theorem statement
theorem point_relationship :
  let y₁ := point1.2
  let y₂ := point2.2
  let y₃ := point3.2
  y₂ > y₃ ∧ y₃ > y₁ := by sorry

end NUMINAMATH_CALUDE_point_relationship_l356_35669


namespace NUMINAMATH_CALUDE_hockey_league_games_l356_35640

/-- The number of games played in a hockey league season -/
def games_in_season (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 17 teams, where each team faces all other teams 10 times,
    the total number of games played in the season is 680. -/
theorem hockey_league_games :
  games_in_season 17 10 = 680 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l356_35640


namespace NUMINAMATH_CALUDE_postage_cost_for_625_ounces_l356_35663

/-- Calculates the postage cost for a letter -/
def postage_cost (weight : ℚ) (base_rate : ℚ) (additional_rate : ℚ) : ℚ :=
  let additional_weight := (weight - 1).ceil
  base_rate + additional_weight * additional_rate

theorem postage_cost_for_625_ounces :
  postage_cost (6.25 : ℚ) (0.50 : ℚ) (0.30 : ℚ) = (2.30 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_postage_cost_for_625_ounces_l356_35663


namespace NUMINAMATH_CALUDE_h_perimeter_is_26_l356_35694

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of an H-shaped figure formed by three rectangles -/
def hPerimeter (r : Rectangle) : ℝ :=
  2 * r.length + 4 * r.width + 2 * r.length

/-- Theorem: The perimeter of an H-shaped figure formed by three 3x5 inch rectangles is 26 inches -/
theorem h_perimeter_is_26 :
  let r : Rectangle := { length := 5, width := 3 }
  hPerimeter r = 26 := by
  sorry

end NUMINAMATH_CALUDE_h_perimeter_is_26_l356_35694


namespace NUMINAMATH_CALUDE_certain_multiple_proof_l356_35693

theorem certain_multiple_proof (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : 7 * n - 15 = m * n + 10) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_multiple_proof_l356_35693


namespace NUMINAMATH_CALUDE_parabola_c_is_negative_eighteen_l356_35681

/-- A parabola passing through two given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_point1 : 2 * 2^2 + b * 2 + c = 6
  pass_through_point2 : 2 * (-3)^2 + b * (-3) + c = -24

/-- The value of c for the parabola -/
def parabola_c_value (p : Parabola) : ℝ := -18

/-- Theorem stating that the value of c for the parabola is -18 -/
theorem parabola_c_is_negative_eighteen (p : Parabola) : 
  parabola_c_value p = p.c := by sorry

end NUMINAMATH_CALUDE_parabola_c_is_negative_eighteen_l356_35681


namespace NUMINAMATH_CALUDE_percentage_6_plus_years_l356_35631

-- Define the number of marks for each year range
def marks : List Nat := [10, 4, 6, 5, 8, 3, 5, 4, 2, 2]

-- Define the total number of marks
def total_marks : Nat := marks.sum

-- Define the number of marks for 6 years or more
def marks_6_plus : Nat := (marks.drop 6).sum

-- Theorem to prove
theorem percentage_6_plus_years (ε : Real) (hε : ε > 0) :
  ∃ (p : Real), abs (p - 26.53) < ε ∧ p = (marks_6_plus * 100 : Real) / total_marks :=
sorry

end NUMINAMATH_CALUDE_percentage_6_plus_years_l356_35631


namespace NUMINAMATH_CALUDE_football_field_fertilizer_l356_35609

/-- Proves that the total amount of fertilizer spread across a football field is 800 pounds,
    given the field's area and a known fertilizer distribution over a portion of the field. -/
theorem football_field_fertilizer (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) :
  total_area = 9600 →
  partial_area = 3600 →
  partial_fertilizer = 300 →
  (partial_fertilizer / partial_area) * total_area = 800 := by
  sorry

end NUMINAMATH_CALUDE_football_field_fertilizer_l356_35609


namespace NUMINAMATH_CALUDE_bills_age_l356_35636

theorem bills_age (caroline : ℕ) 
  (h1 : caroline + (2 * caroline - 1) + (caroline / 2) + (2 * caroline - 1) + (4 * caroline) = 108) : 
  2 * caroline - 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_bills_age_l356_35636


namespace NUMINAMATH_CALUDE_complex_modulus_l356_35660

theorem complex_modulus (z : ℂ) : z * (1 + Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l356_35660


namespace NUMINAMATH_CALUDE_sum_of_irrationals_not_always_irrational_student_claim_incorrect_l356_35675

theorem sum_of_irrationals_not_always_irrational :
  ∃ (a b : ℝ), 
    (¬ ∃ (q : ℚ), a = ↑q) ∧ 
    (¬ ∃ (q : ℚ), b = ↑q) ∧ 
    (∃ (q : ℚ), a + b = ↑q) :=
by sorry

-- Given conditions
axiom sqrt_2_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 = ↑q
axiom sqrt_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q
axiom sum_sqrt_2_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 + Real.sqrt 3 = ↑q

-- The statement to be proved
theorem student_claim_incorrect : 
  ¬ (∀ (a b : ℝ), (¬ ∃ (q : ℚ), a = ↑q) → (¬ ∃ (q : ℚ), b = ↑q) → (¬ ∃ (q : ℚ), a + b = ↑q)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_irrationals_not_always_irrational_student_claim_incorrect_l356_35675


namespace NUMINAMATH_CALUDE_water_usage_fraction_l356_35613

theorem water_usage_fraction (initial_water : ℚ) (car_water : ℚ) (num_cars : ℕ) 
  (plant_water_diff : ℚ) (plate_clothes_water : ℚ) : 
  initial_water = 65 → 
  car_water = 7 → 
  num_cars = 2 → 
  plant_water_diff = 11 → 
  plate_clothes_water = 24 → 
  let total_car_water := car_water * num_cars
  let plant_water := total_car_water - plant_water_diff
  let total_used_water := total_car_water + plant_water
  let remaining_water := initial_water - total_used_water
  plate_clothes_water / remaining_water = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_water_usage_fraction_l356_35613


namespace NUMINAMATH_CALUDE_min_ballots_proof_l356_35600

/-- Represents the number of candidates for each position -/
def candidates : List Nat := [3, 4, 5]

/-- Represents the requirement that each candidate must appear under each number
    an equal number of times -/
def equal_appearance (ballots : Nat) : Prop :=
  ∀ n ∈ candidates, ballots % n = 0

/-- The minimum number of different ballots required -/
def min_ballots : Nat := 5

/-- Theorem stating that the minimum number of ballots satisfying the equal appearance
    requirement is 5 -/
theorem min_ballots_proof :
  (∀ k : Nat, k < min_ballots → ¬(equal_appearance k)) ∧
  (equal_appearance min_ballots) :=
sorry

end NUMINAMATH_CALUDE_min_ballots_proof_l356_35600


namespace NUMINAMATH_CALUDE_football_club_balance_l356_35697

/-- Represents the balance and transactions of a football club --/
structure FootballClub where
  initialBalance : ℝ
  playersSold : ℕ
  sellingPrice : ℝ
  playerAPrice : ℝ
  playerBPrice : ℝ
  playerCPrice : ℝ
  playerDPrice : ℝ
  eurToUsd : ℝ
  gbpToUsd : ℝ
  jpyToUsd : ℝ

/-- Calculates the final balance of the football club after transactions --/
def finalBalance (club : FootballClub) : ℝ :=
  club.initialBalance +
  club.playersSold * club.sellingPrice -
  (club.playerAPrice * club.eurToUsd +
   club.playerBPrice * club.gbpToUsd +
   club.playerCPrice * club.jpyToUsd +
   club.playerDPrice * club.eurToUsd)

/-- Theorem stating that the final balance of the football club is 71.4 million USD --/
theorem football_club_balance (club : FootballClub)
  (h1 : club.initialBalance = 100)
  (h2 : club.playersSold = 2)
  (h3 : club.sellingPrice = 10)
  (h4 : club.playerAPrice = 12)
  (h5 : club.playerBPrice = 8)
  (h6 : club.playerCPrice = 1000)
  (h7 : club.playerDPrice = 9)
  (h8 : club.eurToUsd = 1.3)
  (h9 : club.gbpToUsd = 1.6)
  (h10 : club.jpyToUsd = 0.0085) :
  finalBalance club = 71.4 := by
  sorry

end NUMINAMATH_CALUDE_football_club_balance_l356_35697


namespace NUMINAMATH_CALUDE_exponential_linear_independence_l356_35666

theorem exponential_linear_independence 
  (k₁ k₂ k₃ : ℝ) 
  (h₁ : k₁ ≠ k₂) 
  (h₂ : k₁ ≠ k₃) 
  (h₃ : k₂ ≠ k₃) :
  ∀ (α₁ α₂ α₃ : ℝ), 
  (∀ x : ℝ, α₁ * Real.exp (k₁ * x) + α₂ * Real.exp (k₂ * x) + α₃ * Real.exp (k₃ * x) = 0) → 
  α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_exponential_linear_independence_l356_35666


namespace NUMINAMATH_CALUDE_complex_trajectory_l356_35653

/-- The trajectory of a complex number with given modulus -/
theorem complex_trajectory (x y : ℝ) (h : Complex.abs (x - 2 + y * Complex.I) = 2 * Real.sqrt 2) :
  (x - 2)^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_trajectory_l356_35653


namespace NUMINAMATH_CALUDE_max_value_of_expression_l356_35656

theorem max_value_of_expression (a b : ℕ+) (ha : a < 6) (hb : b < 10) :
  (∀ x y : ℕ+, x < 6 → y < 10 → 2 * x - x * y ≤ 2 * a - a * b) →
  2 * a - a * b = 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l356_35656


namespace NUMINAMATH_CALUDE_abs_three_implies_plus_minus_three_l356_35672

theorem abs_three_implies_plus_minus_three (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by sorry

end NUMINAMATH_CALUDE_abs_three_implies_plus_minus_three_l356_35672


namespace NUMINAMATH_CALUDE_library_budget_is_3000_l356_35650

-- Define the total budget
def total_budget : ℝ := 20000

-- Define the library budget percentage
def library_percentage : ℝ := 0.15

-- Define the parks budget percentage
def parks_percentage : ℝ := 0.24

-- Define the remaining budget
def remaining_budget : ℝ := 12200

-- Theorem to prove
theorem library_budget_is_3000 :
  library_percentage * total_budget = 3000 :=
by
  sorry


end NUMINAMATH_CALUDE_library_budget_is_3000_l356_35650


namespace NUMINAMATH_CALUDE_feb_1_is_sunday_l356_35632

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to get the day n days before a given day
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (prevDay d) n

-- Theorem statement
theorem feb_1_is_sunday (h : DayOfWeek.Saturday = daysBefore DayOfWeek.Saturday 13) :
  DayOfWeek.Sunday = daysBefore DayOfWeek.Saturday 13 :=
by sorry

end NUMINAMATH_CALUDE_feb_1_is_sunday_l356_35632


namespace NUMINAMATH_CALUDE_lcm_of_incremented_numbers_l356_35657

theorem lcm_of_incremented_numbers : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 13 19)) = 8645 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_incremented_numbers_l356_35657


namespace NUMINAMATH_CALUDE_knowledge_competition_theorem_l356_35612

/-- Represents a player in the knowledge competition --/
structure Player where
  correct_prob : ℚ
  deriving Repr

/-- Represents the game setup --/
structure Game where
  player_a : Player
  player_b : Player
  num_questions : ℕ
  deriving Repr

/-- Calculates the probability of a specific score for a player --/
def prob_score (game : Game) (player : Player) (score : ℕ) : ℚ :=
  sorry

/-- Calculates the mathematical expectation of a player's score --/
def expected_score (game : Game) (player : Player) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem knowledge_competition_theorem (game : Game) :
  game.player_a = Player.mk (2/3)
  → game.player_b = Player.mk (4/5)
  → game.num_questions = 2
  → prob_score game game.player_b 10 = 337/900
  ∧ expected_score game game.player_a = 23/3 :=
  sorry

end NUMINAMATH_CALUDE_knowledge_competition_theorem_l356_35612


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_zero_l356_35643

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + Real.log x

theorem tangent_line_implies_b_minus_a_zero (a b : ℝ) :
  (∀ x, f a b x = a * x^2 + b * x + Real.log x) →
  (∃ m c, ∀ x, m * x + c = 4 * x - 2 ∧ f a b 1 = m * 1 + c) →
  b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_zero_l356_35643


namespace NUMINAMATH_CALUDE_complex_equation_solution_l356_35633

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + z) * Complex.I = 1 - Complex.I → z = -2 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l356_35633


namespace NUMINAMATH_CALUDE_otimes_twice_2h_l356_35635

-- Define the operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_twice_2h (h : ℝ) : otimes (2*h) (otimes (2*h) (2*h)) = 2*h := by
  sorry

end NUMINAMATH_CALUDE_otimes_twice_2h_l356_35635


namespace NUMINAMATH_CALUDE_count_assignments_l356_35624

/-- Represents a mathematical statement --/
inductive Statement
  | Assignment (lhs : String) (rhs : String)
  | Equation (lhs : String) (rhs : String)
  | EqualChain (vars : List String) (value : String)

/-- Checks if a statement is an assignment --/
def isAssignment (s : Statement) : Bool :=
  match s with
  | Statement.Assignment _ _ => true
  | _ => false

/-- List of given statements --/
def statements : List Statement :=
  [ Statement.Assignment "m" "x^3 - x^2"
  , Statement.Assignment "T" "T × I"
  , Statement.Equation "32" "A"
  , Statement.Assignment "A" "A + 2"
  , Statement.EqualChain ["a", "b"] "4"
  ]

/-- Main theorem --/
theorem count_assignments :
  (statements.filter isAssignment).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_assignments_l356_35624


namespace NUMINAMATH_CALUDE_triangle_area_is_six_l356_35677

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  (1/2) * b * c * Real.sin A

theorem triangle_area_is_six (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 4)
  (h2 : Real.tan A = 3)
  (h3 : Real.cos C = Real.sqrt 5 / 5) : 
  triangle_area a b c A B C = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_six_l356_35677


namespace NUMINAMATH_CALUDE_sun_division_l356_35698

theorem sun_division (x y z : ℚ) : 
  (∀ r : ℚ, y = (45/100) * r → z = (30/100) * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 54 →                                          -- Y's share is Rs. 54
  x + y + z = 210                                   -- Total amount is Rs. 210
  := by sorry

end NUMINAMATH_CALUDE_sun_division_l356_35698


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l356_35699

/-- A linear function passing through the origin -/
def passes_through_origin (m : ℝ) : Prop :=
  0 = -2 * 0 + (m - 5)

/-- Theorem: If the linear function y = -2x + (m-5) passes through the origin, then m = 5 -/
theorem linear_function_through_origin (m : ℝ) : 
  passes_through_origin m → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_origin_l356_35699


namespace NUMINAMATH_CALUDE_max_apartment_size_l356_35606

/-- Given:
  * The rental rate in Greenview is $1.20 per square foot.
  * Max's monthly budget for rent is $720.
  Prove that the largest apartment size Max can afford is 600 square feet. -/
theorem max_apartment_size (rental_rate : ℝ) (max_budget : ℝ) (max_size : ℝ) : 
  rental_rate = 1.20 →
  max_budget = 720 →
  max_size * rental_rate = max_budget →
  max_size = 600 := by
  sorry

#check max_apartment_size

end NUMINAMATH_CALUDE_max_apartment_size_l356_35606


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l356_35604

theorem fourth_degree_polynomial_roots : ∃ (a b c d : ℝ),
  (a = 1 - Real.sqrt 3) ∧
  (b = 1 + Real.sqrt 3) ∧
  (c = (1 - Real.sqrt 13) / 2) ∧
  (d = (1 + Real.sqrt 13) / 2) ∧
  (∀ x : ℝ, x^4 - 3*x^3 + 3*x^2 - x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l356_35604
