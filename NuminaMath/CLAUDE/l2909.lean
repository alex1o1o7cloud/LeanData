import Mathlib

namespace NUMINAMATH_CALUDE_max_unit_digit_of_2015_divisor_power_l2909_290970

def unit_digit (n : ℕ) : ℕ := n % 10

def is_divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem max_unit_digit_of_2015_divisor_power :
  ∃ (d : ℕ), is_divisor d 2015 ∧
  unit_digit (d^(2015 / d)) = 7 ∧
  ∀ (k : ℕ), is_divisor k 2015 → unit_digit (k^(2015 / k)) ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_unit_digit_of_2015_divisor_power_l2909_290970


namespace NUMINAMATH_CALUDE_coins_in_second_stack_l2909_290945

theorem coins_in_second_stack (total_coins : ℕ) (first_stack : ℕ) (h1 : total_coins = 12) (h2 : first_stack = 4) :
  total_coins - first_stack = 8 := by
  sorry

end NUMINAMATH_CALUDE_coins_in_second_stack_l2909_290945


namespace NUMINAMATH_CALUDE_horse_gram_consumption_l2909_290980

/-- If 15 horses eat 15 bags of gram in 15 days, then 1 horse will eat 1 bag of gram in 15 days. -/
theorem horse_gram_consumption 
  (horses : ℕ) (bags : ℕ) (days : ℕ) 
  (h_horses : horses = 15)
  (h_bags : bags = 15)
  (h_days : days = 15)
  (h_consumption : horses * bags = horses * days) :
  1 * 1 = 1 * days :=
sorry

end NUMINAMATH_CALUDE_horse_gram_consumption_l2909_290980


namespace NUMINAMATH_CALUDE_enclosed_area_theorem_l2909_290977

noncomputable def g (x : ℝ) : ℝ := 2 - Real.sqrt (1 - (2*x/3)^2)

def domain : Set ℝ := Set.Icc (-3/2) (3/2)

theorem enclosed_area_theorem (A : ℝ) :
  A = 2 * (π * (3/2)^2 / 2 - ∫ x in (Set.Icc 0 (3/2)), g x) :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_theorem_l2909_290977


namespace NUMINAMATH_CALUDE_power_sum_simplification_l2909_290932

theorem power_sum_simplification (n : ℕ) : (-3)^n + 2*(-3)^(n-1) = -(-3)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l2909_290932


namespace NUMINAMATH_CALUDE_down_payment_proof_l2909_290920

/-- Calculates the down payment for a car loan given the total price, monthly payment, and loan duration in years. -/
def calculate_down_payment (total_price : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) : ℕ :=
  total_price - monthly_payment * loan_years * 12

/-- Proves that the down payment for a $20,000 car with a 5-year loan and $250 monthly payment is $5,000. -/
theorem down_payment_proof :
  calculate_down_payment 20000 250 5 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_proof_l2909_290920


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_56_l2909_290941

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def count_special_numbers : ℕ :=
  let thousands_digit := 8
  let units_digit := 2
  let hundreds_choices := 8
  let tens_choices := 7
  thousands_digit * units_digit * hundreds_choices * tens_choices

theorem count_special_numbers_eq_56 :
  count_special_numbers = 56 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_56_l2909_290941


namespace NUMINAMATH_CALUDE_total_football_games_l2909_290985

theorem total_football_games (games_missed : ℕ) (games_attended : ℕ) : 
  games_missed = 4 → games_attended = 3 → games_missed + games_attended = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l2909_290985


namespace NUMINAMATH_CALUDE_calculate_partner_b_contribution_b_contribution_is_16200_l2909_290944

/-- Calculates the contribution of partner B given the initial investment of A, 
    the time before B joins, and the profit-sharing ratio. -/
theorem calculate_partner_b_contribution 
  (a_investment : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) : ℕ :=
  let b_investment := 
    (a_investment * total_months * profit_ratio_b) / 
    (profit_ratio_a * (total_months - b_join_month))
  b_investment

/-- Proves that B's contribution is 16200 given the problem conditions -/
theorem b_contribution_is_16200 : 
  calculate_partner_b_contribution 4500 12 7 2 3 = 16200 := by
  sorry

end NUMINAMATH_CALUDE_calculate_partner_b_contribution_b_contribution_is_16200_l2909_290944


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l2909_290965

/-- The number of book shelves -/
def num_shelves : ℕ := 150

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 15

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_on_shelves :
  total_books = 2250 :=
by sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l2909_290965


namespace NUMINAMATH_CALUDE_complex_number_proof_l2909_290917

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_proof (Z : ℂ) 
  (h1 : Complex.abs Z = 3)
  (h2 : is_pure_imaginary (Z + 3*I)) : 
  Z = 3*I := by sorry

end NUMINAMATH_CALUDE_complex_number_proof_l2909_290917


namespace NUMINAMATH_CALUDE_log_xy_value_l2909_290933

theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log (x^3 * y^5) = 2)
  (h2 : Real.log (x^4 * y^2) = 2)
  (h3 : Real.log (x^2 * y^7) = 3) :
  Real.log (x * y) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l2909_290933


namespace NUMINAMATH_CALUDE_parallel_line_theorem_l2909_290907

/-- A line parallel to another line with a given y-intercept -/
def parallel_line_with_y_intercept (a b c : ℝ) (y_intercept : ℝ) : Prop :=
  ∃ k : ℝ, (k ≠ 0) ∧ 
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ a * x + b * y + k = 0) ∧
  (a * 0 + b * y_intercept + k = 0)

/-- The equation x + y + 1 = 0 represents the line parallel to x + y + 4 = 0 with y-intercept -1 -/
theorem parallel_line_theorem :
  parallel_line_with_y_intercept 1 1 4 (-1) →
  ∀ x y : ℝ, x + y + 1 = 0 ↔ parallel_line_with_y_intercept 1 1 4 (-1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_l2909_290907


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2909_290957

/-- The problem statement -/
theorem tangent_line_problem (k : ℝ) (P : ℝ × ℝ) (A : ℝ × ℝ) :
  k > 0 →
  P.1 * k + P.2 + 4 = 0 →
  A.1^2 + A.2^2 - 2*A.2 = 0 →
  (∀ Q : ℝ × ℝ, Q.1^2 + Q.2^2 - 2*Q.2 = 0 → 
    (A.1 - P.1)^2 + (A.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 = 4 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2909_290957


namespace NUMINAMATH_CALUDE_point_on_bisector_l2909_290946

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x (p : Point) : Prop := p.y = p.x

theorem point_on_bisector (a b : ℝ) : 
  let A : Point := ⟨a, b⟩
  let B : Point := ⟨b, a⟩
  A = B → line_y_eq_x A := by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_l2909_290946


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2909_290998

/-- A quadratic function with vertex (2, -1) passing through (-1, -16) has a = -5/3 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 2)^2 - 1) →  -- vertex form
  (a * (-1)^2 + b * (-1) + c = -16) →                 -- passes through (-1, -16)
  a = -5/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2909_290998


namespace NUMINAMATH_CALUDE_equation_equivalence_l2909_290934

theorem equation_equivalence (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0 ↔
  (1 + 1/x) * (1 + 2/y) * (1 + 3/z) * (1 + 4/w) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2909_290934


namespace NUMINAMATH_CALUDE_triangle_condition_implies_a_ge_5_l2909_290938

/-- The function f(x) = x^2 - 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

/-- Theorem: If for any three distinct values in [0, 3], f(x) can form a triangle, then a ≥ 5 -/
theorem triangle_condition_implies_a_ge_5 (a : ℝ) :
  (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 →
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    f a x + f a y > f a z ∧ f a y + f a z > f a x ∧ f a x + f a z > f a y) →
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_a_ge_5_l2909_290938


namespace NUMINAMATH_CALUDE_smallest_delightful_integer_l2909_290979

/-- Definition of a delightful integer -/
def IsDelightful (B : ℤ) : Prop :=
  ∃ n : ℕ, (n + 1) * (2 * B + n) = 6100

/-- The smallest delightful integer -/
theorem smallest_delightful_integer :
  IsDelightful (-38) ∧ ∀ B : ℤ, B < -38 → ¬IsDelightful B :=
by sorry

end NUMINAMATH_CALUDE_smallest_delightful_integer_l2909_290979


namespace NUMINAMATH_CALUDE_exist_three_digits_forming_primes_l2909_290960

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

theorem exist_three_digits_forming_primes :
  ∃ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_prime (two_digit_number a b) ∧
    is_prime (two_digit_number b a) ∧
    is_prime (two_digit_number b c) ∧
    is_prime (two_digit_number c b) ∧
    is_prime (two_digit_number c a) ∧
    is_prime (two_digit_number a c) :=
sorry

end NUMINAMATH_CALUDE_exist_three_digits_forming_primes_l2909_290960


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2909_290921

theorem min_x_prime_factorization_sum (x y a b e : ℕ+) (c d f : ℕ) :
  (∀ x' y' : ℕ+, 7 * x'^5 = 13 * y'^11 → x ≤ x') →
  7 * x^5 = 13 * y^11 →
  x = a^c * b^d * e^f →
  a.val ≠ b.val ∧ b.val ≠ e.val ∧ a.val ≠ e.val →
  Nat.Prime a.val ∧ Nat.Prime b.val ∧ Nat.Prime e.val →
  a.val + b.val + c + d + e.val + f = 37 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l2909_290921


namespace NUMINAMATH_CALUDE_legendre_symbol_three_l2909_290910

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Define the theorem
theorem legendre_symbol_three (p : ℕ) (h_prime : Nat.Prime p) :
  (p % 12 = 1 ∨ p % 12 = 11 → legendre_symbol 3 p = 1) ∧
  (p % 12 = 5 ∨ p % 12 = 7 → legendre_symbol 3 p = -1) := by
  sorry

end NUMINAMATH_CALUDE_legendre_symbol_three_l2909_290910


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2909_290954

theorem gcd_lcm_sum : Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2909_290954


namespace NUMINAMATH_CALUDE_special_set_is_all_reals_l2909_290963

/-- A subset of real numbers with a special property -/
def SpecialSet (A : Set ℝ) : Prop :=
  (∀ x y : ℝ, x + y ∈ A → x * y ∈ A) ∧ Set.Nonempty A

/-- The main theorem: Any special set of real numbers is equal to the entire set of real numbers -/
theorem special_set_is_all_reals (A : Set ℝ) (h : SpecialSet A) : A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_special_set_is_all_reals_l2909_290963


namespace NUMINAMATH_CALUDE_friday_increase_is_forty_percent_l2909_290913

/-- Represents the library borrowing scenario for Krystian --/
structure LibraryBorrowing where
  dailyAverage : ℕ
  weeklyTotal : ℕ
  workdays : ℕ

/-- Calculates the percentage increase of Friday's borrowing compared to the daily average --/
def fridayPercentageIncrease (lb : LibraryBorrowing) : ℚ :=
  let fridayBorrowing := lb.weeklyTotal - (lb.workdays - 1) * lb.dailyAverage
  let increase := fridayBorrowing - lb.dailyAverage
  (increase : ℚ) / lb.dailyAverage * 100

/-- Theorem stating that the percentage increase on Friday is 40% --/
theorem friday_increase_is_forty_percent (lb : LibraryBorrowing) 
    (h1 : lb.dailyAverage = 40)
    (h2 : lb.weeklyTotal = 216)
    (h3 : lb.workdays = 5) : 
  fridayPercentageIncrease lb = 40 := by
  sorry

end NUMINAMATH_CALUDE_friday_increase_is_forty_percent_l2909_290913


namespace NUMINAMATH_CALUDE_dependent_variable_influence_l2909_290948

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ → ℝ  -- Random error term

/-- The dependent variable is influenced by both the independent variable and other factors -/
theorem dependent_variable_influence (model : LinearRegressionModel) :
  ∃ (x₁ x₂ : ℝ), model.y x₁ ≠ model.y x₂ ∧ model.x = model.x :=
by sorry

end NUMINAMATH_CALUDE_dependent_variable_influence_l2909_290948


namespace NUMINAMATH_CALUDE_smallest_hope_number_l2909_290981

def hope_number (n : ℕ+) : Prop :=
  ∃ (a b c : ℕ), 
    (n / 8 : ℚ) = a^2 ∧ 
    (n / 9 : ℚ) = b^3 ∧ 
    (n / 25 : ℚ) = c^5

theorem smallest_hope_number :
  ∃ (n : ℕ+), hope_number n ∧ 
    (∀ (m : ℕ+), hope_number m → n ≤ m) ∧
    n = 2^15 * 3^20 * 5^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_hope_number_l2909_290981


namespace NUMINAMATH_CALUDE_clients_equal_cars_l2909_290950

theorem clients_equal_cars (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 18)
  (h2 : selections_per_client = 3)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / selections_per_client = num_cars :=
by sorry

end NUMINAMATH_CALUDE_clients_equal_cars_l2909_290950


namespace NUMINAMATH_CALUDE_cos_double_angle_proof_l2909_290961

theorem cos_double_angle_proof (α : ℝ) (a : ℝ × ℝ) : 
  a = (Real.cos α, (1 : ℝ) / 2) → 
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 → 
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_proof_l2909_290961


namespace NUMINAMATH_CALUDE_sum_of_roots_even_function_l2909_290904

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    (∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) (heven : IsEven f) (hroots : HasFourRoots f) :
  ∃ (a b c d : ℝ), f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_even_function_l2909_290904


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l2909_290928

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line point.1 (f point.1) ∧
  ∀ x : ℝ, (tangent_line x (f point.1 + (x - point.1) * (3 * point.1^2 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l2909_290928


namespace NUMINAMATH_CALUDE_problem_bottle_height_l2909_290911

/-- Represents a bottle constructed from two cylinders -/
structure Bottle where
  small_radius : ℝ
  large_radius : ℝ
  water_height_right_side_up : ℝ
  water_height_upside_down : ℝ

/-- Calculates the total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  sorry

/-- The specific bottle from the problem -/
def problem_bottle : Bottle :=
  { small_radius := 1
  , large_radius := 3
  , water_height_right_side_up := 20
  , water_height_upside_down := 28 }

/-- Theorem stating that the total height of the problem bottle is 29 cm -/
theorem problem_bottle_height :
  total_height problem_bottle = 29 := by sorry

end NUMINAMATH_CALUDE_problem_bottle_height_l2909_290911


namespace NUMINAMATH_CALUDE_calculation_proof_l2909_290903

theorem calculation_proof :
  (1) * (Real.sqrt 2 + 2)^2 = 6 + 4 * Real.sqrt 2 ∧
  (2) * (Real.sqrt 3 - Real.sqrt 8) - (1/2) * (Real.sqrt 18 + Real.sqrt 12) = -(7/2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2909_290903


namespace NUMINAMATH_CALUDE_no_two_different_three_digit_cubes_l2909_290964

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i < digits.length → j < digits.length → i ≠ j → digits.get ⟨i, by sorry⟩ ≠ digits.get ⟨j, by sorry⟩

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem no_two_different_three_digit_cubes :
  ∀ KUB SHAR : ℕ,
  is_three_digit KUB →
  is_three_digit SHAR →
  all_digits_different KUB →
  all_digits_different SHAR →
  is_cube KUB →
  (∀ d : ℕ, d < 10 → (d ∈ KUB.digits 10 → d ∉ SHAR.digits 10) ∧ (d ∈ SHAR.digits 10 → d ∉ KUB.digits 10)) →
  ¬ is_cube SHAR :=
by sorry

end NUMINAMATH_CALUDE_no_two_different_three_digit_cubes_l2909_290964


namespace NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l2909_290969

/-- Given a car dealership with the following properties:
  * There are 600 cars in total
  * 60% of cars are hybrids
  * 40% of hybrids have only one headlight
  Prove that the number of hybrids with full headlights is 216 -/
theorem hybrid_cars_with_full_headlights 
  (total_cars : ℕ) 
  (hybrid_percentage : ℚ) 
  (one_headlight_percentage : ℚ) 
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : one_headlight_percentage = 40 / 100) :
  ↑total_cars * hybrid_percentage - ↑total_cars * hybrid_percentage * one_headlight_percentage = 216 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l2909_290969


namespace NUMINAMATH_CALUDE_amp_fifteen_amp_l2909_290924

-- Define the ampersand operations
def amp_right (x : ℝ) : ℝ := 8 - x
def amp_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem amp_fifteen_amp : amp_left (amp_right 15) = -16 := by
  sorry

end NUMINAMATH_CALUDE_amp_fifteen_amp_l2909_290924


namespace NUMINAMATH_CALUDE_odd_function_extension_l2909_290971

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_pos : ∀ x > 0, f x = x^2 + 1) :
  ∀ x < 0, f x = -x^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2909_290971


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2909_290936

theorem polynomial_simplification (x : ℝ) :
  (10 * x^3 - 30 * x^2 + 40 * x - 5) - (3 * x^3 - 7 * x^2 - 5 * x + 10) =
  7 * x^3 - 23 * x^2 + 45 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2909_290936


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_355_l2909_290915

theorem ones_digit_73_pow_355 : 73^355 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_355_l2909_290915


namespace NUMINAMATH_CALUDE_proportion_change_l2909_290949

def is_proportion (a b c d : ℚ) : Prop := a * d = b * c

theorem proportion_change (x y : ℚ) :
  is_proportion 3 5 6 10 →
  is_proportion 12 y 6 10 →
  y = 20 := by sorry

end NUMINAMATH_CALUDE_proportion_change_l2909_290949


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_4015_l2909_290916

/-- Represents the polynomial (x-1)^2009 + 3(x-2)^2008 + 5(x-3)^2007 + ⋯ + 4017(x-2009)^2 + 4019(x-4018) -/
def specialPolynomial : Polynomial ℝ := sorry

/-- The sum of the roots of the specialPolynomial -/
def sumOfRoots : ℝ := sorry

/-- Theorem stating that the sum of the roots of the specialPolynomial is -4015 -/
theorem sum_of_roots_is_negative_4015 : sumOfRoots = -4015 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_4015_l2909_290916


namespace NUMINAMATH_CALUDE_meeting_attendees_l2909_290978

/-- The number of handshakes in a meeting where every two people shake hands. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There were 12 people in the meeting given the conditions. -/
theorem meeting_attendees : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l2909_290978


namespace NUMINAMATH_CALUDE_paperclip_theorem_l2909_290996

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the day of the week after n days from Monday -/
def dayAfter (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Number of paperclips after n doublings -/
def paperclips (n : ℕ) : ℕ := 5 * 2^n

theorem paperclip_theorem :
  (∃ n : ℕ, paperclips n > 200 ∧ paperclips (n-1) ≤ 200) ∧
  (∀ n : ℕ, paperclips n > 200 → n ≥ 6) ∧
  dayAfter 12 = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_paperclip_theorem_l2909_290996


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2909_290906

-- Define the quadratic function
def f (x b c : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  f 1 b c = 0 →
  f 3 b c = 0 →
  f (-1) b c = 8 ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = 3) ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≥ -1) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2909_290906


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2909_290930

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = 9) ∧ (x₁^2 - 5*x₁ - 26 = 4*x₁ + 21) ∧ (x₂^2 - 5*x₂ - 26 = 4*x₂ + 21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2909_290930


namespace NUMINAMATH_CALUDE_odot_inequality_range_l2909_290984

-- Define the operation ⊙
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_range :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ x ∈ Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_odot_inequality_range_l2909_290984


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l2909_290982

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y :=
by
  use 167
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l2909_290982


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l2909_290953

theorem tax_reduction_theorem (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_consumption := original_consumption * 1.05
  let new_revenue := original_tax * original_consumption * 0.84
  let new_tax := new_revenue / new_consumption
  (original_tax - new_tax) / original_tax = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l2909_290953


namespace NUMINAMATH_CALUDE_johns_initial_contribution_l2909_290942

theorem johns_initial_contribution 
  (total_initial : ℝ) 
  (total_final : ℝ) 
  (john_initial : ℝ) 
  (kelly_initial : ℝ) 
  (luke_initial : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : total_initial = john_initial + kelly_initial + luke_initial)
  (h4 : total_final = (john_initial - 200) + 3 * kelly_initial + 3 * luke_initial) :
  john_initial = 800 := by
sorry

end NUMINAMATH_CALUDE_johns_initial_contribution_l2909_290942


namespace NUMINAMATH_CALUDE_smallest_cube_ending_888_l2909_290909

theorem smallest_cube_ending_888 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → m^3 % 1000 ≠ 888) ∧ n^3 % 1000 = 888 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_888_l2909_290909


namespace NUMINAMATH_CALUDE_sequence_2023rd_term_l2909_290925

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem sequence_2023rd_term : sequence_term 2023 = -2023 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2023rd_term_l2909_290925


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2909_290973

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  Parallel a c → Parallel b c → Parallel a b := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2909_290973


namespace NUMINAMATH_CALUDE_log_expression_equality_l2909_290958

theorem log_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l2909_290958


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2909_290922

/-- Given a toy store's revenue data for three months, prove that January's revenue is 1/5 of November's revenue. -/
theorem toy_store_revenue_ratio :
  ∀ (nov dec jan : ℝ),
  nov > 0 →
  nov = (2/5) * dec →
  dec = (25/6) * ((nov + jan) / 2) →
  jan = (1/5) * nov :=
by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2909_290922


namespace NUMINAMATH_CALUDE_fraction_equality_l2909_290955

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2909_290955


namespace NUMINAMATH_CALUDE_problem_statement_l2909_290914

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = 3) :
  (x - y)^2 * (x + y)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2909_290914


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2909_290900

/-- Calculate the selling price of a cycle given its cost price and gain percent. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) :
  cost_price = 450 →
  gain_percent = 15.56 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 520.02 := by
  sorry


end NUMINAMATH_CALUDE_cycle_selling_price_l2909_290900


namespace NUMINAMATH_CALUDE_carrie_leftover_money_l2909_290901

/-- Calculates the amount of money Carrie has left after purchasing a bike, helmet, and accessories --/
theorem carrie_leftover_money 
  (hourly_rate : ℝ)
  (hours_per_week : ℝ)
  (weeks_worked : ℝ)
  (bike_cost : ℝ)
  (sales_tax_rate : ℝ)
  (helmet_cost : ℝ)
  (accessories_cost : ℝ)
  (h1 : hourly_rate = 8)
  (h2 : hours_per_week = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400)
  (h5 : sales_tax_rate = 0.06)
  (h6 : helmet_cost = 50)
  (h7 : accessories_cost = 30) :
  hourly_rate * hours_per_week * weeks_worked - 
  (bike_cost * (1 + sales_tax_rate) + helmet_cost + accessories_cost) = 616 :=
by sorry

end NUMINAMATH_CALUDE_carrie_leftover_money_l2909_290901


namespace NUMINAMATH_CALUDE_topsoil_cost_l2909_290908

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- Theorem: The cost of 8 cubic yards of topsoil is $1728 -/
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l2909_290908


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2909_290989

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) - a k = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a n = 39 →                           -- last term is 39
  n ≥ 2 →                              -- ensure at least 3 terms
  a (n - 1) + a (n - 2) = 72 :=         -- sum of last two terms before 39
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2909_290989


namespace NUMINAMATH_CALUDE_octagonal_cube_removed_volume_l2909_290929

/-- The volume of tetrahedra removed from a cube of side length 2 to make octagonal faces -/
theorem octagonal_cube_removed_volume :
  let cube_side : ℝ := 2
  let octagon_side : ℝ := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height : ℝ := 2 / Real.sqrt 2
  let tetrahedron_base_area : ℝ := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume : ℝ := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let total_removed_volume : ℝ := 8 * single_tetrahedron_volume
  total_removed_volume = (80 - 56 * Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_octagonal_cube_removed_volume_l2909_290929


namespace NUMINAMATH_CALUDE_unique_coin_expected_value_l2909_290912

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * (-loss_tails)

theorem unique_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 4
  let loss_tails : ℚ := 3
  coin_flip_expected_value p_heads p_tails win_heads loss_tails = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_unique_coin_expected_value_l2909_290912


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2909_290986

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l2909_290986


namespace NUMINAMATH_CALUDE_cookies_for_guests_l2909_290975

/-- Given the total number of cookies and cookies per guest, calculate the number of guests. -/
def number_of_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem stating that the number of guests is 2 when there are 38 total cookies and 19 cookies per guest. -/
theorem cookies_for_guests : number_of_guests 38 19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_guests_l2909_290975


namespace NUMINAMATH_CALUDE_pocket_knife_value_l2909_290990

def is_fair_division (n : ℕ) (knife_value : ℕ) : Prop :=
  let total_revenue := n * n
  let elder_share := (total_revenue / 20) * 10
  let younger_share := ((total_revenue / 20) * 10) + (total_revenue % 20) + knife_value
  elder_share = younger_share

theorem pocket_knife_value :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (n * n) % 20 = 6 ∧ 
    is_fair_division n 2 :=
by sorry

end NUMINAMATH_CALUDE_pocket_knife_value_l2909_290990


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_fixed_points_condition_l2909_290983

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

/-- The given function f(x) = ax² + (b + 1)x + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

/-- Theorem: The function f has fixed points at 3 and -1 when a = 1 and b = -2 -/
theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) 3 ∧ is_fixed_point (f 1 (-2)) (-1) :=
sorry

/-- Theorem: The function f always has two fixed points for any real b if and only if 0 < a < 1 -/
theorem two_fixed_points_condition (a : ℝ) :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_fixed_points_condition_l2909_290983


namespace NUMINAMATH_CALUDE_square_field_area_l2909_290974

/-- The area of a square field with a diagonal of 30 meters is 450 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 30) : 
  (diagonal ^ 2) / 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2909_290974


namespace NUMINAMATH_CALUDE_expression_evaluation_l2909_290947

theorem expression_evaluation : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2909_290947


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l2909_290962

/-- The number of ways to arrange fruits over a week -/
def arrangeWeekFruits (apples oranges : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + 1) / (Nat.factorial apples * Nat.factorial oranges)

/-- The number of ways to arrange fruits over a week, excluding banana on first day -/
def arrangeWeekFruitsNoBananaFirst (apples oranges : ℕ) : ℕ :=
  (apples + oranges) * arrangeWeekFruits apples oranges

theorem cory_fruit_arrangements :
  arrangeWeekFruitsNoBananaFirst 4 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l2909_290962


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2909_290956

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧  -- four-digit number
    (n % 4 = 0) ∧             -- divisible by 4
    (n % 5 = 0) ∧             -- divisible by 5
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m % 4 = 0) ∧ (m % 5 = 0) → n ≤ m) ∧  -- smallest such number
    n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l2909_290956


namespace NUMINAMATH_CALUDE_negative_three_to_fourth_equals_three_to_fourth_l2909_290988

theorem negative_three_to_fourth_equals_three_to_fourth : (-3) * (-3) * (-3) * (-3) = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_to_fourth_equals_three_to_fourth_l2909_290988


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l2909_290959

def S (n : ℕ+) : ℕ := 2 * n - 1

def a (n : ℕ+) : ℕ := S n - S (n - 1)

theorem a_2017_equals_2 : a 2017 = 2 := by sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l2909_290959


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2909_290951

theorem ceiling_floor_product (y : ℝ) : 
  y > 0 → ⌈y⌉ * ⌊y⌋ = 72 → 8 < y ∧ y < 9 := by
sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2909_290951


namespace NUMINAMATH_CALUDE_parabola_equation_l2909_290918

/-- A parabola with vertex at the origin, focus on the y-axis, and directrix y = 3 has the equation x² = 12y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : p / 2 = 3) :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | x^2 = 2 * p * y} ↔ x^2 = 12 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2909_290918


namespace NUMINAMATH_CALUDE_one_non_negative_solution_condition_l2909_290923

/-- The quadratic equation defined by parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2*(a + 1) * x + 2*(a + 1)

/-- Predicate to check if the equation has only one non-negative solution -/
def has_one_non_negative_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic_equation a x = 0

/-- Theorem stating the condition for the equation to have only one non-negative solution -/
theorem one_non_negative_solution_condition (a : ℝ) :
  has_one_non_negative_solution a ↔ ((-1 ≤ a ∧ a ≤ 1) ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_one_non_negative_solution_condition_l2909_290923


namespace NUMINAMATH_CALUDE_last_three_sum_l2909_290937

theorem last_three_sum (a : Fin 7 → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 13)
  (h2 : (a 3 + a 4 + a 5 + a 6) / 4 = 15)
  (h3 : a 3 ^ 2 = a 6)
  (h4 : a 6 = 25) :
  a 4 + a 5 + a 6 = 55 := by
sorry

end NUMINAMATH_CALUDE_last_three_sum_l2909_290937


namespace NUMINAMATH_CALUDE_system_solutions_l2909_290994

/-- The system of equations -/
def system (x y z a : ℤ) : Prop :=
  (2*y*z + x - y - z = a) ∧
  (2*x*z - x + y - z = a) ∧
  (2*x*y - x - y + z = a)

/-- Condition for a to have four distinct integer solutions -/
def has_four_solutions (a : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ k > 0 ∧ a = (k^2 - 1) / 8

theorem system_solutions (a : ℤ) :
  (¬ ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ x₅ y₅ z₅ : ℤ,
    system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧ system x₃ y₃ z₃ a ∧
    system x₄ y₄ z₄ a ∧ system x₅ y₅ z₅ a ∧
    (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
    (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₁, y₁, z₁) ≠ (x₅, y₅, z₅) ∧
    (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧ (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧
    (x₂, y₂, z₂) ≠ (x₅, y₅, z₅) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄) ∧
    (x₃, y₃, z₃) ≠ (x₅, y₅, z₅) ∧ (x₄, y₄, z₄) ≠ (x₅, y₅, z₅)) ∧
  (has_four_solutions a ↔
    ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ : ℤ,
      system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧
      system x₃ y₃ z₃ a ∧ system x₄ y₄ z₄ a ∧
      (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
      (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧
      (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2909_290994


namespace NUMINAMATH_CALUDE_apples_collected_l2909_290927

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The number of apples Tom picked -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apples_collected : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_apples_collected_l2909_290927


namespace NUMINAMATH_CALUDE_joel_peppers_l2909_290902

/-- Represents the number of peppers picked each day of the week -/
structure WeeklyPeppers where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the number of non-hot peppers given the weekly pepper count -/
def nonHotPeppers (w : WeeklyPeppers) : ℕ :=
  let total := w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday
  (total * 4) / 5

theorem joel_peppers :
  let w : WeeklyPeppers := {
    sunday := 7,
    monday := 12,
    tuesday := 14,
    wednesday := 12,
    thursday := 5,
    friday := 18,
    saturday := 12
  }
  nonHotPeppers w = 64 := by
  sorry

end NUMINAMATH_CALUDE_joel_peppers_l2909_290902


namespace NUMINAMATH_CALUDE_price_difference_pants_belt_l2909_290905

/-- Given the total cost of pants and belt, and the price of pants, 
    calculate the difference in price between the belt and the pants. -/
theorem price_difference_pants_belt 
  (total_cost : ℝ) 
  (pants_price : ℝ) 
  (h1 : total_cost = 70.93)
  (h2 : pants_price = 34.00)
  (h3 : pants_price < total_cost - pants_price) :
  total_cost - pants_price - pants_price = 2.93 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_pants_belt_l2909_290905


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2909_290972

-- Define the line
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant :
  ¬ ∃ (x y : ℝ), line x y ∧ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2909_290972


namespace NUMINAMATH_CALUDE_central_octagon_area_l2909_290931

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square tile -/
structure SquareTile where
  sideLength : ℝ
  center : Point

/-- Theorem: Area of the central octagon in a square tile -/
theorem central_octagon_area (tile : SquareTile) (X Y Z : Point) :
  tile.sideLength = 8 →
  (X.x - Y.x)^2 + (X.y - Y.y)^2 = 2^2 →
  (Y.x - Z.x)^2 + (Y.y - Z.y)^2 = 2^2 →
  (Z.y - Y.y) / (Z.x - Y.x) = 0 →
  let U : Point := { x := (X.x + Z.x) / 2, y := (X.y + Z.y) / 2 }
  let V : Point := { x := (Y.x + Z.x) / 2, y := (Y.y + Z.y) / 2 }
  let octagonArea := (U.x - V.x)^2 + (U.y - V.y)^2 + 4 * ((X.x - U.x)^2 + (X.y - U.y)^2)
  octagonArea = 10 := by
  sorry


end NUMINAMATH_CALUDE_central_octagon_area_l2909_290931


namespace NUMINAMATH_CALUDE_smallest_multiple_of_24_and_36_not_20_l2909_290939

theorem smallest_multiple_of_24_and_36_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 24 ∣ n ∧ 36 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 24 ∣ m → 36 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm 24 36  -- This should output 72

end NUMINAMATH_CALUDE_smallest_multiple_of_24_and_36_not_20_l2909_290939


namespace NUMINAMATH_CALUDE_log_equation_solution_l2909_290987

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * (Real.log x / Real.log 3) = Real.log (4 * x^2) / Real.log 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2909_290987


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2909_290993

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℝ) : ℝ := -x^2 - 6*x + 27

/-- Theorem stating that q satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-9) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2909_290993


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_count_solution_count_equals_10626_l2909_290992

theorem non_negative_integer_solutions_count : Nat :=
  let n : Nat := 20
  let k : Nat := 5
  (n + k - 1).choose (k - 1)

theorem solution_count_equals_10626 : non_negative_integer_solutions_count = 10626 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_count_solution_count_equals_10626_l2909_290992


namespace NUMINAMATH_CALUDE_largest_M_has_property_l2909_290919

/-- The property that for any 10 distinct real numbers in [1, M], 
    there exist three that form a quadratic with no real roots -/
def has_property (M : ℝ) : Prop :=
  ∀ (a : Fin 10 → ℝ), (∀ i j, i ≠ j → a i ≠ a j) → 
  (∀ i, 1 ≤ a i ∧ a i ≤ M) →
  ∃ i j k, i < j ∧ j < k ∧ a i < a j ∧ a j < a k ∧
  (a j)^2 < 4 * (a i) * (a k)

/-- The largest integer M > 1 with the property -/
def largest_M : ℕ := 4^255

theorem largest_M_has_property :
  (has_property (largest_M : ℝ)) ∧
  ∀ n : ℕ, n > largest_M → ¬(has_property (n : ℝ)) :=
by sorry


end NUMINAMATH_CALUDE_largest_M_has_property_l2909_290919


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2909_290997

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, f x₀ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2909_290997


namespace NUMINAMATH_CALUDE_evenResultCombinations_l2909_290968

def Operation : Type := Nat → Nat

def increaseBy2 : Operation := λ n => n + 2
def increaseBy3 : Operation := λ n => n + 3
def multiplyBy2 : Operation := λ n => n * 2

def applyOperations (ops : List Operation) (initial : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) initial

def isEven (n : Nat) : Bool := n % 2 = 0

def allCombinations (n : Nat) : List (List Operation) :=
  sorry -- Implementation of all combinations of 6 operations

theorem evenResultCombinations :
  let initial := 1
  let operations := [increaseBy2, increaseBy3, multiplyBy2]
  let combinations := allCombinations 6
  (combinations.filter (λ ops => isEven (applyOperations ops initial))).length = 486 := by
  sorry

end NUMINAMATH_CALUDE_evenResultCombinations_l2909_290968


namespace NUMINAMATH_CALUDE_f_properties_l2909_290935

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ α, 0 < α ∧ α < Real.pi / 3 → (f α = 2 → α = Real.pi / 6)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2909_290935


namespace NUMINAMATH_CALUDE_convenience_store_analysis_l2909_290976

-- Define the data types
structure YearData :=
  (year : Nat)
  (profit : Real)

-- Define the dataset
def dataset : List YearData := [
  ⟨2014, 27.6⟩, ⟨2015, 42.0⟩, ⟨2016, 38.4⟩, ⟨2017, 48.0⟩, ⟨2018, 63.6⟩,
  ⟨2019, 63.7⟩, ⟨2020, 72.8⟩, ⟨2021, 80.1⟩, ⟨2022, 60.5⟩, ⟨2023, 99.3⟩
]

-- Define the contingency table
def contingencyTable : Matrix (Fin 2) (Fin 2) Nat :=
  ![![2, 5],
    ![3, 0]]

-- Define the chi-square critical value
def chiSquareCritical : Real := 3.841

-- Define the prediction year
def predictionYear : Nat := 2024

-- Define the theorem
theorem convenience_store_analysis :
  -- Chi-square value is greater than the critical value
  ∃ (chiSquareValue : Real),
    chiSquareValue > chiSquareCritical ∧
    -- Predictions from two models are different
    ∃ (prediction1 prediction2 : Real),
      prediction1 ≠ prediction2 ∧
      -- Model 1: Using data from 2014 to 2023 (excluding 2022)
      (∃ (a1 b1 : Real),
        prediction1 = a1 * predictionYear + b1 ∧
        -- Model 2: Using data from 2019 to 2023
        ∃ (a2 b2 : Real),
          prediction2 = a2 * predictionYear + b2) :=
sorry

end NUMINAMATH_CALUDE_convenience_store_analysis_l2909_290976


namespace NUMINAMATH_CALUDE_f_expression_l2909_290966

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, x ≥ 0 → f (Real.sqrt x + 1) = x + 3) →
  (∀ x : ℝ, x ≥ 0 → f (x + 1) = x^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_f_expression_l2909_290966


namespace NUMINAMATH_CALUDE_work_completion_time_l2909_290995

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 4

/-- The number of days it takes C to complete the work -/
def days_C : ℝ := 8

/-- The number of days it takes A, B, and C together to complete the work -/
def days_ABC : ℝ := 2

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + 1 / days_C = 1 / days_ABC) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2909_290995


namespace NUMINAMATH_CALUDE_rhombus_area_triple_diagonals_l2909_290999

/-- The area of a rhombus with diagonals that are 3 times longer than a rhombus
    with diagonals 6 cm and 4 cm is 108 cm². -/
theorem rhombus_area_triple_diagonals (d1 d2 : ℝ) : 
  d1 = 6 → d2 = 4 → (3 * d1 * 3 * d2) / 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_triple_diagonals_l2909_290999


namespace NUMINAMATH_CALUDE_power_sum_equality_l2909_290967

theorem power_sum_equality : (-2 : ℤ) ^ (4 ^ 2) + 2 ^ (3 ^ 2) = 66048 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2909_290967


namespace NUMINAMATH_CALUDE_amp_composition_l2909_290926

-- Define the & operation (postfix)
def amp (x : ℤ) : ℤ := 7 - x

-- Define the & operation (prefix)
def amp_prefix (x : ℤ) : ℤ := x - 10

-- Theorem statement
theorem amp_composition : amp_prefix (amp 12) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_composition_l2909_290926


namespace NUMINAMATH_CALUDE_expression_simplification_l2909_290991

theorem expression_simplification (x : ℚ) :
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2909_290991


namespace NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l2909_290943

-- Define the function f(x) = 2x³ - ax² + 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 1

-- Define the interval [1/2, 2]
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Define the condition of having exactly two zeros in the interval
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z, z ∈ interval ∧ f a z = 0 → z = x ∨ z = y

-- State the theorem
theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_two_zeros a ↔ 3/2 < a ∧ a ≤ 17/4 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l2909_290943


namespace NUMINAMATH_CALUDE_unique_number_l2909_290940

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            n % 2 = 1 ∧ 
            n % 13 = 0 ∧ 
            is_perfect_square (digits_product n) ∧
            n = 91 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2909_290940


namespace NUMINAMATH_CALUDE_ln_power_rational_l2909_290952

theorem ln_power_rational (f : ℝ) (r : ℚ) (hf : f > 0) :
  Real.log (f ^ (r : ℝ)) = r * Real.log f := by
  sorry

end NUMINAMATH_CALUDE_ln_power_rational_l2909_290952
