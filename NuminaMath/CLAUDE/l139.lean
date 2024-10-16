import Mathlib

namespace NUMINAMATH_CALUDE_circle_line_intersection_max_k_l139_13913

theorem circle_line_intersection_max_k : 
  ∃ (k_max : ℝ),
    k_max = 4/3 ∧
    ∀ (k : ℝ),
      (∃ (x₀ y₀ : ℝ),
        y₀ = k * x₀ - 2 ∧
        ∃ (x y : ℝ),
          (x - 4)^2 + y^2 = 1 ∧
          (x - x₀)^2 + (y - y₀)^2 ≤ 1) →
      k ≤ k_max :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_max_k_l139_13913


namespace NUMINAMATH_CALUDE_telescope_visibility_increase_l139_13926

theorem telescope_visibility_increase (min_without max_without min_with max_with : ℝ) 
  (h1 : min_without = 100)
  (h2 : max_without = 110)
  (h3 : min_with = 150)
  (h4 : max_with = 165) :
  let avg_without := (min_without + max_without) / 2
  let avg_with := (min_with + max_with) / 2
  (avg_with - avg_without) / avg_without * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visibility_increase_l139_13926


namespace NUMINAMATH_CALUDE_pollys_age_equals_sum_of_children_ages_l139_13906

/-- Represents Polly's age when it equals the sum of her three children's ages -/
def pollys_age : ℕ := 33

/-- Represents the age of Polly's first child -/
def first_child_age (x : ℕ) : ℕ := x - 20

/-- Represents the age of Polly's second child -/
def second_child_age (x : ℕ) : ℕ := x - 22

/-- Represents the age of Polly's third child -/
def third_child_age (x : ℕ) : ℕ := x - 24

/-- Theorem stating that Polly's age equals the sum of her three children's ages -/
theorem pollys_age_equals_sum_of_children_ages :
  pollys_age = first_child_age pollys_age + second_child_age pollys_age + third_child_age pollys_age :=
by sorry

end NUMINAMATH_CALUDE_pollys_age_equals_sum_of_children_ages_l139_13906


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_odd_sides_l139_13921

theorem right_triangle_consecutive_odd_sides (k : ℤ) :
  let a : ℤ := 2 * k + 1
  let c : ℤ := 2 * k + 3
  let b : ℤ := (c^2 - a^2).sqrt
  (a^2 + b^2 = c^2) → (b^2 = 8 * k + 8) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_odd_sides_l139_13921


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l139_13905

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 - 2
  ((x + 3) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 + x)) = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l139_13905


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l139_13985

theorem sally_bread_consumption :
  let saturday_sandwiches : ℕ := 2
  let sunday_sandwiches : ℕ := 1
  let bread_per_sandwich : ℕ := 2
  let total_sandwiches := saturday_sandwiches + sunday_sandwiches
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l139_13985


namespace NUMINAMATH_CALUDE_max_planes_15_points_l139_13963

/-- The maximum number of planes determined by 15 points in space, where no four points are coplanar -/
def max_planes (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- Theorem stating that the maximum number of planes determined by 15 points in space, 
    where no four points are coplanar, is equal to 455 -/
theorem max_planes_15_points : max_planes 15 = 455 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_15_points_l139_13963


namespace NUMINAMATH_CALUDE_julia_basketball_success_rate_increase_l139_13969

theorem julia_basketball_success_rate_increase :
  let initial_success : ℕ := 3
  let initial_attempts : ℕ := 8
  let subsequent_success : ℕ := 12
  let subsequent_attempts : ℕ := 16
  let total_success := initial_success + subsequent_success
  let total_attempts := initial_attempts + subsequent_attempts
  let initial_rate := initial_success / initial_attempts
  let final_rate := total_success / total_attempts
  final_rate - initial_rate = 1/4 := by sorry

end NUMINAMATH_CALUDE_julia_basketball_success_rate_increase_l139_13969


namespace NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l139_13938

-- Part 1
theorem sqrt_3_times_612_times_3_and_half (x : ℝ) :
  x = Real.sqrt 3 * 612 * (3 + 3/2) → x = 3 := by sorry

-- Part 2
theorem log_squared_difference_plus_log (x : ℝ) :
  x = (Real.log 5 / Real.log 10)^2 - (Real.log 2 / Real.log 10)^2 + (Real.log 4 / Real.log 10) → x = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l139_13938


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_corners_l139_13912

/-- The area of a square with adjacent corners at (1, 2) and (5, 6) is 32. -/
theorem square_area_from_adjacent_corners : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (5, 6)
  let side_length := ((b.1 - a.1)^2 + (b.2 - a.2)^2).sqrt
  side_length^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_corners_l139_13912


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l139_13976

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧ 
  (∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ n) ∧
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l139_13976


namespace NUMINAMATH_CALUDE_grape_rate_proof_l139_13901

/-- The rate of grapes per kilogram -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kilograms -/
def grape_amount : ℝ := 8

/-- The rate of mangoes per kilogram -/
def mango_rate : ℝ := 60

/-- The amount of mangoes purchased in kilograms -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1100

theorem grape_rate_proof : 
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_proof_l139_13901


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l139_13989

/-- The line equation 5y - 3x = 15 intersects the x-axis at the point (-5, 0). -/
theorem line_intersects_x_axis :
  ∃ (x y : ℝ), 5 * y - 3 * x = 15 ∧ y = 0 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l139_13989


namespace NUMINAMATH_CALUDE_machine_selling_price_l139_13991

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 22500 Rs -/
theorem machine_selling_price :
  selling_price 9000 5000 1000 50 = 22500 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l139_13991


namespace NUMINAMATH_CALUDE_unique_functional_equation_l139_13916

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l139_13916


namespace NUMINAMATH_CALUDE_parallelepiped_intersection_length_l139_13949

/-- A parallelepiped with points A, B, C, D, A₁, B₁, C₁, D₁ -/
structure Parallelepiped (V : Type*) [NormedAddCommGroup V] :=
  (A B C D A₁ B₁ C₁ D₁ : V)

/-- Point X on edge A₁D₁ -/
def X {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.A₁ + 5 • (p.D₁ - p.A₁)

/-- Point Y on edge BC -/
def Y {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.B + 3 • (p.C - p.B)

/-- Intersection point Z of plane C₁XY and ray DA -/
noncomputable def Z {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  sorry

/-- Theorem stating that DZ = 20 -/
theorem parallelepiped_intersection_length
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (p : Parallelepiped V) :
  ‖p.D - Z p‖ = 20 ∧ ‖p.B₁ - p.C₁‖ = 14 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_intersection_length_l139_13949


namespace NUMINAMATH_CALUDE_math_problem_time_calculation_l139_13994

theorem math_problem_time_calculation 
  (num_problems : ℕ) 
  (time_per_problem : ℕ) 
  (checking_time : ℕ) : 
  num_problems = 7 → 
  time_per_problem = 4 → 
  checking_time = 3 → 
  num_problems * time_per_problem + checking_time = 31 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_time_calculation_l139_13994


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_of_2940_l139_13920

/-- The number of positive integer factors of 2940 that are perfect squares -/
def num_perfect_square_factors : ℕ := 4

/-- The prime factorization of 2940 -/
def prime_factorization_2940 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1), (7, 1)]

/-- A function to check if a list represents a valid prime factorization -/
def is_valid_prime_factorization (l : List (ℕ × ℕ)) : Prop :=
  l.all (fun (p, e) => Nat.Prime p ∧ e > 0)

/-- A function to compute the product of a prime factorization -/
def product_of_factorization (l : List (ℕ × ℕ)) : ℕ :=
  l.foldl (fun acc (p, e) => acc * p^e) 1

theorem num_perfect_square_factors_of_2940 :
  is_valid_prime_factorization prime_factorization_2940 ∧
  product_of_factorization prime_factorization_2940 = 2940 →
  num_perfect_square_factors = (List.filter (fun (_, e) => e % 2 = 0) prime_factorization_2940).length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_of_2940_l139_13920


namespace NUMINAMATH_CALUDE_keychain_arrangement_count_l139_13911

def num_keys : ℕ := 7

def num_distinct_arrangements (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else (n - 1).factorial / 2

theorem keychain_arrangement_count :
  num_distinct_arrangements (num_keys - 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_keychain_arrangement_count_l139_13911


namespace NUMINAMATH_CALUDE_complex_equation_solution_l139_13900

theorem complex_equation_solution (z : ℂ) : 
  z + 3*I - 3 = 6 - 3*I → z = 9 - 6*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l139_13900


namespace NUMINAMATH_CALUDE_line_for_equal_diagonals_l139_13948

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the line l passing through (-1, 0)
def l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂

-- Define vector OS as the sum of OA and OB
def vectorOS (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x = x₁ + x₂ ∧ y = y₁ + y₂

-- Define the condition for equal diagonals in quadrilateral OASB
def equalDiagonals (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x₁^2 + y₁^2 = x₂^2 + y₂^2

-- Theorem statement
theorem line_for_equal_diagonals :
  ∃! k, intersectionPoints k ∧ equalDiagonals k ∧ k = 1 :=
sorry

end NUMINAMATH_CALUDE_line_for_equal_diagonals_l139_13948


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l139_13918

theorem complex_modulus_problem (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) + 2 * Complex.I → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l139_13918


namespace NUMINAMATH_CALUDE_determinant_zero_l139_13981

theorem determinant_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![![0, Real.cos α, -Real.sin α],
                                           ![-Real.cos α, 0, Real.cos β],
                                           ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l139_13981


namespace NUMINAMATH_CALUDE_problem_solution_l139_13957

theorem problem_solution (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬p) : 
  ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l139_13957


namespace NUMINAMATH_CALUDE_prob_first_qualified_on_third_test_l139_13942

/-- The probability of obtaining the first qualified product on the third test. -/
def P_epsilon_3 (pass_rate : ℝ) (fail_rate : ℝ) : ℝ :=
  fail_rate^2 * pass_rate

/-- The theorem stating that P(ε = 3) is equal to (1/4)² × (3/4) given the specified pass and fail rates. -/
theorem prob_first_qualified_on_third_test :
  let pass_rate : ℝ := 3/4
  let fail_rate : ℝ := 1/4
  P_epsilon_3 pass_rate fail_rate = (1/4)^2 * (3/4) :=
by sorry

end NUMINAMATH_CALUDE_prob_first_qualified_on_third_test_l139_13942


namespace NUMINAMATH_CALUDE_least_four_digit_with_conditions_l139_13983

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def contains_digit (n d : ℕ) : Prop :=
  d ∈ n.digits 10

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_with_conditions :
  ∀ n : ℕ,
    is_four_digit n ∧
    has_different_digits n ∧
    contains_digit n 5 ∧
    divisible_by_digits n →
    5124 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_with_conditions_l139_13983


namespace NUMINAMATH_CALUDE_confectioner_pastries_l139_13927

theorem confectioner_pastries :
  ∀ (total_pastries : ℕ) 
    (regular_customers : ℕ) 
    (actual_customers : ℕ) 
    (pastry_difference : ℕ),
  regular_customers = 28 →
  actual_customers = 49 →
  pastry_difference = 6 →
  regular_customers * (total_pastries / regular_customers) = 
    actual_customers * (total_pastries / regular_customers - pastry_difference) →
  total_pastries = 1176 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastries_l139_13927


namespace NUMINAMATH_CALUDE_three_in_range_of_g_l139_13950

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

-- Theorem statement
theorem three_in_range_of_g (a : ℝ) : ∃ x : ℝ, g a x = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_g_l139_13950


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l139_13909

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x - y + 1 = 0
  line 0 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l139_13909


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l139_13922

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 / (x - 1) = 1 / x) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l139_13922


namespace NUMINAMATH_CALUDE_product_divisibility_l139_13979

/-- Given two lists of positive integers of equal length, where the number of multiples
    of any d > 1 in the first list is no less than that in the second list,
    prove that the product of the first list is divisible by the product of the second list. -/
theorem product_divisibility
  (r : ℕ)
  (m n : List ℕ)
  (h_length : m.length = r ∧ n.length = r)
  (h_positive : ∀ x ∈ m, x > 0) (h_positive' : ∀ y ∈ n, y > 0)
  (h_multiples : ∀ d > 1, (m.filter (· % d = 0)).length ≥ (n.filter (· % d = 0)).length) :
  (m.prod % n.prod = 0) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l139_13979


namespace NUMINAMATH_CALUDE_calculate_expression_l139_13951

theorem calculate_expression : 
  75 * (4 + 1/3 - (5 + 1/4)) / (3 + 1/2 + 2 + 1/5) = -5/31 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l139_13951


namespace NUMINAMATH_CALUDE_bikes_added_per_week_l139_13966

/-- 
Proves that the number of bikes added per week is 3, given the initial stock,
bikes sold in a month, stock after one month, and the number of weeks in a month.
-/
theorem bikes_added_per_week 
  (initial_stock : ℕ) 
  (bikes_sold : ℕ) 
  (final_stock : ℕ) 
  (weeks_in_month : ℕ) 
  (h1 : initial_stock = 51)
  (h2 : bikes_sold = 18)
  (h3 : final_stock = 45)
  (h4 : weeks_in_month = 4)
  : (final_stock - (initial_stock - bikes_sold)) / weeks_in_month = 3 := by
  sorry

end NUMINAMATH_CALUDE_bikes_added_per_week_l139_13966


namespace NUMINAMATH_CALUDE_max_value_constraint_l139_13973

theorem max_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + y - 3 ≤ 0) (h2 : 2 * x + y - 4 ≥ 0) : 
  2 * x + 3 * y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l139_13973


namespace NUMINAMATH_CALUDE_max_side_length_l139_13910

/-- A triangle with three different integer side lengths and a perimeter of 20 units -/
structure TriangleWithConstraints where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 20

/-- The maximum length of any side in a TriangleWithConstraints is 9 -/
theorem max_side_length (t : TriangleWithConstraints) :
  max t.a (max t.b t.c) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_l139_13910


namespace NUMINAMATH_CALUDE_batting_average_increase_l139_13959

theorem batting_average_increase (current_average : ℚ) (matches_played : ℕ) (new_average : ℚ) : 
  current_average = 52 →
  matches_played = 12 →
  new_average = 54 →
  (new_average * (matches_played + 1) - current_average * matches_played : ℚ) = 78 := by
sorry

end NUMINAMATH_CALUDE_batting_average_increase_l139_13959


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_fourteen_l139_13937

/-- A cubic function f(x) with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

/-- Theorem stating that f(-2) = -14 given the conditions -/
theorem f_neg_two_eq_neg_fourteen (a b : ℝ) :
  (f a b 2 = 6) → (f a b (-2) = -14) := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_fourteen_l139_13937


namespace NUMINAMATH_CALUDE_orange_selling_price_l139_13952

/-- Proves that the selling price of each orange is 60 cents given the conditions -/
theorem orange_selling_price (total_cost : ℚ) (num_oranges : ℕ) (profit_per_orange : ℚ) :
  total_cost = 25 / 2 →
  num_oranges = 25 →
  profit_per_orange = 1 / 10 →
  (total_cost / num_oranges + profit_per_orange) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_orange_selling_price_l139_13952


namespace NUMINAMATH_CALUDE_gasoline_price_theorem_l139_13902

/-- Represents the price changes of gasoline over four months -/
def GasolinePriceChanges (initial_price : ℝ) : Prop :=
  ∃ (x : ℝ),
    let jan_price := initial_price * 1.30
    let feb_price := jan_price * 0.90 * 1.15
    let mar_price := feb_price * 1.15
    let apr_price := mar_price * (1 - x / 100)
    (apr_price = initial_price) ∧ (35 ≤ x) ∧ (x < 36)

/-- Theorem stating that there exists a solution to the gasoline price problem -/
theorem gasoline_price_theorem :
  ∃ (initial_price : ℝ), GasolinePriceChanges initial_price :=
sorry

end NUMINAMATH_CALUDE_gasoline_price_theorem_l139_13902


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_ratio_l139_13974

theorem larger_number_given_hcf_lcm_ratio (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 84)
  (lcm_eq : Nat.lcm a b = 21)
  (ratio : a * 4 = b) :
  max a b = 84 := by
sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_ratio_l139_13974


namespace NUMINAMATH_CALUDE_stating_distribution_schemes_count_l139_13971

/-- Represents the number of schools --/
def num_schools : ℕ := 5

/-- Represents the number of computers --/
def num_computers : ℕ := 6

/-- Represents the number of schools that must receive at least 2 computers --/
def num_special_schools : ℕ := 2

/-- Represents the minimum number of computers each special school must receive --/
def min_computers_per_special_school : ℕ := 2

/-- 
Calculates the number of ways to distribute computers to schools 
under the given constraints
--/
def distribution_schemes : ℕ := sorry

/-- 
Theorem stating that the number of distribution schemes is 15
--/
theorem distribution_schemes_count : distribution_schemes = 15 := by sorry

end NUMINAMATH_CALUDE_stating_distribution_schemes_count_l139_13971


namespace NUMINAMATH_CALUDE_max_m_value_l139_13955

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m * a * b / (3 * a + b) ≤ a + 3 * b) ↔ m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l139_13955


namespace NUMINAMATH_CALUDE_negation_of_proposition_l139_13968

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l139_13968


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_product_l139_13970

theorem arithmetic_mean_geq_product (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_product_l139_13970


namespace NUMINAMATH_CALUDE_no_prime_pairs_for_square_diff_l139_13947

theorem no_prime_pairs_for_square_diff (a b : ℕ) : 
  a ≤ 100 → b ≤ 100 → Prime a → Prime b → a^2 - b^2 ≠ 25 :=
by sorry

end NUMINAMATH_CALUDE_no_prime_pairs_for_square_diff_l139_13947


namespace NUMINAMATH_CALUDE_west_movement_l139_13929

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (∀ (d : ℤ), movement d Direction.East = d) →
  (∀ (d : ℤ), movement d Direction.West = -d) →
  movement 5 Direction.West = -5 := by
  sorry

end NUMINAMATH_CALUDE_west_movement_l139_13929


namespace NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l139_13907

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | SimpleRandom

/-- Represents a population with plots -/
structure Population where
  totalPlots : ℕ
  sampleSize : ℕ
  highVariability : Bool

/-- Determines if a sampling method is reasonable given a population -/
def isReasonableSamplingMethod (p : Population) (m : SamplingMethod) : Prop :=
  p.highVariability → m = SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most reasonable method
    for a population with high variability -/
theorem stratified_sampling_most_reasonable (p : Population) 
    (h1 : p.totalPlots = 200)
    (h2 : p.sampleSize = 20)
    (h3 : p.highVariability = true) :
    isReasonableSamplingMethod p SamplingMethod.Stratified :=
  sorry

#check stratified_sampling_most_reasonable

end NUMINAMATH_CALUDE_stratified_sampling_most_reasonable_l139_13907


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l139_13935

/-- Given two vectors a and b in ℝ², prove that if a = (3,1) and b = (x,-1) are parallel, then x = -3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l139_13935


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l139_13945

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-1)*x - m*(m+2) = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*(m-1)*y - m*(m+2) = 0) ∧
  ((-2)^2 - 2*(m-1)*(-2) - m*(m+2) = 0 → 2018 - 3*(m-1)^2 = 2015) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l139_13945


namespace NUMINAMATH_CALUDE_M_closed_under_multiplication_l139_13917

def M : Set ℕ := {n | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem M_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ M → b ∈ M → (a * b) ∈ M :=
by
  sorry

end NUMINAMATH_CALUDE_M_closed_under_multiplication_l139_13917


namespace NUMINAMATH_CALUDE_cosine_rule_with_ratio_l139_13958

theorem cosine_rule_with_ratio (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_rule_with_ratio_l139_13958


namespace NUMINAMATH_CALUDE_hua_luogeng_uses_golden_ratio_l139_13967

-- Define the possible methods for optimal selection
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

-- Define Hua Luogeng's optimal selection method
def huaLuogengMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

-- Theorem stating that Hua Luogeng's method uses the golden ratio
theorem hua_luogeng_uses_golden_ratio :
  huaLuogengMethod = OptimalSelectionMethod.GoldenRatio := by sorry

end NUMINAMATH_CALUDE_hua_luogeng_uses_golden_ratio_l139_13967


namespace NUMINAMATH_CALUDE_prop_p_necessary_not_sufficient_for_q_l139_13941

theorem prop_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, x + y ≠ 4 → (x ≠ 1 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 3) ∧ x + y = 4) :=
by sorry

end NUMINAMATH_CALUDE_prop_p_necessary_not_sufficient_for_q_l139_13941


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l139_13933

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  (Real.sqrt (x^4 + x^2 + 2*x + 1) + Real.sqrt (x^4 - 2*x^3 + 5*x^2 - 4*x + 1)) / x ≥ Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l139_13933


namespace NUMINAMATH_CALUDE_farthest_point_l139_13965

def points : List (ℝ × ℝ) := [(0, 7), (2, 3), (-4, 1), (5, -5), (7, 0)]

def distance_squared (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 2 + p.2 ^ 2

theorem farthest_point :
  ∀ p ∈ points, distance_squared (5, -5) ≥ distance_squared p :=
by sorry

end NUMINAMATH_CALUDE_farthest_point_l139_13965


namespace NUMINAMATH_CALUDE_square_difference_l139_13930

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l139_13930


namespace NUMINAMATH_CALUDE_sin_four_arcsin_one_fourth_l139_13984

theorem sin_four_arcsin_one_fourth :
  Real.sin (4 * Real.arcsin (1/4)) = 7 * Real.sqrt 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_four_arcsin_one_fourth_l139_13984


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l139_13977

/-- Given that the solution set of x² - ax - b < 0 is (2, 3), 
    prove that the solution set of bx² - ax - 1 > 0 is (-1/2, -1/3) -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l139_13977


namespace NUMINAMATH_CALUDE_marble_probability_l139_13924

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (box1 box2 : ℕ) :
  total_marbles = 30 →
  box1 + box2 = total_marbles →
  prob_both_black = 3/5 →
  box1 > 0 ∧ box2 > 0 →
  ∃ (black1 black2 : ℕ),
    black1 ≤ box1 ∧ black2 ≤ box2 ∧
    (black1 : ℚ) / box1 * (black2 : ℚ) / box2 = prob_both_black →
    ((box1 - black1 : ℚ) / box1 * (box2 - black2 : ℚ) / box2 = 4/25) :=
by sorry

#check marble_probability

end NUMINAMATH_CALUDE_marble_probability_l139_13924


namespace NUMINAMATH_CALUDE_distance_from_point_to_y_axis_l139_13997

-- Define a point in 2D Cartesian coordinate system
def point : ℝ × ℝ := (3, -5)

-- Define the distance from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

-- Theorem statement
theorem distance_from_point_to_y_axis :
  distance_to_y_axis point = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_y_axis_l139_13997


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l139_13990

/-- The number of tissues in a mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_size : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_size : ℕ := 10

/-- The number of students in the third kindergartner group -/
def group3_size : ℕ := 11

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group1_size + group2_size + group3_size) * tissues_per_box

theorem kindergarten_tissues : total_tissues = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l139_13990


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l139_13946

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l139_13946


namespace NUMINAMATH_CALUDE_ellipse_k_range_l139_13956

/-- The curve equation --/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

/-- The curve represents an ellipse --/
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

/-- The range of k for which the curve represents an ellipse --/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l139_13956


namespace NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l139_13964

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_square_faces : Nat
  /-- Number of regular hexagonal faces -/
  num_hexagonal_faces : Nat
  /-- Number of regular octagonal faces -/
  num_octagonal_faces : Nat
  /-- Property that exactly one square, one hexagon, and one octagon meet at each vertex -/
  vertex_property : Bool

/-- Calculate the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem stating the number of interior segments in the special polyhedron -/
theorem special_polyhedron_interior_segments 
  (p : SpecialPolyhedron) 
  (h1 : p.is_convex = true)
  (h2 : p.num_square_faces = 12)
  (h3 : p.num_hexagonal_faces = 8)
  (h4 : p.num_octagonal_faces = 6)
  (h5 : p.vertex_property = true) :
  interior_segments p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l139_13964


namespace NUMINAMATH_CALUDE_smaller_to_larger_base_ratio_l139_13987

/-- An isosceles trapezoid with an inscribed equilateral triangle -/
structure IsoscelesTrapezoidWithTriangle where
  /-- Length of the smaller base (and side of the equilateral triangle) -/
  s : ℝ
  /-- Length of the larger base -/
  b : ℝ
  /-- s is positive -/
  s_pos : 0 < s
  /-- b is positive -/
  b_pos : 0 < b
  /-- The larger base is twice the length of a diagonal of the equilateral triangle -/
  diag_relation : b = 2 * s

/-- The ratio of the smaller base to the larger base is 1/2 -/
theorem smaller_to_larger_base_ratio 
  (t : IsoscelesTrapezoidWithTriangle) : t.s / t.b = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_smaller_to_larger_base_ratio_l139_13987


namespace NUMINAMATH_CALUDE_lamp_distribution_and_profit_l139_13996

/-- Represents the types of lamps --/
inductive LampType
| A
| B

/-- Represents the purchase price of a lamp --/
def purchasePrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 40
  | LampType.B => 65

/-- Represents the selling price of a lamp --/
def sellingPrice (t : LampType) : ℕ :=
  match t with
  | LampType.A => 60
  | LampType.B => 100

/-- Represents the profit from selling a lamp --/
def profit (t : LampType) : ℕ := sellingPrice t - purchasePrice t

/-- The total number of lamps --/
def totalLamps : ℕ := 50

/-- The total purchase cost --/
def totalPurchaseCost : ℕ := 2500

/-- The minimum total profit --/
def minTotalProfit : ℕ := 1400

theorem lamp_distribution_and_profit :
  (∃ (x y : ℕ),
    x + y = totalLamps ∧
    x * purchasePrice LampType.A + y * purchasePrice LampType.B = totalPurchaseCost ∧
    x = 30 ∧ y = 20) ∧
  (∃ (m : ℕ),
    m * profit LampType.B + (totalLamps - m) * profit LampType.A ≥ minTotalProfit ∧
    m ≥ 27 ∧
    ∀ (n : ℕ), n * profit LampType.B + (totalLamps - n) * profit LampType.A ≥ minTotalProfit → n ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_lamp_distribution_and_profit_l139_13996


namespace NUMINAMATH_CALUDE_donut_shop_problem_l139_13939

def donut_combinations (total_donuts : ℕ) (types : ℕ) : ℕ :=
  let remaining := total_donuts - types
  (types.choose 1) * (types.choose 1) * (types.choose 1) + 
  (types.choose 2) * (remaining.choose 2) +
  (types.choose 3) * (remaining.choose 1)

theorem donut_shop_problem :
  donut_combinations 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_donut_shop_problem_l139_13939


namespace NUMINAMATH_CALUDE_leftover_pie_share_l139_13972

theorem leftover_pie_share (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 12 / 13 → num_people = 4 → total_pie / num_people = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_leftover_pie_share_l139_13972


namespace NUMINAMATH_CALUDE_prob_not_overcome_is_half_l139_13998

-- Define the set of elements
inductive Element : Type
| Metal : Element
| Wood : Element
| Water : Element
| Fire : Element
| Earth : Element

-- Define the overcoming relation
def overcomes : Element → Element → Prop
| Element.Metal, Element.Wood => True
| Element.Wood, Element.Earth => True
| Element.Earth, Element.Water => True
| Element.Water, Element.Fire => True
| Element.Fire, Element.Metal => True
| _, _ => False

-- Define the probability of selecting two elements that do not overcome each other
def prob_not_overcome : ℚ :=
  let total_pairs := (5 * 4) / 2  -- C(5,2)
  let overcoming_pairs := 5       -- Number of overcoming relationships
  1 - (overcoming_pairs : ℚ) / total_pairs

-- State the theorem
theorem prob_not_overcome_is_half : prob_not_overcome = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_overcome_is_half_l139_13998


namespace NUMINAMATH_CALUDE_polynomial_coefficient_identity_l139_13932

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_identity_l139_13932


namespace NUMINAMATH_CALUDE_solution_set_xfx_lt_zero_l139_13978

/-- The function f satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

/-- Theorem stating the solution set of xf(x) < 0 -/
theorem solution_set_xfx_lt_zero :
  {x : ℝ | x * f x < 0} = {x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_xfx_lt_zero_l139_13978


namespace NUMINAMATH_CALUDE_novel_distribution_count_l139_13960

/-- The number of ways to distribute 4 novels among 5 students -/
def novel_distribution : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of novel distributions is 240 -/
theorem novel_distribution_count : novel_distribution = 240 := by
  sorry

end NUMINAMATH_CALUDE_novel_distribution_count_l139_13960


namespace NUMINAMATH_CALUDE_prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l139_13962

-- Proposition A (false)
theorem prop_a_false : ¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a) := by sorry

-- Proposition B (true)
theorem prop_b_true : ∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2 := by sorry

-- Proposition C (true)
theorem prop_c_true : ∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b) := by sorry

-- Proposition D (true)
theorem prop_d_true : ∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d) := by sorry

-- Combined theorem
theorem propositions_bcd_true_a_false : 
  (¬ (∀ a b : ℝ, a > b → 1 / b > 1 / a)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a*c^2 > b*c^2 → a > b)) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → 1 / (a - c) < 1 / (b - d)) := by
  exact ⟨prop_a_false, prop_b_true, prop_c_true, prop_d_true⟩

end NUMINAMATH_CALUDE_prop_a_false_prop_b_true_prop_c_true_prop_d_true_propositions_bcd_true_a_false_l139_13962


namespace NUMINAMATH_CALUDE_max_roses_theorem_l139_13993

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : Nat  -- Price in cents for an individual rose
  dozen : Nat       -- Price in cents for a dozen roses
  two_dozen : Nat   -- Price in cents for two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def max_roses_purchasable (pricing : RosePricing) (budget : Nat) : Nat :=
  sorry

/-- The theorem stating the maximum number of roses purchasable with the given pricing and budget -/
theorem max_roses_theorem (pricing : RosePricing) (budget : Nat) :
  pricing.individual = 630 →
  pricing.dozen = 3600 →
  pricing.two_dozen = 5000 →
  budget = 68000 →
  max_roses_purchasable pricing budget = 316 :=
sorry

end NUMINAMATH_CALUDE_max_roses_theorem_l139_13993


namespace NUMINAMATH_CALUDE_infinite_series_sum_l139_13961

theorem infinite_series_sum : 
  (∑' n : ℕ, (n^2 + 3*n + 2) / (n * (n + 1) * (n + 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l139_13961


namespace NUMINAMATH_CALUDE_first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l139_13914

-- Define the set of integer solutions
def IntegerSolutions : Set ℤ := {0, 1, 2}

-- Theorem for the first inequality
theorem first_inequality_solution (x : ℝ) :
  3 * (2 * x + 2) > 4 * x - 1 + 7 ↔ x > -3/2 := by sorry

-- Theorem for the system of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x + 1 > 0 ∧ x ≤ (x - 2) / 3 + 2) ↔ (-1 < x ∧ x ≤ 2) := by sorry

-- Theorem for integer solutions
theorem integer_solutions_correct :
  ∀ (n : ℤ), n ∈ IntegerSolutions ↔ (n + 1 > 0 ∧ n ≤ (n - 2) / 3 + 2) := by sorry

end NUMINAMATH_CALUDE_first_inequality_solution_system_of_inequalities_solution_integer_solutions_correct_l139_13914


namespace NUMINAMATH_CALUDE_product_of_integers_l139_13925

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 2 * a * b + 12 * a = 3 * b + 240 → a * b = 255 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l139_13925


namespace NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l139_13975

def repeated_number (n : ℕ) : ℕ := 1001001001 * n

theorem gcd_of_repeated_numbers :
  ∃ (m : ℕ), m > 0 ∧ m < 1000 ∧
  (∀ (n : ℕ), n > 0 ∧ n < 1000 → Nat.gcd (repeated_number m) (repeated_number n) = 1001001001) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l139_13975


namespace NUMINAMATH_CALUDE_customers_per_car_l139_13944

theorem customers_per_car (num_cars : ℕ) (total_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : total_sales = 50) : 
  ∃ (customers_per_car : ℕ), 
    customers_per_car * num_cars = total_sales ∧ 
    customers_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_customers_per_car_l139_13944


namespace NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_zero_power_l139_13928

theorem same_solution_implies_a_plus_b_zero_power (a b : ℝ) :
  (∃ (x y : ℝ), 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 3*x - 5*y = 1 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_zero_power_l139_13928


namespace NUMINAMATH_CALUDE_range_of_a_l139_13954

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (0 < a ∧ a ≤ 1/3) ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l139_13954


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l139_13982

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 7 →
  original_list.sum / original_list.length = 48 →
  let new_list := original_list ++ [x, y, z]
  new_list.sum / new_list.length = 55 →
  (x + y + z) / 3 = 71 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l139_13982


namespace NUMINAMATH_CALUDE_area_of_triangle_AEB_l139_13995

-- Define the points
variable (A B C D E F G : Euclidean_plane)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Euclidean_plane) : Prop := sorry

-- Define the lengths
def length (P Q : Euclidean_plane) : ℝ := sorry

-- Define a point being on a line segment
def on_segment (P Q R : Euclidean_plane) : Prop := sorry

-- Define line intersection
def intersect (P Q R S : Euclidean_plane) : Euclidean_plane := sorry

-- Define triangle area
def triangle_area (P Q R : Euclidean_plane) : ℝ := sorry

theorem area_of_triangle_AEB 
  (h_rect : is_rectangle A B C D)
  (h_AB : length A B = 10)
  (h_BC : length B C = 4)
  (h_F_on_CD : on_segment C D F)
  (h_G_on_CD : on_segment C D G)
  (h_DF : length D F = 2)
  (h_GC : length G C = 3)
  (h_E : E = intersect A F B G) :
  triangle_area A E B = 40 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEB_l139_13995


namespace NUMINAMATH_CALUDE_jimmy_water_consumption_l139_13931

/-- Represents the amount of water Jimmy drinks each time in ounces -/
def water_per_time (times_per_day : ℕ) (days : ℕ) (total_gallons : ℚ) (ounce_to_gallon : ℚ) : ℚ :=
  (total_gallons / ounce_to_gallon) / (times_per_day * days)

/-- Theorem stating that Jimmy drinks 8 ounces of water each time -/
theorem jimmy_water_consumption :
  water_per_time 8 5 (5/2) (1/128) = 8 := by
sorry

end NUMINAMATH_CALUDE_jimmy_water_consumption_l139_13931


namespace NUMINAMATH_CALUDE_day_care_toddlers_l139_13936

/-- Given the initial ratio of toddlers to infants and the ratio after more infants join,
    prove the number of toddlers -/
theorem day_care_toddlers (t i : ℕ) (h1 : t * 3 = i * 7) (h2 : t * 5 = (i + 12) * 7) : t = 42 := by
  sorry

end NUMINAMATH_CALUDE_day_care_toddlers_l139_13936


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_one_l139_13904

theorem sum_of_roots_equals_negative_one :
  ∀ x y : ℝ, (x - 4) * (x + 5) = 33 ∧ (y - 4) * (y + 5) = 33 → x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_one_l139_13904


namespace NUMINAMATH_CALUDE_prob_man_satisfied_correct_expected_satisfied_men_correct_l139_13988

/-- Represents the number of men in the seating arrangement -/
def num_men : ℕ := 50

/-- Represents the number of women in the seating arrangement -/
def num_women : ℕ := 50

/-- Represents the total number of people in the seating arrangement -/
def total_people : ℕ := num_men + num_women

/-- Represents the probability of a specific man being satisfied -/
def prob_man_satisfied : ℚ := 25 / 33

/-- Represents the expected number of satisfied men -/
def expected_satisfied_men : ℚ := 1250 / 33

/-- Theorem stating the probability of a specific man being satisfied -/
theorem prob_man_satisfied_correct : 
  prob_man_satisfied = 1 - (num_men - 1) / (total_people - 1) * (num_men - 2) / (total_people - 2) :=
sorry

/-- Theorem stating the expected number of satisfied men -/
theorem expected_satisfied_men_correct : 
  expected_satisfied_men = num_men * prob_man_satisfied :=
sorry

end NUMINAMATH_CALUDE_prob_man_satisfied_correct_expected_satisfied_men_correct_l139_13988


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l139_13934

-- Definition of a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part 1
theorem fixed_points_of_specific_quadratic :
  let f := quadratic 1 (-1) (-3)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ ∧ x₁ = -1 ∧ x₂ = 3 := by sorry

-- Part 2
theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  let f := quadratic 2 (-(3+a)) (a-1)
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ is_fixed_point f x₁ ∧ is_fixed_point f x₂ →
  (∀ y₁ y₂ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧ is_fixed_point f y₁ ∧ is_fixed_point f y₂ →
    y₁ / y₂ + y₂ / y₁ ≥ 8) := by sorry

-- Part 3
theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point (quadratic a (b+1) (b-1)) x) ↔
  0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l139_13934


namespace NUMINAMATH_CALUDE_equation_three_roots_l139_13992

theorem equation_three_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x : ℝ, x ∈ s ↔ Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)) :=
by sorry

end NUMINAMATH_CALUDE_equation_three_roots_l139_13992


namespace NUMINAMATH_CALUDE_pet_fee_calculation_l139_13999

-- Define the given constants
def daily_rate : ℚ := 125
def stay_duration_days : ℕ := 14
def service_fee_rate : ℚ := 0.2
def security_deposit_rate : ℚ := 0.5
def security_deposit : ℚ := 1110

-- Define the pet fee
def pet_fee : ℚ := 120

-- Theorem statement
theorem pet_fee_calculation :
  let base_cost := daily_rate * stay_duration_days
  let service_fee := service_fee_rate * base_cost
  let total_without_pet_fee := base_cost + service_fee
  let total_with_pet_fee := security_deposit / security_deposit_rate
  total_with_pet_fee - total_without_pet_fee = pet_fee := by
  sorry


end NUMINAMATH_CALUDE_pet_fee_calculation_l139_13999


namespace NUMINAMATH_CALUDE_pencil_sharpener_difference_l139_13903

/-- Proves that the difference in pencils sharpened between electric and hand-crank sharpeners is 10 in 6 minutes -/
theorem pencil_sharpener_difference : 
  let hand_crank_time : ℕ := 45  -- Time in seconds for hand-crank sharpener to sharpen one pencil
  let electric_time : ℕ := 20    -- Time in seconds for electric sharpener to sharpen one pencil
  let total_time : ℕ := 360      -- Total time in seconds (6 minutes)
  (total_time / electric_time) - (total_time / hand_crank_time) = 10 := by
sorry

end NUMINAMATH_CALUDE_pencil_sharpener_difference_l139_13903


namespace NUMINAMATH_CALUDE_age_puzzle_l139_13919

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 50) (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l139_13919


namespace NUMINAMATH_CALUDE_point_on_line_l139_13986

-- Define the points
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (4, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (5, m)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem point_on_line (m : ℝ) :
  collinear A B (C m) → m = 6 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l139_13986


namespace NUMINAMATH_CALUDE_number_greater_than_fifteen_l139_13940

theorem number_greater_than_fifteen (x : ℝ) : 0.4 * x > 0.8 * 5 + 2 → x > 15 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_fifteen_l139_13940


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l139_13980

/-- Represents a point on a 2D grid -/
structure Point where
  x : Int
  y : Int

/-- Represents a rhombus on a 2D grid -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rhombus given its vertices -/
def rhombusArea (r : Rhombus) : ℚ :=
  1 -- placeholder for the actual calculation

/-- Calculates the area of a square grid -/
def gridArea (side : ℕ) : ℕ :=
  side * side

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  let r : Rhombus := {
    v1 := { x := 3, y := 2 },
    v2 := { x := 4, y := 3 },
    v3 := { x := 3, y := 4 },
    v4 := { x := 2, y := 3 }
  }
  let gridSide : ℕ := 6
  rhombusArea r / gridArea gridSide = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l139_13980


namespace NUMINAMATH_CALUDE_topsoil_cost_l139_13915

/-- The cost of topsoil in euros per cubic meter -/
def cost_per_cubic_meter : ℝ := 12

/-- The volume of topsoil to be purchased in cubic meters -/
def volume : ℝ := 3

/-- The total cost of purchasing the topsoil -/
def total_cost : ℝ := cost_per_cubic_meter * volume

/-- Theorem stating that the total cost of purchasing 3 cubic meters of topsoil is 36 euros -/
theorem topsoil_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l139_13915


namespace NUMINAMATH_CALUDE_area_outside_parallel_chords_l139_13953

/-- Given a circle with radius 10 inches and two equal parallel chords 10 inches apart,
    the area of the region outside these chords but inside the circle is (200π/3 - 25√3) square inches. -/
theorem area_outside_parallel_chords (r : ℝ) (d : ℝ) : 
  r = 10 → d = 10 → 
  (2 * π * r^2 / 3 - 5 * r * Real.sqrt 3) = (200 * π / 3 - 25 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_area_outside_parallel_chords_l139_13953


namespace NUMINAMATH_CALUDE_solution_theorem_l139_13908

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 - i)^2 * z = 3 + 2*i

-- State the theorem
theorem solution_theorem :
  ∃ (z : ℂ), given_equation z ∧ z = -1 + (3/2) * i :=
sorry

end NUMINAMATH_CALUDE_solution_theorem_l139_13908


namespace NUMINAMATH_CALUDE_power_multiplication_equals_128_l139_13923

theorem power_multiplication_equals_128 : 
  ∀ b : ℕ, b = 2 → b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equals_128_l139_13923


namespace NUMINAMATH_CALUDE_polygon_with_20_diagonals_has_8_sides_l139_13943

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 diagonals has 8 sides -/
theorem polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_20_diagonals_has_8_sides_l139_13943
