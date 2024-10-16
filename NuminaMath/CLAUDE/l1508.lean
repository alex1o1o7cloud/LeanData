import Mathlib

namespace NUMINAMATH_CALUDE_number_of_divisors_5400_l1508_150855

theorem number_of_divisors_5400 : Nat.card (Nat.divisors 5400) = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_5400_l1508_150855


namespace NUMINAMATH_CALUDE_kaylaScoreEighthLevel_l1508_150810

/-- Fibonacci sequence starting with 1 and 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Kayla's score at a given level -/
def kaylaScore : ℕ → ℤ
  | 0 => 2
  | n + 1 => if n % 2 = 0 then kaylaScore n - fib n else kaylaScore n + fib n

theorem kaylaScoreEighthLevel : kaylaScore 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_kaylaScoreEighthLevel_l1508_150810


namespace NUMINAMATH_CALUDE_addition_problems_l1508_150873

theorem addition_problems :
  (189 + (-9) = 180) ∧
  ((-25) + 56 + (-39) = -8) ∧
  (41 + (-22) + (-33) + 19 = 5) ∧
  ((-0.5) + 13/4 + 2.75 + (-11/2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l1508_150873


namespace NUMINAMATH_CALUDE_club_size_l1508_150844

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := glove_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 3

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (glove_cost + helmet_cost) + cap_cost

/-- The total expenditure for all players' equipment in dollars -/
def total_expenditure : ℕ := 2968

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size : num_players = 72 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l1508_150844


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1508_150875

theorem algebraic_expression_equality (a b : ℝ) : 
  (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) = (2*a + (1/2)*b)^2 - 4*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1508_150875


namespace NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l1508_150890

/-- The derivative of x^2 * sin(x) is 2x * sin(x) + x^2 * cos(x) -/
theorem derivative_x_squared_sin_x (x : ℝ) :
  deriv (fun x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l1508_150890


namespace NUMINAMATH_CALUDE_partner_a_income_increase_l1508_150804

/-- Represents the increase in partner a's income when the profit rate changes --/
def income_increase (capital : ℝ) (initial_rate final_rate : ℝ) (share : ℝ) : ℝ :=
  share * (final_rate - initial_rate) * capital

/-- Theorem stating the increase in partner a's income given the problem conditions --/
theorem partner_a_income_increase :
  let capital : ℝ := 10000
  let initial_rate : ℝ := 0.05
  let final_rate : ℝ := 0.07
  let share : ℝ := 2/3
  income_increase capital initial_rate final_rate share = 400/3 := by sorry

end NUMINAMATH_CALUDE_partner_a_income_increase_l1508_150804


namespace NUMINAMATH_CALUDE_exists_special_number_l1508_150895

/-- A function that checks if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that swaps two digits in a natural number at given positions -/
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

theorem exists_special_number :
  ∃ (N : ℕ),
    N % 2020 = 0 ∧
    has_distinct_digits N ∧
    num_digits N = 6 ∧
    ∀ (i j : ℕ), i ≠ j → (swap_digits N i j) % 2020 ≠ 0 ∧
    ∀ (M : ℕ), M % 2020 = 0 → has_distinct_digits M →
      (∀ (i j : ℕ), i ≠ j → (swap_digits M i j) % 2020 ≠ 0) →
      num_digits M ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_exists_special_number_l1508_150895


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l1508_150846

-- Define the given circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0
def line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  (∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → result_circle x y) ∧
  (∃ h k : ℝ, line h k ∧ ∀ x y : ℝ, result_circle x y ↔ (x - h)^2 + (y - k)^2 = ((h^2 + k^2 - 3) / 2)) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l1508_150846


namespace NUMINAMATH_CALUDE_factory_production_l1508_150836

theorem factory_production (x : ℝ) 
  (h1 : (2200 / x) - (2400 / (1.2 * x)) = 1) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l1508_150836


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1508_150850

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1508_150850


namespace NUMINAMATH_CALUDE_train_platform_time_l1508_150882

/-- Given a train of length 1500 meters that crosses a tree in 100 seconds,
    calculate the time taken to pass a platform of length 500 meters. -/
theorem train_platform_time (train_length platform_length tree_crossing_time : ℝ)
    (h1 : train_length = 1500)
    (h2 : platform_length = 500)
    (h3 : tree_crossing_time = 100) :
    (train_length + platform_length) / (train_length / tree_crossing_time) = 400/3 := by
  sorry

#eval (1500 + 500) / (1500 / 100) -- Should output approximately 133.33333333

end NUMINAMATH_CALUDE_train_platform_time_l1508_150882


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l1508_150802

theorem blocks_used_for_tower (initial_blocks : ℕ) (blocks_left : ℕ) : 
  initial_blocks = 78 → blocks_left = 59 → initial_blocks - blocks_left = 19 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l1508_150802


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1508_150867

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (t s : ℝ), 
    t > 0 ∧ s > 0 ∧  -- positive side lengths
    3 * t - 4 * s = 4020 ∧  -- perimeter difference
    t = |s - 12| + d ∧  -- side length relationship
    4 * s > 0)  -- square perimeter > 0
  ↔ d > 1352 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1508_150867


namespace NUMINAMATH_CALUDE_cao_required_proof_l1508_150851

/-- Represents the balanced chemical equation for the reaction between Calcium oxide and Water to form Calcium hydroxide -/
structure BalancedEquation where
  cao : ℕ
  h2o : ℕ
  caoh2 : ℕ
  balanced : cao = h2o ∧ cao = caoh2

/-- Calculates the required amount of Calcium oxide given the amounts of Water and Calcium hydroxide -/
def calcCaORequired (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) : ℕ :=
  if water = hydroxide then water else 0

theorem cao_required_proof (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) 
  (h1 : water = 3) 
  (h2 : hydroxide = 3) : 
  calcCaORequired water hydroxide eq = 3 := by
  sorry

end NUMINAMATH_CALUDE_cao_required_proof_l1508_150851


namespace NUMINAMATH_CALUDE_ratio_tr_ur_l1508_150827

-- Define the square PQRS
def Square (P Q R S : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  let (sx, sy) := S
  (qx - px)^2 + (qy - py)^2 = 4 ∧
  (rx - qx)^2 + (ry - qy)^2 = 4 ∧
  (sx - rx)^2 + (sy - ry)^2 = 4 ∧
  (px - sx)^2 + (py - sy)^2 = 4

-- Define the quarter circle QS
def QuarterCircle (Q S : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  let (sx, sy) := S
  (sx - qx)^2 + (sy - qy)^2 = 4

-- Define U as the midpoint of QR
def Midpoint (U Q R : ℝ × ℝ) : Prop :=
  let (ux, uy) := U
  let (qx, qy) := Q
  let (rx, ry) := R
  ux = (qx + rx) / 2 ∧ uy = (qy + ry) / 2

-- Define T lying on SR
def PointOnLine (T S R : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (sx, sy) := S
  let (rx, ry) := R
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ tx = sx + t * (rx - sx) ∧ ty = sy + t * (ry - sy)

-- Define TU as tangent to the arc QS
def Tangent (T U Q S : ℝ × ℝ) : Prop :=
  let (tx, ty) := T
  let (ux, uy) := U
  let (qx, qy) := Q
  let (sx, sy) := S
  (tx - ux) * (qy - sy) = (ty - uy) * (qx - sx)

-- Theorem statement
theorem ratio_tr_ur (P Q R S T U : ℝ × ℝ) 
  (h1 : Square P Q R S)
  (h2 : QuarterCircle Q S)
  (h3 : Midpoint U Q R)
  (h4 : PointOnLine T S R)
  (h5 : Tangent T U Q S) :
  let (tx, ty) := T
  let (rx, ry) := R
  let (ux, uy) := U
  (tx - rx)^2 + (ty - ry)^2 = 16/9 * ((ux - rx)^2 + (uy - ry)^2) := by sorry

end NUMINAMATH_CALUDE_ratio_tr_ur_l1508_150827


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1508_150821

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- If a complex number z satisfies (1-i)z = 2i, then z is in the second quadrant -/
theorem z_in_second_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  in_second_quadrant z := by
  sorry


end NUMINAMATH_CALUDE_z_in_second_quadrant_l1508_150821


namespace NUMINAMATH_CALUDE_largest_integer_not_exceeding_700pi_l1508_150803

theorem largest_integer_not_exceeding_700pi :
  ⌊700 * Real.pi⌋ = 2199 := by sorry

end NUMINAMATH_CALUDE_largest_integer_not_exceeding_700pi_l1508_150803


namespace NUMINAMATH_CALUDE_video_game_earnings_l1508_150879

def total_games : ℕ := 10
def non_working_games : ℕ := 2
def price_per_game : ℕ := 4

theorem video_game_earnings :
  (total_games - non_working_games) * price_per_game = 32 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l1508_150879


namespace NUMINAMATH_CALUDE_equation_solution_l1508_150811

theorem equation_solution :
  ∃ y : ℝ, (6 * y / (y + 2) - 4 / (y + 2) = 2 / (y + 2)) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1508_150811


namespace NUMINAMATH_CALUDE_inequality_solution_quadratic_inequality_l1508_150864

-- Part 1
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic (-4) ∪ Set.Ici (1/2))

theorem inequality_solution :
  ∀ x : ℝ, (9 / (x + 4) ≤ 2) ↔ solution_set x := by sorry

-- Part 2
def valid_k (k : ℝ) : Prop :=
  k ∈ (Set.Iio (-Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2))

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → valid_k k := by sorry

end NUMINAMATH_CALUDE_inequality_solution_quadratic_inequality_l1508_150864


namespace NUMINAMATH_CALUDE_trapezoid_area_l1508_150880

/-- Given an outer equilateral triangle with area 64 and an inner equilateral triangle
    with area 4, where the space between them is divided into three congruent trapezoids,
    prove that the area of one trapezoid is 20. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 64) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1508_150880


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1508_150826

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1508_150826


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_coordinate_transformation_l1508_150854

/-- Given a point with rectangular coordinates (4, -3, -2) and corresponding
    spherical coordinates (ρ, θ, φ), prove that the point with spherical
    coordinates (ρ, θ + π, -φ) has rectangular coordinates (-4, 3, -2). -/
theorem spherical_to_rectangular_coordinate_transformation
  (ρ θ φ : ℝ) (h1 : 4 = ρ * Real.sin φ * Real.cos θ)
  (h2 : -3 = ρ * Real.sin φ * Real.sin θ)
  (h3 : -2 = ρ * Real.cos φ) :
  (-4 = ρ * Real.sin (-φ) * Real.cos (θ + π)) ∧
  (3 = ρ * Real.sin (-φ) * Real.sin (θ + π)) ∧
  (-2 = ρ * Real.cos (-φ)) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_coordinate_transformation_l1508_150854


namespace NUMINAMATH_CALUDE_larger_integer_is_nine_l1508_150824

theorem larger_integer_is_nine (x y : ℤ) (h_product : x * y = 36) (h_sum : x + y = 13) :
  max x y = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_is_nine_l1508_150824


namespace NUMINAMATH_CALUDE_intersection_equals_T_l1508_150820

-- Define set S
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}

-- Define set T
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Theorem statement
theorem intersection_equals_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_T_l1508_150820


namespace NUMINAMATH_CALUDE_objective_function_minimum_range_l1508_150893

-- Define the objective function
def objective_function (k x y : ℝ) : ℝ := k * x + 2 * y

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := 2 * x - y ≤ 1
def constraint2 (x y : ℝ) : Prop := x + y ≥ 2
def constraint3 (x y : ℝ) : Prop := y - x ≤ 2

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x y

-- Define the minimum point
def is_minimum_point (k : ℝ) (x y : ℝ) : Prop :=
  feasible_region x y ∧
  ∀ x' y', feasible_region x' y' →
    objective_function k x y ≤ objective_function k x' y'

-- Theorem statement
theorem objective_function_minimum_range :
  ∀ k : ℝ, (is_minimum_point k 1 1 ∧
    ∀ x y, x ≠ 1 ∨ y ≠ 1 → ¬(is_minimum_point k x y)) →
  -4 < k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_objective_function_minimum_range_l1508_150893


namespace NUMINAMATH_CALUDE_exactly_one_integer_solution_l1508_150888

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property for (3n+i)^6 to be an integer
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (3 * n + i : ℂ)^6 = m

-- Theorem statement
theorem exactly_one_integer_solution :
  ∃! n : ℤ, is_integer_power n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_integer_solution_l1508_150888


namespace NUMINAMATH_CALUDE_canned_food_bins_l1508_150840

theorem canned_food_bins (soup : ℝ) (vegetables : ℝ) (pasta : ℝ)
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end NUMINAMATH_CALUDE_canned_food_bins_l1508_150840


namespace NUMINAMATH_CALUDE_percentage_problem_l1508_150814

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (50 / 100) * 1080 → P = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1508_150814


namespace NUMINAMATH_CALUDE_min_sum_for_product_4410_l1508_150883

theorem min_sum_for_product_4410 (a b c d : ℕ+) 
  (h : a * b * c * d = 4410) : 
  (∀ w x y z : ℕ+, w * x * y * z = 4410 → a + b + c + d ≤ w + x + y + z) ∧ 
  (∃ w x y z : ℕ+, w * x * y * z = 4410 ∧ w + x + y + z = 69) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_for_product_4410_l1508_150883


namespace NUMINAMATH_CALUDE_cars_in_north_america_l1508_150808

def total_cars : ℕ := 6755
def cars_in_europe : ℕ := 2871

theorem cars_in_north_america : total_cars - cars_in_europe = 3884 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_north_america_l1508_150808


namespace NUMINAMATH_CALUDE_least_value_quadratic_inequality_l1508_150885

theorem least_value_quadratic_inequality :
  (∀ b : ℝ, b < 4 → -b^2 + 9*b - 20 < 0) ∧
  (-4^2 + 9*4 - 20 = 0) ∧
  (∀ b : ℝ, b > 4 → -b^2 + 9*b - 20 ≤ -b^2 + 9*4 - 20) :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_inequality_l1508_150885


namespace NUMINAMATH_CALUDE_book_sale_earnings_l1508_150852

/-- Calculates the total earnings from a book sale --/
theorem book_sale_earnings (total_books : ℕ) (price_high : ℚ) (price_low : ℚ) : 
  total_books = 10 ∧ 
  price_high = 5/2 ∧ 
  price_low = 2 → 
  (2/5 * total_books : ℚ) * price_high + (3/5 * total_books : ℚ) * price_low = 22 := by
  sorry

#check book_sale_earnings

end NUMINAMATH_CALUDE_book_sale_earnings_l1508_150852


namespace NUMINAMATH_CALUDE_nicky_running_time_l1508_150842

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 400)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  ∃ (t : ℝ), t = 30 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l1508_150842


namespace NUMINAMATH_CALUDE_expand_expression_l1508_150881

theorem expand_expression (y : ℝ) : 12 * (3 * y - 4) = 36 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1508_150881


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1508_150878

theorem quadratic_factorization : 
  ∃ (c d : ℤ), (∀ y : ℝ, 4 * y^2 + 4 * y - 32 = (4 * y + c) * (y + d)) ∧ c - d = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1508_150878


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1508_150889

theorem sqrt_sum_equality (x : ℝ) :
  Real.sqrt (x^2 - 2*x + 4) + Real.sqrt (x^2 + 2*x + 4) =
  Real.sqrt ((x-1)^2 + 3) + Real.sqrt ((x+1)^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1508_150889


namespace NUMINAMATH_CALUDE_solution_equivalence_l1508_150858

theorem solution_equivalence (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (x^2 + 3*x + 1/(x-1) = a + 1/(x-1) ↔ x^2 + 3*x = a)) ∧
  (a = 4 → ∃ x : ℝ, x^2 + 3*x = a ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l1508_150858


namespace NUMINAMATH_CALUDE_total_good_balls_eq_144_l1508_150898

/-- The total number of soccer balls -/
def total_soccer_balls : ℕ := 180

/-- The total number of basketballs -/
def total_basketballs : ℕ := 75

/-- The total number of tennis balls -/
def total_tennis_balls : ℕ := 90

/-- The total number of volleyballs -/
def total_volleyballs : ℕ := 50

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 125

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 49

/-- The number of tennis balls with holes -/
def tennis_balls_with_holes : ℕ := 62

/-- The number of deflated volleyballs -/
def deflated_volleyballs : ℕ := 15

/-- The total number of balls without holes or deflation -/
def total_good_balls : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) +
  (total_basketballs - basketballs_with_holes) +
  (total_tennis_balls - tennis_balls_with_holes) +
  (total_volleyballs - deflated_volleyballs)

theorem total_good_balls_eq_144 : total_good_balls = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_good_balls_eq_144_l1508_150898


namespace NUMINAMATH_CALUDE_flatbread_diameters_exist_l1508_150849

/-- The diameter of the skillet -/
def skillet_diameter : ℕ := 26

/-- Predicate to check if three positive integers satisfy the required conditions -/
def valid_diameters (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y + z = skillet_diameter ∧
  x^2 + y^2 + z^2 = 338 ∧
  (x^2 + y^2 + z^2 : ℚ) / 4 = (skillet_diameter^2 : ℚ) / 8

/-- Theorem stating the existence of three positive integers satisfying the conditions -/
theorem flatbread_diameters_exist : ∃ x y z : ℕ, valid_diameters x y z := by
  sorry

end NUMINAMATH_CALUDE_flatbread_diameters_exist_l1508_150849


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l1508_150801

theorem cubic_equation_ratio (p q r s : ℝ) : 
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) →
  r / s = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l1508_150801


namespace NUMINAMATH_CALUDE_sum_two_longest_altitudes_l1508_150899

/-- A right triangle with sides 6, 8, and 10 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The sum of the lengths of the two longest altitudes in the given right triangle is 14 -/
theorem sum_two_longest_altitudes (t : RightTriangle) : 
  max t.a t.b + min t.a t.b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_longest_altitudes_l1508_150899


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1508_150832

theorem max_product_sum_300 : 
  ∀ a b : ℤ, a + b = 300 → a * b ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1508_150832


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1508_150831

theorem ratio_of_percentages (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.3 * N)
  (hP : P ≠ 0) : 
  M / R = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1508_150831


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1508_150872

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x^6 + x^5 + 3*x^4 + 2*x^2 + 15
def q (x : ℝ) : ℝ := x^6 + x^5 + 4*x^4 - x^3 + x^2 + 18
def r (x : ℝ) : ℝ := x^6 - x^4 + x^3 + x^2 - 3

-- State the theorem
theorem polynomial_simplification (x : ℝ) : p x - q x = r x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1508_150872


namespace NUMINAMATH_CALUDE_composition_f_equals_inverse_e_l1508_150847

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem composition_f_equals_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_composition_f_equals_inverse_e_l1508_150847


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_permutations_l1508_150859

/-- Represents the number of legs a centipede has -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the constraint that for each leg, the sock must be put on before the shoe -/
def sock_before_shoe_constraint (leg : ℕ) : Prop :=
  leg ≤ num_legs ∧ ∃ (sock_pos shoe_pos : ℕ), sock_pos < shoe_pos

/-- The main theorem stating the number of valid permutations -/
theorem centipede_sock_shoe_permutations :
  (Nat.factorial total_items) / (2^num_legs) =
  (Nat.factorial 20) / (2^10) :=
sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_permutations_l1508_150859


namespace NUMINAMATH_CALUDE_green_home_construction_l1508_150835

theorem green_home_construction (x : ℝ) (h : x > 50) : (300 : ℝ) / (x - 50) = 400 / x := by
  sorry

end NUMINAMATH_CALUDE_green_home_construction_l1508_150835


namespace NUMINAMATH_CALUDE_spend_representation_l1508_150806

-- Define a type for monetary transactions
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

-- Theorem statement
theorem spend_representation (amount : ℤ) :
  represent (Transaction.receive amount) = amount →
  represent (Transaction.spend amount) = -amount :=
by
  sorry

end NUMINAMATH_CALUDE_spend_representation_l1508_150806


namespace NUMINAMATH_CALUDE_no_integer_b_with_two_distinct_roots_l1508_150887

theorem no_integer_b_with_two_distinct_roots :
  ¬ ∃ (b : ℤ), ∃ (x y : ℤ), x ≠ y ∧
    x^4 + 4*x^3 + b*x^2 + 16*x + 8 = 0 ∧
    y^4 + 4*y^3 + b*y^2 + 16*y + 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_b_with_two_distinct_roots_l1508_150887


namespace NUMINAMATH_CALUDE_faster_watch_gain_rate_l1508_150874

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : ℕ := sorry

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : Time) : ℕ := sorry

theorem faster_watch_gain_rate (alarmSetTime correctAlarmTime fasterAlarmTime : Time) 
  (h1 : alarmSetTime = ⟨22, 0⟩)  -- Alarm set at 10:00 PM
  (h2 : correctAlarmTime = ⟨4, 0⟩)  -- Correct watch shows 4:00 AM
  (h3 : fasterAlarmTime = ⟨4, 12⟩)  -- Faster watch shows 4:12 AM
  : (timeDifference correctAlarmTime fasterAlarmTime) / 
    (hoursBetween alarmSetTime correctAlarmTime) = 2 := by sorry

end NUMINAMATH_CALUDE_faster_watch_gain_rate_l1508_150874


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1508_150818

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (2 * a^2 - 7 * a + 3 = 0) → 
             (2 * b^2 - 7 * b + 3 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1508_150818


namespace NUMINAMATH_CALUDE_intersection_point_y_axis_l1508_150897

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem intersection_point_y_axis :
  ∃ (y : ℝ), f 0 = y ∧ (0, y) = (0, -2) := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_axis_l1508_150897


namespace NUMINAMATH_CALUDE_unique_solution_2014_l1508_150807

theorem unique_solution_2014 (x : ℝ) (h : x > 0) :
  (x * 2014^(1/x) + (1/x) * 2014^x) / 2 = 2014 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2014_l1508_150807


namespace NUMINAMATH_CALUDE_exponential_decreasing_inequality_l1508_150876

theorem exponential_decreasing_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_inequality_l1508_150876


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1508_150870

/-- The function f(x) = a^(-x-2) + 4 always passes through the point (-2, 5) for a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(-x-2) + 4
  f (-2) = 5 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1508_150870


namespace NUMINAMATH_CALUDE_magic_square_sum_l1508_150860

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_sum : a + 27 + b = sum
  row2_sum : 15 + c + d = sum
  row3_sum : 30 + e + 18 = sum
  col1_sum : 30 + 15 + a = sum
  col2_sum : e + c + 27 = sum
  col3_sum : 18 + d + b = sum
  diag1_sum : 30 + c + b = sum
  diag2_sum : 18 + c + a = sum

/-- Theorem: In a 3x3 magic square with the given known numbers, 
    if the sums of all rows, columns, and diagonals are equal, then d + e = 108 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 108 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1508_150860


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l1508_150896

theorem min_value_of_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y : ℝ, y = 5 * x^2 - 10 * x + 14 → y ≥ min_y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l1508_150896


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1508_150816

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1508_150816


namespace NUMINAMATH_CALUDE_digit_for_divisibility_by_three_l1508_150812

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem digit_for_divisibility_by_three :
  ∃! B : ℕ, is_single_digit B ∧ (952 * 10 + B) % 3 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_for_divisibility_by_three_l1508_150812


namespace NUMINAMATH_CALUDE_evergreen_elementary_grade6_l1508_150819

theorem evergreen_elementary_grade6 (total : ℕ) (grade4 : ℕ) (grade5 : ℕ) 
  (h1 : total = 100)
  (h2 : grade4 = 30)
  (h3 : grade5 = 35) :
  total - grade4 - grade5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_evergreen_elementary_grade6_l1508_150819


namespace NUMINAMATH_CALUDE_converse_zero_product_l1508_150839

theorem converse_zero_product (a b : ℝ) : 
  (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) := by sorry

end NUMINAMATH_CALUDE_converse_zero_product_l1508_150839


namespace NUMINAMATH_CALUDE_line_l_equation_l1508_150894

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if the point (x, y) lies on the given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The reference line y = x + 1 -/
def referenceLine : Line :=
  { slope := 1, yIntercept := 1 }

/-- The line we're trying to prove -/
def lineL : Line :=
  { slope := 2, yIntercept := -3 }

theorem line_l_equation :
  (lineL.slope = 2 * referenceLine.slope) ∧
  (lineL.containsPoint 3 3) →
  ∀ x y : ℝ, lineL.containsPoint x y ↔ y = 2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_l_equation_l1508_150894


namespace NUMINAMATH_CALUDE_angle_point_cosine_l1508_150838

/-- Given an angle α in the first quadrant and a point P(a, √5) on its terminal side,
    if cos α = (√2/4)a, then a = √3 -/
theorem angle_point_cosine (α : Real) (a : Real) :
  0 < α ∧ α < π / 2 →  -- α is in the first quadrant
  (∃ (P : ℝ × ℝ), P = (a, Real.sqrt 5) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = Real.cos α) →  -- P(a, √5) is on the terminal side
  Real.cos α = (Real.sqrt 2 / 4) * a →  -- given condition
  a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_point_cosine_l1508_150838


namespace NUMINAMATH_CALUDE_married_men_fraction_l1508_150886

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women := (3 : ℚ) / 7 * total_women
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  married_men / total_people = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1508_150886


namespace NUMINAMATH_CALUDE_sock_ratio_is_7_19_l1508_150828

/-- Represents the ratio of black socks to blue socks -/
structure SockRatio where
  black : ℕ
  blue : ℕ

/-- Represents the order of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ
  price_ratio : ℚ
  bill_increase : ℚ

/-- Calculates the ratio of black socks to blue socks given a sock order -/
def calculate_sock_ratio (order : SockOrder) : SockRatio :=
  sorry

/-- The specific sock order from the problem -/
def tom_order : SockOrder :=
  { black := 5
  , blue := 0  -- Unknown, to be calculated
  , price_ratio := 3
  , bill_increase := 3/5 }

theorem sock_ratio_is_7_19 : 
  let ratio := calculate_sock_ratio tom_order
  ratio.black = 7 ∧ ratio.blue = 19 := by sorry

end NUMINAMATH_CALUDE_sock_ratio_is_7_19_l1508_150828


namespace NUMINAMATH_CALUDE_inequality_condition_l1508_150845

theorem inequality_condition (x : ℝ) : x * (x + 2) > x * (3 - x) + 1 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l1508_150845


namespace NUMINAMATH_CALUDE_johns_recycling_money_l1508_150884

/-- The weight of a Monday-Saturday newspaper in ounces -/
def weekdayPaperWeight : ℕ := 8

/-- The weight of a Sunday newspaper in ounces -/
def sundayPaperWeight : ℕ := 2 * weekdayPaperWeight

/-- The number of papers John is supposed to deliver daily -/
def dailyPapers : ℕ := 250

/-- The number of weeks John steals the papers -/
def stolenWeeks : ℕ := 10

/-- The recycling value of one ton of paper in dollars -/
def recyclingValuePerTon : ℕ := 20

/-- The number of ounces in a ton -/
def ouncesPerTon : ℕ := 32000

/-- Calculate the total money John makes from recycling stolen newspapers -/
def johnsMoney : ℚ :=
  let totalWeekdayWeight := 6 * stolenWeeks * dailyPapers * weekdayPaperWeight
  let totalSundayWeight := stolenWeeks * dailyPapers * sundayPaperWeight
  let totalWeight := totalWeekdayWeight + totalSundayWeight
  let weightInTons := totalWeight / ouncesPerTon
  weightInTons * recyclingValuePerTon

/-- Theorem stating that John makes $100 from recycling the stolen newspapers -/
theorem johns_recycling_money : johnsMoney = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_recycling_money_l1508_150884


namespace NUMINAMATH_CALUDE_girls_minus_boys_l1508_150843

/-- The number of boys in Grade 7 Class 1 -/
def num_boys (a b : ℤ) : ℤ := 2*a - b

/-- The number of girls in Grade 7 Class 1 -/
def num_girls (a b : ℤ) : ℤ := 3*a + b

/-- The theorem stating the difference between the number of girls and boys -/
theorem girls_minus_boys (a b : ℤ) : 
  num_girls a b - num_boys a b = a + 2*b := by
  sorry

end NUMINAMATH_CALUDE_girls_minus_boys_l1508_150843


namespace NUMINAMATH_CALUDE_g_five_equals_one_l1508_150861

def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (∀ x : ℝ, g x = g (-x))

theorem g_five_equals_one (g : ℝ → ℝ) (h : g_property g) : g 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l1508_150861


namespace NUMINAMATH_CALUDE_p_hyperbola_range_p_necessary_not_sufficient_for_q_l1508_150830

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 4) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (4 - m) = 1

-- Define what it means for p to represent a hyperbola
def p_is_hyperbola (m : ℝ) : Prop := (m - 1) * (m - 4) < 0

-- Define what it means for q to represent an ellipse
def q_is_ellipse (m : ℝ) : Prop := m - 2 > 0 ∧ 4 - m > 0 ∧ m - 2 ≠ 4 - m

-- Theorem 1: The range of m for which p represents a hyperbola
theorem p_hyperbola_range : 
  ∀ m : ℝ, p_is_hyperbola m ↔ (1 < m ∧ m < 4) :=
sorry

-- Theorem 2: p being true is necessary but not sufficient for q being true
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q_is_ellipse m → p_is_hyperbola m) ∧
  (∃ m : ℝ, p_is_hyperbola m ∧ ¬q_is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_p_hyperbola_range_p_necessary_not_sufficient_for_q_l1508_150830


namespace NUMINAMATH_CALUDE_solve_system_l1508_150813

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7) 
  (eq2 : x + 3 * y = 16) : 
  x = 16 := by sorry

end NUMINAMATH_CALUDE_solve_system_l1508_150813


namespace NUMINAMATH_CALUDE_race_time_B_b_finish_time_l1508_150868

/-- Calculates the time taken by runner B to finish a race given the conditions --/
theorem race_time_B (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) : ℝ :=
  let distance_B_in_time_A := race_distance - beat_distance
  let speed_B := distance_B_in_time_A / time_A
  race_distance / speed_B

/-- Proves that B finishes the race in 25 seconds given the specified conditions --/
theorem b_finish_time (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) :
  race_time_B race_distance time_A beat_distance = 25 :=
by
  -- Assuming race_distance = 110, time_A = 20, and beat_distance = 22
  have h1 : race_distance = 110 := by sorry
  have h2 : time_A = 20 := by sorry
  have h3 : beat_distance = 22 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_race_time_B_b_finish_time_l1508_150868


namespace NUMINAMATH_CALUDE_midpoint_locus_l1508_150866

/-- Given real numbers a, b, c forming an arithmetic sequence,
    prove that the locus of the midpoint of the chord of the line
    bx + ay + c = 0 intersecting the parabola y^2 = -1/2 x
    is described by the equation x + 1 = -(2y - 1)^2 -/
theorem midpoint_locus (a b c : ℝ) :
  (2 * b = a + c) →
  ∃ (x y : ℝ), 
    (∃ (x₁ y₁ : ℝ), 
      b * x₁ + a * y₁ + c = 0 ∧ 
      y₁^2 = -1/2 * x₁ ∧
      x = (x₁ - 2) / 2 ∧
      y = (y₁ + 1) / 2) →
    x + 1 = -(2 * y - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l1508_150866


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l1508_150857

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_geq_two_l1508_150857


namespace NUMINAMATH_CALUDE_bacon_vs_mashed_potatoes_l1508_150825

theorem bacon_vs_mashed_potatoes (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 479) 
  (h2 : bacon = 489) : 
  bacon - mashed_potatoes = 10 := by
  sorry

end NUMINAMATH_CALUDE_bacon_vs_mashed_potatoes_l1508_150825


namespace NUMINAMATH_CALUDE_yuan_yuan_delivery_cost_l1508_150865

def express_delivery_cost (weight : ℕ) : ℕ :=
  let base_fee := 13
  let weight_limit := 5
  let additional_fee := 2
  if weight ≤ weight_limit then
    base_fee
  else
    base_fee + (weight - weight_limit) * additional_fee

theorem yuan_yuan_delivery_cost :
  express_delivery_cost 7 = 17 := by sorry

end NUMINAMATH_CALUDE_yuan_yuan_delivery_cost_l1508_150865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1508_150809

/-- An arithmetic sequence with 2036 terms -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ :=
  fun n => a + (n - 1) * d

theorem arithmetic_sequence_sum (a d : ℝ) :
  let t := ArithmeticSequence a d
  t 2018 = 100 →
  t 2000 + 5 * t 2015 + 5 * t 2021 + t 2036 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1508_150809


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l1508_150892

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x (Nat.gcd y z)) * (y - x) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l1508_150892


namespace NUMINAMATH_CALUDE_pirate_loot_value_l1508_150829

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 2, 1, 4]
def silkGarments : List Nat := [1, 2, 0, 2]
def rareSpices : List Nat := [1, 3, 2]

theorem pirate_loot_value :
  base5ToBase10 silverware + base5ToBase10 silkGarments + base5ToBase10 rareSpices = 865 := by
  sorry


end NUMINAMATH_CALUDE_pirate_loot_value_l1508_150829


namespace NUMINAMATH_CALUDE_factorization_equality_l1508_150822

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1508_150822


namespace NUMINAMATH_CALUDE_average_visitors_is_750_l1508_150815

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def average_visitors (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let total_sundays : ℕ := 5
  let total_other_days : ℕ := 30 - total_sundays
  let total_visitors : ℕ := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 750 -/
theorem average_visitors_is_750 :
  average_visitors 1000 700 = 750 := by sorry

end NUMINAMATH_CALUDE_average_visitors_is_750_l1508_150815


namespace NUMINAMATH_CALUDE_estimate_at_25_l1508_150848

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the estimated y value for a given x on a regression line -/
def estimate_y (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- The specific regression line y = 0.5x - 0.81 -/
def specific_line : RegressionLine :=
  { slope := 0.5, intercept := -0.81 }

/-- Theorem: The estimated y value when x = 25 on the specific regression line is 11.69 -/
theorem estimate_at_25 :
  estimate_y specific_line 25 = 11.69 := by sorry

end NUMINAMATH_CALUDE_estimate_at_25_l1508_150848


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l1508_150817

/-- 
Given a scooter purchase, we define the following:
purchase_price: The initial cost of the scooter
selling_price: The price at which the scooter was sold
gain_percent: The percentage gain on the sale
repair_cost: The amount spent on repairs

We prove that the repair cost satisfies the equation relating these variables.
-/
theorem scooter_repair_cost 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (gain_percent : ℝ) 
  (repair_cost : ℝ) 
  (h1 : purchase_price = 4400)
  (h2 : selling_price = 5800)
  (h3 : gain_percent = 0.1154) :
  selling_price = (purchase_price + repair_cost) * (1 + gain_percent) :=
by sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l1508_150817


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1508_150805

/-- A hyperbola is represented by the equation (x^2 / (k-2)) + (y^2 / (5-k)) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / (k-2)) + (y^2 / (5-k)) = 1 ∧ (k-2) * (5-k) < 0

/-- The range of k for which the equation represents a hyperbola -/
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k < 2 ∨ k > 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1508_150805


namespace NUMINAMATH_CALUDE_potato_count_l1508_150841

/-- Given the initial number of potatoes and the number of new potatoes left after rabbits ate some,
    prove that the total number of potatoes is equal to the sum of the initial number and the number of new potatoes left. -/
theorem potato_count (initial : ℕ) (new_left : ℕ) : 
  initial + new_left = initial + new_left :=
by sorry

end NUMINAMATH_CALUDE_potato_count_l1508_150841


namespace NUMINAMATH_CALUDE_like_terms_exponents_l1508_150834

/-- Given two algebraic expressions that are like terms, prove the values of their exponents. -/
theorem like_terms_exponents (a b : ℤ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 2 * x^a * y^2 = k * (-3 * x^3 * y^(b+3))) → 
  (a = 3 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l1508_150834


namespace NUMINAMATH_CALUDE_complex_multiplication_l1508_150891

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number 3+2i -/
def z : ℂ := 3 + 2 * i

theorem complex_multiplication :
  z * i = -2 + 3 * i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1508_150891


namespace NUMINAMATH_CALUDE_total_winter_clothing_l1508_150853

/-- Represents the contents of a box of winter clothing -/
structure BoxContents where
  scarves : ℕ
  mittens : ℕ
  hats : ℕ

/-- Calculates the total number of items in a box -/
def totalItemsInBox (box : BoxContents) : ℕ :=
  box.scarves + box.mittens + box.hats

/-- The contents of the four boxes -/
def box1 : BoxContents := { scarves := 3, mittens := 5, hats := 2 }
def box2 : BoxContents := { scarves := 4, mittens := 3, hats := 1 }
def box3 : BoxContents := { scarves := 2, mittens := 6, hats := 3 }
def box4 : BoxContents := { scarves := 1, mittens := 7, hats := 2 }

/-- Theorem stating that the total number of winter clothing items is 39 -/
theorem total_winter_clothing : 
  totalItemsInBox box1 + totalItemsInBox box2 + totalItemsInBox box3 + totalItemsInBox box4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l1508_150853


namespace NUMINAMATH_CALUDE_festival_profit_margin_is_five_percent_l1508_150877

/-- Represents the pricing and profit information for an item -/
structure ItemPricing where
  regular_discount : ℝ
  regular_profit_margin : ℝ

/-- Calculates the profit margin during a "buy one get one free" offer -/
def festival_profit_margin (item : ItemPricing) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the profit margin during the shopping festival -/
theorem festival_profit_margin_is_five_percent (item : ItemPricing) 
  (h1 : item.regular_discount = 0.3)
  (h2 : item.regular_profit_margin = 0.47) :
  festival_profit_margin item = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_festival_profit_margin_is_five_percent_l1508_150877


namespace NUMINAMATH_CALUDE_exchange_impossibility_l1508_150833

theorem exchange_impossibility : 
  ¬∃ (x y z : ℕ), x + y + z = 10 ∧ x + 3*y + 5*z = 25 :=
sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l1508_150833


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l1508_150800

/-- The quadratic function f(x) = x² - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l1508_150800


namespace NUMINAMATH_CALUDE_fifth_subject_mark_l1508_150862

/-- Given a student's marks in four subjects and the average across five subjects,
    calculate the mark in the fifth subject. -/
theorem fifth_subject_mark (e m p c : ℕ) (avg : ℚ) (h1 : e = 90) (h2 : m = 92) (h3 : p = 85) (h4 : c = 87) (h5 : avg = 87.8) :
  ∃ (b : ℕ), (e + m + p + c + b : ℚ) / 5 = avg ∧ b = 85 := by
  sorry

#check fifth_subject_mark

end NUMINAMATH_CALUDE_fifth_subject_mark_l1508_150862


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l1508_150837

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral :=
  (radius : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

/-- The theorem about the inscribed quadrilateral -/
theorem inscribed_quadrilateral_theorem (q : InscribedQuadrilateral) :
  q.radius = 300 ∧ q.side1 = 300 ∧ q.side2 = 300 ∧ q.side3 = 200 →
  q.side4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l1508_150837


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l1508_150856

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_c_value :
  ∀ c : ℝ, (∀ x y : ℝ, y = 5 * x + 7 ↔ y = (3 * c) * x + 1) → c = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l1508_150856


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l1508_150863

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem largest_n_not_exceeding_500 : ∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l1508_150863


namespace NUMINAMATH_CALUDE_arcsin_arccos_half_pi_l1508_150871

theorem arcsin_arccos_half_pi : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_half_pi_l1508_150871


namespace NUMINAMATH_CALUDE_solution_t_l1508_150869

theorem solution_t : ∃ t : ℝ, (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 1) ∧ t = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_t_l1508_150869


namespace NUMINAMATH_CALUDE_composite_and_prime_divisors_l1508_150823

/-- Given two distinct positive integers a and b where a, b > 1, and s_n = a^n + b^(n+1) -/
theorem composite_and_prime_divisors (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  let s : ℕ → ℕ := fun n => a^n + b^(n+1)
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ¬ Nat.Prime (s n)) ∧
  (∃ (P : Set ℕ), Set.Infinite P ∧ ∀ p ∈ P, Nat.Prime p ∧ ∃ n, p ∣ s n) := by
  sorry

end NUMINAMATH_CALUDE_composite_and_prime_divisors_l1508_150823
