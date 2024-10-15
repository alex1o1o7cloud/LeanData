import Mathlib

namespace NUMINAMATH_CALUDE_sum_three_digit_integers_eq_385550_l3920_392080

/-- The sum of all three-digit positive integers from 200 to 900 -/
def sum_three_digit_integers : ℕ :=
  let first_term := 200
  let last_term := 900
  let common_difference := 1
  let num_terms := (last_term - first_term) / common_difference + 1
  (num_terms * (first_term + last_term)) / 2

theorem sum_three_digit_integers_eq_385550 : 
  sum_three_digit_integers = 385550 := by
  sorry

#eval sum_three_digit_integers

end NUMINAMATH_CALUDE_sum_three_digit_integers_eq_385550_l3920_392080


namespace NUMINAMATH_CALUDE_odd_function_inequality_l3920_392078

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem odd_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≠ 0) →
  is_decreasing_on f (-2) 0 →
  f (1 - m) + f (1 - m^2) < 0 →
  -1 ≤ m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l3920_392078


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l3920_392023

theorem complex_expression_equals_nine :
  (Real.sqrt 2 - 3) ^ (0 : ℝ) - Real.sqrt 9 + |(-2 : ℝ)| + (-1/3 : ℝ) ^ (-2 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l3920_392023


namespace NUMINAMATH_CALUDE_normal_dist_symmetry_normal_dist_property_l3920_392037

/-- A normal distribution with mean 0 and standard deviation σ -/
def normal_dist (σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
def P (σ : ℝ) : Set ℝ → ℝ := sorry

theorem normal_dist_symmetry 
  (σ : ℝ) (a : ℝ) : 
  P σ {x | -a ≤ x ∧ x ≤ 0} = P σ {x | 0 ≤ x ∧ x ≤ a} :=
sorry

theorem normal_dist_property 
  (σ : ℝ) (h : P σ {x | -2 ≤ x ∧ x ≤ 0} = 0.3) : 
  P σ {x | x > 2} = 0.2 :=
sorry

end NUMINAMATH_CALUDE_normal_dist_symmetry_normal_dist_property_l3920_392037


namespace NUMINAMATH_CALUDE_problem_solution_l3920_392033

theorem problem_solution (x : ℝ) (hx_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3920_392033


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l3920_392066

/-- Proves that a squirrel traveling at 6 miles per hour takes 30 minutes to travel 3 miles -/
theorem squirrel_travel_time :
  let speed : ℝ := 6  -- Speed in miles per hour
  let distance : ℝ := 3  -- Distance in miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l3920_392066


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3920_392055

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3920_392055


namespace NUMINAMATH_CALUDE_find_a_solve_inequality_l3920_392074

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

-- Theorem 1: Prove the value of a
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) → a = 2 :=
by sorry

-- Theorem 2: Prove the solution set of the inequality
theorem solve_inequality :
  ∀ x : ℝ, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_solve_inequality_l3920_392074


namespace NUMINAMATH_CALUDE_binomial_threshold_l3920_392097

theorem binomial_threshold (n : ℕ) : 
  (n ≥ 82 → Nat.choose (2*n) n < 4^(n-2)) ∧ 
  (n ≥ 1305 → Nat.choose (2*n) n < 4^(n-3)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_threshold_l3920_392097


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3920_392085

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (-12, x - 4)
  parallel a b → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3920_392085


namespace NUMINAMATH_CALUDE_math_city_intersections_l3920_392088

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets * (c.num_streets - 1)) / 2

/-- Theorem: A city with 10 streets, no parallel streets, and no triple intersections has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel = true → c.no_triple_intersections = true →
  num_intersections c = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_city_intersections_l3920_392088


namespace NUMINAMATH_CALUDE_transformations_result_l3920_392041

/-- Rotates a point (x, y) by 180° counterclockwise around (2, 3) -/
def rotate180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- Applies both transformations to a point (x, y) -/
def applyTransformations (x y : ℝ) : ℝ × ℝ :=
  let (x', y') := rotate180 x y
  reflectAboutYEqX x' y'

theorem transformations_result (c d : ℝ) :
  applyTransformations c d = (1, -4) → d - c = 7 := by
  sorry

end NUMINAMATH_CALUDE_transformations_result_l3920_392041


namespace NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l3920_392082

theorem x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8 :
  (∀ x : ℝ, x^3 < -8 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x^3 ≥ -8) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l3920_392082


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3920_392047

theorem min_distance_between_curves (a b c d : ℝ) :
  (a - 2*Real.exp a)/b = (1 - c)/(d - 1) ∧ (a - 2*Real.exp a)/b = 1 →
  (∀ x y z w : ℝ, (x - 2*Real.exp x)/y = (1 - z)/(w - 1) ∧ (x - 2*Real.exp x)/y = 1 →
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3920_392047


namespace NUMINAMATH_CALUDE_sin_sixty_minus_third_power_zero_l3920_392007

theorem sin_sixty_minus_third_power_zero :
  2 * Real.sin (60 * π / 180) - (1/3)^0 = Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_sin_sixty_minus_third_power_zero_l3920_392007


namespace NUMINAMATH_CALUDE_game_draw_probability_l3920_392027

theorem game_draw_probability (amy_win lily_win eve_win draw : ℚ) : 
  amy_win = 2/5 → lily_win = 1/5 → eve_win = 1/10 → 
  amy_win + lily_win + eve_win + draw = 1 →
  draw = 3/10 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l3920_392027


namespace NUMINAMATH_CALUDE_cos_48_degrees_l3920_392099

theorem cos_48_degrees : Real.cos (48 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l3920_392099


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3920_392039

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) / (1 + Complex.I) = (a : ℂ) + (b : ℂ) * Complex.I → a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3920_392039


namespace NUMINAMATH_CALUDE_log_five_negative_one_l3920_392071

theorem log_five_negative_one (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 5 = -1) : x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_log_five_negative_one_l3920_392071


namespace NUMINAMATH_CALUDE_probability_less_than_four_l3920_392008

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition -/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices at (0,0), (0,3), (3,0), and (3,3) -/
def givenSquare : Square :=
  { bottomLeft := (0, 0),
    topRight := (3, 3) }

/-- The condition x + y < 4 -/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_less_than_four :
  probability givenSquare condition = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_four_l3920_392008


namespace NUMINAMATH_CALUDE_horseback_trip_distance_l3920_392063

/-- Calculates the total distance traveled during a horseback riding trip -/
def total_distance : ℝ :=
  let day1_segment1 := 5 * 7
  let day1_segment2 := 3 * 2
  let day2_segment1 := 6 * 6
  let day2_segment2 := 3 * 3
  let day3_segment1 := 4 * 3
  let day3_segment2 := 7 * 5
  day1_segment1 + day1_segment2 + day2_segment1 + day2_segment2 + day3_segment1 + day3_segment2

theorem horseback_trip_distance : total_distance = 133 := by
  sorry

end NUMINAMATH_CALUDE_horseback_trip_distance_l3920_392063


namespace NUMINAMATH_CALUDE_west_east_correspondence_l3920_392012

-- Define a type for directions
inductive Direction
| East
| West

-- Define a function to represent distance with direction
def distance_with_direction (d : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.East => d
  | Direction.West => -d

-- State the theorem
theorem west_east_correspondence :
  (distance_with_direction 2023 Direction.West = -2023) →
  (distance_with_direction 2023 Direction.East = 2023) :=
by
  sorry

end NUMINAMATH_CALUDE_west_east_correspondence_l3920_392012


namespace NUMINAMATH_CALUDE_stack_height_problem_l3920_392019

/-- Calculates the total height of a stack of discs with a cylindrical item on top -/
def total_height (top_diameter : ℕ) (bottom_diameter : ℕ) (disc_thickness : ℕ) (cylinder_height : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  let discs_height := num_discs * disc_thickness
  discs_height + cylinder_height

/-- The problem statement -/
theorem stack_height_problem :
  let top_diameter := 15
  let bottom_diameter := 1
  let disc_thickness := 2
  let cylinder_height := 10
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = 26 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_problem_l3920_392019


namespace NUMINAMATH_CALUDE_production_days_l3920_392067

theorem production_days (n : ℕ) 
  (h1 : (50 * n) / n = 50)  -- Average production for past n days
  (h2 : ((50 * n + 60) : ℝ) / (n + 1) = 55)  -- New average including today
  : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l3920_392067


namespace NUMINAMATH_CALUDE_power_division_sum_equality_l3920_392076

theorem power_division_sum_equality : (-6)^5 / 6^2 + 4^3 - 7^2 = -201 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_equality_l3920_392076


namespace NUMINAMATH_CALUDE_ellipse_intersection_relation_l3920_392024

/-- Theorem: Relationship between y-coordinates of intersection points on an ellipse --/
theorem ellipse_intersection_relation (a b m : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  a > b ∧ b > 0 ∧ m > a →  -- Conditions on a, b, and m
  (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧  -- A is on the ellipse
  (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧  -- B is on the ellipse
  (∃ k : ℝ, x₁ = k * y₁ + m ∧ x₂ = k * y₂ + m) →  -- A and B are on a line through M
  x₃ = a^2 / m ∧ x₄ = a^2 / m →  -- P and Q are on the line x = a^2/m
  (y₃ * (x₁ + a) = y₁ * (x₃ + a)) ∧  -- P is on line A₁A
  (y₄ * (x₂ + a) = y₂ * (x₄ + a)) →  -- Q is on line A₁B
  1 / y₁ + 1 / y₂ = 1 / y₃ + 1 / y₄ :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_relation_l3920_392024


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l3920_392068

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar form ρ sin(θ - α) = k -/
structure PolarLine where
  alpha : ℝ
  k : ℝ

/-- Calculates the distance from a point in polar coordinates to a line in polar form -/
noncomputable def distanceFromPointToLine (p : PolarPoint) (l : PolarLine) : ℝ :=
  sorry

/-- Theorem stating that the distance from P(2, -π/6) to the line ρ sin(θ - π/6) = 1 is √3 + 1 -/
theorem distance_from_point_to_line :
  let p : PolarPoint := ⟨2, -π/6⟩
  let l : PolarLine := ⟨π/6, 1⟩
  distanceFromPointToLine p l = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l3920_392068


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3920_392032

theorem cubic_equation_solutions :
  ∀ x y : ℤ, x^3 = y^3 + 2*y^2 + 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3920_392032


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l3920_392045

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 8 = m^2 ∧ 
  m = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l3920_392045


namespace NUMINAMATH_CALUDE_pattern_equality_l3920_392058

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => n + i + 1)

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The theorem stating the equality of the observed pattern -/
theorem pattern_equality (n : ℕ) : leftSide n = 2^n * oddProduct n := by
  sorry

#check pattern_equality

end NUMINAMATH_CALUDE_pattern_equality_l3920_392058


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3920_392049

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 8 n = Nat.choose 8 2) → (n = 2 ∨ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3920_392049


namespace NUMINAMATH_CALUDE_equation_solution_l3920_392086

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3920_392086


namespace NUMINAMATH_CALUDE_twelve_lines_formed_l3920_392073

/-- A configuration of points in a plane -/
structure PointConfiguration where
  total_points : ℕ
  collinear_points : ℕ
  noncollinear_points : ℕ
  h_total : total_points = collinear_points + noncollinear_points
  h_collinear : collinear_points ≥ 3
  h_noncollinear : noncollinear_points ≥ 0

/-- The number of lines formed by a given point configuration -/
def num_lines (config : PointConfiguration) : ℕ :=
  1 + config.collinear_points * config.noncollinear_points + 
  (config.noncollinear_points * (config.noncollinear_points - 1)) / 2

/-- Theorem: In the given configuration, 12 lines can be formed -/
theorem twelve_lines_formed (config : PointConfiguration) 
  (h1 : config.total_points = 7)
  (h2 : config.collinear_points = 5)
  (h3 : config.noncollinear_points = 2) :
  num_lines config = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_lines_formed_l3920_392073


namespace NUMINAMATH_CALUDE_highway_vehicles_l3920_392046

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicles_l3920_392046


namespace NUMINAMATH_CALUDE_square_binomial_k_l3920_392044

theorem square_binomial_k (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) → k = 100 := by
sorry

end NUMINAMATH_CALUDE_square_binomial_k_l3920_392044


namespace NUMINAMATH_CALUDE_solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l3920_392011

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (a+2)*x + 2*a < 0

-- Theorem for a = 1
theorem solution_when_a_is_one :
  ∀ x : ℝ, inequality 1 x ↔ 1 < x ∧ x < 2 :=
sorry

-- Theorem for a > 2
theorem solution_when_a_greater_than_two :
  ∀ a x : ℝ, a > 2 → (inequality a x ↔ 2 < x ∧ x < a) :=
sorry

-- Theorem for a = 2
theorem solution_when_a_equals_two :
  ∀ x : ℝ, ¬(inequality 2 x) :=
sorry

-- Theorem for a < 2
theorem solution_when_a_less_than_two :
  ∀ a x : ℝ, a < 2 → (inequality a x ↔ a < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_one_solution_when_a_greater_than_two_solution_when_a_equals_two_solution_when_a_less_than_two_l3920_392011


namespace NUMINAMATH_CALUDE_river_current_speed_l3920_392072

/-- Proves that given a ship with a maximum speed of 20 km/h in still water,
    if it takes the same time to travel 100 km downstream as it does to travel 60 km upstream,
    then the speed of the river current is 5 km/h. -/
theorem river_current_speed :
  let ship_speed : ℝ := 20
  let downstream_distance : ℝ := 100
  let upstream_distance : ℝ := 60
  ∀ current_speed : ℝ,
    (downstream_distance / (ship_speed + current_speed) = upstream_distance / (ship_speed - current_speed)) →
    current_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l3920_392072


namespace NUMINAMATH_CALUDE_x_over_y_equals_one_l3920_392091

-- Define a function that represents the nested absolute value expression
def nestedAbs (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => |nestedAbs y x n - x|

-- State the theorem
theorem x_over_y_equals_one
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h : nestedAbs x y 2019 = nestedAbs y x 2019) :
  x / y = 1 :=
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_one_l3920_392091


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3920_392038

/-- Given that the solution set of ax² + bx + 1 > 0 is {x | -1 < x < 1/3}, prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3920_392038


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_300_l3920_392035

def multiples_of_10 (n : ℕ) : List ℕ :=
  List.filter (fun x => x % 10 = 0) (List.range (n + 1))

theorem average_of_multiples_10_to_300 :
  let sequence := multiples_of_10 300
  (sequence.sum / sequence.length : ℚ) = 155 := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_300_l3920_392035


namespace NUMINAMATH_CALUDE_computer_screen_height_l3920_392090

theorem computer_screen_height (side : ℝ) (height : ℝ) : 
  side = 20 →
  height = 4 * side + 20 →
  height = 100 := by
sorry

end NUMINAMATH_CALUDE_computer_screen_height_l3920_392090


namespace NUMINAMATH_CALUDE_thirteen_binary_l3920_392069

/-- Converts a natural number to its binary representation as a list of booleans -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (l : List Bool) : Prop :=
  to_binary n = l.reverse

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end NUMINAMATH_CALUDE_thirteen_binary_l3920_392069


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l3920_392070

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 1 10, g x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 10, g x = max) ∧
    (∀ x ∈ Set.Icc 1 10, min ≤ g x) ∧
    (∃ x ∈ Set.Icc 1 10, g x = min) ∧
    max + min = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l3920_392070


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_l3920_392083

/-- The sum of 20 consecutive positive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum :
  (∃ n : ℕ, sum_20_consecutive n = 250) ∧
  (∀ m : ℕ, m < 250 → ¬∃ n : ℕ, sum_20_consecutive n = m ∧ is_perfect_square m) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_l3920_392083


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_proof_l3920_392009

/-- The number of bits required to represent ABCDEF₁₆ in binary -/
def abcdef_hex_to_bits : ℕ := 24

/-- The decimal value of ABCDEF₁₆ -/
def abcdef_hex_to_decimal : ℕ := 11293375

theorem abcdef_hex_bits_proof :
  abcdef_hex_to_bits = 24 ∧
  2^23 < abcdef_hex_to_decimal ∧
  abcdef_hex_to_decimal < 2^24 := by
  sorry

#eval abcdef_hex_to_bits
#eval abcdef_hex_to_decimal

end NUMINAMATH_CALUDE_abcdef_hex_bits_proof_l3920_392009


namespace NUMINAMATH_CALUDE_soccer_field_kids_l3920_392059

/-- The number of kids on a soccer field after more kids join -/
def total_kids (initial : ℕ) (joined : ℕ) : ℕ :=
  initial + joined

/-- Theorem: The total number of kids on the soccer field is 36 -/
theorem soccer_field_kids : total_kids 14 22 = 36 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l3920_392059


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3920_392030

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 ∧ (1/m + 1/n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3920_392030


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l3920_392087

/-- A tangential quadrilateral with circumscribed and inscribed circles -/
structure TangentialQuadrilateral where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance between the centers of the circumscribed and inscribed circles -/
  d : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- d is non-negative and less than R -/
  d_bounds : 0 ≤ d ∧ d < R

/-- The main theorem about the relationship between R, r, and d in a tangential quadrilateral -/
theorem tangential_quadrilateral_theorem (q : TangentialQuadrilateral) :
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_theorem_l3920_392087


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3920_392081

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 7*x + 2) * q + (33*x^2 + 10*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3920_392081


namespace NUMINAMATH_CALUDE_car_travel_time_l3920_392057

/-- Given two cars A and B with specific speeds and distance ratios, 
    prove that Car A takes 6 hours to reach its destination. -/
theorem car_travel_time :
  ∀ (speed_A speed_B time_B distance_A distance_B : ℝ),
  speed_A = 50 →
  speed_B = 100 →
  time_B = 1 →
  distance_A / distance_B = 3 →
  distance_B = speed_B * time_B →
  distance_A = speed_A * (distance_A / speed_A) →
  distance_A / speed_A = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l3920_392057


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3920_392020

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 2| + |2*x + 4| < 10) ↔ (x > -4 ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3920_392020


namespace NUMINAMATH_CALUDE_range_of_a_min_value_expression_equality_condition_l3920_392034

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| + |x - 20|

-- Define the property that the solution set is not empty
def solution_set_nonempty (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < 10 * a + 10

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : solution_set_nonempty a → a > 0 := by sorry

-- Theorem for the minimum value of a + 4/a^2
theorem min_value_expression (a : ℝ) (h : a > 0) :
  a + 4 / a^2 ≥ 3 := by sorry

-- Theorem for the equality condition
theorem equality_condition (a : ℝ) (h : a > 0) :
  a + 4 / a^2 = 3 ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_expression_equality_condition_l3920_392034


namespace NUMINAMATH_CALUDE_johns_allowance_l3920_392001

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (A > 0) →                                           -- Allowance is positive
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 96 / 100 = A) →  -- Total spending equals allowance
  (A = 36 / 10) :=                                    -- Allowance is $3.60
by sorry

end NUMINAMATH_CALUDE_johns_allowance_l3920_392001


namespace NUMINAMATH_CALUDE_sum_equals_three_or_seven_l3920_392053

theorem sum_equals_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  x + y + z = 3 ∨ x + y + z = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_or_seven_l3920_392053


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l3920_392003

/-- Represents the gas consumption problem for Dream's car -/
theorem dream_car_gas_consumption 
  (gas_per_mile : ℝ) 
  (miles_today : ℝ) 
  (miles_tomorrow : ℝ) 
  (total_gas : ℝ) :
  miles_today = 400 →
  miles_tomorrow = miles_today + 200 →
  total_gas = 4000 →
  gas_per_mile * miles_today + gas_per_mile * miles_tomorrow = total_gas →
  gas_per_mile = 4 := by
sorry

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l3920_392003


namespace NUMINAMATH_CALUDE_inverse_proportion_point_value_l3920_392043

/-- Prove that for an inverse proportion function y = k/x (k ≠ 0),
    if points A(2,m) and B(m,n) lie on its graph, then n = 2. -/
theorem inverse_proportion_point_value (k m n : ℝ) : 
  k ≠ 0 → m = k / 2 → n = k / m → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_value_l3920_392043


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l3920_392004

/-- The minimum distance between a point in the feasible region and a line -/
theorem min_distance_point_to_line :
  ∀ (x y : ℝ),
  (2 * x + y - 4 ≥ 0) →
  (x - y - 2 ≤ 0) →
  (y - 3 ≤ 0) →
  ∃ (x' y' : ℝ),
  (y' = -2 * x' + 2) →
  ∀ (x'' y'' : ℝ),
  (y'' = -2 * x'' + 2) →
  Real.sqrt ((x - x')^2 + (y - y')^2) ≥ (2 * Real.sqrt 5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l3920_392004


namespace NUMINAMATH_CALUDE_unique_student_count_l3920_392015

theorem unique_student_count : ∃! n : ℕ, n < 600 ∧ n % 25 = 24 ∧ n % 19 = 18 ∧ n = 424 := by
  sorry

end NUMINAMATH_CALUDE_unique_student_count_l3920_392015


namespace NUMINAMATH_CALUDE_arrangement_problem_l3920_392021

def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem arrangement_problem (n_boys n_girls : ℕ) (h_boys : n_boys = 6) (h_girls : n_girls = 4) :
  -- (I) Girls standing together
  (A n_girls n_girls * A (n_boys + 1) (n_boys + 1) = A 4 4 * A 7 7) ∧
  -- (II) No two girls adjacent
  (A n_boys n_boys * A (n_boys + 1) n_girls = A 6 6 * A 7 4) ∧
  -- (III) Boys A, B, C in alphabetical order
  (A (n_boys + n_girls) (n_boys + n_girls - 3) = A 10 7) :=
sorry

end NUMINAMATH_CALUDE_arrangement_problem_l3920_392021


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l3920_392042

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  constant + 1.50 * (x - 2)

/-- The number of axles on an 18-wheel truck with 2 wheels on the front axle and 4 wheels on each other axle -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for the 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 6

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l3920_392042


namespace NUMINAMATH_CALUDE_unique_solution_for_k_squared_minus_2016_equals_3_to_n_l3920_392089

theorem unique_solution_for_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_squared_minus_2016_equals_3_to_n_l3920_392089


namespace NUMINAMATH_CALUDE_floor_product_equals_17_l3920_392000

def solution_set : Set ℝ := Set.Ici 4.25 ∩ Set.Iio 4.5

theorem floor_product_equals_17 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 17 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_17_l3920_392000


namespace NUMINAMATH_CALUDE_min_sum_complementary_events_l3920_392094

theorem min_sum_complementary_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hcomp : 4/x + 1/y = 1) : 
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4/x + 1/y = 1 ∧ x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_complementary_events_l3920_392094


namespace NUMINAMATH_CALUDE_change_eight_dollars_theorem_l3920_392077

theorem change_eight_dollars_theorem :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (combinations : List (ℕ × ℕ × ℕ)),
    combinations.length = n ∧
    ∀ (c : ℕ × ℕ × ℕ), c ∈ combinations →
      let (nickels, dimes, quarters) := c
      nickels > 0 ∧ dimes > 0 ∧ quarters > 0 ∧
      5 * nickels + 10 * dimes + 25 * quarters = 800) :=
by sorry

end NUMINAMATH_CALUDE_change_eight_dollars_theorem_l3920_392077


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l3920_392064

theorem average_age_of_five_students
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 20)
  (h2 : avg_age_all = 20)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 16)
  (h5 : age_last_student = 186)
  : ∃ (avg_age_group2 : ℝ),
    avg_age_group2 = 14 ∧
    avg_age_group2 * (total_students - num_group1 - 1) =
      total_students * avg_age_all - num_group1 * avg_age_group1 - age_last_student :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l3920_392064


namespace NUMINAMATH_CALUDE_puppy_sleeps_16_hours_l3920_392026

def connor_sleep_time : ℕ := 6

def luke_sleep_time (connor_sleep_time : ℕ) : ℕ := connor_sleep_time + 2

def puppy_sleep_time (luke_sleep_time : ℕ) : ℕ := 2 * luke_sleep_time

theorem puppy_sleeps_16_hours :
  puppy_sleep_time (luke_sleep_time connor_sleep_time) = 16 := by
  sorry

end NUMINAMATH_CALUDE_puppy_sleeps_16_hours_l3920_392026


namespace NUMINAMATH_CALUDE_perfect_square_base9_l3920_392052

/-- Represents a number in base 9 of the form ac7b -/
structure Base9Number where
  a : ℕ
  c : ℕ
  b : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.c + 63 + n.b

/-- Theorem stating that if a Base9Number is a perfect square, then b must be 0 -/
theorem perfect_square_base9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base9_l3920_392052


namespace NUMINAMATH_CALUDE_gcd_conditions_and_sum_of_digits_l3920_392036

/-- The least positive integer greater than 1000 satisfying the given GCD conditions -/
def n : ℕ := sorry

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem gcd_conditions_and_sum_of_digits :
  n > 1000 ∧
  Nat.gcd 75 (n + 150) = 25 ∧
  Nat.gcd (n + 75) 150 = 75 ∧
  (∀ k, k > 1000 → Nat.gcd 75 (k + 150) = 25 → Nat.gcd (k + 75) 150 = 75 → k ≥ n) ∧
  sum_of_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_conditions_and_sum_of_digits_l3920_392036


namespace NUMINAMATH_CALUDE_log_inequality_l3920_392093

theorem log_inequality (a b c : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l3920_392093


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3920_392025

/-- Given a right-angled triangle PQR with R at the right angle, 
    PR = 4000, and PQ = 5000, prove that PQ + QR + PR = 16500 -/
theorem triangle_perimeter (PR PQ QR : ℝ) : 
  PR = 4000 → 
  PQ = 5000 → 
  QR^2 = PQ^2 - PR^2 → 
  PQ + QR + PR = 16500 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3920_392025


namespace NUMINAMATH_CALUDE_alyssas_spending_l3920_392029

/-- Calculates the total spending given an amount paid and a refund. -/
def totalSpending (amountPaid refund : ℚ) : ℚ :=
  amountPaid - refund

/-- Proves that Alyssa's total spending is $2.23 given the conditions. -/
theorem alyssas_spending :
  let grapesPayment : ℚ := 12.08
  let cherriesRefund : ℚ := 9.85
  totalSpending grapesPayment cherriesRefund = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_spending_l3920_392029


namespace NUMINAMATH_CALUDE_yogurt_price_is_2_5_l3920_392062

/-- The price of a pack of yogurt in yuan -/
def yogurt_price : ℝ := 2.5

/-- The price of a pack of fresh milk in yuan -/
def milk_price : ℝ := 1

/-- The total cost of 4 packs of yogurt and 4 packs of fresh milk is 14 yuan -/
axiom first_purchase : 4 * yogurt_price + 4 * milk_price = 14

/-- The total cost of 2 packs of yogurt and 8 packs of fresh milk is 13 yuan -/
axiom second_purchase : 2 * yogurt_price + 8 * milk_price = 13

/-- The price of each pack of yogurt is 2.5 yuan -/
theorem yogurt_price_is_2_5 : yogurt_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_price_is_2_5_l3920_392062


namespace NUMINAMATH_CALUDE_sum_inequality_l3920_392013

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 + 2*b^2 + 3) + 1 / (b^2 + 2*c^2 + 3) + 1 / (c^2 + 2*a^2 + 3) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3920_392013


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3920_392028

theorem cafeteria_apples (apples_to_students : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_to_students = 42 →
  num_pies = 9 →
  apples_per_pie = 6 →
  apples_to_students + num_pies * apples_per_pie = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3920_392028


namespace NUMINAMATH_CALUDE_field_fencing_l3920_392018

/-- Proves that a rectangular field with area 80 sq. feet and one side 20 feet requires 28 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length * width = 80 → 
  length = 20 → 
  length + 2 * width = 28 := by sorry

end NUMINAMATH_CALUDE_field_fencing_l3920_392018


namespace NUMINAMATH_CALUDE_point_movement_l3920_392016

/-- Given a point P at (-1, 2), moving it 2 units left and 1 unit up results in point M at (-3, 3) -/
theorem point_movement :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (P.1 - 2, P.2 + 1)
  M = (-3, 3) := by sorry

end NUMINAMATH_CALUDE_point_movement_l3920_392016


namespace NUMINAMATH_CALUDE_pig_farmer_scenario_profit_l3920_392022

/-- Represents the profit calculation for a pig farmer --/
def pig_farmer_profit (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feed_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feed_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- Theorem stating the profit for the given scenario --/
theorem pig_farmer_scenario_profit :
  pig_farmer_profit 6 300 10 12 16 = 960 :=
sorry

end NUMINAMATH_CALUDE_pig_farmer_scenario_profit_l3920_392022


namespace NUMINAMATH_CALUDE_exists_valid_division_l3920_392002

/-- A tiling of a 6x6 board with 2x1 dominos -/
def Tiling := Fin 6 → Fin 6 → Fin 18

/-- A division of the board into two rectangles -/
structure Division where
  horizontal : Bool
  position : Fin 6

/-- Checks if a domino crosses the dividing line -/
def crossesDivision (t : Tiling) (d : Division) : Prop :=
  ∃ (i j : Fin 6), 
    (d.horizontal ∧ i = d.position ∧ t i j = t (i + 1) j) ∨
    (¬d.horizontal ∧ j = d.position ∧ t i j = t i (j + 1))

/-- The main theorem -/
theorem exists_valid_division (t : Tiling) : 
  ∃ (d : Division), ¬crossesDivision t d := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_division_l3920_392002


namespace NUMINAMATH_CALUDE_sum_of_base8_digits_of_888_l3920_392048

/-- Given a natural number n and a base b, returns the list of digits of n in base b -/
def toDigits (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- The sum of a list of natural numbers -/
def sum (l : List ℕ) : ℕ := sorry

theorem sum_of_base8_digits_of_888 :
  sum (toDigits 888 8) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_base8_digits_of_888_l3920_392048


namespace NUMINAMATH_CALUDE_mary_card_count_l3920_392006

/-- The number of Pokemon cards Mary has after receiving gifts from Sam and Alex -/
def final_card_count (initial_cards torn_cards sam_gift alex_gift : ℕ) : ℕ :=
  initial_cards - torn_cards + sam_gift + alex_gift

/-- Theorem stating that Mary has 196 Pokemon cards after the events described -/
theorem mary_card_count : 
  final_card_count 123 18 56 35 = 196 := by
  sorry

end NUMINAMATH_CALUDE_mary_card_count_l3920_392006


namespace NUMINAMATH_CALUDE_bob_always_wins_l3920_392014

/-- The game described in the problem -/
def Game (n : ℕ) : Prop :=
  ∀ (A : Fin (n + 1) → Finset (Fin (2^n))),
    (∀ i, (A i).card = 2^(n-1)) →
    ∃ (a : Fin (n + 1) → Fin (2^n)),
      ∀ t : Fin (2^n),
        ∃ i s, s ∈ A i ∧ (s + a i : Fin (2^n)) = t

/-- Bob always has a winning strategy for any positive n -/
theorem bob_always_wins :
  ∀ n : ℕ, n > 0 → Game n :=
sorry

end NUMINAMATH_CALUDE_bob_always_wins_l3920_392014


namespace NUMINAMATH_CALUDE_recipe_reduction_recipe_reduction_mixed_numbers_l3920_392079

-- Define the original recipe quantities
def flour_original : Rat := 31/4  -- 7 3/4 cups
def sugar_original : Rat := 5/2   -- 2 1/2 cups

-- Define the reduced recipe quantities
def flour_reduced : Rat := 31/12  -- 2 7/12 cups
def sugar_reduced : Rat := 5/6    -- 5/6 cups

-- Theorem to prove the correct reduced quantities
theorem recipe_reduction :
  flour_reduced = (1/3) * flour_original ∧
  sugar_reduced = (1/3) * sugar_original :=
by sorry

-- Helper function to convert rational to mixed number string representation
noncomputable def rat_to_mixed_string (r : Rat) : String :=
  let whole := Int.floor r
  let frac := r - whole
  if frac = 0 then
    s!"{whole}"
  else
    let num := (frac.num : Int)
    let den := (frac.den : Int)
    if whole = 0 then
      s!"{num}/{den}"
    else
      s!"{whole} {num}/{den}"

-- Theorem to prove the correct string representations
theorem recipe_reduction_mixed_numbers :
  rat_to_mixed_string flour_reduced = "2 7/12" ∧
  rat_to_mixed_string sugar_reduced = "5/6" :=
by sorry

end NUMINAMATH_CALUDE_recipe_reduction_recipe_reduction_mixed_numbers_l3920_392079


namespace NUMINAMATH_CALUDE_joe_caught_23_times_l3920_392065

/-- The number of times Joe caught the ball -/
def joe_catches : ℕ := 23

/-- The number of times Derek caught the ball -/
def derek_catches (j : ℕ) : ℕ := 2 * j - 4

/-- The number of times Tammy caught the ball -/
def tammy_catches (d : ℕ) : ℕ := d / 3 + 16

theorem joe_caught_23_times :
  joe_catches = 23 ∧
  derek_catches joe_catches = 2 * joe_catches - 4 ∧
  tammy_catches (derek_catches joe_catches) = 30 :=
sorry

end NUMINAMATH_CALUDE_joe_caught_23_times_l3920_392065


namespace NUMINAMATH_CALUDE_range_of_a_l3920_392050

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → x < -1 → y < -1 → f a y < f a x) →
  (∀ x y, x < y → 1 < x → 1 < y → f a x < f a y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3920_392050


namespace NUMINAMATH_CALUDE_sum_of_digits_in_period_of_one_over_98_squared_l3920_392096

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def sum_of_digits_in_period (n : ℕ) : ℕ :=
  sorry

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 196

/-- Theorem: The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) is 882 -/
theorem sum_of_digits_in_period_of_one_over_98_squared :
  sum_of_digits_in_period period_length = 882 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_period_of_one_over_98_squared_l3920_392096


namespace NUMINAMATH_CALUDE_total_baking_time_l3920_392054

def bread_time_1 : ℕ := 375
def bread_time_2 : ℕ := 160
def bread_time_3 : ℕ := 320

theorem total_baking_time :
  max (max bread_time_1 bread_time_2) bread_time_3 = 375 := by
  sorry

end NUMINAMATH_CALUDE_total_baking_time_l3920_392054


namespace NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_range_l3920_392005

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  9 / (x + 4) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici (1/2) :=
sorry

-- Part 2
theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → 
  k > Real.sqrt 2 ∨ k < -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_quadratic_inequality_range_l3920_392005


namespace NUMINAMATH_CALUDE_initial_coins_l3920_392040

/-- Given a box of coins, prove that the initial number of coins is 21 when 8 coins are added and the total becomes 29. -/
theorem initial_coins (initial_coins added_coins total_coins : ℕ) 
  (h1 : added_coins = 8)
  (h2 : total_coins = 29)
  (h3 : initial_coins + added_coins = total_coins) : 
  initial_coins = 21 := by
sorry

end NUMINAMATH_CALUDE_initial_coins_l3920_392040


namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l3920_392056

-- Define the quadratic functions
def f (x : ℝ) := x^2 - x - 6
def g (x : ℝ) := -2*x^2 + x + 1

-- Theorem for the first inequality
theorem solution_set_f : 
  {x : ℝ | f x > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Theorem for the second inequality
theorem solution_set_g :
  {x : ℝ | g x < 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l3920_392056


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3920_392075

theorem students_in_both_band_and_chorus 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (chorus_students : ℕ) 
  (either_band_or_chorus : ℕ) 
  (h1 : total_students = 300) 
  (h2 : band_students = 150) 
  (h3 : chorus_students = 180) 
  (h4 : either_band_or_chorus = 250) : 
  band_students + chorus_students - either_band_or_chorus = 80 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3920_392075


namespace NUMINAMATH_CALUDE_existence_of_divisor_l3920_392092

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 23 * f (n + 1) + f n

theorem existence_of_divisor (m : ℕ) : ∃ d : ℕ, ∀ n : ℕ, m ∣ f (f n) ↔ d ∣ n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisor_l3920_392092


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l3920_392051

theorem circumscribed_sphere_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 2 * Real.sqrt 6) :
  let diagonal_squared := a^2 + b^2 + c^2
  let sphere_radius := Real.sqrt (diagonal_squared / 4)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 49 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_area_l3920_392051


namespace NUMINAMATH_CALUDE_square_sum_value_l3920_392098

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) :
  x^2 + y^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3920_392098


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3920_392095

theorem complex_equation_solution (z : ℂ) (h : z * (2 - Complex.I) = 5 * Complex.I) :
  z = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3920_392095


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_always_power_of_two_l3920_392017

theorem no_quadratic_trinomial_always_power_of_two : 
  ¬ ∃ (a b c : ℤ), ∀ (x : ℕ), ∃ (n : ℕ), a * x^2 + b * x + c = 2^n := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_always_power_of_two_l3920_392017


namespace NUMINAMATH_CALUDE_contrapositive_example_l3920_392060

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 + x > 2) ↔ (∀ x : ℝ, x^2 + x ≤ 2 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3920_392060


namespace NUMINAMATH_CALUDE_subset_coloring_existence_l3920_392084

theorem subset_coloring_existence (S : Type) [Fintype S] (h : Fintype.card S = 2002) (N : ℕ) (hN : N ≤ 2^2002) :
  ∃ f : Set S → Bool,
    (∀ A B : Set S, f A = true → f B = true → f (A ∪ B) = true) ∧
    (∀ A B : Set S, f A = false → f B = false → f (A ∪ B) = false) ∧
    (Fintype.card {A : Set S | f A = true} = N) :=
by sorry

end NUMINAMATH_CALUDE_subset_coloring_existence_l3920_392084


namespace NUMINAMATH_CALUDE_f_10_equals_144_l3920_392010

def f : ℕ → ℕ
  | 0 => 0  -- define f(0) as 0 for completeness
  | 1 => 2
  | 2 => 3
  | (n + 3) => f (n + 2) + f (n + 1)

theorem f_10_equals_144 : f 10 = 144 := by sorry

end NUMINAMATH_CALUDE_f_10_equals_144_l3920_392010


namespace NUMINAMATH_CALUDE_complex_number_fourth_quadrant_range_l3920_392061

theorem complex_number_fourth_quadrant_range (a : ℝ) : 
  let z : ℂ := (2 + Complex.I) * (a + 2 * Complex.I^3)
  (z.re > 0 ∧ z.im < 0) → -1 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_fourth_quadrant_range_l3920_392061


namespace NUMINAMATH_CALUDE_union_complement_equal_l3920_392031

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equal : B ∪ (U \ A) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_l3920_392031
