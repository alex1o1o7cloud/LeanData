import Mathlib

namespace NUMINAMATH_CALUDE_base_number_problem_l2821_282163

theorem base_number_problem (x y a : ℝ) (h1 : x * y = 1) 
  (h2 : a ^ ((x + y)^2) / a ^ ((x - y)^2) = 625) : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_number_problem_l2821_282163


namespace NUMINAMATH_CALUDE_trapezoid_y_property_l2821_282182

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  c : ℝ
  -- Height of the trapezoid
  k : ℝ
  -- The segment joining midpoints divides the trapezoid into regions with area ratio 3:4
  midpoint_ratio : (c + 75) / (c + 150) = 3 / 4
  -- Length of the segment that divides the trapezoid into two equal areas
  y : ℝ
  -- The segment y divides the trapezoid into two equal areas
  equal_areas : y^2 = 65250

/-- The main theorem stating the property of y -/
theorem trapezoid_y_property (t : Trapezoid) : ⌊t.y^2 / 150⌋ = 435 := by
  sorry

#check trapezoid_y_property

end NUMINAMATH_CALUDE_trapezoid_y_property_l2821_282182


namespace NUMINAMATH_CALUDE_lock_and_key_theorem_l2821_282154

/-- The number of scientists in the team -/
def n : ℕ := 7

/-- The minimum number of scientists required to open the door -/
def k : ℕ := 4

/-- The number of scientists that can be absent -/
def m : ℕ := n - k

/-- The number of unique locks required -/
def num_locks : ℕ := Nat.choose n m

/-- The number of keys each scientist must have -/
def num_keys : ℕ := Nat.choose (n - 1) m

theorem lock_and_key_theorem :
  (num_locks = 35) ∧ (num_keys = 20) :=
sorry

end NUMINAMATH_CALUDE_lock_and_key_theorem_l2821_282154


namespace NUMINAMATH_CALUDE_dealership_van_sales_l2821_282156

/-- Calculates the expected number of vans to be sold given the truck-to-van ratio and the number of trucks expected to be sold. -/
def expected_vans (truck_ratio : ℕ) (van_ratio : ℕ) (trucks_sold : ℕ) : ℕ :=
  (van_ratio * trucks_sold) / truck_ratio

/-- Theorem stating that given a 3:5 ratio of trucks to vans and an expected sale of 45 trucks, 
    the expected number of vans to be sold is 75. -/
theorem dealership_van_sales : expected_vans 3 5 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dealership_van_sales_l2821_282156


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2821_282176

theorem absolute_value_inequality (x : ℝ) :
  |6 - x| / 4 > 1 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2821_282176


namespace NUMINAMATH_CALUDE_no_positive_sequence_with_recurrence_l2821_282166

theorem no_positive_sequence_with_recurrence : 
  ¬ ∃ (a : ℕ → ℝ), 
    (∀ n, a n > 0) ∧ 
    (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sequence_with_recurrence_l2821_282166


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l2821_282167

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l2821_282167


namespace NUMINAMATH_CALUDE_box_width_calculation_l2821_282144

theorem box_width_calculation (length depth : ℕ) (total_cubes : ℕ) (width : ℕ) : 
  length = 49 → 
  depth = 14 → 
  total_cubes = 84 → 
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    width % cube_side = 0 ∧
    (length / cube_side) * (depth / cube_side) * (width / cube_side) = total_cubes) →
  width = 42 := by
sorry

end NUMINAMATH_CALUDE_box_width_calculation_l2821_282144


namespace NUMINAMATH_CALUDE_system_solution_l2821_282185

theorem system_solution :
  ∃ (x y z : ℚ),
    (x + (1/3)*y + (1/3)*z = 14) ∧
    (y + (1/4)*x + (1/4)*z = 8) ∧
    (z + (1/5)*x + (1/5)*y = 8) ∧
    (x = 11) ∧ (y = 4) ∧ (z = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2821_282185


namespace NUMINAMATH_CALUDE_digits_9998_to_10000_of_1_998_l2821_282109

/-- The decimal expansion of 1/998 -/
def decimal_expansion_1_998 : ℕ → ℕ := sorry

/-- The function that extracts a 3-digit number from the decimal expansion -/
def extract_three_digits (start : ℕ) : ℕ := 
  100 * (decimal_expansion_1_998 start) + 
  10 * (decimal_expansion_1_998 (start + 1)) + 
  decimal_expansion_1_998 (start + 2)

/-- The theorem stating that the 9998th through 10000th digits of 1/998 form 042 -/
theorem digits_9998_to_10000_of_1_998 : 
  extract_three_digits 9998 = 42 := by sorry

end NUMINAMATH_CALUDE_digits_9998_to_10000_of_1_998_l2821_282109


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l2821_282100

theorem balloon_count_theorem (fred sam dan total : ℕ) 
  (h1 : fred = 10)
  (h2 : sam = 46)
  (h3 : dan = 16)
  (h4 : total = 72) :
  fred + sam + dan = total :=
sorry

end NUMINAMATH_CALUDE_balloon_count_theorem_l2821_282100


namespace NUMINAMATH_CALUDE_board_game_change_l2821_282122

theorem board_game_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 8 →
  game_cost = 18 →
  payment = 200 →
  change_bill = 10 →
  (payment - num_games * game_cost) / change_bill = 5 := by
sorry

end NUMINAMATH_CALUDE_board_game_change_l2821_282122


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2821_282172

theorem pipe_filling_time (rate_A rate_B : ℝ) (time_B : ℝ) : 
  rate_A = 1 / 12 →
  rate_B = 1 / 36 →
  time_B = 12 →
  ∃ time_A : ℝ, time_A * rate_A + time_B * rate_B = 1 ∧ time_A = 8 :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l2821_282172


namespace NUMINAMATH_CALUDE_equal_probability_sums_l2821_282184

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The minimum face value on each die -/
def min_face : ℕ := 1

/-- The maximum face value on each die -/
def max_face : ℕ := 6

/-- The sum we're comparing against -/
def sum1 : ℕ := 12

/-- The sum that should have the same probability as sum1 -/
def sum2 : ℕ := 44

/-- The probability of obtaining a specific sum when rolling num_dice dice -/
noncomputable def prob_sum (s : ℕ) : ℝ := sorry

theorem equal_probability_sums : prob_sum sum1 = prob_sum sum2 := by sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l2821_282184


namespace NUMINAMATH_CALUDE_train_speed_l2821_282159

/-- The speed of a train given its length, time to cross a bridge, and total length of train and bridge -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 130 →
  crossing_time = 30 →
  total_length = 245 →
  (total_length / crossing_time) * 3.6 = 29.4 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2821_282159


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2821_282113

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2821_282113


namespace NUMINAMATH_CALUDE_counterexample_exists_l2821_282123

theorem counterexample_exists : ∃ a : ℝ, (|a - 1| > 1) ∧ (a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2821_282123


namespace NUMINAMATH_CALUDE_line_characteristics_l2821_282111

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The line y = -x - 3 -/
def line : Line := { slope := -1, y_intercept := -3 }

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on the line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Check if the line passes through a quadrant -/
def Line.passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  ∃ (p : Point), l.contains p ∧
    match q with
    | 1 => p.x > 0 ∧ p.y > 0
    | 2 => p.x < 0 ∧ p.y > 0
    | 3 => p.x < 0 ∧ p.y < 0
    | 4 => p.x > 0 ∧ p.y < 0
    | _ => False

theorem line_characteristics :
  (line.passes_through_quadrant 2 ∧
   line.passes_through_quadrant 3 ∧
   line.passes_through_quadrant 4) ∧
  line.slope < 0 ∧
  line.contains { x := 0, y := -3 } ∧
  ¬ line.contains { x := 3, y := 0 } := by sorry

end NUMINAMATH_CALUDE_line_characteristics_l2821_282111


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2821_282133

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (5026 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (5026 - m) % 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2821_282133


namespace NUMINAMATH_CALUDE_car_acceleration_at_one_second_l2821_282161

-- Define the velocity function
def v (t : ℝ) : ℝ := -t^2 + 10*t

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := -2*t + 10

-- Theorem statement
theorem car_acceleration_at_one_second :
  a 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_acceleration_at_one_second_l2821_282161


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2821_282118

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (5,0) with radius 2.
    The sum of all possible radii of the circle with center C is 14. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2821_282118


namespace NUMINAMATH_CALUDE_power_sum_sequence_l2821_282158

/-- Given real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry


end NUMINAMATH_CALUDE_power_sum_sequence_l2821_282158


namespace NUMINAMATH_CALUDE_shadow_length_l2821_282170

/-- Given a flagpole and a building under similar conditions, this theorem calculates
    the length of the shadow cast by the building. -/
theorem shadow_length
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_height : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 24)
  : (building_height * flagpole_shadow) / flagpole_height = 60 := by
  sorry


end NUMINAMATH_CALUDE_shadow_length_l2821_282170


namespace NUMINAMATH_CALUDE_fruit_salad_weight_l2821_282119

/-- The total weight of fruit in Scarlett's fruit salad is 1.85 pounds. -/
theorem fruit_salad_weight :
  let melon : ℚ := 35/100
  let berries : ℚ := 48/100
  let grapes : ℚ := 29/100
  let pineapple : ℚ := 56/100
  let oranges : ℚ := 17/100
  melon + berries + grapes + pineapple + oranges = 185/100 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_weight_l2821_282119


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2821_282168

theorem solve_exponential_equation :
  ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ∧ x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2821_282168


namespace NUMINAMATH_CALUDE_contractor_engagement_l2821_282115

/-- Represents the contractor's engagement problem -/
def ContractorProblem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) : Prop :=
  ∃ (working_days : ℕ),
    daily_wage * working_days - daily_fine * absent_days = total_earnings ∧
    working_days + absent_days = 30

/-- The theorem states that given the problem conditions, the total engagement days is 30 -/
theorem contractor_engagement :
  ContractorProblem 25 7.5 425 10 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_l2821_282115


namespace NUMINAMATH_CALUDE_circle_line_properties_l2821_282120

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (m + 1)*y - 5*m - 3 = 0

-- Theorem statement
theorem circle_line_properties :
  -- Line l intersects circle C
  (∃ (m x y : ℝ), circle_C x y ∧ line_l m x y) ∧
  -- The chord length intercepted by circle C on the y-axis is 4√6
  (∃ (y1 y2 : ℝ), circle_C 0 y1 ∧ circle_C 0 y2 ∧ y2 - y1 = 4 * Real.sqrt 6) ∧
  -- When the chord length intercepted by circle C is the shortest, the equation of line l is x=1
  (∃ (m : ℝ), ∀ (x y : ℝ), line_l m x y → x = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_line_properties_l2821_282120


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l2821_282199

/-- The speed of cyclist C in mph -/
def speed_C : ℝ := 9

/-- The speed of cyclist D in mph -/
def speed_D : ℝ := speed_C + 6

/-- The distance between Newport and Kingston in miles -/
def distance : ℝ := 80

/-- The distance from Kingston where cyclists meet on D's return journey in miles -/
def meeting_distance : ℝ := 20

theorem cyclist_speed_problem :
  speed_C = 9 ∧
  speed_D = speed_C + 6 ∧
  distance / speed_C = (distance + meeting_distance) / speed_D :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l2821_282199


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2821_282110

theorem min_coefficient_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box*x + 30) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∀ Box' : ℤ, (∀ x, (a*x + b) * (b*x + a) = 30*x^2 + Box'*x + 30) → Box' ≥ Box) →
  Box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2821_282110


namespace NUMINAMATH_CALUDE_tara_book_sales_l2821_282181

/-- The number of books Tara needs to sell to reach her goal -/
def books_to_sell (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let goal := clarinet_cost - initial_savings
  let halfway := goal / 2
  let books_to_halfway := halfway / book_price
  let new_goal := goal + accessory_cost
  let books_after_loss := new_goal / book_price
  books_to_halfway + books_after_loss

/-- Theorem stating that Tara needs to sell 35 books in total -/
theorem tara_book_sales :
  books_to_sell 10 90 4 20 = 35 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_l2821_282181


namespace NUMINAMATH_CALUDE_inequality_proof_l2821_282177

theorem inequality_proof (a m : ℝ) (ha : a > 0) :
  abs (m + a) + abs (m + 1 / a) + abs (-1 / m + a) + abs (-1 / m + 1 / a) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2821_282177


namespace NUMINAMATH_CALUDE_vector_u_satisfies_equation_l2821_282114

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 2]

theorem vector_u_satisfies_equation :
  let u : Matrix (Fin 2) (Fin 1) ℝ := !![5/273; 8/21]
  (B^5 + B^3 + B) * u = !![5; 16] := by
  sorry

end NUMINAMATH_CALUDE_vector_u_satisfies_equation_l2821_282114


namespace NUMINAMATH_CALUDE_percentage_of_70_to_125_l2821_282116

theorem percentage_of_70_to_125 : ∃ p : ℚ, p = 70 / 125 * 100 ∧ p = 56 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_70_to_125_l2821_282116


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2821_282117

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- The main theorem stating that a convex polyhedron with given properties has 310 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 15 ∧
    Q.pentagon_faces = 5 ∧
    space_diagonals Q = 310 :=
  sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l2821_282117


namespace NUMINAMATH_CALUDE_expression_value_l2821_282151

def point_on_terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = -2 ∧ x = Real.cos α ∧ y = Real.sin α

theorem expression_value (α : Real) (h : point_on_terminal_side α) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2821_282151


namespace NUMINAMATH_CALUDE_revenue_change_l2821_282183

theorem revenue_change 
  (P : ℝ) 
  (N : ℝ) 
  (price_decrease : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_decrease = 0.2) 
  (h2 : sales_increase = 0.6) 
  : (1 - price_decrease) * (1 + sales_increase) * (P * N) = 1.28 * (P * N) := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l2821_282183


namespace NUMINAMATH_CALUDE_grade_assignments_l2821_282152

/-- The number of students in the class -/
def num_students : ℕ := 8

/-- The number of distinct grades available -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 65536 := by sorry

end NUMINAMATH_CALUDE_grade_assignments_l2821_282152


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2821_282139

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def solution_count : ℕ := 3

/-- Predicate defining the conditions for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 128

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), ∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧
  (∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_pair p.1 p.2) ∧ s.card = solution_count) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l2821_282139


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l2821_282103

/-- Given a quadratic equation ax² + bx + 3 = 0 with roots -2 and 3,
    prove that the equation a(x+2)² + b(x+2) + 3 = 0 has roots -4 and 1 -/
theorem quadratic_root_transformation (a b : ℝ) :
  (∃ x, a * x^2 + b * x + 3 = 0) →
  ((-2 : ℝ) * (-2 : ℝ) * a + (-2 : ℝ) * b + 3 = 0) →
  ((3 : ℝ) * (3 : ℝ) * a + (3 : ℝ) * b + 3 = 0) →
  (a * ((-4 : ℝ) + 2)^2 + b * ((-4 : ℝ) + 2) + 3 = 0) ∧
  (a * ((1 : ℝ) + 2)^2 + b * ((1 : ℝ) + 2) + 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_transformation_l2821_282103


namespace NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l2821_282135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem intersection_points_sum_greater_than_two (a t x₁ x₂ : ℝ) 
  (ha : a ≤ 0) (ht : -1 < t ∧ t < 0) (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf₁ : f a x₁ = t) (hf₂ : f a x₂ = t) : 
  x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_points_sum_greater_than_two_l2821_282135


namespace NUMINAMATH_CALUDE_expression_simplification_l2821_282131

theorem expression_simplification (x y : ℝ) :
  let P := 2 * x + 3 * y
  let Q := 3 * x + 2 * y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (24 * x^2 + 52 * x * y + 24 * y^2) / (5 * x * y - 5 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2821_282131


namespace NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l2821_282149

/-- Given that James ate 22 carrot sticks before dinner and 37 carrot sticks in total,
    prove that he ate 15 carrot sticks after dinner. -/
theorem carrot_sticks_after_dinner
  (before_dinner : ℕ)
  (total : ℕ)
  (h1 : before_dinner = 22)
  (h2 : total = 37) :
  total - before_dinner = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l2821_282149


namespace NUMINAMATH_CALUDE_final_silver_count_l2821_282129

/-- Represents the number of tokens Alex has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents a token exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.redIn ∧ tokens.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.redIn + booth.redOut,
    blue := tokens.blue - booth.blueIn + booth.blueOut,
    silver := tokens.silver + booth.silverOut }

/-- Checks if any exchanges are possible -/
def exchangesPossible (tokens : TokenCount) (booths : List Booth) : Prop :=
  ∃ b ∈ booths, canExchange tokens b

/-- The main theorem to prove -/
theorem final_silver_count 
  (initialTokens : TokenCount)
  (booth1 booth2 : Booth)
  (h_initial : initialTokens = ⟨75, 75, 0⟩)
  (h_booth1 : booth1 = ⟨2, 0, 0, 1, 1⟩)
  (h_booth2 : booth2 = ⟨0, 3, 1, 0, 1⟩)
  : ∃ (finalTokens : TokenCount), 
    (¬ exchangesPossible finalTokens [booth1, booth2]) ∧ 
    finalTokens.silver = 103 := by
  sorry

end NUMINAMATH_CALUDE_final_silver_count_l2821_282129


namespace NUMINAMATH_CALUDE_quadratic_m_range_l2821_282197

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 - m * x + 1 = 0

def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂

def roots_in_range (m : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_equation m x → x ≥ 1/2 ∧ x ≤ 4

theorem quadratic_m_range :
  ∀ m : ℝ, (has_two_distinct_roots m ∧ roots_in_range m) ↔ (m > 2 * Real.sqrt 2 ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_m_range_l2821_282197


namespace NUMINAMATH_CALUDE_bookstore_shipment_count_l2821_282157

theorem bookstore_shipment_count :
  ∀ (total : ℕ) (displayed : ℕ) (stockroom : ℕ),
    displayed = (30 : ℕ) * total / 100 →
    stockroom = (70 : ℕ) * total / 100 →
    stockroom = 140 →
    total = 200 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_count_l2821_282157


namespace NUMINAMATH_CALUDE_pie_sale_profit_l2821_282180

/-- Calculates the profit from selling pies given the number of pies, costs, and selling price -/
theorem pie_sale_profit
  (num_pumpkin : ℕ)
  (num_cherry : ℕ)
  (cost_pumpkin : ℕ)
  (cost_cherry : ℕ)
  (selling_price : ℕ)
  (h1 : num_pumpkin = 10)
  (h2 : num_cherry = 12)
  (h3 : cost_pumpkin = 3)
  (h4 : cost_cherry = 5)
  (h5 : selling_price = 5) :
  (num_pumpkin + num_cherry) * selling_price - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = 20 :=
by sorry

end NUMINAMATH_CALUDE_pie_sale_profit_l2821_282180


namespace NUMINAMATH_CALUDE_main_theorem_l2821_282150

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def even_symmetric (f : ℝ → ℝ) : Prop := ∀ x, f x - f (-x) = 0

def symmetric_about_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (2 - x)

def increasing_on_zero_two (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorems to be proved
def periodic_four (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

def decreasing_on_two_four (f : ℝ → ℝ) : Prop := 
  ∀ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f y < f x

-- Main theorem
theorem main_theorem (heven : even_symmetric f) 
                     (hsym : symmetric_about_two f) 
                     (hinc : increasing_on_zero_two f) : 
  periodic_four f ∧ decreasing_on_two_four f := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l2821_282150


namespace NUMINAMATH_CALUDE_abc_sum_product_zero_l2821_282107

theorem abc_sum_product_zero (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by sorry

end NUMINAMATH_CALUDE_abc_sum_product_zero_l2821_282107


namespace NUMINAMATH_CALUDE_product_of_roots_l2821_282155

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 375 = 0) → 
  (∃ r₁ r₂ : ℝ, (25 * r₁^2 + 60 * r₁ - 375 = 0) ∧ 
                (25 * r₂^2 + 60 * r₂ - 375 = 0) ∧ 
                (r₁ * r₂ = -15)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2821_282155


namespace NUMINAMATH_CALUDE_outfit_count_l2821_282169

/-- The number of different outfits that can be made with the given clothing items. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits for the given clothing items. -/
theorem outfit_count :
  number_of_outfits 7 4 5 2 = 504 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l2821_282169


namespace NUMINAMATH_CALUDE_sequence_problem_l2821_282188

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_a_sum : a 1 + a 5 + a 9 = 9)
  (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
  (a 2 + a 8) / (1 + b 2 * b 8) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2821_282188


namespace NUMINAMATH_CALUDE_larger_number_proof_l2821_282195

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2500) (h3 : L = 6 * S + 15) : L = 2997 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2821_282195


namespace NUMINAMATH_CALUDE_positive_y_squared_geq_2y_minus_1_l2821_282132

theorem positive_y_squared_geq_2y_minus_1 :
  ∀ y : ℝ, y > 0 → y^2 ≥ 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_y_squared_geq_2y_minus_1_l2821_282132


namespace NUMINAMATH_CALUDE_table_relationship_l2821_282187

def f (x : ℕ) : ℤ := 21 - x^2

theorem table_relationship : 
  (f 0 = 21) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 16) ∧ 
  (f 3 = 9) ∧ 
  (f 4 = 0) := by
  sorry

#check table_relationship

end NUMINAMATH_CALUDE_table_relationship_l2821_282187


namespace NUMINAMATH_CALUDE_inequality_of_four_variables_l2821_282126

theorem inequality_of_four_variables (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_four_variables_l2821_282126


namespace NUMINAMATH_CALUDE_correct_divisor_l2821_282198

theorem correct_divisor (D : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : D = mistaken_divisor * 35)
  (h3 : correct_quotient = 20)
  (h4 : D % (D / correct_quotient) = 0) :
  D / correct_quotient = 21 := by
sorry

end NUMINAMATH_CALUDE_correct_divisor_l2821_282198


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2821_282138

theorem root_equation_implies_expression_value (m : ℝ) : 
  2 * m^2 - 3 * m - 1 = 0 → 6 * m^2 - 9 * m + 2021 = 2024 := by
sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2821_282138


namespace NUMINAMATH_CALUDE_difference_of_squares_305_295_l2821_282193

theorem difference_of_squares_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_305_295_l2821_282193


namespace NUMINAMATH_CALUDE_cubic_harmonic_mean_root_condition_l2821_282190

/-- 
Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and d ≠ 0,
if one of its roots is equal to the harmonic mean of the other two roots,
then the coefficients satisfy the equation 27ad² - 9bcd + 2c³ = 0.
-/
theorem cubic_harmonic_mean_root_condition (a b c d : ℝ) 
  (ha : a ≠ 0) (hd : d ≠ 0) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (a * x₁^3 + b * x₁^2 + c * x₁ + d = 0) ∧ 
    (a * x₂^3 + b * x₂^2 + c * x₂ + d = 0) ∧ 
    (a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) ∧ 
    (x₂ = 2 * x₁ * x₃ / (x₁ + x₃))) →
  27 * a * d^2 - 9 * b * c * d + 2 * c^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_harmonic_mean_root_condition_l2821_282190


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l2821_282140

theorem function_satisfying_conditions : ∃ (f : ℝ → ℝ), 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (2 - x) + f x = 0) ∧ 
  (∀ x, f x = Real.cos (Real.pi / 2 * x)) ∧
  (∃ x y, f x ≠ f y) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l2821_282140


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2821_282164

/-- Given a geometric sequence with first term a₁ = 2 and common ratio r = 3,
    prove that the 5th term (x) is 162 -/
theorem geometric_sequence_fifth_term 
  (a₁ : ℝ) 
  (r : ℝ) 
  (x : ℝ) 
  (h1 : a₁ = 2) 
  (h2 : r = 3) 
  (h3 : x = a₁ * r^4) : x = 162 := by
  sorry

#check geometric_sequence_fifth_term

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2821_282164


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2821_282142

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 16) : 
  x^2 - y^2 = 960 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2821_282142


namespace NUMINAMATH_CALUDE_probability_at_least_three_same_l2821_282134

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def total_outcomes : ℕ := num_sides ^ num_dice

def favorable_outcomes : ℕ :=
  -- Exactly 3 dice showing the same number
  (num_sides * (num_dice.choose 3) * (num_sides - 1)^2) +
  -- Exactly 4 dice showing the same number
  (num_sides * (num_dice.choose 4) * (num_sides - 1)) +
  -- All 5 dice showing the same number
  num_sides

theorem probability_at_least_three_same (h : favorable_outcomes = 4208) :
  (favorable_outcomes : ℚ) / total_outcomes = 1052 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_same_l2821_282134


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2821_282160

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1/3) (h2 : b = -2) :
  ((3*a + b)^2 - (3*a + b)*(3*a - b)) / (2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2821_282160


namespace NUMINAMATH_CALUDE_xy_squared_value_l2821_282165

theorem xy_squared_value (x y : ℤ) (h : y^2 + 2*x^2*y^2 = 20*x^2 + 412) : 2*x*y^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_value_l2821_282165


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2821_282174

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)  -- magnitude of a
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1)  -- magnitude of b
  : Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2821_282174


namespace NUMINAMATH_CALUDE_cookie_averages_l2821_282153

def brand_x_packages : List ℕ := [6, 8, 9, 11, 13]
def brand_y_packages : List ℕ := [14, 15, 18, 20]

theorem cookie_averages :
  let x_total := brand_x_packages.sum
  let y_total := brand_y_packages.sum
  let x_avg : ℚ := x_total / brand_x_packages.length
  let y_avg : ℚ := y_total / brand_y_packages.length
  x_avg = 47 / 5 ∧ y_avg = 67 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_averages_l2821_282153


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l2821_282189

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 20 = 100) :
  ∃ m : ℝ, m = 20 ∧ ∀ x : ℝ, (a 7 + a 14 ≥ x ∧ (∃ y : ℝ, a 7 = y ∧ a 14 = y → a 7 + a 14 = x)) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l2821_282189


namespace NUMINAMATH_CALUDE_tonys_walking_speed_l2821_282194

/-- Proves that Tony's walking speed is 2 MPH given the problem conditions -/
theorem tonys_walking_speed :
  let store_distance : ℝ := 4
  let running_speed : ℝ := 10
  let average_time_minutes : ℝ := 56
  let walking_speed : ℝ := 2

  (walking_speed * store_distance + 2 * (store_distance / running_speed) * 60) / 3 = average_time_minutes
  ∧ walking_speed > 0 := by sorry

end NUMINAMATH_CALUDE_tonys_walking_speed_l2821_282194


namespace NUMINAMATH_CALUDE_election_majority_l2821_282178

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6000 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1200 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l2821_282178


namespace NUMINAMATH_CALUDE_surf_festival_problem_l2821_282179

/-- The Rip Curl Myrtle Beach Surf Festival problem -/
theorem surf_festival_problem (total_surfers : ℝ) (S1 : ℝ) :
  total_surfers = 15000 ∧
  S1 + 0.9 * S1 + 1.5 * S1 + (S1 + 0.9 * S1) + 0.5 * (S1 + 0.9 * S1) = total_surfers →
  S1 = 2400 ∧ total_surfers / 5 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_surf_festival_problem_l2821_282179


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l2821_282137

theorem square_area_equal_perimeter (a b c s : ℝ) : 
  a = 6 → b = 8 → c = 10 → -- Triangle side lengths
  a^2 + b^2 = c^2 →        -- Right-angled triangle condition
  4 * s = a + b + c →      -- Equal perimeter condition
  s^2 = 36 :=              -- Square area
by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l2821_282137


namespace NUMINAMATH_CALUDE_range_of_m_for_necessary_condition_l2821_282148

theorem range_of_m_for_necessary_condition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → (x < m - 1 ∨ x > m + 1)) ∧ 
  (∃ x : ℝ, (x < m - 1 ∨ x > m + 1) ∧ x^2 - 2*x - 3 ≤ 0) ↔ 
  m ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_necessary_condition_l2821_282148


namespace NUMINAMATH_CALUDE_function_inequality_l2821_282191

/-- Given f(x) = e^(2x) - ax, for all x > 0, if f(x) > ax^2 + 1, then a ≤ 2 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, Real.exp (2 * x) - a * x > a * x^2 + 1) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2821_282191


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2821_282104

theorem roots_quadratic_equation (m : ℝ) (a b : ℝ) (s t : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - s*(a^2 + 1/b^2) + t = 0) →
  ((b^2 + 1/a^2)^2 - s*(b^2 + 1/a^2) + t = 0) →
  t = 100/9 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2821_282104


namespace NUMINAMATH_CALUDE_angle_and_complement_differ_by_20_l2821_282105

theorem angle_and_complement_differ_by_20 (α : ℝ) : 
  α - (90 - α) = 20 → α = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_and_complement_differ_by_20_l2821_282105


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_six_l2821_282128

/-- The cost of each chocolate bar, given the total number of bars, 
    the number of unsold bars, and the total revenue from sales. -/
def chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (revenue : ℕ) : ℚ :=
  revenue / (total_bars - unsold_bars)

/-- Theorem stating that the cost of each chocolate bar is $6 under the given conditions. -/
theorem chocolate_bar_cost_is_six : 
  chocolate_bar_cost 13 6 42 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_six_l2821_282128


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2821_282141

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - 3 * Complex.I) / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2821_282141


namespace NUMINAMATH_CALUDE_multiplication_result_l2821_282130

theorem multiplication_result : 9995 * 82519 = 824777405 := by sorry

end NUMINAMATH_CALUDE_multiplication_result_l2821_282130


namespace NUMINAMATH_CALUDE_right_triangle_area_leg_sum_l2821_282108

theorem right_triangle_area_leg_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → 
  a^2 + b^2 = c^2 → 
  (a * b) / 2 + a = 75 ∨ (a * b) / 2 + b = 75 →
  ((a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 23 ∧ b = 6 ∧ c = 25) ∨
   (a = 15 ∧ b = 8 ∧ c = 17) ∨ (a = 8 ∧ b = 15 ∧ c = 17)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_leg_sum_l2821_282108


namespace NUMINAMATH_CALUDE_no_real_roots_l2821_282101

theorem no_real_roots (k : ℝ) (h : 12 - 3 * k < 0) : 
  ∀ x : ℝ, x^2 + 4*x + k ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l2821_282101


namespace NUMINAMATH_CALUDE_find_S_l2821_282127

theorem find_S : ∃ S : ℝ, 
  (∀ a b c d : ℝ, 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1 ∧
    a + b + c + d = S ∧
    1/a + 1/b + 1/c + 1/d = S ∧
    1/(a-1) + 1/(b-1) + 1/(c-1) + 1/(d-1) = S) →
  S = -2 :=
by sorry

end NUMINAMATH_CALUDE_find_S_l2821_282127


namespace NUMINAMATH_CALUDE_jason_retirement_age_l2821_282145

def military_career (join_age time_to_chief : ℕ) : Prop :=
  let time_to_senior_chief : ℕ := time_to_chief + (time_to_chief / 4)
  let time_to_master_chief : ℕ := time_to_senior_chief - (time_to_senior_chief / 10)
  let time_to_command_master_chief : ℕ := time_to_master_chief + (time_to_master_chief / 2)
  let additional_time : ℕ := 5
  let total_service_time : ℕ := time_to_chief + time_to_senior_chief + time_to_master_chief + 
                                 time_to_command_master_chief + additional_time
  join_age + total_service_time = 63

theorem jason_retirement_age : 
  military_career 18 8 := by sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l2821_282145


namespace NUMINAMATH_CALUDE_solution_sets_equivalence_l2821_282146

open Set

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}

-- Define the coefficients a, b, c based on the given conditions
def a : ℝ := -1  -- Assume a = -1 for simplicity, since we know a < 0
def b : ℝ := -2 * a
def c : ℝ := -3 * a

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/3 < x ∧ x < 1}

-- Theorem statement
theorem solution_sets_equivalence : 
  (∀ x : ℝ, x ∈ solution_set_1 ↔ a * x^2 + b * x + c ≤ 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ c * x^2 - b * x + a < 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equivalence_l2821_282146


namespace NUMINAMATH_CALUDE_jill_jack_time_ratio_l2821_282171

/-- The ratio of Jill's time to Jack's time for a given route -/
theorem jill_jack_time_ratio (d : ℝ) (x y : ℝ) : 
  (x = d / (2 * 6) + d / (2 * 12)) →
  (y = d / (3 * 5) + 2 * d / (3 * 15)) →
  x / y = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_jill_jack_time_ratio_l2821_282171


namespace NUMINAMATH_CALUDE_divisibility_condition_l2821_282121

theorem divisibility_condition (x y : ℕ+) :
  (xy^2 + y + 7 ∣ x^2*y + x + y) ↔ 
  (∃ t : ℕ+, x = 7*t^2 ∧ y = 7*t) ∨ (x = 11 ∧ y = 1) ∨ (x = 49 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2821_282121


namespace NUMINAMATH_CALUDE_max_value_e_l2821_282196

theorem max_value_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  e ≤ 16/5 ∧ ∃ (a' b' c' d' e' : ℝ), 
    a' + b' + c' + d' + e' = 8 ∧
    a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 16 ∧
    e' = 16/5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_e_l2821_282196


namespace NUMINAMATH_CALUDE_eddie_dump_rate_l2821_282162

/-- Given that Sam dumps tea for 6 hours at 60 crates per hour,
    and Eddie takes 4 hours to dump the same amount,
    prove that Eddie's rate is 90 crates per hour. -/
theorem eddie_dump_rate 
  (sam_hours : ℕ) 
  (sam_rate : ℕ) 
  (eddie_hours : ℕ) 
  (h1 : sam_hours = 6)
  (h2 : sam_rate = 60)
  (h3 : eddie_hours = 4)
  (h4 : sam_hours * sam_rate = eddie_hours * eddie_rate) :
  eddie_rate = 90 :=
by
  sorry

#check eddie_dump_rate

end NUMINAMATH_CALUDE_eddie_dump_rate_l2821_282162


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2821_282147

theorem triangle_side_difference (x : ℤ) : 
  (∀ y : ℤ, 3 ≤ y ∧ y ≤ 17 → (y + 8 > 10 ∧ y + 10 > 8 ∧ 8 + 10 > y)) →
  (∀ z : ℤ, z < 3 ∨ z > 17 → ¬(z + 8 > 10 ∧ z + 10 > 8 ∧ 8 + 10 > z)) →
  (17 - 3 : ℤ) = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2821_282147


namespace NUMINAMATH_CALUDE_eight_circle_times_three_l2821_282106

-- Define the new operation ⨳
def circle_times (a b : ℤ) : ℤ := 4 * a + 6 * b

-- The theorem to prove
theorem eight_circle_times_three : circle_times 8 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_eight_circle_times_three_l2821_282106


namespace NUMINAMATH_CALUDE_total_spent_is_20_l2821_282186

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 4

/-- The price of a keychain in dollars -/
def keychain_price : ℕ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℕ := 3

/-- The number of bracelets Paula buys -/
def paula_bracelets : ℕ := 2

/-- The number of keychains Paula buys -/
def paula_keychains : ℕ := 1

/-- The number of coloring books Olive buys -/
def olive_coloring_books : ℕ := 1

/-- The number of bracelets Olive buys -/
def olive_bracelets : ℕ := 1

/-- The total amount spent by Paula and Olive -/
def total_spent : ℕ :=
  paula_bracelets * bracelet_price +
  paula_keychains * keychain_price +
  olive_coloring_books * coloring_book_price +
  olive_bracelets * bracelet_price

theorem total_spent_is_20 : total_spent = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_20_l2821_282186


namespace NUMINAMATH_CALUDE_wills_initial_money_l2821_282143

/-- Will's initial amount of money -/
def initial_money : ℕ := 57

/-- Cost of the game Will bought -/
def game_cost : ℕ := 27

/-- Number of toys Will can buy with the remaining money -/
def num_toys : ℕ := 5

/-- Cost of each toy -/
def toy_cost : ℕ := 6

/-- Theorem stating that Will's initial money is correct given the conditions -/
theorem wills_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_wills_initial_money_l2821_282143


namespace NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l2821_282124

/-- A function that checks if three numbers form a valid triangle --/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that generates three consecutive odd integers --/
def consecutive_odd_integers (n : ℕ) : (ℕ × ℕ × ℕ) :=
  (2*n + 1, 2*n + 3, 2*n + 5)

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer side lengths --/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    let (a, b, c) := consecutive_odd_integers n
    is_valid_triangle a b c ∧
    ∀ (m : ℕ), m < n → ¬(is_valid_triangle (2*m + 1) (2*m + 3) (2*m + 5)) ∧
    a + b + c = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l2821_282124


namespace NUMINAMATH_CALUDE_problem_solution_l2821_282125

open Set

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x ≤ a + 6}

theorem problem_solution (a : ℝ) (h : A ⊆ M a) :
  ((𝒰 \ B) ∩ A = {x | -3 ≤ x ∧ x ≤ 2}) ∧ (a ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2821_282125


namespace NUMINAMATH_CALUDE_annieka_free_throws_l2821_282102

def free_throws_problem (deshawn kayla annieka : ℕ) : Prop :=
  deshawn = 12 ∧
  kayla = deshawn + (deshawn / 2) ∧
  annieka = kayla - 4

theorem annieka_free_throws :
  ∀ deshawn kayla annieka : ℕ,
    free_throws_problem deshawn kayla annieka →
    annieka = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_annieka_free_throws_l2821_282102


namespace NUMINAMATH_CALUDE_value_of_a_l2821_282136

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 9 * x^2 + 6 * x - 7

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 18 * x + 6

-- Theorem statement
theorem value_of_a (a : ℝ) : f' a (-1) = 4 → a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2821_282136


namespace NUMINAMATH_CALUDE_median_and_midline_projection_l2821_282192

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a parallel projection
def ParallelProjection := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a median of a triangle
def median (t : Triangle) : (ℝ × ℝ) := sorry

-- Define a midline of a triangle
def midline (t : Triangle) : (ℝ × ℝ) := sorry

-- Theorem statement
theorem median_and_midline_projection 
  (t : Triangle) 
  (p : ParallelProjection) 
  (h : ∃ t', t' = Triangle.mk (p t.A) (p t.B) (p t.C)) :
  (p (median t) = median (Triangle.mk (p t.A) (p t.B) (p t.C))) ∧
  (p (midline t) = midline (Triangle.mk (p t.A) (p t.B) (p t.C))) := by
  sorry

end NUMINAMATH_CALUDE_median_and_midline_projection_l2821_282192


namespace NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l2821_282112

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  let n : ℕ := 12
  let k : ℕ := 2
  let p : ℚ := 1/6
  probability_two_ones n k p = 66 * (1/36) * (9765625/60466176) :=
sorry

end NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l2821_282112


namespace NUMINAMATH_CALUDE_inequality_proof_l2821_282173

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : -1 < b ∧ b < 0) : a * b^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2821_282173


namespace NUMINAMATH_CALUDE_no_prime_sqrt_sum_integer_l2821_282175

theorem no_prime_sqrt_sum_integer :
  ¬ ∃ (p n : ℕ), Prime p ∧ n > 0 ∧ ∃ (k : ℤ), (Int.sqrt (p + n) + Int.sqrt n : ℤ) = k :=
sorry

end NUMINAMATH_CALUDE_no_prime_sqrt_sum_integer_l2821_282175
