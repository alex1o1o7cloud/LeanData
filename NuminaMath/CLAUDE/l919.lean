import Mathlib

namespace babylon_sphere_properties_l919_91973

structure Sphere :=
  (holes : Nat)
  (angle_step : Real)

def ray_pairs (s : Sphere) : Nat :=
  (s.holes * (s.holes - 1)) / 2

def angle_between_rays (s : Sphere) (r1 r2 : Nat × Nat) : Real :=
  sorry  -- Function to calculate angle between two rays

def count_angle_pairs (s : Sphere) (angle : Real) : Nat :=
  sorry  -- Function to count pairs of rays forming a specific angle

def can_construct_polyhedron (s : Sphere) (polyhedron : String) : Prop :=
  sorry  -- Predicate to determine if a polyhedron can be constructed

theorem babylon_sphere_properties (s : Sphere) 
  (h1 : s.holes = 26) 
  (h2 : s.angle_step = 45) : 
  (count_angle_pairs s (45 : Real) = 40) ∧ 
  (count_angle_pairs s (60 : Real) = 48) ∧ 
  (can_construct_polyhedron s "tetrahedron") ∧ 
  (can_construct_polyhedron s "octahedron") ∧ 
  ¬(can_construct_polyhedron s "dual_tetrahedron") :=
by
  sorry

end babylon_sphere_properties_l919_91973


namespace line_intersects_parabola_vertex_l919_91939

/-- The number of real values of b for which the line y = 2x + b intersects
    the parabola y = x^2 - 4x + b^2 at its vertex -/
theorem line_intersects_parabola_vertex : 
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b ∈ s, ∃ x y : ℝ, 
    (y = 2 * x + b) ∧ 
    (y = x^2 - 4 * x + b^2) ∧
    (∀ x' y' : ℝ, (y' = x'^2 - 4 * x' + b^2) → y' ≤ y) :=
sorry

end line_intersects_parabola_vertex_l919_91939


namespace smallest_positive_integer_l919_91980

theorem smallest_positive_integer (x : ℕ+) : (x^3 : ℚ) / (x^2 : ℚ) < 15 → x = 1 := by
  sorry

end smallest_positive_integer_l919_91980


namespace simplify_power_expression_l919_91970

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end simplify_power_expression_l919_91970


namespace divisors_of_300_l919_91976

/-- Given that 300 = 2 × 2 × 3 × 5 × 5, prove that 300 has 18 divisors -/
theorem divisors_of_300 : ∃ (d : Finset Nat), Finset.card d = 18 ∧ 
  (∀ x : Nat, x ∈ d ↔ (x ∣ 300)) := by
  sorry

end divisors_of_300_l919_91976


namespace x_range_for_inequality_l919_91913

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2*x - 1 > m*(x^2 - 1)) ↔ 
  ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2) :=
by sorry

end x_range_for_inequality_l919_91913


namespace smallest_four_digit_mod_8_l919_91977

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 := by
  sorry

end smallest_four_digit_mod_8_l919_91977


namespace cubic_root_function_l919_91914

/-- Given a function y = kx^(1/3) where y = 4 when x = 8, 
    prove that y = 6 when x = 27 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (8 : ℝ)^(1/3) ∧ y = 4) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end cubic_root_function_l919_91914


namespace function_has_one_zero_l919_91986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 - x

theorem function_has_one_zero (a : ℝ) (h1 : |a| ≥ 1 / (2 * Real.exp 1)) 
  (h2 : ∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end function_has_one_zero_l919_91986


namespace student_increase_proof_l919_91955

/-- Represents the increase in the number of students in a hostel -/
def student_increase : ℕ := sorry

/-- The initial number of students in the hostel -/
def initial_students : ℕ := 35

/-- The original daily expenditure of the mess in rupees -/
def original_expenditure : ℕ := 420

/-- The increase in daily mess expenses in rupees when the number of students increases -/
def expense_increase : ℕ := 42

/-- The decrease in average expenditure per student in rupees when the number of students increases -/
def average_expense_decrease : ℕ := 1

/-- Calculates the new total expenditure after the increase in students -/
def new_total_expenditure : ℕ := (initial_students + student_increase) * 
  (original_expenditure / initial_students - average_expense_decrease)

theorem student_increase_proof : 
  new_total_expenditure = original_expenditure + expense_increase ∧ 
  student_increase = 7 := by sorry

end student_increase_proof_l919_91955


namespace field_trip_van_occupancy_l919_91916

/-- Proves the number of people in each van for a field trip --/
theorem field_trip_van_occupancy (num_vans : ℝ) (num_buses : ℝ) (people_per_bus : ℝ) (extra_people_in_buses : ℝ) :
  num_vans = 6.0 →
  num_buses = 8.0 →
  people_per_bus = 18.0 →
  extra_people_in_buses = 108 →
  num_buses * people_per_bus = num_vans * (num_buses * people_per_bus - extra_people_in_buses) / num_vans + extra_people_in_buses →
  (num_buses * people_per_bus - extra_people_in_buses) / num_vans = 6.0 := by
  sorry

#eval (8.0 * 18.0 - 108) / 6.0  -- Should output 6.0

end field_trip_van_occupancy_l919_91916


namespace any_nonzero_to_zero_power_is_one_l919_91938

theorem any_nonzero_to_zero_power_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end any_nonzero_to_zero_power_is_one_l919_91938


namespace floor_sqrt_23_squared_l919_91922

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end floor_sqrt_23_squared_l919_91922


namespace sum_of_squares_l919_91923

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end sum_of_squares_l919_91923


namespace fractional_equation_solution_range_l919_91954

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 3 ∧ x ≠ 2 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → 
  m < 6 ∧ m ≠ 3 := by
sorry

end fractional_equation_solution_range_l919_91954


namespace quadratic_inequality_empty_solution_set_l919_91967

theorem quadratic_inequality_empty_solution_set (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - m * x + m - 1 ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end quadratic_inequality_empty_solution_set_l919_91967


namespace min_value_expression_equality_condition_l919_91987

theorem min_value_expression (x : ℝ) :
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  ∃ x : ℝ, x = 2/3 ∧ 
    Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end min_value_expression_equality_condition_l919_91987


namespace reese_savings_problem_l919_91906

/-- Represents the percentage of savings spent in March -/
def march_spending_percentage : ℝ → Prop := λ M =>
  let initial_savings : ℝ := 11000
  let february_spending : ℝ := 0.2 * initial_savings
  let march_spending : ℝ := M * initial_savings
  let april_spending : ℝ := 1500
  let remaining : ℝ := 2900
  initial_savings - february_spending - march_spending - april_spending = remaining ∧
  M = 0.4

theorem reese_savings_problem :
  ∃ M : ℝ, march_spending_percentage M :=
sorry

end reese_savings_problem_l919_91906


namespace sum_of_digits_l919_91974

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem sum_of_digits (x y : ℕ) : 
  (x < 10) → 
  (y < 10) → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 13 := by
  sorry

end sum_of_digits_l919_91974


namespace inequality_implies_upper_bound_l919_91957

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by
  sorry

end inequality_implies_upper_bound_l919_91957


namespace three_times_value_interval_examples_l919_91961

/-- A function has a k-times value interval if there exists a closed interval [a,b]
    such that the function is monotonic on [a,b] and its range on [a,b] is [ka,kb] --/
def has_k_times_value_interval (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
  (∀ y, f a ≤ y ∧ y ≤ f b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y) ∧
  f a = k * a ∧ f b = k * b

theorem three_times_value_interval_examples :
  (has_k_times_value_interval (fun x => 1 / x) 3) ∧
  (has_k_times_value_interval (fun x => x ^ 2) 3) := by
  sorry

end three_times_value_interval_examples_l919_91961


namespace division_with_remainder_l919_91941

theorem division_with_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 76 → 
  quotient = 4 → 
  divisor = 17 → 
  dividend = divisor * quotient + remainder → 
  remainder = 8 := by
sorry

end division_with_remainder_l919_91941


namespace range_of_z_l919_91975

theorem range_of_z (x y : ℝ) 
  (h1 : -4 ≤ x - y ∧ x - y ≤ -1)
  (h2 : -1 ≤ 4*x - y ∧ 4*x - y ≤ 5) :
  ∃ (z : ℝ), z = 9*x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end range_of_z_l919_91975


namespace area_is_nine_halves_l919_91924

/-- The line in the Cartesian coordinate system -/
def line (x y : ℝ) : Prop := x - y = 0

/-- The curve in the Cartesian coordinate system -/
def curve (x y : ℝ) : Prop := y = x^2 - 2*x

/-- The area enclosed by the line and the curve -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is equal to 9/2 -/
theorem area_is_nine_halves : enclosed_area = 9/2 := by sorry

end area_is_nine_halves_l919_91924


namespace previous_height_l919_91902

theorem previous_height (current_height : ℝ) (growth_rate : ℝ) : 
  current_height = 126 ∧ growth_rate = 0.05 → 
  current_height / (1 + growth_rate) = 120 := by
  sorry

end previous_height_l919_91902


namespace prop_2_prop_4_l919_91989

-- Define real numbers a, b, and c
variable (a b c : ℝ)

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition 2
theorem prop_2 : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition 4
theorem prop_4 : a < 3 → a < 5 := by sorry

end prop_2_prop_4_l919_91989


namespace customers_left_l919_91985

/-- Given a waiter with an initial number of customers and a number of remaining customers,
    prove that the number of customers who left is the difference between the initial and remaining customers. -/
theorem customers_left (initial remaining : ℕ) (h1 : initial = 21) (h2 : remaining = 12) :
  initial - remaining = 9 := by
  sorry

end customers_left_l919_91985


namespace friend_p_distance_at_meeting_l919_91952

-- Define the trail length
def trail_length : ℝ := 22

-- Define the speed ratio between Friend P and Friend Q
def speed_ratio : ℝ := 1.2

-- Theorem statement
theorem friend_p_distance_at_meeting :
  let v : ℝ := trail_length / (speed_ratio + 1)  -- Friend Q's speed
  let t : ℝ := trail_length / (v * (speed_ratio + 1))  -- Time to meet
  speed_ratio * v * t = 12 := by
  sorry

end friend_p_distance_at_meeting_l919_91952


namespace total_cards_is_690_l919_91943

/-- The number of get well cards Mariela received in the hospital -/
def cards_in_hospital : ℕ := 403

/-- The number of get well cards Mariela received at home -/
def cards_at_home : ℕ := 287

/-- The total number of get well cards Mariela received -/
def total_cards : ℕ := cards_in_hospital + cards_at_home

/-- Theorem stating that the total number of get well cards Mariela received is 690 -/
theorem total_cards_is_690 : total_cards = 690 := by
  sorry

end total_cards_is_690_l919_91943


namespace intersection_A_not_B_l919_91978

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / |x - 1| < 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 > 0}

-- Define the complement of B
def not_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_not_B : A ∩ not_B = {x | 2 < x ∧ x ≤ 4} := by sorry

end intersection_A_not_B_l919_91978


namespace student_task_assignment_l919_91991

/-- The number of ways to assign students to tasks under specific conditions -/
def assignment_count (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 1 * Nat.choose m 2 * (n - 1)^(n - 1) + Nat.choose k 2 * (n - 1)^(n - 1)

/-- Theorem stating the number of ways to assign 5 students to 4 tasks under given conditions -/
theorem student_task_assignment :
  assignment_count 4 4 3 = Nat.choose 3 1 * Nat.choose 4 2 * 3^3 + Nat.choose 3 2 * 3^3 :=
by sorry

end student_task_assignment_l919_91991


namespace smallest_angle_solution_l919_91904

theorem smallest_angle_solution (θ : Real) : 
  (θ > 0) → 
  (∀ φ, φ > 0 → φ < θ → Real.sin (10 * Real.pi / 180) ≠ Real.cos (40 * Real.pi / 180) - Real.cos (φ * Real.pi / 180)) →
  Real.sin (10 * Real.pi / 180) = Real.cos (40 * Real.pi / 180) - Real.cos (θ * Real.pi / 180) →
  θ = 30 :=
by sorry

end smallest_angle_solution_l919_91904


namespace bicycle_license_combinations_l919_91937

def license_letter : Nat := 2  -- B or C
def license_digits : Nat := 6
def free_digit_positions : Nat := license_digits - 1  -- All but the last digit
def digits_per_position : Nat := 10  -- 0 to 9
def last_digit : Nat := 1  -- Only 5 is allowed

theorem bicycle_license_combinations :
  license_letter * digits_per_position ^ free_digit_positions * last_digit = 200000 := by
  sorry

end bicycle_license_combinations_l919_91937


namespace figure_area_is_79_l919_91942

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℕ) (height : ℕ) : ℕ := width * height

/-- Represents the dimensions of the figure -/
structure FigureDimensions where
  leftWidth : ℕ
  leftHeight : ℕ
  middleWidth : ℕ
  middleHeight : ℕ
  rightWidth : ℕ
  rightHeight : ℕ

/-- Calculates the total area of the figure -/
def totalArea (d : FigureDimensions) : ℕ :=
  rectangleArea d.leftWidth d.leftHeight +
  rectangleArea d.middleWidth d.middleHeight +
  rectangleArea d.rightWidth d.rightHeight

/-- Theorem: The total area of the figure is 79 square units -/
theorem figure_area_is_79 (d : FigureDimensions) 
  (h1 : d.leftWidth = 6 ∧ d.leftHeight = 7)
  (h2 : d.middleWidth = 4 ∧ d.middleHeight = 3)
  (h3 : d.rightWidth = 5 ∧ d.rightHeight = 5) :
  totalArea d = 79 := by
  sorry

end figure_area_is_79_l919_91942


namespace cubic_equation_solution_l919_91920

theorem cubic_equation_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_solution : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = a ∨ x = -b ∨ x = c) :
  a = 1 ∧ b = -1 ∧ c = -1 := by
  sorry

end cubic_equation_solution_l919_91920


namespace edward_received_amount_l919_91982

def edward_problem (initial_amount spent_amount final_amount received_amount : ℝ) : Prop :=
  initial_amount = 14 ∧
  spent_amount = 17 ∧
  final_amount = 7 ∧
  initial_amount - spent_amount + received_amount = final_amount

theorem edward_received_amount :
  ∃ (received_amount : ℝ), edward_problem 14 17 7 received_amount ∧ received_amount = 10 := by
  sorry

end edward_received_amount_l919_91982


namespace slope_of_specific_midpoint_line_l919_91926

/-- The slope of the line connecting the midpoints of two line segments -/
def slope_of_midpoint_line (x1 y1 x2 y2 x3 y3 x4 y4 : ℚ) : ℚ :=
  let m1x := (x1 + x2) / 2
  let m1y := (y1 + y2) / 2
  let m2x := (x3 + x4) / 2
  let m2y := (y3 + y4) / 2
  (m2y - m1y) / (m2x - m1x)

/-- Theorem: The slope of the line connecting the midpoints of the given segments is -1 -/
theorem slope_of_specific_midpoint_line :
  slope_of_midpoint_line 3 4 7 8 6 2 9 5 = -1 := by sorry

end slope_of_specific_midpoint_line_l919_91926


namespace xy_equals_twelve_l919_91951

theorem xy_equals_twelve (x y : ℝ) 
  (h1 : x * (x + y) = x^2 + 12) 
  (h2 : x - y = 3) : 
  x * y = 12 := by
sorry

end xy_equals_twelve_l919_91951


namespace product_of_repeating_decimals_l919_91996

-- Define the repeating decimal 0.080808...
def repeating_08 : ℚ := 8 / 99

-- Define the repeating decimal 0.333333...
def repeating_3 : ℚ := 1 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_08 * repeating_3 = 8 / 297 := by
  sorry

end product_of_repeating_decimals_l919_91996


namespace smallest_three_digit_multiple_of_9_with_digit_sum_27_l919_91940

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → n ≤ m :=
by sorry

end smallest_three_digit_multiple_of_9_with_digit_sum_27_l919_91940


namespace andy_ant_position_l919_91960

/-- Represents a coordinate point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction (North, East, South, West) -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of Andy the Ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat

/-- Calculates the next direction after a left turn -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.East => Direction.North
  | Direction.South => Direction.East
  | Direction.West => Direction.South

/-- Calculates the movement distance for a given move number -/
def moveDistance (n : Nat) : Int :=
  2 * n + 1

/-- Performs a single move and updates the ant's state -/
def move (state : AntState) : AntState :=
  let dist := moveDistance state.moveCount
  let newPos := match state.direction with
    | Direction.North => { x := state.position.x, y := state.position.y + dist }
    | Direction.East => { x := state.position.x + dist, y := state.position.y }
    | Direction.South => { x := state.position.x, y := state.position.y - dist }
    | Direction.West => { x := state.position.x - dist, y := state.position.y }
  { position := newPos,
    direction := nextDirection state.direction,
    moveCount := state.moveCount + 1 }

/-- Performs n moves and returns the final state -/
def nMoves (n : Nat) (initialState : AntState) : AntState :=
  if n = 0 then initialState else nMoves (n - 1) (move initialState)

/-- The main theorem to prove -/
theorem andy_ant_position :
  let initialState : AntState := {
    position := { x := 10, y := -10 },
    direction := Direction.East,
    moveCount := 0
  }
  let finalState := nMoves 2022 initialState
  finalState.position = { x := 12, y := 4038 } := by sorry

end andy_ant_position_l919_91960


namespace tangent_line_equation_circle_radius_l919_91918

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*x + a = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 8*x - 15*y + 43 = 0

-- Define the dot product of vectors OA and OB
def dot_product_OA_OB (a : ℝ) : ℝ := -6

-- Theorem for part 1
theorem tangent_line_equation :
  ∀ x y : ℝ,
  circle_M (-8) x y →
  (tangent_line_1 x ∨ tangent_line_2 x y) →
  (x - (point_P.1))^2 + (y - (point_P.2))^2 = 
  (x - 1)^2 + y^2 :=
sorry

-- Theorem for part 2
theorem circle_radius :
  ∀ a : ℝ,
  dot_product_OA_OB a = -6 →
  ∃ r : ℝ, r^2 = 7 ∧
  ∀ x y : ℝ, circle_M a x y → (x - 1)^2 + y^2 = r^2 :=
sorry

end tangent_line_equation_circle_radius_l919_91918


namespace solution_implies_k_value_l919_91962

theorem solution_implies_k_value (k x y : ℚ) : 
  x = 3 ∧ y = 2 ∧ k * x + 3 * y = 1 → k = -5/3 := by
  sorry

end solution_implies_k_value_l919_91962


namespace jerrys_shelf_l919_91901

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The number of action figures added later -/
def added_figures : ℕ := 2

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 4

theorem jerrys_shelf :
  initial_figures + added_figures = num_books + difference := by sorry

end jerrys_shelf_l919_91901


namespace ladonnas_cans_correct_l919_91997

/-- The number of cans collected by LaDonna, given that:
    - The total number of cans collected is 85
    - Prikya collected twice as many cans as LaDonna
    - Yoki collected 10 cans
-/
def ladonnas_cans : ℕ := 25

/-- The total number of cans collected -/
def total_cans : ℕ := 85

/-- The number of cans collected by Yoki -/
def yokis_cans : ℕ := 10

theorem ladonnas_cans_correct :
  ladonnas_cans + 2 * ladonnas_cans + yokis_cans = total_cans :=
by sorry

end ladonnas_cans_correct_l919_91997


namespace solve_for_y_l919_91964

theorem solve_for_y (x y : ℚ) (h1 : x = 102) (h2 : x^3*y - 3*x^2*y + 3*x*y = 106200) : 
  y = 10/97 := by
sorry

end solve_for_y_l919_91964


namespace final_books_count_l919_91925

/-- Represents the daily transactions in the library --/
structure DailyTransactions where
  checkouts : ℕ
  returns : ℕ
  renewals : ℕ := 0
  newBooks : ℕ := 0
  damaged : ℕ := 0
  misplaced : ℕ := 0

/-- Calculates the number of available books at the end of the week --/
def availableBooksAtEndOfWeek (initialBooks : ℕ) (monday tuesday wednesday thursday friday : DailyTransactions) : ℕ :=
  let mondayEnd := initialBooks - monday.checkouts + monday.returns
  let tuesdayEnd := mondayEnd - tuesday.checkouts + tuesday.returns + tuesday.newBooks
  let wednesdayEnd := tuesdayEnd - wednesday.checkouts + wednesday.returns - wednesday.damaged
  let thursdayEnd := wednesdayEnd - thursday.checkouts + thursday.returns
  let fridayEnd := thursdayEnd - friday.checkouts + friday.returns
  fridayEnd

/-- Theorem: Given the initial number of books and daily transactions, the final number of books available for checkout at the end of the week is 76 --/
theorem final_books_count (initialBooks : ℕ) (monday tuesday wednesday thursday friday : DailyTransactions) :
  initialBooks = 98 →
  monday = { checkouts := 43, returns := 23, renewals := 5 } →
  tuesday = { checkouts := 28, returns := 0, newBooks := 35 } →
  wednesday = { checkouts := 52, returns := 40, damaged := 3 } →
  thursday = { checkouts := 37, returns := 22 } →
  friday = { checkouts := 29, returns := 50, misplaced := 4 } →
  availableBooksAtEndOfWeek initialBooks monday tuesday wednesday thursday friday = 76 :=
by
  sorry

end final_books_count_l919_91925


namespace chairs_distribution_l919_91998

theorem chairs_distribution (total_chairs : Nat) (h1 : total_chairs = 123) :
  ∃! (num_boys chairs_per_boy : Nat),
    num_boys * chairs_per_boy = total_chairs ∧
    num_boys > 0 ∧
    chairs_per_boy > 0 ∧
    num_boys = 41 ∧
    chairs_per_boy = 3 := by
  sorry

end chairs_distribution_l919_91998


namespace smallest_n_for_quadruplets_l919_91971

/-- The number of ordered quadruplets (a, b, c, d) with given gcd and lcm -/
def count_quadruplets (gcd lcm : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n satisfying the conditions -/
theorem smallest_n_for_quadruplets :
  ∃ n : ℕ, n > 0 ∧ 
  count_quadruplets 72 n = 72000 ∧
  (∀ m : ℕ, m > 0 → m < n → count_quadruplets 72 m ≠ 72000) ∧
  n = 36288 := by
  sorry

end smallest_n_for_quadruplets_l919_91971


namespace cos_n_equals_sin_712_l919_91993

theorem cos_n_equals_sin_712 (n : ℤ) :
  -90 ≤ n ∧ n ≤ 90 ∧ Real.cos (n * π / 180) = Real.sin (712 * π / 180) → n = -82 := by
  sorry

end cos_n_equals_sin_712_l919_91993


namespace rectangle_triangles_l919_91935

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  sorry

/-- Theorem: A rectangle divided into 6 horizontal and 3 vertical parts contains 48 triangles -/
theorem rectangle_triangles :
  let rect : DividedRectangle := { horizontal_divisions := 6, vertical_divisions := 3 }
  count_triangles rect = 48 := by
  sorry

end rectangle_triangles_l919_91935


namespace solution_to_system_of_equations_l919_91956

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 3) ∧ (x = 47/9) ∧ (y = 17/3) := by
  sorry

end solution_to_system_of_equations_l919_91956


namespace example_is_quadratic_l919_91999

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - x + 1 = 0 is a quadratic equation in terms of x -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - x + 1) := by
  sorry

end example_is_quadratic_l919_91999


namespace swap_digits_result_l919_91921

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens < 10 ∧ units < 10

/-- Swaps the digits of a two-digit number -/
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  is_valid := by
    simp [n.is_valid]

/-- Theorem stating the result of swapping digits in the given conditions -/
theorem swap_digits_result (n : TwoDigitNumber) (h_sum : n.tens + n.units = 13) :
  ∃ a : Nat, n.units = a ∧ (swap_digits n).tens * 10 + (swap_digits n).units = 9 * a + 13 := by
  sorry

end swap_digits_result_l919_91921


namespace inequality_solution_l919_91992

theorem inequality_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x < -6 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 74 := by
sorry

end inequality_solution_l919_91992


namespace abs_inequality_equivalence_l919_91900

theorem abs_inequality_equivalence (x : ℝ) : 
  |((x + 4) / 2)| < 3 ↔ -10 < x ∧ x < 2 := by
sorry

end abs_inequality_equivalence_l919_91900


namespace contrapositive_example_l919_91919

theorem contrapositive_example (a b : ℝ) :
  (¬(a + 1 > b) → ¬(a > b)) ↔ ((a + 1 ≤ b) → (a ≤ b)) := by sorry

end contrapositive_example_l919_91919


namespace inequality_solution_l919_91947

theorem inequality_solution (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end inequality_solution_l919_91947


namespace circle_tangent_and_bisecting_point_l919_91911

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 4 * Real.sqrt 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the center of circle C
def center_C : ℝ × ℝ := (0, 0)

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Define point N
def point_N : ℝ × ℝ := (8, 0)

-- Theorem statement
theorem circle_tangent_and_bisecting_point :
  ∃ (N : ℝ × ℝ), N = point_N ∧
  (∀ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 →
    ∃ (k : ℝ), 
      A.2 = k * (A.1 - point_M.1) ∧
      B.2 = k * (B.1 - point_M.1) →
      (A.2 / (A.1 - N.1)) + (B.2 / (B.1 - N.1)) = 0) :=
sorry

end circle_tangent_and_bisecting_point_l919_91911


namespace kate_needs_57_more_l919_91979

/-- The amount of additional money Kate needs to buy all items -/
def additional_money_needed (pen_price notebook_price art_set_price : ℚ)
  (pen_money_ratio : ℚ) (notebook_discount : ℚ) (art_set_money : ℚ) (art_set_discount : ℚ) : ℚ :=
  (pen_price - pen_price * pen_money_ratio) +
  (notebook_price * (1 - notebook_discount)) +
  (art_set_price * (1 - art_set_discount) - art_set_money)

/-- Theorem stating that Kate needs $57 more to buy all items -/
theorem kate_needs_57_more :
  additional_money_needed 30 20 50 (1/3) 0.15 10 0.4 = 57 := by
  sorry

end kate_needs_57_more_l919_91979


namespace max_figures_per_shelf_l919_91908

def initial_shelves : Nat := 3
def shelf1_figures : Nat := 9
def shelf2_figures : Nat := 14
def shelf3_figures : Nat := 7
def additional_shelves : Nat := 2
def new_shelf_max : Nat := 11

def total_figures : Nat := shelf1_figures + shelf2_figures + shelf3_figures
def total_shelves : Nat := initial_shelves + additional_shelves

theorem max_figures_per_shelf :
  ∃ (x : Nat), 
    x ≤ new_shelf_max ∧ 
    x * total_shelves = total_figures ∧
    ∀ (y : Nat), y ≤ new_shelf_max ∧ y * total_shelves = total_figures → y ≤ x :=
by sorry

end max_figures_per_shelf_l919_91908


namespace trig_identity_simplification_l919_91912

theorem trig_identity_simplification (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end trig_identity_simplification_l919_91912


namespace math_and_lang_not_science_l919_91969

def students : ℕ := 120
def math_students : ℕ := 80
def lang_students : ℕ := 70
def science_students : ℕ := 50
def all_three_students : ℕ := 20

theorem math_and_lang_not_science :
  ∃ (math_and_lang math_and_science lang_and_science : ℕ),
    math_and_lang + math_and_science + lang_and_science = 
      math_students + lang_students + science_students - students + all_three_students ∧
    math_and_lang - all_three_students = 30 := by
  sorry

end math_and_lang_not_science_l919_91969


namespace jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l919_91946

theorem jewelry_pattern_purple_beads : ℕ → Prop :=
  fun purple_beads =>
    let green_beads : ℕ := 3
    let red_beads : ℕ := 2 * green_beads
    let pattern_total : ℕ := green_beads + purple_beads + red_beads
    let bracelet_repeats : ℕ := 3
    let necklace_repeats : ℕ := 5
    let bracelet_beads : ℕ := bracelet_repeats * pattern_total
    let necklace_beads : ℕ := necklace_repeats * pattern_total
    let total_beads : ℕ := 742
    let num_bracelets : ℕ := 1
    let num_necklaces : ℕ := 10
    num_bracelets * bracelet_beads + num_necklaces * necklace_beads = total_beads →
    purple_beads = 5

-- Proof
theorem jewelry_pattern_purple_beads_proof : jewelry_pattern_purple_beads 5 := by
  sorry

end jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l919_91946


namespace x_value_on_line_k_l919_91932

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem x_value_on_line_k (x y : ℝ) :
  line_k x 6 → 
  line_k 10 y → 
  x * y = 60 →
  x = 12 := by
  sorry

end x_value_on_line_k_l919_91932


namespace rod_lengths_at_zero_celsius_l919_91965

/-- Theorem: Rod Lengths at 0°C
Given:
- Total length at 0°C is 1 m
- Total length at 100°C is 1.0024 m
- Coefficient of linear expansion for steel is 0.000011
- Coefficient of linear expansion for zinc is 0.000031

Prove:
- Length of steel rod at 0°C is 0.35 m
- Length of zinc rod at 0°C is 0.65 m
-/
theorem rod_lengths_at_zero_celsius 
  (total_length_zero : Real) 
  (total_length_hundred : Real)
  (steel_expansion : Real)
  (zinc_expansion : Real)
  (h1 : total_length_zero = 1)
  (h2 : total_length_hundred = 1.0024)
  (h3 : steel_expansion = 0.000011)
  (h4 : zinc_expansion = 0.000031) :
  ∃ (steel_length zinc_length : Real),
    steel_length = 0.35 ∧ 
    zinc_length = 0.65 ∧
    steel_length + zinc_length = total_length_zero ∧
    steel_length * (1 + 100 * steel_expansion) + 
    zinc_length * (1 + 100 * zinc_expansion) = total_length_hundred :=
by sorry

end rod_lengths_at_zero_celsius_l919_91965


namespace machine_parts_processed_l919_91958

/-- Given two machines processing parts for 'a' hours, where the second machine
    processed 'n' fewer parts and takes 'b' minutes longer per part than the first,
    prove the number of parts processed by each machine. -/
theorem machine_parts_processed
  (a b n : ℝ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  let x := (b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  let y := (-b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  (∀ t, 0 < t ∧ t < a → (t / x = t / (x - n) - b / 60)) ∧
  x > 0 ∧ y > 0 ∧ x - y = n :=
sorry


end machine_parts_processed_l919_91958


namespace concatenated_square_exists_l919_91928

theorem concatenated_square_exists : ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ), 
  (10^n + 1) * A = B^2 ∧ A > 0 ∧ A < 10^n := by
  sorry

end concatenated_square_exists_l919_91928


namespace batsman_average_excluding_extremes_l919_91988

def batting_average : ℝ := 60
def num_innings : ℕ := 46
def highest_score : ℕ := 194
def score_difference : ℕ := 180

theorem batsman_average_excluding_extremes :
  let total_runs : ℝ := batting_average * num_innings
  let lowest_score : ℕ := highest_score - score_difference
  let runs_excluding_extremes : ℝ := total_runs - highest_score - lowest_score
  let innings_excluding_extremes : ℕ := num_innings - 2
  (runs_excluding_extremes / innings_excluding_extremes : ℝ) = 58 := by sorry

end batsman_average_excluding_extremes_l919_91988


namespace area_of_region_l919_91931

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 23 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 6*x - 4*y - 10 = 0) := by
  sorry

end area_of_region_l919_91931


namespace union_necessary_not_sufficient_l919_91917

theorem union_necessary_not_sufficient (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) := by
  sorry

end union_necessary_not_sufficient_l919_91917


namespace special_triangle_side_length_l919_91929

/-- A triangle with special median properties -/
structure SpecialTriangle where
  /-- The length of side EF -/
  EF : ℝ
  /-- The length of side DF -/
  DF : ℝ
  /-- The median from D is perpendicular to the median from E -/
  medians_perpendicular : Bool

/-- Theorem: In a special triangle with EF = 10, DF = 8, and perpendicular medians, DE = 18 -/
theorem special_triangle_side_length (t : SpecialTriangle) 
  (h1 : t.EF = 10) 
  (h2 : t.DF = 8) 
  (h3 : t.medians_perpendicular = true) : 
  ∃ DE : ℝ, DE = 18 := by
  sorry

end special_triangle_side_length_l919_91929


namespace six_and_neg_six_are_opposite_l919_91972

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_neg_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end six_and_neg_six_are_opposite_l919_91972


namespace park_bushes_count_l919_91945

def park_bushes (initial_orchids initial_roses initial_tulips added_orchids removed_roses : ℕ) : ℕ × ℕ × ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let final_roses := initial_roses - removed_roses
  let final_tulips := initial_tulips * 2
  (final_orchids, final_roses, final_tulips)

theorem park_bushes_count : park_bushes 2 5 3 4 1 = (6, 4, 6) := by sorry

end park_bushes_count_l919_91945


namespace max_distance_to_origin_is_three_l919_91903

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- Calculates the maximum distance from any point on a circle to the origin in polar coordinates -/
def maxDistanceToOrigin (c : PolarCircle) : ℝ :=
  c.center.r + c.radius

theorem max_distance_to_origin_is_three :
  let circle := PolarCircle.mk (PolarPoint.mk 2 (π / 6)) 1
  maxDistanceToOrigin circle = 3 := by
  sorry

end max_distance_to_origin_is_three_l919_91903


namespace fraction_zero_implies_x_negative_two_l919_91905

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x ≠ 2) → ((|x| - 2) / (x - 2) = 0) → x = -2 := by
  sorry

end fraction_zero_implies_x_negative_two_l919_91905


namespace sarah_homework_problem_l919_91949

/-- The number of homework problems Sarah had initially -/
def total_problems (finished_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished_problems + remaining_pages * problems_per_page

/-- Theorem stating that Sarah had 60 homework problems initially -/
theorem sarah_homework_problem :
  total_problems 20 5 8 = 60 := by
  sorry

end sarah_homework_problem_l919_91949


namespace triangle_properties_l919_91907

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) 
  (h2 : t.a = Real.sqrt 3)
  (h3 : Real.cos t.C = Real.sqrt 3 / 3) :
  (t.A = π / 3) ∧ (t.c = 2 * Real.sqrt 6 / 3) := by
  sorry


end triangle_properties_l919_91907


namespace cost_of_bananas_l919_91994

/-- The cost of bananas given the total cost of groceries and the costs of other items -/
theorem cost_of_bananas 
  (total_cost : ℕ) 
  (bread_cost milk_cost apple_cost : ℕ) 
  (h1 : total_cost = 42)
  (h2 : bread_cost = 9)
  (h3 : milk_cost = 7)
  (h4 : apple_cost = 14) :
  total_cost - (bread_cost + milk_cost + apple_cost) = 12 := by
sorry

end cost_of_bananas_l919_91994


namespace power_inequality_l919_91933

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ a^b * b^a :=
by sorry

end power_inequality_l919_91933


namespace prime_square_difference_divisibility_l919_91990

theorem prime_square_difference_divisibility (p : ℕ) (hp : Prime p) :
  ∃ (n m : ℕ), n ≠ 0 ∧ m ≠ 0 ∧ n ≠ m ∧
  p - n^2 ≠ 1 ∧
  p - n^2 ≠ p - m^2 ∧
  (p - n^2) ∣ (p - m^2) :=
sorry

end prime_square_difference_divisibility_l919_91990


namespace solution_theorem_l919_91936

/-- A function satisfying the given condition for all non-zero real numbers -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 4 * x

theorem solution_theorem :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    ∀ x : ℝ, x ≠ 0 →
      (f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end solution_theorem_l919_91936


namespace gross_revenue_increase_l919_91983

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 28) :=
by sorry

end gross_revenue_increase_l919_91983


namespace polygon_sides_l919_91953

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (180 * (n - 2) : ℝ) / 360 = 5 / 2 → 
  n = 7 := by
sorry

end polygon_sides_l919_91953


namespace oldest_babysat_age_jane_l919_91981

/-- Represents a person with their current age and baby-sitting history. -/
structure Person where
  currentAge : ℕ
  babySittingStartAge : ℕ
  babySittingEndAge : ℕ

/-- Calculates the maximum age of a child that a person could have babysat. -/
def maxBabysatChildAge (p : Person) : ℕ :=
  p.babySittingEndAge / 2

/-- Calculates the current age of the oldest person that could have been babysat. -/
def oldestBabysatPersonCurrentAge (p : Person) : ℕ :=
  maxBabysatChildAge p + (p.currentAge - p.babySittingEndAge)

/-- Theorem stating the age of the oldest person Jane could have babysat. -/
theorem oldest_babysat_age_jane :
  let jane : Person := {
    currentAge := 32,
    babySittingStartAge := 18,
    babySittingEndAge := 20
  }
  oldestBabysatPersonCurrentAge jane = 22 := by
  sorry


end oldest_babysat_age_jane_l919_91981


namespace solve_for_q_l919_91968

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/72)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 140 := by
  sorry

end solve_for_q_l919_91968


namespace q_round_time_l919_91948

/-- The time it takes for two runners to meet at the starting point again -/
def meeting_time : ℕ := 2772

/-- The time it takes for runner P to complete one round -/
def p_round_time : ℕ := 252

/-- Theorem stating that under given conditions, runner Q takes 2772 seconds to complete a round -/
theorem q_round_time : ∀ (q_round_time : ℕ), 
  (meeting_time % p_round_time = 0) →
  (meeting_time % q_round_time = 0) →
  (meeting_time / p_round_time ≠ meeting_time / q_round_time) →
  q_round_time = meeting_time :=
by sorry

end q_round_time_l919_91948


namespace certain_number_proof_l919_91966

theorem certain_number_proof (h : 2994 / 14.5 = 177) : ∃ x : ℝ, x / 1.45 = 17.7 ∧ x = 25.665 := by
  sorry

end certain_number_proof_l919_91966


namespace remaining_perimeter_is_56_l919_91995

/-- Represents the dimensions of a rectangular piece of paper. -/
structure Paper where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of the remaining rectangle after cutting out the largest square. -/
def remainingPerimeter (p : Paper) : ℝ :=
  2 * (p.width + (p.length - p.width))

/-- Theorem stating that for a 28 cm by 15 cm paper, the perimeter of the remaining rectangle is 56 cm. -/
theorem remaining_perimeter_is_56 :
  let p : Paper := { length := 28, width := 15 }
  remainingPerimeter p = 56 := by sorry

end remaining_perimeter_is_56_l919_91995


namespace incorrect_quotient_calculation_l919_91915

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 40) :
  dividend / incorrect_divisor = 70 := by
  sorry

end incorrect_quotient_calculation_l919_91915


namespace infinite_indices_inequality_l919_91959

def FastGrowingSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ C : ℝ, ∃ N : ℕ, ∀ k > N, (a k : ℝ) > C * k)

theorem infinite_indices_inequality
  (a : ℕ → ℕ)
  (h : FastGrowingSequence a) :
  ∀ M : ℕ, ∃ k > M, 2 * (a k) < (a (k - 1)) + (a (k + 1)) :=
sorry

end infinite_indices_inequality_l919_91959


namespace apple_pear_basket_weights_l919_91963

/-- Given the conditions of the apple and pear basket problem, prove the weights of individual baskets. -/
theorem apple_pear_basket_weights :
  ∀ (apple_weight pear_weight : ℕ),
  -- Total weight of all baskets is 692 kg
  12 * apple_weight + 14 * pear_weight = 692 →
  -- Weight of pear basket is 10 kg less than apple basket
  pear_weight = apple_weight - 10 →
  -- Prove that apple_weight is 32 kg and pear_weight is 22 kg
  apple_weight = 32 ∧ pear_weight = 22 := by
  sorry

end apple_pear_basket_weights_l919_91963


namespace absolute_value_equation_solution_l919_91930

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |n + 6| = 2 - n :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l919_91930


namespace inequality_proof_l919_91984

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c^2 < c*d :=
sorry

end inequality_proof_l919_91984


namespace modular_inverse_of_5_mod_23_l919_91944

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 ∧ x = 14 := by
sorry

end modular_inverse_of_5_mod_23_l919_91944


namespace price_change_l919_91927

/-- Theorem: Price change after 50% decrease and 60% increase --/
theorem price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 - 0.5) * (1 + 0.6) = P * 0.8 := by
  sorry

#check price_change

end price_change_l919_91927


namespace seeds_solution_l919_91909

def seeds_problem (wednesday thursday friday : ℕ) : Prop :=
  wednesday = 5 * thursday ∧
  wednesday + thursday = 156 ∧
  friday = 4

theorem seeds_solution :
  ∃ (wednesday thursday friday : ℕ),
    seeds_problem wednesday thursday friday ∧
    wednesday = 130 ∧
    thursday = 26 ∧
    friday = 4 ∧
    wednesday + thursday + friday = 160 := by
  sorry

end seeds_solution_l919_91909


namespace cubic_roots_sum_squares_l919_91934

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := by
sorry

end cubic_roots_sum_squares_l919_91934


namespace field_division_fraction_l919_91950

theorem field_division_fraction (total_area smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area = 225 →
  larger_area = total_area - smaller_area →
  (larger_area - smaller_area) / ((smaller_area + larger_area) / 2) = 1 / 5 := by
  sorry

end field_division_fraction_l919_91950


namespace blackboard_divisibility_l919_91910

/-- Represents the transformation process on the blackboard -/
def transform (n : ℕ) : ℕ := sorry

/-- The number on the blackboard after n minutes -/
def blackboard_number (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | n+1 => transform (blackboard_number n)

/-- The final number N on the blackboard -/
def N : ℕ := blackboard_number (sorry : ℕ)

theorem blackboard_divisibility :
  (9 ∣ N) → (99 ∣ N) := by sorry

end blackboard_divisibility_l919_91910
