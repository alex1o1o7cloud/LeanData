import Mathlib

namespace NUMINAMATH_CALUDE_max_first_day_volume_l3324_332473

def container_volumes : List Nat := [9, 13, 17, 19, 20, 38]

def is_valid_first_day_selection (selection : List Nat) : Prop :=
  selection.length = 3 ∧ selection.all (λ x => x ∈ container_volumes)

def is_valid_second_day_selection (first_day : List Nat) (second_day : List Nat) : Prop :=
  second_day.length = 2 ∧ 
  second_day.all (λ x => x ∈ container_volumes) ∧
  (∀ x ∈ second_day, x ∉ first_day)

def satisfies_volume_constraint (first_day : List Nat) (second_day : List Nat) : Prop :=
  first_day.sum = 2 * second_day.sum

theorem max_first_day_volume : 
  ∃ (first_day second_day : List Nat),
    is_valid_first_day_selection first_day ∧
    is_valid_second_day_selection first_day second_day ∧
    satisfies_volume_constraint first_day second_day ∧
    first_day.sum = 66 ∧
    (∀ (other_first_day : List Nat),
      is_valid_first_day_selection other_first_day →
      other_first_day.sum ≤ 66) :=
by sorry

end NUMINAMATH_CALUDE_max_first_day_volume_l3324_332473


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l3324_332423

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has exactly 3 digits -/
def isThreeDigit (n : ℕ) : Prop := sorry

theorem no_even_three_digit_sum_27 :
  ¬∃ n : ℕ, isThreeDigit n ∧ digitSum n = 27 ∧ isEven n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l3324_332423


namespace NUMINAMATH_CALUDE_jake_has_one_more_balloon_l3324_332422

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem jake_has_one_more_balloon : balloon_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_one_more_balloon_l3324_332422


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l3324_332448

/-- For positive integers a and b, where b > 1, if a^b is the greatest possible value less than 500, then a + b = 24 -/
theorem greatest_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) 
  (h_greatest : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → a^b ≥ x^y) 
  (h_less_500 : a^b < 500) : a + b = 24 := by
  sorry


end NUMINAMATH_CALUDE_greatest_power_under_500_l3324_332448


namespace NUMINAMATH_CALUDE_zacks_marbles_l3324_332470

theorem zacks_marbles (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 5) → 
  (n = 3 * 20 + 5) → 
  n = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l3324_332470


namespace NUMINAMATH_CALUDE_abs_negative_2023_l3324_332493

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l3324_332493


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3324_332419

theorem diophantine_equation_solutions :
  ∀ n k m : ℕ, 5^n - 3^k = m^2 →
    ((n = 0 ∧ k = 0 ∧ m = 0) ∨ (n = 2 ∧ k = 2 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3324_332419


namespace NUMINAMATH_CALUDE_existence_of_small_triangle_l3324_332488

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of points -/
def PointSet := Set Point

/-- Definition of a square with side length 20 -/
def is_square_20 (A B C D : Point) : Prop := sorry

/-- Check if three points are collinear -/
def are_collinear (P Q R : Point) : Prop := sorry

/-- Check if a point is inside a square -/
def is_inside_square (P : Point) (A B C D : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem existence_of_small_triangle 
  (A B C D : Point) 
  (T : Fin 2000 → Point)
  (h_square : is_square_20 A B C D)
  (h_inside : ∀ i, is_inside_square (T i) A B C D)
  (h_not_collinear : ∀ P Q R, P ≠ Q → Q ≠ R → P ≠ R → 
    P ∈ {A, B, C, D} ∪ (Set.range T) → 
    Q ∈ {A, B, C, D} ∪ (Set.range T) → 
    R ∈ {A, B, C, D} ∪ (Set.range T) → 
    ¬(are_collinear P Q R)) :
  ∃ P Q R, P ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           Q ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           R ∈ {A, B, C, D} ∪ (Set.range T) ∧ 
           triangle_area P Q R < 1/10 :=
sorry

end NUMINAMATH_CALUDE_existence_of_small_triangle_l3324_332488


namespace NUMINAMATH_CALUDE_circle_tangent_and_bisecting_point_l3324_332402

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

end NUMINAMATH_CALUDE_circle_tangent_and_bisecting_point_l3324_332402


namespace NUMINAMATH_CALUDE_adam_father_deposit_l3324_332452

/-- Calculates the total amount after a given period, including initial deposit and interest --/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given the specified conditions, the total amount after 2.5 years is $2400 --/
theorem adam_father_deposit :
  let initialDeposit : ℝ := 2000
  let interestRate : ℝ := 0.08
  let years : ℝ := 2.5
  totalAmount initialDeposit interestRate years = 2400 := by
  sorry

end NUMINAMATH_CALUDE_adam_father_deposit_l3324_332452


namespace NUMINAMATH_CALUDE_other_color_students_l3324_332476

theorem other_color_students (total : ℕ) (blue_percent red_percent green_percent : ℚ) : 
  total = 800 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  green_percent = 15/100 →
  (total : ℚ) * (1 - (blue_percent + red_percent + green_percent)) = 136 := by
  sorry

end NUMINAMATH_CALUDE_other_color_students_l3324_332476


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l3324_332451

/-- Given a bamboo pole and a building with their respective shadows, 
    calculate the height of the building using similar triangles. -/
theorem building_height_from_shadows 
  (bamboo_height : ℝ) 
  (bamboo_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_bamboo_height : bamboo_height = 1.8)
  (h_bamboo_shadow : bamboo_shadow = 3)
  (h_building_shadow : building_shadow = 35)
  : (bamboo_height / bamboo_shadow) * building_shadow = 21 := by
  sorry


end NUMINAMATH_CALUDE_building_height_from_shadows_l3324_332451


namespace NUMINAMATH_CALUDE_division_with_remainder_l3324_332441

theorem division_with_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 76 → 
  quotient = 4 → 
  divisor = 17 → 
  dividend = divisor * quotient + remainder → 
  remainder = 8 := by
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l3324_332441


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l3324_332426

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

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l3324_332426


namespace NUMINAMATH_CALUDE_balanced_quadruple_inequality_l3324_332468

/-- A quadruple of real numbers is balanced if the sum of its elements
    equals the sum of their squares. -/
def IsBalanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

/-- For any positive real number x greater than or equal to 3/2,
    the product (x - a)(x - b)(x - c)(x - d) is non-negative
    for all balanced quadruples (a, b, c, d). -/
theorem balanced_quadruple_inequality (x : ℝ) (hx : x > 0) (hx_ge : x ≥ 3/2) :
  ∀ a b c d : ℝ, IsBalanced a b c d →
  (x - a) * (x - b) * (x - c) * (x - d) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_balanced_quadruple_inequality_l3324_332468


namespace NUMINAMATH_CALUDE_min_distance_MN_l3324_332462

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define a line tangent to the unit circle
def tangent_line (P A B : ℝ × ℝ) : Prop :=
  unit_circle A.1 A.2 ∧ unit_circle B.1 B.2 ∧
  ∃ (t : ℝ), (1 - t) • A + t • B = P

-- Define the intersection points M and N
def intersection_points (A B : ℝ × ℝ) (M N : ℝ × ℝ) : Prop :=
  M.2 = 0 ∧ N.1 = 0 ∧ ∃ (t s : ℝ), (1 - t) • A + t • B = M ∧ (1 - s) • A + s • B = N

-- State the theorem
theorem min_distance_MN (P A B M N : ℝ × ℝ) :
  point_on_ellipse P →
  tangent_line P A B →
  intersection_points A B M N →
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ 
    ∀ (P' A' B' M' N' : ℝ × ℝ), 
      point_on_ellipse P' →
      tangent_line P' A' B' →
      intersection_points A' B' M' N' →
      Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l3324_332462


namespace NUMINAMATH_CALUDE_arcade_time_calculation_l3324_332492

/-- The number of hours spent at the arcade given the rate and total spend -/
def arcade_time (rate : ℚ) (interval : ℚ) (total_spend : ℚ) : ℚ :=
  (total_spend / rate * interval) / 60

/-- Theorem stating that given a rate of $0.50 per 6 minutes and a total spend of $15, 
    the time spent at the arcade is 3 hours -/
theorem arcade_time_calculation :
  arcade_time (1/2) 6 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arcade_time_calculation_l3324_332492


namespace NUMINAMATH_CALUDE_interest_groups_intersection_difference_l3324_332457

theorem interest_groups_intersection_difference (total : ℕ) (math : ℕ) (english : ℕ)
  (h_total : total = 200)
  (h_math : math = 80)
  (h_english : english = 155) :
  (min math english) - (math + english - total) = 45 :=
sorry

end NUMINAMATH_CALUDE_interest_groups_intersection_difference_l3324_332457


namespace NUMINAMATH_CALUDE_range_of_z_l3324_332429

theorem range_of_z (x y : ℝ) 
  (h1 : -4 ≤ x - y ∧ x - y ≤ -1)
  (h2 : -1 ≤ 4*x - y ∧ 4*x - y ≤ 5) :
  ∃ (z : ℝ), z = 9*x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_range_of_z_l3324_332429


namespace NUMINAMATH_CALUDE_edward_received_amount_l3324_332416

def edward_problem (initial_amount spent_amount final_amount received_amount : ℝ) : Prop :=
  initial_amount = 14 ∧
  spent_amount = 17 ∧
  final_amount = 7 ∧
  initial_amount - spent_amount + received_amount = final_amount

theorem edward_received_amount :
  ∃ (received_amount : ℝ), edward_problem 14 17 7 received_amount ∧ received_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_edward_received_amount_l3324_332416


namespace NUMINAMATH_CALUDE_inequality_proof_l3324_332418

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c^2 < c*d :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3324_332418


namespace NUMINAMATH_CALUDE_mean_of_fractions_l3324_332483

theorem mean_of_fractions (a b : ℚ) (ha : a = 2/3) (hb : b = 4/9) :
  (a + b) / 2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_fractions_l3324_332483


namespace NUMINAMATH_CALUDE_prime_square_difference_divisibility_l3324_332415

theorem prime_square_difference_divisibility (p : ℕ) (hp : Prime p) :
  ∃ (n m : ℕ), n ≠ 0 ∧ m ≠ 0 ∧ n ≠ m ∧
  p - n^2 ≠ 1 ∧
  p - n^2 ≠ p - m^2 ∧
  (p - n^2) ∣ (p - m^2) :=
sorry

end NUMINAMATH_CALUDE_prime_square_difference_divisibility_l3324_332415


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3324_332487

/-- Given a mixture of milk and water, prove that the initial volume is 60 litres -/
theorem initial_mixture_volume
  (initial_ratio : ℚ) -- Initial ratio of milk to water
  (final_ratio : ℚ) -- Final ratio of milk to water
  (added_water : ℚ) -- Amount of water added to achieve final ratio
  (h1 : initial_ratio = 2 / 1) -- Initial ratio is 2:1
  (h2 : final_ratio = 1 / 2) -- Final ratio is 1:2
  (h3 : added_water = 60) -- 60 litres of water is added
  : ℚ :=
by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l3324_332487


namespace NUMINAMATH_CALUDE_solve_equations_l3324_332479

theorem solve_equations :
  (∃ x : ℝ, 1 - 3 * (1 - x) = 2 * x ∧ x = 2) ∧
  (∃ x : ℝ, (3 * x + 1) / 2 - (4 * x - 2) / 5 = 1 ∧ x = 1 / 7) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3324_332479


namespace NUMINAMATH_CALUDE_chairs_distribution_l3324_332406

theorem chairs_distribution (total_chairs : Nat) (h1 : total_chairs = 123) :
  ∃! (num_boys chairs_per_boy : Nat),
    num_boys * chairs_per_boy = total_chairs ∧
    num_boys > 0 ∧
    chairs_per_boy > 0 ∧
    num_boys = 41 ∧
    chairs_per_boy = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_distribution_l3324_332406


namespace NUMINAMATH_CALUDE_babylon_sphere_properties_l3324_332436

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

end NUMINAMATH_CALUDE_babylon_sphere_properties_l3324_332436


namespace NUMINAMATH_CALUDE_dollar_three_minus_four_l3324_332472

-- Define the custom operation $
def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y + x

-- Theorem statement
theorem dollar_three_minus_four : dollar 3 (-4) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_minus_four_l3324_332472


namespace NUMINAMATH_CALUDE_andy_ant_position_l3324_332404

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

end NUMINAMATH_CALUDE_andy_ant_position_l3324_332404


namespace NUMINAMATH_CALUDE_friend_p_distance_at_meeting_l3324_332440

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

end NUMINAMATH_CALUDE_friend_p_distance_at_meeting_l3324_332440


namespace NUMINAMATH_CALUDE_three_times_value_interval_examples_l3324_332430

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

end NUMINAMATH_CALUDE_three_times_value_interval_examples_l3324_332430


namespace NUMINAMATH_CALUDE_unique_determination_by_digit_sums_l3324_332450

/-- Given a natural number, compute the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number N, generate a sequence of digit sums for consecutive numbers starting from N+1 -/
def digit_sum_sequence (N : ℕ) (length : ℕ) : List ℕ := sorry

/-- Theorem: For any natural number N, there exists a finite sequence of digit sums that uniquely determines N -/
theorem unique_determination_by_digit_sums (N : ℕ) : 
  ∃ (length : ℕ), ∀ (M : ℕ), M ≠ N → 
    digit_sum_sequence N length ≠ digit_sum_sequence M length := by
  sorry

#check unique_determination_by_digit_sums

end NUMINAMATH_CALUDE_unique_determination_by_digit_sums_l3324_332450


namespace NUMINAMATH_CALUDE_inequality_solution_l3324_332434

theorem inequality_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x < -6 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 74 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3324_332434


namespace NUMINAMATH_CALUDE_complement_of_M_l3324_332438

def M : Set ℝ := {a : ℝ | a^2 - 2*a > 0}

theorem complement_of_M : 
  {a : ℝ | a ∉ M} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3324_332438


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l3324_332414

-- Define real numbers a, b, and c
variable (a b c : ℝ)

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition 2
theorem prop_2 : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition 4
theorem prop_4 : a < 3 → a < 5 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l3324_332414


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_6_l3324_332461

theorem sin_2x_minus_pi_6 (x : ℝ) (h : Real.cos (x + π / 6) + Real.sin (2 * π / 3 + x) = 1 / 2) :
  Real.sin (2 * x - π / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_6_l3324_332461


namespace NUMINAMATH_CALUDE_squirrel_problem_l3324_332469

/-- Theorem: Given the conditions of the squirrel problem, prove the original number of squirrels on each tree. -/
theorem squirrel_problem (s b j : ℕ) : 
  s + b + j = 34 ∧ 
  b + 7 = j + s - 7 ∧ 
  b + 12 = 2 * j → 
  s = 13 ∧ b = 10 ∧ j = 11 := by
sorry

end NUMINAMATH_CALUDE_squirrel_problem_l3324_332469


namespace NUMINAMATH_CALUDE_sum_of_squares_l3324_332437

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3324_332437


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_two_l3324_332475

theorem set_equality_implies_a_equals_two (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 2})
  (h2 : B = {1, a})
  (h3 : A = B) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_two_l3324_332475


namespace NUMINAMATH_CALUDE_july_birth_percentage_l3324_332474

theorem july_birth_percentage (total : ℕ) (july_births : ℕ) 
  (h1 : total = 150) (h2 : july_births = 18) : 
  (july_births : ℚ) / total * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l3324_332474


namespace NUMINAMATH_CALUDE_min_value_xyz_l3324_332459

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3324_332459


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3324_332499

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3324_332499


namespace NUMINAMATH_CALUDE_sequence_properties_l3324_332471

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, S n = -n^2 + 7*n) →
  (∀ n : ℕ, n ≥ 2 → S n - S (n-1) = a n) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, n ≠ 3 ∧ n ≠ 4 → S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3324_332471


namespace NUMINAMATH_CALUDE_periodic_function_value_l3324_332480

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2014 = 5 → f 2015 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l3324_332480


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l3324_332431

theorem solution_implies_k_value (k x y : ℚ) : 
  x = 3 ∧ y = 2 ∧ k * x + 3 * y = 1 → k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l3324_332431


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3324_332408

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    prove that the sum of the first 8 terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 0 + a 1 + a 2 + a 3 = 1) →  -- sum of first 4 terms is 1
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3324_332408


namespace NUMINAMATH_CALUDE_intersection_A_not_B_l3324_332442

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / |x - 1| < 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 > 0}

-- Define the complement of B
def not_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_not_B : A ∩ not_B = {x | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_not_B_l3324_332442


namespace NUMINAMATH_CALUDE_blue_socks_count_l3324_332455

/-- The number of red socks -/
def red_socks : ℕ := 2

/-- The number of black socks -/
def black_socks : ℕ := 2

/-- The number of white socks -/
def white_socks : ℕ := 2

/-- The probability of drawing two socks of the same color -/
def same_color_prob : ℚ := 1/5

theorem blue_socks_count (x : ℕ) (hx : x > 0) :
  let total := red_socks + black_socks + white_socks + x
  (3 * 2 + x * (x - 1)) / (total * (total - 1)) = same_color_prob →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_blue_socks_count_l3324_332455


namespace NUMINAMATH_CALUDE_incorrect_quotient_calculation_l3324_332425

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 40) :
  dividend / incorrect_divisor = 70 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_quotient_calculation_l3324_332425


namespace NUMINAMATH_CALUDE_research_team_composition_l3324_332467

/-- Represents the composition of a research team -/
structure ResearchTeam where
  total : Nat
  male : Nat
  female : Nat

/-- Represents the company's employee composition -/
def company : ResearchTeam :=
  { total := 60,
    male := 45,
    female := 15 }

/-- The size of the research team -/
def team_size : Nat := 4

/-- The probability of an employee being selected for the research team -/
def selection_probability : Rat := team_size / company.total

/-- The composition of the research team -/
def research_team : ResearchTeam :=
  { total := team_size,
    male := 3,
    female := 1 }

/-- The probability of selecting exactly one female when choosing two employees from the research team -/
def prob_one_female : Rat := 1 / 2

theorem research_team_composition :
  selection_probability = 1 / 15 ∧
  research_team.male = 3 ∧
  research_team.female = 1 ∧
  prob_one_female = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_research_team_composition_l3324_332467


namespace NUMINAMATH_CALUDE_intersection_line_l3324_332464

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y - 4 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l3324_332464


namespace NUMINAMATH_CALUDE_triangle_angles_l3324_332478

theorem triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 108 →          -- One angle is 108°
  b = 2 * c →        -- One angle is twice the other
  (b = 48 ∧ c = 24)  -- The two smaller angles are 48° and 24°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angles_l3324_332478


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l3324_332403

theorem trig_identity_simplification (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l3324_332403


namespace NUMINAMATH_CALUDE_cost_of_bananas_l3324_332400

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

end NUMINAMATH_CALUDE_cost_of_bananas_l3324_332400


namespace NUMINAMATH_CALUDE_certain_number_proof_l3324_332412

theorem certain_number_proof (h : 2994 / 14.5 = 177) : ∃ x : ℝ, x / 1.45 = 17.7 ∧ x = 25.665 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3324_332412


namespace NUMINAMATH_CALUDE_meters_equivalence_l3324_332495

-- Define the conversion rates
def meters_to_decimeters : ℝ := 10
def meters_to_centimeters : ℝ := 100

-- Define the theorem
theorem meters_equivalence : 
  7.34 = 7 + (3 / meters_to_decimeters) + (4 / meters_to_centimeters) := by
  sorry

end NUMINAMATH_CALUDE_meters_equivalence_l3324_332495


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3324_332413

theorem quadratic_inequality_empty_solution_set (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - m * x + m - 1 ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3324_332413


namespace NUMINAMATH_CALUDE_max_figures_per_shelf_l3324_332405

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

end NUMINAMATH_CALUDE_max_figures_per_shelf_l3324_332405


namespace NUMINAMATH_CALUDE_final_books_count_l3324_332409

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

end NUMINAMATH_CALUDE_final_books_count_l3324_332409


namespace NUMINAMATH_CALUDE_raccoon_nuts_problem_l3324_332428

theorem raccoon_nuts_problem (raccoon_holes possum_holes : ℕ) : 
  raccoon_holes + possum_holes = 25 →
  possum_holes = raccoon_holes - 3 →
  5 * raccoon_holes = 6 * possum_holes →
  5 * raccoon_holes = 70 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_nuts_problem_l3324_332428


namespace NUMINAMATH_CALUDE_contrapositive_example_l3324_332433

theorem contrapositive_example (a b : ℝ) :
  (¬(a + 1 > b) → ¬(a > b)) ↔ ((a + 1 ≤ b) → (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3324_332433


namespace NUMINAMATH_CALUDE_no_integers_satisfy_conditions_l3324_332491

def f (i : ℕ) : ℕ := 1 + i^(1/3) + i

theorem no_integers_satisfy_conditions :
  ¬∃ i : ℕ, 1 ≤ i ∧ i ≤ 3000 ∧ (∃ m : ℕ, i = m^3) ∧ f i = 1 + i^(1/3) + i :=
by sorry

end NUMINAMATH_CALUDE_no_integers_satisfy_conditions_l3324_332491


namespace NUMINAMATH_CALUDE_tournament_games_l3324_332453

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 30 players, 435 games are played -/
theorem tournament_games :
  num_games 30 = 435 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l3324_332453


namespace NUMINAMATH_CALUDE_packaging_waste_exceeds_target_l3324_332445

/-- The year when the packaging waste exceeds 40 million tons -/
def exceed_year : ℕ := 2021

/-- The initial packaging waste in 2015 (in million tons) -/
def initial_waste : ℝ := 4

/-- The annual growth rate of packaging waste -/
def growth_rate : ℝ := 0.5

/-- The target waste amount to exceed (in million tons) -/
def target_waste : ℝ := 40

/-- Function to calculate the waste amount after n years -/
def waste_after_years (n : ℕ) : ℝ :=
  initial_waste * (1 + growth_rate) ^ n

theorem packaging_waste_exceeds_target :
  waste_after_years (exceed_year - 2015) > target_waste ∧
  ∀ y : ℕ, y < exceed_year - 2015 → waste_after_years y ≤ target_waste :=
by sorry

end NUMINAMATH_CALUDE_packaging_waste_exceeds_target_l3324_332445


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l3324_332446

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.25 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 1.2 * G := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l3324_332446


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3324_332456

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {4, 7, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3324_332456


namespace NUMINAMATH_CALUDE_digit_700_of_3_11_is_7_l3324_332477

/-- The 700th digit past the decimal point in the decimal expansion of 3/11 -/
def digit_700_of_3_11 : ℕ :=
  -- Define the digit here
  7

/-- Theorem stating that the 700th digit past the decimal point
    in the decimal expansion of 3/11 is 7 -/
theorem digit_700_of_3_11_is_7 :
  digit_700_of_3_11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_700_of_3_11_is_7_l3324_332477


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3324_332482

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9 →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3324_332482


namespace NUMINAMATH_CALUDE_tangent_points_x_coordinate_sum_l3324_332460

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the relationship between x-coordinates of tangent points and the point on y = -2p -/
theorem tangent_points_x_coordinate_sum (para : Parabola) (M A B : Point) :
  A.y = A.x^2 / (2 * para.p) →  -- A is on the parabola
  B.y = B.x^2 / (2 * para.p) →  -- B is on the parabola
  M.y = -2 * para.p →  -- M is on the line y = -2p
  (A.y - M.y) / (A.x - M.x) = A.x / para.p →  -- MA is tangent to the parabola
  (B.y - M.y) / (B.x - M.x) = B.x / para.p →  -- MB is tangent to the parabola
  A.x + B.x = 2 * M.x := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_x_coordinate_sum_l3324_332460


namespace NUMINAMATH_CALUDE_example_is_quadratic_l3324_332407

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - x + 1 = 0 is a quadratic equation in terms of x -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l3324_332407


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l3324_332481

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (0, 5)

-- Define point E on BC
def E : ℝ × ℝ := (5, 2)

-- G is on the diagonal BD
def G : ℝ × ℝ := sorry

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_AGE :
  triangleArea A G E = 43.25 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l3324_332481


namespace NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l3324_332489

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  δ : ℝ
  hδ_pos : δ > 0

/-- The probability that a random variable is less than a given value -/
noncomputable def prob_lt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is greater than a given value -/
noncomputable def prob_gt (ξ : NormalRandomVariable) (x : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) (p : ℝ) 
    (h1 : ξ.μ = 1)
    (h2 : prob_lt ξ 1 = 1/2)
    (h3 : prob_gt ξ 2 = p) :
  prob_between ξ 0 1 = 1/2 - p := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l3324_332489


namespace NUMINAMATH_CALUDE_subtract_and_multiply_l3324_332411

theorem subtract_and_multiply (N V : ℝ) : N = 12 → (4 * N - 3 = 9 * (N - V)) → V = 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_and_multiply_l3324_332411


namespace NUMINAMATH_CALUDE_min_members_in_association_l3324_332420

/-- Represents an association with men and women members -/
structure Association where
  men : ℕ
  women : ℕ

/-- Calculates the total number of members in the association -/
def Association.totalMembers (a : Association) : ℕ := a.men + a.women

/-- Calculates the number of homeowners in the association -/
def Association.homeowners (a : Association) : ℚ := 0.1 * a.men + 0.2 * a.women

/-- Theorem stating the minimum number of members in the association -/
theorem min_members_in_association :
  ∃ (a : Association), a.homeowners ≥ 18 ∧
  (∀ (b : Association), b.homeowners ≥ 18 → a.totalMembers ≤ b.totalMembers) ∧
  a.totalMembers = 91 := by
  sorry

end NUMINAMATH_CALUDE_min_members_in_association_l3324_332420


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l3324_332417

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 28) :=
by sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l3324_332417


namespace NUMINAMATH_CALUDE_solution_theorem_l3324_332435

/-- A function satisfying the given condition for all non-zero real numbers -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 4 * x

theorem solution_theorem :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    ∀ x : ℝ, x ≠ 0 →
      (f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_theorem_l3324_332435


namespace NUMINAMATH_CALUDE_expression_simplification_l3324_332447

theorem expression_simplification (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (a - 4) / a / ((a + 2) / (a^2 - 2*a) - (a - 1) / (a^2 - 4*a + 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3324_332447


namespace NUMINAMATH_CALUDE_no_integer_with_five_divisors_sum_square_l3324_332454

theorem no_integer_with_five_divisors_sum_square : ¬ ∃ (n : ℕ+), 
  (∃ (d₁ d₂ d₃ d₄ d₅ : ℕ+), 
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (∀ (d : ℕ+), d ∣ n → d ≥ d₅ ∨ d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄)) ∧
  (∃ (x : ℕ), (d₁ : ℕ)^2 + (d₂ : ℕ)^2 + (d₃ : ℕ)^2 + (d₄ : ℕ)^2 + (d₅ : ℕ)^2 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_with_five_divisors_sum_square_l3324_332454


namespace NUMINAMATH_CALUDE_difference_of_squares_l3324_332449

theorem difference_of_squares (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3324_332449


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_l3324_332421

theorem union_necessary_not_sufficient (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) := by
  sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_l3324_332421


namespace NUMINAMATH_CALUDE_board_numbers_product_l3324_332465

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l3324_332465


namespace NUMINAMATH_CALUDE_remaining_perimeter_is_56_l3324_332401

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

end NUMINAMATH_CALUDE_remaining_perimeter_is_56_l3324_332401


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l3324_332496

/-- Polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- Polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x + x^2 - 6*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that -5/11 is the value of c that makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -5/11 ∧ 
  (∀ (x : ℝ), h c x = (1 + 3*c) + (-12 - 2*c)*x + (3 + c)*x^2 + (-4 - 6*c)*x^3) :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l3324_332496


namespace NUMINAMATH_CALUDE_positive_X_value_l3324_332427

-- Define the ⊠ operation
def boxtimes (X Y : ℤ) : ℤ := X^2 - 2*X + Y^2

-- Theorem statement
theorem positive_X_value :
  ∃ X : ℤ, (boxtimes X 7 = 164) ∧ (X > 0) ∧ (∀ Y : ℤ, (boxtimes Y 7 = 164) ∧ (Y > 0) → Y = X) :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l3324_332427


namespace NUMINAMATH_CALUDE_equation_solution_l3324_332497

theorem equation_solution :
  let f : ℂ → ℂ := λ x => (x - 2)^4 + (x - 6)^4
  ∃ (a b c d : ℂ),
    (f a = 16 ∧ f b = 16 ∧ f c = 16 ∧ f d = 16) ∧
    (a = 4 + Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (b = 4 - Complex.I * Real.sqrt (12 - 8 * Real.sqrt 2)) ∧
    (c = 4 + Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    (d = 4 - Complex.I * Real.sqrt (12 + 8 * Real.sqrt 2)) ∧
    ∀ (x : ℂ), f x = 16 → (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3324_332497


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3324_332486

/-- Proposition p: am² < bm² -/
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2

/-- Proposition q: a < b -/
def q (a b : ℝ) : Prop := a < b

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ a b m : ℝ, p a b m → q a b) ∧
  ¬(∀ a b : ℝ, q a b → ∀ m : ℝ, p a b m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3324_332486


namespace NUMINAMATH_CALUDE_tangent_line_equation_circle_radius_l3324_332432

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

end NUMINAMATH_CALUDE_tangent_line_equation_circle_radius_l3324_332432


namespace NUMINAMATH_CALUDE_range_of_m_l3324_332485

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - m| < 5) ↔ -2 < m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3324_332485


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l3324_332424

/-- The number of real values of b for which the line y = 2x + b intersects
    the parabola y = x^2 - 4x + b^2 at its vertex -/
theorem line_intersects_parabola_vertex : 
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b ∈ s, ∃ x y : ℝ, 
    (y = 2 * x + b) ∧ 
    (y = x^2 - 4 * x + b^2) ∧
    (∀ x' y' : ℝ, (y' = x'^2 - 4 * x' + b^2) → y' ≤ y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l3324_332424


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l3324_332463

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle_a : ℝ
  angle_b : ℝ
  is_isosceles : (angle_a = angle_b) ∨ (angle_a + 2 * angle_b = 180) ∨ (2 * angle_a + angle_b = 180)

-- Define the theorem
theorem isosceles_triangle_angle_b (t : IsoscelesTriangle) 
  (h : t.angle_a = 70) : 
  t.angle_b = 55 ∨ t.angle_b = 70 ∨ t.angle_b = 40 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l3324_332463


namespace NUMINAMATH_CALUDE_triathlete_average_rate_l3324_332466

/-- The average rate of a triathlete's round trip -/
theorem triathlete_average_rate 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (swimming_distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : total_distance = 6) 
  (h2 : running_distance = total_distance / 2) 
  (h3 : swimming_distance = total_distance / 2) 
  (h4 : running_speed = 10) 
  (h5 : swimming_speed = 6) : 
  (total_distance / ((running_distance / running_speed + swimming_distance / swimming_speed) * 60)) = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_average_rate_l3324_332466


namespace NUMINAMATH_CALUDE_no_solution_exists_l3324_332484

theorem no_solution_exists : ¬∃ (k t : ℕ), 
  (1 ≤ k ∧ k ≤ 9) ∧ 
  (1 ≤ t ∧ t ≤ 9) ∧ 
  (808 + 10 * k) - (800 + 88 * k) = 1606 + 10 * t :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3324_332484


namespace NUMINAMATH_CALUDE_machine_working_time_l3324_332498

theorem machine_working_time : ∃ y : ℝ, y > 0 ∧ 1 / (y + 4) + 1 / (y + 2) + 1 / y^2 = 1 / y ∧ y = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l3324_332498


namespace NUMINAMATH_CALUDE_dvd_sales_l3324_332490

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * (cd : ℝ) →
  dvd + cd = 273 →
  dvd = 168 := by
  sorry

end NUMINAMATH_CALUDE_dvd_sales_l3324_332490


namespace NUMINAMATH_CALUDE_xy_equals_twelve_l3324_332439

theorem xy_equals_twelve (x y : ℝ) 
  (h1 : x * (x + y) = x^2 + 12) 
  (h2 : x - y = 3) : 
  x * y = 12 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_twelve_l3324_332439


namespace NUMINAMATH_CALUDE_probability_five_heads_in_six_tosses_l3324_332494

def n : ℕ := 6  -- number of coin tosses
def k : ℕ := 5  -- number of heads we want to get
def p : ℚ := 1/2  -- probability of getting heads on a single toss (fair coin)

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of getting exactly k successes in n trials
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

-- The theorem to prove
theorem probability_five_heads_in_six_tosses :
  probability_k_successes n k p = 0.09375 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_heads_in_six_tosses_l3324_332494


namespace NUMINAMATH_CALUDE_kate_needs_57_more_l3324_332443

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

end NUMINAMATH_CALUDE_kate_needs_57_more_l3324_332443


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l3324_332458

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 9
def S : ℝ := 20

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l3324_332458


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l3324_332444

theorem smallest_positive_integer (x : ℕ+) : (x^3 : ℚ) / (x^2 : ℚ) < 15 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l3324_332444


namespace NUMINAMATH_CALUDE_slope_of_specific_midpoint_line_l3324_332410

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

end NUMINAMATH_CALUDE_slope_of_specific_midpoint_line_l3324_332410
