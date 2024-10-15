import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_right_triangles_l849_84912

/-- A right triangle with integer sides where the hypotenuse and one side are consecutive. -/
structure ConsecutiveRightTriangle where
  a : ℕ  -- One side of the triangle
  b : ℕ  -- The other side of the triangle
  c : ℕ  -- The hypotenuse
  consecutive : c = a + 1
  pythagorean : a^2 + b^2 = c^2

/-- There exist infinitely many ConsecutiveRightTriangles. -/
theorem infinitely_many_consecutive_right_triangles :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ t : ConsecutiveRightTriangle, t.c = m :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_right_triangles_l849_84912


namespace NUMINAMATH_CALUDE_intersection_S_T_l849_84944

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0}

def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l849_84944


namespace NUMINAMATH_CALUDE_convention_handshakes_l849_84940

-- Define the number of companies and representatives per company
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4

-- Define the total number of people
def total_people : ℕ := num_companies * reps_per_company

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := total_people - reps_per_company

-- Theorem statement
theorem convention_handshakes : 
  (total_people * handshakes_per_person) / 2 = 160 := by
  sorry


end NUMINAMATH_CALUDE_convention_handshakes_l849_84940


namespace NUMINAMATH_CALUDE_min_value_product_l849_84988

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1) ≥ 24 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l849_84988


namespace NUMINAMATH_CALUDE_base9_813_equals_base3_220110_l849_84975

/-- Converts a base-9 number to base-3 --/
def base9_to_base3 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: 813 in base 9 is equal to 220110 in base 3 --/
theorem base9_813_equals_base3_220110 : base9_to_base3 813 = 220110 := by
  sorry

end NUMINAMATH_CALUDE_base9_813_equals_base3_220110_l849_84975


namespace NUMINAMATH_CALUDE_correct_costs_l849_84990

/-- The cost of a pen in yuan -/
def pen_cost : ℝ := 10

/-- The cost of an exercise book in yuan -/
def book_cost : ℝ := 1

/-- The total cost of 2 exercise books and 1 pen in yuan -/
def total_cost : ℝ := 12

theorem correct_costs :
  (2 * book_cost + pen_cost = total_cost) ∧
  (book_cost = 0.1 * pen_cost) ∧
  (pen_cost = 10) ∧
  (book_cost = 1) := by sorry

end NUMINAMATH_CALUDE_correct_costs_l849_84990


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l849_84906

theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 2
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ x : ℝ, x^2 - 4*x*(Real.sqrt 2) + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l849_84906


namespace NUMINAMATH_CALUDE_profit_is_12_5_l849_84974

/-- Calculates the profit per piece given the purchase price, markup percentage, and discount percentage. -/
def calculate_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let initial_price := purchase_price * (1 + markup_percent)
  let final_price := initial_price * (1 - discount_percent)
  final_price - purchase_price

/-- Theorem stating that the profit per piece is 12.5 yuan under the given conditions. -/
theorem profit_is_12_5 :
  calculate_profit 100 0.25 0.1 = 12.5 := by
  sorry

#eval calculate_profit 100 0.25 0.1

end NUMINAMATH_CALUDE_profit_is_12_5_l849_84974


namespace NUMINAMATH_CALUDE_f_range_implies_m_plus_n_range_l849_84950

-- Define the function f(x)
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

-- Define the interval [m, n]
def interval (m n : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ n }

-- State the theorem
theorem f_range_implies_m_plus_n_range (m n : ℝ) :
  (∀ x ∈ interval m n, -6 ≤ f x ∧ f x ≤ 2) →
  (0 ≤ m + n ∧ m + n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_m_plus_n_range_l849_84950


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l849_84907

theorem point_movement_on_number_line : 
  let start : ℤ := -2
  let move_right : ℤ := 7
  let move_left : ℤ := 4
  start + move_right - move_left = 1 :=
by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l849_84907


namespace NUMINAMATH_CALUDE_x_is_integer_l849_84997

theorem x_is_integer (x : ℝ) 
  (h1 : ∃ n : ℤ, x^1960 - x^1919 = n)
  (h2 : ∃ m : ℤ, x^2001 - x^1960 = m)
  (h3 : ∃ k : ℤ, x^2001 - x^1919 = k) : 
  ∃ z : ℤ, x = z := by
sorry

end NUMINAMATH_CALUDE_x_is_integer_l849_84997


namespace NUMINAMATH_CALUDE_negation_of_existence_l849_84955

variable (a : ℝ)

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existence_l849_84955


namespace NUMINAMATH_CALUDE_no_mode_in_set_l849_84904

def number_set : Finset ℕ := {91, 85, 80, 83, 84}

def x : ℕ := 504 - (91 + 85 + 80 + 83 + 84)

def complete_set : Finset ℕ := number_set ∪ {x}

theorem no_mode_in_set :
  (Finset.card complete_set = 6) ∧
  (Finset.sum complete_set id / Finset.card complete_set = 84) →
  ∀ n : ℕ, (complete_set.filter (λ m => m = n)).card ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_no_mode_in_set_l849_84904


namespace NUMINAMATH_CALUDE_initial_average_weight_l849_84928

theorem initial_average_weight (a b c d e : ℝ) : 
  -- Initial conditions
  (a + b + c) / 3 = (a + b + c) / 3 →
  -- Adding packet d
  (a + b + c + d) / 4 = 80 →
  -- Replacing a with e
  (b + c + d + e) / 4 = 79 →
  -- Relationship between d and e
  e = d + 3 →
  -- Weight of packet a
  a = 75 →
  -- Conclusion: initial average weight
  (a + b + c) / 3 = 84 := by
sorry


end NUMINAMATH_CALUDE_initial_average_weight_l849_84928


namespace NUMINAMATH_CALUDE_triangle_projection_types_l849_84952

-- Define the possible projection types
inductive ProjectionType
  | Angle
  | Strip
  | TwoAnglesJoined
  | Triangle
  | AngleWithInfiniteFigure

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to determine if a point is on a plane
def isPointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Function to determine if three points are collinear
def areCollinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), p3.x - p1.x = t * (p2.x - p1.x) ∧
              p3.y - p1.y = t * (p2.y - p1.y) ∧
              p3.z - p1.z = t * (p2.z - p1.z)

-- Define the projection function
def project (triangle : Triangle3D) (O : Point3D) (P : Plane3D) : ProjectionType :=
  sorry -- Actual implementation would go here

-- The main theorem
theorem triangle_projection_types 
  (triangle : Triangle3D) 
  (O : Point3D) 
  (P : Plane3D) 
  (h1 : ¬ isPointOnPlane O (Plane3D.mk 0 0 0 0)) -- O is not in the plane of the triangle
  (h2 : ¬ areCollinear triangle.A triangle.B triangle.C) -- ABC is a valid triangle
  : ∃ (projType : ProjectionType), project triangle O P = projType ∧ 
    (projType = ProjectionType.Angle ∨ 
     projType = ProjectionType.Strip ∨ 
     projType = ProjectionType.TwoAnglesJoined ∨ 
     projType = ProjectionType.Triangle ∨ 
     projType = ProjectionType.AngleWithInfiniteFigure) :=
  sorry


end NUMINAMATH_CALUDE_triangle_projection_types_l849_84952


namespace NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l849_84939

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeanDistribution where
  blue : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeanDistribution.total (d : JellyBeanDistribution) : ℕ :=
  d.blue + d.red + d.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeanDistribution :=
  { blue := 1, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeanDistribution :=
  { blue := 2, red := 1, yellow := 2 }

/-- Calculates the probability of matching colors -/
def probability_matching_colors (d1 d2 : JellyBeanDistribution) : ℚ :=
  (d1.blue * d2.blue + d1.red * d2.red : ℚ) / ((d1.total * d2.total) : ℚ)

theorem matching_probability_is_four_fifteenths :
  probability_matching_colors abe bob = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_is_four_fifteenths_l849_84939


namespace NUMINAMATH_CALUDE_airport_gate_probability_l849_84994

/-- The number of gates in the airport -/
def num_gates : ℕ := 16

/-- The distance between adjacent gates in feet -/
def distance_between_gates : ℕ := 75

/-- The maximum distance Dina is willing to walk in feet -/
def max_walking_distance : ℕ := 300

/-- The probability of walking 300 feet or less to the new gate -/
def probability_short_walk : ℚ := 8/15

theorem airport_gate_probability :
  let total_possibilities := num_gates * (num_gates - 1)
  let gates_within_distance := 2 * (max_walking_distance / distance_between_gates)
  let favorable_outcomes := num_gates * gates_within_distance
  (favorable_outcomes : ℚ) / total_possibilities = probability_short_walk :=
sorry

end NUMINAMATH_CALUDE_airport_gate_probability_l849_84994


namespace NUMINAMATH_CALUDE_river_current_speed_proof_l849_84930

def river_current_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  let current_speed := 4
  current_speed

theorem river_current_speed_proof (boat_speed distance total_time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : distance = 60)
  (h3 : total_time = 6.25) :
  river_current_speed boat_speed distance total_time = 4 := by
  sorry

#check river_current_speed_proof

end NUMINAMATH_CALUDE_river_current_speed_proof_l849_84930


namespace NUMINAMATH_CALUDE_simplify_expression_l849_84970

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 0) :
  a - b + (2 * b^2) / (a + b) = (a^2 + b^2) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l849_84970


namespace NUMINAMATH_CALUDE_total_faces_painted_l849_84916

/-- The number of cuboids Amelia painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Amelia -/
theorem total_faces_painted : num_cuboids * faces_per_cuboid = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_l849_84916


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l849_84993

-- Define points A, B, and P
def A : ℝ × ℝ := (6, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  -- Part 1: Perpendicular bisector
  (∀ x y, perp_bisector x y ↔ 
    -- Midpoint condition
    ((x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
     ((A.1 - B.1)/2)^2 + ((A.2 - B.2)/2)^2) ∧
    -- Perpendicularity condition
    ((y - A.2)*(B.1 - A.1) = -(x - A.1)*(B.2 - A.2))) ∧
  -- Part 2: Parallel line
  (∀ x y, parallel_line x y ↔
    -- Point P lies on the line
    (2*P.1 + P.2 - 1 = 0) ∧
    -- Parallel to AB
    ((y - P.2)/(x - P.1) = (B.2 - A.2)/(B.1 - A.1))) :=
sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l849_84993


namespace NUMINAMATH_CALUDE_identity_function_unique_l849_84979

def PositiveNat := {n : ℕ // n > 0}

theorem identity_function_unique 
  (f : PositiveNat → PositiveNat) 
  (h : ∀ (m n : PositiveNat), ∃ (k : ℕ), k * (m.val^2 + (f n).val) = m.val * (f m).val + n.val) : 
  ∀ (n : PositiveNat), f n = n :=
sorry

end NUMINAMATH_CALUDE_identity_function_unique_l849_84979


namespace NUMINAMATH_CALUDE_midpoint_on_yaxis_product_l849_84938

/-- Given a function f(x) = a^x where a > 0 and a ≠ 1, if the midpoint of the line segment
    with endpoints (x₁, f(x₁)) and (x₂, f(x₂)) is on the y-axis, then f(x₁) · f(x₂) = 1 -/
theorem midpoint_on_yaxis_product (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) (ha_ne_one : a ≠ 1) 
  (h_midpoint : x₁ + x₂ = 0) : 
  (a^x₁) * (a^x₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_on_yaxis_product_l849_84938


namespace NUMINAMATH_CALUDE_marble_selection_probability_l849_84917

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly one marble of each color, with one color being chosen twice -/
theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2) /
  Nat.choose total_marbles selected_marbles = 9 / 14 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l849_84917


namespace NUMINAMATH_CALUDE_tank_capacity_l849_84947

/-- The capacity of a tank given specific inlet and outlet pipe rates --/
theorem tank_capacity 
  (outlet_time : ℝ) 
  (inlet_rate1 : ℝ) 
  (inlet_rate2 : ℝ) 
  (extended_time : ℝ) 
  (h1 : outlet_time = 10) 
  (h2 : inlet_rate1 = 4) 
  (h3 : inlet_rate2 = 6) 
  (h4 : extended_time = 8) : 
  ∃ (capacity : ℝ), 
    capacity = 13500 ∧ 
    capacity / outlet_time - (inlet_rate1 * 60 + inlet_rate2 * 60) = capacity / (outlet_time + extended_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l849_84947


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l849_84978

-- Define a point in the grid
structure Point where
  x : Nat
  y : Nat

-- Define the grid size
def gridSize : Nat := 6

-- Define the initially shaded squares
def initialShaded : List Point := [
  { x := 1, y := 1 },
  { x := 1, y := 6 },
  { x := 6, y := 1 },
  { x := 3, y := 4 }
]

-- Function to check if a point is within the grid
def inGrid (p : Point) : Bool :=
  1 ≤ p.x ∧ p.x ≤ gridSize ∧ 1 ≤ p.y ∧ p.y ≤ gridSize

-- Function to check if a set of points has both horizontal and vertical symmetry
def hasSymmetry (points : List Point) : Bool :=
  sorry

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additionalPoints : List Point),
    additionalPoints.length = 4 ∧
    (∀ p ∈ additionalPoints, inGrid p) ∧
    hasSymmetry (initialShaded ++ additionalPoints) ∧
    (∀ (otherPoints : List Point),
      otherPoints.length < 4 →
      ¬ hasSymmetry (initialShaded ++ otherPoints)) :=
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l849_84978


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l849_84908

theorem fractional_equation_m_range : 
  ∀ (x m : ℝ), 
    ((x + m) / (x - 2) - (2 * m) / (x - 2) = 3) →
    (x > 0) →
    (m < 6 ∧ m ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l849_84908


namespace NUMINAMATH_CALUDE_two_friend_visits_count_l849_84963

/-- Represents a friend with a visitation period -/
structure Friend where
  period : ℕ

/-- Calculates the number of days in a given period where exactly two out of three friends visit -/
def countTwoFriendVisits (f1 f2 f3 : Friend) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 27 days in a 365-day period 
    where two out of three friends visit, given their visitation periods -/
theorem two_friend_visits_count : 
  let max : Friend := { period := 5 }
  let nora : Friend := { period := 6 }
  let olivia : Friend := { period := 7 }
  countTwoFriendVisits max nora olivia 365 = 27 := by
  sorry

end NUMINAMATH_CALUDE_two_friend_visits_count_l849_84963


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l849_84935

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧ digit_sum n = 25 ∧ n % 5 = 0

theorem unique_number_with_conditions :
  ∃! n : ℕ, satisfies_conditions n :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l849_84935


namespace NUMINAMATH_CALUDE_solution_value_l849_84918

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 5 = 0) : 2 * a - 3 * b + 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l849_84918


namespace NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l849_84901

-- Define the types for planes and lines
def Plane : Type := sorry
def Line : Type := sorry

-- Define the parallel relation for planes
def parallel (α β : Plane) : Prop := sorry

-- Define the subset relation for a line and a plane
def subset_plane (m : Line) (β : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_not_sufficient_nor_necessary 
  (α β : Plane) (m : Line) (h : parallel α β) :
  ¬(∀ m, subset_plane m β → parallel α β) ∧ 
  ¬(∀ m, parallel α β → subset_plane m β) := by
  sorry

end NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l849_84901


namespace NUMINAMATH_CALUDE_congruence_solution_l849_84969

theorem congruence_solution (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 201) (h3 : 200 * n ≡ 144 [ZMOD 101]) :
  n ≡ 29 [ZMOD 101] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l849_84969


namespace NUMINAMATH_CALUDE_correct_allocation_plans_l849_84956

/-- Represents the number of factories --/
def num_factories : Nat := 4

/-- Represents the number of classes --/
def num_classes : Nat := 3

/-- Represents the requirement that at least one factory must have a class --/
def must_have_class : Nat := 1

/-- The number of different allocation plans --/
def allocation_plans : Nat := 57

/-- Theorem stating that the number of allocation plans is correct --/
theorem correct_allocation_plans :
  (num_factories = 4) →
  (num_classes = 3) →
  (must_have_class = 1) →
  (allocation_plans = 57) := by
  sorry

end NUMINAMATH_CALUDE_correct_allocation_plans_l849_84956


namespace NUMINAMATH_CALUDE_mersenne_factor_square_plus_nine_l849_84910

theorem mersenne_factor_square_plus_nine (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_mersenne_factor_square_plus_nine_l849_84910


namespace NUMINAMATH_CALUDE_factorization_equality_l849_84986

theorem factorization_equality (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l849_84986


namespace NUMINAMATH_CALUDE_list_price_calculation_l849_84958

theorem list_price_calculation (list_price : ℝ) : 
  (list_price ≥ 0) →
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) →
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_calculation_l849_84958


namespace NUMINAMATH_CALUDE_point_C_values_l849_84957

-- Define the points on the number line
def point_A : ℝ := 2
def point_B : ℝ := -4

-- Define the property of equal distances between adjacent points
def equal_distances (c : ℝ) : Prop :=
  abs (point_A - point_B) = abs (point_B - c) ∧ 
  abs (point_A - point_B) = abs (point_A - c)

-- Theorem statement
theorem point_C_values : 
  ∀ c : ℝ, equal_distances c → (c = -10 ∨ c = 8) :=
by sorry

end NUMINAMATH_CALUDE_point_C_values_l849_84957


namespace NUMINAMATH_CALUDE_commission_for_398_machines_l849_84920

/-- Represents the commission structure and pricing model for machine sales -/
structure SalesModel where
  initialPrice : ℝ
  priceDecrease : ℝ
  commissionRate1 : ℝ
  commissionRate2 : ℝ
  commissionRate3 : ℝ
  threshold1 : ℕ
  threshold2 : ℕ

/-- Calculates the total commission for a given number of machines sold -/
def calculateCommission (model : SalesModel) (machinesSold : ℕ) : ℝ :=
  sorry

/-- The specific sales model for the problem -/
def problemModel : SalesModel :=
  { initialPrice := 10000
    priceDecrease := 500
    commissionRate1 := 0.03
    commissionRate2 := 0.04
    commissionRate3 := 0.05
    threshold1 := 150
    threshold2 := 250 }

theorem commission_for_398_machines :
  calculateCommission problemModel 398 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_commission_for_398_machines_l849_84920


namespace NUMINAMATH_CALUDE_william_land_percentage_l849_84925

-- Define the total tax collected from the village
def total_tax : ℝ := 3840

-- Define Mr. William's tax payment
def william_tax : ℝ := 480

-- Define the percentage of cultivated land that is taxed
def tax_percentage : ℝ := 0.9

-- Theorem statement
theorem william_land_percentage :
  (william_tax / total_tax) * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_william_land_percentage_l849_84925


namespace NUMINAMATH_CALUDE_complex_equation_solution_l849_84981

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = 10 → z = 1 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l849_84981


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l849_84984

/-- Represents a class of students with given age statistics -/
structure ClassStats where
  total_students : Nat
  class_average : Float
  group1_size : Nat
  group1_average : Float
  group2_size : Nat
  group3_size : Nat
  group3_average : Float
  remaining_boys_average : Float

/-- Theorem stating the age of the 15th student given the class statistics -/
theorem fifteenth_student_age (stats : ClassStats) 
  (h1 : stats.total_students = 15)
  (h2 : stats.class_average = 15.2)
  (h3 : stats.group1_size = 5)
  (h4 : stats.group1_average = 14)
  (h5 : stats.group2_size = 4)
  (h6 : stats.group3_size = 3)
  (h7 : stats.group3_average = 16.6)
  (h8 : stats.remaining_boys_average = 15.4)
  (h9 : stats.group1_size + stats.group2_size + stats.group3_size + 3 = stats.total_students) :
  ∃ (age : Float), age = 15.7 ∧ age = (stats.class_average * stats.total_students.toFloat
                                      - stats.group1_average * stats.group1_size.toFloat
                                      - stats.group3_average * stats.group3_size.toFloat
                                      - stats.remaining_boys_average * 3)
                                      / stats.group2_size.toFloat :=
by sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l849_84984


namespace NUMINAMATH_CALUDE_rugby_team_average_weight_l849_84983

theorem rugby_team_average_weight 
  (initial_players : ℕ) 
  (new_player_weight : ℝ) 
  (new_average_weight : ℝ) : 
  initial_players = 20 ∧ 
  new_player_weight = 210 ∧ 
  new_average_weight = 181.42857142857142 → 
  (initial_players * (new_average_weight * (initial_players + 1) - new_player_weight)) / initial_players = 180 := by
sorry

end NUMINAMATH_CALUDE_rugby_team_average_weight_l849_84983


namespace NUMINAMATH_CALUDE_no_good_number_l849_84980

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem no_good_number :
  ¬ ∃ n : ℕ, 
    is_divisible_by_sum_of_digits n ∧
    is_divisible_by_sum_of_digits (n + 1) ∧
    is_divisible_by_sum_of_digits (n + 2) ∧
    is_divisible_by_sum_of_digits (n + 3) :=
sorry

end NUMINAMATH_CALUDE_no_good_number_l849_84980


namespace NUMINAMATH_CALUDE_group_size_before_new_member_l849_84982

theorem group_size_before_new_member 
  (avg_after : ℚ) 
  (new_member_amount : ℚ) 
  (avg_before : ℚ) 
  (h1 : avg_after = 20)
  (h2 : new_member_amount = 56)
  (h3 : avg_before = 14) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * avg_before + new_member_amount = (n + 1 : ℚ) * avg_after ∧
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_group_size_before_new_member_l849_84982


namespace NUMINAMATH_CALUDE_systematic_sampling_calculation_l849_84927

def population_size : ℕ := 2005
def sample_size : ℕ := 50

theorem systematic_sampling_calculation :
  let sampling_interval := population_size / sample_size
  let discarded := population_size % sample_size
  sampling_interval = 40 ∧ discarded = 5 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_calculation_l849_84927


namespace NUMINAMATH_CALUDE_segment_length_l849_84913

/-- Given a line segment AB divided by points P and Q, prove that the length of AB is 135/7 -/
theorem segment_length (A B P Q : ℝ) : 
  (∃ x y : ℝ, 
    A < P ∧ P < Q ∧ Q < B ∧  -- P and Q are between A and B
    P - A = 3*x ∧ B - P = 2*x ∧  -- P divides AB in ratio 3:2
    Q - A = 4*y ∧ B - Q = 5*y ∧  -- Q divides AB in ratio 4:5
    Q - P = 3)  -- Distance between P and Q is 3
  → B - A = 135/7 := by
sorry

end NUMINAMATH_CALUDE_segment_length_l849_84913


namespace NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l849_84966

theorem smallest_x_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧ 
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l849_84966


namespace NUMINAMATH_CALUDE_max_b_when_a_is_e_min_a_minus_b_l849_84902

open Real

-- Define the condition that e^x ≥ ax + b for all x
def condition (a b : ℝ) : Prop := ∀ x, exp x ≥ a * x + b

theorem max_b_when_a_is_e :
  (condition e b) → b ≤ 0 :=
sorry

theorem min_a_minus_b :
  ∃ a b, condition a b ∧ ∀ a' b', condition a' b' → a - b ≤ a' - b' ∧ a - b = -1/e :=
sorry

end NUMINAMATH_CALUDE_max_b_when_a_is_e_min_a_minus_b_l849_84902


namespace NUMINAMATH_CALUDE_f_is_quadratic_l849_84948

/-- Definition of a quadratic equation with one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 4x - x² -/
def f (x : ℝ) : ℝ := 4 * x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l849_84948


namespace NUMINAMATH_CALUDE_first_day_distance_l849_84976

/-- Proves the distance covered on the first day of a three-day hike -/
theorem first_day_distance (total_distance : ℝ) (second_day : ℝ) (third_day : ℝ)
  (h1 : total_distance = 50)
  (h2 : second_day = total_distance / 2)
  (h3 : third_day = 15)
  : total_distance - second_day - third_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_day_distance_l849_84976


namespace NUMINAMATH_CALUDE_books_on_shelf_correct_book_count_l849_84922

theorem books_on_shelf (initial_figures : ℕ) (added_figures : ℕ) (extra_books : ℕ) : ℕ :=
  let total_figures := initial_figures + added_figures
  let total_books := total_figures + extra_books
  total_books

theorem correct_book_count : books_on_shelf 2 4 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_correct_book_count_l849_84922


namespace NUMINAMATH_CALUDE_least_valid_number_l849_84996

def is_valid (n : ℕ) : Prop :=
  n % 11 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 7 = 1

theorem least_valid_number : ∀ m : ℕ, m < 2521 → ¬(is_valid m) ∧ is_valid 2521 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l849_84996


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l849_84949

/-- An arithmetic sequence {aₙ} satisfying aₙ₊₁ + aₙ = 4n for all n has a₁ = 1 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)  -- arithmetic sequence condition
  (h_sum : ∀ n, a (n + 1) + a n = 4 * n)                    -- given condition
  : a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l849_84949


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l849_84995

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l849_84995


namespace NUMINAMATH_CALUDE_fraction_problem_l849_84919

theorem fraction_problem (f : ℚ) : 
  (1 / 5 : ℚ)^4 * f^2 = 1 / (10 : ℚ)^4 → f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l849_84919


namespace NUMINAMATH_CALUDE_monkey_rope_system_length_l849_84968

/-- Represents the age and weight of a monkey and its mother, and the properties of a rope system -/
structure MonkeyRopeSystem where
  monkey_age : ℝ
  mother_age : ℝ
  rope_weight_per_foot : ℝ
  weight : ℝ

/-- The conditions of the monkey-rope system problem -/
def monkey_rope_system_conditions (s : MonkeyRopeSystem) : Prop :=
  s.monkey_age + s.mother_age = 4 ∧
  s.monkey_age = s.mother_age / 2 ∧
  s.rope_weight_per_foot = 1/4 ∧
  s.weight = s.mother_age

/-- The theorem stating that under the given conditions, the rope length is 5 feet -/
theorem monkey_rope_system_length
  (s : MonkeyRopeSystem)
  (h : monkey_rope_system_conditions s) :
  (s.weight + s.weight) / (3/4) = 5 :=
sorry

end NUMINAMATH_CALUDE_monkey_rope_system_length_l849_84968


namespace NUMINAMATH_CALUDE_complex_sum_zero_implies_b_equals_two_l849_84924

theorem complex_sum_zero_implies_b_equals_two (b : ℝ) : 
  (2 : ℂ) - Complex.I * b = (2 : ℂ) - Complex.I * b ∧ 
  (2 : ℝ) + (-b) = 0 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_implies_b_equals_two_l849_84924


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l849_84985

theorem quadratic_inequality_solution_set
  (a b c α β : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (h3 : α > 0)
  (h4 : a < 0)
  (h5 : α + β = -b / a)
  (h6 : α * β = c / a) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l849_84985


namespace NUMINAMATH_CALUDE_double_counted_page_l849_84900

theorem double_counted_page (n : ℕ) (m : ℕ) : 
  n > 0 ∧ m > 0 ∧ m ≤ n ∧ (n * (n + 1)) / 2 + m = 2040 → m = 24 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_page_l849_84900


namespace NUMINAMATH_CALUDE_mix_buyer_probability_l849_84933

theorem mix_buyer_probability (total : ℕ) (cake muffin cookie : ℕ) 
  (cake_muffin cake_cookie muffin_cookie : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_cake : cake = 70)
  (h_muffin : muffin = 60)
  (h_cookie : cookie = 50)
  (h_cake_muffin : cake_muffin = 25)
  (h_cake_cookie : cake_cookie = 15)
  (h_muffin_cookie : muffin_cookie = 10)
  (h_all_three : all_three = 5) : 
  (total - (cake + muffin + cookie - cake_muffin - cake_cookie - muffin_cookie + all_three)) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mix_buyer_probability_l849_84933


namespace NUMINAMATH_CALUDE_rectangle_area_l849_84953

theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 → l * w = 1600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l849_84953


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l849_84987

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if there are six consecutive nonprime numbers before a given number -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 97 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 97 ∧ hasSixConsecutiveNonprimes 97 ∧
  ∀ m : ℕ, m < 97 → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l849_84987


namespace NUMINAMATH_CALUDE_tournament_result_l849_84941

/-- Represents a tennis tournament with the given rules --/
structure TennisTournament where
  participants : ℕ
  points_for_win : ℕ
  points_for_loss : ℕ

/-- Calculates the number of participants finishing with a given number of points --/
def participants_with_points (t : TennisTournament) (points : ℕ) : ℕ :=
  Nat.choose (Nat.log 2 t.participants) points

theorem tournament_result (t : TennisTournament) 
  (h1 : t.participants = 512)
  (h2 : t.points_for_win = 1)
  (h3 : t.points_for_loss = 0) :
  participants_with_points t 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_tournament_result_l849_84941


namespace NUMINAMATH_CALUDE_equation_roots_property_l849_84971

-- Define the equation and its properties
def equation (m : ℤ) (x : ℤ) : Prop := x^2 + (m + 1) * x - 2 = 0

-- Define the roots
def is_root (m α β : ℤ) : Prop :=
  equation m (α + 1) ∧ equation m (β + 1) ∧ α < β ∧ m ≠ 0

-- Define d
def d (α β : ℤ) : ℤ := β - α

-- Theorem statement
theorem equation_roots_property :
  ∀ m α β : ℤ, is_root m α β → m = -2 ∧ d α β = 3 := by sorry

end NUMINAMATH_CALUDE_equation_roots_property_l849_84971


namespace NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l849_84946

theorem divisibility_of_square_sum_minus_2017 (n : ℕ) : 
  ∃ x y : ℤ, (n : ℤ) ∣ (x^2 + y^2 - 2017) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l849_84946


namespace NUMINAMATH_CALUDE_min_value_expression_l849_84962

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 < x ∧ x < 1/2) 
  (hy : -1/2 < y ∧ y < 1/2) 
  (hz : -1/2 < z ∧ z < 1/2) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z))) + 
  1/2 ≥ 5/2 ∧ 
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0))) + 
  1/2 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l849_84962


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l849_84911

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, (a = 1 → ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l849_84911


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l849_84915

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬∃ n, P n) ↔ (∀ n, ¬P n) := by sorry

theorem negation_of_proposition :
  (¬∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l849_84915


namespace NUMINAMATH_CALUDE_complex_repairs_is_two_l849_84964

/-- Represents Jim's bike shop operations for a month --/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the number of complex repairs given the shop's operations --/
def complex_repairs_count (shop : BikeShop) : ℕ :=
  sorry

/-- Theorem stating that the number of complex repairs is 2 --/
theorem complex_repairs_is_two (shop : BikeShop) 
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repair_cost = 50)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repairs_count shop = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_repairs_is_two_l849_84964


namespace NUMINAMATH_CALUDE_product_of_digits_for_non_divisible_by_five_l849_84973

def numbers : List Nat := [4750, 4760, 4775, 4785, 4790]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_for_non_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧ 
    units_digit n * tens_digit n = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_for_non_divisible_by_five_l849_84973


namespace NUMINAMATH_CALUDE_circle_construction_l849_84959

/-- Given four lines intersecting at a point with 45° angles between them, and a circle
    intersecting these lines such that two opposite chords have lengths a and b,
    and one chord is three times the length of its opposite chord,
    the circle's center (u, v) and radius r satisfy specific equations. -/
theorem circle_construction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (u v r : ℝ),
    u^2 = (a^2 - b^2) / 8 + Real.sqrt (((a^2 - b^2) / 8)^2 + ((a^2 + b^2) / 10)^2) ∧
    v^2 = r^2 - a^2 / 4 ∧
    r^2 = (u^2 + v^2) / 2 + (a^2 + b^2) / 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_construction_l849_84959


namespace NUMINAMATH_CALUDE_triangle_height_calculation_l849_84998

theorem triangle_height_calculation (base area height : Real) : 
  base = 8.4 → area = 24.36 → area = (base * height) / 2 → height = 5.8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_calculation_l849_84998


namespace NUMINAMATH_CALUDE_D_72_l849_84905

/-- The number of ways to write a positive integer as a product of integers greater than 1, considering order. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
axiom prime_factorization_72 : ∃ (a b : ℕ), 72 = 2^3 * 3^2

/-- Theorem: The number of ways to write 72 as a product of integers greater than 1, considering order, is 26 -/
theorem D_72 : D 72 = 26 := by
  sorry

end NUMINAMATH_CALUDE_D_72_l849_84905


namespace NUMINAMATH_CALUDE_divisor_equation_solution_l849_84903

def is_sixth_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d)

def is_seventh_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (d1 d2 d3 d4 d5 d6 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n ∧ d6 ∣ n ∧
    1 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 < d6 ∧ d6 < d)

theorem divisor_equation_solution (n : ℕ) :
  (∃ (d6 d7 : ℕ), is_sixth_divisor n d6 ∧ is_seventh_divisor n d7 ∧ n = d6^2 + d7^2 - 1) →
  n = 144 ∨ n = 1984 :=
by sorry

end NUMINAMATH_CALUDE_divisor_equation_solution_l849_84903


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l849_84936

theorem floor_ceiling_sum : ⌊(-3.72 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l849_84936


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l849_84931

theorem arithmetic_calculation : 10 * (1/8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l849_84931


namespace NUMINAMATH_CALUDE_boris_candy_problem_l849_84943

/-- Given the initial candy count, amount eaten by daughter, number of bowls, 
    and final count in one bowl, calculate how many pieces Boris took from each bowl. -/
theorem boris_candy_problem (initial_candy : ℕ) (daughter_ate : ℕ) (num_bowls : ℕ) (final_bowl_count : ℕ)
  (h1 : initial_candy = 100)
  (h2 : daughter_ate = 8)
  (h3 : num_bowls = 4)
  (h4 : final_bowl_count = 20)
  (h5 : num_bowls > 0) :
  let remaining_candy := initial_candy - daughter_ate
  let candy_per_bowl := remaining_candy / num_bowls
  candy_per_bowl - final_bowl_count = 3 := by sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l849_84943


namespace NUMINAMATH_CALUDE_face_vertex_assignment_l849_84965

-- Define a planar bipartite graph
class PlanarBipartiteGraph (G : Type) where
  -- Add necessary properties for planar bipartite graphs
  is_planar : Bool
  is_bipartite : Bool

-- Define faces and vertices of a graph
def faces (G : Type) [PlanarBipartiteGraph G] : Set G := sorry
def vertices (G : Type) [PlanarBipartiteGraph G] : Set G := sorry

-- Theorem statement
theorem face_vertex_assignment {G : Type} [PlanarBipartiteGraph G] :
  ∃ f : faces G → vertices G, Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_face_vertex_assignment_l849_84965


namespace NUMINAMATH_CALUDE_min_value_theorem_l849_84934

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 / b = 1) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), 2 / a + 2 * b ≥ x → m ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l849_84934


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l849_84960

theorem prime_divisibility_problem (p q : ℕ) : 
  Prime p → Prime q → p < 2005 → q < 2005 → 
  (q ∣ p^2 + 4) → (p ∣ q^2 + 4) → 
  p = 2 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l849_84960


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_has_10_sides_l849_84921

theorem regular_polygon_with_144_degree_angle_has_10_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (n - 2) * 180 / n = 144 →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_has_10_sides_l849_84921


namespace NUMINAMATH_CALUDE_parakeets_fed_sixty_cups_l849_84942

/-- The number of parakeets fed with a given amount of bird seed -/
def parakeets_fed (cups : ℕ) : ℕ :=
  sorry

theorem parakeets_fed_sixty_cups :
  parakeets_fed 60 = 20 :=
by
  sorry

/-- Assumption: 30 cups of bird seed feed 10 parakeets for 5 days -/
axiom feed_ratio : parakeets_fed 30 = 10

/-- The number of parakeets fed is directly proportional to the amount of bird seed -/
axiom linear_feed : ∀ (c₁ c₂ : ℕ), c₁ ≠ 0 → c₂ ≠ 0 →
  (parakeets_fed c₁ : ℚ) / c₁ = (parakeets_fed c₂ : ℚ) / c₂

end NUMINAMATH_CALUDE_parakeets_fed_sixty_cups_l849_84942


namespace NUMINAMATH_CALUDE_crayons_left_l849_84945

/-- Represents the number of crayons Mary has -/
structure Crayons where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons -/
def total_crayons (c : Crayons) : Nat :=
  c.green + c.blue

/-- Represents the number of crayons Mary gives away -/
structure CrayonsGiven where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons given away -/
def total_given (g : CrayonsGiven) : Nat :=
  g.green + g.blue

/-- Theorem: Mary has 9 crayons left after giving some away -/
theorem crayons_left (initial : Crayons) (given : CrayonsGiven) 
  (h1 : initial.green = 5)
  (h2 : initial.blue = 8)
  (h3 : given.green = 3)
  (h4 : given.blue = 1) :
  total_crayons initial - total_given given = 9 := by
  sorry


end NUMINAMATH_CALUDE_crayons_left_l849_84945


namespace NUMINAMATH_CALUDE_joy_reading_rate_l849_84999

/-- Represents Joy's reading rate in pages per hour -/
def reading_rate (pages_per_20min : ℕ) (pages_per_5hours : ℕ) : ℚ :=
  (pages_per_20min * 3)

/-- Theorem stating Joy's reading rate is 24 pages per hour -/
theorem joy_reading_rate :
  reading_rate 8 120 = 24 := by sorry

end NUMINAMATH_CALUDE_joy_reading_rate_l849_84999


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l849_84991

theorem not_necessary_nor_sufficient_condition (a : ℝ) :
  ¬(∀ x : ℝ, a * x^2 + a * x - 1 < 0 ↔ a < 0) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l849_84991


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l849_84961

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(2 : ℝ) / (x + 3)⌉ 
  else if x < -3 then ⌊(2 : ℝ) / (x + 3)⌋ 
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ¬ ∃ (x : ℝ), g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l849_84961


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l849_84937

-- Define the sum of the first n even numbers
def sumEven (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of the first n odd numbers
def sumOdd (n : ℕ) : ℕ := n^2

-- State the theorem
theorem even_odd_sum_difference : sumEven 100 - sumOdd 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l849_84937


namespace NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l849_84977

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) → n ≥ 169 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) ∧ n = 169 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l849_84977


namespace NUMINAMATH_CALUDE_sum_of_floors_l849_84923

def floor (x : ℚ) : ℤ := Int.floor x

theorem sum_of_floors : 
  (floor (2017 * 3 / 11 : ℚ)) + 
  (floor (2017 * 4 / 11 : ℚ)) + 
  (floor (2017 * 5 / 11 : ℚ)) + 
  (floor (2017 * 6 / 11 : ℚ)) + 
  (floor (2017 * 7 / 11 : ℚ)) + 
  (floor (2017 * 8 / 11 : ℚ)) = 6048 := by
sorry

end NUMINAMATH_CALUDE_sum_of_floors_l849_84923


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l849_84972

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l849_84972


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l849_84992

theorem unique_root_in_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = -x^3 - x) →
  m ≤ n →
  f m * f n < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l849_84992


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l849_84954

/-- Calculates the total amount collected from ticket sales given the number of adults and children, and their respective ticket prices. -/
def totalTicketSales (numAdults numChildren adultPrice childPrice : ℕ) : ℕ :=
  numAdults * adultPrice + numChildren * childPrice

/-- Theorem stating that given the specific conditions of the problem, the total ticket sales amount to $246. -/
theorem theater_ticket_sales :
  let adultPrice : ℕ := 11
  let childPrice : ℕ := 10
  let totalAttendees : ℕ := 23
  let numChildren : ℕ := 7
  let numAdults : ℕ := totalAttendees - numChildren
  totalTicketSales numAdults numChildren adultPrice childPrice = 246 :=
by
  sorry

#check theater_ticket_sales

end NUMINAMATH_CALUDE_theater_ticket_sales_l849_84954


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l849_84929

theorem gasoline_tank_capacity : 
  ∀ (capacity : ℝ),
  (5/6 : ℝ) * capacity - (1/3 : ℝ) * capacity = 20 →
  capacity = 40 := by
sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l849_84929


namespace NUMINAMATH_CALUDE_square_division_l849_84989

theorem square_division (s : ℝ) (w : ℝ) (h : w = 5) :
  ∃ (a b c d e : ℝ),
    s = 20 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + w = s ∧
    c + d = s ∧
    a * c = b * d ∧
    a * c = w * e ∧
    a * c = (s - c) * (s - a - b) :=
by sorry

#check square_division

end NUMINAMATH_CALUDE_square_division_l849_84989


namespace NUMINAMATH_CALUDE_dog_weight_l849_84909

theorem dog_weight (d k r : ℚ) 
  (total_weight : d + k + r = 40)
  (dog_rabbit_weight : d + r = 2 * k)
  (dog_kitten_weight : d + k = r) : 
  d = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_l849_84909


namespace NUMINAMATH_CALUDE_inequality_proof_l849_84926

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l849_84926


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l849_84932

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ a ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l849_84932


namespace NUMINAMATH_CALUDE_system_solution_l849_84951

theorem system_solution (x y : ℝ) (eq1 : 2*x + y = 5) (eq2 : x + 2*y = 4) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l849_84951


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l849_84914

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_participants : ℕ) 
  (physics_participants : ℕ) 
  (no_competition_participants : ℕ) 
  (h1 : total_students = 40)
  (h2 : math_participants = 31)
  (h3 : physics_participants = 20)
  (h4 : no_competition_participants = 8) :
  total_students = math_participants + physics_participants + no_competition_participants - 19 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l849_84914


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l849_84967

/-- The parabola function -/
def f (x : ℝ) : ℝ := (2 - x) * x

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of the parabola y = (2-x)x is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l849_84967
