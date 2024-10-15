import Mathlib

namespace NUMINAMATH_CALUDE_tip_to_cost_ratio_l3308_330844

def pizza_order (boxes : ℕ) (cost_per_box : ℚ) (money_given : ℚ) (change_received : ℚ) : ℚ × ℚ :=
  let total_cost := boxes * cost_per_box
  let amount_paid := money_given - change_received
  let tip := amount_paid - total_cost
  (tip, total_cost)

theorem tip_to_cost_ratio : 
  let (tip, total_cost) := pizza_order 5 7 100 60
  (tip : ℚ) / total_cost = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_tip_to_cost_ratio_l3308_330844


namespace NUMINAMATH_CALUDE_pages_ratio_is_one_to_two_l3308_330869

-- Define the total number of pages in the book
def total_pages : ℕ := 120

-- Define the number of pages read yesterday
def pages_read_yesterday : ℕ := 12

-- Define the number of pages read today
def pages_read_today : ℕ := 2 * pages_read_yesterday

-- Define the number of pages to be read tomorrow
def pages_to_read_tomorrow : ℕ := 42

-- Theorem statement
theorem pages_ratio_is_one_to_two :
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  (pages_to_read_tomorrow : ℚ) / remaining_pages = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pages_ratio_is_one_to_two_l3308_330869


namespace NUMINAMATH_CALUDE_correct_product_with_decimals_l3308_330894

theorem correct_product_with_decimals :
  let x : ℚ := 0.85
  let y : ℚ := 3.25
  let product_without_decimals : ℕ := 27625
  x * y = 2.7625 :=
by sorry

end NUMINAMATH_CALUDE_correct_product_with_decimals_l3308_330894


namespace NUMINAMATH_CALUDE_min_sum_with_constraints_l3308_330849

theorem min_sum_with_constraints (x y z : ℝ) 
  (hx : x ≥ 5) (hy : y ≥ 6) (hz : z ≥ 7) 
  (h_sum_sq : x^2 + y^2 + z^2 ≥ 125) : 
  x + y + z ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 5 ∧ y₀ ≥ 6 ∧ z₀ ≥ 7 ∧ 
    x₀^2 + y₀^2 + z₀^2 ≥ 125 ∧ 
    x₀ + y₀ + z₀ = 19 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraints_l3308_330849


namespace NUMINAMATH_CALUDE_school_average_age_l3308_330855

theorem school_average_age 
  (total_students : ℕ) 
  (boys_avg_age girls_avg_age : ℚ) 
  (num_girls : ℕ) :
  total_students = 640 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 160 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
sorry

end NUMINAMATH_CALUDE_school_average_age_l3308_330855


namespace NUMINAMATH_CALUDE_centroid_coincidence_l3308_330815

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the centroid of a triangle -/
def triangleCentroid (t : Triangle) : Point := sorry

/-- Theorem: The centroid of a triangle coincides with the centroid of its subtriangles -/
theorem centroid_coincidence (ABC : Triangle) : 
  let D : Point := sorry -- D is the foot of the altitude from C to AB
  let ACD : Triangle := ⟨ABC.A, ABC.C, D⟩
  let BCD : Triangle := ⟨ABC.B, ABC.C, D⟩
  let M1 : Point := triangleCentroid ACD
  let M2 : Point := triangleCentroid BCD
  let Z : Point := triangleCentroid ABC
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    Z.x = t * M1.x + (1 - t) * M2.x ∧
    Z.y = t * M1.y + (1 - t) * M2.y ∧
    t = (triangleArea BCD) / (triangleArea ACD + triangleArea BCD) :=
by sorry

end NUMINAMATH_CALUDE_centroid_coincidence_l3308_330815


namespace NUMINAMATH_CALUDE_correct_calculation_l3308_330817

theorem correct_calculation (x : ℝ) : 3 * x - 10 = 50 → 3 * x + 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3308_330817


namespace NUMINAMATH_CALUDE_min_crossing_time_l3308_330852

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  peopleOnIsland : List Person
  peopleOnMainland : List Person
  lampOnIsland : Bool
  totalTime : ℕ

/-- Defines the initial state of the problem -/
def initialState : BridgeState where
  peopleOnIsland := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 8 },
    { crossingTime := 16 }
  ]
  peopleOnMainland := []
  lampOnIsland := true
  totalTime := 0

/-- Represents a valid move across the bridge -/
inductive Move
  | cross (p1 : Person) (p2 : Option Person)
  | returnLamp (p : Person)

/-- Applies a move to the current state -/
def applyMove (state : BridgeState) (move : Move) : BridgeState :=
  sorry

/-- Checks if all people have crossed to the mainland -/
def isComplete (state : BridgeState) : Bool :=
  sorry

/-- Theorem: The minimum time required to cross the bridge is 30 minutes -/
theorem min_crossing_time (initialState : BridgeState) :
  ∃ (moves : List Move), 
    (moves.foldl applyMove initialState).totalTime = 30 ∧ 
    isComplete (moves.foldl applyMove initialState) ∧
    ∀ (otherMoves : List Move), 
      isComplete (otherMoves.foldl applyMove initialState) → 
      (otherMoves.foldl applyMove initialState).totalTime ≥ 30 :=
  sorry

end NUMINAMATH_CALUDE_min_crossing_time_l3308_330852


namespace NUMINAMATH_CALUDE_dans_initial_amount_l3308_330800

/-- Dan's initial amount of money -/
def initial_amount : ℝ := 4

/-- The cost of the candy bar -/
def candy_cost : ℝ := 1

/-- The amount Dan had left after buying the candy bar -/
def remaining_amount : ℝ := 3

/-- Theorem stating that Dan's initial amount equals the sum of the remaining amount and the candy cost -/
theorem dans_initial_amount : initial_amount = remaining_amount + candy_cost := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_amount_l3308_330800


namespace NUMINAMATH_CALUDE_johns_spending_l3308_330835

theorem johns_spending (total : ℚ) 
  (h1 : total = 24)
  (h2 : total * (1/4) + total * (1/3) + 6 + bakery = total) : 
  bakery / total = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_johns_spending_l3308_330835


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_contain_conjugate_diameters_l3308_330829

-- Define an ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

-- Define conjugate diameters of an ellipse
def conjugate_diameters (e : Ellipse) : Set (ℝ × ℝ) := sorry

-- Define the diagonals of a parallelogram
def diagonals (p : Parallelogram) : Set (ℝ × ℝ) := sorry

-- Define what it means for a parallelogram to be inscribed around an ellipse
def is_inscribed (p : Parallelogram) (e : Ellipse) : Prop := sorry

-- Theorem statement
theorem parallelogram_diagonals_contain_conjugate_diameters 
  (e : Ellipse) (p : Parallelogram) (h : is_inscribed p e) :
  diagonals p ⊆ conjugate_diameters e := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_contain_conjugate_diameters_l3308_330829


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3308_330820

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l3308_330820


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3308_330884

theorem complex_equation_solution (x : ℝ) :
  (1 - 2*Complex.I) * (x + Complex.I) = 4 - 3*Complex.I → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3308_330884


namespace NUMINAMATH_CALUDE_rod_pieces_count_l3308_330839

/-- The length of the rod in meters -/
def rod_length_m : ℝ := 34

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 85

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length_m * m_to_cm) / piece_length_cm⌋ = 40 := by sorry

end NUMINAMATH_CALUDE_rod_pieces_count_l3308_330839


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3308_330846

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3308_330846


namespace NUMINAMATH_CALUDE_expression_evaluation_l3308_330897

theorem expression_evaluation :
  let a : ℚ := -1/6
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3308_330897


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3308_330883

/-- Given a line segment with one endpoint at (6, 1) and midpoint at (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 1) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 17

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 1) (5, 7) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3308_330883


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3308_330860

-- Define a cube with 8 vertices
structure Cube :=
  (vertices : Fin 8 → ℝ)

-- Define the sum of numbers on a face
def face_sum (c : Cube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  c.vertices v1 + c.vertices v2 + c.vertices v3 + c.vertices v4

-- Define the sum of all face sums
def total_face_sum (c : Cube) : ℝ :=
  face_sum c 0 1 2 3 +
  face_sum c 0 1 4 5 +
  face_sum c 0 3 4 7 +
  face_sum c 1 2 5 6 +
  face_sum c 2 3 6 7 +
  face_sum c 4 5 6 7

-- Define the sum of all vertex values
def vertex_sum (c : Cube) : ℝ :=
  c.vertices 0 + c.vertices 1 + c.vertices 2 + c.vertices 3 +
  c.vertices 4 + c.vertices 5 + c.vertices 6 + c.vertices 7

-- Theorem statement
theorem cube_sum_theorem (c : Cube) :
  total_face_sum c = 2019 → vertex_sum c = 673 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3308_330860


namespace NUMINAMATH_CALUDE_circles_intersect_l3308_330865

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r1 r2 d : ℝ) : Prop :=
  d > |r1 - r2| ∧ d < r1 + r2

/-- Given two circles with radii 5 and 8, and distance between centers 8, 
    prove that they are intersecting. -/
theorem circles_intersect : are_intersecting 5 8 8 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3308_330865


namespace NUMINAMATH_CALUDE_jane_final_crayons_l3308_330898

/-- The number of crayons Jane ends up with after the hippopotamus incident and finding additional crayons -/
def final_crayon_count (x y : ℕ) : ℕ :=
  y - x + 15

/-- Theorem stating that given the conditions, Jane ends up with 95 crayons -/
theorem jane_final_crayons :
  let x : ℕ := 7  -- number of crayons eaten by the hippopotamus
  let y : ℕ := 87 -- number of crayons Jane had initially
  final_crayon_count x y = 95 := by
  sorry

end NUMINAMATH_CALUDE_jane_final_crayons_l3308_330898


namespace NUMINAMATH_CALUDE_probability_three_blue_jellybeans_l3308_330812

def total_jellybeans : ℕ := 20
def initial_blue_jellybeans : ℕ := 10

def probability_all_blue : ℚ := 2 / 19

theorem probability_three_blue_jellybeans :
  let p1 := initial_blue_jellybeans / total_jellybeans
  let p2 := (initial_blue_jellybeans - 1) / (total_jellybeans - 1)
  let p3 := (initial_blue_jellybeans - 2) / (total_jellybeans - 2)
  p1 * p2 * p3 = probability_all_blue := by
  sorry

end NUMINAMATH_CALUDE_probability_three_blue_jellybeans_l3308_330812


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3308_330879

theorem simplify_trig_expression :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 
    (1 / 2) * Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3308_330879


namespace NUMINAMATH_CALUDE_student_correct_problems_l3308_330810

/-- Represents the number of problems solved correctly by a student. -/
def correct_problems (total : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (final_score : ℤ) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the number of correctly solved problems is 31. -/
theorem student_correct_problems :
  correct_problems 80 5 3 8 = 31 := by sorry

end NUMINAMATH_CALUDE_student_correct_problems_l3308_330810


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3308_330823

/-- Given two regular polygons with equal perimeters, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 → n > 0 → 50 * (3 * s) = n * s → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3308_330823


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3308_330826

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3308_330826


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3308_330832

theorem greatest_power_of_three_in_factorial :
  (∃ (n : ℕ), n = 9 ∧ 
   ∀ (k : ℕ), 3^k ∣ Nat.factorial 22 → k ≤ n) ∧
   (∀ (m : ℕ), m > 9 → ¬(3^m ∣ Nat.factorial 22)) := by
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3308_330832


namespace NUMINAMATH_CALUDE_inscribed_circle_cycle_l3308_330813

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def inscribed_circle (T : Triangle) (i : ℕ) : Circle :=
  sorry

theorem inscribed_circle_cycle (T : Triangle) :
  inscribed_circle T 7 = inscribed_circle T 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_cycle_l3308_330813


namespace NUMINAMATH_CALUDE_exactly_three_statements_true_l3308_330859

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define points A, B, M, and N
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Statement 1
def statement_1 : Prop :=
  polyline_distance A.1 A.2 B.1 B.2 = 5

-- Statement 2
def statement_2 : Prop :=
  ∃ (S : Set (ℝ × ℝ)), S = {p : ℝ × ℝ | polyline_distance p.1 p.2 0 0 = 1} ∧
  ¬(∃ (center : ℝ × ℝ) (radius : ℝ), S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2})

-- Statement 3
def statement_3 : Prop :=
  ∀ (C : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) →
    polyline_distance A.1 A.2 C.1 C.2 + polyline_distance C.1 C.2 B.1 B.2 = polyline_distance A.1 A.2 B.1 B.2

-- Statement 4
def statement_4 : Prop :=
  {p : ℝ × ℝ | polyline_distance p.1 p.2 M.1 M.2 = polyline_distance p.1 p.2 N.1 N.2} =
  {p : ℝ × ℝ | p.1 = 0}

-- Main theorem
theorem exactly_three_statements_true :
  (statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ statement_4) := by sorry

end NUMINAMATH_CALUDE_exactly_three_statements_true_l3308_330859


namespace NUMINAMATH_CALUDE_estimated_y_at_x_100_l3308_330819

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 1.43 * x + 257

-- Theorem statement
theorem estimated_y_at_x_100 :
  regression_equation 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_at_x_100_l3308_330819


namespace NUMINAMATH_CALUDE_only_B_and_C_have_inverses_l3308_330858

-- Define the set of functions
inductive Function : Type
| A | B | C | D | E

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop :=
  match f with
  | Function.A => False
  | Function.B => True
  | Function.C => True
  | Function.D => False
  | Function.E => False

-- Theorem statement
theorem only_B_and_C_have_inverses :
  ∀ f : Function, has_inverse f ↔ (f = Function.B ∨ f = Function.C) :=
by sorry

end NUMINAMATH_CALUDE_only_B_and_C_have_inverses_l3308_330858


namespace NUMINAMATH_CALUDE_brick_length_calculation_l3308_330809

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The problem statement -/
theorem brick_length_calculation (wall : Dimensions) (brick_width : ℝ) (brick_height : ℝ) 
    (num_bricks : ℕ) (h_wall : wall = ⟨800, 600, 22.5⟩) (h_brick_width : brick_width = 11.25) 
    (h_brick_height : brick_height = 6) (h_num_bricks : num_bricks = 2000) :
    ∃ (brick_length : ℝ), 
      volume wall = num_bricks * volume ⟨brick_length, brick_width, brick_height⟩ ∧ 
      brick_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l3308_330809


namespace NUMINAMATH_CALUDE_derivative_of_f_l3308_330880

noncomputable def f (x : ℝ) : ℝ := (2^x * (Real.sin x + Real.cos x * Real.log 2)) / (1 + Real.log 2 ^ 2)

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3308_330880


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3308_330808

theorem cubic_roots_relation (p q r : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 4*x^2 + 5*x - 14 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 34 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3308_330808


namespace NUMINAMATH_CALUDE_number_comparison_l3308_330881

def A : ℕ := 888888888888888888888  -- 19 eights
def B : ℕ := 333333333333333333333333333333333333333333333333333333333333333333333  -- 68 threes
def C : ℕ := 444444444444444444444  -- 19 fours
def D : ℕ := 666666666666666666666666666666666666666666666666666666666666666666667  -- 67 sixes and one seven

theorem number_comparison : C * D - A * B = 444444444444444444444 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l3308_330881


namespace NUMINAMATH_CALUDE_vector_perpendicular_value_l3308_330872

theorem vector_perpendicular_value (k : ℝ) : 
  let a : (ℝ × ℝ) := (3, 1)
  let b : (ℝ × ℝ) := (1, 3)
  let c : (ℝ × ℝ) := (k, -2)
  (((a.1 - c.1) * b.1 + (a.2 - c.2) * b.2) = 0) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_value_l3308_330872


namespace NUMINAMATH_CALUDE_cost_to_fill_displays_l3308_330842

/-- Represents the inventory and pricing of a jewelry store -/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  bracelet_capacity : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ

/-- Calculates the total cost to fill all displays in the jewelry store -/
def total_cost_to_fill (store : JewelryStore) : ℕ :=
  ((store.necklace_capacity - store.current_necklaces) * store.necklace_price) +
  ((store.ring_capacity - store.current_rings) * store.ring_price) +
  ((store.bracelet_capacity - store.current_bracelets) * store.bracelet_price)

/-- Theorem stating that the total cost to fill all displays is $183 -/
theorem cost_to_fill_displays (store : JewelryStore) 
  (h1 : store.necklace_capacity = 12)
  (h2 : store.current_necklaces = 5)
  (h3 : store.ring_capacity = 30)
  (h4 : store.current_rings = 18)
  (h5 : store.bracelet_capacity = 15)
  (h6 : store.current_bracelets = 8)
  (h7 : store.necklace_price = 4)
  (h8 : store.ring_price = 10)
  (h9 : store.bracelet_price = 5) :
  total_cost_to_fill store = 183 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_fill_displays_l3308_330842


namespace NUMINAMATH_CALUDE_sara_letters_count_l3308_330847

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count :
  total_letters = 33 := by sorry

end NUMINAMATH_CALUDE_sara_letters_count_l3308_330847


namespace NUMINAMATH_CALUDE_divisors_of_30_l3308_330801

/-- The number of integer divisors (positive and negative) of 30 -/
def number_of_divisors_of_30 : ℕ :=
  (Finset.filter (· ∣ 30) (Finset.range 31)).card * 2

/-- Theorem stating that the number of integer divisors of 30 is 16 -/
theorem divisors_of_30 : number_of_divisors_of_30 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_30_l3308_330801


namespace NUMINAMATH_CALUDE_cos_90_degrees_zero_l3308_330814

theorem cos_90_degrees_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_zero_l3308_330814


namespace NUMINAMATH_CALUDE_candy_final_temperature_l3308_330892

/-- Calculates the final temperature of a candy mixture given the initial conditions and rates. -/
theorem candy_final_temperature 
  (initial_temp : ℝ) 
  (max_temp : ℝ) 
  (heating_rate : ℝ) 
  (cooling_rate : ℝ) 
  (total_time : ℝ) 
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let heating_time := (max_temp - initial_temp) / heating_rate
  let cooling_time := total_time - heating_time
  let temp_drop := cooling_rate * cooling_time
  max_temp - temp_drop = 170 := by
  sorry

end NUMINAMATH_CALUDE_candy_final_temperature_l3308_330892


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3308_330803

theorem min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x^2 + y^2 = x*y*(x^2*y^2 + 2)) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a*b*(a^2*b^2 + 2) → 
  1/x + 1/y ≤ 1/a + 1/b :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3308_330803


namespace NUMINAMATH_CALUDE_function_transformation_l3308_330806

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3308_330806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3308_330838

/-- An arithmetic sequence {a_n} with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 4 + seq.a 5 = 24) 
  (h2 : seq.S 6 = 48) : 
  common_difference seq = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3308_330838


namespace NUMINAMATH_CALUDE_expansion_coefficient_condition_l3308_330864

/-- The coefficient of the r-th term in the expansion of (2x + 1/x)^n -/
def coefficient (n : ℕ) (r : ℕ) : ℚ :=
  2^(n-r) * (n.choose r)

theorem expansion_coefficient_condition (n : ℕ) :
  (coefficient n 2 = 2 * coefficient n 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_condition_l3308_330864


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3308_330877

theorem quadratic_root_property (a : ℝ) (h : a^2 - 2*a - 3 = 0) : a^2 - 2*a + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3308_330877


namespace NUMINAMATH_CALUDE_base_eight_solution_l3308_330836

theorem base_eight_solution : ∃! (b : ℕ), b > 1 ∧ (3 * b + 2)^2 = b^3 + b + 4 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_solution_l3308_330836


namespace NUMINAMATH_CALUDE_invalid_atomic_number_difference_l3308_330841

/-- Represents a period in the periodic table -/
inductive Period
| Second
| Third
| Fourth
| Fifth
| Sixth

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ
  period : Period

/-- The difference in atomic numbers between elements in groups VIA and IA in the same period -/
def atomicNumberDifference (p : Period) : ℕ :=
  match p with
  | Period.Second => 5
  | Period.Third => 5
  | Period.Fourth => 15
  | Period.Fifth => 15
  | Period.Sixth => 29

theorem invalid_atomic_number_difference (X Y : Element) 
  (h1 : X.period = Y.period)
  (h2 : Y.atomicNumber = X.atomicNumber + atomicNumberDifference X.period) :
  Y.atomicNumber - X.atomicNumber ≠ 9 := by
  sorry

#check invalid_atomic_number_difference

end NUMINAMATH_CALUDE_invalid_atomic_number_difference_l3308_330841


namespace NUMINAMATH_CALUDE_polycarp_kolka_numbers_l3308_330843

/-- The smallest 5-digit number composed of distinct even digits -/
def polycarp_number : ℕ := 20468

/-- Kolka's incorrect 5-digit number -/
def kolka_number : ℕ := 20486

/-- Checks if a number is a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Checks if a number is composed of distinct even digits -/
def has_distinct_even_digits (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    Even a ∧ Even b ∧ Even c ∧ Even d ∧ Even e

theorem polycarp_kolka_numbers :
  (is_five_digit polycarp_number) ∧
  (has_distinct_even_digits polycarp_number) ∧
  (∀ n : ℕ, is_five_digit n → has_distinct_even_digits n → n ≥ polycarp_number) ∧
  (is_five_digit kolka_number) ∧
  (has_distinct_even_digits kolka_number) ∧
  (kolka_number - polycarp_number < 100) ∧
  (kolka_number ≠ polycarp_number) →
  kolka_number = 20486 :=
by sorry

end NUMINAMATH_CALUDE_polycarp_kolka_numbers_l3308_330843


namespace NUMINAMATH_CALUDE_john_received_120_l3308_330840

def grandpa_amount : ℕ := 30

def grandma_amount : ℕ := 3 * grandpa_amount

def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_received_120_l3308_330840


namespace NUMINAMATH_CALUDE_negation_of_implication_l3308_330853

theorem negation_of_implication (a : ℝ) :
  ¬(a > -3 → a > -6) ↔ (a ≤ -3 → a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3308_330853


namespace NUMINAMATH_CALUDE_expand_product_l3308_330861

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3308_330861


namespace NUMINAMATH_CALUDE_problem_solution_l3308_330828

theorem problem_solution :
  let x : ℤ := 5
  let y : ℤ := x + 3
  let z : ℤ := 3 * y + 1
  z = 25 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3308_330828


namespace NUMINAMATH_CALUDE_correct_calculation_l3308_330878

theorem correct_calculation (x : ℚ) (h : 15 / x = 5) : 21 / x = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3308_330878


namespace NUMINAMATH_CALUDE_max_cookies_eaten_l3308_330857

/-- Given two people sharing 30 cookies, where one eats twice as many as the other,
    the maximum number of cookies the person eating fewer can have is 10. -/
theorem max_cookies_eaten (total : ℕ) (andy bella : ℕ) : 
  total = 30 →
  bella = 2 * andy →
  andy + bella = total →
  andy ≤ 10 ∧ ∃ (n : ℕ), n = 10 ∧ n ≤ andy :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l3308_330857


namespace NUMINAMATH_CALUDE_lamplighter_monkey_speed_l3308_330827

/-- A Lamplighter monkey's movement characteristics -/
structure LamplighterMonkey where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ
  totalDistance : ℝ

/-- Theorem: Given the characteristics of a Lamplighter monkey's movement,
    prove that its running speed is 15 feet per second -/
theorem lamplighter_monkey_speed (monkey : LamplighterMonkey)
  (h1 : monkey.swingingSpeed = 10)
  (h2 : monkey.runningTime = 5)
  (h3 : monkey.swingingTime = 10)
  (h4 : monkey.totalDistance = 175) :
  monkey.runningSpeed = 15 := by
  sorry

#check lamplighter_monkey_speed

end NUMINAMATH_CALUDE_lamplighter_monkey_speed_l3308_330827


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l3308_330824

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_monotone_increasing :
  ∀ x y, x ≥ Real.sqrt 2 / 2 → y ≥ Real.sqrt 2 / 2 → x < y → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l3308_330824


namespace NUMINAMATH_CALUDE_expression_factorization_l3308_330834

theorem expression_factorization (x : ℝ) : 
  (18 * x^6 + 50 * x^4 - 8) - (2 * x^6 - 6 * x^4 - 8) = 8 * x^4 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3308_330834


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3308_330899

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3308_330899


namespace NUMINAMATH_CALUDE_min_value_theorem_l3308_330891

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (4/a + 1/b) ≥ 18 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3308_330891


namespace NUMINAMATH_CALUDE_journey_time_proof_l3308_330856

-- Define the journey segments
inductive Segment
| Uphill
| Flat
| Downhill

-- Define the journey parameters
def total_distance : ℝ := 50
def uphill_speed : ℝ := 3

-- Define the ratios
def length_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 1
  | .Flat => 2
  | .Downhill => 3

def time_ratio (s : Segment) : ℝ :=
  match s with
  | .Uphill => 4
  | .Flat => 5
  | .Downhill => 6

-- Define the theorem
theorem journey_time_proof :
  let total_ratio : ℝ := (length_ratio .Uphill) + (length_ratio .Flat) + (length_ratio .Downhill)
  let uphill_distance : ℝ := total_distance * (length_ratio .Uphill) / total_ratio
  let uphill_time : ℝ := uphill_distance / uphill_speed
  let time_ratio_sum : ℝ := (time_ratio .Uphill) + (time_ratio .Flat) + (time_ratio .Downhill)
  let total_time : ℝ := uphill_time * time_ratio_sum / (time_ratio .Uphill)
  total_time = 10 + 5 / 12 :=
by sorry


end NUMINAMATH_CALUDE_journey_time_proof_l3308_330856


namespace NUMINAMATH_CALUDE_remainder_101_103_div_11_l3308_330882

theorem remainder_101_103_div_11 : (101 * 103) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_103_div_11_l3308_330882


namespace NUMINAMATH_CALUDE_circle_line_distance_difference_l3308_330831

/-- Given a circle with equation x² + (y-1)² = 1 and a line x - y - 2 = 0,
    the difference between the maximum and minimum distances from points
    on the circle to the line is (√2)/2 + 1. -/
theorem circle_line_distance_difference :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  let max_distance := Real.sqrt 8
  let min_distance := (3 * Real.sqrt 2) / 2 - 1
  max_distance - min_distance = Real.sqrt 2 / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_distance_difference_l3308_330831


namespace NUMINAMATH_CALUDE_triangle_properties_l3308_330830

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A →
  (A = π / 3) ∧
  (a = 2 → (1 / 2) * b * c * Real.sin A = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3308_330830


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3308_330854

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- Define the first derivative of f
def f' (a b c x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Define the second derivative of f
def f'' (a b : ℝ) (x : ℝ) : ℝ := 6 * a * x + 2 * b

-- State the theorem
theorem cubic_function_properties (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, x = 1 ∨ x = -1 → f' a b c x = 0) →
  f a b c 1 = -1 →
  a = -1/2 ∧ b = 0 ∧ c = 3/2 ∧
  f'' a b 1 < 0 ∧ f'' a b (-1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3308_330854


namespace NUMINAMATH_CALUDE_sqrt_eq_condition_l3308_330863

theorem sqrt_eq_condition (x y : ℝ) (h : x * y ≠ 0) :
  Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0 := by
sorry

end NUMINAMATH_CALUDE_sqrt_eq_condition_l3308_330863


namespace NUMINAMATH_CALUDE_real_part_of_x_l3308_330885

-- Define the variables and their types
variable (x : ℂ) -- x is a complex number
variable (y z : ℝ) -- y and z are real numbers
variable (p q : ℕ) -- p and q are natural numbers (we'll define them as prime later)
variable (n m : ℕ) -- n and m are non-negative integers
variable (k : ℕ) -- k is a natural number (we'll define it as odd prime later)

-- Define the conditions
axiom p_prime : Nat.Prime p
axiom q_prime : Nat.Prime q
axiom p_ne_q : p ≠ q
axiom k_odd_prime : Nat.Prime k ∧ k % 2 = 1
axiom least_p_q : ∀ p' q', Nat.Prime p' → Nat.Prime q' → p' ≠ q' → (p < p' ∨ q < q')

-- Define the specific values
axiom n_val : n = 2
axiom m_val : m = 3
axiom y_val : y = 5
axiom z_val : z = 10

-- Define the system of equations
axiom eq1 : x^n / (12 * ↑p * ↑q) = ↑k
axiom eq2 : x^m + y = z

-- Theorem to prove
theorem real_part_of_x :
  ∃ r : ℝ, (r = 6 * Real.sqrt 6 ∨ r = -6 * Real.sqrt 6) ∧ x.re = r :=
sorry

end NUMINAMATH_CALUDE_real_part_of_x_l3308_330885


namespace NUMINAMATH_CALUDE_cube_of_thousands_l3308_330873

theorem cube_of_thousands (n : ℕ) : n = (n / 1000)^3 ↔ n = 32768 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_thousands_l3308_330873


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3308_330850

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (18 * x - 2)^(1/3) + (16 * x + 2)^(1/3) + (-72 * x)^(1/3) - 6 * x^(1/3)
  {x : ℝ | f x = 0} = {0, 1/9, -1/8} := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3308_330850


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3308_330805

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem stating that the axis of symmetry is x = 1
theorem axis_of_symmetry :
  ∀ y : ℝ, ∃ x : ℝ, parabola (x + 1) = parabola (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3308_330805


namespace NUMINAMATH_CALUDE_quadrilateral_rod_count_l3308_330807

theorem quadrilateral_rod_count : 
  let a : ℕ := 5
  let b : ℕ := 12
  let c : ℕ := 20
  let valid_length (d : ℕ) : Prop := 
    1 ≤ d ∧ d ≤ 40 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a
  (Finset.filter valid_length (Finset.range 41)).card = 30 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_rod_count_l3308_330807


namespace NUMINAMATH_CALUDE_average_equals_one_l3308_330821

theorem average_equals_one (x : ℝ) : 
  (5 + (-1) + (-2) + x) / 4 = 1 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_average_equals_one_l3308_330821


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3308_330896

/-- 
For a quadratic expression of the form x^2 - 16x + k to be the square of a binomial,
k must equal 64.
-/
theorem quadratic_is_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 16*x + k = (a*x + b)^2) ↔ k = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l3308_330896


namespace NUMINAMATH_CALUDE_find_m_l3308_330848

theorem find_m : ∃ m : ℕ, 
  (1 ^ (m + 1) / 5 ^ (m + 1)) * (1 ^ 18 / 4 ^ 18) = 1 / (2 * 10 ^ 35) ∧ m = 34 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3308_330848


namespace NUMINAMATH_CALUDE_glorias_turtle_finish_time_l3308_330895

/-- The finish time of Gloria's turtle in the Key West Turtle Race -/
def glorias_turtle_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finish time is 8 minutes -/
theorem glorias_turtle_finish_time :
  ∃ (gretas_time georges_time : ℕ),
    gretas_time = 6 ∧
    georges_time = gretas_time - 2 ∧
    glorias_turtle_time gretas_time georges_time = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_glorias_turtle_finish_time_l3308_330895


namespace NUMINAMATH_CALUDE_fourth_power_sum_geq_four_times_product_l3308_330887

theorem fourth_power_sum_geq_four_times_product (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a^4 + b^4 + c^4 + d^4 ≥ 4 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_geq_four_times_product_l3308_330887


namespace NUMINAMATH_CALUDE_boys_playing_cards_l3308_330890

theorem boys_playing_cards (total_marble_boys : ℕ) (total_marbles : ℕ) (marbles_per_boy : ℕ) :
  total_marble_boys = 13 →
  total_marbles = 26 →
  marbles_per_boy = 2 →
  total_marbles = total_marble_boys * marbles_per_boy →
  (total_marble_boys : ℤ) - (total_marbles / marbles_per_boy : ℤ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_boys_playing_cards_l3308_330890


namespace NUMINAMATH_CALUDE_abc_inequality_l3308_330862

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3308_330862


namespace NUMINAMATH_CALUDE_hexagon_painting_arrangements_l3308_330845

/-- The number of ways to paint a hexagonal arrangement of equilateral triangles -/
def paint_arrangements : ℕ := 3^6 * 2^6

/-- The hexagonal arrangement consists of 6 inner sticks -/
def inner_sticks : ℕ := 6

/-- The number of available colors -/
def colors : ℕ := 3

/-- The number of triangles in the hexagonal arrangement -/
def triangles : ℕ := 6

/-- The number of ways to paint the inner sticks -/
def inner_stick_arrangements : ℕ := colors^inner_sticks

/-- The number of ways to complete each triangle given the two-color constraint -/
def triangle_completions : ℕ := 2^triangles

theorem hexagon_painting_arrangements :
  paint_arrangements = inner_stick_arrangements * triangle_completions :=
by sorry

end NUMINAMATH_CALUDE_hexagon_painting_arrangements_l3308_330845


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l3308_330875

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l3308_330875


namespace NUMINAMATH_CALUDE_max_value_of_f_l3308_330888

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 1, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3308_330888


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_11_gon_l3308_330874

/-- The number of sides in our regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in an n-sided regular polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in an n-sided regular polygon -/
def shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in an n-sided regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  shortest_diagonals n / total_diagonals n

/-- Theorem: The probability of selecting a shortest diagonal in a regular 11-sided polygon is 1/4 -/
theorem prob_shortest_diagonal_11_gon :
  prob_shortest_diagonal n = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_11_gon_l3308_330874


namespace NUMINAMATH_CALUDE_simplify_expression_l3308_330802

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3308_330802


namespace NUMINAMATH_CALUDE_only_height_weight_correlated_l3308_330870

/-- Represents the relationship between two variables -/
inductive Relationship
  | Functional
  | Correlated
  | Unrelated

/-- Defines the relationship between a cube's volume and its edge length -/
def cube_volume_edge_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between distance traveled and time for constant speed motion -/
def distance_time_relationship : Relationship := Relationship.Functional

/-- Defines the relationship between a person's height and eyesight -/
def height_eyesight_relationship : Relationship := Relationship.Unrelated

/-- Defines the relationship between a person's height and weight -/
def height_weight_relationship : Relationship := Relationship.Correlated

/-- Theorem stating that only height and weight have a correlation among the given pairs -/
theorem only_height_weight_correlated :
  (cube_volume_edge_relationship ≠ Relationship.Correlated) ∧
  (distance_time_relationship ≠ Relationship.Correlated) ∧
  (height_eyesight_relationship ≠ Relationship.Correlated) ∧
  (height_weight_relationship = Relationship.Correlated) :=
sorry

end NUMINAMATH_CALUDE_only_height_weight_correlated_l3308_330870


namespace NUMINAMATH_CALUDE_equation_solution_l3308_330889

theorem equation_solution : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3308_330889


namespace NUMINAMATH_CALUDE_green_peaches_count_l3308_330893

/-- The number of peaches in a basket --/
structure Basket :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)

/-- The basket with the given conditions --/
def my_basket : Basket :=
  { red := 2
  , yellow := 6
  , green := 6 + 8 }

/-- Theorem stating that the number of green peaches is 14 --/
theorem green_peaches_count (b : Basket) 
  (h1 : b.red = 2) 
  (h2 : b.yellow = 6) 
  (h3 : b.green = b.yellow + 8) : 
  b.green = 14 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l3308_330893


namespace NUMINAMATH_CALUDE_ryan_english_study_time_l3308_330816

/-- The number of hours Ryan spends on learning Chinese daily -/
def chinese_hours : ℕ := 5

/-- The number of additional hours Ryan spends on learning English compared to Chinese -/
def additional_english_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English daily -/
def english_hours : ℕ := chinese_hours + additional_english_hours

theorem ryan_english_study_time : english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_study_time_l3308_330816


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3308_330833

/-- The sum of a finite geometric series -/
def geometricSum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometricSum a r n = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3308_330833


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l3308_330886

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_eight_l3308_330886


namespace NUMINAMATH_CALUDE_first_cyclist_overtakes_second_opposite_P_l3308_330851

/-- Represents the circular runway --/
structure CircularRunway where
  radius : ℝ

/-- Represents a moving entity on the circular runway --/
structure MovingEntity where
  velocity : ℝ

/-- Represents the scenario of cyclists and pedestrian on the circular runway --/
structure RunwayScenario where
  runway : CircularRunway
  cyclist1 : MovingEntity
  cyclist2 : MovingEntity
  pedestrian : MovingEntity

/-- The main theorem stating the point where the first cyclist overtakes the second --/
theorem first_cyclist_overtakes_second_opposite_P (scenario : RunwayScenario) 
  (h1 : scenario.cyclist1.velocity > scenario.cyclist2.velocity)
  (h2 : scenario.pedestrian.velocity = (scenario.cyclist1.velocity + scenario.cyclist2.velocity) / 12)
  (h3 : ∃ t1 t2, t2 - t1 = 91 ∧ 
        t1 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity + scenario.pedestrian.velocity) ∧
        t2 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity + scenario.pedestrian.velocity))
  (h4 : ∃ t3 t4, t4 - t3 = 187 ∧
        t3 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity - scenario.pedestrian.velocity) ∧
        t4 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity - scenario.pedestrian.velocity)) :
  ∃ t : ℝ, t * scenario.cyclist1.velocity = π * scenario.runway.radius ∧
          t * scenario.cyclist2.velocity = π * scenario.runway.radius :=
by sorry

end NUMINAMATH_CALUDE_first_cyclist_overtakes_second_opposite_P_l3308_330851


namespace NUMINAMATH_CALUDE_point_coordinates_l3308_330804

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the conditions for the point
def on_x_axis (p : Point) : Prop := p.2 = 0
def right_of_origin (p : Point) : Prop := p.1 > 0
def distance_from_origin (p : Point) (d : ℝ) : Prop := p.1^2 + p.2^2 = d^2

-- Theorem statement
theorem point_coordinates :
  ∀ (p : Point),
    on_x_axis p →
    right_of_origin p →
    distance_from_origin p 3 →
    p = (3, 0) :=
by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l3308_330804


namespace NUMINAMATH_CALUDE_chord_equation_l3308_330818

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- The midpoint of the chord -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line containing the chord -/
def lies_on_chord_line (x y : ℝ) : Prop := 2*x - y - 15 = 0

theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  hyperbola x₁ y₁ →
  hyperbola x₂ y₂ →
  (x₁ + x₂) / 2 = P.1 →
  (y₁ + y₂) / 2 = P.2 →
  lies_on_chord_line x₁ y₁ ∧ lies_on_chord_line x₂ y₂ :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l3308_330818


namespace NUMINAMATH_CALUDE_positive_abc_l3308_330825

theorem positive_abc (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
sorry

end NUMINAMATH_CALUDE_positive_abc_l3308_330825


namespace NUMINAMATH_CALUDE_f_of_three_equals_nine_l3308_330871

theorem f_of_three_equals_nine (f : ℝ → ℝ) (h : ∀ x, f x = x^2) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_nine_l3308_330871


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3308_330811

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationship operators
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line
  (m n : Line) (α : Plane)
  (h1 : contained_in m α)
  (h2 : perpendicular n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3308_330811


namespace NUMINAMATH_CALUDE_harry_change_problem_l3308_330868

theorem harry_change_problem (change : ℕ) : 
  change < 100 ∧ 
  change % 50 = 2 ∧ 
  change % 5 = 4 → 
  change = 52 := by sorry

end NUMINAMATH_CALUDE_harry_change_problem_l3308_330868


namespace NUMINAMATH_CALUDE_boat_fee_ratio_l3308_330867

/-- Proves that the ratio of docking fees to license and registration fees is 3:1 given the conditions of Mitch's boat purchase. -/
theorem boat_fee_ratio :
  let total_savings : ℚ := 20000
  let boat_cost_per_foot : ℚ := 1500
  let license_fee : ℚ := 500
  let max_boat_length : ℚ := 12
  let available_for_boat : ℚ := total_savings - license_fee
  let boat_cost : ℚ := boat_cost_per_foot * max_boat_length
  let docking_fee : ℚ := available_for_boat - boat_cost
  docking_fee / license_fee = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_fee_ratio_l3308_330867


namespace NUMINAMATH_CALUDE_consecutive_integers_base_equation_l3308_330866

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

theorem consecutive_integers_base_equation :
  ∀ C D : ℕ,
  C > 0 →
  D = C + 1 →
  toBase10 154 C + toBase10 52 D = toBase10 76 (C + D) →
  C + D = 11 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_equation_l3308_330866


namespace NUMINAMATH_CALUDE_sum_of_xy_on_circle_l3308_330876

theorem sum_of_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 12*y + 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_on_circle_l3308_330876


namespace NUMINAMATH_CALUDE_inscribed_circle_max_radius_l3308_330837

/-- Given a triangle ABC with side lengths a, b, c, and area A,
    and an inscribed circle with radius r, 
    the radius r is at most (2 * A) / (a + b + c) --/
theorem inscribed_circle_max_radius 
  (a b c : ℝ) 
  (A : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hA : A > 0) 
  (h_triangle : A = a * b * c / (4 * (a * b + b * c + c * a - a * a - b * b - c * c).sqrt)) 
  (r : ℝ) 
  (hr : r > 0) 
  (h_inscribed : r * (a + b + c) ≤ 2 * A) :
  r ≤ 2 * A / (a + b + c) ∧ 
  (r = 2 * A / (a + b + c) ↔ r * (a + b + c) = 2 * A) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_max_radius_l3308_330837


namespace NUMINAMATH_CALUDE_cafe_chairs_minimum_l3308_330822

theorem cafe_chairs_minimum (indoor_tables outdoor_tables : ℕ)
  (indoor_min indoor_max outdoor_min outdoor_max : ℕ)
  (total_customers indoor_customers : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  indoor_min = 6 →
  indoor_max = 10 →
  outdoor_min = 3 →
  outdoor_max = 5 →
  total_customers = 35 →
  indoor_customers = 18 →
  indoor_min ≤ indoor_max →
  outdoor_min ≤ outdoor_max →
  indoor_customers ≤ total_customers →
  (∀ t, t ≤ indoor_tables → indoor_min ≤ t * indoor_min) →
  (∀ t, t ≤ outdoor_tables → outdoor_min ≤ t * outdoor_min) →
  87 ≤ indoor_tables * indoor_min + outdoor_tables * outdoor_min :=
by
  sorry

#check cafe_chairs_minimum

end NUMINAMATH_CALUDE_cafe_chairs_minimum_l3308_330822
