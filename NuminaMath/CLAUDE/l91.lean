import Mathlib

namespace NUMINAMATH_CALUDE_rotate_point_D_l91_9147

/-- Rotates a point (x, y) by 180 degrees around a center (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

theorem rotate_point_D :
  let d : ℝ × ℝ := (2, -3)
  let center : ℝ × ℝ := (3, -2)
  rotate180 d.1 d.2 center.1 center.2 = (4, -1) := by
sorry

end NUMINAMATH_CALUDE_rotate_point_D_l91_9147


namespace NUMINAMATH_CALUDE_edge_probability_is_three_nineteenths_l91_9164

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem stating the probability of selecting two vertices that form an edge -/
theorem edge_probability_is_three_nineteenths (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_edge_probability_is_three_nineteenths_l91_9164


namespace NUMINAMATH_CALUDE_inequality_not_true_l91_9172

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l91_9172


namespace NUMINAMATH_CALUDE_cookies_to_mike_is_23_l91_9128

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike (total cookies_to_tim cookies_in_fridge : ℕ) : ℕ :=
  total - (cookies_to_tim + 2 * cookies_to_tim + cookies_in_fridge)

/-- Theorem: Uncle Jude gave 23 cookies to Mike -/
theorem cookies_to_mike_is_23 :
  cookies_to_mike 256 15 188 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_to_mike_is_23_l91_9128


namespace NUMINAMATH_CALUDE_lower_right_is_four_l91_9174

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Fin 5

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧ 
  (∀ i j k, i ≠ j → g k i ≠ g k j)

-- Define the initial configuration
def initial_config (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 3 ∧
  g 1 0 = 2 ∧ g 1 1 = 3 ∧ g 1 4 = 1 ∧
  g 2 1 = 1 ∧ g 2 3 = 5 ∧
  g 4 2 = 4

-- Theorem statement
theorem lower_right_is_four :
  ∀ g : Grid, is_valid_grid g → initial_config g → g 4 4 = 4 :=
sorry

end NUMINAMATH_CALUDE_lower_right_is_four_l91_9174


namespace NUMINAMATH_CALUDE_average_income_p_q_l91_9116

theorem average_income_p_q (p q r : ℕ) : 
  (q + r) / 2 = 6250 →
  (p + r) / 2 = 5200 →
  p = 4000 →
  (p + q) / 2 = 5050 :=
by
  sorry

#check average_income_p_q

end NUMINAMATH_CALUDE_average_income_p_q_l91_9116


namespace NUMINAMATH_CALUDE_expression_evaluation_l91_9134

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l91_9134


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l91_9195

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l91_9195


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_square_l91_9113

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

theorem sum_to_k_perfect_square (k : ℕ) :
  k ≤ 49 →
  (∃ n : ℕ, n < 100 ∧ sum_to_k k = n^2) ↔
  k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_square_l91_9113


namespace NUMINAMATH_CALUDE_maximum_mark_calculation_maximum_mark_is_500_l91_9186

theorem maximum_mark_calculation (passing_threshold : ℝ) (student_score : ℕ) (failure_margin : ℕ) : ℝ :=
  let passing_mark : ℕ := student_score + failure_margin
  let maximum_mark : ℝ := passing_mark / passing_threshold
  maximum_mark

theorem maximum_mark_is_500 :
  maximum_mark_calculation 0.33 125 40 = 500 := by
  sorry

end NUMINAMATH_CALUDE_maximum_mark_calculation_maximum_mark_is_500_l91_9186


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l91_9182

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l91_9182


namespace NUMINAMATH_CALUDE_subsets_count_l91_9125

theorem subsets_count : ∃ (n : ℕ), n = (Finset.filter (fun X => {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5}) (Finset.powerset {1, 2, 3, 4, 5})).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_subsets_count_l91_9125


namespace NUMINAMATH_CALUDE_greatest_integer_side_length_l91_9196

theorem greatest_integer_side_length (area : ℝ) (h : area < 150) :
  ∃ (s : ℕ), s * s ≤ area ∧ ∀ (t : ℕ), t * t ≤ area → t ≤ s ∧ s = 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_side_length_l91_9196


namespace NUMINAMATH_CALUDE_star_operation_value_l91_9178

def star_operation (a b : ℚ) : ℚ := 1 / a + 1 / b

theorem star_operation_value (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  star_operation a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_value_l91_9178


namespace NUMINAMATH_CALUDE_cauchy_inequality_and_minimum_value_l91_9124

theorem cauchy_inequality_and_minimum_value (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 ∧
  (x^2 + y^2 = 2 ∧ |x| ≠ |y| → ∃ (min : ℝ), min = 50/9 ∧ ∀ z, z = 1/(9*x^2) + 9/y^2 → z ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_inequality_and_minimum_value_l91_9124


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l91_9198

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given vectors a and b, prove that if they are collinear, then y = -2 -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (-3, 1)
  let b : ℝ × ℝ := (6, y)
  collinear a b → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l91_9198


namespace NUMINAMATH_CALUDE_map_scale_proportion_l91_9118

/-- Represents the scale of a map -/
structure MapScale where
  cm : ℝ  -- centimeters on the map
  km : ℝ  -- kilometers in reality

/-- 
Given a map scale where 15 cm represents 90 km, 
proves that 20 cm represents 120 km on the same map
-/
theorem map_scale_proportion (scale : MapScale) 
  (h : scale.cm = 15 ∧ scale.km = 90) : 
  ∃ (new_scale : MapScale), 
    new_scale.cm = 20 ∧ 
    new_scale.km = 120 ∧
    new_scale.km / new_scale.cm = scale.km / scale.cm := by
  sorry


end NUMINAMATH_CALUDE_map_scale_proportion_l91_9118


namespace NUMINAMATH_CALUDE_point_P_coordinates_l91_9154

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the conditions
def lies_on_y_axis (P : Point) : Prop :=
  P.x = 0

def parallel_to_x_axis (P Q : Point) : Prop :=
  P.y = Q.y

def equal_distance_to_axes (P : Point) : Prop :=
  |P.x| = |P.y|

-- Main theorem
theorem point_P_coordinates (a : ℝ) :
  let P : Point := ⟨2*a - 2, a + 5⟩
  let Q : Point := ⟨2, 5⟩
  (lies_on_y_axis P ∨ parallel_to_x_axis P Q ∨ equal_distance_to_axes P) →
  (P = ⟨12, 12⟩ ∨ P = ⟨-12, -12⟩ ∨ P = ⟨-4, 4⟩ ∨ P = ⟨4, -4⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l91_9154


namespace NUMINAMATH_CALUDE_computer_cost_l91_9191

theorem computer_cost (total_budget fridge_cost tv_cost computer_cost : ℕ) : 
  total_budget = 1600 →
  tv_cost = 600 →
  fridge_cost = computer_cost + 500 →
  total_budget = tv_cost + fridge_cost + computer_cost →
  computer_cost = 250 := by
sorry

end NUMINAMATH_CALUDE_computer_cost_l91_9191


namespace NUMINAMATH_CALUDE_pencils_at_meeting_pencils_at_meeting_proof_l91_9105

/-- The number of pencils brought to a committee meeting -/
theorem pencils_at_meeting : ℕ :=
  let associate_prof : ℕ → ℕ := λ x ↦ x  -- Number of associate professors
  let assistant_prof : ℕ → ℕ := λ x ↦ x  -- Number of assistant professors
  let total_people : ℕ := 7  -- Total number of people at the meeting
  let total_charts : ℕ := 11  -- Total number of charts brought to the meeting
  let pencils_per_associate : ℕ := 2  -- Pencils brought by each associate professor
  let pencils_per_assistant : ℕ := 1  -- Pencils brought by each assistant professor
  let charts_per_associate : ℕ := 1  -- Charts brought by each associate professor
  let charts_per_assistant : ℕ := 2  -- Charts brought by each assistant professor

  10  -- The theorem states that the number of pencils is 10

theorem pencils_at_meeting_proof :
  ∀ (x y : ℕ),
  x + y = total_people →
  charts_per_associate * x + charts_per_assistant * y = total_charts →
  pencils_per_associate * x + pencils_per_assistant * y = pencils_at_meeting :=
by
  sorry

#check pencils_at_meeting
#check pencils_at_meeting_proof

end NUMINAMATH_CALUDE_pencils_at_meeting_pencils_at_meeting_proof_l91_9105


namespace NUMINAMATH_CALUDE_divisor_problem_l91_9101

theorem divisor_problem (n m : ℕ) (h1 : n = 987654) (h2 : m = 42) : 
  (n + m) % m = 0 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l91_9101


namespace NUMINAMATH_CALUDE_range_of_m_l91_9131

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  m ∈ Set.Ioo 0 3 := by
sorry

-- Note: Set.Ioo 0 3 represents the open interval (0, 3)

end NUMINAMATH_CALUDE_range_of_m_l91_9131


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l91_9189

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) :
  angle_in_degrees = 1440 →
  (angle_in_degrees * (π / 180)) = 8 * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l91_9189


namespace NUMINAMATH_CALUDE_common_roots_imply_a_b_values_l91_9168

-- Define the two cubic polynomials
def p (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 11*x + 6
def q (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 14*x + 8

-- Define a predicate for having two distinct common roots
def has_two_distinct_common_roots (a b : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ p a r = 0 ∧ p a s = 0 ∧ q b r = 0 ∧ q b s = 0

-- State the theorem
theorem common_roots_imply_a_b_values :
  ∀ a b : ℝ, has_two_distinct_common_roots a b → (a = 6 ∧ b = 7) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_imply_a_b_values_l91_9168


namespace NUMINAMATH_CALUDE_marble_boxes_theorem_l91_9130

/-- Given a number of marbles per box and a total number of marbles,
    calculate the number of boxes. -/
def number_of_boxes (marbles_per_box : ℕ) (total_marbles : ℕ) : ℕ :=
  total_marbles / marbles_per_box

/-- Theorem stating that with 6 marbles per box and 18 total marbles,
    the number of boxes is 3. -/
theorem marble_boxes_theorem :
  number_of_boxes 6 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_boxes_theorem_l91_9130


namespace NUMINAMATH_CALUDE_fraction_inequality_l91_9184

theorem fraction_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l91_9184


namespace NUMINAMATH_CALUDE_no_solution_l91_9102

/-- Q(n) denotes the greatest prime factor of n -/
def Q (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying
    both Q(n) = √n and Q(3n + 16) = √(3n + 16) -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ 
    Q n = Nat.sqrt n ∧ 
    Q (3 * n + 16) = Nat.sqrt (3 * n + 16) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l91_9102


namespace NUMINAMATH_CALUDE_swim_team_capacity_difference_l91_9141

/-- Represents the number of each type of vehicle --/
structure Vehicles where
  cars : Nat
  vans : Nat
  minibuses : Nat

/-- Represents the maximum capacity of each type of vehicle --/
structure VehicleCapacities where
  car : Nat
  van : Nat
  minibus : Nat

/-- Represents the actual number of people in each vehicle --/
structure ActualOccupancy where
  car1 : Nat
  car2 : Nat
  van1 : Nat
  van2 : Nat
  van3 : Nat
  minibus : Nat

def vehicles : Vehicles := {
  cars := 2,
  vans := 3,
  minibuses := 1
}

def capacities : VehicleCapacities := {
  car := 6,
  van := 8,
  minibus := 15
}

def occupancy : ActualOccupancy := {
  car1 := 5,
  car2 := 4,
  van1 := 3,
  van2 := 3,
  van3 := 5,
  minibus := 10
}

def totalMaxCapacity (v : Vehicles) (c : VehicleCapacities) : Nat :=
  v.cars * c.car + v.vans * c.van + v.minibuses * c.minibus

def actualTotalOccupancy (o : ActualOccupancy) : Nat :=
  o.car1 + o.car2 + o.van1 + o.van2 + o.van3 + o.minibus

theorem swim_team_capacity_difference :
  totalMaxCapacity vehicles capacities - actualTotalOccupancy occupancy = 21 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_capacity_difference_l91_9141


namespace NUMINAMATH_CALUDE_point_B_in_first_quadrant_l91_9199

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem to be proved -/
theorem point_B_in_first_quadrant (A : Point2D) (hA : is_in_second_quadrant A) :
  let B : Point2D := ⟨-2 * A.x, (1/3) * A.y⟩
  is_in_first_quadrant B := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_first_quadrant_l91_9199


namespace NUMINAMATH_CALUDE_polynomial_value_range_l91_9152

/-- A polynomial with integer coefficients that equals 5 for five different integer inputs -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a b c d e : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5

theorem polynomial_value_range (P : ℤ → ℤ) (h : IntPolynomial P) :
  ¬∃ x : ℤ, ((-6 : ℤ) ≤ P x ∧ P x ≤ 4) ∨ (6 ≤ P x ∧ P x ≤ 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_range_l91_9152


namespace NUMINAMATH_CALUDE_even_function_f_2_l91_9157

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * (x + 3)

-- State the theorem
theorem even_function_f_2 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : f a 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_even_function_f_2_l91_9157


namespace NUMINAMATH_CALUDE_circle_equation_proof_l91_9149

/-- A circle with center on the x-axis passing through two given points -/
structure CircleOnXAxis where
  center : ℝ  -- x-coordinate of the center
  passesThrough : (ℝ × ℝ) → (ℝ × ℝ) → Prop

/-- The equation of a circle given its center and a point on the circle -/
def circleEquation (h : ℝ) (k : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = ((5 - h)^2 + 2^2)

theorem circle_equation_proof (c : CircleOnXAxis) 
  (h1 : c.passesThrough (5, 2) (-1, 4)) :
  ∀ x y, circleEquation 1 0 x y ↔ (x - 1)^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l91_9149


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l91_9159

theorem theater_ticket_sales
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l91_9159


namespace NUMINAMATH_CALUDE_max_knights_count_l91_9148

/-- Represents the type of islander: Knight or Liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander -/
inductive Statement
  | BothNeighborsLiars
  | OneNeighborLiar

/-- Configuration of islanders around the table -/
structure IslanderConfig where
  total : Nat
  half_both_liars : Nat
  half_one_liar : Nat
  knight_count : Nat

/-- Checks if the given configuration is valid -/
def is_valid_config (config : IslanderConfig) : Prop :=
  config.total = 100 ∧
  config.half_both_liars = 50 ∧
  config.half_one_liar = 50 ∧
  config.knight_count ≤ config.total

/-- Theorem stating the maximum number of knights possible -/
theorem max_knights_count (config : IslanderConfig) 
  (h_valid : is_valid_config config) : 
  config.knight_count ≤ 67 :=
sorry

end NUMINAMATH_CALUDE_max_knights_count_l91_9148


namespace NUMINAMATH_CALUDE_max_volume_rect_prism_l91_9173

/-- A right prism with rectangular bases -/
structure RectPrism where
  a : ℝ  -- length of base
  b : ℝ  -- width of base
  h : ℝ  -- height of prism
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- The sum of areas of three mutually adjacent faces is 48 -/
def adjacent_faces_area (p : RectPrism) : ℝ :=
  p.a * p.h + p.b * p.h + p.a * p.b

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.a * p.b * p.h

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_rect_prism :
  ∃ (p : RectPrism),
    adjacent_faces_area p = 48 ∧
    p.a = p.b ∧  -- two lateral faces are congruent
    ∀ (q : RectPrism),
      adjacent_faces_area q = 48 →
      q.a = q.b →
      volume q ≤ volume p ∧
      volume p = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rect_prism_l91_9173


namespace NUMINAMATH_CALUDE_root_in_interval_l91_9135

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 5 = 0) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l91_9135


namespace NUMINAMATH_CALUDE_range_of_a_l91_9109

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l91_9109


namespace NUMINAMATH_CALUDE_sculpture_height_l91_9106

theorem sculpture_height (base_height : ℝ) (total_height_feet : ℝ) (h1 : base_height = 10) (h2 : total_height_feet = 3.6666666666666665) : 
  total_height_feet * 12 - base_height = 34 := by
sorry

end NUMINAMATH_CALUDE_sculpture_height_l91_9106


namespace NUMINAMATH_CALUDE_frontal_view_correct_l91_9180

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Calculates the maximum height of a column -/
def maxHeight (col : Column) : Nat :=
  col.foldl max 0

/-- Represents the arrangement of cube stacks -/
structure CubeArrangement where
  col1 : Column
  col2 : Column
  col3 : Column

/-- Calculates the frontal view heights of a cube arrangement -/
def frontalView (arr : CubeArrangement) : List Nat :=
  [maxHeight arr.col1, maxHeight arr.col2, maxHeight arr.col3]

/-- The specific cube arrangement described in the problem -/
def problemArrangement : CubeArrangement :=
  { col1 := [4, 2]
    col2 := [3, 0, 3]
    col3 := [1, 5] }

theorem frontal_view_correct :
  frontalView problemArrangement = [4, 3, 5] := by sorry

end NUMINAMATH_CALUDE_frontal_view_correct_l91_9180


namespace NUMINAMATH_CALUDE_ac_length_l91_9142

/-- Two triangles ABC and ADE are similar with given side lengths. -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  AD : ℝ
  DE : ℝ
  EA : ℝ
  similar : True  -- Represents that the triangles are similar
  h_AB : AB = 18
  h_BC : BC = 24
  h_CA : CA = 20
  h_AD : AD = 9
  h_DE : DE = 12
  h_EA : EA = 15

/-- The length of AC in the similar triangles is 20. -/
theorem ac_length (t : SimilarTriangles) : t.CA = 20 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l91_9142


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l91_9188

theorem buffet_meal_combinations : 
  (Nat.choose 4 2) * (Nat.choose 5 3) * (Nat.choose 5 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l91_9188


namespace NUMINAMATH_CALUDE_rectangle_ratio_l91_9132

theorem rectangle_ratio (s w h : ℝ) (h1 : w > 0) (h2 : h > 0) (h3 : s > 0) : 
  (s + 2*w) * (s + h) = 3 * s^2 → h / w = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l91_9132


namespace NUMINAMATH_CALUDE_percentage_gain_calculation_l91_9175

def calculate_percentage_gain (total_bought : ℕ) (cost_per_bowl : ℚ) (total_sold : ℕ) (sell_per_bowl : ℚ) : ℚ :=
  let total_cost := total_bought * cost_per_bowl
  let total_revenue := total_sold * sell_per_bowl
  let profit := total_revenue - total_cost
  (profit / total_cost) * 100

theorem percentage_gain_calculation :
  let total_bought : ℕ := 114
  let cost_per_bowl : ℚ := 13
  let total_sold : ℕ := 108
  let sell_per_bowl : ℚ := 17
  abs (calculate_percentage_gain total_bought cost_per_bowl total_sold sell_per_bowl - 23.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percentage_gain_calculation_l91_9175


namespace NUMINAMATH_CALUDE_u_1990_equals_one_l91_9140

def u : ℕ → ℕ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then 1 - u (n / 2) else u (n / 2)

theorem u_1990_equals_one : u 1990 = 1 := by
  sorry

end NUMINAMATH_CALUDE_u_1990_equals_one_l91_9140


namespace NUMINAMATH_CALUDE_angle_abc_measure_l91_9138

theorem angle_abc_measure (angle_cbd angle_abd angle_abc : ℝ) 
  (h1 : angle_cbd = 90)
  (h2 : angle_abd = 60)
  (h3 : angle_abc + angle_abd + angle_cbd = 190) :
  angle_abc = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l91_9138


namespace NUMINAMATH_CALUDE_subtraction_of_one_and_two_l91_9112

theorem subtraction_of_one_and_two : 1 - 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_one_and_two_l91_9112


namespace NUMINAMATH_CALUDE_cube_square_third_smallest_prime_l91_9153

/-- The third smallest prime number -/
def third_smallest_prime : Nat := 5

/-- The cube of the square of the third smallest prime number -/
def result : Nat := (third_smallest_prime ^ 2) ^ 3

theorem cube_square_third_smallest_prime :
  result = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_square_third_smallest_prime_l91_9153


namespace NUMINAMATH_CALUDE_max_sum_abcd_l91_9110

theorem max_sum_abcd (a b c d : ℤ) 
  (b_pos : b > 0)
  (eq1 : a + b = c)
  (eq2 : b + c = d)
  (eq3 : c + d = a) :
  a + b + c + d ≤ -5 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    b₀ > 0 ∧ 
    a₀ + b₀ = c₀ ∧ 
    b₀ + c₀ = d₀ ∧ 
    c₀ + d₀ = a₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abcd_l91_9110


namespace NUMINAMATH_CALUDE_carl_index_cards_cost_l91_9108

/-- Calculates the total cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (cards_per_pack : ℕ) (cost_per_pack : ℕ) : ℕ :=
  let total_students := periods * students_per_class
  let total_cards := total_students * cards_per_student
  let packs_needed := (total_cards + cards_per_pack - 1) / cards_per_pack  -- Ceiling division
  packs_needed * cost_per_pack

/-- Proves that the total cost of index cards for Carl's classes is $108 -/
theorem carl_index_cards_cost : 
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_cards_cost_l91_9108


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l91_9156

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 18) 
  (h2 : z + x = 19) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 24150.1875 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l91_9156


namespace NUMINAMATH_CALUDE_equal_goals_moment_l91_9121

/-- Represents the state of a football match at any given moment -/
structure MatchState where
  goalsWinner : ℕ
  goalsLoser : ℕ

/-- The final score of the match -/
def finalScore : MatchState := { goalsWinner := 9, goalsLoser := 5 }

/-- Theorem stating that there exists a point during the match where the number of goals
    the winning team still needs to score equals the number of goals the losing team has already scored -/
theorem equal_goals_moment :
  ∃ (state : MatchState), 
    state.goalsWinner ≤ finalScore.goalsWinner ∧ 
    state.goalsLoser ≤ finalScore.goalsLoser ∧
    (finalScore.goalsWinner - state.goalsWinner) = state.goalsLoser :=
sorry

end NUMINAMATH_CALUDE_equal_goals_moment_l91_9121


namespace NUMINAMATH_CALUDE_train_speed_and_length_l91_9160

def bridge_length : ℝ := 1260
def bridge_time : ℝ := 60
def tunnel_length : ℝ := 2010
def tunnel_time : ℝ := 90

theorem train_speed_and_length :
  ∃ (speed length : ℝ),
    (bridge_length + length) / bridge_time = (tunnel_length + length) / tunnel_time ∧
    speed = (bridge_length + length) / bridge_time ∧
    speed = 25 ∧
    length = 240 := by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l91_9160


namespace NUMINAMATH_CALUDE_number_of_ways_to_draw_l91_9166

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The sequence of colors to be drawn -/
def color_sequence : List String := ["Red", "Green", "Blue", "Yellow"]

/-- Function to calculate the number of ways to draw the balls -/
def ways_to_draw : ℕ := (total_balls - 0) * (total_balls - 1) * (total_balls - 2) * (total_balls - 3)

/-- Theorem stating the number of ways to draw the balls -/
theorem number_of_ways_to_draw :
  ways_to_draw = 32760 := by sorry

end NUMINAMATH_CALUDE_number_of_ways_to_draw_l91_9166


namespace NUMINAMATH_CALUDE_divisor_problem_l91_9158

theorem divisor_problem (x d : ℕ) (h1 : x % d = 5) (h2 : (x + 13) % 41 = 18) : d = 41 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l91_9158


namespace NUMINAMATH_CALUDE_effective_discount_l91_9129

theorem effective_discount (initial_discount coupon_discount : ℝ) : 
  initial_discount = 0.6 →
  coupon_discount = 0.3 →
  let sale_price := 1 - initial_discount
  let final_price := sale_price * (1 - coupon_discount)
  1 - final_price = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_effective_discount_l91_9129


namespace NUMINAMATH_CALUDE_vasyas_numbers_l91_9183

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l91_9183


namespace NUMINAMATH_CALUDE_expression_equality_l91_9111

theorem expression_equality : 2 * Real.sin (π / 3) + Real.sqrt 12 + abs (-5) - (π - Real.sqrt 2) ^ 0 = 3 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l91_9111


namespace NUMINAMATH_CALUDE_number_of_carnations_solve_carnation_problem_l91_9162

/-- Proves the number of carnations given the problem conditions --/
theorem number_of_carnations : ℕ → Prop :=
  fun c =>
    let vase_capacity : ℕ := 9
    let num_roses : ℕ := 23
    let num_vases : ℕ := 3
    (c + num_roses = num_vases * vase_capacity) → c = 4

/-- The theorem statement --/
theorem solve_carnation_problem : number_of_carnations 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_carnations_solve_carnation_problem_l91_9162


namespace NUMINAMATH_CALUDE_club_members_count_l91_9165

/-- The number of members in the club -/
def n : ℕ := sorry

/-- The age of the old (replaced) member -/
def O : ℕ := sorry

/-- The age of the new member -/
def N : ℕ := sorry

/-- The average age remains unchanged after replacement and 3 years -/
axiom avg_unchanged : (n * O + 3 * n) / n = (n * N + 3 * n) / n

/-- The difference between the ages of the replaced and new member is 15 -/
axiom age_difference : O - N = 15

/-- Theorem: The number of members in the club is 5 -/
theorem club_members_count : n = 5 := by sorry

end NUMINAMATH_CALUDE_club_members_count_l91_9165


namespace NUMINAMATH_CALUDE_melinda_coffees_l91_9117

/-- The cost of one doughnut in dollars -/
def doughnut_cost : ℚ := 45/100

/-- The total cost of Harold's purchase on Monday in dollars -/
def harold_total : ℚ := 491/100

/-- The number of doughnuts Harold bought on Monday -/
def harold_doughnuts : ℕ := 3

/-- The number of coffees Harold bought on Monday -/
def harold_coffees : ℕ := 4

/-- The total cost of Melinda's purchase on Tuesday in dollars -/
def melinda_total : ℚ := 759/100

/-- The number of doughnuts Melinda bought on Tuesday -/
def melinda_doughnuts : ℕ := 5

/-- Theorem stating that Melinda bought 6 large coffees on Tuesday -/
theorem melinda_coffees : ℕ := by
  sorry


end NUMINAMATH_CALUDE_melinda_coffees_l91_9117


namespace NUMINAMATH_CALUDE_coloring_book_problem_l91_9197

theorem coloring_book_problem (book1 : Nat) (book2 : Nat) (colored : Nat) : 
  book1 = 23 → book2 = 32 → colored = 44 → book1 + book2 - colored = 11 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l91_9197


namespace NUMINAMATH_CALUDE_triangle_existence_l91_9167

/-- Represents a triangle with side lengths a, b, c and angles α, β, γ. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem stating the existence of a triangle satisfying given conditions. -/
theorem triangle_existence (d β γ : ℝ) : 
  ∃ (t : Triangle), 
    t.b + t.c - t.a = d ∧ 
    t.β = β ∧ 
    t.γ = γ ∧
    t.α + t.β + t.γ = π :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l91_9167


namespace NUMINAMATH_CALUDE_ellipse_and_line_property_l91_9100

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ
  b : ℝ
  h : k ≠ 0 ∧ b ≠ 0

/-- Given conditions of the problem -/
axiom ellipse_condition (C : Ellipse) : 
  C.a^2 - C.b^2 = 4 ∧ 2/C.a^2 + 3/C.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_and_line_property (C : Ellipse) (l : Line) :
  (∀ x y, x^2/C.a^2 + y^2/C.b^2 = 1 ↔ x^2/8 + y^2/4 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    x₁^2/8 + y₁^2/4 = 1 ∧
    x₂^2/8 + y₂^2/4 = 1 ∧
    y₁ = l.k * x₁ + l.b ∧
    y₂ = l.k * x₂ + l.b ∧
    x₁ ≠ x₂ ∧
    let xₘ := (x₁ + x₂)/2
    let yₘ := (y₁ + y₂)/2
    (yₘ / xₘ) * l.k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_property_l91_9100


namespace NUMINAMATH_CALUDE_sector_to_cone_l91_9136

/-- Proves that a 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone (sector_angle : Real) (circle_radius : Real) 
  (h1 : sector_angle = 270)
  (h2 : circle_radius = 12) : 
  let base_radius := (sector_angle / 360) * (2 * Real.pi * circle_radius) / (2 * Real.pi)
  let slant_height := circle_radius
  (base_radius = 9 ∧ slant_height = 12) := by
  sorry

end NUMINAMATH_CALUDE_sector_to_cone_l91_9136


namespace NUMINAMATH_CALUDE_storks_joined_l91_9114

theorem storks_joined (initial_birds initial_storks final_difference : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  final_difference = 5 →
  ∃ joined : ℕ, initial_storks + joined = initial_birds + final_difference ∧ joined = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_joined_l91_9114


namespace NUMINAMATH_CALUDE_quarter_circle_arcs_sum_l91_9192

/-- The sum of the lengths of n quarter-circle arcs, each constructed on a segment of length D/n 
    (where D is the diameter of a large circle), approaches πD/8 as n approaches infinity. -/
theorem quarter_circle_arcs_sum (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 8| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_arcs_sum_l91_9192


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l91_9161

def x : ℕ := 5 * 24 * 36

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube : 
  (∀ y < 50, ¬ is_perfect_cube (x * y)) ∧ is_perfect_cube (x * 50) := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l91_9161


namespace NUMINAMATH_CALUDE_green_hats_not_adjacent_probability_l91_9146

def total_children : ℕ := 9
def green_hats : ℕ := 3

theorem green_hats_not_adjacent_probability :
  let total_arrangements := Nat.choose total_children green_hats
  let adjacent_arrangements := (total_children - green_hats + 1) + (total_children - 1) * (total_children - green_hats - 1)
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_not_adjacent_probability_l91_9146


namespace NUMINAMATH_CALUDE_five_solutions_l91_9127

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w θ : ℝ) : 
  x = 2*z + 2*w + z*w*x →
  y = 2*w + 2*x + w*x*y →
  z = 2*x + 2*y + x*y*z →
  w = 2*y + 2*z + y*z*w →
  w = Real.sin θ ^ 2 →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ 
    (a = 2*c + 2*d + c*d*a ∧
     b = 2*d + 2*a + d*a*b ∧
     c = 2*a + 2*b + a*b*c ∧
     d = 2*b + 2*c + b*c*d ∧
     d = Real.sin θ ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_five_solutions_l91_9127


namespace NUMINAMATH_CALUDE_total_value_is_20_31_l91_9145

/-- Represents the value of coins in U.S. Dollars -/
def total_value : ℝ :=
  let us_quarter_value : ℝ := 0.25
  let us_nickel_value : ℝ := 0.05
  let canadian_dime_value : ℝ := 0.10
  let euro_cent_value : ℝ := 0.01
  let british_pence_value : ℝ := 0.01
  let cad_to_usd : ℝ := 0.8
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.4
  let us_quarters : ℝ := 4 * 10 * us_quarter_value
  let us_nickels : ℝ := 9 * 10 * us_nickel_value
  let canadian_dimes : ℝ := 6 * 10 * canadian_dime_value * cad_to_usd
  let euro_cents : ℝ := 5 * 10 * euro_cent_value * eur_to_usd
  let british_pence : ℝ := 3 * 10 * british_pence_value * gbp_to_usd
  us_quarters + us_nickels + canadian_dimes + euro_cents + british_pence

/-- Theorem stating that the total value of Rocco's coins is $20.31 -/
theorem total_value_is_20_31 : total_value = 20.31 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_20_31_l91_9145


namespace NUMINAMATH_CALUDE_mitch_spare_candy_bars_l91_9150

/-- Proves that Mitch wants to have 10 spare candy bars --/
theorem mitch_spare_candy_bars : 
  let bars_per_friend : ℕ := 2
  let total_bars : ℕ := 24
  let num_friends : ℕ := 7
  let spare_bars : ℕ := total_bars - (bars_per_friend * num_friends)
  spare_bars = 10 := by sorry

end NUMINAMATH_CALUDE_mitch_spare_candy_bars_l91_9150


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l91_9194

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def equation (z : ℂ) : Prop := z * (1 - i) = 2 - i

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : equation z) : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l91_9194


namespace NUMINAMATH_CALUDE_wax_needed_l91_9103

theorem wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 288) (h2 : available_wax = 28) :
  total_wax - available_wax = 260 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_l91_9103


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l91_9185

theorem arithmetic_calculations :
  ((-5 + 8 - 2 : ℚ) = 1) ∧
  ((-3 * (5/6) / (-1/4) : ℚ) = 10) ∧
  ((-3/17 + (-3.75) + (-14/17) + 3 * (3/4) : ℚ) = -1) ∧
  ((-1^10 - (13/14 - 11/12) * (4 - (-2)^2) + 1/2 / 3 : ℚ) = -5/6) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l91_9185


namespace NUMINAMATH_CALUDE_smallest_with_14_divisors_l91_9177

/-- Count the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 14 positive divisors -/
def has_14_divisors (n : ℕ) : Prop :=
  count_divisors n = 14

/-- The theorem stating that 192 is the smallest positive integer with exactly 14 positive divisors -/
theorem smallest_with_14_divisors :
  (∀ m : ℕ, m > 0 → m < 192 → ¬(has_14_divisors m)) ∧ has_14_divisors 192 := by sorry

end NUMINAMATH_CALUDE_smallest_with_14_divisors_l91_9177


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l91_9169

theorem largest_number_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 42) 
  (h2 : Nat.lcm a b = 42 * 10 * 20) : max a b = 840 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l91_9169


namespace NUMINAMATH_CALUDE_solution_set_inequality_l91_9123

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | |x + 1| - |x - 5| < 4} = Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l91_9123


namespace NUMINAMATH_CALUDE_connect_four_games_total_l91_9107

/-- Given that Kaleb's ratio of won to lost games is 3:2 and he won 18 games,
    prove that the total number of games played is 30. -/
theorem connect_four_games_total (won lost total : ℕ) : 
  won = 18 → 
  3 * lost = 2 * won → 
  total = won + lost → 
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_connect_four_games_total_l91_9107


namespace NUMINAMATH_CALUDE_sandwich_count_l91_9139

theorem sandwich_count (sandwich_price : ℚ) (soda_price : ℚ) (soda_count : ℕ) (total_cost : ℚ) :
  sandwich_price = 149/100 →
  soda_price = 87/100 →
  soda_count = 4 →
  total_cost = 646/100 →
  ∃ (sandwich_count : ℕ), sandwich_count = 2 ∧ 
    sandwich_count * sandwich_price + soda_count * soda_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_sandwich_count_l91_9139


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l91_9181

theorem eight_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 7) = 80000000 := by sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l91_9181


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l91_9122

theorem complex_magnitude_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l91_9122


namespace NUMINAMATH_CALUDE_new_shoes_average_speed_l91_9126

/-- Calculate the average speed of a hiker using new high-tech shoes over a 4-hour hike -/
theorem new_shoes_average_speed
  (old_speed : ℝ)
  (new_speed_multiplier : ℝ)
  (hike_duration : ℝ)
  (blister_interval : ℝ)
  (speed_reduction_per_blister : ℝ)
  (h_old_speed : old_speed = 6)
  (h_new_speed_multiplier : new_speed_multiplier = 2)
  (h_hike_duration : hike_duration = 4)
  (h_blister_interval : blister_interval = 2)
  (h_speed_reduction : speed_reduction_per_blister = 2)
  : (old_speed * new_speed_multiplier + 
     (old_speed * new_speed_multiplier - speed_reduction_per_blister)) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_average_speed_l91_9126


namespace NUMINAMATH_CALUDE_genevieve_cherries_l91_9179

/-- The number of kilograms of cherries Genevieve bought -/
def cherries_bought : ℕ := 277

/-- The original price of cherries per kilogram in cents -/
def original_price : ℕ := 800

/-- The discount percentage on cherries -/
def discount_percentage : ℚ := 1 / 10

/-- The amount Genevieve was short in cents -/
def short_amount : ℕ := 40000

/-- The amount Genevieve had in cents -/
def genevieve_amount : ℕ := 160000

/-- Theorem stating that given the conditions, Genevieve bought 277 kilograms of cherries -/
theorem genevieve_cherries :
  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_price : ℕ := genevieve_amount + short_amount
  (total_price : ℚ) / discounted_price = cherries_bought := by sorry

end NUMINAMATH_CALUDE_genevieve_cherries_l91_9179


namespace NUMINAMATH_CALUDE_hyperbola_condition_l91_9120

/-- A curve is a hyperbola if it can be represented by an equation of the form
    (x²/a²) - (y²/b²) = 1 or (y²/a²) - (x²/b²) = 1, where a and b are non-zero real numbers. -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
    (∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∨
    (∀ x y, f x y ↔ (y^2 / a^2) - (x^2 / b^2) = 1)

/-- The curve represented by the equation x²/(k-3) - y²/(k+3) = 1 -/
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

theorem hyperbola_condition (k : ℝ) :
  (k > 3 → is_hyperbola (curve k)) ∧
  ¬(is_hyperbola (curve k) → k > 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l91_9120


namespace NUMINAMATH_CALUDE_hash_solution_l91_9190

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem hash_solution :
  ∃ (A : ℝ), (hash A 7 = 225) ∧ (A = 7 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hash_solution_l91_9190


namespace NUMINAMATH_CALUDE_equation_solution_l91_9133

theorem equation_solution : 
  ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ (x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l91_9133


namespace NUMINAMATH_CALUDE_tetrahedron_regularity_l91_9119

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)

-- Define properties of the tetrahedron
def has_inscribed_sphere (t : Tetrahedron) : Prop := sorry

def sphere_touches_incenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_orthocenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_centroid (t : Tetrahedron) : Prop := sorry

def is_regular (t : Tetrahedron) : Prop := sorry

-- Theorem statement
theorem tetrahedron_regularity (t : Tetrahedron) :
  has_inscribed_sphere t ∧
  sphere_touches_incenter t ∧
  sphere_touches_orthocenter t ∧
  sphere_touches_centroid t →
  is_regular t :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_regularity_l91_9119


namespace NUMINAMATH_CALUDE_inequality_equivalence_l91_9155

theorem inequality_equivalence (x : ℝ) : 
  1 / (x - 2) < 4 ↔ x < 2 ∨ x > 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l91_9155


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l91_9104

/-- Represents the amount of corn, either in bushels or individual ears. -/
inductive CornAmount
| bushels (n : ℕ)
| ears (n : ℕ)

/-- Converts CornAmount to total number of ears. -/
def to_ears (amount : CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  match amount with
  | CornAmount.bushels n => n * ears_per_bushel
  | CornAmount.ears n => n

/-- Calculates the remaining ears of corn after giving some away. -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  to_ears initial ears_per_bushel - (given_away.map (λ a => to_ears a ears_per_bushel)).sum

theorem bob_remaining_corn :
  let initial := CornAmount.bushels 120
  let given_away := [
    CornAmount.bushels 15,  -- Terry
    CornAmount.bushels 8,   -- Jerry
    CornAmount.bushels 25,  -- Linda
    CornAmount.ears 42,     -- Stacy
    CornAmount.bushels 9,   -- Susan
    CornAmount.bushels 4,   -- Tim (bushels)
    CornAmount.ears 18      -- Tim (ears)
  ]
  let ears_per_bushel := 15
  remaining_corn initial given_away ears_per_bushel = 825 := by
  sorry

#eval remaining_corn (CornAmount.bushels 120) [
  CornAmount.bushels 15,
  CornAmount.bushels 8,
  CornAmount.bushels 25,
  CornAmount.ears 42,
  CornAmount.bushels 9,
  CornAmount.bushels 4,
  CornAmount.ears 18
] 15

end NUMINAMATH_CALUDE_bob_remaining_corn_l91_9104


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l91_9115

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / i
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l91_9115


namespace NUMINAMATH_CALUDE_remainder_17_49_mod_5_l91_9193

theorem remainder_17_49_mod_5 : 17^49 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_49_mod_5_l91_9193


namespace NUMINAMATH_CALUDE_jeff_donation_proof_l91_9176

/-- The percentage of pencils Jeff donated -/
def jeff_donation_percentage : ℝ := 0.3

theorem jeff_donation_proof :
  let jeff_initial : ℕ := 300
  let vicki_initial : ℕ := 2 * jeff_initial
  let vicki_donation : ℝ := 3/4 * vicki_initial
  let total_remaining : ℕ := 360
  (jeff_initial - jeff_initial * jeff_donation_percentage) +
    (vicki_initial - vicki_donation) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_jeff_donation_proof_l91_9176


namespace NUMINAMATH_CALUDE_shenny_vacation_shirts_l91_9170

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_shenny_vacation_shirts_l91_9170


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l91_9163

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 0 = 25) 
  (h_last : a 5 = 57) : 
  a 3 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l91_9163


namespace NUMINAMATH_CALUDE_x_intercept_is_four_l91_9144

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 4 -/
theorem x_intercept_is_four :
  let l : Line := { x₁ := 10, y₁ := 3, x₂ := -10, y₂ := -7 }
  x_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_four_l91_9144


namespace NUMINAMATH_CALUDE_cube_volume_problem_l91_9151

theorem cube_volume_problem (cube_a_volume : ℝ) (surface_area_ratio : ℝ) :
  cube_a_volume = 8 →
  surface_area_ratio = 3 →
  ∃ (cube_b_volume : ℝ),
    (6 * (cube_a_volume ^ (1/3))^2) * surface_area_ratio = 6 * (cube_b_volume ^ (1/3))^2 ∧
    cube_b_volume = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l91_9151


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l91_9171

def reverse_digits (n : ℕ) : ℕ :=
  sorry

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_11 :
  ∃ m : ℕ,
    is_four_digit m ∧
    is_four_digit (reverse_digits m) ∧
    63 ∣ m ∧
    63 ∣ (reverse_digits m) ∧
    11 ∣ m ∧
    ∀ k : ℕ, (is_four_digit k ∧
              is_four_digit (reverse_digits k) ∧
              63 ∣ k ∧
              63 ∣ (reverse_digits k) ∧
              11 ∣ k) →
              k ≤ m ∧
    m = 9696 :=
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_11_l91_9171


namespace NUMINAMATH_CALUDE_motorcyclist_distance_l91_9143

/-- Represents the motion of a motorcyclist --/
structure Motion where
  initial_speed : ℝ
  acceleration : ℝ
  time_to_b : ℝ
  time_b_to_c : ℝ
  speed_at_c : ℝ

/-- Calculates the distance between points A and C --/
def distance_a_to_c (m : Motion) : ℝ :=
  let speed_at_b := m.initial_speed + m.acceleration * m.time_to_b
  let distance_a_to_b := m.initial_speed * m.time_to_b + 0.5 * m.acceleration * m.time_to_b^2
  let distance_b_to_c := speed_at_b * m.time_b_to_c - 0.5 * m.acceleration * m.time_b_to_c^2
  distance_a_to_b - distance_b_to_c

/-- The main theorem to prove --/
theorem motorcyclist_distance (m : Motion) 
  (h1 : m.initial_speed = 90)
  (h2 : m.time_to_b = 3)
  (h3 : m.time_b_to_c = 2)
  (h4 : m.speed_at_c = 110)
  (h5 : m.acceleration = (m.speed_at_c - m.initial_speed) / (m.time_to_b + m.time_b_to_c)) :
  distance_a_to_c m = 92 := by
  sorry


end NUMINAMATH_CALUDE_motorcyclist_distance_l91_9143


namespace NUMINAMATH_CALUDE_largest_integer_on_card_l91_9187

theorem largest_integer_on_card (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℕ) = {57, 70, 83} →
  max a (max b (max c (max d e))) = 48 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_on_card_l91_9187


namespace NUMINAMATH_CALUDE_power_729_minus_reciprocal_l91_9137

theorem power_729_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * 2) :
  x^729 - 1/(x^729) = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_power_729_minus_reciprocal_l91_9137
