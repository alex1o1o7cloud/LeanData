import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_nine_l1953_195324

theorem sum_of_squares_zero_implies_sum_nine (a b c : ℝ) 
  (h : 2 * (a - 2)^2 + 3 * (b - 3)^2 + 4 * (c - 4)^2 = 0) : 
  a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_nine_l1953_195324


namespace NUMINAMATH_CALUDE_playground_area_l1953_195378

theorem playground_area (w l : ℚ) (h1 : l = 2 * w + 30) (h2 : 2 * (l + w) = 700) : w * l = 233600 / 9 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l1953_195378


namespace NUMINAMATH_CALUDE_original_denominator_proof_l1953_195302

theorem original_denominator_proof : 
  ∀ d : ℚ, (5 : ℚ) / (d + 4) = (1 : ℚ) / 3 → d = 11 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l1953_195302


namespace NUMINAMATH_CALUDE_player_a_winning_strategy_l1953_195305

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a cubic polynomial ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Represents the state of the game -/
structure GameState where
  polynomial : CubicPolynomial
  current_player : Player
  moves_left : Nat

/-- Represents a move in the game -/
structure Move where
  value : ℤ
  position : Nat

/-- Function to apply a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Predicate to check if a polynomial has three distinct integer roots -/
def has_three_distinct_integer_roots (p : CubicPolynomial) : Prop :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initial_state : GameState),
      initial_state.current_player = Player.A →
      initial_state.moves_left = 3 →
      ∀ (b_moves : Fin 2 → Move),
        let final_state := apply_move (apply_move (apply_move initial_state (strategy initial_state)) (b_moves 0)) (b_moves 1)
        has_three_distinct_integer_roots final_state.polynomial :=
  sorry

end NUMINAMATH_CALUDE_player_a_winning_strategy_l1953_195305


namespace NUMINAMATH_CALUDE_onion_harvest_scientific_notation_l1953_195334

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem onion_harvest_scientific_notation :
  toScientificNotation 325000000 = ScientificNotation.mk 3.25 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_onion_harvest_scientific_notation_l1953_195334


namespace NUMINAMATH_CALUDE_factor_polynomial_l1953_195333

theorem factor_polynomial (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1953_195333


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l1953_195389

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) :
  ∀ x y : ℤ, p * (x + y) = x * y ↔
    (x = p * (p + 1) ∧ y = p + 1) ∨
    (x = 2 * p ∧ y = 2 * p) ∨
    (x = 0 ∧ y = 0) ∨
    (x = p * (1 - p) ∧ y = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l1953_195389


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l1953_195373

theorem rabbit_carrot_problem :
  ∀ (rabbit_holes hamster_holes : ℕ),
    rabbit_holes = hamster_holes - 3 →
    4 * rabbit_holes = 5 * hamster_holes →
    4 * rabbit_holes = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l1953_195373


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1953_195348

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1953_195348


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1953_195361

/-- The cubic polynomial whose roots we're interested in -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 5*x + 7

/-- Theorem stating the properties of the cubic polynomial P and its value at 0 -/
theorem cubic_polynomial_property (P : ℝ → ℝ) (a b c : ℝ) 
  (hf : f a = 0 ∧ f b = 0 ∧ f c = 0)
  (hPa : P a = b + c)
  (hPb : P b = c + a)
  (hPc : P c = a + b)
  (hPsum : P (a + b + c) = -16) :
  P 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1953_195361


namespace NUMINAMATH_CALUDE_num_sequences_eq_248832_l1953_195306

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of class sessions per week -/
def sessions_per_week : ℕ := 5

/-- The number of different sequences of students solving problems in one week -/
def num_sequences : ℕ := num_students ^ sessions_per_week

/-- Theorem stating that the number of different sequences of students solving problems in one week is 248,832 -/
theorem num_sequences_eq_248832 : num_sequences = 248832 := by sorry

end NUMINAMATH_CALUDE_num_sequences_eq_248832_l1953_195306


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l1953_195335

/-- Given a Pythagorean triple (a, b, c) where c is the hypotenuse,
    prove that certain expressions are perfect squares and
    that certain equations are solvable in integers. -/
theorem pythagorean_triple_properties (a b c : ℤ) 
  (h : a^2 + b^2 = c^2) : -- Pythagorean triple condition
  (∃ (k₁ k₂ k₃ k₄ : ℤ), 
    2*(c-a)*(c-b) = k₁^2 ∧ 
    2*(c-a)*(c+b) = k₂^2 ∧ 
    2*(c+a)*(c-b) = k₃^2 ∧ 
    2*(c+a)*(c+b) = k₄^2) ∧ 
  (∃ (x y : ℤ), 
    x + y + (2*x*y).sqrt = c ∧ 
    x + y - (2*x*y).sqrt = c) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l1953_195335


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l1953_195304

theorem collinear_points_b_value :
  ∀ b : ℚ,
  (∃ (m c : ℚ), 
    (m * 4 + c = -6) ∧
    (m * (b + 3) + c = -1) ∧
    (m * (-3 * b + 4) + c = 5)) →
  b = 11 / 26 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l1953_195304


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1953_195391

def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 9 * d + (n - 1 : ℝ) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) :
  d ≠ 0 →
  ∃ k : ℕ, k > 0 ∧ 
    (arithmetic_sequence d k) ^ 2 = 
    (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k)) ∧
    k = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1953_195391


namespace NUMINAMATH_CALUDE_all_drawings_fit_three_notebooks_l1953_195372

/-- Proves that all drawings fit in three notebooks after reorganization --/
theorem all_drawings_fit_three_notebooks 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (new_drawings_per_page : Nat) 
  (h1 : initial_notebooks = 5)
  (h2 : pages_per_notebook = 60)
  (h3 : initial_drawings_per_page = 8)
  (h4 : new_drawings_per_page = 15) :
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) ≤ 
  (3 * pages_per_notebook * new_drawings_per_page) := by
  sorry

#check all_drawings_fit_three_notebooks

end NUMINAMATH_CALUDE_all_drawings_fit_three_notebooks_l1953_195372


namespace NUMINAMATH_CALUDE_bisector_sum_squares_l1953_195346

/-- Given a triangle with side lengths a and b, angle C, and its angle bisector l and
    exterior angle bisector l', the sum of squares of these bisectors is equal to
    (64 R^2 S^2) / ((a^2 - b^2)^2), where R is the circumradius and S is the area of the triangle. -/
theorem bisector_sum_squares (a b l l' R S : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) (hl' : 0 < l') (hR : 0 < R) (hS : 0 < S) :
  l'^2 + l^2 = (64 * R^2 * S^2) / ((a^2 - b^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_bisector_sum_squares_l1953_195346


namespace NUMINAMATH_CALUDE_square_equation_solution_l1953_195327

theorem square_equation_solution : ∃! x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1953_195327


namespace NUMINAMATH_CALUDE_sam_distance_l1953_195356

/-- The distance traveled by Sam given his walking speed and duration -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that Sam's traveled distance is 8 miles -/
theorem sam_distance :
  let speed := 4 -- miles per hour
  let time := 2 -- hours
  distance_traveled speed time = 8 := by sorry

end NUMINAMATH_CALUDE_sam_distance_l1953_195356


namespace NUMINAMATH_CALUDE_polygon_sides_l1953_195312

theorem polygon_sides (n : ℕ) (x : ℝ) : 
  n ≥ 3 →
  0 < x →
  x < 180 →
  (n - 2) * 180 - x + (180 - x) = 500 →
  n = 4 ∨ n = 5 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1953_195312


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1953_195395

-- Define the line l passing through (-2, 1) with slope k
def line (k : ℝ) (x y : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the condition for the line to intersect the parabola at only one point
def unique_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, line k p.1 p.2 ∧ parabola p.1 p.2

-- Theorem statement
theorem line_parabola_intersection (k : ℝ) :
  unique_intersection k → k = 0 ∨ k = -1 ∨ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1953_195395


namespace NUMINAMATH_CALUDE_log_sum_greater_than_two_l1953_195375

theorem log_sum_greater_than_two (x y a : ℝ) (m : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < a) (h4 : a < 1)
  (h5 : m = Real.log x / Real.log a + Real.log y / Real.log a) : 
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_greater_than_two_l1953_195375


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt15_l1953_195341

theorem complex_magnitude_equals_sqrt15 (s : ℝ) :
  Complex.abs (-3 + s * Complex.I) = 3 * Real.sqrt 5 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt15_l1953_195341


namespace NUMINAMATH_CALUDE_differential_of_y_l1953_195377

noncomputable def y (x : ℝ) : ℝ := Real.arctan (Real.sinh x) + (Real.sinh x) * Real.log (Real.cosh x)

theorem differential_of_y (x : ℝ) :
  deriv y x = Real.cosh x * (1 + Real.log (Real.cosh x)) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l1953_195377


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1953_195347

theorem parallelogram_base_length
  (area : ℝ)
  (base : ℝ)
  (altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1953_195347


namespace NUMINAMATH_CALUDE_infinite_decimal_digits_l1953_195355

/-- The decimal representation of 1 / (2^3 * 5^4 * 3^2) has infinitely many digits after the decimal point. -/
theorem infinite_decimal_digits (n : ℕ) : ∃ (k : ℕ), k > n ∧ 
  (10^k * (1 : ℚ) / (2^3 * 5^4 * 3^2)).num ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_infinite_decimal_digits_l1953_195355


namespace NUMINAMATH_CALUDE_jacob_dinner_calories_l1953_195319

/-- Calculates Jacob's dinner calories based on his daily goal, breakfast, lunch, and excess calories --/
theorem jacob_dinner_calories
  (daily_goal : ℕ)
  (breakfast : ℕ)
  (lunch : ℕ)
  (excess : ℕ)
  (h1 : daily_goal = 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : excess = 600) :
  daily_goal + excess - (breakfast + lunch) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_jacob_dinner_calories_l1953_195319


namespace NUMINAMATH_CALUDE_lower_variance_implies_more_stable_l1953_195358

/-- Represents a participant in the math competition -/
structure Participant where
  name : String
  average_score : ℝ
  variance : ℝ

/-- Defines what it means for a participant to have more stable performance -/
def has_more_stable_performance (p1 p2 : Participant) : Prop :=
  p1.average_score = p2.average_score ∧ p1.variance < p2.variance

/-- Theorem stating that the participant with lower variance has more stable performance -/
theorem lower_variance_implies_more_stable
  (xiao_li xiao_zhang : Participant)
  (h1 : xiao_li.name = "Xiao Li")
  (h2 : xiao_zhang.name = "Xiao Zhang")
  (h3 : xiao_li.average_score = 95)
  (h4 : xiao_zhang.average_score = 95)
  (h5 : xiao_li.variance = 0.55)
  (h6 : xiao_zhang.variance = 1.35) :
  has_more_stable_performance xiao_li xiao_zhang :=
sorry

end NUMINAMATH_CALUDE_lower_variance_implies_more_stable_l1953_195358


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1953_195343

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (candidate1_votes : ℕ) (candidate2_votes : ℕ),
    valid_votes = (80 * total_votes) / 100 →
    candidate1_votes = (55 * valid_votes) / 100 →
    candidate2_votes = 2700 →
    candidate1_votes + candidate2_votes = valid_votes →
    total_votes = 7500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1953_195343


namespace NUMINAMATH_CALUDE_sculpture_cost_in_pesos_l1953_195323

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Mexican pesos -/
def usd_to_mxn : ℝ := 20

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating that the cost of the sculpture in Mexican pesos is 400 -/
theorem sculpture_cost_in_pesos :
  (sculpture_cost_nad / usd_to_nad) * usd_to_mxn = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_pesos_l1953_195323


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1953_195301

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1953_195301


namespace NUMINAMATH_CALUDE_johns_allowance_l1953_195317

theorem johns_allowance (A : ℝ) : 
  A > 0 →
  (4/15) * A = 0.88 →
  A = 3.30 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l1953_195317


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l1953_195360

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {x | -7 < 2 + 3*x ∧ 2 + 3*x < 5}

-- State the theorem
theorem complement_of_union_A_B :
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l1953_195360


namespace NUMINAMATH_CALUDE_brothers_age_difference_l1953_195325

/-- The age difference between two brothers -/
def age_difference (mark_age john_age : ℕ) : ℕ :=
  mark_age - john_age

theorem brothers_age_difference :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    parents_age = 5 * john_age →
    parents_age = 40 →
    age_difference mark_age john_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l1953_195325


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1953_195340

/-- Given a circle C with equation x^2 + 12y + 57 = -y^2 - 10x, 
    prove that the sum of its center coordinates and radius is -9 -/
theorem circle_center_radius_sum (x y : ℝ) :
  (∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 + 12*y + 57 = -y^2 - 10*x ↔ (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = -9) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1953_195340


namespace NUMINAMATH_CALUDE_equation_solutions_l1953_195320

theorem equation_solutions :
  (∃ x : ℝ, (2 / (x - 2) = 3 / x) ∧ (x = 6)) ∧
  (∃ x : ℝ, (4 / (x^2 - 1) = (x + 2) / (x - 1) - 1) ∧ (x = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1953_195320


namespace NUMINAMATH_CALUDE_triangle_theorem_l1953_195321

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.a + t.b + t.c = 5) :
  Real.sin t.C / Real.sin t.A = 2 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1953_195321


namespace NUMINAMATH_CALUDE_power_of_five_l1953_195337

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l1953_195337


namespace NUMINAMATH_CALUDE_dannys_physics_marks_l1953_195316

/-- Danny's marks in different subjects and average -/
structure DannyMarks where
  english : ℕ
  mathematics : ℕ
  chemistry : ℕ
  biology : ℕ
  average : ℕ

/-- The theorem stating Danny's marks in Physics -/
theorem dannys_physics_marks (marks : DannyMarks) 
  (h1 : marks.english = 76)
  (h2 : marks.mathematics = 65)
  (h3 : marks.chemistry = 67)
  (h4 : marks.biology = 75)
  (h5 : marks.average = 73)
  (h6 : (marks.english + marks.mathematics + marks.chemistry + marks.biology + marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology)) / 5 = marks.average) :
  marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology) = 82 := by
  sorry


end NUMINAMATH_CALUDE_dannys_physics_marks_l1953_195316


namespace NUMINAMATH_CALUDE_max_a_for_four_near_zero_points_l1953_195342

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Definition of a "near-zero point" -/
def is_near_zero_point (f : ℝ → ℝ) (x : ℤ) : Prop :=
  |f x| ≤ 1/4

theorem max_a_for_four_near_zero_points (a b c : ℝ) (ha : a > 0) :
  (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    is_near_zero_point (quadratic_function a b c) x₁ ∧
    is_near_zero_point (quadratic_function a b c) x₂ ∧
    is_near_zero_point (quadratic_function a b c) x₃ ∧
    is_near_zero_point (quadratic_function a b c) x₄) →
  a ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_four_near_zero_points_l1953_195342


namespace NUMINAMATH_CALUDE_range_of_a_l1953_195385

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + Real.cos x < 2) → 
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1953_195385


namespace NUMINAMATH_CALUDE_parabola_vertex_l1953_195331

/-- A quadratic function f(x) = 2x^2 + px + q with roots at -6 and 4 -/
def f (p q : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + q

theorem parabola_vertex (p q : ℝ) :
  (∀ x ∈ Set.Icc (-6 : ℝ) 4, f p q x ≥ 0) ∧
  (∀ x ∉ Set.Icc (-6 : ℝ) 4, f p q x < 0) →
  ∃ vertex : ℝ × ℝ, vertex = (-1, -50) ∧
    ∀ x : ℝ, f p q x ≥ f p q (-1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1953_195331


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_inscribed_octagon_area_is_1400_l1953_195366

/-- The area of an inscribed octagon in a square -/
theorem inscribed_octagon_area (square_perimeter : ℝ) (h1 : square_perimeter = 160) : ℝ :=
  let square_side := square_perimeter / 4
  let triangle_leg := square_side / 4
  let triangle_area := (1 / 2) * triangle_leg * triangle_leg
  let total_triangle_area := 4 * triangle_area
  let square_area := square_side * square_side
  square_area - total_triangle_area

/-- The area of the inscribed octagon is 1400 square centimeters -/
theorem inscribed_octagon_area_is_1400 (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  inscribed_octagon_area square_perimeter h1 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_inscribed_octagon_area_is_1400_l1953_195366


namespace NUMINAMATH_CALUDE_tan_sum_equals_double_tan_l1953_195300

theorem tan_sum_equals_double_tan (α β : Real) 
  (h : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_double_tan_l1953_195300


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l1953_195364

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    (∀ x' : ℕ+, 5 * x'^7 = 13 * y^11 → x' ≥ x) →
    x = a^c * b^d ∧
    Prime a ∧ Prime b ∧
    a + b + c + d = 62 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l1953_195364


namespace NUMINAMATH_CALUDE_tomato_plants_l1953_195311

theorem tomato_plants (n : ℕ) (sum : ℕ) : 
  n = 12 → sum = 186 → 
  ∃ a d : ℕ, 
    (∀ i : ℕ, i ≤ n → a + (i - 1) * d = sum / n + (2 * i - n - 1) / 2) ∧
    (a + (n - 1) * d = 21) :=
by sorry

end NUMINAMATH_CALUDE_tomato_plants_l1953_195311


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1953_195369

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1953_195369


namespace NUMINAMATH_CALUDE_toms_marble_expense_l1953_195390

/-- Given Tom's expenses, prove the amount spent on marbles --/
theorem toms_marble_expense (skateboard_cost shorts_cost total_toys_cost : ℚ)
  (h1 : skateboard_cost = 9.46)
  (h2 : shorts_cost = 14.50)
  (h3 : total_toys_cost = 19.02) :
  total_toys_cost - skateboard_cost = 9.56 := by
  sorry

#check toms_marble_expense

end NUMINAMATH_CALUDE_toms_marble_expense_l1953_195390


namespace NUMINAMATH_CALUDE_equation_solutions_l1953_195315

theorem equation_solutions : 
  {(x, y) : ℕ × ℕ | x^2 + 6*x*y - 7*y^2 = 2009 ∧ x > 0 ∧ y > 0} = 
  {(252, 251), (42, 35), (42, 1)} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1953_195315


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l1953_195363

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 72 - 45

/-- The number of pencils Nancy added to the drawer -/
def added_pencils : ℕ := 45

/-- The total number of pencils in the drawer after Nancy added more -/
def total_pencils : ℕ := 72

theorem pencils_in_drawer :
  initial_pencils + added_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l1953_195363


namespace NUMINAMATH_CALUDE_points_in_circle_l1953_195330

theorem points_in_circle (points : Finset (ℝ × ℝ)) : 
  (points.card = 51) →
  (∀ p ∈ points, p.1 ∈ Set.Icc (0 : ℝ) 1 ∧ p.2 ∈ Set.Icc (0 : ℝ) 1) →
  ∃ c : ℝ × ℝ, ∃ s : Finset (ℝ × ℝ), 
    s ⊆ points ∧ 
    s.card = 3 ∧ 
    (∀ p ∈ s, (p.1 - c.1)^2 + (p.2 - c.2)^2 ≤ (1/7)^2) :=
by sorry

end NUMINAMATH_CALUDE_points_in_circle_l1953_195330


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1953_195310

theorem tangent_line_to_circle (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0) →  -- circle equation
  (x = 1 ∧ y = Real.sqrt 3) →  -- point of tangency
  (x - Real.sqrt 3 * y + 2 = 0)  -- equation of tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1953_195310


namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l1953_195383

theorem power_seven_mod_eight : 7^202 % 8 = 1 := by sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l1953_195383


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l1953_195368

theorem ratio_sum_equality (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : c / d = 3 / 4) 
  (h3 : b ≠ 0) 
  (h4 : d ≠ 0) 
  (h5 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l1953_195368


namespace NUMINAMATH_CALUDE_comparison_theorem_l1953_195332

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ -(-3) > -|(-3)| := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1953_195332


namespace NUMINAMATH_CALUDE_first_car_distance_l1953_195322

theorem first_car_distance (total_distance : ℝ) (second_car_distance : ℝ) (side_distance : ℝ) (final_distance : ℝ) 
  (h1 : total_distance = 113)
  (h2 : second_car_distance = 35)
  (h3 : side_distance = 15)
  (h4 : final_distance = 28) :
  ∃ x : ℝ, x = 17.5 ∧ total_distance - (2 * x + side_distance + second_car_distance) = final_distance :=
by
  sorry


end NUMINAMATH_CALUDE_first_car_distance_l1953_195322


namespace NUMINAMATH_CALUDE_angle_convergence_point_l1953_195376

theorem angle_convergence_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by sorry

end NUMINAMATH_CALUDE_angle_convergence_point_l1953_195376


namespace NUMINAMATH_CALUDE_fred_newspaper_earnings_l1953_195365

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings (total_earnings washing_earnings : ℕ) : ℕ :=
  total_earnings - washing_earnings

theorem fred_newspaper_earnings :
  newspaper_earnings 90 74 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fred_newspaper_earnings_l1953_195365


namespace NUMINAMATH_CALUDE_lcm_18_45_l1953_195392

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_45_l1953_195392


namespace NUMINAMATH_CALUDE_max_expression_proof_l1953_195379

/-- The maximum value of c * a^b - d given the constraints --/
def max_expression : ℕ := 625

/-- The set of possible values for a, b, c, and d --/
def value_set : Finset ℕ := {0, 1, 4, 5}

/-- Proposition: The maximum value of c * a^b - d is 625, given the constraints --/
theorem max_expression_proof :
  ∀ a b c d : ℕ,
    a ∈ value_set → b ∈ value_set → c ∈ value_set → d ∈ value_set →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    c * a^b - d ≤ max_expression :=
by sorry

end NUMINAMATH_CALUDE_max_expression_proof_l1953_195379


namespace NUMINAMATH_CALUDE_probability_A_rolls_correct_l1953_195329

/-- The probability that player A rolls on the n-th turn in a dice game with the following rules:
  - A and B take turns rolling a die, with A going first.
  - If A rolls a 1, A continues to roll; otherwise, it's B's turn.
  - If B rolls a 3, B continues to roll; otherwise, it's A's turn. -/
def probability_A_rolls (n : ℕ) : ℚ :=
  1/2 - 1/3 * (-2/3)^(n-2)

/-- Theorem stating that the probability A rolls on the n-th turn is given by the formula -/
theorem probability_A_rolls_correct (n : ℕ) :
  probability_A_rolls n = 1/2 - 1/3 * (-2/3)^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_probability_A_rolls_correct_l1953_195329


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l1953_195326

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l1953_195326


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1953_195318

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    boys_percentage = 30 / 100 →
    (boys_percentage * total_students : ℚ).num = 90 →
    total_students = 300 :=
by sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1953_195318


namespace NUMINAMATH_CALUDE_monday_temperature_l1953_195351

def sunday_temp : ℝ := 40
def tuesday_temp : ℝ := 65
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem monday_temperature (monday_temp : ℝ) :
  (sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp →
  monday_temp = 50 := by
sorry

end NUMINAMATH_CALUDE_monday_temperature_l1953_195351


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l1953_195380

theorem quadratic_roots_bound (a b : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0) →  -- Equation has real roots
  (α^2 + a*α + b = 0) →           -- α is a root
  (β^2 + a*β + b = 0) →           -- β is a root
  (α ≠ β) →                       -- Roots are distinct
  (2 * abs a < 4 + b ∧ abs b < 4 ↔ abs α < 2 ∧ abs β < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l1953_195380


namespace NUMINAMATH_CALUDE_ampersand_composition_l1953_195313

-- Define the operations
def ampersand (y : ℤ) : ℤ := 2 * (7 - y)
def ampersandbar (y : ℤ) : ℤ := 2 * (y - 7)

-- State the theorem
theorem ampersand_composition : ampersandbar (ampersand (-13)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l1953_195313


namespace NUMINAMATH_CALUDE_square_root_problem_l1953_195386

theorem square_root_problem (y z x : ℝ) (hy : y > 0) (hx : x > 0) :
  y^z = (Real.sqrt 16)^3 → x^2 = y^z → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1953_195386


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1953_195399

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ (1 / 3) := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1953_195399


namespace NUMINAMATH_CALUDE_marta_took_ten_books_l1953_195338

/-- The number of books Marta took off the shelf -/
def books_taken (initial_books : ℝ) (remaining_books : ℕ) : ℝ :=
  initial_books - remaining_books

/-- Theorem stating that Marta took 10 books off the shelf -/
theorem marta_took_ten_books : books_taken 38.0 28 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marta_took_ten_books_l1953_195338


namespace NUMINAMATH_CALUDE_shaded_area_theorem_total_shaded_area_l1953_195388

-- Define the length of the diagonal
def diagonal_length : ℝ := 8

-- Define the number of congruent squares
def num_squares : ℕ := 25

-- Theorem statement
theorem shaded_area_theorem (diagonal : ℝ) (num_squares : ℕ) 
  (h1 : diagonal = diagonal_length) 
  (h2 : num_squares = num_squares) : 
  (diagonal^2 / 2) = 32 := by
  sorry

-- Main theorem connecting the given conditions to the final area
theorem total_shaded_area : 
  (diagonal_length^2 / 2) = 32 := by
  exact shaded_area_theorem diagonal_length num_squares rfl rfl

end NUMINAMATH_CALUDE_shaded_area_theorem_total_shaded_area_l1953_195388


namespace NUMINAMATH_CALUDE_cats_adopted_l1953_195370

/-- Proves the number of cats adopted given the shelter's cat population changes -/
theorem cats_adopted (initial_cats : ℕ) (new_cats : ℕ) (kittens_born : ℕ) (cat_picked_up : ℕ) (final_cats : ℕ) :
  initial_cats = 6 →
  new_cats = 12 →
  kittens_born = 5 →
  cat_picked_up = 1 →
  final_cats = 19 →
  initial_cats + new_cats - (initial_cats + new_cats + kittens_born - cat_picked_up - final_cats) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cats_adopted_l1953_195370


namespace NUMINAMATH_CALUDE_train_average_speed_l1953_195349

/-- 
Given a train that travels two distances in two time periods, 
this theorem proves that its average speed is the total distance divided by the total time.
-/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) 
  (h2 : time1 = 3.5)
  (h3 : distance2 = 470)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 106 := by
sorry

end NUMINAMATH_CALUDE_train_average_speed_l1953_195349


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_approx_248_30_l1953_195328

/-- Calculates the length of a bridge given train and wind conditions --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) 
  (deceleration : ℝ) (headwind_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let headwind_speed_ms := headwind_speed_kmh * 1000 / 3600
  let effective_speed := train_speed_ms - headwind_speed_ms
  let deceleration_distance := effective_speed^2 / (2 * deceleration)
  deceleration_distance + train_length

/-- The bridge length is approximately 248.30 meters --/
theorem bridge_length_approx_248_30 :
  ∃ ε > 0, |bridge_length_calculation 200 60 2 10 - 248.30| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_approx_248_30_l1953_195328


namespace NUMINAMATH_CALUDE_circles_intersect_iff_distance_between_radii_sum_and_diff_l1953_195307

/-- Two circles intersect if and only if the distance between their centers
    is greater than the absolute difference of their radii and less than
    the sum of their radii. -/
theorem circles_intersect_iff_distance_between_radii_sum_and_diff
  (R r d : ℝ) (h : R ≥ r) :
  (∃ (p : ℝ × ℝ), (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧ 
                  (p.1 - d)^2 + p.2^2 = r^2) ↔
  (R - r < d ∧ d < R + r) :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_distance_between_radii_sum_and_diff_l1953_195307


namespace NUMINAMATH_CALUDE_undefined_at_eleven_l1953_195381

theorem undefined_at_eleven (x : ℝ) : 
  (∃ y, (3 * x^2 + 5) / (x^2 - 22*x + 121) = y) ↔ x ≠ 11 :=
by sorry

end NUMINAMATH_CALUDE_undefined_at_eleven_l1953_195381


namespace NUMINAMATH_CALUDE_odd_function_value_l1953_195353

theorem odd_function_value (f : ℝ → ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x^2 + 1/x) →  -- definition of f for x > 0
  f (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l1953_195353


namespace NUMINAMATH_CALUDE_sin_cos_sum_equivalence_l1953_195352

theorem sin_cos_sum_equivalence (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equivalence_l1953_195352


namespace NUMINAMATH_CALUDE_cube_paint_equality_l1953_195396

/-- The number of unit cubes with exactly one face painted in a cube of side length n -/
def one_face_painted (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- The number of unit cubes with exactly two faces painted in a cube of side length n -/
def two_faces_painted (n : ℕ) : ℕ := 12 * (n - 2)

theorem cube_paint_equality (n : ℕ) (h : n > 3) :
  one_face_painted n = two_faces_painted n ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_paint_equality_l1953_195396


namespace NUMINAMATH_CALUDE_probability_x_equals_y_l1953_195367

-- Define the range for x and y
def valid_range (x : ℝ) : Prop := -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi

-- Define the condition for x and y
def condition (x y : ℝ) : Prop := Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Define the total number of valid pairs
def total_pairs : ℕ := 121

-- Define the number of pairs where X = Y
def equal_pairs : ℕ := 11

-- State the theorem
theorem probability_x_equals_y :
  (∀ x y : ℝ, valid_range x → valid_range y → condition x y) →
  (equal_pairs : ℕ) / (total_pairs : ℕ) = 1 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_x_equals_y_l1953_195367


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1953_195309

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) →
  z = -(1/2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1953_195309


namespace NUMINAMATH_CALUDE_rice_weight_calculation_l1953_195394

theorem rice_weight_calculation (total : ℚ) : 
  (total * (1 - 3/10) * (1 - 2/5) = 210) → total = 500 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_calculation_l1953_195394


namespace NUMINAMATH_CALUDE_jimmy_lodging_expenses_l1953_195374

/-- Jimmy's lodging expenses during vacation -/
theorem jimmy_lodging_expenses :
  let hostel_nights : ℕ := 3
  let hostel_rate : ℕ := 15
  let cabin_nights : ℕ := 2
  let cabin_total_rate : ℕ := 45
  let cabin_friends : ℕ := 2
  
  let hostel_cost := hostel_nights * hostel_rate
  let cabin_cost := cabin_nights * (cabin_total_rate / (cabin_friends + 1))
  
  hostel_cost + cabin_cost = 75 := by sorry

end NUMINAMATH_CALUDE_jimmy_lodging_expenses_l1953_195374


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_line_segment_l1953_195362

-- Define the line segment
def line_segment (x y : ℝ) : Prop := x - 2*y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_line_segment :
  ∀ x y : ℝ, line_segment x y →
  ∃ x' y' : ℝ, perpendicular_bisector x' y' ∧
  (x' = (x + (-1))/2 ∧ y' = (y + 0)/2) ∧
  (2*x' - y' - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_line_segment_l1953_195362


namespace NUMINAMATH_CALUDE_halfway_point_fractions_l1953_195393

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/8) (hb : b = 1/3) :
  (a + b) / 2 = 11/48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_fractions_l1953_195393


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l1953_195371

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l1953_195371


namespace NUMINAMATH_CALUDE_river_depth_l1953_195350

/-- Given a river with specified width, flow rate, and volume flow rate, calculate its depth. -/
theorem river_depth (width : ℝ) (flow_rate_kmph : ℝ) (volume_flow_rate : ℝ) :
  width = 75 →
  flow_rate_kmph = 4 →
  volume_flow_rate = 35000 →
  (volume_flow_rate / (flow_rate_kmph * 1000 / 60) / width) = 7 := by
  sorry

#check river_depth

end NUMINAMATH_CALUDE_river_depth_l1953_195350


namespace NUMINAMATH_CALUDE_trig_identity_l1953_195397

theorem trig_identity (x : Real) : 
  (Real.cos x)^4 + (Real.sin x)^4 + 3*(Real.sin x)^2*(Real.cos x)^2 = 
  (Real.cos x)^6 + (Real.sin x)^6 + 4*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1953_195397


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1953_195398

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a - b)^2 = 4 * (a * b)^3) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x - y)^2 = 4 * (x * y)^3 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1953_195398


namespace NUMINAMATH_CALUDE_original_group_size_l1953_195314

theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) :
  initial_days = 8 ∧ absent_men = 3 ∧ final_days = 10 →
  ∃ original_size : ℕ, 
    original_size > absent_men ∧
    (original_size : ℚ) / initial_days = (original_size - absent_men) / final_days ∧
    original_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l1953_195314


namespace NUMINAMATH_CALUDE_max_correct_answers_l1953_195336

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 60 → 
  correct_points = 5 → 
  blank_points = 0 → 
  incorrect_points = -2 → 
  total_score = 150 → 
  (∃ (correct blank incorrect : ℕ), 
    correct + blank + incorrect = total_questions ∧ 
    correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score ∧ 
    ∀ (other_correct : ℕ), 
      (∃ (other_blank other_incorrect : ℕ), 
        other_correct + other_blank + other_incorrect = total_questions ∧ 
        correct_points * other_correct + blank_points * other_blank + incorrect_points * other_incorrect = total_score) → 
      other_correct ≤ 38) ∧ 
  (∃ (blank incorrect : ℕ), 
    38 + blank + incorrect = total_questions ∧ 
    correct_points * 38 + blank_points * blank + incorrect_points * incorrect = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1953_195336


namespace NUMINAMATH_CALUDE_sin_theta_value_l1953_195345

theorem sin_theta_value (a : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real.sin θ = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1953_195345


namespace NUMINAMATH_CALUDE_power_36_equals_power_16_9_l1953_195359

theorem power_36_equals_power_16_9 (m n : ℤ) : 
  (36 : ℝ) ^ (m + n) = (16 : ℝ) ^ (m * n) * (9 : ℝ) ^ (m * n) := by
  sorry

end NUMINAMATH_CALUDE_power_36_equals_power_16_9_l1953_195359


namespace NUMINAMATH_CALUDE_complement_A_union_B_l1953_195382

def U : Set ℕ := {n | n > 0 ∧ n < 9}
def A : Set ℕ := {n ∈ U | n % 2 = 1}
def B : Set ℕ := {n ∈ U | n % 3 = 0}

theorem complement_A_union_B : (U \ (A ∪ B)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l1953_195382


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1953_195344

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x^(1/4) = 18 / (9 - x^(1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1953_195344


namespace NUMINAMATH_CALUDE_divisible_by_42_l1953_195387

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := by sorry

end NUMINAMATH_CALUDE_divisible_by_42_l1953_195387


namespace NUMINAMATH_CALUDE_average_b_c_l1953_195354

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : c - a = 50) :
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_l1953_195354


namespace NUMINAMATH_CALUDE_f_not_increasing_l1953_195339

-- Define the function
def f (x : ℝ) : ℝ := |3 - x|

-- State the theorem
theorem f_not_increasing :
  ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y → f x ≤ f y) :=
sorry

end NUMINAMATH_CALUDE_f_not_increasing_l1953_195339


namespace NUMINAMATH_CALUDE_lulu_ice_cream_expense_l1953_195308

theorem lulu_ice_cream_expense (initial_amount : ℝ) (ice_cream_cost : ℝ) (final_cash : ℝ) :
  initial_amount = 65 →
  final_cash = 24 →
  final_cash = (4/5) * (1/2) * (initial_amount - ice_cream_cost) →
  ice_cream_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_lulu_ice_cream_expense_l1953_195308


namespace NUMINAMATH_CALUDE_fraction_of_pet_owners_l1953_195357

/-- Proves that the fraction of freshmen and sophomores who own a pet is 1/5 -/
theorem fraction_of_pet_owners (total_students : ℕ) (freshmen_sophomores : ℕ) (no_pet : ℕ) :
  total_students = 400 →
  freshmen_sophomores = total_students / 2 →
  no_pet = 160 →
  (freshmen_sophomores - no_pet) / freshmen_sophomores = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_pet_owners_l1953_195357


namespace NUMINAMATH_CALUDE_population_in_scientific_notation_l1953_195303

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population : ℝ := 4.6e9
  toScientificNotation population = ScientificNotation.mk 4.6 9 := by
  sorry

end NUMINAMATH_CALUDE_population_in_scientific_notation_l1953_195303


namespace NUMINAMATH_CALUDE_sum_of_zeros_transformed_parabola_l1953_195384

/-- The sum of zeros of a transformed parabola -/
theorem sum_of_zeros_transformed_parabola : 
  let f (x : ℝ) := (x - 3)^2 + 4
  let g (x : ℝ) := -(x - 7)^2 + 7
  ∃ a b : ℝ, g a = 0 ∧ g b = 0 ∧ a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_transformed_parabola_l1953_195384
