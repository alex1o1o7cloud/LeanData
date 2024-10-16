import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l3044_304443

theorem complex_equation_solution : ∃ (x : ℂ), 5 + 2 * Complex.I * x = -3 - 6 * Complex.I * x ∧ x = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3044_304443


namespace NUMINAMATH_CALUDE_triangle_property_l3044_304412

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - 5)^2 + |t.b - 12| + (t.c - 13)^2 = 0

-- Define what it means to be a right triangle with c as hypotenuse
def is_right_triangle_with_c_hypotenuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  is_right_triangle_with_c_hypotenuse t :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l3044_304412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3044_304465

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_4 + a_8 = 16, a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 16) : 
  a 2 + a 10 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3044_304465


namespace NUMINAMATH_CALUDE_power_sum_inequality_l3044_304415

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_pyth : a^2 + b^2 = c^2) (hn : n > 2) : a^n + b^n < c^n := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l3044_304415


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l3044_304469

theorem triangle_trig_max_value (A B C : ℝ) : 
  A = π / 4 → 
  A + B + C = π → 
  0 < B → 
  B < π → 
  0 < C → 
  C < π → 
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l3044_304469


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3044_304418

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3044_304418


namespace NUMINAMATH_CALUDE_max_profit_optimal_plan_model_b_units_l3044_304482

/-- Represents the profit function for tablet sales -/
def profit_function (x : ℕ) : ℝ := -100 * x + 10000

/-- Represents the total cost function for tablet purchases -/
def total_cost (x : ℕ) : ℝ := 1600 * x + 2500 * (20 - x)

/-- Theorem stating the maximum profit and optimal purchasing plan -/
theorem max_profit_optimal_plan :
  ∃ (x : ℕ),
    x ≤ 20 ∧
    total_cost x ≤ 39200 ∧
    profit_function x ≥ 8500 ∧
    (∀ (y : ℕ), y ≤ 20 → total_cost y ≤ 39200 → profit_function y ≥ 8500 →
      profit_function x ≥ profit_function y) ∧
    x = 12 ∧
    profit_function x = 8800 :=
by sorry

/-- Corollary stating the number of units for model B tablets -/
theorem model_b_units (x : ℕ) (h : x = 12) : 20 - x = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_optimal_plan_model_b_units_l3044_304482


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3044_304416

-- Define the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the given sum of angles
def given_sum : ℝ := 2340

-- Theorem statement
theorem convex_polygon_sides : 
  ∃ (n : ℕ), n > 2 ∧ 
  sum_interior_angles n - given_sum > 0 ∧ 
  sum_interior_angles n - given_sum ≤ 360 ∧
  n = 16 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l3044_304416


namespace NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l3044_304486

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (contained_in : Line → Plane → Prop)

-- Define a property for a line being parallel to countless lines in a plane
variable (parallel_to_countless_lines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines 
  (a b : Line) (α : Plane) :
  parallel a b → contained_in b α → 
  parallel_to_countless_lines a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l3044_304486


namespace NUMINAMATH_CALUDE_percentage_difference_l3044_304448

theorem percentage_difference : 
  (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3044_304448


namespace NUMINAMATH_CALUDE_right_triangle_area_l3044_304481

theorem right_triangle_area (h : ℝ) (angle : ℝ) : 
  h = 13 → angle = 45 → 
  let area := (1/2) * (h / Real.sqrt 2) * (h / Real.sqrt 2)
  area = 84.5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3044_304481


namespace NUMINAMATH_CALUDE_pages_with_text_l3044_304438

/-- Given a book with the following properties:
  * It has 98 pages in total
  * Half of the pages are filled with images
  * 11 pages are for introduction
  * The remaining pages are equally split between blank and text
  Prove that the number of pages with text is 19 -/
theorem pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_pages_with_text_l3044_304438


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3044_304471

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse in 2D space -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if four points form a trapezoid with bases parallel to x-axis -/
def isTrapezoid (p1 p2 p3 p4 : Point) : Prop :=
  (p1.y = p2.y) ∧ (p3.y = p4.y) ∧ (p1.y ≠ p3.y)

/-- Check if a point lies on the vertical bisector of a trapezoid -/
def onVerticalBisector (p : Point) (p1 p2 p3 p4 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2

theorem ellipse_major_axis_length
  (p1 p2 p3 p4 p5 : Point)
  (h1 : p1 = ⟨0, 0⟩)
  (h2 : p2 = ⟨4, 0⟩)
  (h3 : p3 = ⟨1, 3⟩)
  (h4 : p4 = ⟨3, 3⟩)
  (h5 : p5 = ⟨-1, 3/2⟩)
  (h_trapezoid : isTrapezoid p1 p2 p3 p4)
  (h_bisector : onVerticalBisector p5 p1 p2 p3 p4)
  (e : Ellipse)
  (h_on_ellipse : pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ pointOnEllipse p4 e ∧ pointOnEllipse p5 e)
  (h_axes_parallel : e.center.x = (p1.x + p2.x) / 2 ∧ e.center.y = (p1.y + p3.y) / 2) :
  2 * e.a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3044_304471


namespace NUMINAMATH_CALUDE_shirts_remaining_l3044_304457

theorem shirts_remaining (initial_shirts sold_shirts : ℕ) 
  (h1 : initial_shirts = 49)
  (h2 : sold_shirts = 21) :
  initial_shirts - sold_shirts = 28 := by
  sorry

end NUMINAMATH_CALUDE_shirts_remaining_l3044_304457


namespace NUMINAMATH_CALUDE_new_teacher_student_ratio_l3044_304444

/-- Proves that given the initial conditions, the new ratio of teachers to students is 1:25 -/
theorem new_teacher_student_ratio
  (initial_ratio : ℚ)
  (initial_teachers : ℕ)
  (student_increase : ℕ)
  (teacher_increase : ℕ)
  (new_student_ratio : ℚ)
  (h1 : initial_ratio = 50 / 1)
  (h2 : initial_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_ratio = 25 / 1) :
  (initial_teachers + teacher_increase) / (initial_ratio * initial_teachers + student_increase) = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_new_teacher_student_ratio_l3044_304444


namespace NUMINAMATH_CALUDE_no_finite_planes_cover_all_cubes_l3044_304407

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the specifics of a plane for this statement

/-- Represents a cube in the integer grid -/
structure GridCube where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Checks if a plane intersects a grid cube -/
def plane_intersects_cube (p : Plane) (c : GridCube) : Prop :=
  sorry -- Definition not needed for the statement

/-- The main theorem stating that it's impossible to have a finite number of planes
    intersecting all cubes in the integer grid -/
theorem no_finite_planes_cover_all_cubes :
  ∀ (planes : Finset Plane), ∃ (c : GridCube),
    ∀ (p : Plane), p ∈ planes → ¬(plane_intersects_cube p c) := by
  sorry


end NUMINAMATH_CALUDE_no_finite_planes_cover_all_cubes_l3044_304407


namespace NUMINAMATH_CALUDE_exists_non_acute_triangle_with_two_acute_angles_l3044_304404

-- Define what an acute angle is
def is_acute_angle (angle : Real) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define what a right angle is
def is_right_angle (angle : Real) : Prop := angle = Real.pi / 2

-- Define a triangle structure
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_angles : angle1 + angle2 + angle3 = Real.pi

-- Define what an acute triangle is
def is_acute_triangle (t : Triangle) : Prop :=
  is_acute_angle t.angle1 ∧ is_acute_angle t.angle2 ∧ is_acute_angle t.angle3

-- Theorem statement
theorem exists_non_acute_triangle_with_two_acute_angles :
  ∃ (t : Triangle), (is_acute_angle t.angle1 ∧ is_acute_angle t.angle2) ∧ ¬is_acute_triangle t :=
sorry

end NUMINAMATH_CALUDE_exists_non_acute_triangle_with_two_acute_angles_l3044_304404


namespace NUMINAMATH_CALUDE_candidate_c_wins_l3044_304422

/-- Represents a candidate in the election --/
inductive Candidate
  | A
  | B
  | C
  | D
  | E

/-- Returns the vote count for a given candidate --/
def votes (c : Candidate) : Float :=
  match c with
  | Candidate.A => 4237.5
  | Candidate.B => 7298.25
  | Candidate.C => 12498.75
  | Candidate.D => 8157.5
  | Candidate.E => 3748.3

/-- Calculates the total number of votes --/
def totalVotes : Float :=
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E

/-- Calculates the percentage of votes for a given candidate --/
def votePercentage (c : Candidate) : Float :=
  (votes c / totalVotes) * 100

/-- Theorem stating that Candidate C has the highest percentage of votes --/
theorem candidate_c_wins :
  ∀ c : Candidate, c ≠ Candidate.C → votePercentage Candidate.C > votePercentage c :=
by sorry

end NUMINAMATH_CALUDE_candidate_c_wins_l3044_304422


namespace NUMINAMATH_CALUDE_ribbon_cost_comparison_l3044_304462

/-- Represents the cost and quantity of ribbons --/
structure RibbonPurchase where
  cost : ℕ
  quantity : ℕ

/-- Determines if one ribbon is cheaper than another --/
def isCheaper (r1 r2 : RibbonPurchase) : Prop :=
  r1.cost * r2.quantity < r2.cost * r1.quantity

theorem ribbon_cost_comparison 
  (yellow blue : RibbonPurchase)
  (h_yellow : yellow.cost = 24)
  (h_blue : blue.cost = 36) :
  (∃ y b, isCheaper {cost := 24, quantity := y} {cost := 36, quantity := b}) ∧
  (∃ y b, isCheaper {cost := 36, quantity := b} {cost := 24, quantity := y}) ∧
  (∃ y b, yellow.cost * b = blue.cost * y) :=
sorry

end NUMINAMATH_CALUDE_ribbon_cost_comparison_l3044_304462


namespace NUMINAMATH_CALUDE_stuffed_animals_ratio_l3044_304408

/-- Proves the ratio of Kenley's stuffed animals to McKenna's is 2:1 --/
theorem stuffed_animals_ratio :
  let mcKenna : ℕ := 34
  let total : ℕ := 175
  let kenley : ℕ := (total - mcKenna - 5) / 2
  (kenley : ℚ) / mcKenna = 2 := by sorry

end NUMINAMATH_CALUDE_stuffed_animals_ratio_l3044_304408


namespace NUMINAMATH_CALUDE_division_equality_l3044_304413

theorem division_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (3 * a * b)) / (b / (3 * a)) = 1 / (b ^ 2) := by
sorry

end NUMINAMATH_CALUDE_division_equality_l3044_304413


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l3044_304400

-- Define a triangle as three points in a 2D plane
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_inequality (A B C P Q R : ℝ × ℝ) :
  Triangle A B C →
  PointOnSegment P B C →
  PointOnSegment Q C A →
  PointOnSegment R A B →
  min (TriangleArea A Q R) (min (TriangleArea B R P) (TriangleArea C P Q)) ≤ (1/4) * TriangleArea A B C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l3044_304400


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3044_304466

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 7*I
  z₁ / z₂ = 29/53 - (31/53)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3044_304466


namespace NUMINAMATH_CALUDE_average_increase_is_four_l3044_304428

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ

/-- Calculates the increase in average runs after the next innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let currentAverage : ℚ := player.totalRuns / player.innings
  let newTotalRuns : ℕ := player.totalRuns + player.nextInningsRuns
  let newAverage : ℚ := newTotalRuns / (player.innings + 1)
  newAverage - currentAverage

/-- Theorem: The increase in average runs is 4 for the given conditions -/
theorem average_increase_is_four :
  ∀ (player : CricketPlayer),
    player.innings = 10 →
    player.totalRuns = 400 →
    player.nextInningsRuns = 84 →
    averageIncrease player = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_four_l3044_304428


namespace NUMINAMATH_CALUDE_sequence_problem_l3044_304410

theorem sequence_problem (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_relation : ∀ n, (a (n + 1))^2 + (a n)^2 = 2 * n * ((a (n + 1))^2 - (a n)^2)) :
  a 113 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3044_304410


namespace NUMINAMATH_CALUDE_finite_cuboidal_blocks_l3044_304421

theorem finite_cuboidal_blocks :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ a b c : ℕ,
    0 < c ∧ c ≤ b ∧ b ≤ a ∧ a * b * c = 2 * (a - 2) * (b - 2) * (c - 2) →
    (a, b, c) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_cuboidal_blocks_l3044_304421


namespace NUMINAMATH_CALUDE_expected_sum_of_two_marbles_l3044_304453

def marbleSet : Finset ℕ := Finset.range 6

def marblePairs : Finset (ℕ × ℕ) :=
  (marbleSet.product marbleSet).filter (fun p => p.1 < p.2)

def pairSum (p : ℕ × ℕ) : ℕ := p.1 + p.2 + 2

theorem expected_sum_of_two_marbles :
  (marblePairs.sum pairSum) / marblePairs.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_sum_of_two_marbles_l3044_304453


namespace NUMINAMATH_CALUDE_fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3044_304475

theorem fraction_between (a b : ℚ) (t : ℚ) (h : 0 ≤ t ∧ t ≤ 1) :
  a + t * (b - a) = (1 - t) * a + t * b :=
by sorry

theorem one_quarter_between_one_seventh_and_one_fourth :
  (1 : ℚ)/7 + (1/4) * ((1/4) - (1/7)) = 23/112 :=
by sorry

end NUMINAMATH_CALUDE_fraction_between_one_quarter_between_one_seventh_and_one_fourth_l3044_304475


namespace NUMINAMATH_CALUDE_integer_divisibility_in_range_l3044_304485

theorem integer_divisibility_in_range (n : ℕ+) : 
  ∃ (a b c : ℤ), 
    (n : ℤ)^2 < a ∧ a < b ∧ b < c ∧ c < (n : ℤ)^2 + n + 3 * Real.sqrt n ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c) % a = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_in_range_l3044_304485


namespace NUMINAMATH_CALUDE_quadratic_factorization_problem_l3044_304425

theorem quadratic_factorization_problem :
  ∀ (a b : ℕ), 
    (∀ x : ℝ, x^2 - 20*x + 96 = (x - a)*(x - b)) →
    a > b →
    2*b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_problem_l3044_304425


namespace NUMINAMATH_CALUDE_time_spent_on_other_subjects_l3044_304493

def total_time : ℝ := 150

def math_percent : ℝ := 0.20
def science_percent : ℝ := 0.25
def history_percent : ℝ := 0.10
def english_percent : ℝ := 0.15

def min_time_remaining_subject : ℝ := 30

theorem time_spent_on_other_subjects :
  let math_time := total_time * math_percent
  let science_time := total_time * science_percent
  let history_time := total_time * history_percent
  let english_time := total_time * english_percent
  let known_subjects_time := math_time + science_time + history_time + english_time
  let remaining_time := total_time - known_subjects_time
  remaining_time - min_time_remaining_subject = 15 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_other_subjects_l3044_304493


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3044_304433

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) : x + q = 2*q - 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3044_304433


namespace NUMINAMATH_CALUDE_equation_simplification_l3044_304419

theorem equation_simplification (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) + 3) * (x - 1) = 1 + 3 * (x - 1) ∧
  3 * x / (1 - x) * (x - 1) = -3 * x ∧
  1 + 3 * (x - 1) = -3 * x :=
sorry

end NUMINAMATH_CALUDE_equation_simplification_l3044_304419


namespace NUMINAMATH_CALUDE_base7_arithmetic_l3044_304483

/-- Represents a number in base 7 --/
structure Base7 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 7

/-- Addition operation for Base7 numbers --/
def add_base7 (a b : Base7) : Base7 := sorry

/-- Subtraction operation for Base7 numbers --/
def sub_base7 (a b : Base7) : Base7 := sorry

/-- Conversion from a natural number to Base7 --/
def nat_to_base7 (n : Nat) : Base7 := sorry

theorem base7_arithmetic :
  let a := nat_to_base7 24
  let b := nat_to_base7 356
  let c := nat_to_base7 105
  let d := nat_to_base7 265
  sub_base7 (add_base7 a b) c = d := by sorry

end NUMINAMATH_CALUDE_base7_arithmetic_l3044_304483


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3044_304467

theorem expression_equals_negative_one (a : ℝ) (ha : a ≠ 0) :
  ∀ y : ℝ, y ≠ a ∧ y ≠ -a →
    (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3044_304467


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3044_304491

theorem right_angled_triangle (α β γ : Real) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ)
  (h4 : α + β + γ = Real.pi) 
  (h5 : (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ) : 
  γ = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3044_304491


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3044_304427

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3044_304427


namespace NUMINAMATH_CALUDE_local_value_in_product_l3044_304468

/-- The face value of a digit is the digit itself. -/
def faceValue (digit : ℕ) : ℕ := digit

/-- The local value of a digit in a number is the digit multiplied by its place value. -/
def localValue (digit : ℕ) (placeValue : ℕ) : ℕ := digit * placeValue

/-- The product of two numbers. -/
def product (a b : ℕ) : ℕ := a * b

/-- The theorem stating that the local value of 6 in the product of the face value of 7
    and the local value of 8 in 7098060 is equal to 60. -/
theorem local_value_in_product :
  let number := 7098060
  let faceValue7 := faceValue 7
  let localValue8 := localValue 8 1000
  let prod := product faceValue7 localValue8
  localValue 6 10 = 60 :=
by sorry

end NUMINAMATH_CALUDE_local_value_in_product_l3044_304468


namespace NUMINAMATH_CALUDE_sum_of_digits_l3044_304451

theorem sum_of_digits (a b c d e : ℕ) : 
  (10 ≤ 10*a + b) ∧ (10*a + b ≤ 99) ∧
  (100 ≤ 100*c + 10*d + e) ∧ (100*c + 10*d + e ≤ 999) ∧
  (10*a + b + 100*c + 10*d + e = 1079) →
  a + b + c + d + e = 35 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3044_304451


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l3044_304446

/-- A triangle with given altitudes has a specific area -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h_pos₁ : h₁ > 0) (h_pos₂ : h₂ > 0) (h_pos₃ : h₃ > 0) :
  h₁ = 12 → h₂ = 15 → h₃ = 20 → ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * 150) ∧ (b * h₂ = 2 * 150) ∧ (c * h₃ = 2 * 150) :=
by sorry

#check triangle_area_from_altitudes

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l3044_304446


namespace NUMINAMATH_CALUDE_marley_samantha_apple_ratio_l3044_304409

/-- Proves that the ratio of Marley's apples to Samantha's apples is 3:1 -/
theorem marley_samantha_apple_ratio :
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_total_fruits : ℕ := 31
  let marley_apples : ℕ := marley_total_fruits - marley_oranges
  (marley_apples : ℚ) / samantha_apples = 3 / 1 := by
  sorry


end NUMINAMATH_CALUDE_marley_samantha_apple_ratio_l3044_304409


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3044_304426

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3044_304426


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3044_304464

/-- The linear equation 5x - y = 2 is satisfied by the point (1, 3) -/
theorem linear_equation_solution : 5 * 1 - 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3044_304464


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3044_304431

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x^2 + 2 * x + 2 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3044_304431


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l3044_304479

theorem rectangle_measurement_error (x : ℝ) : 
  (((1 + x / 100) * 0.9) = 1.08) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l3044_304479


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3044_304472

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) :
  initial + (initial * percentage) = initial * (1 + percentage) := by sorry

theorem increase_80_by_150_percent :
  80 + (80 * (150 / 100)) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3044_304472


namespace NUMINAMATH_CALUDE_world_population_scientific_notation_l3044_304423

/-- The world's population in billions -/
def world_population : ℝ := 8

/-- Scientific notation representation of a number -/
def scientific_notation (n : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  n = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

/-- Theorem: The world population of 8 billion in scientific notation is 8 × 10^9 -/
theorem world_population_scientific_notation :
  scientific_notation (world_population * 1000000000) 8 9 := by
  sorry

end NUMINAMATH_CALUDE_world_population_scientific_notation_l3044_304423


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3044_304414

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24)
  (h2 : 4 * (a + b + c) = 28) :
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3044_304414


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3044_304488

theorem quadratic_inequality_solution_range (t : ℝ) :
  (∃ c : ℝ, c ≤ 1 ∧ c^2 - 3*c + t ≤ 0) → t ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3044_304488


namespace NUMINAMATH_CALUDE_prime_factors_count_l3044_304442

theorem prime_factors_count (p q r : ℕ) (h1 : p = 4) (h2 : q = 7) (h3 : r = 11) 
  (h4 : p = 2^2) (h5 : Nat.Prime q) (h6 : Nat.Prime r) : 
  (Nat.factors (p^11 * q^7 * r^2)).length = 31 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_count_l3044_304442


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3044_304439

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, 
    the total number of games played is 480. -/
theorem chess_tournament_games : tournament_games 16 * 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3044_304439


namespace NUMINAMATH_CALUDE_range_of_a_l3044_304435

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 32) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = 32) →
  a ∈ Set.Icc 2 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3044_304435


namespace NUMINAMATH_CALUDE_existence_of_1000_consecutive_with_five_primes_l3044_304499

theorem existence_of_1000_consecutive_with_five_primes :
  (∃ n : ℕ, ∀ k ∈ Finset.range 1000, ¬ Nat.Prime (n + k + 2)) →
  (∃ m : ℕ, (Finset.filter (λ k => Nat.Prime (m + k + 1)) (Finset.range 1000)).card = 5) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_1000_consecutive_with_five_primes_l3044_304499


namespace NUMINAMATH_CALUDE_nine_step_paths_through_F_l3044_304476

/-- The number of paths from (0,4) to (3,3) on a grid, moving only right and down -/
def paths_E_to_F : ℕ := Nat.choose 4 1

/-- The number of paths from (3,3) to (5,0) on a grid, moving only right and down -/
def paths_F_to_G : ℕ := Nat.choose 5 2

/-- The total number of 9-step paths from E to G passing through F -/
def total_paths : ℕ := paths_E_to_F * paths_F_to_G

theorem nine_step_paths_through_F (h : total_paths = paths_E_to_F * paths_F_to_G) : 
  total_paths = 40 := by
  sorry

end NUMINAMATH_CALUDE_nine_step_paths_through_F_l3044_304476


namespace NUMINAMATH_CALUDE_smallest_possible_d_l3044_304417

theorem smallest_possible_d : ∃ d : ℝ, d = 3 ∧
  ∀ x : ℝ, (x ≥ 0 ∧ (4 * Real.sqrt 5) ^ 2 + (x + 5) ^ 2 = (4 * x) ^ 2) → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l3044_304417


namespace NUMINAMATH_CALUDE_square_difference_l3044_304447

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 := by sorry

end NUMINAMATH_CALUDE_square_difference_l3044_304447


namespace NUMINAMATH_CALUDE_cost_price_equals_selling_price_l3044_304489

/-- The number of articles whose selling price equals the cost price of 20 articles -/
def x : ℚ :=
  16

/-- The profit percentage -/
def profit_percentage : ℚ :=
  25 / 100

theorem cost_price_equals_selling_price (C : ℚ) (h : C > 0) :
  20 * C = x * C * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_equals_selling_price_l3044_304489


namespace NUMINAMATH_CALUDE_pictures_in_new_galleries_l3044_304456

/-- The number of pictures Alexander draws for the initial exhibition -/
def initial_pictures : ℕ := 9

/-- The number of new galleries -/
def new_galleries : ℕ := 7

/-- The number of pencils Alexander needs for each picture -/
def pencils_per_picture : ℕ := 5

/-- The number of pencils Alexander needs for signing at each exhibition -/
def pencils_for_signing : ℕ := 3

/-- The total number of pencils Alexander uses -/
def total_pencils : ℕ := 218

/-- The list of pictures requested by each new gallery -/
def new_gallery_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]

/-- Theorem: The number of pictures hung in the new galleries is 29 -/
theorem pictures_in_new_galleries : 
  (total_pencils - (pencils_for_signing * (new_galleries + 1))) / pencils_per_picture - initial_pictures = 29 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_new_galleries_l3044_304456


namespace NUMINAMATH_CALUDE_bowling_tournament_sequences_l3044_304458

/-- A tournament with 6 players and 5 matches -/
structure Tournament :=
  (num_players : Nat)
  (num_matches : Nat)
  (outcomes_per_match : Nat)

/-- The number of possible prize distribution sequences in the tournament -/
def prize_sequences (t : Tournament) : Nat :=
  t.outcomes_per_match ^ t.num_matches

/-- Theorem stating that for a tournament with 6 players, 5 matches, and 2 possible outcomes per match,
    the number of possible prize distribution sequences is 32 -/
theorem bowling_tournament_sequences :
  ∀ t : Tournament, t.num_players = 6 → t.num_matches = 5 → t.outcomes_per_match = 2 →
  prize_sequences t = 32 := by
  sorry

end NUMINAMATH_CALUDE_bowling_tournament_sequences_l3044_304458


namespace NUMINAMATH_CALUDE_max_acute_angles_non_convex_polygon_l3044_304449

theorem max_acute_angles_non_convex_polygon (n : ℕ) (h : n ≥ 3) :
  let sum_interior_angles := (n - 2) * 180
  let max_acute_angles := (2 * n) / 3 + 1
  ∃ k : ℕ, k ≤ max_acute_angles ∧
    k * 90 + (n - k) * 360 < sum_interior_angles ∧
    ∀ m : ℕ, m > k → m * 90 + (n - m) * 360 ≥ sum_interior_angles :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_non_convex_polygon_l3044_304449


namespace NUMINAMATH_CALUDE_expression_evaluation_l3044_304496

theorem expression_evaluation : 10 - 9 + 8 * 7^2 + 6 - 5 * 4 + 3 - 2 = 380 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3044_304496


namespace NUMINAMATH_CALUDE_divisor_problem_l3044_304484

theorem divisor_problem (n : ℤ) (d : ℤ) : 
  (∃ k : ℤ, n = 18 * k + 10) → 
  (∃ q : ℤ, 2 * n = d * q + 2) → 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3044_304484


namespace NUMINAMATH_CALUDE_work_completion_proof_l3044_304461

/-- The number of days it takes for person B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A and B work together -/
def work_together_days : ℝ := 7

/-- The fraction of work left after A and B work together for 7 days -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for person A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_proof :
  (work_together_days * (1 / a_days + 1 / b_days) = 1 - work_left) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3044_304461


namespace NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l3044_304402

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (lightSource : Point3D) (s₁ s₂ s₃ s₄ : Sphere),
    ∀ (r : Ray),
      r.origin = lightSource →
      rayIntersectsSphere r s₁ ∨
      rayIntersectsSphere r s₂ ∨
      rayIntersectsSphere r s₃ ∨
      rayIntersectsSphere r s₄ :=
sorry

end NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l3044_304402


namespace NUMINAMATH_CALUDE_perpendicular_projection_vector_l3044_304498

/-- Two-dimensional vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : Vec2
  dir : Vec2

def l : Line :=
  { point := { x := 2, y := 5 }
    dir := { x := 3, y := 2 } }

def m : Line :=
  { point := { x := 1, y := 3 }
    dir := { x := 2, y := 2 } }

def v : Vec2 :=
  { x := 1, y := -1 }

theorem perpendicular_projection_vector :
  (v.x * m.dir.x + v.y * m.dir.y = 0) ∧
  (2 * v.x - v.y = 3) := by sorry

end NUMINAMATH_CALUDE_perpendicular_projection_vector_l3044_304498


namespace NUMINAMATH_CALUDE_spheres_radius_l3044_304411

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The radius of each sphere -/
  radius : ℝ
  /-- The number of spheres is 8 -/
  num_spheres : Nat
  num_spheres_eq : num_spheres = 8
  /-- The cube is a unit cube -/
  cube_edge : ℝ
  cube_edge_eq : cube_edge = 1
  /-- Each sphere touches three adjacent spheres -/
  touches_adjacent : True
  /-- Spheres are inscribed in trihedral angles -/
  inscribed_in_angles : True

/-- The radius of spheres in the specific configuration is 1/4 -/
theorem spheres_radius (config : SpheresInCube) : config.radius = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_l3044_304411


namespace NUMINAMATH_CALUDE_max_product_sum_l3044_304440

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∃ (q : ℕ), q = A * M * C + A * M + M * C + C * A ∧
   ∀ (q' : ℕ), q' = A * M * C + A * M + M * C + C * A → q' ≤ q) ∧
  (A * M * C + A * M + M * C + C * A ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l3044_304440


namespace NUMINAMATH_CALUDE_max_carry_weight_is_1001_l3044_304429

/-- Represents the loader with a waggon and a cart -/
structure Loader :=
  (waggon_capacity : ℕ)
  (cart_capacity : ℕ)

/-- Represents the sand sacks in the storehouse -/
structure Storehouse :=
  (total_weight : ℕ)
  (max_sack_weight : ℕ)

/-- The maximum weight of sand the loader can carry -/
def max_carry_weight (l : Loader) (s : Storehouse) : ℕ :=
  l.waggon_capacity + l.cart_capacity

/-- Theorem stating the maximum weight the loader can carry -/
theorem max_carry_weight_is_1001 (l : Loader) (s : Storehouse) :
  l.waggon_capacity = 1000 →
  l.cart_capacity = 1 →
  s.total_weight > 1001 →
  s.max_sack_weight ≤ 1 →
  max_carry_weight l s = 1001 :=
by sorry

end NUMINAMATH_CALUDE_max_carry_weight_is_1001_l3044_304429


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l3044_304494

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + a*r₁ + b = 0 → 
  r₂^2 + a*r₂ + b = 0 → 
  ∃ t : ℝ, (r₁^2 + 2*r₁*r₂ + r₂^2)^2 + (ab - a^2)*(r₁^2 + 2*r₁*r₂ + r₂^2) + t = 0 ∧ 
           (r₁*r₂*(r₁ + r₂))^2 + (ab - a^2)*(r₁*r₂*(r₁ + r₂)) + t = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l3044_304494


namespace NUMINAMATH_CALUDE_triangle_area_l3044_304480

theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t + 1
  (1 / 2) * base * height = 3 * t^2 + t :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3044_304480


namespace NUMINAMATH_CALUDE_salary_decrease_theorem_l3044_304460

/-- Represents the decrease in average salary of all employees per day -/
def salary_decrease (illiterate_count : ℕ) (literate_count : ℕ) (old_wage : ℕ) (new_wage : ℕ) : ℚ :=
  let total_employees := illiterate_count + literate_count
  let wage_decrease := old_wage - new_wage
  let total_decrease := illiterate_count * wage_decrease
  (total_decrease : ℚ) / total_employees

/-- Theorem stating the decrease in average salary of all employees per day -/
theorem salary_decrease_theorem :
  salary_decrease 20 10 25 10 = 10 := by sorry

end NUMINAMATH_CALUDE_salary_decrease_theorem_l3044_304460


namespace NUMINAMATH_CALUDE_line_chart_most_appropriate_l3044_304452

/-- Represents a chart type -/
inductive ChartType
| LineChart
| BarChart
| PieChart
| ScatterPlot

/-- Represents the requirements for a temperature chart -/
structure TemperatureChartRequirements where
  showsChangeOverTime : Bool
  reflectsAmountAndChanges : Bool
  showsIncreasesAndDecreases : Bool

/-- Defines the properties of a line chart -/
def lineChartProperties : TemperatureChartRequirements :=
  { showsChangeOverTime := true
  , reflectsAmountAndChanges := true
  , showsIncreasesAndDecreases := true }

/-- Determines if a chart type is appropriate for the given requirements -/
def isAppropriateChart (c : ChartType) (r : TemperatureChartRequirements) : Bool :=
  match c with
  | ChartType.LineChart => r.showsChangeOverTime ∧ r.reflectsAmountAndChanges ∧ r.showsIncreasesAndDecreases
  | _ => false

/-- Theorem: A line chart is the most appropriate for recording temperature changes of a feverish patient -/
theorem line_chart_most_appropriate :
  isAppropriateChart ChartType.LineChart lineChartProperties = true :=
sorry

end NUMINAMATH_CALUDE_line_chart_most_appropriate_l3044_304452


namespace NUMINAMATH_CALUDE_sphere_circular_cross_section_l3044_304450

-- Define the types of solids
inductive Solid
| Cylinder
| Cone
| Sphere
| Frustum

-- Define the types of cross-section shapes
inductive CrossSectionShape
| Rectangular
| Triangular
| Circular
| IsoscelesTrapezoid

-- Function to get the cross-section shape of a solid through its axis of rotation
def crossSectionThroughAxis (s : Solid) : CrossSectionShape :=
  match s with
  | Solid.Cylinder => CrossSectionShape.Rectangular
  | Solid.Cone => CrossSectionShape.Triangular
  | Solid.Sphere => CrossSectionShape.Circular
  | Solid.Frustum => CrossSectionShape.IsoscelesTrapezoid

-- Theorem stating that only the Sphere has a circular cross-section through its axis of rotation
theorem sphere_circular_cross_section :
  ∀ s : Solid, crossSectionThroughAxis s = CrossSectionShape.Circular ↔ s = Solid.Sphere :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_circular_cross_section_l3044_304450


namespace NUMINAMATH_CALUDE_f_extrema_l3044_304403

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l3044_304403


namespace NUMINAMATH_CALUDE_max_phi_difference_l3044_304436

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem max_phi_difference (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  (phi (n^2 + 2*n) - phi (n^2) ≤ 72) ∧
  (∃ m : ℕ, 1 ≤ m ∧ m ≤ 100 ∧ phi (m^2 + 2*m) - phi (m^2) = 72) :=
sorry

end NUMINAMATH_CALUDE_max_phi_difference_l3044_304436


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3044_304459

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3044_304459


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3044_304463

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (bounce_ratio : ℝ) (num_bounces : ℕ) : ℝ :=
  let descent_distances := (Finset.range (num_bounces + 1)).sum (fun n => initial_height * bounce_ratio^n)
  let ascent_distances := (Finset.range num_bounces).sum (fun n => initial_height * bounce_ratio^(n+1))
  descent_distances + ascent_distances

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  total_distance 20 (2/3) 4 = 6500/81 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3044_304463


namespace NUMINAMATH_CALUDE_fraction_multiplication_simplification_l3044_304406

theorem fraction_multiplication_simplification :
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_simplification_l3044_304406


namespace NUMINAMATH_CALUDE_start_with_any_digits_l3044_304405

theorem start_with_any_digits :
  ∀ (A : ℕ), ∃ (n m : ℕ), 10^m * A ≤ 2^n ∧ 2^n < 10^m * (A + 1) :=
sorry

end NUMINAMATH_CALUDE_start_with_any_digits_l3044_304405


namespace NUMINAMATH_CALUDE_ellipse_quadrant_area_diff_zero_l3044_304432

/-- Definition of an ellipse with center (h, k) and parameters a, b, c -/
def Ellipse (h k a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - h)^2 / a + (p.2 - k)^2 / b = c}

/-- Areas of the ellipse in each quadrant -/
def QuadrantAreas (e : Set (ℝ × ℝ)) : ℝ × ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The difference of areas in alternating quadrants is zero -/
theorem ellipse_quadrant_area_diff_zero
  (h k a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let e := Ellipse h k a b c
  let (R1, R2, R3, R4) := QuadrantAreas e
  R1 - R2 + R3 - R4 = 0 := by sorry


end NUMINAMATH_CALUDE_ellipse_quadrant_area_diff_zero_l3044_304432


namespace NUMINAMATH_CALUDE_acute_angles_sum_l3044_304473

theorem acute_angles_sum (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_condition : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l3044_304473


namespace NUMINAMATH_CALUDE_power_multiplication_l3044_304434

theorem power_multiplication (x : ℝ) : x^6 * x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3044_304434


namespace NUMINAMATH_CALUDE_ant_count_approximation_l3044_304495

/-- Calculates the approximate number of ants in a rectangular field -/
def approximate_ant_count (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  let width_inches := width_feet * 12
  let length_inches := length_feet * 12
  let area_sq_inches := width_inches * length_inches
  area_sq_inches * ants_per_sq_inch

/-- Theorem stating that the number of ants in the given field is approximately 59 million -/
theorem ant_count_approximation :
  let field_width := 250
  let field_length := 330
  let ants_density := 5
  let calculated_count := approximate_ant_count field_width field_length ants_density
  abs (calculated_count - 59000000) / 59000000 < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ant_count_approximation_l3044_304495


namespace NUMINAMATH_CALUDE_total_shingles_needed_l3044_304478

/-- The number of shingles needed to cover a given area of roof --/
def shingles_per_square_foot : ℕ := 8

/-- The number of roofs to be shingled --/
def number_of_roofs : ℕ := 3

/-- The length of each rectangular side of a roof in feet --/
def roof_side_length : ℕ := 40

/-- The width of each rectangular side of a roof in feet --/
def roof_side_width : ℕ := 20

/-- The number of rectangular sides per roof --/
def sides_per_roof : ℕ := 2

/-- Theorem stating the total number of shingles needed --/
theorem total_shingles_needed :
  (number_of_roofs * sides_per_roof * roof_side_length * roof_side_width * shingles_per_square_foot) = 38400 := by
  sorry

end NUMINAMATH_CALUDE_total_shingles_needed_l3044_304478


namespace NUMINAMATH_CALUDE_altered_prism_edges_l3044_304420

/-- Represents a rectangular prism that has been altered by truncating vertices and cutting faces diagonally -/
structure AlteredPrism where
  initialEdges : Nat
  initialVertices : Nat
  initialFaces : Nat
  newEdgesPerTruncatedVertex : Nat
  newEdgesPerCutFace : Nat

/-- Calculates the total number of edges in the altered prism -/
def totalEdges (p : AlteredPrism) : Nat :=
  p.initialEdges + 
  (p.initialVertices * p.newEdgesPerTruncatedVertex) + 
  (p.initialFaces * p.newEdgesPerCutFace)

/-- Theorem stating that the altered rectangular prism has 42 edges -/
theorem altered_prism_edges :
  ∀ (p : AlteredPrism),
    p.initialEdges = 12 →
    p.initialVertices = 8 →
    p.initialFaces = 6 →
    p.newEdgesPerTruncatedVertex = 3 →
    p.newEdgesPerCutFace = 1 →
    totalEdges p = 42 := by
  sorry

#check altered_prism_edges

end NUMINAMATH_CALUDE_altered_prism_edges_l3044_304420


namespace NUMINAMATH_CALUDE_students_doing_homework_l3044_304477

theorem students_doing_homework (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) 
  (h1 : total = 60)
  (h2 : silent_reading = 3/8)
  (h3 : board_games = 1/4) :
  total - (Int.floor (silent_reading * total) + Int.floor (board_games * total)) = 22 :=
by sorry

end NUMINAMATH_CALUDE_students_doing_homework_l3044_304477


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3044_304490

/-- A quadratic function -/
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ p q : ℝ, p ≠ q → f a b c p = f a b c q → f a b c (p + q) = c) ∧
  (∀ p q : ℝ, p ≠ q → f a b c (p + q) = c → (p + q = 0 ∨ f a b c p = f a b c q)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3044_304490


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l3044_304437

theorem min_distance_to_circle (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∃ (min : ℝ), min = Real.sqrt 5 - 1 ∧ 
    ∀ (u v : ℝ), u^2 + v^2 = 1 → 
      Real.sqrt ((u - 1)^2 + (v - 2)^2) ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l3044_304437


namespace NUMINAMATH_CALUDE_extreme_point_and_tangent_lines_l3044_304430

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - a^2*x

-- State the theorem
theorem extreme_point_and_tangent_lines :
  -- Given conditions
  ∃ (a : ℝ), (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), h ≠ 0 → (f a (x + h) - f a x) / h > 0 ∨ (f a (x + h) - f a x) / h < 0)) →
  -- Conclusions
  (∃ (x : ℝ), f a x = -5 ∧ ∀ (y : ℝ), f a y ≥ -5) ∧
  (f 1 0 = 0 ∧ ∃ (m₁ m₂ : ℝ), m₁ = -1 ∧ m₂ = -5/4 ∧
    ∀ (x : ℝ), (f 1 x = m₁ * x ∨ f 1 x = m₂ * x) → 
      ∀ (y : ℝ), y = m₁ * x ∨ y = m₂ * x → f 1 y = y) :=
by sorry

end NUMINAMATH_CALUDE_extreme_point_and_tangent_lines_l3044_304430


namespace NUMINAMATH_CALUDE_prime_pairs_square_sum_l3044_304424

theorem prime_pairs_square_sum (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) ↔ ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_square_sum_l3044_304424


namespace NUMINAMATH_CALUDE_roots_of_Q_are_fifth_powers_of_roots_of_P_l3044_304497

-- Define the polynomial P
def P (x : ℂ) : ℂ := x^3 - 3*x + 1

-- Define the polynomial Q
def Q (y : ℂ) : ℂ := y^3 + 15*y^2 - 198*y + 1

-- Theorem statement
theorem roots_of_Q_are_fifth_powers_of_roots_of_P :
  ∀ (α : ℂ), P α = 0 → ∃ (β : ℂ), Q (β^5) = 0 ∧ P β = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_Q_are_fifth_powers_of_roots_of_P_l3044_304497


namespace NUMINAMATH_CALUDE_polynomial_root_product_l3044_304487

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b*r - c = 0) → b*c = 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l3044_304487


namespace NUMINAMATH_CALUDE_initial_water_percentage_in_milk_l3044_304492

/-- The initial percentage of water in milk, given that adding 15 liters of pure milk to 10 liters
    of the initial milk reduces the water content to 2%. -/
theorem initial_water_percentage_in_milk :
  ∀ (initial_water_percentage : ℝ),
    (initial_water_percentage ≥ 0) →
    (initial_water_percentage ≤ 100) →
    (10 * (100 - initial_water_percentage) / 100 + 15 = 0.98 * 25) →
    initial_water_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_in_milk_l3044_304492


namespace NUMINAMATH_CALUDE_second_part_interest_rate_l3044_304470

def total_amount : ℝ := 2500
def first_part : ℝ := 500
def first_rate : ℝ := 0.05
def total_income : ℝ := 145

theorem second_part_interest_rate :
  let second_part := total_amount - first_part
  let first_income := first_part * first_rate
  let second_income := total_income - first_income
  let second_rate := second_income / second_part
  second_rate = 0.06 := by sorry

end NUMINAMATH_CALUDE_second_part_interest_rate_l3044_304470


namespace NUMINAMATH_CALUDE_expression_value_l3044_304474

theorem expression_value (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / |x| + |y| / y = 2 ∨ x / |x| + |y| / y = 0 ∨ x / |x| + |y| / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3044_304474


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3044_304454

theorem polynomial_factorization (a : ℝ) : a^2 + 2*a = a*(a+2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3044_304454


namespace NUMINAMATH_CALUDE_expression_eval_at_two_l3044_304455

theorem expression_eval_at_two : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 2
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_at_two_l3044_304455


namespace NUMINAMATH_CALUDE_max_pairs_remaining_l3044_304445

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 27

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- Theorem stating the maximum number of complete pairs remaining after losing shoes -/
theorem max_pairs_remaining (total : ℕ) (lost : ℕ) : 
  total = total_pairs → lost = shoes_lost → total - lost ≤ 18 := by
  sorry

#check max_pairs_remaining

end NUMINAMATH_CALUDE_max_pairs_remaining_l3044_304445


namespace NUMINAMATH_CALUDE_f_abs_x_is_even_l3044_304441

theorem f_abs_x_is_even (f : ℝ → ℝ) : 
  let g := fun (x : ℝ) ↦ f (|x|)
  ∀ x, g (-x) = g x := by sorry

end NUMINAMATH_CALUDE_f_abs_x_is_even_l3044_304441


namespace NUMINAMATH_CALUDE_exists_simultaneous_j_half_no_universal_j_half_l3044_304401

/-- A number is a j-half if it leaves a remainder of j when divided by 2j+1 -/
def is_j_half (n j : ℕ) : Prop := n % (2 * j + 1) = j

/-- For any positive integer k, there exists a number that is simultaneously a j-half for j = 1, 2, ..., k -/
theorem exists_simultaneous_j_half (k : ℕ) : ∃ n : ℕ, ∀ j ∈ Finset.range k, is_j_half n j := by sorry

/-- There is no number which is a j-half for all positive integers j -/
theorem no_universal_j_half : ¬∃ n : ℕ, ∀ j : ℕ, j > 0 → is_j_half n j := by sorry

end NUMINAMATH_CALUDE_exists_simultaneous_j_half_no_universal_j_half_l3044_304401
