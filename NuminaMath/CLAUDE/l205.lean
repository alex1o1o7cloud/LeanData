import Mathlib

namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l205_20527

/-- Given a rectangular pen with a perimeter of 60 feet, 
    the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen : 
  ∀ x y : ℝ, 
    x > 0 → y > 0 → 
    2 * x + 2 * y = 60 → 
    x * y ≤ 225 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l205_20527


namespace NUMINAMATH_CALUDE_volume_ratio_is_three_to_five_l205_20534

/-- A square-based right pyramid where side faces form a 60° angle with the base -/
structure SquarePyramid where
  base_side : ℝ
  side_face_angle : ℝ
  side_face_angle_eq : side_face_angle = π / 3

/-- The ratio of volumes created by the bisector plane -/
def volume_ratio (p : SquarePyramid) : ℝ × ℝ := sorry

/-- Theorem: The ratio of volumes is 3:5 -/
theorem volume_ratio_is_three_to_five (p : SquarePyramid) : 
  volume_ratio p = (3, 5) := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_three_to_five_l205_20534


namespace NUMINAMATH_CALUDE_bob_mary_sheep_ratio_l205_20514

/-- The number of sheep Mary has initially -/
def mary_initial_sheep : ℕ := 300

/-- The number of sheep Mary buys -/
def mary_bought_sheep : ℕ := 266

/-- The difference between Bob's sheep and Mary's sheep after Mary's purchase -/
def sheep_difference : ℕ := 69

/-- Bob's sheep count -/
def bob_sheep : ℕ := mary_initial_sheep + mary_bought_sheep + sheep_difference

theorem bob_mary_sheep_ratio : 
  (bob_sheep : ℚ) / mary_initial_sheep = 635 / 300 := by sorry

end NUMINAMATH_CALUDE_bob_mary_sheep_ratio_l205_20514


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l205_20529

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l205_20529


namespace NUMINAMATH_CALUDE_complement_of_M_l205_20550

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l205_20550


namespace NUMINAMATH_CALUDE_min_value_of_expression_l205_20560

theorem min_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hsum : x + y = 3) :
  (1 / (x - 1) + 3 / (y - 1)) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l205_20560


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_l205_20504

theorem infinite_perfect_squares (k : ℕ+) :
  ∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ n ∈ S, ∃ m : ℕ+, (n * 2^k.val : ℤ) - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_l205_20504


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l205_20571

theorem sum_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 220) (h2 : x * y = 52) : x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l205_20571


namespace NUMINAMATH_CALUDE_real_part_of_squared_reciprocal_l205_20568

theorem real_part_of_squared_reciprocal (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) (h3 : z.re = x) :
  Complex.re ((1 / (2 - z)) ^ 2) = x / (4 * (4 - 4*x + x^2)) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_squared_reciprocal_l205_20568


namespace NUMINAMATH_CALUDE_area_of_six_square_figure_l205_20580

/-- A figure consisting of 6 identical squares with a total perimeter of 84 cm has an area of 216 cm². -/
theorem area_of_six_square_figure (perimeter : ℝ) (num_squares : ℕ) :
  perimeter = 84 →
  num_squares = 6 →
  (perimeter / 14) ^ 2 * num_squares = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_of_six_square_figure_l205_20580


namespace NUMINAMATH_CALUDE_smallest_c_value_l205_20506

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l205_20506


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l205_20565

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) 
  (h₁ : a₁ = 1/2) 
  (h₂ : a₂ = 5/6) 
  (h₃ : a₃ = 7/6) 
  (h₄ : arithmetic_sequence a₁ a₂ a₃ 3 = a₃) :
  arithmetic_sequence a₁ a₂ a₃ 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l205_20565


namespace NUMINAMATH_CALUDE_triangle_area_rational_l205_20564

/-- Represents a point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Condition that the absolute difference between coordinates is at most 2 -/
def coordDiffAtMostTwo (p q : IntPoint) : Prop :=
  (abs (p.x - q.x) ≤ 2) ∧ (abs (p.y - q.y) ≤ 2)

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem stating that the area of a triangle with integer coordinates and limited coordinate differences is rational -/
theorem triangle_area_rational (p1 p2 p3 : IntPoint) 
  (h12 : coordDiffAtMostTwo p1 p2)
  (h23 : coordDiffAtMostTwo p2 p3)
  (h31 : coordDiffAtMostTwo p3 p1) :
  ∃ (q : ℚ), triangleArea p1 p2 p3 = q := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_rational_l205_20564


namespace NUMINAMATH_CALUDE_inequality_proof_l205_20542

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) 
  (h5 : a + b + c + d = 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l205_20542


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l205_20597

/-- Represents the financial state of a person --/
structure FinancialState where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings --/
def calculateExpenditure (fs : FinancialState) : ℕ :=
  fs.income - fs.savings

/-- Calculates the ratio of two numbers --/
def calculateRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- Theorem stating the ratio of income to expenditure --/
theorem income_expenditure_ratio (fs : FinancialState) 
  (h1 : fs.income = 18000) 
  (h2 : fs.savings = 2000) : 
  calculateRatio fs.income (calculateExpenditure fs) = (9, 8) := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l205_20597


namespace NUMINAMATH_CALUDE_abs_greater_than_two_solution_set_l205_20526

theorem abs_greater_than_two_solution_set :
  {x : ℝ | |x| > 2} = {x : ℝ | x > 2 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_abs_greater_than_two_solution_set_l205_20526


namespace NUMINAMATH_CALUDE_sin_cos_identity_l205_20505

theorem sin_cos_identity (x : ℝ) : 
  Real.sin (3 * x - Real.pi / 4) = Real.cos (3 * x - 3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l205_20505


namespace NUMINAMATH_CALUDE_bats_against_left_handed_correct_l205_20562

/-- Represents a baseball player's batting statistics -/
structure BattingStats where
  total_bats : ℕ
  total_hits : ℕ
  left_handed_avg : ℚ
  right_handed_avg : ℚ

/-- Calculates the number of bats against left-handed pitchers -/
def bats_against_left_handed (stats : BattingStats) : ℕ :=
  sorry

/-- Theorem stating the correct number of bats against left-handed pitchers -/
theorem bats_against_left_handed_correct (stats : BattingStats) 
  (h1 : stats.total_bats = 600)
  (h2 : stats.total_hits = 192)
  (h3 : stats.left_handed_avg = 1/4)
  (h4 : stats.right_handed_avg = 7/20)
  (h5 : (stats.total_hits : ℚ) / stats.total_bats = 8/25) :
  bats_against_left_handed stats = 180 :=
sorry

end NUMINAMATH_CALUDE_bats_against_left_handed_correct_l205_20562


namespace NUMINAMATH_CALUDE_triangle_problem_l205_20544

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a + b = 6 →
  c = 2 →
  Real.cos C = 7/9 →
  a = 3 ∧ b = 3 ∧ (1/2 * a * b * Real.sin C = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l205_20544


namespace NUMINAMATH_CALUDE_power_sum_simplification_l205_20532

theorem power_sum_simplification :
  -2^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 := by sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l205_20532


namespace NUMINAMATH_CALUDE_one_fifth_of_five_times_seven_l205_20531

theorem one_fifth_of_five_times_seven :
  (1 / 5 : ℚ) * (5 * 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_five_times_seven_l205_20531


namespace NUMINAMATH_CALUDE_P_is_circle_l205_20538

-- Define the set of points P(x, y) satisfying the given equation
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 10 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 - 4 * p.2|}

-- Theorem stating that the set P forms a circle
theorem P_is_circle : ∃ (c : ℝ × ℝ) (r : ℝ), P = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2} :=
sorry

end NUMINAMATH_CALUDE_P_is_circle_l205_20538


namespace NUMINAMATH_CALUDE_program_output_correct_l205_20576

def program_transformation (a₀ b₀ c₀ : ℕ) : ℕ × ℕ × ℕ :=
  let a₁ := b₀
  let b₁ := c₀
  let c₁ := a₁
  (a₁, b₁, c₁)

theorem program_output_correct :
  program_transformation 2 3 4 = (3, 4, 3) := by sorry

end NUMINAMATH_CALUDE_program_output_correct_l205_20576


namespace NUMINAMATH_CALUDE_three_zeros_iff_a_in_open_interval_l205_20525

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The property of having exactly three distinct real zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- Theorem stating the equivalence between the function having three distinct zeros
    and the parameter a being in the open interval (-2, 2) -/
theorem three_zeros_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_distinct_zeros a ↔ -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_iff_a_in_open_interval_l205_20525


namespace NUMINAMATH_CALUDE_closed_set_properties_l205_20586

-- Definition of a closed set
def is_closed_set (M : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Set M = {-2, -1, 0, 1, 2}
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Set of positive integers
def positive_integers : Set ℤ := {n : ℤ | n > 0}

-- Set of multiples of 3
def multiples_of_three : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  ¬(is_closed_set M) ∧
  ¬(is_closed_set positive_integers) ∧
  (is_closed_set multiples_of_three) ∧
  (∃ A₁ A₂ : Set ℤ, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬(is_closed_set (A₁ ∪ A₂))) :=
by sorry

end NUMINAMATH_CALUDE_closed_set_properties_l205_20586


namespace NUMINAMATH_CALUDE_cos_seven_theta_l205_20573

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (7 * θ) = -37/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_theta_l205_20573


namespace NUMINAMATH_CALUDE_five_triangles_common_vertex_l205_20507

/-- The angle between two adjacent equilateral triangles when five such triangles
    meet at a common vertex with congruent angles between them. -/
def angle_between_triangles : ℝ := 12

theorem five_triangles_common_vertex :
  let n : ℕ := 5 -- number of triangles
  let triangle_angle : ℝ := 60 -- internal angle of an equilateral triangle
  let total_angle : ℝ := 360 -- total angle around a point
  angle_between_triangles = (total_angle - n * triangle_angle) / n := by
  sorry

#check five_triangles_common_vertex

end NUMINAMATH_CALUDE_five_triangles_common_vertex_l205_20507


namespace NUMINAMATH_CALUDE_rohans_age_l205_20594

theorem rohans_age :
  ∀ R : ℕ, (R + 15 = 4 * (R - 15)) → R = 25 := by
  sorry

end NUMINAMATH_CALUDE_rohans_age_l205_20594


namespace NUMINAMATH_CALUDE_quarters_undetermined_l205_20540

/-- Represents Mike's coin collection --/
structure CoinCollection where
  quarters : ℕ
  nickels : ℕ

/-- Represents the borrowing transaction --/
def borrow (c : CoinCollection) (borrowed : ℕ) : CoinCollection :=
  { quarters := c.quarters, nickels := c.nickels - borrowed }

theorem quarters_undetermined (initial_nickels borrowed remaining_nickels : ℕ) 
  (h1 : initial_nickels = 87)
  (h2 : borrowed = 75)
  (h3 : remaining_nickels = 12)
  (h4 : initial_nickels = borrowed + remaining_nickels) :
  ∀ q1 q2 : ℕ, ∃ c1 c2 : CoinCollection,
    c1.nickels = initial_nickels ∧
    c2.nickels = initial_nickels ∧
    c1.quarters = q1 ∧
    c2.quarters = q2 ∧
    (borrow c1 borrowed).nickels = remaining_nickels ∧
    (borrow c2 borrowed).nickels = remaining_nickels :=
sorry

end NUMINAMATH_CALUDE_quarters_undetermined_l205_20540


namespace NUMINAMATH_CALUDE_A_D_independent_l205_20596

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω : Ω | ω.1 = 0}
def D : Set Ω := {ω : Ω | ω.1.val + ω.2.val = 6}

-- State the theorem
theorem A_D_independent : P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_D_independent_l205_20596


namespace NUMINAMATH_CALUDE_jerry_action_figures_l205_20546

theorem jerry_action_figures (initial_books initial_figures added_figures : ℕ) :
  initial_books = 3 →
  initial_figures = 4 →
  initial_books + 3 = initial_figures + added_figures →
  added_figures = 2 := by
sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l205_20546


namespace NUMINAMATH_CALUDE_intersection_problem_l205_20572

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : a * x + b * y + c = 0

/-- Checks if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬ (l1.a * l2.b = l1.b * l2.a)

/-- The main line x + y - 1 = 0 -/
def main_line : Line :=
  { a := 1, b := 1, c := -1, equation := sorry }

/-- Line A: 2x + 2y = 6 -/
def line_a : Line :=
  { a := 2, b := 2, c := -6, equation := sorry }

/-- Line B: x + y = 0 -/
def line_b : Line :=
  { a := 1, b := 1, c := 0, equation := sorry }

/-- Line C: y = -x - 3 -/
def line_c : Line :=
  { a := 1, b := 1, c := 3, equation := sorry }

/-- Line D: y = x - 1 -/
def line_d : Line :=
  { a := 1, b := -1, c := 1, equation := sorry }

theorem intersection_problem :
  intersect main_line line_d ∧
  ¬ intersect main_line line_a ∧
  ¬ intersect main_line line_b ∧
  ¬ intersect main_line line_c :=
sorry

end NUMINAMATH_CALUDE_intersection_problem_l205_20572


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l205_20559

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l205_20559


namespace NUMINAMATH_CALUDE_polynomial_with_negative_roots_l205_20593

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The sum of coefficients of the polynomial -/
def Polynomial4.sum (p : Polynomial4) : ℤ :=
  p.a + p.b + p.c + p.d

/-- Predicate to check if all roots of the polynomial are negative integers -/
def has_all_negative_integer_roots (p : Polynomial4) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ : ℕ), 
    s₁ > 0 ∧ s₂ > 0 ∧ s₃ > 0 ∧ s₄ > 0 ∧
    p.a = s₁ + s₂ + s₃ + s₄ ∧
    p.b = s₁*s₂ + s₁*s₃ + s₁*s₄ + s₂*s₃ + s₂*s₄ + s₃*s₄ ∧
    p.c = s₁*s₂*s₃ + s₁*s₂*s₄ + s₁*s₃*s₄ + s₂*s₃*s₄ ∧
    p.d = s₁*s₂*s₃*s₄

theorem polynomial_with_negative_roots (p : Polynomial4) 
  (h1 : has_all_negative_integer_roots p) 
  (h2 : p.sum = 2003) : 
  p.d = 1992 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_with_negative_roots_l205_20593


namespace NUMINAMATH_CALUDE_minimum_opinion_change_l205_20549

/-- Represents the number of students who like or dislike math at a given time --/
structure MathOpinion where
  like : Nat
  dislike : Nat

/-- Represents the change in students' opinions about math --/
structure OpinionChange where
  dislike_to_like : Nat
  like_to_dislike : Nat

theorem minimum_opinion_change (initial final : MathOpinion) (change : OpinionChange)
    (h1 : initial.like + initial.dislike = 40)
    (h2 : final.like + final.dislike = 40)
    (h3 : initial.like = 18)
    (h4 : initial.dislike = 22)
    (h5 : final.like = 28)
    (h6 : final.dislike = 12)
    (h7 : change.dislike_to_like = 10)
    (h8 : final.like = initial.like + change.dislike_to_like - change.like_to_dislike) :
    change.dislike_to_like + change.like_to_dislike = 10 := by
  sorry

#check minimum_opinion_change

end NUMINAMATH_CALUDE_minimum_opinion_change_l205_20549


namespace NUMINAMATH_CALUDE_abs_neg_2023_l205_20539

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l205_20539


namespace NUMINAMATH_CALUDE_only_first_option_exact_l205_20530

/-- Represents a measurement with a value and whether it's approximate or exact -/
structure Measurement where
  value : ℝ
  isApproximate : Bool

/-- The four options given in the problem -/
def options : List Measurement := [
  ⟨1752, false⟩,  -- A: Dictionary pages
  ⟨150, true⟩,    -- B: Water in teacup
  ⟨13.5, true⟩,   -- C: Running time
  ⟨6.2, true⟩     -- D: World population
]

/-- Theorem stating that only the first option is not an approximate number -/
theorem only_first_option_exact : 
  ∃! i : Fin 4, (options.get i).isApproximate = false := by
  sorry

end NUMINAMATH_CALUDE_only_first_option_exact_l205_20530


namespace NUMINAMATH_CALUDE_square_area_l205_20591

-- Define the square
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the square
def is_valid_square (s : Square) : Prop :=
  let (x₀, _) := s.A
  let (_, y₁) := s.B
  let (x₂, y₂) := s.C
  let (_, y₃) := s.D
  x₀ = x₂ ∧                   -- A and C on same vertical line
  y₁ = 2 ∧ y₂ = 8 ∧ y₃ = 6 ∧  -- y-coordinates are 0, 2, 6, 8
  s.A.2 = 0 ∧ s.C.2 = 8       -- A has y-coordinate 0, C has y-coordinate 8

-- Theorem statement
theorem square_area (s : Square) (h : is_valid_square s) : 
  (s.C.2 - s.A.2) * (s.C.2 - s.A.2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l205_20591


namespace NUMINAMATH_CALUDE_decimal_representation_nonzero_digits_l205_20500

theorem decimal_representation_nonzero_digits :
  let x : ℚ := 720 / (2^6 * 3^5)
  ∃ (a b c d : ℕ) (r : ℚ),
    0 < a ∧ a < 10 ∧
    0 < b ∧ b < 10 ∧
    0 < c ∧ c < 10 ∧
    0 < d ∧ d < 10 ∧
    x = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (d : ℚ) / 10000 + r ∧
    0 ≤ r ∧ r < 1/10000 :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_nonzero_digits_l205_20500


namespace NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l205_20587

/-- Given a cylinder with height 1 and bases on a sphere of diameter 2, its volume is 3π/4 -/
theorem cylinder_volume_on_sphere (h : ℝ) (d : ℝ) (V : ℝ) : 
  h = 1 → d = 2 → V = (3 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_on_sphere_l205_20587


namespace NUMINAMATH_CALUDE_sales_solution_l205_20524

def sales_problem (s1 s2 s3 s5 s6 average : ℕ) : Prop :=
  let total := average * 6
  let known_sum := s1 + s2 + s3 + s5 + s6
  total - known_sum = 6122

theorem sales_solution :
  sales_problem 5266 5744 5864 6588 4916 5750 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l205_20524


namespace NUMINAMATH_CALUDE_trade_calculation_l205_20547

/-- The number of matches per stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of stamps Tonya starts with -/
def tonya_initial_stamps : ℕ := 13

/-- The number of stamps Tonya ends with -/
def tonya_final_stamps : ℕ := 3

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

theorem trade_calculation :
  (tonya_initial_stamps - tonya_final_stamps) * matches_per_stamp = jimmy_matchbooks * matches_per_matchbook :=
by sorry

end NUMINAMATH_CALUDE_trade_calculation_l205_20547


namespace NUMINAMATH_CALUDE_circle_fixed_point_l205_20551

theorem circle_fixed_point (a : ℝ) (ha : a ≠ 1) :
  ∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0 → x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_fixed_point_l205_20551


namespace NUMINAMATH_CALUDE_graph_tangency_l205_20575

noncomputable def tangentPoints (a : ℝ) : Prop :=
  (a > 0 ∧ a ≠ 1) →
  ∃ x > 0, (a^x = x ∧ (a^x * Real.log a = 1 ∨ a^x * Real.log a = -1))

theorem graph_tangency :
  ∀ a : ℝ, tangentPoints a ↔ (a = Real.exp (1 / Real.exp 1) ∨ a = Real.exp (-1 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_graph_tangency_l205_20575


namespace NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l205_20570

/-- Given a > 0 and f(x) = x³ - ax is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotone_cubic (a : ℝ) : 
  a > 0 → 
  (∀ x y, 1 ≤ x → x ≤ y → x^3 - a*x ≤ y^3 - a*y) → 
  a ≤ 3 ∧ ∀ b, (b > 0 ∧ (∀ x y, 1 ≤ x → x ≤ y → x^3 - b*x ≤ y^3 - b*y)) → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l205_20570


namespace NUMINAMATH_CALUDE_multiply_powers_of_ten_l205_20569

theorem multiply_powers_of_ten : (-2 * 10^4) * (4 * 10^5) = -8 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_ten_l205_20569


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_problem_l205_20567

theorem jenny_easter_eggs_problem :
  let total_red : ℕ := 30
  let total_blue : ℕ := 42
  let min_eggs_per_basket : ℕ := 5
  ∃ (eggs_per_basket : ℕ),
    eggs_per_basket ≥ min_eggs_per_basket ∧
    eggs_per_basket ∣ total_red ∧
    eggs_per_basket ∣ total_blue ∧
    ∀ (n : ℕ), n > eggs_per_basket →
      ¬(n ∣ total_red ∧ n ∣ total_blue) →
    eggs_per_basket = 6 :=
by sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_problem_l205_20567


namespace NUMINAMATH_CALUDE_protest_duration_increase_l205_20511

/-- Given two protests with a total duration of 9 days, where the first protest lasts 4 days,
    the percentage increase in duration from the first to the second protest is 25%. -/
theorem protest_duration_increase (d₁ d₂ : ℝ) : 
  d₁ = 4 → d₁ + d₂ = 9 → (d₂ - d₁) / d₁ * 100 = 25 := by
  sorry

#check protest_duration_increase

end NUMINAMATH_CALUDE_protest_duration_increase_l205_20511


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_8191_l205_20543

/-- The greatest prime divisor of a natural number n -/
def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 1

/-- The sum of digits of a natural number n -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

/-- Theorem stating that the sum of the digits of the greatest prime divisor of 8191 is 10 -/
theorem sum_digits_greatest_prime_divisor_8191 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_8191_l205_20543


namespace NUMINAMATH_CALUDE_geometry_theorem_l205_20512

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- line is subset of plane
variable (parallel : Line → Line → Prop)  -- lines are parallel
variable (parallel_lp : Line → Plane → Prop)  -- line is parallel to plane
variable (parallel_pp : Plane → Plane → Prop)  -- planes are parallel
variable (perpendicular : Line → Line → Prop)  -- lines are perpendicular
variable (perpendicular_lp : Line → Plane → Prop)  -- line is perpendicular to plane

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_lp m α ∧ parallel_lp n α → perpendicular m n) ∧ 
  (perpendicular_lp m α ∧ perpendicular_lp m β → parallel_pp α β) ∧
  ¬(∀ m n α, subset m α ∧ parallel_lp n α → parallel m n) ∧
  ¬(∀ m n α, parallel_lp m α ∧ parallel_lp n α → parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l205_20512


namespace NUMINAMATH_CALUDE_remainder_problem_l205_20537

theorem remainder_problem (N : ℤ) (D : ℤ) (h1 : D = 398) (h2 : ∃ Q', 2*N = D*Q' + 112) :
  ∃ Q, N = D*Q + 56 :=
sorry

end NUMINAMATH_CALUDE_remainder_problem_l205_20537


namespace NUMINAMATH_CALUDE_james_purchase_cost_l205_20583

/-- The total cost of James' purchase of dirt bikes and off-road vehicles, including registration fees. -/
def total_cost (dirt_bike_count : ℕ) (dirt_bike_price : ℕ) 
                (offroad_count : ℕ) (offroad_price : ℕ) 
                (registration_fee : ℕ) : ℕ :=
  dirt_bike_count * dirt_bike_price + 
  offroad_count * offroad_price + 
  (dirt_bike_count + offroad_count) * registration_fee

/-- Theorem stating that James' total cost is $1825 -/
theorem james_purchase_cost : 
  total_cost 3 150 4 300 25 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l205_20583


namespace NUMINAMATH_CALUDE_village_population_l205_20598

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 96 / 100 ∧ 
  partial_population = 23040 ∧ 
  (percentage * total_population : ℚ) = partial_population →
  total_population = 24000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l205_20598


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l205_20589

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let f := fun (a b c : ℝ) => (a + b - c)^2 / ((a + b)^2 + c^2)
  f x y z + f y z x + f z x y ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l205_20589


namespace NUMINAMATH_CALUDE_scientific_notation_10870_l205_20588

theorem scientific_notation_10870 :
  10870 = 1.087 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_10870_l205_20588


namespace NUMINAMATH_CALUDE_second_storm_duration_l205_20515

theorem second_storm_duration 
  (storm1_rate : ℝ) 
  (storm2_rate : ℝ) 
  (total_time : ℝ) 
  (total_rainfall : ℝ) 
  (h1 : storm1_rate = 30) 
  (h2 : storm2_rate = 15) 
  (h3 : total_time = 45) 
  (h4 : total_rainfall = 975) :
  ∃ (storm1_duration storm2_duration : ℝ),
    storm1_duration + storm2_duration = total_time ∧
    storm1_rate * storm1_duration + storm2_rate * storm2_duration = total_rainfall ∧
    storm2_duration = 25 := by
  sorry

#check second_storm_duration

end NUMINAMATH_CALUDE_second_storm_duration_l205_20515


namespace NUMINAMATH_CALUDE_square_difference_l205_20503

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l205_20503


namespace NUMINAMATH_CALUDE_max_value_less_than_two_l205_20552

theorem max_value_less_than_two (m : ℝ) (hm1 : 1 < m) (hm2 : m < 1 + Real.sqrt 2) :
  ∀ x y : ℝ, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2 := by
  sorry

#check max_value_less_than_two

end NUMINAMATH_CALUDE_max_value_less_than_two_l205_20552


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l205_20513

theorem unique_number_with_conditions : ∃! n : ℕ,
  50 < n ∧ n < 70 ∧
  n % 5 = 3 ∧
  n % 7 = 2 ∧
  n % 8 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l205_20513


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l205_20517

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back --/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back --/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 15)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  oranges_to_put_back fs = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l205_20517


namespace NUMINAMATH_CALUDE_john_remaining_money_l205_20535

def john_savings : ℕ := 5555
def ticket_cost : ℕ := 1200
def visa_cost : ℕ := 200

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem john_remaining_money :
  base_8_to_10 john_savings - ticket_cost - visa_cost = 1525 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l205_20535


namespace NUMINAMATH_CALUDE_miles_difference_l205_20518

/-- Given that Gervais drove an average of 315 miles for 3 days and Henri drove a total of 1,250 miles,
    prove that Henri drove 305 miles farther than Gervais. -/
theorem miles_difference (gervais_avg_daily : ℕ) (gervais_days : ℕ) (henri_total : ℕ) : 
  gervais_avg_daily = 315 → gervais_days = 3 → henri_total = 1250 → 
  henri_total - (gervais_avg_daily * gervais_days) = 305 := by
sorry

end NUMINAMATH_CALUDE_miles_difference_l205_20518


namespace NUMINAMATH_CALUDE_new_person_age_l205_20501

theorem new_person_age (group_size : ℕ) (age_decrease : ℕ) (replaced_age : ℕ) : 
  group_size = 10 → 
  age_decrease = 3 → 
  replaced_age = 45 → 
  ∃ (original_avg : ℚ) (new_avg : ℚ),
    original_avg - new_avg = age_decrease ∧
    group_size * original_avg - replaced_age = group_size * new_avg - 15 :=
by sorry

end NUMINAMATH_CALUDE_new_person_age_l205_20501


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l205_20574

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧ 
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l205_20574


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_cosine_sum_l205_20554

theorem seventh_root_of_unity_cosine_sum (z : ℂ) (α : ℝ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : z = Complex.exp (Complex.I * α)) :
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α) = -1/2 := by sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_cosine_sum_l205_20554


namespace NUMINAMATH_CALUDE_teacher_remaining_budget_l205_20563

/-- Calculates the remaining budget for a teacher after purchasing school supplies. -/
theorem teacher_remaining_budget 
  (last_year_remaining : ℕ) 
  (this_year_budget : ℕ) 
  (first_purchase : ℕ) 
  (second_purchase : ℕ) 
  (h1 : last_year_remaining = 6)
  (h2 : this_year_budget = 50)
  (h3 : first_purchase = 13)
  (h4 : second_purchase = 24) :
  last_year_remaining + this_year_budget - (first_purchase + second_purchase) = 19 :=
by sorry

end NUMINAMATH_CALUDE_teacher_remaining_budget_l205_20563


namespace NUMINAMATH_CALUDE_correct_calculation_l205_20579

theorem correct_calculation (x : ℝ) : 
  (x + 2.95 = 9.28) → (x - 2.95 = 3.38) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l205_20579


namespace NUMINAMATH_CALUDE_probability_gary_paula_letters_l205_20545

/-- The probability of drawing one letter from Gary's name and one from Paula's name -/
theorem probability_gary_paula_letters : 
  let total_letters : ℕ := 9
  let gary_letters : ℕ := 4
  let paula_letters : ℕ := 5
  let prob_gary_then_paula : ℚ := (gary_letters : ℚ) / total_letters * paula_letters / (total_letters - 1)
  let prob_paula_then_gary : ℚ := (paula_letters : ℚ) / total_letters * gary_letters / (total_letters - 1)
  prob_gary_then_paula + prob_paula_then_gary = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_gary_paula_letters_l205_20545


namespace NUMINAMATH_CALUDE_coins_given_to_laura_l205_20528

def coins_to_laura (piggy_bank : ℕ) (brother : ℕ) (father : ℕ) (final_count : ℕ) : ℕ :=
  piggy_bank + brother + father - final_count

theorem coins_given_to_laura :
  coins_to_laura 15 13 8 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_coins_given_to_laura_l205_20528


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l205_20502

theorem intersection_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -1}
  let B : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}
  (A ∩ B : Set ℝ) = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l205_20502


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l205_20558

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l205_20558


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l205_20519

theorem right_rectangular_prism_volume 
  (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 12) 
  (h_front : front_area = 8) 
  (h_bottom : bottom_area = 4) : 
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l205_20519


namespace NUMINAMATH_CALUDE_final_number_calculation_l205_20566

theorem final_number_calculation (initial_number : ℕ) : 
  initial_number = 8 → 3 * (2 * initial_number + 9) = 75 :=
by sorry

end NUMINAMATH_CALUDE_final_number_calculation_l205_20566


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l205_20508

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 3 = Real.sqrt ((3^a) * (3^b))) : 
  ∀ x y, x > 0 → y > 0 → (1/x + 1/y) ≥ (1/a + 1/b) → (1/a + 1/b) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l205_20508


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l205_20521

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 2 * I
  let z₂ : ℂ := 4 - 2 * I
  let z₃ : ℂ := 4 - 6 * I
  let z₄ : ℂ := 4 + 6 * I
  z₁ / z₂ + z₃ / z₄ = 14 / 65 - 8 / 65 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l205_20521


namespace NUMINAMATH_CALUDE_locus_is_perpendicular_line_l205_20595

/-- Two non-intersecting circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ  -- Center of the first circle
  O₂ : ℝ × ℝ  -- Center of the second circle
  R₁ : ℝ      -- Radius of the first circle
  R₂ : ℝ      -- Radius of the second circle
  h₁ : R₁ > 0
  h₂ : R₂ > 0
  h₃ : ‖O₁ - O₂‖ > R₁ + R₂  -- Circles do not intersect

/-- A point X is on the locus if it's the center of a circle that intersects 
    both given circles at diametrically opposite points -/
def IsOnLocus (tc : TwoCircles) (X : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ 
    r^2 = ‖X - tc.O₁‖^2 + tc.R₁^2 ∧
    r^2 = ‖X - tc.O₂‖^2 + tc.R₂^2

/-- The locus of centers of circles that divide two given non-intersecting circles in half 
    is a straight line perpendicular to the line segment connecting the centers of the given circles -/
theorem locus_is_perpendicular_line (tc : TwoCircles) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ X : ℝ × ℝ, IsOnLocus tc X ↔ a * X.1 + b * X.2 + c = 0) ∧
    a * (tc.O₂.1 - tc.O₁.1) + b * (tc.O₂.2 - tc.O₁.2) = 0 :=
  sorry

end NUMINAMATH_CALUDE_locus_is_perpendicular_line_l205_20595


namespace NUMINAMATH_CALUDE_wally_fraction_given_to_friends_l205_20509

def wally_total_tickets : ℕ := 400
def finley_tickets : ℕ := 220
def jensen_finley_ratio : Rat := 4 / 11

theorem wally_fraction_given_to_friends :
  (finley_tickets + (finley_tickets * jensen_finley_ratio)) / wally_total_tickets = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_wally_fraction_given_to_friends_l205_20509


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l205_20581

/-- Represents a hexagonal figure with red and yellow tiles -/
structure HexagonalFigure where
  redTiles : ℕ
  yellowTiles : ℕ

/-- Adds a border of yellow tiles to a hexagonal figure -/
def addBorder (figure : HexagonalFigure) : HexagonalFigure :=
  { redTiles := figure.redTiles,
    yellowTiles := figure.yellowTiles + 24 }

theorem yellow_red_difference_after_border (figure : HexagonalFigure) 
  (h1 : figure.redTiles = 12)
  (h2 : figure.yellowTiles = 8) :
  let newFigure := addBorder figure
  newFigure.yellowTiles - newFigure.redTiles = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l205_20581


namespace NUMINAMATH_CALUDE_correct_num_shirts_l205_20510

/-- The number of different colored neckties -/
def num_neckties : ℕ := 6

/-- The probability that all boxes contain a necktie and a shirt of the same color -/
def match_probability : ℚ := 1 / 120

/-- The number of different colored shirts -/
def num_shirts : ℕ := 2

/-- Theorem stating that given the number of neckties and the match probability,
    the number of shirts is correct -/
theorem correct_num_shirts :
  (1 : ℚ) / num_shirts ^ num_neckties = match_probability := by sorry

end NUMINAMATH_CALUDE_correct_num_shirts_l205_20510


namespace NUMINAMATH_CALUDE_inequality_solution_set_l205_20584

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2*(a - 2) * x < 4) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l205_20584


namespace NUMINAMATH_CALUDE_intersection_segment_length_l205_20582

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O₂ (x y m : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2 m ∧
  circle_O₁ B.1 B.2 ∧ circle_O₂ B.1 B.2 m

-- Define perpendicular tangents at point A
def perpendicular_tangents (A : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ × ℝ → ℝ × ℝ), 
    (t₁ A = (0, 0)) ∧ (t₂ A = (m, 0)) ∧ 
    (t₁ A • t₂ A = 0)  -- Dot product of tangent vectors is zero

-- Theorem statement
theorem intersection_segment_length 
  (A B : ℝ × ℝ) (m : ℝ) : 
  intersection_points A B m → 
  perpendicular_tangents A m → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l205_20582


namespace NUMINAMATH_CALUDE_reflection_of_M_across_y_axis_l205_20516

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The original point -/
def M : ℝ × ℝ := (3, 2)

theorem reflection_of_M_across_y_axis :
  reflect_y M = (-3, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_across_y_axis_l205_20516


namespace NUMINAMATH_CALUDE_square_area_error_l205_20561

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * 1.06
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l205_20561


namespace NUMINAMATH_CALUDE_school_travel_time_l205_20556

/-- 
If a boy reaches school 4 minutes earlier when walking at 9/8 of his usual rate,
then his usual time to reach the school is 36 minutes.
-/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) :
  usual_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_time_l205_20556


namespace NUMINAMATH_CALUDE_data_mode_and_median_l205_20557

def data : List ℕ := [6, 3, 5, 4, 3, 3]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem data_mode_and_median : 
  mode data = 3 ∧ median data = 3.5 := by sorry

end NUMINAMATH_CALUDE_data_mode_and_median_l205_20557


namespace NUMINAMATH_CALUDE_rice_sales_profit_l205_20578

-- Define the linear function
def sales_function (a b x : ℝ) : ℝ := a * x + b

-- Define the profit function
def profit_function (x y : ℝ) : ℝ := (x - 4) * y

-- Define the theorem
theorem rice_sales_profit 
  (a b : ℝ) 
  (h1 : ∀ x, 4 ≤ x → x ≤ 7 → sales_function a b x ≥ 0)
  (h2 : sales_function a b 5 = 950)
  (h3 : sales_function a b 6 = 900) :
  (a = -50 ∧ b = 1200) ∧
  (profit_function 6 (sales_function a b 6) = 1800) ∧
  (∀ x, 4 ≤ x → x ≤ 7 → profit_function x (sales_function a b x) ≤ 2550) ∧
  (profit_function 7 (sales_function a b 7) = 2550) := by
  sorry

end NUMINAMATH_CALUDE_rice_sales_profit_l205_20578


namespace NUMINAMATH_CALUDE_data_properties_l205_20520

def data : List ℝ := [3, 4, 2, 2, 4]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : List ℝ := sorry

theorem data_properties :
  median data = 3 ∧
  mean data = 3 ∧
  variance data = 0.8 ∧
  mode data ≠ [4] :=
sorry

end NUMINAMATH_CALUDE_data_properties_l205_20520


namespace NUMINAMATH_CALUDE_inequality_system_solution_l205_20590

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → 
  (a = 3 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l205_20590


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l205_20548

/-- Represents a position in the company with its count and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def positions : List Position := [
  ⟨"President", 1, 130000⟩,
  ⟨"Vice-President", 15, 90000⟩,
  ⟨"Director", 10, 80000⟩,
  ⟨"Associate Director", 8, 50000⟩,
  ⟨"Administrative Specialist", 37, 25000⟩
]

/-- The total number of employees --/
def totalEmployees : Nat := positions.foldl (fun acc p => acc + p.count) 0

/-- The median salary of the employees --/
def medianSalary : Nat := 25000

theorem median_salary_is_25000 :
  totalEmployees = 71 → medianSalary = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l205_20548


namespace NUMINAMATH_CALUDE_software_cost_proof_l205_20536

theorem software_cost_proof (total_devices : ℕ) (package1_cost package1_coverage : ℕ) 
  (package2_coverage : ℕ) (savings : ℕ) :
  total_devices = 50 →
  package1_cost = 40 →
  package1_coverage = 5 →
  package2_coverage = 10 →
  savings = 100 →
  (total_devices / package1_coverage * package1_cost - savings) / (total_devices / package2_coverage) = 60 :=
by sorry

end NUMINAMATH_CALUDE_software_cost_proof_l205_20536


namespace NUMINAMATH_CALUDE_income_spent_on_food_l205_20577

/-- Proves the percentage of income spent on food given other expenses -/
theorem income_spent_on_food (F : ℝ) : 
  F ≥ 0 ∧ F ≤ 100 →
  (100 - F - 25 - 0.8 * (75 - 0.75 * F) = 8) →
  F = 46.67 := by
  sorry

end NUMINAMATH_CALUDE_income_spent_on_food_l205_20577


namespace NUMINAMATH_CALUDE_waiter_customers_theorem_l205_20553

def final_customers (initial new left : ℕ) : ℕ :=
  initial - left + new

theorem waiter_customers_theorem (initial new left : ℕ) 
  (h1 : initial ≥ left) : 
  final_customers initial new left = initial - left + new :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_theorem_l205_20553


namespace NUMINAMATH_CALUDE_music_students_percentage_l205_20523

/-- Given a total number of students and the number of students taking dance and art,
    prove that the percentage of students taking music is 20%. -/
theorem music_students_percentage
  (total : ℕ)
  (dance : ℕ)
  (art : ℕ)
  (h1 : total = 400)
  (h2 : dance = 120)
  (h3 : art = 200) :
  (total - dance - art) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_music_students_percentage_l205_20523


namespace NUMINAMATH_CALUDE_one_common_point_condition_l205_20541

/-- A function f(x) = mx² - 4x + 3 has only one common point with the x-axis if and only if m = 0 or m = 4/3 -/
theorem one_common_point_condition (m : ℝ) : 
  (∃! x, m * x^2 - 4 * x + 3 = 0) ↔ (m = 0 ∨ m = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_one_common_point_condition_l205_20541


namespace NUMINAMATH_CALUDE_only_subtraction_correct_l205_20533

theorem only_subtraction_correct : 
  (¬(Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2)) ∧ 
  (¬(3 - 27/64 = 3/4)) ∧ 
  (3 - 8 = -5) ∧ 
  (¬(|Real.sqrt 2 - 1| = 1 - Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_only_subtraction_correct_l205_20533


namespace NUMINAMATH_CALUDE_ravish_failed_by_40_l205_20599

/-- The number of marks Ravish failed by in his board exam -/
def marks_failed (max_marks passing_percentage ravish_marks : ℕ) : ℕ :=
  (max_marks * passing_percentage / 100) - ravish_marks

/-- Proof that Ravish failed by 40 marks -/
theorem ravish_failed_by_40 :
  marks_failed 200 40 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ravish_failed_by_40_l205_20599


namespace NUMINAMATH_CALUDE_exists_product_sum_20000_l205_20555

theorem exists_product_sum_20000 : 
  ∃ k m : ℕ, 1 ≤ k ∧ k < m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_exists_product_sum_20000_l205_20555


namespace NUMINAMATH_CALUDE_inequality_system_solution_l205_20522

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x - a > 1 ∧ 2*x - 3 > a) ↔ x > a + 1) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l205_20522


namespace NUMINAMATH_CALUDE_kerosene_mixture_problem_l205_20585

/-- A mixture problem involving two liquids with different kerosene concentrations -/
theorem kerosene_mixture_problem :
  let first_liquid_kerosene_percent : ℝ := 25
  let second_liquid_kerosene_percent : ℝ := 30
  let second_liquid_parts : ℝ := 4
  let mixture_kerosene_percent : ℝ := 27
  let first_liquid_parts : ℝ := 6

  first_liquid_kerosene_percent / 100 * first_liquid_parts +
  second_liquid_kerosene_percent / 100 * second_liquid_parts =
  mixture_kerosene_percent / 100 * (first_liquid_parts + second_liquid_parts) :=
by sorry

end NUMINAMATH_CALUDE_kerosene_mixture_problem_l205_20585


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l205_20592

/-- Compound interest calculation -/
theorem compound_interest_calculation 
  (initial_deposit : ℝ) 
  (interest_rate : ℝ) 
  (time_period : ℕ) 
  (h1 : initial_deposit = 20000)
  (h2 : interest_rate = 0.03)
  (h3 : time_period = 5) :
  initial_deposit * (1 + interest_rate) ^ time_period = 
    20000 * (1 + 0.03) ^ 5 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l205_20592
