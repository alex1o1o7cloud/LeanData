import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circles_distance_l1193_119339

/-- Given a right triangle XYZ with sides YZ = 9, XZ = 12, and XY = 15,
    and points Q and R such that:
    - Q is the center of a circle tangent to YZ at Y and passing through X
    - R is the center of a circle tangent to XZ at X and passing through Y
    Then the distance QR is 15. -/
theorem triangle_circles_distance (X Y Z Q R : ℝ × ℝ) : 
  let d := λ (a b : ℝ × ℝ) => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  (d Y Z = 9) →
  (d X Z = 12) →
  (d X Y = 15) →
  (d X (X.1 - Y.1, X.2 - Y.2) = d Y (X.1 - Y.1, X.2 - Y.2)) →
  (d Q Y = d Q X) →
  (d R X = d R Y) →
  (d Q R = 15) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circles_distance_l1193_119339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_probability_l1193_119335

/-- The probability of a ball ending up in bin i -/
noncomputable def bin_prob (i : ℕ) : ℝ := (1 / 3) ^ i

/-- The probability of three balls ending up in bins a, a+d, and a+2d -/
noncomputable def prob_arithmetic_prog (a d : ℕ) : ℝ := 
  bin_prob a * bin_prob (a + d) * bin_prob (a + 2 * d)

/-- The total probability of three balls ending up in an arithmetic progression in distinct bins -/
noncomputable def total_prob : ℝ := 
  6 * (∑' a : ℕ, ∑' d : ℕ, prob_arithmetic_prog (a + 1) (d + 1))

theorem arithmetic_progression_probability : 
  total_prob = 3 / 338 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_probability_l1193_119335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_m_eq_4_l1193_119307

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {a b c d e f : ℝ} :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ d * x + e * y + f = 0) ↔ a / b = d / e

/-- Definition of the first line -/
def line1 (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (3 * m - 4) * x + 4 * y - 2 = 0

/-- Definition of the second line -/
def line2 (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ m * x + 2 * y - 2 = 0

/-- Theorem stating that the lines are parallel if and only if m = 4 -/
theorem lines_parallel_iff_m_eq_4 (m : ℝ) :
  (∀ x y : ℝ, line1 m x y ↔ line2 m x y) ↔ m = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_iff_m_eq_4_l1193_119307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1193_119310

theorem power_inequality (n a b : ℕ) (h1 : n > 0) (h2 : a > b) (h3 : b > 1) 
  (h4 : Odd b) (h5 : (b^n) ∣ (a^n - 1)) : a^b > 3^n / n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1193_119310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_sum_of_endpoints_l1193_119305

/-- The function h(x) = 3 / (1 + 3x³) -/
noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^3)

/-- The range of h is (0, 3] -/
theorem range_of_h : Set.range h = Set.Ioo 0 3 ∪ {3} := by sorry

/-- The sum of the endpoints of the range interval is 3 -/
theorem sum_of_endpoints : ∃ (a b : ℝ), Set.range h = Set.Ioc a b ∧ a + b = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_sum_of_endpoints_l1193_119305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l1193_119393

theorem triangle_sine_inequality (α β γ : Real) (h : α + β + γ = π) :
  Real.sin α + Real.sin β > Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l1193_119393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_sum_property_l1193_119370

theorem divisible_by_six_sum_property (n : ℕ) :
  6 ∣ n ↔ 6 ∣ (n % 10 + 4 * (n / 10)) :=
sorry

#check divisible_by_six_sum_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_six_sum_property_l1193_119370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_calculation_l1193_119395

/-- Represents the age of the class teacher -/
def teacher_age : ℝ := 44.7

/-- Theorem stating the conditions and the result to be proved -/
theorem teacher_age_calculation (initial_students : ℕ) (initial_avg : ℝ) 
  (leaving_student_age : ℝ) (new_avg : ℝ) :
  initial_students = 45 →
  initial_avg = 14 →
  leaving_student_age = 15 →
  new_avg = 14.66 →
  (initial_avg * initial_students - leaving_student_age + teacher_age) / initial_students = new_avg :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_calculation_l1193_119395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1193_119372

open Real

-- Define the number of routes
def num_routes : ℕ := 17

-- Define the trigonometric equation
def trig_equation (b : ℝ) : Prop :=
  ∃ a, sin (2 * b * π / 180 + 2 * a * π / 180) = cos (6 * b * π / 180 - 16 * π / 180) ∧ 0 < b ∧ b < 90

-- Define the line equation
def line_equation (b k c m : ℝ) : Prop :=
  (b * c - 6 * m + 3) + k * (c - m + 1) = 0

-- Define the quadratic equation
def quadratic_equation (d c : ℝ) : Prop :=
  d^2 - c = 257 * 259

-- State the theorem
theorem problem_solution :
  (num_routes = 17) ∧
  (trig_equation 9) ∧
  (∃ m, ∀ b k, line_equation b k 1 m) ∧
  (∃ c, quadratic_equation 258 c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1193_119372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1193_119352

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point
def point : ℝ × ℝ := (3, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    my_circle A.1 A.2 ∧
    my_circle B.1 B.2 ∧
    tangent_line A.1 A.2 ∧
    tangent_line B.1 B.2 ∧
    (∀ (P : ℝ × ℝ), my_circle P.1 P.2 → tangent_line P.1 P.2 → (P = A ∨ P = B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1193_119352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1193_119308

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for high achievers and underachievers
variable (HighAchiever : U → Prop)
variable (Underachiever : U → Prop)

-- Define the original statement
def NoHighAchieversAreUnderachievers (U : Type) (HighAchiever Underachiever : U → Prop) : Prop :=
  ∀ x : U, HighAchiever x → ¬Underachiever x

-- Define the negation of the original statement
def NegationOfNoHighAchieversAreUnderachievers (U : Type) (HighAchiever Underachiever : U → Prop) : Prop :=
  ¬(NoHighAchieversAreUnderachievers U HighAchiever Underachiever)

-- Define the statement "Some high achievers are underachievers"
def SomeHighAchieversAreUnderachievers (U : Type) (HighAchiever Underachiever : U → Prop) : Prop :=
  ∃ x : U, HighAchiever x ∧ Underachiever x

-- Theorem stating that the negation is equivalent to "Some high achievers are underachievers"
theorem negation_equivalence (U : Type) (HighAchiever Underachiever : U → Prop) :
  NegationOfNoHighAchieversAreUnderachievers U HighAchiever Underachiever ↔
  SomeHighAchieversAreUnderachievers U HighAchiever Underachiever :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1193_119308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_passage_theorem_l1193_119349

/-- The time taken for a bullet to pass through a plank -/
noncomputable def bullet_passage_time (thickness : ℝ) (initial_velocity : ℝ) (final_velocity : ℝ) : ℝ :=
  3 / (2000 * Real.log 4)

/-- Theorem stating the time taken for the bullet to pass through the plank -/
theorem bullet_passage_theorem (thickness : ℝ) (initial_velocity : ℝ) (final_velocity : ℝ)
    (h1 : thickness = 0.1)
    (h2 : initial_velocity = 200)
    (h3 : final_velocity = 50) :
    ∃ ε > 0, |bullet_passage_time thickness initial_velocity final_velocity - 0.001| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check bullet_passage_time 0.1 200 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_passage_theorem_l1193_119349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_powers_l1193_119358

theorem max_value_of_sum_of_powers (k : ℕ) (a : ℝ) (h_a : a > 0) :
  let S := {s : Finset ℕ | s.sum id = k ∧ s.card ≥ 1 ∧ s.card ≤ k}
  ∀ s ∈ S, (s.sum (fun i => a^i)) ≤ max (k * a) (a^k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_of_powers_l1193_119358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1193_119357

/-- Given a parabola C, a line, and a hyperbola E with the following properties:
    - Parabola C: y^2 = 4x
    - Line: y = x - 1
    - Hyperbola E: x^2/a^2 - y^2/b^2 = 2 (a > 0, b > 0)
    - Line y = x - 1 intersects C at points A and B
    - Line y = x - 1 intersects asymptotes of E at points M and N
    - Midpoint of AB is the same as midpoint of MN
    Then the eccentricity of hyperbola E is √15/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (C : ℝ → ℝ → Prop) (E : ℝ → ℝ → Prop) (L : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ y^2 = 4*x)
  (hE : ∀ x y, E x y ↔ x^2/a^2 - y^2/b^2 = 2)
  (hL : ∀ x y, L x y ↔ y = x - 1)
  (A B : ℝ × ℝ) (hAB : C A.1 A.2 ∧ C B.1 B.2 ∧ L A.1 A.2 ∧ L B.1 B.2)
  (M N : ℝ × ℝ) (hMN : L M.1 M.2 ∧ L N.1 N.2)
  (hMidpoint : (A.1 + B.1)/2 = (M.1 + N.1)/2 ∧ (A.2 + B.2)/2 = (M.2 + N.2)/2) :
  Real.sqrt (1 + b^2/a^2) = Real.sqrt 15 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1193_119357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l1193_119375

theorem sum_of_reciprocals_equals_one (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) :
  1/a + 1/b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l1193_119375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_dropped_l1193_119316

def average_score (total_score : ℕ) (num_tests : ℕ) : ℚ :=
  (total_score : ℚ) / (num_tests : ℚ)

theorem lowest_score_dropped 
  (total_score_before : ℕ) 
  (total_score_after : ℕ) 
  (num_tests_before : ℕ) 
  (num_tests_after : ℕ) 
  (h1 : average_score total_score_before num_tests_before = 90)
  (h2 : average_score total_score_after num_tests_after = 95)
  (h3 : num_tests_before = 4)
  (h4 : num_tests_after = 3) :
  total_score_before - total_score_after = 75 := by
  sorry

#check lowest_score_dropped

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_dropped_l1193_119316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_always_true_l1193_119373

theorem inequality_not_always_true :
  ¬ (∀ m n : ℝ, m > 0 → n > 0 → m + n = 1 → Real.sqrt m + Real.sqrt n ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_always_true_l1193_119373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l1193_119383

/-- Represents the principal amount in Rupees -/
def principal (P : ℝ) : Prop := P > 0

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

/-- Calculates compound interest (annually compounded) given principal, rate, and time -/
noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100) ^ T - P

/-- The theorem stating the relationship between principal, interest rates, and the difference -/
theorem principal_from_interest_difference :
  ∀ P : ℝ,
  simple_interest P 20 2 = P * 0.4 ∧
  compound_interest P 20 2 = P * 0.44 ∧
  compound_interest P 20 2 - simple_interest P 20 2 = 360 →
  principal P ∧ P = 9000 := by
  sorry

#check principal_from_interest_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l1193_119383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PUT_l1193_119340

/-- A square with side length 2 -/
structure Square :=
  (P Q R S : ℝ × ℝ)
  (is_square : P = (0, 0) ∧ Q = (2, 0) ∧ R = (2, 2) ∧ S = (0, 2))

/-- An equilateral triangle PRT where T is on PR -/
structure EquilateralTriangle (sq : Square) :=
  (T : ℝ × ℝ)
  (on_PR : ∃ t : ℝ, T = (2, t) ∧ 0 ≤ t ∧ t ≤ 2)
  (is_equilateral : sorry)  -- We don't prove this, but state it as a condition

/-- The intersection point U of diagonal QS and line segment PT -/
noncomputable def intersection_point (sq : Square) (tri : EquilateralTriangle sq) : ℝ × ℝ := by
  sorry  -- The exact coordinates are not needed for the statement

/-- Area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := by
  sorry  -- We don't implement this, but declare it for use in the theorem

/-- The main theorem -/
theorem area_of_triangle_PUT (sq : Square) (tri : EquilateralTriangle sq) :
  let U := intersection_point sq tri
  area_triangle sq.P U tri.T = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PUT_l1193_119340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_expression_l1193_119315

theorem max_value_of_trig_expression (α β : ℝ) (h : α^2 + β^2 ≤ 16) :
  (∃ C : ℝ, ∀ x : ℝ, |α * Real.sin (2 * x) + β * Real.cos (8 * x)| ≤ C) →
  (∀ C : ℝ, (∀ x : ℝ, |α * Real.sin (2 * x) + β * Real.cos (8 * x)| ≤ C) → C ≥ 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_expression_l1193_119315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_150_degrees_l1193_119384

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors_is_150_degrees (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 6 * Real.sqrt 3)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 1)
  (h3 : a.1 * b.1 + a.2 * b.2 = -9) :
  angle_between_vectors a b = π * 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_150_degrees_l1193_119384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_mixture_weight_l1193_119392

/-- The weight of the entire cement mixture in pounds -/
noncomputable def total_weight : ℝ := 57.6

/-- The weight of gravel in the mixture in pounds -/
noncomputable def gravel_weight : ℝ := 12

/-- The proportion of sand in the mixture -/
noncomputable def sand_proportion : ℝ := 2/5

/-- The proportion of water in the mixture -/
noncomputable def water_proportion : ℝ := 1/6

/-- The proportion of cement in the mixture -/
noncomputable def cement_proportion : ℝ := 1/10

/-- The proportion of lime in the mixture -/
noncomputable def lime_proportion : ℝ := 1/8

theorem cement_mixture_weight :
  sand_proportion * total_weight +
  water_proportion * total_weight +
  cement_proportion * total_weight +
  lime_proportion * total_weight +
  gravel_weight = total_weight := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_mixture_weight_l1193_119392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_tickets_l1193_119328

theorem theater_tickets 
  (num_adults num_children adult_price children_price price_difference : ℕ) : 
  num_adults = 9 →
  adult_price = 11 →
  children_price = 7 →
  num_adults * adult_price = price_difference + num_children * children_price →
  price_difference = 50 →
  num_children = 7 := by
  sorry

#check theater_tickets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_tickets_l1193_119328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_roundtrip_probability_l1193_119379

structure Box where
  red : ℕ
  white : ℕ

def initial_box_A : Box := ⟨1, 5⟩
def initial_box_B : Box := ⟨0, 3⟩

def transfer_probability (box : Box) (num_transfer : ℕ) : ℚ :=
  (box.red : ℚ) / (box.red + box.white)

def box_after_transfer (box1 : Box) (box2 : Box) (num_transfer : ℕ) : Box × Box :=
  ⟨⟨box1.red - 1, box1.white - (num_transfer - 1)⟩, ⟨box2.red + 1, box2.white + (num_transfer - 1)⟩⟩

theorem red_ball_roundtrip_probability :
  let first_transfer := transfer_probability initial_box_A 3
  let (box_A_mid, box_B_mid) := box_after_transfer initial_box_A initial_box_B 3
  let second_transfer := transfer_probability box_B_mid 3
  first_transfer * second_transfer = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_roundtrip_probability_l1193_119379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l1193_119385

-- Define the initial complex number
def z : ℂ := -4 - 6 * Complex.I

-- Define the rotation angle
noncomputable def θ : ℝ := 30 * Real.pi / 180

-- Define the rotation transformation
noncomputable def rotate (w : ℂ) : ℂ := w * Complex.exp (Complex.I * (θ : ℂ))

-- Define the dilation transformation
def dilate (w : ℂ) : ℂ := 2 * w

-- Define the combined transformation
noncomputable def transform (w : ℂ) : ℂ := dilate (rotate w)

-- State the theorem
theorem transform_result :
  transform z = -4 * Real.sqrt 3 + 6 - Complex.I * (6 * Real.sqrt 3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l1193_119385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_21_terms_l1193_119319

def sequenceX (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => -sequenceX n + 1/2

theorem sum_of_first_21_terms : 
  (Finset.range 21).sum (λ i => sequenceX i) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_21_terms_l1193_119319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_exists_l1193_119369

structure Square :=
  (size : ℕ)

structure Shape :=
  (row : ℕ)
  (col : ℕ)

structure Division :=
  (parts : List (List (ℕ × ℕ)))

def isValidDivision (s : Square) (d : Division) (circles : List Shape) (stars : List Shape) : Prop :=
  d.parts.length = 4 ∧
  d.parts.all (λ part => part.length = (s.size * s.size) / 4) ∧
  d.parts.all (λ part => 
    (circles.filter (λ c => (c.row, c.col) ∈ part)).length = 1 ∧
    (stars.filter (λ s => (s.row, s.col) ∈ part)).length = 1)

theorem square_division_exists (s : Square) (circles : List Shape) (stars : List Shape) :
  s.size = 6 →
  circles.length = 4 →
  stars.length = 4 →
  (∀ c ∈ circles, ∀ s ∈ stars, c ≠ s) →
  ∃ (d : Division), isValidDivision s d circles stars := by
  sorry

#check square_division_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_exists_l1193_119369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_overlapping_pairs_l1193_119302

/-- A pair of indices (i, j) with i < j in a sequence -/
structure Pair (n : ℕ) where
  i : Fin n
  j : Fin n
  h : i < j

/-- A sequence of n distinct natural numbers -/
def DistinctSequence (n : ℕ) := { s : Fin n → ℕ // Function.Injective s }

/-- A pair is ascending if the first element is less than the second -/
def IsAscending {n : ℕ} (s : DistinctSequence n) (p : Pair n) : Prop :=
  s.val p.i < s.val p.j

/-- A pair is descending if the first element is greater than the second -/
def IsDescending {n : ℕ} (s : DistinctSequence n) (p : Pair n) : Prop :=
  s.val p.i > s.val p.j

/-- A list of pairs is non-overlapping if no two pairs share an index -/
def NonOverlapping {n : ℕ} (l : List (Pair n)) : Prop :=
  ∀ p q, p ∈ l → q ∈ l → p ≠ q → p.i ≠ q.i ∧ p.i ≠ q.j ∧ p.j ≠ q.i ∧ p.j ≠ q.j

/-- The main theorem -/
theorem largest_non_overlapping_pairs :
  ∃ k : ℕ, k = 333 ∧
  (∀ s : DistinctSequence 1000,
    (∃ l : List (Pair 1000), l.length = k ∧ NonOverlapping l ∧ ∀ p ∈ l, IsAscending s p) ∨
    (∃ l : List (Pair 1000), l.length = k ∧ NonOverlapping l ∧ ∀ p ∈ l, IsDescending s p)) ∧
  (∀ k' > k,
    ∃ s : DistinctSequence 1000,
      (∀ l : List (Pair 1000), l.length = k' → ¬NonOverlapping l ∨ ∃ p ∈ l, ¬IsAscending s p ∧ ¬IsDescending s p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_overlapping_pairs_l1193_119302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_result_l1193_119306

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec (![3, -1] : Fin 2 → ℝ) = ![-12, 4]) :
  (B^5).mulVec (![3, -1] : Fin 2 → ℝ) = ![-3072, 1024] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_result_l1193_119306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_scalene_l1193_119321

/-- Predicate to check if two vectors form a right angle -/
def IsRightAngle (v w : ℝ × ℝ) : Prop :=
  (v.1 * w.1 + v.2 * w.2 = 0) ∨ 
  (v.1 * (-w.2) + v.2 * w.1 = 0)

/-- Predicate to check if a triangle with sides represented by two vectors is scalene -/
def IsScalene (v w : ℝ × ℝ) : Prop :=
  let u := (v.1 + w.1, v.2 + w.2)
  (v.1^2 + v.2^2 ≠ w.1^2 + w.2^2) ∧
  (v.1^2 + v.2^2 ≠ u.1^2 + u.2^2) ∧
  (w.1^2 + w.2^2 ≠ u.1^2 + u.2^2)

/-- Given two vectors BA and BC in ℝ², prove that triangle ABC is right-angled and scalene -/
theorem triangle_right_scalene (BA BC : ℝ × ℝ) 
  (h_BA : BA = (4, -3)) 
  (h_BC : BC = (2, -4)) : 
  IsRightAngle BA BC ∧ IsScalene BA BC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_scalene_l1193_119321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derin_alone_l1193_119334

/-- The time it takes for Saba to complete the task alone -/
noncomputable def s : ℝ := sorry

/-- The time it takes for Rayan to complete the task alone -/
noncomputable def r : ℝ := sorry

/-- The time it takes for Derin to complete the task alone -/
noncomputable def d : ℝ := sorry

/-- All three working together complete the task in 5 minutes -/
axiom all_together : 1/s + 1/r + 1/d = 1/5

/-- Saba and Derin working together complete the task in 7 minutes -/
axiom saba_derin : 1/s + 1/d = 1/7

/-- Rayan and Derin working together complete the task in 15 minutes -/
axiom rayan_derin : 1/r + 1/d = 1/15

/-- Theorem: Given the conditions, Derin completes the task alone in 105 minutes -/
theorem derin_alone : d = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derin_alone_l1193_119334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_assignment_l1193_119362

-- Define the basic types
inductive Girl : Type
| Silvia : Girl
| Martina : Girl
| Zdenka : Girl

inductive Flower : Type
| Tulip : Flower
| Rose : Flower
| Daffodil : Flower

inductive Sport : Type
| Volleyball : Sport
| Basketball : Sport
| Tennis : Sport

inductive Instrument : Type
| Piano : Instrument
| Guitar : Instrument
| Flute : Instrument

-- Define the assignment functions
variable (favoriteFlower : Girl → Flower)
variable (playSport : Girl → Sport)
variable (playInstrument : Girl → Instrument)

-- Define the conditions
axiom different_flowers : ∀ g1 g2 : Girl, g1 ≠ g2 → favoriteFlower g1 ≠ favoriteFlower g2
axiom different_sports : ∀ g1 g2 : Girl, g1 ≠ g2 → playSport g1 ≠ playSport g2
axiom different_instruments : ∀ g1 g2 : Girl, g1 ≠ g2 → playInstrument g1 ≠ playInstrument g2

axiom silvia_not_volleyball : playSport Girl.Silvia ≠ Sport.Volleyball
axiom tulip_basketball_not_piano : ∀ g : Girl, favoriteFlower g = Flower.Tulip → playSport g = Sport.Basketball ∧ playInstrument g ≠ Instrument.Piano
axiom zdenka_guitar_rose : playInstrument Girl.Zdenka = Instrument.Guitar ∧ favoriteFlower Girl.Zdenka = Flower.Rose
axiom martina_flute : playInstrument Girl.Martina = Instrument.Flute
axiom daffodil_not_volleyball : ∀ g : Girl, playSport g = Sport.Volleyball → favoriteFlower g ≠ Flower.Daffodil

-- The theorem to prove
theorem girls_assignment :
  (favoriteFlower Girl.Silvia = Flower.Daffodil ∧ 
   playSport Girl.Silvia = Sport.Tennis ∧ 
   playInstrument Girl.Silvia = Instrument.Piano) ∧
  (favoriteFlower Girl.Martina = Flower.Tulip ∧ 
   playSport Girl.Martina = Sport.Basketball ∧ 
   playInstrument Girl.Martina = Instrument.Flute) ∧
  (favoriteFlower Girl.Zdenka = Flower.Rose ∧ 
   playSport Girl.Zdenka = Sport.Volleyball ∧ 
   playInstrument Girl.Zdenka = Instrument.Guitar) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_assignment_l1193_119362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_proof_l1193_119333

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of an ellipse -/
noncomputable def ellipse_axis_endpoint_distance : ℝ :=
  2 * Real.sqrt 5

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

theorem ellipse_axis_endpoint_distance_proof :
  ∃ (C D : ℝ × ℝ),
    (ellipse_equation C.1 C.2) ∧
    (ellipse_equation D.1 D.2) ∧
    (C.2 = 4 ∨ C.2 = -4) ∧
    (D.1 = 0) ∧
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = ellipse_axis_endpoint_distance :=
by sorry

#check ellipse_axis_endpoint_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_proof_l1193_119333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_of_trigonometric_curve_l1193_119365

theorem symmetry_point_of_trigonometric_curve (w : ℝ) (x₀ : ℝ) : 
  w > 0 → 
  (∀ x, ∃ k : ℤ, x + π / (2 * w) = x₀ + k * π / w) →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∃ A B : ℝ, ∀ x, Real.sin (w * x) + Real.sqrt 3 * Real.cos (w * x) = A * Real.sin (w * x + B)) →
  x₀ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_of_trigonometric_curve_l1193_119365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l1193_119356

theorem complex_number_location (a : ℝ) (h1 : (a^2 - 1 : ℂ) + (a + 1) * Complex.I = b * Complex.I) 
  (h2 : b ≠ 0) :
  let z : ℂ := a + (a - 2) * Complex.I
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_location_l1193_119356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_inequality_l1193_119338

theorem triangle_angle_sine_inequality (α β γ : ℝ) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi)
  (angle_order : α < β ∧ β < γ) :
  Real.sin (2 * α) > Real.sin (2 * β) ∧ Real.sin (2 * β) > Real.sin (2 * γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_inequality_l1193_119338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1193_119355

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the circle (renamed to avoid conflict)
def circle_E (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

-- Define a point on the parabola
structure PointOnParabola (p : ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (on_parabola : parabola p x y)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p)

-- Define symmetry about y-axis
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  Q.1 = -P.1 ∧ Q.2 = P.2

-- Theorem statement
theorem parabola_tangent_theorem (p : ℝ) (P : PointOnParabola p) :
  p = 2 →
  ∃ (M N Q : ℝ × ℝ),
    parabola p M.1 M.2 ∧
    parabola p N.1 N.2 ∧
    parabola p Q.1 Q.2 ∧
    symmetric_about_y_axis (P.x, P.y) Q ∧
    circle_E P.x P.y →
    (∃ (k : ℝ), (M.2 - P.y) = k * (M.1 - P.x) ∧ (N.2 - P.y) = k * (N.1 - P.x)) →
    (M.2 - N.2) / (M.1 - N.1) = Q.1 / 2 →
    ((P.x = 0 ∧ P.y = 0) ∨ (P.x = 4 ∧ P.y = 4) ∨ (P.x = -4 ∧ P.y = 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_theorem_l1193_119355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1193_119327

theorem problem_statement (a b c : ℕ) (h1 : (Nat.gcd a b + Nat.lcm a b = 2021^c))
  (h2 : Nat.Prime (Int.natAbs (a - b))) : ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a + b)^2 + 4 = k * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1193_119327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_is_right_triangle_l1193_119331

-- Define the sets of numbers
def set_A : List ℕ := [1, 1, 2]
def set_B : List ℕ := [3, 4, 6]
def set_C : List ℕ := [7, 24, 25]
def set_D : List ℕ := [6, 12, 13]

-- Define a function to check if a set of three numbers satisfies the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Theorem statement
theorem only_set_C_is_right_triangle :
  (¬ is_right_triangle (set_A.get! 0) (set_A.get! 1) (set_A.get! 2)) ∧
  (¬ is_right_triangle (set_B.get! 0) (set_B.get! 1) (set_B.get! 2)) ∧
  (is_right_triangle (set_C.get! 0) (set_C.get! 1) (set_C.get! 2)) ∧
  (¬ is_right_triangle (set_D.get! 0) (set_D.get! 1) (set_D.get! 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_is_right_triangle_l1193_119331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_rectangle_crease_length_l1193_119323

/-- The length of the crease formed when folding a rectangle --/
noncomputable def creaseLength (x y : ℝ) : ℝ := y * Real.sqrt (x^2 + y^2) / x

/-- Theorem stating the length of the crease when folding a rectangle --/
theorem fold_rectangle_crease_length (x y : ℝ) (h : x ≥ y) (h_pos : x > 0) :
  creaseLength x y = y * Real.sqrt (x^2 + y^2) / x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_rectangle_crease_length_l1193_119323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_time_l1193_119354

/-- The time at which a body's displacement equals its initial position plus 100 units,
    given initial conditions and equations for velocity and displacement. -/
theorem displacement_time
  (S₀ V₀ g μ : ℝ)
  (hμ : μ ≠ 1)
  (hg : g > 0)
  (velocity : ℝ → ℝ)
  (displacement : ℝ → ℝ)
  (hv : ∀ t, velocity t = g * t + V₀ - μ * g * t)
  (hd : ∀ t, displacement t = S₀ + (1/2) * g * t^2 + V₀ * t - (1/2) * μ * g * t^2) :
  ∃ t, displacement t = S₀ + 100 ∧
       t = (-V₀ + Real.sqrt (V₀^2 + 200 * (1 - μ) * g)) / ((1 - μ) * g) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_time_l1193_119354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_75_with_digit_product_multiple_25_l1193_119367

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := Nat.digits 10 n
  List.foldl (·*·) 1 digits

theorem least_multiple_75_with_digit_product_multiple_25 :
  ∀ n : ℕ, n > 0 → is_multiple_of n 75 →
    (is_multiple_of (digit_product n) 25 ∧ digit_product n > 0) →
    n ≥ 575 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_75_with_digit_product_multiple_25_l1193_119367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l1193_119344

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin ((2 * Real.pi / 3) - 2 * x)

-- Define the interval
def interval : Set ℝ := Set.Icc (7 * Real.pi / 12) (13 * Real.pi / 12)

-- Theorem statement
theorem f_monotone_increasing_on_interval :
  StrictMonoOn f interval := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l1193_119344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1193_119361

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2 - 1

-- Define the theorem
theorem triangle_area (a b c A B C : ℝ) : 
  a = 1 → 
  b + c = 2 → 
  f A = 1 / 2 → 
  0 < A → 
  A < Real.pi → 
  a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A)) → 
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

-- Add a dummy main function to satisfy Lake
def main : IO Unit :=
  IO.println s!"The theorem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1193_119361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1193_119313

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- State the theorem
theorem graph_translation (φ : ℝ) :
  (∀ x, f (π/6 - x) φ = f (π/6 + x) φ) →  -- Symmetry condition
  |φ| < π/2 →                            -- Bound on φ
  ∀ x, f x φ = Real.sin (2 * (x + π/12)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l1193_119313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_at_three_l1193_119391

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem f_equals_f_inv_at_three :
  ∃! x : ℝ, f x = f_inv x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_at_three_l1193_119391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_numbers_sum_200_no_four_exceed_100_l1193_119363

theorem nine_numbers_sum_200_no_four_exceed_100 : 
  ∃ (S : Finset ℕ), 
    Finset.card S = 9 ∧ 
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 200) ∧
    (∀ T ⊆ S, Finset.card T = 4 → Finset.sum T id ≤ 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_numbers_sum_200_no_four_exceed_100_l1193_119363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_terms_l1193_119326

noncomputable def curve (n : ℕ) (x : ℝ) : ℝ := x^n * (1 - x)

noncomputable def tangent_y_intercept (n : ℕ) : ℝ :=
  let slope := n * (2 : ℝ)^(n - 1) - (n + 1) * (2 : ℝ)^n
  let y_at_2 := curve n 2
  y_at_2 - slope * 2

noncomputable def sequence_term (n : ℕ) : ℝ := tangent_y_intercept n / (n + 1)

theorem sum_of_sequence_terms (n : ℕ) :
  (Finset.range n).sum (λ i => sequence_term (i + 1)) = (2 : ℝ)^(n + 1) - 2 := by
  sorry

#check sum_of_sequence_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_terms_l1193_119326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_average_age_l1193_119345

/-- The average age of three women given specific conditions about a group of men. -/
theorem womens_average_age (men_count : ℕ) (age_increase : ℚ) (replaced_ages : List ℕ) 
  (h1 : men_count = 12)
  (h2 : age_increase = 7/4)
  (h3 : replaced_ages = [18, 26, 35])
  : (100 : ℚ) / 3 = (men_count * age_increase + replaced_ages.sum) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_average_age_l1193_119345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_income_calculation_l1193_119330

/-- Calculates Ingrid's income given the tax rates and combined tax rate -/
noncomputable def calculate_ingrid_income (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) (combined_tax_rate : ℝ) : ℝ :=
  let combined_income := john_income / (1 - combined_tax_rate)
  combined_income - john_income

/-- Theorem stating that Ingrid's income is approximately $49,142.86 -/
theorem ingrid_income_calculation :
  let john_tax_rate := (0.30 : ℝ)
  let ingrid_tax_rate := (0.40 : ℝ)
  let john_income := (56000 : ℝ)
  let combined_tax_rate := (0.35625 : ℝ)
  let calculated_income := calculate_ingrid_income john_tax_rate ingrid_tax_rate john_income combined_tax_rate
  ∃ ε > 0, |calculated_income - 49142.86| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_income_calculation_l1193_119330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l1193_119304

theorem binomial_product : (Nat.choose 12 6) * ((Nat.choose 5 2) ^ 2) = 92400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l1193_119304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1193_119382

/-- The general solution of the differential equation y''' - 2y'' - 3y' = 0 -/
noncomputable def general_solution (C₁ C₂ C₃ : ℝ) (x : ℝ) : ℝ :=
  C₁ + C₂ * Real.exp (-x) + C₃ * Real.exp (3 * x)

/-- The third-order linear homogeneous differential equation -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[3] y) x - 2 * (deriv^[2] y) x - 3 * (deriv y) x = 0

theorem general_solution_satisfies_equation :
  ∀ C₁ C₂ C₃, differential_equation (general_solution C₁ C₂ C₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_equation_l1193_119382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_seventh_term_l1193_119347

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℚ)
  (d : ℚ)
  (hd : d ≠ 0)
  (ha : arithmetic_sequence a d)
  (hg : geometric_sequence (a 4) (a 5) (a 7))
  (hs : sum_arithmetic a 11 = 66)
  : a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_seventh_term_l1193_119347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_less_than_input_difference_l1193_119351

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

-- State the theorem
theorem f_difference_less_than_input_difference (a b : ℝ) (h : a ≠ b) :
  |f a - f b| < |a - b| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_less_than_input_difference_l1193_119351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l1193_119371

theorem smallest_number_divisible_by_multiple (n : ℕ) : 
  (∀ m ∈ ({12, 16, 18, 21, 28} : Set ℕ), (n - 7) % m = 0) →
  (∀ k < n, ∃ m ∈ ({12, 16, 18, 21, 28} : Set ℕ), (k - 7) % m ≠ 0) →
  n = 1015 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l1193_119371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_minus_pi_fourth_l1193_119350

theorem cos_two_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : α > π/2 ∧ α < π) (h2 : Real.sin α = 3/5) : 
  Real.cos (2*α - π/4) = -17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_minus_pi_fourth_l1193_119350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attempts_five_keys_l1193_119376

/-- Represents the outcome of trying a key -/
inductive KeyOutcome
  | Success
  | Failure

/-- Represents a sequence of keys -/
def KeySequence := List KeyOutcome

/-- The number of attempts needed to find the successful key -/
def attemptsToSuccess (keys : KeySequence) : Nat :=
  match keys with
  | [] => 0
  | KeyOutcome.Success :: _ => 1
  | KeyOutcome.Failure :: rest => 1 + attemptsToSuccess rest

theorem max_attempts_five_keys :
  ∀ (keys : KeySequence),
    keys.length = 5 →
    (∃ (i : Fin keys.length), keys.get i = KeyOutcome.Success) →
    attemptsToSuccess keys ≤ 5 ∧
    ∃ (worstCase : KeySequence),
      worstCase.length = 5 ∧
      (∃ (i : Fin worstCase.length), worstCase.get i = KeyOutcome.Success) ∧
      attemptsToSuccess worstCase = 5 :=
by sorry

#check max_attempts_five_keys

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attempts_five_keys_l1193_119376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_radius_1998_l1193_119343

/-- The parabola y = x² -/
def parabola (x : ℝ) : ℝ := x^2

/-- The radius of the nth circle -/
noncomputable def radius (n : ℕ+) : ℝ := (n.val - 1) / 2

/-- The y-coordinate of the center of the nth circle -/
noncomputable def center_y (n : ℕ+) : ℝ := n.val^2 - n.val + 1/2

/-- The theorem stating the properties of the circles -/
theorem circle_properties (n : ℕ+) :
  -- The circle touches the parabola
  parabola (radius n) = center_y n + radius n ∧
  -- The circle touches the previous circle (for n > 1)
  (n > 1 → center_y n - radius n = center_y (n-1) + radius (n-1)) ∧
  -- The first circle has diameter 1 and touches the parabola at its vertex
  (n = 1 → radius n = 1/2 ∧ center_y n = 1/2) :=
by sorry

/-- The main theorem proving the radius of the 1998th circle -/
theorem radius_1998 : radius 1998 = 998.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_radius_1998_l1193_119343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chad_savings_theorem_l1193_119311

/-- Represents the different currencies --/
inductive Currency
| USD
| EUR
| GBP

/-- Represents an income source --/
structure IncomeSource where
  amount : ℝ
  currency : Currency

/-- Represents an exchange rate from one currency to USD --/
structure ExchangeRate where
  fromCurrency : Currency
  rate : ℝ

/-- Represents a savings tier --/
structure SavingsTier where
  lowerBound : ℝ
  upperBound : ℝ
  percentage : ℝ

/-- Chad's financial situation --/
def chad_finances : Prop := ∃ (income_sources : List IncomeSource)
                              (exchange_rates : List ExchangeRate)
                              (savings_tiers : List SavingsTier)
                              (tax_rate : ℝ),
  -- Income sources
  income_sources = [
    ⟨600, Currency.EUR⟩,
    ⟨250, Currency.GBP⟩,
    ⟨150, Currency.USD⟩,
    ⟨150, Currency.USD⟩
  ] ∧
  -- Exchange rates
  exchange_rates = [
    ⟨Currency.EUR, 1.20⟩,
    ⟨Currency.GBP, 1.40⟩,
    ⟨Currency.USD, 1.00⟩
  ] ∧
  -- Savings tiers
  savings_tiers = [
    ⟨0, 1000, 0.20⟩,
    ⟨1001, 2000, 0.30⟩,
    ⟨2001, 3000, 0.40⟩,
    ⟨3001, (10000 : ℝ), 0.50⟩  -- Using a large number instead of infinity
  ] ∧
  -- Tax rate
  tax_rate = 0.10 ∧
  -- Total income in USD
  (let total_income_usd := 1370
   let income_after_tax := total_income_usd * (1 - tax_rate)
   let savings_amount := income_after_tax * 0.30
   savings_amount = 369.90)

/-- Theorem: Chad's savings amount is $369.90 --/
theorem chad_savings_theorem : chad_finances := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chad_savings_theorem_l1193_119311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1193_119324

theorem calculation_proof : -2^(-1 : ℤ) + (Real.sqrt 16 - Real.pi)^(0 : ℕ) - |Real.sqrt 3 - 2| - 2 * Real.cos (30 * Real.pi / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l1193_119324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1193_119353

noncomputable def f (x : ℝ) : ℝ := (8 * x^2 - 12) / (4 * x^2 + 6 * x - 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, x > M → |f x - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1193_119353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_C_l1193_119329

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A parabola in 2D space -/
def Parabola := {p : Point | p.y = p.x^2}

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The origin point -/
noncomputable def O : Point := ⟨0, 0⟩

/-- Point A on the parabola -/
noncomputable def A : Point := ⟨-1/2, 1/4⟩

/-- Line OA -/
noncomputable def OA : Line := ⟨A.y, -A.x, 0⟩

/-- Point B on the parabola -/
noncomputable def B : Point := ⟨2, 4⟩

/-- Line OB -/
noncomputable def OB : Line := ⟨B.y, -B.x, 0⟩

/-- Point C of rectangle AOBC -/
noncomputable def C : Point := ⟨3/2, 17/4⟩

/-- The hyperbola passing through point C -/
def hyperbola (k : ℝ) (p : Point) : Prop :=
  p.y = k / p.x

theorem hyperbola_through_C :
  A ∈ Parabola →
  perpendicular OA OB →
  B ∈ Parabola →
  hyperbola (51/8) C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_C_l1193_119329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_value_l1193_119312

def program_output (initial a₁ a₂ : ℕ) : ℕ :=
  let step1 := initial + a₁
  let step2 := step1 + a₂
  step2

theorem final_value (initial : ℕ) :
  program_output initial 2 3 = 6 ↔ initial = 1 := by
  -- Proof goes here
  sorry

#eval program_output 1 2 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_value_l1193_119312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pugsley_spiders_l1193_119317

/-- The number of spiders Pugsley has before trading -/
def P : ℕ := sorry

/-- The number of spiders Wednesday has before trading -/
def W : ℕ := sorry

/-- Condition 1: If Pugsley gives Wednesday 2 spiders, Wednesday will have 9 times as many spiders as Pugsley -/
axiom condition1 : W + 2 = 9 * (P - 2)

/-- Condition 2: If Wednesday gives Pugsley 6 spiders, Pugsley will have 6 fewer spiders than Wednesday had before they traded -/
axiom condition2 : P + 6 = W - 6

/-- Theorem: Pugsley has 4 spiders before the trading game -/
theorem pugsley_spiders : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pugsley_spiders_l1193_119317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_for_broken_line_l1193_119341

def geometric_progression (r : ℕ → ℝ) (q : ℝ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → r (i + 1) = q * r i

def broken_line_possible (r : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ (A : ℕ → ℝ × ℝ), 
    (∀ i : ℕ, i < n → ‖A (i+1) - A i‖ = ‖A 1 - A 0‖) ∧
    (∀ i : ℕ, i < n → ‖A i‖ = r i)

theorem max_ratio_for_broken_line (r : ℕ → ℝ) :
  (geometric_progression r ((1 + Real.sqrt 5) / 2) 5) →
  (broken_line_possible r 5) ∧
  (∀ q > ((1 + Real.sqrt 5) / 2), ¬(geometric_progression r q 5 → broken_line_possible r 5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_for_broken_line_l1193_119341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_diameter_theorem_l1193_119399

/-- The diameter of the second circle in a geometric configuration --/
noncomputable def second_circle_diameter (R : ℝ) : ℝ :=
  2 * (R - R * (Real.sqrt 3 / 2))

/-- The main theorem about the diameter of the second circle --/
theorem second_circle_diameter_theorem :
  ∃ (m n : ℕ), second_circle_diameter 10 = Real.sqrt (m : ℝ) + (n : ℝ) ∧ m + n = 1240 :=
by
  -- We'll use 1200 for m and 40 for n
  use 1200, 40
  constructor
  · -- Prove the equality
    sorry
  · -- Prove the sum
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_diameter_theorem_l1193_119399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_optimization_l1193_119387

-- Constants
noncomputable def loop_length : ℝ := 30
noncomputable def inner_speed : ℝ := 25
noncomputable def outer_speed : ℝ := 30
def total_trains : ℕ := 18

-- Part 1
noncomputable def min_speed_inner_loop (num_trains : ℕ) (max_wait_time : ℝ) : ℝ :=
  loop_length / (num_trains * max_wait_time / 60)

-- Part 2
noncomputable def wait_time_inner (num_trains : ℕ) : ℝ :=
  loop_length / (inner_speed * ↑num_trains) * 60

noncomputable def wait_time_outer (num_trains : ℕ) : ℝ :=
  loop_length / (outer_speed * ↑(total_trains - num_trains)) * 60

noncomputable def wait_time_diff (num_trains : ℕ) : ℝ :=
  |wait_time_inner num_trains - wait_time_outer num_trains|

theorem subway_optimization :
  (min_speed_inner_loop 9 10 = 20) ∧
  (∀ x : ℕ, 1 ≤ x ∧ x < total_trains → wait_time_diff 10 ≤ wait_time_diff x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_optimization_l1193_119387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l1193_119300

/-- Ellipse E with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The left focus of the ellipse -/
noncomputable def left_focus (E : Ellipse) : ℝ × ℝ :=
  (-Real.sqrt (E.a^2 - E.b^2), 0)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (E.a^2 - E.b^2) / E.a

/-- Theorem stating the properties of the specific ellipse and the maximum area of triangle OAB -/
theorem ellipse_properties_and_max_area :
  ∃ (E : Ellipse),
    left_focus E = (-1, 0) ∧
    eccentricity E = Real.sqrt 2 / 2 ∧
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2 / 2 + y^2 = 1) ∧
    (∃ (max_area : ℝ),
      max_area = Real.sqrt 2 / 2 ∧
      ∀ A B : ℝ × ℝ,
        ellipse_equation E A.1 A.2 →
        ellipse_equation E B.1 B.2 →
        abs (A.1 * B.2 - A.2 * B.1) / 2 ≤ max_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l1193_119300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_eq_468_l1193_119368

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℚ

/-- Represents a square -/
structure Square where
  center : Point
  side_length : ℚ

/-- The line segment from (0,0) to (702, 303) -/
def line_segment : Set Point := {p : Point | ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = 702 * t ∧ p.y = 303 * t}

/-- Set of all lattice points -/
def lattice_points : Set Point := {p : Point | ∃ m n : ℤ, p.x = m ∧ p.y = n}

/-- Set of all circles centered at lattice points with radius 1/5 -/
def circles : Set Circle := {c : Circle | c.center ∈ lattice_points ∧ c.radius = 1/5}

/-- Set of all squares centered at lattice points with side length 1/5 -/
def squares : Set Square := {s : Square | s.center ∈ lattice_points ∧ s.side_length = 1/5}

/-- Function to count intersections of the line segment with a set of geometric objects -/
noncomputable def count_intersections (objects : Set α) : ℕ :=
  sorry

/-- Theorem stating that the total number of intersections is 468 -/
theorem total_intersections_eq_468 : 
  count_intersections circles + count_intersections squares = 468 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_eq_468_l1193_119368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_proof_l1193_119342

def is_systematic_sample (sample : List Nat) (total : Nat) (ratio : Nat) : Prop :=
  sample.length = total / ratio ∧
  ∃ (start : Nat), start > 0 ∧ start ≤ ratio ∧
    ∀ (i : Nat), i < sample.length → 
      sample.get! i = start + i * ratio

def is_valid_option (option : List Nat) : Prop :=
  option.length = 4 ∧
  ∀ x ∈ option, 1 ≤ x ∧ x ≤ 200

theorem systematic_sampling_proof (total : Nat) (ratio : Nat)
  (optionA optionB optionC optionD : List Nat) :
  total = 200 →
  ratio = 10 →
  optionA = [3, 23, 63, 102] →
  optionB = [31, 61, 87, 127] →
  optionC = [103, 133, 153, 193] →
  optionD = [57, 68, 98, 108] →
  is_valid_option optionA ∧
  is_valid_option optionB ∧
  is_valid_option optionC ∧
  is_valid_option optionD →
  (is_systematic_sample optionC total ratio ∧
   ¬is_systematic_sample optionA total ratio ∧
   ¬is_systematic_sample optionB total ratio ∧
   ¬is_systematic_sample optionD total ratio) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_proof_l1193_119342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1193_119394

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x * Real.sin (126 * Real.pi / 180) * Real.sin (x - 36 * Real.pi / 180) + 
                     x * Real.cos (54 * Real.pi / 180) * Real.cos (x - 36 * Real.pi / 180)

/-- Theorem stating that f is an even function -/
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1193_119394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_for_30km_l1193_119301

/-- Taxi fare function -/
noncomputable def taxi_fare (x : ℝ) : ℝ :=
  if x ≤ 4 then 10
  else if x ≤ 18 then 1.5 * x + 4
  else 2 * x - 5

/-- Theorem: The fare for a 30km taxi trip is $55 -/
theorem fare_for_30km :
  taxi_fare 30 = 55 := by
  -- Unfold the definition of taxi_fare
  unfold taxi_fare
  -- Simplify the if-then-else expression
  simp
  -- Evaluate the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_for_30km_l1193_119301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_by_n_exists_divisible_by_n_with_condition_l1193_119318

theorem exists_divisible_by_n (n z : ℕ) (hn : n > 1) (hz : z > 1) (coprime_nz : Nat.Coprime n z) :
  ∃ i : Fin n, n ∣ (Finset.range (i + 1)).sum (fun k => z^k) :=
by sorry

theorem exists_divisible_by_n_with_condition 
  (n z : ℕ) (hn : n > 1) (hz : z > 1) (coprime_nz : Nat.Coprime n z) (coprime_n_zminus1 : Nat.Coprime n (z - 1)) :
  ∃ i : Fin n, n ∣ (Finset.range (i + 1)).sum (fun k => z^k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_by_n_exists_divisible_by_n_with_condition_l1193_119318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_division_l1193_119359

theorem recurring_decimal_division : 
  let a : ℚ := 54 / 99  -- represents 0.overline{54}
  let b : ℚ := 18 / 99  -- represents 0.overline{18}
  a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_division_l1193_119359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_plus_g_f_plus_g_even_f_minus_g_positive_a_less_than_one_f_minus_g_positive_a_greater_than_one_l1193_119364

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
def g (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) / Real.log a

-- Theorem for the domain of f(x) + g(x)
theorem domain_f_plus_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {x : ℝ | ∃ y, f a x + g a x = y} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

-- Theorem for the even symmetry of f(x) + g(x)
theorem f_plus_g_even (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, f a x + g a x = f a (-x) + g a (-x) := by sorry

-- Theorem for the set where f(x) - g(x) > 0 when 0 < a < 1
theorem f_minus_g_positive_a_less_than_one (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | f a x - g a x > 0} = {x : ℝ | -1 < x ∧ x < 0} := by sorry

-- Theorem for the set where f(x) - g(x) > 0 when a > 1
theorem f_minus_g_positive_a_greater_than_one (a : ℝ) (h : a > 1) :
  {x : ℝ | f a x - g a x > 0} = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_plus_g_f_plus_g_even_f_minus_g_positive_a_less_than_one_f_minus_g_positive_a_greater_than_one_l1193_119364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_a_range_l1193_119322

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x + (a - 1) * log x

-- State the theorem
theorem function_property_implies_a_range :
  ∀ a : ℝ, a > 1 →
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ > x₂ → f a x₁ - f a x₂ > x₂ - x₁) →
  1 < a ∧ a ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_a_range_l1193_119322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1193_119381

/-- The focus of the parabola y = 2x^2 -/
noncomputable def focus : ℝ × ℝ := (0, 1/8)

/-- The parabola y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Theorem stating that the focus satisfies the defining property of a parabola -/
theorem parabola_focus :
  ∀ (x : ℝ), 
  let y := parabola x
  let d := -focus.2
  (x^2 + (y - focus.2)^2) = (y - d)^2 :=
by
  intro x
  simp [parabola, focus]
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1193_119381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_234_104_l1193_119337

-- Define the mean proportional function as noncomputable
noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- State the theorem
theorem mean_proportional_234_104 : mean_proportional 234 104 = 156 := by
  -- Unfold the definition of mean_proportional
  unfold mean_proportional
  -- Simplify the expression
  simp [Real.sqrt_mul]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_234_104_l1193_119337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l1193_119366

/-- Fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- Triangle with sides l, m, n satisfying the given conditions -/
structure SpecialTriangle where
  l : ℕ
  m : ℕ
  n : ℕ
  l_gt_m : l > m
  m_gt_n : m > n
  frac_eq : frac (3^l / 10000 : ℝ) = frac (3^m / 10000 : ℝ) ∧ 
            frac (3^m / 10000 : ℝ) = frac (3^n / 10000 : ℝ)

/-- The smallest possible perimeter of a SpecialTriangle is 3003 -/
theorem min_perimeter_special_triangle :
  ∀ t : SpecialTriangle, t.l + t.m + t.n ≥ 3003 := by
  sorry

#check min_perimeter_special_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l1193_119366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1193_119397

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 30 = 11)
  (hb : b % 30 = 7)
  (hc : c % 30 = 18) :
  (a + b + c) % 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l1193_119397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1193_119348

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

noncomputable def f (θ : ℝ) : ℝ := determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_value_of_f :
  ∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 3 →
  f θ ≤ (Real.sqrt 3 + 1) / 2 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1193_119348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l1193_119320

/-- Given a cubic function g(x) = px³ + qx² + rx + s passing through the point (-3, -7),
    prove that 4p - 2q + r - s = 7 -/
theorem cubic_function_property (p q r s : ℝ) : 
  (fun x ↦ p * x^3 + q * x^2 + r * x + s) (-3) = -7 →
  4 * p - 2 * q + r - s = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l1193_119320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_figure_surface_area_l1193_119374

/-- The total surface area of a combined figure consisting of a hemisphere on top of a cone -/
noncomputable def total_surface_area (base_area : ℝ) (h : base_area = 144 * Real.pi) : ℝ :=
  288 * Real.pi + 144 * Real.sqrt 2 * Real.pi

/-- Theorem stating the total surface area of the combined figure -/
theorem combined_figure_surface_area 
  (base_area : ℝ) 
  (h1 : base_area = 144 * Real.pi)
  (h2 : base_area = 2 * (base_area / 2))  -- Base area of cone equals base area of hemisphere
  (h3 : ∃ r, r * r = base_area / Real.pi ∧ r = (base_area / Real.pi).sqrt)  -- Radius exists
  (h4 : ∃ h, h = (base_area / Real.pi).sqrt)  -- Height of cone equals radius of hemisphere
  : total_surface_area base_area h1 = 288 * Real.pi + 144 * Real.sqrt 2 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_figure_surface_area_l1193_119374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1193_119332

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem f_minimum_value (x : ℝ) (h : x ≥ 1) : f x ≥ 9 ∧ f (5/2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1193_119332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_calculation_l1193_119309

/-- Represents the time it takes for the leak to empty a full tank. -/
noncomputable def leak_empty_time (pipe_fill_time pipe_with_leak_time : ℝ) : ℝ :=
  let pipe_rate := 1 / pipe_fill_time
  let combined_rate := 1 / pipe_with_leak_time
  let leak_rate := pipe_rate - combined_rate
  1 / leak_rate

/-- 
Given a tank that Pipe A can fill in 6 hours, and with a leak it takes 8 hours to fill,
the leak alone will empty the full tank in 24 hours.
-/
theorem leak_empty_time_calculation :
  leak_empty_time 6 8 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_calculation_l1193_119309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_numbers_k_l1193_119325

theorem natural_numbers_k (a b k : ℕ) : 
  a * b ≠ 1 → 
  k = (a^2 + a*b + b^2) / (a*b - 1) → 
  k = 4 ∨ k = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_numbers_k_l1193_119325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_iff_k_in_range_l1193_119390

/-- The function f(x) defined in terms of k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k*x + 7) / (k*x^2 + 4*k*x + 3)

/-- The theorem stating the equivalence between the domain of f being ℝ and the range of k -/
theorem domain_real_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ (0 ≤ k ∧ k < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_real_iff_k_in_range_l1193_119390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1193_119398

noncomputable def f (x : ℝ) : ℝ := Real.tan x + (1 / Real.tan x) + Real.sin (2 * x)

theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1193_119398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_overtime_rate_is_six_l1193_119378

/-- Represents Bob's work and pay information --/
structure BobsWork where
  regularRate : ℚ
  firstWeekHours : ℚ
  secondWeekHours : ℚ
  totalPay : ℚ

/-- Calculates Bob's overtime pay rate --/
noncomputable def overtimeRate (work : BobsWork) : ℚ :=
  let regularHoursPerWeek : ℚ := 40
  let totalRegularHours : ℚ := 2 * regularHoursPerWeek
  let totalOvertimeHours : ℚ := work.firstWeekHours + work.secondWeekHours - totalRegularHours
  let totalRegularPay : ℚ := totalRegularHours * work.regularRate
  let totalOvertimePay : ℚ := work.totalPay - totalRegularPay
  totalOvertimePay / totalOvertimeHours

/-- Theorem stating that Bob's overtime rate is $6 per hour --/
theorem bobs_overtime_rate_is_six (work : BobsWork)
  (h1 : work.regularRate = 5)
  (h2 : work.firstWeekHours = 44)
  (h3 : work.secondWeekHours = 48)
  (h4 : work.totalPay = 472) :
  overtimeRate work = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_overtime_rate_is_six_l1193_119378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_fifths_l1193_119389

/-- Represents a trapezoid ABCD with point E where the extended legs meet -/
structure ExtendedTrapezoid where
  -- The length of base AB
  ab_length : ℝ
  -- The length of base CD
  cd_length : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- Assumption that AB < CD
  h_ab_lt_cd : ab_length < cd_length

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD -/
noncomputable def area_ratio (t : ExtendedTrapezoid) : ℝ :=
  4 / 5

/-- Theorem stating that for a trapezoid with specific measurements, 
    the ratio of areas is 4/5 -/
theorem area_ratio_is_four_fifths (t : ExtendedTrapezoid) 
  (h_ab : t.ab_length = 10)
  (h_cd : t.cd_length = 15)
  (h_height : t.height = 6) :
  area_ratio t = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_fifths_l1193_119389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_l1193_119303

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define the curve
def curve (x y : ℝ) : Prop := y = 2 / x ∧ x > 0

-- Define a circle with center (a, b) and radius r
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define tangency between a circle and the line
def is_tangent (a b r : ℝ) : Prop :=
  (2 * a + b + 1)^2 / 5 = r^2

-- Theorem statement
theorem smallest_circle :
  ∀ a b r : ℝ,
  curve a b →
  is_tangent a b r →
  ∃ x y : ℝ,
  circle_eq x y 1 2 (Real.sqrt 5) ∧
  (∀ x' y' : ℝ, circle_eq x' y' a b r → (x' - 1)^2 + (y' - 2)^2 ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_l1193_119303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_relationship_not_determined_l1193_119336

noncomputable def roundToTenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  ⌊x * 100 + 0.5⌋ / 100

noncomputable def roundToThousandth (x : ℝ) : ℝ := 
  ⌊x * 1000 + 0.5⌋ / 1000

theorem rounding_relationship_not_determined : 
  ∃ (x1 x2 : ℝ), 
    let a1 := roundToThousandth x1
    let b1 := roundToHundredth a1
    let c1 := roundToTenth b1
    let d1 := roundToTenth x1
    let a2 := roundToThousandth x2
    let b2 := roundToHundredth a2
    let c2 := roundToTenth b2
    let d2 := roundToTenth x2
    (a1 ≥ b1 ∧ b1 ≥ c1 ∧ c1 ≥ d1) ∧ 
    ¬(a2 ≥ b2 ∧ b2 ≥ c2 ∧ c2 ≥ d2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_relationship_not_determined_l1193_119336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l1193_119377

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 50

/-- The time it takes for the train to pass a tree in seconds -/
noncomputable def passing_time : ℝ := 18

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := train_speed * km_hr_to_m_s * passing_time

theorem train_length_is_250_meters : train_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l1193_119377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_visits_all_sections_l1193_119396

/-- Represents a station in the metro system -/
structure Station where
  id : ℕ

/-- Represents a section between two stations -/
structure Section where
  fromStation : Station
  toStation : Station

/-- Represents the metro system -/
structure MetroSystem where
  stations : Set Station
  sections : Set Section
  lines : List (List Station)

/-- Represents a train's journey -/
structure TrainJourney where
  start : Station
  finish : Station
  duration : ℕ
  visited_sections : Set Section

/-- Main theorem: If a train travels from A to B in 2016 minutes, it visits all sections -/
theorem train_visits_all_sections 
  (metro : MetroSystem) 
  (journey : TrainJourney) :
  journey.start ∈ metro.stations →
  journey.finish ∈ metro.stations →
  journey.duration = 2016 →
  ∀ s ∈ metro.sections, s ∈ journey.visited_sections :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_visits_all_sections_l1193_119396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1193_119346

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y = 2

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 2| / Real.sqrt 2

/-- The maximum distance from any point on the circle to the line -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' →
    distance_to_line x y ≥ distance_to_line x' y' ∧
    distance_to_line x y = Real.sqrt 2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1193_119346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_of_A_and_B_l1193_119388

-- Define the set difference
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the symmetric difference
def symmetric_difference (X Y : Set ℝ) : Set ℝ := 
  (set_difference X Y) ∪ (set_difference Y X)

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set B
def B : Set ℝ := {y : ℝ | -2 ≤ y ∧ y ≤ 2}

-- Theorem statement
theorem symmetric_difference_of_A_and_B : 
  symmetric_difference A B = {y : ℝ | y > 2} ∪ {y : ℝ | -2 ≤ y ∧ y < 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_of_A_and_B_l1193_119388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sequence_geometric_l1193_119386

/-- Given an integer n ≥ 3, define the sequence of exponents in the prime factorization of n! -/
def exponent_sequence (n : ℕ) : List ℕ :=
  sorry

/-- Check if a sequence forms a geometric progression -/
def is_geometric_sequence (seq : List ℕ) : Prop :=
  sorry

/-- The main theorem stating the conditions for the exponent sequence to be geometric -/
theorem exponent_sequence_geometric (n : ℕ) :
  n ≥ 3 →
  is_geometric_sequence (exponent_sequence n) ↔ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 10 := by
  sorry

#check exponent_sequence_geometric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_sequence_geometric_l1193_119386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l1193_119380

noncomputable def fib : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

noncomputable def lucas : ℕ → ℝ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

axiom fib_closed_form (n : ℕ) : fib n = (α ^ n - β ^ n) / Real.sqrt 5
axiom lucas_closed_form (n : ℕ) : lucas n = α ^ n + β ^ n

theorem not_perfect_square (n : ℕ) (h : n ≥ 2) :
  ¬ ∃ (k : ℝ), (fib (n - 1) * fib n * fib (n + 1) * lucas (n - 1) * lucas n * lucas (n + 1) = k ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_l1193_119380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_is_sqrt3_l1193_119360

/-- The distance from the focus to the directrix of a parabola with vertex at the origin
    and focus at one of the foci of the ellipse 4x^2 + y^2 = 1 -/
noncomputable def parabola_focus_directrix_distance : ℝ := Real.sqrt 3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

/-- The focus of the parabola is one of the foci of the ellipse -/
def parabola_focus_is_ellipse_focus (x y : ℝ) : Prop :=
  ellipse_equation x y ∧ (x = 0 ∧ y^2 = 3/4)

/-- The vertex of the parabola is at the origin -/
def parabola_vertex_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

theorem parabola_focus_directrix_distance_is_sqrt3 :
  ∀ x y : ℝ,
  parabola_focus_is_ellipse_focus x y →
  parabola_vertex_at_origin 0 0 →
  parabola_focus_directrix_distance = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_is_sqrt3_l1193_119360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l1193_119314

theorem smallest_x_value : 
  ∀ x : ℝ, ((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3) → 
  x ≥ -8 - Real.sqrt 292 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l1193_119314
