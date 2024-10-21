import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fold_perimeter_l280_28069

/-- A square with side length 1 and vertices A, B, C, D (in that order) is given.
    C' is a point on AD such that C'D = 1/4.
    E is the intersection point of BC and AB. -/
def square_fold (A B C D C' E : ℝ × ℝ) : Prop :=
  A = (0, 1) ∧ B = (0, 0) ∧ C = (1, 0) ∧ D = (1, 1) ∧
  C' = (1, 1/4) ∧ 
  E.1 = E.2 ∧ -- E is on the line y = x
  E.2 = -3/4 * E.1 + 1 -- E is on the line AC'

/-- The perimeter of triangle AEC' -/
noncomputable def triangle_perimeter (A E C' : ℝ × ℝ) : ℝ :=
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) +
  Real.sqrt ((C'.1 - E.1)^2 + (C'.2 - E.2)^2) +
  Real.sqrt ((A.1 - C'.1)^2 + (A.2 - C'.2)^2)

theorem square_fold_perimeter (A B C D C' E : ℝ × ℝ) :
  square_fold A B C D C' E →
  triangle_perimeter A E C' = (27.5 + 3 * Real.sqrt 5) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_fold_perimeter_l280_28069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l280_28042

noncomputable def f₀ (x : ℝ) : ℝ := x * Real.sin x

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => f₀ x
  | n + 1 => deriv (fun x => f n x) x

theorem f_formula (n : ℕ) (x : ℝ) (h : n > 0) :
  f n x = n * Real.sin (x + (n - 1) * Real.pi / 2) + x * Real.cos (x + (n - 1) * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l280_28042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l280_28048

/-- The universal set U -/
def U (a : ℝ) : Set ℝ := {2, 4, 1 - a}

/-- The set A -/
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

/-- The complement of A in U -/
def C_U_A : Set ℝ := {-1}

/-- Theorem: Given the universal set U, set A, and complement C_U_A as defined above, 
    the value of a is 2. -/
theorem find_a : ∃ a : ℝ, (U a = {2, 4, 1 - a} ∧ A a = {2, a^2 - a + 2} ∧ C_U_A = {-1}) ∧ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l280_28048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_floor_l280_28004

-- Define a custom probability measure type
def ProbMeasure (α : Type) := α → ℝ

-- Define the probability function
noncomputable def ℙ {α : Type} (p : ProbMeasure α) (e : α → Prop) : ℝ := sorry

theorem probability_sqrt_floor :
  ∃ (x : ℝ) (p : ProbMeasure ℝ),
  200 ≤ x ∧ x ≤ 400 →
  ⌊Real.sqrt x⌋ = 18 →
  ℙ p (λ y => ⌊Real.sqrt (100 * y)⌋ = 180) = 361 / 3700 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sqrt_floor_l280_28004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheryls_project_l280_28053

theorem cheryls_project (material1 material2 material3 leftover discount : ℚ)
  (h1 : material1 = 5 / 11)
  (h2 : material2 = 2 / 3)
  (h3 : material3 = 7 / 15)
  (h4 : leftover = 25 / 55)
  (h5 : discount = 15 / 100) :
  (material1 + material2 + material3 - leftover = 187 / 165) ∧
  (1 - discount = 85 / 100) := by
  sorry

#eval (5/11 : ℚ) + (2/3 : ℚ) + (7/15 : ℚ) - (25/55 : ℚ)
#eval (1 : ℚ) - (15/100 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheryls_project_l280_28053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_has_smallest_hypotenuse_l280_28016

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

-- Define the area of a right-angled triangle
noncomputable def area (t : RightTriangle) : ℝ := t.a * t.b / 2

-- Define the hypotenuse of a right-angled triangle
noncomputable def hypotenuse (t : RightTriangle) : ℝ := Real.sqrt (t.a^2 + t.b^2)

-- Theorem statement
theorem isosceles_has_smallest_hypotenuse
  (t : RightTriangle)
  (fixed_area : ℝ)
  (h_area : area t = fixed_area) :
  hypotenuse t ≥ Real.sqrt (2 * fixed_area)
  ∧ (hypotenuse t = Real.sqrt (2 * fixed_area) ↔ t.a = t.b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_has_smallest_hypotenuse_l280_28016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l280_28009

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

def SumDistances (a : ℝ) : ℝ := 2 * a

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : Eccentricity a b = Real.sqrt 2 / 2)
  (h4 : SumDistances a = 2 * Real.sqrt 2) :
  -- 1) Standard equation
  ∃ e : Set (ℝ × ℝ),
    (∀ p : ℝ × ℝ, p ∈ e ↔ p.2^2 / 2 + p.1^2 = 1) ∧
  -- 2) Range of m
  (∀ k : ℝ, k ≠ 0 →
    ∃ m : ℝ, 0 < m ∧ m < 1/2) ∧
  -- 3) Area formula
  (∀ m : ℝ, 0 < m → m < 1/2 →
    ∃ S : ℝ, S = Real.sqrt 2 * Real.sqrt (m * (1 - m)^3)) ∧
  -- 4) Maximum area
  ∃ S_max : ℝ, S_max = 3 * Real.sqrt 6 / 16 ∧
    ∀ m : ℝ, 0 < m → m < 1/2 →
      Real.sqrt 2 * Real.sqrt (m * (1 - m)^3) ≤ S_max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l280_28009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_y_l280_28017

theorem no_integer_solution_for_y (x y : ℝ) : 
  x - (3/8) * x = 25 ∧ x^y = 125 → 
  (x = 40 ∧ ∀ n : ℕ, (40 : ℝ)^n ≠ 125) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_y_l280_28017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_sharing_l280_28010

theorem cake_sharing (n : ℕ) : 
  (∃ (portions : Fin n → ℚ), 
    (∀ i, 0 < portions i) ∧ 
    (∃ i, portions i = 1/11) ∧ 
    (∃ i, portions i = 1/14) ∧ 
    (∀ i, portions i ≤ 1/11) ∧ 
    (∀ i, 1/14 ≤ portions i) ∧ 
    (Finset.sum Finset.univ portions) = 1) ↔ 
  (n = 12 ∨ n = 13) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_sharing_l280_28010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_2006_l280_28000

theorem alternating_sum_to_2006 : 
  (Finset.range 1003).sum (fun i => (2*i + 1 : ℤ) - (2*i + 2)) = -1003 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_to_2006_l280_28000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l280_28056

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors
noncomputable def m (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos t.A
  | 1 => t.b
  | _ => 0

noncomputable def n (t : Triangle) : Fin 2 → Real
  | 0 => Real.sin t.A
  | 1 => t.a
  | _ => 0

-- Define the area function
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.a * t.b * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : ∃ (k : Real), ∀ (i : Fin 2), m t i = k * n t i) -- vectors are collinear
  (h2 : t.B > Real.pi / 2) -- angle B is obtuse
  : 
  (t.B - t.A = Real.pi / 2) ∧ 
  (t.b = 2 * Real.sqrt 3 ∧ t.a = 2 → area t = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l280_28056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_is_1_strictly_increasing_condition_l280_28014

-- Define the function f(x, a)
noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 - 2*a*x

-- Part I: Extreme values when a = 1
theorem extreme_values_when_a_is_1 :
  (∃ x : ℝ, f x 1 = 7/6 ∧ ∀ y : ℝ, f y 1 ≤ f x 1) ∧
  (∃ x : ℝ, f x 1 = -10/3 ∧ ∀ y : ℝ, f y 1 ≥ f x 1) := by
  sorry

-- Part II: Range of a for which f(x) is strictly increasing on (2/3, +∞)
theorem strictly_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, 2/3 < x → x < y → f x a < f y a) ↔ a ≤ -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_when_a_is_1_strictly_increasing_condition_l280_28014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l280_28059

-- Define the circle and ellipse equations
def my_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def my_ellipse (x y : ℝ) : Prop := 9 * x^2 + (y + 1)^2 = 9

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | my_circle p.1 p.2 ∧ my_ellipse p.1 p.2}

-- Define an equilateral triangle
def is_equilateral_triangle (points : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ, points = {a, b, c} ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 =
    (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 =
    (c.1 - a.1)^2 + (c.2 - a.2)^2

-- Theorem statement
theorem intersection_forms_equilateral_triangle :
  is_equilateral_triangle intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_equilateral_triangle_l280_28059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_factorial_difference_tens_digit_zero_tens_digit_of_25_factorial_minus_20_factorial_l280_28065

open BigOperators

theorem tens_digit_of_factorial_difference : ∃ k : ℕ, (Nat.factorial 25) - (Nat.factorial 20) = 10000 * k := by
  sorry

theorem tens_digit_zero (n : ℕ) (h : ∃ k : ℕ, n = 10000 * k) : n % 100 / 10 = 0 := by
  sorry

theorem tens_digit_of_25_factorial_minus_20_factorial : ((Nat.factorial 25) - (Nat.factorial 20)) % 100 / 10 = 0 := by
  have h1 := tens_digit_of_factorial_difference
  exact tens_digit_zero ((Nat.factorial 25) - (Nat.factorial 20)) h1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_factorial_difference_tens_digit_zero_tens_digit_of_25_factorial_minus_20_factorial_l280_28065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l280_28084

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

def vertical_asymptote : ℝ := 3

def horizontal_asymptote : ℝ := 1

def intersection_point : ℝ × ℝ := (vertical_asymptote, horizontal_asymptote)

theorem asymptotes_intersection :
  let (x, y) := intersection_point
  ∀ ε > 0, ∃ δ > 0, ∀ t, t ≠ x → |t - x| < δ → |f t - y| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_intersection_l280_28084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_catches_jerry_l280_28043

/-- The time it takes for Carla to catch up to Jerry -/
noncomputable def catchUpTime (jerrySpeed carlaSpeed : ℝ) (headStart : ℝ) : ℝ :=
  (jerrySpeed * headStart) / (carlaSpeed - jerrySpeed)

/-- Theorem stating that Carla catches up to Jerry in 3 hours -/
theorem carla_catches_jerry :
  catchUpTime 30 35 0.5 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_catches_jerry_l280_28043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_line_l280_28076

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  m : ℝ
  b : ℝ

/-- The area of a triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.b) * abs (l.b / l.m) / 2

/-- The condition that a line passes through a given point -/
def passesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

/-- The condition that a line is in the second quadrant -/
def inSecondQuadrant (l : Line) : Prop :=
  l.m < 0 ∧ l.b > 0

theorem smallest_area_line :
  ∃ (l : Line),
    passesThrough l (Point.mk (-2) 2) ∧
    inSecondQuadrant l ∧
    (∀ (l' : Line),
      passesThrough l' (Point.mk (-2) 2) →
      inSecondQuadrant l' →
      triangleArea l ≤ triangleArea l') ∧
    l.m = -1 ∧
    l.b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_line_l280_28076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l280_28087

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (f : ℝ → ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (f = λ x ↦ sin (2 * x + A)) →
  (∀ x, f x = sin (2 * x + A)) →
  (A = π / 2 → f (-π / 6) = 1 / 2) ∧
  (f (π / 12) = 1 → a = 3 → cos B = 4 / 5 → b = 6 * Real.sqrt 3 / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l280_28087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_is_criminal_l280_28099

-- Define the suspects
inductive Suspect : Type
  | A | B | C | D

-- Define a function to represent whether a suspect is telling the truth
axiom isTellingTruth : Suspect → Prop

-- Define a function to represent whether a suspect is the criminal
axiom isCriminal : Suspect → Prop

-- Axioms based on the problem conditions
axiom only_one_criminal : ∃! s : Suspect, isCriminal s

axiom two_truth_two_lie :
  (isTellingTruth Suspect.A ∧ isTellingTruth Suspect.B ∧ ¬isTellingTruth Suspect.C ∧ ¬isTellingTruth Suspect.D) ∨
  (isTellingTruth Suspect.A ∧ isTellingTruth Suspect.C ∧ ¬isTellingTruth Suspect.B ∧ ¬isTellingTruth Suspect.D) ∨
  (isTellingTruth Suspect.A ∧ isTellingTruth Suspect.D ∧ ¬isTellingTruth Suspect.B ∧ ¬isTellingTruth Suspect.C) ∨
  (isTellingTruth Suspect.B ∧ isTellingTruth Suspect.C ∧ ¬isTellingTruth Suspect.A ∧ ¬isTellingTruth Suspect.D) ∨
  (isTellingTruth Suspect.B ∧ isTellingTruth Suspect.D ∧ ¬isTellingTruth Suspect.A ∧ ¬isTellingTruth Suspect.C) ∨
  (isTellingTruth Suspect.C ∧ isTellingTruth Suspect.D ∧ ¬isTellingTruth Suspect.A ∧ ¬isTellingTruth Suspect.B)

axiom A_statement : isTellingTruth Suspect.A ↔ (isCriminal Suspect.B ∨ isCriminal Suspect.C ∨ isCriminal Suspect.D)

axiom B_statement : isTellingTruth Suspect.B ↔ (¬isCriminal Suspect.B ∧ isCriminal Suspect.C)

axiom C_statement : isTellingTruth Suspect.C ↔ (isCriminal Suspect.A ∨ isCriminal Suspect.B)

axiom D_statement : isTellingTruth Suspect.D ↔ isTellingTruth Suspect.B

-- Theorem to prove
theorem B_is_criminal : isCriminal Suspect.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_is_criminal_l280_28099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_stamp_collection_l280_28063

def divisors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

def proper_divisors (n : ℕ) : Finset ℕ := (divisors n).filter (λ d => d ≠ 1 ∧ d ≠ n)

theorem smallest_stamp_collection :
  ∃ (n : ℕ), n > 2 ∧
    (divisors n).card = 9 ∧
    (proper_divisors n).card = 7 ∧
    ∀ m : ℕ, m > 2 → (divisors m).card = 9 → (proper_divisors m).card = 7 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_stamp_collection_l280_28063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l280_28001

/-- The circle C with equation x^2 + y^2 - 2x + 2√2y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*Real.sqrt 2*y = 0

/-- The center of circle C -/
noncomputable def center_C : ℝ × ℝ := (1, -Real.sqrt 2)

/-- The parabola with equation y^2 = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (∀ x y, parabola x y → (x = 0 ∧ y = 0) → True) ∧  -- vertex at origin
  (parabola (center_C.1) (center_C.2)) ∧             -- passes through center of C
  (∀ x y, parabola x y → x ≥ 0) :=                   -- axis perpendicular to x-axis
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l280_28001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_example_l280_28055

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop := sorry

/-- Adjacent interior angles are complementary -/
def complementary_angles (a b : Real) : Prop := a + b = Real.pi / 2

/-- ∠A and ∠B are adjacent interior angles of two parallel lines -/
def adjacent_interior_angles (A B : Real) (l1 l2 : Line) : Prop :=
  parallel_lines l1 l2 ∧ complementary_angles A B

theorem deductive_reasoning_example 
  (l1 l2 : Line) (A B : Real) 
  (h : adjacent_interior_angles A B l1 l2) : 
  A + B = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_example_l280_28055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l280_28040

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp 1 * (a - 2)

-- State the theorem
theorem f_positive_range (a : ℝ) :
  (∀ x > 1, f a x > 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l280_28040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_transformation_matrix_l280_28080

/-- Transformation matrix for dilation followed by translation -/
def transformation_matrix (scale : ℝ) (translation_x : ℝ) (translation_y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![scale, 0, translation_x],
    ![0, scale, translation_y],
    ![0, 0, 1]]

/-- Theorem stating the correct transformation matrix -/
theorem correct_transformation_matrix :
  transformation_matrix 4 2 3 =
    ![![4, 0, 2],
      ![0, 4, 3],
      ![0, 0, 1]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_transformation_matrix_l280_28080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_threshold_l280_28058

open Real MeasureTheory

noncomputable def series_term (c : ℝ) (n : ℕ) : ℝ := (n.factorial : ℝ) / (c * n : ℝ) ^ n

theorem convergence_threshold (c : ℝ) (h : c > 0) :
  Summable (series_term c) ↔ c > (Real.exp 1)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_threshold_l280_28058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ac_equals_6m_l280_28034

/-- Represents a train station -/
structure Station where
  name : String

/-- Represents a train -/
structure Train where
  name : String

/-- Represents the distance between two stations in kilometers -/
noncomputable def distance (a b : Station) : ℝ := sorry

/-- Represents the time taken to travel between two stations in hours -/
noncomputable def travelTime (t : Train) (a b : Station) : ℝ := sorry

/-- Represents the speed of a train in km/h -/
noncomputable def speed (t : Train) : ℝ := sorry

theorem distance_ac_equals_6m (
  stationA stationB stationC stationD : Station)
  (trainR trainS : Train)
  (M : ℝ)
  (h1 : travelTime trainR stationA stationB = 7)
  (h2 : travelTime trainR stationB stationC = 5)
  (h3 : travelTime trainS stationD stationC = 8)
  (h4 : distance stationA stationD = distance stationA stationB)
  (h5 : distance stationA stationB = distance stationB stationC + M)
  (h6 : speed trainR = speed trainR)  -- constant speed for R train
  (h7 : speed trainS = speed trainS)  -- constant speed for S train
  : distance stationA stationC = 6 * M := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ac_equals_6m_l280_28034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_f_symmetry_l280_28067

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((x - 2) / (x + 2))

-- Define the solution set A
def A (a : ℝ) : Set ℝ := {x | (x - 1)^2 ≤ a^2}

-- Define the domain B of f
def B : Set ℝ := {x | x < -2 ∨ x > 2}

-- Theorem 1
theorem range_of_a (a : ℝ) : 
  a > 0 → (A a ∩ B = ∅) → 0 < a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem f_symmetry (x : ℝ) : 
  x ∈ B → f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_f_symmetry_l280_28067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l280_28088

noncomputable section

variable (f : ℝ → ℝ)

axiom f_condition : ∀ x : ℝ, f x + deriv f x > 1
axiom f_initial : f 0 = 4

theorem solution_set :
  {x : ℝ | Real.exp x * f x > Real.exp x + 3} = {x : ℝ | x > 0} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l280_28088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_coordinates_l280_28025

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the vector relation
def vector_relation (A B : ℝ × ℝ) : Prop :=
  (A.1 - F1.1, A.2 - F1.2) = (5 * (B.1 - F2.1), 5 * (B.2 - F2.2))

theorem ellipse_point_coordinates :
  ∀ A B : ℝ × ℝ,
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  vector_relation A B →
  (A = (0, 1) ∨ A = (0, -1)) := by
  sorry

#check ellipse_point_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_coordinates_l280_28025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_RS_length_l280_28079

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the lengths of the sides
noncomputable def side_length (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the altitude CH
noncomputable def altitude (t : Triangle) : ℝ × ℝ := sorry

-- Define points R and S
noncomputable def R (t : Triangle) : ℝ × ℝ := sorry
noncomputable def S (t : Triangle) : ℝ × ℝ := sorry

-- Main theorem
theorem RS_length (t : Triangle) : 
  side_length t.A t.B = 13 →
  side_length t.A t.C = 12 →
  side_length t.B t.C = 5 →
  side_length (R t) (S t) = 24/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_RS_length_l280_28079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_l280_28035

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 1)
  else x * (x - 1)

theorem f_of_f_neg_one : f (f (-1)) = 6 := by
  -- Prove f(-1) = 2
  have h1 : f (-1) = 2 := by
    simp [f]
    norm_num
  
  -- Prove f(2) = 6
  have h2 : f 2 = 6 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc f (f (-1))
    = f 2 := by rw [h1]
    _ = 6 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_l280_28035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_negative_a_l280_28013

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - (1/2) * a * x^2 - 2 * a * x

-- State the theorem
theorem increasing_function_implies_negative_a :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f a x₁ < f a x₂) →
  a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_negative_a_l280_28013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slips_with_three_count_l280_28041

/-- Represents the number of slips with 3 on them -/
def slips_with_three : ℕ := 10

/-- Total number of slips in the bag -/
def total_slips : ℕ := 15

/-- The value on slips that don't have 3 -/
def other_value : ℕ := 8

/-- The expected value of a randomly drawn slip -/
def expected_value : ℚ := 46/10

/-- Theorem stating that given the conditions, the number of slips with 3 is 10 -/
theorem slips_with_three_count :
  (slips_with_three : ℚ) / total_slips * 3 +
  (total_slips - slips_with_three : ℚ) / total_slips * other_value = expected_value := by
  sorry

#eval slips_with_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slips_with_three_count_l280_28041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l280_28044

theorem tan_inequality : Real.tan (15/8 * Real.pi) > Real.tan (-Real.pi/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l280_28044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l280_28012

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (t.a - t.c) * (Real.sin t.A + Real.sin t.C) = (t.a - t.b) * Real.sin t.B

-- Define the trisection point D
def trisection_point (t : Triangle) (D : Real) : Prop :=
  D = t.a / 3

-- Define the relationship CD = tAD
def cd_relation (t : Triangle) (D : Real) (CD : Real) (t_val : Real) : Prop :=
  CD = t_val * D

-- State the theorem
theorem triangle_theorem (t : Triangle) (D CD t_val : Real) 
  (h1 : given_condition t) 
  (h2 : trisection_point t D) 
  (h3 : cd_relation t D CD t_val) : 
  t.C = Real.pi / 3 ∧ 1 < t_val ∧ t_val ≤ Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l280_28012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l280_28060

/-- Rounds a number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The number to be rounded -/
def number : ℝ := 42.38567

theorem round_to_nearest_tenth_of_number :
  roundToNearestTenth number = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_number_l280_28060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l280_28070

theorem triangle_abc_properties (a b c : ℝ) (A C : ℝ) :
  a = 8 →
  c = 7 →
  Real.cos A = -1/7 →
  b^2 + 14*b - 15 = 0 →
  Real.sin A = 4*Real.sqrt 3/7 →
  Real.sin C = Real.sqrt 3/2 →
  b = 3 ∧ C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l280_28070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_coin_exchange_terminates_l280_28003

/-- Represents a dwarf in the clan -/
structure Dwarf :=
  (id : ℕ)
  (coins : ℕ)

/-- Represents the state of the dwarf clan on a given day -/
structure ClanState :=
  (dwarves : List Dwarf)
  (acquaintances : List (Dwarf × Dwarf))

/-- Represents a single day's coin exchanges -/
def day_exchange (state : ClanState) : ClanState :=
  sorry

/-- The theorem to be proved -/
theorem dwarf_coin_exchange_terminates :
  ∀ (initial_state : ClanState),
  ∃ (n : ℕ), ∀ (m : ℕ),
  m ≥ n → day_exchange^[m] initial_state = day_exchange^[m] initial_state :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_coin_exchange_terminates_l280_28003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_rational_l280_28021

theorem square_of_rational (x : ℤ) : 
  (∃ (n : ℚ), 1 + 105 * (2 : ℚ)^x = n^2) ↔ x ∈ ({3, 4, -4, -6, -8} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_rational_l280_28021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_l280_28015

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x * (x + 2)
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + 2

-- State the theorem
theorem tangent_line_and_minimum (a b : ℝ) (t : ℝ) 
  (h1 : (deriv (f a)) 0 = (deriv (g b)) 0) 
  (h2 : f a 0 = g b 0) 
  (h3 : t > -4) : 
  (∃ (x : ℝ), f 1 x = f a x ∧ g 3 x = g b x) ∧ 
  (∀ x ∈ Set.Icc t (t + 1), 
    (t < -3 → f 1 x ≥ -Real.exp (-3)) ∧
    (t ≥ -3 → f 1 x ≥ f 1 t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_l280_28015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l280_28029

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c d : ℝ) : ℝ :=
  |c - d| / Real.sqrt (a^2 + b^2)

theorem distance_between_given_lines :
  let line1 : ℝ → ℝ → Prop := λ x y ↦ y = x
  let line2 : ℝ → ℝ → Prop := λ x y ↦ x - y + 2 = 0
  distance_between_parallel_lines 1 (-1) 0 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l280_28029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangent_circles_through_point_l280_28093

/-- A circle in the first quadrant that is tangent to both coordinate axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2 ∧ center.1 = radius
  in_first_quadrant : center.1 > 0 ∧ center.2 > 0

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_tangent_circles_through_point
  (C₁ C₂ : TangentCircle)
  (h₁ : distance C₁.center (4, 1) = C₁.radius)
  (h₂ : distance C₂.center (4, 1) = C₂.radius) :
  distance C₁.center C₂.center = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangent_circles_through_point_l280_28093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l280_28089

theorem size_relationship (a b c : ℝ) (ha : a = 0.76) (hb : b = 60.7) (hc : c = Real.log 0.76 / Real.log 10) :
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l280_28089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_2022n_l280_28039

theorem exists_function_2022n : 
  ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f ∘ f ∘ f ∘ f ∘ f) n = 2022 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_2022n_l280_28039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_area_proof_l280_28082

noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ := (b1 + b2) * h / 2

noncomputable def envelope_a_area : ℝ := trapezoid_area 5 7 6
noncomputable def envelope_b_area : ℝ := trapezoid_area 4 6 5

noncomputable def total_area : ℝ := 3 * envelope_a_area + 2 * envelope_b_area

theorem envelope_area_proof : total_area = 158 := by
  -- Unfold definitions
  unfold total_area envelope_a_area envelope_b_area trapezoid_area
  -- Simplify arithmetic expressions
  simp [mul_add, add_mul, mul_assoc, add_assoc]
  -- Perform numerical calculations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_area_proof_l280_28082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_of_all_integers_l280_28071

theorem set_of_all_integers (S : Set ℤ) 
  (h1 : ∃ a b, a ∈ S ∧ b ∈ S ∧ Int.gcd a b = 1 ∧ Int.gcd (a - 2) (b - 2) = 1)
  (h2 : ∀ x y, x ∈ S → y ∈ S → x^2 - y ∈ S) : 
  S = Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_of_all_integers_l280_28071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_difference_l280_28091

/-- The walking speed in miles per hour -/
noncomputable def walking_speed : ℝ := 3

/-- The distance from home to school in miles -/
noncomputable def school_distance : ℝ := 9

/-- The distance Mark walks before turning back in miles -/
noncomputable def mark_initial_distance : ℝ := 3

/-- Time Chris spends walking to school in hours -/
noncomputable def chris_time : ℝ := school_distance / walking_speed

/-- Time Mark spends on initial round trip in hours -/
noncomputable def mark_initial_time : ℝ := (2 * mark_initial_distance) / walking_speed

/-- Time Mark spends walking to school after returning home in hours -/
noncomputable def mark_final_time : ℝ := school_distance / walking_speed

/-- Total time Mark spends walking in hours -/
noncomputable def mark_total_time : ℝ := mark_initial_time + mark_final_time

theorem walking_time_difference : mark_total_time - chris_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_difference_l280_28091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_fraction_l280_28097

-- Define the circular track
structure CircularTrack where
  stations : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, i ≠ j → dist (stations i) (stations j) = dist (stations 0) (stations 1)

-- Define the train system
structure TrainSystem where
  track : CircularTrack
  num_trains : ℕ
  train_positions : Fin num_trains → ℝ × ℝ

-- Define the boarding condition
def boards_earlier (track : CircularTrack) (train_system : TrainSystem) (station : Fin 3) : Prop :=
  ∃ (train : Fin train_system.num_trains), 
    dist (train_system.train_positions train) (track.stations station) < 
    dist (train_system.train_positions train) (track.stations ((station + 1) % 3))

-- Define the forest condition
def forest_condition (track : CircularTrack) (train_system : TrainSystem) : Prop :=
  boards_earlier track train_system 0 ↔ 
    ∃ (train : Fin train_system.num_trains), is_in_forest (train_system.train_positions train)
  where
    is_in_forest : ℝ × ℝ → Prop := sorry  -- Placeholder for the forest region

-- Define the fraction of track in forest
noncomputable def fraction_of_track_in_forest (track : CircularTrack) : ℚ :=
  sorry  -- Placeholder for the actual calculation

-- The main theorem
theorem forest_fraction (track : CircularTrack) (train_system : TrainSystem) :
  forest_condition track train_system →
  (∃ (forest_fraction : ℚ), 
    (forest_fraction = 2/3 ∨ forest_fraction = 1/3) ∧
    forest_fraction = fraction_of_track_in_forest track) :=
by
  sorry  -- Proof to be completed


end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_fraction_l280_28097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_bound_l280_28024

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x) + a * Real.cos (Real.pi / 2 + x)

theorem f_increasing_implies_a_bound (a : ℝ) :
  (∀ x₁ x₂, Real.pi/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi/2 → f a x₁ < f a x₂) →
  a ≤ -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_bound_l280_28024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_correct_locus_Q_is_circle_locus_Q_intersects_circle_O_common_chord_length_l280_28066

noncomputable section

/-- Circle O with equation x^2 + y^2 = 4 -/
def circle_O : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 4}

/-- Point P -/
def point_P : ℝ × ℝ := (2, 1)

/-- Point A on circle O -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B on circle O -/
def point_B : ℝ × ℝ := (0, 2)

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Locus of point Q satisfying QA = √2 * QB -/
def locus_Q : Set (ℝ × ℝ) :=
  {q | distance q point_A = Real.sqrt 2 * distance q point_B}

/-- Tangent lines from P to circle O -/
def tangent_lines : Set (Set (ℝ × ℝ)) :=
  {{p | p.1 = 2}, {p | 3 * p.1 + 4 * p.2 - 10 = 0}}

/-- Definition of a tangent line to a circle -/
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ l ∧ p ∈ c ∧ ∀ q ∈ l, q ≠ p → q ∉ c

theorem tangent_lines_correct :
  ∀ l ∈ tangent_lines, IsTangentLine l circle_O point_P :=
by sorry

theorem locus_Q_is_circle :
  ∃ c r, locus_Q = {p | distance p c = r} :=
by sorry

theorem locus_Q_intersects_circle_O :
  ∃ p q, p ≠ q ∧ p ∈ locus_Q ∩ circle_O ∧ q ∈ locus_Q ∩ circle_O :=
by sorry

theorem common_chord_length :
  ∃ p q, p ≠ q ∧ p ∈ locus_Q ∩ circle_O ∧ q ∈ locus_Q ∩ circle_O ∧
  distance p q = 8 * Real.sqrt 5 / 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_correct_locus_Q_is_circle_locus_Q_intersects_circle_O_common_chord_length_l280_28066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l280_28046

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- A chord of a hyperbola -/
structure Chord (h : Hyperbola a b) where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- The right vertex of a hyperbola -/
def right_vertex (h : Hyperbola a b) : ℝ × ℝ := (a, 0)

/-- Predicate for a chord being perpendicular to the real axis -/
def perpendicular_to_real_axis (c : Chord h) : Prop :=
  c.M.1 = c.N.1

/-- Dot product of vectors AM and AN -/
def dot_product_AM_AN (h : Hyperbola a b) (c : Chord h) : ℝ :=
  let A := right_vertex h
  ((c.M.1 - A.1) * (c.N.1 - A.1) + (c.M.2 - A.2) * (c.N.2 - A.2))

/-- Main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (c : Chord h) (perp : perpendicular_to_real_axis c) 
  (ortho : dot_product_AM_AN h c = 0) : 
  eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l280_28046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perpendicular_to_plane_parallel_l280_28007

-- Define the basic types
class Line : Type
class Plane : Type

-- Define the relations as axioms
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop

-- Theorem 1: Transitivity of parallel lines
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

-- Theorem 2: Lines perpendicular to the same plane are parallel
theorem perpendicular_to_plane_parallel (a b : Line) (y : Plane) :
  perpendicular a y → perpendicular b y → parallel a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_perpendicular_to_plane_parallel_l280_28007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_real_power_one_plus_i_l280_28096

theorem min_n_real_power_one_plus_i :
  (∃ (n : ℕ), n > 0 ∧ ((Complex.I : ℂ) + 1) ^ n ∈ Set.range (Complex.ofReal : ℝ → ℂ)) ∧
  (∀ (n : ℕ), n > 0 ∧ ((Complex.I : ℂ) + 1) ^ n ∈ Set.range (Complex.ofReal : ℝ → ℂ) → n ≥ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_real_power_one_plus_i_l280_28096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_milk_consumption_l280_28002

theorem rachel_milk_consumption : ∃ (initial_milk cat_portion rachel_portion rachel_consumption : ℚ),
  initial_milk = 3/4 ∧
  cat_portion = 1/8 ∧
  rachel_portion = 1/2 ∧
  rachel_consumption = rachel_portion * (initial_milk - cat_portion * initial_milk) ∧
  rachel_consumption = 21/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_milk_consumption_l280_28002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_l280_28019

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x)

-- Define the point of tangency
noncomputable def x₀ : ℝ := 3/4
noncomputable def y₀ : ℝ := 1/2

-- Define the proposed tangent line equation
def tangent_line (x y : ℝ) : Prop := 4*x + 4*y - 5 = 0

-- Theorem statement
theorem tangent_line_correct : 
  (∀ x y, y = f x → (x = x₀ ∧ y = y₀) → tangent_line x y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - (y₀ - (x - x₀))| < ε * |x - x₀|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_l280_28019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_consumption_l280_28083

/-- Represents the number of mangos eaten each day -/
def MangoSequence := Fin 6 → ℕ

/-- The condition that each day's mango consumption is double the previous day -/
def IsDoublingSequence (s : MangoSequence) : Prop :=
  ∀ i : Fin 5, s (i.succ) = 2 * s i

/-- The sum of mangos eaten over 6 days -/
def TotalMangos (s : MangoSequence) : ℕ :=
  (Finset.range 6).sum (λ i => s i)

theorem mango_consumption (s : MangoSequence)
    (doubling : IsDoublingSequence s)
    (total : TotalMangos s = 364) :
    s 5 = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mango_consumption_l280_28083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_m_range_l280_28077

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x + 1/4

noncomputable def g (x : ℝ) : ℝ := -Real.log x

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := min (f m x) (g x)

theorem three_zeros_m_range (m : ℝ) :
  (∀ x, x > 0 → h m x ≥ 0) ∧ 
  (∃ x₁ x₂ x₃, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    h m x₁ = 0 ∧ h m x₂ = 0 ∧ h m x₃ = 0) ∧
  (∀ x₁ x₂ x₃ x₄, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ →
    ¬(h m x₁ = 0 ∧ h m x₂ = 0 ∧ h m x₃ = 0 ∧ h m x₄ = 0)) →
  -5/4 < m ∧ m < -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_m_range_l280_28077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_approx_l280_28098

/-- Represents the score of a participant in the quiz -/
structure Score where
  value : ℝ
  nonneg : 0 ≤ value

/-- The scores of the six participants in the quiz -/
structure QuizScores where
  xander : Score
  tatuya : Score
  ivanna : Score
  dorothy : Score
  olivia : Score
  sam : Score

/-- The conditions of the quiz scores -/
def quiz_conditions (scores : QuizScores) : Prop :=
  scores.tatuya.value = 2 * scores.ivanna.value ∧
  scores.ivanna.value = (3/5) * scores.dorothy.value ∧
  scores.dorothy.value = 90 ∧
  scores.xander.value = (scores.tatuya.value + scores.ivanna.value + scores.dorothy.value) / 3 + 10 ∧
  scores.olivia.value = (3/2) * scores.sam.value ∧
  scores.sam.value = 3.8 * scores.ivanna.value + 5.5

/-- The average score of all participants -/
noncomputable def average_score (scores : QuizScores) : ℝ :=
  (scores.xander.value + scores.tatuya.value + scores.ivanna.value +
   scores.dorothy.value + scores.olivia.value + scores.sam.value) / 6

/-- Theorem stating that the average score is approximately 145.46 -/
theorem average_score_approx (scores : QuizScores) 
  (h : quiz_conditions scores) : 
  |average_score scores - 145.46| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_approx_l280_28098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_iff_m_in_range_l280_28051

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- Define the derivative of f(x)
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*x + m

-- Theorem statement
theorem not_monotonic_iff_m_in_range (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-1) 2 ∧ x₂ ∈ Set.Ioo (-1) 2 ∧ x₁ < x₂ ∧ f m x₁ > f m x₂) ↔ 
  m ∈ Set.Ioo (-16) (1/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_iff_m_in_range_l280_28051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_G_equation_l280_28062

noncomputable def G (a b c d : ℝ) : ℝ := a^b + c/d

theorem solve_G_equation : ∃ x : ℝ, G 3 x 9 3 = 30 ∧ x = 3 := by
  use 3
  constructor
  · simp [G]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_G_equation_l280_28062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunglasses_count_verify_solution_l280_28061

/-- Represents the number of sunglasses pairs in the store before the theft -/
def n : ℕ := 111

/-- The initial average cost per pair of sunglasses -/
def initial_avg : ℚ := 900

/-- The cost of the stolen pair of sunglasses -/
def stolen_cost : ℚ := 2000

/-- The new average cost per pair after the theft -/
def new_avg : ℚ := 890

/-- Theorem stating that the number of sunglasses pairs before the theft was 111 -/
theorem sunglasses_count : n = 111 := by
  -- Proof goes here
  sorry

/-- Verification of the solution using the given conditions -/
theorem verify_solution : (initial_avg * n - stolen_cost) / (n - 1) = new_avg := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunglasses_count_verify_solution_l280_28061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l280_28092

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem f_increasing_interval :
  ∃ (a b : ℝ), a < b ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  a = -Real.pi/3 ∧ b = Real.pi/6 := by
  sorry

#check f_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l280_28092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l280_28020

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

-- State the theorem
theorem f_sum_property (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.sqrt_mul_self]
  -- The rest of the proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l280_28020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l280_28050

/-- Cost function for producing x devices when x < 80 -/
noncomputable def cost_low (x : ℝ) : ℝ := (1/2) * x^2 + 40 * x

/-- Cost function for producing x devices when x ≥ 80 -/
noncomputable def cost_high (x : ℝ) : ℝ := 101 * x + 8100 / x - 2180

/-- Profit function for producing x devices when 0 < x < 80 -/
noncomputable def profit_low (x : ℝ) : ℝ := 100 * x - cost_low x - 500

/-- Profit function for producing x devices when x ≥ 80 -/
noncomputable def profit_high (x : ℝ) : ℝ := 100 * x - cost_high x - 500

/-- The maximum profit is 1500 hundred-thousands yuan when producing 90 devices annually -/
theorem max_profit :
  (∀ x, x > 0 → profit_low x ≤ 1500) ∧
  (∀ x, x ≥ 80 → profit_high x ≤ 1500) ∧
  profit_high 90 = 1500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l280_28050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l280_28031

-- Define the curves C₁ and C₂ in polar coordinates
noncomputable def C₁ (θ : Real) : Real := 4 * Real.cos θ
noncomputable def C₂ (θ : Real) : Real := 4 * Real.sin θ

-- Define the fixed point M
def M : Real × Real := (2, 0)

-- Define the angle of the ray
noncomputable def ray_angle : Real := Real.pi / 3

-- Define the points A and B
noncomputable def A : Real × Real := (C₁ ray_angle * Real.cos ray_angle, C₁ ray_angle * Real.sin ray_angle)
noncomputable def B : Real × Real := (C₂ ray_angle * Real.cos ray_angle, C₂ ray_angle * Real.sin ray_angle)

-- Theorem statement
theorem area_of_triangle_MAB :
  let d := 2 * Real.sin ray_angle
  let AB := C₂ ray_angle - C₁ ray_angle
  (1/2) * d * AB = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l280_28031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_value_x_minus_abs_inequality_implies_a_range_l280_28068

-- Part 1
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 2 ↔ 3*x - |(-2*x + 1)| ≥ a) → a = 3 :=
sorry

-- Part 2
theorem x_minus_abs_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x - |x - a| ≤ 1) →
  a ∈ Set.Iic 1 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_implies_a_value_x_minus_abs_inequality_implies_a_range_l280_28068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l280_28049

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (x - 1)

-- Define IsValidArg
def IsValidArg (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (x + 2 ≥ 0) ∧ (x ≠ 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | IsValidArg f x} = {x : ℝ | x ≥ -2 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l280_28049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l280_28074

noncomputable def f (x : ℝ) : ℝ := (1 + 2^x) / (1 + 4^x)

theorem f_range : 
  (∀ x, 0 < f x ∧ f x ≤ (Real.sqrt 2 + 1) / 2) ∧ 
  (∀ y, 0 < y ∧ y ≤ (Real.sqrt 2 + 1) / 2 → ∃ x, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l280_28074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_2a_l280_28075

/-- A sequence defined recursively -/
noncomputable def u (a : ℝ) : ℕ → ℝ
  | 0 => 2 * a  -- Added case for 0
  | 1 => 2 * a
  | n + 2 => -2 * a / (u a (n + 1) + 2 * a)

/-- Theorem stating that the 16th term of the sequence equals 2a -/
theorem u_16_equals_2a (a : ℝ) (h : a > 0) : u a 16 = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_2a_l280_28075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_40_and_80_divisible_by_3_l280_28018

def is_divisible_by_3 (n : ℕ) : Bool := n % 3 = 0

def numbers_between_40_and_80_divisible_by_3 : List ℕ :=
  (List.range 41).filter (λ n => 40 < n && n ≤ 80 && is_divisible_by_3 n)

theorem average_of_numbers_between_40_and_80_divisible_by_3 :
  (numbers_between_40_and_80_divisible_by_3.sum : ℚ) / numbers_between_40_and_80_divisible_by_3.length = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_between_40_and_80_divisible_by_3_l280_28018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_circle_properties_l280_28005

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (2, 0)

-- Define line AB
def lineAB (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Define line l
def lineL (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the circle
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36/5

-- Theorem 1: Line l passes through C and is perpendicular to AB
theorem line_l_properties :
  (lineL C.1 C.2) ∧
  (∀ x y : ℝ, lineL x y → (x - C.1) * (B.1 - A.1) + (y - C.2) * (B.2 - A.2) = 0) := by
  sorry

-- Theorem 2: The circle is centered at C and tangent to AB
theorem circle_properties :
  (∃ x y : ℝ, lineAB x y ∧ circleC x y) ∧
  (∀ x y : ℝ, circleC x y → (x - C.1)^2 + (y - C.2)^2 = 36/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_circle_properties_l280_28005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l280_28078

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 3 = 6*y - 10*x + 1

/-- The area of the region -/
noncomputable def region_area : ℝ := 32 * Real.pi

/-- Theorem stating that the area of the region defined by the equation is 32π -/
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = π * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l280_28078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_circle_l280_28036

-- Define the line x - y = 1
def line (x y : ℝ) : Prop := x - y = 1

-- Define the circle (x + 2)² + (y - 1)² = 1
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

-- State the theorem
theorem shortest_distance_line_circle :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    line x₁ y₁ → circle_equation x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_line_circle_l280_28036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l280_28032

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x + 2) / (x + 5)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∀ x : ℝ, g x = 1 → x = 1 :=
by
  intro x
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l280_28032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l280_28047

theorem remainder_problem (k : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k % 6 = 5)
  (h3 : k < 38) :
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l280_28047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximizing_price_l280_28011

/-- Represents the revenue function for a company producing mountain bikes -/
noncomputable def revenue (x : ℝ) : ℝ := (1260000 + 300 * x - 3 * x^2) / 7

/-- Theorem stating that the revenue-maximizing price is 650 euros -/
theorem revenue_maximizing_price :
  let initial_price : ℝ := 600
  let initial_quantity : ℝ := 300
  let price_change : ℝ := 7
  let quantity_change : ℝ := 3
  ∃ (max_price : ℝ), max_price = initial_price + 50 ∧
    ∀ (p : ℝ), revenue (p - initial_price) ≤ revenue (max_price - initial_price) :=
by
  sorry

#check revenue_maximizing_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_maximizing_price_l280_28011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_ratio_l280_28052

noncomputable def f (x : ℝ) := x * Real.exp x

theorem perpendicular_lines_ratio (a b : ℝ) :
  (∃ (k : ℝ), k * (1 - 1) + Real.exp 1 = Real.exp 1 ∧ 
   k * (a / b) = -1 ∧
   ∀ (x : ℝ), k = (deriv f) 1) →
  a / b = -1 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_ratio_l280_28052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_hall_construction_cost_l280_28030

/-- Represents the cost function for Team A --/
noncomputable def team_a_cost (x : ℝ) : ℝ := 500 * (1200 / x + 3 * x) + 24000

/-- Represents the cost function for Team B --/
noncomputable def team_b_cost (x a : ℝ) : ℝ := 12000 + 500 * ((a + 1152) / x + a)

theorem sports_hall_construction_cost (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (∀ x > 0, team_a_cost x ≥ 84000) ∧
  (team_a_cost 20 = 84000) ∧
  (∀ x > 0, team_b_cost x a < team_a_cost x ↔ 0 < a ∧ a < 36) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_hall_construction_cost_l280_28030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersectionValues_l280_28033

-- Define the coordinate systems
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the curves
noncomputable def C1 (a t : ℝ) : Point2D :=
  { x := a + Real.sqrt 2 * t, y := 1 + Real.sqrt 2 * t }

def C2 (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 + 4 * Real.cos θ - ρ = 0

-- Define the intersection points
def intersection (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), C2 ((C1 a t₁).x^2 + (C1 a t₁).y^2) (Real.arctan ((C1 a t₁).y / (C1 a t₁).x)) ∧
                 C2 ((C1 a t₂).x^2 + (C1 a t₂).y^2) (Real.arctan ((C1 a t₂).y / (C1 a t₂).x))

-- Define the distance condition
def distanceCondition (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), intersection a ∧ 
    (C1 a t₁).x^2 + ((C1 a t₁).y - 1)^2 = 4 * ((C1 a t₂).x^2 + ((C1 a t₂).y - 1)^2)

-- The theorem to prove
theorem intersectionValues :
  ∀ a : ℝ, distanceCondition a → a = 1/36 ∨ a = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersectionValues_l280_28033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_division_l280_28090

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (2, 2)

-- Define the trapezoid
def trapezoid_ABCD : Set (ℝ × ℝ) := {p | ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ + t₂ ≤ 1 ∧
  ((p = (t₁ * A.1 + t₂ * B.1 + (1 - t₁ - t₂) * C.1,
         t₁ * A.2 + t₂ * B.2 + (1 - t₁ - t₂) * C.2)) ∨
   (p = (t₁ * A.1 + t₂ * D.1 + (1 - t₁ - t₂) * B.1,
         t₁ * A.2 + t₂ * D.2 + (1 - t₁ - t₂) * B.2)))}

-- Define the line BD
def line_BD (x : ℝ) : ℝ := 2

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the dividing line through A
def dividing_line (x : ℝ) : ℝ := x

theorem trapezoid_division :
  ∃ x : ℝ, x = 3/2 ∧
  area {p ∈ trapezoid_ABCD | p.1 ≤ x} = area {p ∈ trapezoid_ABCD | p.1 ≥ x} ∧
  (x, line_BD x) ∈ trapezoid_ABCD ∧
  3 + 2 + 4 + 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_division_l280_28090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_preserves_parallel_l280_28064

/-- An affine transformation in ℝ² -/
def AffineTransformation (A : Matrix (Fin 2) (Fin 2) ℝ) (b : Fin 2 → ℝ) : 
  (Fin 2 → ℝ) → (Fin 2 → ℝ) :=
  λ x ↦ Matrix.mulVec A x + b

/-- A line in ℝ² -/
structure Line where
  point : Fin 2 → ℝ
  direction : Fin 2 → ℝ

/-- Parallel lines in ℝ² -/
def Parallel (L₁ L₂ : Line) : Prop :=
  ∃ (c : ℝ), L₁.direction = c • L₂.direction

/-- Apply an affine transformation to a line -/
def ApplyAffineToLine (A : Matrix (Fin 2) (Fin 2) ℝ) (b : Fin 2 → ℝ) (L : Line) : Line :=
  { point := AffineTransformation A b L.point
  , direction := Matrix.mulVec A L.direction }

/-- The main theorem: affine transformations preserve parallel lines -/
theorem affine_preserves_parallel
  (A : Matrix (Fin 2) (Fin 2) ℝ) (b : Fin 2 → ℝ) (L₁ L₂ : Line) :
  Parallel L₁ L₂ → Parallel (ApplyAffineToLine A b L₁) (ApplyAffineToLine A b L₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_preserves_parallel_l280_28064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melies_meat_purchase_l280_28057

/-- The problem of determining how much meat Méliès bought --/
theorem melies_meat_purchase 
  (meat_price : ℝ) 
  (initial_money : ℝ) 
  (money_left : ℝ) 
  (h1 : meat_price = 82)
  (h2 : initial_money = 180)
  (h3 : money_left = 16) :
  (initial_money - money_left) / meat_price = 2 := by
  sorry

#check melies_meat_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melies_meat_purchase_l280_28057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l280_28086

/-- Parabola type representing y² = 4x -/
structure Parabola where
  equation : (ℝ → ℝ → Prop)
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a parabola y² = 4x, a line through its focus, and points on the parabola,
    prove that the circle formed by the feet of perpendiculars to the directrix
    passing through (-2,3) has the equation (x+1)² + (y-1)² = 5 -/
theorem parabola_circle_theorem (p : Parabola) (l : Line) (A B : ℝ × ℝ) :
  p.equation = (λ x y ↦ y^2 = 4*x) →
  p.focus = (1, 0) →
  p.directrix = (λ x y ↦ x = -1) →
  l.equation A.1 A.2 →
  l.equation B.1 B.2 →
  l.equation 1 0 →
  p.equation A.1 A.2 →
  p.equation B.1 B.2 →
  let A' := (-1, A.2)
  let B' := (-1, B.2)
  let C : Circle := { center := (-1, (A.2 + B.2)/2), radius := |A.2 - B.2|/2 }
  (C.center.1 + 2)^2 + (C.center.2 - 3)^2 = C.radius^2 →
  ∃ (x y : ℝ), (x+1)^2 + (y-1)^2 = 5 ↔ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l280_28086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_trip_time_difference_minutes_l280_28038

-- Define the average speed
noncomputable def average_speed : ℝ := 20

-- Define the distances for each trip
noncomputable def distance_trip1 : ℝ := 100
noncomputable def distance_trip2 : ℝ := 85

-- Define the stop time in hours
noncomputable def stop_time : ℝ := 0.25

-- Function to calculate travel time without stops
noncomputable def travel_time (distance : ℝ) : ℝ := distance / average_speed

-- Theorem statement
theorem trip_time_difference :
  travel_time distance_trip1 + stop_time - travel_time distance_trip2 = 1 := by
  sorry

-- Convert the time difference to minutes
theorem trip_time_difference_minutes :
  (travel_time distance_trip1 + stop_time - travel_time distance_trip2) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_difference_trip_time_difference_minutes_l280_28038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l280_28045

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- The slope of the tangent line at x = 1 -/
noncomputable def m : ℝ := f' 1

theorem tangent_line_equation :
  ∀ x y : ℝ, y = m * (x - point.fst) + point.snd ↔ y = x - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l280_28045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l280_28026

/-- Given circle is defined by the equation x^2 + y^2 + 2x = 0 -/
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

/-- Line of symmetry is defined by the equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- Circle C is symmetric to the given circle with respect to the symmetry line -/
def is_symmetric (C : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), 
    (∀ p : ℝ × ℝ, symmetry_line p.1 p.2 → f p = p) ∧
    (∀ p : ℝ × ℝ, given_circle p.1 p.2 ↔ C (f p).1 (f p).2)

/-- The equation of circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- Theorem stating that if a circle C is symmetric to the given circle
    with respect to the symmetry line, then its equation is circle_C -/
theorem symmetric_circle_equation :
  ∀ C : ℝ → ℝ → Prop, is_symmetric C → (∀ x y, C x y ↔ circle_C x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circle_equation_l280_28026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_power_l280_28027

/-- Given a = log_{0.6}0.5, b = ln0.5, and c = 0.6^0.5, prove that b < c < a -/
theorem order_of_logarithms_and_power (a b c : ℝ) 
  (ha : a = Real.log 0.5 / Real.log 0.6) 
  (hb : b = Real.log 0.5) 
  (hc : c = Real.rpow 0.6 0.5) : 
  b < c ∧ c < a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_power_l280_28027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l280_28073

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x * (a^x - 3*a^2 - 1)

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f a x < f a y) →
  a ≥ Real.sqrt 3 / 3 ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l280_28073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l280_28023

-- Define the circles and point
noncomputable def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
noncomputable def C₂ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 4
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equations
noncomputable def tangent_line₁ (x y : ℝ) : Prop := 3*x - 4*y + 10 = 0
def tangent_line₂ (x : ℝ) : Prop := x = 2

-- Define the length of segment AB
noncomputable def AB_length : ℝ := 4 * Real.sqrt 5 / 5

theorem circle_tangent_and_intersection :
  -- Part 1: Tangent lines
  (∃ (x y : ℝ), C₁ x y ∧ (tangent_line₁ x y ∨ tangent_line₂ x) ∧ 
    ((x - P.1)^2 + (y - P.2)^2 = 1)) ∧
  -- Part 2: Intersection length
  (∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = AB_length) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l280_28023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_intersection_condition_l280_28072

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the line l
def line_l (x y a : ℝ) : Prop := x + a*y + 2*a = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem 1: Line l is tangent to circle C iff a = -3/4
theorem tangent_condition (a : ℝ) :
  (∃ x y : ℝ, circle_C x y ∧ line_l x y a ∧
    ∀ x' y' : ℝ, circle_C x' y' ∧ line_l x' y' a → (x = x' ∧ y = y')) ↔
  a = -3/4 :=
by sorry

-- Theorem 2: When line l intersects circle C at A and B with |AB| = 2√2,
-- the equation of line l is either x-y-2=0 or x-7y-14=0
theorem intersection_condition :
  ∀ a : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l x1 y1 a ∧ line_l x2 y2 a ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2) →
  (a = -1 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_intersection_condition_l280_28072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l280_28006

-- Define the function f(x) = |tan x|
noncomputable def f (x : ℝ) : ℝ := |Real.tan x|

-- State the theorem
theorem f_properties :
  -- f is monotonically increasing on (0, π/2)
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) ∧
  -- f is an even function
  (∀ x, f (-x) = f x) ∧
  -- f has a period of π
  (∀ x, f (x + Real.pi) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l280_28006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_30_l280_28095

/-- Represents the vendor's sunglasses business --/
structure SunglassesVendor where
  cost_price : ℚ
  daily_sales : ℕ
  sign_cost : ℚ

/-- Calculates the selling price of sunglasses given the vendor's business parameters --/
def calculate_selling_price (vendor : SunglassesVendor) : ℚ :=
  (vendor.cost_price * vendor.daily_sales + vendor.sign_cost * 2) / vendor.daily_sales

/-- Theorem stating that the selling price is $30 given the specific conditions --/
theorem selling_price_is_30 (vendor : SunglassesVendor) 
  (h1 : vendor.cost_price = 26)
  (h2 : vendor.daily_sales = 10)
  (h3 : vendor.sign_cost = 20) :
  calculate_selling_price vendor = 30 := by
  sorry

#eval calculate_selling_price { cost_price := 26, daily_sales := 10, sign_cost := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_30_l280_28095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_equals_seven_l280_28037

noncomputable section

/-- Triangle ABC with vertices A(0, 0), B(7, 0), C(3, 4) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 0⟩, ⟨7, 0⟩, ⟨3, 4⟩}

/-- Point through which the line passes -/
noncomputable def line_point : ℝ × ℝ := ⟨6 - 2 * Real.sqrt 2, 3 - Real.sqrt 2⟩

/-- Intersection point P on segment AC -/
noncomputable def point_P : ℝ × ℝ := sorry

/-- Intersection point Q on segment BC -/
noncomputable def point_Q : ℝ × ℝ := sorry

/-- Area of triangle PQC -/
def area_PQC : ℝ := 14 / 3

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: If the area of triangle PQC is 14/3, then |CP| + |CQ| = 7 -/
theorem sum_of_distances_equals_seven :
  area_PQC = 14 / 3 →
  distance ⟨3, 4⟩ point_P + distance ⟨3, 4⟩ point_Q = 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_equals_seven_l280_28037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_minimizes_sum_of_distances_l280_28085

/-- Represents a point on a line -/
structure Point where
  x : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- The sum of distances from a point to a list of points -/
def sumOfDistances (p : Point) (points : List Point) : ℝ :=
  points.foldr (λ q sum => sum + distance p q) 0

theorem median_minimizes_sum_of_distances 
  (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ p₉ : Point)
  (h_order : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ 
             p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x ∧ p₈.x < p₉.x) :
  ∀ p : Point, sumOfDistances p₅ [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈, p₉] ≤ 
               sumOfDistances p [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈, p₉] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_minimizes_sum_of_distances_l280_28085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l280_28081

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - 2 * (Real.sin x) ^ 2) / Real.sin x

-- State the theorem
theorem f_properties :
  ∃ (max_value : ℝ) (monotonic_interval : Set ℝ),
    (∀ x, x ∈ Set.Ioo 0 Real.pi → f x ≤ max_value) ∧
    (max_value = 2 * Real.sqrt 2) ∧
    (monotonic_interval = Set.Ico (3 * Real.pi / 4) Real.pi) ∧
    (∀ x₁ x₂, x₁ ∈ monotonic_interval → x₂ ∈ monotonic_interval → x₁ < x₂ → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l280_28081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_length_theorem_l280_28094

/-- The length of the intersection curve between a sphere and a unit cube -/
noncomputable def intersectionCurveLength : ℝ := (5 * Real.sqrt 3 * Real.pi) / 6

/-- The radius of the sphere -/
noncomputable def sphereRadius : ℝ := (2 * Real.sqrt 3) / 3

/-- Theorem: The length of the curve formed by the intersection of a sphere
(with center at a vertex of a unit cube and radius 2√3/3) and the cube's surface
is equal to 5√3π/6 -/
theorem intersection_curve_length_theorem (cube : Set (ℝ × ℝ × ℝ))
  (sphere : Set (ℝ × ℝ × ℝ)) :
  (∀ (x y z : ℝ), (x, y, z) ∈ cube ↔ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1) →
  (∃ (a b c : ℝ), (a, b, c) ∈ cube ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ sphere ↔
      (x - a)^2 + (y - b)^2 + (z - c)^2 = sphereRadius^2)) →
  Real.sqrt ((5 * Real.sqrt 3 * Real.pi / 6)^2) = intersectionCurveLength :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_length_theorem_l280_28094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l280_28054

-- Define what it means for two numbers to be similar
def similar (a b : ℝ) : Prop := max a b ≤ Real.sqrt 2 * min a b

-- Define a function that checks if a list of numbers satisfies the similarity condition
def all_similar (l : List ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ l → b ∈ l → similar a b

-- State the theorem
theorem no_three_similar_piles :
  ∀ x : ℝ, x > 0 → ¬∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = x ∧ all_similar [a, b, c] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_similar_piles_l280_28054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l280_28022

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (a - c) / (b + c) = Real.sin B / (Real.sin A + Real.sin C) →
  D.1 ≥ 0 → D.1 ≤ c →
  D.2 = b * Real.sin C →
  4 * D.1 = 3 * c →
  A = 2 * π / 3 ∧ Real.cos C = 7 * Real.sqrt 19 / 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l280_28022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l280_28008

noncomputable def f (x : ℝ) : ℝ := (2^x) / (2^x + 1)

theorem inequality_proof (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  f (a - b) + f (b - c) + f (c - a) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l280_28008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teams_for_weights_l280_28028

-- Define the set of weights
def WeightSet : Set ℕ := {w | 1 ≤ w ∧ w ≤ 100}

-- Define the property of no weight being twice another in a subset
def NoDoubleWeight (S : Set ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → y ≠ 2 * x

-- The theorem to prove
theorem min_teams_for_weights :
  ∃ (S₁ S₂ : Set ℕ),
    S₁ ∪ S₂ = WeightSet ∧
    S₁ ∩ S₂ = ∅ ∧
    NoDoubleWeight S₁ ∧
    NoDoubleWeight S₂ :=
by
  sorry

#check min_teams_for_weights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teams_for_weights_l280_28028
