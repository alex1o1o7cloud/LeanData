import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1088_108871

-- Define the hyperbola G
def hyperbola_G (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 9 = 1

-- Define the properties of G
def center_at_origin : Prop := True
def major_axis_on_x : Prop := True
noncomputable def eccentricity : ℝ := Real.sqrt 5 / 2
def distance_difference : ℝ := 12

-- Theorem statement
theorem hyperbola_equation :
  center_at_origin →
  major_axis_on_x →
  eccentricity = Real.sqrt 5 / 2 →
  distance_difference = 12 →
  ∀ x y : ℝ, hyperbola_G x y ↔ x^2 / 36 - y^2 / 9 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1088_108871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_roots_of_unity_and_quadratic_l1088_108810

/-- A root of unity is a complex number that is a solution to z^n = 1 for some positive integer n -/
def RootOfUnity (z : ℂ) : Prop :=
  ∃ n : ℕ+, z ^ (n : ℕ) = 1

/-- The quadratic equation z^2 + az + b = 0 with integer coefficients -/
def QuadraticEquation (z : ℂ) : Prop :=
  ∃ a b : ℤ, z^2 + (a : ℂ)*z + (b : ℂ) = 0

/-- The main theorem stating that there are exactly 8 complex numbers satisfying both conditions -/
theorem eight_roots_of_unity_and_quadratic :
  ∃! (S : Finset ℂ), (∀ z ∈ S, RootOfUnity z ∧ QuadraticEquation z) ∧ S.card = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_roots_of_unity_and_quadratic_l1088_108810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_isosceles_triangles_l1088_108807

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) / 4 * ((((t.leg : ℚ) ^ 2) - (((t.base : ℚ) / 2) ^ 2)).sqrt)

/-- Theorem stating the minimum perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter t1 ≤ perimeter s1) ∧
    perimeter t1 = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_isosceles_triangles_l1088_108807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1088_108860

/-- The volume of a sphere with radius R -/
noncomputable def V (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

/-- The surface area of a sphere with radius R -/
noncomputable def A (R : ℝ) : ℝ := 4 * Real.pi * R^2

theorem sphere_volume_derivative_equals_surface_area {R : ℝ} (h : R > 0) :
  deriv V R = A R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1088_108860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1088_108840

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the foci and endpoints of the minor axis
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
variable (B₁ : ℝ × ℝ)
variable (B₂ : ℝ × ℝ)

-- Define the equilateral triangle condition
def is_equilateral_triangle (F₁ B₁ B₂ : ℝ × ℝ) : Prop :=
  (F₁.1 - B₁.1)^2 + (F₁.2 - B₁.2)^2 = (B₁.1 - B₂.1)^2 + (B₁.2 - B₂.2)^2 ∧
  (F₁.1 - B₁.1)^2 + (F₁.2 - B₁.2)^2 = (F₁.1 - B₂.1)^2 + (F₁.2 - B₂.2)^2

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := x - Real.sqrt 3

-- Define the theorem
theorem ellipse_theorem (C : Ellipse) 
  (h_equilateral : is_equilateral_triangle F₁ B₁ B₂) :
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1}) ∧
  (∃ P Q : ℝ × ℝ, 
    P ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1} ∧
    Q ∈ {(x, y) | x^2 / C.a^2 + y^2 / C.b^2 = 1} ∧
    P.2 = line_l P.1 ∧ Q.2 = line_l Q.1 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (8/5)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1088_108840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_below_curve_l1088_108875

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_a_below_curve (a : ℤ) 
  (h : ∀ x : ℝ, (a : ℝ) * x ≤ f (x + 1)) : 
  a ≤ 3 ∧ ∃ x : ℝ, 3 * x ≤ f (x + 1) :=
by
  sorry

#check max_a_below_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_below_curve_l1088_108875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l1088_108841

/-- Given vectors a and b, if the cosine of the angle between them is 2/15, then the middle component of b is 0. -/
theorem vector_angle_cosine (a b : ℝ × ℝ × ℝ) (h : a = (2, 2, -1)) (k : b.1 = 3 ∧ b.2.2 = 4) :
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) / (Real.sqrt (a.1^2 + a.2.1^2 + a.2.2^2) * Real.sqrt (b.1^2 + b.2.1^2 + b.2.2^2)) = 2/15 → b.2.1 = 0 := by
  sorry

#check vector_angle_cosine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l1088_108841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_f_l1088_108829

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/2)

-- Define the function t(x)
def t (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

-- Define the domain of f(x)
def domain_f (x : ℝ) : Prop := x < 1/2 ∨ x > 1

-- Theorem stating the increasing interval of f(x)
theorem increasing_interval_f :
  ∀ x y, domain_f x → domain_f y → x < y → x < 1/2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_f_l1088_108829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_teaches_Mathematics_l1088_108850

-- Define the teachers and subjects
inductive Teacher : Type
| A : Teacher
| B : Teacher
| C : Teacher
| D : Teacher

inductive Subject : Type
| Mathematics : Subject
| Physics : Subject
| Chemistry : Subject
| English : Subject

-- Define the can_teach relation
def can_teach : Teacher → Subject → Prop := sorry

-- Define the teaches relation
def teaches : Teacher → Subject → Prop := sorry

-- Axioms based on the conditions
axiom can_teach_A : ∀ s, can_teach Teacher.A s ↔ (s = Subject.Physics ∨ s = Subject.Chemistry)
axiom can_teach_B : ∀ s, can_teach Teacher.B s ↔ (s = Subject.Mathematics ∨ s = Subject.English)
axiom can_teach_C : ∀ s, can_teach Teacher.C s ↔ (s = Subject.Mathematics ∨ s = Subject.Physics ∨ s = Subject.Chemistry)
axiom can_teach_D : ∀ s, can_teach Teacher.D s ↔ (s = Subject.Chemistry)

axiom one_subject_per_teacher : ∀ t, ∃! s, teaches t s
axiom one_teacher_per_subject : ∀ s, ∃! t, teaches t s

axiom teaches_implies_can_teach : ∀ t s, teaches t s → can_teach t s

-- Theorem to prove
theorem C_teaches_Mathematics : teaches Teacher.C Subject.Mathematics := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_teaches_Mathematics_l1088_108850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_correct_l1088_108879

/-- An isosceles trapezoid with inscribed circles in its subtrapezoids -/
structure InscribedCircleTrapezoid where
  -- The length of the top base
  c : ℝ
  -- The length of the parallel line segment MN
  d : ℝ
  -- Condition that c > 2d
  h_c_gt_2d : c > 2 * d

/-- The bases of the trapezoid given the conditions -/
noncomputable def trapezoid_bases (t : InscribedCircleTrapezoid) : ℝ × ℝ :=
  let base₁ := t.c - t.d + Real.sqrt (t.c * (t.c - 2 * t.d))
  let base₂ := t.c - t.d - Real.sqrt (t.c * (t.c - 2 * t.d))
  (base₁, base₂)

/-- Theorem stating that the calculated bases are correct for the given trapezoid -/
theorem trapezoid_bases_correct (t : InscribedCircleTrapezoid) :
  trapezoid_bases t = (t.c - t.d + Real.sqrt (t.c * (t.c - 2 * t.d)),
                       t.c - t.d - Real.sqrt (t.c * (t.c - 2 * t.d))) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_correct_l1088_108879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_not_always_decreasing_l1088_108888

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem stating the properties of the function
theorem inverse_proportion_properties :
  (∀ x, x > 0 → f x > 0) ∧  -- First quadrant
  (∀ x, x < 0 → f x < 0) ∧  -- Third quadrant
  (f 2 = 3) ∧               -- Passes through (2,3)
  (∀ x, x ≠ 0 → f x ≠ 0) ∧ -- Doesn't intersect x-axis
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → f x > 1/ε) -- Doesn't intersect y-axis
  := by
  sorry

-- Theorem stating that "y decreases as x increases" is not always true
theorem not_always_decreasing :
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_not_always_decreasing_l1088_108888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_z_in_S_wz_in_S_l1088_108844

-- Define the set S in the complex plane
def S : Set ℂ := {z | Complex.abs z.re ≤ 1 ∧ Complex.abs z.im ≤ 1}

-- Define the complex number (1/2 + 1/2i)
noncomputable def w : ℂ := Complex.mk (1/2) (1/2)

-- Theorem statement
theorem all_z_in_S_wz_in_S :
  ∀ z ∈ S, w * z ∈ S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_z_in_S_wz_in_S_l1088_108844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1088_108802

theorem trigonometric_identity (α : ℝ) (h : Real.tan α + 1 / Real.tan α = 5/2) :
  2 * Real.sin α ^ 2 - 3 * Real.cos α * Real.sin α + 2 = 6/5 ∨
  2 * Real.sin α ^ 2 - 3 * Real.cos α * Real.sin α + 2 = 12/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1088_108802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_two_digit_number_l1088_108808

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def reverse_digits (n : ℕ) : ℕ :=
  (units_digit n) * 10 + tens_digit n

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit_number n ∧ 
  n = factorial (tens_digit n - units_digit n) * (reverse_digits n - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_two_digit_number_l1088_108808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l1088_108862

/-- Represents the time (in hours) it takes to fill or empty the pool -/
structure FillTime where
  hours : ℚ
  positive : hours > 0

/-- Represents the rate at which the pool is filled or emptied (fraction of pool per hour) -/
def fillRate (t : FillTime) : ℚ := 1 / t.hours

theorem pool_fill_time
  (pipe_a pipe_b pipe_c leak : FillTime)
  (h_a : pipe_a.hours = 10)
  (h_b : pipe_b.hours = 6)
  (h_c : pipe_c.hours = 5)
  (h_leak : leak.hours = 15)
  (tarp_factor : ℚ)
  (h_tarp : tarp_factor = 1/2) :
  let combined_rate := (fillRate pipe_a + fillRate pipe_b + fillRate pipe_c) * tarp_factor - fillRate leak
  (1 / combined_rate : ℚ) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l1088_108862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l1088_108842

/-- A function that is symmetric to log₂(x) with respect to the origin -/
noncomputable def f : ℝ → ℝ := sorry

/-- The logarithm base 2 function -/
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

/-- The symmetry condition -/
axiom symmetry_condition (x : ℝ) : x > 0 → f (-x) = -g x

/-- The domain condition for g -/
axiom g_domain (x : ℝ) : x > 0 → g x = Real.log x / Real.log 2

theorem f_expression (x : ℝ) : x < 0 → f x = -g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l1088_108842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1088_108833

-- Define variables x and y as real numbers
variable (x y : ℝ)

-- Define the polynomials A, B, and C as functions
def A : ℝ → ℝ → ℝ := λ x y => x^2 + x*y + 3*y
def B : ℝ → ℝ → ℝ := λ x y => x^2 - x*y

-- Define C implicitly using the given condition
def C : ℝ → ℝ → ℝ := λ x y => 3 * (A x y - (2*x^2 + y))

-- Theorem for the first part of the problem
theorem part_one : ∀ x y : ℝ, 3*(A x y) - B x y = 2*x^2 + 4*x*y + 9*y := by
  sorry

-- Theorem for the second part of the problem
theorem part_two : ∀ x y : ℝ, A x y + (1/3)*(C x y) = 2*x*y + 5*y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1088_108833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1088_108856

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({3, 4, 5, 6} : Set ℕ) → 
  b ∈ ({3, 4, 5, 6} : Set ℕ) → 
  c ∈ ({3, 4, 5, 6} : Set ℕ) → 
  d ∈ ({3, 4, 5, 6} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * c + c * d + a * d) ≤ 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1088_108856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1088_108801

noncomputable section

/-- The curve defined by y = (ax + b) / (cx + d) -/
def curve (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The line y = -x -/
def symmetry_axis (x : ℝ) : ℝ := -x

/-- The necessary condition for y = -x to be an axis of symmetry -/
theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x, curve a b c d (-symmetry_axis x) = symmetry_axis (curve a b c d x)) ↔ a + d = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l1088_108801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_l1088_108861

/-- Represents the cost of tickets for a family visit to a leisure park. -/
structure LeisureParkTickets where
  child_cost : ℕ
  adult_cost : ℕ
  senior_cost : ℕ
  discounted_senior_cost : ℕ
  adult_child_diff : adult_cost = child_cost + 10
  senior_adult_diff : senior_cost = adult_cost - 5
  senior_discount : discounted_senior_cost = senior_cost - 3
  total_cost : 5 * child_cost + 2 * adult_cost + 2 * senior_cost + discounted_senior_cost = 212

/-- Theorem stating that the cost of an adult ticket is $28. -/
theorem adult_ticket_cost (tickets : LeisureParkTickets) : tickets.adult_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_l1088_108861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_with_five_thousands_l1088_108831

/-- A four-digit positive integer with the thousands digit 5 -/
def FourDigitWithFiveThousands : Type := {n : ℕ | 5000 ≤ n ∧ n ≤ 5999}

/-- The count of four-digit positive integers with the thousands digit 5 -/
def CountFourDigitWithFiveThousands : ℕ := Finset.range 1000 |>.card

/-- Theorem: The count of four-digit positive integers with the thousands digit 5 is 1000 -/
theorem count_four_digit_with_five_thousands :
  CountFourDigitWithFiveThousands = 1000 := by
  unfold CountFourDigitWithFiveThousands
  simp [Finset.card_range]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_with_five_thousands_l1088_108831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_theorem_l1088_108834

def factory_problem (output_A output_B output_C defect_rate_A defect_rate_B defect_rate_C : ℝ) : Prop :=
  output_A = 0.25 ∧
  output_B = 0.35 ∧
  output_C = 0.40 ∧
  defect_rate_A = 0.05 ∧
  defect_rate_B = 0.04 ∧
  defect_rate_C = 0.02 ∧
  output_A + output_B + output_C = 1

theorem factory_theorem
  (output_A output_B output_C defect_rate_A defect_rate_B defect_rate_C : ℝ)
  (h : factory_problem output_A output_B output_C defect_rate_A defect_rate_B defect_rate_C) :
  let p_defective := output_A * defect_rate_A + output_B * defect_rate_B + output_C * defect_rate_C
  let p_A_given_defective := (output_A * defect_rate_A) / p_defective
  let p_B_given_defective := (output_B * defect_rate_B) / p_defective
  let p_C_given_defective := (output_C * defect_rate_C) / p_defective
  p_defective = 0.0345 ∧ 
  p_A_given_defective = 25 / 69 ∧
  p_B_given_defective = 28 / 69 ∧
  p_C_given_defective = 16 / 69 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_theorem_l1088_108834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_and_a_range_l1088_108889

theorem angle_between_lines_and_a_range (a : ℝ) :
  let l1 := fun x : ℝ => x
  let l2 := fun x : ℝ => a * x
  let angle := Real.arctan a - Real.arctan 1
  0 < angle ∧ angle < π / 12 →
  (a > Real.sqrt 3 / 3 ∧ a < 1) ∨ (a > 1 ∧ a < Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_lines_and_a_range_l1088_108889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_cosine_right_triangle_area_l1088_108870

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = 2 * (Real.sin t.A) * (Real.sin t.C)

-- Part 1: Isosceles triangle
theorem isosceles_cosine (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : t.a = t.b) : 
  Real.cos t.C = 7/8 := by
sorry

-- Part 2: Right triangle
theorem right_triangle_area (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : t.B = Real.pi/2) 
  (h3 : t.c = Real.sqrt 2) : 
  (1/2) * t.a * t.b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_cosine_right_triangle_area_l1088_108870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1088_108864

/-- Circle C₁ with center (2, 0) and radius 2 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 2^2}

/-- Line C₂ with equation 3x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 1 = 0}

/-- Distance from point (a, b) to line Ax + By + C = 0 -/
noncomputable def distPointToLine (a b A B C : ℝ) : ℝ :=
  |A * a + B * b + C| / Real.sqrt (A^2 + B^2)

theorem circle_line_intersection :
  ∃ (p q : ℝ × ℝ), p ∈ C₁ ∧ p ∈ C₂ ∧ q ∈ C₁ ∧ q ∈ C₂ ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 3 := by
  sorry

#check circle_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1088_108864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1088_108880

/-- The curve C -/
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The line l -/
def line (x y : ℝ) : Prop := 3*x + 4*y - 13 = 0

/-- Distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x + 4*y - 13| / Real.sqrt (3^2 + 4^2)

/-- Maximum distance from curve C to line l -/
theorem max_distance_curve_to_line :
  ∃ (d : ℝ), d = 3 ∧
  (∀ (x y : ℝ), curve x y → distance_to_line x y ≤ d) ∧
  (∃ (x₀ y₀ : ℝ), curve x₀ y₀ ∧ distance_to_line x₀ y₀ = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l1088_108880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1088_108887

/-- Represents the speed of a train given its length and time to cross a pole. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem stating that a train with length 90 meters crossing a pole in 9 seconds
    has a speed of 10 meters per second. -/
theorem train_speed_calculation :
  train_speed 90 9 = 10 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1088_108887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l1088_108884

/-- Represents the volume of a polyhedron in the sequence --/
def polyhedronVolume : ℕ → ℚ := sorry

/-- The volume added at each step --/
def volumeAdded : ℕ → ℚ := sorry

/-- The initial volume of Q0 --/
axiom initial_volume : polyhedronVolume 0 = 1

/-- The ratio of new tetrahedron side length to original side length --/
def sideRatio : ℚ := 2/3

/-- The volume ratio of each new tetrahedron to the previous one --/
def volumeRatio : ℚ := sideRatio ^ 3

/-- The number of new tetrahedra added at each step --/
def numNewTetrahedra : ℕ := 4

/-- The volume added at each step is related to the previous step --/
axiom volume_added_relation : ∀ n : ℕ, volumeAdded (n + 1) = numNewTetrahedra * volumeRatio * volumeAdded n

/-- The volume added at the first step --/
axiom first_volume_added : volumeAdded 1 = numNewTetrahedra * volumeRatio

/-- The total volume at each step is the sum of the previous volume and the added volume --/
axiom volume_recursion : ∀ n : ℕ, polyhedronVolume (n + 1) = polyhedronVolume n + volumeAdded (n + 1)

/-- The main theorem: The volume of Q3 is 149417/19683 --/
theorem volume_Q3 : polyhedronVolume 3 = 149417/19683 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_Q3_l1088_108884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sine_l1088_108837

theorem arithmetic_progression_sine (x y z : ℝ) (α : ℝ) : 
  (∃ d : ℝ, y = x + d ∧ z = y + d) →  -- x, y, z form an arithmetic progression
  α = Real.arccos (2/3) →  -- Common difference
  (∃ k : ℝ, 6 / Real.sin y = 1 / Real.sin x + k ∧ 
            1 / Real.sin z = 6 / Real.sin y + k) →  -- 1/sin(x), 6/sin(y), 1/sin(z) form an arithmetic progression
  Real.sin y ^ 2 = 5/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sine_l1088_108837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_focus_l1088_108857

/-- The ellipse with equation x^2 + 4y^2 = 36 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 4 * p.2^2 = 36}

/-- The left focus of the ellipse -/
noncomputable def LeftFocus : ℝ × ℝ := (-3 * Real.sqrt 3, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum distance between a point on the ellipse and the left focus -/
theorem max_distance_to_focus :
  ∃ (max : ℝ), max = 6 + 3 * Real.sqrt 3 ∧
    ∀ p ∈ Ellipse, distance p LeftFocus ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_focus_l1088_108857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1088_108877

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1088_108877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remaining_sum_l1088_108828

/-- The set of numbers from 1 to 2n - 1 -/
def A (n : ℕ) : Finset ℕ := Finset.range (2*n - 1).succ

/-- The rule that if a is removed, so is 2a -/
def rule1 (S : Finset ℕ) : Prop := ∀ a ∈ S, 2*a ∈ S

/-- The rule that if a and b are removed, so is a + b -/
def rule2 (S : Finset ℕ) : Prop := ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S

/-- The set of removed numbers -/
def removed (n : ℕ) (S : Finset ℕ) : Prop :=
  S ⊆ A n ∧ S.card ≥ n - 1 ∧ rule1 S ∧ rule2 S

/-- The sum of remaining numbers -/
def remainingSum (n : ℕ) (S : Finset ℕ) : ℕ :=
  (A n \ S).sum id

theorem max_remaining_sum (n : ℕ) :
  ∃ S, removed n S ∧ ∀ T, removed n T → remainingSum n S ≥ remainingSum n T ∧
  remainingSum n S = n^2 := by
  sorry

#check max_remaining_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_remaining_sum_l1088_108828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_of_men_l1088_108820

theorem original_number_of_men (original_days actual_days absent_men : ℕ) 
  (h1 : original_days = 20)
  (h2 : absent_men = 2)
  (h3 : actual_days = 22) :
  ∃ original_number : ℕ, 
    original_number * original_days = (original_number - absent_men) * actual_days ∧
    original_number = 22 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_of_men_l1088_108820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_parity_monotonic_count_l1088_108868

/-- Represents a decimal digit (1-9) -/
def Digit := Fin 9

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Checks if a natural number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- Defines a parity-monotonic sequence of digits -/
def isParityMonotonic (digits : List Digit) : Prop :=
  ∀ i j, i < j → (
    (isOdd (digits.get i).val → (digits.get i).val < (digits.get j).val) ∧
    (isEven (digits.get i).val → (digits.get i).val > (digits.get j).val)
  )

/-- Counts the number of parity-monotonic sequences of a given length -/
def countParityMonotonic : ℕ → ℕ
  | 0 => 0
  | 1 => 9
  | n + 1 => 4 * countParityMonotonic n

theorem four_digit_parity_monotonic_count :
  countParityMonotonic 4 = 576 := by sorry

#eval countParityMonotonic 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_parity_monotonic_count_l1088_108868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_gum_packs_l1088_108851

def candy_store_problem (initial_amount : ℚ) (gum_price : ℚ) (chocolate_bars : ℕ) 
  (chocolate_price : ℚ) (candy_canes : ℕ) (candy_cane_price : ℚ) (remaining : ℚ) : ℕ :=
  let total_spent := initial_amount - remaining
  let chocolate_cost := chocolate_bars * chocolate_price
  let candy_cane_cost := candy_canes * candy_cane_price
  let gum_cost := total_spent - chocolate_cost - candy_cane_cost
  (gum_cost / gum_price).floor.toNat

theorem anna_gum_packs : 
  candy_store_problem 10 1 5 1 2 (1/2) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_gum_packs_l1088_108851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_l1088_108806

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

-- Define the points through which the circle passes
def point1 : ℝ × ℝ := (1, 0)
noncomputable def point2 : ℝ × ℝ := (0, Real.sqrt 3)
def point3 : ℝ × ℝ := (-3, 0)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point1.1 point1.2 ∧
  circle_equation point2.1 point2.2 ∧
  circle_equation point3.1 point3.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_passes_through_points_l1088_108806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_h_condition_l1088_108821

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (3 * x + 59) / 5

-- State the theorem
theorem unique_fixed_point_of_h :
  ∃! x : ℝ, h x = x ∧ x = 59 / 2 := by
  sorry

-- Verify the condition h(5x-3) = 3x + 10
theorem h_condition (x : ℝ) :
  h (5 * x - 3) = 3 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_h_condition_l1088_108821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_axes_of_symmetry_l1088_108876

/-- A cube is a three-dimensional shape with 6 square faces. -/
structure Cube where
  faces : Fin 6 → Square

/-- An axis of symmetry for a cube. -/
inductive AxisOfSymmetry
  | BodyDiagonal
  | FaceDiagonal
  | EdgeDiagonal

/-- The total number of axes of symmetry for a cube. -/
def numberOfAxesOfSymmetry (c : Cube) : ℕ :=
  4 + 6 + 3 -- 4 body diagonals, 6 face diagonals, 3 edge diagonals

/-- Theorem: The total number of axes of symmetry for a cube is 9. -/
theorem cube_axes_of_symmetry (c : Cube) :
  numberOfAxesOfSymmetry c = 13 := by
  unfold numberOfAxesOfSymmetry
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_axes_of_symmetry_l1088_108876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt3_sufficient_not_necessary_l1088_108867

theorem tan_sqrt3_sufficient_not_necessary :
  (∀ A : Real, 0 < A → A < π → (Real.tan A > Real.sqrt 3 → A > π/3)) ∧
  (∃ A : Real, 0 < A ∧ A < π ∧ A > π/3 ∧ ¬(Real.tan A > Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt3_sufficient_not_necessary_l1088_108867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_correct_l1088_108811

/-- The plane equation 2x + 3y - z = 0 -/
def plane_equation (x y z : ℝ) : Prop := 2*x + 3*y - z = 0

/-- The vector to be projected -/
def v : Fin 3 → ℝ
| 0 => 2
| 1 => -3
| 2 => 4

/-- The normal vector of the plane -/
def n : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => -1

/-- The projection of v onto the plane -/
noncomputable def projection : Fin 3 → ℝ
| 0 => 23/7
| 1 => -15/14
| 2 => 47/14

theorem projection_is_correct :
  projection = v - ((v • n) / (n • n)) • n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_correct_l1088_108811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_activity_rates_sum_of_squares_l1088_108843

theorem activity_rates_sum_of_squares : ∃ (b j s : ℕ), 
  (3 * b + 1 * j + 5 * s = 89) ∧ 
  (4 * b + 3 * j + 2 * s = 106) ∧ 
  (b^2 + j^2 + s^2 = 821) := by
  -- We'll use existential quantification to state that such b, j, and s exist
  use 1, 26, 12
  -- Now we'll prove each part of the conjunction
  apply And.intro
  · -- Prove Tom's equation
    norm_num
  · apply And.intro
    · -- Prove Jerry's equation
      norm_num
    · -- Prove the sum of squares
      norm_num

#eval 1^2 + 26^2 + 12^2  -- This should output 821

end NUMINAMATH_CALUDE_ERRORFEEDBACK_activity_rates_sum_of_squares_l1088_108843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l1088_108859

theorem smallest_integer_in_set (s : Set ℤ) : 
  (∀ n, n ∈ s → ∃ k : ℤ, n = 2*k + 1) →  -- s contains only odd integers
  (∀ n m, n ∈ s → m ∈ s → n < m → ∃ k : ℤ, m = n + 2*k) →  -- s contains consecutive integers
  155 ∈ s →  -- 155 is in the set
  (∀ n, n ∈ s → n ≤ 155) ∧ (∀ n, n ∈ s → n ≥ 155) →  -- 155 is the median
  167 ∈ s →  -- 167 is in the set
  (∀ n, n ∈ s → n ≤ 167) →  -- 167 is the greatest integer
  143 ∈ s ∧ (∀ n, n ∈ s → n ≥ 143)  -- 143 is the smallest integer
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_set_l1088_108859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_triangles_area_l1088_108815

/-- Predicate to check if a triangle is a 45-45-90 triangle -/
def is_45_45_90_triangle (triangle : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Function to calculate the length of the hypotenuse of a triangle -/
def hypotenuse_length (triangle : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- Function to calculate the distance one triangle is slid along the hypotenuse of another -/
def slide_distance (triangle1 triangle2 : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- Function to calculate the area of overlap between two triangles -/
def area_of_overlap (triangle1 triangle2 : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- The area common to two congruent 45-45-90 triangles with hypotenuse 16 units,
    where one triangle is slid along the hypotenuse of the other by 6 units, is 25 square units. -/
theorem overlapping_triangles_area (triangle1 triangle2 : Set ℝ × Set ℝ) : 
  is_45_45_90_triangle triangle1 →
  is_45_45_90_triangle triangle2 →
  hypotenuse_length triangle1 = 16 →
  hypotenuse_length triangle2 = 16 →
  slide_distance triangle1 triangle2 = 6 →
  area_of_overlap triangle1 triangle2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_triangles_area_l1088_108815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1088_108822

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop := x^2 - a*x + b < 0

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * Real.sqrt (x - 3) + (b - 1) * Real.sqrt (4 - x)

-- Theorem statement
theorem max_value_of_f :
  ∀ a b : ℝ,
  (∀ x : ℝ, quadratic_inequality a b x ↔ 1 < x ∧ x < 2) →
  ∃ M : ℝ, M = Real.sqrt 5 ∧ ∀ x : ℝ, f a b x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1088_108822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_meeting_duration_l1088_108826

/-- The duration of the study period in minutes -/
def study_period : ℕ := 120

/-- The probability of two students meeting in the library -/
def meeting_probability : ℚ := 1/4

/-- Proves that the duration of each student's stay is 120 - 60√3 minutes,
    given the conditions of the problem -/
theorem library_meeting_duration :
  ∃ (n : ℝ) (d e f : ℕ+),
    (n = d - e * Real.sqrt (f : ℝ)) ∧
    (∀ m : ℕ+, f ≠ m ^ 2) ∧
    ((study_period - n) ^ 2 / study_period ^ 2 : ℝ) = (1 - meeting_probability) ∧
    n = 120 - 60 * Real.sqrt 3 :=
by sorry

#check library_meeting_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_meeting_duration_l1088_108826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_l1088_108858

/-- Represents a triangle with its properties -/
structure Triangle where
  isRight : Prop
  acuteAngle1 : ℝ
  acuteAngle2 : ℝ

/-- Represents the complementary relationship between two angles -/
def isComplementaryTo (a b : ℝ) : Prop := a + b = 90

/-- A triangle is right if and only if its two acute angles are complementary. -/
axiom right_triangle_iff_complementary_acute_angles (t : Triangle) :
  t.isRight ↔ isComplementaryTo t.acuteAngle1 t.acuteAngle2

/-- The inverse proposition of "In a right triangle, the two acute angles are complementary" -/
theorem inverse_proposition :
  (∀ t : Triangle, isComplementaryTo t.acuteAngle1 t.acuteAngle2 → t.isRight) ↔
  (∀ t : Triangle, t.isRight → isComplementaryTo t.acuteAngle1 t.acuteAngle2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_l1088_108858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_phase_l1088_108804

noncomputable def original_function (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def shifted_function (x φ : ℝ) : ℝ := original_function (x + Real.pi/6) φ

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem shifted_sine_even_phase (φ : ℝ) : 
  is_even (shifted_function · φ) → φ = Real.pi/6 ∨ ∃ k : ℤ, φ = k * Real.pi + Real.pi/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_even_phase_l1088_108804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1088_108819

open Real

-- Define the function f on the interval (-π/2, π/2)
noncomputable def f : ℝ → ℝ := sorry

-- Define the second derivative of f
noncomputable def f'' : ℝ → ℝ := sorry

-- State the given condition for x ∈ (0, π/2)
axiom f''_condition (x : ℝ) (hx : 0 < x ∧ x < π/2) :
  f'' x > sin (2*x) * f x - cos (2*x) * f'' x

-- Define a, b, and c
noncomputable def a : ℝ := f (π/3)
noncomputable def b : ℝ := 2 * f 0
noncomputable def c : ℝ := Real.sqrt 3 * f (π/6)

-- State the theorem to be proved
theorem f_inequality : a > c ∧ c > b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1088_108819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_digit_number_property_l1088_108838

theorem two_hundred_digit_number_property (N : ℕ) : 
  (∃ (a b c : ℕ) (k : ℕ), 
    N = a * 10^199 + b * 10^198 + c * 10^197 ∧ 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    k = 197 ∧
    N = 44 * (N / 10^199 * 10^198 + N % 10^198 / 10^197 * 10^197 + N % 10^197)) →
  ∃ (c : ℕ), c ∈ ({1, 2, 3} : Set ℕ) ∧ N = 132 * c * 10^197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_digit_number_property_l1088_108838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_l1088_108836

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  2 * Real.sin (ω * x) * Real.sin (ω * x + Real.pi / 3) + Real.cos (2 * ω * x) - 1 / 2

theorem f_two_zeros (ω : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧
    ∀ x ∈ Set.Icc 0 Real.pi, f ω x = 0 → x = x₁ ∨ x = x₂) ↔
  ω ∈ Set.Icc (11/12) (17/12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_l1088_108836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1088_108846

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem parallel_vectors_lambda (a b : V) (lambda : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_parallel : ∃ (k : ℝ), lambda • a + b = k • (a - 2 • b)) :
  lambda = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1088_108846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_frac_sum_eq_quarter_l1088_108835

/-- The Riemann zeta function for real x > 1 -/
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (n : ℝ) ^ (-x)

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The sum of fractional parts of zeta function values -/
noncomputable def zeta_frac_sum : ℝ := ∑' k : ℕ, frac (zeta (2 * (k + 2) - 1))

theorem zeta_frac_sum_eq_quarter :
  zeta_frac_sum = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_frac_sum_eq_quarter_l1088_108835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_into_50_l1088_108855

/-- Represents a square that can be divided into smaller congruent squares -/
structure DivisibleSquare where
  /-- Number of cuts needed to divide the square into 5 congruent squares -/
  cuts_for_5 : ℕ
  /-- Number of cuts needed to divide the square into 10 congruent squares -/
  cuts_for_10 : ℕ
  /-- Condition that 4 cuts result in 5 congruent squares -/
  h_cuts_5 : cuts_for_5 = 4
  /-- Condition that 6 cuts result in 10 congruent squares -/
  h_cuts_10 : cuts_for_10 = 6

/-- Predicate stating that a square can be divided into n congruent squares using c cuts -/
def CanBeDividedInto (s : DivisibleSquare) (n : ℕ) (c : ℕ) : Prop :=
  ∃ (method : ℕ → ℕ → Prop), method n c

/-- Theorem stating that a DivisibleSquare can be divided into 50 congruent squares with 10 cuts -/
theorem divide_into_50 (s : DivisibleSquare) : ∃ (cuts : ℕ), cuts = 10 ∧ 
  CanBeDividedInto s 50 cuts :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_into_50_l1088_108855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_decimal_part_first_two_digits_l1088_108817

theorem sqrt_decimal_part_first_two_digits
  (n : ℕ) (h : n > 100) :
  ∃ (k : ℕ) (ε : ℝ), 
    Real.sqrt (n^2 + 3*n + 1) = k + (1/2 + ε) ∧
    0 < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_decimal_part_first_two_digits_l1088_108817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_installment_plan_l1088_108848

/-- Represents the installment plan for an appliance purchase -/
structure InstallmentPlan where
  initialPrice : ℚ
  initialPayment : ℚ
  monthlyPayment : ℚ
  monthlyInterestRate : ℚ
  numPayments : ℕ

/-- Calculates the nth installment payment -/
def nthInstallment (plan : InstallmentPlan) (n : ℕ) : ℚ :=
  plan.monthlyPayment + (plan.initialPrice - plan.initialPayment - plan.monthlyPayment * (n - 1)) * plan.monthlyInterestRate

/-- Calculates the total cost of the appliance after all payments -/
def totalCost (plan : InstallmentPlan) : ℚ :=
  plan.initialPayment + plan.numPayments * plan.monthlyPayment + 
  (plan.initialPrice - plan.initialPayment) * plan.monthlyInterestRate * plan.numPayments -
  (plan.numPayments * (plan.numPayments - 1) / 2) * plan.monthlyPayment * plan.monthlyInterestRate

/-- Theorem stating the 10th installment payment and total cost for the given appliance purchase plan -/
theorem appliance_installment_plan :
  let plan : InstallmentPlan := {
    initialPrice := 1150
    initialPayment := 150
    monthlyPayment := 50
    monthlyInterestRate := 1/100
    numPayments := 20
  }
  (nthInstallment plan 10 = 111/2) ∧
  (totalCost plan = 1255) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_installment_plan_l1088_108848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_real_root_l1088_108869

/-- Given a cubic polynomial with real coefficients a and b, where -3 - 4i is a root,
    prove that 3 is the real root of the polynomial. -/
theorem cubic_real_root (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (fun x : ℂ ↦ a * x^3 + 4 * x^2 + b * x - 100) (-3 - 4 * Complex.I) = 0 →
  (fun x : ℝ ↦ a * x^3 + 4 * x^2 + b * x - 100) 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_real_root_l1088_108869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1088_108809

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1088_108809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_girls_probability_l1088_108882

/-- The probability of giving birth to a girl -/
noncomputable def prob_girl : ℝ := 1/2

/-- The number of pregnancies allowed -/
def num_pregnancies : ℕ := 2

/-- The probability of having all girls given the number of pregnancies -/
noncomputable def prob_all_girls (n : ℕ) : ℝ := prob_girl ^ n

theorem two_girls_probability :
  prob_all_girls num_pregnancies = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_girls_probability_l1088_108882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reciprocal_sum_bound_l1088_108800

open Set
open BigOperators

/-- A function that checks if all subset sums are distinct -/
def distinct_subset_sums (S : Finset ℕ) : Prop :=
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A.sum id ≠ B.sum id

/-- The main theorem -/
theorem max_reciprocal_sum_bound {n : ℕ} (hn : n ≥ 5) 
  (S : Finset ℕ) (hS : S.card = n) (hd : distinct_subset_sums S) :
  ∃ (max : ℚ), max = 2 - 1 / (2^(n-1)) ∧ 
  (∑ a in S, (1 : ℚ) / a) ≤ max ∧ 
  ∃ S', S'.card = n ∧ distinct_subset_sums S' ∧ ∑ a in S', (1 : ℚ) / a = max :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reciprocal_sum_bound_l1088_108800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l1088_108854

/-- The sum of the geometric series 1 - 1/3 + 1/9 - 1/27 + ... -/
noncomputable def S1 : ℝ := 3/4

/-- The sum of the geometric series 1 + 1/2 - 1/8 + 1/32 - ... -/
noncomputable def S2 : ℝ := 4/5

/-- The equation that defines y -/
def equation (y : ℝ) : Prop :=
  S1 * S2 = 1 + (1/y) + (1/y^2) + (1/y^3) + (1/y^4) + (1/y^5) + (1/y^6) + (1/y^7) + (1/y^8) + (1/y^9)

/-- Theorem stating that there exists a y that satisfies the equation and equals 3/2 -/
theorem y_value : ∃ y : ℝ, equation y ∧ y = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l1088_108854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approx_20000_l1088_108863

/-- Represents the growth rate of the population per year -/
noncomputable def growth_rate : ℝ := 1.10

/-- Represents the population after 3 years -/
noncomputable def final_population : ℝ := 26620

/-- Represents the initial population of the town -/
noncomputable def initial_population : ℝ := final_population / (growth_rate ^ 3)

/-- Theorem stating that the initial population is approximately 20000 -/
theorem initial_population_approx_20000 :
  |initial_population - 20000| < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approx_20000_l1088_108863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l1088_108892

/-- Represents the completion of a project by a team -/
structure ProjectCompletion where
  initial_team : ℕ
  initial_days : ℕ
  initial_completion : ℚ
  additional_team : ℕ

/-- Calculates the total days to complete a project given the initial conditions and additional team members -/
def total_completion_days (p : ProjectCompletion) : ℚ :=
  p.initial_days + (1 - p.initial_completion) / ((p.initial_team + p.additional_team : ℚ) * (p.initial_completion / p.initial_days) / p.initial_team)

/-- The theorem stating that under the given conditions, the project takes 40 days to complete -/
theorem project_completion_time (p : ProjectCompletion) 
  (h1 : p.initial_team = 8)
  (h2 : p.initial_days = 30)
  (h3 : p.initial_completion = 2/3)
  (h4 : p.additional_team = 4) :
  total_completion_days p = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_l1088_108892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1088_108825

noncomputable section

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the focal points
def focal_point_1 : ℝ × ℝ := (-1, 0)
def focal_point_2 : ℝ × ℝ := (1, 0)

-- Define the point P that the ellipse passes through
def point_P : ℝ × ℝ := (4/3, 1/3)

-- Theorem statement
theorem ellipse_equation (a b : ℝ) :
  ellipse_C a b (point_P.1) (point_P.2) ∧
  (∀ x y, ellipse_C a b x y → 
    Real.sqrt ((x - focal_point_1.1)^2 + (y - focal_point_1.2)^2) +
    Real.sqrt ((x - focal_point_2.1)^2 + (y - focal_point_2.2)^2) = 2*a) →
  (∀ x y, x^2 / 2 + y^2 = 1 ↔ ellipse_C (Real.sqrt 2) 1 x y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1088_108825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l1088_108872

/-- A geometric sequence with first term a, common ratio r, and index n -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

/-- The fourth term of the specific geometric sequence -/
def fourth_term : ℝ := -27

theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  (∃ r : ℝ, r ≠ 0 ∧
    geometric_sequence (2 * x) r 1 = 2 * x ∧
    geometric_sequence (2 * x) r 2 = 4 * x + 4 ∧
    geometric_sequence (2 * x) r 3 = 6 * x + 6) →
  geometric_sequence (2 * x) (3/2) 4 = fourth_term := by
  sorry

#check geometric_sequence_fourth_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_fourth_term_l1088_108872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_art_supplies_l1088_108823

/-- Calculates the additional money Albert needs to buy art supplies -/
theorem albert_art_supplies (paintbrush_cost canvas_cost easel_cost paints_cost palette_cost discount_rate tax_rate albert_money : ℚ) 
  (h1 : paintbrush_cost = 15/10)
  (h2 : canvas_cost = 795/100)
  (h3 : easel_cost = 1265/100)
  (h4 : paints_cost = 435/100)
  (h5 : palette_cost = 375/100)
  (h6 : discount_rate = 1/10)
  (h7 : tax_rate = 1/20)
  (h8 : albert_money = 15) :
  (let total_cost := paintbrush_cost + canvas_cost + easel_cost + paints_cost + palette_cost
   let discounted_cost := total_cost * (1 - discount_rate)
   let final_cost := discounted_cost * (1 + tax_rate)
   let additional_money := final_cost - albert_money
   additional_money) = 677/50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_albert_art_supplies_l1088_108823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_sum_proof_l1088_108845

/-- The sum of a geometric sequence with first term a, common ratio r, and n terms -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The number of components manufactured in the first month -/
def initial_production : ℕ := 8

/-- The ratio by which production increases each month -/
def monthly_increase_ratio : ℕ := 3

/-- The number of months in the production period -/
def production_period : ℕ := 4

/-- The total number of components manufactured over the production period -/
def total_production : ℕ := 320

theorem production_sum_proof :
  geometric_sum (initial_production : ℝ) (monthly_increase_ratio : ℝ) production_period = total_production := by
  sorry

#eval total_production

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_sum_proof_l1088_108845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_tangent_l1088_108865

noncomputable def parabola (x : ℝ) : ℝ := x^2

noncomputable def tangent_line (x : ℝ) : ℝ := 2*x - 1

noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), parabola x - ∫ x in (0)..(1/2), tangent_line x

theorem area_enclosed_by_parabola_and_tangent : enclosed_area = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_tangent_l1088_108865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_range_correct_l1088_108878

/-- Represents the properties of a road roller and the road to be compressed -/
structure RoadRoller where
  strip_width : ℝ
  overlap_ratio : ℝ
  road_length : ℝ
  road_width : ℝ
  compression_count : ℕ
  min_time : ℝ
  max_time : ℝ

/-- Calculates the range of speeds required for the road roller to complete its task -/
noncomputable def calculate_speed_range (roller : RoadRoller) : Set ℝ :=
  let effective_width := roller.strip_width * (1 - roller.overlap_ratio)
  let passes := Int.ceil (roller.road_width / effective_width)
  let total_distance := (passes : ℝ) * roller.compression_count * 2 * roller.road_length
  let min_speed := total_distance / roller.max_time
  let max_speed := total_distance / roller.min_time
  { x | min_speed ≤ x ∧ x ≤ max_speed }

/-- Theorem stating that the calculated speed range for the given parameters is correct -/
theorem speed_range_correct (roller : RoadRoller) :
  roller.strip_width = 0.85 ∧
  roller.overlap_ratio = 1/4 ∧
  roller.road_length = 750 ∧
  roller.road_width = 6.5 ∧
  roller.compression_count = 2 ∧
  roller.min_time = 5 ∧
  roller.max_time = 6 →
  ∀ x ∈ calculate_speed_range roller, 2.75 ≤ x ∧ x ≤ 3.3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_range_correct_l1088_108878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1088_108839

/-- The function f(x) = x^2 - x - x ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x - x * Real.log x

/-- The condition that f(x) is non-negative for all positive x -/
axiom f_nonneg : ∀ x > 0, f x ≥ 0

theorem f_properties :
  ∃ x₀ : ℝ,
    (∀ x : ℝ, f x ≤ f x₀) ∧
    (∃! x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
    (Real.exp (-2) < f x₀ ∧ f x₀ < (1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1088_108839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_check_l1088_108897

noncomputable section

/-- A function representing the relationship between x and y --/
def Relationship := ℝ → ℝ

/-- Checks if a relationship is directly proportional --/
def is_directly_proportional (f : Relationship) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- Checks if a relationship is inversely proportional --/
def is_inversely_proportional (f : Relationship) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- Equation A: x^2 + 4y = 16 --/
noncomputable def equation_A : Relationship := λ x => (16 - x^2) / 4

/-- Equation B: 2xy = 8 --/
noncomputable def equation_B : Relationship := λ x => 4 / x

/-- Equation C: x = 3y --/
noncomputable def equation_C : Relationship := λ x => x / 3

/-- Equation D: 4x + 5y = 20 --/
noncomputable def equation_D : Relationship := λ x => (20 - 4*x) / 5

/-- Equation E: x/y = 4 --/
noncomputable def equation_E : Relationship := λ x => x / 4

theorem proportionality_check :
  (¬ is_directly_proportional equation_A ∧ ¬ is_inversely_proportional equation_A) ∧
  (¬ is_directly_proportional equation_D ∧ ¬ is_inversely_proportional equation_D) ∧
  (is_inversely_proportional equation_B ∨ is_directly_proportional equation_B) ∧
  (is_directly_proportional equation_C) ∧
  (is_directly_proportional equation_E) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_check_l1088_108897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_is_ten_l1088_108832

/-- The number of persons in the group -/
def N : ℕ := sorry

/-- The total age of the group before replacement -/
def T : ℕ := sorry

/-- The average age decrease after replacement -/
def age_decrease : ℝ := 3

/-- The age of the person who left the group -/
def age_left : ℕ := 46

/-- The age of the person who joined the group -/
def age_joined : ℕ := 16

/-- Theorem stating that the number of persons in the group is 10 -/
theorem group_size_is_ten :
  (T / N : ℝ) - age_decrease = ((T - age_left + age_joined) / N : ℝ) →
  N = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_is_ten_l1088_108832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l1088_108827

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

-- Define the minimum value function
noncomputable def min_value (a : ℝ) : ℝ :=
  if a ≤ 1 then -a^2 - 4 else -a^2 - 4 * a

-- Theorem statement
theorem min_value_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 0 1, f a x ≥ min_value a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l1088_108827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_numbers_with_two_sums_l1088_108852

/-- Given 10 natural numbers, we can determine them uniquely with two specific sums -/
theorem determine_numbers_with_two_sums 
  (x : Fin 10 → ℕ) 
  (S : ℕ) 
  (n : ℕ) 
  (S' : ℕ) 
  (h_S : S = (Finset.univ.sum x))
  (h_n : 10^n > S)
  (h_S' : S' = (Finset.univ.sum (λ i => x i * 10^(n * i.val)))) :
  ∃! (y : Fin 10 → ℕ), 
    (Finset.univ.sum y = S) ∧ 
    (Finset.univ.sum (λ i => y i * 10^(n * i.val)) = S') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_numbers_with_two_sums_l1088_108852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_value_determinable_l1088_108886

structure CircleSystem where
  values : Fin 6 → ℝ
  segments : Fin 6 → ℝ
  sum_property : ∀ i : Fin 6, values i = segments i + segments ((i + 5) % 6)

theorem circle_value_determinable (cs : CircleSystem) (k : Fin 6) :
  ∃ f : (Fin 6 → ℝ) → ℝ,
    cs.values k = f (λ i ↦ if i = k then 0 else cs.values i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_value_determinable_l1088_108886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_regions_bound_l1088_108881

/-- A type representing a line in a plane. -/
def Line (Plane : Type) : Type := sorry

/-- A type representing a region in a plane. -/
def Region (Plane : Type) : Type := sorry

/-- A function that determines if two regions are adjacent. -/
def adjacent (Plane : Type) (r₁ r₂ : Region Plane) : Prop := sorry

/-- Given n lines (n ≥ 2) in a plane dividing it into regions, with some regions colored
    such that no two colored regions share a common boundary, the number of colored regions
    is at most 1/3(n^2 + n). -/
theorem colored_regions_bound (n : ℕ) (h : n ≥ 2) :
  ∀ (Plane : Type) (lines : Finset (Line Plane))
    (colored_regions : Finset (Region Plane)),
    lines.card = n →
    (∀ r₁ r₂, r₁ ∈ colored_regions → r₂ ∈ colored_regions → r₁ ≠ r₂ → ¬(adjacent Plane r₁ r₂)) →
    colored_regions.card ≤ (n^2 + n) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_regions_bound_l1088_108881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1088_108816

/-- The projection matrix onto a vector (a, b) -/
noncomputable def projectionMatrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a^2 + b^2)) • Matrix.of (λ i j =>
    if i = 0 && j = 0 then a^2
    else if i = 0 && j = 1 then a*b
    else if i = 1 && j = 0 then a*b
    else b^2)

/-- The theorem stating that the determinant of the projection matrix onto (3, -5) is 0 -/
theorem det_projection_matrix_zero :
  let Q := projectionMatrix 3 (-5)
  Matrix.det Q = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_zero_l1088_108816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1088_108830

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 5 = 1

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the given points
def point_P : ℝ × ℝ := (3, 15/4)
def point_Q : ℝ × ℝ := (-16/3, 5)
def point_R : ℝ × ℝ := (2, -2)

-- Define the asymptote hyperbola
def asymptote_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), ellipse x y → hyperbola a b x y) ∧
    (hyperbola a b point_P.1 point_P.2) ∧
    (hyperbola a b point_Q.1 point_Q.2) ∧
    (hyperbola a b point_R.1 point_R.2) ∧
    (∃ (k : ℝ), ∀ (x y : ℝ), asymptote_hyperbola x y ↔ hyperbola a b (k*x) (k*y)) ∧
    ((a = Real.sqrt 3 ∧ b = Real.sqrt 5) ∨ (a = 4 ∧ b = 3) ∨ (a = 2 ∧ b = Real.sqrt 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1088_108830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_non_monotonic_equals_one_third_l1088_108883

open Real MeasureTheory

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a|

-- Define the set of a values that satisfy the condition
def A : Set ℝ := {a | 0 < a ∧ a < 6}

-- Define the set of a values where the function is non-monotonic in (1,2)
def B : Set ℝ := {a | 2 < a ∧ a < 4}

-- State the theorem
theorem probability_non_monotonic_equals_one_third :
  (volume B) / (volume A) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_non_monotonic_equals_one_third_l1088_108883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_cube_with_cylinders_removed_l1088_108805

/-- The remaining volume of a cube after removing two cylindrical sections -/
theorem remaining_volume_cube_with_cylinders_removed (π : ℝ) :
  (let cube_side : ℝ := 6
   let cylinder_radius : ℝ := 1
   let cylinder_height : ℝ := cube_side
   let cube_volume : ℝ := cube_side ^ 3
   let cylinder_volume : ℝ := π * cylinder_radius ^ 2 * cylinder_height
   let total_removed_volume : ℝ := 2 * cylinder_volume
   let remaining_volume : ℝ := cube_volume - total_removed_volume
   remaining_volume) = 216 - 12 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_cube_with_cylinders_removed_l1088_108805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_points_l1088_108824

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_equidistant_points :
  ∀ A : ℝ × ℝ,
  parabola A.1 A.2 →
  distance A focus = distance B focus →
  distance A B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_points_l1088_108824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1088_108890

/-- The general term of the sequence -/
def a (n : ℕ) (lambda : ℝ) : ℝ := n^2 + lambda*n - 2018

/-- The sequence is monotonically increasing -/
def is_monotone_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ, a n lambda < a (n + 1) lambda

theorem lambda_range (lambda : ℝ) :
  is_monotone_increasing lambda → lambda > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1088_108890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_between_1_and_20_l1088_108813

def isPrime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def sumOfPrimesBetween (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).map (· + a)
    |>.filter isPrime
    |>.sum

theorem sum_of_primes_between_1_and_20 :
  sumOfPrimesBetween 1 20 = 77 := by
  -- Evaluate the function
  have h : sumOfPrimesBetween 1 20 = 77 := by native_decide
  -- Use the result
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_primes_between_1_and_20_l1088_108813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_symmetry_l1088_108814

/-- A cubic function with a non-zero cubic term -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_nonzero : a ≠ 0

/-- The value of a cubic function at a given point -/
def CubicFunction.eval (f : CubicFunction) (x : ℝ) : ℝ :=
  f.a * x^3 + f.b * x^2 + f.c * x + f.d

/-- The first derivative of a cubic function -/
def CubicFunction.deriv (f : CubicFunction) (x : ℝ) : ℝ :=
  3 * f.a * x^2 + 2 * f.b * x + f.c

/-- The second derivative of a cubic function -/
def CubicFunction.secondDeriv (f : CubicFunction) (x : ℝ) : ℝ :=
  6 * f.a * x + 2 * f.b

/-- The inflection point of a cubic function -/
noncomputable def CubicFunction.inflectionPoint (f : CubicFunction) : ℝ × ℝ :=
  let x := -f.b / (3 * f.a)
  (x, f.eval x)

/-- Theorem: The inflection point is the center of symmetry for any cubic function -/
theorem cubic_symmetry (f : CubicFunction) (r : ℝ) : 
  let (p, q) := f.inflectionPoint
  f.eval (p + r) + f.eval (p - r) = 2 * q := by
  sorry

#check cubic_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_symmetry_l1088_108814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_ellipse_is_ellipse_equation_curve_is_ellipse_l1088_108873

-- Define the parametric equations for x and y
noncomputable def x (t : ℝ) : ℝ := Real.cos t + Real.sin t
noncomputable def y (t : ℝ) : ℝ := 4 * (Real.cos t - Real.sin t)

-- Theorem stating that the points (x(t), y(t)) lie on an ellipse
theorem points_on_ellipse :
  ∀ t : ℝ, (x t)^2 / 2 + (y t)^2 / 32 = 1 := by
  sorry

-- Define what it means for a point to be on an ellipse
def IsOnEllipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Theorem stating that the equation represents an ellipse
theorem is_ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x y : ℝ, IsOnEllipse x y a b → ∃ t : ℝ, x = a * Real.cos t ∧ y = b * Real.sin t := by
  sorry

-- Main theorem combining the above results
theorem curve_is_ellipse :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ t : ℝ, IsOnEllipse (x t) (y t) a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_ellipse_is_ellipse_equation_curve_is_ellipse_l1088_108873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l1088_108847

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The left focus of the hyperbola is at (-3, 0) -/
def left_focus : ℝ × ℝ := (-3, 0)

/-- The equation of the asymptotes is y = ±√2 x -/
noncomputable def asymptote_slope : ℝ := Real.sqrt 2

/-- Theorem stating the value of a for the given hyperbola -/
theorem hyperbola_a_value (h : Hyperbola) 
  (h_focus : left_focus = (-3, 0))
  (h_asymptote : h.b / h.a = asymptote_slope) : 
  h.a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l1088_108847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l1088_108866

noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference ^ 2 * height) / (4 * Real.pi)

theorem tank_capacity_ratio :
  let tank_a_volume := cylinder_volume 7 8
  let tank_b_volume := cylinder_volume 8 10
  (tank_a_volume / tank_b_volume) = 0.56 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l1088_108866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_eq_149_l1088_108899

def g (x y : ℕ+) : ℕ := sorry

axiom g_self (x : ℕ+) : g x x = x.val ^ 2

axiom g_comm (x y : ℕ+) : g x y = g y x

axiom g_prop (x y : ℕ+) : (x.val + y.val) * g x y = y.val * g x (x + y)

theorem g_sum_eq_149 : g 2 12 + g 5 25 = 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_eq_149_l1088_108899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1088_108891

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : a 0 > 0) (h_geom : is_geometric_sequence a q) :
  (∀ n : ℕ, a (2*n - 1) + a (2*n) < 0) → 
  (q < 0) ∧ 
  ∃ a₀ q₀, a₀ > 0 ∧ q₀ < 0 ∧ 
    is_geometric_sequence (λ n ↦ a₀ * q₀^n) q₀ ∧ 
    ∃ n : ℕ, (λ n ↦ a₀ * q₀^n) (2*n - 1) + (λ n ↦ a₀ * q₀^n) (2*n) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l1088_108891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_schools_count_l1088_108803

/-- Represents a student in the math contest -/
structure Student where
  name : String
  score : ℕ
  rank : ℕ

/-- Represents a school team in the math contest -/
structure Team where
  members : Fin 4 → Student

/-- The math contest -/
structure MathContest where
  schools : ℕ
  teams : Fin schools → Team
  allStudents : Fin (4 * schools) → Student

theorem math_contest_schools_count
  (contest : MathContest)
  (andrea beth carla dan : Student)
  (h1 : ∀ s1 s2 : Student, s1.score = s2.score → s1 = s2)
  (h2 : ∃ t : Fin contest.schools, 
        (contest.teams t).members 0 = andrea ∧ 
        (contest.teams t).members 1 = beth ∧ 
        (contest.teams t).members 2 = carla ∧ 
        (contest.teams t).members 3 = dan)
  (h3 : ∀ (t : Fin contest.schools) (i : Fin 4), andrea.score ≥ ((contest.teams t).members i).score)
  (h4 : beth.rank = 42)
  (h5 : carla.rank = 75)
  (h6 : dan.rank = 95)
  (h7 : ∃ k : ℕ, k = (4 * contest.schools - 1) / 2 ∧ andrea.rank = k + 1)
  : contest.schools = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_contest_schools_count_l1088_108803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_multiple_l1088_108874

def is_valid_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  ∃ (a b c d : Nat), a ∈ ({3, 4, 6, 8} : Set Nat) ∧ 
                     b ∈ ({3, 4, 6, 8} : Set Nat) ∧ 
                     c ∈ ({3, 4, 6, 8} : Set Nat) ∧ 
                     d ∈ ({3, 4, 6, 8} : Set Nat) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = 1000 * a + 100 * b + 10 * c + d

theorem no_valid_multiple : 
  ¬∃ (a b : Nat), is_valid_number a ∧ is_valid_number b ∧ a ≠ b ∧ ∃ (k : Nat), k > 1 ∧ a = k * b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_multiple_l1088_108874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_strategy_correct_l1088_108898

/-- Represents the state of the lamp in the room -/
inductive LampState
| On
| Off

/-- Represents a prisoner's role -/
inductive PrisonerRole
| Counter
| Regular

/-- Represents the action a prisoner can take -/
inductive PrisonerAction
| TurnOn
| TurnOff
| NoAction

/-- The prison escape problem setup -/
structure PrisonEscape where
  totalPrisoners : Nat
  counterCount : Nat
  lampState : LampState

/-- Defines the action a prisoner takes based on their role and the current state -/
def prisonerAction (role : PrisonerRole) (state : PrisonEscape) : PrisonerAction :=
  match role, state.lampState with
  | PrisonerRole.Regular, LampState.Off => PrisonerAction.TurnOn
  | PrisonerRole.Counter, LampState.On => PrisonerAction.TurnOff
  | _, _ => PrisonerAction.NoAction

/-- Updates the prison state based on a prisoner's action -/
def updateState (state : PrisonEscape) (action : PrisonerAction) : PrisonEscape :=
  match action with
  | PrisonerAction.TurnOn => { state with lampState := LampState.On }
  | PrisonerAction.TurnOff => { state with 
      lampState := LampState.Off, 
      counterCount := state.counterCount + 1 
    }
  | PrisonerAction.NoAction => state

/-- Theorem: The counting strategy correctly determines when all prisoners have visited the room -/
theorem counting_strategy_correct (initialState : PrisonEscape) :
  initialState.counterCount = initialState.totalPrisoners - 1 →
  ∀ (role : PrisonerRole), 
    let action := prisonerAction role initialState
    let newState := updateState initialState action
    newState.counterCount = initialState.totalPrisoners - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counting_strategy_correct_l1088_108898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_theorem_l1088_108812

/-- Define the ⊛ operation for real numbers -/
def heartsuit (x y : ℝ) : ℝ := |x - y|

/-- Theorem stating the equivalence of the expression and its value for specific inputs -/
theorem heartsuit_theorem (a b : ℝ) :
  (heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b))) = abs (abs (a - b) - 2 * abs (a - b)) ∧
  (a = 3 ∧ b = -1) → (heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b))) = 4 := by
  sorry

#check heartsuit_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heartsuit_theorem_l1088_108812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1088_108853

theorem indefinite_integral_proof :
  ∀ x : ℝ, 
    Real.sin (2 * x) * (4 * x - 2) = 
    (deriv (λ t ↦ (2 * t - 1) * Real.cos (2 * t) - Real.sin (2 * t))) x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1088_108853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1088_108896

/-- Triangle PQR with point N on side QR -/
structure Triangle (P Q R N : ℝ × ℝ) : Prop where
  is_midpoint : N = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem triangle_side_length 
  (P Q R N : ℝ × ℝ) 
  (h : Triangle P Q R N) 
  (pq_length : distance P Q = 5)
  (pr_length : distance P R = 10)
  (pn_length : distance P N = 6) :
  distance Q R = Real.sqrt 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1088_108896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angle_l1088_108818

theorem cos_sum_special_angle (θ : Real) 
  (h1 : Real.cos θ = -12/13) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.cos (θ + π/4) = -7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angle_l1088_108818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_rate_calculation_l1088_108885

/-- Represents a cellular phone plan -/
structure CellularPlan where
  baseFee : ℚ
  baseMinutes : ℚ
  overageRate : ℚ

/-- Calculates the cost of a plan for a given number of minutes -/
def planCost (plan : CellularPlan) (minutes : ℚ) : ℚ :=
  plan.baseFee + max 0 (minutes - plan.baseMinutes) * plan.overageRate

/-- The problem statement -/
theorem phone_plan_rate_calculation 
  (plan1 : CellularPlan)
  (plan2 : CellularPlan)
  (h1 : plan1.baseFee = 50)
  (h2 : plan1.baseMinutes = 500)
  (h3 : plan2.baseFee = 75)
  (h4 : plan2.baseMinutes = 1000)
  (h5 : plan2.overageRate = 45/100)
  (h6 : planCost plan1 2500 = planCost plan2 2500) :
  plan1.overageRate = 35/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_plan_rate_calculation_l1088_108885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_eq_10_l1088_108894

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_BAD_eq_120 (q : Quadrilateral) : Prop :=
  sorry -- We need to define angle measurement

def angle_BCD_eq_120 (q : Quadrilateral) : Prop :=
  sorry -- We need to define angle measurement

def BC_eq_10 (q : Quadrilateral) : Prop :=
  dist q.B q.C = 10

def CD_eq_10 (q : Quadrilateral) : Prop :=
  dist q.C q.D = 10

-- Theorem statement
theorem AC_eq_10 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle_BAD : angle_BAD_eq_120 q)
  (h_angle_BCD : angle_BCD_eq_120 q)
  (h_BC : BC_eq_10 q)
  (h_CD : CD_eq_10 q) :
  dist q.A q.C = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_eq_10_l1088_108894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_correct_l1088_108893

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ - Real.sin θ, 2 * Real.sin θ + Real.cos θ)

-- Define the intersection points
def intersection_C₁_C₃ : Set (ℝ × ℝ) := {(1/2, 1), (1, 2)}
def intersection_C₂_C₃ : Set (ℝ × ℝ) := {(-1/2, -1), (-1, -2)}

-- Theorem statement
theorem intersection_points_correct :
  (∀ p ∈ intersection_C₁_C₃, ∃ t θ, C₁ t = p ∧ C₃ θ = p) ∧
  (∀ p ∈ intersection_C₂_C₃, ∃ s θ, C₂ s = p ∧ C₃ θ = p) := by
  sorry

#check intersection_points_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_correct_l1088_108893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_integer_l1088_108849

theorem not_both_integer (m n : ℕ+) (h : Nat.Coprime m n) :
  ¬(∃ (a b : ℤ), (a : ℚ) = (n^4 + m : ℚ) / (m^2 + n^2 : ℚ) ∧
                  (b : ℚ) = (n^4 - m : ℚ) / (m^2 - n^2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_integer_l1088_108849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l1088_108895

/-- Represents the time required to complete a project given the total work,
    number of workers, and their efficiency rate. -/
noncomputable def project_completion_time (total_work : ℝ) (num_workers : ℝ) (efficiency_rate : ℝ) : ℝ :=
  total_work / (num_workers * efficiency_rate)

/-- Proves that a project requiring 12 man-days of work will take 1 day for 6 men
    with an efficiency rate of 2.0 man-days per man per day to complete. -/
theorem project_completion_theorem :
  let total_work : ℝ := 12
  let num_workers : ℝ := 6
  let efficiency_rate : ℝ := 2
  project_completion_time total_work num_workers efficiency_rate = 1 := by
  -- Unfold the definition of project_completion_time
  unfold project_completion_time
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

#check project_completion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l1088_108895
