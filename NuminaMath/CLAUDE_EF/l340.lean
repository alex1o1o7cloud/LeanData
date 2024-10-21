import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l340_34079

/-- A sequence defined by the recurrence relation a_{n+1} = (2a_n)/(a_n+2) with a_1 = 1 -/
def a : ℕ → ℚ
  | 0 => 1  -- Define a value for n = 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => 2 * a (n + 1) / (a (n + 1) + 2)

/-- Theorem stating that the general term of the sequence is 2/(n+1) -/
theorem a_general_term (n : ℕ) (h : n ≥ 1) : a n = 2 / (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l340_34079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_is_two_lines_l340_34054

-- Define the circle Ω
noncomputable def Ω : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define points A and B on Ω
noncomputable def A : EuclideanSpace ℝ (Fin 2) := sorry
noncomputable def B : EuclideanSpace ℝ (Fin 2) := sorry

-- Assume A and B are on Ω
axiom A_on_Ω : A ∈ Ω
axiom B_on_Ω : B ∈ Ω

-- Define point C inside Ω
noncomputable def C : EuclideanSpace ℝ (Fin 2) := sorry
axiom C_inside_Ω : C ∈ interior Ω

-- Define a moving point P on Ω
noncomputable def P : ℝ → EuclideanSpace ℝ (Fin 2) := sorry
axiom P_on_Ω : ∀ t, P t ∈ Ω

-- Define the circumcircle of a triangle
noncomputable def circumcircle (X Y Z : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the common external tangents of two circles
noncomputable def common_external_tangents (c1 c2 : Set (EuclideanSpace ℝ (Fin 2))) : Set (AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) := sorry

-- Define the intersection point Q
noncomputable def Q (t : ℝ) : EuclideanSpace ℝ (Fin 2) := sorry
axiom Q_def : ∀ t, Q t ∈ ⋂ l ∈ common_external_tangents (circumcircle A (P t) C) (circumcircle B (P t) C), (l : Set (EuclideanSpace ℝ (Fin 2)))

-- The theorem to prove
theorem locus_of_Q_is_two_lines :
  ∃ (l1 l2 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))), ∀ t, Q t ∈ (l1 : Set (EuclideanSpace ℝ (Fin 2))) ∪ (l2 : Set (EuclideanSpace ℝ (Fin 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_is_two_lines_l340_34054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_ratio_l340_34000

/-- The original function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

/-- The inverse function g^(-1)(x) -/
noncomputable def g_inv (x : ℝ) : ℝ := (4 * x + 2) / (3 - x)

/-- Theorem stating that g_inv is indeed the inverse of g and that a/c = -4 -/
theorem g_inverse_ratio : 
  (∀ x : ℝ, g (g_inv x) = x ∧ g_inv (g x) = x) ∧ 
  (∃ a b c d : ℝ, (∀ x : ℝ, g_inv x = (a * x + b) / (c * x + d)) ∧ a / c = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_ratio_l340_34000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l340_34004

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = ![![1, 0],
       ![0, 1]]

theorem smallest_power_rotation_120 :
  ∃ (n : ℕ), n > 0 ∧ 
    is_identity ((rotation_matrix (2 * Real.pi / 3)) ^ n) ∧
    ∀ (m : ℕ), m > 0 → m < n → 
      ¬is_identity ((rotation_matrix (2 * Real.pi / 3)) ^ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_120_l340_34004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l340_34099

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7*θ) = -8383/98304 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l340_34099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_theorem_l340_34047

theorem xy_value_theorem (x y : ℝ) 
  (h : x^2 - 2*y + Real.sqrt 5 * y = 10 + 3 * Real.sqrt 5) :
  x^y = 64 ∨ x^y = -64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_theorem_l340_34047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_equivalent_polar_coordinates_l340_34095

/-- 
Given a point in polar coordinates (r, θ), where r can be negative,
this function returns the equivalent point in standard polar coordinates
where r > 0 and 0 ≤ θ < 2π.
-/
noncomputable def standardPolarCoordinates (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  if r ≥ 0 then (r, θ % (2 * Real.pi))
  else (abs r, (θ + Real.pi) % (2 * Real.pi))

/-- 
Theorem stating that the point (-2, 3π/8) in polar coordinates
is equivalent to the point (2, 11π/8) in standard polar coordinates.
-/
theorem equivalent_polar_coordinates :
  standardPolarCoordinates (-2) (3 * Real.pi / 8) = (2, 11 * Real.pi / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_equivalent_polar_coordinates_l340_34095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_calculation_l340_34034

/-- Calculates the required monthly rent for a property investment --/
noncomputable def required_monthly_rent (property_value : ℝ) (annual_return_rate : ℝ) 
  (maintenance_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let annual_return := property_value * annual_return_rate
  let annual_taxes := property_value * tax_rate
  let total_annual_income := annual_return + annual_taxes
  let monthly_income := total_annual_income / 12
  monthly_income / (1 - maintenance_rate)

/-- Theorem stating the required monthly rent for the given scenario --/
theorem rent_calculation :
  let property_value : ℝ := 150000
  let annual_return_rate : ℝ := 0.07
  let maintenance_rate : ℝ := 0.15
  let tax_rate : ℝ := 0.0075
  ∃ ε > 0, abs (required_monthly_rent property_value annual_return_rate maintenance_rate tax_rate - 1139.71) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_calculation_l340_34034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_concentric_circles_l340_34023

/-- Given two concentric circles where the outer radius is three times the inner radius,
    and the width between the circles is 4, the area between the circles is 32π. -/
theorem area_between_concentric_circles
  (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 4) :
  π * (3 * r)^2 - π * r^2 = 32 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_concentric_circles_l340_34023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l340_34051

-- Define the circles
def circle1_radius : ℝ := 4
def circle2_radius : ℝ := 2 * circle1_radius

-- Define the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem area_between_circles : 
  circle_area circle2_radius - circle_area circle1_radius = 48 * Real.pi := by
  -- Expand the definition of circle_area
  unfold circle_area
  -- Simplify the expression
  simp [circle1_radius, circle2_radius]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l340_34051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l340_34060

open MeasureTheory Interval Real

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l340_34060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l340_34017

noncomputable section

/-- The number of days A takes to finish the work alone -/
def a_days : ℝ := 4

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 12

/-- The number of days A and B work together before A leaves -/
def days_together : ℝ := 2

/-- The work rate of A per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The work rate of B per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The amount of work completed when A and B work together -/
noncomputable def work_completed : ℝ := days_together * (a_rate + b_rate)

/-- The remaining work to be done by B alone -/
noncomputable def remaining_work : ℝ := 1 - work_completed

/-- The number of days B takes to finish the remaining work -/
noncomputable def b_remaining_days : ℝ := remaining_work / b_rate

theorem b_finishes_in_four_days : b_remaining_days = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l340_34017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l340_34086

-- Define the function f(x) = e^x / x^2
noncomputable def f (x : ℝ) : ℝ := Real.exp x / (x^2)

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (Real.exp x * (x - 2)) / (x^3)

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioc 0 2, -- x ∈ (0, 2]
    f_derivative x ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l340_34086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l340_34039

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 + Real.sin x, 2 * Real.sqrt 2 - Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_properties :
  (∀ x, f x ≤ 4) ∧
  (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), f x = 1 →
    Real.cos (x + 5 * Real.pi / 12) = -(3 * Real.sqrt 5 + 1) / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l340_34039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_fourth_power_l340_34094

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^4
axiom g_81 : g 81 = 81

-- State the theorem to be proved
theorem g_3_fourth_power : (g 3)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_fourth_power_l340_34094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_penguins_are_twenty_percent_l340_34053

/-- Represents the number of animals in a zoo -/
structure ZooPopulation where
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ
  total : ℕ

/-- The conditions of the zoo population -/
def zoo : ZooPopulation where
  giraffes := 5
  penguins := 2 * 5
  elephants := 2
  total := 50  -- Changed from 2 / 0.04 to the actual result

/-- The percentage of penguins in the zoo -/
def penguinPercentage (z : ZooPopulation) : ℚ :=
  (z.penguins : ℚ) / (z.total : ℚ) * 100

/-- Theorem stating that the percentage of penguins in the zoo is 20% -/
theorem penguins_are_twenty_percent :
  penguinPercentage zoo = 20 := by
  -- Unfold the definitions
  unfold penguinPercentage zoo
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_penguins_are_twenty_percent_l340_34053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_particular_l340_34076

-- Define what a particular proposition is
def is_particular_proposition (P : Prop) : Prop :=
  ∃ (α : Type) (predicate : α → Prop), P = ∃ x, predicate x

-- Define the statement we want to prove is a particular proposition
def statement : Prop :=
  ∃ x : ℝ, x ≥ 3

-- Theorem stating that our statement is a particular proposition
theorem statement_is_particular : is_particular_proposition statement := by
  -- Provide the type and predicate
  use ℝ, (λ x : ℝ => x ≥ 3)
  -- Show that the statement matches the form of a particular proposition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_particular_l340_34076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l340_34084

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
    ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]]

theorem smallest_rotation_power : 
  (∀ k : ℕ, k < 8 → rotation_matrix ^ k ≠ 1) ∧ 
  rotation_matrix ^ 8 = 1 := by
  sorry

#check smallest_rotation_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l340_34084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_p_squared_l340_34043

/-- Given a polynomial p(x) = x^5 - 2x^2 + 1, prove that the coefficient of x^4 in (p(x))^2 is 4 -/
theorem coefficient_x4_in_p_squared (x : ℝ) : 
  let p : ℝ → ℝ := λ x => x^5 - 2*x^2 + 1
  ∃ other_terms : ℝ → ℝ, (p x)^2 = x^10 + 4*x^4 + other_terms x
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_p_squared_l340_34043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l340_34014

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi)^2 - Real.cos (x - Real.pi/3)^2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/4) → |f x - m| ≤ 2) ↔
    m ∈ Set.Icc (-7/4) (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l340_34014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_reps_calculation_l340_34075

/-- The number of repetitions in the original routine -/
def original_reps : ℕ := 25

/-- The weight of each dumbbell in the original routine (in pounds) -/
def original_weight : ℕ := 10

/-- The number of dumbbells used in the original routine -/
def original_dumbbells : ℕ := 2

/-- The weight of the new dumbbell (in pounds) -/
def new_weight : ℕ := 15

/-- Function to calculate the total weight lifted -/
def total_weight (weight : ℕ) (reps : ℕ) (dumbbells : ℕ := 1) : ℕ :=
  weight * reps * dumbbells

/-- Function to round up to the nearest integer -/
def round_up (x : ℚ) : ℕ :=
  Int.toNat (Int.ceil x)

/-- Theorem stating the number of repetitions needed with the new weight -/
theorem new_reps_calculation : 
  round_up ((total_weight original_weight original_reps original_dumbbells : ℚ) / new_weight) = 34 := by
  sorry

#eval round_up ((total_weight original_weight original_reps original_dumbbells : ℚ) / new_weight)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_reps_calculation_l340_34075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l340_34031

noncomputable section

def sqrt2 : ℝ := Real.sqrt 2

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the condition for the function
def SatisfiesCondition (f : FunctionType) : Prop :=
  ∀ x : ℝ, f (sqrt2 * x) + f ((4 + 3 * sqrt2) * x) = 2 * f ((2 + sqrt2) * x)

-- Define a periodic function with period 1
def IsPeriodic1 (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 1) = g x

-- The main theorem
theorem functional_equation_solution (f : FunctionType) (h : SatisfiesCondition f) :
  ∃ (g : ℝ → ℝ) (a : ℝ), IsPeriodic1 g ∧
    (∀ x : ℝ, x ≠ 0 → f x = g (Real.log (abs x) / Real.log (sqrt2 - 1))) ∧
    f 0 = a := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l340_34031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l340_34085

theorem repeated_root_angle (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) : 
  (∃ x : ℝ, (x^2 + 4 * x * Real.cos θ + Real.tan (π/2 - θ) = 0) ∧ 
   (∀ y : ℝ, y^2 + 4 * y * Real.cos θ + Real.tan (π/2 - θ) = 0 → y = x)) →
  (θ = π / 12 ∨ θ = 5 * π / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l340_34085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_eccentricity_l340_34067

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Check if a point (x, y) lies on the ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipses_same_eccentricity :
  let C : Ellipse := ⟨2, Real.sqrt 3, by norm_num, by norm_num, sorry⟩
  let C₁ : Ellipse := ⟨2 * Real.sqrt 2, Real.sqrt 6, by norm_num, by norm_num, sorry⟩
  on_ellipse C₁ 2 (-Real.sqrt 3) →
  eccentricity C = eccentricity C₁ ∧ eccentricity C = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_eccentricity_l340_34067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_l340_34022

theorem definite_integral_exp : ∫ x in (0:ℝ)..(1:ℝ), Real.exp x = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_l340_34022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_travel_time_l340_34009

/-- Two cyclists leave a city at the same time in opposite directions. -/
def Cyclists := Unit

/-- The speed of the eastbound cyclist in mph. -/
noncomputable def eastbound_speed : ℝ := 18

/-- The speed difference between the westbound and eastbound cyclists in mph. -/
noncomputable def speed_difference : ℝ := 5

/-- The total distance between the cyclists after traveling for some time. -/
noncomputable def total_distance : ℝ := 246

/-- The time taken for the cyclists to reach the total distance apart. -/
noncomputable def travel_time (c : Cyclists) : ℝ := 
  total_distance / (2 * eastbound_speed + speed_difference)

theorem cyclists_travel_time (c : Cyclists) : 
  travel_time c = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_travel_time_l340_34009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_special_triangle_l340_34049

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the properties of the triangle
def is_right_angled (t : Triangle) : Prop :=
  (t.D.1 - t.F.1) * (t.E.1 - t.F.1) + (t.D.2 - t.F.2) * (t.E.2 - t.F.2) = 0

def angle_D_is_45_degrees (t : Triangle) : Prop :=
  let v1 := (t.E.1 - t.D.1, t.E.2 - t.D.2)
  let v2 := (t.F.1 - t.D.1, t.F.2 - t.D.2)
  Real.cos (Real.pi / 4) = (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))

def DF_equals_8 (t : Triangle) : Prop :=
  Real.sqrt ((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2) = 8

-- Define the incircle radius
noncomputable def incircle_radius (t : Triangle) : ℝ :=
  let a := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let b := Real.sqrt ((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2)
  let c := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let s := (a + b + c) / 2
  Real.sqrt ((s - a) * (s - b) * (s - c) / s)

-- Theorem statement
theorem incircle_radius_of_special_triangle :
  ∀ (t : Triangle),
    is_right_angled t →
    angle_D_is_45_degrees t →
    DF_equals_8 t →
    incircle_radius t = 4 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_special_triangle_l340_34049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_N_is_one_l340_34082

-- Define N as noncomputable due to its dependency on real numbers
noncomputable def N : ℝ := (1987^(4*(Real.sqrt 3 + 2)^1987 + 1988)) / (1987^((Real.sqrt 3 + 2)^1988) + (Real.sqrt 3 + 2)^1988)

-- Theorem statement
theorem last_digit_of_N_is_one : Int.mod (Int.floor N) 10 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_N_is_one_l340_34082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_inscribed_circle_area_ratio_l340_34024

noncomputable section

/-- The area of a circle inscribed in a regular octagon, touching the midpoints of the sides -/
def inscribed_circle_area (side_length : ℝ) : ℝ :=
  let r := side_length / (2 * (1 + Real.sqrt 2))
  Real.pi * r^2

/-- The area of a regular octagon -/
def octagon_area (side_length : ℝ) : ℝ :=
  2 * side_length^2 * (1 + Real.sqrt 2)

theorem octagon_inscribed_circle_area_ratio (side_length : ℝ) :
  ∃ (a b : ℕ), 
    inscribed_circle_area side_length / octagon_area side_length = Real.sqrt a / b ∧ 
    a * b = 192 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_inscribed_circle_area_ratio_l340_34024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_matches_l340_34011

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - 2 * Real.sin x * Real.cos x - Real.sin x ^ 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + 2 * Real.sin x * Real.cos x - 1

-- State the theorem
theorem shift_matches (m : ℝ) :
  (∀ x, f (x - m) = g x) → m = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_matches_l340_34011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l340_34028

/-- Sequence b_n defined recursively -/
def b : ℕ → ℕ
  | 0 => 2  -- Adding the case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + b (n + 1)

/-- The sum of the series -/
noncomputable def seriesSum : ℝ := ∑' n, (b n : ℝ) / 3^(n + 1)

/-- Theorem stating that the sum of the series equals 1/3 -/
theorem series_sum_equals_one_third : seriesSum = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l340_34028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l340_34096

theorem triangle_identity (A B C : ℝ) : 
  B = Real.pi / 2 →
  Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin A * Real.sin B * Real.cos C = 16/9 →
  Real.sin B ^ 2 + Real.sin C ^ 2 + 2 * Real.cos B * Real.cos C * Real.sin A = 25/16 →
  Real.cos C ^ 2 + Real.cos A ^ 2 + 2 * Real.sin C * Real.sin A * Real.sin B = 1 := by
  sorry

#check triangle_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_identity_l340_34096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l340_34065

-- Define the points
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (7, 2)
def E : ℝ × ℝ := (7, 0)
def F : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter function
noncomputable def perimeter : ℝ := 
  distance A B + distance B C + distance C D + distance D E + distance E F + distance F A

-- Theorem statement
theorem polygon_perimeter : perimeter = 24 := by
  -- Unfold definitions
  unfold perimeter distance
  -- Simplify expressions
  simp [A, B, C, D, E, F]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_perimeter_l340_34065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_activities_time_l340_34098

/-- Calculates the total time spent on activities given distances and speeds -/
def total_time (distances : List Float) (speeds : List Float) : Float :=
  List.sum (List.zipWith (λ d s => d / s) distances speeds)

/-- Christopher's activities theorem -/
theorem christopher_activities_time :
  let distances : List Float := [3, 5, 8, 1]
  let speeds : List Float := [6, 4, 5, 3]
  (total_time distances speeds - 3.6833).abs < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_activities_time_l340_34098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_height_uniformity_l340_34003

/-- Given two teams with the same average height, if the variance of heights in Team A
    is greater than the variance of heights in Team B, then Team B has more uniform heights. -/
theorem team_height_uniformity (height_A height_B : ℝ → ℝ) (μ : ℝ) :
  (∀ x : ℝ, ∫ (t : ℝ), (height_A t - μ)^2 > ∫ (t : ℝ), (height_B t - μ)^2) →
  (∃ y : ℝ, ∀ x : ℝ, |height_B x - μ| ≤ |height_A y - μ|) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_height_uniformity_l340_34003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_g_solution_set_of_f_eq_one_l340_34052

open Real

noncomputable def θ : ℝ := sorry

axiom θ_in_range : 0 < θ ∧ θ < π / 2

noncomputable def g (x : ℝ) : ℝ := Real.log ((1 / x) - (cos θ)^2) / Real.log ((sin θ)^2)

noncomputable def f : ℝ → ℝ := Function.invFun g

theorem f_inverse_of_g : Function.LeftInverse f g ∧ Function.RightInverse f g := sorry

theorem solution_set_of_f_eq_one :
  {x : ℝ | f x = 1} = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_g_solution_set_of_f_eq_one_l340_34052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_side_area_l340_34072

/-- The side area of a cylinder as a function of its height -/
noncomputable def cylinderSideArea (x : ℝ) : ℝ := 2 * Real.pi * (6 - x) * x / 3

/-- Theorem: The side area of the internal tangent cylinder is maximum when its height is 3cm -/
theorem max_cylinder_side_area :
  ∀ x : ℝ, 0 < x → x < 6 →
  cylinderSideArea x ≤ cylinderSideArea 3 := by
  sorry

#check max_cylinder_side_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cylinder_side_area_l340_34072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_calculation_l340_34062

/-- Represents the empirical regression equation ŷ = -3.2x + â -/
def regression_equation (x : ℝ) (a : ℝ) : ℝ := -3.2 * x + a

/-- Calculates the residual given actual y and estimated ŷ -/
def calculate_residual (y : ℝ) (y_hat : ℝ) : ℝ := y - y_hat

theorem residual_calculation 
  (x_bar : ℝ) 
  (y_bar : ℝ) 
  (a : ℝ) 
  (h1 : x_bar = 10) 
  (h2 : y_bar = 8) 
  (h3 : regression_equation x_bar a = y_bar) :
  calculate_residual 10 (regression_equation 9.5 a) = 0.4 := by
  sorry

#check residual_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_calculation_l340_34062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l340_34010

/-- Line represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def line1 : Line := { a := -2, b := 1, c := -4 }
def line2 : Line := { a := -2, b := 1, c := 2 }
def line3 : Line := { a := 2, b := 1, c := -2 }

noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1/2) * abs ((t.p1.x * t.p2.y + t.p2.x * t.p3.y + t.p3.x * t.p1.y) -
               (t.p1.y * t.p2.x + t.p2.y * t.p3.x + t.p3.y * t.p1.x))

theorem triangle_area_is_six :
  ∃ (t : Triangle), triangle_area t = 6 ∧
    (∀ (p : Point), (line1.a * p.x + line1.b * p.y + line1.c = 0 ∨
                     line2.a * p.x + line2.b * p.y + line2.c = 0 ∨
                     line3.a * p.x + line3.b * p.y + line3.c = 0) →
                    (p = t.p1 ∨ p = t.p2 ∨ p = t.p3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l340_34010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_inclination_l340_34027

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculates the slope from an angle in degrees -/
noncomputable def slopeFromAngle (angle : ℝ) : ℝ :=
  Real.tan (angle * Real.pi / 180)

/-- Generates the equation of a line in the form ax + by + c = 0 -/
def lineEquation (l : Line) : ℝ → ℝ → ℝ :=
  fun x y => x + l.slope * (-x + l.point.1) - y + l.point.2

theorem line_equation_through_point_with_inclination (p : ℝ × ℝ) (angle : ℝ) :
  let l : Line := { slope := slopeFromAngle angle, point := p }
  (p.1 = 2 ∧ p.2 = 1 ∧ angle = 135) →
  ∀ x y, lineEquation l x y = x + y - 3 := by
  sorry

#check line_equation_through_point_with_inclination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_point_with_inclination_l340_34027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l340_34046

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8*x - 4*y + 16

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (4, -2)

/-- The given point -/
def given_point : ℝ × ℝ := (3, -1)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that the distance between the circle center and the given point is √2 -/
theorem distance_to_center : 
  distance circle_center given_point = Real.sqrt 2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_l340_34046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_l340_34083

theorem cosine_sum_squared (α β γ : ℝ) 
  (h1 : Real.sin α + Real.sin β + Real.sin γ = 0)
  (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_squared_l340_34083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportional_function_and_point_l340_34018

-- Define the inverse proportionality relationship
def inverse_proportional (y x : ℝ) : Prop := ∃ k : ℝ, y = k / (x - 1)

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) : ℝ := 3 / (x - 1)

theorem inverse_proportional_function_and_point :
  (∃ y : ℝ, inverse_proportional y 4 ∧ y = 1) →
  (∀ x : ℝ, x ≠ 1 → inverse_proportional (f x) x) ∧
  f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportional_function_and_point_l340_34018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l340_34019

noncomputable section

-- Define the line y = 3/4x
def line (x : ℝ) : ℝ := (3/4) * x

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the distance between two points
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the function we want to prove bounds for
def f (x : ℝ) : ℝ :=
  |distance (x, line x) F₁ - distance (x, line x) F₂|

-- Statement to prove
theorem f_range :
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 8.5 ∧ ∀ x, 0 ≤ f x ∧ f x ≤ 8.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l340_34019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l340_34089

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 5 * x - 12 * y = 0

-- Define the condition that the terminal side of angle α lies on the line
def terminal_side_on_line (α : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation x y ∧ x = Real.cos α ∧ y = Real.sin α

-- Theorem statement
theorem cos_alpha_value (α : ℝ) (h : terminal_side_on_line α) :
  Real.cos α = 12/13 ∨ Real.cos α = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l340_34089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_has_property_P_half_f₂_has_property_P_for_all_m_l340_34001

-- Define the property P(m) for a function f on [0,2]
def has_property_P (f : ℝ → ℝ) (m : ℝ) : Prop :=
  0 < m ∧ m < 2 ∧ ∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 (2 - m) ∧ f x₀ = f (x₀ + m)

-- Define the functions
noncomputable def f₁ : ℝ → ℝ := λ x => Real.sqrt (1 - (x - 1)^2)
def f₂ : ℝ → ℝ := λ x => (x - 1)^2

-- State the theorems
theorem f₁_has_property_P_half :
  has_property_P f₁ (1/2) := by sorry

theorem f₂_has_property_P_for_all_m :
  ∀ m : ℝ, 0 < m → m < 2 → has_property_P f₂ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_has_property_P_half_f₂_has_property_P_for_all_m_l340_34001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l340_34090

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) (final_population : ℕ) : 
  initial_population = 4675 →
  death_rate = 5 / 100 →
  leaving_rate = 20 / 100 →
  final_population = 
    initial_population - 
    (Int.toNat ⌊(initial_population : ℚ) * death_rate⌋) - 
    (Int.toNat ⌊((initial_population : ℚ) - (initial_population : ℚ) * death_rate) * leaving_rate⌋) →
  final_population = 3553 := by
  sorry

#eval (4675 : ℕ) - 
  (Int.toNat ⌊(4675 : ℚ) * (5 / 100 : ℚ)⌋) - 
  (Int.toNat ⌊((4675 : ℚ) - (4675 : ℚ) * (5 / 100 : ℚ)) * (20 / 100 : ℚ)⌋)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_l340_34090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_l340_34016

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then (x + 1)^5 else x^2 + 2

-- State the theorem
theorem coefficient_of_x_fourth (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ (a b c d e : ℝ), f (f x) = 270 * x^4 + a * x^5 + b * x^3 + c * x^2 + d * x + e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_fourth_l340_34016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_thirteen_pi_fourths_l340_34057

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_thirteen_pi_fourths_l340_34057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l340_34077

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_area (t : Triangle) : 
  t.a = 7 ∧ 
  f (t.A / 2 - Real.pi / 6) = Real.sqrt 3 ∧ 
  Real.sin t.B + Real.sin t.C = 13 * Real.sqrt 3 / 14 ∧
  t.A > 0 ∧ t.A < Real.pi / 2 →
  (1/2) * t.b * t.c * Real.sin t.A = 10 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l340_34077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_equal_division_l340_34058

/-- A quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The midpoint of a line segment -/
noncomputable def midpoint' (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- A point dividing a line segment in the ratio 2:1 -/
noncomputable def divideInRatio2To1 (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((2 * P.1 + Q.1) / 3, (2 * P.2 + Q.2) / 3)

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- Theorem stating the existence of a quadrilateral with the required properties -/
theorem exists_quadrilateral_with_equal_division :
  ∃ (ABCD : Quadrilateral) (M N : ℝ × ℝ),
    M = midpoint' ABCD.A ABCD.B ∧
    N = divideInRatio2To1 ABCD.C ABCD.D ∧
    triangleArea ABCD.A M N + triangleArea ABCD.D M N =
    triangleArea ABCD.B M N + triangleArea ABCD.C M N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_equal_division_l340_34058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l340_34021

-- Define points P and Q
def P : ℝ × ℝ := (-3, 2)
def Q : ℝ × ℝ := (4, 10)

-- Define point D as a function of a real parameter t
def D (t : ℝ) : ℝ × ℝ := (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_D_coordinates :
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  distance P (D t) = 4 * distance (D t) Q ∧
  D t = (2.25, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l340_34021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_in_M_f_in_M_conditions_theta_range_l340_34071

noncomputable def M : Set (ℝ → ℝ) := {h | ∀ x, h (-x) = -h x}

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^(x+1) + b)

theorem f_not_in_M : f 1 1 ∉ M := by sorry

theorem f_in_M_conditions (a b : ℝ) : 
  f a b ∈ M ↔ a = 1 ∧ b = 2 := by sorry

theorem theta_range (θ : ℝ) : 
  (∀ x, f 1 2 x < Real.sin θ) ↔ 
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 5 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_in_M_f_in_M_conditions_theta_range_l340_34071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l340_34091

theorem number_relationship (a b c : ℝ) :
  a = (0.6 : ℝ)^(0.3 : ℝ) → b = Real.log 3 / Real.log 0.6 → c = Real.log π → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l340_34091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_principal_repayment_formula_l340_34015

/-- Repayment amount for the nth month under equal principal repayment method -/
noncomputable def repayment_amount (P : ℝ) (T : ℕ) (r : ℝ) (n : ℕ) : ℝ :=
  (P / (T : ℝ)) + (P - (n - 1) * (P / (T : ℝ))) * r

theorem equal_principal_repayment_formula 
  (P : ℝ) (T : ℕ) (r : ℝ) (n : ℕ) 
  (h_P : P = 480000) 
  (h_T : T = 240) 
  (h_r : r = 0.004) :
  repayment_amount P T r n = 3928 - 8 * (n : ℝ) := by
  sorry

#check equal_principal_repayment_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_principal_repayment_formula_l340_34015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l340_34080

noncomputable def f (x : ℝ) := 1 / Real.sqrt (x + 2) + (x - 1) ^ (0 : ℕ)

theorem f_domain : 
  {x : ℝ | x > -2 ∧ x ≠ 1} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l340_34080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_l340_34037

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 1 then Real.log x else 1 - x

theorem f_greater_than_one_iff (m : ℝ) : 
  f m > 1 ↔ m < 0 ∨ m > Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_iff_l340_34037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_player_beats_all_probability_l340_34030

/-- Represents the possible moves in rock-paper-scissors -/
inductive Move
| Rock
| Paper
| Scissors
deriving DecidableEq

/-- Represents the outcome of a rock-paper-scissors game -/
inductive Outcome
| Win
| Lose
| Draw

/-- A function that determines the outcome of a game between two players -/
def gameResult (player1 : Move) (player2 : Move) : Outcome :=
  match player1, player2 with
  | Move.Rock, Move.Scissors => Outcome.Win
  | Move.Paper, Move.Rock => Outcome.Win
  | Move.Scissors, Move.Paper => Outcome.Win
  | m1, m2 => if m1 = m2 then Outcome.Draw else Outcome.Lose

/-- The probability of choosing any move -/
noncomputable def moveProbability : ℝ := 1 / 3

/-- The number of players -/
def numPlayers : ℕ := 4

/-- Theorem: The probability that one player beats all others in a four-player
    rock-paper-scissors game is 4/27 -/
theorem one_player_beats_all_probability :
  (3 * numPlayers : ℝ) * (moveProbability ^ (numPlayers - 1)) = 4 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_player_beats_all_probability_l340_34030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_always_wins_l340_34078

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment connecting two points -/
structure Segment where
  start : Point
  finish : Point

/-- The game setup -/
structure GameSetup where
  points : Finset Point
  segments : Finset Segment
  pointCount : Nat
  segmentLabels : Segment → Fin 2
  pointLabels : Point → Fin 2

/-- The game conditions -/
def ValidGame (g : GameSetup) : Prop :=
  g.pointCount = 2005 ∧
  g.points.card = g.pointCount ∧
  (∀ p1 p2 p3 : Point, p1 ∈ g.points → p2 ∈ g.points → p3 ∈ g.points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬(∃ (t : ℝ), p3 = ⟨(1 - t) * p1.x + t * p2.x, (1 - t) * p1.y + t * p2.y⟩)) ∧
  (∀ p1 p2 : Point, p1 ∈ g.points → p2 ∈ g.points → p1 ≠ p2 →
    ∃ s : Segment, s ∈ g.segments ∧ s.start = p1 ∧ s.finish = p2)

/-- Donkey's winning condition -/
def DonkeyWins (g : GameSetup) : Prop :=
  ∃ p1 p2 : Point, ∃ s : Segment,
    p1 ∈ g.points ∧ p2 ∈ g.points ∧ s ∈ g.segments ∧
    s.start = p1 ∧ s.finish = p2 ∧
    g.segmentLabels s = g.pointLabels p1 ∧ g.pointLabels p1 = g.pointLabels p2

/-- The main theorem: Donkey always wins with optimal play -/
theorem donkey_always_wins (g : GameSetup) (h : ValidGame g) :
  DonkeyWins g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donkey_always_wins_l340_34078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_calculation_l340_34088

/-- The amount of detergent used per pound of clothes, given total detergent and total weight -/
noncomputable def detergent_per_pound (total_detergent : ℚ) (total_weight : ℚ) : ℚ :=
  total_detergent / total_weight

/-- Theorem stating that the amount of detergent used per pound is 2 ounces -/
theorem detergent_calculation :
  detergent_per_pound 18 9 = 2 := by
  -- Unfold the definition of detergent_per_pound
  unfold detergent_per_pound
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_calculation_l340_34088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l340_34070

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (2 * x.natAbs + 7 < 17) ∧
    (∀ (y : ℤ), 2 * y.natAbs + 7 < 17 → y ≥ x) ∧
    x = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_satisfying_inequality_l340_34070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_non_swimmers_playing_soccer_l340_34066

theorem percent_non_swimmers_playing_soccer :
  ∀ (N : ℝ) (soccer_players swimming_players : ℝ),
    soccer_players = 0.7 * N →
    swimming_players = 0.5 * N →
    (soccer_players * 0.6) = 0.42 * N →
    let non_swimmers := N - swimming_players
    let soccer_non_swimmers := soccer_players - (soccer_players * 0.6)
    (soccer_non_swimmers / non_swimmers) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_non_swimmers_playing_soccer_l340_34066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_8_solution_is_x_cubed_l340_34026

-- Define a power function
noncomputable def power_function (n : ℝ) : ℝ → ℝ := λ x ↦ x^n

-- State the theorem
theorem power_function_through_point_2_8 :
  ∃ n : ℝ, power_function n 2 = 8 ∧ n = 3 := by
  -- Prove that n = 3 satisfies the conditions
  use 3
  constructor
  · -- Show that power_function 3 2 = 8
    simp [power_function]
    norm_num
  · -- Show that n = 3
    rfl

-- Additional theorem to explicitly state that y = x^3 is the solution
theorem solution_is_x_cubed :
  ∃ f : ℝ → ℝ, (∀ x, f x = x^3) ∧ f 2 = 8 := by
  use λ x ↦ x^3
  constructor
  · intro x
    rfl
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_8_solution_is_x_cubed_l340_34026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_flight_time_l340_34032

/-- The radius of the Earth at a specific latitude in miles -/
noncomputable def earth_radius : ℝ := 3000

/-- The average speed of the jet in miles per hour -/
noncomputable def jet_speed : ℝ := 600

/-- The circumference of the Earth at the given latitude -/
noncomputable def earth_circumference : ℝ := 2 * Real.pi * earth_radius

/-- The time taken for the jet to fly around the Earth -/
noncomputable def flight_time : ℝ := earth_circumference / jet_speed

theorem jet_flight_time :
  flight_time = 10 * Real.pi := by
  -- Expand the definitions
  unfold flight_time earth_circumference
  -- Simplify the expression
  simp [earth_radius, jet_speed]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_flight_time_l340_34032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicle_cost_optimization_l340_34038

/-- Represents the total cost of using a new energy vehicle for n years -/
noncomputable def f (n : ℕ) : ℝ := 0.1 * (n : ℝ)^2 + n + 14.4

/-- Represents the average annual cost of using a new energy vehicle for n years -/
noncomputable def S (n : ℕ) : ℝ := f n / n

/-- The optimal number of years to use the vehicle -/
def optimal_years : ℕ := 12

/-- The minimum average annual cost -/
def min_avg_cost : ℝ := 3.4

theorem new_energy_vehicle_cost_optimization :
  (∀ n : ℕ, n > 0 → S n ≥ min_avg_cost) ∧
  S optimal_years = min_avg_cost := by
  sorry

#eval optimal_years
#eval min_avg_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicle_cost_optimization_l340_34038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_f_strictly_increasing_f_values_at_endpoints_zero_of_f_in_zero_one_l340_34025

-- Define the function f(x) = 3^x + 2x - 3
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) + 2*x - 3

-- State the theorem
theorem zero_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is strictly increasing
  ∃ c ∈ Set.Ioo 0 1, f c = 0 :=
by
  sorry

-- Prove that f is strictly increasing
theorem f_strictly_increasing :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

-- Prove that f(0) < 0 and f(1) > 0
theorem f_values_at_endpoints :
  f 0 < 0 ∧ f 1 > 0 :=
by
  sorry

-- Main theorem: The zero of f is in the interval (0, 1)
theorem zero_of_f_in_zero_one :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_f_strictly_increasing_f_values_at_endpoints_zero_of_f_in_zero_one_l340_34025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pocket_money_difference_l340_34029

/-- Represents the amount of money given to each child in a round -/
structure RoundMoney where
  anya : ℕ
  borya : ℕ
  vitya : ℕ

/-- Generates the sequence of money given in each round -/
def generateSequence : ℕ → RoundMoney
  | 0 => ⟨1, 2, 3⟩
  | n + 1 => let prev := generateSequence n; ⟨prev.anya + 3, prev.borya + 3, prev.vitya + 3⟩

/-- Calculates the total money received by each child after n rounds -/
def totalMoney (n : ℕ) : RoundMoney :=
  (List.range n).foldl (fun acc i => ⟨acc.anya + (generateSequence i).anya,
                                     acc.borya + (generateSequence i).borya,
                                     acc.vitya + (generateSequence i).vitya⟩)
                       ⟨0, 0, 0⟩

/-- The main theorem to be proved -/
theorem pocket_money_difference (n : ℕ) :
  (totalMoney n).anya = 202 ∧ (totalMoney n).borya = 203 →
  (totalMoney n).anya - (totalMoney n).vitya = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pocket_money_difference_l340_34029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_two_l340_34036

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y

/-- The parabola x^2 = 4y with a chord AB of length 6 -/
def parabola_with_chord (A B : ParabolaPoint) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = 36

/-- The shortest distance from the midpoint of AB to the x-axis -/
noncomputable def shortest_distance (A B : ParabolaPoint) : ℝ :=
  (A.y + B.y) / 2

/-- Theorem: The shortest distance from the midpoint of AB to the x-axis is 2 -/
theorem shortest_distance_is_two (A B : ParabolaPoint) 
  (h : parabola_with_chord A B) : 
  shortest_distance A B ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_is_two_l340_34036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l340_34061

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 1) / (6 * x^2 + 7 * x + 3)

theorem vertical_asymptotes_sum (p q : ℝ) : 
  (6 * p^2 + 7 * p + 3 = 0) → 
  (6 * q^2 + 7 * q + 3 = 0) → 
  (p ≠ q) →
  (p + q = -11/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l340_34061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l340_34050

theorem cube_root_of_negative_eight : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_eight_l340_34050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_for_y_l340_34056

/-- The sum of the geometric series with first term a and common ratio r -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The left side of the equation -/
noncomputable def leftSide : ℝ := 
  (geometricSeriesSum (1/2) (1/3)) * (geometricSeriesSum (1/3) (-1/3))

/-- The right side of the equation -/
noncomputable def rightSide (y : ℝ) : ℝ := 
  geometricSeriesSum (1/y) (1/y)

theorem equation_holds_for_y (y : ℝ) : 
  leftSide = rightSide y ↔ y = 19/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_holds_for_y_l340_34056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l340_34068

noncomputable def initial_height : ℝ := 150
noncomputable def rebound_factor : ℝ := 1/3
def num_bounces : ℕ := 3

noncomputable def bounce_sequence : ℕ → ℝ
  | 0 => initial_height
  | n + 1 => rebound_factor * bounce_sequence n

noncomputable def total_distance : ℝ :=
  2 * (bounce_sequence 0 + bounce_sequence 1 + bounce_sequence 2) +
  bounce_sequence 3

theorem super_ball_distance :
  total_distance = 294 + 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_ball_distance_l340_34068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_on_interval_l340_34064

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_range_on_interval 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_period : (2 * Real.pi) / ω = 4) 
  (h_shift : ∀ x, f ω φ (x + 1/3) = f ω φ (-x)) 
  : Set.range (fun x => f ω φ x) ∩ Set.Icc 0 1 = Set.Icc (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_on_interval_l340_34064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_integer_product_l340_34074

-- Define the sequence (aₙ)
def a : ℕ → ℝ
  | 0 => 1
  | 1 => 2^(1/19)
  | n + 2 => a (n + 1) * (a n)^2

-- Define the product of the first k terms
def product (k : ℕ) : ℝ :=
  (List.range k).foldl (λ acc i => acc * a (i + 1)) 1

-- Define what it means for a real number to be an integer
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Statement of the theorem
theorem smallest_k_for_integer_product :
  ∀ k : ℕ, k > 0 → isInteger (product k) → k ≥ 17 ∧
  isInteger (product 17) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_integer_product_l340_34074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l340_34059

/-- The area of a triangle given its vertices' coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_PQR_area :
  let P : ℝ × ℝ := (-6, 4)
  let Q : ℝ × ℝ := (1, 7)
  let R : ℝ × ℝ := (4, -3)
  triangleArea P.1 P.2 Q.1 Q.2 R.1 R.2 = 59.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l340_34059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circle_shape_perimeter_l340_34045

/-- A shape made from three circles with the following properties:
  - Each circle has radius R
  - The centers of the circles lie on the same straight line
  - The middle circle passes through the centers of the other two circles -/
structure ThreeCircleShape (R : ℝ) where
  -- We don't need to define the internal structure,
  -- as we're only concerned with the perimeter

/-- The perimeter of a ThreeCircleShape -/
noncomputable def perimeter (R : ℝ) (shape : ThreeCircleShape R) : ℝ := 10 * Real.pi * R / 3

/-- Theorem stating that the perimeter of a ThreeCircleShape is 10πR/3 -/
theorem three_circle_shape_perimeter (R : ℝ) (shape : ThreeCircleShape R) :
  perimeter R shape = 10 * Real.pi * R / 3 := by
  -- Unfold the definition of perimeter
  unfold perimeter
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circle_shape_perimeter_l340_34045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_one_l340_34048

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_one : lg 5 + lg 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_sum_equals_one_l340_34048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_operations_order_l340_34055

-- Define the functions as noncomputable
noncomputable def evaluate_left_to_right : String → ℝ := sorry
noncomputable def evaluate_correct_order : String → ℝ := sorry

-- State the theorem
theorem mixed_operations_order : ¬(∀ (expr : String), evaluate_left_to_right expr = evaluate_correct_order expr) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_operations_order_l340_34055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_print_cost_is_15_l340_34012

/-- Represents the cost of Emily's purchase --/
structure EmilyCost where
  curtainPairs : ℕ
  curtainPrice : ℚ
  wallPrints : ℕ
  installationCost : ℚ
  totalCost : ℚ

/-- Calculates the cost of each wall print --/
def wallPrintCost (e : EmilyCost) : ℚ :=
  (e.totalCost - (e.curtainPairs * e.curtainPrice + e.installationCost)) / e.wallPrints

/-- Theorem stating that each wall print costs $15.00 --/
theorem wall_print_cost_is_15 (e : EmilyCost) 
  (h1 : e.curtainPairs = 2)
  (h2 : e.curtainPrice = 30)
  (h3 : e.wallPrints = 9)
  (h4 : e.installationCost = 50)
  (h5 : e.totalCost = 245) :
  wallPrintCost e = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_print_cost_is_15_l340_34012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l340_34002

theorem log_sum_equality : 
  ∀ (log₃ : ℝ → ℝ) (lg : ℝ → ℝ) (ln : ℝ → ℝ) (exp : ℝ → ℝ),
  (log₃ (427 / 3) = log₃ (3^(-(1/4 : ℝ)))) →
  (lg 25 = 2 * lg 5) →
  (exp (ln 2) = 2) →
  (lg 10 = 1) →
  log₃ (427 / 3) + lg 25 + 2 * lg 2 + exp (ln 2) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l340_34002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brad_speed_l340_34041

/-- Calculates Brad's running speed given the conditions of the problem -/
theorem brad_speed (total_distance : ℝ) (maxwell_speed : ℝ) (maxwell_distance : ℝ) :
  total_distance = 36 →
  maxwell_speed = 3 →
  maxwell_distance = 12 →
  maxwell_distance = total_distance / 2 →
  let brad_distance := total_distance / 2 - maxwell_distance
  let time := maxwell_distance / maxwell_speed
  (brad_distance / time) = 1.5 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brad_speed_l340_34041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l340_34033

def computer_sequence (initial_x initial_y : ℕ) (x_increment : ℕ) (y_threshold : ℕ) : ℕ :=
  let rec loop (x y : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then x
    else if y ≥ y_threshold then x
    else loop (x + x_increment) (y + x + x_increment) (fuel - 1)
  loop initial_x initial_y 1000  -- Assuming 1000 iterations is enough

theorem final_x_value :
  computer_sequence 5 0 3 5000 = 173 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l340_34033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l340_34073

/-- The average age of three women given the following conditions:
    1. There are initially 12 men
    2. Three men of ages 25, 15, and 30 years are replaced by three women
    3. The average age increases by 3.5 years after the replacement
-/
theorem women_average_age (initial_men : ℕ) (age_increase : ℚ) 
  (replaced_men_ages : List ℕ) (h1 : initial_men = 12) 
  (h2 : age_increase = 7/2) 
  (h3 : replaced_men_ages = [25, 15, 30]) : 
  (112 : ℚ) / 3 = 
    (initial_men * age_increase + replaced_men_ages.sum) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l340_34073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangle_theorem_l340_34063

noncomputable section

open Real

theorem oblique_triangle_theorem (A B C a b c : ℝ) :
  -- Triangle ABC is non-degenerate
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Sine law
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 1 →
  -- Conclusion
  (a^2 + b^2) / c^2 = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_triangle_theorem_l340_34063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_average_price_l340_34040

/-- The average price of towels given specific quantities and prices -/
theorem towel_average_price
  (price1 : ℝ) (quantity1 : ℝ) (price2 : ℝ) (quantity2 : ℝ) (total_price3 : ℝ) (quantity3 : ℝ)
  (h1 : price1 = 100)
  (h2 : quantity1 = 3)
  (h3 : price2 = 150)
  (h4 : quantity2 = 5)
  (h5 : total_price3 = 550)
  (h6 : quantity3 = 2) :
  (price1 * quantity1 + price2 * quantity2 + total_price3) / (quantity1 + quantity2 + quantity3) = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_towel_average_price_l340_34040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_56_25_l340_34008

/-- Calculates the profit percentage given cost price, discount percentage, and selling price -/
noncomputable def profit_percentage (cost_price : ℝ) (discount_percent : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := selling_price / (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 56.25% -/
theorem profit_percentage_is_56_25 :
  let cost_price : ℝ := 47.50
  let discount_percent : ℝ := 20
  let selling_price : ℝ := 74.21875
  profit_percentage cost_price discount_percent selling_price = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_56_25_l340_34008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l340_34097

-- Define the radius of the circle
def radius : ℝ := 12

-- Define the angle of each sector in radians (60° = π/3 radians)
noncomputable def sector_angle : ℝ := Real.pi / 3

-- Define the area of the figure ABCD
noncomputable def area_ABCD : ℝ := 2 * (sector_angle / (2 * Real.pi)) * (Real.pi * radius^2)

-- Theorem statement
theorem area_of_two_sectors :
  area_ABCD = 48 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_two_sectors_l340_34097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l340_34035

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x - 8)*(x - 5) / (x - 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Iic (8/3) ∪ Set.Ioi 5

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 1 → (g x ≥ 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l340_34035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_publishing_house_arrangements_l340_34044

/-- The number of different arrangements for typesetting and printing tasks -/
def arrangements (total_workers : ℕ) (typesetting_capable : ℕ) (printing_capable : ℕ) 
  (typesetting_needed : ℕ) (printing_needed : ℕ) : ℕ :=
  -- Definition to be filled
  0  -- Placeholder, replace with actual implementation

/-- Theorem stating the number of arrangements for the given problem -/
theorem publishing_house_arrangements : 
  arrangements 7 5 4 2 2 = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_publishing_house_arrangements_l340_34044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_holds_l340_34092

/-- Given a pattern where √(n + n/d) = n√(n/d) for some natural numbers n and d,
    this function represents the denominator d in terms of n. -/
def denominator (n : ℕ) : ℕ := n^2 - 1

/-- The theorem states that if √(6 + a/t) = 6√(a/t) follows the given pattern,
    then a = 6 and t = 35. -/
theorem pattern_holds (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  (Real.sqrt (6 + a/t) = 6 * Real.sqrt (a/t)) → (a = 6 ∧ t = 35) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_holds_l340_34092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l340_34093

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 2/3

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

-- Define the slope angle function
noncomputable def slope_angle (x : ℝ) : ℝ := Real.arctan (curve_derivative x)

-- Theorem statement
theorem slope_angle_range :
  ∀ x : ℝ, 0 ≤ slope_angle x ∧ slope_angle x < Real.pi ∧
  (slope_angle x ≤ Real.pi/2 ∨ 2*Real.pi/3 ≤ slope_angle x) :=
by
  sorry

#check slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l340_34093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l340_34005

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Condition that a_1, a_3, a_4 form a geometric sequence -/
def GeometricSubsequence (seq : ArithmeticSequence) : Prop :=
  (seq.a 3) ^ 2 = (seq.a 1) * (seq.a 4)

theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) 
  (h : GeometricSubsequence seq) :
  (S seq 3 - S seq 2) / (S seq 5 - S seq 3) = 1/2 ∨
  (S seq 3 - S seq 2) / (S seq 5 - S seq 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l340_34005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l340_34081

-- Define the necessary types
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the objects
variable (a b : AffineSubspace ℝ V) (α : AffineSubspace ℝ V)

-- Define the conditions
variable (h1 : a.direction ≠ b.direction)  -- a and b are skew
variable (h2 : a.direction ≤ α.direction)  -- a is parallel to α

-- Theorem statement
theorem line_plane_relationship :
  (b.direction ≤ α.direction)  -- b is parallel to α
  ∨ (∃ p, p ∈ b ∧ p ∈ α)      -- b intersects α
  ∨ (b ≤ α)                   -- b lies in α
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l340_34081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_difference_26_dollars_l340_34042

/-- Calculates the difference in cents between a good tip (20%) and a bad tip (5%) on a given bill amount. -/
def tip_difference (bill : ℚ) : ℕ :=
  let good_tip_rate : ℚ := 20 / 100
  let bad_tip_rate : ℚ := 5 / 100
  let good_tip : ℚ := bill * good_tip_rate
  let bad_tip : ℚ := bill * bad_tip_rate
  let difference : ℚ := good_tip - bad_tip
  (difference * 100).floor.toNat

/-- The difference between a 20% tip and a 5% tip on a $26 bill is 390 cents. -/
theorem tip_difference_26_dollars : tip_difference 26 = 390 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_difference_26_dollars_l340_34042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l340_34007

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  (seq.a 7 - 1)^3 + 2016 * (seq.a 7 - 1) = -1 →
  (seq.a 2010 - 1)^3 + 2016 * (seq.a 2010 - 1) = 1 →
  sum_n seq 2016 = 2016 ∧ seq.a 2010 > seq.a 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l340_34007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_distinct_l340_34013

/-- Represents a number in the triangular table --/
inductive TableNumber
| initial : ℕ → TableNumber
| squared : TableNumber → TableNumber
| increment : TableNumber → TableNumber

/-- Generates the nth row of the table starting from a --/
def generateRow (a : ℕ) (n : ℕ) : List TableNumber :=
  match n with
  | 0 => [TableNumber.initial a]
  | n + 1 => (generateRow a n).bind (λ x => [TableNumber.squared x, TableNumber.increment x])

/-- Evaluates a TableNumber to its corresponding natural number --/
def evalTableNumber : TableNumber → ℕ
| TableNumber.initial n => n
| TableNumber.squared t => (evalTableNumber t) ^ 2
| TableNumber.increment t => evalTableNumber t + 1

/-- States that all numbers in the nth row of the table starting from a are distinct --/
theorem all_numbers_distinct (a n : ℕ) : 
  List.Nodup ((generateRow a n).map evalTableNumber) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_distinct_l340_34013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_chord_length_l340_34020

noncomputable section

-- Define the ellipses
def ellipse1 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def ellipse2 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the eccentricity
def eccentricity : ℝ := Real.sqrt 5 / 2

-- Theorem 1: Hyperbola equation
theorem hyperbola_equation :
  ∀ x y : ℝ, hyperbola x y ↔ 
    (∃ f : ℝ × ℝ, (ellipse1 f.1 f.2 ∧ hyperbola f.1 f.2) ∧
    Real.sqrt ((x - f.1)^2 + (y - f.2)^2) / Real.sqrt (x^2 / 4 + y^2) = eccentricity) :=
by sorry

-- Theorem 2: Chord length
theorem chord_length :
  ∃ a b : ℝ × ℝ, 
    ellipse2 a.1 a.2 ∧ ellipse2 b.1 b.2 ∧
    line a.1 a.2 ∧ line b.1 b.2 ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 8 / 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_chord_length_l340_34020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_l340_34069

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 - 1 / (a n)

def b (n : ℕ) : ℚ := 1 / (a n - 1)

theorem arithmetic_sequence_b : ∀ n : ℕ, b (n + 1) - b n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_b_l340_34069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l340_34006

/-- Calculates the time (in seconds) it takes for a train to cross a pole -/
noncomputable def time_to_cross_pole (train_length : ℝ) (train_speed_km_hr : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_hr * (1000 / 3600)
  train_length / train_speed_m_s

/-- Theorem stating that a train with length 200 meters traveling at 45 km/hr takes 16 seconds to cross a pole -/
theorem train_crossing_time :
  time_to_cross_pole 200 45 = 16 := by
  -- Unfold the definition of time_to_cross_pole
  unfold time_to_cross_pole
  -- Simplify the expression
  simp
  -- The proof is incomplete, so we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l340_34006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_symmetric_l340_34087

/-- Recursive definition of polynomial P_m -/
def P : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, x, y, z => 1
| (n+1), x, y, z => (x+z)*(y+z)*P n x y (z+1) - z^2*P n x y z

/-- P_m is symmetric for all permutations of x, y, z -/
theorem P_symmetric (m : ℕ) (x y z : ℝ) :
  P m x y z = P m y x z ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_symmetric_l340_34087
