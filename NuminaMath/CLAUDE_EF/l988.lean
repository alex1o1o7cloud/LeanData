import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr3_equals_a_power_5_l988_98862

-- Define a variable for the base
variable (a : ℝ)

-- Define the expressions
noncomputable def expr1 : ℝ → ℝ := λ a ↦ (a^3)^2
noncomputable def expr2 : ℝ → ℝ := λ a ↦ a^10 / a^2
noncomputable def expr3 : ℝ → ℝ := λ a ↦ a^4 * a
noncomputable def expr4 : ℝ → ℝ := λ a ↦ (-1)^(-1 : ℤ) * a^5

-- Theorem statement
theorem only_expr3_equals_a_power_5 (a : ℝ) :
  expr1 a ≠ a^5 ∧
  expr2 a ≠ a^5 ∧
  expr3 a = a^5 ∧
  expr4 a ≠ a^5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr3_equals_a_power_5_l988_98862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_theorem_l988_98824

-- Define the countries
inductive Country
| I
| II
| III
| IV
| V

-- Define the coloring function
def coloring : Country → Fin 2 := sorry

-- Define the adjacency relation
def adjacent : Country → Country → Prop := sorry

-- State the theorem
theorem two_color_theorem :
  ∃ (c : Country → Fin 2),
    (∀ x y : Country, adjacent x y → c x ≠ c y) :=
sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_theorem_l988_98824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l988_98831

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem quadratic_properties :
  (∃ x y : ℝ, (x = 1 ∧ y = -4) ∧ 
    (∀ t : ℝ, f t ≥ f x) ∧ 
    (∀ z : ℝ, f (x - z) = f (x + z))) ∧
  (∀ u : ℝ, f (1 - u) = f (1 + u)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l988_98831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completes_in_seven_days_l988_98892

/-- The number of days A takes to complete the entire work -/
noncomputable def a_total_days : ℝ := 21

/-- The number of days B takes to complete the entire work -/
noncomputable def b_total_days : ℝ := 15

/-- The number of days B worked before leaving -/
noncomputable def b_worked_days : ℝ := 10

/-- The fraction of work completed by B before leaving -/
noncomputable def b_completed_work : ℝ := b_worked_days / b_total_days

/-- The fraction of work remaining after B left -/
noncomputable def remaining_work : ℝ := 1 - b_completed_work

/-- The number of days A takes to complete the remaining work -/
noncomputable def a_remaining_days : ℝ := remaining_work * a_total_days

theorem a_completes_in_seven_days : 
  a_remaining_days = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completes_in_seven_days_l988_98892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_neg_half_l988_98889

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 1, |a + b| = √3, and |b| = 2, 
    prove that the projection of a onto b equals -1/2. -/
theorem projection_equals_neg_half 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b : V) 
  (ha : ‖a‖ = 1) 
  (hab : ‖a + b‖ = Real.sqrt 3) 
  (hb : ‖b‖ = 2) : 
  inner a b / ‖b‖^2 = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_neg_half_l988_98889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standing_time_is_60_l988_98860

/-- Represents the escalator problem with Clea's walking times. -/
structure EscalatorProblem where
  nonop_time : ℚ  -- Time to walk up non-operating escalator
  op_time : ℚ     -- Time to walk up operating escalator
  nonop_time_pos : 0 < nonop_time
  op_time_pos : 0 < op_time
  op_time_less : op_time < nonop_time

/-- Calculates the time for Clea to stand still on the operating escalator. -/
def standingTime (p : EscalatorProblem) : ℚ :=
  (p.nonop_time * p.op_time) / (p.nonop_time - p.op_time)

/-- Theorem stating that for the given problem, the standing time is 60 seconds. -/
theorem standing_time_is_60 (p : EscalatorProblem)
    (h1 : p.nonop_time = 90)
    (h2 : p.op_time = 36) :
    standingTime p = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standing_time_is_60_l988_98860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_addition_theorem_l988_98894

variable (a b : ℝ)

def P (a b : ℝ) : ℝ := sorry

theorem polynomial_addition_theorem (a b : ℝ) :
  (P a b + (-2 * a^3 + 4 * a^2 * b + 5 * b^3) = a^3 - 3 * a^2 * b + 2 * b^3) →
  P a b = 3 * a^3 - 7 * a^2 * b - 3 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_addition_theorem_l988_98894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_triples_condition1_infinitely_many_triples_condition2_l988_98808

-- Define a function to check if a number is representable as the sum of two squares
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

-- Theorem for the first condition
theorem infinitely_many_triples_condition1 :
  ∃ f : ℕ → ℕ, Monotone f ∧
    ∀ n : ℕ, is_sum_of_two_squares (f n) ∧
    ¬is_sum_of_two_squares (f n - 1) ∧
    ¬is_sum_of_two_squares (f n + 1) :=
sorry

-- Theorem for the second condition
theorem infinitely_many_triples_condition2 :
  ∃ g : ℕ → ℕ, Monotone g ∧
    ∀ n : ℕ, is_sum_of_two_squares (g n - 1) ∧
    is_sum_of_two_squares (g n) ∧
    is_sum_of_two_squares (g n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_triples_condition1_infinitely_many_triples_condition2_l988_98808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_ratio_l988_98864

/-- Given a tank of fish with the following properties:
  * There are 30 fish in total
  * One-third of the fish are blue
  * There are 5 blue, spotted fish
  Prove that the ratio of blue, spotted fish to the total number of blue fish is 1:2 -/
theorem fish_tank_ratio : 
  (5 : ℚ) / (30 / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_ratio_l988_98864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_solutions_l988_98879

/-- The equation has exactly two different solutions in the interval (0, π/2) 
    if and only if a is in the specified range -/
theorem equation_two_solutions (a : ℝ) : 
  (∃! (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ t₁ ∈ Set.Ioo 0 (π/2) ∧ t₂ ∈ Set.Ioo 0 (π/2) ∧ 
    ((4 * a * Real.sin t₁^2 + 4 * a * (1 + 2 * Real.sqrt 2) * Real.cos t₁ - 
      4 * (a - 1) * Real.sin t₁ - 5 * a + 2) / 
     (2 * Real.sqrt 2 * Real.cos t₁ - Real.sin t₁) = 4 * a) ∧
    ((4 * a * Real.sin t₂^2 + 4 * a * (1 + 2 * Real.sqrt 2) * Real.cos t₂ - 
      4 * (a - 1) * Real.sin t₂ - 5 * a + 2) / 
     (2 * Real.sqrt 2 * Real.cos t₂ - Real.sin t₂) = 4 * a)) ↔ 
  (a ∈ Set.Ioo 6 (18 + 24 * Real.sqrt 2) ∪ Set.Ioi (18 + 24 * Real.sqrt 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_solutions_l988_98879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l988_98869

-- Define the statements
def statement1 : Prop := ∃ x : ℝ, x^2 = 0.81 ∧ x = 0.09
def statement2 : Prop := ∃ x : ℝ, x^2 = -9 ∧ (x = 3 ∨ x = -3)
def statement3 : Prop := Real.sqrt ((-5)^2) = -5
def statement4 : Prop := ∃ x : ℝ, x^2 = -2 ∧ x < 0
def statement5 : Prop := (0 : ℝ)⁻¹ = 0 ∧ -(0 : ℝ) = 0
def statement6 : Prop := ∀ x : ℝ, x^2 = 4 → (x = 2 ∨ x = -2)
def statement7 : Prop := ∀ x : ℝ, x^(1/3 : ℝ) = x → (x = 1 ∨ x = 0)
def statement8 : Prop := ∀ x : ℝ, ∃! y : ℝ, y = x -- Simplified representation of one-to-one correspondence

theorem correct_statements :
  ¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧
  ¬statement5 ∧ ¬statement6 ∧ ¬statement7 ∧ statement8 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l988_98869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l988_98863

noncomputable def x (t : ℝ) : ℝ := 3 * (Real.cos t + t * Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 3 * (Real.sin t - t * Real.cos t)

noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

theorem curve_arc_length :
  arcLength 0 (π/3) = π^2/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l988_98863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l988_98899

noncomputable def f (x : ℝ) : ℝ := 3 * (1 - x) * Real.log (1 + x) + Real.sin (Real.pi * x)

theorem root_difference_bound (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ ∈ Set.Icc 0 1 →
  x₂ ∈ Set.Icc 0 1 →
  x₁ ≠ x₂ →
  f x₁ = m →
  f x₂ = m →
  |x₁ - x₂| ≤ 1 - (2 * m) / (Real.pi + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l988_98899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_unit_circle_l988_98846

theorem cos_double_angle_unit_circle (α : ℝ) :
  (-(Real.sqrt 5) / 5 : ℝ) ^ 2 + ((2 * Real.sqrt 5) / 5 : ℝ) ^ 2 = 1 →
  Real.cos α = -(Real.sqrt 5) / 5 →
  Real.cos (2 * α) = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_unit_circle_l988_98846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l988_98883

/-- Hyperbola structure -/
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  passesThrough : ℝ × ℝ

/-- Point on the hyperbola -/
structure PointOnHyperbola where
  x : ℝ
  y : ℝ

/-- Given hyperbola -/
noncomputable def givenHyperbola : Hyperbola :=
  { center := (0, 0)
  , eccentricity := Real.sqrt 2
  , passesThrough := (4, -Real.sqrt 10) }

/-- Point M on the hyperbola -/
noncomputable def pointM : PointOnHyperbola :=
  { x := 3
  , y := Real.sqrt 3 }

/-- Theorem statement -/
theorem hyperbola_properties (h : Hyperbola) (m : PointOnHyperbola) 
    (hc : h = givenHyperbola) (hm : m = pointM) :
  let f1 := (Real.sqrt 6, 0)
  let f2 := (-Real.sqrt 6, 0)
  /- 1. Equation of the hyperbola -/
  ((fun (x y : ℝ) => x^2 - y^2 = 6) = 
   (fun (x y : ℝ) => (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / 3) - (p.2^2 / 3) = 1})) ∧ 
  /- 2. Point M is on the circle with diameter F₁F₂ -/
  (((m.x - f1.1) * (m.x - f2.1) + (m.y - f1.2) * (m.y - f2.2)) = 0) ∧
  /- 3. Area of triangle △F₁MF₂ -/
  (1/2 * Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2) * |m.y| = 3 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l988_98883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l988_98840

theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x : ℝ, x ≠ 0 → f x = (x^2 + 1) * (1/x - 1)^5) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  c = -11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l988_98840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_40_equation_l988_98844

theorem angle_40_equation (x y : ℤ) :
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = ↑x + ↑y / Real.cos (40 * π / 180) →
  x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_40_equation_l988_98844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l988_98858

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : Real.log (4/5) / Real.log a < 1) : a ∈ Set.Ioo 0 (4/5) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l988_98858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_one_l988_98849

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
structure Focus where
  x : ℝ
  y : ℝ

-- Define the intersection point
structure IntersectionPoint where
  x : ℝ
  y : ℝ

-- Define perpendicularity to x-axis
def perpendicularToXAxis (f : Focus) (a : IntersectionPoint) : Prop :=
  f.x = a.x

-- Define eccentricity of hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2/a^2)

-- Theorem statement
theorem hyperbola_eccentricity_is_one 
  (p a b : ℝ) 
  (f : Focus) 
  (i : IntersectionPoint) 
  (h1 : parabola p i.x i.y)
  (h2 : hyperbola a b i.x i.y)
  (h3 : perpendicularToXAxis f i) :
  eccentricity a b = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_one_l988_98849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_increasing_function_l988_98850

/-- If y = ax + sin x is monotonically increasing on ℝ, then a ≥ 1 -/
theorem min_a_for_increasing_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x ↦ a * x + Real.sin x)) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_increasing_function_l988_98850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strengthening_cultural_construction_l988_98865

theorem strengthening_cultural_construction :
  ∀ (country : Type) (strength : country → ℕ),
  ∃ (cultural_strength : country → ℕ),
  ∀ (c : country), cultural_strength c ≥ strength c :=
by
  sorry

#check strengthening_cultural_construction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strengthening_cultural_construction_l988_98865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relation_l988_98897

theorem cosine_sine_relation (α m : ℝ) :
  Real.sin (π / 4 - α) = m → Real.cos (π / 4 + α) = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relation_l988_98897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_circle_radius_correct_l988_98875

/-- The radius of a circle externally tangent to three circles inscribed in a 3x3 grid -/
noncomputable def external_circle_radius : ℝ := (5 * Real.sqrt 2 - 3) / 6

/-- Centers of the three inscribed circles -/
noncomputable def circle_centers : Fin 3 → ℝ × ℝ
  | 0 => (1/2, 1/2)    -- lower left
  | 1 => (3/2, 5/2)    -- middle top
  | 2 => (5/2, 3/2)    -- right middle
  | _ => (0, 0)        -- default case (never reached)

/-- Side lengths of the triangle formed by the circle centers -/
noncomputable def triangle_sides : Fin 3 → ℝ
  | 0 => Real.sqrt 5   -- between (1/2, 1/2) and (3/2, 5/2)
  | 1 => Real.sqrt 2   -- between (3/2, 5/2) and (5/2, 3/2)
  | 2 => Real.sqrt 5   -- between (5/2, 3/2) and (1/2, 1/2)
  | _ => 0             -- default case (never reached)

theorem external_circle_radius_correct :
  ∃ (r : ℝ), r = external_circle_radius ∧
  ∀ (i : Fin 3), ∃ (c : ℝ × ℝ), c = circle_centers i ∧
  ∀ (j : Fin 3), ∃ (s : ℝ), s = triangle_sides j ∧
  (∃ (R : ℝ), R = (s * s * s) / (4 * (3/2)) ∧ R = 1/2 + r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_circle_radius_correct_l988_98875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l988_98827

/-- An isosceles triangle has a base of length b and an altitude from the apex to the base of length h.
    A rectangle of height x is inscribed in this triangle such that its base lies along the base of the triangle.
    The lateral sides of the rectangle are parallel to the sides of the triangle.
    The area of the rectangle is (b*x^2)/(2h). -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) :
  let y := (b * x) / (2 * h)
  (x * y) = (b * x^2) / (2 * h) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l988_98827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_most_cost_effective_l988_98828

/-- Represents the size of a juice container -/
inductive JuiceSize
  | Small
  | Medium
  | Large

/-- Represents the cost and quantity of juice for a given size -/
structure JuiceInfo where
  cost : ℝ
  quantity : ℝ

/-- Function to calculate the cost per liter of juice -/
noncomputable def costPerLiter (info : JuiceInfo) : ℝ :=
  info.cost / info.quantity

/-- Theorem stating that Medium is the most cost-effective, followed by Small, then Large -/
theorem medium_most_cost_effective (s m l : JuiceInfo) :
  m.cost = 1.3 * s.cost →
  l.cost = 1.6 * m.cost →
  l.quantity = 1.5 * s.quantity →
  m.quantity = 0.9 * l.quantity →
  costPerLiter m < costPerLiter s ∧ costPerLiter s < costPerLiter l :=
by
  sorry

#check medium_most_cost_effective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medium_most_cost_effective_l988_98828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l988_98898

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

noncomputable def right_focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

theorem distance_right_focus_to_line :
  let a := Real.sqrt 4
  let b := Real.sqrt 5
  let focus := right_focus a b
  distance_point_to_line focus.1 focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l988_98898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_parity_f_l988_98817

def f (n : ℕ) : ℕ := sorry

theorem different_parity_f (n : ℕ) (h : n > 1) : 
  (f n % 2 = 1) ≠ (f (2015 * n) % 2 = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_parity_f_l988_98817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8_plus_abs_2_times_cos_45_l988_98842

theorem sqrt_8_plus_abs_2_times_cos_45 :
  Real.sqrt 8 + abs (-2) * Real.cos (π / 4) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8_plus_abs_2_times_cos_45_l988_98842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_9_l988_98876

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 3  -- Add this case to handle 0
  | 1 => 3
  | 2 => 7
  | n + 3 => (a (n + 1) * a (n + 2)) % 10

-- State the theorem
theorem a_2010_equals_9 : a 2010 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_9_l988_98876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l988_98835

-- Define the function f
def f (x b c : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem function_properties :
  ∀ (b c : ℝ),
  (∀ x, f x b c = f (-x) b c) →  -- f is even
  f 1 b c = 0 →                  -- f(1) = 0
  (∃ (max min : ℝ),
    (∀ x, x ∈ Set.Icc (-1) 3 → f x b c ≤ max) ∧
    (∃ x, x ∈ Set.Icc (-1) 3 ∧ f x b c = max) ∧
    (∀ x, x ∈ Set.Icc (-1) 3 → f x b c ≥ min) ∧
    (∃ x, x ∈ Set.Icc (-1) 3 ∧ f x b c = min) ∧
    max = 8 ∧ min = -1) ∧
  (b ≥ 2 ∨ b ≤ -6 ↔
    (∀ x y, x ∈ Set.Icc (-1) 3 → y ∈ Set.Icc (-1) 3 → x < y → f x b c < f y b c) ∨
    (∀ x y, x ∈ Set.Icc (-1) 3 → y ∈ Set.Icc (-1) 3 → x < y → f x b c > f y b c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l988_98835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_11pi_3_l988_98887

noncomputable def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + Real.pi) = f x + Real.cos x

theorem f_value_at_11pi_3
  (f : ℝ → ℝ)
  (h1 : periodic_function f)
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f x = 0) :
  f (11 * Real.pi / 3) = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_11pi_3_l988_98887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_lighthouse_height_approx_l988_98815

noncomputable section

-- Define the actual lighthouse parameters
def actual_height : ℝ := 50
def actual_visibility : ℝ := 30000  -- 30 km in meters

-- Define the model lighthouse visibility
def model_visibility : ℝ := 0.3  -- 30 cm in meters

-- Define the scale factor
def scale_factor : ℝ := (actual_visibility / model_visibility) ^ (1/3)

-- Define the model height
def model_height : ℝ := actual_height / scale_factor

-- Theorem to prove
theorem model_lighthouse_height_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |model_height - 1.08| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_lighthouse_height_approx_l988_98815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l988_98822

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_condition (t : Triangle) : Prop :=
  (2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A

-- Define the function y
noncomputable def y (t : Triangle) : Real → Real :=
  λ x => Real.sqrt 3 * Real.sin t.B + Real.sin (t.C - Real.pi/6)

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.A = Real.pi/3 ∧ Set.Icc (Real.sqrt 3) 2 = Set.range (y t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l988_98822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_can_determine_score_l988_98826

/-- Represents the score of a football match -/
structure Score :=
  (home : Nat)
  (away : Nat)

/-- The possible scores given the conditions -/
def possible_scores : List Score := [⟨2, 0⟩, ⟨1, 0⟩]

/-- Condition 1: The total number of goals is sufficient to determine the score -/
def total_goals_sufficient (scores : List Score) : Prop :=
  ∀ s₁ s₂, s₁ ∈ scores → s₂ ∈ scores → s₁.home + s₁.away = s₂.home + s₂.away → s₁ = s₂

/-- Condition 2: If the losing team scored one more goal, 
    the total number of goals would still be sufficient to determine the score -/
def total_goals_sufficient_with_extra (scores : List Score) : Prop :=
  ∀ s₁ s₂, s₁ ∈ scores → s₂ ∈ scores → 
    (s₁.home < s₁.away → s₁.home + 1 + s₁.away = s₂.home + s₂.away → s₁ = s₂) ∧
    (s₁.away < s₁.home → s₁.home + s₁.away + 1 = s₂.home + s₂.away → s₁ = s₂)

/-- Condition 3: Shinnik scored at least one goal -/
def shinnik_scored (scores : List Score) : Prop :=
  ∀ s, s ∈ scores → s.home > 0

/-- Condition 4: There must be a winner (no draw) -/
def no_draw (scores : List Score) : Prop :=
  ∀ s, s ∈ scores → s.home ≠ s.away

/-- Main theorem: Given the conditions, there exists a question that allows Petya
    to determine the exact score of the match -/
theorem petya_can_determine_score : 
  total_goals_sufficient possible_scores ∧
  total_goals_sufficient_with_extra possible_scores ∧
  shinnik_scored possible_scores ∧
  no_draw possible_scores →
  ∃ (q : Bool → Bool), ∀ s, s ∈ possible_scores → ∃! result, q result = true ∧ s ∈ possible_scores :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_can_determine_score_l988_98826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l988_98830

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : (2 * abc.a + abc.c) * ((abc.B.1 - abc.A.1) * (abc.C.1 - abc.A.1) + (abc.B.2 - abc.A.2) * (abc.C.2 - abc.A.2)) 
      = abc.c * ((abc.C.1 - abc.B.1) * (abc.A.1 - abc.B.1) + (abc.C.2 - abc.B.2) * (abc.A.2 - abc.B.2))) :
  -- Part 1: Measure of angle B is 2π/3
  let angle_B := Real.arccos (((abc.A.1 - abc.B.1) * (abc.C.1 - abc.B.1) + (abc.A.2 - abc.B.2) * (abc.C.2 - abc.B.2)) 
    / (((abc.A.1 - abc.B.1)^2 + (abc.A.2 - abc.B.2)^2).sqrt * ((abc.C.1 - abc.B.1)^2 + (abc.C.2 - abc.B.2)^2).sqrt))
  angle_B = 2 * Real.pi / 3 ∧
  -- Part 2: If b = √6, then 0 < area ≤ √3/2
  abc.b = Real.sqrt 6 →
    let s := abc.a * abc.c * Real.sin angle_B / 2
    0 < s ∧ s ≤ Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l988_98830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_subset_max_integers_2022_l988_98839

def S (k : ℤ) : Finset ℤ := {6*k + 1, 6*k + 2, 6*k + 3, 6*k + 4, 6*k + 5, 6*k + 6}

theorem difference_in_subset (k : ℤ) (A : Finset ℤ) (h : A ⊆ S k) (h_size : A.card ≥ 3) :
  ∃ (x y : ℤ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (y - x ∈ ({1, 4, 5} : Finset ℤ) ∨ x - y ∈ ({1, 4, 5} : Finset ℤ)) := by
  sorry

def max_integers_with_condition (n : ℕ) : ℕ :=
  (n / 3 : ℕ)

theorem max_integers_2022 :
  max_integers_with_condition 2022 = 674 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_subset_max_integers_2022_l988_98839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l988_98853

/-- The length of the minor axis of the ellipse x²/16 + y²/9 = 1 is 6 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 16 + y^2 / 9 = 1}
  ∃ (minor_axis : ℝ), minor_axis = 6 ∧
    ∀ (p q : ℝ × ℝ), p ∈ ellipse → q ∈ ellipse →
      (p.2 = q.2 ∧ p.2 = 0 ∨ p.1 = q.1 ∧ p.1 = 0) →
      dist p q ≤ minor_axis :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l988_98853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_make_all_white_l988_98880

/-- Represents the color of a cell -/
inductive Color
| Black
| White
deriving Repr, DecidableEq

/-- Represents a 3x3 grid -/
def Grid := Fin 3 → Fin 3 → Color

/-- Represents a flip operation (row or column) -/
inductive Flip
| Row (i : Fin 3)
| Col (j : Fin 3)

/-- Initial grid with one black corner and others white -/
def initial_grid : Grid :=
  fun i j => if i = 0 ∧ j = 0 then Color.Black else Color.White

/-- Apply a single flip operation to a grid -/
def apply_flip (g : Grid) (f : Flip) : Grid :=
  match f with
  | Flip.Row i => fun r c => if r = i then (if g r c = Color.Black then Color.White else Color.Black) else g r c
  | Flip.Col j => fun r c => if c = j then (if g r c = Color.Black then Color.White else Color.Black) else g r c

/-- Apply a sequence of flips to a grid -/
def apply_flips (g : Grid) : List Flip → Grid
| [] => g
| (f::fs) => apply_flips (apply_flip g f) fs

/-- Check if all cells in a grid are white -/
def all_white (g : Grid) : Prop :=
  ∀ i j, g i j = Color.White

/-- The main theorem: it's impossible to make all cells white -/
theorem impossible_to_make_all_white :
  ¬∃ (flips : List Flip), all_white (apply_flips initial_grid flips) := by
  sorry

#eval initial_grid 0 0
#eval initial_grid 1 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_make_all_white_l988_98880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l988_98845

/-- A polynomial of degree 5 with real coefficients -/
def Polynomial5 (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), ∀ x, p x = x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀

/-- The property that if α is a root of p, then 1/α and 1-α are also roots -/
def RootProperty (p : ℝ → ℝ) : Prop :=
  ∀ α : ℝ, p α = 0 → p (1/α) = 0 ∧ p (1-α) = 0

/-- The list of possible polynomial forms -/
def PossibleForms (p : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧
    ((p = λ x ↦ k * (x+1)*(x-1/2)*(x-2)*(x^2 - x + 1)) ∨
     (p = λ x ↦ k * (x+1)^3*(x-1/2)*(x-2)) ∨
     (p = λ x ↦ k * (x+1)*(x-1/2)^3*(x-2)) ∨
     (p = λ x ↦ k * (x+1)*(x-1/2)*(x-2)^3) ∨
     (p = λ x ↦ k * (x+1)^2*(x-1/2)^2*(x-2)) ∨
     (p = λ x ↦ k * (x+1)*(x-1/2)^2*(x-2)^2) ∨
     (p = λ x ↦ k * (x+1)^2*(x-1/2)*(x-2)^2) ∨
     (p = λ x ↦ k * (x-1)^5))

/-- The main theorem -/
theorem polynomial_characterization (p : ℝ → ℝ) :
  Polynomial5 p → RootProperty p → PossibleForms p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l988_98845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l988_98881

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x * Real.log x

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- State the theorem
theorem tangent_perpendicular_to_line (x₀ : ℝ) (h : x₀ > 0) :
  let y₀ := curve x₀
  let slope := Real.log x₀ + 1
  (slope * (-1) = 1) →  -- Perpendicularity condition
  tangent_line x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l988_98881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l988_98820

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The area of a square given its side length -/
def squareArea (side : ℝ) : ℝ :=
  side^2

/-- The perimeter of a square given its side length -/
def squarePerimeter (side : ℝ) : ℝ :=
  4 * side

theorem square_area_perimeter_product :
  let e : Point := ⟨5, 5⟩
  let f : Point := ⟨5, 1⟩
  let g : Point := ⟨1, 1⟩
  let h : Point := ⟨1, 5⟩
  let side := distance e h
  (squareArea side) * (squarePerimeter side) = 256 := by
  sorry

#eval squareArea 4 * squarePerimeter 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_perimeter_product_l988_98820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_perimeter_le_outer_perimeter_l988_98893

/-- A rectangle represented by its vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculate the perimeter of a rectangle -/
noncomputable def perimeter (r : Rectangle) : ℝ :=
  let AB := ((r.B.1 - r.A.1)^2 + (r.B.2 - r.A.2)^2).sqrt
  let BC := ((r.C.1 - r.B.1)^2 + (r.C.2 - r.B.2)^2).sqrt
  2 * (AB + BC)

/-- Predicate to check if one rectangle is contained within another -/
def contained_in (inner outer : Rectangle) : Prop :=
  ∀ (p : ℝ × ℝ), (p = inner.A ∨ p = inner.B ∨ p = inner.C ∨ p = inner.D) →
    outer.A.1 ≤ p.1 ∧ p.1 ≤ outer.C.1 ∧ outer.A.2 ≤ p.2 ∧ p.2 ≤ outer.C.2

/-- Theorem: The perimeter of an inner rectangle is always less than or equal to the perimeter of the outer rectangle -/
theorem inner_perimeter_le_outer_perimeter (inner outer : Rectangle) 
    (h : contained_in inner outer) : 
    perimeter inner ≤ perimeter outer := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_perimeter_le_outer_perimeter_l988_98893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_relation_l988_98825

/-- The ellipse C in the Cartesian coordinate system -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point A on the ellipse C -/
def A (x₁ y₁ : ℝ) : Prop := C x₁ y₁ ∧ x₁ ≠ 0 ∧ y₁ ≠ 0

/-- Point B on the ellipse C, opposite to A -/
def B (x₁ y₁ : ℝ) : Prop := C (-x₁) (-y₁)

/-- Point D on the ellipse C -/
def D (x₂ y₂ : ℝ) : Prop := C x₂ y₂

/-- AD is perpendicular to AB -/
def AD_perp_AB (x₁ y₁ x₂ y₂ : ℝ) : Prop := 
  (y₂ - y₁) * (2 * x₁) + (x₂ - x₁) * (2 * y₁) = 0

/-- M is the intersection of BD and x-axis -/
def M : Prop := True  -- M(3x₁, 0) is always true for any x₁

/-- The slope of BD -/
noncomputable def k₁ (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ + y₁) / (x₂ + x₁)

/-- The slope of AM -/
noncomputable def k₂ (x₁ y₁ : ℝ) : ℝ := y₁ / (3 * x₁)

theorem ellipse_slope_relation (x₁ y₁ x₂ y₂ : ℝ) :
  A x₁ y₁ → B x₁ y₁ → D x₂ y₂ → AD_perp_AB x₁ y₁ x₂ y₂ → M →
  k₁ x₁ y₁ x₂ y₂ = -1/2 * k₂ x₁ y₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_relation_l988_98825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l988_98832

/-- The function J for nonzero real numbers x, y, and z -/
noncomputable def J (x y z : ℝ) : ℝ := x / y + y / z + z / x

/-- Theorem stating that J(-3, 18, 12) = -8/3 -/
theorem J_specific_value : J (-3) 18 12 = -8/3 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform numerical calculations
  norm_num
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l988_98832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_proof_l988_98841

/-- The number of candy bars -/
noncomputable def total_candy_bars : ℝ := 5.0

/-- The number of people -/
noncomputable def number_of_people : ℝ := 3.0

/-- The number of candy bars each person gets -/
noncomputable def candy_bars_per_person : ℝ := total_candy_bars / number_of_people

/-- Proof that the number of candy bars per person is approximately 1.67 -/
theorem candy_distribution_proof : 
  |candy_bars_per_person - 1.67| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_proof_l988_98841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_product_sum_l988_98805

/-- Represents a cubic equation of the form x³ + px² + qx + r = 0 -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The two cubic equations from the problem -/
def eq1 (C : ℝ) : CubicEquation := ⟨0, C, -20⟩
def eq2 (D : ℝ) : CubicEquation := ⟨D, 0, 80⟩

/-- Predicate to check if two cubic equations have two common roots -/
def has_two_common_roots (eq1 eq2 : CubicEquation) : Prop := sorry

/-- The product of the common roots -/
noncomputable def common_roots_product (eq1 eq2 : CubicEquation) : ℝ := sorry

/-- Theorem stating the main result -/
theorem common_roots_product_sum (C D : ℝ) :
  has_two_common_roots (eq1 C) (eq2 D) →
  ∃ (a b c : ℕ), 
    common_roots_product (eq1 C) (eq2 D) = (a : ℝ) * (c ^ (1 / (b : ℝ))) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 17 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_product_sum_l988_98805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_win_probability_l988_98882

/-- Represents a player in the coin-flipping game -/
inductive Player : Type
| Ana : Player
| Carlos : Player
| Juan : Player

/-- The probability of flipping heads -/
noncomputable def headsProbability : ℝ := 1 / 2

/-- The order of players in the game -/
def playerOrder : List Player := [Player.Ana, Player.Carlos, Player.Juan]

/-- The probability that Juan wins the game -/
noncomputable def juanWinProbability : ℝ := 1 / 7

/-- Theorem stating that the probability of Juan winning the game is 1/7 -/
theorem juan_win_probability :
  juanWinProbability = 1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_juan_win_probability_l988_98882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_cosine_l988_98833

theorem triangle_sine_cosine (A B C : ℝ) : 
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  Real.sin A = 5/13 →
  Real.cos B = 3/5 →
  Real.sin C = 63/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_cosine_l988_98833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_approx_3_l988_98886

/-- Calculates the diameter of a cylindrical well given its depth, cost per cubic meter, and total cost. -/
noncomputable def well_diameter (depth : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let volume := total_cost / cost_per_cubic_meter
  let radius := Real.sqrt (volume / (Real.pi * depth))
  2 * radius

/-- The calculated diameter of the well is approximately 3 meters. -/
theorem well_diameter_approx_3 :
  let depth : ℝ := 14
  let cost_per_cubic_meter : ℝ := 15
  let total_cost : ℝ := 1484.40
  ∃ ε > 0, |well_diameter depth cost_per_cubic_meter total_cost - 3| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_diameter_approx_3_l988_98886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_has_four_wheels_l988_98896

-- Define the universe of discourse
def Object : Type := String

-- Define predicates
def isCar (x : Object) : Prop := true
def hasFourWheels (x : Object) : Prop := true
def isBicycle (x : Object) : Prop := true

-- State the theorem
theorem bicycle_has_four_wheels 
  (h1 : ∀ x, isCar x → hasFourWheels x) 
  (h2 : ∀ x, isBicycle x → isCar x) :
  ∀ x, isBicycle x → hasFourWheels x :=
by
  intro x
  intro h_bicycle
  apply h1
  apply h2
  exact h_bicycle

-- Add this line to make the file a valid Lean module
#check bicycle_has_four_wheels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_has_four_wheels_l988_98896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l988_98848

theorem existence_of_special_set (n : ℕ) (h : n > 1) :
  ∃ (S : Finset ℤ), (Finset.card S = n) ∧
  (∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l988_98848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_validWords_eq_recurrence_l988_98814

/-- The number of valid words of length n formed from letters a, b, and c,
    where no two a letters are adjacent. -/
noncomputable def validWords (n : ℕ) : ℝ :=
  let x : ℝ := Real.sqrt 3
  ((2 + x) / (2 * x)) * (1 + x)^n + ((-2 + x) / (2 * x)) * (1 - x)^n

/-- The recurrence relation for the number of valid words. -/
def validWordsRecurrence : ℕ → ℝ
  | 0 => 1  -- Base case: empty word
  | 1 => 3  -- Base case: a, b, c
  | n + 2 => 2 * validWordsRecurrence (n + 1) + 2 * validWordsRecurrence n

/-- Theorem stating that the explicit formula equals the recurrence relation. -/
theorem validWords_eq_recurrence :
  ∀ n : ℕ, validWords n = validWordsRecurrence n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_validWords_eq_recurrence_l988_98814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_ratio_l988_98872

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + ¼c², then a cos B / c = 5/8 -/
theorem triangle_cosine_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 = b^2 + (1/4)*c^2 →  -- Given condition
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) →  -- Cosine rule
  a * Real.cos B / c = 5/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_ratio_l988_98872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l988_98888

/-- The line equation: 3x + y - 6 = 0 -/
def line (x y : ℝ) : Prop := 3 * x + y - 6 = 0

/-- The circle equation: x² + y² - 2y - 4 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

/-- The length of the line segment AB, where A and B are the intersection points of the line and the circle -/
noncomputable def length_AB : ℝ := Real.sqrt 10

/-- Theorem stating that the length of AB is √10 -/
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = length_AB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l988_98888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_solution_l988_98852

/-- The sum of the series 3 + 7y + 11y^2 + 15y^3 + ... -/
noncomputable def seriesSum (y : ℝ) : ℝ := 3 + (7 * y) / (1 - y)

/-- Theorem: If the sum of the series equals 100, then y = 79/100 -/
theorem series_sum_solution :
  ∃ y : ℝ, y > 0 ∧ y < 1 ∧ seriesSum y = 100 → y = 79 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_solution_l988_98852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l988_98800

def U : Set Int := {-2, -1, 0, 1, 2, 3, 4, 5, 6}

def M : Set Int := {x : Int | x > -2 ∧ x < 5}

theorem complement_of_M : U \ M = {-2, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l988_98800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l988_98812

theorem complex_equality (x y : ℝ) : 
  (Complex.I * (x - Complex.I) = y + 2 * Complex.I) → 
  (Complex.ofReal x + Complex.I * Complex.ofReal y = 2 + Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l988_98812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_measurement_l988_98804

/-- Represents a brick with length, width, and height -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the diagonal of a brick -/
noncomputable def brickDiagonal (b : Brick) : ℝ :=
  Real.sqrt (b.length ^ 2 + b.width ^ 2 + b.height ^ 2)

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

/-- Represents the arrangement of three bricks -/
structure BrickArrangement where
  brick : Brick
  pointA : Point3D
  pointB : Point3D

/-- Theorem: The distance between points A and B in the specific arrangement
    of three identical bricks is equal to the diagonal of a single brick -/
theorem brick_diagonal_measurement (arr : BrickArrangement) :
  distance3D arr.pointA arr.pointB = brickDiagonal arr.brick := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_diagonal_measurement_l988_98804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_difference_l988_98859

/-- Represents the characteristics of a car's fuel efficiency --/
structure CarEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving --/
def mpg_difference (car : CarEfficiency) : ℚ :=
  let tank_capacity := car.city_miles_per_tankful / car.city_miles_per_gallon
  let highway_mpg := car.highway_miles_per_tankful / tank_capacity
  highway_mpg - car.city_miles_per_gallon

/-- Theorem stating the difference in miles per gallon for the given car --/
theorem car_efficiency_difference :
  ∃ (car : CarEfficiency),
    car.highway_miles_per_tankful = 462 ∧
    car.city_miles_per_tankful = 336 ∧
    car.city_miles_per_gallon = 48 ∧
    mpg_difference car = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_difference_l988_98859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l988_98856

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

-- State the theorem
theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2) ∧
  (∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≤ 1 / 4) ∧
  (∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≥ -1 / 2) ∧
  (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = 1 / 4) ∧
  (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = -1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l988_98856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_simplification_l988_98829

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  3 * x + 6 ≥ 5 * (x - 2) ∧ (x - 5) / 2 - (4 * x - 3) / 3 < 1

theorem solution_set : 
  ∀ x : ℝ, inequality_system x ↔ -3 < x ∧ x ≤ 8 :=
by sorry

-- Part 2: Expression simplification
noncomputable def original_expression (x : ℝ) : ℝ :=
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x - 1))

theorem simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  original_expression x = 1 / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_simplification_l988_98829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marina_IIA_return_l988_98857

/-- Calculates the annual return on Marina's IIA transactions over three years --/
noncomputable def annual_return_IIA (monthly_salary : ℝ) (tax_rate : ℝ) 
  (contribution_year1 : ℝ) (contribution_year2 : ℝ) (contribution_year3 : ℝ) 
  (max_annual_deduction : ℝ) (deduction_rate : ℝ) : ℝ :=
  let annual_salary := monthly_salary * 12
  let annual_tax := annual_salary * tax_rate
  let deduction_year1 := min (contribution_year1 * deduction_rate) annual_tax
  let deduction_year2 := min (contribution_year2 * deduction_rate) annual_tax
  let deduction_year3 := min (contribution_year3 * deduction_rate) annual_tax
  let total_deduction := deduction_year1 + deduction_year2 + deduction_year3
  let total_contribution := contribution_year1 + contribution_year2 + contribution_year3
  let total_return := total_deduction / total_contribution * 100
  total_return / 3

/-- Theorem stating that Marina's annual return on IIA transactions is approximately 3.55% --/
theorem marina_IIA_return : 
  ∃ ε > 0, |annual_return_IIA 30000 0.13 100000 400000 400000 400000 0.13 - 3.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marina_IIA_return_l988_98857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorization_l988_98809

theorem unique_factorization (m : ℕ) (h_mod : m % 4 = 2) :
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ m = a * b ∧ 0 < a - b ∧ (a - b : ℝ) < Real.sqrt (5 + 4 * Real.sqrt (4 * (m : ℝ) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_factorization_l988_98809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_c_for_degree_four_l988_98801

noncomputable def f (x : ℝ) : ℝ := 2 - 5*x + 4*x^2 - 3*x^3 + 7*x^5
noncomputable def g (x : ℝ) : ℝ := 1 - 7*x + 5*x^2 - 2*x^4 + 9*x^5

theorem unique_c_for_degree_four :
  ∃! c : ℝ, ∃ p : Polynomial ℝ,
    (∀ x : ℝ, f x + c * g x = p.eval x) ∧
    p.degree = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_c_for_degree_four_l988_98801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l988_98819

def sequenceA (a₀ a₁ a₂ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | 2 => a₂
  | (n + 3) => 3 * sequenceA a₀ a₁ a₂ (n + 2) - 3 * sequenceA a₀ a₁ a₂ (n + 1) + sequenceA a₀ a₁ a₂ n

theorem sequence_perfect_squares (a₀ a₁ a₂ : ℤ) 
  (h1 : 2 * a₁ = a₀ + a₂ - 2)
  (h2 : ∀ m : ℕ, ∃ k : ℕ, ∀ i : ℕ, i < m → ∃ y : ℤ, sequenceA a₀ a₁ a₂ (k + i) = y * y) :
  ∃ l : ℤ, ∀ n : ℕ, sequenceA a₀ a₁ a₂ n = (n + l) * (n + l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_perfect_squares_l988_98819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CGE_equals_area_BFG_l988_98810

open Geometry

-- Define the points
variable (A B C D E F G O : EuclideanSpace ℝ (Fin 2))

-- Define the trapezoid ABCD
def is_trapezoid (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D O : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the extension of AC to E
def extend_AC_to_E (A C E O : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist C E = dist A O

-- Define the extension of DB to F
def extend_DB_to_F (B D F O : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist B F = dist D O

-- Define the area of a triangle
noncomputable def triangle_area (X Y Z : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem area_CGE_equals_area_BFG 
  (h1 : is_trapezoid A B C D)
  (h2 : diagonals_intersect A B C D O)
  (h3 : extend_AC_to_E A C E O)
  (h4 : extend_DB_to_F B D F O)
  (h5 : triangle_area B F G = 2015) :
  triangle_area C G E = triangle_area B F G :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CGE_equals_area_BFG_l988_98810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factorization_distinct_positive_factors_l988_98847

/-- The number of distinct, positive factors of 1320 -/
def num_factors : ℕ := 24

/-- 1320 is the product of its prime factors -/
theorem prime_factorization : 1320 = 2^2 * 3 * 5 * 11 := by sorry

/-- The number of distinct, positive factors of 1320 is 24 -/
theorem distinct_positive_factors : num_factors = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factorization_distinct_positive_factors_l988_98847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_to_concentric_inversion_circle_line_to_concentric_l988_98843

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in a plane
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define an inversion transformation
noncomputable def Inversion (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) → (ℝ × ℝ) :=
  fun p => sorry

-- Define membership for Circle
def CircleMembership (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define membership for Line
def LineMembership (p : ℝ × ℝ) (l : Line) : Prop :=
  let (x, y) := p
  let (px, py) := l.point
  let (dx, dy) := l.direction
  (x - px) * dy = (y - py) * dx

-- Theorem statement
theorem inversion_to_concentric
  (S₁ S₂ : Circle)
  (h_non_intersect : ¬ (∃ (p : ℝ × ℝ), CircleMembership p S₁ ∧ CircleMembership p S₂)) :
  ∃ (I : (ℝ × ℝ) → (ℝ × ℝ)) (c : ℝ × ℝ),
    (∃ (center : ℝ × ℝ) (radius : ℝ), I = Inversion center radius) ∧
    (∃ (r₁ r₂ : ℝ), (∀ p, CircleMembership (I p) (Circle.mk c r₁) ↔ CircleMembership p S₁) ∧
                    (∀ p, CircleMembership (I p) (Circle.mk c r₂) ↔ CircleMembership p S₂)) :=
by sorry

-- Alternative theorem for circle and line case
theorem inversion_circle_line_to_concentric
  (S : Circle) (L : Line)
  (h_non_intersect : ¬ (∃ (p : ℝ × ℝ), CircleMembership p S ∧ LineMembership p L)) :
  ∃ (I : (ℝ × ℝ) → (ℝ × ℝ)) (c : ℝ × ℝ),
    (∃ (center : ℝ × ℝ) (radius : ℝ), I = Inversion center radius) ∧
    (∃ (r₁ r₂ : ℝ), (∀ p, CircleMembership (I p) (Circle.mk c r₁) ↔ CircleMembership p S) ∧
                    (∀ p, CircleMembership (I p) (Circle.mk c r₂) ↔ LineMembership p L)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_to_concentric_inversion_circle_line_to_concentric_l988_98843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_tan_x_inequality_l988_98877

theorem sin_x_tan_x_inequality (x : Real) (h1 : 0 < x) (h2 : x < Real.pi / 2) 
  (h3 : Real.sin x < x) (h4 : x < Real.tan x) : 
  (1 / 2) * (Real.sin x + Real.tan x) > x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_tan_x_inequality_l988_98877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_triangle_l988_98834

open Real

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the theorem
theorem solve_triangle : 
  ∀ (a b c A B C : ℝ),
  triangle_ABC a b c A B C →
  a = 5 →
  B = Real.pi/4 →
  C = 7*Real.pi/12 →
  b = 5 * Real.sqrt 2 ∧ 
  c = (5 * Real.sqrt 6 + 5 * Real.sqrt 2) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_triangle_l988_98834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l988_98821

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
noncomputable def market_value (dividend_rate : ℝ) (yield : ℝ) (face_value : ℝ) : ℝ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 9% stock yielding 8% with a face value of $100 has a market value of $112.50 -/
theorem stock_market_value :
  let dividend_rate : ℝ := 0.09
  let yield : ℝ := 0.08
  let face_value : ℝ := 100
  market_value dividend_rate yield face_value = 112.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l988_98821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l988_98873

/-- A function satisfying the given condition for all positive real numbers -/
noncomputable def f (x : ℝ) : ℝ := 
  3/4 * x + 1/x - 1

/-- The theorem stating the minimum value of f on (0, +∞) -/
theorem min_value_of_f :
  (∀ x > 0, f x ≥ Real.sqrt 3 - 1) ∧
  (∃ x > 0, f x = Real.sqrt 3 - 1) :=
by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l988_98873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_12_l988_98816

theorem sin_2alpha_plus_pi_12 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 4 / 5) : 
  Real.sin (2 * α + π / 12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_pi_12_l988_98816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_15_20_in_terms_of_a_and_b_l988_98818

-- Define the given conditions
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3

-- State the theorem
theorem log_15_20_in_terms_of_a_and_b :
  Real.log 20 / Real.log 15 = (2 + a * b) / (a + a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_15_20_in_terms_of_a_and_b_l988_98818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonintersecting_hexagon_l988_98868

/-- A chord in a circle -/
structure Chord where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Circle with chords -/
structure CircleWithChords where
  radius : ℝ
  chords : List Chord
  total_chord_length : ℝ

/-- Regular hexagon inscribed in a circle -/
structure InscribedHexagon where
  center : ℝ × ℝ
  rotation : ℝ  -- Angle of rotation

/-- Helper function to check if a hexagon intersects a chord -/
def intersects (h : InscribedHexagon) (c : Chord) : Prop :=
  sorry  -- Definition of intersection would go here

/-- Theorem: Existence of non-intersecting inscribed hexagon -/
theorem exists_nonintersecting_hexagon (c : CircleWithChords) :
  c.radius = 1 → c.total_chord_length = 1 →
  ∃ (h : InscribedHexagon), ∀ (chord : Chord), chord ∈ c.chords →
    ¬ (intersects h chord) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonintersecting_hexagon_l988_98868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sum_invested_l988_98867

/-- Represents the simple interest calculation -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem original_sum_invested (x : ℝ) :
  ∀ (P R : ℝ),
  (simpleInterest P (R + 8) 15 - simpleInterest P R 15 = x) →
  (P = x * (5/6)) :=
by
  intros P R h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sum_invested_l988_98867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_percentage_l988_98836

theorem cone_water_volume_percentage :
  ∀ (h r : ℝ), h > 0 → r > 0 →
  let water_height := (5 / 6 : ℝ) * h
  let water_radius := (5 / 6 : ℝ) * r
  let cone_volume := (1 / 3 : ℝ) * π * r^2 * h
  let water_volume := (1 / 3 : ℝ) * π * water_radius^2 * water_height
  (water_volume / cone_volume) * 100 = (125 / 216 : ℝ) * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_percentage_l988_98836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l988_98855

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

theorem triangle_theorem (t : Triangle) 
  (h1 : t.b = (2 * Real.sqrt 3 / 3) * t.a * Real.sin t.B)
  (h2 : 0 < t.A) (h3 : t.A < π/2) :
  (t.a = 3 ∧ t.b = Real.sqrt 6 → t.B = π/4) ∧
  (triangleArea t = Real.sqrt 3 / 2 ∧ t.b + t.c = 3 ∧ t.b > t.c → t.b = 2 ∧ t.c = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l988_98855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l988_98885

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2 * Real.exp x

-- State the theorem
theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), ∀ x y : ℝ, 
    y = m * x + b ∧ 
    y = f 0 + (deriv f 0) * x → 
    m = -1 ∧ b = -2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l988_98885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_equals_one_l988_98802

theorem imaginary_sum_equals_one (i : ℂ) (hi : i^2 = -1) : i^8 + i^20 + i^(-34:ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_equals_one_l988_98802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smartphone_battery_life_l988_98870

/-- Represents the battery life of a smartphone --/
structure BatteryLife where
  totalHours : ℚ
  usedHours : ℚ
  unusedHours : ℚ
  unusedRate : ℚ
  usedRate : ℚ

/-- Calculates the remaining battery life --/
def remainingBatteryLife (b : BatteryLife) : ℚ :=
  1 - (b.unusedHours * b.unusedRate + b.usedHours * b.usedRate)

/-- Calculates how long the remaining battery will last when unused --/
def remainingUnusedTime (b : BatteryLife) : ℚ :=
  remainingBatteryLife b / b.unusedRate

/-- Theorem stating the remaining battery life for the given scenario --/
theorem smartphone_battery_life : 
  let b : BatteryLife := {
    totalHours := 12,
    usedHours := 2,
    unusedHours := 10,
    unusedRate := 1 / 36,
    usedRate := 1 / 4
  }
  remainingUnusedTime b = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smartphone_battery_life_l988_98870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_is_79_8_l988_98895

/-- The weight of the new person given the initial conditions -/
def new_person_weight : ℝ :=
  let initial_people : ℕ := 6
  let average_weight_increase : ℝ := 1.8
  let replaced_person_weight : ℝ := 69
  
  -- The weight of the new person
  replaced_person_weight + initial_people * average_weight_increase

/-- Proof that the new person's weight is 79.8 kg -/
theorem new_person_weight_is_79_8 : new_person_weight = 79.8 := by
  -- Unfold the definition of new_person_weight
  unfold new_person_weight
  -- Perform the calculation
  norm_num

-- This will evaluate new_person_weight and print the result
#eval new_person_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_is_79_8_l988_98895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_proof_l988_98874

noncomputable section

/-- The inverse proportion function passing through (1, 3) -/
def f (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The linear function to compare with -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem inverse_proportion_proof (k : ℝ) (m : ℝ) (h1 : k ≠ 0) (h2 : m ≠ 0) :
  (f k 1 = 3 → k = 3) ∧
  (∀ x, 0 < x → x ≤ 1 → f 3 x > g m x ↔ m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_proof_l988_98874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_three_halves_l988_98890

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 2 then 2 * x
  else if x ≥ 2 then x^2 / 2
  else 0  -- We need to define f for all real numbers

-- State the theorem
theorem f_composition_three_halves :
  f (f (3/2)) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_three_halves_l988_98890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_area_l988_98813

/-- Represents the shaded area in the isosceles triangle with inscribed circle problem -/
noncomputable def shaded_area (a b c : ℝ) : ℝ := a * Real.pi - b * Real.sqrt c

/-- Theorem for the isosceles triangle with inscribed circle problem -/
theorem isosceles_triangle_inscribed_circle_area :
  ∃ (a b c : ℝ),
    (10 : ℝ) > 0 ∧  -- side length
    (16 : ℝ) > 0 ∧  -- base length
    (16 : ℝ) < 2 * 10 ∧  -- triangle inequality
    shaded_area a b c > 0 ∧
    shaded_area a b c < (10 * 16) / 2  -- area less than triangle area
    := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_inscribed_circle_area_l988_98813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ED_length_is_20_l988_98871

/-- A plot ABCD with given measurements -/
structure Plot where
  AF : ℝ
  CE : ℝ
  AE : ℝ
  area : ℝ

/-- The length of ED in the plot -/
noncomputable def ED_length (p : Plot) : ℝ := p.AE / 2 - p.CE

/-- Theorem stating that for a plot with given measurements, ED length is 20 -/
theorem ED_length_is_20 (p : Plot) 
  (h1 : p.AF = 30)
  (h2 : p.CE = 40)
  (h3 : p.AE = 120)
  (h4 : p.area = 7200) :
  ED_length p = 20 := by
  sorry

#check ED_length_is_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ED_length_is_20_l988_98871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l988_98891

/-- Expresses the decimal 0.375̄23 as a common fraction -/
theorem decimal_to_fraction : (375/1000) + (23/99)/1000 = 481 / 792 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l988_98891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_nested_expression_l988_98806

theorem evaluate_nested_expression : 
  (4 : ℝ) - 7 * (4 - 7 * (4 - 7)⁻¹)⁻¹ = 19 / 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_nested_expression_l988_98806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_even_and_odd_a_l988_98811

noncomputable def a (n : ℕ) : ℤ := ⌊n * Real.sqrt 2⌋ + ⌊n * Real.sqrt 3⌋

theorem infinitely_many_even_and_odd_a :
  (∀ N : ℕ, ∃ n > N, Even (a n)) ∧
  (∀ N : ℕ, ∃ n > N, Odd (a n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_even_and_odd_a_l988_98811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l988_98854

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = Real.pi / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l988_98854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_existence_l988_98851

-- Define the function f
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x - Real.log (x + m)

-- State the theorem
theorem two_roots_existence (m : ℤ) (hm : m > 1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  x₁ ∈ Set.Icc (Real.exp (-↑m) - ↑m) (Real.exp (2*↑m) - ↑m) ∧
  x₂ ∈ Set.Icc (Real.exp (-↑m) - ↑m) (Real.exp (2*↑m) - ↑m) ∧
  f m x₁ = 0 ∧ f m x₂ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_existence_l988_98851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_ticks_at_6_is_35_l988_98884

/-- A clock that ticks a certain number of times at different hours -/
structure Clock :=
  (ticks_at_6 : ℕ)
  (ticks_at_12 : ℕ)
  (duration_at_12 : ℚ)
  (constant_interval : Bool)

/-- Calculate the time between first and last ticks at 6 o'clock -/
noncomputable def time_between_ticks_at_6 (c : Clock) : ℚ :=
  if c.constant_interval then
    let interval := c.duration_at_12 / (c.ticks_at_12 - 1)
    interval * (c.ticks_at_6 - 1)
  else
    0  -- undefined for non-constant intervals

/-- Theorem stating the time between first and last ticks at 6 o'clock is 35 seconds -/
theorem time_between_ticks_at_6_is_35 (c : Clock) 
  (h1 : c.ticks_at_6 = 6)
  (h2 : c.ticks_at_12 = 12)
  (h3 : c.duration_at_12 = 77)
  (h4 : c.constant_interval = true) :
  time_between_ticks_at_6 c = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_between_ticks_at_6_is_35_l988_98884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l988_98807

-- Define the components of the expression
noncomputable def a : ℝ := -(17.5 / 100 * 4925)
noncomputable def b : ℝ := (2 / 3 * 46 / 100) * 960
noncomputable def c : ℝ := (3 / 5 * 120.75 / 100) * 4500
noncomputable def d : ℝ := 87.625 / 100 * 1203

-- Define the combined result
noncomputable def combined_result : ℝ := a + b + c - d

-- Theorem statement
theorem expression_result : 
  |combined_result - 1638.77| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_result_l988_98807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_one_l988_98861

noncomputable def f (a x : ℝ) : ℝ :=
  (1 + Real.cos (2 * x) + Real.sin (2 * x)) / (Real.sqrt 2 * Real.sin (Real.pi / 2 + x)) + a * Real.sin (x + Real.pi / 4)

theorem max_value_implies_a_eq_one (a : ℝ) :
  (∃ M : ℝ, M = 3 ∧ ∀ x : ℝ, f a x ≤ M) → a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_one_l988_98861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l988_98823

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove that the asymptotes of the hyperbola have a specific form. -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 8*x)
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1)
  (axis_through_focus : ∃ c, c = 2 ∧ c^2 = a^2 + b^2)
  (intersection_length : 2*b^2/a = 6) :
  ∃ f : ℝ → ℝ, (∀ x, f x = Real.sqrt 3 * x ∨ f x = -(Real.sqrt 3 * x)) ∧ 
              (∀ x y, hyperbola x y → (y = f x ∨ y = -f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l988_98823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l988_98866

-- Define the exponential function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem statement
theorem exponential_function_properties (a : ℝ) (h : f a (-2) = 4) :
  (f a 3 = 1/8) ∧ 
  (∀ x : ℝ, f a x + f a (-x) < 5/2 ↔ -1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_properties_l988_98866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_line_l988_98837

/-- Given three points in a plane and three real constants, 
    the set of points satisfying a certain equation forms a line 
    if and only if the sum of the constants is not zero. -/
theorem points_form_line (A B C M : ℝ × ℝ) (l m n : ℝ) :
  (∀ x y : ℝ, M = (x, y) → 
    l * ((x - A.1)^2 + (y - A.2)^2) + 
    m * ((x - B.1)^2 + (y - B.2)^2) + 
    n * ((x - C.1)^2 + (y - C.2)^2) = 0) ↔
  ((∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      l * ((x - A.1)^2 + (y - A.2)^2) + 
      m * ((x - B.1)^2 + (y - B.2)^2) + 
      n * ((x - C.1)^2 + (y - C.2)^2) = 0)) ∧
   l + m + n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_line_l988_98837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l988_98838

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define the centers and radii of the circles
def center_C1 : ℝ × ℝ := (2, 2)
def center_C2 : ℝ × ℝ := (2, 5)
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem: The circles are internally tangent
theorem circles_internally_tangent :
  distance_between_centers = radius_C2 - radius_C1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l988_98838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_and_divisible_pairs_l988_98878

theorem coprime_and_divisible_pairs (n : ℕ) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2 * n + 1) → S.card = n + 1 →
  (∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ Nat.Coprime a b) ∧
  (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ c ≠ d ∧ c ∣ d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_and_divisible_pairs_l988_98878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_20_approximation_l988_98803

-- Define the approximations given in the problem
noncomputable def log10_2_approx : ℝ := 0.300
noncomputable def log10_5_approx : ℝ := 0.699

-- Define the target approximation
def target_approx : ℚ := 13 / 6

-- State the theorem
theorem log4_20_approximation :
  let log10_20 := log10_2_approx + 1
  let log10_4 := 2 * log10_2_approx
  ∃ ε > 0, |log10_20 / log10_4 - target_approx| < ε :=
by
  sorry

-- Note: We use ∃ ε > 0, |...| < ε to represent "approximately equal to"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_20_approximation_l988_98803
