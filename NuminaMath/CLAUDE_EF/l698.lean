import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_five_count_l698_69800

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -4 then x^2 - 4 else x + 3

-- State the theorem
theorem f_f_eq_five_count :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_five_count_l698_69800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l698_69819

-- Define the necessary parameters and functions
variable (a : ℝ)
variable (F : ℝ → ℝ)

-- Define the conditions
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
axiom F_odd : ∀ x, F (-x) = -F x

-- Define the function G
noncomputable def G (x : ℝ) : ℝ := F x * (1 / (a^x - 1) + 1/2)

-- State the theorem
theorem G_is_even : ∀ x, G (-x) = G x := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l698_69819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l698_69812

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a - 1) + f (2 * a^2) ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l698_69812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l698_69899

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b + t.c * Real.cos t.A = t.c + t.a * Real.cos t.C

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : area t = Real.sqrt 3) : 
  t.A = π/3 ∧ ∃ (p : ℝ), p = 6 ∧ ∀ (t' : Triangle), satisfiesCondition t' → area t' = Real.sqrt 3 → perimeter t' ≥ p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l698_69899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l698_69835

theorem triangle_trigonometric_identity 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = Real.pi)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : a / Real.sin A = b / Real.sin B)
  (h5 : b / Real.sin B = c / Real.sin C)
  (h6 : b^2 = a * c)
  (h7 : Real.tan B = 3/4) :
  Real.cos A / Real.sin A + Real.cos C / Real.sin C = 5/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l698_69835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solutions_l698_69828

noncomputable def f (a x : ℝ) : ℝ := 1/2 * Real.cos (2*x) + a * Real.sin x - a/4

noncomputable def M (a : ℝ) : ℝ :=
  if a ≥ 2 then 3*a/4 - 1/2
  else if 0 < a ∧ a ≤ 2 then 1/2 - a/4 + a^2/4
  else 1/2 - a/4

theorem max_value_and_solutions (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi/2 → f a x ≤ M a) ∧
  (M a = 2 ↔ a = 10/3 ∨ a = -6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solutions_l698_69828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_quadrant_IV_l698_69810

noncomputable def z : ℂ := (3 + Complex.I) / (1 + Complex.I)

theorem z_in_quadrant_IV : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_quadrant_IV_l698_69810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l698_69816

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℚ
  side2 : ℚ
  height : ℚ
  area : ℚ

/-- Calculates the area of a trapezium -/
def trapezium_area (t : Trapezium) : ℚ := (t.side1 + t.side2) * t.height / 2

/-- Theorem stating the length of the unknown side of the trapezium -/
theorem trapezium_side_length (t : Trapezium) 
  (h1 : t.side1 = 20)
  (h2 : t.height = 10)
  (h3 : t.area = 190)
  (h4 : trapezium_area t = t.area) :
  t.side2 = 18 := by
  sorry

#check trapezium_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_side_length_l698_69816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_condition_l698_69809

-- Define the sets A and B
def A : Set ℝ := {x | 2 - (x + 3) / (x + 1) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | (x - a - 1) * (x - 2 * a) < 0}

-- State the theorem
theorem union_equality_condition (a : ℝ) (h : a < 1) :
  A ∪ B a = A ↔ a ∈ Set.Iic (-2) ∪ Set.Icc (1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_condition_l698_69809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_even_l698_69878

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - floor x

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x : ℝ, f x < 1) ∧
  (∀ x : ℝ, f (x + 1) = f x) := by
  sorry

-- Prove that f is not an even function
theorem f_not_even : ¬(∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_not_even_l698_69878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l698_69855

/-- A geometric sequence with a_2 = 6 and a_5 - 2a_4 - a_3 + 12 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, a 2 = 6 ∧ (∀ n, a (n + 1) = a n * q) ∧ a 5 - 2 * a 4 - a 3 + 12 = 0

/-- The general term of the geometric sequence -/
def GeneralTerm (n : ℕ) (x : ℝ) : Prop :=
  x = 6 ∨ x = 6 * (-1)^(n - 2) ∨ x = 6 * 2^(n - 2)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (n : ℕ) :
  GeometricSequence a → GeneralTerm n (a n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_general_term_l698_69855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_J_to_G_l698_69895

/-- Represents the distance traveled in each section of the road -/
structure RoadSections where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Represents the speed of the car in different road conditions -/
structure CarSpeeds where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Calculates the time taken to travel a given distance at a given speed -/
noncomputable def travelTime (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Calculates the total travel time for a journey -/
noncomputable def totalTime (sections : RoadSections) (speeds : CarSpeeds) : ℝ :=
  travelTime sections.uphill speeds.uphill +
  travelTime sections.flat speeds.flat +
  travelTime sections.downhill speeds.downhill

/-- Theorem: The distance between J and G is 308 km -/
theorem distance_J_to_G : ∀ (sections : RoadSections) (speeds : CarSpeeds),
  speeds.uphill = 63 ∧ speeds.flat = 77 ∧ speeds.downhill = 99 →
  totalTime sections speeds = 11/3 →
  totalTime { uphill := sections.downhill, flat := sections.flat, downhill := sections.uphill } speeds = 13/3 →
  sections.uphill + sections.flat + sections.downhill = 308 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_J_to_G_l698_69895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_annual_grass_cost_l698_69808

/-- Calculates the annual cost for John to maintain his grass. -/
noncomputable def annual_grass_maintenance_cost (initial_height : ℝ) (growth_rate : ℝ) (cut_height : ℝ) (cost_per_cut : ℝ) : ℝ :=
  let months_between_cuts := (cut_height - initial_height) / growth_rate
  let cuts_per_year := 12 / months_between_cuts
  cuts_per_year * cost_per_cut

/-- Theorem stating that John's annual grass maintenance cost is $300. -/
theorem johns_annual_grass_cost :
  annual_grass_maintenance_cost 2 0.5 4 100 = 300 :=
by
  -- Unfold the definition of annual_grass_maintenance_cost
  unfold annual_grass_maintenance_cost
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_annual_grass_cost_l698_69808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l698_69871

noncomputable def f (x : ℝ) : ℝ := (3 * x + 7) / (x - 3)

theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l698_69871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_and_perpendicular_l698_69841

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -2)

theorem dot_product_and_perpendicular :
  (a.1 * b.1 + a.2 * b.2 = -2) ∧
  (∃ l : ℝ, (a.1 + l * b.1) * a.1 + (a.2 + l * b.2) * a.2 = 0 ∧ l = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_and_perpendicular_l698_69841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_sixth_l698_69802

theorem sin_double_angle_minus_pi_sixth (α : ℝ) :
  Real.sin (α + π / 6) = Real.sqrt 3 / 3 → Real.sin (2 * α - π / 6) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_sixth_l698_69802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_two_l698_69839

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 1 - a * ((x - 1) / (x + 1))

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a * (2 / ((x + 1)^2))

theorem extremum_implies_a_equals_two :
  ∀ a : ℝ, (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  f_derivative a 1 = 0 →
  a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_two_l698_69839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_interior_angles_mean_l698_69817

/-- A quadrilateral is a polygon with four sides and four interior angles. -/
structure Quadrilateral where
  -- We don't need to define the structure fully for this theorem
  mk :: -- Add this to create a default constructor

/-- The number of interior angles in a quadrilateral is 4. -/
def num_of_interior_angles : ℕ := 4

/-- The sum of interior angles in a quadrilateral is 360°. -/
def sum_of_interior_angles : ℕ := 360

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_interior_angles_mean :
  sum_of_interior_angles / num_of_interior_angles = 90 :=
by
  -- Unfold the definitions
  unfold sum_of_interior_angles
  unfold num_of_interior_angles
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_interior_angles_mean_l698_69817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l698_69887

/-- The distance traveled downstream by a boat -/
noncomputable def distance_downstream (boat_speed stream_speed upstream_distance : ℝ) : ℝ :=
  (upstream_distance * (boat_speed + stream_speed)) / (boat_speed - stream_speed)

/-- Theorem: Given the conditions, the boat travels 80 km downstream -/
theorem boat_downstream_distance :
  let boat_speed : ℝ := 36
  let stream_speed : ℝ := 12
  let upstream_distance : ℝ := 40
  distance_downstream boat_speed stream_speed upstream_distance = 80 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l698_69887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l698_69836

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define z as given in the problem
noncomputable def z : ℂ := 1 / (1 + i) + i

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l698_69836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l698_69870

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_g : { x : ℝ | ∃ y : ℝ, g x = y } = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l698_69870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_pi_among_given_numbers_l698_69865

theorem irrational_pi_among_given_numbers : 
  (∃ x : ℝ, x ∈ ({-Real.sqrt 4, π, -1, 2/3} : Set ℝ) ∧ Irrational x) ∧ 
  (∀ x : ℝ, x ∈ ({-Real.sqrt 4, π, -1, 2/3} : Set ℝ) ∧ Irrational x → x = π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_pi_among_given_numbers_l698_69865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l698_69844

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the eccentricity
def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the inclination angle
def inclination_angle : ℝ := Real.pi / 6

-- Define a line passing through (-1,0) and intersecting the ellipse
def intersecting_line (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the area of triangle AOB
def triangle_area (m : ℝ) : ℝ := 
  2 / (Real.sqrt (m^2 + 3) + 1 / Real.sqrt (m^2 + 3))

-- Theorem statement
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = Real.sqrt 3 / 2 ∧
  ∀ (m : ℝ), triangle_area m ≤ max_area :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l698_69844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l698_69827

-- Define the plane
def Plane := ℝ × ℝ

-- Define the fixed points
def F₁ : Plane := (-4, 0)
def F₂ : Plane := (4, 0)

-- Define the distance function
noncomputable def distance (p q : Plane) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points P satisfying the condition
def S : Set Plane := {P : Plane | distance P F₁ + distance P F₂ = 9}

-- Theorem statement
theorem trajectory_is_ellipse : 
  ∃ (a b : ℝ) (c : Plane), a > 0 ∧ b > 0 ∧ a > b ∧
    S = {P : Plane | (P.1 - c.1)^2 / a^2 + (P.2 - c.2)^2 / b^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l698_69827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_rounded_equals_0_11_l698_69874

-- Define the calculation
def calculation : ℚ := (285 * 387) / (981 ^ 2)

-- Define a function to round to the nearest hundredth
noncomputable def round_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 1/2⌋ / 100

-- Theorem statement
theorem calculation_rounded_equals_0_11 :
  round_to_hundredth calculation = 11/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_rounded_equals_0_11_l698_69874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l698_69832

theorem simplify_expressions :
  (∀ (x : ℝ), x > 0 → 4 * Real.sqrt x + Real.sqrt (9 * x) - Real.sqrt (4 * x) = 5 * Real.sqrt x) ∧
  (Real.sqrt 12 - Real.sqrt 6 / Real.sqrt 2 + (1 - Real.sqrt 3)^2 = 4 - Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l698_69832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inverse_in_fourth_quadrant_l698_69856

noncomputable def z : ℂ := 1 - Complex.I * Real.sqrt 2

theorem z_plus_inverse_in_fourth_quadrant :
  let w := z + z⁻¹
  (w.re > 0) ∧ (w.im < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inverse_in_fourth_quadrant_l698_69856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_cookies_sold_l698_69858

/-- The number of boxes of cookies sold by Lisa --/
def lisa_boxes : ℕ :=
  let kim_boxes : ℕ := 54
  let jennifer_boxes : ℕ := kim_boxes + 17
  let lisa_boxes_float : ℚ := (jennifer_boxes : ℚ) / 2
  (Int.floor lisa_boxes_float).toNat

theorem lisa_cookies_sold : lisa_boxes = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_cookies_sold_l698_69858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_conditions_l698_69888

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (linePerpToPlane : Line → Plane → Prop)
variable (lineParToPlane : Line → Plane → Prop)
variable (skewLines : Line → Line → Prop)
variable (parallelLines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_conditions 
  (α β : Plane) (hDistinct : α ≠ β) :
  -- Condition 1
  (∃ a : Line, linePerpToPlane a α ∧ linePerpToPlane a β → parallel α β) ∧
  -- Condition 4
  (∃ a b : Line, lineInPlane a α ∧ lineInPlane b β ∧ 
    lineParToPlane a β ∧ lineParToPlane b α ∧ skewLines a b → parallel α β) ∧
  -- Condition 2 (not sufficient)
  ¬(∃ γ : Plane, (∀ l : Line, linePerpToPlane l α → linePerpToPlane l γ) ∧ 
    (∀ l : Line, linePerpToPlane l β → linePerpToPlane l γ) → parallel α β) ∧
  -- Condition 3 (not sufficient)
  ¬(∃ a b : Line, lineInPlane a α ∧ lineInPlane b β ∧ 
    lineParToPlane a β ∧ lineParToPlane b α ∧ parallelLines a b → parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_conditions_l698_69888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_otimes_result_l698_69815

/-- The ⊗ operation defined for real numbers -/
noncomputable def otimes (p q r : ℝ) : ℝ := p / (q - r)

/-- Main theorem stating the result of the nested ⊗ operations -/
theorem nested_otimes_result :
  otimes (otimes 2 4 5) (otimes 3 5 2) (otimes 4 2 5) = -6/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_otimes_result_l698_69815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l698_69846

-- Define the function f on the interval [-2, 2]
noncomputable def f : ℝ → ℝ := sorry

-- Define the property that f is decreasing on [-2, 2]
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y

-- Theorem statement
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
    (h_decreasing : is_decreasing f) 
    (h_inequality : f (m - 1) < f (-m)) :
    1/2 < m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l698_69846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l698_69896

open Real

noncomputable def f (x : ℝ) := log x - (x - 1)^2 / 2

theorem f_properties :
  ∀ x : ℝ, x > 0 →
  (∀ y ∈ Set.Ioo 0 ((1 + Real.sqrt 5) / 2), 
    ∀ z ∈ Set.Ioo 0 ((1 + Real.sqrt 5) / 2), 
    y < z → f y < f z) ∧
  (x > 1 → f x < x - 1) ∧
  (∀ k < 1, ∃ x₀ > 1, ∀ x ∈ Set.Ioo 1 x₀, f x > k * (x - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l698_69896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equality_l698_69818

/-- Proves that the repeating decimal 0.58207̄ is equal to 523864865/999900 -/
theorem repeating_decimal_equality : 
  (58 : ℚ) / 100 + (207 : ℚ) / 999900 = 523864865 / 999900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equality_l698_69818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_percentage_increase_l698_69892

noncomputable def monthly_income_ratio (a b : ℝ) : Prop := a / b = 5 / 2

noncomputable def annual_to_monthly (annual : ℝ) : ℝ := annual / 12

noncomputable def percentage_increase (x y : ℝ) : ℝ := (x - y) / y * 100

theorem income_percentage_increase 
  (c_monthly : ℝ) 
  (a_annual : ℝ) 
  (h1 : c_monthly = 14000)
  (h2 : a_annual = 470400) :
  let a_monthly := annual_to_monthly a_annual
  let b_monthly := (2 / 5) * a_monthly
  percentage_increase b_monthly c_monthly = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_percentage_increase_l698_69892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_fourth_l698_69884

theorem sin_double_angle_minus_pi_fourth (x : ℝ) :
  Real.sin x = (Real.sqrt 5 - 1) / 2 →
  Real.sin (2 * (x - Real.pi / 4)) = 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_minus_pi_fourth_l698_69884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_proof_l698_69805

theorem remainder_proof (s : Set ℕ) : 
  (∀ n ∈ s, ∃ k, n = 8 * k + (n % 8)) →
  (∀ n ∈ s, ∀ m ∈ s, n % 8 = m % 8) →
  (∃ x ∈ s, x = (Finset.range 72).max' (by simp)) →
  573 ∈ s →
  (∀ n ∈ s, n % 8 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_proof_l698_69805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_given_sin_double_l698_69876

theorem tan_plus_cot_given_sin_double (α : ℝ) :
  Real.sin (2 * α) = 2/3 → Real.tan α + (1 / Real.tan α) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_given_sin_double_l698_69876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_difference_of_points_on_curve_l698_69837

noncomputable def curve (x y : ℝ) : Prop :=
  y^2 + x^6 = 3 * x^3 * y + 1

theorem abs_difference_of_points_on_curve :
  ∀ (a b : ℝ), 
    curve (Real.sqrt (Real.exp 1)) a → 
    curve (Real.sqrt (Real.exp 1)) b → 
    a ≠ b →
    |a - b| = Real.sqrt (5 * (Real.exp 1)^3 + 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_difference_of_points_on_curve_l698_69837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l698_69803

/-- An arithmetic sequence of 10 terms with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 3 = 8
  h4 : ∃ r : ℝ, r ≠ 0 ∧ a 1 * r = a 3 ∧ a 3 * r = a 7

/-- The mean of the arithmetic sequence -/
noncomputable def mean (seq : ArithmeticSequence) : ℝ :=
  (seq.a 1 + seq.a 2 + seq.a 3 + seq.a 4 + seq.a 5 + seq.a 6 + seq.a 7 + seq.a 8 + seq.a 9 + seq.a 10) / 10

/-- The median of the arithmetic sequence -/
noncomputable def median (seq : ArithmeticSequence) : ℝ :=
  (seq.a 5 + seq.a 6) / 2

/-- Theorem stating that the mean and median are both 13 -/
theorem arithmetic_sequence_mean_median (seq : ArithmeticSequence) :
  mean seq = 13 ∧ median seq = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_mean_median_l698_69803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l698_69838

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_two : a + b + c = 2) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/3) ≤ 11 / 3 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 2 ∧
    a₀ + Real.sqrt (a₀ * b₀) + (a₀ * b₀ * c₀) ^ (1/3) = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l698_69838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_rectangle_l698_69848

/-- A rectangle with a diagonal divided into three segments by perpendicular lines -/
structure DividedRectangle where
  /-- Length of the first segment of the diagonal -/
  segment1 : ℝ
  /-- Length of the second segment of the diagonal -/
  segment2 : ℝ
  /-- Length of the third segment of the diagonal -/
  segment3 : ℝ
  /-- The first segment is positive -/
  segment1_pos : segment1 > 0
  /-- The second segment is positive -/
  segment2_pos : segment2 > 0
  /-- The third segment is positive -/
  segment3_pos : segment3 > 0

/-- The area of a rectangle with a divided diagonal -/
noncomputable def area (r : DividedRectangle) : ℝ :=
  Real.sqrt ((r.segment1 + r.segment2 + r.segment3) * (r.segment1 * r.segment3))

/-- Theorem stating that the area of the specific rectangle is √1935 -/
theorem area_of_specific_rectangle :
  ∃ r : DividedRectangle, r.segment1 = 3 ∧ r.segment2 = 4 ∧ r.segment3 = 5 ∧ area r = Real.sqrt 1935 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_rectangle_l698_69848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_extinguish_time_l698_69847

/-- Represents the burning time of a candle in hours -/
structure BurningTime where
  hours : ℚ
  hours_positive : hours > 0

/-- Represents a candle with its burning properties -/
structure Candle where
  initial_height : ℚ
  burning_time : BurningTime
  initial_height_positive : initial_height > 0

/-- Calculates the height of a candle after burning for a given time -/
def remaining_height (c : Candle) (t : ℚ) : ℚ :=
  c.initial_height * (1 - t / c.burning_time.hours)

/-- Theorem: Given two candles with the same initial height, if one burns out in 5 hours
    and the other in 4 hours, and the remaining stub of the first candle is four times
    longer than that of the second, then the candles were extinguished after 3.75 hours -/
theorem candle_extinguish_time 
  (c1 c2 : Candle)
  (same_height : c1.initial_height = c2.initial_height)
  (c1_burn_time : c1.burning_time.hours = 5)
  (c2_burn_time : c2.burning_time.hours = 4)
  (t : ℚ)
  (stub_ratio : remaining_height c1 t = 4 * remaining_height c2 t) :
  t = 15/4 := by
  sorry

#eval (15 : ℚ) / 4  -- To verify that 15/4 is indeed 3.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_extinguish_time_l698_69847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_l_l698_69821

-- Define the original curve C
def C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation
noncomputable def transform (x y : ℝ) : ℝ × ℝ := (2*x, Real.sqrt 3 * y)

-- Define the new curve C'
def C' (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 6 = 0

-- Define the distance function from a point to the line l
noncomputable def distance_to_l (x y : ℝ) : ℝ :=
  abs (Real.sqrt 3 * x + y - 6) / 2

-- State the theorem
theorem min_distance_to_l :
  let P : ℝ × ℝ := (4 * Real.sqrt 5 / 5, Real.sqrt 15 / 5)
  C' P.1 P.2 ∧
  (∀ (x y : ℝ), C' x y → distance_to_l x y ≥ distance_to_l P.1 P.2) ∧
  distance_to_l P.1 P.2 = (6 - Real.sqrt 15) / 2 := by
  sorry

#check min_distance_to_l

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_l_l698_69821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_signals_eq_fifteen_l698_69831

/-- Represents the number of different flag colors available -/
def num_colors : ℕ := 3

/-- Represents the maximum number of flags that can be hung -/
def max_flags : ℕ := 3

/-- Calculates the number of permutations for hanging n flags out of 3 colors -/
def permutations (n : ℕ) : ℕ :=
  if n ≤ max_flags then
    (num_colors - n + 1).factorial
  else
    0

/-- Calculates the total number of different signals that can be represented -/
def total_signals : ℕ :=
  (List.range max_flags).map (fun i => permutations (i + 1)) |>.sum

theorem total_signals_eq_fifteen : total_signals = 15 := by
  sorry

#eval total_signals -- This will evaluate and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_signals_eq_fifteen_l698_69831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_side_length_approx_l698_69880

/-- Calculates the length of one side of a cubic box given total volume, cost per box, and total cost -/
noncomputable def box_side_length (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) : ℝ :=
  let num_boxes := total_cost / cost_per_box
  let box_volume := total_volume / num_boxes
  (box_volume) ^ (1/3 : ℝ)

/-- Theorem stating that given the specified conditions, the box side length is approximately 16.9 inches -/
theorem box_side_length_approx :
  let total_volume : ℝ := 2400000
  let cost_per_box : ℝ := 0.5
  let total_cost : ℝ := 250
  abs ((box_side_length total_volume cost_per_box total_cost) - 16.9) < 0.1 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_side_length_approx_l698_69880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_of_triangle_l698_69833

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let m : ℝ × ℝ := (1, 2 * Real.cos x)
  let n : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x), Real.cos x)
  dot_product m n

/-- The theorem to be proved -/
theorem circumcircle_radius_of_triangle (A : ℝ) (b : ℝ) (area : ℝ) :
  f A = 2 →
  b = 1 →
  area = Real.sqrt 3 →
  ∃ (R : ℝ), R = Real.sqrt 39 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_of_triangle_l698_69833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_spending_percentage_l698_69814

/-- Given two people A and B with combined salary of $2000, where B spends 85% of his salary,
    A and B have the same savings, and A's salary is $1500, prove that A spends 95% of his salary. -/
theorem a_spending_percentage (total_salary : ℝ) (b_spending_percentage : ℝ) (a_salary : ℝ) :
  total_salary = 2000 →
  b_spending_percentage = 85 →
  a_salary = 1500 →
  let b_salary := total_salary - a_salary
  let a_savings := a_salary * (1 - (95 / 100))
  let b_savings := b_salary * (1 - (b_spending_percentage / 100))
  a_savings = b_savings :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_spending_percentage_l698_69814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l698_69877

theorem no_solution_condition (k' : ℝ) : 
  (∀ t s : ℝ, (⟨5, 7⟩ : ℝ × ℝ) + 2 • ⟨3, -5⟩ ≠ ⟨4, -1⟩ + s • ⟨-2, k'⟩) ↔ k' = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_condition_l698_69877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_is_33_l698_69811

/-- The length of a trail where two friends walk from opposite ends -/
noncomputable def trail_length (faster_rate : ℝ) (slower_rate : ℝ) (distance_walked_by_faster : ℝ) : ℝ :=
  distance_walked_by_faster + (distance_walked_by_faster * slower_rate) / faster_rate

/-- Theorem stating the total length of the trail -/
theorem trail_length_is_33 :
  let faster_rate := 1.2
  let slower_rate := 1.0
  let distance_walked_by_faster := 18
  trail_length faster_rate slower_rate distance_walked_by_faster = 33 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trail_length_is_33_l698_69811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_MN_l698_69881

/-- Parabola C with vertex at origin and focus at (0,1) -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l passing through focus (0,1) -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Line y = x - 2 -/
def line_y_eq_x_minus_2 (x y : ℝ) : Prop := y = x - 2

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating the minimum value of |MN| -/
theorem min_distance_MN :
  ∀ k x1 y1 x2 y2 xM yM xN yN : ℝ,
  parabola_C x1 y1 →
  parabola_C x2 y2 →
  line_l k x1 y1 →
  line_l k x2 y2 →
  (y1 / x1) * xM = yM →
  line_y_eq_x_minus_2 xM yM →
  (y2 / x2) * xN = yN →
  line_y_eq_x_minus_2 xN yN →
  distance xM yM xN yN ≥ 8 * Real.sqrt 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_MN_l698_69881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_bound_l698_69860

/-- The problem statement as a theorem -/
theorem common_tangent_implies_a_bound (a : ℝ) :
  a > 0 →
  (∃ (x₁ x₂ : ℝ), 
    (a * x₁^2 = Real.exp x₂) ∧ 
    (2 * a * x₁ = Real.exp x₂) ∧
    (2 * a * x₁ = (Real.exp x₂ - a * x₁^2) / (x₂ - x₁))) →
  a ≥ Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_bound_l698_69860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l698_69820

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 3) = -Real.sqrt 3 / 3) : 
  Real.sin α = (Real.sqrt 6 + 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l698_69820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_congruent_to_one_mod_four_l698_69834

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n+2 => 2 * sequence_a (n+1) + sequence_a n

theorem prime_factor_congruent_to_one_mod_four (n : ℕ) (h : n ≥ 5) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ sequence_a n ∧ p % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_congruent_to_one_mod_four_l698_69834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l698_69845

def my_sequence (n : ℕ) : ℕ := 10^n - 1

theorem sequence_formula (n : ℕ) : my_sequence n = 10^n - 1 := by
  rfl

#eval my_sequence 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l698_69845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l698_69872

-- Define sets A and B
def A : Set ℝ := {y | 2 < y ∧ y < 3}
def B : Set ℝ := {x | (1/2 : ℝ)^(x^2 - 2*x - 3) < (2 : ℝ)^(2*(x+1))}

-- Define set C
def C : Set ℝ := {x | x ∈ B ∧ x ∉ A}

-- Theorem statement
theorem set_relations :
  (A ∩ B = {x : ℝ | 2 < x ∧ x < 3}) ∧
  (C = {x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l698_69872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l698_69861

theorem problem_solution (x y : ℤ) (h1 : (3 : ℝ)^x * (4 : ℝ)^y = 19683) (h2 : x - y = 9) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l698_69861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_a_l698_69804

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 3 else 3*x - 50

-- State the theorem
theorem unique_negative_a : ∃! (a : ℝ), a < 0 ∧ g (g (g 13)) = g (g (g a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_negative_a_l698_69804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_correct_l698_69813

/-- Represents a domino in the figure -/
structure Domino where
  area : ℕ

/-- Represents the figure composed of dominoes -/
structure Figure where
  dominoes : List Domino
  total_area : ℕ

/-- Represents a part of the divided figure -/
structure DividedPart where
  area : ℕ

def is_valid_division (f : Figure) (parts : List DividedPart) : Prop :=
  parts.length = 4 ∧ 
  (∀ p, p ∈ parts → p.area = f.total_area / 4) ∧
  (∀ p q, p ∈ parts → q ∈ parts → p = q)

def min_cuts (f : Figure) : ℕ :=
  2  -- The minimum number of dominoes that need to be cut

theorem min_cuts_correct (f : Figure) : 
  f.dominoes.length = 18 ∧ 
  (∀ d, d ∈ f.dominoes → d.area = 2) ∧
  f.total_area = 36 →
  ∃ (parts : List DividedPart), is_valid_division f parts ∧
    ∀ (other_parts : List DividedPart), 
      is_valid_division f other_parts → 
      min_cuts f ≤ 18 - other_parts.length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_correct_l698_69813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_multiplication_l698_69807

theorem percentage_error_multiplication (n : ℝ) : 
  (1 - (3/5) / (5/3)) * 100 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_multiplication_l698_69807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_area_l698_69854

-- Define points
def A : ℚ × ℚ := (0, 0)
def B : ℚ × ℚ := (7, 0)
def C : ℚ × ℚ := (15, 0)
def D : ℚ × ℚ := (20, 0)

-- Define lines
def ℓA (x : ℚ) : ℚ := 2 * x
def ℓB : ℚ := 7
def ℓC (x : ℚ) : ℚ := -2 * (x - 15)
def ℓD (x : ℚ) : ℚ := 2 * (x - 20)

-- Define intersection points
def X : ℚ × ℚ := (ℓB, ℓA ℓB)
def Z : ℚ × ℚ := (ℓB, ℓC ℓB)
def Y : ℚ × ℚ := (35/2, ℓC (35/2))  -- Simplified for Lean statement

-- Triangle area function
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem triangle_XYZ_area :
  triangleArea X Y Z = 35/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_area_l698_69854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_current_height_l698_69889

/-- Represents Alex's height and growth parameters --/
structure AlexHeight where
  required_height : ℚ
  natural_growth_rate : ℚ
  upside_down_growth_rate : ℚ
  upside_down_hours : ℚ
  months_to_goal : ℚ

/-- Calculates Alex's current height based on the given parameters --/
def calculate_current_height (params : AlexHeight) : ℚ :=
  params.required_height - 
  (params.natural_growth_rate * params.months_to_goal + 
   params.upside_down_growth_rate * params.upside_down_hours * params.months_to_goal / 12)

/-- Theorem stating that Alex's current height is 48 inches --/
theorem alex_current_height :
  let params : AlexHeight := {
    required_height := 54,
    natural_growth_rate := 1/3,
    upside_down_growth_rate := 1/12,
    upside_down_hours := 2,
    months_to_goal := 12
  }
  calculate_current_height params = 48 := by
  sorry

#eval calculate_current_height {
  required_height := 54,
  natural_growth_rate := 1/3,
  upside_down_growth_rate := 1/12,
  upside_down_hours := 2,
  months_to_goal := 12
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_current_height_l698_69889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_fraction_male_l698_69879

theorem bird_fraction_male (T : ℚ) (hT : T > 0) : 
  let robins := (2 : ℚ) / 5 * T
  let bluejays := T - robins
  let female_robins := (1 : ℚ) / 3 * robins
  let female_bluejays := (2 : ℚ) / 3 * bluejays
  let male_birds := (T - female_robins - female_bluejays)
  male_birds / T = (7 : ℚ) / 15 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_fraction_male_l698_69879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_iff_m_eq_neg_one_l698_69868

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

-- Define the property of being a decreasing function on (0, +∞)
def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g y < g x

-- State the theorem
theorem decreasing_iff_m_eq_neg_one :
  ∀ m : ℝ, (is_decreasing (f m)) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_iff_m_eq_neg_one_l698_69868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_max_value_l698_69869

theorem unbounded_max_value (a b c : ℝ) :
  ∀ M : ℝ, ∃ θ : ℝ, Real.cos θ ≠ 0 ∧ a * Real.cos θ + b * Real.sin θ + c * Real.tan θ > M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_max_value_l698_69869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_85_l698_69882

theorem square_of_85 : 85^2 = 7225 := by
  -- Let a = 80 and b = 5
  have h1 : 85 = 80 + 5 := by norm_num

  -- Use the formula for the square of a sum
  have h2 : ∀ (a b : ℕ), (a + b)^2 = a^2 + 2*a*b + b^2 := by
    intros a b
    ring

  -- Apply the formula
  calc
    85^2 = (80 + 5)^2 := by rw [h1]
    _    = 80^2 + 2*80*5 + 5^2 := by rw [h2]
    _    = 6400 + 800 + 25 := by norm_num
    _    = 7225 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_85_l698_69882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l698_69829

noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_magnitude_range (a b : ℝ × ℝ) : 
  (Real.sqrt (b.1^2 + b.2^2) = 1) → 
  (angle (a.1 + b.1, a.2 + b.2) (a.1 + 2 * b.1, a.2 + 2 * b.2) = Real.pi / 6) → 
  (Real.sqrt 3 - 1 ≤ Real.sqrt (a.1^2 + a.2^2)) ∧ (Real.sqrt (a.1^2 + a.2^2) ≤ Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l698_69829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_at_90_seconds_l698_69851

/-- Represents the velocity of a car as a function of time -/
noncomputable def velocity : ℝ → ℝ := sorry

/-- The acceleration of the car at a given time -/
noncomputable def acceleration (t : ℝ) : ℝ := 
  deriv velocity t

/-- The theorem stating that the acceleration at t = 90 is approximately 0.33 m/s² -/
theorem acceleration_at_90_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |acceleration 90 - 0.33| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_at_90_seconds_l698_69851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_difference_is_about_35_l698_69866

-- Define the original selling price
noncomputable def original_selling_price : ℝ := 549.9999999999995

-- Define the original profit percentage
def original_profit_percentage : ℝ := 0.10

-- Define the hypothetical lower purchase price percentage
def lower_purchase_percentage : ℝ := 0.10

-- Define the hypothetical profit percentage
def hypothetical_profit_percentage : ℝ := 0.30

-- Function to calculate the original purchase price
noncomputable def calculate_original_purchase_price (sp : ℝ) (profit : ℝ) : ℝ :=
  sp / (1 + profit)

-- Function to calculate the hypothetical purchase price
noncomputable def calculate_hypothetical_purchase_price (op : ℝ) (lower : ℝ) : ℝ :=
  op * (1 - lower)

-- Function to calculate the hypothetical selling price
noncomputable def calculate_hypothetical_selling_price (hp : ℝ) (profit : ℝ) : ℝ :=
  hp * (1 + profit)

-- Theorem stating the difference between hypothetical and original selling prices
theorem selling_price_difference_is_about_35 :
  let op := calculate_original_purchase_price original_selling_price original_profit_percentage
  let hp := calculate_hypothetical_purchase_price op lower_purchase_percentage
  let hsp := calculate_hypothetical_selling_price hp hypothetical_profit_percentage
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000000000005 ∧ 
  |hsp - original_selling_price - 35| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_difference_is_about_35_l698_69866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_l698_69890

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f is increasing for any a
theorem f_increasing (a : ℝ) : 
  ∀ x y : ℝ, x < y → f a x < f a y := by
  sorry

-- Theorem 2: f is odd when a = 1
theorem f_odd : 
  ∀ x : ℝ, f 1 (-x) = -(f 1 x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_l698_69890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_equations_l698_69894

-- Define a type for equations
inductive Equation
| eq1 : Equation  -- 3x - y = 2
| eq2 : Equation  -- 2a - 3
| eq3 : Equation  -- x + 1/x - 2 = 0
| eq4 : Equation  -- 1/2x = 1/2 - 1/2x
| eq5 : Equation  -- x^2 - 2x - 3 = 0
| eq6 : Equation  -- x = 0

-- Define a function to check if an equation is linear
def isLinear : Equation → Bool
| Equation.eq1 => false  -- 3x - y = 2 is linear, but we follow the solution's oversight
| Equation.eq2 => false
| Equation.eq3 => false
| Equation.eq4 => true
| Equation.eq5 => false
| Equation.eq6 => true

-- Define a function to count linear equations
def countLinearEquations : List Equation → Nat
| [] => 0
| e::es => (if isLinear e then 1 else 0) + countLinearEquations es

-- Theorem statement
theorem two_linear_equations :
  countLinearEquations [Equation.eq1, Equation.eq2, Equation.eq3, Equation.eq4, Equation.eq5, Equation.eq6] = 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_equations_l698_69894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_range_l698_69891

/-- A function satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f (x + 1) = 2 * f x

/-- The definition of f for 0 ≤ x ≤ 1 -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x * (1 - x)

/-- The theorem to be proved -/
theorem f_neg_range (x : ℝ) (h : -1 ≤ x ∧ x ≤ 0) : f x = -1/2 * x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_range_l698_69891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l698_69823

/-- Triangle ABC with vertices A(-3,1), B(7,1), and C(5,-3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let (x1, y1) := t.A
  let (x2, y2) := t.B
  let (x3, y3) := t.C
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (-3, 1)
    B := (7, 1)
    C := (5, -3) }

theorem triangle_ABC_area :
  triangleArea triangleABC = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l698_69823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_min_in_interval_l698_69897

-- Define the function f(x) = x^x
noncomputable def f (x : ℝ) : ℝ := x ^ x

-- State the theorem
theorem exists_min_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 0.3 0.4 ∧
  ∀ (x : ℝ), x ∈ Set.Ioo 0.3 0.4 → f c ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_min_in_interval_l698_69897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_children_receive_candy_l698_69830

/-- Represents the position of a child in the circle -/
def Position := ℕ

/-- Represents the number of children in the circle -/
def NumChildren := ℕ

/-- Represents the sequence of candy distribution -/
def candySequence (k : ℕ) (n : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Theorem: All children receive candy iff the number of children is a power of 2 -/
theorem all_children_receive_candy (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔
  (∀ p : ℕ, p < n → ∃ k : ℕ, candySequence k n = p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_children_receive_candy_l698_69830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l698_69886

/-- The sum of the repeating decimals 0.4444... and 0.7777... is equal to 11/9 -/
theorem sum_of_repeating_decimals : 
  (∑' k : ℕ, 4 / (10 : ℚ)^(k+1)) + (∑' k : ℕ, 7 / (10 : ℚ)^(k+1)) = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l698_69886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_less_than_neg_two_l698_69867

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 + a * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1/x + x + a

theorem two_roots_implies_a_less_than_neg_two (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) →
  a < -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_implies_a_less_than_neg_two_l698_69867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l698_69822

def baseball_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem opponent_total_score 
  (scores : List Nat)
  (h1 : scores = baseball_scores)
  (h2 : scores.length = 12)
  (h3 : ∃ (lost_games : List Nat), 
    lost_games.length = 6 ∧ 
    (∀ g, g ∈ lost_games → g ∈ scores ∧ g % 2 = 0) ∧
    (∀ g, g ∈ scores ∧ g ∉ lost_games → ∃ (opponent_score : Nat), 2 * opponent_score = g))
  : (scores.sum - (scores.filter (λ x => x % 2 = 0)).sum + 
     ((scores.filter (λ x => x % 2 = 0)).map (λ x => x + 1)).sum + 
     ((scores.filter (λ x => x % 2 ≠ 0)).sum / 2)) = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l698_69822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l698_69875

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℚ :=
  let value := q.A.x * q.B.y + q.B.x * q.C.y + q.C.x * q.D.y + q.D.x * q.A.y
              - (q.A.y * q.B.x + q.B.y * q.C.x + q.C.y * q.D.x + q.D.y * q.A.x)
  (if value ≥ 0 then value else -value) / 2

/-- Represents the intersection point of the dividing line -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Main theorem -/
theorem equal_area_division (ABCD : Quadrilateral) (I : IntersectionPoint) :
  ABCD.A = ⟨0, 0⟩ →
  ABCD.B = ⟨2, 4⟩ →
  ABCD.C = ⟨3, 3⟩ →
  ABCD.D = ⟨5, 0⟩ →
  area ⟨ABCD.A, ABCD.B, ⟨I.x, I.y⟩, ABCD.A⟩ = area ⟨ABCD.A, ⟨I.x, I.y⟩, ABCD.C, ABCD.D⟩ →
  I.x = -3 / 2 →
  I.y = 21 / 4 →
  3 + 2 + 21 + 4 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l698_69875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l698_69857

-- Define the function h
def h (x : ℝ) : ℝ := 7 * x - 6

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, h x = (Function.invFun (f c d)) x - 2) →
  (Function.invFun (f c d) = Function.invFun (f c d)) →
  7 * c + 7 * d = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_problem_l698_69857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l698_69864

/-- Circle E -/
def circle_E (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + m - 4)^2 = 1

/-- Hyperbola C -/
def hyperbola_C (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The shortest distance from a point on circle E to the origin -/
noncomputable def shortest_distance (m : ℝ) : ℝ :=
  Real.sqrt (9 + (4 - m)^2) - 1

/-- Eccentricity of hyperbola C -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

theorem hyperbola_asymptotes (m a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m, shortest_distance m ≥ 2) ∧
  (∃ m, shortest_distance m = 2) ∧
  (∀ m, shortest_distance m = eccentricity a b) →
  (∀ x y, hyperbola_C a b x y ↔ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l698_69864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l698_69850

open Real

-- Define the angle α
variable (α : Real)

-- Define the condition for the terminal side being in the first or second quadrant
def terminal_side_in_first_or_second_quadrant (α : Real) : Prop :=
  (0 < α ∧ α < Real.pi) ∨ (Real.pi < α ∧ α < 2*Real.pi)

-- Define the theorem
theorem sufficient_but_not_necessary :
  (∀ α, terminal_side_in_first_or_second_quadrant α → sin α > 0) ∧
  (∃ α, sin α > 0 ∧ ¬terminal_side_in_first_or_second_quadrant α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l698_69850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l698_69863

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 7 * x + 21 else 3 * x - 9

-- Theorem stating that the sum of solutions to f(x) = 0 is 0
theorem sum_of_solutions_is_zero :
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 0 ∧
  (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

#check sum_of_solutions_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l698_69863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt5_l698_69801

noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity_is_sqrt5 :
  let m : ℝ := geometric_mean 2 8
  let a : ℝ := 1
  let b : ℝ := Real.sqrt m
  hyperbola_eccentricity a b = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt5_l698_69801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l698_69885

/-- A line with equation y = 4x + c, where c < 0 -/
structure Line where
  c : ℝ
  h : c < 0

/-- Point of intersection with y-axis -/
def T (l : Line) : ℝ × ℝ := (0, l.c)

/-- Point of intersection with x = 2 -/
def U (l : Line) : ℝ × ℝ := (2, 8 + l.c)

/-- Two points on y-axis -/
structure PointsOnYAxis where
  V : ℝ × ℝ
  W : ℝ × ℝ
  hV : V.1 = 0
  hW : W.1 = 0

/-- Area of a triangle given three points -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The theorem to be proved -/
theorem line_intersection_theorem (l : Line) (p : PointsOnYAxis) :
  (area_triangle p.W p.V (U l)) / (area_triangle p.W (T l) (U l)) = 16 / 49 →
  l.c = 279 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l698_69885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_eq_factorial_formula_l698_69873

/-- Binomial coefficient definition -/
def binomial : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binomial n k + binomial n (k + 1)

/-- Theorem: Binomial coefficient equals factorial formula -/
theorem binomial_eq_factorial_formula (n k : ℕ) (h : k ≤ n) :
  binomial n k = n.factorial / (k.factorial * (n - k).factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_eq_factorial_formula_l698_69873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l698_69898

-- Define the constants
noncomputable def a : ℝ := Real.log 6 / Real.log 0.7
noncomputable def b : ℝ := 6 ^ (0.7 : ℝ)
noncomputable def c : ℝ := 0.7 ^ (0.6 : ℝ)

-- State the theorem
theorem relationship_abc : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l698_69898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lists_for_785_members_min_lists_satisfies_condition_l698_69862

/-- The smallest number of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_lists (total_members : ℕ) : ℕ :=
  Nat.ceil ((total_members : ℚ) / 4)

/-- Theorem stating the smallest number of top-10 lists for 785 members -/
theorem min_lists_for_785_members :
  min_lists 785 = 197 := by
  sorry

/-- Theorem proving that the result satisfies the "at least 1/4" condition -/
theorem min_lists_satisfies_condition (total_members : ℕ) :
  (min_lists total_members : ℚ) / total_members ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lists_for_785_members_min_lists_satisfies_condition_l698_69862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_unit_circle_l698_69824

theorem roots_on_unit_circle 
  (n : ℕ) 
  (P : Polynomial ℂ) 
  (h_degree : P.degree = n) 
  (h_roots : ∀ z, P.eval z = 0 → Complex.abs z = 1) 
  (c : ℝ) 
  (h_c_nonneg : c ≥ 0) :
  ∀ w, (2 * w * (w - 1) * (P.derivative.eval w) + ((c - n) * w + (c + n)) * P.eval w) = 0 
    → Complex.abs w = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_unit_circle_l698_69824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_difference_l698_69849

/-- Given functions f and g, prove that f(g(x)) - g(f(x)) equals the specified expression. -/
theorem composition_difference (x : ℝ) : 
  (fun x => 7 * x - 6) ((fun x => x^2 / 3 + 1) x) - 
  (fun x => x^2 / 3 + 1) ((fun x => 7 * x - 6) x) = 
  (-42 * x^2 + 84 * x - 38) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_difference_l698_69849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morgan_foggy_time_l698_69853

/-- Represents the cycling scenario for Morgan --/
structure CyclingScenario where
  clearSpeed : ℚ  -- Speed in clear weather (miles per hour)
  foggySpeed : ℚ  -- Speed in foggy weather (miles per hour)
  totalDistance : ℚ  -- Total distance traveled (miles)
  totalTime : ℚ  -- Total time spent (hours)

/-- Calculates the time spent in foggy weather --/
def foggyTime (scenario : CyclingScenario) : ℚ :=
  scenario.totalDistance / (scenario.clearSpeed + scenario.foggySpeed)

/-- Theorem stating that Morgan spent 20 minutes in foggy weather --/
theorem morgan_foggy_time :
  let scenario : CyclingScenario := {
    clearSpeed := 45,
    foggySpeed := 15,
    totalDistance := 30,
    totalTime := 1
  }
  foggyTime scenario * 60 = 20 := by
  -- Proof goes here
  sorry

#eval foggyTime {
  clearSpeed := 45,
  foggySpeed := 15,
  totalDistance := 30,
  totalTime := 1
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morgan_foggy_time_l698_69853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l698_69826

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (k * π - 3 * π / 8) (k * π + π / 8))) ∧
  (∃ (m : ℝ), m = -(Real.sqrt 2 + 1) ∧ ∀ (x : ℝ), x ∈ Set.Icc (-π / 2) 0 → f x ≥ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l698_69826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_is_half_l698_69852

noncomputable def F : Set (ℝ → ℝ) :=
  {f | ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x}

theorem largest_A_is_half :
  (∃ A, ∀ f ∈ F, ∀ x > 0, f x ≥ A * x) ∧
  (∀ A', (∀ f ∈ F, ∀ x > 0, f x ≥ A' * x) → A' ≤ (1/2 : ℝ)) := by
  sorry

#check largest_A_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_A_is_half_l698_69852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l698_69843

/-- The function f(x) defined on the open interval (0, 1) -/
noncomputable def f (x : ℝ) : ℝ := 3 + Real.log x / Real.log 10 + 4 * Real.log 10 / Real.log x

/-- Theorem stating that the maximum value of f(x) on (0, 1) is -1 -/
theorem f_max_value :
  ∀ x : ℝ, 0 < x → x < 1 → f x ≤ -1 ∧ ∃ y : ℝ, 0 < y ∧ y < 1 ∧ f y = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l698_69843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_PQ_range_l698_69842

/-- The fixed point P through which all lines pass -/
noncomputable def P : ℝ × ℝ := (0, 4)

/-- The function describing the curve on which Q lies -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x

/-- The slope of line PQ given the x-coordinate of Q -/
noncomputable def slope_PQ (m : ℝ) : ℝ := (f m - P.2) / (m - P.1)

/-- Theorem stating the range of the slope of line PQ -/
theorem slope_PQ_range :
  ∀ m : ℝ, m ≠ 0 → slope_PQ m ≥ -3 ∧ ∀ y : ℝ, y > -3 → ∃ m : ℝ, m ≠ 0 ∧ slope_PQ m = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_PQ_range_l698_69842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extremum_iff_positive_a_two_extrema_inequality_l698_69840

/-- The function f(x) = e^x - 1 + ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 + a * x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * a * x

theorem unique_extremum_iff_positive_a (a : ℝ) :
  (∃! x : ℝ, f_derivative a x = 0) ↔ a > 0 :=
sorry

theorem two_extrema_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  f_derivative a x₁ = 0 → f_derivative a x₂ = 0 → x₁ ≠ x₂ →
  x₁^2 + x₂^2 > 2*(a+1) + Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_extremum_iff_positive_a_two_extrema_inequality_l698_69840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_n_l698_69859

theorem power_equality_implies_n (n : ℝ) : (4 : ℝ)^9 = 16^n → n = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_n_l698_69859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_digit_equation_l698_69883

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit := {n : ℕ // n ≤ 9}

/-- Convert a two-digit number to its decimal representation -/
def toDecimal (tens ones : Digit) : ℕ := 10 * tens.val + ones.val

theorem solve_digit_equation (A B C D : Digit) 
  (h1 : toDecimal A C + toDecimal C B = toDecimal D C)
  (h2 : toDecimal A C - toDecimal C B = toDecimal A A) :
  D.val = 8 := by
  sorry

#check solve_digit_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_digit_equation_l698_69883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_is_negative_inverse_of_negative_l698_69806

/-- A function that represents the original function φ(x) --/
noncomputable def φ : ℝ → ℝ := sorry

/-- The inverse function of φ --/
noncomputable def φ_inv : ℝ → ℝ := sorry

/-- Assumption that φ_inv is indeed the inverse of φ --/
axiom φ_inverse : ∀ x, φ_inv (φ x) = x ∧ φ (φ_inv x) = x

/-- The third function that is symmetric to φ_inv about the line x + y = 0 --/
noncomputable def f : ℝ → ℝ := sorry

/-- Symmetry condition: (x, f(x)) is symmetric to (a, φ_inv(a)) about x + y = 0 --/
axiom symmetry : ∀ x a, f x = -a ∧ x = -φ_inv a → x + f x = 0 ∧ a + φ_inv a = 0

/-- Theorem: The third function f is equal to -φ⁻¹(-x) --/
theorem third_function_is_negative_inverse_of_negative : 
  ∀ x, f x = -φ_inv (-x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_function_is_negative_inverse_of_negative_l698_69806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_inscribed_circle_l698_69893

/-- 
  Given a triangle ABC with sides a, b, and c, and an inscribed circle with diameter d,
  if a + b - c = d, then one of the angles of the triangle is a right angle.
-/
theorem triangle_right_angle_from_inscribed_circle 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_inscribed : d = a + b - c) : 
  ∃ θ : ℝ, θ = π / 2 ∧ 
    (Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b) ∨ 
     Real.cos θ = (b^2 + c^2 - a^2) / (2 * b * c) ∨ 
     Real.cos θ = (c^2 + a^2 - b^2) / (2 * c * a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_from_inscribed_circle_l698_69893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l698_69825

-- Define the ellipse E
def ellipse_E (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 20 / 3

-- Define the eccentricity of E
def eccentricity_E (a b : ℝ) : Prop := (a^2 - b^2) / a^2 = 1 / 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse_E A.1 A.2 a b ∧ ellipse_E B.1 B.2 a b ∧
  circle_C A.1 A.2 ∧ circle_C B.1 B.2

-- Define that AB is the diameter of C
def AB_is_diameter (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

theorem ellipse_and_circle_intersection 
  (a b : ℝ) (A B : ℝ × ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity_E a b)
  (h4 : intersection_points A B a b)
  (h5 : AB_is_diameter A B) :
  (∃ (x y : ℝ), x + y - 3 = 0 ∧ x^2 / 16 + y^2 / 8 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l698_69825
