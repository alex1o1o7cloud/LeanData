import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_k_1_safe_k_range_l373_37367

/-- Environmental compensation fee as a function of wastewater discharge volume -/
noncomputable def P (k : ℝ) (x : ℝ) : ℝ := k * x^3

/-- Gross profit as a function of wastewater discharge volume -/
noncomputable def Q (x : ℝ) : ℝ := (1/2) * x^2 + 10*x

/-- Net profit as a function of wastewater discharge volume and k -/
noncomputable def y (k : ℝ) (x : ℝ) : ℝ := Q x - P k x

/-- Theorem: When k = 1, the wastewater discharge volume that maximizes net profit is 2 -/
theorem max_profit_at_k_1 : 
  ∃ (x : ℝ), x = 2 ∧ ∀ (z : ℝ), y 1 x ≥ y 1 z :=
by sorry

/-- Theorem: The range of k that ensures no health risk (x ≤ 1) while maximizing profit is [11/3, 10] -/
theorem safe_k_range : 
  ∀ (k : ℝ), (k ≥ 11/3 ∧ k ≤ 10) ↔ 
  (∃ (x : ℝ), x ≤ 1 ∧ ∀ (z : ℝ), y k x ≥ y k z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_k_1_safe_k_range_l373_37367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l373_37371

/-- The equation of a hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 3

/-- The equation of the asymptotes -/
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 / 2 * x ∨ y = -Real.sqrt 2 / 2 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ± (√2/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → 
  (∃ ε > 0, ∀ x' y', x'^2 + y'^2 > 1/ε^2 → hyperbola x' y' → asymptotes x' y') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l373_37371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abs_roots_squared_l373_37308

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 23 + 101 / x

-- Define the equation
def equation (x : ℝ) : Prop :=
  x = g (g (g (g (g x))))

-- Define B as the sum of absolute values of roots
noncomputable def B : ℝ :=
  Real.sqrt 427

-- State the theorem
theorem sum_of_abs_roots_squared :
  B^2 = 427 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abs_roots_squared_l373_37308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l373_37390

-- Define a, b, and c
noncomputable def a : ℝ := (1/2 : ℝ) ^ (0.1 : ℝ)
noncomputable def b : ℝ := (1/2 : ℝ) ^ (-0.1 : ℝ)
noncomputable def c : ℝ := (1/2 : ℝ) ^ (0.2 : ℝ)

-- Theorem statement
theorem order_of_numbers : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l373_37390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_polar_l373_37363

/-- Given a line and a circle in polar coordinates, prove that if they are tangent and a > 0, then a = 1 -/
theorem tangent_line_circle_polar (a : ℝ) : 
  a > 0 → 
  (∃ (ρ θ : ℝ), ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0) →
  (∃ (ρ θ : ℝ), ρ = 2 * a * Real.cos θ) →
  (∃ (ρ θ : ℝ), ρ * Real.cos θ + Real.sqrt 3 * ρ * Real.sin θ + 1 = 0 ∧ ρ = 2 * a * Real.cos θ) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_polar_l373_37363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_correct_l373_37395

/-- Represents the walking scenario with given and calculated values -/
structure WalkingScenario where
  actual_speed : ℚ
  actual_distance : ℚ
  additional_distance : ℚ
  faster_speed : ℚ

/-- Calculates the faster speed given the actual speed, distance, and additional distance -/
def calculate_faster_speed (w : WalkingScenario) : ℚ :=
  (w.actual_distance + w.additional_distance) / (w.actual_distance / w.actual_speed)

/-- Theorem stating that the calculated faster speed is correct -/
theorem faster_speed_correct (w : WalkingScenario) 
  (h1 : w.actual_speed = 10)
  (h2 : w.actual_distance = 50)
  (h3 : w.additional_distance = 20)
  (h4 : w.faster_speed = calculate_faster_speed w) :
  w.faster_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_correct_l373_37395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tallest_cylinder_in_torus_l373_37328

/-- Represents a torus in 3D space -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  center : ℝ × ℝ × ℝ
  cross_section_center : ℝ × ℝ × ℝ
  cross_section_radius : ℝ

/-- Represents a cylinder in 3D space -/
structure Cylinder where
  radius : ℝ
  height : ℝ
  axis : ℝ × ℝ × ℝ

/-- The height of the tallest cylinder that can fit inside a torus -/
def tallest_cylinder_height (t : Torus) : ℝ :=
  t.cross_section_center.2.2 + t.cross_section_radius - t.center.2.2

theorem tallest_cylinder_in_torus :
  let t : Torus := {
    inner_radius := 4,
    outer_radius := 6,
    center := (0, 0, 1),
    cross_section_center := (5, 0, 1),
    cross_section_radius := 1
  }
  let c : Cylinder := {
    radius := 1,
    height := tallest_cylinder_height t,
    axis := (0, 0, 1)
  }
  tallest_cylinder_height t = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tallest_cylinder_in_torus_l373_37328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_max_area_l373_37325

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = π
  side_positive : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = (Real.sqrt 3 / 3) * t.c * Real.tan t.B

-- Theorem 1: Angle B is 60 degrees
theorem angle_B_is_60 (t : Triangle) (h : condition t) : t.B = π / 3 := by
  sorry

-- Theorem 2: Maximum area when b = 2
theorem max_area (t : Triangle) (h1 : condition t) (h2 : t.b = 2) :
  ∀ s : Real, s = t.a * t.c * Real.sin t.B / 2 → s ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_60_max_area_l373_37325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l373_37383

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | (n + 1) => sequence_a n / n + n / sequence_a n

theorem sequence_a_property (n : ℕ) (h : n ≥ 4) :
  ↑n ≤ (sequence_a n)^2 ∧ (sequence_a n)^2 < ↑n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_l373_37383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l373_37394

/-- The parabola C defined by y² = -x -/
def parabola_C (x y : ℝ) : Prop := y^2 = -x

/-- The line l defined by x + 2y - 3 = 0 -/
def line_l (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The shortest distance from a point on parabola C to line l -/
noncomputable def shortest_distance : ℝ := 2 * Real.sqrt 5 / 5

theorem shortest_distance_parabola_to_line :
  ∀ (M : ℝ × ℝ), parabola_C M.1 M.2 →
  ∃ (d : ℝ), d = shortest_distance ∧
  ∀ (P : ℝ × ℝ), line_l P.1 P.2 →
  d ≤ Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l373_37394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_walking_rate_l373_37338

/-- The distance between points X and Y in miles -/
noncomputable def total_distance : ℝ := 31

/-- Bob's walking rate in miles per hour -/
noncomputable def bob_rate : ℝ := 4

/-- The distance Bob walked when they met, in miles -/
noncomputable def bob_distance : ℝ := 16

/-- The time difference between Yolanda and Bob starting their walks, in hours -/
noncomputable def time_difference : ℝ := 1

/-- Yolanda's walking rate in miles per hour -/
noncomputable def yolanda_rate : ℝ := (total_distance - bob_distance) / (bob_distance / bob_rate + time_difference)

theorem yolanda_walking_rate :
  yolanda_rate = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yolanda_walking_rate_l373_37338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_floor_value_l373_37317

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem max_floor_value (x : ℝ) : 
  floor ((x + 4) / 10) = 5 → floor (6 * x / 5) ≤ 67 ∧ ∃ y : ℝ, floor ((y + 4) / 10) = 5 ∧ floor (6 * y / 5) = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_floor_value_l373_37317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l373_37359

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of overlap between two circles -/
noncomputable def circleOverlap (c1 c2 : Circle) : ℝ := sorry

/-- The condition that two circles are tangent -/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- The condition that a circle touches another circle at a point -/
def touches (c1 c2 : Circle) : Prop := sorry

/-- The area inside one circle but outside two other circles -/
noncomputable def areaInsideButOutside (c1 c2 c3 : Circle) : ℝ := sorry

theorem circle_area_problem (A B C : Circle) 
  (hA : A.radius = 1)
  (hB : B.radius = 1)
  (hC : C.radius = 2)
  (hAB : areTangent A B)
  (hAC : areTangent A C)
  (hBC : touches B C)
  (hBCnotAB : ¬(hBC = hAB)) :
  ∃ x, 4 * π - x = areaInsideButOutside C A B :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l373_37359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_rate_fraction_proof_l373_37357

/-- Represents the fraction of a week with reduced electricity rates -/
def reducedRateFraction : ℚ := 9/14

/-- Function representing reduced hours on a weekday -/
def reducedHoursWeekday (day : ℕ) : ℕ := 
  if day ≤ 5 then 12 else 0

/-- Function representing reduced hours on a weekend day -/
def reducedHoursWeekend (day : ℕ) : ℕ := 
  if day > 5 ∧ day ≤ 7 then 24 else 0

theorem reduced_rate_fraction_proof :
  -- Given conditions
  (∀ week : ℕ, week = 7) →  -- A week has 7 days
  (∀ day : ℕ, day = 24) →   -- Each day has 24 hours
  (∀ weekday : ℕ, weekday ≤ 5 → reducedHoursWeekday weekday = 12) →  -- Reduced rates apply 12 hours on weekdays
  (∀ weekend : ℕ, weekend > 5 ∧ weekend ≤ 7 → reducedHoursWeekend weekend = 24) →  -- Reduced rates apply 24 hours on weekends
  -- Conclusion
  reducedRateFraction = 9/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_rate_fraction_proof_l373_37357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disinfection_effectiveness_l373_37314

noncomputable def drug_concentration (t : ℝ) : ℝ :=
  if t ≤ 10 then 0.1 * t else (1/2)^(0.1 * t - 1)

theorem disinfection_effectiveness :
  (∃ t : ℝ, t ≥ 30 ∧ ∀ t' ≥ t, drug_concentration t' ≤ 0.25) ∧
  (∃ a b : ℝ, a < b ∧ b - a > 8 ∧ ∀ t ∈ Set.Icc a b, drug_concentration t > 0.5) :=
by
  sorry

#check disinfection_effectiveness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disinfection_effectiveness_l373_37314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_when_a_zero_f_greater_than_one_iff_a_greater_than_two_l373_37373

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem 1: f has no zeros when a = 0
theorem no_zeros_when_a_zero :
  ∀ x > 0, f 0 x ≠ 0 :=
by
  sorry

-- Theorem 2: f(x) > 1 for all x ∈ [1/e, e] iff a > 2
theorem f_greater_than_one_iff_a_greater_than_two :
  ∀ a ≥ 1, (∀ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x > 1) ↔ a > 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_when_a_zero_f_greater_than_one_iff_a_greater_than_two_l373_37373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l373_37374

noncomputable def complex_number : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I

theorem complex_number_in_fourth_quadrant :
  Real.sign (complex_number.re) = 1 ∧ Real.sign (complex_number.im) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l373_37374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l373_37320

/-- Given two fixed points F₁ and F₂ in a metric space, with distance between them equal to 4,
    prove that any point M satisfying |MF₁| + |MF₂| = 4 lies on the line segment F₁F₂. -/
theorem point_on_line_segment 
  {X : Type*} [NormedAddCommGroup X] [InnerProductSpace ℝ X] (F₁ F₂ M : X) :
  ‖F₁ - F₂‖ = 4 →
  ‖M - F₁‖ + ‖M - F₂‖ = 4 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • F₁ + t • F₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l373_37320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_l373_37319

/-- A's work efficiency -/
noncomputable def A : ℝ := sorry

/-- B's work efficiency -/
noncomputable def B : ℝ := 1 / 27

/-- The factor by which A is better than B -/
noncomputable def k : ℝ := sorry

/-- Theorem stating the ratio of A's work efficiency to B's work efficiency -/
theorem work_efficiency_ratio :
  (A + B = 1 / 9) →
  (A = k * B) →
  k = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_l373_37319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_is_sqrt_2_l373_37372

noncomputable def z : ℂ := (6 + 8 * Complex.I) / ((4 + 3 * Complex.I) * (1 + Complex.I))

theorem z_magnitude_is_sqrt_2 : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_is_sqrt_2_l373_37372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l373_37370

/-- The function f(x) = sin x - x --/
noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

/-- Theorem stating the inequality between f(-π/4), f(1), and f(π/3) --/
theorem f_inequality : f (-Real.pi/4) > f 1 ∧ f 1 > f (Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l373_37370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triples_existence_special_triples_existence_proof_l373_37387

/-- A special triple is a triple of nonnegative real numbers that sum to 1 -/
def SpecialTriple : Type := { t : (ℝ × ℝ × ℝ) // t.1 ≥ 0 ∧ t.2.1 ≥ 0 ∧ t.2.2 ≥ 0 ∧ t.1 + t.2.1 + t.2.2 = 1 }

/-- A triple (a₁, a₂, a₃) is better than another triple (b₁, b₂, b₃) if exactly two out of the three inequalities a₁ > b₁, a₂ > b₂, and a₃ > b₃ hold -/
def IsBetter (a b : ℝ × ℝ × ℝ) : Prop :=
  (a.1 > b.1 ∧ a.2.1 > b.2.1 ∧ a.2.2 ≤ b.2.2) ∨
  (a.1 > b.1 ∧ a.2.1 ≤ b.2.1 ∧ a.2.2 > b.2.2) ∨
  (a.1 ≤ b.1 ∧ a.2.1 > b.2.1 ∧ a.2.2 > b.2.2)

/-- For any natural number n ≥ 3, there exists a collection S of special triples with |S| = n
    such that any special triple is bettered by at least one element of S,
    and no such collection exists for n < 3 -/
theorem special_triples_existence (n : ℕ) : Prop :=
  (n ≥ 3 → ∃ (S : Finset SpecialTriple),
    S.card = n ∧
    ∀ (t : SpecialTriple), ∃ (s : SpecialTriple), s ∈ S ∧ IsBetter s.val t.val) ∧
  (n < 3 → ¬∃ (S : Finset SpecialTriple),
    S.card = n ∧
    ∀ (t : SpecialTriple), ∃ (s : SpecialTriple), s ∈ S ∧ IsBetter s.val t.val)

/-- Proof of the theorem -/
theorem special_triples_existence_proof : ∀ (n : ℕ), special_triples_existence n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triples_existence_special_triples_existence_proof_l373_37387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l373_37333

/-- Reflects a point (x, y) across the line y = mx + b -/
noncomputable def reflect (x y m b : ℝ) : ℝ × ℝ :=
  let x' := (x * (1 - m^2) + 2 * m * (y - b)) / (1 + m^2)
  let y' := (2 * m * x - m^2 * y + 2 * b) / (1 + m^2)
  (x', y')

/-- The theorem stating that if (2, 3) is reflected to (10, 7) across y = mx + b, then m + b = 15 -/
theorem reflection_sum (m b : ℝ) : 
  reflect 2 3 m b = (10, 7) → m + b = 15 := by
  sorry

#check reflection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l373_37333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_relation_l373_37309

/-- Theorem about the relationship between face areas and angles in a tetrahedron -/
theorem tetrahedron_face_area_relation 
  (a b c d : ℝ) 
  (α β γ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) :
  d^2 = a^2 + b^2 + c^2 - 2*a*b*Real.cos γ - 2*b*c*Real.cos α - 2*c*a*Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_relation_l373_37309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_difference_l373_37382

/-- The amount of paint in square feet used by Mary for her dragon -/
def mary_paint : ℝ := 3

/-- The amount of paint in square feet used for the sun -/
def sun_paint : ℝ := 5

/-- The total amount of paint in square feet originally in the jar -/
def total_paint : ℝ := 13

/-- The difference in paint usage between Mike and Mary in square feet -/
def paint_difference : ℝ := 2

/-- The amount of paint in square feet used by Mike for his castle -/
def mike_paint : ℝ := mary_paint + paint_difference

theorem paint_usage_difference :
  mary_paint + mike_paint + sun_paint = total_paint ∧
  mike_paint > mary_paint ∧
  paint_difference = mike_paint - mary_paint :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_usage_difference_l373_37382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_theorem_l373_37335

/-- The distance Bob walked when meeting Yolanda -/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) (time_diff : ℝ) : ℝ :=
  let t := (total_distance + bob_rate * time_diff) / (yolanda_rate + bob_rate)
  bob_rate * (t - time_diff)

/-- Theorem stating that Bob walked 28 miles when they met -/
theorem bob_distance_theorem :
  distance_bob_walked 52 3 4 1 = 28 := by
  -- Unfold the definition of distance_bob_walked
  unfold distance_bob_walked
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_theorem_l373_37335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l373_37339

noncomputable def data_set : List ℝ := [4.7, 4.8, 5.1, 5.4, 5.5]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_data_set :
  variance data_set = 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l373_37339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_minimum_value_g_l373_37358

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-x) + a * x - 1 / x

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := f 0 (-x) + 2 * x

-- Theorem for part (I)
theorem extreme_value_condition (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), x ≠ -1 → |f a x| ≤ |f a (-1)|) → 
  a = 0 := by
  sorry

-- Theorem for part (II)
theorem minimum_value_g : 
  (∀ x > 0, g x ≥ g (1/2)) ∧ g (1/2) = 3 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_minimum_value_g_l373_37358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l373_37330

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- State the theorem
theorem zero_point_in_interval :
  -- f is continuous on (0, +∞)
  Continuous (fun x : ℝ => f x) →
  -- f is strictly increasing on (0, +∞)
  StrictMono f →
  -- f(2) < 0
  f 2 < 0 →
  -- f(3) > 0
  f 3 > 0 →
  -- There exists a zero point of f in the interval (2, 3)
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l373_37330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l373_37323

noncomputable def f (x : ℝ) := Real.sqrt (1 - 2 * Real.cos x) + Real.log (Real.sin x - Real.sqrt 2 / 2)

def domain (x : ℝ) : Prop :=
  ∃ k : ℤ, Real.pi / 3 + 2 * k * Real.pi ≤ x ∧ x < 3 * Real.pi / 4 + 2 * k * Real.pi

theorem f_domain :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l373_37323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_AB_l373_37368

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def M : ℝ × ℝ := (4, -4)

-- Assume M is on the parabola
axiom M_on_parabola : parabola M.1 M.2

-- Define points A and B
variable (A B : ℝ × ℝ)

-- Assume A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- Define vector perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop :=
  (v.1 * w.1 + v.2 * w.2 = 0)

-- Assume MA is perpendicular to MB
axiom MA_perp_MB : perpendicular (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the line AB
def line_AB (m n : ℝ) (y : ℝ) : ℝ := m * y + n

-- Theorem: The maximum distance from M to line AB is 4√5
theorem max_distance_M_to_AB :
  ∃ (m n : ℝ), ∀ (y : ℝ),
    (∃ (max_dist : ℝ), max_dist = distance M (line_AB m n y, y) ∧
      ∀ (m' n' y' : ℝ), distance M (line_AB m' n' y', y') ≤ max_dist) ∧
    max_dist = 4 * Real.sqrt 5 := by
  sorry

#check max_distance_M_to_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_AB_l373_37368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_temp_change_l373_37344

/-- Convert Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

/-- Temperature change stages -/
structure TempStages where
  initial : ℝ
  boiling : ℝ
  reduced : ℝ
  final : ℝ

/-- Calculate temperature change -/
noncomputable def temp_change (stages : TempStages) : ℝ :=
  (celsius_to_fahrenheit stages.boiling - celsius_to_fahrenheit stages.initial) +
  (celsius_to_fahrenheit stages.reduced - celsius_to_fahrenheit stages.boiling) +
  (celsius_to_fahrenheit stages.final - celsius_to_fahrenheit stages.reduced)

theorem total_temp_change :
  let initial_temp := 60
  let boiling_point := 100
  let reduced_temp := boiling_point - (boiling_point / 3)
  let final_temp := -10
  let stages := TempStages.mk initial_temp boiling_point reduced_temp final_temp
  temp_change stages = -126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_temp_change_l373_37344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l373_37379

/-- Represents a chessboard square --/
structure Square where
  x : Fin 8
  y : Fin 8

/-- Represents a 2x1 domino --/
structure Domino where
  sq1 : Square
  sq2 : Square

/-- The color of a square (Black or White) --/
inductive Color where
  | Black
  | White

/-- Function to determine the color of a square --/
def squareColor (sq : Square) : Color :=
  if (sq.x.val + sq.y.val) % 2 = 0 then Color.Black else Color.White

/-- The set of all squares on the chessboard --/
def allSquares : Set Square :=
  {sq | sq.x.val < 8 ∧ sq.y.val < 8}

/-- The set of squares after removing two opposite corners --/
def remainingSquares : Set Square :=
  {sq ∈ allSquares | sq ≠ ⟨0, 0⟩ ∧ sq ≠ ⟨7, 7⟩}

/-- A valid tiling of the chessboard --/
def validTiling (tiling : Set Domino) : Prop :=
  (∀ sq ∈ remainingSquares, ∃ d ∈ tiling, sq = d.sq1 ∨ sq = d.sq2) ∧
  (∀ d1 d2, d1 ∈ tiling → d2 ∈ tiling → d1 ≠ d2 → 
    d1.sq1 ≠ d2.sq1 ∧ d1.sq1 ≠ d2.sq2 ∧ d1.sq2 ≠ d2.sq1 ∧ d1.sq2 ≠ d2.sq2)

theorem no_valid_tiling : ¬∃ tiling : Set Domino, validTiling tiling := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l373_37379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_over_f_two_equals_sqrt_three_over_three_l373_37396

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2^(x - 1) else Real.tan ((Real.pi/3) * x)

theorem f_one_over_f_two_equals_sqrt_three_over_three :
  f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_over_f_two_equals_sqrt_three_over_three_l373_37396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l373_37313

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its asymptote equation is y = (√5/2)x, then its eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5 / 2) : 
  Real.sqrt (a^2 + b^2) / a = 3 / 2 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l373_37313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l373_37301

/-- The area of a triangle with base 10 and height 10 is 50 -/
theorem triangle_area
  (base : Real)
  (height : Real)
  (h1 : base = 10)
  (h2 : height = 10) :
  (base * height) / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l373_37301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l373_37322

theorem remainder_theorem (a b : ℕ) (ha : a % 6 = 2) (hb : b % 6 = 5) :
  (b ^ 2 - 3 * a) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l373_37322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_ratio_l373_37343

theorem square_triangle_area_ratio : 
  ∀ (h : ℝ), h > 0 → 
  let s := 3 * h
  let a := 2 * h / Real.sqrt 3
  (s^2) / ((Real.sqrt 3 / 4) * a^2) = 6 := by
  intro h h_pos
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_ratio_l373_37343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relation_l373_37391

-- Define the plane and points
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (P A B C : V)

-- Define the conditions
variable (h1 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h2 : (A - P) + (B - P) + (C - P) = 0)
variable (h3 : ∃ m : ℝ, (B - A) + (C - A) = m • (P - A))

-- State the theorem
theorem point_relation :
  ∃ m : ℝ, (B - A) + (C - A) = m • (P - A) ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_relation_l373_37391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_l373_37364

-- Define the polynomial f(x)
noncomputable def f (x : Real) : Real := -6 * x^5 + 2 * x^4 + 5 * x^2 - 4

-- State the theorem
theorem degree_of_h (h : Polynomial Real) : 
  (∃ (c : Polynomial Real), (∀ x, f x + h.eval x = c.eval x) ∧ c.degree = 2) →
  h.degree = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_l373_37364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_half_l373_37366

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 1 / (x - 2m + 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  1 / (x - 2*m + 1)

/-- If f(x) = 1 / (x - 2m + 1) is an odd function, then m = 1/2 -/
theorem odd_function_implies_m_half :
  ∀ m : ℝ, IsOdd (f m) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_half_l373_37366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l373_37300

/-- The sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 3 * a n + 4

/-- Theorem stating that the general term of the sequence is 3^n - 2 -/
theorem a_general_term (n : ℕ) : a n = 3^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l373_37300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_savings_l373_37349

-- Define the computer models and their prices
def model_A_price : ℝ := 500
def model_B_price : ℝ := 650
def model_C_price : ℝ := 800

-- Define the employee's purchase quantities
def model_A_quantity : ℕ := 5
def model_B_quantity : ℕ := 3
def model_C_quantity : ℕ := 2

-- Define the employee's years of service
def years_of_service : ℝ := 1.5

-- Define the discount percentages
noncomputable def employee_discount (years : ℝ) : ℝ :=
  if years < 1 then 0.10
  else if years < 2 then 0.15
  else 0.20

def bulk_purchase_discount (total_quantity : ℕ) : ℝ :=
  if total_quantity ≥ 5 then 0.05 else 0

-- Define the store credit amount
def store_credit : ℝ := 100

-- Calculate the total retail price
def total_retail_price : ℝ :=
  model_A_price * model_A_quantity +
  model_B_price * model_B_quantity +
  model_C_price * model_C_quantity

-- Calculate the total discount percentage
noncomputable def total_discount_percentage : ℝ :=
  employee_discount years_of_service +
  bulk_purchase_discount (model_A_quantity + model_B_quantity + model_C_quantity)

-- Calculate the final price after discounts and store credit
noncomputable def final_price : ℝ :=
  total_retail_price * (1 - total_discount_percentage) - store_credit

-- Theorem to prove
theorem employee_savings : total_retail_price - final_price = 1310 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_savings_l373_37349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l373_37347

/-- Represents a right circular cone containing liquid -/
structure LiquidCone where
  radius : ℝ
  height : ℝ

/-- Represents the scenario with two cones and marbles -/
structure TwoConesScenario where
  narrow_cone : LiquidCone
  wide_cone : LiquidCone
  marble_radius : ℝ
  initial_volume : ℝ

/-- The rise in liquid level after dropping a marble -/
noncomputable def liquid_rise (cone : LiquidCone) (marble_radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * marble_radius^3 / (Real.pi * cone.radius^2)

theorem liquid_rise_ratio (scenario : TwoConesScenario) 
  (h_narrow_radius : scenario.narrow_cone.radius = 3)
  (h_wide_radius : scenario.wide_cone.radius = 6)
  (h_marble_radius : scenario.marble_radius = 1)
  (h_same_volume : Real.pi * scenario.narrow_cone.radius^2 * scenario.narrow_cone.height = 
                   Real.pi * scenario.wide_cone.radius^2 * scenario.wide_cone.height) :
  liquid_rise scenario.narrow_cone scenario.marble_radius / 
  liquid_rise scenario.wide_cone scenario.marble_radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l373_37347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_improvement_l373_37376

theorem car_efficiency_improvement (original_efficiency : ℚ) (tank_capacity : ℚ)
  (solar_panel_effect : ℚ) (regen_braking_effect : ℚ) (hybrid_effect : ℚ) :
  original_efficiency = 33 →
  tank_capacity = 16 →
  solar_panel_effect = 3/4 →
  regen_braking_effect = 15/100 →
  hybrid_effect = 1/10 →
  ∃ (improved_miles : ℚ),
    improved_miles ≥ 391 ∧
    improved_miles < 392 ∧
    improved_miles = 
      (original_efficiency * tank_capacity * 
        (1 / solar_panel_effect) * 
        (1 / (1 - regen_braking_effect)) * 
        (1 / (1 - hybrid_effect))) -
      (original_efficiency * tank_capacity) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_efficiency_improvement_l373_37376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_lateral_area_l373_37303

/-- Represents a triangular prism with specific properties -/
structure TriangularPrism where
  base_side_length : ℝ
  lateral_edge_length : ℝ
  sphere_radius : ℝ
  /-- The base is an equilateral triangle -/
  base_equilateral : base_side_length > 0
  /-- The ratio of lateral edge to base side is 2:1 -/
  edge_ratio : lateral_edge_length = 2 * base_side_length
  /-- All vertices are on the surface of a sphere -/
  vertices_on_sphere : sphere_radius^2 = base_side_length^2 + (Real.sqrt 3 / 3 * base_side_length)^2
  /-- The surface area of the sphere is 16π/3 -/
  sphere_surface_area : 4 * Real.pi * sphere_radius^2 = 16 * Real.pi / 3

/-- The lateral surface area of the triangular prism is 6 -/
theorem triangular_prism_lateral_area (prism : TriangularPrism) :
  3 * prism.base_side_length * prism.lateral_edge_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_lateral_area_l373_37303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l373_37369

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  B = 60 →           -- Given angle B
  C = 75 →           -- Given angle C
  a = 4 →            -- Given side a
  (a / Real.sin A = b / Real.sin B) →  -- Sine law
  b = 2 :=           -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l373_37369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l373_37378

/-- The power function -/
noncomputable def f (n : ℝ) : ℝ → ℝ := fun x ↦ x^n

/-- The derivative of the power function -/
noncomputable def f_derivative (n : ℝ) : ℝ → ℝ := fun x ↦ n * x^(n-1)

theorem tangent_line_equation :
  ∃ n : ℝ, 
    f n 2 = 8 ∧ 
    (fun x y ↦ 12*x - y - 16 = 0) = 
    (fun x y ↦ y - 8 = (f_derivative n 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l373_37378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_neg_one_l373_37398

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line a^2x - y + 1 = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := a^2

/-- The slope of the second line x - ay - 2 = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := 1/a

/-- The condition for perpendicularity of the given lines -/
def perpendicular_condition (a : ℝ) : Prop :=
  are_perpendicular (slope1 a) (slope2 a)

/-- Theorem stating that a = -1 is a necessary and sufficient condition for perpendicularity -/
theorem perpendicular_iff_neg_one (a : ℝ) :
  perpendicular_condition a ↔ a = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_neg_one_l373_37398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_chain_l373_37318

/-- Represents a chain with a given number of links -/
structure Chain where
  links : ℕ

/-- Represents a cut in the chain -/
structure Cut where
  position : ℕ

/-- Checks if a set of cuts can form all weights from 1 to n grams -/
def canFormAllWeights (chain : Chain) (cuts : List Cut) (n : ℕ) : Prop :=
  ∀ w : ℕ, w ≥ 1 ∧ w ≤ n → ∃ pieces : List ℕ, 
    pieces.sum = w ∧ 
    (∀ piece ∈ pieces, ∃ start finish : ℕ, 
      start < finish ∧ 
      finish ≤ chain.links ∧
      (∀ cut ∈ cuts, cut.position < start ∨ cut.position ≥ finish) ∧
      piece = finish - start)

/-- The main theorem stating the minimum number of cuts required -/
theorem min_cuts_for_chain (chain : Chain) (h : chain.links = 60) :
  ∃ cuts : List Cut, 
    cuts.length = 3 ∧ 
    canFormAllWeights chain cuts 60 ∧
    ∀ cuts' : List Cut, canFormAllWeights chain cuts' 60 → cuts'.length ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cuts_for_chain_l373_37318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_first_five_terms_sum_l373_37341

open BigOperators

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  ∑ i in Finset.range n, a * r ^ i = a * (1 - r^n) / (1 - r) :=
sorry

theorem first_five_terms_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  ∑ i in Finset.range n, a * r^i = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_first_five_terms_sum_l373_37341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_equidistant_from_boston_l373_37351

/-- Represents the distance between two cities -/
structure Distance where
  value : ℝ
  positive : value > 0

/-- Represents the speed of a train -/
structure Speed where
  value : ℝ
  positive : value > 0

theorem trains_meet_equidistant_from_boston 
  (d : Distance) -- distance between Boston and New York
  (v : Speed) -- speed of both trains
  : 
  let t := (d.value + v.value) / (2 * v.value) -- time when trains meet
  let distance_from_boston := (d.value + v.value) / 2 -- distance of both trains from Boston when they meet
  (v.value * t = distance_from_boston) ∧ 
  (d.value - (v.value * (t - 1)) = distance_from_boston) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_equidistant_from_boston_l373_37351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_from_volume_l373_37389

/-- The volume of a region in 3D space within a fixed distance of a line segment. -/
noncomputable def volume_region (r : ℝ) (l : ℝ) : ℝ := 16 * Real.pi * l + (256 / 3) * Real.pi

/-- Theorem stating that if the volume of the region is 544π, the length of the line segment is 86/3. -/
theorem length_of_segment_from_volume (r l : ℝ) (hr : r = 4) (hv : volume_region r l = 544 * Real.pi) :
  l = 86 / 3 := by
  sorry

#check length_of_segment_from_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_from_volume_l373_37389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bvp_integral_equation_equivalence_l373_37375

/-- Green's function for the boundary value problem -/
noncomputable def G (x ξ : ℝ) : ℝ :=
  if x ≤ ξ then (1 - ξ) * x else (x - 1) * ξ

/-- The boundary value problem and its integral equation representation -/
theorem bvp_integral_equation_equivalence
  (f : ℝ → ℝ → ℝ) (y : ℝ → ℝ) :
  (∀ x, x ∈ Set.Icc 0 1 →
    (deriv^[2] y) x = f x (y x) ∧
    y 0 = 0 ∧ y 1 = 0) ↔
  (∀ x, x ∈ Set.Icc 0 1 →
    y x = ∫ ξ in Set.Icc 0 1, G x ξ * f ξ (y ξ)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bvp_integral_equation_equivalence_l373_37375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l373_37302

noncomputable def f (x : ℝ) := Real.sqrt (3 * x + 4) + Real.sqrt (3 - 4 * x)

theorem f_extrema :
  let domain := {x : ℝ | -4/3 ≤ x ∧ x ≤ 3/4}
  (∀ x ∈ domain, f x ≤ 5 * Real.sqrt 21 / 6) ∧
  (∀ x ∈ domain, f x ≥ 5/2) ∧
  (∃ x ∈ domain, f x = 5 * Real.sqrt 21 / 6) ∧
  (∃ x ∈ domain, f x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l373_37302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_fraction_l373_37336

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I * (Complex.ofReal 0) = (Complex.ofReal 2 + Complex.I * Complex.ofReal a) / (1 + Complex.I)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_complex_fraction_l373_37336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stern_brocot_sequence_bijective_l373_37360

-- Define the sequence n_k
def n : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | k + 2 => if k % 2 = 0 then n k + n (k - 1) else n (k / 2)

-- Define the sequence q_k
def q (k : ℕ) : ℚ :=
  if k = 0 then 1 else (n k : ℚ) / (n (k - 1) : ℚ)

-- Statement to prove
theorem stern_brocot_sequence_bijective :
  ∀ (r : ℚ), r > 0 → ∃! (k : ℕ), q k = r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stern_brocot_sequence_bijective_l373_37360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_race_minimum_time_l373_37312

/-- Represents the boat race scenario -/
structure BoatRace where
  riverWidth : ℝ
  boatWidth : ℝ
  minSpace : ℝ
  currentSpeed : ℝ
  boatSpeed : ℝ
  raceDistance : ℝ

/-- Calculates the minimum time for all boats to finish the race -/
noncomputable def minimumRaceTime (race : BoatRace) : ℝ :=
  let spacePerBoat := race.boatWidth + 2 * race.minSpace
  let numBoats := ⌊race.riverWidth / spacePerBoat⌋
  let totalSpeed := race.boatSpeed + race.currentSpeed
  race.raceDistance / totalSpeed * 60  -- Convert to minutes

/-- Theorem stating the minimum race time for the given conditions -/
theorem boat_race_minimum_time :
  let race : BoatRace := {
    riverWidth := 42,
    boatWidth := 3,
    minSpace := 2,
    currentSpeed := 2,
    boatSpeed := 6,
    raceDistance := 1
  }
  minimumRaceTime race = 7.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_race_minimum_time_l373_37312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_minus_alpha_eq_pi_over_four_l373_37385

noncomputable def a (α : Real) : Real × Real := (Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def b (β : Real) : Real × Real := (2 * Real.cos β, 2 * Real.sin β)

theorem beta_minus_alpha_eq_pi_over_four (α β : Real)
  (h1 : π / 6 ≤ α)
  (h2 : α < π / 2)
  (h3 : π / 2 < β)
  (h4 : β ≤ 5 * π / 6)
  (h5 : (a α).1 * ((b β).1 - (a α).1) + (a α).2 * ((b β).2 - (a α).2) = 0) :
  β - α = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_minus_alpha_eq_pi_over_four_l373_37385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l373_37361

theorem min_sum_squares (a b : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * a^(6-k) * b^k = 160) → 
  (∀ c d : ℝ, (∃ k : ℕ, (Nat.choose 6 k) * c^(6-k) * d^k = 160) → a^2 + b^2 ≤ c^2 + d^2) ∧
  a^2 + b^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l373_37361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_6_74_l373_37381

/-- Represents the financial details of a bicycle transaction -/
structure BicycleTransaction where
  purchasePrice : ℚ
  sellingPrice : ℚ
  additionalCosts : ℚ

/-- Calculates the overall loss percentage for a set of bicycle transactions -/
def calculateLossPercentage (transactions : List BicycleTransaction) : ℚ :=
  let totalCost := transactions.foldl (fun acc t => acc + t.purchasePrice + t.additionalCosts) 0
  let totalSale := transactions.foldl (fun acc t => acc + t.sellingPrice) 0
  let loss := totalCost - totalSale
  (loss * 100) / totalCost

/-- The set of bicycle transactions -/
def bicycleTransactions : List BicycleTransaction := [
  { purchasePrice := 900, sellingPrice := 1100, additionalCosts := 0 },
  { purchasePrice := 1200, sellingPrice := 1400, additionalCosts := 0 },
  { purchasePrice := 1700, sellingPrice := 1600, additionalCosts := 0 },
  { purchasePrice := 1500, sellingPrice := 1900, additionalCosts := 200 },
  { purchasePrice := 2100, sellingPrice := 2300, additionalCosts := 300 }
]

/-- Theorem stating that the loss percentage is approximately 6.74% -/
theorem loss_percentage_approx_6_74 :
  ∃ ε > 0, |calculateLossPercentage bicycleTransactions - (674 : ℚ) / 100| < ε ∧ ε < (1 : ℚ) / 100 := by
  sorry

#eval calculateLossPercentage bicycleTransactions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_6_74_l373_37381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l373_37342

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define point M
def M : ℝ × ℝ := (2, 3)

-- Define line l
def l (x : ℝ) : Prop := x = -1

-- Define the distance function between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the distance from a point to the line x = -1
def distanceToLine (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Theorem statement
theorem min_distance_sum :
  ∀ P : ℝ × ℝ, parabola P →
  ∃ minDist : ℝ, minDist = Real.sqrt 10 ∧
  ∀ Q : ℝ × ℝ, parabola Q →
  distance Q M + distanceToLine Q ≥ minDist := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l373_37342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_formula_l373_37377

def seq (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 / 2) * seq n + 1

theorem seq_formula (n : ℕ) : 
  seq n = 2 - (1 / 2) ^ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_formula_l373_37377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_equals_one_l373_37380

/-- An arithmetic sequence {a_n} with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) : ℕ → ℝ := λ n ↦ a + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem a_2_equals_one 
  (a d : ℝ) 
  (h : arithmetic_sum a d 4 = arithmetic_sequence a d 4 + 3) : 
  arithmetic_sequence a d 2 = 1 := by
  sorry

#check a_2_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2_equals_one_l373_37380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_and_polar_line_l373_37305

-- Define the parametric equations of lines l₁ and l₂
noncomputable def l₁ (t k : ℝ) : ℝ × ℝ := (2 + t, k * t)
noncomputable def l₂ (m k : ℝ) : ℝ × ℝ := (-2 + m, m / k)

-- Define the polar equation of line l₃
def l₃ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) - Real.sqrt 2 = 0

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Theorem statement
theorem intersection_curve_and_polar_line :
  -- The general equation of C
  (∀ x y : ℝ, (∃ k t m : ℝ, l₁ t k = (x, y) ∧ l₂ m k = (x, y)) ↔ C x y) ∧
  -- The polar radius of intersection point M
  (∃ x y ρ θ : ℝ, C x y ∧ l₃ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ = Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_and_polar_line_l373_37305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_zero_eq_four_monotone_increasing_interval_l373_37331

noncomputable def f (x : ℝ) : ℝ := 3/4 * (x - 2)^2 + 1

theorem f_monotone_increasing : 
  ∀ x y, 0 ≤ x ∧ x < y ∧ y < 1 → f x < f y :=
by sorry

theorem f_zero_eq_four : f 0 = 4 :=
by sorry

theorem monotone_increasing_interval : 
  ∃ a b, a = 0 ∧ b = 1 ∧ ∀ x y, a ≤ x ∧ x < y ∧ y < b → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_zero_eq_four_monotone_increasing_interval_l373_37331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l373_37310

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem phi_value 
  (ω φ : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ) 
  (h3 : φ < Real.pi / 2) 
  (h4 : f ω φ 0 = - f ω φ (Real.pi / 2)) 
  (h5 : ∀ x, f ω φ (x + Real.pi / 12) = - f ω φ (-x - Real.pi / 12)) : 
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l373_37310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_points_l373_37355

noncomputable def points : List (ℝ × ℝ) := [(0, -7), (2, -3), (-4, 4), (7, 0), (1, 3)]

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem farthest_points : 
  (∀ p ∈ points, distance_from_origin p ≤ distance_from_origin (0, -7)) ∧
  (distance_from_origin (0, -7) = distance_from_origin (7, 0)) ∧
  (∀ q ∈ points, distance_from_origin q = distance_from_origin (0, -7) → q = (0, -7) ∨ q = (7, 0)) :=
by sorry

#check farthest_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_points_l373_37355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l373_37304

def A : Set ℤ := {x : ℤ | -2 < x ∧ x < 1}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l373_37304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_minimum_value_of_f_l373_37345

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / (2 * x) + (3 / 2) * x + 1

-- Theorem 1: If the tangent line at (1, f(1)) is perpendicular to the y-axis, then a = -1
theorem tangent_perpendicular_implies_a_value (a : ℝ) :
  (deriv (f a)) 1 = 0 → a = -1 := by sorry

-- Theorem 2: The minimum value of f(x) with a = -1 is 3, occurring at x = 1
theorem minimum_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f (-1) x = 3 ∧ ∀ (y : ℝ), y > 0 → f (-1) y ≥ 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_minimum_value_of_f_l373_37345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_x12_minus_1_l373_37388

theorem max_factors_x12_minus_1 : ∃ (m : ℕ),
  (∀ (q : List (MvPolynomial ℕ ℝ)), 
    (∀ p ∈ q, MvPolynomial.totalDegree p > 0) →
    (q.prod = MvPolynomial.X 0 ^ 12 - 1) →
    (q.length ≤ m)) ∧
  (∃ (q : List (MvPolynomial ℕ ℝ)),
    (∀ p ∈ q, MvPolynomial.totalDegree p > 0) ∧
    (q.prod = MvPolynomial.X 0 ^ 12 - 1) ∧
    (q.length = m)) ∧
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_x12_minus_1_l373_37388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_intersection_area_l373_37315

/-- The area of the rhombus formed by the intersection of two rectangles -/
noncomputable def rhombusArea (a b α : ℝ) : ℝ :=
  min a b / Real.sin α

theorem rectangle_intersection_area (a b α : ℝ) 
  (ha : a > 0) (hb : b > 0) (hw : α ≠ π / 2) (hα : 0 < α ∧ α < π) :
  let A := rhombusArea a b α
  ∃ (d₁ d₂ : ℝ), d₁ = 1 ∧ 
                 d₂ = 2 * min a b / Real.sin α ∧
                 A = d₁ * d₂ / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_intersection_area_l373_37315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l373_37311

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem probability_calculation :
  let P (A : Finset ℕ) := (A.card : ℚ) / S.card
  (P (S.filter (λ x => x > 3)) = 4/7) ∧
  (P (S.filter (λ x => x % 3 = 0)) = 2/7) ∧
  (P (S.filter (λ x => x > 3 ∨ x % 3 = 0)) = 5/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l373_37311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_function_properties_l373_37316

-- Definition of average value function
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

-- Definition of mean value point
def is_mean_value_point (f : ℝ → ℝ) (a b x₀ : ℝ) : Prop :=
  a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

-- Theorem statement
theorem average_value_function_properties :
  (is_average_value_function (λ x => Real.sin x - 1) (-Real.pi) Real.pi) ∧ 
  (∃ f : ℝ → ℝ, ∃ a b x₀ : ℝ, is_average_value_function f a b ∧ 
    is_mean_value_point f a b x₀ ∧ x₀ > (a + b) / 2) ∧
  (∀ m : ℝ, is_average_value_function (λ x => x^2 + m*x - 1) (-1) 1 → 
    -2 < m ∧ m < 0) ∧
  (∀ a b : ℝ, 1 ≤ a ∧ a < b → 
    ∀ x₀ : ℝ, is_mean_value_point Real.log a b x₀ → 
      Real.log x₀ < 1 / Real.sqrt (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_function_properties_l373_37316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l373_37346

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define the domain of x
def domain : Set ℝ := Set.Icc (-2) 1

-- Theorem statement
theorem f_range : ∀ x ∈ domain, -4 ≤ f x ∧ f x ≤ 0 := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l373_37346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_winning_strategy_l373_37354

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- Represents a cell on the board -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents the game state -/
structure GameState :=
  (board_size : Nat)
  (colored_cells : List Cell)
  (current_player : Player)

/-- Calculates the points earned for coloring a cell -/
def points_for_cell (state : GameState) (cell : Cell) : Nat :=
  sorry

/-- Applies the symmetry strategy for Bob -/
def symmetry_move (cell : Cell) (board_size : Nat) : Cell :=
  { row := cell.row, col := board_size - cell.col + 1 }

/-- Represents the result of a game -/
structure GameResult :=
  (final_score_diff : Int)

/-- Determines if a strategy wins with a given difference -/
def wins_with_difference (strategy : GameState → Cell) (diff : Int) : Prop :=
  sorry

/-- Theorem: Bob can guarantee a win with a maximum point difference of 2040200 -/
theorem bob_winning_strategy (initial_state : GameState) 
  (h1 : initial_state.board_size = 2020)
  (h2 : initial_state.current_player = Player.Alice)
  (h3 : initial_state.colored_cells = []) :
  ∃ (final_score_diff : Int),
    final_score_diff = 2040200 ∧
    ∀ (alice_strategy : GameState → Cell),
      ∃ (bob_strategy : GameState → Cell),
        wins_with_difference bob_strategy final_score_diff :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_winning_strategy_l373_37354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l373_37399

noncomputable def α : ℝ := -1920 * (Real.pi / 180)

theorem angle_properties :
  ∃ (β : ℝ) (k : ℤ),
    α = β + 2 * k * Real.pi ∧
    0 ≤ β ∧ β < 2 * Real.pi ∧
    Real.pi < β ∧ β < 3 * Real.pi / 2 ∧
    ∃ (θ₁ θ₂ : ℝ),
      θ₁ = -2 * Real.pi / 3 ∧
      θ₂ = -8 * Real.pi / 3 ∧
      -4 * Real.pi ≤ θ₁ ∧ θ₁ < 0 ∧
      -4 * Real.pi ≤ θ₂ ∧ θ₂ < 0 ∧
      ∃ (k₁ k₂ : ℤ),
        θ₁ = 2 * k₁ * Real.pi + β - 2 * k * Real.pi ∧
        θ₂ = 2 * k₂ * Real.pi + β - 2 * k * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l373_37399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_tan_l373_37306

noncomputable def f (x : ℝ) : ℝ := Real.tan (4 * x + Real.pi / 3)

theorem min_positive_period_of_tan (x : ℝ) : 
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_tan_l373_37306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_arithmetic_sequence_l373_37362

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1)) / 2

theorem max_sum_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ) (h1 : arithmetic_sequence a d) (h2 : d < 0)
  (h3 : sum_of_arithmetic_sequence a 6 = 5 * a 1 + 10 * d) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ k, sum_of_arithmetic_sequence a k ≤ sum_of_arithmetic_sequence a n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_arithmetic_sequence_l373_37362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_maximum_l373_37334

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) (m : ℕ) : ℤ := -n^2 + 4*n + m

/-- The value of n that maximizes S -/
def n_max : ℕ := 2

theorem S_maximum (m : ℕ) :
  ∀ n : ℕ, S n m ≤ S n_max m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_maximum_l373_37334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l373_37352

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2 + 2*x) →  -- definition of f for x ≥ 0
  f (3 - a^2) > f (2*a) →  -- given inequality
  -3 < a ∧ a < 1 :=  -- conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l373_37352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l373_37386

-- Define the triangle ABC
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Given conditions
  S = (1/4) * (a^2 + b^2 - c^2) →
  b = 2 →
  c = Real.sqrt 6 →
  -- Theorem statements
  C = 45 * (π / 180) ∧ 
  Real.cos B = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l373_37386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_shoveled_ten_driveways_l373_37393

/-- The number of driveways Jimmy shoveled -/
def driveways_shoveled : ℕ := 10

/-- The cost of each candy bar in dollars -/
def candy_bar_cost : ℚ := 3/4

/-- The cost of each lollipop in dollars -/
def lollipop_cost : ℚ := 1/4

/-- The number of candy bars Jimmy bought -/
def candy_bars_bought : ℕ := 2

/-- The number of lollipops Jimmy bought -/
def lollipops_bought : ℕ := 4

/-- The fraction of snow shoveling earnings spent on candy -/
def fraction_spent : ℚ := 1/6

/-- The cost per driveway in dollars -/
def cost_per_driveway : ℚ := 3/2

theorem jimmy_shoveled_ten_driveways :
  let total_spent := candy_bar_cost * candy_bars_bought + lollipop_cost * lollipops_bought
  let total_earned := total_spent / fraction_spent
  driveways_shoveled = (total_earned / cost_per_driveway).floor :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_shoveled_ten_driveways_l373_37393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_w_l373_37326

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_w_l373_37326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_problem_l373_37392

-- Define the circle C: x² + y² = 9
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

-- Define the line l1: 2x + 3y = 0
def l1 (x y : ℝ) : Prop := 2*x + 3*y = 0

-- Define the line l2: x - 2y = 0
def l2 (x y : ℝ) : Prop := x - 2*y = 0

theorem line_equation_problem :
  -- 1. Line through (2, 1) parallel to l1
  (∃ c : ℝ, ∀ x y : ℝ, 2*x + 3*y + c = 0 ↔ (x = 2 ∧ y = 1 ∨ (∃ t : ℝ, x = 2 + 3*t ∧ y = 1 - 2*t))) ∧
  -- 2. Line tangent to C and perpendicular to l2
  (∃ b : ℝ, (∀ x y : ℝ, y = -2*x + b → (x, y) ∈ C → (∃ t : ℝ, x = t ∧ y = 2*t)) ∧
            (∀ x y : ℝ, y = -2*x + b → l2 x y → x = y)) ∧
  -- 3. Line through (3, 2) with equal intercepts
  (∃ k : ℝ, k ≠ 0 ∧
    (∀ x y : ℝ, y - 2 = k*(x - 3) →
      (x = 0 → y = -3*k + 2) ∧
      (y = 0 → x = 3 - 2/k) ∧
      -3*k + 2 = 3 - 2/k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_problem_l373_37392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_of_sequence_l373_37356

def my_sequence (n : ℕ) : ℚ := n / (1 + 2 * n)

theorem hundredth_term_of_sequence :
  my_sequence 100 = 100 / 201 := by
  -- Unfold the definition of my_sequence
  unfold my_sequence
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_of_sequence_l373_37356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_upper_bound_l373_37332

noncomputable def u : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | n + 1 => u n + 1 / u n

theorem u_upper_bound : ∀ n : ℕ, n ≥ 1 → u n ≤ (3 * Real.sqrt n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_upper_bound_l373_37332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_abc_given_abp_l373_37321

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point P on the plane of triangle ABC
variable (P : ℝ × ℝ)

-- Define vector u
variable (u : ℝ × ℝ)

-- Define scalar m
variable (m : ℝ)

-- Define the condition 3PA + 4⃗uP = m⃗B
def condition (A B C P u : ℝ × ℝ) (m : ℝ) : Prop :=
  3 • (A -ᵥ P) + 4 • u = m • B

-- Define the area of a triangle
noncomputable def triangle_area (X Y Z : ℝ × ℝ) : ℝ := 
  abs ((X.1 - Z.1) * (Y.2 - Z.2) - (Y.1 - Z.1) * (X.2 - Z.2)) / 2

-- Theorem statement
theorem area_abc_given_abp 
  (h1 : condition A B C P u m)
  (h2 : m > 0)
  (h3 : triangle_area A B P = 8) :
  triangle_area A B C = 14 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_abc_given_abp_l373_37321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_prob_concentric_circles_l373_37353

/-- The probability that a random chord on the outer circle intersects the inner circle -/
noncomputable def chord_intersection_probability (inner_radius outer_radius : ℝ) : ℝ :=
  1 / 3

/-- Theorem stating that the probability of a random chord on a circle of radius 4
    intersecting a concentric circle of radius 2 is 1/3 -/
theorem chord_intersection_prob_concentric_circles :
  chord_intersection_probability 2 4 = 1 / 3 := by
  sorry

#check chord_intersection_prob_concentric_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_prob_concentric_circles_l373_37353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l373_37324

/-- The amount of oranges used for juice given total production and export percentage -/
noncomputable def oranges_for_juice (total_production : ℝ) (export_percentage : ℝ) : ℝ :=
  total_production * (1 - export_percentage / 100) * 0.6

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem orange_juice_production : 
  round_to_tenth (oranges_for_juice 7.2 30) = 3.0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_juice_production_l373_37324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spanning_tree_with_many_leaves_l373_37365

/-- A graph G is a pair (V, E) where V is the set of vertices and E is the set of edges. -/
structure Graph (V : Type*) where
  edges : Set (V × V)

/-- A graph is connected if there is a path between any two vertices. -/
def Connected {V : Type*} (G : Graph V) : Prop := sorry

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def Degree {V : Type*} (G : Graph V) (v : V) : ℕ := sorry

/-- A spanning tree of a graph G is a tree that includes all vertices of G. -/
def SpanningTree {V : Type*} (G : Graph V) (T : Graph V) : Prop := sorry

/-- The number of leaves in a tree is the number of vertices with degree 1. -/
def NumLeaves {V : Type*} (T : Graph V) : ℕ := sorry

/-- The main theorem: For any connected graph G with n vertices, where all vertices
    have degree at least three, there exists a spanning tree of G with more than 2/9 * n leaves. -/
theorem spanning_tree_with_many_leaves
  {V : Type*} [Fintype V] (G : Graph V) (n : ℕ) (h_connected : Connected G)
  (h_num_vertices : Fintype.card V = n)
  (h_min_degree : ∀ v : V, Degree G v ≥ 3) :
  ∃ T : Graph V, SpanningTree G T ∧ NumLeaves T > 2 * n / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spanning_tree_with_many_leaves_l373_37365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_carbon_count_l373_37327

/-- Represents the number of atoms of a given element in a compound -/
def AtomCount : Type := ℕ

/-- Represents the atomic mass of an element in atomic mass units (amu) -/
def AtomicMass : Type := ℕ

/-- Represents a chemical compound -/
structure Compound where
  c_count : AtomCount
  h_count : AtomCount
  o_count : AtomCount
  molecular_weight : AtomicMass

/-- The atomic masses of carbon, hydrogen, and oxygen -/
def carbon_mass : AtomicMass := (12 : ℕ)
def hydrogen_mass : AtomicMass := (1 : ℕ)
def oxygen_mass : AtomicMass := (16 : ℕ)

/-- The theorem stating that a compound with 1 H, 1 O, and molecular weight 65 amu has 4 C atoms -/
theorem compound_carbon_count 
  (comp : Compound) 
  (h1 : comp.h_count = (1 : ℕ)) 
  (h2 : comp.o_count = (1 : ℕ)) 
  (h3 : comp.molecular_weight = (65 : ℕ)) : 
  comp.c_count = (4 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_carbon_count_l373_37327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l373_37350

noncomputable def f (x : ℝ) := 1 / (x - 3) + Real.sqrt (x - 2)

theorem f_domain : 
  {x : ℝ | x ≥ 2 ∧ x ≠ 3} = {x : ℝ | x ≥ 2 ∧ x ≠ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l373_37350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sqrt_curve_value_l373_37337

/-- The area of the figure bounded by x=1, x=2, y=√x, and the x-axis -/
noncomputable def area_under_sqrt_curve : ℝ := ∫ x in (1:ℝ)..2, Real.sqrt x

/-- The theorem stating that the area under the sqrt curve is (4√2 - 2) / 3 -/
theorem area_under_sqrt_curve_value : 
  area_under_sqrt_curve = (4 * Real.sqrt 2 - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sqrt_curve_value_l373_37337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l373_37397

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Equation of symmetry axis
  (∃ (k : ℤ → ℝ), ∀ (n : ℤ), k n = Real.pi / 6 + n * Real.pi / 2 ∧
    ∀ (x : ℝ), f (k n + x) = f (k n - x)) ∧
  -- Range on the given interval
  (∀ (x : ℝ), -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 →
    -1 / 2 ≤ f x ∧ f x ≤ 1) ∧
  (∃ (x₁ x₂ : ℝ), -Real.pi / 12 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧
    -Real.pi / 12 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧
    f x₁ = -1 / 2 ∧ f x₂ = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l373_37397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_unique_solution_l373_37384

-- Define the set of positive real numbers
def S : Set ℝ := {x : ℝ | x > 0}

-- Define the properties of function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 1 = 0) ∧ 
  (∀ x y, x ∈ S → y ∈ S → f (1 / x + 1 / y) = f x + f y) ∧
  (∀ x y, x ∈ S → y ∈ S → Real.log (x + y) * f (x + y) = Real.log x * f x + Real.log y * f y)

-- Theorem statement
theorem unique_function (f : ℝ → ℝ) (h : satisfies_conditions f) : 
  ∀ x, x ∈ S → f x = 0 := by
  sorry

-- Corollary: There is only one function satisfying the conditions
theorem unique_solution : ∃! f : ℝ → ℝ, satisfies_conditions f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_unique_solution_l373_37384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathcounts_magnet_combinations_l373_37348

def word : Finset Char := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
def vowels : Finset Char := {'A', 'O', 'U'}
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'N', 'S'}

def indistinguishable_t (s : Finset Char) : Nat :=
  if s.card = 2 ∧ s ⊆ consonants then
    if 'T' ∈ s then 1 else 0
  else 0

theorem mathcounts_magnet_combinations :
  (Finset.filter (fun s => s.card = 5 ∧ 
    (s ∩ vowels).card = 2 ∧ 
    (s ∩ consonants).card = 3 ∧ 
    s ⊆ word) (Finset.powerset word)).card = 75 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathcounts_magnet_combinations_l373_37348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_C_grades_l373_37307

def gradeC (score : ℕ) : Bool := 75 ≤ score ∧ score ≤ 84

def scores : List ℕ := [89, 72, 54, 97, 77, 92, 85, 74, 75, 63, 84, 78, 71, 80, 90]

theorem percentage_of_C_grades : 
  (((scores.filter gradeC).length : ℚ) / (scores.length : ℚ)) * 100 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_C_grades_l373_37307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l373_37340

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1) ∧
  (∀ x y : ℝ, y^2/25 - x^2/M = 1) ∧
  (∀ x : ℝ, ∃ y : ℝ, y = (4/3)*x ∨ y = -(4/3)*x) ∧
  (∀ x : ℝ, ∃ y : ℝ, y = (5/Real.sqrt M)*x ∨ y = -(5/Real.sqrt M)*x) →
  M = 144/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l373_37340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l373_37329

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (6 - x ^ (1/3)))

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (0 ≤ x ∧ x ≤ 216) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l373_37329
