import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainder_pigeonhole_l780_78032

theorem square_remainder_pigeonhole (S : Finset ℤ) (h : S.card = 51) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x^2 ≡ y^2 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainder_pigeonhole_l780_78032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l780_78049

/-- Definition of an ellipse with foci and a point satisfying a specific condition -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  F1 : ℝ × ℝ  -- Left focus
  F2 : ℝ × ℝ  -- Right focus
  A : ℝ × ℝ   -- Point on the ellipse
  h3 : A.fst^2 / a^2 + A.snd^2 / b^2 = 1  -- A is on the ellipse
  h4 : 2 * dist A F1 - 3 * dist A F2 = a  -- Given condition

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem stating the range of eccentricity -/
theorem eccentricity_range (e : Ellipse) : 
  2/5 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l780_78049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l780_78043

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt ((2 - x) / x) ∧ x > 0 ∧ x ≤ 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l780_78043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l780_78017

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

-- Define the foci of ellipse C
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define point M on the ellipse
noncomputable def M : ℝ × ℝ := (Real.sqrt 2, 1)

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem ellipse_properties :
  -- (I) The standard equation of ellipse C
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  -- (II) Line AD always passes through (0, 2)
  (∀ A B D : ℝ × ℝ,
    -- A and B are on ellipse C
    ellipse_C A.1 A.2 → ellipse_C B.1 B.2 →
    -- B is symmetric to D with respect to y-axis
    B.1 = -D.1 ∧ B.2 = D.2 →
    -- D is different from A
    D ≠ A →
    -- Line AB passes through P
    (∃ t : ℝ, (1 - t) • P.1 + t • A.1 = B.1 ∧ (1 - t) • P.2 + t • A.2 = B.2) →
    -- Line AD passes through (0, 2)
    ∃ s : ℝ, (1 - s) • A.1 + s • D.1 = 0 ∧ (1 - s) • A.2 + s • D.2 = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l780_78017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l780_78042

theorem max_value_theorem (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 4 → 
    (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≥ (2*a - b)^2 + (2*b - c)^2 + (2*c - a)^2) ∧
  (∃ p q r : ℝ, p^2 + q^2 + r^2 = 4 ∧
    (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 = (2*p - q)^2 + (2*q - r)^2 + (2*r - p)^2) ∧
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 = 28 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l780_78042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_floor_sum_diverges_l780_78095

-- Define the Riemann zeta function
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (n : ℝ) ^ (-x)

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem zeta_floor_sum_diverges :
  ¬ ∃ (S : ℝ), ∑' k : ℕ, (floor (zeta (2 * ↑k))) = S := by
  sorry

#check zeta_floor_sum_diverges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeta_floor_sum_diverges_l780_78095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_power_l780_78008

theorem not_perfect_power (m : ℕ) : ¬∃ (n : ℕ) (k : ℕ), m * (m + 1) = n ^ k ∧ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_power_l780_78008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l780_78025

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

-- State the theorem
theorem solve_for_a (a : ℝ) : f a + f 1 = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l780_78025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l780_78033

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem f_solutions :
  ∀ x : ℝ, f x = 6 ↔ x = -1/2 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solutions_l780_78033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_BO_to_BD_l780_78071

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the midpoint M and intersection point O
noncomputable def M (rect : Rectangle) : ℝ × ℝ := ((rect.B.1 + rect.C.1) / 2, (rect.B.2 + rect.C.2) / 2)
noncomputable def O (rect : Rectangle) : ℝ × ℝ := (0, rect.A.2)

-- Define the lengths
noncomputable def AB (rect : Rectangle) : ℝ := rect.B.1 - rect.A.1
noncomputable def AD (rect : Rectangle) : ℝ := rect.A.2 - rect.D.2
noncomputable def BD (rect : Rectangle) : ℝ := Real.sqrt ((rect.B.1 - rect.D.1)^2 + (rect.B.2 - rect.D.2)^2)
noncomputable def BO (rect : Rectangle) : ℝ := Real.sqrt ((rect.B.1 - (O rect).1)^2 + (rect.B.2 - (O rect).2)^2)

-- Define the theorem
theorem ratio_BO_to_BD (rect : Rectangle) 
  (h1 : AB rect = 6) 
  (h2 : AD rect = 4) 
  (h3 : rect.A = (0, 4)) 
  (h4 : rect.B = (6, 4)) 
  (h5 : rect.C = (6, 0)) 
  (h6 : rect.D = (0, 0)) :
  BO rect / BD rect = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_BO_to_BD_l780_78071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l780_78016

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1/4) :
  (a = 1/2) ∧ (∀ y : ℝ, y ∈ Set.range (f a) → 0 < y ∧ y ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l780_78016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l780_78014

noncomputable def circle_radius : ℝ := 100 * Real.sqrt 5

-- Define the quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)
  (inscribed : a ≤ 2 * circle_radius ∧ b ≤ 2 * circle_radius ∧ c ≤ 2 * circle_radius ∧ d ≤ 2 * circle_radius)
  (three_equal_sides : a = 200 ∧ b = 200 ∧ c = 200)

-- Theorem statement
theorem fourth_side_length (q : Quadrilateral) : q.d = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l780_78014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l780_78072

theorem impossibility_of_arrangement : ¬ ∃ (arr : Matrix (Fin 4) (Fin 4) ℕ),
  (∀ i j, arr i j ∈ Finset.range 16 \ {0, 1}) ∧
  (∀ i, (Finset.sum (Finset.range 4) (λ j ↦ arr i j)) = (Finset.sum (Finset.range 16) (λ k ↦ k + 2)) / 4) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → ¬(arr i j₁ ∣ arr i j₂) ∧ ¬(arr i j₂ ∣ arr i j₁)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_arrangement_l780_78072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l780_78024

theorem power_equation : ∃ x : ℝ, 7776 * (1/6:ℝ)^x = 216 ∧ x = 2 := by
  use 2
  constructor
  · norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l780_78024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_f_values_l780_78007

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define constants for the trigonometric values
noncomputable def sin_50 : ℝ := Real.sin (50 * Real.pi / 180)
noncomputable def cos_50 : ℝ := Real.cos (50 * Real.pi / 180)
noncomputable def tan_50 : ℝ := Real.tan (50 * Real.pi / 180)

-- State the theorem
theorem order_of_f_values
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing : ∀ x y, 0 ≤ x → x < y → f x < f y)
  (h_trig_order : cos_50 < sin_50 ∧ sin_50 < tan_50) :
  f cos_50 < f sin_50 ∧ f sin_50 < f tan_50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_f_values_l780_78007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l780_78070

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the arithmetic sequence property
def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem ellipse_line_theorem (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ →
  is_on_ellipse 4 (9/5) →
  is_on_ellipse x₂ y₂ →
  arithmetic_sequence 
    (distance (x₁, y₁) right_focus)
    (distance (4, 9/5) right_focus)
    (distance (x₂, y₂) right_focus) →
  ∃ (t : ℝ), 
    (∀ x y, 25*x - 20*y = 64 ↔ 
      (y - 9/5) / (x - 4) = (y - 0) / (x - t) ∧
      t = (x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l780_78070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_neg_1125_deg_conversion_l780_78074

theorem angle_conversion (angle_deg : ℚ) : 
  ∃ (α : ℚ) (k : ℤ), angle_deg * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π :=
by
  -- Proof goes here
  sorry

theorem neg_1125_deg_conversion : 
  ∃ (α : ℚ) (k : ℤ), (-1125 : ℚ) * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π ∧ 
  α + 2 * k * π = 7 * π / 4 - 8 * π :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_neg_1125_deg_conversion_l780_78074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_solution_l780_78060

/-- The number of hours Liza and Suzie read -/
def h : ℝ := sorry

/-- Liza's reading rate in pages per hour -/
def liza_rate : ℝ := 20

/-- Suzie's reading rate in pages per hour -/
def suzie_rate : ℝ := 15

/-- The difference in pages read between Liza and Suzie -/
def page_difference : ℝ := 15

theorem reading_time_solution :
  h * liza_rate = h * suzie_rate + page_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_solution_l780_78060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_value_floor_100S_l780_78076

noncomputable def a (k : ℕ) : ℝ := 2^k

noncomputable def S : ℝ := ∑' k, Real.arccos ((2 * (a k)^2 - 6 * (a k) + 5) / 
  Real.sqrt ((a k)^2 - 4 * (a k) + 5) * (4 * (a k)^2 - 8 * (a k) + 5))

theorem S_value : S = π / 2 := by sorry

theorem floor_100S : ⌊100 * S⌋ = 157 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_value_floor_100S_l780_78076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_boats_l780_78052

def initial_boats : ℕ := 30
def fish_percentage : ℚ := 1/5
def arrow_losses : ℕ := 2
def wind_losses : ℕ := 3
def sink_losses : ℕ := 4

theorem remaining_boats :
  initial_boats - 
  (fish_percentage * initial_boats).floor - 
  arrow_losses - 
  wind_losses - 
  sink_losses = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_boats_l780_78052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_ratio_l780_78087

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let BC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let AC := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  let angleC := Real.arccos ((BC^2 + AC^2 - (t.A.1 - t.B.1)^2 - (t.A.2 - t.B.2)^2) / (2 * BC * AC))
  BC = 6 ∧ AC = 3 ∧ angleC = Real.pi/6

-- Define the orthocenter
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the point D where altitude AD intersects BC
noncomputable def pointD (t : Triangle) : ℝ × ℝ := sorry

-- Define the ratio AH:HD
noncomputable def ratioAHHD (t : Triangle) : ℝ :=
  let H := orthocenter t
  let D := pointD t
  let AH := Real.sqrt ((t.A.1 - H.1)^2 + (t.A.2 - H.2)^2)
  let HD := Real.sqrt ((H.1 - D.1)^2 + (H.2 - D.2)^2)
  AH / HD

-- Theorem statement
theorem orthocenter_ratio (t : Triangle) :
  isValidTriangle t → ratioAHHD t = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_ratio_l780_78087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_closed_open_interval_l780_78056

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {y | 2 ≤ y ∧ y ≤ 5}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- State the theorem
theorem intersection_equals_closed_open_interval : 
  A_intersect_B = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_closed_open_interval_l780_78056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l780_78045

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x) / (x + 2)

-- Define the domain
def domain (x : ℝ) : Prop := x ≤ 2 ∧ x ≠ -2

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, (x ∈ Set.Iic 2 \ {-2}) ↔ domain x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l780_78045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_angle_x_l780_78092

/-- Given that the terminal side of angle x passes through point P(-1, 3), 
    prove the following trigonometric identities. -/
theorem trig_identities_for_angle_x (x : ℝ) 
    (h : ∃ (r : ℝ), r > 0 ∧ r * (Real.sin x) = 3 ∧ r * (Real.cos x) = -1) : 
  (Real.sin x + Real.cos x = Real.sqrt 10 / 5) ∧ 
  ((Real.sin (π/2 + x) * Real.cos (π/2 - x)) / (Real.cos (-x) * Real.cos (π - x)) = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_for_angle_x_l780_78092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_segment_length_l780_78061

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the length of the diagonal of a rectangle -/
noncomputable def diagonal (r : Rectangle) : ℝ := Real.sqrt (r.a^2 + r.b^2)

/-- Calculates the sum of segments in the construction -/
noncomputable def segmentSum (r : Rectangle) (n : ℕ) : ℝ :=
  2 * (diagonal r) * (n * (n - 1) / 2) / n

/-- The main theorem stating the total length of segments -/
theorem total_segment_length (r : Rectangle) (h1 : r.a = 5) (h2 : r.b = 4) :
  segmentSum r 200 - diagonal r = 198 * Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_segment_length_l780_78061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_length_approx_l780_78075

/-- Represents the dimensions and properties of a rectangular swimming pool. -/
structure SwimmingPool where
  width : ℝ
  waterLowered : ℝ
  waterRemoved : ℝ
  gallonsPerCubicFoot : ℝ

/-- Calculates the length of a rectangular swimming pool given its properties. -/
noncomputable def calculatePoolLength (pool : SwimmingPool) : ℝ :=
  let volumeCubicFeet := pool.waterRemoved / pool.gallonsPerCubicFoot
  volumeCubicFeet / (pool.width * pool.waterLowered)

/-- Theorem stating that the calculated length of the swimming pool is approximately 50.1 feet. -/
theorem pool_length_approx (pool : SwimmingPool) 
    (h1 : pool.width = 25)
    (h2 : pool.waterLowered = 0.5)
    (h3 : pool.waterRemoved = 4687.5)
    (h4 : pool.gallonsPerCubicFoot = 7.48052) :
    ∃ ε > 0, |calculatePoolLength pool - 50.1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_length_approx_l780_78075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l780_78062

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : 1/m + 1/n = 2*m*n) : m * f n + n * f (-m) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l780_78062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_simplification_l780_78053

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem cos_sum_simplification :
  let x := Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)
  ω ^ 17 = 1 →
  x = (Real.sqrt 13 - 1) / 4 := by
  intro h
  sorry

#check cos_sum_simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_simplification_l780_78053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l780_78021

/-- The radius of the large circle D -/
def R : ℝ := 40

/-- The number of small circles in the ring -/
def n : ℕ := 8

/-- The area of the region inside the large circle and outside all small circles -/
noncomputable def M : ℝ := 
  Real.pi * (R^2 - n * (R / (1 + Real.sqrt 2))^2)

/-- The floor of M -/
noncomputable def M_floor : ℤ := ⌊M⌋

theorem area_difference : M_floor = -26175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l780_78021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_parallel_lines_l780_78039

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Point := ℝ × ℝ

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r = ((1 - t) • p.1 + t • q.1, (1 - t) • p.2 + t • q.2)}

noncomputable def Tangent (c : Circle) (p : Point) : Set Point :=
  sorry

def IntersectionPoints (c₁ c₂ : Circle) : Set Point :=
  sorry

def ParallelLines (l₁ l₂ : Set Point) : Prop :=
  sorry

theorem circles_tangents_parallel_lines 
  (ω₁ ω₂ : Circle) 
  (A C : Point) 
  (h_intersect : {A, C} ⊆ IntersectionPoints ω₁ ω₂) 
  (B : Point) 
  (h_B : B ∈ (Tangent ω₂ A ∩ {p : Point | (p.1 - ω₁.center.1)^2 + (p.2 - ω₁.center.2)^2 = ω₁.radius^2})) 
  (D : Point) 
  (h_D : D ∈ (Tangent ω₁ C ∩ {p : Point | (p.1 - ω₂.center.1)^2 + (p.2 - ω₂.center.2)^2 = ω₂.radius^2})) :
  ParallelLines (Line A D) (Line B C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_parallel_lines_l780_78039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_flooring_area_l780_78010

-- Define the dimensions of each section
def central_side : ℚ := 10
def hallway_length : ℚ := 6
def hallway_width : ℚ := 4
def l_shaped_length : ℚ := 5
def l_shaped_width : ℚ := 2
def triangle_side : ℚ := 3

-- Define the areas of each section
def central_area : ℚ := central_side * central_side
def hallway_area : ℚ := hallway_length * hallway_width
def l_shaped_area : ℚ := l_shaped_length * l_shaped_width
def triangle_area : ℚ := 1/2 * triangle_side * triangle_side

-- Define the total area
def total_area : ℚ := central_area + hallway_area + l_shaped_area + triangle_area

-- Theorem statement
theorem bathroom_flooring_area : total_area = 277/2 := by
  -- Unfold definitions
  unfold total_area central_area hallway_area l_shaped_area triangle_area
  unfold central_side hallway_length hallway_width l_shaped_length l_shaped_width triangle_side
  -- Simplify the expression
  simp [mul_comm, mul_assoc, add_comm, add_assoc]
  -- Check equality
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_flooring_area_l780_78010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_power_sum_divisibility_l780_78091

theorem coprime_power_sum_divisibility (a b m n : ℕ) : 
  a ≠ 0 → b ≠ 0 → a ≥ 2 → Nat.Coprime a b → 
  (a^m + b^m) ∣ (a^n + b^n) → m ∣ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_power_sum_divisibility_l780_78091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l780_78057

open Set
open Function
open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x > 0}

-- State the conditions
axiom f_domain : ∀ x ∈ dom_f, f x ≠ 0
axiom f_condition : ∀ x ∈ dom_f, f x + x * (deriv^[2] f x) > 0

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 1) * f (x^2 - 1) < f (x + 1)

-- State the theorem
theorem solution_set :
  {x : ℝ | inequality x} = Ioo 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l780_78057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tunnel_time_l780_78011

noncomputable def tunnel_time (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then
    3480 / x
  else if 12 < x ∧ x ≤ 25 then
    5 * x + 2880 / x + 10
  else
    0  -- undefined for other x values

def problem_conditions : Prop :=
  ∃ (num_cars : ℕ) (car_length tunnel_length max_speed : ℝ),
    num_cars = 31 ∧
    car_length = 5 ∧
    tunnel_length = 2725 ∧
    max_speed = 25 ∧
    ∀ x : ℝ, 0 < x → x ≤ max_speed →
      tunnel_time x = if x ≤ 12 then
                        (tunnel_length + car_length * num_cars + 20 * (num_cars - 1)) / x
                      else
                        (tunnel_length + car_length * num_cars + ((1/6) * x^2 + (1/3) * x) * (num_cars - 1)) / x

theorem min_tunnel_time :
  problem_conditions →
  ∃ (min_time : ℝ) (optimal_speed : ℝ),
    min_time = 250 ∧
    optimal_speed = 24 ∧
    ∀ x : ℝ, 0 < x → x ≤ 25 → tunnel_time x ≥ min_time :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tunnel_time_l780_78011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l780_78058

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) + f x * f y = x^2 * y^2 + 2 * x * y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l780_78058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l780_78082

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := λ x ↦ (2 : ℝ)^x + b

-- State the theorem
theorem inverse_function_point (b : ℝ) 
  (g : ℝ → ℝ) 
  (h_inv : Function.LeftInverse g (f b) ∧ Function.RightInverse g (f b)) 
  (h_point : g 5 = 2) : 
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l780_78082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l780_78078

noncomputable def point1 : ℝ × ℝ := (2, 3)
noncomputable def point2 : ℝ × ℝ := (-3, 1)
noncomputable def point3 : ℝ × ℝ := (4, -5)
noncomputable def point4 : ℝ × ℝ := (-6, 2)
noncomputable def point5 : ℝ × ℝ := (0, -7)

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem farthest_point :
  ∀ p ∈ ({point1, point2, point3, point4, point5} : Set (ℝ × ℝ)),
    distance_from_origin point3 ≥ distance_from_origin p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l780_78078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_4_20_l780_78023

noncomputable def hour_hand_speed : ℝ := 360 / 12
noncomputable def minute_hand_speed : ℝ := 360 / 60

noncomputable def hour_hand_position (hours : ℝ) (minutes : ℝ) : ℝ :=
  (hours * hour_hand_speed) + (minutes * hour_hand_speed / 60)

noncomputable def minute_hand_position (minutes : ℝ) : ℝ :=
  minutes * minute_hand_speed

noncomputable def angle_between_hands (hours : ℝ) (minutes : ℝ) : ℝ :=
  |hour_hand_position hours minutes - minute_hand_position minutes|

theorem angle_at_4_20 : angle_between_hands 4 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_4_20_l780_78023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l780_78002

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin (π/2 + α) = -3/5) :
  Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l780_78002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_task_is_stabilize_fertility_l780_78096

/-- Represents the change in population proportions and total population over 10 years -/
structure PopulationChange where
  youth_decrease : ℝ  -- Decrease in proportion of population aged 0-14
  elderly_increase : ℝ  -- Increase in proportion of population aged 65 and above
  total_increase : ℕ  -- Increase in total population

/-- Represents possible main tasks for population work -/
inductive PopulationTask
  | ControlMovement
  | StabilizeLowFertility
  | CurbAging
  | IncreaseYouth

/-- Given population changes, determines the main task for population work -/
def determineMainTask (change : PopulationChange) : PopulationTask := 
  PopulationTask.StabilizeLowFertility

/-- Theorem stating that given the observed population changes, 
    the main task is to stabilize low fertility -/
theorem main_task_is_stabilize_fertility 
  (change : PopulationChange)
  (h1 : change.youth_decrease = 4.8)
  (h2 : change.elderly_increase = 1.39)
  (h3 : change.total_increase = 130000000) :
  determineMainTask change = PopulationTask.StabilizeLowFertility := by
  sorry

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

notation x " ≈ " y => approx_equal x y 0.01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_task_is_stabilize_fertility_l780_78096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_theorem_l780_78029

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite -/
structure Kite where
  K : Point
  I : Point
  T : Point
  E : Point

/-- Calculate the area of a quadrilateral given its four vertices -/
noncomputable def area (A B C D : Point) : ℝ := sorry

/-- Calculate the distance between two points -/
noncomputable def distance (A B : Point) : ℝ := sorry

/-- Check if a line is perpendicular to another line -/
def isPerpendicular (A B C D : Point) : Prop := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M A B : Point) : Prop := sorry

/-- Main theorem -/
theorem kite_area_theorem (kite : Kite) (R A N M D : Point) : 
  isPerpendicular kite.I kite.E kite.K kite.T →
  R.x = 0 ∧ R.y = 0 →
  isMidpoint A kite.K kite.I →
  isMidpoint N kite.I kite.T →
  isMidpoint M kite.T kite.E →
  isMidpoint D kite.E kite.K →
  area M A kite.K kite.E = 18 →
  distance kite.I kite.T = 10 →
  area R A kite.I N = 4 →
  area D kite.I M kite.E = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_theorem_l780_78029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_is_zero_l780_78001

-- Define the matrices and vector
def R (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def S (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x^2, x*y, x*z],
    ![x*y, y^2, y*z],
    ![x*z, y*z, z^2]]

def u (x y z : ℝ) : Fin 3 → ℝ :=
  ![x, y, z]

-- State the theorem
theorem matrix_product_is_zero (d e f x y z : ℝ)
  (h1 : d*y = e*z)
  (h2 : d*x = f*z)
  (h3 : e*x = f*y) :
  R d e f • S x y z = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_is_zero_l780_78001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l780_78088

theorem right_triangle_sin_c (A B C : Real) : 
  -- Triangle ABC is a right triangle with A as the right angle
  Real.sin A = 1 →
  -- Given cos B
  Real.cos B = 3/5 →
  -- Prove that sin C equals 3/5
  Real.sin C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sin_c_l780_78088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_c_range_l780_78030

open Set
open Function
open Real

-- Define the function
noncomputable def f (c : ℝ) : ℝ → ℝ := λ x => Real.log (x^2 + 2*x - c)

-- Define the propositions
def prop_p (c : ℝ) : Prop := Set.range (f c) = Set.univ
def prop_q (c : ℝ) : Prop := Set.range (f c) = Set.univ

-- State the theorem
theorem log_function_c_range :
  (∀ c : ℝ, (prop_p c ∧ ¬prop_q c) ∨ (¬prop_p c ∧ prop_q c)) →
  {c : ℝ | c < -1} = {c : ℝ | ∃ x, (prop_p x ∧ ¬prop_q x) ∨ (¬prop_p x ∧ prop_q x)} :=
by
  sorry

#check log_function_c_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_c_range_l780_78030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l780_78037

/-- A triangle is defined by its three angles A, B, and C -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

/-- Condition for the inequality in the problem -/
def condition (t : Triangle) : Prop :=
  Real.sin (t.A - t.B) * Real.cos t.B + Real.cos (t.A - t.B) * Real.sin t.B ≥ 1

/-- A triangle is right-angled if one of its angles is π/2 -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

/-- The main theorem to be proved -/
theorem condition_sufficient_not_necessary :
  (∀ t : Triangle, condition t → is_right_triangle t) ∧
  ¬(∀ t : Triangle, is_right_triangle t → condition t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l780_78037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l780_78093

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m * x + b ∧ (0, 1) ∈ {(x, y) | y = m * x + b}

-- Define points A and B as intersections of the line and parabola
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_focus A.1 A.2 ∧ line_through_focus B.1 B.2 ∧
  A ≠ B

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1*(p2.2 - p3.2) + p2.1*(p3.2 - p1.2) + p3.1*(p1.2 - p2.2))

-- Theorem statement
theorem parabola_intersection_ratio (A B : ℝ × ℝ) :
  intersection_points A B →
  distance A focus = 3 →
  triangle_area A (0, 0) focus / triangle_area B (0, 0) focus = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l780_78093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_l780_78035

/-- Represents the time taken to fill a tank with two pipes -/
noncomputable def fill_time (time_a : ℝ) (time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: Two pipes that can fill a tank in 10 and 20 hours respectively
    will fill the tank in 20/3 hours when opened simultaneously -/
theorem simultaneous_fill_time :
  fill_time 10 20 = 20 / 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_l780_78035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perimeter_ellipse_triangle_perimeter_l780_78055

theorem ellipse_perimeter (a b c : ℝ) (h1 : a > 5) (h2 : b = 5) (h3 : c = 4) : 
  let perimeter := 4 * a
  perimeter = 4 * Real.sqrt 41 := by
  have h4 : a^2 = b^2 + c^2 := by
    -- Proof of a^2 = b^2 + c^2
    sorry
  have h5 : a = Real.sqrt 41 := by
    -- Proof that a = √41
    sorry
  -- Final calculation
  calc
    4 * a = 4 * Real.sqrt 41 := by rw [h5]

-- The main theorem
theorem ellipse_triangle_perimeter : 
  ∃ (a b c : ℝ), a > 5 ∧ b = 5 ∧ c = 4 ∧
  let perimeter := 4 * a
  perimeter = 4 * Real.sqrt 41 := by
  -- Existential introduction
  use Real.sqrt 41, 5, 4
  -- Prove the conjunction
  constructor
  · -- Prove a > 5
    sorry
  constructor
  · -- Prove b = 5
    rfl
  constructor
  · -- Prove c = 4
    rfl
  · -- Apply the previous theorem
    exact ellipse_perimeter (Real.sqrt 41) 5 4 sorry rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perimeter_ellipse_triangle_perimeter_l780_78055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_digit_divisible_by_eleven_l780_78064

theorem nineteen_digit_divisible_by_eleven : ∃ n : ℕ, 
  (10^18 ≤ n) ∧ (n < 10^19) ∧ 
  (∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0) ∧
  (n % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteen_digit_divisible_by_eleven_l780_78064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l780_78097

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

-- Theorem statement
theorem f_properties (a : ℝ) :
  (f a 1 = -3/10) →
  (∀ x, f a x = -f a (-x)) ∧
  (∀ x, -1/6 ≤ f a x ∧ f a x ≤ 0 ↔ 0 ≤ x ∧ x ≤ 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l780_78097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_special_triangle_l780_78051

noncomputable section

/-- The radius of the circumscribed circle of a triangle --/
def circumradius (a b : ℝ) (angle : ℝ) : ℝ :=
  b * a * Real.sin angle / (2 * Real.sin ((Real.pi / 2) - (angle / 2)))

/-- Theorem: The radius of the circle circumscribed around a triangle with sides 5 and 8
    and an angle of 60° between them is 7√3/3 --/
theorem circumradius_special_triangle :
  circumradius 5 8 (Real.pi / 3) = 7 * Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_special_triangle_l780_78051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l780_78065

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem f_min_max : 
  let a : ℝ := 3
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≥ 2/3) ∧ 
  (∀ x ∈ Set.Icc a b, f x ≤ 1) ∧
  (f b = 2/3) ∧ 
  (f a = 1) := by
  sorry

#check f_min_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l780_78065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_hour_study_score_l780_78044

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  /-- The score for 4 hours of study -/
  base_score : ℚ
  /-- The study time for the base score -/
  base_time : ℚ
  /-- The improvement factor from tutoring -/
  tutoring_improvement : ℚ

/-- Calculates the final score based on study time and tutoring -/
def final_score (r : StudyScoreRelation) (study_time : ℚ) : ℚ :=
  (r.base_score * study_time / r.base_time) * (1 + r.tutoring_improvement)

/-- Theorem stating the final score for 5 hours of study with tutoring -/
theorem five_hour_study_score (r : StudyScoreRelation) 
  (h1 : r.base_score = 80)
  (h2 : r.base_time = 4)
  (h3 : r.tutoring_improvement = 1/10) :
  final_score r 5 = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_hour_study_score_l780_78044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_equality_l780_78005

-- Define the angle in radians (since Lean typically works with radians)
noncomputable def theta : ℝ := 25 * Real.pi / 180

-- State the theorem
theorem acute_angle_equality : 
  (Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin theta + Real.cos theta) ∧ 
  (0 < theta) ∧ (theta < Real.pi / 2) → 
  theta = 25 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_equality_l780_78005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l780_78031

noncomputable section

-- Define the function pairs
def f_A (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)
def g_A (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

def f_B (x : ℝ) : ℝ := x^2
def g_B (x : ℝ) : ℝ := (x^2)^(1/3)

def f_C (x : ℝ) : ℝ := (x^2 - 1) / (x - 1)
def g_C (x : ℝ) : ℝ := x + 1

def f_D (x : ℝ) : ℝ := Real.sqrt (x^2)
def g_D (x : ℝ) : ℝ := (Real.sqrt x)^2

-- Define the domains of the functions
def domain_f_A : Set ℝ := {x | x > 1}
def domain_g_A : Set ℝ := {x | x > 1 ∨ x < -1}

def domain_f_C : Set ℝ := {x | x ≠ 1}
def domain_g_C : Set ℝ := Set.univ

def domain_f_D : Set ℝ := Set.univ
def domain_g_D : Set ℝ := {x | x ≥ 0}

-- Theorem statement
theorem function_equality :
  (∃ x, f_A x ≠ g_A x ∨ domain_f_A ≠ domain_g_A) ∧
  (∀ x, f_B x = g_B x) ∧
  (∃ x, f_C x ≠ g_C x ∨ domain_f_C ≠ domain_g_C) ∧
  (∃ x, f_D x ≠ g_D x ∨ domain_f_D ≠ domain_g_D) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l780_78031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covariance_operator_and_gaussian_process_l780_78041

/-- Continuous real covariance function on [a, b]² ⊂ ℝ² -/
def ContinuousCovariance (K : ℝ → ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous (fun p : ℝ × ℝ ↦ K p.1 p.2) ∧ 
  ∀ s t, a ≤ s ∧ s ≤ b ∧ a ≤ t ∧ t ≤ b → K s t = K t s

/-- Integral operator A defined by K -/
noncomputable def IntegralOperator (K : ℝ → ℝ → ℝ) (a b : ℝ) (f : ℝ → ℝ) (s : ℝ) : ℝ :=
  ∫ t in a..b, K s t * f t

/-- Eigenfunction property for K -/
def IsEigenfunction (K : ℝ → ℝ → ℝ) (a b : ℝ) (lambda : ℝ) (phi : ℝ → ℝ) : Prop :=
  ∀ s, a ≤ s ∧ s ≤ b → IntegralOperator K a b phi s = lambda * phi s

/-- Mercer's theorem representation of K -/
def MercerRepresentation (K : ℝ → ℝ → ℝ) (a b : ℝ) 
  (lambda : ℕ → ℝ) (phi : ℕ → ℝ → ℝ) : Prop :=
  ∀ s t, a ≤ s ∧ s ≤ b ∧ a ≤ t ∧ t ≤ b → 
    K s t = ∑' n, lambda n * phi n s * phi n t

/-- Main theorem statement -/
theorem covariance_operator_and_gaussian_process
  (K : ℝ → ℝ → ℝ) (a b : ℝ) (h : a < b)
  (hK : ContinuousCovariance K a b) :
  ∃ (A : (ℝ → ℝ) → (ℝ → ℝ))
    (lambda : ℕ → ℝ) (phi : ℕ → ℝ → ℝ) (xi : ℕ → ℝ → ℝ),
    (∀ f g, ∫ s in a..b, (A f s) * g s = ∫ s in a..b, f s * (A g s)) ∧ 
    (MercerRepresentation K a b lambda phi) ∧
    (∀ t, a ≤ t ∧ t ≤ b → 
      ∃ X : ℝ → ℝ, X t = ∑' n, xi n t * Real.sqrt (lambda n) * phi n t) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covariance_operator_and_gaussian_process_l780_78041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l780_78047

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 4 * Real.pi / 3

/-- Theorem stating that the area of the ellipse defined by the given equation is 4π/3 -/
theorem ellipse_area_proof : 
  ∃ (a b : ℝ), 
    (∀ x y : ℝ, ellipse_equation x y ↔ (x - a)^2 + (y - b)^2 / (4/9) = 1) ∧
    ellipse_area = Real.pi * 1 * (4/3) :=
by
  sorry

#check ellipse_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_proof_l780_78047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l780_78022

theorem no_solutions_in_interval : ∀ x : ℝ, -π < x ∧ x < π →
  (Real.cos (4 * x) - Real.tan (5 * x) ≠ Real.sin (5 * x) / Real.cos (5 * x) - Real.sin (6 * x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l780_78022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_percentage_l780_78073

noncomputable section

-- Define the initial price
def initial_price : ℚ := 150

-- Define the second discount percentage
def second_discount : ℚ := 10

-- Define the final price
def final_price : ℚ := 108

-- Define the function that calculates the price after both discounts
def price_after_discounts (first_discount : ℚ) : ℚ :=
  initial_price * (1 - first_discount / 100) * (1 - second_discount / 100)

-- Theorem statement
theorem first_discount_percentage :
  ∃ (x : ℚ), price_after_discounts x = final_price ∧ x = 20 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_percentage_l780_78073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_l780_78094

/-- Proves that a rabbit running at 5 miles per hour takes 24 minutes to travel 2 miles -/
theorem rabbit_travel_time : 
  let rabbit_speed : ℝ := 5  -- miles per hour
  let travel_distance : ℝ := 2  -- miles
  let minutes_per_hour : ℝ := 60
  let travel_time_minutes : ℝ := travel_distance / rabbit_speed * minutes_per_hour
  travel_time_minutes = 24 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_l780_78094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l780_78015

/-- An odd function f: ℝ → ℝ such that f(x) = x - sin x for x ≥ 0 -/
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≥ 0 then x - Real.sin x else -((-x) - Real.sin (-x))

theorem odd_function_inequality (m : ℝ) :
  (∀ t : ℝ, f (-4 * t) > f (2 * m + m * t^2)) → m < -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_l780_78015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_prime_value_l780_78027

def f (y : ℤ) : ℕ := Int.natAbs (4 * y^2 - 19 * y + 5)

theorem greatest_integer_prime_value :
  (∀ y : ℤ, y > 0 → ¬ Nat.Prime (f y)) ∧ Nat.Prime (f 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_prime_value_l780_78027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l780_78085

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) / (2 - Real.sin x * Real.cos x)

theorem range_of_f :
  Set.range f = { y : ℝ | -2 * Real.sqrt 2 / 5 ≤ y ∧ y ≤ 2 * Real.sqrt 2 / 5 } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l780_78085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l780_78004

/-- A complex number in the first quadrant with magnitude 4 -/
structure FirstQuadrantComplexNumber where
  x : ℝ
  y : ℝ
  first_quadrant : x > 0 ∧ y > 0
  magnitude : x^2 + y^2 = 16

/-- A hyperbola with axes lengths corresponding to a complex number -/
structure Hyperbola (z : FirstQuadrantComplexNumber) where
  real_axis : ℝ := z.x
  imag_axis : ℝ := z.y

/-- The focal length of a hyperbola -/
noncomputable def focal_length (z : FirstQuadrantComplexNumber) : ℝ :=
  2 * Real.sqrt ((z.x / 2)^2 + (z.y / 2)^2)

/-- Theorem stating that the focal length of the hyperbola is 4 -/
theorem hyperbola_focal_length (z : FirstQuadrantComplexNumber) :
  focal_length z = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l780_78004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_theorem_l780_78013

def photo_arrangement_count : Nat :=
  let num_students : Nat := 4
  let num_teacher : Nat := 1
  let total_people : Nat := num_students + num_teacher
  -- We don't need to calculate the teacher's position for this problem
  num_students.factorial

#eval photo_arrangement_count  -- This should evaluate to 24

theorem photo_arrangement_theorem :
  photo_arrangement_count = 24 := by
  -- Unfold the definition of photo_arrangement_count
  unfold photo_arrangement_count
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_theorem_l780_78013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sisters_money_ratio_l780_78000

/-- Represents the money of Joanna's sister as a fraction of Joanna's money -/
def sister_money (joanna_money : ℚ) (total_money : ℚ) : ℚ :=
  total_money - joanna_money - (3 * joanna_money)

/-- Proves that the ratio of Joanna's sister's money to Joanna's money is 1:2 given the conditions -/
theorem sisters_money_ratio (joanna_money : ℚ) (total_money : ℚ) : 
  joanna_money = 8 →
  total_money = 36 →
  (sister_money joanna_money total_money) / joanna_money = 1 / 2 := by
  intro h1 h2
  unfold sister_money
  simp [h1, h2]
  norm_num
  
#eval sister_money 8 36 / 8 -- This should evaluate to 1/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sisters_money_ratio_l780_78000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_properties_l780_78019

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (Real.exp x) / (x^2) + x - 2 * Real.log x + m

-- Define the theorem
theorem function_zeros_properties (m : ℝ) 
  (h1 : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ m = 0 ∧ f x₂ m = 0) :
  (m < 2 * Real.log 2 - 2 - Real.exp 2 / 4) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m = 0 → f x₂ m = 0 → x₁ + x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_properties_l780_78019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l780_78036

structure Student where
  weight : ℝ
  height : ℝ
  age : ℝ

noncomputable def averageWeight (students : List Student) : ℝ :=
  (students.map (·.weight)).sum / students.length

theorem new_student_weight (initialStudents : List Student)
    (h1 : initialStudents.length = 29)
    (h2 : (initialStudents.map (·.height)).sum / initialStudents.length = 1.5)
    (h3 : averageWeight initialStudents = 28)
    (h4 : (initialStudents.map (·.age)).sum / initialStudents.length = 14.2)
    (newStudent : Student)
    (h5 : newStudent.height = 1.65)
    (h6 : newStudent.age = 15)
    (h7 : averageWeight (newStudent :: initialStudents) = 27.4) :
  newStudent.weight = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l780_78036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l780_78003

/-- A right rectangular prism with given face areas and reciprocal product of side lengths -/
structure RectangularPrism where
  side_area : ℝ
  front_area : ℝ
  bottom_area : ℝ
  reciprocal_product : ℝ

/-- The volume of a rectangular prism given its properties -/
noncomputable def volume (p : RectangularPrism) : ℝ :=
  Real.sqrt (p.side_area * p.front_area * p.bottom_area)

/-- Theorem stating the volume of the specific rectangular prism -/
theorem specific_prism_volume :
  let p : RectangularPrism := {
    side_area := 12,
    front_area := 18,
    bottom_area := 9,
    reciprocal_product := 1 / 216
  }
  abs (volume p - 44) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l780_78003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l780_78028

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ :=
  abs (c₂ / a₂ - c₁ / a₁) / Real.sqrt ((a₁ / b₁ - a₂ / b₂)^2 + 1)

/-- Theorem: The distance between the lines x + y - 1 = 0 and 2x + 2y + 1 = 0 is 3√2/4 -/
theorem distance_specific_lines :
  distance_between_parallel_lines 1 1 (-1) 2 2 1 = 3 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_lines_l780_78028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_proof_l780_78054

def geometric_series (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^n

def series_sum (y : ℝ) : ℝ := 2 + 7*y + 12*y^2 + 17*y^3 + (geometric_series 22 y 4)

theorem solution_proof (y : ℝ) (h1 : |y| < 1) (h2 : series_sum y = 92) : y = 18/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_proof_l780_78054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_cost_l780_78077

-- Define the prices and quantities
noncomputable def pasta_price : ℚ := 3/2
noncomputable def pasta_quantity : ℚ := 2
noncomputable def beef_price : ℚ := 8
noncomputable def beef_quantity : ℚ := 1/4
noncomputable def sauce_price : ℚ := 2
noncomputable def sauce_quantity : ℚ := 2
noncomputable def quesadilla_price : ℚ := 6

-- Define the total cost function
noncomputable def total_cost : ℚ :=
  pasta_price * pasta_quantity +
  beef_price * beef_quantity +
  sauce_price * sauce_quantity +
  quesadilla_price

-- Theorem statement
theorem grocery_cost : total_cost = 15 := by
  -- Unfold the definition of total_cost
  unfold total_cost
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_cost_l780_78077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l780_78067

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then -x - 1 else 1 - x

theorem f_properties : 
  (∀ x ∈ ({x : ℝ | x > 0} ∪ {x : ℝ | x < 0}), f (-x) = -f x) ∧ 
  {x : ℝ | f x > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l780_78067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l780_78040

-- Define the points A and B
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, -2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  distance P A + distance P B = 4

-- Define what it means for a point to be on a line segment
def on_line_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- The theorem to be proved
theorem trajectory_is_line_segment :
  ∀ P : ℝ × ℝ, satisfies_condition P → on_line_segment P A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l780_78040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_wage_ratio_calculation_l780_78038

/-- Represents the regression line equation and consumption data -/
structure RegressionData where
  slope : ℝ
  intercept : ℝ
  avg_consumption : ℝ

/-- Calculates the average wage given regression data -/
noncomputable def calc_avg_wage (data : RegressionData) : ℝ :=
  (data.avg_consumption - data.intercept) / data.slope

/-- Calculates the ratio of average consumption to average wage -/
noncomputable def consumption_wage_ratio (data : RegressionData) : ℝ :=
  data.avg_consumption / calc_avg_wage data

/-- Theorem stating the consumption to wage ratio for given data -/
theorem consumption_wage_ratio_calculation (data : RegressionData)
  (h1 : data.slope = 0.7)
  (h2 : data.intercept = 2.1)
  (h3 : data.avg_consumption = 10.5) :
  consumption_wage_ratio data = 0.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_wage_ratio_calculation_l780_78038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l780_78009

open Real

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f (f x + y) = f (f x) + 2 * y * f x - f y + 2 * y^2 + 1

/-- The theorem stating that x^2 + 1 is the unique solution -/
theorem unique_solution :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → (∀ x : ℝ, x > 0 → f x = x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l780_78009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l780_78026

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 1/2| + a*|x - 3/2|

-- Part I
theorem part_one :
  {x : ℝ | f (-1) x ≤ 3*x} = {x : ℝ | x ≥ -1/2} := by sorry

-- Part II
theorem part_two :
  ∀ b : ℝ, (∀ x : ℝ, 2*(f 2 x) + 1 ≥ |1 - b|) → b ∈ Set.Icc (-7) 9 := by sorry

#check part_one
#check part_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l780_78026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_xyz_abc_l780_78083

-- Define the points
def A : ℚ × ℚ := (2, 0)
def B : ℚ × ℚ := (8, 12)
def C : ℚ × ℚ := (14, 0)
def X : ℚ × ℚ := (6, 0)
def Y : ℚ × ℚ := (8, 4)
def Z : ℚ × ℚ := (10, 0)

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem area_ratio_xyz_abc :
  (triangleArea X Y Z) / (triangleArea A B C) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_xyz_abc_l780_78083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l780_78012

theorem complex_number_opposite_parts (a : ℝ) : 
  (((1 : ℂ) - Complex.I * a) * Complex.I).re = 
  -(((1 : ℂ) - Complex.I * a) * Complex.I).im → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_opposite_parts_l780_78012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_3_10_times_b_10_l780_78066

noncomputable def a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (2/3) * a n + (4/3) * Real.sqrt (9^n - (a n)^2)

noncomputable def b (n : ℕ) : ℝ := a n / 3^n

theorem a_10_equals_3_10_times_b_10 : a 10 = 3^10 * b 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_3_10_times_b_10_l780_78066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_covered_l780_78059

/-- The distance covered by a wheel with given radius and number of revolutions -/
noncomputable def distance_covered (radius : ℝ) (revolutions : ℕ) : ℝ :=
  2 * Real.pi * radius * (revolutions : ℝ)

/-- Theorem: The distance covered by a wheel with radius 20.4 cm making 400 revolutions
    is approximately 512.707488 meters -/
theorem wheel_distance_covered :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000001 ∧ 
  |distance_covered 0.204 400 - 512.707488| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_covered_l780_78059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_non_real_l780_78050

theorem equation_solution_non_real (a b : ℂ) : 
  (2 * a ≠ 0) → 
  (2 * a + 3 * b ≠ 0) → 
  ((2 * a + 3 * b) / (2 * a) = 3 * b / (2 * a + 3 * b)) → 
  (¬(a.re = a ∧ a.im = 0) ∨ ¬(b.re = b ∧ b.im = 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_non_real_l780_78050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_850_l780_78046

/-- The area of a square with side length 40 units -/
noncomputable def square_area : ℝ := 40 * 40

/-- The area of the first triangle with base 30 units and height 30 units -/
noncomputable def triangle1_area : ℝ := (1 / 2) * 30 * 30

/-- The area of the second triangle with base 20 units and height 30 units -/
noncomputable def triangle2_area : ℝ := (1 / 2) * 20 * 30

/-- The theorem stating that the area of the shaded region is 850 square units -/
theorem shaded_area_is_850 : square_area - (triangle1_area + triangle2_area) = 850 := by
  -- Unfold definitions
  unfold square_area triangle1_area triangle2_area
  -- Simplify arithmetic expressions
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_850_l780_78046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l780_78090

def A (a : ℝ) : Set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3 * x + 2 * a = 0}

theorem problem_solution :
  ∃ (a : ℝ),
    (A a ∩ B a = {2}) ∧
    (a = -5) ∧
    (A a = {2, 1/2}) ∧
    (B a = {2, -5}) ∧
    (let U := A a ∪ B a;
     (Uᶜ ∩ A a)ᶜ ∩ (Uᶜ ∩ B a)ᶜ = {1/2, -5}) ∧
    (Set.powerset ((Uᶜ ∩ A a)ᶜ ∩ (Uᶜ ∩ B a)ᶜ) = {∅, {1/2}, {-5}, {1/2, -5}}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l780_78090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l780_78006

noncomputable def f (x : ℝ) : ℝ := (x^3 - 64) / (x + 64)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -64} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l780_78006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l780_78069

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ∫ t in Set.Icc 0 1, |t - x| * t

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = (2 - Real.sqrt 2) / 6 ∧
  ∀ (x : ℝ), f x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l780_78069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_a_range_l780_78099

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem f_min_value :
  ∃ (min : ℝ), min = 5 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

-- Theorem for the range of a
theorem a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), x ∈ Set.Icc (-3) 2 → f x ≥ |x + a|) ↔
    a ∈ Set.Icc (-2) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_a_range_l780_78099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_2_power_divides_3_power_greatest_2_power_greatest_3_power_main_result_l780_78086

/-- The greatest power of 2 that divides 360 -/
def x : ℕ := 3

/-- The greatest power of 3 that divides 360 -/
def y : ℕ := 2

/-- 360 is divisible by 2^x -/
theorem divides_2_power : 2^x ∣ 360 :=
sorry

/-- 360 is divisible by 3^y -/
theorem divides_3_power : 3^y ∣ 360 :=
sorry

/-- 2^x is the greatest power of 2 that divides 360 -/
theorem greatest_2_power : ¬(2^(x+1) ∣ 360) :=
sorry

/-- 3^y is the greatest power of 3 that divides 360 -/
theorem greatest_3_power : ¬(3^(y+1) ∣ 360) :=
sorry

/-- The main theorem: (1/3)^(y-x) = 3 -/
theorem main_result : (1/3 : ℚ)^(y-x) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_2_power_divides_3_power_greatest_2_power_greatest_3_power_main_result_l780_78086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l780_78048

-- Define the constant e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Define the variables a, b, and c
noncomputable def a : ℝ := Real.log 0.99
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := (0.99 : ℝ) ^ e

-- State the theorem
theorem order_of_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l780_78048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l780_78034

/-- The nth term of a geometric sequence with first term a and common ratio r -/
noncomputable def geometric_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- The eighth term of the specific geometric sequence -/
noncomputable def eighth_term : ℝ := geometric_term 12 (1/4) 8

theorem eighth_term_value : eighth_term = 3/4096 := by
  -- Unfold the definitions
  unfold eighth_term geometric_term
  -- Simplify the expression
  simp [pow_succ, pow_zero]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l780_78034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l780_78068

theorem problem_1 (a b : ℝ) (h : a - b = 3) : 1 + 2*b - (a + b) = -2 := by sorry

theorem problem_2 (x : ℝ) (h : (2 : ℝ)^x = 3) : (2 : ℝ)^(2*x - 3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l780_78068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l780_78081

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm := Real.sqrt (v.1^2 + v.2^2)
  let a := v.1 / norm
  let b := v.2 / norm
  !![a^2, a*b; a*b, b^2]

theorem projection_matrix_not_invertible :
  let v : ℝ × ℝ := (4, 5)
  let Q := projection_matrix v
  ¬ IsUnit (Q.det) := by
  sorry

#check projection_matrix_not_invertible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l780_78081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solution_count_l780_78098

theorem congruence_solution_count (a b : ℕ) (h1 : a = 3^100) (h2 : b = 5454) :
  ∀ z : ℕ, 1 ≤ z ∧ z < 3^99 →
  (∀ c : ℕ, Nat.Coprime c 3 →
    (∀ x : ℕ, x^z ≡ c [MOD a] ↔ x^b ≡ c [MOD a])) ↔
  ∃ t k : ℕ, t ≥ 1 ∧ Nat.Coprime k 6 ∧ z = 2^t * 3^3 * k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solution_count_l780_78098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_relation_l780_78089

theorem circle_point_relation (b lambda : ℝ) : 
  b ≠ -2 →
  (∀ x y : ℝ, x^2 + y^2 = 1 → 
    (x - b)^2 + y^2 = lambda^2 * ((x + 2)^2 + y^2)) →
  b = -1/2 ∧ lambda = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_relation_l780_78089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_7_simplest_l780_78063

/-- A quadratic radical is simplest if it cannot be simplified further -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y z : ℝ, y^2 * z = x → (y = 1 ∨ z = 1)

/-- The given set of quadratic radicals -/
def quadratic_radicals : Set ℝ := {Real.sqrt 12, Real.sqrt (2/3), Real.sqrt 0.3, Real.sqrt 7}

theorem sqrt_7_simplest :
  Real.sqrt 7 ∈ quadratic_radicals ∧
  is_simplest_quadratic_radical (Real.sqrt 7) ∧
  ∀ x ∈ quadratic_radicals, is_simplest_quadratic_radical x → x = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_7_simplest_l780_78063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_81_l780_78020

theorem power_of_81 : (81 : ℝ)^(5/4) = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_81_l780_78020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l780_78080

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I →
  Complex.ofReal a + Complex.I * Complex.ofReal b = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l780_78080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_hits_l780_78018

def target_shooting (p : ℝ) (k : ℕ) : Set ℕ :=
  {n : ℕ | (p * (n : ℝ) - (1 - p) ≤ (k : ℝ)) ∧ ((k : ℝ) ≤ p * (n : ℝ) + p)}

theorem most_probable_hits (p : ℝ) (k : ℕ) (hp : p = 0.7) (hk : k = 16) :
  target_shooting p k = {22, 23} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_probable_hits_l780_78018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l780_78084

theorem power_equality (y : ℝ) (h : (8 : ℝ)^y - (8 : ℝ)^(y-1) = 448) : (3*y)^y = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l780_78084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_bound_l780_78079

/-- Given a function f(x) = ax - x^3 on the interval (0, 1), 
    if f(x₂) - f(x₁) > x₂ - x₁ for any x₁, x₂ ∈ (0, 1) where x₁ < x₂, 
    then a ≥ 4 -/
theorem function_condition_implies_a_bound (a : ℝ) : 
  (∀ x, x ∈ Set.Ioo 0 1 → ∃ f : ℝ → ℝ, f x = a * x - x^3) →
  (∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ < x₂ → 
    ∃ f : ℝ → ℝ, f x₂ - f x₁ > x₂ - x₁) →
  a ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_bound_l780_78079
