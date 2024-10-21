import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_currents_l903_90354

/-- Represents a circuit with given EMF and resistances -/
structure Circuit where
  ε : ℝ  -- EMF
  R : ℝ  -- Base resistance
  R₁ : ℝ -- Resistance 1
  R₂ : ℝ -- Resistance 2
  R₃ : ℝ -- Resistance 3
  R₄ : ℝ -- Resistance 4

/-- Calculates the current through R₁ in the given circuit -/
noncomputable def current_R₁ (c : Circuit) : ℝ :=
  (7 * c.ε) / (23 * c.R₁)

/-- Calculates the current through the ammeter in the given circuit -/
noncomputable def current_ammeter (c : Circuit) : ℝ :=
  ((7 * c.ε) / (23 * c.R₁)) - ((16 * c.ε) / (69 * c.R₁))

/-- Theorem stating the currents in the circuit -/
theorem circuit_currents (c : Circuit) 
    (h1 : c.ε = 69)
    (h2 : c.R = 10)
    (h3 : c.R₁ = c.R)
    (h4 : c.R₂ = 3 * c.R)
    (h5 : c.R₃ = 3 * c.R)
    (h6 : c.R₄ = 4 * c.R) :
    current_R₁ c = 2.1 ∧ current_ammeter c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_currents_l903_90354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l903_90353

theorem power_of_three (m n : ℝ) (h1 : (9 : ℝ)^m = 3) (h2 : (27 : ℝ)^n = 4) : 
  (3 : ℝ)^(2*m + 3*n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l903_90353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_sum_asymptotic_l903_90397

/-- Bernoulli random variable with p = 1/2 -/
def BernoulliHalf : Type := Bool

/-- Sum of n independent Bernoulli random variables -/
def S (n : ℕ) : ℕ → ℤ := sorry

/-- Expected value of |S_n| -/
noncomputable def E_abs_S (n : ℕ) : ℝ := sorry

/-- Asymptotic equivalence -/
def asymp_equiv (f g : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n / g n - 1| < ε

/-- Main theorem: E|S_n| ~ √(2n/π) as n → ∞ -/
theorem bernoulli_sum_asymptotic :
  asymp_equiv E_abs_S (λ n => Real.sqrt (2 * n / Real.pi)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_sum_asymptotic_l903_90397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l903_90309

-- Define the curve
noncomputable def curve (t : ℝ) : ℝ × ℝ := (Real.sin t, Real.cos t)

-- Define the parameter value
noncomputable def t₀ : ℝ := Real.pi / 6

-- Define the point on the curve at t₀
noncomputable def point_on_curve : ℝ × ℝ := curve t₀

-- Define the slope of the tangent line at t₀
noncomputable def tangent_slope : ℝ := -Real.tan t₀

-- Define the slope of the normal line at t₀
noncomputable def normal_slope : ℝ := -1 / tangent_slope

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  ∀ x y : ℝ, 
  y = -1/Real.sqrt 3 * x + 2/Real.sqrt 3 ↔ 
  y - point_on_curve.2 = tangent_slope * (x - point_on_curve.1) := by
  sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  ∀ x y : ℝ,
  y = Real.sqrt 3 * x ↔
  y - point_on_curve.2 = normal_slope * (x - point_on_curve.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l903_90309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l903_90312

noncomputable def f (x : ℝ) := 3 * Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x : ℝ, f (π/3 - x) = f (π/3 + x)) ∧
  (∀ x ∈ Set.Icc (-π/3) (-π/6), ∀ y ∈ Set.Icc (-π/3) (-π/6), x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l903_90312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l903_90322

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / ⌊2 * x^2 - 10 * x + 16⌋

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≤ 5/2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l903_90322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_min_at_40_l903_90367

/-- The power consumption function for an electric bicycle -/
noncomputable def power_consumption (x : ℝ) : ℝ := (1/3) * x^3 - (39/2) * x^2 - 40 * x

/-- Theorem stating that the power consumption is minimized at speed 40 -/
theorem power_consumption_min_at_40 :
  ∀ x > 0, power_consumption x ≥ power_consumption 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_consumption_min_at_40_l903_90367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l903_90364

/-- The sum of the infinite series $\sum_{n = 1}^\infty \frac{3n - 2}{n(n + 1)(n + 3)}$ -/
noncomputable def infiniteSeries : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

/-- Theorem: The sum of the infinite series equals 31/24 -/
theorem infiniteSeriesSum : infiniteSeries = 31 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l903_90364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l903_90336

/-- Definition of the ellipse E -/
noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- Definition of point P -/
noncomputable def P : ℝ × ℝ := (0, 2 * Real.sqrt 3)

/-- Definition of the line l with slope k passing through P -/
noncomputable def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + P.2

/-- Theorem stating the slope of line l -/
theorem ellipse_line_slope :
  ∃ (k : ℝ), (k = 4 * Real.sqrt 2 / 3 ∨ k = -4 * Real.sqrt 2 / 3) ∧
  ∃ (M N : ℝ × ℝ),
    ellipse M.1 M.2 ∧
    ellipse N.1 N.2 ∧
    line k M.1 M.2 ∧
    line k N.1 N.2 ∧
    (N.1 - P.1, N.2 - P.2) = (3 * (M.1 - P.1), 3 * (M.2 - P.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l903_90336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_division_possible_l903_90328

/-- A prism is a polyhedron with two parallel polygonal bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  base : Set Point
  height : ℝ
  is_polygonal : Prop  -- Changed from IsPolygon to Prop
  is_positive_height : height > 0

/-- A plane is a flat, two-dimensional surface that extends infinitely in all directions. -/
structure Plane where
  normal : ℝ → ℝ → ℝ → ℝ  -- Changed from Vector to a function type
  point : Point

/-- Represents the result of intersecting a prism with a plane. -/
inductive PrismIntersection
  | TwoPrisms (upper : Prism) (lower : Prism)
  | Other

/-- Function to intersect a prism with a plane -/
def intersect_prism_plane (P : Prism) (π : Plane) : PrismIntersection :=
  sorry  -- Placeholder implementation

/-- States that it is possible for a plane to divide a prism into two prisms. -/
theorem prism_division_possible (P : Prism) : 
  ∃ (π : Plane), ∃ (upper lower : Prism), 
    PrismIntersection.TwoPrisms upper lower = intersect_prism_plane P π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_division_possible_l903_90328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l903_90357

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I : ℂ) / (1 + Complex.I) = ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l903_90357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_ice_cream_amount_l903_90325

/-- Given Blake's milkshake-making scenario, prove he had 192 ounces of ice cream initially. -/
theorem blake_ice_cream_amount 
  (milk_per_shake : ℕ)
  (ice_cream_per_shake : ℕ)
  (initial_milk : ℕ)
  (leftover_milk : ℕ)
  (h1 : milk_per_shake = 4)
  (h2 : ice_cream_per_shake = 12)
  (h3 : initial_milk = 72)
  (h4 : leftover_milk = 8) :
  (initial_milk - leftover_milk) / milk_per_shake * ice_cream_per_shake = 192 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blake_ice_cream_amount_l903_90325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l903_90362

/-- The time (in seconds) it takes for two trains to meet, given their initial distance and speeds. -/
noncomputable def time_to_meet (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance / (speed1 + speed2)

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem trains_meet_time :
  let initial_distance : ℝ := 273.2
  let speed1 : ℝ := kmph_to_mps 65
  let speed2 : ℝ := kmph_to_mps 88
  let meet_time := time_to_meet initial_distance speed1 speed2
  |meet_time - 2.576| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l903_90362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_range_l903_90307

def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (|m| - 1) - y^2 / (m - 2) = 1

theorem hyperbola_m_range :
  (∀ m : ℝ, is_hyperbola m → m ∈ Set.Ioo (-1 : ℝ) 1 ∪ Set.Ioi (2 : ℝ)) ∧
  (∀ m : ℝ, m ∈ Set.Ioo (-1 : ℝ) 1 ∪ Set.Ioi (2 : ℝ) → is_hyperbola m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_range_l903_90307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_diff_distributive_l903_90381

noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def diff_avg (a b : ℝ) : ℝ := (a - b) / 2

theorem avg_diff_distributive (x y z : ℝ) :
  diff_avg x (avg y z) = avg (diff_avg x y) (diff_avg x z) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_diff_distributive_l903_90381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_l903_90301

theorem max_value_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 5 * y < 100) :
  x * y * (100 - 2*x - 5*y) ≤ 3703.7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_l903_90301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_master_practice_l903_90360

theorem chess_master_practice (games : Fin 77 → ℕ) 
  (h1 : ∀ i, games i ≥ 1)
  (h2 : ∀ i, (Finset.range 7).sum (λ j ↦ games ((i + j) % 77)) ≤ 12) :
  ∃ n : ℕ, ∃ start : Fin 77, 
    (Finset.range n).sum (λ j ↦ games ((start + j) % 77)) = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_master_practice_l903_90360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_play_area_is_correct_l903_90361

/-- The area in which Chuck the llama can play when tied to the corner of a rectangular shed --/
noncomputable def chucks_play_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let main_area := (3/4) * Real.pi * leash_length^2
  let additional_area := (1/4) * Real.pi * (leash_length - shed_length)^2
  main_area + additional_area

/-- Theorem: Chuck's play area is 12.25π square meters --/
theorem chucks_play_area_is_correct :
  chucks_play_area 3 4 4 = (49/4) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_play_area_is_correct_l903_90361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90349

-- Define the line
def line (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x + 2 * y - 2 * m - 2 = 0

-- Define the fixed point C
def C : ℝ × ℝ := (2, 0)

-- Define the circle (renamed to avoid conflict)
def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define C'
def C' : ℝ × ℝ := (-2, 0)

-- Define curve E
def curveE (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Additional definitions (placeholders)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry
def is_tangent (P Q : ℝ × ℝ) : Prop := sorry
def vector (A B : ℝ × ℝ) : ℝ × ℝ := sorry
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

theorem problem_solution :
  (∀ m : ℝ, line m C.1 C.2) ∧ 
  (∀ x y : ℝ, circleC x y ↔ (x - C.1)^2 + (y - C.2)^2 = 4) ∧
  (∀ M : ℝ × ℝ, 
    (∃ θ : ℝ, angle C' M C = 2*θ ∧ area_triangle M C C' = 4*Real.tan θ) 
    ↔ curveE M.1 M.2) ∧
  (∀ P : ℝ × ℝ, curveE P.1 P.2 → 
    (∃ Q R : ℝ × ℝ, 
      circleC Q.1 Q.2 ∧ circleC R.1 R.2 ∧ 
      is_tangent P Q ∧ is_tangent P R ∧
      dot_product (vector P Q) (vector P R) ≥ 8 * Real.sqrt 2 - 12)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l903_90398

noncomputable def small_pump_time : ℝ := 2
noncomputable def larger_pump_time : ℝ := 1/3
noncomputable def extra_large_pump_time : ℝ := 1/5
noncomputable def mini_pump_time : ℝ := 2.5

noncomputable def combined_fill_time (t1 t2 t3 t4 : ℝ) : ℝ :=
  1 / (1/t1 + 1/t2 + 1/t3 + 1/t4)

theorem pump_fill_time :
  combined_fill_time small_pump_time larger_pump_time extra_large_pump_time mini_pump_time = 10/89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l903_90398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l903_90350

noncomputable def slope1 (a : ℝ) : ℝ := -1/a
noncomputable def slope2 (a : ℝ) : ℝ := -(a-2)/3

def are_parallel (a : ℝ) : Prop := slope1 a = slope2 a

theorem parallel_condition :
  ∀ a : ℝ, a ≠ 0 → (are_parallel a ↔ a = -1) :=
by
  intro a ha
  simp [are_parallel, slope1, slope2]
  field_simp [ha]
  ring_nf
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l903_90350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90333

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x) / (4^x + 2) + a

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem problem_solution :
  (∃ a : ℝ, f a (lg 2) + f a (lg 5) = 3 ∧ a = 1) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f 1 x ≥ 4^x + m) → m ≤ -7/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_A_ethanol_percentage_approx_l903_90344

noncomputable def tank_capacity : ℝ := 214
noncomputable def fuel_A_volume : ℝ := 106
noncomputable def fuel_B_ethanol_percentage : ℝ := 0.16
noncomputable def total_ethanol : ℝ := 30

noncomputable def fuel_B_volume : ℝ := tank_capacity - fuel_A_volume

noncomputable def fuel_A_ethanol_percentage : ℝ := (total_ethanol - fuel_B_volume * fuel_B_ethanol_percentage) / fuel_A_volume

theorem fuel_A_ethanol_percentage_approx :
  ∃ ε > 0, |fuel_A_ethanol_percentage - 0.12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_A_ethanol_percentage_approx_l903_90344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_ratio_is_one_l903_90332

/-- Represents a cylindrical frustum -/
structure CylindricalFrustum where
  h : ℝ  -- height
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  l : ℝ  -- slant height

/-- Calculates the lateral surface area of a cylindrical frustum -/
noncomputable def lateralSurfaceArea (cf : CylindricalFrustum) : ℝ :=
  Real.pi * (cf.R + cf.r) * cf.l

/-- Transforms a cylindrical frustum according to the given conditions -/
noncomputable def transformFrustum (cf : CylindricalFrustum) (n : ℝ) : CylindricalFrustum :=
  { h := n * cf.h
    R := cf.R / n
    r := cf.r / n
    l := cf.l }

/-- Theorem stating that the ratio of lateral surface areas is 1 after transformation -/
theorem lateral_surface_area_ratio_is_one (cf : CylindricalFrustum) (n : ℝ) (hn : n ≠ 0) :
  lateralSurfaceArea (transformFrustum cf n) = lateralSurfaceArea cf := by
  sorry

#check lateral_surface_area_ratio_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_ratio_is_one_l903_90332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_volume_is_270_l903_90385

/-- Calculates the volume of a trapezoidal prism-shaped swimming pool. -/
noncomputable def swimming_pool_volume (width length shallow_depth deep_depth : ℝ) : ℝ :=
  (1 / 2) * (shallow_depth + deep_depth) * width * length

/-- Theorem stating that a swimming pool with given dimensions has a volume of 270 cubic meters. -/
theorem swimming_pool_volume_is_270 :
  swimming_pool_volume 9 12 1 4 = 270 := by
  -- Unfold the definition of swimming_pool_volume
  unfold swimming_pool_volume
  -- Simplify the arithmetic expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the result is equal to 270
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_volume_is_270_l903_90385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_squared_value_l903_90363

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 9)*(3*w + 6)) : 
  ∃ ε > 0, abs (w^2 - 91.44) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_squared_value_l903_90363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l903_90345

theorem find_n (n : ℕ) (a : ℕ → ℝ) : 
  (∀ x : ℝ, (Finset.range (n + 1)).sum (λ i => (1 + x)^i) = 
    (Finset.range (n + 1)).sum (λ i => a i * x^i)) →
  ((Finset.range (n - 1)).sum (λ i => a (i + 1))) = 29 - n →
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l903_90345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_with_plane_l903_90382

-- Define the necessary types if they're not already in Mathlib
def Line3 : Type := sorry
def Plane3 : Type := sorry

-- Define the necessary operations if they're not already in Mathlib
def Line3.intersects (l : Line3) (p : Plane3) : Prop := sorry
def angle_line_plane (l : Line3) (p : Plane3) : ℝ := sorry

-- Define parallel for lines if it's not already in Mathlib
def parallel (l₁ l₂ : Line3) : Prop := sorry

infixl:50 " ∥ " => parallel

theorem parallel_lines_equal_angles_with_plane (l₁ l₂ : Line3) (p : Plane3) 
  (h_parallel : l₁ ∥ l₂) (h_intersect₁ : l₁.intersects p) (h_intersect₂ : l₂.intersects p) :
  angle_line_plane l₁ p = angle_line_plane l₂ p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_with_plane_l903_90382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l903_90355

/-- The ellipse with equation x²/49 + y²/24 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 49) + (p.2^2 / 24) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

/-- A point P on the ellipse -/
noncomputable def P : ℝ × ℝ := sorry

/-- The condition that P forms perpendicular lines with F₁ and F₂ -/
def perpendicularLines (P : ℝ × ℝ) : Prop :=
  (P.2 / (P.1 + 5)) * (P.2 / (P.1 - 5)) = -1

/-- The area of triangle PF₁F₂ -/
noncomputable def triangleArea (P : ℝ × ℝ) : ℝ :=
  (1/2) * 10 * |P.2|

theorem ellipse_triangle_area :
  P ∈ Ellipse → perpendicularLines P → triangleArea P = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l903_90355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_simplify_series_l903_90346

-- Problem 1
theorem simplify_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  2 / (a.sqrt - b.sqrt) = a.sqrt + b.sqrt :=
by sorry

-- Problem 2
theorem simplify_series :
  (Finset.range 1010).sum (λ i => 1 / (Real.sqrt (2 * i + 4) + Real.sqrt (2 * i + 2))) =
  (Real.sqrt 2022 - Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_simplify_series_l903_90346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l903_90384

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 4)

theorem periodic_function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :
  -- 1. Range and minimum on [0, π/2]
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → -Real.sqrt 2 / 2 ≤ f ω x ∧ f ω x ≤ 1) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f ω x ≥ f ω (Real.pi / 2)) ∧
  -- 2. Intervals of monotonic increase
  (∀ k : ℤ, ∀ x y, x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    x ≤ y → f ω x ≤ f ω y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_l903_90384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_or_eight_l903_90375

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  decr : d < 0
  seq : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Theorem: For a decreasing arithmetic sequence where S_5 = S_10, 
    the value of n that maximizes S_n is either 7 or 8 -/
theorem max_sum_at_seven_or_eight (seq : ArithmeticSequence) 
    (h : S seq 5 = S seq 10) : 
    ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ 
    (∀ m : ℕ, S seq m ≤ S seq n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_seven_or_eight_l903_90375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l903_90326

/-- The fraction of a 7×6 grid covered by a triangle with vertices (2,4), (6,2), and (5,5) -/
theorem triangle_area_fraction : 
  let A : ℝ × ℝ := (2, 4)
  let B : ℝ × ℝ := (6, 2)
  let C : ℝ × ℝ := (5, 5)
  let grid_width : ℝ := 7
  let grid_height : ℝ := 6
  let triangle_area := (1/2) * abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))
  let grid_area := grid_width * grid_height
  triangle_area / grid_area = 5/42 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l903_90326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_value_l903_90337

def calculation : ℝ := 2.4 * 8.2 * (4.8 + 5.2)

def options : List ℝ := [150, 200, 250, 300, 350]

theorem closest_value :
  ∃ x ∈ options, ∀ y ∈ options, |calculation - x| ≤ |calculation - y| :=
by
  -- The proof goes here
  sorry

#eval calculation
#eval options

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_value_l903_90337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l903_90383

noncomputable section

variable (t : ℝ)
variable (α β : ℝ)
variable (u₁ u₂ u₃ : ℝ)

def f (t : ℝ) (x : ℝ) : ℝ := (2 * x - t) / (x^2 + 1)

noncomputable def g (t : ℝ) : ℝ := 
  (8 * Real.sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25)

theorem inequality_proof
  (h1 : α ≠ β)
  (h2 : 4 * α^2 - 4 * t * α - 1 = 0)
  (h3 : 4 * β^2 - 4 * t * β - 1 = 0)
  (h4 : α < β)
  (h5 : 0 < u₁) (h6 : u₁ < π/2)
  (h7 : 0 < u₂) (h8 : u₂ < π/2)
  (h9 : 0 < u₃) (h10 : u₃ < π/2)
  (h11 : Real.sin u₁ + Real.sin u₂ + Real.sin u₃ = 1) :
  1 / g (Real.tan u₁) + 1 / g (Real.tan u₂) + 1 / g (Real.tan u₃) < 3/4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l903_90383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_percentage_gain_l903_90377

/-- Calculate the percentage gain for a shoe sale given the manufacturing cost, transportation cost, and selling price. -/
theorem shoe_percentage_gain
  (manufacturing_cost : ℝ)
  (transportation_cost_per_100 : ℝ)
  (selling_price : ℝ)
  (h1 : manufacturing_cost = 220)
  (h2 : transportation_cost_per_100 = 500)
  (h3 : selling_price = 270) :
  (selling_price - (manufacturing_cost + transportation_cost_per_100 / 100)) /
  (manufacturing_cost + transportation_cost_per_100 / 100) * 100 = 20 := by
  sorry

#check shoe_percentage_gain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_percentage_gain_l903_90377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l903_90386

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Represents that a point lies on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Represents that a point lies on a line segment between two other points -/
def PointOnSegment (p q r : Point) : Prop := sorry

/-- Represents that two circles are tangent at a point -/
def CirclesTangentAt (c1 c2 : Circle) (p : Point) : Prop := sorry

/-- Represents the intersection points of two circles -/
def CirclesIntersection (c1 c2 : Circle) : Set Point := sorry

/-- Represents the intersection point of two lines -/
noncomputable def LinesIntersection (p1 q1 p2 q2 : Point) : Point := sorry

theorem fixed_intersection_point 
  (S : Circle) (A B C : Point) (S' : Circle → Circle) :
  PointOnCircle A S →
  PointOnCircle B S →
  PointOnSegment A C B →
  (∀ s', CirclesTangentAt (S' s') S C) →
  (∀ s', ∃ P Q, P ∈ CirclesIntersection S (S' s') ∧ Q ∈ CirclesIntersection S (S' s')) →
  ∃ M, ∀ s', 
    ∃ P Q, P ∈ CirclesIntersection S (S' s') ∧ 
            Q ∈ CirclesIntersection S (S' s') ∧ 
            M = LinesIntersection A B P Q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l903_90386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_74_between_8_and_9_l903_90302

theorem sqrt_74_between_8_and_9 : ∃ (a b : ℕ), a = 8 ∧ b = 9 ∧ (a : ℝ) < Real.sqrt 74 ∧ Real.sqrt 74 < (b : ℝ) ∧ a * b = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_74_between_8_and_9_l903_90302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_dilution_l903_90379

/-- Given a mixture of alcohol and water, calculate the new alcohol percentage after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (initial_volume_positive : 0 < initial_volume)
  (initial_alcohol_percentage_valid : 0 ≤ initial_alcohol_percentage ∧ initial_alcohol_percentage ≤ 100)
  (added_water_nonnegative : 0 ≤ added_water) :
  let initial_alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let new_total_volume := initial_volume + added_water
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  (initial_volume = 15 ∧ 
   initial_alcohol_percentage = 20 ∧ 
   added_water = 2) →
  (abs (new_alcohol_percentage - 17.65) < 0.01) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_after_dilution_l903_90379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l903_90313

/-- Given a parabola y² = 2px with 0 < p < 6, if a point M(3, y₀) on the parabola has a distance
    to the focus that is twice its distance to the line x = p/2, then p = 2. -/
theorem parabola_focus_distance (p y₀ : ℝ) : 
  0 < p → p < 6 → y₀^2 = 2 * p * 3 → 
  (3 + p / 2)^2 + y₀^2 = 4 * ((3 - p / 2)^2 + y₀^2) → 
  p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l903_90313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log18_not_directly_computable_log18_not_computable_l903_90329

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Given approximations
axiom log5_approx : ∃ ε > 0, |log 5 - 0.6990| < ε
axiom log10_approx : ∃ ε > 0, |log 10 - 1.0000| < ε

-- Define a predicate for directly computable logarithms
def directly_computable (x : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ), log x = f (log 5) (log 10)

-- Theorem statement
theorem log18_not_directly_computable :
  ¬ directly_computable 18 ∧
  directly_computable (6/5) ∧
  directly_computable 12 ∧
  directly_computable 450 ∧
  directly_computable 0.5 :=
by
  sorry -- Skip the proof for now

-- Additional theorem to show log 18 is not directly computable
theorem log18_not_computable : ¬ directly_computable 18 :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log18_not_directly_computable_log18_not_computable_l903_90329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l903_90347

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + Real.pi/4)

theorem amplitude_and_phase_shift :
  (∃ A : ℝ, ∀ x, |f x| ≤ A ∧ (∃ x₀, f x₀ = A ∨ f x₀ = -A)) ∧
  (∃ C : ℝ, ∀ x, f x = 5 * Real.cos (x - C)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l903_90347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l903_90316

/-- Represents the journey parameters and calculates the total time -/
noncomputable def Journey (totalDistance : ℝ) (carSpeed : ℝ) (harryWalkSpeed : ℝ) (dickWalkSpeed : ℝ) (harryCarDistance : ℝ) : ℝ :=
  let harryTime := harryCarDistance / carSpeed + (totalDistance - harryCarDistance) / harryWalkSpeed
  let dickWalkDistance := dickWalkSpeed * harryTime
  let tomDickTime := harryCarDistance / carSpeed + dickWalkDistance / carSpeed + (totalDistance - harryCarDistance + dickWalkDistance) / carSpeed
  harryTime

/-- Theorem stating that the journey time is approximately 7.47 hours -/
theorem journey_time_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (totalDistance carSpeed harryWalkSpeed dickWalkSpeed harryCarDistance : ℝ),
    totalDistance = 120 ∧
    carSpeed = 30 ∧
    harryWalkSpeed = 5 ∧
    dickWalkSpeed = 3 ∧
    harryCarDistance = 40 ∧
    |Journey totalDistance carSpeed harryWalkSpeed dickWalkSpeed harryCarDistance - 7.47| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l903_90316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l903_90315

/-- Predicate to check if a triangle is acute-angled -/
def acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the orthocenter of a triangle -/
def is_orthocenter (M A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the circumcenter of a triangle -/
def is_circumcenter (O A B C : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is the midpoint of a line segment -/
def is_midpoint (F B C : ℝ × ℝ) : Prop := sorry

/-- Given an acute-angled triangle ABC with orthocenter M, circumcenter O, and F as the midpoint of side BC, 
    the length of AM is twice the length of OF. -/
theorem orthocenter_circumcenter_relation (A B C M O F : ℝ × ℝ) : 
  acute_triangle A B C →
  is_orthocenter M A B C →
  is_circumcenter O A B C →
  is_midpoint F B C →
  dist A M = 2 * dist O F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l903_90315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l903_90358

/-- A game state represented by a list of natural numbers, where each number is the count of nuts in a pile --/
def GameState := List Nat

/-- Predicate to check if two numbers are coprime --/
def coprime (a b : Nat) : Prop := Nat.gcd a b = 1

/-- Function to combine two piles in a game state --/
def combinePiles (state : GameState) (i j : Fin state.length) : GameState :=
  sorry

/-- Predicate to check if a move is valid --/
def validMove (state : GameState) (i j : Fin state.length) : Prop :=
  i ≠ j ∧ coprime (state.get i) (state.get j)

/-- Predicate to check if a game state is a winning state (only one pile left) --/
def isWinningState (state : GameState) : Prop :=
  state.length = 1

/-- Theorem stating that the second player has a winning strategy for all N > 2 --/
theorem second_player_wins (N : Nat) (h : N > 2) :
  ∃ (strategy : GameState → Fin 2 × Fin 2),
    ∀ (firstPlayerMoves : List (Fin 2 × Fin 2)),
      let game := List.replicate N 1
      let finalState := sorry  -- Apply moves alternating between first player and strategy
      isWinningState finalState ∧ finalState.length % 2 = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l903_90358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_problem_l903_90389

/-- Given two monic cubic polynomials h and j, and a real number s, 
    satisfying the conditions specified, prove that s = 48. -/
theorem cubic_polynomial_root_problem (h j : ℝ → ℝ) (s : ℝ) : 
  (∀ x, h x - j x = 2*s) →  -- h(x) - j(x) = 2s for all real x
  (∃ c, h = λ x ↦ (x - (s + 2)) * (x - (s + 8)) * (x - c)) →  -- h has roots s+2 and s+8
  (∃ d, j = λ x ↦ (x - (s + 5)) * (x - (s + 11)) * (x - d)) →  -- j has roots s+5 and s+11
  (∀ x, ∃ a b c, h x = x^3 + a*x^2 + b*x + c) →  -- h is monic cubic
  (∀ x, ∃ a b c, j x = x^3 + a*x^2 + b*x + c) →  -- j is monic cubic
  s = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_problem_l903_90389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_S_union_T_equals_interval_l903_90366

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_equals_interval :
  (Set.univ \ S) ∪ T = Set.Iic (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_S_union_T_equals_interval_l903_90366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l903_90304

noncomputable def f (x : ℝ) := 3 / (x + 1)

theorem f_properties :
  let a := 3
  let b := 5
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y) ∧
  (∀ x, x ∈ Set.Icc a b → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc a b → f x ≥ f b) ∧
  f a = 3/4 ∧
  f b = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l903_90304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_interval_l903_90399

/-- The function f(x) = √3 * sin(x) + cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

/-- The monotonically increasing interval of f(x) -/
def monotonic_interval (k : ℤ) : Set ℝ := Set.Ioo (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + Real.pi / 3)

/-- Theorem: The monotonically increasing interval of f(x) is (2kπ - 2π/3, 2kπ + π/3), where k ∈ Z -/
theorem f_monotonic_interval : ∀ (k : ℤ), StrictMonoOn f (monotonic_interval k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_interval_l903_90399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_specific_circle_satisfies_conditions_l903_90311

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 3 = 0

-- Define the curve (x > 0)
def curve (x y : ℝ) : Prop := x > 0 ∧ y = -3 / x

-- Define a circle with center (a, b) and radius r
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define tangency condition
def is_tangent (a b r : ℝ) : Prop := 
  abs (3 * a - 4 * b + 3) / 5 = r

-- Define the specific circle we're proving is smallest
def specific_circle (x y : ℝ) : Prop := circle_eq x y 2 (-3/2) 3

theorem smallest_circle : 
  ∀ a b r : ℝ, 
    curve a b → 
    is_tangent a b r → 
    r ≥ 3 :=
by
  sorry

-- Prove that the specific circle satisfies the conditions
theorem specific_circle_satisfies_conditions :
  ∃ x y : ℝ,
    curve 2 (-3/2) ∧
    is_tangent 2 (-3/2) 3 ∧
    specific_circle x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_specific_circle_satisfies_conditions_l903_90311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_density_in_liquids_l903_90338

/-- The weight of an object is p. The weight of this object in a liquid with density d₁ is p₁, 
    and in a liquid with density x is p₂. This theorem proves the relation between x, d, and the given parameters. -/
theorem object_density_in_liquids 
  (p p₁ p₂ d₁ : ℝ) 
  (hp : p > 0) 
  (hp₁ : p₁ > 0) 
  (hp₂ : p₂ > 0) 
  (hd₁ : d₁ > 0) 
  (h_p_p₁ : p > p₁) 
  (h_p_p₂ : p > p₂) : 
  ∃ (d x : ℝ), 
    d = d₁ * p / (p - p₁) ∧ 
    x = d₁ * (p - p₂) / (p - p₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_density_in_liquids_l903_90338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_baking_time_l903_90387

/-- Represents the number of cupcakes sold -/
def n : ℕ → ℕ := sorry

/-- The price in cents for n cupcakes -/
def price (n : ℕ) : ℕ := (n + 20) * (n + 15)

/-- The number of cupcakes Roy can bake per hour -/
def baking_rate : ℕ := 10

/-- The price of the order in cents -/
def order_price : ℕ := 1050

theorem roy_baking_time :
  ∃ (n : ℕ), price n = order_price ∧ 
    (n : ℚ) / baking_rate * 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_baking_time_l903_90387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l903_90365

def compare_integers (a b : ℕ) : Prop :=
  (Nat.repr a).length < (Nat.repr b).length ∨
  ((Nat.repr a).length = (Nat.repr b).length ∧
    (Nat.repr a).data < (Nat.repr b).data)

theorem ascending_order :
  compare_integers 865 6503 ∧
  compare_integers 6503 8506 ∧
  compare_integers 8506 8560 :=
by
  sorry

#check ascending_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l903_90365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l903_90303

theorem sin_double_angle_special_case (θ : ℝ) 
  (h1 : Real.cos (θ + π/2) = 4/5) 
  (h2 : -π/2 < θ ∧ θ < π/2) : 
  Real.sin (2*θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l903_90303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l903_90370

def sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda : ℝ) : Prop :=
  a 1 = 1 ∧
  (∀ n, a n ≠ 0) ∧
  (∀ n, a n * a (n + 1) = lambda * S n - 1)

theorem sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda : ℝ) 
  (h : sequence_property a S lambda) : 
  (∀ n, a (n + 2) - a n = lambda) ∧
  (∃ d : ℝ, ∀ n, a (n + 1) - a n = d) ↔ lambda = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l903_90370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_neg_2_l903_90348

theorem square_of_3x_plus_4_when_x_is_neg_2 :
  ∀ x : ℝ, x = -2 → (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_neg_2_l903_90348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l903_90324

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x - 3)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≤ f y} = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l903_90324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_is_sqrt_a_squared_plus_three_l903_90376

/-- Represents the concept of an expression being in its simplest form -/
def IsSimplestForm (x : ℝ → ℝ) : Prop := sorry

/-- The given options for square roots -/
noncomputable def options : List (ℝ → ℝ) := [
  (λ _ ↦ Real.sqrt 8),
  (λ _ ↦ Real.sqrt (1/9)),
  (λ a ↦ Real.sqrt (a^2)),
  (λ a ↦ Real.sqrt (a^2 + 3))
]

/-- Statement: Among the given options, √(a^2 + 3) is in the simplest form -/
theorem simplest_form_is_sqrt_a_squared_plus_three :
  ∃ (f : ℝ → ℝ), f ∈ options ∧ IsSimplestForm f ∧ f = (λ a ↦ Real.sqrt (a^2 + 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_is_sqrt_a_squared_plus_three_l903_90376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l903_90335

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A = (1, 2) ∧
  (∃ k : ℝ, t.C.1 - 2 * t.C.2 + 1 = 0) ∧
  (∃ D : ℝ × ℝ, D = ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2) ∧
                   7 * D.1 + 5 * D.2 - 5 = 0)

-- Define the area function
noncomputable def area (t : Triangle) : ℝ :=
  let s := (dist t.A t.B + dist t.B t.C + dist t.C t.A) / 2
  Real.sqrt (s * (s - dist t.A t.B) * (s - dist t.B t.C) * (s - dist t.C t.A))

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = (5, -6) ∧
  (area t = 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l903_90335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_preserves_sum_of_squares_impossible_transformation_l903_90327

noncomputable def transform (a b : ℝ) : ℝ × ℝ :=
  ((a + b) / Real.sqrt 2, (a - b) / Real.sqrt 2)

noncomputable def sum_of_squares (triple : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := triple
  x^2 + y^2 + z^2

theorem transform_preserves_sum_of_squares (a b : ℝ) :
  let (x, y) := transform a b
  x^2 + y^2 = a^2 + b^2 := by sorry

theorem impossible_transformation :
  ¬ ∃ (steps : ℕ), ∃ (final : ℝ × ℝ × ℝ),
    (∀ i < steps, ∃ (intermediate : ℝ × ℝ × ℝ),
      (sum_of_squares intermediate = sum_of_squares (2, Real.sqrt 2, 1 / Real.sqrt 2))) ∧
    final = (1, Real.sqrt 2, 1 + Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_preserves_sum_of_squares_impossible_transformation_l903_90327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l903_90396

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = p.1 + 4

def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = p.1

-- Define area function (placeholder)
def area (s : Square) : ℝ := sorry

-- Define solution (placeholder)
def solution : ℝ := sorry

-- Main theorem
theorem square_area (ABCD : Square) 
  (hAB : on_line ABCD.A ∧ on_line ABCD.B)
  (hCD : on_parabola ABCD.C ∧ on_parabola ABCD.D) :
  area ABCD = solution := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_l903_90396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l903_90374

/-- The time (in minutes) the cyclist must wait for the hiker to catch up -/
noncomputable def waiting_time (hiker_speed cyclist_speed : ℝ) (cyclist_travel_time : ℝ) : ℝ :=
  (cyclist_speed * cyclist_travel_time / hiker_speed - cyclist_travel_time) * 60

theorem cyclist_waiting_time :
  let hiker_speed : ℝ := 5  -- miles per hour
  let cyclist_speed : ℝ := 20  -- miles per hour
  let cyclist_travel_time : ℝ := 5 / 60  -- 5 minutes converted to hours
  waiting_time hiker_speed cyclist_speed cyclist_travel_time = 20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_waiting_time_l903_90374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l903_90300

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | (p.1^2 / 4 + p.2^2 / 3) = 1}

-- Define the line l
def l (k m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + m}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the distance ratio condition
def distanceRatio (p : ℝ × ℝ) : Prop :=
  dist p F / dist p (4, p.2) = 1/2

-- Define parallelogram predicate
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2) ∧
  (c.1 - a.1 = d.1 - b.1) ∧ (c.2 - a.2 = d.2 - b.2)

-- Define triangle area function
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)) / 2

-- Main theorem
theorem constant_triangle_area 
  (k m : ℝ) 
  (hk : k ≠ 0) 
  (hl : l k m ∩ C ≠ ∅) 
  (hP : ∃ P ∈ C, isParallelogram O P (A : ℝ × ℝ) (B : ℝ × ℝ)) 
  (hA : A ∈ C ∩ l k m) 
  (hB : B ∈ C ∩ l k m) 
  (hAB : A ≠ B) :
  triangleArea O A B = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l903_90300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_in_fourth_quadrant_l903_90321

noncomputable def M : ℝ × ℝ := (2, Real.tan (300 * Real.pi / 180))

theorem M_in_fourth_quadrant : 
  M.1 > 0 ∧ M.2 < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_in_fourth_quadrant_l903_90321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_count_l903_90390

/-- The number of shelves in the library -/
def num_shelves : ℕ := 25794

/-- The number of books per shelf -/
def books_per_shelf : ℚ := 13.2

/-- The total number of books in the library -/
def total_books : ℕ := 340481

/-- Theorem stating the relationship between the number of shelves, 
    books per shelf, and the total number of books -/
theorem library_book_count : 
  ⌈(num_shelves : ℚ) * books_per_shelf⌉ = total_books := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_book_count_l903_90390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l903_90306

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem: The speed of the train is approximately 11.11 m/s -/
theorem train_speed_approx :
  let train_length : ℝ := 200
  let bridge_length : ℝ := 300
  let crossing_time : ℝ := 45
  ∃ ε > 0, |train_speed train_length bridge_length crossing_time - 11.11| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l903_90306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_closest_points_of_circles_l903_90320

/-- Given two circles with centers at (4,5) and (20,15), both tangent to the x-axis,
    the distance between their closest points is √356 - 20. -/
theorem distance_between_closest_points_of_circles : ∃ d : ℝ,
  let c1 : ℝ × ℝ := (4, 5)
  let c2 : ℝ × ℝ := (20, 15)
  let r1 : ℝ := c1.2
  let r2 : ℝ := c2.2
  let center_distance : ℝ := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  d = center_distance - r1 - r2 ∧ d = Real.sqrt 356 - 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_closest_points_of_circles_l903_90320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l903_90330

theorem unique_solution_condition (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  (∃! x : ℝ, x > 0 ∧ a^x = x^b) ↔ ∃ t : ℝ, t > 1 ∧ a = t ∧ b = Real.exp 1 * Real.log t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l903_90330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l903_90388

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 2

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 0 1, f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l903_90388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l903_90340

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := (1/2) * Real.sin (ω * x + φ)

theorem angle_C_measure 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < π) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_PQ : ∃ P Q : ℝ × ℝ, 
    (P.2 = (f ω φ P.1) ∧ Q.2 = (f ω φ Q.1)) ∧ 
    (∀ x, (f ω φ x) ≤ P.2 ∧ (f ω φ x) ≥ Q.2) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 2)
  (h_triangle : ∃ A B C : ℝ, 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    Real.sin A = 1 / Real.sqrt 2 ∧
    Real.sin B = Real.sqrt 2 / 2 ∧
    f π (π/2) (A/π) = Real.sqrt 3 / 4) :
  C = 7*π/12 ∨ C = π/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l903_90340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_c_y_coordinate_l903_90317

noncomputable section

structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

def has_vertical_symmetry (p : Pentagon) : Prop :=
  p.A.1 + p.E.1 = p.B.1 + p.D.1 ∧ p.C.1 = (p.B.1 + p.D.1) / 2

def pentagon_area (p : Pentagon) : ℝ :=
  let triangle_area := abs ((p.B.1 - p.D.1) * (p.C.2 - p.B.2)) / 2
  (p.D.1 - p.A.1) * (p.B.2 - p.A.2) + triangle_area

theorem pentagon_c_y_coordinate :
  ∀ (p : Pentagon),
    has_vertical_symmetry p →
    p.A = (0, 0) →
    p.B = (0, 6) →
    p.D = (6, 6) →
    p.E = (6, 0) →
    pentagon_area p = 100 →
    p.C.2 = 82 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_c_y_coordinate_l903_90317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_figure_center_cut_l903_90378

/-- Represents a grid-based figure -/
structure GridFigure where
  rows : ℕ
  cols : ℕ
  cells : Set (ℕ × ℕ)

/-- Represents a vertical cut on a grid -/
def VerticalCut (g : GridFigure) (col : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 < g.rows ∧ p.2 = col}

/-- The left part of a figure after a vertical cut -/
def LeftPart (g : GridFigure) (cut : ℕ) : Set (ℕ × ℕ) :=
  {p ∈ g.cells | p.2 < cut}

/-- The right part of a figure after a vertical cut -/
def RightPart (g : GridFigure) (cut : ℕ) : Set (ℕ × ℕ) :=
  {p ∈ g.cells | p.2 > cut}

/-- A figure is symmetric about a vertical line if mirroring any point
    in the left half results in a point in the right half -/
def IsSymmetricAboutVertical (g : GridFigure) (center : ℕ) : Prop :=
  ∀ p ∈ g.cells, p.2 < center →
    (p.1, 2 * center - p.2 - 1) ∈ g.cells

/-- Theorem: For a symmetric figure with an even number of columns,
    a vertical cut through the center divides it into identical parts -/
theorem symmetric_figure_center_cut
  (g : GridFigure)
  (h_even : Even g.cols)
  (h_sym : IsSymmetricAboutVertical g (g.cols / 2)) :
  LeftPart g (g.cols / 2) = RightPart g (g.cols / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_figure_center_cut_l903_90378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l903_90314

/-- Given a triangle with area t and sides in ratio 4:5:6, prove the side lengths -/
theorem triangle_side_lengths
  (t : ℝ)
  (h_t : t = 357.18)
  (a b c : ℝ)
  (h_ratio : (a : ℝ) / 4 = b / 5 ∧ b / 5 = c / 6)
  (h_area : t = (a * b * Real.sqrt (1 - (a^2 + b^2 - c^2)^2 / (4 * a^2 * b^2))) / 4)
  : ∃ (ε : ℝ), ε > 0 ∧ 
    (abs (a - 24) < ε) ∧ 
    (abs (b - 30) < ε) ∧ 
    (abs (c - 36) < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l903_90314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_calculation_l903_90343

/-- The height of a tray formed from a square paper --/
noncomputable def trayHeight (side_length : ℝ) (cut_start : ℝ) (cut_angle : ℝ) : ℝ :=
  let diagonal_cut := cut_start * Real.sqrt 2
  let pr := (cut_start + diagonal_cut) / Real.cos (cut_angle / 2)
  pr * Real.sin (cut_angle / 2)

/-- Theorem stating the height of the tray for the given conditions --/
theorem tray_height_calculation :
  trayHeight 120 (Real.sqrt 20) (π / 4) = (Real.sqrt 20 + Real.sqrt 40) * Real.sqrt (2 - Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_calculation_l903_90343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l903_90356

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : Real.cos (α + π/4) = 4/5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l903_90356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_y_l903_90310

open Real

noncomputable def y (x : ℝ) : ℝ := (log x) / (log 2 * x^3)

theorem third_derivative_y (x : ℝ) (h : x > 0) : 
  (deriv^[3] y) x = (47 - 60 * log x) / (log 2 * x^6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_y_l903_90310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l903_90341

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_polar_point_to_line (ρ : ℝ) (θ : ℝ) (a : ℝ) : ℝ :=
  |ρ * Real.sin θ - a|

/-- Theorem stating the distance from the point (2, π/6) to the line ρ sin θ = 3 is 2 --/
theorem distance_to_line_is_two :
  distance_polar_point_to_line 2 (Real.pi/6) 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l903_90341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_composition_l903_90394

-- Define the function to count the number of smaller decimal units in a larger decimal
def countDecimalUnits (larger smaller : Float) : Nat :=
  (larger / smaller).toUInt64.toNat

-- Theorem statement
theorem decimal_composition :
  (countDecimalUnits 0.4 0.1 = 4) ∧
  (countDecimalUnits 0.023 0.001 = 23) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_composition_l903_90394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_on_cone_l903_90331

/-- Given a cone with slant height 4 cm and base radius 1 cm, 
    the shortest distance an ant travels when crawling around 
    the lateral surface for one round is 4√2 cm. -/
theorem ant_path_on_cone (slant_height base_radius : ℝ) 
  (h1 : slant_height = 4)
  (h2 : base_radius = 1) : 
  Real.sqrt (2 * slant_height ^ 2) = 4 * Real.sqrt 2 := by
  sorry

#check ant_path_on_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_on_cone_l903_90331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_exponent_l903_90352

theorem linear_equation_exponent (a : ℝ) : 
  (∀ x : ℝ, ∃ m b : ℝ, x^(a-1) - 5 = m*x + b) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_exponent_l903_90352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_circumference_approx_l903_90339

/-- The circumference of a semicircle with diameter equal to the side of a square,
    where the square's perimeter is equal to the perimeter of a rectangle with
    length 18 cm and breadth 10 cm. -/
noncomputable def semicircle_circumference : ℝ :=
  let rectangle_length : ℝ := 18
  let rectangle_breadth : ℝ := 10
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_breadth)
  let square_side : ℝ := rectangle_perimeter / 4
  (Real.pi * square_side) / 2 + square_side

/-- Theorem stating that the semicircle_circumference is approximately 36.02 -/
theorem semicircle_circumference_approx :
  ‖semicircle_circumference - 36.02‖ < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_circumference_approx_l903_90339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_valid_arrangement_l903_90359

/-- A type representing the faces of a dodecahedron -/
def DodecahedronFace := Fin 12

/-- A function type representing an arrangement of numbers on a dodecahedron -/
def Arrangement := DodecahedronFace → Fin 12

/-- Predicate to check if two faces are adjacent or opposite on a dodecahedron -/
def are_adjacent_or_opposite (f1 f2 : DodecahedronFace) : Prop :=
  sorry

/-- Predicate to check if two numbers are consecutive (including 12 and 1) -/
def are_consecutive (n1 n2 : Fin 12) : Prop :=
  (n1 = n2 + 1) ∨ (n2 = n1 + 1) ∨ (n1 = 0 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 0)

/-- Predicate to check if an arrangement is valid -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  (∀ f1 f2 : DodecahedronFace, f1 ≠ f2 → arr f1 ≠ arr f2) ∧
  (∀ f1 f2 : DodecahedronFace, are_adjacent_or_opposite f1 f2 →
    ¬are_consecutive (arr f1) (arr f2))

/-- The number of valid arrangements -/
noncomputable def num_valid_arrangements : ℕ :=
  sorry

/-- The total number of possible arrangements -/
def total_arrangements : ℕ :=
  Nat.factorial 11

/-- The main theorem: probability of a valid arrangement -/
theorem probability_valid_arrangement :
  (num_valid_arrangements : ℚ) / (total_arrangements : ℚ) =
  (num_valid_arrangements : ℚ) / (Nat.factorial 11 : ℚ) := by
  sorry

#eval Nat.factorial 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_valid_arrangement_l903_90359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l903_90371

/-- The length of a chord in an ellipse -/
theorem ellipse_chord_length (x y : ℝ) (A B : ℝ × ℝ) : 
  (∀ x y, x^2 / 9 + y^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c = 2 * Real.sqrt 2) →  -- Distance from center to focus
  (∃ m : ℝ, m = Real.sqrt 3 / 3) →  -- Slope of the line
  (∃ F₁ : ℝ × ℝ, F₁ = (-2 * Real.sqrt 2, 0)) →  -- Left focus coordinates
  (∀ t : ℝ, 
    let (x, y) := A
    x = -2 * Real.sqrt 2 + t * 1 ∧ y = t * Real.sqrt 3 / 3) →  -- Line equation passing through F₁
  (∀ t : ℝ, 
    let (x, y) := B
    x = -2 * Real.sqrt 2 + t * 1 ∧ y = t * Real.sqrt 3 / 3) →  -- Line equation passing through F₁
  (let (x₁, y₁) := A; x₁^2 / 9 + y₁^2 = 1) →  -- A is on the ellipse
  (let (x₂, y₂) := B; x₂^2 / 9 + y₂^2 = 1) →  -- B is on the ellipse
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2  -- Length of chord AB is 2
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l903_90371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l903_90369

def word : String := "EXAMPLE"

def valid_sequence (s : List Char) : Bool :=
  s.length = 4 &&
  s.head? = some 'X' &&
  s.getLast? ≠ some 'E' &&
  s.toFinset ⊆ word.toList.toFinset &&
  s.toFinset.card = 4

theorem distinct_sequences_count :
  (List.filter valid_sequence (List.permutations word.toList)).length = 80 := by
  sorry

#eval (List.filter valid_sequence (List.permutations word.toList)).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sequences_count_l903_90369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l903_90318

-- Define the function f(x) = 2x - 4 + 3^x
noncomputable def f (x : ℝ) := 2 * x - 4 + (3 : ℝ)^x

-- State the theorem
theorem zero_in_interval :
  (f (1/2) < 0) → (f 1 > 0) → ∃ c ∈ Set.Ioo (1/2) 1, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l903_90318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_factorization_l903_90308

/-- Represents a polynomial in two variables -/
structure MyPolynomial (α : Type) where
  coeffs : List (α × ℕ × ℕ)  -- List of (coefficient, x exponent, y exponent)

/-- Checks if a polynomial can be factored as a square of a binomial -/
def isSquareOfBinomial (p : MyPolynomial ℚ) : Prop :=
  ∃ a b : ℚ, p.coeffs = [(a^2, 2, 0), (2*a*b, 1, 1), (b^2, 0, 2)] ∨
              p.coeffs = [(a^2, 2, 0), (-2*a*b, 1, 1), (b^2, 0, 2)]

/-- The given polynomials -/
def polyA : MyPolynomial ℚ := ⟨[(1, 2, 0), (9, 0, 2)]⟩
def polyB : MyPolynomial ℚ := ⟨[(3, 2, 0), (-9, 0, 1)]⟩
def polyC : MyPolynomial ℚ := ⟨[(-1/4, 2, 0), (1/9, 0, 2)]⟩
def polyD : MyPolynomial ℚ := ⟨[(-1/4, 2, 0), (-1/9, 0, 2)]⟩

/-- The main theorem -/
theorem square_of_binomial_factorization :
  isSquareOfBinomial polyC ∧
  ¬isSquareOfBinomial polyA ∧
  ¬isSquareOfBinomial polyB ∧
  ¬isSquareOfBinomial polyD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_binomial_factorization_l903_90308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_m_l903_90395

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := |x - 2*a| + |x + 1/a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | |x - 2| + |x + 1| > 4} = {x : ℝ | x < -3/2 ∨ x > 5/2} := by sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  (∀ x a : ℝ, f x a ≥ m^2 - m + 2*Real.sqrt 2) → 0 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_m_l903_90395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l903_90334

/-- The constant term in the binomial expansion of (1+2x^2)(x - 1/x)^8 -/
def constant_term : ℤ := -42

/-- The binomial expansion of (1+2x^2)(x - 1/x)^8 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (1 + 2*x^2) * (x - 1/x)^8

/-- Theorem stating that the constant term of the binomial expansion is -42 -/
theorem constant_term_proof :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = binomial_expansion x) ∧ 
  (f 0 = constant_term) := by
  sorry

#check constant_term_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l903_90334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l903_90393

/-- Definition of the ellipse C -/
noncomputable def ellipse_C : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 / 9 + y^2 = 1}

/-- The ellipse C passes through the given points -/
axiom passes_through_points :
  (-1, 2 * Real.sqrt 2 / 3) ∈ ellipse_C ∧ (2, Real.sqrt 5 / 3) ∈ ellipse_C

/-- Definition of the top vertex B -/
def top_vertex : ℝ × ℝ := (0, 1)

/-- Definition of perpendicular lines through B -/
def perpendicular_lines (k : ℝ) : (Set (ℝ × ℝ)) × (Set (ℝ × ℝ)) :=
  ({p | ∃ (x y : ℝ), p = (x, y) ∧ y = k * x + 1},
   {p | ∃ (x y : ℝ), p = (x, y) ∧ y = -1/k * x + 1})

/-- Definition of the intersection points P and Q -/
noncomputable def intersection_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-18*k / (1 + 9*k^2), (1 - 9*k^2) / (1 + 9*k^2)),
   (18*k / (9 + k^2), (k^2 - 9) / (9 + k^2)))

/-- The main theorem to be proved -/
theorem ellipse_C_properties :
  (∀ (x y : ℝ), (x, y) ∈ ellipse_C ↔ x^2 / 9 + y^2 = 1) ∧
  (∀ (k : ℝ), k ≠ 0 →
    let (P, Q) := intersection_points k
    (0, -4/5) ∈ {p | ∃ (t : ℝ), p = (1-t) • P + t • Q}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_C_properties_l903_90393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l903_90319

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x - 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 8} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l903_90319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l903_90373

open Real Matrix

theorem determinant_max_value :
  let A : ℝ → Matrix (Fin 3) (Fin 3) ℝ := λ θ => !![1, 1, 1;
                                                    1, 1 + sin θ ^ 2, 1;
                                                    1 + cos θ ^ 2, 1, 1]
  (∀ θ, det (A θ) ≤ (3 : ℝ) / 4) ∧ ∃ θ₀, det (A θ₀) = (3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l903_90373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l903_90342

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + Real.sqrt x - 1

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (x > 0) → (f x > f (2*x - 4) ↔ x > 2 ∧ x < 4) :=
by
  sorry

#check solution_set_of_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l903_90342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_recorded_positive_l903_90351

/-- Represents the amount of money in yuan -/
def Amount := ℤ

/-- Records a financial transaction -/
noncomputable def record (amount : ℝ) : Amount :=
  if amount < 0 then Int.floor amount else Int.ceil amount

/-- Theorem: If spending is recorded as negative, income is recorded as positive -/
theorem income_recorded_positive (spending income : ℝ) 
  (h : record (-spending) = -Int.ceil spending) :
  record income = Int.ceil income :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_recorded_positive_l903_90351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_expression_close_to_approximation_l903_90380

-- Define the approximations for √2 and √3
noncomputable def sqrt2_approx : ℝ := 1.414
noncomputable def sqrt3_approx : ℝ := 1.732

-- Define the expression to be calculated
noncomputable def expression : ℝ := Real.sqrt 0.08 + (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2)

-- Theorem statement
theorem expression_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |expression - 1.28| < ε := by
  sorry

-- Additional helper theorem to show the approximation is close to the actual value
theorem expression_close_to_approximation :
  ∃ (δ : ℝ), δ > 0 ∧ δ < 0.01 ∧ |expression - (sqrt2_approx / 5 + 1)| < δ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_expression_close_to_approximation_l903_90380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l903_90372

/-- Sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The sum of the geometric series -2, -6, -18, -54, -162, -486, -1458 is -2186 -/
theorem geometric_series_sum :
  geometricSum (-2 : ℝ) 3 7 = -2186 := by
  -- Unfold the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [Real.rpow_nat_cast]
  -- Perform numerical computation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l903_90372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l903_90323

theorem solve_exponential_equation (x : ℝ) : (1000 : ℝ)^(5/2) = 10^x → x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l903_90323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coeff_sum_quadratic_l903_90368

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  b : ℤ
  c : ℤ

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (qf : QuadraticFunction) (x : ℝ) : ℝ :=
  x^2 + qf.b * x + qf.c

/-- The composition f(f(x)) -/
def f_comp_f (qf : QuadraticFunction) (x : ℝ) : ℝ :=
  f qf (f qf x)

/-- Predicate: f(f(x)) = 0 has four distinct real roots in arithmetic sequence -/
def has_four_roots_in_arithmetic_seq (qf : QuadraticFunction) : Prop :=
  ∃ (a d : ℝ), d ≠ 0 ∧ 
    (∀ x : ℝ, f_comp_f qf x = 0 ↔ x = a - d ∨ x = a ∨ x = a + d ∨ x = a + 2*d)

/-- The sum of coefficients of f(x) -/
def coeff_sum (qf : QuadraticFunction) : ℤ :=
  1 + qf.b + qf.c

/-- The main theorem -/
theorem min_coeff_sum_quadratic :
  ∃ qf : QuadraticFunction, 
    has_four_roots_in_arithmetic_seq qf ∧
    (∀ qf' : QuadraticFunction, has_four_roots_in_arithmetic_seq qf' → 
      coeff_sum qf ≤ coeff_sum qf') ∧
    qf.b = 22 ∧ qf.c = 105 := by
  sorry

#check min_coeff_sum_quadratic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coeff_sum_quadratic_l903_90368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90392

noncomputable def m (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x) + Real.sin (ω * x), Real.cos (ω * x))

noncomputable def n (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x))

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

def is_periodic (g : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, g (x + T) = g x

def arithmetic_sequence (a b c : ℝ) : Prop := 2 * b = a + c

theorem problem_solution (ω : ℝ) (A B C : ℝ) (a b c : ℝ) 
  (h_ω : ω > 0)
  (h_periodic : is_periodic (f ω) π)
  (h_arithmetic : arithmetic_sequence a b c)
  (h_f_B : f ω B = 1) :
  ω = 1 ∧ A = B ∧ B = C ∧ a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l903_90392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_shift_roots_l903_90305

-- Define the polynomial P(x)
noncomputable def P : ℝ → ℝ := sorry

-- Define the degree of P(x)
def degree_P : ℕ := sorry

-- Define the set of roots of P(x)
noncomputable def roots_P : Finset ℤ := sorry

-- Theorem statement
theorem polynomial_shift_roots
  (h1 : degree_P > 5)
  (h2 : Finset.card roots_P = degree_P)
  (h3 : ∀ (x : ℤ), x ∈ roots_P → P (↑x) = 0)
  (h4 : ∀ (x y : ℤ), x ∈ roots_P → y ∈ roots_P → x ≠ y → P (↑x) ≠ P (↑y)) :
  ∃ (roots_P_plus_3 : Finset ℝ),
    Finset.card roots_P_plus_3 = degree_P ∧
    (∀ (x : ℝ), x ∈ roots_P_plus_3 → P x + 3 = 0) ∧
    (∀ (x y : ℝ), x ∈ roots_P_plus_3 → y ∈ roots_P_plus_3 → x ≠ y → P x + 3 ≠ P y + 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_shift_roots_l903_90305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_l903_90391

noncomputable section

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := x^2 / 2
def line (x y : ℝ) : Prop := 3 * x - 2 * y - 2 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 = p.2 ∧ line p.1 p.2}

-- Define the slope of the tangent line to the parabola at a point
def tangent_slope (x : ℝ) : ℝ := x

-- Define the slope of the given line
def line_slope : ℝ := 3 / 2

-- Define the angle between two slopes
def angle_between_slopes (m1 m2 : ℝ) : ℝ :=
  Real.arctan (abs ((m2 - m1) / (1 + m1 * m2)))

-- Theorem statement
theorem intersection_angles :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧
  (angle_between_slopes (tangent_slope p1.1) line_slope = Real.arctan (1 / 5) ∨
   angle_between_slopes (tangent_slope p1.1) line_slope = Real.arctan (1 / 8)) ∧
  (angle_between_slopes (tangent_slope p2.1) line_slope = Real.arctan (1 / 5) ∨
   angle_between_slopes (tangent_slope p2.1) line_slope = Real.arctan (1 / 8)) ∧
  angle_between_slopes (tangent_slope p1.1) line_slope ≠
  angle_between_slopes (tangent_slope p2.1) line_slope :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_l903_90391
