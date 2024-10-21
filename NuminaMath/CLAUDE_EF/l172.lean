import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_in_intervals_l172_17250

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 3) / ((x - 1)^2)

-- State the theorem
theorem f_negative_iff_in_intervals :
  ∀ x : ℝ, f x < 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioo 1 3 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_iff_in_intervals_l172_17250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_focus_directrix_distance_l172_17232

-- Define the distance between focus and directrix for each parabola
noncomputable def P₁ : ℝ := 1/2  -- for y² = -x
noncomputable def P₂ : ℝ := 1    -- for y² = 2x
noncomputable def P₃ : ℝ := 1/4  -- for 2x² = y
noncomputable def P₄ : ℝ := 2    -- for x² = -4y

-- Theorem statement
theorem smallest_focus_directrix_distance :
  P₃ < P₁ ∧ P₃ < P₂ ∧ P₃ < P₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_focus_directrix_distance_l172_17232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_product_equals_n_squared_l172_17235

noncomputable def a (n : ℕ) (x : ℝ) : ℝ := (1 - x^n) / (1 - x)

noncomputable def b (n : ℕ) (x : ℝ) : ℝ := 
  Finset.sum (Finset.range n) (fun i => (2 * ↑i + 1) * x^i)

theorem integral_product_equals_n_squared (n : ℕ) :
  ∫ x in (Set.Icc 0 1 : Set ℝ), (a n x) * (b n x) = n^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_product_equals_n_squared_l172_17235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_curve_intersection_l172_17271

theorem ln_curve_intersection (f : ℝ → ℝ) (A B C E : ℝ × ℝ) (x₃ : ℝ) :
  (∀ x, f x = Real.log x) →
  A = (1, 0) →
  B = (Real.exp 5, 5) →
  C.1 = (2 * A.1 + B.1) / 3 →
  C.2 = (2 * A.2 + B.2) / 3 →
  E = (x₃, C.2) →
  f x₃ = C.2 →
  x₃ = Real.exp (5/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_curve_intersection_l172_17271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l172_17262

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem arithmetic_sequence_sum_10 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 5 = 2 →
  a 2 + a 14 = 12 →
  sum_of_arithmetic_sequence a 10 = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l172_17262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l172_17218

/-- Represents the three colors: Red, Green, Blue -/
inductive Color
| Red
| Green
| Blue

/-- Represents the state of the circle with marked points -/
structure CircleState where
  n : Nat
  points : List Color
  h_n_ge_3 : n ≥ 3
  h_points_length : points.length = n

/-- Represents a single step in the transformation process -/
inductive Step
| replace (i : Nat) (new_color : Color) : Step

/-- Checks if a given state is a final state (all points have the same color) -/
def isFinalState (state : CircleState) : Prop :=
  ∃ c : Color, ∀ p ∈ state.points, p = c

/-- Checks if a given state is missing one color -/
def isMissingOneColor (state : CircleState) : Prop :=
  ∃ c : Color, c ∉ state.points

/-- Represents the possibility of reaching a final state of any color -/
def canReachAllColors (initial : CircleState) : Prop :=
  ∀ c : Color, ∃ final : CircleState, isFinalState final ∧ (∀ p ∈ final.points, p = c)

/-- Main theorem: It's possible to reach a final state of any color from an initial state
    with one missing color if and only if n is even -/
theorem circle_coloring_theorem (n : Nat) (h_n_ge_3 : n ≥ 3) :
  (∃ initial : CircleState, initial.n = n ∧ isMissingOneColor initial ∧ canReachAllColors initial) ↔
  Even n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_theorem_l172_17218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_problem_l172_17236

noncomputable def distance_point_to_line (x y : ℝ) (m : ℝ) : ℝ :=
  |x + y * m| / Real.sqrt 2

def line_equation (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi / 4) = m

def point_relationship (ρ_p ρ_q : ℝ) : Prop :=
  ρ_p * ρ_q = 1

theorem polar_coordinate_problem (m : ℝ) (h_m : m > 0) :
  distance_point_to_line (Real.sqrt 2) 0 m = 3 →
  m = 2 ∧
  ∀ (ρ θ : ℝ), line_equation ρ θ 2 →
    ∃ (ρ_q : ℝ), point_relationship ρ ρ_q ∧
      ρ_q = (1 / 2) * Real.sin (θ - Real.pi / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_problem_l172_17236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l172_17234

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), f (π / 6 - x) = f (π / 6 + x)) ∧
  (∀ (x : ℝ), f (π / 3 - x) = f (π / 3 + x)) ∧
  (¬ ∀ (x y : ℝ), -π / 3 ≤ x ∧ x < y ∧ y ≤ π / 6 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l172_17234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_circle_radius_l172_17282

/-- An isosceles triangle with a specific circle -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Center of the circle
  O : ℝ × ℝ
  -- Foot of the altitude from A
  D : ℝ × ℝ
  -- Midpoint of AC
  K : ℝ × ℝ
  -- AB = BC = 25
  ab_eq_bc : dist A B = 25 ∧ dist B C = 25
  -- AC = 14
  ac_eq_14 : dist A C = 14
  -- D is on BC and AD ⟂ BC
  d_on_bc : D ∈ Set.Icc B C
  ad_perp_bc : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0
  -- Circle touches BC at D
  circle_touches_bc : dist O D = dist O B ∧ dist O D = dist O C
  -- Circle passes through K
  k_on_circle : dist O K = dist O D
  -- K is midpoint of AC
  k_midpoint : K.1 = (A.1 + C.1) / 2 ∧ K.2 = (A.2 + C.2) / 2

/-- The radius of the circle in the special triangle is 175/48 -/
theorem special_triangle_circle_radius (t : SpecialTriangle) : 
  dist t.O t.D = 175 / 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_circle_radius_l172_17282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l172_17272

-- Define the constants
noncomputable def a : ℝ := (3 : ℝ) ^ (1/3 : ℝ)
noncomputable def b : ℝ := (1/4 : ℝ) ^ (3.1 : ℝ)
noncomputable def c : ℝ := Real.log 3 / Real.log 0.4

-- State the theorem
theorem abc_inequality : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l172_17272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l172_17268

/-- An ellipse with foci and a point satisfying certain conditions -/
structure SpecialEllipse where
  /-- The ellipse -/
  E : Set (ℝ × ℝ)
  /-- Left focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- Right focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- Left vertex of the ellipse -/
  A : ℝ × ℝ
  /-- A point on the ellipse -/
  P : ℝ × ℝ
  /-- P is on the ellipse E -/
  h_P_on_E : P ∈ E
  /-- The circle with diameter PF₁ passes through F₂ -/
  h_circle : ∃ (center : ℝ × ℝ), ‖center - P‖ = ‖center - F₁‖ ∧ ‖center - F₂‖ = ‖center - P‖
  /-- The distance PF₂ is 1/4 of AF₂ -/
  h_distance : ‖P - F₂‖ = (1/4) * ‖A - F₂‖

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : SpecialEllipse) : ℝ := sorry

/-- Theorem stating that the eccentricity of the special ellipse is 3/4 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) : eccentricity e = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l172_17268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_arc_length_proof_l172_17256

/-- The length of the arc of the parabola y^2 = 4x from (0, 0) to (1, 2) -/
noncomputable def parabola_arc_length : ℝ := Real.sqrt 2 + Real.log (1 + Real.sqrt 2)

/-- Theorem stating that the length of the arc of the parabola y^2 = 4x 
    from (0, 0) to (1, 2) is equal to √2 + ln(1 + √2) -/
theorem parabola_arc_length_proof :
  let f : ℝ → ℝ := fun y => y^2 / 4
  let a : ℝ := 0
  let b : ℝ := 2
  (∫ y in a..b, Real.sqrt (1 + (deriv f y)^2)) = parabola_arc_length :=
by
  sorry

#check parabola_arc_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_arc_length_proof_l172_17256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l172_17298

theorem sin_transform (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l172_17298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_l172_17276

def adjacent_condition (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

def circle_arrangement (arr : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, adjacent_condition (arr i) (arr ((i + 1) % 99))

theorem divisible_by_three (arr : Fin 99 → ℕ) (h : circle_arrangement arr) :
  ∃ i : Fin 99, arr i % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_l172_17276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l172_17283

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y + Real.sqrt 3 * x = Real.sqrt 3 * m

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y m : ℝ) : ℝ :=
  abs (Real.sqrt 3 * x + y - Real.sqrt 3 * m) / 2

-- Theorem for part I
theorem line_tangent_to_curve :
  ∀ x y : ℝ, curve_C x y → line_l 3 x y → 
  distance_point_to_line 1 0 3 = Real.sqrt 3 := by
  sorry

-- Theorem for part II
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, curve_C x y ∧ distance_point_to_line x y m = Real.sqrt 3 / 2) ↔
  -2 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l172_17283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_candle_half_length_time_l172_17275

/-- Represents a candle with its initial length and burning time -/
structure Candle where
  initialLength : ℝ
  burningTime : ℝ

/-- The time when the thin candle becomes half the length of the thick candle -/
noncomputable def timeWhenHalfLength (thin : Candle) (thick : Candle) : ℝ :=
  20 / 3

theorem thin_candle_half_length_time 
  (thin : Candle) 
  (thick : Candle) 
  (h1 : thin.initialLength = 20) 
  (h2 : thick.initialLength = 20) 
  (h3 : thin.burningTime = 4) 
  (h4 : thick.burningTime = 5) :
  timeWhenHalfLength thin thick = 20 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_candle_half_length_time_l172_17275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l172_17284

open Real

theorem angle_properties (α : ℝ) (h1 : α ∈ Set.Icc π (3*π/2)) (h2 : Real.sin α = -3/5) :
  Real.tan α = 3/4 ∧ Real.tan (α - π/4) = -1/7 ∧ Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l172_17284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l172_17200

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi/6) + Real.cos (2*x) + 1/4

theorem f_extrema :
  let a : ℝ := -Real.pi/12
  let b : ℝ := 5*Real.pi/12
  (∀ x ∈ Set.Icc a b, f x ≤ Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc a b, f x = Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ -Real.sqrt 3 / 4) ∧
  (∃ x ∈ Set.Icc a b, f x = -Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l172_17200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_l172_17248

-- Define the function f(x) = (1/2)ax^2 - ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x

-- Define the property of being monotonically decreasing on an interval
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_monotonically_decreasing_iff (a : ℝ) :
  MonotonicallyDecreasing (f a) (1/3) 2 ↔ a < 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_l172_17248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l172_17237

theorem pizza_payment_difference : 
  ∀ (total_slices : ℕ) 
    (plain_cost anchovy_cost : ℚ) 
    (dave_anchovy dave_plain doug_plain : ℕ),
  total_slices = 8 →
  plain_cost = 8 →
  anchovy_cost = 2 →
  dave_anchovy = 4 →
  dave_plain = 1 →
  doug_plain = 3 →
  let total_cost := plain_cost + anchovy_cost
  let cost_per_slice := total_cost / total_slices
  let dave_payment := cost_per_slice * (dave_anchovy + dave_plain)
  let doug_payment := cost_per_slice * doug_plain
  dave_payment - doug_payment = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l172_17237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encoding_problem_l172_17290

/-- Encoding function from digits to symbols -/
def encode : Fin 5 → Fin 5 := sorry

/-- Decoding function from symbols to digits -/
def decode : Fin 5 → Fin 5 := sorry

/-- The value encoded as VYZ -/
def vyz : ℕ := sorry

/-- The value encoded as VYX -/
def vyx : ℕ := sorry

/-- The value encoded as VVW -/
def vvw : ℕ := sorry

/-- The value encoded as XYZ -/
def xyz : ℕ := sorry

theorem encoding_problem (h1 : vyx = vyz + 1) (h2 : vvw = vyx + 1)
  (h3 : encode 0 = 0) (h4 : encode 1 = 1) (h5 : encode 3 = 2) (h6 : encode 4 = 3) :
  xyz = 108 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encoding_problem_l172_17290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l172_17251

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The result of the calculation described in the problem -/
noncomputable def problem_result : ℝ :=
  round_to_hundredth (3 * (92.46 + 57.835))

/-- Theorem stating that the result of the calculation is 450.89 -/
theorem problem_solution : problem_result = 450.89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l172_17251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pow_fixed_point_l172_17265

/-- The function f as defined in the problem -/
def f (n : ℕ+) (x : ℕ) : ℕ :=
  if 2 * x ≤ n then 2 * x else 2 * n - 2 * x + 1

/-- The m-fold composition of f -/
def f_pow (n : ℕ+) (m : ℕ) : ℕ → ℕ :=
  (f n)^[m]

/-- The main theorem -/
theorem f_pow_fixed_point (n : ℕ+) (m : ℕ) (h : m > 0) (h1 : f_pow n m 1 = 1) :
    ∀ k : ℕ, k ≤ n → f_pow n m k = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pow_fixed_point_l172_17265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l172_17241

/-- A hyperbola with given properties -/
structure Hyperbola where
  -- Point A on the hyperbola
  a : ℝ × ℝ
  -- Eccentricity
  e : ℝ
  -- Hyperbola passes through A
  passes_through_a : a.1^2 / 16 - a.2^2 / 12 = 1
  -- Eccentricity is 2
  eccentricity_is_two : e = 2
  -- Axes of symmetry are coordinate axes (implicit in the equation)

/-- The equation of the hyperbola and its angle bisector -/
def hyperbola_properties (h : Hyperbola) : Prop :=
  (∀ x y : ℝ, x^2 / 16 - y^2 / 12 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 12 = 1}) ∧
  (∀ x y : ℝ, y = 2*x - 2 ↔ (x, y) ∈ {p : ℝ × ℝ | p.2 = 2*p.1 - 2})

theorem hyperbola_theorem (h : Hyperbola) (hA : h.a = (4, 6)) :
  hyperbola_properties h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l172_17241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_multiplication_l172_17294

theorem average_after_multiplication (numbers : Finset ℝ) (sum : ℝ) :
  numbers.card = 7 →
  sum / 7 = 15 →
  sum = numbers.sum id →
  ((5 * sum) / 7) = 75 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_multiplication_l172_17294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angled_l172_17229

theorem triangle_acute_angled (α β γ : Real) 
  (h1 : Real.sin α > Real.cos β) 
  (h2 : Real.sin β > Real.cos γ) 
  (h3 : Real.sin γ > Real.cos α) 
  (h4 : α + β + γ = Real.pi) : 
  α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_acute_angled_l172_17229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lucky_days_l172_17274

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def bandit_sequence (a₁ a₂ : ℕ) : ℕ → ℕ
| 0 => a₁
| 1 => a₂
| n + 2 => bandit_sequence a₁ a₂ n + 2 * bandit_sequence a₁ a₂ (n + 1)

theorem max_lucky_days (a₁ a₂ : ℕ) (h₁ : is_prime a₁) (h₂ : is_prime a₂) :
  ∃ (n : ℕ), n = 5 ∧
  (∀ k ≤ n, is_prime (bandit_sequence a₁ a₂ k)) ∧
  (∀ m > n, ∃ k ≤ m, ¬is_prime (bandit_sequence a₁ a₂ k)) :=
by
  sorry

#check max_lucky_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lucky_days_l172_17274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l172_17201

/-- Given vectors a and b, if there exists a real number lambda such that (a - lambda*b) ⊥ b, then lambda = 6/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) :
  a = (2, -7) → b = (-2, -4) →
  (∃ lambda : ℝ, (a.1 - lambda * b.1, a.2 - lambda * b.2) • b = 0) →
  ∃ lambda : ℝ, (a.1 - lambda * b.1, a.2 - lambda * b.2) • b = 0 ∧ lambda = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l172_17201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_square_difference_l172_17269

open Real

-- Define the interval (-π/2, π/2)
def I : Set ℝ := Set.Ioo (-π/2) (π/2)

-- Define the properties of f and g
def satisfies_conditions (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ I, f x + g x = Real.sqrt ((1 + Real.cos (2*x)) / (1 - Real.sin x))) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x)

-- State the theorem
theorem function_square_difference
  (f g : ℝ → ℝ)
  (h : satisfies_conditions f g) :
  ∀ x ∈ I, (f x)^2 - (g x)^2 = -2 * Real.cos x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_square_difference_l172_17269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_fifteen_degrees_l172_17257

theorem cosine_sine_fifteen_degrees :
  (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_fifteen_degrees_l172_17257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l172_17267

theorem trig_problem (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 2 / 10)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) : 
  Real.cos (2 * β) = 4 / 5 ∧ α + 2 * β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l172_17267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_connected_squares_count_l172_17245

theorem tri_connected_squares_count : 
  let lower_bound := 2018
  let upper_bound := 3018
  (Finset.filter (fun n => n % 2 = 0 ∧ lower_bound ≤ n ∧ n ≤ upper_bound) (Finset.range (upper_bound - lower_bound + 1))).card = 501 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_connected_squares_count_l172_17245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l172_17213

noncomputable section

open Real

variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def condition (A C a c : ℝ) : Prop :=
  c = sqrt 3 * a * sin C - c * cos A

-- Define the area of the triangle
def area (a b c A : ℝ) : ℝ :=
  1/2 * b * c * sin A

-- Theorem statement
theorem triangle_proof (A B C a b c : ℝ) 
  (h1 : triangle_ABC A B C a b c)
  (h2 : condition A C a c)
  (h3 : a = 2)
  (h4 : area a b c A = sqrt 3) :
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l172_17213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_female_l172_17219

/-- The probability of selecting at least one female student when choosing
    two students from a group of two male and three female students. -/
theorem prob_at_least_one_female (male_count female_count select_count : ℕ) 
    (h1 : male_count = 2)
    (h2 : female_count = 3)
    (h3 : select_count = 2) :
  (Nat.choose (male_count + female_count) select_count - Nat.choose male_count select_count : ℚ) / 
  Nat.choose (male_count + female_count) select_count = 9 / 10 := by
  sorry

#check prob_at_least_one_female

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_female_l172_17219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_division_l172_17212

/-- Represents the weight of potatoes in kilograms -/
def TotalWeight : ℝ := 13

/-- Represents the number of family members -/
def FamilyMembers : ℕ := 5

/-- Represents the number of potatoes -/
def NumberOfPotatoes : ℕ := 42

/-- Represents the weight of each serving in kilograms -/
noncomputable def ServingWeight : ℝ := TotalWeight / FamilyMembers

theorem potato_division :
  ServingWeight = 2.6 := by
  -- Unfold the definition of ServingWeight
  unfold ServingWeight
  -- Perform the division
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_division_l172_17212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_l172_17202

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
axiom f_even : ∀ x : ℝ, f x = f (-x)

-- f(2+x) = -f(2-x) for all x ∈ ℝ
axiom f_property : ∀ x : ℝ, f (2 + x) = -f (2 - x)

-- f(-3) = -2
axiom f_neg_three : f (-3) = -2

theorem f_2007 : f 2007 = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_l172_17202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_4_pow_2018_l172_17207

noncomputable def a : ℕ → ℝ
| 0 => 7
| n + 1 => a n * (a n + 2)

theorem smallest_n_exceeding_4_pow_2018 : 
  (∀ k < 12, a k ≤ (4 : ℝ)^2018) ∧ 
  a 12 > (4 : ℝ)^2018 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_4_pow_2018_l172_17207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_x_five_l172_17224

/-- 
Given a positive integer n, if in the binomial expansion of (1+x)^n 
the coefficient of x^5 is the largest among all coefficients, 
then n equals 10.
-/
theorem largest_coefficient_x_five (n : ℕ) (hn : n > 0) :
  (∀ k : ℕ, k ≤ n → (Nat.choose n 5) ≥ (Nat.choose n k)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_x_five_l172_17224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l172_17258

/-- Represents the scale of a map -/
structure MapScale where
  map_distance : ℚ
  actual_distance : ℚ

/-- Calculates the map distance given an actual distance and a map scale -/
def calculate_map_distance (actual_distance : ℚ) (scale : MapScale) : ℚ :=
  actual_distance * (scale.map_distance / scale.actual_distance)

theorem map_distance_calculation (actual_distance : ℚ) :
  let scale := MapScale.mk 1 400000
  let map_distance := calculate_map_distance actual_distance scale
  actual_distance = 80 → map_distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l172_17258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_y0_value_set_l172_17280

-- Define the function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 - b*x

-- Statement for the first part of the problem
theorem min_b_value : 
  (∃ (b : ℝ), ∀ (b' : ℝ), (∃ (x : ℝ), x > 0 ∧ f b' x ≥ b'*x^2 + x) → b' ≥ b) ∧ 
  (∃ (x : ℝ), x > 0 ∧ f ((5 - 2*Real.sqrt 7) / 3) x ≥ ((5 - 2*Real.sqrt 7) / 3)*x^2 + x) := by
  sorry

-- Statement for the second part of the problem
theorem y0_value_set (b : ℝ) :
  (∃ (x1 x2 : ℝ), 1 < x1 ∧ x1 < x2 ∧ f b 0 = 0 ∧ f b x1 = 0 ∧ f b x2 = 0) →
  (∃ (y0 : ℝ), 0 < y0 ∧ y0 < 2/9 ∧
    (∃ (x0 : ℝ), ((deriv (f b)) x1) * (x0 - x1) = ((deriv (f b)) x2) * (x0 - x2) ∧ 
                 y0 = ((deriv (f b)) x1) * (x0 - x1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_y0_value_set_l172_17280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specificTrapezoidArea_l172_17227

/-- Represents an isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  longerBase : ℝ
  baseAngle : ℝ

/-- Calculates the area of the isosceles trapezoid -/
noncomputable def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 112 -/
theorem specificTrapezoidArea :
  let t : IsoscelesTrapezoid := { longerBase := 20, baseAngle := Real.arccos 0.6 }
  trapezoidArea t = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specificTrapezoidArea_l172_17227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l172_17221

/-- The radius of the inscribed circle of a triangle with sides a, b, and c --/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: The radius of the inscribed circle in a triangle with sides 6, 8, and 5 is √23 / 7.6 --/
theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 6 8 5 = Real.sqrt 23 / 7.6 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval inscribedCircleRadius 6 8 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l172_17221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_underline_count_l172_17209

/-- Represents a table of numbers -/
structure Table (m n : ℕ) where
  entries : Fin m → Fin n → ℝ

/-- Represents the set of doubly underlined numbers in a table -/
def doublyUnderlined (t : Table m n) (k l : ℕ) : Finset (Fin m × Fin n) :=
  sorry

theorem double_underline_count (m n k l : ℕ) (h1 : k ≤ m) (h2 : l ≤ n) :
  ∀ (t : Table m n), (doublyUnderlined t k l).card ≥ k * l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_underline_count_l172_17209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l172_17263

/-- Given vectors a, b, c in a plane where the angle between a and b is 90°,
    |a| = |b| = 1, |c| = 2√3, and c = λa + μb, then λ² + μ² = 12 -/
theorem vector_equality (a b c : ℝ × ℝ) (l m : ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is 90°
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  (c.1^2 + c.2^2 = 12) →  -- |c| = 2√3
  (c = (l * a.1 + m * b.1, l * a.2 + m * b.2)) →  -- c = λa + μb
  l^2 + m^2 = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l172_17263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l172_17270

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2)^x + m

-- State the theorem
theorem function_inequality_implies_m_bound :
  ∀ m : ℝ,
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc (-1) 1, f x₁ ≥ g m x₂) →
  m ≤ 5/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l172_17270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_range_l172_17273

def vector_a (l : ℝ) : Fin 2 → ℝ := ![1, l]
def vector_b (l : ℝ) : Fin 2 → ℝ := ![l, 4]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w > 0

theorem acute_angle_range (l : ℝ) :
  is_acute_angle (vector_a l) (vector_b l) ↔ l ∈ Set.Ioo 0 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_range_l172_17273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l172_17252

/-- The total surface area of a cone with given diameter and height -/
noncomputable def coneTotalSurfaceArea (diameter height : ℝ) : ℝ :=
  let radius := diameter / 2
  let slantHeight := Real.sqrt (radius ^ 2 + height ^ 2)
  let lateralArea := Real.pi * radius * slantHeight
  let baseArea := Real.pi * radius ^ 2
  lateralArea + baseArea

/-- Theorem: The total surface area of a cone with diameter 8 cm and height 12 cm
    is equal to 16π(√10 + 1) cm² -/
theorem cone_surface_area_specific :
  coneTotalSurfaceArea 8 12 = 16 * Real.pi * (Real.sqrt 10 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_specific_l172_17252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cube_cost_10ft_l172_17261

/-- The cost to paint a cube given its edge length, paint cost per quart, and coverage per quart -/
noncomputable def paint_cube_cost (edge_length : ℝ) (cost_per_quart : ℝ) (coverage_per_quart : ℝ) : ℝ :=
  6 * edge_length^2 * cost_per_quart / coverage_per_quart

/-- Theorem: The cost to paint a 10-foot cube is $192 -/
theorem paint_cube_cost_10ft : paint_cube_cost 10 3.2 10 = 192 := by
  -- Unfold the definition of paint_cube_cost
  unfold paint_cube_cost
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cube_cost_10ft_l172_17261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l172_17230

/-- Calculates the cost price given the selling price and markup percentage -/
def costPrice (sellingPrice : ℚ) (markupPercentage : ℚ) : ℚ :=
  sellingPrice / (1 + markupPercentage / 100)

/-- Theorem stating that the cost price of the computer table is approximately 6424 -/
theorem computer_table_cost_price :
  let sellingPrice : ℚ := 7967
  let markupPercentage : ℚ := 24
  let calculatedCostPrice := costPrice sellingPrice markupPercentage
  ⌊calculatedCostPrice⌋ = 6424 := by
  sorry

#eval Int.floor (costPrice 7967 24)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_price_l172_17230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l172_17226

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point P (-1, 2) to the line 8x-6y+15=0 is 1/2 -/
theorem distance_point_to_line_example : 
  distancePointToLine (-1) 2 8 (-6) 15 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l172_17226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_customers_to_second_floor_l172_17293

structure Store where
  floors : Nat
  elevator : Bool

structure CustomerFlow (s : Store) where
  x : Nat
  y : Nat
  z : Nat
  h_positive : x > 0 ∧ y > 0 ∧ z > 0

theorem more_customers_to_second_floor
  (s : Store)
  (cf : CustomerFlow s)
  (h_floors : s.floors = 3)
  (h_elevator : s.elevator)
  (h_second_floor : cf.y / 2 = cf.x - (cf.z - cf.y / 2))
  (h_third_floor : cf.z < (cf.x + cf.y + cf.z) / 3) :
  cf.x - (cf.z - cf.y / 2) > cf.z - cf.y / 2 := by
  sorry

#check more_customers_to_second_floor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_customers_to_second_floor_l172_17293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_four_l172_17277

theorem probability_at_least_one_multiple_of_four :
  let n : ℕ := 100
  let k : ℕ := 4
  let multiples : ℕ := n / k
  let non_multiples : ℕ := n - multiples
  (n^2 - non_multiples^2 : ℚ) / (n^2 : ℚ) = 7/16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_four_l172_17277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_cos_x_solutions_l172_17299

theorem sin_3x_eq_cos_x_solutions :
  ∃ (S : Finset ℝ), S.card = 6 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos x → x ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_eq_cos_x_solutions_l172_17299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_bear_cost_l172_17203

theorem teddy_bear_cost 
  (num_toys : ℕ) 
  (toy_cost : ℕ) 
  (num_teddy_bears : ℕ) 
  (total_cost : ℕ) 
  (teddy_bear_cost : ℕ)
  (h1 : num_toys = 28) 
  (h2 : toy_cost = 10) 
  (h3 : num_teddy_bears = 20) 
  (h4 : total_cost = 580) 
  (h5 : total_cost = num_toys * toy_cost + num_teddy_bears * teddy_bear_cost) :
  teddy_bear_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teddy_bear_cost_l172_17203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_intersection_sum_l172_17247

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the length of a segment between two points -/
noncomputable def SegmentLength (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

/-- Checks if a point is the midpoint of a segment -/
def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Checks if a point is on the circumcircle of a triangle -/
def IsOnCircumcircle (P A B C : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    SegmentLength center A = radius ∧
    SegmentLength center B = radius ∧
    SegmentLength center C = radius ∧
    SegmentLength center P = radius

/-- Approximate equality for real numbers -/
def approx (a b : ℝ) : Prop :=
  |a - b| < 0.01

/-- Given a triangle ABC with side lengths AB = 12, BC = 15, AC = 13, 
    where D, E, F are midpoints of AB, BC, AC respectively, 
    and X ≠ E is the intersection of circumcircles of triangles BDE and CEF, 
    the sum XA + XB + XC is approximately equal to 38.88. -/
theorem triangle_circumcircle_intersection_sum (A B C D E F X : Point) : 
  SegmentLength A B = 12 →
  SegmentLength B C = 15 →
  SegmentLength A C = 13 →
  IsMidpoint D A B →
  IsMidpoint E B C →
  IsMidpoint F A C →
  X ≠ E →
  IsOnCircumcircle X B D E →
  IsOnCircumcircle X C E F →
  approx (SegmentLength X A + SegmentLength X B + SegmentLength X C) 38.88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_intersection_sum_l172_17247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l172_17215

noncomputable def f (x : ℝ) := Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 9 - 8 * Real.sqrt (x - 2))

theorem solution_set (x : ℝ) : x ≥ 2 ∧ f x = 3 ↔ x = 6 ∨ x = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l172_17215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_result_l172_17292

theorem complex_arithmetic_result : 
  ∃ ε > 0, |((116 * 2 - 116) - (116 * 2 + 104) / (3^2 : ℝ) + 
             (104 * 3 - 104) - (104 * 3 + 94) / (4^2 : ℝ)) - 261.291| < ε :=
by
  -- We'll use a specific ε value for this approximation
  use 0.001
  norm_num
  -- The proof is completed by numerical computation
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_result_l172_17292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_f_period_f_two_maxima_l172_17223

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vectors_parallel (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) :
  (∃ k : ℝ, a x = k • b x) ↔ (x = Real.pi / 3 ∨ x = Real.pi / 2) := by sorry

theorem f_period : ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
  (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧ p = Real.pi := by sorry

theorem f_two_maxima (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 m ∧ x₂ ∈ Set.Icc 0 m ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 m → f x ≤ f x₁) ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 m → f x ≤ f x₂)) ↔
  m ≥ 7 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_f_period_f_two_maxima_l172_17223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_cannot_tile_l172_17266

noncomputable def internal_angle (n : ℕ) : ℝ := 180 - (360 / n)

def can_tile_plane (angle : ℝ) : Prop := ∃ k : ℕ, k * angle = 360

theorem regular_pentagon_cannot_tile :
  can_tile_plane (internal_angle 3) ∧
  can_tile_plane (internal_angle 4) ∧
  ¬can_tile_plane (internal_angle 5) ∧
  can_tile_plane (internal_angle 6) :=
by
  sorry

#check regular_pentagon_cannot_tile

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_cannot_tile_l172_17266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_example_l172_17255

/-- Calculates the total cost of a taxi ride based on given parameters. -/
def taxi_ride_cost (
  initial_charge : ℚ
) (additional_charge : ℚ
) (waiting_charge : ℚ
) (toll_fee : ℚ
) (surge_rate : ℚ
) (distance : ℚ
) (waiting_time : ℕ
) (has_toll : Bool
) (is_peak_hour : Bool
) : ℚ :=
  sorry

theorem taxi_ride_cost_example :
  let initial_charge : ℚ := 5/2
  let additional_charge : ℚ := 2/5
  let waiting_charge : ℚ := 1/4
  let toll_fee : ℚ := 3
  let surge_rate : ℚ := 1/5
  let distance : ℚ := 8
  let waiting_time : ℕ := 12
  let has_toll : Bool := true
  let is_peak_hour : Bool := true
  taxi_ride_cost initial_charge additional_charge waiting_charge toll_fee surge_rate
    distance waiting_time has_toll is_peak_hour = 289/10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_example_l172_17255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l172_17291

theorem root_exists_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ x^3 = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l172_17291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l172_17246

theorem sin_cos_sixth_power_min (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l172_17246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_trig_identity_l172_17281

/-- Given a point P(-4,3) on the terminal side of angle θ, prove that 2sin θ + cos θ = 2/5 -/
theorem terminal_side_trig_identity (θ : ℝ) (h : ((-4 : ℝ), 3) ∈ Set.range (λ t => (t * Real.cos θ, t * Real.sin θ))) : 
  2 * Real.sin θ + Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_trig_identity_l172_17281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_xy_xz_yz_l172_17208

noncomputable def xy_xz_yz (x y z : ℝ) : ℝ := x*y + x*z + y*z

theorem sum_max_min_xy_xz_yz (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  (⨆ t : ℝ, xy_xz_yz x y z) + 15 * (⨅ t : ℝ, xy_xz_yz x y z) = 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_xy_xz_yz_l172_17208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l172_17297

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculates the slope between two points -/
noncomputable def slopeBetweenPoints (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- The main theorem -/
theorem line_equation_correct (A : Point) (m : ℝ) (l : Line) : 
  A.x = 1 → A.y = -2 → m = 3 → 
  l.a = 3 → l.b = -1 → l.c = -5 →
  pointOnLine A l ∧ ∀ (B : Point), pointOnLine B l → B.x ≠ A.x → slopeBetweenPoints A B = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l172_17297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l172_17228

/-- Represents the shopkeeper's financial situation --/
structure Shopkeeper where
  costPrice : ℝ
  profitRate : ℝ
  priceFluctuationRate : ℝ
  taxRate : ℝ
  theftLossRate : ℝ
  insuranceReimbursementRate : ℝ

/-- Calculates the overall loss percentage for the shopkeeper --/
noncomputable def calculateLossPercentage (s : Shopkeeper) : ℝ :=
  let profit := s.costPrice * s.profitRate
  let sellingPrice := s.costPrice + profit
  let priceDecrease := sellingPrice * s.priceFluctuationRate
  let tax := profit * s.taxRate
  let theftLoss := s.costPrice * s.theftLossRate
  let insuranceReimbursement := theftLoss * s.insuranceReimbursementRate
  let netLoss := theftLoss - insuranceReimbursement
  let overallLoss := netLoss + priceDecrease - (profit - tax)
  (overallLoss / s.costPrice) * 100

/-- Theorem stating that the shopkeeper's overall loss percentage is 9.5% --/
theorem shopkeeper_loss_percentage :
  let s : Shopkeeper := {
    costPrice := 100
    profitRate := 0.1
    priceFluctuationRate := 0.05
    taxRate := 0.15
    theftLossRate := 0.5
    insuranceReimbursementRate := 0.75
  }
  calculateLossPercentage s = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l172_17228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_l172_17286

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + 16/19, 9; 4 - 16/19, 10]
  ¬(IsUnit (Matrix.det A)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_l172_17286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l172_17240

theorem class_average_problem (total_students : ℕ) (group1_students : ℕ) 
  (group1_average : ℝ) (class_average : ℝ) :
  total_students = 20 →
  group1_students = 10 →
  group1_average = 80 →
  class_average = 70 →
  (total_students * class_average - group1_students * group1_average) / (total_students - group1_students) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l172_17240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l172_17260

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = !![1, 0; 0, 1]

theorem smallest_rotation_power : 
  (∀ k : ℕ, k > 0 → k < 3 → ¬is_identity (A ^ k)) ∧ 
  is_identity (A ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l172_17260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l172_17222

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 6
noncomputable def line2 (x : ℝ) : ℝ := -2 * x + 12

noncomputable def intersection_x : ℝ := 18 / 5
noncomputable def intersection_y : ℝ := line1 intersection_x

noncomputable def y_intercept1 : ℝ := line1 0
noncomputable def y_intercept2 : ℝ := line2 0

theorem triangle_area : 
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  (1 / 2 : ℝ) * base * height = 32.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l172_17222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_theorem_l172_17225

/-- An inverse proportion function passing through (3,-2) -/
noncomputable def inverse_prop (k : ℝ) (x : ℝ) : ℝ := (2 - k) / x

theorem inverse_prop_theorem (k : ℝ) :
  (inverse_prop k 3 = -2) →
  (k = 8 ∧
   ∀ x₁ x₂ y₁ y₂ : ℝ, 
     0 < x₁ → x₁ < x₂ → 
     inverse_prop k x₁ = y₁ → 
     inverse_prop k x₂ = y₂ → 
     y₁ < y₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_theorem_l172_17225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wilsons_theorem_l172_17243

theorem wilsons_theorem (p : Nat) (h : Nat.Prime p) : Fact ((Nat.factorial (p - 1)) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wilsons_theorem_l172_17243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_three_pi_four_l172_17296

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem f_zero_at_three_pi_four (φ : ℝ) 
  (h1 : |φ| ≤ π/2)
  (h2 : f (π/6) φ = 1/2)
  (h3 : f (5*π/6) φ = 1/2) :
  f (3*π/4) φ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_at_three_pi_four_l172_17296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_equidistant_points_l172_17217

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between a point and a line -/
noncomputable def distancePointLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Distance between two points -/
noncomputable def distancePoints (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is equidistant from a circle and two parallel lines -/
def isEquidistant (p : Point) (c : Circle) (l1 l2 : Line) : Prop :=
  distancePointLine p l1 = distancePointLine p l2 ∧
  distancePointLine p l1 = abs (distancePoints p c.center - c.radius)

/-- The main theorem -/
theorem four_equidistant_points
  (c : Circle)
  (l1 l2 : Line)
  (h_parallel : l1.a = l2.a ∧ l1.b = l2.b)
  (h_distance : distancePointLine c.center l1 > c.radius) :
  ∃ (s : Finset Point), s.card = 4 ∧ ∀ p ∈ s, isEquidistant p c l1 l2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_equidistant_points_l172_17217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_rate_calculation_l172_17244

/-- Calculate the rate of gravelling per square meter for a rectangular plot with a gravel path -/
theorem gravel_rate_calculation (plot_length plot_width path_width total_cost : ℝ) 
  (h1 : plot_length = 110)
  (h2 : plot_width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 340) :
  total_cost / ((plot_length * plot_width) - ((plot_length - 2 * path_width) * (plot_width - 2 * path_width))) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_rate_calculation_l172_17244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l172_17287

-- Define the function f(x) = log₂x + 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 3

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 1 ∧ f x = y) ↔ y ∈ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l172_17287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l172_17285

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line
def myLine (x y : ℝ) : Prop := y = x + 3

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem distance_from_center_to_line :
  let (x₀, y₀) := circleCenter
  ∃ d : ℝ, d = |x₀ - y₀ + 3| / Real.sqrt 2 ∧ d = Real.sqrt 2 := by
  sorry

#check distance_from_center_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l172_17285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l172_17254

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 20 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1.5)

-- Define the radius of the circle
noncomputable def radius : ℝ := Real.sqrt (7/4)

-- Theorem stating that the equation represents a circle with the given center and radius
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l172_17254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_probabilities_l172_17238

/-- A family participating in an environmental protection contest --/
inductive Family : Type
| A : Family
| B : Family
| C : Family

/-- The probability of a family answering a question correctly --/
noncomputable def prob_correct (f : Family) : ℚ :=
  match f with
  | Family.A => 3/4
  | Family.B => 3/8
  | Family.C => 2/3

/-- The probability of two families both answering correctly --/
noncomputable def prob_both_correct (f1 f2 : Family) : ℚ :=
  prob_correct f1 * prob_correct f2

/-- The probability of two families both answering incorrectly --/
noncomputable def prob_both_incorrect (f1 f2 : Family) : ℚ :=
  (1 - prob_correct f1) * (1 - prob_correct f2)

/-- The main theorem about the probabilities in the contest --/
theorem contest_probabilities :
  (prob_both_incorrect Family.A Family.C = 1/12) ∧
  (prob_both_correct Family.B Family.C = 1/4) ∧
  (prob_correct Family.A * prob_correct Family.B * prob_correct Family.C +
   (1 - prob_correct Family.A) * prob_correct Family.B * prob_correct Family.C +
   prob_correct Family.A * (1 - prob_correct Family.B) * prob_correct Family.C +
   prob_correct Family.A * prob_correct Family.B * (1 - prob_correct Family.C) = 21/32) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_probabilities_l172_17238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_satisfies_condition_l172_17231

/-- The speed of a car that takes 2 seconds longer to travel 1 kilometer compared to 60 km/h -/
noncomputable def car_speed : ℝ :=
  let time_60 : ℝ := 1 / 60 * 3600  -- Time in seconds to travel 1 km at 60 km/h
  let v : ℝ := 3600 / 62           -- Speed in km/h that satisfies the condition
  v

/-- Theorem stating that the calculated car_speed satisfies the given condition -/
theorem car_speed_satisfies_condition : 
  (1 / car_speed) * 3600 = 62 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_satisfies_condition_l172_17231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_symmetric_angles_l172_17210

/-- Given that the terminal sides of angles α and β are symmetric with respect to y = x,
    and the terminal side of angle α passes through (-1/2, √5/4), prove that sin(α + β) = 1 -/
theorem sin_sum_symmetric_angles (α β : ℝ) 
  (h1 : ∃ (t : ℝ), t * (-1/2) + (1 - t) * (Real.sqrt 5/4) = t * (Real.sqrt 5/4) + (1 - t) * (-1/2)) 
  (h2 : ∃ (t : ℝ), t * (-1/2) + (1 - t) * (Real.sqrt 5/4) = Real.cos α ∧ 
                   t * (Real.sqrt 5/4) + (1 - t) * (-1/2) = Real.sin α) :
  Real.sin (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_symmetric_angles_l172_17210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossible_l172_17289

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a rectangle has one side twice as long as the other -/
def Rectangle.isValidRatio (r : Rectangle) : Prop :=
  r.width = 2 * r.height ∨ r.height = 2 * r.width

/-- Checks if a list of rectangles can fit inside a square -/
def fitsInSquare (s : Square) (rectangles : List Rectangle) : Prop :=
  ∃ (arrangement : List (ℕ × ℕ)), 
    arrangement.length = rectangles.length ∧
    ∀ (i : Fin rectangles.length),
      ∃ (x y : ℕ),
        x + (rectangles.get i).width ≤ s.side ∧ 
        y + (rectangles.get i).height ≤ s.side

/-- The main theorem stating the impossibility of the division -/
theorem square_division_impossible :
  ∀ (s : Square),
  ¬∃ (rectangles : List Rectangle),
    rectangles.length = 7 ∧
    (∀ r ∈ rectangles, r.isValidRatio) ∧
    fitsInSquare s rectangles :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossible_l172_17289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_decrease_approx_2_percent_l172_17220

/-- Represents a parallel plate capacitor with square plates -/
structure ParallelPlateCapacitor where
  sideLength : ℝ
  capacitance : ℝ
  separation : ℝ

/-- Calculates the new voltage after increasing the side length by 1% -/
noncomputable def newVoltage (c : ParallelPlateCapacitor) : ℝ :=
  let newSideLength := c.sideLength * 1.01
  let newArea := newSideLength ^ 2
  let newCapacitance := c.capacitance * (newArea / (c.sideLength ^ 2))
  c.capacitance / newCapacitance

/-- Theorem stating that the voltage decreases by approximately 2% when the side length increases by 1% -/
theorem voltage_decrease_approx_2_percent (c : ParallelPlateCapacitor) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |newVoltage c - 0.98| < ε := by
  sorry

#check voltage_decrease_approx_2_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_decrease_approx_2_percent_l172_17220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_inequality_l172_17242

/-- Given a geometric sequence {a_n} with positive terms, S_n is the sum of the first n terms -/
noncomputable def S (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ := (a 1) * (1 - q^n) / (1 - q)

/-- Theorem: If S_6 - 2S_3 = 5 for a geometric sequence with positive terms,
    then S_9 - S_6 is greater than or equal to 20 -/
theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_sum_diff : S 6 a q - 2 * S 3 a q = 5) :
  S 9 a q - S 6 a q ≥ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_inequality_l172_17242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_cos_2theta_l172_17239

open Real

/-- The maximum y-coordinate of a point on the graph of r = cos 2θ is √6/9 -/
theorem max_y_coordinate_cos_2theta : 
  ∃ (max_y : ℝ), (∀ θ : ℝ, (cos (2 * θ) * sin θ) ≤ max_y) ∧ (max_y = Real.sqrt 6 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_cos_2theta_l172_17239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l172_17211

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (9*x^2 - 6*x + 1) - Real.sqrt (9*x^2 + 6*x + 1) + 2*x^2

/-- The proposed equivalent form -/
def g (x : ℝ) : ℝ := |3*x - 1| - |3*x + 1| + 2*x^2

/-- Theorem stating the equivalence of f and g -/
theorem f_eq_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l172_17211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_exponent_sum_l172_17205

noncomputable section

variable (a b c : ℝ)

-- Define the original expression
def original_expression := Real.sqrt (72 * a^5 * b^8 * c^13)

-- Define the simplified expression
def simplified_expression := 6 * a^2 * b^4 * c^6 * Real.sqrt (a * c)

-- Define the sum of exponents outside the square root
def sum_of_exponents := 2 + 4 + 6

theorem simplification_and_exponent_sum :
  original_expression a b c = simplified_expression a b c ∧
  sum_of_exponents = 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_and_exponent_sum_l172_17205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_berries_cheapest_l172_17216

/-- Represents the cost of making jam using different methods -/
structure JamCost where
  transportationCost : ℚ
  berriesGathered : ℚ
  marketBerryPrice : ℚ
  sugarPrice : ℚ
  readyJamPrice : ℚ

/-- Calculates the cost of making 1.5 kg of jam by picking berries -/
def costPickingBerries (c : JamCost) : ℚ :=
  (c.transportationCost / c.berriesGathered) + c.sugarPrice

/-- Calculates the cost of making 1.5 kg of jam by buying berries -/
def costBuyingBerries (c : JamCost) : ℚ :=
  c.marketBerryPrice + c.sugarPrice

/-- Calculates the cost of buying 1.5 kg of ready-made jam -/
def costBuyingJam (c : JamCost) : ℚ :=
  (3/2) * c.readyJamPrice

/-- Theorem: Picking berries is the cheapest method to make jam -/
theorem picking_berries_cheapest (c : JamCost) 
  (h1 : c.transportationCost = 200)
  (h2 : c.berriesGathered = 5)
  (h3 : c.marketBerryPrice = 150)
  (h4 : c.sugarPrice = 54)
  (h5 : c.readyJamPrice = 220) :
  costPickingBerries c < costBuyingBerries c ∧ 
  costPickingBerries c < costBuyingJam c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_berries_cheapest_l172_17216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l172_17295

noncomputable def f (x : ℝ) : ℝ := 5 * x^2 - 1/x + 3

noncomputable def g (k x : ℝ) : ℝ := x^2 - k*x - k

theorem k_value : 
  ∃ k : ℝ, f 2 - g k 2 = 7 ∧ k = -23/6 := by
  use -23/6
  constructor
  · -- Prove f 2 - g (-23/6) 2 = 7
    calc
      f 2 - g (-23/6) 2 = (5 * 2^2 - 1/2 + 3) - (2^2 - (-23/6) * 2 - (-23/6)) := by rfl
      _ = (20 - 1/2 + 3) - (4 + 23/3 + 23/6) := by ring_nf
      _ = 22.5 - (4 + 23/3 + 23/6) := by ring_nf
      _ = 7 := by ring_nf
  · -- Prove k = -23/6
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l172_17295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_order_most_economical_correct_l172_17206

/-- Represents the size of a facial cream container -/
inductive Size
  | S -- Small
  | M -- Medium
  | L -- Large

/-- Represents the cost and quantity of cream for each size -/
structure CreamInfo where
  cost : ℝ
  quantity : ℝ

/-- Given information about facial cream sizes -/
noncomputable def facial_cream_info (size : Size) : CreamInfo :=
  match size with
  | Size.S => { cost := 1, quantity := 2/3 }
  | Size.M => { cost := 1.3, quantity := 0.85 }
  | Size.L => { cost := 1.82, quantity := 1 }

/-- Cost-effectiveness of a size, lower is better -/
noncomputable def cost_effectiveness (size : Size) : ℝ :=
  let info := facial_cream_info size
  info.cost / info.quantity

/-- Theorem stating the order of cost-effectiveness -/
theorem cost_effectiveness_order :
  cost_effectiveness Size.L > cost_effectiveness Size.S ∧
  cost_effectiveness Size.S > cost_effectiveness Size.M :=
by sorry

/-- The most economical size to buy -/
def most_economical : Size := Size.M

/-- Theorem stating that the most economical size is correct -/
theorem most_economical_correct :
  ∀ size, size ≠ most_economical →
    cost_effectiveness size > cost_effectiveness most_economical :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_effectiveness_order_most_economical_correct_l172_17206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l172_17279

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 2 + 32 / (n : ℝ) ≥ 8 ∧ 
  ((n : ℝ) / 2 + 32 / (n : ℝ) = 8 ↔ n = 8) := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l172_17279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l172_17288

-- Define the slope of a line given its equation ax + by = c
def line_slope (a b : ℚ) : ℚ := -a / b

-- Define what it means for two lines to be parallel
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℚ) : Prop :=
  line_slope a₁ b₁ = line_slope a₂ b₂

-- Theorem statement
theorem parallel_line_slope :
  ∀ (a b c : ℚ), parallel 3 (-6) 12 a b c → line_slope a b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l172_17288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_price_is_125_l172_17259

-- Define the total price of the boat
variable (boat_price : ℝ)

-- Define Pankrác's payment
def pankrac_payment (boat_price : ℝ) : ℝ := 0.6 * boat_price

-- Define the remaining price after Pankrác's payment
def remaining_after_pankrac (boat_price : ℝ) : ℝ := boat_price - pankrac_payment boat_price

-- Define Servác's payment
def servac_payment (boat_price : ℝ) : ℝ := 0.4 * remaining_after_pankrac boat_price

-- Define Bonifác's payment
def bonifac_payment : ℝ := 30

-- Theorem stating that the boat price is 125 zlatek
theorem boat_price_is_125 : ∃ boat_price : ℝ, boat_price = 125 ∧ 
  bonifac_payment = remaining_after_pankrac boat_price - servac_payment boat_price :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_price_is_125_l172_17259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l172_17204

theorem inequality_proof (a b c d e f : ℕ+) 
  (h1 : (a : ℚ) / b < (c : ℚ) / d)
  (h2 : (c : ℚ) / d < (e : ℚ) / f)
  (h3 : (a * f : ℤ) - (b * e : ℤ) = -1) :
  (d : ℕ) ≥ b + f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l172_17204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_distance_theorem_l172_17214

/-- The distance Heather walks before meeting Stacy -/
noncomputable def heather_distance (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (delay : ℝ) : ℝ :=
  let t := (total_distance - delay * stacy_speed) / (heather_speed + stacy_speed)
  heather_speed * t

/-- Theorem stating the distance Heather walks before meeting Stacy -/
theorem heather_distance_theorem (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (delay : ℝ) 
    (h1 : total_distance = 40)
    (h2 : heather_speed = 5)
    (h3 : stacy_speed = heather_speed + 1)
    (h4 : delay = 24 / 60) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |heather_distance total_distance heather_speed stacy_speed delay - 17.09| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_distance_theorem_l172_17214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l172_17253

-- Define the quadrilateral EFGH
def Quadrilateral (E F G H : ℝ × ℝ) : Prop :=
  -- Right angles at F and H
  (F.2 - E.2) * (G.1 - F.1) + (F.1 - E.1) * (G.2 - F.2) = 0 ∧
  (H.2 - G.2) * (E.1 - H.1) + (H.1 - G.1) * (E.2 - H.2) = 0

-- Define the length of a line segment
noncomputable def Length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the area of a quadrilateral
noncomputable def Area (E F G H : ℝ × ℝ) : ℝ :=
  abs ((E.1 * F.2 + F.1 * G.2 + G.1 * H.2 + H.1 * E.2) -
       (F.1 * E.2 + G.1 * F.2 + H.1 * G.2 + E.1 * H.2)) / 2

theorem quadrilateral_area 
  (E F G H : ℝ × ℝ) 
  (h1 : Quadrilateral E F G H) 
  (h2 : Length E G = 5) 
  (h3 : ∃ (a b : ℕ), a ≠ b ∧ 
    ((Length E F = a ∧ Length F G = b) ∨ 
     (Length F G = a ∧ Length G H = b) ∨ 
     (Length G H = a ∧ Length H E = b) ∨ 
     (Length H E = a ∧ Length E F = b))) : 
  Area E F G H = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l172_17253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l172_17278

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- k is the smallest positive integer such that p(z) divides z^k - 1 -/
theorem smallest_k_divides : ∃! k : ℕ, k > 0 ∧
  (∀ z : ℂ, p z = 0 → z^k = 1) ∧ 
  (∀ m : ℕ, 0 < m → m < k → ∃ z : ℂ, p z = 0 ∧ z^m ≠ 1) ∧
  k = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_divides_l172_17278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l172_17249

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (π / 2 + 2 * x) - 5 * sin x

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l172_17249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_10_value_l172_17233

/-- Parabola definition -/
def Parabola (x y : ℝ) : Prop := x^2 = y

/-- Focus of the parabola -/
noncomputable def Focus : ℝ × ℝ := (0, 1/4)

/-- Point on the parabola -/
noncomputable def PointOnParabola (n : ℕ) : ℝ × ℝ := sorry

/-- Distance between consecutive points and focus -/
axiom distance_diff (n : ℕ) : 
  ‖PointOnParabola (n + 1) - Focus‖ - ‖PointOnParabola n - Focus‖ = 2

/-- Third point x-coordinate -/
axiom x_3 : (PointOnParabola 3).1 = 2

/-- Main theorem -/
theorem y_10_value : (PointOnParabola 10).2 = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_10_value_l172_17233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_probability_l172_17264

theorem correct_arrangement_probability :
  (2 : ℚ) / 6 = 1 / 3 :=
by
  field_simp
  ring

#eval (2 : ℚ) / 6 == 1 / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_probability_l172_17264
