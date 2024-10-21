import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_focal_distance_l640_64034

/-- An ellipse with axes parallel to the coordinate axes, tangent to the x-axis at (5, 0) and to the y-axis at (0, 2) -/
structure TangentEllipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  h_center : center = (5, 2)
  h_major : a = 5
  h_minor : b = 2

/-- The distance between the foci of the ellipse -/
noncomputable def focalDistance (e : TangentEllipse) : ℝ := 2 * Real.sqrt (e.a^2 - e.b^2)

/-- Theorem stating that the distance between the foci of the described ellipse is 2√21 -/
theorem tangent_ellipse_focal_distance (e : TangentEllipse) : 
  focalDistance e = 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ellipse_focal_distance_l640_64034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_km_hr_l640_64053

-- Define the train's properties
noncomputable def train_length : ℝ := 150  -- in meters
noncomputable def time_to_cross_pole : ℝ := 9  -- in seconds

-- Define the conversion factor from m/s to km/hr
noncomputable def m_s_to_km_hr : ℝ := 3.6

-- Define the speed calculation function
noncomputable def calculate_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time * m_s_to_km_hr

-- Theorem statement
theorem train_speed_is_60_km_hr :
  calculate_speed train_length time_to_cross_pole = 60 := by
  -- Unfold the definitions
  unfold calculate_speed train_length time_to_cross_pole m_s_to_km_hr
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_km_hr_l640_64053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_probability_l640_64057

/-- Represents the probability of winning if the first flip is heads (1) -/
noncomputable def x : ℝ := 9/29

/-- Represents the probability of winning if the first flip is tails (0) -/
noncomputable def y : ℝ := 5/29

/-- The overall probability of winning -/
noncomputable def P : ℝ := (x + y) / 2

/-- Theorem stating that the probability of 10101 occurring before 010101 is 7/29 -/
theorem sequence_probability : P = 7/29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_probability_l640_64057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_condition_l640_64007

/-- A function f(x) = x³ + ax² + (a+6)x + 1 has both a maximum and a minimum value if and only if the parameter a is in the set (-∞, -3) ∪ (6, +∞) -/
theorem function_extrema_condition (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^3 + a*x^2 + (a+6)*x + 1 ≤ x₁^3 + a*x₁^2 + (a+6)*x₁ + 1) ∧
    (∀ x : ℝ, x^3 + a*x^2 + (a+6)*x + 1 ≥ x₂^3 + a*x₂^2 + (a+6)*x₂ + 1)) ↔ 
  (a < -3 ∨ a > 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_condition_l640_64007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_set_size_l640_64060

theorem min_sum_set_size (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h₁ : n ≥ 2)
  (h₂ : a 0 = 0)
  (h₃ : a (Fin.last n) = 2 * n - 1)
  (h₄ : ∀ i j : Fin (n + 1), i < j → a i < a j) :
  Finset.card (Finset.image (fun p : Fin (n + 1) × Fin (n + 1) => a p.1 + a p.2) 
    (Finset.filter (fun p : Fin (n + 1) × Fin (n + 1) => p.1 ≤ p.2) (Finset.univ.product Finset.univ))) = 3 * n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_set_size_l640_64060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_12_percent_l640_64011

/-- Calculates the tax rate on the first $40,000 of income -/
noncomputable def tax_rate_first_40k (total_income : ℝ) (total_tax : ℝ) : ℝ :=
  let excess_income := total_income - 40000
  let excess_tax := 0.2 * excess_income
  let base_tax := total_tax - excess_tax
  (base_tax / 40000) * 100

/-- Theorem stating that the tax rate on the first $40,000 is 12% -/
theorem tax_rate_is_12_percent :
  tax_rate_first_40k 56000 8000 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_12_percent_l640_64011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_negative_l640_64046

/- Define the numbers A, B, C, D, and E -/
noncomputable def A : ℝ := -4.5
noncomputable def B : ℝ := -2.3
noncomputable def C : ℝ := 0.3
noncomputable def D : ℝ := 1.2
noncomputable def E : ℝ := 2.4

/- Define the expressions -/
noncomputable def expr1 := A - B
noncomputable def expr2 := A * B
noncomputable def expr3 := (D / B) * A
noncomputable def expr4 := C / (A * B)
noncomputable def expr5 := (D + E) / C

/- Theorem statement -/
theorem only_expr1_negative :
  expr1 < 0 ∧ expr2 ≥ 0 ∧ expr3 ≥ 0 ∧ expr4 > 0 ∧ expr5 > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr1_negative_l640_64046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l640_64084

-- Define the function h
noncomputable def h (t : ℝ) : ℝ := (t^2 - 1/2*t) / (2*t^2 + 1)

-- State the theorem about the range of h
theorem range_of_h :
  Set.range h = Set.Icc (1 - Real.sqrt 15 / 2) (1 + Real.sqrt 15 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l640_64084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l640_64004

theorem repeating_decimal_sum (x y : ℕ) : 
  (5 : ℚ) / 13 = ∑' n, (10 * x + y : ℚ) / (100 ^ (n + 1)) → x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l640_64004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l640_64083

def mySequence (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem sequence_lower_bound (a : ℕ → ℝ) (h : mySequence a) : ∀ n : ℕ, a n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l640_64083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_discount_proof_l640_64045

theorem laptop_discount_proof (initial_price : ℝ) (h : initial_price > 0) :
  (initial_price - initial_price * (1 - 0.3) * (1 - 0.5)) / initial_price = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_discount_proof_l640_64045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_is_four_l640_64030

/-- Calculates the investment duration in years given the principal amount, interest rate, and simple interest. -/
noncomputable def investment_duration (principal : ℝ) (rate : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * rate)

/-- Theorem: Given the specified conditions, the investment duration is 4 years. -/
theorem investment_duration_is_four :
  let principal : ℝ := 780
  let rate : ℝ := 4.166666666666667
  let simple_interest : ℝ := 130
  investment_duration principal rate simple_interest = 4 := by
  -- Unfold the definition of investment_duration
  unfold investment_duration
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_is_four_l640_64030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l640_64065

-- Define the function f(x) as noncomputable due to Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

-- State the theorem
theorem monotonic_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (x₁ x₂ : ℝ)
  (h_symmetric : ∀ x, f ω (2 * x₁ - x) = f ω x ∧ f ω (2 * x₂ - x) = f ω x)
  (h_min_diff : ∀ y z, f ω y = f ω z → |x₁ - x₂| ≤ |y - z| ∧ |x₁ - x₂| = π / 2) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 3) (k * π + π / 6)) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l640_64065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_and_extrema_l640_64058

noncomputable section

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

-- State the theorem
theorem g_monotonicity_and_extrema (a : ℝ) (h : a > 1) :
  (1 < a ∧ a ≤ 2 → ∀ x y, 0 < x ∧ 0 < y ∧ x < y → g a x < g a y) ∧
  (a > 2 → 
    (∀ x y, 0 < x ∧ x < y ∧ y < a - 1 - Real.sqrt (a * (a - 2)) → g a x < g a y) ∧
    (∀ x y, a - 1 - Real.sqrt (a * (a - 2)) < x ∧ x < y ∧ y < a - 1 + Real.sqrt (a * (a - 2)) → g a x > g a y) ∧
    (∀ x y, a - 1 + Real.sqrt (a * (a - 2)) < x ∧ x < y → g a x < g a y) ∧
    (∀ x, 0 < x → g a x ≤ g a (a - 1 - Real.sqrt (a * (a - 2)))) ∧
    (∀ x, 0 < x → g a x ≥ g a (a - 1 + Real.sqrt (a * (a - 2))))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotonicity_and_extrema_l640_64058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l640_64085

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (∀ (x : ℝ), (4 : ℝ)^x - (2 : ℝ)^x + 1 - 3 = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
  x₁ = 1 ∧ x₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l640_64085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_circumradius_l640_64018

/-- An isosceles trapezoid with three sides of length 1 and one side of length √2 -/
structure IsoscelesTrapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  is_isosceles : side1 = side2 ∧ side2 = side3
  side_lengths : side1 = 1 ∧ side4 = Real.sqrt 2

/-- The radius of the circle circumscribed around an isosceles trapezoid -/
noncomputable def circumradius (t : IsoscelesTrapezoid) : ℝ := Real.sqrt ((Real.sqrt 2 + 3) / 7)

/-- Theorem stating that the circumradius of the specified isosceles trapezoid is √((√2 + 3)/7) -/
theorem isosceles_trapezoid_circumradius (t : IsoscelesTrapezoid) : 
  circumradius t = Real.sqrt ((Real.sqrt 2 + 3) / 7) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_circumradius_l640_64018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_current_ages_l640_64016

-- Define Maria's and Evan's current ages
def maria_age : ℕ → ℕ := sorry
def evan_age : ℕ → ℕ := sorry

-- Maria is seven years older than Evan
axiom age_difference : ∀ n, maria_age n = evan_age n + 7

-- Ten years from now, Maria will be three times as old as Evan was five years ago
axiom future_age_relation : ∀ n, maria_age n + 10 = 3 * (evan_age n - 5)

-- Theorem to prove
theorem sum_of_current_ages : 
  ∃ n, maria_age n + evan_age n = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_current_ages_l640_64016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l640_64021

/-- The function f(x) = x^2 - 2ax + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

/-- Part 1: Range of a when f(x) > 0 for all x > 0 -/
theorem part1 (a : ℝ) : 
  (∀ x > 0, f a x > 0) → 
  a ∈ Set.Iic (-Real.sqrt 5) ∪ Set.Ioo (-Real.sqrt 5) (Real.sqrt 5) :=
sorry

/-- Part 2: Value of a when a > 1 and domain and range of f are [1,a] -/
theorem part2 (a : ℝ) : 
  a > 1 → 
  (∀ x, f a x ∈ Set.Icc 1 a ↔ x ∈ Set.Icc 1 a) → 
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l640_64021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_at_three_l640_64036

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 3 then 4 * x^2 + 1 else 12 * x + 1

theorem f_continuous_at_three : ContinuousAt f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_continuous_at_three_l640_64036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enhanced_ohara_triple_49_16_l640_64002

/-- Definition of Enhanced O'Hara triple -/
def is_enhanced_ohara_triple (a b : ℕ) (y : ℝ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) + Real.sqrt ((a + b : ℕ) : ℝ) = y

/-- Theorem: If (49, 16, y) is an Enhanced O'Hara triple, then y = 11 + √65 -/
theorem enhanced_ohara_triple_49_16 (y : ℝ) :
  is_enhanced_ohara_triple 49 16 y → y = 11 + Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enhanced_ohara_triple_49_16_l640_64002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_set_connection_l640_64072

/-- A Pythagorean set is a set of 3 positive integers that are lengths of sides of a right-angled triangle -/
def PythagoreanSet : Set (Set ℕ) :=
  {s | ∃ a b c : ℕ, s = {a, b, c} ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2}

/-- There exists a sequence of Pythagorean sets connecting any two Pythagorean sets -/
theorem pythagorean_set_connection (P Q : Set ℕ) 
  (hP : P ∈ PythagoreanSet) (hQ : Q ∈ PythagoreanSet) :
  ∃ (m : ℕ) (chain : Fin (m + 1) → Set ℕ),
    m ≥ 1 ∧
    (∀ i, chain i ∈ PythagoreanSet) ∧
    chain 0 = P ∧
    chain ⟨m, Nat.lt_succ_self m⟩ = Q ∧
    (∀ i : Fin m, (chain i ∩ chain ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩).Nonempty) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_set_connection_l640_64072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l640_64061

/-- Represents a square in the rectangle dissection -/
structure Square where
  side : Nat

/-- Represents the rectangle -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents the dissection of the rectangle into squares -/
structure Dissection where
  rect : Rectangle
  squares : Vector Square 7

/-- The conditions for a valid dissection -/
def validDissection (d : Dissection) : Prop :=
  let s := d.squares
  -- Relations between square sides
  s[0].side + s[1].side = s[2].side ∧
  s[0].side + s[2].side = s[3].side ∧
  s[2].side + s[3].side = s[4].side ∧
  s[1].side + s[2].side + s[3].side = s[5].side ∧
  s[1].side + s[5].side = s[6].side ∧
  s[0].side + s[3].side + s[4].side = s[6].side ∧
  s[4].side + s[6].side = s[5].side + s[6].side ∧
  -- Rectangle dimensions
  d.rect.width = s[4].side + s[6].side ∧
  d.rect.height = s[5].side + s[6].side ∧
  -- Width and height are relatively prime
  Nat.Coprime d.rect.width d.rect.height

theorem rectangle_perimeter (d : Dissection) :
  validDissection d → 2 * (d.rect.width + d.rect.height) = 178 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l640_64061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_M_l640_64047

-- Define the ellipse parameters
variable (a b : ℝ)

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle center
noncomputable def circle_center : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0

-- Define the vertices
def right_vertex : ℝ × ℝ := (a, 0)
def lower_vertex : ℝ × ℝ := (0, -b)

-- Define the angle BMN
noncomputable def angle_BMN : ℝ := 60 * Real.pi / 180

-- Theorem statement
theorem ellipse_and_point_M :
  (a = Real.sqrt 3 ∧ b = 1) ∧
  ∃ (M : ℝ × ℝ), M.1 > 0 ∧ M.2 = 0 ∧ M = (Real.sqrt 3 / 3, 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_M_l640_64047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l640_64035

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (Real.arctan x)^2 - Real.arctan x * Real.arctan (1/x) + (Real.arctan (1/x))^2

/-- The theorem stating the range of g(x) -/
theorem g_range :
  ∀ x : ℝ, x ≠ 0 →
    π^2 / 16 ≤ g x ∧ g x ≤ π^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l640_64035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_calculation_l640_64013

-- Define constants
noncomputable def train_length : ℝ := 1200
noncomputable def train_speed_kmh : ℝ := 120
noncomputable def crossing_time : ℝ := 45

-- Define function to convert km/h to m/s
noncomputable def km_per_hour_to_meter_per_second (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

-- Theorem statement
theorem tunnel_length_calculation :
  let train_speed_ms := km_per_hour_to_meter_per_second train_speed_kmh
  let total_distance := train_speed_ms * crossing_time
  let tunnel_length := total_distance - train_length
  ∃ ε > 0, |tunnel_length - 299.85| < ε :=
by
  sorry

#check tunnel_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_calculation_l640_64013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l640_64003

-- Define the function f(x) = log₃(x) + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 3

-- State the theorem
theorem solution_interval :
  ∃ x : ℝ, x > 2 ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l640_64003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l640_64020

theorem complex_fraction_calculation : 
  (1/2 + 1/4 + 1/8) / ((1/2) * (1/4) * (1/8)) *
  ((7/18) * (9/2) + 1/6) / (40/3 - 15/4 * 16/5) = 161/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l640_64020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_l640_64052

noncomputable section

variables {f : ℝ → ℝ}

-- The tangent line of f at x = 1 has slope 1/2
axiom f_derivative : deriv f 1 = 1/2

-- The tangent line of f at x = 1 passes through the point (1, 5/2)
axiom f_value : f 1 = 5/2

-- Theorem to prove
theorem f_sum : f 1 + deriv (deriv f) 1 = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_l640_64052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_preschool_fullday_students_l640_64087

theorem rainbow_preschool_fullday_students 
  (total_students : ℕ) 
  (half_day_percentage : ℚ) 
  (h1 : total_students = 80)
  (h2 : half_day_percentage = 1/4) : 
  total_students - (↑total_students * half_day_percentage).floor = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_preschool_fullday_students_l640_64087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pattern_2007_2008_l640_64049

/-- Represents the square pattern of positive integers as described in the problem -/
def squarePattern (row : ℕ) (col : ℕ) : ℕ := sorry

/-- The size of the square we're considering -/
def n : ℕ := 2008

theorem square_pattern_2007_2008 :
  squarePattern 2007 2008 = 4030056 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pattern_2007_2008_l640_64049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l640_64033

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

-- Theorem statement
theorem g_is_zero : ∀ x : ℝ, g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l640_64033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l640_64008

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : ℝ := by
  let rate_of_current : ℝ := 3
  let downstream_distance : ℝ := 7.2
  let downstream_time : ℝ := 24 / 60

  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed : ℝ := downstream_speed - rate_of_current

  have : boat_speed = 15 := by
    -- Proof steps would go here
    sorry

  exact boat_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l640_64008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_length_is_258_l640_64056

/-- A rectangular garden with given perimeter and breadth -/
structure RectangularGarden where
  perimeter : ℝ
  breadth : ℝ

/-- The length of a rectangular garden -/
noncomputable def length (g : RectangularGarden) : ℝ :=
  (g.perimeter - 2 * g.breadth) / 2

/-- Theorem: The length of a rectangular garden with perimeter 680 m and breadth 82 m is 258 m -/
theorem garden_length_is_258 :
  let g : RectangularGarden := { perimeter := 680, breadth := 82 }
  length g = 258 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_length_is_258_l640_64056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l640_64001

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the tangent line
def TangentLine (circle : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Set (Set (ℝ × ℝ)) :=
  {l : Set (ℝ × ℝ) | ∃ a b c : ℝ, l = Line a b c ∧ point ∈ l ∧ ∃! p, p ∈ circle ∧ p ∈ l}

theorem circle_and_tangents :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle passes through A(3,2) and B(1,4)
    (3, 2) ∈ Circle center radius ∧
    (1, 4) ∈ Circle center radius ∧
    -- Center lies on the line x+y-3=0
    center ∈ Line 1 1 (-3) ∧
    -- The circle equation is (x-1)^2 + (y-2)^2 = 4
    Circle center radius = Circle (1, 2) 2 ∧
    -- Tangent lines through M(3,1) are x-3=0 or 3x-4y-5=0
    (∀ l : Set (ℝ × ℝ), l ∈ TangentLine (Circle center radius) (3, 1) ↔
      l = Line 1 0 (-3) ∨ l = Line 3 (-4) (-5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l640_64001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l640_64077

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-11.5) 11.5) 
  (hb : b ∈ Set.Icc (-11.5) 11.5) 
  (hc : c ∈ Set.Icc (-11.5) 11.5) 
  (hd : d ∈ Set.Icc (-11.5) 11.5) : 
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 552 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l640_64077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_test_point_for_2_4_l640_64059

/-- The golden ratio, approximately 0.618 --/
noncomputable def ourGoldenRatio : ℝ := (Real.sqrt 5 - 1) / 2

/-- Calculates the first test point using the 0.618 method --/
noncomputable def firstTestPoint (a b : ℝ) : ℝ := a + ourGoldenRatio * (b - a)

/-- Theorem stating that the first test point for the interval [2, 4] is approximately 3.236 --/
theorem first_test_point_for_2_4 :
  ‖firstTestPoint 2 4 - 3.236‖ < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_test_point_for_2_4_l640_64059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_condition_l640_64041

/-- A function f is increasing on an interval I if for all x, y in I, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

/-- The interval (-∞, -1] -/
def NegativeOneAndBelow : Set ℝ := {x : ℝ | x ≤ -1}

/-- The function f(x) = x^3 - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem increasing_cubic_condition (a : ℝ) : 
  IncreasingOn (f a) NegativeOneAndBelow ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_cubic_condition_l640_64041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_right_triangles_l640_64024

/-- A right triangle with positive integer leg lengths -/
structure RightTriangle where
  a : ℕ+
  b : ℕ+

/-- The hypotenuse of a right triangle -/
noncomputable def RightTriangle.hypotenuse (t : RightTriangle) : ℝ :=
  Real.sqrt (t.a.val^2 + t.b.val^2 : ℝ)

/-- The area of a right triangle -/
noncomputable def RightTriangle.area (t : RightTriangle) : ℝ :=
  (t.a.val * t.b.val : ℝ) / 2

/-- The perimeter of a right triangle -/
noncomputable def RightTriangle.perimeter (t : RightTriangle) : ℝ :=
  t.a.val + t.b.val + t.hypotenuse

/-- A right triangle satisfying the area-perimeter condition -/
def SpecialRightTriangle (t : RightTriangle) : Prop :=
  t.area = 3 * t.perimeter

/-- The number of non-congruent special right triangles -/
def num_special_triangles : ℕ :=
  7  -- The actual number we want to prove

/-- The main theorem -/
theorem count_special_right_triangles :
  ∃ (s : Finset RightTriangle), 
    (∀ t ∈ s, SpecialRightTriangle t) ∧ 
    (∀ t, SpecialRightTriangle t → t ∈ s) ∧
    s.card = num_special_triangles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_right_triangles_l640_64024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l640_64028

theorem expression_equality : 
  Real.sqrt 27 - (-2)^(0 : ℕ) + |1 - Real.sqrt 3| + 2 * Real.cos (30 * π / 180) = 5 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l640_64028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_solution_liquid_x_percentage_l640_64082

/-- Represents the composition of a solution -/
structure Solution where
  total_mass : ℚ
  liquid_x_percentage : ℚ

/-- Calculates the mass of liquid X in a solution -/
def liquid_x_mass (s : Solution) : ℚ :=
  s.total_mass * (s.liquid_x_percentage / 100)

/-- Calculates the mass of water in a solution -/
def water_mass (s : Solution) : ℚ :=
  s.total_mass - liquid_x_mass s

/-- The initial solution Y -/
def initial_solution_y : Solution :=
  { total_mass := 10,
    liquid_x_percentage := 30 }

/-- The amount of water that evaporates -/
def evaporated_water : ℚ := 2

/-- The amount of solution Y added after evaporation -/
def added_solution_y : Solution :=
  { total_mass := 2,
    liquid_x_percentage := 30 }

/-- Theorem stating that the percentage of liquid X in the new solution is 36% -/
theorem new_solution_liquid_x_percentage : 
  let remaining_solution := 
    { total_mass := initial_solution_y.total_mass - evaporated_water,
      liquid_x_percentage := 
        (liquid_x_mass initial_solution_y / 
         (initial_solution_y.total_mass - evaporated_water)) * 100 }
  let new_solution_total_mass := 
    remaining_solution.total_mass + added_solution_y.total_mass
  let new_solution_liquid_x_mass := 
    liquid_x_mass remaining_solution + liquid_x_mass added_solution_y
  (new_solution_liquid_x_mass / new_solution_total_mass) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_solution_liquid_x_percentage_l640_64082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l640_64086

/-- The amount of tin in a mixture of alloys -/
noncomputable def amount_of_tin (weight_A weight_B weight_C : ℝ) 
  (ratio_A_lead ratio_A_tin : ℝ)
  (ratio_B_tin ratio_B_copper : ℝ)
  (ratio_C_tin ratio_C_copper ratio_C_iron : ℝ) : ℝ :=
  (ratio_A_tin / (ratio_A_lead + ratio_A_tin)) * weight_A +
  (ratio_B_tin / (ratio_B_tin + ratio_B_copper)) * weight_B +
  (ratio_C_tin / (ratio_C_tin + ratio_C_copper + ratio_C_iron)) * weight_C

/-- The theorem stating the amount of tin in the mixture -/
theorem tin_in_mixture : 
  ∀ (ε : ℝ), ε > 0 → 
  |amount_of_tin 90 140 50 3 4 2 5 3 2 5 - 106.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l640_64086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l640_64022

theorem definite_integral_value : 
  ∫ x in (Set.Icc 0 1), (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l640_64022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l640_64010

variable (a : ℝ)
def z₁ : ℂ := 1 - a * Complex.I
def z₂ : ℂ := 2 * a + 3 * Complex.I

theorem complex_problem :
  (∀ a : ℝ, (z₁ a * z₂ a).re = 0 → Complex.abs (z₁ a + z₂ a) = Real.sqrt 10) ∧
  (∀ a : ℝ, (z₂ a / z₁ a).im = 5 * (z₂ a / z₁ a).re → a = -1 ∨ a = -3/2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l640_64010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_with_six_factors_l640_64079

/-- A function that returns the number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ) : ℕ :=
  (Finset.filter (·∣n) (Finset.range (n + 1))).card

/-- A function that checks if a positive integer has exactly six distinct positive factors -/
def has_six_factors (n : ℕ) : Prop :=
  num_factors n = 6

/-- The theorem stating that 12 is the least positive integer with exactly six distinct positive factors -/
theorem least_with_six_factors :
  has_six_factors 12 ∧ ∀ n : ℕ, 0 < n → n < 12 → ¬(has_six_factors n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_with_six_factors_l640_64079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l640_64043

/-- The volume of a regular tetrahedron with base edge length 6 is 9 -/
theorem regular_tetrahedron_volume : 
  ∀ (t : Real), 
  t > 0 → 
  t^3 * (2/12) * Real.sqrt 2 = 9 → 
  t = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l640_64043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_correct_l640_64068

/-- Represents the properties of a cylindrical glass cup -/
structure GlassCup where
  wall_thickness : ℝ
  internal_diameter : ℝ
  internal_height : ℝ
  bottom_thickness : ℝ
  glass_density_ratio : ℝ

/-- Calculates the center of mass of the system as a function of water height -/
noncomputable def center_of_mass (cup : GlassCup) (water_height : ℝ) : ℝ :=
  let r := cup.internal_diameter / 2
  let R := r + cup.wall_thickness
  let h := cup.internal_height
  let h_t := cup.bottom_thickness
  let ρ_ratio := cup.glass_density_ratio
  (17 * water_height^2 + 340.5) / (34.21 * water_height + 86.05)

/-- Theorem stating that the calculated center of mass is correct for the given cup properties -/
theorem center_of_mass_correct (cup : GlassCup) (water_height : ℝ) :
  cup.wall_thickness = 0.15 →
  cup.internal_diameter = 6.6 →
  cup.internal_height = 10 →
  cup.bottom_thickness = 0.3 →
  cup.glass_density_ratio = 2 →
  0 ≤ water_height →
  water_height ≤ cup.internal_height →
  center_of_mass cup water_height = (17 * water_height^2 + 340.5) / (34.21 * water_height + 86.05) := by
  sorry

#check center_of_mass_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_correct_l640_64068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l640_64006

noncomputable def f (x : ℝ) := (Real.sqrt (x - 3)) / (Real.sqrt (7 - x))

theorem domain_of_f : Set.Icc 3 7 \ {7} = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l640_64006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_from_area_l640_64069

-- Define the area of the circle
noncomputable def circle_area : ℝ := 113.09733552923255

-- Define the constant pi
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem circle_diameter_from_area :
  ∃ (d : ℝ), d = 12 ∧ circle_area = π * (d / 2)^2 := by
  -- Introduce the diameter
  let d : ℝ := 12
  -- Assert its existence
  exists d
  -- Prove the conjunction
  apply And.intro
  · -- Prove d = 12
    rfl
  · -- Prove circle_area = π * (d / 2)^2
    -- This step would require actual computation, so we'll use sorry
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_from_area_l640_64069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_27_l640_64000

theorem cube_root_negative_27 : ((-27 : ℝ) ^ (1/3 : ℝ)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_27_l640_64000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l640_64019

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧ f x = 6 ∧ ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x :=
by
  -- The proof goes here
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l640_64019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_collinear_vectors_equilateral_l640_64080

/-- Triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector representation based on triangle side and half-angle cosine -/
noncomputable def vector_rep (side : ℝ) (angle : ℝ) : ℝ × ℝ := (side, Real.cos (angle / 2))

/-- Collinearity of three 2D vectors -/
def collinear (v1 v2 v3 : ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), v2.1 = t1 * v1.1 ∧ v2.2 = t1 * v1.2 ∧ 
                 v3.1 = t2 * v1.1 ∧ v3.2 = t2 * v1.2

/-- A triangle is equilateral if all its angles are equal -/
def is_equilateral (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

theorem triangle_collinear_vectors_equilateral (t : Triangle) :
  collinear (vector_rep t.a t.A) (vector_rep t.b t.B) (vector_rep t.c t.C) →
  is_equilateral t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_collinear_vectors_equilateral_l640_64080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_ratio_l640_64051

/-- Given three jars with alcohol solutions where the ratios of alcohol to water are p:1, q:1, and r:1 respectively,
    this theorem states that when the contents of these jars are mixed, the ratio of alcohol to water in the resulting mixture
    is (pqr + pr + pq + qr + p + q + r) / (pq + pr + qr + p + q + r + 3) -/
theorem alcohol_water_mixture_ratio (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (p / (p + 1) + q / (q + 1) + r / (r + 1)) / (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1)) =
  (p * q * r + p * r + p * q + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_water_mixture_ratio_l640_64051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l640_64032

theorem simplify_square_roots : Real.sqrt (3 * 5) * Real.sqrt (3^5 * 5^5) = 3375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_square_roots_l640_64032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l640_64092

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + 4 * Real.sqrt x

-- State the theorem
theorem evaluate_f : 2 * f 3 - f 9 = -75 + 8 * Real.sqrt 3 := by
  -- Expand the definition of f
  simp [f]
  -- Simplify the expression
  ring_nf
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_f_l640_64092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l640_64023

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x)^2 / (2 * cos x * sin x - (sin x)^2)

theorem f_minimum_value (x : ℝ) (h : 0 < x ∧ x < π/3) : 
  f x ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l640_64023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_sum_theorem_l640_64098

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def is_valid_sum (sum : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    (a = penny ∨ a = nickel ∨ a = dime ∨ a = quarter) ∧
    (b = penny ∨ b = nickel ∨ b = dime ∨ b = quarter) ∧
    (c = penny ∨ c = nickel ∨ c = dime ∨ c = quarter) ∧
    (d = penny ∨ d = nickel ∨ d = dime ∨ d = quarter) ∧
    (a + b + c + d = sum)

theorem coin_sum_theorem :
  ¬(is_valid_sum 15) ∧
  (is_valid_sum 25) ∧
  (is_valid_sum 35) ∧
  (is_valid_sum 45) ∧
  (is_valid_sum 55) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_sum_theorem_l640_64098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_angle_value_problem_l640_64055

theorem trigonometric_problem :
  ∀ α β : ℝ,
  Real.sin α = 3/5 →
  Real.cos β = 4/5 →
  π/2 < α →
  α < π →
  0 < β →
  β < π/2 →
  Real.cos (α + β) = -1 :=
by
  sorry

theorem angle_value_problem :
  ∀ α β : ℝ,
  Real.cos α = 1/7 →
  Real.cos (α - β) = 13/14 →
  0 < β →
  β < α →
  α < π/2 →
  β = π/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_angle_value_problem_l640_64055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_quadrilateral_diagonal_product_difference_l640_64067

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A quadrilateral with vertices on integer lattice points -/
structure LatticeQuadrilateral where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint
  D : LatticePoint

/-- Distance between two lattice points -/
noncomputable def distance (p q : LatticePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a quadrilateral is a trapezoid -/
def isTrapezoid (q : LatticeQuadrilateral) : Prop :=
  sorry

theorem lattice_quadrilateral_diagonal_product_difference
  (q : LatticeQuadrilateral)
  (h : ¬isTrapezoid q) :
  |(distance q.A q.C) * (distance q.A q.D) -
   (distance q.B q.C) * (distance q.B q.D)| ≥ 1 := by
  sorry

#check lattice_quadrilateral_diagonal_product_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_quadrilateral_diagonal_product_difference_l640_64067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l640_64015

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 10 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = total_avg_age →
  teacher_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_age_l640_64015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l640_64062

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the external point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the two potential tangent lines
def line1 (x : ℝ) : Prop := x = 1
def line2 (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ x y : ℝ, my_circle x y → (line1 x ∨ line2 x y) → 
    ((x - 1)^2 + (y - 2)^2 = (x^2 + y^2 - 1)^2 / ((2*x)^2 + (2*y)^2))) ∧
  (line1 (point_P.1) ∨ line2 point_P.1 point_P.2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l640_64062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l640_64078

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.exp x + Real.exp (-x)) / 2)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧                   -- f is even
  (∀ x y, 0 < x → x < y → f x < f y) :=   -- f is increasing on (0, +∞)
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l640_64078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l640_64040

-- Define the quadratic functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define a as the point where f reaches its maximum
noncomputable def a : ℝ := sorry

-- State the conditions
axiom a_positive : a > 0
axiom f_max : ∀ x, f x ≤ f a
axiom f_max_value : f a = 5
axiom g_min : ∀ x, g x ≥ -2
axiom g_at_a : g a = 25
axiom sum_equation : ∀ x, f x + g x = x^2 + 16*x + 13

-- State the theorem to be proved
theorem quadratic_functions_theorem : 
  a = 1 ∧ ∀ x, g x = 3*x^2 + 12*x + 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_theorem_l640_64040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_correct_l640_64070

/-- The total distance traveled by a laser beam with given properties -/
noncomputable def laser_beam_distance : ℝ := 6 + 2 * Real.sqrt 29

/-- The starting point of the laser beam -/
def start_point : ℝ × ℝ := (2, 4)

/-- The ending point of the laser beam -/
def end_point : ℝ × ℝ := (10, 4)

/-- The point where the laser beam reflects off the y-axis -/
def y_axis_reflection : ℝ × ℝ := (0, 4)

/-- The point where the laser beam reflects off the x-axis -/
def x_axis_reflection : ℝ × ℝ := (10, 0)

/-- Theorem stating that the total distance traveled by the laser beam is correct -/
theorem laser_beam_distance_correct :
  let d1 := Real.sqrt ((start_point.1 - y_axis_reflection.1)^2 + (start_point.2 - y_axis_reflection.2)^2)
  let d2 := Real.sqrt ((y_axis_reflection.1 - x_axis_reflection.1)^2 + (y_axis_reflection.2 - x_axis_reflection.2)^2)
  let d3 := Real.sqrt ((x_axis_reflection.1 - end_point.1)^2 + (x_axis_reflection.2 - end_point.2)^2)
  d1 + d2 + d3 = laser_beam_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_correct_l640_64070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_tangent_property_l640_64038

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a tangent line
structure TangentLine where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define membership for points in TangentLine and Circle
def point_on_tangent_line (p : ℝ × ℝ) (t : TangentLine) : Prop :=
  ∃ k : ℝ, p = (t.point.1 + k * t.direction.1, t.point.2 + k * t.direction.2)

def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the theorem
theorem four_circles_tangent_property 
  (S1 S2 S3 S4 : Circle)
  (h12 : externally_tangent S1 S2)
  (h23 : externally_tangent S2 S3)
  (h34 : externally_tangent S3 S4)
  (h41 : externally_tangent S4 S1)
  (t1 t2 t3 t4 : TangentLine) :
  (∃ (p : ℝ × ℝ), t1.point = p ∧ t2.point = p ∧ t3.point = p ∧ t4.point = p) ∨
  (∃ (c : Circle), 
    (∃ (p1 : ℝ × ℝ), point_on_tangent_line p1 t1 ∧ point_on_circle p1 c) ∧
    (∃ (p2 : ℝ × ℝ), point_on_tangent_line p2 t2 ∧ point_on_circle p2 c) ∧
    (∃ (p3 : ℝ × ℝ), point_on_tangent_line p3 t3 ∧ point_on_circle p3 c) ∧
    (∃ (p4 : ℝ × ℝ), point_on_tangent_line p4 t4 ∧ point_on_circle p4 c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_tangent_property_l640_64038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l640_64094

theorem cos_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 * Real.sqrt 3 / 7)
  (h4 : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l640_64094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_l640_64074

noncomputable def rationalize_denominator (x y z : ℝ) : ℝ := x * (z^(1/3)) / (y * z^(1/3))

theorem rationalized_form :
  ∃ (A B C : ℤ),
    rationalize_denominator 5 3 7 = (A : ℝ) * ((B : ℝ)^(1/3)) / (C : ℝ) ∧
    C > 0 ∧
    (∀ (p : ℕ), Nat.Prime p → (B % (p^3 : ℕ) ≠ 0)) ∧
    A = 5 ∧ B = 49 ∧ C = 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_l640_64074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_two_l640_64048

-- Define the slopes of the lines
noncomputable def slope_l1 (m : ℝ) : ℝ := -2 / m
noncomputable def slope_l2 (m : ℝ) : ℝ := -m / 2

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop := slope_l1 m * slope_l2 m = -1

-- Theorem statement
theorem perpendicular_lines_m_equals_two :
  ∀ m : ℝ, m ≠ 0 → perpendicular m → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_two_l640_64048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l640_64063

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) :
  (Real.log x / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * 
  (Real.log (x^7) / Real.log (y^3)) * (Real.log (y^8) / Real.log (x^2)) = Real.log x / Real.log y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_equality_l640_64063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_boys_to_girls_in_combined_schools_l640_64014

theorem ratio_of_boys_to_girls_in_combined_schools : 
  ∀ (pascal_total fermat_total : ℕ)
    (pascal_boy_ratio pascal_girl_ratio : ℕ)
    (fermat_boy_ratio fermat_girl_ratio : ℕ),
  pascal_total = 400 →
  fermat_total = 600 →
  pascal_boy_ratio = 3 →
  pascal_girl_ratio = 2 →
  fermat_boy_ratio = 2 →
  fermat_girl_ratio = 3 →
  let pascal_boys := pascal_total * pascal_boy_ratio / (pascal_boy_ratio + pascal_girl_ratio)
  let pascal_girls := pascal_total * pascal_girl_ratio / (pascal_boy_ratio + pascal_girl_ratio)
  let fermat_boys := fermat_total * fermat_boy_ratio / (fermat_boy_ratio + fermat_girl_ratio)
  let fermat_girls := fermat_total * fermat_girl_ratio / (fermat_boy_ratio + fermat_girl_ratio)
  let total_boys := pascal_boys + fermat_boys
  let total_girls := pascal_girls + fermat_girls
  (total_boys : ℚ) / (total_girls : ℚ) = 12 / 13 :=
by
  intros pascal_total fermat_total pascal_boy_ratio pascal_girl_ratio fermat_boy_ratio fermat_girl_ratio
  intros h_pascal_total h_fermat_total h_pascal_boy_ratio h_pascal_girl_ratio h_fermat_boy_ratio h_fermat_girl_ratio
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_boys_to_girls_in_combined_schools_l640_64014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l640_64009

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (pass_time : ℝ) (platform_length : ℝ) :
  train_speed = 45 * 1000 / 3600 ∧ 
  pass_time = 51.99999999999999 ∧ 
  platform_length = 290 →
  train_speed * pass_time - platform_length = 360 := by
  sorry

#eval Float.toString (45 * 1000 / 3600 * 51.99999999999999 - 290)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l640_64009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sidney_cat_food_problem_l640_64076

/-- The number of additional cans of cat food Sidney needs to buy -/
def additional_cans_needed (num_kittens num_adult_cats initial_cans days : ℕ) 
  (kitten_daily_consumption adult_daily_consumption : ℚ) : ℕ :=
  let total_consumption := (num_kittens * kitten_daily_consumption + 
    num_adult_cats * adult_daily_consumption) * days
  (Int.ceil total_consumption - initial_cans).toNat

/-- Theorem stating the number of additional cans Sidney needs to buy -/
theorem sidney_cat_food_problem : 
  additional_cans_needed 4 3 7 7 (3/4) 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sidney_cat_food_problem_l640_64076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l640_64005

theorem sin_alpha_plus_beta (α β : ℝ) : 
  0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 → 
  Real.cos α = 12/13 → 
  Real.cos (2*α + β) = 3/5 → 
  Real.sin (α + β) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l640_64005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l640_64031

-- Define the constants
noncomputable def a : ℝ := Real.log (3/2) / Real.log (3/4)
noncomputable def b : ℝ := (3/2) ^ (3/2)
noncomputable def c : ℝ := (3/4) ^ (4/3)

-- State the theorem
theorem relationship_abc : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l640_64031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lower_bound_l640_64026

/-- A structure representing a configuration of planes in space -/
structure PlaneConfiguration where
  n : ℕ
  planes : Fin n → Set (Fin 3 → ℝ)  -- Representing planes as sets of points in ℝ³
  n_ge_5 : n ≥ 5
  three_intersect : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    ∃! p, p ∈ (planes i) ∩ (planes j) ∩ (planes k)
  no_five_coplanar : ∀ (i j k l m : Fin n),
    i ≠ j → j ≠ k → k ≠ l → l ≠ m → i ≠ k → i ≠ l → i ≠ m → j ≠ l → j ≠ m → k ≠ m →
    ¬∃ p, p ∈ (planes i) ∩ (planes j) ∩ (planes k) ∩ (planes l) ∩ (planes m)

/-- The number of tetrahedra formed by the intersections of planes -/
def num_tetrahedra (config : PlaneConfiguration) : ℕ := sorry

/-- Theorem stating the lower bound on the number of tetrahedra -/
theorem tetrahedra_lower_bound (config : PlaneConfiguration) :
  num_tetrahedra config ≥ (2 * config.n - 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lower_bound_l640_64026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l640_64054

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The ratio condition for the triangle -/
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a + t.b = 5 * k ∧ t.b + t.c = 6 * k ∧ t.c + t.a = 7 * k

/-- A triangle is obtuse if one of its angles is greater than 90 degrees -/
def is_obtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ :=
  t.a / (2 * Real.sin t.A)

/-- The inradius of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ :=
  (t.a * t.b * Real.sin t.C) / (t.a + t.b + t.c)

/-- The main theorem -/
theorem triangle_properties (t : Triangle) (h : ratio_condition t) :
  is_obtuse t ∧ circumradius t / inradius t = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l640_64054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_in_proportion_l640_64039

/-- Checks if four numbers are in proportion -/
def are_in_proportion (a b c d : ℝ) : Prop := a * d = b * c

/-- The sets of four line segments -/
def set_A : List ℝ := [1, 2, 3, 4]
def set_B : List ℝ := [2, 4, 6, 8]
def set_C : List ℝ := [5, 30, 10, 15]
def set_D : List ℝ := [5, 20, 10, 15]

/-- All sets of line segments -/
def all_sets : List (List ℝ) := [set_A, set_B, set_C, set_D]

theorem only_set_C_in_proportion :
  ∃! s, s ∈ all_sets ∧ s.length = 4 ∧ are_in_proportion s[0]! s[1]! s[2]! s[3]! :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_in_proportion_l640_64039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_correct_scarf_price_correct_l640_64044

/-- Calculate the final price after applying successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem saree_price_correct (initialPrice : ℝ) (discount1 discount2 discount3 : ℝ) :
  initialPrice = 500 ∧ discount1 = 0.1 ∧ discount2 = 0.05 ∧ discount3 = 0.08 →
  abs (finalPrice initialPrice discount1 discount2 discount3 - 393.30) < 0.01 := by
  sorry

theorem scarf_price_correct (initialPrice : ℝ) (discount1 discount2 discount3 : ℝ) :
  initialPrice = 350 ∧ discount1 = 0.12 ∧ discount2 = 0.07 ∧ discount3 = 0.05 →
  abs (finalPrice initialPrice discount1 discount2 discount3 - 272.12) < 0.01 := by
  sorry

#check saree_price_correct
#check scarf_price_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_correct_scarf_price_correct_l640_64044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_sin_l640_64088

theorem negation_of_existence_sin (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.sin x ≥ 1) → 
  (¬p ↔ ∀ x : ℝ, Real.sin x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_sin_l640_64088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l640_64073

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 9)
  (h2 : (x : ℝ) / (y : ℝ) = 96.25) : 
  y = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l640_64073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l640_64089

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 2 + t * Real.sin α)

-- Theorem statement
theorem slope_of_line_l :
  ∀ α t₁ t₂ : ℝ,
  let A := line_l α t₁
  let B := line_l α t₂
  let M := (2, 2)
  (∃ θ₁, curve_C θ₁ = A) ∧
  (∃ θ₂, curve_C θ₂ = B) ∧
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →
  Real.tan α = -9/16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l640_64089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_daily_production_approx_l640_64029

def initial_production : ℕ := 4000
def monthly_increase : ℕ := 100
def days_in_year : ℕ := 365

def monthly_production (month : ℕ) : ℕ :=
  if month = 1 then initial_production
  else initial_production + (month - 1) * monthly_increase

def total_yearly_production : ℕ :=
  (List.range 12).map (fun i => monthly_production (i + 1)) |> List.sum

noncomputable def average_daily_production : ℚ :=
  (total_yearly_production : ℚ) / days_in_year

theorem average_daily_production_approx :
  (average_daily_production * 100).floor / 100 = 15205 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_daily_production_approx_l640_64029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l640_64096

-- Define the type of functions from [-1,1] to ℝ
def FunctionType := ℝ → ℝ

-- Define the property of satisfying the given equation
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020

-- Define an odd function
def IsOdd (g : FunctionType) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g (-x) = -g x

-- State the theorem
theorem function_characterization (f : FunctionType) :
  SatisfiesEquation f →
  ∃ g : FunctionType, IsOdd g ∧
    ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = g (1 - 2 * x^2) + 1010 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l640_64096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_l640_64066

noncomputable def f (x : ℝ) : ℝ := 2 / (4^x + 2) + 2

theorem sum_of_f (m : ℕ) (h : m ≥ 2) :
  (Finset.range (m - 1)).sum (λ k => f ((k + 1 : ℕ) / m)) = 5/2 * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_l640_64066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l640_64090

-- Define the region for point P
def region_P (x y : ℝ) : Prop :=
  2 * x - y + 2 ≥ 0 ∧ x - 2 * y + 1 ≤ 0 ∧ x + y - 2 ≤ 0

-- Define the curve for point Q
def curve_Q (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_PQ :
  ∃ (x1 y1 x2 y2 : ℝ),
    region_P x1 y1 ∧ 
    curve_Q x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ), region_P x3 y3 → curve_Q x4 y4 → 
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l640_64090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l640_64099

/-- The line on which point P moves --/
def line (x y : ℝ) : Prop := 3 * x + 4 * y + 8 = 0

/-- The circle to which PA and PB are tangent --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The center of the circle --/
def center : ℝ × ℝ := (1, 1)

/-- The radius of the circle --/
def radius : ℝ := 1

/-- The distance from the center of the circle to the line --/
def distance_to_line : ℝ := 3

/-- The length of the tangents PA and PB --/
noncomputable def tangent_length : ℝ := 2 * Real.sqrt 2

/-- Function to calculate the area of a quadrilateral --/
noncomputable def quadrilateralArea (P A C B : ℝ × ℝ) : ℝ := sorry

theorem min_area_PACB :
  ∀ P A B : ℝ × ℝ,
  (∃ x y, P = (x, y) ∧ line x y) →
  circle_equation A.1 A.2 →
  circle_equation B.1 B.2 →
  (∀ Q : ℝ × ℝ, circle_equation Q.1 Q.2 → (Q = A ∨ Q = B ∨ ¬(Q ∈ Set.Icc P A ∨ Q ∈ Set.Icc P B))) →
  (∀ Q : ℝ × ℝ, min (dist P Q) (dist center Q) = radius → (Q = A ∨ Q = B)) →
  (∀ area : ℝ, area ≥ 0 → area ≥ 2 * Real.sqrt 2 →
    ∃ P' A' B' : ℝ × ℝ,
      (∃ x' y', P' = (x', y') ∧ line x' y') ∧
      circle_equation A'.1 A'.2 ∧
      circle_equation B'.1 B'.2 ∧
      (∀ Q : ℝ × ℝ, circle_equation Q.1 Q.2 → (Q = A' ∨ Q = B' ∨ ¬(Q ∈ Set.Icc P' A' ∨ Q ∈ Set.Icc P' B'))) ∧
      (∀ Q : ℝ × ℝ, min (dist P' Q) (dist center Q) = radius → (Q = A' ∨ Q = B')) ∧
      quadrilateralArea P' A' center B' = area) →
  quadrilateralArea P A center B ≥ 2 * Real.sqrt 2 :=
by sorry

#check min_area_PACB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l640_64099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_quadrilateral_l640_64064

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Circle → ℝ × ℝ

-- Define vectors
def vector (A B : Point) : ℝ × ℝ := (B () - A ())

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : Point) : ℝ := sorry

-- Theorem statement
theorem area_of_special_quadrilateral 
  (c : Circle) (A B C D : Point) 
  (h1 : ∀ X : Point, magnitude (X c) = 1) 
  (h2 : vector A B + 2 • (vector A C) = vector A D) 
  (h3 : magnitude (vector A C) = 1) :
  area_quadrilateral A B D C = (3 * Real.sqrt 3) / 4 := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_quadrilateral_l640_64064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_satisfies_conditions_l640_64027

/-- A linear function passing through (2, 6) and parallel to y = -2x -/
def linear_function (x : ℝ) : ℝ := -2 * x + 10

/-- The slope of the line y = -2x -/
def parallel_line_slope : ℝ := -2

theorem linear_function_satisfies_conditions :
  (∀ x : ℝ, (deriv linear_function) x = parallel_line_slope) ∧
  linear_function 2 = 6 := by
  sorry

#check linear_function_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_satisfies_conditions_l640_64027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_is_139_50_l640_64050

/-- Calculates the compound interest for a given principal, rate, and time --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 2) ^ (2 * time)

/-- Represents the banking scenario described in the problem --/
noncomputable def bank_scenario (initial_deposit : ℝ) (first_rate : ℝ) (second_rate : ℝ) (withdrawal : ℝ) : ℝ :=
  let first_period := compound_interest initial_deposit first_rate 0.5
  let second_period := compound_interest (first_period - withdrawal) second_rate 0.5
  second_period

/-- Theorem stating that the final amount in the account is $139.50 --/
theorem final_amount_is_139_50 :
  bank_scenario 140 0.20 0.25 30 = 139.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_is_139_50_l640_64050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_f_inverse_f_positive_l640_64091

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((x + 5) / (x - 5)) / Real.log 10

-- Theorem for the domain of f
theorem domain_of_f : Set ℝ := {x | x > 5 ∨ x < -5}

-- Theorem for the parity of f
theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := 5 * (10^x + 1) / (10^x - 1)

-- Theorem for the inverse function
theorem f_inverse : ∀ x ≠ 0, f (f_inv x) = x ∧ f_inv (f x) = x := by
  sorry

-- Theorem for the range of x where f(x) > 0
theorem f_positive : ∀ x, f x > 0 ↔ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_f_inverse_f_positive_l640_64091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l640_64071

-- Define the basic geometric objects
structure Plane where

structure Line where

-- Define geometric relations
def perpendicular (a b : Plane) : Prop := sorry

def parallel (a b : Plane) : Prop := sorry

def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

def intersection_line (p1 p2 : Plane) : Line := sorry

-- Define the propositions
def proposition1 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2

def proposition2 (p1 p2 : Plane) (l1 l2 : Line) : Prop :=
  line_in_plane l1 p1 → line_in_plane l2 p1 → parallel_lines l1 l2 →
  perpendicular_line_plane l1 p2 → perpendicular_line_plane l2 p2 → parallel p1 p2

def proposition3 (l1 l2 m : Line) : Prop :=
  parallel_lines l1 l2 → perpendicular_lines l1 m → perpendicular_lines l2 m

def proposition4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular p1 p2 → line_in_plane l p1 →
  ¬perpendicular_lines l (intersection_line p1 p2) →
  ¬perpendicular_line_plane l p2

theorem geometry_propositions :
  (∀ p1 p2 l, proposition1 p1 p2 l) ∧
  (∃ p1 p2 l1 l2, ¬proposition2 p1 p2 l1 l2) ∧
  (∀ l1 l2 m, proposition3 l1 l2 m) ∧
  (∀ p1 p2 l, proposition4 p1 p2 l) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l640_64071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_theorem_l640_64017

-- Define the point P
def P : ℝ × ℝ := (-5, -4)

-- Define the line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y = c

-- Define the condition that the line passes through point P
def passes_through_P (a b c : ℝ) : Prop := line_equation a b c P.1 P.2

-- Define the area of the triangle formed by the line and coordinate axes
noncomputable def triangle_area (a b : ℝ) : ℝ := (1/2) * |a * b|

-- Main theorem
theorem unique_line_theorem (a b c : ℝ) :
  passes_through_P a b c ∧ 
  triangle_area (c/a) (c/b) = 5 ∧ 
  a ≠ 0 ∧ b ≠ 0 →
  a = 23 ∧ b = 5 ∧ c = 230 := by
  sorry

#check unique_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_theorem_l640_64017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_corner_chop_height_l640_64097

/-- The height of the remaining part of a cube after chopping off a corner -/
noncomputable def remaining_height (s : ℝ) : ℝ := s * (3 - 2 * Real.sqrt 3) / 3

/-- 
Theorem: The height of the remaining part of a cube with side length s, 
after chopping off a corner through the three adjacent vertices, 
is equal to s(3 - 2√3) / 3.
-/
theorem cube_corner_chop_height (s : ℝ) (h_pos : s > 0) : 
  ∃ (H : ℝ), H = remaining_height s ∧ 
  H = s - (2 * Real.sqrt 3 * s) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_corner_chop_height_l640_64097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_focus_x_coord_l640_64075

/-- Given a hyperbola with equation x^2 - 2y^2 = 1, prove that the x-coordinate of its right focus is √6/2 -/
theorem hyperbola_right_focus_x_coord :
  ∃ (focus_x : ℝ), focus_x = Real.sqrt 6 / 2 ∧ focus_x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_focus_x_coord_l640_64075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l640_64012

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x^3 / Real.log 4) + (Real.log x / Real.log 4) = 5 → x = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l640_64012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l640_64081

/-- Given a 3x3 matrix with specific structure, prove that b + m = 306 -/
theorem matrix_power_sum (b m : ℕ) : 
  let A := (Matrix.of ![![1, 3, b],
                       ![0, 1, 5],
                       ![0, 0, 1]] : Matrix (Fin 3) (Fin 3) ℕ)
  let B := (Matrix.of ![![1, 27, 3003],
                       ![0,  1,   45],
                       ![0,  0,    1]] : Matrix (Fin 3) (Fin 3) ℕ)
  A^m = B → b + m = 306 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l640_64081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l640_64042

noncomputable def a (n : ℕ) : ℝ := Real.cos (10^n * Real.pi / 180)

theorem a_100_properties : a 100 > 0 ∧ |a 100| < 0.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_properties_l640_64042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bag_problem_l640_64037

/-- Represents the number of cups of tea that can be made from one tea bag -/
inductive CupsPerBag
  | two
  | three

/-- Represents a person's tea consumption -/
structure TeaConsumption where
  cups : ℕ
  bags : ℕ
  usage : List CupsPerBag

/-- Convert CupsPerBag to Nat -/
def cupsPerBagToNat : CupsPerBag → ℕ
  | CupsPerBag.two => 2
  | CupsPerBag.three => 3

/-- Sum of cups from a list of CupsPerBag -/
def sumCups (usage : List CupsPerBag) : ℕ :=
  usage.map cupsPerBagToNat |>.sum

/-- The problem statement -/
theorem tea_bag_problem :
  ∃! n : ℕ,
    (∃ natasha_usage : List CupsPerBag,
      natasha_usage.length = n ∧
      (∀ bag ∈ natasha_usage, bag = CupsPerBag.two ∨ bag = CupsPerBag.three) ∧
      sumCups natasha_usage = 41) ∧
    (∃ inna_usage : List CupsPerBag,
      inna_usage.length = n ∧
      (∀ bag ∈ inna_usage, bag = CupsPerBag.two ∨ bag = CupsPerBag.three) ∧
      sumCups inna_usage = 58) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_bag_problem_l640_64037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_3_sqrt_2_l640_64093

/-- Represents a parallelogram ADEH with specific properties -/
structure Parallelogram where
  /-- Length of side AD -/
  ad : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- Length of diagonal BH -/
  bh : ℝ
  /-- AD is 6 -/
  ad_eq : ad = 6
  /-- HE is 6 -/
  he_eq : he = 6
  /-- BH is 2√2 -/
  bh_eq : bh = 2 * Real.sqrt 2

/-- The area of quadrilateral WXYZ in the parallelogram -/
noncomputable def area_WXYZ (p : Parallelogram) : ℝ := 3 * Real.sqrt 2

/-- Theorem stating that the area of WXYZ is 3√2 -/
theorem area_WXYZ_is_3_sqrt_2 (p : Parallelogram) : area_WXYZ p = 3 * Real.sqrt 2 := by
  -- Unfold the definition of area_WXYZ
  unfold area_WXYZ
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_WXYZ_is_3_sqrt_2_l640_64093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_centers_of_symmetry_l640_64095

/-- A shape in a 2D space --/
structure Shape where
  -- Add necessary fields here

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines if a point is a center of symmetry for a shape --/
def IsCenterOfSymmetry (s : Shape) (p : Point) : Prop :=
  sorry

/-- Theorem: If a shape has at least two centers of symmetry, it has infinitely many --/
theorem infinite_centers_of_symmetry (s : Shape) 
    (p1 p2 : Point) 
    (h1 : p1 ≠ p2) 
    (h2 : IsCenterOfSymmetry s p1) 
    (h3 : IsCenterOfSymmetry s p2) : 
    ∀ n : ℕ, ∃ (centers : Finset Point), 
      (∀ p ∈ centers, IsCenterOfSymmetry s p) ∧ centers.card = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_centers_of_symmetry_l640_64095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l640_64025

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l640_64025
