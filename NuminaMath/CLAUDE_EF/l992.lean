import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l992_99299

/-- The parabola equation y = x^2 - 4x + 3 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

/-- The area of triangle ABC given points A(x₁,y₁), B(x₂,y₂), C(x₃,y₃) -/
noncomputable def triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs (x₁*y₂ + x₂*y₃ + x₃*y₁ - y₁*x₂ - y₂*x₃ - y₃*x₁)

theorem max_triangle_area :
  ∀ p q : ℝ,
  2 ≤ p → p ≤ 5 →
  parabola 2 0 →
  parabola 5 2 →
  parabola p q →
  (∀ p' q' : ℝ, 2 ≤ p' → p' ≤ 5 → parabola p' q' →
    triangle_area 2 0 5 2 p' q' ≤ triangle_area 2 0 5 2 5 2) →
  triangle_area 2 0 5 2 5 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l992_99299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_without_three_together_l992_99274

/-- The number of ways to arrange 6 people in a row such that 3 specific people are not all standing together -/
def arrangementsWithoutThreeTogether : ℕ := 576

/-- Total number of people -/
def totalPeople : ℕ := 6

/-- Number of specific people (A, B, C) -/
def specificPeople : ℕ := 3

theorem arrangements_without_three_together :
  arrangementsWithoutThreeTogether = 
    Nat.factorial totalPeople - Nat.factorial (totalPeople - specificPeople + 1) * Nat.factorial specificPeople :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_without_three_together_l992_99274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l992_99243

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-x)

noncomputable def tangent_slope (x : ℝ) : ℝ := -Real.exp (-x)

def parallel_line_slope : ℝ := -2

theorem tangent_point_coordinates : 
  let P : ℝ × ℝ := (-Real.log 2, 2)
  (curve P.1 = P.2) ∧ 
  (tangent_slope P.1 = parallel_line_slope) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l992_99243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_bounded_l992_99264

def sequence_bounded (p : ℕ → ℕ) : Prop :=
  ∃ P, ∀ n, p n ≤ P

theorem prime_sequence_bounded
  (p : ℕ → ℕ)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_init : ∃ p₀ p₁, p 0 = p₀ ∧ p 1 = p₁)
  (h_next : ∀ n ≥ 1, p (n + 1) = (p (n - 1) + p n + 100).factors.last') :
  sequence_bounded p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_bounded_l992_99264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laran_poster_business_l992_99252

/-- Represents the number of large posters sold per day -/
def large_posters : ℕ := sorry

/-- Represents the number of small posters sold per day -/
def small_posters : ℕ := sorry

/-- The total number of posters sold per day -/
def total_posters : ℕ := 5

/-- The profit from a large poster -/
def large_profit : ℕ := 5

/-- The profit from a small poster -/
def small_profit : ℕ := 3

/-- The total daily profit -/
def daily_profit : ℕ := 19

theorem laran_poster_business :
  large_posters + small_posters = total_posters ∧
  large_posters * large_profit + small_posters * small_profit = daily_profit →
  large_posters = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laran_poster_business_l992_99252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_exceeds_60_percent_on_wednesday_l992_99202

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℚ
  total : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet * (1 - 0.3) + 0.4,
    total := 1 }

/-- Checks if the proportion of millet exceeds 60% -/
def milletExceeds60Percent (state : FeederState) : Bool :=
  state.millet / state.total > 0.6

/-- Initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, millet := 0.4, total := 1 }

theorem millet_exceeds_60_percent_on_wednesday :
  let state3 := nextDay (nextDay initialState)
  milletExceeds60Percent state3 ∧
  ∀ (n : ℕ), n < 3 → ¬milletExceeds60Percent (Nat.iterate nextDay n initialState) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_exceeds_60_percent_on_wednesday_l992_99202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_g_zeros_l992_99226

noncomputable section

def f (a x : ℝ) : ℝ := Real.log (1 / x + a) / Real.log 2

def g (a x : ℝ) : ℝ := f a x - Real.log ((a - 4) * x + 2 * a - 5) / Real.log 2

theorem f_range_and_g_zeros (a : ℝ) :
  (f a 1 < 2 ↔ -1 < a ∧ a < 3) ∧
  ((∀ x, g a x ≠ 0) ↔ a ≤ 4/5) ∧
  ((∃! x, g a x = 0) ↔ (4/5 < a ∧ a ≤ 1) ∨ a = 3 ∨ a = 4) ∧
  ((∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0) ↔ (1 < a ∧ a ≠ 3 ∧ a ≠ 4)) :=
by sorry

#check f_range_and_g_zeros


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_g_zeros_l992_99226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_bags_combined_weight_l992_99225

/-- The weight of James's bag in kilograms -/
noncomputable def james_bag_weight : ℝ := 18

/-- The ratio of Oliver's bag weight to James's bag weight -/
noncomputable def oliver_bag_ratio : ℝ := 1 / 6

/-- The number of bags Oliver has -/
def oliver_bag_count : ℕ := 2

/-- Theorem: The combined weight of Oliver's bags is 6 kg -/
theorem oliver_bags_combined_weight :
  (oliver_bag_count : ℝ) * (oliver_bag_ratio * james_bag_weight) = 6 := by
  sorry

#eval oliver_bag_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_bags_combined_weight_l992_99225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l992_99294

-- Define the hyperbola
def hyperbola (a b : ℝ) : Prop := 
  a > 0 ∧ b > 0

-- Define the asymptote line
def asymptote_line (x y : ℝ) : Prop := x + 2*y + 5 = 0

-- Define the condition that the asymptote is parallel to the line
def asymptote_parallel (a b : ℝ) : Prop := b / a = 1 / 2

-- Define the condition that a focus lies on the line
def focus_on_line (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), asymptote_line x y ∧ x^2 + y^2 = a^2 + b^2

-- State the theorem
theorem hyperbola_equation (a b : ℝ) : 
  hyperbola a b ∧ 
  asymptote_parallel a b ∧ 
  focus_on_line a b → 
  a^2 = 20 ∧ b^2 = 5 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l992_99294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_markers_count_l992_99240

theorem blue_markers_count (total : ℕ) (blue_percentage : ℚ) : 
  total = 96 → blue_percentage = 30 / 100 → 
  ∃ (blue : ℕ), blue = 29 ∧ (blue : ℚ) = blue_percentage * total :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_markers_count_l992_99240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l992_99289

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem function_properties (a : ℝ) :
  (f a 1 = 5) →
  (a = 4 ∧
   (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) ∧
   (∀ x₁ x₂ : ℝ, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l992_99289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscar_cd_packing_l992_99204

theorem oscar_cd_packing (rock classical pop : ℕ) 
  (h_rock : rock = 14) 
  (h_classical : classical = 12) 
  (h_pop : pop = 8) : 
  Nat.gcd rock (Nat.gcd classical pop) = 
  Finset.sup (Finset.filter (fun d => d ∣ rock ∧ d ∣ classical ∧ d ∣ pop) (Finset.range (min rock (min classical pop) + 1))) id := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscar_cd_packing_l992_99204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_slope_line_properties_l992_99287

-- Define a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define a line with rational slope
structure RationalSlopeLine where
  a : ℤ
  b : ℤ
  c : ℤ
  hNonZero : a ≠ 0 ∨ b ≠ 0

-- Define a function to check if a point is on the line
def isOnLine (p : LatticePoint) (l : RationalSlopeLine) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to calculate the distance from a point to the line
noncomputable def distanceToLine (p : LatticePoint) (l : RationalSlopeLine) : ℝ :=
  (|l.a * p.x + l.b * p.y + l.c|) / Real.sqrt (l.a^2 + l.b^2)

theorem rational_slope_line_properties 
  (l : RationalSlopeLine) : 
  (∀ p : LatticePoint, ¬isOnLine p l) ∨ 
  (∃ d : ℝ, d > 0 ∧ 
    (∀ p : LatticePoint, 
      ¬isOnLine p l → distanceToLine p l ≥ d) ∧
    (∃ f : ℕ → LatticePoint, ∀ n : ℕ, isOnLine (f n) l)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_slope_line_properties_l992_99287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_fee_calculation_l992_99261

/-- Represents the gas fee calculation for a city -/
noncomputable def gas_fee (consumption : ℝ) : ℝ :=
  if consumption ≤ 60 then
    consumption * 0.8
  else
    60 * 0.8 + (consumption - 60) * 1.2

/-- Theorem: If the average gas cost is 0.88 yuan/m³ and consumption > 60m³,
    then the consumption is 75m³ and the total fee is 66 yuan -/
theorem gas_fee_calculation (consumption : ℝ) 
    (h1 : consumption > 60) 
    (h2 : gas_fee consumption / consumption = 0.88) : 
    consumption = 75 ∧ gas_fee consumption = 66 := by
  sorry

#check gas_fee_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_fee_calculation_l992_99261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l992_99211

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := -1 / ((2 * x - 1)^2)

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l992_99211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_maps_R_to_R_l992_99290

/-- The region R in the complex plane -/
def R : Set ℂ := {z : ℂ | Complex.abs z.re ≤ 2 ∧ Complex.abs z.im ≤ 2}

/-- The transformation applied to z -/
noncomputable def T (z : ℂ) : ℂ := (2/3 + 2/3*Complex.I) * z

/-- Theorem stating that the transformation T maps R to itself -/
theorem T_maps_R_to_R : ∀ z ∈ R, T z ∈ R := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_maps_R_to_R_l992_99290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_properties_l992_99212

/-- Represents an electrical circuit with two capacitors and a resistor. -/
structure Circuit where
  C : ℝ  -- Capacitance of the first capacitor
  R : ℝ  -- Resistance of the resistor
  U₀ : ℝ  -- Initial voltage on the first capacitor

variable (circuit : Circuit)

/-- The initial current in the circuit. -/
noncomputable def initial_current (circuit : Circuit) : ℝ := circuit.U₀ / circuit.R

/-- The steady-state voltage on the capacitor with capacitance C. -/
noncomputable def steady_state_voltage (circuit : Circuit) : ℝ := circuit.U₀ / 5

/-- The amount of heat generated in the circuit after connection. -/
noncomputable def heat_generated (circuit : Circuit) : ℝ := (2 / 5) * circuit.C * circuit.U₀^2

/-- Theorem stating the properties of the circuit. -/
theorem circuit_properties (circuit : Circuit) :
  initial_current circuit = circuit.U₀ / circuit.R ∧
  steady_state_voltage circuit = circuit.U₀ / 5 ∧
  heat_generated circuit = (2 / 5) * circuit.C * circuit.U₀^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_properties_l992_99212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_size_l992_99282

/-- Represents the number of men needed to asphalt a road given specific conditions -/
noncomputable def men_needed (road_length : ℝ) (days : ℝ) (hours_per_day : ℝ) : ℝ :=
  (road_length * 2880) / (days * hours_per_day)

/-- Theorem stating that 20 men are needed for the second group -/
theorem second_group_size :
  men_needed 2 19.2 15 = 20 :=
by
  -- The proof goes here
  sorry

/-- Given conditions as lemmas -/
lemma first_group_condition :
  men_needed 1 12 8 = 30 :=
by
  -- The proof goes here
  sorry

#check second_group_size
#check first_group_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_size_l992_99282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l992_99210

noncomputable def g (A : ℝ) : ℝ :=
  (Real.cos A * (2 * Real.sin A ^ 2 + Real.sin A ^ 4 + 2 * Real.cos A ^ 2 + Real.cos A ^ 2 * Real.sin A ^ 2)) /
  ((Real.cos A / Real.sin A) * (1 / Real.sin A - Real.cos A * (Real.cos A / Real.sin A)))

theorem g_range (A : ℝ) (h : ∀ n : ℤ, A ≠ n * Real.pi) :
  2 ≤ g A ∧ g A ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l992_99210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_differential_equation_l992_99273

/-- The function F that is a solution to the differential equation -/
noncomputable def F (x y : ℝ) : ℝ := x^4 / 4 + x * y - y^2 / 2

/-- The differential form from the original equation -/
noncomputable def ω (x y : ℝ) : (ℝ × ℝ) → ℝ := λ (dx, dy) ↦ x^3 * dx + y * dx + x * dy - y * dy

theorem solution_to_differential_equation :
  ∀ (x y : ℝ), ∃ (C : ℝ),
    (∀ (dx dy : ℝ),
      (deriv (λ x' ↦ F x' y) x) * dx + (deriv (λ y' ↦ F x y') y) * dy = ω x y (dx, dy)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_differential_equation_l992_99273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_l992_99292

/-- The curve C -/
noncomputable def C (x y : ℝ) : Prop := Real.sqrt (x^2 / 25) + Real.sqrt (y^2 / 9) = 1

/-- Point F₁ -/
def F₁ : ℝ × ℝ := (-4, 0)

/-- Point F₂ -/
def F₂ : ℝ × ℝ := (4, 0)

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem curve_property :
  ∀ (x y : ℝ), C x y → distance (x, y) F₁ + distance (x, y) F₂ ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_property_l992_99292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_251_pattern_l992_99221

def has_251_pattern (m n : ℕ) : Prop := 
  ∃ k : ℕ, (1000 * m) / n - (100 * k) = 251

theorem smallest_n_with_251_pattern : 
  (∃ m : ℕ, m < 127 ∧ Nat.Coprime m 127 ∧ has_251_pattern m 127) ∧ 
  (∀ n : ℕ, n < 127 → ¬∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ has_251_pattern m n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_251_pattern_l992_99221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_two_neither_necessary_nor_sufficient_l992_99280

def z (a : ℝ) : ℂ := (a - 4) + (a + 2) * Complex.I

theorem a_eq_two_neither_necessary_nor_sufficient :
  ¬(∀ a : ℝ, z a = Complex.I * (z a).im) ∧
  ¬(∀ a : ℝ, a = 2 → z a = Complex.I * (z a).im) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_two_neither_necessary_nor_sufficient_l992_99280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_two_injective_injective_neq_implies_fn_neq_monotone_implies_injective_l992_99258

-- Definition of injective function
def Injective {α β : Type*} (f : α → β) : Prop :=
  ∀ x y, f x = f y → x = y

-- Statement 1
theorem exp_two_injective : Injective (fun x : ℝ => Real.exp (x * Real.log 2)) := by sorry

-- Statement 2
theorem injective_neq_implies_fn_neq {α β : Type*} (f : α → β) (hf : Injective f) :
  ∀ x y, x ≠ y → f x ≠ f y := by sorry

-- Statement 3
theorem monotone_implies_injective {α : Type*} [LinearOrder α] {β : Type*} [Preorder β] 
  (f : α → β) (hf : Monotone f) : Injective f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_two_injective_injective_neq_implies_fn_neq_monotone_implies_injective_l992_99258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_plan_interest_rate_l992_99209

/-- Calculates the interest rate for a payment plan, rounded to the nearest tenth of a percent. -/
def calculate_interest_rate (purchase_price down_payment monthly_payment num_payments : ℚ) : ℚ :=
  let total_paid := down_payment + monthly_payment * num_payments
  let interest_amount := total_paid - purchase_price
  let interest_rate := (interest_amount / purchase_price) * 100
  (interest_rate * 10).floor / 10

/-- Theorem stating that the given payment plan results in a 15.7% interest rate. -/
theorem payment_plan_interest_rate :
  calculate_interest_rate 127 27 10 12 = 157 / 10 := by
  sorry

#eval calculate_interest_rate 127 27 10 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_plan_interest_rate_l992_99209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_sum_l992_99253

/-- 
Given a quadratic function f(x) = 3x^2 - 2x + 8, when shifted 6 units to the right,
it becomes a new quadratic function g(x) = ax^2 + bx + c.
This theorem states that the sum of the coefficients a, b, and c equals 93.
-/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 3 * (x - 6)^2 - 2 * (x - 6) + 8 = a * x^2 + b * x + c) → 
  a + b + c = 93 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_shift_sum_l992_99253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_expansion_l992_99271

noncomputable def f (x : ℝ) : ℝ := (x - 4)^2

noncomputable def fourierCoefficient (m : ℕ) : ℝ :=
  if m = 0 then 32 / 3 else 64 / (m^2 * Real.pi^2)

noncomputable def fourierSeries (x : ℝ) : ℝ :=
  fourierCoefficient 0 / 2 + ∑' m, fourierCoefficient m * Real.cos (m * Real.pi * x / 4)

theorem fourier_series_expansion :
  ∀ x ∈ Set.Icc 0 4, f x = fourierSeries x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_expansion_l992_99271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_l992_99222

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 ≤ 3}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Define the expected intersection
def expected_intersection : Set ℝ := Set.Iic (-2) ∪ {2}

-- Theorem statement
theorem intersection_equality : A ∩ B = expected_intersection := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equality_l992_99222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l992_99248

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x + Real.cos x > 1) ↔ (∃ x : ℝ, Real.sin x + Real.cos x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l992_99248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_inequality_holds_l992_99234

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 1 - a * Real.log x

-- Theorem for part (I)
theorem one_common_point (a : ℝ) :
  (∃! x, f a x = 0 ∧ x > 0) ↔ (a ≤ 0 ∨ a = 2) := by
  sorry

-- Theorem for part (II)
theorem inequality_holds (a : ℝ) :
  (∀ x ≥ 1, f a x ≤ Real.exp (x - 1) + x^2 - x - 1) ↔ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_inequality_holds_l992_99234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_study_time_approx_l992_99245

/-- Calculates the number of minutes Mia spends studying each day -/
noncomputable def mia_study_time : ℝ :=
  let total_hours : ℝ := 24
  let sleep_hours : ℝ := 7
  let waking_hours : ℝ := total_hours - sleep_hours
  let waking_minutes : ℝ := waking_hours * 60
  let tv_time : ℝ := (1 / 5) * waking_minutes
  let exercise_time : ℝ := (1 / 8) * waking_minutes
  let remaining_after_tv_exercise : ℝ := waking_minutes - (tv_time + exercise_time)
  let social_media_time : ℝ := (1 / 6) * remaining_after_tv_exercise
  let remaining_after_social : ℝ := remaining_after_tv_exercise - social_media_time
  let chores_time : ℝ := (1 / 3) * remaining_after_social
  let remaining_after_chores : ℝ := remaining_after_social - chores_time
  (1 / 4) * remaining_after_chores

theorem mia_study_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |mia_study_time - 96| < ε := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval mia_study_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_study_time_approx_l992_99245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_bound_l992_99293

/-- Represents a country with cities and postal routes -/
structure Country where
  cities : Finset Nat
  routes : Nat → Nat → Nat
  connected : ∀ i j, i ∈ cities → j ∈ cities → ∃ (path : List Nat), path.head? = some i ∧ path.getLast? = some j

/-- The number of stamps required for all mayors to send letters to each other -/
def totalStamps (c : Country) : Nat :=
  c.cities.sum (λ i => c.cities.sum (λ j => if i ≠ j then c.routes i j else 0))

/-- Theorem stating the maximum number of stamps required -/
theorem max_stamps_bound (c : Country) (h : c.cities.card = 9) : totalStamps c ≤ 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_stamps_bound_l992_99293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_population_west_percentage_l992_99237

/-- Represents the population of Asians in millions for each region of the U.S. -/
structure AsianPopulation where
  NE : ℕ
  MW : ℕ
  South : ℕ
  West : ℕ

/-- Calculates the percentage of Asians living in the West, rounded to the nearest percent -/
def percentAsianInWest (pop : AsianPopulation) : ℕ :=
  let totalAsian := pop.NE + pop.MW + pop.South + pop.West
  let westPercentage := (pop.West : ℚ) / (totalAsian : ℚ) * 100
  (westPercentage + 1/2).floor.toNat

/-- The given population data from the 1980 U.S. census -/
def census1980 : AsianPopulation := ⟨2, 2, 2, 5⟩

theorem asian_population_west_percentage :
  percentAsianInWest census1980 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_population_west_percentage_l992_99237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_triangle_area_l992_99241

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the hyperbola
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  x₁ = -x₂ ∧ y₁ = y₂ ∧ x₁^2 - y₁^2 = 5

-- Define a right angle
def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  abs ((x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) / 2)

-- Theorem statement
theorem hyperbola_right_angle_triangle_area 
  (F₁ F₂ P : ℝ × ℝ) 
  (h₁ : are_foci F₁ F₂) 
  (h₂ : is_on_hyperbola P.1 P.2) 
  (h₃ : is_right_angle F₁ P F₂) : 
  triangle_area F₁ P F₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_triangle_area_l992_99241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l992_99270

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (1 ∉ A a) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l992_99270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_rectangle_l992_99208

theorem six_points_in_rectangle (points : Finset (ℝ × ℝ)) : 
  (points.card = 6) →
  (∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3) →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_rectangle_l992_99208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l992_99288

/-- The x-coordinate of the right focus of the ellipse x²/5 + y² = 1 -/
noncomputable def right_focus_x : ℝ := 2

/-- The x-coordinate of the focus of the parabola y² = 2mx -/
noncomputable def parabola_focus_x (m : ℝ) : ℝ := m / 2

/-- The statement that the focus of the parabola y² = 2mx coincides with 
    the right focus of the ellipse x²/5 + y² = 1 implies m = 4 -/
theorem parabola_ellipse_focus_coincide (m : ℝ) : 
  parabola_focus_x m = right_focus_x → m = 4 := by
  intro h
  have : m / 2 = 2 := h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l992_99288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_sevenths_three_fourths_times_two_l992_99266

/-- The remainder function for real numbers -/
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

/-- Proof that rem(5/7, 3/4) * 2 = 10/7 -/
theorem rem_five_sevenths_three_fourths_times_two :
  rem (5/7 : ℝ) (3/4 : ℝ) * 2 = 10/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_sevenths_three_fourths_times_two_l992_99266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_digit_inequality_l992_99231

theorem single_digit_inequality (A : ℕ) : 
  (A < 10) →
  (3.4 < (3 : ℝ) + A / 10 ∧ (3 : ℝ) + A / 10 < 4) →
  (4/3 > (4 : ℝ)/A ∧ (4 : ℝ)/A > 4/6) →
  A = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_digit_inequality_l992_99231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_l992_99255

theorem sum_of_multiples (m n : ℕ) 
  (hm : m > 0)
  (hn : n > 0)
  (h1 : n * (m * (m + 1) / 2) = 120)
  (h2 : n^3 * (m^3 * (m^3 + 1) / 2) = 4032000) :
  n^2 * (m^2 * (m^2 + 1) / 2) = 20800 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_multiples_l992_99255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_16_l992_99286

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 18

theorem g_composition_equals_16 : g (g (g 20)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_16_l992_99286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_15_and_3a_minus_b_plus_c_l992_99275

theorem sqrt_15_and_3a_minus_b_plus_c (a b c : ℝ) : 
  (5 * a + 2)^(1/3) = 3 →
  (3 * a + b - 1)^(1/2) = 4 →
  c = ⌊Real.sqrt 15⌋ →
  (Real.sqrt 15 - ⌊Real.sqrt 15⌋ = Real.sqrt 15 - 3) ∧ 
  ((3 * a - b + c)^(1/2) = 4 ∨ (3 * a - b + c)^(1/2) = -4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_15_and_3a_minus_b_plus_c_l992_99275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_is_circle_l992_99223

/-- The equation of the cookie: x^2 + y^2 + 16 = 6x + 14y -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 16 = 6*x + 14*y

/-- The radius of the cookie -/
noncomputable def cookie_radius : ℝ := Real.sqrt 42

/-- Theorem stating that the cookie equation represents a circle -/
theorem cookie_is_circle :
  ∃ (h k : ℝ), ∀ (x y : ℝ), cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 := by
  -- Proof goes here
  sorry

#check cookie_is_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_is_circle_l992_99223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l992_99297

/-- Represents the production rates and order quantities for an assembly line --/
structure AssemblyLine where
  rate1 : ℚ  -- Initial production rate in cogs per hour
  order1 : ℚ  -- First order quantity in cogs
  rate2 : ℚ  -- Second production rate in cogs per hour
  order2 : ℚ  -- Second order quantity in cogs
  rate3 : ℚ  -- Third production rate in cogs per hour
  order3 : ℚ  -- Third order quantity in cogs

/-- Calculates the overall average output of the assembly line --/
def overallAverageOutput (a : AssemblyLine) : ℚ :=
  (a.order1 + a.order2 + a.order3) / 
  (a.order1 / a.rate1 + a.order2 / a.rate2 + a.order3 / a.rate3)

/-- Theorem stating that for the given production rates and order quantities,
    the overall average output is 1620/35 cogs per hour --/
theorem assembly_line_output : 
  let a := AssemblyLine.mk 36 60 60 90 45 120
  overallAverageOutput a = 1620 / 35 := by
  sorry

#eval overallAverageOutput (AssemblyLine.mk 36 60 60 90 45 120)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l992_99297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_satisfy_equations_l992_99250

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def x_k (k : ℕ) : ℝ := -5 + 9/8 * (k - 1)

theorem solutions_satisfy_equations (k : ℕ) (h1 : k ≥ 1) (h2 : k ≤ 8) :
  (floor (x_k k) = 8/9 * (x_k k - 4) + 3) ∧
  (x_k k - floor (x_k k) = 1/9 * (x_k k + 5)) := by
  sorry

#check solutions_satisfy_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_satisfy_equations_l992_99250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sin_double_l992_99215

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := Real.tan (x / 2)

noncomputable def f (y : ℝ) : ℝ := 
  4 * y * (1 - y^2) / (1 + y^2)^2

-- State the theorem
theorem tan_half_sin_double (k : ℝ) : 
  (∀ x, 0 < x → x < Real.pi → f (g x) = Real.sin (2 * x)) →
  k * f (Real.sqrt 2 / 2) = 36 * Real.sqrt 2 ↔ k = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sin_double_l992_99215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weighted_average_tax_rate_l992_99257

/-- Calculate the combined weighted average tax rate for John and Ingrid --/
theorem combined_weighted_average_tax_rate : 
  ∃ (combined_weighted_average_tax_rate : ℝ),
  let john_job_income : ℝ := 57000;
  let john_job_tax_rate : ℝ := 0.30;
  let john_rental_income : ℝ := 11000;
  let john_rental_tax_rate : ℝ := 0.25;
  let ingrid_employment_income : ℝ := 72000;
  let ingrid_employment_tax_rate : ℝ := 0.40;
  let ingrid_investment_income : ℝ := 4500;
  let ingrid_investment_tax_rate : ℝ := 0.15;

  let john_total_tax : ℝ := john_job_income * john_job_tax_rate + john_rental_income * john_rental_tax_rate;
  let ingrid_total_tax : ℝ := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate;
  let combined_total_tax : ℝ := john_total_tax + ingrid_total_tax;

  let john_total_income : ℝ := john_job_income + john_rental_income;
  let ingrid_total_income : ℝ := ingrid_employment_income + ingrid_investment_income;
  let combined_total_income : ℝ := john_total_income + ingrid_total_income;

  combined_weighted_average_tax_rate = combined_total_tax / combined_total_income * 100 ∧
  abs (combined_weighted_average_tax_rate - 34.14) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_weighted_average_tax_rate_l992_99257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weak_multiple_exists_l992_99229

/-- Two positive integers are coprime if their greatest common divisor is 1 -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A positive integer is weak if it cannot be written as ax + by for non-negative integers x and y -/
def Weak (n a b : ℕ) : Prop := ∀ x y : ℕ, n ≠ a * x + b * y

/-- Main theorem statement -/
theorem weak_multiple_exists (a b n : ℕ) (h_coprime : Coprime a b) (h_weak : Weak n a b) 
    (h_bound : n < a * b / 6) : ∃ k : ℕ, k ≥ 2 ∧ Weak (k * n) a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weak_multiple_exists_l992_99229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l992_99279

-- Define the vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x/4), 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x/4), Real.cos (x/4) ^ 2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Part 1
theorem part_one (x : ℝ) (h : f x = 1) :
  Real.cos (2 * Real.pi / 3 - x) = 1 / 2 := by sorry

-- Part 2
theorem part_two (A B C : ℝ) (a b c : ℝ) 
  (h : a * Real.cos C + c / 2 = b) :
  1 < f B ∧ f B < 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l992_99279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_to_square_l992_99247

/-- A cross made of 5 unit squares -/
structure Cross where
  center : Set (ℝ × ℝ)
  arms : Fin 4 → Set (ℝ × ℝ)

/-- A dissection of the cross into 5 parts -/
structure Dissection (c : Cross) where
  parts : Fin 5 → Set (ℝ × ℝ)
  cover : (⋃ i, parts i) = c.center ∪ (⋃ i, c.arms i)
  disjoint : ∀ i j, i ≠ j → parts i ∩ parts j = ∅

/-- A 2x2 square -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- Theorem: A cross can be dissected and rearranged into a 2x2 square -/
theorem cross_to_square :
  ∃ (c : Cross) (d : Dissection c) (f : Fin 5 → ℝ × ℝ → ℝ × ℝ),
    (∀ i, f i '' d.parts i ⊆ Square) ∧
    (⋃ i, f i '' d.parts i) = Square :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_to_square_l992_99247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_S8_S9_l992_99268

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Define the sum of the first n terms
noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := 
  n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem min_sum_S8_S9 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h1 : a₁ < 0) 
  (h2 : d > 0) 
  (h3 : S a₁ d 6 = S a₁ d 11) :
  ∀ n : ℕ, n ≥ 1 → S a₁ d 8 ≤ S a₁ d n ∧ S a₁ d 9 ≤ S a₁ d n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_S8_S9_l992_99268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l992_99203

def a (n : ℕ+) (lambda : ℝ) : ℝ := n^2 - 2*lambda*n + 1

theorem sequence_increasing_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ+, a n lambda < a (n + 1) lambda) → lambda < (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l992_99203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_pounds_is_three_l992_99260

/-- The number of pounds of raisins mixed with nuts -/
def raisin_pounds : ℝ := sorry

/-- The cost of a pound of raisins -/
def raisin_cost : ℝ := sorry

/-- The cost of a pound of nuts -/
def nut_cost : ℝ := sorry

/-- The relationship between nut cost and raisin cost -/
axiom nut_cost_relation : nut_cost = 4 * raisin_cost

/-- The total cost ratio of raisins to the mixture -/
axiom raisin_cost_ratio : raisin_pounds * raisin_cost = 
  0.15789473684210525 * (raisin_pounds * raisin_cost + 4 * nut_cost)

/-- The theorem stating that the number of pounds of raisins is 3 -/
theorem raisin_pounds_is_three : raisin_pounds = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_pounds_is_three_l992_99260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l992_99224

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : (- t.b + Real.sqrt 2 * t.c) / Real.cos t.B = t.a / Real.cos t.A) :
  t.A = π / 4 ∧ 
  (t.a = 2 → ∃ (S : ℝ), S ≤ Real.sqrt 2 + 1 ∧ 
    ∀ (S' : ℝ), (∃ (b' c' : ℝ), S' = 1/2 * b' * c' * Real.sin t.A) → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l992_99224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l992_99296

-- Define the function f(x) = √(2x - 1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 1)

-- Theorem stating the domain of f
theorem f_domain : 
  ∀ x : ℝ, (2 * x - 1 ≥ 0) ↔ x ≥ (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l992_99296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l992_99244

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices --/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

/-- Theorem: The area of the triangle formed by the intersection of three specific lines is 8.625 --/
theorem triangle_area_is_8_625 
  (line1 line2 : Line)
  (intersection_point : Point)
  (h1 : line1.slope = 1/3)
  (h2 : line2.slope = 3)
  (h3 : intersection_point.x = 3 ∧ intersection_point.y = 3)
  (line3 : Set Point)
  (h4 : ∀ p, p ∈ line3 ↔ p.x + p.y = 12) :
  ∃ (p1 p2 : Point), 
    p1 ∈ line3 ∧ 
    p2 ∈ line3 ∧ 
    triangleArea intersection_point p1 p2 = 8.625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_625_l992_99244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_2x_l992_99235

theorem parallel_vectors_sin_2x (x : ℝ) 
  (a b : ℝ × ℝ)
  (ha : a = (Real.cos x, -2))
  (hb : b = (Real.sin x, 1))
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  Real.sin (2 * x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_2x_l992_99235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_minimum_l992_99249

noncomputable section

variable (a : ℝ)

def room_area : ℝ := 12
def front_cost : ℝ := 400
def side_cost : ℝ := 150
def roof_floor_cost : ℝ := 5800
def wall_height : ℝ := 3

def total_cost (x : ℝ) : ℝ := 900 * (x + 16 / x) + 5800

def is_valid_length (x : ℝ) : Prop := 0 < x ∧ x ≤ a

theorem total_cost_minimum (h : 0 < a) :
  (∀ x, is_valid_length a x → total_cost x ≥ 
    (if a ≥ 4 then 13000 else 900 * (a + 16 / a) + 5800)) ∧
  (if a ≥ 4 
   then is_valid_length a 4 ∧ total_cost 4 = 13000
   else total_cost a = 900 * (a + 16 / a) + 5800) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_minimum_l992_99249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_difference_l992_99230

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 1⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Statement of the theorem -/
theorem parabola_point_difference (A B : Point) 
  (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_diff : distance A focus - distance B focus = 2) :
  A.y + A.x^2 - B.y - B.x^2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_difference_l992_99230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_to_decimal_fraction_l992_99283

theorem octal_to_decimal_fraction (a b : ℕ) : 
  (3 * 8^2 + 7 * 8 + 4 = 10 * a + b) → 
  (a ≤ 9) → 
  (b ≤ 9) → 
  (a + b : ℝ) / 20 = 0.35 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_to_decimal_fraction_l992_99283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l992_99276

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem stating that the domain of f is [1,2) ∪ (2,+∞)
theorem domain_of_f : domain_f = Set.Icc 1 2 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l992_99276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reforestation_growth_rates_l992_99259

/-- The original area available for reforestation in acres -/
def original_area : ℚ := 637680

/-- The area converted to forest in 2000 in acres -/
def area_2000 : ℚ := 80000

/-- The total area reforested by the end of 2002 in acres -/
def total_area_2002 : ℚ := 291200

/-- The average annual growth rate for 2001 and 2002 -/
def avg_growth_rate : ℚ := 1/5

/-- The function relating the reforested area y (in ten thousand acres) to the growth rate x in 2003 -/
def reforested_area (x : ℚ) : ℚ := 1152/100 * x + 1152/100

/-- The minimum reforested area for 2003 in ten thousand acres -/
def min_area_2003 : ℚ := 144/10

/-- The maximum possible reforested area for 2003 in ten thousand acres -/
def max_area_2003 : ℚ := (original_area - total_area_2002) / 10000

/-- Theorem stating the correctness of the average annual growth rate and the range of growth rate for 2003 -/
theorem reforestation_growth_rates :
  (area_2000 + area_2000 * (1 + avg_growth_rate) + area_2000 * (1 + avg_growth_rate)^2 = total_area_2002) ∧
  (∀ x, min_area_2003 ≤ reforested_area x ∧ reforested_area x ≤ max_area_2003 ↔ 1/4 ≤ x ∧ x ≤ 2) :=
by
  sorry

#eval original_area
#eval area_2000
#eval total_area_2002
#eval avg_growth_rate
#eval reforested_area 1
#eval min_area_2003
#eval max_area_2003

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reforestation_growth_rates_l992_99259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l992_99201

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclinationAngle (a b : ℝ) : ℝ :=
  Real.pi - Real.arctan (a / b)

/-- The equation of the line: √2x + √6y + 1 = 0 -/
theorem line_inclination_angle :
  inclinationAngle (Real.sqrt 2) (Real.sqrt 6) = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l992_99201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_profit_at_two_critical_points_l992_99206

-- Define the profit function
noncomputable def g (x : ℝ) : ℝ := -x^3/2 + (9/2)*x^2 - 2

-- Define the derivative of the profit function
noncomputable def g' (x : ℝ) : ℝ := -3*x^2/2 + 9*x

-- State the theorem
theorem max_profit :
  ∃ (x : ℝ), x > 0 ∧ x ≤ 8 ∧
  (∀ y, y > 0 → y ≤ 8 → g y ≤ g x) ∧
  x = 6 ∧ g x = 52 := by
  sorry

-- Additional theorem to show that g(2) = 0.12
theorem profit_at_two :
  g 2 = 0.12 := by
  sorry

-- Theorem to show that g'(x) = 0 has two solutions
theorem critical_points :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g' x₁ = 0 ∧ g' x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_profit_at_two_critical_points_l992_99206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_progression_length_l992_99262

/-- The set of reciprocals of the first 2016 positive integers -/
def S : Set ℚ := {x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2016 ∧ x = 1 / n}

/-- A function to check if a list of rationals forms an arithmetic progression -/
def isArithmeticProgression (l : List ℚ) : Prop :=
  l.length > 1 ∧ ∀ i : ℕ, i + 2 < l.length → 
    l.get ⟨i + 2, by sorry⟩ - l.get ⟨i + 1, by sorry⟩ = 
    l.get ⟨i + 1, by sorry⟩ - l.get ⟨i, by sorry⟩

/-- The maximum length of an arithmetic progression in S -/
def maxAPLength : ℕ := 6

theorem max_arithmetic_progression_length :
  ∀ l : List ℚ, (∀ x ∈ l, x ∈ S) → isArithmeticProgression l → l.length ≤ maxAPLength :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_progression_length_l992_99262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l992_99232

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 9*x + 20) / (|x - 5| + |x + 2|)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≤ 4 ∨ x ≥ 5

-- Theorem statement
theorem f_domain : ∀ x : ℝ, x ∈ {x | f x ≠ 0} ↔ domain x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l992_99232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l992_99207

/-- Represents a sampling method -/
inductive SamplingMethod
| Stratified
| Lottery
| Random
| Systematic

/-- Represents a class of students -/
structure StudentClass where
  size : Nat
  numbers : Finset Nat

/-- Represents the entire population -/
structure Population where
  classes : Finset StudentClass
  selectedNumber : Nat

/-- Defines the conditions of the problem -/
def systematicSamplingConditions (pop : Population) : Prop :=
  ∀ c ∈ pop.classes, 
    c.size = 56 ∧ 
    c.numbers = Finset.range 56 ∧
    pop.selectedNumber = 16

/-- The theorem to be proved -/
theorem systematic_sampling 
  (pop : Population) 
  (h : systematicSamplingConditions pop) : 
  SamplingMethod.Systematic = 
    (if pop.classes.card > 1 ∧ 
        (∀ c ∈ pop.classes, c.size > 1) ∧ 
        (∀ c ∈ pop.classes, pop.selectedNumber ∈ c.numbers)
    then SamplingMethod.Systematic
    else SamplingMethod.Random) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_l992_99207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l992_99263

noncomputable def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def sample_mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length : ℝ)

noncomputable def significant_improvement (old_data new_data : List ℝ) : Prop :=
  let x_bar := sample_mean old_data
  let y_bar := sample_mean new_data
  let s1_sq := sample_variance old_data
  let s2_sq := sample_variance new_data
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_sq + s2_sq) / 10)

theorem new_device_improvement :
  significant_improvement old_data new_data := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l992_99263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_from_clay_l992_99278

/-- Represents the number of students from each school and gender --/
structure StudentDistribution where
  total : Nat
  boys : Nat
  girls : Nat
  jonas : Nat
  clay : Nat
  birch : Nat
  jonasBoys : Nat
  birchGirls : Nat

/-- The given student distribution --/
def givenDistribution : StudentDistribution :=
  { total := 180
  , boys := 94
  , girls := 86
  , jonas := 60
  , clay := 80
  , birch := 40
  , jonasBoys := 30
  , birchGirls := 24
  }

/-- Theorem stating that the number of boys from Clay Middle School is 48 --/
theorem boys_from_clay (d : StudentDistribution) : 
  d = givenDistribution → 
  d.boys - (d.jonasBoys + (d.birch - d.birchGirls)) = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_from_clay_l992_99278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l992_99269

theorem tan_pi_fourth_minus_alpha (α : Real) 
  (h1 : α > π) 
  (h2 : α < 3 * π / 2) 
  (h3 : Real.cos α = -4/5) : 
  Real.tan (π/4 - α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l992_99269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_boxes_theorem_l992_99217

/-- Proves that given two boxes with specific capacities and filling conditions, 
    the fraction of the first box filled with oranges is 3/4. -/
theorem orange_boxes_theorem :
  ∃ (box1_fraction : ℚ), 
    box1_fraction * 80 + 3/5 * 50 = 90 ∧
    box1_fraction = 3/4 := by
  -- We'll use 3/4 as our witness for box1_fraction
  use 3/4
  constructor
  · -- Prove that (3/4 * 80) + (3/5 * 50) = 90
    norm_num
  · -- Prove that 3/4 = 3/4
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_boxes_theorem_l992_99217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l992_99220

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * Complex.ofReal b = (11 - 7 * Complex.I) / (1 - 2 * Complex.I)) → 
  a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_sum_l992_99220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_mountain_road_l992_99246

theorem average_speed_mountain_road (road_length uphill_speed downhill_speed : ℝ) :
  road_length > 0 ∧ uphill_speed > 0 ∧ downhill_speed > 0 →
  (road_length = 400 ∧ uphill_speed = 50 ∧ downhill_speed = 80) →
  (2 * road_length) / (road_length / uphill_speed + road_length / downhill_speed) = 800 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_mountain_road_l992_99246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario3_faster_than_scenario2_l992_99236

/-- The speed of the pedestrian in km/h -/
noncomputable def pedestrian_speed : ℝ := 7

/-- The speed of the cyclist in km/h -/
noncomputable def cyclist_speed : ℝ := 15

/-- The distance traveled by the pedestrian in one hour -/
noncomputable def distance_traveled : ℝ := 7

/-- The distance between point B and point A -/
noncomputable def distance_BA : ℝ := 4 * Real.pi - 7

/-- The time taken in the third scenario -/
noncomputable def time_scenario3 : ℝ := (4 * Real.pi - 7) / (cyclist_speed + pedestrian_speed)

/-- The time taken in the second scenario -/
noncomputable def time_scenario2 : ℝ := (11 - 2 * Real.pi) / (cyclist_speed + pedestrian_speed)

theorem scenario3_faster_than_scenario2 : time_scenario3 < time_scenario2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario3_faster_than_scenario2_l992_99236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l992_99213

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  a > 0 →
  b > 0 →
  c > 0 →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  2 * Real.cos C * (a * Real.cos C + c * Real.cos A) + b = 0 →
  b = 2 →
  c = 2 * Real.sqrt 3 →
  C = 2 * Real.pi / 3 ∧ 
  (1/2) * a * b * Real.sin C = Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l992_99213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_height_l992_99251

/-- Given five points A, B, C, D, and O on a field, with A directly north of O,
    B directly west of O, C directly south of O, and D directly east of O.
    K is a point directly above O. The distance between C and D is 160 meters,
    KC has a length of 170 meters, and KD has a length of 150 meters. -/
structure KiteConfiguration where
  O : ℝ × ℝ × ℝ
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ
  K : ℝ × ℝ × ℝ
  h_A_north : A.2.1 = O.2.1 ∧ A.2.2 > O.2.2 ∧ A.1 = O.1
  h_B_west : B.2.2 = O.2.2 ∧ B.2.1 < O.2.1 ∧ B.1 = O.1
  h_C_south : C.2.1 = O.2.1 ∧ C.2.2 < O.2.2 ∧ C.1 = O.1
  h_D_east : D.2.2 = O.2.2 ∧ D.2.1 > O.2.1 ∧ D.1 = O.1
  h_K_above : K.2.1 = O.2.1 ∧ K.2.2 = O.2.2 ∧ K.1 > O.1
  h_CD_distance : (C.2.1 - D.2.1)^2 + (C.2.2 - D.2.2)^2 = 160^2
  h_KC_length : (K.2.1 - C.2.1)^2 + (K.2.2 - C.2.2)^2 + (K.1 - C.1)^2 = 170^2
  h_KD_length : (K.2.1 - D.2.1)^2 + (K.2.2 - D.2.2)^2 + (K.1 - D.1)^2 = 150^2

/-- The height of the kite (length of OK) is 30√43 meters. -/
theorem kite_height (config : KiteConfiguration) :
  (config.K.1 - config.O.1)^2 = (30 * Real.sqrt 43)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_height_l992_99251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l992_99214

/-- A track with straight sides and semicircular ends -/
structure Track where
  inner_radius : ℝ
  straight_length : ℝ
  width : ℝ

/-- The time difference between walking the outer and inner edges of the track -/
noncomputable def time_difference (track : Track) (speed : ℝ) : ℝ :=
  (2 * track.straight_length + 2 * Real.pi * (track.inner_radius + track.width)) / speed -
  (2 * track.straight_length + 2 * Real.pi * track.inner_radius) / speed

/-- The theorem stating the walker's speed given the track properties -/
theorem walker_speed (track : Track) (h1 : track.width = 10) (h2 : time_difference track (Real.pi / 3) = 60) :
  ∃ (speed : ℝ), time_difference track speed = 60 ∧ speed = Real.pi / 3 := by
  sorry

#check walker_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l992_99214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_arctan_over_x_power_l992_99227

/-- The limit of (arctan(3x)/x)^(x+2) as x approaches 0 is 9 -/
theorem limit_arctan_over_x_power : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → 
    |((Real.arctan (3*x)) / x)^(x+2) - 9| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_arctan_over_x_power_l992_99227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_a_4_value_of_a_for_inequality_l992_99272

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

-- Part 1: Theorem for the number of zeros when a = 4
theorem zeros_of_f_when_a_4 :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  f 4 x₁ = 0 ∧ f 4 x₂ = 0 ∧
  ∀ x : ℝ, x > 0 → f 4 x = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

-- Part 2: Theorem for the value of a
theorem value_of_a_for_inequality :
  (∀ x : ℝ, x > -1 → Real.exp x + a * Real.log (x + 1) ≥ 1) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_f_when_a_4_value_of_a_for_inequality_l992_99272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_properties_l992_99291

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents an ellipse -/
structure Ellipse where
  f1 : Point  -- First focus
  f2 : Point  -- Second focus
  sum_distances : ℝ  -- Sum of distances from any point on ellipse to both foci

/-- Theorem: For the given ellipse, h + k + a + b = 9 + √15 -/
theorem ellipse_sum_properties (e : Ellipse) 
    (h_f1 : e.f1 = ⟨0, 0⟩) 
    (h_f2 : e.f2 = ⟨6, 2⟩) 
    (h_sum : e.sum_distances = 10) : 
  ∃ (h k a b : ℝ), 
    h + k + a + b = 9 + Real.sqrt 15 ∧ 
    (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
      distance ⟨x, y⟩ e.f1 + distance ⟨x, y⟩ e.f2 = e.sum_distances) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_properties_l992_99291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l992_99238

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}

-- Define set B
def B : Set ℝ := {x | Real.sqrt (x - 1) < 1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l992_99238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_ten_l992_99219

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of a triangle --/
noncomputable def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The area of a triangle --/
noncomputable def area (t : Triangle) : ℝ := (1 / 2) * t.b * t.c * Real.sin t.A

theorem triangle_perimeter_is_ten (t : Triangle) 
  (h_acute : t.A < π / 2 ∧ t.B < π / 2 ∧ t.C < π / 2)
  (h_area : area t = Real.sqrt 15)
  (h_sine : t.c * Real.sin t.A = 2 * t.a * Real.sin t.B)
  (h_b : t.b = 2) :
  perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_ten_l992_99219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saturated_function_fixed_point_l992_99239

def Saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (f^[f^[f n] n]) n = n

theorem saturated_function_fixed_point (m : ℕ) :
  (∀ f : ℕ → ℕ, Saturated f → (f^[2014]) m = m) ↔ m ∣ 2014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saturated_function_fixed_point_l992_99239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l992_99218

theorem negation_of_conditional (α : Real) :
  ¬(α = π/4 → Real.tan α = 1) ↔ (α ≠ π/4 → Real.tan α ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l992_99218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_excavation_theorem_l992_99267

/-- Represents the tunnel excavation project -/
structure TunnelProject where
  totalLength : ℚ
  originalDays : ℕ
  increasedDays : ℕ
  efficiencyIncrease : ℚ
  completionRatio : ℚ
  remainingDays : ℕ

/-- Calculates the original daily excavation plan -/
def originalDailyPlan (project : TunnelProject) : ℚ :=
  project.totalLength * project.completionRatio / 
    (project.originalDays + project.increasedDays * (1 + project.efficiencyIncrease))

/-- Calculates the minimum daily excavation for Team B -/
def teamBDailyMinimum (project : TunnelProject) (originalPlan : ℚ) : ℚ :=
  (project.totalLength * (1 - project.completionRatio) - 
   project.remainingDays * originalPlan * (1 + project.efficiencyIncrease)) / 
  project.remainingDays

theorem tunnel_excavation_theorem (project : TunnelProject) 
  (h_total : project.totalLength = 2400)
  (h_original : project.originalDays = 6)
  (h_increased : project.increasedDays = 8)
  (h_efficiency : project.efficiencyIncrease = 1/4)
  (h_completion : project.completionRatio = 2/3)
  (h_remaining : project.remainingDays = 4) :
  originalDailyPlan project = 100 ∧ 
  teamBDailyMinimum project (originalDailyPlan project) = 75 := by
  sorry

#eval originalDailyPlan { totalLength := 2400, originalDays := 6, increasedDays := 8, 
                          efficiencyIncrease := 1/4, completionRatio := 2/3, remainingDays := 4 }
#eval teamBDailyMinimum { totalLength := 2400, originalDays := 6, increasedDays := 8, 
                          efficiencyIncrease := 1/4, completionRatio := 2/3, remainingDays := 4 } 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_excavation_theorem_l992_99267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_l992_99233

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  lines : Finset (Set (ℝ × ℝ))
  intersections : Finset (ℝ × ℝ)

/-- Predicate to check if a configuration is valid -/
def isValidConfiguration (config : LineConfiguration) : Prop :=
  config.lines.card = 5 ∧
  config.intersections.card = 7 ∧
  ∀ p ∈ config.intersections, ∃ l₁ l₂, l₁ ∈ config.lines ∧ l₂ ∈ config.lines ∧ l₁ ≠ l₂ ∧ p ∈ l₁ ∩ l₂

/-- Theorem stating the existence of a valid configuration -/
theorem exists_valid_configuration : ∃ config : LineConfiguration, isValidConfiguration config :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_l992_99233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_l992_99228

noncomputable section

/-- The perimeter of a circular sector with given radius and missing arc angle --/
def sectorPerimeter (radius : ℝ) (missingAngle : ℝ) : ℝ :=
  (2 - missingAngle / 360) * 2 * Real.pi * radius + 2 * radius

theorem monster_perimeter :
  sectorPerimeter 2 90 = 3 * Real.pi + 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_l992_99228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l992_99295

def number_list : List Nat := [14, 17, 19, 22, 26, 31]

def is_prime (n : Nat) : Bool := Nat.Prime n

def arithmetic_mean (l : List Nat) : Rat :=
  (l.sum : Rat) / l.length

theorem arithmetic_mean_of_primes :
  arithmetic_mean (number_list.filter is_prime) = 67 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l992_99295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l992_99242

noncomputable def root₁ (q : ℝ) : ℝ := (q + Real.sqrt (4*q - 3)) / 2
noncomputable def root₂ (q : ℝ) : ℝ := (q - Real.sqrt (4*q - 3)) / 2

theorem root_difference (q : ℝ) : 
  let r := max (root₁ q) (root₂ q)
  let s := min (root₁ q) (root₂ q)
  r - s = 2 * Real.sqrt (q - 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l992_99242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l992_99216

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  Real.sin (C / 2) = Real.sqrt 10 / 4 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 15) / 4 →
  Real.sin A ^ 2 + Real.sin B ^ 2 = (13 / 16) * Real.sin C ^ 2 →
  Real.cos C = -1 / 4 ∧
  ((a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 2 ∧ c = 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l992_99216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l992_99254

noncomputable def rhombus_side_length : ℝ := 2
noncomputable def rhombus_angle : ℝ := 30 * Real.pi / 180
noncomputable def prism_height : ℝ := 1
noncomputable def cross_section_angle : ℝ := 60 * Real.pi / 180

theorem cross_section_area :
  let base_width := rhombus_side_length
  let cross_section_height := prism_height / Real.sin cross_section_angle
  base_width * cross_section_height = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l992_99254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l992_99277

theorem simplify_trig_expression :
  (Real.tan (60 * π / 180))^3 + (Real.tan ((90 - 60) * π / 180))^3 /
  (Real.tan (60 * π / 180) + Real.tan ((90 - 60) * π / 180)) = 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l992_99277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l992_99281

/-- Calculates the present value of a future sum given an interest rate and time period. -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- The problem statement -/
theorem investment_problem (ε : ℝ) (hε : ε > 0) : 
  ∃ (presentVal : ℝ), 
    |presentValue 500000 0.05 10 - presentVal| < ε ∧ 
    |presentVal - 306956.63| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l992_99281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_general_l992_99205

variable (x₁ x₂ x₃ x₄ x₅ : ℝ)

def equation1 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := x₂ + 2 * x₃ - 3 * x₄ = -1
def equation2 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := 2 * x₁ - x₂ + 3 * x₃ + 4 * x₅ = 5
def equation3 (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop := 2 * x₁ + 5 * x₃ - 3 * x₄ + 4 * x₅ = 4

noncomputable def generalSolution (C₁ C₂ C₃ : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ :=
  (2 - 5/2*C₁ + 3/2*C₂ - 2*C₃,
   -1 - 2*C₁ + 3*C₂,
   C₁,
   C₂,
   C₃)

theorem solution_is_general (C₁ C₂ C₃ : ℝ) :
  let (x₁, x₂, x₃, x₄, x₅) := generalSolution C₁ C₂ C₃
  equation1 x₁ x₂ x₃ x₄ x₅ ∧
  equation2 x₁ x₂ x₃ x₄ x₅ ∧
  equation3 x₁ x₂ x₃ x₄ x₅ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_general_l992_99205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l992_99200

/-- Given a hyperbola and a circle with specific properties, prove that the asymptotes of the hyperbola have equations 2x ± y = 0 -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (C : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) (F1 T P : ℝ × ℝ),
    C = λ x y => b^2 * x^2 / a^2 - a^2 * y^2 / b^2 = a^2 * b^2 ∧
    circle = λ x y => x^2 + y^2 = a^2 ∧
    F1 = (-Real.sqrt (a^2 + b^2), 0) ∧
    circle T.1 T.2 ∧
    C P.1 P.2 ∧
    P.1 > 0 ∧  -- P is on the right branch
    T = ((F1.1 + P.1) / 2, (F1.2 + P.2) / 2) →  -- T is midpoint of F1P
    (∀ x y, C x y → (2 * x = y ∨ 2 * x = -y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l992_99200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_iff_normal_vectors_parallel_parallel_planes_k_value_l992_99256

/-- Definition of a plane -/
structure Plane where
  normalVector : ℝ × ℝ × ℝ

/-- Definition of parallel planes -/
def Plane.IsParallelTo (α β : Plane) : Prop := 
  ∃ (c : ℝ), c ≠ 0 ∧ α.normalVector = c • β.normalVector

/-- Two planes are parallel if and only if their normal vectors are parallel -/
theorem planes_parallel_iff_normal_vectors_parallel {α β : Plane} :
  α.IsParallelTo β ↔ ∃ (c : ℝ), c ≠ 0 ∧ α.normalVector = c • β.normalVector := by
  sorry

theorem parallel_planes_k_value (α β : Plane) (k : ℝ) :
  α.normalVector = (1, 2, -2) →
  β.normalVector = (-2, -4, k) →
  α.IsParallelTo β →
  k = 4 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_iff_normal_vectors_parallel_parallel_planes_k_value_l992_99256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l992_99284

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle
def circleEq (c_x c_y r : ℝ) (x y : ℝ) : Prop :=
  (x - c_x)^2 + (y - c_y)^2 = r^2

-- State the theorem
theorem circle_equation :
  ∃ (c_x c_y r : ℝ),
    (∀ x y, line x y → circleEq c_x c_y r x y) ∧
    circleEq c_x c_y r 3 0 ∧
    circleEq c_x c_y r 0 1 ∧
    c_x = 1 ∧ c_y = -1 ∧ r^2 = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l992_99284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_n_l992_99265

/-- A function that checks if a quadratic polynomial ax^2 + bx + c
    can be factored into two linear factors with integer coefficients -/
def is_factorable (a b c : ℤ) : Prop :=
  ∃ p q r s : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (p * x + q) * (r * x + s)

/-- The theorem stating that 46 is the smallest value of n for which
    5x^2 + nx + 48 can be factored into two linear factors with integer coefficients -/
theorem smallest_factorable_n :
  (∀ n : ℤ, n < 46 → ¬is_factorable 5 n 48) ∧ is_factorable 5 46 48 := by
  sorry

#check smallest_factorable_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_n_l992_99265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_intersection_l992_99285

-- Define the parameterized equation of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := ((1/2) * t, Real.sqrt 2/2 + (Real.sqrt 3/2) * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi/4)

-- State the theorem
theorem line_and_curve_intersection :
  -- Slope angle of line l is 60°
  (∃ (angle : ℝ), angle = 60 * (Real.pi/180) ∧ 
    ∀ t, (line_l t).2 - Real.sqrt 2/2 = Real.tan angle * (line_l t).1) ∧
  -- Length of chord AB is √10/2
  (∃ A B : ℝ × ℝ, 
    (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B) ∧
    (∃ θ₁ θ₂ : ℝ, curve_C θ₁ = Real.sqrt ((A.1)^2 + (A.2)^2) ∧ 
                  curve_C θ₂ = Real.sqrt ((B.1)^2 + (B.2)^2)) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_intersection_l992_99285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_for_point_l992_99298

/-- Given an angle α whose terminal side passes through a point P(3a, 4a) where a ≠ 0,
    prove the trigonometric ratios. -/
theorem trig_ratios_for_point (α : ℝ) (a : ℝ) (ha : a ≠ 0) 
  (h : ∃ (t : ℝ), t > 0 ∧ Real.cos α = 3 * a * t ∧ Real.sin α = 4 * a * t) : 
  (abs (Real.sin α)) = 4/5 ∧ (abs (Real.cos α)) = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_for_point_l992_99298
