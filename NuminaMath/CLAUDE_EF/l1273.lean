import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_l1273_127374

noncomputable def A : ℝ × ℝ := (Real.sqrt 3, 1)

def unit_circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 1

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem max_vector_sum :
  ∀ B : ℝ × ℝ, unit_circle B →
    ∃ max_length : ℝ, max_length = 3 ∧
      ∀ C : ℝ × ℝ, unit_circle C →
        vector_length (A.1 + C.1, A.2 + C.2) ≤ max_length :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_sum_l1273_127374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_parabola_l1273_127394

/-- The set of points defined by x = y^2 -/
def parabola (x y : ℝ) : Prop := x = y^2

/-- The circle with radius 5 centered at (11, 1) -/
def circle_eq (x y : ℝ) : Prop := (x - 11)^2 + (y - 1)^2 = 25

/-- The resulting parabola equation -/
def result_parabola (x y : ℝ) : Prop := y = (1/2) * x^2 - (21/2) * x + 97/2

theorem intersection_points_on_parabola :
  ∀ x y : ℝ, parabola x y ∧ circle_eq x y → result_parabola x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_parabola_l1273_127394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_positive_sufficient_not_necessary_l1273_127369

theorem sin_cos_product_positive_sufficient_not_necessary :
  ∃ θ₁ θ₂ : Real,
    (Real.sin θ₁ * Real.cos θ₁ > 0 ∧ 0 < θ₁ ∧ θ₁ < Real.pi / 2) ∧
    (0 < θ₂ ∧ θ₂ < Real.pi / 2 ∧ ¬(Real.sin θ₂ * Real.cos θ₂ > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_positive_sufficient_not_necessary_l1273_127369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1273_127330

theorem expression_evaluation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |4 - |Real.sqrt 3 - 3| + 6 - 8.732| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1273_127330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_given_job_l1273_127375

/-- Represents the job completion problem --/
structure JobCompletion where
  totalDays : ℕ
  elapsedDays : ℕ
  initialWorkers : ℕ
  completedPortion : ℚ

/-- Calculates the minimum number of workers needed to complete the job on time --/
def minWorkersNeeded (job : JobCompletion) : ℕ := 
  let remainingDays := job.totalDays - job.elapsedDays
  let remainingPortion := 1 - job.completedPortion
  let dailyRatePerWorker := job.completedPortion / (job.initialWorkers : ℚ) / (job.elapsedDays : ℚ)
  let requiredDailyRate := remainingPortion / (remainingDays : ℚ)
  (requiredDailyRate / dailyRatePerWorker).ceil.toNat

/-- Theorem stating that for the given job conditions, the minimum number of workers needed is 5 --/
theorem min_workers_for_given_job :
  let job := JobCompletion.mk 40 10 10 (2/5)
  minWorkersNeeded job = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_given_job_l1273_127375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1273_127398

-- Define the function f using Real.log instead of a custom log10 function
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log 10

-- State the theorem
theorem f_properties :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (0 : ℝ) 1 → x₂ ∈ Set.Ioo (0 : ℝ) 1 → x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1273_127398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1273_127371

noncomputable def purchase_price : ℝ := 130
noncomputable def down_payment : ℝ := 30
noncomputable def monthly_payment : ℝ := 10
def num_payments : ℕ := 12

noncomputable def total_paid : ℝ := down_payment + monthly_payment * (num_payments : ℝ)
noncomputable def interest_paid : ℝ := total_paid - purchase_price
noncomputable def interest_percent : ℝ := (interest_paid / purchase_price) * 100

theorem interest_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |interest_percent - 15.4| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1273_127371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1273_127346

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

-- Define the property of being an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem statement
theorem odd_function_values :
  ∀ a b : ℝ, (is_odd (f a b)) → (a = -1/2 ∧ b = Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1273_127346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rulers_equation_original_rulers_count_l1273_127389

/-- The number of rulers originally in the drawer -/
def original_rulers : ℕ := sorry

/-- The number of rulers Tim added to the drawer -/
def added_rulers : ℕ := 14

/-- The total number of rulers after Tim added some -/
def total_rulers : ℕ := 25

/-- Theorem stating that the original number of rulers plus the added rulers equals the total rulers -/
theorem rulers_equation : original_rulers + added_rulers = total_rulers := by
  sorry

/-- Theorem proving that the original number of rulers was 11 -/
theorem original_rulers_count : original_rulers = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rulers_equation_original_rulers_count_l1273_127389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_property_l1273_127305

/-- A quadrilateral with side lengths satisfying a specific equation is either a parallelogram or has perpendicular diagonals -/
theorem quadrilateral_property (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2*m*n + 2*p*q) :
  (m = n ∧ p = q) ∨ (∃ (d1 d2 : ℝ × ℝ), d1.1 * d2.1 + d1.2 * d2.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_property_l1273_127305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_chord_length_k_value_l1273_127382

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersectionPoints (k : ℝ) : 
  ∃ (xA yA xB yB : ℝ), parabola xA yA ∧ line k xA yA ∧ parabola xB yB ∧ line k xB yB := by sorry

-- Define the chord length
noncomputable def chordLength (xA yA xB yB : ℝ) : ℝ := 
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

-- Theorem 1: OA ⊥ OB
theorem perpendicular_OA_OB (k : ℝ) : 
  ∃ (xA yA xB yB : ℝ), 
    parabola xA yA ∧ line k xA yA ∧ parabola xB yB ∧ line k xB yB →
    xA * xB + yA * yB = 0 := by sorry

-- Theorem 2: When chord length AB = √10, k = ±1/6
theorem chord_length_k_value : 
  ∃ (k xA yA xB yB : ℝ),
    parabola xA yA ∧ line k xA yA ∧ parabola xB yB ∧ line k xB yB ∧
    chordLength xA yA xB yB = Real.sqrt 10 →
    k = 1/6 ∨ k = -1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_OA_OB_chord_length_k_value_l1273_127382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_range_l1273_127329

noncomputable def f (x : ℝ) := 3 * Real.cos (2 * x)

theorem f_max_value_and_range :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ (∃ (k : ℤ), f (k * π) = M)) ∧
  (∀ (x : ℝ), f x = 3 → ∃ (k : ℤ), x = k * π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_range_l1273_127329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_of_squares_zero_purely_imaginary_number_max_abs_z_bijection_real_purely_imaginary_l1273_127327

-- Statement 1
theorem complex_sum_of_squares_zero :
  ∃ (z₁ z₂ : ℂ), z₁^2 + z₂^2 = 0 ∧ (z₁ ≠ 0 ∨ z₂ ≠ 0) := by sorry

-- Statement 2
theorem purely_imaginary_number (z : ℂ) (b : ℝ) (h : z = Complex.I * b) (h_nonzero : b ≠ 0) :
  z.re = 0 ∧ z.im ≠ 0 := by sorry

-- Statement 3
theorem max_abs_z (z : ℂ) (h : Complex.abs (z + (Complex.ofReal (Real.sqrt 3)) + Complex.I) = 1) :
  ∃ (z_max : ℂ), Complex.abs z_max = 3 ∧ 
    ∀ (w : ℂ), Complex.abs (w + (Complex.ofReal (Real.sqrt 3)) + Complex.I) = 1 → 
      Complex.abs w ≤ Complex.abs z_max := by sorry

-- Statement 4
theorem bijection_real_purely_imaginary :
  ∃ (f : ℝ → ℂ), Function.Bijective f ∧ ∀ (x : ℝ), (f x).re = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_of_squares_zero_purely_imaginary_number_max_abs_z_bijection_real_purely_imaginary_l1273_127327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_day_is_450_seconds_l1273_127349

/-- The number of seconds in a fraction of a day -/
def seconds_in_fraction_of_day (day_fraction : ℚ) : ℕ :=
  ((day_fraction * 24 * 60 * 60).num).toNat

/-- Theorem stating that 1/4 of 1/6 of 1/8 of a day is 450 seconds -/
theorem fraction_of_day_is_450_seconds : 
  seconds_in_fraction_of_day ((1 : ℚ) / 4 * (1 : ℚ) / 6 * (1 : ℚ) / 8) = 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_day_is_450_seconds_l1273_127349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcd_is_common_denominator_l1273_127347

/-- The least common denominator of the given rational expressions -/
noncomputable def lcd (a : ℝ) : ℝ := a^4 - 2*a^2 + 1

/-- The first rational expression -/
noncomputable def expr1 (a : ℝ) : ℝ := 1 / (a^2 - 2*a + 1)

/-- The second rational expression -/
noncomputable def expr2 (a : ℝ) : ℝ := 1 / (a^2 - 1)

/-- The third rational expression -/
noncomputable def expr3 (a : ℝ) : ℝ := 1 / (a^2 + 2*a + 1)

theorem lcd_is_common_denominator (a : ℝ) :
  (∃ k1 k2 k3 : ℝ, 
    expr1 a = k1 / lcd a ∧ 
    expr2 a = k2 / lcd a ∧ 
    expr3 a = k3 / lcd a) ∧
  (∀ d : ℝ, (∃ k1 k2 k3 : ℝ, 
    expr1 a = k1 / d ∧ 
    expr2 a = k2 / d ∧ 
    expr3 a = k3 / d) → 
  lcd a ∣ d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcd_is_common_denominator_l1273_127347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisibility_implies_constant_ratio_l1273_127336

theorem integer_divisibility_implies_constant_ratio 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ+) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃ ∧ b₁ ≠ b₂ ∧ b₁ ≠ b₃ ∧ b₂ ≠ b₃)
  (h_divides : ∀ n : ℕ+, 
    ((n + 1) * a₁^(n:ℕ) + n * a₂^(n:ℕ) + (n - 1) * a₃^(n:ℕ)) ∣ 
    ((n + 1) * b₁^(n:ℕ) + n * b₂^(n:ℕ) + (n - 1) * b₃^(n:ℕ))) :
  ∃ k : ℕ+, b₁ = k * a₁ ∧ b₂ = k * a₂ ∧ b₃ = k * a₃ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_divisibility_implies_constant_ratio_l1273_127336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_representation_l1273_127366

/-- Given a triangle ABC, prove that the intersection point Q of lines BG and AF
    can be represented as a specific linear combination of A, B, and C. -/
theorem intersection_point_representation (A B C F G Q : ℝ × ℝ) : 
  -- Triangle ABC exists (implicitly assumed by using A, B, C as points)
  -- F lies on BC extended past C (represented by the ratio condition)
  (∃ t : ℝ, t > 1 ∧ F = t • C + (1 - t) • B) →
  -- BF:FC = 2:1
  (F = (2/3) • C + (1/3) • B) →
  -- G lies on AC (represented by the ratio condition)
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ G = s • A + (1 - s) • C) →
  -- AG:GC = 3:2
  (G = (2/5) • A + (3/5) • C) →
  -- Q is on BG
  (∃ u : ℝ, Q = u • B + (1 - u) • G) →
  -- Q is on AF
  (∃ v : ℝ, Q = v • A + (1 - v) • F) →
  -- Conclusion: Q can be represented as (2/15)A + (2/15)B + (11/15)C
  Q = (2/15) • A + (2/15) • B + (11/15) • C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_representation_l1273_127366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_munificence_of_p_l1273_127381

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + x^2 - 3*x + 1

-- Define the munificence of a function on an interval
noncomputable def munificence (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : a ≤ x ∧ x ≤ b), |f x|

-- Theorem statement
theorem munificence_of_p :
  munificence p (-1) 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_munificence_of_p_l1273_127381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_retirement_age_l1273_127372

/-- Calculate Jason's retirement age based on his military career progression --/
theorem jason_retirement_age :
  let join_age : ℝ := 18
  let chief_time : ℝ := 8
  let senior_chief_time : ℝ := chief_time * 1.255
  let master_chief_time : ℝ := senior_chief_time * 0.875
  let command_master_chief_time : ℝ := master_chief_time * 1.475
  let last_rank_time : ℝ := 2.5
  let total_service_time : ℝ := chief_time + senior_chief_time + master_chief_time + command_master_chief_time + last_rank_time
  let retirement_age : ℝ := join_age + total_service_time
  Int.floor retirement_age = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_retirement_age_l1273_127372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l1273_127323

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)
noncomputable def C2 (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the fixed point M
def M : ℝ × ℝ := (2, 0)

-- Define the intersection angle
noncomputable def θ_intersect : ℝ := Real.pi / 3

-- Define points A and B
noncomputable def A : ℝ × ℝ := C1 θ_intersect
noncomputable def B : ℝ × ℝ := (C2 θ_intersect * Real.cos θ_intersect, C2 θ_intersect * Real.sin θ_intersect)

-- State the theorem
theorem area_of_triangle_MAB :
  let d := Real.sqrt 3  -- distance from M to the ray
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)  -- distance between A and B
  (1/2) * AB * d = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MAB_l1273_127323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_alpha_l1273_127385

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem range_of_f_and_alpha (α : ℝ) :
  (∀ x, -Real.pi/6 ≤ x ∧ x ≤ Real.pi/3 → -1/2 ≤ f x ∧ f x ≤ 1) ∧
  ((∀ x, -Real.pi/6 ≤ x ∧ x ≤ α → -1/2 ≤ f x ∧ f x ≤ 1) →
   Real.pi/6 ≤ α ∧ α ≤ Real.pi/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_and_alpha_l1273_127385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l1273_127360

-- Define the complex number z
noncomputable def z : ℂ := (1 - Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem modulus_of_z_is_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l1273_127360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_l1273_127350

/-- Energy stored between two charges -/
noncomputable def energy (distance : ℝ) (charge1 : ℝ) (charge2 : ℝ) : ℝ :=
  (charge1 * charge2) / distance

/-- Configuration of charges -/
structure ChargeConfig where
  sideLength : ℝ
  charge : ℝ

/-- Total energy in initial square configuration -/
noncomputable def initialEnergy (config : ChargeConfig) : ℝ := 20

/-- Total energy when one charge is moved to center -/
noncomputable def centerEnergy (config : ChargeConfig) : ℝ :=
  initialEnergy config + 40

/-- Theorem statement -/
theorem energy_increase (config : ChargeConfig) (h1 : config.sideLength > 0) (h2 : config.charge > 0) :
  centerEnergy config - initialEnergy config = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_l1273_127350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_3_l1273_127308

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of interest
def point : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at the point
noncomputable def tangent_slope : ℝ := 3 * point.1^2

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := tangent_slope * (x - point.1) + point.2

-- Define the x-coordinate of the intersection of the tangent line with the x-axis
noncomputable def x_intercept : ℝ := point.1 - point.2 / tangent_slope

-- Define the area of the triangle
noncomputable def triangle_area : ℝ := (1/2) * (2 - x_intercept) * tangent_line 2

-- Theorem statement
theorem triangle_area_is_8_3 : triangle_area = 8/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_8_3_l1273_127308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_steps_l1273_127367

/-- The number of steps Piglet counted going down the moving escalator -/
def steps_down : ℕ := 66

/-- The number of steps Piglet counted going up the moving escalator -/
def steps_up : ℕ := 198

/-- Piglet's speed in steps per unit time -/
noncomputable def u : ℝ := sorry

/-- The speed of the escalator in steps per unit time -/
noncomputable def v : ℝ := sorry

/-- The length of the escalator in steps -/
def L : ℕ := sorry

/-- Theorem stating that the number of steps on the escalator is 99 -/
theorem escalator_steps :
  (L : ℝ) * u / (u + v) = steps_down ∧
  (L : ℝ) * u / (u - v) = steps_up →
  L = 99 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_steps_l1273_127367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_age_proof_l1273_127343

theorem oldest_age_proof (ages : List Nat) : 
  ages.length = 25 ∧ 
  (∀ i j, i < j → i < ages.length → j < ages.length → ages[i]! < ages[j]!) ∧
  (∀ i, i + 1 < ages.length → ages[i + 1]! = ages[i]! + 1) ∧
  (ages.sum + 50 = 2000) →
  ages.maximum? = some 88 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oldest_age_proof_l1273_127343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_power_minus_nine_l1273_127322

/-- The sum of the digits of 10^1001 - 9 is 9001 -/
theorem sum_digits_power_minus_nine : 
  (Nat.digits 10 (10^1001 - 9)).sum = 9001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_power_minus_nine_l1273_127322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neighboring_root_equation_conditions_part1_not_neighboring_root_part2_find_k_l1273_127391

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ 
  b^2 - 4*a*c ≥ 0 ∧
  ((-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)) - ((-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) = 1

/-- Theorem: Conditions for a quadratic equation to be a neighboring root equation -/
theorem neighboring_root_equation_conditions (a b c : ℝ) :
  is_neighboring_root_equation a b c ↔ 
  a ≠ 0 ∧ b^2 - 4*a*c ≥ 0 ∧ 
  ((-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)) - ((-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) = 1 :=
by sorry

/-- Part 1: Checking if x^2 + x - 2 = 0 is a neighboring root equation -/
theorem part1_not_neighboring_root : ¬ is_neighboring_root_equation 1 1 (-2) :=
by sorry

/-- Part 2: Finding k for which x^2 - (k-3)x - 3k = 0 is a neighboring root equation -/
theorem part2_find_k : 
  ∀ k : ℝ, is_neighboring_root_equation 1 (3-k) (-3*k) ↔ k = -2 ∨ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_neighboring_root_equation_conditions_part1_not_neighboring_root_part2_find_k_l1273_127391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_triangle_area_l1273_127364

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.A = Real.sqrt 10 / 10 ∧
  Real.cos t.C = Real.sqrt 5 / 5

-- Theorem 1: Measure of angle B
theorem angle_B_measure (t : Triangle) (h : triangle_conditions t) : t.B = Real.pi / 4 := by
  sorry

-- Theorem 2: Area of triangle ABC
theorem triangle_area (t : Triangle) (h : triangle_conditions t) (hc : t.c = 4) : 
  (1 / 2) * t.b * t.c * Real.sin t.A = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_measure_triangle_area_l1273_127364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_divisible_by_13_l1273_127301

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := sorry

/-- Evaluate a polynomial at a point -/
def evaluate (p : IntPolynomial n) (x : Fin n → ℤ) : ℤ := sorry

/-- Count solutions modulo 13 -/
def countSolutions (p : IntPolynomial n) : ℕ :=
  Fintype.card {x : Fin n → Fin 13 | evaluate p (fun i => x i) % 13 = 0}

theorem solutions_divisible_by_13 (n : ℕ) (p : IntPolynomial n) :
  degree p < n → 13 ∣ countSolutions p := by
  sorry

#check solutions_divisible_by_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_divisible_by_13_l1273_127301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_formula_specific_area_calculation_l1273_127315

/-- Represents an isosceles trapezoid with given parameters -/
structure IsoscelesTrapezoid where
  h : ℝ       -- height of the trapezoid
  α : ℝ       -- half of the angle at which upper base is viewed from midpoint of lower base
  β : ℝ       -- half of the angle at which lower base is viewed from midpoint of upper base

/-- Calculates the area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  t.h^2 * (Real.tan t.α + Real.tan t.β)

/-- Theorem stating that the area formula for an isosceles trapezoid is correct -/
theorem isosceles_trapezoid_area_formula (t : IsoscelesTrapezoid) :
  area t = t.h^2 * (Real.tan t.α + Real.tan t.β) :=
by
  sorry

/-- Calculates the specific area for h = 2, α = 15°, β = 75° -/
noncomputable def specific_area : ℝ :=
  area { h := 2, α := 15 * Real.pi / 180, β := 75 * Real.pi / 180 }

/-- Theorem stating that the specific area calculation is correct -/
theorem specific_area_calculation :
  specific_area = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_formula_specific_area_calculation_l1273_127315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robots_convergence_l1273_127390

-- Define the grid
structure Grid :=
  (height : ℕ)
  (width : ℕ)
  (is_passable : ℕ → ℕ → Bool → Bool → Bool)

-- Define a robot's position
structure Position :=
  (x : ℕ)
  (y : ℕ)

-- Define the set of possible commands
inductive Command
  | Up
  | Down
  | Left
  | Right

-- Define a function to check if a position is within the grid
def is_valid_position (g : Grid) (p : Position) : Prop :=
  p.x < g.width ∧ p.y < g.height

-- Define a function to check if a move is valid
def is_valid_move (g : Grid) (p : Position) (c : Command) : Bool :=
  match c with
  | Command.Up    => g.is_passable p.x p.y true false ∧ p.y + 1 < g.height
  | Command.Down  => g.is_passable p.x p.y false true ∧ p.y > 0
  | Command.Left  => g.is_passable p.x p.y false false ∧ p.x > 0
  | Command.Right => g.is_passable p.x p.y true true ∧ p.x + 1 < g.width

-- Define a function to apply a command to a position
def apply_command (g : Grid) (p : Position) (c : Command) : Position :=
  if is_valid_move g p c then
    match c with
    | Command.Up    => { x := p.x, y := p.y + 1 }
    | Command.Down  => { x := p.x, y := p.y - 1 }
    | Command.Left  => { x := p.x - 1, y := p.y }
    | Command.Right => { x := p.x + 1, y := p.y }
  else p

-- Define the theorem
theorem robots_convergence
  (g : Grid)
  (robots : List Position)
  (h1 : ∀ r, r ∈ robots → is_valid_position g r)
  (h2 : ∀ (r : Position) (target : Position),
        is_valid_position g r → is_valid_position g target →
        ∃ (commands : List Command),
          (commands.foldl (apply_command g) r) = target) :
  ∃ (commands : List Command),
    ∀ r₁ r₂, r₁ ∈ robots → r₂ ∈ robots →
      (commands.foldl (apply_command g) r₁) = (commands.foldl (apply_command g) r₂) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robots_convergence_l1273_127390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_coverage_l1273_127320

/-- A sequence of points defined as 1/n for n ∈ ℕ₊ -/
def harmonic_sequence (n : ℕ+) : ℚ := 1 / n

/-- Checks if a point is covered by a closed interval -/
def is_covered (point : ℚ) (interval : Set ℚ) : Prop :=
  point ∈ interval

/-- Checks if all points in a sequence up to n are covered by a set of intervals -/
def all_covered (n : ℕ) (intervals : List (Set ℚ)) : Prop :=
  ∀ i : ℕ+, i.val ≤ n → ∃ interval ∈ intervals, is_covered (harmonic_sequence i) interval

/-- The main theorem stating the smallest k for which all points can be covered -/
theorem smallest_k_for_coverage : 
  ∀ k : ℕ+, 
    (∃ intervals : List (Set ℚ), 
      intervals.length = 5 ∧ 
      (∀ interval ∈ intervals, ∃ a b : ℚ, interval = Set.Icc a b ∧ b - a = 1 / k) ∧
      (∀ n : ℕ, all_covered n intervals)) 
    ↔ k ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_coverage_l1273_127320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_101st_term_l1273_127309

def next_term (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 + 1 else (n + 1) / 2

def sequence_term (n : ℕ) : ℕ :=
  match n with
  | 0 => 16
  | m + 1 => next_term (sequence_term m)

theorem sequence_101st_term :
  sequence_term 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_101st_term_l1273_127309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_problem_l1273_127302

/-- Represents a circle in the coordinate plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center_x - c2.center_x)^2 + (c1.center_y - c2.center_y)^2 = (c1.radius + c2.radius)^2

/-- Generates the kth circle based on a_k -/
noncomputable def generate_circle (a_k : ℝ) : Circle :=
  { center_x := a_k
  , center_y := (1/4) * a_k^2
  , radius := (1/4) * a_k^2 }

/-- The main theorem to prove -/
theorem circles_problem (a : Fin 2018 → ℝ) : 
  (∀ k : Fin 2017, a (k.succ) < a k) → 
  (∀ k : Fin 2017, externally_tangent (generate_circle (a k)) (generate_circle (a k.succ))) →
  a (Fin.last 2017) = 1 / 2018 →
  a 0 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_problem_l1273_127302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_eq_cbrt_54_times_sqrt_2_l1273_127307

/-- A two-dimensional coordinate system with special points O, A, and B -/
structure SpecialTriangle where
  /-- The length of OA -/
  oa : ℝ
  /-- The angle AOB in radians -/
  angle_aob : ℝ
  /-- Assertion that O is the origin -/
  o_is_origin : True
  /-- Assertion that A is on the positive x-axis -/
  a_on_x_axis : True
  /-- Assertion that B is on the positive y-axis -/
  b_on_y_axis : True
  /-- The length of OA is the cube root of 54 -/
  oa_eq_cbrt_54 : oa = (54 : ℝ) ^ (1/3)
  /-- The angle AOB is 45 degrees (π/4 radians) -/
  angle_aob_eq_pi_div_4 : angle_aob = Real.pi / 4

/-- The length of AB in the special triangle -/
noncomputable def length_AB (t : SpecialTriangle) : ℝ :=
  t.oa * Real.sqrt 2

/-- Theorem: The length of AB in the special triangle is the cube root of 54 times the square root of 2 -/
theorem length_AB_eq_cbrt_54_times_sqrt_2 (t : SpecialTriangle) :
  length_AB t = (54 : ℝ) ^ (1/3) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_eq_cbrt_54_times_sqrt_2_l1273_127307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_earnings_l1273_127356

noncomputable def main_job_hourly_rate : ℝ := 20
noncomputable def second_job_rate_reduction : ℝ := 0.2
noncomputable def main_job_hours : ℝ := 30

noncomputable def second_job_hourly_rate : ℝ := main_job_hourly_rate * (1 - second_job_rate_reduction)
noncomputable def second_job_hours : ℝ := main_job_hours / 2

noncomputable def weekly_earnings : ℝ := main_job_hourly_rate * main_job_hours + second_job_hourly_rate * second_job_hours

theorem james_weekly_earnings :
  weekly_earnings = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_earnings_l1273_127356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_values_l1273_127339

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (|x + 3| - |x - 7|)

-- Define the set of m values for which f(x) > m has a solution
def M : Set ℝ := {m : ℝ | ∃ x : ℝ, f x > m}

-- Theorem statement
theorem range_of_m_values : M = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_values_l1273_127339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_number_of_zeros_l1273_127388

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x + a * x

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a ≥ 0 → Monotone (f a)) ∧
  (a < 0 → ∃ c : ℝ, c = Real.log (-a/2) ∧
    (∀ x y, x < y → x < c → f a y < f a x) ∧
    (∀ x y, x < y → c < x → f a x < f a y)) :=
sorry

-- Theorem for number of zeros
theorem number_of_zeros (a : ℝ) :
  (a ≥ 0 → ∀ x > 0, f a x ≠ 0) ∧
  (a = -2 * Real.exp 1 → ∃! x, x > 0 ∧ f a x = 0) ∧
  (a < -2 * Real.exp 1 → ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_number_of_zeros_l1273_127388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1273_127340

/-- The range of real number a, given that p is a sufficient but not necessary condition for q -/
theorem range_of_a : ∃ (S : Set ℝ), S = Set.Ioc (-2) (-1) ∧
  ∀ (a : ℝ), a ∈ S ↔
    (∀ x : ℝ, (1 / (x - 1) < 1) → (x^2 + (a - 1)*x - a > 0)) ∧
    (∃ x : ℝ, (x^2 + (a - 1)*x - a > 0) ∧ (1 / (x - 1) ≥ 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1273_127340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_70_l1273_127304

/-- Represents the distance traveled by a car in each hour of a 6-hour journey --/
def hourly_distances : Fin 6 → ℝ
  | 0 => 80  -- first hour
  | 1 => 40  -- second hour
  | 2 => 60  -- third hour
  | 3 => 50  -- fourth hour
  | 4 => 90  -- fifth hour
  | 5 => 100 -- sixth hour

/-- The total time of the journey in hours --/
def total_time : ℝ := 6

/-- Calculates the total distance traveled --/
def total_distance : ℝ := (Finset.range 6).sum (λ i => hourly_distances i)

/-- Theorem stating that the average speed of the car is 70 km/h --/
theorem average_speed_is_70 : total_distance / total_time = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_70_l1273_127304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_g_l1273_127312

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3

noncomputable def g (x : ℝ) : ℝ := 3^x - 2

theorem solution_set_f_g (x : ℝ) :
  f (g x) > 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioi (Real.log 5 / Real.log 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_g_l1273_127312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l1273_127313

/-- Given a line segment with one endpoint at (7, 4) and midpoint at (9, -15),
    the sum of the coordinates of the other endpoint is -23. -/
theorem endpoint_coordinate_sum (A M B : ℝ × ℝ) : 
  A = (7, 4) ∧ 
  M = (9, -15) ∧ 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B.1 + B.2 = -23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l1273_127313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l1273_127368

theorem triangle_tangent_ratio (A B C : ℝ) (a b c : ℝ) :
  A = 60 * Real.pi / 180 →
  a > 0 → b > 0 → c > 0 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (Real.tan A - Real.tan B) / (Real.tan A + Real.tan B) = (c - b) / c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l1273_127368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1273_127333

-- Define the hyperbola
def hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the foci
noncomputable def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Define angle function (placeholder)
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define area_triangle function (placeholder)
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem hyperbola_properties
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (P : ℝ × ℝ) (hP : P ∈ hyperbola a b h₁ h₂)
  (hP_right : P.1 > 0)
  (hAngle : angle (foci a b).1 P (foci a b).2 = π / 3)
  (hArea : area_triangle (foci a b).1 P (foci a b).2 = 3 * Real.sqrt 3 * a^2) :
  (eccentricity a b = 2) ∧
  (∀ Q : ℝ × ℝ, Q ∈ hyperbola a b h₁ h₂ → Q.1 ≥ 0 → Q.2 ≥ 0 →
    angle Q (foci a b).2 (-a, 0) = 2 * angle Q (-a, 0) (foci a b).2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1273_127333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_unit_vectors_l1273_127379

/-- Given a vector a = (4, 3), prove that (4/5, 3/5) and (-4/5, -3/5) are the only unit vectors collinear with a. -/
theorem collinear_unit_vectors (a : ℝ × ℝ) (h : a = (4, 3)) :
  let unit_vectors := {v : ℝ × ℝ | ∃ (k : ℝ), v = k • a ∧ ‖v‖ = 1}
  unit_vectors = {(4/5, 3/5), (-4/5, -3/5)} := by
  sorry

#check collinear_unit_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_unit_vectors_l1273_127379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1273_127341

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := 1 / x + 9 / (2 - x)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = 8 ∧ 
  (∀ x : ℝ, 0 < x → x < 2 → f x ≥ m) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = m) := by
  -- Proof skeleton
  -- We'll use 1/2 as the critical point
  let x₀ : ℝ := 1/2
  -- Define m as f(x₀)
  let m : ℝ := f x₀
  
  -- Prove the three parts of the conjunction
  have h1 : m = 8 := by sorry
  have h2 : ∀ x : ℝ, 0 < x → x < 2 → f x ≥ m := by sorry
  have h3 : 0 < x₀ ∧ x₀ < 2 ∧ f x₀ = m := by sorry
  
  -- Combine the parts to prove the theorem
  exact ⟨m, h1, h2, ⟨x₀, h3⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1273_127341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1273_127392

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (3 * x^2 - 2 * a * x) / Real.log a

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, 1/2 ≤ x ∧ x < y ∧ y ≤ 1 → f a x > f a y) →
  (0 < a ∧ a < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1273_127392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1273_127387

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x-value in a quadratic function -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the x-coordinate of the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertexX (f : QuadraticFunction) : ℝ :=
  -f.b / (2 * f.a)

/-- Calculates the y-coordinate of the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertexY (f : QuadraticFunction) : ℝ :=
  f.evaluate (f.vertexX)

/-- Represents the vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Represents a point on the y-axis -/
structure YIntercept where
  y : ℝ

theorem parabola_properties (f : QuadraticFunction) 
    (h : f = { a := 3, b := -6, c := 2 }) : 
    (Vertex.mk (f.vertexX) (f.vertexY) = Vertex.mk 1 (-1)) ∧ 
    (YIntercept.mk (f.evaluate 0) = YIntercept.mk 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1273_127387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_O₂_l1273_127359

/-- The radius of the sector AOB -/
noncomputable def r : ℝ := Real.pi

/-- The angle θ, where the central angle of sector AOB is 2θ -/
noncomputable def θ : ℝ := Real.pi / 4

/-- The radius of circle O₁ -/
noncomputable def r₁ (r θ : ℝ) : ℝ := (r * Real.sin θ) / (1 + Real.sin θ)

/-- The radius of circle O₂ -/
noncomputable def r₂ (r θ : ℝ) : ℝ := (r * Real.sin θ * (1 - Real.sin θ)) / ((1 + Real.sin θ)^2)

/-- The area of circle O₂ -/
noncomputable def area_O₂ (r θ : ℝ) : ℝ := Real.pi * (r₂ r θ)^2

theorem max_area_O₂ (r : ℝ) (hr : r > 0) (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi/2) :
  (∃ (θ_max : ℝ), θ_max ∈ Set.Ioo 0 (Real.pi/2) ∧ 
    (∀ θ' ∈ Set.Ioo 0 (Real.pi/2), area_O₂ r θ' ≤ area_O₂ r θ_max)) ∧
  (∃ (θ_max : ℝ), θ_max ∈ Set.Ioo 0 (Real.pi/2) ∧ 
    area_O₂ r θ_max = (r^2 * Real.pi) / 64 ∧
    Real.sin θ_max = 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_O₂_l1273_127359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_carriages_count_l1273_127397

theorem train_carriages_count (train_speed carriage_length engine_length bridge_length crossing_time : ℝ) 
  (h1 : train_speed = 60)
  (h2 : carriage_length = 60)
  (h3 : engine_length = 60)
  (h4 : bridge_length = 4.5)
  (h5 : crossing_time = 6) : ℕ := by
  -- The number of carriages is 24
  sorry

#check train_carriages_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_carriages_count_l1273_127397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_correct_seats_probability_general_last_two_correct_seats_probability_l1273_127337

/-- Represents the seating arrangement process for students in a saloon. -/
def seatingProcess (n : ℕ) : Type :=
  Fin n → Fin n

/-- The probability that the last two students sit on their correct seats. -/
noncomputable def lastTwoCorrectProbability (n : ℕ) : ℝ :=
  1 / 3

/-- Theorem stating that for 100 students and seats, the probability of the last two sitting correctly is 1/3. -/
theorem last_two_correct_seats_probability :
  lastTwoCorrectProbability 100 = 1 / 3 := by
  sorry

/-- Generalization: For n students, the probability of the last two sitting correctly is 1/3. -/
theorem general_last_two_correct_seats_probability (n : ℕ) (h : n ≥ 3) :
  lastTwoCorrectProbability n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_correct_seats_probability_general_last_two_correct_seats_probability_l1273_127337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_120_l1273_127393

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the properties of the rectangle
def is_rectangle (r : Rectangle) : Prop :=
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧
  r.B.1 - r.A.1 = w ∧ r.B.2 - r.A.2 = 0 ∧
  r.C.1 - r.B.1 = 0 ∧ r.C.2 - r.B.2 = -h ∧
  r.D.1 - r.A.1 = 0 ∧ r.D.2 - r.A.2 = -h

-- Define the length of AB
noncomputable def AB_length (r : Rectangle) : ℝ :=
  Real.sqrt ((r.B.1 - r.A.1)^2 + (r.B.2 - r.A.2)^2)

-- Define the length of AC (diagonal)
noncomputable def AC_length (r : Rectangle) : ℝ :=
  Real.sqrt ((r.C.1 - r.A.1)^2 + (r.C.2 - r.A.2)^2)

-- Define the area of the rectangle
def rectangle_area (r : Rectangle) : ℝ :=
  (r.B.1 - r.A.1) * (r.A.2 - r.D.2)

-- Theorem statement
theorem rectangle_area_is_120 (r : Rectangle) 
  (h_rect : is_rectangle r)
  (h_AB : AB_length r = 15)
  (h_AC : AC_length r = 17) :
  rectangle_area r = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_120_l1273_127393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_AFB_l1273_127328

/-- Parabola with focus F and points A and B -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_parabola_A : (A.2)^2 = 2 * p * A.1
  h_parabola_B : (B.2)^2 = 2 * p * B.1
  h_condition : A.1 + B.1 + p = (2 * Real.sqrt 3 / 3) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The maximum angle AFB in the parabola configuration -/
theorem max_angle_AFB (para : Parabola) : 
  ∃ (θ : ℝ), θ ≤ 2 * Real.pi / 3 ∧ 
  ∀ (θ' : ℝ), θ' = Real.arccos ((para.A.1 - para.F.1)^2 + (para.A.2 - para.F.2)^2 + 
                               (para.B.1 - para.F.1)^2 + (para.B.2 - para.F.2)^2 - 
                               ((para.A.1 - para.B.1)^2 + (para.A.2 - para.B.2)^2)) / 
                              (2 * Real.sqrt ((para.A.1 - para.F.1)^2 + (para.A.2 - para.F.2)^2) * 
                                    Real.sqrt ((para.B.1 - para.F.1)^2 + (para.B.2 - para.F.2)^2)) →
  θ' ≤ θ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_AFB_l1273_127328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_in_second_quadrant_l1273_127361

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sin 5, Real.cos 5)

-- Define the property of being in the second quadrant
def is_in_second_quadrant (point : ℝ × ℝ) : Prop :=
  point.1 < 0 ∧ point.2 > 0

-- Theorem statement
theorem P_in_second_quadrant : is_in_second_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_in_second_quadrant_l1273_127361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l1273_127358

/-- A complex number is in the third quadrant if its real part is negative and its imaginary part is negative. -/
def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The given complex number -/
def z : ℂ := -1 - 3*Complex.I

/-- Theorem: The complex number z is in the third quadrant -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l1273_127358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_of_f_l1273_127377

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - 2*x - x^2)

-- State the theorem
theorem monotonically_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = -1 ∧ b = 1 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x) ∧
  (∀ c d, c < a ∨ b < d → ¬(∀ x y, c ≤ x ∧ x < y ∧ y ≤ d → f y ≤ f x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_of_f_l1273_127377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_equals_2sqrt2Q_l1273_127314

/-- A right parallelepiped with a rhombic base -/
structure RhombicParallelepiped where
  /-- The side length of the rhombic base -/
  base_side : ℝ
  /-- The height of the parallelepiped -/
  height : ℝ

/-- The cross-section formed by a plane passing through one side of the lower base
    and the opposite side of the upper base, forming a 45° angle with the base plane -/
noncomputable def cross_section (p : RhombicParallelepiped) : ℝ := 
  p.base_side * (Real.sqrt 2 * p.height)

/-- The lateral surface area of the parallelepiped -/
def lateral_surface_area (p : RhombicParallelepiped) : ℝ := 
  4 * p.base_side * p.height

theorem lateral_surface_area_equals_2sqrt2Q (p : RhombicParallelepiped) (Q : ℝ) 
    (h : cross_section p = Q) : 
  lateral_surface_area p = 2 * Real.sqrt 2 * Q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_equals_2sqrt2Q_l1273_127314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_37_l1273_127306

/-- The angle of the minute hand at a given minute -/
def minuteHandAngle (minutes : ℕ) : ℝ := 6 * minutes

/-- The angle of the hour hand at a given hour and minute -/
def hourHandAngle (hours minutes : ℕ) : ℝ := 30 * hours + 0.5 * minutes

/-- The smaller angle between two angles on a circle -/
noncomputable def smallerAngle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

/-- The theorem stating that the smaller angle between clock hands at 5:37 is 53.5 degrees -/
theorem clock_angle_at_5_37 :
  smallerAngle (hourHandAngle 5 37) (minuteHandAngle 37) = 53.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_37_l1273_127306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l1273_127351

noncomputable section

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi/6) = 1/2

-- Define the curve C in parametric form
def curve_C (α : ℝ) : ℝ × ℝ := (1 + 3 * Real.cos α, 3 * Real.sin α)

-- State the theorem
theorem line_curve_intersection :
  -- Given the line l and curve C as defined above
  (∀ ρ θ : ℝ, line_l ρ θ ↔ ρ * Real.sin (θ - Real.pi/6) = 1/2) →
  (∀ α : ℝ, curve_C α = (1 + 3 * Real.cos α, 3 * Real.sin α)) →
  -- 1. The rectangular equation of line l
  (∀ x y : ℝ, (x - Real.sqrt 3 * y + 1 = 0) ↔ ∃ ρ θ : ℝ, line_l ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  -- 2. The ordinary equation of curve C
  (∀ x y : ℝ, ((x - 1)^2 + y^2 = 9) ↔ ∃ α : ℝ, curve_C α = (x, y)) →
  -- 3. Line l intersects curve C
  (∃ x y : ℝ, (x - Real.sqrt 3 * y + 1 = 0) ∧ ((x - 1)^2 + y^2 = 9)) →
  -- 4. The length of the chord of intersection
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - Real.sqrt 3 * y₁ + 1 = 0) ∧ ((x₁ - 1)^2 + y₁^2 = 9) ∧
    (x₂ - Real.sqrt 3 * y₂ + 1 = 0) ∧ ((x₂ - 1)^2 + y₂^2 = 9) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 32)) →
  -- Conclusion: All of the above statements are true
  True := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l1273_127351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_not_one_l1273_127344

-- Define IsTriangle as a predicate
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ IsTriangle a b c :=
sorry

theorem third_side_not_one (x : ℝ) :
  x > 0 → IsTriangle 5 7 x → x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_not_one_l1273_127344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angleAt9am_l1273_127383

/-- The number of hours on a clock face -/
def clockHours : ℕ := 12

/-- The angle corresponding to each hour on a clock face -/
noncomputable def anglePerHour : ℚ := 360 / clockHours

/-- The position of the hour hand at 9:00 a.m. in terms of hours from 12 o'clock -/
def hourHandPosition : ℕ := 9

/-- The position of the minute hand at 9:00 a.m. in terms of hours from 12 o'clock -/
def minuteHandPosition : ℕ := 0

/-- The angle between the minute hand and the hour hand at 9:00 a.m. -/
theorem angleAt9am :
  (hourHandPosition * anglePerHour : ℚ) - (minuteHandPosition * anglePerHour : ℚ) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angleAt9am_l1273_127383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_from_tax_l1273_127396

/-- Special municipal payroll tax function -/
noncomputable def payroll_tax (payroll : ℝ) : ℝ :=
  if payroll ≤ 200000 then 0 else 0.002 * (payroll - 200000)

/-- Theorem: If a company pays $200 in the special municipal payroll tax, 
    their total payroll is $300,000 -/
theorem payroll_from_tax (tax_paid : ℝ) (h : tax_paid = 200) :
  ∃ (payroll : ℝ), payroll_tax payroll = tax_paid ∧ payroll = 300000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_from_tax_l1273_127396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accident_increase_is_ten_percent_l1273_127316

/-- Calculates the percentage increase for accidents given insurance premium information -/
noncomputable def accident_percentage_increase (initial_premium : ℝ) (ticket_increase : ℝ) (num_tickets : ℕ) (new_premium : ℝ) : ℝ :=
  let total_increase := new_premium - initial_premium
  let ticket_total_increase := ticket_increase * (num_tickets : ℝ)
  let accident_increase := total_increase - ticket_total_increase
  (accident_increase / initial_premium) * 100

/-- Theorem stating that the percentage increase for accidents is 10% given the problem conditions -/
theorem accident_increase_is_ten_percent :
  let initial_premium : ℝ := 50
  let ticket_increase : ℝ := 5
  let num_tickets : ℕ := 3
  let new_premium : ℝ := 70
  accident_percentage_increase initial_premium ticket_increase num_tickets new_premium = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accident_increase_is_ten_percent_l1273_127316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_age_2003_l1273_127321

/-- Walter's age at the end of 1998 -/
def walter_age_1998 : ℝ := sorry

/-- Walter's grandmother's age at the end of 1998 -/
def grandmother_age_1998 : ℝ := sorry

/-- The year Walter was born -/
def walter_birth_year : ℝ := sorry

/-- The year Walter's grandmother was born -/
def grandmother_birth_year : ℝ := sorry

theorem walter_age_2003 :
  walter_age_1998 = (1 / 3) * grandmother_age_1998 →
  walter_birth_year + grandmother_birth_year = 3858 →
  walter_birth_year = 1998 - walter_age_1998 →
  grandmother_birth_year = 1998 - grandmother_age_1998 →
  walter_age_1998 + 5 = 39.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_age_2003_l1273_127321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_polynomial_sum_of_squares_l1273_127363

/-- A polynomial that takes only positive values for all real x -/
def PositivePolynomial (P : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, P x > 0) ∧ (∃ n : ℕ, ∀ x : ℝ, ∃ c : Polynomial ℝ, P x = c.eval x ∧ c.degree ≤ n)

/-- The main theorem stating that any positive polynomial can be expressed as the sum of two squared polynomials -/
theorem positive_polynomial_sum_of_squares 
  (P : ℝ → ℝ) (h : PositivePolynomial P) : 
  ∃ (a b : ℝ → ℝ), 
    (∃ n : ℕ, ∀ x : ℝ, ∃ ca : Polynomial ℝ, a x = ca.eval x ∧ ca.natDegree ≤ n) ∧
    (∃ m : ℕ, ∀ x : ℝ, ∃ cb : Polynomial ℝ, b x = cb.eval x ∧ cb.natDegree ≤ m) ∧
    (∀ x : ℝ, P x = (a x)^2 + (b x)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_polynomial_sum_of_squares_l1273_127363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_weighted_score_l1273_127357

def weightedScore (s1 s2 w : ℝ) : ℝ := (s1 + s2) * w

theorem eva_weighted_score :
  let maths_s2 : ℝ := 80
  let arts_s2 : ℝ := 90
  let science_s2 : ℝ := 90
  let history_s2 : ℝ := 85

  let maths_s1 : ℝ := maths_s2 + 10
  let arts_s1 : ℝ := arts_s2 - 15
  let science_s1 : ℝ := science_s2 * (2/3)
  let history_s1 : ℝ := history_s2 + 5

  let maths_weight : ℝ := 0.3
  let arts_weight : ℝ := 0.25
  let science_weight : ℝ := 0.35
  let history_weight : ℝ := 0.1

  weightedScore maths_s1 maths_s2 maths_weight +
  weightedScore arts_s1 arts_s2 arts_weight +
  weightedScore science_s1 science_s2 science_weight +
  weightedScore history_s1 history_s2 history_weight = 164.875 := by
  sorry

def main : IO Unit := do
  let maths_s2 : Float := 80
  let arts_s2 : Float := 90
  let science_s2 : Float := 90
  let history_s2 : Float := 85

  let maths_s1 : Float := maths_s2 + 10
  let arts_s1 : Float := arts_s2 - 15
  let science_s1 : Float := science_s2 * (2/3)
  let history_s1 : Float := history_s2 + 5

  let maths_weight : Float := 0.3
  let arts_weight : Float := 0.25
  let science_weight : Float := 0.35
  let history_weight : Float := 0.1

  let result : Float := 
    (maths_s1 + maths_s2) * maths_weight +
    (arts_s1 + arts_s2) * arts_weight +
    (science_s1 + science_s2) * science_weight +
    (history_s1 + history_s2) * history_weight

  IO.println s!"The total weighted score is {result}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_weighted_score_l1273_127357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_in_rolls_l1273_127362

def dice_roll := Fin 6

def median (rolls : List ℕ) : ℚ :=
  sorry

def variance (rolls : List ℕ) : ℚ :=
  sorry

theorem no_six_in_rolls (rolls : List ℕ) 
  (h_size : rolls.length = 5)
  (h_median : median rolls = 3)
  (h_variance : variance rolls = 16/100) : 
  ∀ x ∈ rolls, x ≠ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_in_rolls_l1273_127362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l1273_127300

/-- Represents an alloy with a given weight and ratio of components -/
structure Alloy where
  weight : ℝ
  ratio1 : ℝ
  ratio2 : ℝ

/-- Calculates the amount of the second component in an alloy -/
noncomputable def amountOfSecondComponent (a : Alloy) : ℝ :=
  (a.ratio2 / (a.ratio1 + a.ratio2)) * a.weight

/-- Alloy A with 100 kg and lead:tin ratio of 5:3 -/
def alloyA : Alloy :=
  { weight := 100
    ratio1 := 5
    ratio2 := 3 }

/-- Alloy B with 200 kg and tin:copper ratio of 2:3 -/
def alloyB : Alloy :=
  { weight := 200
    ratio1 := 2
    ratio2 := 3 }

/-- Theorem stating the amount of tin in the mixture of alloys A and B -/
theorem tin_in_mixture : amountOfSecondComponent alloyA + amountOfSecondComponent alloyB = 117.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_in_mixture_l1273_127300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1273_127355

/-- Given two mutually perpendicular unit vectors a and b in a plane,
    and a vector c satisfying (a-c)⋅(b-c) = 0,
    the maximum value of |c| is √2. -/
theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • b = 0) → 
  ((a - c) • (b - c) = 0) → 
  (∃ (k : ℝ), ‖c‖ ≤ k ∧ k = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l1273_127355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1273_127331

def is_arithmetic_progression (a : List ℝ) : Prop :=
  a.length ≥ 3 ∧ ∀ i : Fin (a.length - 2), a[i.val + 1] - a[i.val] = a[i.val + 2] - a[i.val + 1]

theorem arithmetic_sequence_middle_term (y : ℝ) : 
  y > 0 → 
  is_arithmetic_progression [2^2, y^2, 5^2] → 
  y = Real.sqrt 14.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1273_127331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_symmetry_l1273_127310

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, f (-x) = -f x)
variable (x₀ : ℝ)
variable (k : ℝ)
variable (h3 : k ≠ 0)
variable (h4 : deriv f (-x₀) = k)

-- State the theorem
theorem derivative_symmetry : deriv f x₀ = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_symmetry_l1273_127310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1273_127354

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem trajectory_of_P (x y : ℝ) :
  distance (x, y) A = 2 * distance (x, y) B →
  (x - 2)^2 + y^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l1273_127354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1273_127318

/-- The function f(x) = ln(1+x) - x + (k/2)x^2 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - x + (k / 2) * x^2

theorem f_inequality (k : ℝ) (x : ℝ) (hk : k ≥ 0) (hx : x > -1) :
  k = 0 → Real.log (x + 1) ≥ 1 - 1 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1273_127318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_area_perimeter_is_500_l1273_127378

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A square on a 2D grid --/
structure GridSquare where
  p : GridPoint
  q : GridPoint
  r : GridPoint
  s : GridPoint

/-- Calculate the distance between two grid points --/
noncomputable def distance (a b : GridPoint) : ℝ :=
  Real.sqrt (((a.x - b.x : ℤ) ^ 2 + (a.y - b.y : ℤ) ^ 2) : ℝ)

/-- Calculate the side length of a grid square --/
noncomputable def sideLength (square : GridSquare) : ℝ :=
  distance square.p square.q

/-- Calculate the area of a grid square --/
noncomputable def area (square : GridSquare) : ℝ :=
  (sideLength square) ^ 2

/-- Calculate the perimeter of a grid square --/
noncomputable def perimeter (square : GridSquare) : ℝ :=
  4 * (sideLength square)

/-- The specific square PQRS from the problem --/
def squarePQRS : GridSquare :=
  { p := { x := 1, y := 6 },
    q := { x := 6, y := 6 },
    r := { x := 6, y := 1 },
    s := { x := 1, y := 1 } }

theorem product_area_perimeter_is_500 :
  area squarePQRS * perimeter squarePQRS = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_area_perimeter_is_500_l1273_127378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parabola_l1273_127352

/-- The equation of the tangent line to the parabola y = x^2 - 4x at x = 1 is y = -2x - 1 -/
theorem tangent_line_parabola :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  let x₀ : ℝ := 1
  let m : ℝ := (deriv f) x₀
  let b : ℝ := f x₀ - m * x₀
  (λ x y ↦ y = m * x + b) = (λ x y ↦ y = -2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parabola_l1273_127352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1273_127326

-- Define the function f
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := n * x^(n^2 + 2*n)

-- Define what it means for a function to be a power function
def isPowerFunction (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, g x = a * x^b

-- Define what it means for a function to be monotonically increasing on (0, +∞)
def isMonoIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → g x < g y

-- State the theorem
theorem problem_statement :
  (∃ n : ℝ, isPowerFunction (f n) ∧ isMonoIncreasing (f n)) ∧
  ¬(∀ x : ℝ, x^2 + 2 < 3*x ↔ ¬(∃ x : ℝ, x^2 + 2 > 3*x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1273_127326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_l1273_127395

theorem least_marbles (n : ℕ) : 
  (∀ d ∈ ({2, 4, 5, 7, 8, 10} : Set ℕ), d ∣ n) → n ≥ 280 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_l1273_127395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_width_to_perimeter_ratio_l1273_127365

/-- Proves that for a rectangular garden with length 25 feet and width 15 feet,
    the ratio of its width to its perimeter is equal to 3:16. -/
theorem garden_width_to_perimeter_ratio :
  let length : ℚ := 25
  let width : ℚ := 15
  let perimeter : ℚ := 2 * (length + width)
  (width / perimeter) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_width_to_perimeter_ratio_l1273_127365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1273_127376

/-- Given an initial amount that grows to a final amount over a certain time period,
    calculate the annual interest rate. -/
noncomputable def calculate_interest_rate (initial_amount final_amount : ℝ) (time : ℝ) : ℝ :=
  (final_amount / initial_amount - 1) / time

/-- The problem statement -/
theorem interest_rate_problem (initial_amount final_amount time : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : final_amount = 1000)
  (h3 : time = 5) :
  calculate_interest_rate initial_amount final_amount time = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1273_127376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1273_127334

def sequence_a : ℕ → ℚ
  | 0 => 2/3  -- Add case for 0
  | 1 => 2/3
  | (n+2) => 2 * sequence_a (n+1) / (sequence_a (n+1) + 1)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a n - 1

def sequence_c (n : ℕ) : ℚ := n / sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n+1) = (1/2) * sequence_b n) ∧
  (∀ n : ℕ, n ≥ 1 → 
    (Finset.range n).sum (λ i => sequence_c (i+1)) = 
      (n^2 + n + 4) / 2 - (2 + n : ℚ) / (2^n)) := by
  sorry

#eval sequence_a 3  -- Test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1273_127334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_absolute_sum_diff_l1273_127386

/-- The area enclosed by the graph of |x + y| + |x - y| ≤ 6 is 36 square units. -/
theorem area_of_absolute_sum_diff : 
  (MeasureTheory.volume (Set.Icc (-3 : ℝ) 3 ×ˢ Set.Icc (-3 : ℝ) 3)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_absolute_sum_diff_l1273_127386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l1273_127370

noncomputable section

-- Define the number of observations
def n : ℕ := 8

-- Define the sums from the given conditions
def sum_x : ℝ := 52
def sum_y : ℝ := 228
def sum_x_sq : ℝ := 478
def sum_xy : ℝ := 1849

-- Define the mean of x and y
def mean_x : ℝ := sum_x / n
def mean_y : ℝ := sum_y / n

-- Define the regression coefficients
def b : ℝ := (sum_xy - n * mean_x * mean_y) / (sum_x_sq - n * mean_x^2)
def a : ℝ := mean_y - b * mean_x

-- Define a function to round to two decimal places
def roundToTwoDP (x : ℝ) : ℝ := 
  (↑(round (x * 100)) : ℝ) / 100

-- State the theorem
theorem regression_line_equation : 
  ∃ (a' b' : ℝ), 
    (roundToTwoDP a' = 11.47 ∧ roundToTwoDP b' = 2.62) ∧ 
    (∀ x y : ℝ, y = a' + b' * x → y = a + b * x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l1273_127370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_T_eq_T_T_is_valid_l1273_127342

/-- Represents a programming statement --/
inductive Statement
  | Input (s : String)
  | Assignment (s : String)
  | Print (s : String)

/-- Checks if a statement is a valid input statement --/
def isValidInput : Statement → Bool
  | Statement.Input s => s.contains ';' && !s.contains '='
  | _ => false

/-- Checks if a statement is a valid assignment statement --/
def isValidAssignment : Statement → Bool
  | Statement.Assignment s => !s.contains '='  -- simplified check
  | _ => false

/-- Checks if a statement is a valid print statement --/
def isValidPrint : Statement → Bool
  | Statement.Print s => s.contains ';' && !s.contains '='
  | _ => false

/-- The main theorem to prove --/
theorem only_T_eq_T_T_is_valid :
  let stmtA := Statement.Input "x=3"
  let stmtB := Statement.Assignment "A=B=2"
  let stmtC := Statement.Assignment "T=T*T"
  let stmtD := Statement.Print "A=4"
  (¬ isValidInput stmtA) ∧
  (¬ isValidAssignment stmtB) ∧
  (isValidAssignment stmtC) ∧
  (¬ isValidPrint stmtD) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_T_eq_T_T_is_valid_l1273_127342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1273_127348

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if a/b = d/e -/
axiom lines_parallel (a b c d e f : ℝ) : 
  (a * e = b * d) ↔ (∀ x y : ℝ, a*x + b*y + c = 0 ↔ d*x + e*y + f = 0)

/-- Definition of the first line -/
def line1 (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 2 * x + (m + 1) * y + 4 = 0

/-- Definition of the second line -/
def line2 (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => m * x + 3 * y - 2 = 0

/-- Theorem stating the condition for the lines to be parallel -/
theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, line1 m x y ↔ line2 m x y) ↔ (m = 2 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l1273_127348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_dth_power_l1273_127332

theorem existence_of_non_dth_power (d : ℕ) (ε : ℝ) (p : ℕ) (h_d : d > 0) (h_ε : ε > 0) (h_p : Nat.Prime p) (h_gcd : Nat.gcd (p - 1) d ≠ 1) :
  ∃ (r : ℝ), r > 0 ∧ Real.log r = ε - (1 / (Nat.gcd d (p - 1) : ℝ)) →
  ∃ (a : ℕ), 0 < a ∧ a < p^(Int.toNat ⌈r⌉) ∧ ¬∃ (x : ℕ), x^d ≡ a [ZMOD p] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_dth_power_l1273_127332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_vertex_ratio_l1273_127399

/-- Parabola type -/
structure Parabola where
  a : ℝ
  q : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given parabola P -/
noncomputable def P : Parabola :=
  { a := 4, q := 0 }

/-- Vertex of a parabola -/
noncomputable def vertex (p : Parabola) : Point :=
  { x := 0, y := p.q }

/-- Focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 0, y := 1 / (4 * p.a) + p.q }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Parabola Q derived from P -/
noncomputable def Q : Parabola :=
  { a := 8, q := 1/2 }

/-- Theorem: The ratio of focus-to-focus distance to vertex-to-vertex distance is 15/16 -/
theorem focus_vertex_ratio :
  distance (focus P) (focus Q) / distance (vertex P) (vertex Q) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_vertex_ratio_l1273_127399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1273_127373

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := -(Real.log 3 / Real.log 2)

theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1273_127373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1273_127324

theorem inequality_solution_set (x : ℝ) : 
  (2 : ℝ)^(|x - 2| + |x - 4|) > (2 : ℝ)^6 ↔ x < 0 ∨ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1273_127324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_japanese_students_fraction_l1273_127325

theorem japanese_students_fraction (j : ℕ) (h1 : j > 0) : 
  (((3 * j) / 4 + (2 * j) / 8 : ℚ) / (3 * j) : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_japanese_students_fraction_l1273_127325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_from_parallel_lines_perpendicular_line_from_perpendicular_planes_line_in_or_parallel_to_plane_l1273_127353

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Proposition ②
theorem parallel_planes_from_parallel_lines 
  (α β : Plane) (a b : Line) :
  skew a b →
  contains α a →
  contains β b →
  parallel_line_plane a β →
  parallel_line_plane b α →
  parallel_planes α β :=
sorry

-- Proposition ③
theorem perpendicular_line_from_perpendicular_planes 
  (α β γ : Plane) (l : Line) :
  perpendicular_planes α γ →
  perpendicular_planes β γ →
  intersect α β = l →
  perpendicular_plane_line γ l :=
sorry

-- Proposition ④
theorem line_in_or_parallel_to_plane 
  (α β : Plane) (a : Line) :
  perpendicular_planes α β →
  perpendicular_plane_line β a →
  (contains α a ∨ parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_from_parallel_lines_perpendicular_line_from_perpendicular_planes_line_in_or_parallel_to_plane_l1273_127353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_squared_special_angle_l1273_127335

/-- 
Given an angle θ with vertex at the origin, initial side along the positive x-axis, 
and terminal side on the line y = 2x, prove that cos²θ - sin²θ = -3/5 
-/
theorem cos_squared_minus_sin_squared_special_angle (θ : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ Real.cos θ = a / (Real.sqrt 5 * |a|) ∧ Real.sin θ = (2 * a) / (Real.sqrt 5 * |a|)) →
  Real.cos θ ^ 2 - Real.sin θ ^ 2 = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_squared_special_angle_l1273_127335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1273_127338

theorem problem_statement (k x y n : ℕ) : 
  (k % (x^2) = 0) →
  (k % (y^2) = 0) →
  (k / (x^2) = n) →
  (k / (y^2) = n + 148) →
  (Nat.gcd x y = 1) →
  k = 467856 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1273_127338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l1273_127317

/-- A regular square pyramid with an inscribed sphere -/
structure SquarePyramid where
  a : ℝ  -- base side length
  r : ℝ  -- radius of inscribed sphere
  h : ℝ  -- height of the pyramid
  h_eq_r : h = r  -- height equals radius of inscribed sphere

/-- The volume of a regular square pyramid -/
noncomputable def volume (p : SquarePyramid) : ℝ := (1 / 3) * p.a^2 * p.r

/-- Theorem: The volume of a regular square pyramid with base side length a
    and height equal to the radius r of its inscribed sphere is (1/3) * a^2 * r -/
theorem square_pyramid_volume (p : SquarePyramid) :
  volume p = (1 / 3) * p.a^2 * p.r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_pyramid_volume_l1273_127317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1273_127319

/-- Given a triangle ABC where angle A is 60 degrees, side b is 1, and the area is √3,
    prove that (a+b+c)/(sin A + sin B + sin C) = 2√39/3 -/
theorem triangle_ratio (a b c A B C : ℝ) (h_angle : A = π/3) (h_side : b = 1)
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l1273_127319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_redemption_days_l1273_127345

def DayOfWeek : Type := Fin 7

def is_weekend (day : Fin 7) : Prop :=
  day = 5 ∨ day = 6  -- 5 represents Saturday, 6 represents Sunday

def next_redemption_day (start : Fin 7) (n : Nat) : Fin 7 :=
  (start + 15 * n : Nat) % 7

def valid_start_day (start : Fin 7) : Prop :=
  ∀ n : Fin 5, ¬is_weekend (next_redemption_day start n)

theorem coupon_redemption_days :
  ∀ start : Fin 7, valid_start_day start ↔ (start = 0 ∨ start = 1 ∨ start = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_redemption_days_l1273_127345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equilateral_config_l1273_127384

/-- A point in a 2D plane -/
def Plane := ℝ × ℝ

/-- Check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : Plane) : Prop :=
  let d12 := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let d23 := Real.sqrt ((p2.1 - p3.1)^2 + (p2.2 - p3.2)^2)
  let d31 := Real.sqrt ((p3.1 - p1.1)^2 + (p3.2 - p1.2)^2)
  d12 = d23 ∧ d23 = d31

/-- A configuration of points on a plane satisfying the equilateral triangle property -/
structure EquilateralConfig where
  n : ℕ
  points : Fin n → Plane
  h_gt_two : n > 2
  h_equilateral : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    IsEquilateralTriangle (points i) (points j) (points k)

/-- The theorem stating that the only valid configuration has exactly 3 points -/
theorem unique_equilateral_config : 
  ∀ (config : EquilateralConfig), config.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equilateral_config_l1273_127384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focalDistanceIs6Sqrt3_l1273_127303

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  center : ℝ × ℝ
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- The ellipse is tangent to the x-axis at (6, 0) and to the y-axis at (0, 3) -/
def tangentEllipse : ParallelAxisEllipse :=
  { center := (6, 3)
  , semiMajorAxis := 6
  , semiMinorAxis := 3 }

/-- The distance between the foci of the ellipse -/
noncomputable def focalDistance (e : ParallelAxisEllipse) : ℝ :=
  2 * Real.sqrt (e.semiMajorAxis ^ 2 - e.semiMinorAxis ^ 2)

/-- Theorem: The distance between the foci of the given ellipse is 6√3 -/
theorem focalDistanceIs6Sqrt3 : focalDistance tangentEllipse = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focalDistanceIs6Sqrt3_l1273_127303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_value_l1273_127380

def mySequence (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => 5
  | n + 1 => if n % 2 = 0 then f (mySequence f n) else f (mySequence f n + 1)

theorem tenth_finger_value (f : ℕ → ℕ) (h1 : f 5 = 4) (h2 : f 4 = 3) (h3 : f 3 = 6)
  (h4 : ∀ n, n ≥ 5 → mySequence f n = 3) : mySequence f 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_finger_value_l1273_127380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_target_l1273_127311

-- Define the type of polynomials we can construct
inductive ConstructiblePoly : Type
  | x : ConstructiblePoly
  | combine : ConstructiblePoly → ConstructiblePoly → ConstructiblePoly

-- Define the evaluation function for our polynomials
def eval : ConstructiblePoly → ℝ → ℝ
  | ConstructiblePoly.x => id
  | ConstructiblePoly.combine p q => fun x => eval p x * eval q x + eval p x + eval q x + 1

-- Define our target polynomial
noncomputable def target (x : ℝ) : ℝ := (x^1983 - 1) / (x - 1)

-- The theorem we want to prove
theorem cannot_construct_target :
  ¬ ∃ p : ConstructiblePoly, ∀ x : ℝ, x ≠ 1 → eval p x = target x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_construct_target_l1273_127311
