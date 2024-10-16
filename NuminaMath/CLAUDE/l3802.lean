import Mathlib

namespace NUMINAMATH_CALUDE_camdens_dogs_legs_count_l3802_380298

theorem camdens_dogs_legs_count :
  ∀ (justin_dogs : ℕ) (rico_dogs : ℕ) (camden_dogs : ℕ),
    justin_dogs = 14 →
    rico_dogs = justin_dogs + 10 →
    camden_dogs = (3 * rico_dogs) / 4 →
    camden_dogs * 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_camdens_dogs_legs_count_l3802_380298


namespace NUMINAMATH_CALUDE_remainder_sum_l3802_380210

theorem remainder_sum (n : ℤ) : n % 18 = 11 → (n % 3 + n % 6 = 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3802_380210


namespace NUMINAMATH_CALUDE_exchange_result_l3802_380285

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate : ℝ := 85

/-- The amount of Canadian dollars to be exchanged -/
def cad_amount : ℝ := 5

/-- Theorem stating that exchanging 5 CAD results in 425 JPY -/
theorem exchange_result : cad_amount * exchange_rate = 425 := by
  sorry

end NUMINAMATH_CALUDE_exchange_result_l3802_380285


namespace NUMINAMATH_CALUDE_flower_bed_lilies_l3802_380212

/-- Given a flower bed with roses, tulips, and lilies, prove the number of lilies. -/
theorem flower_bed_lilies (roses tulips lilies : ℕ) : 
  roses = 57 → 
  tulips = 82 → 
  tulips = roses + lilies + 13 → 
  lilies = 12 := by
sorry


end NUMINAMATH_CALUDE_flower_bed_lilies_l3802_380212


namespace NUMINAMATH_CALUDE_only_lottery_is_event_l3802_380252

/-- Represents an experiment --/
inductive Experiment
  | TossCoin
  | Shoot
  | BoilWater
  | WinLottery

/-- Defines what constitutes an event --/
def is_event (e : Experiment) : Prop :=
  match e with
  | Experiment.WinLottery => True
  | _ => False

/-- Theorem stating that only winning the lottery constitutes an event --/
theorem only_lottery_is_event (e : Experiment) :
  is_event e ↔ e = Experiment.WinLottery :=
by sorry

end NUMINAMATH_CALUDE_only_lottery_is_event_l3802_380252


namespace NUMINAMATH_CALUDE_abs_value_inequality_l3802_380292

theorem abs_value_inequality (x : ℝ) : 2 ≤ |x - 3| ∧ |x - 3| ≤ 4 ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l3802_380292


namespace NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3802_380208

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ (2^x + a) / (2^x + b)

theorem odd_function_values_and_monotonicity_and_inequality
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = -1 ∧ b = 1) ∧
  (∀ x y, x < y → f a b x < f a b y) ∧
  (∀ x, f a b x + f a b (6 - x^2) ≤ 0 ↔ x ≤ -2 ∨ x ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_values_and_monotonicity_and_inequality_l3802_380208


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3802_380226

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem f_monotone_decreasing :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → f x1 > f x2 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3802_380226


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3802_380206

/-- If rowing against a stream takes twice as long as rowing with the stream for the same distance,
    then the ratio of the boat's speed in still water to the stream's speed is 3:1. -/
theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) : 
  boat_speed / stream_speed = 3 := by
sorry


end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3802_380206


namespace NUMINAMATH_CALUDE_tan_squared_to_sin_squared_l3802_380257

noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

theorem tan_squared_to_sin_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.sin t ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_squared_to_sin_squared_l3802_380257


namespace NUMINAMATH_CALUDE_line_equation_l3802_380294

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (m b x y : ℝ) : Prop := y = m * x + b

/-- Predicate to check if a point bisects a chord -/
def bisects_chord (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    on_line 8 (-15) x₁ y₁ ∧
    on_line 8 (-15) x₂ y₂ ∧
    bisects_chord point.1 point.2 x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3802_380294


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l3802_380276

theorem sum_of_x_solutions (x₁ x₂ : ℝ) (y : ℝ) (h1 : y = 5) (h2 : x₁^2 + y^2 = 169) (h3 : x₂^2 + y^2 = 169) (h4 : x₁ ≠ x₂) : x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l3802_380276


namespace NUMINAMATH_CALUDE_sandys_water_goal_l3802_380213

/-- Sandy's water drinking goal problem -/
theorem sandys_water_goal (water_per_interval : ℕ) (hours_per_interval : ℕ) (total_hours : ℕ) : 
  water_per_interval = 500 →
  hours_per_interval = 2 →
  total_hours = 12 →
  (water_per_interval * (total_hours / hours_per_interval)) / 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_water_goal_l3802_380213


namespace NUMINAMATH_CALUDE_complex_expression_equals_19_l3802_380255

-- Define lg as base 2 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem complex_expression_equals_19 :
  27 ^ (2/3) - 2 ^ (lg 3) * lg (1/8) + 2 * lg (Real.sqrt (3 + Real.sqrt 5) + Real.sqrt (3 - Real.sqrt 5)) = 19 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_19_l3802_380255


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l3802_380209

theorem exists_integer_divisible_by_15_with_sqrt_between_25_and_26 :
  ∃ n : ℕ+, 15 ∣ n ∧ (25 : ℝ) < Real.sqrt n ∧ Real.sqrt n < 26 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_25_and_26_l3802_380209


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3802_380291

/-- Given a quadratic function f(x) = ax^2 + bx + a satisfying certain conditions,
    prove its expression and range. -/
theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + a
  (∀ x, f (x + 7/4) = f (7/4 - x)) →
  (∃! x, f x = 7 * x + a) →
  (f = λ x ↦ -2 * x^2 + 7 * x - 2) ∧
  (Set.range f = Set.Iic (33/8)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3802_380291


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l3802_380219

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 3 / w + 3 / x = 3 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l3802_380219


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3802_380288

/-- Given polynomials P, Q, R, and S with integer coefficients satisfying the equation
    P(x^5) + x Q(x^5) + x^2 R(x^5) = (1 + x + x^2 + x^3 + x^4) S(x),
    prove that there exists a polynomial p such that P(x) = (x - 1) * p(x). -/
theorem polynomial_divisibility 
  (P Q R S : Polynomial ℤ) 
  (h : P.comp (X^5 : Polynomial ℤ) + X * Q.comp (X^5 : Polynomial ℤ) + X^2 * R.comp (X^5 : Polynomial ℤ) = 
       (1 + X + X^2 + X^3 + X^4) * S) : 
  ∃ p : Polynomial ℤ, P = (X - 1) * p := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3802_380288


namespace NUMINAMATH_CALUDE_area_of_triangle_PF1F2_l3802_380254

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the hyperbola C2
def C2 (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define a point P that satisfies both C1 and C2
def P : ℝ × ℝ := sorry

-- Assume P is on both C1 and C2
axiom P_on_C1 : C1 P.1 P.2
axiom P_on_C2 : C2 P.1 P.2

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_PF1F2 : 
  triangle_area P F1 F2 = Real.sqrt 2 := sorry

end NUMINAMATH_CALUDE_area_of_triangle_PF1F2_l3802_380254


namespace NUMINAMATH_CALUDE_division_multiplication_chain_l3802_380243

theorem division_multiplication_chain : (180 / 6) * 3 / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_chain_l3802_380243


namespace NUMINAMATH_CALUDE_devices_delivered_l3802_380241

/-- Represents the properties of the energy-saving devices delivery -/
structure DeviceDelivery where
  totalWeight : ℕ
  lightestThreeWeight : ℕ
  heaviestThreeWeight : ℕ
  allWeightsDifferent : Bool

/-- The number of devices in the delivery -/
def numDevices (d : DeviceDelivery) : ℕ := sorry

/-- Theorem stating that given the specific conditions, the number of devices is 10 -/
theorem devices_delivered (d : DeviceDelivery) 
  (h1 : d.totalWeight = 120)
  (h2 : d.lightestThreeWeight = 31)
  (h3 : d.heaviestThreeWeight = 41)
  (h4 : d.allWeightsDifferent = true) :
  numDevices d = 10 := by sorry

end NUMINAMATH_CALUDE_devices_delivered_l3802_380241


namespace NUMINAMATH_CALUDE_larger_integer_value_l3802_380260

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℝ) / (b : ℝ) = 7 / 3)
  (h_product : (a : ℕ) * b = 168) : 
  (a : ℝ) = 14 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3802_380260


namespace NUMINAMATH_CALUDE_equation_positive_roots_l3802_380234

theorem equation_positive_roots (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * |x| + |x + a| = 0) ↔ -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_roots_l3802_380234


namespace NUMINAMATH_CALUDE_B_is_smallest_l3802_380235

def A : ℕ := 32 + 7
def B : ℕ := 3 * 10 + 3
def C : ℕ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end NUMINAMATH_CALUDE_B_is_smallest_l3802_380235


namespace NUMINAMATH_CALUDE_cinnamon_blend_probability_l3802_380259

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem cinnamon_blend_probability : 
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_blend_probability_l3802_380259


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3802_380242

theorem arithmetic_calculations : 
  (26 - 7 + (-6) + 17 = 30) ∧ 
  (-81 / (9/4) * (-4/9) / (-16) = -1) ∧ 
  ((2/3 - 3/4 + 1/6) * (-36) = -3) ∧ 
  (-1^4 + 12 / (-2)^2 + 1/4 * (-8) = 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3802_380242


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l3802_380297

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l3802_380297


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l3802_380204

theorem fuel_cost_calculation (original_cost : ℝ) : 
  (2 * original_cost * 1.2 = 480) → original_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l3802_380204


namespace NUMINAMATH_CALUDE_cricket_target_runs_l3802_380274

/-- Calculates the target number of runs in a cricket game given specific conditions -/
theorem cricket_target_runs (total_overs run_rate_first_12 run_rate_remaining_38 : ℝ) 
  (h1 : total_overs = 50)
  (h2 : run_rate_first_12 = 4.5)
  (h3 : run_rate_remaining_38 = 8.052631578947368) : 
  ∃ (target : ℕ), target = 360 ∧ 
  target = ⌊run_rate_first_12 * 12 + run_rate_remaining_38 * (total_overs - 12)⌋ := by
  sorry

#check cricket_target_runs

end NUMINAMATH_CALUDE_cricket_target_runs_l3802_380274


namespace NUMINAMATH_CALUDE_problem_solution_l3802_380245

/-- Calculates the total earnings given investment ratios, percentage return ratios, and the difference between B's and A's earnings -/
def totalEarnings (investmentRatio : Fin 3 → ℕ) (returnRatio : Fin 3 → ℕ) (bMinusAEarnings : ℕ) : ℕ :=
  let earnings := λ i => investmentRatio i * returnRatio i
  let totalEarnings := (earnings 0) + (earnings 1) + (earnings 2)
  totalEarnings * (bMinusAEarnings / ((investmentRatio 1 * returnRatio 1) - (investmentRatio 0 * returnRatio 0)))

/-- The total earnings for the given problem -/
theorem problem_solution :
  totalEarnings
    (λ i => [3, 4, 5].get i)
    (λ i => [6, 5, 4].get i)
    250 = 7250 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3802_380245


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_3_5_7_l3802_380231

theorem least_four_digit_multiple_of_3_5_7 :
  (∀ n : ℕ, n ≥ 1000 ∧ n < 1050 → ¬(3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000 ∧ 3 ∣ 1050 ∧ 5 ∣ 1050 ∧ 7 ∣ 1050) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_3_5_7_l3802_380231


namespace NUMINAMATH_CALUDE_inequality_proof_l3802_380227

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2) 
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3802_380227


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l3802_380225

theorem points_on_line_procedure (x : ℕ) : ∃ x > 0, 9*x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l3802_380225


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3802_380263

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3802_380263


namespace NUMINAMATH_CALUDE_parrot_phrases_l3802_380280

def phrases_learned (days : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) : ℕ :=
  initial_phrases + (days / 7) * phrases_per_week

theorem parrot_phrases :
  phrases_learned 49 2 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_parrot_phrases_l3802_380280


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_equals_two_l3802_380290

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_implies_x_equals_two :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  parallel (a.1 + b.1, a.2 + b.2) (4 * b.1 - 2 * a.1, 4 * b.2 - 2 * a.2) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_equals_two_l3802_380290


namespace NUMINAMATH_CALUDE_intersection_A_B_l3802_380216

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | x * (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3802_380216


namespace NUMINAMATH_CALUDE_max_mn_and_min_4m2_plus_n2_l3802_380286

theorem max_mn_and_min_4m2_plus_n2 (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → m * n ≥ x * y) ∧
  (m * n = 1/8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  (4 * m^2 + n^2 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_mn_and_min_4m2_plus_n2_l3802_380286


namespace NUMINAMATH_CALUDE_janes_cans_l3802_380232

theorem janes_cans (total_seeds : ℕ) (seeds_per_can : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end NUMINAMATH_CALUDE_janes_cans_l3802_380232


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3802_380293

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 4 - a 2 = -2)
  (h_seventh : a 7 = -3) :
  a 9 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3802_380293


namespace NUMINAMATH_CALUDE_four_valid_m_l3802_380256

/-- The number of positive integers m for which 2310 / (m^2 - 4) is a positive integer -/
def count_valid_m : ℕ := 4

/-- Predicate to check if 2310 / (m^2 - 4) is a positive integer -/
def is_valid (m : ℕ) : Prop :=
  m > 0 ∧ ∃ k : ℕ+, k * (m^2 - 4) = 2310

/-- Theorem stating that there are exactly 4 positive integers m satisfying the condition -/
theorem four_valid_m :
  (∃! (s : Finset ℕ), s.card = count_valid_m ∧ ∀ m, m ∈ s ↔ is_valid m) :=
sorry

end NUMINAMATH_CALUDE_four_valid_m_l3802_380256


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l3802_380205

theorem abc_sum_sqrt (a b c : ℝ) 
  (eq1 : b + c = 17) 
  (eq2 : c + a = 18) 
  (eq3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 60 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l3802_380205


namespace NUMINAMATH_CALUDE_average_weight_increase_l3802_380278

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)  -- number of people in the group
  (w_old : ℝ)  -- weight of the person being replaced
  (w_new : ℝ)  -- weight of the new person
  (h_n : n = 8)  -- given that there are 8 people
  (h_w_old : w_old = 67)  -- given that the old person weighs 67 kg
  (h_w_new : w_new = 87)  -- given that the new person weighs 87 kg
  : (w_new - w_old) / n = 2.5 := by
sorry


end NUMINAMATH_CALUDE_average_weight_increase_l3802_380278


namespace NUMINAMATH_CALUDE_angle_X_measure_l3802_380249

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)
  (sum_angles : X + Y + Z = 180)
  (all_positive : 0 < X ∧ 0 < Y ∧ 0 < Z)

-- State the theorem
theorem angle_X_measure (t : Triangle) 
  (h1 : t.Z = 3 * t.Y) 
  (h2 : t.Y = 15) : 
  t.X = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_X_measure_l3802_380249


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l3802_380289

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 2

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) (-1) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) (-1) → f y ≥ f x) ∧
  f x = -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l3802_380289


namespace NUMINAMATH_CALUDE_light_reflection_theorem_l3802_380202

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Reflects a point across a line -/
def reflectPoint (p : Point) (l : Line) : Point :=
  sorry

/-- Constructs a line passing through two points -/
def lineThrough (p1 p2 : Point) : Line :=
  sorry

theorem light_reflection_theorem (P A : Point) (mirror : Line) :
  P = Point.mk 2 3 →
  A = Point.mk 1 1 →
  mirror = Line.mk 1 1 1 →
  let Q := reflectPoint P mirror
  let incidentRay := lineThrough P (Point.mk mirror.a mirror.b)
  let reflectedRay := lineThrough Q A
  incidentRay = Line.mk 2 (-1) (-1) ∧
  reflectedRay = Line.mk 4 (-5) 1 :=
sorry

end NUMINAMATH_CALUDE_light_reflection_theorem_l3802_380202


namespace NUMINAMATH_CALUDE_pennys_bakery_revenue_l3802_380223

/-- Represents the price and quantity of a type of cheesecake -/
structure Cheesecake where
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the total revenue from a type of cheesecake -/
def revenue (c : Cheesecake) (slices_per_pie : ℕ) : ℕ :=
  c.price_per_slice * c.pies_sold * slices_per_pie

/-- The main theorem about Penny's bakery revenue -/
theorem pennys_bakery_revenue : 
  let slices_per_pie : ℕ := 6
  let blueberry : Cheesecake := { price_per_slice := 7, pies_sold := 7 }
  let strawberry : Cheesecake := { price_per_slice := 8, pies_sold := 5 }
  let chocolate : Cheesecake := { price_per_slice := 9, pies_sold := 3 }
  revenue blueberry slices_per_pie + revenue strawberry slices_per_pie + revenue chocolate slices_per_pie = 696 := by
  sorry


end NUMINAMATH_CALUDE_pennys_bakery_revenue_l3802_380223


namespace NUMINAMATH_CALUDE_room_population_problem_l3802_380218

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  initial_men + 2 = 14 →  -- Final number of men is 14
  (2 * (initial_women - 3) = 24) →  -- Final number of women is 24
  True :=
by sorry

end NUMINAMATH_CALUDE_room_population_problem_l3802_380218


namespace NUMINAMATH_CALUDE_triangle_ratio_equals_two_l3802_380281

/-- In triangle ABC, if angle A is 60 degrees and side a is √3, 
    then (a + b) / (sin A + sin B) = 2 -/
theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  A = π / 3 ∧ 
  a = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  (a + b) / (Real.sin A + Real.sin B) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equals_two_l3802_380281


namespace NUMINAMATH_CALUDE_power_multiplication_l3802_380296

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3802_380296


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3802_380239

/-- The point of intersection between a line and a plane in 3D space. -/
theorem line_plane_intersection
  (A1 A2 A3 A4 : ℝ × ℝ × ℝ)
  (h1 : A1 = (1, 2, -3))
  (h2 : A2 = (1, 0, 1))
  (h3 : A3 = (-2, -1, 6))
  (h4 : A4 = (0, -5, -4)) :
  ∃ P : ℝ × ℝ × ℝ,
    (∃ t : ℝ, P = A4 + t • (A4 - A1)) ∧
    (∃ u v : ℝ, P = A1 + u • (A2 - A1) + v • (A3 - A1)) :=
by sorry


end NUMINAMATH_CALUDE_line_plane_intersection_l3802_380239


namespace NUMINAMATH_CALUDE_difference_of_squares_l3802_380266

theorem difference_of_squares (x y : ℚ) 
  (h1 : x + y = 15/26) 
  (h2 : x - y = 2/65) : 
  x^2 - y^2 = 15/845 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3802_380266


namespace NUMINAMATH_CALUDE_length_of_AB_l3802_380265

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x^2 + y^2 - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 2 * y + 9 = 0

-- Define the point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the condition that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop :=
  ∃ (center_x center_y : ℝ), line_l k center_x center_y ∧
    ∀ (x y : ℝ), circle_C x y ↔ circle_C (2 * center_x - x) (2 * center_y - y)

-- Define the tangency condition
def is_tangent (k : ℝ) (B : ℝ × ℝ) : Prop :=
  circle_C B.1 B.2 ∧
  ∃ (t : ℝ), B = (t, k * t + 2) ∧
    ∀ (x y : ℝ), line_l k x y → (circle_C x y → x = B.1 ∧ y = B.2)

-- State the theorem
theorem length_of_AB (k : ℝ) (B : ℝ × ℝ) :
  is_axis_of_symmetry k →
  is_tangent k B →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l3802_380265


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l3802_380237

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l3802_380237


namespace NUMINAMATH_CALUDE_solution_set_equality_l3802_380271

/-- The solution set of the inequality (x^2 - 2x - 3)(x^2 - 4x + 4) < 0 -/
def SolutionSet : Set ℝ :=
  {x | (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0}

/-- The set {x | -1 < x < 3 and x ≠ 2} -/
def TargetSet : Set ℝ :=
  {x | -1 < x ∧ x < 3 ∧ x ≠ 2}

theorem solution_set_equality : SolutionSet = TargetSet := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3802_380271


namespace NUMINAMATH_CALUDE_expression_equals_185_l3802_380222

theorem expression_equals_185 : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_185_l3802_380222


namespace NUMINAMATH_CALUDE_goldbach_138_max_diff_l3802_380214

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_max_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → s - r ≤ 124 :=
sorry

end NUMINAMATH_CALUDE_goldbach_138_max_diff_l3802_380214


namespace NUMINAMATH_CALUDE_efficiency_increase_sakshi_to_tanya_l3802_380253

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Sakshi's work rate in parts per day -/
def sakshi_rate : ℚ := 1 / 25

/-- Tanya's work rate in parts per day -/
def tanya_rate : ℚ := 1 / 20

theorem efficiency_increase_sakshi_to_tanya :
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_efficiency_increase_sakshi_to_tanya_l3802_380253


namespace NUMINAMATH_CALUDE_seven_distinct_reverse_numbers_l3802_380220

def is_reverse_after_adding_18 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ n + 18 = (n % 10) * 10 + (n / 10)

theorem seven_distinct_reverse_numbers :
  ∃ (S : Finset ℕ), S.card = 7 ∧ ∀ n ∈ S, is_reverse_after_adding_18 n ∧
    ∀ m ∈ S, m ≠ n → m ≠ n := by
  sorry

end NUMINAMATH_CALUDE_seven_distinct_reverse_numbers_l3802_380220


namespace NUMINAMATH_CALUDE_sugar_water_concentration_l3802_380238

theorem sugar_water_concentration (a : ℝ) : 
  (100 * 0.4 + a * 0.2) / (100 + a) = 0.25 → a = 300 := by
  sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_l3802_380238


namespace NUMINAMATH_CALUDE_nonagon_prism_edges_l3802_380272

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Nat  -- number of sides in the base shape

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : Nat :=
  3 * p.base

theorem nonagon_prism_edges :
  ∀ p : Prism, p.base = 9 → p.edges = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_prism_edges_l3802_380272


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3802_380201

/-- 
Given a boat traveling downstream with the following conditions:
1. The rate of the stream is 5 km/hr
2. The boat takes 3 hours to cover a distance of 63 km downstream

This theorem proves that the speed of the boat in still water is 16 km/hr.
-/
theorem boat_speed_in_still_water : 
  ∀ (stream_rate : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ),
  stream_rate = 5 →
  downstream_time = 3 →
  downstream_distance = 63 →
  ∃ (still_water_speed : ℝ),
    still_water_speed = 16 ∧
    downstream_distance = (still_water_speed + stream_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3802_380201


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l3802_380228

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The total number of books Bryan has -/
def total_books : ℕ := 621

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := total_books / books_per_shelf

theorem bryan_bookshelves : num_bookshelves = 23 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l3802_380228


namespace NUMINAMATH_CALUDE_tree_height_when_boy_is_36_inches_l3802_380279

/-- Calculates the final height of a tree given initial heights and growth rates -/
def final_tree_height (initial_tree_height : ℝ) (initial_boy_height : ℝ) (final_boy_height : ℝ) : ℝ :=
  initial_tree_height + 2 * (final_boy_height - initial_boy_height)

/-- Proves that the tree will be 40 inches tall when the boy is 36 inches tall -/
theorem tree_height_when_boy_is_36_inches :
  final_tree_height 16 24 36 = 40 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_when_boy_is_36_inches_l3802_380279


namespace NUMINAMATH_CALUDE_tylenol_dosage_l3802_380262

/-- Represents the dosage schedule and total amount of medication taken -/
structure DosageInfo where
  interval : ℕ  -- Time interval between doses in hours
  duration : ℕ  -- Total duration of medication in hours
  tablets_per_dose : ℕ  -- Number of tablets taken per dose
  total_grams : ℕ  -- Total amount of medication taken in grams

/-- Calculates the milligrams per tablet given dosage information -/
def milligrams_per_tablet (info : DosageInfo) : ℕ :=
  let total_milligrams := info.total_grams * 1000
  let num_doses := info.duration / info.interval
  let milligrams_per_dose := total_milligrams / num_doses
  milligrams_per_dose / info.tablets_per_dose

/-- Theorem stating that under the given conditions, each tablet contains 500 milligrams -/
theorem tylenol_dosage (info : DosageInfo) 
  (h1 : info.interval = 4)
  (h2 : info.duration = 12)
  (h3 : info.tablets_per_dose = 2)
  (h4 : info.total_grams = 3) :
  milligrams_per_tablet info = 500 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_dosage_l3802_380262


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_l3802_380207

/-- Represents a chocolate bar with grooves -/
structure ChocolateBar where
  longitudinal_grooves : Nat
  transverse_grooves : Nat

/-- Calculates the minimum number of breaks required to separate the chocolate bar into pieces with no grooves -/
def min_breaks (bar : ChocolateBar) : Nat :=
  4

/-- Theorem stating that a chocolate bar with 2 longitudinal grooves and 3 transverse grooves requires 4 breaks -/
theorem chocolate_bar_breaks (bar : ChocolateBar) 
  (h1 : bar.longitudinal_grooves = 2) 
  (h2 : bar.transverse_grooves = 3) : 
  min_breaks bar = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_breaks_l3802_380207


namespace NUMINAMATH_CALUDE_stream_speed_l3802_380284

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 216 km downstream in 8 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) :
  boat_speed = 22 →
  distance = 216 →
  time = 8 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3802_380284


namespace NUMINAMATH_CALUDE_roots_of_f_eq_x_none_or_infinite_l3802_380268

theorem roots_of_f_eq_x_none_or_infinite (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = f x + 1) :
  (∀ x : ℝ, f x ≠ x) ∨ (∃ S : Set ℝ, Set.Infinite S ∧ ∀ x ∈ S, f x = x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_f_eq_x_none_or_infinite_l3802_380268


namespace NUMINAMATH_CALUDE_art_fair_sales_l3802_380211

theorem art_fair_sales (total_visitors : ℕ) (two_painting_buyers : ℕ) (one_painting_buyers : ℕ) (total_paintings_sold : ℕ) :
  total_visitors = 20 →
  two_painting_buyers = 4 →
  one_painting_buyers = 12 →
  total_paintings_sold = 36 →
  ∃ (four_painting_buyers : ℕ),
    four_painting_buyers * 4 + two_painting_buyers * 2 + one_painting_buyers = total_paintings_sold ∧
    four_painting_buyers + two_painting_buyers + one_painting_buyers ≤ total_visitors ∧
    four_painting_buyers = 4 :=
by sorry

end NUMINAMATH_CALUDE_art_fair_sales_l3802_380211


namespace NUMINAMATH_CALUDE_addition_problem_l3802_380240

theorem addition_problem (x y : ℕ) :
  (x + y = x + 2000) ∧ (x + y = y + 6) →
  (x = 6 ∧ y = 2000 ∧ x + y = 2006) :=
by sorry

end NUMINAMATH_CALUDE_addition_problem_l3802_380240


namespace NUMINAMATH_CALUDE_larger_number_problem_l3802_380299

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3802_380299


namespace NUMINAMATH_CALUDE_factorial_ratio_l3802_380282

theorem factorial_ratio (n : ℕ) (h : n > 0) : (Nat.factorial n) / (Nat.factorial (n - 1)) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3802_380282


namespace NUMINAMATH_CALUDE_no_points_above_diagonal_l3802_380277

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ + t₂ ≤ 1 ∧
    p = (4 * t₁ + 4 * t₂, 10 * t₂)}

-- Theorem statement
theorem no_points_above_diagonal (a b : ℝ) :
  (a, b) ∈ triangle → a - b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_no_points_above_diagonal_l3802_380277


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l3802_380269

theorem initial_number_of_girls (initial_boys : ℕ) (boys_dropout : ℕ) (girls_dropout : ℕ) (remaining_students : ℕ) : 
  initial_boys = 14 →
  boys_dropout = 4 →
  girls_dropout = 3 →
  remaining_students = 17 →
  initial_boys - boys_dropout + (initial_girls - girls_dropout) = remaining_students →
  initial_girls = 10 :=
by
  sorry

#check initial_number_of_girls

end NUMINAMATH_CALUDE_initial_number_of_girls_l3802_380269


namespace NUMINAMATH_CALUDE_z_power_sum_l3802_380295

theorem z_power_sum (z : ℂ) (h : z = (Real.sqrt 2) / (1 - Complex.I)) : 
  z^100 + z^50 + 1 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_power_sum_l3802_380295


namespace NUMINAMATH_CALUDE_power_difference_squared_l3802_380264

theorem power_difference_squared (n : ℕ) :
  (5^(1001 : ℕ) + 6^(1002 : ℕ))^2 - (5^(1001 : ℕ) - 6^(1002 : ℕ))^2 = 24 * 30^(1001 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_squared_l3802_380264


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3802_380217

theorem trigonometric_identity (a b c : Real) :
  (Real.sin (a - b)) / (Real.sin a * Real.sin b) +
  (Real.sin (b - c)) / (Real.sin b * Real.sin c) +
  (Real.sin (c - a)) / (Real.sin c * Real.sin a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3802_380217


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l3802_380230

/-- The number of chocolate bars in a large box -/
def chocolate_bars_in_large_box (small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  small_boxes * bars_per_small_box

/-- Theorem: There are 640 chocolate bars in the large box -/
theorem total_chocolate_bars :
  chocolate_bars_in_large_box 20 32 = 640 := by
sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l3802_380230


namespace NUMINAMATH_CALUDE_expression_value_l3802_380275

theorem expression_value :
  let a : ℤ := 10
  let b : ℤ := 15
  let c : ℤ := 3
  let d : ℤ := 2
  (a * (b - c)) - ((a - b) * c) + d = 137 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3802_380275


namespace NUMINAMATH_CALUDE_valid_outfits_count_l3802_380224

/-- The number of shirts available -/
def num_shirts : ℕ := 7

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for shirts and hats -/
def num_colors : ℕ := 7

/-- Calculate the number of valid outfits -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_colors * num_pants

theorem valid_outfits_count :
  num_valid_outfits = 210 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l3802_380224


namespace NUMINAMATH_CALUDE_coefficient_sum_equality_l3802_380233

theorem coefficient_sum_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_equality_l3802_380233


namespace NUMINAMATH_CALUDE_second_wing_floors_is_seven_l3802_380273

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_halls_per_floor : ℕ
  wing2_rooms_per_hall : ℕ

/-- Calculate the number of floors in the second wing -/
def second_wing_floors (h : Hotel) : ℕ :=
  let wing1_rooms := h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall
  let wing2_rooms := h.total_rooms - wing1_rooms
  let rooms_per_floor_wing2 := h.wing2_halls_per_floor * h.wing2_rooms_per_hall
  wing2_rooms / rooms_per_floor_wing2

/-- The theorem stating that the number of floors in the second wing is 7 -/
theorem second_wing_floors_is_seven (h : Hotel) 
    (h_total : h.total_rooms = 4248)
    (h_wing1_floors : h.wing1_floors = 9)
    (h_wing1_halls : h.wing1_halls_per_floor = 6)
    (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
    (h_wing2_halls : h.wing2_halls_per_floor = 9)
    (h_wing2_rooms : h.wing2_rooms_per_hall = 40) : 
  second_wing_floors h = 7 := by
  sorry

#eval second_wing_floors {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_halls_per_floor := 9,
  wing2_rooms_per_hall := 40
}

end NUMINAMATH_CALUDE_second_wing_floors_is_seven_l3802_380273


namespace NUMINAMATH_CALUDE_remainder_double_n_l3802_380203

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l3802_380203


namespace NUMINAMATH_CALUDE_some_number_value_l3802_380246

theorem some_number_value (n m : ℚ) : 
  n = 40 → (n / 20) * (n / m) = 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3802_380246


namespace NUMINAMATH_CALUDE_soccer_team_biology_count_l3802_380229

theorem soccer_team_biology_count :
  ∀ (total_players physics_count chemistry_count all_three_count physics_and_chemistry_count : ℕ),
    total_players = 15 →
    physics_count = 8 →
    chemistry_count = 6 →
    all_three_count = 3 →
    physics_and_chemistry_count = 4 →
    ∃ (biology_count : ℕ),
      biology_count = 9 ∧
      biology_count = total_players - (physics_count - physics_and_chemistry_count) - (chemistry_count - physics_and_chemistry_count) :=
by
  sorry

#check soccer_team_biology_count

end NUMINAMATH_CALUDE_soccer_team_biology_count_l3802_380229


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3802_380247

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) :
  area = 162 →
  (∀ base height, altitude_base_relation base height → height = 2 * base) →
  ∃ base : ℝ, altitude_base_relation base (2 * base) ∧ 
    area = base * (2 * base) ∧ 
    base = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3802_380247


namespace NUMINAMATH_CALUDE_ellipse_and_circle_problem_l3802_380221

theorem ellipse_and_circle_problem 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2^2 = a^2 - b^2) -- condition for right focus at (2,0)
  : 
  (∀ x y : ℝ, x^2/6 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ 
  (∃ m : ℝ, ∃ c : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ c ↔ (p.1^2 + (p.2 - 1/3)^2 = (1/3)^2)) ∧
    (∃ p1 p2 p3 p4 : ℝ × ℝ, 
      p1 ∈ c ∧ p2 ∈ c ∧ p3 ∈ c ∧ p4 ∈ c ∧
      p1.2 = p1.1^2 + m ∧ p2.2 = p2.1^2 + m ∧ p3.2 = p3.1^2 + m ∧ p4.2 = p4.1^2 + m ∧
      p1.1^2/6 + p1.2^2/2 = 1 ∧ p2.1^2/6 + p2.2^2/2 = 1 ∧ p3.1^2/6 + p3.2^2/2 = 1 ∧ p4.1^2/6 + p4.2^2/2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_problem_l3802_380221


namespace NUMINAMATH_CALUDE_unique_single_digit_cube_equation_l3802_380283

theorem unique_single_digit_cube_equation :
  ∃! (A : ℕ), A ∈ Finset.range 10 ∧ A ≠ 0 ∧ A^3 = 210 + A :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_single_digit_cube_equation_l3802_380283


namespace NUMINAMATH_CALUDE_set_A_equals_neg_one_zero_l3802_380244

def A : Set ℤ := {x | x^2 + x ≤ 0}

theorem set_A_equals_neg_one_zero : A = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_neg_one_zero_l3802_380244


namespace NUMINAMATH_CALUDE_polynomial_sum_l3802_380236

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) :
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3802_380236


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l3802_380251

theorem complex_fraction_problem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2)
  (h3 : x + y ≠ 0)
  (h4 : x^3 + y^3 ≠ 0)
  (h5 : x^5 + y^5 ≠ 0) :
  (x^6 + y^6) / (x^5 + y^5) = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l3802_380251


namespace NUMINAMATH_CALUDE_f_4_1981_l3802_380267

/-- A function satisfying the given recursive properties -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Power tower of 2 with given height -/
def power_tower_2 : ℕ → ℕ
| 0 => 1
| n + 1 => 2^(power_tower_2 n)

/-- The main theorem to prove -/
theorem f_4_1981 : f 4 1981 = power_tower_2 1984 - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l3802_380267


namespace NUMINAMATH_CALUDE_stair_climbing_time_l3802_380200

theorem stair_climbing_time : 
  let n : ℕ := 4  -- number of flights
  let a : ℕ := 30 -- time for first flight
  let d : ℕ := 10 -- time increase for each subsequent flight
  let S := n * (2 * a + (n - 1) * d) / 2  -- sum formula for arithmetic sequence
  S = 180 := by sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l3802_380200


namespace NUMINAMATH_CALUDE_quadratic_common_root_theorem_l3802_380215

theorem quadratic_common_root_theorem (a b : ℕ+) :
  (∃ x : ℝ, (a - 1 : ℝ) * x^2 - (a^2 + 2 : ℝ) * x + (a^2 + 2*a : ℝ) = 0 ∧
             (b - 1 : ℝ) * x^2 - (b^2 + 2 : ℝ) * x + (b^2 + 2*b : ℝ) = 0) →
  (a^(b : ℕ) + b^(a : ℕ) : ℝ) / (a^(-(b : ℤ)) + b^(-(a : ℤ)) : ℝ) = 256 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_common_root_theorem_l3802_380215


namespace NUMINAMATH_CALUDE_point_rotation_on_circle_l3802_380248

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

def rotation_45_ccw (x y x' y' : ℝ) : Prop :=
  x' = x * (Real.sqrt 2 / 2) - y * (Real.sqrt 2 / 2) ∧
  y' = x * (Real.sqrt 2 / 2) + y * (Real.sqrt 2 / 2)

theorem point_rotation_on_circle :
  ∀ (x' y' : ℝ),
    circle_equation 3 4 →
    rotation_45_ccw 3 4 x' y' →
    x' = -(Real.sqrt 2 / 2) ∧ y' = 7 * (Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_point_rotation_on_circle_l3802_380248


namespace NUMINAMATH_CALUDE_tan_alpha_two_l3802_380270

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 * Real.cos α - Real.cos α ^ 2 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_l3802_380270


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3802_380250

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3802_380250


namespace NUMINAMATH_CALUDE_right_triangle_proof_l3802_380261

theorem right_triangle_proof (n : ℝ) (hn : n > 0) :
  let a := 2*n^2 + 2*n + 1
  let b := 2*n^2 + 2*n
  let c := 2*n + 1
  a^2 = b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_proof_l3802_380261


namespace NUMINAMATH_CALUDE_complement_intersection_MN_l3802_380258

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem complement_intersection_MN :
  (M ∩ N)ᶜ = {1, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_MN_l3802_380258


namespace NUMINAMATH_CALUDE_legos_lost_l3802_380287

def initial_legos : ℕ := 380
def given_to_sister : ℕ := 24
def current_legos : ℕ := 299

theorem legos_lost : initial_legos - given_to_sister - current_legos = 57 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l3802_380287
