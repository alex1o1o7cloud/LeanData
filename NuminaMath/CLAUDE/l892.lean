import Mathlib

namespace sum_product_theorem_l892_89224

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = -4)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 485 / 9 := by
sorry

end sum_product_theorem_l892_89224


namespace equation_positive_root_implies_m_eq_neg_one_l892_89286

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 1) - m / (1 - x) = 2

-- State the theorem
theorem equation_positive_root_implies_m_eq_neg_one :
  ∃ (x : ℝ), x > 0 ∧ equation x m → m = -1 :=
sorry

end equation_positive_root_implies_m_eq_neg_one_l892_89286


namespace union_of_A_and_B_intersection_empty_iff_l892_89281

def A (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 5}

theorem union_of_A_and_B (m : ℝ) :
  m = -3 → A m ∪ B = {x | -7 < x ∧ x ≤ 5} := by sorry

theorem intersection_empty_iff (m : ℝ) :
  A m ∩ B = ∅ ↔ m ≤ -4 ∨ 1 ≤ m := by sorry

end union_of_A_and_B_intersection_empty_iff_l892_89281


namespace sum_of_b_values_l892_89204

theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 9 * x^2 + b₁ * x + 15 * x + 16 = 0) ∧
  (∃! x, 9 * x^2 + b₂ * x + 15 * x + 16 = 0) →
  b₁ + b₂ = -30 := by
sorry

end sum_of_b_values_l892_89204


namespace car_fuel_efficiency_l892_89200

def miles_to_school : ℝ := 15
def miles_to_softball : ℝ := 6
def miles_to_restaurant : ℝ := 2
def miles_to_friend : ℝ := 4
def miles_to_home : ℝ := 11
def initial_gas : ℝ := 2

def total_miles : ℝ := miles_to_school + miles_to_softball + miles_to_restaurant + miles_to_friend + miles_to_home

theorem car_fuel_efficiency :
  total_miles / initial_gas = 19 := by sorry

end car_fuel_efficiency_l892_89200


namespace trigonometric_identity_l892_89211

theorem trigonometric_identity (α β γ : ℝ) :
  3.400 * Real.cos (α + β) * Real.cos γ + Real.cos α + Real.cos β + Real.cos γ - Real.sin (α + β) * Real.sin γ =
  4 * Real.cos ((α + β) / 2) * Real.cos ((α + γ) / 2) * Real.cos ((β + γ) / 2) := by
  sorry

end trigonometric_identity_l892_89211


namespace solve_for_x_l892_89276

theorem solve_for_x (y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 := by
  sorry

end solve_for_x_l892_89276


namespace remainder_444_power_222_mod_13_l892_89277

theorem remainder_444_power_222_mod_13 : 444^222 ≡ 1 [ZMOD 13] := by
  sorry

end remainder_444_power_222_mod_13_l892_89277


namespace only_one_divides_power_plus_one_l892_89202

theorem only_one_divides_power_plus_one :
  ∀ n : ℕ+, n.val % 2 = 1 ∧ (n.val ∣ 3^n.val + 1) → n = 1 := by sorry

end only_one_divides_power_plus_one_l892_89202


namespace quadratic_inequalities_l892_89206

theorem quadratic_inequalities (x : ℝ) :
  (((1/2 : ℝ) * x^2 - 4*x + 6 < 0) ↔ (2 < x ∧ x < 6)) ∧
  ((4*x^2 - 4*x + 1 ≥ 0) ↔ True) ∧
  ((2*x^2 - x - 1 ≤ 0) ↔ (-1/2 ≤ x ∧ x ≤ 1)) ∧
  ((3*(x-2)*(x+2) - 4*(x+1)^2 + 1 < 0) ↔ (x < -5 ∨ x > -3)) :=
by sorry

end quadratic_inequalities_l892_89206


namespace interval_length_implies_difference_l892_89265

theorem interval_length_implies_difference (c d : ℝ) :
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) →
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d ↔ (c + 2) / 3 ≤ x ∧ x ≤ (d + 2) / 3) →
  ((d + 2) / 3 - (c + 2) / 3 = 15) →
  d - c = 45 := by
  sorry

end interval_length_implies_difference_l892_89265


namespace james_fish_purchase_l892_89264

theorem james_fish_purchase (fish_per_roll : ℕ) (bad_fish_percent : ℚ) (rolls_made : ℕ) :
  fish_per_roll = 40 →
  bad_fish_percent = 1/5 →
  rolls_made = 8 →
  ∃ (total_fish : ℕ), total_fish = 400 ∧ 
    (total_fish : ℚ) * (1 - bad_fish_percent) = (fish_per_roll * rolls_made : ℚ) :=
by sorry

end james_fish_purchase_l892_89264


namespace inverse_variation_result_l892_89221

/-- A function representing the inverse variation of 7y with the cube of x -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 7 * y = k / (x ^ 3)

/-- The theorem stating that given the inverse variation and initial condition,
    when x = 4, y = 1 -/
theorem inverse_variation_result :
  (∃ y₀ : ℝ, inverse_variation 2 y₀ ∧ y₀ = 8) →
  (∃ y : ℝ, inverse_variation 4 y ∧ y = 1) :=
by sorry

end inverse_variation_result_l892_89221


namespace ratio_of_sums_l892_89291

/-- An arithmetic sequence with common difference d, first term 8d, and sum of first n terms S_n -/
structure ArithmeticSequence (d : ℝ) where
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 8 * d
  h4 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The ratio of 7S_5 to 5S_7 is 10/11 for the given arithmetic sequence -/
theorem ratio_of_sums (d : ℝ) (seq : ArithmeticSequence d) :
  7 * seq.S 5 / (5 * seq.S 7) = 10 / 11 :=
sorry

end ratio_of_sums_l892_89291


namespace quadratic_root_fraction_l892_89209

theorem quadratic_root_fraction (a b : ℝ) (h1 : a ≠ b) (h2 : a + b - 20 = 0) :
  (a^2 - b^2) / (2*a - 2*b) = 10 := by
sorry

end quadratic_root_fraction_l892_89209


namespace initial_blocks_l892_89269

theorem initial_blocks (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 9 → total = 95 → initial + added = total → initial = 86 := by
  sorry

end initial_blocks_l892_89269


namespace prime_pairs_dividing_sum_of_powers_l892_89223

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q ∣ 2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end prime_pairs_dividing_sum_of_powers_l892_89223


namespace banana_weights_l892_89292

/-- A scale with a constant displacement --/
structure DisplacedScale where
  displacement : ℝ

/-- Measurements of banana bunches on a displaced scale --/
structure BananaMeasurements where
  small_bunch : ℝ
  large_bunch : ℝ
  combined_bunches : ℝ

/-- The actual weights of the banana bunches --/
def actual_weights (s : DisplacedScale) (m : BananaMeasurements) : Prop :=
  ∃ (small large : ℝ),
    small = m.small_bunch - s.displacement ∧
    large = m.large_bunch - s.displacement ∧
    small + large = m.combined_bunches - s.displacement ∧
    small = 1 ∧ large = 2

/-- Theorem stating that given the measurements, the actual weights are 1 kg and 2 kg --/
theorem banana_weights (s : DisplacedScale) (m : BananaMeasurements) 
  (h1 : m.small_bunch = 1.5)
  (h2 : m.large_bunch = 2.5)
  (h3 : m.combined_bunches = 3.5) :
  actual_weights s m :=
by sorry

end banana_weights_l892_89292


namespace correct_average_after_error_correction_l892_89298

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℝ) 
  (wrong_mark correct_mark : ℝ) :
  n = 25 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (n * initial_average - wrong_mark + correct_mark) / n = 98 := by
sorry

end correct_average_after_error_correction_l892_89298


namespace parabola_inequality_l892_89251

def f (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_inequality : f (-1) < f 4 ∧ f 4 < f 1 := by
  sorry

end parabola_inequality_l892_89251


namespace right_triangle_3_4_5_l892_89263

theorem right_triangle_3_4_5 (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_3_4_5

end right_triangle_3_4_5_l892_89263


namespace arithmetic_sequence_fifth_term_l892_89233

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 1)
  (h_8 : a 8 = 2 * a 6 + a 4) :
  a 5 = -1/2 := by
  sorry

end arithmetic_sequence_fifth_term_l892_89233


namespace games_missed_l892_89278

/-- Given a total number of soccer games and the number of games Jessica attended,
    calculate the number of games Jessica missed. -/
theorem games_missed (total_games attended_games : ℕ) : 
  total_games = 6 → attended_games = 2 → total_games - attended_games = 4 := by
  sorry

end games_missed_l892_89278


namespace binary_rep_of_31_l892_89212

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 31 is [true, true, true, true, true] -/
theorem binary_rep_of_31 : toBinary 31 = [true, true, true, true, true] := by
  sorry

end binary_rep_of_31_l892_89212


namespace angle_equality_l892_89229

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end angle_equality_l892_89229


namespace gcd_lcm_identity_l892_89253

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end gcd_lcm_identity_l892_89253


namespace meet_time_l892_89293

/-- Represents the scenario of Petya and Vasya's journey --/
structure Journey where
  distance : ℝ  -- Total distance between Petya and Vasya
  speed_dirt : ℝ  -- Speed on dirt road
  speed_paved : ℝ  -- Speed on paved road
  time_to_bridge : ℝ  -- Time for Petya to reach the bridge

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.speed_paved = 3 * j.speed_dirt ∧
  j.time_to_bridge = 1 ∧
  j.distance / 2 = j.speed_paved * j.time_to_bridge

/-- The theorem to be proved --/
theorem meet_time (j : Journey) (h : journey_conditions j) : 
  ∃ (t : ℝ), t = 2 ∧ t = j.time_to_bridge + (j.distance / 2 - j.speed_dirt * j.time_to_bridge) / (2 * j.speed_dirt) :=
sorry

end meet_time_l892_89293


namespace line_not_in_third_quadrant_l892_89252

/-- The line x + y - 1 = 0 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ∀ x y : ℝ, x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) := by
  sorry

end line_not_in_third_quadrant_l892_89252


namespace alice_burger_spending_l892_89216

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of burgers Alice purchases each day -/
def burgers_per_day : ℕ := 4

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June -/
def total_spent : ℕ := june_days * burgers_per_day * burger_cost

theorem alice_burger_spending :
  total_spent = 1560 := by
  sorry

end alice_burger_spending_l892_89216


namespace readers_of_both_genres_l892_89207

theorem readers_of_both_genres (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h_total : total = 150)
  (h_sci_fi : sci_fi = 120)
  (h_literary : literary = 90) :
  sci_fi + literary - total = 60 := by
  sorry

end readers_of_both_genres_l892_89207


namespace max_triangle_sum_l892_89243

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9, 10}

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
  d ∈ triangle_numbers ∧ e ∈ triangle_numbers ∧ f ∈ triangle_numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def side_sum (a b c : ℕ) : ℕ := a + b + c

def equal_sums (a b c d e f : ℕ) : Prop :=
  side_sum a b c = side_sum c d e ∧
  side_sum c d e = side_sum e f a

theorem max_triangle_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    equal_sums a b c d e f →
    side_sum a b c ≤ 24 :=
sorry

end max_triangle_sum_l892_89243


namespace derivatives_verification_l892_89271

theorem derivatives_verification :
  (∀ x : ℝ, deriv (λ x => x^2) x = 2 * x) ∧
  (∀ x : ℝ, deriv Real.sin x = Real.cos x) ∧
  (∀ x : ℝ, deriv (λ x => Real.exp (-x)) x = -Real.exp (-x)) ∧
  (∀ x : ℝ, x ≠ -1 → deriv (λ x => Real.log (x + 1)) x = 1 / (x + 1)) := by
  sorry

end derivatives_verification_l892_89271


namespace sin_cos_identity_l892_89299

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l892_89299


namespace car_pedestrian_speed_ratio_l892_89210

/-- The ratio of car speed to pedestrian speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (vp vc : ℝ),
  L > 0 →  -- The bridge has positive length
  vp > 0 →  -- The pedestrian's speed is positive
  vc > 0 →  -- The car's speed is positive
  (2 / 5 * L) / vp = L / vc →  -- Time for pedestrian to return equals time for car to reach start
  (3 / 5 * L) / vp = L / vc →  -- Time for pedestrian to finish equals time for car to finish
  vc / vp = 5 := by
sorry

end car_pedestrian_speed_ratio_l892_89210


namespace perimeter_plus_area_of_specific_parallelogram_l892_89232

/-- A parallelogram in a 2D coordinate plane -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The sum of perimeter and area of a specific parallelogram -/
theorem perimeter_plus_area_of_specific_parallelogram :
  let p := Parallelogram.mk (2, 1) (7, 1) (5, 6) (10, 6)
  perimeter p + area p = 35 + 2 * Real.sqrt 34 := by sorry

end perimeter_plus_area_of_specific_parallelogram_l892_89232


namespace direct_proportion_conditions_l892_89213

/-- A function representing a potential direct proportion -/
def f (k b x : ℝ) : ℝ := (k - 4) * x + b

/-- Definition of a direct proportion function -/
def is_direct_proportion (g : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, g x = m * x

/-- Theorem stating the necessary and sufficient conditions for f to be a direct proportion -/
theorem direct_proportion_conditions (k b : ℝ) :
  is_direct_proportion (f k b) ↔ k ≠ 4 ∧ b = 0 :=
sorry

end direct_proportion_conditions_l892_89213


namespace power_equation_solution_l892_89275

theorem power_equation_solution (y : ℕ) : 8^5 + 8^5 + 2 * 8^5 = 2^y → y = 17 := by
  sorry

end power_equation_solution_l892_89275


namespace p_and_not_q_is_true_l892_89234

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 1

-- Theorem statement
theorem p_and_not_q_is_true : p ∧ ¬q := by sorry

end p_and_not_q_is_true_l892_89234


namespace sqrt_53_between_consecutive_integers_l892_89290

theorem sqrt_53_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ)^2 < 53 ∧ 53 < (n + 1 : ℝ)^2 ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_53_between_consecutive_integers_l892_89290


namespace largest_x_sqrt_3x_eq_6x_l892_89270

theorem largest_x_sqrt_3x_eq_6x :
  ∃ (x_max : ℚ), x_max = 1/12 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 6 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 6 * x_max :=
sorry

end largest_x_sqrt_3x_eq_6x_l892_89270


namespace inequality_range_l892_89272

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ a ∈ Set.Ioc (-8/5) 0 :=
sorry

end inequality_range_l892_89272


namespace divisor_problem_l892_89296

/-- 
Given a dividend of 23, a quotient of 5, and a remainder of 3, 
prove that the divisor is 4.
-/
theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = 23 → quotient = 5 → remainder = 3 → 
  dividend = divisor * quotient + remainder → 
  divisor = 4 := by
sorry

end divisor_problem_l892_89296


namespace union_of_A_and_B_l892_89246

def set_A : Set ℝ := {x | x < -1 ∨ x > 3}
def set_B : Set ℝ := {x | x - 2 ≥ 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x | x < -1 ∨ x ≥ 2} := by sorry

end union_of_A_and_B_l892_89246


namespace triple_digit_sum_of_2012_pow_2012_l892_89227

/-- The sum of the digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The function that applies digit_sum three times -/
def triple_digit_sum (n : ℕ) : ℕ := digit_sum (digit_sum (digit_sum n))

theorem triple_digit_sum_of_2012_pow_2012 :
  triple_digit_sum (2012^2012) = 7 := by sorry

end triple_digit_sum_of_2012_pow_2012_l892_89227


namespace infinite_series_sum_l892_89267

/-- The sum of the infinite series ∑(n=1 to ∞) (2n + 1) / (n(n + 1)(n + 2)) is equal to 1 -/
theorem infinite_series_sum : 
  (∑' n : ℕ+, (2 * n.val + 1 : ℝ) / (n.val * (n.val + 1) * (n.val + 2))) = 1 := by
  sorry

end infinite_series_sum_l892_89267


namespace pen_average_price_l892_89214

/-- Given the purchase of pens and pencils with specific quantities and prices,
    prove that the average price of a pen is $12. -/
theorem pen_average_price
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (total_cost : ℚ)
  (pencil_avg_price : ℚ)
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 510)
  (h4 : pencil_avg_price = 2) :
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 12 :=
by sorry

end pen_average_price_l892_89214


namespace investment_period_l892_89215

theorem investment_period (emma_investment briana_investment : ℝ)
  (emma_yield briana_yield : ℝ) (difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  briana_yield = 0.10 →
  difference = 10 →
  ∃ t : ℝ, t = 2 ∧ 
    t * (briana_investment * briana_yield - emma_investment * emma_yield) = difference :=
by sorry

end investment_period_l892_89215


namespace coin_toss_recurrence_l892_89287

/-- The probability of having a group of length k or more in n tosses of a symmetric coin. -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem coin_toss_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end coin_toss_recurrence_l892_89287


namespace uniform_payment_proof_l892_89255

theorem uniform_payment_proof :
  ∃ (x y : ℕ), 
    5 * x - 3 * y = 24 ∧ 
    x > 0 ∧ 
    y ≥ 0 ∧ 
    ∀ (x' y' : ℕ), 5 * x' - 3 * y' = 24 → x' > 0 → y' ≥ 0 → x ≤ x' ∧ y ≤ y' :=
by sorry

end uniform_payment_proof_l892_89255


namespace product_pricing_equation_l892_89261

/-- 
Given a product with:
- Marked price of 1375 yuan
- Sold at 80% of the marked price
- Making a profit of 100 yuan
Prove that the equation relating the cost price x to these values is:
1375 * 80% = x + 100
-/
theorem product_pricing_equation (x : ℝ) : 
  1375 * (80 / 100) = x + 100 := by sorry

end product_pricing_equation_l892_89261


namespace regular_pentagon_side_length_l892_89289

/-- A regular pentagon with a perimeter of 23.4 cm has sides of length 4.68 cm. -/
theorem regular_pentagon_side_length : 
  ∀ (p : ℝ) (s : ℝ), 
  p = 23.4 →  -- perimeter is 23.4 cm
  s = p / 5 →  -- side length is perimeter divided by 5 (number of sides in a pentagon)
  s = 4.68 := by
sorry

end regular_pentagon_side_length_l892_89289


namespace blue_pill_cost_is_21_l892_89231

/-- The cost of a blue pill given the conditions of Ben's medication regimen -/
def blue_pill_cost (total_cost : ℚ) (duration_days : ℕ) (blue_red_diff : ℚ) : ℚ :=
  let daily_cost : ℚ := total_cost / duration_days
  let x : ℚ := (daily_cost + blue_red_diff) / 2
  x

theorem blue_pill_cost_is_21 :
  blue_pill_cost 819 21 3 = 21 := by
  sorry

end blue_pill_cost_is_21_l892_89231


namespace inequality_and_equality_l892_89274

theorem inequality_and_equality (x : ℝ) (h : x > 0) : 
  (x + 1/x ≥ 2) ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end inequality_and_equality_l892_89274


namespace bill_split_correct_l892_89203

/-- The number of people splitting the bill -/
def num_people : ℕ := 9

/-- The total bill amount in cents -/
def total_bill : ℕ := 51416

/-- The amount each person should pay in cents, rounded to the nearest cent -/
def amount_per_person : ℕ := 5713

/-- Theorem stating that the calculated amount per person is correct -/
theorem bill_split_correct : 
  (total_bill + num_people - 1) / num_people = amount_per_person :=
sorry

end bill_split_correct_l892_89203


namespace abc_inequality_l892_89222

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^3 - a^2 + 3)) + (b / (b^3 - b^2 + 3)) + (c / (c^3 - c^2 + 3)) ≤ 1 := by
  sorry

end abc_inequality_l892_89222


namespace average_after_removal_l892_89226

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 10 →
  sum = Finset.sum numbers id →
  sum / 10 = 85 →
  72 ∈ numbers →
  78 ∈ numbers →
  ((sum - 72 - 78) / 8) = 87.5 :=
sorry

end average_after_removal_l892_89226


namespace cow_value_increase_is_600_l892_89245

/-- Calculates the increase in a cow's value after weight gain -/
def cow_value_increase (initial_weight : ℝ) (weight_increase_factor : ℝ) (price_per_pound : ℝ) : ℝ :=
  (initial_weight * weight_increase_factor * price_per_pound) - (initial_weight * price_per_pound)

/-- Theorem stating that the increase in the cow's value is $600 -/
theorem cow_value_increase_is_600 :
  cow_value_increase 400 1.5 3 = 600 := by
  sorry

end cow_value_increase_is_600_l892_89245


namespace half_of_four_power_2022_l892_89262

theorem half_of_four_power_2022 : (4 ^ 2022) / 2 = 2 ^ 4043 := by
  sorry

end half_of_four_power_2022_l892_89262


namespace gcd_459_357_l892_89250

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l892_89250


namespace intersection_uniqueness_l892_89217

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := -2 * y = 7 * x - 3

/-- The intersection point -/
def intersection_point : ℚ × ℚ := (-3/17, 36/17)

theorem intersection_uniqueness :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point := by
  sorry

end intersection_uniqueness_l892_89217


namespace speed_equivalence_l892_89273

/-- Conversion factor from m/s to km/h -/
def meters_per_second_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 30.0024

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 108.00864

/-- Theorem stating that the given speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * meters_per_second_to_kmph := by
  sorry

end speed_equivalence_l892_89273


namespace power_of_six_seven_equals_product_of_seven_sixes_l892_89282

theorem power_of_six_seven_equals_product_of_seven_sixes :
  6^7 = (List.replicate 7 6).prod := by
  sorry

end power_of_six_seven_equals_product_of_seven_sixes_l892_89282


namespace vector_linear_combination_l892_89280

/-- Given vectors a, b, and c in ℝ², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end vector_linear_combination_l892_89280


namespace Betty_wallet_contribution_ratio_l892_89297

theorem Betty_wallet_contribution_ratio :
  let wallet_cost : ℚ := 100
  let initial_savings : ℚ := wallet_cost / 2
  let parents_contribution : ℚ := 15
  let remaining_need : ℚ := 5
  let grandparents_contribution : ℚ := wallet_cost - initial_savings - parents_contribution - remaining_need
  grandparents_contribution / parents_contribution = 2 := by
    sorry

end Betty_wallet_contribution_ratio_l892_89297


namespace parallel_lines_m_equals_one_l892_89248

/-- Two lines are parallel if their slopes are equal -/
def parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line: x + (1+m)y + (m-2) = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (1+m)*y + (m-2) = 0

/-- The second line: mx + 2y + 8 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m*x + 2*y + 8 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) m 2 → m = 1 := by
  sorry

end parallel_lines_m_equals_one_l892_89248


namespace problem_1_l892_89230

theorem problem_1 (x : ℕ) : 2 * 8^x * 16^x = 2^22 → x = 3 := by
  sorry

end problem_1_l892_89230


namespace cost_of_3200_pencils_l892_89249

/-- The cost of a given number of pencils based on a known price for a box of pencils -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (num_pencils : ℕ) : ℚ :=
  (box_cost * num_pencils) / box_size

/-- Theorem: Given a box of 160 personalized pencils costs $48, the cost of 3200 pencils is $960 -/
theorem cost_of_3200_pencils :
  pencil_cost 160 48 3200 = 960 := by
  sorry

end cost_of_3200_pencils_l892_89249


namespace total_spent_is_520_l892_89218

/-- Shopping expenses for Lisa and Carly -/
def shopping_expenses (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : Prop :=
  T_L = 40 ∧
  J_L = T_L / 2 ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = T_L / 4 ∧
  J_C = 3 * J_L ∧
  C_C = C_L / 2 ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = J_C / 2

/-- The total amount spent by Lisa and Carly -/
def total_spent (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : ℝ :=
  T_L + J_L + C_L + S_L + T_C + J_C + C_C + S_C + D_C + A_C

/-- Theorem stating that the total amount spent is $520 -/
theorem total_spent_is_520 :
  ∀ T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ,
  shopping_expenses T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_spent T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C = 520 :=
by sorry

end total_spent_is_520_l892_89218


namespace largest_prime_factor_of_sum_of_divisors_300_l892_89208

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_300 :
  largest_prime_factor (sum_of_divisors 300) = 31 := by sorry

end largest_prime_factor_of_sum_of_divisors_300_l892_89208


namespace compound_interest_initial_sum_l892_89295

/-- Given an initial sum of money P and an annual compound interest rate r,
    if P(1 + r)² = 8880 and P(1 + r)³ = 9261, then P is approximately equal to 8160. -/
theorem compound_interest_initial_sum (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8880)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 8160| < ε :=
sorry

end compound_interest_initial_sum_l892_89295


namespace perpendicular_distance_is_six_l892_89247

/-- A rectangular parallelepiped with dimensions 6 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ := 6
  width : ℝ := 5
  height : ℝ := 4

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a point to a plane -/
def perpendicularDistance (S : Point3D) (P Q R : Point3D) : ℝ := sorry

theorem perpendicular_distance_is_six :
  let p : Parallelepiped := { }
  let S : Point3D := ⟨6, 0, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  let Q : Point3D := ⟨0, 5, 0⟩
  let R : Point3D := ⟨0, 0, 4⟩
  perpendicularDistance S P Q R = 6 := by sorry

end perpendicular_distance_is_six_l892_89247


namespace roses_sold_l892_89279

/-- Proves that the number of roses sold is 2, given the initial, picked, and final numbers of roses. -/
theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) 
  (h1 : initial = 11) 
  (h2 : picked = 32) 
  (h3 : final = 41) : 
  initial - (final - picked) = 2 := by
  sorry

end roses_sold_l892_89279


namespace vector_subtraction_magnitude_l892_89240

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_subtraction_magnitude : ‖a - b‖ = 5 := by sorry

end vector_subtraction_magnitude_l892_89240


namespace equation_with_increasing_roots_l892_89254

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x + 1) - (m + 1) / (x^2 + x) = (x + 1) / x

-- Define the concept of increasing roots
def has_increasing_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < y ∧ equation x m ∧ equation y m

-- Theorem statement
theorem equation_with_increasing_roots (m : ℝ) :
  has_increasing_roots m → m = -2 ∨ m = 0 := by
  sorry

end equation_with_increasing_roots_l892_89254


namespace new_year_cards_cost_l892_89241

def card_price_1 : ℚ := 10 / 100
def card_price_2 : ℚ := 15 / 100
def card_price_3 : ℚ := 25 / 100
def card_price_4 : ℚ := 40 / 100

def total_cards : ℕ := 30

theorem new_year_cards_cost (q1 q2 q3 q4 : ℕ) 
  (h1 : q1 + q2 + q3 + q4 = total_cards)
  (h2 : (q1 = 5 ∧ q2 = 5) ∨ (q1 = 5 ∧ q3 = 5) ∨ (q1 = 5 ∧ q4 = 5) ∨ 
        (q2 = 5 ∧ q3 = 5) ∨ (q2 = 5 ∧ q4 = 5) ∨ (q3 = 5 ∧ q4 = 5))
  (h3 : (q1 = 10 ∧ q2 = 10) ∨ (q1 = 10 ∧ q3 = 10) ∨ (q1 = 10 ∧ q4 = 10) ∨ 
        (q2 = 10 ∧ q3 = 10) ∨ (q2 = 10 ∧ q4 = 10) ∨ (q3 = 10 ∧ q4 = 10))
  (h4 : ∃ (n : ℕ), q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = n) :
  q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = 7 := by
sorry


end new_year_cards_cost_l892_89241


namespace largest_k_inequality_l892_89285

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∃ k : ℕ+, k = 4 ∧
    (∀ m : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ (m : ℝ) / (a - c)) → m ≤ k) ∧
    (1 / (a - b) + 1 / (b - c) ≥ (k : ℝ) / (a - c))) :=
sorry

end largest_k_inequality_l892_89285


namespace new_person_weight_l892_89235

theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 70 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 90 :=
by sorry

end new_person_weight_l892_89235


namespace right_triangle_sides_l892_89288

/-- A right triangle with perimeter 60 and height to hypotenuse 12 has sides 15, 20, and 35 -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  a^2 + b^2 = c^2 →
  a * b = 12 * c →
  h = 12 →
  (a = 15 ∧ b = 20 ∧ c = 35) ∨ (a = 20 ∧ b = 15 ∧ c = 35) :=
by sorry


end right_triangle_sides_l892_89288


namespace anya_hair_growth_l892_89244

/-- The number of hairs Anya washes down the drain -/
def washed_hairs : ℕ := 32

/-- The number of hairs Anya brushes out -/
def brushed_hairs : ℕ := washed_hairs / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := washed_hairs + brushed_hairs + 1

theorem anya_hair_growth :
  hairs_to_grow = 49 :=
by sorry

end anya_hair_growth_l892_89244


namespace age_difference_proof_l892_89283

theorem age_difference_proof (younger_age elder_age : ℕ) : 
  younger_age = 35 →
  elder_age - 15 = 2 * (younger_age - 15) →
  elder_age - younger_age = 20 :=
by
  sorry

end age_difference_proof_l892_89283


namespace lisa_photos_l892_89201

def photo_problem (animal_photos flower_photos scenery_photos this_weekend last_weekend : ℕ) : Prop :=
  animal_photos = 10 ∧
  flower_photos = 3 * animal_photos ∧
  scenery_photos = flower_photos - 10 ∧
  this_weekend = animal_photos + flower_photos + scenery_photos ∧
  last_weekend = this_weekend - 15

theorem lisa_photos :
  ∀ animal_photos flower_photos scenery_photos this_weekend last_weekend,
  photo_problem animal_photos flower_photos scenery_photos this_weekend last_weekend →
  last_weekend = 45 := by
sorry

end lisa_photos_l892_89201


namespace special_function_properties_l892_89225

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧
  (∀ x > 0, f x > 1) ∧
  (∀ a b : ℝ, f (a + b) = f a * f b)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 1) ∧
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end special_function_properties_l892_89225


namespace prob_same_foot_is_three_sevenths_l892_89219

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 4

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes out of the total shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

/-- The number of ways to select 2 shoes from the same foot -/
def same_foot_selections : ℕ := 2 * Nat.choose num_pairs selected_shoes

/-- The probability of selecting two shoes from the same foot -/
def prob_same_foot : ℚ := same_foot_selections / total_selections

theorem prob_same_foot_is_three_sevenths :
  prob_same_foot = 3 / 7 := by sorry

end prob_same_foot_is_three_sevenths_l892_89219


namespace brandon_sales_theorem_l892_89256

def total_sales : ℝ := 80

theorem brandon_sales_theorem :
  let credit_sales_ratio : ℝ := 2/5
  let cash_sales_ratio : ℝ := 1 - credit_sales_ratio
  let cash_sales_amount : ℝ := 48
  cash_sales_ratio * total_sales = cash_sales_amount :=
by sorry

end brandon_sales_theorem_l892_89256


namespace three_zeros_implies_a_equals_four_l892_89268

-- Define the piecewise function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≠ 3 then 2 / |x - 3| else a

-- Define the function y
noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

-- Theorem statement
theorem three_zeros_implies_a_equals_four (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    y x₁ a = 0 ∧ y x₂ a = 0 ∧ y x₃ a = 0) →
  (∀ x : ℝ, y x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  a = 4 :=
by sorry


end three_zeros_implies_a_equals_four_l892_89268


namespace sum_equals_zero_l892_89242

/-- The number of numbers satisfying: "there is no other number whose absolute value is equal to the absolute value of a" -/
def a : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number whose square is equal to the square of b" -/
def b : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number that, when multiplied by c, results in a product greater than 1" -/
def c : ℕ := sorry

theorem sum_equals_zero : a + b + c = 0 := by sorry

end sum_equals_zero_l892_89242


namespace general_term_formula_min_value_S_min_value_n_l892_89266

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem for the general term
theorem general_term_formula : ∀ n : ℕ, a n = S n - S (n - 1) :=
sorry

-- Theorem for the minimum value of S_n
theorem min_value_S : ∃ n : ℕ, S n = -112 ∧ ∀ m : ℕ, S m ≥ -112 :=
sorry

-- Theorem for the values of n that give the minimum
theorem min_value_n : ∀ n : ℕ, S n = -112 ↔ (n = 7 ∨ n = 8) :=
sorry

end general_term_formula_min_value_S_min_value_n_l892_89266


namespace milk_for_flour_batch_l892_89239

/-- Given that 60 mL of milk is used for every 300 mL of flour,
    prove that 300 mL of milk is needed for 1500 mL of flour. -/
theorem milk_for_flour_batch (milk_per_portion : ℝ) (flour_per_portion : ℝ) 
    (total_flour : ℝ) (h1 : milk_per_portion = 60) 
    (h2 : flour_per_portion = 300) (h3 : total_flour = 1500) : 
    (total_flour / flour_per_portion) * milk_per_portion = 300 :=
by sorry

end milk_for_flour_batch_l892_89239


namespace total_savings_is_40_l892_89220

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the conversion rates
def pennies_per_dollar : ℕ := 100
def nickels_per_dollar : ℕ := 20
def dimes_per_dollar : ℕ := 10

-- Define the total savings
def total_savings : ℚ :=
  (teagan_pennies : ℚ) / pennies_per_dollar +
  (rex_nickels : ℚ) / nickels_per_dollar +
  (toni_dimes : ℚ) / dimes_per_dollar

-- Theorem statement
theorem total_savings_is_40 : total_savings = 40 := by
  sorry

end total_savings_is_40_l892_89220


namespace point_coordinates_l892_89257

/-- Given a point P that is 2 units right and 4 units up from the origin (0,0),
    prove that the coordinates of P are (2,4). -/
theorem point_coordinates (P : ℝ × ℝ) 
  (h1 : P.1 = 2)  -- P is 2 units right from the origin
  (h2 : P.2 = 4)  -- P is 4 units up from the origin
  : P = (2, 4) := by
  sorry

end point_coordinates_l892_89257


namespace jean_domino_friends_l892_89258

theorem jean_domino_friends :
  ∀ (total_dominoes : ℕ) (dominoes_per_player : ℕ) (total_players : ℕ),
    total_dominoes = 28 →
    dominoes_per_player = 7 →
    total_players * dominoes_per_player = total_dominoes →
    total_players - 1 = 3 :=
by
  sorry

end jean_domino_friends_l892_89258


namespace fair_coin_five_tosses_l892_89259

/-- The probability of a fair coin landing on the same side for all tosses -/
def same_side_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_five_tosses :
  same_side_probability 5 = 1 / 32 := by
  sorry


end fair_coin_five_tosses_l892_89259


namespace pages_left_to_read_l892_89284

/-- Given a book with 400 pages, prove that after reading 20% of it, 320 pages are left to read. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) : 
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

#eval (400 : ℕ) - ((400 : ℕ) * (20 / 100 : ℚ)).floor

end pages_left_to_read_l892_89284


namespace division_problem_l892_89205

theorem division_problem (divisor : ℕ) : 
  (171 / divisor = 8) ∧ (171 % divisor = 3) → divisor = 21 := by
  sorry

end division_problem_l892_89205


namespace main_divisors_equal_implies_equal_l892_89238

/-- The two largest proper divisors of a composite natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d ≠ 1 ∧ ∀ k ∈ Nat.divisors n, k ≠ n → k ≠ 1 → d ≥ k}

/-- A natural number is composite if it has at least one proper divisor -/
def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem main_divisors_equal_implies_equal (a b : ℕ) 
  (ha : is_composite a) (hb : is_composite b) 
  (h : main_divisors a = main_divisors b) : 
  a = b :=
sorry

end main_divisors_equal_implies_equal_l892_89238


namespace absolute_difference_inequality_l892_89294

theorem absolute_difference_inequality (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |x - y| < |1 - x*y| := by
  sorry

end absolute_difference_inequality_l892_89294


namespace third_term_is_four_l892_89236

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem: The third term of the sequence is 4 -/
theorem third_term_is_four : a 3 = 4 := by
  sorry

end third_term_is_four_l892_89236


namespace deposit_calculation_l892_89237

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) : 
  deposit_percentage = 0.1 →
  remaining_amount = 945 →
  total_price * (1 - deposit_percentage) = remaining_amount →
  total_price * deposit_percentage = 105 := by
sorry

end deposit_calculation_l892_89237


namespace geometric_sequence_common_ratio_l892_89228

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    if a₁ + a₃ = 10 and a₄ + a₆ = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 1 + a 3 = 10) 
  (h3 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by sorry

end geometric_sequence_common_ratio_l892_89228


namespace tomato_price_per_pound_l892_89260

/-- Calculates the price per pound of a tomato based on grocery shopping details. -/
theorem tomato_price_per_pound
  (meat_weight : Real)
  (meat_price_per_pound : Real)
  (buns_price : Real)
  (lettuce_price : Real)
  (tomato_weight : Real)
  (pickles_price : Real)
  (pickles_coupon : Real)
  (paid_amount : Real)
  (change_received : Real)
  (h1 : meat_weight = 2)
  (h2 : meat_price_per_pound = 3.5)
  (h3 : buns_price = 1.5)
  (h4 : lettuce_price = 1)
  (h5 : tomato_weight = 1.5)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_coupon = 1)
  (h8 : paid_amount = 20)
  (h9 : change_received = 6) :
  (paid_amount - change_received - (meat_weight * meat_price_per_pound + buns_price + lettuce_price + (pickles_price - pickles_coupon))) / tomato_weight = 2 := by
  sorry


end tomato_price_per_pound_l892_89260
