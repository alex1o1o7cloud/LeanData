import Mathlib

namespace lottery_win_probability_l4071_407149

/-- A lottery event with two prize categories -/
structure LotteryEvent where
  firstPrizeProb : ℝ
  secondPrizeProb : ℝ

/-- The probability of winning a prize in the lottery event -/
def winPrizeProb (event : LotteryEvent) : ℝ :=
  event.firstPrizeProb + event.secondPrizeProb

/-- Theorem stating the probability of winning a prize in the given lottery event -/
theorem lottery_win_probability :
  ∃ (event : LotteryEvent), 
    event.firstPrizeProb = 0.1 ∧ 
    event.secondPrizeProb = 0.1 ∧ 
    winPrizeProb event = 0.2 := by
  sorry

end lottery_win_probability_l4071_407149


namespace right_triangle_hypotenuse_l4071_407138

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by sorry

end right_triangle_hypotenuse_l4071_407138


namespace smallest_prime_square_mod_six_is_five_l4071_407178

theorem smallest_prime_square_mod_six_is_five :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p^2 % 6 = 1 ∧ 
    (∀ (q : ℕ), Nat.Prime q → q^2 % 6 = 1 → p ≤ q) ∧
    p = 5 := by
  sorry

end smallest_prime_square_mod_six_is_five_l4071_407178


namespace distinct_values_count_l4071_407101

-- Define the expression
def base : ℕ := 3
def expr := base^(base^(base^base))

-- Define the possible parenthesizations
def p1 := base^(base^(base^base))
def p2 := base^((base^base)^base)
def p3 := ((base^base)^base)^base
def p4 := (base^(base^base))^base
def p5 := (base^base)^(base^base)

-- Theorem statement
theorem distinct_values_count :
  ∃ (s : Finset ℕ), (∀ x : ℕ, x ∈ s ↔ (x = p1 ∨ x = p2 ∨ x = p3 ∨ x = p4 ∨ x = p5)) ∧ Finset.card s = 2 :=
sorry

end distinct_values_count_l4071_407101


namespace stating_sweet_apple_percentage_correct_l4071_407183

/-- Represents the percentage of sweet apples in Chang's Garden. -/
def sweet_apple_percentage : ℝ := 75

/-- Represents the total number of apples sold. -/
def total_apples : ℕ := 100

/-- Represents the price of a sweet apple in dollars. -/
def sweet_apple_price : ℝ := 0.5

/-- Represents the price of a sour apple in dollars. -/
def sour_apple_price : ℝ := 0.1

/-- Represents the total earnings from selling all apples in dollars. -/
def total_earnings : ℝ := 40

/-- 
Theorem stating that the percentage of sweet apples is correct given the conditions.
-/
theorem sweet_apple_percentage_correct : 
  sweet_apple_price * (sweet_apple_percentage / 100 * total_apples) + 
  sour_apple_price * ((100 - sweet_apple_percentage) / 100 * total_apples) = 
  total_earnings := by sorry

end stating_sweet_apple_percentage_correct_l4071_407183


namespace sequence_formula_l4071_407187

theorem sequence_formula (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 1 :=
by
  sorry

end sequence_formula_l4071_407187


namespace equation_solution_l4071_407147

theorem equation_solution : 
  ∃! x : ℝ, (2 + x ≠ 0 ∧ 3 * x - 1 ≠ 0) ∧ (1 / (2 + x) = 2 / (3 * x - 1)) ∧ x = 5 := by
  sorry

end equation_solution_l4071_407147


namespace parabola_has_one_x_intercept_l4071_407191

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -3 * y^2 + 2 * y + 2

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola_equation x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end parabola_has_one_x_intercept_l4071_407191


namespace smallest_three_digit_multiple_l4071_407105

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end smallest_three_digit_multiple_l4071_407105


namespace multiplication_result_l4071_407107

theorem multiplication_result : 2.68 * 0.74 = 1.9832 := by
  sorry

end multiplication_result_l4071_407107


namespace correct_rounding_l4071_407126

def round_to_thousandth (x : ℚ) : ℚ :=
  (⌊x * 1000 + 0.5⌋ : ℚ) / 1000

theorem correct_rounding :
  round_to_thousandth 2.098176 = 2.098 := by sorry

end correct_rounding_l4071_407126


namespace binary_arrangements_count_l4071_407176

/-- The number of ways to arrange 3 ones and 3 zeros in a binary string -/
def binaryArrangements : ℕ := 20

/-- The length of the binary string -/
def stringLength : ℕ := 6

/-- The number of ones in the binary string -/
def numberOfOnes : ℕ := 3

theorem binary_arrangements_count :
  binaryArrangements = Nat.choose stringLength numberOfOnes := by
  sorry

end binary_arrangements_count_l4071_407176


namespace ellipse_equation_l4071_407116

/-- The locus of points P such that |F₁F₂| is the arithmetic mean of |PF₁| and |PF₂| -/
def EllipseLocus (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist F₁ F₂ = (dist P F₁ + dist P F₂) / 2}

theorem ellipse_equation (P : ℝ × ℝ) :
  P ∈ EllipseLocus (-2, 0) (2, 0) ↔ P.1^2 / 16 + P.2^2 / 12 = 1 := by
  sorry

end ellipse_equation_l4071_407116


namespace marble_difference_l4071_407150

theorem marble_difference (total : ℕ) (yellow : ℕ) (h1 : total = 913) (h2 : yellow = 514) :
  yellow - (total - yellow) = 115 := by
  sorry

end marble_difference_l4071_407150


namespace diamond_equation_solution_l4071_407163

/-- Define the diamond operation -/
def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

/-- Theorem: If A ◇ 5 = 82, then A = 12 -/
theorem diamond_equation_solution :
  ∀ A : ℝ, diamond A 5 = 82 → A = 12 := by
  sorry

end diamond_equation_solution_l4071_407163


namespace julies_landscaping_hours_l4071_407131

/-- Julie's landscaping business problem -/
theorem julies_landscaping_hours (mowing_rate pulling_rate pulling_hours total_earnings : ℕ) :
  mowing_rate = 4 →
  pulling_rate = 8 →
  pulling_hours = 3 →
  total_earnings = 248 →
  ∃ (mowing_hours : ℕ),
    2 * (mowing_rate * mowing_hours + pulling_rate * pulling_hours) = total_earnings ∧
    mowing_hours = 25 :=
by sorry

end julies_landscaping_hours_l4071_407131


namespace unique_quadratic_solution_l4071_407110

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, 2 * b * x^2 + 16 * x + 5 = 0) :
  ∃ x, 2 * b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8 := by
  sorry

end unique_quadratic_solution_l4071_407110


namespace inequality_solution_range_l4071_407158

theorem inequality_solution_range (a : ℝ) : 
  (∃! (x y : ℤ), x ≠ y ∧ 
    (∀ (z : ℤ), z^2 - (a+1)*z + a < 0 ↔ (z = x ∨ z = y))) ↔ 
  (a ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 3 4) :=
sorry

end inequality_solution_range_l4071_407158


namespace sally_cost_theorem_l4071_407188

def lightning_cost : ℝ := 140000

def mater_cost : ℝ := 0.1 * lightning_cost

def sally_cost : ℝ := 3 * mater_cost

theorem sally_cost_theorem : sally_cost = 42000 := by
  sorry

end sally_cost_theorem_l4071_407188


namespace complement_of_N_in_M_l4071_407197

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end complement_of_N_in_M_l4071_407197


namespace smallest_block_size_l4071_407124

theorem smallest_block_size (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 378 → 
  l * m * n ≥ 560 :=
by sorry

end smallest_block_size_l4071_407124


namespace karls_total_distance_l4071_407180

/-- Represents the problem of calculating Karl's total driving distance --/
def karls_drive (miles_per_gallon : ℚ) (tank_capacity : ℚ) (initial_distance : ℚ) 
  (refuel_amount : ℚ) (final_tank_fraction : ℚ) : Prop :=
  let initial_fuel_used : ℚ := initial_distance / miles_per_gallon
  let remaining_fuel : ℚ := refuel_amount - (tank_capacity * final_tank_fraction)
  let additional_distance : ℚ := remaining_fuel * miles_per_gallon
  let total_distance : ℚ := initial_distance + additional_distance
  total_distance = 517

/-- Theorem stating that Karl drove 517 miles given the problem conditions --/
theorem karls_total_distance : 
  karls_drive 25 16 400 10 (1/3) :=
by
  sorry

end karls_total_distance_l4071_407180


namespace triangle_angle_and_vector_dot_product_l4071_407119

theorem triangle_angle_and_vector_dot_product 
  (A B C : ℝ) (a b c : ℝ) (k : ℝ) :
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  k > 1 ∧
  (∀ t : ℝ, 0 < t ∧ t ≤ 1 → -2 * t^2 + 4 * k * t + 1 ≤ 5) ∧
  -2 + 4 * k + 1 = 5 →
  B = π / 3 ∧ k = 3 / 2 := by
sorry

end triangle_angle_and_vector_dot_product_l4071_407119


namespace max_y_over_x_on_circle_l4071_407136

theorem max_y_over_x_on_circle :
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - Real.sqrt 3)^2 = 3}
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (p : ℝ × ℝ), p ∈ circle → p.2 / p.1 ≤ max := by
  sorry

end max_y_over_x_on_circle_l4071_407136


namespace c_rent_share_is_72_l4071_407120

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℚ) (oxen_a : ℕ) (months_a : ℕ) (oxen_b : ℕ) (months_b : ℕ) (oxen_c : ℕ) (months_c : ℕ) : ℚ :=
  let total_oxen_months := oxen_a * months_a + oxen_b * months_b + oxen_c * months_c
  let c_oxen_months := oxen_c * months_c
  (c_oxen_months : ℚ) / total_oxen_months * total_rent

/-- Theorem stating that C's share of the rent is approximately 72 -/
theorem c_rent_share_is_72 :
  let rent_share := RentShare 280 10 7 12 5 15 3
  ∃ ε > 0, abs (rent_share - 72) < ε :=
by
  sorry


end c_rent_share_is_72_l4071_407120


namespace basketball_score_proof_l4071_407190

-- Define the scores for each quarter
def alpha_scores (a r : ℝ) : Fin 4 → ℝ
| 0 => a
| 1 => a * r
| 2 => a * r^2
| 3 => a * r^3

def beta_scores (b d : ℝ) : Fin 4 → ℝ
| 0 => b
| 1 => b + d
| 2 => b + 2*d
| 3 => b + 3*d

-- Define the theorem
theorem basketball_score_proof 
  (a r b d : ℝ) 
  (h1 : 0 < r) -- Ensure increasing geometric sequence
  (h2 : 0 < d) -- Ensure increasing arithmetic sequence
  (h3 : alpha_scores a r 0 + alpha_scores a r 1 = beta_scores b d 0 + beta_scores b d 1) -- Tied at second quarter
  (h4 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) = 
        (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) + 2) -- Alpha wins by 2
  (h5 : (alpha_scores a r 0 + alpha_scores a r 1 + alpha_scores a r 2 + alpha_scores a r 3) ≤ 100) -- Alpha's total ≤ 100
  (h6 : (beta_scores b d 0 + beta_scores b d 1 + beta_scores b d 2 + beta_scores b d 3) ≤ 100) -- Beta's total ≤ 100
  : (alpha_scores a r 0 + alpha_scores a r 1 + beta_scores b d 0 + beta_scores b d 1) = 24 :=
by sorry


end basketball_score_proof_l4071_407190


namespace cubic_sum_theorem_l4071_407157

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (prod_sum_eq : a * b + a * c + b * c = 5) 
  (prod_eq : a * b * c = -12) : 
  a^3 + b^3 + c^3 = 90 := by
sorry

end cubic_sum_theorem_l4071_407157


namespace square_root_of_four_l4071_407174

theorem square_root_of_four :
  {y : ℝ | y^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l4071_407174


namespace complex_modulus_problem_l4071_407102

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l4071_407102


namespace easter_egg_distribution_l4071_407162

theorem easter_egg_distribution (red_total orange_total : ℕ) 
  (h_red : red_total = 20) (h_orange : orange_total = 30) : ∃ (eggs_per_basket : ℕ), 
  eggs_per_basket ≥ 5 ∧ 
  red_total % eggs_per_basket = 0 ∧ 
  orange_total % eggs_per_basket = 0 ∧
  ∀ (n : ℕ), n ≥ 5 ∧ red_total % n = 0 ∧ orange_total % n = 0 → n ≥ eggs_per_basket :=
by
  sorry

end easter_egg_distribution_l4071_407162


namespace binomial_sum_problem_l4071_407186

theorem binomial_sum_problem (n : ℕ) (M N : ℝ) : 
  M = (5 - 1/2)^n →  -- Sum of coefficients of (5x - 1/√x)^n
  N = 2^n →          -- Sum of binomial coefficients
  M - N = 240 → 
  N = 16 := by
sorry

end binomial_sum_problem_l4071_407186


namespace angle_conversion_and_coterminal_l4071_407133

-- Define α in degrees
def α : ℝ := 1680

-- Theorem statement
theorem angle_conversion_and_coterminal (α : ℝ) :
  ∃ (k : ℤ) (β : ℝ), 
    (α * π / 180 = 2 * k * π + β) ∧ 
    (0 ≤ β) ∧ (β < 2 * π) ∧
    (∃ (θ : ℝ), 
      (θ = -8 * π / 3) ∧ 
      (-4 * π < θ) ∧ (θ < -2 * π) ∧
      (∃ (m : ℤ), θ = 2 * m * π + β)) := by
  sorry

end angle_conversion_and_coterminal_l4071_407133


namespace zoo_consumption_theorem_l4071_407173

/-- Represents the daily consumption of fish for an animal -/
structure DailyConsumption where
  trout : Float
  salmon : Float

/-- Calculates the total monthly consumption for all animals -/
def totalMonthlyConsumption (animals : List DailyConsumption) (days : Nat) : Float :=
  let dailyTotal := animals.foldl (fun acc x => acc + x.trout + x.salmon) 0
  dailyTotal * days.toFloat

/-- Theorem stating the total monthly consumption for the given animals -/
theorem zoo_consumption_theorem (pb1 pb2 pb3 sl1 sl2 : DailyConsumption)
    (h1 : pb1 = { trout := 0.2, salmon := 0.4 })
    (h2 : pb2 = { trout := 0.3, salmon := 0.5 })
    (h3 : pb3 = { trout := 0.25, salmon := 0.45 })
    (h4 : sl1 = { trout := 0.1, salmon := 0.15 })
    (h5 : sl2 = { trout := 0.2, salmon := 0.25 }) :
    totalMonthlyConsumption [pb1, pb2, pb3, sl1, sl2] 30 = 84 := by
  sorry


end zoo_consumption_theorem_l4071_407173


namespace gcf_lcm_sum_4_10_l4071_407129

theorem gcf_lcm_sum_4_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 := by
  sorry

end gcf_lcm_sum_4_10_l4071_407129


namespace boisjoli_farm_egg_production_l4071_407170

/-- Represents the chicken coop at Boisjoli farm -/
structure ChickenCoop where
  total_hens : ℕ
  total_roosters : ℕ
  laying_percentage : ℚ
  morning_laying_percentage : ℚ
  afternoon_laying_percentage : ℚ
  unusable_egg_percentage : ℚ
  eggs_per_box : ℕ
  days_per_week : ℕ

/-- Calculates the number of boxes of usable eggs filled in a week -/
def boxes_filled_per_week (coop : ChickenCoop) : ℕ :=
  sorry

/-- The main theorem stating the number of boxes filled per week -/
theorem boisjoli_farm_egg_production :
  let coop : ChickenCoop := {
    total_hens := 270,
    total_roosters := 3,
    laying_percentage := 9/10,
    morning_laying_percentage := 4/10,
    afternoon_laying_percentage := 5/10,
    unusable_egg_percentage := 1/20,
    eggs_per_box := 7,
    days_per_week := 7
  }
  boxes_filled_per_week coop = 203 := by
  sorry

end boisjoli_farm_egg_production_l4071_407170


namespace container_volume_ratio_l4071_407167

/-- Theorem: Container Volume Ratio
Given two containers where the first is 3/7 full and transfers all its water to the second,
making it 2/3 full, the ratio of the volume of the first container to the volume of the second
container is 14/9.
-/
theorem container_volume_ratio (container1 container2 : ℝ) :
  container1 > 0 ∧ container2 > 0 →  -- Ensure containers have positive volume
  (3 / 7 : ℝ) * container1 = (2 / 3 : ℝ) * container2 → -- Water transfer equation
  container1 / container2 = 14 / 9 := by
sorry


end container_volume_ratio_l4071_407167


namespace round_robin_tournament_l4071_407111

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 := by
  sorry

end round_robin_tournament_l4071_407111


namespace root_sum_reciprocal_diff_l4071_407192

-- Define the polynomial
def f (x : ℝ) : ℝ := 20 * x^3 - 40 * x^2 + 24 * x - 2

-- State the theorem
theorem root_sum_reciprocal_diff (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- roots are distinct
  0 < a → a < 1 →                -- a is between 0 and 1
  0 < b → b < 1 →                -- b is between 0 and 1
  0 < c → c < 1 →                -- c is between 0 and 1
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 1 :=
by sorry

end root_sum_reciprocal_diff_l4071_407192


namespace not_both_prime_2n_plus_minus_one_l4071_407155

theorem not_both_prime_2n_plus_minus_one (n : ℕ) (h : n > 2) :
  ¬(Nat.Prime (2^n - 1) ∧ Nat.Prime (2^n + 1)) :=
by sorry

end not_both_prime_2n_plus_minus_one_l4071_407155


namespace horner_method_equals_polynomial_f_at_5_equals_4881_l4071_407140

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

def horner_method (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc c => acc * x + c) 0

theorem horner_method_equals_polynomial (x : ℝ) :
  horner_method [6, 5, 4, 3, 2, 1] x = f x :=
sorry

theorem f_at_5_equals_4881 :
  f 5 = 4881 :=
sorry

end horner_method_equals_polynomial_f_at_5_equals_4881_l4071_407140


namespace exists_infinite_ap_not_in_polynomial_image_l4071_407145

/-- A polynomial of degree 10 with integer coefficients -/
def IntPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
    ∀ x, P x = a₁₀ * x^10 + a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + 
             a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

/-- An infinite arithmetic progression -/
def InfiniteArithmeticProgression (a d : ℤ) : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = a + k * d}

/-- The main theorem -/
theorem exists_infinite_ap_not_in_polynomial_image (P : ℤ → ℤ) 
    (h : IntPolynomial P) :
  ∃ (a d : ℤ), d ≠ 0 ∧ 
    ∀ n ∈ InfiniteArithmeticProgression a d, 
      ∀ k : ℤ, P k ≠ n :=
by sorry

end exists_infinite_ap_not_in_polynomial_image_l4071_407145


namespace arithmetic_sequence_property_l4071_407115

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 + a 12 = 90) →
  (a 10 - (1/3) * a 14 = 12) := by
  sorry

end arithmetic_sequence_property_l4071_407115


namespace lee_overall_percentage_l4071_407134

theorem lee_overall_percentage (t : ℝ) (h1 : t > 0) : 
  let james_solo := 0.70 * (t / 2)
  let james_total := 0.85 * t
  let together := james_total - james_solo
  let lee_solo := 0.75 * (t / 2)
  lee_solo + together = 0.875 * t := by
sorry

end lee_overall_percentage_l4071_407134


namespace largest_prime_divisor_of_ribbons_l4071_407130

/-- The lengths of Amanda's ribbons in inches -/
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

/-- A function to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem stating that 2 is the largest prime that divides all ribbon lengths -/
theorem largest_prime_divisor_of_ribbons :
  ∃ (p : ℕ), is_prime p ∧ 
    (∀ (length : ℕ), length ∈ ribbon_lengths → p ∣ length) ∧
    (∀ (q : ℕ), is_prime q → 
      (∀ (length : ℕ), length ∈ ribbon_lengths → q ∣ length) → q ≤ p) ∧
    p = 2 :=
  sorry

end largest_prime_divisor_of_ribbons_l4071_407130


namespace grandma_gift_amount_l4071_407104

/-- Calculates the amount grandma gave each person given the initial amount, expenses, and remaining amount. -/
theorem grandma_gift_amount
  (initial_amount : ℝ)
  (gasoline_cost : ℝ)
  (lunch_cost : ℝ)
  (gift_cost_per_person : ℝ)
  (num_people : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : gasoline_cost = 8)
  (h3 : lunch_cost = 15.65)
  (h4 : gift_cost_per_person = 5)
  (h5 : num_people = 2)
  (h6 : remaining_amount = 36.35) :
  (remaining_amount - (initial_amount - (gasoline_cost + lunch_cost + gift_cost_per_person * num_people))) / num_people = 10 :=
by sorry

end grandma_gift_amount_l4071_407104


namespace determinant_max_value_l4071_407146

theorem determinant_max_value :
  let f : ℝ → ℝ := λ θ => 2 * Real.sqrt 2 * Real.cos θ + Real.cos (2 * θ)
  ∃ (θ : ℝ), f θ = 2 * Real.sqrt 2 + 1 ∧ ∀ (φ : ℝ), f φ ≤ 2 * Real.sqrt 2 + 1 := by
  sorry

end determinant_max_value_l4071_407146


namespace decreasing_function_range_l4071_407177

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, x < y → x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → f x > f y) →
  (1 - a ∈ Set.Ioo (-1) 1) →
  (a^2 - 1 ∈ Set.Ioo (-1) 1) →
  f (1 - a) < f (a^2 - 1) →
  0 < a ∧ a < Real.sqrt 2 := by
sorry

end decreasing_function_range_l4071_407177


namespace problem_solution_l4071_407184

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

end problem_solution_l4071_407184


namespace final_collection_is_55_l4071_407117

def museum_donations (initial_collection : ℕ) : ℕ :=
  let guggenheim_donation := 51
  let metropolitan_donation := 2 * guggenheim_donation
  let damaged_sets := 20
  let after_damage := initial_collection - guggenheim_donation - metropolitan_donation - damaged_sets
  let louvre_donation := after_damage / 2
  let after_louvre := after_damage - louvre_donation
  let british_donation := (2 * after_louvre) / 3
  after_louvre - british_donation

theorem final_collection_is_55 :
  museum_donations 500 = 55 := by
  sorry

end final_collection_is_55_l4071_407117


namespace total_stairs_l4071_407112

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2 + 18) ∧
  ravi = (veronica * 3 / 2 : ℕ) ∧  -- Using integer division
  samir + veronica + ravi = 761

theorem total_stairs : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
sorry

end total_stairs_l4071_407112


namespace xyz_absolute_value_l4071_407144

theorem xyz_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq1 : x + 1 / y = y + 1 / z)
  (heq2 : y + 1 / z = z + 1 / x + 1) :
  |x * y * z| = 1 := by sorry

end xyz_absolute_value_l4071_407144


namespace polynomial_value_l4071_407181

/-- A quadratic polynomial of the form a(x^3 - x^2 + 3x) + b(2x^2 + x) + x^3 - 5 -/
def p (a b x : ℝ) : ℝ := a*(x^3 - x^2 + 3*x) + b*(2*x^2 + x) + x^3 - 5

/-- If p(2) = -17, then p(-2) = -1 -/
theorem polynomial_value (a b : ℝ) (h : p a b 2 = -17) : p a b (-2) = -1 := by
  sorry

end polynomial_value_l4071_407181


namespace range_of_a_l4071_407152

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l4071_407152


namespace pascal_third_element_51st_row_l4071_407153

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth element in the nth row of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_third_element_51st_row : 
  pascal_element 51 2 = 1275 :=
sorry

end pascal_third_element_51st_row_l4071_407153


namespace set_A_equals_neg_one_zero_l4071_407199

def A : Set ℤ := {x | x^2 + x ≤ 0}

theorem set_A_equals_neg_one_zero : A = {-1, 0} := by sorry

end set_A_equals_neg_one_zero_l4071_407199


namespace collinearity_necessary_not_sufficient_l4071_407179

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- The statement to be proved -/
theorem collinearity_necessary_not_sufficient :
  ¬(∀ a : ℝ, collinear (a, a^2) (1, 2) → a = 2) ∧
  (a = 2 → collinear (a, a^2) (1, 2)) :=
by sorry

end collinearity_necessary_not_sufficient_l4071_407179


namespace probability_is_correct_l4071_407128

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def needed_stickers : ℕ := 6
def collected_stickers : ℕ := total_stickers - needed_stickers

def probability_complete_collection : ℚ :=
  (Nat.choose needed_stickers needed_stickers * Nat.choose collected_stickers (selected_stickers - needed_stickers)) /
  Nat.choose total_stickers selected_stickers

theorem probability_is_correct : probability_complete_collection = 5 / 442 := by
  sorry

end probability_is_correct_l4071_407128


namespace linear_function_quadrants_l4071_407135

/-- A proportional function with a negative slope -/
structure NegativeSlopeProportionalFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  k_negative : k < 0

/-- The linear function y = 2x + k -/
def linear_function (f : NegativeSlopeProportionalFunction) (x : ℝ) : ℝ := 2 * x + f.k

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Check if a point (x, y) is in a given quadrant -/
def in_quadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- The theorem stating that the linear function passes through Quadrants I, III, and IV -/
theorem linear_function_quadrants (f : NegativeSlopeProportionalFunction) :
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.I) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.III) ∧
  (∃ x y : ℝ, y = linear_function f x ∧ in_quadrant x y Quadrant.IV) :=
sorry

end linear_function_quadrants_l4071_407135


namespace max_value_of_f_l4071_407100

theorem max_value_of_f (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  (∃ (a b : ℝ), f 0 + f 2017 ≤ a ∧ f 0 + f 2017 ≥ b) ∧ 
  (∀ (y : ℝ), f 0 + f 2017 ≤ y → y ≤ 1 + Real.sqrt 2 / 2) := by
  sorry

end max_value_of_f_l4071_407100


namespace quadratic_function_properties_l4071_407109

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- State the theorem
theorem quadratic_function_properties :
  -- f is a quadratic function
  (∃ (a b c : ℝ), ∀ x, f x = a*x^2 + b*x + c) ∧
  -- f'(x) = 2x + 2
  (∀ x, deriv f x = 2*x + 2) ∧
  -- f(x) = 0 has two equal real roots
  (∃! x : ℝ, f x = 0) →
  -- 1. f(x) = x^2 + 2x + 1
  (∀ x, f x = x^2 + 2*x + 1) ∧
  -- 2. The area enclosed by f(x) and the coordinate axes is 1/3
  (∫ x in (-1)..0, f x = 1/3) ∧
  -- 3. The value of t that divides the enclosed area into two equal parts is 1 - 1/32
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧
    ∫ x in (-1)..(-t), f x = ∫ x in (-t)..0, f x ∧
    t = 1 - 1/(2^5)^(1/3)) :=
by sorry

end quadratic_function_properties_l4071_407109


namespace reading_time_per_page_l4071_407141

theorem reading_time_per_page 
  (planned_hours : ℝ) 
  (actual_fraction : ℝ) 
  (pages_read : ℕ) : 
  planned_hours = 3 → 
  actual_fraction = 3/4 → 
  pages_read = 9 → 
  (planned_hours * actual_fraction * 60) / pages_read = 15 := by
sorry

end reading_time_per_page_l4071_407141


namespace solve_system_l4071_407169

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end solve_system_l4071_407169


namespace min_distance_on_feb_9th_l4071_407159

/-- Represents the squared distance between a space probe and Mars as a function of time -/
def D (a b c : ℝ) (t : ℝ) : ℝ := a * t^2 + b * t + c

/-- Theorem stating that the minimum distance occurs on February 9th -/
theorem min_distance_on_feb_9th (a b c : ℝ) :
  D a b c (-9) = 25 →
  D a b c 0 = 4 →
  D a b c 3 = 9 →
  ∃ (t_min : ℝ), t_min = -1 ∧ ∀ (t : ℝ), D a b c t_min ≤ D a b c t :=
sorry

end min_distance_on_feb_9th_l4071_407159


namespace parabola_focus_directrix_distance_l4071_407166

/-- For a parabola with equation y² = -x, the distance from its focus to its directrix is 1/2. -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = -x → 
  ∃ (focus_x focus_y directrix_x : ℝ),
    (focus_x = -1/4 ∧ focus_y = 0) ∧
    directrix_x = 1/4 ∧
    |focus_x - directrix_x| = 1/2 :=
by sorry

end parabola_focus_directrix_distance_l4071_407166


namespace dans_age_l4071_407114

theorem dans_age : 
  ∀ x : ℕ, (x + 18 = 5 * (x - 6)) → x = 12 :=
by
  sorry

end dans_age_l4071_407114


namespace always_odd_l4071_407108

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end always_odd_l4071_407108


namespace max_fourth_power_sum_l4071_407194

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), m = 64 / (4^(1/3)) ∧ a^4 + b^4 + c^4 + d^4 ≤ m :=
by sorry

end max_fourth_power_sum_l4071_407194


namespace unpainted_cubes_count_l4071_407168

/-- Represents a 6x6x6 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)
  (unpainted_columns : Nat)
  (unpainted_rows : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_per_face = 10)
  (h4 : c.unpainted_columns = 2)
  (h5 : c.unpainted_rows = 2) :
  unpainted_cubes c = 168 := by
  sorry

end unpainted_cubes_count_l4071_407168


namespace percentage_equation_l4071_407122

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 617.50) → x = 190 := by
  sorry

end percentage_equation_l4071_407122


namespace red_lettuce_cost_l4071_407171

/-- The amount spent on red lettuce given the total cost and cost of green lettuce -/
def amount_spent_on_red_lettuce (total_cost green_cost : ℕ) : ℕ :=
  total_cost - green_cost

/-- Proof that the amount spent on red lettuce is $6 -/
theorem red_lettuce_cost : amount_spent_on_red_lettuce 14 8 = 6 := by
  sorry

end red_lettuce_cost_l4071_407171


namespace some_number_value_l4071_407185

theorem some_number_value (n m : ℚ) : 
  n = 40 → (n / 20) * (n / m) = 1 → m = 2 := by
  sorry

end some_number_value_l4071_407185


namespace roots_are_irrational_l4071_407189

/-- Given a real number k, this function represents the quadratic equation x^2 - 3kx + 2k^2 - 1 = 0 --/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*k*x + 2*k^2 - 1 = 0

/-- The product of the roots of the quadratic equation is 7 --/
axiom root_product (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ x₁ * x₂ = 7

/-- Definition of an irrational number --/
def is_irrational (x : ℝ) : Prop :=
  ∀ p q : ℤ, q ≠ 0 → x ≠ p / q

/-- The main theorem: the roots of the quadratic equation are irrational --/
theorem roots_are_irrational (k : ℝ) :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ ∧ 
             is_irrational x₁ ∧ is_irrational x₂ :=
by sorry

end roots_are_irrational_l4071_407189


namespace complex_magnitude_problem_l4071_407154

theorem complex_magnitude_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2*I) = Real.sqrt 10 := by
  sorry

end complex_magnitude_problem_l4071_407154


namespace largest_degree_with_asymptote_l4071_407165

-- Define the denominator of our rational function
def q (x : ℝ) : ℝ := 3 * x^6 + 2 * x^3 - x + 4

-- Define a proposition that checks if a polynomial has a horizontal asymptote when divided by q(x)
def has_horizontal_asymptote (p : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x > M, |p x / q x - L| < ε

-- Define a function to get the degree of a polynomial
noncomputable def poly_degree (p : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem largest_degree_with_asymptote :
  ∃ (p : ℝ → ℝ), poly_degree p = 6 ∧ has_horizontal_asymptote p ∧
  ∀ (p' : ℝ → ℝ), poly_degree p' > 6 → ¬(has_horizontal_asymptote p') :=
sorry

end largest_degree_with_asymptote_l4071_407165


namespace trigonometric_identity_l4071_407125

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) 
  (hyp : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) : 
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end trigonometric_identity_l4071_407125


namespace birds_flew_up_l4071_407127

/-- The number of birds that flew up to a tree -/
theorem birds_flew_up (initial : ℕ) (total : ℕ) (h1 : initial = 14) (h2 : total = 35) :
  total - initial = 21 := by
  sorry

end birds_flew_up_l4071_407127


namespace jane_mean_score_l4071_407151

def jane_scores : List ℝ := [96, 95, 90, 87, 91, 75]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℝ) = 89 := by
  sorry

end jane_mean_score_l4071_407151


namespace jogger_difference_l4071_407143

/-- The number of joggers bought by each person -/
structure Joggers where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the problem -/
def jogger_problem (j : Joggers) : Prop :=
  j.christopher = 20 * j.tyson ∧
  j.christopher = 80 ∧
  j.alexander = j.tyson + 22

/-- The theorem to prove -/
theorem jogger_difference (j : Joggers) (h : jogger_problem j) :
  j.christopher - j.alexander = 54 := by
  sorry

end jogger_difference_l4071_407143


namespace max_remainder_eleven_l4071_407172

theorem max_remainder_eleven (y : ℕ+) : ∃ (q r : ℕ), y = 11 * q + r ∧ r < 11 ∧ r ≤ 10 :=
sorry

end max_remainder_eleven_l4071_407172


namespace correct_calculation_l4071_407193

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end correct_calculation_l4071_407193


namespace triangle_height_equals_30_l4071_407137

/-- Given a rectangle with perimeter 60 cm and a right triangle with base 15 cm,
    if their areas are equal, then the height of the triangle is 30 cm. -/
theorem triangle_height_equals_30 (rectangle_perimeter : ℝ) (triangle_base : ℝ) (h : ℝ) :
  rectangle_perimeter = 60 →
  triangle_base = 15 →
  (rectangle_perimeter / 4) * (rectangle_perimeter / 4) = (1 / 2) * triangle_base * h →
  h = 30 := by
  sorry

end triangle_height_equals_30_l4071_407137


namespace race_heartbeats_l4071_407113

/-- The number of heartbeats during a race, given the race distance, heart rate, and pace. -/
def heartbeats_during_race (distance : ℕ) (heart_rate : ℕ) (pace : ℕ) : ℕ :=
  distance * pace * heart_rate

/-- Theorem stating that the number of heartbeats during a 30-mile race
    with a heart rate of 160 beats per minute and a pace of 6 minutes per mile
    is equal to 28800. -/
theorem race_heartbeats :
  heartbeats_during_race 30 160 6 = 28800 := by
  sorry

#eval heartbeats_during_race 30 160 6

end race_heartbeats_l4071_407113


namespace arithmetic_sequence_sum_l4071_407175

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by sorry

end arithmetic_sequence_sum_l4071_407175


namespace sum_of_cubes_l4071_407198

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) : a^3 + b^3 = 638 := by
  sorry

end sum_of_cubes_l4071_407198


namespace trig_expression_equals_one_l4071_407106

theorem trig_expression_equals_one :
  (Real.sqrt 3 * Real.sin (20 * π / 180) + Real.sin (70 * π / 180)) /
  Real.sqrt (2 - 2 * Real.cos (100 * π / 180)) = 1 := by
  sorry

end trig_expression_equals_one_l4071_407106


namespace binomial_coefficient_22_5_l4071_407196

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end binomial_coefficient_22_5_l4071_407196


namespace imaginary_unit_sum_l4071_407118

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → Complex.abs (-i) + i^2018 = 0 := by
  sorry

end imaginary_unit_sum_l4071_407118


namespace sequence_periodicity_l4071_407123

def isEventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ (n k : ℕ), k > 0 ∧ ∀ m, m ≥ n → a (m + k) = a m

theorem sequence_periodicity (a : ℕ → ℕ) 
    (h : ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
    isEventuallyPeriodic a := by
  sorry

end sequence_periodicity_l4071_407123


namespace quadratic_solution_sum_l4071_407142

theorem quadratic_solution_sum (a b : ℕ+) : 
  (∃ x : ℝ, x^2 + 16*x = 96 ∧ x = Real.sqrt a - b) → a + b = 168 := by
  sorry

end quadratic_solution_sum_l4071_407142


namespace subtraction_multiplication_equality_l4071_407182

theorem subtraction_multiplication_equality : (3.65 - 1.25) * 2 = 4.80 := by
  sorry

end subtraction_multiplication_equality_l4071_407182


namespace final_rope_length_l4071_407148

/-- Calculates the final length of a rope made by tying multiple pieces together -/
theorem final_rope_length
  (rope_lengths : List ℝ)
  (knot_loss : ℝ)
  (h_lengths : rope_lengths = [8, 20, 2, 2, 2, 7])
  (h_knot_loss : knot_loss = 1.2)
  : (rope_lengths.sum - knot_loss * (rope_lengths.length - 1 : ℝ)) = 35 := by
  sorry

end final_rope_length_l4071_407148


namespace square_of_1007_l4071_407121

theorem square_of_1007 : (1007 : ℕ) ^ 2 = 1014049 := by
  sorry

end square_of_1007_l4071_407121


namespace five_chairs_cost_l4071_407160

/-- The cost of a single plastic chair -/
def chair_cost : ℝ := sorry

/-- The cost of a portable table -/
def table_cost : ℝ := sorry

/-- Three chairs cost the same as one table -/
axiom chair_table_relation : 3 * chair_cost = table_cost

/-- One table and two chairs cost $55 -/
axiom total_cost : table_cost + 2 * chair_cost = 55

/-- The cost of five plastic chairs is $55 -/
theorem five_chairs_cost : 5 * chair_cost = 55 := by sorry

end five_chairs_cost_l4071_407160


namespace function_inequality_implies_m_bound_l4071_407103

open Real

theorem function_inequality_implies_m_bound (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 ℯ ∧ m * (x₀ - 1 / x₀) - 2 * log x₀ < -m / x₀) →
  m < 2 / ℯ := by
  sorry

end function_inequality_implies_m_bound_l4071_407103


namespace height_range_selection_probability_overall_avg_height_l4071_407156

-- Define the track and field team
def num_male : ℕ := 12
def num_female : ℕ := 8
def total_athletes : ℕ := num_male + num_female
def max_height : ℕ := 190
def min_height : ℕ := 160
def avg_height_male : ℝ := 175
def avg_height_female : ℝ := 165

-- Theorem 1: The range of heights is 30cm
theorem height_range : max_height - min_height = 30 := by sorry

-- Theorem 2: The probability of an athlete being selected in a random sample of 10 is 1/2
theorem selection_probability : (10 : ℝ) / total_athletes = (1 : ℝ) / 2 := by sorry

-- Theorem 3: The overall average height of the team is 171cm
theorem overall_avg_height :
  (num_male : ℝ) / total_athletes * avg_height_male +
  (num_female : ℝ) / total_athletes * avg_height_female = 171 := by sorry

end height_range_selection_probability_overall_avg_height_l4071_407156


namespace range_of_f_l4071_407164

def f (x : ℝ) := x^2 - 2*x + 4

theorem range_of_f :
  ∀ y ∈ Set.Icc 3 7, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, ∃ y ∈ Set.Icc 3 7, f x = y :=
sorry

end range_of_f_l4071_407164


namespace exist_six_numbers_l4071_407132

theorem exist_six_numbers : ∃ (a b c d e f : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f) = 2012 := by
  sorry

end exist_six_numbers_l4071_407132


namespace olivers_card_collection_l4071_407195

/-- Oliver's card collection problem -/
theorem olivers_card_collection :
  ∀ (alien_baseball monster_club battle_gremlins : ℕ),
  monster_club = 2 * alien_baseball →
  battle_gremlins = 48 →
  battle_gremlins = 3 * alien_baseball →
  monster_club = 32 := by
sorry

end olivers_card_collection_l4071_407195


namespace smallest_t_for_complete_circle_l4071_407161

/-- The smallest value of t such that when r = sin θ is plotted for 0 ≤ θ ≤ t,
    the resulting graph represents the entire circle is π. -/
theorem smallest_t_for_complete_circle : 
  ∃ t : ℝ, t > 0 ∧ 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = Real.sin θ) ∧
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ x = (Real.sin θ) * (Real.cos θ) ∧ y = Real.sin θ) ∧
  (∀ t' : ℝ, t' < t → 
    ∃ x y : ℝ, x^2 + y^2 ≤ 1 ∧ 
    ∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t' → x ≠ (Real.sin θ) * (Real.cos θ) ∨ y ≠ Real.sin θ)) ∧
  t = Real.pi :=
by sorry

end smallest_t_for_complete_circle_l4071_407161


namespace double_inequality_proof_l4071_407139

theorem double_inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end double_inequality_proof_l4071_407139
