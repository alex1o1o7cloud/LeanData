import Mathlib

namespace NUMINAMATH_CALUDE_root_product_theorem_l3930_393096

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^5 + 3*x^2 + 1

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 5

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (hroots : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = 131 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3930_393096


namespace NUMINAMATH_CALUDE_pool_capacity_l3930_393008

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C - 0.5 * C = 300) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l3930_393008


namespace NUMINAMATH_CALUDE_dividend_calculation_l3930_393030

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 15) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 140 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3930_393030


namespace NUMINAMATH_CALUDE_stratified_sampling_probability_l3930_393092

theorem stratified_sampling_probability 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (third_year : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = first_year + second_year + third_year)
  (h2 : total_students = 600)
  (h3 : first_year = 100)
  (h4 : second_year = 200)
  (h5 : third_year = 300)
  (h6 : selected = 30) :
  (selected : ℚ) / (total_students : ℚ) = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_probability_l3930_393092


namespace NUMINAMATH_CALUDE_chelsea_cupcake_time_l3930_393097

/-- Calculates the total time Chelsea spent making and decorating cupcakes --/
def total_cupcake_time (num_batches : ℕ) 
                       (bake_time_per_batch : ℕ) 
                       (ice_time_per_batch : ℕ)
                       (cupcakes_per_batch : ℕ)
                       (decor_time_per_cupcake : List ℕ) : ℕ :=
  let base_time := num_batches * (bake_time_per_batch + ice_time_per_batch)
  let decor_time := (List.map (· * cupcakes_per_batch) decor_time_per_cupcake).sum
  base_time + decor_time

/-- Theorem stating that Chelsea's total time making and decorating cupcakes is 542 minutes --/
theorem chelsea_cupcake_time : 
  total_cupcake_time 4 20 30 6 [10, 15, 12, 20] = 542 := by
  sorry


end NUMINAMATH_CALUDE_chelsea_cupcake_time_l3930_393097


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3930_393011

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3930_393011


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_product_l3930_393024

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

def product_of_first_six : ℕ := (List.range 6).map (· + 1) |>.prod

theorem smallest_binary_divisible_by_product :
  let n : ℕ := 1111111110000
  (is_binary_number n) ∧
  (n % product_of_first_six = 0) ∧
  (∀ m : ℕ, m < n → is_binary_number m → m % product_of_first_six ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_product_l3930_393024


namespace NUMINAMATH_CALUDE_min_value_sum_l3930_393068

theorem min_value_sum (a b : ℝ) (h : a^2 + 2*b^2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (∃ (u v : ℝ), u^2 + 2*v^2 = 6 ∧ m = u + v) ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l3930_393068


namespace NUMINAMATH_CALUDE_min_value_theorem_l3930_393085

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 / y = 3) :
  ∀ z, z = 2 / x + y → z ≥ 8 / 3 ∧ ∃ w, w = 2 / x + y ∧ w = 8 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3930_393085


namespace NUMINAMATH_CALUDE_constant_term_proof_l3930_393013

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to find the maximum coefficient term
def max_coeff_term (n : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constant_term (n : ℕ) : ℕ := sorry

theorem constant_term_proof (n : ℕ) :
  max_coeff_term n = 6 → constant_term n = 180 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_proof_l3930_393013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3930_393034

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  product_condition : a 7 * a 11 = 6
  sum_condition : a 4 + a 14 = 5

/-- The common difference of an arithmetic sequence is either 1/4 or -1/4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  (∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ (d = 1/4 ∨ d = -1/4)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3930_393034


namespace NUMINAMATH_CALUDE_remainder_17_pow_2090_mod_23_l3930_393083

theorem remainder_17_pow_2090_mod_23 : 17^2090 % 23 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2090_mod_23_l3930_393083


namespace NUMINAMATH_CALUDE_division_remainder_l3930_393022

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 190 →
  divisor = 21 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3930_393022


namespace NUMINAMATH_CALUDE_spinner_probability_l3930_393045

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3930_393045


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3930_393079

def M : Set ℝ := {x | (2 - x) / (x + 1) ≥ 0}
def N : Set ℝ := {x | ∃ y, y = Real.log x}

theorem set_intersection_equality : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3930_393079


namespace NUMINAMATH_CALUDE_remaining_lawn_after_one_hour_l3930_393061

/-- Given that Mary can mow the entire lawn in 3 hours, 
    this function calculates the fraction of the lawn mowed in a given time. -/
def fraction_mowed (hours : ℚ) : ℚ := hours / 3

/-- This theorem states that if Mary works for 1 hour, 
    then 2/3 of the lawn remains to be mowed. -/
theorem remaining_lawn_after_one_hour : 
  1 - (fraction_mowed 1) = 2/3 := by sorry

end NUMINAMATH_CALUDE_remaining_lawn_after_one_hour_l3930_393061


namespace NUMINAMATH_CALUDE_simplify_sqrt_three_l3930_393038

theorem simplify_sqrt_three : 3 * Real.sqrt 3 - 2 * Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_three_l3930_393038


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3930_393042

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 1 / (x - 2) ≥ 4 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 2) :
  ∃ x₀ > 2, x₀ + 1 / (x₀ - 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l3930_393042


namespace NUMINAMATH_CALUDE_complex_square_roots_l3930_393088

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -104 + 63 * I ∧ (5 + 8 * I) ^ 2 = -104 + 63 * I → 
  (-5 - 8 * I) ^ 2 = -104 + 63 * I := by
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l3930_393088


namespace NUMINAMATH_CALUDE_chlorine_discount_is_20_percent_l3930_393027

def original_chlorine_price : ℝ := 10
def original_soap_price : ℝ := 16
def soap_discount : ℝ := 0.25
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5
def total_savings : ℝ := 26

theorem chlorine_discount_is_20_percent :
  ∃ (chlorine_discount : ℝ),
    chlorine_discount = 0.20 ∧
    (chlorine_quantity : ℝ) * original_chlorine_price * (1 - chlorine_discount) +
    soap_quantity * original_soap_price * (1 - soap_discount) =
    chlorine_quantity * original_chlorine_price +
    soap_quantity * original_soap_price - total_savings :=
by sorry

end NUMINAMATH_CALUDE_chlorine_discount_is_20_percent_l3930_393027


namespace NUMINAMATH_CALUDE_jenny_recycling_problem_l3930_393015

/-- The weight of each can in ounces -/
def can_weight : ℚ := 2

theorem jenny_recycling_problem :
  let total_weight : ℚ := 100
  let bottle_weight : ℚ := 6
  let num_cans : ℚ := 20
  let cents_per_bottle : ℚ := 10
  let cents_per_can : ℚ := 3
  let total_cents : ℚ := 160
  (total_weight - num_cans * can_weight) / bottle_weight * cents_per_bottle + num_cans * cents_per_can = total_cents :=
by sorry

end NUMINAMATH_CALUDE_jenny_recycling_problem_l3930_393015


namespace NUMINAMATH_CALUDE_mirror_area_l3930_393021

theorem mirror_area (frame_length frame_width frame_side_width : ℝ) 
  (h1 : frame_length = 80)
  (h2 : frame_width = 60)
  (h3 : frame_side_width = 10) :
  (frame_length - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 2400 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l3930_393021


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l3930_393049

/-- The equation of a reflected light ray given specific conditions -/
theorem reflected_ray_equation :
  let origin : ℝ × ℝ := (0, 0)
  let incident_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 5 = 0
  let reflection_point : ℝ × ℝ := (1, 3)
  let reflected_line : ℝ → ℝ → Prop := λ x y => x - 5 * y + 14 = 0
  ∀ (x y : ℝ), reflected_line x y ↔
    ∃ (p : ℝ × ℝ),
      incident_line p.1 p.2 ∧
      (p.1 - origin.1) * (y - p.2) = (x - p.1) * (p.2 - origin.2) ∧
      (p.1 - reflection_point.1) * (y - p.2) = (x - p.1) * (p.2 - reflection_point.2) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l3930_393049


namespace NUMINAMATH_CALUDE_exponential_inequality_l3930_393059

theorem exponential_inequality (x y a b : ℝ) 
  (h1 : x > y) (h2 : y > 1) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : 
  a^x < b^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3930_393059


namespace NUMINAMATH_CALUDE_chicken_problem_l3930_393010

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_skylar = 3 * chickens_colten - 4 ∧
    chickens_quentin = 2 * chickens_skylar + 32 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total :=
by
  sorry

#check chicken_problem

end NUMINAMATH_CALUDE_chicken_problem_l3930_393010


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3930_393001

theorem gcd_of_specific_numbers : Nat.gcd 333333 9999999 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3930_393001


namespace NUMINAMATH_CALUDE_erwin_chocolate_consumption_l3930_393089

/-- Represents Erwin's chocolate consumption pattern and total chocolates eaten --/
structure ChocolateConsumption where
  weekday_consumption : ℕ  -- chocolates eaten per weekday
  weekend_consumption : ℕ  -- chocolates eaten per weekend day
  total_chocolates : ℕ     -- total chocolates eaten

/-- Calculates the number of weeks it took to eat all chocolates --/
def weeks_to_finish (consumption : ChocolateConsumption) : ℚ :=
  consumption.total_chocolates / (5 * consumption.weekday_consumption + 2 * consumption.weekend_consumption)

/-- Theorem stating it took Erwin 2 weeks to finish the chocolates --/
theorem erwin_chocolate_consumption :
  let consumption : ChocolateConsumption := {
    weekday_consumption := 2,
    weekend_consumption := 1,
    total_chocolates := 24
  }
  weeks_to_finish consumption = 2 := by sorry

end NUMINAMATH_CALUDE_erwin_chocolate_consumption_l3930_393089


namespace NUMINAMATH_CALUDE_not_prime_3999991_l3930_393076

theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
  sorry

end NUMINAMATH_CALUDE_not_prime_3999991_l3930_393076


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_299_l3930_393069

theorem greatest_prime_factor_of_299 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 299 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 299 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_299_l3930_393069


namespace NUMINAMATH_CALUDE_platform_length_l3930_393056

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 72 →
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 380 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_l3930_393056


namespace NUMINAMATH_CALUDE_digit_equation_sum_l3930_393055

theorem digit_equation_sum : 
  ∀ (E M V Y : ℕ),
  (E < 10) → (M < 10) → (V < 10) → (Y < 10) →
  (V ≥ 1) →
  (Y ≠ 0) → (M ≠ 0) →
  (E ≠ M) → (E ≠ V) → (E ≠ Y) → 
  (M ≠ V) → (M ≠ Y) → 
  (V ≠ Y) →
  ((10 * Y + E) * (10 * M + E) = 111 * V) →
  (E + M + V + Y = 21) := by
sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l3930_393055


namespace NUMINAMATH_CALUDE_boy_speed_around_square_l3930_393086

/-- The speed of a boy running around a square field -/
theorem boy_speed_around_square (side_length : ℝ) (time : ℝ) : 
  side_length = 20 → time = 24 → 
  (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boy_speed_around_square_l3930_393086


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3930_393075

/-- Proves that the interest rate is 8% per annum given the conditions of the problem -/
theorem interest_rate_calculation (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) :
  P = 2500 →
  t = 8 →
  I = P - 900 →
  I = P * r * t / 100 →
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3930_393075


namespace NUMINAMATH_CALUDE_remaining_bird_families_l3930_393081

/-- The number of bird families left near the mountain after some flew away -/
def bird_families_left (initial : ℕ) (flew_away : ℕ) : ℕ :=
  initial - flew_away

/-- Theorem stating that 237 bird families were left near the mountain -/
theorem remaining_bird_families :
  bird_families_left 709 472 = 237 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bird_families_l3930_393081


namespace NUMINAMATH_CALUDE_product_inequality_l3930_393031

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 + x + y^2) * (1 + y + x^2) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3930_393031


namespace NUMINAMATH_CALUDE_equation_solution_l3930_393032

theorem equation_solution (x : ℝ) : 4 / (1 + 3/x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3930_393032


namespace NUMINAMATH_CALUDE_expected_rounds_four_players_l3930_393054

/-- Represents the expected number of rounds in a rock-paper-scissors game -/
def expected_rounds (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 3/2
  | 3 => 9/4
  | 4 => 81/14
  | _ => 0  -- undefined for n > 4

/-- The rules of the rock-paper-scissors game -/
axiom game_rules : ∀ (n : ℕ), n > 0 → n ≤ 4 → 
  expected_rounds n = 
    if n = 1 then 0
    else if n = 2 then 3/2
    else if n = 3 then 9/4
    else 81/14

/-- The main theorem: expected number of rounds for 4 players is 81/14 -/
theorem expected_rounds_four_players :
  expected_rounds 4 = 81/14 :=
by
  exact game_rules 4 (by norm_num) (by norm_num)


end NUMINAMATH_CALUDE_expected_rounds_four_players_l3930_393054


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3930_393060

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3930_393060


namespace NUMINAMATH_CALUDE_book_price_difference_l3930_393087

def necklace_price : ℕ := 34
def spending_limit : ℕ := 70
def overspent_amount : ℕ := 3

theorem book_price_difference (book_price : ℕ) : 
  book_price > necklace_price →
  book_price + necklace_price = spending_limit + overspent_amount →
  book_price - necklace_price = 5 := by
sorry

end NUMINAMATH_CALUDE_book_price_difference_l3930_393087


namespace NUMINAMATH_CALUDE_min_ab_is_16_l3930_393071

-- Define the points
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def B (b : ℝ) : ℝ × ℝ := (0, b)
def C : ℝ × ℝ := (-2, -2)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem min_ab_is_16 (a b : ℝ) (h1 : a * b > 0) (h2 : collinear (A a) (B b) C) :
  ∀ x y : ℝ, x * y > 0 → collinear (A x) (B y) C → a * b ≤ x * y ∧ a * b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_ab_is_16_l3930_393071


namespace NUMINAMATH_CALUDE_carnival_tickets_l3930_393009

theorem carnival_tickets (tickets : ℕ) (extra : ℕ) : 
  let F := Nat.minFac (tickets + extra)
  F ∣ (tickets + extra) ∧ ¬(F ∣ tickets) →
  F = 3 :=
by
  sorry

#check carnival_tickets 865 8

end NUMINAMATH_CALUDE_carnival_tickets_l3930_393009


namespace NUMINAMATH_CALUDE_segment_ratio_l3930_393082

/-- Given five consecutive points on a line, prove that the ratio of two specific segments is 2:1 -/
theorem segment_ratio (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (d - e = 4) ∧        -- de = 4
  (a - b = 5) ∧        -- ab = 5
  (a - c = 11) ∧       -- ac = 11
  (a - e = 18) →       -- ae = 18
  (c - b) / (d - c) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_segment_ratio_l3930_393082


namespace NUMINAMATH_CALUDE_work_completion_time_l3930_393066

theorem work_completion_time (p q : ℕ) (work_left : ℚ) : 
  p = 15 → q = 20 → work_left = 8/15 → 
  (1 : ℚ) - (1/p + 1/q) * (days_worked : ℚ) = work_left → 
  days_worked = 4 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3930_393066


namespace NUMINAMATH_CALUDE_derivative_of_f_l3930_393073

-- Define the function f(x) = (5x - 4)^3
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

-- State the theorem that the derivative of f(x) is 15(5x - 4)^2
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 15 * (5 * x - 4) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3930_393073


namespace NUMINAMATH_CALUDE_conference_teams_l3930_393098

/-- The number of games played in a conference where each team plays every other team twice -/
def games_played (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 12 teams in the conference -/
theorem conference_teams : ∃ n : ℕ, n > 0 ∧ games_played n = 132 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_conference_teams_l3930_393098


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l3930_393044

def driving_distances : List Int := [-5, 3, 6, -4, 7, -2]

def base_fare : Nat := 8
def extra_fare_per_km : Nat := 2
def base_distance : Nat := 3

def cumulative_distance (n : Nat) : Int :=
  (driving_distances.take n).sum

def trip_fare (distance : Int) : Nat :=
  base_fare + max 0 (distance.natAbs - base_distance) * extra_fare_per_km

def total_earnings : Nat :=
  (driving_distances.map trip_fare).sum

theorem taxi_driver_theorem :
  (cumulative_distance 4 = 0) ∧
  (cumulative_distance driving_distances.length = 5) ∧
  (total_earnings = 68) := by
  sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l3930_393044


namespace NUMINAMATH_CALUDE_people_not_playing_sports_l3930_393099

theorem people_not_playing_sports (total_people : ℕ) (tennis_players : ℕ) (baseball_players : ℕ) (both_players : ℕ) :
  total_people = 310 →
  tennis_players = 138 →
  baseball_players = 255 →
  both_players = 94 →
  total_people - (tennis_players + baseball_players - both_players) = 11 :=
by sorry

end NUMINAMATH_CALUDE_people_not_playing_sports_l3930_393099


namespace NUMINAMATH_CALUDE_veg_eaters_count_l3930_393006

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  onlyNonVeg : ℕ
  bothVegAndNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (diet : FamilyDiet) : ℕ :=
  diet.onlyVeg + diet.bothVegAndNonVeg

/-- Theorem stating that for a given family diet, the total number of vegetarian eaters
    is equal to the sum of those who eat only vegetarian and those who eat both -/
theorem veg_eaters_count (diet : FamilyDiet) :
  totalVegEaters diet = diet.onlyVeg + diet.bothVegAndNonVeg := by
  sorry

/-- Example family with the given dietary information -/
def exampleFamily : FamilyDiet where
  onlyVeg := 19
  onlyNonVeg := 9
  bothVegAndNonVeg := 12

#eval totalVegEaters exampleFamily

end NUMINAMATH_CALUDE_veg_eaters_count_l3930_393006


namespace NUMINAMATH_CALUDE_divisible_by_77_l3930_393094

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_77_l3930_393094


namespace NUMINAMATH_CALUDE_six_inch_gold_cube_value_l3930_393039

/-- Represents the properties of a gold cube -/
structure GoldCube where
  side_length : ℝ
  value : ℝ

/-- Calculates the volume of a cube given its side length -/
def cube_volume (side : ℝ) : ℝ := side ^ 3

/-- Theorem: The value of a 6-inch gold cube is $2700 -/
theorem six_inch_gold_cube_value :
  let cube4 : GoldCube := { side_length := 4, value := 800 }
  let cube6 : GoldCube := { side_length := 6, value := 2700 }
  cube6.value = cube4.value * (cube_volume cube6.side_length / cube_volume cube4.side_length) := by
  sorry

end NUMINAMATH_CALUDE_six_inch_gold_cube_value_l3930_393039


namespace NUMINAMATH_CALUDE_sum_after_removing_terms_l3930_393019

theorem sum_after_removing_terms : 
  let sequence := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let removed_terms := [1/12, 1/15]
  let remaining_terms := sequence.filter (λ x => x ∉ removed_terms)
  (remaining_terms.sum = 1) := by sorry

end NUMINAMATH_CALUDE_sum_after_removing_terms_l3930_393019


namespace NUMINAMATH_CALUDE_worker_days_calculation_l3930_393037

theorem worker_days_calculation (wages_group1 wages_group2 : ℚ)
  (workers_group1 workers_group2 : ℕ) (days_group2 : ℕ) :
  wages_group1 = 9450 →
  wages_group2 = 9975 →
  workers_group1 = 15 →
  workers_group2 = 19 →
  days_group2 = 5 →
  ∃ (days_group1 : ℕ),
    (wages_group1 / (workers_group1 * days_group1 : ℚ)) =
    (wages_group2 / (workers_group2 * days_group2 : ℚ)) ∧
    days_group1 = 6 :=
by sorry

end NUMINAMATH_CALUDE_worker_days_calculation_l3930_393037


namespace NUMINAMATH_CALUDE_shop_profit_per_tshirt_l3930_393050

/-- The amount the shop makes off each t-shirt -/
def T : ℝ := 25

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 115

/-- The number of t-shirts sold -/
def t_shirts_sold : ℕ := 113

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 78

/-- The price difference between a jersey and a t-shirt -/
def price_difference : ℝ := 90

theorem shop_profit_per_tshirt :
  T = 25 ∧
  jersey_profit = 115 ∧
  t_shirts_sold = 113 ∧
  jerseys_sold = 78 ∧
  jersey_profit = T + price_difference ∧
  price_difference = 90 →
  T = 25 := by sorry

end NUMINAMATH_CALUDE_shop_profit_per_tshirt_l3930_393050


namespace NUMINAMATH_CALUDE_convenience_store_soda_sales_l3930_393072

/-- Represents the weekly soda sales of a convenience store -/
structure SodaSales where
  gallons_per_box : ℕ
  cost_per_box : ℕ
  weekly_syrup_cost : ℕ

/-- Calculates the number of gallons of soda sold per week -/
def gallons_sold_per_week (s : SodaSales) : ℕ :=
  (s.weekly_syrup_cost / s.cost_per_box) * s.gallons_per_box

/-- Theorem: Given the conditions, the store sells 180 gallons of soda per week -/
theorem convenience_store_soda_sales :
  ∀ (s : SodaSales),
    s.gallons_per_box = 30 →
    s.cost_per_box = 40 →
    s.weekly_syrup_cost = 240 →
    gallons_sold_per_week s = 180 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_soda_sales_l3930_393072


namespace NUMINAMATH_CALUDE_power_of_power_l3930_393064

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l3930_393064


namespace NUMINAMATH_CALUDE_ratio_theorem_l3930_393095

theorem ratio_theorem (a b c d r : ℝ) 
  (h1 : (b + c + d) / a = r)
  (h2 : (a + c + d) / b = r)
  (h3 : (a + b + d) / c = r)
  (h4 : (a + b + c) / d = r)
  : r = 3 ∨ r = -1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l3930_393095


namespace NUMINAMATH_CALUDE_rain_duration_theorem_l3930_393080

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem rain_duration_theorem :
  total_rain_duration rain_duration_day1
    (rain_duration_day2 rain_duration_day1)
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_theorem_l3930_393080


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l3930_393012

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (b c x : ℂ) : ℂ := x^2 + b*x + c

theorem quadratic_root_implies_coefficients :
  ∀ (b c : ℝ), quadratic b c (2 - i) = 0 → b = -4 ∧ c = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l3930_393012


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3930_393070

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallelLines : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the "lies within" relation between a line and a plane
variable (liesWithin : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : parallelLines a b) 
  (h2 : parallelLinePlane a α) :
  parallelLinePlane b α ∨ liesWithin b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3930_393070


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l3930_393016

/-- The simple interest rate problem -/
theorem simple_interest_rate_problem 
  (simple_interest : ℝ) 
  (principal : ℝ) 
  (time : ℝ) 
  (h1 : simple_interest = 16.32)
  (h2 : principal = 34)
  (h3 : time = 8)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l3930_393016


namespace NUMINAMATH_CALUDE_percent_relationship_l3930_393002

theorem percent_relationship (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) :
  y / x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percent_relationship_l3930_393002


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3930_393000

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  let P : ℝ × ℝ := (-2, 1)
  reflect_x P = (-2, -1) := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3930_393000


namespace NUMINAMATH_CALUDE_inequality_solution_and_geometric_mean_l3930_393078

theorem inequality_solution_and_geometric_mean (a b m : ℝ) : 
  (∀ x, (x - 2) / (a * x + b) > 0 ↔ -1 < x ∧ x < 2) →
  m^2 = a * b →
  (3 * m^2 * a) / (a^3 + 2 * b^3) = 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_geometric_mean_l3930_393078


namespace NUMINAMATH_CALUDE_whitney_spent_440_l3930_393065

/-- Calculates the total amount spent by Whitney on books and magazines. -/
def whitneyTotalSpent (whaleBooks fishBooks sharkBooks magazines : ℕ) 
  (whaleCost fishCost sharkCost magazineCost : ℕ) : ℕ :=
  whaleBooks * whaleCost + fishBooks * fishCost + sharkBooks * sharkCost + magazines * magazineCost

/-- Proves that Whitney spent $440 in total. -/
theorem whitney_spent_440 : 
  whitneyTotalSpent 15 12 5 8 14 13 10 3 = 440 := by
  sorry

end NUMINAMATH_CALUDE_whitney_spent_440_l3930_393065


namespace NUMINAMATH_CALUDE_dance_group_composition_l3930_393018

/-- Represents a dance group --/
structure DanceGroup where
  boy_dancers : ℕ
  girl_dancers : ℕ
  boy_escorts : ℕ
  girl_escorts : ℕ

/-- The problem statement --/
theorem dance_group_composition 
  (group_a group_b : DanceGroup)
  (h1 : group_a.boy_dancers + group_a.girl_dancers = group_b.boy_dancers + group_b.girl_dancers + 1)
  (h2 : group_a.boy_escorts + group_a.girl_escorts = group_b.boy_escorts + group_b.girl_escorts + 1)
  (h3 : group_a.boy_dancers + group_b.boy_dancers = group_a.girl_dancers + group_b.girl_dancers + 1)
  (h4 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 484)
  (h5 : (group_a.boy_dancers + group_a.boy_escorts) * (group_b.girl_dancers + group_b.girl_escorts) +
        (group_b.boy_dancers + group_b.boy_escorts) * (group_a.girl_dancers + group_a.girl_escorts) = 246)
  (h6 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 306)
  (h7 : group_a.boy_dancers * group_a.girl_dancers + group_b.boy_dancers * group_b.girl_dancers = 150)
  (h8 : let total := group_a.boy_dancers + group_a.girl_dancers + group_a.boy_escorts + group_a.girl_escorts +
                     group_b.boy_dancers + group_b.girl_dancers + group_b.boy_escorts + group_b.girl_escorts
        (total * (total - 1)) / 2 = 946) :
  group_a = { boy_dancers := 8, girl_dancers := 10, boy_escorts := 2, girl_escorts := 3 } ∧
  group_b = { boy_dancers := 10, girl_dancers := 7, boy_escorts := 2, girl_escorts := 2 } :=
by sorry

end NUMINAMATH_CALUDE_dance_group_composition_l3930_393018


namespace NUMINAMATH_CALUDE_johanna_turtle_loss_l3930_393084

/-- The fraction of turtles Johanna loses -/
def johanna_loss_fraction (owen_initial : ℕ) (johanna_diff : ℕ) (owen_final : ℕ) : ℚ :=
  let owen_after_month := 2 * owen_initial
  let johanna_initial := owen_initial - johanna_diff
  1 - (owen_final - owen_after_month) / johanna_initial

theorem johanna_turtle_loss 
  (owen_initial : ℕ) 
  (johanna_diff : ℕ) 
  (owen_final : ℕ) 
  (h1 : owen_initial = 21)
  (h2 : johanna_diff = 5)
  (h3 : owen_final = 50) :
  johanna_loss_fraction owen_initial johanna_diff owen_final = 1/2 := by
  sorry

#eval johanna_loss_fraction 21 5 50

end NUMINAMATH_CALUDE_johanna_turtle_loss_l3930_393084


namespace NUMINAMATH_CALUDE_sum_and_product_positive_iff_both_positive_l3930_393035

theorem sum_and_product_positive_iff_both_positive (a b : ℝ) :
  (a + b > 0 ∧ a * b > 0) ↔ (a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_positive_iff_both_positive_l3930_393035


namespace NUMINAMATH_CALUDE_factorization_equality_l3930_393052

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3930_393052


namespace NUMINAMATH_CALUDE_five_times_seven_and_two_fifths_l3930_393048

theorem five_times_seven_and_two_fifths (x : ℚ) : x = 5 * (7 + 2/5) → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_five_times_seven_and_two_fifths_l3930_393048


namespace NUMINAMATH_CALUDE_rs_length_l3930_393046

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let XY := dist t.X t.Y
  let YZ := dist t.Y t.Z
  let ZX := dist t.Z t.X
  XY = 13 ∧ YZ = 14 ∧ ZX = 15

-- Define the median XM
def isMedian (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.X M = dist M ((t.Y.1 + t.Z.1, t.Y.2 + t.Z.2))

-- Define points G and F
def isOnSide (A B P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ P = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

-- Define angle bisectors
def isAngleBisector (t : Triangle) (P : ℝ × ℝ) (V : ℝ × ℝ) : Prop :=
  ∃ G F : ℝ × ℝ, 
    isOnSide t.Z t.X G ∧ 
    isOnSide t.X t.Y F ∧
    dist t.Y G * dist t.Z V = dist t.Z G * dist t.Y V ∧
    dist t.Z F * dist t.X V = dist t.X F * dist t.Z V

-- Define the theorem
theorem rs_length (t : Triangle) (M R S : ℝ × ℝ) :
  isValidTriangle t →
  isMedian t M →
  isAngleBisector t R t.Y →
  isAngleBisector t S t.Z →
  dist R S = 129 / 203 :=
sorry


end NUMINAMATH_CALUDE_rs_length_l3930_393046


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3930_393063

/-- The total percentage allocated to categories other than basic astrophysics -/
def other_categories_percentage : ℝ := 98

/-- The total degrees in a circle -/
def circle_degrees : ℝ := 360

/-- The percentage allocated to basic astrophysics -/
def basic_astrophysics_percentage : ℝ := 100 - other_categories_percentage

theorem basic_astrophysics_degrees :
  (basic_astrophysics_percentage / 100) * circle_degrees = 7.2 := by sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3930_393063


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3930_393028

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 6 meters, width 5 meters, 
    and depth 2 meters is 104 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 6 5 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3930_393028


namespace NUMINAMATH_CALUDE_third_power_four_five_l3930_393074

theorem third_power_four_five (x y : ℚ) : 
  x = 5/6 → y = 6/5 → (1/3) * x^4 * y^5 = 44/111 := by
  sorry

end NUMINAMATH_CALUDE_third_power_four_five_l3930_393074


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l3930_393067

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9 / 2) + Real.sqrt (2 / 9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l3930_393067


namespace NUMINAMATH_CALUDE_prohor_receives_all_money_l3930_393003

/-- Represents a person with their initial number of flatbreads -/
structure Person where
  name : String
  flatbreads : ℕ

/-- Represents the situation with the woodcutters and hunter -/
structure WoodcutterSituation where
  ivan : Person
  prohor : Person
  hunter : Person
  total_flatbreads : ℕ
  total_people : ℕ
  hunter_payment : ℕ

/-- Calculates the fair compensation for a person based on shared flatbreads -/
def fair_compensation (situation : WoodcutterSituation) (person : Person) : ℕ :=
  let shared_flatbreads := person.flatbreads - (situation.total_flatbreads / situation.total_people)
  shared_flatbreads * (situation.hunter_payment / situation.total_flatbreads)

/-- Theorem stating that Prohor should receive all the money -/
theorem prohor_receives_all_money (situation : WoodcutterSituation) : 
  situation.ivan.flatbreads = 4 →
  situation.prohor.flatbreads = 8 →
  situation.total_flatbreads = 12 →
  situation.total_people = 3 →
  situation.hunter_payment = 60 →
  fair_compensation situation situation.prohor = situation.hunter_payment :=
sorry

end NUMINAMATH_CALUDE_prohor_receives_all_money_l3930_393003


namespace NUMINAMATH_CALUDE_combination_equality_implies_x_values_l3930_393047

theorem combination_equality_implies_x_values (x : ℕ) : 
  (Nat.choose 25 (2 * x) = Nat.choose 25 (x + 4)) → (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_x_values_l3930_393047


namespace NUMINAMATH_CALUDE_vector_inequality_l3930_393007

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality (a b c : V) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖a + b + c‖ ≥ ‖a + b‖ + ‖b + c‖ + ‖c + a‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l3930_393007


namespace NUMINAMATH_CALUDE_snake_length_ratio_l3930_393020

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := 70

/-- The ratio of the boa constrictor's length to the garden snake's length -/
def length_ratio : ℚ := boa_constrictor_length / garden_snake_length

theorem snake_length_ratio :
  length_ratio = 7 := by sorry

end NUMINAMATH_CALUDE_snake_length_ratio_l3930_393020


namespace NUMINAMATH_CALUDE_xyz_sum_root_l3930_393051

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 16) 
  (h2 : z + x = 18) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = 9 * Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l3930_393051


namespace NUMINAMATH_CALUDE_tv_sales_effect_l3930_393058

theorem tv_sales_effect (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 54.26 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l3930_393058


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l3930_393017

/-- The number of digits in the representation of an integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth digit in the sequence of concatenated integers from 1 to 500 -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_1234_is_4 :
  nthDigit 1234 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l3930_393017


namespace NUMINAMATH_CALUDE_different_terminal_sides_not_equal_l3930_393036

-- Define an angle
def Angle : Type := ℝ

-- Define the initial side of an angle
def initial_side (a : Angle) : ℝ × ℝ := sorry

-- Define the terminal side of an angle
def terminal_side (a : Angle) : ℝ × ℝ := sorry

-- Define equality of angles
def angle_eq (a b : Angle) : Prop := 
  initial_side a = initial_side b ∧ terminal_side a = terminal_side b

-- Theorem statement
theorem different_terminal_sides_not_equal (a b : Angle) :
  initial_side a = initial_side b → 
  terminal_side a ≠ terminal_side b → 
  ¬(angle_eq a b) := by sorry

end NUMINAMATH_CALUDE_different_terminal_sides_not_equal_l3930_393036


namespace NUMINAMATH_CALUDE_tiling_comparison_l3930_393005

/-- Number of ways to tile a grid with rectangles -/
def tiling_count (grid_size : ℕ × ℕ) (tile_size : ℕ × ℕ) : ℕ := sorry

/-- Theorem: For any n > 1, the number of ways to tile a 3n × 3n grid with 1 × 3 rectangles
    is greater than the number of ways to tile a 2n × 2n grid with 1 × 2 rectangles -/
theorem tiling_comparison (n : ℕ) (h : n > 1) :
  tiling_count (3*n, 3*n) (1, 3) > tiling_count (2*n, 2*n) (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_tiling_comparison_l3930_393005


namespace NUMINAMATH_CALUDE_rabbit_speed_l3930_393041

theorem rabbit_speed (x : ℕ) : ((2 * x + 4) * 2 = 188) ↔ (x = 45) := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l3930_393041


namespace NUMINAMATH_CALUDE_power_division_equals_square_l3930_393014

theorem power_division_equals_square (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_square_l3930_393014


namespace NUMINAMATH_CALUDE_minimum_additional_games_minimum_additional_games_is_146_l3930_393023

theorem minimum_additional_games : ℕ → Prop :=
  fun n =>
    let initial_games : ℕ := 4
    let initial_lions_wins : ℕ := 3
    let initial_eagles_wins : ℕ := 1
    let total_games : ℕ := initial_games + n
    let total_eagles_wins : ℕ := initial_eagles_wins + n
    (total_eagles_wins : ℚ) / (total_games : ℚ) ≥ 98 / 100 ∧
    ∀ m : ℕ, m < n →
      let total_games_m : ℕ := initial_games + m
      let total_eagles_wins_m : ℕ := initial_eagles_wins + m
      (total_eagles_wins_m : ℚ) / (total_games_m : ℚ) < 98 / 100

theorem minimum_additional_games_is_146 : minimum_additional_games 146 := by
  sorry

#check minimum_additional_games_is_146

end NUMINAMATH_CALUDE_minimum_additional_games_minimum_additional_games_is_146_l3930_393023


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3930_393090

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3930_393090


namespace NUMINAMATH_CALUDE_vector_subtraction_l3930_393040

/-- Given two vectors OM and ON in R², prove that MN = ON - OM -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3930_393040


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3930_393033

theorem simplify_sqrt_expression :
  (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 147 / Real.sqrt 63) = (42 - 7 * Real.sqrt 21) / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3930_393033


namespace NUMINAMATH_CALUDE_problem_solution_l3930_393053

theorem problem_solution (x y : ℝ) (h : y = Real.sqrt (x - 4) - Real.sqrt (4 - x) + 2023) :
  y - x^2 + 17 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3930_393053


namespace NUMINAMATH_CALUDE_min_value_zero_l3930_393093

/-- The expression for which we want to find the minimum value -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (4*k^2 + 3)*y^2 - 6*x - 6*y + 9

/-- The theorem stating the condition for the minimum value of f to be 0 -/
theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ∧ (∃ x y : ℝ, f k x y = 0) ↔ k = 4/3 := by sorry

end NUMINAMATH_CALUDE_min_value_zero_l3930_393093


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l3930_393004

/-- Proves that a measured weight loss of 9.22% with 2% added clothing weight
    corresponds to an actual weight loss of approximately 5.55% -/
theorem weight_loss_calculation (measured_loss : Real) (clothing_weight : Real) :
  measured_loss = 9.22 ∧ clothing_weight = 2 →
  ∃ actual_loss : Real,
    (100 - actual_loss) * (1 + clothing_weight / 100) = 100 - measured_loss ∧
    abs (actual_loss - 5.55) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l3930_393004


namespace NUMINAMATH_CALUDE_modulus_product_complex_l3930_393029

theorem modulus_product_complex : |(7 - 4*I)*(3 + 11*I)| = Real.sqrt 8450 := by
  sorry

end NUMINAMATH_CALUDE_modulus_product_complex_l3930_393029


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3930_393062

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {1, 3, 4}

theorem complement_of_intersection (h : U = {1, 2, 3, 4} ∧ M = {1, 2, 3} ∧ N = {1, 3, 4}) :
  (M ∩ N)ᶜ = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3930_393062


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3930_393077

theorem complex_magnitude_proof : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3930_393077


namespace NUMINAMATH_CALUDE_angle_relation_l3930_393091

theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1/2) (h6 : Real.cos β = -7 * Real.sqrt 2 / 10) :
  2 * α - β = -3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l3930_393091


namespace NUMINAMATH_CALUDE_quotient_calculation_l3930_393026

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 166)
  (h2 : divisor = 18)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_calculation_l3930_393026


namespace NUMINAMATH_CALUDE_train_length_l3930_393057

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  post_time = 15 →
  platform_length = 100 →
  platform_time = 25 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3930_393057


namespace NUMINAMATH_CALUDE_customer_money_problem_l3930_393043

/-- Represents the initial amount of money a customer has --/
structure Money where
  dollars : ℕ
  cents : ℕ

/-- Represents the conditions of the problem --/
def satisfiesConditions (m : Money) : Prop :=
  let totalCents := 100 * m.dollars + m.cents
  let remainingCents := totalCents / 2
  let remainingDollars := remainingCents / 100
  let remainingCentsOnly := remainingCents % 100
  remainingCentsOnly = m.dollars ∧ remainingDollars = 2 * m.cents

/-- The theorem to be proved --/
theorem customer_money_problem :
  ∃ (m : Money), satisfiesConditions m ∧ m.dollars = 99 ∧ m.cents = 98 := by
  sorry

end NUMINAMATH_CALUDE_customer_money_problem_l3930_393043


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3930_393025

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3930_393025
