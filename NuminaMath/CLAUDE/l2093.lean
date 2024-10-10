import Mathlib

namespace standard_deviation_of_numbers_l2093_209387

def numbers : List ℝ := [9.8, 9.8, 9.9, 9.9, 10.0, 10.0, 10.1, 10.5]

theorem standard_deviation_of_numbers :
  let mean : ℝ := 10
  let count_within_one_std : ℕ := 7
  let n : ℕ := numbers.length
  ∀ σ : ℝ,
    (mean = (numbers.sum / n)) →
    (count_within_one_std = (numbers.filter (λ x => |x - mean| ≤ σ)).length) →
    (count_within_one_std = (n * 875 / 1000)) →
    σ = 0.5 := by
  sorry

end standard_deviation_of_numbers_l2093_209387


namespace sum_between_bounds_l2093_209367

theorem sum_between_bounds : 
  (21/2 : ℚ) < (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) ∧ 
  (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) < 11 := by
  sorry

end sum_between_bounds_l2093_209367


namespace f_symmetry_l2093_209323

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b 2 = -1 → f a b (-2) = 3 := by
  sorry

end f_symmetry_l2093_209323


namespace car_savings_calculation_l2093_209392

theorem car_savings_calculation 
  (monthly_earnings : ℕ) 
  (car_cost : ℕ) 
  (total_earnings : ℕ) 
  (h1 : monthly_earnings = 4000)
  (h2 : car_cost = 45000)
  (h3 : total_earnings = 360000) :
  car_cost / (total_earnings / monthly_earnings) = 500 := by
sorry

end car_savings_calculation_l2093_209392


namespace sprint_medal_awards_l2093_209354

/-- The number of ways to award medals in the international sprint final -/
def medal_awards (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_wins := non_american_sprinters.descFactorial medals
  let one_american_wins := american_sprinters * medals * (non_american_sprinters.descFactorial (medals - 1))
  no_american_wins + one_american_wins

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem sprint_medal_awards :
  medal_awards 10 4 3 = 480 := by
  sorry

end sprint_medal_awards_l2093_209354


namespace total_amount_cows_and_goats_l2093_209335

/-- The total amount spent on cows and goats -/
def total_amount (num_cows num_goats cow_price goat_price : ℕ) : ℕ :=
  num_cows * cow_price + num_goats * goat_price

/-- Theorem: The total amount spent on 2 cows at Rs. 460 each and 8 goats at Rs. 60 each is Rs. 1400 -/
theorem total_amount_cows_and_goats :
  total_amount 2 8 460 60 = 1400 := by
  sorry

end total_amount_cows_and_goats_l2093_209335


namespace arrival_time_difference_l2093_209337

/-- Represents the distance to the pool in miles -/
def distance_to_pool : ℝ := 3

/-- Represents Jill's speed in miles per hour -/
def jill_speed : ℝ := 12

/-- Represents Jack's speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculates the time difference in minutes between Jill and Jack's arrival at the pool -/
theorem arrival_time_difference : 
  hours_to_minutes (distance_to_pool / jill_speed - distance_to_pool / jack_speed) = 45 := by
  sorry

end arrival_time_difference_l2093_209337


namespace min_cost_to_1981_l2093_209348

/-- Cost of multiplying by 3 -/
def mult_cost : ℕ := 5

/-- Cost of adding 4 -/
def add_cost : ℕ := 2

/-- The target number to reach -/
def target : ℕ := 1981

/-- A step in the calculation process -/
inductive Step
| Mult : Step  -- Multiply by 3
| Add : Step   -- Add 4

/-- A sequence of steps -/
def Sequence := List Step

/-- Calculate the result of applying a sequence of steps starting from 1 -/
def apply_sequence (s : Sequence) : ℕ :=
  s.foldl (λ n step => match step with
    | Step.Mult => n * 3
    | Step.Add => n + 4) 1

/-- Calculate the cost of a sequence of steps -/
def sequence_cost (s : Sequence) : ℕ :=
  s.foldl (λ cost step => cost + match step with
    | Step.Mult => mult_cost
    | Step.Add => add_cost) 0

/-- Theorem: The minimum cost to reach 1981 is 42 kopecks -/
theorem min_cost_to_1981 :
  ∃ (s : Sequence), apply_sequence s = target ∧
    sequence_cost s = 42 ∧
    ∀ (s' : Sequence), apply_sequence s' = target →
      sequence_cost s' ≥ sequence_cost s :=
by sorry

end min_cost_to_1981_l2093_209348


namespace quadratic_solution_implies_sum_l2093_209314

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (a * 2^2 - b * 2 + 2 = 0) → (2024 + 2*a - b = 2023) := by
  sorry

end quadratic_solution_implies_sum_l2093_209314


namespace legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l2093_209322

-- Define the necessary variables and functions
variable (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
variable (a : ℕ) (hcoprime : Nat.Coprime a p)

-- Theorem 1
theorem legendre_symbol_values :
  (a ^ ((p - 1) / 2)) % p = 1 ∨ (a ^ ((p - 1) / 2)) % p = p - 1 :=
sorry

-- Theorem 2
theorem legendre_symbol_square_equivalence :
  (a ^ ((p - 1) / 2)) % p = 1 ↔ ∃ x, (x * x) % p = a % p :=
sorry

-- Theorem 3
theorem minus_one_square_mod_p :
  (∃ x, (x * x) % p = p - 1) ↔ p % 4 = 1 :=
sorry

-- Theorem 4
theorem eleven_power_sum_of_squares (n : ℕ) :
  ∀ a b : ℕ, 11^n = a^2 + b^2 →
    ∃ k : ℕ, n = 2*k ∧ ((a = 11^k ∧ b = 0) ∨ (a = 0 ∧ b = 11^k)) :=
sorry

end legendre_symbol_values_legendre_symbol_square_equivalence_minus_one_square_mod_p_eleven_power_sum_of_squares_l2093_209322


namespace committee_formation_count_l2093_209336

def total_members : ℕ := 12
def committee_size : ℕ := 5
def incompatible_members : ℕ := 2

theorem committee_formation_count :
  (Nat.choose total_members committee_size) -
  (Nat.choose (total_members - incompatible_members) (committee_size - incompatible_members)) = 672 := by
  sorry

end committee_formation_count_l2093_209336


namespace quadratic_inequality_range_l2093_209353

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ a ≥ (1/4 : ℝ) :=
sorry

end quadratic_inequality_range_l2093_209353


namespace village_cats_l2093_209365

theorem village_cats (total : ℕ) (spotted : ℕ) (fluffy_spotted : ℕ) : 
  spotted = total / 3 →
  fluffy_spotted = spotted / 4 →
  fluffy_spotted = 10 →
  total = 120 := by
sorry

end village_cats_l2093_209365


namespace port_distance_l2093_209325

/-- The distance between two ports given travel times and current speed -/
theorem port_distance (downstream_time upstream_time current_speed : ℝ) 
  (h_downstream : downstream_time = 3)
  (h_upstream : upstream_time = 4)
  (h_current : current_speed = 5) : 
  ∃ (distance boat_speed : ℝ),
    distance = downstream_time * (boat_speed + current_speed) ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = 120 := by
  sorry

end port_distance_l2093_209325


namespace hyperbola_standard_equation_l2093_209340

/-- A hyperbola passing through the point (3, -√2) with eccentricity √5/2 has the standard equation x²/1 - y²/(1/4) = 1 -/
theorem hyperbola_standard_equation (x y a b : ℝ) : 
  (x = 3 ∧ y = -Real.sqrt 2) →  -- Point on the hyperbola
  (Real.sqrt 5 / 2 = Real.sqrt (a^2 + b^2) / a) →  -- Eccentricity
  (x^2 / a^2 - y^2 / b^2 = 1) →  -- General equation of hyperbola
  (a^2 = 1 ∧ b^2 = 1/4) :=  -- Standard equation coefficients
by sorry

end hyperbola_standard_equation_l2093_209340


namespace calculate_expression_l2093_209324

theorem calculate_expression : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end calculate_expression_l2093_209324


namespace fraction_2011_l2093_209327

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ := sorry

/-- The sum of Euler's totient function up to n -/
def phi_sum (n : ℕ) : ℕ := sorry

theorem fraction_2011 : fraction_sequence 2011 = 49 / 111 := by sorry

end fraction_2011_l2093_209327


namespace hyperbola_asymptote_slope_l2093_209318

theorem hyperbola_asymptote_slope (m : ℝ) (α : ℝ) :
  (∀ x y : ℝ, x^2 + y^2/m = 1) →
  (0 < α ∧ α < π/3) →
  (-3 < m ∧ m < 0) :=
sorry

end hyperbola_asymptote_slope_l2093_209318


namespace gum_pack_size_l2093_209393

/-- The number of pieces of banana gum Luke has initially -/
def banana_gum : ℕ := 28

/-- The number of pieces of apple gum Luke has initially -/
def apple_gum : ℕ := 36

/-- The number of pieces of gum in each complete pack -/
def y : ℕ := 14

theorem gum_pack_size :
  (banana_gum - 2 * y) * (apple_gum + 3 * y) = banana_gum * apple_gum := by
  sorry

#check gum_pack_size

end gum_pack_size_l2093_209393


namespace gcd_97_power_plus_one_l2093_209360

theorem gcd_97_power_plus_one (p : Nat) (h_prime : Nat.Prime p) (h_p : p = 97) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end gcd_97_power_plus_one_l2093_209360


namespace cistern_filling_time_l2093_209308

theorem cistern_filling_time (t : ℝ) : t > 0 → 
  (4 * (1 / t + 1 / 18) + 8 / 18 = 1) → t = 12 := by sorry

end cistern_filling_time_l2093_209308


namespace linear_function_properties_l2093_209332

def f (x : ℝ) : ℝ := -2 * x - 4

theorem linear_function_properties :
  (f (-1) = -2) ∧
  (f 0 ≠ -2) ∧
  (∀ x, x < -2 → f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → f x ≠ y) :=
by sorry

end linear_function_properties_l2093_209332


namespace bill_denomination_l2093_209338

-- Define the problem parameters
def total_bill : ℕ := 285
def coin_value : ℕ := 5
def total_items : ℕ := 24
def num_bills : ℕ := 11
def num_coins : ℕ := 11

-- Theorem to prove
theorem bill_denomination :
  ∃ (x : ℕ), 
    x * num_bills + coin_value * num_coins = total_bill ∧
    num_bills + num_coins = total_items ∧
    x = 20 := by
  sorry

end bill_denomination_l2093_209338


namespace unique_solution_divisor_system_l2093_209375

theorem unique_solution_divisor_system :
  ∀ a b : ℕ+,
  (∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ+),
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ ∧ a₈ < a₉ ∧ a₉ < a₁₀ ∧ a₁₀ < a₁₁ ∧
    a₁ ∣ a ∧ a₂ ∣ a ∧ a₃ ∣ a ∧ a₄ ∣ a ∧ a₅ ∣ a ∧ a₆ ∣ a ∧ a₇ ∣ a ∧ a₈ ∣ a ∧ a₉ ∣ a ∧ a₁₀ ∣ a ∧ a₁₁ ∣ a) →
  (∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ+),
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆ ∧ b₆ < b₇ ∧ b₇ < b₈ ∧ b₈ < b₉ ∧ b₉ < b₁₀ ∧ b₁₀ < b₁₁ ∧
    b₁ ∣ b ∧ b₂ ∣ b ∧ b₃ ∣ b ∧ b₄ ∣ b ∧ b₅ ∣ b ∧ b₆ ∣ b ∧ b₇ ∣ b ∧ b₈ ∣ b ∧ b₉ ∣ b ∧ b₁₀ ∣ b ∧ b₁₁ ∣ b) →
  a₁₀ + b₁₀ = a →
  a₁₁ + b₁₁ = b →
  a = 1024 ∧ b = 2048 :=
by sorry

end unique_solution_divisor_system_l2093_209375


namespace interest_tax_rate_proof_l2093_209305

/-- The tax rate for interest tax on savings deposits in China --/
def interest_tax_rate : ℝ := 0.20

theorem interest_tax_rate_proof (initial_deposit : ℝ) (interest_rate : ℝ) (total_received : ℝ)
  (h1 : initial_deposit = 10000)
  (h2 : interest_rate = 0.0225)
  (h3 : total_received = 10180) :
  initial_deposit + initial_deposit * interest_rate * (1 - interest_tax_rate) = total_received :=
by sorry

end interest_tax_rate_proof_l2093_209305


namespace triangle_properties_l2093_209346

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2))
  (h_obtuse : t.C > π/2) :
  t.B = π/3 ∧ ∃ (k : ℝ), k > 2 ∧ t.c / t.a = k :=
by sorry

end triangle_properties_l2093_209346


namespace x_value_proof_l2093_209357

def star_operation (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_proof :
  ∀ x : ℝ, star_operation 3 x = 27 → x = 6 := by
  sorry

end x_value_proof_l2093_209357


namespace inscribed_rectangle_circle_circumference_l2093_209377

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    circle_circumference = π * (rectangle_width^2 + rectangle_height^2).sqrt →
    circle_circumference = 15 * π := by
  sorry

end inscribed_rectangle_circle_circumference_l2093_209377


namespace xyz_sum_reciprocal_l2093_209306

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 7)
  (h_sum_y : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end xyz_sum_reciprocal_l2093_209306


namespace cube_sum_reciprocal_l2093_209302

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end cube_sum_reciprocal_l2093_209302


namespace sodas_per_pack_james_sodas_problem_l2093_209359

theorem sodas_per_pack (packs : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) (sodas_per_day : ℕ) : ℕ :=
  let total_sodas := sodas_per_day * days_in_week
  let new_sodas := total_sodas - initial_sodas
  new_sodas / packs

theorem james_sodas_problem : sodas_per_pack 5 10 7 10 = 12 := by
  sorry

end sodas_per_pack_james_sodas_problem_l2093_209359


namespace recurring_decimal_to_fraction_l2093_209326

theorem recurring_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + 7 / 9 * (1 / 10^n)) → x = 34 / 9 := by
  sorry

end recurring_decimal_to_fraction_l2093_209326


namespace total_spent_proof_l2093_209316

def jayda_stall1 : ℝ := 400
def jayda_stall2 : ℝ := 120
def jayda_stall3 : ℝ := 250
def aitana_multiplier : ℝ := 1.4 -- 1 + 2/5
def jayda_discount1 : ℝ := 0.05
def aitana_discount2 : ℝ := 0.10
def sales_tax : ℝ := 0.10
def exchange_rate : ℝ := 1.25

def total_spent_cad : ℝ :=
  ((jayda_stall1 * (1 - jayda_discount1) + jayda_stall2 + jayda_stall3) * (1 + sales_tax) +
   (jayda_stall1 * aitana_multiplier + 
    jayda_stall2 * aitana_multiplier * (1 - aitana_discount2) + 
    jayda_stall3 * aitana_multiplier) * (1 + sales_tax)) * exchange_rate

theorem total_spent_proof : total_spent_cad = 2490.40 := by
  sorry

end total_spent_proof_l2093_209316


namespace value_of_N_l2093_209317

theorem value_of_N : 
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / 
           Real.sqrt (Real.sqrt 5 + 1) - 
           Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end value_of_N_l2093_209317


namespace square_divides_power_plus_one_l2093_209361

theorem square_divides_power_plus_one (n : ℕ) : 
  n ^ 2 ∣ 2 ^ n + 1 ↔ n = 1 ∨ n = 3 := by
sorry

end square_divides_power_plus_one_l2093_209361


namespace t_value_l2093_209355

/-- Linear regression equation for the given data points -/
def linear_regression (x : ℝ) : ℝ := 1.04 * x + 1.9

/-- The value of t in the data set (4, t) -/
def t : ℝ := linear_regression 4

theorem t_value : t = 6.06 := by
  sorry

end t_value_l2093_209355


namespace corn_amount_approx_l2093_209397

/-- The cost of corn per pound -/
def corn_cost : ℝ := 1.05

/-- The cost of beans per pound -/
def bean_cost : ℝ := 0.39

/-- The total pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 23.10

/-- The amount of corn bought (in pounds) -/
noncomputable def corn_amount : ℝ := 
  (total_cost - bean_cost * total_pounds) / (corn_cost - bean_cost)

theorem corn_amount_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |corn_amount - 17.3| < ε :=
sorry

end corn_amount_approx_l2093_209397


namespace subscription_difference_l2093_209382

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  b_subscription : ℕ
  c_subscription : ℕ

/-- Theorem stating the difference between b's and c's subscriptions given the problem conditions. -/
theorem subscription_difference (bv : BusinessVenture) : 
  bv.total_subscription = 50000 ∧
  bv.total_profit = 36000 ∧
  bv.a_profit = 15120 ∧
  bv.b_subscription + 4000 + bv.b_subscription + bv.c_subscription = bv.total_subscription ∧
  bv.a_profit * bv.total_subscription = bv.total_profit * (bv.b_subscription + 4000) →
  bv.b_subscription - bv.c_subscription = 5000 := by
  sorry

#check subscription_difference

end subscription_difference_l2093_209382


namespace probability_five_digit_palindrome_div_11_l2093_209347

-- Define a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ∃ a b c : ℕ, 
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

-- Define divisibility by 11
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

-- Count of five-digit palindromes
def count_five_digit_palindromes : ℕ := 900

-- Count of five-digit palindromes divisible by 11
def count_five_digit_palindromes_div_11 : ℕ := 90

-- Theorem statement
theorem probability_five_digit_palindrome_div_11 :
  (count_five_digit_palindromes_div_11 : ℚ) / count_five_digit_palindromes = 1 / 10 :=
sorry

end probability_five_digit_palindrome_div_11_l2093_209347


namespace problem_solution_l2093_209369

theorem problem_solution (s P k : ℝ) (h : P = s / Real.sqrt ((1 + k) ^ n)) :
  n = (2 * Real.log (s / P)) / Real.log (1 + k) :=
by sorry

end problem_solution_l2093_209369


namespace corn_increase_factor_l2093_209372

theorem corn_increase_factor (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 1) 
  (h3 : 1 - x + x = 1/2) 
  (h4 : 1 - x + x/2 = 1/2) : 
  (3/2 * x) / (1/2 * x) = 3 := by sorry

end corn_increase_factor_l2093_209372


namespace constant_function_l2093_209344

theorem constant_function (α : ℝ) (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos x ^ 2 + Real.cos (x + α) ^ 2 - 2 * Real.cos α * Real.cos x * Real.cos (x + α)
  f x = (1 - Real.cos (2 * α)) / 2 :=
by sorry

end constant_function_l2093_209344


namespace arithmetic_sequence_proof_l2093_209343

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (k b : ℝ) :
  let a : ℕ → ℝ := λ n => k * n + b
  is_arithmetic_sequence a ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = k) :=
by
  sorry

end arithmetic_sequence_proof_l2093_209343


namespace largest_integer_k_for_real_roots_l2093_209312

theorem largest_integer_k_for_real_roots : ∃ (k : ℤ),
  (∀ (j : ℤ), (∀ (x : ℝ), ∃ (y : ℝ), x * (j * x + 1) - x^2 + 3 = 0) → j ≤ k) ∧
  (∀ (x : ℝ), ∃ (y : ℝ), x * (k * x + 1) - x^2 + 3 = 0) ∧
  k = 1 :=
sorry

end largest_integer_k_for_real_roots_l2093_209312


namespace quadratic_monotonicity_l2093_209380

/-- A function f is monotonic on an open interval (a, b) if it is either
    strictly increasing or strictly decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f y < f x)

theorem quadratic_monotonicity (a : ℝ) :
  IsMonotonic (fun x ↦ x^2 + 2*(a-1)*x + 2) 2 4 →
  a ≤ -3 ∨ a ≥ -1 := by
  sorry

end quadratic_monotonicity_l2093_209380


namespace gcd_1734_816_l2093_209390

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end gcd_1734_816_l2093_209390


namespace parallelogram_area_calculation_l2093_209362

-- Define the parallelogram
def parallelogram_base : ℝ := 32
def parallelogram_height : ℝ := 14

-- Define the area formula for a parallelogram
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_area_calculation :
  parallelogram_area parallelogram_base parallelogram_height = 448 := by
  sorry

end parallelogram_area_calculation_l2093_209362


namespace quadratic_properties_l2093_209376

/-- Represents a quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function satisfying the given conditions -/
def f : QuadraticFunction := {
  a := -1,
  b := 2,
  c := 3,
  a_nonzero := by norm_num
}

/-- Evaluation of the quadratic function -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_properties (t : ℝ) :
  let f := f
  (∃ x y, x > 0 ∧ y > 0 ∧ eval f x = y ∧ ∀ x', eval f x' ≤ eval f x) ∧ 
  (∀ m n, 0 < m → m < 4 → eval f m = n → -5 < n ∧ n ≤ 4) ∧
  (eval f (-2) = t ∧ eval f 4 = t) ∧
  (∀ p, (∀ x, eval f x < 2*x + p) → p > 3) := by
  sorry


end quadratic_properties_l2093_209376


namespace removed_triangles_area_l2093_209350

/-- The combined area of four isosceles right triangles removed from the corners of a square
    with side length 20 units to form a regular octagon is 512 square units. -/
theorem removed_triangles_area (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 20 →
  (square_side - 2 * triangle_leg)^2 + (triangle_leg - (square_side - 2 * triangle_leg))^2 = square_side^2 →
  4 * (1/2 * triangle_leg^2) = 512 :=
by sorry

end removed_triangles_area_l2093_209350


namespace valid_mixture_weight_l2093_209373

/-- A cement mixture composed of sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ
  gravel_weight : ℝ

/-- The cement mixture satisfies the given conditions -/
def is_valid_mixture (m : CementMixture) : Prop :=
  m.sand_fraction = 1/3 ∧
  m.water_fraction = 1/4 ∧
  m.gravel_weight = 10 ∧
  m.sand_fraction * m.total_weight + m.water_fraction * m.total_weight + m.gravel_weight = m.total_weight

/-- The theorem stating that a valid mixture has a total weight of 24 pounds -/
theorem valid_mixture_weight (m : CementMixture) (h : is_valid_mixture m) : m.total_weight = 24 := by
  sorry

end valid_mixture_weight_l2093_209373


namespace intersection_M_N_l2093_209352

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l2093_209352


namespace existence_of_points_with_derivative_sum_zero_l2093_209307

theorem existence_of_points_with_derivative_sum_zero
  {f : ℝ → ℝ} {a b : ℝ} (h_diff : DifferentiableOn ℝ f (Set.Icc a b))
  (h_eq : f a = f b) (h_lt : a < b) :
  ∃ x y, x ∈ Set.Icc a b ∧ y ∈ Set.Icc a b ∧ x ≠ y ∧
    (deriv f x) + 5 * (deriv f y) = 0 := by
  sorry

end existence_of_points_with_derivative_sum_zero_l2093_209307


namespace parallel_perpendicular_implication_l2093_209351

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

end parallel_perpendicular_implication_l2093_209351


namespace kate_bouncy_balls_difference_l2093_209320

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The total number of red bouncy balls Kate bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Kate bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

/-- The difference between the number of red and yellow bouncy balls -/
def difference_in_balls : ℕ := total_red_balls - total_yellow_balls

theorem kate_bouncy_balls_difference :
  difference_in_balls = 18 := by sorry

end kate_bouncy_balls_difference_l2093_209320


namespace school_section_problem_l2093_209339

/-- Calculates the maximum number of equal-sized mixed-gender sections
    that can be formed given the number of boys and girls and the required ratio. -/
def max_sections (boys girls : ℕ) (boy_ratio girl_ratio : ℕ) : ℕ :=
  min (boys / boy_ratio) (girls / girl_ratio)

/-- Theorem stating the solution to the school section problem -/
theorem school_section_problem :
  max_sections 2040 1728 3 2 = 680 := by
  sorry

end school_section_problem_l2093_209339


namespace total_arrangements_l2093_209331

/-- Represents the three elective math courses -/
inductive Course
| MatrixTransformation
| InfoSecCrypto
| SwitchCircuits

/-- Represents a teacher and their teaching capabilities -/
structure Teacher where
  id : Nat
  canTeach : Course → Bool

/-- The pool of available teachers -/
def teacherPool : Finset Teacher := sorry

/-- Teachers who can teach only Matrix and Transformation -/
def matrixOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Information Security and Cryptography -/
def cryptoOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Switch Circuits and Boolean Algebra -/
def switchOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach all three courses -/
def versatileTeachers : Finset Teacher := sorry

/-- A valid selection of teachers for the courses -/
def isValidSelection (selection : Finset Teacher) : Prop := sorry

/-- The number of different valid arrangements -/
def numArrangements : Nat := sorry

theorem total_arrangements :
  (Finset.card teacherPool = 10) →
  (Finset.card matrixOnlyTeachers = 3) →
  (Finset.card cryptoOnlyTeachers = 2) →
  (Finset.card switchOnlyTeachers = 3) →
  (Finset.card versatileTeachers = 2) →
  (∀ s : Finset Teacher, isValidSelection s → Finset.card s = 9) →
  (∀ c : Course, ∀ s : Finset Teacher, isValidSelection s →
    Finset.card (s.filter (fun t => t.canTeach c)) = 3) →
  numArrangements = 16 := by
  sorry

end total_arrangements_l2093_209331


namespace translation_of_line_segment_l2093_209356

/-- Given a line segment AB with endpoints A(1,0) and B(3,2), if it is translated to a new position
    where the new endpoints are A₁(a,1) and B₁(4,b), then a = 2 and b = 3. -/
theorem translation_of_line_segment (a b : ℝ) : 
  (∃ (dx dy : ℝ), (1 + dx = a ∧ 0 + dy = 1) ∧ (3 + dx = 4 ∧ 2 + dy = b)) → 
  (a = 2 ∧ b = 3) :=
by sorry

end translation_of_line_segment_l2093_209356


namespace symmetric_points_ab_power_l2093_209371

/-- Given two points M(2a, 2) and N(-8, a+b) that are symmetric with respect to the y-axis,
    prove that a^b = 1/16 -/
theorem symmetric_points_ab_power (a b : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (2*a, 2) ∧ 
    N = (-8, a+b) ∧ 
    (M.1 = -N.1) ∧  -- x-coordinates are opposite
    (M.2 = N.2))    -- y-coordinates are equal
  → a^b = 1/16 := by
sorry

end symmetric_points_ab_power_l2093_209371


namespace distance_inequality_l2093_209374

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the "in" relation for lines and planes
variable (in_plane : Line → Plane → Prop)

-- Define the "on" relation for points and lines
variable (on_line : Point → Line → Prop)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_to_line : Point → Line → ℝ)
variable (distance_line_to_line : Line → Line → ℝ)

-- Define the specific objects in our problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the theorem
theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : in_plane m α)
  (h_n_in_β : in_plane n β)
  (h_A_on_m : on_line A m)
  (h_B_on_n : on_line B n)
  (h_a : distance A B = a)
  (h_b : distance_point_to_line A n = b)
  (h_c : distance_line_to_line m n = c)
  : c ≤ a ∧ a ≤ b :=
by sorry

end distance_inequality_l2093_209374


namespace triangle_cosine_sum_max_l2093_209334

theorem triangle_cosine_sum_max (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π →
  Real.cos A + Real.cos B * Real.cos C ≤ 1 :=
sorry

end triangle_cosine_sum_max_l2093_209334


namespace barbara_scrap_paper_heaps_l2093_209341

/-- Represents the number of sheets in a bundle of paper -/
def sheets_per_bundle : ℕ := 2

/-- Represents the number of sheets in a bunch of paper -/
def sheets_per_bunch : ℕ := 4

/-- Represents the number of sheets in a heap of paper -/
def sheets_per_heap : ℕ := 20

/-- Represents the number of bundles of colored paper Barbara found -/
def colored_bundles : ℕ := 3

/-- Represents the number of bunches of white paper Barbara found -/
def white_bunches : ℕ := 2

/-- Represents the total number of sheets Barbara removed -/
def total_sheets_removed : ℕ := 114

/-- Theorem stating the number of heaps of scrap paper Barbara found -/
theorem barbara_scrap_paper_heaps :
  (total_sheets_removed - (colored_bundles * sheets_per_bundle + white_bunches * sheets_per_bunch)) / sheets_per_heap = 5 := by
  sorry

end barbara_scrap_paper_heaps_l2093_209341


namespace triangle_altitude_reciprocal_sum_bounds_l2093_209396

/-- For any triangle, the sum of the reciprocals of two altitudes lies between the reciprocal of the radius of the inscribed circle and the reciprocal of its diameter. -/
theorem triangle_altitude_reciprocal_sum_bounds (a b c m_a m_b m_c ρ s t : ℝ) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_altitudes : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perimeter : a + b + c = 2 * s)
  (h_area : t > 0)
  (h_inscribed_radius : ρ > 0)
  (h_altitude_a : a * m_a = 2 * t)
  (h_altitude_b : b * m_b = 2 * t)
  (h_altitude_c : c * m_c = 2 * t)
  (h_inscribed_radius_def : s * ρ = t) :
  1 / (2 * ρ) < 1 / m_a + 1 / m_b ∧ 1 / m_a + 1 / m_b < 1 / ρ :=
by sorry

end triangle_altitude_reciprocal_sum_bounds_l2093_209396


namespace system_solution_l2093_209368

theorem system_solution :
  ∃! (x y : ℝ), 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 :=
by
  -- Proof goes here
  sorry

end system_solution_l2093_209368


namespace max_perpendicular_faces_theorem_l2093_209313

/-- The maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
def max_perpendicular_faces (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

/-- Theorem stating the maximum number of lateral faces of an n-sided pyramid that can be perpendicular to the base -/
theorem max_perpendicular_faces_theorem (n : ℕ) (h : n > 0) :
  max_perpendicular_faces n = 
    if n % 2 = 0 
    then n / 2 
    else (n + 1) / 2 :=
by sorry

end max_perpendicular_faces_theorem_l2093_209313


namespace brianna_cd_purchase_l2093_209311

theorem brianna_cd_purchase (m : ℚ) (c : ℚ) (n : ℚ) (h1 : m > 0) (h2 : c > 0) (h3 : n > 0) :
  (1 / 4 : ℚ) * m = (1 / 4 : ℚ) * n * c →
  m - n * c = 0 :=
by sorry

end brianna_cd_purchase_l2093_209311


namespace methane_required_moles_l2093_209304

/-- Represents a chemical species in a reaction -/
structure ChemicalSpecies where
  formula : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure ChemicalReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

def methane_chlorine_reaction : ChemicalReaction :=
  { reactants := [
      { formula := "CH4", moles := 1 },
      { formula := "Cl2", moles := 1 }
    ],
    products := [
      { formula := "CH3Cl", moles := 1 },
      { formula := "HCl", moles := 1 }
    ]
  }

/-- Theorem stating that 2 moles of CH4 are required to react with 2 moles of Cl2 -/
theorem methane_required_moles 
  (reaction : ChemicalReaction)
  (h_reaction : reaction = methane_chlorine_reaction)
  (h_cl2_moles : ∃ cl2 ∈ reaction.reactants, cl2.formula = "Cl2" ∧ cl2.moles = 2)
  (h_hcl_moles : ∃ hcl ∈ reaction.products, hcl.formula = "HCl" ∧ hcl.moles = 2) :
  ∃ ch4 ∈ reaction.reactants, ch4.formula = "CH4" ∧ ch4.moles = 2 :=
sorry

end methane_required_moles_l2093_209304


namespace f_monotone_decreasing_l2093_209378

def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem f_monotone_decreasing : 
  MonotoneOn f (Set.Ici 2) := by sorry

end f_monotone_decreasing_l2093_209378


namespace complex_square_root_of_negative_four_l2093_209345

theorem complex_square_root_of_negative_four (z : ℂ) : 
  z^2 = -4 ∧ z.im > 0 → z = 2*I :=
by sorry

end complex_square_root_of_negative_four_l2093_209345


namespace solve_for_a_l2093_209381

theorem solve_for_a : ∀ a : ℝ, (a * 1 - (-3) = 1) → a = -2 := by
  sorry

end solve_for_a_l2093_209381


namespace dist_to_left_focus_is_ten_l2093_209391

/-- The hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (pos_a : a > 0)
  (pos_b : b > 0)

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The distance from a point to the right focus of the hyperbola -/
def distToRightFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x - h.a|

/-- The distance from a point to the left focus of the hyperbola -/
def distToLeftFocus (h : Hyperbola) (p : PointOnHyperbola h) : ℝ :=
  |p.x + h.a|

/-- The main theorem -/
theorem dist_to_left_focus_is_ten
  (h : Hyperbola)
  (p : PointOnHyperbola h)
  (right_focus_dist : distToRightFocus h p = 4)
  (h_eq : h.a = 3 ∧ h.b = 4) :
  distToLeftFocus h p = 10 := by
  sorry

end dist_to_left_focus_is_ten_l2093_209391


namespace constant_term_when_sum_is_64_l2093_209301

-- Define the sum of binomial coefficients
def sum_binomial_coeffs (n : ℕ) : ℕ := 2^n

-- Define the constant term in the expansion
def constant_term (n : ℕ) : ℤ :=
  (-1)^(n/2) * (n.choose (n/2))

-- Theorem statement
theorem constant_term_when_sum_is_64 :
  ∃ n : ℕ, sum_binomial_coeffs n = 64 ∧ constant_term n = 15 :=
sorry

end constant_term_when_sum_is_64_l2093_209301


namespace smallest_a_for_equation_l2093_209310

theorem smallest_a_for_equation : ∃ (p : ℕ) (b : ℕ), 
  Nat.Prime p ∧ 
  b ≥ 2 ∧ 
  (9^p - 9) / p = b^2 ∧ 
  ∀ (a : ℕ) (q : ℕ) (c : ℕ), 
    a > 0 ∧ a < 9 → 
    Nat.Prime q → 
    c ≥ 2 → 
    (a^q - a) / q ≠ c^2 :=
by sorry

end smallest_a_for_equation_l2093_209310


namespace pancakes_needed_l2093_209303

/-- Given a family of 8 people and 12 pancakes already made, prove that 4 more pancakes are needed for everyone to have a second pancake. -/
theorem pancakes_needed (family_size : ℕ) (pancakes_made : ℕ) : 
  family_size = 8 → pancakes_made = 12 → 
  (family_size * 2 - pancakes_made : ℕ) = 4 := by sorry

end pancakes_needed_l2093_209303


namespace complex_multiplication_l2093_209384

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) :
  (1 + i) * (1 - 2*i) = 3 - i := by
  sorry

end complex_multiplication_l2093_209384


namespace inverse_function_value_l2093_209383

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x

-- Define the domain of f
def domain (x : ℝ) : Prop := x < -2

-- Define the inverse function property
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, domain x → f (f_inv (f x)) = f x ∧ f_inv (f x) = x

-- Theorem statement
theorem inverse_function_value :
  ∃ f_inv : ℝ → ℝ, is_inverse f f_inv ∧ f_inv 12 = -6 :=
sorry

end inverse_function_value_l2093_209383


namespace planning_committee_combinations_l2093_209385

theorem planning_committee_combinations (n : ℕ) (k : ℕ) : n = 20 ∧ k = 3 → Nat.choose n k = 1140 := by
  sorry

end planning_committee_combinations_l2093_209385


namespace years_until_26_l2093_209300

/-- Kiril's current age -/
def current_age : ℕ := sorry

/-- Kiril's target age -/
def target_age : ℕ := 26

/-- Condition that current age is a multiple of 5 -/
axiom current_age_multiple_of_5 : ∃ k : ℕ, current_age = 5 * k

/-- Condition that last year's age was a multiple of 7 -/
axiom last_year_age_multiple_of_7 : ∃ m : ℕ, current_age - 1 = 7 * m

/-- Theorem stating the number of years until Kiril is 26 -/
theorem years_until_26 : target_age - current_age = 11 := by sorry

end years_until_26_l2093_209300


namespace diagonal_cubes_180_270_360_l2093_209330

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_on_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: The internal diagonal of a 180 × 270 × 360 rectangular solid passes through 540 cubes -/
theorem diagonal_cubes_180_270_360 :
  cubes_on_diagonal 180 270 360 = 540 := by sorry

end diagonal_cubes_180_270_360_l2093_209330


namespace first_grade_students_l2093_209370

theorem first_grade_students (total : ℕ) (difference : ℕ) (first_grade : ℕ) : 
  total = 1256 → 
  difference = 408 →
  first_grade + difference = total - first_grade →
  first_grade = 424 := by
sorry

end first_grade_students_l2093_209370


namespace x_value_l2093_209329

theorem x_value : ∃ x : ℝ, x * 0.65 = 552.50 * 0.20 ∧ x = 170 := by
  sorry

end x_value_l2093_209329


namespace original_denominator_proof_l2093_209342

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ (1 : ℚ) / 2 ∧ (2 + 5 : ℚ) / (d + 5) = (1 : ℚ) / 2 → d = 9 := by
  sorry

end original_denominator_proof_l2093_209342


namespace at_least_three_same_purchase_l2093_209394

/-- Represents a purchase combination of items -/
structure Purchase where
  threeYuanItems : Nat
  fiveYuanItems : Nat
  deriving Repr

/-- The set of all valid purchase combinations -/
def validPurchases : Finset Purchase :=
  sorry

/-- The number of valid purchase combinations -/
def numCombinations : Nat :=
  Finset.card validPurchases

theorem at_least_three_same_purchase (n : Nat) (h : n = 25) :
  ∀ (purchases : Fin n → Purchase),
    (∀ i, purchases i ∈ validPurchases) →
    ∃ (p : Purchase) (i j k : Fin n),
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      purchases i = p ∧ purchases j = p ∧ purchases k = p :=
  sorry

end at_least_three_same_purchase_l2093_209394


namespace sarah_copies_3600_pages_l2093_209328

/-- The total number of pages Sarah will copy for two contracts -/
def total_pages (num_people : ℕ) (contract1_pages : ℕ) (contract1_copies : ℕ) 
                (contract2_pages : ℕ) (contract2_copies : ℕ) : ℕ :=
  num_people * (contract1_pages * contract1_copies + contract2_pages * contract2_copies)

/-- Theorem: Sarah will copy 3600 pages in total -/
theorem sarah_copies_3600_pages : 
  total_pages 20 30 3 45 2 = 3600 := by
  sorry

end sarah_copies_3600_pages_l2093_209328


namespace ants_sugar_harvesting_l2093_209395

def sugar_harvesting (initial_sugar : ℝ) (removal_rate : ℝ) (time_passed : ℝ) : Prop :=
  let remaining_sugar := initial_sugar - removal_rate * time_passed
  let remaining_time := remaining_sugar / removal_rate
  remaining_time = 3

theorem ants_sugar_harvesting :
  sugar_harvesting 24 4 3 :=
sorry

end ants_sugar_harvesting_l2093_209395


namespace geometric_sum_first_seven_l2093_209366

-- Define the geometric sequence
def geometric_sequence (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

-- Define the sum of the first n terms of the geometric sequence
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a₀ else a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 7
  geometric_sum a₀ r n = 1093/2187 := by
  sorry


end geometric_sum_first_seven_l2093_209366


namespace excess_amount_l2093_209379

theorem excess_amount (x : ℝ) (a : ℝ) : 
  x = 0.16 * x + a → x = 50 → a = 42 := by
  sorry

end excess_amount_l2093_209379


namespace solve_equation_l2093_209321

theorem solve_equation : ∃ x : ℝ, (3/2 : ℝ) * x - 3 = 15 ∧ x = 12 := by
  sorry

end solve_equation_l2093_209321


namespace scaled_circle_equation_l2093_209349

/-- Given a circle and a scaling transformation, prove the equation of the resulting curve -/
theorem scaled_circle_equation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle equation
  (x' = 2*x) →       -- Scaling for x
  (y' = 3*y) →       -- Scaling for y
  (x'^2/4 + y'^2/9 = 1) -- Resulting curve equation
:= by sorry

end scaled_circle_equation_l2093_209349


namespace product_173_240_l2093_209399

theorem product_173_240 : 
  (∃ n : ℕ, n * 12 = 173 * 240 ∧ n = 3460) → 173 * 240 = 41520 := by
  sorry

end product_173_240_l2093_209399


namespace saree_stripe_ratio_l2093_209388

theorem saree_stripe_ratio :
  ∀ (gold brown blue : ℕ),
    gold = brown →
    blue = 5 * gold →
    brown = 4 →
    blue = 60 →
    (gold : ℚ) / brown = 3 / 1 :=
by
  sorry

end saree_stripe_ratio_l2093_209388


namespace inequality_proof_l2093_209309

def f (x : ℝ) : ℝ := 2 * abs (x - 1) + x - 1

def g (x : ℝ) : ℝ := 16 * x^2 - 8 * x + 1

def M : Set ℝ := {x | f x ≤ 1}

def N : Set ℝ := {x | g x ≤ 4}

theorem inequality_proof (x : ℝ) (hx : x ∈ M ∩ N) : x^2 * f x + x * (f x)^2 ≤ 1/4 := by
  sorry

end inequality_proof_l2093_209309


namespace fabric_price_system_l2093_209319

/-- Represents the price per foot of damask fabric in wen -/
def damask_price : ℝ := sorry

/-- Represents the price per foot of gauze fabric in wen -/
def gauze_price : ℝ := sorry

/-- The length of the damask fabric in feet -/
def damask_length : ℝ := 7

/-- The length of the gauze fabric in feet -/
def gauze_length : ℝ := 9

/-- The price difference per foot between damask and gauze fabrics in wen -/
def price_difference : ℝ := 36

theorem fabric_price_system :
  (damask_length * damask_price = gauze_length * gauze_price) ∧
  (damask_price - gauze_price = price_difference) := by sorry

end fabric_price_system_l2093_209319


namespace larger_cuboid_length_l2093_209333

/-- Proves that the length of a larger cuboid is 18m, given the specified conditions --/
theorem larger_cuboid_length : 
  ∀ (small_length small_width small_height : ℝ)
    (large_width large_height : ℝ)
    (num_small_cuboids : ℕ),
  small_length = 5 →
  small_width = 6 →
  small_height = 3 →
  large_width = 15 →
  large_height = 2 →
  num_small_cuboids = 6 →
  ∃ (large_length : ℝ),
    large_length * large_width * large_height = 
    num_small_cuboids * (small_length * small_width * small_height) ∧
    large_length = 18 := by
sorry


end larger_cuboid_length_l2093_209333


namespace final_distance_after_checkpoints_l2093_209386

/-- Represents the state of a car on the highway -/
structure CarState where
  position : ℝ
  speed : ℝ

/-- Represents a checkpoint on the highway -/
structure Checkpoint where
  position : ℝ
  new_speed : ℝ

/-- Updates the car state after passing a checkpoint -/
def update_car_state (car : CarState) (checkpoint : Checkpoint) : CarState :=
  { position := checkpoint.position, speed := checkpoint.new_speed }

/-- Calculates the final distance between two cars after passing checkpoints -/
def final_distance (initial_distance : ℝ) (initial_speed : ℝ) (checkpoints : List Checkpoint) : ℝ :=
  sorry

/-- Theorem stating the final distance between the cars -/
theorem final_distance_after_checkpoints :
  let initial_distance := 100
  let initial_speed := 60
  let checkpoints := [
    { position := 1000, new_speed := 80 },
    { position := 2000, new_speed := 100 },
    { position := 3000, new_speed := 120 }
  ]
  final_distance initial_distance initial_speed checkpoints = 200 := by
  sorry

end final_distance_after_checkpoints_l2093_209386


namespace cone_sphere_ratio_l2093_209364

/-- Given a sphere and a right circular cone, where:
    - The radius of the cone's base is twice the radius of the sphere
    - The volume of the cone is one-third the volume of the sphere
    Prove that the ratio of the cone's altitude to its base radius is 1/6 -/
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  (4 / 3 * Real.pi * r^2 * h = 1 / 3 * (4 / 3 * Real.pi * r^3)) → 
  (h / (2 * r) = 1 / 6) := by
sorry

end cone_sphere_ratio_l2093_209364


namespace cube_surface_area_l2093_209363

/-- Given three points A, B, and C as vertices of a cube, prove that its surface area is 294 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (1, 4, 2) → B = (2, 0, -7) → C = (5, -5, 1) → 
  (let surface_area := 6 * (dist A B)^2
   surface_area = 294) := by
  sorry

end cube_surface_area_l2093_209363


namespace snowball_theorem_l2093_209398

def snowball_distribution (lucy_snowballs : ℕ) (charlie_extra : ℕ) : Prop :=
  let charlie_initial := lucy_snowballs + charlie_extra
  let linus_received := charlie_initial / 2
  let charlie_final := charlie_initial / 2
  let sally_received := linus_received / 3
  let linus_final := linus_received - sally_received
  charlie_final = 25 ∧ lucy_snowballs = 19 ∧ linus_final = 17 ∧ sally_received = 8

theorem snowball_theorem : snowball_distribution 19 31 := by
  sorry

end snowball_theorem_l2093_209398


namespace pasta_sauce_cost_l2093_209358

/-- The cost of pasta sauce given grocery shopping conditions -/
theorem pasta_sauce_cost 
  (mustard_oil_quantity : ℝ) 
  (mustard_oil_price : ℝ) 
  (pasta_quantity : ℝ) 
  (pasta_price : ℝ) 
  (pasta_sauce_quantity : ℝ) 
  (initial_money : ℝ) 
  (money_left : ℝ) 
  (h1 : mustard_oil_quantity = 2) 
  (h2 : mustard_oil_price = 13) 
  (h3 : pasta_quantity = 3) 
  (h4 : pasta_price = 4) 
  (h5 : pasta_sauce_quantity = 1) 
  (h6 : initial_money = 50) 
  (h7 : money_left = 7) : 
  (initial_money - money_left - (mustard_oil_quantity * mustard_oil_price + pasta_quantity * pasta_price)) / pasta_sauce_quantity = 5 := by
sorry

end pasta_sauce_cost_l2093_209358


namespace peach_tree_count_l2093_209315

theorem peach_tree_count (almond_trees : ℕ) (peach_trees : ℕ) : 
  almond_trees = 300 →
  peach_trees = 2 * almond_trees - 30 →
  peach_trees = 570 := by
sorry

end peach_tree_count_l2093_209315


namespace complex_equation_solution_l2093_209389

theorem complex_equation_solution (a : ℝ) : (a - Complex.I)^2 = 2 * Complex.I → a = -1 := by
  sorry

end complex_equation_solution_l2093_209389
