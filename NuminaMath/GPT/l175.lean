import Mathlib

namespace car_speed_l175_175558

theorem car_speed (v t Δt : ℝ) (h1: 90 = v * t) (h2: 90 = (v + 30) * (t - Δt)) (h3: Δt = 0.5) : 
  ∃ v, 90 = v * t ∧ 90 = (v + 30) * (t - Δt) :=
by {
  sorry
}

end car_speed_l175_175558


namespace problem_1_problem_2_l175_175280

-- Definitions for set A and B when a = 3 for (1)
def A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 ≤ 0 }

-- Theorem for (1)
theorem problem_1 : A ∪ (Bᶜ) = Set.univ := sorry

-- Function to describe B based on a for (2)
def B_a (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a ≤ 0 }
def A_set : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }

-- Theorem for (2)
theorem problem_2 (a : ℝ) : (1 < a ∧ a < 4) → (A_set ∩ B_a a ≠ ∅ ∧ B_a a ⊆ A_set ∧ B_a a ≠ A_set) := sorry

end problem_1_problem_2_l175_175280


namespace mango_production_l175_175193

-- Conditions
def num_papaya_trees := 2
def papayas_per_tree := 10
def num_mango_trees := 3
def total_fruits := 80

-- Definition to be proven
def mangos_per_mango_tree : Nat :=
  (total_fruits - num_papaya_trees * papayas_per_tree) / num_mango_trees

theorem mango_production :
  mangos_per_mango_tree = 20 := by
  sorry

end mango_production_l175_175193


namespace minimum_value_l175_175792

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 128) : 
  ∃ (m : ℝ), (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a * b * c = 128 → (a^2 + 8 * a * b + 4 * b^2 + 8 * c^2) ≥ m) 
  ∧ m = 384 :=
sorry


end minimum_value_l175_175792


namespace desired_salt_percentage_is_ten_percent_l175_175710

-- Define the initial conditions
def initial_pure_water_volume : ℝ := 100
def saline_solution_percentage : ℝ := 0.25
def added_saline_volume : ℝ := 66.67
def total_volume : ℝ := initial_pure_water_volume + added_saline_volume
def added_salt : ℝ := saline_solution_percentage * added_saline_volume
def desired_salt_percentage (P : ℝ) : Prop := added_salt = P * total_volume

-- State the theorem and its result
theorem desired_salt_percentage_is_ten_percent (P : ℝ) (h : desired_salt_percentage P) : P = 0.1 :=
sorry

end desired_salt_percentage_is_ten_percent_l175_175710


namespace monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l175_175370

-- Definition of given conditions regarding tourists count in February and April
def tourists_in_february : ℕ := 16000
def tourists_in_april : ℕ := 25000

-- Theorem 1: Monthly average growth rate of tourists from February to April is 25%.
theorem monthly_avg_growth_rate_25 :
  (tourists_in_april : ℝ) = tourists_in_february * (1 + 0.25)^2 :=
sorry

-- Definition of given conditions for tourists count from May 1st to May 21st
def tourists_may_1_to_21 : ℕ := 21250
def max_total_tourists_may : ℕ := 31250 -- Expressed in thousands as 31.25 in millions

-- Theorem 2: Maximum average number of tourists per day in the next 10 days of May.
theorem max_avg_tourists_next_10_days :
  ∀ (a : ℝ), tourists_may_1_to_21 + 10 * a ≤ max_total_tourists_may →
  a ≤ 10000 :=
sorry

end monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l175_175370


namespace trigonometric_identity_l175_175227

theorem trigonometric_identity :
  Real.tan (70 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) * (Real.sqrt 3 * Real.tan (20 * Real.pi / 180) - 1) = -1 :=
by
  sorry

end trigonometric_identity_l175_175227


namespace smaller_two_digit_product_l175_175513

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l175_175513


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l175_175666

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l175_175666


namespace simplify_expression_l175_175678

variables {a b c : ℝ}
variable (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
variable (h₃ : b - 2 / c ≠ 0)

theorem simplify_expression :
  (a - 2 / b) / (b - 2 / c) = c / b :=
sorry

end simplify_expression_l175_175678


namespace fraction_of_weight_kept_l175_175057

-- Definitions of the conditions
def hunting_trips_per_month := 6
def months_in_season := 3
def deers_per_trip := 2
def weight_per_deer := 600
def weight_kept_per_year := 10800

-- Definition calculating total weight caught in the hunting season
def total_trips := hunting_trips_per_month * months_in_season
def weight_per_trip := deers_per_trip * weight_per_deer
def total_weight_caught := total_trips * weight_per_trip

-- The theorem to prove the fraction
theorem fraction_of_weight_kept : (weight_kept_per_year : ℚ) / (total_weight_caught : ℚ) = 1 / 2 := by
  -- Proof goes here
  sorry

end fraction_of_weight_kept_l175_175057


namespace weight_of_packet_a_l175_175552

theorem weight_of_packet_a
  (A B C D E F : ℝ)
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 79)
  (h5 : F = (A + E) / 2)
  (h6 : (B + C + D + E + F) / 5 = 81) :
  A = 75 :=
by sorry

end weight_of_packet_a_l175_175552


namespace remainder_of_x_squared_div_20_l175_175286

theorem remainder_of_x_squared_div_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 4 * x ≡ 12 [ZMOD 20]) :
  (x * x) % 20 = 4 :=
sorry

end remainder_of_x_squared_div_20_l175_175286


namespace gcd_38_23_l175_175213

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem gcd_38_23 : gcd 38 23 = 1 := by
  sorry

end gcd_38_23_l175_175213


namespace greatest_common_factor_of_three_digit_palindromes_l175_175364

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l175_175364


namespace seats_per_table_l175_175599

-- Definitions based on conditions
def tables := 4
def total_people := 32

-- Statement to prove
theorem seats_per_table : (total_people / tables) = 8 :=
by 
  sorry

end seats_per_table_l175_175599


namespace system_of_equations_solutions_l175_175843

theorem system_of_equations_solutions (x y : ℝ) (h1 : x ^ 5 + y ^ 5 = 1) (h2 : x ^ 6 + y ^ 6 = 1) :
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end system_of_equations_solutions_l175_175843


namespace moles_of_water_produced_l175_175163

theorem moles_of_water_produced (H₃PO₄ NaOH NaH₂PO₄ H₂O : ℝ) (h₁ : H₃PO₄ = 3) (h₂ : NaOH = 3) (h₃ : NaH₂PO₄ = 3) (h₄ : NaH₂PO₄ / H₂O = 1) : H₂O = 3 :=
by
  sorry

end moles_of_water_produced_l175_175163


namespace minimize_sum_of_digits_l175_175863

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the expression in the problem
def expression (p : ℕ) : ℕ :=
  p^4 - 5 * p^2 + 13

-- Proposition stating the conditions and the expected result
theorem minimize_sum_of_digits (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∀ q : ℕ, Nat.Prime q → q % 2 = 1 → sum_of_digits (expression q) ≥ sum_of_digits (expression 5)) →
  p = 5 :=
by
  sorry

end minimize_sum_of_digits_l175_175863


namespace arithmetic_mean_l175_175414

variable (x b : ℝ)

theorem arithmetic_mean (hx : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
  sorry

end arithmetic_mean_l175_175414


namespace sum_of_cubes_l175_175270

theorem sum_of_cubes
  (a b c : ℝ)
  (h₁ : a + b + c = 7)
  (h₂ : ab + ac + bc = 9)
  (h₃ : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end sum_of_cubes_l175_175270


namespace magazines_per_bookshelf_l175_175730

noncomputable def total_books : ℕ := 23
noncomputable def total_books_and_magazines : ℕ := 2436
noncomputable def total_bookshelves : ℕ := 29

theorem magazines_per_bookshelf : (total_books_and_magazines - total_books) / total_bookshelves = 83 :=
by
  sorry

end magazines_per_bookshelf_l175_175730


namespace find_f_l175_175112

theorem find_f (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > 0)
  (h2 : f 1 = 1)
  (h3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2) : ∀ x : ℝ, f x = x := by
  sorry

end find_f_l175_175112


namespace number_of_ways_to_put_balls_in_boxes_l175_175768

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l175_175768


namespace number_of_draws_l175_175145

-- Definition of the competition conditions
def competition_conditions (A B C D E : ℕ) : Prop :=
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = E) ∧
  15 ∣ (10000 * A + 1000 * B + 100 * C + 10 * D + E)

-- The main theorem stating the number of draws
theorem number_of_draws :
  ∃ (A B C D E : ℕ), competition_conditions A B C D E ∧ 
  (∃ (draws : ℕ), draws = 3) :=
by
  sorry

end number_of_draws_l175_175145


namespace problem_statement_l175_175425

theorem problem_statement (m : ℂ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2005 = 2006 :=
  sorry

end problem_statement_l175_175425


namespace daniel_utility_equation_solution_l175_175861

theorem daniel_utility_equation_solution (t : ℚ) :
  t * (10 - t) = (4 - t) * (t + 4) → t = 8 / 5 := by
  sorry

end daniel_utility_equation_solution_l175_175861


namespace geometric_series_first_term_l175_175406

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l175_175406


namespace clean_house_time_l175_175970

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l175_175970


namespace eight_painters_finish_in_required_days_l175_175906

/- Conditions setup -/
def initial_painters : ℕ := 6
def initial_days : ℕ := 2
def job_constant := initial_painters * initial_days

def new_painters : ℕ := 8
def required_days := 3 / 2

/- Theorem statement -/
theorem eight_painters_finish_in_required_days : new_painters * required_days = job_constant :=
sorry

end eight_painters_finish_in_required_days_l175_175906


namespace total_distance_is_1095_l175_175316

noncomputable def totalDistanceCovered : ℕ :=
  let running_first_3_months := 3 * 3 * 10
  let running_next_3_months := 3 * 3 * 20
  let running_last_6_months := 3 * 6 * 30
  let total_running := running_first_3_months + running_next_3_months + running_last_6_months

  let swimming_first_6_months := 3 * 6 * 5
  let total_swimming := swimming_first_6_months

  let total_hiking := 13 * 15

  total_running + total_swimming + total_hiking

theorem total_distance_is_1095 : totalDistanceCovered = 1095 := by
  sorry

end total_distance_is_1095_l175_175316


namespace carla_final_payment_l175_175130

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l175_175130


namespace roots_of_quadratic_expression_l175_175151

theorem roots_of_quadratic_expression :
    (∀ x: ℝ, (x^2 + 3 * x - 2 = 0) → ∃ x₁ x₂: ℝ, x = x₁ ∨ x = x₂) ∧ 
    (∀ x₁ x₂ : ℝ, (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -2) → x₁^2 + 2 * x₁ - x₂ = 5) :=
by
  sorry

end roots_of_quadratic_expression_l175_175151


namespace fruit_selling_price_3640_l175_175499

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l175_175499


namespace allowance_spent_l175_175807

variable (A x y : ℝ)
variable (h1 : x = 0.20 * (A - y))
variable (h2 : y = 0.05 * (A - x))

theorem allowance_spent : (x + y) / A = 23 / 100 :=
by 
  sorry

end allowance_spent_l175_175807


namespace remaining_wax_l175_175905

-- Define the conditions
def ounces_for_car : ℕ := 3
def ounces_for_suv : ℕ := 4
def initial_wax : ℕ := 11
def spilled_wax : ℕ := 2

-- Define the proof problem: Show remaining wax after detailing car and SUV
theorem remaining_wax {ounces_for_car ounces_for_suv initial_wax spilled_wax : ℕ} :
  initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv) = 2 :=
by
  -- Defining the variables according to the conditions
  have h1 : ounces_for_car = 3 := rfl
  have h2 : ounces_for_suv = 4 := rfl
  have h3 : initial_wax = 11 := rfl
  have h4 : spilled_wax = 2 := rfl
  -- Using the conditions to calculate the remaining wax
  calc
    initial_wax - spilled_wax - (ounces_for_car + ounces_for_suv)
        = 11 - 2 - (3 + 4) : by rw [h1, h2, h3, h4]
    ... = 11 - 2 - 7 : rfl
    ... = 9 - 7 : rfl
    ... = 2 : rfl

end remaining_wax_l175_175905


namespace total_amount_earned_is_90_l175_175114

variable (W : ℕ)

-- Define conditions
def work_capacity_condition : Prop :=
  5 = W ∧ W = 8

-- Define wage per man in Rs.
def wage_per_man : ℕ := 6

-- Define total amount earned by 5 men
def total_earned_by_5_men : ℕ := 5 * wage_per_man

-- Define total amount for the problem
def total_earned (W : ℕ) : ℕ :=
  3 * total_earned_by_5_men

-- The final proof statement
theorem total_amount_earned_is_90 (W : ℕ) (h : work_capacity_condition W) : total_earned W = 90 := by
  sorry

end total_amount_earned_is_90_l175_175114


namespace nails_needed_l175_175161

theorem nails_needed (nails_own nails_found nails_total_needed : ℕ) 
  (h1 : nails_own = 247) 
  (h2 : nails_found = 144) 
  (h3 : nails_total_needed = 500) : 
  nails_total_needed - (nails_own + nails_found) = 109 := 
by
  sorry

end nails_needed_l175_175161


namespace tv_cost_l175_175795

theorem tv_cost (savings : ℕ) (fraction_spent_on_furniture : ℚ) (amount_spent_on_furniture : ℚ) (remaining_savings : ℚ) :
  savings = 1000 →
  fraction_spent_on_furniture = 3/5 →
  amount_spent_on_furniture = fraction_spent_on_furniture * savings →
  remaining_savings = savings - amount_spent_on_furniture →
  remaining_savings = 400 :=
by
  sorry

end tv_cost_l175_175795


namespace matt_twice_james_age_in_5_years_l175_175380

theorem matt_twice_james_age_in_5_years :
  (∃ x : ℕ, (3 + 27 = 30) ∧ (Matt_current_age = 65) ∧ 
  (Matt_age_in_x_years = Matt_current_age + x) ∧ 
  (James_age_in_x_years = James_current_age + x) ∧ 
  (Matt_age_in_x_years = 2 * James_age_in_x_years) → x = 5) :=
sorry

end matt_twice_james_age_in_5_years_l175_175380


namespace find_original_sales_tax_percentage_l175_175516

noncomputable def original_sales_tax_percentage (x : ℝ) : Prop :=
∃ (x : ℝ),
  let reduced_tax := 10 / 3 / 100;
  let market_price := 9000;
  let difference := 14.999999999999986;
  (x / 100 * market_price - reduced_tax * market_price = difference) ∧ x = 0.5

theorem find_original_sales_tax_percentage : original_sales_tax_percentage 0.5 :=
sorry

end find_original_sales_tax_percentage_l175_175516


namespace correct_calculation_of_exponentiation_l175_175367

theorem correct_calculation_of_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end correct_calculation_of_exponentiation_l175_175367


namespace sum_of_n_values_l175_175535

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l175_175535


namespace b_investment_l175_175582

theorem b_investment (A_invest C_invest total_profit A_profit x : ℝ) 
(h1 : A_invest = 2400) 
(h2 : C_invest = 9600) 
(h3 : total_profit = 9000) 
(h4 : A_profit = 1125)
(h5 : x = (8100000 / 1125)) : 
x = 7200 := by
  rw [h5]
  sorry

end b_investment_l175_175582


namespace solve_for_x_l175_175925

-- Defining the given conditions
def y : ℕ := 6
def lhs (x : ℕ) : ℕ := Nat.pow x y
def rhs : ℕ := Nat.pow 3 12

-- Theorem statement to prove
theorem solve_for_x (x : ℕ) (hypothesis : lhs x = rhs) : x = 9 :=
by sorry

end solve_for_x_l175_175925


namespace geom_series_first_term_l175_175399

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l175_175399


namespace isosceles_triangle_sides_l175_175614

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l175_175614


namespace largest_among_trig_expressions_l175_175127

theorem largest_among_trig_expressions :
  let a := Real.tan 48 + 1 / Real.tan 48
  let b := Real.sin 48 + Real.cos 48
  let c := Real.tan 48 + Real.cos 48
  let d := 1 / Real.tan 48 + Real.sin 48
  a > b ∧ a > c ∧ a > d :=
by
  sorry

end largest_among_trig_expressions_l175_175127


namespace find_shirt_numbers_calculate_profit_l175_175846

def total_shirts_condition (x y : ℕ) : Prop := x + y = 200
def total_cost_condition (x y : ℕ) : Prop := 25 * x + 15 * y = 3500
def profit_calculation (x y : ℕ) : ℕ := (50 - 25) * x + (35 - 15) * y

theorem find_shirt_numbers (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  x = 50 ∧ y = 150 :=
sorry

theorem calculate_profit (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  profit_calculation x y = 4250 :=
sorry

end find_shirt_numbers_calculate_profit_l175_175846


namespace contributions_before_john_l175_175897

theorem contributions_before_john (n : ℕ) (A : ℚ) 
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 225) / (n + 1) = 75) : n = 6 :=
by {
  sorry
}

end contributions_before_john_l175_175897


namespace find_a2_geometric_sequence_l175_175875

theorem find_a2_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) 
  (h_a1 : a 1 = 1 / 4) (h_eq : a 3 * a 5 = 4 * (a 4 - 1)) : a 2 = 1 / 8 :=
by
  sorry

end find_a2_geometric_sequence_l175_175875


namespace evaluate_expression_l175_175857

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 :=
by
  sorry

end evaluate_expression_l175_175857


namespace clean_house_time_l175_175968

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l175_175968


namespace area_is_rational_l175_175877

-- Definitions of the vertices of the triangle
def point1 : (ℤ × ℤ) := (2, 3)
def point2 : (ℤ × ℤ) := (5, 7)
def point3 : (ℤ × ℤ) := (3, 4)

-- Define a function to calculate the area of a triangle given vertices with integer coordinates
def triangle_area (A B C: (ℤ × ℤ)) : ℚ :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

-- Define the area of our specific triangle
noncomputable def area_of_triangle_with_given_vertices := triangle_area point1 point2 point3

-- Proof statement
theorem area_is_rational : ∃ (Q : ℚ), Q = area_of_triangle_with_given_vertices := 
sorry

end area_is_rational_l175_175877


namespace smaller_two_digit_product_l175_175511

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l175_175511


namespace part_a_part_b_part_c_l175_175036

open Real

-- Definition of Parabola
def parabola (x : ℝ) : ℝ := x^2 - x - 2

-- Points on the parabola
def point_on_parabola (x y : ℝ) : Prop := y = parabola x

-- Definition of a line equation
def line_eq (m b x : ℝ) : ℝ := m * x + b

-- Part (a)
theorem part_a (x1 x2 y1 y2 : ℝ) (hP : point_on_parabola x1 y1) (hQ : point_on_parabola x2 y2)
  (hm : (x1 + x2) / 2 = 0) (hm' : (y1 + y2) / 2 = 0) : 
  ∃ m, (∀ x, line_eq m 0 x = -x) := sorry

-- Part (b)
theorem part_b (x1 x2 y1 y2 : ℝ) (hP : point_on_parabola x1 y1) (hQ : point_on_parabola x2 y2)
  (hr : (2 * x2 + x1) / 3 = 0) (hr' : (2 * y2 + y1) / 3 = 0) : 
  ∃ m, (∀ x, line_eq m 0 x = -2 * x) := sorry

-- Part (c)
theorem part_c (x1 x2 : ℝ) (h_int : parabola x1 = line_eq (-2) 0 x1 ∧ parabola x2 = line_eq (-2) 0 x2)
  (hP : x1 < x2) : 
  ∃ A, A = 9 / 2 := sorry

end part_a_part_b_part_c_l175_175036


namespace collinear_k_perpendicular_k_l175_175890

def vector := ℝ × ℝ

def a : vector := (1, 3)
def b : vector := (3, -4)

def collinear (u v : vector) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : vector) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def k_vector_a_minus_b (k : ℝ) (a b : vector) : vector :=
  (k * a.1 - b.1, k * a.2 - b.2)

def a_plus_b (a b : vector) : vector :=
  (a.1 + b.1, a.2 + b.2)

theorem collinear_k (k : ℝ) : collinear (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = -1 :=
sorry

theorem perpendicular_k (k : ℝ) : perpendicular (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = 16 :=
sorry

end collinear_k_perpendicular_k_l175_175890


namespace distinct_points_count_l175_175736

-- Definitions based on conditions
def eq1 (x y : ℝ) : Prop := (x + y = 7) ∨ (2 * x - 3 * y = -7)
def eq2 (x y : ℝ) : Prop := (x - y = 3) ∨ (3 * x + 2 * y = 18)

-- The statement combining conditions and requiring the proof of 3 distinct solutions
theorem distinct_points_count : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (eq1 p1.1 p1.2 ∧ eq2 p1.1 p1.2) ∧ 
    (eq1 p2.1 p2.2 ∧ eq2 p2.1 p2.2) ∧ 
    (eq1 p3.1 p3.2 ∧ eq2 p3.1 p3.2) ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end distinct_points_count_l175_175736


namespace find_m_if_purely_imaginary_l175_175927

theorem find_m_if_purely_imaginary : ∀ m : ℝ, (m^2 - 5*m + 6 = 0) → (m = 2) :=
by 
  intro m
  intro h
  sorry

end find_m_if_purely_imaginary_l175_175927


namespace eval_expression_l175_175554

theorem eval_expression : 1999^2 - 1998 * 2002 = -3991 := 
by
  sorry

end eval_expression_l175_175554


namespace time_for_embankments_l175_175355

theorem time_for_embankments (rate : ℚ) (t1 t2 : ℕ) (w1 w2 : ℕ)
    (h1 : w1 = 75) (h2 : w2 = 60) (h3 : t1 = 4)
    (h4 : rate = 1 / (w1 * t1 : ℚ)) 
    (h5 : t2 = 1 / (w2 * rate)) : 
    t1 + t2 = 9 :=
sorry

end time_for_embankments_l175_175355


namespace ethan_presents_l175_175600

variable (A E : ℝ)

theorem ethan_presents (h1 : A = 9) (h2 : A = E - 22.0) : E = 31 := 
by
  sorry

end ethan_presents_l175_175600


namespace sum_of_solutions_abs_eq_l175_175540

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l175_175540


namespace roger_left_money_correct_l175_175077

noncomputable def roger_left_money (P : ℝ) (q : ℝ) (E : ℝ) (r1 : ℝ) (C : ℝ) (r2 : ℝ) : ℝ :=
  let feb_expense := q * P
  let after_feb := P - feb_expense
  let mar_expense := E * r1
  let after_mar := after_feb - mar_expense
  let mom_gift := C * r2
  after_mar + mom_gift

theorem roger_left_money_correct :
  roger_left_money 45 0.35 20 1.2 46 0.8 = 42.05 :=
by
  sorry

end roger_left_money_correct_l175_175077


namespace arithmetic_sequence_common_difference_l175_175278

theorem arithmetic_sequence_common_difference (a : Nat → Int)
  (h1 : a 1 = 2) 
  (h3 : a 3 = 8)
  (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- General form for an arithmetic sequence given two terms
  : a 2 - a 1 = 3 :=
by
  -- The main steps of the proof will follow from the arithmetic progression properties
  sorry

end arithmetic_sequence_common_difference_l175_175278


namespace p_over_q_at_neg1_l175_175506

-- Definitions of p(x) and q(x) based on given conditions
noncomputable def q (x : ℝ) := (x + 3) * (x - 2)
noncomputable def p (x : ℝ) := 2 * x

-- Define the main function y = p(x) / q(x)
noncomputable def y (x : ℝ) := p x / q x

-- Statement to prove the value of p(-1) / q(-1)
theorem p_over_q_at_neg1 : y (-1) = (1 : ℝ) / 3 :=
by
  sorry

end p_over_q_at_neg1_l175_175506


namespace participation_schemes_count_l175_175421

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count_l175_175421


namespace roots_cubic_sum_cubes_l175_175318

theorem roots_cubic_sum_cubes (a b c : ℝ) 
    (h1 : 6 * a^3 - 803 * a + 1606 = 0)
    (h2 : 6 * b^3 - 803 * b + 1606 = 0)
    (h3 : 6 * c^3 - 803 * c + 1606 = 0) :
    (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := 
by
  sorry

end roots_cubic_sum_cubes_l175_175318


namespace largest_bucket_capacity_l175_175038

-- Let us define the initial conditions
def capacity_5_liter_bucket : ℕ := 5
def capacity_3_liter_bucket : ℕ := 3
def remaining_after_pour := capacity_5_liter_bucket - capacity_3_liter_bucket
def additional_capacity_without_overflow : ℕ := 4

-- Problem statement: Prove that the capacity of the largest bucket is 6 liters
theorem largest_bucket_capacity : ∀ (c : ℕ), remaining_after_pour + additional_capacity_without_overflow = c → c = 6 := 
by
  sorry

end largest_bucket_capacity_l175_175038


namespace percentage_exceeds_l175_175568

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l175_175568


namespace perpendicular_vectors_vector_sum_norm_min_value_f_l175_175891

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3*x/2), Real.sin (3*x/2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem perpendicular_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 ↔ x = Real.pi / 4 := sorry

theorem vector_sum_norm (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ 1 ↔ 0 ≤ x ∧ x ≤ Real.pi / 3 := sorry

theorem min_value_f (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≥ -2) ↔ m = Real.sqrt 2 / 2 := sorry

end perpendicular_vectors_vector_sum_norm_min_value_f_l175_175891


namespace jim_reads_less_hours_l175_175311

-- Conditions
def initial_speed : ℕ := 40 -- pages per hour
def initial_pages_per_week : ℕ := 600 -- pages
def speed_increase_factor : ℚ := 1.5
def new_pages_per_week : ℕ := 660 -- pages

-- Calculations based on conditions
def initial_hours_per_week : ℚ := initial_pages_per_week / initial_speed
def new_speed : ℚ := initial_speed * speed_increase_factor
def new_hours_per_week : ℚ := new_pages_per_week / new_speed

-- Theorem Statement
theorem jim_reads_less_hours :
  initial_hours_per_week - new_hours_per_week = 4 :=
  sorry

end jim_reads_less_hours_l175_175311


namespace infinite_equal_pairs_of_equal_terms_l175_175148

theorem infinite_equal_pairs_of_equal_terms {a : ℤ → ℤ}
  (h : ∀ n, a n = (a (n - 1) + a (n + 1)) / 4)
  (i j : ℤ) (hij : a i = a j) :
  ∃ (infinitely_many_pairs : ℕ → ℤ × ℤ), ∀ k, a (infinitely_many_pairs k).1 = a (infinitely_many_pairs k).2 :=
sorry

end infinite_equal_pairs_of_equal_terms_l175_175148


namespace number_of_students_in_first_class_l175_175088

theorem number_of_students_in_first_class 
  (x : ℕ) -- number of students in the first class
  (avg_first_class : ℝ := 50) 
  (num_second_class : ℕ := 50)
  (avg_second_class : ℝ := 60)
  (avg_all_students : ℝ := 56.25)
  (total_avg_eqn : (avg_first_class * x + avg_second_class * num_second_class) / (x + num_second_class) = avg_all_students) : 
  x = 30 :=
by sorry

end number_of_students_in_first_class_l175_175088


namespace housewife_more_oil_l175_175139

theorem housewife_more_oil 
    (reduction_percent : ℝ := 10)
    (reduced_price : ℝ := 16)
    (budget : ℝ := 800)
    (approx_answer : ℝ := 5.01) :
    let P := reduced_price / (1 - reduction_percent / 100)
    let Q_original := budget / P
    let Q_reduced := budget / reduced_price
    let delta_Q := Q_reduced - Q_original
    abs (delta_Q - approx_answer) < 0.02 := 
by
  -- Let the goal be irrelevant to the proof because the proof isn't provided
  sorry

end housewife_more_oil_l175_175139


namespace largest_number_from_hcf_factors_l175_175083

/-- This statement checks the largest number derivable from given HCF and factors. -/
theorem largest_number_from_hcf_factors (HCF factor1 factor2 : ℕ) (hHCF : HCF = 52) (hfactor1 : factor1 = 11) (hfactor2 : factor2 = 12) :
  max (HCF * factor1) (HCF * factor2) = 624 :=
by
  sorry

end largest_number_from_hcf_factors_l175_175083


namespace smallest_b_for_factorization_l175_175869

theorem smallest_b_for_factorization : ∃ (p q : ℕ), p * q = 2007 ∧ p + q = 232 :=
by
  sorry

end smallest_b_for_factorization_l175_175869


namespace inverseP_l175_175754

-- Mathematical definitions
def isOdd (a : ℕ) : Prop := a % 2 = 1
def isPrime (a : ℕ) : Prop := Nat.Prime a

-- Given proposition P (hypothesis)
def P (a : ℕ) : Prop := isOdd a → isPrime a

-- Inverse proposition: if a is prime, then a is odd
theorem inverseP (a : ℕ) (h : isPrime a) : isOdd a :=
sorry

end inverseP_l175_175754


namespace prob_red_ball_is_three_fifths_l175_175902

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - (yellow_balls + green_balls)
def total_probability : ℚ := 1
def probability_of_red_ball : ℚ := red_balls / total_balls

theorem prob_red_ball_is_three_fifths :
  probability_of_red_ball = 3 / 5 :=
begin
  sorry
end

end prob_red_ball_is_three_fifths_l175_175902


namespace find_X_l175_175442

theorem find_X : 
  let M := 3012 / 4
  let N := M / 4
  let X := M - N
  X = 564.75 :=
by
  sorry

end find_X_l175_175442


namespace find_eccentricity_l175_175343

variables {a b x_N x_M : ℝ}
variable {e : ℝ}

-- Conditions
def line_passes_through_N (x_N : ℝ) (x_M : ℝ) : Prop :=
x_N ≠ 0 ∧ x_N = 4 * x_M

def hyperbola (x y a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def midpoint_x_M (x_M : ℝ) : Prop :=
∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = x_M

-- Proof Problem
theorem find_eccentricity
  (hN : line_passes_through_N x_N x_M)
  (hC : hyperbola x_N 0 a b)
  (hM : midpoint_x_M x_M) :
  e = 2 :=
sorry

end find_eccentricity_l175_175343


namespace a_range_of_proposition_l175_175748

theorem a_range_of_proposition (a : ℝ) : (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 <= a * x) ↔ a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end a_range_of_proposition_l175_175748


namespace find_number_90_l175_175454

theorem find_number_90 {x y : ℝ} (h1 : x = y + 0.11 * y) (h2 : x = 99.9) : y = 90 :=
sorry

end find_number_90_l175_175454


namespace add_inequality_of_greater_l175_175150

theorem add_inequality_of_greater (a b c d : ℝ) (h₁ : a > b) (h₂ : c > d) : a + c > b + d := 
by sorry

end add_inequality_of_greater_l175_175150


namespace geom_series_first_term_l175_175397

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l175_175397


namespace least_possible_sum_l175_175173

theorem least_possible_sum (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x + y + z = 26 :=
by sorry

end least_possible_sum_l175_175173


namespace minimum_workers_needed_l175_175522

noncomputable def units_per_first_worker : Nat := 48
noncomputable def units_per_second_worker : Nat := 32
noncomputable def units_per_third_worker : Nat := 28

def minimum_workers_first_process : Nat := 14
def minimum_workers_second_process : Nat := 21
def minimum_workers_third_process : Nat := 24

def lcm_3_nat (a b c : Nat) : Nat :=
  Nat.lcm (Nat.lcm a b) c

theorem minimum_workers_needed (a b c : Nat) (w1 w2 w3 : Nat)
  (h1 : a = 48) (h2 : b = 32) (h3 : c = 28)
  (hw1 : w1 = minimum_workers_first_process )
  (hw2 : w2 = minimum_workers_second_process )
  (hw3 : w3 = minimum_workers_third_process ) :
  lcm_3_nat a b c / a = w1 ∧ lcm_3_nat a b c / b = w2 ∧ lcm_3_nat a b c / c = w3 :=
by
  sorry

end minimum_workers_needed_l175_175522


namespace margo_total_distance_travelled_l175_175483

noncomputable def total_distance_walked (walking_time_in_minutes: ℝ) (stopping_time_in_minutes: ℝ) (additional_walking_time_in_minutes: ℝ) (walking_speed: ℝ) : ℝ :=
  walking_speed * ((walking_time_in_minutes + stopping_time_in_minutes + additional_walking_time_in_minutes) / 60)

noncomputable def total_distance_cycled (cycling_time_in_minutes: ℝ) (cycling_speed: ℝ) : ℝ :=
  cycling_speed * (cycling_time_in_minutes / 60)

theorem margo_total_distance_travelled :
  let walking_time := 10
  let stopping_time := 15
  let additional_walking_time := 10
  let cycling_time := 15
  let walking_speed := 4
  let cycling_speed := 10

  total_distance_walked walking_time stopping_time additional_walking_time walking_speed +
  total_distance_cycled cycling_time cycling_speed = 4.8333 := 
by 
  sorry

end margo_total_distance_travelled_l175_175483


namespace hilary_corn_shucking_l175_175634

theorem hilary_corn_shucking : 
    (total_ears : ℕ) (total_stalks : ℕ) (half_ears_kernels : ℕ) (other_half_ears_kernels : ℕ) 
    (ears_per_stalk : ℕ) (stalks : ℕ) 
    (h1 : ears_per_stalk = 4) 
    (h2 : stalks = 108) 
    (h3 : half_ears_kernels = 500) 
    (h4 : other_half_ears_kernels = 600) : 
    let total_ears := stalks * ears_per_stalk
    let half_ears := total_ears / 2 in
    total_ears * half_ears_kernels / 2 + total_ears * other_half_ears_kernels / 2 = 237600 :=
by 
    intros
    rw [h1, h2, h3, h4]
    sorry

end hilary_corn_shucking_l175_175634


namespace carla_final_payment_l175_175129

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l175_175129


namespace total_people_served_l175_175320

variable (total_people : ℕ)
variable (people_not_buy_coffee : ℕ := 10)

theorem total_people_served (H : (2 / 5 : ℚ) * total_people = people_not_buy_coffee) : total_people = 25 := 
by
  sorry

end total_people_served_l175_175320


namespace hyperbola_standard_eq_proof_l175_175290

noncomputable def real_axis_length := 6
noncomputable def asymptote_slope := 3 / 2

def hyperbola_standard_eq (a b : ℝ) :=
  ∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1)

theorem hyperbola_standard_eq_proof (a b : ℝ) 
  (h_a : 2 * a = real_axis_length)
  (h_b : a / b = asymptote_slope) :
  hyperbola_standard_eq 3 2 := 
by
  sorry

end hyperbola_standard_eq_proof_l175_175290


namespace number_of_moles_of_H2O_l175_175267

def reaction_stoichiometry (n_NaOH m_Cl2 : ℕ) : ℕ :=
  1  -- Moles of H2O produced according to the balanced equation with the given reactants

theorem number_of_moles_of_H2O 
  (n_NaOH : ℕ) (m_Cl2 : ℕ) 
  (h_NaOH : n_NaOH = 2) 
  (h_Cl2 : m_Cl2 = 1) :
  reaction_stoichiometry n_NaOH m_Cl2 = 1 :=
by
  rw [h_NaOH, h_Cl2]
  -- Would typically follow with the proof using the conditions and stoichiometric relation
  sorry  -- Proof step omitted

end number_of_moles_of_H2O_l175_175267


namespace three_distinct_solutions_no_solution_for_2009_l175_175066

-- Problem 1: Show that the equation has at least three distinct solutions if it has one
theorem three_distinct_solutions (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3*x1*y1^2 + y1^3 = n ∧ 
    x2^3 - 3*x2*y2^2 + y2^3 = n ∧ 
    x3^3 - 3*x3*y3^2 + y3^3 = n ∧ 
    (x1, y1) ≠ (x2, y2) ∧ 
    (x1, y1) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x3, y3)) :=
sorry

-- Problem 2: Show that the equation has no solutions when n = 2009
theorem no_solution_for_2009 :
  ¬ ∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = 2009 :=
sorry

end three_distinct_solutions_no_solution_for_2009_l175_175066


namespace pre_image_of_f_5_1_l175_175092

def f (x y : ℝ) : ℝ × ℝ := (x + y, 2 * x - y)

theorem pre_image_of_f_5_1 : ∃ (x y : ℝ), f x y = (5, 1) ∧ (x, y) = (2, 3) :=
by
  sorry

end pre_image_of_f_5_1_l175_175092


namespace general_term_arithmetic_sequence_l175_175178

-- Consider an arithmetic sequence {a_n}
variable (a : ℕ → ℤ)

-- Conditions
def a1 : Prop := a 1 = 1
def a3 : Prop := a 3 = -3
def is_arithmetic_sequence : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Theorem statement
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 1) (h3 : a 3 = -3) (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 3 - 2 * n :=
by
  sorry  -- proof is not required

end general_term_arithmetic_sequence_l175_175178


namespace keiko_speed_l175_175908

theorem keiko_speed (a b s : ℝ) 
  (width : ℝ := 8) 
  (radius_inner := b) 
  (radius_outer := b + width)
  (time_difference := 48) 
  (L_inner := 2 * a + 2 * Real.pi * radius_inner)
  (L_outer := 2 * a + 2 * Real.pi * radius_outer) :
  (L_outer / s = L_inner / s + time_difference) → 
  s = Real.pi / 3 :=
by 
  sorry

end keiko_speed_l175_175908


namespace red_ball_probability_l175_175115

theorem red_ball_probability 
  (red_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : black_balls = 9)
  (h3 : total_balls = red_balls + black_balls) :
  (red_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end red_ball_probability_l175_175115


namespace g_triple_of_10_l175_175983

def g (x : Int) : Int :=
  if x < 4 then x^2 - 9 else x + 7

theorem g_triple_of_10 : g (g (g 10)) = 31 := by
  sorry

end g_triple_of_10_l175_175983


namespace total_cats_l175_175796

theorem total_cats (a b c d : ℝ) (ht : a = 15.5) (hs : b = 11.6) (hg : c = 24.2) (hr : d = 18.3) :
  a + b + c + d = 69.6 :=
by
  sorry

end total_cats_l175_175796


namespace graph_passes_through_quadrants_l175_175945

-- Definitions based on the conditions
def linear_function (x : ℝ) : ℝ := -2 * x + 1

-- The property to be proven
theorem graph_passes_through_quadrants :
  (∃ x > 0, linear_function x > 0) ∧  -- Quadrant I
  (∃ x < 0, linear_function x > 0) ∧  -- Quadrant II
  (∃ x > 0, linear_function x < 0) := -- Quadrant IV
sorry

end graph_passes_through_quadrants_l175_175945


namespace arithmetic_sequence_term_number_l175_175345

theorem arithmetic_sequence_term_number :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (a 1 = 1) →
    (∀ m, a (m + 1) = a m + 3) →
    (a n = 2014) →
    n = 672 :=
by
  -- conditions
  intro a n h1 h2 h3
  -- proof skipped
  sorry

end arithmetic_sequence_term_number_l175_175345


namespace gcd_256_180_720_l175_175829

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l175_175829


namespace lcm_48_180_l175_175417

theorem lcm_48_180 : Int.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l175_175417


namespace percentage_exceeds_l175_175567

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l175_175567


namespace carter_total_drum_sticks_l175_175131

def sets_per_show_used := 5
def sets_per_show_tossed := 6
def nights := 30

theorem carter_total_drum_sticks : 
  (sets_per_show_used + sets_per_show_tossed) * nights = 330 := by
  sorry

end carter_total_drum_sticks_l175_175131


namespace Da_Yan_sequence_20th_term_l175_175334

noncomputable def Da_Yan_sequence_term (n: ℕ) : ℕ :=
  if n % 2 = 0 then
    (n^2) / 2
  else
    (n^2 - 1) / 2

theorem Da_Yan_sequence_20th_term : Da_Yan_sequence_term 20 = 200 :=
by
  sorry

end Da_Yan_sequence_20th_term_l175_175334


namespace slope_of_line_dividing_rectangle_l175_175519

theorem slope_of_line_dividing_rectangle (h_vertices : 
  ∃ (A B C D : ℝ × ℝ), A = (1, 0) ∧ B = (9, 0) ∧ C = (1, 2) ∧ D = (9, 2) ∧ 
  (∃ line : ℝ × ℝ, line = (0, 0) ∧ line = (5, 1))) : 
  ∃ m : ℝ, m = 1 / 5 :=
sorry

end slope_of_line_dividing_rectangle_l175_175519


namespace mean_weight_of_cats_l175_175098

def weight_list : List ℝ :=
  [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

noncomputable def total_weight : ℝ := weight_list.sum

noncomputable def mean_weight : ℝ := total_weight / weight_list.length

theorem mean_weight_of_cats : mean_weight = 101.64 := by
  sorry

end mean_weight_of_cats_l175_175098


namespace find_positive_number_l175_175010

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l175_175010


namespace remainder_5_pow_2048_mod_17_l175_175868

theorem remainder_5_pow_2048_mod_17 : (5 ^ 2048) % 17 = 0 :=
by
  sorry

end remainder_5_pow_2048_mod_17_l175_175868


namespace sandy_savings_l175_175789

-- Definition and conditions
def last_year_savings (S : ℝ) : ℝ := 0.06 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_savings (S : ℝ) : ℝ := 1.8333333333333333 * last_year_savings S

-- The percentage P of this year's salary that Sandy saved
def this_year_savings_perc (S : ℝ) (P : ℝ) : Prop :=
  P * this_year_salary S = this_year_savings S

-- The proof statement: Sandy saved 10% of her salary this year
theorem sandy_savings (S : ℝ) (P : ℝ) (h: this_year_savings_perc S P) : P = 0.10 :=
  sorry

end sandy_savings_l175_175789


namespace find_some_value_l175_175372

theorem find_some_value (m n : ℝ) (some_value : ℝ) 
  (h₁ : m = n / 2 - 2 / 5)
  (h₂ : m + 2 = (n + some_value) / 2 - 2 / 5) :
  some_value = 4 := 
sorry

end find_some_value_l175_175372


namespace paco_salty_cookies_left_l175_175191

theorem paco_salty_cookies_left (S₁ S₂ : ℕ) (h₁ : S₁ = 6) (e1_eaten : ℕ) (a₁ : e1_eaten = 3)
(h₂ : S₂ = 24) (r1_ratio : ℚ) (a_ratio : r1_ratio = (2/3)) :
  S₁ - e1_eaten + r1_ratio * S₂ = 19 :=
by
  sorry

end paco_salty_cookies_left_l175_175191


namespace equal_sum_sequence_S_9_l175_175412

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions taken from the problem statement
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Lean statement of the problem
theorem equal_sum_sequence_S_9
  (h1 : equal_sum_sequence a 5)
  (h2 : a 1 = 2)
  : sum_first_n_terms a 9 = 22 :=
sorry

end equal_sum_sequence_S_9_l175_175412


namespace sqrt_of_product_eq_540_l175_175253

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l175_175253


namespace chord_length_of_concentric_circles_l175_175085

theorem chord_length_of_concentric_circles 
  (R r : ℝ) (h1 : R^2 - r^2 = 15) (h2 : ∀ s, s = 2 * R) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ ∀ x, x = c := 
by 
  sorry

end chord_length_of_concentric_circles_l175_175085


namespace line_parallel_not_coincident_l175_175813

theorem line_parallel_not_coincident (a : ℝ) :
  (a = 3) ↔ (∀ x y, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y + 7 - a = 0) → 
              (∃ k : Real, a / 3 = k ∧ k ≠ 3 * a / (7 - a))) :=
by
  sorry

end line_parallel_not_coincident_l175_175813


namespace original_triangle_area_l175_175339

theorem original_triangle_area (A_new : ℝ) (scale_factor : ℝ) (A_original : ℝ) 
  (h1: scale_factor = 5) (h2: A_new = 200) (h3: A_new = scale_factor^2 * A_original) : 
  A_original = 8 :=
by
  sorry

end original_triangle_area_l175_175339


namespace time_to_travel_to_shop_l175_175005

-- Define the distance and speed as given conditions
def distance : ℕ := 184
def speed : ℕ := 23

-- Define the time taken for the journey
def time_taken (d : ℕ) (s : ℕ) : ℕ := d / s

-- Statement to prove that the time taken is 8 hours
theorem time_to_travel_to_shop : time_taken distance speed = 8 := by
  -- The proof is omitted
  sorry

end time_to_travel_to_shop_l175_175005


namespace roshini_spent_on_sweets_l175_175948

theorem roshini_spent_on_sweets
  (initial_amount : Real)
  (amount_given_per_friend : Real)
  (num_friends : Nat)
  (total_amount_given : Real)
  (amount_spent_on_sweets : Real) :
  initial_amount = 10.50 →
  amount_given_per_friend = 3.40 →
  num_friends = 2 →
  total_amount_given = amount_given_per_friend * num_friends →
  amount_spent_on_sweets = initial_amount - total_amount_given →
  amount_spent_on_sweets = 3.70 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end roshini_spent_on_sweets_l175_175948


namespace islanders_statements_l175_175839

-- Define the basic setup: roles of individuals
inductive Role
| knight
| liar
open Role

-- Define each individual's statement
def A_statement := λ (distance : ℕ), distance = 1
def B_statement := λ (distance : ℕ), distance = 2

-- Prove that given the conditions, the possible distances mentioned by the third and fourth islanders can be as specified
theorem islanders_statements :
  ∃ (C_statement D_statement : ℕ → Prop),
  (∀ distance, C_statement distance ↔ distance ∈ {1, 3, 4}) ∧ (∀ distance, D_statement distance ↔ distance = 2) :=
by
  sorry

end islanders_statements_l175_175839


namespace exists_n_l175_175658

open Nat

theorem exists_n (p a k : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hk : p^a < k ∧ k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (binom n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
by
  sorry

end exists_n_l175_175658


namespace geom_series_first_term_l175_175403

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l175_175403


namespace range_of_m_l175_175880

variable (a b c m y1 y2 y3 : Real)

-- Given points and the parabola equation
def on_parabola (x y a b c : Real) : Prop := y = a * x^2 + b * x + c

-- Conditions
variable (hP : on_parabola (-2) y1 a b c)
variable (hQ : on_parabola 4 y2 a b c)
variable (hM : on_parabola m y3 a b c)
variable (h_vertex : 2 * a * m + b = 0)
variable (h_y_order : y3 ≥ y2 ∧ y2 > y1)

-- Theorem to prove m > 1
theorem range_of_m : m > 1 :=
sorry

end range_of_m_l175_175880


namespace valid_interval_for_a_l175_175410

theorem valid_interval_for_a (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) ↔ (2 / 3 ≤ a ∧ a < 2 ∧ a ≠ 5 / 3) :=
by
  sorry

end valid_interval_for_a_l175_175410


namespace remainder_div_13_l175_175847

theorem remainder_div_13 {N : ℕ} (k : ℕ) (h : N = 39 * k + 18) : N % 13 = 5 := sorry

end remainder_div_13_l175_175847


namespace find_positive_number_l175_175011

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l175_175011


namespace final_price_is_correct_l175_175020

-- Define the original price
def original_price : ℝ := 10

-- Define the first reduction percentage
def first_reduction_percentage : ℝ := 0.30

-- Define the second reduction percentage
def second_reduction_percentage : ℝ := 0.50

-- Define the price after the first reduction
def price_after_first_reduction : ℝ := original_price * (1 - first_reduction_percentage)

-- Define the final price after the second reduction
def final_price : ℝ := price_after_first_reduction * (1 - second_reduction_percentage)

-- Theorem to prove the final price is $3.50
theorem final_price_is_correct : final_price = 3.50 := by
  sorry

end final_price_is_correct_l175_175020


namespace total_marbles_l175_175561

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l175_175561


namespace minimum_cubes_required_l175_175240

def cube_snaps_visible (n : Nat) : Prop := 
  ∀ (cubes : Fin n → Fin 6 → Bool),
    (∀ i, (cubes i 0 ∧ cubes i 1) ∨ ¬(cubes i 0 ∨ cubes i 1)) → 
    ∃ i j, (i ≠ j) ∧ 
            (cubes i 0 ↔ ¬ cubes j 0) ∧ 
            (cubes i 1 ↔ ¬ cubes j 1)

theorem minimum_cubes_required : 
  ∃ n, cube_snaps_visible n ∧ n = 4 := 
  by sorry

end minimum_cubes_required_l175_175240


namespace college_girls_count_l175_175294

theorem college_girls_count (B G : ℕ) (h1 : B / G = 8 / 5) (h2 : B + G = 546) : G = 210 :=
by
  sorry

end college_girls_count_l175_175294


namespace race_permutations_l175_175583

-- Define the number of participants
def num_participants : ℕ := 4

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n + 1) * factorial n

-- Theorem: Given 4 participants, the number of different possible orders they can finish the race is 24.
theorem race_permutations : factorial num_participants = 24 := by
  -- sorry added to skip the proof
  sorry

end race_permutations_l175_175583


namespace value_of_expression_l175_175025

theorem value_of_expression (a b : ℤ) (h : a - b = 1) : 3 * a - 3 * b - 4 = -1 :=
by {
  sorry
}

end value_of_expression_l175_175025


namespace carlos_marbles_l175_175978

theorem carlos_marbles :
  ∃ N : ℕ, 
    (N % 9 = 2) ∧ 
    (N % 10 = 2) ∧ 
    (N % 11 = 2) ∧ 
    (N > 1) ∧ 
    N = 992 :=
by {
  -- We need this for the example; you would remove it in a real proof.
  sorry
}

end carlos_marbles_l175_175978


namespace sufficient_not_necessary_condition_l175_175892

theorem sufficient_not_necessary_condition (x k : ℝ) (p : x ≥ k) (q : (2 - x) / (x + 1) < 0) :
  (∀ x, x ≥ k → ((2 - x) / (x + 1) < 0)) ∧ (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → k > 2 := by
  sorry

end sufficient_not_necessary_condition_l175_175892


namespace problem1_problem2_l175_175707

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ 0) : 
  (a - b^2 / a) / ((a^2 + 2 * a * b + b^2) / a) = (a - b) / (a + b) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (6 - 2 * x ≥ 4) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (x ≤ 1) :=
by
  sorry

end problem1_problem2_l175_175707


namespace probability_perfect_square_l175_175811

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def successful_outcomes : Finset ℕ := {1, 4}

def total_possible_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_perfect_square :
  (successful_outcomes.card : ℚ) / (total_possible_outcomes.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_perfect_square_l175_175811


namespace tim_total_spent_l175_175110

variable (lunch_cost : ℝ)
variable (tip_percentage : ℝ)
variable (total_spent : ℝ)

theorem tim_total_spent (h_lunch_cost : lunch_cost = 60.80)
                        (h_tip_percentage : tip_percentage = 0.20)
                        (h_total_spent : total_spent = lunch_cost + (tip_percentage * lunch_cost)) :
                        total_spent = 72.96 :=
sorry

end tim_total_spent_l175_175110


namespace webinar_active_minutes_l175_175581

theorem webinar_active_minutes :
  let hours := 13
  let extra_minutes := 17
  let break_minutes := 22
  (hours * 60 + extra_minutes) - break_minutes = 775 := by
  sorry

end webinar_active_minutes_l175_175581


namespace min_value_expr_l175_175791

open Real

theorem min_value_expr(p q r : ℝ)(hp : 0 < p)(hq : 0 < q)(hr : 0 < r) :
  (5 * r / (3 * p + q) + 5 * p / (q + 3 * r) + 4 * q / (2 * p + 2 * r)) ≥ 5 / 2 :=
sorry

end min_value_expr_l175_175791


namespace real_no_impure_l175_175166

theorem real_no_impure {x : ℝ} (h1 : x^2 - 1 = 0) (h2 : x^2 + 3 * x + 2 ≠ 0) : x = 1 :=
by
  sorry

end real_no_impure_l175_175166


namespace solve_for_y_l175_175196

theorem solve_for_y (y : ℝ) (hy : y ≠ -2) : 
  (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ↔ y = 7 / 6 :=
by sorry

end solve_for_y_l175_175196


namespace product_simplification_l175_175105

theorem product_simplification :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) = 7 :=
by
  sorry

end product_simplification_l175_175105


namespace second_polygon_sides_l175_175942

theorem second_polygon_sides 
  (s : ℝ) -- side length of the second polygon
  (n1 n2 : ℕ) -- n1 = number of sides of the first polygon, n2 = number of sides of the second polygon
  (h1 : n1 = 40) -- first polygon has 40 sides
  (h2 : ∀ s1 s2 : ℝ, s1 = 3 * s2 → n1 * s1 = n2 * s2 → n2 = 120)
  : n2 = 120 := 
by
  sorry

end second_polygon_sides_l175_175942


namespace smallest_positive_period_of_f_max_min_values_of_f_in_interval_l175_175757

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

theorem max_min_values_of_f_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ f x ≥ -1 / 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_in_interval_l175_175757


namespace length_of_second_train_l175_175232

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (relative_speed : ℝ)
  (total_distance_covered : ℝ)
  (L : ℝ)
  (h1 : length_first_train = 210)
  (h2 : speed_first_train = 120 * 1000 / 3600)
  (h3 : speed_second_train = 80 * 1000 / 3600)
  (h4 : time_to_cross = 9)
  (h5 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600))
  (h6 : total_distance_covered = relative_speed * time_to_cross)
  (h7 : total_distance_covered = length_first_train + L) : 
  L = 289.95 :=
by {
  sorry
}

end length_of_second_train_l175_175232


namespace bridge_length_l175_175507

noncomputable def speed_km_per_hr_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def distance_travelled (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_condition : train_length = 150) 
  (train_speed_condition : train_speed_kmph = 45) 
  (crossing_time_condition : crossing_time_s = 30) :
  (distance_travelled (speed_km_per_hr_to_m_per_s train_speed_kmph) crossing_time_s - train_length) = 225 :=
by 
  sorry

end bridge_length_l175_175507


namespace hilary_total_kernels_l175_175633

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end hilary_total_kernels_l175_175633


namespace ball_distribution_l175_175164

-- Definitions as per conditions
def num_distinguishable_balls : ℕ := 5
def num_indistinguishable_boxes : ℕ := 3

-- Problem statement to prove
theorem ball_distribution : 
  let ways_to_distribute_balls := 1 + 5 + 10 + 10 + 30 in
  ways_to_distribute_balls = 56 :=
by
  -- proof required here
  sorry

end ball_distribution_l175_175164


namespace inequality_a3_b3_c3_l175_175061

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end inequality_a3_b3_c3_l175_175061


namespace toy_cars_in_third_box_l175_175314

theorem toy_cars_in_third_box (total_cars first_box second_box : ℕ) (H1 : total_cars = 71) 
    (H2 : first_box = 21) (H3 : second_box = 31) : total_cars - (first_box + second_box) = 19 :=
by
  sorry

end toy_cars_in_third_box_l175_175314


namespace debts_equal_in_25_days_l175_175258

-- Define the initial debts and the interest rates
def Darren_initial_debt : ℝ := 200
def Darren_interest_rate : ℝ := 0.08
def Fergie_initial_debt : ℝ := 300
def Fergie_interest_rate : ℝ := 0.04

-- Define the debts as a function of days passed t
def Darren_debt (t : ℝ) : ℝ := Darren_initial_debt * (1 + Darren_interest_rate * t)
def Fergie_debt (t : ℝ) : ℝ := Fergie_initial_debt * (1 + Fergie_interest_rate * t)

-- Prove that Darren and Fergie will owe the same amount in 25 days
theorem debts_equal_in_25_days : ∃ t, Darren_debt t = Fergie_debt t ∧ t = 25 := by
  sorry

end debts_equal_in_25_days_l175_175258


namespace sum_of_solutions_abs_eq_l175_175539

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l175_175539


namespace age_problem_solution_l175_175955

namespace AgeProblem

variables (S M : ℕ) (k : ℕ)

-- Condition: The present age of the son is 22
def son_age (S : ℕ) := S = 22

-- Condition: The man is 24 years older than his son
def man_age (M S : ℕ) := M = S + 24

-- Condition: In two years, man's age will be a certain multiple of son's age
def age_multiple (M S k : ℕ) := M + 2 = k * (S + 2)

-- Question: The ratio of man's age to son's age in two years
def age_ratio (M S : ℕ) := (M + 2) / (S + 2)

theorem age_problem_solution (S M : ℕ) (k : ℕ) 
  (h1 : son_age S)
  (h2 : man_age M S)
  (h3 : age_multiple M S k)
  : age_ratio M S = 2 :=
by
  rw [son_age, man_age, age_multiple, age_ratio] at *
  sorry

end AgeProblem

end age_problem_solution_l175_175955


namespace cary_initial_wage_l175_175132

noncomputable def initial_hourly_wage (x : ℝ) : Prop :=
  let first_year_wage := 1.20 * x
  let second_year_wage := 0.75 * first_year_wage
  second_year_wage = 9

theorem cary_initial_wage : ∃ x : ℝ, initial_hourly_wage x ∧ x = 10 := 
by
  use 10
  unfold initial_hourly_wage
  simp
  sorry

end cary_initial_wage_l175_175132


namespace warehouse_bins_total_l175_175849

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total_l175_175849


namespace find_x_square_l175_175742

theorem find_x_square (x : ℝ) (h_pos : x > 0) (h_condition : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end find_x_square_l175_175742


namespace determine_a_l175_175611

open Complex

noncomputable def complex_eq_real_im_part (a : ℝ) : Prop :=
  let z := (a - I) * (1 + I) / I
  (z.re, z.im) = ((a - 1 : ℝ), -(a + 1 : ℝ))

theorem determine_a (a : ℝ) (h : complex_eq_real_im_part a) : a = -1 :=
sorry

end determine_a_l175_175611


namespace total_photos_newspaper_l175_175300

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l175_175300


namespace find_a8_l175_175054

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def sum_of_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

theorem find_a8
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_terms a_n S)
  (h_S15 : S 15 = 45) :
  a_n 8 = 3 :=
sorry

end find_a8_l175_175054


namespace number_of_factors_l175_175440

theorem number_of_factors (K : ℕ) (hK : K = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, (∀ d e f g : ℕ, (0 ≤ d ∧ d ≤ 4) → (0 ≤ e ∧ e ≤ 3) → (0 ≤ f ∧ f ≤ 2) → (0 ≤ g ∧ g ≤ 1) → n = 120) :=
sorry

end number_of_factors_l175_175440


namespace solve_inequality_l175_175987

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l175_175987


namespace average_loss_per_loot_box_l175_175313

theorem average_loss_per_loot_box
  (cost_per_loot_box : ℝ := 5)
  (value_standard_item : ℝ := 3.5)
  (probability_rare_item_A : ℝ := 0.05)
  (value_rare_item_A : ℝ := 10)
  (probability_rare_item_B : ℝ := 0.03)
  (value_rare_item_B : ℝ := 15)
  (probability_rare_item_C : ℝ := 0.02)
  (value_rare_item_C : ℝ := 20) 
  : (cost_per_loot_box 
      - (0.90 * value_standard_item 
      + probability_rare_item_A * value_rare_item_A 
      + probability_rare_item_B * value_rare_item_B 
      + probability_rare_item_C * value_rare_item_C)) = 0.50 := by 
  sorry

end average_loss_per_loot_box_l175_175313


namespace number_of_monsters_l175_175331

theorem number_of_monsters
    (M S : ℕ)
    (h1 : 4 * M + 3 = S)
    (h2 : 5 * M = S - 6) :
  M = 9 :=
sorry

end number_of_monsters_l175_175331


namespace num_digits_difference_l175_175702

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end num_digits_difference_l175_175702


namespace negation_abs_lt_one_l175_175344

theorem negation_abs_lt_one (x : ℝ) : (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_abs_lt_one_l175_175344


namespace find_m_l175_175160

def U : Set ℤ := {-1, 2, 3, 6}
def A (m : ℤ) : Set ℤ := {x | x^2 - 5 * x + m = 0}
def complement_U_A (m : ℤ) : Set ℤ := U \ A m

theorem find_m (m : ℤ) (hU : U = {-1, 2, 3, 6}) (hcomp : complement_U_A m = {2, 3}) :
  m = -6 := by
  sorry

end find_m_l175_175160


namespace triangles_fit_in_pan_l175_175794

theorem triangles_fit_in_pan (pan_length pan_width triangle_base triangle_height : ℝ)
  (h1 : pan_length = 15) (h2 : pan_width = 24) (h3 : triangle_base = 3) (h4 : triangle_height = 4) :
  (pan_length * pan_width) / (1/2 * triangle_base * triangle_height) = 60 :=
by
  sorry

end triangles_fit_in_pan_l175_175794


namespace number_under_35_sampled_l175_175383

-- Define the conditions
def total_employees : ℕ := 500
def employees_under_35 : ℕ := 125
def employees_35_to_49 : ℕ := 280
def employees_over_50 : ℕ := 95
def sample_size : ℕ := 100

-- Define the theorem stating the desired result
theorem number_under_35_sampled : (employees_under_35 * sample_size / total_employees) = 25 :=
by
  sorry

end number_under_35_sampled_l175_175383


namespace find_possible_values_of_y_l175_175659

theorem find_possible_values_of_y (x : ℝ) (h : x^2 + 9 * (3 * x / (x - 3))^2 = 90) :
  y = (x - 3)^3 * (x + 2) / (2 * x - 4) → y = 28 / 3 ∨ y = 169 :=
by
  sorry

end find_possible_values_of_y_l175_175659


namespace yi_jianlian_shots_l175_175250

theorem yi_jianlian_shots (x y : ℕ) 
  (h1 : x + y = 16 - 3) 
  (h2 : 2 * x + y = 28 - 3 * 3) : 
  x = 6 ∧ y = 7 := 
by 
  sorry

end yi_jianlian_shots_l175_175250


namespace find_a2_plus_b2_l175_175271

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := 
by
  sorry

end find_a2_plus_b2_l175_175271


namespace multiplication_value_l175_175700

theorem multiplication_value (x : ℝ) (h : (2.25 / 3) * x = 9) : x = 12 :=
by
  sorry

end multiplication_value_l175_175700


namespace proportion_correct_l175_175042

theorem proportion_correct (x y : ℝ) (h : 3 * x = 2 * y) (hy : y ≠ 0) : x / 2 = y / 3 :=
by
  sorry

end proportion_correct_l175_175042


namespace first_term_of_geometric_series_l175_175400

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l175_175400


namespace find_k_l175_175631

def system_of_equations (x y k : ℝ) : Prop :=
  x - y = k - 3 ∧
  3 * x + 5 * y = 2 * k + 8 ∧
  x + y = 2

theorem find_k (x y k : ℝ) (h : system_of_equations x y k) : k = 1 := 
sorry

end find_k_l175_175631


namespace nate_search_time_l175_175798

def sectionG_rows : ℕ := 15
def sectionG_cars_per_row : ℕ := 10
def sectionH_rows : ℕ := 20
def sectionH_cars_per_row : ℕ := 9
def cars_per_minute : ℕ := 11

theorem nate_search_time :
  (sectionG_rows * sectionG_cars_per_row + sectionH_rows * sectionH_cars_per_row) / cars_per_minute = 30 :=
  by
    sorry

end nate_search_time_l175_175798


namespace num_people_price_item_equation_l175_175465

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l175_175465


namespace exists_2009_integers_with_gcd_condition_l175_175598

theorem exists_2009_integers_with_gcd_condition : 
  ∃ (S : Finset ℕ), S.card = 2009 ∧ (∀ x ∈ S, ∀ y ∈ S, x ≠ y → |x - y| = Nat.gcd x y) :=
sorry

end exists_2009_integers_with_gcd_condition_l175_175598


namespace right_triangle_area_l175_175624

theorem right_triangle_area (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) :
  (1/2) * (a : ℝ) * b = 30 :=
by
  sorry

end right_triangle_area_l175_175624


namespace find_m_l175_175838

theorem find_m (m : ℕ) (h : 8 ^ 36 * 6 ^ 21 = 3 * 24 ^ m) : m = 43 :=
sorry

end find_m_l175_175838


namespace balls_in_boxes_l175_175758

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l175_175758


namespace joe_dropped_score_l175_175473

theorem joe_dropped_score (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 60) (h2 : (A + B + C) / 3 = 65) :
  min A (min B (min C D)) = D → D = 45 :=
by sorry

end joe_dropped_score_l175_175473


namespace solve_for_x_l175_175699

theorem solve_for_x (x : ℚ) : 
  x + 5 / 6 = 11 / 18 - 2 / 9 → x = -4 / 9 := 
by
  intro h
  sorry

end solve_for_x_l175_175699


namespace solution_to_quadratic_inequality_l175_175988

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l175_175988


namespace intersection_is_singleton_zero_l175_175284

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

-- Define the theorem to be proved
theorem intersection_is_singleton_zero : M ∩ N = {0} :=
by
  -- Proof is provided by the steps above but not needed here
  sorry

end intersection_is_singleton_zero_l175_175284


namespace total_games_in_season_l175_175126

-- Definitions based on the conditions
def num_teams := 16
def teams_per_division := 8
def num_divisions := num_teams / teams_per_division

-- Each team plays every other team in its division twice
def games_within_division_per_team := (teams_per_division - 1) * 2

-- Each team plays every team in the other division once
def games_across_divisions_per_team := teams_per_division

-- Total games per team
def games_per_team := games_within_division_per_team + games_across_divisions_per_team

-- Total preliminary games for all teams (each game is counted twice)
def preliminary_total_games := games_per_team * num_teams

-- Since each game is counted twice, the final number of games
def total_games := preliminary_total_games / 2

theorem total_games_in_season : total_games = 176 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end total_games_in_season_l175_175126


namespace ball_distribution_l175_175763

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l175_175763


namespace find_selling_price_l175_175500

-- Define the parameters based on the problem conditions
constant cost_price : ℝ := 22
constant selling_price_original : ℝ := 38
constant sales_volume_original : ℝ := 160
constant price_reduction_step : ℝ := 3
constant sales_increase_step : ℝ := 120
constant daily_profit_target : ℝ := 3640

-- Define the function representing the sales volume as a function of price reduction
def sales_volume (x : ℝ) : ℝ :=
  sales_volume_original + (x / price_reduction_step) * sales_increase_step

-- Define the function representing the daily profit as a function of price reduction
def daily_profit (x : ℝ) : ℝ :=
  (selling_price_original - x - cost_price) * (sales_volume x)

-- State the main theorem: the new selling price ensuring the desired profit
theorem find_selling_price : ∃ x : ℝ, daily_profit x = daily_profit_target ∧ (selling_price_original - x = 29) :=
by
  sorry

end find_selling_price_l175_175500


namespace number_of_people_joining_group_l175_175852

theorem number_of_people_joining_group (x : ℕ) (h1 : 180 / 18 = 10) 
  (h2 : 180 / (18 + x) = 9) : x = 2 :=
by
  sorry

end number_of_people_joining_group_l175_175852


namespace arithmetic_mean_eq_2_l175_175860

theorem arithmetic_mean_eq_2 (a x : ℝ) (hx: x ≠ 0) :
  (1/2) * (((2 * x + a) / x) + ((2 * x - a) / x)) = 2 :=
by
  sorry

end arithmetic_mean_eq_2_l175_175860


namespace cone_volume_in_liters_l175_175543

theorem cone_volume_in_liters (d h : ℝ) (pi : ℝ) (liters_conversion : ℝ) :
  d = 12 → h = 10 → liters_conversion = 1000 → (1/3) * pi * (d/2)^2 * h * (1 / liters_conversion) = 0.12 * pi :=
by
  intros hd hh hc
  sorry

end cone_volume_in_liters_l175_175543


namespace prime_factorial_division_l175_175790

theorem prime_factorial_division (p k n : ℕ) (hp : Prime p) (h : p^k ∣ n!) : (p!)^k ∣ n! :=
sorry

end prime_factorial_division_l175_175790


namespace value_of_b_minus_a_l175_175640

open Real

def condition (a b : ℝ) : Prop := 
  abs a = 3 ∧ abs b = 2 ∧ a + b > 0

theorem value_of_b_minus_a (a b : ℝ) (h : condition a b) :
  b - a = -1 ∨ b - a = -5 :=
  sorry

end value_of_b_minus_a_l175_175640


namespace time_to_school_l175_175523

theorem time_to_school (total_distance walk_speed run_speed distance_ran : ℕ) (h_total : total_distance = 1800)
    (h_walk_speed : walk_speed = 70) (h_run_speed : run_speed = 210) (h_distance_ran : distance_ran = 600) :
    total_distance / walk_speed + distance_ran / run_speed = 20 := by
  sorry

end time_to_school_l175_175523


namespace find_first_offset_l175_175264

theorem find_first_offset (x : ℝ) : 
  let area := 180
  let diagonal := 24
  let offset2 := 6
  (area = (diagonal * (x + offset2)) / 2) -> x = 9 :=
sorry

end find_first_offset_l175_175264


namespace fraction_not_integer_l175_175194

theorem fraction_not_integer (a b : ℤ) : ¬ (∃ k : ℤ, (a^2 + b^2) = k * (a^2 - b^2)) :=
sorry

end fraction_not_integer_l175_175194


namespace total_weight_of_full_bucket_l175_175220

variable (a b x y : ℝ)

def bucket_weights :=
  (x + (1/3) * y = a) → (x + (3/4) * y = b) → (x + y = (16/5) * b - (11/5) * a)

theorem total_weight_of_full_bucket :
  bucket_weights a b x y :=
by
  intro h1 h2
  -- proof goes here, can be omitted as per instructions
  sorry

end total_weight_of_full_bucket_l175_175220


namespace interior_angles_of_n_plus_4_sided_polygon_l175_175679

theorem interior_angles_of_n_plus_4_sided_polygon (n : ℕ) (hn : 180 * (n - 2) = 1800) : 
  180 * (n + 4 - 2) = 2520 :=
by sorry

end interior_angles_of_n_plus_4_sided_polygon_l175_175679


namespace balls_in_boxes_l175_175759

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 56 ∧ (∃ f : ℕ → ℕ, (∀ n, f n = choose (n + 3) 3) ∧ f 5 = 56) :=
by
  sorry

end balls_in_boxes_l175_175759


namespace least_5_digit_divisible_l175_175866

theorem least_5_digit_divisible (n : ℕ) (h1 : n ≥ 10000) (h2 : n < 100000)
  (h3 : 15 ∣ n) (h4 : 12 ∣ n) (h5 : 18 ∣ n) : n = 10080 :=
by
  sorry

end least_5_digit_divisible_l175_175866


namespace find_number_l175_175230

theorem find_number (x : ℝ) (h : 0.50 * x = 0.30 * 50 + 13) : x = 56 :=
by
  sorry

end find_number_l175_175230


namespace statues_created_first_year_l175_175438

-- Definition of the initial conditions and the variable representing the number of statues created in the first year.
variables (S : ℕ)

-- Condition 1: In the second year, statues are quadrupled.
def second_year_statues : ℕ := 4 * S

-- Condition 2: In the third year, 12 statues are added, and 3 statues are broken.
def third_year_statues : ℕ := second_year_statues S + 12 - 3

-- Condition 3: In the fourth year, twice as many new statues are added as had been broken the previous year (2 * 3).
def fourth_year_added_statues : ℕ := 2 * 3
def fourth_year_statues : ℕ := third_year_statues S + fourth_year_added_statues

-- Condition 4: Total number of statues at the end of four years is 31.
def total_statues : ℕ := fourth_year_statues S

theorem statues_created_first_year : total_statues S = 31 → S = 4 :=
by {
  sorry
}

end statues_created_first_year_l175_175438


namespace soccer_team_percentage_l175_175957

theorem soccer_team_percentage (total_games won_games : ℕ) (h1 : total_games = 140) (h2 : won_games = 70) :
  (won_games / total_games : ℚ) * 100 = 50 := by
  sorry

end soccer_team_percentage_l175_175957


namespace field_length_l175_175817

theorem field_length 
  (w l : ℝ)
  (pond_area : ℝ := 25)
  (h1 : l = 2 * w)
  (h2 : pond_area = 25)
  (h3 : pond_area = (1 / 8) * (l * w)) :
  l = 20 :=
by
  sorry

end field_length_l175_175817


namespace solve_inequality_l175_175040

theorem solve_inequality (a x : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) :
  ((0 ≤ a ∧ a < 1/2 → a < x ∧ x < 1 - a) ∧ 
   (a = 1/2 → false) ∧ 
   (1/2 < a ∧ a ≤ 1 → 1 - a < x ∧ x < a)) ↔ (x - a) * (x + a - 1) < 0 := 
by
  sorry

end solve_inequality_l175_175040


namespace intersection_A_B_l175_175889

def A := {x : ℝ | 2 < x ∧ x < 4}
def B := {x : ℝ | (x-1) * (x-3) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end intersection_A_B_l175_175889


namespace sum_series_1_to_60_l175_175589

-- Define what it means to be the sum of the first n natural numbers
def sum_n (n : Nat) : Nat := n * (n + 1) / 2

theorem sum_series_1_to_60 : sum_n 60 = 1830 :=
by
  sorry

end sum_series_1_to_60_l175_175589


namespace euclid1976_partb_problem2_l175_175676

theorem euclid1976_partb_problem2
  (x y : ℝ)
  (geo_prog : y^2 = 2 * x)
  (arith_prog : 2 / y = 1 / x + 9 / x^2) :
  x * y = 27 / 2 := by 
  sorry

end euclid1976_partb_problem2_l175_175676


namespace find_c_l175_175737

open Real

theorem find_c (c : ℝ) (h : ∀ x, (x ∈ Set.Iio 2 ∨ x ∈ Set.Ioi 7) → -x^2 + c * x - 9 < -4) : 
  c = 9 :=
sorry

end find_c_l175_175737


namespace downstream_speed_l175_175233

noncomputable def V_b : ℝ := 7
noncomputable def V_up : ℝ := 4
noncomputable def V_s : ℝ := V_b - V_up

theorem downstream_speed :
  V_b + V_s = 10 := sorry

end downstream_speed_l175_175233


namespace central_angle_of_sector_l175_175335

theorem central_angle_of_sector {r l : ℝ} 
  (h1 : 2 * r + l = 4) 
  (h2 : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by 
  sorry

end central_angle_of_sector_l175_175335


namespace compute_exponent_multiplication_l175_175591

theorem compute_exponent_multiplication : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_exponent_multiplication_l175_175591


namespace parallel_vectors_xy_l175_175437

theorem parallel_vectors_xy {x y : ℝ} (h : ∃ k : ℝ, (1, y, -3) = (k * x, k * (-2), k * 5)) : x * y = -2 :=
by sorry

end parallel_vectors_xy_l175_175437


namespace intersection_with_x_axis_l175_175090

theorem intersection_with_x_axis (a : ℝ) (h : 2 * a - 4 = 0) : a = 2 := by
  sorry

end intersection_with_x_axis_l175_175090


namespace inequality_proof_l175_175470

theorem inequality_proof (a b c : ℝ) (hab : a * b < 0) : 
  a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := 
by 
  sorry

end inequality_proof_l175_175470


namespace number_of_yellow_marbles_l175_175954

theorem number_of_yellow_marbles (total_marbles blue_marbles red_marbles green_marbles yellow_marbles : ℕ)
    (h_total : total_marbles = 164) 
    (h_blue : blue_marbles = total_marbles / 2)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27) :
    yellow_marbles = total_marbles - (blue_marbles + red_marbles + green_marbles) →
    yellow_marbles = 14 := by
  sorry

end number_of_yellow_marbles_l175_175954


namespace simplify_expression_l175_175919

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (5 * x) * (x^4) = 27 * x^5 :=
by
  sorry

end simplify_expression_l175_175919


namespace expression_at_x_equals_2_l175_175041

theorem expression_at_x_equals_2 (a b : ℝ) (h : 2 * a - b = -1) : (2 * b - 4 * a) = 2 :=
by {
  sorry
}

end expression_at_x_equals_2_l175_175041


namespace complete_squares_l175_175895

def valid_solutions (x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = -2 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = 6) ∨
  (x = 0 ∧ y = -2 ∧ z = 6) ∨
  (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 4 ∧ y = -2 ∧ z = 0) ∨
  (x = 4 ∧ y = 0 ∧ z = 6) ∨
  (x = 4 ∧ y = -2 ∧ z = 6)

theorem complete_squares (x y z : ℝ) : 
  (x - 2)^2 + (y + 1)^2 = 5 →
  (x - 2)^2 + (z - 3)^2 = 13 →
  (y + 1)^2 + (z - 3)^2 = 10 →
  valid_solutions x y z :=
by
  intros h1 h2 h3
  sorry

end complete_squares_l175_175895


namespace pow_five_2010_mod_seven_l175_175217

theorem pow_five_2010_mod_seven :
  (5 ^ 2010) % 7 = 1 :=
by
  have h : (5 ^ 6) % 7 = 1 := sorry
  sorry

end pow_five_2010_mod_seven_l175_175217


namespace range_of_a_l175_175422

theorem range_of_a (a b : ℝ) (h1 : 0 ≤ a - b ∧ a - b ≤ 1) (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1 / 2 ≤ a ∧ a ≤ 5 / 2 := 
sorry

end range_of_a_l175_175422


namespace range_of_a_l175_175027

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : a ≤ -3 :=
by
  sorry

end range_of_a_l175_175027


namespace six_digit_palindromes_count_l175_175995

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l175_175995


namespace election_votes_l175_175353

theorem election_votes (T V : ℕ) 
    (hT : 8 * T = 11 * 20000) 
    (h_total_votes : T = 2500 + V + 20000) :
    V = 5000 :=
by
    sorry

end election_votes_l175_175353


namespace prime_has_two_square_numbers_l175_175113

noncomputable def isSquareNumber (p q : ℕ) : Prop :=
  p > q ∧ Nat.Prime p ∧ Nat.Prime q ∧ ¬ p^2 ∣ (q^(p-1) - 1)

theorem prime_has_two_square_numbers (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5) :
  ∃ q1 q2 : ℕ, isSquareNumber p q1 ∧ isSquareNumber p q2 ∧ q1 ≠ q2 :=
by 
  sorry

end prime_has_two_square_numbers_l175_175113


namespace suff_and_not_necessary_l175_175619

theorem suff_and_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) :
  (|a| > |b|) ∧ (¬(∀ x y : ℝ, (|x| > |y|) → (x > y ∧ y > 0))) :=
by
  sorry

end suff_and_not_necessary_l175_175619


namespace work_done_in_a_day_l175_175709

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l175_175709


namespace find_possible_values_l175_175182

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end find_possible_values_l175_175182


namespace last_two_digits_of_9_pow_2008_l175_175340

theorem last_two_digits_of_9_pow_2008 : (9 ^ 2008) % 100 = 21 := 
by
  sorry

end last_two_digits_of_9_pow_2008_l175_175340


namespace at_least_one_variety_has_27_apples_l175_175209

theorem at_least_one_variety_has_27_apples (total_apples : ℕ) (varieties : ℕ) 
  (h_total : total_apples = 105) (h_varieties : varieties = 4) : 
  ∃ v : ℕ, v ≥ 27 := 
sorry

end at_least_one_variety_has_27_apples_l175_175209


namespace fred_baseball_cards_l175_175608

variable (initial_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_baseball_cards (h1 : initial_cards = 5) (h2 : bought_cards = 3) : initial_cards - bought_cards = 2 := by
  sorry

end fred_baseball_cards_l175_175608


namespace cost_price_of_article_l175_175932

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 := 
by 
  sorry

end cost_price_of_article_l175_175932


namespace cos_b4_b6_l175_175282

theorem cos_b4_b6 (a b : ℕ → ℝ) (d : ℝ) 
  (ha_geom : ∀ n, a (n + 1) / a n = a 1)
  (hb_arith : ∀ n, b (n + 1) = b n + d)
  (ha_prod : a 1 * a 5 * a 9 = -8)
  (hb_sum : b 2 + b 5 + b 8 = 6 * Real.pi) : 
  Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1 / 2 :=
sorry

end cos_b4_b6_l175_175282


namespace additional_days_needed_is_15_l175_175049

-- Definitions and conditions from the problem statement
def good_days_2013 : ℕ := 365 * 479 / 100  -- Number of good air quality days in 2013
def target_increase : ℕ := 20              -- Target increase in percentage for 2014
def additional_days_first_half_2014 : ℕ := 20 -- Additional good air quality days in first half of 2014 compared to 2013
def half_good_days_2013 : ℕ := good_days_2013 / 2 -- Good air quality days in first half of 2013

-- Target number of good air quality days for 2014
def target_days_2014 : ℕ := good_days_2013 * (100 + target_increase) / 100

-- Good air quality days in the first half of 2014
def good_days_first_half_2014 : ℕ := half_good_days_2013 + additional_days_first_half_2014

-- Additional good air quality days needed in the second half of 2014
def additional_days_2014_second_half (target_days good_days_first_half_2014 : ℕ) : ℕ := 
  target_days - good_days_first_half_2014 - half_good_days_2013

-- Final theorem verifying the number of additional days needed in the second half of 2014 is 15
theorem additional_days_needed_is_15 : 
  additional_days_2014_second_half target_days_2014 good_days_first_half_2014 = 15 :=
sorry

end additional_days_needed_is_15_l175_175049


namespace find_number_of_students_l175_175087

-- Definitions for the conditions
def avg_age_students := 14
def teacher_age := 65
def new_avg_age := 15

-- The total age of students is n multiplied by their average age
def total_age_students (n : ℕ) := n * avg_age_students

-- The total age including teacher
def total_age_incl_teacher (n : ℕ) := total_age_students n + teacher_age

-- The new average age when teacher is included
def new_avg_age_incl_teacher (n : ℕ) := total_age_incl_teacher n / (n + 1)

theorem find_number_of_students (n : ℕ) (h₁ : avg_age_students = 14) (h₂ : teacher_age = 65) (h₃ : new_avg_age = 15) 
  (h_averages_eq : new_avg_age_incl_teacher n = new_avg_age) : n = 50 :=
  sorry

end find_number_of_students_l175_175087


namespace max_triangle_perimeter_l175_175576

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l175_175576


namespace acute_triangle_l175_175048

-- Given the lengths of three line segments
def length1 : ℝ := 5
def length2 : ℝ := 6
def length3 : ℝ := 7

-- Conditions (C): The lengths of the three line segments
def triangle_inequality : Prop :=
  length1 + length2 > length3 ∧
  length1 + length3 > length2 ∧
  length2 + length3 > length1

-- Question (Q) and Answer (A): They form an acute triangle
theorem acute_triangle (h : triangle_inequality) : (length1^2 + length2^2 - length3^2 > 0) :=
by
  sorry

end acute_triangle_l175_175048


namespace find_xyz_l175_175067

open Complex

theorem find_xyz (a b c x y z : ℂ)
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0)
  (ha : a = (b + c) / (x + 1))
  (hb : b = (a + c) / (y + 1))
  (hc : c = (a + b) / (z + 1))
  (hxy_z_1 : x * y + x * z + y * z = 9)
  (hxy_z_2 : x + y + z = 5) :
  x * y * z = 13 := 
sorry

end find_xyz_l175_175067


namespace disjunction_of_p_and_q_l175_175030

-- Define the propositions p and q
variable (p q : Prop)

-- Assume that p is true and q is false
theorem disjunction_of_p_and_q (h1 : p) (h2 : ¬q) : p ∨ q := 
sorry

end disjunction_of_p_and_q_l175_175030


namespace find_f_2_l175_175625

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem find_f_2 (a b : ℝ) (hf_neg2 : f a b (-2) = 7) : f a b 2 = -13 :=
by
  sorry

end find_f_2_l175_175625


namespace num_people_price_item_equation_l175_175463

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l175_175463


namespace solve_for_x_l175_175239

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h : (x / 100) * (x ^ 2) = 9) : x = 10 * (3 ^ (1 / 3)) :=
by
  sorry

end solve_for_x_l175_175239


namespace find_a_plus_b_l175_175476

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end find_a_plus_b_l175_175476


namespace find_initial_crayons_l175_175488

namespace CrayonProblem

variable (gave : ℕ) (lost : ℕ) (additional_lost : ℕ) 

def correct_answer (gave lost additional_lost : ℕ) :=
  gave + lost = gave + (gave + additional_lost) ∧ gave + lost = 502

theorem find_initial_crayons
  (gave := 90)
  (lost := 412)
  (additional_lost := 322)
  : correct_answer gave lost additional_lost :=
by 
  sorry

end CrayonProblem

end find_initial_crayons_l175_175488


namespace gcd_256_180_720_l175_175830

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l175_175830


namespace correct_equation_l175_175466

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l175_175466


namespace total_photos_l175_175298

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l175_175298


namespace base6_multiplication_l175_175731

-- Definitions of the base-six numbers
def base6_132 := [1, 3, 2] -- List representing 132_6
def base6_14 := [1, 4] -- List representing 14_6

-- Function to convert a base-6 list to a base-10 number
def base6_to_base10 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc x, acc * 6 + x) 0

-- The conversion of our specified numbers to base-10
def base10_132 := base6_to_base10 base6_132
def base10_14 := base6_to_base10 base6_14

-- The product of the conversions
def base10_product := base10_132 * base10_14

-- Function to convert a base-10 number to a base-6 list
def base10_to_base6 (n : ℕ) : List ℕ :=
  let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else loop (n / 6) ((n % 6) :: acc)
  loop n []

-- The conversion of the product back to base-6
def base6_product := base10_to_base6 base10_product

-- The expected base-6 product
def expected_base6_product := [1, 3, 3, 2]

-- The formal theorem statement
theorem base6_multiplication :
  base6_product = expected_base6_product := by
  sorry

end base6_multiplication_l175_175731


namespace total_pages_in_book_l175_175917

-- Define the conditions
def pagesDay1To5 : Nat := 5 * 25
def pagesDay6To9 : Nat := 4 * 40
def pagesLastDay : Nat := 30

-- Total calculation
def totalPages (p1 p2 pLast : Nat) : Nat := p1 + p2 + pLast

-- The proof problem statement
theorem total_pages_in_book :
  totalPages pagesDay1To5 pagesDay6To9 pagesLastDay = 315 :=
  by
    sorry

end total_pages_in_book_l175_175917


namespace steven_shirts_l175_175327

theorem steven_shirts : 
  (∀ (S A B : ℕ), S = 4 * A ∧ A = 6 * B ∧ B = 3 → S = 72) := 
by
  intro S A B
  intro h
  cases h with h1 h2
  cases h2 with hA hB
  rw [hB, hA]
  sorry

end steven_shirts_l175_175327


namespace distinct_license_plates_l175_175386

noncomputable def license_plates : ℕ :=
  let digits_possibilities := 10^5
  let letters_possibilities := 26^3
  let positions := 6
  positions * digits_possibilities * letters_possibilities

theorem distinct_license_plates : 
  license_plates = 105456000 := by
  sorry

end distinct_license_plates_l175_175386


namespace problem_statement_l175_175026

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f a b α β 2007 = 5) :
  f a b α β 2008 = 3 := 
by
  sorry

end problem_statement_l175_175026


namespace smallest_term_of_bn_div_an_is_four_l175_175277

theorem smallest_term_of_bn_div_an_is_four
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = 2 * S n)
  (h3 : b 1 = 16)
  (h4 : ∀ n, b (n + 1) - b n = 2 * n) :
  ∃ n : ℕ, ∀ m : ℕ, (m ≠ 4 → b m / a m > b 4 / a 4) ∧ (n = 4) := sorry

end smallest_term_of_bn_div_an_is_four_l175_175277


namespace no_common_point_in_all_circles_l175_175648

variable {Point : Type}
variable {Circle : Type}
variable (center : Circle → Point)
variable (contains : Circle → Point → Prop)

-- Given six circles in the plane
variables (C1 C2 C3 C4 C5 C6 : Circle)

-- Condition: None of the circles contain the center of any other circle
axiom condition_1 : ∀ (C D : Circle), C ≠ D → ¬ contains C (center D)

-- Question: Prove that there does not exist a point P that lies in all six circles
theorem no_common_point_in_all_circles : 
  ¬ ∃ (P : Point), (contains C1 P) ∧ (contains C2 P) ∧ (contains C3 P) ∧ (contains C4 P) ∧ (contains C5 P) ∧ (contains C6 P) :=
sorry

end no_common_point_in_all_circles_l175_175648


namespace problemI_solution_set_problemII_range_of_a_l175_175431

section ProblemI

def f (x : ℝ) := |2 * x - 2| + 2

theorem problemI_solution_set :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end ProblemI

section ProblemII

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

theorem problemII_range_of_a :
  {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 3} = {a : ℝ | 2 ≤ a} :=
by
  sorry

end ProblemII

end problemI_solution_set_problemII_range_of_a_l175_175431


namespace sqrt_expression_l175_175251

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l175_175251


namespace daughter_age_l175_175940

-- Define the conditions and the question as a theorem
theorem daughter_age (D F : ℕ) (h1 : F = 3 * D) (h2 : F + 12 = 2 * (D + 12)) : D = 12 :=
by
  -- We need to provide a proof or placeholder for now
  sorry

end daughter_age_l175_175940


namespace circle_and_tangent_lines_l175_175274

-- Define the problem conditions
def passes_through (a b r : ℝ) : Prop :=
  (a - (-2))^2 + (b - 2)^2 = r^2 ∧
  (a - (-5))^2 + (b - 5)^2 = r^2

def lies_on_line (a b : ℝ) : Prop :=
  a + b + 3 = 0

-- Define the standard equation of the circle
def is_circle_eq (a b r : ℝ) : Prop := ∀ x y : ℝ, 
  (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 5)^2 + (y - 2)^2 = 9

-- Define the tangent lines
def is_tangent_lines (x y k : ℝ) : Prop :=
  (k = (20 / 21) ∨ x = -2) → (20 * x - 21 * y + 229 = 0 ∨ x = -2)

-- The theorem statement in Lean 4
theorem circle_and_tangent_lines (a b r : ℝ) (x y k : ℝ) :
  passes_through a b r →
  lies_on_line a b →
  is_circle_eq a b r →
  is_tangent_lines x y k :=
by {
  sorry
}

end circle_and_tangent_lines_l175_175274


namespace smallest_sum_of_squares_l175_175338

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 91) : x^2 + y^2 ≥ 109 :=
sorry

end smallest_sum_of_squares_l175_175338


namespace solve_x_division_l175_175542

theorem solve_x_division :
  ∀ x : ℝ, (3 / x + 4 / x / (8 / x) = 1.5) → x = 3 := 
by
  intro x
  intro h
  sorry

end solve_x_division_l175_175542


namespace MsElizabethInvestmentsCount_l175_175188

variable (MrBanksRevPerInvestment : ℕ) (MsElizabethRevPerInvestment : ℕ) (MrBanksInvestments : ℕ) (MsElizabethExtraRev : ℕ)

def MrBanksTotalRevenue := MrBanksRevPerInvestment * MrBanksInvestments
def MsElizabethTotalRevenue := MrBanksTotalRevenue + MsElizabethExtraRev
def MsElizabethInvestments := MsElizabethTotalRevenue / MsElizabethRevPerInvestment

theorem MsElizabethInvestmentsCount (h1 : MrBanksRevPerInvestment = 500) 
  (h2 : MsElizabethRevPerInvestment = 900)
  (h3 : MrBanksInvestments = 8)
  (h4 : MsElizabethExtraRev = 500) : 
  MsElizabethInvestments MrBanksRevPerInvestment MsElizabethRevPerInvestment MrBanksInvestments MsElizabethExtraRev = 5 :=
by
  sorry

end MsElizabethInvestmentsCount_l175_175188


namespace total_votes_l175_175245

theorem total_votes (votes_veggies : ℕ) (votes_meat : ℕ) (H1 : votes_veggies = 337) (H2 : votes_meat = 335) : votes_veggies + votes_meat = 672 :=
by
  sorry

end total_votes_l175_175245


namespace no_real_roots_iff_l175_175047

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem no_real_roots_iff (a : ℝ) : (∀ x : ℝ, f x a ≠ 0) → a > 1 :=
  by
    sorry

end no_real_roots_iff_l175_175047


namespace rectangle_in_triangle_area_l175_175672

theorem rectangle_in_triangle_area
  (PR : ℝ) (h_PR : PR = 15)
  (Q_altitude : ℝ) (h_Q_altitude : Q_altitude = 9)
  (x : ℝ)
  (AD : ℝ) (h_AD : AD = x)
  (AB : ℝ) (h_AB : AB = x / 3) :
  (AB * AD = 675 / 64) :=
by
  sorry

end rectangle_in_triangle_area_l175_175672


namespace sqrt_diff_eq_neg_four_sqrt_five_l175_175007

theorem sqrt_diff_eq_neg_four_sqrt_five : 
  (Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5)) = -4 * Real.sqrt 5 := 
sorry

end sqrt_diff_eq_neg_four_sqrt_five_l175_175007


namespace red_yellow_flowers_l175_175584

theorem red_yellow_flowers
  (total : ℕ)
  (yellow_white : ℕ)
  (red_white : ℕ)
  (extra_red_over_white : ℕ)
  (H1 : total = 44)
  (H2 : yellow_white = 13)
  (H3 : red_white = 14)
  (H4 : extra_red_over_white = 4) :
  ∃ (red_yellow : ℕ), red_yellow = 17 := by
  sorry

end red_yellow_flowers_l175_175584


namespace BothNormal_l175_175726

variable (Normal : Type) (Person : Type) (MrA MrsA : Person)
variables (isNormal : Person → Prop)

-- Conditions given in the problem
axiom MrA_statement : ∀ p : Person, p = MrsA → isNormal MrA → isNormal MrsA
axiom MrsA_statement : ∀ p : Person, p = MrA → isNormal MrsA → isNormal MrA

-- Question (translated to proof problem): 
-- prove that Mr. A and Mrs. A are both normal persons
theorem BothNormal : isNormal MrA ∧ isNormal MrsA := 
  by 
    sorry -- proof is omitted

end BothNormal_l175_175726


namespace quadratic_root_ratio_l175_175019

theorem quadratic_root_ratio (k : ℝ) (h : ∃ r : ℝ, r ≠ 0 ∧ 3 * r * r = k * r - 12 * r + k ∧ r * r = k + 9 * r - k) : k = 27 :=
sorry

end quadratic_root_ratio_l175_175019


namespace volume_of_rectangular_solid_l175_175099

theorem volume_of_rectangular_solid
  (a b c : ℝ)
  (h1 : a * b = 3)
  (h2 : a * c = 5)
  (h3 : b * c = 15) :
  a * b * c = 15 :=
sorry

end volume_of_rectangular_solid_l175_175099


namespace walnut_trees_l175_175653

theorem walnut_trees (logs_per_pine logs_per_maple logs_per_walnut pine_trees maple_trees total_logs walnut_trees : ℕ)
  (h1 : logs_per_pine = 80)
  (h2 : logs_per_maple = 60)
  (h3 : logs_per_walnut = 100)
  (h4 : pine_trees = 8)
  (h5 : maple_trees = 3)
  (h6 : total_logs = 1220)
  (h7 : total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut) :
  walnut_trees = 4 :=
by
  sorry

end walnut_trees_l175_175653


namespace remainder_a6_mod_n_eq_1_l175_175479

theorem remainder_a6_mod_n_eq_1 
  (n : ℕ) (a : ℤ) (h₁ : n > 0) (h₂ : a^3 ≡ 1 [MOD n]) : a^6 ≡ 1 [MOD n] := 
by 
  sorry

end remainder_a6_mod_n_eq_1_l175_175479


namespace ratio_of_savings_to_earnings_l175_175913

-- Definitions based on the given conditions
def earnings_washing_cars : ℤ := 20
def earnings_walking_dogs : ℤ := 40
def total_savings : ℤ := 150
def months : ℤ := 5

-- Statement to prove the ratio of savings per month to total earnings per month
theorem ratio_of_savings_to_earnings :
  (total_savings / months) = (earnings_washing_cars + earnings_walking_dogs) / 2 := by
  sorry

end ratio_of_savings_to_earnings_l175_175913


namespace total_amount_paid_correct_l175_175189

-- Define variables for prices of the pizzas
def first_pizza_price : ℝ := 8
def second_pizza_price : ℝ := 12
def third_pizza_price : ℝ := 10

-- Define variables for discount rate and tax rate
def discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.05

-- Define the total amount paid by Mrs. Hilt
def total_amount_paid : ℝ :=
  let total_cost := first_pizza_price + second_pizza_price + third_pizza_price
  let discount := total_cost * discount_rate
  let discounted_total := total_cost - discount
  let sales_tax := discounted_total * sales_tax_rate
  discounted_total + sales_tax

-- Prove that the total amount paid is $25.20
theorem total_amount_paid_correct : total_amount_paid = 25.20 := 
  by
  sorry

end total_amount_paid_correct_l175_175189


namespace principal_amount_borrowed_l175_175547

theorem principal_amount_borrowed (P R T SI : ℕ) (h₀ : SI = (P * R * T) / 100) (h₁ : SI = 5400) (h₂ : R = 12) (h₃ : T = 3) : P = 15000 :=
by
  sorry

end principal_amount_borrowed_l175_175547


namespace total_concrete_weight_l175_175848

theorem total_concrete_weight (w1 w2 : ℝ) (c1 c2 : ℝ) (total_weight : ℝ)
  (h1 : w1 = 1125)
  (h2 : w2 = 1125)
  (h3 : c1 = 0.093)
  (h4 : c2 = 0.113)
  (h5 : (w1 * c1 + w2 * c2) / (w1 + w2) = 0.108) :
  total_weight = w1 + w2 :=
by
  sorry

end total_concrete_weight_l175_175848


namespace sequence_explicit_formula_l175_175649

-- Define the sequence
def a : ℕ → ℤ
| 0     := 2
| (n+1) := a n - n - 1 + 3

-- Define the function to prove
def explicit_formula (n : ℕ) : ℤ := -(n * (n + 1)) / 2 + 3 * n + 2

-- The proof problem statement
theorem sequence_explicit_formula (n : ℕ) : a n = explicit_formula n :=
sorry

end sequence_explicit_formula_l175_175649


namespace cleaning_time_with_doubled_an_speed_l175_175967

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l175_175967


namespace find_y_l175_175082

variable (t : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := x = 3 - t
def condition2 : Prop := y = 2 * t + 11
def condition3 : Prop := x = 1

theorem find_y (h1 : condition1 x t) (h2 : condition2 t y) (h3 : condition3 x) : y = 15 := by
  sorry

end find_y_l175_175082


namespace total_marbles_l175_175563

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l175_175563


namespace john_climbs_9_flights_l175_175784

variable (fl : Real := 10)  -- Each flight of stairs is 10 feet
variable (step_height_inches : Real := 18)  -- Each step is 18 inches
variable (steps : Nat := 60)  -- John climbs 60 steps

theorem john_climbs_9_flights :
  (steps * (step_height_inches / 12) / fl = 9) :=
by
  sorry

end john_climbs_9_flights_l175_175784


namespace smaller_two_digit_product_l175_175512

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l175_175512


namespace length_of_platform_l175_175392

-- Given conditions
def train_length : ℝ := 100
def time_pole : ℝ := 15
def time_platform : ℝ := 40

-- Theorem to prove the length of the platform
theorem length_of_platform (L : ℝ) 
    (h_train_length : train_length = 100)
    (h_time_pole : time_pole = 15)
    (h_time_platform : time_platform = 40)
    (h_speed : (train_length / time_pole) = (100 + L) / time_platform) : 
    L = 500 / 3 :=
by
  sorry

end length_of_platform_l175_175392


namespace number_with_1_before_and_after_l175_175046

theorem number_with_1_before_and_after (n : ℕ) (hn : n < 10) : 100 * 1 + 10 * n + 1 = 101 + 10 * n := by
    sorry

end number_with_1_before_and_after_l175_175046


namespace shooter_probability_l175_175575

theorem shooter_probability (hit_prob : ℝ) (n : ℕ) (k : ℕ) (hit_prob_condition : hit_prob = 0.8) (n_condition : n = 5) (k_condition : k = 2) :
  (probability (at_least_k_hits n hit_prob k) = 0.9929) :=
sorry

end shooter_probability_l175_175575


namespace num_ways_distribute_balls_l175_175770

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l175_175770


namespace maximize_NPM_l175_175106

theorem maximize_NPM :
  ∃ (M N P : ℕ), 
    (∀ M, M < 10 → (11 * M * M) = N * 100 + P * 10 + M) →
    N * 100 + P * 10 + M = 396 :=
by
  sorry

end maximize_NPM_l175_175106


namespace salary_net_change_l175_175706

variable {S : ℝ}

theorem salary_net_change (S : ℝ) : (1.4 * S - 0.4 * (1.4 * S)) - S = -0.16 * S :=
by
  sorry

end salary_net_change_l175_175706


namespace distinct_strings_after_operations_l175_175072

def valid_strings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else valid_strings (n-1) + valid_strings (n-2)

theorem distinct_strings_after_operations :
  valid_strings 10 = 144 := by
  sorry

end distinct_strings_after_operations_l175_175072


namespace tennis_ball_ratio_problem_solution_l175_175390

def tennis_ball_ratio_problem (total_balls ordered_white ordered_yellow dispatched_yellow extra_yellow : ℕ) : Prop :=
  total_balls = 114 ∧ 
  ordered_white = total_balls / 2 ∧ 
  ordered_yellow = total_balls / 2 ∧ 
  dispatched_yellow = ordered_yellow + extra_yellow → 
  (ordered_white / dispatched_yellow = 57 / 107)

theorem tennis_ball_ratio_problem_solution :
  tennis_ball_ratio_problem 114 57 57 107 50 := by 
  sorry

end tennis_ball_ratio_problem_solution_l175_175390


namespace complement_union_M_N_eq_set_l175_175035

open Set

-- Define the universe U
def U : Set (ℝ × ℝ) := { p | True }

-- Define the set M
def M : Set (ℝ × ℝ) := { p | (p.snd - 3) / (p.fst - 2) ≠ 1 }

-- Define the set N
def N : Set (ℝ × ℝ) := { p | p.snd ≠ p.fst + 1 }

-- Define the complement of M ∪ N in U
def complement_MN : Set (ℝ × ℝ) := compl (M ∪ N)

theorem complement_union_M_N_eq_set : complement_MN = { (2, 3) } :=
  sorry

end complement_union_M_N_eq_set_l175_175035


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l175_175667

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l175_175667


namespace balls_in_boxes_l175_175766

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l175_175766


namespace measure_of_angle_B_l175_175342

noncomputable def angle_opposite_side (a b c : ℝ) (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : ℝ :=
  if h : (c^2)/(a+b) + (a^2)/(b+c) = b then 60 else 0

theorem measure_of_angle_B {a b c : ℝ} (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : 
  angle_opposite_side a b c h = 60 :=
by
  sorry

end measure_of_angle_B_l175_175342


namespace problem_statement_l175_175477

open Complex

theorem problem_statement (a b : ℝ) (h : (1 + (I : ℂ) * (sqrt 3))^3 + a * (1 + (I * sqrt 3)) + b = 0) :
  a + b = 8 := 
sorry

end problem_statement_l175_175477


namespace cleaning_time_together_l175_175375

theorem cleaning_time_together (t : ℝ) (h_t : 3 = t / 3) (h_john_time : 6 = 6) : 
  (5 / (1 / 6 + 1 / 9)) = 3.6 :=
by
  sorry

end cleaning_time_together_l175_175375


namespace solve_fractional_equation_l175_175923

theorem solve_fractional_equation (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ -6) :
    (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9 / 4 :=
by
  sorry

end solve_fractional_equation_l175_175923


namespace estimated_probability_mouth_upwards_l175_175170

theorem estimated_probability_mouth_upwards :
  let num_tosses := 200
  let occurrences_upwards := 48
  let estimated_probability := (occurrences_upwards : ℝ) / (num_tosses : ℝ)
  estimated_probability = 0.24 := 
by {
  sorry,
}

end estimated_probability_mouth_upwards_l175_175170


namespace num_points_on_ellipse_with_area_l175_175094

-- Define the line equation
def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Define the area condition for the triangle
def area_condition (xA yA xB yB xP yP : ℝ) : Prop :=
  abs (xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB)) = 6

-- Define the main theorem statement
theorem num_points_on_ellipse_with_area (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∃ P1 P2 : ℝ × ℝ, 
    (ellipse_eq P1.1 P1.2) ∧ 
    (ellipse_eq P2.1 P2.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P1.1 P1.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P2.1 P2.2) ∧ 
    P1 ≠ P2 := sorry

end num_points_on_ellipse_with_area_l175_175094


namespace biology_to_general_ratio_l175_175521

variable (g b m : ℚ)

theorem biology_to_general_ratio (h1 : g = 30) 
                                (h2 : m = (3/5) * (g + b)) 
                                (h3 : g + b + m = 144) : 
                                b / g = 2 / 1 := 
by 
  sorry

end biology_to_general_ratio_l175_175521


namespace arrangements_TOOTH_l175_175137
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end arrangements_TOOTH_l175_175137


namespace ryan_time_learning_l175_175864

variable (t : ℕ) (c : ℕ)

/-- Ryan spends a total of 3 hours on both languages every day. Assume further that he spends 1 hour on learning Chinese every day, and you need to find how many hours he spends on learning English. --/
theorem ryan_time_learning (h_total : t = 3) (h_chinese : c = 1) : (t - c) = 2 := 
by
  -- Proof goes here
  sorry

end ryan_time_learning_l175_175864


namespace indoor_table_chairs_l175_175952

theorem indoor_table_chairs (x : ℕ) :
  (9 * x) + (11 * 3) = 123 → x = 10 :=
by
  intro h
  sorry

end indoor_table_chairs_l175_175952


namespace gcf_of_all_three_digit_palindromes_l175_175363

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l175_175363


namespace minimum_meals_needed_l175_175475

theorem minimum_meals_needed (total_jam : ℝ) (max_per_meal : ℝ) (jars : ℕ) (max_jar_weight : ℝ):
  (total_jam = 50) → (max_per_meal = 5) → (jars ≥ 50) → (max_jar_weight ≤ 1) →
  (jars * max_jar_weight = total_jam) →
  jars ≥ 12 := sorry

end minimum_meals_needed_l175_175475


namespace total_marbles_l175_175562

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l175_175562


namespace no_solutions_to_cubic_sum_l175_175143

theorem no_solutions_to_cubic_sum (x y z : ℤ) : 
    ¬ (x^3 + y^3 = z^3 + 4) :=
by 
  sorry

end no_solutions_to_cubic_sum_l175_175143


namespace solve_equation_l175_175741

theorem solve_equation :
  ∀ (x m n : ℕ), 
    0 < x → 0 < m → 0 < n → 
    x^m = 2^(2 * n + 1) + 2^n + 1 →
    (x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4) :=
by
  sorry

end solve_equation_l175_175741


namespace pills_needed_for_week_l175_175918

def pill_mg : ℕ := 50 -- Each pill has 50 mg of Vitamin A.
def recommended_daily_mg : ℕ := 200 -- The recommended daily serving of Vitamin A is 200 mg.
def days_in_week : ℕ := 7 -- There are 7 days in a week.

theorem pills_needed_for_week : (recommended_daily_mg / pill_mg) * days_in_week = 28 := 
by 
  sorry

end pills_needed_for_week_l175_175918


namespace wall_paint_area_l175_175244

theorem wall_paint_area
  (A₁ : ℕ) (A₂ : ℕ) (A₃ : ℕ) (A₄ : ℕ)
  (H₁ : A₁ = 32)
  (H₂ : A₂ = 48)
  (H₃ : A₃ = 32)
  (H₄ : A₄ = 48) :
  A₁ + A₂ + A₃ + A₄ = 160 :=
by
  sorry

end wall_paint_area_l175_175244


namespace find_parallelogram_height_l175_175739

def parallelogram_height (base area : ℕ) : ℕ := area / base

theorem find_parallelogram_height :
  parallelogram_height 32 448 = 14 :=
by {
  sorry
}

end find_parallelogram_height_l175_175739


namespace total_photos_newspaper_l175_175305

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l175_175305


namespace range_of_a_l175_175881

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a * x > 0) → a < 1 :=
by
  sorry

end range_of_a_l175_175881


namespace number_of_even_factors_of_n_l175_175065

def n : ℕ := 2^4 * 3^3 * 5 * 7^2

theorem number_of_even_factors_of_n : 
  (∃ k : ℕ, n = 2^4 * 3^3 * 5 * 7^2 ∧ k = 96) → 
  ∃ count : ℕ, 
    count = 96 ∧ 
    (∀ m : ℕ, 
      (m ∣ n ∧ m % 2 = 0) ↔ 
      (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d)) :=
by
  sorry

end number_of_even_factors_of_n_l175_175065


namespace bus_average_speed_excluding_stoppages_l175_175009

theorem bus_average_speed_excluding_stoppages :
  ∀ v : ℝ, (32 / 60) * v = 40 → v = 75 :=
by
  intro v
  intro h
  sorry

end bus_average_speed_excluding_stoppages_l175_175009


namespace stratified_sampling_school_C_l175_175050

theorem stratified_sampling_school_C 
  (teachers_A : ℕ) 
  (teachers_B : ℕ) 
  (teachers_C : ℕ) 
  (total_teachers : ℕ)
  (total_drawn : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 140)
  (hC : teachers_C = 160)
  (hTotal : total_teachers = teachers_A + teachers_B + teachers_C)
  (hDraw : total_drawn = 60) :
  (total_drawn * teachers_C / total_teachers) = 20 := 
by
  sorry

end stratified_sampling_school_C_l175_175050


namespace solution_exists_unique_l175_175910

variable (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)

theorem solution_exists_unique (x y z : ℝ)
  (hx : x = (b + c) / 2)
  (hy : y = (c + a) / 2)
  (hz : z = (a + b) / 2)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by
  sorry

end solution_exists_unique_l175_175910


namespace problem_proof_l175_175755

open Set

noncomputable def A : Set ℝ := {x | abs (4 * x - 1) < 9}
noncomputable def B : Set ℝ := {x | x / (x + 3) ≥ 0}
noncomputable def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 5 / 2}
noncomputable def correct_answer : Set ℝ := Iio (-3) ∪ Ici (5 / 2)

theorem problem_proof : (compl A) ∩ B = correct_answer := 
  by
    sorry

end problem_proof_l175_175755


namespace complete_residue_system_infinitely_many_positive_integers_l175_175715

def is_complete_residue_system (n m : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → (i^n % m ≠ j^n % m)

theorem complete_residue_system_infinitely_many_positive_integers (m : ℕ) (h_pos : 0 < m) :
  ∃ᶠ n in at_top, is_complete_residue_system n m :=
sorry

end complete_residue_system_infinitely_many_positive_integers_l175_175715


namespace number_of_squares_l175_175490

def side_plywood : ℕ := 50
def side_square_1 : ℕ := 10
def side_square_2 : ℕ := 20
def total_cut_length : ℕ := 280

/-- Number of squares obtained given the side lengths of the plywood and the cut lengths -/
theorem number_of_squares (x y : ℕ) (h1 : 100 * x + 400 * y = side_plywood^2)
  (h2 : 40 * x + 80 * y = total_cut_length) : x + y = 16 :=
sorry

end number_of_squares_l175_175490


namespace relationship_between_vars_l175_175443

-- Define the variables a, b, c, d as real numbers
variables (a b c d : ℝ)

-- Define the initial condition
def initial_condition := (a + 2 * b) / (2 * b + c) = (c + 2 * d) / (2 * d + a)

-- State the theorem to be proved
theorem relationship_between_vars (h : initial_condition a b c d) : 
  a = c ∨ a + c + 2 * (b + d) = 0 :=
sorry

end relationship_between_vars_l175_175443


namespace solve_xy_l175_175738

theorem solve_xy (x y : ℝ) :
  (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1 / 3 → 
  x = 34 / 3 ∧ y = 35 / 3 :=
by
  intro h
  sorry

end solve_xy_l175_175738


namespace transform_negation_l175_175723

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l175_175723


namespace inverse_function_value_l175_175774

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ y : ℝ, f (3^y) = y) : f 3 = 1 :=
sorry

end inverse_function_value_l175_175774


namespace max_trading_cards_l175_175652

variable (money : ℝ) (cost_per_card : ℝ) (max_cards : ℕ)

theorem max_trading_cards (h_money : money = 9) (h_cost : cost_per_card = 1) : max_cards ≤ 9 :=
sorry

end max_trading_cards_l175_175652


namespace balls_in_boxes_l175_175767

-- Here we declare the variables for the balls and boxes
variables (balls : Nat) (boxes : Nat)

-- Define the appropriate conditions
def problem_conditions := balls = 5 ∧ boxes = 4

-- Define the desired result
def desired_result := 56

-- The statement of the problem in Lean 4
theorem balls_in_boxes : problem_conditions balls boxes → (number_of_ways balls boxes = desired_result) :=
by
  sorry

end balls_in_boxes_l175_175767


namespace annual_population_increase_l175_175096

theorem annual_population_increase (x : ℝ) (initial_pop : ℝ) :
    (initial_pop * (1 + (x - 1) / 100)^3 = initial_pop * 1.124864) → x = 5.04 :=
by
  -- Provided conditions
  intros h
  -- The hypothesis conditionally establishes that this will derive to show x = 5.04
  sorry

end annual_population_increase_l175_175096


namespace ball_distribution_l175_175762

theorem ball_distribution (n_balls : ℕ) (n_boxes : ℕ) (h_balls : n_balls = 5) (h_boxes : n_boxes = 4) : 
  (∃ ways : ℕ, ways = 68) :=
by
  use 68
  exact sorry

end ball_distribution_l175_175762


namespace cube_root_of_64_is_4_l175_175642

theorem cube_root_of_64_is_4 (x : ℝ) (h1 : 0 < x) (h2 : x^3 = 64) : x = 4 :=
by
  sorry

end cube_root_of_64_is_4_l175_175642


namespace number_of_poison_frogs_l175_175179

theorem number_of_poison_frogs
  (total_frogs : ℕ) (tree_frogs : ℕ) (wood_frogs : ℕ) (poison_frogs : ℕ)
  (h₁ : total_frogs = 78)
  (h₂ : tree_frogs = 55)
  (h₃ : wood_frogs = 13)
  (h₄ : total_frogs = tree_frogs + wood_frogs + poison_frogs) :
  poison_frogs = 10 :=
by sorry

end number_of_poison_frogs_l175_175179


namespace matches_between_withdrawn_players_l175_175778

theorem matches_between_withdrawn_players (n r : ℕ) (h : 50 = (n - 3).choose 2 + (6 - r) + r) : r = 1 :=
sorry

end matches_between_withdrawn_players_l175_175778


namespace nested_fraction_value_l175_175590

theorem nested_fraction_value : 
  let expr := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))
  expr = 21 / 55 :=
by 
  sorry

end nested_fraction_value_l175_175590


namespace total_photos_newspaper_l175_175302

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l175_175302


namespace max_students_equal_division_l175_175950

theorem max_students_equal_division (pens pencils : ℕ) (h_pens : pens = 640) (h_pencils : pencils = 520) : 
  Nat.gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  have : Nat.gcd 640 520 = 40 := by norm_num
  exact this

end max_students_equal_division_l175_175950


namespace part_I_part_II_l175_175432

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end part_I_part_II_l175_175432


namespace hunter_ants_l175_175809

variable (spiders : ℕ) (ladybugs_before : ℕ) (ladybugs_flew : ℕ) (total_insects : ℕ)

theorem hunter_ants (h1 : spiders = 3)
                    (h2 : ladybugs_before = 8)
                    (h3 : ladybugs_flew = 2)
                    (h4 : total_insects = 21) :
  ∃ ants : ℕ, ants = total_insects - (spiders + (ladybugs_before - ladybugs_flew)) ∧ ants = 12 :=
by
  sorry

end hunter_ants_l175_175809


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l175_175665

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l175_175665


namespace actual_total_area_in_acres_l175_175953

-- Define the conditions
def base_cm : ℝ := 20
def height_cm : ℝ := 12
def rect_length_cm : ℝ := 20
def rect_width_cm : ℝ := 5
def scale_cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

-- Define the total area in acres calculation
def total_area_cm_squared : ℝ := 120 + 100
def total_area_miles_squared : ℝ := total_area_cm_squared * (scale_cm_to_miles ^ 2)
def total_area_acres : ℝ := total_area_miles_squared * sq_mile_to_acres

-- The theorem statement
theorem actual_total_area_in_acres : total_area_acres = 1267200 :=
by
  sorry

end actual_total_area_in_acres_l175_175953


namespace jungkook_mother_age_four_times_jungkook_age_l175_175939

-- Definitions of conditions
def jungkoo_age : ℕ := 16
def mother_age : ℕ := 46

-- Theorem statement for the problem
theorem jungkook_mother_age_four_times_jungkook_age :
  ∃ (x : ℕ), (mother_age - x = 4 * (jungkoo_age - x)) ∧ x = 6 :=
by
  sorry

end jungkook_mother_age_four_times_jungkook_age_l175_175939


namespace strawberries_per_person_l175_175787

noncomputable def total_strawberries (baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  baskets * strawberries_per_basket

noncomputable def kimberly_strawberries (brother_strawberries : ℕ) : ℕ :=
  8 * brother_strawberries

noncomputable def parents_strawberries (kimberly_strawberries : ℕ) : ℕ :=
  kimberly_strawberries - 93

noncomputable def total_family_strawberries (kimberly : ℕ) (brother : ℕ) (parents : ℕ) : ℕ :=
  kimberly + brother + parents

noncomputable def equal_division (total_strawberries : ℕ) (people : ℕ) : ℕ :=
  total_strawberries / people

theorem strawberries_per_person :
  let brother_baskets := 3
  let strawberries_per_basket := 15
  let brother_strawberries := total_strawberries brother_baskets strawberries_per_basket
  let kimberly_straw := kimberly_strawberries brother_strawberries
  let parents_straw := parents_strawberries kimberly_straw
  let total := total_family_strawberries kimberly_straw brother_strawberries parents_straw
  equal_division total 4 = 168 :=
by
  simp [total_strawberries, kimberly_strawberries, parents_strawberries, total_family_strawberries, equal_division]
  sorry

end strawberries_per_person_l175_175787


namespace sum_of_n_values_l175_175533

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l175_175533


namespace percentage_of_number_l175_175570

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l175_175570


namespace john_bought_six_bagels_l175_175803

theorem john_bought_six_bagels (b m : ℕ) (expenditure_in_dollars_whole : (90 * b + 60 * m) % 100 = 0) (total_items : b + m = 7) : 
b = 6 :=
by
  -- The proof goes here. For now, we skip it with sorry.
  sorry

end john_bought_six_bagels_l175_175803


namespace base3_to_base10_equiv_l175_175594

theorem base3_to_base10_equiv : 
  let repr := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  repr = 142 :=
by
  sorry

end base3_to_base10_equiv_l175_175594


namespace bob_mean_score_l175_175674

-- Conditions
def scores : List ℝ := [68, 72, 76, 80, 85, 90]
def alice_scores (a1 a2 a3 : ℝ) : Prop := a1 < a2 ∧ a2 < a3 ∧ a1 + a2 + a3 = 225
def bob_scores (b1 b2 b3 : ℝ) : Prop := b1 + b2 + b3 = 246

-- Theorem statement proving Bob's mean score
theorem bob_mean_score (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1 ∈ scores) (h2 : a2 ∈ scores) (h3 : a3 ∈ scores)
  (h4 : b1 ∈ scores) (h5 : b2 ∈ scores) (h6 : b3 ∈ scores)
  (h7 : alice_scores a1 a2 a3)
  (h8 : bob_scores b1 b2 b3)
  (h9 : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3)
  : (b1 + b2 + b3) / 3 = 82 :=
sorry

end bob_mean_score_l175_175674


namespace negation_equiv_l175_175201

noncomputable def negate_existential : Prop :=
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0

noncomputable def universal_negation : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0

theorem negation_equiv : negate_existential = universal_negation :=
by
  -- Proof to be filled in
  sorry

end negation_equiv_l175_175201


namespace islander_distances_l175_175840

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end islander_distances_l175_175840


namespace transformed_system_solution_l175_175034

theorem transformed_system_solution :
  (∀ (a b : ℝ), 2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 → a = 8.3 ∧ b = 1.2) →
  (∀ (x y : ℝ), 2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9 →
    x = 6.3 ∧ y = 2.2) :=
by
  intro h1
  intro x y
  intro hy
  sorry

end transformed_system_solution_l175_175034


namespace soja_finished_fraction_l175_175675

def pages_finished (x pages_left total_pages : ℕ) : Prop :=
  x - pages_left = 100 ∧ x + pages_left = total_pages

noncomputable def fraction_finished (x total_pages : ℕ) : ℚ :=
  x / total_pages

theorem soja_finished_fraction (x : ℕ) (h1 : pages_finished x (x - 100) 300) :
  fraction_finished x 300 = 2 / 3 :=
by
  sorry

end soja_finished_fraction_l175_175675


namespace a_lt_one_l175_175063

-- Define the function f(x) = |x-3| + |x+7|
def f (x : ℝ) : ℝ := |x-3| + |x+7|

-- The statement of the problem
theorem a_lt_one (a : ℝ) :
  (∀ x : ℝ, a < Real.log (f x)) → a < 1 :=
by
  intro h
  have H : f (-7) = 10 := by sorry -- piecewise definition
  have H1 : Real.log (f (-7)) = 1 := by sorry -- minimum value of log
  specialize h (-7)
  rw [H1] at h
  exact h

end a_lt_one_l175_175063


namespace balls_in_boxes_l175_175764

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l175_175764


namespace cleaning_time_l175_175975

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l175_175975


namespace part1_solution_part2_no_solution_l175_175378

theorem part1_solution (x y : ℚ) :
  x + y = 5 ∧ 3 * x + 10 * y = 30 ↔ x = 20 / 7 ∧ y = 15 / 7 :=
by
  sorry

theorem part2_no_solution (x : ℚ) :
  (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 ↔ False :=
by
  sorry

end part1_solution_part2_no_solution_l175_175378


namespace measles_cases_1995_l175_175293

-- Definitions based on the conditions
def initial_cases_1970 : ℕ := 300000
def final_cases_2000 : ℕ := 200
def cases_1990 : ℕ := 1000
def decrease_rate : ℕ := 14950 -- Annual linear decrease from 1970-1990
def a : ℤ := -8 -- Coefficient for the quadratic phase

-- Function modeling the number of cases in the quadratic phase (1990-2000)
def measles_cases (x : ℕ) : ℤ := a * (x - 1990)^2 + cases_1990

-- The statement we want to prove
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end measles_cases_1995_l175_175293


namespace distance_to_lateral_face_l175_175929

theorem distance_to_lateral_face 
  (height : ℝ) 
  (angle : ℝ) 
  (h_height : height = 6 * Real.sqrt 6)
  (h_angle : angle = Real.pi / 4) : 
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 30 / 5 :=
by
  sorry

end distance_to_lateral_face_l175_175929


namespace polynomial_abc_l175_175452

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l175_175452


namespace balls_in_boxes_l175_175765

theorem balls_in_boxes : 
  let boxes := 4
      balls := 5
      arrangements := 
        4  -- (5, 0, 0, 0)
        + (4 * 3)  -- (4, 1, 0, 0)
        + (4 * 3)  -- (3, 2, 0, 0)
        + (4 * 3)  -- (3, 1, 1, 0)
        + (binom 4 2 * 2)  -- (2, 2, 1, 0)
        + 4  -- (2, 1, 1, 1)
  in arrangements = 56 := by
  sorry

end balls_in_boxes_l175_175765


namespace smallest_6_digit_div_by_111_l175_175705

theorem smallest_6_digit_div_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 := by
  sorry

end smallest_6_digit_div_by_111_l175_175705


namespace probability_of_point_in_smaller_square_l175_175128

-- Definitions
def A_large : ℝ := 5 * 5
def A_small : ℝ := 2 * 2

-- Theorem statement
theorem probability_of_point_in_smaller_square 
  (side_large : ℝ) (side_small : ℝ)
  (hle : side_large = 5) (hse : side_small = 2) :
  (side_large * side_large ≠ 0) ∧ (side_small * side_small ≠ 0) → 
  (A_small / A_large = 4 / 25) :=
sorry

end probability_of_point_in_smaller_square_l175_175128


namespace sum_of_solutions_of_absolute_value_l175_175527

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l175_175527


namespace solve_z_solutions_l175_175870

noncomputable def z_solutions (z : ℂ) : Prop :=
  z ^ 6 = -16

theorem solve_z_solutions :
  {z : ℂ | z_solutions z} = {2 * Complex.I, -2 * Complex.I} :=
by {
  sorry
}

end solve_z_solutions_l175_175870


namespace probability_at_least_one_even_l175_175573

theorem probability_at_least_one_even 
  (cards : Finset ℕ) (h_card_count : cards.card = 9) (h_card_range : ∀ x ∈ cards, 1 ≤ x ∧ x ≤ 9) :
  let draws := cards.powerset.filter (λ s, s.card = 2)
  let evens := {x ∈ cards | x % 2 = 0}
  let favorable := draws.filter (λ s, ¬ s.disjoint evens)
  (favorable.card : ℚ) / (draws.card : ℚ) = 13/18 :=
sorry

end probability_at_least_one_even_l175_175573


namespace first_term_of_geometric_series_l175_175401

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l175_175401


namespace nonneg_triple_inequality_l175_175656

theorem nonneg_triple_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) :=
by
  sorry

end nonneg_triple_inequality_l175_175656


namespace farmer_land_l175_175486

theorem farmer_land (A : ℝ) (h1 : 0.9 * A = A_cleared) (h2 : 0.3 * A_cleared = A_soybeans) 
  (h3 : 0.6 * A_cleared = A_wheat) (h4 : 0.1 * A_cleared = 540) : A = 6000 :=
by
  sorry

end farmer_land_l175_175486


namespace rational_sum_zero_l175_175618

theorem rational_sum_zero (x1 x2 x3 x4 : ℚ)
  (h1 : x1 = x2 + x3 + x4)
  (h2 : x2 = x1 + x3 + x4)
  (h3 : x3 = x1 + x2 + x4)
  (h4 : x4 = x1 + x2 + x3) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 := 
sorry

end rational_sum_zero_l175_175618


namespace find_number_l175_175688

theorem find_number (x : ℤ) (h : 33 + 3 * x = 48) : x = 5 :=
by
  sorry

end find_number_l175_175688


namespace range_of_m_l175_175643

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x ^ 2 - 2 * (4 - m) * x + 1
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end range_of_m_l175_175643


namespace min_value_3x_4y_l175_175677

open Real

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y = 5 :=
by
  sorry

end min_value_3x_4y_l175_175677


namespace sugar_ratio_l175_175409

theorem sugar_ratio (total_sugar : ℕ)  (bags : ℕ) (remaining_sugar : ℕ) (sugar_each_bag : ℕ) (sugar_fell : ℕ)
  (h1 : total_sugar = 24) (h2 : bags = 4) (h3 : total_sugar - remaining_sugar = sugar_fell) 
  (h4 : total_sugar / bags = sugar_each_bag) (h5 : remaining_sugar = 21) : 
  2 * sugar_fell = sugar_each_bag := by
  -- proof goes here
  sorry

end sugar_ratio_l175_175409


namespace radian_measure_of_acute_angle_l175_175941

theorem radian_measure_of_acute_angle 
  (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
  (θ : ℝ) (S U : ℝ) 
  (hS : S = U * 9 / 14) (h_total_area : (π * r1^2) + (π * r2^2) + (π * r3^2) = S + U) :
  θ = 1827 * π / 3220 :=
by
  -- proof goes here
  sorry

end radian_measure_of_acute_angle_l175_175941


namespace increasing_interval_l175_175093

noncomputable def f (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + (3 / 2) * x ^ 2 - x

theorem increasing_interval : ∀ x y : ℝ, (1/2) ≤ x → y ≤ 1 → x ≤ y → f x ≤ f y :=
by
  intro x y hx hy hxy
  -- Proof goes here (to be filled in with the proof steps)
  sorry

end increasing_interval_l175_175093


namespace cleaning_time_l175_175976

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l175_175976


namespace integer_solution_unique_l175_175080

theorem integer_solution_unique (w x y z : ℤ) :
  w^2 + 11 * x^2 - 8 * y^2 - 12 * y * z - 10 * z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry
 
end integer_solution_unique_l175_175080


namespace find_two_digit_number_l175_175142

theorem find_two_digit_number : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 10^6 ≤ n^3 ∧ n^3 < 10^7 ∧ 101010 * x + 1 = n^3 ∧ x = 93) := 
 by
  sorry

end find_two_digit_number_l175_175142


namespace paths_from_A_to_B_grid_l175_175190

open Nat

theorem paths_from_A_to_B_grid :
  let grid_width := 6
  let grid_height := 5
  let total_moves := 10
  let right_moves := 6
  let up_moves := 4
  total_moves = right_moves + up_moves →
  grid_width = right_moves →
  grid_height = up_moves →
  ∃ paths : ℕ, paths = choose total_moves up_moves ∧ paths = 210 :=
begin
  intros,
  sorry
end

end paths_from_A_to_B_grid_l175_175190


namespace compare_tan_neg_values_l175_175981

theorem compare_tan_neg_values :
  tan (- (13 * Real.pi / 7)) > tan (- (15 * Real.pi / 8)) :=
by
  sorry

end compare_tan_neg_values_l175_175981


namespace not_super_lucky_years_l175_175718

def sum_of_month_and_day (m d : ℕ) : ℕ := m + d
def product_of_month_and_day (m d : ℕ) : ℕ := m * d
def sum_of_last_two_digits (y : ℕ) : ℕ :=
  let d1 := y / 10 % 10
  let d2 := y % 10
  d1 + d2

def is_super_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), sum_of_month_and_day m d = 24 ∧
               product_of_month_and_day m d = 2 * sum_of_last_two_digits y

theorem not_super_lucky_years :
  ¬ is_super_lucky_year 2070 ∧
  ¬ is_super_lucky_year 2081 ∧
  ¬ is_super_lucky_year 2092 :=
by {
  sorry
}

end not_super_lucky_years_l175_175718


namespace alchemerion_age_problem_l175_175394

theorem alchemerion_age_problem
  (A S F : ℕ)  -- Declare the ages as natural numbers
  (h1 : A = 3 * S)  -- Condition 1: Alchemerion is 3 times his son's age
  (h2 : F = 2 * A + 40)  -- Condition 2: His father’s age is 40 years more than twice his age
  (h3 : A + S + F = 1240)  -- Condition 3: Together they are 1240 years old
  (h4 : A = 360)  -- Condition 4: Alchemerion is 360 years old
  : 40 = F - 2 * A :=  -- Conclusion: The number of years more than twice Alchemerion’s age is 40
by
  sorry  -- Proof can be filled in here

end alchemerion_age_problem_l175_175394


namespace ratio_c_d_l175_175208

theorem ratio_c_d (a b c d : ℝ) (h_eq : ∀ x, a * x^3 + b * x^2 + c * x + d = 0) 
    (h_roots : ∀ r, r = 2 ∨ r = 4 ∨ r = 5 ↔ (a * r^3 + b * r^2 + c * r + d = 0)) :
    c / d = 19 / 20 :=
by
  sorry

end ratio_c_d_l175_175208


namespace area_increase_by_40_percent_l175_175517

theorem area_increase_by_40_percent (s : ℝ) : 
  let A1 := s^2 
  let new_side := 1.40 * s 
  let A2 := new_side^2 
  (A2 - A1) / A1 * 100 = 96 := 
by 
  sorry

end area_increase_by_40_percent_l175_175517


namespace jakes_digging_time_l175_175309

theorem jakes_digging_time
  (J : ℕ)
  (Paul_work_rate : ℚ := 1/24)
  (Hari_work_rate : ℚ := 1/48)
  (Combined_work_rate : ℚ := 1/8)
  (Combined_work_eq : 1 / J + Paul_work_rate + Hari_work_rate = Combined_work_rate) :
  J = 16 := sorry

end jakes_digging_time_l175_175309


namespace moles_NaOH_to_form_H2O_2_moles_l175_175867

-- Define the reaction and moles involved
def reaction : String := "NH4NO3 + NaOH -> NaNO3 + NH3 + H2O"
def moles_H2O_produced : Nat := 2
def moles_NaOH_required (moles_H2O : Nat) : Nat := moles_H2O

-- Theorem stating the required moles of NaOH to produce 2 moles of H2O
theorem moles_NaOH_to_form_H2O_2_moles : moles_NaOH_required moles_H2O_produced = 2 := 
by
  sorry

end moles_NaOH_to_form_H2O_2_moles_l175_175867


namespace Shane_current_age_44_l175_175691

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end Shane_current_age_44_l175_175691


namespace find_a_l175_175436

theorem find_a
  (a : ℝ)
  (h_perpendicular : ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 3 * x - 6 * y - 1 = 0 → true) :
  a = 4 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end find_a_l175_175436


namespace surface_area_of_sphere_l175_175714

theorem surface_area_of_sphere (l w h : ℝ) (s t : ℝ) :
  l = 3 ∧ w = 2 ∧ h = 1 ∧ (s = (l^2 + w^2 + h^2).sqrt / 2) → t = 4 * Real.pi * s^2 → t = 14 * Real.pi :=
by
  intros
  sorry

end surface_area_of_sphere_l175_175714


namespace fraction_calculation_l175_175858

theorem fraction_calculation : 
  (1 / 4 + 1 / 6 - 1 / 2) / (-1 / 24) = 2 := 
by 
  sorry

end fraction_calculation_l175_175858


namespace students_in_class_l175_175260

theorem students_in_class
 (S : ℕ)
 (erasers_original pencils_original erasers_left pencils_left : ℕ)
 (h_erasers : erasers_original = 49)
 (h_pencils : pencils_original = 66)
 (h_erasers_left : erasers_left = 4)
 (h_pencils_left : pencils_left = 6)
 (erasers_to_divide : ℕ := erasers_original - erasers_left)
 (pencils_to_divide : ℕ := pencils_original - pencils_left)
 (h_erasers_to_divide : erasers_to_divide = 45)
 (h_pencils_to_divide : pencils_to_divide = 60)
 (h_divide_erasers : 45 % S = 0) 
 (h_divide_pencils : 60 % S = 0):
 S = 15 := 
sorry

end students_in_class_l175_175260


namespace volume_of_rotated_solid_l175_175346

theorem volume_of_rotated_solid (unit_cylinder_r1 h1 r2 h2 : ℝ) :
  unit_cylinder_r1 = 6 → h1 = 1 → r2 = 3 → h2 = 4 → 
  (π * unit_cylinder_r1^2 * h1 + π * r2^2 * h2) = 72 * π :=
by 
-- We place the arguments and sorry for skipping the proof
  sorry

end volume_of_rotated_solid_l175_175346


namespace distance_at_40_kmph_l175_175835

theorem distance_at_40_kmph (x : ℝ) (h1 : x / 40 + (250 - x) / 60 = 5) : x = 100 := 
by
  sorry

end distance_at_40_kmph_l175_175835


namespace gcd_of_all_three_digit_palindromes_is_one_l175_175365

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l175_175365


namespace highest_y_coordinate_l175_175984

theorem highest_y_coordinate : 
  (∀ x y : ℝ, ((x - 4)^2 / 25 + y^2 / 49 = 0) → y = 0) := 
by
  sorry

end highest_y_coordinate_l175_175984


namespace total_grazing_area_l175_175122

-- Define the dimensions of the field
def field_width : ℝ := 46
def field_height : ℝ := 20

-- Define the length of the rope
def rope_length : ℝ := 17

-- Define the radius and position of the fenced area
def fenced_radius : ℝ := 5
def fenced_distance_x : ℝ := 25
def fenced_distance_y : ℝ := 10

-- Given the conditions, prove the total grazing area
theorem total_grazing_area (field_width field_height rope_length fenced_radius fenced_distance_x fenced_distance_y : ℝ) :
  (π * rope_length^2 / 4) = 227.07 :=
by
  sorry

end total_grazing_area_l175_175122


namespace min_time_calculation_l175_175916

noncomputable def min_time_to_receive_keys (diameter cyclist_speed_road cyclist_speed_alley pedestrian_speed : ℝ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let distance_pedestrian := pedestrian_speed * 1
  let min_time := (2 * Real.pi * radius - 2 * distance_pedestrian) / (cyclist_speed_road + cyclist_speed_alley)
  min_time

theorem min_time_calculation :
  min_time_to_receive_keys 4 15 20 6 = (2 * Real.pi - 2) / 21 :=
by
  sorry

end min_time_calculation_l175_175916


namespace correct_dispersion_statements_l175_175585

def statement1 (make_use_of_data : Prop) : Prop :=
make_use_of_data = true

def statement2 (multi_numerical_values : Prop) : Prop :=
multi_numerical_values = true

def statement3 (dispersion_large_value_small : Prop) : Prop :=
dispersion_large_value_small = false

theorem correct_dispersion_statements
  (make_use_of_data : Prop)
  (multi_numerical_values : Prop)
  (dispersion_large_value_small : Prop)
  (h1 : statement1 make_use_of_data)
  (h2 : statement2 multi_numerical_values)
  (h3 : statement3 dispersion_large_value_small) :
  (make_use_of_data ∧ multi_numerical_values ∧ ¬ dispersion_large_value_small) = true :=
by
  sorry

end correct_dispersion_statements_l175_175585


namespace train_a_distance_at_meeting_l175_175374

noncomputable def train_a_speed : ℝ := 75 / 3
noncomputable def train_b_speed : ℝ := 75 / 2
noncomputable def relative_speed : ℝ := train_a_speed + train_b_speed
noncomputable def time_until_meet : ℝ := 75 / relative_speed
noncomputable def distance_traveled_by_train_a : ℝ := train_a_speed * time_until_meet

theorem train_a_distance_at_meeting : distance_traveled_by_train_a = 30 := by
  sorry

end train_a_distance_at_meeting_l175_175374


namespace company_pays_per_month_l175_175108

theorem company_pays_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1.08 * 10^6)
  (h5 : cost_per_box = 0.6) :
  (total_volume / (length * width * height) * cost_per_box) = 360 :=
by
  -- sorry to skip proof
  sorry

end company_pays_per_month_l175_175108


namespace negation_of_proposition_l175_175033

theorem negation_of_proposition (a b : ℝ) : 
  (¬ (∀ (a b : ℝ), (ab > 0 → a > 0)) ↔ ∀ (a b : ℝ), (ab ≤ 0 → a ≤ 0)) := 
sorry

end negation_of_proposition_l175_175033


namespace bacteria_seventh_generation_l175_175670

/-- Represents the effective multiplication factor per generation --/
def effective_mult_factor : ℕ := 4

/-- The number of bacteria in the first generation --/
def first_generation : ℕ := 1

/-- A helper function to compute the number of bacteria in the nth generation --/
def bacteria_count (n : ℕ) : ℕ :=
  first_generation * effective_mult_factor ^ n

/-- The number of bacteria in the seventh generation --/
theorem bacteria_seventh_generation : bacteria_count 7 = 4096 := by
  sorry

end bacteria_seventh_generation_l175_175670


namespace conditionA_is_necessary_for_conditionB_l175_175075

-- Definitions for conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (area : ℝ) -- area of the triangle

def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

def conditionA (t1 t2 : Triangle) : Prop :=
  t1.area = t2.area ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem conditionA_is_necessary_for_conditionB (t1 t2 : Triangle) :
  congruent t1 t2 → conditionA t1 t2 :=
by sorry

end conditionA_is_necessary_for_conditionB_l175_175075


namespace eulerian_orientation_exists_l175_175457

theorem eulerian_orientation_exists {V : Type*} (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj]
  (h_connected : G.Connected) (h_even_degree : ∀ v : V, Even (G.degree v)) :
  ∃ (G_oriented : SimpleGraph V), (∀ v : V, (G_oriented.degree v = G.degree v / 2)) ∧ (∀ u v : V, G_oriented.Connected) :=
by
  sorry

end eulerian_orientation_exists_l175_175457


namespace shares_of_valuable_stock_l175_175662

theorem shares_of_valuable_stock 
  (price_val : ℕ := 78)
  (price_oth : ℕ := 39)
  (shares_oth : ℕ := 26)
  (total_asset : ℕ := 2106)
  (x : ℕ) 
  (h_val_stock : total_asset = 78 * x + 39 * 26) : 
  x = 14 :=
by
  sorry

end shares_of_valuable_stock_l175_175662


namespace boat_speed_in_still_water_l175_175053

-- Identifying the speeds of the boat in still water and the stream
variables (b s : ℝ)

-- Conditions stated in terms of equations
axiom boat_along_stream : b + s = 7
axiom boat_against_stream : b - s = 5

-- Prove that the boat speed in still water is 6 km/hr
theorem boat_speed_in_still_water : b = 6 :=
by
  sorry

end boat_speed_in_still_water_l175_175053


namespace sum_of_first_six_terms_geometric_sequence_l175_175003

theorem sum_of_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r ^ n) / (1 - r)
  S_n = 1365 / 4096 := by
  sorry

end sum_of_first_six_terms_geometric_sequence_l175_175003


namespace not_square_l175_175319

open Int

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ k : ℤ, (a^2 : ℤ) + ⌈(4 * a^2 : ℤ) / b⌉ = k^2 :=
by
  sorry

end not_square_l175_175319


namespace compute_result_l175_175592

-- Define the operations a # b and b # c
def operation (a b : ℤ) : ℤ := a * b - b + b^2

-- Define the expression for (3 # 8) # z given the operations
def evaluate (z : ℤ) : ℤ := operation (operation 3 8) z

-- Prove that (3 # 8) # z = 79z + z^2
theorem compute_result (z : ℤ) : evaluate z = 79 * z + z^2 := 
by
  sorry

end compute_result_l175_175592


namespace thirteenth_result_is_128_l175_175198

theorem thirteenth_result_is_128 
  (avg_all : ℕ → ℕ → ℕ) (avg_first : ℕ → ℕ → ℕ) (avg_last : ℕ → ℕ → ℕ) :
  avg_all 25 20 = (avg_first 12 14) + (avg_last 12 17) + 128 :=
by
  sorry

end thirteenth_result_is_128_l175_175198


namespace distance_focus_to_asymptote_l175_175433

theorem distance_focus_to_asymptote (m : ℝ) (x y : ℝ) (h1 : (x^2) / 9 - (y^2) / m = 1) 
  (h2 : (Real.sqrt 14) / 3 = (Real.sqrt (9 + m)) / 3) : 
  ∃ d : ℝ, d = Real.sqrt 5 := 
by 
  sorry

end distance_focus_to_asymptote_l175_175433


namespace trajectory_equation_l175_175756

open Real

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the moving point P
def P (x y : ℝ) : Prop := 
  (4 * Real.sqrt ((x + 2) ^ 2 + y ^ 2) + 4 * (x - 2) = 0) → 
  (y ^ 2 = -8 * x)

-- The theorem stating the desired proof problem
theorem trajectory_equation (x y : ℝ) : P x y :=
sorry

end trajectory_equation_l175_175756


namespace parabola_problem_l175_175888

-- defining the geometric entities and conditions
variables {x y k x1 y1 x2 y2 : ℝ}

-- the definition for the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- the definition for point M
def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 2)

-- the definition for line passing through focus with slope k intersecting the parabola at A and B
def line_through_focus_and_k (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1))

-- the definition for vectors MA and MB having dot product zero
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 - 2 * (y1 + y2) + 4 = 0)

-- the main statement to be proved
theorem parabola_problem
  (h_parabola_A : parabola x1 y1)
  (h_parabola_B : parabola x2 y2)
  (h_point_M : point_M 0 2)
  (h_line_through_focus_and_k : line_through_focus_and_k x1 y1 x2 y2 k)
  (h_orthogonal_vectors : orthogonal_vectors x1 y1 x2 y2) :
  k = 1 :=
sorry

end parabola_problem_l175_175888


namespace min_coins_cover_99_l175_175696

def coin_values : List ℕ := [1, 5, 10, 25, 50]

noncomputable def min_coins_cover (n : ℕ) : ℕ := sorry

theorem min_coins_cover_99 : min_coins_cover 99 = 9 :=
  sorry

end min_coins_cover_99_l175_175696


namespace mod_equivalence_l175_175045

theorem mod_equivalence (x y m : ℤ) (h1 : x ≡ 25 [ZMOD 60]) (h2 : y ≡ 98 [ZMOD 60]) (h3 : m = 167) :
  x - y ≡ m [ZMOD 60] :=
sorry

end mod_equivalence_l175_175045


namespace total_toothpicks_grid_area_l175_175357

open Nat

-- Definitions
def grid_length : Nat := 30
def grid_width : Nat := 50

-- Prove the total number of toothpicks
theorem total_toothpicks : (31 * grid_width + 51 * grid_length) = 3080 := by
  sorry

-- Prove the area enclosed by the grid
theorem grid_area : (grid_length * grid_width) = 1500 := by
  sorry

end total_toothpicks_grid_area_l175_175357


namespace cost_of_six_burritos_and_seven_sandwiches_l175_175859

variable (b s : ℝ)
variable (h1 : 4 * b + 2 * s = 5.00)
variable (h2 : 3 * b + 5 * s = 6.50)

theorem cost_of_six_burritos_and_seven_sandwiches : 6 * b + 7 * s = 11.50 :=
  sorry

end cost_of_six_burritos_and_seven_sandwiches_l175_175859


namespace find_diameter_l175_175604

noncomputable def cost_per_meter : ℝ := 2
noncomputable def total_cost : ℝ := 188.49555921538757
noncomputable def circumference (c : ℝ) (p : ℝ) : ℝ := c / p
noncomputable def diameter (c : ℝ) : ℝ := c / Real.pi

theorem find_diameter :
  diameter (circumference total_cost cost_per_meter) = 30 := by
  sorry

end find_diameter_l175_175604


namespace hypotenuse_length_l175_175242

theorem hypotenuse_length (x y h : ℝ)
  (hx : (1 / 3) * π * y * x^2 = 1620 * π)
  (hy : (1 / 3) * π * x * y^2 = 3240 * π) :
  h = Real.sqrt 507 :=
by
  sorry

end hypotenuse_length_l175_175242


namespace courtyard_length_l175_175636

theorem courtyard_length 
  (stone_area : ℕ) 
  (stones_total : ℕ) 
  (width : ℕ)
  (total_area : ℕ) 
  (L : ℕ) 
  (h1 : stone_area = 4)
  (h2 : stones_total = 135)
  (h3 : width = 18)
  (h4 : total_area = stones_total * stone_area)
  (h5 : total_area = L * width) :
  L = 30 :=
by
  -- Proof steps would go here
  sorry

end courtyard_length_l175_175636


namespace volunteer_count_change_l175_175717

theorem volunteer_count_change :
  let x := 1
  let fall_increase := 1.09
  let winter_increase := 1.15
  let spring_decrease := 0.81
  let summer_increase := 1.12
  let summer_end_decrease := 0.95
  let final_ratio := x * fall_increase * winter_increase * spring_decrease * summer_increase * summer_end_decrease
  (final_ratio - x) / x * 100 = 19.13 :=
by
  sorry

end volunteer_count_change_l175_175717


namespace students_last_year_l175_175307

theorem students_last_year (students_this_year : ℝ) (increase_percent : ℝ) (last_year_students : ℝ) 
  (h1 : students_this_year = 960) 
  (h2 : increase_percent = 0.20) 
  (h3 : students_this_year = last_year_students * (1 + increase_percent)) : 
  last_year_students = 800 :=
by 
  sorry

end students_last_year_l175_175307


namespace smallest_m_plus_n_l175_175657

theorem smallest_m_plus_n (m n : ℕ) (hmn : m > n) (hid : (2012^m : ℕ) % 1000 = (2012^n) % 1000) : m + n = 104 :=
sorry

end smallest_m_plus_n_l175_175657


namespace polynomial_abc_value_l175_175450

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l175_175450


namespace sum_of_squares_is_perfect_square_l175_175992

theorem sum_of_squares_is_perfect_square (n p k : ℤ) : 
  (∃ m : ℤ, n^2 + p^2 + k^2 = m^2) ↔ (n * k = (p / 2)^2) :=
by
  sorry

end sum_of_squares_is_perfect_square_l175_175992


namespace problem_solution_l175_175275

def f (x : ℕ) : ℝ := sorry

axiom f_add_eq_mul (p q : ℕ) : f (p + q) = f p * f q
axiom f_one_eq_three : f 1 = 3

theorem problem_solution :
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 + 
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 = 24 := 
by
  sorry

end problem_solution_l175_175275


namespace only_point_D_lies_on_graph_l175_175107

def point := ℤ × ℤ

def lies_on_graph (f : ℤ → ℤ) (p : point) : Prop :=
  f p.1 = p.2

def f (x : ℤ) : ℤ := 2 * x - 1

theorem only_point_D_lies_on_graph :
  (lies_on_graph f (-1, 3) = false) ∧ 
  (lies_on_graph f (0, 1) = false) ∧ 
  (lies_on_graph f (1, -1) = false) ∧ 
  (lies_on_graph f (2, 3)) := 
by
  sorry

end only_point_D_lies_on_graph_l175_175107


namespace axis_symmetry_shifted_graph_l175_175291

open Real

theorem axis_symmetry_shifted_graph :
  ∀ k : ℤ, ∃ x : ℝ, (y = 2 * sin (2 * x)) ∧
  y = 2 * sin (2 * (x + π / 12)) ↔
  x = k * π / 2 + π / 6 :=
sorry

end axis_symmetry_shifted_graph_l175_175291


namespace sqrt_of_product_eq_540_l175_175254

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l175_175254


namespace cost_of_two_pencils_and_one_pen_l175_175199

variable (a b : ℝ)

-- Given conditions
def condition1 : Prop := (5 * a + b = 2.50)
def condition2 : Prop := (a + 2 * b = 1.85)

-- Statement to prove
theorem cost_of_two_pencils_and_one_pen
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  2 * a + b = 1.45 :=
sorry

end cost_of_two_pencils_and_one_pen_l175_175199


namespace fraction_without_cable_or_vcr_l175_175780

theorem fraction_without_cable_or_vcr (T : ℕ) (h1 : ℚ) (h2 : ℚ) (h3 : ℚ) 
  (h1 : h1 = 1 / 5 * T) 
  (h2 : h2 = 1 / 10 * T) 
  (h3 : h3 = 1 / 3 * (1 / 5 * T)) 
: (T - (1 / 5 * T + 1 / 10 * T - 1 / 3 * (1 / 5 * T))) / T = 23 / 30 := 
by 
  sorry

end fraction_without_cable_or_vcr_l175_175780


namespace sequence_a_b_10_l175_175800

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l175_175800


namespace arithmetic_sequence_a5_l175_175623

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) = a n + 2

-- Statement of the theorem with conditions and conclusion
theorem arithmetic_sequence_a5 :
  ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ a 1 = 1 ∧ a 5 = 9 :=
by
  sorry

end arithmetic_sequence_a5_l175_175623


namespace geom_series_first_term_l175_175405

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l175_175405


namespace solve_for_pure_imaginary_l175_175441

theorem solve_for_pure_imaginary (x : ℝ) 
  (h1 : x^2 - 1 = 0) 
  (h2 : x - 1 ≠ 0) 
  : x = -1 :=
sorry

end solve_for_pure_imaginary_l175_175441


namespace train_length_correct_l175_175371

noncomputable def speed_kmph : ℝ := 60
noncomputable def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
noncomputable def conversion_factor := (1000 : ℝ) / 3600

-- Speed in m/s
noncomputable def speed_mps := speed_kmph * conversion_factor

-- Length of the train
noncomputable def train_length := speed_mps * time_sec

theorem train_length_correct :
  train_length = 100.02 :=
by
  sorry

end train_length_correct_l175_175371


namespace minimum_sum_of_x_and_y_l175_175620

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 4 * y = x * y

theorem minimum_sum_of_x_and_y (x y : ℝ) (h : conditions x y) : x + y ≥ 9 := by
  sorry

end minimum_sum_of_x_and_y_l175_175620


namespace quadratic_solution_unique_l175_175332

noncomputable def solve_quad_eq (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) : ℝ :=
-2 / 3

theorem quadratic_solution_unique (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) :
  (∃! x : ℝ, a * x^2 + 36 * x + 12 = 0) ∧ (solve_quad_eq a h h_uniq) = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l175_175332


namespace triangle_area_is_sqrt3_over_4_l175_175427

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem triangle_area_is_sqrt3_over_4
  (a b c A B : ℝ)
  (h1 : A = Real.pi / 3)
  (h2 : b = 2 * a * Real.cos B)
  (h3 : c = 1)
  (h4 : B = Real.pi / 3)
  (h5 : a = 1)
  (h6 : b = 1) :
  area_of_triangle a b c A B (Real.pi - A - B) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_is_sqrt3_over_4_l175_175427


namespace probability_of_two_accurate_forecasts_l175_175684

noncomputable def event_A : Type := {forecast : ℕ | forecast = 1}

def prob_A : ℝ := 0.9
def prob_A' : ℝ := 1 - prob_A

-- Define that there are 3 independent trials
def num_forecasts : ℕ := 3

-- Given
def probability_two_accurate (x : ℕ) : ℝ :=
if x = 2 then 3 * (prob_A^2 * prob_A') else 0

-- Statement to be proved
theorem probability_of_two_accurate_forecasts : probability_two_accurate 2 = 0.243 := by
  -- Proof will go here
  sorry

end probability_of_two_accurate_forecasts_l175_175684


namespace geometric_series_first_term_l175_175407

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l175_175407


namespace total_photos_newspaper_l175_175303

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l175_175303


namespace sequence_monotonic_decreasing_l175_175426

theorem sequence_monotonic_decreasing (t : ℝ) :
  (∀ n : ℕ, n > 0 → (- (n + 1) ^ 2 + t * (n + 1)) - (- n ^ 2 + t * n) < 0) ↔ (t < 3) :=
by 
  sorry

end sequence_monotonic_decreasing_l175_175426


namespace found_bottle_caps_is_correct_l175_175734

def initial_bottle_caps : ℕ := 6
def total_bottle_caps : ℕ := 28

theorem found_bottle_caps_is_correct : total_bottle_caps - initial_bottle_caps = 22 := by
  sorry

end found_bottle_caps_is_correct_l175_175734


namespace convert_255_to_base8_l175_175411

-- Define the conversion function from base 10 to base 8
def base10_to_base8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let r2 := n % 64
  let d1 := r2 / 8
  let r1 := r2 % 8
  d2 * 100 + d1 * 10 + r1

-- Define the specific number and base in the conditions
def num10 : ℕ := 255
def base8_result : ℕ := 377

-- The theorem stating the proof problem
theorem convert_255_to_base8 : base10_to_base8 num10 = base8_result :=
by
  -- You would provide the proof steps here
  sorry

end convert_255_to_base8_l175_175411


namespace evaluate_expression_l175_175609

theorem evaluate_expression (x : ℝ) (h : 3 * x^3 - x = 1) : 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := 
by
  sorry

end evaluate_expression_l175_175609


namespace work_done_in_a_day_l175_175708

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l175_175708


namespace third_median_length_l175_175055

noncomputable def triangle_median_length (m₁ m₂ : ℝ) (area : ℝ) : ℝ :=
  if m₁ = 5 ∧ m₂ = 4 ∧ area = 6 * Real.sqrt 5 then
    3 * Real.sqrt 7
  else
    0

theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ)
  (h₁ : m₁ = 5) (h₂ : m₂ = 4) (h₃ : area = 6 * Real.sqrt 5) :
  triangle_median_length m₁ m₂ area = 3 * Real.sqrt 7 :=
by
  -- Proof is skipped
  sorry

end third_median_length_l175_175055


namespace sam_initial_pennies_l175_175673

def initial_pennies_spent (spent: Nat) (left: Nat) : Nat :=
  spent + left

theorem sam_initial_pennies (spent: Nat) (left: Nat) : spent = 93 ∧ left = 5 → initial_pennies_spent spent left = 98 :=
by
  sorry

end sam_initial_pennies_l175_175673


namespace mass_scientific_notation_l175_175931

def mass := 37e-6

theorem mass_scientific_notation : mass = 3.7 * 10^(-5) :=
by
  sorry

end mass_scientific_notation_l175_175931


namespace gcd_eq_55_l175_175247

theorem gcd_eq_55 : Nat.gcd 5280 12155 = 55 := sorry

end gcd_eq_55_l175_175247


namespace ratio_a_b_l175_175629

theorem ratio_a_b (a b c : ℝ) (h1 : a * (-1) ^ 2 + b * (-1) + c = 1) (h2 : a * 3 ^ 2 + b * 3 + c = 1) : 
  a / b = -2 :=
by 
  sorry

end ratio_a_b_l175_175629


namespace population_net_increase_l175_175306

def birth_rate : ℕ := 8
def birth_time : ℕ := 2
def death_rate : ℕ := 6
def death_time : ℕ := 2
def seconds_per_minute : ℕ := 60
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

theorem population_net_increase :
  (birth_rate / birth_time - death_rate / death_time) * (seconds_per_minute * minutes_per_hour * hours_per_day) = 86400 :=
by
  sorry

end population_net_increase_l175_175306


namespace sum_of_n_values_l175_175534

theorem sum_of_n_values : sum (λ n, |3 * n - 8| = 5) = 16/3 := 
by
  sorry

end sum_of_n_values_l175_175534


namespace minimum_toothpicks_to_remove_l175_175607

-- Definitions related to the problem statement
def total_toothpicks : Nat := 40
def initial_triangles : Nat := 36

-- Ensure that the minimal number of toothpicks to be removed to destroy all triangles is correct.
theorem minimum_toothpicks_to_remove : ∃ (n : Nat), n = 15 ∧ (∀ (t : Nat), t ≤ total_toothpicks - n → t = 0) :=
sorry

end minimum_toothpicks_to_remove_l175_175607


namespace james_added_8_fish_l175_175938

theorem james_added_8_fish
  (initial_fish : ℕ := 60)
  (fish_eaten_per_day : ℕ := 2)
  (total_days_with_worm : ℕ := 21)
  (fish_remaining_when_discovered : ℕ := 26) :
  ∃ (additional_fish : ℕ), additional_fish = 8 :=
by
  let total_fish_eaten := total_days_with_worm * fish_eaten_per_day
  let fish_remaining_without_addition := initial_fish - total_fish_eaten
  let additional_fish := fish_remaining_when_discovered - fish_remaining_without_addition
  exact ⟨additional_fish, sorry⟩

end james_added_8_fish_l175_175938


namespace triangle_inequality_theorem_l175_175546

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l175_175546


namespace number_of_girls_l175_175933

-- Given conditions
def ratio_girls_boys_teachers (girls boys teachers : ℕ) : Prop :=
  3 * (girls + boys + teachers) = 3 * girls + 2 * boys + 1 * teachers

def total_people (total girls boys teachers : ℕ) : Prop :=
  total = girls + boys + teachers

-- Define the main theorem
theorem number_of_girls 
  (k total : ℕ)
  (h1 : ratio_girls_boys_teachers (3 * k) (2 * k) k)
  (h2 : total_people total (3 * k) (2 * k) k)
  (h_total : total = 60) : 
  3 * k = 30 :=
  sorry

end number_of_girls_l175_175933


namespace Lisa_flight_time_l175_175482

theorem Lisa_flight_time :
  let distance := 500
  let speed := 45
  (distance : ℝ) / (speed : ℝ) = 500 / 45 := by
  sorry

end Lisa_flight_time_l175_175482


namespace sum_of_roots_is_zero_l175_175621

variable {R : Type*} [LinearOrderedField R]

-- Define the function f : R -> R and its properties
variable (f : R → R)
variable (even_f : ∀ x, f x = f (-x))
variable (roots_f : Finset R)
variable (roots_f_four : roots_f.card = 4)
variable (roots_f_set : ∀ x, x ∈ roots_f → f x = 0)

theorem sum_of_roots_is_zero : (roots_f.sum id) = 0 := 
sorry

end sum_of_roots_is_zero_l175_175621


namespace global_chess_tournament_total_games_global_chess_tournament_player_wins_l175_175296

theorem global_chess_tournament_total_games (num_players : ℕ) (h200 : num_players = 200) :
  (num_players * (num_players - 1)) / 2 = 19900 := by
  sorry

theorem global_chess_tournament_player_wins (num_players losses : ℕ) 
  (h200 : num_players = 200) (h30 : losses = 30) :
  (num_players - 1) - losses = 169 := by
  sorry

end global_chess_tournament_total_games_global_chess_tournament_player_wins_l175_175296


namespace sum_of_coefficients_l175_175044

theorem sum_of_coefficients (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℝ) :
  (5 * 1 - 2)^6 = b_6 * 1^6 + b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0
  → b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 = 729 := by
  sorry

end sum_of_coefficients_l175_175044


namespace boxes_needed_for_loose_crayons_l175_175420

-- Definitions based on conditions
def boxes_francine : ℕ := 5
def loose_crayons_francine : ℕ := 5
def loose_crayons_friend : ℕ := 27
def total_crayons_francine : ℕ := 85
def total_boxes_needed : ℕ := 2

-- The theorem to prove
theorem boxes_needed_for_loose_crayons 
  (hf : total_crayons_francine = boxes_francine * 16 + loose_crayons_francine)
  (htotal_loose : loose_crayons_francine + loose_crayons_friend = 32)
  (hboxes : boxes_francine = 5) : 
  total_boxes_needed = 2 :=
sorry

end boxes_needed_for_loose_crayons_l175_175420


namespace phi_value_for_unique_symmetry_center_l175_175283

theorem phi_value_for_unique_symmetry_center :
  ∃ (φ : ℝ), (0 < φ ∧ φ < π / 2) ∧
  (φ = π / 12 ∨ φ = π / 6 ∨ φ = π / 3 ∨ φ = 5 * π / 12) ∧
  ((∃ x : ℝ, 2 * x + φ = π ∧ π / 6 < x ∧ x < π / 3) ↔ φ = 5 * π / 12) :=
  sorry

end phi_value_for_unique_symmetry_center_l175_175283


namespace a_pow_10_plus_b_pow_10_l175_175802

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l175_175802


namespace transform_negation_l175_175722

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l175_175722


namespace jake_delay_l175_175728

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end jake_delay_l175_175728


namespace number_of_ways_to_put_balls_in_boxes_l175_175769

noncomputable def count_ways_to_put_balls_in_boxes : ℕ :=
  ∑ (x : Fin 5) in Finset.powerset (Finset.range 4), if (x.sum = 5) then 1 else 0

theorem number_of_ways_to_put_balls_in_boxes : count_ways_to_put_balls_in_boxes = 68 := 
by 
-- sorry placeholder for the actual proof
sorry

end number_of_ways_to_put_balls_in_boxes_l175_175769


namespace n_minus_m_eq_zero_l175_175896

-- Definitions based on the conditions
def m : ℝ := sorry
def n : ℝ := sorry
def i := Complex.I
def condition : Prop := m + i = (1 + 2 * i) - n * i

-- The theorem stating the equivalence proof problem
theorem n_minus_m_eq_zero (h : condition) : n - m = 0 :=
sorry

end n_minus_m_eq_zero_l175_175896


namespace min_score_needed_l175_175654

/-- 
Given the list of scores and the targeted increase in the average score,
ascertain that the minimum score required on the next test to achieve the
new average is 110.
 -/
theorem min_score_needed 
  (scores : List ℝ) 
  (target_increase : ℝ) 
  (new_score : ℝ) 
  (total_scores : ℝ)
  (current_average : ℝ) 
  (target_average : ℝ) 
  (needed_score : ℝ) :
  (total_scores = 86 + 92 + 75 + 68 + 88 + 84) ∧
  (current_average = total_scores / 6) ∧
  (target_average = current_average + target_increase) ∧
  (new_score = total_scores + needed_score) ∧
  (target_average = new_score / 7) ->
  needed_score = 110 :=
by
  sorry

end min_score_needed_l175_175654


namespace rate_per_sq_meter_l175_175930

theorem rate_per_sq_meter
  (length : Float := 9)
  (width : Float := 4.75)
  (total_cost : Float := 38475)
  : (total_cost / (length * width)) = 900 := 
by
  sorry

end rate_per_sq_meter_l175_175930


namespace total_weight_of_load_l175_175958

def weight_of_crate : ℕ := 4
def weight_of_carton : ℕ := 3
def number_of_crates : ℕ := 12
def number_of_cartons : ℕ := 16

theorem total_weight_of_load :
  number_of_crates * weight_of_crate + number_of_cartons * weight_of_carton = 96 :=
by sorry

end total_weight_of_load_l175_175958


namespace quartic_poly_roots_l175_175016

noncomputable def roots_polynomial : List ℝ := [
  (1 + Real.sqrt 5) / 2,
  (1 - Real.sqrt 5) / 2,
  (3 + Real.sqrt 13) / 6,
  (3 - Real.sqrt 13) / 6
]

theorem quartic_poly_roots :
  ∀ x : ℝ, x ∈ roots_polynomial ↔ 3*x^4 - 4*x^3 - 5*x^2 - 4*x + 3 = 0 :=
by sorry

end quartic_poly_roots_l175_175016


namespace unique_arrangements_of_TOOTH_l175_175136

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem unique_arrangements_of_TOOTH : 
  let word := "TOOTH" in
  let n := 5 in
  let t_count := 3 in
  let o_count := 2 in
  n.factorial / (t_count.factorial * o_count.factorial) = 10 :=
sorry

end unique_arrangements_of_TOOTH_l175_175136


namespace find_t_l175_175865

-- Define the logarithm base 3 function
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Given Condition
def condition (t : ℝ) : Prop := 4 * log_base_3 t = log_base_3 (4 * t) + 2

-- Theorem stating if the given condition holds, then t must be 6
theorem find_t (t : ℝ) (ht : condition t) : t = 6 := 
by
  sorry

end find_t_l175_175865


namespace polynomial_abc_value_l175_175449

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l175_175449


namespace arithmetic_sequence_fifth_term_l175_175882

theorem arithmetic_sequence_fifth_term (x : ℝ) (a₂ : ℝ := x) (a₃ : ℝ := 3) 
    (a₁ : ℝ := -1) (h₁ : a₂ = a₁ + (1*(x + 1))) (h₂ : a₃ = a₁ + 2*(x + 1)) : 
    a₁ + 4*(a₃ - a₂ + 1) = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l175_175882


namespace sum_of_ages_l175_175084

variable {P M Mo : ℕ}

-- Conditions
axiom ratio1 : 3 * M = 5 * P
axiom ratio2 : 3 * Mo = 5 * M
axiom age_difference : Mo - P = 80

-- Statement that needs to be proved
theorem sum_of_ages : P + M + Mo = 245 := by
  sorry

end sum_of_ages_l175_175084


namespace base_conversion_and_addition_l175_175985

theorem base_conversion_and_addition :
  let n1 := 2 * (8:ℕ)^2 + 4 * 8^1 + 3 * 8^0
  let d1 := 1 * 4^1 + 3 * 4^0
  let n2 := 2 * 7^2 + 0 * 7^1 + 4 * 7^0
  let d2 := 2 * 5^1 + 3 * 5^0
  n1 / d1 + n2 / d2 = 31 + 51 / 91 := by
  sorry

end base_conversion_and_addition_l175_175985


namespace roots_greater_than_one_implies_range_l175_175154

theorem roots_greater_than_one_implies_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + a = 0 → x > 1) → 3 < a ∧ a ≤ 4 :=
by
  sorry

end roots_greater_than_one_implies_range_l175_175154


namespace factory_ill_days_l175_175174

theorem factory_ill_days
  (average_first_25_days : ℝ)
  (total_days : ℝ)
  (overall_average : ℝ)
  (ill_days_average : ℝ)
  (production_first_25_days_total : ℝ)
  (production_ill_days_total : ℝ)
  (x : ℝ) :
  average_first_25_days = 50 →
  total_days = 25 + x →
  overall_average = 48 →
  ill_days_average = 38 →
  production_first_25_days_total = 25 * 50 →
  production_ill_days_total = x * 38 →
  (25 * 50 + x * 38 = (25 + x) * 48) →
  x = 5 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end factory_ill_days_l175_175174


namespace f_of_f_of_f_of_3_l175_175735

def f (x : ℕ) : ℕ := 
  if x > 9 then x - 1 
  else x ^ 3

theorem f_of_f_of_f_of_3 : f (f (f 3)) = 25 :=
by sorry

end f_of_f_of_f_of_3_l175_175735


namespace unit_digit_product_7858_1086_4582_9783_l175_175219

-- Define the unit digits of the given numbers
def unit_digit_7858 : ℕ := 8
def unit_digit_1086 : ℕ := 6
def unit_digit_4582 : ℕ := 2
def unit_digit_9783 : ℕ := 3

-- Define a function to calculate the unit digit of a product of two numbers based on their unit digits
def unit_digit_product (a b : ℕ) : ℕ :=
  (a * b) % 10

-- The theorem that states the unit digit of the product of the numbers is 4
theorem unit_digit_product_7858_1086_4582_9783 :
  unit_digit_product (unit_digit_product (unit_digit_product unit_digit_7858 unit_digit_1086) unit_digit_4582) unit_digit_9783 = 4 :=
  by
  sorry

end unit_digit_product_7858_1086_4582_9783_l175_175219


namespace fraction_product_equals_l175_175248

def frac1 := 7 / 4
def frac2 := 8 / 14
def frac3 := 9 / 6
def frac4 := 10 / 25
def frac5 := 28 / 21
def frac6 := 15 / 45
def frac7 := 32 / 16
def frac8 := 50 / 100

theorem fraction_product_equals : 
  (frac1 * frac2 * frac3 * frac4 * frac5 * frac6 * frac7 * frac8) = (4 / 5) := 
by
  sorry

end fraction_product_equals_l175_175248


namespace no_real_quadruples_solutions_l175_175015

theorem no_real_quadruples_solutions :
  ¬ ∃ (a b c d : ℝ),
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := 
sorry

end no_real_quadruples_solutions_l175_175015


namespace initial_blue_balls_l175_175224

-- Define the initial conditions
variables (B : ℕ) (total_balls : ℕ := 15) (removed_blue_balls : ℕ := 3)
variable (prob_after_removal : ℚ := 1 / 3)
variable (remaining_balls : ℕ := total_balls - removed_blue_balls)
variable (remaining_blue_balls : ℕ := B - removed_blue_balls)

-- State the theorem
theorem initial_blue_balls : 
  remaining_balls = 12 → remaining_blue_balls = remaining_balls * prob_after_removal → B = 7 :=
by
  intros h1 h2
  sorry

end initial_blue_balls_l175_175224


namespace distributive_property_example_l175_175815

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = (3/4) * (-36) + (7/12) * (-36) - (5/9) * (-36) :=
by
  sorry

end distributive_property_example_l175_175815


namespace central_angle_relation_l175_175292

theorem central_angle_relation
  (R L : ℝ)
  (α : ℝ)
  (r l β : ℝ)
  (h1 : r = 0.5 * R)
  (h2 : l = 1.5 * L)
  (h3 : L = R * α)
  (h4 : l = r * β) : 
  β = 3 * α :=
by
  sorry

end central_angle_relation_l175_175292


namespace num_ways_distribute_balls_l175_175771

theorem num_ways_distribute_balls : ∃ (n : ℕ), n = 60 ∧ (∃ (balls boxes : ℕ), balls = 5 ∧ boxes = 4 ∧ (compositions.balls_in_boxes balls boxes = n)) := 
begin
    sorry
end

end num_ways_distribute_balls_l175_175771


namespace find_common_difference_l175_175460

noncomputable def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 4 = 7 ∧ a 3 + a 6 = 16

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ) (h : common_difference a d) : d = 2 :=
by
  sorry

end find_common_difference_l175_175460


namespace area_of_triangle_A2B2C2_l175_175462

noncomputable def area_DA1B1 : ℝ := 15 / 4
noncomputable def area_DA1C1 : ℝ := 10
noncomputable def area_DB1C1 : ℝ := 6
noncomputable def area_DA2B2 : ℝ := 40
noncomputable def area_DA2C2 : ℝ := 30
noncomputable def area_DB2C2 : ℝ := 50

theorem area_of_triangle_A2B2C2 : ∃ area : ℝ, 
  area = (50 * Real.sqrt 2) ∧ 
  (area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 ∧ 
  area_DA2C2 = 30 ∧ 
  area_DB2C2 = 50) := 
by
  sorry

end area_of_triangle_A2B2C2_l175_175462


namespace speed_increase_percentage_l175_175472

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end speed_increase_percentage_l175_175472


namespace sam_gave_2_puppies_l175_175192

theorem sam_gave_2_puppies (original_puppies given_puppies remaining_puppies : ℕ) 
  (h1 : original_puppies = 6) (h2 : remaining_puppies = 4) :
  given_puppies = original_puppies - remaining_puppies := by 
  sorry

end sam_gave_2_puppies_l175_175192


namespace overall_profit_refrigerator_mobile_phone_l175_175495

theorem overall_profit_refrigerator_mobile_phone
  (purchase_price_refrigerator : ℕ)
  (purchase_price_mobile_phone : ℕ)
  (loss_percentage_refrigerator : ℕ)
  (profit_percentage_mobile_phone : ℕ)
  (selling_price_refrigerator : ℕ)
  (selling_price_mobile_phone : ℕ)
  (total_cost_price : ℕ)
  (total_selling_price : ℕ)
  (overall_profit : ℕ) :
  purchase_price_refrigerator = 15000 →
  purchase_price_mobile_phone = 8000 →
  loss_percentage_refrigerator = 4 →
  profit_percentage_mobile_phone = 10 →
  selling_price_refrigerator = purchase_price_refrigerator - (purchase_price_refrigerator * loss_percentage_refrigerator / 100) →
  selling_price_mobile_phone = purchase_price_mobile_phone + (purchase_price_mobile_phone * profit_percentage_mobile_phone / 100) →
  total_cost_price = purchase_price_refrigerator + purchase_price_mobile_phone →
  total_selling_price = selling_price_refrigerator + selling_price_mobile_phone →
  overall_profit = total_selling_price - total_cost_price →
  overall_profit = 200 :=
  by sorry

end overall_profit_refrigerator_mobile_phone_l175_175495


namespace complete_the_square_l175_175366

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x - 10 = 0

-- State the proof problem
theorem complete_the_square (x : ℝ) (h : quadratic_eq x) : (x - 3)^2 = 19 :=
by 
  -- Skip the proof using sorry
  sorry

end complete_the_square_l175_175366


namespace johns_percentage_increase_l175_175551

def original_amount : ℕ := 60
def new_amount : ℕ := 84

def percentage_increase (original new : ℕ) := ((new - original : ℕ) * 100) / original 

theorem johns_percentage_increase : percentage_increase original_amount new_amount = 40 :=
by
  sorry

end johns_percentage_increase_l175_175551


namespace rook_placement_non_attacking_l175_175461

theorem rook_placement_non_attacking (n : ℕ) (w b : ℕ) : 
  w = 8 * 8 ∧ b = (8 * (8 - 1) + (8 - 1) * (8 - 1)) → 
  w * b = 3136 :=
by 
  intro h.
  cases h with hw hb.
  sorry

end rook_placement_non_attacking_l175_175461


namespace new_foreign_students_l175_175586

theorem new_foreign_students 
  (total_students : ℕ)
  (percent_foreign : ℕ)
  (foreign_students_next_sem : ℕ)
  (current_foreign_students : ℕ := total_students * percent_foreign / 100) : 
  total_students = 1800 → 
  percent_foreign = 30 → 
  foreign_students_next_sem = 740 → 
  foreign_students_next_sem - current_foreign_students = 200 :=
by
  intros
  sorry

end new_foreign_students_l175_175586


namespace dragon_boat_festival_problem_l175_175501

theorem dragon_boat_festival_problem :
  ∃ (x : ℝ), 
    let cp := 22 in
    let sp := 38 in
    let q := 160 in
    let dp := 3640 in
    let profit := (sp - x - cp) in
    let new_q := q + (x / 3) * 120 in
    profit * new_q = dp ∧ 
    sp - x = 29 :=
begin
  sorry
end

end dragon_boat_festival_problem_l175_175501


namespace triangle_inequality_theorem_l175_175545

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l175_175545


namespace find_beta_plus_two_alpha_is_pi_l175_175273

noncomputable def find_beta_plus_two_alpha (α β : ℝ) : ℝ :=
if h : α ∈ (0 : ℝ, Real.pi) ∧ β ∈ (0 : ℝ, Real.pi) ∧ (Real.cos α + Real.cos β - Real.cos (α + β) = 3 / 2) then
  2 * α + β
else
  0

theorem find_beta_plus_two_alpha_is_pi (α β : ℝ) (hα : α ∈ (0 : ℝ, Real.pi))
  (hβ : β ∈ (0 : ℝ, Real.pi)) (hcos : Real.cos α + Real.cos β - Real.cos (α + β) = 3 / 2) :
  find_beta_plus_two_alpha α β = Real.pi :=
by
  rw [find_beta_plus_two_alpha, dif_pos]
  swap
  use ⟨hα, hβ, hcos⟩
  sorry

end find_beta_plus_two_alpha_is_pi_l175_175273


namespace solve_problem_l175_175289

theorem solve_problem :
  ∃ a b c d e f : ℤ,
  (208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f) ∧
  (0 ≤ a ∧ a ≤ 7) ∧ (0 ≤ b ∧ b ≤ 7) ∧ (0 ≤ c ∧ c ≤ 7) ∧
  (0 ≤ d ∧ d ≤ 7) ∧ (0 ≤ e ∧ e ≤ 7) ∧ (0 ≤ f ∧ f ≤ 7) ∧
  (a * b * c + d * e * f = 72) :=
by
  sorry

end solve_problem_l175_175289


namespace intersection_of_M_and_N_l175_175434

noncomputable def M : Set ℝ := {x | x - 2 > 0}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {x | x > 2} :=
sorry

end intersection_of_M_and_N_l175_175434


namespace total_number_of_squares_l175_175491

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l175_175491


namespace more_pie_eaten_l175_175727

theorem more_pie_eaten (e f : ℝ) (h1 : e = 0.67) (h2 : f = 0.33) : e - f = 0.34 :=
by sorry

end more_pie_eaten_l175_175727


namespace length_of_AB_l175_175238

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def slope_of_line : ℝ := Real.tan (Real.pi / 6)

-- Equation of the line in point-slope form
noncomputable def line_eq (x : ℝ) : ℝ :=
  (slope_of_line * x) + 1

-- Intersection points of the line with the parabola y = (1/4)x^2
noncomputable def parabola_eq (x : ℝ) : ℝ :=
  (1/4) * x ^ 2

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A.2 = parabola_eq A.1) ∧
    (B.2 = parabola_eq B.1) ∧ 
    (A.2 = line_eq A.1) ∧
    (B.2 = line_eq B.1) ∧
    ((((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1 / 2)) = 16 / 3) :=
by
  sorry

end length_of_AB_l175_175238


namespace solve_system_l175_175197

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y - z = 4 ∧ x^2 - y^2 + z^2 = -4 ∧ xyz = 6) ↔ 
    (x, y, z) = (2, 3, 1) ∨ (x, y, z) = (-1, 3, -2) :=
by
  sorry

end solve_system_l175_175197


namespace probability_both_selected_l175_175225

def P_X : ℚ := 1 / 3
def P_Y : ℚ := 2 / 7

theorem probability_both_selected : P_X * P_Y = 2 / 21 :=
by
  sorry

end probability_both_selected_l175_175225


namespace tens_digit_2015_pow_2016_minus_2017_l175_175698

theorem tens_digit_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 = 8 := 
sorry

end tens_digit_2015_pow_2016_minus_2017_l175_175698


namespace parallel_lines_m_l175_175159

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 3 * m * x + (m + 2) * y + 1 = 0) ∧
  (∀ (x y : ℝ), (m - 2) * x + (m + 2) * y + 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (3 * m) / (m + 2) = (m - 2) / (m + 2)) →
  (m = -1 ∨ m = -2) :=
sorry

end parallel_lines_m_l175_175159


namespace clothing_store_gross_profit_l175_175235

theorem clothing_store_gross_profit :
  ∃ S : ℝ, S = 81 + 0.25 * S ∧
  ∃ new_price : ℝ,
    new_price = S - 0.20 * S ∧
    ∃ profit : ℝ,
      profit = new_price - 81 ∧
      profit = 5.40 :=
by
  sorry

end clothing_store_gross_profit_l175_175235


namespace milk_rate_proof_l175_175354

theorem milk_rate_proof
  (initial_milk : ℕ := 30000)
  (time_pumped_out : ℕ := 4)
  (rate_pumped_out : ℕ := 2880)
  (time_adding_milk : ℕ := 7)
  (final_milk : ℕ := 28980) :
  ((final_milk - (initial_milk - time_pumped_out * rate_pumped_out)) / time_adding_milk = 1500) :=
by {
  sorry
}

end milk_rate_proof_l175_175354


namespace problem_product_xyzw_l175_175168

theorem problem_product_xyzw
    (x y z w : ℝ)
    (h1 : x + 1 / y = 1)
    (h2 : y + 1 / z + w = 1)
    (h3 : w = 2) :
    xyzw = -2 * y^2 + 2 * y :=
by
    sorry

end problem_product_xyzw_l175_175168


namespace triangle_B_angle_range_triangle_perimeter_l175_175455

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {R : ℝ}

-- Conditions
def triangle_sin_condition (A B C : ℝ) : Prop :=
  sin A + sin C = 2 * sin B

def geometric_sequence_condition (a b c : ℝ) : Prop :=
  9 * b * b = 10 * a * c

def circumcircle_radius_condition (R : ℝ) : Prop :=
  R = 3

-- Problem Statement
theorem triangle_B_angle_range 
  (h1 : triangle_sin_condition A B C)
  : B > 0 ∧ B ≤ π / 3 :=
sorry

theorem triangle_perimeter 
  (h1 : triangle_sin_condition A B C)
  (h2 : geometric_sequence_condition a b c)
  (h3 : circumcircle_radius_condition R)
  : a + b + c = 6 * sqrt 5 :=
sorry

end triangle_B_angle_range_triangle_perimeter_l175_175455


namespace percentage_of_number_l175_175569

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l175_175569


namespace base_7_digits_956_l175_175894

theorem base_7_digits_956 : ∃ n : ℕ, ∀ k : ℕ, 956 < 7^k → n = k ∧ 956 ≥ 7^(k-1) := sorry

end base_7_digits_956_l175_175894


namespace handshakes_l175_175140

open Nat

theorem handshakes : ∃ x : ℕ, 4 + 3 + 2 + 1 + x = 10 ∧ x = 2 :=
by
  existsi 2
  simp
  sorry

end handshakes_l175_175140


namespace monthly_salary_l175_175808

variable {S : ℝ}

-- Conditions based on the problem description
def spends_on_food (S : ℝ) : ℝ := 0.40 * S
def spends_on_house_rent (S : ℝ) : ℝ := 0.20 * S
def spends_on_entertainment (S : ℝ) : ℝ := 0.10 * S
def spends_on_conveyance (S : ℝ) : ℝ := 0.10 * S
def savings (S : ℝ) : ℝ := 0.20 * S

-- Given savings
def savings_amount : ℝ := 2500

-- The proof statement for the monthly salary
theorem monthly_salary (h : savings S = savings_amount) : S = 12500 := by
  sorry

end monthly_salary_l175_175808


namespace range_of_a_l175_175032

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x / x) - a * (x ^ 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f a x1 / x2) - (f a x2 / x1) < 0) ↔ (a ≤ Real.exp 2 / 12) := by
  sorry

end range_of_a_l175_175032


namespace cost_prices_l175_175853

variable {C1 C2 : ℝ}

theorem cost_prices (h1 : 0.30 * C1 - 0.15 * C1 = 120) (h2 : 0.25 * C2 - 0.10 * C2 = 150) :
  C1 = 800 ∧ C2 = 1000 := 
by
  sorry

end cost_prices_l175_175853


namespace vanessa_video_files_initial_l175_175526

theorem vanessa_video_files_initial (m v r d t : ℕ) (h1 : m = 13) (h2 : r = 33) (h3 : d = 10) (h4 : t = r + d) (h5 : t = m + v) : v = 30 :=
by
  sorry

end vanessa_video_files_initial_l175_175526


namespace required_folders_l175_175572

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_count : ℕ := 24
def total_cost : ℝ := 30

theorem required_folders : ∃ (folders : ℕ), folders = 20 ∧ 
  (pencil_count * pencil_cost + folders * folder_cost = total_cost) :=
sorry

end required_folders_l175_175572


namespace triangle_solutions_l175_175747

theorem triangle_solutions :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 7.012 ∧
  c - b = 1.753 ∧
  B = 38 + 12/60 + 48/3600 ∧
  A = 81 + 47/60 + 12.5/3600 ∧
  C = 60 ∧
  b = 4.3825 ∧
  c = 6.1355 :=
sorry -- Proof goes here

end triangle_solutions_l175_175747


namespace all_round_trips_miss_capital_same_cost_l175_175824

open Set

variable {City : Type} [Inhabited City]
variable {f : City → City → ℝ}
variable (capital : City)
variable (round_trip_cost : List City → ℝ)

-- The conditions
axiom flight_cost_symmetric (A B : City) : f A B = f B A
axiom equal_round_trip_cost (R1 R2 : List City) :
  (∀ (city : City), city ∈ R1 ↔ city ∈ R2) → 
  round_trip_cost R1 = round_trip_cost R2

noncomputable def constant_trip_cost := 
  ∀ (cities1 cities2 : List City),
     (∀ (city : City), city ∈ cities1 ↔ city ∈ cities2) →
     ¬(capital ∈ cities1 ∨ capital ∈ cities2) →
     round_trip_cost cities1 = round_trip_cost cities2

-- Goal to prove
theorem all_round_trips_miss_capital_same_cost : constant_trip_cost capital round_trip_cost := 
  sorry

end all_round_trips_miss_capital_same_cost_l175_175824


namespace sum_of_n_for_3n_minus_8_eq_5_l175_175531

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l175_175531


namespace f_zero_eq_one_positive_for_all_x_l175_175187

variables {R : Type*} [LinearOrderedField R] (f : R → R)

-- Conditions
axiom domain (x : R) : true -- This translates that f has domain (-∞, ∞)
axiom non_constant (x1 x2 : R) (h : x1 ≠ x2) : f x1 ≠ f x2
axiom functional_eq (x y : R) : f (x + y) = f x * f y

-- Questions
theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem positive_for_all_x (x : R) : f x > 0 :=
sorry

end f_zero_eq_one_positive_for_all_x_l175_175187


namespace minimum_value_of_y_exists_l175_175257

theorem minimum_value_of_y_exists :
  ∃ (y : ℝ), (∀ (x : ℝ), (y + x) = (y - x)^2 + 3 * (y - x) + 3) ∧ y = -1/2 :=
by sorry

end minimum_value_of_y_exists_l175_175257


namespace find_vector_BC_l175_175879

structure Point2D where
  x : ℝ
  y : ℝ

def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨3, 2⟩
def AC : Point2D := ⟨-4, -3⟩

def vector_add (p1 p2 : Point2D) : Point2D := ⟨p1.x + p2.x, p1.y + p2.y⟩
def vector_sub (p1 p2 : Point2D) : Point2D := ⟨p1.x - p2.x, p1.y - p2.y⟩

def C : Point2D := vector_add A AC
def BC : Point2D := vector_sub C B

theorem find_vector_BC : BC = ⟨-7, -4⟩ := by
  sorry

end find_vector_BC_l175_175879


namespace sqrt_expression_l175_175252

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l175_175252


namespace cost_of_article_l175_175288

theorem cost_of_article (C G1 G2 : ℝ) (h1 : G1 = 380 - C) (h2 : G2 = 450 - C) (h3 : G2 = 1.10 * G1) : 
  C = 320 :=
by
  sorry

end cost_of_article_l175_175288


namespace gcd_three_digit_palindromes_l175_175359

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l175_175359


namespace max_number_of_9_letter_palindromes_l175_175265

theorem max_number_of_9_letter_palindromes : 26^5 = 11881376 :=
by sorry

end max_number_of_9_letter_palindromes_l175_175265


namespace original_savings_eq_920_l175_175912

variable (S : ℝ) -- Define S as a real number representing Linda's savings
variable (h1 : S * (1 / 4) = 230) -- Given condition

theorem original_savings_eq_920 :
  S = 920 :=
by
  sorry

end original_savings_eq_920_l175_175912


namespace smaller_two_digit_product_l175_175510

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l175_175510


namespace largest_of_five_consecutive_integers_with_product_15120_l175_175418

theorem largest_of_five_consecutive_integers_with_product_15120 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 12 :=
begin
  sorry
end

end largest_of_five_consecutive_integers_with_product_15120_l175_175418


namespace find_constant_a_l175_175448

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (Real.exp x - 1)

theorem find_constant_a (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 := 
by
  sorry

end find_constant_a_l175_175448


namespace value_of_x_l175_175447

-- Define the conditions extracted from problem (a)
def condition1 (x : ℝ) : Prop := x^2 - 1 = 0
def condition2 (x : ℝ) : Prop := x - 1 ≠ 0

-- The statement to be proved
theorem value_of_x : ∀ x : ℝ, condition1 x → condition2 x → x = -1 :=
by
  intros x h1 h2
  sorry

end value_of_x_l175_175447


namespace reflect_across_x_axis_l175_175903

-- Definitions for the problem conditions
def initial_point : ℝ × ℝ := (-2, 1)
def reflected_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The statement to be proved
theorem reflect_across_x_axis :
  reflected_point initial_point = (-2, -1) :=
  sorry

end reflect_across_x_axis_l175_175903


namespace min_value_of_f_l175_175681

noncomputable def f (x : ℝ) : ℝ :=
  1 / (Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_l175_175681


namespace new_person_weight_l175_175812

theorem new_person_weight :
  (8 * 2.5 + 75 = 95) :=
by sorry

end new_person_weight_l175_175812


namespace count_even_divisors_8_l175_175439

theorem count_even_divisors_8! :
  ∃ (even_divisors total : ℕ),
    even_divisors = 84 ∧
    total = 56 :=
by
  /-
    To formulate the problem in Lean:
    We need to establish two main facts:
    1. The count of even divisors of 8! is 84.
    2. The count of those even divisors that are multiples of both 2 and 3 is 56.
  -/
  sorry

end count_even_divisors_8_l175_175439


namespace find_smallest_b_l175_175018

theorem find_smallest_b :
  ∃ b : ℕ, 
    (∀ r s : ℤ, r * s = 3960 → r + s ≠ b ∨ r + s > 0) ∧ 
    (∀ r s : ℤ, r * s = 3960 → (r + s < b → r + s ≤ 0)) ∧ 
    b = 126 :=
by
  sorry

end find_smallest_b_l175_175018


namespace find_daily_wage_of_c_l175_175834

def dailyWagesInRatio (a b c : ℕ) : Prop :=
  4 * a = 3 * b ∧ 5 * a = 3 * c

def totalEarnings (a b c : ℕ) (total : ℕ) : Prop :=
  6 * a + 9 * b + 4 * c = total

theorem find_daily_wage_of_c (a b c : ℕ) (total : ℕ) 
  (h1 : dailyWagesInRatio a b c) 
  (h2 : totalEarnings a b c total) 
  (h3 : total = 1406) : 
  c = 95 :=
by
  -- We assume the conditions and solve the required proof.
  sorry

end find_daily_wage_of_c_l175_175834


namespace xy_square_sum_l175_175445

theorem xy_square_sum (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 :=
by
  sorry

end xy_square_sum_l175_175445


namespace solve_missing_figure_l175_175704

theorem solve_missing_figure (x : ℝ) (h : 0.25/100 * x = 0.04) : x = 16 :=
by
  sorry

end solve_missing_figure_l175_175704


namespace calculate_fraction_l175_175544

variables (n_bl: ℕ) (deg_warm: ℕ) (total_deg: ℕ) (total_bl: ℕ)

def blanket_fraction_added := total_deg / deg_warm

theorem calculate_fraction (h1: deg_warm = 3) (h2: total_deg = 21) (h3: total_bl = 14) :
  (blanket_fraction_added total_deg deg_warm) / total_bl = 1 / 2 :=
by {
  sorry
}

end calculate_fraction_l175_175544


namespace transform_negation_l175_175719

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l175_175719


namespace poster_width_l175_175123
   
   theorem poster_width (h : ℕ) (A : ℕ) (w : ℕ) (h_eq : h = 7) (A_eq : A = 28) (area_eq : w * h = A) : w = 4 :=
   by
   sorry
   
end poster_width_l175_175123


namespace forty_percent_of_number_l175_175949

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 0.40 * N = 120 :=
sorry

end forty_percent_of_number_l175_175949


namespace arrival_probability_l175_175595

-- Definitions for arrival times and conditions
variables {t1 t2 t3 : ℝ} -- The arrival times
variables (ht1 : t1 < t2) -- Given condition: t1 < t2

-- Assume arrival times are uniformly random and independent
-- Note: Uniform distribution over a session duration can be conceptualized but is not explicitly defined here

noncomputable def prob_t1_lt_t3_given_t1_lt_t2 : ℝ := 
real.to_nnreal (2 / 3)

-- Theorem statement: prove the probability calculation
theorem arrival_probability : prob_t1_lt_t3_given_t1_lt_t2 = 2 / 3 := sorry

end arrival_probability_l175_175595


namespace livestock_min_count_l175_175566

/-- A livestock trader bought some horses at $344 each and some oxen at $265 each. 
The total cost of all the horses was $33 more than the total cost of all the oxen. 
Prove that the minimum number of horses and oxen he could have bought under these conditions 
are x = 36 and y = 25. -/
theorem livestock_min_count 
    (x y: ℤ) (horses_cost oxen_cost : ℤ) (price_diff : ℤ)
    (h_horses_cost : horses_cost = 344) (h_oxen_cost : oxen_cost = 265) (h_price_diff : price_diff = 33) 
    (h_eq: horses_cost * x = oxen_cost * y + price_diff): 
    (x = 36) ∧ (y = 25) :=
by
    sorry

end livestock_min_count_l175_175566


namespace sum_of_solutions_abs_eq_l175_175541

-- Define the condition on n
def abs_eq (n : ℝ) : Prop := abs (3 * n - 8) = 5

-- Statement we want to prove
theorem sum_of_solutions_abs_eq : ∑ n in { n | abs_eq n }, n = 16/3 :=
by
  sorry

end sum_of_solutions_abs_eq_l175_175541


namespace simplify_expression_l175_175333

variable (b : ℤ)

theorem simplify_expression :
  (3 * b + 6 - 6 * b) / 3 = -b + 2 :=
sorry

end simplify_expression_l175_175333


namespace range_of_a_l175_175428

variable (f : ℝ → ℝ)

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f y < f x

theorem range_of_a 
  (decreasing_f : is_decreasing f)
  (hfdef : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (2 * x - 3) < f (x - 2)) :
  ∃ a : ℝ, 1 < a ∧ a ≤ 2  :=
by 
  sorry

end range_of_a_l175_175428


namespace solve_quadratic_eq_l175_175498

theorem solve_quadratic_eq (x : ℝ) (h : x > 0) (eq : 4 * x^2 + 8 * x - 20 = 0) : 
  x = Real.sqrt 6 - 1 :=
sorry

end solve_quadratic_eq_l175_175498


namespace lcm_8_13_14_is_728_l175_175144

-- Define the numbers and their factorizations
def num1 := 8
def fact1 := 2 ^ 3

def num2 := 13  -- 13 is prime

def num3 := 14
def fact3 := 2 * 7

-- Define the function to calculate the LCM of three integers
def lcm (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- State the theorem to prove that the LCM of 8, 13, and 14 is 728
theorem lcm_8_13_14_is_728 : lcm num1 num2 num3 = 728 :=
by
  -- Prove the equality, skipping proof details with sorry
  sorry

end lcm_8_13_14_is_728_l175_175144


namespace number_of_chocolates_bought_l175_175898

theorem number_of_chocolates_bought (C S : ℝ) 
  (h1 : ∃ n : ℕ, n * C = 21 * S) 
  (h2 : (S - C) / C * 100 = 66.67) : 
  ∃ n : ℕ, n = 35 := 
by
  sorry

end number_of_chocolates_bought_l175_175898


namespace cleaning_time_if_anne_doubled_l175_175973

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l175_175973


namespace sophie_total_spend_l175_175325

def total_cost_with_discount_and_tax : ℝ :=
  let cupcakes_price := 5 * 2
  let doughnuts_price := 6 * 1
  let apple_pie_price := 4 * 2
  let cookies_price := 15 * 0.60
  let chocolate_bars_price := 8 * 1.50
  let soda_price := 12 * 1.20
  let gum_price := 3 * 0.80
  let chips_price := 10 * 1.10
  let total_before_discount := cupcakes_price + doughnuts_price + apple_pie_price + cookies_price + chocolate_bars_price + soda_price + gum_price + chips_price
  let discount := 0.10 * total_before_discount
  let subtotal_after_discount := total_before_discount - discount
  let sales_tax := 0.06 * subtotal_after_discount
  let total_cost := subtotal_after_discount + sales_tax
  total_cost

theorem sophie_total_spend :
  total_cost_with_discount_and_tax = 69.45 :=
sorry

end sophie_total_spend_l175_175325


namespace sequence_a_b_10_l175_175799

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l175_175799


namespace sum_of_solutions_of_absolute_value_l175_175528

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l175_175528


namespace binomial_coefficient_x5_l175_175415

theorem binomial_coefficient_x5 :
  let binomial_term (r : ℕ) : ℕ := Nat.choose 7 r * (21 - 4 * r)
  35 = binomial_term 4 :=
by
  sorry

end binomial_coefficient_x5_l175_175415


namespace problems_per_page_l175_175226

theorem problems_per_page (total_problems finished_problems pages_left problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : pages_left = 2)
  (h4 : total_problems - finished_problems = pages_left * problems_per_page) :
  problems_per_page = 7 :=
by
  sorry

end problems_per_page_l175_175226


namespace molecular_weight_of_compound_l175_175215

noncomputable def molecularWeight (Ca_wt : ℝ) (O_wt : ℝ) (H_wt : ℝ) (nCa : ℕ) (nO : ℕ) (nH : ℕ) : ℝ :=
  (nCa * Ca_wt) + (nO * O_wt) + (nH * H_wt)

theorem molecular_weight_of_compound :
  molecularWeight 40.08 15.999 1.008 1 2 2 = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l175_175215


namespace movie_duration_l175_175387

theorem movie_duration :
  let start_time := (13, 30)
  let end_time := (14, 50)
  let hours := end_time.1 - start_time.1
  let minutes := end_time.2 - start_time.2
  (if minutes < 0 then (hours - 1, minutes + 60) else (hours, minutes)) = (1, 20) := by
    sorry

end movie_duration_l175_175387


namespace total_marbles_l175_175564

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l175_175564


namespace q_evaluation_l175_175660

def q (x y : ℤ) : ℤ :=
if x ≥ 0 ∧ y ≤ 0 then x - y
else if x < 0 ∧ y > 0 then x + 3 * y
else 4 * x - 2 * y

theorem q_evaluation : q (q 2 (-3)) (q (-4) 1) = 6 :=
by
  sorry

end q_evaluation_l175_175660


namespace range_of_a_l175_175775

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → x^2 + 2 * a * x + 1 ≥ 0) ↔ a ≥ -1 := 
by
  sorry

end range_of_a_l175_175775


namespace x_squared_plus_y_squared_l175_175556

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := 
by
  sorry

end x_squared_plus_y_squared_l175_175556


namespace find_other_sides_of_triangle_l175_175616

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l175_175616


namespace ellipse_parameters_sum_l175_175680

theorem ellipse_parameters_sum 
  (h k a b : ℤ) 
  (h_def : h = 3) 
  (k_def : k = -5) 
  (a_def : a = 7) 
  (b_def : b = 2) : 
  h + k + a + b = 7 := 
by 
  -- definitions and sums will be handled by autogenerated proof
  sorry

end ellipse_parameters_sum_l175_175680


namespace expand_polynomials_eq_l175_175141

-- Define the polynomials P(z) and Q(z)
def P (z : ℝ) : ℝ := 3 * z^3 + 2 * z^2 - 4 * z + 1
def Q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the result polynomial R(z)
def R (z : ℝ) : ℝ := 12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2

-- State the theorem that proves P(z) * Q(z) = R(z)
theorem expand_polynomials_eq :
  ∀ (z : ℝ), (P z) * (Q z) = R z :=
by
  intros z
  sorry

end expand_polynomials_eq_l175_175141


namespace cost_price_of_watch_l175_175960

theorem cost_price_of_watch :
  ∃ (CP : ℝ), (CP * 1.07 = CP * 0.88 + 250) ∧ CP = 250 / 0.19 :=
sorry

end cost_price_of_watch_l175_175960


namespace result_prob_a_l175_175668

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l175_175668


namespace steven_has_72_shirts_l175_175328

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l175_175328


namespace cleaning_time_with_doubled_an_speed_l175_175966

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l175_175966


namespace minimum_value_correct_l175_175330

noncomputable def minimum_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_eq : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z + 1)^2 / (2 * x * y * z)

theorem minimum_value_correct {x y z : ℝ}
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 + y^2 + z^2 = 1) :
  minimum_value x y z h_pos h_eq = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_correct_l175_175330


namespace geom_seq_product_a2_a3_l175_175647

theorem geom_seq_product_a2_a3 :
  ∃ (a_n : ℕ → ℝ), (a_n 1 * a_n 4 = -3) ∧ (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1) ^ (n - 1)) → a_n 2 * a_n 3 = -3 :=
by
  sorry

end geom_seq_product_a2_a3_l175_175647


namespace min_straight_line_cuts_l175_175635

theorem min_straight_line_cuts (can_overlap : Prop) : 
  ∃ (cuts : ℕ), cuts = 4 ∧ 
  (∀ (square : ℕ), square = 3 →
   ∀ (unit : ℕ), unit = 1 → 
   ∀ (divided : Prop), divided = True → 
   (unit * unit) * 9 = (square * square)) :=
by
  sorry

end min_straight_line_cuts_l175_175635


namespace tom_paid_correct_amount_l175_175951

def quantity_of_apples : ℕ := 8
def rate_per_kg_apples : ℕ := 70
def quantity_of_mangoes : ℕ := 9
def rate_per_kg_mangoes : ℕ := 45

def cost_of_apples : ℕ := quantity_of_apples * rate_per_kg_apples
def cost_of_mangoes : ℕ := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid : ℕ := cost_of_apples + cost_of_mangoes

theorem tom_paid_correct_amount :
  total_amount_paid = 965 :=
sorry

end tom_paid_correct_amount_l175_175951


namespace balls_in_boxes_l175_175761

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l175_175761


namespace find_cost_of_books_l175_175772

theorem find_cost_of_books
  (C_L C_G1 C_G2 : ℝ)
  (h1 : C_L + C_G1 + C_G2 = 1080)
  (h2 : 0.9 * C_L = 1.15 * C_G1 + 1.25 * C_G2)
  (h3 : C_G1 + C_G2 = 1080 - C_L) :
  C_L = 784 :=
sorry

end find_cost_of_books_l175_175772


namespace cauchy_inequality_minimum_value_inequality_l175_175842

-- Part 1: Prove Cauchy Inequality
theorem cauchy_inequality (a b x y : ℝ) : 
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

-- Part 2: Find the minimum value under the given conditions
theorem minimum_value_inequality (x y : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x ≠ y ∨ x ≠ -y) : 
  ∃ m, m = (1 / (9 * x^2) + 9 / y^2) ∧ m = 50 / 9 :=
by
  sorry

end cauchy_inequality_minimum_value_inequality_l175_175842


namespace eccentricity_of_ellipse_l175_175152

variables {a b c e : ℝ}

-- Definition of geometric progression condition for the ellipse axes and focal length
def geometric_progression_condition (a b c : ℝ) : Prop :=
  (2 * b) ^ 2 = 2 * c * 2 * a

-- Eccentricity calculation
def eccentricity {a c : ℝ} (e : ℝ) : Prop :=
  e = (a^2 - c^2) / a^2

-- Theorem that states the eccentricity under the given condition
theorem eccentricity_of_ellipse (h : geometric_progression_condition a b c) : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_ellipse_l175_175152


namespace correlation_statements_l175_175687

variables {x y : ℝ}
variables (r : ℝ) (h1 : r > 0) (h2 : r = 1) (h3 : r = -1)

theorem correlation_statements :
  (r > 0 → (∀ x y, x > 0 → y > 0)) ∧
  (r = 1 ∨ r = -1 → (∀ x y, ∃ m b : ℝ, y = m * x + b)) :=
sorry

end correlation_statements_l175_175687


namespace angle_between_diagonal_and_base_l175_175243

theorem angle_between_diagonal_and_base 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sin (α / 2)) :=
sorry

end angle_between_diagonal_and_base_l175_175243


namespace percent_difference_l175_175549

variable (w e y z : ℝ)

-- Definitions based on the given conditions
def condition1 : Prop := w = 0.60 * e
def condition2 : Prop := e = 0.60 * y
def condition3 : Prop := z = 0.54 * y

-- Statement of the theorem to prove
theorem percent_difference (h1 : condition1 w e) (h2 : condition2 e y) (h3 : condition3 z y) : 
  (z - w) / w * 100 = 50 := 
by
  sorry

end percent_difference_l175_175549


namespace sum_of_n_values_l175_175537

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l175_175537


namespace original_number_of_turtles_l175_175121

-- Define the problem
theorem original_number_of_turtles (T : ℕ) (h1 : 17 = (T + 3 * T - 2) / 2) : T = 9 := by
  sorry

end original_number_of_turtles_l175_175121


namespace count_interesting_quadruples_l175_175982

def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d > b + c

def num_interesting_quadruples : ℕ :=
  (finset.range 11).card (λ quad : ℕ × ℕ × ℕ × ℕ, is_interesting_quadruple quad.1 quad.2.1 quad.2.2.1 quad.2.2.2)

theorem count_interesting_quadruples : num_interesting_quadruples = 80 :=
  sorry

end count_interesting_quadruples_l175_175982


namespace polygon_sides_eq_six_l175_175205

theorem polygon_sides_eq_six (n : ℕ) :
  ((n - 2) * 180 = 2 * 360) → n = 6 := by
  intro h
  have : (n - 2) * 180 = 720 := by exact h
  have : n - 2 = 4 := by linarith
  have : n = 6 := by linarith
  exact this

end polygon_sides_eq_six_l175_175205


namespace surface_area_of_sphere_with_diameter_4_l175_175207

theorem surface_area_of_sphere_with_diameter_4 :
    let diameter := 4
    let radius := diameter / 2
    let surface_area := 4 * Real.pi * radius^2
    surface_area = 16 * Real.pi :=
by
  -- Sorry is used in place of the actual proof.
  sorry

end surface_area_of_sphere_with_diameter_4_l175_175207


namespace bisection_min_calculations_l175_175943

theorem bisection_min_calculations 
  (a b : ℝ)
  (h_interval : a = 1.4 ∧ b = 1.5)
  (delta : ℝ)
  (h_delta : delta = 0.001) :
  ∃ n : ℕ, 0.1 / (2 ^ n) ≤ delta ∧ n = 7 :=
sorry

end bisection_min_calculations_l175_175943


namespace max_perimeter_triangle_l175_175578

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l175_175578


namespace simplify_expression_l175_175920

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 18 = 45 * w + 18 := by
  sorry

end simplify_expression_l175_175920


namespace johns_overall_profit_l175_175785

def cost_price_grinder : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_grinder : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10

noncomputable def loss_amount_grinder : ℝ := loss_percent_grinder * cost_price_grinder
noncomputable def selling_price_grinder : ℝ := cost_price_grinder - loss_amount_grinder

noncomputable def profit_amount_mobile : ℝ := profit_percent_mobile * cost_price_mobile
noncomputable def selling_price_mobile : ℝ := cost_price_mobile + profit_amount_mobile

noncomputable def total_cost_price : ℝ := cost_price_grinder + cost_price_mobile
noncomputable def total_selling_price : ℝ := selling_price_grinder + selling_price_mobile
noncomputable def overall_profit : ℝ := total_selling_price - total_cost_price

theorem johns_overall_profit :
  overall_profit = 50 := 
by
  sorry

end johns_overall_profit_l175_175785


namespace carrots_problem_l175_175841

def total_carrots (faye_picked : Nat) (mother_picked : Nat) : Nat :=
  faye_picked + mother_picked

def bad_carrots (total_carrots : Nat) (good_carrots : Nat) : Nat :=
  total_carrots - good_carrots

theorem carrots_problem (faye_picked : Nat) (mother_picked : Nat) (good_carrots : Nat) (bad_carrots : Nat) 
  (h1 : faye_picked = 23) 
  (h2 : mother_picked = 5)
  (h3 : good_carrots = 12) :
  bad_carrots = 16 := sorry

end carrots_problem_l175_175841


namespace find_mean_of_two_l175_175593

-- Define the set of numbers
def numbers : List ℕ := [1879, 1997, 2023, 2029, 2113, 2125]

-- Define the mean of the four selected numbers
def mean_of_four : ℕ := 2018

-- Define the sum of all numbers
def total_sum : ℕ := numbers.sum

-- Define the sum of the four numbers with a given mean
def sum_of_four : ℕ := 4 * mean_of_four

-- Define the sum of the remaining two numbers
def sum_of_two (total sum_of_four : ℕ) : ℕ := total - sum_of_four

-- Define the mean of the remaining two numbers
def mean_of_two (sum_two : ℕ) : ℕ := sum_two / 2

-- Define the condition theorem to be proven
theorem find_mean_of_two : mean_of_two (sum_of_two total_sum sum_of_four) = 2047 := 
by
  sorry

end find_mean_of_two_l175_175593


namespace maximize_garden_area_l175_175071

def optimal_dimensions_area : Prop :=
  let l := 100
  let w := 60
  let area := 6000
  (2 * l) + (2 * w) = 320 ∧ l >= 100 ∧ (l * w) = area

theorem maximize_garden_area : optimal_dimensions_area := by
  sorry

end maximize_garden_area_l175_175071


namespace num_people_price_item_equation_l175_175464

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l175_175464


namespace line_connecting_centers_l175_175134

-- Define the first circle equation
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x + 6*y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_eq (x y : ℝ) := 3*x - y - 9 = 0

-- Prove that the line connecting the centers of the circles has the given equation
theorem line_connecting_centers :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → line_eq x y := 
sorry

end line_connecting_centers_l175_175134


namespace no_primes_divisible_by_60_l175_175637

theorem no_primes_divisible_by_60 (p : ℕ) (prime_p : Nat.Prime p) : ¬ (60 ∣ p) :=
by
  sorry

end no_primes_divisible_by_60_l175_175637


namespace union_complement_eq_l175_175435

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {-1, 0, 3}

theorem union_complement_eq :
  A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by
  sorry

end union_complement_eq_l175_175435


namespace gcf_of_three_digit_palindromes_is_one_l175_175360

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l175_175360


namespace willy_crayons_difference_l175_175369

def willy : Int := 5092
def lucy : Int := 3971
def jake : Int := 2435

theorem willy_crayons_difference : willy - (lucy + jake) = -1314 := by
  sorry

end willy_crayons_difference_l175_175369


namespace card_draw_probability_l175_175101

-- Define a function to compute the probability of a sequence of draws
noncomputable def probability_of_event : Rat :=
  (4 / 52) * (4 / 51) * (1 / 50)

theorem card_draw_probability :
  probability_of_event = 4 / 33150 :=
by
  -- Proof goes here
  sorry

end card_draw_probability_l175_175101


namespace no_integer_pairs_satisfy_equation_l175_175740

def equation_satisfaction (m n : ℤ) : Prop :=
  m^3 + 3 * m^2 + 2 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ (m n : ℤ), equation_satisfaction m n :=
by
  sorry

end no_integer_pairs_satisfy_equation_l175_175740


namespace sum_of_decimals_l175_175379

theorem sum_of_decimals : (1 / 10) + (9 / 100) + (9 / 1000) + (7 / 10000) = 0.1997 := 
sorry

end sum_of_decimals_l175_175379


namespace derivative_at_pi_over_3_l175_175430

noncomputable def f (x : Real) : Real := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

theorem derivative_at_pi_over_3 : (deriv f) (π / 3) = -1 / 2 := 
by
  sorry

end derivative_at_pi_over_3_l175_175430


namespace largest_lcm_value_is_90_l175_175214

def lcm_vals (a b : ℕ) : ℕ := Nat.lcm a b

theorem largest_lcm_value_is_90 :
  max (lcm_vals 18 3)
      (max (lcm_vals 18 9)
           (max (lcm_vals 18 6)
                (max (lcm_vals 18 12)
                     (max (lcm_vals 18 15)
                          (lcm_vals 18 18))))) = 90 :=
by
  -- Use the fact that the calculations of LCMs are as follows:
  -- lcm(18, 3) = 18
  -- lcm(18, 9) = 18
  -- lcm(18, 6) = 18
  -- lcm(18, 12) = 36
  -- lcm(18, 15) = 90
  -- lcm(18, 18) = 18
  -- therefore, the largest value among these is 90
  sorry

end largest_lcm_value_is_90_l175_175214


namespace six_digit_palindromes_count_l175_175998

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l175_175998


namespace never_prime_l175_175004

theorem never_prime (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 105) := sorry

end never_prime_l175_175004


namespace kernels_popped_in_final_bag_l175_175073

/-- Parker wants to find out what the average percentage of kernels that pop in a bag is.
In the first bag he makes, 60 kernels pop and the bag has 75 kernels.
In the second bag, 42 kernels pop and there are 50 in the bag.
In the final bag, some kernels pop and the bag has 100 kernels.
The average percentage of kernels that pop in a bag is 82%.
How many kernels popped in the final bag?
We prove that given these conditions, the number of popped kernels in the final bag is 82.
-/
noncomputable def kernelsPoppedInFirstBag := 60
noncomputable def totalKernelsInFirstBag := 75
noncomputable def kernelsPoppedInSecondBag := 42
noncomputable def totalKernelsInSecondBag := 50
noncomputable def totalKernelsInFinalBag := 100
noncomputable def averagePoppedPercentage := 82

theorem kernels_popped_in_final_bag (x : ℕ) :
  (kernelsPoppedInFirstBag * 100 / totalKernelsInFirstBag +
   kernelsPoppedInSecondBag * 100 / totalKernelsInSecondBag +
   x * 100 / totalKernelsInFinalBag) / 3 = averagePoppedPercentage →
  x = 82 := 
by
  sorry

end kernels_popped_in_final_bag_l175_175073


namespace fraction_of_positive_number_l175_175382

theorem fraction_of_positive_number (x : ℝ) (f : ℝ) (h : x = 0.4166666666666667 ∧ f * x = (25/216) * (1/x)) : f = 2/3 :=
sorry

end fraction_of_positive_number_l175_175382


namespace reflect_parabola_x_axis_l175_175308

theorem reflect_parabola_x_axis (x : ℝ) (a b c : ℝ) :
  (∀ y : ℝ, y = x^2 + x - 2 → -y = x^2 + x - 2) →
  (∀ y : ℝ, -y = x^2 + x - 2 → y = -x^2 - x + 2) :=
by
  intros h₁ h₂
  intro y
  sorry

end reflect_parabola_x_axis_l175_175308


namespace max_quarters_l175_175496

-- Definitions stating the conditions
def total_money_in_dollars : ℝ := 4.80
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10

-- Theorem statement
theorem max_quarters (q : ℕ) (h1 : total_money_in_dollars = (q * value_of_quarter) + (2 * q * value_of_dime)) : q ≤ 10 :=
by {
  -- Injecting a placeholder to facilitate proof development
  sorry
}

end max_quarters_l175_175496


namespace actual_time_when_car_clock_shows_10PM_l175_175022

def car_clock_aligned (aligned_time wristwatch_time : ℕ) : Prop :=
  aligned_time = wristwatch_time

def car_clock_time (rate: ℚ) (hours_elapsed_real_time hours_elapsed_car_time : ℚ) : Prop :=
  rate = hours_elapsed_car_time / hours_elapsed_real_time

def actual_time (current_car_time car_rate : ℚ) : ℚ :=
  current_car_time / car_rate

theorem actual_time_when_car_clock_shows_10PM :
  let accurate_start_time := 9 -- 9:00 AM
  let car_start_time := 9 -- Synchronized at 9:00 AM
  let wristwatch_time_wristwatch := 13 -- 1:00 PM in hours
  let car_time_car := 13 + 48 / 60 -- 1:48 PM in hours
  let rate := car_time_car / wristwatch_time_wristwatch
  let current_car_time := 22 -- 10:00 PM in hours
  let real_time := actual_time current_car_time rate
  real_time = 19.8333 := -- which converts to 7:50 PM (Option B)
sorry

end actual_time_when_car_clock_shows_10PM_l175_175022


namespace area_of_rotated_squares_l175_175102

noncomputable def side_length : ℝ := 8
noncomputable def rotation_middle : ℝ := 45
noncomputable def rotation_top : ℝ := 75

-- Theorem: The area of the resulting 24-sided polygon.
theorem area_of_rotated_squares :
  (∃ (polygon_area : ℝ), polygon_area = 96) :=
sorry

end area_of_rotated_squares_l175_175102


namespace Seokjin_tangerines_per_day_l175_175321

theorem Seokjin_tangerines_per_day 
  (T_initial : ℕ) (D : ℕ) (T_remaining : ℕ) 
  (h1 : T_initial = 29) 
  (h2 : D = 8) 
  (h3 : T_remaining = 5) : 
  (T_initial - T_remaining) / D = 3 := 
by
  sorry

end Seokjin_tangerines_per_day_l175_175321


namespace initial_persons_count_l175_175336

open Real

def average_weight_increase (n : ℕ) (increase_per_person : ℝ) : ℝ :=
  increase_per_person * n

def weight_difference (new_weight old_weight : ℝ) : ℝ :=
  new_weight - old_weight

theorem initial_persons_count :
  ∀ (n : ℕ),
  average_weight_increase n 2.5 = weight_difference 95 75 → n = 8 :=
by
  intro n h
  sorry

end initial_persons_count_l175_175336


namespace josh_money_left_l175_175474

theorem josh_money_left :
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  money_left = 15.87 :=
by
  let initial_money := 100.00
  let shirt_cost := 12.67
  let meal_cost := 25.39
  let magazine_cost := 14.25
  let debt_payment := 4.32
  let gadget_cost := 27.50
  let total_spent := shirt_cost + meal_cost + magazine_cost + debt_payment + gadget_cost
  let money_left := initial_money - total_spent
  have h1 : total_spent = 84.13 := sorry
  have h2 : money_left = initial_money - 84.13 := sorry
  have h3 : money_left = 15.87 := sorry
  exact h3

end josh_money_left_l175_175474


namespace asterisk_replacement_l175_175111

theorem asterisk_replacement (x : ℝ) : 
  (x / 20) * (x / 80) = 1 ↔ x = 40 :=
by sorry

end asterisk_replacement_l175_175111


namespace inequality_solution_l175_175610

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then 
    {x : ℝ | 1 < x}
  else if 0 < a ∧ a < 2 then 
    {x : ℝ | 1 < x ∧ x < (2 / a)}
  else if a = 2 then 
    ∅
  else if a > 2 then 
    {x : ℝ | (2 / a) < x ∧ x < 1}
  else 
    {x : ℝ | x < (2 / a)} ∪ {x : ℝ | 1 < x}

theorem inequality_solution (a : ℝ) :
  ∀ x : ℝ, (ax^2 - (a + 2) * x + 2 < 0) ↔ (x ∈ solve_inequality a) :=
sorry

end inequality_solution_l175_175610


namespace platform_length_l175_175222

theorem platform_length (length_train : ℝ) (speed_train_kmph : ℝ) (time_sec : ℝ) (length_platform : ℝ) :
  length_train = 1020 → speed_train_kmph = 102 → time_sec = 50 →
  length_platform = (speed_train_kmph * 1000 / 3600) * time_sec - length_train :=
by
  intros
  sorry

end platform_length_l175_175222


namespace number_of_people_l175_175671

theorem number_of_people (total_eggs : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : 
  total_eggs = 36 → eggs_per_omelet = 4 → omelets_per_person = 3 → 
  (total_eggs / eggs_per_omelet) / omelets_per_person = 3 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_l175_175671


namespace number_of_books_l175_175663

theorem number_of_books (original_books new_books : ℕ) (h1 : original_books = 35) (h2 : new_books = 56) : 
  original_books + new_books = 91 :=
by {
  -- the proof will go here, but is not required for the statement
  sorry
}

end number_of_books_l175_175663


namespace parallel_lines_cond_l175_175337

theorem parallel_lines_cond (a c : ℝ) :
    (∀ (x y : ℝ), (a * x - 2 * y - 1 = 0) ↔ (6 * x - 4 * y + c = 0)) → 
        (a = 3 ∧ ∃ (c : ℝ), c ≠ -2) ∨ (a = 3 ∧ c = -2) := 
sorry

end parallel_lines_cond_l175_175337


namespace problem1_correct_problem2_correct_l175_175249

noncomputable def problem1 := 5 + (-6) + 3 - 8 - (-4)
noncomputable def problem2 := -2^2 - 3 * (-1)^3 - (-1) / (-1 / 2)^2

theorem problem1_correct : problem1 = -2 := by
  rw [problem1]
  sorry

theorem problem2_correct : problem2 = 3 := by
  rw [problem2]
  sorry

end problem1_correct_problem2_correct_l175_175249


namespace non_congruent_rectangles_with_even_dimensions_l175_175389

/-- Given a rectangle with perimeter 120 inches and even integer dimensions,
    prove that there are 15 non-congruent rectangles that meet these criteria. -/
theorem non_congruent_rectangles_with_even_dimensions (h w : ℕ) (h_even : h % 2 = 0) (w_even : w % 2 = 0) (perimeter_condition : 2 * (h + w) = 120) :
  ∃ n : ℕ, n = 15 := sorry

end non_congruent_rectangles_with_even_dimensions_l175_175389


namespace steven_has_72_shirts_l175_175329

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l175_175329


namespace total_production_by_june_l175_175956

def initial_production : ℕ := 10

def common_ratio : ℕ := 3

def production_june : ℕ :=
  let a := initial_production
  let r := common_ratio
  a * ((r^6 - 1) / (r - 1))

theorem total_production_by_june : production_june = 3640 :=
by sorry

end total_production_by_june_l175_175956


namespace number_picked_by_person_announcing_average_5_l175_175195

-- Definition of given propositions and assumptions
def numbers_picked (b : Fin 6 → ℕ) (average : Fin 6 → ℕ) :=
  (b 4 = 15) ∧
  (average 4 = 8) ∧
  (average 1 = 5) ∧
  (b 2 + b 4 = 16) ∧
  (b 0 + b 2 = 10) ∧
  (b 4 + b 0 = 12)

-- Prove that given the conditions, the number picked by the person announcing an average of 5 is 7
theorem number_picked_by_person_announcing_average_5 (b : Fin 6 → ℕ) (average : Fin 6 → ℕ)
  (h : numbers_picked b average) : b 2 = 7 :=
  sorry

end number_picked_by_person_announcing_average_5_l175_175195


namespace lilies_per_centerpiece_l175_175484

def centerpieces := 6
def roses_per_centerpiece := 8
def orchids_per_rose := 2
def total_flowers := 120
def ratio_roses_orchids_lilies_centerpiece := 1 / 2 / 3

theorem lilies_per_centerpiece :
  ∀ (c : ℕ) (r : ℕ) (o : ℕ) (l : ℕ),
  c = centerpieces → r = roses_per_centerpiece →
  o = orchids_per_rose * r →
  total_flowers = 6 * (r + o + l) →
  ratio_roses_orchids_lilies_centerpiece = r / o / l →
  l = 10 := by sorry

end lilies_per_centerpiece_l175_175484


namespace first_player_wins_l175_175525

theorem first_player_wins :
  ∀ {table : Type} {coin : Type} 
  (can_place : table → coin → Prop) -- function defining if a coin can be placed on the table
  (not_overlap : ∀ (t : table) (c1 c2 : coin), (can_place t c1 ∧ can_place t c2) → c1 ≠ c2) -- coins do not overlap
  (first_move_center : table → coin) -- first player places the coin at the center
  (mirror_move : table → coin → coin), -- function to place a coin symmetrically
  (∃ strategy : (table → Prop) → (coin → Prop),
    (∀ (t : table) (p : table → Prop), p t → strategy p (mirror_move t (first_move_center t))) ∧ 
    (∀ (t : table) (p : table → Prop), strategy p (first_move_center t) → p t)) := sorry

end first_player_wins_l175_175525


namespace garden_yield_l175_175797

theorem garden_yield
  (steps_length : ℕ)
  (steps_width : ℕ)
  (step_to_feet : ℕ → ℝ)
  (yield_per_sqft : ℝ)
  (h1 : steps_length = 18)
  (h2 : steps_width = 25)
  (h3 : ∀ n : ℕ, step_to_feet n = n * 2.5)
  (h4 : yield_per_sqft = 2 / 3)
  : (step_to_feet steps_length * step_to_feet steps_width) * yield_per_sqft = 1875 :=
by
  sorry

end garden_yield_l175_175797


namespace number_of_combinations_l175_175180

noncomputable def countOddNumbers (n : ℕ) : ℕ := (n + 1) / 2

noncomputable def countPrimesLessThan30 : ℕ := 9 -- {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def countMultiplesOfFour (n : ℕ) : ℕ := n / 4

theorem number_of_combinations : countOddNumbers 40 * countPrimesLessThan30 * countMultiplesOfFour 40 = 1800 := by
  sorry

end number_of_combinations_l175_175180


namespace three_digit_number_equality_l175_175413

theorem three_digit_number_equality :
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧
  (100 * x + 10 * y + z = x^2 + y + z^3) ∧
  (100 * x + 10 * y + z = 357) :=
by
  sorry

end three_digit_number_equality_l175_175413


namespace intersection_points_l175_175000

open Real

def parabola1 (x : ℝ) : ℝ := x^2 - 3 * x + 2
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem intersection_points : 
  ∃ x y : ℝ, 
  (parabola1 x = y ∧ parabola2 x = y) ∧ 
  ((x = 1/2 ∧ y = 3/4) ∨ (x = -3 ∧ y = 20)) :=
by sorry

end intersection_points_l175_175000


namespace length_of_AB_l175_175711

noncomputable def parabola_intersection (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
|x1 - x2|

theorem length_of_AB : 
  ∀ (x1 x2 y1 y2 : ℝ),
    (x1 + x2 = 6) →
    (A = (x1, y1)) →
    (B = (x2, y2)) →
    (y1^2 = 4 * x1) →
    (y2^2 = 4 * x2) →
    parabola_intersection x1 x2 y1 y2 = 8 :=
by
  sorry

end length_of_AB_l175_175711


namespace order_of_f_values_l175_175186

noncomputable def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else 0 -- define f such that it handles the missing part

theorem order_of_f_values :
  (∀ x: ℝ, f (2 - x) = f (1 + x)) ∧ (∀ x: ℝ, x >= 1 → f x = 3^x - 1) →
  f 0 < f 3 ∧ f 3 < f (-2) :=
by
  sorry

end order_of_f_values_l175_175186


namespace sum_of_n_values_l175_175538

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l175_175538


namespace find_remainder_in_division_l175_175900

theorem find_remainder_in_division
  (D : ℕ)
  (r : ℕ) -- the remainder when using the incorrect divisor
  (R : ℕ) -- the remainder when using the correct divisor
  (h1 : D = 12 * 63 + r)
  (h2 : D = 21 * 36 + R)
  : R = 0 :=
by
  sorry

end find_remainder_in_division_l175_175900


namespace count_rectangles_with_perimeter_twenty_two_l175_175162

theorem count_rectangles_with_perimeter_twenty_two : 
  (∃! (n : ℕ), n = 11) :=
by
  sorry

end count_rectangles_with_perimeter_twenty_two_l175_175162


namespace problem_statement_l175_175184

noncomputable def f (x : ℝ) : ℝ := ∫ t in -x..x, Real.cos t

theorem problem_statement : f (f (Real.pi / 4)) = 2 * Real.sin (Real.sqrt 2) := 
by
  sorry

end problem_statement_l175_175184


namespace speed_of_first_train_is_correct_l175_175212

-- Define the lengths of the trains
def length_train1 : ℕ := 110
def length_train2 : ℕ := 200

-- Define the speed of the second train in kmph
def speed_train2 : ℕ := 65

-- Define the time they take to clear each other in seconds
def time_clear_seconds : ℚ := 7.695936049253991

-- Define the speed of the first train
def speed_train1 : ℚ :=
  let time_clear_hours : ℚ := time_clear_seconds / 3600
  let total_distance_km : ℚ := (length_train1 + length_train2) / 1000
  let relative_speed_kmph : ℚ := total_distance_km / time_clear_hours 
  relative_speed_kmph - speed_train2

-- The proof problem is to show that the speed of the first train is 80.069 kmph
theorem speed_of_first_train_is_correct : speed_train1 = 80.069 := by
  sorry

end speed_of_first_train_is_correct_l175_175212


namespace length_of_other_parallel_side_l175_175603

theorem length_of_other_parallel_side (a b h area : ℝ) 
  (h_area : area = 190) 
  (h_parallel1 : b = 18) 
  (h_height : h = 10) : 
  a = 20 :=
by
  sorry

end length_of_other_parallel_side_l175_175603


namespace solve_for_n_l175_175833

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by
  sorry

end solve_for_n_l175_175833


namespace possible_values_of_sum_l175_175183

theorem possible_values_of_sum (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 2) : 
  set.Ici (2 : ℝ) = {x : ℝ | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b)} :=
by
  sorry

end possible_values_of_sum_l175_175183


namespace square_side_length_same_area_l175_175612

theorem square_side_length_same_area (length width : ℕ) (l_eq : length = 72) (w_eq : width = 18) : 
  ∃ side_length : ℕ, side_length * side_length = length * width ∧ side_length = 36 :=
by
  sorry

end square_side_length_same_area_l175_175612


namespace pet_shop_ways_l175_175124

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways_l175_175124


namespace find_minimum_m_l175_175021

noncomputable def volume_space (m : ℝ) : ℝ := m^4
noncomputable def favorable_event (m : ℝ) : ℝ := (m - 1)^4
noncomputable def probability_favorable (m : ℝ) : ℝ := favorable_event m / volume_space m

theorem find_minimum_m :
  ∃ m : ℕ, 
    (probability_favorable m > 2/3) ∧
    (∀ n : ℕ, n < m → ¬(probability_favorable n > 2/3)) ∧
    m = 12 :=
by
  sorry

end find_minimum_m_l175_175021


namespace farmer_plants_rows_per_bed_l175_175119

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed_l175_175119


namespace arithmetic_sequence_sum_l175_175883

noncomputable def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℕ :=
  n * 2^n

def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : a_n 1 + a_n 2 + a_n 3 = 6)
  (h2 : a_n 5 = 5)
  (h3 : ∀ n, b_n n = a_n n * 2^(a_n n)) :
  (∀ n, a_n n = n) ∧ (∀ n, S_n n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end arithmetic_sequence_sum_l175_175883


namespace geom_series_first_term_l175_175398

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l175_175398


namespace lines_coplanar_l175_175487

/-
Given:
- Line 1 parameterized as (2 + s, 4 - k * s, -1 + k * s)
- Line 2 parameterized as (2 * t, 2 + t, 3 - t)
Prove: If these lines are coplanar, then k = -1/2
-/
theorem lines_coplanar (k : ℚ) (s t : ℚ)
  (line1 : ℚ × ℚ × ℚ := (2 + s, 4 - k * s, -1 + k * s))
  (line2 : ℚ × ℚ × ℚ := (2 * t, 2 + t, 3 - t))
  (coplanar : ∃ (s t : ℚ), line1 = line2) :
  k = -1 / 2 := 
sorry

end lines_coplanar_l175_175487


namespace find_original_shirt_price_l175_175312

noncomputable def original_shirt_price (S pants_orig_price jacket_orig_price total_paid : ℝ) :=
  let discounted_shirt := S * 0.5625
  let discounted_pants := pants_orig_price * 0.70
  let discounted_jacket := jacket_orig_price * 0.64
  let total_before_loyalty := discounted_shirt + discounted_pants + discounted_jacket
  let total_after_loyalty := total_before_loyalty * 0.90
  let total_after_tax := total_after_loyalty * 1.15
  total_after_tax = total_paid

theorem find_original_shirt_price : 
  original_shirt_price S 50 75 150 → S = 110.07 :=
by
  intro h
  sorry

end find_original_shirt_price_l175_175312


namespace transform_negation_l175_175724

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l175_175724


namespace cups_remaining_l175_175069

-- Definitions based on problem conditions
def initial_cups : ℕ := 12
def mary_morning_cups : ℕ := 1
def mary_evening_cups : ℕ := 1
def frank_afternoon_cups : ℕ := 1
def frank_late_evening_cups : ℕ := 2 * frank_afternoon_cups

-- Hypothesis combining all conditions:
def total_given_cups : ℕ :=
  mary_morning_cups + mary_evening_cups + frank_afternoon_cups + frank_late_evening_cups

-- Theorem to prove
theorem cups_remaining : initial_cups - total_given_cups = 7 :=
  sorry

end cups_remaining_l175_175069


namespace train_length_is_500_l175_175959

def speed_kmph : ℕ := 360
def time_sec : ℕ := 5

def speed_mps (v_kmph : ℕ) : ℕ :=
  v_kmph * 1000 / 3600

def length_of_train (v_mps : ℕ) (t_sec : ℕ) : ℕ :=
  v_mps * t_sec

theorem train_length_is_500 :
  length_of_train (speed_mps speed_kmph) time_sec = 500 := 
sorry

end train_length_is_500_l175_175959


namespace part1_part2_l175_175753

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part1 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a ≥ 0)) ↔ (0 < a ∧ a ≤ 2) := sorry

theorem part2 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (x - 1) * f x a ≥ 0) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_part2_l175_175753


namespace divisibility_expression_l175_175497

variable {R : Type*} [CommRing R] (x a b : R)

theorem divisibility_expression :
  ∃ k : R, (x + a + b) ^ 3 - x ^ 3 - a ^ 3 - b ^ 3 = (x + a) * (x + b) * k :=
sorry

end divisibility_expression_l175_175497


namespace blue_bird_high_school_team_arrangement_l175_175926

theorem blue_bird_high_school_team_arrangement : 
  let girls := 2
  let boys := 3
  let girls_permutations := Nat.factorial girls
  let boys_permutations := Nat.factorial boys
  girls_permutations * boys_permutations = 12 := by
  sorry

end blue_bird_high_school_team_arrangement_l175_175926


namespace math_problem_l175_175915

noncomputable def problem_statement : Prop :=
  let A : ℝ × ℝ := (5, 6)
  let B : ℝ × ℝ := (8, 3)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  let y_intercept : ℝ := A.snd - slope * A.fst
  slope + y_intercept = 10

theorem math_problem : problem_statement := sorry

end math_problem_l175_175915


namespace range_of_k_l175_175281

theorem range_of_k (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + k * x + 1 / 2 ≥ 0) → k ∈ Set.Ioc 0 4 := 
by 
  sorry

end range_of_k_l175_175281


namespace brown_stripes_l175_175827

theorem brown_stripes (B G Bl : ℕ) (h1 : G = 3 * B) (h2 : Bl = 5 * G) (h3 : Bl = 60) : B = 4 :=
by {
  sorry
}

end brown_stripes_l175_175827


namespace inequality_solution_l175_175820

theorem inequality_solution (x : ℝ) : (x - 1) / 3 > 2 → x > 7 :=
by
  intros h
  sorry

end inequality_solution_l175_175820


namespace largest_a_mul_b_l175_175185

-- Given conditions and proof statement
theorem largest_a_mul_b {m k q a b : ℕ} (hm : m = 720 * k + 83)
  (ha : m = a * q + b) (h_b_lt_a: b < a): a * b = 5112 :=
sorry

end largest_a_mul_b_l175_175185


namespace no_non_congruent_right_triangles_l175_175039

theorem no_non_congruent_right_triangles (a b : ℝ) (c : ℝ) (h_right_triangle : c = Real.sqrt (a^2 + b^2)) (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2)) : a = 0 ∨ b = 0 :=
by
  sorry

end no_non_congruent_right_triangles_l175_175039


namespace ratio_perimeters_of_squares_l175_175695

theorem ratio_perimeters_of_squares (a b : ℝ) (h_diag : (a * Real.sqrt 2) / (b * Real.sqrt 2) = 2.5) : (4 * a) / (4 * b) = 10 :=
by
  sorry

end ratio_perimeters_of_squares_l175_175695


namespace total_books_l175_175493

def numberOfMysteryShelves := 6
def numberOfPictureShelves := 2
def booksPerShelf := 9

theorem total_books (hMystery : numberOfMysteryShelves = 6) 
                    (hPicture : numberOfPictureShelves = 2) 
                    (hBooksPerShelf : booksPerShelf = 9) :
  numberOfMysteryShelves * booksPerShelf + numberOfPictureShelves * booksPerShelf = 72 :=
  by 
  sorry

end total_books_l175_175493


namespace handshake_count_l175_175175

-- Definitions based on conditions
def groupA_size : ℕ := 25
def groupB_size : ℕ := 15

-- Total number of handshakes is calculated as product of their sizes
def total_handshakes : ℕ := groupA_size * groupB_size

-- The theorem we need to prove
theorem handshake_count : total_handshakes = 375 :=
by
  -- skipped proof
  sorry

end handshake_count_l175_175175


namespace gcd_three_digit_palindromes_l175_175362

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l175_175362


namespace smallest_even_x_l175_175831

theorem smallest_even_x (x : ℤ) (h1 : x < 3 * x - 10) (h2 : ∃ k : ℤ, x = 2 * k) : x = 6 :=
by {
  sorry
}

end smallest_even_x_l175_175831


namespace total_weight_cashew_nuts_and_peanuts_l175_175712

theorem total_weight_cashew_nuts_and_peanuts (weight_cashew_nuts weight_peanuts : ℕ) (h1 : weight_cashew_nuts = 3) (h2 : weight_peanuts = 2) : 
  weight_cashew_nuts + weight_peanuts = 5 := 
by
  sorry

end total_weight_cashew_nuts_and_peanuts_l175_175712


namespace max_distance_l175_175574

theorem max_distance (x y : ℝ) (u v w : ℝ)
  (h1 : u = Real.sqrt (x^2 + y^2))
  (h2 : v = Real.sqrt ((x - 1)^2 + y^2))
  (h3 : w = Real.sqrt ((x - 1)^2 + (y - 1)^2))
  (h4 : u^2 + v^2 = w^2) :
  ∃ (P : ℝ), P = 2 + Real.sqrt 2 :=
sorry

end max_distance_l175_175574


namespace trigonometric_identity_l175_175043

theorem trigonometric_identity 
  (α : ℝ)
  (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l175_175043


namespace lcm_852_1491_l175_175947

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l175_175947


namespace percentage_cut_l175_175520

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem percentage_cut : (cut_amount / original_budget) * 100 = 70 :=
by
  sorry

end percentage_cut_l175_175520


namespace candy_box_price_l175_175961

theorem candy_box_price (c s : ℝ) 
  (h1 : 1.50 * s = 6) 
  (h2 : c + s = 16) 
  (h3 : ∀ c, 1.25 * c = 1.25 * 12) : 
  (1.25 * c = 15) :=
by
  sorry

end candy_box_price_l175_175961


namespace tom_read_books_l175_175356

theorem tom_read_books :
  let books_may := 2
  let books_june := 6
  let books_july := 10
  books_may + books_june + books_july = 18 := by
  sorry

end tom_read_books_l175_175356


namespace max_triangle_perimeter_l175_175577

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l175_175577


namespace product_of_two_numbers_l175_175349

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l175_175349


namespace quadratic_real_roots_range_l175_175276

theorem quadratic_real_roots_range (m : ℝ) : (∃ x y : ℝ, x ≠ y ∧ mx^2 + 2*x + 1 = 0 ∧ yx^2 + 2*y + 1 = 0) → m ≤ 1 ∧ m ≠ 0 :=
by 
sorry

end quadratic_real_roots_range_l175_175276


namespace sum_of_number_is_8_l175_175204

theorem sum_of_number_is_8 (x v : ℝ) (h1 : 0.75 * x + 2 = v) (h2 : x = 8.0) : v = 8.0 :=
by
  sorry

end sum_of_number_is_8_l175_175204


namespace max_ab_value_l175_175844

variable (a b c : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < 1
axiom h2 : 0 < b ∧ b < 1
axiom h3 : 0 < c ∧ c < 1
axiom h4 : 3 * a + 2 * b = 1

-- Goal
theorem max_ab_value : ab = 1 / 24 :=
by
  sorry

end max_ab_value_l175_175844


namespace log2_q_for_tournament_l175_175262

noncomputable def tournament_game_log2 (num_teams : ℕ) : ℕ :=
  let num_games := num_teams * (num_teams - 1) / 2
  let factorial_powers_2 := ∑ n in Finset.range (num_teams + 1), num_teams / 2^n
  num_games - factorial_powers_2

theorem log2_q_for_tournament :
  tournament_game_log2 50 = 1178 :=
by
  sorry

end log2_q_for_tournament_l175_175262


namespace sqrt_product_of_powers_eq_l175_175256

theorem sqrt_product_of_powers_eq :
  ∃ (x y z : ℕ), prime x ∧ prime y ∧ prime z ∧ x = 2 ∧ y = 3 ∧ z = 5 ∧
  sqrt (x^4 * y^6 * z^2) = 540 := by
  use 2, 3, 5
  show prime 2, from prime_two
  show prime 3, from prime_three
  show prime 5, from prime_five
  show 2 = 2, from rfl
  show 3 = 3, from rfl
  show 5 = 5, from rfl
  sorry

end sqrt_product_of_powers_eq_l175_175256


namespace inverse_square_variation_l175_175376

theorem inverse_square_variation (k : ℝ) (y x : ℝ) (h1: x = k / y^2) (h2: 0.25 = k / 36) : 
  x = 1 :=
by
  -- Here, you would provide further Lean code to complete the proof
  -- using the given hypothesis h1 and h2, along with some computation.
  sorry

end inverse_square_variation_l175_175376


namespace no_duplicate_among_expressions_l175_175713

theorem no_duplicate_among_expressions
  (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ)
  (ha : a1 = x^2)
  (hb : b1 = y^3)
  (hc : c1 = z^5)
  (hd : d1 = w^7)
  (ha2 : a2 = m^2)
  (hb2 : b2 = n^3)
  (hc2 : c2 = p^5)
  (hd2 : d2 = q^7)
  (h1 : N = a1 - a2)
  (h2 : N = b1 - b2)
  (h3 : N = c1 - c2)
  (h4 : N = d1 - d2) :
  ¬ (a1 = b1 ∨ a1 = c1 ∨ a1 = d1 ∨ b1 = c1 ∨ b1 = d1 ∨ c1 = d1) :=
by
  -- Begin proof here
  sorry

end no_duplicate_among_expressions_l175_175713


namespace geom_series_first_term_l175_175404

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l175_175404


namespace find_angle_B_find_sin_C_l175_175456

-- Statement for proving B = π / 4 given the conditions
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B) 
  (hABC : A + B + C = Real.pi) :
  B = Real.pi / 4 := 
sorry

-- Statement for proving sin C when cos A = 1 / 3
theorem find_sin_C (A C : ℝ) 
  (hA : Real.cos A = 1 / 3)
  (hABC : A + Real.pi / 4 + C = Real.pi) :
  Real.sin C = (4 + Real.sqrt 2) / 6 := 
sorry

end find_angle_B_find_sin_C_l175_175456


namespace number_of_students_l175_175081

theorem number_of_students (B S : ℕ) 
  (h1 : S = 9 * B + 1) 
  (h2 : S = 10 * B - 10) : 
  S = 100 := 
by 
  { sorry }

end number_of_students_l175_175081


namespace factory_a_min_hours_l175_175559

theorem factory_a_min_hours (x : ℕ) :
  (550 * x + (700 - 55 * x) / 45 * 495 ≤ 7260) → (8 ≤ x) :=
by
  sorry

end factory_a_min_hours_l175_175559


namespace isosceles_triangle_sides_l175_175615

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l175_175615


namespace monotonic_decreasing_interval_l175_175200

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 (Real.sqrt 3 / 3), (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l175_175200


namespace lcm_ac_is_420_l175_175341

theorem lcm_ac_is_420 (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
    Nat.lcm a c = 420 :=
sorry

end lcm_ac_is_420_l175_175341


namespace profit_difference_l175_175223

variable (a_capital b_capital c_capital b_profit : ℕ)

theorem profit_difference (h₁ : a_capital = 8000) (h₂ : b_capital = 10000) 
                          (h₃ : c_capital = 12000) (h₄ : b_profit = 2000) : 
  c_capital * (b_profit / b_capital) - a_capital * (b_profit / b_capital) = 800 := 
sorry

end profit_difference_l175_175223


namespace a_pow_10_plus_b_pow_10_l175_175801

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end a_pow_10_plus_b_pow_10_l175_175801


namespace height_of_cuboid_l175_175100

theorem height_of_cuboid (A l w : ℝ) (h : ℝ) (hA : A = 442) (hl : l = 7) (hw : w = 8) : h = 11 :=
by
  sorry

end height_of_cuboid_l175_175100


namespace range_of_a_l175_175155

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l175_175155


namespace remainder_when_a6_divided_by_n_l175_175480

theorem remainder_when_a6_divided_by_n (n : ℕ) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := 
sorry

end remainder_when_a6_divided_by_n_l175_175480


namespace future_skyscraper_climb_proof_l175_175171

variable {H_f H_c H_fut : ℝ}

theorem future_skyscraper_climb_proof
  (H_f : ℝ)
  (H_c : ℝ := 3 * H_f)
  (H_fut : ℝ := 1.25 * H_c)
  (T_f : ℝ := 1) :
  (H_fut * T_f / H_f) > 2 * T_f :=
by
  -- specific calculations would go here
  sorry

end future_skyscraper_climb_proof_l175_175171


namespace digit_difference_base2_150_950_l175_175701

def largest_power_of_2_lt (n : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ k, 2^k ≤ n).last' getLastRange

def base2_digits (n : ℕ) : ℕ := largest_power_of_2_lt n + 1

theorem digit_difference_base2_150_950 :
  base2_digits 950 - base2_digits 150 = 2 :=
by {
  sorry
}

end digit_difference_base2_150_950_l175_175701


namespace shane_current_age_l175_175692

theorem shane_current_age (Garret_age : ℕ) (h : Garret_age = 12) : 
  (let Shane_age_twenty_years_ago := 2 * Garret_age in
   let Shane_current := Shane_age_twenty_years_ago + 20 in
   Shane_current = 44) :=
by
  sorry

end shane_current_age_l175_175692


namespace total_photos_l175_175299

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l175_175299


namespace part1_part2_l175_175661

def z1 (a : ℝ) : Complex := Complex.mk 2 a
def z2 : Complex := Complex.mk 3 (-4)

-- Part 1: Prove that the product of z1 and z2 equals 10 - 5i when a = 1.
theorem part1 : z1 1 * z2 = Complex.mk 10 (-5) :=
by
  -- proof to be filled in
  sorry

-- Part 2: Prove that a = 4 when z1 + z2 is a real number.
theorem part2 (a : ℝ) (h : (z1 a + z2).im = 0) : a = 4 :=
by
  -- proof to be filled in
  sorry

end part1_part2_l175_175661


namespace number_of_pages_in_bible_l175_175907

-- Definitions based on conditions
def hours_per_day := 2
def pages_per_hour := 50
def weeks := 4
def days_per_week := 7

-- Hypotheses transformed into mathematical facts
def total_days := weeks * days_per_week
def total_hours := total_days * hours_per_day
def total_pages := total_hours * pages_per_hour

-- Theorem to prove the Bible length based on conditions
theorem number_of_pages_in_bible : total_pages = 2800 := 
by
  sorry

end number_of_pages_in_bible_l175_175907


namespace mean_score_74_9_l175_175646

/-- 
In a class of 100 students, the score distribution is as follows:
- 10 students scored 100%
- 15 students scored 90%
- 20 students scored 80%
- 30 students scored 70%
- 20 students scored 60%
- 4 students scored 50%
- 1 student scored 40%

Prove that the mean percentage score of the class is 74.9.
-/
theorem mean_score_74_9 : 
  let scores := [100, 90, 80, 70, 60, 50, 40]
  let counts := [10, 15, 20, 30, 20, 4, 1]
  let total_students := 100
  let total_score := 1000 + 1350 + 1600 + 2100 + 1200 + 200 + 40
  (total_score / total_students : ℝ) = 74.9 :=
by {
  -- The detailed proof steps are omitted with sorry.
  sorry
}

end mean_score_74_9_l175_175646


namespace hyperbola_eccentricity_l175_175446

theorem hyperbola_eccentricity (h : ∀ x y m : ℝ, x^2 - y^2 / m = 1 → m > 0 → (Real.sqrt (1 + m) = Real.sqrt 3)) : ∃ m : ℝ, m = 2 := sorry

end hyperbola_eccentricity_l175_175446


namespace points_satisfy_equation_l175_175037

theorem points_satisfy_equation (x y : ℝ) : 
  (2 * x^2 + 3 * x * y + y^2 + x = 1) ↔ (y = -x - 1) ∨ (y = -2 * x + 1) := by
  sorry

end points_satisfy_equation_l175_175037


namespace polynomial_zero_pairs_l175_175602

theorem polynomial_zero_pairs (r s : ℝ) :
  (∀ x : ℝ, (x = 0 ∨ x = 0) ↔ x^2 - 2 * r * x + r = 0) ∧
  (∀ x : ℝ, (x = 0 ∨ x = 0 ∨ x = 0) ↔ 27 * x^3 - 27 * r * x^2 + s * x - r^6 = 0) → 
  (r, s) = (0, 0) ∨ (r, s) = (1, 9) :=
by
  sorry

end polynomial_zero_pairs_l175_175602


namespace complex_numbers_satisfying_conditions_l175_175014

theorem complex_numbers_satisfying_conditions (x y z : ℂ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : x = 1 ∧ y = 1 ∧ z = 1 := 
by sorry

end complex_numbers_satisfying_conditions_l175_175014


namespace sqrt_product_of_powers_eq_l175_175255

theorem sqrt_product_of_powers_eq :
  ∃ (x y z : ℕ), prime x ∧ prime y ∧ prime z ∧ x = 2 ∧ y = 3 ∧ z = 5 ∧
  sqrt (x^4 * y^6 * z^2) = 540 := by
  use 2, 3, 5
  show prime 2, from prime_two
  show prime 3, from prime_three
  show prime 5, from prime_five
  show 2 = 2, from rfl
  show 3 = 3, from rfl
  show 5 = 5, from rfl
  sorry

end sqrt_product_of_powers_eq_l175_175255


namespace total_photos_l175_175297

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l175_175297


namespace exists_n_good_but_not_succ_good_l175_175064

def S (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), 
    a_seq n = a ∧ (∀ i : Fin n, a_seq (Fin.succ i) = a_seq i - S (a_seq i))

theorem exists_n_good_but_not_succ_good (n : ℕ) : 
  ∃ a, n_good n a ∧ ¬ n_good (n + 1) a := 
sorry

end exists_n_good_but_not_succ_good_l175_175064


namespace percentage_decrease_is_14_percent_l175_175181

-- Definitions based on conditions
def original_price_per_pack : ℚ := 7 / 3
def new_price_per_pack : ℚ := 8 / 4

-- Statement to prove that percentage decrease is 14%
theorem percentage_decrease_is_14_percent :
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100 = 14 := by
  sorry

end percentage_decrease_is_14_percent_l175_175181


namespace flag_design_combinations_l175_175935

-- Definitions
def colors : Nat := 3  -- Number of colors: purple, gold, and silver
def stripes : Nat := 3  -- Number of horizontal stripes in the flag

-- The Lean statement
theorem flag_design_combinations :
  (colors ^ stripes) = 27 :=
by
  sorry

end flag_design_combinations_l175_175935


namespace arithmetic_seq_infinitely_many_squares_l175_175855

theorem arithmetic_seq_infinitely_many_squares 
  (a d : ℕ) 
  (h : ∃ (n y : ℕ), a + n * d = y^2) : 
  ∃ (m : ℕ), ∀ k : ℕ, ∃ n' y' : ℕ, a + n' * d = y'^2 :=
by sorry

end arithmetic_seq_infinitely_many_squares_l175_175855


namespace find_sister_candy_initially_l175_175059

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end find_sister_candy_initially_l175_175059


namespace cyclists_meet_at_starting_point_l175_175373

-- Define the conditions: speeds of cyclists and the circumference of the circle
def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def circumference : ℝ := 300

-- Define the total speed by summing individual speeds
def relative_speed : ℝ := speed_cyclist1 + speed_cyclist2

-- Define the time required to meet at the starting point
def meeting_time : ℝ := 20

-- The theorem statement which states that given the conditions, the cyclists will meet after 20 seconds
theorem cyclists_meet_at_starting_point :
  meeting_time = circumference / relative_speed :=
sorry

end cyclists_meet_at_starting_point_l175_175373


namespace divide_equally_l175_175786

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end divide_equally_l175_175786


namespace common_external_tangent_b_l175_175524

def circle1_center := (1, 3)
def circle1_radius := 3
def circle2_center := (10, 6)
def circle2_radius := 7

theorem common_external_tangent_b :
  ∃ (b : ℝ), ∀ (m : ℝ), m = 3 / 4 ∧ b = 9 / 4 := sorry

end common_external_tangent_b_l175_175524


namespace effective_writing_speed_is_750_l175_175388

-- Definitions based on given conditions in problem part a)
def total_words : ℕ := 60000
def total_hours : ℕ := 100
def break_hours : ℕ := 20
def effective_hours : ℕ := total_hours - break_hours
def effective_writing_speed : ℕ := total_words / effective_hours

-- Statement to be proved
theorem effective_writing_speed_is_750 : effective_writing_speed = 750 := by
  sorry

end effective_writing_speed_is_750_l175_175388


namespace polynomial_bound_l175_175793

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (h : ∀ x : ℝ, |x| < 1 → |P a b c d x| ≤ 1) :
  |a| + |b| + |c| + |d| ≤ 7 :=
sorry

end polynomial_bound_l175_175793


namespace min_abc_sum_l175_175029

theorem min_abc_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 8) : a + b + c ≥ 6 :=
by {
  sorry
}

end min_abc_sum_l175_175029


namespace investment_growth_theorem_l175_175690

variable (x : ℝ)

-- Defining the initial and final investments
def initial_investment : ℝ := 800
def final_investment : ℝ := 960

-- Defining the growth equation
def growth_equation (x : ℝ) : Prop := initial_investment * (1 + x) ^ 2 = final_investment

-- The theorem statement that needs to be proven
theorem investment_growth_theorem : growth_equation x := sorry

end investment_growth_theorem_l175_175690


namespace amelia_painted_faces_l175_175246

def faces_of_cuboid : ℕ := 6
def number_of_cuboids : ℕ := 6

theorem amelia_painted_faces : faces_of_cuboid * number_of_cuboids = 36 :=
by {
  sorry
}

end amelia_painted_faces_l175_175246


namespace swimmers_meeting_times_l175_175176

theorem swimmers_meeting_times (l : ℕ) (vA vB t : ℕ) (T : ℝ) :
  l = 120 →
  vA = 4 →
  vB = 3 →
  t = 15 →
  T = 21 :=
  sorry

end swimmers_meeting_times_l175_175176


namespace rth_term_l175_175269

-- Given arithmetic progression sum formula
def Sn (n : ℕ) : ℕ := 3 * n^2 + 4 * n + 5

-- Prove that the r-th term of the sequence is 6r + 1
theorem rth_term (r : ℕ) : (Sn r) - (Sn (r - 1)) = 6 * r + 1 :=
by
  sorry

end rth_term_l175_175269


namespace proof_inequality_l175_175272

noncomputable def inequality_proof (α : ℝ) (a b : ℝ) (m : ℕ) : Prop :=
  (0 < α) → (α < Real.pi / 2) →
  (m ≥ 1) →
  (0 < a) → (0 < b) →
  (a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2))

-- Statement of the proof problem
theorem proof_inequality (α : ℝ) (a b : ℝ) (m : ℕ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : 1 ≤ m) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ 
    (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2) :=
by
  sorry

end proof_inequality_l175_175272


namespace solve_equation1_solve_equation2_pos_solve_equation2_neg_l175_175871

theorem solve_equation1 (x : ℝ) (h : 2 * x^3 = 16) : x = 2 :=
sorry

theorem solve_equation2_pos (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 :=
sorry

theorem solve_equation2_neg (x : ℝ) (h : (x - 1)^2 = 4) : x = -1 :=
sorry

end solve_equation1_solve_equation2_pos_solve_equation2_neg_l175_175871


namespace min_days_to_triple_loan_l175_175395

theorem min_days_to_triple_loan (amount_borrowed : ℕ) (interest_rate : ℝ) :
  ∀ x : ℕ, x ≥ 20 ↔ amount_borrowed + (amount_borrowed * (interest_rate / 10)) * x ≥ 3 * amount_borrowed :=
sorry

end min_days_to_triple_loan_l175_175395


namespace sum_of_n_for_3n_minus_8_eq_5_l175_175530

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l175_175530


namespace derivative_of_y_l175_175416

noncomputable def y (x : ℝ) : ℝ :=
  1/2 * Real.tanh x + 1/(4 * Real.sqrt 2) * Real.log ((1 + Real.sqrt 2 * Real.tanh x) / (1 - Real.sqrt 2 * Real.tanh x))

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 1/(Real.cosh x ^ 2 * (1 - Real.sinh x ^ 2)) := 
by
  sorry

end derivative_of_y_l175_175416


namespace unit_cost_calculation_l175_175231

theorem unit_cost_calculation : 
  ∀ (total_cost : ℕ) (ounces : ℕ), total_cost = 84 → ounces = 12 → (total_cost / ounces = 7) :=
by
  intros total_cost ounces h1 h2
  sorry

end unit_cost_calculation_l175_175231


namespace middle_number_probability_l175_175682

noncomputable theory
open_locale classical

/-- 
  Define the problem conditions.
  - The numbers 1 to 11 are arranged in a line.
  - The middle number in the line is larger than exactly one number to its left.
-/
def middle_larger_left (l : list ℤ) : Prop :=
  l.length = 11 ∧ l.nth 5 > l.take 5.to_finset.count (< l.nth 5)

/--
  Define the probability calculation function for the given problem's conditions.
-/
def probability_larger_right (l : list ℤ) : ℚ :=
  if middle_larger_left l then
    let valid_arrangements := filter middle_larger_left (list.permutations (list.range' 1 11)) in
    (valid_arrangements.filter (λ m, m.nth 5 > m.drop 6.head!)).length / valid_arrangements.length
  else 0

/--
  Statement of the mathematical proof problem in Lean.
  - Prove the probability that the middle number is larger than exactly one number to its right is 10/33.
-/
theorem middle_number_probability :
  ∀ l : list ℤ, probability_larger_right l = 10 / 33 :=
by
  sorry

end middle_number_probability_l175_175682


namespace triangle_area_l175_175241

theorem triangle_area (h : ℝ) (hypotenuse : h = 12) (angle : ∃θ : ℝ, θ = 30 ∧ θ = 30) :
  ∃ (A : ℝ), A = 18 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l175_175241


namespace problem1_problem2_l175_175023

section problems

variables (m n a b : ℕ)
variables (h1 : 4 ^ m = a) (h2 : 8 ^ n = b)

theorem problem1 : 2 ^ (2 * m + 3 * n) = a * b :=
sorry

theorem problem2 : 2 ^ (4 * m - 6 * n) = a ^ 2 / b ^ 2 :=
sorry

end problems

end problem1_problem2_l175_175023


namespace total_stoppage_time_l175_175116

theorem total_stoppage_time (stop1 stop2 stop3 : ℕ) (h1 : stop1 = 5)
  (h2 : stop2 = 8) (h3 : stop3 = 10) : stop1 + stop2 + stop3 = 23 :=
sorry

end total_stoppage_time_l175_175116


namespace speed_increase_needed_l175_175471

-- Definitions based on the conditions
def usual_speed := ℝ
def usual_travel_time := ℝ -- in minutes
def late_departure := 40   -- in minutes
def increased_speed_factor := 1.6
def early_arrival := 9*60 - (8*60 + 35) -- 25 minutes (from 9:00 AM to 8:35 AM)

-- The problem statement in Lean 4
theorem speed_increase_needed (v : usual_speed) (T : usual_travel_time) :
  let T_late := T + late_departure in
  let T_increased_speed := (T / increased_speed_factor) in
  T = usual_travel_time →
  v = usual_speed →
  T_late - T_increased_speed = late_departure + early_arrival → 
  (T - late_departure) / T = 1 / (1 - 40 / (T * (1 / (1.6)))) →
  (v * (3 / 4)) / v = 1.3 := 
sorry

end speed_increase_needed_l175_175471


namespace six_digit_palindromes_count_l175_175994

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l175_175994


namespace range_of_a_l175_175628

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a_l175_175628


namespace exists_unique_adjacent_sums_in_circle_l175_175703

theorem exists_unique_adjacent_sums_in_circle :
  ∃ (f : Fin 10 → Fin 11),
    (∀ (i j : Fin 10), i ≠ j → (f i + f (i + 1)) % 11 ≠ (f j + f (j + 1)) % 11) :=
sorry

end exists_unique_adjacent_sums_in_circle_l175_175703


namespace original_price_of_sarees_l175_175934

theorem original_price_of_sarees (P : ℝ) (h : 0.72 * P = 108) : P = 150 := 
by 
  sorry

end original_price_of_sarees_l175_175934


namespace min_value_polynomial_expression_at_k_eq_1_is_0_l175_175597

-- Definition of the polynomial expression
def polynomial_expression (k x y : ℝ) : ℝ :=
  3 * x^2 - 4 * k * x * y + (2 * k^2 + 1) * y^2 - 6 * x - 2 * y + 4

-- Proof statement
theorem min_value_polynomial_expression_at_k_eq_1_is_0 :
  (∀ x y : ℝ, polynomial_expression 1 x y ≥ 0) ∧ (∃ x y : ℝ, polynomial_expression 1 x y = 0) :=
by
  -- Expected proof here. For now, we indicate sorry to skip the proof.
  sorry

end min_value_polynomial_expression_at_k_eq_1_is_0_l175_175597


namespace sum_of_solutions_of_absolute_value_l175_175529

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l175_175529


namespace line_slope_and_intersection_l175_175268

theorem line_slope_and_intersection:
  (∀ x y : ℝ, x^2 + x / 4 + y / 5 = 1 → ∀ m : ℝ, m = -5 / 4) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → ¬ (x^2 + x / 4 + y / 5 = 1)) :=
by
  sorry

end line_slope_and_intersection_l175_175268


namespace isosceles_triangle_angle_l175_175826

theorem isosceles_triangle_angle
  (A B C : ℝ)
  (h1 : A = C)
  (h2 : B = 2 * A - 40)
  (h3 : A + B + C = 180) :
  B = 70 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_angle_l175_175826


namespace final_price_l175_175571

def initial_price : ℝ := 200
def discount_morning : ℝ := 0.40
def increase_noon : ℝ := 0.25
def discount_afternoon : ℝ := 0.20

theorem final_price : 
  let price_after_morning := initial_price * (1 - discount_morning)
  let price_after_noon := price_after_morning * (1 + increase_noon)
  let final_price := price_after_noon * (1 - discount_afternoon)
  final_price = 120 := 
by
  sorry

end final_price_l175_175571


namespace six_digit_palindromes_count_l175_175996

theorem six_digit_palindromes_count : 
  let digit_count := 10 in
  let a_choices := 9 in
  let b_choices := digit_count in
  let c_choices := digit_count in
  a_choices * b_choices * c_choices = 900 :=
by
  let digit_count := 10
  let a_choices := 9
  let b_choices := digit_count
  let c_choices := digit_count
  show a_choices * b_choices * c_choices = 900
  sorry

end six_digit_palindromes_count_l175_175996


namespace max_squares_covered_by_card_l175_175118

noncomputable def card_coverage_max_squares (card_side : ℝ) (square_side : ℝ) : ℕ :=
  if card_side = 2 ∧ square_side = 1 then 9 else 0

theorem max_squares_covered_by_card : card_coverage_max_squares 2 1 = 9 := by
  sorry

end max_squares_covered_by_card_l175_175118


namespace trig_evaluation_trig_identity_value_l175_175228

-- Problem 1: Prove the trigonometric evaluation
theorem trig_evaluation :
  (Real.cos (9 * Real.pi / 4)) + (Real.tan (-Real.pi / 4)) + (Real.sin (21 * Real.pi)) = (Real.sqrt 2 / 2) - 1 :=
by
  sorry

-- Problem 2: Prove the value given the trigonometric identity
theorem trig_identity_value (θ : ℝ) (h : Real.sin θ = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by
  sorry

end trig_evaluation_trig_identity_value_l175_175228


namespace rectangle_not_equal_118_l175_175125

theorem rectangle_not_equal_118 
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (A : ℕ) (P : ℕ)
  (h₂ : A = a * b) (h₃ : P = 2 * (a + b)) :
  (a + 2) * (b + 2) - 2 ≠ 118 :=
sorry

end rectangle_not_equal_118_l175_175125


namespace complex_pow_simplify_l175_175515

noncomputable def i : ℂ := Complex.I

theorem complex_pow_simplify :
  (1 + Real.sqrt 3 * Complex.I) ^ 3 * Complex.I = -8 * Complex.I :=
by
  sorry

end complex_pow_simplify_l175_175515


namespace chef_additional_wings_l175_175120

theorem chef_additional_wings
    (n : ℕ) (w_initial : ℕ) (w_per_friend : ℕ) (w_additional : ℕ)
    (h1 : n = 4)
    (h2 : w_initial = 9)
    (h3 : w_per_friend = 4)
    (h4 : w_additional = 7) :
    n * w_per_friend - w_initial = w_additional :=
by
  sorry

end chef_additional_wings_l175_175120


namespace unique_elements_set_l175_175904

theorem unique_elements_set (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 0 ↔ 3 ≠ x ∧ x ≠ (x ^ 2 - 2 * x) ∧ (x ^ 2 - 2 * x) ≠ 3 := by
  sorry

end unique_elements_set_l175_175904


namespace ratio_yellow_jelly_beans_l175_175146

theorem ratio_yellow_jelly_beans :
  let bag_A_total := 24
  let bag_B_total := 30
  let bag_C_total := 32
  let bag_D_total := 34
  let bag_A_yellow_ratio := 0.40
  let bag_B_yellow_ratio := 0.30
  let bag_C_yellow_ratio := 0.25 
  let bag_D_yellow_ratio := 0.10
  let bag_A_yellow := bag_A_total * bag_A_yellow_ratio
  let bag_B_yellow := bag_B_total * bag_B_yellow_ratio
  let bag_C_yellow := bag_C_total * bag_C_yellow_ratio
  let bag_D_yellow := bag_D_total * bag_D_yellow_ratio
  let total_yellow := bag_A_yellow + bag_B_yellow + bag_C_yellow + bag_D_yellow
  let total_beans := bag_A_total + bag_B_total + bag_C_total + bag_D_total
  (total_yellow / total_beans) = 0.25 := by
  sorry

end ratio_yellow_jelly_beans_l175_175146


namespace max_abs_f_lower_bound_l175_175873

theorem max_abs_f_lower_bound (a b M : ℝ) (hM : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → abs (x^2 + a*x + b) ≤ M) : 
  M ≥ 1/2 :=
sorry

end max_abs_f_lower_bound_l175_175873


namespace number_of_semesters_l175_175310

-- Define the given conditions
def units_per_semester : ℕ := 20
def cost_per_unit : ℕ := 50
def total_cost : ℕ := 2000

-- Define the cost per semester using the conditions
def cost_per_semester := units_per_semester * cost_per_unit

-- Prove the number of semesters is 2 given the conditions
theorem number_of_semesters : total_cost / cost_per_semester = 2 := by
  -- Add a placeholder "sorry" to skip the actual proof
  sorry

end number_of_semesters_l175_175310


namespace determine_constant_l175_175172

/-- If the function f(x) = a * sin x + 3 * cos x has a maximum value of 5,
then the constant a must be ± 4. -/
theorem determine_constant (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x + 3 * Real.cos x ≤ 5) :
  a = 4 ∨ a = -4 :=
sorry

end determine_constant_l175_175172


namespace Jake_later_than_Austin_l175_175729

theorem Jake_later_than_Austin 
    (floors : ℕ) 
    (steps_per_floor : ℕ) 
    (elevator_time_sec : ℕ)
    (steps_per_sec : ℕ) 
    (jh : floors = 9) 
    (spf : steps_per_floor = 30) 
    (et : elevator_time_sec = 60) 
    (steps_sec : steps_per_sec = 3) 
    : 90 - 60 = 30 := 
by
  have total_steps := 9 * 30
  have time_jake := total_steps / 3
  have Jake_additional_time := time_jake - 60
  rw [eq_one_of_eq_succ_eq_succ jh, eq_one_of_eq_succ_eq_succ spf, eq_one_of_eq_succ_eq_succ et, eq_one_of_eq_succ_eq_succ steps_sec] at *
  norm_num at *
  exact Jake_additional_time

end Jake_later_than_Austin_l175_175729


namespace roots_greater_than_one_implies_range_l175_175153

theorem roots_greater_than_one_implies_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + a = 0 → x > 1) → 3 < a ∧ a ≤ 4 :=
by
  sorry

end roots_greater_than_one_implies_range_l175_175153


namespace solution_to_quadratic_inequality_l175_175989

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l175_175989


namespace find_constants_l175_175887

noncomputable def f (x m n : ℝ) := (m * x + 1) / (x + n)

theorem find_constants (m n : ℝ) (h_symm : ∀ x y, f x m n = y → f (4 - x) m n = 8 - y) : 
  m = 4 ∧ n = -2 := 
by
  sorry

end find_constants_l175_175887


namespace arithmetic_sequence_common_difference_l175_175469

-- Arithmetic sequence with condition and proof of common difference
theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) :
  (a 2015 = a 2013 + 6) → ((a 2015 - a 2013) = 2 * d) → (d = 3) :=
by
  intro h1 h2
  sorry

end arithmetic_sequence_common_difference_l175_175469


namespace joyce_apples_l175_175058

theorem joyce_apples (initial_apples given_apples remaining_apples : ℕ) (h1 : initial_apples = 75) (h2 : given_apples = 52) (h3 : remaining_apples = initial_apples - given_apples) : remaining_apples = 23 :=
by
  rw [h1, h2] at h3
  exact h3

end joyce_apples_l175_175058


namespace find_a_b_l175_175453

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x + 2 > a ∧ x - 1 < b) ↔ (1 < x ∧ x < 3)) → a = 3 ∧ b = 2 :=
by
  intro h
  sorry

end find_a_b_l175_175453


namespace base_seven_to_ten_l175_175358

theorem base_seven_to_ten :
  (6 * 7^4 + 5 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0) = 16244 :=
by sorry

end base_seven_to_ten_l175_175358


namespace triangle_lattice_points_l175_175962

theorem triangle_lattice_points :
  ∀ (A B C : ℕ) (AB AC BC : ℕ), 
    AB = 2016 → AC = 1533 → BC = 1533 → 
    ∃ lattice_points: ℕ, lattice_points = 1165322 := 
by
  sorry

end triangle_lattice_points_l175_175962


namespace chocolate_and_gum_l175_175788

/--
Kolya says that two chocolate bars are more expensive than five gum sticks, 
while Sasha claims that three chocolate bars are more expensive than eight gum sticks. 
When this was checked, only one of them was right. Is it true that seven chocolate bars 
are more expensive than nineteen gum sticks?
-/
theorem chocolate_and_gum (c g : ℝ) (hk : 2 * c > 5 * g) (hs : 3 * c > 8 * g) (only_one_correct : ¬((2 * c > 5 * g) ∧ (3 * c > 8 * g)) ∧ (2 * c > 5 * g ∨ 3 * c > 8 * g)) : 7 * c < 19 * g :=
by
  sorry

end chocolate_and_gum_l175_175788


namespace nesting_rectangles_exists_l175_175874

theorem nesting_rectangles_exists :
  ∀ (rectangles : List (ℕ × ℕ)), rectangles.length = 101
    ∧ (∀ r ∈ rectangles, r.fst ≤ 100 ∧ r.snd ≤ 100) 
    → ∃ (A B C : ℕ × ℕ), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles 
    ∧ (A.fst < B.fst ∧ A.snd < B.snd) 
    ∧ (B.fst < C.fst ∧ B.snd < C.snd) := 
by sorry

end nesting_rectangles_exists_l175_175874


namespace no_real_solutions_l175_175259

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x ^ 2 - 6 * x + 5) ^ 2 + 1 = -|x|

-- Declare the theorem which states there are no real solutions to the given equation
theorem no_real_solutions : ∀ x : ℝ, ¬ equation x :=
by
  intro x
  sorry

end no_real_solutions_l175_175259


namespace remainder_div_3973_28_l175_175697

theorem remainder_div_3973_28 : (3973 % 28) = 9 := by
  sorry

end remainder_div_3973_28_l175_175697


namespace first_player_winning_strategy_l175_175391

-- Defining the type for the positions on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq

-- Initial position C1
def C1 : Position := ⟨3, 1⟩

-- Winning position H8
def H8 : Position := ⟨8, 8⟩

-- Function to check if a position is a winning position
-- the target winning position is H8
def isWinningPosition (p : Position) : Bool :=
  p = H8

-- Function to determine the next possible positions
-- from the current position based on the allowed moves
def nextPositions (p : Position) : List Position :=
  (List.range (8 - p.x)).map (λ dx => ⟨p.x + dx + 1, p.y⟩) ++
  (List.range (8 - p.y)).map (λ dy => ⟨p.x, p.y + dy + 1⟩) ++
  (List.range (min (8 - p.x) (8 - p.y))).map (λ d => ⟨p.x + d + 1, p.y + d + 1⟩)

-- Statement of the problem: First player has a winning strategy from C1
theorem first_player_winning_strategy : 
  ∃ move : Position, move ∈ nextPositions C1 ∧
  ∀ next_move : Position, next_move ∈ nextPositions move → isWinningPosition next_move :=
sorry

end first_player_winning_strategy_l175_175391


namespace solve_inequality_system_l175_175502

theorem solve_inequality_system (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

end solve_inequality_system_l175_175502


namespace kayak_total_until_May_l175_175964

noncomputable def kayak_number (n : ℕ) : ℕ :=
  if n = 0 then 5
  else 3 * kayak_number (n - 1)

theorem kayak_total_until_May : kayak_number 0 + kayak_number 1 + kayak_number 2 + kayak_number 3 = 200 := by
  sorry

end kayak_total_until_May_l175_175964


namespace original_cost_of_pencil_l175_175091

theorem original_cost_of_pencil (final_price discount: ℝ) (h_final: final_price = 3.37) (h_disc: discount = 0.63) : 
  final_price + discount = 4 :=
by
  sorry

end original_cost_of_pencil_l175_175091


namespace kerosene_cost_l175_175777

theorem kerosene_cost (R E K : ℕ) (h1 : E = R) (h2 : K = 6 * E) (h3 : R = 24) : 2 * K = 288 :=
by
  sorry

end kerosene_cost_l175_175777


namespace correct_optionD_l175_175368

def operationA (a : ℝ) : Prop := a^3 + 3 * a^3 = 5 * a^6
def operationB (a : ℝ) : Prop := 7 * a^2 * a^3 = 7 * a^6
def operationC (a : ℝ) : Prop := (-2 * a^3)^2 = 4 * a^5
def operationD (a : ℝ) : Prop := a^8 / a^2 = a^6

theorem correct_optionD (a : ℝ) : ¬ operationA a ∧ ¬ operationB a ∧ ¬ operationC a ∧ operationD a :=
by
  unfold operationA operationB operationC operationD
  sorry

end correct_optionD_l175_175368


namespace calculate_expression_l175_175733

noncomputable def f (x : ℝ) : ℝ :=
  (x^3 + 5 * x^2 + 6 * x) / (x^3 - x^2 - 2 * x)

def num_holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -2
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- vertical asymptotes at x = 0 and x = 1
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no horizontal asymptote
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- oblique asymptote at y = x + 4

theorem calculate_expression : num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 9 :=
by
  -- Provide the proof here
  sorry

end calculate_expression_l175_175733


namespace quadratic_eq_solutions_l175_175937

theorem quadratic_eq_solutions : ∃ x1 x2 : ℝ, (x^2 = x) ∨ (x = 0 ∧ x = 1) := by
  sorry

end quadratic_eq_solutions_l175_175937


namespace find_m_l175_175424

def l1 (m x y: ℝ) : Prop := 2 * x + m * y - 2 = 0
def l2 (m x y: ℝ) : Prop := m * x + 2 * y - 1 = 0
def perpendicular (m : ℝ) : Prop :=
  let slope_l1 := -2 / m
  let slope_l2 := -m / 2
  slope_l1 * slope_l2 = -1

theorem find_m (m : ℝ) (h : perpendicular m) : m = 2 :=
sorry

end find_m_l175_175424


namespace dryer_weight_l175_175006

theorem dryer_weight 
(empty_truck_weight crates_soda_weight num_crates soda_weight_factor 
    fresh_produce_weight_factor num_dryers fully_loaded_truck_weight : ℕ) 

  (h1 : empty_truck_weight = 12000) 
  (h2 : crates_soda_weight = 50) 
  (h3 : num_crates = 20) 
  (h4 : soda_weight_factor = crates_soda_weight * num_crates) 
  (h5 : fresh_produce_weight_factor = 2 * soda_weight_factor) 
  (h6 : num_dryers = 3) 
  (h7 : fully_loaded_truck_weight = 24000) 

  : (fully_loaded_truck_weight - empty_truck_weight 
      - (soda_weight_factor + fresh_produce_weight_factor)) / num_dryers = 3000 := 
by sorry

end dryer_weight_l175_175006


namespace solution_l175_175028

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℝ) (a b : ℝ) (x : ℝ), 
    (|A - B| = 3) ∧
    (|A - C| = 1) ∧
    (A = Real.pi / 2) ∧  -- This typically signifies angle A is 90 degrees.
    (a > 0) ∧
    (b > 0) ∧
    (a = 1) ∧
    (|A - D| = x) ∧
    (|B - D| = 3 - x) ∧
    (|C - D| = Real.sqrt (x^2 + 1)) ∧
    (Real.sqrt (x^2 + 1) - (3 - x) = 2) ∧
    (|A - D| / |B - D| = 4)

theorem solution : problem_statement :=
sorry

end solution_l175_175028


namespace series_converges_l175_175135

theorem series_converges :
  ∑' n, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_converges_l175_175135


namespace marbles_problem_l175_175725

theorem marbles_problem (a : ℚ) (h1: 34 * a = 156) : a = 78 / 17 := 
by
  sorry

end marbles_problem_l175_175725


namespace find_units_min_selling_price_l175_175825

-- Definitions for the given conditions
def total_units : ℕ := 160
def cost_A : ℕ := 150
def cost_B : ℕ := 350
def total_cost : ℕ := 36000
def min_profit : ℕ := 11000

-- Part 1: Proving number of units purchased
theorem find_units :
  ∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x :=
by
  sorry

-- Part 2: Finding the minimum selling price per unit of model A for the profit condition
theorem min_selling_price (t : ℕ) :
  (∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x) →
  100 * (t - cost_A) + 60 * 2 * (t - cost_A) ≥ min_profit →
  t ≥ 200 :=
by
  sorry

end find_units_min_selling_price_l175_175825


namespace part_I_part_II_l175_175885

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - (2 * a + 1) * x

theorem part_I (a : ℝ) (ha : a = -2) : 
  (∃ x : ℝ, f a x = 1) ∧ ∀ x : ℝ, f a x ≤ 1 :=
by sorry

theorem part_II (a : ℝ) (ha : a < 1/2) :
  (∃ x : ℝ, 0 < x ∧ x < exp 1 ∧ f a x < 0) → a < (exp 1 - 1) / (exp 1 * (exp 1 - 2)) :=
by sorry

end part_I_part_II_l175_175885


namespace airplane_total_luggage_weight_l175_175351

def num_people := 6
def bags_per_person := 5
def weight_per_bag := 50
def additional_bags := 90

def total_weight_people := num_people * bags_per_person * weight_per_bag
def total_weight_additional_bags := additional_bags * weight_per_bag

def total_luggage_weight := total_weight_people + total_weight_additional_bags

theorem airplane_total_luggage_weight : total_luggage_weight = 6000 :=
by
  sorry

end airplane_total_luggage_weight_l175_175351


namespace cleaning_time_l175_175974

-- Define the rate of cleaning for Bruce and Anne
variables (B A : ℝ)

-- Define the given conditions
def condition1 := B + A = 1/4
def condition2 := A = 1/12
def condition3 := ∀ {B A : ℝ}, B + 2 * A = 1/3 → B = 1/6 → A = 1/12 → B + 2 * A = 1/3

-- State the theorem to be proven
theorem cleaning_time (B A : ℝ) (h1 : condition1 B A) (h2 : condition2 A) : 
  (B + 2 * A = 1/3) → (1 / (B + 2 * A) = 3) :=
begin
  intros h3,
  have hA : A = 1 / 12 := h2,
  have hB : B = 1 / 6 := by linarith [h1, hA],
  field_simp [hB, hA] at h3,
  rw [h3],
  norm_num,
end

end cleaning_time_l175_175974


namespace minimum_value_of_xy_l175_175149

noncomputable def minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : ℝ :=
  if hmin : 4 * x + y + 12 = x * y then 36 else sorry

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : 
  minimum_value_xy x y hx hy h = 36 :=
sorry

end minimum_value_of_xy_l175_175149


namespace perfect_square_value_of_b_l175_175928

theorem perfect_square_value_of_b :
  (∃ b : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + b * b) = (11.98 + b)^2) →
  (∃ b : ℝ, b = 0.02) :=
sorry

end perfect_square_value_of_b_l175_175928


namespace greatest_common_divisor_l175_175694

theorem greatest_common_divisor (n : ℕ) (h1 : ∃ d : ℕ, d = gcd 180 n ∧ (∃ (l : List ℕ), l.length = 5 ∧ ∀ x : ℕ, x ∈ l → x ∣ d)) :
  ∃ x : ℕ, x = 27 :=
by
  sorry

end greatest_common_divisor_l175_175694


namespace smallest_n_mod_equiv_l175_175002

theorem smallest_n_mod_equiv (n : ℕ) (h : 0 < n ∧ 2^n ≡ n^5 [MOD 4]) : n = 2 :=
by
  sorry

end smallest_n_mod_equiv_l175_175002


namespace six_digit_palindromes_l175_175999

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l175_175999


namespace no_solution_frac_eq_l175_175644

theorem no_solution_frac_eq (k : ℝ) : (∀ x : ℝ, ¬(1 / (x + 1) = 3 * k / x)) ↔ (k = 0 ∨ k = 1 / 3) :=
by
  sorry

end no_solution_frac_eq_l175_175644


namespace sum_of_distinct_abc_eq_roots_l175_175060

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 * ((x + 2*y)^2 - y^2 + x - 1)

-- Main theorem statement
theorem sum_of_distinct_abc_eq_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : f a (b+c) = f b (c+a)) (h2 : f b (c+a) = f c (a+b)) :
  a + b + c = (1 + Real.sqrt 5) / 2 ∨ a + b + c = (1 - Real.sqrt 5) / 2 :=
sorry

end sum_of_distinct_abc_eq_roots_l175_175060


namespace range_of_a_l175_175630

theorem range_of_a (A B C : Set ℝ) (a : ℝ) :
  A = { x | -1 < x ∧ x < 4 } →
  B = { x | -5 < x ∧ x < (3 / 2) } →
  C = { x | (1 - 2 * a) < x ∧ x < (2 * a) } →
  (C ⊆ (A ∩ B)) →
  a ≤ (3 / 4) :=
by
  intros hA hB hC hSubset
  sorry

end range_of_a_l175_175630


namespace largest_of_five_consecutive_integers_l175_175419

theorem largest_of_five_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120) : n + 4 = 9 :=
sorry

end largest_of_five_consecutive_integers_l175_175419


namespace find_triangle_C_coordinates_find_triangle_area_l175_175751

noncomputable def triangle_C_coordinates (A B : (ℝ × ℝ)) (median_eq altitude_eq : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ C : ℝ × ℝ, C = (3, 1) ∧
    let A := (1,2)
    let B := (3, 4)
    let median_eq := (2, 1, -7)
    let altitude_eq := (2, -1, -2)
    true

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : Prop :=
  ∃ S : ℝ, S = 3 ∧
    let A := (1,2)
    let B := (3, 4)
    let C := (3, 1)
    true

theorem find_triangle_C_coordinates : triangle_C_coordinates (1,2) (3,4) (2, 1, -7) (2, -1, -2) :=
by { sorry }

theorem find_triangle_area : triangle_area (1,2) (3,4) (3,1) :=
by { sorry }

end find_triangle_C_coordinates_find_triangle_area_l175_175751


namespace smaller_two_digit_product_l175_175509

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end smaller_two_digit_product_l175_175509


namespace smaller_two_digit_product_l175_175514

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l175_175514


namespace f_of_f_eq_f_l175_175062

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem f_of_f_eq_f (x : ℝ) : f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 :=
by
  sorry

end f_of_f_eq_f_l175_175062


namespace smallest_positive_integer_divisible_by_8_11_15_l175_175743

-- Define what it means for a number to be divisible by another
def divisible_by (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

-- Define a function to find the least common multiple of three numbers
noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Statement of the theorem
theorem smallest_positive_integer_divisible_by_8_11_15 : 
  ∀ n : ℕ, (n > 0) ∧ divisible_by n 8 ∧ divisible_by n 11 ∧ divisible_by n 15 ↔ n = 1320 :=
sorry -- Proof is omitted

end smallest_positive_integer_divisible_by_8_11_15_l175_175743


namespace eagles_min_additional_wins_l175_175503

theorem eagles_min_additional_wins {N : ℕ} (eagles_initial_wins falcons_initial_wins : ℕ) (initial_games : ℕ)
  (total_games_won_fraction : ℚ) (required_fraction : ℚ) :
  eagles_initial_wins = 3 →
  falcons_initial_wins = 4 →
  initial_games = eagles_initial_wins + falcons_initial_wins →
  total_games_won_fraction = (3 + N) / (7 + N) →
  required_fraction = 9 / 10 →
  total_games_won_fraction = required_fraction →
  N = 33 :=
by
  sorry

end eagles_min_additional_wins_l175_175503


namespace dust_storm_acres_l175_175565

def total_acres : ℕ := 64013
def untouched_acres : ℕ := 522
def dust_storm_covered : ℕ := total_acres - untouched_acres

theorem dust_storm_acres :
  dust_storm_covered = 63491 := by
  sorry

end dust_storm_acres_l175_175565


namespace sequence_explicit_formula_l175_175650

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end sequence_explicit_formula_l175_175650


namespace inequality_proof_l175_175745

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  (1 / (a^2 + 1)) + (1 / (b^2 + 1)) + (1 / (c^2 + 1)) ≤ 9 / 4 :=
by
  sorry

end inequality_proof_l175_175745


namespace carson_seed_l175_175980

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l175_175980


namespace num_3_digit_multiples_l175_175285

def is_3_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999
def multiple_of (k n : Nat) : Prop := ∃ m : Nat, n = m * k

theorem num_3_digit_multiples (count_35_not_70 : Nat) (h : count_35_not_70 = 13) :
  let count_multiples_35 := (980 / 35) - (105 / 35) + 1
  let count_multiples_70 := (980 / 70) - (140 / 70) + 1
  count_multiples_35 - count_multiples_70 = count_35_not_70 := sorry

end num_3_digit_multiples_l175_175285


namespace total_wet_surface_area_l175_175109

def length : ℝ := 8
def width : ℝ := 4
def depth : ℝ := 1.25

theorem total_wet_surface_area : length * width + 2 * (length * depth) + 2 * (width * depth) = 62 :=
by
  sorry

end total_wet_surface_area_l175_175109


namespace coeff_z_in_third_eq_l175_175157

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq_l175_175157


namespace remaining_student_number_l175_175776

theorem remaining_student_number (s1 s2 s3 : ℕ) (h1 : s1 = 5) (h2 : s2 = 29) (h3 : s3 = 41) (N : ℕ) (hN : N = 48) :
  ∃ s4, s4 < N ∧ s4 ≠ s1 ∧ s4 ≠ s2 ∧ s4 ≠ s3 ∧ (s4 = 17) :=
by
  sorry

end remaining_student_number_l175_175776


namespace planting_cost_l175_175816

-- Define the costs of the individual items
def cost_of_flowers : ℝ := 9
def cost_of_clay_pot : ℝ := cost_of_flowers + 20
def cost_of_soil : ℝ := cost_of_flowers - 2
def cost_of_fertilizer : ℝ := cost_of_flowers + (0.5 * cost_of_flowers)
def cost_of_tools : ℝ := cost_of_clay_pot - (0.25 * cost_of_clay_pot)

-- Define the total cost
def total_cost : ℝ :=
  cost_of_flowers + cost_of_clay_pot + cost_of_soil + cost_of_fertilizer + cost_of_tools

-- The statement to prove
theorem planting_cost : total_cost = 80.25 :=
by
  sorry

end planting_cost_l175_175816


namespace clean_house_time_l175_175969

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l175_175969


namespace simplify_and_evaluate_expression_l175_175921

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -3) : (1 + 1/(x+1)) / ((x^2 + 4*x + 4) / (x+1)) = -1 :=
by
  sorry

end simplify_and_evaluate_expression_l175_175921


namespace clock_tick_intervals_l175_175856

theorem clock_tick_intervals (intervals_6: ℕ) (intervals_12: ℕ) (total_time_12: ℕ) (interval_time: ℕ):
  intervals_6 = 5 →
  intervals_12 = 11 →
  total_time_12 = 88 →
  interval_time = total_time_12 / intervals_12 →
  intervals_6 * interval_time = 40 :=
by
  intros h1 h2 h3 h4
  -- will continue proof here
  sorry

end clock_tick_intervals_l175_175856


namespace min_value_a_2b_l175_175423

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 3 / b = 1) :
  a + 2 * b = 7 + 2 * Real.sqrt 6 :=
sorry

end min_value_a_2b_l175_175423


namespace arithmetic_sequence_solution_geometric_sequence_solution_l175_175557

-- Problem 1: Arithmetic sequence
noncomputable def arithmetic_general_term (n : ℕ) : ℕ := 30 - 3 * n
noncomputable def arithmetic_sum_terms (n : ℕ) : ℝ := -1.5 * n^2 + 28.5 * n

theorem arithmetic_sequence_solution (n : ℕ) (a8 a10 : ℕ) (sequence : ℕ → ℝ) :
  a8 = 6 → a10 = 0 → (sequence n = arithmetic_general_term n) ∧ (sequence n = arithmetic_sum_terms n) ∧ (n = 9 ∨ n = 10) := 
sorry

-- Problem 2: Geometric sequence
noncomputable def geometric_general_term (n : ℕ) : ℝ := 2^(n-2)
noncomputable def geometric_sum_terms (n : ℕ) : ℝ := 2^(n-1) - 0.5

theorem geometric_sequence_solution (n : ℕ) (a1 a4 : ℝ) (sequence : ℕ → ℝ):
  a1 = 0.5 → a4 = 4 → (sequence n = geometric_general_term n) ∧ (sequence n = geometric_sum_terms n) := 
sorry

end arithmetic_sequence_solution_geometric_sequence_solution_l175_175557


namespace first_term_of_geometric_series_l175_175402

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end first_term_of_geometric_series_l175_175402


namespace find_other_sides_of_triangle_l175_175617

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l175_175617


namespace probability_abcd_16_l175_175221

theorem probability_abcd_16 :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let events := {t | (t ∈ outcomes × outcomes × outcomes × outcomes) ∧ (∃ a b c d, t = (a, b, c, d) ∧ a * b * c * d = 16)}
  @Prob (outcomes : Set ℕ) (fun x => discreteUniform outcomes) events = 7 / 1296 := 
sorry

end probability_abcd_16_l175_175221


namespace find_number_l175_175012

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l175_175012


namespace calculate_probability_l175_175381

-- Definitions
def total_coins : ℕ := 16  -- Total coins (3 pennies + 5 nickels + 8 dimes)
def draw_coins : ℕ := 8    -- Coins drawn
def successful_outcomes : ℕ := 321  -- Number of successful outcomes
def total_outcomes : ℕ := Nat.choose total_coins draw_coins  -- Total number of ways to choose draw_coins from total_coins

-- Question statement in Lean 4: Probability of drawing coins worth at least 75 cents
theorem calculate_probability : (successful_outcomes : ℝ) / (total_outcomes : ℝ) = 321 / 12870 := by
  sorry

end calculate_probability_l175_175381


namespace volleyball_team_selection_l175_175580

noncomputable def volleyball_squad_count (n m k : ℕ) : ℕ :=
  n * (Nat.choose m k)

theorem volleyball_team_selection :
  volleyball_squad_count 12 11 7 = 3960 :=
by
  sorry

end volleyball_team_selection_l175_175580


namespace find_x_in_interval_l175_175990

theorem find_x_in_interval (x : ℝ) 
  (h₁ : 4 ≤ (x + 1) / (3 * x - 7)) 
  (h₂ : (x + 1) / (3 * x - 7) < 9) : 
  x ∈ Set.Ioc (32 / 13) (29 / 11) := 
sorry

end find_x_in_interval_l175_175990


namespace curve_is_line_l175_175605

theorem curve_is_line (θ : ℝ) (hθ : θ = 5 * Real.pi / 6) : 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), r = 0 ↔
  (∃ p : ℝ × ℝ, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧
                p.1 * a + p.2 * b = 0) :=
sorry

end curve_is_line_l175_175605


namespace smallest_possible_value_of_M_l175_175352

theorem smallest_possible_value_of_M :
  ∃ (N M : ℕ), N > 0 ∧ M > 0 ∧ 
               ∃ (r_6 r_36 r_216 r_M : ℕ), 
               r_6 < 6 ∧ 
               r_6 < r_36 ∧ r_36 < 36 ∧ 
               r_36 < r_216 ∧ r_216 < 216 ∧ 
               r_216 < r_M ∧ 
               r_36 = (r_6 * r) ∧ 
               r_216 = (r_6 * r^2) ∧ 
               r_M = (r_6 * r^3) ∧ 
               Nat.mod N 6 = r_6 ∧ 
               Nat.mod N 36 = r_36 ∧ 
               Nat.mod N 216 = r_216 ∧ 
               Nat.mod N M = r_M ∧ 
               M = 2001 :=
sorry

end smallest_possible_value_of_M_l175_175352


namespace parallel_lines_find_m_l175_175878

theorem parallel_lines_find_m (m : ℝ) :
  (((3 + m) / 2 = 4 / (5 + m)) ∧ ((3 + m) / 2 ≠ (5 - 3 * m) / 8)) → m = -7 :=
sorry

end parallel_lines_find_m_l175_175878


namespace dealership_sales_l175_175588

theorem dealership_sales (sports_cars sedans suvs : ℕ) (h_sc : sports_cars = 35)
  (h_ratio_sedans : 5 * sedans = 8 * sports_cars) 
  (h_ratio_suvs : 5 * suvs = 3 * sports_cars) : 
  sedans = 56 ∧ suvs = 21 := by
  sorry

#print dealership_sales

end dealership_sales_l175_175588


namespace find_f_correct_l175_175505

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_con1 : ∀ x : ℝ, 2 * f x + f (-x) = 2 * x

theorem find_f_correct : ∀ x : ℝ, f x = 2 * x :=
by
  sorry

end find_f_correct_l175_175505


namespace sin_70_equals_1_minus_2a_squared_l175_175024

variable (a : ℝ)

theorem sin_70_equals_1_minus_2a_squared (h : Real.sin (10 * Real.pi / 180) = a) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * a^2 := 
sorry

end sin_70_equals_1_minus_2a_squared_l175_175024


namespace closest_point_on_plane_l175_175017

theorem closest_point_on_plane 
  (x y z : ℝ) 
  (h : 4 * x - 3 * y + 2 * z = 40) 
  (h_closest : ∀ (px py pz : ℝ), (4 * px - 3 * py + 2 * pz = 40) → dist (px, py, pz) (3, 1, 4) ≥ dist (x, y, z) (3, 1, 4)) :
  (x, y, z) = (139/19, -58/19, 86/19) :=
sorry

end closest_point_on_plane_l175_175017


namespace functional_eq_solution_l175_175263

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (x) ^ 2 + f (y)) = x * f (x) + y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := sorry

end functional_eq_solution_l175_175263


namespace cheap_feed_amount_l175_175689

theorem cheap_feed_amount (x y : ℝ) (h1 : x + y = 27) (h2 : 0.17 * x + 0.36 * y = 7.02) : 
  x = 14.21 :=
sorry

end cheap_feed_amount_l175_175689


namespace sum_squares_6_to_14_l175_175218

def sum_of_squares (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_squares_6_to_14 :
  (sum_of_squares 14) - (sum_of_squares 5) = 960 :=
by
  sorry

end sum_squares_6_to_14_l175_175218


namespace geometric_series_first_term_l175_175408

theorem geometric_series_first_term (r a s : ℝ) (h₁ : r = 1 / 4) (h₂ : s = 80) (h₃ : s = a / (1 - r)) : a = 60 :=
by
  rw [h₁] at h₃
  rw [h₂] at h₃
  norm_num at h₃
  linarith

# Examples utilized:
-- r : common ratio
-- a : first term of the series
-- s : sum of the series
-- h₁ : condition that the common ratio is 1/4
-- h₂ : condition that the sum is 80
-- h₃ : condition representing the formula for the sum of an infinite geometric series

end geometric_series_first_term_l175_175408


namespace original_price_l175_175078

theorem original_price (x : ℝ) (h : x * (1 / 8) = 8) : x = 64 := by
  -- To be proved
  sorry

end original_price_l175_175078


namespace polynomial_abc_l175_175451

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l175_175451


namespace tan_3theta_l175_175287

theorem tan_3theta (θ : ℝ) (h : Real.tan θ = 3 / 4) : Real.tan (3 * θ) = -12.5 :=
sorry

end tan_3theta_l175_175287


namespace min_variance_l175_175876

/--
Given a sample x, 1, y, 5 with an average of 2,
prove that the minimum value of the variance of this sample is 3.
-/
theorem min_variance (x y : ℝ) 
  (h_avg : (x + 1 + y + 5) / 4 = 2) :
  3 ≤ (1 / 4) * ((x - 2) ^ 2 + (y - 2) ^ 2 + (1 - 2) ^ 2 + (5 - 2) ^ 2) :=
sorry

end min_variance_l175_175876


namespace value_op_and_add_10_l175_175202

def op_and (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem value_op_and_add_10 : op_and 8 5 + 10 = 49 :=
by
  sorry

end value_op_and_add_10_l175_175202


namespace lower_bound_third_inequality_l175_175899

theorem lower_bound_third_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 8 > x ∧ x > 0)
  (h4 : x + 1 < 9) :
  x = 7 → ∃ l < 7, ∀ y, l < y ∧ y < 9 → y = x := 
sorry

end lower_bound_third_inequality_l175_175899


namespace variance_X_binomial_l175_175177

theorem variance_X_binomial :
  ∀ σ : ℝ, ∀ (X : ℝ → Prop), (∀ x, X x → x ∼ normal 90 σ^2)
  ∧ (P (λ x, x < 70) 0.2)
  ∧ (∀ x, P (λ x, 90 ≤ x ∧ x ≤ 110) 0.3) →
  (∃ (X : ℕ → Prop), ∃ n : ℕ, n = 10 ∧ X 0.3 →
  ∀ p: ℝ, ∀ k: ℕ, X k ∼ binomial n 0.3 ∧ variance X = 2.1) :=
by sorry

end variance_X_binomial_l175_175177


namespace intervals_of_monotonicity_and_min_value_l175_175886

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem intervals_of_monotonicity_and_min_value : 
  (∀ x, (x < -1 → f x < f (x + 0.0001)) ∧ (x > -1 ∧ x < 3 → f x > f (x + 0.0001)) ∧ (x > 3 → f x < f (x + 0.0001))) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≥ f 2) :=
by
  sorry

end intervals_of_monotonicity_and_min_value_l175_175886


namespace donovan_points_needed_l175_175138

-- Definitions based on conditions
def average_points := 26
def games_played := 15
def total_games := 20
def goal_average := 30

-- Assertion
theorem donovan_points_needed :
  let total_points_needed := goal_average * total_games
  let points_already_scored := average_points * games_played
  let remaining_games := total_games - games_played
  let remaining_points_needed := total_points_needed - points_already_scored
  let points_per_game_needed := remaining_points_needed / remaining_games
  points_per_game_needed = 42 :=
  by
    -- Proof skipped
    sorry

end donovan_points_needed_l175_175138


namespace heartsuit_ratio_l175_175909

def k : ℝ := 3

def heartsuit (n m : ℕ) : ℝ := k * n^3 * m^2

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 := 
by
  sorry

end heartsuit_ratio_l175_175909


namespace transform_negation_l175_175720

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l175_175720


namespace min_value_of_f_l175_175606

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  (∀ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) → f x y ≥ 12 / 35) ∧
  ∃ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) ∧ f x y = 12 / 35 :=
by
  sorry

end min_value_of_f_l175_175606


namespace distance_P_to_y_axis_l175_175279

-- Define the Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Condition: Point P with coordinates (-3, 5)
def P : Point := ⟨-3, 5⟩

-- Definition of distance from a point to the y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  abs p.x

-- Proof problem statement
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 := 
  sorry

end distance_P_to_y_axis_l175_175279


namespace total_students_in_school_l175_175716

noncomputable def small_school_students (boys girls : ℕ) (total_students : ℕ) : Prop :=
boys = 42 ∧ 
(girls : ℕ) = boys / 7 ∧
total_students = boys + girls

theorem total_students_in_school : small_school_students 42 6 48 :=
by
  sorry

end total_students_in_school_l175_175716


namespace initial_meals_is_70_l175_175385

-- Define variables and conditions
variables (A : ℕ)
def initial_meals_for_adults := A

-- Given conditions
def condition_1 := true  -- Group of 55 adults and some children (not directly used in proving A)
def condition_2 := true  -- Either a certain number of adults or 90 children (implicitly used in equation)
def condition_3 := (A - 21) * (90 / A) = 63  -- 21 adults have their meal, remaining food serves 63 children

-- The proof statement
theorem initial_meals_is_70 (h : (A - 21) * (90 / A) = 63) : A = 70 :=
sorry

end initial_meals_is_70_l175_175385


namespace find_a_2b_l175_175261

theorem find_a_2b 
  (a b : ℤ) 
  (h1 : a * b = -150) 
  (h2 : a + b = -23) : 
  a + 2 * b = -55 :=
sorry

end find_a_2b_l175_175261


namespace fluorescent_bulbs_switched_on_percentage_l175_175963

theorem fluorescent_bulbs_switched_on_percentage (I F : ℕ) (x : ℝ) (Inc_on F_on total_on Inc_on_ratio : ℝ) 
  (h1 : Inc_on = 0.3 * I) 
  (h2 : total_on = 0.7 * (I + F)) 
  (h3 : Inc_on_ratio = 0.08571428571428571) 
  (h4 : Inc_on_ratio = Inc_on / total_on) 
  (h5 : total_on = Inc_on + F_on) 
  (h6 : F_on = x * F) :
  x = 0.9 :=
sorry

end fluorescent_bulbs_switched_on_percentage_l175_175963


namespace cylindrical_container_volume_increase_l175_175133

theorem cylindrical_container_volume_increase (R H : ℝ)
  (initial_volume : ℝ)
  (x : ℝ) : 
  R = 10 ∧ H = 5 ∧ initial_volume = π * R^2 * H →
  π * (R + 2 * x)^2 * H = π * R^2 * (H + 3 * x) →
  x = 5 :=
by
  -- Given conditions
  intro conditions volume_equation
  obtain ⟨hR, hH, hV⟩ := conditions
  -- Simplifying and solving the resulting equation
  sorry

end cylindrical_container_volume_increase_l175_175133


namespace friends_cant_go_to_movies_l175_175485

theorem friends_cant_go_to_movies (total_friends : ℕ) (friends_can_go : ℕ) (H1 : total_friends = 15) (H2 : friends_can_go = 8) : (total_friends - friends_can_go) = 7 :=
by
  sorry

end friends_cant_go_to_movies_l175_175485


namespace geometric_sequence_not_sufficient_nor_necessary_l175_175746

theorem geometric_sequence_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) → 
  (¬ (q > 1 → ∀ n : ℕ, a n < a (n + 1))) ∧ (¬ (∀ n : ℕ, a n < a (n + 1) → q > 1)) :=
by
  sorry

end geometric_sequence_not_sufficient_nor_necessary_l175_175746


namespace total_miles_walked_l175_175683

-- Definition of the conditions
def num_islands : ℕ := 4
def miles_per_day_island1 : ℕ := 20
def miles_per_day_island2 : ℕ := 25
def days_per_island : ℚ := 1.5

-- Mathematically Equivalent Proof Problem
theorem total_miles_walked :
  let total_miles_island1 := 2 * (miles_per_day_island1 * days_per_island)
  let total_miles_island2 := 2 * (miles_per_day_island2 * days_per_island)
  total_miles_island1 + total_miles_island2 = 135 := by
  sorry

end total_miles_walked_l175_175683


namespace max_value_of_quadratic_in_interval_l175_175555

theorem max_value_of_quadratic_in_interval :
  ∃ x ∈ Icc (-1 : ℝ) 1, ∀ y ∈ Icc (-1 : ℝ) 1, f y ≤ f x ∧ f x = 4 := by
  let f := λ x : ℝ, x^2 + 4 * x + 1
  have h : ∀ x ∈ Icc (-1 : ℝ) 1, f x = 4 → x = 1 ∨ x = -1 := sorry
  use 1
  split
  { exact ⟨le_of_lt zero_lt_one, le_refl 1⟩ }
  { intros y hy
    have h' : y ∈ Icc (-1 : ℝ) 1 := hy
    sorry }


end max_value_of_quadratic_in_interval_l175_175555


namespace simplify_expression_l175_175922

theorem simplify_expression (y : ℝ) : 
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) = 8 * y^3 - 12 * y^2 + 20 * y - 24 := 
by
  sorry

end simplify_expression_l175_175922


namespace coin_flip_sequences_l175_175560

theorem coin_flip_sequences : 
  let flips := 10
  let choices := 2
  let total_sequences := choices ^ flips
  total_sequences = 1024 :=
by
  sorry

end coin_flip_sequences_l175_175560


namespace imag_part_z_l175_175884

theorem imag_part_z {z : ℂ} (h : i * (z - 3) = -1 + 3 * i) : z.im = 1 :=
sorry

end imag_part_z_l175_175884


namespace range_of_x_l175_175773

theorem range_of_x (x : ℝ) (h : ∃ y : ℝ, y = (x - 3) ∧ y > 0) : x > 3 :=
sorry

end range_of_x_l175_175773


namespace initial_bacteria_count_l175_175504

theorem initial_bacteria_count (n : ℕ) : 
  (n * 4^10 = 4194304) → n = 4 :=
by
  sorry

end initial_bacteria_count_l175_175504


namespace max_volume_l175_175237

variable (x y z : ℝ) (V : ℝ)
variable (k : ℝ)

-- Define the constraint
def constraint := x + 2 * y + 3 * z = 180

-- Define the volume
def volume := x * y * z

-- The goal is to show that under the constraint, the maximum possible volume is 36000 cubic cm.
theorem max_volume :
  (∀ (x y z : ℝ) (h : constraint x y z), volume x y z ≤ 36000) :=
  sorry

end max_volume_l175_175237


namespace product_of_two_numbers_l175_175348

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l175_175348


namespace max_n_for_positive_sum_l175_175622

theorem max_n_for_positive_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith: ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2)
  (h_cond1: a 9 + a 12 < 0)
  (h_cond2: a 10 * a 11 < 0)
  : ∀ n, S 19 > 0 ∧ S 20 < 0 → n ≤ 19 :=
begin
  sorry
end

end max_n_for_positive_sum_l175_175622


namespace value_of_2a_minus_b_minus_4_l175_175167

theorem value_of_2a_minus_b_minus_4 (a b : ℝ) (h : 2 * a - b = 2) : 2 * a - b - 4 = -2 :=
by
  sorry

end value_of_2a_minus_b_minus_4_l175_175167


namespace det_scaled_matrix_l175_175872

theorem det_scaled_matrix (a b c d : ℝ) (h : a * d - b * c = 5) : 
  (3 * a) * (3 * d) - (3 * b) * (3 * c) = 45 :=
by 
  sorry

end det_scaled_matrix_l175_175872


namespace find_b_l175_175810

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  2 * x / (x^2 + b * x + 1)

noncomputable def f_inverse (y : ℝ) : ℝ :=
  (1 - y) / y

theorem find_b (b : ℝ) (h : ∀ x, f_inverse (f x b) = x) : b = 4 :=
sorry

end find_b_l175_175810


namespace how_many_large_glasses_l175_175804

theorem how_many_large_glasses (cost_small cost_large : ℕ) 
                               (total_money money_left change : ℕ) 
                               (num_small : ℕ) : 
  cost_small = 3 -> 
  cost_large = 5 -> 
  total_money = 50 -> 
  money_left = 26 ->
  change = 1 ->
  num_small = 8 ->
  (money_left - change) / cost_large = 5 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end how_many_large_glasses_l175_175804


namespace exists_digit_sum_divisible_by_11_l175_175074

-- Define a function to compute the sum of the digits of a natural number
def digit_sum (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

-- The main theorem to be proven
theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k) % 11 = 0) := 
sorry

end exists_digit_sum_divisible_by_11_l175_175074


namespace ratio_of_sides_l175_175651

theorem ratio_of_sides (a b : ℝ) (h1 : a + b = 3 * a) (h2 : a + b - Real.sqrt (a^2 + b^2) = (1 / 3) * b) : a / b = 1 / 2 :=
sorry

end ratio_of_sides_l175_175651


namespace roots_ratio_sum_l175_175317

theorem roots_ratio_sum (a b m : ℝ) 
  (m1 m2 : ℝ)
  (h_roots : a ≠ b ∧ b ≠ 0 ∧ m ≠ 0 ∧ a ≠ 0 ∧ 
    ∀ x : ℝ, m * (x^2 - 3 * x) + 2 * x + 7 = 0 → (x = a ∨ x = b)) 
  (h_ratio : (a / b) + (b / a) = 7 / 3)
  (h_m1_m2_eq : ((3 * m - 2) ^ 2) / (7 * m) - 2 = 7 / 3)
  (h_m_vieta : (3 * m - 2) ^ 2 - 27 * m * (91 / 3) = 0) :
  (m1 + m2 = 127 / 27) ∧ (m1 * m2 = 4 / 9) →
  ((m1 / m2) + (m2 / m1) = 47.78) :=
sorry

end roots_ratio_sum_l175_175317


namespace problem_statement_l175_175639

-- Define a : ℝ such that (a + 1/a)^3 = 7
variables (a : ℝ) (h : (a + 1/a)^3 = 7)

-- Goal: Prove that a^4 + 1/a^4 = 1519/81
theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 7) : a^4 + 1/a^4 = 1519 / 81 := 
sorry

end problem_statement_l175_175639


namespace principal_trebled_after_5_years_l175_175518

-- Definitions of the conditions
def original_simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100
def total_simple_interest (P R n T : ℕ) : ℕ := (P * R * n) / 100 + (3 * P * R * (T - n)) / 100

-- The theorem statement
theorem principal_trebled_after_5_years :
  ∀ (P R : ℕ), original_simple_interest P R 10 = 800 →
              total_simple_interest P R 5 10 = 1600 →
              5 = 5 :=
by
  intros P R h1 h2
  sorry

end principal_trebled_after_5_years_l175_175518


namespace find_n_l175_175832

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 26) : n = 5 :=
by {
  sorry,
}

end find_n_l175_175832


namespace product_of_two_numbers_l175_175347

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l175_175347


namespace students_chose_water_l175_175459

theorem students_chose_water (total_students : ℕ)
  (h1 : 75 * total_students / 100 = 90)
  (h2 : 25 * total_students / 100 = x) :
  x = 30 := 
sorry

end students_chose_water_l175_175459


namespace find_m_l175_175645

noncomputable def f (m : ℝ) (x : ℝ) := (x^2 + m * x) * Real.exp x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m 
  (a b : ℝ) 
  (h_interval : a = -3/2 ∧ b = 1)
  (h_decreasing : is_monotonically_decreasing_on_interval (f m) a b) :
  m = -3/2 := 
sorry

end find_m_l175_175645


namespace natural_number_property_l175_175601

theorem natural_number_property (N k : ℕ) (hk : k > 0)
    (h1 : 10^(k-1) ≤ N) (h2 : N < 10^k) (h3 : N * 10^(k-1) ≤ N^2) (h4 : N^2 ≤ N * 10^k) :
    N = 10^(k-1) := 
sorry

end natural_number_property_l175_175601


namespace calculate_division_l175_175977

theorem calculate_division :
  (- (3 / 4) - 5 / 9 + 7 / 12) / (- 1 / 36) = 26 := by
  sorry

end calculate_division_l175_175977


namespace second_cannibal_wins_l175_175234

/-- Define a data structure for the position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  deriving Inhabited, DecidableEq

/-- Check if two positions are adjacent in a legal move (vertical or horizontal) -/
def isAdjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1))

/-- Define the initial positions of the cannibals -/
def initialPositionFirstCannibal : Position := ⟨1, 1⟩
def initialPositionSecondCannibal : Position := ⟨8, 8⟩

/-- Define a move function for a cannibal (a valid move should keep it on the board) -/
def move (p : Position) (direction : String) : Position :=
  match direction with
  | "up"     => if p.y < 8 then ⟨p.x, p.y + 1⟩ else p
  | "down"   => if p.y > 1 then ⟨p.x, p.y - 1⟩ else p
  | "left"   => if p.x > 1 then ⟨p.x - 1, p.y⟩ else p
  | "right"  => if p.x < 8 then ⟨p.x + 1, p.y⟩ else p
  | _        => p

/-- Predicate determining if a cannibal can eat the other by moving to its position -/
def canEat (p1 p2 : Position) : Bool :=
  p1 = p2

/-- 
  Prove that the second cannibal will eat the first cannibal with the correct strategy. 
  We formalize the fact that with correct play, starting from the initial positions, 
  the second cannibal (initially at ⟨8, 8⟩) can always force a win.
-/
theorem second_cannibal_wins :
  ∀ (p1 p2 : Position), 
  p1 = initialPositionFirstCannibal →
  p2 = initialPositionSecondCannibal →
  (∃ strategy : (Position → String), ∀ positionFirstCannibal : Position, canEat (move p2 (strategy p2)) positionFirstCannibal) :=
by
  sorry

end second_cannibal_wins_l175_175234


namespace hyperbola_asymptotes_l175_175001

theorem hyperbola_asymptotes (x y : ℝ) (h : y^2 / 16 - x^2 / 9 = (1 : ℝ)) :
  ∃ (m : ℝ), (m = 4 / 3) ∨ (m = -4 / 3) :=
sorry

end hyperbola_asymptotes_l175_175001


namespace difference_between_number_and_its_3_5_l175_175266

theorem difference_between_number_and_its_3_5 (x : ℕ) (h : x = 155) :
  x - (3 / 5 : ℚ) * x = 62 := by
  sorry

end difference_between_number_and_its_3_5_l175_175266


namespace cleaning_time_with_doubled_an_speed_l175_175965

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l175_175965


namespace determine_k_l175_175627

noncomputable def f (x k : ℝ) : ℝ := -4 * x^3 + k * x

theorem determine_k : ∀ k : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x k ≤ 1) → k = 3 :=
by
  sorry

end determine_k_l175_175627


namespace trig_identity_l175_175822

theorem trig_identity :
  (Real.cos (80 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) + 
   Real.sin (80 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  sorry

end trig_identity_l175_175822


namespace sum_of_n_values_l175_175536

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l175_175536


namespace cleaning_time_if_anne_doubled_l175_175971

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l175_175971


namespace carson_seed_l175_175979

variable (s f : ℕ) -- Define the variables s and f as nonnegative integers

-- Conditions given in the problem
axiom h1 : s = 3 * f
axiom h2 : s + f = 60

-- The theorem to prove
theorem carson_seed : s = 45 :=
by
  -- Proof would go here
  sorry

end carson_seed_l175_175979


namespace total_age_proof_l175_175295

variable (K : ℕ) -- Kaydence's age
variable (T : ℕ) -- Total age of people in the gathering

def Kaydence_father_age : ℕ := 60
def Kaydence_mother_age : ℕ := Kaydence_father_age - 2
def Kaydence_brother_age : ℕ := Kaydence_father_age / 2
def Kaydence_sister_age : ℕ := 40
def elder_cousin_age : ℕ := Kaydence_brother_age + 2 * Kaydence_sister_age
def younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
def grandmother_age : ℕ := 3 * Kaydence_mother_age - 5

theorem total_age_proof (K : ℕ) : T = 525 + K :=
by 
  sorry

end total_age_proof_l175_175295


namespace value_of_a_plus_b_l175_175638

theorem value_of_a_plus_b (a b : ℝ) (h : (2 * a + 2 * b - 1) * (2 * a + 2 * b + 1) = 99) :
  a + b = 5 ∨ a + b = -5 :=
sorry

end value_of_a_plus_b_l175_175638


namespace solve_for_y_l175_175924

theorem solve_for_y (y : ℝ) (h : 7 - y = 12) : y = -5 := sorry

end solve_for_y_l175_175924


namespace find_number_l175_175013

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l175_175013


namespace transform_negation_l175_175721

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l175_175721


namespace six_digit_palindromes_count_l175_175997

theorem six_digit_palindromes_count :
  (∃ (a b c d : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9) →
  9 * 10 * 10 * 10 = 9000 :=
by
  sorry

end six_digit_palindromes_count_l175_175997


namespace part_a_part_b_l175_175669

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l175_175669


namespace jackie_break_duration_l175_175781

noncomputable def push_ups_no_breaks : ℕ := 30

noncomputable def push_ups_with_breaks : ℕ := 22

noncomputable def total_breaks : ℕ := 2

theorem jackie_break_duration :
  (5 * 6 - push_ups_with_breaks) * (10 / 5) / total_breaks = 8 := by
-- Given that
-- 1) Jackie does 5 push-ups in 10 seconds
-- 2) Jackie takes 2 breaks in one minute and performs 22 push-ups
-- We need to prove the duration of each break
sorry

end jackie_break_duration_l175_175781


namespace g_correct_l175_175862

-- Define the polynomials involved
def p1 (x : ℝ) : ℝ := 2 * x^5 + 4 * x^3 - 3 * x
def p2 (x : ℝ) : ℝ := 7 * x^3 + 5 * x - 2

-- Define g(x) as the polynomial we need to find
def g (x : ℝ) : ℝ := -2 * x^5 + 3 * x^3 + 8 * x - 2

-- Now, state the condition
def condition (x : ℝ) : Prop := p1 x + g x = p2 x

-- Prove the condition holds with the defined polynomials
theorem g_correct (x : ℝ) : condition x :=
by
  change p1 x + g x = p2 x
  sorry

end g_correct_l175_175862


namespace minimum_daily_production_to_avoid_losses_l175_175821

theorem minimum_daily_production_to_avoid_losses (x : ℕ) :
  (∀ x, (10 * x) ≥ (5 * x + 4000)) → (x ≥ 800) :=
sorry

end minimum_daily_production_to_avoid_losses_l175_175821


namespace ratio_of_blue_to_purple_beads_l175_175396

theorem ratio_of_blue_to_purple_beads :
  ∃ (B G : ℕ), 
    7 + B + G = 46 ∧ 
    G = B + 11 ∧ 
    B / 7 = 2 :=
by
  sorry

end ratio_of_blue_to_purple_beads_l175_175396


namespace employee_n_salary_l175_175211

theorem employee_n_salary (x : ℝ) (h : x + 1.2 * x = 583) : x = 265 := sorry

end employee_n_salary_l175_175211


namespace polygon_sides_eq_six_l175_175206

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l175_175206


namespace find_expression_l175_175086

theorem find_expression (E a : ℝ) 
  (h1 : (E + (3 * a - 8)) / 2 = 69) 
  (h2 : a = 26) : 
  E = 68 :=
sorry

end find_expression_l175_175086


namespace ratio_of_kits_to_students_l175_175068

theorem ratio_of_kits_to_students (art_kits students : ℕ) (h1 : art_kits = 20) (h2 : students = 10) : art_kits / Nat.gcd art_kits students = 2 ∧ students / Nat.gcd art_kits students = 1 := by
  sorry

end ratio_of_kits_to_students_l175_175068


namespace maggie_total_income_l175_175070

def total_income (h_tractor : ℕ) (r_office r_tractor : ℕ) :=
  let h_office := 2 * h_tractor
  (h_tractor * r_tractor) + (h_office * r_office)

theorem maggie_total_income :
  total_income 13 10 12 = 416 := 
  sorry

end maggie_total_income_l175_175070


namespace melanie_total_plums_l175_175914

-- Define the initial conditions
def melaniePlums : Float := 7.0
def samGavePlums : Float := 3.0

-- State the theorem to prove
theorem melanie_total_plums : melaniePlums + samGavePlums = 10.0 := 
by
  sorry

end melanie_total_plums_l175_175914


namespace solve_inequality_l175_175986

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end solve_inequality_l175_175986


namespace right_triangle_third_side_l175_175051

theorem right_triangle_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : c = Real.sqrt (7) ∨ c = 5) :
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
  sorry

end right_triangle_third_side_l175_175051


namespace time_for_B_and_C_l175_175851

variables (a b c : ℝ)

-- Conditions
axiom cond1 : a = (1 / 2) * b
axiom cond2 : b = 2 * c
axiom cond3 : a + b + c = 1 / 26
axiom cond4 : a + b = 1 / 13
axiom cond5 : a + c = 1 / 39

-- Statement to prove
theorem time_for_B_and_C (a b c : ℝ) (cond1 : a = (1 / 2) * b)
                                      (cond2 : b = 2 * c)
                                      (cond3 : a + b + c = 1 / 26)
                                      (cond4 : a + b = 1 / 13)
                                      (cond5 : a + c = 1 / 39) :
  (1 / (b + c)) = 104 / 3 :=
sorry

end time_for_B_and_C_l175_175851


namespace cleaning_time_if_anne_doubled_l175_175972

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l175_175972


namespace ratio_of_compositions_l175_175478

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem ratio_of_compositions :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 :=
by
  -- Proof will go here
  sorry

end ratio_of_compositions_l175_175478


namespace sector_area_l175_175828

theorem sector_area (theta : ℝ) (d : ℝ) (r : ℝ := d / 2) (circle_area : ℝ := π * r^2) 
    (sector_area : ℝ := (theta / 360) * circle_area) : 
  theta = 120 → d = 6 → sector_area = 3 * π :=
by
  intro htheta hd
  sorry

end sector_area_l175_175828


namespace sophomores_more_than_first_graders_l175_175210

def total_students : ℕ := 95
def first_graders : ℕ := 32
def second_graders : ℕ := total_students - first_graders

theorem sophomores_more_than_first_graders : second_graders - first_graders = 31 := by
  sorry

end sophomores_more_than_first_graders_l175_175210


namespace num_of_dogs_l175_175823

theorem num_of_dogs (num_puppies : ℕ) (dog_food_per_meal : ℕ) (dog_meals_per_day : ℕ) (total_food : ℕ)
  (h1 : num_puppies = 4)
  (h2 : dog_food_per_meal = 4)
  (h3 : dog_meals_per_day = 3)
  (h4 : total_food = 108)
  : ∃ (D : ℕ), num_puppies * (dog_food_per_meal / 2) * (dog_meals_per_day * 3) + D * (dog_food_per_meal * dog_meals_per_day) = total_food ∧ D = 3 :=
by
  sorry

end num_of_dogs_l175_175823


namespace average_age_inhabitants_Campo_Verde_l175_175097

theorem average_age_inhabitants_Campo_Verde
  (H M : ℕ)
  (ratio_h_m : H / M = 2 / 3)
  (avg_age_men : ℕ := 37)
  (avg_age_women : ℕ := 42) :
  ((37 * H + 42 * M) / (H + M) : ℕ) = 40 := 
sorry

end average_age_inhabitants_Campo_Verde_l175_175097


namespace sphere_cube_volume_ratio_l175_175216

theorem sphere_cube_volume_ratio (d a : ℝ) (h_d : d = 12) (h_a : a = 6) :
  let r := d / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cube := a^3
  V_sphere / V_cube = (4 * π) / 3 :=
by
  sorry

end sphere_cube_volume_ratio_l175_175216


namespace find_f2_l175_175031

-- Define the conditions
variable {f g : ℝ → ℝ} {a : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Assume g is an even function
axiom even_g : ∀ x : ℝ, g (-x) = g x

-- Condition given in the problem
axiom f_g_relation : ∀ x : ℝ, f x + g x = a^x - a^(-x) + 2

-- Condition that g(2) = a
axiom g_at_2 : g 2 = a

-- Condition for a
axiom a_cond : a > 0 ∧ a ≠ 1

-- Proof problem
theorem find_f2 : f 2 = 15 / 4 := by
  sorry

end find_f2_l175_175031


namespace average_speed_l175_175103

theorem average_speed (D : ℝ) (h1 : 0 < D) :
  let s1 := 60   -- speed from Q to B in miles per hour
  let s2 := 20   -- speed from B to C in miles per hour
  let d1 := 2 * D  -- distance from Q to B
  let d2 := D     -- distance from B to C
  let t1 := d1 / s1  -- time to travel from Q to B
  let t2 := d2 / s2  -- time to travel from B to C
  let total_distance := d1 + d2  -- total distance
  let total_time := t1 + t2   -- total time
  let average_speed := total_distance / total_time  -- average speed
  average_speed = 36 :=
by
  sorry

end average_speed_l175_175103


namespace good_divisors_n_l175_175911

def is_good_divisor (n d : ℕ) : Prop := d ∣ n ∧ d + 1 ∣ n

theorem good_divisors_n (n : ℕ) (h : 1 < n) :
  (∃ S : Finset ℕ, S.card > 0 ∧ (∀ d ∈ S, is_good_divisor n d)) →
  n = 2 ∨ n = 6 ∨ n = 12 := by
  sorry

end good_divisors_n_l175_175911


namespace area_of_rectangle_l175_175854

theorem area_of_rectangle (w l : ℕ) (hw : w = 10) (hl : l = 2) : (w * l) = 20 :=
by
  sorry

end area_of_rectangle_l175_175854


namespace red_ball_probability_l175_175901

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end red_ball_probability_l175_175901


namespace amount_after_2_years_l175_175548

noncomputable def amount_after_n_years (present_value : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate_of_increase)^years

theorem amount_after_2_years :
  amount_after_n_years 6400 (1/8) 2 = 8100 :=
by
  sorry

end amount_after_2_years_l175_175548


namespace correct_equation_l175_175468

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l175_175468


namespace final_amount_after_two_years_l175_175008

open BigOperators

/-- Given an initial amount A0 and a percentage increase p, calculate the amount after n years -/
def compound_increase (A0 : ℝ) (p : ℝ) (n : ℕ) : ℝ :=
  (A0 * (1 + p)^n)

theorem final_amount_after_two_years (A0 : ℝ) (p : ℝ) (A2 : ℝ) :
  A0 = 1600 ∧ p = 1 / 8 ∧ compound_increase 1600 (1 / 8) 2 = 2025 :=
  sorry

end final_amount_after_two_years_l175_175008


namespace geometric_progression_common_ratio_l175_175814

/--
If \( a_1, a_2, a_3 \) are terms of an arithmetic progression with common difference \( d \neq 0 \),
and the products \( a_1 a_2, a_2 a_3, a_3 a_1 \) form a geometric progression,
then the common ratio of this geometric progression is \(-2\).
-/
theorem geometric_progression_common_ratio (a₁ a₂ a₃ d : ℝ) (h₀ : d ≠ 0) (h₁ : a₂ = a₁ + d)
  (h₂ : a₃ = a₁ + 2 * d) (h₃ : (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) :
  (a₂ * a₃) / (a₁ * a₂) = -2 :=
by
  sorry

end geometric_progression_common_ratio_l175_175814


namespace value_of_expression_l175_175613

variable {a : Nat → Int}

def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n m : Nat, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_expression
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 :=
  sorry

end value_of_expression_l175_175613


namespace solution_set_of_inequality_l175_175685

variable (a b c : ℝ)

theorem solution_set_of_inequality 
  (h1 : a < 0)
  (h2 : b = a)
  (h3 : c = -2 * a)
  (h4 : ∀ x : ℝ, -2 < x ∧ x < 1 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, (x ≤ -1 / 2 ∨ x ≥ 1) ↔ cx^2 + ax + b ≥ 0 :=
sorry

end solution_set_of_inequality_l175_175685


namespace number_of_squares_l175_175489

def side_plywood : ℕ := 50
def side_square_1 : ℕ := 10
def side_square_2 : ℕ := 20
def total_cut_length : ℕ := 280

/-- Number of squares obtained given the side lengths of the plywood and the cut lengths -/
theorem number_of_squares (x y : ℕ) (h1 : 100 * x + 400 * y = side_plywood^2)
  (h2 : 40 * x + 80 * y = total_cut_length) : x + y = 16 :=
sorry

end number_of_squares_l175_175489


namespace major_axis_of_ellipse_l175_175508

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define the length of the major axis
def major_axis_length : ℝ := 8

-- The theorem to prove
theorem major_axis_of_ellipse : 
  (∀ x y : ℝ, ellipse_eq x y) → major_axis_length = 8 :=
by
  sorry

end major_axis_of_ellipse_l175_175508


namespace total_photos_newspaper_l175_175301

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l175_175301


namespace initial_pipes_count_l175_175324

theorem initial_pipes_count (n r : ℝ) 
  (h1 : n * r = 1 / 12) 
  (h2 : (n + 10) * r = 1 / 4) : 
  n = 5 := 
by 
  sorry

end initial_pipes_count_l175_175324


namespace garageHasWheels_l175_175236

-- Define the conditions
def bikeWheelsPerBike : Nat := 2
def bikesInGarage : Nat := 10

-- State the theorem to be proved
theorem garageHasWheels : bikesInGarage * bikeWheelsPerBike = 20 := by
  sorry

end garageHasWheels_l175_175236


namespace range_of_a_l175_175156

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l175_175156


namespace inequality_proof_l175_175322

-- Define the main theorem with the conditions
theorem inequality_proof 
  (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a = b ∧ b = c ∧ c = d) ↔ (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a)) := 
sorry

end inequality_proof_l175_175322


namespace find_x_l175_175632

  -- Definition of the vectors
  def a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
  def b : ℝ × ℝ := (2, 1)

  -- Condition that vectors are parallel
  def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

  -- Theorem statement
  theorem find_x (x : ℝ) (h : are_parallel (a x) b) : x = 5 :=
  sorry
  
end find_x_l175_175632


namespace problem_statement_l175_175655

variables {p q r s : ℝ}

theorem problem_statement 
  (h : (p - q) * (r - s) / (q - r) * (s - p) = 3 / 7) : 
  (p - r) * (q - s) / (p - q) * (r - s) = -4 / 3 :=
by sorry

end problem_statement_l175_175655


namespace base_six_product_correct_l175_175732

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end base_six_product_correct_l175_175732


namespace list_price_of_article_l175_175818

theorem list_price_of_article (P : ℝ) (h : 0.882 * P = 57.33) : P = 65 :=
by
  sorry

end list_price_of_article_l175_175818


namespace pages_in_first_chapter_l175_175845

theorem pages_in_first_chapter
  (total_pages : ℕ)
  (second_chapter_pages : ℕ)
  (first_chapter_pages : ℕ)
  (h1 : total_pages = 81)
  (h2 : second_chapter_pages = 68) :
  first_chapter_pages = 81 - 68 :=
sorry

end pages_in_first_chapter_l175_175845


namespace largest_four_digit_negative_congruent_to_1_pmod_17_l175_175104

theorem largest_four_digit_negative_congruent_to_1_pmod_17 :
  ∃ n : ℤ, 17 * n + 1 < -1000 ∧ 17 * n + 1 ≥ -9999 ∧ 17 * n + 1 ≡ 1 [ZMOD 17] := 
sorry

end largest_four_digit_negative_congruent_to_1_pmod_17_l175_175104


namespace dog_running_direction_undeterminable_l175_175384

/-- Given the conditions:
 1. A dog is tied to a tree with a nylon cord of length 10 feet.
 2. The dog runs from one side of the tree to the opposite side with the cord fully extended.
 3. The dog runs approximately 30 feet.
 Prove that it is not possible to determine the specific starting direction of the dog.
-/
theorem dog_running_direction_undeterminable (r : ℝ) (full_length : r = 10) (distance_ran : ℝ) (approx_distance : distance_ran = 30) : (
  ∀ (d : ℝ), d < 2 * π * r → ¬∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π ∧ (distance_ran = r * θ)
  ) :=
by
  sorry

end dog_running_direction_undeterminable_l175_175384


namespace valid_combinations_l175_175089

theorem valid_combinations :
  ∀ (x y z : ℕ), 
  10 ≤ x ∧ x ≤ 20 → 
  10 ≤ y ∧ y ≤ 20 →
  10 ≤ z ∧ z ≤ 20 →
  3 * x^2 - y^2 - 7 * z = 99 →
  (x, y, z) = (15, 10, 12) ∨ (x, y, z) = (16, 12, 11) ∨ (x, y, z) = (18, 15, 13) := 
by
  intros x y z hx hy hz h
  sorry

end valid_combinations_l175_175089


namespace sum_of_n_for_3n_minus_8_eq_5_l175_175532

theorem sum_of_n_for_3n_minus_8_eq_5 : 
  let S := {n | |3 * n - 8| = 5}
  in S.sum id = 16 / 3 :=
by
  sorry

end sum_of_n_for_3n_minus_8_eq_5_l175_175532


namespace question1_question2_l175_175626

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Problem 1: Prove the valid solution of x when f(x) = 3 and x ∈ [0, 4]
theorem question1 (h₀ : 0 ≤ 3) (h₁ : 4 ≥ 3) : 
  ∃ (x : ℝ), (f x = 3 ∧ 0 ≤ x ∧ x ≤ 4) → x = 3 :=
by
  sorry

-- Problem 2: Prove the range of f(x) when x ∈ [0, 4]
theorem question2 : 
  ∃ (a b : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 4 → a ≤ f x ∧ f x ≤ b) → a = -1 ∧ b = 8 :=
by
  sorry

end question1_question2_l175_175626


namespace max_perimeter_triangle_l175_175579

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l175_175579


namespace problem_27_integer_greater_than_B_over_pi_l175_175553

noncomputable def B : ℕ := 22

theorem problem_27_integer_greater_than_B_over_pi :
  Nat.ceil (B / Real.pi) = 8 := sorry

end problem_27_integer_greater_than_B_over_pi_l175_175553


namespace possible_number_of_students_l175_175850

theorem possible_number_of_students (n : ℕ) 
  (h1 : n ≥ 1) 
  (h2 : ∃ k : ℕ, 120 = 2 * n + 2 * k) :
  n = 58 ∨ n = 60 :=
sorry

end possible_number_of_students_l175_175850


namespace alpha_epsilon_time_difference_l175_175203

def B := 100
def M := 120
def A := B - 10

theorem alpha_epsilon_time_difference : M - A = 30 := by
  sorry

end alpha_epsilon_time_difference_l175_175203


namespace minimum_value_of_expression_l175_175752

theorem minimum_value_of_expression {k x1 x2 : ℝ} 
  (h1 : x1 + x2 = -2 * k)
  (h2 : x1 * x2 = k^2 + k + 3) : 
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 :=
sorry

end minimum_value_of_expression_l175_175752


namespace boxes_left_l175_175783

-- Define the initial number of boxes
def initial_boxes : ℕ := 10

-- Define the number of boxes sold
def boxes_sold : ℕ := 5

-- Define a theorem stating that the number of boxes left is 5
theorem boxes_left : initial_boxes - boxes_sold = 5 :=
by
  sorry

end boxes_left_l175_175783


namespace steven_shirts_l175_175326

theorem steven_shirts : 
  (∀ (S A B : ℕ), S = 4 * A ∧ A = 6 * B ∧ B = 3 → S = 72) := 
by
  intro S A B
  intro h
  cases h with h1 h2
  cases h2 with hA hB
  rw [hB, hA]
  sorry

end steven_shirts_l175_175326


namespace julia_ink_containers_l175_175315

-- Definitions based on conditions
def total_posters : Nat := 60
def posters_remaining : Nat := 45
def lost_containers : Nat := 1

-- Required to be proven statement
theorem julia_ink_containers : 
  (total_posters - posters_remaining) = 15 → 
  posters_remaining / 15 = 3 := 
by 
  sorry

end julia_ink_containers_l175_175315


namespace monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l175_175481

noncomputable def f (x m : ℝ) : ℝ := x - m * (x + 1) * Real.log (x + 1)

theorem monotonicity_intervals_m0 :
  ∀ x : ℝ, x > -1 → f x 0 = x - 0 * (x + 1) * Real.log (x + 1) ∧ f x 0 > 0 := 
sorry

theorem monotonicity_intervals_m_positive (m : ℝ) (hm : m > 0) :
  ∀ x : ℝ, x > -1 → 
  (f x m > f (x + e ^ ((1 - m) / m) - 1) m ∧ 
  f (x + e ^ ((1 - m) / m) - 1) m < f (x + e ^ ((1 - m) / m) - 1 + 1) m) :=
sorry

theorem intersection_points_m1 (t : ℝ) (hx_rng : -1 / 2 ≤ t ∧ t < 1) :
  (∃ x1 x2 : ℝ, x1 > -1/2 ∧ x1 ≤ 1 ∧ x2 > -1/2 ∧ x2 ≤ 1 ∧ f x1 1 = t ∧ f x2 1 = t) ↔ 
  (-1 / 2 + 1 / 2 * Real.log 2 ≤ t ∧ t < 0) :=
sorry

theorem inequality_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (1 + a) ^ b < (1 + b) ^ a :=
sorry

end monotonicity_intervals_m0_monotonicity_intervals_m_positive_intersection_points_m1_inequality_a_b_l175_175481


namespace ice_cream_cones_sixth_day_l175_175587

theorem ice_cream_cones_sixth_day (cones_day1 cones_day2 cones_day3 cones_day4 cones_day5 cones_day7 : ℝ)
  (mean : ℝ) (h1 : cones_day1 = 100) (h2 : cones_day2 = 92) 
  (h3 : cones_day3 = 109) (h4 : cones_day4 = 96) 
  (h5 : cones_day5 = 103) (h7 : cones_day7 = 105) 
  (h_mean : mean = 100.1) : 
  ∃ cones_day6 : ℝ, cones_day6 = 95.7 :=
by 
  sorry

end ice_cream_cones_sixth_day_l175_175587


namespace problem_solution_l175_175377

variables {p q r : ℝ}

theorem problem_solution (h1 : (p + q) * (q + r) * (r + p) / (p * q * r) = 24)
  (h2 : (p - 2 * q) * (q - 2 * r) * (r - 2 * p) / (p * q * r) = 10) :
  ∃ m n : ℕ, (m.gcd n = 1 ∧ (p/q + q/r + r/p = m/n) ∧ m + n = 39) :=
sorry

end problem_solution_l175_175377


namespace combined_age_in_ten_years_l175_175779

theorem combined_age_in_ten_years (B A: ℕ) (hA : A = 20) (h1: A + 10 = 2 * (B + 10)): 
  (A + 10) + (B + 10) = 45 := 
by
  sorry

end combined_age_in_ten_years_l175_175779


namespace k_value_and_set_exists_l175_175686

theorem k_value_and_set_exists
  (x1 x2 x3 x4 : ℚ)
  (h1 : (x1 + x2) / (x3 + x4) = -1)
  (h2 : (x1 + x3) / (x2 + x4) = -1)
  (h3 : (x1 + x4) / (x2 + x3) = -1)
  (hne : x1 ≠ x2 ∨ x1 ≠ x3 ∨ x1 ≠ x4 ∨ x2 ≠ x3 ∨ x2 ≠ x4 ∨ x3 ≠ x4) :
  ∃ (A B C : ℚ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ x1 = A ∧ x2 = B ∧ x3 = C ∧ x4 = -A - B - C := 
sorry

end k_value_and_set_exists_l175_175686


namespace sequence_formula_l175_175936

open Nat

def a : ℕ → ℤ
| 0     => 0  -- Defining a(0) though not used
| 1     => 1
| (n+2) => 3 * a (n+1) + 2^(n+2)

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 5 * 3^(n-1) - 2^(n+1) :=
by
  sorry

end sequence_formula_l175_175936


namespace calculate_meals_l175_175664

-- Given conditions
def meal_cost : ℕ := 7
def total_spent : ℕ := 21

-- The expected number of meals Olivia's dad paid for
def expected_meals : ℕ := 3

-- Proof statement
theorem calculate_meals : total_spent / meal_cost = expected_meals :=
by
  sorry
  -- Proof can be completed using arithmetic simplification.

end calculate_meals_l175_175664


namespace min_bottles_needed_l175_175229

theorem min_bottles_needed (num_people : ℕ) (exchange_rate : ℕ) (bottles_needed_per_person : ℕ) (total_bottles_purchased : ℕ):
  num_people = 27 → exchange_rate = 3 → bottles_needed_per_person = 1 → total_bottles_purchased = 18 → 
  ∀ n, n = num_people → (n / bottles_needed_per_person) = 27 ∧ (num_people * 2 / 3) = 18 :=
by
  intros
  sorry

end min_bottles_needed_l175_175229


namespace counting_indistinguishable_boxes_l175_175165

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l175_175165


namespace Yan_distance_ratio_l175_175946

theorem Yan_distance_ratio (d x : ℝ) (v : ℝ) (h1 : d > 0) (h2 : x > 0) (h3 : x < d)
  (h4 : 7 * (d - x) = x + d) : 
  x / (d - x) = 3 / 4 :=
by
  sorry

end Yan_distance_ratio_l175_175946


namespace product_of_two_numbers_l175_175350

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l175_175350


namespace trucks_initial_count_l175_175079

theorem trucks_initial_count (x : ℕ) (h : x - 13 = 38) : x = 51 :=
by sorry

end trucks_initial_count_l175_175079


namespace find_n_l175_175749

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - imaginary_unit) = 1 + n * imaginary_unit) : n = 1 :=
sorry

end find_n_l175_175749


namespace like_terms_sum_l175_175750

theorem like_terms_sum (n m : ℕ) 
  (h1 : n + 1 = 3) 
  (h2 : m - 1 = 3) : 
  m + n = 6 := 
  sorry

end like_terms_sum_l175_175750


namespace total_number_of_squares_l175_175492

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l175_175492


namespace twenty_four_is_75_percent_of_what_number_l175_175693

theorem twenty_four_is_75_percent_of_what_number :
  ∃ x : ℝ, 24 = (75 / 100) * x ∧ x = 32 :=
by {
  use 32,
  split,
  { norm_num },
  { norm_num }
} -- sorry

end twenty_four_is_75_percent_of_what_number_l175_175693


namespace accessory_factory_growth_l175_175117

theorem accessory_factory_growth (x : ℝ) :
  600 + 600 * (1 + x) + 600 * (1 + x) ^ 2 = 2180 :=
sorry

end accessory_factory_growth_l175_175117


namespace negation_p_equiv_l175_175805

noncomputable def negation_of_proposition_p : Prop :=
∀ m : ℝ, ¬ ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem negation_p_equiv (p : Prop) (h : p = ∃ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0) :
  ¬ p ↔ negation_of_proposition_p :=
by {
  sorry
}

end negation_p_equiv_l175_175805


namespace coin_tails_probability_l175_175169

theorem coin_tails_probability:
  let p := 0.5 in
  let n := 3 in
  let k := 2 in
  (Nat.choose n k) * (p^k) * ((1-p)^(n-k)) = 0.375 :=
by
  sorry

end coin_tails_probability_l175_175169


namespace complement_intersection_l175_175429

-- Definitions
def A : Set ℝ := { x | x^2 + x - 6 < 0 }
def B : Set ℝ := { x | x > 1 }

-- Stating the problem
theorem complement_intersection (x : ℝ) : x ∈ (Aᶜ ∩ B) ↔ x ∈ Set.Ici 2 :=
by sorry

end complement_intersection_l175_175429


namespace find_original_number_of_men_l175_175837

theorem find_original_number_of_men (x : ℕ) (h1 : x * 12 = (x - 6) * 14) : x = 42 :=
  sorry

end find_original_number_of_men_l175_175837


namespace kopecks_problem_l175_175806

theorem kopecks_problem (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b :=
sorry

end kopecks_problem_l175_175806


namespace rahul_share_l175_175494

theorem rahul_share :
  let total_payment := 370
  let bonus := 30
  let remaining_payment := total_payment - bonus
  let rahul_work_per_day := 1 / 3
  let rajesh_work_per_day := 1 / 2
  let ramesh_work_per_day := 1 / 4
  
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day + ramesh_work_per_day
  let rahul_share_of_work := rahul_work_per_day / total_work_per_day
  let rahul_payment := rahul_share_of_work * remaining_payment

  rahul_payment = 80 :=
by {
  sorry
}

end rahul_share_l175_175494


namespace vector_BC_calculation_l175_175444

/--
If \(\overrightarrow{AB} = (3, 6)\) and \(\overrightarrow{AC} = (1, 2)\),
then \(\overrightarrow{BC} = (-2, -4)\).
-/
theorem vector_BC_calculation (AB AC BC : ℤ × ℤ) 
  (hAB : AB = (3, 6))
  (hAC : AC = (1, 2)) : 
  BC = (-2, -4) := 
by
  sorry

end vector_BC_calculation_l175_175444


namespace balls_in_boxes_l175_175760

theorem balls_in_boxes : ∃ n : ℕ, n = 56 ∧
  (∀ k : ℕ, 0 < k →
    (∃ x : ℕ, x = k ∧ x = 5) →
    (∃ y : ℕ, y = 4)) :=
by
  existsi 56
  split
  { refl }
  intros k hk Hyk
  existsi 4
  split
  { assumption }
  assumption

end balls_in_boxes_l175_175760


namespace compute_nested_operations_l175_175095

def operation (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem compute_nested_operations :
  operation 5 (operation 6 (operation 7 (operation 8 9))) = 3588 / 587 :=
  sorry

end compute_nested_operations_l175_175095


namespace trivia_team_total_points_l175_175393

def totalPoints : Nat := 182

def points_member_A : Nat := 3 * 2
def points_member_B : Nat := 5 * 4 + 1 * 6
def points_member_C : Nat := 2 * 6
def points_member_D : Nat := 4 * 2 + 2 * 4
def points_member_E : Nat := 1 * 2 + 3 * 4
def points_member_F : Nat := 5 * 6
def points_member_G : Nat := 2 * 4 + 1 * 2
def points_member_H : Nat := 3 * 6 + 2 * 2
def points_member_I : Nat := 1 * 4 + 4 * 6
def points_member_J : Nat := 7 * 2 + 1 * 4

theorem trivia_team_total_points : 
  points_member_A + points_member_B + points_member_C + points_member_D + points_member_E + 
  points_member_F + points_member_G + points_member_H + points_member_I + points_member_J = totalPoints := 
by
  repeat { sorry }

end trivia_team_total_points_l175_175393


namespace gcd_three_digit_palindromes_l175_175361

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l175_175361


namespace total_photos_newspaper_l175_175304

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l175_175304


namespace whisker_ratio_l175_175744

theorem whisker_ratio 
  (p : ℕ) (c : ℕ) (h1 : p = 14) (h2 : c = 22) (s := c + 6) :
  s / p = 2 := 
by
  sorry

end whisker_ratio_l175_175744


namespace percentage_of_students_who_received_certificates_l175_175052

theorem percentage_of_students_who_received_certificates
  (total_boys : ℕ)
  (total_girls : ℕ)
  (perc_boys_certificates : ℝ)
  (perc_girls_certificates : ℝ)
  (h1 : total_boys = 30)
  (h2 : total_girls = 20)
  (h3 : perc_boys_certificates = 0.1)
  (h4 : perc_girls_certificates = 0.2)
  : (3 + 4) / (30 + 20) * 100 = 14 :=
by
  sorry

end percentage_of_students_who_received_certificates_l175_175052


namespace no_solutions_for_a_l175_175991

theorem no_solutions_for_a (a : ℝ) : (∀ x : ℝ, 5 * |x - 4 * a| + |x - a^2| + 4 * x - 4 * a ≠ 0) ↔ a ∈ set.Iio (-8) ∪ set.Ioi 0 := by
  sorry

end no_solutions_for_a_l175_175991


namespace polygon_sides_arithmetic_progression_l175_175819

theorem polygon_sides_arithmetic_progression 
  (n : ℕ) 
  (h1 : ∀ n, ∃ a_1, ∃ a_n, ∀ i, a_n = 172 ∧ (a_i = a_1 + (i - 1) * 4) ∧ (i ≤ n))
  (h2 : ∀ S, S = 180 * (n - 2)) 
  (h3 : ∀ S, S = n * ((172 - 4 * (n - 1) + 172) / 2)) 
  : n = 12 := 
by 
  sorry

end polygon_sides_arithmetic_progression_l175_175819


namespace machines_used_l175_175836

variable (R S : ℕ)

/-- 
  A company has two types of machines, type R and type S. 
  Operating at a constant rate, a machine of type R does a certain job in 36 hours, 
  and a machine of type S does the job in 9 hours. 
  If the company used the same number of each type of machine to do the job in 12 hours, 
  then the company used 15 machines of type R.
-/
theorem machines_used (hR : ∀ ⦃n⦄, n * (1 / 36) + n * (1 / 9) = (1 / 12)) :
  R = 15 := 
by 
  sorry

end machines_used_l175_175836


namespace can_capacity_l175_175550

theorem can_capacity (x : ℝ) (milk water : ℝ) (full_capacity : ℝ) : 
  5 * x = milk ∧ 
  3 * x = water ∧ 
  full_capacity = milk + water + 8 ∧ 
  (milk + 8) / water = 2 → 
  full_capacity = 72 := 
sorry

end can_capacity_l175_175550


namespace robin_gum_count_l175_175076

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (final_gum : ℝ) 
  (h1 : initial_gum = 18.0) (h2 : additional_gum = 44.0) : final_gum = 62.0 :=
by {
  sorry
}

end robin_gum_count_l175_175076


namespace six_digit_palindromes_count_l175_175993

def is_six_digit_palindrome (n : ℕ) : Prop :=
  let digits := Int.toDigits n in
  n ≥ 100000 ∧ n < 1000000 ∧ (digits = List.reverse digits)

def count_six_digit_palindromes : ℕ :=
  let valid_digits (a b c : ℕ) := a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 in
  (List.range 9) -- a can be 1 through 9, List.range 9 generates [0, 1, 2, ..., 8]
  .map (λ i, i+1) -- Now it generates [1, 2, ..., 9]
  .map (λ a, (List.range 10) -- b can be 0 through 9
    .map (λ b, (List.range 10) -- c can be 0 through 9
      .map (λ c, (List.range 10) -- d can be 0 through 9
        .filter (valid_digits a b c) -- only keep valid combinations
        .length) -- count combinations for a single a, b combination
      .sum)) -- sum over all c {for this b}
    .sum) -- sum over all b
  .sum -- sum over all a

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindromes_count_l175_175993


namespace price_per_rose_is_2_l175_175893

-- Definitions from conditions
def has_amount (total_dollars : ℕ) : Prop := total_dollars = 300
def total_roses (R : ℕ) : Prop := ∃ (j : ℕ) (i : ℕ), R / 3 = j ∧ R / 2 = i ∧ j + i = 125

-- Theorem stating the price per rose
theorem price_per_rose_is_2 (R : ℕ) : 
  has_amount 300 → total_roses R → 300 / R = 2 :=
sorry

end price_per_rose_is_2_l175_175893


namespace no_solution_exists_l175_175596

theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ¬(2 / a + 2 / b = 1 / (a + b)) :=
sorry

end no_solution_exists_l175_175596


namespace Jerry_needs_72_dollars_l175_175782

def action_figures_current : ℕ := 7
def action_figures_total : ℕ := 16
def cost_per_figure : ℕ := 8
def money_needed : ℕ := 72

theorem Jerry_needs_72_dollars : 
  (action_figures_total - action_figures_current) * cost_per_figure = money_needed :=
by
  sorry

end Jerry_needs_72_dollars_l175_175782


namespace find_values_l175_175147

theorem find_values (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 :=
by 
  sorry

end find_values_l175_175147


namespace MH_greater_than_MK_l175_175458

-- Defining the conditions: BH perpendicular to HK and BH = 2
def BH := 2

-- Defining the conditions: CK perpendicular to HK and CK = 5
def CK := 5

-- M is the midpoint of BC, which implicitly means MB = MC in length
def M_midpoint_BC (MB MC : ℝ) :=
  MB = MC

theorem MH_greater_than_MK (MB MC MH MK : ℝ) 
  (hM_midpoint : M_midpoint_BC MB MC)
  (hMH : MH^2 + BH^2 = MB^2)
  (hMK : MK^2 + CK^2 = MC^2) :
  MH > MK :=
by
  sorry

end MH_greater_than_MK_l175_175458


namespace correct_equation_l175_175467

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l175_175467


namespace sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l175_175056

theorem sqrt_three_is_irrational_and_infinite_non_repeating_decimal :
    ∀ r : ℝ, r = Real.sqrt 3 → ¬ ∃ (m n : ℤ), n ≠ 0 ∧ r = m / n := by
    sorry

end sqrt_three_is_irrational_and_infinite_non_repeating_decimal_l175_175056


namespace find_f_at_2_l175_175158

variable {R : Type} [Ring R]

def f (a b x : R) : R := a * x ^ 3 + b * x - 3

theorem find_f_at_2 (a b : R) (h : f a b (-2) = 7) : f a b 2 = -13 := 
by 
  have h₁ : f a b (-2) + f a b 2 = -6 := sorry
  have h₂ : f a b 2 = -6 - f a b (-2) := sorry
  rw [h₂, h]
  norm_num

end find_f_at_2_l175_175158


namespace c_plus_d_l175_175641

theorem c_plus_d (a b c d : ℝ) (h1 : a + b = 11) (h2 : b + c = 9) (h3 : a + d = 5) :
  c + d = 3 + b :=
by
  sorry

end c_plus_d_l175_175641


namespace verify_condition_C_l175_175944

variable (x y z : ℤ)

-- Given conditions
def condition_C : Prop := x = y ∧ y = z + 1

-- The theorem/proof problem
theorem verify_condition_C (h : condition_C x y z) : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 := 
by 
  sorry

end verify_condition_C_l175_175944


namespace sum_of_areas_is_858_l175_175323

def first_six_odd_squares : List ℕ := [1^2, 3^2, 5^2, 7^2, 9^2, 11^2]

def rectangle_area (width length : ℕ) : ℕ := width * length

def sum_of_areas : ℕ := (first_six_odd_squares.map (rectangle_area 3)).sum

theorem sum_of_areas_is_858 : sum_of_areas = 858 := 
by
  -- Our aim is to show that sum_of_areas is 858
  -- The proof will be developed here
  sorry

end sum_of_areas_is_858_l175_175323
