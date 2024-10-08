import Mathlib

namespace expression_evaluation_l78_78793

theorem expression_evaluation (p q : ℝ) (h : p / q = 4 / 5) : (25 / 7 + (2 * q - p) / (2 * q + p)) = 4 :=
by {
  sorry
}

end expression_evaluation_l78_78793


namespace daughter_weight_l78_78968

def main : IO Unit :=
  IO.println s!"The weight of the daughter is 50 kg."

theorem daughter_weight :
  ∀ (G D C : ℝ), G + D + C = 110 → D + C = 60 → C = (1/5) * G → D = 50 :=
by
  intros G D C h1 h2 h3
  sorry

end daughter_weight_l78_78968


namespace total_votes_l78_78390

-- Define the given conditions
def candidate_votes (V : ℝ) : ℝ := 0.35 * V
def rival_votes (V : ℝ) : ℝ := 0.35 * V + 1800

-- Prove the total number of votes cast
theorem total_votes (V : ℝ) (h : candidate_votes V + rival_votes V = V) : V = 6000 :=
by
  sorry

end total_votes_l78_78390


namespace accessories_per_doll_l78_78879

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end accessories_per_doll_l78_78879


namespace derek_dogs_l78_78262

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l78_78262


namespace binom_12_11_l78_78336

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l78_78336


namespace smallest_egg_count_l78_78096

theorem smallest_egg_count : ∃ n : ℕ, n > 100 ∧ n % 12 = 10 ∧ n = 106 :=
by {
  sorry
}

end smallest_egg_count_l78_78096


namespace problem_1_problem_2_problem_3_l78_78748

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : Real.tan α = -2)

theorem problem_1 : Real.sin (α + (π / 6)) = (2 * Real.sqrt 15 - Real.sqrt 5) / 10 := by
  sorry

theorem problem_2 : (2 * Real.cos ((π / 2) + α) - Real.cos (π - α)) / (Real.sin ((π / 2) - α) - 3 * Real.sin (π + α)) = 5 / 7 := by
  sorry

theorem problem_3 : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end problem_1_problem_2_problem_3_l78_78748


namespace system_of_equations_n_eq_1_l78_78899

theorem system_of_equations_n_eq_1 {x y n : ℝ} 
  (h₁ : 5 * x - 4 * y = n) 
  (h₂ : 3 * x + 5 * y = 8)
  (h₃ : x = y) : 
  n = 1 := 
by
  sorry

end system_of_equations_n_eq_1_l78_78899


namespace max_f_l78_78571

theorem max_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x : ℝ, (-1 < x) →  ∀ y : ℝ, (y > -1) → ((1 + y)^a - a*y ≤ 1) :=
sorry

end max_f_l78_78571


namespace rate_of_interest_is_12_percent_l78_78940

variables (P r : ℝ)
variables (A5 A8 : ℝ)

-- Given conditions: 
axiom A5_condition : A5 = 9800
axiom A8_condition : A8 = 12005
axiom simple_interest_5_year : A5 = P + 5 * P * r / 100
axiom simple_interest_8_year : A8 = P + 8 * P * r / 100

-- The statement we aim to prove
theorem rate_of_interest_is_12_percent : r = 12 := 
sorry

end rate_of_interest_is_12_percent_l78_78940


namespace Patrick_hours_less_than_twice_Greg_l78_78044

def J := 18
def G := J - 6
def total_hours := 50
def P : ℕ := sorry -- To be defined, we need to establish the proof later with the condition J + G + P = 50
def X : ℕ := sorry -- To be defined, we need to establish the proof later with the condition P = 2 * G - X

theorem Patrick_hours_less_than_twice_Greg : X = 4 := by
  -- Placeholder definitions for P and X based on the given conditions
  let P := total_hours - (J + G)
  let X := 2 * G - P
  sorry -- Proof details to be filled in

end Patrick_hours_less_than_twice_Greg_l78_78044


namespace paul_is_19_years_old_l78_78720

theorem paul_is_19_years_old
  (mark_age : ℕ)
  (alice_age : ℕ)
  (paul_age : ℕ)
  (h1 : mark_age = 20)
  (h2 : alice_age = mark_age + 4)
  (h3 : paul_age = alice_age - 5) : 
  paul_age = 19 := by 
  sorry

end paul_is_19_years_old_l78_78720


namespace large_box_count_l78_78850

variable (x y : ℕ)

theorem large_box_count (h₁ : x + y = 21) (h₂ : 120 * x + 80 * y = 2000) : x = 8 := by
  sorry

end large_box_count_l78_78850


namespace days_of_supply_l78_78015

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l78_78015


namespace correctly_calculated_value_l78_78457

theorem correctly_calculated_value : 
  ∃ x : ℝ, (x + 4 = 40) ∧ (x / 4 = 9) :=
sorry

end correctly_calculated_value_l78_78457


namespace verify_solution_l78_78798

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x - y = 9
def condition2 : Prop := 4 * x + 3 * y = 1

-- Proof problem statement
theorem verify_solution
  (h1 : condition1 x y)
  (h2 : condition2 x y) :
  x = 4 ∧ y = -5 :=
sorry

end verify_solution_l78_78798


namespace eval_composition_l78_78533

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 - 2

theorem eval_composition : f (g 2) = -7 := 
by {
  sorry
}

end eval_composition_l78_78533


namespace total_cleaning_time_is_100_l78_78056

def outsideCleaningTime : ℕ := 80
def insideCleaningTime : ℕ := outsideCleaningTime / 4
def totalCleaningTime : ℕ := outsideCleaningTime + insideCleaningTime

theorem total_cleaning_time_is_100 : totalCleaningTime = 100 := by
  sorry

end total_cleaning_time_is_100_l78_78056


namespace negative_expression_P_minus_Q_l78_78692

theorem negative_expression_P_minus_Q :
  ∀ (P Q R S T : ℝ), 
    P = -4.0 → 
    Q = -2.0 → 
    R = 0.2 → 
    S = 1.1 → 
    T = 1.7 → 
    P - Q < 0 := 
by 
  intros P Q R S T hP hQ hR hS hT
  rw [hP, hQ]
  sorry

end negative_expression_P_minus_Q_l78_78692


namespace calculate_f_at_5_l78_78550

noncomputable def g (y : ℝ) : ℝ := (1 / 2) * y^2

noncomputable def f (x y : ℝ) : ℝ := 2 * x^2 + g y

theorem calculate_f_at_5 (y : ℝ) (h1 : f 2 y = 50) (h2 : y = 2*Real.sqrt 21) :
  f 5 y = 92 :=
by
  sorry

end calculate_f_at_5_l78_78550


namespace distance_between_red_lights_in_feet_l78_78063

theorem distance_between_red_lights_in_feet :
  let inches_between_lights := 6
  let pattern := [2, 3]
  let foot_in_inches := 12
  let pos_3rd_red := 6
  let pos_21st_red := 51
  let number_of_gaps := pos_21st_red - pos_3rd_red
  let total_distance_in_inches := number_of_gaps * inches_between_lights
  let total_distance_in_feet := total_distance_in_inches / foot_in_inches
  total_distance_in_feet = 22 := by
  sorry

end distance_between_red_lights_in_feet_l78_78063


namespace complex_div_eq_l78_78790

open Complex

def z := 4 - 2 * I

theorem complex_div_eq :
  (z + I = 4 - I) →
  (z / (4 + 2 * I) = (3 - 4 * I) / 5) :=
by
  sorry

end complex_div_eq_l78_78790


namespace cars_on_wednesday_more_than_monday_l78_78971

theorem cars_on_wednesday_more_than_monday:
  let cars_tuesday := 25
  let cars_monday := 0.8 * cars_tuesday
  let cars_thursday := 10
  let cars_friday := 10
  let cars_saturday := 5
  let cars_sunday := 5
  let total_cars := 97
  ∃ (cars_wednesday : ℝ), cars_wednesday - cars_monday = 2 :=
by
  sorry

end cars_on_wednesday_more_than_monday_l78_78971


namespace min_value_of_x2_y2_z2_l78_78578

noncomputable def min_square_sum (x y z k : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_of_x2_y2_z2 (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = k) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (x y z k : ℝ), (x^3 + y^3 + z^3 - 3 * x * y * z = k ∧ k ≥ -1) -> min_square_sum x y z k ≥ min_val :=
by
  sorry

end min_value_of_x2_y2_z2_l78_78578


namespace cubic_difference_l78_78637

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l78_78637


namespace remainder_div_741147_6_l78_78824

theorem remainder_div_741147_6 : 741147 % 6 = 3 :=
by
  sorry

end remainder_div_741147_6_l78_78824


namespace algebraic_expression_evaluation_l78_78784

-- Given condition and goal statement
theorem algebraic_expression_evaluation (a b : ℝ) (h : a - 2 * b + 3 = 0) : 5 + 2 * b - a = 8 :=
by sorry

end algebraic_expression_evaluation_l78_78784


namespace first_tap_fill_time_l78_78891

theorem first_tap_fill_time (T : ℚ) :
  (∀ (second_tap_empty_time : ℚ), second_tap_empty_time = 8) →
  (∀ (combined_fill_time : ℚ), combined_fill_time = 40 / 3) →
  (1/T - 1/8 = 3/40) →
  T = 5 :=
by
  intros h1 h2 h3
  sorry

end first_tap_fill_time_l78_78891


namespace linda_needs_additional_batches_l78_78139

theorem linda_needs_additional_batches:
  let classmates := 24
  let cookies_per_classmate := 10
  let dozen := 12
  let cookies_per_batch := 4 * dozen
  let cookies_needed := classmates * cookies_per_classmate
  let chocolate_chip_batches := 2
  let oatmeal_raisin_batches := 1
  let cookies_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let remaining_cookies := cookies_needed - cookies_made
  let additional_batches := remaining_cookies / cookies_per_batch
  additional_batches = 2 :=
by
  sorry

end linda_needs_additional_batches_l78_78139


namespace factor_expression_l78_78094

variable (b : ℤ)

theorem factor_expression : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) :=
by
  sorry

end factor_expression_l78_78094


namespace xyz_value_l78_78508

theorem xyz_value (x y z : ℝ) (h1 : 2 * x + 3 * y + z = 13) 
                              (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := 
by 
  sorry

end xyz_value_l78_78508


namespace scarves_per_yarn_correct_l78_78140

def scarves_per_yarn (total_yarns total_scarves : ℕ) : ℕ :=
  total_scarves / total_yarns

theorem scarves_per_yarn_correct :
  scarves_per_yarn (2 + 6 + 4) 36 = 3 :=
by
  sorry

end scarves_per_yarn_correct_l78_78140


namespace sum_of_solutions_l78_78279

-- Define the polynomial equation and the condition
def equation (x : ℝ) : Prop := 3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

-- Sum of solutions for the given polynomial equation under the constraint
theorem sum_of_solutions :
  (∀ x : ℝ, equation x → x ≠ -3) →
  ∃ (a b : ℝ), equation a ∧ equation b ∧ a + b = 4 := 
by
  intros h
  sorry

end sum_of_solutions_l78_78279


namespace min_a2_plus_b2_l78_78639

theorem min_a2_plus_b2 (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a2_plus_b2_l78_78639


namespace exponent_calculation_l78_78894

theorem exponent_calculation :
  ((19 ^ 11) / (19 ^ 8) * (19 ^ 3) = 47015881) :=
by
  sorry

end exponent_calculation_l78_78894


namespace exist_odd_a_b_k_l78_78434

theorem exist_odd_a_b_k (m : ℤ) : 
  ∃ (a b k : ℤ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (k ≥ 0) ∧ (2 * m = a^19 + b^99 + k * 2^1999) :=
by {
  sorry
}

end exist_odd_a_b_k_l78_78434


namespace velocity_is_zero_at_t_equals_2_l78_78534

def displacement (t : ℝ) : ℝ := -2 * t^2 + 8 * t

theorem velocity_is_zero_at_t_equals_2 : (deriv displacement 2 = 0) :=
by
  -- The definition step from (a). 
  let v := deriv displacement
  -- This would skip the proof itself, as instructed.
  sorry

end velocity_is_zero_at_t_equals_2_l78_78534


namespace trigonometric_ratio_l78_78382

theorem trigonometric_ratio (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 :=
sorry

end trigonometric_ratio_l78_78382


namespace interest_rate_l78_78309

noncomputable def simple_interest (P r t: ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t: ℝ) : ℝ := P * (1 + r / 100) ^ t - P

theorem interest_rate (P r: ℝ) (h1: simple_interest P r 2 = 50) (h2: compound_interest P r 2 = 51.25) : r = 5 :=
by
  sorry

end interest_rate_l78_78309


namespace zilla_savings_deposit_l78_78144

-- Definitions based on problem conditions
def monthly_earnings (E : ℝ) : Prop :=
  0.07 * E = 133

def tax_deduction (E : ℝ) : ℝ :=
  E - 0.10 * E

def expenditure (earnings : ℝ) : ℝ :=
  133 +  0.30 * earnings + 0.20 * earnings + 0.12 * earnings

def savings_deposit (remaining_earnings : ℝ) : ℝ :=
  0.15 * remaining_earnings

-- The final proof statement
theorem zilla_savings_deposit (E : ℝ) (total_spent : ℝ) (earnings_after_tax : ℝ) (remaining_earnings : ℝ) : 
  monthly_earnings E →
  tax_deduction E = earnings_after_tax →
  expenditure earnings_after_tax = total_spent →
  remaining_earnings = earnings_after_tax - total_spent →
  savings_deposit remaining_earnings = 77.52 :=
by
  intros
  sorry

end zilla_savings_deposit_l78_78144


namespace a_5_is_31_l78_78071

/-- Define the sequence a_n recursively -/
def a : Nat → Nat
| 0        => 1
| (n + 1)  => 2 * a n + 1

/-- Prove that the 5th term in the sequence is 31 -/
theorem a_5_is_31 : a 5 = 31 := 
sorry

end a_5_is_31_l78_78071


namespace largest_possible_value_l78_78931

noncomputable def largest_log_expression (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) : ℝ := 
  Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b

theorem largest_possible_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) (h3 : a = b) : 
  largest_log_expression a b h1 h2 = 0 :=
by
  sorry

end largest_possible_value_l78_78931


namespace find_11th_place_l78_78659

def placement_problem (Amara Bindu Carlos Devi Eshan Farel: ℕ): Prop :=
  (Carlos + 5 = Amara) ∧
  (Bindu = Eshan + 3) ∧
  (Carlos = Devi + 2) ∧
  (Devi = 6) ∧
  (Eshan + 1 = Farel) ∧
  (Bindu + 4 = Amara) ∧
  (Farel = 9)

theorem find_11th_place (Amara Bindu Carlos Devi Eshan Farel: ℕ) 
  (h : placement_problem Amara Bindu Carlos Devi Eshan Farel) : 
  Eshan = 11 := 
sorry

end find_11th_place_l78_78659


namespace solve_for_r_l78_78258

theorem solve_for_r (r : ℚ) (h : 4 * (r - 10) = 3 * (3 - 3 * r) + 9) : r = 58 / 13 :=
by
  sorry

end solve_for_r_l78_78258


namespace ten_crates_probability_l78_78548

theorem ten_crates_probability (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) :
  let num_crates := 10
  let crate_dimensions := [3, 4, 6]
  let target_height := 41

  -- Definition of the generating function coefficients and constraints will be complex,
  -- so stating the specific problem directly.
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ m = 190 ∧ n = 2187 →
  let probability := (m : ℚ) / (n : ℚ)
  probability = (190 : ℚ) / 2187 := 
by
  sorry

end ten_crates_probability_l78_78548


namespace fishing_probability_correct_l78_78172

-- Definitions for probabilities
def P_sunny : ℝ := 0.3
def P_rainy : ℝ := 0.5
def P_cloudy : ℝ := 0.2

def P_fishing_given_sunny : ℝ := 0.7
def P_fishing_given_rainy : ℝ := 0.3
def P_fishing_given_cloudy : ℝ := 0.5

-- The total probability function
def P_fishing : ℝ :=
  P_sunny * P_fishing_given_sunny +
  P_rainy * P_fishing_given_rainy +
  P_cloudy * P_fishing_given_cloudy

theorem fishing_probability_correct : P_fishing = 0.46 :=
by 
  sorry -- Proof goes here

end fishing_probability_correct_l78_78172


namespace at_least_one_ge_one_l78_78029

theorem at_least_one_ge_one (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  let a := x1 / x2
  let b := x2 / x3
  let c := x3 / x1
  a + b + c ≥ 3 → (a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1) :=
by
  intros
  sorry

end at_least_one_ge_one_l78_78029


namespace used_more_brown_sugar_l78_78905

-- Define the amounts of sugar used
def brown_sugar : ℝ := 0.62
def white_sugar : ℝ := 0.25

-- Define the statement to prove
theorem used_more_brown_sugar : brown_sugar - white_sugar = 0.37 :=
by
  sorry

end used_more_brown_sugar_l78_78905


namespace quotient_product_larger_integer_l78_78095

theorem quotient_product_larger_integer
  (x y : ℕ)
  (h1 : y / x = 7 / 3)
  (h2 : x * y = 189)
  : y = 21 := 
sorry

end quotient_product_larger_integer_l78_78095


namespace Nigel_initial_amount_l78_78562

-- Defining the initial amount Olivia has
def Olivia_initial : ℕ := 112

-- Defining the amount left after buying the tickets
def amount_left : ℕ := 83

-- Defining the cost per ticket and the number of tickets bought
def cost_per_ticket : ℕ := 28
def number_of_tickets : ℕ := 6

-- Calculating the total cost of the tickets
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Calculating the total amount Olivia spent
def Olivia_spent : ℕ := Olivia_initial - amount_left

-- Defining the total amount they spent
def total_spent : ℕ := total_cost

-- Main theorem to prove that Nigel initially had $139
theorem Nigel_initial_amount : ∃ (n : ℕ), (n + Olivia_initial - Olivia_spent = total_spent) → n = 139 :=
by {
  sorry
}

end Nigel_initial_amount_l78_78562


namespace quadratic_has_negative_root_iff_l78_78520

theorem quadratic_has_negative_root_iff (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by
  sorry

end quadratic_has_negative_root_iff_l78_78520


namespace final_price_of_bicycle_l78_78026

def original_price : ℝ := 200
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.25

theorem final_price_of_bicycle :
  let first_sale_price := original_price - (first_discount_rate * original_price)
  let final_sale_price := first_sale_price - (second_discount_rate * first_sale_price)
  final_sale_price = 90 := by
  sorry

end final_price_of_bicycle_l78_78026


namespace log_equation_solution_l78_78955

theorem log_equation_solution (a b x : ℝ) (h : 5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) :
    b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5) :=
sorry

end log_equation_solution_l78_78955


namespace masha_lives_on_seventh_floor_l78_78080

/-- Masha lives in apartment No. 290, which is in the 4th entrance of a 17-story building.
The number of apartments is the same in all entrances of the building on all 17 floors; apartment numbers start from 1.
We need to prove that Masha lives on the 7th floor. -/
theorem masha_lives_on_seventh_floor 
  (n_apartments_per_floor : ℕ) 
  (total_floors : ℕ := 17) 
  (entrances : ℕ := 4) 
  (masha_apartment : ℕ := 290) 
  (start_apartment : ℕ := 1) 
  (h1 : (masha_apartment - start_apartment + 1) > 0) 
  (h2 : masha_apartment ≤ entrances * total_floors * n_apartments_per_floor)
  (h4 : masha_apartment > (entrances - 1) * total_floors * n_apartments_per_floor)  
   : ((masha_apartment - ((entrances - 1) * total_floors * n_apartments_per_floor) - 1) / n_apartments_per_floor) + 1 = 7 := 
by
  sorry

end masha_lives_on_seventh_floor_l78_78080


namespace eggs_divided_l78_78874

theorem eggs_divided (boxes : ℝ) (eggs_per_box : ℝ) (total_eggs : ℝ) :
  boxes = 2.0 → eggs_per_box = 1.5 → total_eggs = boxes * eggs_per_box → total_eggs = 3.0 :=
by
  intros
  sorry

end eggs_divided_l78_78874


namespace ordering_of_variables_l78_78617

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem ordering_of_variables 
  (a b c : ℝ)
  (ha : a - 2 = Real.log (a / 2))
  (hb : b - 3 = Real.log (b / 3))
  (hc : c - 3 = Real.log (c / 2))
  (ha_pos : 0 < a) (ha_lt_one : a < 1)
  (hb_pos : 0 < b) (hb_lt_one : b < 1)
  (hc_pos : 0 < c) (hc_lt_one : c < 1) :
  c < b ∧ b < a :=
sorry

end ordering_of_variables_l78_78617


namespace exists_equidistant_point_l78_78530

-- Define three points A, B, and C in 2D space
variables {A B C P: ℝ × ℝ}

-- Assume the points A, B, and C are not collinear
def not_collinear (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

-- Define the concept of a point being equidistant from three given points
def equidistant (P A B C : ℝ × ℝ) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

-- Define the intersection of the perpendicular bisectors of the sides of the triangle formed by A, B, and C
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- placeholder for the actual construction

-- The main theorem statement: If A, B, and C are not collinear, then there exists a unique point P that is equidistant from A, B, and C
theorem exists_equidistant_point (h: not_collinear A B C) :
  ∃! P, equidistant P A B C := 
sorry

end exists_equidistant_point_l78_78530


namespace total_money_from_tshirts_l78_78363

def num_tshirts_sold := 20
def money_per_tshirt := 215

theorem total_money_from_tshirts :
  num_tshirts_sold * money_per_tshirt = 4300 :=
by
  sorry

end total_money_from_tshirts_l78_78363


namespace train_length_is_400_l78_78125

-- Conditions from a)
def train_speed_kmph : ℕ := 180
def crossing_time_sec : ℕ := 8

-- The corresponding length in meters
def length_of_train : ℕ := 400

-- The problem statement to prove
theorem train_length_is_400 :
  (train_speed_kmph * 1000 / 3600) * crossing_time_sec = length_of_train := by
  -- Proof is skipped as per the requirement
  sorry

end train_length_is_400_l78_78125


namespace exterior_angle_BAC_l78_78315

-- Definitions for the problem conditions
def regular_nonagon_interior_angle :=
  140

def square_interior_angle :=
  90

-- The proof statement
theorem exterior_angle_BAC (regular_nonagon_interior_angle square_interior_angle : ℝ) : 
  regular_nonagon_interior_angle = 140 ∧ square_interior_angle = 90 -> 
  ∃ (BAC : ℝ), BAC = 130 :=
by
  sorry

end exterior_angle_BAC_l78_78315


namespace min_chocolates_for_most_l78_78936

theorem min_chocolates_for_most (a b c d : ℕ) (h : a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 50) : d ≥ 14 := sorry

end min_chocolates_for_most_l78_78936


namespace garden_length_l78_78513

theorem garden_length (w l : ℝ) (h1 : l = 2 + 3 * w) (h2 : 2 * l + 2 * w = 100) : l = 38 :=
sorry

end garden_length_l78_78513


namespace alan_total_cost_is_84_l78_78200

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l78_78200


namespace maximum_rectangle_area_l78_78384

theorem maximum_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 44) : 
  ∃ (l_max w_max : ℕ), l_max * w_max = 121 :=
by
  sorry

end maximum_rectangle_area_l78_78384


namespace dennis_rocks_left_l78_78150

-- Definitions based on conditions:
def initial_rocks : ℕ := 10
def rocks_eaten_by_fish (initial : ℕ) : ℕ := initial / 2
def rocks_spat_out_by_fish : ℕ := 2

-- Total rocks left:
def total_rocks_left (initial : ℕ) (spat_out : ℕ) : ℕ :=
  (rocks_eaten_by_fish initial) + spat_out

-- Statement to be proved:
theorem dennis_rocks_left : total_rocks_left initial_rocks rocks_spat_out_by_fish = 7 :=
by
  -- Conclusion by calculation (Proved in steps)
  sorry

end dennis_rocks_left_l78_78150


namespace find_ab_l78_78137

-- Define the statement to be proven
theorem find_ab (a b : ℕ) (h1 : (a + b) % 3 = 2)
                           (h2 : b % 5 = 3)
                           (h3 : (b - a) % 11 = 1) :
  10 * a + b = 23 := 
sorry

end find_ab_l78_78137


namespace eccentricity_of_ellipse_l78_78149

noncomputable def calculate_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_ellipse : 
  (calculate_eccentricity 5 4) = 3 / 5 :=
by
  sorry

end eccentricity_of_ellipse_l78_78149


namespace proof_problem_l78_78267

def f (x : ℤ) : ℤ := 3 * x + 5
def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_problem : 
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = (380 / 653) := 
  by 
    sorry

end proof_problem_l78_78267


namespace find_B_l78_78719

theorem find_B (A B C : ℝ) (h1 : A = B + C) (h2 : A + B = 1/25) (h3 : C = 1/35) : B = 1/175 :=
by
  sorry

end find_B_l78_78719


namespace three_digit_number_condition_l78_78509

theorem three_digit_number_condition (x y z : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 9) (h₁ : 0 ≤ y ∧ y ≤ 9) (h₂ : 0 ≤ z ∧ z ≤ 9)
(h₃ : 100 * x + 10 * y + z = 34 * (x + y + z)) : 
100 * x + 10 * y + z = 102 ∨ 100 * x + 10 * y + z = 204 ∨ 100 * x + 10 * y + z = 306 ∨ 100 * x + 10 * y + z = 408 :=
sorry

end three_digit_number_condition_l78_78509


namespace remainder_sum_mod9_l78_78134

theorem remainder_sum_mod9 :
  ((2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9) = 6 := 
by 
  sorry

end remainder_sum_mod9_l78_78134


namespace muffins_count_l78_78702

-- Lean 4 Statement
theorem muffins_count (doughnuts muffins : ℕ) (ratio_doughnuts_muffins : ℕ → ℕ → Prop)
  (h_ratio : ratio_doughnuts_muffins 5 1) (h_doughnuts : doughnuts = 50) :
  muffins = 10 :=
by
  sorry

end muffins_count_l78_78702


namespace arithmetic_sequence_proof_l78_78769

variable (n : ℕ)
variable (a_n S_n : ℕ → ℤ)

noncomputable def a : ℕ → ℤ := 48 - 8 * n
noncomputable def S : ℕ → ℤ := -4 * (n ^ 2) + 44 * n

axiom a_3 : a 3 = 24
axiom S_11 : S 11 = 0

theorem arithmetic_sequence_proof :
  a n = 48 - 8 * n ∧
  S n = -4 * n ^ 2 + 44 * n ∧
  ∃ n, S n = 120 ∧ (n = 5 ∨ n = 6) :=
by
  unfold a S
  sorry

end arithmetic_sequence_proof_l78_78769


namespace sin_double_angle_l78_78795

open Real

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l78_78795


namespace no_natural_numbers_for_squares_l78_78800

theorem no_natural_numbers_for_squares :
  ∀ x y : ℕ, ¬(∃ k m : ℕ, k^2 = x^2 + y ∧ m^2 = y^2 + x) :=
by sorry

end no_natural_numbers_for_squares_l78_78800


namespace largest_band_members_l78_78557

theorem largest_band_members :
  ∃ (r x : ℕ), r * x + 3 = 107 ∧ (r - 3) * (x + 2) = 107 ∧ r * x < 147 :=
sorry

end largest_band_members_l78_78557


namespace annual_income_before_tax_l78_78670

theorem annual_income_before_tax (I : ℝ) (h1 : 0.42 * I - 0.28 * I = 4830) : I = 34500 :=
sorry

end annual_income_before_tax_l78_78670


namespace find_fifth_term_l78_78780

noncomputable def geometric_sequence_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : ℝ :=
  a * r^4

theorem find_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : geometric_sequence_fifth_term a r h₁ h₂ = 2 := sorry

end find_fifth_term_l78_78780


namespace final_number_after_increase_l78_78863

-- Define the original number and the percentage increase
def original_number : ℕ := 70
def increase_percentage : ℝ := 0.50

-- Define the function to calculate the final number after the increase
def final_number : ℝ := original_number * (1 + increase_percentage)

-- The proof statement that the final number is 105
theorem final_number_after_increase : final_number = 105 :=
by
  sorry

end final_number_after_increase_l78_78863


namespace restore_triangle_ABC_l78_78605

-- let I be the incenter of triangle ABC
variable (I : Point)
-- let Ic be the C-excenter of triangle ABC
variable (I_c : Point)
-- let H be the foot of the altitude from vertex C to side AB
variable (H : Point)

-- Claim: Given I, I_c, H, we can recover the original triangle ABC
theorem restore_triangle_ABC (I I_c H : Point) : ExistsTriangleABC :=
sorry

end restore_triangle_ABC_l78_78605


namespace closest_pressure_reading_l78_78538

theorem closest_pressure_reading (x : ℝ) (h : 102.4 ≤ x ∧ x ≤ 102.8) :
    (|x - 102.5| > |x - 102.6| ∧ |x - 102.6| < |x - 102.7| ∧ |x - 102.6| < |x - 103.0|) → x = 102.6 :=
by
  sorry

end closest_pressure_reading_l78_78538


namespace intersection_eq_l78_78185

def A : Set ℝ := {x : ℝ | (x - 2) / (x + 3) ≤ 0 }
def B : Set ℝ := {x : ℝ | x ≤ 1 }

theorem intersection_eq : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ 1 } :=
sorry

end intersection_eq_l78_78185


namespace find_a_l78_78237

open Real

def are_perpendicular (l1 l2 : Real × Real × Real) : Prop :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  a1 * a2 + b1 * b2 = 0

theorem find_a (a : Real) :
  let l1 := (a + 2, 1 - a, -1)
  let l2 := (a - 1, 2 * a + 3, 2)
  are_perpendicular l1 l2 → a = 1 ∨ a = -1 :=
by
  intro h
  sorry

end find_a_l78_78237


namespace cash_calculation_l78_78161

theorem cash_calculation 
  (value_gold_coin : ℕ) (value_silver_coin : ℕ) 
  (num_gold_coins : ℕ) (num_silver_coins : ℕ) 
  (total_money : ℕ) : 
  value_gold_coin = 50 → 
  value_silver_coin = 25 → 
  num_gold_coins = 3 → 
  num_silver_coins = 5 → 
  total_money = 305 → 
  (total_money - (num_gold_coins * value_gold_coin + num_silver_coins * value_silver_coin) = 30) := 
by
  intros h1 h2 h3 h4 h5
  sorry

end cash_calculation_l78_78161


namespace no_2014_ambiguous_integer_exists_l78_78510

theorem no_2014_ambiguous_integer_exists :
  ∀ k : ℕ, (∃ m : ℤ, k^2 - 8056 = m^2) → (∃ n : ℤ, k^2 + 8056 = n^2) → false :=
by
  -- Proof is omitted as per the instructions
  sorry

end no_2014_ambiguous_integer_exists_l78_78510


namespace max_wrestlers_more_than_131_l78_78537

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end max_wrestlers_more_than_131_l78_78537


namespace tim_weekly_earnings_l78_78189

-- Definitions based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- The theorem that we need to prove
theorem tim_weekly_earnings :
  (tasks_per_day * pay_per_task * days_per_week : ℝ) = 720 :=
by
  sorry -- Skipping the proof

end tim_weekly_earnings_l78_78189


namespace correct_inequality_l78_78783

theorem correct_inequality (a b c d : ℝ)
    (hab : a > b) (hb0 : b > 0)
    (hcd : c > d) (hd0 : d > 0) :
    Real.sqrt (a / d) > Real.sqrt (b / c) :=
by
    sorry

end correct_inequality_l78_78783


namespace inverse_of_97_mod_98_l78_78322

theorem inverse_of_97_mod_98 : 97 * 97 ≡ 1 [MOD 98] :=
by
  sorry

end inverse_of_97_mod_98_l78_78322


namespace problem_l78_78049

theorem problem
: 15 * (1 / 17) * 34 = 30 := by
  sorry

end problem_l78_78049


namespace arc_length_of_sector_l78_78307

theorem arc_length_of_sector (θ r : ℝ) (h1 : θ = 120) (h2 : r = 2) : 
  (θ / 360) * (2 * Real.pi * r) = (4 * Real.pi) / 3 :=
by
  sorry

end arc_length_of_sector_l78_78307


namespace JohnsonsYield_l78_78594

def JohnsonYieldPerTwoMonths (J : ℕ) : Prop :=
  ∀ (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ),
    neighbor_hectares = 2 →
    neighbor_yield_per_hectare = 2 * J →
    total_yield_six_months = 1200 →
    3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months →
    J = 80

theorem JohnsonsYield
  (J : ℕ)
  (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ)
  (h1 : neighbor_hectares = 2)
  (h2 : neighbor_yield_per_hectare = 2 * J)
  (h3 : total_yield_six_months = 1200)
  (h4 : 3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months) :
  J = 80 :=
by
  sorry

end JohnsonsYield_l78_78594


namespace combine_terms_implies_mn_l78_78018

theorem combine_terms_implies_mn {m n : ℕ} (h1 : m = 2) (h2 : n = 3) : m ^ n = 8 :=
by
  -- We will skip the proof here
  sorry

end combine_terms_implies_mn_l78_78018


namespace geom_seq_a7_a10_sum_l78_78460

theorem geom_seq_a7_a10_sum (a_n : ℕ → ℝ) (q a1 : ℝ)
  (h_seq : ∀ n, a_n (n + 1) = a1 * (q ^ n))
  (h1 : a1 + a1 * q = 2)
  (h2 : a1 * (q ^ 2) + a1 * (q ^ 3) = 4) :
  a_n 7 + a_n 8 + a_n 9 + a_n 10 = 48 := 
sorry

end geom_seq_a7_a10_sum_l78_78460


namespace index_cards_per_student_l78_78241

theorem index_cards_per_student
    (periods_per_day : ℕ)
    (students_per_class : ℕ)
    (cost_per_pack : ℕ)
    (total_spent : ℕ)
    (cards_per_pack : ℕ)
    (total_packs : ℕ)
    (total_index_cards : ℕ)
    (total_students : ℕ)
    (index_cards_per_student : ℕ)
    (h1 : periods_per_day = 6)
    (h2 : students_per_class = 30)
    (h3 : cost_per_pack = 3)
    (h4 : total_spent = 108)
    (h5 : cards_per_pack = 50)
    (h6 : total_packs = total_spent / cost_per_pack)
    (h7 : total_index_cards = total_packs * cards_per_pack)
    (h8 : total_students = periods_per_day * students_per_class)
    (h9 : index_cards_per_student = total_index_cards / total_students) :
    index_cards_per_student = 10 := 
  by
    sorry

end index_cards_per_student_l78_78241


namespace greatest_divisor_540_180_under_60_l78_78259

theorem greatest_divisor_540_180_under_60 : ∃ d, d ∣ 540 ∧ d ∣ 180 ∧ d < 60 ∧ ∀ k, k ∣ 540 → k ∣ 180 → k < 60 → k ≤ d :=
by
  sorry

end greatest_divisor_540_180_under_60_l78_78259


namespace proof_of_calculation_l78_78918

theorem proof_of_calculation : (7^2 - 5^2)^4 = 331776 := by
  sorry

end proof_of_calculation_l78_78918


namespace halfway_fraction_between_l78_78978

theorem halfway_fraction_between (a b : ℚ) (h_a : a = 1/6) (h_b : b = 1/4) : (a + b) / 2 = 5 / 24 :=
by
  have h1 : a = (1 : ℚ) / 6 := h_a
  have h2 : b = (1 : ℚ) / 4 := h_b
  sorry

end halfway_fraction_between_l78_78978


namespace greatest_possible_perimeter_l78_78574

theorem greatest_possible_perimeter (x : ℤ) (hx1 : 3 * x > 17) (hx2 : 17 > x) : 
  (3 * x + 17 ≤ 65) :=
by
  have Hx : x ≤ 16 := sorry -- Derived from inequalities hx1 and hx2
  have Hx_ge_6 : x ≥ 6 := sorry -- Derived from integer constraint and hx1, hx2
  sorry -- Show 3 * x + 17 has maximum value 65 when x = 16

end greatest_possible_perimeter_l78_78574


namespace find_sale_in_fourth_month_l78_78036

variable (sale1 sale2 sale3 sale5 sale6 : ℕ)
variable (TotalSales : ℕ)
variable (AverageSales : ℕ)

theorem find_sale_in_fourth_month (h1 : sale1 = 6335)
                                   (h2 : sale2 = 6927)
                                   (h3 : sale3 = 6855)
                                   (h4 : sale5 = 6562)
                                   (h5 : sale6 = 5091)
                                   (h6 : AverageSales = 6500)
                                   (h7 : TotalSales = AverageSales * 6) :
  ∃ sale4, TotalSales = sale1 + sale2 + sale3 + sale4 + sale5 + sale6 ∧ sale4 = 7230 :=
by
  sorry

end find_sale_in_fourth_month_l78_78036


namespace min_value_of_square_sum_l78_78778

theorem min_value_of_square_sum (x y : ℝ) 
  (h1 : (x + 5) ^ 2 + (y - 12) ^ 2 = 14 ^ 2) : 
  x^2 + y^2 = 1 := 
sorry

end min_value_of_square_sum_l78_78778


namespace ratio_expression_l78_78204

theorem ratio_expression 
  (m n r t : ℚ)
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := 
by 
  sorry

end ratio_expression_l78_78204


namespace total_planks_l78_78744

-- Define the initial number of planks
def initial_planks : ℕ := 15

-- Define the planks Charlie got
def charlie_planks : ℕ := 10

-- Define the planks Charlie's father got
def father_planks : ℕ := 10

-- Prove the total number of planks
theorem total_planks : (initial_planks + charlie_planks + father_planks) = 35 :=
by sorry

end total_planks_l78_78744


namespace remainder_of_3_pow_19_div_10_l78_78590

def w : ℕ := 3 ^ 19

theorem remainder_of_3_pow_19_div_10 : w % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_div_10_l78_78590


namespace p_implies_q_l78_78868

theorem p_implies_q (x : ℝ) :
  (|2*x - 3| < 1) → (x*(x - 3) < 0) :=
by
  intros hp
  sorry

end p_implies_q_l78_78868


namespace problem_statement_l78_78061

variable {a b c : ℝ}

theorem problem_statement (h : a < b) (hc : c < 0) : ¬ (a * c < b * c) :=
by sorry

end problem_statement_l78_78061


namespace stamps_total_l78_78052

theorem stamps_total (x : ℕ) (a_initial : ℕ := 5 * x) (b_initial : ℕ := 4 * x)
                     (a_after : ℕ := a_initial - 5) (b_after : ℕ := b_initial + 5)
                     (h_ratio_initial : a_initial / b_initial = 5 / 4)
                     (h_ratio_final : a_after / b_after = 4 / 5) :
                     a_initial + b_initial = 45 :=
by
  sorry

end stamps_total_l78_78052


namespace trapezoid_PR_length_l78_78871

noncomputable def PR_length (PQ RS QS PR : ℝ) (angle_QSP angle_SRP : ℝ) : Prop :=
  PQ < RS ∧ 
  QS = 2 ∧ 
  angle_QSP = 30 ∧ 
  angle_SRP = 60 ∧ 
  RS / PQ = 7 / 3 ∧ 
  PR = 8 / 3

theorem trapezoid_PR_length (PQ RS QS PR : ℝ) 
  (angle_QSP angle_SRP : ℝ) 
  (h1 : PQ < RS) 
  (h2 : QS = 2) 
  (h3 : angle_QSP = 30) 
  (h4 : angle_SRP = 60) 
  (h5 : RS / PQ = 7 / 3) :
  PR = 8 / 3 := 
by
  sorry

end trapezoid_PR_length_l78_78871


namespace cos_C_values_l78_78996

theorem cos_C_values (sin_A : ℝ) (cos_B : ℝ) (cos_C : ℝ) 
  (h1 : sin_A = 4 / 5) 
  (h2 : cos_B = 12 / 13) 
  : cos_C = -16 / 65 ∨ cos_C = 56 / 65 :=
by
  sorry

end cos_C_values_l78_78996


namespace det_dilation_matrix_l78_78852

section DilationMatrixProof

def E : Matrix (Fin 3) (Fin 3) ℝ := !![5, 0, 0; 0, 5, 0; 0, 0, 5]

theorem det_dilation_matrix :
  Matrix.det E = 125 :=
by {
  sorry
}

end DilationMatrixProof

end det_dilation_matrix_l78_78852


namespace regular_polygon_sides_l78_78662

theorem regular_polygon_sides (n : ℕ) (h : (180 * (n - 2) = 135 * n)) : n = 8 := by
  sorry

end regular_polygon_sides_l78_78662


namespace common_point_of_geometric_progression_l78_78935

theorem common_point_of_geometric_progression (a b c x y : ℝ) (r : ℝ) 
  (h1 : b = a * r) (h2 : c = a * r^2) 
  (h3 : a * x + b * y = c) : 
  x = 1 / 2 ∧ y = -1 / 2 := 
sorry

end common_point_of_geometric_progression_l78_78935


namespace no_unique_symbols_for_all_trains_l78_78473

def proposition (a b c d : Prop) : Prop :=
  (¬a ∧  b ∧ ¬c ∧  d)
∨ ( a ∧ ¬b ∧ ¬c ∧ ¬d)

theorem no_unique_symbols_for_all_trains 
    (a b c d : Prop)
    (p : proposition a b c d)
    (s1 : ¬a ∧  b ∧ ¬c ∧  d)
    (s2 :  a ∧ ¬b ∧ ¬c ∧ ¬d) : 
    False :=
by {cases s1; cases s2; contradiction}

end no_unique_symbols_for_all_trains_l78_78473


namespace probability_bernardo_larger_l78_78727

-- Define the sets from which Bernardo and Silvia are picking numbers
def set_B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to calculate the probability as described in the problem statement
def bernardo_larger_probability : ℚ := sorry -- The step by step calculations will be inserted here

-- Main theorem stating what needs to be proved
theorem probability_bernardo_larger : bernardo_larger_probability = 61 / 80 := 
sorry

end probability_bernardo_larger_l78_78727


namespace sum_mod_13_l78_78837

theorem sum_mod_13 :
  (9023 % 13 = 5) → 
  (9024 % 13 = 6) → 
  (9025 % 13 = 7) → 
  (9026 % 13 = 8) → 
  ((9023 + 9024 + 9025 + 9026) % 13 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_mod_13_l78_78837


namespace length_of_bridge_l78_78010

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_hr : ℝ)
  (time_sec : ℝ)
  (h_train_length : length_of_train = 155)
  (h_train_speed : speed_km_hr = 45)
  (h_time : time_sec = 30) :
  ∃ (length_of_bridge : ℝ),
    length_of_bridge = 220 :=
by
  sorry

end length_of_bridge_l78_78010


namespace total_jellybeans_l78_78265

theorem total_jellybeans (G : ℕ) (H1 : G = 8 + 2) (H2 : ∀ O : ℕ, O = G - 1) : 
  8 + G + (G - 1) = 27 := 
by 
  sorry

end total_jellybeans_l78_78265


namespace no_integer_n_gt_1_satisfies_inequality_l78_78417

open Int

theorem no_integer_n_gt_1_satisfies_inequality :
  ∀ (n : ℤ), n > 1 → ¬ (⌊(Real.sqrt (↑n - 2) + 2 * Real.sqrt (↑n + 2))⌋ < ⌊Real.sqrt (9 * (↑n : ℝ) + 6)⌋) :=
by
  intros n hn
  sorry

end no_integer_n_gt_1_satisfies_inequality_l78_78417


namespace positive_integer_prime_condition_l78_78729

theorem positive_integer_prime_condition (n : ℕ) 
  (h1 : 0 < n)
  (h2 : ∀ (k : ℕ), k < n → Nat.Prime (4 * k^2 + n)) : 
  n = 3 ∨ n = 7 := 
sorry

end positive_integer_prime_condition_l78_78729


namespace quadratic_passes_through_point_l78_78656

theorem quadratic_passes_through_point (a b : ℝ) (h : a ≠ 0) (h₁ : ∃ y : ℝ, y = a * 1^2 + b * 1 - 1 ∧ y = 1) : a + b + 1 = 3 :=
by
  obtain ⟨y, hy1, hy2⟩ := h₁
  sorry

end quadratic_passes_through_point_l78_78656


namespace special_collection_books_l78_78075

theorem special_collection_books (loaned_books : ℕ) (returned_percentage : ℝ) (end_of_month_books : ℕ)
    (H1 : loaned_books = 160)
    (H2 : returned_percentage = 0.65)
    (H3 : end_of_month_books = 244) :
    let books_returned := returned_percentage * loaned_books
    let books_not_returned := loaned_books - books_returned
    let original_books := end_of_month_books + books_not_returned
    original_books = 300 :=
by
  sorry

end special_collection_books_l78_78075


namespace foxes_wolves_bears_num_l78_78944

-- Definitions and theorem statement
def num_hunters := 45
def num_rabbits := 2008
def rabbits_per_fox := 59
def rabbits_per_wolf := 41
def rabbits_per_bear := 40

theorem foxes_wolves_bears_num (x y z : ℤ) : 
  x + y + z = num_hunters → 
  rabbits_per_wolf * x + rabbits_per_fox * y + rabbits_per_bear * z = num_rabbits → 
  x = 18 ∧ y = 10 ∧ z = 17 :=
by 
  intro h1 h2 
  sorry

end foxes_wolves_bears_num_l78_78944


namespace sum_of_first_3n_terms_l78_78372

theorem sum_of_first_3n_terms (n : ℕ) (sn s2n s3n : ℕ) 
  (h1 : sn = 48) (h2 : s2n = 60)
  (h3 : s2n - sn = s3n - s2n) (h4 : 2 * (s2n - sn) = sn + (s3n - s2n)) :
  s3n = 36 := 
by {
  sorry
}

end sum_of_first_3n_terms_l78_78372


namespace proof_problem_l78_78576

theorem proof_problem (a b c : ℝ) (h1 : 4 * a - 2 * b + c > 0) (h2 : a + b + c < 0) : b^2 > a * c :=
sorry

end proof_problem_l78_78576


namespace num_terms_in_expansion_eq_3_pow_20_l78_78885

-- Define the expression 
def expr (x y : ℝ) := (1 + x + y) ^ 20

-- Statement of the problem
theorem num_terms_in_expansion_eq_3_pow_20 (x y : ℝ) : (3 : ℝ)^20 = (1 + x + y) ^ 20 :=
by sorry

end num_terms_in_expansion_eq_3_pow_20_l78_78885


namespace xyz_sum_eq_eleven_l78_78565

theorem xyz_sum_eq_eleven (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 :=
sorry

end xyz_sum_eq_eleven_l78_78565


namespace charlie_has_54_crayons_l78_78845

theorem charlie_has_54_crayons
  (crayons_Billie : ℕ)
  (crayons_Bobbie : ℕ)
  (crayons_Lizzie : ℕ)
  (crayons_Charlie : ℕ)
  (h1 : crayons_Billie = 18)
  (h2 : crayons_Bobbie = 3 * crayons_Billie)
  (h3 : crayons_Lizzie = crayons_Bobbie / 2)
  (h4 : crayons_Charlie = 2 * crayons_Lizzie) : 
  crayons_Charlie = 54 := 
sorry

end charlie_has_54_crayons_l78_78845


namespace train_speed_is_180_kmh_l78_78860

-- Defining the conditions
def train_length : ℕ := 1500  -- meters
def platform_length : ℕ := 1500  -- meters
def crossing_time : ℕ := 1  -- minute

-- Function to compute the speed in km/hr
def speed_in_km_per_hr (length : ℕ) (time : ℕ) : ℕ :=
  let distance := length + length
  let speed_m_per_min := distance / time
  let speed_km_per_hr := speed_m_per_min * 60 / 1000
  speed_km_per_hr

-- The main theorem we need to prove
theorem train_speed_is_180_kmh :
  speed_in_km_per_hr train_length crossing_time = 180 :=
by
  sorry

end train_speed_is_180_kmh_l78_78860


namespace segment_AC_length_l78_78567

-- Define segments AB and BC
def AB : ℝ := 4
def BC : ℝ := 3

-- Define segment AC in terms of the conditions given
def AC_case1 : ℝ := AB - BC
def AC_case2 : ℝ := AB + BC

-- The proof problem statement
theorem segment_AC_length : AC_case1 = 1 ∨ AC_case2 = 7 := by
  sorry

end segment_AC_length_l78_78567


namespace rectangle_new_area_l78_78858

theorem rectangle_new_area (original_area : ℝ) (new_length_factor : ℝ) (new_width_factor : ℝ) 
  (h1 : original_area = 560) (h2 : new_length_factor = 1.2) (h3 : new_width_factor = 0.85) : 
  new_length_factor * new_width_factor * original_area = 571 := 
by 
  sorry

end rectangle_new_area_l78_78858


namespace pills_per_week_l78_78111

theorem pills_per_week (hours_per_pill : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) 
(h1: hours_per_pill = 6) (h2: hours_per_day = 24) (h3: days_per_week = 7) :
(hours_per_day / hours_per_pill) * days_per_week = 28 :=
by
  sorry

end pills_per_week_l78_78111


namespace coords_of_A_l78_78035

theorem coords_of_A :
  ∃ (x y : ℝ), y = Real.exp x ∧ (Real.exp x = 1) ∧ y = 1 :=
by
  use 0, 1
  have hx : Real.exp 0 = 1 := Real.exp_zero
  have hy : 1 = Real.exp 0 := hx.symm
  exact ⟨hy, hx, rfl⟩

end coords_of_A_l78_78035


namespace chewbacca_gum_pieces_l78_78132

theorem chewbacca_gum_pieces (y : ℚ)
  (h1 : ∀ x : ℚ, x ≠ 0 → (15 - y) = 15 * (25 + 2 * y) / 25) :
  y = 5 / 2 :=
by
  sorry

end chewbacca_gum_pieces_l78_78132


namespace least_possible_number_l78_78461

theorem least_possible_number {x : ℕ} (h1 : x % 6 = 2) (h2 : x % 4 = 3) : x = 50 :=
sorry

end least_possible_number_l78_78461


namespace cuboid_height_l78_78067

theorem cuboid_height (l b A : ℝ) (hl : l = 10) (hb : b = 8) (hA : A = 480) :
  ∃ h : ℝ, A = 2 * (l * b + b * h + l * h) ∧ h = 320 / 36 := by
  sorry

end cuboid_height_l78_78067


namespace count_divisors_2022_2022_l78_78013

noncomputable def num_divisors_2022_2022 : ℕ :=
  let fac2022 := 2022
  let factor_triplets := [(2, 3, 337), (3, 337, 2), (2, 337, 3), (337, 2, 3), (337, 3, 2), (3, 2, 337)]
  factor_triplets.length

theorem count_divisors_2022_2022 :
  num_divisors_2022_2022 = 6 :=
  by {
    sorry
  }

end count_divisors_2022_2022_l78_78013


namespace eq_nine_l78_78745

theorem eq_nine (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : (x - y)^2 = 9 := by
  sorry

end eq_nine_l78_78745


namespace school_children_count_l78_78797

theorem school_children_count (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by sorry

end school_children_count_l78_78797


namespace number_of_ants_in_section_correct_l78_78202

noncomputable def ants_in_section := 
  let width_feet : ℝ := 600
  let length_feet : ℝ := 800
  let ants_per_square_inch : ℝ := 5
  let side_feet : ℝ := 200
  let feet_to_inches : ℝ := 12
  let side_inches := side_feet * feet_to_inches
  let area_section_square_inches := side_inches^2
  ants_per_square_inch * area_section_square_inches

theorem number_of_ants_in_section_correct :
  ants_in_section = 28800000 := 
by 
  unfold ants_in_section 
  sorry

end number_of_ants_in_section_correct_l78_78202


namespace gcd_153_119_l78_78959

theorem gcd_153_119 : Nat.gcd 153 119 = 17 :=
by
  sorry

end gcd_153_119_l78_78959


namespace smallest_interesting_number_l78_78512

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l78_78512


namespace arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l78_78162

-- Statement for Problem 1: Number of ways to draw five numbers forming an arithmetic progression
theorem arithmetic_progression_five_numbers :
  ∃ (N : ℕ), N = 968 :=
  sorry

-- Statement for Problem 2: Number of ways to draw four numbers forming an arithmetic progression with a fifth number being arbitrary
theorem arithmetic_progression_four_numbers :
  ∃ (N : ℕ), N = 111262 :=
  sorry

end arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l78_78162


namespace rhombus_area_l78_78040

theorem rhombus_area 
  (a b : ℝ)
  (side_length : ℝ)
  (diff_diag : ℝ)
  (h_side_len : side_length = Real.sqrt 89)
  (h_diff_diag : diff_diag = 6)
  (h_diag : a - b = diff_diag ∨ b - a = diff_diag)
  (h_side_eq : side_length = Real.sqrt (a^2 + b^2)) :
  (1 / 2 * a * b) * 4 = 80 :=
by
  sorry

end rhombus_area_l78_78040


namespace first_divisor_is_six_l78_78641

theorem first_divisor_is_six {d : ℕ} 
  (h1: (1394 - 14) % d = 0)
  (h2: (2535 - 1929) % d = 0)
  (h3: (40 - 34) % d = 0)
  : d = 6 :=
sorry

end first_divisor_is_six_l78_78641


namespace decreasing_interval_of_even_function_l78_78499

-- Defining the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := (k-2) * x^2 + (k-1) * x + 3

-- Defining the condition that f is an even function
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem decreasing_interval_of_even_function (k : ℝ) :
  isEvenFunction (f · k) → k = 1 ∧ ∀ x ≥ 0, f x k ≤ f 0 k :=
by
  sorry

end decreasing_interval_of_even_function_l78_78499


namespace value_of_a_minus_b_l78_78626

variables (a b : ℚ)

theorem value_of_a_minus_b (h1 : |a| = 5) (h2 : |b| = 2) (h3 : |a + b| = a + b) : a - b = 3 ∨ a - b = 7 :=
sorry

end value_of_a_minus_b_l78_78626


namespace constant_term_correct_l78_78453

theorem constant_term_correct:
    ∀ (a k n : ℤ), 
      (∀ x : ℤ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
      → a - n + k = 7 
      → n = -6 := 
by
    intros a k n h h2
    have h1 := h 0
    sorry

end constant_term_correct_l78_78453


namespace inequality_proof_l78_78126

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l78_78126


namespace sin_sum_less_than_zero_l78_78177

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ 0 < γ ∧ γ < Real.pi / 2

theorem sin_sum_less_than_zero (n : ℕ) :
  (∀ (α β γ : ℝ), is_acute_triangle α β γ → (Real.sin (n * α) + Real.sin (n * β) + Real.sin (n * γ) < 0)) ↔ n = 4 :=
by
  sorry

end sin_sum_less_than_zero_l78_78177


namespace lollipops_remainder_l78_78353

theorem lollipops_remainder :
  let total_lollipops := 8362
  let lollipops_per_package := 12
  total_lollipops % lollipops_per_package = 10 :=
by
  let total_lollipops := 8362
  let lollipops_per_package := 12
  sorry

end lollipops_remainder_l78_78353


namespace inequality_one_solution_inequality_two_solution_cases_l78_78645

-- Setting up the problem for the first inequality
theorem inequality_one_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 4} = {x : ℝ |  -x ^ 2 + 3 * x + 4 ≥ 0} :=
sorry

-- Setting up the problem for the second inequality with different cases of 'a'
theorem inequality_two_solution_cases (a : ℝ) :
  (a = 0 ∧ {x : ℝ | true} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a > 0 ∧ {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a < 0 ∧ {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0}) :=
sorry

end inequality_one_solution_inequality_two_solution_cases_l78_78645


namespace negation_prop_l78_78799

theorem negation_prop (x : ℝ) : (¬ (∀ x : ℝ, Real.exp x > x^2)) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) :=
by
  sorry

end negation_prop_l78_78799


namespace ratio_of_cows_sold_l78_78661

-- Condition 1: The farmer originally has 51 cows.
def original_cows : ℕ := 51

-- Condition 2: The farmer adds 5 new cows to the herd.
def new_cows : ℕ := 5

-- Condition 3: The farmer has 42 cows left after selling a portion of the herd.
def remaining_cows : ℕ := 42

-- Defining total cows after adding new cows
def total_cows_after_addition : ℕ := original_cows + new_cows

-- Defining cows sold
def cows_sold : ℕ := total_cows_after_addition - remaining_cows

-- The theorem states the ratio of 'cows sold' to 'total cows after addition' is 1 : 4
theorem ratio_of_cows_sold : (cows_sold : ℚ) / (total_cows_after_addition : ℚ) = 1 / 4 := by
  -- Proof would go here
  sorry


end ratio_of_cows_sold_l78_78661


namespace curve_is_line_l78_78017

def curve := {p : ℝ × ℝ | ∃ (θ : ℝ), (p.1 = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ
                                        ∧ p.2 = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ)}

-- Problem: Prove that the curve defined by the polar equation is a line.
theorem curve_is_line : ∀ (p : ℝ × ℝ), p ∈ curve → p.1 + p.2 = 1 :=
by
  -- The proof is omitted.
  sorry

end curve_is_line_l78_78017


namespace triangle_base_length_l78_78395

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 24) (h_height : height = 8) (h_area_formula : area = (base * height) / 2) :
  base = 6 :=
by
  sorry

end triangle_base_length_l78_78395


namespace determinant_matrices_equivalence_l78_78400

-- Define the problem as a Lean theorem statement
theorem determinant_matrices_equivalence (p q r s : ℝ) 
  (h : p * s - q * r = 3) : 
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 12 := 
by 
  sorry

end determinant_matrices_equivalence_l78_78400


namespace relationship_between_y_coordinates_l78_78016

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l78_78016


namespace find_a_l78_78722

theorem find_a (a b c : ℂ) (ha : a.re = a) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 6) : a = 1 :=
by
  sorry

end find_a_l78_78722


namespace product_xyz_one_l78_78236

theorem product_xyz_one (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) : x * y * z = 1 := 
by {
    sorry
}

end product_xyz_one_l78_78236


namespace initial_number_2008_l78_78992

theorem initial_number_2008 
  (numbers_on_blackboard : ℕ → Prop)
  (x : ℕ)
  (Ops : ∀ x, numbers_on_blackboard x → (numbers_on_blackboard (2 * x + 1) ∨ numbers_on_blackboard (x / (x + 2)))) 
  (initial_apearing : numbers_on_blackboard 2008) :
  numbers_on_blackboard 2008 = true :=
sorry

end initial_number_2008_l78_78992


namespace cheryl_material_used_l78_78671

noncomputable def total_material_needed : ℚ :=
  (5 / 11) + (2 / 3)

noncomputable def material_left : ℚ :=
  25 / 55

noncomputable def material_used : ℚ :=
  total_material_needed - material_left

theorem cheryl_material_used :
  material_used = 22 / 33 :=
by
  sorry

end cheryl_material_used_l78_78671


namespace largest_common_value_lt_1000_l78_78691

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l78_78691


namespace find_children_tickets_l78_78232

variable (A C S : ℝ)

theorem find_children_tickets 
  (h1 : A + C + S = 600)
  (h2 : 6 * A + 4.5 * C + 5 * S = 3250) :
  C = (350 - S) / 1.5 := 
sorry

end find_children_tickets_l78_78232


namespace reflected_light_ray_equation_l78_78078

-- Definitions for the points and line
structure Point := (x : ℝ) (y : ℝ)

-- Given points M and N
def M : Point := ⟨2, 6⟩
def N : Point := ⟨-3, 4⟩

-- Given line l
def l (p : Point) : Prop := p.x - p.y + 3 = 0

-- The target equation of the reflected light ray
def target_equation (p : Point) : Prop := p.x - 6 * p.y + 27 = 0

-- Statement to prove
theorem reflected_light_ray_equation :
  (∃ K : Point, (M.x = 2 ∧ M.y = 6) ∧ l (⟨K.x + (K.x - M.x), K.y + (K.y - M.y)⟩)
     ∧ (N.x = -3 ∧ N.y = 4)) →
  (∀ P : Point, target_equation P ↔ (P.x - 6 * P.y + 27 = 0)) := by
sorry

end reflected_light_ray_equation_l78_78078


namespace smallest_angle_measure_in_triangle_l78_78856

theorem smallest_angle_measure_in_triangle (a b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : c > 2 * Real.sqrt 2) :
  ∃ x : ℝ, x = 140 ∧ C < x :=
sorry

end smallest_angle_measure_in_triangle_l78_78856


namespace marcy_pets_cat_time_l78_78945

theorem marcy_pets_cat_time (P : ℝ) (h1 : P + (1/3)*P = 16) : P = 12 :=
by
  sorry

end marcy_pets_cat_time_l78_78945


namespace slope_of_perpendicular_line_l78_78957

theorem slope_of_perpendicular_line (m1 m2 : ℝ) : 
  (5*x - 2*y = 10) →  ∃ m2, m2 = (-2/5) :=
by sorry

end slope_of_perpendicular_line_l78_78957


namespace prime_gt_3_divides_exp_l78_78133

theorem prime_gt_3_divides_exp (p : ℕ) (hprime : Nat.Prime p) (hgt3 : p > 3) :
  42 * p ∣ 3^p - 2^p - 1 :=
sorry

end prime_gt_3_divides_exp_l78_78133


namespace total_balls_l78_78318

def num_white : ℕ := 50
def num_green : ℕ := 30
def num_yellow : ℕ := 10
def num_red : ℕ := 7
def num_purple : ℕ := 3

def prob_neither_red_nor_purple : ℝ := 0.9

theorem total_balls (T : ℕ) 
  (h : prob_red_purple = 1 - prob_neither_red_nor_purple) 
  (h_prob : prob_red_purple = (num_red + num_purple : ℝ) / (T : ℝ)) :
  T = 100 :=
by sorry

end total_balls_l78_78318


namespace cost_of_bag_l78_78449

variable (cost_per_bag : ℝ)
variable (chips_per_bag : ℕ := 24)
variable (calories_per_chip : ℕ := 10)
variable (total_calories : ℕ := 480)
variable (total_cost : ℝ := 4)

theorem cost_of_bag :
  (chips_per_bag * (total_calories / calories_per_chip / chips_per_bag) = (total_calories / calories_per_chip)) →
  (total_cost / (total_calories / (calories_per_chip * chips_per_bag))) = 2 :=
by
  sorry

end cost_of_bag_l78_78449


namespace heaviest_person_is_42_27_l78_78841

-- Define the main parameters using the conditions
def heaviest_person_weight (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real) (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) : Real :=
  let h := P 2 + 7.7
  h

-- State the theorem
theorem heaviest_person_is_42_27 (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real)
  (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) :
  heaviest_person_weight M P Q H L S = 42.27 :=
sorry

end heaviest_person_is_42_27_l78_78841


namespace gcd_polynomial_l78_78890

theorem gcd_polynomial {b : ℕ} (h : 570 ∣ b) : Nat.gcd (4*b^3 + 2*b^2 + 5*b + 95) b = 95 := 
sorry

end gcd_polynomial_l78_78890


namespace thousandths_place_digit_of_7_div_32_l78_78333

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l78_78333


namespace rolls_for_mode_of_two_l78_78649

theorem rolls_for_mode_of_two (n : ℕ) (p : ℚ := 1/6) (m0 : ℕ := 32) : 
  (n : ℚ) * p - (1 - p) ≤ m0 ∧ m0 ≤ (n : ℚ) * p + p ↔ 191 ≤ n ∧ n ≤ 197 := 
by
  sorry

end rolls_for_mode_of_two_l78_78649


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l78_78843

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l78_78843


namespace original_number_is_fraction_l78_78690

theorem original_number_is_fraction (x : ℚ) (h : 1 + 1/x = 7/3) : x = 3/4 :=
sorry

end original_number_is_fraction_l78_78690


namespace find_other_number_l78_78613

theorem find_other_number (y : ℕ) : Nat.lcm 240 y = 5040 ∧ Nat.gcd 240 y = 24 → y = 504 :=
by
  sorry

end find_other_number_l78_78613


namespace find_greater_number_l78_78377

theorem find_greater_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 6) (h3 : x * y = 216) (h4 : x > y) : x = 18 := 
sorry

end find_greater_number_l78_78377


namespace ratio_of_efficacy_l78_78676

-- Define original conditions
def original_sprigs_of_mint := 3
def green_tea_leaves_per_sprig := 2

-- Define new condition
def new_green_tea_leaves := 12

-- Calculate the number of sprigs of mint corresponding to the new green tea leaves in the new mud
def new_sprigs_of_mint := new_green_tea_leaves / green_tea_leaves_per_sprig

-- Statement of the theorem: ratio of the efficacy of new mud to original mud is 1:2
theorem ratio_of_efficacy : new_sprigs_of_mint = 2 * original_sprigs_of_mint :=
by
    sorry

end ratio_of_efficacy_l78_78676


namespace number_of_men_in_larger_group_l78_78028

-- Define the constants and conditions
def men1 := 36         -- men in the first group
def days1 := 18        -- days taken by the first group
def men2 := 108       -- men in the larger group (what we want to prove)
def days2 := 6         -- days taken by the second group

-- Given conditions as lean definitions
def total_work (men : Nat) (days : Nat) := men * days
def condition1 := (total_work men1 days1 = 648)
def condition2 := (total_work men2 days2 = 648)

-- Problem statement 
-- proving that men2 is 108
theorem number_of_men_in_larger_group : condition1 → condition2 → men2 = 108 :=
by
  intros
  sorry

end number_of_men_in_larger_group_l78_78028


namespace third_median_length_l78_78965

theorem third_median_length 
  (m_A m_B : ℝ) -- lengths of the first two medians
  (area : ℝ)   -- area of the triangle
  (h_median_A : m_A = 5) -- the first median is 5 inches
  (h_median_B : m_B = 8) -- the second median is 8 inches
  (h_area : area = 6 * Real.sqrt 15) -- the area of the triangle is 6√15 square inches
  : ∃ m_C : ℝ, m_C = Real.sqrt 31 := -- the length of the third median is √31
sorry

end third_median_length_l78_78965


namespace cooperative_payment_divisibility_l78_78623

theorem cooperative_payment_divisibility (T_old : ℕ) (N : ℕ) 
  (hN : N = 99 * T_old / 100) : 99 ∣ N :=
by
  sorry

end cooperative_payment_divisibility_l78_78623


namespace find_x10_l78_78107

theorem find_x10 (x : ℕ → ℝ) :
  x 1 = 1 ∧ x 2 = 1 ∧ (∀ n ≥ 2, x (n + 1) = (x n * x (n - 1)) / (x n + x (n - 1))) →
  x 10 = 1 / 55 :=
by sorry

end find_x10_l78_78107


namespace abs_neg_three_l78_78990

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l78_78990


namespace length_of_box_l78_78415

theorem length_of_box 
  (width height num_cubes length : ℕ)
  (h_width : width = 16)
  (h_height : height = 13)
  (h_cubes : num_cubes = 3120)
  (h_volume : length * width * height = num_cubes) :
  length = 15 :=
by
  sorry

end length_of_box_l78_78415


namespace arithmetic_sequence_a4_l78_78357

def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else if n = 2 then 4 else 2 + (n - 1) * 2

theorem arithmetic_sequence_a4 :
  a 4 = 8 :=
by {
  sorry
}

end arithmetic_sequence_a4_l78_78357


namespace trigonometric_identity_l78_78511

open Real

variable (α : ℝ)

theorem trigonometric_identity (h : tan (π - α) = 2) :
  (sin (π / 2 + α) + sin (π - α)) / (cos (3 * π / 2 + α) + 2 * cos (π + α)) = 1 / 4 :=
  sorry

end trigonometric_identity_l78_78511


namespace inverse_of_B_squared_l78_78673

theorem inverse_of_B_squared (B_inv : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B_inv = ![![3, -2], ![0, 5]]) : 
  (B_inv * B_inv) = ![![9, -16], ![0, 25]] :=
by
  sorry

end inverse_of_B_squared_l78_78673


namespace complex_quadrant_example_l78_78021

open Complex

def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_quadrant_example (z : ℂ) (h : (1 - I) * z = (1 + I) ^ 2) : in_second_quadrant z :=
by
  sorry

end complex_quadrant_example_l78_78021


namespace lcm_is_perfect_square_l78_78338

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l78_78338


namespace simplify_sqrt_neg_five_squared_l78_78325

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end simplify_sqrt_neg_five_squared_l78_78325


namespace jerusha_earnings_l78_78746

theorem jerusha_earnings (L : ℕ) (h1 : 5 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l78_78746


namespace sum_of_arithmetic_sequence_has_remainder_2_l78_78493

def arithmetic_sequence_remainder : ℕ := 
  let first_term := 1
  let common_difference := 6
  let last_term := 259
  -- Calculate number of terms
  let n := (last_term + 5) / common_difference
  -- Sum of remainders of each term when divided by 6
  let sum_of_remainders := n * 1
  -- The remainder when this sum is divided by 6
  sum_of_remainders % 6 
theorem sum_of_arithmetic_sequence_has_remainder_2 : 
  arithmetic_sequence_remainder = 2 := by 
  sorry

end sum_of_arithmetic_sequence_has_remainder_2_l78_78493


namespace find_omega_l78_78878

noncomputable def f (x : ℝ) (ω φ : ℝ) := Real.sin (ω * x + φ)

theorem find_omega (ω φ : ℝ) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
  (h_symm : ∀ x : ℝ, f (3 * π / 4 + x) ω φ = f (3 * π / 4 - x) ω φ)
  (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ω φ ≤ f x2 ω φ) :
  ω = 2 / 3 ∨ ω = 2 :=
sorry

end find_omega_l78_78878


namespace simplify_expression_eq_square_l78_78764

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l78_78764


namespace DanGreenMarbles_l78_78329

theorem DanGreenMarbles : 
  ∀ (initial_green marbles_taken remaining_green : ℕ), 
  initial_green = 32 →
  marbles_taken = 23 →
  remaining_green = initial_green - marbles_taken →
  remaining_green = 9 :=
by sorry

end DanGreenMarbles_l78_78329


namespace trigonometric_simplification_l78_78130

open Real

theorem trigonometric_simplification (α : ℝ) :
  (3.4113 * sin α * cos (3 * α) + 9 * sin α * cos α - sin (3 * α) * cos (3 * α) - 3 * sin (3 * α) * cos α) = 
  2 * sin (2 * α)^3 :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_simplification_l78_78130


namespace product_of_g_on_roots_l78_78433

-- Define the given polynomials f and g
def f (x : ℝ) : ℝ := x^5 + 3 * x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 5

-- Define the roots of the polynomial f
axiom roots : ∃ (x1 x2 x3 x4 x5 : ℝ), 
  f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0

theorem product_of_g_on_roots : 
  (∃ x1 x2 x3 x4 x5: ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0) 
  → g x1 * g x2 * g x3 * g x4 * g x5 = 131 := 
by
  sorry

end product_of_g_on_roots_l78_78433


namespace division_of_203_by_single_digit_l78_78835

theorem division_of_203_by_single_digit (d : ℕ) (h : 1 ≤ d ∧ d < 10) : 
  ∃ q : ℕ, q = 203 / d ∧ (10 ≤ q ∧ q < 100 ∨ 100 ≤ q ∧ q < 1000) := 
by
  sorry

end division_of_203_by_single_digit_l78_78835


namespace problem_I_II_l78_78465

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

def seq_a (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 0 = a1 ∧ (∀ n, a (n + 1) = f (a n))

theorem problem_I_II (a : ℕ → ℝ) (a1 : ℝ) (h_a1 : 0 < a1 ∧ a1 < 1) (h_seq : seq_a a a1) :
  (∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n, a (n + 1) < (1 / 6) * (a n) ^ 3) :=
  sorry

end problem_I_II_l78_78465


namespace cost_of_pack_of_socks_is_5_l78_78966

-- Conditions definitions
def shirt_price : ℝ := 12.00
def short_price : ℝ := 15.00
def trunks_price : ℝ := 14.00
def shirts_count : ℕ := 3
def shorts_count : ℕ := 2
def total_bill : ℝ := 102.00
def total_known_cost : ℝ := 3 * shirt_price + 2 * short_price + trunks_price

-- Definition of the problem statement
theorem cost_of_pack_of_socks_is_5 (S : ℝ) : total_bill = total_known_cost + S + 0.2 * (total_known_cost + S) → S = 5 := 
by
  sorry

end cost_of_pack_of_socks_is_5_l78_78966


namespace john_remaining_amount_l78_78679

theorem john_remaining_amount (initial_amount games: ℕ) (food souvenirs: ℕ) :
  initial_amount = 100 →
  games = 20 →
  food = 3 * games →
  souvenirs = (1 / 2 : ℚ) * games →
  initial_amount - (games + food + souvenirs) = 10 :=
by
  sorry

end john_remaining_amount_l78_78679


namespace fractionD_is_unchanged_l78_78540

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l78_78540


namespace parabola_directrix_l78_78950

theorem parabola_directrix (p : ℝ) (h : p > 0) (h_directrix : -p / 2 = -4) : p = 8 :=
by
  sorry

end parabola_directrix_l78_78950


namespace rectangle_area_l78_78953

-- Define length and width
def width : ℕ := 6
def length : ℕ := 3 * width

-- Define area of the rectangle
def area (length width : ℕ) : ℕ := length * width

-- Statement to prove
theorem rectangle_area : area length width = 108 := by
  sorry

end rectangle_area_l78_78953


namespace closest_integer_to_cbrt_250_l78_78653

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l78_78653


namespace product_of_ys_l78_78165

theorem product_of_ys (x y : ℤ) (h1 : x^3 + y^2 - 3 * y + 1 < 0)
                                     (h2 : 3 * x^3 - y^2 + 3 * y > 0) : 
  (y = 1 ∨ y = 2) → (1 * 2 = 2) :=
by {
  sorry
}

end product_of_ys_l78_78165


namespace product_of_roots_l78_78699

theorem product_of_roots (Q : Polynomial ℚ) (hQ : Q.degree = 1) (h_root : Q.eval 6 = 0) :
  (Q.roots : Multiset ℚ).prod = 6 :=
sorry

end product_of_roots_l78_78699


namespace good_numbers_l78_78951

/-- Definition of a good number -/
def is_good (n : ℕ) : Prop :=
  ∃ (k_1 k_2 k_3 k_4 : ℕ), 
    (1 ≤ k_1 ∧ 1 ≤ k_2 ∧ 1 ≤ k_3 ∧ 1 ≤ k_4) ∧
    (n + k_1 ∣ n + k_1^2) ∧ 
    (n + k_2 ∣ n + k_2^2) ∧ 
    (n + k_3 ∣ n + k_3^2) ∧ 
    (n + k_4 ∣ n + k_4^2) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ 
    (k_3 ≠ k_4)

/-- The main theorem to prove -/
theorem good_numbers : 
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → 
  (Prime p ∧ Prime (2 * p + 1) ↔ is_good (2 * p)) :=
by
  sorry

end good_numbers_l78_78951


namespace gcd_max_value_l78_78282

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l78_78282


namespace percent_of_b_is_50_l78_78414

variable (a b c : ℝ)

-- Conditions
def c_is_25_percent_of_a : Prop := c = 0.25 * a
def b_is_50_percent_of_a : Prop := b = 0.50 * a

-- Proof
theorem percent_of_b_is_50 :
  c_is_25_percent_of_a c a → b_is_50_percent_of_a b a → c = 0.50 * b :=
by sorry

end percent_of_b_is_50_l78_78414


namespace max_proj_area_of_regular_tetrahedron_l78_78553

theorem max_proj_area_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) : 
    ∃ max_area : ℝ, max_area = a^2 / 2 :=
by
  existsi (a^2 / 2)
  sorry

end max_proj_area_of_regular_tetrahedron_l78_78553


namespace A_finishes_work_in_9_days_l78_78423

noncomputable def B_work_rate : ℝ := 1 / 15
noncomputable def B_work_10_days : ℝ := 10 * B_work_rate
noncomputable def remaining_work_by_A : ℝ := 1 - B_work_10_days

theorem A_finishes_work_in_9_days (A_days : ℝ) (B_days : ℝ) (B_days_worked : ℝ) (A_days_worked : ℝ) :
  (B_days = 15) ∧ (B_days_worked = 10) ∧ (A_days_worked = 3) ∧ 
  (remaining_work_by_A = (1 / 3)) → A_days = 9 :=
by sorry

end A_finishes_work_in_9_days_l78_78423


namespace container_capacity_l78_78380

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 18 = 0.75 * C) : 
  C = 40 :=
by
  -- proof steps would go here
  sorry

end container_capacity_l78_78380


namespace exists_prime_q_and_positive_n_l78_78011

theorem exists_prime_q_and_positive_n (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ q n : ℕ, Nat.Prime q ∧ q < p ∧ 0 < n ∧ p ∣ (n^2 - q) :=
by
  sorry

end exists_prime_q_and_positive_n_l78_78011


namespace probability_four_red_four_blue_l78_78768

noncomputable def urn_probability : ℚ :=
  let initial_red := 2
  let initial_blue := 1
  let operations := 5
  let final_red := 4
  let final_blue := 4
  -- calculate the probability using given conditions, this result is directly derived as 2/7
  2 / 7

theorem probability_four_red_four_blue :
  urn_probability = 2 / 7 :=
by
  sorry

end probability_four_red_four_blue_l78_78768


namespace class_average_gpa_l78_78335

theorem class_average_gpa (n : ℕ) (hn : 0 < n) :
  ((1/3 * n) * 45 + (2/3 * n) * 60) / n = 55 :=
by
  sorry

end class_average_gpa_l78_78335


namespace function_range_l78_78547

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 - 1) * (x^2 + a * x + b)

theorem function_range (a b : ℝ) (h_symm : ∀ x : ℝ, f (6 - x) a b = f x a b) :
  a = -12 ∧ b = 35 ∧ (∀ y, ∃ x : ℝ, f x (-12) 35 = y ↔ -36 ≤ y) :=
by
  sorry

end function_range_l78_78547


namespace sqrt_expression_value_l78_78740

theorem sqrt_expression_value :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 :=
by
  sorry

end sqrt_expression_value_l78_78740


namespace average_weight_of_eight_boys_l78_78881

theorem average_weight_of_eight_boys :
  let avg16 := 50.25
  let avg24 := 48.55
  let total_weight_16 := 16 * avg16
  let total_weight_all := 24 * avg24
  let W := (total_weight_all - total_weight_16) / 8
  W = 45.15 :=
by
  sorry

end average_weight_of_eight_boys_l78_78881


namespace tan_double_angle_l78_78584

theorem tan_double_angle (α : ℝ) (h1 : Real.cos (Real.pi - α) = 4 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_l78_78584


namespace nat_divisor_problem_l78_78249

open Nat

theorem nat_divisor_problem (n : ℕ) (d : ℕ → ℕ) (k : ℕ)
    (h1 : 1 = d 1)
    (h2 : ∀ i, 1 < i → i ≤ k → d i < d (i + 1))
    (hk : d k = n)
    (hdiv : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
    (heq : n = d 2 * d 3 + d 2 * d 5 + d 3 * d 5) :
    k = 8 ∨ k = 9 :=
sorry

end nat_divisor_problem_l78_78249


namespace number_of_always_true_inequalities_l78_78526

theorem number_of_always_true_inequalities (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  (a + c > b + d) ∧
  (¬(a - c > b - d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 - 3 > -2 - (-2))) ∧
  (¬(a * c > b * d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 * 3 > -2 * (-2))) ∧
  (¬(a / c > b / d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 / 3 > (-2) / (-2))) :=
by
  sorry

end number_of_always_true_inequalities_l78_78526


namespace number_of_roosters_l78_78920

def chickens := 9000
def ratio_roosters_hens := 2 / 1

theorem number_of_roosters (h : ratio_roosters_hens = 2 / 1) (c : chickens = 9000) : ∃ r : ℕ, r = 6000 := 
by sorry

end number_of_roosters_l78_78920


namespace find_number_l78_78092

theorem find_number (x N : ℕ) (h₁ : x = 32) (h₂ : N - (23 - (15 - x)) = (12 * 2 / 1 / 2)) : N = 88 :=
sorry

end find_number_l78_78092


namespace exponent_multiplication_l78_78674

theorem exponent_multiplication :
  (-1 / 2 : ℝ) ^ 2022 * (2 : ℝ) ^ 2023 = 2 :=
by sorry

end exponent_multiplication_l78_78674


namespace father_l78_78389

-- Definitions based on conditions in a)
def cost_MP3_player : ℕ := 120
def cost_CD : ℕ := 19
def total_cost : ℕ := cost_MP3_player + cost_CD
def savings : ℕ := 55
def amount_lacking : ℕ := 64

-- Statement of the proof problem
theorem father's_contribution : (savings + (148:ℕ) - amount_lacking = total_cost) := by
  -- Add sorry to skip the proof
  sorry

end father_l78_78389


namespace tic_tac_toe_board_configurations_l78_78340

theorem tic_tac_toe_board_configurations :
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  total_configurations = 592 :=
by 
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  sorry

end tic_tac_toe_board_configurations_l78_78340


namespace percentage_increase_is_20_l78_78934

noncomputable def total_stocks : ℕ := 1980
noncomputable def stocks_higher : ℕ := 1080
noncomputable def stocks_lower : ℕ := total_stocks - stocks_higher

/--
Given that the total number of stocks is 1,980, and 1,080 stocks closed at a higher price today than yesterday.
Furthermore, the number of stocks that closed higher today is greater than the number that closed lower.

Prove that the percentage increase in the number of stocks that closed at a higher price today compared to the number that closed at a lower price is 20%.
-/
theorem percentage_increase_is_20 :
  (stocks_higher - stocks_lower) / stocks_lower * 100 = 20 := by
  sorry

end percentage_increase_is_20_l78_78934


namespace transformed_curve_eq_l78_78585

/-- Given the initial curve equation and the scaling transformation,
    prove that the resulting curve has the transformed equation. -/
theorem transformed_curve_eq 
  (x y x' y' : ℝ)
  (h_curve : x^2 + 9*y^2 = 9)
  (h_transform_x : x' = x)
  (h_transform_y : y' = 3*y) :
  (x')^2 + y'^2 = 9 := 
sorry

end transformed_curve_eq_l78_78585


namespace vanya_faster_speed_l78_78366

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l78_78366


namespace total_interest_l78_78687

def P : ℝ := 1000
def r : ℝ := 0.1
def n : ℕ := 3

theorem total_interest : (P * (1 + r)^n) - P = 331 := by
  sorry

end total_interest_l78_78687


namespace trigonometric_relationship_l78_78289

-- Given conditions
variables (x : ℝ) (a b c : ℝ)

-- Required conditions
variables (h1 : π / 4 < x) (h2 : x < π / 2)
variables (ha : a = Real.sin x)
variables (hb : b = Real.cos x)
variables (hc : c = Real.tan x)

-- Proof goal
theorem trigonometric_relationship : b < a ∧ a < c :=
by
  -- Proof will go here
  sorry

end trigonometric_relationship_l78_78289


namespace problem_equivalent_proof_statement_l78_78388

-- Definition of a line with a definite slope
def has_definite_slope (m : ℝ) : Prop :=
  ∃ slope : ℝ, slope = -m 

-- Definition of the equation of a line passing through two points being correct
def line_through_two_points (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2) : Prop :=
  ∀ x y : ℝ, (y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)) ↔ y = ((y2 - y1) * (x - x1) / (x2 - x1)) + y1 

-- Formalizing and proving the given conditions
theorem problem_equivalent_proof_statement : 
  (∀ m : ℝ, has_definite_slope m) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2), line_through_two_points x1 y1 x2 y2 h) :=
by 
  sorry

end problem_equivalent_proof_statement_l78_78388


namespace simplify_expression_l78_78356

theorem simplify_expression (x : ℝ) (h : x ≤ 2) : 
  (Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9)) = -1 :=
by 
  sorry

end simplify_expression_l78_78356


namespace tan_225_eq_1_l78_78032

theorem tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
  sorry

end tan_225_eq_1_l78_78032


namespace percent_increase_l78_78700

def initial_price : ℝ := 15
def final_price : ℝ := 16

theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 6.67 :=
by
  sorry

end percent_increase_l78_78700


namespace problem1_problem2_l78_78122

section Problems

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Problem 1: Tangent line problem for a = 1
def tangent_line_eqn (x : ℝ) : Prop :=
  let a := 1
  let f := f a
  (∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b)

-- Problem 2: Minimum value problem
def min_value_condition (a : ℝ) : Prop :=
  f a (1 / 4) = (11 / 12)

theorem problem1 : tangent_line_eqn 0 :=
  sorry

theorem problem2 : min_value_condition (1 / 4) :=
  sorry

end Problems

end problem1_problem2_l78_78122


namespace sin_minus_cos_eq_l78_78681

variable {α : ℝ} (h₁ : 0 < α ∧ α < π) (h₂ : Real.sin α + Real.cos α = 1/3)

theorem sin_minus_cos_eq : Real.sin α - Real.cos α = Real.sqrt 17 / 3 :=
by 
  -- Proof goes here
  sorry

end sin_minus_cos_eq_l78_78681


namespace bruce_anne_clean_in_4_hours_l78_78375

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l78_78375


namespace hyperbola_focus_l78_78618

theorem hyperbola_focus (m : ℝ) (h : (0, 5) = (0, 5)) : 
  (∀ x y : ℝ, (y^2 / m - x^2 / 9 = 1) → m = 16) :=
sorry

end hyperbola_focus_l78_78618


namespace maximize_triangle_area_l78_78299

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end maximize_triangle_area_l78_78299


namespace quadratic_function_range_l78_78037

def range_of_quadratic_function : Set ℝ :=
  {y : ℝ | y ≥ 2}

theorem quadratic_function_range :
  ∀ x : ℝ, (∃ y : ℝ, y = x^2 - 4*x + 6 ∧ y ∈ range_of_quadratic_function) :=
by
  sorry

end quadratic_function_range_l78_78037


namespace amount_of_rice_distributed_in_first_5_days_l78_78569

-- Definitions from conditions
def workers_day (d : ℕ) : ℕ := if d = 1 then 64 else 64 + 7 * (d - 1)

-- The amount of rice each worker receives per day
def rice_per_worker : ℕ := 3

-- Total workers dispatched in the first 5 days
def total_workers_first_5_days : ℕ := (workers_day 1 + workers_day 2 + workers_day 3 + workers_day 4 + workers_day 5)

-- Given these definitions, we now state the theorem to prove
theorem amount_of_rice_distributed_in_first_5_days : total_workers_first_5_days * rice_per_worker = 1170 :=
by
  sorry

end amount_of_rice_distributed_in_first_5_days_l78_78569


namespace quiz_competition_l78_78883

theorem quiz_competition (x : ℕ) :
  (10 * x - 4 * (20 - x) ≥ 88) ↔ (x ≥ 12) :=
by 
  sorry

end quiz_competition_l78_78883


namespace rahul_and_sham_together_complete_task_in_35_days_l78_78535

noncomputable def rahul_rate (W : ℝ) : ℝ := W / 60
noncomputable def sham_rate (W : ℝ) : ℝ := W / 84
noncomputable def combined_rate (W : ℝ) := rahul_rate W + sham_rate W

theorem rahul_and_sham_together_complete_task_in_35_days (W : ℝ) :
  (W / combined_rate W) = 35 :=
by
  sorry

end rahul_and_sham_together_complete_task_in_35_days_l78_78535


namespace smallest_possible_area_square_l78_78923

theorem smallest_possible_area_square : 
  ∃ (c : ℝ), (∀ (x y : ℝ), ((y = 3 * x - 20) ∨ (y = x^2)) ∧ 
      (10 * (9 + 4 * c) = ((c + 20) / Real.sqrt 10) ^ 2) ∧ 
      (c = 80) ∧ 
      (10 * (9 + 4 * c) = 3290)) :=
by {
  use 80,
  sorry
}

end smallest_possible_area_square_l78_78923


namespace inequality_positive_real_numbers_l78_78159

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l78_78159


namespace radius_of_tangent_circle_l78_78866

def is_tangent_coor_axes_and_leg (r : ℝ) : Prop :=
  -- Circle with radius r is tangent to coordinate axes and one leg of the triangle
  ∃ O B C : ℝ × ℝ, 
  -- Conditions: centers and tangency
  O = (r, r) ∧ 
  B = (0, 2) ∧ 
  C = (2, 0) ∧ 
  r = 1

theorem radius_of_tangent_circle :
  ∀ r : ℝ, is_tangent_coor_axes_and_leg r → r = 1 :=
by
  sorry

end radius_of_tangent_circle_l78_78866


namespace product_of_sequence_is_243_l78_78767

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l78_78767


namespace john_total_distance_l78_78129

theorem john_total_distance :
  let speed := 55 -- John's speed in mph
  let time1 := 2 -- Time before lunch in hours
  let time2 := 3 -- Time after lunch in hours
  let distance1 := speed * time1 -- Distance before lunch
  let distance2 := speed * time2 -- Distance after lunch
  let total_distance := distance1 + distance2 -- Total distance

  total_distance = 275 :=
by
  sorry

end john_total_distance_l78_78129


namespace distance_between_parallel_lines_l78_78155

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end distance_between_parallel_lines_l78_78155


namespace find_a_l78_78391

def M : Set ℝ := {x | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem find_a (a : ℝ) : N a ⊆ M ↔ a = -1 ∨ a = 0 ∨ a = 2/3 := 
by
  sorry

end find_a_l78_78391


namespace hilt_books_difference_l78_78846

noncomputable def original_price : ℝ := 11
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discount_price (price : ℝ) (rate : ℝ) : ℝ := price * (1 - rate)
noncomputable def quantity : ℕ := 15
noncomputable def sale_price : ℝ := 25
noncomputable def tax_rate : ℝ := 0.10
noncomputable def price_with_tax (price : ℝ) (rate : ℝ) : ℝ := price * (1 + rate)

noncomputable def total_cost : ℝ := discount_price original_price discount_rate * quantity
noncomputable def total_revenue : ℝ := price_with_tax sale_price tax_rate * quantity
noncomputable def profit : ℝ := total_revenue - total_cost

theorem hilt_books_difference : profit = 280.50 :=
by
  sorry

end hilt_books_difference_l78_78846


namespace union_M_N_l78_78787

def M : Set ℕ := {1, 2}
def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l78_78787


namespace product_of_total_points_l78_78646

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def Allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def Betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem product_of_total_points :
  total_points Allie_rolls * total_points Betty_rolls = 256 :=
by
  sorry

end product_of_total_points_l78_78646


namespace count_valid_triples_l78_78331

theorem count_valid_triples :
  ∃! (a c : ℕ), a ≤ 101 ∧ 101 ≤ c ∧ a * c = 101^2 :=
sorry

end count_valid_triples_l78_78331


namespace pond_fish_approximation_l78_78209

noncomputable def total_number_of_fish
  (tagged_first: ℕ) (total_caught_second: ℕ) (tagged_second: ℕ) : ℕ :=
  (tagged_first * total_caught_second) / tagged_second

theorem pond_fish_approximation :
  total_number_of_fish 60 50 2 = 1500 :=
by
  -- calculation of the total number of fish based on given conditions
  sorry

end pond_fish_approximation_l78_78209


namespace total_wait_time_difference_l78_78106

theorem total_wait_time_difference :
  let kids_swings := 6
  let kids_slide := 4 * kids_swings
  let wait_time_swings := [210, 420, 840] -- in seconds
  let total_wait_time_swings := wait_time_swings.sum
  let wait_time_slide := [45, 90, 180] -- in seconds
  let total_wait_time_slide := wait_time_slide.sum
  let total_wait_time_all_kids_swings := kids_swings * total_wait_time_swings
  let total_wait_time_all_kids_slide := kids_slide * total_wait_time_slide
  let difference := total_wait_time_all_kids_swings - total_wait_time_all_kids_slide
  difference = 1260 := sorry

end total_wait_time_difference_l78_78106


namespace find_fraction_value_l78_78668

variable (a b : ℝ)

theorem find_fraction_value (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 := 
  sorry

end find_fraction_value_l78_78668


namespace perpendicular_tangents_l78_78233

theorem perpendicular_tangents (a b : ℝ) (h1 : ∀ (x y : ℝ), y = x^3 → y = (3 * x^2) * (x - 1) + 1 → y = 3 * (x - 1) + 1) (h2 : (a : ℝ) * 1 - (b : ℝ) * 1 = 2) 
 (h3 : (a : ℝ)/(b : ℝ) * 3 = -1) : a / b = -1 / 3 :=
by
  sorry

end perpendicular_tangents_l78_78233


namespace min_ab_l78_78215

variable (a b : ℝ)

theorem min_ab (h1 : a > 1) (h2 : b > 2) (h3 : a * b = 2 * a + b) : a + b ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_ab_l78_78215


namespace rosie_pie_count_l78_78943

-- Conditions and definitions
def apples_per_pie (total_apples pies : ℕ) : ℕ := total_apples / pies

-- Theorem statement (mathematical proof problem)
theorem rosie_pie_count :
  ∀ (a p : ℕ), a = 12 → p = 3 → (36 : ℕ) / (apples_per_pie a p) = 9 :=
by
  intros a p ha hp
  rw [ha, hp]
  -- Skipping the proof
  sorry

end rosie_pie_count_l78_78943


namespace cos_eq_cos_of_n_l78_78827

theorem cos_eq_cos_of_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (283 * Real.pi / 180)) : n = 77 :=
by sorry

end cos_eq_cos_of_n_l78_78827


namespace probability_points_one_unit_apart_l78_78967

theorem probability_points_one_unit_apart :
  let total_points := 16
  let total_pairs := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  let probability := favorable_pairs / total_pairs
  probability = (1 : ℚ) / 10 :=
by
  sorry

end probability_points_one_unit_apart_l78_78967


namespace bobs_income_after_changes_l78_78154

variable (initial_salary : ℝ) (february_increase_rate : ℝ) (march_reduction_rate : ℝ)

def february_salary (initial_salary : ℝ) (increase_rate : ℝ) : ℝ :=
  initial_salary * (1 + increase_rate)

def march_salary (february_salary : ℝ) (reduction_rate : ℝ) : ℝ :=
  february_salary * (1 - reduction_rate)

theorem bobs_income_after_changes (h1 : initial_salary = 2750)
  (h2 : february_increase_rate = 0.15)
  (h3 : march_reduction_rate = 0.10) :
  march_salary (february_salary initial_salary february_increase_rate) march_reduction_rate = 2846.25 := 
sorry

end bobs_income_after_changes_l78_78154


namespace distance_from_P_to_y_axis_l78_78285

theorem distance_from_P_to_y_axis 
  (x y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 25) = 1)
  (F1 : ℝ × ℝ := (0, -3))
  (F2 : ℝ × ℝ := (0, 3))
  (h2 : (F1.1 - x)^2 + (F1.2 - y)^2 = 9 ∨ (F2.1 - x)^2 + (F2.2 - y)^2 = 9 
          ∨ (F1.1 - x)^2 + (F1.2 - y)^2 + (F2.1 - x)^2 + (F2.2 - y)^2 = (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) :
  |x| = 16 / 5 :=
by
  sorry

end distance_from_P_to_y_axis_l78_78285


namespace equal_amounts_hot_and_cold_water_l78_78525

theorem equal_amounts_hot_and_cold_water (time_to_fill_cold : ℕ) (time_to_fill_hot : ℕ) (t_c : ℤ) : 
  time_to_fill_cold = 19 → 
  time_to_fill_hot = 23 → 
  t_c = 2 :=
by
  intros h_c h_h
  sorry

end equal_amounts_hot_and_cold_water_l78_78525


namespace exists_nat_number_gt_1000_l78_78420

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_nat_number_gt_1000 (S : ℕ → ℕ) :
  (∀ n : ℕ, S (2^n) = sum_of_digits (2^n)) →
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n + 1)) :=
by sorry

end exists_nat_number_gt_1000_l78_78420


namespace find_c_l78_78323

-- Definition of the function f
def f (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

-- Theorem statement
theorem find_c (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : f a a b c = a^3)
  (h4 : f b a b c = b^3) : c = 16 :=
by
  sorry

end find_c_l78_78323


namespace problem_solution_l78_78060

theorem problem_solution :
  ((8 * 2.25 - 5 * 0.85) / 2.5 + (3 / 5 * 1.5 - 7 / 8 * 0.35) / 1.25) = 5.975 :=
by
  sorry

end problem_solution_l78_78060


namespace cars_minus_trucks_l78_78432

theorem cars_minus_trucks (total : ℕ) (trucks : ℕ) (h_total : total = 69) (h_trucks : trucks = 21) :
  (total - trucks) - trucks = 27 :=
by
  sorry

end cars_minus_trucks_l78_78432


namespace problem1_problem2_l78_78153

-- Define the first problem as a proof statement in Lean
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 25 → (x = 7 ∨ x = -3) := sorry

-- Define the second problem as a proof statement in Lean
theorem problem2 (x : ℝ) : (x - 5) ^ 2 = 2 * (5 - x) → (x = 5 ∨ x = 3) := sorry

end problem1_problem2_l78_78153


namespace max_d_value_l78_78089

theorem max_d_value : ∀ (d e : ℕ), (d < 10) → (e < 10) → (5 * 10^5 + d * 10^4 + 5 * 10^3 + 2 * 10^2 + 2 * 10 + e ≡ 0 [MOD 22]) → (e % 2 = 0) → (d + e = 10) → d ≤ 8 :=
by
  intros d e h1 h2 h3 h4 h5
  sorry

end max_d_value_l78_78089


namespace profit_margin_increase_l78_78844

theorem profit_margin_increase (CP : ℝ) (SP : ℝ) (NSP : ℝ) (initial_margin : ℝ) (desired_margin : ℝ) :
  initial_margin = 0.25 → desired_margin = 0.40 → SP = (1 + initial_margin) * CP → NSP = (1 + desired_margin) * CP →
  ((NSP - SP) / SP) * 100 = 12 := 
by 
  intros h1 h2 h3 h4
  sorry

end profit_margin_increase_l78_78844


namespace probability_of_two_boys_given_one_boy_l78_78770

-- Define the events and probabilities
def P_BB : ℚ := 1/4
def P_BG : ℚ := 1/4
def P_GB : ℚ := 1/4
def P_GG : ℚ := 1/4

def P_at_least_one_boy : ℚ := 1 - P_GG

def P_two_boys_given_at_least_one_boy : ℚ := P_BB / P_at_least_one_boy

-- Statement to be proven
theorem probability_of_two_boys_given_one_boy : P_two_boys_given_at_least_one_boy = 1/3 :=
by sorry

end probability_of_two_boys_given_one_boy_l78_78770


namespace simplest_form_fraction_l78_78136

theorem simplest_form_fraction 
  (m n a : ℤ) (h_f1 : (2 * m) / (10 * m * n) = 1 / (5 * n))
  (h_f2 : (m^2 - n^2) / (m + n) = (m - n))
  (h_f3 : (2 * a) / (a^2) = 2 / a) : 
  ∀ (f : ℤ), f = (m^2 + n^2) / (m + n) → 
    (∀ (k : ℤ), k ≠ 1 → (m^2 + n^2) / (m + n) ≠ k * f) :=
by
  intros f h_eq k h_kneq1
  sorry

end simplest_form_fraction_l78_78136


namespace john_total_expenses_l78_78023

theorem john_total_expenses :
  (let epiPenCost := 500
   let yearlyMedicalExpenses := 2000
   let firstEpiPenInsuranceCoverage := 0.75
   let secondEpiPenInsuranceCoverage := 0.60
   let medicalExpensesCoverage := 0.80
   let firstEpiPenCost := epiPenCost * (1 - firstEpiPenInsuranceCoverage)
   let secondEpiPenCost := epiPenCost * (1 - secondEpiPenInsuranceCoverage)
   let totalEpiPenCost := firstEpiPenCost + secondEpiPenCost
   let yearlyMedicalExpensesCost := yearlyMedicalExpenses * (1 - medicalExpensesCoverage)
   let totalCost := totalEpiPenCost + yearlyMedicalExpensesCost
   totalCost) = 725 := sorry

end john_total_expenses_l78_78023


namespace rhombus_diagonal_length_l78_78760

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length_l78_78760


namespace correct_answer_l78_78582

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

theorem correct_answer (m : ℝ) : 
  (∀ x₁ x₂, 1 < x₁ → 1 < x₂ → (f x₁ m - f x₂ m) / (x₁ - x₂) > 0) → m ≥ -4 :=
by
  sorry

end correct_answer_l78_78582


namespace same_oxidation_state_HNO3_N2O5_l78_78621

def oxidation_state_HNO3 (H O: Int) : Int := 1 + 1 + (3 * (-2))
def oxidation_state_N2O5 (H O: Int) : Int := (2 * 1) + (5 * (-2))
def oxidation_state_substances_equal : Prop :=
  oxidation_state_HNO3 1 (-2) = oxidation_state_N2O5 1 (-2)

theorem same_oxidation_state_HNO3_N2O5 : oxidation_state_substances_equal :=
  by
  sorry

end same_oxidation_state_HNO3_N2O5_l78_78621


namespace fraction_subtraction_l78_78636

theorem fraction_subtraction :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end fraction_subtraction_l78_78636


namespace car_speeds_l78_78684

theorem car_speeds (d x : ℝ) (small_car_speed large_car_speed : ℝ) 
  (h1 : d = 135) 
  (h2 : small_car_speed = 5 * x) 
  (h3 : large_car_speed = 2 * x) 
  (h4 : 135 / small_car_speed + (4 + 0.5) = 135 / large_car_speed)
  : small_car_speed = 45 ∧ large_car_speed = 18 := by
  sorry

end car_speeds_l78_78684


namespace actual_distance_traveled_l78_78960

theorem actual_distance_traveled (D t : ℝ) 
  (h1 : D = 15 * t)
  (h2 : D + 50 = 35 * t) : 
  D = 37.5 :=
by
  sorry

end actual_distance_traveled_l78_78960


namespace playground_area_l78_78514

theorem playground_area (L B : ℕ) (h1 : B = 6 * L) (h2 : B = 420)
  (A_total A_playground : ℕ) (h3 : A_total = L * B) 
  (h4 : A_playground = A_total / 7) :
  A_playground = 4200 :=
by sorry

end playground_area_l78_78514


namespace sugar_percentage_l78_78665

theorem sugar_percentage 
  (initial_volume : ℝ) (initial_water_perc : ℝ) (initial_kola_perc: ℝ) (added_sugar : ℝ) (added_water : ℝ) (added_kola : ℝ)
  (initial_solution: initial_volume = 340) 
  (perc_water : initial_water_perc = 0.75) 
  (perc_kola: initial_kola_perc = 0.05)
  (added_sugar_amt : added_sugar = 3.2) 
  (added_water_amt : added_water = 12) 
  (added_kola_amt : added_kola = 6.8) : 
  (71.2 / 362) * 100 = 19.67 := 
by 
  sorry

end sugar_percentage_l78_78665


namespace neg_prop1_true_neg_prop2_false_l78_78121

-- Proposition 1: The logarithm of a positive number is always positive
def prop1 : Prop := ∀ x : ℝ, x > 0 → Real.log x > 0

-- Negation of Proposition 1: There exists a positive number whose logarithm is not positive
def neg_prop1 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0

-- Proposition 2: For all x in the set of integers Z, the last digit of x^2 is not 3
def prop2 : Prop := ∀ x : ℤ, (x * x % 10 ≠ 3)

-- Negation of Proposition 2: There exists an x in the set of integers Z such that the last digit of x^2 is 3
def neg_prop2 : Prop := ∃ x : ℤ, (x * x % 10 = 3)

-- Proof that the negation of Proposition 1 is true
theorem neg_prop1_true : neg_prop1 := 
  by sorry

-- Proof that the negation of Proposition 2 is false
theorem neg_prop2_false : ¬ neg_prop2 := 
  by sorry

end neg_prop1_true_neg_prop2_false_l78_78121


namespace equal_total_areas_of_checkerboard_pattern_l78_78416

-- Definition representing the convex quadrilateral and its subdivisions
structure ConvexQuadrilateral :=
  (A B C D : ℝ × ℝ) -- vertices of the quadrilateral

-- Predicate indicating the subdivision and coloring pattern
inductive CheckerboardColor
  | Black
  | White

-- Function to determine the area of the resulting smaller quadrilateral
noncomputable def area_of_subquadrilateral 
  (quad : ConvexQuadrilateral) 
  (subdivision : ℕ) -- subdivision factor
  (color : CheckerboardColor) 
  : ℝ := -- returns the area based on the subdivision and color
  -- Simplified implementation of area calculation
  -- (detailed geometric computation should replace this placeholder)
  sorry

-- Function to determine the total area of quadrilaterals of a given color
noncomputable def total_area_of_color 
  (quad : ConvexQuadrilateral) 
  (substution : ℕ) 
  (color : CheckerboardColor) 
  : ℝ := -- Total area of subquadrilaterals of the given color
  sorry

-- Theorem stating the required proof
theorem equal_total_areas_of_checkerboard_pattern
  (quad : ConvexQuadrilateral)
  (subdivision : ℕ)
  : total_area_of_color quad subdivision CheckerboardColor.Black = total_area_of_color quad subdivision CheckerboardColor.White :=
  sorry

end equal_total_areas_of_checkerboard_pattern_l78_78416


namespace walt_age_l78_78188

theorem walt_age (T W : ℕ) 
  (h1 : T = 3 * W)
  (h2 : T + 12 = 2 * (W + 12)) : 
  W = 12 :=
by
  sorry

end walt_age_l78_78188


namespace quadratic_inequality_solution_l78_78549

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
by
  sorry

end quadratic_inequality_solution_l78_78549


namespace cube_roll_sums_l78_78156

def opposite_faces_sum_to_seven (a b : ℕ) : Prop := a + b = 7

def valid_cube_faces : Prop := 
  opposite_faces_sum_to_seven 1 6 ∧
  opposite_faces_sum_to_seven 2 5 ∧
  opposite_faces_sum_to_seven 3 4

def max_min_sums : ℕ × ℕ := (342, 351)

theorem cube_roll_sums (faces_sum_seven : valid_cube_faces) : 
  ∃ cube_sums : ℕ × ℕ, cube_sums = max_min_sums := sorry

end cube_roll_sums_l78_78156


namespace minimum_pizzas_needed_l78_78393

variables (p : ℕ)

def income_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4
def maintenance_cost_per_pizza : ℕ := 1
def car_cost : ℕ := 6500

theorem minimum_pizzas_needed :
  p ≥ 929 ↔ (income_per_pizza * p - (gas_cost_per_pizza + maintenance_cost_per_pizza) * p) ≥ car_cost :=
sorry

end minimum_pizzas_needed_l78_78393


namespace rice_cake_slices_length_l78_78974

noncomputable def slice_length (cake_length : ℝ) (num_cakes : ℕ) (overlap : ℝ) (num_slices : ℕ) : ℝ :=
  let total_original_length := num_cakes * cake_length
  let total_overlap := (num_cakes - 1) * overlap
  let actual_length := total_original_length - total_overlap
  actual_length / num_slices

theorem rice_cake_slices_length : 
  slice_length 2.7 5 0.3 6 = 2.05 :=
by
  sorry

end rice_cake_slices_length_l78_78974


namespace frac_equiv_l78_78596

theorem frac_equiv (a b : ℚ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end frac_equiv_l78_78596


namespace ratio_perimeter_pentagon_to_square_l78_78771

theorem ratio_perimeter_pentagon_to_square
  (a : ℝ) -- Let a be the length of each side of the square
  (T_perimeter S_perimeter : ℝ) 
  (h1 : T_perimeter = S_perimeter) -- Given the perimeter of the triangle equals the perimeter of the square
  (h2 : S_perimeter = 4 * a) -- Given the perimeter of the square is 4 times the length of its side
  (P_perimeter : ℝ)
  (h3 : P_perimeter = (T_perimeter + S_perimeter) - 2 * a) -- Perimeter of the pentagon considering shared edge
  :
  P_perimeter / S_perimeter = 3 / 2 := 
sorry

end ratio_perimeter_pentagon_to_square_l78_78771


namespace triangle_area_is_60_l78_78616

noncomputable def triangle_area (P r : ℝ) : ℝ :=
  (r * P) / 2

theorem triangle_area_is_60 (hP : 48 = 48) (hr : 2.5 = 2.5) : triangle_area 48 2.5 = 60 := by
  sorry

end triangle_area_is_60_l78_78616


namespace max_marks_paper_I_l78_78984

theorem max_marks_paper_I (M : ℝ) (h1 : 0.40 * M = 60) : M = 150 :=
by
  sorry

end max_marks_paper_I_l78_78984


namespace smallest_prime_factor_of_2939_l78_78275

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

theorem smallest_prime_factor_of_2939 : smallest_prime_factor 2939 13 :=
by
  sorry

end smallest_prime_factor_of_2939_l78_78275


namespace gcd_irreducible_fraction_l78_78038

theorem gcd_irreducible_fraction (n : ℕ) (hn: 0 < n) : gcd (3*n + 1) (5*n + 2) = 1 :=
  sorry

end gcd_irreducible_fraction_l78_78038


namespace sector_area_l78_78489

theorem sector_area (r α S : ℝ) (h1 : α = 2) (h2 : 2 * r + α * r = 8) : S = 4 :=
sorry

end sector_area_l78_78489


namespace rachel_math_homework_l78_78057

theorem rachel_math_homework (reading_hw math_hw : ℕ) 
  (h1 : reading_hw = 4) 
  (h2 : math_hw = reading_hw + 3) : 
  math_hw = 7 := by
  sorry

end rachel_math_homework_l78_78057


namespace parallelogram_side_lengths_l78_78072

theorem parallelogram_side_lengths (x y : ℝ) (h₁ : 3 * x + 6 = 12) (h₂ : 10 * y - 3 = 15) : x + y = 3.8 :=
by
  sorry

end parallelogram_side_lengths_l78_78072


namespace symmetric_point_reflection_l78_78135

theorem symmetric_point_reflection (x y : ℝ) : (2, -(-5)) = (2, 5) := by
  sorry

end symmetric_point_reflection_l78_78135


namespace prove_m_eq_n_l78_78761

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end prove_m_eq_n_l78_78761


namespace probability_of_specific_combination_l78_78429

theorem probability_of_specific_combination :
  let shirts := 6
  let shorts := 8
  let socks := 7
  let total_clothes := shirts + shorts + socks
  let ways_total := Nat.choose total_clothes 4
  let ways_shirts := Nat.choose shirts 2
  let ways_shorts := Nat.choose shorts 1
  let ways_socks := Nat.choose socks 1
  let ways_favorable := ways_shirts * ways_shorts * ways_socks
  let probability := (ways_favorable: ℚ) / ways_total
  probability = 56 / 399 :=
by
  simp
  sorry

end probability_of_specific_combination_l78_78429


namespace inequalities_correct_l78_78750

theorem inequalities_correct (a b : ℝ) (h : a * b > 0) :
  |b| > |a| ∧ |a + b| < |b| := sorry

end inequalities_correct_l78_78750


namespace max_students_received_less_than_given_l78_78351

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l78_78351


namespace find_a_range_l78_78425

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x| + 2 else x + 2 / x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end find_a_range_l78_78425


namespace arccos_one_eq_zero_l78_78815

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l78_78815


namespace dice_product_divisibility_probability_l78_78002

theorem dice_product_divisibility_probability :
  let p := 1 - ((5 / 18)^6 : ℚ)
  p = (33996599 / 34012224 : ℚ) :=
by
  -- This is the condition where the probability p is computed as the complementary probability.
  sorry

end dice_product_divisibility_probability_l78_78002


namespace measurement_units_correct_l78_78820

structure Measurement (A : Type) where
  value : A
  unit : String

def height_of_desk : Measurement ℕ := ⟨70, "centimeters"⟩
def weight_of_apple : Measurement ℕ := ⟨240, "grams"⟩
def duration_of_soccer_game : Measurement ℕ := ⟨90, "minutes"⟩
def dad_daily_work_duration : Measurement ℕ := ⟨8, "hours"⟩

theorem measurement_units_correct :
  height_of_desk.unit = "centimeters" ∧
  weight_of_apple.unit = "grams" ∧
  duration_of_soccer_game.unit = "minutes" ∧
  dad_daily_work_duration.unit = "hours" :=
by
  sorry

end measurement_units_correct_l78_78820


namespace sparkling_water_cost_l78_78362

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end sparkling_water_cost_l78_78362


namespace inscribed_sphere_radius_of_tetrahedron_l78_78014

variables (V S1 S2 S3 S4 R : ℝ)

theorem inscribed_sphere_radius_of_tetrahedron
  (hV_pos : 0 < V)
  (hS_pos : 0 < S1) (hS2_pos : 0 < S2) (hS3_pos : 0 < S3) (hS4_pos : 0 < S4) :
  R = 3 * V / (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_radius_of_tetrahedron_l78_78014


namespace number_of_true_propositions_l78_78725

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end number_of_true_propositions_l78_78725


namespace arithmetic_expression_l78_78466

theorem arithmetic_expression : 125 - 25 * 4 = 25 := 
by
  sorry

end arithmetic_expression_l78_78466


namespace shortest_side_length_rectangular_solid_geometric_progression_l78_78625

theorem shortest_side_length_rectangular_solid_geometric_progression
  (b s : ℝ)
  (h1 : (b^3 / s) = 512)
  (h2 : 2 * ((b^2 / s) + (b^2 * s) + b^2) = 384)
  : min (b / s) (min b (b * s)) = 8 := 
sorry

end shortest_side_length_rectangular_solid_geometric_progression_l78_78625


namespace number_of_passed_boys_l78_78181

theorem number_of_passed_boys 
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 36 * 120) :
  P = 105 := 
sorry

end number_of_passed_boys_l78_78181


namespace frustum_lateral_area_l78_78332

def frustum_upper_base_radius : ℝ := 3
def frustum_lower_base_radius : ℝ := 4
def frustum_slant_height : ℝ := 6

theorem frustum_lateral_area : 
  (1 / 2) * (frustum_upper_base_radius + frustum_lower_base_radius) * 2 * Real.pi * frustum_slant_height = 42 * Real.pi :=
by
  sorry

end frustum_lateral_area_l78_78332


namespace negation_abs_val_statement_l78_78925

theorem negation_abs_val_statement (x : ℝ) :
  ¬ (|x| ≤ 3 ∨ |x| > 5) ↔ (|x| > 3 ∧ |x| ≤ 5) :=
by sorry

end negation_abs_val_statement_l78_78925


namespace condition_on_p_l78_78451

theorem condition_on_p (p q r M : ℝ) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : 0 < M) :
  p > (100 * (q + r)) / (100 - q - r) → 
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M :=
by
  intro h
  -- The proof will go here
  sorry

end condition_on_p_l78_78451


namespace dog_age_64_human_years_l78_78217

def dog_years (human_years : ℕ) : ℕ :=
if human_years = 0 then
  0
else if human_years = 1 then
  1
else if human_years = 2 then
  2
else
  2 + (human_years - 2) / 5

theorem dog_age_64_human_years : dog_years 64 = 10 :=
by 
    sorry

end dog_age_64_human_years_l78_78217


namespace angle_A_in_triangle_l78_78581

theorem angle_A_in_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 2 → B = π / 4 → 
  (A = π / 3 ∨ A = 2 * π / 3) :=
by
  intros A B C a b c ha hb hB
  sorry

end angle_A_in_triangle_l78_78581


namespace percentage_y_of_x_l78_78033

variable {x y : ℝ}

theorem percentage_y_of_x 
  (h : 0.15 * x = 0.20 * y) : y = 0.75 * x := 
sorry

end percentage_y_of_x_l78_78033


namespace range_of_f_l78_78306

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l78_78306


namespace setC_is_not_pythagorean_triple_l78_78208

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end setC_is_not_pythagorean_triple_l78_78208


namespace eight_digit_descending_numbers_count_l78_78000

theorem eight_digit_descending_numbers_count : (Nat.choose 10 2) = 45 :=
by
  sorry

end eight_digit_descending_numbers_count_l78_78000


namespace no_prime_pair_summing_to_53_l78_78152

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l78_78152


namespace sum_of_prime_factors_l78_78908

theorem sum_of_prime_factors (n : ℕ) (h : n = 257040) : 
  (2 + 5 + 3 + 107 = 117) :=
by sorry

end sum_of_prime_factors_l78_78908


namespace vector_expression_l78_78483

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j k a b : V)
variables (h_i_j_k_non_coplanar : ∃ (l m n : ℝ), l • i + m • j + n • k = 0 → l = 0 ∧ m = 0 ∧ n = 0)
variables (h_a : a = (1 / 2 : ℝ) • i - j + k)
variables (h_b : b = 5 • i - 2 • j - k)

theorem vector_expression :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
by
  sorry

end vector_expression_l78_78483


namespace integer_b_if_integer_a_l78_78292

theorem integer_b_if_integer_a (a b : ℤ) (h : 2 * a + a^2 = 2 * b + b^2) : (∃ a' : ℤ, a = a') → ∃ b' : ℤ, b = b' :=
by
-- proof will be filled in here
sorry

end integer_b_if_integer_a_l78_78292


namespace stephen_female_worker_ants_l78_78926

-- Define the conditions
def stephen_ants : ℕ := 110
def worker_ants (total_ants : ℕ) : ℕ := total_ants / 2
def male_worker_ants (workers : ℕ) : ℕ := (20 / 100) * workers

-- Define the question and correct answer
def female_worker_ants (total_ants : ℕ) : ℕ :=
  let workers := worker_ants total_ants
  workers - male_worker_ants workers

-- The theorem to prove
theorem stephen_female_worker_ants : female_worker_ants stephen_ants = 44 :=
  by sorry -- Skip the proof for now

end stephen_female_worker_ants_l78_78926


namespace sam_paid_amount_l78_78521

theorem sam_paid_amount (F : ℝ) (Joe Peter Sam : ℝ) 
  (h1 : Joe = (1/4)*F + 7) 
  (h2 : Peter = (1/3)*F - 7) 
  (h3 : Sam = (1/2)*F - 12)
  (h4 : Joe + Peter + Sam = F) : 
  Sam = 60 := 
by 
  sorry

end sam_paid_amount_l78_78521


namespace purchase_in_april_l78_78270

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l78_78270


namespace unknown_rate_of_blankets_l78_78828

theorem unknown_rate_of_blankets (x : ℝ) :
  2 * 100 + 5 * 150 + 2 * x = 9 * 150 → x = 200 :=
by
  sorry

end unknown_rate_of_blankets_l78_78828


namespace geometric_series_sum_example_l78_78612

-- Define the finite geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- State the theorem
theorem geometric_series_sum_example :
  geometric_series_sum (1/2) (1/2) 8 = 255 / 256 :=
by
  sorry

end geometric_series_sum_example_l78_78612


namespace total_cost_is_9220_l78_78776

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end total_cost_is_9220_l78_78776


namespace work_completion_l78_78875

theorem work_completion (a b : Type) (work_done_together work_done_by_a work_done_by_b : ℝ) 
  (h1 : work_done_together = 1 / 12) 
  (h2 : work_done_by_a = 1 / 20) 
  (h3 : work_done_by_b = work_done_together - work_done_by_a) : 
  work_done_by_b = 1 / 30 :=
by
  sorry

end work_completion_l78_78875


namespace percentage_increase_in_allowance_l78_78216

def middle_school_allowance : ℕ := 8 + 2
def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

theorem percentage_increase_in_allowance : 
  (senior_year_allowance - middle_school_allowance) * 100 / middle_school_allowance = 150 := 
  by
    sorry

end percentage_increase_in_allowance_l78_78216


namespace probability_red_ball_l78_78422

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l78_78422


namespace balcony_more_than_orchestra_l78_78888

theorem balcony_more_than_orchestra (O B : ℕ) 
  (h1 : O + B = 355) 
  (h2 : 12 * O + 8 * B = 3320) : 
  B - O = 115 :=
by 
  -- Sorry, this will skip the proof.
  sorry

end balcony_more_than_orchestra_l78_78888


namespace line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l78_78775

-- Problem 1: The line passes through a fixed point
theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (1, -2) ∧ (∀ x y, k * x - y - 2 - k = 0 → P = (x, y)) :=
by
  sorry

-- Problem 2: Range of values for k if the line does not pass through the second quadrant
theorem range_of_k_no_second_quadrant (k : ℝ) : ¬ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ k * x - y - 2 - k = 0) → k ∈ Set.Ici (0) :=
by
  sorry

-- Problem 3: Minimum area of triangle AOB
theorem min_area_triangle (k : ℝ) :
  let A := (2 + k) / k
  let B := -2 - k
  (∀ x y, k * x - y - 2 - k = 0 ↔ (x = A ∧ y = 0) ∨ (x = 0 ∧ y = B)) →
  ∃ S : ℝ, S = 4 ∧ (∀ x y : ℝ, (k = 2 ∧ k * x - y - 4 = 0) → S = 4) :=
by
  sorry

end line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l78_78775


namespace quadratic_equation_solution_diff_l78_78073

theorem quadratic_equation_solution_diff :
  let a := 1
  let b := -6
  let c := -40
  let discriminant := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 := (-b - Real.sqrt discriminant) / (2 * a)
  abs (root1 - root2) = 14 := by
  -- placeholder for the proof
  sorry

end quadratic_equation_solution_diff_l78_78073


namespace prove_percentage_cats_adopted_each_month_l78_78752

noncomputable def percentage_cats_adopted_each_month
    (initial_dogs : ℕ)
    (initial_cats : ℕ)
    (initial_lizards : ℕ)
    (adopted_dogs_percent : ℕ)
    (adopted_lizards_percent : ℕ)
    (new_pets_each_month : ℕ)
    (total_pets_after_month : ℕ)
    (adopted_cats_percent : ℕ) : Prop :=
  initial_dogs = 30 ∧
  initial_cats = 28 ∧
  initial_lizards = 20 ∧
  adopted_dogs_percent = 50 ∧
  adopted_lizards_percent = 20 ∧
  new_pets_each_month = 13 ∧
  total_pets_after_month = 65 →
  adopted_cats_percent = 25

-- The condition to prove
theorem prove_percentage_cats_adopted_each_month :
  percentage_cats_adopted_each_month 30 28 20 50 20 13 65 25 :=
by 
  sorry

end prove_percentage_cats_adopted_each_month_l78_78752


namespace solve_equation_l78_78268

theorem solve_equation (x : ℝ) (h : x + 3 ≠ 0) : (2 / (x + 3) = 1) → (x = -1) :=
by
  intro h1
  -- Proof skipped
  sorry

end solve_equation_l78_78268


namespace find_k_l78_78892

def vector := ℝ × ℝ  -- Define a vector as a pair of real numbers

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a (k : ℝ) : vector := (k, 3)
def b : vector := (1, 4)
def c : vector := (2, 1)
def linear_combination (k : ℝ) : vector := ((2 * k - 3), -6)

theorem find_k (k : ℝ) (h : dot_product (linear_combination k) c = 0) : k = 3 := by
  sorry

end find_k_l78_78892


namespace value_of_a_minus_b_l78_78343

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 :=
sorry

end value_of_a_minus_b_l78_78343


namespace relationship_between_n_and_m_l78_78726

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def geometric_sequence (a q : ℝ) (m : ℕ) : ℝ :=
  a * q ^ (m - 1)

theorem relationship_between_n_and_m
  (a d q : ℝ) (n m : ℕ)
  (h_d_ne_zero : d ≠ 0)
  (h1 : arithmetic_sequence a d 1 = geometric_sequence a q 1)
  (h2 : arithmetic_sequence a d 3 = geometric_sequence a q 3)
  (h3 : arithmetic_sequence a d 7 = geometric_sequence a q 5)
  (q_pos : 0 < q) (q_sqrt2 : q^2 = 2)
  :
  n = 2 ^ ((m + 1) / 2) - 1 := sorry

end relationship_between_n_and_m_l78_78726


namespace stratified_sampling_third_grade_l78_78310

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end stratified_sampling_third_grade_l78_78310


namespace exam_students_count_l78_78595

theorem exam_students_count (n : ℕ) (T : ℕ) (h1 : T = 90 * n) 
                            (h2 : (T - 90) / (n - 2) = 95) : n = 20 :=
by {
  sorry
}

end exam_students_count_l78_78595


namespace area_ratio_of_regular_polygons_l78_78849

noncomputable def area_ratio (r : ℝ) : ℝ :=
  let A6 := (3 * Real.sqrt 3 / 2) * r^2
  let s8 := r * Real.sqrt (2 - Real.sqrt 2)
  let A8 := 2 * (1 + Real.sqrt 2) * (s8 ^ 2)
  A8 / A6

theorem area_ratio_of_regular_polygons (r : ℝ) :
  area_ratio r = 4 * (1 + Real.sqrt 2) * (2 - Real.sqrt 2) / (3 * Real.sqrt 3) :=
  sorry

end area_ratio_of_regular_polygons_l78_78849


namespace number_of_ideal_subsets_l78_78441

def is_ideal_subset (p q : ℕ) (S : Set ℕ) : Prop :=
  0 ∈ S ∧ ∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S

theorem number_of_ideal_subsets (p q : ℕ) (hpq : Nat.Coprime p q) :
  ∃ n, n = Nat.choose (p + q) p / (p + q) :=
sorry

end number_of_ideal_subsets_l78_78441


namespace calculation_of_product_l78_78546

theorem calculation_of_product : (0.09)^3 * 0.0007 = 0.0000005103 := 
by
  sorry

end calculation_of_product_l78_78546


namespace problem_statement_l78_78300
open Real

noncomputable def l1 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x - (cos α) * y + 1 = 0
noncomputable def l2 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x + (cos α) * y + 1 = 0
noncomputable def l3 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x - (sin α) * y + 1 = 0
noncomputable def l4 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x + (sin α) * y + 1 = 0

theorem problem_statement:
  (∃ (α : ℝ), ∀ (x y : ℝ), l1 α x y → l2 α x y) ∧
  (∀ (α : ℝ), ∀ (x y : ℝ), l1 α x y → (sin α) * (cos α) + (-cos α) * (sin α) = 0) ∧
  (∃ (p : ℝ × ℝ), ∀ (α : ℝ), abs ((sin α) * p.1 - (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((sin α) * p.1 + (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((cos α) * p.1 - (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1 ∧
                        abs ((cos α) * p.1 + (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1) :=
sorry

end problem_statement_l78_78300


namespace marys_age_l78_78229

variable (M R : ℕ) -- Define M (Mary's current age) and R (Rahul's current age) as natural numbers

theorem marys_age
  (h1 : R = M + 40)       -- Rahul is 40 years older than Mary
  (h2 : R + 30 = 3 * (M + 30))  -- In 30 years, Rahul will be three times as old as Mary
  : M = 20 := 
sorry  -- The proof goes here

end marys_age_l78_78229


namespace systematic_sampling_correct_l78_78176

-- Conditions as definitions
def total_bags : ℕ := 50
def num_samples : ℕ := 5
def interval (total num : ℕ) : ℕ := total / num
def correct_sequence : List ℕ := [5, 15, 25, 35, 45]

-- Statement
theorem systematic_sampling_correct :
  ∃ l : List ℕ, (l.length = num_samples) ∧ 
               (∀ i ∈ l, i ≤ total_bags) ∧
               (∀ i j, i < j → l.indexOf i < l.indexOf j → j - i = interval total_bags num_samples) ∧
               l = correct_sequence :=
by
  sorry

end systematic_sampling_correct_l78_78176


namespace Dima_claim_false_l78_78240

theorem Dima_claim_false (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a*x^2 + b*x + c = 0) → ∃ α β, α < 0 ∧ β < 0 ∧ (α + β = -b/a) ∧ (α*β = c/a)) :
  ¬ ∃ α' β', α' > 0 ∧ β' > 0 ∧ (α' + β' = -c/b) ∧ (α'*β' = a/b) :=
sorry

end Dima_claim_false_l78_78240


namespace f_2018_eq_2017_l78_78411

-- Define f(1) and f(2)
def f : ℕ → ℕ 
| 1 => 1
| 2 => 1
| n => if h : n ≥ 3 then (f (n - 1) - f (n - 2) + n) else 0

-- State the theorem to prove f(2018) = 2017
theorem f_2018_eq_2017 : f 2018 = 2017 := 
by 
  sorry

end f_2018_eq_2017_l78_78411


namespace inequality_and_equality_condition_l78_78051

theorem inequality_and_equality_condition (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 ≤ a * b) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧ (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) :=
by
  sorry

end inequality_and_equality_condition_l78_78051


namespace digits_subtraction_eq_zero_l78_78437

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero_l78_78437


namespace pears_weight_l78_78474

theorem pears_weight (x : ℕ) (h : 2 * x + 50 = 250) : x = 100 :=
sorry

end pears_weight_l78_78474


namespace smallest_four_digit_multiple_of_18_l78_78956

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n > 999 ∧ n < 10000 ∧ 18 ∣ n ∧ (∀ m : ℕ, m > 999 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l78_78956


namespace tan_add_l78_78847

theorem tan_add (α β : ℝ) (h1 : Real.tan (α - π / 6) = 3 / 7) (h2 : Real.tan (π / 6 + β) = 2 / 5) : Real.tan (α + β) = 1 := by
  sorry

end tan_add_l78_78847


namespace sin_double_angle_of_tan_l78_78632

theorem sin_double_angle_of_tan (α : ℝ) (hα1 : Real.tan α = 2) (hα2 : 0 < α ∧ α < Real.pi / 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_double_angle_of_tan_l78_78632


namespace fraction_of_25_exists_l78_78522

theorem fraction_of_25_exists :
  ∃ x : ℚ, 0.60 * 40 = x * 25 + 4 ∧ x = 4 / 5 :=
by
  simp
  sorry

end fraction_of_25_exists_l78_78522


namespace cost_prices_max_units_B_possible_scenarios_l78_78001

-- Part 1: Prove cost prices of Product A and B
theorem cost_prices (x : ℝ) (A B : ℝ) 
  (h₁ : B = x ∧ A = x - 2) 
  (h₂ : 80 / A = 100 / B) 
  : B = 10 ∧ A = 8 :=
by 
  sorry

-- Part 2: Prove maximum units of product B that can be purchased
theorem max_units_B (y : ℕ) 
  (h₁ : ∀ y : ℕ, 3 * y - 5 + y ≤ 95) 
  : y ≤ 25 :=
by 
  sorry

-- Part 3: Prove possible scenarios for purchasing products A and B
theorem possible_scenarios (y : ℕ) 
  (h₁ : y > 23 * 9/17 ∧ y ≤ 25) 
  : y = 24 ∨ y = 25 :=
by 
  sorry

end cost_prices_max_units_B_possible_scenarios_l78_78001


namespace locus_of_orthocenter_l78_78274

theorem locus_of_orthocenter (A_x A_y : ℝ) (h_A : A_x = 0 ∧ A_y = 2)
    (c_r : ℝ) (h_c : c_r = 2) 
    (M_x M_y Q_x Q_y : ℝ)
    (h_circle : Q_x^2 + Q_y^2 = c_r^2)
    (h_tangent : M_x ≠ 0 ∧ (M_y - 2) / M_x = -Q_x / Q_y)
    (h_M_on_tangent : M_x^2 + (M_y - 2)^2 = 4 ∧ M_x ≠ 0)
    (H_x H_y : ℝ)
    (h_orthocenter : (H_x - A_x)^2 + (H_y - A_y + 2)^2 = 4) :
    (H_x^2 + (H_y - 2)^2 = 4) ∧ (H_x ≠ 0) := 
sorry

end locus_of_orthocenter_l78_78274


namespace near_square_qoutient_l78_78192

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient_l78_78192


namespace megan_bottles_left_l78_78255

-- Defining the initial conditions
def initial_bottles : Nat := 17
def bottles_drank : Nat := 3

-- Theorem stating that Megan has 14 bottles left
theorem megan_bottles_left : initial_bottles - bottles_drank = 14 := by
  sorry

end megan_bottles_left_l78_78255


namespace knights_round_table_l78_78497

theorem knights_round_table (n : ℕ) (h : ∃ (f e : ℕ), f = e ∧ f + e = n) : n % 4 = 0 :=
sorry

end knights_round_table_l78_78497


namespace find_integer_n_l78_78467

theorem find_integer_n : ∃ n, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ n = 9 :=   
by 
  -- The proof will be written here.
  sorry

end find_integer_n_l78_78467


namespace city_of_archimedes_schools_l78_78541

noncomputable def numberOfSchools : ℕ := 32

theorem city_of_archimedes_schools :
  ∃ n : ℕ, (∀ s : Set ℕ, s = {45, 68, 113} →
  (∀ x ∈ s, x > 1 → 4 * n = x + 1 → (2 * n ≤ x ∧ 2 * n + 1 ≥ x) ))
  ∧ n = numberOfSchools :=
sorry

end city_of_archimedes_schools_l78_78541


namespace sequence_property_l78_78484

theorem sequence_property : 
  ∀ (a : ℕ → ℝ), 
    a 1 = 1 →
    a 2 = 1 → 
    (∀ n, a (n + 2) = a (n + 1) + 1 / a n) →
    a 180 > 19 :=
by
  intros a h1 h2 h3
  sorry

end sequence_property_l78_78484


namespace jason_borrowed_amount_l78_78544

def earning_per_six_hours : ℤ :=
  2 + 4 + 6 + 2 + 4 + 6

def total_hours_worked : ℤ :=
  48

def cycle_length : ℤ :=
  6

def total_cycles : ℤ :=
  total_hours_worked / cycle_length

def total_amount_borrowed : ℤ :=
  total_cycles * earning_per_six_hours

theorem jason_borrowed_amount : total_amount_borrowed = 192 :=
  by
    -- Here we use the definition and conditions to prove the equivalence
    -- of the calculation to the problem statement.
    sorry

end jason_borrowed_amount_l78_78544


namespace average_M_possibilities_l78_78848

theorem average_M_possibilities (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
    (12 = (8 + 15 + M) / 3) ∨ (15 = (8 + 15 + M) / 3) :=
  sorry

end average_M_possibilities_l78_78848


namespace solution_set_for_fractional_inequality_l78_78334

theorem solution_set_for_fractional_inequality :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end solution_set_for_fractional_inequality_l78_78334


namespace hoses_fill_time_l78_78179

noncomputable def time_to_fill_pool {P A B C : ℝ} (h₁ : A + B = P / 3) (h₂ : A + C = P / 4) (h₃ : B + C = P / 5) : ℝ :=
  (120 / 47 : ℝ)

theorem hoses_fill_time {P A B C : ℝ} 
  (h₁ : A + B = P / 3) 
  (h₂ : A + C = P / 4) 
  (h₃ : B + C = P / 5) 
  : time_to_fill_pool h₁ h₂ h₃ = (120 / 47 : ℝ) :=
sorry

end hoses_fill_time_l78_78179


namespace log_relationship_l78_78839

noncomputable def log_m (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem log_relationship (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  log_m m 0.3 > log_m m 0.5 :=
by
  sorry

end log_relationship_l78_78839


namespace correct_technology_used_l78_78980

-- Define the condition that the program title is "Back to the Dinosaur Era"
def program_title : String := "Back to the Dinosaur Era"

-- Define the condition that the program vividly recreated various dinosaurs and their living environments
def recreated_living_environments : Bool := true

-- Define the options for digital Earth technologies
inductive DigitalEarthTechnology
| InformationSuperhighway
| HighResolutionSatelliteTechnology
| SpatialInformationTechnology
| VisualizationAndVirtualRealityTechnology

-- Define the correct answer
def correct_technology := DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology

-- The proof problem: Prove that given the conditions, the technology used is the correct one
theorem correct_technology_used
  (title : program_title = "Back to the Dinosaur Era")
  (recreated : recreated_living_environments) :
  correct_technology = DigitalEarthTechnology.VisualizationAndVirtualRealityTechnology :=
by
  sorry

end correct_technology_used_l78_78980


namespace value_of_f_at_2_l78_78669

def f (x : ℤ) : ℤ := x^3 - x

theorem value_of_f_at_2 : f 2 = 6 := by
  sorry

end value_of_f_at_2_l78_78669


namespace mystery_book_shelves_l78_78319

-- Define the conditions from the problem
def total_books : ℕ := 72
def picture_book_shelves : ℕ := 2
def books_per_shelf : ℕ := 9

-- Determine the number of mystery book shelves
theorem mystery_book_shelves : 
  let books_on_picture_shelves := picture_book_shelves * books_per_shelf
  let mystery_books := total_books - books_on_picture_shelves
  let mystery_shelves := mystery_books / books_per_shelf
  mystery_shelves = 6 :=
by {
  -- This space is intentionally left incomplete, as the proof itself is not required.
  sorry
}

end mystery_book_shelves_l78_78319


namespace relationship_among_f_l78_78302

theorem relationship_among_f (
  f : ℝ → ℝ
) (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_increasing : ∀ a b, (0 ≤ a ∧ a < b ∧ b ≤ 1) → f a < f b) :
  f 2 < f (-5.5) ∧ f (-5.5) < f (-1) :=
by
  sorry

end relationship_among_f_l78_78302


namespace remaining_problems_to_grade_l78_78007

-- Define the conditions
def problems_per_worksheet : ℕ := 3
def total_worksheets : ℕ := 15
def graded_worksheets : ℕ := 7

-- The remaining worksheets to grade
def remaining_worksheets : ℕ := total_worksheets - graded_worksheets

-- Theorems stating the amount of problems left to grade
theorem remaining_problems_to_grade : problems_per_worksheet * remaining_worksheets = 24 :=
by
  sorry

end remaining_problems_to_grade_l78_78007


namespace probability_crisp_stops_on_dime_l78_78298

noncomputable def crisp_stops_on_dime_probability : ℚ :=
  let a := (2/3 : ℚ)
  let b := (1/3 : ℚ)
  let a1 := (15/31 : ℚ)
  let b1 := (30/31 : ℚ)
  (2 / 3) * a1 + (1 / 3) * b1

theorem probability_crisp_stops_on_dime :
  crisp_stops_on_dime_probability = 20 / 31 :=
by
  sorry

end probability_crisp_stops_on_dime_l78_78298


namespace positive_difference_is_329_l78_78782

-- Definitions of the fractions involved
def fraction1 : ℚ := (7^2 + 7^2) / 7
def fraction2 : ℚ := (7^2 * 7^2) / 7

-- Statement of the positive difference proof
theorem positive_difference_is_329 : abs (fraction2 - fraction1) = 329 := by
  -- Skipping the proof here
  sorry

end positive_difference_is_329_l78_78782


namespace exists_2016_integers_with_product_9_and_sum_0_l78_78672

theorem exists_2016_integers_with_product_9_and_sum_0 :
  ∃ (L : List ℤ), L.length = 2016 ∧ L.prod = 9 ∧ L.sum = 0 := by
  sorry

end exists_2016_integers_with_product_9_and_sum_0_l78_78672


namespace total_customers_is_40_l78_78593

-- The number of tables the waiter is attending
def num_tables : ℕ := 5

-- The number of women at each table
def women_per_table : ℕ := 5

-- The number of men at each table
def men_per_table : ℕ := 3

-- The total number of customers at each table
def customers_per_table : ℕ := women_per_table + men_per_table

-- The total number of customers the waiter has
def total_customers : ℕ := num_tables * customers_per_table

theorem total_customers_is_40 : total_customers = 40 :=
by
  -- Proof goes here
  sorry

end total_customers_is_40_l78_78593


namespace Jackson_missed_one_wednesday_l78_78611

theorem Jackson_missed_one_wednesday (weeks total_sandwiches missed_fridays sandwiches_eaten : ℕ) 
  (h1 : weeks = 36)
  (h2 : total_sandwiches = 2 * weeks)
  (h3 : missed_fridays = 2)
  (h4 : sandwiches_eaten = 69) :
  (total_sandwiches - missed_fridays - sandwiches_eaten) / 2 = 1 :=
by
  -- sorry to skip the proof.
  sorry

end Jackson_missed_one_wednesday_l78_78611


namespace total_teachers_in_all_departments_is_637_l78_78838

noncomputable def total_teachers : ℕ :=
  let major_departments := 9
  let minor_departments := 8
  let teachers_per_major := 45
  let teachers_per_minor := 29
  (major_departments * teachers_per_major) + (minor_departments * teachers_per_minor)

theorem total_teachers_in_all_departments_is_637 : total_teachers = 637 := 
  by
  sorry

end total_teachers_in_all_departments_is_637_l78_78838


namespace incorrect_statement_l78_78698

theorem incorrect_statement : ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) :=
by
  -- Proof goes here
  sorry

end incorrect_statement_l78_78698


namespace NicoleEndsUpWith36Pieces_l78_78444

namespace ClothingProblem

noncomputable def NicoleClothesStart := 10
noncomputable def FirstOlderSisterClothes := NicoleClothesStart / 2
noncomputable def NextOldestSisterClothes := NicoleClothesStart + 2
noncomputable def OldestSisterClothes := (NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes) / 3

theorem NicoleEndsUpWith36Pieces : 
  NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes + OldestSisterClothes = 36 :=
  by
    sorry

end ClothingProblem

end NicoleEndsUpWith36Pieces_l78_78444


namespace find_a_if_f_is_odd_l78_78206

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_if_f_is_odd :
  (∀ x : ℝ, f 1 x = -f 1 (-x)) ↔ (1 = 1) :=
by
  sorry

end find_a_if_f_is_odd_l78_78206


namespace arithmetic_geom_seq_l78_78608

noncomputable def geom_seq (a q : ℝ) : ℕ → ℝ 
| 0     => a
| (n+1) => q * (geom_seq a q n)

theorem arithmetic_geom_seq
  (a q : ℝ)
  (h_arith : 2 * geom_seq a q 1 = 1 + (geom_seq a q 2 - 1))
  (h_q : q = 2) :
  (geom_seq a q 2 + geom_seq a q 3) / (geom_seq a q 4 + geom_seq a q 5) = 1 / 4 :=
by
  sorry

end arithmetic_geom_seq_l78_78608


namespace range_of_a_l78_78452

variable {x a : ℝ}

def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (ha : a < 0) 
  (H : (∀ x, ¬ p a x → q x) ∧ ∃ x, q x ∧ ¬ p a x ∧ ¬ q x) : a ≤ -4 := 
sorry

end range_of_a_l78_78452


namespace real_roots_for_all_K_l78_78330

theorem real_roots_for_all_K (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x-1) * (x-2) + 2 * x :=
sorry

end real_roots_for_all_K_l78_78330


namespace steve_final_height_l78_78792

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l78_78792


namespace find_other_digits_l78_78909

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem find_other_digits (n : ℕ) (h : ℕ) :
  tens_digit n = h →
  h = 1 →
  is_divisible_by_9 n →
  ∃ m : ℕ, m < 9 ∧ n = 10 * ((n / 10) / 10) * 10 + h * 10 + m ∧ (∃ k : ℕ, k * 9 = h + m + (n / 100)) :=
sorry

end find_other_digits_l78_78909


namespace rectangle_sides_l78_78886

theorem rectangle_sides (x y : ℕ) :
  (2 * x + 2 * y = x * y) →
  x > 0 →
  y > 0 →
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) :=
by
  sorry

end rectangle_sides_l78_78886


namespace rain_puddle_depth_l78_78979

theorem rain_puddle_depth
  (rain_rate : ℝ) (wait_time : ℝ) (puddle_area : ℝ) 
  (h_rate : rain_rate = 10) (h_time : wait_time = 3) (h_area : puddle_area = 300) :
  ∃ (depth : ℝ), depth = rain_rate * wait_time :=
by
  use 30
  simp [h_rate, h_time]
  sorry

end rain_puddle_depth_l78_78979


namespace cost_of_each_barbell_l78_78427

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l78_78427


namespace rank_friends_l78_78354

variable (Amy Bill Celine : Prop)

-- Statement definitions
def statement_I := Bill
def statement_II := ¬Amy
def statement_III := ¬Celine

-- Exactly one of the statements is true
def exactly_one_true (s1 s2 s3 : Prop) :=
  (s1 ∧ ¬s2 ∧ ¬s3) ∨ (¬s1 ∧ s2 ∧ ¬s3) ∨ (¬s1 ∧ ¬s2 ∧ s3)

theorem rank_friends (h : exactly_one_true (statement_I Bill) (statement_II Amy) (statement_III Celine)) :
  (Amy ∧ ¬Bill ∧ Celine) :=
sorry

end rank_friends_l78_78354


namespace direct_proportion_l78_78529

theorem direct_proportion : 
  ∃ k, (∀ x, y = k * x) ↔ (y = -2 * x) :=
by
  sorry

end direct_proportion_l78_78529


namespace smallest_int_with_18_divisors_l78_78627

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l78_78627


namespace max_value_of_function_l78_78981

/-- Let y(x) = a^(2*x) + 2 * a^x - 1 for a positive real number a and x in [-1, 1].
    Prove that the maximum value of y on the interval [-1, 1] is 14 when a = 1/3 or a = 3. -/
theorem max_value_of_function (a : ℝ) (a_pos : 0 < a) (h : a = 1 / 3 ∨ a = 3) : 
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 14 := 
sorry

end max_value_of_function_l78_78981


namespace odd_function_product_nonpositive_l78_78263

noncomputable def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_product_nonpositive (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) : 
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by 
  sorry

end odd_function_product_nonpositive_l78_78263


namespace mass_of_man_l78_78291

variable (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)

-- Given conditions
def boatLength := L = 3
def boatBreadth := B = 2
def sinkingDepth := h = 0.018
def waterDensity := ρ = 1000

-- The mass of the man
theorem mass_of_man (L B h ρ : ℝ) (H1 : boatLength L) (H2 : boatBreadth B) (H3 : sinkingDepth h) (H4 : waterDensity ρ) : 
  ρ * L * B * h = 108 := by
  sorry

end mass_of_man_l78_78291


namespace unit_stratified_sampling_l78_78988

theorem unit_stratified_sampling 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (selected_elderly : ℕ)
  (total : ℕ) (n : ℕ)
  (h1 : elderly = 27)
  (h2 : middle_aged = 54)
  (h3 : young = 81)
  (h4 : selected_elderly = 3)
  (h5 : total = elderly + middle_aged + young)
  (h6 : 3 / 27 = selected_elderly / elderly)
  (h7 : n / total = selected_elderly / elderly) : 
  n = 18 := 
by
  sorry

end unit_stratified_sampling_l78_78988


namespace value_of_certain_number_l78_78927

theorem value_of_certain_number (a b : ℕ) (h : 1 / 7 * 8 = 5) (h2 : 1 / 5 * b = 35) : b = 175 :=
by
  -- by assuming the conditions hold, we need to prove b = 175
  sorry

end value_of_certain_number_l78_78927


namespace a_pow_m_minus_a_pow_n_divisible_by_30_l78_78854

theorem a_pow_m_minus_a_pow_n_divisible_by_30
  (a m n k : ℕ)
  (h_n_ge_two : n ≥ 2)
  (h_m_gt_n : m > n)
  (h_m_n_diff : m = n + 4 * k) :
  30 ∣ (a ^ m - a ^ n) :=
sorry

end a_pow_m_minus_a_pow_n_divisible_by_30_l78_78854


namespace picture_area_l78_78475

theorem picture_area (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h_area : (3 * x + 4) * (y + 3) = 60) : x * y = 15 := 
by 
  sorry

end picture_area_l78_78475


namespace total_theme_parks_l78_78524

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end total_theme_parks_l78_78524


namespace cycling_problem_l78_78365

theorem cycling_problem (x : ℝ) (h₀ : x > 0) :
  30 / x - 30 / (x + 3) = 2 / 3 :=
sorry

end cycling_problem_l78_78365


namespace meters_of_cloth_sold_l78_78716

-- Definitions based on conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 20
def cost_price_per_meter : ℕ := 85
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof statement
theorem meters_of_cloth_sold : ∃ x : ℕ, selling_price_per_meter * x = total_selling_price ∧ x = 85 := by
  sorry

end meters_of_cloth_sold_l78_78716


namespace joan_missed_games_l78_78575

variable (total_games : ℕ) (night_games : ℕ) (attended_games : ℕ)

theorem joan_missed_games (h1 : total_games = 864) (h2 : night_games = 128) (h3 : attended_games = 395) : 
  total_games - attended_games = 469 :=
  by
    sorry

end joan_missed_games_l78_78575


namespace solve_inequality_l78_78490

open Set

theorem solve_inequality :
  { x : ℝ | (2 * x - 2) / (x^2 - 5*x + 6) ≤ 3 } = Ioo (5/3) 2 ∪ Icc 3 4 :=
by
  sorry

end solve_inequality_l78_78490


namespace real_part_of_z1_is_zero_l78_78976

-- Define the imaginary unit i with its property
def i := Complex.I

-- Define z1 using the given expression
noncomputable def z1 := (1 - 2 * i) / (2 + i^5)

-- State the theorem about the real part of z1
theorem real_part_of_z1_is_zero : z1.re = 0 :=
by
  sorry

end real_part_of_z1_is_zero_l78_78976


namespace value_of_a2_l78_78173

theorem value_of_a2 (a0 a1 a2 a3 a4 : ℝ) (x : ℝ) 
  (h : x^4 = a0 + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3 + a4 * (x - 2)^4) :
  a2 = 24 :=
sorry

end value_of_a2_l78_78173


namespace tank_empty_time_l78_78817

def tank_capacity : ℝ := 6480
def leak_time : ℝ := 6
def inlet_rate_per_minute : ℝ := 4.5
def inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

theorem tank_empty_time : tank_capacity / (tank_capacity / leak_time - inlet_rate_per_hour) = 8 := 
by
  sorry

end tank_empty_time_l78_78817


namespace tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l78_78083

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem tangent_line_at_x_equals_1 (a : ℝ) (x : ℝ) (h₀ : a = 2) (h₁ : x = 1) : 
  3 * x - (f a 1) - 1 = 0 := 
sorry

theorem monotonic_intervals (a x : ℝ) (h₀ : x > 0) :
  ((a >= 0 ∧ ∀ (x : ℝ), x > 0 → (f a x) > (f a (x - 1))) ∨ 
  (a < 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < -1/a → (f a x) > (f a (x - 1)) ∧ ∀ (x : ℝ), x > -1/a → (f a x) < (f a (x - 1)))) :=
sorry

theorem range_of_a (a x : ℝ) (h₀ : 0 < x) (h₁ : f a x < 2) : a < -1 / Real.exp (3) :=
sorry

end tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l78_78083


namespace twenty_is_80_percent_of_what_number_l78_78345

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l78_78345


namespace ratio_of_selling_prices_l78_78224

theorem ratio_of_selling_prices (C SP1 SP2 : ℝ)
  (h1 : SP1 = C + 0.20 * C)
  (h2 : SP2 = C + 1.40 * C) :
  SP2 / SP1 = 2 := by
  sorry

end ratio_of_selling_prices_l78_78224


namespace volume_increase_l78_78678

theorem volume_increase (l w h: ℕ) 
(h1: l * w * h = 4320) 
(h2: l * w + w * h + h * l = 852) 
(h3: l + w + h = 52) : 
(l + 1) * (w + 1) * (h + 1) = 5225 := 
by 
  sorry

end volume_increase_l78_78678


namespace moles_of_CH4_needed_l78_78069

theorem moles_of_CH4_needed
  (moles_C6H6_needed : ℕ)
  (reaction_balance : ∀ (C6H6 CH4 C6H5CH3 H2 : ℕ), 
    C6H6 + CH4 = C6H5CH3 + H2 → C6H6 = 1 ∧ CH4 = 1 ∧ C6H5CH3 = 1 ∧ H2 = 1)
  (H : moles_C6H6_needed = 3) :
  (3 : ℕ) = 3 :=
by 
  -- The actual proof would go here
  sorry

end moles_of_CH4_needed_l78_78069


namespace theta_in_fourth_quadrant_l78_78650

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan (θ + Real.pi / 4) = 1 / 3) : 
  (θ > 3 * Real.pi / 2) ∧ (θ < 2 * Real.pi) :=
sorry

end theta_in_fourth_quadrant_l78_78650


namespace train_to_platform_ratio_l78_78551

-- Define the given conditions as assumptions
def speed_kmh : ℕ := 54 -- speed of the train in km/hr
def train_length_m : ℕ := 450 -- length of the train in meters
def crossing_time_min : ℕ := 1 -- time to cross the platform in minutes

-- Conversion from km/hr to m/min
def speed_mpm : ℕ := (speed_kmh * 1000) / 60

-- Calculate the total distance covered in one minute
def total_distance_m : ℕ := speed_mpm * crossing_time_min

-- Define the length of the platform
def platform_length_m : ℕ := total_distance_m - train_length_m

-- The proof statement to show the ratio of the lengths
theorem train_to_platform_ratio : train_length_m = platform_length_m :=
by 
  -- following from the definition of platform_length_m
  sorry

end train_to_platform_ratio_l78_78551


namespace find_chosen_number_l78_78418

-- Define the conditions
def condition (x : ℝ) : Prop := (3 / 2) * x + 53.4 = -78.9

-- State the theorem
theorem find_chosen_number : ∃ x : ℝ, condition x ∧ x = -88.2 :=
sorry

end find_chosen_number_l78_78418


namespace solve_system_of_equations_l78_78816

theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : 2 * x + y = 21) : x + y = 9 := by
  sorry

end solve_system_of_equations_l78_78816


namespace angle_measure_l78_78986

-- Define the complement function
def complement (α : ℝ) : ℝ := 180 - α

-- Given condition
variable (α : ℝ)
variable (h : complement α = 120)

-- Theorem to prove
theorem angle_measure : α = 60 :=
by sorry

end angle_measure_l78_78986


namespace average_of_solutions_l78_78946

theorem average_of_solutions (a b : ℝ) :
  (∃ x1 x2 : ℝ, 3 * a * x1^2 - 6 * a * x1 + 2 * b = 0 ∧
                3 * a * x2^2 - 6 * a * x2 + 2 * b = 0 ∧
                x1 ≠ x2) →
  (1 + 1) / 2 = 1 :=
by
  intros
  sorry

end average_of_solutions_l78_78946


namespace find_n_l78_78518

noncomputable def b_0 : ℝ := Real.cos (Real.pi / 18) ^ 2

noncomputable def b_n (n : ℕ) : ℝ :=
if n = 0 then b_0 else 4 * (b_n (n - 1)) * (1 - (b_n (n - 1)))

theorem find_n : ∀ n : ℕ, b_n n = b_0 → n = 24 := 
sorry

end find_n_l78_78518


namespace solution_set_empty_l78_78409

theorem solution_set_empty (x : ℝ) : ¬ (|x| + |2023 - x| < 2023) :=
by
  sorry

end solution_set_empty_l78_78409


namespace problem1_problem2_problem3_l78_78386

-- Problem 1
theorem problem1
  (α : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (hα : 0 < α ∧ α < 2 * Real.pi / 3) :
  (a + b) • (a - b) = 0 :=
sorry

-- Problem 2
theorem problem2
  (α k : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (x : ℝ × ℝ := k • a + 3 • b)
  (y : ℝ × ℝ := a + (1 / k) • b)
  (hk : 0 < k)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hxy : x • y = 0) :
  k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0 :=
sorry

-- Problem 3
theorem problem3
  (α k : ℝ)
  (h_eq : k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hk : 0 < k) :
  Real.pi / 2 ≤ α ∧ α < 2 * Real.pi / 3 :=
sorry

end problem1_problem2_problem3_l78_78386


namespace min_value_x2_y2_l78_78043

theorem min_value_x2_y2 (x y : ℝ) (h : x^3 + y^3 + 3 * x * y = 1) : x^2 + y^2 ≥ 1 / 2 :=
by
  -- We are required to prove the minimum value of x^2 + y^2 given the condition is 1/2
  sorry

end min_value_x2_y2_l78_78043


namespace total_payment_correct_l78_78592

-- Define the conditions for each singer
def firstSingerPayment : ℝ := 2 * 25
def secondSingerPayment : ℝ := 3 * 35
def thirdSingerPayment : ℝ := 4 * 20
def fourthSingerPayment : ℝ := 2.5 * 30

def firstSingerTip : ℝ := 0.15 * firstSingerPayment
def secondSingerTip : ℝ := 0.20 * secondSingerPayment
def thirdSingerTip : ℝ := 0.25 * thirdSingerPayment
def fourthSingerTip : ℝ := 0.18 * fourthSingerPayment

def firstSingerTotal : ℝ := firstSingerPayment + firstSingerTip
def secondSingerTotal : ℝ := secondSingerPayment + secondSingerTip
def thirdSingerTotal : ℝ := thirdSingerPayment + thirdSingerTip
def fourthSingerTotal : ℝ := fourthSingerPayment + fourthSingerTip

-- Define the total amount paid
def totalPayment : ℝ := firstSingerTotal + secondSingerTotal + thirdSingerTotal + fourthSingerTotal

-- The proof problem: Prove the total amount paid
theorem total_payment_correct : totalPayment = 372 := by
  sorry

end total_payment_correct_l78_78592


namespace solve_quadratic_eqn_l78_78477

theorem solve_quadratic_eqn :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ (x = 2 ∨ x = -3) :=
by
  intros
  simp
  sorry

end solve_quadratic_eqn_l78_78477


namespace incorrect_weight_estimation_l78_78804

variables (x y : ℝ)

/-- Conditions -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.71

/-- Incorrect conclusion -/
theorem incorrect_weight_estimation : regression_equation 160 ≠ 50.29 :=
by 
  sorry

end incorrect_weight_estimation_l78_78804


namespace polynomial_quotient_l78_78554

theorem polynomial_quotient : 
  (12 * x^3 + 20 * x^2 - 7 * x + 4) / (3 * x + 4) = 4 * x^2 + (4/3) * x - 37/9 :=
by
  sorry

end polynomial_quotient_l78_78554


namespace total_points_l78_78347

noncomputable def Noa_score : ℕ := 30
noncomputable def Phillip_score : ℕ := 2 * Noa_score
noncomputable def Lucy_score : ℕ := (3 / 2) * Phillip_score

theorem total_points : 
  Noa_score + Phillip_score + Lucy_score = 180 := 
by
  sorry

end total_points_l78_78347


namespace total_bulbs_is_118_l78_78615

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end total_bulbs_is_118_l78_78615


namespace tire_circumference_constant_l78_78438

/--
Given the following conditions:
1. Car speed v = 120 km/h
2. Tire rotation rate n = 400 rpm
3. Tire pressure P = 32 psi
4. Tire radius changes according to the formula R = R_0(1 + kP)
5. R_0 is the initial tire radius
6. k is a constant relating to the tire's elasticity
7. Change in tire pressure due to the incline is negligible

Prove that the circumference C of the tire is 5 meters.
-/
theorem tire_circumference_constant (v : ℝ) (n : ℝ) (P : ℝ) (R_0 : ℝ) (k : ℝ) 
  (h1 : v = 120 * 1000 / 3600) -- Car speed in m/s
  (h2 : n = 400 / 60)           -- Tire rotation rate in rps
  (h3 : P = 32)                 -- Tire pressure in psi
  (h4 : ∀ R P, R = R_0 * (1 + k * P)) -- Tire radius formula
  (h5 : ∀ P, P = 0)             -- Negligible change in tire pressure
  : C = 5 :=
  sorry

end tire_circumference_constant_l78_78438


namespace quotient_calculation_l78_78711

theorem quotient_calculation (dividend divisor remainder expected_quotient : ℕ)
  (h₁ : dividend = 166)
  (h₂ : divisor = 18)
  (h₃ : remainder = 4)
  (h₄ : dividend = divisor * expected_quotient + remainder) :
  expected_quotient = 9 :=
by
  sorry

end quotient_calculation_l78_78711


namespace curve_intersection_l78_78410

theorem curve_intersection (a m : ℝ) (a_pos : 0 < a) :
  (∀ x y : ℝ, 
     (x^2 / a^2 + y^2 = 1) ∧ (y^2 = 2 * (x + m)) 
     → 
     (1 / 2 * (a^2 + 1) = m) ∨ (-a < m ∧ m <= a))
  ∨ (a >= 1 → -a < m ∧ m < a) := 
sorry

end curve_intersection_l78_78410


namespace sum_of_reflected_coordinates_l78_78900

noncomputable def sum_of_coordinates (C D : ℝ × ℝ) : ℝ :=
  C.1 + C.2 + D.1 + D.2

theorem sum_of_reflected_coordinates (y : ℝ) :
  let C := (3, y)
  let D := (3, -y)
  sum_of_coordinates C D = 6 :=
by
  sorry

end sum_of_reflected_coordinates_l78_78900


namespace Tammy_earnings_3_weeks_l78_78131

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l78_78131


namespace order_of_values_l78_78009

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log Real.pi / Real.pi
noncomputable def d : ℝ := Real.log 2.72 / 2.72
noncomputable def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_values : a < f ∧ f < c ∧ c < b ∧ b < d :=
by
  sorry

end order_of_values_l78_78009


namespace solve_fractional_eq_l78_78184

-- Defining the fractional equation as a predicate
def fractional_eq (x : ℝ) : Prop :=
  (5 / x) = (7 / (x - 2))

-- The main theorem to be proven
theorem solve_fractional_eq : ∃ x : ℝ, fractional_eq x ∧ x = -5 := by
  sorry

end solve_fractional_eq_l78_78184


namespace sally_cost_is_42000_l78_78492

-- Definitions for conditions
def lightningCost : ℕ := 140000
def materCost : ℕ := (10 * lightningCost) / 100
def sallyCost : ℕ := 3 * materCost

-- Theorem statement
theorem sally_cost_is_42000 : sallyCost = 42000 := by
  sorry

end sally_cost_is_42000_l78_78492


namespace power_division_result_l78_78421

theorem power_division_result : (-2)^(2014) / (-2)^(2013) = -2 :=
by
  sorry

end power_division_result_l78_78421


namespace product_of_perimeters_correct_l78_78239

noncomputable def area (side_length : ℝ) : ℝ := side_length * side_length

theorem product_of_perimeters_correct (x y : ℝ)
  (h1 : area x + area y = 85)
  (h2 : area x - area y = 45) :
  4 * x * 4 * y = 32 * Real.sqrt 325 :=
by sorry

end product_of_perimeters_correct_l78_78239


namespace value_of_f_l78_78120

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom f_has_derivative : ∀ x, deriv f x = f' x
axiom f_equation : ∀ x, f x = 3 * x^2 + 2 * x * (f' 1)

-- Proof goal
theorem value_of_f'_at_3 : f' 3 = 6 := by
  sorry

end value_of_f_l78_78120


namespace g_10_plus_g_neg10_eq_6_l78_78272

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end g_10_plus_g_neg10_eq_6_l78_78272


namespace algebraic_expression_value_l78_78251

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^2 + a + 1 = 2 :=
sorry

end algebraic_expression_value_l78_78251


namespace unique_solution_pair_l78_78220

open Real

theorem unique_solution_pair :
  ∃! (x y : ℝ), y = (x-1)^2 ∧ x * y - y = -3 :=
sorry

end unique_solution_pair_l78_78220


namespace find_investment_amount_l78_78933

noncomputable def brokerage_fee (market_value : ℚ) : ℚ := (1 / 4 / 100) * market_value

noncomputable def actual_cost (market_value : ℚ) : ℚ := market_value + brokerage_fee market_value

noncomputable def income_per_100_face_value (interest_rate : ℚ) : ℚ := (interest_rate / 100) * 100

noncomputable def investment_amount (income : ℚ) (actual_cost_per_100 : ℚ) (income_per_100 : ℚ) : ℚ :=
  (income * actual_cost_per_100) / income_per_100

theorem find_investment_amount :
  investment_amount 756 (actual_cost 124.75) (income_per_100_face_value 10.5) = 9483.65625 :=
sorry

end find_investment_amount_l78_78933


namespace intersection_of_A_and_B_l78_78187

section intersection_proof

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x + 1 > 0}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := 
by {
  sorry
}

end intersection_proof

end intersection_of_A_and_B_l78_78187


namespace fraction_equation_solution_l78_78118

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) : 
  (1 / (x - 2) = 3 / x) → x = 3 := 
by 
  sorry

end fraction_equation_solution_l78_78118


namespace smallest_number_of_students_in_debate_club_l78_78034

-- Define conditions
def ratio_8th_to_6th (x₈ x₆ : ℕ) : Prop := 7 * x₆ = 4 * x₈
def ratio_8th_to_7th (x₈ x₇ : ℕ) : Prop := 6 * x₇ = 5 * x₈
def ratio_8th_to_9th (x₈ x₉ : ℕ) : Prop := 9 * x₉ = 2 * x₈

-- Problem statement
theorem smallest_number_of_students_in_debate_club 
  (x₈ x₆ x₇ x₉ : ℕ) 
  (h₁ : ratio_8th_to_6th x₈ x₆) 
  (h₂ : ratio_8th_to_7th x₈ x₇) 
  (h₃ : ratio_8th_to_9th x₈ x₉) : 
  x₈ + x₆ + x₇ + x₉ = 331 := 
sorry

end smallest_number_of_students_in_debate_club_l78_78034


namespace g_of_neg_2_l78_78447

def f (x : ℚ) : ℚ := 4 * x - 9

def g (y : ℚ) : ℚ :=
  3 * ((y + 9) / 4)^2 - 4 * ((y + 9) / 4) + 2

theorem g_of_neg_2 : g (-2) = 67 / 16 :=
by
  sorry

end g_of_neg_2_l78_78447


namespace correct_average_of_15_numbers_l78_78175

theorem correct_average_of_15_numbers
  (initial_average : ℝ)
  (num_numbers : ℕ)
  (incorrect1 incorrect2 correct1 correct2 : ℝ)
  (initial_average_eq : initial_average = 37)
  (num_numbers_eq : num_numbers = 15)
  (incorrect1_eq : incorrect1 = 52)
  (incorrect2_eq : incorrect2 = 39)
  (correct1_eq : correct1 = 64)
  (correct2_eq : correct2 = 27) :
  (initial_average * num_numbers - incorrect1 - incorrect2 + correct1 + correct2) / num_numbers = 37 :=
by
  rw [initial_average_eq, num_numbers_eq, incorrect1_eq, incorrect2_eq, correct1_eq, correct2_eq]
  sorry

end correct_average_of_15_numbers_l78_78175


namespace max_mass_of_grain_l78_78932

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l78_78932


namespace num_integers_satisfying_inequality_l78_78346

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l78_78346


namespace relationship_y1_y2_y3_l78_78991

def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variable (m : ℝ) (y1 y2 y3 : ℝ)

-- Given conditions
axiom m_gt_zero : m > 0
axiom point1_on_graph : y1 = quadratic m (-1)
axiom point2_on_graph : y2 = quadratic m (5 / 2)
axiom point3_on_graph : y3 = quadratic m 6

-- Prove the relationship between y1, y2, and y3
theorem relationship_y1_y2_y3 : y3 > y1 ∧ y1 > y2 :=
by sorry

end relationship_y1_y2_y3_l78_78991


namespace final_height_of_tree_in_4_months_l78_78186

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end final_height_of_tree_in_4_months_l78_78186


namespace part_one_part_two_l78_78248

noncomputable def problem_conditions (θ : ℝ) : Prop :=
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  ∃ m : ℝ, (∀ x : ℝ, x^2 - (Real.sqrt 3 - 1) * x + m = 0 → (x = sin_theta ∨ x = cos_theta))

theorem part_one (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let m := sin_theta * cos_theta
  m = (3 - 2 * Real.sqrt 3) / 2 :=
sorry

theorem part_two (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let tan_theta := sin_theta / cos_theta
  (cos_theta - sin_theta * tan_theta) / (1 - tan_theta) = Real.sqrt 3 - 1 :=
sorry

end part_one_part_two_l78_78248


namespace sausage_left_l78_78712

variables (S x y : ℝ)

-- Conditions
axiom dog_bites : y = x + 300
axiom cat_bites : x = y + 500

-- Theorem Statement
theorem sausage_left {S x y : ℝ}
  (h1 : y = x + 300)
  (h2 : x = y + 500) : S - x - y = 400 :=
by
  sorry

end sausage_left_l78_78712


namespace steel_scrap_problem_l78_78813

theorem steel_scrap_problem 
  (x y : ℝ)
  (h1 : x + y = 140)
  (h2 : 0.05 * x + 0.40 * y = 42) :
  x = 40 ∧ y = 100 :=
by
  -- Solution steps are not required here
  sorry

end steel_scrap_problem_l78_78813


namespace total_triangles_in_grid_l78_78789

-- Conditions
def bottom_row_triangles : Nat := 3
def next_row_triangles : Nat := 2
def top_row_triangles : Nat := 1
def additional_triangle : Nat := 1

def small_triangles := bottom_row_triangles + next_row_triangles + top_row_triangles + additional_triangle

-- Combining the triangles into larger triangles
def larger_triangles := 1 -- Formed by combining 4 small triangles
def largest_triangle := 1 -- Formed by combining all 7 small triangles

-- Math proof problem
theorem total_triangles_in_grid : small_triangles + larger_triangles + largest_triangle = 9 :=
by
  sorry

end total_triangles_in_grid_l78_78789


namespace necklace_ratio_l78_78961

variable {J Q H : ℕ}

theorem necklace_ratio (h1 : H = J + 5) (h2 : H = 25) (h3 : H = Q + 15) : Q / J = 1 / 2 := by
  sorry

end necklace_ratio_l78_78961


namespace executive_board_elections_l78_78260

noncomputable def num_candidates : ℕ := 18
noncomputable def num_positions : ℕ := 6
noncomputable def num_former_board_members : ℕ := 8

noncomputable def total_selections := Nat.choose num_candidates num_positions
noncomputable def no_former_board_members_selections := Nat.choose (num_candidates - num_former_board_members) num_positions

noncomputable def valid_selections := total_selections - no_former_board_members_selections

theorem executive_board_elections : valid_selections = 18354 :=
by sorry

end executive_board_elections_l78_78260


namespace waiters_hired_correct_l78_78397

noncomputable def waiters_hired (W H : ℕ) : Prop :=
  let cooks := 9
  (cooks / W = 3 / 8) ∧ (cooks / (W + H) = 1 / 4) ∧ (H = 12)

theorem waiters_hired_correct (W H : ℕ) : waiters_hired W H :=
  sorry

end waiters_hired_correct_l78_78397


namespace evaluate_expression_l78_78906

-- Define the base value
def base := 3000

-- Define the exponential expression
def exp_value := base ^ base

-- Prove that base * exp_value equals base ^ (1 + base)
theorem evaluate_expression : base * exp_value = base ^ (1 + base) := by
  sorry

end evaluate_expression_l78_78906


namespace unique_positive_integer_b_quadratic_solution_l78_78758

theorem unique_positive_integer_b_quadratic_solution (c : ℝ) :
  (∃! (b : ℕ), ∀ (x : ℝ), x^2 + (b^2 + (1 / b^2)) * x + c = 3) ↔ c = 5 :=
sorry

end unique_positive_integer_b_quadratic_solution_l78_78758


namespace initial_plan_days_l78_78830

-- Define the given conditions in Lean
variables (D : ℕ) -- Initial planned days for completing the job
variables (P : ℕ) -- Number of people initially hired
variables (Q : ℕ) -- Number of people fired
variables (W1 : ℚ) -- Portion of the work done before firing people
variables (D1 : ℕ) -- Days taken to complete W1 portion of work
variables (W2 : ℚ) -- Remaining portion of the work done after firing people
variables (D2 : ℕ) -- Days taken to complete W2 portion of work

-- Conditions from the problem
axiom h1 : P = 10
axiom h2 : Q = 2
axiom h3 : W1 = 1 / 4
axiom h4 : D1 = 20
axiom h5 : W2 = 3 / 4
axiom h6 : D2 = 75

-- The main theorem that proves the total initially planned days were 80
theorem initial_plan_days : D = 80 :=
sorry

end initial_plan_days_l78_78830


namespace game_cost_proof_l78_78214

variable (initial : ℕ) (allowance : ℕ) (final : ℕ) (cost : ℕ)

-- Initial amount
def initial_money : ℕ := 11
-- Allowance received
def allowance_money : ℕ := 14
-- Final amount of money
def final_money : ℕ := 22
-- Cost of the new game is to be proved
def game_cost : ℕ :=  initial_money - (final_money - allowance_money)

theorem game_cost_proof : game_cost = 3 := by
  sorry

end game_cost_proof_l78_78214


namespace remainder_2011_2015_mod_23_l78_78579

theorem remainder_2011_2015_mod_23 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := 
by
  sorry

end remainder_2011_2015_mod_23_l78_78579


namespace probability_at_least_one_humanities_l78_78210

theorem probability_at_least_one_humanities :
  let morning_classes := ["mathematics", "Chinese", "politics", "geography"]
  let afternoon_classes := ["English", "history", "physical_education"]
  let humanities := ["politics", "history", "geography"]
  let total_choices := List.length morning_classes * List.length afternoon_classes
  let favorable_morning := List.length (List.filter (fun x => x ∈ humanities) morning_classes)
  let favorable_afternoon := List.length (List.filter (fun x => x ∈ humanities) afternoon_classes)
  let favorable_choices := favorable_morning * List.length afternoon_classes + favorable_afternoon * (List.length morning_classes - favorable_morning)
  (favorable_choices / total_choices) = (2 / 3) := by sorry

end probability_at_least_one_humanities_l78_78210


namespace weight_of_replaced_student_l78_78114

-- Define the conditions as hypotheses
variable (W : ℝ)
variable (h : W - 46 = 40)

-- Prove that W = 86
theorem weight_of_replaced_student : W = 86 :=
by
  -- We should conclude the proof; for now, we leave a placeholder
  sorry

end weight_of_replaced_student_l78_78114


namespace coat_price_reduction_l78_78741

theorem coat_price_reduction:
  ∀ (original_price reduction_amount : ℕ),
  original_price = 500 →
  reduction_amount = 350 →
  (reduction_amount : ℝ) / original_price * 100 = 70 :=
by
  intros original_price reduction_amount h1 h2
  sorry

end coat_price_reduction_l78_78741


namespace sine_of_negative_90_degrees_l78_78160

theorem sine_of_negative_90_degrees : Real.sin (-(Real.pi / 2)) = -1 := 
sorry

end sine_of_negative_90_degrees_l78_78160


namespace find_numbers_l78_78012

theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b) = Real.sqrt 5) ∧ 
  (2 * a * b / (a + b) = 5 / 3) → 
  (a = 5 ∧ b = 1) ∨ (a = 1 ∧ b = 5) := 
sorry

end find_numbers_l78_78012


namespace cos_value_given_sin_l78_78902

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (π / 3 - α) = 3 / 5 :=
sorry

end cos_value_given_sin_l78_78902


namespace sum_first_five_terms_l78_78190

-- Define the geometric sequence
noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n
  else a1 * (1 - q^(n + 1)) / (1 - q)

-- Given conditions
def a1 : ℝ := 1
def q : ℝ := 2
def n : ℕ := 5

-- The theorem to be proven
theorem sum_first_five_terms : sum_geometric_sequence a1 q (n-1) = 31 := by
  sorry

end sum_first_five_terms_l78_78190


namespace calculate_value_of_expression_l78_78982

theorem calculate_value_of_expression :
  (2523 - 2428)^2 / 121 = 75 :=
by
  -- calculation steps here
  sorry

end calculate_value_of_expression_l78_78982


namespace replace_asterisks_l78_78314

theorem replace_asterisks (x : ℝ) (h : (x / 20) * (x / 80) = 1) : x = 40 :=
sorry

end replace_asterisks_l78_78314


namespace cannot_be_n_plus_2_l78_78762

theorem cannot_be_n_plus_2 (n : ℕ) : 
  ¬(∃ Y, (Y = n + 2) ∧ 
         ((Y = n - 3) ∨ (Y = n - 1) ∨ (Y = n + 5))) := 
by {
  sorry
}

end cannot_be_n_plus_2_l78_78762


namespace ratio_of_intercepts_l78_78253

variable (b1 b2 : ℝ)
variable (s t : ℝ)
variable (Hs : s = -b1 / 8)
variable (Ht : t = -b2 / 3)

theorem ratio_of_intercepts (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0) : s / t = 3 * b1 / (8 * b2) :=
by
  sorry

end ratio_of_intercepts_l78_78253


namespace range_of_x_for_a_range_of_a_l78_78685

-- Define propositions p and q
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (I)
theorem range_of_x_for_a (a x : ℝ) (ha : a = 1) (hpq : prop_p a x ∧ prop_q x) : 2 < x ∧ x < 3 :=
by
  sorry

-- Part (II)
theorem range_of_a (p q : ℝ → Prop) (hpq : ∀ x : ℝ, ¬p x → ¬q x) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_for_a_range_of_a_l78_78685


namespace probability_of_two_red_two_green_l78_78515

def red_balls : ℕ := 10
def green_balls : ℕ := 8
def total_balls : ℕ := red_balls + green_balls
def drawn_balls : ℕ := 4

def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_two_red_two_green : ℚ :=
  (combination red_balls 2 * combination green_balls 2 : ℚ) / combination total_balls drawn_balls

theorem probability_of_two_red_two_green :
  prob_two_red_two_green = 7 / 17 := 
sorry

end probability_of_two_red_two_green_l78_78515


namespace solutions_periodic_with_same_period_l78_78219

variable {y z : ℝ → ℝ}
variable (f g : ℝ → ℝ)

-- defining the conditions
variable (h1 : ∀ x, deriv y x = - (z x)^3)
variable (h2 : ∀ x, deriv z x = (y x)^3)
variable (h3 : y 0 = 1)
variable (h4 : z 0 = 0)
variable (h5 : ∀ x, y x = f x)
variable (h6 : ∀ x, z x = g x)

-- proving periodicity
theorem solutions_periodic_with_same_period : ∃ k > 0, (∀ x, f (x + k) = f x ∧ g (x + k) = g x) := by
  sorry

end solutions_periodic_with_same_period_l78_78219


namespace problem1_solution_l78_78818

theorem problem1_solution : ∀ x : ℝ, x^2 - 6 * x + 9 = (5 - 2 * x)^2 → (x = 8/3 ∨ x = 2) :=
sorry

end problem1_solution_l78_78818


namespace orthogonal_planes_k_value_l78_78895

theorem orthogonal_planes_k_value
  (k : ℝ)
  (h : 3 * (-1) + 1 * 1 + (-2) * k = 0) : 
  k = -1 :=
sorry

end orthogonal_planes_k_value_l78_78895


namespace distance_to_grandmas_house_is_78_l78_78486

-- Define the conditions
def miles_to_pie_shop : ℕ := 35
def miles_to_gas_station : ℕ := 18
def miles_remaining : ℕ := 25

-- Define the mathematical claim
def total_distance_to_grandmas_house : ℕ :=
  miles_to_pie_shop + miles_to_gas_station + miles_remaining

-- Prove the claim
theorem distance_to_grandmas_house_is_78 :
  total_distance_to_grandmas_house = 78 :=
by
  sorry

end distance_to_grandmas_house_is_78_l78_78486


namespace alcohol_percentage_in_mixed_solution_l78_78254

theorem alcohol_percentage_in_mixed_solution :
  let vol1 := 8
  let perc1 := 0.25
  let vol2 := 2
  let perc2 := 0.12
  let total_alcohol := (vol1 * perc1) + (vol2 * perc2)
  let total_volume := vol1 + vol2
  (total_alcohol / total_volume) * 100 = 22.4 := by
  sorry

end alcohol_percentage_in_mixed_solution_l78_78254


namespace percentage_failed_hindi_l78_78642

theorem percentage_failed_hindi 
  (F_E F_B P_BE : ℕ) 
  (h₁ : F_E = 42) 
  (h₂ : F_B = 28) 
  (h₃ : P_BE = 56) :
  ∃ F_H, F_H = 30 := 
by
  sorry

end percentage_failed_hindi_l78_78642


namespace smallest_n_fact_expr_l78_78108

theorem smallest_n_fact_expr : ∃ n : ℕ, (∀ m : ℕ, m = 6 → n! = (n - 4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1)) ∧ n = 23 := by
  sorry

end smallest_n_fact_expr_l78_78108


namespace estimate_y_value_l78_78193

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value_l78_78193


namespace house_trailer_payment_difference_l78_78962

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end house_trailer_payment_difference_l78_78962


namespace right_triangle_acute_angles_l78_78638

theorem right_triangle_acute_angles (a b : ℝ)
  (h_right_triangle : a + b = 90)
  (h_ratio : a / b = 3 / 2) :
  (a = 54) ∧ (b = 36) :=
by
  sorry

end right_triangle_acute_angles_l78_78638


namespace quadratic_inequality_solution_set_l78_78958

-- Define the necessary variables and conditions
variable (a b c α β : ℝ)
variable (h1 : 0 < α)
variable (h2 : α < β)
variable (h3 : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (α < x ∧ x < β))

-- Statement to be proved
theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, ((a + c - b) * x^2 + (b - 2 * a) * x + a > 0) ↔ ((1 / (1 + β) < x) ∧ (x < 1 / (1 + α))) :=
sorry

end quadratic_inequality_solution_set_l78_78958


namespace power_multiplication_l78_78303

theorem power_multiplication (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := 
by 
  sorry

end power_multiplication_l78_78303


namespace time_for_new_circle_l78_78833

theorem time_for_new_circle 
  (rounds : ℕ) (time : ℕ) (k : ℕ) (original_time_per_round new_time_per_round : ℝ) 
  (h1 : rounds = 8) 
  (h2 : time = 40) 
  (h3 : k = 10) 
  (h4 : original_time_per_round = time / rounds)
  (h5 : new_time_per_round = original_time_per_round * k) :
  new_time_per_round = 50 :=
by {
  sorry
}

end time_for_new_circle_l78_78833


namespace clock_angle_34030_l78_78504

noncomputable def calculate_angle (h m s : ℕ) : ℚ :=
  abs ((60 * h - 11 * (m + s / 60)) / 2)

theorem clock_angle_34030 : calculate_angle 3 40 30 = 130 :=
by
  sorry

end clock_angle_34030_l78_78504


namespace beth_should_charge_42_cents_each_l78_78227

theorem beth_should_charge_42_cents_each (n_alan_cookies : ℕ) (price_alan_cookie : ℕ) (n_beth_cookies : ℕ) (total_earnings : ℕ) (price_beth_cookie : ℕ):
  n_alan_cookies = 15 → 
  price_alan_cookie = 50 → 
  n_beth_cookies = 18 → 
  total_earnings = n_alan_cookies * price_alan_cookie → 
  price_beth_cookie = total_earnings / n_beth_cookies → 
  price_beth_cookie = 42 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beth_should_charge_42_cents_each_l78_78227


namespace planes_parallel_if_line_perpendicular_to_both_l78_78066

variables {Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Assume we have a function parallel that checks if a line is parallel to a plane
-- and a function perpendicular that checks if a line is perpendicular to a plane. 
-- Also, we assume a function parallel_planes that checks if two planes are parallel.
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

theorem planes_parallel_if_line_perpendicular_to_both
  (h1 : perpendicular l α) (h2 : perpendicular l β) : parallel_planes α β :=
sorry

end planes_parallel_if_line_perpendicular_to_both_l78_78066


namespace sculpture_paint_area_l78_78394

/-- An artist creates a sculpture using 15 cubes, each with a side length of 1 meter. 
The cubes are organized into a wall-like structure with three layers: 
the top layer consists of 3 cubes, 
the middle layer consists of 5 cubes, 
and the bottom layer consists of 7 cubes. 
Some of the cubes in the middle and bottom layers are spaced apart, exposing additional side faces. 
Prove that the total exposed surface area painted is 49 square meters. -/
theorem sculpture_paint_area :
  let cubes_sizes : ℕ := 15
  let layer_top : ℕ := 3
  let layer_middle : ℕ := 5
  let layer_bottom : ℕ := 7
  let side_exposed_area_layer_top : ℕ := layer_top * 5
  let side_exposed_area_layer_middle : ℕ := 2 * 3 + 3 * 2
  let side_exposed_area_layer_bottom : ℕ := layer_bottom * 1
  let exposed_side_faces : ℕ := side_exposed_area_layer_top + side_exposed_area_layer_middle + side_exposed_area_layer_bottom
  let exposed_top_faces : ℕ := layer_top * 1 + layer_middle * 1 + layer_bottom * 1
  let total_exposed_area : ℕ := exposed_side_faces + exposed_top_faces
  total_exposed_area = 49 := 
sorry

end sculpture_paint_area_l78_78394


namespace polynomial_simplification_l78_78921

variable (x : ℝ)

theorem polynomial_simplification :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 :=
by
  sorry

end polynomial_simplification_l78_78921


namespace water_wheel_effective_horsepower_l78_78379

noncomputable def effective_horsepower 
  (velocity : ℝ) (width : ℝ) (thickness : ℝ) (density : ℝ) 
  (diameter : ℝ) (efficiency : ℝ) (g : ℝ) (hp_conversion : ℝ) : ℝ :=
  let mass_flow_rate := velocity * width * thickness * density
  let kinetic_energy_per_second := 0.5 * mass_flow_rate * velocity^2
  let potential_energy_per_second := mass_flow_rate * diameter * g
  let indicated_power := kinetic_energy_per_second + potential_energy_per_second
  let horsepower := indicated_power / hp_conversion
  efficiency * horsepower

theorem water_wheel_effective_horsepower :
  effective_horsepower 1.4 0.5 0.13 1000 3 0.78 9.81 745.7 = 2.9 :=
by
  sorry

end water_wheel_effective_horsepower_l78_78379


namespace probability_A_wins_probability_A_wins_2_l78_78733

def binomial (n k : ℕ) := Nat.choose n k

noncomputable def P (n : ℕ) : ℚ := 
  1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n))

theorem probability_A_wins (n : ℕ) : P n = 1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n)) := 
by sorry

theorem probability_A_wins_2 : P 2 = 5 / 16 := 
by sorry

end probability_A_wins_probability_A_wins_2_l78_78733


namespace total_dolls_l78_78402

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l78_78402


namespace necessary_but_not_sufficient_l78_78436

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by sorry

end necessary_but_not_sufficient_l78_78436


namespace milk_needed_for_cookies_l78_78281

-- Definition of the problem conditions
def cookies_per_milk_usage := 24
def milk_in_liters := 5
def liters_to_milliliters := 1000
def milk_for_6_cookies := 1250

-- Prove that 1250 milliliters of milk are needed to bake 6 cookies
theorem milk_needed_for_cookies
  (h1 : cookies_per_milk_usage = 24)
  (h2 : milk_in_liters = 5)
  (h3 : liters_to_milliliters = 1000) :
  milk_for_6_cookies = 1250 :=
by
  -- Proof is omitted with sorry
  sorry

end milk_needed_for_cookies_l78_78281


namespace relationship_between_x_y_l78_78644

theorem relationship_between_x_y (x y : ℝ) (h1 : x^2 - y^2 > 2 * x) (h2 : x * y < y) : x < y ∧ y < 0 := 
sorry

end relationship_between_x_y_l78_78644


namespace initial_bananas_per_child_l78_78911

theorem initial_bananas_per_child 
    (absent : ℕ) (present : ℕ) (total : ℕ) (x : ℕ) (B : ℕ)
    (h1 : absent = 305)
    (h2 : present = 305)
    (h3 : total = 610)
    (h4 : B = present * (x + 2))
    (h5 : B = total * x) : 
    x = 2 :=
by
  sorry

end initial_bananas_per_child_l78_78911


namespace mary_daily_tasks_l78_78589

theorem mary_daily_tasks :
  ∃ (x y : ℕ), (x + y = 15) ∧ (4 * x + 7 * y = 85) ∧ (y = 8) :=
by
  sorry

end mary_daily_tasks_l78_78589


namespace derek_age_l78_78556

theorem derek_age (aunt_beatrice_age : ℕ) (emily_age : ℕ) (derek_age : ℕ)
  (h1 : aunt_beatrice_age = 54)
  (h2 : emily_age = aunt_beatrice_age / 2)
  (h3 : derek_age = emily_age - 7) : derek_age = 20 :=
by
  sorry

end derek_age_l78_78556


namespace union_M_N_eq_interval_l78_78591

variable {α : Type*} [PartialOrder α]

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem union_M_N_eq_interval :
  M ∪ N = {x | -1/2 < x ∧ x ≤ 1} :=
by
  sorry

end union_M_N_eq_interval_l78_78591


namespace find_a6_l78_78922

theorem find_a6 (a : ℕ → ℚ) (h₁ : ∀ n, a (n + 1) = 2 * a n - 1) (h₂ : a 8 = 16) : a 6 = 19 / 4 :=
sorry

end find_a6_l78_78922


namespace omega_not_real_root_l78_78963

theorem omega_not_real_root {ω : ℂ} (h1 : ω^3 = 1) (h2 : ω ≠ 1) (h3 : ω^2 + ω + 1 = 0) :
  (2 + 3 * ω - ω^2)^3 + (2 - 3 * ω + ω^2)^3 = -68 + 96 * ω :=
by sorry

end omega_not_real_root_l78_78963


namespace contractor_initial_hire_l78_78196

theorem contractor_initial_hire :
  ∃ (P : ℕ), 
    (∀ (total_work : ℝ), 
      (P * 20 = (1/4) * total_work) ∧ 
      ((P - 2) * 75 = (3/4) * total_work)) → 
    P = 10 :=
by
  sorry

end contractor_initial_hire_l78_78196


namespace functional_equation_solution_l78_78470

-- The mathematical problem statement in Lean 4

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_monotonic : ∀ x y : ℝ, (f x) * (f y) = f (x + y))
  (h_mono : ∀ x y : ℝ, x < y → f x < f y ∨ f x > f y) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end functional_equation_solution_l78_78470


namespace area_under_parabola_l78_78597

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- State the theorem about the area under the curve
theorem area_under_parabola : (∫ x in (1 : ℝ)..3, parabola x) = 4 / 3 :=
by
  -- Proof goes here
  sorry

end area_under_parabola_l78_78597


namespace smallest_possible_perimeter_l78_78212

-- Definitions for prime numbers and scalene triangles
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions
def valid_sides (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ is_scalene_triangle a b c

def valid_perimeter (a b c : ℕ) : Prop :=
  is_prime (a + b + c)

-- The goal statement
theorem smallest_possible_perimeter : ∃ a b c : ℕ, valid_sides a b c ∧ valid_perimeter a b c ∧ (a + b + c) = 23 :=
by
  sorry

end smallest_possible_perimeter_l78_78212


namespace find_a_l78_78559

theorem find_a {a : ℝ} :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 → (x < -1 ∨ x > -1 / 2)) → a = -2 :=
by 
  intros h
  sorry

end find_a_l78_78559


namespace find_other_root_l78_78822

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end find_other_root_l78_78822


namespace bill_sun_vs_sat_l78_78123

theorem bill_sun_vs_sat (B_Sat B_Sun J_Sun : ℕ) 
  (h1 : B_Sun = 6)
  (h2 : J_Sun = 2 * B_Sun)
  (h3 : B_Sat + B_Sun + J_Sun = 20) : 
  B_Sun - B_Sat = 4 :=
by
  sorry

end bill_sun_vs_sat_l78_78123


namespace sum_of_fractions_and_decimal_l78_78145

theorem sum_of_fractions_and_decimal :
  (6 / 5 : ℝ) + (1 / 10 : ℝ) + 1.56 = 2.86 :=
by
  sorry

end sum_of_fractions_and_decimal_l78_78145


namespace john_guests_count_l78_78851

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def additional_fractional_guests : ℝ := 0.60
def total_cost_when_wife_gets_her_way : ℕ := 50000

theorem john_guests_count (G : ℕ) :
  venue_cost + cost_per_guest * (1 + additional_fractional_guests) * G = 
  total_cost_when_wife_gets_her_way →
  G = 50 :=
by
  sorry

end john_guests_count_l78_78851


namespace pagoda_top_story_lanterns_l78_78458

/--
Given a 7-story pagoda where each story has twice as many lanterns as the one above it, 
and a total of 381 lanterns across all stories, prove the number of lanterns on the top (7th) story is 3.
-/
theorem pagoda_top_story_lanterns (a : ℕ) (n : ℕ) (r : ℚ) (sum_lanterns : ℕ) :
  n = 7 → r = 1 / 2 → sum_lanterns = 381 →
  (a * (1 - r^n) / (1 - r) = sum_lanterns) → (a * r^(n - 1) = 3) :=
by
  intros h_n h_r h_sum h_geo_sum
  let a_val := 192 -- from the solution steps
  rw [h_n, h_r, h_sum] at h_geo_sum
  have h_a : a = a_val := by sorry
  rw [h_a, h_n, h_r]
  exact sorry

end pagoda_top_story_lanterns_l78_78458


namespace domain_of_v_l78_78374

def domain_v (x : ℝ) : Prop :=
  x ≥ 2 ∧ x ≠ 5

theorem domain_of_v :
  {x : ℝ | domain_v x} = { x | 2 < x ∧ x < 5 } ∪ { x | 5 < x }
:= by
  sorry

end domain_of_v_l78_78374


namespace consecutive_numbers_sum_39_l78_78832

theorem consecutive_numbers_sum_39 (n : ℕ) (hn : n + (n + 1) = 39) : n + 1 = 20 :=
sorry

end consecutive_numbers_sum_39_l78_78832


namespace tan_sub_pi_over_4_l78_78654

-- Define the conditions and the problem statement
variable (α : ℝ) (h : Real.tan α = 2)

-- State the problem as a theorem
theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_4_l78_78654


namespace sum_of_c_n_l78_78555

variable {a_n : ℕ → ℕ}    -- Sequence {a_n}
variable {b_n : ℕ → ℕ}    -- Sequence {b_n}
variable {c_n : ℕ → ℕ}    -- Sequence {c_n}
variable {S_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {a_n}
variable {T_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {c_n}

axiom a3 : a_n 3 = 7
axiom S6 : S_n 6 = 48
axiom b_recur : ∀ n : ℕ, 2 * b_n (n + 1) = b_n n + 2
axiom b1 : b_n 1 = 3
axiom c_def : ∀ n : ℕ, c_n n = a_n n * (b_n n - 2)

theorem sum_of_c_n : ∀ n : ℕ, T_n n = 10 - (2*n + 5) * (1 / (2^(n-1))) :=
by
  -- Proof omitted
  sorry

end sum_of_c_n_l78_78555


namespace adjusted_retail_price_l78_78643

variable {a : ℝ} {m n : ℝ}

theorem adjusted_retail_price (h : 0 ≤ m ∧ 0 ≤ n) : (a * (1 + m / 100) * (n / 100)) = a * (1 + m / 100) * (n / 100) :=
by
  sorry

end adjusted_retail_price_l78_78643


namespace perfect_square_for_n_l78_78763

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l78_78763


namespace sum_powers_eq_34_over_3_l78_78620

theorem sum_powers_eq_34_over_3 (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6):
  a^4 + b^4 + c^4 = 34 / 3 :=
by
  sorry

end sum_powers_eq_34_over_3_l78_78620


namespace total_exercise_time_l78_78910

-- Definitions based on given conditions
def javier_daily : ℕ := 50
def javier_days : ℕ := 7
def sanda_daily : ℕ := 90
def sanda_days : ℕ := 3

-- Proof problem to verify the total exercise time for both Javier and Sanda
theorem total_exercise_time : javier_daily * javier_days + sanda_daily * sanda_days = 620 := by
  sorry

end total_exercise_time_l78_78910


namespace bells_ring_together_l78_78952

theorem bells_ring_together (church school day_care library noon : ℕ) :
  church = 18 ∧ school = 24 ∧ day_care = 30 ∧ library = 35 ∧ noon = 0 →
  ∃ t : ℕ, t = 2520 ∧ ∀ n, (t - noon) % n = 0 := by
  sorry

end bells_ring_together_l78_78952


namespace sufficient_condition_l78_78683

theorem sufficient_condition (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_condition_l78_78683


namespace tg_pi_over_12_eq_exists_two_nums_l78_78271

noncomputable def tg (x : ℝ) := Real.tan x

theorem tg_pi_over_12_eq : tg (Real.pi / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

theorem exists_two_nums (a : Fin 13 → ℝ) (h_diff : Function.Injective a) :
  ∃ x y, 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

end tg_pi_over_12_eq_exists_two_nums_l78_78271


namespace visual_range_percent_increase_l78_78697

-- Define the original and new visual ranges
def original_range : ℝ := 90
def new_range : ℝ := 150

-- Define the desired percent increase as a real number
def desired_percent_increase : ℝ := 66.67

-- The theorem to prove that the visual range is increased by the desired percentage
theorem visual_range_percent_increase :
  ((new_range - original_range) / original_range) * 100 = desired_percent_increase := 
sorry

end visual_range_percent_increase_l78_78697


namespace remainder_product_div_10_l78_78938

def unitsDigit (n : ℕ) : ℕ := n % 10

theorem remainder_product_div_10 :
  let a := 1734
  let b := 5389
  let c := 80607
  let p := a * b * c
  unitsDigit p = 2 := by
  sorry

end remainder_product_div_10_l78_78938


namespace puppies_left_l78_78088

namespace AlyssaPuppies

def initPuppies : ℕ := 12
def givenAway : ℕ := 7
def remainingPuppies : ℕ := 5

theorem puppies_left (initPuppies givenAway remainingPuppies : ℕ) : 
  initPuppies - givenAway = remainingPuppies :=
by
  sorry

end AlyssaPuppies

end puppies_left_l78_78088


namespace remainder_of_base12_integer_divided_by_9_l78_78213

-- Define the base-12 integer
def base12_integer := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3

-- Define the condition for our problem
def divisor := 9

-- State the theorem to be proved
theorem remainder_of_base12_integer_divided_by_9 :
  base12_integer % divisor = 0 :=
sorry

end remainder_of_base12_integer_divided_by_9_l78_78213


namespace freddy_age_l78_78406

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l78_78406


namespace cos_of_largest_angle_is_neg_half_l78_78304

-- Lean does not allow forward references to elements yet to be declared, 
-- hence we keep a strict order for declarations
namespace TriangleCosine

open Real

-- Define the side lengths of the triangle as constants
def a : ℝ := 3
def b : ℝ := 5
def c : ℝ := 7

-- Define the expression using cosine rule to find cos C
noncomputable def cos_largest_angle : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Declare the theorem statement
theorem cos_of_largest_angle_is_neg_half : cos_largest_angle = -1 / 2 := 
by 
  sorry

end TriangleCosine

end cos_of_largest_angle_is_neg_half_l78_78304


namespace total_earnings_correct_l78_78884

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l78_78884


namespace find_t_for_area_of_triangle_l78_78715

theorem find_t_for_area_of_triangle :
  ∃ (t : ℝ), 
  (∀ (A B C T U: ℝ × ℝ),
    A = (0, 10) → 
    B = (3, 0) → 
    C = (9, 0) → 
    T = (3/10 * (10 - t), t) →
    U = (9/10 * (10 - t), t) →
    2 * 15 = 3/10 * (10 - t) ^ 2) →
  t = 2.93 :=
by sorry

end find_t_for_area_of_triangle_l78_78715


namespace range_of_a_l78_78098

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a^2 ≤ 0 → false) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l78_78098


namespace triangle_area_of_tangent_line_l78_78472

theorem triangle_area_of_tangent_line (a : ℝ) 
  (h : a > 0) 
  (ha : (1/2) * 3 * a * (3 / (2 * a ^ (1/2))) = 18)
  : a = 64 := 
sorry

end triangle_area_of_tangent_line_l78_78472


namespace polynomial_remainder_l78_78869

theorem polynomial_remainder (x : ℝ) :
  (x^4 + 3 * x^2 - 4) % (x^2 + 2) = x^2 - 4 :=
sorry

end polynomial_remainder_l78_78869


namespace find_a_and_b_l78_78124

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x + 1))

theorem find_a_and_b
  (a b : ℝ)
  (h1 : a < b)
  (h2 : f a = f ((- (b + 1)) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = - 2 / 5 ∧ b = - 1 / 3 :=
sorry

end find_a_and_b_l78_78124


namespace sin_value_l78_78728

open Real

-- Define the given conditions
variables (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x) (h3 : x < 2 * π)

-- State the problem to be proved
theorem sin_value : sin x = - 4 / 5 :=
by
  sorry

end sin_value_l78_78728


namespace tyrone_gave_marbles_l78_78450

theorem tyrone_gave_marbles :
  ∃ x : ℝ, (120 - x = 3 * (30 + x)) ∧ x = 7.5 :=
by
  sorry

end tyrone_gave_marbles_l78_78450


namespace monitor_height_l78_78321

theorem monitor_height (width_in_inches : ℕ) (pixels_per_inch : ℕ) (total_pixels : ℕ) 
  (h1 : width_in_inches = 21) (h2 : pixels_per_inch = 100) (h3 : total_pixels = 2520000) : 
  total_pixels / (width_in_inches * pixels_per_inch) / pixels_per_inch = 12 :=
by
  sorry

end monitor_height_l78_78321


namespace skateboard_price_after_discounts_l78_78350

-- Defining all necessary conditions based on the given problem.
def original_price : ℝ := 150
def discount1 : ℝ := 0.40 * original_price
def price_after_discount1 : ℝ := original_price - discount1
def discount2 : ℝ := 0.25 * price_after_discount1
def final_price : ℝ := price_after_discount1 - discount2

-- Goal: Prove that the final price after both discounts is $67.50.
theorem skateboard_price_after_discounts : final_price = 67.50 := by
  sorry

end skateboard_price_after_discounts_l78_78350


namespace value_of_expression_l78_78166

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 :=
by
  -- proof goes here
  sorry

end value_of_expression_l78_78166


namespace line_equations_through_point_with_intercepts_l78_78324

theorem line_equations_through_point_with_intercepts (x y : ℝ) :
  (x = -10 ∧ y = 10) ∧ (∃ a : ℝ, 4 * a = intercept_x ∧ a = intercept_y) →
  (x + y = 0 ∨ x + 4 * y - 30 = 0) :=
by
  sorry

end line_equations_through_point_with_intercepts_l78_78324


namespace daniel_total_spent_l78_78046

/-
Daniel buys various items with given prices, receives a 10% coupon discount,
a store credit of $1.50, a 5% student discount, and faces a 6.5% sales tax.
Prove that the total amount he spends is $8.23.
-/
def total_spent (prices : List ℝ) (coupon_discount store_credit student_discount sales_tax : ℝ) : ℝ :=
  let initial_total := prices.sum
  let after_coupon := initial_total * (1 - coupon_discount)
  let after_student := after_coupon * (1 - student_discount)
  let after_store_credit := after_student - store_credit
  let final_total := after_store_credit * (1 + sales_tax)
  final_total

theorem daniel_total_spent :
  total_spent 
    [0.85, 0.50, 1.25, 3.75, 2.99, 1.45] -- prices of items
    0.10 -- 10% coupon discount
    1.50 -- $1.50 store credit
    0.05 -- 5% student discount
    0.065 -- 6.5% sales tax
  = 8.23 :=
by
  sorry

end daniel_total_spent_l78_78046


namespace geometric_sequence_a1_range_l78_78252

theorem geometric_sequence_a1_range (a : ℕ → ℝ) (b : ℕ → ℝ) (a1 : ℝ) :
  (∀ n, a (n+1) = a n / 2) ∧ (∀ n, b n = n / 2) ∧ (∃! n : ℕ, a n > b n) →
  (6 < a1 ∧ a1 ≤ 16) :=
by
  sorry

end geometric_sequence_a1_range_l78_78252


namespace jillian_apartment_size_l78_78924

theorem jillian_apartment_size :
  ∃ (s : ℝ), (1.20 * s = 720) ∧ s = 600 := by
sorry

end jillian_apartment_size_l78_78924


namespace ratio_of_inscribed_to_circumscribed_l78_78361

theorem ratio_of_inscribed_to_circumscribed (a : ℝ) :
  let r' := a * Real.sqrt 6 / 12
  let R' := a * Real.sqrt 6 / 4
  r' / R' = 1 / 3 := by
  sorry

end ratio_of_inscribed_to_circumscribed_l78_78361


namespace triangle_problem_l78_78564

/--
Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C, respectively, 
where:
1. b * (sin B - sin C) = a * sin A - c * sin C
2. a = 2 * sqrt 3
3. the area of triangle ABC is 2 * sqrt 3

Prove:
1. A = π / 3
2. The perimeter of triangle ABC is 2 * sqrt 3 + 6
-/
theorem triangle_problem 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := 
sorry

end triangle_problem_l78_78564


namespace pyarelal_loss_l78_78913

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (h1 : total_loss = 670) (h2 : 1 / 9 * P + P = 10 / 9 * P):
  (9 / (1 + 9)) * total_loss = 603 :=
by
  sorry

end pyarelal_loss_l78_78913


namespace abc_divisibility_l78_78005

theorem abc_divisibility (a b c : ℕ) (h₁ : a ∣ (b * c - 1)) (h₂ : b ∣ (c * a - 1)) (h₃ : c ∣ (a * b - 1)) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 1 ∧ b = 1 ∧ ∃ n : ℕ, n ≥ 1 ∧ c = n) :=
by
  sorry

end abc_divisibility_l78_78005


namespace cube_volume_is_64_l78_78199

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end cube_volume_is_64_l78_78199


namespace curve_is_parabola_l78_78201

theorem curve_is_parabola (t : ℝ) : 
  ∃ (x y : ℝ), (x = 3^t - 2) ∧ (y = 9^t - 4 * 3^t + 2 * t - 4) ∧ (∃ a b c : ℝ, y = a * x^2 + b * x + c) :=
by sorry

end curve_is_parabola_l78_78201


namespace hike_length_l78_78542

-- Definitions of conditions
def initial_water : ℕ := 6
def final_water : ℕ := 1
def hike_duration : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_drunk : ℕ := 1
def first_part_drink_rate : ℚ := 2 / 3

-- Statement to prove
theorem hike_length (hike_duration : ℕ) (initial_water : ℕ) (final_water : ℕ) (leak_rate : ℕ) 
  (last_mile_drunk : ℕ) (first_part_drink_rate : ℚ) : 
  hike_duration = 2 → 
  initial_water = 6 → 
  final_water = 1 → 
  leak_rate = 1 → 
  last_mile_drunk = 1 → 
  first_part_drink_rate = 2 / 3 → 
  ∃ miles : ℕ, miles = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof placeholder
  sorry

end hike_length_l78_78542


namespace find_c_l78_78320

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 6) (h3 : ((6 - c) / c) = 4 / 9) : c = 54 / 13 :=
sorry

end find_c_l78_78320


namespace not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l78_78371

def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ * x₁ + b * x₁ + c = 0 ∧ a * x₂ * x₂ + b * x₂ + c = 0 
  ∧ (x₁ - x₂ = 1 ∨ x₂ - x₁ = 1)

theorem not_neighboring_root_equation_x2_x_2 : 
  ¬ is_neighboring_root_equation 1 1 (-2) :=
sorry

theorem neighboring_root_equation_k_values (k : ℝ) : 
  is_neighboring_root_equation 1 (-(k-3)) (-3*k) ↔ k = -2 ∨ k = -4 :=
sorry

end not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l78_78371


namespace find_pink_highlighters_l78_78693

def yellow_highlighters : ℕ := 7
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 15

theorem find_pink_highlighters : (total_highlighters - (yellow_highlighters + blue_highlighters)) = 3 :=
by
  sorry

end find_pink_highlighters_l78_78693


namespace order_of_magnitude_l78_78929

theorem order_of_magnitude (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end order_of_magnitude_l78_78929


namespace perpendicular_planes_implies_perpendicular_line_l78_78809

-- Definitions of lines and planes and their properties in space
variable {Space : Type}
variable (m n l : Line Space) -- Lines in space
variable (α β γ : Plane Space) -- Planes in space

-- Conditions: m, n, and l are non-intersecting lines, α, β, and γ are non-intersecting planes
axiom non_intersecting_lines : ¬ (m = n) ∧ ¬ (m = l) ∧ ¬ (n = l)
axiom non_intersecting_planes : ¬ (α = β) ∧ ¬ (α = γ) ∧ ¬ (β = γ)

-- To prove: if α ⊥ γ, β ⊥ γ, and α ∩ β = l, then l ⊥ γ
theorem perpendicular_planes_implies_perpendicular_line
  (h1 : α ⊥ γ) 
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = l) : l ⊥ γ := 
  sorry

end perpendicular_planes_implies_perpendicular_line_l78_78809


namespace diameter_of_triple_sphere_l78_78747

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end diameter_of_triple_sphere_l78_78747


namespace find_length_of_AC_l78_78405

theorem find_length_of_AC
  (A B C : Type)
  (AB : Real)
  (AC : Real)
  (Area : Real)
  (angle_A : Real)
  (h1 : AB = 8)
  (h2 : angle_A = (30 * Real.pi / 180)) -- converting degrees to radians
  (h3 : Area = 16) :
  AC = 8 :=
by
  -- Skipping proof as requested
  sorry

end find_length_of_AC_l78_78405


namespace problem1_problem2_l78_78724

theorem problem1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2 * y :=
by 
  sorry

theorem problem2 (m : ℝ) (h : m ≠ 2) : (1 - m / (m + 2)) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 2 / (m - 2) :=
by 
  sorry

end problem1_problem2_l78_78724


namespace original_two_digit_number_is_52_l78_78970

theorem original_two_digit_number_is_52 (x : ℕ) (h1 : 10 * x + 6 = x + 474) (h2 : 10 ≤ x ∧ x < 100) : x = 52 :=
sorry

end original_two_digit_number_is_52_l78_78970


namespace inequality_proof_l78_78396

theorem inequality_proof (x : ℝ) (hx : 0 < x) : (1 / x) + 4 * (x ^ 2) ≥ 3 :=
by
  sorry

end inequality_proof_l78_78396


namespace half_of_number_l78_78151

theorem half_of_number (N : ℝ)
  (h1 : (4 / 15) * (5 / 7) * N = (4 / 9) * (2 / 5) * N + 8) : 
  (N / 2) = 315 := 
sorry

end half_of_number_l78_78151


namespace percentage_in_excess_l78_78647

theorem percentage_in_excess 
  (A B : ℝ) (x : ℝ)
  (h1 : ∀ A',  A' = A * (1 + x / 100))
  (h2 : ∀ B',  B' = 0.94 * B)
  (h3 : ∀ A' B', A' * B' = A * B * (1 + 0.0058)) :
  x = 7 :=
by
  sorry

end percentage_in_excess_l78_78647


namespace four_digit_integer_product_l78_78435

theorem four_digit_integer_product :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
  a^2 + b^2 + c^2 + d^2 = 65 ∧ a * b * c * d = 140 :=
by
  sorry

end four_digit_integer_product_l78_78435


namespace intersection_point_of_lines_l78_78689

theorem intersection_point_of_lines :
  let line1 (x : ℝ) := 3 * x - 4
  let line2 (x : ℝ) := - (1 / 3) * x + 5
  (∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ x = 2.7 ∧ y = 4.1) :=
by {
    sorry
}

end intersection_point_of_lines_l78_78689


namespace ashton_remaining_items_l78_78774

variables (pencil_boxes : ℕ) (pens_boxes : ℕ) (pencils_per_box : ℕ) (pens_per_box : ℕ)
          (given_pencils_brother : ℕ) (distributed_pencils_friends : ℕ)
          (distributed_pens_friends : ℕ)

def total_initial_pencils := 3 * 14
def total_initial_pens := 2 * 10

def remaining_pencils := total_initial_pencils - 6 - 12
def remaining_pens := total_initial_pens - 8
def remaining_items := remaining_pencils + remaining_pens

theorem ashton_remaining_items : remaining_items = 36 :=
sorry

end ashton_remaining_items_l78_78774


namespace parabola_condition_l78_78757

/-- Given the point (3,0) lies on the parabola y = 2x^2 + (k + 2)x - k,
    prove that k = -12. -/
theorem parabola_condition (k : ℝ) (h : 0 = 2 * 3^2 + (k + 2) * 3 - k) : k = -12 :=
by 
  sorry

end parabola_condition_l78_78757


namespace simplify_fraction_l78_78619

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : (2 + 4 * i) / (1 - 5 * i) = (-9 / 13) + (7 / 13) * i :=
by sorry

end simplify_fraction_l78_78619


namespace plane_distance_l78_78401

variable (a b c p : ℝ)

def plane_intercept := (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (p = 1 / (Real.sqrt ((1 / a^2) + (1 / b^2) + (1 / c^2))))

theorem plane_distance
  (h : plane_intercept a b c p) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 1 / p^2 := 
sorry

end plane_distance_l78_78401


namespace transform_expression_l78_78234

theorem transform_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) : 
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := 
by 
  sorry

end transform_expression_l78_78234


namespace value_of_k_l78_78358

theorem value_of_k (k : ℝ) : (2 - k * 2 = -4 * (-1)) → k = -1 :=
by
  intro h
  sorry

end value_of_k_l78_78358


namespace inequality_solution_set_l78_78987

theorem inequality_solution_set (x : ℝ) : ((x - 1) * (x^2 - x + 1) > 0) ↔ (x > 1) :=
by
  sorry

end inequality_solution_set_l78_78987


namespace heat_capacity_at_100K_l78_78221

noncomputable def heat_capacity (t : ℝ) : ℝ :=
  0.1054 + 0.000004 * t

theorem heat_capacity_at_100K :
  heat_capacity 100 = 0.1058 := 
by
  sorry

end heat_capacity_at_100K_l78_78221


namespace function_relation4_l78_78191

open Set

section
  variable (M : Set ℤ) (N : Set ℤ)

  def relation1 (x : ℤ) := x ^ 2
  def relation2 (x : ℤ) := x + 1
  def relation3 (x : ℤ) := x - 1
  def relation4 (x : ℤ) := abs x

  theorem function_relation4 : 
    M = {-1, 1, 2, 4} →
    N = {1, 2, 4} →
    (∀ x ∈ M, relation4 x ∈ N) :=
  by
    intros hM hN
    simp [relation4]
    sorry
end

end function_relation4_l78_78191


namespace discount_percentage_l78_78667

variable (P : ℝ)  -- Original price of the car
variable (D : ℝ)  -- Discount percentage in decimal form
variable (S : ℝ)  -- Selling price of the car

theorem discount_percentage
  (h1 : S = P * (1 - D) * 1.70)
  (h2 : S = P * 1.1899999999999999) :
  D = 0.3 :=
by
  -- The proof goes here
  sorry

end discount_percentage_l78_78667


namespace repair_cost_total_l78_78734

def hourly_labor_cost : ℝ := 75
def labor_hours : ℝ := 16
def part_cost : ℝ := 1200
def labor_cost : ℝ := hourly_labor_cost * labor_hours
def total_cost : ℝ := labor_cost + part_cost

theorem repair_cost_total : total_cost = 2400 := 
by
  -- Proof omitted
  sorry

end repair_cost_total_l78_78734


namespace clock_hands_angle_seventy_degrees_l78_78103

theorem clock_hands_angle_seventy_degrees (t : ℝ) (h : t ≥ 0 ∧ t ≤ 60):
    let hour_angle := 210 + 30 * (t / 60)
    let minute_angle := 360 * (t / 60)
    let angle := abs (hour_angle - minute_angle)
    (angle = 70 ∨ angle = 290) ↔ (t = 25 ∨ t = 52) :=
by apply sorry

end clock_hands_angle_seventy_degrees_l78_78103


namespace sum_of_consecutive_page_numbers_l78_78225

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 283 := 
sorry

end sum_of_consecutive_page_numbers_l78_78225


namespace mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l78_78634

def molar_mass_carbon : Float := 12.01
def molar_mass_hydrogen : Float := 1.008
def moles_of_nonane : Float := 23.0
def num_carbons_in_nonane : Float := 9.0
def num_hydrogens_in_nonane : Float := 20.0

theorem mass_of_23_moles_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  mass_23_moles = 2950.75 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  have molar_mass_C9H20_val : molar_mass_C9H20 = 128.25 := sorry
  have mass_23_moles_val : mass_23_moles = 2950.75 := sorry
  exact mass_23_moles_val

theorem percentage_composition_C_H_O_in_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  percentage_carbon = 84.27 ∧ percentage_hydrogen = 15.73 ∧ percentage_oxygen = 0 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  have percentage_carbon_val : percentage_carbon = 84.27 := sorry
  have percentage_hydrogen_val : percentage_hydrogen = 15.73 := sorry
  have percentage_oxygen_val : percentage_oxygen = 0 := by rfl
  exact ⟨percentage_carbon_val, percentage_hydrogen_val, percentage_oxygen_val⟩

end mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l78_78634


namespace value_of_a_l78_78203

theorem value_of_a (a x y : ℤ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - 3 * y = 1) : a = 2 := by
  sorry

end value_of_a_l78_78203


namespace fraction_replaced_l78_78857

theorem fraction_replaced :
  ∃ x : ℚ, (0.60 * (1 - x) + 0.25 * x = 0.35) ∧ x = 5 / 7 := by
    sorry

end fraction_replaced_l78_78857


namespace candidate_total_score_l78_78269

theorem candidate_total_score (written_score : ℝ) (interview_score : ℝ) (written_weight : ℝ) (interview_weight : ℝ) :
    written_score = 90 → interview_score = 80 → written_weight = 0.70 → interview_weight = 0.30 →
    written_score * written_weight + interview_score * interview_weight = 87 :=
by
  intros
  sorry

end candidate_total_score_l78_78269


namespace average_a_b_l78_78031

theorem average_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27) : (A + B) / 2 = 40 := 
by
  sorry

end average_a_b_l78_78031


namespace arnold_plates_count_l78_78714

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end arnold_plates_count_l78_78714


namespace find_red_peaches_l78_78663

def num_red_peaches (red yellow green : ℕ) : Prop :=
  (green = red + 1) ∧ yellow = 71 ∧ green = 8

theorem find_red_peaches (red : ℕ) :
  num_red_peaches red 71 8 → red = 7 :=
by
  sorry

end find_red_peaches_l78_78663


namespace initial_guppies_l78_78446

theorem initial_guppies (total_gups : ℕ) (dozen_gups : ℕ) (extra_gups : ℕ) (baby_gups_initial : ℕ) (baby_gups_later : ℕ) :
  total_gups = 52 → dozen_gups = 12 → extra_gups = 3 → baby_gups_initial = 3 * 12 → baby_gups_later = 9 → 
  total_gups - (baby_gups_initial + baby_gups_later) = 7 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end initial_guppies_l78_78446


namespace conversion_base8_to_base10_l78_78463

theorem conversion_base8_to_base10 : 5 * 8^3 + 2 * 8^2 + 1 * 8^1 + 4 * 8^0 = 2700 :=
by 
  sorry

end conversion_base8_to_base10_l78_78463


namespace smallest_b_value_l78_78907

theorem smallest_b_value (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a - b = 7) 
    (h₄ : (Nat.gcd ((a^3 + b^3) / (a + b)) (a^2 * b)) = 12) : b = 6 :=
by
    -- proof goes here
    sorry

end smallest_b_value_l78_78907


namespace minimum_and_maximum_S_l78_78197

theorem minimum_and_maximum_S (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) :
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * a^2 - 3 * b^2 - 3 * c^2 - 3 * d^2 = 7.5 :=
sorry

end minimum_and_maximum_S_l78_78197


namespace required_speed_l78_78047

noncomputable def distance_travelled_late (d: ℝ) (t: ℝ) : ℝ :=
  50 * (t + 1/12)

noncomputable def distance_travelled_early (d: ℝ) (t: ℝ) : ℝ :=
  70 * (t - 1/12)

theorem required_speed :
  ∃ (s: ℝ), s = 58 ∧ 
  (∀ (d t: ℝ), distance_travelled_late d t = d ∧ distance_travelled_early d t = d → 
  d / t = s) :=
by
  sorry

end required_speed_l78_78047


namespace combined_value_l78_78359

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l78_78359


namespace smallest_n_l78_78713

theorem smallest_n (n : ℕ) (h : n ≥ 2) : 
  (∃ m : ℕ, m * m = (n + 1) * (2 * n + 1) / 6) ↔ n = 337 :=
by
  sorry

end smallest_n_l78_78713


namespace unknown_number_value_l78_78836

theorem unknown_number_value (x n : ℝ) (h1 : 0.75 / x = n / 8) (h2 : x = 2) : n = 3 :=
by
  sorry

end unknown_number_value_l78_78836


namespace no_rational_roots_l78_78328

theorem no_rational_roots (x : ℚ) : ¬(3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1 = 0) :=
by sorry

end no_rational_roots_l78_78328


namespace diff_of_squares_l78_78042

variable (a : ℝ)

theorem diff_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end diff_of_squares_l78_78042


namespace factors_and_divisors_l78_78709

theorem factors_and_divisors :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (¬(∃ n : ℕ, 209 = 19 * n ∧ ¬ (∃ m : ℕ, 57 = 19 * m))) ∧
  (¬(¬(∃ n : ℕ, 90 = 30 * n) ∧ ¬(∃ m : ℕ, 75 = 30 * m))) ∧
  (¬(∃ n : ℕ, 51 = 17 * n ∧ ¬ (∃ m : ℕ, 68 = 17 * m))) ∧
  (∃ n : ℕ, 171 = 9 * n) :=
by {
  sorry
}

end factors_and_divisors_l78_78709


namespace parabola_focus_equals_hyperbola_focus_l78_78648

noncomputable def hyperbola_right_focus : (Float × Float) := (2, 0)

noncomputable def parabola_focus (p : Float) : (Float × Float) := (p / 2, 0)

theorem parabola_focus_equals_hyperbola_focus (p : Float) :
  parabola_focus p = hyperbola_right_focus → p = 4 := by
  intro h
  sorry

end parabola_focus_equals_hyperbola_focus_l78_78648


namespace preimages_of_f_l78_78403

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem preimages_of_f (k : ℝ) : (∃ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ ≠ x₂) ↔ k < 1 := by
  sorry

end preimages_of_f_l78_78403


namespace mean_is_six_greater_than_median_l78_78660

theorem mean_is_six_greater_than_median (x a : ℕ) 
  (h1 : (x + a) + (x + 4) + (x + 7) + (x + 37) + x == 5 * (x + 10)) :
  a = 2 :=
by
  -- proof goes here
  sorry

end mean_is_six_greater_than_median_l78_78660


namespace Henry_age_l78_78640

-- Define the main proof statement
theorem Henry_age (h s : ℕ) 
(h1 : h + 8 = 3 * (s - 1))
(h2 : (h - 25) + (s - 25) = 83) : h = 97 :=
by
  sorry

end Henry_age_l78_78640


namespace medical_bills_value_l78_78742

variable (M : ℝ)
variable (property_damage : ℝ := 40000)
variable (insurance_coverage : ℝ := 0.80)
variable (carl_coverage : ℝ := 0.20)
variable (carl_owes : ℝ := 22000)

theorem medical_bills_value : 0.20 * (property_damage + M) = carl_owes → M = 70000 := 
by
  intro h
  sorry

end medical_bills_value_l78_78742


namespace width_of_carpet_is_1000_cm_l78_78796

noncomputable def width_of_carpet_in_cm (total_cost : ℝ) (cost_per_meter : ℝ) (length_of_room : ℝ) : ℝ :=
  let total_length_of_carpet := total_cost / cost_per_meter
  let width_of_carpet_in_meters := total_length_of_carpet / length_of_room
  width_of_carpet_in_meters * 100

theorem width_of_carpet_is_1000_cm :
  width_of_carpet_in_cm 810 4.50 18 = 1000 :=
by sorry

end width_of_carpet_is_1000_cm_l78_78796


namespace unique_a_for_three_distinct_real_solutions_l78_78731

theorem unique_a_for_three_distinct_real_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - 2 * x + 1 - 3 * |x|) ∧
  ((∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧
  (∀ x4 : ℝ, f x4 = 0 → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3) )) ) ↔
  a = 1 / 4 :=
sorry

end unique_a_for_three_distinct_real_solutions_l78_78731


namespace farmer_pays_per_acre_per_month_l78_78399

-- Define the conditions
def total_payment : ℕ := 600
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Define the problem to prove
theorem farmer_pays_per_acre_per_month :
  length_of_plot * width_of_plot / square_feet_per_acre > 0 ∧
  total_payment / (length_of_plot * width_of_plot / square_feet_per_acre) = 60 :=
by
  -- skipping the actual proof for now
  sorry

end farmer_pays_per_acre_per_month_l78_78399


namespace cost_of_whitewashing_l78_78606

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end cost_of_whitewashing_l78_78606


namespace donation_percentage_correct_l78_78973

noncomputable def percentage_donated_to_orphan_house (income remaining : ℝ) (given_to_children_percentage : ℝ) (given_to_wife_percentage : ℝ) (remaining_after_donation : ℝ)
    (before_donation_remaining : income * (1 - given_to_children_percentage / 100 - given_to_wife_percentage / 100) = remaining)
    (after_donation_remaining : remaining - remaining_after_donation * remaining = 500) : Prop :=
    100 * (remaining - 500) / remaining = 16.67

theorem donation_percentage_correct 
    (income : ℝ) 
    (child_percentage : ℝ := 10)
    (num_children : ℕ := 2)
    (wife_percentage : ℝ := 20)
    (final_amount : ℝ := 500)
    (income_value : income = 1000 ) : 
    percentage_donated_to_orphan_house income 
    (income * (1 - (child_percentage * num_children) / 100 - wife_percentage / 100)) 
    (child_percentage * num_children)
    wife_percentage 
    final_amount 
    sorry 
    sorry :=
sorry

end donation_percentage_correct_l78_78973


namespace max_proj_area_l78_78141

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem max_proj_area : 
  ∃ max_area : ℝ, max_area = Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end max_proj_area_l78_78141


namespace ratio_of_larger_to_smaller_l78_78003

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
a / b

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a * b = 50) (h3 : a > b) :
  ratio_of_numbers a b = 4 / 3 :=
sorry

end ratio_of_larger_to_smaller_l78_78003


namespace intersection_with_y_axis_l78_78588

theorem intersection_with_y_axis (x y : ℝ) : (x + y - 3 = 0 ∧ x = 0) → (x = 0 ∧ y = 3) :=
by {
  sorry
}

end intersection_with_y_axis_l78_78588


namespace inequality_flip_l78_78117

theorem inequality_flip (a b : ℤ) (c : ℤ) (h1 : a < b) (h2 : c < 0) : 
  c * a > c * b :=
sorry

end inequality_flip_l78_78117


namespace speed_boat_upstream_l78_78825

-- Define the conditions provided in the problem
def V_b : ℝ := 8.5  -- Speed of the boat in still water (in km/hr)
def V_downstream : ℝ := 13 -- Speed of the boat downstream (in km/hr)
def V_s : ℝ := V_downstream - V_b  -- Speed of the stream (in km/hr), derived from V_downstream and V_b
def V_upstream (V_b : ℝ) (V_s : ℝ) : ℝ := V_b - V_s  -- Speed of the boat upstream (in km/hr)

-- Statement to prove: the speed of the boat upstream is 4 km/hr
theorem speed_boat_upstream :
  V_upstream V_b V_s = 4 :=
by
  -- This line is for illustration, replace with an actual proof
  sorry

end speed_boat_upstream_l78_78825


namespace smallest_positive_multiple_of_45_l78_78147

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l78_78147


namespace Ann_is_16_l78_78749

variable (A S : ℕ)

theorem Ann_is_16
  (h1 : A = S + 5)
  (h2 : A + S = 27) :
  A = 16 :=
by
  sorry

end Ann_is_16_l78_78749


namespace find_n_l78_78284

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 14 = 56) (h_gcf : Nat.gcd n 14 = 12) : n = 48 :=
by
  sorry

end find_n_l78_78284


namespace expression_simplifies_to_one_l78_78045

-- Define x in terms of the given condition
def x : ℚ := (1 / 2) ^ (-1 : ℤ) + (-3) ^ (0 : ℤ)

-- Define the given expression
def expr (x : ℚ) : ℚ := (((x^2 - 1) / (x^2 - 2 * x + 1)) - (1 / (x - 1))) / (3 / (x - 1))

-- Define the theorem stating the equivalence
theorem expression_simplifies_to_one : expr x = 1 := by
  sorry

end expression_simplifies_to_one_l78_78045


namespace complete_the_square_l78_78290

theorem complete_the_square (x : ℝ) :
  (x^2 + 14*x + 60) = ((x + 7) ^ 2 + 11) :=
by
  sorry

end complete_the_square_l78_78290


namespace surface_area_increase_l78_78048

structure RectangularSolid (length : ℝ) (width : ℝ) (height : ℝ) where
  surface_area : ℝ := 2 * (length * width + length * height + width * height)

def cube_surface_contributions (side : ℝ) : ℝ := side ^ 2 * 3

theorem surface_area_increase
  (original : RectangularSolid 4 3 5)
  (cube_side : ℝ := 1) :
  let new_cube_contribution := cube_surface_contributions cube_side
  let removed_face : ℝ := cube_side ^ 2
  let original_surface_area := original.surface_area
  original_surface_area + new_cube_contribution - removed_face = original_surface_area + 2 :=
by
  sorry

end surface_area_increase_l78_78048


namespace breadth_remains_the_same_l78_78148

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l78_78148


namespace proper_sets_exist_l78_78864

def proper_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, (1 ≤ w ∧ w ≤ 500) → ∃ (used_weights : List ℕ), (used_weights ⊆ weights) ∧ (used_weights.sum = w ∧ ∀ (alternative_weights : List ℕ), (alternative_weights ⊆ weights ∧ alternative_weights.sum = w) → used_weights = alternative_weights)

theorem proper_sets_exist (weights : List ℕ) :
  (weights.sum = 500) → 
  ∃ (sets : List (List ℕ)), sets.length = 3 ∧ (∀ s ∈ sets, proper_set s) :=
by
  sorry

end proper_sets_exist_l78_78864


namespace largest_positive_real_root_l78_78392

theorem largest_positive_real_root (b2 b1 b0 : ℤ) (h2 : |b2| ≤ 3) (h1 : |b1| ≤ 3) (h0 : |b0| ≤ 3) :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + (b2 : ℝ) * r^2 + (b1 : ℝ) * r + (b0 : ℝ) = 0) ∧ 3.5 < r ∧ r < 4.0 :=
sorry

end largest_positive_real_root_l78_78392


namespace least_clock_equivalent_to_square_greater_than_4_l78_78430

theorem least_clock_equivalent_to_square_greater_than_4 : 
  ∃ (x : ℕ), x > 4 ∧ (x^2 - x) % 12 = 0 ∧ ∀ (y : ℕ), y > 4 → (y^2 - y) % 12 = 0 → x ≤ y :=
by
  -- The proof will go here
  sorry

end least_clock_equivalent_to_square_greater_than_4_l78_78430


namespace expected_carrot_yield_l78_78167

-- Condition definitions
def num_steps_width : ℕ := 16
def num_steps_length : ℕ := 22
def step_length : ℝ := 1.75
def avg_yield_per_sqft : ℝ := 0.75

-- Theorem statement
theorem expected_carrot_yield : 
  (num_steps_width * step_length) * (num_steps_length * step_length) * avg_yield_per_sqft = 808.5 :=
by
  sorry

end expected_carrot_yield_l78_78167


namespace binom_10_3_l78_78603

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l78_78603


namespace votes_cast_l78_78772

theorem votes_cast (V : ℝ) (hv1 : 0.35 * V + (0.35 * V + 1800) = V) : V = 6000 :=
sorry

end votes_cast_l78_78772


namespace selling_price_before_clearance_l78_78916

-- Define the cost price (CP)
def CP : ℝ := 100

-- Define the gain percent before the clearance sale
def gain_percent_before : ℝ := 0.35

-- Define the discount percent during the clearance sale
def discount_percent : ℝ := 0.10

-- Define the gain percent during the clearance sale
def gain_percent_sale : ℝ := 0.215

-- Calculate the selling price before the clearance sale (SP_before)
def SP_before : ℝ := CP * (1 + gain_percent_before)

-- Calculate the selling price during the clearance sale (SP_sale)
def SP_sale : ℝ := SP_before * (1 - discount_percent)

-- Proof statement in Lean 4
theorem selling_price_before_clearance : SP_before = 135 :=
by
  -- Place to fill in the proof later
  sorry

end selling_price_before_clearance_l78_78916


namespace g_triple_apply_l78_78919

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_triple_apply : g (g (g 20)) = 1 :=
by
  sorry

end g_triple_apply_l78_78919


namespace prism_volume_l78_78573

theorem prism_volume (a b c : ℝ) (h1 : a * b = 12) (h2 : b * c = 8) (h3 : a * c = 4) : a * b * c = 8 * Real.sqrt 6 :=
by 
  sorry

end prism_volume_l78_78573


namespace binomial_sum_eval_l78_78664

theorem binomial_sum_eval :
  (Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 5)) +
  (Nat.factorial 6 / (Nat.factorial 4 * Nat.factorial 2)) = 36 := by
sorry

end binomial_sum_eval_l78_78664


namespace blake_change_l78_78244

theorem blake_change :
  let lollipop_count := 4
  let chocolate_count := 6
  let lollipop_cost := 2
  let chocolate_cost := 4 * lollipop_cost
  let total_received := 6 * 10
  let total_cost := (lollipop_count * lollipop_cost) + (chocolate_count * chocolate_cost)
  let change := total_received - total_cost
  change = 4 :=
by
  sorry

end blake_change_l78_78244


namespace equilateral_triangle_perimeter_l78_78053

theorem equilateral_triangle_perimeter (a P : ℕ) 
  (h1 : 2 * a + 10 = 40)  -- Condition: perimeter of isosceles triangle is 40
  (h2 : P = 3 * a) :      -- Definition of perimeter of equilateral triangle
  P = 45 :=               -- Expected result
by
  sorry

end equilateral_triangle_perimeter_l78_78053


namespace arithmetic_progression_infinite_kth_powers_l78_78288

theorem arithmetic_progression_infinite_kth_powers {a d k : ℕ} (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∀ n : ℕ, ¬ ∃ b : ℕ, a + n * d = b ^ k) ∨ (∀ b : ℕ, ∃ n : ℕ, a + n * d = b ^ k) :=
sorry

end arithmetic_progression_infinite_kth_powers_l78_78288


namespace tire_swap_distance_l78_78182

theorem tire_swap_distance : ∃ x : ℕ, 
  (1 - x / 11000) * 9000 = (1 - x / 9000) * 11000 ∧ x = 4950 := 
by
  sorry

end tire_swap_distance_l78_78182


namespace second_chick_eats_52_l78_78753

theorem second_chick_eats_52 (days : ℕ) (first_chick_eats : ℕ → ℕ) (second_chick_eats : ℕ → ℕ) :
  (∀ n, first_chick_eats n + second_chick_eats n = 12) →
  (∃ a b, first_chick_eats a = 7 ∧ second_chick_eats a = 5 ∧
          first_chick_eats b = 7 ∧ second_chick_eats b = 5 ∧
          12 * days = first_chick_eats a * 2 + first_chick_eats b * 6 + second_chick_eats a * 2 + second_chick_eats b * 6) →
  (first_chick_eats a * 2 + first_chick_eats b * 6 = 44) →
  (second_chick_eats a * 2 + second_chick_eats b * 6 = 52) :=
by
  sorry

end second_chick_eats_52_l78_78753


namespace find_speeds_l78_78999

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l78_78999


namespace smallest_m_plus_n_l78_78819

theorem smallest_m_plus_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 3 * m^3 = 5 * n^5) : m + n = 720 :=
by
  sorry

end smallest_m_plus_n_l78_78819


namespace elvins_first_month_bill_l78_78939

-- Define the variables involved
variables (F C : ℝ)

-- State the given conditions
def condition1 : Prop := F + C = 48
def condition2 : Prop := F + 2 * C = 90

-- State the theorem we need to prove
theorem elvins_first_month_bill (F C : ℝ) (h1 : F + C = 48) (h2 : F + 2 * C = 90) : F + C = 48 :=
by sorry

end elvins_first_month_bill_l78_78939


namespace larger_number_is_2997_l78_78785

theorem larger_number_is_2997 (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := 
by
  sorry

end larger_number_is_2997_l78_78785


namespace math_problem_l78_78527

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l78_78527


namespace molecular_weight_CaOH2_correct_l78_78054

/-- Molecular weight of Calcium hydroxide -/
def molecular_weight_CaOH2 (Ca O H : ℝ) : ℝ :=
  Ca + 2 * (O + H)

theorem molecular_weight_CaOH2_correct :
  molecular_weight_CaOH2 40.08 16.00 1.01 = 74.10 :=
by 
  -- This statement requires a proof that would likely involve arithmetic on real numbers
  sorry

end molecular_weight_CaOH2_correct_l78_78054


namespace problem_l78_78610

theorem problem (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5) : 5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
by 
  sorry

end problem_l78_78610


namespace min_value_tan_product_l78_78583

theorem min_value_tan_product (A B C : ℝ) (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (sin_eq : Real.sin A = 3 * Real.sin B * Real.sin C) :
  ∃ t : ℝ, t = Real.tan A * Real.tan B * Real.tan C ∧ t = 12 :=
sorry

end min_value_tan_product_l78_78583


namespace vegetable_price_l78_78022

theorem vegetable_price (v : ℝ) 
  (beef_cost : ∀ (b : ℝ), b = 3 * v)
  (total_cost : 4 * (3 * v) + 6 * v = 36) : 
  v = 2 :=
by {
  -- The proof would go here.
  sorry
}

end vegetable_price_l78_78022


namespace find_salary_J_l78_78100

variables (J F M A May : ℝ)

def avg_salary_J_F_M_A (J F M A : ℝ) : Prop :=
  (J + F + M + A) / 4 = 8000

def avg_salary_F_M_A_May (F M A May : ℝ) : Prop :=
  (F + M + A + May) / 4 = 8700

def salary_May (May : ℝ) : Prop :=
  May = 6500

theorem find_salary_J (h1 : avg_salary_J_F_M_A J F M A) (h2 : avg_salary_F_M_A_May F M A May) (h3 : salary_May May) :
  J = 3700 :=
sorry

end find_salary_J_l78_78100


namespace weight_of_new_person_l78_78086

theorem weight_of_new_person {avg_increase : ℝ} (n : ℕ) (p : ℝ) (w : ℝ) (h : n = 8) (h1 : avg_increase = 2.5) (h2 : w = 67):
  p = 87 :=
by
  sorry

end weight_of_new_person_l78_78086


namespace range_of_a_l78_78495

variable (a x : ℝ)
def A (a : ℝ) := {x : ℝ | 2 * a ≤ x ∧ x ≤ a ^ 2 + 1}
def B (a : ℝ) := {x : ℝ | (x - 2) * (x - (3 * a + 1)) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ A a → x ∈ B a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by sorry

end range_of_a_l78_78495


namespace postal_service_revenue_l78_78443

theorem postal_service_revenue 
  (price_colored : ℝ := 0.50)
  (price_bw : ℝ := 0.35)
  (price_golden : ℝ := 2.00)
  (sold_colored : ℕ := 578833)
  (sold_bw : ℕ := 523776)
  (sold_golden : ℕ := 120456) : 
  (price_colored * (sold_colored : ℝ) + 
  price_bw * (sold_bw : ℝ) + 
  price_golden * (sold_golden : ℝ) = 713650.10) :=
by
  sorry

end postal_service_revenue_l78_78443


namespace linear_coefficient_l78_78577

theorem linear_coefficient (a b c : ℤ) (h : a = 1 ∧ b = -2 ∧ c = -1) :
    b = -2 := 
by
  -- Use the given hypothesis directly
  exact h.2.1

end linear_coefficient_l78_78577


namespace valid_assignment_l78_78969

/-- A function to check if an expression is a valid assignment expression -/
def is_assignment (lhs : String) (rhs : String) : Prop :=
  lhs = "x" ∧ (rhs = "3" ∨ rhs = "x + 1")

theorem valid_assignment :
  (is_assignment "x" "x + 1") ∧
  ¬(is_assignment "3" "x") ∧
  ¬(is_assignment "x" "3") ∧
  ¬(is_assignment "x" "x2 + 1") :=
by
  sorry

end valid_assignment_l78_78969


namespace sum_even_integers_102_to_200_l78_78552

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l78_78552


namespace sum_powers_divisible_by_10_l78_78019

theorem sum_powers_divisible_by_10 (n : ℕ) (hn : n % 4 ≠ 0) : 
  ∃ k : ℕ, 1^n + 2^n + 3^n + 4^n = 10 * k :=
  sorry

end sum_powers_divisible_by_10_l78_78019


namespace calculate_expression_l78_78777

theorem calculate_expression :
    (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := 
by 
  sorry

end calculate_expression_l78_78777


namespace simplify_polynomial_subtraction_l78_78867

/--
  Given the polynomials (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) and (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5),
  prove that their difference simplifies to x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3.
-/
theorem simplify_polynomial_subtraction  (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) = x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 :=
sorry

end simplify_polynomial_subtraction_l78_78867


namespace wet_surface_area_is_correct_l78_78297

-- Define the dimensions of the cistern
def cistern_length : ℝ := 6  -- in meters
def cistern_width  : ℝ := 4  -- in meters
def water_depth    : ℝ := 1.25  -- in meters

-- Compute areas for each surface in contact with water
def bottom_area : ℝ := cistern_length * cistern_width
def long_sides_area : ℝ := 2 * (cistern_length * water_depth)
def short_sides_area : ℝ := 2 * (cistern_width * water_depth)

-- Calculate the total area of the wet surface
def total_wet_surface_area : ℝ := bottom_area + long_sides_area + short_sides_area

-- Statement to prove
theorem wet_surface_area_is_correct : total_wet_surface_area = 49 := by
  sorry

end wet_surface_area_is_correct_l78_78297


namespace jack_mopping_rate_l78_78426

variable (bathroom_floor_area : ℕ) (kitchen_floor_area : ℕ) (time_mopped : ℕ)

theorem jack_mopping_rate
  (h_bathroom : bathroom_floor_area = 24)
  (h_kitchen : kitchen_floor_area = 80)
  (h_time : time_mopped = 13) :
  (bathroom_floor_area + kitchen_floor_area) / time_mopped = 8 :=
by
  sorry

end jack_mopping_rate_l78_78426


namespace weight_of_packet_a_l78_78737

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

end weight_of_packet_a_l78_78737


namespace unique_quotient_is_9742_l78_78558

theorem unique_quotient_is_9742 :
  ∃ (d4 d3 d2 d1 : ℕ),
    (d2 = d1 + 2) ∧
    (d4 = d3 + 2) ∧
    (0 ≤ d1 ∧ d1 ≤ 9) ∧
    (0 ≤ d2 ∧ d2 ≤ 9) ∧
    (0 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d4 * 1000 + d3 * 100 + d2 * 10 + d1 = 9742) :=
by sorry

end unique_quotient_is_9742_l78_78558


namespace find_sum_of_numbers_l78_78545

variables (a b c : ℕ) (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300)

theorem find_sum_of_numbers (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300) :
  a + b + c = 14700 :=
sorry

end find_sum_of_numbers_l78_78545


namespace max_togs_possible_l78_78531

def tag_cost : ℕ := 3
def tig_cost : ℕ := 4
def tog_cost : ℕ := 8
def total_budget : ℕ := 100
def min_tags : ℕ := 1
def min_tigs : ℕ := 1
def min_togs : ℕ := 1

theorem max_togs_possible : 
  ∃ (tags tigs togs : ℕ), tags ≥ min_tags ∧ tigs ≥ min_tigs ∧ togs ≥ min_togs ∧ 
  tag_cost * tags + tig_cost * tigs + tog_cost * togs = total_budget ∧ togs = 11 :=
sorry

end max_togs_possible_l78_78531


namespace Janice_age_l78_78877

theorem Janice_age (x : ℝ) (h : x + 12 = 8 * (x - 2)) : x = 4 := by
  sorry

end Janice_age_l78_78877


namespace equation_of_line_through_points_l78_78948

-- Definitions for the problem conditions
def point1 : ℝ × ℝ := (-1, 2)
def point2 : ℝ × ℝ := (-3, -2)

-- The theorem stating the equation of the line passing through the given points
theorem equation_of_line_through_points :
  ∃ a b c : ℝ, (a * point1.1 + b * point1.2 + c = 0) ∧ (a * point2.1 + b * point2.2 + c = 0) ∧ 
             (a = 2) ∧ (b = -1) ∧ (c = 4) :=
by
  sorry

end equation_of_line_through_points_l78_78948


namespace no_three_digit_numbers_divisible_by_30_l78_78566

def digits_greater_than_6 (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d > 6

theorem no_three_digit_numbers_divisible_by_30 :
  ∀ n, (100 ≤ n ∧ n < 1000 ∧ digits_greater_than_6 n ∧ n % 30 = 0) → false :=
by
  sorry

end no_three_digit_numbers_divisible_by_30_l78_78566


namespace intersection_of_M_and_N_l78_78305

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l78_78305


namespace area_BCD_sixteen_area_BCD_with_new_ABD_l78_78349

-- Define the conditions and parameters of the problem.
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions from part (a)
variable (AB_length : Real) (BC_length : Real) (area_ABD : Real)

-- Define the lengths and areas in our problem.
axiom AB_eq_five : AB_length = 5
axiom BC_eq_eight : BC_length = 8
axiom area_ABD_eq_ten : area_ABD = 10

-- Part (a) problem statement
theorem area_BCD_sixteen (AB_length BC_length area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → area_ABD = 10 → (∃ area_BCD : Real, area_BCD = 16) :=
by
  sorry

-- Given conditions from part (b)
variable (new_area_ABD : Real)

-- Define the new area.
axiom new_area_ABD_eq_hundred : new_area_ABD = 100

-- Part (b) problem statement
theorem area_BCD_with_new_ABD (AB_length BC_length new_area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → new_area_ABD = 100 → (∃ area_BCD : Real, area_BCD = 160) :=
by
  sorry

end area_BCD_sixteen_area_BCD_with_new_ABD_l78_78349


namespace find_f_at_6_5_l78_78801

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom functional_equation (x : ℝ) : f (x + 2) = - (1 / f x)
axiom initial_condition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : f x = x - 2

theorem find_f_at_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_at_6_5_l78_78801


namespace stock_yield_percentage_l78_78287

def annualDividend (parValue : ℕ) (rate : ℕ) : ℕ :=
  (parValue * rate) / 100

def yieldPercentage (dividend : ℕ) (marketPrice : ℕ) : ℕ :=
  (dividend * 100) / marketPrice

theorem stock_yield_percentage :
  let par_value := 100
  let rate := 8
  let market_price := 80
  yieldPercentage (annualDividend par_value rate) market_price = 10 :=
by
  sorry

end stock_yield_percentage_l78_78287


namespace puppies_per_dog_l78_78273

/--
Chuck breeds dogs. He has 3 pregnant dogs.
They each give birth to some puppies. Each puppy needs 2 shots and each shot costs $5.
The total cost of the shots is $120. Prove that each pregnant dog gives birth to 4 puppies.
-/
theorem puppies_per_dog :
  let num_dogs := 3
  let cost_per_shot := 5
  let shots_per_puppy := 2
  let total_cost := 120
  let cost_per_puppy := shots_per_puppy * cost_per_shot
  let total_puppies := total_cost / cost_per_puppy
  (total_puppies / num_dogs) = 4 := by
  sorry

end puppies_per_dog_l78_78273


namespace minimum_connected_components_l78_78599

/-- We start with two points A, B on a 6*7 lattice grid. We say two points 
  X, Y are connected if one can reflect several times with respect to points A, B 
  and reach from X to Y. Prove that the minimum number of connected components 
  over all choices of A, B is 8. -/
theorem minimum_connected_components (A B : ℕ × ℕ) 
  (hA : A.1 < 6 ∧ A.2 < 7) (hB : B.1 < 6 ∧ B.2 < 7) :
  ∃ k, k = 8 :=
sorry

end minimum_connected_components_l78_78599


namespace probability_three_heads_in_a_row_l78_78834

theorem probability_three_heads_in_a_row (h : ℝ) (p_head : h = 1/2) (ind_flips : ∀ (n : ℕ), true) : 
  (1/2 * 1/2 * 1/2 = 1/8) :=
by
  sorry

end probability_three_heads_in_a_row_l78_78834


namespace sale_in_fifth_month_l78_78570

theorem sale_in_fifth_month (s1 s2 s3 s4 s5 s6 : ℤ) (avg_sale : ℤ) (h1 : s1 = 6435) (h2 : s2 = 6927)
  (h3 : s3 = 6855) (h4 : s4 = 7230) (h6 : s6 = 7391) (h_avg_sale : avg_sale = 6900) :
    (s1 + s2 + s3 + s4 + s5 + s6) / 6 = avg_sale → s5 = 6562 :=
by
  sorry

end sale_in_fifth_month_l78_78570


namespace arcsin_one_half_l78_78341

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l78_78341


namespace satisfy_equation_l78_78296

theorem satisfy_equation (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end satisfy_equation_l78_78296


namespace boat_speed_determination_l78_78904

theorem boat_speed_determination :
  ∃ x : ℝ, 
    (∀ u d : ℝ, u = 170 / (x + 6) ∧ d = 170 / (x - 6))
    ∧ (u + d = 68)
    ∧ (x = 9) := 
by
  sorry

end boat_speed_determination_l78_78904


namespace algebraic_expression_value_l78_78861

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 23 - 1) : x^2 + 2 * x + 2 = 24 :=
by
  -- Start of the proof
  sorry -- Proof is omitted as per instructions

end algebraic_expression_value_l78_78861


namespace solution_set_of_inequality_l78_78286

noncomputable def f : ℝ → ℝ := sorry 

axiom f_cond : ∀ x : ℝ, f x + deriv f x > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_of_inequality : {x : ℝ | f x > 3 / Real.exp x + 1} = { x : ℝ | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l78_78286


namespace range_of_a_l78_78228

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * x * log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by 
  sorry

end range_of_a_l78_78228


namespace problem_statement_l78_78998

-- Defining the terms x, y, and d as per the problem conditions
def x : ℕ := 2351
def y : ℕ := 2250
def d : ℕ := 121

-- Stating the proof problem in Lean
theorem problem_statement : (x - y)^2 / d = 84 := by
  sorry

end problem_statement_l78_78998


namespace players_taking_chemistry_l78_78428

theorem players_taking_chemistry (total_players biology_players both_sci_players: ℕ) 
  (h1 : total_players = 12)
  (h2 : biology_players = 7)
  (h3 : both_sci_players = 2)
  (h4 : ∀ p, p <= total_players) : 
  ∃ chemistry_players, chemistry_players = 7 := 
sorry

end players_taking_chemistry_l78_78428


namespace tan_a_values_l78_78280

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 :=
by
  sorry

end tan_a_values_l78_78280


namespace problem_a4_inv_a4_eq_seven_l78_78543

theorem problem_a4_inv_a4_eq_seven (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + (1/a)^4 = 7 :=
sorry

end problem_a4_inv_a4_eq_seven_l78_78543


namespace digital_earth_sustainable_development_l78_78496

theorem digital_earth_sustainable_development :
  (after_realization_digital_earth : Prop) → (scientists_can : Prop) :=
sorry

end digital_earth_sustainable_development_l78_78496


namespace solution_set_of_inequality_l78_78972

theorem solution_set_of_inequality (x : ℝ) :
  2 * x ≤ -1 → x > -1 → -1 < x ∧ x ≤ -1 / 2 :=
by
  intro h1 h2
  have h3 : x ≤ -1 / 2 := by linarith
  exact ⟨h2, h3⟩

end solution_set_of_inequality_l78_78972


namespace ceil_floor_sum_l78_78342

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l78_78342


namespace part_I_part_II_l78_78143

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2 * m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem part_I (m : ℝ) : (∀ x, x ∈ A ∧ x ∈ B → x ∈ C m) → m ≥ 5 / 2 :=
sorry

theorem part_II (m : ℝ) : 
  (B ∪ (C m) = Set.univ) ∧ 
  (C m ⊆ D) → 
  7 / 2 ≤ m ∧ m < 4 :=
sorry

end part_I_part_II_l78_78143


namespace find_equation_l78_78810

theorem find_equation (x : ℝ) : 
  (3 + x < 1 → false) ∧
  ((x - 67 + 63 = x - 4) → false) ∧
  ((4.8 + x = x + 4.8) → false) ∧
  (x + 0.7 = 12 → true) := 
sorry

end find_equation_l78_78810


namespace employees_bonus_l78_78502

theorem employees_bonus (x y z : ℝ) 
  (h1 : x + y + z = 2970) 
  (h2 : y = (1 / 3) * x + 180) 
  (h3 : z = (1 / 3) * y + 130) :
  x = 1800 ∧ y = 780 ∧ z = 390 :=
by
  sorry

end employees_bonus_l78_78502


namespace probability_xiaoming_l78_78261

variable (win_probability : ℚ) 
          (xiaoming_goal : ℕ)
          (xiaojie_goal : ℕ)
          (rounds_needed_xiaoming : ℕ)
          (rounds_needed_xiaojie : ℕ)

def probability_xiaoming_wins_2_consecutive_rounds
   (win_probability : ℚ) 
   (rounds_needed_xiaoming : ℕ) : ℚ :=
  (win_probability ^ 2) + 
  2 * win_probability ^ 3 * (1 - win_probability) + 
  win_probability ^ 4

theorem probability_xiaoming :
    win_probability = (1/2) ∧ 
    rounds_needed_xiaoming = 2 ∧
    rounds_needed_xiaojie = 3 →
    probability_xiaoming_wins_2_consecutive_rounds (1 / 2) 2 = 7 / 16 :=
by
  -- Proof steps placeholder
  sorry

end probability_xiaoming_l78_78261


namespace new_milk_water_ratio_l78_78183

theorem new_milk_water_ratio
  (original_milk : ℚ)
  (original_water : ℚ)
  (added_water : ℚ)
  (h_ratio : original_milk / original_water = 2 / 1)
  (h_milk_qty : original_milk = 45)
  (h_added_water : added_water = 10) :
  original_milk / (original_water + added_water) = 18 / 13 :=
by
  sorry

end new_milk_water_ratio_l78_78183


namespace maximum_value_of_parabola_eq_24_l78_78158

theorem maximum_value_of_parabola_eq_24 (x : ℝ) : 
  ∃ x, x = -2 ∧ (-2 * x^2 - 8 * x + 16) = 24 :=
by
  use -2
  sorry

end maximum_value_of_parabola_eq_24_l78_78158


namespace solve_fractional_eq_l78_78293

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x - 1) = 1 / x) ↔ (x = -1) :=
by 
  sorry

end solve_fractional_eq_l78_78293


namespace find_a_l78_78743

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  2 * a * Real.cos (2 * x) - Real.cos (3 * x)

theorem find_a (a : ℝ) (h : f_prime a (Real.pi / 3) = 0) : a = 1 :=
by
  sorry

end find_a_l78_78743


namespace find_function_f_l78_78110

-- The function f maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- The statement to be proved
theorem find_function_f (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, (f m)^2 + f n ∣ (m^2 + n)^2) : ∀ n : ℕ+, f n = n :=
sorry

end find_function_f_l78_78110


namespace jared_annual_earnings_l78_78480

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l78_78480


namespace parallel_lines_slopes_l78_78755

theorem parallel_lines_slopes (k : ℝ) :
  (∀ x y : ℝ, x + (1 + k) * y = 2 - k → k * x + 2 * y + 8 = 0 → k = 1) :=
by
  intro h1 h2
  -- We can see that there should be specifics here about how the conditions lead to k = 1
  sorry

end parallel_lines_slopes_l78_78755


namespace coefficients_of_polynomial_l78_78563

theorem coefficients_of_polynomial (a_5 a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x : ℝ, x^5 = a_5 * (2*x + 1)^5 + a_4 * (2*x + 1)^4 + a_3 * (2*x + 1)^3 + a_2 * (2*x + 1)^2 + a_1 * (2*x + 1) + a_0) →
  a_5 = 1/32 ∧ a_4 = -5/32 :=
by sorry

end coefficients_of_polynomial_l78_78563


namespace cubic_root_expression_l78_78180

theorem cubic_root_expression (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + p * r + q * r = -2) (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -24 :=
sorry

end cubic_root_expression_l78_78180


namespace cubic_identity_l78_78831

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end cubic_identity_l78_78831


namespace mike_spend_on_plants_l78_78485

def Mike_buys : Prop :=
  let rose_bushes_total := 6
  let rose_bush_cost := 75
  let friend_rose_bushes := 2
  let self_rose_bushes := rose_bushes_total - friend_rose_bushes
  let self_rose_bush_cost := self_rose_bushes * rose_bush_cost
  let tiger_tooth_aloe_total := 2
  let aloe_cost := 100
  let self_aloe_cost := tiger_tooth_aloe_total * aloe_cost
  self_rose_bush_cost + self_aloe_cost = 500

theorem mike_spend_on_plants :
  Mike_buys := by
  sorry

end mike_spend_on_plants_l78_78485


namespace conic_sections_l78_78975

theorem conic_sections (x y : ℝ) : 
  y^4 - 16*x^4 = 8*y^2 - 4 → 
  (y^2 - 4 * x^2 = 4 ∨ y^2 + 4 * x^2 = 4) :=
sorry

end conic_sections_l78_78975


namespace cost_per_steak_knife_l78_78614

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end cost_per_steak_knife_l78_78614


namespace find_number_of_valid_polynomials_l78_78081

noncomputable def number_of_polynomials_meeting_constraints : Nat :=
  sorry

theorem find_number_of_valid_polynomials : number_of_polynomials_meeting_constraints = 11 :=
  sorry

end find_number_of_valid_polynomials_l78_78081


namespace mean_cars_l78_78708

theorem mean_cars (a b c d e : ℝ) (h1 : a = 30) (h2 : b = 14) (h3 : c = 14) (h4 : d = 21) (h5 : e = 25) : 
  (a + b + c + d + e) / 5 = 20.8 :=
by
  -- The proof will be provided here
  sorry

end mean_cars_l78_78708


namespace solution_set_of_inequality_l78_78407

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x ≥ 0} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by
  sorry

end solution_set_of_inequality_l78_78407


namespace opposite_of_2023_is_minus_2023_l78_78855

def opposite (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_2023_is_minus_2023 : opposite 2023 (-2023) :=
by
  sorry

end opposite_of_2023_is_minus_2023_l78_78855


namespace A_completion_time_l78_78481

theorem A_completion_time :
  ∃ A : ℝ, (A > 0) ∧ (
    (2 * (1 / A + 1 / 10) + 3.0000000000000004 * (1 / 10) = 1) ↔ A = 4
  ) :=
by
  have B_workday := 10
  sorry -- proof would go here

end A_completion_time_l78_78481


namespace chairperson_and_committee_ways_l78_78821

-- Definitions based on conditions
def total_people : ℕ := 10
def ways_to_choose_chairperson : ℕ := total_people
def ways_to_choose_committee (remaining_people : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose remaining_people committee_size

-- The resulting theorem
theorem chairperson_and_committee_ways :
  ways_to_choose_chairperson * ways_to_choose_committee (total_people - 1) 3 = 840 :=
by
  sorry

end chairperson_and_committee_ways_l78_78821


namespace trigonometric_identity_l78_78171

theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (-1 + Real.sqrt 3) / 2 :=
sorry

end trigonometric_identity_l78_78171


namespace percent_children_with_both_colors_l78_78635

theorem percent_children_with_both_colors
  (F : ℕ) (C : ℕ) 
  (even_F : F % 2 = 0)
  (children_pick_two_flags : C = F / 2)
  (sixty_percent_blue : 6 * C / 10 = 6 * C / 10)
  (fifty_percent_red : 5 * C / 10 = 5 * C / 10)
  : (6 * C / 10) + (5 * C / 10) - C = C / 10 :=
by
  sorry

end percent_children_with_both_colors_l78_78635


namespace intersection_complement_l78_78431

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem intersection_complement : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_complement_l78_78431


namespace percent_greater_than_average_l78_78954

variable (M N : ℝ)

theorem percent_greater_than_average (h : M > N) :
  (200 * (M - N)) / (M + N) = ((M - ((M + N) / 2)) / ((M + N) / 2)) * 100 :=
by 
  sorry

end percent_greater_than_average_l78_78954


namespace find_probability_between_0_and_1_l78_78735

-- Define a random variable X following a normal distribution N(μ, σ²)
variables {X : ℝ → ℝ} {μ σ : ℝ}
-- Define conditions:
-- Condition 1: X follows a normal distribution with mean μ and variance σ²
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  sorry  -- Assume properties of normal distribution are satisfied

-- Condition 2: P(X < 1) = 1/2
def P_X_lt_1 : Prop := 
  sorry  -- Assume that P(X < 1) = 1/2

-- Condition 3: P(X > 2) = p
def P_X_gt_2 (p : ℝ) : Prop := 
  sorry  -- Assume that P(X > 2) = p

noncomputable
def probability_X_between_0_and_1 (p : ℝ) : ℝ :=
  1/2 - p

theorem find_probability_between_0_and_1 (X : ℝ → ℝ) {μ σ p : ℝ} 
  (hX : normal_dist X μ σ)
  (h1 : P_X_lt_1)
  (h2 : P_X_gt_2 p) :
  probability_X_between_0_and_1 p = 1/2 - p := 
  sorry

end find_probability_between_0_and_1_l78_78735


namespace least_tiles_required_l78_78455

def room_length : ℕ := 7550
def room_breadth : ℕ := 2085
def tile_size : ℕ := 5
def total_area : ℕ := room_length * room_breadth
def tile_area : ℕ := tile_size * tile_size
def number_of_tiles : ℕ := total_area / tile_area

theorem least_tiles_required : number_of_tiles = 630270 := by
  sorry

end least_tiles_required_l78_78455


namespace female_democrats_l78_78419

theorem female_democrats :
  ∀ (F M : ℕ),
  F + M = 720 →
  F/2 + M/4 = 240 →
  F / 2 = 120 :=
by
  intros F M h1 h2
  sorry

end female_democrats_l78_78419


namespace milk_exchange_l78_78694

theorem milk_exchange (initial_empty_bottles : ℕ) (exchange_rate : ℕ) (start_full_bottles : ℕ) : initial_empty_bottles = 43 → exchange_rate = 4 → start_full_bottles = 0 → ∃ liters_of_milk : ℕ, liters_of_milk = 14 :=
by
  intro h1 h2 h3
  sorry

end milk_exchange_l78_78694


namespace carrot_price_l78_78666

variables (total_tomatoes : ℕ) (total_carrots : ℕ) (price_per_tomato : ℝ) (total_revenue : ℝ)

theorem carrot_price :
  total_tomatoes = 200 →
  total_carrots = 350 →
  price_per_tomato = 1 →
  total_revenue = 725 →
  (total_revenue - total_tomatoes * price_per_tomato) / total_carrots = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end carrot_price_l78_78666


namespace find_number_l78_78600

theorem find_number (x : ℤ) (h : 45 - (28 - (37 - (x - 18))) = 57) : x = 15 :=
by
  sorry

end find_number_l78_78600


namespace part1_part2_l78_78163

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part (1): Prove range for m
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic 1 (-5) m x = 0) ↔ m ≤ 25 / 4 := sorry

-- Part (2): Prove value of m given conditions on roots
theorem part2 (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : 3 * x1 - 2 * x2 = 5) : 
  m = x1 * x2 → m = 6 := sorry

end part1_part2_l78_78163


namespace find_x_solution_l78_78587

noncomputable def find_x (x y : ℝ) (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : Prop := 
  x = (3 + Real.sqrt 17) / 2

theorem find_x_solution (x y : ℝ) 
(h1 : x - y^2 = 3) 
(h2 : x^2 + y^4 = 13) 
(hx_pos : 0 < x) 
(hy_pos : 0 < y) : 
  find_x x y h1 h2 :=
sorry

end find_x_solution_l78_78587


namespace recurring_decimals_sum_l78_78257

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l78_78257


namespace oil_consumption_relation_l78_78264

noncomputable def initial_oil : ℝ := 62

noncomputable def remaining_oil (x : ℝ) : ℝ :=
  if x = 100 then 50
  else if x = 200 then 38
  else if x = 300 then 26
  else if x = 400 then 14
  else 62 - 0.12 * x

theorem oil_consumption_relation (x : ℝ) :
  remaining_oil x = 62 - 0.12 * x := by
  sorry

end oil_consumption_relation_l78_78264


namespace probability_detecting_drunk_driver_l78_78707

namespace DrunkDrivingProbability

def P_A : ℝ := 0.05
def P_B_given_A : ℝ := 0.99
def P_B_given_not_A : ℝ := 0.01

def P_not_A : ℝ := 1 - P_A

def P_B : ℝ := P_A * P_B_given_A + P_not_A * P_B_given_not_A

theorem probability_detecting_drunk_driver :
  P_B = 0.059 :=
by
  sorry

end DrunkDrivingProbability

end probability_detecting_drunk_driver_l78_78707


namespace ratio_bee_eaters_leopards_l78_78686

variables (s f l c a t e r : ℕ)

-- Define the conditions from the problem.
def conditions : Prop :=
  s = 100 ∧
  f = 80 ∧
  l = 20 ∧
  c = s / 2 ∧
  a = 2 * (f + l) ∧
  t = 670 ∧
  e = t - (s + f + l + c + a)

-- The theorem statement proving the ratio.
theorem ratio_bee_eaters_leopards (h : conditions s f l c a t e) : r = (e / l) := by
  sorry

end ratio_bee_eaters_leopards_l78_78686


namespace divisor_of_109_l78_78887

theorem divisor_of_109 (d : ℕ) (h : 109 = 9 * d + 1) : d = 12 :=
sorry

end divisor_of_109_l78_78887


namespace lcm_12_20_correct_l78_78174

def lcm_12_20_is_60 : Nat := Nat.lcm 12 20

theorem lcm_12_20_correct : Nat.lcm 12 20 = 60 := by
  -- assumed factorization conditions as prerequisites
  have h₁ : Nat.primeFactors 12 = {2, 3} := sorry
  have h₂ : Nat.primeFactors 20 = {2, 5} := sorry
  -- the main proof goal
  exact sorry

end lcm_12_20_correct_l78_78174


namespace intercepts_sum_eq_seven_l78_78119

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l78_78119


namespace jane_total_drawing_paper_l78_78732

theorem jane_total_drawing_paper (brown_sheets : ℕ) (yellow_sheets : ℕ) 
    (h1 : brown_sheets = 28) (h2 : yellow_sheets = 27) : 
    brown_sheets + yellow_sheets = 55 := 
by
    sorry

end jane_total_drawing_paper_l78_78732


namespace problem1_problem2_l78_78912

theorem problem1 : 
  (5 / 7 : ℚ) * (-14 / 3) / (5 / 3) = -2 := 
by 
  sorry

theorem problem2 : 
  (-15 / 7 : ℚ) / (-6 / 5) * (-7 / 5) = -5 / 2 := 
by 
  sorry

end problem1_problem2_l78_78912


namespace find_alpha_l78_78705

def point (α : ℝ) : Prop := 3^α = Real.sqrt 3

theorem find_alpha (α : ℝ) (h : point α) : α = 1/2 := 
by 
  sorry

end find_alpha_l78_78705


namespace find_2xy2_l78_78791

theorem find_2xy2 (x y : ℤ) (h : y^2 + 2 * x^2 * y^2 = 20 * x^2 + 412) : 2 * x * y^2 = 288 :=
sorry

end find_2xy2_l78_78791


namespace oil_spent_amount_l78_78090

theorem oil_spent_amount :
  ∀ (P R M : ℝ), R = 25 → P = (R / 0.75) → ((M / R) - (M / P) = 5) → M = 500 :=
by
  intros P R M hR hP hOil
  sorry

end oil_spent_amount_l78_78090


namespace chord_square_length_eq_512_l78_78503

open Real

/-
The conditions are:
1. The radii of two smaller circles are 4 and 8.
2. These circles are externally tangent to each other.
3. Both smaller circles are internally tangent to a larger circle with radius 12.
4. A common external tangent to the two smaller circles serves as a chord of the larger circle.
-/

noncomputable def radius_small1 : ℝ := 4
noncomputable def radius_small2 : ℝ := 8
noncomputable def radius_large : ℝ := 12

/-- Show that the square of the length of the chord formed by the common external tangent of two smaller circles 
which are externally tangent to each other and internally tangent to a larger circle is 512. -/
theorem chord_square_length_eq_512 : ∃ (PQ : ℝ), PQ^2 = 512 := by
  sorry

end chord_square_length_eq_512_l78_78503


namespace range_of_a_for_f_ge_a_l78_78657

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a_for_f_ge_a :
  (∀ x : ℝ, (-1 ≤ x → f x a ≥ a)) ↔ (-3 ≤ a ∧ a ≤ 1) :=
  sorry

end range_of_a_for_f_ge_a_l78_78657


namespace tall_wins_min_voters_l78_78781

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end tall_wins_min_voters_l78_78781


namespace largest_tile_size_l78_78807

def length_cm : ℕ := 378
def width_cm : ℕ := 525

theorem largest_tile_size :
  Nat.gcd length_cm width_cm = 21 := by
  sorry

end largest_tile_size_l78_78807


namespace notebooks_ratio_l78_78739

variable (C N : Nat)

theorem notebooks_ratio (h1 : 512 = C * N)
  (h2 : 512 = 16 * (C / 2)) :
  N = C / 8 :=
by
  sorry

end notebooks_ratio_l78_78739


namespace machine_bottle_caps_l78_78250

variable (A_rate : ℕ)
variable (A_time : ℕ)
variable (B_rate : ℕ)
variable (B_time : ℕ)
variable (C_rate : ℕ)
variable (C_time : ℕ)
variable (D_rate : ℕ)
variable (D_time : ℕ)
variable (E_rate : ℕ)
variable (E_time : ℕ)

def A_bottles := A_rate * A_time
def B_bottles := B_rate * B_time
def C_bottles := C_rate * C_time
def D_bottles := D_rate * D_time
def E_bottles := E_rate * E_time

theorem machine_bottle_caps (hA_rate : A_rate = 24)
                            (hA_time : A_time = 10)
                            (hB_rate : B_rate = A_rate - 3)
                            (hB_time : B_time = 12)
                            (hC_rate : C_rate = B_rate + 6)
                            (hC_time : C_time = 15)
                            (hD_rate : D_rate = C_rate - 4)
                            (hD_time : D_time = 8)
                            (hE_rate : E_rate = D_rate + 5)
                            (hE_time : E_time = 5) :
  A_bottles A_rate A_time = 240 ∧ 
  B_bottles B_rate B_time = 252 ∧ 
  C_bottles C_rate C_time = 405 ∧ 
  D_bottles D_rate D_time = 184 ∧ 
  E_bottles E_rate E_time = 140 := by
    sorry

end machine_bottle_caps_l78_78250


namespace base_6_digit_divisibility_l78_78064

theorem base_6_digit_divisibility (d : ℕ) (h1 : d < 6) : ∃ t : ℤ, (655 + 42 * d) = 13 * t :=
by sorry

end base_6_digit_divisibility_l78_78064


namespace neg_i_pow_four_l78_78055

-- Define i as the imaginary unit satisfying i^2 = -1
def i : ℂ := Complex.I

-- The proof problem: Prove (-i)^4 = 1 given i^2 = -1
theorem neg_i_pow_four : (-i)^4 = 1 :=
by
  -- sorry is used to skip proof
  sorry

end neg_i_pow_four_l78_78055


namespace sandwiches_per_person_l78_78476

-- Definitions derived from conditions
def cost_of_12_croissants := 8.0
def number_of_people := 24
def total_spending := 32.0
def croissants_per_set := 12

-- Statement to be proved
theorem sandwiches_per_person :
  ∀ (cost_of_12_croissants total_spending croissants_per_set number_of_people : ℕ),
  total_spending / cost_of_12_croissants * croissants_per_set / number_of_people = 2 :=
by
  sorry

end sandwiches_per_person_l78_78476


namespace work_finished_days_earlier_l78_78964

theorem work_finished_days_earlier
  (D : ℕ) (M : ℕ) (A : ℕ) (Work : ℕ) (D_new : ℕ) (E : ℕ)
  (hD : D = 8)
  (hM : M = 30)
  (hA : A = 10)
  (hWork : Work = M * D)
  (hTotalWork : Work = 240)
  (hD_new : D_new = Work / (M + A))
  (hDnew_calculated : D_new = 6)
  (hE : E = D - D_new)
  (hE_calculated : E = 2) : 
  E = 2 :=
by
  sorry

end work_finished_days_earlier_l78_78964


namespace least_distance_fly_crawled_l78_78898

noncomputable def leastDistance (baseRadius height startDist endDist : ℝ) : ℝ :=
  let C := 2 * Real.pi * baseRadius
  let slantHeight := Real.sqrt (baseRadius ^ 2 + height ^ 2)
  let theta := C / slantHeight
  let x1 := startDist * Real.cos 0
  let y1 := startDist * Real.sin 0
  let x2 := endDist * Real.cos (theta / 2)
  let y2 := endDist * Real.sin (theta / 2)
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem least_distance_fly_crawled (baseRadius height startDist endDist : ℝ) (h1 : baseRadius = 500) (h2 : height = 150 * Real.sqrt 7) (h3 : startDist = 150) (h4 : endDist = 300 * Real.sqrt 2) :
  leastDistance baseRadius height startDist endDist = 150 * Real.sqrt 13 := by
  sorry

end least_distance_fly_crawled_l78_78898


namespace reduced_rates_start_l78_78730

theorem reduced_rates_start (reduced_fraction : ℝ) (total_hours : ℝ) (weekend_hours : ℝ) (weekday_hours : ℝ) 
  (start_time : ℝ) (end_time : ℝ) : 
  reduced_fraction = 0.6428571428571429 → 
  total_hours = 168 → 
  weekend_hours = 48 → 
  weekday_hours = 60 - weekend_hours → 
  end_time = 8 → 
  start_time = end_time - weekday_hours → 
  start_time = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end reduced_rates_start_l78_78730


namespace maximize_sales_volume_l78_78448

open Real

def profit (x : ℝ) : ℝ := (x - 20) * (400 - 20 * (x - 30))

theorem maximize_sales_volume : 
  ∃ x : ℝ, (∀ x' : ℝ, profit x' ≤ profit x) ∧ x = 35 := 
by
  sorry

end maximize_sales_volume_l78_78448


namespace initial_cards_l78_78138

theorem initial_cards (taken left initial : ℕ) (h1 : taken = 59) (h2 : left = 17) (h3 : initial = left + taken) : initial = 76 :=
by
  sorry

end initial_cards_l78_78138


namespace nail_polishes_total_l78_78977

theorem nail_polishes_total :
  let k := 25
  let h := k + 8
  let r := k - 6
  h + r = 52 :=
by
  sorry

end nail_polishes_total_l78_78977


namespace find_values_of_a_and_b_l78_78246

theorem find_values_of_a_and_b (a b : ℚ) (h1 : 4 * a + 2 * b = 92) (h2 : 6 * a - 4 * b = 60) : 
  a = 122 / 7 ∧ b = 78 / 7 :=
by {
  sorry
}

end find_values_of_a_and_b_l78_78246


namespace discount_difference_l78_78367

theorem discount_difference (bill_amt : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  bill_amt = 12000 → d1 = 0.42 → d2 = 0.35 → d3 = 0.05 →
  (bill_amt * (1 - d2) * (1 - d3) - bill_amt * (1 - d1) = 450) :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end discount_difference_l78_78367


namespace log_expression_defined_l78_78993

theorem log_expression_defined (x : ℝ) : ∃ c : ℝ, (∀ x > c, (x > 7^8)) :=
by
  existsi 7^8
  intro x hx
  sorry

end log_expression_defined_l78_78993


namespace swap_numbers_l78_78794

-- Define the initial state
variables (a b c : ℕ)
axiom initial_state : a = 8 ∧ b = 17

-- Define the assignment sequence
axiom swap_statement1 : c = b 
axiom swap_statement2 : b = a
axiom swap_statement3 : a = c

-- Define the theorem to be proved
theorem swap_numbers (a b c : ℕ) (initial_state : a = 8 ∧ b = 17)
  (swap_statement1 : c = b) (swap_statement2 : b = a) (swap_statement3 : a = c) :
  (a = 17 ∧ b = 8) :=
sorry

end swap_numbers_l78_78794


namespace least_positive_x_multiple_of_53_l78_78718

theorem least_positive_x_multiple_of_53 :
  ∃ (x : ℕ), (x > 0) ∧ ((2 * x)^2 + 2 * 47 * (2 * x) + 47^2) % 53 = 0 ∧ x = 6 :=
by
  sorry

end least_positive_x_multiple_of_53_l78_78718


namespace each_boy_brought_nine_cups_l78_78074

/--
There are 30 students in Ms. Leech's class. Twice as many girls as boys are in the class.
There are 10 boys in the class and the total number of cups brought by the students 
in the class is 90. Prove that each boy brought 9 cups.
-/
theorem each_boy_brought_nine_cups (students girls boys cups : ℕ) 
  (h1 : students = 30) 
  (h2 : girls = 2 * boys) 
  (h3 : boys = 10) 
  (h4 : cups = 90) 
  : cups / boys = 9 := 
sorry

end each_boy_brought_nine_cups_l78_78074


namespace solve_abs_quadratic_eq_l78_78442

theorem solve_abs_quadratic_eq (x : ℝ) (h : |2 * x + 4| = 1 - 3 * x + x ^ 2) :
    x = (5 + Real.sqrt 37) / 2 ∨ x = (5 - Real.sqrt 37) / 2 := by
  sorry

end solve_abs_quadratic_eq_l78_78442


namespace more_red_than_yellow_l78_78313

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l78_78313


namespace beads_removed_l78_78949

def total_beads (blue yellow : Nat) : Nat := blue + yellow

def beads_per_part (total : Nat) (parts : Nat) : Nat := total / parts

def beads_remaining (per_part : Nat) (removed : Nat) : Nat := per_part - removed

def doubled_beads (remaining : Nat) : Nat := 2 * remaining

theorem beads_removed {x : Nat} 
  (blue : Nat) (yellow : Nat) (parts : Nat) (final_per_part : Nat) :
  total_beads blue yellow = 39 →
  parts = 3 →
  beads_per_part 39 parts = 13 →
  doubled_beads (beads_remaining 13 x) = 6 →
  x = 10 := by
  sorry

end beads_removed_l78_78949


namespace value_of_expression_l78_78327

variable (a b c : ℝ)

theorem value_of_expression (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
                            (h2 : abc = 1)
                            (h3 : a^2 + b^2 + c^2 - ((1 / (a^2)) + (1 / (b^2)) + (1 / (c^2))) = 8 * (a + b + c) - 8 * (ab + bc + ca)) :
                            (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) = -3/2 :=
by
  sorry

end value_of_expression_l78_78327


namespace machineB_produces_100_parts_in_40_minutes_l78_78604

-- Define the given conditions
def machineA_rate := 50 / 10 -- Machine A's rate in parts per minute
def machineB_rate := machineA_rate / 2 -- Machine B's rate in parts per minute

-- Machine A produces 50 parts in 10 minutes
def machineA_50_parts_time : ℝ := 10

-- Machine B's time to produce 100 parts (The question)
def machineB_100_parts_time : ℝ := 40

-- Proving that Machine B takes 40 minutes to produce 100 parts
theorem machineB_produces_100_parts_in_40_minutes :
    machineB_100_parts_time = 40 :=
by
  sorry

end machineB_produces_100_parts_in_40_minutes_l78_78604


namespace probability_triangle_or_hexagon_l78_78039

theorem probability_triangle_or_hexagon 
  (total_shapes : ℕ) 
  (num_triangles : ℕ) 
  (num_squares : ℕ) 
  (num_circles : ℕ) 
  (num_hexagons : ℕ)
  (htotal : total_shapes = 10)
  (htriangles : num_triangles = 3)
  (hsquares : num_squares = 4)
  (hcircles : num_circles = 2)
  (hhexagons : num_hexagons = 1):
  (num_triangles + num_hexagons) / total_shapes = 2 / 5 := 
by 
  sorry

end probability_triangle_or_hexagon_l78_78039


namespace Maggie_age_l78_78277

theorem Maggie_age (Kate Maggie Sue : ℕ) (h1 : Kate + Maggie + Sue = 48) (h2 : Kate = 19) (h3 : Sue = 12) : Maggie = 17 := by
  sorry

end Maggie_age_l78_78277


namespace c_work_rate_l78_78479

/--
A can do a piece of work in 4 days.
B can do it in 8 days.
With the assistance of C, A and B completed the work in 2 days.
Prove that C alone can do the work in 8 days.
-/
theorem c_work_rate :
  (1 / 4 + 1 / 8 + 1 / c = 1 / 2) → c = 8 :=
by
  intro h
  sorry

end c_work_rate_l78_78479


namespace jacob_dimes_l78_78889

-- Definitions of the conditions
def mrs_hilt_total_cents : ℕ := 2 * 1 + 2 * 10 + 2 * 5
def jacob_base_cents : ℕ := 4 * 1 + 1 * 5
def difference : ℕ := 13

-- The proof problem: prove Jacob has 1 dime.
theorem jacob_dimes (d : ℕ) (h : mrs_hilt_total_cents - (jacob_base_cents + 10 * d) = difference) : d = 1 := by
  sorry

end jacob_dimes_l78_78889


namespace johns_total_spending_l78_78243

theorem johns_total_spending:
  ∀ (X : ℝ), (3/7 * X + 2/5 * X + 1/4 * X + 1/14 * X + 12 = X) → X = 80 :=
by
  intro X h
  sorry

end johns_total_spending_l78_78243


namespace petya_vasya_same_sum_l78_78765

theorem petya_vasya_same_sum :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2^99 * (2^100 - 1) :=
by
  sorry

end petya_vasya_same_sum_l78_78765


namespace compass_legs_cannot_swap_l78_78560

-- Define the problem conditions: compass legs on infinite grid, constant distance d.
def on_grid (p q : ℤ × ℤ) : Prop := 
  ∃ d : ℕ, d * d = (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) ∧ d > 0

-- Define the main theorem as a Lean 4 statement
theorem compass_legs_cannot_swap (p q : ℤ × ℤ) (h : on_grid p q) : 
  ¬ ∃ r s : ℤ × ℤ, on_grid r p ∧ on_grid s p ∧ p ≠ q ∧ r = q ∧ s = p :=
sorry

end compass_legs_cannot_swap_l78_78560


namespace find_g_l78_78041

noncomputable def g (x : ℝ) := -4 * x ^ 4 + x ^ 3 - 6 * x ^ 2 + x - 1

theorem find_g (x : ℝ) :
  4 * x ^ 4 + 2 * x ^ 2 - x + 7 + g x = x ^ 3 - 4 * x ^ 2 + 6 :=
by
  sorry

end find_g_l78_78041


namespace number_of_boxes_in_each_case_l78_78312

theorem number_of_boxes_in_each_case (a b : ℕ) :
    a + b = 2 → 9 = a * 8 + b :=
by
    intro h
    sorry

end number_of_boxes_in_each_case_l78_78312


namespace aras_current_height_l78_78266

-- Define the variables and conditions
variables (x : ℝ) (sheas_original_height : ℝ := x) (ars_original_height : ℝ := x)
variables (sheas_growth_factor : ℝ := 0.30) (sheas_current_height : ℝ := 65)
variables (sheas_growth : ℝ := sheas_current_height - sheas_original_height)
variables (aras_growth : ℝ := sheas_growth / 3)

-- Define a theorem for Ara's current height
theorem aras_current_height (h1 : sheas_current_height = (1 + sheas_growth_factor) * sheas_original_height)
                           (h2 : sheas_original_height = ars_original_height) :
                           aras_growth + ars_original_height = 55 :=
by
  sorry

end aras_current_height_l78_78266


namespace jane_paints_correct_area_l78_78914

def height_of_wall : ℕ := 10
def length_of_wall : ℕ := 15
def width_of_door : ℕ := 3
def height_of_door : ℕ := 5

def area_of_wall := height_of_wall * length_of_wall
def area_of_door := width_of_door * height_of_door
def area_to_be_painted := area_of_wall - area_of_door

theorem jane_paints_correct_area : area_to_be_painted = 135 := by
  sorry

end jane_paints_correct_area_l78_78914


namespace solve_for_x_l78_78316

theorem solve_for_x : ∀ (x : ℕ), (1000 = 10^3) → (40 = 2^3 * 5) → 1000^5 = 40^x → x = 15 :=
by
  intros x h1 h2 h3
  sorry

end solve_for_x_l78_78316


namespace book_pages_l78_78703

theorem book_pages (total_pages : ℝ) : 
  (0.1 * total_pages + 0.25 * total_pages + 30 = 0.5 * total_pages) → 
  total_pages = 240 :=
by
  sorry

end book_pages_l78_78703


namespace min_value_of_f_value_of_a_l78_78456

-- Definition of the function f
def f (x : ℝ) : ℝ := abs (x + 2) + 2 * abs (x - 1)

-- Problem: Prove that the minimum value of f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := sorry

-- Additional definitions for the second part of the problem
def g (x a : ℝ) : ℝ := f x + x - a

-- Problem: Given that the solution set of g(x,a) < 0 is (m, n) and n - m = 6, prove that a = 8
theorem value_of_a (a : ℝ) (m n : ℝ) (h : ∀ x : ℝ, g x a < 0 ↔ m < x ∧ x < n) (h_interval : n - m = 6) : a = 8 := sorry

end min_value_of_f_value_of_a_l78_78456


namespace union_of_A_and_B_l78_78655

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set B
def B := {x : ℝ | x < 1}

-- The proof problem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} :=
by sorry

end union_of_A_and_B_l78_78655


namespace cubes_in_fig_6_surface_area_fig_10_l78_78523

-- Define the function to calculate the number of unit cubes in Fig. n
def cubes_in_fig (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define the function to calculate the surface area of the solid figure for Fig. n
def surface_area_fig (n : ℕ) : ℕ := 6 * n * n

-- Theorem statements
theorem cubes_in_fig_6 : cubes_in_fig 6 = 91 :=
by sorry

theorem surface_area_fig_10 : surface_area_fig 10 = 600 :=
by sorry

end cubes_in_fig_6_surface_area_fig_10_l78_78523


namespace total_hours_worked_l78_78717

theorem total_hours_worked
  (x : ℕ)
  (h1 : 5 * x = 55)
  : 2 * x + 3 * x + 5 * x = 110 :=
by 
  sorry

end total_hours_worked_l78_78717


namespace inequality_relationship_cannot_be_established_l78_78128

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_relationship_cannot_be_established :
  ¬ (1 / (a - b) > 1 / a) :=
by sorry

end inequality_relationship_cannot_be_established_l78_78128


namespace roots_cubic_sum_l78_78710

theorem roots_cubic_sum:
  (∃ p q r : ℝ, 
     (p^3 - p^2 + p - 2 = 0) ∧ 
     (q^3 - q^2 + q - 2 = 0) ∧ 
     (r^3 - r^2 + r - 2 = 0)) 
  → 
  (∃ p q r : ℝ, p^3 + q^3 + r^3 = 4) := 
by 
  sorry

end roots_cubic_sum_l78_78710


namespace simplify_expression_and_evaluate_at_zero_l78_78462

theorem simplify_expression_and_evaluate_at_zero :
  ((2 * (0 : ℝ) - 1) / (0 + 1) - 0 + 1) / ((0 - 2) / ((0 ^ 2) + 2 * 0 + 1)) = 0 :=
by
  -- proof omitted
  sorry

end simplify_expression_and_evaluate_at_zero_l78_78462


namespace a_horses_is_18_l78_78194

-- Definitions of given conditions
def total_cost : ℕ := 435
def b_share : ℕ := 180
def horses_b : ℕ := 16
def months_b : ℕ := 9
def cost_b : ℕ := horses_b * months_b

def horses_c : ℕ := 18
def months_c : ℕ := 6
def cost_c : ℕ := horses_c * months_c

def total_cost_eq (x : ℕ) : Prop :=
  x * 8 + cost_b + cost_c = total_cost

-- Statement of the proof problem
theorem a_horses_is_18 (x : ℕ) : total_cost_eq x → x = 18 := 
sorry

end a_horses_is_18_l78_78194


namespace twenty_twenty_third_term_l78_78170

def sequence_denominator (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_numerator_pos (n : ℕ) : ℕ :=
  (n + 1) / 2

def sequence_numerator_neg (n : ℕ) : ℤ :=
  -((n + 1) / 2 : ℤ)

def sequence_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then 
    (sequence_numerator_pos n) / (sequence_denominator n) 
  else 
    (sequence_numerator_neg n : ℚ) / (sequence_denominator n)

theorem twenty_twenty_third_term :
  sequence_term 2023 = 1012 / 4045 := 
sorry

end twenty_twenty_third_term_l78_78170


namespace ab_zero_if_conditions_l78_78383

theorem ab_zero_if_conditions 
  (a b : ℤ)
  (h : |a - b| + |a * b| = 2) : a * b = 0 :=
  sorry

end ab_zero_if_conditions_l78_78383


namespace soldiers_arrival_time_l78_78058

open Function

theorem soldiers_arrival_time
    (num_soldiers : ℕ) (distance : ℝ) (car_speed : ℝ) (car_capacity : ℕ) (walk_speed : ℝ) (start_time : ℝ) :
    num_soldiers = 12 →
    distance = 20 →
    car_speed = 20 →
    car_capacity = 4 →
    walk_speed = 4 →
    start_time = 0 →
    ∃ arrival_time, arrival_time = 2 + 36/60 :=
by
  intros
  sorry

end soldiers_arrival_time_l78_78058


namespace triangle_inequality_l78_78766

variables (a b c : ℝ)

theorem triangle_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b > c) (h₄ : b + c > a) (h₅ : c + a > b) :
  (|a^2 - b^2| / c) + (|b^2 - c^2| / a) ≥ (|c^2 - a^2| / b) :=
by
  sorry

end triangle_inequality_l78_78766


namespace intersection_A_B_l78_78507

def A : Set ℤ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 2, 3} :=
by sorry

end intersection_A_B_l78_78507


namespace inequality_bounds_of_xyz_l78_78516

theorem inequality_bounds_of_xyz
  (x y z : ℝ)
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 6)
  (h4 : x * y + y * z + z * x = 9) :
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := 
sorry

end inequality_bounds_of_xyz_l78_78516


namespace find_height_on_BC_l78_78896

noncomputable def height_on_BC (a b : ℝ) (A B C : ℝ) : ℝ := b * (Real.sin C)

theorem find_height_on_BC (A B C a b h : ℝ)
  (h_a: a = Real.sqrt 3)
  (h_b: b = Real.sqrt 2)
  (h_cos: 1 + 2 * Real.cos (B + C) = 0)
  (h_A: A = Real.pi / 3)
  (h_B: B = Real.pi / 4)
  (h_C: C = 5 * Real.pi / 12)
  (h_h: h = height_on_BC a b A B C) :
  h = (Real.sqrt 3 + 1) / 2 :=
sorry

end find_height_on_BC_l78_78896


namespace range_of_a_l78_78937

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd and monotonically increasing function, to be defined later.

noncomputable def g (x a : ℝ) : ℝ :=
  f (x^2) + f (a - 2 * |x|)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ g x4 a = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l78_78937


namespace α_plus_2β_eq_pi_div_2_l78_78226

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : 3 * sin α ^ 2 + 2 * sin β ^ 2 = 1
axiom h4 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0

theorem α_plus_2β_eq_pi_div_2 : α + 2 * β = π / 2 :=
by
  sorry

end α_plus_2β_eq_pi_div_2_l78_78226


namespace rhombus_longer_diagonal_l78_78087

theorem rhombus_longer_diagonal 
  (a b : ℝ) 
  (h₁ : a = 61) 
  (h₂ : b = 44) :
  ∃ d₂ : ℝ, d₂ = 2 * Real.sqrt (a * a - (b / 2) * (b / 2)) :=
sorry

end rhombus_longer_diagonal_l78_78087


namespace odd_if_and_only_if_m_even_l78_78997

variables (o n m : ℕ)

theorem odd_if_and_only_if_m_even
  (h_o_odd : o % 2 = 1) :
  ((o^3 + n*o + m) % 2 = 1) ↔ (m % 2 = 0) :=
sorry

end odd_if_and_only_if_m_even_l78_78997


namespace polar_to_cartesian_eq_polar_circle_area_l78_78805

theorem polar_to_cartesian_eq (p θ x y : ℝ) (h : p = 2 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 - 2 * x + y^2 = 0 := sorry

theorem polar_circle_area (p θ : ℝ) (h : p = 2 * Real.cos θ) :
  Real.pi = Real.pi := (by ring)


end polar_to_cartesian_eq_polar_circle_area_l78_78805


namespace sequence_formula_l78_78308

-- Defining the sequence and the conditions
def bounded_seq (a : ℕ → ℝ) : Prop :=
  ∃ C > 0, ∀ n, |a n| ≤ C

-- Statement of the problem in Lean
theorem sequence_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 3 * a n - 4) →
  bounded_seq a →
  ∀ n : ℕ, a n = 2 :=
by
  intros h1 h2
  sorry

end sequence_formula_l78_78308


namespace parabola_intersection_probability_correct_l78_78127

noncomputable def parabola_intersection_probability : ℚ := sorry

theorem parabola_intersection_probability_correct :
  parabola_intersection_probability = 209 / 216 := sorry

end parabola_intersection_probability_correct_l78_78127


namespace simplify_expression_l78_78440

theorem simplify_expression (x : ℝ) :
  4*x^3 + 5*x + 6*x^2 + 10 - (3 - 6*x^2 - 4*x^3 + 2*x) = 8*x^3 + 12*x^2 + 3*x + 7 :=
by
  sorry

end simplify_expression_l78_78440


namespace product_roots_example_l78_78721

def cubic_eq (a b c d : ℝ) (x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

noncomputable def product_of_roots (a b c d : ℝ) : ℝ := -d / a

theorem product_roots_example : product_of_roots 4 (-2) (-25) 36 = -9 := by
  sorry

end product_roots_example_l78_78721


namespace units_digit_of_n_l78_78788

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_of_n 
  (m n : ℕ) 
  (h1 : m * n = 21 ^ 6) 
  (h2 : units_digit m = 7) : 
  units_digit n = 3 := 
sorry

end units_digit_of_n_l78_78788


namespace proposition_D_l78_78808

theorem proposition_D (a b c d : ℝ) (h1 : a < b) (h2 : c < d) : a + c < b + d :=
sorry

end proposition_D_l78_78808


namespace loraine_wax_usage_l78_78020

/-
Loraine makes wax sculptures of animals. Large animals take eight sticks of wax, medium animals take five sticks, and small animals take three sticks.
She made twice as many small animals as large animals, and four times as many medium animals as large animals. She used 36 sticks of wax for small animals.
Prove that Loraine used 204 sticks of wax to make all the animals.
-/

theorem loraine_wax_usage :
  ∃ (L M S : ℕ), (S = 2 * L) ∧ (M = 4 * L) ∧ (3 * S = 36) ∧ (8 * L + 5 * M + 3 * S = 204) :=
by {
  sorry
}

end loraine_wax_usage_l78_78020


namespace inequality_for_a_ne_1_l78_78598

theorem inequality_for_a_ne_1 (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3 * (1 + a^2 + a^4) :=
sorry

end inequality_for_a_ne_1_l78_78598


namespace compute_diameter_of_garden_roller_l78_78882

noncomputable def diameter_of_garden_roller (length : ℝ) (area_per_revolution : ℝ) (pi : ℝ) :=
  let radius := (area_per_revolution / (2 * pi * length))
  2 * radius

theorem compute_diameter_of_garden_roller :
  diameter_of_garden_roller 3 (66 / 5) (22 / 7) = 1.4 := by
  sorry

end compute_diameter_of_garden_roller_l78_78882


namespace find_b_l78_78404

variable {a b d m : ℝ}

theorem find_b (h : m = d * a * b / (a + b)) : b = m * a / (d * a - m) :=
sorry

end find_b_l78_78404


namespace total_prime_ending_starting_numerals_l78_78004

def single_digit_primes : List ℕ := [2, 3, 5, 7]
def number_of_possible_digits := 10

def count_3digit_numerals : ℕ :=
  4 * number_of_possible_digits * 4

def count_4digit_numerals : ℕ :=
  4 * number_of_possible_digits * number_of_possible_digits * 4

theorem total_prime_ending_starting_numerals : 
  count_3digit_numerals + count_4digit_numerals = 1760 := by
sorry

end total_prime_ending_starting_numerals_l78_78004


namespace problem_l78_78104

-- Definitions according to the conditions
def red_balls : ℕ := 1
def black_balls (n : ℕ) : ℕ := n
def total_balls (n : ℕ) : ℕ := red_balls + black_balls n
noncomputable def probability_red (n : ℕ) : ℚ := (red_balls : ℚ) / (total_balls n : ℚ)
noncomputable def variance (n : ℕ) : ℚ := (black_balls n : ℚ) / (total_balls n : ℚ)^2

-- The theorem we want to prove
theorem problem (n : ℕ) (h : 0 < n) : 
  (∀ m : ℕ, n < m → probability_red m < probability_red n) ∧ 
  (∀ m : ℕ, n < m → variance m < variance n) :=
sorry

end problem_l78_78104


namespace license_plate_difference_l78_78873

theorem license_plate_difference :
  (26^3 * 10^4) - (26^4 * 10^3) = -281216000 :=
by
  sorry

end license_plate_difference_l78_78873


namespace probability_other_side_green_l78_78198

-- Definitions based on the conditions
def Card : Type := ℕ
def num_cards : ℕ := 8
def blue_blue : ℕ := 4
def blue_green : ℕ := 2
def green_green : ℕ := 2

def total_green_sides : ℕ := (green_green * 2) + blue_green
def green_opposite_green_side : ℕ := green_green * 2

theorem probability_other_side_green (h_total_green_sides : total_green_sides = 6)
(h_green_opposite_green_side : green_opposite_green_side = 4) :
  (green_opposite_green_side / total_green_sides : ℚ) = 2 / 3 := 
by
  sorry

end probability_other_side_green_l78_78198


namespace parabola_shift_right_by_3_l78_78373

theorem parabola_shift_right_by_3 :
  ∀ (x : ℝ), (∃ y₁ y₂ : ℝ, y₁ = 2 * x^2 ∧ y₂ = 2 * (x - 3)^2) →
  (∃ (h : ℝ), h = 3) :=
sorry

end parabola_shift_right_by_3_l78_78373


namespace max_chords_through_line_l78_78680

noncomputable def maxChords (n : ℕ) : ℕ :=
  let k := n / 2
  k * k + n

theorem max_chords_through_line (points : ℕ) (h : points = 2017) : maxChords 2016 = 1018080 :=
by
  have h1 : (2016 / 2) * (2016 / 2) + 2016 = 1018080 := by norm_num
  rw [← h1]; sorry

end max_chords_through_line_l78_78680


namespace inequality_represents_area_l78_78803

theorem inequality_represents_area (a : ℝ) :
  (if a > 1 then ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y < - (x + 3) / (a - 1)
  else ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y > - (x + 3) / (a - 1)) :=
by sorry

end inequality_represents_area_l78_78803


namespace find_right_triangle_conditions_l78_78339

def is_right_triangle (A B C : ℝ) : Prop := 
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem find_right_triangle_conditions (A B C : ℝ):
  (A + B = C ∧ is_right_triangle A B C) ∨ 
  (A = B ∧ B = 2 * C ∧ is_right_triangle A B C) ∨ 
  (A / 30 = 1 ∧ B / 30 = 2 ∧ C / 30 = 3 ∧ is_right_triangle A B C) :=
sorry

end find_right_triangle_conditions_l78_78339


namespace find_k_l78_78842

theorem find_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 :=
sorry

end find_k_l78_78842


namespace product_eq_1519000000_div_6561_l78_78880

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end product_eq_1519000000_div_6561_l78_78880


namespace minimum_guests_economical_option_l78_78601

theorem minimum_guests_economical_option :
  ∀ (x : ℕ), (150 + 20 * x > 300 + 15 * x) → x > 30 :=
by 
  intro x
  sorry

end minimum_guests_economical_option_l78_78601


namespace no_real_roots_of_quadratic_l78_78469

theorem no_real_roots_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -Real.sqrt 5 → c = Real.sqrt 2 →
  (b^2 - 4 * a * c < 0) → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c ha hb hc hD
  rw [ha, hb, hc] at hD
  sorry

end no_real_roots_of_quadratic_l78_78469


namespace inequality_Cauchy_Schwarz_l78_78491

theorem inequality_Cauchy_Schwarz (a b : ℝ) : 
  (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 :=
by
  sorry

end inequality_Cauchy_Schwarz_l78_78491


namespace maria_should_buy_more_l78_78682

-- Define the conditions as assumptions.
variables (needs total_cartons : ℕ) (strawberries blueberries : ℕ)

-- Specify the given conditions.
def maria_conditions (needs total_cartons strawberries blueberries : ℕ) : Prop :=
  needs = 21 ∧ strawberries = 4 ∧ blueberries = 8 ∧ total_cartons = strawberries + blueberries

-- State the theorem to be proven.
theorem maria_should_buy_more
  (needs total_cartons : ℕ) (strawberries blueberries : ℕ)
  (h : maria_conditions needs total_cartons strawberries blueberries) :
  needs - total_cartons = 9 :=
sorry

end maria_should_buy_more_l78_78682


namespace ratio_65_13_l78_78276

theorem ratio_65_13 : 65 / 13 = 5 := 
by
  sorry

end ratio_65_13_l78_78276


namespace calculate_power_expression_l78_78947

theorem calculate_power_expression : 4 ^ 2009 * (-0.25) ^ 2008 - 1 = 3 := 
by
  -- steps and intermediate calculations go here
  sorry

end calculate_power_expression_l78_78947


namespace derivative_quadrant_l78_78059

theorem derivative_quadrant (b c : ℝ) (H_b : b = -4) : ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 2*x + b = y := by
  sorry

end derivative_quadrant_l78_78059


namespace fraction_addition_l78_78482

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l78_78482


namespace longer_side_length_l78_78025

-- Define the conditions as parameters
variables (W : ℕ) (poles : ℕ) (distance : ℕ) (P : ℕ)

-- Assume the fixed conditions given in the problem
axiom shorter_side : W = 10
axiom number_of_poles : poles = 24
axiom distance_between_poles : distance = 5

-- Define the total perimeter based on the number of segments formed by the poles
noncomputable def perimeter (poles : ℕ) (distance : ℕ) : ℕ :=
  (poles - 4) * distance

-- The total perimeter of the rectangle
axiom total_perimeter : P = perimeter poles distance

-- Definition of the perimeter of the rectangle in terms of its sides
axiom rectangle_perimeter : ∀ (L W : ℕ), P = 2 * L + 2 * W

-- The theorem we need to prove
theorem longer_side_length (L : ℕ) : L = 40 :=
by
  -- Sorry is used to skip the actual proof for now
  sorry

end longer_side_length_l78_78025


namespace ak_not_perfect_square_l78_78109

theorem ak_not_perfect_square (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k1 k2, a k1 = 1988 ∧ b k2 = 1988) :
  ∀ k, ¬ ∃ n, a k = n * n :=
by
  sorry

end ak_not_perfect_square_l78_78109


namespace sum_of_areas_of_circles_l78_78352

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l78_78352


namespace error_percent_in_area_l78_78572

theorem error_percent_in_area
  (L W : ℝ)
  (hL : L > 0)
  (hW : W > 0) :
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 0.8 := by
  sorry

end error_percent_in_area_l78_78572


namespace must_true_l78_78498

axiom p : Prop
axiom q : Prop
axiom h1 : ¬ (p ∧ q)
axiom h2 : p ∨ q

theorem must_true : (¬ p) ∨ (¬ q) := by
  sorry

end must_true_l78_78498


namespace carts_needed_each_day_last_two_days_l78_78247

-- Define capacities as per conditions
def daily_capacity_large_truck : ℚ := 1 / (3 * 4)
def daily_capacity_small_truck : ℚ := 1 / (4 * 5)
def daily_capacity_cart : ℚ := 1 / (20 * 6)

-- Define the number of carts required each day in the last two days
def required_carts_last_two_days : ℚ :=
  let total_work_done_by_large_trucks := 2 * daily_capacity_large_truck * 2
  let total_work_done_by_small_trucks := 3 * daily_capacity_small_truck * 2
  let total_work_done_by_carts := 7 * daily_capacity_cart * 2
  let total_work_done := total_work_done_by_large_trucks + total_work_done_by_small_trucks + total_work_done_by_carts
  let remaining_work := 1 - total_work_done
  remaining_work / (2 * daily_capacity_cart)

-- Assertion of the number of carts required
theorem carts_needed_each_day_last_two_days :
  required_carts_last_two_days = 15 := by
  sorry

end carts_needed_each_day_last_two_days_l78_78247


namespace final_selling_price_l78_78802

-- Define the conditions in Lean
def cost_price_A : ℝ := 150
def profit_A_rate : ℝ := 0.20
def profit_B_rate : ℝ := 0.25

-- Define the function to calculate selling price based on cost price and profit rate
def selling_price (cost_price : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_price + (profit_rate * cost_price)

-- The theorem to be proved
theorem final_selling_price :
  selling_price (selling_price cost_price_A profit_A_rate) profit_B_rate = 225 :=
by
  -- The proof is omitted
  sorry

end final_selling_price_l78_78802


namespace eval_expr1_eval_expr2_l78_78168

theorem eval_expr1 : (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
by
  -- proof goes here
  sorry

theorem eval_expr2 : (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) / (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180))) = Real.sqrt 2 :=
by
  -- proof goes here
  sorry

end eval_expr1_eval_expr2_l78_78168


namespace measure_of_alpha_l78_78008

theorem measure_of_alpha
  (A B D α : ℝ)
  (hA : A = 50)
  (hB : B = 150)
  (hD : D = 140)
  (quadrilateral_sum : A + B + D + α = 360) : α = 20 :=
by
  rw [hA, hB, hD] at quadrilateral_sum
  sorry

end measure_of_alpha_l78_78008


namespace num_real_values_for_integer_roots_l78_78983

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l78_78983


namespace probability_comparison_l78_78901

variables (M N : ℕ) (m n : ℝ)
variable (h₁ : m > 10^6)
variable (h₂ : n ≤ 10^6)

theorem probability_comparison (h₃: 0 < M) (h₄: 0 < N):
  (m * M) / (m * M + n * N) > (M / (M + N)) :=
by
  have h₅: n / m < 1 := sorry
  have h₆: M > 0 := by linarith
  have h₇: 1 + (n / m) * (N / M) < 2 := sorry
  sorry

end probability_comparison_l78_78901


namespace mass_percentage_of_Br_in_BaBr2_l78_78536

theorem mass_percentage_of_Br_in_BaBr2 :
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  mass_percentage_Br = 53.80 :=
by
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  sorry

end mass_percentage_of_Br_in_BaBr2_l78_78536


namespace slope_of_line_joining_solutions_l78_78704

theorem slope_of_line_joining_solutions (x1 x2 y1 y2 : ℝ) :
  (4 / x1 + 5 / y1 = 1) → (4 / x2 + 5 / y2 = 1) →
  (x1 ≠ x2) → (y1 = 5 * x1 / (4 * x1 - 1)) → (y2 = 5 * x2 / (4 * x2 - 1)) →
  (x1 ≠ 1 / 4) → (x2 ≠ 1 / 4) →
  ((y2 - y1) / (x2 - x1) = - (5 / 21)) :=
by
  intros h_eq1 h_eq2 h_neq h_y1 h_y2 h_x1 h_x2
  -- Proof omitted for brevity
  sorry

end slope_of_line_joining_solutions_l78_78704


namespace product_of_consecutive_even_numbers_divisible_by_8_l78_78105

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 
  8 ∣ (2 * n) * (2 * n + 2) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l78_78105


namespace distance_from_neg2_l78_78099

theorem distance_from_neg2 (x : ℝ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 := 
by sorry

end distance_from_neg2_l78_78099


namespace goods_train_speed_l78_78607

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed_l78_78607


namespace correct_option_is_c_l78_78528

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c_l78_78528


namespace units_digit_of_k_squared_plus_2_to_the_k_l78_78006

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k_l78_78006


namespace Megan_acorns_now_l78_78942

def initial_acorns := 16
def given_away_acorns := 7
def remaining_acorns := initial_acorns - given_away_acorns

theorem Megan_acorns_now : remaining_acorns = 9 := by
  sorry

end Megan_acorns_now_l78_78942


namespace points_on_decreasing_line_y1_gt_y2_l78_78050
-- Import the necessary library

-- Necessary conditions and definitions
variable {x y : ℝ}

-- Given points P(3, y1) and Q(4, y2)
def y1 : ℝ := -2*3 + 4
def y2 : ℝ := -2*4 + 4

-- Lean statement to prove y1 > y2
theorem points_on_decreasing_line_y1_gt_y2 (h1 : y1 = -2 * 3 +4) (h2 : y2 = -2 * 4 + 4) : 
  y1 > y2 :=
sorry  -- Proof steps go here

end points_on_decreasing_line_y1_gt_y2_l78_78050


namespace print_shop_X_charge_l78_78027

-- Define the given conditions
def cost_per_copy_X (x : ℝ) : Prop := x > 0
def cost_per_copy_Y : ℝ := 2.75
def total_copies : ℕ := 40
def extra_cost_Y : ℝ := 60

-- Define the main problem
theorem print_shop_X_charge (x : ℝ) (h : cost_per_copy_X x) :
  total_copies * cost_per_copy_Y = total_copies * x + extra_cost_Y → x = 1.25 :=
by
  sorry

end print_shop_X_charge_l78_78027


namespace total_cost_of_groceries_l78_78561

noncomputable def M (R : ℝ) : ℝ := 24 * R / 10
noncomputable def F : ℝ := 22

theorem total_cost_of_groceries (R : ℝ) (hR : 2 * R = 22) :
  10 * M R = 24 * R ∧ F = 2 * R ∧ F = 22 →
  4 * M R + 3 * R + 5 * F = 248.6 := by
  sorry

end total_cost_of_groceries_l78_78561


namespace range_of_m_l78_78736

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 + m * x + 1 = 0 → x ≠ 0) ∧ ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l78_78736


namespace integer_sequence_perfect_square_l78_78082

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ a 2 = 4 ∧ ∀ n ≥ 2, a n = (a (n - 1) * a (n + 1) + 1) ^ (1 / 2)

theorem integer_sequence {a : ℕ → ℝ} : 
  seq a → ∀ n, ∃ k : ℤ, a n = k := 
by sorry

theorem perfect_square {a : ℕ → ℝ} :
  seq a → ∀ n, ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k ^ 2 :=
by sorry

end integer_sequence_perfect_square_l78_78082


namespace function_relation_l78_78941

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem function_relation:
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by 
  sorry

end function_relation_l78_78941


namespace simplify_fraction_l78_78205

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) :
  (x - 1/y) / (y - 1/x) = x / y :=
sorry

end simplify_fraction_l78_78205


namespace taco_beef_per_taco_l78_78369

open Real

theorem taco_beef_per_taco
  (total_beef : ℝ)
  (sell_price : ℝ)
  (cost_per_taco : ℝ)
  (profit : ℝ)
  (h1 : total_beef = 100)
  (h2 : sell_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : profit = 200) :
  ∃ (x : ℝ), x = 1/4 := 
by
  -- The proof will go here.
  sorry

end taco_beef_per_taco_l78_78369


namespace smallest_four_digit_number_divisible_by_smallest_primes_l78_78853

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end smallest_four_digit_number_divisible_by_smallest_primes_l78_78853


namespace sixth_number_of_11_consecutive_odd_sum_1991_is_181_l78_78723

theorem sixth_number_of_11_consecutive_odd_sum_1991_is_181 :
  (∃ (n : ℤ), (2 * n + 1) + (2 * n + 3) + (2 * n + 5) + (2 * n + 7) + (2 * n + 9) + (2 * n + 11) + (2 * n + 13) + (2 * n + 15) + (2 * n + 17) + (2 * n + 19) + (2 * n + 21) = 1991) →
  2 * 85 + 11 = 181 := 
by
  sorry

end sixth_number_of_11_consecutive_odd_sum_1991_is_181_l78_78723


namespace percent_y_of_x_l78_78917

theorem percent_y_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y / x = 1 / 3 :=
by
  -- proof steps would be provided here
  sorry

end percent_y_of_x_l78_78917


namespace solve_equation_l78_78398

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end solve_equation_l78_78398


namespace baking_time_one_batch_l78_78930

theorem baking_time_one_batch (x : ℕ) (time_icing_per_batch : ℕ) (num_batches : ℕ) (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : time_icing_per_batch = 30)
  (h3 : total_time = 200)
  (h4 : total_time = num_batches * x + num_batches * time_icing_per_batch) :
  x = 20 :=
by
  rw [h1, h2, h3] at h4
  sorry

end baking_time_one_batch_l78_78930


namespace hyperbola_equation_l78_78062

theorem hyperbola_equation 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_gt : a > b)
  (parallel_asymptote : ∃ k : ℝ, k = 2)
  (focus_on_line : ∃ cₓ : ℝ, ∃ c : ℝ, c = 5 ∧ cₓ = -5 ∧ (y = -2 * cₓ - 10)) :
  ∃ (a b : ℝ), (a^2 = 5) ∧ (b^2 = 20) ∧ (a^2 > b^2) ∧ c = 5 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 5 - y^2 / 20 = 1)) :=
sorry

end hyperbola_equation_l78_78062


namespace initial_average_l78_78317

theorem initial_average (A : ℝ) (h : (15 * A + 14 * 15) / 15 = 54) : A = 40 :=
by
  sorry

end initial_average_l78_78317


namespace prob_both_successful_prob_at_least_one_successful_l78_78696

variables (P_A P_B : ℚ)
variables (h1 : P_A = 1 / 2)
variables (h2 : P_B = 2 / 5)

/-- Prove that the probability that both A and B score in one shot each is 1 / 5. -/
theorem prob_both_successful (P_A P_B : ℚ) (h1 : P_A = 1 / 2) (h2 : P_B = 2 / 5) :
  P_A * P_B = 1 / 5 :=
by sorry

variables (P_A_miss P_B_miss : ℚ)
variables (h3 : P_A_miss = 1 / 2)
variables (h4 : P_B_miss = 3 / 5)

/-- Prove that the probability that at least one shot is successful is 7 / 10. -/
theorem prob_at_least_one_successful (P_A_miss P_B_miss : ℚ) (h3 : P_A_miss = 1 / 2) (h4 : P_B_miss = 3 / 5) :
  1 - P_A_miss * P_B_miss = 7 / 10 :=
by sorry

end prob_both_successful_prob_at_least_one_successful_l78_78696


namespace trip_time_l78_78928

theorem trip_time (T : ℝ) (x : ℝ) : 
  (150 / 4 = 50 / 30 + (x - 50) / 4 + (150 - x) / 30) → (T = 37.5) :=
by
  sorry

end trip_time_l78_78928


namespace neg_mod_eq_1998_l78_78624

theorem neg_mod_eq_1998 {a : ℤ} (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end neg_mod_eq_1998_l78_78624


namespace inequality_inequation_l78_78829

theorem inequality_inequation (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x + y + z = 1) :
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 :=
by
  sorry

end inequality_inequation_l78_78829


namespace smallest_x_remainder_l78_78519

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l78_78519


namespace min_value_of_n_l78_78355

theorem min_value_of_n :
  ∀ (h : ℝ), ∃ n : ℝ, (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -x^2 + 2 * h * x - h ≤ n) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -x^2 + 2 * h * x - h = n) ∧
  n = -1 / 4 := 
by
  sorry

end min_value_of_n_l78_78355


namespace find_larger_number_l78_78256

-- Definitions based on the conditions
variables (x y : ℕ)

-- Main theorem
theorem find_larger_number (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 :=
by
  sorry

end find_larger_number_l78_78256


namespace problem_statement_l78_78378

theorem problem_statement (p q : ℝ)
  (α β : ℝ) (h1 : α ≠ β) (h1' : α + β = -p) (h1'' : α * β = -2)
  (γ δ : ℝ) (h2 : γ ≠ δ) (h2' : γ + δ = -q) (h2'' : γ * δ = -3) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2) - 2 * q + 1 := by
  sorry

end problem_statement_l78_78378


namespace calculate_expression_l78_78235

theorem calculate_expression : 
  (1007^2 - 995^2 - 1005^2 + 997^2) = 8008 := 
by {
  sorry
}

end calculate_expression_l78_78235


namespace grandfather_7_times_older_after_8_years_l78_78840

theorem grandfather_7_times_older_after_8_years :
  ∃ x : ℕ, ∀ (g_age ng_age : ℕ), 50 < g_age ∧ g_age < 90 ∧ g_age = 31 * ng_age → g_age + x = 7 * (ng_age + x) → x = 8 :=
by
  sorry

end grandfather_7_times_older_after_8_years_l78_78840


namespace count_three_digit_distinct_under_800_l78_78068

-- Definitions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 800
def distinct_digits (n : ℕ) : Prop := (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) 

-- Theorem
theorem count_three_digit_distinct_under_800 : ∃ k : ℕ, k = 504 ∧ ∀ n : ℕ, is_three_digit n → distinct_digits n → n < 800 :=
by 
  exists 504
  sorry

end count_three_digit_distinct_under_800_l78_78068


namespace min_value_of_a_plus_b_minus_c_l78_78870

open Real

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ) :
  (∀ (x y : ℝ), 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) →
  (∃ c_min, c_min = 2 ∧ ∀ c', c' = a + b - c → c' ≥ c_min) :=
by
  sorry

end min_value_of_a_plus_b_minus_c_l78_78870


namespace t_minus_d_l78_78242

-- Define amounts paid by Tom, Dorothy, and Sammy
def tom_paid : ℕ := 140
def dorothy_paid : ℕ := 90
def sammy_paid : ℕ := 220

-- Define the total amount and required equal share
def total_paid : ℕ := tom_paid + dorothy_paid + sammy_paid
def equal_share : ℕ := total_paid / 3

-- Define the amounts t and d where Tom and Dorothy balance the costs by paying Sammy
def t : ℤ := equal_share - tom_paid -- Amount Tom gave to Sammy
def d : ℤ := equal_share - dorothy_paid -- Amount Dorothy gave to Sammy

-- Prove that t - d = -50
theorem t_minus_d : t - d = -50 := by
  sorry

end t_minus_d_l78_78242


namespace mean_height_calc_l78_78464

/-- Heights of players on the soccer team -/
def heights : List ℕ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

/-- Total number of players -/
def total_players : ℕ := heights.length

/-- Sum of heights of players -/
def sum_heights : ℕ := heights.sum

/-- Mean height of players on the soccer team -/
def mean_height : ℚ := sum_heights / total_players

/-- Proof that the mean height is correct -/
theorem mean_height_calc : mean_height = 56.25 := by
  sorry

end mean_height_calc_l78_78464


namespace dogs_not_liking_any_food_l78_78169

-- Declare variables
variable (n w s ws c cs : ℕ)

-- Define problem conditions
def total_dogs := n
def dogs_like_watermelon := w
def dogs_like_salmon := s
def dogs_like_watermelon_and_salmon := ws
def dogs_like_chicken := c
def dogs_like_chicken_and_salmon_but_not_watermelon := cs

-- Define the statement proving the number of dogs that do not like any of the three foods
theorem dogs_not_liking_any_food : 
  n = 75 → 
  w = 15 → 
  s = 54 → 
  ws = 12 → 
  c = 20 → 
  cs = 7 → 
  (75 - ((w - ws) + (s - ws - cs) + (c - cs) + ws + cs) = 5) :=
by
  intros _ _ _ _ _ _
  sorry

end dogs_not_liking_any_food_l78_78169


namespace inequality_solution_l78_78759

theorem inequality_solution (x : ℝ) :
  (3 / 16) + abs (x - 17 / 64) < 7 / 32 ↔ (15 / 64) < x ∧ x < (19 / 64) :=
by
  sorry

end inequality_solution_l78_78759


namespace solution_exists_solution_unique_l78_78326

noncomputable def abc_solutions : Finset (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (2, 2, 4), (2, 4, 8), (3, 5, 15), 
   (2, 4, 2), (4, 2, 2), (4, 2, 8), (8, 4, 2), 
   (2, 8, 4), (8, 2, 4), (5, 3, 15), (15, 3, 5), (3, 15, 5),
   (2, 2, 4), (4, 2, 2), (4, 8, 2)}

theorem solution_exists (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a * b * c - 1 = (a - 1) * (b - 1) * (c - 1)) ↔ (a, b, c) ∈ abc_solutions := 
by
  sorry

theorem solution_unique (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a, b, c) ∈ abc_solutions → a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) :=
by
  sorry

end solution_exists_solution_unique_l78_78326


namespace evaluated_result_l78_78633

noncomputable def evaluate_expression (y : ℝ) (hy : y ≠ 0) : ℝ :=
  (18 * y^3) * (4 * y^2) * (1 / (2 * y)^3)

theorem evaluated_result (y : ℝ) (hy : y ≠ 0) : evaluate_expression y hy = 9 * y^2 :=
by
  sorry

end evaluated_result_l78_78633


namespace train_crossing_time_l78_78301

-- Define the length of the train
def train_length : ℝ := 120

-- Define the speed of the train
def train_speed : ℝ := 15

-- Define the target time to cross the man
def target_time : ℝ := 8

-- Proposition to prove
theorem train_crossing_time :
  target_time = train_length / train_speed :=
by
  sorry

end train_crossing_time_l78_78301


namespace length_of_AB_l78_78370

theorem length_of_AB 
  (P Q A B : ℝ)
  (h_P_on_AB : P > 0 ∧ P < B)
  (h_Q_on_AB : Q > P ∧ Q < B)
  (h_ratio_P : P = 3 / 7 * B)
  (h_ratio_Q : Q = 4 / 9 * B)
  (h_PQ : Q - P = 3) 
: B = 189 := 
sorry

end length_of_AB_l78_78370


namespace parallel_vectors_l78_78097

variable {k m : ℝ}

theorem parallel_vectors (h₁ : (2 : ℝ) = k * m) (h₂ : m = 2 * k) : m = 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l78_78097


namespace another_seat_in_sample_l78_78445

-- Definition of the problem
def total_students := 56
def sample_size := 4
def sample_set : Finset ℕ := {3, 17, 45}

-- Lean 4 statement for the proof problem
theorem another_seat_in_sample :
  (sample_set = sample_set ∪ {31}) ∧
  (31 ∉ sample_set) ∧
  (∀ x ∈ sample_set ∪ {31}, x ≤ total_students) :=
by
  sorry

end another_seat_in_sample_l78_78445


namespace paint_cost_per_quart_l78_78580

-- Definitions of conditions
def edge_length (cube_edge_length : ℝ) : Prop := cube_edge_length = 10
def surface_area (s_area : ℝ) : Prop := s_area = 6 * (10^2)
def coverage_per_quart (coverage : ℝ) : Prop := coverage = 120
def total_cost (cost : ℝ) : Prop := cost = 16
def required_quarts (quarts : ℝ) : Prop := quarts = 600 / 120
def cost_per_quart (cost : ℝ) (quarts : ℝ) (price_per_quart : ℝ) : Prop := price_per_quart = cost / quarts

-- Main theorem statement translating the problem into Lean
theorem paint_cost_per_quart {cube_edge_length s_area coverage cost quarts price_per_quart : ℝ} :
  edge_length cube_edge_length →
  surface_area s_area →
  coverage_per_quart coverage →
  total_cost cost →
  required_quarts quarts →
  quarts = s_area / coverage →
  cost_per_quart cost quarts 3.20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- proof will go here
  sorry

end paint_cost_per_quart_l78_78580


namespace solve_for_z_l78_78337

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l78_78337


namespace find_k_l78_78586

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x = -2 ∧ x^2 - k * x + 2 = 0) : k = -3 := by
  sorry

end find_k_l78_78586


namespace average_of_original_set_l78_78859

theorem average_of_original_set
  (A : ℝ)
  (n : ℕ)
  (B : ℝ)
  (h1 : n = 7)
  (h2 : B = 5 * A)
  (h3 : B / n = 100)
  : A = 20 :=
by
  sorry

end average_of_original_set_l78_78859


namespace base_conversion_and_addition_l78_78295

def C : ℕ := 12

def base9_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 9^2) + (d1 * 9^1) + (d0 * 9^0)

def base13_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 13^2) + (d1 * 13^1) + (d0 * 13^0)

def num1 := base9_to_nat 7 5 2
def num2 := base13_to_nat 6 C 3

theorem base_conversion_and_addition :
  num1 + num2 = 1787 :=
by
  sorry

end base_conversion_and_addition_l78_78295


namespace construction_company_doors_needed_l78_78501

-- Definitions based on conditions
def num_floors_per_building : ℕ := 20
def num_apartments_per_floor : ℕ := 8
def num_buildings : ℕ := 3
def num_doors_per_apartment : ℕ := 10

-- Total number of apartments
def total_apartments : ℕ :=
  num_floors_per_building * num_apartments_per_floor * num_buildings

-- Total number of doors
def total_doors_needed : ℕ :=
  num_doors_per_apartment * total_apartments

-- Theorem statement to prove the number of doors needed
theorem construction_company_doors_needed :
  total_doors_needed = 4800 :=
sorry

end construction_company_doors_needed_l78_78501


namespace machine_present_value_l78_78101

theorem machine_present_value
  (rate_of_decay : ℝ) (n_periods : ℕ) (final_value : ℝ) (initial_value : ℝ)
  (h_decay : rate_of_decay = 0.25)
  (h_periods : n_periods = 2)
  (h_final_value : final_value = 225) :
  initial_value = 400 :=
by
  -- The proof would go here. 
  sorry

end machine_present_value_l78_78101


namespace find_h_from_quadratic_l78_78915

theorem find_h_from_quadratic (
  p q r : ℝ) (h₁ : ∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) :
  ∀ m k h, (∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - h)^2 + k) → h = 5 :=
by
  intros m k h h₂
  sorry

end find_h_from_quadratic_l78_78915


namespace complementary_angle_difference_l78_78115

theorem complementary_angle_difference (x : ℝ) (h : 3 * x + 5 * x = 90) : 
    abs ((5 * x) - (3 * x)) = 22.5 :=
by
  -- placeholder proof
  sorry

end complementary_angle_difference_l78_78115


namespace min_n_for_constant_term_l78_78478

theorem min_n_for_constant_term (n : ℕ) (h : 0 < n) : 
  (∃ (r : ℕ), 0 = n - 4 * r / 3) → n = 4 :=
by
  sorry

end min_n_for_constant_term_l78_78478


namespace smallest_prime_12_less_than_square_l78_78381

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l78_78381


namespace total_value_of_coins_l78_78893

variables {p n : ℕ}

-- Ryan has 17 coins consisting of pennies and nickels
axiom coins_eq : p + n = 17

-- The number of pennies is equal to the number of nickels
axiom pennies_eq_nickels : p = n

-- Prove that the total value of Ryan's coins is 49 cents
theorem total_value_of_coins : (p * 1 + n * 5) = 49 :=
by sorry

end total_value_of_coins_l78_78893


namespace coin_toss_sequences_count_l78_78701

theorem coin_toss_sequences_count :
  (∃ (seq : List Char), 
    seq.length = 15 ∧ 
    (seq == ['H', 'H']) = 5 ∧ 
    (seq == ['H', 'T']) = 3 ∧ 
    (seq == ['T', 'H']) = 2 ∧ 
    (seq == ['T', 'T']) = 4) → 
  (count_sequences == 775360) :=
by
  sorry

end coin_toss_sequences_count_l78_78701


namespace extreme_value_at_one_l78_78862

noncomputable def f (x : ℝ) (a : ℝ) := (x^2 + a) / (x + 1)

theorem extreme_value_at_one (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y-1) < δ → abs (f y a - f 1 a) < ε)) →
  a = 3 :=
by
  sorry

end extreme_value_at_one_l78_78862


namespace mean_age_of_euler_family_children_l78_78222

noncomputable def euler_family_children_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

theorem mean_age_of_euler_family_children : 
  (List.sum euler_family_children_ages : ℚ) / (List.length euler_family_children_ages) = 96 / 7 := 
by
  sorry

end mean_age_of_euler_family_children_l78_78222


namespace science_students_count_l78_78751

def total_students := 400 + 120
def local_arts_students := 0.50 * 400
def local_commerce_students := 0.85 * 120
def total_local_students := 327

theorem science_students_count :
  0.25 * S = 25 →
  S = 100 :=
by
  sorry

end science_students_count_l78_78751


namespace problem_solution_l78_78085

theorem problem_solution (x : ℝ) (h : x - 29 = 63) : (x - 47 = 45) :=
by
  sorry

end problem_solution_l78_78085


namespace newly_grown_uneaten_potatoes_l78_78178

variable (u : ℕ)

def initially_planted : ℕ := 8
def total_now : ℕ := 11

theorem newly_grown_uneaten_potatoes : u = total_now - initially_planted := by
  sorry

end newly_grown_uneaten_potatoes_l78_78178


namespace total_whales_seen_is_178_l78_78070

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end total_whales_seen_is_178_l78_78070


namespace notebooks_if_students_halved_l78_78532

-- Definitions based on the problem conditions
def totalNotebooks: ℕ := 512
def notebooksPerStudent (students: ℕ) : ℕ := students / 8
def notebooksWhenStudentsHalved (students notebooks: ℕ) : ℕ := notebooks / (students / 2)

-- Theorem statement
theorem notebooks_if_students_halved (S : ℕ) (h : S * (S / 8) = totalNotebooks) :
    notebooksWhenStudentsHalved S totalNotebooks = 16 :=
by
  sorry

end notebooks_if_students_halved_l78_78532


namespace percent_of_part_l78_78230

variable (Part : ℕ) (Whole : ℕ)

theorem percent_of_part (hPart : Part = 70) (hWhole : Whole = 280) :
  (Part / Whole) * 100 = 25 := by
  sorry

end percent_of_part_l78_78230


namespace johnny_tables_l78_78311

theorem johnny_tables :
  ∀ (T : ℕ),
  (∀ (T : ℕ), 4 * T + 5 * T = 45) →
  T = 5 :=
  sorry

end johnny_tables_l78_78311


namespace find_angle_A_find_area_l78_78630

-- Define the geometric and trigonometric conditions of the triangle
def triangle (A B C a b c : ℝ) :=
  a = 4 * Real.sqrt 3 ∧ b + c = 8 ∧
  2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Prove angle A is 60 degrees
theorem find_angle_A (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : A = Real.pi / 3 := sorry

-- Prove the area of triangle ABC is 4 * sqrt(3) / 3
theorem find_area (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : 
  (1 / 2) * (a * b * Real.sin C) = (4 * Real.sqrt 3) / 3 := sorry

end find_angle_A_find_area_l78_78630


namespace product_of_numbers_l78_78995

theorem product_of_numbers (x y : ℤ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end product_of_numbers_l78_78995


namespace find_g_2_l78_78278

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 2 * g x - 3 * g (1 / x) = x ^ 2

theorem find_g_2 : g 2 = 8.25 :=
by {
  sorry
}

end find_g_2_l78_78278


namespace range_of_a_minus_abs_b_l78_78030

theorem range_of_a_minus_abs_b (a b : ℝ) (h1: 1 < a) (h2: a < 3) (h3: -4 < b) (h4: b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l78_78030


namespace multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l78_78609

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l78_78609


namespace find_P_l78_78897

theorem find_P (P Q R S : ℕ) (h1 : P ≠ Q) (h2 : P ≠ R) (h3 : P ≠ S) (h4 : Q ≠ R) (h5 : Q ≠ S) (h6 : R ≠ S)
  (h7 : P > 0) (h8 : Q > 0) (h9 : R > 0) (h10 : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDiff : P - Q = R + S) : P = 12 :=
by
  sorry

end find_P_l78_78897


namespace cone_height_l78_78756

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end cone_height_l78_78756


namespace distance_between_lines_l78_78164

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines : 
  ∀ (x y : ℝ), line1 x y → line2 x y → (|1 - (-1)| / Real.sqrt (1^2 + (-1)^2)) = Real.sqrt 2 := 
by 
  sorry

end distance_between_lines_l78_78164


namespace number_of_solutions_l78_78424

theorem number_of_solutions (n : ℕ) : (4 * n) = 80 ↔ n = 20 :=
by
  sorry

end number_of_solutions_l78_78424


namespace Peter_initially_had_33_marbles_l78_78077

-- Definitions based on conditions
def lostMarbles : Nat := 15
def currentMarbles : Nat := 18

-- Definition for the initial marbles calculation
def initialMarbles (lostMarbles : Nat) (currentMarbles : Nat) : Nat :=
  lostMarbles + currentMarbles

-- Theorem statement
theorem Peter_initially_had_33_marbles : initialMarbles lostMarbles currentMarbles = 33 := by
  sorry

end Peter_initially_had_33_marbles_l78_78077


namespace dilation_transformation_result_l78_78651

theorem dilation_transformation_result
  (x y x' y' : ℝ)
  (h₀ : x'^2 / 4 + y'^2 / 9 = 1) 
  (h₁ : x' = 2 * x)
  (h₂ : y' = 3 * y)
  (h₃ : x^2 + y^2 = 1)
  : x'^2 / 4 + y'^2 / 9 = 1 := 
by
  sorry

end dilation_transformation_result_l78_78651


namespace value_of_expression_l78_78065

theorem value_of_expression (x : ℝ) (h : x^2 - 5 * x + 6 < 0) : x^2 - 5 * x + 10 = 4 :=
sorry

end value_of_expression_l78_78065


namespace find_range_of_a_l78_78157

noncomputable def range_of_a (a : ℝ) : Prop :=
∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ (Real.pi / 2)) → 
  let α := (x + 3, x)
  let β := (2 * Real.sin θ * Real.cos θ, a * Real.sin θ + a * Real.cos θ)
  let sum := (α.1 + β.1, α.2 + β.2)
  (sum.1^2 + sum.2^2)^(1/2) ≥ Real.sqrt 2

theorem find_range_of_a : range_of_a a ↔ (a ≤ 1 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l78_78157


namespace difference_between_sums_l78_78471

open Nat

-- Sum of the first 'n' positive odd integers formula: n^2
def sum_of_first_odd (n : ℕ) : ℕ := n * n

-- Sum of the first 'n' positive even integers formula: n(n+1)
def sum_of_first_even (n : ℕ) : ℕ := n * (n + 1)

-- The main theorem stating the difference between the sums
theorem difference_between_sums (n : ℕ) (h : n = 3005) :
  sum_of_first_even n - sum_of_first_odd n = 3005 :=
by
  sorry

end difference_between_sums_l78_78471


namespace vera_operations_impossible_l78_78413

theorem vera_operations_impossible (N : ℕ) : (N % 3 ≠ 0) → ¬(∃ k : ℕ, ((N + 3 * k) % 5 = 0) → ((N + 3 * k) / 5) = 1) :=
by
  sorry

end vera_operations_impossible_l78_78413


namespace overall_average_of_25_results_l78_78468

theorem overall_average_of_25_results (first_12_avg last_12_avg thirteenth_result : ℝ) 
  (h1 : first_12_avg = 14) (h2 : last_12_avg = 17) (h3 : thirteenth_result = 78) :
  (12 * first_12_avg + thirteenth_result + 12 * last_12_avg) / 25 = 18 :=
by
  sorry

end overall_average_of_25_results_l78_78468


namespace John_completes_work_alone_10_days_l78_78994

theorem John_completes_work_alone_10_days
  (R : ℕ)
  (T : ℕ)
  (W : ℕ)
  (H1 : R = 40)
  (H2 : T = 8)
  (H3 : 1/10 = (1/R) + (1/W))
  : W = 10 := sorry

end John_completes_work_alone_10_days_l78_78994


namespace minimum_bail_rate_l78_78505

theorem minimum_bail_rate
  (distance : ℝ) (leak_rate : ℝ) (rain_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) (time_in_minutes : ℝ) (bail_rate : ℝ) : 
  (distance = 2) → 
  (leak_rate = 15) → 
  (rain_rate = 5) →
  (sink_threshold = 60) →
  (rowing_speed = 3) →
  (time_in_minutes = (2 / 3) * 60) →
  (bail_rate = sink_threshold / (time_in_minutes) - (rain_rate + leak_rate)) →
  bail_rate ≥ 18.5 :=
by
  intros h_distance h_leak_rate h_rain_rate h_sink_threshold h_rowing_speed h_time_in_minutes h_bail_rate
  sorry

end minimum_bail_rate_l78_78505


namespace volume_difference_is_867_25_l78_78231

noncomputable def charlie_volume : ℝ :=
  let h_C := 9
  let circumference_C := 7
  let r_C := circumference_C / (2 * Real.pi)
  let v_C := Real.pi * r_C^2 * h_C
  v_C

noncomputable def dana_volume : ℝ :=
  let h_D := 5
  let circumference_D := 10
  let r_D := circumference_D / (2 * Real.pi)
  let v_D := Real.pi * r_D^2 * h_D
  v_D

noncomputable def volume_difference : ℝ :=
  Real.pi * (abs (charlie_volume - dana_volume))

theorem volume_difference_is_867_25 : volume_difference = 867.25 := by
  sorry

end volume_difference_is_867_25_l78_78231


namespace arithmetic_seq_a5_value_l78_78658

theorem arithmetic_seq_a5_value (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := 
sorry

end arithmetic_seq_a5_value_l78_78658


namespace cost_of_one_package_of_berries_l78_78459

noncomputable def martin_daily_consumption : ℚ := 1 / 2

noncomputable def package_content : ℚ := 1

noncomputable def total_period_days : ℚ := 30

noncomputable def total_spent : ℚ := 30

theorem cost_of_one_package_of_berries :
  (total_spent / (total_period_days * martin_daily_consumption / package_content)) = 2 :=
sorry

end cost_of_one_package_of_berries_l78_78459


namespace equation_of_directrix_l78_78294

theorem equation_of_directrix (x y : ℝ) (h : y^2 = 2 * x) : 
  x = - (1/2) :=
sorry

end equation_of_directrix_l78_78294


namespace expansion_of_binomials_l78_78102

theorem expansion_of_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 :=
  sorry

end expansion_of_binomials_l78_78102


namespace convert_spherical_to_cartesian_l78_78283

theorem convert_spherical_to_cartesian :
  let ρ := 5
  let θ₁ := 3 * Real.pi / 4
  let φ₁ := 9 * Real.pi / 5
  let φ' := 2 * Real.pi - φ₁
  let θ' := θ₁ + Real.pi
  ∃ (θ : ℝ) (φ : ℝ),
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    (∃ (x y z : ℝ),
      x = ρ * Real.sin φ' * Real.cos θ' ∧
      y = ρ * Real.sin φ' * Real.sin θ' ∧
      z = ρ * Real.cos φ') ∧
    θ = θ' ∧ φ = φ' :=
by
  sorry

end convert_spherical_to_cartesian_l78_78283


namespace rebecca_has_more_eggs_than_marbles_l78_78773

-- Given conditions
def eggs : Int := 20
def marbles : Int := 6

-- Mathematically equivalent statement to prove
theorem rebecca_has_more_eggs_than_marbles :
    eggs - marbles = 14 :=
by
    sorry

end rebecca_has_more_eggs_than_marbles_l78_78773


namespace first_recipe_cups_l78_78439

-- Definitions based on the given conditions
def ounces_per_bottle : ℕ := 16
def ounces_per_cup : ℕ := 8
def cups_second_recipe : ℕ := 1
def cups_third_recipe : ℕ := 3
def total_bottles : ℕ := 3
def total_ounces : ℕ := total_bottles * ounces_per_bottle
def total_cups_needed : ℕ := total_ounces / ounces_per_cup

-- Proving the amount of cups of soy sauce needed for the first recipe
theorem first_recipe_cups : 
  total_cups_needed - (cups_second_recipe + cups_third_recipe) = 2 
:= by 
-- Proof omitted
  sorry

end first_recipe_cups_l78_78439


namespace bus_speed_with_stoppages_l78_78238

theorem bus_speed_with_stoppages :
  ∀ (speed_excluding_stoppages : ℕ) (stop_minutes : ℕ) (total_minutes : ℕ)
  (speed_including_stoppages : ℕ),
  speed_excluding_stoppages = 80 →
  stop_minutes = 15 →
  total_minutes = 60 →
  speed_including_stoppages = (speed_excluding_stoppages * (total_minutes - stop_minutes) / total_minutes) →
  speed_including_stoppages = 60 := by
  sorry

end bus_speed_with_stoppages_l78_78238


namespace smallest_sum_p_q_l78_78506

theorem smallest_sum_p_q (p q : ℕ) (h_pos : 1 < p) (h_cond : (p^2 * q - 1) = (2021 * p * q) / 2021) : p + q = 44 :=
sorry

end smallest_sum_p_q_l78_78506


namespace probability_of_event_l78_78195

noncomputable def drawing_probability : ℚ := 
  let total_outcomes := 81
  let successful_outcomes :=
    (9 + 9 + 9 + 9 + 9 + 7 + 5 + 3 + 1)
  successful_outcomes / total_outcomes

theorem probability_of_event :
  drawing_probability = 61 / 81 := 
by
  sorry

end probability_of_event_l78_78195


namespace train_length_l78_78113

theorem train_length (x : ℕ) (h1 : (310 + x) / 18 = x / 8) : x = 248 :=
  sorry

end train_length_l78_78113


namespace hyperbola_condition_sufficiency_l78_78568

theorem hyperbola_condition_sufficiency (k : ℝ) :
  (k > 3) → (∃ x y : ℝ, (x^2)/(3-k) + (y^2)/(k-1) = 1) :=
by
  sorry

end hyperbola_condition_sufficiency_l78_78568


namespace john_weekly_loss_is_525000_l78_78494

-- Define the constants given in the problem
def daily_production : ℕ := 1000
def production_cost_per_tire : ℝ := 250
def selling_price_factor : ℝ := 1.5
def potential_daily_sales : ℕ := 1200
def days_in_week : ℕ := 7

-- Define the selling price per tire
def selling_price_per_tire : ℝ := production_cost_per_tire * selling_price_factor

-- Define John's current daily earnings from selling 1000 tires
def current_daily_earnings : ℝ := daily_production * selling_price_per_tire

-- Define John's potential daily earnings from selling 1200 tires
def potential_daily_earnings : ℝ := potential_daily_sales * selling_price_per_tire

-- Define the daily loss by not being able to produce all the tires
def daily_loss : ℝ := potential_daily_earnings - current_daily_earnings

-- Define the weekly loss
def weekly_loss : ℝ := daily_loss * days_in_week

-- Statement: Prove that John's weekly financial loss is $525,000
theorem john_weekly_loss_is_525000 : weekly_loss = 525000 :=
by
  sorry

end john_weekly_loss_is_525000_l78_78494


namespace distance_between_circle_centers_l78_78823

-- Define the given side lengths of the triangle
def DE : ℝ := 12
def DF : ℝ := 15
def EF : ℝ := 9

-- Define the problem and assertion
theorem distance_between_circle_centers :
  ∃ d : ℝ, d = 12 * Real.sqrt 13 :=
sorry

end distance_between_circle_centers_l78_78823


namespace simplify_expression_l78_78629

theorem simplify_expression :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
  sorry

end simplify_expression_l78_78629


namespace car_speed_without_red_light_l78_78500

theorem car_speed_without_red_light (v : ℝ) :
  (∃ k : ℕ+, v = 10 / k) ↔ 
  ∀ (dist : ℝ) (green_duration red_duration total_cycle : ℝ),
    dist = 1500 ∧ green_duration = 90 ∧ red_duration = 60 ∧ total_cycle = 150 →
    v * total_cycle = dist / (green_duration + red_duration) := 
by
  sorry

end car_speed_without_red_light_l78_78500


namespace probability_no_3x3_red_square_l78_78360

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l78_78360


namespace average_salary_l78_78903

theorem average_salary (a b c d e : ℕ) (h₁ : a = 8000) (h₂ : b = 5000) (h₃ : c = 15000) (h₄ : d = 7000) (h₅ : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by sorry

end average_salary_l78_78903


namespace omitted_decimal_sum_is_integer_l78_78786

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end omitted_decimal_sum_is_integer_l78_78786


namespace least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l78_78344

noncomputable def sum_of_cubes (n : ℕ) : ℕ :=
  (n * (n + 1)/2)^2

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem least_m_for_sum_of_cubes_is_perfect_cube 
  (h : ∃ m : ℕ, ∀ (a : ℕ), (sum_of_cubes (2*m+1) = a^3) → a = 6):
  m = 1 := sorry

theorem least_k_for_sum_of_squares_is_perfect_square 
  (h : ∃ k : ℕ, ∀ (b : ℕ), (sum_of_squares (2*k+1) = b^2) → b = 77):
  k = 5 := sorry

end least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l78_78344


namespace sum_of_products_two_at_a_time_l78_78695

theorem sum_of_products_two_at_a_time
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 222)
  (h2 : a + b + c = 22) :
  a * b + b * c + c * a = 131 := 
sorry

end sum_of_products_two_at_a_time_l78_78695


namespace problem_a_problem_b_problem_c_l78_78989

theorem problem_a : (7 * (2 / 3) + 16 * (5 / 12)) = 11.3333 := by
  sorry

theorem problem_b : (5 - (2 / (5 / 3))) = 3.8 := by
  sorry

theorem problem_c : (1 + 2 / (1 + 3 / (1 + 4))) = 2.25 := by
  sorry

end problem_a_problem_b_problem_c_l78_78989


namespace boat_speed_of_stream_l78_78811

theorem boat_speed_of_stream :
  ∀ (x : ℝ), 
    (∀ s_b : ℝ, s_b = 18) → 
    (∀ d1 d2 : ℝ, d1 = 48 → d2 = 32 → d1 / (18 + x) = d2 / (18 - x)) → 
    x = 3.6 :=
by 
  intros x h_speed h_distance
  sorry

end boat_speed_of_stream_l78_78811


namespace segments_divide_ratio_3_to_1_l78_78112

-- Define points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (A B : Point)

-- Define T-shaped figure consisting of 22 unit squares
noncomputable def T_shaped_figure : ℕ := 22

-- Define line p passing through point V
structure Line :=
  (p : Point → Point)
  (passes_through : Point)

-- Define equal areas condition
def equal_areas (white_area gray_area : ℝ) : Prop := 
  white_area = gray_area

-- Define the problem
theorem segments_divide_ratio_3_to_1
  (AB : Segment)
  (V : Point)
  (white_area gray_area : ℝ)
  (p : Line)
  (h1 : equal_areas white_area gray_area)
  (h2 : T_shaped_figure = 22)
  (h3 : p.passes_through = V) :
  ∃ (C : Point), (p.p AB.A = C) ∧ ((abs (AB.A.x - C.x)) / (abs (C.x - AB.B.x))) = 3 :=
sorry

end segments_divide_ratio_3_to_1_l78_78112


namespace students_taking_neither_l78_78348

theorem students_taking_neither (total biology chemistry both : ℕ)
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  (total - (biology + chemistry - both)) = 10 :=
by {
  sorry
}

end students_taking_neither_l78_78348


namespace base_b_square_l78_78076

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, k^2 = b^2 + 4 * b + 4 := 
by 
  sorry

end base_b_square_l78_78076


namespace dwarf_diamond_distribution_l78_78677

-- Definitions for conditions
def dwarves : Type := Fin 8
structure State :=
  (diamonds : dwarves → ℕ)

-- Initial condition: Each dwarf has 3 diamonds
def initial_state : State := 
  { diamonds := fun _ => 3 }

-- Transition function: Each dwarf divides diamonds into two piles and passes them to neighbors
noncomputable def transition (s : State) : State := sorry

-- Proof goal: At a certain point in time, 3 specific dwarves have 24 diamonds in total,
-- with one dwarf having 7 diamonds, then prove the other two dwarves have 12 and 5 diamonds.
theorem dwarf_diamond_distribution (s : State)
  (h1 : ∃ t, s = (transition^[t]) initial_state ∧ ∃ i j k : dwarves, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    s.diamonds i + s.diamonds j + s.diamonds k = 24 ∧
    s.diamonds i = 7)
  : ∃ a b : dwarves, a ≠ b ∧ s.diamonds a = 12 ∧ s.diamonds b = 5 := sorry

end dwarf_diamond_distribution_l78_78677


namespace solve_coffee_problem_l78_78779

variables (initial_stock new_purchase : ℕ)
           (initial_decaf_percentage new_decaf_percentage : ℚ)
           (total_stock total_decaf weight_percentage_decaf : ℚ)

def coffee_problem :=
  initial_stock = 400 ∧
  initial_decaf_percentage = 0.20 ∧
  new_purchase = 100 ∧
  new_decaf_percentage = 0.50 ∧
  total_stock = initial_stock + new_purchase ∧
  total_decaf = initial_stock * initial_decaf_percentage + new_purchase * new_decaf_percentage ∧
  weight_percentage_decaf = (total_decaf / total_stock) * 100

theorem solve_coffee_problem : coffee_problem 400 100 0.20 0.50 500 130 26 :=
by {
  sorry
}

end solve_coffee_problem_l78_78779


namespace melted_ice_cream_depth_l78_78368

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ),
    r_sphere = 3 ∧ r_cylinder = 12 ∧
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder →
    h_cylinder = 1 / 4 :=
by
  intros r_sphere r_cylinder h_cylinder h
  have r_sphere_eq : r_sphere = 3 := h.1
  have r_cylinder_eq : r_cylinder = 12 := h.2.1
  have volume_eq : (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder := h.2.2
  sorry

end melted_ice_cream_depth_l78_78368


namespace hyperbola_sum_l78_78091

noncomputable def h : ℝ := -3
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 50
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem hyperbola_sum :
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end hyperbola_sum_l78_78091


namespace petya_mistake_l78_78517

theorem petya_mistake :
  (35 + 10 - 41 = 42 + 12 - 50) →
  (35 + 10 - 45 = 42 + 12 - 54) →
  (5 * (7 + 2 - 9) = 6 * (7 + 2 - 9)) →
  False :=
by
  intros h1 h2 h3
  sorry

end petya_mistake_l78_78517


namespace find_a7_l78_78706

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l78_78706


namespace find_m_l78_78826

variables (AB AC AD : ℝ × ℝ)
variables (m : ℝ)

-- Definitions of vectors
def vector_AB : ℝ × ℝ := (-1, 2)
def vector_AC : ℝ × ℝ := (2, 3)
def vector_AD (m : ℝ) : ℝ × ℝ := (m, -3)

-- Conditions
def collinear (B C D : ℝ × ℝ) : Prop := ∃ k : ℝ, B = k • C ∨ C = k • D ∨ D = k • B

-- Main statement to prove
theorem find_m (h1 : vector_AB = (-1, 2))
               (h2 : vector_AC = (2, 3))
               (h3 : vector_AD m = (m, -3))
               (h4 : collinear vector_AB vector_AC (vector_AD m)) :
  m = -16 :=
sorry

end find_m_l78_78826


namespace total_erasers_l78_78245

def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

theorem total_erasers : cases * boxes_per_case * erasers_per_box = 2100 := by
  sorry

end total_erasers_l78_78245


namespace f_fraction_neg_1987_1988_l78_78754

-- Define the function f and its properties
def f : ℚ → ℝ := sorry

axiom functional_eq (x y : ℚ) : f (x + y) = f x * f y - f (x * y) + 1
axiom not_equal_f : f 1988 ≠ f 1987

-- Prove the desired equality
theorem f_fraction_neg_1987_1988 : f (-1987 / 1988) = 1 / 1988 :=
by
  sorry

end f_fraction_neg_1987_1988_l78_78754


namespace g_value_at_4_l78_78376

noncomputable def g : ℝ → ℝ := sorry -- We will define g here

def functional_condition (g : ℝ → ℝ) := ∀ x y : ℝ, x * g y = y * g x
def g_value_at_12 := g 12 = 30

theorem g_value_at_4 (g : ℝ → ℝ) (h₁ : functional_condition g) (h₂ : g_value_at_12) : g 4 = 10 := 
sorry

end g_value_at_4_l78_78376


namespace bobby_total_candy_l78_78408

theorem bobby_total_candy (candy1 candy2 : ℕ) (h1 : candy1 = 26) (h2 : candy2 = 17) : candy1 + candy2 = 43 := 
by 
  sorry

end bobby_total_candy_l78_78408


namespace fraction_of_women_married_l78_78631

theorem fraction_of_women_married (total : ℕ) (women men married: ℕ) (h1 : total = women + men)
(h2 : women = 76 * total / 100) (h3 : married = 60 * total / 100) (h4 : 2 * (men - married) = 3 * men):
 (married - (total - women - married) * 1 / 3) = 13 * women / 19 :=
sorry

end fraction_of_women_married_l78_78631


namespace geometric_sequence_fifth_term_l78_78738

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 16) (h2 : a * r^6 = 2) : a * r^4 = 2 :=
sorry

end geometric_sequence_fifth_term_l78_78738


namespace sum_zero_implies_inequality_l78_78487

variable {a b c d : ℝ}

theorem sum_zero_implies_inequality
  (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := 
sorry

end sum_zero_implies_inequality_l78_78487


namespace Susan_initial_amount_l78_78675

def initial_amount (S : ℝ) : Prop :=
  let Spent_in_September := (1/6) * S
  let Spent_in_October := (1/8) * S
  let Spent_in_November := 0.3 * S
  let Spent_in_December := 100
  let Remaining := 480
  S - (Spent_in_September + Spent_in_October + Spent_in_November + Spent_in_December) = Remaining

theorem Susan_initial_amount : ∃ S : ℝ, initial_amount S ∧ S = 1420 :=
by
  sorry

end Susan_initial_amount_l78_78675


namespace simplify_and_evaluate_expression_l78_78539

theorem simplify_and_evaluate_expression
    (a b : ℤ)
    (h1 : a = -1/3)
    (h2 : b = -2) :
  ((3 * a + b)^2 - (3 * a + b) * (3 * a - b)) / (2 * b) = -3 :=
by
  sorry

end simplify_and_evaluate_expression_l78_78539


namespace smaller_number_l78_78806

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l78_78806


namespace gasohol_problem_l78_78024

noncomputable def initial_gasohol_volume (x : ℝ) : Prop :=
  let ethanol_in_initial_mix := 0.05 * x
  let ethanol_to_add := 2
  let total_ethanol := ethanol_in_initial_mix + ethanol_to_add
  let total_volume := x + 2
  0.1 * total_volume = total_ethanol

theorem gasohol_problem (x : ℝ) : initial_gasohol_volume x → x = 36 := by
  intro h
  sorry

end gasohol_problem_l78_78024


namespace garden_area_difference_l78_78079
-- Import the entire Mathlib

-- Lean Statement
theorem garden_area_difference :
  let length_Alice := 15
  let width_Alice := 30
  let length_Bob := 18
  let width_Bob := 28
  let area_Alice := length_Alice * width_Alice
  let area_Bob := length_Bob * width_Bob
  let difference := area_Bob - area_Alice
  difference = 54 :=
by
  sorry

end garden_area_difference_l78_78079


namespace point_in_second_quadrant_l78_78084

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant_l78_78084


namespace pear_sales_ratio_l78_78142

theorem pear_sales_ratio : 
  ∀ (total_sold afternoon_sold morning_sold : ℕ), 
  total_sold = 420 ∧ afternoon_sold = 280 ∧ total_sold = afternoon_sold + morning_sold 
  → afternoon_sold / morning_sold = 2 :=
by 
  intros total_sold afternoon_sold morning_sold 
  intro h 
  have h_total : total_sold = 420 := h.1 
  have h_afternoon : afternoon_sold = 280 := h.2.1 
  have h_morning : total_sold = afternoon_sold + morning_sold := h.2.2
  sorry

end pear_sales_ratio_l78_78142


namespace exists_unique_xy_l78_78488

theorem exists_unique_xy (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end exists_unique_xy_l78_78488


namespace habitable_land_area_l78_78387

noncomputable def area_of_habitable_land : ℝ :=
  let length : ℝ := 23
  let diagonal : ℝ := 33
  let radius_of_pond : ℝ := 3
  let width : ℝ := Real.sqrt (diagonal ^ 2 - length ^ 2)
  let area_of_rectangle : ℝ := length * width
  let area_of_pond : ℝ := Real.pi * (radius_of_pond ^ 2)
  area_of_rectangle - area_of_pond

theorem habitable_land_area :
  abs (area_of_habitable_land - 515.91) < 0.01 :=
by
  sorry

end habitable_land_area_l78_78387


namespace quadratic_roots_one_is_twice_l78_78814

theorem quadratic_roots_one_is_twice (a b c : ℝ) (m : ℝ) :
  (∃ x1 x2 : ℝ, 2 * x1^2 - (2 * m + 1) * x1 + m^2 - 9 * m + 39 = 0 ∧ x2 = 2 * x1) ↔ m = 10 ∨ m = 7 :=
by 
  sorry

end quadratic_roots_one_is_twice_l78_78814


namespace coin_probability_not_unique_l78_78146

variables (p : ℝ) (w : ℝ)
def binomial_prob := 10 * p^3 * (1 - p)^2

theorem coin_probability_not_unique (h : binomial_prob p = 144 / 625) : 
  ∃ p1 p2, p1 ≠ p2 ∧ binomial_prob p1 = 144 / 625 ∧ binomial_prob p2 = 144 / 625 :=
by 
  sorry

end coin_probability_not_unique_l78_78146


namespace range_of_m_l78_78865

variable {m x : ℝ}

theorem range_of_m (h : ∀ x, -1 < x ∧ x < 4 ↔ x > 2 * m ^ 2 - 3) : m ∈ [-1, 1] :=
sorry

end range_of_m_l78_78865


namespace middle_number_is_9_l78_78872

-- Define the problem conditions
variable (x y z : ℕ)

-- Lean proof statement
theorem middle_number_is_9 
  (h1 : x + y = 16)
  (h2 : x + z = 21)
  (h3 : y + z = 23)
  (h4 : x < y)
  (h5 : y < z) : y = 9 :=
by
  sorry

end middle_number_is_9_l78_78872


namespace range_x_minus_q_l78_78628

theorem range_x_minus_q (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) : x - q < 3 - 2*q :=
by
  sorry

end range_x_minus_q_l78_78628


namespace sum_of_squares_eq_229_l78_78622

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l78_78622


namespace start_of_range_l78_78211

variable (x : ℕ)

theorem start_of_range (h : ∃ (n : ℕ), n ≤ 79 ∧ n % 11 = 0 ∧ x = 79 - 3 * 11) 
(h4 : ∀ (k : ℕ), 0 ≤ k ∧ k < 4 → ∃ (y : ℕ), y = 79 - (k * 11) ∧ y % 11 = 0) :
  x = 44 := by
  sorry

end start_of_range_l78_78211


namespace WalterWorksDaysAWeek_l78_78602

theorem WalterWorksDaysAWeek (hourlyEarning : ℕ) (hoursPerDay : ℕ) (schoolAllocationFraction : ℚ) (schoolAllocation : ℕ) 
  (dailyEarning : ℕ) (weeklyEarning : ℕ) (daysWorked : ℕ) :
  hourlyEarning = 5 →
  hoursPerDay = 4 →
  schoolAllocationFraction = 3 / 4 →
  schoolAllocation = 75 →
  dailyEarning = hourlyEarning * hoursPerDay →
  weeklyEarning = (schoolAllocation : ℚ) / schoolAllocationFraction →
  daysWorked = weeklyEarning / dailyEarning →
  daysWorked = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end WalterWorksDaysAWeek_l78_78602


namespace wall_width_l78_78985

theorem wall_width (V h l w : ℝ) (h_cond : h = 6 * w) (l_cond : l = 42 * w) (vol_cond : 252 * w^3 = 129024) : w = 8 := 
by
  -- Proof is omitted; required to produce lean statement only
  sorry

end wall_width_l78_78985


namespace remainder_of_55_pow_55_plus_15_mod_8_l78_78688

theorem remainder_of_55_pow_55_plus_15_mod_8 :
  (55^55 + 15) % 8 = 6 := by
  -- This statement does not include any solution steps.
  sorry

end remainder_of_55_pow_55_plus_15_mod_8_l78_78688


namespace red_grapes_in_salad_l78_78876

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l78_78876


namespace profit_percentage_correct_l78_78812

noncomputable def CP : ℝ := 460
noncomputable def SP : ℝ := 542.8
noncomputable def profit : ℝ := SP - CP
noncomputable def profit_percentage : ℝ := (profit / CP) * 100

theorem profit_percentage_correct :
  profit_percentage = 18 := by
  sorry

end profit_percentage_correct_l78_78812


namespace non_neg_int_solutions_l78_78093

theorem non_neg_int_solutions (n : ℕ) (a b : ℤ) :
  n^2 = a + b ∧ n^3 = a^2 + b^2 → n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end non_neg_int_solutions_l78_78093


namespace square_root_problem_l78_78116

theorem square_root_problem
  (x : ℤ) (y : ℤ)
  (hx : x = Nat.sqrt 16)
  (hy : y^2 = 9) :
  x^2 + y^2 + x - 2 = 27 := by
  sorry

end square_root_problem_l78_78116


namespace distance_light_travels_250_years_l78_78223

def distance_light_travels_one_year : ℝ := 5.87 * 10^12
def years : ℝ := 250

theorem distance_light_travels_250_years :
  distance_light_travels_one_year * years = 1.4675 * 10^15 :=
by
  sorry

end distance_light_travels_250_years_l78_78223


namespace eval_infinite_product_l78_78412

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, (3:ℝ)^(2 * n / (3:ℝ)^n)

theorem eval_infinite_product : infinite_product = (3:ℝ)^(9 / 2) := by
  sorry

end eval_infinite_product_l78_78412


namespace sum_of_common_ratios_l78_78385

theorem sum_of_common_ratios (k p r : ℝ) (h1 : k ≠ 0) (h2 : k * (p^2) - k * (r^2) = 5 * (k * p - k * r)) (h3 : p ≠ r) : p + r = 5 :=
sorry

end sum_of_common_ratios_l78_78385


namespace find_a_for_max_y_l78_78207

theorem find_a_for_max_y (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 2 * (x - 1)^2 - 3 ≤ 15) →
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ a ∧ 2 * (x - 1)^2 - 3 = 15) →
  a = 4 :=
by sorry

end find_a_for_max_y_l78_78207


namespace largest_sum_pairs_l78_78218

theorem largest_sum_pairs (a b c d : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a < b) (h₇ : b < c) (h₈ : c < d)
(h₉ : a + b = 9 ∨ a + b = 10) (h₁₀ : b + c = 9 ∨ b + c = 10)
(h₁₁ : b + d = 12) (h₁₂ : c + d = 13) :
d = 8 ∨ d = 7.5 :=
sorry

end largest_sum_pairs_l78_78218


namespace cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l78_78652

-- Problem Part 1
theorem cos_alpha_implies_sin_alpha (alpha : ℝ) (h1 : Real.cos alpha = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin alpha = -3/5 := sorry

-- Problem Part 2
theorem tan_theta_implies_expr (theta : ℝ) (h1 : Real.tan theta = 3) : 
  (Real.sin theta + Real.cos theta) / (2 * Real.sin theta + Real.cos theta) = 4 / 7 := sorry

end cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l78_78652


namespace bob_repay_l78_78364

theorem bob_repay {x : ℕ} (h : 50 + 10 * x >= 150) : x >= 10 :=
by
  sorry

end bob_repay_l78_78364


namespace total_sacks_after_6_days_l78_78454

-- Define the conditions
def sacks_per_day : ℕ := 83
def days : ℕ := 6

-- Prove the total number of sacks after 6 days is 498
theorem total_sacks_after_6_days : sacks_per_day * days = 498 := by
  -- Proof Content Placeholder
  sorry

end total_sacks_after_6_days_l78_78454
