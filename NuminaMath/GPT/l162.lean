import Mathlib

namespace polynomial_even_or_odd_polynomial_divisible_by_3_l162_16299

theorem polynomial_even_or_odd (p q : ℤ) :
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 0 ↔ (q % 2 = 0) ∧ (p % 2 = 1)) ∧
  (∀ x : ℤ, (x^2 + p * x + q) % 2 = 1 ↔ (q % 2 = 1) ∧ (p % 2 = 1)) := 
sorry

theorem polynomial_divisible_by_3 (p q : ℤ) :
  (∀ x : ℤ, (x^3 + p * x + q) % 3 = 0) ↔ (q % 3 = 0) ∧ (p % 3 = 2) := 
sorry

end polynomial_even_or_odd_polynomial_divisible_by_3_l162_16299


namespace inequality_for_five_real_numbers_l162_16252

open Real

theorem inequality_for_five_real_numbers
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (h4 : 1 < a4)
  (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_for_five_real_numbers_l162_16252


namespace quadrilateral_count_l162_16288

-- Define the number of points
def num_points := 9

-- Define the number of vertices in a quadrilateral
def vertices_in_quadrilateral := 4

-- Use a combination function to find the number of ways to choose 4 points out of 9
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem that asserts the number of quadrilaterals that can be formed
theorem quadrilateral_count : combination num_points vertices_in_quadrilateral = 126 :=
by
  -- The proof would go here
  sorry

end quadrilateral_count_l162_16288


namespace min_value_of_F_on_negative_half_l162_16278

variable (f g : ℝ → ℝ)
variable (a b : ℝ)

def F (x : ℝ) := a * f x + b * g x + 2

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_value_of_F_on_negative_half
  (h_f : is_odd f) (h_g : is_odd g)
  (max_F_positive_half : ∃ x, x > 0 ∧ F f g a b x = 5) :
  ∃ x, x < 0 ∧ F f g a b x = -3 :=
by {
  sorry
}

end min_value_of_F_on_negative_half_l162_16278


namespace find_b_l162_16211

theorem find_b
  (b : ℝ)
  (hx : ∃ y : ℝ, 4 * 3 + 2 * y = b ∧ 3 * 3 + 4 * y = 3 * b) :
  b = -15 :=
sorry

end find_b_l162_16211


namespace evaluate_series_l162_16248

theorem evaluate_series : 1 + (1 / 2) + (1 / 4) + (1 / 8) = 15 / 8 := by
  sorry

end evaluate_series_l162_16248


namespace sum_greater_than_product_l162_16205

theorem sum_greater_than_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
by { sorry }

end sum_greater_than_product_l162_16205


namespace facemasks_per_box_l162_16293

theorem facemasks_per_box (x : ℝ) :
  (3 * x * 0.50) - 15 = 15 → x = 20 :=
by
  intros h
  sorry

end facemasks_per_box_l162_16293


namespace simplify_expression_l162_16281

theorem simplify_expression (y : ℝ) : (y - 2) ^ 2 + 2 * (y - 2) * (4 + y) + (4 + y) ^ 2 = 4 * (y + 1) ^ 2 := 
by 
  sorry

end simplify_expression_l162_16281


namespace final_price_correct_l162_16260

-- Definitions that follow the given conditions
def initial_price : ℝ := 150
def increase_percentage_year1 : ℝ := 1.5
def decrease_percentage_year2 : ℝ := 0.3

-- Compute intermediate values
noncomputable def price_end_year1 : ℝ := initial_price + (increase_percentage_year1 * initial_price)
noncomputable def price_end_year2 : ℝ := price_end_year1 - (decrease_percentage_year2 * price_end_year1)

-- The final theorem stating the price at the end of the second year
theorem final_price_correct : price_end_year2 = 262.5 := by
  sorry

end final_price_correct_l162_16260


namespace function_always_negative_iff_l162_16283

theorem function_always_negative_iff (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 :=
by
  -- Proof skipped
  sorry

end function_always_negative_iff_l162_16283


namespace min_unattainable_score_l162_16219

theorem min_unattainable_score : ∀ (score : ℕ), (¬ ∃ (a b c : ℕ), 
  (a = 1 ∨ a = 3 ∨ a = 8 ∨ a = 12 ∨ a = 0) ∧ 
  (b = 1 ∨ b = 3 ∨ b = 8 ∨ b = 12 ∨ b = 0) ∧ 
  (c = 1 ∨ c = 3 ∨ c = 8 ∨ c = 12 ∨ c = 0) ∧ 
  score = a + b + c) ↔ score = 22 := 
by
  sorry

end min_unattainable_score_l162_16219


namespace tom_catches_up_in_60_minutes_l162_16272

-- Definitions of the speeds and initial distance
def lucy_speed : ℝ := 4  -- Lucy's speed in miles per hour
def tom_speed : ℝ := 6   -- Tom's speed in miles per hour
def initial_distance : ℝ := 2  -- Initial distance between Tom and Lucy in miles

-- Conclusion that needs to be proved
theorem tom_catches_up_in_60_minutes :
  (initial_distance / (tom_speed - lucy_speed)) * 60 = 60 :=
by
  sorry

end tom_catches_up_in_60_minutes_l162_16272


namespace three_at_five_l162_16286

def op_at (a b : ℤ) : ℤ := 3 * a - 3 * b

theorem three_at_five : op_at 3 5 = -6 :=
by
  sorry

end three_at_five_l162_16286


namespace vector_parallel_x_is_neg1_l162_16224

variables (a b : ℝ × ℝ)
variable (x : ℝ)

def vectors_parallel : Prop := 
  (a = (1, -1)) ∧ (b = (x, 1)) ∧ (a.1 * b.2 - a.2 * b.1 = 0)

theorem vector_parallel_x_is_neg1 (h : vectors_parallel a b x) : x = -1 :=
sorry

end vector_parallel_x_is_neg1_l162_16224


namespace rectangle_perimeter_l162_16202

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangle_perimeter (x y : ℝ) (A : ℝ) (E : ℝ) (fA fB : Real) (p : ℝ) 
  (h1 : y = 2 * x)
  (h2 : x * y = 2015)
  (h3 : E = 2006 * π)
  (h4 : fA = x + y)
  (h5 : fB ^ 2 = (3 / 2)^2 * 1007.5 - (p / 2)^2)
  (h6 : 2 * (3 / 2 * sqrt 1007.5 * sqrt 1009.375) = 2006 / π) :
  2 * (x + y) = 6 * sqrt 1007.5 := 
by
  sorry

end rectangle_perimeter_l162_16202


namespace part_I_min_value_part_II_nonexistence_l162_16268

theorem part_I_min_value (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : a^2 + 16 * b^2 ≥ 32 :=
by
  sorry

theorem part_II_nonexistence (a b : ℝ) (hab : a > 0 ∧ b > 0 ∧ a + 4 * b = (a * b)^(3/2)) : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 6 :=
by
  sorry

end part_I_min_value_part_II_nonexistence_l162_16268


namespace pete_ran_least_distance_l162_16200

theorem pete_ran_least_distance
  (phil_distance : ℕ := 4)
  (tom_distance : ℕ := 6)
  (pete_distance : ℕ := 2)
  (amal_distance : ℕ := 8)
  (sanjay_distance : ℕ := 7) :
  pete_distance ≤ phil_distance ∧
  pete_distance ≤ tom_distance ∧
  pete_distance ≤ amal_distance ∧
  pete_distance ≤ sanjay_distance :=
by {
  sorry
}

end pete_ran_least_distance_l162_16200


namespace rectangle_area_at_stage_8_l162_16266

-- Declare constants for the conditions.
def square_side_length : ℕ := 4
def number_of_stages : ℕ := 8
def area_of_single_square : ℕ := square_side_length * square_side_length

-- The statement to prove
theorem rectangle_area_at_stage_8 : number_of_stages * area_of_single_square = 128 := by
  sorry

end rectangle_area_at_stage_8_l162_16266


namespace Carter_card_number_l162_16227

-- Definitions based on conditions
def Marcus_cards : ℕ := 210
def difference : ℕ := 58

-- Definition to infer the number of Carter's baseball cards
def Carter_cards : ℕ := Marcus_cards - difference

-- Theorem statement asserting the number of baseball cards Carter has
theorem Carter_card_number : Carter_cards = 152 := by
  sorry

end Carter_card_number_l162_16227


namespace sector_angle_l162_16203

noncomputable def central_angle_of_sector (r l : ℝ) : ℝ := l / r

theorem sector_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  central_angle_of_sector r l = 1 ∨ central_angle_of_sector r l = 4 :=
by
  sorry

end sector_angle_l162_16203


namespace total_drums_hit_l162_16218

/-- 
Given the conditions of the problem, Juanita hits 4500 drums in total. 
-/
theorem total_drums_hit (entry_fee cost_per_drum_hit earnings_per_drum_hit_beyond_200_double
                         net_loss: ℝ) 
                         (first_200_drums hits_after_200: ℕ) :
  entry_fee = 10 → 
  cost_per_drum_hit = 0.02 →
  earnings_per_drum_hit_beyond_200_double = 0.025 →
  net_loss = -7.5 →
  hits_after_200 = 4300 →
  first_200_drums = 200 →
  (-net_loss = entry_fee + (first_200_drums * cost_per_drum_hit) +
   (hits_after_200 * (earnings_per_drum_hit_beyond_200_double - cost_per_drum_hit))) →
  first_200_drums + hits_after_200 = 4500 :=
by
  intro h_entry_fee h_cost_per_drum_hit h_earnings_per_drum_hit_beyond_200_double h_net_loss h_hits_after_200
       h_first_200_drums h_loss_equation
  sorry

end total_drums_hit_l162_16218


namespace spencer_total_distance_l162_16285

-- Definitions for the given conditions
def distance_house_to_library : ℝ := 0.3
def distance_library_to_post_office : ℝ := 0.1
def distance_post_office_to_home : ℝ := 0.4

-- Define the total distance based on the given conditions
def total_distance : ℝ := distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home

-- Statement to prove
theorem spencer_total_distance : total_distance = 0.8 := by
  sorry

end spencer_total_distance_l162_16285


namespace sales_tax_calculation_l162_16241

theorem sales_tax_calculation 
  (total_amount_paid : ℝ)
  (tax_rate : ℝ)
  (cost_tax_free : ℝ) :
  total_amount_paid = 30 → tax_rate = 0.08 → cost_tax_free = 12.72 → 
  (∃ sales_tax : ℝ, sales_tax = 1.28) :=
by
  intros H1 H2 H3
  sorry

end sales_tax_calculation_l162_16241


namespace ratio_of_area_to_breadth_is_15_l162_16298

-- Definitions for our problem
def breadth := 5
def length := 15 -- since l - b = 10 and b = 5

-- Given conditions
axiom area_is_ktimes_breadth (k : ℝ) : length * breadth = k * breadth
axiom length_breadth_difference : length - breadth = 10

-- The proof statement
theorem ratio_of_area_to_breadth_is_15 : (length * breadth) / breadth = 15 := by
  sorry

end ratio_of_area_to_breadth_is_15_l162_16298


namespace towel_bleach_percentage_decrease_l162_16207

theorem towel_bleach_percentage_decrease :
  ∀ (L B : ℝ), (L > 0) → (B > 0) → 
  let L' := 0.70 * L 
  let B' := 0.75 * B 
  let A := L * B 
  let A' := L' * B' 
  (A - A') / A * 100 = 47.5 :=
by sorry

end towel_bleach_percentage_decrease_l162_16207


namespace contrapositive_example_l162_16267

theorem contrapositive_example (a b : ℝ) :
  (a > b → a - 1 > b - 2) ↔ (a - 1 ≤ b - 2 → a ≤ b) := 
by
  sorry

end contrapositive_example_l162_16267


namespace length_of_BC_l162_16296

-- Definitions of given conditions
def AB : ℝ := 4
def AC : ℝ := 3
def dot_product_AC_BC : ℝ := 1

-- Hypothesis used in the problem
axiom nonneg_AC (AC : ℝ) : AC ≥ 0
axiom nonneg_AB (AB : ℝ) : AB ≥ 0

-- Statement to be proved
theorem length_of_BC (AB AC dot_product_AC_BC : ℝ)
  (h1 : AB = 4) (h2 : AC = 3) (h3 : dot_product_AC_BC = 1) : exists (BC : ℝ), BC = 3 := by
  sorry

end length_of_BC_l162_16296


namespace total_lives_l162_16237

theorem total_lives (initial_players new_players lives_per_person : ℕ)
  (h_initial : initial_players = 8)
  (h_new : new_players = 2)
  (h_lives : lives_per_person = 6)
  : (initial_players + new_players) * lives_per_person = 60 := 
by
  sorry

end total_lives_l162_16237


namespace pow_mod_eq_l162_16290

theorem pow_mod_eq : (6 ^ 2040) % 50 = 26 := by
  sorry

end pow_mod_eq_l162_16290


namespace abs_value_expression_l162_16280

theorem abs_value_expression : abs (3 * Real.pi - abs (3 * Real.pi - 10)) = 6 * Real.pi - 10 :=
by sorry

end abs_value_expression_l162_16280


namespace real_solutions_eq_l162_16225

def satisfies_equations (x y : ℝ) : Prop :=
  (4 * x + 5 * y = 13) ∧ (2 * x - 3 * y = 1)

theorem real_solutions_eq {x y : ℝ} : satisfies_equations x y ↔ (x = 2 ∧ y = 1) :=
by sorry

end real_solutions_eq_l162_16225


namespace find_incorrect_statement_l162_16289

variable (q n x y : ℚ)

theorem find_incorrect_statement :
  (∀ q, q < -1 → q < 1/q) ∧
  (∀ n, n ≥ 0 → -n ≥ n) ∧
  (∀ x, x < 0 → x^3 < x) ∧
  (∀ y, y < 0 → y^2 > y) →
  (∃ x, x < 0 ∧ ¬ (x^3 < x)) :=
by
  sorry

end find_incorrect_statement_l162_16289


namespace sequence_general_term_l162_16228

theorem sequence_general_term (a : ℕ → ℤ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, 1 < n → a n = 2 * (n + a (n - 1))) :
  ∀ n, 1 ≤ n → a n = 2 ^ (n + 2) - 2 * n - 4 :=
by
  sorry

end sequence_general_term_l162_16228


namespace students_per_row_l162_16250

theorem students_per_row (x : ℕ) : 45 = 11 * x + 1 → x = 4 :=
by
  intro h
  sorry

end students_per_row_l162_16250


namespace compute_division_l162_16222

variable (a b c : ℕ)
variable (ha : a = 3)
variable (hb : b = 2)
variable (hc : c = 2)

theorem compute_division : (c * a^3 + c * b^3) / (a^2 - a * b + b^2) = 10 := by
  sorry

end compute_division_l162_16222


namespace decipher_numbers_l162_16269

variable (K I S : Nat)

theorem decipher_numbers
  (h1: 1 ≤ K ∧ K < 5)
  (h2: I ≠ 0)
  (h3: I ≠ K)
  (h_eq: K * 100 + I * 10 + S + K * 10 + S * 10 + I = I * 100 + S * 10 + K):
  (K, I, S) = (4, 9, 5) :=
by sorry

end decipher_numbers_l162_16269


namespace children_eating_porridge_today_l162_16249

theorem children_eating_porridge_today
  (eat_every_day : ℕ)
  (eat_every_other_day : ℕ)
  (ate_yesterday : ℕ) :
  eat_every_day = 5 →
  eat_every_other_day = 7 →
  ate_yesterday = 9 →
  (eat_every_day + (eat_every_other_day - (ate_yesterday - eat_every_day)) = 8) :=
by
  intros h1 h2 h3
  sorry

end children_eating_porridge_today_l162_16249


namespace solve_for_x_l162_16255

theorem solve_for_x :
  ∃ x : ℝ, ((17.28 / x) / (3.6 * 0.2)) = 2 ∧ x = 12 :=
by
  sorry

end solve_for_x_l162_16255


namespace part1_part2_l162_16264

-- Prove Part (1)
theorem part1 (M : ℕ) (N : ℕ) (h : M = 9) (h2 : N - 4 + 6 = M) : N = 7 :=
sorry

-- Prove Part (2)
theorem part2 (M : ℕ) (h : M = 9) : M - 4 = 5 ∨ M + 4 = 13 :=
sorry

end part1_part2_l162_16264


namespace probability_even_heads_after_60_flips_l162_16257

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (3 / 4) - (1 / 2) * P_n (n - 1)

theorem probability_even_heads_after_60_flips :
  P_n 60 = 1 / 2 * (1 + 1 / 2^60) :=
sorry

end probability_even_heads_after_60_flips_l162_16257


namespace intersection_eq_l162_16209

-- Given conditions
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | x > 1 }

-- Statement of the problem to be proved
theorem intersection_eq : M ∩ N = { x | 1 < x ∧ x < 3 } :=
sorry

end intersection_eq_l162_16209


namespace number_of_chords_with_integer_length_l162_16210

theorem number_of_chords_with_integer_length 
(centerP_dist radius : ℝ) 
(h1 : centerP_dist = 12) 
(h2 : radius = 20) : 
  ∃ n : ℕ, n = 9 := 
by 
  sorry

end number_of_chords_with_integer_length_l162_16210


namespace max_annual_profit_l162_16231

noncomputable def R (x : ℝ) : ℝ :=
  if x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

noncomputable def W (x : ℝ) : ℝ :=
  if x < 40 then -10 * x^2 + 600 * x - 260
  else -x + 9190 - 10000 / x

theorem max_annual_profit : ∃ x : ℝ, W 100 = 8990 :=
by {
  use 100,
  sorry
}

end max_annual_profit_l162_16231


namespace expression_evaluation_l162_16226

variable {x y : ℝ}

theorem expression_evaluation (h : (x-2)^2 + |y-3| = 0) :
  ( (x - 2 * y) * (x + 2 * y) - (x - y) ^ 2 + y * (y + 2 * x) ) / (-2 * y) = 2 :=
by
  sorry

end expression_evaluation_l162_16226


namespace intersection_M_N_l162_16214

def M : Set ℝ := { x | (x - 2) / (x - 3) < 0 }
def N : Set ℝ := { x | Real.log (x - 2) / Real.log (1 / 2) ≥ 1 }

theorem intersection_M_N : M ∩ N = { x | 2 < x ∧ x ≤ 5 / 2 } :=
by
  sorry

end intersection_M_N_l162_16214


namespace A_is_11_years_older_than_B_l162_16223

-- Define the constant B as given in the problem
def B : ℕ := 41

-- Define the condition based on the problem statement
def condition (A : ℕ) := A + 10 = 2 * (B - 10)

-- Prove the main statement that A is 11 years older than B
theorem A_is_11_years_older_than_B (A : ℕ) (h : condition A) : A - B = 11 :=
by
  sorry

end A_is_11_years_older_than_B_l162_16223


namespace marbles_per_box_l162_16204

-- Define the total number of marbles
def total_marbles : Nat := 18

-- Define the number of boxes
def number_of_boxes : Nat := 3

-- Prove there are 6 marbles in each box
theorem marbles_per_box : total_marbles / number_of_boxes = 6 := by
  sorry

end marbles_per_box_l162_16204


namespace find_gamma_delta_l162_16240

theorem find_gamma_delta (γ δ : ℝ) (h : ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90 * x + 1980) / (x^2 + 60 * x - 3240)) : 
  γ + δ = 140 :=
sorry

end find_gamma_delta_l162_16240


namespace smallest_n_for_factorable_quadratic_l162_16229

open Int

theorem smallest_n_for_factorable_quadratic : ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 72 → 3 * B + A = n) ∧ n = 35 :=
by
  sorry

end smallest_n_for_factorable_quadratic_l162_16229


namespace log_pi_inequality_l162_16243

theorem log_pi_inequality (a b : ℝ) (π : ℝ) (h1 : 2^a = π) (h2 : 5^b = π) (h3 : a = Real.log π / Real.log 2) (h4 : b = Real.log π / Real.log 5) :
  (1 / a) + (1 / b) > 2 :=
by
  sorry

end log_pi_inequality_l162_16243


namespace correct_operation_l162_16292

variable (a b : ℝ)

theorem correct_operation :
  -a^6 / a^3 = -a^3 := by
  sorry

end correct_operation_l162_16292


namespace probability_of_Y_l162_16246

variable (P_X : ℝ) (P_X_and_Y : ℝ) (P_Y : ℝ)

theorem probability_of_Y (h1 : P_X = 1 / 7)
                         (h2 : P_X_and_Y = 0.031746031746031744) :
  P_Y = 0.2222222222222222 :=
sorry

end probability_of_Y_l162_16246


namespace find_first_parrot_weight_l162_16215

def cats_weights := [7, 10, 13, 15]
def cats_sum := List.sum cats_weights
def dog1 := cats_sum - 2
def dog2 := cats_sum + 7
def dog3 := (dog1 + dog2) / 2
def dogs_sum := dog1 + dog2 + dog3
def total_parrots_weight := 2 / 3 * dogs_sum

noncomputable def parrot1 := 2 / 5 * total_parrots_weight
noncomputable def parrot2 := 3 / 5 * total_parrots_weight

theorem find_first_parrot_weight : parrot1 = 38 :=
by
  sorry

end find_first_parrot_weight_l162_16215


namespace calculate_rent_l162_16201

def monthly_income : ℝ := 3200
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def car_payment : ℝ := 350
def gas_maintenance : ℝ := 350

def total_expenses : ℝ := utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance
def rent : ℝ := monthly_income - total_expenses

theorem calculate_rent : rent = 1250 := by
  -- condition proof here
  sorry

end calculate_rent_l162_16201


namespace max_value_of_function_l162_16216

noncomputable def function_to_maximize (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + 1 / ((Real.sin x)^2 + (Real.cos x)^2 + 1)

theorem max_value_of_function :
  ∃ x : ℝ, function_to_maximize x = 7 / 4 :=
sorry

end max_value_of_function_l162_16216


namespace correct_flowchart_requirement_l162_16294

def flowchart_requirement (option : String) : Prop := 
  option = "From left to right, from top to bottom" ∨
  option = "From right to left, from top to bottom" ∨
  option = "From left to right, from bottom to top" ∨
  option = "From right to left, from bottom to top"

theorem correct_flowchart_requirement : 
  (∀ option, flowchart_requirement option → option = "From left to right, from top to bottom") :=
by
  sorry

end correct_flowchart_requirement_l162_16294


namespace only_natural_number_solution_l162_16208

theorem only_natural_number_solution (n : ℕ) :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = n * x * y * z) ↔ (n = 3) := 
sorry

end only_natural_number_solution_l162_16208


namespace problem_statement_l162_16220

/-
If x is equal to the sum of the even integers from 40 to 60 inclusive,
y is the number of even integers from 40 to 60 inclusive,
and z is the sum of the odd integers from 41 to 59 inclusive,
prove that x + y + z = 1061.
-/
theorem problem_statement :
  let x := (11 / 2) * (40 + 60)
  let y := 11
  let z := (10 / 2) * (41 + 59)
  x + y + z = 1061 :=
by
  sorry

end problem_statement_l162_16220


namespace compute_alpha_l162_16291

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end compute_alpha_l162_16291


namespace gcd_fact_8_10_l162_16245

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l162_16245


namespace parabola_vertex_l162_16263

theorem parabola_vertex (x y : ℝ) :
  (x^2 - 4 * x + 3 * y + 8 = 0) → (x, y) = (2, -4 / 3) :=
by
  sorry

end parabola_vertex_l162_16263


namespace total_pets_count_l162_16259

/-- Taylor and his six friends have a total of 45 pets, given the specified conditions about the number of each type of pet they have. -/
theorem total_pets_count
  (Taylor_cats : ℕ := 4)
  (Friend1_pets : ℕ := 8 * 3)
  (Friend2_dogs : ℕ := 3)
  (Friend2_birds : ℕ := 1)
  (Friend3_dogs : ℕ := 5)
  (Friend3_cats : ℕ := 2)
  (Friend4_reptiles : ℕ := 2)
  (Friend4_birds : ℕ := 3)
  (Friend4_cats : ℕ := 1) :
  Taylor_cats + Friend1_pets + Friend2_dogs + Friend2_birds + Friend3_dogs + Friend3_cats + Friend4_reptiles + Friend4_birds + Friend4_cats = 45 :=
sorry

end total_pets_count_l162_16259


namespace prob_three_red_cards_l162_16213

noncomputable def probability_of_three_red_cards : ℚ :=
  let total_ways := 52 * 51 * 50
  let ways_to_choose_red_cards := 26 * 25 * 24
  ways_to_choose_red_cards / total_ways

theorem prob_three_red_cards : probability_of_three_red_cards = 4 / 17 := sorry

end prob_three_red_cards_l162_16213


namespace door_solution_l162_16230

def door_problem (x : ℝ) : Prop :=
  let w := x - 4
  let h := x - 2
  let diagonal := x
  (diagonal ^ 2 - (h) ^ 2 = (w) ^ 2)

theorem door_solution (x : ℝ) : door_problem x :=
  sorry

end door_solution_l162_16230


namespace range_of_x_plus_one_over_x_l162_16253

theorem range_of_x_plus_one_over_x (x : ℝ) (h : x < 0) : x + 1/x ≤ -2 := by
  sorry

end range_of_x_plus_one_over_x_l162_16253


namespace sabrina_cookies_l162_16279

theorem sabrina_cookies :
  let S0 : ℕ := 28
  let S1 : ℕ := S0 - 10
  let S2 : ℕ := S1 + 3 * 10
  let S3 : ℕ := S2 - S2 / 3
  let S4 : ℕ := S3 + 16 / 4
  let S5 : ℕ := S4 - S4 / 2
  S5 = 18 := 
by
  -- begin proof here
  sorry

end sabrina_cookies_l162_16279


namespace exponentiation_example_l162_16232

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l162_16232


namespace simplify_expression_l162_16295

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a :=
by
  sorry

end simplify_expression_l162_16295


namespace sum_of_side_lengths_l162_16238

theorem sum_of_side_lengths (A B C : ℕ) (hA : A = 10) (h_nat_B : B > 0) (h_nat_C : C > 0)
(h_eq_area : B^2 + C^2 = A^2) : B + C = 14 :=
sorry

end sum_of_side_lengths_l162_16238


namespace people_per_van_is_six_l162_16217

noncomputable def n_vans : ℝ := 6.0
noncomputable def n_buses : ℝ := 8.0
noncomputable def p_bus : ℝ := 18.0
noncomputable def people_difference : ℝ := 108

theorem people_per_van_is_six (x : ℝ) (h : n_buses * p_bus = n_vans * x + people_difference) : x = 6.0 := 
by
  sorry

end people_per_van_is_six_l162_16217


namespace range_of_x_sq_add_y_sq_l162_16262

theorem range_of_x_sq_add_y_sq (x y : ℝ) (h : x^2 + y^2 = 4 * x) : 
  ∃ (a b : ℝ), a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b ∧ a = 0 ∧ b = 16 :=
by
  sorry

end range_of_x_sq_add_y_sq_l162_16262


namespace z_in_third_quadrant_l162_16274

def i := Complex.I

def z := i + 2 * (i^2) + 3 * (i^3)

theorem z_in_third_quadrant : 
    let z_real := Complex.re z
    let z_imag := Complex.im z
    z_real < 0 ∧ z_imag < 0 :=
by
  sorry

end z_in_third_quadrant_l162_16274


namespace solution_set_inequality1_solution_set_inequality2_l162_16212

def inequality1 (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def inequality2 (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0

theorem solution_set_inequality1 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality1 x} :=
sorry

theorem solution_set_inequality2 : {x : ℝ | (-1 / 2 : ℝ) <= x ∧ x < 3} = {x : ℝ | inequality2 x} :=
sorry

end solution_set_inequality1_solution_set_inequality2_l162_16212


namespace total_number_of_numbers_l162_16239

theorem total_number_of_numbers (avg : ℝ) (sum1 sum2 sum3 : ℝ) (N : ℝ) :
  avg = 3.95 →
  sum1 = 2 * 3.8 →
  sum2 = 2 * 3.85 →
  sum3 = 2 * 4.200000000000001 →
  avg = (sum1 + sum2 + sum3) / N →
  N = 6 :=
by
  intros h_avg h_sum1 h_sum2 h_sum3 h_total
  sorry

end total_number_of_numbers_l162_16239


namespace min_value_f_l162_16271

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f_l162_16271


namespace part_I_part_II_l162_16261

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 4)

-- Part I
theorem part_I : f (Real.pi / 6) + f (-Real.pi / 6) = Real.sqrt 6 / 2 :=
by
  sorry

-- Part II
theorem part_II (x : ℝ) (h : f x = Real.sqrt 2 / 3) : Real.sin (2 * x) = 5 / 9 :=
by
  sorry

end part_I_part_II_l162_16261


namespace estimate_flight_time_around_earth_l162_16254

theorem estimate_flight_time_around_earth 
  (radius : ℝ) 
  (speed : ℝ)
  (h_radius : radius = 6000) 
  (h_speed : speed = 600) 
  : abs (20 * Real.pi - 63) < 1 :=
by
  sorry

end estimate_flight_time_around_earth_l162_16254


namespace problem1_problem2_l162_16273

variables (a b : ℝ)

theorem problem1 : ((a^2)^3 / (-a)^2) = a^4 :=
sorry

theorem problem2 : ((a + 2 * b) * (a + b) - 3 * a * (a + b)) = -2 * a^2 + 2 * b^2 :=
sorry

end problem1_problem2_l162_16273


namespace rosy_fish_count_l162_16233

theorem rosy_fish_count (L R T : ℕ) (hL : L = 10) (hT : T = 19) : R = T - L := by
  sorry

end rosy_fish_count_l162_16233


namespace find_f_2_l162_16242

def f (a b x : ℝ) := a * x^3 - b * x + 1

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = -1) : f a b 2 = 3 :=
by
  sorry

end find_f_2_l162_16242


namespace car_trip_problem_l162_16221

theorem car_trip_problem (a b c : ℕ) (x : ℕ) 
(h1 : 1 ≤ a) 
(h2 : a + b + c ≤ 9)
(h3 : 100 * b + 10 * c + a - 100 * a - 10 * b - c = 60 * x) 
: a^2 + b^2 + c^2 = 14 := 
by
  sorry

end car_trip_problem_l162_16221


namespace triangle_angle_sixty_degrees_l162_16206

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) : 
  ∃ (θ : ℝ), θ = 60 ∧ ∃ (a b c : ℝ), a * b * c ≠ 0 ∧ ∀ {α β γ : ℝ}, (a + b + c = α + β + γ + θ) := 
sorry

end triangle_angle_sixty_degrees_l162_16206


namespace number_of_divisors_36_l162_16234

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l162_16234


namespace find_m_n_sum_l162_16275

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem find_m_n_sum (x₀ : ℝ) (m n : ℤ) 
  (hmn_adj : n = m + 1) 
  (hx₀_zero : f x₀ = 0) 
  (hx₀_interval : (m : ℝ) < x₀ ∧ x₀ < (n : ℝ)) :
  m + n = 1 :=
sorry

end find_m_n_sum_l162_16275


namespace intersection_complement_l162_16287

open Set

noncomputable def N := {x : ℕ | true}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def C_N (B : Set ℕ) : Set ℕ := {n ∈ N | n ∉ B}

theorem intersection_complement :
  A ∩ (C_N B) = {1} :=
by
  sorry

end intersection_complement_l162_16287


namespace divisibility_by_7_l162_16247

theorem divisibility_by_7 (n : ℕ) (h : 0 < n) : 7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) :=
sorry

end divisibility_by_7_l162_16247


namespace find_y_l162_16297

variable (a b y : ℝ)
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(3 * b) = a^b * y^b)

theorem find_y : y = 27 * a^2 :=
  by sorry

end find_y_l162_16297


namespace rational_numbers_satisfying_conditions_l162_16256

theorem rational_numbers_satisfying_conditions :
  (∃ n : ℕ, n = 166 ∧ ∀ (m : ℚ),
  abs m < 500 → (∃ x : ℤ, 3 * x^2 + m * x + 25 = 0) ↔ n = 166)
:=
sorry

end rational_numbers_satisfying_conditions_l162_16256


namespace least_five_digit_congruent_to_6_mod_19_l162_16236

theorem least_five_digit_congruent_to_6_mod_19 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 19 = 6 ∧ n = 10011 :=
by
  sorry

end least_five_digit_congruent_to_6_mod_19_l162_16236


namespace inverse_f_of_7_l162_16282

def f (x : ℝ) : ℝ := 2 * x^2 + 3

theorem inverse_f_of_7:
  ∀ y : ℝ, f (7) = y ↔ y = 101 :=
by
  sorry

end inverse_f_of_7_l162_16282


namespace value_of_b_l162_16284

theorem value_of_b (a b : ℕ) (r : ℝ) (h₁ : a = 2020) (h₂ : r = a / b) (h₃ : r = 0.5) : b = 4040 := 
by
  -- Hint: The proof takes steps to transform the conditions using basic algebraic manipulations.
  sorry

end value_of_b_l162_16284


namespace sin_of_angle_l162_16265

theorem sin_of_angle (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1/3) :
  Real.sin (2*θ + Real.pi/2) = -7/9 :=
by
  sorry

end sin_of_angle_l162_16265


namespace expenditure_ratio_l162_16244

theorem expenditure_ratio (I_A I_B E_A E_B : ℝ) (h1 : I_A / I_B = 5 / 6)
  (h2 : I_B = 7200) (h3 : 1800 = I_A - E_A) (h4 : 1600 = I_B - E_B) :
  E_A / E_B = 3 / 4 :=
sorry

end expenditure_ratio_l162_16244


namespace num_of_nickels_l162_16251

theorem num_of_nickels (n : ℕ) (h1 : n = 17) (h2 : (17 * n) - 1 = 18 * (n - 1)) : n = 17 → 17 * n = 289 → ∃ k, k = 2 :=
by 
  intros hn hv
  sorry

end num_of_nickels_l162_16251


namespace a_minus_b_eq_one_l162_16258

variable (a b : ℕ)

theorem a_minus_b_eq_one
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.sqrt 18 = a * Real.sqrt 2) 
  (h4 : Real.sqrt 8 = 2 * Real.sqrt b) : 
  a - b = 1 := 
sorry

end a_minus_b_eq_one_l162_16258


namespace combined_distance_correct_l162_16276

-- Define the conditions
def wheelA_rotations_per_minute := 20
def wheelA_distance_per_rotation_cm := 35
def wheelB_rotations_per_minute := 30
def wheelB_distance_per_rotation_cm := 50

-- Calculate distances in meters
def wheelA_distance_per_minute_m :=
  (wheelA_rotations_per_minute * wheelA_distance_per_rotation_cm) / 100

def wheelB_distance_per_minute_m :=
  (wheelB_rotations_per_minute * wheelB_distance_per_rotation_cm) / 100

def wheelA_distance_per_hour_m :=
  wheelA_distance_per_minute_m * 60

def wheelB_distance_per_hour_m :=
  wheelB_distance_per_minute_m * 60

def combined_distance_per_hour_m :=
  wheelA_distance_per_hour_m + wheelB_distance_per_hour_m

theorem combined_distance_correct : combined_distance_per_hour_m = 1320 := by
  -- skip the proof here with sorry
  sorry

end combined_distance_correct_l162_16276


namespace bob_grade_is_35_l162_16270

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

theorem bob_grade_is_35 : bob_grade = 35 :=
by
  -- Proof will go here
  sorry

end bob_grade_is_35_l162_16270


namespace book_original_price_l162_16277

noncomputable def original_price : ℝ := 420 / 1.40

theorem book_original_price (new_price : ℝ) (percentage_increase : ℝ) : 
  new_price = 420 → percentage_increase = 0.40 → original_price = 300 :=
by
  intros h1 h2
  exact sorry

end book_original_price_l162_16277


namespace james_total_vegetables_l162_16235

def james_vegetable_count (a b c d e : ℕ) : ℕ :=
  a + b + c + d + e

theorem james_total_vegetables 
    (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
    a = 22 → b = 18 → c = 15 → d = 10 → e = 12 →
    james_vegetable_count a b c d e = 77 :=
by
  intros ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end james_total_vegetables_l162_16235
