import Mathlib

namespace min_choir_members_l138_138775

theorem min_choir_members (n : ℕ) : 
  (∀ (m : ℕ), m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) → 
  n = 990 :=
by
  sorry

end min_choir_members_l138_138775


namespace building_height_l138_138456

noncomputable def height_of_building (flagpole_height shadow_of_flagpole shadow_of_building : ℝ) : ℝ :=
  (flagpole_height / shadow_of_flagpole) * shadow_of_building

theorem building_height : height_of_building 18 45 60 = 24 := by {
  sorry
}

end building_height_l138_138456


namespace intersection_of_M_and_N_is_correct_l138_138673

-- Definitions according to conditions
def M : Set ℤ := {-4, -2, 0, 2, 4, 6}
def N : Set ℤ := {x | -3 ≤ x ∧ x ≤ 4}

-- Proof statement
theorem intersection_of_M_and_N_is_correct : (M ∩ N) = {-2, 0, 2, 4} := by
  sorry

end intersection_of_M_and_N_is_correct_l138_138673


namespace total_tickets_l138_138163

-- Definitions based on given conditions
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def additional_tickets : ℕ := 6

-- Proof statement (only statement, proof is not required)
theorem total_tickets : (initial_tickets - spent_tickets + additional_tickets = 30) :=
  sorry

end total_tickets_l138_138163


namespace crates_probability_numerator_l138_138039

theorem crates_probability_numerator {a b c m n : ℕ} (h1 : 3 * a + 4 * b + 6 * c = 50) (h2 : a + b + c = 12)
(h3 : Nat.gcd m n = 1) (h4 : ∃ k, m = 30690 * k ∧ n = 531441 * k) : m = 10230 :=
by {
  -- Problem translation and conditions
  sorry
}

end crates_probability_numerator_l138_138039


namespace min_value_of_y_l138_138096

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (∃ y : ℝ, y = 1 / a + 4 / b ∧ (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ y)) ∧ 
  (∀ y : ℝ, y = 1 / a + 4 / b → y ≥ 9) :=
sorry

end min_value_of_y_l138_138096


namespace four_digit_numbers_permutations_l138_138104

theorem four_digit_numbers_permutations (a b : ℕ) (h1 : a = 3) (h2 : b = 0) : 
  (if a = 3 ∧ b = 0 then 3 else 0) = 3 :=
by
  sorry

end four_digit_numbers_permutations_l138_138104


namespace bucket_initial_amount_l138_138158

theorem bucket_initial_amount (A B : ℝ) 
  (h1 : A - 6 = (1 / 3) * (B + 6)) 
  (h2 : B - 6 = (1 / 2) * (A + 6)) : 
  A = 13.2 := 
sorry

end bucket_initial_amount_l138_138158


namespace train_pass_time_l138_138181

-- Definitions based on conditions
def train_length : Float := 250
def pole_time : Float := 10
def platform_length : Float := 1250
def incline_angle : Float := 5 -- degrees
def speed_reduction_factor : Float := 0.75

-- The statement to be proved
theorem train_pass_time :
  let original_speed := train_length / pole_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  let time_to_pass_platform := total_distance / incline_speed
  time_to_pass_platform = 80 := by
  simp [train_length, pole_time, platform_length, incline_angle, speed_reduction_factor]
  sorry

end train_pass_time_l138_138181


namespace find_n_value_l138_138902

theorem find_n_value : (15 * 25 + 20 * 5) = (10 * 25 + 45 * 5) := 
  sorry

end find_n_value_l138_138902


namespace planes_parallel_l138_138095

variables {a b c : Type} {α β γ : Type}
variables (h_lines : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Conditions based on the propositions
variables (h1 : parallel α γ)
variables (h2 : parallel β γ)

-- Theorem to prove
theorem planes_parallel (h1: parallel α γ) (h2 : parallel β γ) : parallel α β := 
sorry

end planes_parallel_l138_138095


namespace marble_problem_l138_138958

-- Defining the problem in Lean statement
theorem marble_problem 
  (m : ℕ) (n k : ℕ) (hx : m = 220) (hy : n = 20) : 
  (∀ x : ℕ, (k = n + x) → (m / n = 11) → (m / k = 10)) → (x = 2) :=
by {
  sorry
}

end marble_problem_l138_138958


namespace fewest_toothpicks_proof_l138_138203

noncomputable def fewest_toothpicks_to_remove (total_toothpicks : ℕ) (additional_row_and_column : ℕ) (triangles : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) (max_destroyed_per_toothpick : ℕ) (horizontal_toothpicks : ℕ) : ℕ :=
  horizontal_toothpicks

theorem fewest_toothpicks_proof 
  (total_toothpicks : ℕ := 40) 
  (additional_row_and_column : ℕ := 1) 
  (triangles : ℕ := 35) 
  (upward_triangles : ℕ := 15) 
  (downward_triangles : ℕ := 10)
  (max_destroyed_per_toothpick : ℕ := 1)
  (horizontal_toothpicks : ℕ := 15) :
  fewest_toothpicks_to_remove total_toothpicks additional_row_and_column triangles upward_triangles downward_triangles max_destroyed_per_toothpick horizontal_toothpicks = 15 := 
by 
  sorry

end fewest_toothpicks_proof_l138_138203


namespace value_of_x_l138_138371

theorem value_of_x (x c m n : ℝ) (hne: m≠n) (hneq : c ≠ 0) 
  (h1: c = 3) (h2: m = 2) (h3: n = 5)
  (h4: (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) : 
  x = -11 := by
  sorry

end value_of_x_l138_138371


namespace cranberries_left_in_bog_l138_138326

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l138_138326


namespace product_of_integers_l138_138144

theorem product_of_integers :
  ∃ (A B C : ℤ), A + B + C = 33 ∧ C = 3 * B ∧ A = C - 23 ∧ A * B * C = 192 :=
by
  sorry

end product_of_integers_l138_138144


namespace abs_neg_eq_iff_nonpos_l138_138107

theorem abs_neg_eq_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by sorry

end abs_neg_eq_iff_nonpos_l138_138107


namespace sqrt_domain_l138_138742

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l138_138742


namespace geom_seq_sum_first_four_terms_l138_138871

noncomputable def sum_first_n_terms_geom (a₁ q: ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_first_four_terms
  (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (h₂ : a₁ * q^3 = 27) :
  sum_first_n_terms_geom a₁ q 4 = 40 :=
by
  sorry

end geom_seq_sum_first_four_terms_l138_138871


namespace range_q_l138_138701

def q (x : ℝ) : ℝ :=
  if (Nat.floor x).prime then x + 3
  else q (Int.gpf (Nat.floor x)) + 2 * (x + 1 - Nat.floor x)

theorem range_q : set.range q = set.Ico 5 18 :=
by
  sorry

end range_q_l138_138701


namespace rain_forest_animals_l138_138725

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l138_138725


namespace ratio_of_x_to_y_l138_138865

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) : 
  x / y = Real.sqrt (17 / 8) :=
by
  sorry

end ratio_of_x_to_y_l138_138865


namespace calculate_amount_left_l138_138397

def base_income : ℝ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℝ := 500
def utilities : ℝ := 100
def food : ℝ := 300
def miscellaneous_percentage : ℝ := 0.10
def savings_percentage : ℝ := 0.07
def investment_percentage : ℝ := 0.05
def medical_expense : ℝ := 250
def tax_percentage : ℝ := 0.15

def total_income (base_income : ℝ) (bonus_percentage : ℝ) : ℝ :=
  base_income + (bonus_percentage * base_income)

def taxes (base_income : ℝ) (tax_percentage : ℝ) : ℝ :=
  tax_percentage * base_income

def total_fixed_expenses (rent : ℝ) (utilities : ℝ) (food : ℝ) : ℝ :=
  rent + utilities + food

def public_transport_expense (total_income : ℝ) (public_transport_percentage : ℝ) : ℝ :=
  public_transport_percentage * total_income

def miscellaneous_expense (total_income : ℝ) (miscellaneous_percentage : ℝ) : ℝ :=
  miscellaneous_percentage * total_income

def variable_expenses (public_transport_expense : ℝ) (miscellaneous_expense : ℝ) : ℝ :=
  public_transport_expense + miscellaneous_expense

def savings (total_income : ℝ) (savings_percentage : ℝ) : ℝ :=
  savings_percentage * total_income

def investment (total_income : ℝ) (investment_percentage : ℝ) : ℝ :=
  investment_percentage * total_income

def total_savings_investments (savings : ℝ) (investment : ℝ) : ℝ :=
  savings + investment

def total_expenses_contributions 
  (fixed_expenses : ℝ) 
  (variable_expenses : ℝ) 
  (medical_expense : ℝ) 
  (total_savings_investments : ℝ) : ℝ :=
  fixed_expenses + variable_expenses + medical_expense + total_savings_investments

def amount_left (income_after_taxes : ℝ) (total_expenses_contributions : ℝ) : ℝ :=
  income_after_taxes - total_expenses_contributions

theorem calculate_amount_left 
  (base_income : ℝ)
  (bonus_percentage : ℝ)
  (public_transport_percentage : ℝ)
  (rent : ℝ)
  (utilities : ℝ)
  (food : ℝ)
  (miscellaneous_percentage : ℝ)
  (savings_percentage : ℝ)
  (investment_percentage : ℝ)
  (medical_expense : ℝ)
  (tax_percentage : ℝ)
  (total_income : ℝ := total_income base_income bonus_percentage)
  (taxes : ℝ := taxes base_income tax_percentage)
  (income_after_taxes : ℝ := total_income - taxes)
  (fixed_expenses : ℝ := total_fixed_expenses rent utilities food)
  (public_transport_expense : ℝ := public_transport_expense total_income public_transport_percentage)
  (miscellaneous_expense : ℝ := miscellaneous_expense total_income miscellaneous_percentage)
  (variable_expenses : ℝ := variable_expenses public_transport_expense miscellaneous_expense)
  (savings : ℝ := savings total_income savings_percentage)
  (investment : ℝ := investment total_income investment_percentage)
  (total_savings_investments : ℝ := total_savings_investments savings investment)
  (total_expenses_contributions : ℝ := total_expenses_contributions fixed_expenses variable_expenses medical_expense total_savings_investments)
  : amount_left income_after_taxes total_expenses_contributions = 229 := 
sorry

end calculate_amount_left_l138_138397


namespace skylar_current_age_l138_138022

theorem skylar_current_age (started_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) (h1 : started_age = 17) (h2 : annual_donation = 8000) (h3 : total_donation = 440000) : 
  (started_age + total_donation / annual_donation = 72) :=
by
  sorry

end skylar_current_age_l138_138022


namespace units_digit_probability_l138_138178

noncomputable def probability_units_digit_less_than_seven : ℚ :=
  let favorable_outcomes := 7
  let total_possible_outcomes := 10
  favorable_outcomes / total_possible_outcomes

theorem units_digit_probability :
  probability_units_digit_less_than_seven = 7 / 10 :=
sorry

end units_digit_probability_l138_138178


namespace convert_base_10_to_base_5_l138_138341

theorem convert_base_10_to_base_5 :
  (256 : ℕ) = 2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 :=
by
  sorry

end convert_base_10_to_base_5_l138_138341


namespace problem_statement_l138_138745

-- Definitions for the conditions in the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p
def has_three_divisors (k : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ k = p^2

-- Given conditions
def m : ℕ := 3 -- the smallest odd prime
def n : ℕ := 49 -- the largest integer less than 50 with exactly three positive divisors

-- The proof statement
theorem problem_statement : m + n = 52 :=
by sorry

end problem_statement_l138_138745


namespace susie_investment_l138_138723

theorem susie_investment :
  ∃ x : ℝ, x * (1 + 0.04)^3 + (2000 - x) * (1 + 0.06)^3 = 2436.29 → x = 820 :=
by
  sorry

end susie_investment_l138_138723


namespace three_nabla_four_l138_138527

noncomputable def modified_operation (a b : ℝ) : ℝ :=
  (a + b^2) / (1 + a * b^2)

theorem three_nabla_four : modified_operation 3 4 = 19 / 49 := 
  by 
  sorry

end three_nabla_four_l138_138527


namespace count_integers_congruent_to_7_mod_13_l138_138522

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l138_138522


namespace solve_triangle_l138_138265

theorem solve_triangle (a b m₁ m₂ k₃ : ℝ) (h1 : a = m₂ / Real.sin γ) (h2 : b = m₁ / Real.sin γ) : 
  a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := 
  by 
  sorry

end solve_triangle_l138_138265


namespace number_of_classes_l138_138616

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 := by
  sorry

end number_of_classes_l138_138616


namespace B_subset_A_implies_m_le_5_l138_138674

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def setB (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

theorem B_subset_A_implies_m_le_5 :
  B ⊆ A → (∀ k : ℝ, k ∈ setB m → k ∈ setA) → m ≤ 5 :=
by
  sorry

end B_subset_A_implies_m_le_5_l138_138674


namespace mod_3_pow_2040_eq_1_mod_5_l138_138048

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l138_138048


namespace canoes_more_than_kayaks_l138_138459

noncomputable def canoes_and_kayaks (C K : ℕ) : Prop :=
  (2 * C = 3 * K) ∧ (12 * C + 18 * K = 504) ∧ (C - K = 7)

theorem canoes_more_than_kayaks (C K : ℕ) (h : canoes_and_kayaks C K) : C - K = 7 :=
sorry

end canoes_more_than_kayaks_l138_138459


namespace circumference_in_scientific_notation_l138_138915

noncomputable def circumference_m : ℝ := 4010000

noncomputable def scientific_notation (m: ℝ) : Prop :=
  m = 4.01 * 10^6

theorem circumference_in_scientific_notation : scientific_notation circumference_m :=
by
  sorry

end circumference_in_scientific_notation_l138_138915


namespace simplify_trig_expression_l138_138106

open Real

theorem simplify_trig_expression (theta : ℝ) (h : 0 < theta ∧ theta < π / 4) :
  sqrt (1 - 2 * sin (π + theta) * sin (3 * π / 2 - theta)) = cos theta - sin theta :=
sorry

end simplify_trig_expression_l138_138106


namespace find_angle_ACB_l138_138686

open EuclideanGeometry

variable (A B C D : Point)
variable (h_parallel : ParallelLine DC AB)
variable (h_angle_DCA : Angle DCA = 55º)
variable (h_angle_ABC : Angle ABC = 60º)

theorem find_angle_ACB :
  Angle ACB = 65º :=
by
  sorry

end find_angle_ACB_l138_138686


namespace total_heads_l138_138778

theorem total_heads (h : ℕ) (c : ℕ) (total_feet : ℕ) 
  (h_count : h = 30)
  (hen_feet : h * 2 + c * 4 = total_feet)
  (total_feet_val : total_feet = 140) 
  : h + c = 50 :=
by
  sorry

end total_heads_l138_138778


namespace man_age_twice_son_age_in_two_years_l138_138172

theorem man_age_twice_son_age_in_two_years :
  ∀ (S M X : ℕ), S = 30 → M = S + 32 → (M + X = 2 * (S + X)) → X = 2 :=
by
  intros S M X hS hM h
  sorry

end man_age_twice_son_age_in_two_years_l138_138172


namespace points_after_perfect_games_l138_138175

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l138_138175


namespace winston_cents_left_l138_138451

def initial_amount_quarters (num_quarters : Nat) (value_per_quarter : Nat) : Nat :=
-num_quarters * value_per_quarter

def amount_spent (dollar_spent : Nat) : Nat := 
-dollar_spent * 50

theorem winston_cents_left (quarters : Nat) (value_per_quarter : Nat) (dollar_spent : Nat) : 
quarters = 14 ∧ value_per_quarter = 25 ∧ dollar_spent = 1/2 → 
initial_amount_quarters quarters value_per_quarter - amount_spent 1/2 = 300 :=
by 
intro h
cases h with h1 h_value_per_quarter 
cases h_value_per_quarter with h2 h_dollar_spent 
have h_amount_quarters : initial_amount_quarters 14 25 = 350 := 
by norm_num
have h_amount_spent : amount_spent 1 = 50 := 
by norm_num
rw [h_amount_quarters, h_amount_spent] 
norm_num
sorry

end winston_cents_left_l138_138451


namespace train_cross_time_l138_138245

theorem train_cross_time (length : ℝ) (speed_kmh : ℝ) (expected_time : ℝ) 
  (h_length : length = 140)
  (h_speed_kmh : speed_kmh = 108) 
  (h_expected_time : expected_time = 4.67) : 
  let speed_ms := speed_kmh * 1000 / 3600 in
  let time := length / speed_ms in
  time = expected_time :=
by 
  sorry

end train_cross_time_l138_138245


namespace product_of_integers_l138_138755

theorem product_of_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) : x * y = 168 := by
  sorry

end product_of_integers_l138_138755


namespace integer_solutions_l138_138796

theorem integer_solutions (n : ℤ) : ∃ m : ℤ, n^2 + 15 = m^2 ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 :=
by
  sorry

end integer_solutions_l138_138796


namespace hall_width_l138_138240

theorem hall_width (w : ℝ) (length height cost_per_m2 total_expenditure : ℝ)
  (h_length : length = 20)
  (h_height : height = 5)
  (h_cost : cost_per_m2 = 50)
  (h_expenditure : total_expenditure = 47500)
  (h_area : total_expenditure = cost_per_m2 * (2 * (length * w) + 2 * (length * height) + 2 * (w * height))) :
  w = 15 := 
sorry

end hall_width_l138_138240


namespace largest_int_mod_6_less_than_100_l138_138823

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l138_138823


namespace problem_l138_138851

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem problem (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1))
                (h2 : f 1 = 0) :
  f 0 = -2 ∧ ∀ x : ℝ, f x = x^2 + x - 2 := by
  sorry

end problem_l138_138851


namespace three_pow_2040_mod_5_l138_138043

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l138_138043


namespace abs_sum_le_abs_one_plus_mul_l138_138660

theorem abs_sum_le_abs_one_plus_mul {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  |x + y| ≤ |1 + x * y| :=
sorry

end abs_sum_le_abs_one_plus_mul_l138_138660


namespace digit_properties_l138_138917

theorem digit_properties {x : ℕ} 
  (h1: 12 * x - 21 * x = 36)   -- Condition: difference between numbers is 36
  (h2: ∃ k, 2 * k = x)        -- Condition: ratio of 1:2 between digits
  (hx : x < 10)               -- Condition: digits have to be less than 10
  : ((10 * x + 2 * x) % 10 + ((2 * 10 * x) % 10)) - ((2 * (10 * x) % 10 ) - x) = 6 := 
begin
  sorry
end

end digit_properties_l138_138917


namespace slices_eaten_l138_138259

theorem slices_eaten (slices_cheese : ℕ) (slices_pepperoni : ℕ) (slices_left_per_person : ℕ) (phil_andre_slices_left : ℕ) :
  (slices_cheese + slices_pepperoni = 22) →
  (slices_left_per_person = 2) →
  (phil_andre_slices_left = 2 + 2) →
  (slices_cheese + slices_pepperoni - phil_andre_slices_left = 18) :=
by
  intros
  sorry

end slices_eaten_l138_138259


namespace room_length_l138_138409

-- Defining conditions
def room_height : ℝ := 5
def room_width : ℝ := 7
def door_height : ℝ := 3
def door_width : ℝ := 1
def num_doors : ℝ := 2
def window1_height : ℝ := 1.5
def window1_width : ℝ := 2
def window2_height : ℝ := 1.5
def window2_width : ℝ := 1
def num_window2 : ℝ := 2
def paint_cost_per_sq_m : ℝ := 3
def total_paint_cost : ℝ := 474

-- Defining the problem as a statement to prove x (room length) is 10 meters
theorem room_length {x : ℝ} 
  (H1 : total_paint_cost = paint_cost_per_sq_m * ((2 * (x * room_height) + 2 * (room_width * room_height)) - (num_doors * (door_height * door_width) + (window1_height * window1_width) + num_window2 * (window2_height * window2_width)))) 
  : x = 10 :=
by 
  sorry

end room_length_l138_138409


namespace tangent_lines_parallel_l138_138935

-- Definitions and conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2
def line (x : ℝ) : ℝ := 4 * x - 1
def tangent_line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem tangent_lines_parallel (tangent_line : ℝ → ℝ) :
  (∃ x : ℝ, tangent_line_eq 4 (-1) 0 x (curve x)) ∧ 
  (∃ x : ℝ, tangent_line_eq 4 (-1) (-4) x (curve x)) :=
sorry

end tangent_lines_parallel_l138_138935


namespace minimum_value_property_l138_138199

noncomputable def min_value_expression (x : ℝ) (h : x > 10) : ℝ :=
  (x^2 + 36) / (x - 10)

noncomputable def min_value : ℝ := 4 * Real.sqrt 34 + 20

theorem minimum_value_property (x : ℝ) (h : x > 10) :
  min_value_expression x h >= min_value := by
  sorry

end minimum_value_property_l138_138199


namespace banana_price_l138_138771

theorem banana_price (b : ℝ) : 
    (∃ x : ℕ, 0.70 * x + b * (9 - x) = 5.60 ∧ x + (9 - x) = 9) → b = 0.60 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- equations to work with:
  -- 0.70 * x + b * (9 - x) = 5.60
  -- x + (9 - x) = 9
  sorry

end banana_price_l138_138771


namespace element_with_36_36_percentage_is_O_l138_138842

-- Define the chemical formula N2O and atomic masses
def chemical_formula : String := "N2O"
def atomic_mass_N : Float := 14.01
def atomic_mass_O : Float := 16.00

-- Define the molar mass of N2O
def molar_mass_N2O : Float := (2 * atomic_mass_N) + (1 * atomic_mass_O)

-- Mass of nitrogen in N2O
def mass_N_in_N2O : Float := 2 * atomic_mass_N

-- Mass of oxygen in N2O
def mass_O_in_N2O : Float := 1 * atomic_mass_O

-- Mass percentages
def mass_percentage_N : Float := (mass_N_in_N2O / molar_mass_N2O) * 100
def mass_percentage_O : Float := (mass_O_in_N2O / molar_mass_N2O) * 100

-- Prove that the element with a mass percentage of 36.36% is oxygen
theorem element_with_36_36_percentage_is_O : mass_percentage_O = 36.36 := sorry

end element_with_36_36_percentage_is_O_l138_138842


namespace correct_statement_l138_138056

-- Definition of quadrants
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_third_quadrant (θ : ℝ) : Prop := -180 < θ ∧ θ < -90
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement of the problem
theorem correct_statement : is_obtuse_angle θ → is_second_quadrant θ :=
by sorry

end correct_statement_l138_138056


namespace base_k_to_decimal_is_5_l138_138167

theorem base_k_to_decimal_is_5 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 42) : k = 5 := sorry

end base_k_to_decimal_is_5_l138_138167


namespace digit_in_2017th_place_l138_138070

def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_in_2017th_place :
  digit_at_position 2017 = 7 :=
by sorry

end digit_in_2017th_place_l138_138070


namespace largest_nonrepresentable_by_17_11_l138_138241

/--
In the USA, standard letter-size paper is 8.5 inches wide and 11 inches long. The largest integer that cannot be written as a sum of a whole number (possibly zero) of 17's and a whole number (possibly zero) of 11's is 159.
-/
theorem largest_nonrepresentable_by_17_11 : 
  ∀ (a b : ℕ), (∀ (n : ℕ), n = 17 * a + 11 * b -> n ≠ 159) ∧ 
               ¬ (∃ (a b : ℕ), 17 * a + 11 * b = 159) :=
by
  sorry

end largest_nonrepresentable_by_17_11_l138_138241


namespace line_equation_l138_138302

variable (θ : ℝ) (b : ℝ) (y x : ℝ)

-- Conditions: 
-- Slope angle θ = 45°
def slope_angle_condition : θ = 45 := by
  sorry

-- Y-intercept b = 2
def y_intercept_condition : b = 2 := by
  sorry

-- Given these conditions, we want to prove the line equation
theorem line_equation (x : ℝ) (θ : ℝ) (b : ℝ) :
  θ = 45 → b = 2 → y = x + 2 := by
  sorry

end line_equation_l138_138302


namespace sum_of_interior_edges_l138_138315

def frame_width : ℝ := 1
def outer_length : ℝ := 5
def frame_area : ℝ := 18
def inner_length1 : ℝ := outer_length - 2 * frame_width

/-- Given conditions and required to prove:
1. The frame is made of one-inch-wide pieces of wood.
2. The area of just the frame is 18 square inches.
3. One of the outer edges of the frame is 5 inches long.
Prove: The sum of the lengths of the four interior edges is 14 inches.
-/
theorem sum_of_interior_edges (inner_length2 : ℝ) 
  (h1 : (outer_length * (inner_length2 + 2) - inner_length1 * inner_length2) = frame_area)
  (h2 : (inner_length2 - 2) / 2 = 1) : 
  inner_length1 + inner_length1 + inner_length2 + inner_length2 = 14 :=
by
  sorry

end sum_of_interior_edges_l138_138315


namespace inscribed_squares_ratio_l138_138152

theorem inscribed_squares_ratio (x y : ℝ) 
  (h₁ : 5^2 + 12^2 = 13^2)
  (h₂ : x = 144 / 17)
  (h₃ : y = 5) :
  x / y = 144 / 85 :=
by
  sorry

end inscribed_squares_ratio_l138_138152


namespace max_distance_origin_perpendicular_bisector_l138_138847

theorem max_distance_origin_perpendicular_bisector :
  ∀ (k m : ℝ), k ≠ 0 → 
  (|m| = Real.sqrt (1 + k^2)) → 
  ∃ (d : ℝ), d = 4 / 3 :=
by
  sorry

end max_distance_origin_perpendicular_bisector_l138_138847


namespace scores_are_sample_l138_138037

-- Define the total number of students
def total_students : ℕ := 5000

-- Define the number of selected students for sampling
def selected_students : ℕ := 200

-- Define a predicate that checks if a selection is a sample
def is_sample (total selected : ℕ) : Prop :=
  selected < total

-- The proposition that needs to be proven
theorem scores_are_sample : is_sample total_students selected_students := 
by 
  -- Proof of the theorem is omitted.
  sorry

end scores_are_sample_l138_138037


namespace sequence_a_l138_138672

theorem sequence_a (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n ≥ 2, a n / a (n + 1) + a n / a (n - 1) = 2) :
  a 12 = 1 / 6 :=
sorry

end sequence_a_l138_138672


namespace quadratic_has_real_root_l138_138036

theorem quadratic_has_real_root (a b : ℝ) : (∃ x : ℝ, x^2 + a * x + b = 0) :=
by
  -- To use contradiction, we assume the negation
  have h : ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry
  -- By contradiction, this assumption should lead to a contradiction
  sorry

end quadratic_has_real_root_l138_138036


namespace triangle_area_from_squares_l138_138991

/-- 
Given three squares with areas 36, 64, and 100, 
prove that the area of the triangle formed by their side lengths is 24.
-/
theorem triangle_area_from_squares :
  let a := 6
  let b := 8
  let c := 10
  let area_a := a ^ 2
  let area_b := b ^ 2
  let area_c := c ^ 2
  (area_a = 36) →
  (area_b = 64) →
  (area_c = 100) →
  (a^2 + b^2 = c^2) →
  (1 / 2 * a * b = 24) :=
by
  intros a b c area_a area_b area_c h1 h2 h3 h4
  exact h4
  sorry

end triangle_area_from_squares_l138_138991


namespace boys_on_trip_l138_138710

theorem boys_on_trip (B G : ℕ) 
    (h1 : G = B + (2 / 5 : ℚ) * B) 
    (h2 : 1 + 1 + 1 + B + G = 123) : 
    B = 50 := 
by 
  -- Proof skipped 
  sorry

end boys_on_trip_l138_138710


namespace machine_b_finishes_in_12_hours_l138_138705

noncomputable def machine_b_time : ℝ :=
  let rA := 1 / 4  -- rate of Machine A
  let rC := 1 / 6  -- rate of Machine C
  let rTotalTogether := 1 / 2  -- rate of all machines working together
  let rB := (rTotalTogether - rA - rC)  -- isolate the rate of Machine B
  1 / rB  -- time for Machine B to finish the job

theorem machine_b_finishes_in_12_hours : machine_b_time = 12 :=
by
  sorry

end machine_b_finishes_in_12_hours_l138_138705


namespace least_multiple_of_11_not_lucky_l138_138310

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l138_138310


namespace largest_n_unique_k_l138_138747

theorem largest_n_unique_k : ∃! (n : ℕ), ∃ (k : ℤ),
  (7 / 16 : ℚ) < (n : ℚ) / (n + k : ℚ) ∧ (n : ℚ) / (n + k : ℚ) < (8 / 17 : ℚ) ∧ n = 112 := 
sorry

end largest_n_unique_k_l138_138747


namespace fifth_number_in_21st_row_l138_138482

theorem fifth_number_in_21st_row : 
  let nth_odd_number (n : ℕ) := 2 * n - 1 
  let sum_first_n_rows (n : ℕ) := n * (n + (n - 1))
  nth_odd_number 405 = 809 := 
by
  sorry

end fifth_number_in_21st_row_l138_138482


namespace sqrt_domain_l138_138741

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l138_138741


namespace solve_inequality_l138_138135

theorem solve_inequality (x : ℝ) :
  (4 ≤ x^2 - 3 * x - 6 ∧ x^2 - 3 * x - 6 ≤ 2 * x + 8) ↔ (5 ≤ x ∧ x ≤ 7 ∨ x = -2) :=
by
  sorry

end solve_inequality_l138_138135


namespace num_sets_C_l138_138205

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

theorem num_sets_C : {C : Set ℕ // B ∪ C = A}.1.card = 4 := 
  sorry

end num_sets_C_l138_138205


namespace find_second_projection_l138_138211

noncomputable def second_projection (plane : Prop) (first_proj : Prop) (distance : ℝ) : Prop :=
∃ second_proj : Prop, true

theorem find_second_projection 
  (plane : Prop) 
  (first_proj : Prop) 
  (distance : ℝ) :
  ∃ second_proj : Prop, true :=
sorry

end find_second_projection_l138_138211


namespace total_egg_collection_l138_138332

theorem total_egg_collection (
  -- Conditions
  (Benjamin_collects : Nat) (h1 : Benjamin_collects = 6) 
  (Carla_collects : Nat) (h2 : Carla_collects = 3 * Benjamin_collects) 
  (Trisha_collects : Nat) (h3 : Trisha_collects = Benjamin_collects - 4)
  ) : 
  -- Question and answer
  (Total_collects : Nat) (h_total : Total_collects = Benjamin_collects + Carla_collects + Trisha_collects) => 
  (Total_collects = 26) := 
  by
  sorry

end total_egg_collection_l138_138332


namespace no_triangles_with_geometric_progression_angles_l138_138520

theorem no_triangles_with_geometric_progression_angles :
  ¬ ∃ (a r : ℕ), a ≥ 10 ∧ (a + a * r + a * r^2 = 180) ∧ (a ≠ a * r) ∧ (a ≠ a * r^2) ∧ (a * r ≠ a * r^2) :=
sorry

end no_triangles_with_geometric_progression_angles_l138_138520


namespace ratio_of_width_to_length_l138_138870

-- Definitions of length, width, perimeter
def l : ℕ := 10
def P : ℕ := 30

-- Define the condition for the width
def width_from_perimeter (l P : ℕ) : ℕ :=
  (P - 2 * l) / 2

-- Calculate the width using the given length and perimeter
def w : ℕ := width_from_perimeter l P

-- Theorem stating the ratio of width to length
theorem ratio_of_width_to_length : (w : ℚ) / l = 1 / 2 := by
  -- Proof steps will go here
  sorry

end ratio_of_width_to_length_l138_138870


namespace imaginary_part_of_product_l138_138848

def imaginary_unit : ℂ := Complex.I

def z : ℂ := 2 + imaginary_unit

theorem imaginary_part_of_product : (z * imaginary_unit).im = 2 := by
  sorry

end imaginary_part_of_product_l138_138848


namespace locus_of_centers_l138_138407

-- Statement of the problem
theorem locus_of_centers :
  ∀ (a b : ℝ),
    ((∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (3 - r)^2))) ↔ (4 * a^2 + 4 * b^2 - 25 = 0) := by
  sorry

end locus_of_centers_l138_138407


namespace repeating_decimals_sum_as_fraction_l138_138085

noncomputable def repeating_decimal_to_fraction (n : Int) (d : Nat) : Rat :=
  n / (10^d - 1)

theorem repeating_decimals_sum_as_fraction :
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  x1 + x2 + x3 = (283 / 11111 : Rat) :=
by
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  have : x1 = 0.2, by sorry
  have : x2 = 0.03, by sorry
  have : x3 = 0.0004, by sorry
  show x1 + x2 + x3 = 283 / 11111
  sorry

end repeating_decimals_sum_as_fraction_l138_138085


namespace determine_values_a_b_l138_138642

theorem determine_values_a_b (a b x : ℝ) (h₁ : x > 1)
  (h₂ : 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = (10 * (Real.log x)^2) / (Real.log a + Real.log b)) :
  b = a ^ ((5 + Real.sqrt 10) / 3) ∨ b = a ^ ((5 - Real.sqrt 10) / 3) :=
by sorry

end determine_values_a_b_l138_138642


namespace set_representation_l138_138351

def is_nat_star (n : ℕ) : Prop := n > 0
def satisfies_eqn (x y : ℕ) : Prop := y = 6 / (x + 3)

theorem set_representation :
  {p : ℕ × ℕ | is_nat_star p.fst ∧ is_nat_star p.snd ∧ satisfies_eqn p.fst p.snd } = { (3, 1) } :=
by
  sorry

end set_representation_l138_138351


namespace book_arrangement_count_l138_138857

theorem book_arrangement_count :
  let total_books := 7
  let identical_math_books := 3
  let identical_physics_books := 2
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2)) = 420 := 
by
  sorry

end book_arrangement_count_l138_138857


namespace parabola_equation_l138_138098

-- Define the conditions and the claim
theorem parabola_equation (p : ℝ) (hp : p > 0) (h_symmetry : -p / 2 = -1 / 2) : 
  (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = 2 * y) :=
by 
  sorry

end parabola_equation_l138_138098


namespace remainder_of_power_modulo_l138_138051

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l138_138051


namespace solve_for_s_l138_138973

noncomputable def compute_s : Set ℝ :=
  { s | ∀ (x : ℝ), (x ≠ -1) → ((s * x - 3) / (x + 1) = x ↔ x^2 + (1 - s) * x + 3 = 0) ∧
    ((1 - s) ^ 2 - 4 * 3 = 0) }

theorem solve_for_s (h : ∀ s ∈ compute_s, s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :
  compute_s = {1 + 2 * Real.sqrt 3, 1 - 2 * Real.sqrt 3} :=
by
  sorry

end solve_for_s_l138_138973


namespace claudia_total_earnings_l138_138339

def cost_per_beginner_class : Int := 15
def cost_per_advanced_class : Int := 20
def num_beginner_kids_saturday : Int := 20
def num_advanced_kids_saturday : Int := 10
def num_sibling_pairs : Int := 5
def sibling_discount : Int := 3

theorem claudia_total_earnings : 
  let beginner_earnings_saturday := num_beginner_kids_saturday * cost_per_beginner_class
  let advanced_earnings_saturday := num_advanced_kids_saturday * cost_per_advanced_class
  let total_earnings_saturday := beginner_earnings_saturday + advanced_earnings_saturday
  
  let num_beginner_kids_sunday := num_beginner_kids_saturday / 2
  let num_advanced_kids_sunday := num_advanced_kids_saturday / 2
  let beginner_earnings_sunday := num_beginner_kids_sunday * cost_per_beginner_class
  let advanced_earnings_sunday := num_advanced_kids_sunday * cost_per_advanced_class
  let total_earnings_sunday := beginner_earnings_sunday + advanced_earnings_sunday

  let total_earnings_no_discount := total_earnings_saturday + total_earnings_sunday

  let total_sibling_discount := num_sibling_pairs * 2 * sibling_discount
  
  let total_earnings := total_earnings_no_discount - total_sibling_discount
  total_earnings = 720 := 
by
  sorry

end claudia_total_earnings_l138_138339


namespace collinear_points_k_value_l138_138588

theorem collinear_points_k_value : 
  (∀ k : ℝ, ∃ (a : ℝ) (b : ℝ), ∀ (x : ℝ) (y : ℝ),
    ((x, y) = (1, -2) ∨ (x, y) = (3, 2) ∨ (x, y) = (6, k / 3)) → y = a * x + b) → k = 24 :=
by
sorry

end collinear_points_k_value_l138_138588


namespace largest_integer_is_222_l138_138906

theorem largest_integer_is_222
  (a b c d : ℤ)
  (h_distinct : a < b ∧ b < c ∧ c < d)
  (h_mean : (a + b + c + d) / 4 = 72)
  (h_min_a : a ≥ 21) 
  : d = 222 :=
sorry

end largest_integer_is_222_l138_138906


namespace students_study_both_l138_138061

-- Define variables and conditions
variable (total_students G B G_and_B : ℕ)
variable (G_percent B_percent : ℝ)
variable (total_students_eq : total_students = 300)
variable (G_percent_eq : G_percent = 0.8)
variable (B_percent_eq : B_percent = 0.5)
variable (G_eq : G = G_percent * total_students)
variable (B_eq : B = B_percent * total_students)
variable (students_eq : total_students = G + B - G_and_B)

-- Theorem statement
theorem students_study_both :
  G_and_B = 90 :=
by
  sorry

end students_study_both_l138_138061


namespace maximum_value_omega_l138_138237

theorem maximum_value_omega (ω : ℝ) (h : ∃! x, 0 < x ∧ x < π / 2 ∧ sin (ω * x) + 1 = 0) : ω ≤ 7 := 
by {
  sorry
}

end maximum_value_omega_l138_138237


namespace largest_int_less_than_100_mod_6_eq_4_l138_138820

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l138_138820


namespace max_value_x_sq_y_l138_138853

theorem max_value_x_sq_y (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end max_value_x_sq_y_l138_138853


namespace green_pens_l138_138921

theorem green_pens (blue_pens green_pens : ℕ) (ratio_blue_to_green : blue_pens / green_pens = 4 / 3) (total_blue : blue_pens = 16) : green_pens = 12 :=
by sorry

end green_pens_l138_138921


namespace sum_of_distances_l138_138394

theorem sum_of_distances (a b : ℤ) (k : ℕ) 
  (h1 : |k - a| + |(k + 1) - a| + |(k + 2) - a| + |(k + 3) - a| + |(k + 4) - a| + |(k + 5) - a| + |(k + 6) - a| = 609)
  (h2 : |k - b| + |(k + 1) - b| + |(k + 2) - b| + |(k + 3) - b| + |(k + 4) - b| + |(k + 5) - b| + |(k + 6) - b| = 721)
  (h3 : a + b = 192) :
  a = 1 ∨ a = 104 ∨ a = 191 := 
sorry

end sum_of_distances_l138_138394


namespace total_money_shared_l138_138168

def A_share (B : ℕ) : ℕ := B / 2
def B_share (C : ℕ) : ℕ := C / 2
def C_share : ℕ := 400

theorem total_money_shared (A B C : ℕ) (h1 : A = A_share B) (h2 : B = B_share C) (h3 : C = C_share) : A + B + C = 700 :=
by
  sorry

end total_money_shared_l138_138168


namespace balls_removal_l138_138147

theorem balls_removal (total_balls : ℕ) (percent_green initial_green initial_yellow remaining_percent : ℝ)
    (h_percent_green : percent_green = 0.7)
    (h_total_balls : total_balls = 600)
    (h_initial_green : initial_green = percent_green * total_balls)
    (h_initial_yellow : initial_yellow = total_balls - initial_green)
    (h_remaining_percent : remaining_percent = 0.6) :
    ∃ x : ℝ, (initial_green - x) / (total_balls - x) = remaining_percent ∧ x = 150 := 
by 
  sorry

end balls_removal_l138_138147


namespace max_notebooks_lucy_can_buy_l138_138125

-- Definitions given in the conditions
def lucyMoney : ℕ := 2145
def notebookCost : ℕ := 230

-- Theorem to prove the number of notebooks Lucy can buy
theorem max_notebooks_lucy_can_buy : lucyMoney / notebookCost = 9 := 
by
  sorry

end max_notebooks_lucy_can_buy_l138_138125


namespace range_of_a_for_negative_root_l138_138797

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) →
  - (1/2 : ℝ) < a ∧ a ≤ (1/16 : ℝ) :=
by
  sorry

end range_of_a_for_negative_root_l138_138797


namespace smallest_two_digit_number_product_12_l138_138438

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l138_138438


namespace repeating_decimal_sum_l138_138350

open Real

noncomputable def repeating_decimal_to_fraction (d: ℕ) : ℚ :=
  if d = 3 then 1/3 else if d = 7 then 7/99 else if d = 9 then 1/111 else 0 -- specific case of 3, 7, 9.

theorem repeating_decimal_sum:
  let x := repeating_decimal_to_fraction 3
  let y := repeating_decimal_to_fraction 7
  let z := repeating_decimal_to_fraction 9
  x + y + z = 499 / 1189 :=
by
  sorry -- Proof is omitted

end repeating_decimal_sum_l138_138350


namespace min_value_correct_l138_138269

noncomputable def min_value (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) : ℝ :=
(1 / m) + (2 / n)

theorem min_value_correct :
  ∃ m n : ℝ, ∃ h₁ : m > 0, ∃ h₂ : n > 0, ∃ h₃ : m + n = 1,
  min_value m n h₁ h₂ h₃ = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_correct_l138_138269


namespace cube_vertex_adjacency_l138_138460

noncomputable def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem cube_vertex_adjacency :
  ∀ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ), 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8 ∧
  beautiful_face v1 v2 v3 v4 ∧ beautiful_face v5 v6 v7 v8 ∧
  beautiful_face v1 v3 v5 v7 ∧ beautiful_face v2 v4 v6 v8 ∧
  beautiful_face v1 v2 v5 v6 ∧ beautiful_face v3 v4 v7 v8 →
  (v6 = 6 → (v1 = 2 ∧ v2 = 3 ∧ v3 = 5) ∨ 
   (v1 = 3 ∧ v2 = 5 ∧ v3 = 7) ∨ 
   (v1 = 2 ∧ v2 = 3 ∧ v3 = 7)) :=
sorry

end cube_vertex_adjacency_l138_138460


namespace find_a_value_l138_138984

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x + 1)

def slope_of_tangent_line (a : ℝ) : Prop :=
  (deriv (fun x => f x a) 1) = -1

theorem find_a_value : ∃ a : ℝ, slope_of_tangent_line a ∧ a = 7 := by
  sorry

end find_a_value_l138_138984


namespace siblings_ate_two_slices_l138_138948

-- Let slices_after_dinner be the number of slices left after eating one-fourth of 16 slices
def slices_after_dinner : ℕ := 16 - 16 / 4

-- Let slices_after_yves be the number of slices left after Yves ate one-fourth of the remaining pizza
def slices_after_yves : ℕ := slices_after_dinner - slices_after_dinner / 4

-- Let slices_left be the number of slices left after Yves's siblings ate some slices
def slices_left : ℕ := 5

-- Let slices_eaten_by_siblings be the number of slices eaten by Yves's siblings
def slices_eaten_by_siblings : ℕ := slices_after_yves - slices_left

-- Since there are two siblings, each ate half of the slices_eaten_by_siblings
def slices_per_sibling : ℕ := slices_eaten_by_siblings / 2

-- The theorem stating that each sibling ate 2 slices
theorem siblings_ate_two_slices : slices_per_sibling = 2 :=
by
  -- Definition of slices_after_dinner
  have h1 : slices_after_dinner = 12 := by sorry
  -- Definition of slices_after_yves
  have h2 : slices_after_yves = 9 := by sorry
  -- Definition of slices_eaten_by_siblings
  have h3 : slices_eaten_by_siblings = 4 := by sorry
  -- Final assertion of slices_per_sibling
  have h4 : slices_per_sibling = 2 := by sorry
  exact h4

end siblings_ate_two_slices_l138_138948


namespace find_1995th_remaining_number_l138_138358

theorem find_1995th_remaining_number :
  let seq := { n : ℕ | ¬ (n % 4 = 0 ∨ n % 7 = 0) ∨ n % 5 = 0 }
  (seq.to_finset.sort (≤)).nth (1994) = some 2795 :=
by
  have seq := { n : ℕ | ¬ (n % 4 = 0 ∨ n % 7 = 0) ∨ n % 5 = 0 }
  sorry

end find_1995th_remaining_number_l138_138358


namespace line_equation_through_point_and_area_l138_138622

theorem line_equation_through_point_and_area (k b : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4/3, 2)) ∧
  (∀ (A B : ℝ × ℝ), A = (- b / k, 0) ∧ B = (0, b) → 
  1 / 2 * abs ((- b / k) * b) = 6) →
  (y = k * x + b ↔ (y = -3/4 * x + 3 ∨ y = -3 * x + 6)) :=
by
  sorry

end line_equation_through_point_and_area_l138_138622


namespace abc_value_l138_138860

noncomputable def find_abc (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) : ℝ :=
  a * b * c

theorem abc_value (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 :=
by
  -- We skip the proof by providing sorry.
  sorry

end abc_value_l138_138860


namespace tax_refund_l138_138759

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end tax_refund_l138_138759


namespace tetrahedron_max_volume_l138_138924

noncomputable def tetrahedron_volume (AC AB BD CD : ℝ) : ℝ :=
  let x := (2 : ℝ) * (Real.sqrt 3) / 3
  let m := Real.sqrt (1 - x^2 / 4)
  let α := Real.pi / 2 -- Maximize with sin α = 1
  x * m^2 * Real.sin α / 6

theorem tetrahedron_max_volume : ∀ (AC AB BD CD : ℝ),
  AC = 1 → AB = 1 → BD = 1 → CD = 1 →
  tetrahedron_volume AC AB BD CD = 2 * Real.sqrt 3 / 27 :=
by
  intros AC AB BD CD hAC hAB hBD hCD
  rw [hAC, hAB, hBD, hCD]
  dsimp [tetrahedron_volume]
  norm_num
  sorry

end tetrahedron_max_volume_l138_138924


namespace abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l138_138232

theorem abs_x_minus_one_eq_one_minus_x_implies_x_le_one (x : ℝ) (h : |x - 1| = 1 - x) : x ≤ 1 :=
by
  sorry

end abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l138_138232


namespace graph_disjoint_paths_and_separation_l138_138122

theorem graph_disjoint_paths_and_separation (G : Type*) [graph G] {V : set G} (A B : set V) :
  ∃ (P : set (path G)) (S : set V), 
    (∀ p ∈ P, ends_of_path p.1 ∈ A ∧ ends_of_path p.2 ∈ B ∧ disjoint (p.interior ∩ p ∈ P)) ∧
    separation_set (A ∪ B) P S := 
sorry

end graph_disjoint_paths_and_separation_l138_138122


namespace committee_count_l138_138787

theorem committee_count :
  let num_ways_first3_dept := (nat.choose 3 2) * (nat.choose 3 2)
  let num_ways_physics := (nat.choose 1 1) * (nat.choose 1 1)
  let total_ways := num_ways_first3_dept ^ 3 * num_ways_physics
  total_ways = 729 :=
by
  sorry

end committee_count_l138_138787


namespace quiz_score_of_dropped_student_l138_138266

theorem quiz_score_of_dropped_student 
    (avg_all : ℝ) (num_all : ℕ) (new_avg_remaining : ℝ) (num_remaining : ℕ)
    (total_all : ℝ := num_all * avg_all) (total_remaining : ℝ := num_remaining * new_avg_remaining) :
    avg_all = 61.5 → num_all = 16 → new_avg_remaining = 64 → num_remaining = 15 → (total_all - total_remaining = 24) :=
by
  intros h_avg_all h_num_all h_new_avg_remaining h_num_remaining
  rw [h_avg_all, h_new_avg_remaining, h_num_all, h_num_remaining]
  sorry

end quiz_score_of_dropped_student_l138_138266


namespace total_pets_combined_l138_138577

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end total_pets_combined_l138_138577


namespace profit_per_box_type_A_and_B_maximize_profit_l138_138306

-- Condition definitions
def total_boxes : ℕ := 600
def profit_type_A : ℕ := 40000
def profit_type_B : ℕ := 160000
def profit_difference : ℕ := 200

-- Question 1: Proving the profit per box for type A and B
theorem profit_per_box_type_A_and_B (x : ℝ) :
  (profit_type_A / x + profit_type_B / (x + profit_difference) = total_boxes)
  → (x = 200) ∧ (x + profit_difference = 400) :=
sorry

-- Condition definitions for question 2
def price_reduction_per_box_A (a : ℕ) : ℕ := 5 * a
def price_increase_per_box_B (a : ℕ) : ℕ := 5 * a

-- Initial number of boxes sold for type A and B
def initial_boxes_sold_A : ℕ := 200
def initial_boxes_sold_B : ℕ := 400

-- General profit function
def profit (a : ℕ) : ℝ :=
  (initial_boxes_sold_A + 2 * a) * (200 - price_reduction_per_box_A a) +
  (initial_boxes_sold_B - 2 * a) * (400 + price_increase_per_box_B a)

-- Question 2: Proving the price reduction and maximum profit
theorem maximize_profit (a : ℕ) :
  ((price_reduction_per_box_A a = 75) ∧ (profit a = 204500)) :=
sorry

end profit_per_box_type_A_and_B_maximize_profit_l138_138306


namespace base4_to_base10_conversion_l138_138077

theorem base4_to_base10_conversion : 
  (1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 2 * 4^0) = 102 :=
by
  sorry

end base4_to_base10_conversion_l138_138077


namespace stratified_sampling_result_l138_138631

-- Define the total number of students in each grade
def students_grade10 : ℕ := 1600
def students_grade11 : ℕ := 1200
def students_grade12 : ℕ := 800

-- Define the condition
def stratified_sampling (x : ℕ) : Prop :=
  (x / (students_grade10 + students_grade11 + students_grade12) = (20 / students_grade12))

-- The main statement to be proven
theorem stratified_sampling_result 
  (students_grade10 : ℕ)
  (students_grade11 : ℕ)
  (students_grade12 : ℕ)
  (sampled_from_grade12 : ℕ)
  (h_sampling : stratified_sampling 90)
  (h_sampled12 : sampled_from_grade12 = 20) :
  (90 - sampled_from_grade12 = 70) :=
  by
    sorry

end stratified_sampling_result_l138_138631


namespace parabola_vertex_coordinates_l138_138609

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), y = (x - 2)^2 ∧ (x, y) = (2, 0) :=
sorry

end parabola_vertex_coordinates_l138_138609


namespace sky_falls_distance_l138_138321

def distance_from_city (x : ℕ) (y : ℕ) : Prop := 50 * x = y

theorem sky_falls_distance :
    ∃ D_s : ℕ, distance_from_city D_s 400 ∧ D_s = 8 :=
by
  sorry

end sky_falls_distance_l138_138321


namespace most_probable_standard_parts_in_batch_l138_138779

theorem most_probable_standard_parts_in_batch :
  let q := 0.075
  let p := 1 - q
  let n := 39
  ∃ k₀ : ℤ, 36 ≤ k₀ ∧ k₀ ≤ 37 := 
by
  sorry

end most_probable_standard_parts_in_batch_l138_138779


namespace sin_300_eq_neg_sqrt3_div_2_l138_138489

-- Defining the problem statement as a Lean theorem
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l138_138489


namespace equation_one_solution_equation_two_no_solution_l138_138263

theorem equation_one_solution (x : ℝ) (hx1 : x ≠ 3) : (2 * x + 9) / (3 - x) = (4 * x - 7) / (x - 3) ↔ x = -1 / 3 := 
by 
    sorry

theorem equation_two_no_solution (x : ℝ) (hx2 : x ≠ 1) (hx3 : x ≠ -1) : 
    (x + 1) / (x - 1) - 4 / (x ^ 2 - 1) = 1 → False := 
by 
    sorry

end equation_one_solution_equation_two_no_solution_l138_138263


namespace problem1_l138_138461

noncomputable def sqrt7_minus_1_pow_0 : ℝ := (Real.sqrt 7 - 1)^0
noncomputable def minus_half_pow_neg_2 : ℝ := (-1 / 2)^(-2 : ℤ)
noncomputable def sqrt3_tan_30 : ℝ := Real.sqrt 3 * Real.tan (Real.pi / 6)

theorem problem1 : sqrt7_minus_1_pow_0 - minus_half_pow_neg_2 + sqrt3_tan_30 = -2 := by
  sorry

end problem1_l138_138461


namespace remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l138_138053

theorem remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14 
  (a b c d e f g h : ℤ) 
  (h1 : a = 11085)
  (h2 : b = 11087)
  (h3 : c = 11089)
  (h4 : d = 11091)
  (h5 : e = 11093)
  (h6 : f = 11095)
  (h7 : g = 11097)
  (h8 : h = 11099) :
  (2 * (a + b + c + d + e + f + g + h)) % 14 = 2 := 
by
  sorry

end remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l138_138053


namespace min_value_frac_l138_138091

theorem min_value_frac (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ (∀ y, 0 < y ∧ y < 1 → (a * a / y + b * b / (1 - y)) ≥ (a + b) * (a + b)) ∧ 
       a * a / x + b * b / (1 - x) = (a + b) * (a + b) := 
by {
  sorry
}

end min_value_frac_l138_138091


namespace total_increase_by_five_l138_138734

-- Let B be the number of black balls
variable (B : ℕ)
-- Let W be the number of white balls
variable (W : ℕ)
-- Initially the total number of balls
def T := B + W
-- If the number of black balls is increased to 5 times the original, the total becomes twice the original
axiom h1 : 5 * B + W = 2 * (B + W)
-- If the number of white balls is increased to 5 times the original 
def k : ℕ := 5
-- The new total number of balls 
def new_total := B + k * W

-- Prove that the new total is 4 times the original total.
theorem total_increase_by_five : new_total = 4 * T :=
by
sorry

end total_increase_by_five_l138_138734


namespace exists_x_odd_n_l138_138252

theorem exists_x_odd_n (n : ℤ) (h : n % 2 = 1) : 
  ∃ x : ℤ, n^2 ∣ x^2 - n*x - 1 := by
  sorry

end exists_x_odd_n_l138_138252


namespace decreasing_implies_inequality_l138_138343

variable (f : ℝ → ℝ)

theorem decreasing_implies_inequality (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) : f 3 < f 2 ∧ f 2 < f 1 :=
  sorry

end decreasing_implies_inequality_l138_138343


namespace largest_int_less_than_100_remainder_4_l138_138837

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l138_138837


namespace sufficient_but_not_necessary_l138_138300

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 0 → x^2 + x > 0) ∧ (∃ y : ℝ, y < -1 ∧ y^2 + y > 0) :=
by
  sorry

end sufficient_but_not_necessary_l138_138300


namespace p_necessary_condition_q_l138_138089

variable (a b : ℝ) (p : ab = 0) (q : a^2 + b^2 ≠ 0)

theorem p_necessary_condition_q : (∀ a b : ℝ, (ab = 0) → (a^2 + b^2 ≠ 0)) ∧ (∃ a b : ℝ, (a^2 + b^2 ≠ 0) ∧ ¬ (ab = 0)) := sorry

end p_necessary_condition_q_l138_138089


namespace additional_vegetables_can_be_planted_l138_138711

-- Defines the garden's initial conditions.
def tomatoes_kinds := 3
def tomatoes_each := 5
def cucumbers_kinds := 5
def cucumbers_each := 4
def potatoes := 30
def rows := 10
def spaces_per_row := 15

-- The proof statement.
theorem additional_vegetables_can_be_planted (total_tomatoes : ℕ := tomatoes_kinds * tomatoes_each)
                                              (total_cucumbers : ℕ := cucumbers_kinds * cucumbers_each)
                                              (total_potatoes : ℕ := potatoes)
                                              (total_spaces : ℕ := rows * spaces_per_row) :
  total_spaces - (total_tomatoes + total_cucumbers + total_potatoes) = 85 := 
by 
  sorry

end additional_vegetables_can_be_planted_l138_138711


namespace min_baseball_cards_divisible_by_15_l138_138120

theorem min_baseball_cards_divisible_by_15 :
  ∀ (j m c e t : ℕ),
    j = m →
    m = c - 6 →
    c = 20 →
    e = 2 * (j + m) →
    t = c + m + j + e →
    t ≥ 104 →
    ∃ k : ℕ, t = 15 * k ∧ t = 105 :=
by
  intros j m c e t h1 h2 h3 h4 h5 h6
  sorry

end min_baseball_cards_divisible_by_15_l138_138120


namespace distinct_extreme_points_range_l138_138110

theorem distinct_extreme_points_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = -1/2 * x^2 + 4 * x - 2 * a * Real.log x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = 0 ∧ f' x2 = 0) →
  0 < a ∧ a < 2 :=
sorry

end distinct_extreme_points_range_l138_138110


namespace price_of_kid_ticket_l138_138568

theorem price_of_kid_ticket (k a : ℤ) (hk : k = 6) (ha : a = 2)
  (price_kid price_adult : ℤ)
  (hprice_adult : price_adult = 2 * price_kid)
  (hcost_total : 6 * price_kid + 2 * price_adult = 50) :
  price_kid = 5 :=
by
  sorry

end price_of_kid_ticket_l138_138568


namespace points_after_perfect_games_l138_138173

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l138_138173


namespace sequence_count_is_correct_l138_138251

def has_integer_root (a_i a_i_plus_1 : ℕ) : Prop :=
  ∃ r : ℕ, r^2 - a_i * r + a_i_plus_1 = 0

def valid_sequence (seq : Fin 16 → ℕ) : Prop :=
  ∀ i : Fin 15, has_integer_root (seq i.val + 1) (seq (i + 1).val + 1) ∧ seq 15 = seq 0

-- This noncomputable definition is used because we are estimating a specific number without providing a concrete computable function.
noncomputable def sequence_count : ℕ :=
  1409

theorem sequence_count_is_correct :
  ∃ N, valid_sequence seq → N = 1409 :=
sorry 

end sequence_count_is_correct_l138_138251


namespace min_weight_of_lightest_l138_138932

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l138_138932


namespace fraction_value_l138_138146

theorem fraction_value : (2 + 3 + 4 : ℚ) / (2 * 3 * 4) = 3 / 8 := 
by sorry

end fraction_value_l138_138146


namespace find_19a_20b_21c_l138_138390

theorem find_19a_20b_21c (a b c : ℕ) (h₁ : 29 * a + 30 * b + 31 * c = 366) 
  (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 19 * a + 20 * b + 21 * c = 246 := 
sorry

end find_19a_20b_21c_l138_138390


namespace different_signs_abs_value_larger_l138_138528

variable {a b : ℝ}

theorem different_signs_abs_value_larger (h1 : a + b < 0) (h2 : ab < 0) : 
  (a > 0 ∧ b < 0 ∧ |a| < |b|) ∨ (a < 0 ∧ b > 0 ∧ |b| < |a|) :=
sorry

end different_signs_abs_value_larger_l138_138528


namespace find_b_l138_138589

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: (-(a / 3) = -c)) (h2 : (-(a / 3) = 1 + a + b + c)) (h3: c = 2) : b = -11 :=
by
  sorry

end find_b_l138_138589


namespace tom_remaining_money_l138_138150

def monthly_allowance : ℝ := 12
def first_week_spending : ℝ := monthly_allowance * (1 / 3)
def remaining_after_first_week : ℝ := monthly_allowance - first_week_spending
def second_week_spending : ℝ := remaining_after_first_week * (1 / 4)
def remaining_after_second_week : ℝ := remaining_after_first_week - second_week_spending

theorem tom_remaining_money : remaining_after_second_week = 6 :=
by 
  sorry

end tom_remaining_money_l138_138150


namespace pressure_relation_l138_138758

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end pressure_relation_l138_138758


namespace gcd_bn_bn1_l138_138884

def b (n : ℕ) : ℤ := (7^n - 1) / 6
def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 1))

theorem gcd_bn_bn1 (n : ℕ) : e n = 1 := by
  sorry

end gcd_bn_bn1_l138_138884


namespace operation_results_in_m4_l138_138294

variable (m : ℤ)

theorem operation_results_in_m4 :
  (-m^2)^2 = m^4 :=
sorry

end operation_results_in_m4_l138_138294


namespace cubic_expansion_solution_l138_138449

theorem cubic_expansion_solution (x y : ℕ) (h_x : x = 27) (h_y : y = 9) : 
  x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3 = 46656 :=
by
  sorry

end cubic_expansion_solution_l138_138449


namespace part1_part2_part3_l138_138992

-- Given conditions and definitions
def A : ℝ := 1
def B : ℝ := 3
def y1 : ℝ := sorry  -- simply a placeholder value as y1 == y2
def y2 : ℝ := y1
def y (x m n : ℝ) : ℝ := x^2 + m * x + n

-- (1) Proof of m = -4
theorem part1 (n : ℝ) (h1 : y A m n = y1) (h2 : y B m n = y2) : m = -4 := sorry

-- (2) Proof of n = 4 when the parabola intersects the x-axis at one point
theorem part2 (h : ∃ n, ∀ x : ℝ, y x (-4) n = 0 → x = (x - 2)^2) : n = 4 := sorry

-- (3) Proof of the range of real number values for a
theorem part3 (a : ℝ) (b1 b2 : ℝ) (n : ℝ) (h1 : y a (-4) n = b1) 
  (h2 : y B (-4) n = b2) (h3 : b1 > b2) : a < 1 ∨ a > 3 := sorry

end part1_part2_part3_l138_138992


namespace rhombus_shorter_diagonal_l138_138218

theorem rhombus_shorter_diagonal (perimeter : ℝ) (angle_ratio : ℝ) (side_length diagonal_length : ℝ)
  (h₁ : perimeter = 9.6) 
  (h₂ : angle_ratio = 1 / 2) 
  (h₃ : side_length = perimeter / 4) 
  (h₄ : diagonal_length = side_length) :
  diagonal_length = 2.4 := 
sorry

end rhombus_shorter_diagonal_l138_138218


namespace mod_congruent_integers_l138_138523

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l138_138523


namespace a_and_b_together_time_eq_4_over_3_l138_138596

noncomputable def work_together_time (a b c h : ℝ) :=
  (1 / a) + (1 / b) + (1 / c) = (1 / (a - 6)) ∧
  (1 / a) + (1 / b) = 1 / h ∧
  (1 / (a - 6)) = (1 / (b - 1)) ∧
  (1 / (a - 6)) = 2 / c

theorem a_and_b_together_time_eq_4_over_3 (a b c h : ℝ) (h_wt : work_together_time a b c h) : 
  h = 4 / 3 :=
  sorry

end a_and_b_together_time_eq_4_over_3_l138_138596


namespace problem_solution_l138_138990

-- Definition of the geometric sequence and the arithmetic condition
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def arithmetic_condition (a : ℕ → ℕ) := 2 * (a 3 + 1) = a 2 + a 4

-- Definitions used in the proof
def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) := a_n n + n
def S_5 := b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5

-- Proof statement
theorem problem_solution : 
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_condition a ∧ a 1 = 1 ∧ (∀ n, a n = 2^(n-1))) ∧
  S_5 = 46 :=
by
  sorry

end problem_solution_l138_138990


namespace quadratic_solution_interval_l138_138844

noncomputable def quadratic_inequality (z : ℝ) : Prop :=
  z^2 - 56*z + 360 ≤ 0

theorem quadratic_solution_interval :
  {z : ℝ // quadratic_inequality z} = {z : ℝ // 8 ≤ z ∧ z ≤ 45} :=
by
  sorry

end quadratic_solution_interval_l138_138844


namespace find_a_l138_138806

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l138_138806


namespace find_V_D_l138_138946

noncomputable def V_A : ℚ := sorry
noncomputable def V_B : ℚ := sorry
noncomputable def V_C : ℚ := sorry
noncomputable def V_D : ℚ := sorry
noncomputable def V_E : ℚ := sorry

axiom condition1 : V_A + V_B + V_C + V_D + V_E = 1 / 7.5
axiom condition2 : V_A + V_C + V_E = 1 / 5
axiom condition3 : V_A + V_C + V_D = 1 / 6
axiom condition4 : V_B + V_D + V_E = 1 / 4

theorem find_V_D : V_D = 1 / 12 := 
  by
    sorry

end find_V_D_l138_138946


namespace most_stable_performance_l138_138506

-- Define the variances for each player
def variance_A : ℝ := 0.66
def variance_B : ℝ := 0.52
def variance_C : ℝ := 0.58
def variance_D : ℝ := 0.62

-- State the theorem
theorem most_stable_performance : variance_B < variance_C ∧ variance_C < variance_D ∧ variance_D < variance_A :=
by
  -- Since we are tasked to write only the statement, the proof part is skipped.
  sorry

end most_stable_performance_l138_138506


namespace fish_caught_together_l138_138696

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l138_138696


namespace tan_alpha_second_quadrant_l138_138097

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_second_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π)
  (h2 : Real.cos (π / 2 - α) = 4 / 5) :
  tan_alpha α = -4 / 3 :=
by
  sorry

end tan_alpha_second_quadrant_l138_138097


namespace area_of_sector_l138_138217

theorem area_of_sector
  (θ : ℝ) (l : ℝ) (r : ℝ := l / θ)
  (h1 : θ = 2)
  (h2 : l = 4) :
  1 / 2 * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l138_138217


namespace sum_of_repeating_decimals_l138_138082

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l138_138082


namespace find_maximum_marks_l138_138756

variable (percent_marks : ℝ := 0.92)
variable (obtained_marks : ℝ := 368)
variable (max_marks : ℝ := obtained_marks / percent_marks)

theorem find_maximum_marks : max_marks = 400 := by
  sorry

end find_maximum_marks_l138_138756


namespace blueberries_count_l138_138127

theorem blueberries_count
  (initial_apples : ℕ)
  (initial_oranges : ℕ)
  (initial_blueberries : ℕ)
  (apples_eaten : ℕ)
  (oranges_eaten : ℕ)
  (remaining_fruits : ℕ)
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 9)
  (h3 : apples_eaten = 1)
  (h4 : oranges_eaten = 1)
  (h5 : remaining_fruits = 26) :
  initial_blueberries = 5 := 
by
  sorry

end blueberries_count_l138_138127


namespace intersection_nonempty_iff_m_lt_one_l138_138249

open Set Real

variable {m : ℝ}

theorem intersection_nonempty_iff_m_lt_one 
  (A : Set ℝ) (B : Set ℝ) (U : Set ℝ := univ) 
  (hA : A = {x | x + m >= 0}) 
  (hB : B = {x | -1 < x ∧ x < 5}) : 
  (U \ A ∩ B ≠ ∅) ↔ m < 1 := by
  sorry

end intersection_nonempty_iff_m_lt_one_l138_138249


namespace bottles_last_days_l138_138795

theorem bottles_last_days :
  let total_bottles := 8066
  let bottles_per_day := 109
  total_bottles / bottles_per_day = 74 :=
by
  sorry

end bottles_last_days_l138_138795


namespace range_subtraction_l138_138514

theorem range_subtraction {a b : ℝ} (h_range : set.image (fun x => 2 * real.cos x) (set.Icc (real.pi / 3) (4 * real.pi / 3)) = set.Icc a b) :
  b - a = 3 :=
by
  sorry

end range_subtraction_l138_138514


namespace allison_total_video_hours_l138_138071

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end allison_total_video_hours_l138_138071


namespace diameter_inscribed_circle_l138_138271

noncomputable def diameter_of_circle (r : ℝ) : ℝ :=
2 * r

theorem diameter_inscribed_circle (r : ℝ) (h : 8 * r = π * r ^ 2) : diameter_of_circle r = 16 / π := by
  sorry

end diameter_inscribed_circle_l138_138271


namespace problem_xyz_l138_138689

noncomputable def distance_from_intersection_to_side_CD (s : ℝ) : ℝ :=
  s * ((8 - Real.sqrt 15) / 8)

theorem problem_xyz
  (s : ℝ)
  (ABCD_is_square : (0 ≤ s))
  (X_is_intersection: ∃ (X : ℝ × ℝ), (X.1^2 + X.2^2 = s^2) ∧ ((X.1 - s)^2 + X.2^2 = (s / 2)^2))
  : distance_from_intersection_to_side_CD s = (s * (8 - Real.sqrt 15) / 8) :=
sorry

end problem_xyz_l138_138689


namespace cora_cookies_per_day_l138_138843

theorem cora_cookies_per_day :
  (∀ (day : ℕ), day ∈ (Finset.range 30) →
    ∃ cookies_per_day : ℕ,
    cookies_per_day * 30 = 1620 / 18) →
  cookies_per_day = 3 := by
  sorry

end cora_cookies_per_day_l138_138843


namespace problem_solved_by_half_participants_l138_138377

variables (n m : ℕ)
variable (solve : ℕ → ℕ → Prop)  -- solve i j means participant i solved problem j

axiom half_n_problems_solved : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)

theorem problem_solved_by_half_participants (h : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)) : 
  ∃ j, j < n ∧ (∃ count, count ≥ m / 2 ∧ (∃ i, i < m → solve i j)) :=
  sorry

end problem_solved_by_half_participants_l138_138377


namespace smallest_two_digit_product_12_l138_138440

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l138_138440


namespace stormi_cars_washed_l138_138899

-- Definitions based on conditions
def cars_earning := 10
def lawns_number := 2
def lawn_earning := 13
def bicycle_cost := 80
def needed_amount := 24

-- Auxiliary calculations
def lawns_total_earning := lawns_number * lawn_earning
def already_earning := bicycle_cost - needed_amount
def cars_total_earning := already_earning - lawns_total_earning

-- Main problem statement
theorem stormi_cars_washed : (cars_total_earning / cars_earning) = 3 :=
  by sorry

end stormi_cars_washed_l138_138899


namespace total_cost_of_new_movie_l138_138876

noncomputable def previous_movie_length_hours : ℕ := 2
noncomputable def new_movie_length_increase_percent : ℕ := 60
noncomputable def previous_movie_cost_per_minute : ℕ := 50
noncomputable def new_movie_cost_per_minute_factor : ℕ := 2 

theorem total_cost_of_new_movie : 
  let new_movie_length_hours := previous_movie_length_hours + (previous_movie_length_hours * new_movie_length_increase_percent / 100)
  let new_movie_length_minutes := new_movie_length_hours * 60
  let new_movie_cost_per_minute := previous_movie_cost_per_minute * new_movie_cost_per_minute_factor
  let total_cost := new_movie_length_minutes * new_movie_cost_per_minute
  total_cost = 19200 := 
by
  sorry

end total_cost_of_new_movie_l138_138876


namespace scientific_notation_of_300_million_l138_138591

theorem scientific_notation_of_300_million : 
  300000000 = 3 * 10^8 := 
by
  sorry

end scientific_notation_of_300_million_l138_138591


namespace largest_integer_less_100_leaves_remainder_4_l138_138832

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l138_138832


namespace average_hamburgers_per_day_l138_138629

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l138_138629


namespace angle_BDE_60_l138_138242

noncomputable def is_isosceles_triangle (A B C : Type) (angle_BAC : ℝ) : Prop :=
angle_BAC = 20

noncomputable def equal_sides (BC BD BE : ℝ) : Prop :=
BC = BD ∧ BD = BE

theorem angle_BDE_60 (A B C D E : Type) (BC BD BE : ℝ) 
  (h1 : is_isosceles_triangle A B C 20) 
  (h2 : equal_sides BC BD BE) : 
  ∃ (angle_BDE : ℝ), angle_BDE = 60 :=
by
  sorry

end angle_BDE_60_l138_138242


namespace average_of_new_sequence_l138_138261

theorem average_of_new_sequence (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_new_sequence_l138_138261


namespace find_a_l138_138809

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l138_138809


namespace probability_first_ge_second_l138_138309

-- Define the number of faces
def faces : ℕ := 10

-- Define the total number of outcomes excluding the duplicates
def total_outcomes : ℕ := faces * faces - faces

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ := 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_first_ge_second :
  probability = 11 / 18 :=
sorry

end probability_first_ge_second_l138_138309


namespace smallest_pretty_num_l138_138177

-- Define the notion of a pretty number
def is_pretty (n : ℕ) : Prop :=
  ∃ d1 d2 : ℕ, (1 ≤ d1 ∧ d1 ≤ n) ∧ (1 ≤ d2 ∧ d2 ≤ n) ∧ d2 - d1 ∣ n ∧ (1 < d1)

-- Define the statement to prove that 160400 is the smallest pretty number greater than 401 that is a multiple of 401
theorem smallest_pretty_num (n : ℕ) (hn1 : n > 401) (hn2 : n % 401 = 0) : n = 160400 :=
  sorry

end smallest_pretty_num_l138_138177


namespace Madelyn_daily_pizza_expense_l138_138184

theorem Madelyn_daily_pizza_expense (total_expense : ℕ) (days_in_may : ℕ) 
  (h1 : total_expense = 465) (h2 : days_in_may = 31) : 
  total_expense / days_in_may = 15 := 
by
  sorry

end Madelyn_daily_pizza_expense_l138_138184


namespace sqrt_inequality_l138_138849

theorem sqrt_inequality (x : ℝ) (h₁ : 3 / 2 ≤ x) (h₂ : x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := 
sorry

end sqrt_inequality_l138_138849


namespace inequality_holds_l138_138372

variables (a b c : ℝ)

theorem inequality_holds 
  (h1 : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) :=
sorry

end inequality_holds_l138_138372


namespace unique_solution_2023_plus_2_pow_n_eq_k_sq_l138_138189

theorem unique_solution_2023_plus_2_pow_n_eq_k_sq (n k : ℕ) (h : 2023 + 2^n = k^2) :
  (n = 1 ∧ k = 45) :=
by
  sorry

end unique_solution_2023_plus_2_pow_n_eq_k_sq_l138_138189


namespace cats_left_l138_138961

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) (total_initial_cats : ℕ) (remaining_cats : ℕ) :
  siamese_cats = 15 → house_cats = 49 → cats_sold = 19 → total_initial_cats = siamese_cats + house_cats → remaining_cats = total_initial_cats - cats_sold → remaining_cats = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h4, h3] at h5
  exact h5

end cats_left_l138_138961


namespace minimum_positive_period_of_f_l138_138584

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_positive_period_of_f : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi := 
sorry

end minimum_positive_period_of_f_l138_138584


namespace amount_paid_l138_138691

-- Defining the conditions as constants
def cost_of_apple : ℝ := 0.75
def change_received : ℝ := 4.25

-- Stating the theorem that needs to be proved
theorem amount_paid (a : ℝ) : a = cost_of_apple + change_received :=
by
  sorry

end amount_paid_l138_138691


namespace polynomial_evaluation_l138_138498

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l138_138498


namespace evaluate_f_3_minus_f_neg_3_l138_138679

def f (x : ℝ) : ℝ := x^4 + x^2 + 7 * x

theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_3_minus_f_neg_3_l138_138679


namespace increase_by_40_percent_l138_138762

theorem increase_by_40_percent (initial_number : ℕ) (increase_rate : ℕ) :
  initial_number = 150 → increase_rate = 40 →
  initial_number + (increase_rate / 100 * initial_number) = 210 := by
  sorry

end increase_by_40_percent_l138_138762


namespace minimize_at_five_halves_five_sixths_l138_138357

noncomputable def minimize_expression (x y : ℝ) : ℝ :=
  (y - 1)^2 + (x + y - 3)^2 + (2 * x + y - 6)^2

theorem minimize_at_five_halves_five_sixths (x y : ℝ) :
  minimize_expression x y = 1 / 6 ↔ (x = 5 / 2 ∧ y = 5 / 6) :=
sorry

end minimize_at_five_halves_five_sixths_l138_138357


namespace reflect_center_is_image_center_l138_138406

def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem reflect_center_is_image_center : 
  reflect_over_y_eq_neg_x (3, -4) = (4, -3) :=
by
  -- Proof is omitted as per instructions.
  -- This proof would show the reflection of the point (3, -4) over the line y = -x resulting in (4, -3).
  sorry

end reflect_center_is_image_center_l138_138406


namespace smallest_two_digit_number_product_12_l138_138443

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l138_138443


namespace value_of_f_two_l138_138364

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_two :
  (∀ x : ℝ, f (1 / x) = 1 / (x + 1)) → f 2 = 2 / 3 := by
  intro h
  -- The proof would go here
  sorry

end value_of_f_two_l138_138364


namespace value_of_a_squared_plus_2a_l138_138231

theorem value_of_a_squared_plus_2a (a x : ℝ) (h1 : x = -5) (h2 : 2 * x + 8 = x / 5 - a) : a^2 + 2 * a = 3 :=
by {
  sorry
}

end value_of_a_squared_plus_2a_l138_138231


namespace length_to_width_ratio_l138_138080

-- Define the conditions: perimeter and length
variable (P : ℕ) (l : ℕ) (w : ℕ)

-- Given conditions
def conditions : Prop := (P = 100) ∧ (l = 40) ∧ (P = 2 * l + 2 * w)

-- The proposition we want to prove
def ratio : Prop := l / w = 4

-- The main theorem
theorem length_to_width_ratio (h : conditions P l w) : ratio l w :=
by sorry

end length_to_width_ratio_l138_138080


namespace product_of_consecutive_integers_l138_138904

theorem product_of_consecutive_integers
  (a b : ℕ) (n : ℕ)
  (h1 : a = 12)
  (h2 : b = 22)
  (mean_five_numbers : (a + b + n + (n + 1) + (n + 2)) / 5 = 17) :
  (n * (n + 1) * (n + 2)) = 4896 := by
  sorry

end product_of_consecutive_integers_l138_138904


namespace solve_for_x_l138_138092

noncomputable def valid_x (x : ℝ) : Prop :=
  let l := 4 * x
  let w := 2 * x + 6
  l * w = 2 * (l + w)

theorem solve_for_x : 
  ∃ (x : ℝ), valid_x x ↔ x = (-3 + Real.sqrt 33) / 4 :=
by
  sorry

end solve_for_x_l138_138092


namespace largest_rectangle_in_circle_l138_138535

theorem largest_rectangle_in_circle {r : ℝ} (h : r = 6) : 
  ∃ A : ℝ, A = 72 := 
by 
  sorry

end largest_rectangle_in_circle_l138_138535


namespace radius_of_inscribed_circle_in_COD_l138_138408

theorem radius_of_inscribed_circle_in_COD
  (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
  (H1 : r1 = 6)
  (H2 : r2 = 2)
  (H3 : r3 = 1.5)
  (H4 : 1/r1 + 1/r3 = 1/r2 + 1/r4) :
  r4 = 3 :=
by
  sorry

end radius_of_inscribed_circle_in_COD_l138_138408


namespace find_ordered_pair_l138_138571

theorem find_ordered_pair {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = -2 * a ∨ x = b)
  (h2 : b = -2 * -2 * a) : (a, b) = (-1/2, -1/2) :=
by
  sorry

end find_ordered_pair_l138_138571


namespace avg_variance_stability_excellent_performance_probability_l138_138379

-- Define the scores of players A and B in seven games
def scores_A : List ℕ := [26, 28, 32, 22, 37, 29, 36]
def scores_B : List ℕ := [26, 29, 32, 28, 39, 29, 27]

-- Define the mean and variance calculations
def mean (scores : List ℕ) : ℚ := (scores.sum : ℚ) / scores.length
def variance (scores : List ℕ) : ℚ := 
  (scores.map (λ x => (x - mean scores) ^ 2)).sum / scores.length

theorem avg_variance_stability :
  mean scores_A = 30 ∧ mean scores_B = 30 ∧
  variance scores_A = 174 / 7 ∧ variance scores_B = 116 / 7 ∧
  variance scores_A > variance scores_B := 
by
  sorry

-- Define the probabilities of scoring higher than 30
def probability_excellent (scores : List ℕ) : ℚ := 
  (scores.filter (λ x => x > 30)).length / scores.length

theorem excellent_performance_probability :
  probability_excellent scores_A = 3 / 7 ∧ probability_excellent scores_B = 2 / 7 ∧
  (probability_excellent scores_A * probability_excellent scores_B = 6 / 49) :=
by
  sorry

end avg_variance_stability_excellent_performance_probability_l138_138379


namespace tangent_line_eq_l138_138918

theorem tangent_line_eq
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (P : ℝ × ℝ)
  (hP : P = (1, Real.sqrt 3))
  : x - Real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_eq_l138_138918


namespace min_value_of_expression_l138_138000

theorem min_value_of_expression 
  (a b : ℝ) 
  (h : a > 0) 
  (h₀ : b > 0) 
  (h₁ : 2*a + b = 2) : 
  ∃ c : ℝ, c = (8*a + b) / (a*b) ∧ c = 9 :=
sorry

end min_value_of_expression_l138_138000


namespace negation_of_p_l138_138014

def f (a x : ℝ) : ℝ := a * x - x - a

theorem negation_of_p :
  (¬ ∀ a > 0, a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔ (∃ a > 0, a ≠ 1 ∧ ¬ ∃ x : ℝ, f a x = 0) :=
by {
  sorry
}

end negation_of_p_l138_138014


namespace prob_D_correct_l138_138964

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 3
def prob_C : ℚ := 1 / 6
def total_prob (prob_D : ℚ) : Prop := prob_A + prob_B + prob_C + prob_D = 1

theorem prob_D_correct : ∃ (prob_D : ℚ), total_prob prob_D ∧ prob_D = 1 / 4 :=
by
  -- Proof omitted
  sorry

end prob_D_correct_l138_138964


namespace max_students_distributing_pens_and_pencils_l138_138458

theorem max_students_distributing_pens_and_pencils :
  Nat.gcd 1001 910 = 91 :=
by
  -- remaining proof required
  sorry

end max_students_distributing_pens_and_pencils_l138_138458


namespace angle_AMC_is_70_l138_138690

theorem angle_AMC_is_70 (A B C M : Type) (angle_MBA angle_MAB angle_ACB : ℝ) (AC BC : ℝ) :
  AC = BC → 
  angle_MBA = 30 → 
  angle_MAB = 10 → 
  angle_ACB = 80 → 
  ∃ angle_AMC : ℝ, angle_AMC = 70 :=
by
  sorry

end angle_AMC_is_70_l138_138690


namespace length_ac_l138_138295

theorem length_ac (a b c d e : ℝ) (h1 : bc = 3 * cd) (h2 : de = 7) (h3 : ab = 5) (h4 : ae = 20) :
    ac = 11 :=
by
  sorry

end length_ac_l138_138295


namespace xyz_value_l138_138512

theorem xyz_value (x y z : ℚ)
  (h1 : x + y + z = 1)
  (h2 : x + y - z = 2)
  (h3 : x - y - z = 3) :
  x * y * z = 1/2 :=
by
  sorry

end xyz_value_l138_138512


namespace find_a_l138_138811

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l138_138811


namespace reciprocal_neg_one_thirteen_l138_138415

theorem reciprocal_neg_one_thirteen : -(1:ℝ) / 13⁻¹ = -13 := 
sorry

end reciprocal_neg_one_thirteen_l138_138415


namespace matrix_determinant_zero_implies_sum_of_squares_l138_138664

theorem matrix_determinant_zero_implies_sum_of_squares (a b : ℝ)
  (h : (Matrix.det ![![a - Complex.I, b - 2 * Complex.I],
                       ![1, 1 + Complex.I]]) = 0) :
  a^2 + b^2 = 1 :=
sorry

end matrix_determinant_zero_implies_sum_of_squares_l138_138664


namespace sled_dog_race_l138_138319

theorem sled_dog_race (d t : ℕ) (h1 : d + t = 315) (h2 : (1.2 : ℚ) * d + t = (1 / 2 : ℚ) * (2 * d + 3 * t)) :
  d = 225 ∧ t = 90 :=
sorry

end sled_dog_race_l138_138319


namespace problem_equiv_l138_138109

variable (a b c d e f : ℝ)

theorem problem_equiv :
  a * b * c = 65 → 
  b * c * d = 65 → 
  c * d * e = 1000 → 
  d * e * f = 250 → 
  (a * f) / (c * d) = 1 / 4 := 
by 
  intros h1 h2 h3 h4
  sorry

end problem_equiv_l138_138109


namespace workers_contribution_eq_l138_138752

variable (W C : ℕ)

theorem workers_contribution_eq :
  W * C = 300000 → W * (C + 50) = 320000 → W = 400 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end workers_contribution_eq_l138_138752


namespace milk_price_increase_l138_138872

theorem milk_price_increase
  (P : ℝ) (C : ℝ) (P_new : ℝ)
  (h1 : P * C = P_new * (5 / 6) * C) :
  (P_new - P) / P * 100 = 20 :=
by
  sorry

end milk_price_increase_l138_138872


namespace total_driving_time_is_40_l138_138620

noncomputable def totalDrivingTime
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ) : ℕ :=
  let trips := totalCattle / truckCapacity
  let timePerRoundTrip := 2 * (distance / speed)
  trips * timePerRoundTrip

theorem total_driving_time_is_40
  (totalCattle : ℕ)
  (truckCapacity : ℕ)
  (distance : ℕ)
  (speed : ℕ)
  (hCattle : totalCattle = 400)
  (hCapacity : truckCapacity = 20)
  (hDistance : distance = 60)
  (hSpeed : speed = 60) :
  totalDrivingTime totalCattle truckCapacity distance speed = 40 := by
  sorry

end total_driving_time_is_40_l138_138620


namespace length_AE_l138_138543

theorem length_AE (A B C D E : Type) 
  (AB AC AD AE : ℝ) 
  (angle_BAC : ℝ)
  (h1 : AB = 4.5) 
  (h2 : AC = 5) 
  (h3 : angle_BAC = 30) 
  (h4 : AD = 1.5) 
  (h5 : AD / AB = AE / AC) : 
  AE = 1.6667 := 
sorry

end length_AE_l138_138543


namespace integral_sin_pi_half_to_three_pi_half_l138_138495

theorem integral_sin_pi_half_to_three_pi_half :
  ∫ x in (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), Real.sin x = 0 :=
by
  sorry

end integral_sin_pi_half_to_three_pi_half_l138_138495


namespace star_j_l138_138344

def star (x y : ℝ) : ℝ := x^3 - x * y

theorem star_j (j : ℝ) : star j (star j j) = 2 * j^3 - j^4 := 
by
  sorry

end star_j_l138_138344


namespace non_congruent_squares_on_6_by_6_grid_l138_138225

theorem non_congruent_squares_on_6_by_6_grid :
  let n := 6 in
  (sum (list.map (λ (k : ℕ), (n - k) * (n - k)) [1, 2, 3, 4, 5]) +
  25 + 9 + 1 + 20 + 10 + 8) = 128 := by
  sorry

end non_congruent_squares_on_6_by_6_grid_l138_138225


namespace find_a_l138_138808

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l138_138808


namespace exponential_function_solution_l138_138375

theorem exponential_function_solution (a : ℝ) (h₁ : ∀ x : ℝ, a ^ x > 0) :
  (∃ y : ℝ, y = a ^ 2 ∧ y = 4) → a = 2 :=
by
  sorry

end exponential_function_solution_l138_138375


namespace expected_adjacent_red_pairs_correct_l138_138908

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l138_138908


namespace route_down_distance_l138_138957

theorem route_down_distance
  (rate_up : ℕ)
  (time_up : ℕ)
  (rate_down_rate_factor : ℚ)
  (time_down : ℕ)
  (h1 : rate_up = 4)
  (h2 : time_up = 2)
  (h3 : rate_down_rate_factor = (3 / 2))
  (h4 : time_down = time_up) :
  rate_down_rate_factor * rate_up * time_up = 12 := 
by
  rw [h1, h2, h3]
  sorry

end route_down_distance_l138_138957


namespace simplify_frac_l138_138337

variable (m : ℝ)

theorem simplify_frac : m^2 ≠ 9 → (3 / (m^2 - 9) + m / (9 - m^2)) = - (1 / (m + 3)) :=
by
  intro h
  sorry

end simplify_frac_l138_138337


namespace total_money_at_least_108_l138_138004

-- Definitions for the problem
def tram_ticket_cost : ℕ := 1
def passenger_coins (n : ℕ) : Prop := n = 2 ∨ n = 5

-- Condition that conductor had no change initially
def initial_conductor_money : ℕ := 0

-- Condition that each passenger can pay exactly 1 Ft and receive change
def can_pay_ticket_with_change (coins : List ℕ) : Prop := 
  ∀ c ∈ coins, passenger_coins c → 
    ∃ change : List ℕ, (change.sum = c - tram_ticket_cost) ∧ 
      (∀ x ∈ change, passenger_coins x)

-- Assume we have 20 passengers with only 2 Ft and 5 Ft coins
def passengers_coins : List (List ℕ) :=
  -- Simplified representation
  List.replicate 20 [2, 5]

noncomputable def total_passenger_money : ℕ :=
  (passengers_coins.map List.sum).sum

-- Lean statement for the proof problem
theorem total_money_at_least_108 : total_passenger_money ≥ 108 :=
sorry

end total_money_at_least_108_l138_138004


namespace meaningful_sqrt_range_l138_138740

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l138_138740


namespace regular_price_of_one_tire_l138_138726

theorem regular_price_of_one_tire
  (x : ℝ) -- Define the variable \( x \) as the regular price of one tire
  (h1 : 3 * x + 10 = 250) -- Set up the equation based on the condition

  : x = 80 := 
sorry

end regular_price_of_one_tire_l138_138726


namespace largest_int_less_than_100_remainder_4_l138_138836

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l138_138836


namespace cost_price_250_l138_138962

theorem cost_price_250 (C : ℝ) (h1 : 0.90 * C = C - 0.10 * C) (h2 : 1.10 * C = C + 0.10 * C) (h3 : 1.10 * C - 0.90 * C = 50) : C = 250 := 
by
  sorry

end cost_price_250_l138_138962


namespace discount_percentage_for_two_pairs_of_jeans_l138_138317

theorem discount_percentage_for_two_pairs_of_jeans
  (price_per_pair : ℕ := 40)
  (price_for_three_pairs : ℕ := 112)
  (discount : ℕ := 8)
  (original_price_for_two_pairs : ℕ := price_per_pair * 2)
  (discount_percentage : ℕ := (discount * 100) / original_price_for_two_pairs) :
  discount_percentage = 10 := 
by
  sorry

end discount_percentage_for_two_pairs_of_jeans_l138_138317


namespace service_center_milepost_l138_138580

theorem service_center_milepost :
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  service_center = 120 :=
by
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  sorry

end service_center_milepost_l138_138580


namespace y_intercept_of_line_l138_138981

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by sorry

end y_intercept_of_line_l138_138981


namespace expected_pairs_of_red_in_circle_deck_l138_138912

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l138_138912


namespace Lizzie_group_number_l138_138702

theorem Lizzie_group_number (x : ℕ) (h1 : x + (x + 17) = 91) : x + 17 = 54 :=
by
  sorry

end Lizzie_group_number_l138_138702


namespace one_fifth_of_ten_x_plus_three_l138_138235

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : 
  (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := 
  sorry

end one_fifth_of_ten_x_plus_three_l138_138235


namespace abs_sum_fraction_le_sum_abs_fraction_l138_138551

variable (a b : ℝ)

theorem abs_sum_fraction_le_sum_abs_fraction (a b : ℝ) :
  (|a + b| / (1 + |a + b|)) ≤ (|a| / (1 + |a|)) + (|b| / (1 + |b|)) :=
sorry

end abs_sum_fraction_le_sum_abs_fraction_l138_138551


namespace katie_ds_games_l138_138388

theorem katie_ds_games (new_friends_games old_friends_games total_friends_games katie_games : ℕ) 
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_friends_games = 141)
  (h4 : total_friends_games = new_friends_games + old_friends_games + katie_games) :
  katie_games = 0 :=
by
  sorry

end katie_ds_games_l138_138388


namespace quadratic_roots_primes_4_possible_k_l138_138789

theorem quadratic_roots_primes_4_possible_k :
  ∃ k_set: set ℕ, k_set.card = 4 ∧
    (∀ k ∈ k_set, ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧
      p + q = 58 ∧ p * q = k) :=
by sorry

end quadratic_roots_primes_4_possible_k_l138_138789


namespace specific_n_values_l138_138491

theorem specific_n_values (n : ℕ) : 
  ∃ m : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → m % k = 0) ∧ 
    (m % (n + 1) ≠ 0) ∧ 
    (m % (n + 2) ≠ 0) ∧ 
    (m % (n + 3) ≠ 0) ↔ n = 1 ∨ n = 2 ∨ n = 6 := 
by
  sorry

end specific_n_values_l138_138491


namespace remainder_three_l138_138055

-- Define the condition that x % 6 = 3
def condition (x : ℕ) : Prop := x % 6 = 3

-- Proof statement that if condition is met, then (3 * x) % 6 = 3
theorem remainder_three {x : ℕ} (h : condition x) : (3 * x) % 6 = 3 :=
sorry

end remainder_three_l138_138055


namespace probability_of_winning_five_tickets_l138_138729

def probability_of_winning_one_ticket := 1 / 10000000
def number_of_tickets_bought := 5

theorem probability_of_winning_five_tickets : 
  (number_of_tickets_bought * probability_of_winning_one_ticket) = 5 / 10000000 :=
by
  sorry

end probability_of_winning_five_tickets_l138_138729


namespace remainder_when_sum_divided_by_29_l138_138395

theorem remainder_when_sum_divided_by_29 (c d : ℤ) (k j : ℤ) 
  (hc : c = 52 * k + 48) 
  (hd : d = 87 * j + 82) : 
  (c + d) % 29 = 22 := 
by 
  sorry

end remainder_when_sum_divided_by_29_l138_138395


namespace unique_real_root_count_l138_138982

theorem unique_real_root_count :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 := by
  sorry

end unique_real_root_count_l138_138982


namespace product_of_N1_N2_l138_138391

theorem product_of_N1_N2 :
  (∃ (N1 N2 : ℤ),
    (∀ (x : ℚ),
      (47 * x - 35) * (x - 1) * (x - 2) = N1 * (x - 2) * (x - 1) + N2 * (x - 1) * (x - 2)) ∧
    N1 * N2 = -708) :=
sorry

end product_of_N1_N2_l138_138391


namespace min_value_expression_l138_138250

theorem min_value_expression (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (min ((1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + (x * y * z)) 2) = 2 :=
by 
  sorry

end min_value_expression_l138_138250


namespace average_hamburgers_per_day_l138_138628

theorem average_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h₁ : total_hamburgers = 63) (h₂ : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end average_hamburgers_per_day_l138_138628


namespace product_of_numbers_l138_138145

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x * y = 375 :=
sorry

end product_of_numbers_l138_138145


namespace limit_sine_power_l138_138971

open Real Filter

theorem limit_sine_power (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = (sin (2 * x) / x) ^ (1 + x)) →
  ∃ l : ℝ, tendsto (λ x : ℝ, f x) (𝓝 0) (𝓝 l) ∧ l = 2 :=
begin
  intro h_f,
  have h_sin : ∀ x, sin (2 * x) = 2 * x * sin x / x,
    from sorry,
  have h_tendsto_base : tendsto (λ x, (sin (2 * x) / x)) (𝓝 0) (𝓝 2),
    from sorry,
  have h_tendsto_exponent : tendsto (λ x, (1 + x)) (𝓝 0) (𝓝 1),
    from tendsto_add tendsto_const_nhds tendsto_id,
  rw ← h_f,
  rw ← tendsto.comp h_tendsto_base h_tendsto_exponent,
  rw pow_one,
  exact 2,
end

end limit_sine_power_l138_138971


namespace evelyn_found_caps_l138_138197

theorem evelyn_found_caps (start_caps end_caps found_caps : ℕ) 
    (h1 : start_caps = 18) 
    (h2 : end_caps = 81) 
    (h3 : found_caps = end_caps - start_caps) :
  found_caps = 63 := by
  sorry

end evelyn_found_caps_l138_138197


namespace simplest_quadratic_radical_l138_138480

noncomputable def optionA := Real.sqrt 7
noncomputable def optionB := Real.sqrt 9
noncomputable def optionC := Real.sqrt 12
noncomputable def optionD := Real.sqrt (2 / 3)

theorem simplest_quadratic_radical :
  optionA = Real.sqrt 7 ∧
  optionB = Real.sqrt 9 ∧
  optionC = Real.sqrt 12 ∧
  optionD = Real.sqrt (2 / 3) ∧
  (optionB = 3 ∧ optionC = 2 * Real.sqrt 3 ∧ optionD = Real.sqrt 6 / 3) ∧
  (optionA < 3 ∧ optionA < 2 * Real.sqrt 3 ∧ optionA < Real.sqrt 6 / 3) :=
  by {
    sorry
  }

end simplest_quadratic_radical_l138_138480


namespace point_in_second_or_third_quadrant_l138_138850

theorem point_in_second_or_third_quadrant (k b : ℝ) (h₁ : k < 0) (h₂ : b ≠ 0) : 
  (k < 0 ∧ b > 0) ∨ (k < 0 ∧ b < 0) :=
by
  sorry

end point_in_second_or_third_quadrant_l138_138850


namespace mod_3_pow_2040_eq_1_mod_5_l138_138046

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l138_138046


namespace tax_rate_correct_l138_138617

/-- The tax rate in dollars per $100.00 is $82.00, given that the tax rate as a percent is 82%. -/
theorem tax_rate_correct (x : ℝ) (h : x = 82) : (x / 100) * 100 = 82 :=
by
  rw [h]
  sorry

end tax_rate_correct_l138_138617


namespace probability_prime_sum_l138_138153

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def dice_roll_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6)

def prime_sum_outcomes : Finset (ℕ × ℕ) :=
  dice_roll_outcomes.filter (λ p, is_prime (p.1 + p.2))

theorem probability_prime_sum : (prime_sum_outcomes.card : ℚ) / (dice_roll_outcomes.card : ℚ) = 5 / 12 :=
by
  sorry

end probability_prime_sum_l138_138153


namespace simplify_expression_l138_138185

theorem simplify_expression :
  1 + (1 / (1 + Real.sqrt 2)) - (1 / (1 - Real.sqrt 5)) =
  1 + ((-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10)) :=
by
  sorry

end simplify_expression_l138_138185


namespace keith_gave_away_p_l138_138546

theorem keith_gave_away_p (k_init : Nat) (m_init : Nat) (final_pears : Nat) (k_gave_away : Nat) (total_init: Nat := k_init + m_init) :
  k_init = 47 →
  m_init = 12 →
  final_pears = 13 →
  k_gave_away = total_init - final_pears →
  k_gave_away = 46 :=
by
  -- Insert proof here (skip using sorry)
  sorry

end keith_gave_away_p_l138_138546


namespace circles_fit_l138_138874

noncomputable def fit_circles_in_rectangle : Prop :=
  ∃ (m n : ℕ) (α : ℝ), (m * n * α * α = 1) ∧ (m * n * α / 2 = 1962)

theorem circles_fit : fit_circles_in_rectangle :=
by sorry

end circles_fit_l138_138874


namespace molecular_weight_of_compound_l138_138501

-- Define the atomic weights for Hydrogen, Chlorine, and Oxygen
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_O : ℝ := 15.999

-- Define the molecular weight of the compound
def molecular_weight (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

-- The proof problem statement
theorem molecular_weight_of_compound :
  molecular_weight atomic_weight_H atomic_weight_Cl atomic_weight_O = 68.456 :=
sorry

end molecular_weight_of_compound_l138_138501


namespace largest_int_less_than_100_remainder_4_l138_138834

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l138_138834


namespace five_fourths_of_fifteen_fourths_l138_138647

theorem five_fourths_of_fifteen_fourths :
  (5 / 4) * (15 / 4) = 75 / 16 := by
  sorry

end five_fourths_of_fifteen_fourths_l138_138647


namespace complex_div_eq_half_add_half_i_l138_138666

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem to be proven
theorem complex_div_eq_half_add_half_i :
  (i / (1 + i)) = (1 / 2 + (1 / 2) * i) :=
by
  -- The proof will go here
  sorry

end complex_div_eq_half_add_half_i_l138_138666


namespace ellipse_focal_distance_l138_138634

theorem ellipse_focal_distance :
  let a := 9
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 4 * Real.sqrt 14 :=
by
  sorry

end ellipse_focal_distance_l138_138634


namespace flowers_per_pot_l138_138034

def total_gardens : ℕ := 10
def pots_per_garden : ℕ := 544
def total_flowers : ℕ := 174080

theorem flowers_per_pot  :
  (total_flowers / (total_gardens * pots_per_garden)) = 32 :=
by
  -- Here would be the place to provide the proof, but we use sorry for now
  sorry

end flowers_per_pot_l138_138034


namespace no_integer_solution_for_expression_l138_138754

theorem no_integer_solution_for_expression (x y z : ℤ) :
  x^4 + y^4 + z^4 - 2 * x^2 * y^2 - 2 * y^2 * z^2 - 2 * z^2 * x^2 ≠ 2000 :=
by sorry

end no_integer_solution_for_expression_l138_138754


namespace days_matt_and_son_eat_only_l138_138890

theorem days_matt_and_son_eat_only (x y : ℕ) 
  (h1 : x + y = 7)
  (h2 : 2 * x + 8 * y = 38) : 
  x = 3 :=
by
  sorry

end days_matt_and_son_eat_only_l138_138890


namespace inequality_transitive_l138_138676

theorem inequality_transitive (a b c : ℝ) (h : a < b) (h' : b < c) : a - c < b - c :=
by
  sorry

end inequality_transitive_l138_138676


namespace mod_3_pow_2040_eq_1_mod_5_l138_138047

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l138_138047


namespace proportional_segments_l138_138677

-- Define the problem
theorem proportional_segments :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → (a * d = b * c) → d = 18 :=
by
  intros a b c d ha hb hc hrat
  rw [ha, hb, hc] at hrat
  exact sorry

end proportional_segments_l138_138677


namespace problem1_problem2_l138_138608

/-- Proof statement for the first mathematical problem -/
theorem problem1 (x : ℝ) (h : (x - 2) ^ 2 = 9) : x = 5 ∨ x = -1 :=
by {
  -- Proof goes here
  sorry
}

/-- Proof statement for the second mathematical problem -/
theorem problem2 (x : ℝ) (h : 27 * (x + 1) ^ 3 + 8 = 0) : x = -5 / 3 :=
by {
  -- Proof goes here
  sorry
}

end problem1_problem2_l138_138608


namespace solution_set_for_inequality_l138_138651

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 4 * x + 5 < 0} = {x : ℝ | x > 5 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l138_138651


namespace ratio_of_members_l138_138635

theorem ratio_of_members (r p : ℕ) (h1 : 5 * r + 12 * p = 8 * (r + p)) : (r / p : ℚ) = 4 / 3 := by
  sorry -- This is a placeholder for the actual proof.

end ratio_of_members_l138_138635


namespace sanAntonioToAustin_passes_austinToSanAntonio_l138_138074

noncomputable def buses_passed : ℕ :=
  let austinToSanAntonio (n : ℕ) : ℕ := n * 2
  let sanAntonioToAustin (n : ℕ) : ℕ := n * 2 + 1
  let tripDuration : ℕ := 3
  if (austinToSanAntonio 3 - 0) <= tripDuration then 2 else 0

-- Proof statement
theorem sanAntonioToAustin_passes_austinToSanAntonio :
  buses_passed = 2 :=
  sorry

end sanAntonioToAustin_passes_austinToSanAntonio_l138_138074


namespace dice_sum_not_20_l138_138557

/-- Given that Louise rolls four standard six-sided dice (with faces numbered from 1 to 6)
    and the product of the numbers on the upper faces is 216, prove that it is not possible
    for the sum of the upper faces to be 20. -/
theorem dice_sum_not_20 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
                        (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
                        (product : a * b * c * d = 216) : a + b + c + d ≠ 20 := 
by sorry

end dice_sum_not_20_l138_138557


namespace solve_m_l138_138715

theorem solve_m (x y m : ℝ) (h1 : 4 * x + 2 * y = 3 * m) (h2 : 3 * x + y = m + 2) (h3 : y = -x) : m = 1 := 
by {
  sorry
}

end solve_m_l138_138715


namespace trapezoid_lower_side_length_l138_138404

variable (U L : ℝ) (height area : ℝ)

theorem trapezoid_lower_side_length
  (h1 : L = U - 3.4)
  (h2 : height = 5.2)
  (h3 : area = 100.62)
  (h4 : area = (1 / 2) * (U + L) * height) :
  L = 17.65 :=
by
  sorry

end trapezoid_lower_side_length_l138_138404


namespace winston_cents_left_l138_138452

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end winston_cents_left_l138_138452


namespace find_missing_number_l138_138304

theorem find_missing_number:
  ∃ x : ℕ, (306 / 34) * 15 + x = 405 := sorry

end find_missing_number_l138_138304


namespace range_of_distances_l138_138553

theorem range_of_distances (t θ : ℝ)
    (x_line y_line : ℝ := 1 - t)
    (y_line : ℝ := 8 - 2 * t)
    (x_curve y_curve : ℝ := 1 + sqrt 5 * cos θ)
    (y_curve : ℝ := -2 + sqrt 5 * sin θ) :
    set_of (λ d : ℝ, 
        ∃ P Q : ℝ × ℝ,
        P = (x_line, y_line) ∧ 
        Q = (x_curve, y_curve) ∧ 
        d = (dist P Q)
    ) = {d | d > sqrt 5} := 
sorry

end range_of_distances_l138_138553


namespace problem1_problem2_l138_138301

-- For problem (1)
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := sorry

-- For problem (2)
theorem problem2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b^2 = a * c) :
  a^2 + b^2 + c^2 > (a - b + c)^2 := sorry

end problem1_problem2_l138_138301


namespace find_dividend_l138_138162

theorem find_dividend :
  ∀ (Divisor Quotient Remainder : ℕ), Divisor = 15 → Quotient = 9 → Remainder = 5 → (Divisor * Quotient + Remainder) = 140 :=
by
  intros Divisor Quotient Remainder hDiv hQuot hRem
  subst hDiv
  subst hQuot
  subst hRem
  sorry

end find_dividend_l138_138162


namespace part1_max_price_part2_min_sales_volume_l138_138138

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def original_revenue : ℝ := original_price * original_sales_volume
noncomputable def max_new_price (t : ℝ) : Prop := t * (130000 - 2000 * t) ≥ original_revenue

theorem part1_max_price (t : ℝ) (ht : max_new_price t) : t ≤ 40 :=
sorry

noncomputable def investment (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600) + 50 + (x / 5)
noncomputable def min_sales_volume (x : ℝ) (a : ℝ) : Prop := a * x ≥ original_revenue + investment x

theorem part2_min_sales_volume (a : ℝ) : min_sales_volume 30 a → a ≥ 10.2 :=
sorry

end part1_max_price_part2_min_sales_volume_l138_138138


namespace integer_triangle_answer_l138_138538

def integer_triangle_condition :=
∀ a r : ℕ, (1 ≤ a ∧ a ≤ 19) → 
(a = 12) → (r = 3) → 
(r = 96 / (20 + a))

theorem integer_triangle_answer : 
  integer_triangle_condition := 
by
  sorry

end integer_triangle_answer_l138_138538


namespace golf_tournament_percentage_increase_l138_138486

theorem golf_tournament_percentage_increase:
  let electricity_bill := 800
  let cell_phone_expenses := electricity_bill + 400
  let golf_tournament_cost := 1440
  (golf_tournament_cost - cell_phone_expenses) / cell_phone_expenses * 100 = 20 :=
by
  sorry

end golf_tournament_percentage_increase_l138_138486


namespace find_coals_per_bag_l138_138773

open Nat

variable (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ)

def coal_per_bag (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ) : ℕ :=
  (totalTime / timePerSet) * burnRate / totalBags

theorem find_coals_per_bag :
  coal_per_bag 15 20 240 3 = 60 :=
by
  sorry

end find_coals_per_bag_l138_138773


namespace product_divisibility_l138_138013

theorem product_divisibility (a b c : ℤ)
  (h₁ : (a + b + c) ^ 2 = -(a * b + a * c + b * c))
  (h₂ : a + b ≠ 0)
  (h₃ : b + c ≠ 0)
  (h₄ : a + c ≠ 0) :
  (a + b) * (a + c) % (b + c) = 0 ∧
  (a + b) * (b + c) % (a + c) = 0 ∧
  (a + c) * (b + c) % (a + b) = 0 := by
  sorry

end product_divisibility_l138_138013


namespace units_digit_k_squared_plus_2_k_l138_138552

noncomputable def k : ℕ := 2009^2 + 2^2009 - 3

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_k_squared_plus_2_k : units_digit (k^2 + 2^k) = 1 := by
  sorry

end units_digit_k_squared_plus_2_k_l138_138552


namespace Vasya_and_Petya_no_mistake_exists_l138_138600

def is_prime (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem Vasya_and_Petya_no_mistake_exists :
  ∃ x : ℝ, (∃ p : ℕ, is_prime p ∧ 10 * x = p) ∧ 
           (∃ q : ℕ, is_prime q ∧ 15 * x = q) :=
sorry

end Vasya_and_Petya_no_mistake_exists_l138_138600


namespace inequality_solution_l138_138354

theorem inequality_solution :
  { x : ℝ | 0 < x ∧ x ≤ 7/3 ∨ 3 ≤ x } = { x : ℝ | (0 < x ∧ x ≤ 7/3) ∨ 3 ≤ x } :=
sorry

end inequality_solution_l138_138354


namespace sum_of_divisors_143_l138_138943

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l138_138943


namespace find_m_of_power_fn_and_increasing_l138_138581

theorem find_m_of_power_fn_and_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) > 0) →
  m^2 - m - 5 = 1 →
  1 < m →
  m = 3 :=
sorry

end find_m_of_power_fn_and_increasing_l138_138581


namespace cevian_concurrency_l138_138887

theorem cevian_concurrency
  (A B C Z X Y : ℝ)
  (a b c s : ℝ)
  (h1 : s = (a + b + c) / 2)
  (h2 : AZ = s - c) (h3 : ZB = s - b)
  (h4 : BX = s - a) (h5 : XC = s - c)
  (h6 : CY = s - b) (h7 : YA = s - a)
  : (AZ / ZB) * (BX / XC) * (CY / YA) = 1 :=
by
  sorry

end cevian_concurrency_l138_138887


namespace percentage_both_correct_l138_138234

theorem percentage_both_correct (p1 p2 pn : ℝ) (h1 : p1 = 0.85) (h2 : p2 = 0.80) (h3 : pn = 0.05) :
  ∃ x, x = 0.70 ∧ x = p1 + p2 - 1 + pn := by
  sorry

end percentage_both_correct_l138_138234


namespace probability_three_white_balls_l138_138463

def total_balls := 11
def white_balls := 5
def black_balls := 6
def balls_drawn := 5
def white_balls_drawn := 3
def black_balls_drawn := 2

theorem probability_three_white_balls :
  let total_outcomes := Nat.choose total_balls balls_drawn
  let favorable_outcomes := (Nat.choose white_balls white_balls_drawn) * (Nat.choose black_balls black_balls_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 77 :=
by
  sorry

end probability_three_white_balls_l138_138463


namespace average_hamburgers_per_day_l138_138627

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l138_138627


namespace min_packs_for_soda_l138_138569

theorem min_packs_for_soda (max_packs : ℕ) (packs : List ℕ) : 
  let num_cans := 95
  let max_each_pack := 4
  let pack_8 := packs.count 8 
  let pack_15 := packs.count 15
  let pack_18 := packs.count 18
  pack_8 ≤ max_each_pack ∧ pack_15 ≤ max_each_pack ∧ pack_18 ≤ max_each_pack ∧ 
  pack_8 * 8 + pack_15 * 15 + pack_18 * 18 = num_cans ∧ 
  pack_8 + pack_15 + pack_18 = max_packs → max_packs = 6 :=
sorry

end min_packs_for_soda_l138_138569


namespace number_multiplied_by_approx_l138_138464

variable (X : ℝ)

theorem number_multiplied_by_approx (h : (0.0048 * X) / (0.05 * 0.1 * 0.004) = 840) : X = 3.5 :=
by
  sorry

end number_multiplied_by_approx_l138_138464


namespace average_price_per_book_l138_138161

theorem average_price_per_book 
  (amount1 : ℝ)
  (books1 : ℕ)
  (amount2 : ℝ)
  (books2 : ℕ)
  (h1 : amount1 = 581)
  (h2 : books1 = 27)
  (h3 : amount2 = 594)
  (h4 : books2 = 20) :
  (amount1 + amount2) / (books1 + books2) = 25 := 
by
  sorry

end average_price_per_book_l138_138161


namespace prob_exactly_two_trains_on_time_is_0_398_l138_138493

-- Definitions and conditions
def eventA := true
def eventB := true
def eventC := true

def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Question definition (to be proved)
def exact_two_on_time : ℝ :=
  P_A * P_B * P_not_C + P_A * P_not_B * P_C + P_not_A * P_B * P_C

-- Theorem statement
theorem prob_exactly_two_trains_on_time_is_0_398 :
  exact_two_on_time = 0.398 := sorry

end prob_exactly_two_trains_on_time_is_0_398_l138_138493


namespace beans_in_jar_l138_138572

theorem beans_in_jar (B : ℕ) 
  (h1 : B / 4 = number_of_red_beans)
  (h2 : number_of_red_beans = B / 4)
  (h3 : number_of_white_beans = (B * 3 / 4) / 3)
  (h4 : number_of_white_beans = B / 4)
  (h5 : number_of_remaining_beans_after_white = B / 2)
  (h6 : 143 = B / 4):
  B = 572 :=
by
  sorry

end beans_in_jar_l138_138572


namespace Martha_cards_l138_138396

theorem Martha_cards :
  let initial_cards := 76.0
  let given_away_cards := 3.0
  initial_cards - given_away_cards = 73.0 :=
by 
  let initial_cards := 76.0
  let given_away_cards := 3.0
  have h : initial_cards - given_away_cards = 73.0 := by sorry
  exact h

end Martha_cards_l138_138396


namespace price_of_each_cake_is_correct_l138_138845

-- Define the conditions
def total_flour : ℕ := 6
def flour_for_cakes : ℕ := 4
def flour_per_cake : ℚ := 0.5
def remaining_flour := total_flour - flour_for_cakes
def flour_per_cupcake : ℚ := 1 / 5
def total_earnings : ℚ := 30
def cupcake_price : ℚ := 1

-- Number of cakes and cupcakes
def number_of_cakes := flour_for_cakes / flour_per_cake
def number_of_cupcakes := remaining_flour / flour_per_cupcake

-- Earnings from cupcakes
def earnings_from_cupcakes := number_of_cupcakes * cupcake_price

-- Earnings from cakes
def earnings_from_cakes := total_earnings - earnings_from_cupcakes

-- Price per cake
def price_per_cake := earnings_from_cakes / number_of_cakes

-- Final statement to prove
theorem price_of_each_cake_is_correct : price_per_cake = 2.50 := by
  sorry

end price_of_each_cake_is_correct_l138_138845


namespace number_of_red_candies_is_4_l138_138594

-- Define the parameters as given in the conditions
def number_of_green_candies : ℕ := 5
def number_of_blue_candies : ℕ := 3
def likelihood_of_blue_candy : ℚ := 25 / 100

-- Define the total number of candies
def total_number_of_candies (number_of_red_candies : ℕ) : ℕ :=
  number_of_green_candies + number_of_blue_candies + number_of_red_candies

-- Define the proof statement
theorem number_of_red_candies_is_4 (R : ℕ) :
  (3 / total_number_of_candies R = 25 / 100) → R = 4 :=
sorry

end number_of_red_candies_is_4_l138_138594


namespace car_traveled_miles_per_gallon_city_l138_138454

noncomputable def miles_per_gallon_city (H C G : ℝ) : Prop :=
  (C = H - 18) ∧ (462 = H * G) ∧ (336 = C * G)

theorem car_traveled_miles_per_gallon_city :
  ∃ H G, miles_per_gallon_city H 48 G :=
by
  sorry

end car_traveled_miles_per_gallon_city_l138_138454


namespace p_sufficient_not_necessary_for_q_l138_138090

noncomputable def p (x : ℝ) : Prop := |x - 3| < 1
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (p x → q x) ∧ (¬ (q x → p x)) := by
  sorry

end p_sufficient_not_necessary_for_q_l138_138090


namespace notebooks_difference_l138_138556

noncomputable def price_more_than_dime (p : ℝ) : Prop := p > 0.10
noncomputable def payment_equation (nL nN : ℕ) (p : ℝ) : Prop :=
  (nL * p = 2.10 ∧ nN * p = 2.80)

theorem notebooks_difference (nL nN : ℕ) (p : ℝ) (h1 : price_more_than_dime p) (h2 : payment_equation nL nN p) :
  nN - nL = 2 :=
by sorry

end notebooks_difference_l138_138556


namespace sin_double_angle_l138_138213

theorem sin_double_angle 
  (α β : ℝ)
  (h1 : 0 < β)
  (h2 : β < α)
  (h3 : α < π / 4)
  (h_cos_diff : Real.cos (α - β) = 12 / 13)
  (h_sin_sum : Real.sin (α + β) = 4 / 5) :
  Real.sin (2 * α) = 63 / 65 := 
sorry

end sin_double_angle_l138_138213


namespace trigonometric_identity_l138_138094

theorem trigonometric_identity (α : Real) (h : (1 + Real.sin α) / Real.cos α = -1 / 2) :
  (Real.cos α) / (Real.sin α - 1) = 1 / 2 :=
sorry

end trigonometric_identity_l138_138094


namespace least_integer_value_x_l138_138430

theorem least_integer_value_x (x : ℤ) (h : |(2 : ℤ) * x + 3| ≤ 12) : x = -7 :=
by
  sorry

end least_integer_value_x_l138_138430


namespace joan_apples_after_giving_l138_138247

-- Definitions of the conditions
def initial_apples : ℕ := 43
def given_away_apples : ℕ := 27

-- Statement to prove
theorem joan_apples_after_giving : (initial_apples - given_away_apples = 16) :=
by sorry

end joan_apples_after_giving_l138_138247


namespace mean_of_remaining_three_l138_138905

theorem mean_of_remaining_three (a b c : ℝ) (h₁ : (a + b + c + 105) / 4 = 93) : (a + b + c) / 3 = 89 :=
  sorry

end mean_of_remaining_three_l138_138905


namespace correct_equation_l138_138859

namespace MathProblem

def is_two_digit_positive_integer (P : ℤ) : Prop :=
  10 ≤ P ∧ P < 100

def equation_A : Prop :=
  ∀ x : ℤ, x^2 + (-98)*x + 2001 = (x - 29) * (x - 69)

def equation_B : Prop :=
  ∀ x : ℤ, x^2 + (-110)*x + 2001 = (x - 23) * (x - 87)

def equation_C : Prop :=
  ∀ x : ℤ, x^2 + 110*x + 2001 = (x + 23) * (x + 87)

def equation_D : Prop :=
  ∀ x : ℤ, x^2 + 98*x + 2001 = (x + 29) * (x + 69)

theorem correct_equation :
  is_two_digit_positive_integer 98 ∧ equation_D :=
  sorry

end MathProblem

end correct_equation_l138_138859


namespace smallest_two_digit_number_product_12_l138_138437

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l138_138437


namespace consecutive_integers_sum_l138_138215

theorem consecutive_integers_sum :
  ∃ (a b : ℤ), a < sqrt 33 ∧ sqrt 33 < b ∧ a + 1 = b ∧ a + b = 11 :=
by
  sorry

end consecutive_integers_sum_l138_138215


namespace area_inner_square_l138_138007

theorem area_inner_square (ABCD_side : ℝ) (BE : ℝ) (EFGH_area : ℝ) 
  (h1 : ABCD_side = Real.sqrt 50) 
  (h2 : BE = 1) :
  EFGH_area = 36 :=
by
  sorry

end area_inner_square_l138_138007


namespace sum_of_roots_of_P_is_8029_l138_138196

-- Define the polynomial
noncomputable def P : Polynomial ℚ :=
  (Polynomial.X - 1)^2008 + 
  3 * (Polynomial.X - 2)^2007 + 
  5 * (Polynomial.X - 3)^2006 + 
  -- Continue defining all terms up to:
  2009 * (Polynomial.X - 2008)^2 + 
  2011 * (Polynomial.X - 2009)

-- The proof problem statement
theorem sum_of_roots_of_P_is_8029 :
  (P.roots.sum = 8029) :=
sorry

end sum_of_roots_of_P_is_8029_l138_138196


namespace ratio_distance_traveled_by_foot_l138_138318

theorem ratio_distance_traveled_by_foot (D F B C : ℕ) (hD : D = 40) 
(hB : B = D / 2) (hC : C = 10) (hF : F = D - (B + C)) : F / D = 1 / 4 := 
by sorry

end ratio_distance_traveled_by_foot_l138_138318


namespace polygon_sides_eq_7_l138_138298

theorem polygon_sides_eq_7 (n : ℕ) (h : n * (n - 3) / 2 = 2 * n) : n = 7 := 
by 
  sorry

end polygon_sides_eq_7_l138_138298


namespace smallest_two_digit_l138_138433

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l138_138433


namespace tangent_point_x_coordinate_l138_138002

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 2 * x

-- The statement to be proved.
theorem tangent_point_x_coordinate (x : ℝ) (h : derivative x = 4) : x = 2 :=
sorry

end tangent_point_x_coordinate_l138_138002


namespace find_a_extreme_values_l138_138513

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)
noncomputable def f' (x a : ℝ) : ℝ := 2 * (x - a) + 4 / (x + 1)

-- Given conditions
theorem find_a (a : ℝ) :
  f' 1 a = 0 ↔ a = 2 :=
by
  sorry

theorem extreme_values :
  ∃ x : ℝ, -1 < x ∧ f (0 : ℝ) 2 = 4 ∨ f (1 : ℝ) 2 = 1 + 4 * Real.log 2 :=
by
  sorry

end find_a_extreme_values_l138_138513


namespace painting_prices_l138_138564

theorem painting_prices (P : ℝ) (h₀ : 55000 = 3.5 * P - 500) : 
  P = 15857.14 :=
by
  -- P represents the average price of the previous three paintings.
  -- Given the condition: 55000 = 3.5 * P - 500
  -- We need to prove: P = 15857.14
  sorry

end painting_prices_l138_138564


namespace Bowen_total_spent_l138_138785

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end Bowen_total_spent_l138_138785


namespace smallest_a_exists_l138_138650

theorem smallest_a_exists : ∃ a b c : ℤ, a > 0 ∧ b^2 > 4*a*c ∧ 
  (∀ x : ℝ, x > 0 ∧ x < 1 → (a * x^2 - b * x + c) = 0 → false) 
  ∧ a = 5 :=
by sorry

end smallest_a_exists_l138_138650


namespace stratified_sampling_major_C_l138_138115

theorem stratified_sampling_major_C
  (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (students_D : ℕ)
  (total_students : ℕ) (sample_size : ℕ)
  (hA : students_A = 150) (hB : students_B = 150) (hC : students_C = 400) (hD : students_D = 300)
  (hTotal : total_students = students_A + students_B + students_C + students_D)
  (hSample : sample_size = 40)
  : students_C * (sample_size / total_students) = 16 :=
by
  sorry

end stratified_sampling_major_C_l138_138115


namespace angle_CAB_EQ_angle_EAD_l138_138885

variable {A B C D E : Type}

-- Define the angles as variables for the pentagon ABCDE
variable (ABC ADE CEA BDA CAB EAD : ℝ)

-- Given conditions
axiom angle_ABC_EQ_angle_ADE : ABC = ADE
axiom angle_CEA_EQ_angle_BDA : CEA = BDA

-- Prove that angle CAB equals angle EAD
theorem angle_CAB_EQ_angle_EAD : CAB = EAD :=
by
  sorry

end angle_CAB_EQ_angle_EAD_l138_138885


namespace average_hamburgers_per_day_l138_138626

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l138_138626


namespace sangeun_initial_money_l138_138566

theorem sangeun_initial_money :
  ∃ (X : ℝ), 
  ((X / 2 - 2000) / 2 - 2000 = 0) ∧ 
  X = 12000 :=
by sorry

end sangeun_initial_money_l138_138566


namespace total_wage_l138_138462

theorem total_wage (work_days_A work_days_B : ℕ) (wage_A : ℕ) (total_wage : ℕ) 
  (h1 : work_days_A = 10) 
  (h2 : work_days_B = 15) 
  (h3 : wage_A = 1980)
  (h4 : (wage_A / (wage_A / (total_wage * 3 / 5))) = 3)
  : total_wage = 3300 :=
sorry

end total_wage_l138_138462


namespace determine_a_l138_138525

theorem determine_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x + 1 > 0) ↔ a = 2 := by
  sorry

end determine_a_l138_138525


namespace percent_difference_z_w_l138_138376

theorem percent_difference_z_w (w x y z : ℝ)
  (h1 : w = 0.60 * x)
  (h2 : x = 0.60 * y)
  (h3 : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
sorry

end percent_difference_z_w_l138_138376


namespace rectangular_field_area_l138_138730

theorem rectangular_field_area
  (x : ℝ) 
  (length := 3 * x) 
  (breadth := 4 * x) 
  (perimeter := 2 * (length + breadth))
  (cost_per_meter : ℝ := 0.25) 
  (total_cost : ℝ := 87.5) 
  (paise_per_rupee : ℝ := 100)
  (perimeter_eq_cost : 14 * x * cost_per_meter * paise_per_rupee = total_cost * paise_per_rupee) :
  (length * breadth = 7500) := 
by
  -- proof omitted
  sorry

end rectangular_field_area_l138_138730


namespace total_strictly_monotonous_positive_integers_l138_138793

def is_strictly_monotonous (n : ℕ) : Prop :=
  (n > 0) ∧
  (n < 10 ∨
    ((list.of_digits (n.digits 10)).nodup ∧
     (list.of_digits (n.digits 10)).pairwise (<) ∨
     (list.of_digits (n.digits 10)).pairwise (>)))

def number_of_strictly_monotonous_numbers : ℕ :=
  9 + ∑ n in finset.range (10 - 2 + 1), 2 * nat.choose 9 n

theorem total_strictly_monotonous_positive_integers: number_of_strictly_monotonous_numbers = 1013 :=
by
  -- Proof would go here
  sorry

end total_strictly_monotonous_positive_integers_l138_138793


namespace machine_a_sprockets_per_hour_l138_138457

theorem machine_a_sprockets_per_hour (s h : ℝ)
    (H1 : 1.1 * s * h = 550)
    (H2 : s * (h + 10) = 550) : s = 5 := by
  sorry

end machine_a_sprockets_per_hour_l138_138457


namespace find_seventh_term_l138_138008

theorem find_seventh_term :
  ∃ r : ℚ, ∃ (a₁ a₇ a₁₀ : ℚ), 
    a₁ = 12 ∧ 
    a₁₀ = 78732 ∧ 
    a₇ = a₁ * r^6 ∧ 
    a₁₀ = a₁ * r^9 ∧ 
    a₇ = 8748 :=
by
  sorry

end find_seventh_term_l138_138008


namespace farm_problem_l138_138561

theorem farm_problem
    (initial_cows : ℕ := 12)
    (initial_pigs : ℕ := 34)
    (remaining_animals : ℕ := 30)
    (C : ℕ)
    (P : ℕ)
    (h1 : P = 3 * C)
    (h2 : initial_cows - C + (initial_pigs - P) = remaining_animals) :
    C = 4 :=
by
  sorry

end farm_problem_l138_138561


namespace fermat_little_theorem_l138_138129

theorem fermat_little_theorem (N p : ℕ) (hp : Nat.Prime p) (hNp : ¬ p ∣ N) : p ∣ (N ^ (p - 1) - 1) := 
sorry

end fermat_little_theorem_l138_138129


namespace mystical_words_count_l138_138139

-- We define a function to count words given the conditions
def count_possible_words : ℕ := 
  let total_words : ℕ := (20^1 - 19^1) + (20^2 - 19^2) + (20^3 - 19^3) + (20^4 - 19^4) + (20^5 - 19^5)
  total_words

theorem mystical_words_count : count_possible_words = 755761 :=
by 
  unfold count_possible_words
  sorry

end mystical_words_count_l138_138139


namespace greatest_positive_integer_N_l138_138500

def condition (x : Int) (y : Int) : Prop :=
  (x^2 - x * y) % 1111 ≠ 0

theorem greatest_positive_integer_N :
  ∃ N : Nat, (∀ (x : Fin N) (y : Fin N), x ≠ y → condition x y) ∧ N = 1000 :=
by
  sorry

end greatest_positive_integer_N_l138_138500


namespace consecutive_integers_sum_l138_138216

theorem consecutive_integers_sum (a b : ℤ) (sqrt_33 : ℝ) (h1 : a < sqrt_33) (h2 : sqrt_33 < b) (h3 : b = a + 1) (h4 : sqrt_33 = Real.sqrt 33) : a + b = 11 :=
  sorry

end consecutive_integers_sum_l138_138216


namespace bodhi_yacht_animals_l138_138559

def total_animals (cows foxes zebras sheep : ℕ) : ℕ :=
  cows + foxes + zebras + sheep

theorem bodhi_yacht_animals :
  ∀ (cows foxes sheep : ℕ), foxes = 15 → cows = 20 → sheep = 20 → total_animals cows foxes (3 * foxes) sheep = 100 :=
by
  intros cows foxes sheep h1 h2 h3
  rw [h1, h2, h3]
  show total_animals 20 15 (3 * 15) 20 = 100
  sorry

end bodhi_yacht_animals_l138_138559


namespace codecracker_total_combinations_l138_138380

theorem codecracker_total_combinations (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) :
  colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num

end codecracker_total_combinations_l138_138380


namespace find_difference_l138_138142

theorem find_difference (m n : ℕ) (hm : ∃ x, m = 111 * x) (hn : ∃ y, n = 31 * y) (h_sum : m + n = 2017) :
  n - m = 463 :=
sorry

end find_difference_l138_138142


namespace largest_int_less_than_100_remainder_4_l138_138835

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l138_138835


namespace train_speed_l138_138476

theorem train_speed (length_m : ℝ) (time_s : ℝ) (h_length : length_m = 133.33333333333334) (h_time : time_s = 8) : 
  let length_km := length_m / 1000
  let time_hr := time_s / 3600
  length_km / time_hr = 60 :=
by
  sorry

end train_speed_l138_138476


namespace train_cross_pole_time_l138_138477

-- Defining the given conditions
def speed_km_hr : ℕ := 54
def length_m : ℕ := 135

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℤ := (54 * 1000) / 3600

-- Statement to be proved
theorem train_cross_pole_time : (length_m : ℤ) / speed_m_s = 9 := by
  sorry

end train_cross_pole_time_l138_138477


namespace range_of_expression_l138_138223

noncomputable def f (x : ℝ) := |Real.log x / Real.log 2|

theorem range_of_expression (a b : ℝ) (h_f_eq : f a = f b) (h_a_lt_b : a < b) :
  f a = f b → a < b → (∃ c > 3, c = (2 / a) + (1 / b)) := by
  sorry

end range_of_expression_l138_138223


namespace three_pow_2040_mod_5_l138_138045

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l138_138045


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138841

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138841


namespace number_of_points_l138_138684

theorem number_of_points (a b : ℤ) : (|a| = 3 ∧ |b| = 2) → ∃! (P : ℤ × ℤ), P = (a, b) :=
by sorry

end number_of_points_l138_138684


namespace example_theorem_l138_138685

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end example_theorem_l138_138685


namespace max_tulips_l138_138427

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end max_tulips_l138_138427


namespace milk_water_ratio_l138_138683

theorem milk_water_ratio (total_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) (added_water : ℕ)
  (h₁ : total_volume = 45) (h₂ : initial_milk_ratio = 4) (h₃ : initial_water_ratio = 1) (h₄ : added_water = 9) :
  (36 : ℕ) / (18 : ℕ) = 2 :=
by sorry

end milk_water_ratio_l138_138683


namespace optimal_playground_dimensions_and_area_l138_138485

theorem optimal_playground_dimensions_and_area:
  ∃ (l w : ℝ), 2 * l + 2 * w = 380 ∧ l ≥ 100 ∧ w ≥ 60 ∧ l * w = 9000 :=
by
  sorry

end optimal_playground_dimensions_and_area_l138_138485


namespace union_M_N_eq_interval_l138_138224

variable {α : Type*} [PartialOrder α]

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem union_M_N_eq_interval :
  M ∪ N = {x | -1/2 < x ∧ x ≤ 1} :=
by
  sorry

end union_M_N_eq_interval_l138_138224


namespace root_diff_condition_l138_138210

noncomputable def g (x : ℝ) : ℝ := 4^x + 2*x - 2
noncomputable def f (x : ℝ) : ℝ := 4*x - 1

theorem root_diff_condition :
  ∃ x₀, g x₀ = 0 ∧ |x₀ - 1/4| ≤ 1/4 ∧ ∃ y₀, f y₀ = 0 ∧ |y₀ - x₀| ≤ 0.25 :=
sorry

end root_diff_condition_l138_138210


namespace unique_function_property_l138_138253

theorem unique_function_property (f : ℕ → ℕ) (h : ∀ m n : ℕ, f m + f n ∣ m + n) :
  ∀ m : ℕ, f m = m :=
by
  sorry

end unique_function_property_l138_138253


namespace prob_three_cards_hearts_king_spade_l138_138737

noncomputable def probability_hearts_king_spade : ℚ :=
  let total_cards := 52
  let total_hearts := 13
  let hearts_kings := 1 -- one king in hearts
  let rest_hearts := 12 -- remaining hearts cards
  let others_suit := 39 -- cards that are not hearts
  let total_kings := 4 -- total number of kings
  let spades := 13 -- total number of spades
  
  -- Probability of the first card is a hearts suit
  let P_first_hearts : ℚ := total_hearts / total_cards
  
  -- Probability of the first card is king of hearts
  let P_first_king_hearts : ℚ := hearts_kings / total_cards
  
  -- Probability of the second card is a king (excluding already drawn king if king of hearts is drawn first)
  let P_second_king_given_first_king_hearts : ℚ := (total_kings - hearts_kings) / (total_cards - 1)
  
  -- Probability of third card being a spade if the first king of hearts is drawn
  let P_third_spade_given_first_king_hearts : ℚ := spades / (total_cards - 2)
  
  -- Combine probabilities for the scenario where the first card is the king of hearts
  let case_king_hearts := P_first_king_hearts * P_second_king_given_first_king_hearts * P_third_spade_given_first_king_hearts
  
  -- Probability of the first card being a hearts (excluding king of hearts)
  let P_first_hearts_not_king : ℚ := rest_hearts / total_cards
  
  -- Probability of second card being any king
  let P_second_king_given_first_hearts_not_king : ℚ := total_kings / (total_cards - 1)
  
  -- Probability of third card being a spade
  let P_third_spade_given_first_hearts_not_king : ℚ := spades / (total_cards - 2)
  
  -- Combine probabilities for the scenario where the first card is a hearts but not the king
  let case_hearts_not_king := P_first_hearts_not_king * P_second_king_given_first_hearts_not_king * P_third_spade_given_first_hearts_not_king
  
  -- Total probability
  case_king_hearts + case_hearts_not_king
  
theorem prob_three_cards_hearts_king_spade : 
  probability_hearts_king_spade = 1 / 200 :=
by
  sorry

end prob_three_cards_hearts_king_spade_l138_138737


namespace evaluate_expression_l138_138749

theorem evaluate_expression : 12^2 + 2 * 12 * 5 + 5^2 = 289 := by
  sorry

end evaluate_expression_l138_138749


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138817

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138817


namespace rain_forest_animals_l138_138724

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l138_138724


namespace f_g_of_1_l138_138861

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 5 * x + 6
def g (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

-- The statement we need to prove
theorem f_g_of_1 : f (g 1) = 132 := by
  sorry

end f_g_of_1_l138_138861


namespace least_positive_multiple_of_17_gt_500_l138_138287

theorem least_positive_multiple_of_17_gt_500 : ∃ n: ℕ, n > 500 ∧ n % 17 = 0 ∧ n = 510 := by
  sorry

end least_positive_multiple_of_17_gt_500_l138_138287


namespace f_2019_eq_2019_l138_138502

def f : ℝ → ℝ := sorry

axiom f_pos : ∀ x, x > 0 → f x > 0
axiom f_one : f 1 = 1
axiom f_eq : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

theorem f_2019_eq_2019 : f 2019 = 2019 :=
by sorry

end f_2019_eq_2019_l138_138502


namespace probability_AC_adjacent_BE_not_adjacent_l138_138420

-- Define the 5 students as elements of a set
inductive Student
| A | B | C | D | E
deriving DecidableEq

open Student

-- Define a function to count valid arrangements with given conditions
def count_valid_arrangements (students : List Student) : Nat := sorry

-- Define the total number of permutations of 5 elements
def total_permutations : Nat := 5!

-- The proof statement
theorem probability_AC_adjacent_BE_not_adjacent :
  (count_valid_arrangements [A, B, C, D, E] / (total_permutations : ℚ) = 1 / 5) :=
by
  sorry

end probability_AC_adjacent_BE_not_adjacent_l138_138420


namespace largest_int_mod_6_less_than_100_l138_138825

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l138_138825


namespace bus_driver_limit_of_hours_l138_138615

theorem bus_driver_limit_of_hours (r o T H L : ℝ)
  (h_reg_rate : r = 16)
  (h_ot_rate : o = 1.75 * r)
  (h_total_comp : T = 752)
  (h_hours_worked : H = 44)
  (h_equation : r * L + o * (H - L) = T) :
  L = 40 :=
  sorry

end bus_driver_limit_of_hours_l138_138615


namespace chicken_price_per_pound_l138_138386

theorem chicken_price_per_pound (beef_pounds chicken_pounds : ℕ) (beef_price chicken_price : ℕ)
    (total_amount : ℕ)
    (h_beef_quantity : beef_pounds = 1000)
    (h_beef_cost : beef_price = 8)
    (h_chicken_quantity : chicken_pounds = 2 * beef_pounds)
    (h_total_price : 1000 * beef_price + chicken_pounds * chicken_price = total_amount)
    (h_total_amount : total_amount = 14000) : chicken_price = 3 :=
by
  sorry

end chicken_price_per_pound_l138_138386


namespace monochromatic_triangle_in_K6_l138_138536

theorem monochromatic_triangle_in_K6 :
  ∀ (color : Fin 6 → Fin 6 → Prop),
  (∀ (a b : Fin 6), a ≠ b → (color a b ↔ color b a)) →
  (∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (color x y = color y z ∧ color y z = color z x)) :=
by
  sorry

end monochromatic_triangle_in_K6_l138_138536


namespace length_of_goods_train_l138_138605

theorem length_of_goods_train 
  (speed_kmh : ℕ) 
  (platform_length_m : ℕ) 
  (cross_time_s : ℕ) :
  speed_kmh = 72 → platform_length_m = 280 → cross_time_s = 26 → 
  ∃ train_length_m : ℕ, train_length_m = 240 :=
by
  intros h1 h2 h3
  sorry

end length_of_goods_train_l138_138605


namespace compute_f_six_l138_138548

def f (x : Int) : Int :=
  if x ≥ 0 then -x^2 - 1 else x + 10

theorem compute_f_six (x : Int) : f (f (f (f (f (f 1))))) = -35 :=
by
  sorry

end compute_f_six_l138_138548


namespace rational_square_root_l138_138507

theorem rational_square_root {x y : ℚ} 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (xy + 1)^2 = 0) : 
  ∃ r : ℚ, r * r = 1 + x * y := 
sorry

end rational_square_root_l138_138507


namespace find_numer_denom_n_l138_138450

theorem find_numer_denom_n (n : ℕ) 
    (h : (2 + n) / (7 + n) = (3 : ℤ) / 4) : n = 13 := sorry

end find_numer_denom_n_l138_138450


namespace map_distance_to_real_distance_l138_138579

theorem map_distance_to_real_distance (d_map : ℝ) (scale : ℝ) (d_real : ℝ) 
    (h1 : d_map = 7.5) (h2 : scale = 8) : d_real = 60 :=
by
  sorry

end map_distance_to_real_distance_l138_138579


namespace smallest_two_digit_number_product_12_l138_138446

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l138_138446


namespace unique_function_property_l138_138812

def f (n : Nat) : Nat := sorry

theorem unique_function_property :
  (∀ x y : ℕ+, x < y → f x < f y) ∧
  (∀ y x : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ n : ℕ+, f n = n^2 :=
by
  intros h
  sorry

end unique_function_property_l138_138812


namespace sum_of_divisors_143_l138_138944

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l138_138944


namespace find_x_l138_138233

theorem find_x (x : ℚ) (h : (35 / 100) * x = (40 / 100) * 50) : 
  x = 400 / 7 :=
sorry

end find_x_l138_138233


namespace divide_8_friends_among_4_teams_l138_138856

def num_ways_to_divide_friends (n : ℕ) (teams : ℕ) :=
  teams ^ n

theorem divide_8_friends_among_4_teams :
  num_ways_to_divide_friends 8 4 = 65536 :=
by sorry

end divide_8_friends_among_4_teams_l138_138856


namespace number_of_boys_l138_138774

-- Definitions of the conditions
def total_members (B G : ℕ) : Prop := B + G = 26
def meeting_attendance (B G : ℕ) : Prop := (1 / 2 : ℚ) * G + B = 16

-- Theorem statement
theorem number_of_boys (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : B = 6 := by
  sorry

end number_of_boys_l138_138774


namespace no_solution_frac_eq_l138_138719

theorem no_solution_frac_eq (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  3 / x + 6 / (x - 1) - (x + 5) / (x * (x - 1)) ≠ 0 :=
by {
  sorry
}

end no_solution_frac_eq_l138_138719


namespace height_of_fourth_person_l138_138299

/-- There are 4 people of different heights standing in order of increasing height.
    The difference is 2 inches between the first person and the second person,
    and also between the second person and the third person.
    The difference between the third person and the fourth person is 6 inches.
    The average height of the four people is 76 inches.
    Prove that the height of the fourth person is 82 inches. -/
theorem height_of_fourth_person 
  (h1 h2 h3 h4 : ℕ) 
  (h2_def : h2 = h1 + 2)
  (h3_def : h3 = h2 + 2)
  (h4_def : h4 = h3 + 6)
  (average_height : (h1 + h2 + h3 + h4) / 4 = 76) 
  : h4 = 82 :=
by sorry

end height_of_fourth_person_l138_138299


namespace modified_pyramid_volume_l138_138625

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) 
  (hV : V = 1/3 * s^2 * h) (hV_eq : V = 72) :
  (1/3) * (3 * s)^2 * (2 * h) = 1296 := by
  sorry

end modified_pyramid_volume_l138_138625


namespace max_comic_books_l138_138010

namespace JasmineComicBooks

-- Conditions
def total_money : ℝ := 12.50
def comic_book_cost : ℝ := 1.15

-- Statement of the theorem
theorem max_comic_books (n : ℕ) (h : n * comic_book_cost ≤ total_money) : n ≤ 10 := by
  sorry

end JasmineComicBooks

end max_comic_books_l138_138010


namespace trigonometric_identity_l138_138987

variable {a b c A B C : ℝ}

theorem trigonometric_identity (h1 : 2 * c^2 - 2 * a^2 = b^2) 
  (cos_A : ℝ) (cos_C : ℝ) 
  (h_cos_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_C : cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * c * cos_A - 2 * a * cos_C = b := 
sorry

end trigonometric_identity_l138_138987


namespace year_proof_l138_138182

variable (n : ℕ)

def packaging_waste_exceeds_threshold (y0 : ℝ) (rate : ℝ) (threshold : ℝ) : Prop :=
  let y := y0 * (rate^n)
  y > threshold

noncomputable def year_when_waste_exceeds := 
  let initial_year := 2015
  let y0 := 4 * 10^6 -- in tons
  let rate := (3.0 / 2.0) -- growth rate per year
  let threshold := 40 * 10^6 -- threshold in tons
  ∃ n, packaging_waste_exceeds_threshold n y0 rate threshold ∧ (initial_year + n = 2021)

theorem year_proof : year_when_waste_exceeds :=
  sorry

end year_proof_l138_138182


namespace remaining_black_cards_l138_138305

-- Define the conditions of the problem
def total_cards : ℕ := 52
def colors : ℕ := 2
def cards_per_color := total_cards / colors
def black_cards_taken_out : ℕ := 5
def total_black_cards : ℕ := cards_per_color

-- Prove the remaining black cards
theorem remaining_black_cards : total_black_cards - black_cards_taken_out = 21 := 
by
  -- Logic to calculate remaining black cards
  sorry

end remaining_black_cards_l138_138305


namespace hockey_team_ties_l138_138519

theorem hockey_team_ties (W T : ℕ) (h1 : 2 * W + T = 60) (h2 : W = T + 12) : T = 12 :=
by
  sorry

end hockey_team_ties_l138_138519


namespace least_multiple_of_11_not_lucky_l138_138311

-- Define what it means for a number to be a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

-- Define what it means for a number to be a multiple of 11
def is_multiple_of_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- State the problem: the least positive multiple of 11 that is not a lucky integer is 132
theorem least_multiple_of_11_not_lucky :
  ∃ n : ℕ, is_multiple_of_11 n ∧ ¬ is_lucky n ∧ n = 132 :=
sorry

end least_multiple_of_11_not_lucky_l138_138311


namespace evaluate_expression_l138_138798

theorem evaluate_expression :
  let a := (1 : ℚ) / 5
  let b := (1 : ℚ) / 3
  let c := (3 : ℚ) / 7
  let d := (1 : ℚ) / 4
  (a + b) / (c - d) = 224 / 75 := by
sorry

end evaluate_expression_l138_138798


namespace find_value_of_c_l138_138960

noncomputable def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem find_value_of_c (b c : ℝ) 
    (h1 : parabola b c 1 = 2)
    (h2 : parabola b c 5 = 2) :
    c = 7 :=
by
  sorry

end find_value_of_c_l138_138960


namespace identify_letter_R_l138_138256

variable (x y : ℕ)

def date_A : ℕ := x + 2
def date_B : ℕ := x + 5
def date_E : ℕ := x

def y_plus_x := y + x
def combined_dates := date_A x + 2 * date_B x

theorem identify_letter_R (h1 : y_plus_x x y = combined_dates x) : 
  y = 2 * x + 12 ∧ ∃ (letter : String), letter = "R" := sorry

end identify_letter_R_l138_138256


namespace rachel_picked_2_apples_l138_138896

def apples_picked (initial_apples picked_apples final_apples : ℕ) : Prop :=
  initial_apples - picked_apples = final_apples

theorem rachel_picked_2_apples (initial_apples final_apples : ℕ)
  (h_initial : initial_apples = 9)
  (h_final : final_apples = 7) :
  apples_picked initial_apples 2 final_apples :=
by
  rw [h_initial, h_final]
  sorry

end rachel_picked_2_apples_l138_138896


namespace complex_quadrant_l138_138602

theorem complex_quadrant (z : ℂ) (h : z * (2 - I) = 2 + I) : 0 < z.re ∧ 0 < z.im := 
sorry

end complex_quadrant_l138_138602


namespace valid_outfit_combinations_l138_138858

theorem valid_outfit_combinations (shirts pants hats shoes : ℕ) (colors : ℕ) 
  (h₁ : shirts = 6) (h₂ : pants = 6) (h₃ : hats = 6) (h₄ : shoes = 6) (h₅ : colors = 6) :
  ∀ (valid_combinations : ℕ),
  (valid_combinations = colors * (colors - 1) * (colors - 2) * (colors - 3)) → valid_combinations = 360 := 
by
  intros valid_combinations h_valid_combinations
  sorry

end valid_outfit_combinations_l138_138858


namespace tangent_line_to_curve_determines_m_l138_138112

theorem tangent_line_to_curve_determines_m :
  ∃ m : ℝ, (∀ x : ℝ, y = x ^ 4 + m * x) ∧ (2 * -1 + y' + 3 = 0) ∧ (y' = -2) → (m = 2) :=
by
  sorry

end tangent_line_to_curve_determines_m_l138_138112


namespace quadratic_inequality_solution_l138_138863

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 9*x + 14 < 0) : 2 < x ∧ x < 7 :=
by
  sorry

end quadratic_inequality_solution_l138_138863


namespace triangle_max_third_side_l138_138901

theorem triangle_max_third_side (D E F : ℝ) (a b : ℝ) (h1 : a = 8) (h2 : b = 15) 
(h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1) 
: ∃ c : ℝ, c = 13 :=
by
  sorry

end triangle_max_third_side_l138_138901


namespace exists_positive_integers_abcd_l138_138509

theorem exists_positive_integers_abcd (m : ℤ) : ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a * b - c * d = m) := by
  sorry

end exists_positive_integers_abcd_l138_138509


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138838

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138838


namespace extreme_points_range_l138_138111

noncomputable def f (a x : ℝ) : ℝ := - (1/2) * x^2 + 4 * x - 2 * a * Real.log x

theorem extreme_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ 0 < a ∧ a < 2 := 
sorry

end extreme_points_range_l138_138111


namespace town_population_growth_l138_138922

noncomputable def populationAfterYears (population : ℝ) (year1Increase : ℝ) (year2Increase : ℝ) : ℝ :=
  let populationAfterFirstYear := population * (1 + year1Increase)
  let populationAfterSecondYear := populationAfterFirstYear * (1 + year2Increase)
  populationAfterSecondYear

theorem town_population_growth :
  ∀ (initialPopulation : ℝ) (year1Increase : ℝ) (year2Increase : ℝ),
    initialPopulation = 1000 → year1Increase = 0.10 → year2Increase = 0.20 →
      populationAfterYears initialPopulation year1Increase year2Increase = 1320 :=
by
  intros initialPopulation year1Increase year2Increase h1 h2 h3
  rw [h1, h2, h3]
  have h4 : populationAfterYears 1000 0.10 0.20 = 1320 := sorry
  exact h4

end town_population_growth_l138_138922


namespace fish_catch_l138_138694

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l138_138694


namespace largest_integer_with_remainder_l138_138829

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l138_138829


namespace number_of_candidates_is_three_l138_138278

variable (votes : List ℕ) (totalVotes : ℕ)

def determineNumberOfCandidates (votes : List ℕ) (totalVotes : ℕ) : ℕ :=
  votes.length

theorem number_of_candidates_is_three (V : ℕ) 
  (h_votes : [2500, 5000, 20000].sum = V) 
  (h_percent : 20000 = 7273 / 10000 * V): 
  determineNumberOfCandidates [2500, 5000, 20000] V = 3 := 
by 
  sorry

end number_of_candidates_is_three_l138_138278


namespace count_integers_congruent_to_7_mod_13_l138_138521

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l138_138521


namespace problem_statement_l138_138888

-- Define the "24-pretty" number as given in the problem
def is_24_pretty (n : ℕ) : Prop :=
  Nat.dvd 24 n ∧ (Nat.divisors n).card = 24

-- Sum of all 24-pretty numbers less than 3000
def S : ℕ := ∑ n in (List.finRange 3000).filter is_24_pretty, n

-- The main statement to prove
theorem problem_statement : S / 24 = 219 := by
  sorry

end problem_statement_l138_138888


namespace mary_shirts_left_l138_138017

theorem mary_shirts_left :
  let blue_shirts := 35
  let brown_shirts := 48
  let red_shirts := 27
  let yellow_shirts := 36
  let green_shirts := 18
  let blue_given_away := 4 / 5 * blue_shirts
  let brown_given_away := 5 / 6 * brown_shirts
  let red_given_away := 2 / 3 * red_shirts
  let yellow_given_away := 3 / 4 * yellow_shirts
  let green_given_away := 1 / 3 * green_shirts
  let blue_left := blue_shirts - blue_given_away
  let brown_left := brown_shirts - brown_given_away
  let red_left := red_shirts - red_given_away
  let yellow_left := yellow_shirts - yellow_given_away
  let green_left := green_shirts - green_given_away
  blue_left + brown_left + red_left + yellow_left + green_left = 45 := by
  sorry

end mary_shirts_left_l138_138017


namespace find_a_b_and_tangent_lines_l138_138368

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

theorem find_a_b_and_tangent_lines (a b : ℝ) :
  (3 * (-2 / 3)^2 + 2 * a * (-2 / 3) + b = 0) ∧
  (3 * 1^2 + 2 * a * 1 + b = 0) →
  a = -1 / 2 ∧ b = -2 ∧
  (∀ t : ℝ, f t a b = (t^3 + (a - 1 / 2) * t^2 - 2 * t + 1) → 
     (f t a b - (3 * t^2 - t - 2) * (0 - t) = 1) →
       (3 * t^2 - t - 2 = (t * (3 * (t - t))) ) → 
          ((2 * 0 + f 0 a b) = 1) ∨ (33 * 0 + 16 * 1 - 16 = 1)) :=
sorry

end find_a_b_and_tangent_lines_l138_138368


namespace population_of_village_l138_138764

-- Define the given condition
def total_population (P : ℝ) : Prop :=
  0.4 * P = 23040

-- The theorem to prove that the total population is 57600
theorem population_of_village : ∃ P : ℝ, total_population P ∧ P = 57600 :=
by
  sorry

end population_of_village_l138_138764


namespace mrs_taylor_total_payment_l138_138255

-- Declaring the price of items and discounts
def price_tv : ℝ := 750
def price_soundbar : ℝ := 300

def discount_tv : ℝ := 0.15
def discount_soundbar : ℝ := 0.10

-- Total number of each items
def num_tv : ℕ := 2
def num_soundbar : ℕ := 3

-- Total cost calculation after discounts
def total_cost_tv := num_tv * price_tv * (1 - discount_tv)
def total_cost_soundbar := num_soundbar * price_soundbar * (1 - discount_soundbar)
def total_cost := total_cost_tv + total_cost_soundbar

-- The theorem we want to prove
theorem mrs_taylor_total_payment : total_cost = 2085 := by
  -- Skipping the proof
  sorry

end mrs_taylor_total_payment_l138_138255


namespace total_meters_examined_l138_138183

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end total_meters_examined_l138_138183


namespace simplify_expression_l138_138262

variable (z : ℝ)

theorem simplify_expression: (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z :=
by sorry

end simplify_expression_l138_138262


namespace largest_divisor_prime_cube_diff_l138_138088

theorem largest_divisor_prime_cube_diff (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge5 : p ≥ 5) : 
  ∃ k, k = 12 ∧ ∀ n, n ∣ (p^3 - p) ↔ n ∣ 12 :=
by
  sorry

end largest_divisor_prime_cube_diff_l138_138088


namespace max_tulips_l138_138426

theorem max_tulips (r y : ℕ) (h₁ : r + y = 2 * (y : ℕ) + 1) (h₂ : |r - y| = 1) (h₃ : 50 * y + 31 * r ≤ 600) :
    r + y = 15 :=
sorry

end max_tulips_l138_138426


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138840

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138840


namespace management_sampled_count_l138_138466

variable (total_employees salespeople management_personnel logistical_support staff_sample_size : ℕ)
variable (proportional_sampling : Prop)
variable (n_management_sampled : ℕ)

axiom h1 : total_employees = 160
axiom h2 : salespeople = 104
axiom h3 : management_personnel = 32
axiom h4 : logistical_support = 24
axiom h5 : proportional_sampling
axiom h6 : staff_sample_size = 20

theorem management_sampled_count : n_management_sampled = 4 :=
by
  -- The proof is omitted as per instructions
  sorry

end management_sampled_count_l138_138466


namespace smallest_two_digit_l138_138432

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l138_138432


namespace quadratic_equation_problems_l138_138220

noncomputable def quadratic_has_real_roots (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  Δ ≥ 0

noncomputable def valid_m_values (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  1 = m ∨ -1 / 3 = m

theorem quadratic_equation_problems (m : ℝ) :
  quadratic_has_real_roots m ∧
  (∀ x1 x2 : ℝ, 
      (x1 ≠ x2) →
      x1 + x2 = -(3 * m - 1) / m →
      x1 * x2 = (2 * m - 2) / m →
      abs (x1 - x2) = 2 →
      valid_m_values m) :=
by 
  sorry

end quadratic_equation_problems_l138_138220


namespace arccos_equivalence_l138_138206

open Real

theorem arccos_equivalence (α : ℝ) (h₀ : α ∈ Set.Icc 0 (2 * π)) (h₁ : cos α = 1 / 3) :
  α = arccos (1 / 3) ∨ α = 2 * π - arccos (1 / 3) := 
by 
  sorry

end arccos_equivalence_l138_138206


namespace triangle_area_l138_138533

theorem triangle_area (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 := 
by 
  sorry

end triangle_area_l138_138533


namespace sufficient_but_not_necessary_condition_for_x_1_l138_138474

noncomputable def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
(x = 1 → (x = 1 ∨ x = 2)) ∧ ¬ ((x = 1 ∨ x = 2) → x = 1)

theorem sufficient_but_not_necessary_condition_for_x_1 :
  sufficient_but_not_necessary_condition 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_for_x_1_l138_138474


namespace denomination_other_currency_notes_l138_138011

noncomputable def denomination_proof : Prop :=
  ∃ D x y : ℕ, 
  (x + y = 85) ∧
  (100 * x + D * y = 5000) ∧
  (D * y = 3500) ∧
  (D = 50)

theorem denomination_other_currency_notes :
  denomination_proof :=
sorry

end denomination_other_currency_notes_l138_138011


namespace initial_num_families_eq_41_l138_138453

-- Definitions based on the given conditions
def num_families_flew_away : ℕ := 27
def num_families_left : ℕ := 14

-- Statement to prove
theorem initial_num_families_eq_41 : num_families_flew_away + num_families_left = 41 := by
  sorry

end initial_num_families_eq_41_l138_138453


namespace polygon_is_quadrilateral_l138_138314

-- Problem statement in Lean 4
theorem polygon_is_quadrilateral 
  (n : ℕ) 
  (h₁ : (n - 2) * 180 = 360) :
  n = 4 :=
by
  sorry

end polygon_is_quadrilateral_l138_138314


namespace point_on_graph_l138_138369

theorem point_on_graph (g : ℝ → ℝ) (h : g 8 = 10) :
  ∃ x y : ℝ, 3 * y = g (3 * x - 1) + 3 ∧ x = 3 ∧ y = 13 / 3 ∧ x + y = 22 / 3 :=
by
  sorry

end point_on_graph_l138_138369


namespace cos_difference_identity_cos_phi_value_l138_138761

variables (α β θ φ : ℝ)
variables (a b : ℝ × ℝ)

-- Part I
theorem cos_difference_identity (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) (hβ : 0 ≤ β ∧ β ≤ 2 * Real.pi) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
sorry

-- Part II
theorem cos_phi_value (hθ : 0 < θ ∧ θ < Real.pi / 2) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (ha : a = (Real.sin θ, -2)) (hb : b = (1, Real.cos θ)) (dot_ab_zero : a.1 * b.1 + a.2 * b.2 = 0)
  (h_sin_diff : Real.sin (theta - phi) = Real.sqrt 10 / 10) :
  Real.cos φ = Real.sqrt 2 / 2 :=
sorry

end cos_difference_identity_cos_phi_value_l138_138761


namespace union_of_A_and_B_l138_138001

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set B
def B := {x : ℝ | x < 1}

-- The proof problem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} :=
by sorry

end union_of_A_and_B_l138_138001


namespace probability_two_red_balls_l138_138614

def total_balls : ℕ := 15
def red_balls_initial : ℕ := 7
def blue_balls_initial : ℕ := 8
def red_balls_after_first_draw : ℕ := 6
def remaining_balls_after_first_draw : ℕ := 14

theorem probability_two_red_balls :
  (red_balls_initial / total_balls) *
  (red_balls_after_first_draw / remaining_balls_after_first_draw) = 1 / 5 :=
by sorry

end probability_two_red_balls_l138_138614


namespace slope_tangent_line_at_origin_l138_138348

open Real

theorem slope_tangent_line_at_origin :
  deriv (λ x : ℝ, exp x) 0 = exp 0 := by
  sorry

end slope_tangent_line_at_origin_l138_138348


namespace count_C_sets_l138_138204

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

-- The predicate that a set C satisfies B ∪ C = A
def satisfies_condition (C : Set ℕ) : Prop := B ∪ C = A

-- The claim that there are exactly 4 such sets C
theorem count_C_sets : 
  ∃ (C1 C2 C3 C4 : Set ℕ), 
    (satisfies_condition C1 ∧ satisfies_condition C2 ∧ satisfies_condition C3 ∧ satisfies_condition C4) 
    ∧ 
    (∀ C', satisfies_condition C' → C' = C1 ∨ C' = C2 ∨ C' = C3 ∨ C' = C4)
    ∧ 
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4) := 
sorry

end count_C_sets_l138_138204


namespace factorize_difference_of_squares_factorize_cubic_l138_138799

-- Problem 1: Prove that 4x^2 - 36 = 4(x + 3)(x - 3)
theorem factorize_difference_of_squares (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := 
  sorry

-- Problem 2: Prove that x^3 - 2x^2y + xy^2 = x(x - y)^2
theorem factorize_cubic (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
  sorry

end factorize_difference_of_squares_factorize_cubic_l138_138799


namespace identity_proof_l138_138531

theorem identity_proof (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 55) : x^2 - y^2 = 1 / 121 :=
by 
  sorry

end identity_proof_l138_138531


namespace number_of_people_in_group_l138_138405

theorem number_of_people_in_group 
    (N : ℕ)
    (old_person_weight : ℕ) (new_person_weight : ℕ)
    (average_weight_increase : ℕ) :
    old_person_weight = 70 →
    new_person_weight = 94 →
    average_weight_increase = 3 →
    N * average_weight_increase = new_person_weight - old_person_weight →
    N = 8 :=
by
  sorry

end number_of_people_in_group_l138_138405


namespace least_t_geometric_progression_exists_l138_138076

open Real

theorem least_t_geometric_progression_exists :
  ∃ (t : ℝ),
  (∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧
             (arcsin (sin α) = α ∧
              arcsin (sin (3 * α)) = 3 * α ∧
              arcsin (sin (8 * α)) = 8 * α) ∧
              (arcsin (sin (t * α)) = (some_ratio) * (arcsin (sin (8 * α))) )) ∧ 
   0 < t := 
by 
  sorry

end least_t_geometric_progression_exists_l138_138076


namespace find_k_l138_138370

theorem find_k (x : ℝ) (k : ℝ) (h : 2 * x - 3 = 3 * x - 2 + k) (h_solution : x = 2) : k = -3 := by
  sorry

end find_k_l138_138370


namespace max_tulips_count_l138_138428

theorem max_tulips_count : ∃ (r y n : ℕ), 
  n = r + y ∧ 
  n % 2 = 1 ∧ 
  |r - y| = 1 ∧ 
  50 * y + 31 * r ≤ 600 ∧ 
  n = 15 := 
by
  sorry

end max_tulips_count_l138_138428


namespace min_cubes_l138_138068

-- Define the conditions
structure Cube := (x : ℕ) (y : ℕ) (z : ℕ)
def shares_face (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z = c2.z - 1)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

def front_view (cubes : List Cube) : Prop :=
  -- Representation of L-shape in xy-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 1 ∧ c2.y = 0 ∧ c2.z = 0) ∧
  (c3.x = 2 ∧ c3.y = 0 ∧ c3.z = 0) ∧
  (c4.x = 2 ∧ c4.y = 1 ∧ c4.z = 0) ∧
  (c5.x = 1 ∧ c5.y = 2 ∧ c5.z = 0)

def side_view (cubes : List Cube) : Prop :=
  -- Representation of Z-shape in yz-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 0 ∧ c2.y = 1 ∧ c2.z = 0) ∧
  (c3.x = 0 ∧ c3.y = 1 ∧ c3.z = 1) ∧
  (c4.x = 0 ∧ c4.y = 2 ∧ c4.z = 1) ∧
  (c5.x = 0 ∧ c5.y = 2 ∧ c5.z = 2)

-- Proof statement
theorem min_cubes (cubes : List Cube) (h1 : front_view cubes) (h2 : side_view cubes) : cubes.length = 5 :=
by sorry

end min_cubes_l138_138068


namespace selling_price_of_cricket_bat_l138_138307

variable (profit : ℝ) (profit_percentage : ℝ)
variable (selling_price : ℝ)

theorem selling_price_of_cricket_bat 
  (h1 : profit = 215)
  (h2 : profit_percentage = 33.85826771653544) : 
  selling_price = 849.70 :=
sorry

end selling_price_of_cricket_bat_l138_138307


namespace flower_selection_l138_138776

theorem flower_selection : Nat.choose 10 6 = 210 := 
by
  sorry

end flower_selection_l138_138776


namespace non_congruent_squares_count_l138_138228

theorem non_congruent_squares_count (n : ℕ) (h : n = 6) : 
  let standard_squares := (finset.range 5).sum (λ k, (n - k)^2)
  let tilted_squares := (finset.range 5).sum (λ i, (match i with
    | 0 => (n-1)^2
    | 1 => (n-2)^2
    | 2 => 2 * (n-2) * (n-1)
    | 3 => 2 * (n-3) * (n-1)
    | 4 => 0
    | _ => 0))
  in standard_squares + tilted_squares = 201 :=
by
  sorry

end non_congruent_squares_count_l138_138228


namespace tank_capacity_is_48_l138_138028

-- Define the conditions
def num_4_liter_bucket_used : ℕ := 12
def num_3_liter_bucket_used : ℕ := num_4_liter_bucket_used + 4

-- Define the capacities of the buckets and the tank
def bucket_4_liters_capacity : ℕ := 4 * num_4_liter_bucket_used
def bucket_3_liters_capacity : ℕ := 3 * num_3_liter_bucket_used

-- Tank capacity
def tank_capacity : ℕ := 48

-- Statement to prove
theorem tank_capacity_is_48 : 
    bucket_4_liters_capacity = tank_capacity ∧
    bucket_3_liters_capacity = tank_capacity := by
  sorry

end tank_capacity_is_48_l138_138028


namespace omega_eq_six_l138_138038

theorem omega_eq_six (A ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : -π / 2 < φ ∧ φ < π / 2) (h4 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h5 : ∀ x, f (-x) = -f x) 
  (h6 : ∀ x, f (x + π / 6) = -f (x - π / 6)) :
  ω = 6 :=
sorry

end omega_eq_six_l138_138038


namespace complement_subset_lemma_l138_138103

-- Definitions for sets P and Q
def P : Set ℝ := {x | 0 < x ∧ x < 1}

def Q : Set ℝ := {x | x^2 + x - 2 ≤ 0}

-- Definition for complement of a set
def C_ℝ (A : Set ℝ) : Set ℝ := {x | ¬(x ∈ A)}

-- Prove the required relationship
theorem complement_subset_lemma : C_ℝ Q ⊆ C_ℝ P :=
by
  -- The proof steps will go here
  sorry

end complement_subset_lemma_l138_138103


namespace find_increase_in_perimeter_l138_138950

variable (L B y : ℕ)

theorem find_increase_in_perimeter (h1 : 2 * (L + y + (B + y)) = 2 * (L + B) + 16) : y = 4 := by
  sorry

end find_increase_in_perimeter_l138_138950


namespace gym_class_students_l138_138777

theorem gym_class_students :
  ∃ n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 6 = 3 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ (n = 165 ∨ n = 237) :=
by
  sorry

end gym_class_students_l138_138777


namespace sum_is_zero_l138_138073

-- Define the conditions: the function f is invertible, and f(a) = 3, f(b) = 7
variables {α β : Type} [Inhabited α] [Inhabited β]

def invertible {α β : Type} (f : α → β) :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

variables (f : ℝ → ℝ) (a b : ℝ)

-- Assume f is invertible and the given conditions f(a) = 3 and f(b) = 7
axiom f_invertible : invertible f
axiom f_a : f a = 3
axiom f_b : f b = 7

-- Prove that a + b = 0
theorem sum_is_zero : a + b = 0 :=
sorry

end sum_is_zero_l138_138073


namespace pqr_value_l138_138900

theorem pqr_value (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h1 : p + q + r = 24)
  (h2 : (1 / p : ℚ) + (1 / q) + (1 / r) + 240 / (p * q * r) = 1): 
  p * q * r = 384 :=
by
  sorry

end pqr_value_l138_138900


namespace gondor_laptops_wednesday_l138_138517

/-- Gondor's phone repair earnings per unit -/
def phone_earning : ℕ := 10

/-- Gondor's laptop repair earnings per unit -/
def laptop_earning : ℕ := 20

/-- Number of phones repaired on Monday -/
def phones_monday : ℕ := 3

/-- Number of phones repaired on Tuesday -/
def phones_tuesday : ℕ := 5

/-- Number of laptops repaired on Thursday -/
def laptops_thursday : ℕ := 4

/-- Total earnings of Gondor -/
def total_earnings : ℕ := 200

/-- Number of laptops repaired on Wednesday, which we need to prove equals 2 -/
def laptops_wednesday : ℕ := 2

theorem gondor_laptops_wednesday : 
    (phones_monday * phone_earning + phones_tuesday * phone_earning + 
    laptops_thursday * laptop_earning + laptops_wednesday * laptop_earning = total_earnings) :=
by
    sorry

end gondor_laptops_wednesday_l138_138517


namespace two_digit_integers_count_l138_138854

def digits : Set ℕ := {3, 5, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem two_digit_integers_count : 
  ∃ (count : ℕ), count = 16 ∧
  (∀ (t : ℕ), t ∈ digits → 
  ∀ (u : ℕ), u ∈ digits → 
  t ≠ u ∧ is_odd u → 
  (∃ n : ℕ, 10 * t + u = n)) :=
by
  -- The total number of unique two-digit integers is 16
  use 16
  -- Proof skipped
  sorry

end two_digit_integers_count_l138_138854


namespace fifth_term_arithmetic_sequence_l138_138732

theorem fifth_term_arithmetic_sequence (a d : ℤ) 
  (h_twentieth : a + 19 * d = 12) 
  (h_twenty_first : a + 20 * d = 16) : 
  a + 4 * d = -48 := 
by sorry

end fifth_term_arithmetic_sequence_l138_138732


namespace expand_polynomials_l138_138349

-- Define the given polynomials
def poly1 (x : ℝ) : ℝ := 12 * x^2 + 5 * x - 3
def poly2 (x : ℝ) : ℝ := 3 * x^3 + 2

-- Define the expected result of the polynomial multiplication
def expected (x : ℝ) : ℝ := 36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6

-- State the theorem
theorem expand_polynomials (x : ℝ) :
  (poly1 x) * (poly2 x) = expected x :=
by
  sorry

end expand_polynomials_l138_138349


namespace find_13_points_within_radius_one_l138_138868

theorem find_13_points_within_radius_one (points : Fin 25 → ℝ × ℝ)
  (h : ∀ i j k : Fin 25, min (dist (points i) (points j)) (min (dist (points i) (points k)) (dist (points j) (points k))) < 1) :
  ∃ (subset : Finset (Fin 25)), subset.card = 13 ∧ ∃ (center : ℝ × ℝ), ∀ i ∈ subset, dist (points i) center < 1 :=
  sorry

end find_13_points_within_radius_one_l138_138868


namespace sum_of_divisors_of_143_l138_138939

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l138_138939


namespace eval_polynomial_eq_5_l138_138496

noncomputable def eval_polynomial_at_root : ℝ :=
let x := (3 + 3 * Real.sqrt 5) / 2 in
if h : x^2 - 3*x - 9 = 0 then x^3 - 3*x^2 - 9*x + 5 else 0

theorem eval_polynomial_eq_5 :
  eval_polynomial_at_root = 5 :=
by
  sorry

end eval_polynomial_eq_5_l138_138496


namespace jane_doe_gift_l138_138698

theorem jane_doe_gift (G : ℝ) (h1 : 0.25 * G + 0.1125 * (0.75 * G) = 15000) : G = 41379 := 
sorry

end jane_doe_gift_l138_138698


namespace find_number_of_lines_l138_138116

theorem find_number_of_lines (n : ℕ) (h : (n * (n - 1) / 2) * 8 = 280) : n = 10 :=
by
  sorry

end find_number_of_lines_l138_138116


namespace double_root_divisors_l138_138471

theorem double_root_divisors (b3 b2 b1 s : ℤ) (h : 0 = (s^2) • (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50)) : 
  s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 :=
by
  sorry

end double_root_divisors_l138_138471


namespace exists_smallest_positive_period_even_function_l138_138782

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

noncomputable def functions : List (ℝ → ℝ) :=
  [
    (λ x => Real.sin (2 * x + Real.pi / 2)),
    (λ x => Real.cos (2 * x + Real.pi / 2)),
    (λ x => Real.sin (2 * x) + Real.cos (2 * x)),
    (λ x => Real.sin x + Real.cos x)
  ]

def smallest_positive_period_even_function : ℝ → Prop :=
  λ T => ∃ f ∈ functions, is_even_function f ∧ period f T ∧ T > 0

theorem exists_smallest_positive_period_even_function :
  smallest_positive_period_even_function Real.pi :=
sorry

end exists_smallest_positive_period_even_function_l138_138782


namespace solve_for_a_and_b_l138_138212

noncomputable def A := {x : ℝ | (-2 < x ∧ x < -1) ∨ (x > 1)}
noncomputable def B (a b : ℝ) := {x : ℝ | a ≤ x ∧ x < b}

theorem solve_for_a_and_b (a b : ℝ) :
  (A ∪ B a b = {x : ℝ | x > -2}) ∧ (A ∩ B a b = {x : ℝ | 1 < x ∧ x < 3}) →
  a = -1 ∧ b = 3 :=
by
  sorry

end solve_for_a_and_b_l138_138212


namespace opposite_of_neg_sqrt_two_l138_138586

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end opposite_of_neg_sqrt_two_l138_138586


namespace misread_signs_in_front_of_6_terms_l138_138603

/-- Define the polynomial function --/
def poly (x : ℝ) : ℝ :=
  10 * x ^ 9 + 9 * x ^ 8 + 8 * x ^ 7 + 7 * x ^ 6 + 6 * x ^ 5 + 5 * x ^ 4 + 4 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

/-- Xiao Ming's mistaken result --/
def mistaken_result : ℝ := 7

/-- Correct value of the expression at x = -1 --/
def correct_value : ℝ := poly (-1)

/-- The difference due to misreading signs --/
def difference : ℝ := mistaken_result - correct_value

/-- Prove that Xiao Ming misread the signs in front of 6 terms --/
theorem misread_signs_in_front_of_6_terms :
  difference / 2 = 6 :=
by
  simp [difference, correct_value, poly]
  -- the proof steps would go here
  sorry

#eval poly (-1)  -- to validate the correct value
#eval mistaken_result - poly (-1)  -- to validate the difference

end misread_signs_in_front_of_6_terms_l138_138603


namespace calculate_cells_after_12_days_l138_138322

theorem calculate_cells_after_12_days :
  let initial_cells := 5
  let division_factor := 3
  let days := 12
  let period := 3
  let n := days / period
  initial_cells * division_factor ^ (n - 1) = 135 := by
  sorry

end calculate_cells_after_12_days_l138_138322


namespace exists_triplet_with_gcd_conditions_l138_138547

-- Given the conditions as definitions in Lean.
variables (S : Set ℕ)
variable [Infinite S] -- S is an infinite set of positive integers.
variables {a b c d x y z : ℕ}
variable (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
variable (hdistinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) 
variable (hgcd_neq : gcd a b ≠ gcd c d)

-- The formal proof statement.
theorem exists_triplet_with_gcd_conditions :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x :=
sorry

end exists_triplet_with_gcd_conditions_l138_138547


namespace area_of_square_l138_138661

noncomputable def square_area (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) : ℝ :=
  (v * v) / 4

theorem area_of_square (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) (h_cond : ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → B = (u, 0) → C = (u, v) → 
  (u - 0) * (u - 0) + (v - 0) * (v - 0) = (u - 0) * (u - 0)) :
  square_area u v h_u h_v = v * v / 4 := 
by 
  sorry

end area_of_square_l138_138661


namespace length_gh_parallel_lines_l138_138187

theorem length_gh_parallel_lines (
    AB CD EF GH : ℝ
) (
    h1 : AB = 300
) (
    h2 : CD = 200
) (
    h3 : EF = (AB + CD) / 2 * (1 / 2)
) (
    h4 : GH = EF * (1 - 1 / 4)
) :
    GH = 93.75 :=
by
    sorry

end length_gh_parallel_lines_l138_138187


namespace find_smallest_in_arithmetic_progression_l138_138654

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end find_smallest_in_arithmetic_progression_l138_138654


namespace eval_polynomial_eq_5_l138_138497

noncomputable def eval_polynomial_at_root : ℝ :=
let x := (3 + 3 * Real.sqrt 5) / 2 in
if h : x^2 - 3*x - 9 = 0 then x^3 - 3*x^2 - 9*x + 5 else 0

theorem eval_polynomial_eq_5 :
  eval_polynomial_at_root = 5 :=
by
  sorry

end eval_polynomial_eq_5_l138_138497


namespace number_of_diamonds_in_F10_l138_138340

def sequence_of_figures (F : ℕ → ℕ) : Prop :=
  F 1 = 4 ∧
  (∀ n ≥ 2, F n = F (n-1) + 4 * (n + 2)) ∧
  F 3 = 28

theorem number_of_diamonds_in_F10 (F : ℕ → ℕ) (h : sequence_of_figures F) : F 10 = 336 :=
by
  sorry

end number_of_diamonds_in_F10_l138_138340


namespace new_concentration_of_mixture_l138_138753

theorem new_concentration_of_mixture
  (v1_cap : ℝ) (v1_alcohol_percent : ℝ)
  (v2_cap : ℝ) (v2_alcohol_percent : ℝ)
  (new_vessel_cap : ℝ) (poured_liquid : ℝ)
  (filled_water : ℝ) :
  v1_cap = 2 →
  v1_alcohol_percent = 0.25 →
  v2_cap = 6 →
  v2_alcohol_percent = 0.50 →
  new_vessel_cap = 10 →
  poured_liquid = 8 →
  filled_water = (new_vessel_cap - poured_liquid) →
  ((v1_cap * v1_alcohol_percent + v2_cap * v2_alcohol_percent) / new_vessel_cap) = 0.35 :=
by
  intros v1_h v1_per_h v2_h v2_per_h v_new_h poured_h filled_h
  sorry

end new_concentration_of_mixture_l138_138753


namespace mod_congruent_integers_l138_138524

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l138_138524


namespace candy_bars_saved_l138_138881

theorem candy_bars_saved
  (candy_bars_per_week : ℕ)
  (weeks : ℕ)
  (candy_bars_eaten_per_4_weeks : ℕ) :
  candy_bars_per_week = 2 →
  weeks = 16 →
  candy_bars_eaten_per_4_weeks = 1 →
  (candy_bars_per_week * weeks) - (weeks / 4 * candy_bars_eaten_per_4_weeks) = 28 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end candy_bars_saved_l138_138881


namespace max_value_a_l138_138202

theorem max_value_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) ↔ a ≤ 6 := by
  sorry

end max_value_a_l138_138202


namespace range_of_a1_l138_138996

theorem range_of_a1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq : ∀ n, 12 * S n = 4 * a (n + 1) + 5^n - 13)
  (h_S4 : ∀ n, S n ≤ S 4):
  13 / 48 ≤ a 1 ∧ a 1 ≤ 59 / 64 :=
sorry

end range_of_a1_l138_138996


namespace initial_balls_in_bag_l138_138378

theorem initial_balls_in_bag (n : ℕ) 
  (h_add_white : ∀ x : ℕ, x = n + 1)
  (h_probability : (5 / 8) = 0.625):
  n = 7 :=
sorry

end initial_balls_in_bag_l138_138378


namespace value_of_b_l138_138374

variable (a b c : ℕ)
variable (h_a_nonzero : a ≠ 0)
variable (h_a : a < 8)
variable (h_b : b < 8)
variable (h_c : c < 8)
variable (h_square : ∃ k, k^2 = a * 8^3 + 3 * 8^2 + b * 8 + c)

theorem value_of_b : b = 1 :=
by sorry

end value_of_b_l138_138374


namespace intersection_of_A_and_B_l138_138993

-- Definitions based on conditions
def A : Set ℝ := { x | x + 2 = 0 }
def B : Set ℝ := { x | x^2 - 4 = 0 }

-- Theorem statement proving the question == answer given conditions
theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by 
  sorry

end intersection_of_A_and_B_l138_138993


namespace grandmother_age_l138_138709

theorem grandmother_age (minyoung_age_current : ℕ)
                         (minyoung_age_future : ℕ)
                         (grandmother_age_future : ℕ)
                         (h1 : minyoung_age_future = minyoung_age_current + 3)
                         (h2 : grandmother_age_future = 65)
                         (h3 : minyoung_age_future = 10) : grandmother_age_future - (minyoung_age_future -minyoung_age_current) = 62 := by
  sorry

end grandmother_age_l138_138709


namespace icosahedron_to_octahedron_l138_138159

theorem icosahedron_to_octahedron : 
  ∃ (f : Finset (Fin 20)), f.card = 8 ∧ 
  (∀ {o : Finset (Fin 8)}, (True ∧ True)) ∧
  (∃ n : ℕ, n = 5) := by
  sorry

end icosahedron_to_octahedron_l138_138159


namespace Rachel_brought_25_cookies_l138_138558

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Total_cookies : ℕ := 60

theorem Rachel_brought_25_cookies : (Total_cookies - (Mona_cookies + Jasmine_cookies) = 25) :=
by
  sorry

end Rachel_brought_25_cookies_l138_138558


namespace find_units_digit_l138_138423

theorem find_units_digit (A : ℕ) (h : 10 * A + 2 = 20 + A + 9) : A = 3 :=
by
  sorry

end find_units_digit_l138_138423


namespace lcm_quadruples_count_l138_138062

-- Define the problem conditions
variables (r s : ℕ) (hr : r > 0) (hs : s > 0)

-- Define the mathematical problem statement
theorem lcm_quadruples_count :
  ( ∀ (a b c d : ℕ),
    lcm (lcm a b) c = lcm (lcm a b) d ∧
    lcm (lcm a b) c = lcm (lcm a c) d ∧
    lcm (lcm a b) c = lcm (lcm b c) d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a = 3 ^ r * 7 ^ s ∧
    b = 3 ^ r * 7 ^ s ∧
    c = 3 ^ r * 7 ^ s ∧
    d = 3 ^ r * 7 ^ s 
  → ∃ n, n = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2)) :=
sorry

end lcm_quadruples_count_l138_138062


namespace max_value_of_f_l138_138290

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := by
  use Real.sqrt 5
  sorry

end max_value_of_f_l138_138290


namespace min_a4_in_arithmetic_sequence_l138_138537

noncomputable def arithmetic_sequence_min_a4 (a1 d : ℝ) 
(S4 : ℝ := 4 * a1 + 6 * d)
(S5 : ℝ := 5 * a1 + 10 * d)
(a4 : ℝ := a1 + 3 * d) : Prop :=
  S4 ≤ 4 ∧ S5 ≥ 15 → a4 = 7

theorem min_a4_in_arithmetic_sequence (a1 d : ℝ) (h1 : 4 * a1 + 6 * d ≤ 4) 
(h2 : 5 * a1 + 10 * d ≥ 15) : 
arithmetic_sequence_min_a4 a1 d := 
by {
  sorry -- Proof is omitted
}

end min_a4_in_arithmetic_sequence_l138_138537


namespace find_a_l138_138214

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 3) (h3 : a * x - 2 * y = 4) : a = 10 :=
by {
  sorry
}

end find_a_l138_138214


namespace express_in_scientific_notation_l138_138280

theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 159600 = a * 10 ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.596 ∧ b = 5 :=
by
  sorry

end express_in_scientific_notation_l138_138280


namespace coloring_problem_l138_138699

theorem coloring_problem (a : ℕ → ℕ) (n t : ℕ) 
  (h1 : ∀ i j, i < j → a i < a j) 
  (h2 : ∀ x : ℤ, ∃ i, 0 < i ∧ i ≤ n ∧ ((x + a (i - 1)) % t) = 0) : 
  n ∣ t :=
by
  sorry

end coloring_problem_l138_138699


namespace find_other_root_l138_138367

variable {m : ℝ} -- m is a real number
variable (x : ℝ)

theorem find_other_root (h : x^2 + m * x - 5 = 0) (hx1 : x = -1) : x = 5 :=
sorry

end find_other_root_l138_138367


namespace dot_product_AB_BC_l138_138873

theorem dot_product_AB_BC 
  (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (cosB : ℝ)
  (h3 : cosB = 3 / 4) : 
  (a * c * (-cosB) = -3/2) :=
by 
  -- Given conditions
  sorry

end dot_product_AB_BC_l138_138873


namespace find_abs_diff_of_average_and_variance_l138_138267

noncomputable def absolute_difference (x y : ℝ) (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  |x - y|

theorem find_abs_diff_of_average_and_variance (x y : ℝ) (h1 : (x + y + 30 + 29 + 31) / 5 = 30)
  (h2 : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) :
  absolute_difference x y 30 30 29 31 = 4 :=
by
  sorry

end find_abs_diff_of_average_and_variance_l138_138267


namespace simplify_and_evaluate_expression_l138_138567

theorem simplify_and_evaluate_expression (x y : ℚ) (h_x : x = -2) (h_y : y = 1/2) :
  (x + 2 * y)^2 - (x + y) * (x - y) = -11/4 := by
  sorry

end simplify_and_evaluate_expression_l138_138567


namespace debt_payments_l138_138063

noncomputable def average_payment (total_amount : ℕ) (payments : ℕ) : ℕ := total_amount / payments

theorem debt_payments (x : ℕ) :
  8 * x + 44 * (x + 65) = 52 * 465 → x = 410 :=
by
  intros h
  sorry

end debt_payments_l138_138063


namespace combined_total_pets_l138_138575

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l138_138575


namespace inequality_with_equality_condition_l138_138020

variables {a b c d : ℝ}

theorem inequality_with_equality_condition (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 1) : 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) ∧ 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1 / 2 ↔ a = b ∧ b = c ∧ c = d) := 
sorry

end inequality_with_equality_condition_l138_138020


namespace largest_int_mod_6_less_than_100_l138_138824

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l138_138824


namespace f_le_g_for_a_eq_neg1_l138_138998

noncomputable def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * Real.exp x

noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * x - Real.log x + t

theorem f_le_g_for_a_eq_neg1 (t : ℝ) :
  let b := 3
  ∃ x ∈ Set.Ioi 0, f (-1) b x ≤ g t x ↔ t ≤ Real.exp 2 - 1 / 2 :=
by
  sorry

end f_le_g_for_a_eq_neg1_l138_138998


namespace solve_equation_l138_138570

theorem solve_equation : ∃ x : ℝ, (x^3 - ⌊x⌋ = 3) := 
sorry

end solve_equation_l138_138570


namespace perfect_square_trinomial_m_l138_138530

theorem perfect_square_trinomial_m (m : ℤ) : (∃ (a : ℤ), (x : ℝ) → x^2 + m * x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) :=
sorry

end perfect_square_trinomial_m_l138_138530


namespace cost_of_gravelling_path_l138_138949

theorem cost_of_gravelling_path (length width path_width : ℝ) (cost_per_sq_m : ℝ)
  (h1 : length = 110) (h2 : width = 65) (h3 : path_width = 2.5) (h4 : cost_per_sq_m = 0.50) :
  (length * width - (length - 2 * path_width) * (width - 2 * path_width)) * cost_per_sq_m = 425 := by
  sorry

end cost_of_gravelling_path_l138_138949


namespace original_total_thumbtacks_l138_138889

-- Conditions
def num_cans : ℕ := 3
def num_boards_tested : ℕ := 120
def thumbtacks_per_board : ℕ := 3
def thumbtacks_remaining_per_can : ℕ := 30

-- Question
theorem original_total_thumbtacks :
  (num_cans * num_boards_tested * thumbtacks_per_board) + (num_cans * thumbtacks_remaining_per_can) = 450 :=
sorry

end original_total_thumbtacks_l138_138889


namespace smallest_two_digit_product_12_l138_138442

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l138_138442


namespace distinct_real_roots_l138_138102

theorem distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, (k - 2) * x^2 + 2 * x - 1 = 0 → ∃ y : ℝ, (k - 2) * y^2 + 2 * y - 1 = 0 ∧ y ≠ x) ↔
  (k > 1 ∧ k ≠ 2) := 
by sorry

end distinct_real_roots_l138_138102


namespace city_mpg_l138_138455

-- Define the conditions
variables {T H C : ℝ}
axiom cond1 : H * T = 560
axiom cond2 : (H - 6) * T = 336

-- The formal proof goal
theorem city_mpg : C = 9 :=
by
  have h1 : H = 560 / T := by sorry
  have h2 : (560 / T - 6) * T = 336 := by sorry
  have h3 : C = H - 6 := by sorry
  have h4 :  C = 9 := by sorry
  exact h4

end city_mpg_l138_138455


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138814

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138814


namespace shortest_chord_line_through_P_longest_chord_line_through_P_l138_138995

theorem shortest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = 1/2 * x + 5/2 → a * x + b * y + c = 0)
  ∧ (a = 1) ∧ (b = -2) ∧ (c = 5) := sorry

theorem longest_chord_line_through_P (P : ℝ × ℝ) (circle : (ℝ × ℝ) → Prop) (hP : P = (-1, 2))
  (h_circle_eq : ∀ (x y : ℝ), circle (x, y) ↔ x ^ 2 + y ^ 2 = 8) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ (x y : ℝ), y = -2 * x → a * x + b * y + c = 0)
  ∧ (a = 2) ∧ (b = 1) ∧ (c = 0) := sorry

end shortest_chord_line_through_P_longest_chord_line_through_P_l138_138995


namespace smallest_two_digit_l138_138431

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l138_138431


namespace tan_x_eq_sqrt3_l138_138207

theorem tan_x_eq_sqrt3 (x : Real) (h : Real.sin (x + 20 * Real.pi / 180) = Real.cos (x + 10 * Real.pi / 180) + Real.cos (x - 10 * Real.pi / 180)) : Real.tan x = Real.sqrt 3 := 
by
  sorry

end tan_x_eq_sqrt3_l138_138207


namespace sequence_nth_term_l138_138382

/-- The nth term of the sequence {a_n} defined by a_1 = 1 and
    the recurrence relation a_{n+1} = 2a_n + 2 for all n ∈ ℕ*,
    is given by the formula a_n = 3 * 2 ^ (n - 1) - 2. -/
theorem sequence_nth_term (n : ℕ) (h : n > 0) : 
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n > 0, a (n + 1) = 2 * a n + 2) ∧ a n = 3 * 2 ^ (n - 1) - 2 :=
  sorry

end sequence_nth_term_l138_138382


namespace sum_of_divisors_143_l138_138940

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l138_138940


namespace each_trainer_hours_l138_138413

theorem each_trainer_hours (dolphins : ℕ) (hours_per_dolphin : ℕ) (trainers : ℕ) :
  dolphins = 4 →
  hours_per_dolphin = 3 →
  trainers = 2 →
  (dolphins * hours_per_dolphin) / trainers = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end each_trainer_hours_l138_138413


namespace logarithm_equation_l138_138054

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem logarithm_equation (a : ℝ) : 
  (1 / log_base 2 a + 1 / log_base 3 a + 1 / log_base 4 a = 1) → a = 24 :=
by
  sorry

end logarithm_equation_l138_138054


namespace shaded_area_l138_138381

-- Definition for the conditions provided in the problem
def side_length := 6
def area_square := side_length ^ 2
def area_square_unit := area_square * 4

-- The problem and proof statement
theorem shaded_area (sl : ℕ) (asq : ℕ) (nsq : ℕ):
    sl = 6 ∧
    asq = sl ^ 2 ∧
    nsq = asq * 4 →
    nsq - (4 * (sl^2 / 2)) = 72 :=
by
  sorry

end shaded_area_l138_138381


namespace find_smallest_in_arith_prog_l138_138653

theorem find_smallest_in_arith_prog (a d : ℝ) 
    (h1 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
    (h2 : (a - 2 * d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2 * d)^4 = 136) :
    a = -2 * Real.sqrt 2 ∨ a = 2 * Real.sqrt 2 :=
begin
  -- sorry placeholder for proof steps
  sorry
end

end find_smallest_in_arith_prog_l138_138653


namespace find_a_l138_138669

variable {x y a : ℝ}

theorem find_a (h1 : 2 * x - y + a ≥ 0) (h2 : 3 * x + y ≤ 3) (h3 : ∀ (x y : ℝ), 4 * x + 3 * y ≤ 8) : a = 2 := 
sorry

end find_a_l138_138669


namespace initial_fraction_of_larger_jar_l138_138194

theorem initial_fraction_of_larger_jar (S L W : ℝ) 
  (h1 : W = 1/6 * S) 
  (h2 : W = 1/3 * L) : 
  W / L = 1 / 3 := 
by 
  sorry

end initial_fraction_of_larger_jar_l138_138194


namespace at_least_one_true_l138_138272

-- Definitions (Conditions)
variables (p q : Prop)

-- Statement
theorem at_least_one_true (h : p ∨ q) : p ∨ q := by
  sorry

end at_least_one_true_l138_138272


namespace average_speed_l138_138727

theorem average_speed (D T : ℝ) (hD : D = 200) (hT : T = 6) : D / T = 33.33 := by
  -- Sorry is used to skip the proof, only the statement is provided as per instruction
  sorry

end average_speed_l138_138727


namespace recurring_decimal_sum_as_fraction_l138_138083

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l138_138083


namespace exists_n_prime_divides_exp_sum_l138_138123

theorem exists_n_prime_divides_exp_sum (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) :=
by
  sorry

end exists_n_prime_divides_exp_sum_l138_138123


namespace transformation_correct_l138_138929

noncomputable def original_function (x : ℝ) : ℝ := 2^x
noncomputable def transformed_function (x : ℝ) : ℝ := 2^x - 1
noncomputable def log_function (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = log_function (original_function x) :=
by
  intros x
  rw [transformed_function, log_function, original_function]
  sorry

end transformation_correct_l138_138929


namespace find_p_l138_138244

variables {m n p : ℚ}

theorem find_p (h1 : m = 3 * n + 5) (h2 : (m + 2) = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end find_p_l138_138244


namespace distance_A_C_15_l138_138895

noncomputable def distance_from_A_to_C : ℝ := 
  let AB := 6
  let AC := AB + (3 * AB) / 2
  AC

theorem distance_A_C_15 (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 24) (h5 : D - B = 3 * (B - A)) 
  (h6 : C = (B + D) / 2) :
  distance_from_A_to_C = 15 :=
by sorry

end distance_A_C_15_l138_138895


namespace distribution_of_balls_l138_138421

-- Definition for the problem conditions
inductive Ball : Type
| one : Ball
| two : Ball
| three : Ball
| four : Ball

inductive Box : Type
| box1 : Box
| box2 : Box
| box3 : Box

-- Function to count the number of ways to distribute the balls according to the conditions
noncomputable def num_ways_to_distribute_balls : Nat := 18

-- Theorem statement
theorem distribution_of_balls :
  num_ways_to_distribute_balls = 18 := by
  sorry

end distribution_of_balls_l138_138421


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138839

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l138_138839


namespace smallest_two_digit_number_product_12_l138_138436

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l138_138436


namespace james_chess_learning_time_l138_138246

theorem james_chess_learning_time (R : ℝ) 
    (h1 : R + 49 * R + 100 * (R + 49 * R) = 10100) 
    : R = 2 :=
by 
  sorry

end james_chess_learning_time_l138_138246


namespace FastFoodCost_l138_138786

theorem FastFoodCost :
  let sandwich_cost := 4
  let soda_cost := 1.5
  let fries_cost := 2.5
  let num_sandwiches := 4
  let num_sodas := 6
  let num_fries := 3
  let discount := 5
  let total_cost := (sandwich_cost * num_sandwiches) + (soda_cost * num_sodas) + (fries_cost * num_fries) - discount
  total_cost = 27.5 := 
by
  sorry

end FastFoodCost_l138_138786


namespace find_cashew_kilos_l138_138065

variables (x : ℕ)

def cashew_cost_per_kilo := 210
def peanut_cost_per_kilo := 130
def total_weight := 5
def peanuts_weight := 2
def avg_price_per_kilo := 178

-- Given conditions
def cashew_total_cost := cashew_cost_per_kilo * x
def peanut_total_cost := peanut_cost_per_kilo * peanuts_weight
def total_price := total_weight * avg_price_per_kilo

theorem find_cashew_kilos (h1 : cashew_total_cost + peanut_total_cost = total_price) : x = 3 :=
by
  sorry

end find_cashew_kilos_l138_138065


namespace gcd_4557_1953_5115_l138_138648

def problem_conditions : Prop := (4557 > 0) ∧ (1953 > 0) ∧ (5115 > 0)

theorem gcd_4557_1953_5115 : Int.gcd (Int.gcd 4557 1953) 5115 = 93 := by
  have h1 : problem_conditions := by
    -- Since 4557, 1953, and 5115 are all greater than 0, we have:
    sorry
  -- Use the Euclidean algorithm to find the GCD of the numbers 4557, 1953, and 5115.
  sorry

end gcd_4557_1953_5115_l138_138648


namespace find_c_in_terms_of_a_and_b_l138_138722

theorem find_c_in_terms_of_a_and_b (a b : ℝ) :
  (∃ α β : ℝ, (α + β = -a) ∧ (α * β = b)) →
  (∃ c d : ℝ, (∃ α β : ℝ, (α^3 + β^3 = -c) ∧ (α^3 * β^3 = d))) →
  c = a^3 - 3 * a * b :=
by
  intros h1 h2
  sorry

end find_c_in_terms_of_a_and_b_l138_138722


namespace min_y_squared_isosceles_trapezoid_l138_138012

theorem min_y_squared_isosceles_trapezoid:
  ∀ (EF GH y : ℝ) (circle_center : ℝ)
    (isosceles_trapezoid : Prop)
    (tangent_EH : Prop)
    (tangent_FG : Prop),
  isosceles_trapezoid ∧ EF = 72 ∧ GH = 45 ∧ EH = y ∧ FG = y ∧
  (∃ (circle : ℝ), circle_center = (EF / 2) ∧ tangent_EH ∧ tangent_FG)
  → y^2 = 486 :=
by sorry

end min_y_squared_isosceles_trapezoid_l138_138012


namespace meiosis_and_fertilization_outcome_l138_138193

-- Definitions corresponding to the conditions:
def increases_probability_of_genetic_mutations (x : Type) := 
  ∃ (p : x), false -- Placeholder for the actual mutation rate being low

def inherits_all_genetic_material (x : Type) :=
  ∀ (p : x), false -- Parents do not pass all genes to offspring

def receives_exactly_same_genetic_information (x : Type) :=
  ∀ (p : x), false -- Offspring do not receive exact genetic information from either parent

def produces_genetic_combination_different (x : Type) :=
  ∃ (o : x), true -- The offspring has different genetic information from either parent

-- The main statement to be proven:
theorem meiosis_and_fertilization_outcome (x : Type) 
  (cond1 : ¬ increases_probability_of_genetic_mutations x)
  (cond2 : ¬ inherits_all_genetic_material x)
  (cond3 : ¬ receives_exactly_same_genetic_information x) :
  produces_genetic_combination_different x :=
sorry

end meiosis_and_fertilization_outcome_l138_138193


namespace angle_MON_l138_138994

theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  (O = (0, 0)) →
  (M = (-2, 2)) →
  (N = (2, 2)) →
  (x^2 + y^2 + D * x - 4 * y = 0) →
  (D = 0) →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end angle_MON_l138_138994


namespace determine_x_l138_138192

theorem determine_x (x : ℝ) (hx : 0 < x) (h : (⌊x⌋ : ℝ) * x = 120) : x = 120 / 11 := 
sorry

end determine_x_l138_138192


namespace cabin_price_correct_l138_138516

noncomputable def cabin_price 
  (cash : ℤ)
  (cypress_trees : ℤ) (pine_trees : ℤ) (maple_trees : ℤ)
  (price_cypress : ℤ) (price_pine : ℤ) (price_maple : ℤ)
  (remaining_cash : ℤ)
  (expected_price : ℤ) : Prop :=
   cash + (cypress_trees * price_cypress + pine_trees * price_pine + maple_trees * price_maple) - remaining_cash = expected_price

theorem cabin_price_correct :
  cabin_price 150 20 600 24 100 200 300 350 130000 :=
by
  sorry

end cabin_price_correct_l138_138516


namespace find_triplets_l138_138646

theorem find_triplets (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1 ∣ (a + 1)^n) ↔ ((a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end find_triplets_l138_138646


namespace sum_of_repeating_decimals_l138_138645

-- Definitions for periodic decimals
def repeating_five := 5 / 9
def repeating_seven := 7 / 9

-- Theorem statement
theorem sum_of_repeating_decimals : (repeating_five + repeating_seven) = 4 / 3 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_repeating_decimals_l138_138645


namespace group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l138_138735

def mats_weaved (weavers mats days : ℕ) : ℕ :=
  (mats / days) * weavers

theorem group_a_mats_in_12_days (mats_req : ℕ) :
  let weavers := 4
  let mats_per_period := 4
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_b_mats_in_12_days (mats_req : ℕ) :
  let weavers := 6
  let mats_per_period := 9
  let period_days := 3
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_c_mats_in_12_days (mats_req : ℕ) :
  let weavers := 8
  let mats_per_period := 16
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

end group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l138_138735


namespace cubic_equation_three_distinct_real_roots_l138_138985

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x^2 - a

theorem cubic_equation_three_distinct_real_roots (a : ℝ) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃
  ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ↔ -4 < a ∧ a < 0 :=
sorry

end cubic_equation_three_distinct_real_roots_l138_138985


namespace DeMorgansLaws_l138_138504

variable (U : Type) (A B : Set U)

theorem DeMorgansLaws :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∧ (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ :=
by
  -- Statement of the theorems, proof is omitted
  sorry

end DeMorgansLaws_l138_138504


namespace find_a_l138_138805

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l138_138805


namespace union_sets_l138_138393

def A := { x : ℝ | x^2 ≤ 1 }
def B := { x : ℝ | 0 < x }

theorem union_sets : A ∪ B = { x | -1 ≤ x } :=
by {
  sorry -- Proof is omitted as per the instructions
}

end union_sets_l138_138393


namespace three_pow_2040_mod_5_l138_138044

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l138_138044


namespace tylers_age_l138_138599

theorem tylers_age (B T : ℕ) 
  (h1 : T = B - 3) 
  (h2 : T + B = 11) : 
  T = 4 :=
sorry

end tylers_age_l138_138599


namespace inequality_1_inequality_3_l138_138360

variable (a b : ℝ)
variable (hab : a > b ∧ b ≥ 2)

theorem inequality_1 (hab : a > b ∧ b ≥ 2) : b ^ 2 > 3 * b - a :=
by sorry

theorem inequality_3 (hab : a > b ∧ b ≥ 2) : a * b > a + b :=
by sorry

end inequality_1_inequality_3_l138_138360


namespace power_multiplication_l138_138792

variable (p : ℝ)  -- Assuming p is a real number

theorem power_multiplication :
  (-p)^2 * (-p)^3 = -p^5 :=
sorry

end power_multiplication_l138_138792


namespace remainder_of_power_modulo_l138_138050

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l138_138050


namespace find_a_l138_138802

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l138_138802


namespace find_a_l138_138032

theorem find_a (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 = 180)
  (h2 : x2 = 182)
  (h3 : x3 = 173)
  (h4 : x4 = 175)
  (h6 : x6 = 178)
  (h7 : x7 = 176)
  (h_avg : (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 178) : x5 = 182 := by
  sorry

end find_a_l138_138032


namespace snail_max_distance_300_meters_l138_138633
-- Import required library

-- Define the problem statement
theorem snail_max_distance_300_meters 
  (n : ℕ) (left_turns : ℕ) (right_turns : ℕ) 
  (total_distance : ℕ)
  (h1 : n = 300)
  (h2 : left_turns = 99)
  (h3 : right_turns = 200)
  (h4 : total_distance = n) : 
  ∃ d : ℝ, d = 100 * Real.sqrt 2 :=
by
  sorry

end snail_max_distance_300_meters_l138_138633


namespace jerry_age_is_10_l138_138891

-- Define the ages of Mickey and Jerry
def MickeyAge : ℝ := 20
def mickey_eq_jerry (JerryAge : ℝ) : Prop := MickeyAge = 2.5 * JerryAge - 5

theorem jerry_age_is_10 : ∃ JerryAge : ℝ, mickey_eq_jerry JerryAge ∧ JerryAge = 10 :=
by
  -- By solving the equation MickeyAge = 2.5 * JerryAge - 5,
  -- we can find that Jerry's age must be 10.
  use 10
  sorry

end jerry_age_is_10_l138_138891


namespace hours_per_trainer_l138_138414

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end hours_per_trainer_l138_138414


namespace small_order_peanuts_l138_138285

theorem small_order_peanuts (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) 
    (small_orders : ℕ) (peanuts_per_small : ℕ) : 
    total_peanuts = large_orders * peanuts_per_large + small_orders * peanuts_per_small → 
    total_peanuts = 800 → 
    large_orders = 3 → 
    peanuts_per_large = 200 → 
    small_orders = 4 → 
    peanuts_per_small = 50 := by
  intros h1 h2 h3 h4 h5
  sorry

end small_order_peanuts_l138_138285


namespace probability_all_six_draws_white_l138_138953

theorem probability_all_six_draws_white :
  let total_balls := 14
  let white_balls := 7
  let single_draw_white_probability := (white_balls : ℚ) / total_balls
  (single_draw_white_probability ^ 6 = (1 : ℚ) / 64) :=
by
  sorry

end probability_all_six_draws_white_l138_138953


namespace meaningful_sqrt_range_l138_138739

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l138_138739


namespace kim_candy_bars_saved_l138_138882

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end kim_candy_bars_saved_l138_138882


namespace find_value_of_a3_plus_a5_l138_138243

variable {a : ℕ → ℝ}
variable {r : ℝ}

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_value_of_a3_plus_a5 (h_geom : geometric_seq a r) (h_pos: ∀ n, 0 < a n)
  (h_eq: a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end find_value_of_a3_plus_a5_l138_138243


namespace terminating_decimals_count_l138_138655

theorem terminating_decimals_count : 
  (Finset.card (Finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 499)) (Finset.range 500))) = 499 :=
by
  sorry

end terminating_decimals_count_l138_138655


namespace workers_work_5_days_a_week_l138_138308

def total_weekly_toys : ℕ := 5500
def daily_toys : ℕ := 1100
def days_worked : ℕ := total_weekly_toys / daily_toys

theorem workers_work_5_days_a_week : days_worked = 5 := 
by 
  sorry

end workers_work_5_days_a_week_l138_138308


namespace amount_spent_per_trip_l138_138131

def trips_per_month := 4
def months_per_year := 12
def initial_amount := 200
def final_amount := 104

def total_amount_spent := initial_amount - final_amount
def total_trips := trips_per_month * months_per_year

theorem amount_spent_per_trip :
  (total_amount_spent / total_trips) = 2 := 
by 
  sorry

end amount_spent_per_trip_l138_138131


namespace new_plants_description_l138_138282

-- Condition: Anther culture of diploid corn treated with colchicine.
def diploid_corn := Type
def colchicine_treatment (plant : diploid_corn) : Prop := -- assume we have some method to define it
sorry

def anther_culture (plant : diploid_corn) (treated : colchicine_treatment plant) : Type := -- assume we have some method to define it
sorry

-- Describe the properties of new plants
def is_haploid (plant : diploid_corn) : Prop := sorry
def has_no_homologous_chromosomes (plant : diploid_corn) : Prop := sorry
def cannot_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def has_homologous_chromosomes_in_somatic_cells (plant : diploid_corn) : Prop := sorry
def can_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def is_homozygous_or_heterozygous (plant : diploid_corn) : Prop := sorry
def is_definitely_homozygous (plant : diploid_corn) : Prop := sorry
def is_diploid (plant : diploid_corn) : Prop := sorry

-- Equivalent math proof problem
theorem new_plants_description (plant : diploid_corn) (treated : colchicine_treatment plant) : 
  is_haploid (anther_culture plant treated) ∧ 
  has_homologous_chromosomes_in_somatic_cells (anther_culture plant treated) ∧ 
  can_form_fertile_gametes (anther_culture plant treated) ∧ 
  is_homozygous_or_heterozygous (anther_culture plant treated) := sorry

end new_plants_description_l138_138282


namespace exists_unique_x_l138_138136

-- Given constants a and b with the condition a > b > 0
variables {a b : ℝ} (h : a > b) (h' : b > 0)

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b)

-- Define the target value
def target_value : ℝ := (a ^ (1/3) + b ^ (1/3)) / 2

-- Prove the existence and uniqueness of x
theorem exists_unique_x : ∃! (x : ℝ), x > 0 ∧ f a b x = target_value ^ 3 :=
sorry

end exists_unique_x_l138_138136


namespace sqrt3_pow_log_sqrt3_8_eq_8_l138_138448

theorem sqrt3_pow_log_sqrt3_8_eq_8 : (Real.sqrt 3) ^ (Real.log 8 / Real.log (Real.sqrt 3)) = 8 :=
by
  sorry

end sqrt3_pow_log_sqrt3_8_eq_8_l138_138448


namespace inequality_max_k_l138_138649

theorem inequality_max_k (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2 * d)^5) ≥ 174960 * a * b * c * d^3 :=
sorry

end inequality_max_k_l138_138649


namespace problem1_problem2_l138_138972

-- Problem 1
theorem problem1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 :=
by decide  -- automatically prove simple arithmetic

-- Problem 2
variables {x : ℝ} (hx1 : x ≠ 1) (hx2 : x ≠ -1)

theorem problem2 : ((x^2 / (x + 1)) - (1 / (x + 1))) * (x + 1) / (x - 1) = x + 1 :=
by sorry  -- proof to be completed

end problem1_problem2_l138_138972


namespace action_movies_rented_l138_138977

-- Defining the conditions as hypotheses
theorem action_movies_rented (a M A D : ℝ) (h1 : 0.64 * M = 10 * a)
                             (h2 : D = 5 * A)
                             (h3 : D + A = 0.36 * M) :
    A = 0.9375 * a :=
sorry

end action_movies_rented_l138_138977


namespace square_side_length_l138_138728

theorem square_side_length (x : ℝ) (h : 4 * x = x^2) : x = 4 := 
by
  sorry

end square_side_length_l138_138728


namespace total_fence_poles_needed_l138_138604

def number_of_poles_per_side := 27

theorem total_fence_poles_needed (n : ℕ) (h : n = number_of_poles_per_side) : 
  4 * n - 4 = 104 :=
by sorry

end total_fence_poles_needed_l138_138604


namespace find_power_l138_138108

theorem find_power (some_power : ℕ) (k : ℕ) :
  k = 8 → (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → some_power = 16 :=
by
  intro h1 h2
  rw [h1] at h2
  sorry

end find_power_l138_138108


namespace percentage_x_of_yz_l138_138254

theorem percentage_x_of_yz (x y z w : ℝ) (h1 : x = 0.07 * y) (h2 : y = 0.35 * z) (h3 : z = 0.60 * w) :
  (x / (y + z) * 100) = 1.8148 :=
by
  sorry

end percentage_x_of_yz_l138_138254


namespace taxi_fare_distance_l138_138416

theorem taxi_fare_distance (x : ℝ) : 
  (8 + if x ≤ 3 then 0 else if x ≤ 8 then 2.15 * (x - 3) else 2.15 * 5 + 2.85 * (x - 8)) + 1 = 31.15 → x = 11.98 :=
by 
  sorry

end taxi_fare_distance_l138_138416


namespace max_area_of_rotating_lines_l138_138540

structure Point :=
(x : ℝ)
(y : ℝ)

def slope (p1 p2 : Point) : ℝ :=
(p2.y - p1.y) / (p2.x - p1.x)

def y_intercept (p : Point) (m : ℝ) : ℝ :=
p.y - m * p.x

def line (m : ℝ) (b : ℝ) : ℝ → ℝ :=
λ x, m * x + b

def vertical_line (x : ℝ) : ℝ → Point :=
λ y, ⟨x, y⟩

noncomputable def rotate (angle : ℝ) (p : Point) : Point :=
sorry -- Implementation of rotation around specific point is beyond this example.

noncomputable def intersection (ℓ₁ ℓ₂ : ℝ → ℝ) : Point :=
sorry -- Intersection calculation is beyond this example.

def area_of_triangle (p1 p2 p3 : Point) : ℝ :=
(1 / 2) * | p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) |

theorem max_area_of_rotating_lines : 
  let A := Point.mk 0 0,
      B := Point.mk 8 0,
      C := Point.mk 15 0,
      ℓA := line 2 0,
      ℓB := vertical_line 8,
      ℓC := line (-2) (y_intercept (Point.mk 15 0) (-2))
  in ∃ time : ℝ, ∃ p1 p2 p3 : Point, 
      p1 = intersection ℓB ℓC ∧
      p2 = intersection ℓA ℓC ∧
      p3 = intersection ℓA ℓB ∧
      area_of_triangle p1 p2 p3 = 0.5 := 
sorry

end max_area_of_rotating_lines_l138_138540


namespace circle_k_range_l138_138410

def circle_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem circle_k_range (k : ℝ) (h : ∃ x y, circle_equation k x y) : k > 4 ∨ k < -1 :=
by
  sorry

end circle_k_range_l138_138410


namespace expected_adjacent_red_pairs_l138_138910

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l138_138910


namespace simplify_expression_l138_138134

theorem simplify_expression (x : ℝ) : 120 * x - 72 * x + 15 * x - 9 * x = 54 * x := 
by
  sorry

end simplify_expression_l138_138134


namespace total_cost_is_correct_l138_138260

def goldfish_price := 3
def goldfish_quantity := 15
def blue_fish_price := 6
def blue_fish_quantity := 7
def neon_tetra_price := 2
def neon_tetra_quantity := 10
def angelfish_price := 8
def angelfish_quantity := 5

def total_cost := goldfish_quantity * goldfish_price 
                 + blue_fish_quantity * blue_fish_price 
                 + neon_tetra_quantity * neon_tetra_price 
                 + angelfish_quantity * angelfish_price

theorem total_cost_is_correct : total_cost = 147 :=
by
  -- Summary of the proof steps goes here
  sorry

end total_cost_is_correct_l138_138260


namespace integer_solution_for_equation_l138_138746

theorem integer_solution_for_equation :
  ∃ (M : ℤ), 14^2 * 35^2 = 10^2 * (M - 10)^2 ∧ M = 59 :=
by
  sorry

end integer_solution_for_equation_l138_138746


namespace axis_of_symmetry_l138_138678

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) : 
  ∀ x : ℝ, f x = f (4 - x) := 
  by sorry

end axis_of_symmetry_l138_138678


namespace recurring_decimal_sum_as_fraction_l138_138084

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l138_138084


namespace sum_of_coordinates_of_D_l138_138712

theorem sum_of_coordinates_of_D (P C D : ℝ × ℝ)
  (hP : P = (4, 9))
  (hC : C = (10, 5))
  (h_mid : P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 11 :=
sorry

end sum_of_coordinates_of_D_l138_138712


namespace reduced_cost_per_meter_l138_138470

theorem reduced_cost_per_meter (original_cost total_cost new_length original_length : ℝ) :
  original_cost = total_cost / original_length →
  new_length = original_length + 4 →
  total_cost = total_cost →
  original_cost - (total_cost / new_length) = 1 :=
by sorry

end reduced_cost_per_meter_l138_138470


namespace maximum_possible_shortest_piece_length_l138_138765

theorem maximum_possible_shortest_piece_length :
  ∃ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ 
  C = 140 ∧ (A + B + C + D + E = 640) ∧ A = 80 :=
by
  sorry

end maximum_possible_shortest_piece_length_l138_138765


namespace cost_of_six_hotdogs_and_seven_burgers_l138_138387

theorem cost_of_six_hotdogs_and_seven_burgers :
  ∀ (h b : ℝ), 4 * h + 5 * b = 3.75 → 5 * h + 3 * b = 3.45 → 6 * h + 7 * b = 5.43 :=
by
  intros h b h_eqn b_eqn
  sorry

end cost_of_six_hotdogs_and_seven_burgers_l138_138387


namespace mandy_pieces_eq_fifteen_l138_138770

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l138_138770


namespace product_of_five_consecutive_integers_divisible_by_120_l138_138936

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by 
  sorry

end product_of_five_consecutive_integers_divisible_by_120_l138_138936


namespace equation_one_solution_equation_two_no_solution_l138_138264

-- Problem 1
theorem equation_one_solution (x : ℝ) (h : x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) : x = 0 := 
by 
  sorry

-- Problem 2
theorem equation_two_no_solution (x : ℝ) (h : 2 * x + 9 / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2) : False := 
by 
  sorry

end equation_one_solution_equation_two_no_solution_l138_138264


namespace todd_savings_l138_138281

def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def credit_card_discount : ℝ := 0.10
def rebate : ℝ := 0.05
def sales_tax : ℝ := 0.08

def calculate_savings (original_price sale_discount coupon credit_card_discount rebate sales_tax : ℝ) : ℝ :=
  let after_sale := original_price * (1 - sale_discount)
  let after_coupon := after_sale - coupon
  let after_credit_card := after_coupon * (1 - credit_card_discount)
  let after_rebate := after_credit_card * (1 - rebate)
  let tax := after_credit_card * sales_tax
  let final_price := after_rebate + tax
  original_price - final_price

theorem todd_savings : calculate_savings 125 0.20 10 0.10 0.05 0.08 = 41.57 :=
by
  sorry

end todd_savings_l138_138281


namespace yogurt_combinations_l138_138069

-- Definitions based on conditions
def flavors : ℕ := 5
def toppings : ℕ := 8
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The problem statement to be proved
theorem yogurt_combinations :
  flavors * choose toppings 3 = 280 :=
by
  sorry

end yogurt_combinations_l138_138069


namespace max_product_l138_138346

theorem max_product (a b : ℝ) (h1 : 9 * a ^ 2 + 16 * b ^ 2 = 25) (h2 : a > 0) (h3 : b > 0) :
  a * b ≤ 25 / 24 :=
sorry

end max_product_l138_138346


namespace percentage_saving_l138_138618

theorem percentage_saving 
  (p_coat p_pants : ℝ)
  (d_coat d_pants : ℝ)
  (h_coat : p_coat = 100)
  (h_pants : p_pants = 50)
  (h_d_coat : d_coat = 0.30)
  (h_d_pants : d_pants = 0.40) :
  (p_coat * d_coat + p_pants * d_pants) / (p_coat + p_pants) = 0.333 :=
by
  sorry

end percentage_saving_l138_138618


namespace find_fourth_number_l138_138951

variable (x : ℝ)

theorem find_fourth_number
  (h : 3 + 33 + 333 + x = 399.6) :
  x = 30.6 :=
sorry

end find_fourth_number_l138_138951


namespace speed_in_still_water_l138_138623

-- Given conditions
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 41

-- Question: Prove the speed of the man in still water is 33 kmph.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 33 := 
by 
  sorry

end speed_in_still_water_l138_138623


namespace even_function_value_l138_138852

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_value (h_even : ∀ x, f a b x = f a b (-x))
    (h_domain : a - 1 = -2 * a) :
    f a (0 : ℝ) (1 / 2) = 13 / 12 :=
by
  sorry

end even_function_value_l138_138852


namespace find_n_l138_138087

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n % 11 = 99999 % 11) : n = 9 :=
sorry

end find_n_l138_138087


namespace unique_positive_integers_l138_138035

theorem unique_positive_integers (x y : ℕ) (h1 : x^2 + 84 * x + 2008 = y^2) : x + y = 80 :=
  sorry

end unique_positive_integers_l138_138035


namespace option1_distribution_and_expectation_best_participation_option_l138_138067

open ProbabilityTheory

-- Define the main problem conditions
def traditional_event_points : ℕ → ℕ
| 0 => 0
| _ => 30

def new_event_points : ℕ → ℕ
| 0 => 0
| 1 => 40
| _ => 90

def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
(n.choose k) * p^k * (1-p)^(n-k)

-- Lean 4 theorem statements
theorem option1_distribution_and_expectation :
  let X := 30 * (Nat.rchoose 3 ∏ i, ∑ (k : ℝ), if k < 1/2 then 0 else 1);
  E[X] = 45 :=
sorry

theorem best_participation_option :
  let X_score := 45
  let Y_score := (0 * 2/9) + (30 * 2/9) + (40 * 2/9) + (70 * 2/9) + (90 * 1/18) + (120 * 1/18);
  X_score > Y_score :=
sorry

end option1_distribution_and_expectation_best_participation_option_l138_138067


namespace grace_crayon_selection_l138_138277

def crayons := {i // 1 ≤ i ∧ i ≤ 15}
def red_crayons := {i // 1 ≤ i ∧ i ≤ 3}

def total_ways := Nat.choose 15 5
def non_favorable := Nat.choose 12 5

theorem grace_crayon_selection : total_ways - non_favorable = 2211 :=
by
  sorry

end grace_crayon_selection_l138_138277


namespace twins_age_l138_138296

theorem twins_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 :=
by
  sorry

end twins_age_l138_138296


namespace ratio_of_turtles_l138_138894

noncomputable def initial_turtles_owen : ℕ := 21
noncomputable def initial_turtles_johanna : ℕ := initial_turtles_owen - 5
noncomputable def turtles_johanna_after_month : ℕ := initial_turtles_johanna / 2
noncomputable def turtles_owen_after_month : ℕ := 50 - turtles_johanna_after_month

theorem ratio_of_turtles (a b : ℕ) (h1 : a = 21) (h2 : b = 5) (h3 : initial_turtles_owen = a) (h4 : initial_turtles_johanna = initial_turtles_owen - b) 
(h5 : turtles_johanna_after_month = initial_turtles_johanna / 2) (h6 : turtles_owen_after_month = 50 - turtles_johanna_after_month) : 
turtles_owen_after_month / initial_turtles_owen = 2 := by
  sorry

end ratio_of_turtles_l138_138894


namespace parking_lot_total_spaces_l138_138469

theorem parking_lot_total_spaces (ratio_fs_cc : ℕ) (ratio_cc_fs : ℕ) (fs_spaces : ℕ) (total_spaces : ℕ) 
  (h1 : ratio_fs_cc = 11) (h2 : ratio_cc_fs = 4) (h3 : fs_spaces = 330) :
  total_spaces = 450 :=
by
  sorry

end parking_lot_total_spaces_l138_138469


namespace lg_eight_plus_three_lg_five_l138_138079

theorem lg_eight_plus_three_lg_five : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  sorry

end lg_eight_plus_three_lg_five_l138_138079


namespace partnership_investment_l138_138058

theorem partnership_investment
  (a_investment : ℕ := 30000)
  (b_investment : ℕ)
  (c_investment : ℕ := 50000)
  (c_profit_share : ℕ := 36000)
  (total_profit : ℕ := 90000)
  (total_investment := a_investment + b_investment + c_investment)
  (c_defined_share : ℚ := 2/5)
  (profit_proportionality : (c_profit_share : ℚ) / total_profit = (c_investment : ℚ) / total_investment) :
  b_investment = 45000 :=
by
  sorry

end partnership_investment_l138_138058


namespace max_value_x_plus_2y_l138_138363

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x * y = 4) :
  x + 2 * y ≤ 4 :=
sorry

end max_value_x_plus_2y_l138_138363


namespace class_books_transfer_l138_138488

theorem class_books_transfer :
  ∀ (A B n : ℕ), 
    A = 200 → B = 200 → 
    (B + n = 3/2 * (A - n)) →
    n = 40 :=
by sorry

end class_books_transfer_l138_138488


namespace problem_l138_138248

section Problem
variables {n : ℕ } {k : ℕ} 

theorem problem (n : ℕ) (k : ℕ) (a : ℕ) (n_i : Fin k → ℕ) (h1 : ∀ i j, i ≠ j → Nat.gcd (n_i i) (n_i j) = 1) 
  (h2 : ∀ i, a^n_i i % n_i i = 1) (h3 : ∀ i, ¬(n_i i ∣ a - 1)) :
  ∃ (x : ℕ), x > 1 ∧ a^x % x = 1 ∧ x ≥ 2^(k + 1) - 2 := by
  sorry
end Problem

end problem_l138_138248


namespace cave_depth_l138_138682

theorem cave_depth 
  (total_depth : ℕ) 
  (remaining_depth : ℕ) 
  (h1 : total_depth = 974) 
  (h2 : remaining_depth = 386) : 
  total_depth - remaining_depth = 588 := 
by 
  sorry

end cave_depth_l138_138682


namespace closest_to_fraction_l138_138494

theorem closest_to_fraction (options : List ℝ) (h1 : options = [2000, 1500, 200, 2500, 3000]) :
  ∃ closest : ℝ, closest ∈ options ∧ closest = 2000 :=
by
  sorry

end closest_to_fraction_l138_138494


namespace total_loss_l138_138969

theorem total_loss (P : ℝ) (A : ℝ) (L : ℝ) (h1 : A = (1/9) * P) (h2 : 603 = (P / (A + P)) * L) : 
  L = 670 :=
by
  sorry

end total_loss_l138_138969


namespace non_congruent_squares_on_6x6_grid_l138_138227

theorem non_congruent_squares_on_6x6_grid : 
  let grid := (6,6)
  ∃ (n : ℕ), n = 89 ∧ 
  (∀ k, (1 ≤ k ∧ k ≤ 6) → (lattice_squares_count grid k = k * k),
  tilted_squares_count grid 2 = 25,
  tilted_squares_count grid 4 = 9)
  :=
sorry

end non_congruent_squares_on_6x6_grid_l138_138227


namespace prove_f_cos_eq_l138_138529

variable (f : ℝ → ℝ)

theorem prove_f_cos_eq :
  (∀ x : ℝ, f (Real.sin x) = 3 - Real.cos (2 * x)) →
  (∀ x : ℝ, f (Real.cos x) = 3 + Real.cos (2 * x)) :=
by
  sorry

end prove_f_cos_eq_l138_138529


namespace tree_shadow_length_l138_138692

theorem tree_shadow_length (jane_shadow : ℝ) (jane_height : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h₁ : jane_shadow = 0.5)
  (h₂ : jane_height = 1.5)
  (h₃ : tree_height = 30)
  (h₄ : jane_height / jane_shadow = tree_height / tree_shadow)
  : tree_shadow = 10 :=
by
  -- skipping the proof steps
  sorry

end tree_shadow_length_l138_138692


namespace sum_a1_to_a5_l138_138359

noncomputable def f (x : ℝ) : ℝ := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5
noncomputable def g (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : ℝ := a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5

theorem sum_a1_to_a5 (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, f x = g x a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 1 = g 1 a_0 a_1 a_2 a_3 a_4 a_5) →
  (f 0 = g 0 a_0 a_1 a_2 a_3 a_4 a_5) →
  a_0 = 62 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -57 :=
by
  intro hf1 hf2 hf3 ha0 hsum
  sorry

end sum_a1_to_a5_l138_138359


namespace dan_balloons_l138_138986

theorem dan_balloons (fred_balloons sam_balloons total_balloons dan_balloons : ℕ) 
  (h₁ : fred_balloons = 10) 
  (h₂ : sam_balloons = 46) 
  (h₃ : total_balloons = 72) : 
  dan_balloons = total_balloons - (fred_balloons + sam_balloons) :=
by
  sorry

end dan_balloons_l138_138986


namespace average_rainfall_feb_1983_l138_138866

theorem average_rainfall_feb_1983 (total_rainfall : ℕ) (days_in_february : ℕ) (hours_per_day : ℕ) 
  (H1 : total_rainfall = 789) (H2 : days_in_february = 28) (H3 : hours_per_day = 24) : 
  total_rainfall / (days_in_february * hours_per_day) = 789 / 672 :=
by
  sorry

end average_rainfall_feb_1983_l138_138866


namespace new_profit_is_220_percent_l138_138959

noncomputable def cost_price (CP : ℝ) : ℝ := 100

def initial_profit_percentage : ℝ := 60

noncomputable def initial_selling_price (CP : ℝ) : ℝ :=
  CP + (initial_profit_percentage / 100) * CP

noncomputable def new_selling_price (SP : ℝ) : ℝ :=
  2 * SP

noncomputable def new_profit_percentage (CP SP2 : ℝ) : ℝ :=
  ((SP2 - CP) / CP) * 100

theorem new_profit_is_220_percent : 
  new_profit_percentage (cost_price 100) (new_selling_price (initial_selling_price (cost_price 100))) = 220 :=
by
  sorry

end new_profit_is_220_percent_l138_138959


namespace problem1_problem2_l138_138484

-- Problem 1: Prove that 2023 * 2023 - 2024 * 2022 = 1
theorem problem1 : 2023 * 2023 - 2024 * 2022 = 1 := 
by 
  sorry

-- Problem 2: Prove that (-4 * x * y^3) * (1/2 * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4
theorem problem2 (x y : ℝ) : (-4 * x * y^3) * ((1/2) * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4 := 
by 
  sorry

end problem1_problem2_l138_138484


namespace no_solution_iff_n_eq_neg2_l138_138862

noncomputable def has_no_solution (n : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬ (n * x + y + z = 2 ∧ 
                  x + n * y + z = 2 ∧ 
                  x + y + n * z = 2)

theorem no_solution_iff_n_eq_neg2 (n : ℝ) : has_no_solution n ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg2_l138_138862


namespace amy_tips_calculation_l138_138968

theorem amy_tips_calculation 
  (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) 
  (h_wage : hourly_wage = 2)
  (h_hours : hours_worked = 7)
  (h_total : total_earnings = 23) : 
  total_earnings - (hourly_wage * hours_worked) = 9 := 
sorry

end amy_tips_calculation_l138_138968


namespace necessary_but_not_sufficient_l138_138989

def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient (a b c : ℝ) (p : a * b < 0) (q : is_hyperbola a b c) :
  (∀ (a b c : ℝ), is_hyperbola a b c → a * b < 0) ∧ (¬ ∀ (a b c : ℝ), a * b < 0 → is_hyperbola a b c) :=
by
  sorry

end necessary_but_not_sufficient_l138_138989


namespace ratio_of_length_to_width_l138_138583

-- Definitions of conditions
def width := 5
def area := 75

-- Theorem statement proving the ratio is 3
theorem ratio_of_length_to_width {l : ℕ} (h1 : l * width = area) : l / width = 3 :=
by sorry

end ratio_of_length_to_width_l138_138583


namespace find_angle_l138_138100

theorem find_angle (θ : ℝ) (h : 180 - θ = 3 * (90 - θ)) : θ = 45 :=
by
  sorry

end find_angle_l138_138100


namespace rain_on_first_day_l138_138927

theorem rain_on_first_day (x : ℝ) (h1 : x >= 0)
  (h2 : (2 * x) + 50 / 100 * (2 * x) = 3 * x) 
  (h3 : 6 * 12 = 72)
  (h4 : 3 * 3 = 9)
  (h5 : x + 2 * x + 3 * x = 6 * x)
  (h6 : 6 * x + 21 - 9 = 72) : x = 10 :=
by 
  -- Proof would go here, but we skip it according to instructions
  sorry

end rain_on_first_day_l138_138927


namespace find_b_l138_138587

theorem find_b (b c x1 x2 : ℝ)
  (h_parabola_intersects_x_axis : (x1 ≠ x2) ∧ x1 * x2 = c ∧ x1 + x2 = -b ∧ x2 - x1 = 1)
  (h_parabola_intersects_y_axis : c ≠ 0)
  (h_length_ab : x2 - x1 = 1)
  (h_area_abc : (1 / 2) * (x2 - x1) * |c| = 1)
  : b = -3 :=
sorry

end find_b_l138_138587


namespace eggs_collection_l138_138331

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l138_138331


namespace asymptotic_lines_of_hyperbola_l138_138813

open Real

-- Given: Hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- To Prove: Asymptotic lines equation
theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, hyperbola x y → (y = x ∨ y = -x) :=
by
  intros x y h
  sorry

end asymptotic_lines_of_hyperbola_l138_138813


namespace jasmine_spent_l138_138385

theorem jasmine_spent 
  (original_cost : ℝ)
  (discount : ℝ)
  (h_original : original_cost = 35)
  (h_discount : discount = 17) : 
  original_cost - discount = 18 := 
by
  sorry

end jasmine_spent_l138_138385


namespace lower_limit_of_range_with_multiples_l138_138733

theorem lower_limit_of_range_with_multiples (n : ℕ) (h : 2000 - n ≥ 198 * 10 ∧ n % 10 = 0 ∧ n + 1980 ≤ 2000) :
  n = 30 :=
by
  sorry

end lower_limit_of_range_with_multiples_l138_138733


namespace simplify_expression_eq_l138_138356

theorem simplify_expression_eq (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by 
  sorry

end simplify_expression_eq_l138_138356


namespace booth_earnings_after_5_days_l138_138954

def booth_daily_popcorn_earnings := 50
def booth_daily_cotton_candy_earnings := 3 * booth_daily_popcorn_earnings
def booth_total_daily_earnings := booth_daily_popcorn_earnings + booth_daily_cotton_candy_earnings
def booth_total_expenses := 30 + 75

theorem booth_earnings_after_5_days :
  5 * booth_total_daily_earnings - booth_total_expenses = 895 :=
by
  sorry

end booth_earnings_after_5_days_l138_138954


namespace problem_statement_l138_138003

noncomputable def percent_of_y (y : ℝ) (z : ℂ) : ℝ :=
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10).re

theorem problem_statement (y : ℝ) (z : ℂ) (hy : y > 0) : percent_of_y y z = 0.6 * y :=
by
  sorry

end problem_statement_l138_138003


namespace total_distance_collinear_centers_l138_138487

theorem total_distance_collinear_centers (r1 r2 r3 : ℝ) (d12 d13 d23 : ℝ) 
  (h1 : r1 = 6) 
  (h2 : r2 = 14) 
  (h3 : d12 = r1 + r2) 
  (h4 : d13 = r3 - r1) 
  (h5 : d23 = r3 - r2) :
  d13 = d12 + r1 := by
  -- proof follows here
  sorry

end total_distance_collinear_centers_l138_138487


namespace ratio_of_pq_l138_138555

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem ratio_of_pq (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (H : is_pure_imaginary ((Complex.ofReal 3 - Complex.ofReal 4 * Complex.I) * (Complex.ofReal p + Complex.ofReal q * Complex.I))) :
  p / q = -4 / 3 :=
by
  sorry

end ratio_of_pq_l138_138555


namespace joyce_pencils_given_l138_138644

def original_pencils : ℕ := 51
def total_pencils_after : ℕ := 57

theorem joyce_pencils_given : total_pencils_after - original_pencils = 6 :=
by
  sorry

end joyce_pencils_given_l138_138644


namespace triangle_is_isosceles_l138_138688

theorem triangle_is_isosceles 
  (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_sin_identity : Real.sin A = 2 * Real.sin C * Real.cos B) : 
  (B = C) :=
sorry

end triangle_is_isosceles_l138_138688


namespace mary_total_baseball_cards_l138_138016

noncomputable def mary_initial_baseball_cards : ℕ := 18
noncomputable def torn_baseball_cards : ℕ := 8
noncomputable def fred_given_baseball_cards : ℕ := 26
noncomputable def mary_bought_baseball_cards : ℕ := 40

theorem mary_total_baseball_cards :
  mary_initial_baseball_cards - torn_baseball_cards + fred_given_baseball_cards + mary_bought_baseball_cards = 76 :=
by
  sorry

end mary_total_baseball_cards_l138_138016


namespace cost_price_per_meter_l138_138693

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (h1 : total_cost = 397.75) (h2 : total_length = 9.25) : total_cost / total_length = 43 :=
by
  -- Proof omitted
  sorry

end cost_price_per_meter_l138_138693


namespace notebook_cost_l138_138877

-- Define the conditions
def cost_pen := 1
def num_pens := 3
def num_notebooks := 4
def cost_folder := 5
def num_folders := 2
def initial_bill := 50
def change_back := 25

-- Calculate derived values
def total_spent := initial_bill - change_back
def total_cost_pens := num_pens * cost_pen
def total_cost_folders := num_folders * cost_folder
def total_cost_notebooks := total_spent - total_cost_pens - total_cost_folders

-- Calculate the cost per notebook
def cost_per_notebook := total_cost_notebooks / num_notebooks

-- Proof statement
theorem notebook_cost : cost_per_notebook = 3 := by
  sorry

end notebook_cost_l138_138877


namespace sum_arithmetic_series_remainder_l138_138052

theorem sum_arithmetic_series_remainder :
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S % 9 = 5 :=
by
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  show S % 9 = 5
  sorry

end sum_arithmetic_series_remainder_l138_138052


namespace inequality_for_a_ne_1_l138_138898

theorem inequality_for_a_ne_1 (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3 * (1 + a^2 + a^4) :=
sorry

end inequality_for_a_ne_1_l138_138898


namespace least_non_lucky_multiple_of_11_l138_138312

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l138_138312


namespace meaningful_expression_l138_138373

theorem meaningful_expression (x : ℝ) : (1 / (x - 2) ≠ 0) ↔ (x ≠ 2) :=
by
  sorry

end meaningful_expression_l138_138373


namespace price_first_oil_l138_138610

theorem price_first_oil (P : ℝ) (h1 : 10 * P + 5 * 66 = 15 * 58.67) : P = 55.005 :=
sorry

end price_first_oil_l138_138610


namespace fraction_to_decimal_l138_138980

theorem fraction_to_decimal (n d : ℕ) (hn : n = 53) (hd : d = 160) (gcd_nd : Nat.gcd n d = 1)
  (prime_factorization_d : ∃ k l : ℕ, d = 2^k * 5^l) : ∃ dec : ℚ, (n:ℚ) / (d:ℚ) = dec ∧ dec = 0.33125 :=
by sorry

end fraction_to_decimal_l138_138980


namespace long_sleeve_shirts_l138_138399

variable (short_sleeve long_sleeve : Nat)
variable (total_shirts washed_shirts : Nat)
variable (not_washed_shirts : Nat)

-- Given conditions
axiom h1 : short_sleeve = 9
axiom h2 : total_shirts = 29
axiom h3 : not_washed_shirts = 1
axiom h4 : washed_shirts = total_shirts - not_washed_shirts

-- The question to be proved
theorem long_sleeve_shirts : long_sleeve = washed_shirts - short_sleeve := by
  sorry

end long_sleeve_shirts_l138_138399


namespace probability_spade_then_king_l138_138744

theorem probability_spade_then_king :
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  prob_case1 + prob_case2 = (17 / 884) :=
by {
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  have h1 : prob_case1 = 48 / 2652 := sorry,
  have h2 : prob_case2 = 3 / 2652 := sorry,
  calc
    prob_case1 + prob_case2 = (48 / 2652) + (3 / 2652) : by rw [h1, h2]
    ... = 51 / 2652 : by norm_num
    ... = 17 / 884 : by norm_num
}

end probability_spade_then_king_l138_138744


namespace smallest_sum_of_bases_l138_138425

theorem smallest_sum_of_bases :
  ∃ (c d : ℕ), 8 * c + 9 = 9 * d + 8 ∧ c + d = 19 := 
by
  sorry

end smallest_sum_of_bases_l138_138425


namespace smallest_interesting_rectangle_area_l138_138066

/-- 
  A rectangle is interesting if both its side lengths are integers and 
  it contains exactly four lattice points strictly in its interior.
  Prove that the area of the smallest such interesting rectangle is 10.
-/
theorem smallest_interesting_rectangle_area :
  ∃ (a b : ℕ), (a - 1) * (b - 1) = 4 ∧ a * b = 10 :=
by
  sorry

end smallest_interesting_rectangle_area_l138_138066


namespace fraction_product_l138_138186

theorem fraction_product :
  (7 / 4 : ℚ) * (14 / 35) * (21 / 12) * (28 / 56) * (49 / 28) * (42 / 84) * (63 / 36) * (56 / 112) = (1201 / 12800) := 
by
  sorry

end fraction_product_l138_138186


namespace combined_total_pets_l138_138574

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l138_138574


namespace population_Lake_Bright_l138_138597

-- Definition of total population
def T := 80000

-- Definition of population of Gordonia
def G := (1 / 2) * T

-- Definition of population of Toadon
def Td := (60 / 100) * G

-- Proof that the population of Lake Bright is 16000
theorem population_Lake_Bright : T - (G + Td) = 16000 :=
by {
    -- Leaving the proof as sorry
    sorry
}

end population_Lake_Bright_l138_138597


namespace sum_of_reciprocals_of_roots_eq_17_div_8_l138_138983

theorem sum_of_reciprocals_of_roots_eq_17_div_8 :
  ∀ p q : ℝ, (p + q = 17) → (p * q = 8) → (1 / p + 1 / q = 17 / 8) :=
by
  intros p q h1 h2
  sorry

end sum_of_reciprocals_of_roots_eq_17_div_8_l138_138983


namespace exists_plane_perpendicular_l138_138681

-- Definitions of line, plane and perpendicularity intersection etc.
variables (Point : Type) (Line Plane : Type)
variables (l : Line) (α : Plane) (intersects : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)
variables (β : Plane) (subset : Line → Plane → Prop)

-- Conditions
axiom line_intersects_plane (h1 : intersects l α) : Prop
axiom line_not_perpendicular_plane (h2 : ¬perpendicular l α) : Prop

-- The main statement to prove
theorem exists_plane_perpendicular (h1 : intersects l α) (h2 : ¬perpendicular l α) :
  ∃ (β : Plane), (subset l β) ∧ (perpendicular_planes β α) :=
sorry

end exists_plane_perpendicular_l138_138681


namespace probability_two_digit_between_21_and_30_l138_138707

theorem probability_two_digit_between_21_and_30 (dice1 dice2 : ℤ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 6) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 6) :
∃ (p : ℚ), p = 11 / 36 := 
sorry

end probability_two_digit_between_21_and_30_l138_138707


namespace first_spade_second_king_prob_l138_138743

-- Definitions and conditions of the problem
def total_cards := 52
def total_spades := 13
def total_kings := 4
def spades_excluding_king := 12 -- Number of spades excluding the king of spades
def remaining_kings_after_king_spade := 3

-- Calculate probabilities for each case
def first_non_king_spade_prob := spades_excluding_king / total_cards
def second_king_after_non_king_spade_prob := total_kings / (total_cards - 1)
def case1_prob := first_non_king_spade_prob * second_king_after_non_king_spade_prob

def first_king_spade_prob := 1 / total_cards
def second_king_after_king_spade_prob := remaining_kings_after_king_spade / (total_cards - 1)
def case2_prob := first_king_spade_prob * second_king_after_king_spade_prob

def combined_prob := case1_prob + case2_prob

-- The proof statement
theorem first_spade_second_king_prob :
  combined_prob = 1 / total_cards := by
  sorry

end first_spade_second_king_prob_l138_138743


namespace find_a_if_f_is_odd_l138_138671

noncomputable def f (a x : ℝ) : ℝ := (Real.logb 2 ((a - x) / (1 + x))) 

theorem find_a_if_f_is_odd (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

end find_a_if_f_is_odd_l138_138671


namespace smallest_two_digit_product_12_l138_138441

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l138_138441


namespace sum_of_divisors_143_l138_138941

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l138_138941


namespace canoe_total_weight_calculation_canoe_maximum_weight_limit_l138_138562

def canoe_max_people : ℕ := 8
def people_with_pets_ratio : ℚ := 3 / 4
def adult_weight : ℚ := 150
def child_weight : ℚ := adult_weight / 2
def dog_weight : ℚ := adult_weight / 3
def cat1_weight : ℚ := adult_weight / 10
def cat2_weight : ℚ := adult_weight / 8

def canoe_capacity_with_pets : ℚ := people_with_pets_ratio * canoe_max_people

def total_weight_adults_and_children : ℚ := 4 * adult_weight + 2 * child_weight
def total_weight_pets : ℚ := dog_weight + cat1_weight + cat2_weight
def total_weight : ℚ := total_weight_adults_and_children + total_weight_pets

def max_weight_limit : ℚ := canoe_max_people * adult_weight

theorem canoe_total_weight_calculation :
  total_weight = 833 + 3 / 4 := by
  sorry

theorem canoe_maximum_weight_limit :
  max_weight_limit = 1200 := by
  sorry

end canoe_total_weight_calculation_canoe_maximum_weight_limit_l138_138562


namespace sales_worth_l138_138630

def old_scheme_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_scheme_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)
def remuneration_difference (S : ℝ) : ℝ := new_scheme_remuneration S - old_scheme_remuneration S

theorem sales_worth (S : ℝ) (h : remuneration_difference S = 600) : S = 24000 :=
by
  sorry

end sales_worth_l138_138630


namespace abs_neg_four_minus_six_l138_138075

theorem abs_neg_four_minus_six : abs (-4 - 6) = 10 := 
by
  sorry

end abs_neg_four_minus_six_l138_138075


namespace area_of_triangle_ABC_is_24_l138_138355

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the area calculation
def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  0.5 * |(v.1 * w.2 - v.2 * w.1)|

theorem area_of_triangle_ABC_is_24 :
  triangleArea A B C = 24 := by
  sorry

end area_of_triangle_ABC_is_24_l138_138355


namespace kristen_turtles_l138_138283

variable (K : ℕ)
variable (T : ℕ)
variable (R : ℕ)

-- Conditions
def kris_turtles (K : ℕ) : ℕ := K / 4
def trey_turtles (R : ℕ) : ℕ := 7 * R
def trey_more_than_kristen (T K : ℕ) : Prop := T = K + 9

-- Theorem to prove 
theorem kristen_turtles (K : ℕ) (R : ℕ) (T : ℕ) (h1 : R = kris_turtles K) (h2 : T = trey_turtles R) (h3 : trey_more_than_kristen T K) : K = 12 :=
by
  sorry

end kristen_turtles_l138_138283


namespace cartons_in_load_l138_138179

theorem cartons_in_load 
  (crate_weight : ℕ)
  (carton_weight : ℕ)
  (num_crates : ℕ)
  (total_load_weight : ℕ)
  (h1 : crate_weight = 4)
  (h2 : carton_weight = 3)
  (h3 : num_crates = 12)
  (h4 : total_load_weight = 96) :
  ∃ C : ℕ, num_crates * crate_weight + C * carton_weight = total_load_weight ∧ C = 16 := 
by 
  sorry

end cartons_in_load_l138_138179


namespace simplify_composite_product_fraction_l138_138643

def first_four_composite_product : ℤ := 4 * 6 * 8 * 9
def next_four_composite_product : ℤ := 10 * 12 * 14 * 15
def expected_fraction_num : ℤ := 12
def expected_fraction_den : ℤ := 175

theorem simplify_composite_product_fraction :
  (first_four_composite_product / next_four_composite_product : ℚ) = (expected_fraction_num / expected_fraction_den) :=
by
  rw [first_four_composite_product, next_four_composite_product]
  norm_num
  sorry

end simplify_composite_product_fraction_l138_138643


namespace largest_int_less_than_100_mod_6_eq_4_l138_138819

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l138_138819


namespace point_on_x_axis_l138_138236

theorem point_on_x_axis (m : ℝ) (h : 3 * m + 1 = 0) : m = -1 / 3 :=
by 
  sorry

end point_on_x_axis_l138_138236


namespace prob_AC_adjacent_BE_not_adjacent_l138_138419

open Finset
open Perm
open Probability

-- Define the students as a Finset
def students : Finset (Fin 5) := {0, 1, 2, 3, 4}

-- Define that A is 0, B is 1, C is 2, D is 3, E is 4
def A := 0
def B := 1
def C := 2
def D := 3
def E := 4

-- Define the event of A and C being adjacent
def adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  (List.indexOf x p + 1 = List.indexOf y p) ∨ (List.indexOf y p + 1 = List.indexOf x p)

-- Define the event of B and E not being adjacent
def not_adjacent (x y : Fin 5) (p : List (Fin 5)) : Prop := 
  ¬ adjacent x y p

-- Lean 4 statement: Calculate the probability that A and C are adjacent while B and E are not adjacent
noncomputable def probabilityAC_adjacent_BE_not_adjacent : ℚ :=
  let total_permutations := (univ.perm 5).toFinset.card
  let valid_permutations := (univ.filter (λ p, adjacent A C p ∧ not_adjacent B E p)).perm.toFinset.card
  valid_permutations / total_permutations

theorem prob_AC_adjacent_BE_not_adjacent : probabilityAC_adjacent_BE_not_adjacent = 1/5 :=
  sorry

end prob_AC_adjacent_BE_not_adjacent_l138_138419


namespace amelia_drove_tuesday_l138_138025

-- Define the known quantities
def total_distance : ℕ := 8205
def distance_monday : ℕ := 907
def remaining_distance : ℕ := 6716

-- Define the distance driven on Tuesday and state the theorem
def distance_tuesday : ℕ := total_distance - (distance_monday + remaining_distance)

-- Theorem stating the distance driven on Tuesday is 582 kilometers
theorem amelia_drove_tuesday : distance_tuesday = 582 := 
by
  -- We skip the proof for now
  sorry

end amelia_drove_tuesday_l138_138025


namespace people_got_on_at_third_stop_l138_138736

theorem people_got_on_at_third_stop
  (initial : ℕ)
  (got_off_first : ℕ)
  (got_off_second : ℕ)
  (got_on_second : ℕ)
  (got_off_third : ℕ)
  (people_after_third : ℕ) :
  initial = 50 →
  got_off_first = 15 →
  got_off_second = 8 →
  got_on_second = 2 →
  got_off_third = 4 →
  people_after_third = 28 →
  ∃ got_on_third : ℕ, got_on_third = 3 :=
by
  sorry

end people_got_on_at_third_stop_l138_138736


namespace area_RWP_l138_138721

-- Definitions
variables (X Y Z W P Q R : ℝ × ℝ)
variables (h₁ : (X.1 - Z.1) * (X.1 - Z.1) + (X.2 - Z.2) * (X.2 - Z.2) = 144)
variables (h₂ : P.1 = X.1 - 8 ∧ P.2 = X.2)
variables (h₃ : Q.1 = (Z.1 + P.1) / 2 ∧ Q.2 = (Z.2 + P.2) / 2)
variables (h₄ : R.1 = (Y.1 + P.1) / 2 ∧ R.2 = (Y.2 + P.2) / 2)
variables (h₅ : 1 / 2 * ((Z.1 - X.1) * (W.2 - X.2) - (Z.2 - X.2) * (W.1 - X.1)) = 72)
variables (h₆ : 1 / 2 * abs ((Q.1 - X.1) * (W.2 - X.2) - (Q.2 - X.2) * (W.1 - X.1)) = 20)

-- Theorem statement
theorem area_RWP : 
  1 / 2 * abs ((R.1 - W.1) * (P.2 - W.2) - (R.2 - W.2) * (P.1 - W.1)) = 12 :=
sorry

end area_RWP_l138_138721


namespace range_of_m_l138_138606

noncomputable def f (x m : ℝ) := Real.exp x * (Real.log x + (1 / 2) * x ^ 2 - m * x)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → ((Real.exp x * ((1 / x) + x - m)) > 0)) → m < 2 := by
  sorry

end range_of_m_l138_138606


namespace range_distance_PQ_l138_138362

noncomputable def point_P (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def point_Q (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

noncomputable def distance_PQ (α β : ℝ) : ℝ :=
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 +
             (3 * Real.sin α - 2 * Real.sin β)^2 +
             (1 - 1)^2)

theorem range_distance_PQ : 
  ∀ α β : ℝ, 1 ≤ distance_PQ α β ∧ distance_PQ α β ≤ 5 := 
by
  intros
  sorry

end range_distance_PQ_l138_138362


namespace find_a_l138_138810

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l138_138810


namespace largest_integer_with_remainder_l138_138827

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l138_138827


namespace find_f_g_2_l138_138760

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 - 6

theorem find_f_g_2 : f (g 2) = 1 := 
  by
  -- Proof goes here
  sorry

end find_f_g_2_l138_138760


namespace largest_int_less_than_100_mod_6_eq_4_l138_138821

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l138_138821


namespace remainder_of_13_pow_13_plus_13_div_14_l138_138750

theorem remainder_of_13_pow_13_plus_13_div_14 : ((13 ^ 13 + 13) % 14) = 12 :=
by
  sorry

end remainder_of_13_pow_13_plus_13_div_14_l138_138750


namespace bowen_spending_l138_138784

noncomputable def total_amount_spent (pen_cost pencil_cost : ℕ) (number_of_pens number_of_pencils : ℕ) : ℕ :=
  number_of_pens * pen_cost + number_of_pencils * pencil_cost

theorem bowen_spending :
  let pen_cost := 15 in
  let pencil_cost := 25 in
  let number_of_pens := 40 in
  let number_of_pencils := number_of_pens + (2 * number_of_pens / 5) in
  total_amount_spent pen_cost pencil_cost number_of_pens number_of_pencils = 2000 :=
by
  sorry

end bowen_spending_l138_138784


namespace number_of_black_squares_in_58th_row_l138_138766

theorem number_of_black_squares_in_58th_row :
  let pattern := [1, 0, 0] -- pattern where 1 represents a black square
  let n := 58
  let total_squares := 2 * n - 1 -- total squares in the 58th row
  let black_count := total_squares / 3 -- number of black squares in the repeating pattern
  black_count = 38 :=
by
  let pattern := [1, 0, 0]
  let n := 58
  let total_squares := 2 * n - 1
  let black_count := total_squares / 3
  have black_count_eq_38 : 38 = (115 / 3) := by sorry
  exact black_count_eq_38.symm

end number_of_black_squares_in_58th_row_l138_138766


namespace prime_ge_7_p2_sub1_div_by_30_l138_138345

theorem prime_ge_7_p2_sub1_div_by_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) :=
sorry

end prime_ge_7_p2_sub1_div_by_30_l138_138345


namespace compare_quadratics_maximize_rectangle_area_l138_138164

-- (Ⅰ) Problem statement for comparing quadratic expressions
theorem compare_quadratics (x : ℝ) : (x + 1) * (x - 3) > (x + 2) * (x - 4) := by
  sorry

-- (Ⅱ) Problem statement for maximizing rectangular area with given perimeter
theorem maximize_rectangle_area (x y : ℝ) (h : 2 * (x + y) = 36) : 
  x = 9 ∧ y = 9 ∧ x * y = 81 := by
  sorry

end compare_quadratics_maximize_rectangle_area_l138_138164


namespace solve_for_b_l138_138974

theorem solve_for_b 
  (b : ℝ)
  (h : (25 * b^2) - 84 = 0) :
  b = (2 * Real.sqrt 21) / 5 ∨ b = -(2 * Real.sqrt 21) / 5 :=
by sorry

end solve_for_b_l138_138974


namespace lottery_jackpot_probability_l138_138687

noncomputable def C (n k : ℕ) : ℕ := Fact.factorial n / (Fact.factorial k * Fact.factorial (n - k))

theorem lottery_jackpot_probability :
  (C 45 6 = 8145060) →
  (100: ℚ) / (C 45 6: ℚ) = 0.0000123 :=
by
  sorry

end lottery_jackpot_probability_l138_138687


namespace second_term_of_arithmetic_sequence_l138_138532

-- Define the statement of the problem
theorem second_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h : a + (a + 2 * d) = 10) : 
  a + d = 5 := 
by 
  sorry

end second_term_of_arithmetic_sequence_l138_138532


namespace same_color_probability_l138_138117

-- Define the total number of balls
def total_balls : ℕ := 4 + 6 + 5

-- Define the number of each color of balls
def white_balls : ℕ := 4
def black_balls : ℕ := 6
def red_balls : ℕ := 5

-- Define the events and probabilities
def pr_event (n : ℕ) (total : ℕ) : ℚ := n / total
def pr_cond_event (n : ℕ) (total : ℕ) : ℚ := n / total

-- Define the probabilities for each compound event
def pr_C1 : ℚ := pr_event white_balls total_balls * pr_cond_event (white_balls - 1) (total_balls - 1)
def pr_C2 : ℚ := pr_event black_balls total_balls * pr_cond_event (black_balls - 1) (total_balls - 1)
def pr_C3 : ℚ := pr_event red_balls total_balls * pr_cond_event (red_balls - 1) (total_balls - 1)

-- Define the total probability
def pr_C : ℚ := pr_C1 + pr_C2 + pr_C3

-- The goal is to prove that the total probability pr_C is equal to 31 / 105
theorem same_color_probability : pr_C = 31 / 105 := 
  by sorry

end same_color_probability_l138_138117


namespace least_lcm_possible_l138_138920

theorem least_lcm_possible (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) : Nat.lcm a c = 12 :=
sorry

end least_lcm_possible_l138_138920


namespace total_eggs_collected_l138_138334

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l138_138334


namespace expected_adjacent_red_pairs_l138_138914

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l138_138914


namespace largest_integer_less_100_leaves_remainder_4_l138_138833

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l138_138833


namespace tenth_term_of_arithmetic_sequence_l138_138641

-- Define the initial conditions: first term 'a' and the common difference 'd'
def a : ℤ := 2
def d : ℤ := 1 - a

-- Define the n-th term of an arithmetic sequence formula
def nth_term (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Statement to prove
theorem tenth_term_of_arithmetic_sequence :
  nth_term a d 10 = -7 := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l138_138641


namespace mac_total_loss_l138_138704

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end mac_total_loss_l138_138704


namespace select_student_B_l138_138030

-- Define the average scores for the students A, B, C, D
def avg_A : ℝ := 85
def avg_B : ℝ := 90
def avg_C : ℝ := 90
def avg_D : ℝ := 85

-- Define the variances for the students A, B, C, D
def var_A : ℝ := 50
def var_B : ℝ := 42
def var_C : ℝ := 50
def var_D : ℝ := 42

-- Theorem stating the selected student should be B
theorem select_student_B (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ)
  (h_avg_A : avg_A = 85) (h_avg_B : avg_B = 90) (h_avg_C : avg_C = 90) (h_avg_D : avg_D = 85)
  (h_var_A : var_A = 50) (h_var_B : var_B = 42) (h_var_C : var_C = 50) (h_var_D : var_D = 42) :
  (avg_B = 90 ∧ avg_C = 90 ∧ avg_B ≥ avg_A ∧ avg_B ≥ avg_D ∧ var_B < var_C) → 
  (select_student = "B") :=
by
  sorry

end select_student_B_l138_138030


namespace correct_addition_result_l138_138947

-- Define the particular number x and state the condition.
variable (x : ℕ) (h₁ : x + 21 = 52)

-- Assert that the correct result when adding 40 to x is 71.
theorem correct_addition_result : x + 40 = 71 :=
by
  -- Proof would go here; represented as a placeholder for now.
  sorry

end correct_addition_result_l138_138947


namespace maximize_greenhouse_planting_area_l138_138478

theorem maximize_greenhouse_planting_area
    (a b : ℝ)
    (h : a * b = 800)
    (planting_area : ℝ := (a - 4) * (b - 2)) :
  (a = 40 ∧ b = 20) ↔ planting_area = 648 :=
by
  sorry

end maximize_greenhouse_planting_area_l138_138478


namespace train_departure_time_l138_138592

-- Conditions
def arrival_time : ℕ := 1000  -- Representing 10:00 as 1000 (in minutes since midnight)
def travel_time : ℕ := 15  -- 15 minutes

-- Definition of time subtraction
def time_sub (arrival : ℕ) (travel : ℕ) : ℕ :=
arrival - travel

-- Proof that the train left at 9:45
theorem train_departure_time : time_sub arrival_time travel_time = 945 := by
  sorry

end train_departure_time_l138_138592


namespace area_of_triangle_MAB_l138_138539

noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_MAB :
  let C1 (p : ℝ × ℝ) := p.1^2 - p.2^2 = 2
  let C2 (p : ℝ × ℝ) := ∃ θ, p.1 = 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ
  let M := (3.0, 0.0)
  let A := (2, 2 * Real.sin (Real.pi / 6))
  let B := (2 * Real.sqrt 3, 2 * Real.sin (Real.pi / 6))
  triangle_area A B M = (3 * Real.sqrt 3 - 3) / 2 :=
by
  sorry

end area_of_triangle_MAB_l138_138539


namespace complement_A_is_correct_l138_138099

-- Let A be the set representing the domain of the function y = log2(x - 1)
def A : Set ℝ := { x : ℝ | x > 1 }

-- The universal set is ℝ
def U : Set ℝ := Set.univ

-- Complement of A with respect to ℝ
def complement_A (U : Set ℝ) (A : Set ℝ) : Set ℝ := U \ A

-- Prove that the complement of A with respect to ℝ is (-∞, 1]
theorem complement_A_is_correct : complement_A U A = { x : ℝ | x ≤ 1 } :=
by {
 sorry
}

end complement_A_is_correct_l138_138099


namespace inequality_am_gm_l138_138554

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / c + c / b) ≥ (4 * a / (a + b)) ∧ (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by
  -- Proof steps
  sorry

end inequality_am_gm_l138_138554


namespace smallest_nat_number_l138_138292

theorem smallest_nat_number : ∃ a : ℕ, (a % 3 = 2) ∧ (a % 5 = 4) ∧ (a % 7 = 4) ∧ (∀ b : ℕ, (b % 3 = 2) ∧ (b % 5 = 4) ∧ (b % 7 = 4) → a ≤ b) ∧ a = 74 := 
sorry

end smallest_nat_number_l138_138292


namespace andrew_vacation_days_l138_138481

-- Andrew's working days and vacation accrual rate
def days_worked : ℕ := 300
def vacation_rate : Nat := 10
def vacation_days_earned : ℕ := days_worked / vacation_rate

-- Days off in March and September
def days_off_march : ℕ := 5
def days_off_september : ℕ := 2 * days_off_march
def total_days_off : ℕ := days_off_march + days_off_september

-- Remaining vacation days calculation
def remaining_vacation_days : ℕ := vacation_days_earned - total_days_off

-- Problem statement to prove
theorem andrew_vacation_days : remaining_vacation_days = 15 :=
by
  -- Substitute the known values and perform the calculation
  unfold remaining_vacation_days vacation_days_earned total_days_off vacation_rate days_off_march days_off_september days_worked
  norm_num
  sorry

end andrew_vacation_days_l138_138481


namespace polyhedron_space_diagonals_l138_138170

theorem polyhedron_space_diagonals (V E F T P : ℕ) (total_pairs_of_vertices total_edges total_face_diagonals : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 40)
  (hT : T = 30)
  (hP : P = 10)
  (h_total_pairs_of_vertices : total_pairs_of_vertices = 30 * 29 / 2)
  (h_total_face_diagonals : total_face_diagonals = 5 * 10)
  :
  total_pairs_of_vertices - E - total_face_diagonals = 315 := 
by
  sorry

end polyhedron_space_diagonals_l138_138170


namespace terminating_decimal_of_fraction_l138_138190

theorem terminating_decimal_of_fraction (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 624) : 
  (∃ m : ℕ, 10^m * (n / 625) = k) → ∃ m, m = 624 :=
sorry

end terminating_decimal_of_fraction_l138_138190


namespace smallest_n_divisible_by_2022_l138_138573

theorem smallest_n_divisible_by_2022 (n : ℕ) (h1 : n > 1) (h2 : (n^7 - 1) % 2022 = 0) : n = 79 :=
sorry

end smallest_n_divisible_by_2022_l138_138573


namespace triangle_inequality_l138_138130

variable (a b c p : ℝ)
variable (triangle : a + b > c ∧ a + c > b ∧ b + c > a)
variable (h_p : p = (a + b + c) / 2)

theorem triangle_inequality : 2 * Real.sqrt ((p - b) * (p - c)) ≤ a :=
sorry

end triangle_inequality_l138_138130


namespace find_a_l138_138800

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l138_138800


namespace general_term_find_n_l138_138275

-- Preconditions
def a_10 := 30
def a_20 := 50

-- General term a_n
theorem general_term (n : ℕ) (a_n : ℕ → ℤ) (d : ℤ) (a_1 : ℤ) :
  (a_1 + 9 * d = a_10) ∧ (a_1 + 19 * d = a_20) → a_n n = 2 * n + 10 :=
by
sorry

-- Sum of first n terms
def S_n (n : ℕ) : ℤ := 242
def a_1_val := 12
def d_val := 2

theorem find_n (n : ℕ) : S_n n = n * a_1_val + (n * (n - 1) / 2) * d_val → n = 11 :=
by
sorry

end general_term_find_n_l138_138275


namespace mike_ride_distance_l138_138708

theorem mike_ride_distance (M : ℕ) 
  (cost_Mike : ℝ) 
  (cost_Annie : ℝ) 
  (annies_miles : ℕ := 26) 
  (annies_toll : ℝ := 5) 
  (mile_cost : ℝ := 0.25) 
  (initial_fee : ℝ := 2.5)
  (hc_Mike : cost_Mike = initial_fee + mile_cost * M)
  (hc_Annie : cost_Annie = initial_fee + annies_toll + mile_cost * annies_miles)
  (heq : cost_Mike = cost_Annie) :
  M = 46 := by 
  sorry

end mike_ride_distance_l138_138708


namespace smallest_positive_number_div_conditions_is_perfect_square_l138_138200

theorem smallest_positive_number_div_conditions_is_perfect_square :
  ∃ n : ℕ,
    (n % 11 = 10) ∧
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    (∃ k : ℕ, n = k * k) ∧
    n = 2782559 :=
by
  sorry

end smallest_positive_number_div_conditions_is_perfect_square_l138_138200


namespace dividend_calculation_l138_138429

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 6
  let dividend := 159
  (divisor * quotient) + remainder = dividend :=
by
  sorry

end dividend_calculation_l138_138429


namespace proof_a_eq_x_and_b_eq_x_pow_x_l138_138121

theorem proof_a_eq_x_and_b_eq_x_pow_x
  {a b x : ℕ}
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_x : 0 < x)
  (h : x^(a + b) = a^b * b) :
  a = x ∧ b = x^x := 
by
  sorry

end proof_a_eq_x_and_b_eq_x_pow_x_l138_138121


namespace points_after_perfect_games_l138_138176

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l138_138176


namespace pigeons_percentage_l138_138389

theorem pigeons_percentage (total_birds pigeons sparrows crows doves non_sparrows : ℕ)
  (h_total : total_birds = 100)
  (h_pigeons : pigeons = 40)
  (h_sparrows : sparrows = 20)
  (h_crows : crows = 15)
  (h_doves : doves = 25)
  (h_non_sparrows : non_sparrows = total_birds - sparrows) :
  (pigeons / non_sparrows : ℚ) * 100 = 50 :=
sorry

end pigeons_percentage_l138_138389


namespace joe_money_fraction_l138_138545

theorem joe_money_fraction :
  ∃ f : ℝ,
    (200 : ℝ) = 160 + (200 - 160) ∧
    160 - 160 * f - 20 = 40 + 160 * f + 20 ∧
    f = 1 / 4 :=
by
  -- The proof should go here.
  sorry

end joe_money_fraction_l138_138545


namespace factorize_expression_l138_138353

theorem factorize_expression (x y : ℝ) : 
  x^3 - x*y^2 = x * (x + y) * (x - y) :=
sorry

end factorize_expression_l138_138353


namespace totalBirdsOnFence_l138_138763

/-
Statement: Given initial birds and additional birds joining, the total number
           of birds sitting on the fence is 10.
Conditions:
1. Initially, there are 4 birds.
2. 6 more birds join them.
3. There are 46 storks on the fence, but they do not affect the number of birds.
-/

def initialBirds : Nat := 4
def additionalBirds : Nat := 6

theorem totalBirdsOnFence : initialBirds + additionalBirds = 10 := by
  sorry

end totalBirdsOnFence_l138_138763


namespace zero_point_in_interval_l138_138593

noncomputable def f (x a : ℝ) := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : 
  (∃ x, 1 < x ∧ x < 2 ∧ f x a = 0) → 0 < a ∧ a < 3 :=
by
  sorry

end zero_point_in_interval_l138_138593


namespace find_a_l138_138807

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l138_138807


namespace max_area_of_garden_l138_138468

theorem max_area_of_garden
  (w : ℕ) (l : ℕ)
  (h1 : l = 2 * w)
  (h2 : l + 2 * w = 480) : l * w = 28800 :=
sorry

end max_area_of_garden_l138_138468


namespace find_fourth_vertex_l138_138595

open Complex

theorem find_fourth_vertex (A B C: ℂ) (hA: A = 2 + 3 * Complex.I) 
                            (hB: B = -3 + 2 * Complex.I) 
                            (hC: C = -2 - 3 * Complex.I) : 
                            ∃ D : ℂ, D = 2.5 + 0.5 * Complex.I :=
by 
  sorry

end find_fourth_vertex_l138_138595


namespace cube_edge_probability_l138_138040

theorem cube_edge_probability :
  let num_vertices := 8
  let edges_per_vertex := 3
  let total_vertex_pairs := (Nat.choose num_vertices 2)
  let total_edges := (num_vertices * edges_per_vertex) / 2
  (total_edges : ℚ) / (total_vertex_pairs : ℚ) = 3 / 7 :=
by
  sorry

end cube_edge_probability_l138_138040


namespace jack_keeps_10800_pounds_l138_138875

def number_of_months_in_a_quarter := 12 / 4
def monthly_hunting_trips := 6
def total_hunting_trips := monthly_hunting_trips * number_of_months_in_a_quarter
def deers_per_trip := 2
def total_deers := total_hunting_trips * deers_per_trip
def weight_per_deer := 600
def total_weight := total_deers * weight_per_deer
def kept_weight_fraction := 1 / 2
def kept_weight := total_weight * kept_weight_fraction

theorem jack_keeps_10800_pounds :
  kept_weight = 10800 :=
by
  -- This is a stub for the automated proof
  sorry

end jack_keeps_10800_pounds_l138_138875


namespace cost_of_new_shoes_l138_138465

theorem cost_of_new_shoes 
    (R : ℝ) 
    (L_r : ℝ) 
    (L_n : ℝ) 
    (increase_percent : ℝ) 
    (H_R : R = 13.50) 
    (H_L_r : L_r = 1) 
    (H_L_n : L_n = 2) 
    (H_inc_percent : increase_percent = 0.1852) : 
    2 * (R * (1 + increase_percent) / L_n) = 32.0004 := 
by
    sorry

end cost_of_new_shoes_l138_138465


namespace right_triangle_side_length_l138_138201

theorem right_triangle_side_length (x : ℝ) (hx : x > 0) (h_area : (1 / 2) * x * (3 * x) = 108) :
  x = 6 * Real.sqrt 2 :=
sorry

end right_triangle_side_length_l138_138201


namespace possible_values_on_Saras_card_l138_138126

theorem possible_values_on_Saras_card :
  ∀ (y : ℝ), (0 < y ∧ y < π / 2) →
  let sin_y := Real.sin y
  let cos_y := Real.cos y
  let tan_y := Real.tan y
  (∃ (s l k : ℝ), s = sin_y ∧ l = cos_y ∧ k = tan_y ∧
  (s = l ∨ s = k ∨ l = k) ∧ (s = l ∧ l ≠ k) ∧ s = l ∧ s = 1) :=
sorry

end possible_values_on_Saras_card_l138_138126


namespace largest_int_mod_6_less_than_100_l138_138822

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l138_138822


namespace multiple_of_savings_l138_138781

theorem multiple_of_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1 / 4) * P
  let monthly_non_savings := (3 / 4) * P
  let total_yearly_savings := 12 * monthly_savings
  ∃ M : ℝ, total_yearly_savings = M * monthly_non_savings ∧ M = 4 := 
by
  sorry

end multiple_of_savings_l138_138781


namespace solution_set_l138_138864

-- Define the function and the conditions
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem solution_set (hf_even : is_even f)
                     (hf_increasing : increasing_on f (Set.Ioi 0))
                     (hf_value : f (-2013) = 0) :
  {x | x * f x < 0} = {x | x < -2013 ∨ (0 < x ∧ x < 2013)} :=
by
  sorry

end solution_set_l138_138864


namespace mark_age_l138_138706

-- Definitions based on the conditions in the problem
variables (M J P : ℕ)  -- Current ages of Mark, John, and their parents respectively

-- Condition definitions
def condition1 : Prop := J = M - 10
def condition2 : Prop := P = 5 * J
def condition3 : Prop := P - 22 = M

-- The theorem to prove the correct answer
theorem mark_age : condition1 M J ∧ condition2 J P ∧ condition3 P M → M = 18 := by
  sorry

end mark_age_l138_138706


namespace curler_ratio_l138_138195

theorem curler_ratio
  (total_curlers : ℕ)
  (pink_curlers : ℕ)
  (blue_curlers : ℕ)
  (green_curlers : ℕ)
  (h1 : total_curlers = 16)
  (h2 : blue_curlers = 2 * pink_curlers)
  (h3 : green_curlers = 4) :
  pink_curlers / total_curlers = 1 / 4 := by
  sorry

end curler_ratio_l138_138195


namespace janice_bought_30_fifty_cent_items_l138_138544

theorem janice_bought_30_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 150 * y + 300 * z = 4500) : x = 30 :=
by
  sorry

end janice_bought_30_fifty_cent_items_l138_138544


namespace g_1_5_l138_138026

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g x ≠ 0

axiom g_zero : g 0 = 0

axiom g_mono (x y : ℝ) (hx : 0 ≤ x ∧ x < y ∧ y ≤ 1) : g x ≤ g y

axiom g_symmetry (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (1 - x) = 1 - g x

axiom g_scaling (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (x/4) = g x / 2

theorem g_1_5 : g (1 / 5) = 1 / 4 := 
sorry

end g_1_5_l138_138026


namespace max_handshakes_25_people_l138_138165

theorem max_handshakes_25_people : 
  (∃ n : ℕ, n = 25) → 
  (∀ p : ℕ, p ≤ 24) → 
  ∃ m : ℕ, m = 300 :=
by sorry

end max_handshakes_25_people_l138_138165


namespace lightest_weight_minimum_l138_138933

theorem lightest_weight_minimum (distinct_masses : ∀ {w : set ℤ}, ∀ (x ∈ w) (y ∈ w), x = y → x = y)
  (lightest_weight_ratio : ∀ {weights : list ℤ} (m : ℤ), m = list.minimum weights →
     sum (list.filter (≠ m) weights) = 71 * m)
  (two_lightest_weights_ratio : ∀ {weights : list ℤ} (n m : ℤ), m ∈ weights → n ∈ weights →
     n + m = list.minimum (m :: list.erase weights m) →
     sum (list.filter (≠ n + m) weights) = 34 * (n + m)) :
  ∃ (m : ℤ), m = 35 := 
sorry

end lightest_weight_minimum_l138_138933


namespace sandy_money_left_l138_138565

theorem sandy_money_left (total_money : ℝ) (spent_percentage : ℝ) (money_left : ℝ) : 
  total_money = 320 → spent_percentage = 0.30 → money_left = (total_money * (1 - spent_percentage)) → 
  money_left = 224 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end sandy_money_left_l138_138565


namespace smallest_positive_integer_x_for_2520x_eq_m_cubed_l138_138976

theorem smallest_positive_integer_x_for_2520x_eq_m_cubed :
  ∃ (M x : ℕ), x > 0 ∧ 2520 * x = M^3 ∧ (∀ y, y > 0 ∧ 2520 * y = M^3 → x ≤ y) :=
sorry

end smallest_positive_integer_x_for_2520x_eq_m_cubed_l138_138976


namespace pay_for_notebook_with_change_l138_138114

theorem pay_for_notebook_with_change : ∃ (a b : ℤ), 16 * a - 27 * b = 1 :=
by
  sorry

end pay_for_notebook_with_change_l138_138114


namespace digit_problem_l138_138916

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end digit_problem_l138_138916


namespace calculate_product_l138_138006

variable (EF FG GH HE : ℚ)
variable (x y : ℚ)

-- Conditions
axiom h1 : EF = 110
axiom h2 : FG = 16 * y^3
axiom h3 : GH = 6 * x + 2
axiom h4 : HE = 64
-- Parallelogram properties
axiom h5 : EF = GH
axiom h6 : FG = HE

theorem calculate_product (EF FG GH HE : ℚ) (x y : ℚ)
  (h1 : EF = 110) (h2 : FG = 16 * y ^ 3) (h3 : GH = 6 * x + 2) (h4 : HE = 64) (h5 : EF = GH) (h6 : FG = HE) :
  x * y = 18 * (4) ^ (1/3) := by
  sorry

end calculate_product_l138_138006


namespace probability_first_less_than_second_die_l138_138064

theorem probability_first_less_than_second_die :
  let die := Finset.range 1 7 in
  let outcomes := die.product die in
  let favorable_outcomes := outcomes.filter (λ p : ℕ × ℕ, p.1 < p.2) in
  (favorable_outcomes.card : ℚ) / outcomes.card = 5 / 12 := sorry

end probability_first_less_than_second_die_l138_138064


namespace cost_of_socks_l138_138276

theorem cost_of_socks (S : ℝ) (players : ℕ) (jersey : ℝ) (shorts : ℝ) 
                      (total_cost : ℝ) 
                      (h1 : players = 16) 
                      (h2 : jersey = 25) 
                      (h3 : shorts = 15.20) 
                      (h4 : total_cost = 752) 
                      (h5 : total_cost = players * (jersey + shorts + S)) 
                      : S = 6.80 := 
by
  sorry

end cost_of_socks_l138_138276


namespace fifth_inequality_l138_138018

theorem fifth_inequality :
  1 + (1 / (2^2 : ℝ)) + (1 / (3^2 : ℝ)) + (1 / (4^2 : ℝ)) + (1 / (5^2 : ℝ)) + (1 / (6^2 : ℝ)) < (11 / 6 : ℝ) :=
by
  sorry

end fifth_inequality_l138_138018


namespace trapezium_division_l138_138320

theorem trapezium_division (h : ℝ) (m n : ℕ) (h_pos : 0 < h) 
  (areas_equal : 4 / (3 * ↑m) = 7 / (6 * ↑n)) :
  m + n = 15 := by
  sorry

end trapezium_division_l138_138320


namespace find_values_l138_138657

noncomputable def equation_satisfaction (x y : ℝ) : Prop :=
  x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3

theorem find_values (x y : ℝ) :
  equation_satisfaction x y → x = 1 / 3 ∧ y = 2 / 3 :=
by
  intro h
  sorry

end find_values_l138_138657


namespace correct_calculation_l138_138157

variable (a : ℝ)

theorem correct_calculation :
  a^6 / (1/2 * a^2) = 2 * a^4 :=
by
  sorry

end correct_calculation_l138_138157


namespace fouad_age_l138_138965

theorem fouad_age (F : ℕ) (Ahmed_current_age : ℕ) (H : Ahmed_current_age = 11) (H2 : F + 4 = 2 * Ahmed_current_age) : F = 18 :=
by
  -- We do not need to write the proof steps, just a placeholder.
  sorry

end fouad_age_l138_138965


namespace bing_location_subject_l138_138925

-- Defining entities
inductive City
| Beijing
| Shanghai
| Chongqing

inductive Subject
| Mathematics
| Chinese
| ForeignLanguage

inductive Teacher
| Jia
| Yi
| Bing

-- Defining the conditions
variables (works_in : Teacher → City) (teaches : Teacher → Subject)

axiom cond1_jia_not_beijing : works_in Teacher.Jia ≠ City.Beijing
axiom cond1_yi_not_shanghai : works_in Teacher.Yi ≠ City.Shanghai
axiom cond2_beijing_not_foreign : ∀ t, works_in t = City.Beijing → teaches t ≠ Subject.ForeignLanguage
axiom cond3_shanghai_math : ∀ t, works_in t = City.Shanghai → teaches t = Subject.Mathematics
axiom cond4_yi_not_chinese : teaches Teacher.Yi ≠ Subject.Chinese

-- The question
theorem bing_location_subject : 
  works_in Teacher.Bing = City.Beijing ∧ teaches Teacher.Bing = Subject.Chinese :=
by
  sorry

end bing_location_subject_l138_138925


namespace points_on_curve_l138_138384

theorem points_on_curve (x y : ℝ) :
  (∃ p : ℝ, y = p^2 + (2 * p - 1) * x + 2 * x^2) ↔ y ≥ x^2 - x :=
by
  sorry

end points_on_curve_l138_138384


namespace cost_of_materials_l138_138966

theorem cost_of_materials (initial_bracelets given_away : ℕ) (sell_price profit : ℝ)
  (h1 : initial_bracelets = 52) 
  (h2 : given_away = 8) 
  (h3 : sell_price = 0.25) 
  (h4 : profit = 8) :
  let remaining_bracelets := initial_bracelets - given_away
  let total_revenue := remaining_bracelets * sell_price
  let cost_of_materials := total_revenue - profit
  cost_of_materials = 3 := 
by
  sorry

end cost_of_materials_l138_138966


namespace robin_camera_pictures_l138_138714

-- Given conditions
def pictures_from_phone : Nat := 35
def num_albums : Nat := 5
def pics_per_album : Nat := 8

-- Calculate total pictures and the number of pictures from the camera
theorem robin_camera_pictures : num_albums * pics_per_album - pictures_from_phone = 5 := by
  sorry

end robin_camera_pictures_l138_138714


namespace marbles_sum_l138_138352

variable {K M : ℕ}

theorem marbles_sum (hFabian_kyle : 15 = 3 * K) (hFabian_miles : 15 = 5 * M) :
  K + M = 8 :=
by
  sorry

end marbles_sum_l138_138352


namespace average_mileage_correct_l138_138180

def total_distance : ℕ := 150 * 2
def sedan_mileage : ℕ := 25
def hybrid_mileage : ℕ := 50
def sedan_gas_used : ℕ := 150 / sedan_mileage
def hybrid_gas_used : ℕ := 150 / hybrid_mileage
def total_gas_used : ℕ := sedan_gas_used + hybrid_gas_used
def average_gas_mileage : ℚ := total_distance / total_gas_used

theorem average_mileage_correct :
  average_gas_mileage = 33 + 1 / 3 :=
by
  sorry

end average_mileage_correct_l138_138180


namespace total_meters_built_l138_138472

/-- Define the length of the road -/
def road_length (L : ℕ) := L = 1000

/-- Define the average meters built per day -/
def average_meters_per_day (A : ℕ) := A = 120

/-- Define the number of days worked from July 29 to August 2 -/
def number_of_days_worked (D : ℕ) := D = 5

/-- The total meters built by the time they finished -/
theorem total_meters_built
  (L A D : ℕ)
  (h1 : road_length L)
  (h2 : average_meters_per_day A)
  (h3 : number_of_days_worked D)
  : L / D * A = 600 := by
  sorry

end total_meters_built_l138_138472


namespace translation_preserves_parallel_and_equal_length_l138_138479

theorem translation_preserves_parallel_and_equal_length
    (A B C D : ℝ)
    (after_translation : (C - A) = (D - B))
    (connecting_parallel : C - A = D - B) :
    (C - A = D - B) ∧ (C - A = D - B) :=
by
  sorry

end translation_preserves_parallel_and_equal_length_l138_138479


namespace least_multiple_17_gt_500_l138_138289

theorem least_multiple_17_gt_500 (n : ℕ) (h : (n = 17)) : ∃ m : ℤ, (m * n > 500 ∧ m * n = 510) :=
  sorry

end least_multiple_17_gt_500_l138_138289


namespace cyclist_C_speed_l138_138284

theorem cyclist_C_speed 
  (dist_XY : ℝ)
  (speed_diff : ℝ)
  (meet_point : ℝ)
  (c d : ℝ)
  (h1 : dist_XY = 90)
  (h2 : speed_diff = 5)
  (h3 : meet_point = 15)
  (h4 : d = c + speed_diff)
  (h5 : 75 = dist_XY - meet_point)
  (h6 : 105 = dist_XY + meet_point)
  (h7 : 75 / c = 105 / d) :
  c = 12.5 :=
sorry

end cyclist_C_speed_l138_138284


namespace initial_percent_l138_138945

theorem initial_percent (x : ℝ) :
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := 
by 
  sorry

end initial_percent_l138_138945


namespace cranberries_left_in_bog_l138_138327

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l138_138327


namespace mandy_chocolate_l138_138768

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l138_138768


namespace license_plates_count_l138_138230

/-
Problem:
I want to choose a license plate that is 4 characters long,
where the first character is a letter,
the last two characters are either a letter or a digit,
and the second character can be a letter or a digit 
but must be the same as either the first or the third character.
Additionally, the fourth character must be different from the first three characters.
-/

def is_letter (c : Char) : Prop := c.isAlpha
def is_digit_or_letter (c : Char) : Prop := c.isAlpha || c.isDigit
noncomputable def count_license_plates : ℕ :=
  let first_char_options := 26
  let third_char_options := 36
  let second_char_options := 2
  let fourth_char_options := 34
  first_char_options * third_char_options * second_char_options * fourth_char_options

theorem license_plates_count : count_license_plates = 59904 := by
  sorry

end license_plates_count_l138_138230


namespace alpha_beta_square_eq_eight_l138_138526

open Real

theorem alpha_beta_square_eq_eight :
  ∃ α β : ℝ, 
  (∀ x : ℝ, x^2 - 2 * x - 1 = 0 ↔ x = α ∨ x = β) → 
  (α ≠ β) → 
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_eq_eight_l138_138526


namespace fish_catch_l138_138695

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l138_138695


namespace midpoint_reflection_sum_l138_138400

/-- 
Points P and R are located at (2, 1) and (12, 15) respectively. 
Point M is the midpoint of segment PR. 
Segment PR is reflected over the y-axis.
We want to prove that the sum of the coordinates of the image of point M (the midpoint of the reflected segment) is 1.
-/
theorem midpoint_reflection_sum : 
  let P := (2, 1)
  let R := (12, 15)
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P_image := (-P.1, P.2)
  let R_image := (-R.1, R.2)
  let M' := ((P_image.1 + R_image.1) / 2, (P_image.2 + R_image.2) / 2)
  (M'.1 + M'.2) = 1 :=
by
  sorry

end midpoint_reflection_sum_l138_138400


namespace citrus_grove_total_orchards_l138_138467

theorem citrus_grove_total_orchards (lemons_orchards oranges_orchards grapefruits_orchards limes_orchards total_orchards : ℕ) 
  (h1 : lemons_orchards = 8) 
  (h2 : oranges_orchards = lemons_orchards / 2) 
  (h3 : grapefruits_orchards = 2) 
  (h4 : limes_orchards = grapefruits_orchards) 
  (h5 : total_orchards = lemons_orchards + oranges_orchards + grapefruits_orchards + limes_orchards) : 
  total_orchards = 16 :=
by 
  sorry

end citrus_grove_total_orchards_l138_138467


namespace largest_integer_with_remainder_l138_138826

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l138_138826


namespace cheryl_material_need_l138_138338

-- Cheryl's conditions
def cheryl_material_used (x : ℚ) : Prop :=
  x + 2/3 - 4/9 = 2/3

-- The proof problem statement
theorem cheryl_material_need : ∃ x : ℚ, cheryl_material_used x ∧ x = 4/9 :=
  sorry

end cheryl_material_need_l138_138338


namespace non_congruent_squares_on_6x6_grid_l138_138226

def lattice_points := finset (ℕ × ℕ)

def squares_of_integer_side_length (n : ℕ) : ℕ :=
  n * n

def squares_diagonal_of_rectangles (a b : ℕ) : ℕ :=
  (6 - a) * (6 - b)

def count_squares : ℕ :=
  (squares_of_integer_side_length 5) + 
  (squares_of_integer_side_length 4) + 
  (squares_of_integer_side_length 3) + 
  (squares_of_integer_side_length 2) + 
  (squares_of_integer_side_length 1) +
  (squares_diagonal_of_rectangles 1 2) + 
  (squares_diagonal_of_rectangles 1 3)

theorem non_congruent_squares_on_6x6_grid :
  count_squares = 90 :=
by 
  unfold count_squares 
  unfold squares_of_integer_side_length 
  unfold squares_diagonal_of_rectangles 
  simp
  sorry

end non_congruent_squares_on_6x6_grid_l138_138226


namespace fraction_operation_correct_l138_138293

theorem fraction_operation_correct {a b : ℝ} :
  (0.2 * a + 0.5 * b) ≠ 0 →
  (2 * a + 5 * b) ≠ 0 →
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
by
  intros h1 h2
  sorry

end fraction_operation_correct_l138_138293


namespace nancy_packs_of_crayons_l138_138398

theorem nancy_packs_of_crayons (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l138_138398


namespace distance_internal_tangent_l138_138273

noncomputable def radius_O := 5
noncomputable def distance_external := 9

theorem distance_internal_tangent (radius_O radius_dist_external : ℝ) 
  (h1 : radius_O = 5) (h2: radius_dist_external = 9) : 
  ∃ r : ℝ, r = 4 ∧ abs (r - radius_O) = 1 := by
  sorry

end distance_internal_tangent_l138_138273


namespace new_class_mean_l138_138238

theorem new_class_mean :
  let students1 := 45
  let mean1 := 80
  let students2 := 4
  let mean2 := 85
  let students3 := 1
  let score3 := 90
  let total_students := students1 + students2 + students3
  let total_score := (students1 * mean1) + (students2 * mean2) + (students3 * score3)
  let class_mean := total_score / total_students
  class_mean = 80.6 := 
by
  sorry

end new_class_mean_l138_138238


namespace complement_union_is_correct_l138_138515

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_is_correct : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end complement_union_is_correct_l138_138515


namespace cranberries_left_l138_138324

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l138_138324


namespace proof_problem_l138_138663

open Real

def p : Prop := ∀ x : ℝ, 2^x + 1 / 2^x > 2
def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ sin x + cos x = 1 / 2

theorem proof_problem : ¬p ∧ ¬q :=
by
  sorry

end proof_problem_l138_138663


namespace exist_pair_sum_to_12_l138_138717

theorem exist_pair_sum_to_12 (S : Set ℤ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (chosen : Set ℤ) (hchosen : chosen ⊆ S) (hsize : chosen.card = 7) :
  ∃x ∈ chosen, ∃y ∈ chosen, x ≠ y ∧ x + y = 12 := 
sorry

end exist_pair_sum_to_12_l138_138717


namespace smallest_two_digit_product_12_l138_138439

theorem smallest_two_digit_product_12 : 
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) ∧ (∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (a * b = 12) ∧ (n = 10 * a + b)) ∧
  (∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (∃ (c d : ℕ), (1 ≤ c ∧ c < 10) ∧ (1 ≤ d ∧ d < 10) ∧ (c * d = 12) ∧ (m = 10 * c + d)) → n ≤ m) ↔ n = 26 :=
by
  sorry

end smallest_two_digit_product_12_l138_138439


namespace least_non_lucky_multiple_of_11_l138_138313

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l138_138313


namespace trigonometric_identity_proof_l138_138508

theorem trigonometric_identity_proof (alpha : Real)
(h1 : Real.tan (alpha + π / 4) = 1 / 2)
(h2 : -π / 2 < alpha ∧ alpha < 0) :
  (2 * Real.sin alpha ^ 2 + Real.sin (2 * alpha)) / Real.cos (alpha - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trigonometric_identity_proof_l138_138508


namespace derivative_of_f_at_pi_over_2_l138_138846

noncomputable def f (x : Real) := 5 * Real.sin x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = 0 :=
by
  -- The proof is omitted
  sorry

end derivative_of_f_at_pi_over_2_l138_138846


namespace eggs_collection_l138_138330

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l138_138330


namespace points_after_perfect_games_l138_138174

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l138_138174


namespace ratio_of_sides_l138_138132

theorem ratio_of_sides (s x y : ℝ) 
    (h1 : 0.1 * s^2 = 0.25 * x * y)
    (h2 : x = s / 10)
    (h3 : y = 4 * s) : x / y = 1 / 40 :=
by
  sorry

end ratio_of_sides_l138_138132


namespace pressure_ratio_l138_138757

-- Define Q, Δu, and A
def Q (Δu A : ℝ) : ℝ := Δu + A

-- Define the relationship for Q = 0
def Q_zero (Δu A : ℝ) : Prop := Q Δu A = 0

-- Define Δu in terms of cv, T, and T0
def Δu (cv T T0 : ℝ) : ℝ := cv * (T - T0)

-- Define A in terms of k, x
def A (k x : ℝ) : ℝ := (k * x^2) / 2

-- Define relationship between k, x, P, and S
def pressure_relation (k x P S : ℝ) : Prop := k * x = P * S

-- Define change in volume
def ΔV (S x : ℝ) : ℝ := S * x

-- Define the expanded volume
def V (n S x : ℝ) : ℝ := (n / (n - 1)) * S * x

-- Ideal gas law relationships
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

-- Initial and expanded states
def initial_state (P0 V0 R T0 : ℝ) : Prop := P0 * V0 = R * T0
def expanded_state (P n V0 R T : ℝ) : Prop := P * n * V0 = R * T

-- Define target proof statement
theorem pressure_ratio (cv k x P S n R T T0 P0 V0 : ℝ)
  (hQ_zero : Q_zero (Δu cv T T0) (A k x))
  (hPressRel : pressure_relation k x P S)
  (hIdealGasLaw : ideal_gas_law P (V n S x) R T)
  (hInitialState : initial_state P0 V0 R T0)
  (hExpandedState : expanded_state P n V0 R T) :
  P / P0 = 1 / (n * (1 + ((n - 1) * R) / (2 * n * cv))) :=
by sorry

end pressure_ratio_l138_138757


namespace range_of_m_l138_138883

theorem range_of_m (A : Set ℝ) (m : ℝ) (h : ∃ x, x ∈ A ∩ {x | x ≠ 0}) :
  -4 < m ∧ m < 0 :=
by
  have A_def : A = {x | x^2 + (m+2)*x + 1 = 0} := sorry
  have h_non_empty : ∃ x, x ∈ A ∧ x ≠ 0 := sorry
  have discriminant : (m+2)^2 - 4 < 0 := sorry
  exact ⟨sorry, sorry⟩

end range_of_m_l138_138883


namespace distance_between_trees_l138_138166

-- Define the conditions
def yard_length : ℝ := 325
def number_of_trees : ℝ := 26
def number_of_intervals : ℝ := number_of_trees - 1

-- Define what we need to prove
theorem distance_between_trees:
  (yard_length / number_of_intervals) = 13 := 
  sorry

end distance_between_trees_l138_138166


namespace expected_winnings_l138_138624

theorem expected_winnings (roll_1_2: ℝ) (roll_3_4: ℝ) (roll_5_6: ℝ) (p1_2 p3_4 p5_6: ℝ) :
    roll_1_2 = 2 →
    roll_3_4 = 4 →
    roll_5_6 = -6 →
    p1_2 = 1 / 8 →
    p3_4 = 1 / 4 →
    p5_6 = 1 / 8 →
    (2 * p1_2 + 2 * p1_2 + 4 * p3_4 + 4 * p3_4 + roll_5_6 * p5_6 + roll_5_6 * p5_6) = 1 := by
  intros
  sorry

end expected_winnings_l138_138624


namespace negation_of_universal_proposition_l138_138585

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
sorry

end negation_of_universal_proposition_l138_138585


namespace find_n_values_l138_138209

theorem find_n_values (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 91 = k^2) : n = 9 ∨ n = 10 :=
sorry

end find_n_values_l138_138209


namespace find_cost_of_pencil_and_pen_l138_138140

variable (p q r : ℝ)

-- Definitions based on conditions
def condition1 := 3 * p + 2 * q + r = 4.20
def condition2 := p + 3 * q + 2 * r = 4.75
def condition3 := 2 * r = 3.00

-- The theorem to prove
theorem find_cost_of_pencil_and_pen (p q r : ℝ) (h1 : condition1 p q r) (h2 : condition2 p q r) (h3 : condition3 r) :
  p + q = 1.12 :=
by
  sorry

end find_cost_of_pencil_and_pen_l138_138140


namespace slope_of_tangent_line_at_origin_l138_138347

theorem slope_of_tangent_line_at_origin : 
  (deriv (λ x : ℝ, Real.exp x) 0) = 1 :=
by
  sorry

end slope_of_tangent_line_at_origin_l138_138347


namespace no_possible_values_of_k_l138_138790

theorem no_possible_values_of_k (k : ℤ) :
  (∀ p q : ℤ, p * q = k ∧ p + q = 58 → ¬ (nat.prime p ∧ nat.prime q)) := 
by
  sorry

end no_possible_values_of_k_l138_138790


namespace point_B_value_l138_138219

/-- Given that point A represents the number 7 on a number line
    and point A is moved 3 units to the right to point B,
    prove that point B represents the number 10 -/
theorem point_B_value (A B : ℤ) (h1: A = 7) (h2: B = A + 3) : B = 10 :=
  sorry

end point_B_value_l138_138219


namespace find_a_find_A_l138_138534

-- Part (I)
theorem find_a (b c : ℝ) (A : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = 5 * Real.pi / 6) :
  ∃ a : ℝ, a = 2 * Real.sqrt 7 :=
by {
  sorry
}

-- Part (II)
theorem find_A (b c : ℝ) (C : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 2 + A) :
  ∃ A : ℝ, A = Real.pi / 6 :=
by {
  sorry
}

end find_a_find_A_l138_138534


namespace find_min_values_l138_138503

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 - 2 * x * y + 6 * y^2 - 14 * x - 6 * y + 72

theorem find_min_values :
  (∀x y : ℝ, f x y ≥ f (15 / 2) (1 / 2)) ∧ f (15 / 2) (1 / 2) = 22.5 :=
by
  sorry

end find_min_values_l138_138503


namespace bertha_gave_away_balls_l138_138788

def balls_initial := 2
def balls_worn_out := 20 / 10
def balls_lost := 20 / 5
def balls_purchased := (20 / 4) * 3
def balls_after_20_games_without_giveaway := balls_initial - balls_worn_out - balls_lost + balls_purchased
def balls_after_20_games := 10

theorem bertha_gave_away_balls : balls_after_20_games_without_giveaway - balls_after_20_games = 1 := by
  sorry

end bertha_gave_away_balls_l138_138788


namespace find_m_n_condition_l138_138198

theorem find_m_n_condition (m n : ℕ) :
  m ≥ 1 ∧ n > m ∧ (42 ^ n ≡ 42 ^ m [MOD 100]) ∧ m + n = 24 :=
sorry

end find_m_n_condition_l138_138198


namespace total_teaching_time_l138_138979

def teaching_times :=
  let eduardo_math_time := 3 * 60
  let eduardo_science_time := 4 * 90
  let eduardo_history_time := 2 * 120
  let total_eduardo_time := eduardo_math_time + eduardo_science_time + eduardo_history_time

  let frankie_math_time := 2 * (3 * 60)
  let frankie_science_time := 2 * (4 * 90)
  let frankie_history_time := 2 * (2 * 120)
  let total_frankie_time := frankie_math_time + frankie_science_time + frankie_history_time

  let georgina_math_time := 3 * (3 * 80)
  let georgina_science_time := 3 * (4 * 100)
  let georgina_history_time := 3 * (2 * 150)
  let total_georgina_time := georgina_math_time + georgina_science_time + georgina_history_time

  total_eduardo_time + total_frankie_time + total_georgina_time

theorem total_teaching_time : teaching_times = 5160 := by
  -- calculations omitted
  sorry

end total_teaching_time_l138_138979


namespace taxi_division_number_of_ways_to_divide_six_people_l138_138021

theorem taxi_division (people : Finset ℕ) (h : people.card = 6) (taxi1 taxi2 : Finset ℕ) 
  (h1 : taxi1.card ≤ 4) (h2 : taxi2.card ≤ 4) (h_union : people = taxi1 ∪ taxi2) (h_disjoint : Disjoint taxi1 taxi2) :
  (taxi1.card = 3 ∧ taxi2.card = 3) ∨ 
  (taxi1.card = 4 ∧ taxi2.card = 2) :=
sorry

theorem number_of_ways_to_divide_six_people : 
  ∃ n : ℕ, n = 50 :=
sorry

end taxi_division_number_of_ways_to_divide_six_people_l138_138021


namespace minimum_value_proof_l138_138667

noncomputable def minimum_value (x : ℝ) (h : x > 1) : ℝ :=
  (x^2 + x + 1) / (x - 1)

theorem minimum_value_proof : ∃ x : ℝ, x > 1 ∧ minimum_value x (by sorry) = 3 + 2*Real.sqrt 3 :=
sorry

end minimum_value_proof_l138_138667


namespace total_driving_time_l138_138621

theorem total_driving_time
    (TotalCattle : ℕ) (Distance : ℝ) (TruckCapacity : ℕ) (Speed : ℝ)
    (hcattle : TotalCattle = 400)
    (hdistance : Distance = 60)
    (hcapacity : TruckCapacity = 20)
    (hspeed : Speed = 60) :
    let Trips := TotalCattle / TruckCapacity,
        OneWayTime := Distance / Speed,
        RoundTripTime := 2 * OneWayTime,
        TotalTime := Trips * RoundTripTime
    in TotalTime = 40 :=
by
  sorry

end total_driving_time_l138_138621


namespace at_least_502_friendly_numbers_l138_138783

def friendly (a : ℤ) : Prop :=
  ∃ (m n : ℤ), m > 0 ∧ n > 0 ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem at_least_502_friendly_numbers :
  ∃ S : Finset ℤ, (∀ a ∈ S, friendly a) ∧ 502 ≤ S.card ∧ ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2012 :=
by
  sorry

end at_least_502_friendly_numbers_l138_138783


namespace pentagon_angles_l138_138323

def is_point_in_convex_pentagon (O A B C D E : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry -- Assume definition of angle in radians

theorem pentagon_angles (O A B C D E: Point) (hO : is_point_in_convex_pentagon O A B C D E)
  (h1: angle A O B = angle B O C) (h2: angle B O C = angle C O D)
  (h3: angle C O D = angle D O E) (h4: angle D O E = angle E O A) :
  (angle E O A = angle A O B) ∨ (angle E O A + angle A O B = π) :=
sorry

end pentagon_angles_l138_138323


namespace wheel_radius_l138_138160

theorem wheel_radius 
(D: ℝ) (N: ℕ) (r: ℝ) 
(hD: D = 88 * 1000) 
(hN: N = 1000) 
(hC: 2 * Real.pi * r * N = D) : 
r = 88 / (2 * Real.pi) :=
by
  sorry

end wheel_radius_l138_138160


namespace nicole_initial_candies_l138_138128

theorem nicole_initial_candies (x : ℕ) (h1 : x / 3 + 5 + 10 = x) : x = 23 := by
  sorry

end nicole_initial_candies_l138_138128


namespace repeating_decimals_sum_as_fraction_l138_138086

noncomputable def repeating_decimal_to_fraction (n : Int) (d : Nat) : Rat :=
  n / (10^d - 1)

theorem repeating_decimals_sum_as_fraction :
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  x1 + x2 + x3 = (283 / 11111 : Rat) :=
by
  let x1 := repeating_decimal_to_fraction 2 1
  let x2 := repeating_decimal_to_fraction 3 2
  let x3 := repeating_decimal_to_fraction 4 4
  have : x1 = 0.2, by sorry
  have : x2 = 0.03, by sorry
  have : x3 = 0.0004, by sorry
  show x1 + x2 + x3 = 283 / 11111
  sorry

end repeating_decimals_sum_as_fraction_l138_138086


namespace positive_integers_square_less_than_three_times_l138_138748

theorem positive_integers_square_less_than_three_times (n : ℕ) (hn : 0 < n) (ineq : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
by sorry

end positive_integers_square_less_than_three_times_l138_138748


namespace probability_of_drawing_ball_three_twice_sum_six_l138_138505

def balls : Finset ℕ := {1, 2, 3, 4}

-- Definitions for the problem
def draws : Type := list ℕ

def valid_draws (d : draws) : Prop :=
  d.length = 3 ∧ d.all (λ n, n ∈ balls)

def sum_is_six (d : draws) : Prop :=
  d.sum = 6

def count_ball_three (d : draws) : ℕ :=
  d.count (3)

def favorable (d : draws) : Prop :=
  count_ball_three d = 2

-- The main theorem we want to prove
theorem probability_of_drawing_ball_three_twice_sum_six : 
  (∑ d in (Finset.filter valid_draws (draws.ball_combinations 4 3)), if favorable d ∧ sum_is_six d then 1 else 0).to_real = 0 := 
  sorry

end probability_of_drawing_ball_three_twice_sum_six_l138_138505


namespace unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l138_138027

theorem unique_solution_x_ln3_plus_x_ln4_eq_x_ln5 :
  ∃! x : ℝ, 0 < x ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) := sorry

end unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l138_138027


namespace expected_adjacent_red_pairs_l138_138913

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l138_138913


namespace ticket_sales_revenue_l138_138424

theorem ticket_sales_revenue :
  let student_ticket_price := 4
  let general_admission_ticket_price := 6
  let total_tickets_sold := 525
  let general_admission_tickets_sold := 388
  let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold
  let money_from_student_tickets := student_tickets_sold * student_ticket_price
  let money_from_general_admission_tickets := general_admission_tickets_sold * general_admission_ticket_price
  let total_money_collected := money_from_student_tickets + money_from_general_admission_tickets
  total_money_collected = 2876 :=
by
  sorry

end ticket_sales_revenue_l138_138424


namespace average_monthly_income_P_and_R_l138_138024

theorem average_monthly_income_P_and_R 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : P = 4000) :
  (P + R) / 2 = 5200 :=
sorry

end average_monthly_income_P_and_R_l138_138024


namespace range_of_a_l138_138999

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x ≤ f a (x + ε))) → a ≤ -1/4 :=
sorry

end range_of_a_l138_138999


namespace sequence_all_integers_l138_138274

open Nat

def a : ℕ → ℤ
| 0 => 1
| 1 => 1
| n+2 => (a (n+1))^2 + 2 / a n

theorem sequence_all_integers :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
by
  sorry

end sequence_all_integers_l138_138274


namespace platform_length_l138_138611

noncomputable def length_of_platform (L : ℝ) : Prop :=
  ∃ (a : ℝ), 
    -- Train starts from rest
    (0 : ℝ) * 24 + (1/2) * a * 24^2 = 300 ∧
    -- Train crosses a platform in 39 seconds
    (0 : ℝ) * 39 + (1/2) * a * 39^2 = 300 + L ∧
    -- Constant acceleration found
    a = (25 : ℝ) / 24

-- Claim that length of platform should be 492.19 meters
theorem platform_length : length_of_platform 492.19 :=
sorry

end platform_length_l138_138611


namespace exponential_inequality_l138_138510

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) :=
sorry

end exponential_inequality_l138_138510


namespace problem_statement_l138_138155

-- Define the expression in Lean
def expr : ℤ := 120 * (120 - 5) - (120 * 120 - 10 + 2)

-- Theorem stating the value of the expression
theorem problem_statement : expr = -592 := by
  sorry

end problem_statement_l138_138155


namespace solve_for_x_l138_138919

theorem solve_for_x (x : ℝ) :
  let area_square1 := (2 * x) ^ 2
  let area_square2 := (5 * x) ^ 2
  let area_triangle := 0.5 * (2 * x) * (5 * x)
  (area_square1 + area_square2 + area_triangle = 850) → x = 5 := by
  sorry

end solve_for_x_l138_138919


namespace polynomial_evaluation_l138_138499

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l138_138499


namespace length_of_football_field_l138_138893

theorem length_of_football_field :
  ∃ x : ℝ, (4 * x + 500 = 1172) ∧ x = 168 :=
by
  use 168
  simp
  sorry

end length_of_football_field_l138_138893


namespace longest_pencil_l138_138720

/-- Hallway dimensions and the longest pencil problem -/
theorem longest_pencil (L : ℝ) : 
    (∃ P : ℝ, P = 3 * L) :=
sorry

end longest_pencil_l138_138720


namespace expected_adjacent_red_pairs_correct_l138_138907

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l138_138907


namespace enemy_plane_hit_probability_l138_138952

theorem enemy_plane_hit_probability : 
  let p_A_hit := 0.6
  let p_B_hit := 0.5
  let p_A_miss := 1 - p_A_hit
  let p_B_miss := 1 - p_B_hit
  let p_both_miss := p_A_miss * p_B_miss
  let p_hit := 1 - p_both_miss
  p_hit = 0.8 :=
by
  simp only [p_A_hit, p_B_hit, p_A_miss, p_B_miss, p_both_miss, p_hit]
  norm_num
  sorry

end enemy_plane_hit_probability_l138_138952


namespace second_smallest_three_digit_in_pascal_triangle_l138_138291

theorem second_smallest_three_digit_in_pascal_triangle (m n : ℕ) :
  (∀ k : ℕ, ∃! r c : ℕ, r ≥ c ∧ r.choose c = k) →
  (∃! r : ℕ, r ≥ 2 ∧ 100 = r.choose 1) →
  (m = 101 ∧ n = 101) :=
by
  sorry

end second_smallest_three_digit_in_pascal_triangle_l138_138291


namespace arithmetic_sequence_general_term_sequence_sum_l138_138662

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (a_5 : a 5 = a 1 + 4 * (a 2 - a 1)) 
  (a_6 : a 6 = a 1 + 5 * (a 2 - a 1)) 
  (h₁ : a 5 + a 6 = 24) 
  (h₂ : S 3 = 15) 
  (Sn_formula : ∀ n, S n = (n / 2 : ℤ) * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  ∀ n, a n = 2 * n + 1 := 
sorry

theorem sequence_sum 
  (a : ℕ → ℤ) (b : ℕ → ℤ) 
  (h : ∀ n, a n = 2 * n + 1) 
  (bn_def : ∀ n, b n = 1 / (a n ^ 2 - 1)) : 
  ∀ n, (Finset.range n).sum (λ n, b n) = n / (4 * (n + 1)) := 
sorry

end arithmetic_sequence_general_term_sequence_sum_l138_138662


namespace find_a_l138_138582

theorem find_a (b c : ℤ) 
  (vertex_condition : ∀ (x : ℝ), x = -1 → (ax^2 + b*x + c) = -2)
  (point_condition : ∀ (x : ℝ), x = 0 → (a*x^2 + b*x + c) = -1) :
  ∃ (a : ℤ), a = 1 :=
by
  sorry

end find_a_l138_138582


namespace smallest_two_digit_number_product_12_l138_138444

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l138_138444


namespace arrangement_volunteers_l138_138019

-- Definitions based on conditions:
def num_volunteers : ℕ := 5

def ways_friday : ℕ := num_volunteers.choose 1
def ways_saturday : ℕ := (num_volunteers - 1).choose 2
def ways_sunday : ℕ := (num_volunteers - 1 - 2).choose 1

-- The proof problem statement:
theorem arrangement_volunteers :
  ways_friday * ways_saturday * ways_sunday = 60 :=
by
  sorry

end arrangement_volunteers_l138_138019


namespace probability_ace_king_queen_l138_138279

-- Definitions based on the conditions
def total_cards := 52
def aces := 4
def kings := 4
def queens := 4

def probability_first_ace := aces / total_cards
def probability_second_king := kings / (total_cards - 1)
def probability_third_queen := queens / (total_cards - 2)

theorem probability_ace_king_queen :
  (probability_first_ace * probability_second_king * probability_third_queen) = (8 / 16575) :=
by sorry

end probability_ace_king_queen_l138_138279


namespace lemons_and_oranges_for_100_gallons_l138_138149

-- Given conditions
def lemons_per_gallon := 30 / 40
def oranges_per_gallon := 20 / 40

-- Theorem to be proven
theorem lemons_and_oranges_for_100_gallons : 
  lemons_per_gallon * 100 = 75 ∧ oranges_per_gallon * 100 = 50 := by
  sorry

end lemons_and_oranges_for_100_gallons_l138_138149


namespace ratio_of_sopranos_to_altos_l138_138169

theorem ratio_of_sopranos_to_altos (S A : ℕ) :
  (10 = 5 * S) ∧ (15 = 5 * A) → (S : ℚ) / (A : ℚ) = 2 / 3 :=
by sorry

end ratio_of_sopranos_to_altos_l138_138169


namespace cranberries_left_l138_138325

def initial_cranberries : ℕ := 60000
def harvested_by_humans : ℕ := initial_cranberries * 40 / 100
def eaten_by_elk : ℕ := 20000

theorem cranberries_left (c : ℕ) : c = initial_cranberries - harvested_by_humans - eaten_by_elk → c = 16000 := by
  sorry

end cranberries_left_l138_138325


namespace find_a_b_and_m_range_l138_138855

-- Definitions and initial conditions
def f (x : ℝ) (a b m : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + m
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 2*a*x + b

-- Problem statement
theorem find_a_b_and_m_range (a b m : ℝ) :
  (∀ x, f_prime x a b = 6 * (x + 0.5)^2 - k) →
  f_prime 1 a b = 0 →
  a = 3 ∧ b = -12 ∧ -20 < m ∧ m < 7 :=
sorry

end find_a_b_and_m_range_l138_138855


namespace chocolate_cost_is_correct_l138_138078

def total_spent : ℕ := 13
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := total_spent - candy_bar_cost

theorem chocolate_cost_is_correct : chocolate_cost = 6 :=
by
  sorry

end chocolate_cost_is_correct_l138_138078


namespace price_change_l138_138143

theorem price_change (P : ℝ) : 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  P4 = P * 0.9216 := 
by 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  show P4 = P * 0.9216
  sorry

end price_change_l138_138143


namespace sum_of_digits_l138_138738

theorem sum_of_digits (A T M : ℕ) (h1 : T = A + 3) (h2 : M = 3)
    (h3 : (∃ k : ℕ, T = k^2 * M) ∧ (∃ l : ℕ, T = 33)) : 
    ∃ x : ℕ, ∃ dsum : ℕ, (A + x) % (M + x) = 0 ∧ dsum = 12 :=
by
  sorry

end sum_of_digits_l138_138738


namespace find_common_difference_l138_138997

noncomputable def common_difference (a₁ d : ℤ) : Prop :=
  let a₂ := a₁ + d
  let a₃ := a₁ + 2 * d
  let S₅ := 5 * a₁ + 10 * d
  a₂ + a₃ = 8 ∧ S₅ = 25 → d = 2

-- Statement of the proof problem
theorem find_common_difference (a₁ d : ℤ) (h : common_difference a₁ d) : d = 2 :=
by sorry

end find_common_difference_l138_138997


namespace sharon_trip_distance_l138_138897

noncomputable def usual_speed (x : ℝ) : ℝ := x / 200

noncomputable def reduced_speed (x : ℝ) : ℝ := x / 200 - 30 / 60

theorem sharon_trip_distance (x : ℝ) (h1 : (x / 3) / usual_speed x + (2 * x / 3) / reduced_speed x = 310) : 
x = 220 :=
by
  sorry

end sharon_trip_distance_l138_138897


namespace geometric_sequence_product_l138_138118

theorem geometric_sequence_product (a1 a5 : ℚ) (a b c : ℚ) (q : ℚ) 
  (h1 : a1 = 8 / 3) 
  (h5 : a5 = 27 / 2)
  (h_common_ratio_pos : q = 3 / 2)
  (h_a : a = a1 * q)
  (h_b : b = a * q)
  (h_c : c = b * q)
  (h5_eq : a5 = a1 * q^4)
  (h_common_ratio_neg : q = -3 / 2 ∨ q = 3 / 2) :
  a * b * c = 216 := by
    sorry

end geometric_sequence_product_l138_138118


namespace sum_of_divisors_of_143_l138_138942

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l138_138942


namespace octagon_perimeter_l138_138937

theorem octagon_perimeter (n : ℕ) (side_length : ℝ) (h1 : n = 8) (h2 : side_length = 2) : 
  n * side_length = 16 :=
by
  sorry

end octagon_perimeter_l138_138937


namespace find_x_l138_138059

theorem find_x (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by
  -- Proof goes here
  sorry

end find_x_l138_138059


namespace number_of_toothpicks_l138_138928

def num_horizontal_toothpicks(lines width : Nat) : Nat := lines * width
def num_vertical_toothpicks(lines height : Nat) : Nat := lines * height

theorem number_of_toothpicks (high wide : Nat) (missing : Nat) 
  (h_high : high = 15) (h_wide : wide = 15) (h_missing : missing = 1) : 
  num_horizontal_toothpicks (high + 1) wide + num_vertical_toothpicks (wide + 1) high - missing = 479 := by
  sorry

end number_of_toothpicks_l138_138928


namespace total_egg_collection_l138_138333

theorem total_egg_collection (
  -- Conditions
  (Benjamin_collects : Nat) (h1 : Benjamin_collects = 6) 
  (Carla_collects : Nat) (h2 : Carla_collects = 3 * Benjamin_collects) 
  (Trisha_collects : Nat) (h3 : Trisha_collects = Benjamin_collects - 4)
  ) : 
  -- Question and answer
  (Total_collects : Nat) (h_total : Total_collects = Benjamin_collects + Carla_collects + Trisha_collects) => 
  (Total_collects = 26) := 
  by
  sorry

end total_egg_collection_l138_138333


namespace ratio_amyl_alcohol_to_ethanol_l138_138675

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end ratio_amyl_alcohol_to_ethanol_l138_138675


namespace quadratic_solution_l138_138031

theorem quadratic_solution :
  (∀ x : ℝ, (x^2 - x - 1 = 0) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2)) :=
by
  intro x
  rw [sub_eq_neg_add, sub_eq_neg_add]
  sorry

end quadratic_solution_l138_138031


namespace Tamika_hours_l138_138137

variable (h : ℕ)

theorem Tamika_hours :
  (45 * h = 55 * 5 + 85) → h = 8 :=
by 
  sorry

end Tamika_hours_l138_138137


namespace john_tax_rate_l138_138879

theorem john_tax_rate { P: Real → Real → Real → Real → Prop }:
  ∀ (cNikes cBoots totalPaid taxRate: ℝ), 
  cNikes = 150 →
  cBoots = 120 →
  totalPaid = 297 →
  taxRate = ((totalPaid - (cNikes + cBoots)) / (cNikes + cBoots)) * 100 →
  taxRate = 10 :=
by
  intros cNikes cBoots totalPaid taxRate HcNikes HcBoots HtotalPaid HtaxRate
  sorry

end john_tax_rate_l138_138879


namespace least_positive_multiple_of_17_gt_500_l138_138286

theorem least_positive_multiple_of_17_gt_500 : ∃ n: ℕ, n > 500 ∧ n % 17 = 0 ∧ n = 510 := by
  sorry

end least_positive_multiple_of_17_gt_500_l138_138286


namespace choir_members_count_l138_138578

theorem choir_members_count (n : ℕ) 
  (h1 : 150 < n) 
  (h2 : n < 300) 
  (h3 : n % 6 = 1) 
  (h4 : n % 8 = 3) 
  (h5 : n % 9 = 2) : 
  n = 163 :=
sorry

end choir_members_count_l138_138578


namespace transformed_curve_l138_138268

theorem transformed_curve :
  (∀ x y : ℝ, 3*x = x' ∧ 4*y = y' → x^2 + y^2 = 1) ↔ (x'^2 / 9 + y'^2 / 16 = 1) :=
by
  sorry

end transformed_curve_l138_138268


namespace product_of_two_numbers_l138_138598

theorem product_of_two_numbers (a b : ℝ)
  (h1 : a + b = 8 * (a - b))
  (h2 : a * b = 30 * (a - b)) :
  a * b = 400 / 7 :=
by
  sorry

end product_of_two_numbers_l138_138598


namespace tap_C_fills_in_6_l138_138297

-- Definitions for the rates at which taps fill the tank
def rate_A := 1/10
def rate_B := 1/15
def rate_combined := 1/3

-- Proof problem: Given the conditions, prove that the third tap fills the tank in 6 hours
theorem tap_C_fills_in_6 (rate_A rate_B rate_combined : ℚ) (h : rate_A + rate_B + 1/x = rate_combined) : x = 6 :=
sorry

end tap_C_fills_in_6_l138_138297


namespace has_zero_when_a_gt_0_l138_138222

noncomputable def f (x a : ℝ) : ℝ :=
  x * Real.log (x - 1) - a

theorem has_zero_when_a_gt_0 (a : ℝ) (h : a > 0) :
  ∃ x0 : ℝ, f x0 a = 0 ∧ 2 < x0 :=
sorry

end has_zero_when_a_gt_0_l138_138222


namespace sum_to_12_of_7_chosen_l138_138716

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end sum_to_12_of_7_chosen_l138_138716


namespace curve_crossing_point_l138_138483

theorem curve_crossing_point :
  (∃ t : ℝ, (t^2 - 4 = 2) ∧ (t^3 - 6 * t + 4 = 4)) ∧
  (∃ t' : ℝ, t ≠ t' ∧ (t'^2 - 4 = 2) ∧ (t'^3 - 6 * t' + 4 = 4)) :=
sorry

end curve_crossing_point_l138_138483


namespace largest_integer_less_100_leaves_remainder_4_l138_138831

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l138_138831


namespace original_number_is_seven_l138_138156

theorem original_number_is_seven (x : ℤ) (h : 3 * x - 6 = 15) : x = 7 :=
by
  sorry

end original_number_is_seven_l138_138156


namespace sum_of_k_l138_138447

theorem sum_of_k (k : ℕ) :
  ((∃ x, x^2 - 4 * x + 3 = 0 ∧ x^2 - 7 * x + k = 0) →
  (k = 6 ∨ k = 12)) →
  (6 + 12 = 18) :=
by sorry

end sum_of_k_l138_138447


namespace city_phone_number_remainder_l138_138713

theorem city_phone_number_remainder :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := sorry

end city_phone_number_remainder_l138_138713


namespace track_length_is_320_l138_138636

noncomputable def length_of_track (x : ℝ) : Prop :=
  (∃ v_b v_s : ℝ, (v_b > 0 ∧ v_s > 0 ∧ v_b + v_s = x / 2 ∧ -- speeds of Brenda and Sally must sum up to half the track length against each other
                    80 / v_b = (x / 2 - 80) / v_s ∧ -- First meeting condition
                    120 / v_s + 80 / v_b = (x / 2 + 40) / v_s + (x - 80) / v_b -- Second meeting condition
                   )) ∧ x = 320

theorem track_length_is_320 : ∃ x : ℝ, length_of_track x :=
by
  use 320
  unfold length_of_track
  simp
  sorry

end track_length_is_320_l138_138636


namespace sum_of_consecutive_integers_l138_138923

theorem sum_of_consecutive_integers (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + 1 = b) (h4 : b + 1 = c) (h5 : a * b * c = 336) : a + b + c = 21 :=
sorry

end sum_of_consecutive_integers_l138_138923


namespace nuts_consumed_range_l138_138403

def diet_day_nuts : Nat := 1
def normal_day_nuts : Nat := diet_day_nuts + 2

def total_nuts_consumed (start_with_diet_day : Bool) : Nat :=
  if start_with_diet_day then
    (10 * diet_day_nuts) + (9 * normal_day_nuts)
  else
    (10 * normal_day_nuts) + (9 * diet_day_nuts)

def min_nuts_consumed : Nat :=
  Nat.min (total_nuts_consumed true) (total_nuts_consumed false)

def max_nuts_consumed : Nat :=
  Nat.max (total_nuts_consumed true) (total_nuts_consumed false)

theorem nuts_consumed_range :
  min_nuts_consumed = 37 ∧ max_nuts_consumed = 39 := by
  sorry

end nuts_consumed_range_l138_138403


namespace f_no_zero_point_l138_138221

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem f_no_zero_point (x : ℝ) (h : x > 0) : f x ≠ 0 :=
by 
  sorry

end f_no_zero_point_l138_138221


namespace veggies_count_l138_138171

def initial_tomatoes := 500
def picked_tomatoes := 325
def initial_potatoes := 400
def picked_potatoes := 270
def initial_cucumbers := 300
def planted_cucumber_plants := 200
def cucumbers_per_plant := 2
def initial_cabbages := 100
def picked_cabbages := 50
def planted_cabbage_plants := 80
def cabbages_per_cabbage_plant := 3

noncomputable def remaining_tomatoes : Nat :=
  initial_tomatoes - picked_tomatoes

noncomputable def remaining_potatoes : Nat :=
  initial_potatoes - picked_potatoes

noncomputable def remaining_cucumbers : Nat :=
  initial_cucumbers + planted_cucumber_plants * cucumbers_per_plant

noncomputable def remaining_cabbages : Nat :=
  (initial_cabbages - picked_cabbages) + planted_cabbage_plants * cabbages_per_cabbage_plant

theorem veggies_count :
  remaining_tomatoes = 175 ∧
  remaining_potatoes = 130 ∧
  remaining_cucumbers = 700 ∧
  remaining_cabbages = 290 :=
by
  sorry

end veggies_count_l138_138171


namespace probability_two_packs_approximately_l138_138060

-- Defining the problem conditions
def initial_pills : ℕ := 10   -- Initial number of pills
def consumption_rate : ℝ := 1 -- Assuming the rate of consumption is 1 pill per time unit
def refill_threshold : ℕ := 1  -- Order new pack when only one pill is left
def total_days : ℕ := 365       -- Total days in a year

-- Defining the probability assertion based on steady-state analysis
def probability_two_packs (n : ℕ) : ℝ :=
  let probability_k_1 (k : ℕ) := (1 : ℝ) / (2 ^ (n - k) * n) in
  let sum_probabilities := ∑ k in finset.range n, probability_k_1 (k + 1) in
  sum_probabilities

theorem probability_two_packs_approximately : probability_two_packs 10 ≈ 0.1998 :=
  sorry  -- Proof is omitted

end probability_two_packs_approximately_l138_138060


namespace base_8_to_10_conversion_l138_138934

theorem base_8_to_10_conversion : (2 * 8^4 + 3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 6 * 8^0) = 10030 := by 
  -- specify the summation directly 
  sorry

end base_8_to_10_conversion_l138_138934


namespace total_popsicle_sticks_l138_138658

def Gino_popsicle_sticks : ℕ := 63
def My_popsicle_sticks : ℕ := 50
def Nick_popsicle_sticks : ℕ := 82

theorem total_popsicle_sticks : Gino_popsicle_sticks + My_popsicle_sticks + Nick_popsicle_sticks = 195 := by
  sorry

end total_popsicle_sticks_l138_138658


namespace smallest_two_digit_number_product_12_l138_138435

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_product_to_twelve (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ), n = 10 * d₁ + d₂ ∧ d₁ * d₂ = 12

theorem smallest_two_digit_number_product_12 :
  ∃ (n : ℕ), is_valid_two_digit_number n ∧ digits_product_to_twelve n ∧ ∀ m, (is_valid_two_digit_number m ∧ digits_product_to_twelve m) → n ≤ m :=
by {
  use 26,
  split,
  { -- Proof that 26 is a valid two-digit number
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Proof that the digits of 26 multiply to 12
    use [2, 6],
    exact ⟨rfl, by norm_num⟩ },
  { -- Proof that 26 is the smallest such number
    intros m hm,
    rcases hm with ⟨⟨hm₁, hm₂⟩, hd⟩,
    cases hd with d₁ hd,
    cases hd with d₂ hd,
    cases hd with hm_eq hd_mul,
    interval_cases m,
    sorry
  }
}

end smallest_two_digit_number_product_12_l138_138435


namespace non_congruent_rectangles_unique_l138_138780

theorem non_congruent_rectangles_unique (P : ℕ) (w : ℕ) (h : ℕ) :
  P = 72 ∧ w = 14 ∧ 2 * (w + h) = P → 
  (∃ h, w = 14 ∧ 2 * (w + h) = 72 ∧ 
  ∀ w' h', w' = w → 2 * (w' + h') = 72 → (h' = h)) :=
by
  sorry

end non_congruent_rectangles_unique_l138_138780


namespace students_suggested_tomatoes_l138_138718

theorem students_suggested_tomatoes (students_total mashed_potatoes bacon tomatoes : ℕ) 
  (h_total : students_total = 826)
  (h_mashed_potatoes : mashed_potatoes = 324)
  (h_bacon : bacon = 374)
  (h_tomatoes : students_total = mashed_potatoes + bacon + tomatoes) :
  tomatoes = 128 :=
by {
  sorry
}

end students_suggested_tomatoes_l138_138718


namespace ants_on_track_l138_138257

/-- Given that ants move on a circular track of length 60 cm at a speed of 1 cm/s
and that there are 48 pairwise collisions in a minute, prove that the possible 
total number of ants on the track is 10, 11, 14, or 25. -/
theorem ants_on_track (x y : ℕ) (h : x * y = 24) : x + y = 10 ∨ x + y = 11 ∨ x + y = 14 ∨ x + y = 25 :=
by sorry

end ants_on_track_l138_138257


namespace intersection_when_a_minus2_range_of_a_if_A_subset_B_l138_138124

namespace ProofProblem

open Set

-- Definitions
def A (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Theorem (1)
theorem intersection_when_a_minus2 : 
  A (-2) ∩ B = { x : ℝ | -5 ≤ x ∧ x < -1 } :=
by
  sorry

-- Theorem (2)
theorem range_of_a_if_A_subset_B : 
  A a ⊆ B → (a ∈ Iic (-4) ∨ a ∈ Ici 3) :=
by
  sorry

end ProofProblem

end intersection_when_a_minus2_range_of_a_if_A_subset_B_l138_138124


namespace max_cosine_value_l138_138550

theorem max_cosine_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 1 ≥ Real.cos a :=
sorry

end max_cosine_value_l138_138550


namespace solve_quadratic_eq_l138_138023

theorem solve_quadratic_eq (a b x : ℝ) :
  12 * a * b * x^2 - (16 * a^2 - 9 * b^2) * x - 12 * a * b = 0 ↔ (x = 4 * a / (3 * b)) ∨ (x = -3 * b / (4 * a)) :=
by
  sorry

end solve_quadratic_eq_l138_138023


namespace range_of_x_l138_138670

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (x / (1 + 2 * x))

theorem range_of_x (x : ℝ) :
  f (x * (3 * x - 2)) < -1 / 3 ↔ (-(1 / 3) < x ∧ x < 0) ∨ ((2 / 3) < x ∧ x < 1) :=
by
  sorry

end range_of_x_l138_138670


namespace total_animals_l138_138970

variable (rats chihuahuas : ℕ)
variable (h1 : rats = 60)
variable (h2 : rats = 6 * chihuahuas)

theorem total_animals (rats : ℕ) (chihuahuas : ℕ) (h1 : rats = 60) (h2 : rats = 6 * chihuahuas) : rats + chihuahuas = 70 := by
  sorry

end total_animals_l138_138970


namespace largest_integer_less_100_leaves_remainder_4_l138_138830

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l138_138830


namespace least_multiple_17_gt_500_l138_138288

theorem least_multiple_17_gt_500 (n : ℕ) (h : (n = 17)) : ∃ m : ℤ, (m * n > 500 ∧ m * n = 510) :=
  sorry

end least_multiple_17_gt_500_l138_138288


namespace total_pets_combined_l138_138576

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end total_pets_combined_l138_138576


namespace unique_k_n_m_solution_l138_138492

-- Problem statement
theorem unique_k_n_m_solution :
  ∃ (k : ℕ) (n : ℕ) (m : ℕ), k = 1 ∧ n = 2 ∧ m = 3 ∧ 3^k + 5^k = n^m ∧
  ∀ (k₀ : ℕ) (n₀ : ℕ) (m₀ : ℕ), (3^k₀ + 5^k₀ = n₀^m₀ ∧ m₀ ≥ 2) → (k₀ = 1 ∧ n₀ = 2 ∧ m₀ = 3) :=
by
  sorry

end unique_k_n_m_solution_l138_138492


namespace sphere_surface_area_l138_138418

theorem sphere_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (32 / 3) * π) : 4 * π * R^2 = 16 * π :=
sorry

end sphere_surface_area_l138_138418


namespace dolphins_scored_15_l138_138867

theorem dolphins_scored_15 (s d : ℤ) 
  (h1 : s + d = 48) 
  (h2 : s - d = 18) : 
  d = 15 := 
sorry

end dolphins_scored_15_l138_138867


namespace negation_of_p_l138_138093

theorem negation_of_p (p : Prop) : (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ (∀ x : ℝ, x > 0 → ¬ ((x + 1) * Real.exp x > 1)) :=
by
  sorry

end negation_of_p_l138_138093


namespace Tom_allowance_leftover_l138_138151

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end Tom_allowance_leftover_l138_138151


namespace sin_theta_value_l138_138668

open Real

theorem sin_theta_value
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo (3 * π / 4) (5 * π / 4))
  (h2 : sin (θ - π / 4) = 5 / 13) :
  sin θ = - (7 * sqrt 2) / 26 :=
  sorry

end sin_theta_value_l138_138668


namespace q_evaluation_l138_138975

def q (x y : ℤ) : ℤ :=
if x >= 0 ∧ y >= 0 then x - y
else if x < 0 ∧ y < 0 then x + 3 * y
else 2 * x + 2 * y

theorem q_evaluation : q (q 1 (-1)) (q (-2) (-3)) = -22 := by
sorry

end q_evaluation_l138_138975


namespace function_increment_l138_138191

theorem function_increment (x₁ x₂ : ℝ) (f : ℝ → ℝ) (h₁ : x₁ = 2) 
                           (h₂ : x₂ = 2.5) (h₃ : ∀ x, f x = x ^ 2) :
  f x₂ - f x₁ = 2.25 :=
by
  sorry

end function_increment_l138_138191


namespace bricks_in_chimney_proof_l138_138638

noncomputable def bricks_in_chimney (h : ℕ) : Prop :=
  let brenda_rate := h / 8
  let brandon_rate := h / 12
  let combined_rate_with_decrease := (brenda_rate + brandon_rate) - 12
  (6 * combined_rate_with_decrease = h) 

theorem bricks_in_chimney_proof : ∃ h : ℕ, bricks_in_chimney h ∧ h = 288 :=
sorry

end bricks_in_chimney_proof_l138_138638


namespace minimum_lightest_weight_l138_138931

-- Definitions
def lightest_weight (m : ℕ) : Prop := ∃ n, 72 * m = 35 * n ∧ m % 35 = 0 ∧ m ≥ 35

-- Theorem statement
theorem minimum_lightest_weight : ∃ m, lightest_weight m ∧ m = 35 :=
by
  use 35
  split
  sorry
  exact rfl

end minimum_lightest_weight_l138_138931


namespace smallest_two_digit_number_product_12_l138_138445

-- Defining the conditions
def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def valid_pair (a b : ℕ) : Prop := 
  is_digit a ∧ is_digit b ∧ (a * b = 12)

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Proving the smallest two-digit number formed by valid pairs
theorem smallest_two_digit_number_product_12 :
  ∃ (a b : ℕ), valid_pair a b ∧ (∀ (c d : ℕ), valid_pair c d → two_digit_number a b ≤ two_digit_number c d) ∧ two_digit_number a b = 26 := 
by
  -- Define the valid pairs (2, 6) and (3, 4)
  let pairs := [(2,6), (6,2), (3,4), (4,3)]
  -- Shortcut: manually verify each resulting two-digit number
  have h1 : two_digit_number 2 6 = 26 := rfl
  have h2 : two_digit_number 6 2 = 62 := rfl
  have h3 : two_digit_number 3 4 = 34 := rfl
  have h4 : two_digit_number 4 3 = 43 := rfl
  -- Verify that 26 is the smallest
  have min_26 : ∀ {a b}, (a, b) ∈ pairs → two_digit_number 2 6 ≤ two_digit_number a b := by
    rintro _ _ (rfl | rfl | rfl | rfl)
    · exact Nat.le_refl 26
    · exact Nat.le_refl 26
    · exact Nat.le_of_lt (by linarith)
    · exact Nat.le_of_lt (by linarith)

  -- Conclude the proof
  use [2, 6]
  split
  · -- Validate the pair (2, 6)
    simp [valid_pair, is_digit]
  split
  · -- Validate that (2, 6) forms the smallest number
    exact min_26
  · -- Validate the resulting smallest number
    exact h1

end smallest_two_digit_number_product_12_l138_138445


namespace line_intersects_parabola_once_l138_138141

theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, -3 * y^2 + 2 * y + 7 = k) ↔ k = 22 / 3 :=
by {
  sorry
}

end line_intersects_parabola_once_l138_138141


namespace first_player_winning_strategy_l138_138869

-- Definitions based on conditions
def initial_position (m n : ℕ) : ℕ × ℕ := (m - 1, n - 1)

-- Main theorem statement
theorem first_player_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (initial_position m n).fst ≠ (initial_position m n).snd ↔ m ≠ n :=
by
  sorry

end first_player_winning_strategy_l138_138869


namespace toothpaste_last_day_l138_138417

theorem toothpaste_last_day (total_toothpaste : ℝ)
  (dad_use_per_brush : ℝ) (dad_brushes_per_day : ℕ)
  (mom_use_per_brush : ℝ) (mom_brushes_per_day : ℕ)
  (anne_use_per_brush : ℝ) (anne_brushes_per_day : ℕ)
  (brother_use_per_brush : ℝ) (brother_brushes_per_day : ℕ)
  (sister_use_per_brush : ℝ) (sister_brushes_per_day : ℕ)
  (grandfather_use_per_brush : ℝ) (grandfather_brushes_per_day : ℕ)
  (guest_use_per_brush : ℝ) (guest_brushes_per_day : ℕ) (guest_days : ℕ)
  (total_usage_per_day : ℝ) :
  total_toothpaste = 80 →
  dad_use_per_brush * dad_brushes_per_day = 16 →
  mom_use_per_brush * mom_brushes_per_day = 12 →
  anne_use_per_brush * anne_brushes_per_day = 8 →
  brother_use_per_brush * brother_brushes_per_day = 4 →
  sister_use_per_brush * sister_brushes_per_day = 2 →
  grandfather_use_per_brush * grandfather_brushes_per_day = 6 →
  guest_use_per_brush * guest_brushes_per_day * guest_days = 6 * 4 →
  total_usage_per_day = 54 →
  80 / 54 = 1 → 
  total_toothpaste / total_usage_per_day = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end toothpaste_last_day_l138_138417


namespace ratio_of_rooms_l138_138640

def rooms_in_danielle_apartment : Nat := 6
def rooms_in_heidi_apartment : Nat := 3 * rooms_in_danielle_apartment
def rooms_in_grant_apartment : Nat := 2

theorem ratio_of_rooms :
  (rooms_in_grant_apartment : ℚ) / (rooms_in_heidi_apartment : ℚ) = 1 / 9 := 
by 
  sorry

end ratio_of_rooms_l138_138640


namespace reciprocal_of_neg3_l138_138029

theorem reciprocal_of_neg3 : 1 / (-3: ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l138_138029


namespace triangle_ABC_area_l138_138639

open Real

-- Define points A, B, and C
structure Point :=
  (x: ℝ)
  (y: ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨8, 2⟩
def C : Point := ⟨6, -1⟩

-- Function to calculate the area of a triangle given vertices A, B, and C
noncomputable def triangle_area (A B C : Point) : ℝ := 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

-- The statement to be proved
theorem triangle_ABC_area : triangle_area A B C = 13.5 :=
by
  sorry

end triangle_ABC_area_l138_138639


namespace students_taking_neither_l138_138560

variable (total_students math_students physics_students both_students : ℕ)
variable (h1 : total_students = 80)
variable (h2 : math_students = 50)
variable (h3 : physics_students = 40)
variable (h4 : both_students = 25)

theorem students_taking_neither (h1 : total_students = 80)
    (h2 : math_students = 50)
    (h3 : physics_students = 40)
    (h4 : both_students = 25) :
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 15 :=
by
    sorry

end students_taking_neither_l138_138560


namespace largest_integer_with_remainder_l138_138828

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l138_138828


namespace necessary_not_sufficient_l138_138988

theorem necessary_not_sufficient (a b : ℝ) : (a > b - 1) ∧ ¬ (a > b - 1 → a > b) := 
sorry

end necessary_not_sufficient_l138_138988


namespace at_least_one_false_l138_138511

theorem at_least_one_false (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
  by
  sorry

end at_least_one_false_l138_138511


namespace ratio_of_female_democrats_l138_138422

theorem ratio_of_female_democrats (F M : ℕ) (total_participants : ℕ) (total_democrats : ℕ) (female_democrats : ℕ) :
    total_participants = 720 →
    (M / 4 : ℝ) = 120 →
    (total_participants / 3 : ℝ) = 240 →
    female_democrats = 120 →
    total_democrats = 240 →
    M + F = total_participants →
    120 / (F : ℝ) = 1 / 2 :=
by sorry

end ratio_of_female_democrats_l138_138422


namespace negation_of_prop1_equiv_l138_138270

-- Given proposition: if x > 1 then x > 0
def prop1 (x : ℝ) : Prop := x > 1 → x > 0

-- Negation of the given proposition: if x ≤ 1 then x ≤ 0
def neg_prop1 (x : ℝ) : Prop := x ≤ 1 → x ≤ 0

-- The theorem to prove that the negation of the proposition "If x > 1, then x > 0" 
-- is "If x ≤ 1, then x ≤ 0"
theorem negation_of_prop1_equiv (x : ℝ) : ¬(prop1 x) ↔ neg_prop1 x :=
by
  sorry

end negation_of_prop1_equiv_l138_138270


namespace expected_pairs_of_red_in_circle_deck_l138_138911

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l138_138911


namespace orthogonal_vectors_l138_138886

open Real

variables (r s : ℝ)

def a : ℝ × ℝ × ℝ := (5, r, -3)
def b : ℝ × ℝ × ℝ := (-1, 2, s)

theorem orthogonal_vectors
  (orthogonality : 5 * (-1) + r * 2 + (-3) * s = 0)
  (magnitude_condition : 34 + r^2 = 4 * (5 + s^2)) :
  ∃ (r s : ℝ), (2 * r - 3 * s = 5) ∧ (r^2 - 4 * s^2 = -14) :=
  sorry

end orthogonal_vectors_l138_138886


namespace convert_base7_to_base2_l138_138490

-- Definitions and conditions
def base7_to_decimal (n : ℕ) : ℕ :=
  2 * 7^1 + 5 * 7^0

def decimal_to_binary (n : ℕ) : ℕ :=
  -- Reversing the binary conversion steps
  -- 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 19
  1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Proof problem
theorem convert_base7_to_base2 : decimal_to_binary (base7_to_decimal 25) = 10011 :=
by {
  sorry
}

end convert_base7_to_base2_l138_138490


namespace equilateral_triangle_intersection_impossible_l138_138009

noncomputable def trihedral_angle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ β = 90 ∧ γ = 90 ∧ α > 0

theorem equilateral_triangle_intersection_impossible :
  ¬ ∀ (α : ℝ), ∀ (β γ : ℝ), trihedral_angle α β γ → 
    ∃ (plane : ℝ → ℝ → ℝ), 
      ∀ (x y z : ℝ), plane x y = z → x = y ∧ y = z ∧ z = x ∧ 
                      x + y + z = 60 :=
sorry

end equilateral_triangle_intersection_impossible_l138_138009


namespace N_eq_M_union_P_l138_138015

open Set

def M : Set ℝ := { x | ∃ n : ℤ, x = n }
def N : Set ℝ := { x | ∃ n : ℤ, x = n / 2 }
def P : Set ℝ := { x | ∃ n : ℤ, x = n + 1/2 }

theorem N_eq_M_union_P : N = M ∪ P := 
sorry

end N_eq_M_union_P_l138_138015


namespace time_for_B_work_alone_l138_138612

def work_rate_A : ℚ := 1 / 6
def work_rate_combined : ℚ := 1 / 3
def work_share_C : ℚ := 1 / 8

theorem time_for_B_work_alone : 
  ∃ x : ℚ, (work_rate_A + 1 / x = work_rate_combined - work_share_C) → x = 24 := 
sorry

end time_for_B_work_alone_l138_138612


namespace cylindrical_to_rectangular_coordinates_l138_138342

theorem cylindrical_to_rectangular_coordinates (r θ z : ℝ) (h1 : r = 6) (h2 : θ = 5 * Real.pi / 3) (h3 : z = 7) :
    (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 7) :=
by
  rw [h1, h2, h3]
  -- Using trigonometric identities:
  have hcos : Real.cos (5 * Real.pi / 3) = 1 / 2 := sorry
  have hsin : Real.sin (5 * Real.pi / 3) = -(Real.sqrt 3) / 2 := sorry
  rw [hcos, hsin]
  simp
  sorry

end cylindrical_to_rectangular_coordinates_l138_138342


namespace smallest_two_digit_l138_138434

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def prod_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  d₁ * d₂ = n

noncomputable def smallest_two_digit_with_digit_product_12 : ℕ :=
  -- this is the digit product of 2 and 6
  if is_two_digit (2 * 10 + 6) then 2 * 10 + 6 else sorry

theorem smallest_two_digit : smallest_two_digit_with_digit_product_12 = 26 := by
  sorry

end smallest_two_digit_l138_138434


namespace asymptote_equation_of_hyperbola_l138_138101

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : Prop :=
  let e := Real.sqrt 2
  e = Real.sqrt (1 + a^2) / a

theorem asymptote_equation_of_hyperbola :
  ∀ (a : ℝ) (h : a > 0), hyperbola_eccentricity a h → (∀ x y : ℝ, (x^2 - y^2 = 1 → y = x ∨ y = -x)) :=
by
  intro a h he
  sorry

end asymptote_equation_of_hyperbola_l138_138101


namespace closest_integer_to_99_times_9_l138_138751

theorem closest_integer_to_99_times_9 :
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  1000 ∈ choices ∧ ∀ (n : ℤ), n ∈ choices → dist 1000 ≤ dist n :=
by
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  sorry

end closest_integer_to_99_times_9_l138_138751


namespace terminating_decimal_numbers_count_l138_138656

def is_terminating_decimal (n: ℕ) : Prop :=
  ∃ m : ℕ, ∃ k : ℤ, n = k * (2^m) * (5^m)

theorem terminating_decimal_numbers_count :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → is_terminating_decimal (n / 500) = true :=
begin
  sorry
end

end terminating_decimal_numbers_count_l138_138656


namespace sqrt_abs_eq_zero_imp_power_eq_neg_one_l138_138105

theorem sqrt_abs_eq_zero_imp_power_eq_neg_one (m n : ℤ) (h : (Real.sqrt (m - 2) + abs (n + 3) = 0)) : (m + n) ^ 2023 = -1 := by
  sorry

end sqrt_abs_eq_zero_imp_power_eq_neg_one_l138_138105


namespace find_alpha_l138_138665

theorem find_alpha (α β : ℝ) (h1 : Real.arctan α = 1/2) (h2 : Real.arctan (α - β) = 1/3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α = π/4 := by
  sorry

end find_alpha_l138_138665


namespace probability_area_greater_than_half_circumference_l138_138473

theorem probability_area_greater_than_half_circumference :
  (∑ r in {1, 2, 3, 4, 5, 6}.toFinset.filter (λ r, r > 1), (1/6 : ℝ)) = 5/6 :=
by
  -- here goes the proof
  sorry

end probability_area_greater_than_half_circumference_l138_138473


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138816

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138816


namespace fish_caught_together_l138_138697

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l138_138697


namespace non_congruent_squares_6x6_grid_l138_138229

theorem non_congruent_squares_6x6_grid : 
  let count_squares (n: ℕ) : ℕ := 
    let horizontal_or_vertical := (6 - n) * (6 - n)
    let diagonal := if n * n <= 6 * 6 then (6 - n + 1) * (6 - n + 1) else 0
    horizontal_or_vertical + diagonal
  (count_squares 1) + (count_squares 2) + (count_squares 3) + (count_squares 4) + (count_squares 5) = 141 :=
by
  sorry

end non_congruent_squares_6x6_grid_l138_138229


namespace grazing_area_proof_l138_138632

noncomputable def grazing_area (s r : ℝ) : ℝ :=
  let A_circle := 3.14 * r^2
  let A_sector := (300 / 360) * A_circle
  let A_triangle := (1.732 / 4) * s^2
  let A_triangle_part := A_triangle / 3
  let A_grazing := A_sector - A_triangle_part
  3 * A_grazing

theorem grazing_area_proof : grazing_area 5 7 = 136.59 :=
  by
  sorry

end grazing_area_proof_l138_138632


namespace max_books_john_can_buy_l138_138878

-- Define the key variables and conditions
def johns_money : ℕ := 3745
def book_cost : ℕ := 285
def sales_tax_rate : ℚ := 0.05

-- Define the total cost per book including tax
def total_cost_per_book : ℝ := book_cost + book_cost * sales_tax_rate

-- Define the inequality problem
theorem max_books_john_can_buy : ∃ (x : ℕ), 300 * x ≤ johns_money ∧ 300 * (x + 1) > johns_money :=
by
  sorry

end max_books_john_can_buy_l138_138878


namespace julie_initial_savings_l138_138880

def calculate_earnings (lawns newspapers dogs : ℕ) (price_lawn price_newspaper price_dog : ℝ) : ℝ :=
  (lawns * price_lawn) + (newspapers * price_newspaper) + (dogs * price_dog)

def calculate_total_spent_bike (earnings remaining_money : ℝ) : ℝ :=
  earnings + remaining_money

def calculate_initial_savings (cost_bike total_spent : ℝ) : ℝ :=
  cost_bike - total_spent

theorem julie_initial_savings :
  let cost_bike := 2345
  let lawns := 20
  let newspapers := 600
  let dogs := 24
  let price_lawn := 20
  let price_newspaper := 0.40
  let price_dog := 15
  let remaining_money := 155
  let earnings := calculate_earnings lawns newspapers dogs price_lawn price_newspaper price_dog
  let total_spent := calculate_total_spent_bike earnings remaining_money
  calculate_initial_savings cost_bike total_spent = 1190 :=
by
  -- Although the proof is not required, this setup assumes correctness.
  sorry

end julie_initial_savings_l138_138880


namespace statement_A_statement_B_statement_C_l138_138208

variable {a b : ℝ}
variable (ha : a > 0) (hb : b > 0)

theorem statement_A : (ab ≤ 1) → (1/a + 1/b ≥ 2) :=
by
  sorry

theorem statement_B : (a + b = 4) → (∀ x, (x = 1/a + 9/b) → (x ≥ 4)) :=
by
  sorry

theorem statement_C : (a^2 + b^2 = 4) → (ab ≤ 2) :=
by
  sorry

end statement_A_statement_B_statement_C_l138_138208


namespace largest_int_less_than_100_mod_6_eq_4_l138_138818

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l138_138818


namespace nested_abs_expression_eval_l138_138336

theorem nested_abs_expression_eval :
  abs (abs (-abs (-2 + 3) - 2) + 3) = 6 := sorry

end nested_abs_expression_eval_l138_138336


namespace chapters_ratio_l138_138637

theorem chapters_ratio
  (c1 : ℕ) (c2 : ℕ) (total : ℕ) (x : ℕ)
  (h1 : c1 = 20)
  (h2 : c2 = 15)
  (h3 : total = 75)
  (h4 : x = (c1 + 2 * c2) / 2)
  (h5 : c1 + 2 * c2 + x = total) :
  (x : ℚ) / (c1 + 2 * c2 : ℚ) = 1 / 2 :=
by
  sorry

end chapters_ratio_l138_138637


namespace sum_of_repeating_decimals_l138_138081

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end sum_of_repeating_decimals_l138_138081


namespace mac_loses_l138_138703

def dime_value := 0.10
def nickel_value := 0.05
def quarter_value := 0.25

def dimes_per_quarter := 3
def nickels_per_quarter := 7

def num_quarters_with_dimes := 20
def num_quarters_with_nickels := 20

def total_quarters := num_quarters_with_dimes + num_quarters_with_nickels

def value_of_quarters_received := total_quarters * quarter_value
def value_of_dimes_traded := num_quarters_with_dimes * dimes_per_quarter * dime_value
def value_of_nickels_traded := num_quarters_with_nickels * nickels_per_quarter * nickel_value

def total_value_traded := value_of_dimes_traded + value_of_nickels_traded

theorem mac_loses
  : total_value_traded - value_of_quarters_received = 3.00 :=
by sorry

end mac_loses_l138_138703


namespace cos_seventh_eq_sum_of_cos_l138_138148

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end cos_seventh_eq_sum_of_cos_l138_138148


namespace remainder_of_power_modulo_l138_138049

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l138_138049


namespace sum_of_divisors_143_l138_138938

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l138_138938


namespace positive_integers_divide_n_plus_7_l138_138601

theorem positive_integers_divide_n_plus_7 (n : ℕ) (hn_pos : 0 < n) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 :=
by 
  sorry

end positive_integers_divide_n_plus_7_l138_138601


namespace compare_flavors_l138_138955

def flavor_ratings_A := [7, 9, 8, 6, 10]
def flavor_ratings_B := [5, 6, 10, 10, 9]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem compare_flavors : 
  mean flavor_ratings_A = mean flavor_ratings_B ∧ variance flavor_ratings_A < variance flavor_ratings_B := by
  sorry

end compare_flavors_l138_138955


namespace abs_neg_one_over_2023_l138_138903

theorem abs_neg_one_over_2023 : abs (-1 / 2023) = 1 / 2023 :=
by
  sorry

end abs_neg_one_over_2023_l138_138903


namespace xenia_weekly_earnings_l138_138057

theorem xenia_weekly_earnings
  (hours_week_1 : ℕ)
  (hours_week_2 : ℕ)
  (week2_additional_earnings : ℕ)
  (hours_week_3 : ℕ)
  (bonus_week_3 : ℕ)
  (hourly_wage : ℚ)
  (earnings_week_1 : ℚ)
  (earnings_week_2 : ℚ)
  (earnings_week_3 : ℚ)
  (total_earnings : ℚ) :
  hours_week_1 = 18 →
  hours_week_2 = 25 →
  week2_additional_earnings = 60 →
  hours_week_3 = 28 →
  bonus_week_3 = 30 →
  hourly_wage = (60 : ℚ) / (25 - 18) →
  earnings_week_1 = hours_week_1 * hourly_wage →
  earnings_week_2 = hours_week_2 * hourly_wage →
  earnings_week_2 = earnings_week_1 + 60 →
  earnings_week_3 = hours_week_3 * hourly_wage + 30 →
  total_earnings = earnings_week_1 + earnings_week_2 + earnings_week_3 →
  hourly_wage = (857 : ℚ) / 1000 ∧
  total_earnings = (63947 : ℚ) / 100
:= by
  intros h1 h2 h3 h4 h5 hw he1 he2 he2_60 he3 hte
  sorry

end xenia_weekly_earnings_l138_138057


namespace vector_subtraction_l138_138607

def a : ℝ × ℝ := (3, 5)
def b : ℝ × ℝ := (-2, 1)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)

theorem vector_subtraction : (a.1 - two_b.1, a.2 - two_b.2) = (7, 3) := by
  sorry

end vector_subtraction_l138_138607


namespace john_saves_money_l138_138119

def original_spending (coffees_per_day: ℕ) (price_per_coffee: ℕ) : ℕ :=
  coffees_per_day * price_per_coffee

def new_price (original_price: ℕ) (increase_percentage: ℕ) : ℕ :=
  original_price + (original_price * increase_percentage / 100)

def new_coffees_per_day (original_coffees_per_day: ℕ) (reduction_fraction: ℕ) : ℕ :=
  original_coffees_per_day / reduction_fraction

def current_spending (new_coffees_per_day: ℕ) (new_price_per_coffee: ℕ) : ℕ :=
  new_coffees_per_day * new_price_per_coffee

theorem john_saves_money
  (coffees_per_day : ℕ := 4)
  (price_per_coffee : ℕ := 2)
  (increase_percentage : ℕ := 50)
  (reduction_fraction : ℕ := 2) :
  original_spending coffees_per_day price_per_coffee
  - current_spending (new_coffees_per_day coffees_per_day reduction_fraction)
                     (new_price price_per_coffee increase_percentage)
  = 2 := by
{
  sorry
}

end john_saves_money_l138_138119


namespace books_on_shelf_after_removal_l138_138033

theorem books_on_shelf_after_removal :
  let initial_books : ℝ := 38.0
  let books_removed : ℝ := 10.0
  initial_books - books_removed = 28.0 :=
by 
  sorry

end books_on_shelf_after_removal_l138_138033


namespace intersection_is_expected_l138_138549

open Set

def setA : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def expectedIntersection : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_is_expected :
  (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_is_expected_l138_138549


namespace abs_eq_neg_self_iff_l138_138680

theorem abs_eq_neg_self_iff (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by
  -- skipping proof with sorry
  sorry

end abs_eq_neg_self_iff_l138_138680


namespace fifteen_times_number_eq_150_l138_138303

theorem fifteen_times_number_eq_150 (n : ℕ) (h : 15 * n = 150) : n = 10 :=
sorry

end fifteen_times_number_eq_150_l138_138303


namespace find_pair_not_satisfying_equation_l138_138258

theorem find_pair_not_satisfying_equation :
  ¬ (187 * 314 - 104 * 565 = 41) :=
by
  sorry

end find_pair_not_satisfying_equation_l138_138258


namespace range_of_a_l138_138365

theorem range_of_a (f : ℝ → ℝ)
  (h1 : ∀ x < 1, f x = (2 * a - 1) * x + 4 * a)
  (h2 : ∀ x ≥ 1, f x = Real.log x / Real.log a)
  (h_decreasing : ∀ x y, x < y → f x ≥ f y) :
  a ∈ Icc (1/6 : ℝ) (1/2 : ℝ) :=
sorry

end range_of_a_l138_138365


namespace cranberries_left_l138_138328

theorem cranberries_left (total_cranberries : ℕ) (harvested_percent: ℝ) (cranberries_eaten : ℕ) 
  (h1 : total_cranberries = 60000) 
  (h2 : harvested_percent = 0.40) 
  (h3 : cranberries_eaten = 20000) : 
  total_cranberries - (harvested_percent * total_cranberries).to_nat - cranberries_eaten = 16000 := 
by 
  sorry

end cranberries_left_l138_138328


namespace sum_of_cubes_8001_l138_138731
-- Import the entire Mathlib library

-- Define a property on integers
def approx (x y : ℝ) := abs (x - y) < 0.000000000000004

-- Define the variables a and b
variables (a b : ℤ)

-- State the theorem
theorem sum_of_cubes_8001 (h : approx (a * b : ℝ) 19.999999999999996) : a^3 + b^3 = 8001 := 
sorry

end sum_of_cubes_8001_l138_138731


namespace expected_adjacent_red_pairs_l138_138909

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l138_138909


namespace fewer_servings_per_day_l138_138133

theorem fewer_servings_per_day :
  ∀ (daily_consumption servings_old servings_new: ℕ),
    daily_consumption = 64 →
    servings_old = 8 →
    servings_new = 16 →
    (daily_consumption / servings_old) - (daily_consumption / servings_new) = 4 :=
by
  intros daily_consumption servings_old servings_new h1 h2 h3
  sorry

end fewer_servings_per_day_l138_138133


namespace find_a_l138_138803

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l138_138803


namespace hexagon_area_correct_l138_138791

open Real

def point := (ℝ × ℝ)

def hexagon_vertices : list point :=
  [(0,0), (2,4), (6,4), (8,0), (6,-4), (2,-4)]

def hexagon_area (vertices : list point) : ℝ :=
  -- Using the Shoelace formula here as a placeholder for an actual formula or method
  -- Implementation is skipped with 'sorry'
  sorry

theorem hexagon_area_correct :
  hexagon_area hexagon_vertices = 32 := 
  sorry

end hexagon_area_correct_l138_138791


namespace highest_temperature_l138_138518

theorem highest_temperature (lowest_temp : ℝ) (max_temp_diff : ℝ) :
  lowest_temp = 18 → max_temp_diff = 4 → lowest_temp + max_temp_diff = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end highest_temperature_l138_138518


namespace factor_is_2_l138_138772

variable (x : ℕ) (f : ℕ)

theorem factor_is_2 (h₁ : x = 36)
                    (h₂ : ((f * (x + 10)) / 2) - 2 = 44) : f = 2 :=
by {
  sorry
}

end factor_is_2_l138_138772


namespace find_numbers_l138_138652

theorem find_numbers (a b c : ℝ) (x y z: ℝ) (h1 : x + y = z + a) (h2 : x + z = y + b) (h3 : y + z = x + c) :
    x = (a + b - c) / 2 ∧ y = (a - b + c) / 2 ∧ z = (-a + b + c) / 2 := by
  sorry

end find_numbers_l138_138652


namespace mandy_chocolate_l138_138767

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l138_138767


namespace interval_contains_root_l138_138412

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem interval_contains_root :
  f (-1) < 0 → 
  f 0 < 0 → 
  f 1 < 0 → 
  f 2 > 0 → 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intro h1 h2 h3 h4
  sorry

end interval_contains_root_l138_138412


namespace arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l138_138113

-- Definitions based on conditions
def performances : Nat := 8
def singing : Nat := 2
def dance : Nat := 3
def variety : Nat := 3

-- Problem 1: Prove arrangement with a singing program at the beginning and end
theorem arrange_singing_begin_end : 1440 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 2: Prove arrangement with singing programs not adjacent
theorem arrange_singing_not_adjacent : 30240 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 3: Prove arrangement with singing programs adjacent and dance not adjacent
theorem arrange_singing_adjacent_dance_not_adjacent : 2880 = sorry :=
by
  -- proof goes here
  sorry

end arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l138_138113


namespace amy_uploaded_photos_l138_138967

theorem amy_uploaded_photos (albums photos_per_album : ℕ) (h1 : albums = 9) (h2 : photos_per_album = 20) :
  albums * photos_per_album = 180 :=
by {
  sorry
}

end amy_uploaded_photos_l138_138967


namespace find_general_equation_of_line_l138_138590

variables {x y k b : ℝ}

-- Conditions: slope of the line is -2 and sum of its intercepts is 12.
def slope_of_line (l : ℝ → ℝ → Prop) : Prop := ∃ b, ∀ x y, l x y ↔ y = -2 * x + b
def sum_of_intercepts (l : ℝ → ℝ → Prop) : Prop := ∃ b, b + (b / 2) = 12

-- Question: What is the general equation of the line?
noncomputable def general_equation (l : ℝ → ℝ → Prop) : Prop :=
  slope_of_line l ∧ sum_of_intercepts l → ∀ x y, l x y ↔ 2 * x + y - 8 = 0

-- The theorem we need to prove
theorem find_general_equation_of_line (l : ℝ → ℝ → Prop) : general_equation l :=
sorry

end find_general_equation_of_line_l138_138590


namespace log_expression_evaluation_l138_138402

theorem log_expression_evaluation : 
  (4 * Real.log 2 + 3 * Real.log 5 - Real.log (1/5)) = 4 := 
  sorry

end log_expression_evaluation_l138_138402


namespace shepherd_initial_sheep_l138_138316

def sheep_pass_gate (sheep : ℕ) : ℕ :=
  sheep / 2 + 1

noncomputable def shepherd_sheep (initial_sheep : ℕ) : ℕ :=
  (sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate) initial_sheep

theorem shepherd_initial_sheep (initial_sheep : ℕ) (h : shepherd_sheep initial_sheep = 2) :
  initial_sheep = 2 :=
sorry

end shepherd_initial_sheep_l138_138316


namespace exterior_angle_regular_octagon_l138_138542

-- Definition and proof statement
theorem exterior_angle_regular_octagon :
  let n := 8 -- The number of sides of the polygon (octagon)
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  exterior_angle = 45 :=
by
  let n := 8
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  sorry

end exterior_angle_regular_octagon_l138_138542


namespace find_length_of_AB_l138_138563

theorem find_length_of_AB (x y : ℝ) (AP PB AQ QB PQ AB : ℝ) 
  (h1 : AP = 3 * x) 
  (h2 : PB = 4 * x) 
  (h3 : AQ = 4 * y) 
  (h4 : QB = 5 * y)
  (h5 : PQ = 5) 
  (h6 : AP + PB = AB)
  (h7 : AQ + QB = AB)
  (h8 : PQ = AQ - AP)
  (h9 : 7 * x = 9 * y) : 
  AB = 315 := 
by
  sorry

end find_length_of_AB_l138_138563


namespace f_1984_can_be_any_real_l138_138154

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition : ∀ (x y : ℤ), f (x - y^2) = f x + (y^2 - 2 * x) * f y

theorem f_1984_can_be_any_real
    (a : ℝ)
    (h : f 1 = a) : f 1984 = 1984^2 * a := sorry

end f_1984_can_be_any_real_l138_138154


namespace no_all_nine_odd_l138_138005

theorem no_all_nine_odd
  (a1 a2 a3 a4 a5 b1 b2 b3 b4 : ℤ)
  (h1 : a1 % 2 = 1) (h2 : a2 % 2 = 1) (h3 : a3 % 2 = 1)
  (h4 : a4 % 2 = 1) (h5 : a5 % 2 = 1) (h6 : b1 % 2 = 1)
  (h7 : b2 % 2 = 1) (h8 : b3 % 2 = 1) (h9 : b4 % 2 = 1)
  (sum_eq : a1 + a2 + a3 + a4 + a5 = b1 + b2 + b3 + b4) : 
  false :=
sorry

end no_all_nine_odd_l138_138005


namespace cube_volume_ratio_l138_138042

theorem cube_volume_ratio (edge1 edge2 : ℕ) (h1 : edge1 = 10) (h2 : edge2 = 36) :
  (edge1^3 : ℚ) / (edge2^3) = 125 / 5832 :=
by
  sorry

end cube_volume_ratio_l138_138042


namespace possible_last_digits_count_l138_138892

theorem possible_last_digits_count : 
  ∃ s : Finset Nat, s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ n ∈ s, ∃ m, (m % 10 = n) ∧ (m % 3 = 0) := 
sorry

end possible_last_digits_count_l138_138892


namespace cos_pi_minus_half_alpha_l138_138366

-- Conditions given in the problem
variable (α : ℝ)
variable (hα1 : 0 < α ∧ α < π / 2)
variable (hα2 : Real.sin α = 3 / 5)

-- The proof problem statement
theorem cos_pi_minus_half_alpha (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.sin α = 3 / 5) : 
  Real.cos (π - α / 2) = -3 * Real.sqrt 10 / 10 := 
sorry

end cos_pi_minus_half_alpha_l138_138366


namespace arrange_cyclic_sequence_l138_138956

-- Variables and definitions based on the problem conditions
variable {G : Type*} [Group G] [Fintype G] (g h : G) [IsGenerated G {g, h}]
 
-- Lean statement for proof problem
theorem arrange_cyclic_sequence (n : ℕ) (G : Type*) [Group G] [Fintype G]
  (hn : Fintype.card G = n) (g h : G) [IsGeneratedBy G {g, h}] :
  ∃ (s : Fin 2n → G), 
    (∀ i : Fin 2n, s ⟨i + 1 % 2n, (Nat.mod_lt _ (Nat.succ_pos')).1⟩ = g * s ⟨i, sorry⟩ ∨ s ⟨i + 1 % 2n, _⟩ = h * s ⟨i, sorry⟩) ∧
    (s 0 = g * s ⟨2n-1, sorry⟩ ∨ s 0 = h * s ⟨2n-1, sorry⟩) :=
  sorry

end arrange_cyclic_sequence_l138_138956


namespace number_of_matches_in_first_set_l138_138613

theorem number_of_matches_in_first_set
  (avg_next_13_matches : ℕ := 15)
  (total_matches : ℕ := 35)
  (avg_all_matches : ℚ := 23.17142857142857)
  (x : ℕ := total_matches - 13) :
  x = 22 := by
  sorry

end number_of_matches_in_first_set_l138_138613


namespace Allison_uploads_videos_l138_138072

theorem Allison_uploads_videos :
  let halfway := 30 / 2 in
  let first_half_videos := 10 * halfway in
  let second_half_videos_per_day := 10 * 2 in
  let second_half_videos := second_half_videos_per_day * halfway in
  let total_videos := first_half_videos + second_half_videos in
  total_videos = 450 := 
by
  sorry

end Allison_uploads_videos_l138_138072


namespace cranberries_left_l138_138329

theorem cranberries_left (total_cranberries : ℕ) (harvested_percent: ℝ) (cranberries_eaten : ℕ) 
  (h1 : total_cranberries = 60000) 
  (h2 : harvested_percent = 0.40) 
  (h3 : cranberries_eaten = 20000) : 
  total_cranberries - (harvested_percent * total_cranberries).to_nat - cranberries_eaten = 16000 := 
by 
  sorry

end cranberries_left_l138_138329


namespace find_a_l138_138804

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l138_138804


namespace exists_point_P_equal_distance_squares_l138_138926

-- Define the points in the plane representing the vertices of the triangles
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℝ × ℝ}
-- Define the function that calculates the square distance between two points
def sq_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Define the proof statement
theorem exists_point_P_equal_distance_squares :
  ∃ P : ℝ × ℝ,
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P B1 + sq_distance P B2 + sq_distance P B3 ∧
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P C1 + sq_distance P C2 + sq_distance P C3 := sorry

end exists_point_P_equal_distance_squares_l138_138926


namespace max_area_triangle_l138_138541

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

noncomputable def line_eq (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

theorem max_area_triangle (x1 y1 x2 y2 xp yp : ℝ) (h1 : circle_eq x1 y1) (h2 : circle_eq x2 y2) (h3 : circle_eq xp yp)
  (h4 : line_eq x1 y1) (h5 : line_eq x2 y2) (h6 : (xp, yp) ≠ (x1, y1)) (h7 : (xp, yp) ≠ (x2, y2)) :
  ∃ S : ℝ, S = 10 * Real.sqrt 5 / 9 :=
by
  sorry

end max_area_triangle_l138_138541


namespace train_speed_is_36_kph_l138_138963

-- Define the given conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 3

-- Convert distance from meters to kilometers
def distance_kilometers : ℕ -> ℕ := fun d => d / 1000
-- Convert time from minutes to hours
def time_hours : ℕ -> ℚ := fun t => (t : ℚ) / 60

-- Calculate speed in kilometers per hour
def speed_kph (d : ℕ) (t : ℕ) : ℚ :=
  let d_km := d / 1000
  let t_hr := (t : ℚ) / 60
  d_km / t_hr

-- The theorem to prove the speed
theorem train_speed_is_36_kph :
  speed_kph distance_meters time_minutes = 36 := by
  sorry

end train_speed_is_36_kph_l138_138963


namespace draw_from_unit_D_l138_138978

variable (d : ℕ)

-- Variables representing the number of questionnaires drawn from A, B, C, and D
def QA : ℕ := 30 - d
def QB : ℕ := 30
def QC : ℕ := 30 + d
def QD : ℕ := 30 + 2 * d

-- Total number of questionnaires drawn
def TotalDrawn : ℕ := QA d + QB + QC d + QD d

theorem draw_from_unit_D :
  (TotalDrawn d = 150) →
  QD d = 60 := sorry

end draw_from_unit_D_l138_138978


namespace bounded_sequence_range_l138_138361

theorem bounded_sequence_range (a : ℝ) (a_n : ℕ → ℝ) (h1 : a_n 1 = a)
    (hrec : ∀ n : ℕ, a_n (n + 1) = 3 * (a_n n)^3 - 7 * (a_n n)^2 + 5 * (a_n n))
    (bounded : ∃ M : ℝ, ∀ n : ℕ, abs (a_n n) ≤ M) :
    0 ≤ a ∧ a ≤ 4/3 :=
by
  sorry

end bounded_sequence_range_l138_138361


namespace hashN_of_25_l138_138188

def hashN (N : ℝ) : ℝ := 0.6 * N + 2

theorem hashN_of_25 : hashN (hashN (hashN (hashN 25))) = 7.592 :=
by
  sorry

end hashN_of_25_l138_138188


namespace intersection_of_function_and_inverse_l138_138411

theorem intersection_of_function_and_inverse (c k : ℤ) (f : ℤ → ℤ)
  (hf : ∀ x:ℤ, f x = 4 * x + c) 
  (hf_inv : ∀ y:ℤ, (∃ x:ℤ, f x = y) → (∃ x:ℤ, f y = x))
  (h_intersection : ∀ k:ℤ, f 2 = k ∧ f k = 2 ) 
  : k = 2 :=
sorry

end intersection_of_function_and_inverse_l138_138411


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138815

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l138_138815


namespace isosceles_triangle_area_l138_138041

theorem isosceles_triangle_area (a b c : ℝ) (h: a = 5 ∧ b = 5 ∧ c = 6)
  (altitude_splits_base : ∀ (h : 3^2 + x^2 = 25), x = 4) : 
  ∃ (area : ℝ), area = 12 := 
by
  sorry

end isosceles_triangle_area_l138_138041


namespace simplify_triangle_expression_l138_138659

theorem simplify_triangle_expression (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c :=
by
  sorry

end simplify_triangle_expression_l138_138659


namespace geometric_sequence_ratio_l138_138930

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (A B : ℕ → ℝ)
  (hA9 : A 9 = (a 5) ^ 9)
  (hB9 : B 9 = (b 5) ^ 9)
  (h_ratio : a 5 / b 5 = 2) :
  (A 9 / B 9) = 512 := by
  sorry

end geometric_sequence_ratio_l138_138930


namespace sequence_a_n_l138_138401

theorem sequence_a_n {a : ℕ → ℤ}
  (h1 : a 2 = 5)
  (h2 : a 1 = 1)
  (h3 : ∀ n ≥ 2, a (n+1) - 2 * a n + a (n-1) = 7) :
  a 17 = 905 :=
  sorry

end sequence_a_n_l138_138401


namespace mandy_pieces_eq_fifteen_l138_138769

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l138_138769


namespace find_a_l138_138801

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l138_138801


namespace total_eggs_collected_l138_138335

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l138_138335


namespace gcd_g102_g103_l138_138392

def g (x : ℕ) : ℕ := x^2 - x + 2007

theorem gcd_g102_g103 : 
  Nat.gcd (g 102) (g 103) = 3 :=
by
  sorry

end gcd_g102_g103_l138_138392


namespace sum_of_reciprocals_of_root_products_eq_4_l138_138700

theorem sum_of_reciprocals_of_root_products_eq_4
  (p q r s t : ℂ)
  (h_poly : ∀ x : ℂ, x^5 + 10*x^4 + 20*x^3 + 15*x^2 + 8*x + 5 = 0 ∨ (x - p)*(x - q)*(x - r)*(x - s)*(x - t) = 0)
  (h_vieta_2 : p*q + p*r + p*s + p*t + q*r + q*s + q*t + r*s + r*t + s*t = 20)
  (h_vieta_all : p*q*r*s*t = 5) :
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := 
sorry

end sum_of_reciprocals_of_root_products_eq_4_l138_138700


namespace total_savings_percentage_l138_138619

theorem total_savings_percentage
  (original_coat_price : ℕ) (original_pants_price : ℕ)
  (coat_discount_percent : ℚ) (pants_discount_percent : ℚ)
  (original_total_price : ℕ) (total_savings : ℕ)
  (savings_percentage : ℚ) :
  original_coat_price = 120 →
  original_pants_price = 60 →
  coat_discount_percent = 0.30 →
  pants_discount_percent = 0.60 →
  original_total_price = original_coat_price + original_pants_price →
  total_savings = original_coat_price * coat_discount_percent + original_pants_price * pants_discount_percent →
  savings_percentage = (total_savings / original_total_price) * 100 →
  savings_percentage = 40 := 
by
  intros
  sorry

end total_savings_percentage_l138_138619


namespace plane_equation_intercept_l138_138383

theorem plane_equation_intercept (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x y z : ℝ, ∃ k : ℝ, k = 1 → (x / a + y / b + z / c) = k :=
by sorry

end plane_equation_intercept_l138_138383


namespace range_of_a_for_common_tangents_l138_138794

theorem range_of_a_for_common_tangents :
  ∃ (a : ℝ), ∀ (x y : ℝ),
    ((x - 2)^2 + y^2 = 4) ∧ ((x - a)^2 + (y + 3)^2 = 9) →
    (-2 < a) ∧ (a < 6) := by
  sorry

end range_of_a_for_common_tangents_l138_138794


namespace total_yardage_progress_l138_138239

def teamA_moves : List Int := [-5, 8, -3, 6]
def teamB_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress :
  (teamA_moves.sum + teamB_moves.sum) = 10 :=
by
  sorry

end total_yardage_progress_l138_138239


namespace pipes_fill_tank_in_8_hours_l138_138475

theorem pipes_fill_tank_in_8_hours (A B C : ℝ) (hA : A = 1 / 56) (hB : B = 2 * A) (hC : C = 2 * B) :
  1 / (A + B + C) = 8 :=
by
  sorry

end pipes_fill_tank_in_8_hours_l138_138475
