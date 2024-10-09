import Mathlib

namespace mary_total_payment_l1235_123575

def fixed_fee : ℕ := 17
def hourly_charge : ℕ := 7
def rental_duration : ℕ := 9
def total_payment (f : ℕ) (h : ℕ) (r : ℕ) : ℕ := f + (h * r)

theorem mary_total_payment:
  total_payment fixed_fee hourly_charge rental_duration = 80 :=
by
  sorry

end mary_total_payment_l1235_123575


namespace field_trip_count_l1235_123556

theorem field_trip_count (vans: ℕ) (buses: ℕ) (people_per_van: ℕ) (people_per_bus: ℕ)
  (hv: vans = 9) (hb: buses = 10) (hpv: people_per_van = 8) (hpb: people_per_bus = 27):
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end field_trip_count_l1235_123556


namespace interest_rate_and_years_l1235_123579

theorem interest_rate_and_years
    (P : ℝ)
    (n : ℕ)
    (e : ℝ)
    (h1 : P * (e ^ n) * e = P * (e ^ (n + 1)) + 4156.02)
    (h2 : P * (e ^ (n - 1)) = P * (e ^ n) - 3996.12) :
    (e = 1.04) ∧ (P = 60000) ∧ (E = 4/100) ∧ (n = 14) := by
  sorry

end interest_rate_and_years_l1235_123579


namespace ben_final_salary_is_2705_l1235_123584

def initial_salary : ℕ := 3000

def salary_after_raise (salary : ℕ) : ℕ :=
  salary * 110 / 100

def salary_after_pay_cut (salary : ℕ) : ℕ :=
  salary * 85 / 100

def final_salary (initial : ℕ) : ℕ :=
  (salary_after_pay_cut (salary_after_raise initial)) - 100

theorem ben_final_salary_is_2705 : final_salary initial_salary = 2705 := 
by 
  sorry

end ben_final_salary_is_2705_l1235_123584


namespace find_positive_real_solutions_l1235_123580

open Real

theorem find_positive_real_solutions 
  (x : ℝ) 
  (h : (1/3 * (4 * x^2 - 2)) = ((x^2 - 60 * x - 15) * (x^2 + 30 * x + 3))) :
  x = 30 + sqrt 917 ∨ x = -15 + (sqrt 8016) / 6 :=
by sorry

end find_positive_real_solutions_l1235_123580


namespace dorothy_total_sea_glass_l1235_123522

def Blanche_red : ℕ := 3
def Rose_red : ℕ := 9
def Rose_blue : ℕ := 11

def Dorothy_red : ℕ := 2 * (Blanche_red + Rose_red)
def Dorothy_blue : ℕ := 3 * Rose_blue

theorem dorothy_total_sea_glass : Dorothy_red + Dorothy_blue = 57 :=
by
  sorry

end dorothy_total_sea_glass_l1235_123522


namespace quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l1235_123573

-- Proof Problem (1)
theorem quadratic_inequality_roots_a_eq_neg1
  (a : ℝ)
  (h : ∀ x, (-1 < x ∧ x < 3) → ax^2 - 2 * a * x + 3 > 0) :
  a = -1 :=
sorry

-- Proof Problem (2)
theorem quadratic_inequality_for_all_real_a_range
  (a : ℝ)
  (h : ∀ x, ax^2 - 2 * a * x + 3 > 0) :
  0 ≤ a ∧ a < 3 :=
sorry

end quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l1235_123573


namespace power_function_at_point_l1235_123553

theorem power_function_at_point (f : ℝ → ℝ) (h : ∃ α, ∀ x, f x = x^α) (hf : f 2 = 4) : f 3 = 9 :=
sorry

end power_function_at_point_l1235_123553


namespace math_expr_evaluation_l1235_123506

theorem math_expr_evaluation :
  3 + 15 / 3 - 2^2 + 1 = 5 :=
by
  -- The proof will be filled here
  sorry

end math_expr_evaluation_l1235_123506


namespace arithmetic_contains_geometric_l1235_123586

theorem arithmetic_contains_geometric (a d : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) : 
  ∃ b q : ℕ, (b = a) ∧ (q = 1 + d) ∧ (∀ n : ℕ, ∃ k : ℕ, a * (1 + d)^n = a + k * d) :=
by
  sorry

end arithmetic_contains_geometric_l1235_123586


namespace value_of_m_minus_n_l1235_123501

theorem value_of_m_minus_n (m n : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (m : ℂ) / (1 + i) = 1 - n * i) : m - n = 1 :=
sorry

end value_of_m_minus_n_l1235_123501


namespace unique_k_solves_eq_l1235_123550

theorem unique_k_solves_eq (k : ℕ) (hpos_k : k > 0) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = k * a * b) ↔ k = 2 :=
by
  sorry

end unique_k_solves_eq_l1235_123550


namespace unique_pair_a_b_l1235_123531

open Complex

theorem unique_pair_a_b :
  ∃! (a b : ℂ), a^4 * b^3 = 1 ∧ a^6 * b^7 = 1 := by
  sorry

end unique_pair_a_b_l1235_123531


namespace ratio_of_kids_l1235_123567

theorem ratio_of_kids (k2004 k2005 k2006 : ℕ) 
  (h2004: k2004 = 60) 
  (h2005: k2005 = k2004 / 2)
  (h2006: k2006 = 20) :
  (k2006 : ℚ) / k2005 = 2 / 3 :=
by
  sorry

end ratio_of_kids_l1235_123567


namespace volume_conversion_l1235_123500

theorem volume_conversion (v_feet : ℕ) (h : v_feet = 250) : (v_feet / 27 : ℚ) = 250 / 27 := by
  sorry

end volume_conversion_l1235_123500


namespace smallest_num_conditions_l1235_123552

theorem smallest_num_conditions :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧ n = 11 :=
by
  sorry

end smallest_num_conditions_l1235_123552


namespace count_two_digit_decimals_between_0_40_and_0_50_l1235_123544

theorem count_two_digit_decimals_between_0_40_and_0_50 : 
  ∃ (n : ℕ), n = 9 ∧ ∀ x : ℝ, 0.40 < x ∧ x < 0.50 → (exists d : ℕ, (1 ≤ d ∧ d ≤ 9 ∧ x = 0.4 + d * 0.01)) :=
by
  sorry

end count_two_digit_decimals_between_0_40_and_0_50_l1235_123544


namespace reciprocal_of_2022_l1235_123541

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l1235_123541


namespace percentage_deposited_to_wife_is_33_l1235_123566

-- Definitions based on the conditions
def total_income : ℝ := 800000
def children_distribution_rate : ℝ := 0.20
def number_of_children : ℕ := 3
def donation_rate : ℝ := 0.05
def final_amount : ℝ := 40000

-- We can compute the intermediate values to use them in the final proof
def amount_distributed_to_children : ℝ := total_income * children_distribution_rate * number_of_children
def remaining_after_distribution : ℝ := total_income - amount_distributed_to_children
def donation_amount : ℝ := remaining_after_distribution * donation_rate
def remaining_after_donation : ℝ := remaining_after_distribution - donation_amount
def deposited_to_wife : ℝ := remaining_after_donation - final_amount

-- The statement to prove
theorem percentage_deposited_to_wife_is_33 :
  (deposited_to_wife / total_income) * 100 = 33 := by
  sorry

end percentage_deposited_to_wife_is_33_l1235_123566


namespace hawks_score_l1235_123534

theorem hawks_score (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 18) : y = 16 := by
  sorry

end hawks_score_l1235_123534


namespace solve_system_of_equations_l1235_123530

-- Define the given system of equations and conditions
theorem solve_system_of_equations (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : yz / (y + z) = a) 
  (h2 : xz / (x + z) = b) 
  (h3 : xy / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧ 
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧ 
  z = 2 * a * b * c / (a * c + b * c - a * b) := sorry

end solve_system_of_equations_l1235_123530


namespace stripe_width_l1235_123533

theorem stripe_width (x : ℝ) (h : 60 * x - x^2 = 400) : x = 30 - 5 * Real.sqrt 5 := 
  sorry

end stripe_width_l1235_123533


namespace points_per_correct_answer_hard_round_l1235_123592

theorem points_per_correct_answer_hard_round (total_points easy_points_per average_points_per hard_correct : ℕ) 
(easy_correct average_correct : ℕ) : 
  (total_points = (easy_correct * easy_points_per + average_correct * average_points_per) + (hard_correct * 5)) →
  (easy_correct = 6) →
  (easy_points_per = 2) →
  (average_correct = 2) →
  (average_points_per = 3) →
  (hard_correct = 4) →
  (total_points = 38) →
  5 = 5 := 
by
  intros
  sorry

end points_per_correct_answer_hard_round_l1235_123592


namespace cubic_solution_unique_real_l1235_123597

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end cubic_solution_unique_real_l1235_123597


namespace find_m_l1235_123521

theorem find_m (y x m : ℝ) (h1 : 2 - 3 * (1 - y) = 2 * y) (h2 : y = x) (h3 : m * (x - 3) - 2 = -8) : m = 3 :=
sorry

end find_m_l1235_123521


namespace benny_seashells_l1235_123517

-- Define the initial number of seashells Benny found
def seashells_found : ℝ := 66.5

-- Define the percentage of seashells Benny gave away
def percentage_given_away : ℝ := 0.75

-- Calculate the number of seashells Benny gave away
def seashells_given_away : ℝ := percentage_given_away * seashells_found

-- Calculate the number of seashells Benny now has
def seashells_left : ℝ := seashells_found - seashells_given_away

-- Prove that Benny now has 16.625 seashells
theorem benny_seashells : seashells_left = 16.625 :=
by
  sorry

end benny_seashells_l1235_123517


namespace consecutive_sum_to_20_has_one_set_l1235_123558

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l1235_123558


namespace geometric_sequence_a5_l1235_123559

theorem geometric_sequence_a5
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_ratio : ∀ n, a (n + 1) = 2 * a n)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := 
sorry

end geometric_sequence_a5_l1235_123559


namespace chris_initial_donuts_l1235_123572

theorem chris_initial_donuts (D : ℝ) (H1 : D * 0.90 - 4 = 23) : D = 30 := 
by
sorry

end chris_initial_donuts_l1235_123572


namespace max_a_condition_l1235_123508

theorem max_a_condition (a : ℝ) :
  (∀ x : ℝ, x < a → |x| > 2) ∧ (∃ x : ℝ, |x| > 2 ∧ ¬ (x < a)) →
  a ≤ -2 :=
by 
  sorry

end max_a_condition_l1235_123508


namespace injured_player_age_l1235_123524

noncomputable def average_age_full_team := 22
noncomputable def number_of_players := 11
noncomputable def average_age_remaining_players := 21
noncomputable def number_of_remaining_players := 10
noncomputable def total_age_full_team := number_of_players * average_age_full_team
noncomputable def total_age_remaining_players := number_of_remaining_players * average_age_remaining_players

theorem injured_player_age :
  (number_of_players * average_age_full_team) -
  (number_of_remaining_players * average_age_remaining_players) = 32 :=
by
  sorry

end injured_player_age_l1235_123524


namespace square_area_l1235_123574

theorem square_area (x y : ℝ) 
  (h1 : x = 20 ∧ y = 20)
  (h2 : x = 20 ∧ y = 5)
  (h3 : x = x ∧ y = 5)
  (h4 : x = x ∧ y = 20)
  : (∃ a : ℝ, a = 225) :=
sorry

end square_area_l1235_123574


namespace factor_poly_eq_factored_form_l1235_123542

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l1235_123542


namespace area_of_rectangle_at_stage_4_l1235_123539

def area_at_stage (n : ℕ) : ℕ :=
  let square_area := 16
  let initial_squares := 2
  let common_difference := 2
  let total_squares := initial_squares + common_difference * (n - 1)
  total_squares * square_area

theorem area_of_rectangle_at_stage_4 :
  area_at_stage 4 = 128 :=
by
  -- computation and transformations are omitted
  sorry

end area_of_rectangle_at_stage_4_l1235_123539


namespace transformed_center_coordinates_l1235_123562

theorem transformed_center_coordinates (S : (ℝ × ℝ)) (hS : S = (3, -4)) : 
  let reflected_S := (S.1, -S.2)
  let translated_S := (reflected_S.1, reflected_S.2 + 5)
  translated_S = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l1235_123562


namespace average_of_remaining_two_numbers_l1235_123523

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 4.60)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.8) :
  ((e + f) / 2) = 6.6 :=
sorry

end average_of_remaining_two_numbers_l1235_123523


namespace average_age_when_youngest_born_l1235_123507

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_youngest_age total_age_when_youngest_born : ℝ) 
  (h1 : n = 7) (h2 : avg_age = 30) (h3 : current_youngest_age = 8) (h4 : total_age_when_youngest_born = (n * avg_age - n * current_youngest_age)) : 
  total_age_when_youngest_born / n = 22 :=
by
  sorry

end average_age_when_youngest_born_l1235_123507


namespace unit_price_first_purchase_l1235_123504

theorem unit_price_first_purchase (x y : ℝ) (h1 : x * y = 500000) 
    (h2 : 1.4 * x * (y + 10000) = 770000) : x = 5 :=
by
  -- Proof details here
  sorry

end unit_price_first_purchase_l1235_123504


namespace tangent_line_slope_l1235_123589

/-- Given the line y = mx is tangent to the circle x^2 + y^2 - 4x + 2 = 0, 
    the slope m must be ±1. -/
theorem tangent_line_slope (m : ℝ) :
  (∃ x y : ℝ, y = m * x ∧ (x ^ 2 + y ^ 2 - 4 * x + 2 = 0)) →
  (m = 1 ∨ m = -1) :=
by
  sorry

end tangent_line_slope_l1235_123589


namespace value_after_increase_l1235_123512

def original_number : ℝ := 400
def percentage_increase : ℝ := 0.20

theorem value_after_increase : original_number * (1 + percentage_increase) = 480 := by
  sorry

end value_after_increase_l1235_123512


namespace complement_union_l1235_123596

-- Definitions of sets A and B based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0}

def B : Set ℝ := {x | x ≥ 1}

-- Theorem to prove the complement of the union of sets A and B within U
theorem complement_union (x : ℝ) : x ∉ (A ∪ B) ↔ (0 < x ∧ x < 1) := by
  sorry

end complement_union_l1235_123596


namespace exists_odd_k_l1235_123526

noncomputable def f (n : ℕ) : ℕ :=
sorry

theorem exists_odd_k : 
  (∀ m n : ℕ, f (m * n) = f m * f n) → 
  (∀ m n : ℕ, (m + n) ∣ (f m + f n)) → 
  ∃ k : ℕ, (k % 2 = 1) ∧ (∀ n : ℕ, f n = n ^ k) :=
sorry

end exists_odd_k_l1235_123526


namespace negation_example_l1235_123502

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := sorry

end negation_example_l1235_123502


namespace degree_of_p_is_unbounded_l1235_123540

theorem degree_of_p_is_unbounded (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval (x^2 - 1) = (p.eval x) * (p.eval (-x))) : False :=
sorry

end degree_of_p_is_unbounded_l1235_123540


namespace find_sum_squares_l1235_123535

variables (x y : ℝ)

theorem find_sum_squares (h1 : y + 4 = (x - 2)^2) (h2 : x + 4 = (y - 2)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 15 :=
sorry

end find_sum_squares_l1235_123535


namespace ratio_of_areas_l1235_123528

noncomputable def area_ratio (a : ℝ) : ℝ :=
  let side_triangle : ℝ := a
  let area_triangle : ℝ := (1 / 2) * side_triangle * side_triangle
  let height_rhombus : ℝ := side_triangle * Real.sin (Real.pi / 3)
  let area_rhombus : ℝ := height_rhombus * side_triangle
  area_rhombus / area_triangle

theorem ratio_of_areas (a : ℝ) (h : a > 0) : area_ratio a = 3 := by
  -- The proof would be here
  sorry

end ratio_of_areas_l1235_123528


namespace gcd_204_85_l1235_123538

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l1235_123538


namespace solve_for_x_l1235_123594

-- Define the problem
def equation (x : ℝ) : Prop := x + 2 * x + 12 = 500 - (3 * x + 4 * x)

-- State the theorem that we want to prove
theorem solve_for_x : ∃ (x : ℝ), equation x ∧ x = 48.8 := by
  sorry

end solve_for_x_l1235_123594


namespace fraction_value_l1235_123599

theorem fraction_value (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 :=
sorry

end fraction_value_l1235_123599


namespace greatest_value_2q_sub_r_l1235_123564

theorem greatest_value_2q_sub_r : 
  ∃ (q r : ℕ), 965 = 22 * q + r ∧ 2 * q - r = 67 := 
by 
  sorry

end greatest_value_2q_sub_r_l1235_123564


namespace smallest_w_correct_l1235_123510

-- Define the conditions
def is_factor (a b : ℕ) : Prop := ∃ k, a = b * k

-- Given conditions
def cond1 (w : ℕ) : Prop := is_factor (2^6) (1152 * w)
def cond2 (w : ℕ) : Prop := is_factor (3^4) (1152 * w)
def cond3 (w : ℕ) : Prop := is_factor (5^3) (1152 * w)
def cond4 (w : ℕ) : Prop := is_factor (7^2) (1152 * w)
def cond5 (w : ℕ) : Prop := is_factor (11) (1152 * w)
def is_positive (w : ℕ) : Prop := w > 0

-- The smallest possible value of w given all conditions
def smallest_w : ℕ := 16275

-- Proof statement
theorem smallest_w_correct : 
  ∀ (w : ℕ), cond1 w ∧ cond2 w ∧ cond3 w ∧ cond4 w ∧ cond5 w ∧ is_positive w ↔ w = smallest_w := sorry

end smallest_w_correct_l1235_123510


namespace remainder_1493827_div_4_l1235_123590

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := 
by
  sorry

end remainder_1493827_div_4_l1235_123590


namespace remainder_7_pow_63_mod_8_l1235_123537

theorem remainder_7_pow_63_mod_8 : 7^63 % 8 = 7 :=
by sorry

end remainder_7_pow_63_mod_8_l1235_123537


namespace find_third_number_l1235_123545

theorem find_third_number (x y z : ℝ) 
  (h1 : y = 3 * x - 7)
  (h2 : z = 2 * x + 2)
  (h3 : x + y + z = 168) : z = 60 :=
sorry

end find_third_number_l1235_123545


namespace problem_1_solution_set_problem_2_range_of_T_l1235_123515

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

theorem problem_1_solution_set :
  {x : ℝ | f x > 2} = {x | x < -5 ∨ 1 < x} :=
by 
  -- to be proven
  sorry

theorem problem_2_range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 2.5 * T - 1) →
  (T ≤ -3 ∨ T ≥ 0.5) :=
by
  -- to be proven
  sorry

end problem_1_solution_set_problem_2_range_of_T_l1235_123515


namespace find_a_l1235_123547

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end find_a_l1235_123547


namespace find_x_plus_y_l1235_123595

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3005) (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 3004 :=
by 
  sorry

end find_x_plus_y_l1235_123595


namespace number_of_correct_statements_l1235_123509

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end number_of_correct_statements_l1235_123509


namespace train_length_proof_l1235_123568

noncomputable def speed_km_per_hr : ℝ := 108
noncomputable def time_seconds : ℝ := 9
noncomputable def length_of_train : ℝ := 270
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

theorem train_length_proof : 
  (speed_km_per_hr * (km_to_m / hr_to_s) * time_seconds) = length_of_train :=
  by
  sorry

end train_length_proof_l1235_123568


namespace probability_top_card_heart_l1235_123588

def specially_designed_deck (n_cards n_ranks n_suits cards_per_suit : ℕ) : Prop :=
  n_cards = 60 ∧ n_ranks = 15 ∧ n_suits = 4 ∧ cards_per_suit = n_ranks

theorem probability_top_card_heart (n_cards n_ranks n_suits cards_per_suit : ℕ)
  (h_deck : specially_designed_deck n_cards n_ranks n_suits cards_per_suit) :
  (15 / 60 : ℝ) = 1 / 4 :=
by
  sorry

end probability_top_card_heart_l1235_123588


namespace symmetric_points_y_axis_l1235_123571

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : (a, 3) = (-2, 3)) (h₂ : (2, b) = (2, 3)) : (a + b) ^ 2015 = 1 := by
  sorry

end symmetric_points_y_axis_l1235_123571


namespace interest_rate_per_annum_l1235_123582
noncomputable def interest_rate_is_10 : ℝ := 10
theorem interest_rate_per_annum (P R : ℝ) : 
  (1200 * ((1 + R / 100)^2 - 1) - 1200 * R * 2 / 100 = 12) → P = 1200 → R = 10 := 
by sorry

end interest_rate_per_annum_l1235_123582


namespace number_of_books_in_box_l1235_123505

theorem number_of_books_in_box (total_weight : ℕ) (weight_per_book : ℕ) 
  (h1 : total_weight = 42) (h2 : weight_per_book = 3) : total_weight / weight_per_book = 14 :=
by sorry

end number_of_books_in_box_l1235_123505


namespace find_ck_l1235_123511

-- Definitions based on the conditions
def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

def geometric_sequence (r : ℕ) (n : ℕ) : ℕ :=
  r^(n - 1)

def combined_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

-- Given conditions
variable {d r k : ℕ}
variable (hd : combined_sequence d r (k-1) = 250)
variable (hk : combined_sequence d r (k+1) = 1250)

-- The theorem statement
theorem find_ck : combined_sequence d r k = 502 :=
  sorry

end find_ck_l1235_123511


namespace find_ratio_l1235_123548

-- Definition of the function
def f (x : ℝ) (a b: ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

-- Statement to be proved
theorem find_ratio (a b : ℝ) (h1: f 1 a b = 10) (h2 : (3 * 1^2 + 2 * a * 1 + b = 0)) : b = -a / 2 :=
by
  sorry

end find_ratio_l1235_123548


namespace shoes_remaining_l1235_123516

theorem shoes_remaining (monthly_goal : ℕ) (sold_last_week : ℕ) (sold_this_week : ℕ) (remaining_shoes : ℕ) :
  monthly_goal = 80 →
  sold_last_week = 27 →
  sold_this_week = 12 →
  remaining_shoes = monthly_goal - sold_last_week - sold_this_week →
  remaining_shoes = 41 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end shoes_remaining_l1235_123516


namespace calc_fraction_l1235_123518

variable {x y : ℝ}

theorem calc_fraction (h : x + y = x * y - 1) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 1 - 1 / (x * y) := 
by 
  sorry

end calc_fraction_l1235_123518


namespace price_of_brand_y_pen_l1235_123577

-- Definitions based on the conditions
def num_brand_x_pens : ℕ := 8
def price_per_brand_x_pen : ℝ := 4.0
def total_spent : ℝ := 40.0
def total_pens : ℕ := 12

-- price of brand Y that needs to be proven
def price_per_brand_y_pen : ℝ := 2.0

-- Proof statement
theorem price_of_brand_y_pen :
  let num_brand_y_pens := total_pens - num_brand_x_pens
  let spent_on_brand_x_pens := num_brand_x_pens * price_per_brand_x_pen
  let spent_on_brand_y_pens := total_spent - spent_on_brand_x_pens
  spent_on_brand_y_pens / num_brand_y_pens = price_per_brand_y_pen :=
by
  sorry

end price_of_brand_y_pen_l1235_123577


namespace yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l1235_123554

-- Defining "Yazhong point"
def yazhong (A B M : ℝ) : Prop := abs (M - A) = abs (M - B)

-- Problem 1
theorem yazhong_point_1 {A B M : ℝ} (hA : A = -5) (hB : B = 1) (hM : yazhong A B M) : M = -2 :=
sorry

-- Problem 2
theorem yazhong_point_2 {A B M : ℝ} (hM : M = 2) (hAB : B - A = 9) (h_order : A < B) (hY : yazhong A B M) :
  (A = -5/2) ∧ (B = 13/2) :=
sorry

-- Problem 3 Part ①
theorem yazhong_point_3_part1 (A : ℝ) (B : ℝ) (m : ℤ) 
  (hA : A = -6) (hB_range : -4 ≤ B ∧ B ≤ -2) (hM : yazhong A B m) : 
  m = -5 ∨ m = -4 :=
sorry

-- Problem 3 Part ②
theorem yazhong_point_3_part2 (C D : ℝ) (n : ℤ)
  (hC : C = -4) (hD : D = -2) (hM : yazhong (-6) (C + D + 2 * n) 0) : 
  8 ≤ n ∧ n ≤ 10 :=
sorry

end yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l1235_123554


namespace circle_sector_cones_sum_radii_l1235_123520

theorem circle_sector_cones_sum_radii :
  let r := 5
  let a₁ := 1
  let a₂ := 2
  let a₃ := 3
  let total_area := π * r * r
  let θ₁ := (a₁ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₂ := (a₂ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₃ := (a₃ / (a₁ + a₂ + a₃)) * 2 * π
  let r₁ := (a₁ / (a₁ + a₂ + a₃)) * r
  let r₂ := (a₂ / (a₁ + a₂ + a₃)) * r
  let r₃ := (a₃ / (a₁ + a₂ + a₃)) * r
  r₁ + r₂ + r₃ = 5 :=
by {
  sorry
}

end circle_sector_cones_sum_radii_l1235_123520


namespace total_cans_given_away_l1235_123560

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l1235_123560


namespace baby_guppies_l1235_123549

theorem baby_guppies (x : ℕ) (h1 : 7 + x + 9 = 52) : x = 36 :=
by
  sorry

end baby_guppies_l1235_123549


namespace animal_eyes_count_l1235_123543

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end animal_eyes_count_l1235_123543


namespace base_three_to_decimal_l1235_123557

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end base_three_to_decimal_l1235_123557


namespace max_volume_pyramid_l1235_123514

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end max_volume_pyramid_l1235_123514


namespace find_triangle_value_l1235_123578

variables (triangle q r : ℝ)
variables (h1 : triangle + q = 75) (h2 : triangle + q + r = 138) (h3 : r = q / 3)

theorem find_triangle_value : triangle = -114 :=
by
  sorry

end find_triangle_value_l1235_123578


namespace time_for_Q_to_finish_job_alone_l1235_123563

theorem time_for_Q_to_finish_job_alone (T_Q : ℝ) 
  (h1 : 0 < T_Q)
  (rate_P : ℝ := 1 / 4) 
  (rate_Q : ℝ := 1 / T_Q)
  (combined_work_rate : ℝ := 3 * (rate_P + rate_Q))
  (remaining_work : ℝ := 0.1) -- 0.4 * rate_P
  (total_work_done : ℝ := 0.9) -- 1 - remaining_work
  (h2 : combined_work_rate = total_work_done) : T_Q = 20 :=
by sorry

end time_for_Q_to_finish_job_alone_l1235_123563


namespace gallons_bought_l1235_123570

variable (total_needed : ℕ) (existing_paint : ℕ) (needed_more : ℕ)

theorem gallons_bought (H : total_needed = 70) (H1 : existing_paint = 36) (H2 : needed_more = 11) : 
  total_needed - existing_paint - needed_more = 23 := 
sorry

end gallons_bought_l1235_123570


namespace imaginary_part_of_z_l1235_123551

open Complex

-- Define the context
variables (z : ℂ) (a b : ℂ)

-- Define the condition
def condition := (1 - 2*I) * z = 5 * I

-- Lean 4 statement to prove the imaginary part of z 
theorem imaginary_part_of_z (h : condition z) : z.im = 1 :=
sorry

end imaginary_part_of_z_l1235_123551


namespace exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l1235_123583

theorem exists_n_such_that_an_is_cube_and_bn_is_fifth_power
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), n ≥ 1 ∧ (∃ k : ℤ, a * n = k^3) ∧ (∃ l : ℤ, b * n = l^5) := 
by
  sorry

end exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l1235_123583


namespace cubic_difference_pos_l1235_123585

theorem cubic_difference_pos {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cubic_difference_pos_l1235_123585


namespace total_spent_on_concert_tickets_l1235_123519

theorem total_spent_on_concert_tickets : 
  let price_per_ticket := 4
  let number_of_tickets := 3 + 5
  let discount_threshold := 5
  let discount_rate := 0.10
  let service_fee_per_ticket := 2
  let initial_cost := number_of_tickets * price_per_ticket
  let discount := if number_of_tickets > discount_threshold then discount_rate * initial_cost else 0
  let discounted_cost := initial_cost - discount
  let service_fee := number_of_tickets * service_fee_per_ticket
  let total_cost := discounted_cost + service_fee
  total_cost = 44.8 :=
by
  sorry

end total_spent_on_concert_tickets_l1235_123519


namespace complementary_angles_decrease_percent_l1235_123536

theorem complementary_angles_decrease_percent
    (a b : ℝ) 
    (h1 : a + b = 90) 
    (h2 : a / b = 3 / 7) 
    (h3 : new_a = a * 1.15) 
    (h4 : new_a + new_b = 90) : 
    (new_b / b * 100) = 93.57 := 
sorry

end complementary_angles_decrease_percent_l1235_123536


namespace bernardo_wins_at_5_l1235_123576

theorem bernardo_wins_at_5 :
  (∀ N : ℕ, (16 * N + 900 < 1000) → (920 ≤ 16 * N + 840) → N ≥ 5)
    ∧ (5 < 10 ∧ 16 * 5 + 900 < 1000 ∧ 920 ≤ 16 * 5 + 840) := by
{
  sorry
}

end bernardo_wins_at_5_l1235_123576


namespace brick_length_correct_l1235_123525

-- Define the constants
def courtyard_length_meters : ℝ := 25
def courtyard_width_meters : ℝ := 18
def courtyard_area_meters : ℝ := courtyard_length_meters * courtyard_width_meters
def bricks_number : ℕ := 22500
def brick_width_cm : ℕ := 10

-- We want to prove the length of each brick
def brick_length_cm : ℕ := 20

-- Convert courtyard area to square centimeters
def courtyard_area_cm : ℝ := courtyard_area_meters * 10000

-- Define the proof statement
theorem brick_length_correct :
  courtyard_area_cm = (brick_length_cm * brick_width_cm) * bricks_number :=
by
  sorry

end brick_length_correct_l1235_123525


namespace number_of_persons_in_group_l1235_123503

theorem number_of_persons_in_group 
    (n : ℕ)
    (h1 : average_age_before - average_age_after = 3)
    (h2 : person_replaced_age = 40)
    (h3 : new_person_age = 10)
    (h4 : total_age_decrease = 3 * n):
  n = 10 := 
sorry

end number_of_persons_in_group_l1235_123503


namespace infinite_chain_resistance_l1235_123569

noncomputable def resistance_of_infinite_chain (R₀ : ℝ) : ℝ :=
  (R₀ * (1 + Real.sqrt 5)) / 2

theorem infinite_chain_resistance : resistance_of_infinite_chain 10 = 5 + 5 * Real.sqrt 5 :=
by
  sorry

end infinite_chain_resistance_l1235_123569


namespace product_of_N_l1235_123555

theorem product_of_N (M L : ℝ) (N : ℝ) 
  (h1 : M = L + N) 
  (h2 : ∀ M4 L4 : ℝ, M4 = M - 7 → L4 = L + 5 → |M4 - L4| = 4) :
  N = 16 ∨ N = 8 ∧ (16 * 8 = 128) := 
by 
  sorry

end product_of_N_l1235_123555


namespace distance_between_trees_l1235_123561

theorem distance_between_trees
  (num_trees : ℕ)
  (length_of_yard : ℝ)
  (one_tree_at_each_end : True)
  (h1 : num_trees = 26)
  (h2 : length_of_yard = 400) :
  length_of_yard / (num_trees - 1) = 16 :=
by
  sorry

end distance_between_trees_l1235_123561


namespace kolya_start_time_l1235_123581

-- Definitions of conditions as per the initial problem statement
def angle_moved_by_minute_hand (x : ℝ) : ℝ := 6 * x
def angle_moved_by_hour_hand (x : ℝ) : ℝ := 30 + 0.5 * x

theorem kolya_start_time (x : ℝ) :
  (angle_moved_by_minute_hand x = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) ∨
  (angle_moved_by_minute_hand x - 180 = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) :=
sorry

end kolya_start_time_l1235_123581


namespace regular_polygon_sides_l1235_123532

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l1235_123532


namespace find_number_l1235_123527

theorem find_number (n : ℕ) (h : Nat.factorial 4 / Nat.factorial (4 - n) = 24) : n = 3 :=
by
  sorry

end find_number_l1235_123527


namespace find_k_l1235_123565

theorem find_k (x y k : ℝ) (hx1 : x - 4 * y + 3 ≤ 0) (hx2 : 3 * x + 5 * y - 25 ≤ 0) (hx3 : x ≥ 1)
  (hmax : ∃ (z : ℝ), z = 12 ∧ z = k * x + y) (hmin : ∃ (z : ℝ), z = 3 ∧ z = k * x + y) :
  k = 2 :=
sorry

end find_k_l1235_123565


namespace largest_among_abc_l1235_123587

variable {a b c : ℝ}

theorem largest_among_abc 
  (hn1 : a < 0) 
  (hn2 : b < 0) 
  (hn3 : c < 0) 
  (h : (c / (a + b)) < (a / (b + c)) ∧ (a / (b + c)) < (b / (c + a))) : c > a ∧ c > b :=
by
  sorry

end largest_among_abc_l1235_123587


namespace chests_content_l1235_123598

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end chests_content_l1235_123598


namespace rectangle_area_correct_l1235_123593

theorem rectangle_area_correct (l r s : ℝ) (b : ℝ := 10) (h1 : l = (1 / 4) * r) (h2 : r = s) (h3 : s^2 = 1225) :
  l * b = 87.5 :=
by
  sorry

end rectangle_area_correct_l1235_123593


namespace total_spent_l1235_123546

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l1235_123546


namespace probability_no_rain_five_days_probability_drought_alert_approx_l1235_123513

theorem probability_no_rain_five_days (p : ℚ) (h : p = 1/3) :
  (p ^ 5) = 1 / 243 :=
by
  -- Add assumptions and proceed
  sorry

theorem probability_drought_alert_approx (p : ℚ) (h : p = 1/3) :
  4 * (p ^ 2) = 4 / 9 :=
by
  -- Add assumptions and proceed
  sorry

end probability_no_rain_five_days_probability_drought_alert_approx_l1235_123513


namespace necessary_and_sufficient_condition_l1235_123529

theorem necessary_and_sufficient_condition {a : ℝ} :
    (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end necessary_and_sufficient_condition_l1235_123529


namespace village_connection_possible_l1235_123591

variable (V : Type) -- Type of villages
variable (Villages : List V) -- List of 26 villages
variable (connected_by_tractor connected_by_train : V → V → Prop) -- Connections

-- Define the hypothesis
variable (bidirectional_connections : ∀ (v1 v2 : V), v1 ≠ v2 → (connected_by_tractor v1 v2 ∨ connected_by_train v1 v2))

-- Main theorem statement
theorem village_connection_possible :
  ∃ (mode : V → V → Prop), (∀ v1 v2 : V, v1 ≠ v2 → v1 ∈ Villages → v2 ∈ Villages → mode v1 v2) ∧
  (∀ v1 v2 : V, v1 ∈ Villages → v2 ∈ Villages → ∃ (path : List (V × V)), (∀ edge ∈ path, mode edge.fst edge.snd) ∧ path ≠ []) :=
by
  sorry

end village_connection_possible_l1235_123591
