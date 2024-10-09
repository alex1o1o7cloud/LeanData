import Mathlib

namespace inequality_example_l1207_120778

theorem inequality_example (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 :=
sorry

end inequality_example_l1207_120778


namespace fraction_of_boxes_loaded_by_day_crew_l1207_120768

theorem fraction_of_boxes_loaded_by_day_crew
    (dayCrewBoxesPerWorker : ℚ)
    (dayCrewWorkers : ℚ)
    (nightCrewBoxesPerWorker : ℚ := (3 / 4) * dayCrewBoxesPerWorker)
    (nightCrewWorkers : ℚ := (3 / 4) * dayCrewWorkers) :
    (dayCrewBoxesPerWorker * dayCrewWorkers) / ((dayCrewBoxesPerWorker * dayCrewWorkers) + (nightCrewBoxesPerWorker * nightCrewWorkers)) = 16 / 25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l1207_120768


namespace range_of_b_l1207_120700

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b) (h2 : a + b < 1) (h3 : 2 ≤ a - b) (h4 : a - b < 3) :
  -3 / 2 < b ∧ b < -1 / 2 :=
by
  sorry

end range_of_b_l1207_120700


namespace units_digit_of_m_squared_plus_two_to_the_m_is_seven_l1207_120733

def m := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_the_m_is_seven :
  (m^2 + 2^m) % 10 = 7 := by
sorry

end units_digit_of_m_squared_plus_two_to_the_m_is_seven_l1207_120733


namespace polynomial_divisible_iff_l1207_120702

theorem polynomial_divisible_iff (a b : ℚ) : 
  ((a + b) * 1^5 + (a * b) * 1^2 + 1 = 0) ∧ 
  ((a + b) * 2^5 + (a * b) * 2^2 + 1 = 0) ↔ 
  (a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1) := 
by 
  sorry

end polynomial_divisible_iff_l1207_120702


namespace difference_in_spending_l1207_120760

-- Condition: original prices and discounts
def original_price_candy_bar : ℝ := 6
def discount_candy_bar : ℝ := 0.25
def original_price_chocolate : ℝ := 3
def discount_chocolate : ℝ := 0.10

-- The theorem to prove
theorem difference_in_spending : 
  (original_price_candy_bar * (1 - discount_candy_bar) - original_price_chocolate * (1 - discount_chocolate)) = 1.80 :=
by
  sorry

end difference_in_spending_l1207_120760


namespace initial_dogs_count_is_36_l1207_120759

-- Conditions
def initial_cats := 29
def adopted_dogs := 20
def additional_cats := 12
def total_pets := 57

-- Calculate total cats
def total_cats := initial_cats + additional_cats

-- Calculate initial dogs
def initial_dogs (initial_dogs : ℕ) : Prop :=
(initial_dogs - adopted_dogs) + total_cats = total_pets

-- Prove that initial dogs (D) is 36
theorem initial_dogs_count_is_36 : initial_dogs 36 :=
by
-- Here should contain the proof which is omitted
sorry

end initial_dogs_count_is_36_l1207_120759


namespace sam_average_speed_l1207_120740

theorem sam_average_speed :
  let total_time := 7 -- total time from 7 a.m. to 2 p.m.
  let rest_time := 1 -- rest period from 9 a.m. to 10 a.m.
  let effective_time := total_time - rest_time
  let total_distance := 200 -- total miles covered
  let avg_speed := total_distance / effective_time
  avg_speed = 33.3 :=
sorry

end sam_average_speed_l1207_120740


namespace original_radius_new_perimeter_l1207_120763

variable (r : ℝ)

theorem original_radius_new_perimeter (h : (π * (r + 5)^2 = 4 * π * r^2)) :
  r = 5 ∧ 2 * π * (r + 5) = 20 * π :=
by
  sorry

end original_radius_new_perimeter_l1207_120763


namespace no_triangle_sides_exist_l1207_120756

theorem no_triangle_sides_exist (x y z : ℝ) (h_triangle_sides : x > 0 ∧ y > 0 ∧ z > 0)
  (h_triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
sorry

end no_triangle_sides_exist_l1207_120756


namespace gwen_money_received_from_dad_l1207_120797

variables (D : ℕ)

-- Conditions
def mom_received := 8
def mom_more_than_dad := 3

-- Question and required proof
theorem gwen_money_received_from_dad : 
  (mom_received = D + mom_more_than_dad) -> D = 5 := 
by
  sorry

end gwen_money_received_from_dad_l1207_120797


namespace determine_value_of_product_l1207_120725

theorem determine_value_of_product (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := 
by 
  sorry

end determine_value_of_product_l1207_120725


namespace evaluate_expression_l1207_120753

theorem evaluate_expression : 
  (1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))) = 5 / 7 :=
by
  sorry

end evaluate_expression_l1207_120753


namespace work_done_l1207_120737

noncomputable def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

theorem work_done (W : ℝ) (h : W = ∫ x in (1:ℝ)..(5:ℝ), F x) : W = 112 :=
by sorry

end work_done_l1207_120737


namespace value_of_x_l1207_120714

theorem value_of_x : 
  ∀ (x y z : ℕ), 
  (x = y / 3) ∧ 
  (y = z / 6) ∧ 
  (z = 72) → 
  x = 4 :=
by
  intros x y z h
  have h1 : y = z / 6 := h.2.1
  have h2 : z = 72 := h.2.2
  have h3 : x = y / 3 := h.1
  sorry

end value_of_x_l1207_120714


namespace is_factorization_l1207_120793

-- Define the conditions
def A_transformation : Prop := (∀ x : ℝ, (x + 1) * (x - 1) = x ^ 2 - 1)
def B_transformation : Prop := (∀ m : ℝ, m ^ 2 + m - 4 = (m + 3) * (m - 2) + 2)
def C_transformation : Prop := (∀ x : ℝ, x ^ 2 + 2 * x = x * (x + 2))
def D_transformation : Prop := (∀ x : ℝ, 2 * x ^ 2 + 2 * x = 2 * x ^ 2 * (1 + (1 / x)))

-- The goal is to prove that transformation C is a factorization
theorem is_factorization : C_transformation :=
by
  sorry

end is_factorization_l1207_120793


namespace range_of_a_l1207_120728

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ↔ a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l1207_120728


namespace certain_number_eq_1000_l1207_120769

theorem certain_number_eq_1000 (x : ℝ) (h : 3500 - x / 20.50 = 3451.2195121951218) : x = 1000 := 
by
  sorry

end certain_number_eq_1000_l1207_120769


namespace evaluate_ratio_l1207_120727

theorem evaluate_ratio : (2^3002 * 3^3005 / 6^3003 : ℚ) = 9 / 2 := 
sorry

end evaluate_ratio_l1207_120727


namespace solve_for_x_l1207_120795

theorem solve_for_x (x : ℝ) : (|2 * x + 8| = 4 - 3 * x) → x = -4 / 5 :=
  sorry

end solve_for_x_l1207_120795


namespace common_difference_arithmetic_sequence_l1207_120742

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end common_difference_arithmetic_sequence_l1207_120742


namespace probability_of_one_fork_one_spoon_one_knife_l1207_120750

theorem probability_of_one_fork_one_spoon_one_knife 
  (num_forks : ℕ) (num_spoons : ℕ) (num_knives : ℕ) (total_pieces : ℕ)
  (h_forks : num_forks = 7) (h_spoons : num_spoons = 8) (h_knives : num_knives = 5)
  (h_total : total_pieces = num_forks + num_spoons + num_knives) :
  (∃ (prob : ℚ), prob = 14 / 57) :=
by
  sorry

end probability_of_one_fork_one_spoon_one_knife_l1207_120750


namespace solution_set_inequality_l1207_120799

theorem solution_set_inequality {x : ℝ} : 
  ((x - 1)^2 < 1) ↔ (0 < x ∧ x < 2) := by
  sorry

end solution_set_inequality_l1207_120799


namespace value_range_of_func_l1207_120709

-- Define the function y = x^2 - 4x + 6 for x in the interval [1, 4]
def func (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem value_range_of_func : 
  ∀ y, ∃ x, (1 ≤ x ∧ x ≤ 4) ∧ y = func x ↔ 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end value_range_of_func_l1207_120709


namespace inequality_proof_l1207_120781

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : 0 < c)
  : a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c :=
sorry

end inequality_proof_l1207_120781


namespace baby_whales_on_second_trip_l1207_120708

def iwishmael_whales_problem : Prop :=
  let male1 := 28
  let female1 := 2 * male1
  let male3 := male1 / 2
  let female3 := female1
  let total_whales := 178
  let total_without_babies := (male1 + female1) + (male3 + female3)
  total_whales - total_without_babies = 24

theorem baby_whales_on_second_trip : iwishmael_whales_problem :=
  by
  sorry

end baby_whales_on_second_trip_l1207_120708


namespace evaluate_expression_l1207_120752

open BigOperators

theorem evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 1 → 2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l1207_120752


namespace total_photos_l1207_120706

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l1207_120706


namespace second_exponent_base_ends_in_1_l1207_120729

theorem second_exponent_base_ends_in_1 
  (x : ℕ) 
  (h : ((1023 ^ 3923) + (x ^ 3921)) % 10 = 8) : 
  x % 10 = 1 := 
by sorry

end second_exponent_base_ends_in_1_l1207_120729


namespace base_k_sum_l1207_120715

theorem base_k_sum (k : ℕ) (t : ℕ) (h1 : (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5)
    (h2 : t = (k + 3) + (k + 4) + (k + 7)) :
    t = 50 := sorry

end base_k_sum_l1207_120715


namespace twenty_percent_greater_than_40_l1207_120782

theorem twenty_percent_greater_than_40 (x : ℝ) (h : x = 40 + 0.2 * 40) : x = 48 := by
sorry

end twenty_percent_greater_than_40_l1207_120782


namespace number_of_moles_of_NaCl_l1207_120785

theorem number_of_moles_of_NaCl
  (moles_NaOH : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : 2 * moles_NaOH + moles_Cl2 = 2 * moles_NaOH + 1) :
  2 * moles_Cl2 = 2 := by 
  sorry

end number_of_moles_of_NaCl_l1207_120785


namespace root_in_interval_implies_a_in_range_l1207_120745

theorem root_in_interval_implies_a_in_range {a : ℝ} (h : ∃ x : ℝ, x ≤ 1 ∧ 2^x - a^2 - a = 0) : 0 < a ∧ a ≤ 1 := sorry

end root_in_interval_implies_a_in_range_l1207_120745


namespace b_cong_zero_l1207_120766

theorem b_cong_zero (a b c m : ℤ) (h₀ : 1 < m) (h : ∀ (n : ℕ), (a ^ n + b * n + c) % m = 0) : b % m = 0 :=
  sorry

end b_cong_zero_l1207_120766


namespace total_carrots_l1207_120730

def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11

theorem total_carrots : Joan_carrots + Jessica_carrots = 40 := by
  sorry

end total_carrots_l1207_120730


namespace geometric_sequence_a_equals_minus_four_l1207_120713

theorem geometric_sequence_a_equals_minus_four (a : ℝ) 
(h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : a = -4 :=
sorry

end geometric_sequence_a_equals_minus_four_l1207_120713


namespace students_no_A_l1207_120776

def total_students : Nat := 40
def students_A_chemistry : Nat := 10
def students_A_physics : Nat := 18
def students_A_both : Nat := 6

theorem students_no_A : (total_students - (students_A_chemistry + students_A_physics - students_A_both)) = 18 :=
by
  sorry

end students_no_A_l1207_120776


namespace field_length_l1207_120734

theorem field_length 
  (w l : ℝ)
  (pond_area : ℝ := 25)
  (h1 : l = 2 * w)
  (h2 : pond_area = 25)
  (h3 : pond_area = (1 / 8) * (l * w)) :
  l = 20 :=
by
  sorry

end field_length_l1207_120734


namespace problem_value_of_m_l1207_120741

theorem problem_value_of_m (m : ℝ)
  (h1 : (m + 1) * x ^ (m ^ 2 - 3) = y)
  (h2 : m ^ 2 - 3 = 1)
  (h3 : m + 1 < 0) : 
  m = -2 := 
  sorry

end problem_value_of_m_l1207_120741


namespace credit_limit_l1207_120735

theorem credit_limit (paid_tuesday : ℕ) (paid_thursday : ℕ) (remaining_payment : ℕ) (full_payment : ℕ) 
  (h1 : paid_tuesday = 15) 
  (h2 : paid_thursday = 23) 
  (h3 : remaining_payment = 62) 
  (h4 : full_payment = paid_tuesday + paid_thursday + remaining_payment) : 
  full_payment = 100 := 
by
  sorry

end credit_limit_l1207_120735


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l1207_120717

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) :=
sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l1207_120717


namespace intersection_A_B_l1207_120704

def set_A : Set ℕ := {x | x^2 - 2 * x = 0}
def set_B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : set_A ∩ set_B = {0, 2} := 
by sorry

end intersection_A_B_l1207_120704


namespace ratio_larger_to_smaller_l1207_120720

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
  a / b

theorem ratio_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : ratio_of_numbers a b = 9 / 5 := 
  sorry

end ratio_larger_to_smaller_l1207_120720


namespace sourav_distance_l1207_120765

def D (t : ℕ) : ℕ := 20 * t

theorem sourav_distance :
  ∀ (t : ℕ), 20 * t = 25 * (t - 1) → 20 * t = 100 :=
by
  intros t h
  sorry

end sourav_distance_l1207_120765


namespace boys_more_than_girls_l1207_120796

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l1207_120796


namespace initial_men_count_l1207_120770

theorem initial_men_count (M : ℕ) (A : ℕ) (H1 : 58 - (20 + 22) = 2 * M) : M = 8 :=
by
  sorry

end initial_men_count_l1207_120770


namespace waiter_earned_in_tips_l1207_120783

def waiter_customers := 7
def customers_didnt_tip := 5
def tip_per_customer := 3
def customers_tipped := waiter_customers - customers_didnt_tip
def total_earnings := customers_tipped * tip_per_customer

theorem waiter_earned_in_tips : total_earnings = 6 :=
by
  sorry

end waiter_earned_in_tips_l1207_120783


namespace zero_integers_satisfy_conditions_l1207_120744

noncomputable def satisfies_conditions (n : ℤ) : Prop :=
  ∃ k : ℤ, n * (25 - n) = k^2 * (25 - n)^2 ∧ n % 3 = 0

theorem zero_integers_satisfy_conditions :
  (∃ n : ℤ, satisfies_conditions n) → False := by
  sorry

end zero_integers_satisfy_conditions_l1207_120744


namespace find_two_digit_number_l1207_120718

theorem find_two_digit_number : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 10^6 ≤ n^3 ∧ n^3 < 10^7 ∧ 101010 * x + 1 = n^3 ∧ x = 93) := 
 by
  sorry

end find_two_digit_number_l1207_120718


namespace problem_l1207_120736

noncomputable def F (x : ℝ) : ℝ :=
  (1 + x^2 - x^3) / (2 * x * (1 - x))

theorem problem (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  F x + F ((x - 1) / x) = 1 + x :=
by
  sorry

end problem_l1207_120736


namespace daughter_age_l1207_120716

-- Define the conditions and the question as a theorem
theorem daughter_age (D F : ℕ) (h1 : F = 3 * D) (h2 : F + 12 = 2 * (D + 12)) : D = 12 :=
by
  -- We need to provide a proof or placeholder for now
  sorry

end daughter_age_l1207_120716


namespace total_points_l1207_120743

def jon_points (sam_points : ℕ) : ℕ := 2 * sam_points + 3
def sam_points (alex_points : ℕ) : ℕ := alex_points / 2
def jack_points (jon_points : ℕ) : ℕ := jon_points + 5
def tom_points (jon_points jack_points : ℕ) : ℕ := jon_points + jack_points - 4
def alex_points : ℕ := 18

theorem total_points : jon_points (sam_points alex_points) + 
                       jack_points (jon_points (sam_points alex_points)) + 
                       tom_points (jon_points (sam_points alex_points)) 
                       (jack_points (jon_points (sam_points alex_points))) + 
                       sam_points alex_points + 
                       alex_points = 117 :=
by sorry

end total_points_l1207_120743


namespace percentage_increase_l1207_120711

noncomputable def price_increase (d new_price : ℝ) : ℝ :=
  ((new_price - d) / d) * 100

theorem percentage_increase 
  (d new_price : ℝ)
  (h1 : 2 * d = 585)
  (h2 : new_price = 351) :
  price_increase d new_price = 20 :=
by
  sorry

end percentage_increase_l1207_120711


namespace line_intersections_with_parabola_l1207_120789

theorem line_intersections_with_parabola :
  ∃! (L : ℝ → ℝ) (l_count : ℕ),  
    l_count = 3 ∧
    (∀ x : ℝ, (L x) ∈ {x | (L 0 = 2) ∧ ∃ y, y * y = 8 * x ∧ L x = y}) := sorry

end line_intersections_with_parabola_l1207_120789


namespace supremum_neg_frac_bound_l1207_120777

noncomputable def supremum_neg_frac (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_neg_frac_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  supremum_neg_frac a b ≤ - 9 / 2 :=
sorry

end supremum_neg_frac_bound_l1207_120777


namespace right_triangle_345_l1207_120775

theorem right_triangle_345 :
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  -- Here, we should construct the proof later
  sorry
}

end right_triangle_345_l1207_120775


namespace third_month_sale_l1207_120739

theorem third_month_sale
  (avg_sale : ℕ)
  (num_months : ℕ)
  (sales : List ℕ)
  (sixth_month_sale : ℕ)
  (total_sales_req : ℕ) :
  avg_sale = 6500 →
  num_months = 6 →
  sales = [6435, 6927, 7230, 6562] →
  sixth_month_sale = 4991 →
  total_sales_req = avg_sale * num_months →
  total_sales_req - (sales.sum + sixth_month_sale) = 6855 := by
  sorry

end third_month_sale_l1207_120739


namespace max_a_avoiding_lattice_points_l1207_120748

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points_l1207_120748


namespace inequality_proof_l1207_120780

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) : ab < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by
  sorry

end inequality_proof_l1207_120780


namespace fewest_four_dollar_frisbees_l1207_120719

-- Definitions based on the conditions
variables (x y : ℕ) -- The numbers of $3 and $4 frisbees, respectively.
def total_frisbees (x y : ℕ) : Prop := x + y = 60
def total_receipts (x y : ℕ) : Prop := 3 * x + 4 * y = 204

-- The statement to prove
theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : total_frisbees x y) (h2 : total_receipts x y) : y = 24 :=
sorry

end fewest_four_dollar_frisbees_l1207_120719


namespace representation_of_1_l1207_120746

theorem representation_of_1 (x y z : ℕ) (h : 1 = 1/x + 1/y + 1/z) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end representation_of_1_l1207_120746


namespace fifth_grade_soccer_students_l1207_120754

variable (T B Gnp GP S : ℕ)
variable (p : ℝ)

theorem fifth_grade_soccer_students
  (hT : T = 420)
  (hB : B = 296)
  (hp_percent : p = 86 / 100)
  (hGnp : Gnp = 89)
  (hpercent_boys_playing_soccer : (1 - p) * S = GP)
  (hpercent_girls_playing_soccer : GP = 35) :
  S = 250 := by
  sorry

end fifth_grade_soccer_students_l1207_120754


namespace remainder_of_n_div_4_is_1_l1207_120747

noncomputable def n : ℕ := sorry  -- We declare n as a noncomputable natural number to proceed with the proof complexity

theorem remainder_of_n_div_4_is_1 (n : ℕ) (h : (2 * n) % 4 = 2) : n % 4 = 1 :=
by
  sorry  -- skip the proof

end remainder_of_n_div_4_is_1_l1207_120747


namespace savings_after_purchase_l1207_120791

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end savings_after_purchase_l1207_120791


namespace total_packs_of_groceries_l1207_120751

-- Definitions based on conditions
def packs_of_cookies : Nat := 4
def packs_of_cake : Nat := 22
def packs_of_chocolate : Nat := 16

-- The proof statement
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake + packs_of_chocolate = 42 :=
by
  -- Proof skipped using sorry
  sorry

end total_packs_of_groceries_l1207_120751


namespace find_m_l1207_120792

-- Circle equation: x^2 + y^2 + 2x - 6y + 1 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y + 1 = 0

-- Line equation: x + m * y + 4 = 0
def line_eq (x y m : ℝ) : Prop := x + m * y + 4 = 0

-- Prove that the value of m such that the center of the circle lies on the line is -1
theorem find_m (m : ℝ) : 
  (∃ x y : ℝ, circle_eq x y ∧ (x, y) = (-1, 3) ∧ line_eq x y m) → m = -1 :=
by {
  sorry
}

end find_m_l1207_120792


namespace emily_necklaces_l1207_120779

theorem emily_necklaces (n beads_per_necklace total_beads : ℕ) (h1 : beads_per_necklace = 8) (h2 : total_beads = 16) : n = total_beads / beads_per_necklace → n = 2 :=
by sorry

end emily_necklaces_l1207_120779


namespace find_x_l1207_120772

noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := 1 / a
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem find_x (a₀ : a = 0.3010) : 
  ∃ x : ℝ, (log2_5 ^ 2 - a * log2_5 + x * b = 0) → 
  x = (log2_5 ^ 2 * 0.3010) :=
by
  sorry

end find_x_l1207_120772


namespace value_is_correct_l1207_120764

-- Define the number
def initial_number : ℝ := 4400

-- Define the value calculation in Lean
def value : ℝ := 0.15 * (0.30 * (0.50 * initial_number))

-- The theorem statement
theorem value_is_correct : value = 99 := by
  sorry

end value_is_correct_l1207_120764


namespace least_number_to_divisible_by_11_l1207_120774

theorem least_number_to_divisible_by_11 (n : ℕ) (h : n = 11002) : ∃ k : ℕ, (n + k) % 11 = 0 ∧ ∀ m : ℕ, (n + m) % 11 = 0 → m ≥ k :=
by
  sorry

end least_number_to_divisible_by_11_l1207_120774


namespace velocity_at_t_10_time_to_reach_max_height_max_height_l1207_120771

-- Define the height function H(t)
def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

-- Define the velocity function v(t) as the derivative of H(t)
def v (t : ℝ) : ℝ := 200 - 9.8 * t

-- Theorem: The velocity of the body at t = 10 seconds
theorem velocity_at_t_10 : v 10 = 102 := by
  sorry

-- Theorem: The time to reach maximum height
theorem time_to_reach_max_height : (∃ t : ℝ, v t = 0 ∧ t = 200 / 9.8) := by
  sorry

-- Theorem: The maximum height the body will reach
theorem max_height : H (200 / 9.8) = 2040.425 := by
  sorry

end velocity_at_t_10_time_to_reach_max_height_max_height_l1207_120771


namespace units_digit_periodic_10_l1207_120762

theorem units_digit_periodic_10:
  ∀ n: ℕ, (n * (n + 1) * (n + 2)) % 10 = ((n + 10) * (n + 11) * (n + 12)) % 10 :=
by
  sorry

end units_digit_periodic_10_l1207_120762


namespace daughter_age_is_10_l1207_120726

variable (D : ℕ)

-- Conditions
def father_current_age (D : ℕ) : ℕ := 4 * D
def father_age_in_20_years (D : ℕ) : ℕ := father_current_age D + 20
def daughter_age_in_20_years (D : ℕ) : ℕ := D + 20

-- Theorem statement
theorem daughter_age_is_10 :
  father_current_age D = 40 →
  father_age_in_20_years D = 2 * daughter_age_in_20_years D →
  D = 10 :=
by
  -- Here would be the proof steps to show that D = 10 given the conditions
  sorry

end daughter_age_is_10_l1207_120726


namespace rook_placement_5x5_l1207_120758

theorem rook_placement_5x5 :
  ∀ (board : Fin 5 → Fin 5) (distinct : Function.Injective board),
  ∃ (ways : Nat), ways = 120 := by
  sorry

end rook_placement_5x5_l1207_120758


namespace remainder_M_divided_by_1000_l1207_120722

/-- Define flag problem parameters -/
def flagpoles: ℕ := 2
def blue_flags: ℕ := 15
def green_flags: ℕ := 10

/-- Condition: Two flagpoles, 15 blue flags and 10 green flags -/
def arrangable_flags (flagpoles blue_flags green_flags: ℕ) : Prop :=
  blue_flags + green_flags = 25 ∧ flagpoles = 2

/-- Condition: Each pole contains at least one flag -/
def each_pole_has_flag (arranged_flags: ℕ) : Prop :=
  arranged_flags > 0

/-- Condition: No two green flags are adjacent in any arrangement -/
def no_adjacent_green_flags (arranged_greens: ℕ) : Prop :=
  arranged_greens > 0

/-- Main theorem statement with correct answer -/
theorem remainder_M_divided_by_1000 (M: ℕ) : 
  arrangable_flags flagpoles blue_flags green_flags ∧ 
  each_pole_has_flag M ∧ 
  no_adjacent_green_flags green_flags ∧ 
  M % 1000 = 122
:= sorry

end remainder_M_divided_by_1000_l1207_120722


namespace number_of_correct_statements_l1207_120712

theorem number_of_correct_statements (a : ℚ) : 
  (¬ (a < 0 → -a < 0) ∧ ¬ (|a| > 0) ∧ ¬ ((a < 0 ∨ -a < 0) ∧ ¬ (a = 0))) 
  → 0 = 0 := 
by
  intro h
  sorry

end number_of_correct_statements_l1207_120712


namespace total_balloons_l1207_120703

theorem total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) 
  (h1 : fred_balloons = 10) 
  (h2 : sam_balloons = 46) 
  (h3 : dan_balloons = 16) 
  (total : fred_balloons + sam_balloons + dan_balloons = 72) :
  fred_balloons + sam_balloons + dan_balloons = 72 := 
sorry

end total_balloons_l1207_120703


namespace prize_amount_l1207_120755

theorem prize_amount (P : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : n = 40)
  (h2 : a = 40)
  (h3 : b = (2 / 5) * P)
  (h4 : c = (3 / 5) * 40)
  (h5 : b / c = 120) :
  P = 7200 := 
sorry

end prize_amount_l1207_120755


namespace jia_profits_1_yuan_l1207_120787

-- Definition of the problem conditions
def initial_cost : ℝ := 1000
def profit_rate : ℝ := 0.1
def loss_rate : ℝ := 0.1
def resale_rate : ℝ := 0.9

-- Defined transactions with conditions
def jia_selling_price1 : ℝ := initial_cost * (1 + profit_rate)
def yi_selling_price_to_jia : ℝ := jia_selling_price1 * (1 - loss_rate)
def jia_selling_price2 : ℝ := yi_selling_price_to_jia * resale_rate

-- Final net income calculation
def jia_net_income : ℝ := -initial_cost + jia_selling_price1 - yi_selling_price_to_jia + jia_selling_price2

-- Lean statement to be proved
theorem jia_profits_1_yuan : jia_net_income = 1 := sorry

end jia_profits_1_yuan_l1207_120787


namespace number_of_toys_sold_l1207_120705

theorem number_of_toys_sold (total_selling_price gain_per_toy cost_price_per_toy : ℕ)
  (h1 : total_selling_price = 25200)
  (h2 : gain_per_toy = 3 * cost_price_per_toy)
  (h3 : cost_price_per_toy = 1200) : 
  (total_selling_price - gain_per_toy) / cost_price_per_toy = 18 :=
by 
  sorry

end number_of_toys_sold_l1207_120705


namespace car_speed_second_hour_l1207_120784

theorem car_speed_second_hour (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : (s1 + s2) / 2 = 35) : s2 = 60 := by
  sorry

end car_speed_second_hour_l1207_120784


namespace T_n_correct_l1207_120710

def a_n (n : ℕ) : ℤ := 2 * n - 5

def b_n (n : ℕ) : ℤ := 2^n

def C_n (n : ℕ) : ℤ := |a_n n| * b_n n

def T_n : ℕ → ℤ
| 1     => 6
| 2     => 10
| n     => if n >= 3 then 34 + (2 * n - 7) * 2^(n + 1) else 0  -- safeguard for invalid n

theorem T_n_correct (n : ℕ) (hyp : n ≥ 1) : 
  T_n n = 
  if n = 1 then 6 
  else if n = 2 then 10 
  else if n ≥ 3 then 34 + (2 * n - 7) * 2^(n + 1) 
  else 0 := 
by 
sorry

end T_n_correct_l1207_120710


namespace true_propositions_l1207_120731

-- Definitions according to conditions:
def p (x y : ℝ) : Prop := x > y → -x < -y
def q (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Given that p is true and q is false.
axiom p_true {x y : ℝ} : p x y
axiom q_false {x y : ℝ} : ¬ q x y

-- Proving the actual propositions that are true:
theorem true_propositions (x y : ℝ) : 
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  have h1 : p x y := p_true
  have h2 : ¬ q x y := q_false
  constructor
  · left; exact h1
  · constructor; assumption; assumption

end true_propositions_l1207_120731


namespace quadratic_complete_square_l1207_120723

theorem quadratic_complete_square (c r s k : ℝ) (h1 : 8 * k^2 - 6 * k + 16 = c * (k + r)^2 + s) 
  (h2 : c = 8) 
  (h3 : r = -3 / 8) 
  (h4 : s = 119 / 8) : 
  s / r = -119 / 3 := 
by 
  sorry

end quadratic_complete_square_l1207_120723


namespace anna_correct_percentage_l1207_120761

theorem anna_correct_percentage :
  let test1_problems := 30
  let test1_score := 0.75
  let test2_problems := 50
  let test2_score := 0.85
  let test3_problems := 20
  let test3_score := 0.65
  let correct_test1 := test1_score * test1_problems
  let correct_test2 := test2_score * test2_problems
  let correct_test3 := test3_score * test3_problems
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := correct_test1 + correct_test2 + correct_test3
  (total_correct / total_problems) * 100 = 78 :=
by
  sorry

end anna_correct_percentage_l1207_120761


namespace temperature_decrease_is_negative_l1207_120749

-- Condition: A temperature rise of 3°C is denoted as +3°C.
def temperature_rise (c : Int) : String := if c > 0 then "+" ++ toString c ++ "°C" else toString c ++ "°C"

-- Specification: Prove a decrease of 4°C is denoted as -4°C.
theorem temperature_decrease_is_negative (h : temperature_rise 3 = "+3°C") : temperature_rise (-4) = "-4°C" :=
by
  -- Proof
  sorry

end temperature_decrease_is_negative_l1207_120749


namespace broccoli_sales_l1207_120701

theorem broccoli_sales (B C S Ca : ℝ) (h1 : C = 2 * B) (h2 : S = B / 2 + 16) (h3 : Ca = 136) (total_sales : B + C + S + Ca = 380) :
  B = 57 :=
by
  sorry

end broccoli_sales_l1207_120701


namespace volume_rect_prism_l1207_120757

variables (a d h : ℝ)
variables (ha : a > 0) (hd : d > 0) (hh : h > 0)

theorem volume_rect_prism : a * d * h = adh :=
by
  sorry

end volume_rect_prism_l1207_120757


namespace find_point_B_l1207_120786

theorem find_point_B (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, -5)) 
  (ha : a = (2, 3)) 
  (hAB : B - A = 3 • a) : 
  B = (5, 4) := sorry

end find_point_B_l1207_120786


namespace fill_pipe_fraction_l1207_120773

theorem fill_pipe_fraction (x : ℝ) (h : x = 1 / 2) : x = 1 / 2 :=
by
  sorry

end fill_pipe_fraction_l1207_120773


namespace Elaine_rent_percentage_l1207_120767

variable (E : ℝ) (last_year_rent : ℝ) (this_year_rent : ℝ)

def Elaine_last_year_earnings (E : ℝ) : ℝ := E

def Elaine_last_year_rent (E : ℝ) : ℝ := 0.20 * E

def Elaine_this_year_earnings (E : ℝ) : ℝ := 1.25 * E

def Elaine_this_year_rent (E : ℝ) : ℝ := 0.30 * (1.25 * E)

theorem Elaine_rent_percentage 
  (E : ℝ) 
  (last_year_rent := Elaine_last_year_rent E)
  (this_year_rent := Elaine_this_year_rent E) :
  (this_year_rent / last_year_rent) * 100 = 187.5 := 
by sorry

end Elaine_rent_percentage_l1207_120767


namespace find_c_value_l1207_120790

theorem find_c_value 
  (a b c : ℝ)
  (h_a : a = 5 / 2)
  (h_b : b = 17)
  (roots : ∀ x : ℝ, x = (-b + Real.sqrt 23) / 5 ∨ x = (-b - Real.sqrt 23) / 5)
  (discrim_eq : ∀ c : ℝ, b ^ 2 - 4 * a * c = 23) :
  c = 26.6 := by
  sorry

end find_c_value_l1207_120790


namespace greatest_b_l1207_120707

theorem greatest_b (b : ℤ) (h : ∀ x : ℝ, x^2 + b * x + 20 ≠ -6) : b = 10 := sorry

end greatest_b_l1207_120707


namespace min_inverse_ab_l1207_120788

theorem min_inverse_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) : 
  ∃ (m : ℝ), (m = 2 / 9) ∧ (∀ (a b : ℝ), a > 0 → b > 0 → a + 2 * b = 6 → 1/(a * b) ≥ m) :=
by
  sorry

end min_inverse_ab_l1207_120788


namespace find_m_value_l1207_120738

theorem find_m_value (x: ℝ) (m: ℝ) (hx: x > 2) (hm: m > 0) (h_min: ∀ y, (y = x + m / (x - 2)) → y ≥ 6) : m = 4 := 
sorry

end find_m_value_l1207_120738


namespace total_lemonade_poured_l1207_120721

-- Define the amounts of lemonade served during each intermission.
def first_intermission : ℝ := 0.25
def second_intermission : ℝ := 0.42
def third_intermission : ℝ := 0.25

-- State the theorem that the total amount of lemonade poured is 0.92 pitchers.
theorem total_lemonade_poured : first_intermission + second_intermission + third_intermission = 0.92 :=
by
  -- Placeholders to skip the proof.
  sorry

end total_lemonade_poured_l1207_120721


namespace xy_value_l1207_120798

namespace ProofProblem

variables {x y : ℤ}

theorem xy_value (h1 : x * (x + y) = x^2 + 12) (h2 : x - y = 3) : x * y = 12 :=
by
  -- The proof is not required here
  sorry

end ProofProblem

end xy_value_l1207_120798


namespace congruence_is_sufficient_but_not_necessary_for_equal_area_l1207_120724

-- Definition of conditions
def Congruent (Δ1 Δ2 : Type) : Prop := sorry -- Definition of congruent triangles
def EqualArea (Δ1 Δ2 : Type) : Prop := sorry -- Definition of triangles with equal area

-- Theorem statement
theorem congruence_is_sufficient_but_not_necessary_for_equal_area 
  (Δ1 Δ2 : Type) :
  (Congruent Δ1 Δ2 → EqualArea Δ1 Δ2) ∧ (¬ (EqualArea Δ1 Δ2 → Congruent Δ1 Δ2)) :=
sorry

end congruence_is_sufficient_but_not_necessary_for_equal_area_l1207_120724


namespace infinitely_many_n_l1207_120794

-- Definition capturing the condition: equation \( (x + y + z)^3 = n^2 xyz \)
def equation (x y z n : ℕ) : Prop := (x + y + z)^3 = n^2 * x * y * z

-- The main statement: proving the existence of infinitely many positive integers n such that the equation has a solution
theorem infinitely_many_n :
  ∃ᶠ n : ℕ in at_top, ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n :=
sorry

end infinitely_many_n_l1207_120794


namespace range_of_function_is_correct_l1207_120732

def range_of_quadratic_function : Set ℝ :=
  {y | ∃ x : ℝ, y = -x^2 - 6 * x - 5}

theorem range_of_function_is_correct :
  range_of_quadratic_function = {y | y ≤ 4} :=
by
  -- sorry allows skipping the actual proof step
  sorry

end range_of_function_is_correct_l1207_120732
