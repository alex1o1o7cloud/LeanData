import Mathlib

namespace faucets_fill_time_l1621_162156

theorem faucets_fill_time (fill_time_4faucets_200gallons_12min : 4 * 12 * faucet_rate = 200) 
    (fill_time_m_50gallons_seconds : ∃ (rate: ℚ), 8 * t_to_seconds * rate = 50) : 
    8 * t_to_seconds / 33.33 = 90 :=
by sorry


end faucets_fill_time_l1621_162156


namespace sin_240_eq_neg_sqrt3_div_2_l1621_162106

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l1621_162106


namespace john_worked_period_l1621_162133

theorem john_worked_period (A : ℝ) (n : ℕ) (h1 : 6 * A = 1 / 2 * (6 * A + n * A)) : n + 1 = 7 :=
by
  sorry

end john_worked_period_l1621_162133


namespace single_cakes_needed_l1621_162155

theorem single_cakes_needed :
  ∀ (layer_cake_frosting single_cake_frosting cupcakes_frosting brownies_frosting : ℝ)
  (layer_cakes cupcakes brownies total_frosting : ℕ)
  (single_cakes_needed : ℝ),
  layer_cake_frosting = 1 →
  single_cake_frosting = 0.5 →
  cupcakes_frosting = 0.5 →
  brownies_frosting = 0.5 →
  layer_cakes = 3 →
  cupcakes = 6 →
  brownies = 18 →
  total_frosting = 21 →
  single_cakes_needed = (total_frosting - (layer_cakes * layer_cake_frosting + cupcakes * cupcakes_frosting + brownies * brownies_frosting)) / single_cake_frosting →
  single_cakes_needed = 12 :=
by
  intros
  sorry

end single_cakes_needed_l1621_162155


namespace distance_from_diagonal_intersection_to_base_l1621_162157

theorem distance_from_diagonal_intersection_to_base (AD BC AB R : ℝ) (O : ℝ → Prop) (M N Q : ℝ) :
  (AD + BC + 2 * AB = 8) ∧
  (AD + BC) = 4 ∧
  (R = 1 / 2) ∧
  (2 = R * (AD + BC) / 2) ∧
  (BC = AD + 2 * AB) ∧
  (∀ x, x * (2 - x) = (1 / 2) ^ 2)  →
  (Q = (2 - Real.sqrt 3) / 4) :=
by
  intros
  sorry

end distance_from_diagonal_intersection_to_base_l1621_162157


namespace integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l1621_162128

open Real

noncomputable def integral_abs_x_plus_2 : ℝ :=
  ∫ x in (-4 : ℝ)..(3 : ℝ), |x + 2|

noncomputable def integral_inv_x_minus_1 : ℝ :=
  ∫ x in (2 : ℝ)..(Real.exp 1 + 1 : ℝ), 1 / (x - 1)

theorem integral_abs_x_plus_2_eq_29_div_2 :
  integral_abs_x_plus_2 = 29 / 2 :=
sorry

theorem integral_inv_x_minus_1_eq_1 :
  integral_inv_x_minus_1 = 1 :=
sorry

end integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l1621_162128


namespace c_over_e_l1621_162177

theorem c_over_e (a b c d e : ℝ) (h1 : 1 * 2 * 3 * a + 1 * 2 * 4 * a + 1 * 3 * 4 * a + 2 * 3 * 4 * a = -d)
  (h2 : 1 * 2 * 3 * 4 = e / a)
  (h3 : 1 * 2 * a + 1 * 3 * a + 1 * 4 * a + 2 * 3 * a + 2 * 4 * a + 3 * 4 * a = c) :
  c / e = 35 / 24 :=
by
  sorry

end c_over_e_l1621_162177


namespace min_sides_regular_polygon_l1621_162176

/-- A regular polygon can accurately be placed back in its original position 
    when rotated by 50°.  Prove that the minimum number of sides the polygon 
    should have is 36. -/

theorem min_sides_regular_polygon (n : ℕ) (h : ∃ k : ℕ, 50 * k = 360 / n) : n = 36 :=
  sorry

end min_sides_regular_polygon_l1621_162176


namespace amount_subtracted_for_new_ratio_l1621_162107

theorem amount_subtracted_for_new_ratio (x a : ℝ) (h1 : 3 * x = 72) (h2 : 8 * x = 192)
(h3 : (3 * x - a) / (8 * x - a) = 4 / 9) : a = 24 := by
  -- Proof will go here
  sorry

end amount_subtracted_for_new_ratio_l1621_162107


namespace combine_like_terms_problem1_combine_like_terms_problem2_l1621_162131

-- Problem 1 Statement
theorem combine_like_terms_problem1 (x y : ℝ) : 
  2*x - (x - y) + (x + y) = 2*x + 2*y :=
by
  sorry

-- Problem 2 Statement
theorem combine_like_terms_problem2 (x : ℝ) : 
  3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 :=
by
  sorry

end combine_like_terms_problem1_combine_like_terms_problem2_l1621_162131


namespace fg_of_one_eq_onehundredandfive_l1621_162178

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2)^3

theorem fg_of_one_eq_onehundredandfive : f (g 1) = 105 :=
by
  -- proof would go here
  sorry

end fg_of_one_eq_onehundredandfive_l1621_162178


namespace positive_diff_between_two_numbers_l1621_162161

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l1621_162161


namespace number_of_bushes_needed_l1621_162121

-- Definitions from the conditions
def containers_per_bush : ℕ := 10
def containers_per_zucchini : ℕ := 3
def zucchinis_required : ℕ := 72

-- Statement to prove
theorem number_of_bushes_needed : 
  ∃ bushes_needed : ℕ, bushes_needed = 22 ∧ 
  (zucchinis_required * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush = bushes_needed := 
by
  sorry

end number_of_bushes_needed_l1621_162121


namespace yoongi_flowers_left_l1621_162172

theorem yoongi_flowers_left (initial_flowers given_to_eunji given_to_yuna : ℕ) 
  (h_initial : initial_flowers = 28) 
  (h_eunji : given_to_eunji = 7) 
  (h_yuna : given_to_yuna = 9) : 
  initial_flowers - (given_to_eunji + given_to_yuna) = 12 := 
by 
  sorry

end yoongi_flowers_left_l1621_162172


namespace distance_traveled_l1621_162165

theorem distance_traveled :
  ∫ t in (3:ℝ)..(5:ℝ), (2 * t + 3 : ℝ) = 22 :=
by
  sorry

end distance_traveled_l1621_162165


namespace carly_cooks_in_72_minutes_l1621_162171

def total_time_to_cook_burgers (total_guests : ℕ) (cook_time_per_side : ℕ) (burgers_per_grill : ℕ) : ℕ :=
  let guests_who_want_two_burgers := total_guests / 2
  let guests_who_want_one_burger := total_guests - guests_who_want_two_burgers
  let total_burgers := (guests_who_want_two_burgers * 2) + guests_who_want_one_burger
  let total_batches := (total_burgers + burgers_per_grill - 1) / burgers_per_grill  -- ceil division for total batches
  total_batches * (2 * cook_time_per_side)  -- total time

theorem carly_cooks_in_72_minutes : 
  total_time_to_cook_burgers 30 4 5 = 72 :=
by 
  sorry

end carly_cooks_in_72_minutes_l1621_162171


namespace equal_sides_length_of_isosceles_right_triangle_l1621_162129

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c^2 = 2 * a^2 ∧ a^2 + a^2 + c^2 = 725

theorem equal_sides_length_of_isosceles_right_triangle (a c : ℝ) 
  (h : isosceles_right_triangle a c) : 
  a = 13.5 :=
by
  sorry

end equal_sides_length_of_isosceles_right_triangle_l1621_162129


namespace insurance_plan_percentage_l1621_162169

theorem insurance_plan_percentage
(MSRP : ℝ) (I : ℝ) (total_cost : ℝ) (state_tax_rate : ℝ)
(hMSRP : MSRP = 30)
(htotal_cost : total_cost = 54)
(hstate_tax_rate : state_tax_rate = 0.5)
(h_total_cost_eq : MSRP + I + state_tax_rate * (MSRP + I) = total_cost) :
(I / MSRP) * 100 = 20 :=
by
  -- You can leave the proof as sorry, as it's not needed for the problem
  sorry

end insurance_plan_percentage_l1621_162169


namespace dividing_by_10_l1621_162100

theorem dividing_by_10 (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 :=
by
  sorry

end dividing_by_10_l1621_162100


namespace find_a_l1621_162119

theorem find_a 
  {a : ℝ} 
  (h : ∀ x : ℝ, (ax / (x - 1) < 1) ↔ (x < 1 ∨ x > 3)) : 
  a = 2 / 3 := 
sorry

end find_a_l1621_162119


namespace chocolate_cost_proof_l1621_162164

/-- The initial amount of money Dan has. -/
def initial_amount : ℕ := 7

/-- The cost of the candy bar. -/
def candy_bar_cost : ℕ := 2

/-- The remaining amount of money Dan has after the purchases. -/
def remaining_amount : ℕ := 2

/-- The cost of the chocolate. -/
def chocolate_cost : ℕ := initial_amount - candy_bar_cost - remaining_amount

/-- Expected cost of the chocolate. -/
def expected_chocolate_cost : ℕ := 3

/-- Prove that the cost of the chocolate equals the expected cost. -/
theorem chocolate_cost_proof : chocolate_cost = expected_chocolate_cost :=
by
  sorry

end chocolate_cost_proof_l1621_162164


namespace percent_employed_females_l1621_162112

theorem percent_employed_females (percent_employed : ℝ) (percent_employed_males : ℝ) :
  percent_employed = 0.64 →
  percent_employed_males = 0.55 →
  (percent_employed - percent_employed_males) / percent_employed * 100 = 14.0625 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end percent_employed_females_l1621_162112


namespace speed_ratio_l1621_162125

noncomputable def k_value {u v x y : ℝ} (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : ℝ :=
  1 + Real.sqrt 2

theorem speed_ratio (u v x y : ℝ) (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : 
  u / v = k_value h_uv h_v h_x h_y h_ratio :=
sorry

end speed_ratio_l1621_162125


namespace force_required_l1621_162122

theorem force_required 
  (F : ℕ → ℕ)
  (h_inv : ∀ L L' : ℕ, F L * L = F L' * L')
  (h1 : F 12 = 300) :
  F 18 = 200 :=
by
  sorry

end force_required_l1621_162122


namespace complex_product_l1621_162158

theorem complex_product : (3 + 4 * I) * (-2 - 3 * I) = -18 - 17 * I :=
by
  sorry

end complex_product_l1621_162158


namespace rectangle_area_l1621_162146

theorem rectangle_area (x : ℕ) (L W : ℕ) (h₁ : L * W = 864) (h₂ : L + W = 60) (h₃ : L = W + x) : 
  ((60 - x) / 2) * ((60 + x) / 2) = 864 :=
sorry

end rectangle_area_l1621_162146


namespace geometric_sequence_product_l1621_162145

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 * a 11 = 8) :
  a 2 * a 8 = 4 :=
sorry

end geometric_sequence_product_l1621_162145


namespace number_of_eggs_in_each_basket_l1621_162140

theorem number_of_eggs_in_each_basket 
  (total_blue_eggs : ℕ)
  (total_yellow_eggs : ℕ)
  (h1 : total_blue_eggs = 30)
  (h2 : total_yellow_eggs = 42)
  (exists_basket_count : ∃ n : ℕ, 6 ≤ n ∧ total_blue_eggs % n = 0 ∧ total_yellow_eggs % n = 0) :
  ∃ n : ℕ, n = 6 := 
sorry

end number_of_eggs_in_each_basket_l1621_162140


namespace child_l1621_162183

noncomputable def child's_ticket_cost : ℕ :=
  let adult_ticket_price := 7
  let total_tickets := 900
  let total_revenue := 5100
  let childs_tickets_sold := 400
  let adult_tickets_sold := total_tickets - childs_tickets_sold
  let total_adult_revenue := adult_tickets_sold * adult_ticket_price
  let total_child_revenue := total_revenue - total_adult_revenue
  let child's_ticket_price := total_child_revenue / childs_tickets_sold
  child's_ticket_price

theorem child's_ticket_cost_is_4 : child's_ticket_cost = 4 :=
by
  have adult_ticket_price := 7
  have total_tickets := 900
  have total_revenue := 5100
  have childs_tickets_sold := 400
  have adult_tickets_sold := total_tickets - childs_tickets_sold
  have total_adult_revenue := adult_tickets_sold * adult_ticket_price
  have total_child_revenue := total_revenue - total_adult_revenue
  have child's_ticket_price := total_child_revenue / childs_tickets_sold
  show child's_ticket_cost = 4
  sorry

end child_l1621_162183


namespace secretaries_ratio_l1621_162193

theorem secretaries_ratio (A B C : ℝ) (hA: A = 75) (h_total: A + B + C = 120) : B + C = 45 :=
by {
  -- sorry: We define this part to be explored by the theorem prover
  sorry
}

end secretaries_ratio_l1621_162193


namespace job_completion_in_time_l1621_162160

theorem job_completion_in_time (t_total t_1 w_1 : ℕ) (work_done : ℚ) (h : (t_total = 30) ∧ (t_1 = 6) ∧ (w_1 = 8) ∧ (work_done = 1/3)) :
  ∃ w : ℕ, w = 4 ∧ (t_total - t_1) * w_1 / t_1 * (1 / work_done) / w = 3 :=
by
  sorry

end job_completion_in_time_l1621_162160


namespace find_number_of_adults_l1621_162181

variable (A : ℕ) -- Variable representing the number of adults.
def C : ℕ := 5  -- Number of children.

def meal_cost : ℕ := 3  -- Cost per meal in dollars.
def total_cost (A : ℕ) : ℕ := (A + C) * meal_cost  -- Total cost formula.

theorem find_number_of_adults 
  (h1 : meal_cost = 3)
  (h2 : total_cost A = 21)
  (h3 : C = 5) :
  A = 2 :=
sorry

end find_number_of_adults_l1621_162181


namespace book_cost_l1621_162159

theorem book_cost (C_1 C_2 : ℝ)
  (h1 : C_1 + C_2 = 420)
  (h2 : C_1 * 0.85 = C_2 * 1.19) :
  C_1 = 245 :=
by
  -- We skip the proof here using sorry.
  sorry

end book_cost_l1621_162159


namespace tangent_to_parabola_k_l1621_162136

theorem tangent_to_parabola_k (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 32 * x ∧ 
  ∀ (a b : ℝ) (ha : a * y^2 + b * y + k = 0), b^2 - 4 * a * k = 0) → k = 98 :=
by
  sorry

end tangent_to_parabola_k_l1621_162136


namespace arithmetic_sequence_ratio_l1621_162186

noncomputable def sum_first_n_terms (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_ratio (d : ℚ) (h : d ≠ 0) :
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  (7 * S₅) / (5 * S₇) = 10 / 11 :=
by 
  let a₁ := 8 * d
  let S₅ := sum_first_n_terms a₁ d 5
  let S₇ := sum_first_n_terms a₁ d 7
  sorry

end arithmetic_sequence_ratio_l1621_162186


namespace son_l1621_162135

variable (S M : ℤ)

-- Conditions
def condition1 : Prop := M = S + 24
def condition2 : Prop := M + 2 = 2 * (S + 2)

theorem son's_age : condition1 S M ∧ condition2 S M → S = 22 :=
by
  sorry

end son_l1621_162135


namespace same_grades_percentage_l1621_162149

theorem same_grades_percentage (total_students same_grades_A same_grades_B same_grades_C same_grades_D : ℕ) 
  (total_eq : total_students = 50) 
  (same_A : same_grades_A = 3) 
  (same_B : same_grades_B = 6) 
  (same_C : same_grades_C = 7) 
  (same_D : same_grades_D = 2) : 
  (same_grades_A + same_grades_B + same_grades_C + same_grades_D) * 100 / total_students = 36 := 
by
  sorry

end same_grades_percentage_l1621_162149


namespace gear_revolutions_l1621_162115

theorem gear_revolutions (t : ℝ) (r_p r_q : ℝ) (h1 : r_q = 40) (h2 : t = 20)
 (h3 : (r_q / 60) * t = ((r_p / 60) * t) + 10) :
 r_p = 10 :=
 sorry

end gear_revolutions_l1621_162115


namespace sector_area_l1621_162175

theorem sector_area (radius area : ℝ) (θ : ℝ) (h1 : 2 * radius + θ * radius = 16) (h2 : θ = 2) : area = 16 :=
  sorry

end sector_area_l1621_162175


namespace unit_digit_2_pow_2024_l1621_162104

theorem unit_digit_2_pow_2024 : (2 ^ 2024) % 10 = 6 := by
  -- We observe the repeating pattern in the unit digits of powers of 2:
  -- 2^1 = 2 -> unit digit is 2
  -- 2^2 = 4 -> unit digit is 4
  -- 2^3 = 8 -> unit digit is 8
  -- 2^4 = 16 -> unit digit is 6
  -- The cycle repeats every 4 powers: 2, 4, 8, 6
  -- 2024 ≡ 0 (mod 4), so it corresponds to the unit digit of 2^4, which is 6
  sorry

end unit_digit_2_pow_2024_l1621_162104


namespace inequality_pos_real_l1621_162111

theorem inequality_pos_real (
  a b c : ℝ
) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  abc ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧ 
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end inequality_pos_real_l1621_162111


namespace couple_tickets_sold_l1621_162173

theorem couple_tickets_sold (S C : ℕ) :
  20 * S + 35 * C = 2280 ∧ S + 2 * C = 128 -> C = 56 :=
by
  intro h
  sorry

end couple_tickets_sold_l1621_162173


namespace sum_of_decimals_l1621_162197

theorem sum_of_decimals :
  0.3 + 0.04 + 0.005 + 0.0006 + 0.00007 = (34567 / 100000 : ℚ) :=
by
  -- The proof details would go here
  sorry

end sum_of_decimals_l1621_162197


namespace number_of_items_l1621_162137

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l1621_162137


namespace x_varies_as_half_power_of_z_l1621_162126

variable {x y z : ℝ} -- declare variables as real numbers

-- Assume the conditions, which are the relationships between x, y, and z
variable (k j : ℝ) (k_pos : k > 0) (j_pos : j > 0)
axiom xy_relationship : ∀ y, x = k * y^2
axiom yz_relationship : ∀ z, y = j * z^(1/4)

-- The theorem we want to prove
theorem x_varies_as_half_power_of_z (z : ℝ) (h : z ≥ 0) : ∃ m, m > 0 ∧ x = m * z^(1/2) :=
sorry

end x_varies_as_half_power_of_z_l1621_162126


namespace probability_of_sequence_l1621_162134

noncomputable def prob_first_card_diamond : ℚ := 13 / 52
noncomputable def prob_second_card_spade_given_first_diamond : ℚ := 13 / 51
noncomputable def prob_third_card_heart_given_first_diamond_and_second_spade : ℚ := 13 / 50

theorem probability_of_sequence : 
  prob_first_card_diamond * prob_second_card_spade_given_first_diamond * 
  prob_third_card_heart_given_first_diamond_and_second_spade = 169 / 10200 := 
by
  -- Proof goes here
  sorry

end probability_of_sequence_l1621_162134


namespace tenth_term_arithmetic_sequence_l1621_162127

def arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

theorem tenth_term_arithmetic_sequence :
  arithmetic_sequence (1 / 2) (1 / 2) 10 = 5 :=
by
  sorry

end tenth_term_arithmetic_sequence_l1621_162127


namespace average_correct_l1621_162188

theorem average_correct :
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1252140 + 2345) / 10 = 125831.9 := 
sorry

end average_correct_l1621_162188


namespace simplify_f_value_f_second_quadrant_l1621_162190

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) / 
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : 
  f α = Real.cos α := 
sorry

theorem value_f_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (hcosα : Real.cos (π / 2 + α) = -1 / 3) :
  f α = - (2 * Real.sqrt 2) / 3 := 
sorry

end simplify_f_value_f_second_quadrant_l1621_162190


namespace vector_calculation_l1621_162108

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, -2)

theorem vector_calculation : 2 • a - b = (5, 8) := by
  sorry

end vector_calculation_l1621_162108


namespace classroom_has_total_books_l1621_162148

-- Definitions for the conditions
def num_children : Nat := 10
def books_per_child : Nat := 7
def additional_books : Nat := 8

-- Total number of books the children have
def total_books_from_children : Nat := num_children * books_per_child

-- The expected total number of books in the classroom
def total_books : Nat := total_books_from_children + additional_books

-- The main theorem to be proven
theorem classroom_has_total_books : total_books = 78 :=
by
  sorry

end classroom_has_total_books_l1621_162148


namespace classrooms_student_hamster_difference_l1621_162143

-- Define the problem conditions
def students_per_classroom := 22
def hamsters_per_classroom := 3
def number_of_classrooms := 5

-- Define the problem statement
theorem classrooms_student_hamster_difference :
  (students_per_classroom * number_of_classrooms) - 
  (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end classrooms_student_hamster_difference_l1621_162143


namespace rachel_math_homework_l1621_162196

/-- Rachel had to complete some pages of math homework. 
Given:
- 4 more pages of math homework than reading homework
- 3 pages of reading homework
Prove that Rachel had to complete 7 pages of math homework.
--/
theorem rachel_math_homework
  (r : ℕ) (h_r : r = 3)
  (m : ℕ) (h_m : m = r + 4) :
  m = 7 := by
  sorry

end rachel_math_homework_l1621_162196


namespace cost_price_of_article_l1621_162194

theorem cost_price_of_article (x : ℝ) :
  (86 - x = x - 42) → x = 64 :=
by
  intro h
  sorry

end cost_price_of_article_l1621_162194


namespace prime_divisibility_l1621_162110

theorem prime_divisibility (a b : ℕ) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := 
by
  sorry

end prime_divisibility_l1621_162110


namespace complementary_event_A_l1621_162118

-- Define the events
def EventA (defective : ℕ) : Prop := defective ≥ 2

def ComplementaryEvent (defective : ℕ) : Prop := defective ≤ 1

-- Question: Prove that the complementary event of event A ("at least 2 defective products") 
-- is "at most 1 defective product" given the conditions.
theorem complementary_event_A (defective : ℕ) (total : ℕ) (h_total : total = 10) :
  EventA defective ↔ ComplementaryEvent defective :=
by sorry

end complementary_event_A_l1621_162118


namespace plane_parallel_l1621_162102

-- Definitions for planes and lines within a plane
variable (Plane : Type) (Line : Type)
variables (lines_in_plane1 : Set Line)
variables (parallel_to_plane2 : Line → Prop)
variables (Plane1 Plane2 : Plane)

-- Conditions
axiom infinite_lines_in_plane1_parallel_to_plane2 : ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l
axiom planes_are_parallel : ∀ (P1 P2 : Plane), (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) → P1 = Plane1 → P2 = Plane2 → (Plane1 ≠ Plane2 ∧ (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l))

-- The proof that Plane 1 and Plane 2 are parallel based on the conditions
theorem plane_parallel : Plane1 ≠ Plane2 → ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l → (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) := 
by
  sorry

end plane_parallel_l1621_162102


namespace find_b_from_quadratic_l1621_162153

theorem find_b_from_quadratic (b n : ℤ)
  (h1 : b > 0)
  (h2 : (x : ℤ) → (x + n)^2 - 6 = x^2 + b * x + 19) :
  b = 10 :=
sorry

end find_b_from_quadratic_l1621_162153


namespace oil_truck_radius_l1621_162195

/-- 
A full stationary oil tank that is a right circular cylinder has a radius of 100 feet 
and a height of 25 feet. Oil is pumped from the stationary tank to an oil truck that 
has a tank that is a right circular cylinder. The oil level dropped 0.025 feet in the stationary tank. 
The oil truck's tank has a height of 10 feet. The radius of the oil truck's tank is 5 feet. 
--/
theorem oil_truck_radius (r_stationary : ℝ) (h_stationary : ℝ) (h_truck : ℝ) 
  (Δh : ℝ) (r_truck : ℝ) 
  (h_stationary_pos : 0 < h_stationary) (h_truck_pos : 0 < h_truck) (r_stationary_pos : 0 < r_stationary) :
  r_stationary = 100 → h_stationary = 25 → Δh = 0.025 → h_truck = 10 → r_truck = 5 → 
  π * (r_stationary ^ 2) * Δh = π * (r_truck ^ 2) * h_truck :=
by 
  -- Use the conditions and perform algebra to show the equality.
  sorry

end oil_truck_radius_l1621_162195


namespace total_oysters_and_crabs_is_195_l1621_162116

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end total_oysters_and_crabs_is_195_l1621_162116


namespace daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l1621_162189

-- Definitions based on conditions
def purchase_price : ℝ := 30
def max_selling_price : ℝ := 55
def linear_relationship (x : ℝ) : ℝ := -2 * x + 140
def profit (x : ℝ) : ℝ := (x - purchase_price) * linear_relationship x

-- Part 1: Daily profit when selling price is 35 yuan
theorem daily_profit_35 : profit 35 = 350 :=
  sorry

-- Part 2: Selling price for a daily profit of 600 yuan
theorem selling_price_for_600_profit (x : ℝ) (h1 : 30 ≤ x) (h2 : x ≤ 55) : profit x = 600 → x = 40 :=
  sorry

-- Part 3: Possibility of daily profit of 900 yuan
theorem no_900_profit_possible (h1 : ∀ x, 30 ≤ x ∧ x ≤ 55 → profit x ≠ 900) : ¬ ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ profit x = 900 :=
  sorry

end daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l1621_162189


namespace batsman_average_after_11th_inning_l1621_162120

theorem batsman_average_after_11th_inning 
  (x : ℝ) 
  (h1 : (10 * x + 95) / 11 = x + 5) : 
  x + 5 = 45 :=
by 
  sorry

end batsman_average_after_11th_inning_l1621_162120


namespace max_ab_min_3x_4y_max_f_l1621_162167

-- Proof Problem 1
theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : ab <= 1/16 :=
  sorry

-- Proof Problem 2
theorem min_3x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y >= 5 :=
  sorry

-- Proof Problem 3
theorem max_f (x : ℝ) (h1 : x < 5/4) : 4 * x - 2 + 1 / (4 * x - 5) <= 1 :=
  sorry

end max_ab_min_3x_4y_max_f_l1621_162167


namespace absolute_inequality_solution_l1621_162151

theorem absolute_inequality_solution (x : ℝ) (hx : x > 0) :
  |5 - 2 * x| ≤ 8 ↔ 0 ≤ x ∧ x ≤ 6.5 :=
by sorry

end absolute_inequality_solution_l1621_162151


namespace inequality_solution_l1621_162152

theorem inequality_solution (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < (5 / 9) :=
by
  sorry

end inequality_solution_l1621_162152


namespace find_amount_of_alcohol_l1621_162180

theorem find_amount_of_alcohol (A W : ℝ) (h₁ : A / W = 4 / 3) (h₂ : A / (W + 7) = 4 / 5) : A = 14 := 
sorry

end find_amount_of_alcohol_l1621_162180


namespace relationship_even_increasing_l1621_162198

-- Even function definition
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Monotonically increasing function definition on interval
def increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

variable {f : ℝ → ℝ}

-- The proof problem statement
theorem relationship_even_increasing (h_even : even_function f) (h_increasing : increasing_on f 0 1) :
  f 0 < f (-0.5) ∧ f (-0.5) < f (-1) :=
by
  sorry

end relationship_even_increasing_l1621_162198


namespace find_phi_l1621_162123

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, 2 * Real.sin (2 * x + φ - π / 6) = 2 * Real.cos (2 * x)) → φ = 5 * π / 6 :=
by
  sorry

end find_phi_l1621_162123


namespace hyperbola_range_m_l1621_162147

theorem hyperbola_range_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1)) - (y^2 / (m - 2)) = 1) ↔ (m < -1) ∨ (m > 2) := 
by
  sorry

end hyperbola_range_m_l1621_162147


namespace find_y_in_terms_of_x_and_n_l1621_162199

variable (x n y : ℝ)

theorem find_y_in_terms_of_x_and_n
  (h : n = 3 * x * y / (x - y)) :
  y = n * x / (3 * x + n) :=
  sorry

end find_y_in_terms_of_x_and_n_l1621_162199


namespace cost_of_eraser_l1621_162109

theorem cost_of_eraser
  (total_money: ℕ)
  (n_sharpeners n_notebooks n_erasers n_highlighters: ℕ)
  (price_sharpener price_notebook price_highlighter: ℕ)
  (heaven_spent brother_spent remaining_money final_spent: ℕ) :
  total_money = 100 →
  n_sharpeners = 2 →
  price_sharpener = 5 →
  n_notebooks = 4 →
  price_notebook = 5 →
  n_highlighters = 1 →
  price_highlighter = 30 →
  heaven_spent = n_sharpeners * price_sharpener + n_notebooks * price_notebook →
  brother_spent = 30 →
  remaining_money = total_money - heaven_spent →
  final_spent = remaining_money - brother_spent →
  final_spent = 40 →
  n_erasers = 10 →
  ∀ cost_per_eraser: ℕ, final_spent = cost_per_eraser * n_erasers →
  cost_per_eraser = 4 := by
  intros h_total_money h_n_sharpeners h_price_sharpener h_n_notebooks h_price_notebook
    h_n_highlighters h_price_highlighter h_heaven_spent h_brother_spent h_remaining_money
    h_final_spent h_n_erasers cost_per_eraser h_final_cost
  sorry

end cost_of_eraser_l1621_162109


namespace spring_problem_l1621_162141

theorem spring_problem (x y : ℝ) : 
  (∀ x, y = 0.5 * x + 12) →
  (0.5 * 3 + 12 = 13.5) ∧
  (y = 0.5 * x + 12) ∧
  (0.5 * 5.5 + 12 = 14.75) ∧
  (20 = 0.5 * 16 + 12) :=
by 
  sorry

end spring_problem_l1621_162141


namespace negation_equivalence_l1621_162101

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end negation_equivalence_l1621_162101


namespace eq1_eq2_eq3_eq4_l1621_162138

/-
  First, let's define each problem and then state the equivalency of the solutions.
  We will assume the real number type for the domain of x.
-/

-- Assume x is a real number
variable (x : ℝ)

theorem eq1 (x : ℝ) : (x - 3)^2 = 4 -> (x = 5 ∨ x = 1) := sorry

theorem eq2 (x : ℝ) : x^2 - 5 * x + 1 = 0 -> (x = (5 - Real.sqrt 21) / 2 ∨ x = (5 + Real.sqrt 21) / 2) := sorry

theorem eq3 (x : ℝ) : x * (3 * x - 2) = 2 * (3 * x - 2) -> (x = 2 / 3 ∨ x = 2) := sorry

theorem eq4 (x : ℝ) : (x + 1)^2 = 4 * (1 - x)^2 -> (x = 1 / 3 ∨ x = 3) := sorry

end eq1_eq2_eq3_eq4_l1621_162138


namespace prove_x_eq_one_l1621_162184

variables (x y : ℕ)

theorem prove_x_eq_one 
  (hx : x > 0) 
  (hy : y > 0) 
  (hdiv : ∀ n : ℕ, n > 0 → (2^n * y + 1) ∣ (x^2^n - 1)) : 
  x = 1 :=
sorry

end prove_x_eq_one_l1621_162184


namespace evaluate_gg2_l1621_162144

noncomputable def g (x : ℚ) : ℚ := 1 / (x^2) + (x^2) / (1 + x^2)

theorem evaluate_gg2 : g (g 2) = 530881 / 370881 :=
by
  sorry

end evaluate_gg2_l1621_162144


namespace line_passes_through_point_l1621_162103

theorem line_passes_through_point (k : ℝ) :
  ∀ k : ℝ, (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 :=
by
  intro k
  sorry

end line_passes_through_point_l1621_162103


namespace apples_in_each_basket_l1621_162162

theorem apples_in_each_basket (total_apples : ℕ) (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : baskets = 19) 
  (h3 : apples_per_basket = total_apples / baskets) : 
  apples_per_basket = 26 :=
by 
  rw [h1, h2] at h3
  exact h3

end apples_in_each_basket_l1621_162162


namespace border_area_is_correct_l1621_162174

def framed_area (height width border: ℝ) : ℝ :=
  (height + 2 * border) * (width + 2 * border)

def photograph_area (height width: ℝ) : ℝ :=
  height * width

theorem border_area_is_correct (h w b : ℝ) (h6 : h = 6) (w8 : w = 8) (b3 : b = 3) :
  (framed_area h w b - photograph_area h w) = 120 := by
  sorry

end border_area_is_correct_l1621_162174


namespace volume_ratio_sum_is_26_l1621_162142

noncomputable def volume_of_dodecahedron (s : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * s ^ 3 / 4

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def volume_ratio_sum (s : ℝ) : ℝ :=
  let ratio := (volume_of_dodecahedron s) / (volume_of_cube s)
  let numerator := 15 + 7 * Real.sqrt 5
  let denominator := 4
  numerator + denominator

theorem volume_ratio_sum_is_26 (s : ℝ) : volume_ratio_sum s = 26 := by
  sorry

end volume_ratio_sum_is_26_l1621_162142


namespace initial_population_correct_l1621_162185

-- Definitions based on conditions
def initial_population (P : ℝ) := P
def population_after_bombardment (P : ℝ) := 0.9 * P
def population_after_fear (P : ℝ) := 0.8 * (population_after_bombardment P)
def final_population := 3240

-- Theorem statement
theorem initial_population_correct (P : ℝ) (h : population_after_fear P = final_population) :
  initial_population P = 4500 :=
sorry

end initial_population_correct_l1621_162185


namespace work_completion_time_l1621_162114

theorem work_completion_time (x : ℕ) (h1 : ∀ B, ∀ A, A = 2 * B) (h2 : (1/x + 1/(2*x)) * 4 = 1) : x = 12 := 
sorry

end work_completion_time_l1621_162114


namespace minimum_value_2_l1621_162132

noncomputable def minimum_value (x y : ℝ) : ℝ := 2 * x + 3 * y ^ 2

theorem minimum_value_2 (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2 * y = 1) : minimum_value x y = 2 :=
sorry

end minimum_value_2_l1621_162132


namespace employees_participating_in_game_l1621_162124

theorem employees_participating_in_game 
  (managers players : ℕ)
  (teams people_per_team : ℕ)
  (h_teams : teams = 3)
  (h_people_per_team : people_per_team = 2)
  (h_managers : managers = 3)
  (h_total_players : players = teams * people_per_team) :
  players - managers = 3 :=
sorry

end employees_participating_in_game_l1621_162124


namespace bryan_more_than_ben_l1621_162139

theorem bryan_more_than_ben :
  let Bryan_candies := 50
  let Ben_candies := 20
  Bryan_candies - Ben_candies = 30 :=
by
  let Bryan_candies := 50
  let Ben_candies := 20
  sorry

end bryan_more_than_ben_l1621_162139


namespace opposite_of_a_is_2_l1621_162170

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2_l1621_162170


namespace perfect_square_trinomial_m_value_l1621_162113

theorem perfect_square_trinomial_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ y : ℤ, y^2 + my + 9 = (y + a)^2) ↔ (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_value_l1621_162113


namespace box_width_l1621_162192

theorem box_width (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) : 
  rate = 4 ∧ time = 21 ∧ length = 7 ∧ depth = 2 ∧ volume = rate * time ∧ volume = length * width * depth → width = 6 :=
by
  sorry

end box_width_l1621_162192


namespace ab_necessary_not_sufficient_l1621_162117

theorem ab_necessary_not_sufficient (a b : ℝ) : 
  (ab > 0) ↔ ((a ≠ 0) ∧ (b ≠ 0) ∧ ((b / a + a / b > 2) → (ab > 0))) := 
sorry

end ab_necessary_not_sufficient_l1621_162117


namespace total_number_of_workers_l1621_162105

theorem total_number_of_workers 
    (W : ℕ) 
    (average_salary_all : ℕ := 8000) 
    (average_salary_technicians : ℕ := 12000) 
    (average_salary_rest : ℕ := 6000) 
    (total_salary_all : ℕ := average_salary_all * W) 
    (salary_technicians : ℕ := 6 * average_salary_technicians) 
    (N : ℕ := W - 6) 
    (salary_rest : ℕ := average_salary_rest * N) 
    (salary_equation : total_salary_all = salary_technicians + salary_rest) 
  : W = 18 := 
sorry

end total_number_of_workers_l1621_162105


namespace odd_square_mod_eight_l1621_162179

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l1621_162179


namespace total_selling_price_correct_l1621_162130

-- Define the cost prices of the three articles
def cost_A : ℕ := 400
def cost_B : ℕ := 600
def cost_C : ℕ := 800

-- Define the desired profit percentages for the three articles
def profit_percent_A : ℚ := 40 / 100
def profit_percent_B : ℚ := 35 / 100
def profit_percent_C : ℚ := 25 / 100

-- Define the selling prices of the three articles
def selling_price_A : ℚ := cost_A * (1 + profit_percent_A)
def selling_price_B : ℚ := cost_B * (1 + profit_percent_B)
def selling_price_C : ℚ := cost_C * (1 + profit_percent_C)

-- Define the total selling price
def total_selling_price : ℚ := selling_price_A + selling_price_B + selling_price_C

-- The proof statement
theorem total_selling_price_correct : total_selling_price = 2370 :=
sorry

end total_selling_price_correct_l1621_162130


namespace minimize_S_n_l1621_162168

noncomputable def S_n (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 - 30 * (n : ℝ)

theorem minimize_S_n :
  ∃ n : ℕ, S_n n = 2 * (7 : ℝ) ^ 2 - 30 * (7 : ℝ) ∨ S_n n = 2 * (8 : ℝ) ^ 2 - 30 * (8 : ℝ) := by
  sorry

end minimize_S_n_l1621_162168


namespace division_of_decimals_l1621_162154

theorem division_of_decimals :
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end division_of_decimals_l1621_162154


namespace quadratic_radicals_x_le_10_l1621_162182

theorem quadratic_radicals_x_le_10 (a x : ℝ) (h1 : 3 * a - 8 = 17 - 2 * a) (h2 : 4 * a - 2 * x ≥ 0) : x ≤ 10 :=
by
  sorry

end quadratic_radicals_x_le_10_l1621_162182


namespace am_hm_inequality_l1621_162187

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : ℝ :=
  (a + b + c) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem am_hm_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  smallest_possible_value a b c d h1 h2 h3 h4 ≥ 9 / 2 :=
by
  sorry

end am_hm_inequality_l1621_162187


namespace labor_productivity_increase_l1621_162166

noncomputable def regression_equation (x : ℝ) : ℝ := 50 + 60 * x

theorem labor_productivity_increase (Δx : ℝ) (hx : Δx = 1) :
  regression_equation (x + Δx) - regression_equation x = 60 :=
by
  sorry

end labor_productivity_increase_l1621_162166


namespace games_per_season_l1621_162163

-- Define the problem parameters
def total_goals : ℕ := 1244
def louie_last_match_goals : ℕ := 4
def louie_previous_goals : ℕ := 40
def louie_season_total_goals := louie_last_match_goals + louie_previous_goals
def brother_goals_per_game := 2 * louie_last_match_goals
def seasons : ℕ := 3

-- Prove the number of games in each season
theorem games_per_season : ∃ G : ℕ, louie_season_total_goals + (seasons * brother_goals_per_game * G) = total_goals ∧ G = 50 := 
by {
  sorry
}

end games_per_season_l1621_162163


namespace trapezoid_bd_length_l1621_162150

theorem trapezoid_bd_length
  (AB CD AC BD : ℝ)
  (tanC tanB : ℝ)
  (h1 : AB = 24)
  (h2 : CD = 15)
  (h3 : AC = 30)
  (h4 : tanC = 2)
  (h5 : tanB = 1.25)
  (h6 : AC ^ 2 = AB ^ 2 + (CD - AB) ^ 2) :
  BD = 9 * Real.sqrt 11 := by
  sorry

end trapezoid_bd_length_l1621_162150


namespace num_lineups_l1621_162191

-- Define the given conditions
def num_players : ℕ := 12
def num_lineman : ℕ := 4
def num_qb_among_lineman : ℕ := 2
def num_running_backs : ℕ := 3

-- State the problem and the result as a theorem
theorem num_lineups : 
  (num_lineman * (num_qb_among_lineman) * (num_running_backs) * (num_players - num_lineman - num_qb_among_lineman - num_running_backs + 3) = 216) := 
by
  -- The proof will go here
  sorry

end num_lineups_l1621_162191
