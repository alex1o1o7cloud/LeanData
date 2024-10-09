import Mathlib

namespace volume_expression_correct_l817_81719

variable (x : ℝ)

def volume (x : ℝ) := x * (30 - 2 * x) * (20 - 2 * x)

theorem volume_expression_correct (h : x < 10) :
  volume x = 4 * x^3 - 100 * x^2 + 600 * x :=
by sorry

end volume_expression_correct_l817_81719


namespace a_perp_a_add_b_l817_81783

def vector (α : Type*) := α × α

def a : vector ℤ := (2, -1)
def b : vector ℤ := (1, 7)

def dot_product (v1 v2 : vector ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def add_vector (v1 v2 : vector ℤ) : vector ℤ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def perpendicular (v1 v2 : vector ℤ) : Prop :=
  dot_product v1 v2 = 0

theorem a_perp_a_add_b :
  perpendicular a (add_vector a b) :=
by {
  sorry
}

end a_perp_a_add_b_l817_81783


namespace arithmetic_sequence_sum_l817_81747

theorem arithmetic_sequence_sum :
  ∃ (a_n : ℕ → ℝ) (d : ℝ), 
  (∀ n, a_n n = a_n 0 + n * d) ∧
  d > 0 ∧
  a_n 0 + a_n 1 + a_n 2 = 15 ∧
  a_n 0 * a_n 1 * a_n 2 = 80 →
  a_n 10 + a_n 11 + a_n 12 = 135 :=
by
  sorry

end arithmetic_sequence_sum_l817_81747


namespace average_salary_excluding_manager_l817_81752

theorem average_salary_excluding_manager (A : ℝ) 
  (num_employees : ℝ := 20)
  (manager_salary : ℝ := 3300)
  (salary_increase : ℝ := 100)
  (total_salary_with_manager : ℝ := 21 * (A + salary_increase)) :
  20 * A + manager_salary = total_salary_with_manager → A = 1200 := 
by
  intro h
  sorry

end average_salary_excluding_manager_l817_81752


namespace original_population_multiple_of_5_l817_81794

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem original_population_multiple_of_5 (x y z : ℕ) 
  (H1 : is_perfect_square (x * x)) 
  (H2 : x * x + 200 = y * y) 
  (H3 : y * y + 180 = z * z) : 
  ∃ k : ℕ, x * x = 5 * k := 
sorry

end original_population_multiple_of_5_l817_81794


namespace solve_system_l817_81764

theorem solve_system (x y : ℝ) :
  (2 * y = (abs (2 * x + 3)) - (abs (2 * x - 3))) ∧ 
  (4 * x = (abs (y + 2)) - (abs (y - 2))) → 
  (-1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x) := 
by
  sorry

end solve_system_l817_81764


namespace congruent_triangles_solve_x_l817_81733

theorem congruent_triangles_solve_x (x : ℝ) (h1 : x > 0)
    (h2 : x^2 - 1 = 3) (h3 : x^2 + 1 = 5) (h4 : x^2 + 3 = 7) : x = 2 :=
by
  sorry

end congruent_triangles_solve_x_l817_81733


namespace elem_of_M_l817_81741

variable (U M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : U \ M = {1, 3})

theorem elem_of_M : 2 ∈ M :=
by {
  sorry
}

end elem_of_M_l817_81741


namespace probability_red_balls_fourth_draw_l817_81780

theorem probability_red_balls_fourth_draw :
  let p_red := 2 / 10
  let p_white := 8 / 10
  p_red * p_red * p_white * p_white * 3 / 10 + 
  p_red * p_white * p_red * p_white * 2 / 10 + 
  p_white * p_red * p_red * p_red = 0.0434 :=
sorry

end probability_red_balls_fourth_draw_l817_81780


namespace kabadi_kho_kho_players_l817_81758

theorem kabadi_kho_kho_players (total_players kabadi_only kho_kho_only both_games : ℕ)
  (h1 : kabadi_only = 10)
  (h2 : kho_kho_only = 40)
  (h3 : total_players = 50)
  (h4 : kabadi_only + kho_kho_only - both_games = total_players) :
  both_games = 0 := by
  sorry

end kabadi_kho_kho_players_l817_81758


namespace coin_problem_l817_81755

theorem coin_problem :
  ∃ n : ℕ, (n % 8 = 5) ∧ (n % 7 = 2) ∧ (n % 9 = 1) := 
sorry

end coin_problem_l817_81755


namespace find_a_cubed_l817_81744

-- Definitions based on conditions
def varies_inversely (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement with given conditions
theorem find_a_cubed (a b : ℝ) (k : ℝ) (h1 : varies_inversely a b)
    (h2 : a = 2) (h3 : b = 4) (k_val : k = 2048) (b_new : b = 8) : a^3 = 1 / 2 :=
sorry

end find_a_cubed_l817_81744


namespace blue_pigment_percentage_l817_81717

-- Define weights and pigments in the problem
variables (S G : ℝ)
-- Conditions
def sky_blue_paint := 0.9 * S = 4.5
def total_weight := S + G = 10
def sky_blue_blue_pigment := 0.1
def green_blue_pigment := 0.7

-- Prove the percentage of blue pigment in brown paint is 40%
theorem blue_pigment_percentage :
  sky_blue_paint S →
  total_weight S G →
  (0.1 * (4.5 / 0.9) + 0.7 * (10 - (4.5 / 0.9))) / 10 * 100 = 40 :=
by
  intros h1 h2
  sorry

end blue_pigment_percentage_l817_81717


namespace anna_original_money_l817_81739

theorem anna_original_money (x : ℝ) (h : (3 / 4) * x = 24) : x = 32 :=
by
  sorry

end anna_original_money_l817_81739


namespace meet_without_contact_probability_l817_81703

noncomputable def prob_meet_without_contact : ℝ :=
  let total_area := 1
  let outside_area := (1 / 8) * 2
  total_area - outside_area

theorem meet_without_contact_probability :
  prob_meet_without_contact = 3 / 4 :=
by
  sorry

end meet_without_contact_probability_l817_81703


namespace part_one_part_two_l817_81738

def f (x : ℝ) : ℝ := |x| + |x - 1|

theorem part_one (m : ℝ) (h : ∀ x, f x ≥ |m - 1|) : m ≤ 2 := by
  sorry

theorem part_two (a b : ℝ) (M : ℝ) (ha : 0 < a) (hb : 0 < b) (hM : a^2 + b^2 = M) (hM_value : M = 2) : a + b ≥ 2 * a * b := by
  sorry

end part_one_part_two_l817_81738


namespace base_8_addition_l817_81707

theorem base_8_addition (X Y : ℕ) (h1 : Y + 2 % 8 = X % 8) (h2 : X + 3 % 8 = 2 % 8) : X + Y = 12 := by
  sorry

end base_8_addition_l817_81707


namespace sandwich_cost_l817_81795

theorem sandwich_cost (S : ℝ) (h : 2 * S + 4 * 0.87 = 8.36) : S = 2.44 :=
by sorry

end sandwich_cost_l817_81795


namespace hyperbola_standard_equation_l817_81722

theorem hyperbola_standard_equation
  (passes_through : ∀ {x y : ℝ}, (x, y) = (1, 1) → 2 * x + y = 0 ∨ 2 * x - y = 0)
  (asymptote1 : ∀ {x y : ℝ}, 2 * x + y = 0 → y = -2 * x)
  (asymptote2 : ∀ {x y : ℝ}, 2 * x - y = 0 → y = 2 * x) :
  ∃ a b : ℝ, a = 4 / 3 ∧ b = 1 / 3 ∧ ∀ x y : ℝ, (x, y) = (1, 1) → (x^2 / a - y^2 / b = 1) := 
sorry

end hyperbola_standard_equation_l817_81722


namespace eq_holds_for_n_l817_81788

theorem eq_holds_for_n (n : ℕ) (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a + b + c + d = n * Real.sqrt (a * b * c * d) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end eq_holds_for_n_l817_81788


namespace max_value_of_y_l817_81742

open Classical

noncomputable def satisfies_equation (x y : ℝ) : Prop := y * x * (x + y) = x - y

theorem max_value_of_y : 
  ∀ (y : ℝ), (∃ (x : ℝ), x > 0 ∧ satisfies_equation x y) → y ≤ 1 / 3 := 
sorry

end max_value_of_y_l817_81742


namespace Lisa_photos_l817_81778

variable (a f s : ℕ)

theorem Lisa_photos (h1: a = 10) (h2: f = 3 * a) (h3: s = f - 10) : a + f + s = 60 := by
  sorry

end Lisa_photos_l817_81778


namespace boys_in_2nd_l817_81706

def students_in_3rd := 19
def students_in_4th := 2 * students_in_3rd
def girls_in_2nd := 19
def total_students := 86
def students_in_2nd := total_students - students_in_3rd - students_in_4th

theorem boys_in_2nd : students_in_2nd - girls_in_2nd = 10 := by
  sorry

end boys_in_2nd_l817_81706


namespace graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l817_81724

theorem graph_of_3x2_minus_12y2_is_pair_of_straight_lines :
  ∀ (x y : ℝ), 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l817_81724


namespace solve_inequality_system_l817_81770

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l817_81770


namespace simplify_complex_expr_correct_l817_81746

noncomputable def simplify_complex_expr (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (4 - 2 * i) - 2 * i * (3 - 2 * i) + (1 + i) * (2 + i)

theorem simplify_complex_expr_correct (i : ℂ) (h : i^2 = -1) : 
  simplify_complex_expr i h = 9 - 9 * i :=
by
  sorry

end simplify_complex_expr_correct_l817_81746


namespace toy_cars_in_third_box_l817_81791

theorem toy_cars_in_third_box (total_cars first_box second_box : ℕ) (H1 : total_cars = 71) 
    (H2 : first_box = 21) (H3 : second_box = 31) : total_cars - (first_box + second_box) = 19 :=
by
  sorry

end toy_cars_in_third_box_l817_81791


namespace age_difference_is_58_l817_81751

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Age_difference : ℕ := Grandfather_age - Milena_age

theorem age_difference_is_58 : Age_difference = 58 := by
  sorry

end age_difference_is_58_l817_81751


namespace victory_saved_less_l817_81734

-- Definitions based on conditions
def total_savings : ℕ := 1900
def sam_savings : ℕ := 1000
def victory_savings : ℕ := total_savings - sam_savings

-- Prove that Victory saved $100 less than Sam
theorem victory_saved_less : sam_savings - victory_savings = 100 := by
  -- placeholder for the proof
  sorry

end victory_saved_less_l817_81734


namespace pencils_bought_at_cost_price_l817_81737

variable (C S : ℝ)
variable (n : ℕ)

theorem pencils_bought_at_cost_price (h1 : n * C = 8 * S) (h2 : S = 1.5 * C) : n = 12 := 
by sorry

end pencils_bought_at_cost_price_l817_81737


namespace circle_radius_l817_81721

theorem circle_radius (r₂ : ℝ) : 
  (∃ r₁ : ℝ, r₁ = 5 ∧ (∀ d : ℝ, d = 7 → (d = r₁ + r₂ ∨ d = abs (r₁ - r₂)))) → (r₂ = 2 ∨ r₂ = 12) :=
by
  sorry

end circle_radius_l817_81721


namespace percent_diamond_jewels_l817_81793

def percent_beads : ℝ := 0.3
def percent_ruby_jewels : ℝ := 0.5

theorem percent_diamond_jewels (percent_beads percent_ruby_jewels : ℝ) : 
  (1 - percent_beads) * (1 - percent_ruby_jewels) = 0.35 :=
by
  -- We insert the proof steps here
  sorry

end percent_diamond_jewels_l817_81793


namespace sandy_age_when_record_l817_81750

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l817_81750


namespace exists_fi_l817_81779

theorem exists_fi (f : ℝ → ℝ) (h_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧ 
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧ 
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧ 
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧ 
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
by
  sorry

end exists_fi_l817_81779


namespace abc_eq_ab_bc_ca_l817_81705

variable {u v w A B C : ℝ}
variable (Huvw : u * v * w = 1)
variable (HA : A = u * v + u + 1)
variable (HB : B = v * w + v + 1)
variable (HC : C = w * u + w + 1)

theorem abc_eq_ab_bc_ca 
  (Huvw : u * v * w = 1)
  (HA : A = u * v + u + 1)
  (HB : B = v * w + v + 1)
  (HC : C = w * u + w + 1) : 
  A * B * C = A * B + B * C + C * A := 
by
  sorry

end abc_eq_ab_bc_ca_l817_81705


namespace solve_inequality_l817_81712

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l817_81712


namespace f_at_2_l817_81767

noncomputable def f (x : ℝ) (a b : ℝ) := a * Real.log x + b / x + x
noncomputable def g (x : ℝ) (a b : ℝ) := (a / x) - (b / (x ^ 2)) + 1

theorem f_at_2 (a b : ℝ) (ha : g 1 a b = 0) (hb : g 3 a b = 0) : f 2 a b = 1 / 2 - 4 * Real.log 2 :=
by
  sorry

end f_at_2_l817_81767


namespace min_value_of_f_range_of_a_l817_81702

def f (x : ℝ) : ℝ := 2 * |x - 2| - x + 5

theorem min_value_of_f : ∃ (m : ℝ), m = 3 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use 3
  sorry

theorem range_of_a (a : ℝ) : (|a + 2| ≥ 3 ↔ a ≤ -5 ∨ a ≥ 1) :=
sorry

end min_value_of_f_range_of_a_l817_81702


namespace hyperbola_eccentricity_sqrt2_l817_81714

theorem hyperbola_eccentricity_sqrt2
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h : (c + a)^2 + (b^2 / a)^2 = 2 * c * (c + a)) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_sqrt2_l817_81714


namespace young_employees_l817_81735

theorem young_employees (ratio_young : ℕ)
                        (ratio_middle : ℕ)
                        (ratio_elderly : ℕ)
                        (sample_selected : ℕ)
                        (prob_selection : ℚ)
                        (h_ratio : ratio_young = 10 ∧ ratio_middle = 8 ∧ ratio_elderly = 7)
                        (h_sample : sample_selected = 200)
                        (h_prob : prob_selection = 0.2) :
                        10 * (sample_selected / prob_selection) / 25 = 400 :=
by {
  sorry
}

end young_employees_l817_81735


namespace largest_n_for_ap_interior_angles_l817_81723

theorem largest_n_for_ap_interior_angles (n : ℕ) (d : ℤ) (a : ℤ) :
  (∀ i ∈ Finset.range n, a + i * d < 180) → 720 = d * (n - 1) * n → n ≤ 27 :=
by
  sorry

end largest_n_for_ap_interior_angles_l817_81723


namespace log_sqrt_pi_simplification_l817_81790

theorem log_sqrt_pi_simplification:
  2 * Real.log 4 + Real.log (5 / 8) + Real.sqrt ((Real.sqrt 3 - Real.pi) ^ 2) = 1 + Real.pi - Real.sqrt 3 :=
sorry

end log_sqrt_pi_simplification_l817_81790


namespace sum_of_a_and_b_l817_81725

theorem sum_of_a_and_b (a b : ℤ) (h1 : a + 2 * b = 8) (h2 : 2 * a + b = 4) : a + b = 4 := by
  sorry

end sum_of_a_and_b_l817_81725


namespace sum_of_integers_990_l817_81797

theorem sum_of_integers_990 :
  ∃ (n m : ℕ), (n * (n + 1) = 990 ∧ (m - 1) * m * (m + 1) = 990 ∧ (n + n + 1 + m - 1 + m + m + 1 = 90)) :=
sorry

end sum_of_integers_990_l817_81797


namespace inheritance_amount_l817_81704

-- Definitions of conditions
def inheritance (y : ℝ) : Prop :=
  let federalTaxes := 0.25 * y
  let remainingAfterFederal := 0.75 * y
  let stateTaxes := 0.1125 * y
  let totalTaxes := federalTaxes + stateTaxes
  totalTaxes = 12000

-- Theorem statement
theorem inheritance_amount (y : ℝ) (h : inheritance y) : y = 33103 :=
sorry

end inheritance_amount_l817_81704


namespace form_square_from_trapezoid_l817_81798

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem form_square_from_trapezoid (a b h : ℝ) (trapezoid_area_eq_five : trapezoid_area a b h = 5) :
  ∃ s : ℝ, s^2 = 5 :=
by
  use (Real.sqrt 5)
  sorry

end form_square_from_trapezoid_l817_81798


namespace MrBensonPaidCorrectAmount_l817_81772

-- Definitions based on the conditions
def generalAdmissionTicketPrice : ℤ := 40
def VIPTicketPrice : ℤ := 60
def premiumTicketPrice : ℤ := 80

def generalAdmissionTicketsBought : ℤ := 10
def VIPTicketsBought : ℤ := 3
def premiumTicketsBought : ℤ := 2

def generalAdmissionExcessThreshold : ℤ := 8
def VIPExcessThreshold : ℤ := 2
def premiumExcessThreshold : ℤ := 1

def generalAdmissionDiscountPercentage : ℤ := 3
def VIPDiscountPercentage : ℤ := 7
def premiumDiscountPercentage : ℤ := 10

-- Function to calculate the cost without discounts
def costWithoutDiscount : ℤ :=
  (generalAdmissionTicketsBought * generalAdmissionTicketPrice) +
  (VIPTicketsBought * VIPTicketPrice) +
  (premiumTicketsBought * premiumTicketPrice)

-- Function to calculate the total discount
def totalDiscount : ℤ :=
  let generalAdmissionDiscount := if generalAdmissionTicketsBought > generalAdmissionExcessThreshold then 
    (generalAdmissionTicketsBought - generalAdmissionExcessThreshold) * generalAdmissionTicketPrice * generalAdmissionDiscountPercentage / 100 else 0
  let VIPDiscount := if VIPTicketsBought > VIPExcessThreshold then 
    (VIPTicketsBought - VIPExcessThreshold) * VIPTicketPrice * VIPDiscountPercentage / 100 else 0
  let premiumDiscount := if premiumTicketsBought > premiumExcessThreshold then 
    (premiumTicketsBought - premiumExcessThreshold) * premiumTicketPrice * premiumDiscountPercentage / 100 else 0
  generalAdmissionDiscount + VIPDiscount + premiumDiscount

-- Function to calculate the total cost after discounts
def totalCostAfterDiscount : ℤ := costWithoutDiscount - totalDiscount

-- Proof statement
theorem MrBensonPaidCorrectAmount :
  totalCostAfterDiscount = 723 :=
by
  sorry

end MrBensonPaidCorrectAmount_l817_81772


namespace track_circumference_is_180_l817_81759

noncomputable def track_circumference : ℕ :=
  let brenda_first_meeting_dist := 120
  let sally_second_meeting_dist := 180
  let brenda_speed_factor : ℕ := 2
  -- circumference of the track
  let circumference := 3 * brenda_first_meeting_dist / brenda_speed_factor
  circumference

theorem track_circumference_is_180 :
  track_circumference = 180 :=
by 
  sorry

end track_circumference_is_180_l817_81759


namespace value_of_ratio_l817_81765

theorem value_of_ratio (x y : ℝ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 2 * x + 3 * y = 8) :
    (2 / x + 3 / y) = 25 / 8 := 
by
  sorry

end value_of_ratio_l817_81765


namespace smallest_positive_divisor_l817_81713

theorem smallest_positive_divisor
  (a b x₀ y₀ : ℤ)
  (h₀ : a ≠ 0 ∨ b ≠ 0)
  (h₁ : ∀ x y, a * x₀ + b * y₀ ≤ 0 ∨ a * x + b * y ≥ a * x₀ + b * y₀)
  (h₂ : 0 < a * x₀ + b * y₀):
  ∀ x y : ℤ, a * x₀ + b * y₀ ∣ a * x + b * y := 
sorry

end smallest_positive_divisor_l817_81713


namespace rate_of_mixed_oil_per_litre_l817_81789

theorem rate_of_mixed_oil_per_litre :
  let oil1_litres := 10
  let oil1_rate := 55
  let oil2_litres := 5
  let oil2_rate := 66
  let total_cost := oil1_litres * oil1_rate + oil2_litres * oil2_rate
  let total_volume := oil1_litres + oil2_litres
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 58.67 :=
by
  sorry

end rate_of_mixed_oil_per_litre_l817_81789


namespace find_k_l817_81753

theorem find_k (k α β : ℝ)
  (h1 : (∀ x : ℝ, x^2 - (k-1) * x - 3*k - 2 = 0 → x = α ∨ x = β))
  (h2 : α^2 + β^2 = 17) :
  k = 2 :=
sorry

end find_k_l817_81753


namespace max_distance_without_fuel_depots_l817_81743

def exploration_max_distance : ℕ :=
  360

-- Define the conditions
def cars_count : ℕ :=
  9

def full_tank_distance : ℕ :=
  40

def additional_gal_capacity : ℕ :=
  9

def total_gallons_per_car : ℕ :=
  1 + additional_gal_capacity

-- Define the distance calculation under the given constraints
theorem max_distance_without_fuel_depots (n : ℕ) (d_tank : ℕ) (d_add : ℕ) :
  ∀ (cars : ℕ), (cars = cars_count) →
  (d_tank = full_tank_distance) →
  (d_add = additional_gal_capacity) →
  ((cars * (1 + d_add)) * d_tank) / (2 * cars - 1) = exploration_max_distance :=
by
  intros _ hc ht ha
  rw [hc, ht, ha]
  -- Proof skipped
  sorry

end max_distance_without_fuel_depots_l817_81743


namespace william_washed_2_normal_cars_l817_81757

def time_spent_on_one_normal_car : Nat := 4 + 7 + 4 + 9

def time_spent_on_suv : Nat := 2 * time_spent_on_one_normal_car

def total_time_spent : Nat := 96

def time_spent_on_normal_cars : Nat := total_time_spent - time_spent_on_suv

def number_of_normal_cars : Nat := time_spent_on_normal_cars / time_spent_on_one_normal_car

theorem william_washed_2_normal_cars : number_of_normal_cars = 2 := by
  sorry

end william_washed_2_normal_cars_l817_81757


namespace percent_increase_l817_81708

theorem percent_increase (original new : ℕ) (h1 : original = 30) (h2 : new = 60) :
  ((new - original) / original) * 100 = 100 := 
by
  sorry

end percent_increase_l817_81708


namespace wang_hua_withdrawal_correct_l817_81777

noncomputable def wang_hua_withdrawal : ℤ :=
  let d : ℤ := 14
  let c : ℤ := 32
  -- The amount Wang Hua was supposed to withdraw in yuan
  (d * 100 + c)

theorem wang_hua_withdrawal_correct (d c : ℤ) :
  let initial_amount := (100 * d + c)
  let incorrect_amount := (100 * c + d)
  let amount_spent := 350
  let remaining_amount := incorrect_amount - amount_spent
  let expected_remaining := 2 * initial_amount
  remaining_amount = expected_remaining ∧ 
  d = 14 ∧ 
  c = 32 :=
by
  sorry

end wang_hua_withdrawal_correct_l817_81777


namespace maximum_value_of_f_l817_81766

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l817_81766


namespace find_original_number_l817_81745

theorem find_original_number (x y : ℕ) (h1 : x + y = 8) (h2 : 10 * y + x = 10 * x + y + 18) : 10 * x + y = 35 := 
sorry

end find_original_number_l817_81745


namespace sum_of_digits_in_binary_representation_of_315_l817_81786

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l817_81786


namespace stable_scores_l817_81709

theorem stable_scores (S_A S_B S_C S_D : ℝ) (hA : S_A = 2.2) (hB : S_B = 6.6) (hC : S_C = 7.4) (hD : S_D = 10.8) : 
  S_A ≤ S_B ∧ S_A ≤ S_C ∧ S_A ≤ S_D :=
by
  sorry

end stable_scores_l817_81709


namespace probability_zero_after_2017_days_l817_81775

-- Define the people involved
inductive Person
| Lunasa | Merlin | Lyrica
deriving DecidableEq, Inhabited

open Person

-- Define the initial state with each person having their own distinct hat
def initial_state : Person → Person
| Lunasa => Lunasa
| Merlin => Merlin
| Lyrica => Lyrica

-- Define a function that represents switching hats between two people
def switch_hats (p1 p2 : Person) (state : Person → Person) : Person → Person :=
  λ p => if p = p1 then state p2 else if p = p2 then state p1 else state p

-- Define a function to represent the state after n days (iterations)
def iter_switch_hats (n : ℕ) : Person → Person :=
  sorry -- This would involve implementing the iterative random switching

-- Proposition: The probability that after 2017 days, every person has their own hat back is 0
theorem probability_zero_after_2017_days :
  iter_switch_hats 2017 = initial_state → false :=
by
  sorry

end probability_zero_after_2017_days_l817_81775


namespace presidency_meeting_ways_l817_81711

theorem presidency_meeting_ways : 
  ∃ (ways : ℕ), ways = 4 * 6 * 3 * 225 := sorry

end presidency_meeting_ways_l817_81711


namespace new_percentage_of_managers_is_98_l817_81776

def percentage_of_managers (initial_employees : ℕ) (initial_percentage_managers : ℕ) (managers_leaving : ℕ) : ℕ :=
  let initial_managers := initial_percentage_managers * initial_employees / 100
  let remaining_managers := initial_managers - managers_leaving
  let remaining_employees := initial_employees - managers_leaving
  (remaining_managers * 100) / remaining_employees

theorem new_percentage_of_managers_is_98 :
  percentage_of_managers 500 99 250 = 98 :=
by
  sorry

end new_percentage_of_managers_is_98_l817_81776


namespace odd_function_value_l817_81727

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

theorem odd_function_value (a b : ℝ) (h1 : ∀ x, f x b = -f (-x) b) (h2 : a - 4 + 2 * a - 2 = 0) : f a b + f (2 * -a) b = 0 := by
  sorry

end odd_function_value_l817_81727


namespace vacation_fund_percentage_l817_81736

variable (s : ℝ) (vs : ℝ)
variable (d : ℝ)
variable (v : ℝ)

-- conditions:
-- 1. Jill's net monthly salary
#check (s = 3700)
-- 2. Jill's discretionary income is one fifth of her salary
#check (d = s / 5)
-- 3. Savings percentage
#check (0.20 * d)
-- 4. Eating out and socializing percentage
#check (0.35 * d)
-- 5. Gifts and charitable causes
#check (111)

-- Prove: 
theorem vacation_fund_percentage : 
  s = 3700 -> d = s / 5 -> 
  (v * d + 0.20 * d + 0.35 * d + 111 = d) -> 
  v = 222 / 740 :=
by
  sorry -- proof skipped

end vacation_fund_percentage_l817_81736


namespace dandelion_seeds_percentage_approx_29_27_l817_81762

/-
Mathematical conditions:
- Carla has the following set of plants and seeds per plant:
  - 6 sunflowers with 9 seeds each
  - 8 dandelions with 12 seeds each
  - 4 roses with 7 seeds each
  - 10 tulips with 15 seeds each.
- Calculate:
  - total seeds
  - percentage of seeds from dandelions
-/ 

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def num_roses : ℕ := 4
def num_tulips : ℕ := 10

def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12
def seeds_per_rose : ℕ := 7
def seeds_per_tulip : ℕ := 15

def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion
def total_rose_seeds : ℕ := num_roses * seeds_per_rose
def total_tulip_seeds : ℕ := num_tulips * seeds_per_tulip

def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds + total_rose_seeds + total_tulip_seeds

def percentage_dandelion_seeds : ℚ := (total_dandelion_seeds : ℚ) / total_seeds * 100

theorem dandelion_seeds_percentage_approx_29_27 : abs (percentage_dandelion_seeds - 29.27) < 0.01 :=
sorry

end dandelion_seeds_percentage_approx_29_27_l817_81762


namespace city_phone_number_remainder_l817_81768

theorem city_phone_number_remainder :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := sorry

end city_phone_number_remainder_l817_81768


namespace remainder_of_polynomial_l817_81769

theorem remainder_of_polynomial (x : ℤ) : 
  (x^4 - 1) * (x^2 - 1) % (x^2 + x + 1) = 3 := 
sorry

end remainder_of_polynomial_l817_81769


namespace proof_problem_l817_81771

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem_l817_81771


namespace find_product_of_roots_plus_one_l817_81784

-- Define the problem conditions
variables (x1 x2 : ℝ)
axiom sum_roots : x1 + x2 = 3
axiom prod_roots : x1 * x2 = 2

-- State the theorem corresponding to the proof problem
theorem find_product_of_roots_plus_one : (x1 + 1) * (x2 + 1) = 6 :=
by 
  sorry

end find_product_of_roots_plus_one_l817_81784


namespace radius_of_garden_outer_boundary_l817_81799

-- Definitions based on the conditions from the problem statement
def fountain_diameter : ℝ := 12
def garden_width : ℝ := 10

-- Question translated to a proof statement
theorem radius_of_garden_outer_boundary :
  (fountain_diameter / 2 + garden_width) = 16 := 
by 
  sorry

end radius_of_garden_outer_boundary_l817_81799


namespace weight_comparison_l817_81740

theorem weight_comparison :
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  average = 45 ∧ median = 25 ∧ average - median = 20 :=
by
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  have h1 : average = 45 := sorry
  have h2 : median = 25 := sorry
  have h3 : average - median = 20 := sorry
  exact ⟨h1, h2, h3⟩

end weight_comparison_l817_81740


namespace jeffrey_walks_to_mailbox_l817_81748

theorem jeffrey_walks_to_mailbox :
  ∀ (D total_steps net_gain_per_set steps_per_set sets net_gain : ℕ),
    steps_per_set = 3 ∧ 
    net_gain = 1 ∧ 
    total_steps = 330 ∧ 
    net_gain_per_set = net_gain ∧ 
    sets = total_steps / steps_per_set ∧ 
    D = sets * net_gain →
    D = 110 :=
by
  intro D total_steps net_gain_per_set steps_per_set sets net_gain
  intro h
  sorry

end jeffrey_walks_to_mailbox_l817_81748


namespace find_two_digit_integers_l817_81763

theorem find_two_digit_integers :
  ∃ (m n : ℕ), 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 ∧
    (∃ (a b : ℚ), a = m ∧ b = n ∧ (a + b) / 2 = b + a / 100) ∧ (m + n < 150) ∧ m = 50 ∧ n = 49 := 
by
  sorry

end find_two_digit_integers_l817_81763


namespace initial_men_count_l817_81728

theorem initial_men_count (x : ℕ) 
  (h1 : ∀ t : ℕ, t = 25 * x) 
  (h2 : ∀ t : ℕ, t = 12 * 75) : 
  x = 36 := 
by
  sorry

end initial_men_count_l817_81728


namespace cement_percentage_of_second_concrete_l817_81700

theorem cement_percentage_of_second_concrete 
  (total_weight : ℝ) (final_percentage : ℝ) (partial_weight : ℝ) 
  (percentage_first_concrete : ℝ) :
  total_weight = 4500 →
  final_percentage = 0.108 →
  partial_weight = 1125 →
  percentage_first_concrete = 0.108 →
  ∃ percentage_second_concrete : ℝ, 
    percentage_second_concrete = 0.324 :=
by
  intros h1 h2 h3 h4
  let total_cement := total_weight * final_percentage
  let cement_first_concrete := partial_weight * percentage_first_concrete
  let cement_second_concrete := total_cement - cement_first_concrete
  let percentage_second_concrete := cement_second_concrete / partial_weight
  use percentage_second_concrete
  sorry

end cement_percentage_of_second_concrete_l817_81700


namespace largest_angle_in_triangle_l817_81792

theorem largest_angle_in_triangle (k : ℕ) (h : 3 * k + 4 * k + 5 * k = 180) : 5 * k = 75 :=
  by
  -- This is a placeholder for the proof, which is not required as per instructions
  sorry

end largest_angle_in_triangle_l817_81792


namespace general_term_a_l817_81774

noncomputable def S (n : ℕ) : ℤ := 3^n - 2

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * 3^(n - 1)

theorem general_term_a (n : ℕ) (hn : n > 0) : a n = if n = 1 then 1 else 2 * 3^(n - 1) := by
  -- Proof goes here
  sorry

end general_term_a_l817_81774


namespace chromium_percentage_l817_81720

noncomputable def chromium_percentage_in_new_alloy 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) : ℝ :=
    (((chromium_percentage_first * weight_first / 100) + (chromium_percentage_second * weight_second / 100)) 
    / (weight_first + weight_second)) * 100

theorem chromium_percentage 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) 
    (h1 : chromium_percentage_first = 10) 
    (h2 : weight_first = 15) 
    (h3 : chromium_percentage_second = 8) 
    (h4 : weight_second = 35) :
    chromium_percentage_in_new_alloy chromium_percentage_first weight_first chromium_percentage_second weight_second = 8.6 :=
by 
  rw [h1, h2, h3, h4]
  simp [chromium_percentage_in_new_alloy]
  norm_num


end chromium_percentage_l817_81720


namespace isosceles_triangle_perimeter_l817_81787

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2))
  (h2 : ∃ x y z : ℕ, (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  a + a + b = 12 :=
by
  sorry

end isosceles_triangle_perimeter_l817_81787


namespace least_number_to_add_l817_81730

theorem least_number_to_add (n d : ℕ) (h₁ : n = 1054) (h₂ : d = 23) : ∃ x, (n + x) % d = 0 ∧ x = 4 := by
  sorry

end least_number_to_add_l817_81730


namespace olya_candies_l817_81729

theorem olya_candies (P M T O : ℕ) (h1 : P + M + T + O = 88) (h2 : 1 ≤ P) (h3 : 1 ≤ M) (h4 : 1 ≤ T) (h5 : 1 ≤ O) (h6 : M + T = 57) (h7 : P > M) (h8 : P > T) (h9 : P > O) : O = 1 :=
by
  sorry

end olya_candies_l817_81729


namespace expected_value_min_of_subset_l817_81760

noncomputable def expected_value_min (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : ℚ :=
  (n + 1) / (r + 1)

theorem expected_value_min_of_subset (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  expected_value_min n r h = (n + 1) / (r + 1) :=
sorry

end expected_value_min_of_subset_l817_81760


namespace problem_sufficient_necessary_condition_l817_81731

open Set

variable {x : ℝ}

def P (x : ℝ) : Prop := abs (x - 2) < 3
def Q (x : ℝ) : Prop := x^2 - 8 * x + 15 < 0

theorem problem_sufficient_necessary_condition :
    (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
by
  sorry

end problem_sufficient_necessary_condition_l817_81731


namespace expression_comparison_l817_81773

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) :
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  (exprI = exprII ∨ exprI = exprIII ∨ exprII = exprIII ∨ 
   (exprI > exprII ∧ exprI > exprIII) ∨
   (exprII > exprI ∧ exprII > exprIII) ∨
   (exprIII > exprI ∧ exprIII > exprII)) ∧
  ¬((exprI > exprII ∧ exprI > exprIII) ∨
    (exprII > exprI ∧ exprII > exprIII) ∨
    (exprIII > exprI ∧ exprIII > exprII)) :=
by
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  sorry

end expression_comparison_l817_81773


namespace jill_present_age_l817_81710

-- Define the main proof problem
theorem jill_present_age (H J : ℕ) (h1 : H + J = 33) (h2 : H - 6 = 2 * (J - 6)) : J = 13 :=
by
  sorry

end jill_present_age_l817_81710


namespace find_m_for_perfect_square_trinomial_l817_81754

theorem find_m_for_perfect_square_trinomial :
  ∃ m : ℤ, (∀ (x y : ℝ), (9 * x^2 + m * x * y + 16 * y^2 = (3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (3 * x - 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x + 4 * y)^2) ∨ 
                   (9 * x^2 + m * x * y + 16 * y^2 = (-3 * x - 4 * y)^2)) ↔ 
          (m = 24 ∨ m = -24) := 
by
  sorry

end find_m_for_perfect_square_trinomial_l817_81754


namespace circle_diameter_l817_81781
open Real

theorem circle_diameter (A : ℝ) (hA : A = 50.26548245743669) : ∃ d : ℝ, d = 8 :=
by
  sorry

end circle_diameter_l817_81781


namespace bus_trip_speed_l817_81782

theorem bus_trip_speed :
  ∃ v : ℝ, v > 0 ∧ (660 / v - 1 = 660 / (v + 5)) ∧ v = 55 :=
by
  sorry

end bus_trip_speed_l817_81782


namespace image_of_center_l817_81726

-- Define the initial coordinates
def initial_coordinate : ℝ × ℝ := (-3, 4)

-- Function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to translate a point up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Definition of the final coordinate
noncomputable def final_coordinate : ℝ × ℝ :=
  translate_up (reflect_x initial_coordinate) 5

-- Theorem stating the final coordinate after transformations
theorem image_of_center : final_coordinate = (-3, 1) := by
  -- Proof is omitted
  sorry

end image_of_center_l817_81726


namespace car_dealership_sales_l817_81701

theorem car_dealership_sales (x : ℕ)
  (h1 : 5 * x = 30 * 8)
  (h2 : 30 + x = 78) : 
  x = 48 :=
sorry

end car_dealership_sales_l817_81701


namespace cone_base_circumference_l817_81716

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (C : ℝ) : 
  r = 5 → θ = 300 → C = (θ / 360) * (2 * Real.pi * r) → C = (25 / 3) * Real.pi :=
by
  sorry

end cone_base_circumference_l817_81716


namespace even_function_of_shift_sine_l817_81715

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (x - 6)^2 * Real.sin (ω * x)

theorem even_function_of_shift_sine :
  ∃ ω : ℝ, (∀ x : ℝ, f x ω = f (-x) ω) → ω = π / 4 :=
by
  sorry

end even_function_of_shift_sine_l817_81715


namespace problem_a_b_l817_81756

theorem problem_a_b (a b : ℝ) (h₁ : a + b = 10) (h₂ : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end problem_a_b_l817_81756


namespace employee_monthly_wage_l817_81732

theorem employee_monthly_wage 
(revenue : ℝ)
(tax_rate : ℝ)
(marketing_rate : ℝ)
(operational_cost_rate : ℝ)
(wage_rate : ℝ)
(num_employees : ℕ)
(h_revenue : revenue = 400000)
(h_tax_rate : tax_rate = 0.10)
(h_marketing_rate : marketing_rate = 0.05)
(h_operational_cost_rate : operational_cost_rate = 0.20)
(h_wage_rate : wage_rate = 0.15)
(h_num_employees : num_employees = 10) :
(revenue * (1 - tax_rate) * (1 - marketing_rate) * (1 - operational_cost_rate) * wage_rate / num_employees = 4104) :=
by
  sorry

end employee_monthly_wage_l817_81732


namespace george_money_left_after_donations_and_groceries_l817_81796

def monthly_income : ℕ := 240
def donation (income : ℕ) : ℕ := income / 2
def post_donation_money (income : ℕ) : ℕ := income - donation income
def groceries_cost : ℕ := 20
def money_left (income : ℕ) : ℕ := post_donation_money income - groceries_cost

theorem george_money_left_after_donations_and_groceries :
  money_left monthly_income = 100 :=
by
  sorry

end george_money_left_after_donations_and_groceries_l817_81796


namespace tom_swim_time_l817_81785

theorem tom_swim_time (t : ℝ) :
  (2 * t + 4 * t = 12) → t = 2 :=
by
  intro h
  have eq1 : 6 * t = 12 := by linarith
  linarith

end tom_swim_time_l817_81785


namespace largest_perfect_square_factor_of_1764_l817_81718

theorem largest_perfect_square_factor_of_1764 : ∃ m, m * m = 1764 ∧ ∀ n, n * n ∣ 1764 → n * n ≤ 1764 :=
by
  sorry

end largest_perfect_square_factor_of_1764_l817_81718


namespace arc_length_l817_81761

/-- Given a circle with a radius of 5 cm and a sector area of 11.25 cm², 
prove that the length of the arc forming the sector is 4.5 cm. --/
theorem arc_length (r : ℝ) (A : ℝ) (θ : ℝ) (arc_length : ℝ) 
  (h_r : r = 5) 
  (h_A : A = 11.25) 
  (h_area_formula : A = (θ / (2 * π)) * π * r ^ 2) 
  (h_arc_length_formula : arc_length = r * θ) :
  arc_length = 4.5 :=
sorry

end arc_length_l817_81761


namespace incorrect_statement_A_l817_81749

/-- Let prob_beijing be the probability of rainfall in Beijing and prob_shanghai be the probability
of rainfall in Shanghai. We assert that statement (A) which claims "It is certain to rain in Beijing today, 
while it is certain not to rain in Shanghai" is incorrect given the probabilities. 
-/
theorem incorrect_statement_A (prob_beijing prob_shanghai : ℝ) 
  (h_beijing : prob_beijing = 0.8)
  (h_shanghai : prob_shanghai = 0.2)
  (statement_A : ¬ (prob_beijing = 1 ∧ prob_shanghai = 0)) : 
  true := 
sorry

end incorrect_statement_A_l817_81749
