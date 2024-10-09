import Mathlib

namespace volume_percentage_error_l68_6825

theorem volume_percentage_error (L W H : ℝ) (hL : L > 0) (hW : W > 0) (hH : H > 0) :
  let V_true := L * W * H
  let L_meas := 1.08 * L
  let W_meas := 1.12 * W
  let H_meas := 1.05 * H
  let V_calc := L_meas * W_meas * H_meas
  let percentage_error := ((V_calc - V_true) / V_true) * 100
  percentage_error = 25.424 :=
by
  sorry

end volume_percentage_error_l68_6825


namespace sum_three_consecutive_odd_integers_l68_6897

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l68_6897


namespace gcd_459_357_is_51_l68_6854

-- Define the problem statement
theorem gcd_459_357_is_51 : Nat.gcd 459 357 = 51 :=
by
  -- Proof here
  sorry

end gcd_459_357_is_51_l68_6854


namespace determine_y_minus_x_l68_6878

theorem determine_y_minus_x (x y : ℝ) (h1 : x + y = 360) (h2 : x / y = 3 / 5) : y - x = 90 := sorry

end determine_y_minus_x_l68_6878


namespace problem_statement_l68_6882

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement_l68_6882


namespace angle_of_inclination_45_l68_6842

def plane (x y z : ℝ) : Prop := (x = y) ∧ (y = z)
def image_planes (x y : ℝ) : Prop := (x = 45 ∧ y = 45)

theorem angle_of_inclination_45 (t₁₂ : ℝ) :
  ∃ θ: ℝ, (plane t₁₂ t₁₂ t₁₂ → image_planes 45 45 → θ = 45) :=
sorry

end angle_of_inclination_45_l68_6842


namespace functional_equation_solution_l68_6824

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by
  intro h
  sorry

end functional_equation_solution_l68_6824


namespace quadratic_distinct_roots_l68_6821

theorem quadratic_distinct_roots (m : ℝ) : 
  ((m - 2) * x ^ 2 + 2 * x + 1 = 0) → (m < 3 ∧ m ≠ 2) :=
by
  sorry

end quadratic_distinct_roots_l68_6821


namespace value_of_expression_l68_6841

theorem value_of_expression (x y : ℤ) (hx : x = -5) (hy : y = 8) : 2 * (x - y) ^ 2 - x * y = 378 :=
by
  rw [hx, hy]
  -- The proof goes here.
  sorry

end value_of_expression_l68_6841


namespace equilateral_triangle_of_condition_l68_6870

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 0) :
  a = b ∧ b = c := 
sorry

end equilateral_triangle_of_condition_l68_6870


namespace absolute_inequality_l68_6879

theorem absolute_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := 
sorry

end absolute_inequality_l68_6879


namespace campaign_donation_ratio_l68_6806

theorem campaign_donation_ratio (max_donation : ℝ) 
  (total_money : ℝ) 
  (percent_donations : ℝ) 
  (num_max_donors : ℕ) 
  (half_max_donation : ℝ) 
  (total_raised : ℝ) 
  (half_donation : ℝ) :
  total_money = total_raised * percent_donations →
  half_donation = max_donation / 2 →
  half_max_donation = num_max_donors * max_donation →
  total_money - half_max_donation = 1500 * half_donation →
  (1500 : ℝ) / (num_max_donors : ℝ) = 3 :=
sorry

end campaign_donation_ratio_l68_6806


namespace lydia_current_age_l68_6899

def years_for_apple_tree_to_bear_fruit : ℕ := 7
def lydia_age_when_planted_tree : ℕ := 4
def lydia_age_when_eats_apple : ℕ := 11

theorem lydia_current_age 
  (h : lydia_age_when_eats_apple - lydia_age_when_planted_tree = years_for_apple_tree_to_bear_fruit) :
  lydia_age_when_eats_apple = 11 := 
by
  sorry

end lydia_current_age_l68_6899


namespace tan_difference_identity_l68_6808

theorem tan_difference_identity {α : ℝ} (h : Real.tan α = 4 * Real.sin (7 * Real.pi / 3)) :
  Real.tan (α - Real.pi / 3) = Real.sqrt 3 / 7 := 
sorry

end tan_difference_identity_l68_6808


namespace common_difference_d_l68_6823

theorem common_difference_d (a_1 d : ℝ) (h1 : a_1 + 2 * d = 4) (h2 : 9 * a_1 + 36 * d = 18) : d = -1 :=
by sorry

end common_difference_d_l68_6823


namespace find_number_of_ducks_l68_6891

variable {D H : ℕ}

-- Definition of the conditions
def total_animals (D H : ℕ) : Prop := D + H = 11
def total_legs (D H : ℕ) : Prop := 2 * D + 4 * H = 30
def number_of_ducks (D : ℕ) : Prop := D = 7

-- Lean statement for the proof problem
theorem find_number_of_ducks (D H : ℕ) (h1 : total_animals D H) (h2 : total_legs D H) : number_of_ducks D :=
by
  sorry

end find_number_of_ducks_l68_6891


namespace relationship_among_p_q_a_b_l68_6857

open Int

variables (a b p q : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = p) (h3 : Nat.lcm a b = q)

theorem relationship_among_p_q_a_b : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end relationship_among_p_q_a_b_l68_6857


namespace handshakes_4_handshakes_n_l68_6827

-- Defining the number of handshakes for n people
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

-- Proving that the number of handshakes for 4 people is 6
theorem handshakes_4 : handshakes 4 = 6 := by
  sorry

-- Proving that the number of handshakes for n people is (n * (n - 1)) / 2
theorem handshakes_n (n : ℕ) : handshakes n = (n * (n - 1)) / 2 := by 
  sorry

end handshakes_4_handshakes_n_l68_6827


namespace perpendicular_vectors_x_value_l68_6810

theorem perpendicular_vectors_x_value
  (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (1, -2)) (hb : b = (-3, x))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -3 / 2 := by
  sorry

end perpendicular_vectors_x_value_l68_6810


namespace James_delivers_2565_bags_in_a_week_l68_6876

noncomputable def total_bags_delivered_in_a_week
  (days_15_bags : ℕ)
  (trips_per_day_15_bags : ℕ)
  (bags_per_trip_15 : ℕ)
  (days_20_bags : ℕ)
  (trips_per_day_20_bags : ℕ)
  (bags_per_trip_20 : ℕ) : ℕ :=
  (days_15_bags * trips_per_day_15_bags * bags_per_trip_15) + (days_20_bags * trips_per_day_20_bags * bags_per_trip_20)

theorem James_delivers_2565_bags_in_a_week :
  total_bags_delivered_in_a_week 3 25 15 4 18 20 = 2565 :=
by
  sorry

end James_delivers_2565_bags_in_a_week_l68_6876


namespace sum_divisible_by_10_l68_6818

-- Define the problem statement
theorem sum_divisible_by_10 {n : ℕ} : (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 10 = 0 ↔ ∃ t : ℕ, n = 5 * t + 1 :=
by sorry

end sum_divisible_by_10_l68_6818


namespace record_expenditure_l68_6890

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end record_expenditure_l68_6890


namespace school_should_purchase_bookshelves_l68_6804

theorem school_should_purchase_bookshelves
  (x : ℕ)
  (h₁ : x ≥ 20)
  (cost_A : ℕ := 20 * 300 + 100 * (x - 20))
  (cost_B : ℕ := (20 * 300 + 100 * x) * 80 / 100)
  (h₂ : cost_A = cost_B) : x = 40 :=
by sorry

end school_should_purchase_bookshelves_l68_6804


namespace completing_the_square_solution_l68_6805

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l68_6805


namespace find_tangent_line_l68_6812

def curve := fun x : ℝ => x^3 + 2 * x + 1
def tangent_point := 1
def tangent_line (x y : ℝ) := 5 * x - y - 1 = 0

theorem find_tangent_line :
  tangent_line tangent_point (curve tangent_point) :=
by
  sorry

end find_tangent_line_l68_6812


namespace cards_thrown_away_l68_6893

theorem cards_thrown_away (h1 : 3 * (52 / 2) + 3 * 52 - 200 = 34) : 34 = 34 :=
by sorry

end cards_thrown_away_l68_6893


namespace packing_big_boxes_l68_6896

def total_items := 8640
def items_per_small_box := 12
def small_boxes_per_big_box := 6

def num_big_boxes (total_items items_per_small_box small_boxes_per_big_box : ℕ) : ℕ :=
  (total_items / items_per_small_box) / small_boxes_per_big_box

theorem packing_big_boxes : num_big_boxes total_items items_per_small_box small_boxes_per_big_box = 120 :=
by
  sorry

end packing_big_boxes_l68_6896


namespace total_number_of_people_l68_6839

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people_l68_6839


namespace distance_range_l68_6845

theorem distance_range (A_school_distance : ℝ) (B_school_distance : ℝ) (x : ℝ)
  (hA : A_school_distance = 3) (hB : B_school_distance = 2) :
  1 ≤ x ∧ x ≤ 5 :=
sorry

end distance_range_l68_6845


namespace sequence_expression_l68_6873

theorem sequence_expression (a : ℕ → ℕ) (h₀ : a 1 = 33) (h₁ : ∀ n, a (n + 1) - a n = 2 * n) : 
  ∀ n, a n = n^2 - n + 33 :=
by
  sorry

end sequence_expression_l68_6873


namespace hyperbola_asymptotes_l68_6830

theorem hyperbola_asymptotes (x y : ℝ) : x^2 - 4 * y^2 = -1 → (x = 2 * y) ∨ (x = -2 * y) := 
by
  intro h
  sorry

end hyperbola_asymptotes_l68_6830


namespace compare_logs_l68_6881

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

theorem compare_logs : b > c ∧ c > a :=
by
  sorry

end compare_logs_l68_6881


namespace determine_initial_sum_l68_6822

def initial_sum_of_money (P r : ℝ) : Prop :=
  (600 = P + 2 * P * r) ∧ (700 = P + 2 * P * (r + 0.1))

theorem determine_initial_sum (P r : ℝ) (h : initial_sum_of_money P r) : P = 500 :=
by
  cases h with
  | intro h1 h2 =>
    sorry

end determine_initial_sum_l68_6822


namespace intersection_of_M_and_N_l68_6856

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- The theorem to be proved
theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_M_and_N_l68_6856


namespace geometric_mean_problem_l68_6836

theorem geometric_mean_problem
  (a : Nat) (a1 : Nat) (a8 : Nat) (r : Rat) 
  (h1 : a1 = 6) (h2 : a8 = 186624) 
  (h3 : a8 = a1 * r^7) 
  : a = a1 * r^3 → a = 1296 := 
by
  sorry

end geometric_mean_problem_l68_6836


namespace minimum_value_of_expression_l68_6832

variable (a b c : ℝ)

noncomputable def expression (a b c : ℝ) := (a + b) / c + (a + c) / b + (b + c) / a

theorem minimum_value_of_expression (hp1 : 0 < a) (hp2 : 0 < b) (hp3 : 0 < c) (h1 : a = 2 * b) (h2 : a = 2 * c) :
  expression a b c = 9.25 := 
sorry

end minimum_value_of_expression_l68_6832


namespace johns_leisure_travel_miles_per_week_l68_6888

-- Define the given conditions
def mpg : Nat := 30
def work_round_trip_miles : Nat := 20 * 2  -- 20 miles to work + 20 miles back home
def work_days_per_week : Nat := 5
def weekly_fuel_usage_gallons : Nat := 8

-- Define the property to prove
theorem johns_leisure_travel_miles_per_week :
  let work_miles_per_week := work_round_trip_miles * work_days_per_week
  let total_possible_miles := weekly_fuel_usage_gallons * mpg
  let leisure_miles := total_possible_miles - work_miles_per_week
  leisure_miles = 40 :=
by
  sorry

end johns_leisure_travel_miles_per_week_l68_6888


namespace solution_set_of_inequality_l68_6848

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x^2 + 2 < x} = {x : ℝ | x < -2 / 3} ∪ {x : ℝ | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l68_6848


namespace minimum_area_isosceles_trapezoid_l68_6820

theorem minimum_area_isosceles_trapezoid (r x a d : ℝ) (h_circumscribed : a + d = 2 * x) (h_minimal : x ≥ 2 * r) :
  4 * r^2 ≤ (a + d) * r :=
by sorry

end minimum_area_isosceles_trapezoid_l68_6820


namespace option_B_correct_l68_6862

theorem option_B_correct (a b : ℝ) (h : a < b) : a^3 < b^3 := sorry

end option_B_correct_l68_6862


namespace cricket_target_runs_l68_6849

def target_runs (first_10_overs_run_rate remaining_40_overs_run_rate : ℝ) : ℝ :=
  10 * first_10_overs_run_rate + 40 * remaining_40_overs_run_rate

theorem cricket_target_runs : target_runs 4.2 6 = 282 := by
  sorry

end cricket_target_runs_l68_6849


namespace cone_base_circumference_l68_6838

theorem cone_base_circumference (radius : ℝ) (angle : ℝ) (c_base : ℝ) :
  radius = 6 ∧ angle = 180 ∧ c_base = 6 * Real.pi →
  (c_base = (angle / 360) * (2 * Real.pi * radius)) :=
by
  intros h
  rcases h with ⟨h_radius, h_angle, h_c_base⟩
  rw [h_radius, h_angle]
  norm_num
  sorry

end cone_base_circumference_l68_6838


namespace factorization_problem1_factorization_problem2_l68_6887

variables {a b x y : ℝ}

theorem factorization_problem1 (a b x y : ℝ) : a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) :=
by sorry

theorem factorization_problem2 (a b : ℝ) : a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 :=
by sorry

end factorization_problem1_factorization_problem2_l68_6887


namespace calculate_lunch_break_duration_l68_6833

noncomputable def paula_rate (p : ℝ) : Prop := p > 0
noncomputable def helpers_rate (h : ℝ) : Prop := h > 0
noncomputable def apprentice_rate (a : ℝ) : Prop := a > 0
noncomputable def lunch_break_duration (L : ℝ) : Prop := L >= 0

-- Monday's work equation
noncomputable def monday_work (p h a L : ℝ) (monday_work_done : ℝ) :=
  0.6 = (9 - L) * (p + h + a)

-- Tuesday's work equation
noncomputable def tuesday_work (h a L : ℝ) (tuesday_work_done : ℝ) :=
  0.3 = (7 - L) * (h + a)

-- Wednesday's work equation
noncomputable def wednesday_work (p a L : ℝ) (wednesday_work_done : ℝ) :=
  0.1 = (1.2 - L) * (p + a)

-- Final proof statement
theorem calculate_lunch_break_duration (p h a L : ℝ)
  (H1 : paula_rate p)
  (H2 : helpers_rate h)
  (H3 : apprentice_rate a)
  (H4 : lunch_break_duration L)
  (H5 : monday_work p h a L 0.6)
  (H6 : tuesday_work h a L 0.3)
  (H7 : wednesday_work p a L 0.1) :
  L = 1.4 :=
sorry

end calculate_lunch_break_duration_l68_6833


namespace proof_op_l68_6859

def op (A B : ℕ) : ℕ := (A * B) / 2

theorem proof_op (a b c : ℕ) : op (op 4 6) 9 = 54 := by
  sorry

end proof_op_l68_6859


namespace least_sum_p_q_r_l68_6801

theorem least_sum_p_q_r (p q r : ℕ) (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (h : 17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1)) : p + q + r = 290 :=
  sorry

end least_sum_p_q_r_l68_6801


namespace car_speed_second_hour_l68_6803

theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (avg_speed : ℝ)
  (hours : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_first_hour : ℝ)
  (distance_second_hour : ℝ) :
  speed_first_hour = 90 →
  avg_speed = 75 →
  hours = 2 →
  total_time = hours →
  total_distance = avg_speed * total_time →
  distance_first_hour = speed_first_hour * 1 →
  distance_second_hour = total_distance - distance_first_hour →
  distance_second_hour / 1 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end car_speed_second_hour_l68_6803


namespace slightly_used_crayons_count_l68_6826

-- Definitions
def total_crayons := 120
def new_crayons := total_crayons * (1/3)
def broken_crayons := total_crayons * (20/100)
def slightly_used_crayons := total_crayons - new_crayons - broken_crayons

-- Theorem statement
theorem slightly_used_crayons_count :
  slightly_used_crayons = 56 :=
by
  sorry

end slightly_used_crayons_count_l68_6826


namespace gcd_of_12347_and_9876_l68_6814

theorem gcd_of_12347_and_9876 : Nat.gcd 12347 9876 = 7 :=
by
  sorry

end gcd_of_12347_and_9876_l68_6814


namespace solve_equation_l68_6894

theorem solve_equation (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2)
  (h₃ : (3 * x + 6)/(x^2 + 5 * x + 6) = (3 - x)/(x - 2)) :
  x = 3 ∨ x = -3 :=
sorry

end solve_equation_l68_6894


namespace goods_train_speed_l68_6869

theorem goods_train_speed
  (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
  (h_train_length : length_train = 250.0416)
  (h_platform_length : length_platform = 270)
  (h_time : time_seconds = 26) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
    sorry

end goods_train_speed_l68_6869


namespace coprime_repeating_decimal_sum_l68_6800

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l68_6800


namespace activity_probability_l68_6867

noncomputable def total_basic_events : ℕ := 3^4
noncomputable def favorable_events : ℕ := Nat.choose 4 2 * Nat.factorial 3

theorem activity_probability :
  (favorable_events : ℚ) / total_basic_events = 4 / 9 :=
by
  sorry

end activity_probability_l68_6867


namespace find_number_l68_6868

theorem find_number (N : ℝ) (h1 : (4/5) * (3/8) * N = some_number)
                    (h2 : 2.5 * N = 199.99999999999997) :
  N = 79.99999999999999 := 
sorry

end find_number_l68_6868


namespace solve_4_times_3_l68_6807

noncomputable def custom_operation (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem solve_4_times_3 : custom_operation 4 3 = 13 := by
  -- Here the proof would be provided, for now we use sorry
  sorry

end solve_4_times_3_l68_6807


namespace overall_percentage_loss_l68_6874

noncomputable def original_price : ℝ := 100
noncomputable def increased_price : ℝ := original_price * 1.36
noncomputable def first_discount_price : ℝ := increased_price * 0.90
noncomputable def second_discount_price : ℝ := first_discount_price * 0.85
noncomputable def third_discount_price : ℝ := second_discount_price * 0.80
noncomputable def final_price_with_tax : ℝ := third_discount_price * 1.05
noncomputable def percentage_change : ℝ := ((final_price_with_tax - original_price) / original_price) * 100

theorem overall_percentage_loss : percentage_change = -12.6064 :=
by
  sorry

end overall_percentage_loss_l68_6874


namespace sugar_per_batch_l68_6852

variable (S : ℝ)

theorem sugar_per_batch :
  (8 * (4 + S) = 44) → (S = 1.5) :=
by
  intro h
  sorry

end sugar_per_batch_l68_6852


namespace largest_D_l68_6885

theorem largest_D (D : ℝ) : (∀ x y : ℝ, x^2 + 2 * y^2 + 3 ≥ D * (3 * x + 4 * y)) → D ≤ Real.sqrt (12 / 17) :=
by
  sorry

end largest_D_l68_6885


namespace leak_drain_time_l68_6880

theorem leak_drain_time (P L : ℕ → ℕ) (H1 : ∀ t, P t = 1 / 2) (H2 : ∀ t, P t - L t = 1 / 3) : 
  (1 / L 1) = 6 :=
by
  sorry

end leak_drain_time_l68_6880


namespace johns_elevation_after_descent_l68_6829

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l68_6829


namespace lowest_dropped_score_l68_6860

theorem lowest_dropped_score (A B C D : ℕ)
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : (A + B + C) / 3 = 55) :
  D = 35 :=
by
  sorry

end lowest_dropped_score_l68_6860


namespace complex_modulus_z_l68_6817

-- Define the complex number z with given conditions
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I + Complex.I

-- State the theorem to be proven
theorem complex_modulus_z : Complex.abs z = Real.sqrt 2 := 
sorry

end complex_modulus_z_l68_6817


namespace frac_eq_l68_6834

def my_at (a b : ℕ) := a * b + b^2
def my_hash (a b : ℕ) := a^2 + b + a * b^2

theorem frac_eq : my_at 4 3 / my_hash 4 3 = 21 / 55 :=
by
  sorry

end frac_eq_l68_6834


namespace find_value_l68_6875

variable (y : ℝ) (Q : ℝ)
axiom condition : 5 * (3 * y + 7 * Real.pi) = Q

theorem find_value : 10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by
  sorry

end find_value_l68_6875


namespace service_fee_calculation_l68_6843

-- Problem definitions based on conditions
def cost_food : ℝ := 50
def tip : ℝ := 5
def total_spent : ℝ := 61
def service_fee_percentage (x : ℝ) : Prop := x = (12 / 50) * 100

-- The main statement to be proven, showing that the service fee percentage is 24%
theorem service_fee_calculation : service_fee_percentage 24 :=
by {
  sorry
}

end service_fee_calculation_l68_6843


namespace exists_a_solution_iff_l68_6811

theorem exists_a_solution_iff (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1 / 4 := 
by 
  sorry

end exists_a_solution_iff_l68_6811


namespace solve_for_x_l68_6837

theorem solve_for_x (a b c x : ℝ) (h : x^2 + b^2 + c = (a + x)^2) : 
  x = (b^2 + c - a^2) / (2 * a) :=
by sorry

end solve_for_x_l68_6837


namespace largest_possible_median_l68_6871

theorem largest_possible_median 
  (l : List ℕ)
  (h_l : l = [4, 5, 3, 7, 9, 6])
  (h_pos : ∀ n ∈ l, 0 < n)
  (additional : List ℕ)
  (h_additional_pos : ∀ n ∈ additional, 0 < n)
  (h_length : l.length + additional.length = 9) : 
  ∃ median, median = 7 :=
by
  sorry

end largest_possible_median_l68_6871


namespace difference_between_neutrons_and_electrons_l68_6853

def proton_number : Nat := 118
def mass_number : Nat := 293

def number_of_neutrons : Nat := mass_number - proton_number
def number_of_electrons : Nat := proton_number

theorem difference_between_neutrons_and_electrons :
  (number_of_neutrons - number_of_electrons) = 57 := by
  sorry

end difference_between_neutrons_and_electrons_l68_6853


namespace relationship_y1_y2_l68_6895

theorem relationship_y1_y2 (k y1 y2 : ℝ) 
  (h1 : y1 = (k^2 + 1) * (-3) - 5) 
  (h2 : y2 = (k^2 + 1) * 4 - 5) : 
  y1 < y2 :=
sorry

end relationship_y1_y2_l68_6895


namespace sum_first_10_terms_arithmetic_seq_l68_6835

theorem sum_first_10_terms_arithmetic_seq (a : ℕ → ℤ) (h : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9) :
  ∃ S, S = 10 * (a 4 + a 7) / 2 ∧ (S = 15 ∨ S = -15) := 
by
  sorry

end sum_first_10_terms_arithmetic_seq_l68_6835


namespace exam_score_impossible_l68_6861

theorem exam_score_impossible (x y : ℕ) : 
  (5 * x + y = 97) ∧ (x + y ≤ 20) → false :=
by
  sorry

end exam_score_impossible_l68_6861


namespace remainder_of_expression_l68_6892

theorem remainder_of_expression (x y u v : ℕ) (h : x = u * y + v) (Hv : 0 ≤ v ∧ v < y) :
  (if v + 2 < y then (x + 3 * u * y + 2) % y = v + 2
   else (x + 3 * u * y + 2) % y = v + 2 - y) :=
by sorry

end remainder_of_expression_l68_6892


namespace total_jokes_l68_6851

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l68_6851


namespace max_single_student_books_l68_6850

-- Definitions and conditions
variable (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ)
variable (total_avg_books_per_student : ℕ)

-- Given data
def given_data : Prop :=
  total_students = 20 ∧ no_books = 2 ∧ one_book = 8 ∧
  two_books = 3 ∧ total_avg_books_per_student = 2

-- Maximum number of books any single student could borrow
theorem max_single_student_books (total_students no_books one_book two_books total_avg_books_per_student : ℕ) 
  (h : given_data total_students no_books one_book two_books total_avg_books_per_student) : 
  ∃ max_books_borrowed, max_books_borrowed = 8 :=
by
  sorry

end max_single_student_books_l68_6850


namespace Q_subset_P_l68_6858

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Statement to prove the relationship between P and Q
theorem Q_subset_P : Q ⊆ P :=
by
  sorry

end Q_subset_P_l68_6858


namespace solution_set_l68_6883

theorem solution_set:
  (∃ x y : ℝ, x - y = 0 ∧ x^2 + y = 2) ↔ (∃ x y : ℝ, (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -2)) :=
by
  sorry

end solution_set_l68_6883


namespace parabola_equation_l68_6846

theorem parabola_equation (a : ℝ) : 
(∀ x y : ℝ, y = x → y = a * x^2)
∧ (∃ P : ℝ × ℝ, P = (2, 2) ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
  → A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ y₁ = x₁ ∧ y₂ = x₂ ∧ x₂ = x₁ → 
  ∃ f : ℝ × ℝ, f.fst ≠ 0 ∧ f.snd = 0) →
  a = (1 : ℝ) / 7 := 
sorry

end parabola_equation_l68_6846


namespace number_with_150_quarters_is_37_point_5_l68_6840

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end number_with_150_quarters_is_37_point_5_l68_6840


namespace larger_integer_is_30_l68_6816

-- Define the problem statement using the given conditions
theorem larger_integer_is_30 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h1 : a / b = 5 / 2) (h2 : a * b = 360) :
  max a b = 30 :=
sorry

end larger_integer_is_30_l68_6816


namespace number_of_factors_60_l68_6809

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l68_6809


namespace beth_speed_l68_6815

noncomputable def beth_average_speed (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ) : ℚ :=
  let jerry_time_hours := jerry_time_minutes / 60
  let jerry_distance := jerry_speed * jerry_time_hours
  let beth_distance := jerry_distance + beth_extra_miles
  let beth_time_hours := (jerry_time_minutes + beth_extra_time_minutes) / 60
  beth_distance / beth_time_hours

theorem beth_speed {beth_avg_speed : ℚ}
  (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ)
  (h_jerry_speed : jerry_speed = 40)
  (h_jerry_time : jerry_time_minutes = 30)
  (h_beth_extra_miles : beth_extra_miles = 5)
  (h_beth_extra_time : beth_extra_time_minutes = 20) :
  beth_average_speed jerry_speed jerry_time_minutes beth_extra_miles beth_extra_time_minutes = 30 := 
by 
  -- Leaving out the proof steps
  sorry

end beth_speed_l68_6815


namespace turnip_weight_possible_l68_6864

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l68_6864


namespace find_xyz_squares_l68_6819

theorem find_xyz_squares (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end find_xyz_squares_l68_6819


namespace volume_of_larger_part_of_pyramid_proof_l68_6884

noncomputable def volume_of_larger_part_of_pyramid (a b : ℝ) (inclined_angle : ℝ) (area_ratio : ℝ) : ℝ :=
let h_trapezoid := Real.sqrt ((2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 / 4)
let height_pyramid := (1 / 2) * h_trapezoid * Real.tan (inclined_angle)
let volume_total := (1 / 3) * (((a + b) / 2) * Real.sqrt ((a - b) ^ 2 + 4 * h_trapezoid ^ 2) * height_pyramid)
let volume_smaller := (1 / (5 + 7)) * 7 * volume_total
(volume_total - volume_smaller)

theorem volume_of_larger_part_of_pyramid_proof  :
  (volume_of_larger_part_of_pyramid 2 (Real.sqrt 3) (Real.pi / 6) (5 / 7) = 0.875) :=
by
sorry

end volume_of_larger_part_of_pyramid_proof_l68_6884


namespace girls_boys_ratio_l68_6844

-- Let g be the number of girls and b be the number of boys.
-- From the conditions, we have:
-- 1. Total students: g + b = 32
-- 2. More girls than boys: g = b + 6

theorem girls_boys_ratio
  (g b : ℕ) -- Declare number of girls and boys as natural numbers
  (h1 : g + b = 32) -- Total number of students
  (h2 : g = b + 6)  -- 6 more girls than boys
  : g = 19 ∧ b = 13 := 
sorry

end girls_boys_ratio_l68_6844


namespace part1_part2_l68_6898

noncomputable def f (a x : ℝ) := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x
noncomputable def g (a x : ℝ) := x^2 + 5 * a^2
noncomputable def F (a x : ℝ) := f a x + g a x

theorem part1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ a ≤ 0 :=
by sorry

theorem part2 (a : ℝ) : ∀ x : ℝ, F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
by sorry

end part1_part2_l68_6898


namespace distance_between_points_on_parabola_l68_6865

theorem distance_between_points_on_parabola :
  ∀ (x1 x2 y1 y2 : ℝ), 
    (y1^2 = 4 * x1) → (y2^2 = 4 * x2) → (x2 = x1 + 2) → (|y2 - y1| = 4 * Real.sqrt x2 - 4 * Real.sqrt x1) →
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4
  sorry

end distance_between_points_on_parabola_l68_6865


namespace vans_capacity_l68_6802

-- Definitions based on the conditions
def num_students : ℕ := 22
def num_adults : ℕ := 2
def num_vans : ℕ := 3

-- The Lean statement (theorem to be proved)
theorem vans_capacity :
  (num_students + num_adults) / num_vans = 8 := 
by
  sorry

end vans_capacity_l68_6802


namespace thread_length_l68_6813

theorem thread_length (x : ℝ) (h : x + (3/4) * x = 21) : x = 12 :=
  sorry

end thread_length_l68_6813


namespace value_of_a_5_l68_6877

-- Define the sequence with the general term formula
def a (n : ℕ) : ℕ := 4 * n - 3

-- Prove that the value of a_5 is 17
theorem value_of_a_5 : a 5 = 17 := by
  sorry

end value_of_a_5_l68_6877


namespace geometric_sequence_third_term_l68_6847

theorem geometric_sequence_third_term :
  ∀ (a_1 a_5 : ℚ) (r : ℚ), 
    a_1 = 1 / 2 →
    (a_1 * r^4) = a_5 →
    a_5 = 16 →
    (a_1 * r^2) = 2 := 
by
  intros a_1 a_5 r h1 h2 h3
  sorry

end geometric_sequence_third_term_l68_6847


namespace maximize_parabola_area_l68_6866

variable {a b : ℝ}

/--
The parabola y = ax^2 + bx is tangent to the line x + y = 4 within the first quadrant. 
Prove that the values of a and b that maximize the area S enclosed by this parabola and 
the x-axis are a = -1 and b = 3, and that the maximum value of S is 9/2.
-/
theorem maximize_parabola_area (hab_tangent : ∃ x y, y = a * x^2 + b * x ∧ y = 4 - x ∧ x > 0 ∧ y > 0) 
  (area_eqn : S = 1/6 * (b^3 / a^2)) : 
  a = -1 ∧ b = 3 ∧ S = 9/2 := 
sorry

end maximize_parabola_area_l68_6866


namespace expression_equals_24_l68_6863

-- Given values
def a := 7
def b := 4
def c := 1
def d := 7

-- Statement to prove
theorem expression_equals_24 : (a - b) * (c + d) = 24 := by
  sorry

end expression_equals_24_l68_6863


namespace savings_calculation_l68_6872

-- Define the conditions
def income := 17000
def ratio_income_expenditure := 5 / 4

-- Prove that the savings are Rs. 3400
theorem savings_calculation (h : income = 5 * 3400): (income - 4 * 3400) = 3400 :=
by sorry

end savings_calculation_l68_6872


namespace turnips_bag_l68_6855

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l68_6855


namespace no_pos_int_mult_5005_in_form_l68_6886

theorem no_pos_int_mult_5005_in_form (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 49) :
  ¬ ∃ k : ℕ, 5005 * k = 10^j - 10^i := by
  sorry

end no_pos_int_mult_5005_in_form_l68_6886


namespace crown_cost_before_tip_l68_6889

theorem crown_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (crown_cost : ℝ) :
  total_paid = 22000 → tip_percentage = 0.10 → total_paid = crown_cost * (1 + tip_percentage) → crown_cost = 20000 :=
by
  sorry

end crown_cost_before_tip_l68_6889


namespace twelve_point_five_minutes_in_seconds_l68_6828

-- Definitions
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- Theorem: Prove that 12.5 minutes is 750 seconds
theorem twelve_point_five_minutes_in_seconds : minutes_to_seconds 12.5 = 750 :=
by 
  sorry

end twelve_point_five_minutes_in_seconds_l68_6828


namespace find_DG_l68_6831

theorem find_DG (a b k l : ℕ) (h1 : a * k = 37 * (a + b)) (h2 : b * l = 37 * (a + b)) : 
  k = 1406 :=
by
  sorry

end find_DG_l68_6831
