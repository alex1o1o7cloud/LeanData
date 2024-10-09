import Mathlib

namespace total_fishermen_count_l1744_174478

theorem total_fishermen_count (F T F1 F2 : ℕ) (hT : T = 10000) (hF1 : F1 = 19 * 400) (hF2 : F2 = 2400) (hTotal : F1 + F2 = T) : F = 20 :=
by
  sorry

end total_fishermen_count_l1744_174478


namespace flowers_per_bouquet_l1744_174462

theorem flowers_per_bouquet :
  let red_seeds := 125
  let yellow_seeds := 125
  let orange_seeds := 125
  let purple_seeds := 125
  let red_killed := 45
  let yellow_killed := 61
  let orange_killed := 30
  let purple_killed := 40
  let bouquets := 36
  let red_flowers := red_seeds - red_killed
  let yellow_flowers := yellow_seeds - yellow_killed
  let orange_flowers := orange_seeds - orange_killed
  let purple_flowers := purple_seeds - purple_killed
  let total_flowers := red_flowers + yellow_flowers + orange_flowers + purple_flowers
  let flowers_per_bouquet := total_flowers / bouquets
  flowers_per_bouquet = 9 :=
by
  sorry

end flowers_per_bouquet_l1744_174462


namespace anne_bob_total_difference_l1744_174412

-- Define specific values as constants
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.08

-- Define the calculations according to Anne's method
def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define the calculations according to Bob's method
def bob_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- State the theorem that the difference between Anne's and Bob's totals is zero
theorem anne_bob_total_difference : anne_total - bob_total = 0 :=
by sorry  -- Proof not required

end anne_bob_total_difference_l1744_174412


namespace smallest_n_l1744_174443

def n_expr (n : ℕ) : ℕ :=
  n * (2^7) * (3^2) * (7^3)

theorem smallest_n (n : ℕ) (h1: 25 ∣ n_expr n) (h2: 27 ∣ n_expr n) : n = 75 :=
sorry

end smallest_n_l1744_174443


namespace abs_add_lt_abs_sub_l1744_174428

-- Define the conditions
variables {a b : ℝ} (h1 : a * b < 0)

-- Prove the statement
theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| := sorry

end abs_add_lt_abs_sub_l1744_174428


namespace rectangle_sides_l1744_174419

theorem rectangle_sides (k : ℝ) (μ : ℝ) (a b : ℝ) 
  (h₀ : k = 8) 
  (h₁ : μ = 3/10) 
  (h₂ : 2 * (a + b) = k) 
  (h₃ : a * b = μ * (a^2 + b^2)) : 
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) :=
sorry

end rectangle_sides_l1744_174419


namespace roots_cubic_l1744_174465

theorem roots_cubic (a b c d r s t : ℂ) 
    (h1 : a ≠ 0)
    (h2 : r + s + t = -b / a)
    (h3 : r * s + r * t + s * t = c / a)
    (h4 : r * s * t = -d / a) :
    (1 / r^2) + (1 / s^2) + (1 / t^2) = (b^2 - 2 * a * c) / (d^2) :=
by
    sorry

end roots_cubic_l1744_174465


namespace derivative_value_at_pi_over_2_l1744_174474

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem derivative_value_at_pi_over_2 : deriv f (Real.pi / 2) = -1 :=
by
  sorry

end derivative_value_at_pi_over_2_l1744_174474


namespace tan_half_angle_product_zero_l1744_174400

theorem tan_half_angle_product_zero (a b : ℝ) 
  (h: 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) 
  : Real.tan (a / 2) * Real.tan (b / 2) = 0 := 
by 
  sorry

end tan_half_angle_product_zero_l1744_174400


namespace jason_total_expenditure_l1744_174435

theorem jason_total_expenditure :
  let stove_cost := 1200
  let wall_repair_ratio := 1 / 6
  let wall_repair_cost := stove_cost * wall_repair_ratio
  let total_cost := stove_cost + wall_repair_cost
  total_cost = 1400 := by
  {
    sorry
  }

end jason_total_expenditure_l1744_174435


namespace initial_interest_rate_l1744_174407

variable (P r : ℕ)

theorem initial_interest_rate (h1 : 405 = (P * r) / 100) (h2 : 450 = (P * (r + 5)) / 100) : r = 45 :=
sorry

end initial_interest_rate_l1744_174407


namespace consecutive_odd_numbers_sum_power_fourth_l1744_174432

theorem consecutive_odd_numbers_sum_power_fourth :
  ∃ x1 x2 x3 : ℕ, 
  x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ 
  x1 + 2 = x2 ∧ x2 + 2 = x3 ∧ 
  (∃ n : ℕ, n < 10 ∧ (x1 + x2 + x3 = n^4)) :=
sorry

end consecutive_odd_numbers_sum_power_fourth_l1744_174432


namespace sphere_radius_l1744_174457

theorem sphere_radius (r : ℝ) (π : ℝ)
    (h1 : Volume = (4 / 3) * π * r^3)
    (h2 : SurfaceArea = 4 * π * r^2)
    (h3 : Volume = SurfaceArea) :
    r = 3 :=
by
  -- Here starts the proof, but we use 'sorry' to skip it as per the instructions.
  sorry

end sphere_radius_l1744_174457


namespace find_numbers_l1744_174452

theorem find_numbers (x y : ℝ) (r : ℝ) (d : ℝ) 
  (h_geom_x : x = 5 * r) 
  (h_geom_y : y = 5 * r^2)
  (h_arith_1 : y = x + d) 
  (h_arith_2 : 15 = y + d) : 
  x + y = 10 :=
by
  sorry

end find_numbers_l1744_174452


namespace proof_min_value_a3_and_a2b2_l1744_174494

noncomputable def min_value_a3_and_a2b2 (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧
  (a2 = a1 + b1) ∧ (a3 = a1 + 2 * b1) ∧ (b2 = b1 * a1) ∧ 
  (b3 = b1 * a1^2) ∧ (a3 = b3) ∧ 
  (a3 = 3 * Real.sqrt 6 / 2) ∧
  (a2 * b2 = 15 * Real.sqrt 6 / 8) 

theorem proof_min_value_a3_and_a2b2 : ∃ (a1 a2 a3 b1 b2 b3 : ℝ), min_value_a3_and_a2b2 a1 a2 a3 b1 b2 b3 :=
by
  use 2*Real.sqrt 6/3, 5*Real.sqrt 6/4, 3*Real.sqrt 6/2, Real.sqrt 6/4, 3/2, 3*Real.sqrt 6/2
  sorry

end proof_min_value_a3_and_a2b2_l1744_174494


namespace distance_BC_in_circle_l1744_174498

theorem distance_BC_in_circle
    (r : ℝ) (A B C : ℝ × ℝ)
    (h_radius : r = 10)
    (h_diameter : dist A B = 2 * r)
    (h_chord : dist A C = 12) :
    dist B C = 16 := by
  sorry

end distance_BC_in_circle_l1744_174498


namespace coefficient_x2y2_l1744_174404

theorem coefficient_x2y2 : 
  let expr1 := (1 + x) ^ 3
  let expr2 := (1 + y) ^ 4
  let C3_2 := Nat.choose 3 2
  let C4_2 := Nat.choose 4 2
  (C3_2 * C4_2 = 18) := by
    sorry

end coefficient_x2y2_l1744_174404


namespace amount_of_bill_l1744_174485

theorem amount_of_bill (TD R FV T : ℝ) (hTD : TD = 270) (hR : R = 16) (hT : T = 9/12) 
(h_formula : TD = (R * T * FV) / (100 + (R * T))) : FV = 2520 :=
by
  sorry

end amount_of_bill_l1744_174485


namespace greatest_int_less_than_150_with_gcd_30_eq_5_l1744_174421

theorem greatest_int_less_than_150_with_gcd_30_eq_5 : ∃ (n : ℕ), n < 150 ∧ gcd n 30 = 5 ∧ n = 145 := by
  sorry

end greatest_int_less_than_150_with_gcd_30_eq_5_l1744_174421


namespace sara_gave_dan_limes_l1744_174484

theorem sara_gave_dan_limes (initial_limes : ℕ) (final_limes : ℕ) (d : ℕ) 
  (h1: initial_limes = 9) (h2: final_limes = 13) (h3: final_limes = initial_limes + d) : d = 4 := 
by sorry

end sara_gave_dan_limes_l1744_174484


namespace area_before_halving_l1744_174441

theorem area_before_halving (A : ℝ) (h : A / 2 = 7) : A = 14 :=
sorry

end area_before_halving_l1744_174441


namespace find_possible_values_of_a_l1744_174479

noncomputable def find_a (x y a : ℝ) : Prop :=
  (x + y = a) ∧ (x^3 + y^3 = a) ∧ (x^5 + y^5 = a)

theorem find_possible_values_of_a (a : ℝ) :
  (∃ x y : ℝ, find_a x y a) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
sorry

end find_possible_values_of_a_l1744_174479


namespace polynomial_factors_sum_l1744_174486

open Real

theorem polynomial_factors_sum
  (a b c : ℝ)
  (h1 : ∀ x, (x^2 + x + 2) * (a * x + b - a) + (c - a - b) * x + 5 + 2 * a - 2 * b = 0)
  (h2 : a * (1/2)^3 + b * (1/2)^2 + c * (1/2) - 25/16 = 0) :
  a + b + c = 45 / 11 :=
by
  sorry

end polynomial_factors_sum_l1744_174486


namespace friday_lending_tuesday_vs_thursday_total_lending_l1744_174403

def standard_lending_rate : ℕ := 50
def monday_excess : ℤ := 0
def tuesday_excess : ℤ := 8
def wednesday_excess : ℤ := 6
def thursday_shortfall : ℤ := -3
def friday_shortfall : ℤ := -7

theorem friday_lending : (standard_lending_rate + friday_shortfall) = 43 := by
  sorry

theorem tuesday_vs_thursday : (tuesday_excess - thursday_shortfall) = 11 := by
  sorry

theorem total_lending : 
  (5 * standard_lending_rate + (monday_excess + tuesday_excess + wednesday_excess + thursday_shortfall + friday_shortfall)) = 254 := by
  sorry

end friday_lending_tuesday_vs_thursday_total_lending_l1744_174403


namespace intersection_at_one_point_l1744_174438

theorem intersection_at_one_point (m : ℝ) :
  (∃ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 ∧
            ∀ x' : ℝ, (m - 4) * x'^2 - 2 * m * x' - m - 6 = 0 → x' = x) ↔
  m = -4 ∨ m = 3 ∨ m = 4 := 
by
  sorry

end intersection_at_one_point_l1744_174438


namespace parallel_lines_slope_eq_l1744_174439

theorem parallel_lines_slope_eq (k : ℝ) : (∀ x : ℝ, 3 = 6 * k) → k = 1 / 2 :=
by
  intro h
  sorry

end parallel_lines_slope_eq_l1744_174439


namespace largest_integral_value_of_y_l1744_174464

theorem largest_integral_value_of_y : 
  (1 / 4 : ℝ) < (y / 7 : ℝ) ∧ (y / 7 : ℝ) < (3 / 5 : ℝ) → y ≤ 4 :=
by
  sorry

end largest_integral_value_of_y_l1744_174464


namespace factor_polynomials_l1744_174408

theorem factor_polynomials :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) :=
by
  sorry

end factor_polynomials_l1744_174408


namespace factors_180_count_l1744_174477

theorem factors_180_count : 
  ∃ (n : ℕ), 180 = 2^2 * 3^2 * 5^1 ∧ n = 18 ∧ 
  ∀ p a b c, 
  180 = p^a * p^b * p^c →
  (a+1) * (b+1) * (c+1) = 18 :=
by {
  sorry
}

end factors_180_count_l1744_174477


namespace number_of_integer_solutions_l1744_174454

theorem number_of_integer_solutions : 
  (∃ (sols : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ sols ↔ (1 : ℚ)/x + (1 : ℚ)/y = 1/7) ∧ sols.length = 5) := 
sorry

end number_of_integer_solutions_l1744_174454


namespace real_part_implies_value_of_a_l1744_174499

theorem real_part_implies_value_of_a (a b : ℝ) (h : a = 2 * b) (hb : b = 1) : a = 2 := by
  sorry

end real_part_implies_value_of_a_l1744_174499


namespace range_of_a_l1744_174480

theorem range_of_a (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5 * x + 15 / 2 * a <= 0) -> a > 5 / 6 :=
by
  sorry

end range_of_a_l1744_174480


namespace triangle_area_is_3_max_f_l1744_174437

noncomputable def triangle_area :=
  let a : ℝ := 2
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 2
  let A : ℝ := Real.pi / 3
  (1 / 2) * b * c * Real.sin A

theorem triangle_area_is_3 :
  triangle_area = 3 := by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * (Real.sin x * Real.cos (Real.pi / 3) + Real.cos x * Real.sin (Real.pi / 3))

theorem max_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 3), f x = 2 + Real.sqrt 3 ∧ x = Real.pi / 12 := by
  sorry

end triangle_area_is_3_max_f_l1744_174437


namespace spherical_cap_surface_area_l1744_174468

theorem spherical_cap_surface_area (V : ℝ) (h : ℝ) (A : ℝ) (r : ℝ) 
  (volume_eq : V = (4 / 3) * π * r^3) 
  (cap_height : h = 2) 
  (sphere_volume : V = 288 * π) 
  (cap_surface_area : A = 2 * π * r * h) : 
  A = 24 * π := 
sorry

end spherical_cap_surface_area_l1744_174468


namespace average_speed_of_train_l1744_174483

theorem average_speed_of_train (x : ℝ) (h1 : 0 < x) : 
  let Time1 := x / 40
  let Time2 := x / 10
  let TotalDistance := 3 * x
  let TotalTime := x / 8
  (TotalDistance / TotalTime = 24) :=
by
  sorry

end average_speed_of_train_l1744_174483


namespace max_regular_hours_l1744_174433

/-- A man's regular pay is $3 per hour up to a certain number of hours, and his overtime pay rate
    is twice the regular pay rate. The man was paid $180 and worked 10 hours overtime.
    Prove that the maximum number of hours he can work at his regular pay rate is 40 hours.
-/
theorem max_regular_hours (P R OT : ℕ) (hP : P = 180) (hOT : OT = 10) (reg_rate overtime_rate : ℕ)
  (hreg_rate : reg_rate = 3) (hovertime_rate : overtime_rate = 2 * reg_rate) :
  P = reg_rate * R + overtime_rate * OT → R = 40 :=
by
  sorry

end max_regular_hours_l1744_174433


namespace bottle_caps_weight_l1744_174461

theorem bottle_caps_weight :
  (∀ n : ℕ, n = 7 → 1 = 1) → -- 7 bottle caps weigh exactly 1 ounce
  (∀ m : ℕ, m = 2016 → 1 = 1) → -- Josh has 2016 bottle caps
  2016 / 7 = 288 := -- The weight of Josh's entire bottle cap collection is 288 ounces
by
  intros h1 h2
  sorry

end bottle_caps_weight_l1744_174461


namespace neg_all_cups_full_l1744_174495

variable (x : Type) (cup : x → Prop) (full : x → Prop)

theorem neg_all_cups_full :
  ¬ (∀ x, cup x → full x) = ∃ x, cup x ∧ ¬ full x := by
sorry

end neg_all_cups_full_l1744_174495


namespace sales_tax_per_tire_l1744_174496

def cost_per_tire : ℝ := 7
def number_of_tires : ℕ := 4
def final_total_cost : ℝ := 30

theorem sales_tax_per_tire :
  (final_total_cost - number_of_tires * cost_per_tire) / number_of_tires = 0.5 :=
sorry

end sales_tax_per_tire_l1744_174496


namespace average_age_combined_l1744_174450

-- Definitions of the given conditions
def avg_age_fifth_graders := 10
def number_fifth_graders := 40
def avg_age_parents := 40
def number_parents := 60

-- The theorem we need to prove
theorem average_age_combined : 
  (avg_age_fifth_graders * number_fifth_graders + avg_age_parents * number_parents) / (number_fifth_graders + number_parents) = 28 := 
by
  sorry

end average_age_combined_l1744_174450


namespace students_per_bus_l1744_174453

theorem students_per_bus
  (total_students : ℕ)
  (buses : ℕ)
  (students_in_cars : ℕ)
  (h1 : total_students = 375)
  (h2 : buses = 7)
  (h3 : students_in_cars = 4) :
  (total_students - students_in_cars) / buses = 53 :=
by
  sorry

end students_per_bus_l1744_174453


namespace inequality_solution_l1744_174473

noncomputable def solution_set : Set ℝ := {x : ℝ | x < 4 ∨ x > 5}

theorem inequality_solution (x : ℝ) :
  (x - 2) / (x - 4) ≤ 3 ↔ x ∈ solution_set :=
by
  sorry

end inequality_solution_l1744_174473


namespace min_abs_sum_of_x1_x2_l1744_174445

open Real

theorem min_abs_sum_of_x1_x2 (x1 x2 : ℝ) (h : 1 / ((2 + sin x1) * (2 + sin (2 * x2))) = 1) : 
  abs (x1 + x2) = π / 4 :=
sorry

end min_abs_sum_of_x1_x2_l1744_174445


namespace JakeMowingEarnings_l1744_174414

theorem JakeMowingEarnings :
  (∀ rate hours_mowing hours_planting (total_charge : ℝ),
      rate = 20 →
      hours_mowing = 1 →
      hours_planting = 2 →
      total_charge = 45 →
      (total_charge = hours_planting * rate + 5) →
      hours_mowing * rate = 20) :=
by
  intros rate hours_mowing hours_planting total_charge
  sorry

end JakeMowingEarnings_l1744_174414


namespace no_solutions_for_a_gt_1_l1744_174410

theorem no_solutions_for_a_gt_1 (a b : ℝ) (h_a_gt_1 : 1 < a) :
  ¬∃ x : ℝ, a^(2-2*x^2) + (b+4) * a^(1-x^2) + 3*b + 4 = 0 ↔ 0 < b ∧ b < 4 :=
by
  sorry

end no_solutions_for_a_gt_1_l1744_174410


namespace david_course_hours_l1744_174481

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l1744_174481


namespace volleyball_match_prob_A_win_l1744_174491

-- Definitions of given probabilities and conditions
def rally_scoring_system := true
def first_to_25_wins := true
def tie_at_24_24_continues_until_lead_by_2 := true
def prob_team_A_serves_win : ℚ := 2/3
def prob_team_B_serves_win : ℚ := 2/5
def outcomes_independent := true
def score_22_22_team_A_serves := true

-- The problem to prove
theorem volleyball_match_prob_A_win :
  rally_scoring_system ∧
  first_to_25_wins ∧
  tie_at_24_24_continues_until_lead_by_2 ∧
  prob_team_A_serves_win = 2/3 ∧
  prob_team_B_serves_win = 2/5 ∧
  outcomes_independent ∧
  score_22_22_team_A_serves →
  (prob_team_A_serves_win ^ 3 + (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win ^ 2 + prob_team_A_serves_win * (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win + prob_team_A_serves_win ^ 2 * (1 - prob_team_A_serves_win) * prob_team_B_serves_win) = 64/135 :=
by
  sorry

end volleyball_match_prob_A_win_l1744_174491


namespace smallest_four_digit_divisible_by_33_l1744_174448

theorem smallest_four_digit_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 ∧ n = 1023 := by 
  sorry

end smallest_four_digit_divisible_by_33_l1744_174448


namespace find_number_of_children_l1744_174487

def admission_cost_adult : ℝ := 30
def admission_cost_child : ℝ := 15
def total_people : ℕ := 10
def soda_cost : ℝ := 5
def discount_rate : ℝ := 0.8
def total_paid : ℝ := 197

def total_cost_with_discount (adults children : ℕ) : ℝ :=
  discount_rate * (adults * admission_cost_adult + children * admission_cost_child)

theorem find_number_of_children (A C : ℕ) 
  (h1 : A + C = total_people)
  (h2 : total_cost_with_discount A C + soda_cost = total_paid) :
  C = 4 :=
sorry

end find_number_of_children_l1744_174487


namespace children_too_heavy_l1744_174406

def Kelly_weight : ℝ := 34
def Sam_weight : ℝ := 40
def Daisy_weight : ℝ := 28
def Megan_weight := 1.1 * Kelly_weight
def Mike_weight := Megan_weight + 5

def Total_weight := Kelly_weight + Sam_weight + Daisy_weight + Megan_weight + Mike_weight
def Bridge_limit : ℝ := 130

theorem children_too_heavy :
  Total_weight - Bridge_limit = 51.8 :=
by
  sorry

end children_too_heavy_l1744_174406


namespace smallest_integer_l1744_174440

theorem smallest_integer :
  ∃ (M : ℕ), M > 0 ∧
             M % 3 = 2 ∧
             M % 4 = 3 ∧
             M % 5 = 4 ∧
             M % 6 = 5 ∧
             M % 7 = 6 ∧
             M % 11 = 10 ∧
             M = 4619 :=
by
  sorry

end smallest_integer_l1744_174440


namespace driers_drying_time_l1744_174470

noncomputable def drying_time (r1 r2 r3 : ℝ) : ℝ := 1 / (r1 + r2 + r3)

theorem driers_drying_time (Q : ℝ) (r1 r2 r3 : ℝ)
  (h1 : r1 = Q / 24) 
  (h2 : r2 = Q / 2) 
  (h3 : r3 = Q / 8) : 
  drying_time r1 r2 r3 = 1.5 :=
by
  sorry

end driers_drying_time_l1744_174470


namespace problem_BD_l1744_174469

variable (a b c : ℝ)

theorem problem_BD (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) :=
by
  sorry

end problem_BD_l1744_174469


namespace circle_through_A_B_C_l1744_174471

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (1, 12)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (-9, 2)

-- Definition of the expected standard equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 100

-- Theorem stating that the expected equation is the equation of the circle through points A, B, and C
theorem circle_through_A_B_C : 
  ∀ (x y : ℝ),
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → 
  circle_eq x y := sorry

end circle_through_A_B_C_l1744_174471


namespace square_of_larger_number_is_1156_l1744_174426

theorem square_of_larger_number_is_1156
  (x y : ℕ)
  (h1 : x + y = 60)
  (h2 : x - y = 8) :
  x^2 = 1156 := by
  sorry

end square_of_larger_number_is_1156_l1744_174426


namespace shopping_people_count_l1744_174423

theorem shopping_people_count :
  ∃ P : ℕ, P = 10 ∧
  ∃ (stores : ℕ) (total_visits : ℕ) (two_store_visitors : ℕ) 
    (at_least_one_store_visitors : ℕ) (max_stores_visited : ℕ),
    stores = 8 ∧
    total_visits = 22 ∧
    two_store_visitors = 8 ∧
    at_least_one_store_visitors = P ∧
    max_stores_visited = 3 ∧
    total_visits = (two_store_visitors * 2) + 6 ∧
    P = two_store_visitors + 2 :=
by {
    sorry
}

end shopping_people_count_l1744_174423


namespace shaded_fraction_is_one_eighth_l1744_174411

noncomputable def total_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def half_area (length : ℕ) (width : ℕ) : ℚ :=
  total_area length width / 2

noncomputable def shaded_area (length : ℕ) (width : ℕ) : ℚ :=
  half_area length width / 4

theorem shaded_fraction_is_one_eighth : 
  ∀ (length width : ℕ), length = 15 → width = 21 → shaded_area length width / total_area length width = 1 / 8 :=
by
  sorry

end shaded_fraction_is_one_eighth_l1744_174411


namespace balance_of_three_squares_and_two_heartsuits_l1744_174467

-- Definitions
variable {x y z w : ℝ}

-- Given conditions
axiom h1 : 3 * x + 4 * y + z = 12 * w
axiom h2 : x = z + 2 * w

-- Problem to prove
theorem balance_of_three_squares_and_two_heartsuits :
  (3 * y + 2 * z) = (26 / 9) * w :=
sorry

end balance_of_three_squares_and_two_heartsuits_l1744_174467


namespace differentiate_and_evaluate_l1744_174449

theorem differentiate_and_evaluate (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ) :
  (2*x - 1)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 = 12 :=
sorry

end differentiate_and_evaluate_l1744_174449


namespace area_bounded_by_curve_and_line_l1744_174492

theorem area_bounded_by_curve_and_line :
  let curve_x (t : ℝ) := 10 * (t - Real.sin t)
  let curve_y (t : ℝ) := 10 * (1 - Real.cos t)
  let y_line := 15
  (∫ t in (2/3) * Real.pi..(4/3) * Real.pi, 100 * (1 - Real.cos t)^2) = 100 * Real.pi + 200 * Real.sqrt 3 :=
by
  sorry

end area_bounded_by_curve_and_line_l1744_174492


namespace symmetric_point_origin_l1744_174422

-- Define the original point P with given coordinates
def P : ℝ × ℝ := (-2, 3)

-- Define the symmetric point P' with respect to the origin
def P'_symmetric (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The theorem states that the symmetric point of P is (2, -3)
theorem symmetric_point_origin : P'_symmetric P = (2, -3) := 
by
  sorry

end symmetric_point_origin_l1744_174422


namespace no_four_distinct_nat_dividing_pairs_l1744_174413

theorem no_four_distinct_nat_dividing_pairs (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∣ (b - c)) (h8 : a ∣ (b - d))
  (h9 : a ∣ (c - d)) (h10 : b ∣ (a - c)) (h11 : b ∣ (a - d)) (h12 : b ∣ (c - d))
  (h13 : c ∣ (a - b)) (h14 : c ∣ (a - d)) (h15 : c ∣ (b - d)) (h16 : d ∣ (a - b))
  (h17 : d ∣ (a - c)) (h18 : d ∣ (b - c)) : False := 
sorry

end no_four_distinct_nat_dividing_pairs_l1744_174413


namespace saturday_price_is_correct_l1744_174434

-- Define Thursday's price
def thursday_price : ℝ := 50

-- Define the price increase rate on Friday
def friday_increase_rate : ℝ := 0.2

-- Define the discount rate on Saturday
def saturday_discount_rate : ℝ := 0.15

-- Calculate the price on Friday
def friday_price : ℝ := thursday_price * (1 + friday_increase_rate)

-- Calculate the discount amount on Saturday
def saturday_discount : ℝ := friday_price * saturday_discount_rate

-- Calculate the price on Saturday
def saturday_price : ℝ := friday_price - saturday_discount

-- Theorem stating the price on Saturday
theorem saturday_price_is_correct : saturday_price = 51 := by
  -- Definitions are already embedded into the conditions
  -- so here we only state the property to be proved.
  sorry

end saturday_price_is_correct_l1744_174434


namespace division_remainder_l1744_174446

theorem division_remainder (dividend divisor quotient : ℕ) (h_dividend : dividend = 131) (h_divisor : divisor = 14) (h_quotient : quotient = 9) :
  ∃ remainder : ℕ, dividend = divisor * quotient + remainder ∧ remainder = 5 :=
by
  sorry

end division_remainder_l1744_174446


namespace last_digit_p_adic_l1744_174401

theorem last_digit_p_adic (a : ℤ) (p : ℕ) (hp : Nat.Prime p) (h_last_digit_nonzero : a % p ≠ 0) : (a ^ (p - 1) - 1) % p = 0 :=
by
  sorry

end last_digit_p_adic_l1744_174401


namespace min_value_of_f_min_value_at_x_1_l1744_174482

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + 1 / (2 - 3 * x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 35 :=
by
  sorry

-- As an additional statement, we can check the specific case at x = 1
theorem min_value_at_x_1 :
  f 1 = 35 :=
by
  sorry

end min_value_of_f_min_value_at_x_1_l1744_174482


namespace gcd_solution_l1744_174416

noncomputable def gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : ℤ :=
  Int.gcd (4 * b^2 + 35 * b + 72) (3 * b + 8)

theorem gcd_solution (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : gcd_problem b h = 2 :=
by
  sorry

end gcd_solution_l1744_174416


namespace square_side_increase_factor_l1744_174415

theorem square_side_increase_factor (s k : ℕ) (x new_x : ℕ) (h1 : x = 4 * s) (h2 : new_x = 4 * x) (h3 : new_x = 4 * (k * s)) : k = 4 :=
by
  sorry

end square_side_increase_factor_l1744_174415


namespace sale_saving_percentage_l1744_174493

theorem sale_saving_percentage (P : ℝ) : 
  let original_price := 8 * P
  let sale_price := 6 * P
  let amount_saved := original_price - sale_price
  let percentage_saved := (amount_saved / original_price) * 100
  percentage_saved = 25 :=
by
  sorry

end sale_saving_percentage_l1744_174493


namespace second_quadrant_point_l1744_174418

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point_l1744_174418


namespace gcd_14m_21n_126_l1744_174460

theorem gcd_14m_21n_126 {m n : ℕ} (hm_pos : 0 < m) (hn_pos : 0 < n) (h_gcd : Nat.gcd m n = 18) : 
  Nat.gcd (14 * m) (21 * n) = 126 :=
by
  sorry

end gcd_14m_21n_126_l1744_174460


namespace fourth_function_form_l1744_174417

variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)
variable (hf : Function.LeftInverse f_inv f)
variable (hf_inv : Function.RightInverse f_inv f)

theorem fourth_function_form :
  (∀ x, y = (-(f (-x - 1)) + 2) ↔ y = f_inv (x + 2) + 1 ↔ -(x + y) = 0) :=
  sorry

end fourth_function_form_l1744_174417


namespace smallest_of_x_y_z_l1744_174456

variables {a b c d : ℕ}

/-- Given that x, y, and z are in the ratio a, b, c respectively, 
    and their sum x + y + z equals d, and 0 < a < b < c,
    prove that the smallest of x, y, and z is da / (a + b + c). -/
theorem smallest_of_x_y_z (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : 0 < d)
    (h_sum : ∀ k : ℚ, x = k * a → y = k * b → z = k * c → x + y + z = d) : 
    (∃ k : ℚ, x = k * a ∧ y = k * b ∧ z = k * c ∧ k = d / (a + b + c) ∧ x = da / (a + b + c)) :=
by 
  sorry

end smallest_of_x_y_z_l1744_174456


namespace deductible_increase_l1744_174427

theorem deductible_increase (current_deductible : ℝ) (increase_fraction : ℝ) (next_year_deductible : ℝ) : 
  current_deductible = 3000 ∧ increase_fraction = 2 / 3 ∧ next_year_deductible = (1 + increase_fraction) * current_deductible →
  next_year_deductible - current_deductible = 2000 :=
by
  intros h
  sorry

end deductible_increase_l1744_174427


namespace game_result_l1744_174436

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_sequence : List ℕ := [2, 5, 6, 4, 3]
def jill_sequence : List ℕ := [1, 6, 3, 2, 5]

def calculate_score (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + g x) 0

theorem game_result : calculate_score jack_sequence * calculate_score jill_sequence = 420 :=
by
  sorry

end game_result_l1744_174436


namespace b_95_mod_49_l1744_174490

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := 
by sorry

end b_95_mod_49_l1744_174490


namespace simplify_f_l1744_174409

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) (h : Real.sin (α - 3 / 2 * Real.pi) = 1 / 5) : f α = -1 / 5 := by
  sorry

end simplify_f_l1744_174409


namespace range_of_m_l1744_174442

variable (m : ℝ)

def hyperbola (m : ℝ) := (x y : ℝ) → (x^2 / (1 + m)) - (y^2 / (3 - m)) = 1

def eccentricity_condition (m : ℝ) := (2 / (Real.sqrt (1 + m)) > Real.sqrt 2)

theorem range_of_m (m : ℝ) (h1 : 1 + m > 0) (h2 : 3 - m > 0) (h3 : eccentricity_condition m) :
 -1 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l1744_174442


namespace hexagon_ratio_l1744_174455

theorem hexagon_ratio (A B : ℝ) (h₁ : A = 8) (h₂ : B = 2)
                      (A_above : ℝ) (h₃ : A_above = (3 + B))
                      (H : 3 + B = 1 / 2 * (A + B)) 
                      (XQ QY : ℝ) (h₄ : XQ + QY = 4)
                      (h₅ : 3 + B = 4 + B / 2) :
  XQ / QY = 2 := 
by
  sorry

end hexagon_ratio_l1744_174455


namespace balloon_permutations_count_l1744_174402

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l1744_174402


namespace problem_statement_l1744_174476

noncomputable def seq_sub_triples: ℚ :=
  let a := (5 / 6 : ℚ)
  let b := (1 / 6 : ℚ)
  let c := (1 / 4 : ℚ)
  a - b - c

theorem problem_statement : seq_sub_triples = 5 / 12 := by
  sorry

end problem_statement_l1744_174476


namespace game_A_probability_greater_than_B_l1744_174424

-- Defining the probabilities of heads and tails for the biased coin
def prob_heads : ℚ := 2 / 3
def prob_tails : ℚ := 1 / 3

-- Defining the winning probabilities for Game A
def prob_winning_A : ℚ := (prob_heads^4) + (prob_tails^4)

-- Defining the winning probabilities for Game B
def prob_winning_B : ℚ := (prob_heads^3 * prob_tails) + (prob_tails^3 * prob_heads)

-- The statement we want to prove
theorem game_A_probability_greater_than_B : prob_winning_A - prob_winning_B = 7 / 81 := by
  sorry

end game_A_probability_greater_than_B_l1744_174424


namespace length_of_each_movie_l1744_174466

-- Defining the amount of time Grandpa Lou watched movies on Tuesday in minutes
def time_tuesday : ℕ := 4 * 60 + 30   -- 4 hours and 30 minutes

-- Defining the number of movies watched on Tuesday
def movies_tuesday (x : ℕ) : Prop := time_tuesday / x = 90

-- Defining the total number of movies watched in both days
def total_movies_two_days (x : ℕ) : Prop := x + 2 * x = 9

theorem length_of_each_movie (x : ℕ) (h₁ : total_movies_two_days x) (h₂ : movies_tuesday x) : time_tuesday / x = 90 :=
by
  -- Given the conditions, we can prove the statement:
  sorry

end length_of_each_movie_l1744_174466


namespace overall_gain_percent_l1744_174497

variables (C_A S_A C_B S_B : ℝ)

def cost_price_A (n : ℝ) : ℝ := n * C_A
def selling_price_A (n : ℝ) : ℝ := n * S_A

def cost_price_B (n : ℝ) : ℝ := n * C_B
def selling_price_B (n : ℝ) : ℝ := n * S_B

theorem overall_gain_percent :
  (selling_price_A 25 = cost_price_A 50) →
  (selling_price_B 30 = cost_price_B 60) →
  ((S_A - C_A) / C_A * 100 = 100) ∧ ((S_B - C_B) / C_B * 100 = 100) :=
by
  sorry

end overall_gain_percent_l1744_174497


namespace asymptote_equation_l1744_174447

theorem asymptote_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + Real.sqrt (a^2 + b^2) = 2 * b) →
  (4 * x = 3 * y) ∨ (4 * x = -3 * y) :=
by
  sorry

end asymptote_equation_l1744_174447


namespace BANANA_arrangements_l1744_174489

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l1744_174489


namespace divisor_count_l1744_174429

theorem divisor_count (m : ℕ) (h : m = 2^15 * 5^12) :
  let m_squared := m * m
  let num_divisors_m := (15 + 1) * (12 + 1)
  let num_divisors_m_squared := (30 + 1) * (24 + 1)
  let divisors_of_m_squared_less_than_m := (num_divisors_m_squared - 1) / 2
  num_divisors_m_squared - num_divisors_m = 179 :=
by
  subst h
  sorry

end divisor_count_l1744_174429


namespace slopes_product_no_circle_MN_A_l1744_174459

-- Define the equation of the ellipse E and the specific points A and B
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the point P which lies on the ellipse
def P (x0 y0 : ℝ) : Prop := ellipse_eq x0 y0 ∧ x0 ≠ -2 ∧ x0 ≠ 2

-- Prove the product of the slopes of lines PA and PB
theorem slopes_product (x0 y0 : ℝ) (hP : P x0 y0) : 
  (y0 / (x0 + 2)) * (y0 / (x0 - 2)) = -1 / 4 := sorry

-- Define point Q
def Q : ℝ × ℝ := (-1, 0)

-- Define points M and N which are intersections of line and ellipse
def MN_line (t y : ℝ) : ℝ := t * y - 1

-- Prove there is no circle with diameter MN passing through A
theorem no_circle_MN_A (t : ℝ) : 
  ¬ ∃ M N : ℝ × ℝ, ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧
  (∃ x1 y1 x2 y2, (M = (x1, y1) ∧ N = (x2, y2)) ∧
  (MN_line t y1 = x1 ∧ MN_line t y2 = x2) ∧ 
  ((x1 + 2) * (x2 + 2) + y1 * y2 = 0)) := sorry

end slopes_product_no_circle_MN_A_l1744_174459


namespace benny_bought_books_l1744_174463

theorem benny_bought_books :
  ∀ (initial_books sold_books remaining_books bought_books : ℕ),
    initial_books = 22 →
    sold_books = initial_books / 2 →
    remaining_books = initial_books - sold_books →
    remaining_books + bought_books = 17 →
    bought_books = 6 :=
by
  intros initial_books sold_books remaining_books bought_books
  sorry

end benny_bought_books_l1744_174463


namespace gcd_exponentiation_gcd_fermat_numbers_l1744_174488

-- Part (a)
theorem gcd_exponentiation (m n : ℕ) (a : ℕ) (h1 : m ≠ n) (h2 : a > 1) : 
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by
sorry

-- Part (b)
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (m n : ℕ) (h1 : m ≠ n) : 
  Nat.gcd (fermat_number m) (fermat_number n) = 1 :=
by
sorry

end gcd_exponentiation_gcd_fermat_numbers_l1744_174488


namespace solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l1744_174444

-- Case 1: a ≠ 0
theorem solve_eq_nonzero (a b : ℝ) (h : a ≠ 0) : ∃ x : ℝ, x = -b / a ∧ a * x + b = 0 :=
by
  sorry

-- Case 2: a = 0 and b = 0
theorem solve_eq_zero_zero (a b : ℝ) (h1 : a = 0) (h2 : b = 0) : ∀ x : ℝ, a * x + b = 0 :=
by
  sorry

-- Case 3: a = 0 and b ≠ 0
theorem solve_eq_zero_nonzero (a b : ℝ) (h1 : a = 0) (h2 : b ≠ 0) : ¬ ∃ x : ℝ, a * x + b = 0 :=
by
  sorry

end solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l1744_174444


namespace perimeter_ACFHK_is_correct_l1744_174420

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the points of the pentagon within the dodecagon
def ACFHK_points : ℕ := 5

-- Define the perimeter of the pentagon ACFHK in the dodecagon
noncomputable def perimeter_of_ACFHK : ℝ :=
  let triangle_side := radius
  let isosceles_right_triangle_side := radius * Real.sqrt 2
  3 * triangle_side + 2 * isosceles_right_triangle_side

-- Verify that the calculated perimeter matches the expected value
theorem perimeter_ACFHK_is_correct : perimeter_of_ACFHK = 18 + 12 * Real.sqrt 2 :=
  sorry

end perimeter_ACFHK_is_correct_l1744_174420


namespace correct_calculation_l1744_174430

theorem correct_calculation :
  ∃ (a : ℤ), (a^2 + a^2 = 2 * a^2) ∧ 
  (¬(3*a + 4*(a : ℤ) = 12*a*(a : ℤ))) ∧ 
  (¬((a*(a : ℤ)^2)^3 = a*(a : ℤ)^6)) ∧ 
  (¬((a + 3)^2 = a^2 + 9)) :=
by
  sorry

end correct_calculation_l1744_174430


namespace arithmetic_seq_perfect_sixth_power_l1744_174472

theorem arithmetic_seq_perfect_sixth_power 
  (a h : ℤ)
  (seq : ∀ n : ℕ, ℤ)
  (h_seq : ∀ n, seq n = a + n * h)
  (h1 : ∃ s₁ x, seq s₁ = x^2)
  (h2 : ∃ s₂ y, seq s₂ = y^3) :
  ∃ k s, seq s = k^6 := 
sorry

end arithmetic_seq_perfect_sixth_power_l1744_174472


namespace solve_system_l1744_174451

theorem solve_system :
  ∃ (x y : ℤ), (x * (1/7 : ℚ)^2 = 7^3) ∧ (x + y = 7^2) ∧ (x = 16807) ∧ (y = -16758) :=
by
  sorry

end solve_system_l1744_174451


namespace percentage_cities_in_range_l1744_174425

-- Definitions of percentages as given conditions
def percentage_cities_between_50k_200k : ℕ := 40
def percentage_cities_below_50k : ℕ := 35
def percentage_cities_above_200k : ℕ := 25

-- Statement of the problem
theorem percentage_cities_in_range :
  percentage_cities_between_50k_200k = 40 := 
by
  sorry

end percentage_cities_in_range_l1744_174425


namespace total_people_is_120_l1744_174475

def num_children : ℕ := 80

def num_adults (num_children : ℕ) : ℕ := num_children / 2

def total_people (num_children num_adults : ℕ) : ℕ := num_children + num_adults

theorem total_people_is_120 : total_people num_children (num_adults num_children) = 120 := by
  sorry

end total_people_is_120_l1744_174475


namespace distinct_roots_difference_l1744_174431

theorem distinct_roots_difference (r s : ℝ) (h₀ : r ≠ s) (h₁ : r > s) (h₂ : ∀ x, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) :
  r - s = Real.sqrt 29 :=
by
  sorry

end distinct_roots_difference_l1744_174431


namespace min_final_exam_score_l1744_174405

theorem min_final_exam_score (q1 q2 q3 q4 final_exam : ℤ)
    (H1 : q1 = 90) (H2 : q2 = 85) (H3 : q3 = 77) (H4 : q4 = 96) :
    (1/2) * (q1 + q2 + q3 + q4) / 4 + (1/2) * final_exam ≥ 90 ↔ final_exam ≥ 93 :=
by
    sorry

end min_final_exam_score_l1744_174405


namespace abs_neg_three_eq_three_l1744_174458

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l1744_174458
