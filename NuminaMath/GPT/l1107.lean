import Mathlib

namespace min_restoration_time_l1107_110774

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end min_restoration_time_l1107_110774


namespace sally_initial_poems_l1107_110750

theorem sally_initial_poems (recited: ℕ) (forgotten: ℕ) (h1 : recited = 3) (h2 : forgotten = 5) : 
  recited + forgotten = 8 := 
by
  sorry

end sally_initial_poems_l1107_110750


namespace problem_statement_l1107_110788

def p (x y : ℝ) : Prop :=
  (x^2 + y^2 ≠ 0) → ¬ (x = 0 ∧ y = 0)

def q (m : ℝ) : Prop :=
  (m > -2) → ∃ x : ℝ, x^2 + 2*x - m = 0

theorem problem_statement : ∀ (x y m : ℝ), p x y ∨ q m :=
sorry

end problem_statement_l1107_110788


namespace range_of_a_l1107_110772

noncomputable def A (a : ℝ) : Set ℝ := { x | 3 + a ≤ x ∧ x ≤ 4 + 3 * a }
noncomputable def B : Set ℝ := { x | -4 ≤ x ∧ x < 5 }

theorem range_of_a (a : ℝ) : A a ⊆ B ↔ -1/2 ≤ a ∧ a < 1/3 :=
  sorry

end range_of_a_l1107_110772


namespace number_of_squares_is_five_l1107_110745

-- A function that computes the number of squares obtained after the described operations on a piece of paper.
def folded_and_cut_number_of_squares (initial_shape : Type) (folds : ℕ) (cuts : ℕ) : ℕ :=
  -- sorry is used here as a placeholder for the actual implementation
  sorry

-- The main theorem stating that after two folds and two cuts, we obtain five square pieces.
theorem number_of_squares_is_five (initial_shape : Type) (h_initial_square : initial_shape = square)
  (h_folds : folds = 2) (h_cuts : cuts = 2) : folded_and_cut_number_of_squares initial_shape folds cuts = 5 :=
  sorry

end number_of_squares_is_five_l1107_110745


namespace identify_false_condition_l1107_110743

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
def condition_A (a b c : ℝ) : Prop := quadratic_function a b c (-1) = 0
def condition_B (a b c : ℝ) : Prop := 2 * a + b = 0
def condition_C (a b c : ℝ) : Prop := quadratic_function a b c 1 = 3
def condition_D (a b c : ℝ) : Prop := quadratic_function a b c 2 = 8

-- Main theorem stating which condition is false
theorem identify_false_condition (a b c : ℝ) (ha : a ≠ 0) : ¬ condition_A a b c ∨ ¬ condition_B a b c ∨ ¬ condition_C a b c ∨  ¬ condition_D a b c :=
by
sorry

end identify_false_condition_l1107_110743


namespace bridget_heavier_than_martha_l1107_110739

def bridget_weight := 39
def martha_weight := 2

theorem bridget_heavier_than_martha :
  bridget_weight - martha_weight = 37 :=
by
  sorry

end bridget_heavier_than_martha_l1107_110739


namespace time_against_current_l1107_110756

-- Define the conditions:
def swimming_speed_still_water : ℝ := 6  -- Speed in still water (km/h)
def current_speed : ℝ := 2  -- Speed of the water current (km/h)
def time_with_current : ℝ := 3.5  -- Time taken to swim with the current (hours)

-- Define effective speeds:
def effective_speed_against_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water - current_speed

def effective_speed_with_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water + current_speed

-- Calculate the distance covered with the current:
def distance_with_current (time_with_current effective_speed_with_current: ℝ) : ℝ :=
  time_with_current * effective_speed_with_current

-- Define the proof goal:
theorem time_against_current (h1 : swimming_speed_still_water = 6) (h2 : current_speed = 2)
  (h3 : time_with_current = 3.5) :
  ∃ (t : ℝ), t = 7 := by
  sorry

end time_against_current_l1107_110756


namespace ace_first_king_second_prob_l1107_110776

def cards : Type := { x : ℕ // x < 52 }

def ace (c : cards) : Prop := 
  c.1 = 0 ∨ c.1 = 1 ∨ c.1 = 2 ∨ c.1 = 3

def king (c : cards) : Prop := 
  c.1 = 4 ∨ c.1 = 5 ∨ c.1 = 6 ∨ c.1 = 7

def prob_ace_first_king_second : ℚ := 4 / 52 * 4 / 51

theorem ace_first_king_second_prob :
  prob_ace_first_king_second = 4 / 663 := by
  sorry

end ace_first_king_second_prob_l1107_110776


namespace minimum_value_inequality_l1107_110714

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end minimum_value_inequality_l1107_110714


namespace stock_value_order_l1107_110775

-- Define the initial investment and yearly changes
def initialInvestment : Float := 100
def firstYearChangeA : Float := 1.30
def firstYearChangeB : Float := 0.70
def firstYearChangeG : Float := 1.10
def firstYearChangeD : Float := 1.00 -- unchanged

def secondYearChangeA : Float := 0.90
def secondYearChangeB : Float := 1.35
def secondYearChangeG : Float := 1.05
def secondYearChangeD : Float := 1.10

-- Calculate the final values after two years
def finalValueA : Float := initialInvestment * firstYearChangeA * secondYearChangeA
def finalValueB : Float := initialInvestment * firstYearChangeB * secondYearChangeB
def finalValueG : Float := initialInvestment * firstYearChangeG * secondYearChangeG
def finalValueD : Float := initialInvestment * firstYearChangeD * secondYearChangeD

-- Theorem statement - Prove that the final order of the values is B < D < G < A
theorem stock_value_order : finalValueB < finalValueD ∧ finalValueD < finalValueG ∧ finalValueG < finalValueA := by
  sorry

end stock_value_order_l1107_110775


namespace rectangle_sides_l1107_110789

theorem rectangle_sides (x y : ℝ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  (x = 7 / 2 ∧ y = 14 / 3) ∨ (x = 14 / 3 ∧ y = 7 / 2) :=
by {
  sorry
}

end rectangle_sides_l1107_110789


namespace total_clouds_count_l1107_110786

-- Definitions based on the conditions
def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2

-- The theorem statement that needs to be proved
theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds = 78 := by
  -- Definitions
  have h1 : carson_clouds = 12 := rfl
  have h2 : little_brother_clouds = 5 * 12 := rfl
  have h3 : older_sister_clouds = 12 / 2 := rfl
  sorry

end total_clouds_count_l1107_110786


namespace dk_is_odd_l1107_110773

def NTypePermutations (k : ℕ) (x : Fin (3 * k + 1) → ℕ) : Prop :=
  (∀ i j : Fin (k + 1), i < j → x i < x j) ∧
  (∀ i j : Fin (k + 1), i < j → x (k + 1 + i) > x (k + 1 + j)) ∧
  (∀ i j : Fin (k + 1), i < j → x (2 * k + 1 + i) < x (2 * k + 1 + j))

def countNTypePermutations (k : ℕ) : ℕ :=
  sorry -- This would be the count of all N-type permutations, use advanced combinatorics or algorithms

theorem dk_is_odd (k : ℕ) (h : 0 < k) : ∃ d : ℕ, countNTypePermutations k = 2 * d + 1 :=
  sorry

end dk_is_odd_l1107_110773


namespace smallest_positive_angle_equivalent_neg_1990_l1107_110764

theorem smallest_positive_angle_equivalent_neg_1990:
  ∃ k : ℤ, 0 ≤ (θ : ℤ) ∧ θ < 360 ∧ -1990 + 360 * k = θ := by
  use 6
  sorry

end smallest_positive_angle_equivalent_neg_1990_l1107_110764


namespace prob_two_red_balls_in_four_draws_l1107_110709

noncomputable def probability_red_balls (draws : ℕ) (red_in_draw : ℕ) (total_balls : ℕ) (red_balls : ℕ) : ℝ :=
  let prob_red := (red_balls : ℝ) / (total_balls : ℝ)
  let prob_white := 1 - prob_red
  (Nat.choose draws red_in_draw : ℝ) * (prob_red ^ red_in_draw) * (prob_white ^ (draws - red_in_draw))

theorem prob_two_red_balls_in_four_draws :
  probability_red_balls 4 2 10 4 = 0.3456 :=
by
  sorry

end prob_two_red_balls_in_four_draws_l1107_110709


namespace age_twice_in_two_years_l1107_110744

-- conditions
def father_age (S : ℕ) : ℕ := S + 24
def present_son_age : ℕ := 22
def present_father_age : ℕ := father_age present_son_age

-- theorem statement
theorem age_twice_in_two_years (S M Y : ℕ) (h1 : S = present_son_age) (h2 : M = present_father_age) : 
  M + 2 = 2 * (S + 2) :=
by
  sorry

end age_twice_in_two_years_l1107_110744


namespace intersection_interval_l1107_110751

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (x: ℝ) : ℝ := 7 - 2 * x

theorem intersection_interval : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = g x := 
sorry

end intersection_interval_l1107_110751


namespace bins_of_soup_l1107_110702

theorem bins_of_soup (total_bins : ℝ) (bins_of_vegetables : ℝ) (bins_of_pasta : ℝ) 
(h1 : total_bins = 0.75) (h2 : bins_of_vegetables = 0.125) (h3 : bins_of_pasta = 0.5) :
  total_bins - (bins_of_vegetables + bins_of_pasta) = 0.125 := by
  -- proof
  sorry

end bins_of_soup_l1107_110702


namespace bill_project_days_l1107_110793

theorem bill_project_days (naps: ℕ) (hours_per_nap: ℕ) (working_hours: ℕ) : 
  (naps = 6) → (hours_per_nap = 7) → (working_hours = 54) → 
  (naps * hours_per_nap + working_hours) / 24 = 4 := 
by
  intros h1 h2 h3
  sorry

end bill_project_days_l1107_110793


namespace adult_ticket_cost_is_16_l1107_110795

-- Define the problem
def group_size := 6 + 10 -- Total number of people
def child_tickets := 6 -- Number of children
def adult_tickets := 10 -- Number of adults
def child_ticket_cost := 10 -- Cost per child ticket
def total_ticket_cost := 220 -- Total cost for all tickets

-- Prove the cost of an adult ticket
theorem adult_ticket_cost_is_16 : 
  (total_ticket_cost - (child_tickets * child_ticket_cost)) / adult_tickets = 16 := by
  sorry

end adult_ticket_cost_is_16_l1107_110795


namespace x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l1107_110736

theorem x_sq_plus_3x_minus_2_ge_zero (x : ℝ) (h : x ≥ 1) : x^2 + 3 * x - 2 ≥ 0 :=
sorry

theorem neg_x_sq_plus_3x_minus_2_lt_zero (x : ℝ) (h : x < 1) : x^2 + 3 * x - 2 < 0 :=
sorry

end x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l1107_110736


namespace polygon_diagonals_30_l1107_110710

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end polygon_diagonals_30_l1107_110710


namespace length_of_third_side_l1107_110748

theorem length_of_third_side (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 12) (h2 : c = 18) (h3 : B = 2 * C) :
  ∃ a, a = 15 :=
by {
  sorry
}

end length_of_third_side_l1107_110748


namespace parabola_properties_l1107_110730

noncomputable def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties
  (a b c t m n x₀ : ℝ)
  (ha : a > 0)
  (h1 : parabola a b c 1 = m)
  (h4 : parabola a b c 4 = n)
  (ht : t = -b / (2 * a))
  (h3ab : 3 * a + b = 0) 
  (hmnc : m < c ∧ c < n)
  (hx₀ym : parabola a b c x₀ = m) :
  m < n ∧ (1 / 2) < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 :=
  sorry

end parabola_properties_l1107_110730


namespace area_difference_of_square_screens_l1107_110758

theorem area_difference_of_square_screens (d1 d2 : ℝ) (A1 A2 : ℝ) 
  (h1 : d1 = 18) (h2 : d2 = 16) 
  (hA1 : A1 = d1^2 / 2) (hA2 : A2 = d2^2 / 2) : 
  A1 - A2 = 34 := by
  sorry

end area_difference_of_square_screens_l1107_110758


namespace no_three_nat_numbers_with_sum_power_of_three_l1107_110735

noncomputable def powers_of_3 (n : ℕ) : ℕ := 3^n

theorem no_three_nat_numbers_with_sum_power_of_three :
  ¬ ∃ (a b c : ℕ) (k m n : ℕ), a + b = powers_of_3 k ∧ b + c = powers_of_3 m ∧ c + a = powers_of_3 n :=
by
  sorry

end no_three_nat_numbers_with_sum_power_of_three_l1107_110735


namespace cubic_meter_to_cubic_centimeters_l1107_110713

theorem cubic_meter_to_cubic_centimeters :
  (1 : ℝ) ^ 3 = (100 : ℝ) ^ 3 := by
  sorry

end cubic_meter_to_cubic_centimeters_l1107_110713


namespace cubic_coefficient_determination_l1107_110707

def f (x : ℚ) (A B C D : ℚ) : ℚ := A*x^3 + B*x^2 + C*x + D

theorem cubic_coefficient_determination {A B C D : ℚ}
  (h1 : f 1 A B C D = 0)
  (h2 : f (2/3) A B C D = -4)
  (h3 : f (4/5) A B C D = -16/5) :
  A = 15 ∧ B = -37 ∧ C = 30 ∧ D = -8 :=
  sorry

end cubic_coefficient_determination_l1107_110707


namespace sin_arcsin_plus_arctan_l1107_110708

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l1107_110708


namespace diego_annual_savings_l1107_110733

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end diego_annual_savings_l1107_110733


namespace alan_glasses_drank_l1107_110798

-- Definition for the rate of drinking water
def glass_per_minutes := 1 / 20

-- Definition for the total time in minutes
def total_minutes := 5 * 60

-- Theorem stating the number of glasses Alan will drink in the given time
theorem alan_glasses_drank : (glass_per_minutes * total_minutes) = 15 :=
by 
  sorry

end alan_glasses_drank_l1107_110798


namespace part1_part2_l1107_110760

-- Condition definitions
def income2017 : ℝ := 2500
def income2019 : ℝ := 3600
def n : ℕ := 2

-- Part 1: Prove the annual growth rate
theorem part1 (x : ℝ) (hx : income2019 = income2017 * (1 + x) ^ n) : x = 0.2 :=
by sorry

-- Part 2: Prove reaching 4200 yuan with the same growth rate
theorem part2 (hx : income2019 = income2017 * (1 + 0.2) ^ n) : 3600 * (1 + 0.2) ≥ 4200 :=
by sorry

end part1_part2_l1107_110760


namespace gcd_a_b_l1107_110740

def a (n : ℤ) : ℤ := n^5 + 6 * n^3 + 8 * n
def b (n : ℤ) : ℤ := n^4 + 4 * n^2 + 3

theorem gcd_a_b (n : ℤ) : ∃ d : ℤ, d = Int.gcd (a n) (b n) ∧ (d = 1 ∨ d = 3) :=
by
  sorry

end gcd_a_b_l1107_110740


namespace cost_price_of_ball_l1107_110749

theorem cost_price_of_ball (x : ℝ) (h : 17 * x - 5 * x = 720) : x = 60 :=
by {
  sorry
}

end cost_price_of_ball_l1107_110749


namespace soda_consumption_l1107_110783

theorem soda_consumption 
    (dozens : ℕ)
    (people_per_dozen : ℕ)
    (cost_per_box : ℕ)
    (cans_per_box : ℕ)
    (family_members : ℕ)
    (payment_per_member : ℕ)
    (dozens_eq : dozens = 5)
    (people_per_dozen_eq : people_per_dozen = 12)
    (cost_per_box_eq : cost_per_box = 2)
    (cans_per_box_eq : cans_per_box = 10)
    (family_members_eq : family_members = 6)
    (payment_per_member_eq : payment_per_member = 4) :
  (60 * (cans_per_box)) / 60 = 2 :=
by
  -- proof would go here eventually
  sorry

end soda_consumption_l1107_110783


namespace train_length_l1107_110705

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 3.499720022398208 ∧ 
  speed_kmh = 144 ∧ 
  speed_ms = 40 ∧ 
  length = speed_ms * time → 
  length = 139.98880089592832 :=
by sorry

end train_length_l1107_110705


namespace initial_percentage_liquid_X_l1107_110716

theorem initial_percentage_liquid_X (P : ℝ) :
  let original_solution_kg := 8
  let evaporated_water_kg := 2
  let added_solution_kg := 2
  let remaining_solution_kg := original_solution_kg - evaporated_water_kg
  let new_solution_kg := remaining_solution_kg + added_solution_kg
  let new_solution_percentage := 0.25
  let initial_liquid_X_kg := (P / 100) * original_solution_kg
  let final_liquid_X_kg := initial_liquid_X_kg + (P / 100) * added_solution_kg
  let final_liquid_X_kg' := new_solution_percentage * new_solution_kg
  (final_liquid_X_kg = final_liquid_X_kg') → 
  P = 20 :=
by
  intros
  let original_solution_kg_p0 := 8
  let evaporated_water_kg_p1 := 2
  let added_solution_kg_p2 := 2
  let remaining_solution_kg_p3 := (original_solution_kg_p0 - evaporated_water_kg_p1)
  let new_solution_kg_p4 := (remaining_solution_kg_p3 + added_solution_kg_p2)
  let new_solution_percentage : ℝ := 0.25
  let initial_liquid_X_kg_p6 := ((P / 100) * original_solution_kg_p0)
  let final_liquid_X_kg_p7 := initial_liquid_X_kg_p6 + ((P / 100) * added_solution_kg_p2)
  let final_liquid_X_kg_p8 := (new_solution_percentage * new_solution_kg_p4)
  exact sorry

end initial_percentage_liquid_X_l1107_110716


namespace fraction_of_money_left_l1107_110782

theorem fraction_of_money_left 
  (m c : ℝ) 
  (h1 : (1/4 : ℝ) * m = (1/2) * c) : 
  (m - c) / m = (1/2 : ℝ) :=
by
  -- the proof will be written here
  sorry

end fraction_of_money_left_l1107_110782


namespace side_length_of_square_l1107_110779

theorem side_length_of_square (m : ℕ) (a : ℕ) (hm : m = 100) (ha : a^2 = m) : a = 10 :=
by 
  sorry

end side_length_of_square_l1107_110779


namespace regular_12gon_symmetry_and_angle_l1107_110781

theorem regular_12gon_symmetry_and_angle :
  ∀ (L R : ℕ), 
  (L = 12) ∧ (R = 30) → 
  (L + R = 42) :=
by
  -- placeholder for the actual proof
  sorry

end regular_12gon_symmetry_and_angle_l1107_110781


namespace amount_collected_from_ii_and_iii_class_l1107_110712

theorem amount_collected_from_ii_and_iii_class
  (P1 P2 P3 : ℕ) (F1 F2 F3 : ℕ) (total_amount amount_ii_iii : ℕ)
  (H1 : P1 / P2 = 1 / 50)
  (H2 : P1 / P3 = 1 / 100)
  (H3 : F1 / F2 = 5 / 2)
  (H4 : F1 / F3 = 5 / 1)
  (H5 : total_amount = 3575)
  (H6 : total_amount = (P1 * F1) + (P2 * F2) + (P3 * F3))
  (H7 : amount_ii_iii = (P2 * F2) + (P3 * F3)) :
  amount_ii_iii = 3488 := sorry

end amount_collected_from_ii_and_iii_class_l1107_110712


namespace fraction_of_value_l1107_110790

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end fraction_of_value_l1107_110790


namespace combined_PP_curve_l1107_110767

-- Definitions based on the given conditions
def M1 (K : ℝ) : ℝ := 40 - 2 * K
def M2 (K : ℝ) : ℝ := 64 - K ^ 2
def combinedPPC (K1 K2 : ℝ) : ℝ := 128 - 0.5 * K1^2 + 40 - 2 * K2

theorem combined_PP_curve (K : ℝ) :
  (K ≤ 2 → combinedPPC K 0 = 168 - 0.5 * K^2) ∧
  (2 < K ∧ K ≤ 22 → combinedPPC 2 (K - 2) = 170 - 2 * K) ∧
  (22 < K ∧ K ≤ 36 → combinedPPC (K - 20) 20 = 20 * K - 0.5 * K^2 - 72) :=
by
  sorry

end combined_PP_curve_l1107_110767


namespace find_ding_score_l1107_110780

noncomputable def jia_yi_bing_avg_score : ℕ := 89
noncomputable def four_avg_score := jia_yi_bing_avg_score + 2
noncomputable def four_total_score := 4 * four_avg_score
noncomputable def jia_yi_bing_total_score := 3 * jia_yi_bing_avg_score
noncomputable def ding_score := four_total_score - jia_yi_bing_total_score

theorem find_ding_score : ding_score = 97 := 
by
  sorry

end find_ding_score_l1107_110780


namespace green_paint_mixture_l1107_110792

theorem green_paint_mixture :
  ∀ (x : ℝ), 
    let light_green_paint := 5
    let darker_green_paint := x
    let final_paint := light_green_paint + darker_green_paint
    1 + 0.4 * darker_green_paint = 0.25 * final_paint -> x = 5 / 3 := 
by 
  intros x
  let light_green_paint := 5
  let darker_green_paint := x
  let final_paint := light_green_paint + darker_green_paint
  sorry

end green_paint_mixture_l1107_110792


namespace multiplier_for_average_grade_l1107_110706

/-- Conditions -/
def num_of_grades_2 : ℕ := 3
def num_of_grades_3 : ℕ := 4
def num_of_grades_4 : ℕ := 1
def num_of_grades_5 : ℕ := 1
def cash_reward : ℕ := 15

-- Definitions for sums and averages based on the conditions
def sum_of_grades : ℕ :=
  num_of_grades_2 * 2 + num_of_grades_3 * 3 + num_of_grades_4 * 4 + num_of_grades_5 * 5

def total_grades : ℕ :=
  num_of_grades_2 + num_of_grades_3 + num_of_grades_4 + num_of_grades_5

def average_grade : ℕ :=
  sum_of_grades / total_grades

/-- Proof statement -/
theorem multiplier_for_average_grade : cash_reward / average_grade = 5 := by
  sorry

end multiplier_for_average_grade_l1107_110706


namespace age_of_b_l1107_110765

-- Definition of conditions
variable (a b c : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : a + b + c = 12)

-- The statement of the proof problem
theorem age_of_b : b = 4 :=
by {
   sorry
}

end age_of_b_l1107_110765


namespace find_interest_rate_l1107_110720

theorem find_interest_rate
  (P : ℝ) (t : ℕ) (I : ℝ)
  (hP : P = 3000)
  (ht : t = 5)
  (hI : I = 750) :
  ∃ r : ℝ, I = P * r * t / 100 ∧ r = 5 :=
by 
  sorry

end find_interest_rate_l1107_110720


namespace not_sixth_power_of_integer_l1107_110797

theorem not_sixth_power_of_integer (n : ℕ) : ¬ ∃ k : ℤ, 6 * n^3 + 3 = k^6 :=
by
  sorry

end not_sixth_power_of_integer_l1107_110797


namespace marble_ratio_l1107_110715

theorem marble_ratio (A J C : ℕ) (h1 : 3 * (A + J + C) = 60) (h2 : A = 4) (h3 : C = 8) : A / J = 1 / 2 :=
by sorry

end marble_ratio_l1107_110715


namespace binom_8_3_eq_56_l1107_110770

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l1107_110770


namespace y_value_solution_l1107_110755

theorem y_value_solution (y : ℝ) (h : (3 / y) - ((4 / y) * (2 / y)) = 1.5) : 
  y = 1 + Real.sqrt (19 / 3) := 
sorry

end y_value_solution_l1107_110755


namespace find_m_eq_zero_l1107_110746

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end find_m_eq_zero_l1107_110746


namespace find_pair_l1107_110754

theorem find_pair :
  ∃ x y : ℕ, (1984 * x - 1983 * y = 1985) ∧ (x = 27764) ∧ (y = 27777) :=
by
  sorry

end find_pair_l1107_110754


namespace banker_l1107_110728

theorem banker's_discount (BD TD FV : ℝ) (hBD : BD = 18) (hTD : TD = 15) 
(h : BD = TD + (TD^2 / FV)) : FV = 75 := by
  sorry

end banker_l1107_110728


namespace garrison_reinforcement_l1107_110719

theorem garrison_reinforcement (x : ℕ) (h1 : ∀ (n m p : ℕ), n * m = p → x = n - m) :
  (150 * (31 - x) = 450 * 5) → x = 16 :=
by sorry

end garrison_reinforcement_l1107_110719


namespace chloe_probability_l1107_110778

theorem chloe_probability :
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4_prob := 3 / 4
  let neither_multiple_of_4_prob := (non_multiples_of_4_prob) ^ 2
  let at_least_one_multiple_of_4_prob := 1 - neither_multiple_of_4_prob
  at_least_one_multiple_of_4_prob = 7 / 16 := by
  sorry

end chloe_probability_l1107_110778


namespace part_I_part_II_l1107_110785

noncomputable def f (x b c : ℝ) := x^2 + b*x + c

theorem part_I (x_1 x_2 b c : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) :
  b^2 > 2 * (b + 2 * c) :=
sorry

theorem part_II (x_1 x_2 b c t : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) (h5 : 0 < t ∧ t < x_1) :
  f t b c > x_1 :=
sorry

end part_I_part_II_l1107_110785


namespace xiaoli_estimate_smaller_l1107_110734

variable (x y z : ℝ)
variable (hx : x > y) (hz : z > 0)

theorem xiaoli_estimate_smaller :
  (x - z) - (y + z) < x - y := 
by
  sorry

end xiaoli_estimate_smaller_l1107_110734


namespace sin_theta_add_pi_over_3_l1107_110753

theorem sin_theta_add_pi_over_3 (θ : ℝ) (h : Real.cos (π / 6 - θ) = 2 / 3) : 
  Real.sin (θ + π / 3) = 2 / 3 :=
sorry

end sin_theta_add_pi_over_3_l1107_110753


namespace integer_solution_of_floor_equation_l1107_110747

theorem integer_solution_of_floor_equation (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 11) :=
by sorry

end integer_solution_of_floor_equation_l1107_110747


namespace analysis_hours_l1107_110704

theorem analysis_hours (n t : ℕ) (h1 : n = 206) (h2 : t = 1) : n * t = 206 := by
  sorry

end analysis_hours_l1107_110704


namespace journey_total_distance_l1107_110703

def miles_driven : ℕ := 923
def miles_to_go : ℕ := 277
def total_distance : ℕ := 1200

theorem journey_total_distance : miles_driven + miles_to_go = total_distance := by
  sorry

end journey_total_distance_l1107_110703


namespace distance_difference_l1107_110701

-- Given conditions
def speed_train1 : ℕ := 20
def speed_train2 : ℕ := 25
def total_distance : ℕ := 675

-- Define the problem statement
theorem distance_difference : ∃ t : ℝ, (speed_train2 * t - speed_train1 * t) = 75 ∧ (speed_train1 * t + speed_train2 * t) = total_distance := by 
  sorry

end distance_difference_l1107_110701


namespace polyhedron_inequality_proof_l1107_110757

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l1107_110757


namespace factorization_c_minus_d_l1107_110724

theorem factorization_c_minus_d : 
  ∃ (c d : ℤ), (∀ (x : ℤ), (4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d))) ∧ (c - d = 8) :=
by
  sorry

end factorization_c_minus_d_l1107_110724


namespace triangle_right_angle_l1107_110791

theorem triangle_right_angle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B - C) : B = 90 :=
by sorry

end triangle_right_angle_l1107_110791


namespace num_divisors_m2_less_than_m_not_divide_m_l1107_110768

namespace MathProof

def m : ℕ := 2^20 * 3^15 * 5^6

theorem num_divisors_m2_less_than_m_not_divide_m :
  let m2 := m ^ 2
  let total_divisors_m2 := 41 * 31 * 13
  let total_divisors_m := 21 * 16 * 7
  let divisors_m2_less_than_m := (total_divisors_m2 - 1) / 2
  divisors_m2_less_than_m - total_divisors_m = 5924 :=
by sorry

end MathProof

end num_divisors_m2_less_than_m_not_divide_m_l1107_110768


namespace merchant_marking_percentage_l1107_110717

theorem merchant_marking_percentage (L : ℝ) (p : ℝ) (d : ℝ) (c : ℝ) (profit : ℝ) 
  (purchase_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (list_price : ℝ) : 
  L = 100 ∧ p = 30 ∧ d = 20 ∧ c = 20 ∧ profit = 20 ∧ 
  purchase_price = L - L * (p / 100) ∧ 
  marked_price = 109.375 ∧ 
  selling_price = marked_price - marked_price * (d / 100) ∧ 
  selling_price - purchase_price = profit * (selling_price / 100) 
  → marked_price = 109.375 := by sorry

end merchant_marking_percentage_l1107_110717


namespace students_still_inward_l1107_110725

theorem students_still_inward (num_students : ℕ) (turns : ℕ) : (num_students = 36) ∧ (turns = 36) → ∃ n, n = 26 :=
by
  sorry

end students_still_inward_l1107_110725


namespace find_b_l1107_110711

def point := ℝ × ℝ

def dir_vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def scale_vector (v : point) (s : ℝ) : point := (s * v.1, s * v.2)

theorem find_b (p1 p2 : point) (b : ℝ) :
  p1 = (-5, 0) → p2 = (-2, 2) →
  dir_vector p1 p2 = (3, 2) →
  scale_vector (3, 2) (2 / 3) = (2, b) →
  b = 4 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_b_l1107_110711


namespace virginia_sweettarts_l1107_110769

theorem virginia_sweettarts (total_sweettarts : ℕ) (sweettarts_per_person : ℕ) (friends : ℕ) (sweettarts_left : ℕ) 
  (h1 : total_sweettarts = 13) 
  (h2 : sweettarts_per_person = 3) 
  (h3 : total_sweettarts = sweettarts_per_person * (friends + 1) + sweettarts_left) 
  (h4 : sweettarts_left < sweettarts_per_person) :
  friends = 3 :=
by
  sorry

end virginia_sweettarts_l1107_110769


namespace union_of_setA_and_setB_l1107_110742

def setA : Set ℕ := {1, 2, 4}
def setB : Set ℕ := {2, 6}

theorem union_of_setA_and_setB :
  setA ∪ setB = {1, 2, 4, 6} :=
by sorry

end union_of_setA_and_setB_l1107_110742


namespace solve_equation_l1107_110787

theorem solve_equation (x : ℝ) (h₀ : x ≠ -3) (h₁ : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : x = 9 :=
by
  sorry

end solve_equation_l1107_110787


namespace no_such_polyhedron_l1107_110759

theorem no_such_polyhedron (n : ℕ) (S : Fin n → ℝ) (H : ∀ i j : Fin n, i ≠ j → S i ≥ 2 * S j) : False :=
by
  sorry

end no_such_polyhedron_l1107_110759


namespace solve_for_x_l1107_110796

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end solve_for_x_l1107_110796


namespace sector_to_cone_base_area_l1107_110721

theorem sector_to_cone_base_area
  (r_sector : ℝ) (theta : ℝ) (h1 : r_sector = 2) (h2 : theta = 120) :
  ∃ (A : ℝ), A = (4 / 9) * Real.pi :=
by
  sorry

end sector_to_cone_base_area_l1107_110721


namespace heartsuit_3_8_l1107_110762

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end heartsuit_3_8_l1107_110762


namespace prop1_prop3_l1107_110727

def custom_op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

theorem prop1 (x y : ℝ) : custom_op x y = custom_op y x :=
by sorry

theorem prop3 (x : ℝ) : custom_op (x + 1) (x - 1) = custom_op x x - 1 :=
by sorry

end prop1_prop3_l1107_110727


namespace complement_A_is_correct_l1107_110732

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

end complement_A_is_correct_l1107_110732


namespace solve_for_a_l1107_110741

theorem solve_for_a (a x : ℝ) (h : 2 * x + 3 * a = 10) (hx : x = 2) : a = 2 :=
by
  rw [hx] at h
  linarith

end solve_for_a_l1107_110741


namespace remainder_of_k_divided_by_7_l1107_110752

theorem remainder_of_k_divided_by_7 :
  ∃ k < 42, k % 5 = 2 ∧ k % 6 = 5 ∧ k % 7 = 3 :=
by {
  -- The proof is supplied here
  sorry
}

end remainder_of_k_divided_by_7_l1107_110752


namespace loan_amount_calculation_l1107_110700

theorem loan_amount_calculation
  (annual_interest : ℝ) (interest_rate : ℝ) (time : ℝ) (loan_amount : ℝ)
  (h1 : annual_interest = 810)
  (h2 : interest_rate = 0.09)
  (h3 : time = 1)
  (h4 : loan_amount = annual_interest / (interest_rate * time)) :
  loan_amount = 9000 := by
sorry

end loan_amount_calculation_l1107_110700


namespace total_participants_l1107_110763

theorem total_participants
  (F M : ℕ) 
  (half_female_democrats : F / 2 = 125)
  (one_third_democrats : (F + M) / 3 = (125 + M / 4))
  : F + M = 1750 :=
by
  sorry

end total_participants_l1107_110763


namespace average_non_prime_squares_approx_l1107_110726

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of non-prime numbers between 50 and 100
def non_prime_numbers : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70,
   72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,
   92, 93, 94, 95, 96, 98, 99]

-- Define the sum of squares of the elements in a list
def sum_of_squares (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * x + acc) 0

-- Define the count of non-prime numbers
def count_non_prime : ℕ :=
  non_prime_numbers.length

-- Calculate the average
def average_non_prime_squares : ℚ :=
  sum_of_squares non_prime_numbers / count_non_prime

-- Theorem to state that the average of the sum of squares of non-prime numbers
-- between 50 and 100 is approximately 6417.67
theorem average_non_prime_squares_approx :
  abs ((average_non_prime_squares : ℝ) - 6417.67) < 0.01 := 
  sorry

end average_non_prime_squares_approx_l1107_110726


namespace jellybean_total_count_l1107_110718

theorem jellybean_total_count :
  let black := 8
  let green := 2 * black
  let orange := (2 * green) - 5
  let red := orange + 3
  let yellow := black / 2
  let purple := red + 4
  let brown := (green + purple) - 3
  black + green + orange + red + yellow + purple + brown = 166 := by
  -- skipping proof for brevity
  sorry

end jellybean_total_count_l1107_110718


namespace math_problem_l1107_110771

noncomputable def f (x : ℝ) := |Real.exp x - 1|

theorem math_problem (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : x2 > 0)
  (h3 : - Real.exp x1 * Real.exp x2 = -1) :
  (x1 + x2 = 0) ∧
  (0 < (Real.exp x2 + Real.exp x1 - 2) / (x2 - x1)) ∧
  (0 < Real.exp x1 ∧ Real.exp x1 < 1) :=
by
  sorry

end math_problem_l1107_110771


namespace solve_quadratic_eq1_solve_quadratic_eq2_l1107_110738

theorem solve_quadratic_eq1 (x : ℝ) :
  x^2 - 4 * x + 3 = 0 ↔ (x = 3 ∨ x = 1) :=
sorry

theorem solve_quadratic_eq2 (x : ℝ) :
  x^2 - x - 3 = 0 ↔ (x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) :=
sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l1107_110738


namespace trig_identity_l1107_110737

theorem trig_identity :
  (Real.sin (17 * Real.pi / 180) * Real.cos (47 * Real.pi / 180) - 
   Real.sin (73 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = -1/2 := 
by
  sorry

end trig_identity_l1107_110737


namespace mike_initial_cards_l1107_110794

-- Define the conditions
def initial_cards (x : ℕ) := x + 13 = 100

-- Define the proof statement
theorem mike_initial_cards : initial_cards 87 :=
by
  sorry

end mike_initial_cards_l1107_110794


namespace circles_intersect_l1107_110766

def C1 (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1
def C2 (x y a : ℝ) : Prop := (x-a)^2 + (y-1)^2 = 16

theorem circles_intersect (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, C2 x' y' a) ↔ 3 < a ∧ a < 4 :=
sorry

end circles_intersect_l1107_110766


namespace simplify_expr1_simplify_expr2_l1107_110799

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l1107_110799


namespace four_digit_palindrome_divisible_by_11_probability_zero_l1107_110784

theorem four_digit_palindrome_divisible_by_11_probability_zero :
  (∃ a b : ℕ, 2 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (1001 * a + 110 * b) % 11 = 0) = false :=
by sorry

end four_digit_palindrome_divisible_by_11_probability_zero_l1107_110784


namespace solution_condition1_solution_condition2_solution_condition3_solution_condition4_l1107_110777

-- Define the conditions
def Condition1 : Prop :=
  ∃ (total_population box1 box2 sampled : Nat),
  total_population = 30 ∧ box1 = 21 ∧ box2 = 9 ∧ sampled = 10

def Condition2 : Prop :=
  ∃ (total_population produced_by_A produced_by_B sampled : Nat),
  total_population = 30 ∧ produced_by_A = 21 ∧ produced_by_B = 9 ∧ sampled = 10

def Condition3 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 10

def Condition4 : Prop :=
  ∃ (total_population sampled : Nat),
  total_population = 300 ∧ sampled = 50

-- Define the appropriate methods
def LotteryMethod : Prop := ∃ method : String, method = "Lottery method"
def StratifiedSampling : Prop := ∃ method : String, method = "Stratified sampling"
def RandomNumberMethod : Prop := ∃ method : String, method = "Random number method"
def SystematicSampling : Prop := ∃ method : String, method = "Systematic sampling"

-- Statements to prove the appropriate methods for each condition
theorem solution_condition1 : Condition1 → LotteryMethod := by sorry
theorem solution_condition2 : Condition2 → StratifiedSampling := by sorry
theorem solution_condition3 : Condition3 → RandomNumberMethod := by sorry
theorem solution_condition4 : Condition4 → SystematicSampling := by sorry

end solution_condition1_solution_condition2_solution_condition3_solution_condition4_l1107_110777


namespace vertices_divisible_by_three_l1107_110729

namespace PolygonDivisibility

theorem vertices_divisible_by_three (v : Fin 2018 → ℤ) 
  (h_initial : (Finset.univ.sum v) = 1) 
  (h_move : ∀ i : Fin 2018, ∃ j : Fin 2018, abs (v i - v j) = 1) :
  ¬ ∃ (k : Fin 2018 → ℤ), (∀ n : Fin 2018, k n % 3 = 0) :=
by {
  sorry
}

end PolygonDivisibility

end vertices_divisible_by_three_l1107_110729


namespace find_angle_and_area_of_triangle_l1107_110761

theorem find_angle_and_area_of_triangle (a b : ℝ) 
  (h_a : a = Real.sqrt 7) (h_b : b = 2)
  (angle_A : ℝ) (angle_A_eq : angle_A = Real.pi / 3)
  (angle_B : ℝ)
  (vec_m : ℝ × ℝ := (a, Real.sqrt 3 * b))
  (vec_n : ℝ × ℝ := (Real.cos angle_A, Real.sin angle_B))
  (colinear : vec_m.1 * vec_n.2 = vec_m.2 * vec_n.1)
  (sin_A : Real.sin angle_A = (Real.sqrt 3) / 2)
  (cos_A : Real.cos angle_A = 1 / 2) :
  angle_A = Real.pi / 3 ∧ 
  ∃ (c : ℝ), c = 3 ∧
  (1/2) * b * c * Real.sin angle_A = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_angle_and_area_of_triangle_l1107_110761


namespace value_of_m_l1107_110731

theorem value_of_m :
  ∃ m : ℝ, (3 - 1) / (m + 2) = 1 → m = 0 :=
by 
  sorry

end value_of_m_l1107_110731


namespace trapezium_area_l1107_110722

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l1107_110722


namespace proof_goal_l1107_110723

noncomputable def exp_value (k m n : ℕ) : ℤ :=
  (6^k - k^6 + 2^m - 4^m + n^3 - 3^n : ℤ)

theorem proof_goal (k m n : ℕ) (h_k : 18^k ∣ 624938) (h_m : 24^m ∣ 819304) (h_n : n = 2 * k + m) :
  exp_value k m n = 0 := by
  sorry

end proof_goal_l1107_110723
