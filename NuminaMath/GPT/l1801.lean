import Mathlib

namespace divides_number_of_ones_l1801_180132

theorem divides_number_of_ones (n : ℕ) (h1 : ¬(2 ∣ n)) (h2 : ¬(5 ∣ n)) : ∃ k : ℕ, n ∣ ((10^k - 1) / 9) :=
by
  sorry

end divides_number_of_ones_l1801_180132


namespace no_two_adj_or_opposite_same_num_l1801_180154

theorem no_two_adj_or_opposite_same_num :
  ∃ (prob : ℚ), prob = 25 / 648 ∧ 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧
    (A ≠ D ∧ B ≠ E ∧ C ≠ F) ∧ 
    (1 ≤ A ∧ A ≤ 6) ∧ (1 ≤ B ∧ B ≤ 6) ∧ (1 ≤ C ∧ C ≤ 6) ∧ 
    (1 ≤ D ∧ D ≤ 6) ∧ (1 ≤ E ∧ E ≤ 6) ∧ (1 ≤ F ∧ F ≤ 6) →
    prob = (6 * 5 * 4 * 5 * 3 * 3) / (6^6) := 
sorry

end no_two_adj_or_opposite_same_num_l1801_180154


namespace billy_picked_36_dandelions_initially_l1801_180164

namespace Dandelions

/-- The number of dandelions Billy picked initially. -/
def billy_initial (B : ℕ) : ℕ := B

/-- The number of dandelions George picked initially. -/
def george_initial (B : ℕ) : ℕ := B / 3

/-- The additional dandelions picked by Billy and George respectively. -/
def additional_dandelions : ℕ := 10

/-- The total dandelions picked by Billy and George initially and additionally. -/
def total_dandelions (B : ℕ) : ℕ :=
  billy_initial B + additional_dandelions + george_initial B + additional_dandelions

/-- The average number of dandelions picked by both Billy and George, given as 34. -/
def average_dandelions (total : ℕ) : Prop := total / 2 = 34

/-- The main theorem stating that Billy picked 36 dandelions initially. -/
theorem billy_picked_36_dandelions_initially :
  ∀ B : ℕ, average_dandelions (total_dandelions B) ↔ B = 36 :=
by
  intro B
  sorry

end Dandelions

end billy_picked_36_dandelions_initially_l1801_180164


namespace exists_prime_divisor_in_sequence_l1801_180166

theorem exists_prime_divisor_in_sequence
  (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d)
  (a : ℕ → ℕ)
  (h0 : a 1 = c)
  (hs : ∀ n, a (n+1) = a n ^ d + c) :
  ∀ (n : ℕ), 2 ≤ n →
  ∃ (p : ℕ), Prime p ∧ p ∣ a n ∧ ∀ i, 1 ≤ i ∧ i < n → ¬ p ∣ a i := sorry

end exists_prime_divisor_in_sequence_l1801_180166


namespace dakotas_medical_bill_l1801_180153

theorem dakotas_medical_bill :
  let days_in_hospital := 3
  let hospital_bed_cost_per_day := 900
  let specialists_rate_per_hour := 250
  let specialist_minutes_per_day := 15
  let num_specialists := 2
  let ambulance_cost := 1800

  let hospital_bed_cost := hospital_bed_cost_per_day * days_in_hospital
  let specialists_total_minutes := specialist_minutes_per_day * num_specialists
  let specialists_hours := specialists_total_minutes / 60.0
  let specialists_cost := specialists_hours * specialists_rate_per_hour

  let total_medical_bill := hospital_bed_cost + specialists_cost + ambulance_cost

  total_medical_bill = 4625 := 
by
  sorry

end dakotas_medical_bill_l1801_180153


namespace calc_remainder_l1801_180151

theorem calc_remainder : 
  (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 +
   90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 -
   90^7 * Nat.choose 10 7 + 90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 +
   90^10 * Nat.choose 10 10) % 88 = 1 := 
by sorry

end calc_remainder_l1801_180151


namespace nigella_sold_3_houses_l1801_180190

noncomputable def houseA_cost : ℝ := 60000
noncomputable def houseB_cost : ℝ := 3 * houseA_cost
noncomputable def houseC_cost : ℝ := 2 * houseA_cost - 110000
noncomputable def commission_rate : ℝ := 0.02

noncomputable def houseA_commission : ℝ := houseA_cost * commission_rate
noncomputable def houseB_commission : ℝ := houseB_cost * commission_rate
noncomputable def houseC_commission : ℝ := houseC_cost * commission_rate

noncomputable def total_commission : ℝ := houseA_commission + houseB_commission + houseC_commission
noncomputable def base_salary : ℝ := 3000
noncomputable def total_earnings : ℝ := base_salary + total_commission

theorem nigella_sold_3_houses 
  (H1 : total_earnings = 8000) 
  (H2 : houseA_cost = 60000) 
  (H3 : houseB_cost = 3 * houseA_cost) 
  (H4 : houseC_cost = 2 * houseA_cost - 110000) 
  (H5 : commission_rate = 0.02) :
  3 = 3 :=
by 
  -- Proof not required
  sorry

end nigella_sold_3_houses_l1801_180190


namespace calculation_is_correct_l1801_180192

-- Define the numbers involved in the calculation
def a : ℝ := 12.05
def b : ℝ := 5.4
def c : ℝ := 0.6

-- Expected result of the calculation
def expected_result : ℝ := 65.67

-- Prove that the calculation is correct
theorem calculation_is_correct : (a * b + c) = expected_result :=
by
  sorry

end calculation_is_correct_l1801_180192


namespace arithmetic_expression_evaluation_l1801_180174

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l1801_180174


namespace function_y_increases_when_x_gt_1_l1801_180183

theorem function_y_increases_when_x_gt_1 :
  ∀ (x : ℝ), (x > 1 → 2*x^2 > 2*(x-1)^2) :=
by
  sorry

end function_y_increases_when_x_gt_1_l1801_180183


namespace travel_time_correct_l1801_180115

def luke_bus_to_work : ℕ := 70
def paula_bus_to_work : ℕ := (70 * 3) / 5
def jane_train_to_work : ℕ := 120
def michael_cycle_to_work : ℕ := 120 / 4

def luke_bike_back_home : ℕ := 70 * 5
def paula_bus_back_home: ℕ := paula_bus_to_work
def jane_train_back_home : ℕ := 120 * 2
def michael_cycle_back_home : ℕ := michael_cycle_to_work

def luke_total_travel : ℕ := luke_bus_to_work + luke_bike_back_home
def paula_total_travel : ℕ := paula_bus_to_work + paula_bus_back_home
def jane_total_travel : ℕ := jane_train_to_work + jane_train_back_home
def michael_total_travel : ℕ := michael_cycle_to_work + michael_cycle_back_home

def total_travel_time : ℕ := luke_total_travel + paula_total_travel + jane_total_travel + michael_total_travel

theorem travel_time_correct : total_travel_time = 924 :=
by sorry

end travel_time_correct_l1801_180115


namespace gcd_C_D_eq_6_l1801_180133

theorem gcd_C_D_eq_6
  (C D : ℕ)
  (h_lcm : Nat.lcm C D = 180)
  (h_ratio : C = 5 * D / 6) :
  Nat.gcd C D = 6 := 
by
  sorry

end gcd_C_D_eq_6_l1801_180133


namespace certain_number_105_l1801_180146

theorem certain_number_105 (a x : ℕ) (h0 : a = 105) (h1 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end certain_number_105_l1801_180146


namespace number_of_columns_per_section_l1801_180188

variables (S C : ℕ)

-- Define the first condition: S * C + (S - 1) / 2 = 1223
def condition1 := S * C + (S - 1) / 2 = 1223

-- Define the second condition: S = 2 * C + 5
def condition2 := S = 2 * C + 5

-- Formulate the theorem that C = 23 given the two conditions
theorem number_of_columns_per_section
  (h1 : condition1 S C)
  (h2 : condition2 S C) :
  C = 23 :=
sorry

end number_of_columns_per_section_l1801_180188


namespace quadratic_neq_l1801_180180

theorem quadratic_neq (m : ℝ) : (m-2) ≠ 0 ↔ m ≠ 2 :=
sorry

end quadratic_neq_l1801_180180


namespace bobArrivesBefore845Prob_l1801_180159

noncomputable def probabilityBobBefore845 (totalTime: ℕ) (cutoffTime: ℕ) : ℚ :=
  let totalArea := (totalTime * totalTime) / 2
  let areaOfInterest := (cutoffTime * cutoffTime) / 2
  (areaOfInterest : ℚ) / totalArea

theorem bobArrivesBefore845Prob (totalTime: ℕ) (cutoffTime: ℕ) (ht: totalTime = 60) (hc: cutoffTime = 45) :
  probabilityBobBefore845 totalTime cutoffTime = 9 / 16 := by
  sorry

end bobArrivesBefore845Prob_l1801_180159


namespace remainder_of_144_div_k_l1801_180107

theorem remainder_of_144_div_k
  (k : ℕ)
  (h1 : 0 < k)
  (h2 : 120 % k^2 = 12) :
  144 % k = 0 :=
by
  sorry

end remainder_of_144_div_k_l1801_180107


namespace Brandy_caffeine_intake_l1801_180103

theorem Brandy_caffeine_intake :
  let weight := 60
  let recommended_limit_per_kg := 2.5
  let tolerance := 50
  let coffee_cups := 2
  let coffee_per_cup := 95
  let energy_drinks := 4
  let caffeine_per_energy_drink := 120
  let max_safe_caffeine := weight * recommended_limit_per_kg + tolerance
  let caffeine_from_coffee := coffee_cups * coffee_per_cup
  let caffeine_from_energy_drinks := energy_drinks * caffeine_per_energy_drink
  let total_caffeine_consumed := caffeine_from_coffee + caffeine_from_energy_drinks
  max_safe_caffeine - total_caffeine_consumed = -470 := 
by
  sorry

end Brandy_caffeine_intake_l1801_180103


namespace total_expenditure_is_3000_l1801_180152

/-- Define the Hall dimensions -/
def length : ℝ := 20
def width : ℝ := 15
def cost_per_square_meter : ℝ := 10

/-- Statement to prove --/
theorem total_expenditure_is_3000 
  (h_length : length = 20)
  (h_width : width = 15)
  (h_cost : cost_per_square_meter = 10) : 
  length * width * cost_per_square_meter = 3000 :=
sorry

end total_expenditure_is_3000_l1801_180152


namespace students_per_group_l1801_180186

-- Definitions for conditions
def number_of_boys : ℕ := 28
def number_of_girls : ℕ := 4
def number_of_groups : ℕ := 8
def total_students : ℕ := number_of_boys + number_of_girls

-- The Theorem we want to prove
theorem students_per_group : total_students / number_of_groups = 4 := by
  sorry

end students_per_group_l1801_180186


namespace least_number_to_be_added_l1801_180135

theorem least_number_to_be_added (k : ℕ) (h₁ : Nat.Prime 29) (h₂ : Nat.Prime 37) (H : Nat.gcd 29 37 = 1) : 
  (433124 + k) % Nat.lcm 29 37 = 0 → k = 578 :=
by 
  sorry

end least_number_to_be_added_l1801_180135


namespace solve_for_x_l1801_180162

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 :=
by
  -- reduce the problem to its final steps
  sorry

end solve_for_x_l1801_180162


namespace Mark_marbles_correct_l1801_180194

def Connie_marbles : ℕ := 323
def Juan_marbles : ℕ := Connie_marbles + 175
def Mark_marbles : ℕ := 3 * Juan_marbles

theorem Mark_marbles_correct : Mark_marbles = 1494 := 
by
  sorry

end Mark_marbles_correct_l1801_180194


namespace angle_difference_l1801_180139

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference_l1801_180139


namespace gcf_360_180_l1801_180167

theorem gcf_360_180 : Nat.gcd 360 180 = 180 :=
by
  sorry

end gcf_360_180_l1801_180167


namespace three_digit_number_ends_with_same_three_digits_l1801_180120

theorem three_digit_number_ends_with_same_three_digits (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, k ≥ 1 → N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) := 
sorry

end three_digit_number_ends_with_same_three_digits_l1801_180120


namespace roots_of_poly_l1801_180163

theorem roots_of_poly (a b c : ℂ) :
  ∀ x, x = a ∨ x = b ∨ x = c → x^4 - a*x^3 - b*x + c = 0 :=
sorry

end roots_of_poly_l1801_180163


namespace difference_between_numbers_l1801_180191

noncomputable def L : ℕ := 1614
noncomputable def Q : ℕ := 6
noncomputable def R : ℕ := 15

theorem difference_between_numbers (S : ℕ) (h : L = Q * S + R) : L - S = 1348 :=
by {
  -- proof skipped
  sorry
}

end difference_between_numbers_l1801_180191


namespace music_library_avg_disk_space_per_hour_l1801_180157

theorem music_library_avg_disk_space_per_hour 
  (days_of_music: ℕ) (total_space_MB: ℕ) (hours_in_day: ℕ) 
  (h1: days_of_music = 15) 
  (h2: total_space_MB = 18000) 
  (h3: hours_in_day = 24) : 
  (total_space_MB / (days_of_music * hours_in_day)) = 50 := 
by
  sorry

end music_library_avg_disk_space_per_hour_l1801_180157


namespace train_speed_is_60_kmph_l1801_180141

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end train_speed_is_60_kmph_l1801_180141


namespace remainder_of_12345678910_div_101_l1801_180179

theorem remainder_of_12345678910_div_101 :
  12345678910 % 101 = 31 :=
sorry

end remainder_of_12345678910_div_101_l1801_180179


namespace part1_part2_l1801_180124

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1 / 3) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) :
  (a ≠ 0 → ( a > 0 ↔ -1/a < x ∧ x < 1)
  ∧ (a = 0 ↔ x < 1)
  ∧ (-1 < a ∧ a < 0 ↔ x < 1 ∨ x > -1/a)
  ∧ (a = -1 ↔ x ≠ 1)
  ∧ (a < -1 ↔ x < -1/a ∨ x > 1)) :=
sorry

end part1_part2_l1801_180124


namespace hannah_age_double_july_age_20_years_ago_l1801_180121

/-- Define the current ages of July (J) and her husband (H) -/
def current_age_july : ℕ := 23
def current_age_husband : ℕ := 25

/-- Assertion that July's husband is 2 years older than her -/
axiom husband_older : current_age_husband = current_age_july + 2

/-- We denote the ages 20 years ago -/
def age_july_20_years_ago := current_age_july - 20
def age_hannah_20_years_ago := current_age_husband - 20 - 2 * (current_age_july - 20)

theorem hannah_age_double_july_age_20_years_ago :
  age_hannah_20_years_ago = 6 :=
by sorry

end hannah_age_double_july_age_20_years_ago_l1801_180121


namespace no_integer_solutions_l1801_180199

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
  -x^2 + 4 * y * z + 3 * z^2 = 36 ∧
  x^2 + 2 * x * y + 9 * z^2 = 121 → false :=
by
  sorry

end no_integer_solutions_l1801_180199


namespace fifth_equation_sum_first_17_even_sum_even_28_to_50_l1801_180113

-- Define a function to sum the first n even numbers
def sum_even (n : ℕ) : ℕ := n * (n + 1)

-- Part (1) According to the pattern, write down the ⑤th equation
theorem fifth_equation : sum_even 5 = 30 := by
  sorry

-- Part (2) Calculate according to this pattern:
-- ① Sum of first 17 even numbers
theorem sum_first_17_even : sum_even 17 = 306 := by
  sorry

-- ② Sum of even numbers from 28 to 50
theorem sum_even_28_to_50 : 
  let sum_even_50 := sum_even 25
  let sum_even_26 := sum_even 13
  sum_even_50 - sum_even_26 = 468 := by
  sorry

end fifth_equation_sum_first_17_even_sum_even_28_to_50_l1801_180113


namespace consecutive_numbers_l1801_180189

theorem consecutive_numbers (x : ℕ) (h : (4 * x + 2) * (4 * x^2 + 6 * x + 6) = 3 * (4 * x^3 + 4 * x^2 + 18 * x + 8)) :
  x = 2 :=
sorry

end consecutive_numbers_l1801_180189


namespace find_m_l1801_180148

theorem find_m (x y m : ℝ) 
  (h1 : x + y = 8)
  (h2 : y - m * x = 7)
  (h3 : y - x = 7.5) : m = 3 := 
  sorry

end find_m_l1801_180148


namespace number_of_second_graders_l1801_180129

-- Define the number of kindergartners
def kindergartners : ℕ := 34

-- Define the number of first graders
def first_graders : ℕ := 48

-- Define the total number of students
def total_students : ℕ := 120

-- Define the proof statement
theorem number_of_second_graders : total_students - (kindergartners + first_graders) = 38 := by
  -- omit the proof details
  sorry

end number_of_second_graders_l1801_180129


namespace simplify_fraction_l1801_180158

theorem simplify_fraction : 1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 :=
by
sorry

end simplify_fraction_l1801_180158


namespace sum_of_first_11_odd_numbers_l1801_180127

theorem sum_of_first_11_odd_numbers : 
  (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) = 121 :=
by
  sorry

end sum_of_first_11_odd_numbers_l1801_180127


namespace product_of_digits_of_N_l1801_180134

theorem product_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2485) : 
  (N.digits 10).prod = 0 :=
sorry

end product_of_digits_of_N_l1801_180134


namespace probability_point_in_region_l1801_180109

theorem probability_point_in_region (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 2010) 
  (h2 : 0 ≤ y ∧ y ≤ 2009) 
  (h3 : ∃ (u v : ℝ), (u, v) = (x, y) ∧ x > 2 * y ∧ y > 500) : 
  ∃ p : ℚ, p = 1505 / 4018 := 
sorry

end probability_point_in_region_l1801_180109


namespace compare_logs_l1801_180176

theorem compare_logs (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 3)
                              (h2 : b = Real.log 8 / Real.log 4)
                              (h3 : c = Real.log 10 / Real.log 5) : 
                              a > b ∧ b > c :=
by
  sorry

end compare_logs_l1801_180176


namespace largest_even_number_l1801_180171

theorem largest_even_number (x : ℤ) 
  (h : x + (x + 2) + (x + 4) = x + 18) : x + 4 = 10 :=
by
  sorry

end largest_even_number_l1801_180171


namespace complex_expression_l1801_180160

theorem complex_expression (z : ℂ) (i : ℂ) (h1 : z^2 + 1 = 0) (h2 : i^2 = -1) : 
  (z^4 + i) * (z^4 - i) = 0 :=
sorry

end complex_expression_l1801_180160


namespace change_in_opinion_difference_l1801_180125

theorem change_in_opinion_difference :
  let initially_liked_pct := 0.4;
  let initially_disliked_pct := 0.6;
  let finally_liked_pct := 0.8;
  let finally_disliked_pct := 0.2;
  let max_change := finally_liked_pct + (initially_disliked_pct - finally_disliked_pct);
  let min_change := finally_liked_pct - initially_liked_pct;
  max_change - min_change = 0.2 :=
by
  sorry

end change_in_opinion_difference_l1801_180125


namespace luis_can_make_sum_multiple_of_4_l1801_180187

noncomputable def sum_of_dice (dice: List ℕ) : ℕ :=
  dice.sum 

theorem luis_can_make_sum_multiple_of_4 (d1 d2 d3: ℕ) 
  (h1: 1 ≤ d1 ∧ d1 ≤ 6) 
  (h2: 1 ≤ d2 ∧ d2 ≤ 6) 
  (h3: 1 ≤ d3 ∧ d3 ≤ 6) : 
  ∃ (dice: List ℕ), dice.length = 3 ∧ 
  sum_of_dice dice % 4 = 0 := 
by
  sorry

end luis_can_make_sum_multiple_of_4_l1801_180187


namespace cube_volume_l1801_180128

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end cube_volume_l1801_180128


namespace probability_two_people_between_l1801_180181

theorem probability_two_people_between (total_people : ℕ) (favorable_arrangements : ℕ) (total_arrangements : ℕ) :
  total_people = 6 ∧ favorable_arrangements = 144 ∧ total_arrangements = 720 →
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 5 :=
by
  intros h
  -- We substitute the given conditions
  have ht : total_people = 6 := h.1
  have hf : favorable_arrangements = 144 := h.2.1
  have ha : total_arrangements = 720 := h.2.2
  -- We need to calculate the probability considering the favorable and total arrangements
  sorry

end probability_two_people_between_l1801_180181


namespace find_numbers_l1801_180168

theorem find_numbers (A B: ℕ) (h1: A + B = 581) (h2: (Nat.lcm A B) / (Nat.gcd A B) = 240) : 
  (A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560) :=
by
  sorry

end find_numbers_l1801_180168


namespace triangle_angles_l1801_180185

theorem triangle_angles (r_a r_b r_c R : ℝ) (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
  ∃ (α β γ : ℝ), α = 90 ∧ γ = 60 ∧ β = 30 :=
by
  sorry

end triangle_angles_l1801_180185


namespace valid_call_time_at_15_l1801_180136

def time_difference := 5 -- Beijing is 5 hours ahead of Moscow

def beijing_start_time := 14 -- Start time in Beijing corresponding to 9:00 in Moscow
def beijing_end_time := 17  -- End time in Beijing corresponding to 17:00 in Beijing

-- Define the call time in Beijing
def call_time_beijing := 15

-- The time window during which they can start the call in Beijing
def valid_call_time (t : ℕ) : Prop :=
  beijing_start_time <= t ∧ t <= beijing_end_time

-- The theorem to prove that 15:00 is a valid call time in Beijing
theorem valid_call_time_at_15 : valid_call_time call_time_beijing :=
by
  sorry

end valid_call_time_at_15_l1801_180136


namespace total_cookies_l1801_180117

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l1801_180117


namespace parabola_standard_equation_l1801_180122

theorem parabola_standard_equation (h : ∀ y, y = 1/2) : ∃ c : ℝ, c = -2 ∧ (∀ x y, x^2 = c * y) :=
by
  -- Considering 'h' provides the condition for the directrix
  sorry

end parabola_standard_equation_l1801_180122


namespace sin_double_angle_l1801_180196

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
by
  sorry

end sin_double_angle_l1801_180196


namespace juan_faster_than_peter_l1801_180178

theorem juan_faster_than_peter (J : ℝ) :
  (Peter_speed : ℝ) = 5.0 →
  (time : ℝ) = 1.5 →
  (distance_apart : ℝ) = 19.5 →
  (J + 5.0) * time = distance_apart →
  J - 5.0 = 3 := 
by
  intros Peter_speed_eq time_eq distance_apart_eq relative_speed_eq
  sorry

end juan_faster_than_peter_l1801_180178


namespace rectangle_area_1600_l1801_180108

theorem rectangle_area_1600
  (l w : ℝ)
  (h1 : l = 4 * w)
  (h2 : 2 * l + 2 * w = 200) :
  l * w = 1600 :=
by
  sorry

end rectangle_area_1600_l1801_180108


namespace cos_double_angle_l1801_180140

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = 1 / 3) : 
  Real.cos (2 * α) = -7 / 9 := 
by 
  sorry

end cos_double_angle_l1801_180140


namespace avg_speed_l1801_180169

variable (d1 d2 t1 t2 : ℕ)

-- Conditions
def distance_first_hour : ℕ := 80
def distance_second_hour : ℕ := 40
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Ensure that total distance and total time are defined correctly from conditions
def total_distance : ℕ := distance_first_hour + distance_second_hour
def total_time : ℕ := time_first_hour + time_second_hour

-- Theorem to prove the average speed
theorem avg_speed : total_distance / total_time = 60 := by
  sorry

end avg_speed_l1801_180169


namespace ceil_sqrt_fraction_eq_neg2_l1801_180142

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l1801_180142


namespace expected_value_of_boy_girl_pairs_l1801_180143

noncomputable def expected_value_of_T (boys girls : ℕ) : ℚ :=
  24 * ((boys / 24) * (girls / 23) + (girls / 24) * (boys / 23))

theorem expected_value_of_boy_girl_pairs (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 14) :
  expected_value_of_T boys girls = 12 :=
by
  rw [h_boys, h_girls]
  norm_num
  sorry

end expected_value_of_boy_girl_pairs_l1801_180143


namespace average_price_per_book_l1801_180111

theorem average_price_per_book (books1_cost : ℕ) (books1_count : ℕ)
    (books2_cost : ℕ) (books2_count : ℕ)
    (h1 : books1_cost = 6500) (h2 : books1_count = 65)
    (h3 : books2_cost = 2000) (h4 : books2_count = 35) :
    (books1_cost + books2_cost) / (books1_count + books2_count) = 85 :=
by
    sorry

end average_price_per_book_l1801_180111


namespace simplify_fraction_l1801_180161

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hx2 : x^2 - (1 / y) ≠ 0) (hy2 : y^2 - (1 / x) ≠ 0) :
  (x^2 - 1 / y) / (y^2 - 1 / x) = (x * (x^2 * y - 1)) / (y * (y^2 * x - 1)) :=
sorry

end simplify_fraction_l1801_180161


namespace find_number_l1801_180145

-- Given conditions:
def sum_and_square (n : ℕ) : Prop := n^2 + n = 252
def is_factor (n d : ℕ) : Prop := d % n = 0

-- Equivalent proof problem statement
theorem find_number : ∃ n : ℕ, sum_and_square n ∧ is_factor n 180 ∧ n > 0 ∧ n = 14 :=
by
  sorry

end find_number_l1801_180145


namespace arithmetic_sequence_a9_l1801_180149

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Assume arithmetic sequence: a(n) = a1 + (n-1)d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Given conditions
axiom condition1 : arithmetic_sequence a d 5 + arithmetic_sequence a d 7 = 16
axiom condition2 : arithmetic_sequence a d 3 = 1

-- Prove that a₉ = 15
theorem arithmetic_sequence_a9 : arithmetic_sequence a d 9 = 15 := by
  sorry

end arithmetic_sequence_a9_l1801_180149


namespace find_angle_A_l1801_180137

noncomputable def angle_A (a b c S : ℝ) := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

theorem find_angle_A (a b c S : ℝ) (hb : 0 < b) (hc : 0 < c) (hS : S = (1/2) * b * c * Real.sin (angle_A a b c S)) 
    (h_eq : b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S) : 
    angle_A a b c S = π / 6 := by 
  sorry

end find_angle_A_l1801_180137


namespace fish_ratio_l1801_180184

theorem fish_ratio (k : ℕ) (kendra_fish : ℕ) (home_fish : ℕ)
    (h1 : kendra_fish = 30)
    (h2 : home_fish = 87)
    (h3 : k - 3 + kendra_fish = home_fish) :
  k = 60 ∧ (k / 3, kendra_fish / 3) = (19, 10) :=
by
  sorry

end fish_ratio_l1801_180184


namespace table_length_l1801_180102

theorem table_length (L : ℕ) (H1 : ∃ n : ℕ, 80 = n * L)
  (H2 : L ≥ 16) (H3 : ∃ m : ℕ, 16 = m * 4)
  (H4 : L % 4 = 0) : L = 20 := by 
sorry

end table_length_l1801_180102


namespace correct_inequality_l1801_180150

theorem correct_inequality (x : ℝ) : (1 / (x^2 + 1)) > (1 / (x^2 + 2)) :=
by {
  -- Lean proof steps would be here, but we will use 'sorry' instead to indicate the proof is omitted.
  sorry
}

end correct_inequality_l1801_180150


namespace inequality_proof_l1801_180144

theorem inequality_proof
  (a b c d e f : ℝ)
  (h1 : 1 ≤ a)
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d)
  (h5 : d ≤ e)
  (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := 
by 
  sorry

end inequality_proof_l1801_180144


namespace meeting_point_l1801_180193

def same_start (x : ℝ) (y : ℝ) : Prop := x = y

def walk_time (x : ℝ) (y : ℝ) (t : ℝ) : Prop := 
  x * t + y * t = 24

def hector_speed (s : ℝ) : ℝ := s

def jane_speed (s : ℝ) : ℝ := 3 * s

theorem meeting_point (s t : ℝ) :
  same_start 0 0 ∧ walk_time (hector_speed s) (jane_speed s) t → t = 6 / s ∧ (6 : ℝ) = 6 :=
by
  intros h
  sorry

end meeting_point_l1801_180193


namespace karen_wrong_questions_l1801_180119

theorem karen_wrong_questions (k l n : ℕ) (h1 : k + l = 6 + n) (h2 : k + n = l + 9) : k = 6 := 
by
  sorry

end karen_wrong_questions_l1801_180119


namespace calculate_gross_income_l1801_180105
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end calculate_gross_income_l1801_180105


namespace cordelia_bleach_time_l1801_180131

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end cordelia_bleach_time_l1801_180131


namespace sum_abs_binom_coeff_l1801_180110

theorem sum_abs_binom_coeff (a a1 a2 a3 a4 a5 a6 a7 : ℤ)
    (h : (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
    |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3 ^ 7 - 1 := sorry

end sum_abs_binom_coeff_l1801_180110


namespace sum_of_coefficients_l1801_180155

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a_0 + a_1 * (x + 3) + 
           a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 + 
           a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + 
           a_8 * (x + 3)^8 + a_9 * (x + 3)^9 + a_10 * (x + 3)^10) →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 9 := 
by
  -- proof steps skipped
  sorry

end sum_of_coefficients_l1801_180155


namespace total_goals_in_league_l1801_180106

variables (g1 g2 T : ℕ)

-- Conditions
def equal_goals : Prop := g1 = g2
def players_goals : Prop := g1 = 30
def total_goals_percentage : Prop := (g1 + g2) * 5 = T

-- Theorem to prove: Given the conditions, the total number of goals T should be 300
theorem total_goals_in_league (h1 : equal_goals g1 g2) (h2 : players_goals g1) (h3 : total_goals_percentage g1 g2 T) : T = 300 :=
sorry

end total_goals_in_league_l1801_180106


namespace roots_quadratic_l1801_180138

theorem roots_quadratic (a b : ℝ) (h₁ : a + b = 6) (h₂ : a * b = 8) :
  a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 :=
by
  sorry

end roots_quadratic_l1801_180138


namespace number_of_freshmen_l1801_180177

theorem number_of_freshmen (n : ℕ) : n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 → n = 265 := by
  sorry

end number_of_freshmen_l1801_180177


namespace trajectory_of_midpoint_l1801_180165

theorem trajectory_of_midpoint (A B P : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : ∃ m n : ℝ, B = (m, n) ∧ n^2 = 2 * m)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.2 - 2)^2 = P.1 - 1 :=
sorry

end trajectory_of_midpoint_l1801_180165


namespace percent_students_elected_to_learn_from_home_l1801_180126

theorem percent_students_elected_to_learn_from_home (H : ℕ) : 
  (100 - H) / 2 = 30 → H = 40 := 
by
  sorry

end percent_students_elected_to_learn_from_home_l1801_180126


namespace archer_hits_less_than_8_l1801_180104

variables (P10 P9 P8 : ℝ)

-- Conditions
def hitting10_ring := P10 = 0.3
def hitting9_ring := P9 = 0.3
def hitting8_ring := P8 = 0.2

-- Statement to prove
theorem archer_hits_less_than_8 (P10 P9 P8 : ℝ)
  (h10 : hitting10_ring P10)
  (h9 : hitting9_ring P9)
  (h8 : hitting8_ring P8)
  (mutually_exclusive: P10 + P9 + P8 <= 1):
  1 - (P10 + P9 + P8) = 0.2 :=
by
  -- Here goes the proof 
  sorry

end archer_hits_less_than_8_l1801_180104


namespace range_of_a_l1801_180123

def S : Set ℝ := {x | (x - 2) ^ 2 > 9 }
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8 }

theorem range_of_a (a : ℝ) : (S ∪ T a) = Set.univ ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l1801_180123


namespace setC_not_pythagorean_l1801_180172

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l1801_180172


namespace vitamin_D_scientific_notation_l1801_180170

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l1801_180170


namespace germination_estimate_l1801_180156

theorem germination_estimate (germination_rate : ℝ) (total_pounds : ℝ) 
  (hrate_nonneg : 0 ≤ germination_rate) (hrate_le_one : germination_rate ≤ 1) 
  (h_germination_value : germination_rate = 0.971) 
  (h_total_pounds_value : total_pounds = 1000) : 
  total_pounds * (1 - germination_rate) = 29 := 
by 
  sorry

end germination_estimate_l1801_180156


namespace specified_time_is_30_total_constuction_cost_is_180000_l1801_180147

noncomputable def specified_time (x : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  (teamA_rate + teamB_rate) * 15 + 5 * teamA_rate = 1

theorem specified_time_is_30 : specified_time 30 :=
  by 
    sorry

noncomputable def total_constuction_cost (x : ℕ) (costA : ℕ) (costB : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  let total_time := 1 / (teamA_rate + teamB_rate)
  total_time * (costA + costB)

theorem total_constuction_cost_is_180000 : total_constuction_cost 30 6500 3500 = 180000 :=
  by 
    sorry

end specified_time_is_30_total_constuction_cost_is_180000_l1801_180147


namespace inequality_proof_l1801_180116

noncomputable def inequality_holds (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop :=
  (a ^ 2) / (b - 1) + (b ^ 2) / (a - 1) ≥ 8

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  inequality_holds a b ha hb :=
sorry

end inequality_proof_l1801_180116


namespace transformed_parabola_l1801_180175

theorem transformed_parabola (x : ℝ) : 
  (λ x => -x^2 + 1) (x - 2) - 2 = - (x - 2)^2 - 1 := 
by 
  sorry 

end transformed_parabola_l1801_180175


namespace yellow_mugs_count_l1801_180118

variables (R B Y O : ℕ)
variables (B_eq_3R : B = 3 * R)
variables (R_eq_Y_div_2 : R = Y / 2)
variables (O_eq_4 : O = 4)
variables (mugs_eq_40 : R + B + Y + O = 40)

theorem yellow_mugs_count : Y = 12 :=
by 
  sorry

end yellow_mugs_count_l1801_180118


namespace solve_quadratic_l1801_180112

theorem solve_quadratic : ∀ x, x^2 - 4 * x + 3 = 0 ↔ x = 3 ∨ x = 1 := 
by
  sorry

end solve_quadratic_l1801_180112


namespace Mark_time_spent_l1801_180182

theorem Mark_time_spent :
  let parking_time := 5
  let walking_time := 3
  let long_wait_time := 30
  let short_wait_time := 10
  let long_wait_days := 2
  let short_wait_days := 3
  let work_days := 5
  (parking_time + walking_time) * work_days + 
    long_wait_time * long_wait_days + 
    short_wait_time * short_wait_days = 130 :=
by
  sorry

end Mark_time_spent_l1801_180182


namespace solve_equation_l1801_180100

theorem solve_equation (x y z : ℕ) :
  (∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 2) ↔ (x^2 + 3 * y^2 = 2^z) :=
by
  sorry

end solve_equation_l1801_180100


namespace count_squares_below_line_l1801_180114

theorem count_squares_below_line (units : ℕ) :
  let intercept_x := 221;
  let intercept_y := 7;
  let total_squares := intercept_x * intercept_y;
  let diagonal_squares := intercept_x - 1 + intercept_y - 1 + 1; 
  let non_diag_squares := total_squares - diagonal_squares;
  let below_line := non_diag_squares / 2;
  below_line = 660 :=
by
  sorry

end count_squares_below_line_l1801_180114


namespace arrangement_of_mississippi_no_adjacent_s_l1801_180130

-- Conditions: The word "MISSISSIPPI" has 11 letters with specific frequencies: 1 M, 4 I's, 4 S's, 2 P's.
-- No two S's can be adjacent.
def ways_to_arrange_mississippi_no_adjacent_s: Nat :=
  let total_non_s_arrangements := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)
  let gaps_for_s := Nat.choose 8 4
  total_non_s_arrangements * gaps_for_s

theorem arrangement_of_mississippi_no_adjacent_s : ways_to_arrange_mississippi_no_adjacent_s = 7350 :=
by
  unfold ways_to_arrange_mississippi_no_adjacent_s
  sorry

end arrangement_of_mississippi_no_adjacent_s_l1801_180130


namespace larger_fraction_of_two_l1801_180101

theorem larger_fraction_of_two (x y : ℚ) (h1 : x + y = 7/8) (h2 : x * y = 1/4) : max x y = 1/2 :=
sorry

end larger_fraction_of_two_l1801_180101


namespace rectangle_area_l1801_180195

theorem rectangle_area {A_s A_r : ℕ} (s l w : ℕ) (h1 : A_s = 36) (h2 : A_s = s * s)
  (h3 : w = s) (h4 : l = 3 * w) (h5 : A_r = w * l) : A_r = 108 :=
by
  sorry

end rectangle_area_l1801_180195


namespace g_difference_l1801_180197

variable (g : ℝ → ℝ)

-- Condition: g is a linear function
axiom linear_g : ∃ a b : ℝ, ∀ x : ℝ, g x = a * x + b

-- Condition: g(10) - g(4) = 18
axiom g_condition : g 10 - g 4 = 18

theorem g_difference : g 16 - g 4 = 36 :=
by
  sorry

end g_difference_l1801_180197


namespace oliver_spent_amount_l1801_180173

theorem oliver_spent_amount :
  ∀ (S : ℕ), (33 - S + 32 = 61) → S = 4 :=
by
  sorry

end oliver_spent_amount_l1801_180173


namespace value_of_fraction_of_power_l1801_180198

-- Define the values in the problem
def a : ℝ := 6
def b : ℝ := 30

-- The problem asks us to prove
theorem value_of_fraction_of_power : 
  (1 / 3) * (a ^ b) = 2 * (a ^ (b - 1)) :=
by
  -- Initial Setup
  let c := (1 / 3) * (a ^ b)
  let d := 2 * (a ^ (b - 1))
  -- The main claim
  show c = d
  sorry

end value_of_fraction_of_power_l1801_180198
