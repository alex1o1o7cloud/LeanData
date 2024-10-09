import Mathlib

namespace tan_product_l1961_196118

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l1961_196118


namespace find_first_number_in_sequence_l1961_196195

theorem find_first_number_in_sequence :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℚ),
    (a3 = a2 * a1) ∧ 
    (a4 = a3 * a2) ∧ 
    (a5 = a4 * a3) ∧ 
    (a6 = a5 * a4) ∧ 
    (a7 = a6 * a5) ∧ 
    (a8 = a7 * a6) ∧ 
    (a9 = a8 * a7) ∧ 
    (a10 = a9 * a8) ∧ 
    (a8 = 36) ∧ 
    (a9 = 324) ∧ 
    (a10 = 11664) ∧ 
    (a1 = 59049 / 65536) := 
sorry

end find_first_number_in_sequence_l1961_196195


namespace value_of_expression_l1961_196114

open Real

theorem value_of_expression {a : ℝ} (h : a^2 + 4 * a - 5 = 0) : 3 * a^2 + 12 * a = 15 :=
by sorry

end value_of_expression_l1961_196114


namespace quadratic_has_exactly_one_solution_l1961_196155

theorem quadratic_has_exactly_one_solution (k : ℚ) :
  (3 * x^2 - 8 * x + k = 0) → ((-8)^2 - 4 * 3 * k = 0) → k = 16 / 3 :=
by
  sorry

end quadratic_has_exactly_one_solution_l1961_196155


namespace correct_idiom_l1961_196190

-- Define the conditions given in the problem
def context := "The vast majority of office clerks read a significant amount of materials"
def idiom_usage := "to say _ of additional materials"

-- Define the proof problem
theorem correct_idiom (context: String) (idiom_usage: String) : idiom_usage.replace "_ of additional materials" "nothing of newspapers and magazines" = "to say nothing of newspapers and magazines" :=
sorry

end correct_idiom_l1961_196190


namespace sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l1961_196132

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) : (x + 1) * (x - 2) > 0 → x > 2 :=
by sorry

theorem converse_not_true 
  (x : ℝ) : x > 2 → (x + 1) * (x - 2) > 0 :=
by sorry

theorem cond_x_gt_2_iff_sufficient_not_necessary 
  (x : ℝ) : (x > 2 → (x + 1) * (x - 2) > 0) ∧ 
            ((x + 1) * (x - 2) > 0 → x > 2) :=
by sorry

end sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l1961_196132


namespace percentage_return_on_investment_l1961_196141

theorem percentage_return_on_investment (dividend_rate : ℝ) (face_value : ℝ) (purchase_price : ℝ) (return_percentage : ℝ) :
  dividend_rate = 0.125 → face_value = 40 → purchase_price = 20 → return_percentage = 25 :=
by
  intros h1 h2 h3
  sorry

end percentage_return_on_investment_l1961_196141


namespace meters_examined_l1961_196192

theorem meters_examined (x : ℝ) (h1 : 0.07 / 100 * x = 2) : x = 2857 :=
by
  -- using the given setup and simplification
  sorry

end meters_examined_l1961_196192


namespace arithmetic_sequence_condition_l1961_196125

theorem arithmetic_sequence_condition (a : ℕ → ℝ) :
  (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2)) ↔
  (∀ n ∈ {k : ℕ | k > 0}, a (n+1) - a n = a (n+2) - a (n+1)) ∧ ¬ (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2) → a (n+1) = a n) :=
sorry

end arithmetic_sequence_condition_l1961_196125


namespace number_of_math_students_l1961_196127

-- Definitions for the problem conditions
variables (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
variable (total_students_eq : total_students = 100)
variable (both_classes_eq : both_classes = 10)
variable (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))

-- Theorem statement
theorem number_of_math_students (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
  (total_students_eq : total_students = 100)
  (both_classes_eq : both_classes = 10)
  (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))
  (total_students_eq : total_students = physics_class + math_class - both_classes) :
  math_class = 88 :=
sorry

end number_of_math_students_l1961_196127


namespace avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l1961_196117

variable (c d : ℤ)
variable (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7 :
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7) / 7 = c + 7 :=
by
  sorry

end avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l1961_196117


namespace average_rainfall_in_normal_year_l1961_196128

def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def rainfall_difference : ℕ := 58

theorem average_rainfall_in_normal_year :
  (total_rainfall_this_year + rainfall_difference) = 140 :=
by
  sorry

end average_rainfall_in_normal_year_l1961_196128


namespace rationalize_sqrt_5_div_18_l1961_196168

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l1961_196168


namespace claire_earning_l1961_196151

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end claire_earning_l1961_196151


namespace pyramid_volume_l1961_196184

noncomputable def volume_of_pyramid 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2)
  (isosceles_pyramid : Prop) : ℝ :=
  sorry

theorem pyramid_volume 
  (EFGH_rect : ℝ × ℝ) 
  (EF_len : EFGH_rect.1 = 15 * Real.sqrt 2) 
  (FG_len : EFGH_rect.2 = 14 * Real.sqrt 2) 
  (isosceles_pyramid : Prop) : 
  volume_of_pyramid EFGH_rect EF_len FG_len isosceles_pyramid = 735 := 
sorry

end pyramid_volume_l1961_196184


namespace diagonal_cells_crossed_l1961_196129

theorem diagonal_cells_crossed (m n : ℕ) (h_m : m = 199) (h_n : n = 991) :
  (m + n - Nat.gcd m n) = 1189 := by
  sorry

end diagonal_cells_crossed_l1961_196129


namespace circle_standard_equation_l1961_196126

theorem circle_standard_equation:
  ∃ (x y : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

end circle_standard_equation_l1961_196126


namespace correct_statement_a_l1961_196177

theorem correct_statement_a (x y : ℝ) (h : x + y < 0) : x^2 - y > x :=
sorry

end correct_statement_a_l1961_196177


namespace initial_pens_count_l1961_196181

theorem initial_pens_count (P : ℕ) (h : 2 * (P + 22) - 19 = 75) : P = 25 :=
by
  sorry

end initial_pens_count_l1961_196181


namespace population_increase_duration_l1961_196122

noncomputable def birth_rate := 6 / 2 -- people every 2 seconds = 3 people per second
noncomputable def death_rate := 2 / 2 -- people every 2 seconds = 1 person per second
noncomputable def net_increase_per_second := (birth_rate - death_rate) -- net increase per second

def total_net_increase := 172800

theorem population_increase_duration :
  (total_net_increase / net_increase_per_second) / 3600 = 24 :=
by
  sorry

end population_increase_duration_l1961_196122


namespace correct_sum_104th_parenthesis_l1961_196179

noncomputable def sum_104th_parenthesis : ℕ := sorry

theorem correct_sum_104th_parenthesis :
  sum_104th_parenthesis = 2072 := 
by 
  sorry

end correct_sum_104th_parenthesis_l1961_196179


namespace cost_of_kid_ticket_l1961_196110

theorem cost_of_kid_ticket (total_people kids adults : ℕ) 
  (adult_ticket_cost kid_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (h_people : total_people = kids + adults)
  (h_adult_cost : adult_ticket_cost = 28)
  (h_kids : kids = 203)
  (h_total_sales : total_sales = 3864)
  (h_calculate_sales : adults * adult_ticket_cost + kids * kid_ticket_cost = total_sales)
  : kid_ticket_cost = 12 :=
by
  sorry -- Proof will be filled in

end cost_of_kid_ticket_l1961_196110


namespace storage_house_blocks_needed_l1961_196108

noncomputable def volume_of_storage_house
  (L_o : ℕ) (W_o : ℕ) (H_o : ℕ) (T : ℕ) : ℕ :=
  let interior_length := L_o - 2 * T
  let interior_width := W_o - 2 * T
  let interior_height := H_o - T
  let outer_volume := L_o * W_o * H_o
  let interior_volume := interior_length * interior_width * interior_height
  outer_volume - interior_volume

theorem storage_house_blocks_needed :
  volume_of_storage_house 15 12 8 2 = 912 :=
  by
    sorry

end storage_house_blocks_needed_l1961_196108


namespace amount_after_two_years_l1961_196121

def present_value : ℝ := 70400
def rate : ℝ := 0.125
def years : ℕ := 2
def final_amount := present_value * (1 + rate) ^ years

theorem amount_after_two_years : final_amount = 89070 :=
by sorry

end amount_after_two_years_l1961_196121


namespace part_I_part_II_l1961_196158

section problem_1

def f (x : ℝ) (a : ℝ) := |x - 3| - |x + a|

theorem part_I (x : ℝ) (hx : f x 2 < 1) : 0 < x :=
by
  sorry

theorem part_II (a : ℝ) (h : ∀ (x : ℝ), f x a ≤ 2 * a) : 3 ≤ a :=
by
  sorry

end problem_1

end part_I_part_II_l1961_196158


namespace factorize_expression_l1961_196156

theorem factorize_expression (a b x y : ℝ) : 
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = ab * (x - y)^2 * (a * x - a * y - b) :=
by
  sorry

end factorize_expression_l1961_196156


namespace angle_ABC_is_83_l1961_196142

-- Definitions of angles and the quadrilateral
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angleBAC angleCAD angleACD : ℝ)
variables (AB AD AC : ℝ)

-- Conditions as hypotheses
axiom h1 : angleBAC = 60
axiom h2 : angleCAD = 60
axiom h3 : AB + AD = AC
axiom h4 : angleACD = 23

-- The theorem to prove
theorem angle_ABC_is_83 (h1 : angleBAC = 60) (h2 : angleCAD = 60) (h3 : AB + AD = AC) (h4 : angleACD = 23) : 
  ∃ angleABC : ℝ, angleABC = 83 :=
sorry

end angle_ABC_is_83_l1961_196142


namespace least_possible_value_f_1998_l1961_196144

theorem least_possible_value_f_1998 
  (f : ℕ → ℕ)
  (h : ∀ m n, f (n^2 * f m) = m * (f n)^2) : 
  f 1998 = 120 :=
sorry

end least_possible_value_f_1998_l1961_196144


namespace annual_income_before_tax_l1961_196193

variable (I : ℝ) -- Define I as the annual income before tax

-- Conditions
def original_tax (I : ℝ) : ℝ := 0.42 * I
def new_tax (I : ℝ) : ℝ := 0.32 * I
def differential_savings (I : ℝ) : ℝ := original_tax I - new_tax I

-- Theorem: Given the conditions, the taxpayer's annual income before tax is $42,400
theorem annual_income_before_tax : differential_savings I = 4240 → I = 42400 := by
  sorry

end annual_income_before_tax_l1961_196193


namespace beth_overall_score_l1961_196130

-- Definitions for conditions
def percent_score (score_pct : ℕ) (total_problems : ℕ) : ℕ :=
  (score_pct * total_problems) / 100

def total_correct_answers : ℕ :=
  percent_score 60 15 + percent_score 85 20 + percent_score 75 25

def total_problems : ℕ := 15 + 20 + 25

def combined_percentage : ℕ :=
  (total_correct_answers * 100) / total_problems

-- The statement to be proved
theorem beth_overall_score : combined_percentage = 75 := by
  sorry

end beth_overall_score_l1961_196130


namespace reduced_price_correct_l1961_196103

theorem reduced_price_correct (P R Q: ℝ) (h1 : R = 0.75 * P) (h2 : 900 = Q * P) (h3 : 900 = (Q + 5) * R)  :
  R = 45 := by 
  sorry

end reduced_price_correct_l1961_196103


namespace lines_parallel_l1961_196159

theorem lines_parallel (m : ℝ) : 
  (m = 2 ↔ ∀ x y : ℝ, (2 * x - m * y - 1 = 0) ∧ ((m - 1) * x - y + 1 = 0) → 
  (∃ k : ℝ, (2 * x - m * y - 1 = k * ((m - 1) * x - y + 1)))) :=
by sorry

end lines_parallel_l1961_196159


namespace cube_strictly_increasing_l1961_196163

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l1961_196163


namespace p_scale_measurement_l1961_196167

theorem p_scale_measurement (a b P S : ℝ) (h1 : 30 = 6 * a + b) (h2 : 60 = 24 * a + b) (h3 : 100 = a * P + b) : P = 48 :=
by
  sorry

end p_scale_measurement_l1961_196167


namespace number_of_ordered_pairs_l1961_196157

noncomputable def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k

theorem number_of_ordered_pairs :
  (∃ (n : ℕ), n = 29 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ x ≤ 2020 ∧ y ≤ 2020 →
    is_power_of_prime (3 * x^2 + 10 * x * y + 3 * y^2) → n = 29) :=
by
  sorry

end number_of_ordered_pairs_l1961_196157


namespace compare_expr_l1961_196136

theorem compare_expr (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) :=
sorry

end compare_expr_l1961_196136


namespace sandy_net_amount_spent_l1961_196165

def amount_spent_shorts : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def amount_received_return : ℝ := 7.43

theorem sandy_net_amount_spent :
  amount_spent_shorts + amount_spent_shirt - amount_received_return = 18.70 :=
by
  sorry

end sandy_net_amount_spent_l1961_196165


namespace probability_is_1_over_90_l1961_196196

/-- Probability Calculation -/
noncomputable def probability_of_COLD :=
  (1 / (Nat.choose 5 3)) * (2 / 3) * (1 / (Nat.choose 4 2))

theorem probability_is_1_over_90 :
  probability_of_COLD = (1 / 90) :=
by
  sorry

end probability_is_1_over_90_l1961_196196


namespace smaller_circle_radius_l1961_196134

theorem smaller_circle_radius
  (radius_largest : ℝ)
  (h1 : radius_largest = 10)
  (aligned_circles : ℝ)
  (h2 : 4 * aligned_circles = 2 * radius_largest) :
  aligned_circles / 2 = 2.5 :=
by
  sorry

end smaller_circle_radius_l1961_196134


namespace reema_simple_interest_l1961_196109

-- Definitions and conditions
def principal : ℕ := 1200
def rate_of_interest : ℕ := 6
def time_period : ℕ := rate_of_interest

-- Simple interest calculation
def calculate_simple_interest (P R T: ℕ) : ℕ :=
  (P * R * T) / 100

-- The theorem to prove that Reema paid Rs 432 as simple interest.
theorem reema_simple_interest : calculate_simple_interest principal rate_of_interest time_period = 432 := 
  sorry

end reema_simple_interest_l1961_196109


namespace domain_of_function_l1961_196147

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x * (x - 1) ≥ 0) ↔ (x = 0 ∨ x ≥ 1) :=
by sorry

end domain_of_function_l1961_196147


namespace volume_of_cone_l1961_196186

theorem volume_of_cone (d h : ℝ) (d_eq : d = 12) (h_eq : h = 9) : 
  (1 / 3) * π * (d / 2)^2 * h = 108 * π := 
by 
  rw [d_eq, h_eq] 
  sorry

end volume_of_cone_l1961_196186


namespace frood_game_least_n_l1961_196172

theorem frood_game_least_n (n : ℕ) (h : n > 0) (drop_score : ℕ := n * (n + 1) / 2) (eat_score : ℕ := 15 * n) 
  : drop_score > eat_score ↔ n ≥ 30 :=
by
  sorry

end frood_game_least_n_l1961_196172


namespace fg_at_3_l1961_196113

def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := x^2 + 5

theorem fg_at_3 : f (g 3) = 10 := by
  sorry

end fg_at_3_l1961_196113


namespace worker_efficiency_l1961_196146

theorem worker_efficiency (Wq : ℝ) (x : ℝ) : 
  (1.4 * (1 / x) = 1 / (1.4 * x)) → 
  (14 * (1 / x + 1 / (1.4 * x)) = 1) → 
  x = 24 :=
by
  sorry

end worker_efficiency_l1961_196146


namespace product_mod_7_zero_l1961_196198

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l1961_196198


namespace PersonX_job_completed_time_l1961_196162

-- Definitions for conditions
def Dan_job_time := 15 -- hours
def PersonX_job_time (x : ℝ) := x -- hours
def Dan_work_time := 3 -- hours
def PersonX_remaining_work_time := 8 -- hours

-- Given Dan's and Person X's work time, prove Person X's job completion time
theorem PersonX_job_completed_time (x : ℝ) (h1 : Dan_job_time > 0)
    (h2 : PersonX_job_time x > 0)
    (h3 : Dan_work_time > 0)
    (h4 : PersonX_remaining_work_time * (1 - Dan_work_time / Dan_job_time) = 1 / x * 8) :
    x = 10 :=
  sorry

end PersonX_job_completed_time_l1961_196162


namespace distinct_positive_roots_log_sum_eq_5_l1961_196104

theorem distinct_positive_roots_log_sum_eq_5 (a b : ℝ)
  (h : ∀ (x : ℝ), (8 * x ^ 3 + 6 * a * x ^ 2 + 3 * b * x + a = 0) → x > 0) 
  (h_sum : ∀ u v w : ℝ, (8 * u ^ 3 + 6 * a * u ^ 2 + 3 * b * u + a = 0) ∧
                       (8 * v ^ 3 + 6 * a * v ^ 2 + 3 * b * v + a = 0) ∧
                       (8 * w ^ 3 + 6 * a * w ^ 2 + 3 * b * w + a = 0) → 
                       u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
                       (Real.log (u) / Real.log (3) + Real.log (v) / Real.log (3) + Real.log (w) / Real.log (3) = 5)) :
  a = -1944 :=
sorry

end distinct_positive_roots_log_sum_eq_5_l1961_196104


namespace number_of_special_divisors_l1961_196154

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end number_of_special_divisors_l1961_196154


namespace Jungkook_fewest_erasers_l1961_196106

-- Define the number of erasers each person has.
def Jungkook_erasers : ℕ := 6
def Jimin_erasers : ℕ := Jungkook_erasers + 4
def Seokjin_erasers : ℕ := Jimin_erasers - 3

-- Prove that Jungkook has the fewest erasers.
theorem Jungkook_fewest_erasers : Jungkook_erasers < Jimin_erasers ∧ Jungkook_erasers < Seokjin_erasers :=
by
  -- Proof goes here
  sorry

end Jungkook_fewest_erasers_l1961_196106


namespace jason_total_spending_l1961_196143

def cost_of_shorts : ℝ := 14.28
def cost_of_jacket : ℝ := 4.74
def total_spent : ℝ := 19.02

theorem jason_total_spending : cost_of_shorts + cost_of_jacket = total_spent :=
by
  sorry

end jason_total_spending_l1961_196143


namespace sum_at_simple_interest_l1961_196120

theorem sum_at_simple_interest (P R : ℝ) (h1: ((3 * P * (R + 1))/ 100) = ((3 * P * R) / 100 + 72)) : P = 2400 := 
by 
  sorry

end sum_at_simple_interest_l1961_196120


namespace tangent_line_through_P_l1961_196140

theorem tangent_line_through_P (x y : ℝ) :
  (∃ l : ℝ, l = 3*x - 4*y + 5) ∨ (x = 1) :=
by
  sorry

end tangent_line_through_P_l1961_196140


namespace find_n_l1961_196182

theorem find_n (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (h2 : b n = 0)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 :=
sorry

end find_n_l1961_196182


namespace pipe_fill_time_l1961_196170

theorem pipe_fill_time (T : ℝ) 
  (h1 : ∃ T : ℝ, 0 < T) 
  (h2 : T + (1/2) > 0) 
  (h3 : ∃ leak_rate : ℝ, leak_rate = 1/10) 
  (h4 : ∃ pipe_rate : ℝ, pipe_rate = 1/T) 
  (h5 : ∃ effective_rate : ℝ, effective_rate = pipe_rate - leak_rate) 
  (h6 : effective_rate = 1 / (T + 1/2))  : 
  T = Real.sqrt 5 :=
  sorry

end pipe_fill_time_l1961_196170


namespace yulia_max_candies_l1961_196176

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies_l1961_196176


namespace quadratic_conversion_l1961_196164

theorem quadratic_conversion (x : ℝ) :
  (2*x - 1)^2 = (x + 1)*(3*x + 4) →
  ∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a*x^2 + b*x + c = 0 :=
by simp [pow_two, mul_add, add_mul, mul_comm]; sorry

end quadratic_conversion_l1961_196164


namespace custom_op_example_l1961_196197

def custom_op (a b : ℕ) : ℕ := (a + 1) / b

theorem custom_op_example : custom_op 2 (custom_op 3 4) = 3 := 
by
  sorry

end custom_op_example_l1961_196197


namespace polygon_diagonals_l1961_196161

theorem polygon_diagonals (n : ℕ) (h : n - 3 ≤ 6) : n = 9 :=
by sorry

end polygon_diagonals_l1961_196161


namespace unit_vector_perpendicular_to_a_l1961_196178

-- Definitions of a vector and the properties of unit and perpendicular vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def is_unit_vector (v : Vector2D) : Prop :=
  v.x ^ 2 + v.y ^ 2 = 1

def is_perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

-- Given vector a
def a : Vector2D := ⟨3, 4⟩

-- Coordinates of the unit vector that is perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ (b : Vector2D), is_unit_vector b ∧ is_perpendicular a b ∧
  (b = ⟨-4 / 5, 3 / 5⟩ ∨ b = ⟨4 / 5, -3 / 5⟩) :=
sorry

end unit_vector_perpendicular_to_a_l1961_196178


namespace smallest_integer_in_set_of_seven_l1961_196169

theorem smallest_integer_in_set_of_seven (n : ℤ) (h : n + 6 < 3 * (n + 3)) : n = -1 :=
sorry

end smallest_integer_in_set_of_seven_l1961_196169


namespace conjecture_l1961_196111

noncomputable def f (x : ℝ) : ℝ :=
  1 / (3^x + Real.sqrt 3)

theorem conjecture (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 3 / 3 := sorry

end conjecture_l1961_196111


namespace number_of_8_tuples_l1961_196180

-- Define the constraints for a_k
def valid_a (a : ℕ) (k : ℕ) : Prop := 0 ≤ a ∧ a ≤ k

-- Define the condition for the 8-tuple
def valid_8_tuple (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ) : Prop :=
  valid_a a1 1 ∧ valid_a a2 2 ∧ valid_a a3 3 ∧ valid_a a4 4 ∧ 
  (a1 + a2 + a3 + a4 + 2 * b1 + 3 * b2 + 4 * b3 + 5 * b4 = 19)

theorem number_of_8_tuples : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ (a1 a2 a3 a4 b1 b2 b3 b4 : ℕ), valid_8_tuple a1 a2 a3 a4 b1 b2 b3 b4 := 
sorry

end number_of_8_tuples_l1961_196180


namespace toothpicks_in_arithmetic_sequence_l1961_196189

theorem toothpicks_in_arithmetic_sequence :
  let a1 := 5
  let d := 3
  let n := 15
  let a_n n := a1 + (n - 1) * d
  let sum_to_n n := n * (2 * a1 + (n - 1) * d) / 2
  sum_to_n n = 390 := by
  sorry

end toothpicks_in_arithmetic_sequence_l1961_196189


namespace right_triangle_third_side_l1961_196137

theorem right_triangle_third_side (a b : ℕ) (c : ℝ) (h₁: a = 3) (h₂: b = 4) (h₃: ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2)) ∨ (c^2 + b^2 = a^2)):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end right_triangle_third_side_l1961_196137


namespace solution_set_of_inequality_l1961_196105

theorem solution_set_of_inequality (a : ℝ) (h : 0 < a) :
  {x : ℝ | x ^ 2 - 4 * a * x - 5 * a ^ 2 < 0} = {x : ℝ | -a < x ∧ x < 5 * a} :=
sorry

end solution_set_of_inequality_l1961_196105


namespace value_of_x_l1961_196112

theorem value_of_x (x : ℤ) : (x + 1) * (x + 1) = 16 ↔ (x = 3 ∨ x = -5) := 
by sorry

end value_of_x_l1961_196112


namespace sum_of_cubes_three_consecutive_divisible_by_three_l1961_196135

theorem sum_of_cubes_three_consecutive_divisible_by_three (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 3 = 0 := 
by 
  sorry

end sum_of_cubes_three_consecutive_divisible_by_three_l1961_196135


namespace geometric_sequence_a9_value_l1961_196160

theorem geometric_sequence_a9_value {a : ℕ → ℝ} (q a1 : ℝ) 
  (h_geom : ∀ n, a n = a1 * q ^ n)
  (h_a3 : a 3 = 2)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (h_sum : S 12 = 4 * S 6) : a 9 = 2 := 
by 
  sorry

end geometric_sequence_a9_value_l1961_196160


namespace spadesuit_problem_l1961_196173

def spadesuit (x y : ℝ) := (x + y) * (x - y)

theorem spadesuit_problem : spadesuit 5 (spadesuit 3 2) = 0 := by
  sorry

end spadesuit_problem_l1961_196173


namespace geometric_sequence_common_ratio_l1961_196145

theorem geometric_sequence_common_ratio
  (q : ℝ) (a_n : ℕ → ℝ)
  (h_inc : ∀ n, a_n (n + 1) = q * a_n n ∧ q > 1)
  (h_a2 : a_n 2 = 2)
  (h_a4_a3 : a_n 4 - a_n 3 = 4) : 
  q = 2 :=
sorry

end geometric_sequence_common_ratio_l1961_196145


namespace lawn_unmowed_fraction_l1961_196166

noncomputable def rate_mary : ℚ := 1 / 6
noncomputable def rate_tom : ℚ := 1 / 3

theorem lawn_unmowed_fraction :
  (1 : ℚ) - ((1 * rate_tom) + (2 * (rate_mary + rate_tom))) = 1 / 6 :=
by
  -- This part will be the actual proof which we are skipping
  sorry

end lawn_unmowed_fraction_l1961_196166


namespace smaller_number_l1961_196149

theorem smaller_number (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end smaller_number_l1961_196149


namespace conversion_problems_l1961_196133

def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 2 + 10 * decimal_to_binary (n / 2)

def largest_two_digit_octal : ℕ := 77

theorem conversion_problems :
  decimal_to_binary 111 = 1101111 ∧ (7 * 8 + 7) = 63 :=
by
  sorry

end conversion_problems_l1961_196133


namespace number_of_schools_l1961_196185

-- Define the conditions as parameters and assumptions
structure CityContest (n : ℕ) :=
  (students_per_school : ℕ := 4)
  (total_students : ℕ := students_per_school * n)
  (andrea_percentile : ℕ := 75)
  (andrea_highest_team : Prop)
  (beth_rank : ℕ := 20)
  (carla_rank : ℕ := 47)
  (david_rank : ℕ := 78)
  (andrea_position : ℕ)
  (h3 : andrea_position = (3 * total_students + 1) / 4)
  (h4 : 3 * n > 78)

-- Define the main theorem statement
theorem number_of_schools (n : ℕ) (contest : CityContest n) (h5 : contest.andrea_highest_team) : n = 20 :=
  by {
    -- You would insert the detailed proof of the theorem based on the conditions here.
    sorry
  }

end number_of_schools_l1961_196185


namespace pencils_per_person_l1961_196152

theorem pencils_per_person (x : ℕ) (h : 3 * x = 24) : x = 8 :=
by
  -- sorry we are skipping the actual proof
  sorry

end pencils_per_person_l1961_196152


namespace f_at_3_l1961_196100

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_of_2 : f 2 = 1
axiom f_rec (x : ℝ) : f (x + 2) = f x + f 2

theorem f_at_3 : f 3 = 3 / 2 := 
by 
  sorry

end f_at_3_l1961_196100


namespace Justin_run_home_time_l1961_196115

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l1961_196115


namespace geometric_sequence_increasing_condition_l1961_196194

theorem geometric_sequence_increasing_condition (a₁ a₂ a₄ : ℝ) (q : ℝ) (n : ℕ) (a : ℕ → ℝ):
  (∀ n, a n = a₁ * q^n) →
  (a₁ < a₂ ∧ a₂ < a₄) → 
  ¬ (∀ n, a n < a (n + 1)) → 
  (a₁ < a₂ ∧ a₂ < a₄) ∧ ¬ (∀ n, a n < a (n + 1)) :=
sorry

end geometric_sequence_increasing_condition_l1961_196194


namespace paint_needed_270_statues_l1961_196174

theorem paint_needed_270_statues:
  let height_large := 12
  let paint_large := 2
  let height_small := 3
  let num_statues := 270
  let ratio_height := (height_small : ℝ) / (height_large : ℝ)
  let ratio_area := ratio_height ^ 2
  let paint_small := paint_large * ratio_area
  let total_paint := num_statues * paint_small
  total_paint = 33.75 := by
  sorry

end paint_needed_270_statues_l1961_196174


namespace count_3_digit_integers_with_product_36_l1961_196175

theorem count_3_digit_integers_with_product_36 : 
  ∃ n, n = 21 ∧ 
         (∀ d1 d2 d3 : ℕ, 
           1 ≤ d1 ∧ d1 ≤ 9 ∧ 
           1 ≤ d2 ∧ d2 ≤ 9 ∧ 
           1 ≤ d3 ∧ d3 ≤ 9 ∧
           d1 * d2 * d3 = 36 → 
           (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0)) := sorry

end count_3_digit_integers_with_product_36_l1961_196175


namespace liar_and_truth_tellers_l1961_196153

-- Define the characters and their nature (truth-teller or liar)
inductive Character : Type
| Kikimora
| Leshy
| Vodyanoy

def always_truthful (c : Character) : Prop := sorry
def always_lying (c : Character) : Prop := sorry

axiom kikimora_statement : always_lying Character.Kikimora
axiom leshy_statement : ∃ l₁ l₂ : Character, l₁ ≠ l₂ ∧ always_lying l₁ ∧ always_lying l₂
axiom vodyanoy_statement : true -- Vodyanoy's silence

-- Proof that Kikimora and Vodyanoy are liars and Leshy is truthful
theorem liar_and_truth_tellers :
  always_lying Character.Kikimora ∧
  always_lying Character.Vodyanoy ∧
  always_truthful Character.Leshy := sorry

end liar_and_truth_tellers_l1961_196153


namespace compare_abc_l1961_196131

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l1961_196131


namespace find_simple_interest_rate_l1961_196119

variable (P : ℝ) (n : ℕ) (r_c : ℝ) (t : ℝ) (I_c : ℝ) (I_s : ℝ) (r_s : ℝ)

noncomputable def compound_interest_amount (P r_c : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r_c / n) ^ (n * t)

noncomputable def simple_interest_amount (P r_s : ℝ) (t : ℝ) : ℝ :=
  P * r_s * t

theorem find_simple_interest_rate
  (hP : P = 5000)
  (hr_c : r_c = 0.16)
  (hn : n = 2)
  (ht : t = 1)
  (hI_c : I_c = compound_interest_amount P r_c n t - P)
  (hI_s : I_s = I_c - 16)
  (hI_s_def : I_s = simple_interest_amount P r_s t) :
  r_s = 0.1632 := sorry

end find_simple_interest_rate_l1961_196119


namespace real_solutions_x4_plus_3_minus_x4_eq_82_l1961_196150

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end real_solutions_x4_plus_3_minus_x4_eq_82_l1961_196150


namespace mike_travel_distance_l1961_196138

theorem mike_travel_distance
  (mike_start : ℝ := 2.50)
  (mike_per_mile : ℝ := 0.25)
  (annie_start : ℝ := 2.50)
  (annie_toll : ℝ := 5.00)
  (annie_per_mile : ℝ := 0.25)
  (annie_miles : ℝ := 14)
  (mike_cost : ℝ)
  (annie_cost : ℝ) :
  mike_cost = annie_cost → mike_cost = mike_start + mike_per_mile * 34 := by
  sorry

end mike_travel_distance_l1961_196138


namespace game_ends_in_36_rounds_l1961_196199

theorem game_ends_in_36_rounds 
    (tokens_A : ℕ := 17) (tokens_B : ℕ := 16) (tokens_C : ℕ := 15)
    (rounds : ℕ) 
    (game_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop) 
    (extra_discard_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop)  
    (game_ends_when_token_zero : (tokens_A tokens_B tokens_C : ℕ) → Prop) :
    game_rule tokens_A tokens_B tokens_C rounds ∧
    extra_discard_rule tokens_A tokens_B tokens_C rounds ∧
    game_ends_when_token_zero tokens_A tokens_B tokens_C → 
    rounds = 36 := by
    sorry

end game_ends_in_36_rounds_l1961_196199


namespace find_larger_number_l1961_196148

theorem find_larger_number
  (x y : ℝ)
  (h1 : y = 2 * x + 3)
  (h2 : x + y = 27)
  : y = 19 :=
by
  sorry

end find_larger_number_l1961_196148


namespace problem_part1_problem_part2_l1961_196107

theorem problem_part1 :
  ∀ m : ℝ, (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
sorry

theorem problem_part2 :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧
    (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
sorry

end problem_part1_problem_part2_l1961_196107


namespace tan_eleven_pi_over_three_l1961_196188

theorem tan_eleven_pi_over_three : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := 
    sorry

end tan_eleven_pi_over_three_l1961_196188


namespace mike_total_investment_l1961_196101

variable (T : ℝ)
variable (H1 : 0.09 * 1800 + 0.11 * (T - 1800) = 624)

theorem mike_total_investment : T = 6000 :=
by
  sorry

end mike_total_investment_l1961_196101


namespace ratio_of_men_to_women_l1961_196191

/-- Define the number of men and women on a co-ed softball team. -/
def number_of_men : ℕ := 8
def number_of_women : ℕ := 12

/--
  Given:
  1. There are 4 more women than men.
  2. The total number of players is 20.
  Prove that the ratio of men to women is 2 : 3.
-/
theorem ratio_of_men_to_women 
  (h1 : number_of_women = number_of_men + 4)
  (h2 : number_of_men + number_of_women = 20) :
  (number_of_men * 3) = (number_of_women * 2) :=
by
  have h3 : number_of_men = 8 := by sorry
  have h4 : number_of_women = 12 := by sorry
  sorry

end ratio_of_men_to_women_l1961_196191


namespace randy_initial_blocks_l1961_196187

theorem randy_initial_blocks (used_blocks left_blocks total_blocks : ℕ) (h1 : used_blocks = 19) (h2 : left_blocks = 59) : total_blocks = used_blocks + left_blocks → total_blocks = 78 :=
by 
  intros
  sorry

end randy_initial_blocks_l1961_196187


namespace two_x_plus_y_eq_12_l1961_196139

-- Variables representing the prime numbers x and y
variables {x y : ℕ}

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Prime n
def lcm_eq (a b c : ℕ) : Prop := Nat.lcm a b = c

-- The theorem statement
theorem two_x_plus_y_eq_12 (h1 : lcm_eq x y 10) (h2 : is_prime x) (h3 : is_prime y) (h4 : x > y) :
    2 * x + y = 12 :=
sorry

end two_x_plus_y_eq_12_l1961_196139


namespace computer_price_in_2016_l1961_196102

def price (p₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ := p₀ * (r ^ (n / 4))

theorem computer_price_in_2016 :
  price 8100 (2/3 : ℚ) 16 = 1600 :=
by
  sorry

end computer_price_in_2016_l1961_196102


namespace pump_fill_time_without_leak_l1961_196183

theorem pump_fill_time_without_leak
    (P : ℝ)
    (h1 : 2 + 1/7 = (15:ℝ)/7)
    (h2 : 1 / P - 1 / 30 = 7 / 15) :
  P = 2 := by
  sorry

end pump_fill_time_without_leak_l1961_196183


namespace sum_of_four_numbers_eq_zero_l1961_196171

theorem sum_of_four_numbers_eq_zero
  (x y s t : ℝ)
  (h₀ : x ≠ y)
  (h₁ : x ≠ s)
  (h₂ : x ≠ t)
  (h₃ : y ≠ s)
  (h₄ : y ≠ t)
  (h₅ : s ≠ t)
  (h_eq : (x + s) / (x + t) = (y + t) / (y + s)) :
  x + y + s + t = 0 := by
sorry

end sum_of_four_numbers_eq_zero_l1961_196171


namespace original_bet_l1961_196116

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l1961_196116


namespace vector_addition_dot_product_l1961_196123

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vector_addition :
  let c := (1, 2) + (3, 1)
  c = (4, 3) := by
  sorry

theorem dot_product :
  let d := (1 * 3 + 2 * 1)
  d = 5 := by
  sorry

end vector_addition_dot_product_l1961_196123


namespace emir_needs_more_money_l1961_196124

noncomputable def dictionary_cost : ℝ := 5.50
noncomputable def dinosaur_book_cost : ℝ := 11.25
noncomputable def childrens_cookbook_cost : ℝ := 5.75
noncomputable def science_experiment_kit_cost : ℝ := 8.50
noncomputable def colored_pencils_cost : ℝ := 3.60
noncomputable def world_map_poster_cost : ℝ := 2.40
noncomputable def puzzle_book_cost : ℝ := 4.65
noncomputable def sketchpad_cost : ℝ := 6.20

noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def dinosaur_discount_rate : ℝ := 0.10
noncomputable def saved_amount : ℝ := 28.30

noncomputable def total_cost_before_tax : ℝ :=
  dictionary_cost +
  (dinosaur_book_cost - dinosaur_discount_rate * dinosaur_book_cost) +
  childrens_cookbook_cost +
  science_experiment_kit_cost +
  colored_pencils_cost +
  world_map_poster_cost +
  puzzle_book_cost +
  sketchpad_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + total_sales_tax

noncomputable def additional_amount_needed : ℝ := total_cost_after_tax - saved_amount

theorem emir_needs_more_money : additional_amount_needed = 21.81 := by
  sorry

end emir_needs_more_money_l1961_196124
