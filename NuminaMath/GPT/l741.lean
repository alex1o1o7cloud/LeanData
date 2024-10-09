import Mathlib

namespace div_by_eight_l741_74190

theorem div_by_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 :=
by
  sorry

end div_by_eight_l741_74190


namespace income_percent_greater_l741_74162

variable (A B : ℝ)

-- Condition: A's income is 25% less than B's income
def income_condition (A B : ℝ) : Prop :=
  A = 0.75 * B

-- Statement: B's income is 33.33% greater than A's income
theorem income_percent_greater (A B : ℝ) (h : income_condition A B) :
  B = A * (4 / 3) := by
sorry

end income_percent_greater_l741_74162


namespace simplify_expr1_simplify_expr2_l741_74173

-- Define the first problem with necessary conditions
theorem simplify_expr1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (b - a)) = (a + b) / (a - b) :=
by
  sorry

-- Define the second problem with necessary conditions
theorem simplify_expr2 (x : ℝ) (hx1 : x ≠ -3) (hx2 : x ≠ 4) (hx3 : x ≠ -4) :
  ((x - 4) / (x + 3)) / (x - 3 - (7 / (x + 3))) = 1 / (x + 4) :=
by
  sorry

end simplify_expr1_simplify_expr2_l741_74173


namespace cindy_marbles_problem_l741_74146

theorem cindy_marbles_problem
  (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 500) (h2 : friends = 4) (h3 : marbles_per_friend = 80) :
  4 * (initial_marbles - (marbles_per_friend * friends)) = 720 :=
by
  sorry

end cindy_marbles_problem_l741_74146


namespace roots_twice_other_p_values_l741_74102

theorem roots_twice_other_p_values (p : ℝ) :
  (∃ (a : ℝ), (a^2 = 9) ∧ (x^2 + p*x + 18 = 0) ∧
  ((x - a)*(x - 2*a) = (0:ℝ))) ↔ (p = 9 ∨ p = -9) :=
sorry

end roots_twice_other_p_values_l741_74102


namespace jakes_class_boys_count_l741_74167

theorem jakes_class_boys_count 
    (ratio_girls_boys : ℕ → ℕ → Prop)
    (students_total : ℕ)
    (ratio_condition : ratio_girls_boys 3 4)
    (total_condition : students_total = 35) :
    ∃ boys : ℕ, boys = 20 :=
by
  sorry

end jakes_class_boys_count_l741_74167


namespace determinant_nonnegative_of_skew_symmetric_matrix_l741_74182

theorem determinant_nonnegative_of_skew_symmetric_matrix
  (a b c d e f : ℝ)
  (A : Matrix (Fin 4) (Fin 4) ℝ)
  (hA : A = ![
    ![0, a, b, c],
    ![-a, 0, d, e],
    ![-b, -d, 0, f],
    ![-c, -e, -f, 0]]) :
  0 ≤ Matrix.det A := by
  sorry

end determinant_nonnegative_of_skew_symmetric_matrix_l741_74182


namespace Pablo_is_70_cm_taller_than_Charlene_l741_74145

variable (Ruby Pablo Charlene Janet : ℕ)

-- Conditions
axiom h1 : Ruby + 2 = Pablo
axiom h2 : Charlene = 2 * Janet
axiom h3 : Janet = 62
axiom h4 : Ruby = 192

-- The statement to prove
theorem Pablo_is_70_cm_taller_than_Charlene : Pablo - Charlene = 70 :=
by
  -- Formalizing the proof
  sorry

end Pablo_is_70_cm_taller_than_Charlene_l741_74145


namespace differential_savings_l741_74195

def original_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

theorem differential_savings : (original_tax_rate * annual_income) - (new_tax_rate * annual_income) = 7200 := by
  sorry

end differential_savings_l741_74195


namespace evaluate_expression_at_3_l741_74165

-- Define the expression
def expression (x : ℕ) : ℕ := x^2 - 3*x + 2

-- Statement of the problem
theorem evaluate_expression_at_3 : expression 3 = 2 := by
    sorry -- Proof is omitted

end evaluate_expression_at_3_l741_74165


namespace ratio_of_wages_l741_74171

def hours_per_day_josh : ℕ := 8
def days_per_week : ℕ := 5
def weeks_per_month : ℕ := 4
def wage_per_hour_josh : ℕ := 9
def monthly_total_payment : ℚ := 1980

def hours_per_day_carl : ℕ := hours_per_day_josh - 2

def monthly_hours_josh : ℕ := hours_per_day_josh * days_per_week * weeks_per_month
def monthly_hours_carl : ℕ := hours_per_day_carl * days_per_week * weeks_per_month

def monthly_earnings_josh : ℚ := wage_per_hour_josh * monthly_hours_josh
def monthly_earnings_carl : ℚ := monthly_total_payment - monthly_earnings_josh

def hourly_wage_carl : ℚ := monthly_earnings_carl / monthly_hours_carl

theorem ratio_of_wages : hourly_wage_carl / wage_per_hour_josh = 1 / 2 := by
  sorry

end ratio_of_wages_l741_74171


namespace min_rounds_for_expected_value_l741_74143

theorem min_rounds_for_expected_value 
  (p1 p2 : ℝ) (h0 : 0 ≤ p1 ∧ p1 ≤ 1) (h1 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h2 : p1 + p2 = 3 / 2)
  (indep : true) -- Assuming independence implicitly
  (X : ℕ → ℕ) (n : ℕ)
  (E_X_eq_24 : (n : ℕ) * (3 * p1 * p2 * (1 - p1 * p2)) = 24) :
  n = 32 := 
sorry

end min_rounds_for_expected_value_l741_74143


namespace find_the_number_l741_74164

theorem find_the_number :
  ∃ x : ℕ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := 
  sorry

end find_the_number_l741_74164


namespace boys_tried_out_l741_74169

theorem boys_tried_out (G B C N : ℕ) (hG : G = 9) (hC : C = 2) (hN : N = 21) (h : G + B - C = N) : B = 14 :=
by
  -- The proof is omitted, focusing only on stating the theorem
  sorry

end boys_tried_out_l741_74169


namespace cole_round_trip_time_l741_74148

-- Define the relevant quantities
def speed_to_work : ℝ := 70 -- km/h
def speed_to_home : ℝ := 105 -- km/h
def time_to_work_mins : ℝ := 72 -- minutes

-- Define the theorem to be proved
theorem cole_round_trip_time : 
  (time_to_work_mins / 60 + (speed_to_work * time_to_work_mins / 60) / speed_to_home) = 2 :=
by
  sorry

end cole_round_trip_time_l741_74148


namespace min_bought_chocolates_l741_74163

variable (a b : ℕ)

theorem min_bought_chocolates :
    ∃ a : ℕ, 
        ∃ b : ℕ, 
            b = a + 41 
            ∧ (376 - a - b = 3 * a) 
            ∧ a = 67 :=
by
  sorry

end min_bought_chocolates_l741_74163


namespace gray_area_is_50pi_l741_74156

noncomputable section

-- Define the radii of the inner and outer circles
def R_inner : ℝ := 2.5
def R_outer : ℝ := 3 * R_inner

-- Area of circles
def A_inner : ℝ := Real.pi * R_inner^2
def A_outer : ℝ := Real.pi * R_outer^2

-- Define width of the gray region
def gray_width : ℝ := R_outer - R_inner

-- Gray area calculation
def A_gray : ℝ := A_outer - A_inner

-- The theorem stating the area of the gray region
theorem gray_area_is_50pi :
  gray_width = 5 → A_gray = 50 * Real.pi := by
  -- Here we assume the proof continues
  sorry

end gray_area_is_50pi_l741_74156


namespace arithmetic_sequence_a2_a9_sum_l741_74110

theorem arithmetic_sequence_a2_a9_sum 
  (a : ℕ → ℝ) (d a₁ : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S10 : 10 * a 1 + 45 * d = 120) :
  a 2 + a 9 = 24 :=
sorry

end arithmetic_sequence_a2_a9_sum_l741_74110


namespace decreasing_on_negative_interval_and_max_value_l741_74189

open Classical

noncomputable def f : ℝ → ℝ := sorry  -- Define f later

variables {f : ℝ → ℝ}

-- Hypotheses
axiom h_even : ∀ x, f x = f (-x)
axiom h_increasing_0_7 : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → y ≤ 7 → f x ≤ f y
axiom h_decreasing_7_inf : ∀ ⦃x y : ℝ⦄, 7 ≤ x → x ≤ y → f x ≥ f y
axiom h_f_7_6 : f 7 = 6

-- Theorem Statement
theorem decreasing_on_negative_interval_and_max_value :
  (∀ ⦃x y : ℝ⦄, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
by
  sorry

end decreasing_on_negative_interval_and_max_value_l741_74189


namespace largest_possible_sum_l741_74191

theorem largest_possible_sum :
  let a := 12
  let b := 6
  let c := 6
  let d := 12
  a + b = c + d ∧ a + b + 15 = 33 :=
by
  have h1 : 12 + 6 = 6 + 12 := by norm_num
  have h2 : 12 + 6 + 15 = 33 := by norm_num
  exact ⟨h1, h2⟩

end largest_possible_sum_l741_74191


namespace similarity_coefficient_l741_74117

theorem similarity_coefficient (α : ℝ) :
  (2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)))
  = 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end similarity_coefficient_l741_74117


namespace units_digit_product_l741_74136

theorem units_digit_product (a b : ℕ) (ha : a % 10 = 7) (hb : b % 10 = 4) :
  (a * b) % 10 = 8 := 
by
  sorry

end units_digit_product_l741_74136


namespace median_to_hypotenuse_of_right_triangle_l741_74144

theorem median_to_hypotenuse_of_right_triangle (DE DF : ℝ) (h₁ : DE = 6) (h₂ : DF = 8) :
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  N = 5 :=
by
  let EF := Real.sqrt (DE^2 + DF^2)
  let N := EF / 2
  have h : N = 5 :=
    by
      sorry
  exact h

end median_to_hypotenuse_of_right_triangle_l741_74144


namespace unique_3_digit_number_with_conditions_l741_74121

def valid_3_digit_number (n : ℕ) : Prop :=
  let d2 := n / 100
  let d1 := (n / 10) % 10
  let d0 := n % 10
  (d2 > 0) ∧ (d2 < 10) ∧ (d1 < 10) ∧ (d0 < 10) ∧ (d2 + d1 + d0 = 28) ∧ (d0 < 7) ∧ (d0 % 2 = 0)

theorem unique_3_digit_number_with_conditions :
  (∃! n : ℕ, valid_3_digit_number n) :=
sorry

end unique_3_digit_number_with_conditions_l741_74121


namespace compute_n_pow_m_l741_74103

-- Given conditions
variables (n m : ℕ)
axiom n_eq : n = 3
axiom n_plus_one_eq_2m : n + 1 = 2 * m

-- Goal: Prove n^m = 9
theorem compute_n_pow_m : n^m = 9 :=
by {
  -- Proof goes here
  sorry
}

end compute_n_pow_m_l741_74103


namespace percentage_of_alcohol_in_second_vessel_l741_74140

-- Define the problem conditions
def capacity1 : ℝ := 2
def percentage1 : ℝ := 0.35
def alcohol1 := capacity1 * percentage1

def capacity2 : ℝ := 6 
def percentage2 (x : ℝ) : ℝ := 0.01 * x
def alcohol2 (x : ℝ) := capacity2 * percentage2 x

def total_capacity : ℝ := 8
def final_percentage : ℝ := 0.37
def total_alcohol := total_capacity * final_percentage

theorem percentage_of_alcohol_in_second_vessel (x : ℝ) :
  alcohol1 + alcohol2 x = total_alcohol → x = 37.67 :=
by sorry

end percentage_of_alcohol_in_second_vessel_l741_74140


namespace solve_for_q_l741_74127

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l741_74127


namespace part1_part2_part3_l741_74175

-- Part 1
theorem part1 (x : ℝ) (h : abs (x + 2) = abs (x - 4)) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : abs (x + 2) + abs (x - 4) = 8) : x = -3 ∨ x = 5 :=
by
  sorry

-- Part 3
theorem part3 (t : ℝ) :
  let M := -2 - t
  let N := 4 - 3 * t
  (abs M = abs (M - N) → t = 1/2) ∧ 
  (N = 0 → t = 4/3) ∧
  (abs N = abs (N - M) → t = 2) ∧
  (M = N → t = 3) ∧
  (abs (M - N) = abs (2 * M) → t = 8) :=
by
  sorry

end part1_part2_part3_l741_74175


namespace charges_equal_at_x_4_cost_effectiveness_l741_74172

-- Defining the conditions
def full_price : ℕ := 240

def yA (x : ℕ) : ℕ := 120 * x + 240
def yB (x : ℕ) : ℕ := 144 * x + 144

-- (Ⅰ) Establishing the expressions for the charges is already encapsulated in the definitions.

-- (Ⅱ) Proving the equivalence of the two charges for a specific number of students x.
theorem charges_equal_at_x_4 : ∀ x : ℕ, yA x = yB x ↔ x = 4 := 
by {
  sorry
}

-- (Ⅲ) Discussing which travel agency is more cost-effective based on the number of students x.
theorem cost_effectiveness (x : ℕ) :
  (x < 4 → yA x > yB x) ∧ (x > 4 → yA x < yB x) :=
by {
  sorry
}

end charges_equal_at_x_4_cost_effectiveness_l741_74172


namespace find_triples_l741_74152

theorem find_triples 
  (x y z : ℝ)
  (h1 : x + y * z = 2)
  (h2 : y + z * x = 2)
  (h3 : z + x * y = 2)
 : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end find_triples_l741_74152


namespace ms_warren_walking_speed_correct_l741_74194

noncomputable def walking_speed_proof : Prop :=
  let running_speed := 6 -- mph
  let running_time := 20 / 60 -- hours
  let total_distance := 3 -- miles
  let distance_ran := running_speed * running_time
  let distance_walked := total_distance - distance_ran
  let walking_time := 30 / 60 -- hours
  let walking_speed := distance_walked / walking_time
  walking_speed = 2

theorem ms_warren_walking_speed_correct (walking_speed_proof : Prop) : walking_speed_proof :=
by sorry

end ms_warren_walking_speed_correct_l741_74194


namespace andrew_worked_days_l741_74157

-- Definitions per given conditions
def vacation_days_per_work_days (W : ℕ) : ℕ := W / 10
def days_taken_off_in_march := 5
def days_taken_off_in_september := 2 * days_taken_off_in_march
def total_days_off_taken := days_taken_off_in_march + days_taken_off_in_september
def remaining_vacation_days := 15
def total_vacation_days := total_days_off_taken + remaining_vacation_days

theorem andrew_worked_days (W : ℕ) :
  vacation_days_per_work_days W = total_vacation_days → W = 300 := by
  sorry

end andrew_worked_days_l741_74157


namespace part1_part2_l741_74197

theorem part1 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (cos_AB: ℝ), cos_AB = 56 / 65 :=
by {
  sorry
}

theorem part2 (A B : ℝ) (c : ℝ) (cos_A : ℝ) (tan_half_B_add_cot_half_B: ℝ) 
  (h1: cos_A = 5 / 13) 
  (h2: tan_half_B_add_cot_half_B = 10 / 3) 
  (pos_c: c = 21) :
  ∃ (area: ℝ), area = 126 :=
by {
  sorry
}

end part1_part2_l741_74197


namespace second_month_sales_l741_74134

def sales_first_month : ℝ := 7435
def sales_third_month : ℝ := 7855
def sales_fourth_month : ℝ := 8230
def sales_fifth_month : ℝ := 7562
def sales_sixth_month : ℝ := 5991
def average_sales : ℝ := 7500

theorem second_month_sales : 
  ∃ (second_month_sale : ℝ), 
    (sales_first_month + second_month_sale + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = average_sales ∧
    second_month_sale = 7927 := by
  sorry

end second_month_sales_l741_74134


namespace volume_of_sphere_l741_74132

theorem volume_of_sphere (V : ℝ) (r : ℝ) : r = 1 / 3 → (2 * r) = (16 / 9 * V)^(1/3) → V = 1 / 6 :=
by
  intro h_radius h_diameter
  sorry

end volume_of_sphere_l741_74132


namespace riley_outside_fraction_l741_74137

theorem riley_outside_fraction
  (awake_jonsey : ℚ := 2 / 3)
  (jonsey_outside_fraction : ℚ := 1 / 2)
  (awake_riley : ℚ := 3 / 4)
  (total_inside_time : ℚ := 10)
  (hours_per_day : ℕ := 24) :
  let jonsey_inside_time := 1 / 3 * hours_per_day
  let riley_inside_time := (1 - (8 / 9)) * (3 / 4) * hours_per_day
  jonsey_inside_time + riley_inside_time = total_inside_time :=
by
  sorry

end riley_outside_fraction_l741_74137


namespace average_of_seven_starting_with_d_l741_74170

theorem average_of_seven_starting_with_d (c d : ℕ) (h : d = (c + 3)) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_seven_starting_with_d_l741_74170


namespace solve_equation_l741_74139

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l741_74139


namespace paper_sufficient_to_cover_cube_l741_74186

noncomputable def edge_length_cube : ℝ := 1
noncomputable def side_length_sheet : ℝ := 2.5

noncomputable def surface_area_cube : ℝ := 6
noncomputable def area_sheet : ℝ := 6.25

theorem paper_sufficient_to_cover_cube : area_sheet ≥ surface_area_cube :=
  by
    sorry

end paper_sufficient_to_cover_cube_l741_74186


namespace eliminate_denominators_l741_74178

variable {x : ℝ}

theorem eliminate_denominators (h : 3 / (2 * x) = 1 / (x - 1)) :
  3 * x - 3 = 2 * x := 
by
  sorry

end eliminate_denominators_l741_74178


namespace units_digit_of_7_power_exp_is_1_l741_74111

-- Define the periodicity of units digits of powers of 7
def units_digit_seq : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit_power_7 (n : ℕ) : ℕ :=
  units_digit_seq.get! (n % 4)

-- Define the exponent
def exp : ℕ := 8^5

-- Define the modular operation result
def exp_modulo : ℕ := exp % 4

-- Define the main statement
theorem units_digit_of_7_power_exp_is_1 :
  units_digit_power_7 exp = 1 :=
by
  simp [units_digit_power_7, units_digit_seq, exp, exp_modulo]
  sorry

end units_digit_of_7_power_exp_is_1_l741_74111


namespace dvds_still_fit_in_book_l741_74151

def total_capacity : ℕ := 126
def dvds_already_in_book : ℕ := 81

theorem dvds_still_fit_in_book : (total_capacity - dvds_already_in_book = 45) :=
by
  sorry

end dvds_still_fit_in_book_l741_74151


namespace perfect_square_trinomial_coeff_l741_74153

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end perfect_square_trinomial_coeff_l741_74153


namespace snakes_in_each_cage_l741_74100

theorem snakes_in_each_cage (total_snakes : ℕ) (total_cages : ℕ) (h_snakes: total_snakes = 4) (h_cages: total_cages = 2) 
  (h_even_distribution : (total_snakes % total_cages) = 0) : (total_snakes / total_cages) = 2 := 
by sorry

end snakes_in_each_cage_l741_74100


namespace divisor_is_five_l741_74181

theorem divisor_is_five (n d : ℕ) (h1 : ∃ k, n = k * d + 3) (h2 : ∃ l, n^2 = l * d + 4) : d = 5 :=
sorry

end divisor_is_five_l741_74181


namespace molecular_weight_AlPO4_correct_l741_74196

-- Noncomputable because we are working with specific numerical values.
noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_P : ℝ := 30.97
noncomputable def atomic_weight_O : ℝ := 16.00

noncomputable def molecular_weight_AlPO4 : ℝ := 
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

theorem molecular_weight_AlPO4_correct : molecular_weight_AlPO4 = 121.95 := by
  sorry

end molecular_weight_AlPO4_correct_l741_74196


namespace find_m_from_split_l741_74125

theorem find_m_from_split (m : ℕ) (h1 : m > 1) (h2 : m^2 - m + 1 = 211) : True :=
by
  -- This theorem states that under the conditions that m is a positive integer greater than 1
  -- and m^2 - m + 1 = 211, there exists an integer value for m that satisfies these conditions.
  trivial

end find_m_from_split_l741_74125


namespace degree_of_g_l741_74107

theorem degree_of_g 
  (f : Polynomial ℤ)
  (g : Polynomial ℤ) 
  (h₁ : f = -9 * Polynomial.X^5 + 4 * Polynomial.X^3 - 2 * Polynomial.X + 6)
  (h₂ : (f + g).degree = 2) :
  g.degree = 5 :=
sorry

end degree_of_g_l741_74107


namespace cutting_wire_random_event_l741_74138

noncomputable def length : ℝ := sorry

def is_random_event (a : ℝ) : Prop := sorry

theorem cutting_wire_random_event (a : ℝ) (h : a > 0) :
  is_random_event a := 
by
  sorry

end cutting_wire_random_event_l741_74138


namespace tamara_is_17_over_6_times_taller_than_kim_l741_74198

theorem tamara_is_17_over_6_times_taller_than_kim :
  ∀ (T K : ℕ), T = 68 → T + K = 92 → (T : ℚ) / K = 17 / 6 :=
by
  intros T K hT hSum
  -- proof steps go here, but we use sorry to skip the proof
  sorry

end tamara_is_17_over_6_times_taller_than_kim_l741_74198


namespace permits_cost_l741_74176

-- Definitions based on conditions
def total_cost : ℕ := 2950
def contractor_hourly_rate : ℕ := 150
def contractor_hours_per_day : ℕ := 5
def contractor_days : ℕ := 3
def inspector_discount_rate : ℕ := 80

-- Proving the cost of permits
theorem permits_cost : ∃ (permits_cost : ℕ), permits_cost = 250 :=
by
  let contractor_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (contractor_hourly_rate * inspector_discount_rate / 100)
  let inspector_cost := contractor_hours * inspector_hourly_rate
  let total_cost_without_permits := contractor_cost + inspector_cost
  let permits_cost := total_cost - total_cost_without_permits
  use permits_cost
  sorry

end permits_cost_l741_74176


namespace min_perimeter_triangle_l741_74108

theorem min_perimeter_triangle (a b c : ℝ) (cosC : ℝ) :
  a + b = 10 ∧ cosC = -1/2 ∧ c^2 = (a - 5)^2 + 75 →
  a + b + c = 10 + 5 * Real.sqrt 3 :=
by
  sorry

end min_perimeter_triangle_l741_74108


namespace swapped_digits_greater_by_18_l741_74142

theorem swapped_digits_greater_by_18 (x : ℕ) : 
  (10 * x + 1) - (10 + x) = 18 :=
  sorry

end swapped_digits_greater_by_18_l741_74142


namespace train_average_speed_l741_74149

theorem train_average_speed (speed : ℕ) (stop_time : ℕ) (running_time : ℕ) (total_time : ℕ)
  (h1 : speed = 60)
  (h2 : stop_time = 24)
  (h3 : running_time = total_time - stop_time)
  (h4 : running_time = 36)
  (h5 : total_time = 60) :
  (speed * running_time / total_time = 36) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end train_average_speed_l741_74149


namespace percentage_calculation_l741_74118

theorem percentage_calculation :
  ∀ (P : ℝ),
  (0.3 * 0.5 * 4400 = 99) →
  (P * 4400 = 99) →
  P = 0.0225 :=
by
  intros P condition1 condition2
  -- From the given conditions, it follows directly
  sorry

end percentage_calculation_l741_74118


namespace initial_geese_count_l741_74179

-- Define the number of geese that flew away
def geese_flew_away : ℕ := 28

-- Define the number of geese left in the field
def geese_left : ℕ := 23

-- Prove that the initial number of geese in the field was 51
theorem initial_geese_count : geese_left + geese_flew_away = 51 := by
  sorry

end initial_geese_count_l741_74179


namespace equal_opposite_roots_eq_m_l741_74180

theorem equal_opposite_roots_eq_m (a b c : ℝ) (m : ℝ) (h : (∃ x : ℝ, (a * x - c ≠ 0) ∧ (((x^2 - b * x) / (a * x - c)) = ((m - 1) / (m + 1)))) ∧
(∀ x : ℝ, ((x^2 - b * x) = 0 → x = 0) ∧ (∃ t : ℝ, t > 0 ∧ ((x = t) ∨ (x = -t))))):
  m = (a - b) / (a + b) :=
by
  sorry

end equal_opposite_roots_eq_m_l741_74180


namespace monotonic_if_and_only_if_extreme_point_inequality_l741_74120

noncomputable def f (x a : ℝ) : ℝ := x^2 - 1 + a * Real.log (1 - x)

def is_monotonic (a : ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x a ≤ f y a

theorem monotonic_if_and_only_if (a : ℝ) : 
  is_monotonic a ↔ a ≥ 0.5 :=
sorry

theorem extreme_point_inequality (a : ℝ) (x1 x2 : ℝ) (hₐ : 0 < a ∧ a < 0.5) 
  (hx : x1 < x2) (hx₁₂ : f x1 a = f x2 a) : 
  f x1 a / x2 > f x2 a / x1 :=
sorry

end monotonic_if_and_only_if_extreme_point_inequality_l741_74120


namespace largest_angle_of_convex_hexagon_l741_74114

noncomputable def hexagon_largest_angle (x : ℚ) : ℚ :=
  max (6 * x - 3) (max (5 * x + 1) (max (4 * x - 4) (max (3 * x) (max (2 * x + 2) x))))

theorem largest_angle_of_convex_hexagon (x : ℚ) (h : x + (2*x+2) + 3*x + (4*x-4) + (5*x+1) + (6*x-3) = 720) : 
  hexagon_largest_angle x = 4281 / 21 := 
sorry

end largest_angle_of_convex_hexagon_l741_74114


namespace problem_1_problem_2_l741_74133

-- Problem 1 statement
theorem problem_1 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_cond_a : a = 1/4) (h_cond_q : (1 : ℝ) / 2 < x ∧ x < 1) (h_cond_p : a < x ∧ x < 3 * a): 1 / 2 < x ∧ x < 3 / 4 :=
by sorry

-- Problem 2 statement
theorem problem_2 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_neg_p : ¬(a < x ∧ x < 3 * a)) (h_neg_q : ¬((1 / (2 : ℝ))^(m - 1) < x ∧ x < 1)): 1 / 3 ≤ a ∧ a ≤ 1 / 2 :=
by sorry

end problem_1_problem_2_l741_74133


namespace exists_complex_on_line_y_eq_neg_x_l741_74130

open Complex

theorem exists_complex_on_line_y_eq_neg_x :
  ∃ (z : ℂ), ∃ (a b : ℝ), z = a + b * I ∧ b = -a :=
by
  use 1 - I
  use 1, -1
  sorry

end exists_complex_on_line_y_eq_neg_x_l741_74130


namespace midpoint_reflection_sum_l741_74113

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

end midpoint_reflection_sum_l741_74113


namespace average_score_remaining_students_l741_74160

theorem average_score_remaining_students (n : ℕ) (h : n > 15) (avg_all : ℚ) (avg_15 : ℚ) :
  avg_all = 12 → avg_15 = 20 →
  (∃ avg_remaining : ℚ, avg_remaining = (12 * n - 300) / (n - 15)) :=
by
  sorry

end average_score_remaining_students_l741_74160


namespace either_p_or_q_false_suff_not_p_true_l741_74129

theorem either_p_or_q_false_suff_not_p_true (p q : Prop) : (p ∨ q = false) → (¬p = true) :=
by
  sorry

end either_p_or_q_false_suff_not_p_true_l741_74129


namespace sum_of_g_49_l741_74177

def f (x : ℝ) := 4 * x^2 - 3
def g (y : ℝ) := y^2 + 2 * y + 2

theorem sum_of_g_49 : (g 49) = 30 :=
  sorry

end sum_of_g_49_l741_74177


namespace sum_and_product_of_white_are_white_l741_74126

-- Definitions based on the conditions
def is_colored_black_or_white (n : ℕ) : Prop :=
  true -- This is a simplified assumption since this property is always true.

def is_black (n : ℕ) : Prop := (n % 2 = 0)
def is_white (n : ℕ) : Prop := (n % 2 = 1)

-- Conditions given in the problem
axiom sum_diff_colors_is_black (a b : ℕ) (ha : is_black a) (hb : is_white b) : is_black (a + b)
axiom infinitely_many_whites : ∀ n, ∃ m ≥ n, is_white m

-- Statement to prove that the sum and product of two white numbers are white
theorem sum_and_product_of_white_are_white (a b : ℕ) (ha : is_white a) (hb : is_white b) : 
  is_white (a + b) ∧ is_white (a * b) :=
sorry

end sum_and_product_of_white_are_white_l741_74126


namespace problem_equivalent_l741_74119

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) / Real.log 4 - 1

theorem problem_equivalent : 
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = Real.log (x + 2) / Real.log 4 - 1) →
  {x : ℝ | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h_even h_def
  sorry

end problem_equivalent_l741_74119


namespace mn_eq_neg_infty_to_0_l741_74104

-- Definitions based on the conditions
def M : Set ℝ := {y | y ≤ 2}
def N : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Set difference definition
def set_diff (A B : Set ℝ) : Set ℝ := {y | y ∈ A ∧ y ∉ B}

-- The proof statement we need to prove
theorem mn_eq_neg_infty_to_0 : set_diff M N = {y | y < 0} :=
  sorry  -- Proof will go here

end mn_eq_neg_infty_to_0_l741_74104


namespace set_of_points_l741_74150

theorem set_of_points : {p : ℝ × ℝ | (2 * p.1 - p.2 = 1) ∧ (p.1 + 4 * p.2 = 5)} = { (1, 1) } :=
by
  sorry

end set_of_points_l741_74150


namespace old_barbell_cost_l741_74187

theorem old_barbell_cost (x : ℝ) (new_barbell_cost : ℝ) (h1 : new_barbell_cost = 1.30 * x) (h2 : new_barbell_cost = 325) : x = 250 :=
by
  sorry

end old_barbell_cost_l741_74187


namespace pyramid_levels_l741_74101

theorem pyramid_levels (n : ℕ) (h : (n * (n + 1) * (2 * n + 1)) / 6 = 225) : n = 6 :=
by
  sorry

end pyramid_levels_l741_74101


namespace calculate_f3_l741_74116

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^7 + a * x^5 + b * x - 5

theorem calculate_f3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := 
by
  sorry

end calculate_f3_l741_74116


namespace expression_evaluation_l741_74128

theorem expression_evaluation (x : ℤ) (hx : x = 4) : 5 * x + 3 - x^2 = 7 :=
by
  sorry

end expression_evaluation_l741_74128


namespace train_length_is_95_l741_74141

noncomputable def train_length (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ := 
  let speed_ms := speed_kmh * 1000 / 3600 
  speed_ms * time_seconds

theorem train_length_is_95 : train_length 1.5980030008814248 214 = 95 := by
  sorry

end train_length_is_95_l741_74141


namespace length_of_other_leg_l741_74192

theorem length_of_other_leg (c a b : ℕ) (h1 : c = 10) (h2 : a = 6) (h3 : c^2 = a^2 + b^2) : b = 8 :=
by
  sorry

end length_of_other_leg_l741_74192


namespace fraction_equality_l741_74184

theorem fraction_equality 
  (a b c d : ℝ)
  (h1 : a + c = 2 * b)
  (h2 : 2 * b * d = c * (b + d))
  (hb : b ≠ 0)
  (hd : d ≠ 0) :
  a / b = c / d :=
sorry

end fraction_equality_l741_74184


namespace bill_apples_left_l741_74183

-- Definitions based on the conditions
def total_apples : Nat := 50
def apples_per_child : Nat := 3
def number_of_children : Nat := 2
def apples_per_pie : Nat := 10
def number_of_pies : Nat := 2

-- The main statement to prove
theorem bill_apples_left : total_apples - ((apples_per_child * number_of_children) + (apples_per_pie * number_of_pies)) = 24 := by
sorry

end bill_apples_left_l741_74183


namespace lateral_surface_area_of_prism_l741_74105

theorem lateral_surface_area_of_prism 
  (a : ℝ) (α β V : ℝ) :
  let sin (x : ℝ) := Real.sin x 
  ∃ S : ℝ,
    S = (2 * V * sin ((α + β) / 2)) / (a * sin (α / 2) * sin (β / 2)) := 
sorry

end lateral_surface_area_of_prism_l741_74105


namespace sum_infinite_series_l741_74154

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l741_74154


namespace sum_ages_l741_74158

variables (uncle_age eunji_age yuna_age : ℕ)

def EunjiAge (uncle_age : ℕ) := uncle_age - 25
def YunaAge (eunji_age : ℕ) := eunji_age + 3

theorem sum_ages (h_uncle : uncle_age = 41) (h_eunji : EunjiAge uncle_age = eunji_age) (h_yuna : YunaAge eunji_age = yuna_age) :
  eunji_age + yuna_age = 35 :=
sorry

end sum_ages_l741_74158


namespace triangle_angle_B_l741_74109

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h : a / b = 3 / Real.sqrt 7) (h2 : b / c = Real.sqrt 7 / 2) : B = Real.pi / 3 :=
by
  sorry

end triangle_angle_B_l741_74109


namespace eric_has_more_than_500_paperclips_on_saturday_l741_74155

theorem eric_has_more_than_500_paperclips_on_saturday :
  ∃ k : ℕ, (4 * 3 ^ k > 500) ∧ (∀ m : ℕ, m < k → 4 * 3 ^ m ≤ 500) ∧ ((k + 1) % 7 = 6) :=
by
  sorry

end eric_has_more_than_500_paperclips_on_saturday_l741_74155


namespace no_real_roots_of_quadratic_l741_74122

theorem no_real_roots_of_quadratic (k : ℝ) (h : 12 - 3 * k < 0) : ∀ (x : ℝ), ¬ (x^2 + 4 * x + k = 0) := by
  sorry

end no_real_roots_of_quadratic_l741_74122


namespace value_of_complex_fraction_l741_74123

theorem value_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) : ((1 - i) / (1 + i)) ^ 2 = -1 :=
by
  sorry

end value_of_complex_fraction_l741_74123


namespace tom_has_hours_to_spare_l741_74188

theorem tom_has_hours_to_spare 
  (num_walls : ℕ) 
  (wall_length wall_height : ℕ) 
  (painting_rate : ℕ) 
  (total_hours : ℕ) 
  (num_walls_eq : num_walls = 5) 
  (wall_length_eq : wall_length = 2) 
  (wall_height_eq : wall_height = 3) 
  (painting_rate_eq : painting_rate = 10) 
  (total_hours_eq : total_hours = 10)
  : total_hours - (num_walls * wall_length * wall_height * painting_rate) / 60 = 5 := 
sorry

end tom_has_hours_to_spare_l741_74188


namespace cone_surface_area_l741_74159

-- Define the surface area formula for a cone with radius r and slant height l
theorem cone_surface_area (r l : ℝ) : 
  let S := π * r^2 + π * r * l
  S = π * r^2 + π * r * l :=
by sorry

end cone_surface_area_l741_74159


namespace find_x_l741_74115

def operation (a b : ℝ) : ℝ := a * b^(1/2)

theorem find_x (x : ℝ) : operation x 9 = 12 → x = 4 :=
by
  intro h
  sorry

end find_x_l741_74115


namespace event_eq_conds_l741_74199

-- Definitions based on conditions
def Die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }
def sum_points (d1 d2 : Die) : ℕ := d1.val + d2.val

def event_xi_eq_4 (d1 d2 : Die) : Prop := 
  sum_points d1 d2 = 4

def condition_a (d1 d2 : Die) : Prop := 
  d1.val = 2 ∧ d2.val = 2

def condition_b (d1 d2 : Die) : Prop := 
  (d1.val = 3 ∧ d2.val = 1) ∨ (d1.val = 1 ∧ d2.val = 3)

def event_condition (d1 d2 : Die) : Prop :=
  condition_a d1 d2 ∨ condition_b d1 d2

-- The main Lean statement
theorem event_eq_conds (d1 d2 : Die) : 
  event_xi_eq_4 d1 d2 ↔ event_condition d1 d2 := 
by
  sorry

end event_eq_conds_l741_74199


namespace combined_cost_of_apples_and_strawberries_l741_74161

theorem combined_cost_of_apples_and_strawberries :
  let cost_of_apples := 15
  let cost_of_strawberries := 26
  cost_of_apples + cost_of_strawberries = 41 :=
by
  sorry

end combined_cost_of_apples_and_strawberries_l741_74161


namespace sixth_employee_salary_l741_74106

def salaries : List Real := [1000, 2500, 3100, 3650, 1500]

def mean_salary_of_six : Real := 2291.67

theorem sixth_employee_salary : 
  let total_five := salaries.sum 
  let total_six := mean_salary_of_six * 6
  (total_six - total_five) = 2000.02 :=
by
  sorry

end sixth_employee_salary_l741_74106


namespace monkeys_and_bananas_l741_74124

theorem monkeys_and_bananas :
  (∀ (m n t : ℕ), m * t = n → (∀ (m' n' t' : ℕ), n = m * (t / t') → n' = (m' * t') / t → n' = n → m' = m)) →
  (6 : ℕ) = 6 :=
by
  intros H
  let m := 6
  let n := 6
  let t := 6
  have H1 : m * t = n := by sorry
  let k := 18
  let t' := 18
  have H2 : n = m * (t / t') := by sorry
  let n' := 18
  have H3 : n' = (m * t') / t := by sorry
  have H4 : n' = n := by sorry
  exact H m n t H1 6 n' t' H2 H3 H4

end monkeys_and_bananas_l741_74124


namespace shed_width_l741_74112

theorem shed_width (backyard_length backyard_width shed_length area_needed : ℝ)
  (backyard_area : backyard_length * backyard_width = 260)
  (sod_area : area_needed = 245)
  (shed_dim : shed_length = 3) :
  (backyard_length * backyard_width - area_needed) / shed_length = 5 :=
by
  -- We need to prove the width of the shed given the conditions
  sorry

end shed_width_l741_74112


namespace f_96_l741_74185

noncomputable def f : ℕ → ℝ := sorry -- assume f is defined somewhere

axiom f_property (a b k : ℕ) (h : a + b = 3 * 2^k) : f a + f b = 2 * k^2

theorem f_96 : f 96 = 20 :=
by
  -- Here we should provide the proof, but for now we use sorry
  sorry

end f_96_l741_74185


namespace count_positive_integers_m_l741_74174

theorem count_positive_integers_m :
  ∃ m_values : Finset ℕ, m_values.card = 4 ∧ ∀ m ∈ m_values, 
    ∃ k : ℕ, k > 0 ∧ (7 * m + 2 = m * k + 2 * m) := 
sorry

end count_positive_integers_m_l741_74174


namespace henrietta_paint_gallons_l741_74193

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l741_74193


namespace least_number_to_add_l741_74166

theorem least_number_to_add (n : ℕ) : (3457 + n) % 103 = 0 ↔ n = 45 :=
by sorry

end least_number_to_add_l741_74166


namespace no_integers_satisfy_l741_74135

def P (x a b c d : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integers_satisfy :
  ∀ a b c d : ℤ, ¬ (P 19 a b c d = 1 ∧ P 62 a b c d = 2) :=
by
  intro a b c d
  sorry

end no_integers_satisfy_l741_74135


namespace find_a_l741_74147

noncomputable def P (a : ℚ) (k : ℕ) : ℚ := a * (1 / 2)^(k)

theorem find_a (a : ℚ) : (P a 1 + P a 2 + P a 3 = 1) → (a = 8 / 7) :=
by
  sorry

end find_a_l741_74147


namespace geometric_seq_condition_l741_74131

variable (n : ℕ) (a : ℕ → ℝ)

-- The definition of a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n + 1) = a n * a (n + 2)

-- The main theorem statement
theorem geometric_seq_condition :
  (is_geometric_seq a n → ∀ n, |a n| ≥ 0) →
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = a (n + 1) * a (n + 1)) →
  (∀ m, a m = 0 → ¬(is_geometric_seq a n)) :=
sorry

end geometric_seq_condition_l741_74131


namespace sum_final_numbers_l741_74168

theorem sum_final_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_final_numbers_l741_74168
