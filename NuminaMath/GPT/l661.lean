import Mathlib

namespace min_cubes_required_l661_66107

def volume_of_box (L W H : ℕ) : ℕ := L * W * H
def volume_of_cube (v_cube : ℕ) : ℕ := v_cube
def minimum_number_of_cubes (V_box V_cube : ℕ) : ℕ := V_box / V_cube

theorem min_cubes_required :
  minimum_number_of_cubes (volume_of_box 12 16 6) (volume_of_cube 3) = 384 :=
by sorry

end min_cubes_required_l661_66107


namespace min_value_expression_l661_66176

theorem min_value_expression :
  ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ (5 * Real.sqrt 6) / 3 :=
by
  sorry

end min_value_expression_l661_66176


namespace fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l661_66137

theorem fixed_point_of_line (a : ℝ) (A : ℝ × ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> A = (1, 6)) :=
sorry

theorem range_of_a_to_avoid_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> x * y < 0 -> a ≤ -5) :=
sorry

end fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l661_66137


namespace years_later_l661_66165

variables (R F Y : ℕ)

-- Conditions
def condition1 := F = 4 * R
def condition2 := F + Y = 5 * (R + Y) / 2
def condition3 := F + Y + 8 = 2 * (R + Y + 8)

-- The result to be proved
theorem years_later (R F Y : ℕ) (h1 : condition1 R F) (h2 : condition2 R F Y) (h3 : condition3 R F Y) : 
  Y = 8 := by
  sorry

end years_later_l661_66165


namespace imaginary_part_inv_z_l661_66196

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_inv_z : Complex.im (1 / z) = 2 / 5 :=
by
  -- proof to be filled in
  sorry

end imaginary_part_inv_z_l661_66196


namespace find_x_l661_66169

theorem find_x (x y z : ℕ) (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_eq1 : x + y + z = 37) (h_eq2 : 5 * y = 6 * z) : x = 21 :=
sorry

end find_x_l661_66169


namespace distance_between_foci_l661_66142

theorem distance_between_foci :
  let x := ℝ
  let y := ℝ
  ∀ (x y : ℝ), 9*x^2 + 36*x + 4*y^2 - 8*y + 1 = 0 →
  ∃ (d : ℝ), d = (Real.sqrt 351) / 3 :=
sorry

end distance_between_foci_l661_66142


namespace coin_probability_l661_66110

theorem coin_probability (a r : ℝ) (h : r < a / 2) :
  let favorable_cells := 3
  let larger_cell_area := 9 * a^2
  let favorable_area_per_cell := (a - 2 * r)^2
  let favorable_area := favorable_cells * favorable_area_per_cell
  let probability := favorable_area / larger_cell_area
  probability = (a - 2 * r)^2 / (3 * a^2) :=
by
  sorry

end coin_probability_l661_66110


namespace matthew_total_time_on_failure_day_l661_66116

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end matthew_total_time_on_failure_day_l661_66116


namespace xenia_weekly_earnings_l661_66187

theorem xenia_weekly_earnings
  (hours_week_1 : ℕ)
  (hours_week_2 : ℕ)
  (week2_additional_earnings : ℕ)
  (hours_week_3 : ℕ)
  (bonus_week_3 : ℕ)
  (hourly_wage : ℚ)
  (earnings_week_1 : ℚ)
  (earnings_week_2 : ℚ)
  (earnings_week_3 : ℚ)
  (total_earnings : ℚ) :
  hours_week_1 = 18 →
  hours_week_2 = 25 →
  week2_additional_earnings = 60 →
  hours_week_3 = 28 →
  bonus_week_3 = 30 →
  hourly_wage = (60 : ℚ) / (25 - 18) →
  earnings_week_1 = hours_week_1 * hourly_wage →
  earnings_week_2 = hours_week_2 * hourly_wage →
  earnings_week_2 = earnings_week_1 + 60 →
  earnings_week_3 = hours_week_3 * hourly_wage + 30 →
  total_earnings = earnings_week_1 + earnings_week_2 + earnings_week_3 →
  hourly_wage = (857 : ℚ) / 1000 ∧
  total_earnings = (63947 : ℚ) / 100
:= by
  intros h1 h2 h3 h4 h5 hw he1 he2 he2_60 he3 hte
  sorry

end xenia_weekly_earnings_l661_66187


namespace subtraction_example_l661_66160

theorem subtraction_example : 6102 - 2016 = 4086 := by
  sorry

end subtraction_example_l661_66160


namespace range_of_a_l661_66178

theorem range_of_a (a x : ℝ) (h_eq : 2 * x - 1 = x + a) (h_pos : x > 0) : a > -1 :=
sorry

end range_of_a_l661_66178


namespace value_of_expression_l661_66158

theorem value_of_expression (x : ℝ) (hx : 23 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 5 :=
by
  sorry

end value_of_expression_l661_66158


namespace badminton_members_count_l661_66179

def total_members := 30
def neither_members := 2
def both_members := 6

def members_play_badminton_and_tennis (B T : ℕ) : Prop :=
  B + T - both_members = total_members - neither_members

theorem badminton_members_count (B T : ℕ) (hbt : B = T) :
  members_play_badminton_and_tennis B T → B = 17 :=
by
  intros h
  sorry

end badminton_members_count_l661_66179


namespace stamens_in_bouquet_l661_66111

-- Define the number of pistils, leaves, stamens for black roses and crimson flowers
def pistils_black_rose : ℕ := 4
def stamens_black_rose : ℕ := 4
def leaves_black_rose : ℕ := 2

def pistils_crimson_flower : ℕ := 8
def stamens_crimson_flower : ℕ := 10
def leaves_crimson_flower : ℕ := 3

-- Define the number of black roses and crimson flowers (as variables x and y)
variables (x y : ℕ)

-- Define the total number of pistils and leaves in the bouquet
def total_pistils : ℕ := pistils_black_rose * x + pistils_crimson_flower * y
def total_leaves : ℕ := leaves_black_rose * x + leaves_crimson_flower * y

-- Condition: There are 108 fewer leaves than pistils
axiom leaves_pistils_relation : total_leaves = total_pistils - 108

-- Calculate the total number of stamens in the bouquet
def total_stamens : ℕ := stamens_black_rose * x + stamens_crimson_flower * y

-- The theorem to be proved
theorem stamens_in_bouquet : total_stamens = 216 :=
by
  sorry

end stamens_in_bouquet_l661_66111


namespace sin_cos_identity_l661_66173

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := 
by
  sorry

end sin_cos_identity_l661_66173


namespace garden_least_cost_l661_66199

-- Define the costs per flower type
def cost_sunflower : ℝ := 0.75
def cost_tulip : ℝ := 2
def cost_marigold : ℝ := 1.25
def cost_orchid : ℝ := 4
def cost_violet : ℝ := 3.5

-- Define the areas of each section
def area_top_left : ℝ := 5 * 2
def area_bottom_left : ℝ := 5 * 5
def area_top_right : ℝ := 3 * 5
def area_bottom_right : ℝ := 3 * 4
def area_central_right : ℝ := 5 * 3

-- Calculate the total costs after assigning the most cost-effective layout
def total_cost : ℝ :=
  (area_top_left * cost_orchid) +
  (area_bottom_right * cost_violet) +
  (area_central_right * cost_tulip) +
  (area_bottom_left * cost_marigold) +
  (area_top_right * cost_sunflower)

-- Prove that the total cost is $154.50
theorem garden_least_cost : total_cost = 154.50 :=
by sorry

end garden_least_cost_l661_66199


namespace number_of_planting_methods_l661_66122

noncomputable def num_planting_methods : ℕ :=
  -- Six different types of crops
  let crops := ['A', 'B', 'C', 'D', 'E', 'F']
  -- Six trial fields arranged in a row, numbered 1 through 6
  -- Condition: Crop A cannot be planted in the first two fields
  -- Condition: Crop B must not be adjacent to crop A
  -- Answer: 240 different planting methods
  240

theorem number_of_planting_methods :
  num_planting_methods = 240 :=
  by
    -- Proof omitted
    sorry

end number_of_planting_methods_l661_66122


namespace increasing_function_range_a_l661_66191

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem increasing_function_range_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 1 < a ∧ a ≤ 5 / 3 :=
sorry

end increasing_function_range_a_l661_66191


namespace new_average_height_is_184_l661_66109

-- Define the initial conditions
def original_num_students : ℕ := 35
def original_avg_height : ℕ := 180
def left_num_students : ℕ := 7
def left_avg_height : ℕ := 120
def joined_num_students : ℕ := 7
def joined_avg_height : ℕ := 140

-- Calculate the initial total height
def original_total_height := original_avg_height * original_num_students

-- Calculate the total height of the students who left
def left_total_height := left_avg_height * left_num_students

-- Calculate the new total height after the students left
def new_total_height1 := original_total_height - left_total_height

-- Calculate the total height of the new students who joined
def joined_total_height := joined_avg_height * joined_num_students

-- Calculate the new total height after the new students joined
def new_total_height2 := new_total_height1 + joined_total_height

-- Calculate the new average height
def new_avg_height := new_total_height2 / original_num_students

-- The theorem stating the result
theorem new_average_height_is_184 : new_avg_height = 184 := by
  sorry

end new_average_height_is_184_l661_66109


namespace find_values_l661_66190

noncomputable def equation_satisfaction (x y : ℝ) : Prop :=
  x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3

theorem find_values (x y : ℝ) :
  equation_satisfaction x y → x = 1 / 3 ∧ y = 2 / 3 :=
by
  intro h
  sorry

end find_values_l661_66190


namespace find_f_neg5_l661_66138

theorem find_f_neg5 (a b : ℝ) (Sin : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b * (Sin x) ^ 3 + 1)
  (h_f5 : f 5 = 7) :
  f (-5) = -5 := 
by
  sorry

end find_f_neg5_l661_66138


namespace q_poly_correct_l661_66147

open Polynomial

noncomputable def q : Polynomial ℚ := 
  -(C 1) * X^6 + C 4 * X^4 + C 21 * X^3 + C 15 * X^2 + C 14 * X + C 3

theorem q_poly_correct : 
  ∀ x : Polynomial ℚ,
  q + (X^6 + 4 * X^4 + 5 * X^3 + 12 * X) = 
  (8 * X^4 + 26 * X^3 + 15 * X^2 + 26 * X + C 3) := by sorry

end q_poly_correct_l661_66147


namespace problem_statement_l661_66125

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l661_66125


namespace unique_12_tuple_l661_66105

theorem unique_12_tuple : 
  ∃! (x : Fin 12 → ℝ), 
    ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2 + 
    (x 11)^2 = 1 / 13) ∧ (x 0 + x 11 = 1 / 2) :=
by
  sorry

end unique_12_tuple_l661_66105


namespace compare_M_N_l661_66161

theorem compare_M_N (a b c : ℝ) (h1 : a > 0) (h2 : b < -2 * a) : 
  (|a - b + c| + |2 * a + b|) < (|a + b + c| + |2 * a - b|) :=
by
  sorry

end compare_M_N_l661_66161


namespace difference_in_students_specific_case_diff_l661_66171

-- Define the variables and conditions
variables (a b : ℕ)

-- Condition: a > b
axiom h1 : a > b

-- Definition of eighth grade students
def eighth_grade_students := (3 * a + b) * (2 * a + 2 * b)

-- Definition of seventh grade students
def seventh_grade_students := (2 * (a + b)) ^ 2

-- Theorem for the difference in the number of students
theorem difference_in_students : (eighth_grade_students a b) - (seventh_grade_students a b) = 2 * a^2 - 2 * b^2 :=
sorry

-- Theorem for the specific example when a = 10 and b = 2
theorem specific_case_diff : eighth_grade_students 10 2 - seventh_grade_students 10 2 = 192 :=
sorry

end difference_in_students_specific_case_diff_l661_66171


namespace remainder_of_3_pow_20_mod_7_l661_66145

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l661_66145


namespace prove_p_and_q_l661_66180

def p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1 / 4

def q (A B : ℝ) : Prop :=
  A > B ↔ Real.sin A > Real.sin B

theorem prove_p_and_q :
  (∀ m : ℝ, p m) ∧ (∀ A B : ℝ, q A B) :=
by
  sorry

end prove_p_and_q_l661_66180


namespace greatest_integer_l661_66131

theorem greatest_integer (m : ℕ) (h1 : 0 < m) (h2 : m < 150)
  (h3 : ∃ a : ℤ, m = 9 * a - 2) (h4 : ∃ b : ℤ, m = 5 * b + 4) :
  m = 124 := 
sorry

end greatest_integer_l661_66131


namespace D_score_l661_66135

noncomputable def score_A : ℕ := 94

variables (A B C D E : ℕ)

-- Conditions
def A_scored : A = score_A := sorry
def B_highest : B > A := sorry
def C_average_AD : (C * 2) = A + D := sorry
def D_average_five : (D * 5) = A + B + C + D + E := sorry
def E_score_C2 : E = C + 2 := sorry

-- Question
theorem D_score : D = 96 :=
by {
  sorry
}

end D_score_l661_66135


namespace expression_equals_answer_l661_66106

noncomputable def evaluate_expression : ℚ :=
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 +
  (2013^2 * 2014 - 2015) / Nat.factorial 2014

theorem expression_equals_answer :
  evaluate_expression = 
  1 / Nat.factorial 2009 + 
  1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 
  1 / Nat.factorial 2014 :=
by
  sorry

end expression_equals_answer_l661_66106


namespace mediant_fraction_of_6_11_and_5_9_minimized_is_31_l661_66168

theorem mediant_fraction_of_6_11_and_5_9_minimized_is_31 
  (p q : ℕ) (h_pos : 0 < p ∧ 0 < q)
  (h_bounds : (6 : ℝ) / 11 < p / q ∧ p / q < 5 / 9)
  (h_min_q : ∀ r s : ℕ, (6 : ℝ) / 11 < r / s ∧ r / s < 5 / 9 → s ≥ q) :
  p + q = 31 :=
sorry

end mediant_fraction_of_6_11_and_5_9_minimized_is_31_l661_66168


namespace combinatorics_sum_l661_66126

theorem combinatorics_sum :
  (Nat.choose 20 6 + Nat.choose 20 5 = 62016) :=
by
  sorry

end combinatorics_sum_l661_66126


namespace inequality_of_f_log2015_l661_66184

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_of_f_log2015 :
  (∀ x : ℝ, deriv f x > f x) →
  f (Real.log 2015) > 2015 * f 0 :=
by sorry

end inequality_of_f_log2015_l661_66184


namespace mother_returns_home_at_8_05_l661_66167

noncomputable
def xiaoMing_home_time : Nat := 7 * 60 -- 7:00 AM in minutes
def xiaoMing_speed : Nat := 40 -- in meters per minute
def mother_home_time : Nat := 7 * 60 + 20 -- 7:20 AM in minutes
def meet_point : Nat := 1600 -- in meters
def stay_time : Nat := 5 -- in minutes
def return_duration_by_bike : Nat := 20 -- in minutes

theorem mother_returns_home_at_8_05 :
    (xiaoMing_home_time + (meet_point / xiaoMing_speed) + stay_time + return_duration_by_bike) = (8 * 60 + 5) :=
by
    sorry

end mother_returns_home_at_8_05_l661_66167


namespace cost_per_person_trip_trips_rental_cost_l661_66155

-- Define the initial conditions
def ticket_price_per_person := 60
def total_employees := 70
def small_car_seats := 4
def large_car_seats := 11
def extra_cost_small_car_per_person := 5
def extra_revenue_large_car := 50
def max_total_cost := 5000

-- Define the costs per person per trip for small and large cars
def large_car_cost_per_person := 10
def small_car_cost_per_person := large_car_cost_per_person + extra_cost_small_car_per_person

-- Define the number of trips for four-seater and eleven-seater cars
def four_seater_trips := 1
def eleven_seater_trips := 6

-- Prove the lean statements
theorem cost_per_person_trip : 
  (11 * large_car_cost_per_person) - (small_car_seats * small_car_cost_per_person) = extra_revenue_large_car := 
sorry

theorem trips_rental_cost (x y : ℕ) : 
  (small_car_seats * x + large_car_seats * y = total_employees) ∧
  ((total_employees * ticket_price_per_person) + (small_car_cost_per_person * small_car_seats * x) + (large_car_cost_per_person * large_car_seats * y) ≤ max_total_cost) :=
sorry

end cost_per_person_trip_trips_rental_cost_l661_66155


namespace total_books_l661_66104

def books_per_shelf_mystery : ℕ := 7
def books_per_shelf_picture : ℕ := 5
def books_per_shelf_sci_fi : ℕ := 8
def books_per_shelf_biography : ℕ := 6

def shelves_mystery : ℕ := 8
def shelves_picture : ℕ := 2
def shelves_sci_fi : ℕ := 3
def shelves_biography : ℕ := 4

theorem total_books :
  (books_per_shelf_mystery * shelves_mystery) + 
  (books_per_shelf_picture * shelves_picture) + 
  (books_per_shelf_sci_fi * shelves_sci_fi) + 
  (books_per_shelf_biography * shelves_biography) = 114 :=
by
  sorry

end total_books_l661_66104


namespace balcony_more_than_orchestra_l661_66103

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 340) 
  (h2 : 12 * x + 8 * y = 3320) : 
  y - x = 40 := 
sorry

end balcony_more_than_orchestra_l661_66103


namespace number_of_books_in_library_l661_66193

def number_of_bookcases : ℕ := 28
def shelves_per_bookcase : ℕ := 6
def books_per_shelf : ℕ := 19

theorem number_of_books_in_library : number_of_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end number_of_books_in_library_l661_66193


namespace sequence_formula_l661_66123

theorem sequence_formula (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l661_66123


namespace largest_divisor_of_n_l661_66132

-- Definitions and conditions from the problem
def is_positive_integer (n : ℕ) := n > 0
def is_divisible_by (a b : ℕ) := ∃ k : ℕ, a = k * b

-- Lean 4 statement encapsulating the problem
theorem largest_divisor_of_n (n : ℕ) (h1 : is_positive_integer n) (h2 : is_divisible_by (n * n) 72) : 
  ∃ v : ℕ, v = 12 ∧ is_divisible_by n v := 
sorry

end largest_divisor_of_n_l661_66132


namespace fraction_value_l661_66102

theorem fraction_value :
  (0.02 ^ 2 + 0.52 ^ 2 + 0.035 ^ 2) / (0.002 ^ 2 + 0.052 ^ 2 + 0.0035 ^ 2) = 100 := by
    sorry

end fraction_value_l661_66102


namespace max_difference_is_correct_l661_66124

noncomputable def max_y_difference : ℝ := 
  let x1 := Real.sqrt (2 / 3)
  let y1 := 2 + (x1 ^ 2) + (x1 ^ 3)
  let x2 := -x1
  let y2 := 2 + (x2 ^ 2) + (x2 ^ 3)
  abs (y1 - y2)

theorem max_difference_is_correct : max_y_difference = 4 * Real.sqrt 2 / 9 := 
  sorry -- Proof is omitted

end max_difference_is_correct_l661_66124


namespace prob_of_caps_given_sunglasses_l661_66189

theorem prob_of_caps_given_sunglasses (n_sunglasses n_caps n_both : ℕ) (P_sunglasses_given_caps : ℚ) 
  (h_nsunglasses : n_sunglasses = 80) (h_ncaps : n_caps = 45)
  (h_Psunglasses_given_caps : P_sunglasses_given_caps = 3/8)
  (h_nboth : n_both = P_sunglasses_given_caps * n_sunglasses) :
  (n_both / n_caps) = 2/3 := 
by
  sorry

end prob_of_caps_given_sunglasses_l661_66189


namespace hats_cost_l661_66144

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l661_66144


namespace boat_speed_in_still_water_l661_66153

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := by
  sorry

end boat_speed_in_still_water_l661_66153


namespace y_in_terms_of_x_l661_66108

theorem y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = -2 * x + 5 :=
sorry

end y_in_terms_of_x_l661_66108


namespace inequality_addition_l661_66151

theorem inequality_addition (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end inequality_addition_l661_66151


namespace second_candidate_more_marks_30_l661_66175

noncomputable def total_marks : ℝ := 600
def passing_marks_approx : ℝ := 240

def candidate_marks (percentage : ℝ) (total : ℝ) : ℝ :=
  percentage * total

def more_marks (second_candidate : ℝ) (passing : ℝ) : ℝ :=
  second_candidate - passing

theorem second_candidate_more_marks_30 :
  more_marks (candidate_marks 0.45 total_marks) passing_marks_approx = 30 := by
  sorry

end second_candidate_more_marks_30_l661_66175


namespace result_of_subtraction_l661_66114

theorem result_of_subtraction (N : ℝ) (h1 : N = 100) : 0.80 * N - 20 = 60 :=
by
  sorry

end result_of_subtraction_l661_66114


namespace alice_additional_cookies_proof_l661_66139

variable (alice_initial_cookies : ℕ)
variable (bob_initial_cookies : ℕ)
variable (cookies_thrown_away : ℕ)
variable (bob_additional_cookies : ℕ)
variable (total_edible_cookies : ℕ)

theorem alice_additional_cookies_proof 
    (h1 : alice_initial_cookies = 74)
    (h2 : bob_initial_cookies = 7)
    (h3 : cookies_thrown_away = 29)
    (h4 : bob_additional_cookies = 36)
    (h5 : total_edible_cookies = 93) :
  alice_initial_cookies + bob_initial_cookies - cookies_thrown_away + bob_additional_cookies + (93 - (74 + 7 - 29 + 36)) = total_edible_cookies :=
by
  sorry

end alice_additional_cookies_proof_l661_66139


namespace equal_intercepts_line_l661_66185

theorem equal_intercepts_line (x y : ℝ)
  (h1 : x + 2*y - 6 = 0) 
  (h2 : x - 2*y + 2 = 0) 
  (hx : x = 2) 
  (hy : y = 2) :
  (y = x) ∨ (x + y = 4) :=
sorry

end equal_intercepts_line_l661_66185


namespace triangle_properties_l661_66115

theorem triangle_properties (A B C a b c : ℝ) (h1 : a * Real.tan C = 2 * c * Real.sin A)
  (h2 : C > 0 ∧ C < Real.pi)
  (h3 : a / Real.sin A = c / Real.sin C) :
  C = Real.pi / 3 ∧ (1 / 2 < Real.sin (A + Real.pi / 6) ∧ Real.sin (A + Real.pi / 6) ≤ 1) →
  (Real.sqrt 3 / 2 < Real.sin A + Real.sin B ∧ Real.sin A + Real.sin B ≤ Real.sqrt 3) :=
by
  intro h4
  sorry

end triangle_properties_l661_66115


namespace find_k_l661_66159

theorem find_k (x y k : ℝ) 
  (h1 : 4 * x + 2 * y = 5 * k - 4) 
  (h2 : 2 * x + 4 * y = -1) 
  (h3 : x - y = 1) : 
  k = 1 := 
by sorry

end find_k_l661_66159


namespace exists_nonneg_poly_div_l661_66197

theorem exists_nonneg_poly_div (P : Polynomial ℝ) 
  (hP_pos : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ n, Q.coeff n ≥ 0) ∧ (∀ n, R.coeff n ≥ 0) ∧ (P = Q / R) := 
sorry

end exists_nonneg_poly_div_l661_66197


namespace exists_no_zero_digits_divisible_by_2_pow_100_l661_66186

theorem exists_no_zero_digits_divisible_by_2_pow_100 :
  ∃ (N : ℕ), (2^100 ∣ N) ∧ (∀ d ∈ (N.digits 10), d ≠ 0) := sorry

end exists_no_zero_digits_divisible_by_2_pow_100_l661_66186


namespace matrix_det_evaluation_l661_66157

noncomputable def matrix_det (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1,   x,     y,     z],
    ![1, x + y,   y,     z],
    ![1,   x, x + y,     z],
    ![1,   x,     y, x + y + z]
  ]

theorem matrix_det_evaluation (x y z : ℝ) :
  matrix_det x y z = y * x * x + y * y * x :=
by sorry

end matrix_det_evaluation_l661_66157


namespace probability_intersection_three_elements_l661_66166

theorem probability_intersection_three_elements (U : Finset ℕ) (hU : U = {1, 2, 3, 4, 5}) : 
  ∃ (p : ℚ), p = 5 / 62 :=
by
  sorry

end probability_intersection_three_elements_l661_66166


namespace find_b1_b7_b10_value_l661_66181

open Classical

theorem find_b1_b7_b10_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith_seq : ∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2))
  (h_geom_seq : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r)
  (a3_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (b6_a6_eq : b 6 = a 6)
  (non_zero_seq : ∀ n : ℕ, a n ≠ 0) :
  b 1 * b 7 * b 10 = 8 := 
by 
  sorry

end find_b1_b7_b10_value_l661_66181


namespace jump_difference_l661_66140

-- Definitions based on conditions
def grasshopper_jump : ℕ := 13
def frog_jump : ℕ := 11

-- Proof statement
theorem jump_difference : grasshopper_jump - frog_jump = 2 := by
  sorry

end jump_difference_l661_66140


namespace total_cookies_l661_66194

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1: num_people = 4) (h2: cookies_per_person = 22) : total_cookies = 88 :=
by
  sorry

end total_cookies_l661_66194


namespace area_of_transformed_region_l661_66130

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end area_of_transformed_region_l661_66130


namespace sum_of_80th_equation_l661_66128

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation_l661_66128


namespace no_snow_five_days_l661_66141

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l661_66141


namespace goblin_treasure_l661_66198

theorem goblin_treasure : 
  (∃ d : ℕ, 8000 + 300 * d = 5000 + 500 * d) ↔ ∃ (d : ℕ), d = 15 :=
by
  sorry

end goblin_treasure_l661_66198


namespace length_of_first_train_l661_66127

theorem length_of_first_train
  (speed1_kmph : ℝ) (speed2_kmph : ℝ)
  (time_s : ℝ) (length2_m : ℝ)
  (relative_speed_mps : ℝ := (speed1_kmph + speed2_kmph) * 1000 / 3600)
  (total_distance_m : ℝ := relative_speed_mps * time_s)
  (length1_m : ℝ := total_distance_m - length2_m) :
  speed1_kmph = 80 →
  speed2_kmph = 65 →
  time_s = 7.199424046076314 →
  length2_m = 180 →
  length1_m = 110 :=
by
  sorry

end length_of_first_train_l661_66127


namespace find_cost_price_l661_66162

/-- Statement: Given Mohit sold an article for $18000 and 
if he offered a discount of 10% on the selling price, he would have earned a profit of 8%, 
prove that the cost price (CP) of the article is $15000. -/

def discounted_price (sp : ℝ) := sp - (0.10 * sp)
def profit_price (cp : ℝ) := cp * 1.08

theorem find_cost_price (sp : ℝ) (discount: sp = 18000) (profit_discount: profit_price (discounted_price sp) = discounted_price sp):
    ∃ (cp : ℝ), cp = 15000 :=
by
    sorry

end find_cost_price_l661_66162


namespace train_crosses_pole_in_9_seconds_l661_66121

theorem train_crosses_pole_in_9_seconds
  (speed_kmh : ℝ) (train_length_m : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 58) 
  (h2 : train_length_m = 145) 
  (h3 : time_s = train_length_m / (speed_kmh * 1000 / 3600)) :
  time_s = 9 :=
by
  sorry

end train_crosses_pole_in_9_seconds_l661_66121


namespace tan_30_degrees_correct_l661_66188

noncomputable def tan_30_degrees : ℝ := Real.tan (Real.pi / 6)

theorem tan_30_degrees_correct : tan_30_degrees = Real.sqrt 3 / 3 :=
by
  sorry

end tan_30_degrees_correct_l661_66188


namespace arithmetic_sequences_sum_l661_66195

theorem arithmetic_sequences_sum
  (a b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d1)
  (h2 : ∀ n, b (n + 1) = b n + d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end arithmetic_sequences_sum_l661_66195


namespace negation_of_proposition_l661_66177

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by
  sorry

end negation_of_proposition_l661_66177


namespace carlotta_total_time_l661_66182

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l661_66182


namespace find_a8_l661_66163

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) = a n / n) (h2 : a 5 = 15) : a 8 = 24 :=
sorry

end find_a8_l661_66163


namespace hyperbola_eccentricity_l661_66117

-- Let's define the variables and conditions first
variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variable (h_asymptote : b = a)

-- We need to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l661_66117


namespace problem_statement_l661_66143

theorem problem_statement (x y : ℝ) (h₁ : |x| = 3) (h₂ : |y| = 4) (h₃ : x > y) : 2 * x - y = 10 := 
by {
  sorry
}

end problem_statement_l661_66143


namespace probability_failed_both_tests_eq_l661_66156

variable (total_students pass_test1 pass_test2 pass_both : ℕ)

def students_failed_both_tests (total pass1 pass2 both : ℕ) : ℕ :=
  total - (pass1 + pass2 - both)

theorem probability_failed_both_tests_eq 
  (h_total : total_students = 100)
  (h_pass1 : pass_test1 = 60)
  (h_pass2 : pass_test2 = 40)
  (h_pass_both : pass_both = 20) :
  students_failed_both_tests total_students pass_test1 pass_test2 pass_both / (total_students : ℚ) = 0.2 :=
by
  sorry

end probability_failed_both_tests_eq_l661_66156


namespace sum_of_floors_of_square_roots_l661_66113

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l661_66113


namespace value_of_f_5_l661_66101

variable (f : ℕ → ℕ) (x y : ℕ)

theorem value_of_f_5 (h1 : f 2 = 50) (h2 : ∀ x, f x = 2 * x ^ 2 + y) : f 5 = 92 :=
by
  sorry

end value_of_f_5_l661_66101


namespace Aunt_Zhang_expenditure_is_negative_l661_66100

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l661_66100


namespace area_parallelogram_proof_l661_66129

/-- We are given a rectangle with a length of 10 cm and a width of 8 cm.
    We transform it into a parallelogram with a height of 9 cm.
    We need to prove that the area of the parallelogram is 72 square centimeters. -/
def area_of_parallelogram_from_rectangle (length width height : ℝ) : ℝ :=
  width * height

theorem area_parallelogram_proof
  (length width height : ℝ)
  (h_length : length = 10)
  (h_width : width = 8)
  (h_height : height = 9) :
  area_of_parallelogram_from_rectangle length width height = 72 :=
by
  sorry

end area_parallelogram_proof_l661_66129


namespace choir_members_max_l661_66119

theorem choir_members_max (s x : ℕ) (h1 : s * x < 147) (h2 : s * x + 3 = (s - 3) * (x + 2)) : s * x = 84 :=
sorry

end choir_members_max_l661_66119


namespace cut_half_meter_from_two_thirds_l661_66134

theorem cut_half_meter_from_two_thirds (L : ℝ) (hL : L = 2 / 3) : L - 1 / 6 = 1 / 2 :=
by
  rw [hL]
  norm_num

end cut_half_meter_from_two_thirds_l661_66134


namespace mod_equiv_n_l661_66120

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end mod_equiv_n_l661_66120


namespace problem_rewrite_expression_l661_66112

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l661_66112


namespace first_half_speed_l661_66174

noncomputable def speed_first_half : ℝ := 21

theorem first_half_speed (total_distance first_half_distance second_half_distance second_half_speed total_time : ℝ)
  (h1 : total_distance = 224)
  (h2 : first_half_distance = total_distance / 2)
  (h3 : second_half_distance = total_distance / 2)
  (h4 : second_half_speed = 24)
  (h5 : total_time = 10)
  (h6 : total_time = first_half_distance / speed_first_half + second_half_distance / second_half_speed) :
  speed_first_half = 21 :=
sorry

end first_half_speed_l661_66174


namespace graph_of_equation_is_line_and_hyperbola_l661_66146

theorem graph_of_equation_is_line_and_hyperbola :
  ∀ (x y : ℝ), ((x^2 - 1) * (x + y) = y^2 * (x + y)) ↔ (y = -x) ∨ ((x + y) * (x - y) = 1) := by
  intro x y
  sorry

end graph_of_equation_is_line_and_hyperbola_l661_66146


namespace prove_ln10_order_l661_66150

def ln10_order_proof : Prop :=
  let a := Real.log 10
  let b := Real.log 100
  let c := (Real.log 10) ^ 2
  c > b ∧ b > a

theorem prove_ln10_order : ln10_order_proof := 
sorry

end prove_ln10_order_l661_66150


namespace test_point_third_l661_66183

def interval := (1000, 2000)
def phi := 0.618
def x1 := 1000 + phi * (2000 - 1000)
def x2 := 1000 + 2000 - x1

-- By definition and given the conditions, x3 is computed in a specific manner
def x3 := x2 + 2000 - x1

theorem test_point_third : x3 = 1764 :=
by
  -- Skipping the proof for now
  sorry

end test_point_third_l661_66183


namespace coefficient_of_x7_in_expansion_eq_15_l661_66149

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := n.choose k

theorem coefficient_of_x7_in_expansion_eq_15 (a : ℝ) (hbinom : binomial 10 3 * (-a) ^ 3 = 15) : a = -1 / 2 := by
  sorry

end coefficient_of_x7_in_expansion_eq_15_l661_66149


namespace max_students_with_equal_distribution_l661_66164

theorem max_students_with_equal_distribution (pens pencils : ℕ) (h_pens : pens = 3540) (h_pencils : pencils = 2860) :
  gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  -- Proof steps will go here
  sorry

end max_students_with_equal_distribution_l661_66164


namespace highway_length_on_map_l661_66170

theorem highway_length_on_map (total_length_km : ℕ) (scale : ℚ) (length_on_map_cm : ℚ) 
  (h1 : total_length_km = 155) (h2 : scale = 1 / 500000) :
  length_on_map_cm = 31 :=
by
  sorry

end highway_length_on_map_l661_66170


namespace lowest_possible_price_l661_66136

def typeADiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 15 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 20 / 100
  discountedPrice - additionalDiscount

def typeBDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 25 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 15 / 100
  discountedPrice - additionalDiscount

def typeCDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 30 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 10 / 100
  discountedPrice - additionalDiscount

def finalPrice (discountedPrice : ℕ) : ℕ :=
  let tax := discountedPrice * 7 / 100
  discountedPrice + tax

theorem lowest_possible_price : 
  min (finalPrice (typeADiscountedPrice 4500)) 
      (min (finalPrice (typeBDiscountedPrice 5500)) 
           (finalPrice (typeCDiscountedPrice 5000))) = 3274 :=
by {
  sorry
}

end lowest_possible_price_l661_66136


namespace imo_hosting_arrangements_l661_66192

structure IMOCompetition where
  countries : Finset String
  continents : Finset String
  assignments : Finset (String × String)
  constraints : String → String
  assignments_must_be_unique : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                 (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                 constraints c1 ≠ constraints c2 → c1 ≠ c2
  no_consecutive_same_continent : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                   (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                   (c1, cnt1) ≠ (c2, cnt2) →
                                   constraints c1 ≠ constraints c2

def number_of_valid_arrangements (comp: IMOCompetition) : Nat := 240

theorem imo_hosting_arrangements (comp : IMOCompetition) :
  number_of_valid_arrangements comp = 240 := by
  sorry

end imo_hosting_arrangements_l661_66192


namespace complex_magnitude_l661_66154

open Complex

theorem complex_magnitude {x y : ℝ} (h : (1 + Complex.I) * x = 1 + y * Complex.I) : abs (x + y * Complex.I) = Real.sqrt 2 :=
sorry

end complex_magnitude_l661_66154


namespace journey_time_difference_l661_66118

theorem journey_time_difference :
  let speed := 40  -- mph
  let distance1 := 360  -- miles
  let distance2 := 320  -- miles
  (distance1 / speed - distance2 / speed) * 60 = 60 := 
by
  sorry

end journey_time_difference_l661_66118


namespace circle_sine_intersection_l661_66133

theorem circle_sine_intersection (h k r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n > 16 ∧
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, (x - h)^2 + (2 * Real.sin x - k)^2 = r^2) ∧ xs.card = n :=
by
  sorry

end circle_sine_intersection_l661_66133


namespace conic_sections_ab_value_l661_66172

theorem conic_sections_ab_value
  (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
by
  -- Proof will be filled in later
  sorry

end conic_sections_ab_value_l661_66172


namespace altitude_angle_bisector_inequality_l661_66152

theorem altitude_angle_bisector_inequality
  (h l R r : ℝ) 
  (triangle_condition : ∀ (h l : ℝ) (R r : ℝ), (h > 0 ∧ l > 0 ∧ R > 0 ∧ r > 0)) :
  h / l ≥ Real.sqrt (2 * r / R) :=
by
  sorry

end altitude_angle_bisector_inequality_l661_66152


namespace find_fraction_l661_66148

noncomputable def some_fraction_of_number_is (N f : ℝ) : Prop :=
  1 + f * N = 0.75 * N

theorem find_fraction (N : ℝ) (hN : N = 12.0) :
  ∃ f : ℝ, some_fraction_of_number_is N f ∧ f = 2 / 3 :=
by
  sorry

end find_fraction_l661_66148
