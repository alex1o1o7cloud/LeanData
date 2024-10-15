import Mathlib

namespace NUMINAMATH_GPT_Ashok_took_six_subjects_l1652_165208

theorem Ashok_took_six_subjects
  (n : ℕ) -- number of subjects Ashok took
  (T : ℕ) -- total marks secured in those subjects
  (h_avg_n : T = n * 72) -- condition: average of marks in n subjects is 72
  (h_avg_5 : 5 * 74 = 370) -- condition: average of marks in 5 subjects is 74
  (h_6th_mark : 62 > 0) -- condition: the 6th subject's mark is 62
  (h_T : T = 370 + 62) -- condition: total marks including the 6th subject
  : n = 6 := 
sorry


end NUMINAMATH_GPT_Ashok_took_six_subjects_l1652_165208


namespace NUMINAMATH_GPT_least_number_four_digits_divisible_by_15_25_40_75_l1652_165215

noncomputable def least_four_digit_multiple : ℕ :=
  1200

theorem least_number_four_digits_divisible_by_15_25_40_75 :
  (∀ n, (n ∣ 15) ∧ (n ∣ 25) ∧ (n ∣ 40) ∧ (n ∣ 75)) → least_four_digit_multiple = 1200 :=
sorry

end NUMINAMATH_GPT_least_number_four_digits_divisible_by_15_25_40_75_l1652_165215


namespace NUMINAMATH_GPT_fireworks_display_l1652_165276

-- Define numbers and conditions
def display_fireworks_for_number (n : ℕ) : ℕ := 6
def display_fireworks_for_letter (c : Char) : ℕ := 5
def fireworks_per_box : ℕ := 8
def number_boxes : ℕ := 50

-- Calculate fireworks for the year 2023
def fireworks_for_year : ℕ :=
  display_fireworks_for_number 2 * 2 +
  display_fireworks_for_number 0 * 1 +
  display_fireworks_for_number 3 * 1

-- Calculate fireworks for "HAPPY NEW YEAR"
def fireworks_for_phrase : ℕ :=
  12 * display_fireworks_for_letter 'H'

-- Calculate fireworks for 50 boxes
def fireworks_for_boxes : ℕ := number_boxes * fireworks_per_box

-- Total fireworks calculation
def total_fireworks : ℕ := fireworks_for_year + fireworks_for_phrase + fireworks_for_boxes

-- Proof statement
theorem fireworks_display : total_fireworks = 476 := 
  by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_fireworks_display_l1652_165276


namespace NUMINAMATH_GPT_find_divided_number_l1652_165299

theorem find_divided_number :
  ∃ (Number : ℕ), ∃ (q r d : ℕ), q = 8 ∧ r = 3 ∧ d = 21 ∧ Number = d * q + r ∧ Number = 171 :=
by
  sorry

end NUMINAMATH_GPT_find_divided_number_l1652_165299


namespace NUMINAMATH_GPT_difference_of_cats_l1652_165288

-- Definitions based on given conditions
def number_of_cats_sheridan : ℕ := 11
def number_of_cats_garrett : ℕ := 24

-- Theorem statement (proof problem) based on the question and correct answer
theorem difference_of_cats : (number_of_cats_garrett - number_of_cats_sheridan) = 13 := by
  sorry

end NUMINAMATH_GPT_difference_of_cats_l1652_165288


namespace NUMINAMATH_GPT_complex_number_solution_l1652_165202

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1652_165202


namespace NUMINAMATH_GPT_games_given_to_neil_is_five_l1652_165265

variable (x : ℕ)

def initial_games_henry : ℕ := 33
def initial_games_neil : ℕ := 2
def games_given_to_neil : ℕ := x

theorem games_given_to_neil_is_five
  (H : initial_games_henry - games_given_to_neil = 4 * (initial_games_neil + games_given_to_neil)) :
  games_given_to_neil = 5 := by
  sorry

end NUMINAMATH_GPT_games_given_to_neil_is_five_l1652_165265


namespace NUMINAMATH_GPT_find_abs_xyz_l1652_165258

noncomputable def conditions_and_question (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1)

theorem find_abs_xyz (x y z : ℝ) (h : conditions_and_question x y z) : |x * y * z| = 1 :=
  sorry

end NUMINAMATH_GPT_find_abs_xyz_l1652_165258


namespace NUMINAMATH_GPT_possible_values_of_a_l1652_165262

def setA := {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | 2 * a - x > 1}
def complementB (a : ℝ) := {x : ℝ | x ≥ (2 * a - 1)}

theorem possible_values_of_a (a : ℝ) :
  (∀ x, x ∈ setA → x ∈ complementB a) ↔ (a = -2 ∨ a = 0 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l1652_165262


namespace NUMINAMATH_GPT_fraction_incorrect_like_music_l1652_165259

-- Define the conditions as given in the problem
def total_students : ℕ := 100
def like_music_percentage : ℝ := 0.7
def dislike_music_percentage : ℝ := 1 - like_music_percentage

def correct_like_percentage : ℝ := 0.75
def incorrect_like_percentage : ℝ := 1 - correct_like_percentage

def correct_dislike_percentage : ℝ := 0.85
def incorrect_dislike_percentage : ℝ := 1 - correct_dislike_percentage

-- The number of students liking music
def like_music_students : ℝ := total_students * like_music_percentage
-- The number of students disliking music
def dislike_music_students : ℝ := total_students * dislike_music_percentage

-- The number of students who correctly say they like music
def correct_like_music_say : ℝ := like_music_students * correct_like_percentage
-- The number of students who incorrectly say they dislike music
def incorrect_dislike_music_say : ℝ := like_music_students * incorrect_like_percentage

-- The number of students who correctly say they dislike music
def correct_dislike_music_say : ℝ := dislike_music_students * correct_dislike_percentage
-- The number of students who incorrectly say they like music
def incorrect_like_music_say : ℝ := dislike_music_students * incorrect_dislike_percentage

-- The total number of students who say they like music
def total_say_like_music : ℝ := correct_like_music_say + incorrect_like_music_say

-- The final theorem we want to prove
theorem fraction_incorrect_like_music : ((incorrect_like_music_say : ℝ) / total_say_like_music) = (5 / 58) :=
by
  -- here we would provide the proof, but for now, we use sorry
  sorry

end NUMINAMATH_GPT_fraction_incorrect_like_music_l1652_165259


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l1652_165214

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), (0 ≤ a ∧ a < 5) ∧ (0 ≤ b ∧ b < 13) ∧ (1 / 2015 = a / 5 + b / 13 + c / 31) ∧ (a + b = 14) :=
sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l1652_165214


namespace NUMINAMATH_GPT_cucumbers_after_purchase_l1652_165281

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end NUMINAMATH_GPT_cucumbers_after_purchase_l1652_165281


namespace NUMINAMATH_GPT_increase_in_area_correct_l1652_165291

-- Define the dimensions of the original rectangular garden
def length_rect := 60
def width_rect := 20

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Calculate the side length of the square garden using the same perimeter.
def side_square := perimeter_rect / 4

-- Define the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Define the area of the square garden
def area_square := side_square * side_square

-- Define the increase in area after reshaping
def increase_in_area := area_square - area_rect

-- Prove that the increase in the area is 400 square feet
theorem increase_in_area_correct : increase_in_area = 400 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_increase_in_area_correct_l1652_165291


namespace NUMINAMATH_GPT_simplify_fraction_l1652_165223

theorem simplify_fraction (x : ℝ) (hx : x ≠ 1) : (x^2 / (x-1)) - (1 / (x-1)) = x + 1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1652_165223


namespace NUMINAMATH_GPT_relationship_between_a_b_c_d_l1652_165269

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x)
noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.sin x)

open Real

theorem relationship_between_a_b_c_d :
  ∀ (x : ℝ) (a b c d : ℝ),
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, f x ≤ a ∧ b ≤ f x) →
  (∀ x, g x ≤ c ∧ d ≤ g x) →
  a = sin 1 →
  b = -sin 1 →
  c = 1 →
  d = cos 1 →
  b < d ∧ d < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_d_l1652_165269


namespace NUMINAMATH_GPT_compare_values_of_even_and_monotone_function_l1652_165253

variable (f : ℝ → ℝ)

def is_even_function := ∀ x : ℝ, f x = f (-x)
def is_monotone_increasing_on_nonneg := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem compare_values_of_even_and_monotone_function
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  sorry

end NUMINAMATH_GPT_compare_values_of_even_and_monotone_function_l1652_165253


namespace NUMINAMATH_GPT_perfect_square_condition_l1652_165266

theorem perfect_square_condition (m : ℤ) : 
  (∃ k : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = k^2) ↔ m = 196 :=
by sorry

end NUMINAMATH_GPT_perfect_square_condition_l1652_165266


namespace NUMINAMATH_GPT_solveEquation1_proof_solveEquation2_proof_l1652_165206

noncomputable def solveEquation1 : Set ℝ :=
  { x | 2 * x^2 - 5 * x = 0 }

theorem solveEquation1_proof :
  solveEquation1 = { 0, (5 / 2 : ℝ) } :=
by
  sorry

noncomputable def solveEquation2 : Set ℝ :=
  { x | x^2 + 3 * x - 3 = 0 }

theorem solveEquation2_proof :
  solveEquation2 = { ( (-3 + Real.sqrt 21) / 2 : ℝ ), ( (-3 - Real.sqrt 21) / 2 : ℝ ) } :=
by
  sorry

end NUMINAMATH_GPT_solveEquation1_proof_solveEquation2_proof_l1652_165206


namespace NUMINAMATH_GPT_new_solution_percentage_l1652_165233

theorem new_solution_percentage 
  (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution_weight : ℝ) 
  (percentage_X : ℝ) (percentage_water : ℝ)
  (total_initial_X : ℝ := initial_weight * percentage_X)
  (initial_water : ℝ := initial_weight * percentage_water)
  (post_evaporation_weight : ℝ := initial_weight - evaporated_water)
  (post_evaporation_X : ℝ := total_initial_X)
  (post_evaporation_water : ℝ := post_evaporation_weight - total_initial_X)
  (added_X : ℝ := added_solution_weight * percentage_X)
  (added_water : ℝ := added_solution_weight * percentage_water)
  (total_X : ℝ := post_evaporation_X + added_X)
  (total_water : ℝ := post_evaporation_water + added_water)
  (new_total_weight : ℝ := post_evaporation_weight + added_solution_weight) :
  (total_X / new_total_weight) * 100 = 41.25 := 
by {
  sorry
}

end NUMINAMATH_GPT_new_solution_percentage_l1652_165233


namespace NUMINAMATH_GPT_car_drive_highway_distance_l1652_165298

theorem car_drive_highway_distance
  (d_local : ℝ)
  (s_local : ℝ)
  (s_highway : ℝ)
  (s_avg : ℝ)
  (d_total := d_local + s_avg * (d_local / s_local + d_local / s_highway))
  (t_local := d_local / s_local)
  (t_highway : ℝ := (d_total - d_local) / s_highway)
  (t_total := t_local + t_highway)
  (avg_speed := (d_total) / t_total)
  : d_local = 60 → s_local = 20 → s_highway = 60 → s_avg = 36 → avg_speed = 36 → d_total - d_local = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_car_drive_highway_distance_l1652_165298


namespace NUMINAMATH_GPT_money_given_to_each_friend_l1652_165207

-- Define the conditions
def initial_amount : ℝ := 20.10
def money_spent_on_sweets : ℝ := 1.05
def amount_left : ℝ := 17.05
def number_of_friends : ℝ := 2.0

-- Theorem statement
theorem money_given_to_each_friend :
  (initial_amount - amount_left - money_spent_on_sweets) / number_of_friends = 1.00 :=
by
  sorry

end NUMINAMATH_GPT_money_given_to_each_friend_l1652_165207


namespace NUMINAMATH_GPT_prob_a_prob_b_l1652_165237

-- Given conditions and question for Part a
def election_prob (p q : ℕ) (h : p > q) : ℚ :=
  (p - q) / (p + q)

theorem prob_a : election_prob 3 2 (by decide) = 1 / 5 :=
  sorry

-- Given conditions and question for Part b
theorem prob_b : election_prob 1010 1009 (by decide) = 1 / 2019 :=
  sorry

end NUMINAMATH_GPT_prob_a_prob_b_l1652_165237


namespace NUMINAMATH_GPT_tan_pi_over_4_plus_alpha_l1652_165293

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_tan_pi_over_4_plus_alpha_l1652_165293


namespace NUMINAMATH_GPT_sample_size_120_l1652_165234

theorem sample_size_120
  (x y : ℕ)
  (h_ratio : x / 2 = y / 3 ∧ y / 3 = 60 / 5)
  (h_max : max x (max y 60) = 60) :
  x + y + 60 = 120 := by
  sorry

end NUMINAMATH_GPT_sample_size_120_l1652_165234


namespace NUMINAMATH_GPT_continuous_arrow_loop_encircling_rectangle_l1652_165270

def total_orientations : ℕ := 2^4

def favorable_orientations : ℕ := 2 * 2

def probability_loop : ℚ := favorable_orientations / total_orientations

theorem continuous_arrow_loop_encircling_rectangle : probability_loop = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_continuous_arrow_loop_encircling_rectangle_l1652_165270


namespace NUMINAMATH_GPT_find_power_l1652_165213

theorem find_power (a b c d e : ℕ) (h1 : a = 105) (h2 : b = 21) (h3 : c = 25) (h4 : d = 45) (h5 : e = 49) 
(h6 : a ^ (3 : ℕ) = b * c * d * e) : 3 = 3 := by
  sorry

end NUMINAMATH_GPT_find_power_l1652_165213


namespace NUMINAMATH_GPT_solve_inequality_l1652_165212

-- Define the inequality as a function
def inequality_holds (x : ℝ) : Prop :=
  (2 * x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10)

-- Define the solution set as intervals excluding the points
def solution_set (x : ℝ) : Prop :=
  x < -5 / 2 ∨ x > -2

theorem solve_inequality (x : ℝ) : inequality_holds x ↔ solution_set x :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l1652_165212


namespace NUMINAMATH_GPT_integer_solution_for_system_l1652_165249

theorem integer_solution_for_system 
    (x y z : ℕ) 
    (h1 : 3 * x - 4 * y + 5 * z = 10) 
    (h2 : 7 * y + 8 * x - 3 * z = 13) : 
    x = 1 ∧ y = 2 ∧ z = 3 :=
by 
  sorry

end NUMINAMATH_GPT_integer_solution_for_system_l1652_165249


namespace NUMINAMATH_GPT_rain_on_Tuesday_correct_l1652_165264

-- Let the amount of rain on Monday be represented by m
def rain_on_Monday : ℝ := 0.9

-- Let the difference in rain between Monday and Tuesday be represented by d
def rain_difference : ℝ := 0.7

-- Define the calculated amount of rain on Tuesday
def rain_on_Tuesday : ℝ := rain_on_Monday - rain_difference

-- The statement we need to prove
theorem rain_on_Tuesday_correct : rain_on_Tuesday = 0.2 := 
by
  -- Proof omitted (to be provided)
  sorry

end NUMINAMATH_GPT_rain_on_Tuesday_correct_l1652_165264


namespace NUMINAMATH_GPT_inequality_pos_real_l1652_165267

theorem inequality_pos_real (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ (2 / 3) := 
sorry

end NUMINAMATH_GPT_inequality_pos_real_l1652_165267


namespace NUMINAMATH_GPT_fraction_C_D_l1652_165236

noncomputable def C : ℝ := ∑' n, if n % 6 = 0 then 0 else if n % 2 = 0 then ((-1)^(n/2 + 1) / (↑n^2)) else 0
noncomputable def D : ℝ := ∑' n, if n % 6 = 0 then ((-1)^(n/6 + 1) / (↑n^2)) else 0

theorem fraction_C_D : C / D = 37 := sorry

end NUMINAMATH_GPT_fraction_C_D_l1652_165236


namespace NUMINAMATH_GPT_harmonic_mean_lcm_gcd_sum_l1652_165211

theorem harmonic_mean_lcm_gcd_sum {m n : ℕ} (h_lcm : Nat.lcm m n = 210) (h_gcd : Nat.gcd m n = 6) (h_sum : m + n = 72) :
  (1 / (m : ℚ) + 1 / (n : ℚ)) = 2 / 35 := 
sorry

end NUMINAMATH_GPT_harmonic_mean_lcm_gcd_sum_l1652_165211


namespace NUMINAMATH_GPT_math_problem_l1652_165277

noncomputable def problem_statement : Prop :=
  let A : ℝ × ℝ := (5, 6)
  let B : ℝ × ℝ := (8, 3)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  let y_intercept : ℝ := A.snd - slope * A.fst
  slope + y_intercept = 10

theorem math_problem : problem_statement := sorry

end NUMINAMATH_GPT_math_problem_l1652_165277


namespace NUMINAMATH_GPT_worm_length_difference_is_correct_l1652_165252

-- Define the lengths of the worms
def worm1_length : ℝ := 0.8
def worm2_length : ℝ := 0.1

-- Define the difference in length between the longer worm and the shorter worm
def length_difference (a b : ℝ) : ℝ := a - b

-- State the theorem that the length difference is 0.7 inches
theorem worm_length_difference_is_correct (h1 : worm1_length = 0.8) (h2 : worm2_length = 0.1) :
  length_difference worm1_length worm2_length = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_worm_length_difference_is_correct_l1652_165252


namespace NUMINAMATH_GPT_solveAdultsMonday_l1652_165203

def numAdultsMonday (A : ℕ) : Prop :=
  let childrenMondayCost := 7 * 3
  let childrenTuesdayCost := 4 * 3
  let adultsTuesdayCost := 2 * 4
  let totalChildrenCost := childrenMondayCost + childrenTuesdayCost
  let totalAdultsCost := A * 4 + adultsTuesdayCost
  let totalRevenue := totalChildrenCost + totalAdultsCost
  totalRevenue = 61

theorem solveAdultsMonday : numAdultsMonday 5 := 
  by 
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_solveAdultsMonday_l1652_165203


namespace NUMINAMATH_GPT_years_of_interest_l1652_165239

noncomputable def principal : ℝ := 2600
noncomputable def interest_difference : ℝ := 78

theorem years_of_interest (R : ℝ) (N : ℝ) (h : (principal * (R + 1) * N / 100) - (principal * R * N / 100) = interest_difference) : N = 3 :=
sorry

end NUMINAMATH_GPT_years_of_interest_l1652_165239


namespace NUMINAMATH_GPT_solution_mod_5_l1652_165238

theorem solution_mod_5 (a : ℤ) : 
  (a^3 + 3 * a + 1) % 5 = 0 ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  sorry

end NUMINAMATH_GPT_solution_mod_5_l1652_165238


namespace NUMINAMATH_GPT_cost_of_five_dozens_l1652_165255

-- Define cost per dozen given the total cost for two dozen
noncomputable def cost_per_dozen : ℝ := 15.60 / 2

-- Define the number of dozen apples we want to calculate the cost for
def number_of_dozens := 5

-- Define the total cost for the given number of dozens
noncomputable def total_cost (n : ℕ) : ℝ := n * cost_per_dozen

-- State the theorem
theorem cost_of_five_dozens : total_cost number_of_dozens = 39 :=
by
  unfold total_cost cost_per_dozen
  sorry

end NUMINAMATH_GPT_cost_of_five_dozens_l1652_165255


namespace NUMINAMATH_GPT_evaluate_expression_at_3_l1652_165295

theorem evaluate_expression_at_3 : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_3_l1652_165295


namespace NUMINAMATH_GPT_sally_lost_two_balloons_l1652_165250

-- Condition: Sally originally had 9 orange balloons.
def original_orange_balloons := 9

-- Condition: Sally now has 7 orange balloons.
def current_orange_balloons := 7

-- Problem: Prove that Sally lost 2 orange balloons.
theorem sally_lost_two_balloons : original_orange_balloons - current_orange_balloons = 2 := by
  sorry

end NUMINAMATH_GPT_sally_lost_two_balloons_l1652_165250


namespace NUMINAMATH_GPT_fraction_product_l1652_165232

theorem fraction_product : 
  (7 / 5) * (8 / 16) * (21 / 15) * (14 / 28) * (35 / 25) * (20 / 40) * (49 / 35) * (32 / 64) = 2401 / 10000 :=
by
  -- This line is to skip the proof
  sorry

end NUMINAMATH_GPT_fraction_product_l1652_165232


namespace NUMINAMATH_GPT_determine_a_l1652_165290

theorem determine_a (x y a : ℝ) 
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) : 
  a = 0 := 
sorry

end NUMINAMATH_GPT_determine_a_l1652_165290


namespace NUMINAMATH_GPT_count_quadruples_l1652_165222

open Real

theorem count_quadruples:
  ∃ qs : Finset (ℝ × ℝ × ℝ × ℝ),
  (∀ (a b c k : ℝ), (a, b, c, k) ∈ qs ↔ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a^k = b * c ∧
    b^k = c * a ∧
    c^k = a * b
  ) ∧
  qs.card = 8 :=
sorry

end NUMINAMATH_GPT_count_quadruples_l1652_165222


namespace NUMINAMATH_GPT_square_of_number_ending_in_5_l1652_165247

theorem square_of_number_ending_in_5 (a : ℤ) :
  (10 * a + 5) * (10 * a + 5) = 100 * a * (a + 1) + 25 := by
  sorry

end NUMINAMATH_GPT_square_of_number_ending_in_5_l1652_165247


namespace NUMINAMATH_GPT_fraction_ratio_l1652_165297

variable (M Q P N R : ℝ)

theorem fraction_ratio (h1 : M = 0.40 * Q)
                       (h2 : Q = 0.25 * P)
                       (h3 : N = 0.40 * R)
                       (h4 : R = 0.75 * P) :
  M / N = 1 / 3 := 
by
  -- proof steps can be provided here
  sorry

end NUMINAMATH_GPT_fraction_ratio_l1652_165297


namespace NUMINAMATH_GPT_bus_problem_l1652_165286

theorem bus_problem (x : ℕ)
  (h1 : 28 + 82 - x = 30) :
  82 - x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_bus_problem_l1652_165286


namespace NUMINAMATH_GPT_clowns_to_guppies_ratio_l1652_165283

theorem clowns_to_guppies_ratio
  (C : ℕ)
  (tetra : ℕ)
  (guppies : ℕ)
  (total_animals : ℕ)
  (h1 : tetra = 4 * C)
  (h2 : guppies = 30)
  (h3 : total_animals = 330)
  (h4 : total_animals = tetra + C + guppies) :
  C / guppies = 2 :=
by
  sorry

end NUMINAMATH_GPT_clowns_to_guppies_ratio_l1652_165283


namespace NUMINAMATH_GPT_find_angle_A_l1652_165254

open Real

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = sqrt 2) 
  (hb : b = 2) 
  (hB : sin B + cos B = sqrt 2) :
  A = π / 6 := 
  sorry

end NUMINAMATH_GPT_find_angle_A_l1652_165254


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1652_165248

open Real

theorem solve_equation_1 (x : ℝ) (h_ne1 : x + 1 ≠ 0) (h_ne2 : x - 3 ≠ 0) : 
  (5 / (x + 1) = 1 / (x - 3)) → x = 4 :=
by
    intro h
    sorry

theorem solve_equation_2 (x : ℝ) (h_ne1 : x - 4 ≠ 0) (h_ne2 : 4 - x ≠ 0) :
    (3 - x) / (x - 4) = 1 / (4 - x) - 2 → False :=
by
    intro h
    sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1652_165248


namespace NUMINAMATH_GPT_dust_particles_calculation_l1652_165205

theorem dust_particles_calculation (D : ℕ) (swept : ℝ) (left_by_shoes : ℕ) (total_after_walk : ℕ)  
  (h_swept : swept = 9 / 10)
  (h_left_by_shoes : left_by_shoes = 223)
  (h_total_after_walk : total_after_walk = 331)
  (h_equation : (1 - swept) * D + left_by_shoes = total_after_walk) : 
  D = 1080 := 
by
  sorry

end NUMINAMATH_GPT_dust_particles_calculation_l1652_165205


namespace NUMINAMATH_GPT_train_speed_l1652_165278

noncomputable def distance : ℝ := 45  -- 45 km
noncomputable def time_minutes : ℝ := 30  -- 30 minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert minutes to hours

theorem train_speed (d : ℝ) (t_m : ℝ) : d = 45 → t_m = 30 → d / (t_m / 60) = 90 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_train_speed_l1652_165278


namespace NUMINAMATH_GPT_cosine_inequality_l1652_165224

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y :=
sorry

end NUMINAMATH_GPT_cosine_inequality_l1652_165224


namespace NUMINAMATH_GPT_nominal_rate_of_interest_correct_l1652_165245

noncomputable def nominal_rate_of_interest (EAR : ℝ) (n : ℕ) : ℝ :=
  let i := by 
    sorry
  i

theorem nominal_rate_of_interest_correct :
  nominal_rate_of_interest 0.0609 2 = 0.0598 :=
by 
  sorry

end NUMINAMATH_GPT_nominal_rate_of_interest_correct_l1652_165245


namespace NUMINAMATH_GPT_horner_eval_hex_to_decimal_l1652_165200

-- Problem 1: Evaluate the polynomial using Horner's method
theorem horner_eval (x : ℤ) (f : ℤ → ℤ) (v3 : ℤ) :
  (f x = 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12) →
  x = -4 →
  v3 = (((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12 →
  v3 = -57 :=
by
  intros hf hx hv
  sorry

-- Problem 2: Convert hexadecimal base-6 to decimal
theorem hex_to_decimal (hex : ℕ) (dec : ℕ) :
  hex = 210 →
  dec = 0 * 6^0 + 1 * 6^1 + 2 * 6^2 →
  dec = 78 :=
by
  intros hhex hdec
  sorry

end NUMINAMATH_GPT_horner_eval_hex_to_decimal_l1652_165200


namespace NUMINAMATH_GPT_modified_cube_surface_area_l1652_165242

noncomputable def total_surface_area_modified_cube : ℝ :=
  let side_length := 10
  let triangle_side := 7 * Real.sqrt 2
  let tunnel_wall_area := 3 * (Real.sqrt 3 / 4 * triangle_side^2)
  let original_surface_area := 6 * side_length^2
  original_surface_area + tunnel_wall_area

theorem modified_cube_surface_area : 
  total_surface_area_modified_cube = 600 + 73.5 * Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_modified_cube_surface_area_l1652_165242


namespace NUMINAMATH_GPT_avg_of_multiples_l1652_165272

theorem avg_of_multiples (n : ℝ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n + 6 * n + 7 * n + 8 * n + 9 * n + 10 * n) / 10 = 60.5) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_multiples_l1652_165272


namespace NUMINAMATH_GPT_pieces_after_10_cuts_l1652_165225

-- Define the number of cuts
def cuts : ℕ := 10

-- Define the function that calculates the number of pieces
def pieces (k : ℕ) : ℕ := k + 1

-- State the theorem to prove the number of pieces given 10 cuts
theorem pieces_after_10_cuts : pieces cuts = 11 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pieces_after_10_cuts_l1652_165225


namespace NUMINAMATH_GPT_nancy_pensils_total_l1652_165260

theorem nancy_pensils_total
  (initial: ℕ) 
  (mult_factor: ℕ) 
  (add_pencils: ℕ) 
  (final_total: ℕ) 
  (h1: initial = 27)
  (h2: mult_factor = 4)
  (h3: add_pencils = 45):
  final_total = initial * mult_factor + add_pencils := 
by
  sorry

end NUMINAMATH_GPT_nancy_pensils_total_l1652_165260


namespace NUMINAMATH_GPT_total_amount_division_l1652_165217

variables (w x y z : ℝ)

theorem total_amount_division (h_w : w = 2)
                              (h_x : x = 0.75)
                              (h_y : y = 1.25)
                              (h_z : z = 0.85)
                              (h_share_y : y * Rs48_50 = Rs48_50) :
                              total_amount = 4.85 * 38.80 := sorry

end NUMINAMATH_GPT_total_amount_division_l1652_165217


namespace NUMINAMATH_GPT_correct_conclusions_l1652_165228

variable (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)

def f (x : ℝ) : ℝ := x^2

theorem correct_conclusions (h_distinct : x1 ≠ x2) :
  (f x1 * x2 = f x1 * f x2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f ((x1 + x2) / 2) < (f x1 + f x2) / 2) :=
by
  sorry

end NUMINAMATH_GPT_correct_conclusions_l1652_165228


namespace NUMINAMATH_GPT_find_selling_price_l1652_165230

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end NUMINAMATH_GPT_find_selling_price_l1652_165230


namespace NUMINAMATH_GPT_integers_in_range_of_f_l1652_165251

noncomputable def f (x : ℝ) := x^2 + x + 1/2

def count_integers_in_range (n : ℕ) : ℕ :=
  2 * (n + 1)

theorem integers_in_range_of_f (n : ℕ) :
  (count_integers_in_range n) = (2 * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_integers_in_range_of_f_l1652_165251


namespace NUMINAMATH_GPT_sum_of_integers_l1652_165244

theorem sum_of_integers:
  ∀ (m n p q : ℕ),
    m ≠ n → m ≠ p → m ≠ q → n ≠ p → n ≠ q → p ≠ q →
    (8 - m) * (8 - n) * (8 - p) * (8 - q) = 9 →
    m + n + p + q = 32 :=
by
  intros m n p q hmn hmp hmq hnp hnq hpq heq
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1652_165244


namespace NUMINAMATH_GPT_find_y_value_l1652_165219

-- Define the linear relationship
def linear_eq (k b x : ℝ) : ℝ := k * x + b

-- Given conditions
variables (k b : ℝ)
axiom h1 : linear_eq k b 0 = -1
axiom h2 : linear_eq k b (1/2) = 2

-- Prove that the value of y when x = -1/2 is -4
theorem find_y_value : linear_eq k b (-1/2) = -4 :=
by sorry

end NUMINAMATH_GPT_find_y_value_l1652_165219


namespace NUMINAMATH_GPT_cos_seven_pi_over_four_l1652_165209

theorem cos_seven_pi_over_four : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_four_l1652_165209


namespace NUMINAMATH_GPT_students_helped_on_third_day_l1652_165226

theorem students_helped_on_third_day (books_total : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day4 : ℕ) (books_day3 : ℕ) :
  books_total = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day4 = 9 →
  books_day3 = books_total - ((students_day1 + students_day2 + students_day4) * books_per_student) →
  books_day3 / books_per_student = 6 :=
by
  sorry

end NUMINAMATH_GPT_students_helped_on_third_day_l1652_165226


namespace NUMINAMATH_GPT_range_of_a_l1652_165285

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (a > (1 / 2))

theorem range_of_a (hpq_true: p a ∨ q a) (hpq_false: ¬ (p a ∧ q a)) :
  (0 < a ∧ a ≤ (1 / 2)) ∨ (a ≥ 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1652_165285


namespace NUMINAMATH_GPT_max_x_values_l1652_165240

noncomputable def y (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

theorem max_x_values :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} = {x : ℝ | y x = y (x)} :=
sorry

end NUMINAMATH_GPT_max_x_values_l1652_165240


namespace NUMINAMATH_GPT_wendy_albums_used_l1652_165243

def total_pictures : ℕ := 45
def pictures_in_one_album : ℕ := 27
def pictures_per_album : ℕ := 2

theorem wendy_albums_used :
  let remaining_pictures := total_pictures - pictures_in_one_album
  let albums_used := remaining_pictures / pictures_per_album
  albums_used = 9 :=
by
  sorry

end NUMINAMATH_GPT_wendy_albums_used_l1652_165243


namespace NUMINAMATH_GPT_trapezoid_bisector_segment_length_l1652_165282

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end NUMINAMATH_GPT_trapezoid_bisector_segment_length_l1652_165282


namespace NUMINAMATH_GPT_missing_fraction_l1652_165204

theorem missing_fraction (x : ℕ) (h1 : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_missing_fraction_l1652_165204


namespace NUMINAMATH_GPT_mass_of_man_l1652_165296

theorem mass_of_man (L B h ρ V m: ℝ) (boat_length: L = 3) (boat_breadth: B = 2) 
  (boat_sink_depth: h = 0.01) (water_density: ρ = 1000) 
  (displaced_volume: V = L * B * h) (displaced_mass: m = ρ * V): m = 60 := 
by 
  sorry

end NUMINAMATH_GPT_mass_of_man_l1652_165296


namespace NUMINAMATH_GPT_loan_amount_principal_l1652_165221

-- Definitions based on conditions
def rate_of_interest := 3
def time_period := 3
def simple_interest := 108

-- Question translated to Lean 4 statement
theorem loan_amount_principal : ∃ P, (simple_interest = (P * rate_of_interest * time_period) / 100) ∧ P = 1200 :=
sorry

end NUMINAMATH_GPT_loan_amount_principal_l1652_165221


namespace NUMINAMATH_GPT_find_point_B_l1652_165268

-- Definition of Point
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of conditions
def A : Point := ⟨1, 2⟩
def d : ℝ := 3
def AB_parallel_x (A B : Point) : Prop := A.y = B.y

theorem find_point_B (B : Point) (h_parallel : AB_parallel_x A B) (h_dist : abs (B.x - A.x) = d) :
  (B = ⟨4, 2⟩) ∨ (B = ⟨-2, 2⟩) :=
by
  sorry

end NUMINAMATH_GPT_find_point_B_l1652_165268


namespace NUMINAMATH_GPT_divisor_of_2n_when_remainder_is_two_l1652_165235

theorem divisor_of_2n_when_remainder_is_two (n : ℤ) (k : ℤ) : 
  (n = 22 * k + 12) → ∃ d : ℤ, d = 22 ∧ (2 * n) % d = 2 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_2n_when_remainder_is_two_l1652_165235


namespace NUMINAMATH_GPT_girls_came_in_classroom_l1652_165294

theorem girls_came_in_classroom (initial_boys initial_girls boys_left final_children girls_in_classroom : ℕ)
  (h1 : initial_boys = 5)
  (h2 : initial_girls = 4)
  (h3 : boys_left = 3)
  (h4 : final_children = 8)
  (h5 : girls_in_classroom = final_children - (initial_boys - boys_left)) :
  girls_in_classroom - initial_girls = 2 :=
by
  sorry

end NUMINAMATH_GPT_girls_came_in_classroom_l1652_165294


namespace NUMINAMATH_GPT_natural_number_x_l1652_165275

theorem natural_number_x (x : ℕ) (A : ℕ → ℕ) (h : 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2) : x = 4 :=
sorry

end NUMINAMATH_GPT_natural_number_x_l1652_165275


namespace NUMINAMATH_GPT_inequality_proof_l1652_165229

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1652_165229


namespace NUMINAMATH_GPT_min_value_inv_sum_l1652_165271

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end NUMINAMATH_GPT_min_value_inv_sum_l1652_165271


namespace NUMINAMATH_GPT_steel_mill_production_2010_l1652_165201

noncomputable def steel_mill_production (P : ℕ → ℕ) : Prop :=
  (P 1990 = 400000) ∧ (P 2000 = 500000) ∧ ∀ n, (P n) = (P (n-1)) + (500000 - 400000) / 10

theorem steel_mill_production_2010 (P : ℕ → ℕ) (h : steel_mill_production P) : P 2010 = 630000 :=
by
  sorry -- proof omitted

end NUMINAMATH_GPT_steel_mill_production_2010_l1652_165201


namespace NUMINAMATH_GPT_intersection_sets_l1652_165287

-- defining sets A and B
def A : Set ℤ := {-1, 2, 4}
def B : Set ℤ := {0, 2, 6}

-- the theorem to be proved
theorem intersection_sets:
  A ∩ B = {2} :=
sorry

end NUMINAMATH_GPT_intersection_sets_l1652_165287


namespace NUMINAMATH_GPT_max_value_expr_l1652_165220

theorem max_value_expr (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (hxyz : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (x - y + z) ≤ 2187 / 216 :=
sorry

end NUMINAMATH_GPT_max_value_expr_l1652_165220


namespace NUMINAMATH_GPT_years_before_marriage_l1652_165216

theorem years_before_marriage {wedding_anniversary : ℕ} 
  (current_year : ℕ) (met_year : ℕ) (years_before_dating : ℕ) :
  wedding_anniversary = 20 →
  current_year = 2025 →
  met_year = 2000 →
  years_before_dating = 2 →
  met_year + years_before_dating + (current_year - met_year - wedding_anniversary) = current_year - wedding_anniversary - years_before_dating + wedding_anniversary - current_year :=
by
  sorry

end NUMINAMATH_GPT_years_before_marriage_l1652_165216


namespace NUMINAMATH_GPT_SWE4_l1652_165289

theorem SWE4 (a : ℕ → ℕ) (n : ℕ) :
  a 0 = 0 →
  (∀ n, a (n + 1) = 2 * a n + 2^n) →
  (∃ k : ℕ, n = 2^k) →
  ∃ m : ℕ, a n = 2^m :=
by
  intros h₀ h_recurrence h_power
  sorry

end NUMINAMATH_GPT_SWE4_l1652_165289


namespace NUMINAMATH_GPT_carol_weight_l1652_165241

variable (a c : ℝ)

theorem carol_weight (h1 : a + c = 240) (h2 : c - a = (2 / 3) * c) : c = 180 :=
by
  sorry

end NUMINAMATH_GPT_carol_weight_l1652_165241


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1652_165274

open Set Int

def A : Set ℝ := { x | x ^ 2 - 6 * x + 8 ≤ 0 }
def B : Set ℤ := { x | abs (x - 3) < 2 }

theorem intersection_of_A_and_B :
  (A ∩ (coe '' B) = { x : ℝ | x = 2 ∨ x = 3 ∨ x = 4 }) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1652_165274


namespace NUMINAMATH_GPT_shoe_length_size_15_l1652_165210

theorem shoe_length_size_15 : 
  ∀ (length : ℕ → ℝ), 
    (∀ n, 8 ≤ n ∧ n ≤ 17 → length (n + 1) = length n + 1 / 4) → 
    length 17 = (1 + 0.10) * length 8 →
    length 15 = 24.25 :=
by
  intro length h_increase h_largest
  sorry

end NUMINAMATH_GPT_shoe_length_size_15_l1652_165210


namespace NUMINAMATH_GPT_find_m_given_root_exists_l1652_165263

theorem find_m_given_root_exists (x m : ℝ) (h : ∃ x, x ≠ 2 ∧ (x / (x - 2) - 2 = m / (x - 2))) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_given_root_exists_l1652_165263


namespace NUMINAMATH_GPT_sum_div_minuend_eq_two_l1652_165256

variable (Subtrahend Minuend Difference : ℝ)

theorem sum_div_minuend_eq_two
  (h : Subtrahend + Difference = Minuend) :
  (Subtrahend + Minuend + Difference) / Minuend = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_div_minuend_eq_two_l1652_165256


namespace NUMINAMATH_GPT_shortest_ribbon_length_l1652_165246

theorem shortest_ribbon_length :
  ∃ (L : ℕ), (∀ (n : ℕ), n = 2 ∨ n = 5 ∨ n = 7 → L % n = 0) ∧ L = 70 :=
by
  sorry

end NUMINAMATH_GPT_shortest_ribbon_length_l1652_165246


namespace NUMINAMATH_GPT_circle_center_radius_sum_18_l1652_165273

-- Conditions from the problem statement
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * y - 9 = -y^2 + 18 * x + 9

-- Goal is to prove a + b + r = 18
theorem circle_center_radius_sum_18 :
  (∃ a b r : ℝ, 
     (∀ x y : ℝ, circle_eq x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
     a + b + r = 18) :=
sorry

end NUMINAMATH_GPT_circle_center_radius_sum_18_l1652_165273


namespace NUMINAMATH_GPT_workman_problem_l1652_165261

theorem workman_problem (x : ℝ) (h : (1 / x) + (1 / (2 * x)) = 1 / 32): x = 48 :=
sorry

end NUMINAMATH_GPT_workman_problem_l1652_165261


namespace NUMINAMATH_GPT_tree_planting_campaign_l1652_165257

theorem tree_planting_campaign
  (P : ℝ)
  (h1 : 456 = P * (1 - 1/20))
  (h2 : P ≥ 0)
  : (P * (1 + 0.1)) = (456 / (1 - 1/20) * 1.1) :=
by
  sorry

end NUMINAMATH_GPT_tree_planting_campaign_l1652_165257


namespace NUMINAMATH_GPT_find_certain_number_l1652_165231

theorem find_certain_number (x certain_number : ℤ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (certain_number + 62 + 98 + 124 + x) / 5 = 78) : 
  certain_number = 106 := 
by 
  sorry

end NUMINAMATH_GPT_find_certain_number_l1652_165231


namespace NUMINAMATH_GPT_susie_vacuums_each_room_in_20_minutes_l1652_165279

theorem susie_vacuums_each_room_in_20_minutes
  (total_time_hours : ℕ)
  (number_of_rooms : ℕ)
  (total_time_minutes : ℕ)
  (time_per_room : ℕ)
  (h1 : total_time_hours = 2)
  (h2 : number_of_rooms = 6)
  (h3 : total_time_minutes = total_time_hours * 60)
  (h4 : time_per_room = total_time_minutes / number_of_rooms) :
  time_per_room = 20 :=
by
  sorry

end NUMINAMATH_GPT_susie_vacuums_each_room_in_20_minutes_l1652_165279


namespace NUMINAMATH_GPT_pages_written_on_wednesday_l1652_165218

variable (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ)
variable (totalPages : ℕ)

def pagesOnMonday (minutesMonday rateMonday : ℕ) : ℕ :=
  minutesMonday / rateMonday

def pagesOnTuesday (minutesTuesday rateTuesday : ℕ) : ℕ :=
  minutesTuesday / rateTuesday

def totalPagesMondayAndTuesday (minutesMonday minutesTuesday rateMonday rateTuesday : ℕ) : ℕ :=
  pagesOnMonday minutesMonday rateMonday + pagesOnTuesday minutesTuesday rateTuesday

def pagesOnWednesday (minutesMonday minutesTuesday rateMonday rateTuesday totalPages : ℕ) : ℕ :=
  totalPages - totalPagesMondayAndTuesday minutesMonday minutesTuesday rateMonday rateTuesday

theorem pages_written_on_wednesday :
  pagesOnWednesday 60 45 30 15 10 = 5 := by
  sorry

end NUMINAMATH_GPT_pages_written_on_wednesday_l1652_165218


namespace NUMINAMATH_GPT_max_value_min_4x_y_4y_x2_5y2_l1652_165280

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_min_4x_y_4y_x2_5y2_l1652_165280


namespace NUMINAMATH_GPT_extra_birds_l1652_165284

def num_sparrows : ℕ := 10
def num_robins : ℕ := 5
def num_bluebirds : ℕ := 3
def nests_for_sparrows : ℕ := 4
def nests_for_robins : ℕ := 2
def nests_for_bluebirds : ℕ := 2

theorem extra_birds (num_sparrows : ℕ)
                    (num_robins : ℕ)
                    (num_bluebirds : ℕ)
                    (nests_for_sparrows : ℕ)
                    (nests_for_robins : ℕ)
                    (nests_for_bluebirds : ℕ) :
    num_sparrows = 10 ∧ 
    num_robins = 5 ∧ 
    num_bluebirds = 3 ∧ 
    nests_for_sparrows = 4 ∧ 
    nests_for_robins = 2 ∧ 
    nests_for_bluebirds = 2 ->
    num_sparrows - nests_for_sparrows = 6 ∧ 
    num_robins - nests_for_robins = 3 ∧ 
    num_bluebirds - nests_for_bluebirds = 1 :=
by sorry

end NUMINAMATH_GPT_extra_birds_l1652_165284


namespace NUMINAMATH_GPT_distinct_prime_factors_of_90_l1652_165227

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end NUMINAMATH_GPT_distinct_prime_factors_of_90_l1652_165227


namespace NUMINAMATH_GPT_matrix_zero_product_or_rank_one_l1652_165292

variables {n : ℕ}
variables (A B C : matrix (fin n) (fin n) ℝ)

theorem matrix_zero_product_or_rank_one
  (h1 : A * B * C = 0)
  (h2 : B.rank = 1) :
  A * B = 0 ∨ B * C = 0 :=
sorry

end NUMINAMATH_GPT_matrix_zero_product_or_rank_one_l1652_165292
