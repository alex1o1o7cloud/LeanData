import Mathlib

namespace recommended_water_intake_l136_13645

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end recommended_water_intake_l136_13645


namespace mike_planted_50_l136_13670

-- Definitions for conditions
def mike_morning (M : ℕ) := M
def ted_morning (M : ℕ) := 2 * M
def mike_afternoon := 60
def ted_afternoon := 40
def total_planted (M : ℕ) := mike_morning M + ted_morning M + mike_afternoon + ted_afternoon

-- Statement to prove
theorem mike_planted_50 (M : ℕ) (h : total_planted M = 250) : M = 50 :=
by
  sorry

end mike_planted_50_l136_13670


namespace sector_area_l136_13665

theorem sector_area
  (r : ℝ) (s : ℝ) (h_r : r = 1) (h_s : s = 1) : 
  (1 / 2) * r * s = 1 / 2 := by
  sorry

end sector_area_l136_13665


namespace quilt_shading_fraction_l136_13618

/-- 
Statement:
Given a quilt block made from nine unit squares, where two unit squares are divided diagonally into triangles, 
and one unit square is divided into four smaller equal squares with one of the smaller squares shaded, 
the fraction of the quilt that is shaded is \( \frac{5}{36} \).
-/
theorem quilt_shading_fraction : 
  let total_area := 9 
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2 
  shaded_area / total_area = 5 / 36 :=
by
  -- Definitions based on conditions
  let total_area := 9
  let shaded_area := 1 / 4 + 1 / 2 + 1 / 2
  -- The proof statement (fraction of shaded area)
  have h : shaded_area / total_area = 5 / 36 := sorry
  exact h

end quilt_shading_fraction_l136_13618


namespace value_of_a_l136_13678

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end value_of_a_l136_13678


namespace expected_coincidences_l136_13687

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l136_13687


namespace woman_working_days_l136_13636

-- Define the conditions
def man_work_rate := 1 / 6
def boy_work_rate := 1 / 18
def combined_work_rate := 1 / 4

-- Question statement in Lean 4
theorem woman_working_days :
  ∃ W : ℚ, (man_work_rate + W + boy_work_rate = combined_work_rate) ∧ (1 / W = 1296) :=
sorry

end woman_working_days_l136_13636


namespace solve_for_x_l136_13664

-- Define the conditions as mathematical statements in Lean
def conditions (x y : ℝ) : Prop :=
  (2 * x - 3 * y = 10) ∧ (y = -x)

-- State the theorem that needs to be proven
theorem solve_for_x : ∃ x : ℝ, ∃ y : ℝ, conditions x y ∧ x = 2 :=
by 
  -- Provide a sketch of the proof to show that the statement is well-formed
  sorry

end solve_for_x_l136_13664


namespace find_2alpha_minus_beta_l136_13627

theorem find_2alpha_minus_beta (α β : ℝ) (tan_diff : Real.tan (α - β) = 1 / 2) 
  (cos_β : Real.cos β = -7 * Real.sqrt 2 / 10) (α_range : 0 < α ∧ α < Real.pi) 
  (β_range : 0 < β ∧ β < Real.pi) : 2 * α - β = -3 * Real.pi / 4 :=
sorry

end find_2alpha_minus_beta_l136_13627


namespace optimal_optimism_coefficient_l136_13630

theorem optimal_optimism_coefficient (a b : ℝ) (x : ℝ) (h_b_gt_a : b > a) (h_x : 0 < x ∧ x < 1) 
  (h_c : ∀ (c : ℝ), c = a + x * (b - a) → (c - a) * (c - a) = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end optimal_optimism_coefficient_l136_13630


namespace unit_triangle_count_bound_l136_13646

variable {L : ℝ} (L_pos : L > 0)
variable {n : ℕ}

/--
  Let \( \Delta \) be an equilateral triangle with side length \( L \), and suppose that \( n \) unit 
  equilateral triangles are drawn inside \( \Delta \) with non-overlapping interiors and each having 
  sides parallel to \( \Delta \) but with opposite orientation. Then,
  we must have \( n \leq \frac{2}{3} L^2 \).
-/
theorem unit_triangle_count_bound (L_pos : L > 0) (n : ℕ) :
  n ≤ (2 / 3) * (L ^ 2) := 
sorry

end unit_triangle_count_bound_l136_13646


namespace fraction_meaningful_domain_l136_13661

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l136_13661


namespace soccer_ball_cost_l136_13692

theorem soccer_ball_cost (F S : ℝ) 
  (h1 : 3 * F + S = 155) 
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 := 
sorry

end soccer_ball_cost_l136_13692


namespace average_of_remaining_numbers_l136_13609

theorem average_of_remaining_numbers (S : ℕ) (h1 : S = 12 * 90) :
  ((S - 65 - 75 - 85) / 9) = 95 :=
by
  sorry

end average_of_remaining_numbers_l136_13609


namespace probability_male_female_ratio_l136_13640

theorem probability_male_female_ratio :
  let total_possibilities := Nat.choose 9 5
  let specific_scenarios := Nat.choose 5 2 * Nat.choose 4 3 + Nat.choose 5 3 * Nat.choose 4 2
  let probability := specific_scenarios / (total_possibilities : ℚ)
  probability = 50 / 63 :=
by 
  sorry

end probability_male_female_ratio_l136_13640


namespace cara_age_is_40_l136_13693

-- Defining the conditions
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Proving the question
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end cara_age_is_40_l136_13693


namespace max_value_expression_l136_13680

theorem max_value_expression (a b c : ℝ) (h : a * b * c + a + c - b = 0) : 
  ∃ m, (m = (1/(1+a^2) - 1/(1+b^2) + 1/(1+c^2))) ∧ (m = 5 / 4) :=
by 
  sorry

end max_value_expression_l136_13680


namespace geometric_sequence_product_l136_13673

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_product (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
  (h : a 3 = -1) : a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by
  sorry

end geometric_sequence_product_l136_13673


namespace intersection_of_A_and_B_l136_13621

theorem intersection_of_A_and_B :
  let A := {0, 1, 2, 3, 4}
  let B := {x | ∃ n ∈ A, x = 2 * n}
  A ∩ B = {0, 2, 4} :=
by
  sorry

end intersection_of_A_and_B_l136_13621


namespace total_packages_of_gum_l136_13688

theorem total_packages_of_gum (R_total R_extra R_per_package A_total A_extra A_per_package : ℕ) 
  (hR1 : R_total = 41) (hR2 : R_extra = 6) (hR3 : R_per_package = 7)
  (hA1 : A_total = 23) (hA2 : A_extra = 3) (hA3 : A_per_package = 5) :
  (R_total - R_extra) / R_per_package + (A_total - A_extra) / A_per_package = 9 :=
by
  sorry

end total_packages_of_gum_l136_13688


namespace find_m_l136_13674

theorem find_m (x : ℝ) (m : ℝ) (h1 : x > 2) (h2 : x - 3 * m + 1 > 0) : m = 1 :=
sorry

end find_m_l136_13674


namespace eval_sin_570_l136_13682

theorem eval_sin_570:
  2 * Real.sin (570 * Real.pi / 180) = -1 := 
by sorry

end eval_sin_570_l136_13682


namespace tony_average_time_l136_13695

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end tony_average_time_l136_13695


namespace bobby_shoes_l136_13697

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l136_13697


namespace elementary_sampling_count_l136_13696

theorem elementary_sampling_count :
  ∃ (a : ℕ), (a + (a + 600) + (a + 1200) = 3600) ∧
             (a = 600) ∧
             (a + 1200 = 1800) ∧
             (1800 * 1 / 100 = 18) :=
by {
  sorry
}

end elementary_sampling_count_l136_13696


namespace inscribed_square_area_l136_13655

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = (a ^ 2 + b ^ 2) ^ (1 / 2)

def square_area (s : ℝ) : ℝ := s * s

theorem inscribed_square_area
  (a b c : ℝ) (s₁ s₂ : ℝ)
  (ha : a = 16 * 2) -- Leg lengths equal to 2 * 16 cm
  (hb : b = 16 * 2)
  (hc : c = 32 * Real.sqrt 2) -- Hypotenuse of the triangle
  (hiso : isosceles_right_triangle a b c)
  (harea₁ : square_area 16 = 256) -- Given square area
  (hS : s₂ = 16 * Real.sqrt 2 - 8) -- Side length of the new square
  : square_area s₂ = 576 - 256 * Real.sqrt 2 := sorry

end inscribed_square_area_l136_13655


namespace king_total_payment_l136_13639

/-- 
A king gets a crown made that costs $20,000. He tips the person 10%. Prove that the total amount the king paid after the tip is $22,000.
-/
theorem king_total_payment (C : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h1 : C = 20000) 
  (h2 : tip_percentage = 0.1) 
  (h3 : total_paid = C + C * tip_percentage) : 
  total_paid = 22000 := 
by 
  sorry

end king_total_payment_l136_13639


namespace final_S_is_correct_l136_13605

/-- Define a function to compute the final value of S --/
def final_value_of_S : ℕ :=
  let S := 0
  let I_values := List.range' 1 27 3 -- generate list [1, 4, 7, ..., 28]
  I_values.foldl (fun S I => S + I) 0  -- compute the sum of the list

/-- Theorem stating the final value of S is 145 --/
theorem final_S_is_correct : final_value_of_S = 145 := by
  sorry

end final_S_is_correct_l136_13605


namespace range_of_a_for_empty_solution_set_l136_13654

theorem range_of_a_for_empty_solution_set : 
  (∀ a : ℝ, (∀ x : ℝ, |x - 4| + |3 - x| < a → false) ↔ a ≤ 1) := 
sorry

end range_of_a_for_empty_solution_set_l136_13654


namespace cos_squared_pi_over_4_minus_alpha_l136_13653

theorem cos_squared_pi_over_4_minus_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 3 / 4) :
  Real.cos (Real.pi / 4 - α) ^ 2 = 9 / 25 :=
by
  sorry

end cos_squared_pi_over_4_minus_alpha_l136_13653


namespace sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l136_13667

-- Define a convex n-gon and prove that the sum of its interior angles is (n-2) * 180 degrees
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (sum_of_angles : ℝ), sum_of_angles = (n-2) * 180 :=
sorry

-- Define a convex n-gon and prove that the number of triangles formed by dividing with non-intersecting diagonals is n-2
theorem number_of_triangles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (num_of_triangles : ℕ), num_of_triangles = n-2 :=
sorry

end sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l136_13667


namespace inverse_36_mod_53_l136_13669

theorem inverse_36_mod_53 (h : 17 * 26 ≡ 1 [MOD 53]) : 36 * 27 ≡ 1 [MOD 53] :=
sorry

end inverse_36_mod_53_l136_13669


namespace average_speed_l136_13616

-- Define the speeds and times
def speed1 : ℝ := 120 -- km/h
def time1 : ℝ := 1 -- hour

def speed2 : ℝ := 150 -- km/h
def time2 : ℝ := 2 -- hours

def speed3 : ℝ := 80 -- km/h
def time3 : ℝ := 0.5 -- hour

-- Define the conversion factor
def km_to_miles : ℝ := 0.62

-- Calculate total distance (in kilometers)
def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time2
def distance3 : ℝ := speed3 * time3

def total_distance_km : ℝ := distance1 + distance2 + distance3

-- Convert total distance to miles
def total_distance_miles : ℝ := total_distance_km * km_to_miles

-- Calculate total time (in hours)
def total_time : ℝ := time1 + time2 + time3

-- Final proof statement for average speed
theorem average_speed : total_distance_miles / total_time = 81.49 := by {
  sorry
}

end average_speed_l136_13616


namespace find_product_of_roots_l136_13683

noncomputable def equation (x : ℝ) : ℝ := (Real.sqrt 2023) * x^3 - 4047 * x^2 + 3

theorem find_product_of_roots (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) 
  (h3 : equation x1 = 0) (h4 : equation x2 = 0) (h5 : equation x3 = 0) :
  x2 * (x1 + x3) = 3 :=
by
  sorry

end find_product_of_roots_l136_13683


namespace y_equals_px_div_5x_p_l136_13691

variable (p x y : ℝ)

theorem y_equals_px_div_5x_p (h : p = 5 * x * y / (x - y)) : y = p * x / (5 * x + p) :=
sorry

end y_equals_px_div_5x_p_l136_13691


namespace tan_alpha_l136_13644

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 3/5) (h2 : Real.pi / 2 < α ∧ α < Real.pi) : Real.tan α = -3/4 := 
  sorry

end tan_alpha_l136_13644


namespace cylinder_lateral_surface_area_l136_13601

theorem cylinder_lateral_surface_area 
  (diameter height : ℝ) 
  (h1 : diameter = 2) 
  (h2 : height = 2) : 
  2 * Real.pi * (diameter / 2) * height = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l136_13601


namespace num_boys_l136_13635

-- Definitions as per the conditions
def boys (d : ℕ) := 2 * d
def reducedGirls (d : ℕ) := d - 1

-- Lean statement for the proof problem
theorem num_boys (d b : ℕ) 
  (h1 : b = boys d)
  (h2 : b = reducedGirls d + 8) : b = 14 :=
by {
  sorry
}

end num_boys_l136_13635


namespace weeks_saved_l136_13613

theorem weeks_saved (w : ℕ) :
  (10 * w / 2) - ((10 * w / 2) / 4) = 15 → 
  w = 4 := 
by
  sorry

end weeks_saved_l136_13613


namespace problem_DE_length_l136_13651

theorem problem_DE_length
  (AB AD : ℝ)
  (AB_eq : AB = 7)
  (AD_eq : AD = 10)
  (area_eq : 7 * CE = 140)
  (DC CE DE : ℝ)
  (DC_eq : DC = 7)
  (CE_eq : CE = 20)
  : DE = Real.sqrt 449 :=
by
  sorry

end problem_DE_length_l136_13651


namespace abs_diff_of_two_numbers_l136_13642

theorem abs_diff_of_two_numbers (x y : ℝ) (h_sum : x + y = 42) (h_prod : x * y = 437) : |x - y| = 4 :=
sorry

end abs_diff_of_two_numbers_l136_13642


namespace cost_of_four_books_l136_13619

theorem cost_of_four_books
  (H : 2 * book_cost = 36) :
  4 * book_cost = 72 :=
by
  sorry

end cost_of_four_books_l136_13619


namespace fraction_bounds_l136_13677

theorem fraction_bounds (x y : ℝ) (h : x^2 * y^2 + x * y + 1 = 3 * y^2) : 
0 ≤ (y - x) / (x + 4 * y) ∧ (y - x) / (x + 4 * y) ≤ 4 := 
sorry

end fraction_bounds_l136_13677


namespace a_plus_b_l136_13600

open Complex

theorem a_plus_b (a b : ℝ) (h : (a - I) * I = -b + 2 * I) : a + b = 1 := by
  sorry

end a_plus_b_l136_13600


namespace shares_difference_l136_13675

-- conditions: the ratio is 3:7:12, and the difference between q and r's share is Rs. 3000
theorem shares_difference (x : ℕ) (h : 12 * x - 7 * x = 3000) : 7 * x - 3 * x = 2400 :=
by
  -- simply skip the proof since it's not required in the prompt
  sorry

end shares_difference_l136_13675


namespace MN_eq_l136_13679

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}
def operation (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

theorem MN_eq : operation M N = {1, 4} :=
sorry

end MN_eq_l136_13679


namespace range_of_x_add_y_l136_13626

noncomputable def floor_not_exceeding (z : ℝ) : ℤ := ⌊z⌋

theorem range_of_x_add_y (x y : ℝ) (h1 : y = 3 * floor_not_exceeding x + 4) 
    (h2 : y = 4 * floor_not_exceeding (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) : 
    40 < x + y ∧ x + y < 41 :=
by 
  sorry 

end range_of_x_add_y_l136_13626


namespace pink_cookies_eq_fifty_l136_13666

-- Define the total number of cookies
def total_cookies : ℕ := 86

-- Define the number of red cookies
def red_cookies : ℕ := 36

-- The property we want to prove
theorem pink_cookies_eq_fifty : (total_cookies - red_cookies = 50) :=
by
  sorry

end pink_cookies_eq_fifty_l136_13666


namespace common_root_cubic_polynomials_l136_13617

open Real

theorem common_root_cubic_polynomials (a b c : ℝ)
  (h1 : ∃ α : ℝ, α^3 - a * α^2 + b = 0 ∧ α^3 - b * α^2 + c = 0)
  (h2 : ∃ β : ℝ, β^3 - b * β^2 + c = 0 ∧ β^3 - c * β^2 + a = 0)
  (h3 : ∃ γ : ℝ, γ^3 - c * γ^2 + a = 0 ∧ γ^3 - a * γ^2 + b = 0)
  : a = b ∧ b = c :=
sorry

end common_root_cubic_polynomials_l136_13617


namespace Thabo_books_l136_13634

theorem Thabo_books :
  ∃ (H : ℕ), ∃ (P : ℕ), ∃ (F : ℕ), 
  (H + P + F = 220) ∧ 
  (P = H + 20) ∧ 
  (F = 2 * P) ∧ 
  (H = 40) :=
by
  -- Here will be the formal proof, which is not required for this task.
  sorry

end Thabo_books_l136_13634


namespace circle_positions_n_l136_13652

theorem circle_positions_n (n : ℕ) (h1 : n ≥ 23) (h2 : (23 - 7) * 2 + 2 = n) : n = 32 :=
sorry

end circle_positions_n_l136_13652


namespace twenty_five_percent_of_five_hundred_l136_13648

theorem twenty_five_percent_of_five_hundred : 0.25 * 500 = 125 := 
by 
  sorry

end twenty_five_percent_of_five_hundred_l136_13648


namespace longer_side_of_rectangle_l136_13643

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
noncomputable def rectangle_area : ℝ := 3 * circle_area
noncomputable def shorter_side : ℝ := 2 * circle_radius

theorem longer_side_of_rectangle :
    ∃ (l : ℝ), l = rectangle_area / shorter_side ∧ l = 9 * Real.pi :=
by
  sorry

end longer_side_of_rectangle_l136_13643


namespace points_6_units_away_from_neg1_l136_13615

theorem points_6_units_away_from_neg1 (A : ℝ) (h : A = -1) :
  { x : ℝ | abs (x - A) = 6 } = { -7, 5 } :=
by
  sorry

end points_6_units_away_from_neg1_l136_13615


namespace maximum_distance_value_of_m_l136_13658

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := y = m * x - m - 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the problem statement
theorem maximum_distance_value_of_m :
  ∃ (m : ℝ), (∀ x y : ℝ, circle_eq x y → ∃ P : ℝ × ℝ, line_eq m P.fst P.snd) →
  m = -0.5 :=
sorry

end maximum_distance_value_of_m_l136_13658


namespace no_unique_p_l136_13623

-- Define the probabilities P_1 and P_2 given p
def P1 (p : ℝ) : ℝ := 3 * p^2 - 2 * p^3
def P2 (p : ℝ) : ℝ := 3 * p^2 - 3 * p^3

-- Define the expected value E(xi)
def E_xi (p : ℝ) : ℝ := P1 p + P2 p

-- Prove that there does not exist a unique p in (0, 1) such that E(xi) = 1.5
theorem no_unique_p (p : ℝ) (h : 0 < p ∧ p < 1) : E_xi p ≠ 1.5 := by
  sorry

end no_unique_p_l136_13623


namespace num_people_for_new_avg_l136_13629

def avg_salary := 430
def old_supervisor_salary := 870
def new_supervisor_salary := 870
def num_workers := 8
def total_people_before := num_workers + 1
def total_salary_before := total_people_before * avg_salary
def workers_salary := total_salary_before - old_supervisor_salary
def total_salary_after := workers_salary + new_supervisor_salary

theorem num_people_for_new_avg :
    ∃ (x : ℕ), x * avg_salary = total_salary_after ∧ x = 9 :=
by
  use 9
  field_simp
  sorry

end num_people_for_new_avg_l136_13629


namespace least_num_subtracted_l136_13699

theorem least_num_subtracted 
  {x : ℤ} 
  (h5 : (642 - x) % 5 = 4) 
  (h7 : (642 - x) % 7 = 4) 
  (h9 : (642 - x) % 9 = 4) : 
  x = 4 := 
sorry

end least_num_subtracted_l136_13699


namespace shaded_area_in_6x6_grid_l136_13637

def total_shaded_area (grid_size : ℕ) (triangle_squares : ℕ) (num_triangles : ℕ)
  (trapezoid_squares : ℕ) (num_trapezoids : ℕ) : ℕ :=
  (triangle_squares * num_triangles) + (trapezoid_squares * num_trapezoids)

theorem shaded_area_in_6x6_grid :
  total_shaded_area 6 2 2 3 4 = 16 :=
by
  -- Proof omitted for demonstration purposes
  sorry

end shaded_area_in_6x6_grid_l136_13637


namespace remaining_miles_to_be_built_l136_13685

-- Definitions from problem conditions
def current_length : ℕ := 200
def target_length : ℕ := 650
def first_day_miles : ℕ := 50
def second_day_miles : ℕ := 3 * first_day_miles

-- Lean theorem statement
theorem remaining_miles_to_be_built : 
  (target_length - current_length) - (first_day_miles + second_day_miles) = 250 := 
by 
  sorry

end remaining_miles_to_be_built_l136_13685


namespace tenth_term_is_correct_l136_13625

-- Define the conditions
def first_term : ℚ := 3
def last_term : ℚ := 88
def num_terms : ℕ := 30
def common_difference : ℚ := (last_term - first_term) / (num_terms - 1)

-- Define the function for the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℚ := first_term + (n - 1) * common_difference

-- Prove that the 10th term is 852/29
theorem tenth_term_is_correct : nth_term 10 = 852 / 29 := 
by 
  -- Add the proof later, the statement includes the setup and conditions
  sorry

end tenth_term_is_correct_l136_13625


namespace value_of_a_l136_13608

theorem value_of_a (a : ℝ) (h₁ : ∀ x : ℝ, (2 * x - (1/3) * a ≤ 0) → (x ≤ 2)) : a = 12 :=
sorry

end value_of_a_l136_13608


namespace arithmetic_seq_sum_mod_9_l136_13607

def sum_arithmetic_seq := 88230 + 88231 + 88232 + 88233 + 88234 + 88235 + 88236 + 88237 + 88238 + 88239 + 88240

theorem arithmetic_seq_sum_mod_9 : 
  sum_arithmetic_seq % 9 = 0 :=
by
-- proof will be provided here
sorry

end arithmetic_seq_sum_mod_9_l136_13607


namespace sequence_m_value_l136_13624

theorem sequence_m_value (m : ℕ) (a : ℕ → ℝ) (h₀ : a 0 = 37) (h₁ : a 1 = 72)
  (hm : a m = 0) (h_rec : ∀ k, 1 ≤ k ∧ k < m → a (k + 1) = a (k - 1) - 3 / a k) : m = 889 :=
sorry

end sequence_m_value_l136_13624


namespace sampling_methods_match_l136_13614

inductive SamplingMethod
| simple_random
| stratified
| systematic

open SamplingMethod

def commonly_used_sampling_methods : List SamplingMethod := 
  [simple_random, stratified, systematic]

def option_C : List SamplingMethod := 
  [simple_random, stratified, systematic]

theorem sampling_methods_match : commonly_used_sampling_methods = option_C := by
  sorry

end sampling_methods_match_l136_13614


namespace ted_age_solution_l136_13603

theorem ted_age_solution (t s : ℝ) (h1 : t = 3 * s - 10) (h2 : t + s = 60) : t = 42.5 :=
by {
  sorry
}

end ted_age_solution_l136_13603


namespace prime_divisor_exponent_l136_13662

theorem prime_divisor_exponent (a n : ℕ) (p : ℕ) 
    (ha : a ≥ 2)
    (hn : n ≥ 1) 
    (hp : Nat.Prime p) 
    (hdiv : p ∣ a^(2^n) + 1) :
    2^(n+1) ∣ (p-1) :=
by
  sorry

end prime_divisor_exponent_l136_13662


namespace arrangement_non_adjacent_l136_13668

theorem arrangement_non_adjacent :
  let total_arrangements := Nat.factorial 30
  let adjacent_arrangements := 2 * Nat.factorial 29
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements = 28 * Nat.factorial 29 :=
by
  sorry

end arrangement_non_adjacent_l136_13668


namespace frequency_of_rolling_six_is_0_point_19_l136_13681

theorem frequency_of_rolling_six_is_0_point_19 :
  ∀ (total_rolls number_six_appeared : ℕ), total_rolls = 100 → number_six_appeared = 19 → 
  (number_six_appeared : ℝ) / (total_rolls : ℝ) = 0.19 := 
by 
  intros total_rolls number_six_appeared h_total_rolls h_number_six_appeared
  sorry

end frequency_of_rolling_six_is_0_point_19_l136_13681


namespace how_many_years_older_is_a_than_b_l136_13650

variable (a b c : ℕ)

theorem how_many_years_older_is_a_than_b
  (hb : b = 4)
  (hc : c = b / 2)
  (h_ages_sum : a + b + c = 12) :
  a - b = 2 := by
  sorry

end how_many_years_older_is_a_than_b_l136_13650


namespace conditional_probability_l136_13628

theorem conditional_probability :
  let P_B : ℝ := 0.15
  let P_A : ℝ := 0.05
  let P_A_and_B : ℝ := 0.03
  let P_B_given_A := P_A_and_B / P_A
  P_B_given_A = 0.6 :=
by
  sorry

end conditional_probability_l136_13628


namespace divide_number_l136_13694

theorem divide_number (x : ℝ) (h : 0.3 * x = 0.2 * (80 - x) + 10) : min x (80 - x) = 28 := 
by 
  sorry

end divide_number_l136_13694


namespace mia_spent_total_l136_13649

theorem mia_spent_total (sibling_cost parent_cost : ℕ) (num_siblings num_parents : ℕ)
    (h1 : sibling_cost = 30)
    (h2 : parent_cost = 30)
    (h3 : num_siblings = 3)
    (h4 : num_parents = 2) :
    sibling_cost * num_siblings + parent_cost * num_parents = 150 :=
by
  sorry

end mia_spent_total_l136_13649


namespace num_dogs_with_spots_l136_13647

variable (D P : ℕ)

theorem num_dogs_with_spots (h1 : D / 2 = D / 2) (h2 : D / 5 = P) : (5 * P) / 2 = D / 2 := 
by
  have h3 : 5 * P = D := by
    sorry
  have h4 : (5 * P) / 2 = D / 2 := by
    rw [h3]
  exact h4

end num_dogs_with_spots_l136_13647


namespace find_x_when_y_neg4_l136_13631

variable {x y : ℝ}
variable (k : ℝ)

-- Condition: x is inversely proportional to y
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop :=
  x * y = k

theorem find_x_when_y_neg4 (h : inversely_proportional 5 10 50) :
  inversely_proportional x (-4) 50 → x = -25 / 2 :=
by sorry

end find_x_when_y_neg4_l136_13631


namespace roots_quadratic_sum_squares_l136_13620

theorem roots_quadratic_sum_squares :
  (∃ a b : ℝ, (∀ x : ℝ, x^2 - 4 * x + 4 = 0 → (x = a ∨ x = b)) ∧ a^2 + b^2 = 8) :=
by
  sorry

end roots_quadratic_sum_squares_l136_13620


namespace beth_crayon_packs_l136_13656

theorem beth_crayon_packs (P : ℕ) (h1 : 10 * P + 6 = 46) : P = 4 :=
by
  sorry

end beth_crayon_packs_l136_13656


namespace sum_of_z_values_l136_13602

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem sum_of_z_values (z1 z2 : ℝ) (hz1 : f (3 * z1) = 11) (hz2 : f (3 * z2) = 11) :
  z1 + z2 = - (2 / 9) :=
sorry

end sum_of_z_values_l136_13602


namespace square_garden_area_l136_13671

theorem square_garden_area (P A : ℕ)
  (h1 : P = 40)
  (h2 : A = 2 * P + 20) :
  A = 100 :=
by
  rw [h1] at h2 -- Substitute h1 (P = 40) into h2 (A = 2P + 20)
  norm_num at h2 -- Normalize numeric expressions in h2
  exact h2 -- Conclude by showing h2 (A = 100) holds

-- The output should be able to build successfully without solving the proof.

end square_garden_area_l136_13671


namespace find_original_number_l136_13689

theorem find_original_number (x : ℕ) 
    (h1 : (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183) : x = 32 := 
by
  sorry

end find_original_number_l136_13689


namespace proof_problem_l136_13604

def problem : Prop :=
  ∃ (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004

theorem proof_problem : 
  problem → 
  ∃! (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004 :=
sorry

end proof_problem_l136_13604


namespace total_cans_collected_l136_13638

theorem total_cans_collected 
  (bags_saturday : ℕ) 
  (bags_sunday : ℕ) 
  (cans_per_bag : ℕ) 
  (h1 : bags_saturday = 6) 
  (h2 : bags_sunday = 3) 
  (h3 : cans_per_bag = 8) : 
  bags_saturday + bags_sunday * cans_per_bag = 72 := 
by 
  simp [h1, h2, h3]; -- Simplify using the given conditions
  sorry -- Placeholder for the computation proof

end total_cans_collected_l136_13638


namespace value_of_a5_l136_13622

variable (a_n : ℕ → ℝ)
variable (a1 a9 a5 : ℝ)

-- Given conditions
axiom a1_plus_a9_eq_10 : a1 + a9 = 10
axiom arithmetic_sequence : ∀ n, a_n n = a1 + (n - 1) * (a_n 2 - a1)

-- Prove that a5 = 5
theorem value_of_a5 : a5 = 5 :=
by
  sorry

end value_of_a5_l136_13622


namespace largest_real_number_mu_l136_13672

noncomputable def largest_mu : ℝ := 13 / 2

theorem largest_real_number_mu (
  a b c d : ℝ
) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) :
  (a^2 + b^2 + c^2 + d^2) ≥ (largest_mu * a * b + b * c + 2 * c * d) :=
sorry

end largest_real_number_mu_l136_13672


namespace quadratic_smallest_root_a_quadratic_smallest_root_b_l136_13610

-- For Part (a)
theorem quadratic_smallest_root_a (a : ℝ) 
  (h : a^2 - 9 * a - 10 = 0 ∧ ∀ x, x^2 - 9 * x - 10 = 0 → x ≥ a) : 
  a^4 - 909 * a = 910 :=
by sorry

-- For Part (b)
theorem quadratic_smallest_root_b (b : ℝ) 
  (h : b^2 - 9 * b + 10 = 0 ∧ ∀ x, x^2 - 9 * x + 10 = 0 → x ≥ b) : 
  b^4 - 549 * b = -710 :=
by sorry

end quadratic_smallest_root_a_quadratic_smallest_root_b_l136_13610


namespace trig_expression_simplify_l136_13632

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l136_13632


namespace mother_hen_heavier_l136_13606

-- Define the weights in kilograms
def weight_mother_hen : ℝ := 2.3
def weight_baby_chick : ℝ := 0.4

-- State the theorem with the final correct answer
theorem mother_hen_heavier :
  weight_mother_hen - weight_baby_chick = 1.9 :=
by
  sorry

end mother_hen_heavier_l136_13606


namespace max_ab_bc_cd_l136_13663

theorem max_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
    ab + bc + cd ≤ 10000 := by
  sorry

end max_ab_bc_cd_l136_13663


namespace find_values_of_a_l136_13676

theorem find_values_of_a :
  ∃ (a : ℝ), 
    (∀ x y, (|y + 2| + |x - 11| - 3) * (x^2 + y^2 - 13) = 0 ∧ 
             (x - 5)^2 + (y + 2)^2 = a) ↔ 
    a = 9 ∨ a = 42 + 2 * Real.sqrt 377 :=
sorry

end find_values_of_a_l136_13676


namespace max_constant_N_l136_13698

theorem max_constant_N (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0):
  (c^2 + d^2) ≠ 0 → ∃ N, N = 1 ∧ (a^2 + b^2) / (c^2 + d^2) ≤ 1 :=
by
  sorry

end max_constant_N_l136_13698


namespace number_of_five_dollar_bills_l136_13611

theorem number_of_five_dollar_bills (total_money denomination expected_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : denomination = 5) 
  (h3 : expected_bills = total_money / denomination) : 
  expected_bills = 9 :=
by
  sorry

end number_of_five_dollar_bills_l136_13611


namespace paint_leftover_l136_13686

theorem paint_leftover (containers total_walls tiles_wall paint_ceiling : ℕ) 
  (h_containers : containers = 16) 
  (h_total_walls : total_walls = 4) 
  (h_tiles_wall : tiles_wall = 1) 
  (h_paint_ceiling : paint_ceiling = 1) : 
  containers - ((total_walls - tiles_wall) * (containers / total_walls)) - paint_ceiling = 3 :=
by 
  sorry

end paint_leftover_l136_13686


namespace check_correct_conditional_expression_l136_13684
-- importing the necessary library for basic algebraic constructions and predicates

-- defining a predicate to denote the symbolic representation of conditional expressions validity
def valid_conditional_expression (expr: String) : Prop :=
  expr = "x <> 1" ∨ expr = "x > 1" ∨ expr = "x >= 1" ∨ expr = "x < 1" ∨ expr = "x <= 1" ∨ expr = "x = 1"

-- theorem to check for the valid conditional expression among the given options
theorem check_correct_conditional_expression :
  (valid_conditional_expression "1 < x < 2") = false ∧ 
  (valid_conditional_expression "x > < 1") = false ∧ 
  (valid_conditional_expression "x <> 1") = true ∧ 
  (valid_conditional_expression "x ≤ 1") = true :=
by sorry

end check_correct_conditional_expression_l136_13684


namespace solve_for_x_l136_13660

theorem solve_for_x (x y : ℝ) (h : (x + 1) / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 5)) : 
  x = (y^2 + 3 * y - 1) / 7 := 
by 
  sorry

end solve_for_x_l136_13660


namespace slices_dinner_l136_13612

variable (lunch_slices : ℕ) (total_slices : ℕ)
variable (h1 : lunch_slices = 7) (h2 : total_slices = 12)

theorem slices_dinner : total_slices - lunch_slices = 5 :=
by sorry

end slices_dinner_l136_13612


namespace calculate_value_is_neg_seventeen_l136_13690

theorem calculate_value_is_neg_seventeen : -3^2 + (-2)^3 = -17 :=
by
  sorry

end calculate_value_is_neg_seventeen_l136_13690


namespace geom_progression_n_eq_6_l136_13641

theorem geom_progression_n_eq_6
  (a r : ℝ)
  (h_r : r = 6)
  (h_ratio : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217) :
  n = 6 :=
by
  sorry

end geom_progression_n_eq_6_l136_13641


namespace find_opposite_endpoint_l136_13657

/-- A utility function to model coordinate pairs as tuples -/
def coord_pair := (ℝ × ℝ)

-- Define the center and one endpoint
def center : coord_pair := (4, 6)
def endpoint1 : coord_pair := (2, 1)

-- Define the expected endpoint
def expected_endpoint2 : coord_pair := (6, 11)

/-- Definition of the opposite endpoint given the center and one endpoint -/
def opposite_endpoint (c : coord_pair) (p : coord_pair) : coord_pair :=
  let dx := c.1 - p.1
  let dy := c.2 - p.2
  (c.1 + dx, c.2 + dy)

/-- The proof statement for the problem -/
theorem find_opposite_endpoint :
  opposite_endpoint center endpoint1 = expected_endpoint2 :=
sorry

end find_opposite_endpoint_l136_13657


namespace ratio_of_cost_to_selling_price_l136_13659

-- Define the given conditions
def cost_price (CP : ℝ) := CP
def selling_price (CP : ℝ) : ℝ := CP + 0.25 * CP

-- Lean statement for the problem
theorem ratio_of_cost_to_selling_price (CP SP : ℝ) (h1 : SP = selling_price CP) : CP / SP = 4 / 5 :=
by
  sorry

end ratio_of_cost_to_selling_price_l136_13659


namespace kylie_total_apples_l136_13633

-- Define the conditions as given in the problem.
def first_hour_apples : ℕ := 66
def second_hour_apples : ℕ := 2 * first_hour_apples
def third_hour_apples : ℕ := first_hour_apples / 3

-- Define the mathematical proof problem.
theorem kylie_total_apples : 
  first_hour_apples + second_hour_apples + third_hour_apples = 220 :=
by
  -- Proof goes here
  sorry

end kylie_total_apples_l136_13633
