import Mathlib

namespace NUMINAMATH_GPT_yoki_cans_correct_l279_27965

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end NUMINAMATH_GPT_yoki_cans_correct_l279_27965


namespace NUMINAMATH_GPT_work_together_days_l279_27954

theorem work_together_days (A B : ℝ) (h1 : A = 1/2 * B) (h2 : B = 1/48) :
  1 / (A + B) = 32 :=
by
  sorry

end NUMINAMATH_GPT_work_together_days_l279_27954


namespace NUMINAMATH_GPT_fill_tank_with_leak_l279_27974

theorem fill_tank_with_leak (P L T : ℝ) 
  (hP : P = 1 / 2)  -- Rate of the pump
  (hL : L = 1 / 6)  -- Rate of the leak
  (hT : T = 3)  -- Time taken to fill the tank with the leak
  : 1 / (P - L) = T := 
by
  sorry

end NUMINAMATH_GPT_fill_tank_with_leak_l279_27974


namespace NUMINAMATH_GPT_min_cost_to_form_closed_chain_l279_27926

/-- Definition for the cost model -/
def cost_separate_link : ℕ := 1
def cost_attach_link : ℕ := 2
def total_cost (n : ℕ) : ℕ := n * (cost_separate_link + cost_attach_link)

-- Number of pieces of gold chain and links in each chain
def num_pieces : ℕ := 13

/-- Minimum cost calculation proof statement -/
theorem min_cost_to_form_closed_chain : total_cost (num_pieces - 1) = 36 := 
by
  sorry

end NUMINAMATH_GPT_min_cost_to_form_closed_chain_l279_27926


namespace NUMINAMATH_GPT_part1_part2_l279_27904

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - 5 * a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + 3

-- (1)
theorem part1 (x : ℝ) : abs (g x) < 8 → -4 < x ∧ x < 6 :=
by
  sorry

-- (2)
theorem part2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ 0.4 ∨ a ≤ -0.8) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l279_27904


namespace NUMINAMATH_GPT_both_true_of_neg_and_false_l279_27945

variable (P Q : Prop)

theorem both_true_of_neg_and_false (h : ¬ (P ∧ Q) = False) : P ∧ Q :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_both_true_of_neg_and_false_l279_27945


namespace NUMINAMATH_GPT_hyperbola_equation_l279_27903

theorem hyperbola_equation 
  (h k a c : ℝ)
  (center_cond : (h, k) = (3, -1))
  (vertex_cond : a = abs (2 - (-1)))
  (focus_cond : c = abs (7 - (-1)))
  (b : ℝ)
  (b_square : c^2 = a^2 + b^2) :
  h + k + a + b = 5 + Real.sqrt 55 := 
by
  -- Prove that given the conditions, the value of h + k + a + b is 5 + √55.
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l279_27903


namespace NUMINAMATH_GPT_water_added_l279_27940

theorem water_added (initial_volume : ℕ) (ratio_milk_water_initial : ℚ) 
  (ratio_milk_water_final : ℚ) (w : ℕ)
  (initial_volume_eq : initial_volume = 45)
  (ratio_milk_water_initial_eq : ratio_milk_water_initial = 4 / 1)
  (ratio_milk_water_final_eq : ratio_milk_water_final = 6 / 5)
  (final_ratio_eq : ratio_milk_water_final = 36 / (9 + w)) :
  w = 21 := 
sorry

end NUMINAMATH_GPT_water_added_l279_27940


namespace NUMINAMATH_GPT_value_of_x2y_plus_xy2_l279_27944

-- Define variables x and y as real numbers
variables (x y : ℝ)

-- Define the conditions
def condition1 : Prop := x + y = -2
def condition2 : Prop := x * y = -3

-- Define the proof problem
theorem value_of_x2y_plus_xy2 (h1 : condition1 x y) (h2 : condition2 x y) : x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_x2y_plus_xy2_l279_27944


namespace NUMINAMATH_GPT_inequality_sum_leq_three_l279_27909

theorem inequality_sum_leq_three
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) + 
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) + 
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2 + z^2) ≤ 3 := 
sorry

end NUMINAMATH_GPT_inequality_sum_leq_three_l279_27909


namespace NUMINAMATH_GPT_find_fraction_l279_27930

-- Let f be a real number representing the fraction
theorem find_fraction (f : ℝ) (h : f * 12 + 5 = 11) : f = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_l279_27930


namespace NUMINAMATH_GPT_proof_problem_l279_27987

variable {a b : ℝ}
variable (cond : sqrt a > sqrt b)

theorem proof_problem (h1 : a > b) (h2 : 0 ≤ a) (h3 : 0 ≤ b) :
  (a^2 > b^2) ∧
  ((b + 1) / (a + 1) > b / a) ∧
  (b + 1 / (b + 1) ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l279_27987


namespace NUMINAMATH_GPT_smallest_x_l279_27960

theorem smallest_x (a b x : ℤ) (h1 : x = 2 * a^5) (h2 : x = 5 * b^2) (pos_x : x > 0) : x = 200000 := sorry

end NUMINAMATH_GPT_smallest_x_l279_27960


namespace NUMINAMATH_GPT_chips_calories_l279_27985

-- Define the conditions
def calories_from_breakfast : ℕ := 560
def calories_from_lunch : ℕ := 780
def calories_from_cake : ℕ := 110
def calories_from_coke : ℕ := 215
def daily_calorie_limit : ℕ := 2500
def remaining_calories : ℕ := 525

-- Define the total calories consumed so far
def total_consumed : ℕ := calories_from_breakfast + calories_from_lunch + calories_from_cake + calories_from_coke

-- Define the total allowable calories without exceeding the limit
def total_allowed : ℕ := daily_calorie_limit - remaining_calories

-- Define the calories in the chips
def calories_in_chips : ℕ := total_allowed - total_consumed

-- Prove that the number of calories in the chips is 310
theorem chips_calories :
  calories_in_chips = 310 :=
by
  sorry

end NUMINAMATH_GPT_chips_calories_l279_27985


namespace NUMINAMATH_GPT_area_of_45_45_90_triangle_l279_27914

theorem area_of_45_45_90_triangle (h : ℝ) (h_eq : h = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 32 := 
by
  sorry

end NUMINAMATH_GPT_area_of_45_45_90_triangle_l279_27914


namespace NUMINAMATH_GPT_fraction_in_pairing_l279_27928

open Function

theorem fraction_in_pairing (s t : ℕ) (h : (t : ℚ) / 4 = s / 3) : 
  ((t / 4 : ℚ) + (s / 3)) / (t + s) = 2 / 7 :=
by sorry

end NUMINAMATH_GPT_fraction_in_pairing_l279_27928


namespace NUMINAMATH_GPT_problem1_problem2_l279_27923

-- Problem (1)
theorem problem1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) : a^2 + (3 / 2) * b - 5 = -2 := 
sorry

-- Problem (2)
theorem problem2 (x : ℝ) (h : 14 * x + 5 - 21 * x^2 = -2) : 6 * x^2 - 4 * x + 5 = 7 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l279_27923


namespace NUMINAMATH_GPT_total_supervisors_l279_27964

theorem total_supervisors (buses : ℕ) (supervisors_per_bus : ℕ) (h1 : buses = 7) (h2 : supervisors_per_bus = 3) :
  buses * supervisors_per_bus = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_supervisors_l279_27964


namespace NUMINAMATH_GPT_tiles_needed_to_cover_floor_l279_27915

-- Definitions of the conditions
def room_length : ℕ := 2
def room_width : ℕ := 12
def tile_area : ℕ := 4

-- The proof statement: calculate the number of tiles needed to cover the entire floor
theorem tiles_needed_to_cover_floor : 
  (room_length * room_width) / tile_area = 6 := 
by 
  sorry

end NUMINAMATH_GPT_tiles_needed_to_cover_floor_l279_27915


namespace NUMINAMATH_GPT_problem_solution_l279_27961

noncomputable def a_sequence : ℕ → ℕ := sorry
noncomputable def S_n : ℕ → ℕ := sorry
noncomputable def b_sequence : ℕ → ℕ := sorry
noncomputable def c_sequence : ℕ → ℕ := sorry
noncomputable def T_n : ℕ → ℕ := sorry

theorem problem_solution (n : ℕ) (a_condition : ∀ n : ℕ, 2 * S_n = (n + 1) ^ 2 * a_sequence n - n ^ 2 * a_sequence (n + 1))
                        (b_condition : ∀ n : ℕ, b_sequence 1 = a_sequence 1 ∧ (n ≠ 0 → n * b_sequence (n + 1) = a_sequence n * b_sequence n)) :
  (∀ n, a_sequence n = 2 * n) ∧
  (∀ n, b_sequence n = 2 ^ n) ∧
  (∀ n, T_n n = 2 ^ (n + 1) + n ^ 2 + n - 2) :=
sorry


end NUMINAMATH_GPT_problem_solution_l279_27961


namespace NUMINAMATH_GPT_quadratic_positive_difference_l279_27933

theorem quadratic_positive_difference :
  ∀ x : ℝ, x^2 - 5 * x + 15 = x + 55 → x = 10 ∨ x = -4 →
  |10 - (-4)| = 14 :=
by
  intro x h1 h2
  have h3 : x = 10 ∨ x = -4 := h2
  have h4 : |10 - (-4)| = 14 := by norm_num
  exact h4

end NUMINAMATH_GPT_quadratic_positive_difference_l279_27933


namespace NUMINAMATH_GPT_exist_x_y_l279_27962

theorem exist_x_y (a b c : ℝ) (h₁ : abs a > 2) (h₂ : a^2 + b^2 + c^2 = a * b * c + 4) :
  ∃ x y : ℝ, a = x + 1/x ∧ b = y + 1/y ∧ c = x*y + 1/(x*y) :=
sorry

end NUMINAMATH_GPT_exist_x_y_l279_27962


namespace NUMINAMATH_GPT_max_ab_sum_l279_27900

theorem max_ab_sum (a b: ℤ) (h1: a ≠ b) (h2: a * b = -132) (h3: a ≤ b): a + b = -1 :=
sorry

end NUMINAMATH_GPT_max_ab_sum_l279_27900


namespace NUMINAMATH_GPT_total_revenue_correct_l279_27984

-- Definitions based on the problem conditions
def price_per_kg_first_week : ℝ := 10
def quantity_sold_first_week : ℝ := 50
def discount_percentage : ℝ := 0.25
def multiplier_next_week : ℝ := 3

-- Derived definitions
def revenue_first_week := quantity_sold_first_week * price_per_kg_first_week
def quantity_sold_second_week := multiplier_next_week * quantity_sold_first_week
def discounted_price_per_kg := price_per_kg_first_week * (1 - discount_percentage)
def revenue_second_week := quantity_sold_second_week * discounted_price_per_kg
def total_revenue := revenue_first_week + revenue_second_week

-- The theorem that needs to be proven
theorem total_revenue_correct : total_revenue = 1625 := 
by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l279_27984


namespace NUMINAMATH_GPT_solve_ax_plus_b_l279_27949

theorem solve_ax_plus_b (a b : ℝ) : 
  (if a ≠ 0 then "unique solution, x = -b / a"
   else if b ≠ 0 then "no solution"
   else "infinitely many solutions") = "A conditional control structure should be adopted" :=
sorry

end NUMINAMATH_GPT_solve_ax_plus_b_l279_27949


namespace NUMINAMATH_GPT_Seokjin_total_fish_l279_27913

-- Define the conditions
def fish_yesterday := 10
def cost_yesterday := 3000
def additional_cost := 6000
def price_per_fish := cost_yesterday / fish_yesterday
def total_cost_today := cost_yesterday + additional_cost
def fish_today := total_cost_today / price_per_fish

-- Define the goal
theorem Seokjin_total_fish (h1 : fish_yesterday = 10)
                           (h2 : cost_yesterday = 3000)
                           (h3 : additional_cost = 6000)
                           (h4 : price_per_fish = cost_yesterday / fish_yesterday)
                           (h5 : total_cost_today = cost_yesterday + additional_cost)
                           (h6 : fish_today = total_cost_today / price_per_fish) :
  fish_yesterday + fish_today = 40 :=
by
  sorry

end NUMINAMATH_GPT_Seokjin_total_fish_l279_27913


namespace NUMINAMATH_GPT_curve_is_parabola_l279_27916

-- Define the condition: the curve is defined by the given polar equation
def polar_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- The main theorem statement: Prove that the curve defined by the equation is a parabola
theorem curve_is_parabola (r θ : ℝ) (h : polar_eq r θ) : ∃ x y : ℝ, x = 1 + 2 * y :=
sorry

end NUMINAMATH_GPT_curve_is_parabola_l279_27916


namespace NUMINAMATH_GPT_probability_exactly_one_win_l279_27978

theorem probability_exactly_one_win :
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  P_exactly_one_win = 8 / 15 :=
by
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  have h1 : P_exactly_one_win = 8 / 15 := sorry
  exact h1

end NUMINAMATH_GPT_probability_exactly_one_win_l279_27978


namespace NUMINAMATH_GPT_largest_square_area_l279_27992

theorem largest_square_area (XY XZ YZ : ℝ)
  (h1 : XZ^2 = 2 * XY^2)
  (h2 : XY^2 + YZ^2 = XZ^2)
  (h3 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_largest_square_area_l279_27992


namespace NUMINAMATH_GPT_find_value_l279_27963

theorem find_value 
  (x1 x2 x3 x4 x5 : ℝ)
  (condition1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 = 2)
  (condition2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 = 15)
  (condition3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 = 130) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 = 347 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l279_27963


namespace NUMINAMATH_GPT_h_inv_f_neg3_does_not_exist_real_l279_27902

noncomputable def h : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

theorem h_inv_f_neg3_does_not_exist_real (h_inv : ℝ → ℝ)
  (h_cond : ∀ (x : ℝ), f (h_inv (h x)) = 7 * x ^ 2 + 4) :
  ¬ ∃ x : ℝ, h_inv (f (-3)) = x :=
by 
  sorry

end NUMINAMATH_GPT_h_inv_f_neg3_does_not_exist_real_l279_27902


namespace NUMINAMATH_GPT_students_per_row_first_scenario_l279_27927

theorem students_per_row_first_scenario 
  (S R x : ℕ)
  (h1 : S = x * R + 6)
  (h2 : S = 12 * (R - 3))
  (h3 : S = 6 * R) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_students_per_row_first_scenario_l279_27927


namespace NUMINAMATH_GPT_find_w_squared_l279_27994

theorem find_w_squared (w : ℝ) :
  (w + 15)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = ((-21 + Real.sqrt 7965) / 22)^2 ∨ 
        w^2 = ((-21 - Real.sqrt 7965) / 22)^2 :=
by sorry

end NUMINAMATH_GPT_find_w_squared_l279_27994


namespace NUMINAMATH_GPT_common_area_approximation_l279_27906

noncomputable def elliptical_domain (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2) ≤ 1

noncomputable def circular_domain (x y : ℝ) : Prop :=
  (x^2 + y^2) ≤ 2

noncomputable def intersection_area : ℝ :=
  7.27

theorem common_area_approximation :
  ∃ area, 
    elliptical_domain x y ∧ circular_domain x y →
    abs (area - intersection_area) < 0.01 :=
sorry

end NUMINAMATH_GPT_common_area_approximation_l279_27906


namespace NUMINAMATH_GPT_zac_strawberries_l279_27980

theorem zac_strawberries (J M Z : ℕ) 
  (h1 : J + M + Z = 550) 
  (h2 : J + M = 350) 
  (h3 : M + Z = 250) : 
  Z = 200 :=
sorry

end NUMINAMATH_GPT_zac_strawberries_l279_27980


namespace NUMINAMATH_GPT_proof_m_n_sum_l279_27986

-- Definitions based on conditions
def m : ℕ := 2
def n : ℕ := 49

-- Problem statement as a Lean theorem
theorem proof_m_n_sum : m + n = 51 :=
by
  -- This is where the detailed proof would go. Using sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_proof_m_n_sum_l279_27986


namespace NUMINAMATH_GPT_average_student_headcount_is_10983_l279_27901

def student_headcount_fall_03_04 := 11500
def student_headcount_spring_03_04 := 10500
def student_headcount_fall_04_05 := 11600
def student_headcount_spring_04_05 := 10700
def student_headcount_fall_05_06 := 11300
def student_headcount_spring_05_06 := 10300 -- Assume value

def total_student_headcount :=
  student_headcount_fall_03_04 + student_headcount_spring_03_04 +
  student_headcount_fall_04_05 + student_headcount_spring_04_05 +
  student_headcount_fall_05_06 + student_headcount_spring_05_06

def average_student_headcount := total_student_headcount / 6

theorem average_student_headcount_is_10983 :
  average_student_headcount = 10983 :=
by -- Will prove the theorem
sorry

end NUMINAMATH_GPT_average_student_headcount_is_10983_l279_27901


namespace NUMINAMATH_GPT_repeating_decimal_denominators_l279_27959

theorem repeating_decimal_denominators (a b c : ℕ) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) (h_not_all_nine : ¬(a = 9 ∧ b = 9 ∧ c = 9)) : 
  ∃ denominators : Finset ℕ, denominators.card = 7 ∧ (∀ d ∈ denominators, d ∣ 999) ∧ ¬ 1 ∈ denominators :=
sorry

end NUMINAMATH_GPT_repeating_decimal_denominators_l279_27959


namespace NUMINAMATH_GPT_number_of_students_preferring_dogs_l279_27998

-- Define the conditions
def total_students : ℕ := 30
def dogs_video_games_chocolate_percentage : ℚ := 0.50
def dogs_movies_vanilla_percentage : ℚ := 0.10
def cats_video_games_chocolate_percentage : ℚ := 0.20
def cats_movies_vanilla_percentage : ℚ := 0.15

-- Define the target statement to prove
theorem number_of_students_preferring_dogs : 
  (dogs_video_games_chocolate_percentage + dogs_movies_vanilla_percentage) * total_students = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_preferring_dogs_l279_27998


namespace NUMINAMATH_GPT_captain_smollett_problem_l279_27970

/-- 
Given the captain's age, the number of children he has, and the length of his schooner, 
prove that the unique solution to the product condition is age = 53 years, children = 6, 
and length = 101 feet, under the given constraints.
-/
theorem captain_smollett_problem
  (age children length : ℕ)
  (h1 : age < 100)
  (h2 : children > 3)
  (h3 : age * children * length = 32118) : age = 53 ∧ children = 6 ∧ length = 101 :=
by {
  -- Proof will be filled in later
  sorry
}

end NUMINAMATH_GPT_captain_smollett_problem_l279_27970


namespace NUMINAMATH_GPT_crates_needed_l279_27976

-- Conditions as definitions
def novels : ℕ := 145
def comics : ℕ := 271
def documentaries : ℕ := 419
def albums : ℕ := 209
def crate_capacity : ℕ := 9

-- Total items calculation
def total_items : ℕ := novels + comics + documentaries + albums

-- Proof statement
theorem crates_needed : (total_items / crate_capacity) = 116 := by
  sorry

end NUMINAMATH_GPT_crates_needed_l279_27976


namespace NUMINAMATH_GPT_distance_to_other_focus_of_ellipse_l279_27911

noncomputable def ellipse_param (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def is_focus_distance (a distF1 distF2 : ℝ) : Prop :=
  ∀ P₁ P₂ : ℝ, distF1 + distF2 = 2 * a

theorem distance_to_other_focus_of_ellipse (x y : ℝ) (distF1 : ℝ) :
  ellipse_param 4 5 x y ∧ distF1 = 6 → is_focus_distance 5 distF1 4 :=
by
  simp [ellipse_param, is_focus_distance]
  sorry

end NUMINAMATH_GPT_distance_to_other_focus_of_ellipse_l279_27911


namespace NUMINAMATH_GPT_age_ratio_in_two_years_l279_27966

variable (S M : ℕ)

-- Conditions
def sonCurrentAge : Prop := S = 18
def manCurrentAge : Prop := M = S + 20
def multipleCondition : Prop := ∃ k : ℕ, M + 2 = k * (S + 2)

-- Statement to prove
theorem age_ratio_in_two_years (h1 : sonCurrentAge S) (h2 : manCurrentAge S M) (h3 : multipleCondition S M) : 
  (M + 2) / (S + 2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_age_ratio_in_two_years_l279_27966


namespace NUMINAMATH_GPT_find_real_triples_l279_27957

theorem find_real_triples :
  ∀ (a b c : ℝ), a^2 + a * b + c = 0 ∧ b^2 + b * c + a = 0 ∧ c^2 + c * a + b = 0
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_find_real_triples_l279_27957


namespace NUMINAMATH_GPT_sum_same_probability_l279_27919

-- Definition for standard dice probability problem
def dice_problem (n : ℕ) (target_sum : ℕ) (target_sum_of_faces : ℕ) : Prop :=
  let faces := [1, 2, 3, 4, 5, 6]
  let min_sum := n * 1
  let max_sum := n * 6
  let average_sum := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * average_sum - target_sum
  symmetric_sum = target_sum_of_faces

-- The proof statement (no proof included, just the declaration)
theorem sum_same_probability : dice_problem 8 12 44 :=
by sorry

end NUMINAMATH_GPT_sum_same_probability_l279_27919


namespace NUMINAMATH_GPT_mean_greater_than_median_by_l279_27955

-- Define the data: number of students missing specific days
def studentsMissingDays := [3, 1, 4, 1, 1, 5] -- corresponding to 0, 1, 2, 3, 4, 5 days missed

-- Total number of students
def totalStudents := 15

-- Function to calculate the sum of missed days weighted by the number of students
def totalMissedDays := (0 * 3) + (1 * 1) + (2 * 4) + (3 * 1) + (4 * 1) + (5 * 5)

-- Calculate the mean number of missed days
def meanDaysMissed := totalMissedDays / totalStudents

-- Select the median number of missed days (8th student) from the ordered list
def medianDaysMissed := 2

-- Calculate the difference between the mean and median
def difference := meanDaysMissed - medianDaysMissed

-- Define the proof problem statement
theorem mean_greater_than_median_by : 
  difference = 11 / 15 :=
by
  -- This is where the actual proof would be written
  sorry

end NUMINAMATH_GPT_mean_greater_than_median_by_l279_27955


namespace NUMINAMATH_GPT_evaluate_expression_l279_27975

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l279_27975


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l279_27908

-- Define System (1) and prove its solution
theorem solve_system1 (x y : ℝ) (h1 : x = 5 - y) (h2 : x - 3 * y = 1) : x = 4 ∧ y = 1 := by
  sorry

-- Define System (2) and prove its solution
theorem solve_system2 (x y : ℝ) (h1 : x - 2 * y = 6) (h2 : 2 * x + 3 * y = -2) : x = 2 ∧ y = -2 := by
  sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l279_27908


namespace NUMINAMATH_GPT_least_y_value_l279_27956

theorem least_y_value (y : ℝ) : 2 * y ^ 2 + 7 * y + 3 = 5 → y ≥ -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_least_y_value_l279_27956


namespace NUMINAMATH_GPT_even_not_div_by_4_not_sum_consecutive_odds_l279_27999

theorem even_not_div_by_4_not_sum_consecutive_odds
  (e : ℤ) (h_even: e % 2 = 0) (h_nondiv4: ¬ (e % 4 = 0)) :
  ∀ n : ℤ, e ≠ n + (n + 2) :=
by
  sorry

end NUMINAMATH_GPT_even_not_div_by_4_not_sum_consecutive_odds_l279_27999


namespace NUMINAMATH_GPT_range_f_real_l279_27917

noncomputable def f (a : ℝ) (x : ℝ) :=
  if x > 1 then (a ^ x) else (4 - a / 2) * x + 2

theorem range_f_real (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ (1 < a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_f_real_l279_27917


namespace NUMINAMATH_GPT_only_selected_A_is_20_l279_27941

def cardinality_A (x : ℕ) : ℕ := x
def cardinality_B (x : ℕ) : ℕ := x + 8
def cardinality_union (x : ℕ) : ℕ := 54
def cardinality_intersection (x : ℕ) : ℕ := 6

theorem only_selected_A_is_20 (x : ℕ) (h_total : cardinality_union x = 54) 
  (h_inter : cardinality_intersection x = 6) (h_B : cardinality_B x = x + 8) :
  cardinality_A x - cardinality_intersection x = 20 :=
by
  sorry

end NUMINAMATH_GPT_only_selected_A_is_20_l279_27941


namespace NUMINAMATH_GPT_max_value_l279_27972

open Real

theorem max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 75) : 
  xy * (75 - 2 * x - 5 * y) ≤ 1562.5 := 
sorry

end NUMINAMATH_GPT_max_value_l279_27972


namespace NUMINAMATH_GPT_speed_against_current_l279_27990

theorem speed_against_current (V_m V_c : ℕ) (h1 : V_m + V_c = 20) (h2 : V_c = 3) : V_m - V_c = 14 :=
by 
  sorry

end NUMINAMATH_GPT_speed_against_current_l279_27990


namespace NUMINAMATH_GPT_middle_group_frequency_l279_27981

theorem middle_group_frequency (f : ℕ) (A : ℕ) (h_total : A + f = 100) (h_middle : f = A) : f = 50 :=
by
  sorry

end NUMINAMATH_GPT_middle_group_frequency_l279_27981


namespace NUMINAMATH_GPT_find_y_coordinate_l279_27935

theorem find_y_coordinate (y : ℝ) (h : y > 0) (dist_eq : (10 - 2)^2 + (y - 5)^2 = 13^2) : y = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_y_coordinate_l279_27935


namespace NUMINAMATH_GPT_find_range_m_l279_27947

variables (m : ℝ)

def p (m : ℝ) : Prop :=
  (∀ x y : ℝ, (x^2 / (2 * m)) - (y^2 / (m - 1)) = 1) → false

def q (m : ℝ) : Prop :=
  (∀ e : ℝ, (1 < e ∧ e < 2) → (∀ x y : ℝ, (y^2 / 5) - (x^2 / m) = 1)) → false

noncomputable def range_m (m : ℝ) : Prop :=
  p m = false ∧ q m = false ∧ (p m ∨ q m) = true → (1/3 ≤ m ∧ m < 15)

theorem find_range_m : ∀ m : ℝ, range_m m :=
by
  intro m
  simp [range_m, p, q]
  sorry

end NUMINAMATH_GPT_find_range_m_l279_27947


namespace NUMINAMATH_GPT_sum_of_real_y_values_l279_27995

theorem sum_of_real_y_values :
  (∀ (x y : ℝ), x^2 + x^2 * y^2 + x^2 * y^4 = 525 ∧ x + x * y + x * y^2 = 35 → y = 1 / 2 ∨ y = 2) →
    (1 / 2 + 2 = 5 / 2) :=
by
  intro h
  have := h (1 / 2)
  have := h 2
  sorry  -- Proof steps showing 1/2 and 2 are the solutions, leading to the sum 5/2

end NUMINAMATH_GPT_sum_of_real_y_values_l279_27995


namespace NUMINAMATH_GPT_product_of_numbers_l279_27934

-- Definitions of the conditions
variables (x y : ℝ)

-- The conditions themselves
def cond1 : Prop := x + y = 20
def cond2 : Prop := x^2 + y^2 = 200

-- Statement of the proof problem
theorem product_of_numbers (h1 : cond1 x y) (h2 : cond2 x y) : x * y = 100 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l279_27934


namespace NUMINAMATH_GPT_right_isosceles_areas_no_relations_l279_27931

theorem right_isosceles_areas_no_relations :
  let W := 1 / 2 * 5 * 5
  let X := 1 / 2 * 12 * 12
  let Y := 1 / 2 * 13 * 13
  ¬ (X + Y = 2 * W + X ∨ W + X = Y ∨ 2 * X = W + Y ∨ X + W = W ∨ W + Y = 2 * X) :=
by
  sorry

end NUMINAMATH_GPT_right_isosceles_areas_no_relations_l279_27931


namespace NUMINAMATH_GPT_geometric_sequence_product_l279_27948

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n, a (n + 1) = a n * r

noncomputable def quadratic_roots (a1 a10 : ℝ) : Prop :=
3 * a1^2 - 2 * a1 - 6 = 0 ∧ 3 * a10^2 - 2 * a10 - 6 = 0

theorem geometric_sequence_product {a : ℕ → ℝ}
  (h_geom : geometric_sequence a)
  (h_roots : quadratic_roots (a 1) (a 10)) :
  a 4 * a 7 = -2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l279_27948


namespace NUMINAMATH_GPT_find_x_l279_27943

def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
def vector_b : ℝ × ℝ := (-3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular (vector_a x + vector_b) vector_b) : 
  x = -7 / 2 :=
  sorry

end NUMINAMATH_GPT_find_x_l279_27943


namespace NUMINAMATH_GPT_binomial_expansion_evaluation_l279_27938

theorem binomial_expansion_evaluation : 
  (8 ^ 4 + 4 * (8 ^ 3) * 2 + 6 * (8 ^ 2) * (2 ^ 2) + 4 * 8 * (2 ^ 3) + 2 ^ 4) = 10000 := 
by 
  sorry

end NUMINAMATH_GPT_binomial_expansion_evaluation_l279_27938


namespace NUMINAMATH_GPT_remainder_8_pow_215_mod_9_l279_27982

theorem remainder_8_pow_215_mod_9 : (8 ^ 215) % 9 = 8 := by
  -- condition
  have pattern : ∀ n, (8 ^ (2 * n + 1)) % 9 = 8 := by sorry
  -- final proof
  exact pattern 107

end NUMINAMATH_GPT_remainder_8_pow_215_mod_9_l279_27982


namespace NUMINAMATH_GPT_deepak_present_age_l279_27991

theorem deepak_present_age (x : ℕ) (h1 : ∀ current_age_rahul current_age_deepak, 
  4 * x = current_age_rahul ∧ 3 * x = current_age_deepak)
  (h2 : ∀ current_age_rahul, current_age_rahul + 6 = 22) :
  3 * x = 12 :=
by
  have h3 : 4 * x + 6 = 22 := h2 (4 * x)
  linarith

end NUMINAMATH_GPT_deepak_present_age_l279_27991


namespace NUMINAMATH_GPT_binary_arithmetic_l279_27936

def a : ℕ := 0b10110  -- 10110_2
def b : ℕ := 0b1101   -- 1101_2
def c : ℕ := 0b11100  -- 11100_2
def d : ℕ := 0b11101  -- 11101_2
def e : ℕ := 0b101    -- 101_2

theorem binary_arithmetic :
  (a + b - c + d + e) = 0b101101 := by
  sorry

end NUMINAMATH_GPT_binary_arithmetic_l279_27936


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l279_27968

-- (1) Simplify the expression: 3a(a+1) - (3+a)(3-a) - (2a-1)^2 == 7a - 10
theorem simplify_expr1 (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1) ^ 2 = 7 * a - 10 :=
sorry

-- (2) Simplify the expression: ((x^2 - 2x + 4) / (x - 1) + 2 - x) / (x^2 + 4x + 4) / (1 - x) == -2 / (x + 2)^2
theorem simplify_expr2 (x : ℝ) (h : x ≠ 1) (h1 : x ≠ 0) : 
  (((x^2 - 2 * x + 4) / (x - 1) + 2 - x) / ((x^2 + 4 * x + 4) / (1 - x))) = -2 / (x + 2)^2 :=
sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l279_27968


namespace NUMINAMATH_GPT_set_difference_P_M_l279_27967

open Set

noncomputable def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2009}
noncomputable def P : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2010}

theorem set_difference_P_M : P \ M = {2010} :=
by
  sorry

end NUMINAMATH_GPT_set_difference_P_M_l279_27967


namespace NUMINAMATH_GPT_total_weight_correct_l279_27951

-- Definitions for the problem conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

def fill1 : ℝ := 0.7
def fill2 : ℝ := 0.6
def fill3 : ℝ := 0.5

def density1 : ℝ := 5
def density2 : ℝ := 4
def density3 : ℝ := 3

-- The weights of the sand in each jug
def weight1 : ℝ := fill1 * jug1_capacity * density1
def weight2 : ℝ := fill2 * jug2_capacity * density2
def weight3 : ℝ := fill3 * jug3_capacity * density3

-- The total weight of the sand in all jugs
def total_weight : ℝ := weight1 + weight2 + weight3

-- The proof statement
theorem total_weight_correct : total_weight = 20.2 := by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l279_27951


namespace NUMINAMATH_GPT_corresponding_angles_equal_l279_27922

-- Definition: Corresponding angles and their equality
def corresponding_angles (α β : ℝ) : Prop :=
  -- assuming definition of corresponding angles can be defined
  sorry

theorem corresponding_angles_equal {α β : ℝ} (h : corresponding_angles α β) : α = β :=
by
  -- the proof is provided in the problem statement
  sorry

end NUMINAMATH_GPT_corresponding_angles_equal_l279_27922


namespace NUMINAMATH_GPT_total_tiles_needed_l279_27932

-- Definitions of the given conditions
def blue_tiles : Nat := 48
def red_tiles : Nat := 32
def additional_tiles_needed : Nat := 20

-- Statement to prove the total number of tiles needed to complete the pool
theorem total_tiles_needed : blue_tiles + red_tiles + additional_tiles_needed = 100 := by
  sorry

end NUMINAMATH_GPT_total_tiles_needed_l279_27932


namespace NUMINAMATH_GPT_basketball_team_wins_l279_27918

theorem basketball_team_wins (wins_first_60 : ℕ) (remaining_games : ℕ) (total_games : ℕ) (target_win_percentage : ℚ) (winning_games : ℕ) : 
  wins_first_60 = 45 → remaining_games = 40 → total_games = 100 → target_win_percentage = 0.75 → 
  winning_games = 30 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_basketball_team_wins_l279_27918


namespace NUMINAMATH_GPT_most_likely_sum_exceeding_twelve_l279_27907

-- Define a die with faces 0, 1, 2, 3, 4, 5
def die_faces : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define a function to get the sum of rolled results exceeding 12
noncomputable def sum_exceeds_twelve (rolls : List ℕ) : ℕ :=
  let sum := rolls.foldl (· + ·) 0
  if sum > 12 then sum else 0

-- Define a function to simulate the die roll until the sum exceeds 12
noncomputable def roll_die_until_exceeds_twelve : ℕ :=
  sorry -- This would contain the logic to simulate the rolling process

-- The theorem statement that the most likely value of the sum exceeding 12 is 13
theorem most_likely_sum_exceeding_twelve : roll_die_until_exceeds_twelve = 13 :=
  sorry

end NUMINAMATH_GPT_most_likely_sum_exceeding_twelve_l279_27907


namespace NUMINAMATH_GPT_shelves_needed_l279_27924

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (shelves : ℕ) :
  total_books = 34 →
  books_taken = 7 →
  books_per_shelf = 3 →
  remaining_books = total_books - books_taken →
  shelves = remaining_books / books_per_shelf →
  shelves = 9 :=
by
  intros h_total h_taken h_per_shelf h_remaining h_shelves
  rw [h_total, h_taken, h_per_shelf] at *
  sorry

end NUMINAMATH_GPT_shelves_needed_l279_27924


namespace NUMINAMATH_GPT_negation_exists_negation_proposition_l279_27910

theorem negation_exists (P : ℝ → Prop) :
  (∃ x : ℝ, P x) ↔ ¬ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_exists_negation_proposition_l279_27910


namespace NUMINAMATH_GPT_inequality_no_solution_l279_27950

-- Define the quadratic inequality.
def quadratic_ineq (m x : ℝ) : Prop :=
  (m + 1) * x^2 - m * x + (m - 1) > 0

-- Define the condition for m.
def range_of_m (m : ℝ) : Prop :=
  m ≤ - (2 * Real.sqrt 3) / 3

-- Theorem stating that if the inequality has no solution, m gets restricted.
theorem inequality_no_solution (m : ℝ) :
  (∀ x : ℝ, ¬ quadratic_ineq m x) ↔ range_of_m m :=
by sorry

end NUMINAMATH_GPT_inequality_no_solution_l279_27950


namespace NUMINAMATH_GPT_restaurant_donates_24_l279_27971

def restaurant_donation (customer_donation_per_person : ℕ) (num_customers : ℕ) (restaurant_donation_per_ten_dollars : ℕ) : ℕ :=
  let total_customer_donation := customer_donation_per_person * num_customers
  let increments_of_ten := total_customer_donation / 10
  increments_of_ten * restaurant_donation_per_ten_dollars

theorem restaurant_donates_24 :
  restaurant_donation 3 40 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_donates_24_l279_27971


namespace NUMINAMATH_GPT_intersection_of_sets_l279_27929

open Set

theorem intersection_of_sets (p q : ℝ) :
  (M = {x : ℝ | x^2 - 5 * x < 0}) →
  (M = {x : ℝ | 0 < x ∧ x < 5}) →
  (N = {x : ℝ | p < x ∧ x < 6}) →
  (M ∩ N = {x : ℝ | 2 < x ∧ x < q}) →
  p + q = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l279_27929


namespace NUMINAMATH_GPT_monotonic_intervals_and_extreme_points_l279_27983

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x

theorem monotonic_intervals_and_extreme_points (a : ℝ) (h : 1 < a) :
  ∃ x1 x2, x1 = 1 ∧ x2 = a ∧ x1 < x2 ∧ f x2 a < - (3 / 2) * x1 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_and_extreme_points_l279_27983


namespace NUMINAMATH_GPT_sum_of_distinct_FGHJ_values_l279_27921

theorem sum_of_distinct_FGHJ_values (A B C D E F G H I J K : ℕ)
  (h1: 0 ≤ A ∧ A ≤ 9)
  (h2: 0 ≤ B ∧ B ≤ 9)
  (h3: 0 ≤ C ∧ C ≤ 9)
  (h4: 0 ≤ D ∧ D ≤ 9)
  (h5: 0 ≤ E ∧ E ≤ 9)
  (h6: 0 ≤ F ∧ F ≤ 9)
  (h7: 0 ≤ G ∧ G ≤ 9)
  (h8: 0 ≤ H ∧ H ≤ 9)
  (h9: 0 ≤ I ∧ I ≤ 9)
  (h10: 0 ≤ J ∧ J ≤ 9)
  (h11: 0 ≤ K ∧ K ≤ 9)
  (h_divisibility_16: ∃ x, GHJK = x ∧ x % 16 = 0)
  (h_divisibility_9: (1 + B + C + D + E + F + G + H + I + J + K) % 9 = 0) :
  (F * G * H * J = 12 ∨ F * G * H * J = 120 ∨ F * G * H * J = 448) →
  (12 + 120 + 448 = 580) := 
by sorry

end NUMINAMATH_GPT_sum_of_distinct_FGHJ_values_l279_27921


namespace NUMINAMATH_GPT_find_coordinates_of_C_l279_27988

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 4, y := -1, z := 2 }
def B : Point := { x := 2, y := -3, z := 0 }

def satisfies_condition (C : Point) : Prop :=
  (C.x - B.x, C.y - B.y, C.z - B.z) = (2 * (A.x - C.x), 2 * (A.y - C.y), 2 * (A.z - C.z))

theorem find_coordinates_of_C (C : Point) (h : satisfies_condition C) : C = { x := 10/3, y := -5/3, z := 4/3 } :=
  sorry -- Proof is omitted as requested

end NUMINAMATH_GPT_find_coordinates_of_C_l279_27988


namespace NUMINAMATH_GPT_solve_rebus_l279_27942

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end NUMINAMATH_GPT_solve_rebus_l279_27942


namespace NUMINAMATH_GPT_f_3_1_plus_f_3_4_l279_27979

def f (a b : ℕ) : ℚ :=
  if a + b < 5 then (a * b - a + 4) / (2 * a)
  else (a * b - b - 5) / (-2 * b)

theorem f_3_1_plus_f_3_4 :
  f 3 1 + f 3 4 = 7 / 24 :=
by
  sorry

end NUMINAMATH_GPT_f_3_1_plus_f_3_4_l279_27979


namespace NUMINAMATH_GPT_weight_of_b_l279_27946

variable (A B C : ℕ)

theorem weight_of_b 
  (h1 : A + B + C = 180) 
  (h2 : A + B = 140) 
  (h3 : B + C = 100) :
  B = 60 :=
sorry

end NUMINAMATH_GPT_weight_of_b_l279_27946


namespace NUMINAMATH_GPT_distance_between_meeting_points_is_48_l279_27952

noncomputable def distance_between_meeting_points 
    (d : ℝ) -- total distance between points A and B
    (first_meeting_from_B : ℝ)   -- distance of the first meeting point from B
    (second_meeting_from_A : ℝ) -- distance of the second meeting point from A
    (second_meeting_from_B : ℝ) : ℝ :=
    (second_meeting_from_B - first_meeting_from_B)

theorem distance_between_meeting_points_is_48 
    (d : ℝ)
    (hm1 : first_meeting_from_B = 108)
    (hm2 : second_meeting_from_A = 84) 
    (hm3 : second_meeting_from_B = d - 24) :
    distance_between_meeting_points d first_meeting_from_B second_meeting_from_A second_meeting_from_B = 48 := by
  sorry

end NUMINAMATH_GPT_distance_between_meeting_points_is_48_l279_27952


namespace NUMINAMATH_GPT_sum_polynomial_coefficients_l279_27977

theorem sum_polynomial_coefficients :
  let a := 1
  let a_sum := -2
  (2009 * a + a_sum) = 2007 :=
by
  sorry

end NUMINAMATH_GPT_sum_polynomial_coefficients_l279_27977


namespace NUMINAMATH_GPT_find_k_l279_27969

theorem find_k (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_parabola_A : y₁ = x₁^2)
  (h_parabola_B : y₂ = x₂^2)
  (h_line_A : y₁ = x₁ - k)
  (h_line_B : y₂ = x₂ - k)
  (h_midpoint : (y₁ + y₂) / 2 = 1) 
  (h_sum_x : x₁ + x₂ = 1) :
  k = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_find_k_l279_27969


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l279_27989

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l279_27989


namespace NUMINAMATH_GPT_sum_of_numbers_l279_27973

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l279_27973


namespace NUMINAMATH_GPT_problem_conditions_imply_options_l279_27937

theorem problem_conditions_imply_options (a b : ℝ) 
  (h1 : a + 1 > b) 
  (h2 : b > 2 / a) 
  (h3 : 2 / a > 0) : 
  (a = 2 ∧ a + 1 > 2 / a ∧ b > 2 / 2) ∨
  (a = 1 → a + 1 ≤ 2 / a) ∨
  (b = 1 → ∃ a, a > 1 ∧ a + 1 > 1 ∧ 1 > 2 / a) ∨
  (a * b = 1 → ab ≤ 2) := 
sorry

end NUMINAMATH_GPT_problem_conditions_imply_options_l279_27937


namespace NUMINAMATH_GPT_smallest_m_for_probability_l279_27939

-- Define the conditions in Lean
def nonWithInTwoUnits (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 2 ∧ abs (y - z) ≥ 2 ∧ abs (z - x) ≥ 2

def probabilityCondition (m : ℝ) : Prop :=
  (m - 4)^3 / m^3 > 2/3

-- The theorem statement
theorem smallest_m_for_probability : ∃ m : ℕ, 0 < m ∧ (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ m ∧ 0 ≤ y ∧ y ≤ m ∧ 0 ≤ z ∧ z ≤ m → nonWithInTwoUnits x y z) → probabilityCondition m ∧ m = 14 :=
by sorry

end NUMINAMATH_GPT_smallest_m_for_probability_l279_27939


namespace NUMINAMATH_GPT_percentage_of_mothers_l279_27953

open Real

-- Define the constants based on the conditions provided.
def P : ℝ := sorry -- Total number of parents surveyed
def M : ℝ := sorry -- Number of mothers
def F : ℝ := sorry -- Number of fathers

-- The equations derived from the conditions.
axiom condition1 : M + F = P
axiom condition2 : (1/8)*M + (1/4)*F = 17.5/100 * P

-- The proof goal: to show the percentage of mothers.
theorem percentage_of_mothers :
  M / P = 3 / 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_mothers_l279_27953


namespace NUMINAMATH_GPT_sum_opposite_numbers_correct_opposite_sum_numbers_correct_l279_27905

def opposite (x : Int) : Int := -x

def sum_opposite_numbers (a b : Int) : Int := opposite a + opposite b

def opposite_sum_numbers (a b : Int) : Int := opposite (a + b)

theorem sum_opposite_numbers_correct (a b : Int) : sum_opposite_numbers (-6) 4 = 2 := 
by sorry

theorem opposite_sum_numbers_correct (a b : Int) : opposite_sum_numbers (-6) 4 = 2 := 
by sorry

end NUMINAMATH_GPT_sum_opposite_numbers_correct_opposite_sum_numbers_correct_l279_27905


namespace NUMINAMATH_GPT_infinite_series_sum_l279_27993

theorem infinite_series_sum : (∑' n : ℕ, if n % 3 = 0 then 1 / (3 * 2^(((n - n % 3) / 3) + 1)) 
                                 else if n % 3 = 1 then -1 / (6 * 2^(((n - n % 3) / 3)))
                                 else -1 / (12 * 2^(((n - n % 3) / 3)))) = 1 / 72 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l279_27993


namespace NUMINAMATH_GPT_Rachel_homework_difference_l279_27996

theorem Rachel_homework_difference (m r : ℕ) (hm : m = 8) (hr : r = 14) : r - m = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Rachel_homework_difference_l279_27996


namespace NUMINAMATH_GPT_dodecahedron_decagon_area_sum_l279_27920

theorem dodecahedron_decagon_area_sum {a b c : ℕ} (h1 : Nat.Coprime a c) (h2 : b ≠ 0) (h3 : ¬ ∃ p : ℕ, p.Prime ∧ p * p ∣ b) 
  (area_eq : (5 + 5 * Real.sqrt 5) / 4 = (a * Real.sqrt b) / c) : a + b + c = 14 :=
sorry

end NUMINAMATH_GPT_dodecahedron_decagon_area_sum_l279_27920


namespace NUMINAMATH_GPT_mean_proportional_AC_is_correct_l279_27912

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end NUMINAMATH_GPT_mean_proportional_AC_is_correct_l279_27912


namespace NUMINAMATH_GPT_domain_shift_l279_27958

theorem domain_shift (f : ℝ → ℝ) (h : ∀ (x : ℝ), (-2 < x ∧ x < 2) → (f (x + 2) = f x)) :
  ∀ (y : ℝ), (3 < y ∧ y < 7) ↔ (y - 3 < 4 ∧ y - 3 > -2) :=
by
  sorry

end NUMINAMATH_GPT_domain_shift_l279_27958


namespace NUMINAMATH_GPT_circle_tangent_to_directrix_and_yaxis_on_parabola_l279_27925

noncomputable def circle1_eq (x y : ℝ) := (x - 1)^2 + (y - 1 / 2)^2 = 1
noncomputable def circle2_eq (x y : ℝ) := (x + 1)^2 + (y - 1 / 2)^2 = 1

theorem circle_tangent_to_directrix_and_yaxis_on_parabola :
  ∀ (x y : ℝ), (x^2 = 2 * y) → 
  ((y = -1 / 2 → circle1_eq x y) ∨ (y = -1 / 2 → circle2_eq x y)) :=
by
  intro x y h_parabola
  sorry

end NUMINAMATH_GPT_circle_tangent_to_directrix_and_yaxis_on_parabola_l279_27925


namespace NUMINAMATH_GPT_sequences_converge_and_find_limits_l279_27997

theorem sequences_converge_and_find_limits (x y : ℕ → ℝ)
  (h1 : x 1 = 1)
  (h2 : y 1 = Real.sqrt 3)
  (h3 : ∀ n : ℕ, x (n + 1) * y (n + 1) = x n)
  (h4 : ∀ n : ℕ, x (n + 1)^2 + y n = 2) :
  ∃ (Lx Ly : ℝ), (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |x n - Lx| < ε) ∧ 
                  (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |y n - Ly| < ε) ∧ 
                  Lx = 0 ∧ 
                  Ly = 2 := 
sorry

end NUMINAMATH_GPT_sequences_converge_and_find_limits_l279_27997
