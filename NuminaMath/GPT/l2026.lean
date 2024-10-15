import Mathlib

namespace NUMINAMATH_GPT_proportion_solution_l2026_202663

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := 
by 
suffices h₀ : x = 6 / 5 by sorry
suffices h₁ : 6 / 5 = 1.2 by sorry
-- Proof steps go here
sorry

end NUMINAMATH_GPT_proportion_solution_l2026_202663


namespace NUMINAMATH_GPT_tommy_gum_given_l2026_202674

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tommy_gum_given_l2026_202674


namespace NUMINAMATH_GPT_mrs_hilt_bakes_loaves_l2026_202613

theorem mrs_hilt_bakes_loaves :
  let total_flour := 5
  let flour_per_loaf := 2.5
  (total_flour / flour_per_loaf) = 2 := 
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_bakes_loaves_l2026_202613


namespace NUMINAMATH_GPT_infinitely_many_composite_values_l2026_202643

theorem infinitely_many_composite_values (k m : ℕ) 
  (h_k : k ≥ 2) : 
  ∃ n : ℕ, n = 4 * k^4 ∧ ∀ m : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ m^4 + n = x * y :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_composite_values_l2026_202643


namespace NUMINAMATH_GPT_star_value_l2026_202699

variable (a b : ℤ)
noncomputable def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem star_value
  (h1 : a + b = 11)
  (h2 : a * b = 24)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  star a b = 11 / 24 := by
  sorry

end NUMINAMATH_GPT_star_value_l2026_202699


namespace NUMINAMATH_GPT_olivia_total_payment_l2026_202600

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end NUMINAMATH_GPT_olivia_total_payment_l2026_202600


namespace NUMINAMATH_GPT_product_of_odd_implies_sum_is_odd_l2026_202662

theorem product_of_odd_implies_sum_is_odd (a b c : ℤ) (h : a * b * c % 2 = 1) : (a + b + c) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_product_of_odd_implies_sum_is_odd_l2026_202662


namespace NUMINAMATH_GPT_rectangle_width_to_length_ratio_l2026_202676

theorem rectangle_width_to_length_ratio {w : ℕ} 
  (h1 : ∀ (l : ℕ), l = 10)
  (h2 : ∀ (p : ℕ), p = 32)
  (h3 : ∀ (P : ℕ), P = 2 * 10 + 2 * w) :
  (w : ℚ) / 10 = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_to_length_ratio_l2026_202676


namespace NUMINAMATH_GPT_Nicky_time_before_catchup_l2026_202628

-- Define the given speeds and head start time as constants
def v_C : ℕ := 5 -- Cristina's speed in meters per second
def v_N : ℕ := 3 -- Nicky's speed in meters per second
def t_H : ℕ := 12 -- Head start in seconds

-- Define the running time until catch up
def time_Nicky_run : ℕ := t_H + (36 / (v_C - v_N))

-- Prove that the time Nicky has run before Cristina catches up to him is 30 seconds
theorem Nicky_time_before_catchup : time_Nicky_run = 30 :=
by
  -- Add the steps for the proof
  sorry

end NUMINAMATH_GPT_Nicky_time_before_catchup_l2026_202628


namespace NUMINAMATH_GPT_max_planes_15_points_l2026_202632

theorem max_planes_15_points (P : Finset (Fin 15)) (hP : ∀ (p1 p2 p3 : Fin 15), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3) :
  P.card = 15 → (∃ planes : Finset (Finset (Fin 15)), planes.card = 455) := by
  sorry

end NUMINAMATH_GPT_max_planes_15_points_l2026_202632


namespace NUMINAMATH_GPT_mean_of_xyz_l2026_202623

theorem mean_of_xyz (x y z : ℝ) (h1 : 9 * x + 3 * y - 5 * z = -4) (h2 : 5 * x + 2 * y - 2 * z = 13) : 
  (x + y + z) / 3 = 10 := 
sorry

end NUMINAMATH_GPT_mean_of_xyz_l2026_202623


namespace NUMINAMATH_GPT_percentage_of_students_who_speak_lies_l2026_202685

theorem percentage_of_students_who_speak_lies
  (T : ℝ)    -- percentage of students who speak the truth
  (I : ℝ)    -- percentage of students who speak both truth and lies
  (U : ℝ)    -- probability of a randomly selected student speaking the truth or lies
  (H_T : T = 0.3)
  (H_I : I = 0.1)
  (H_U : U = 0.4) :
  ∃ (L : ℝ), L = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_who_speak_lies_l2026_202685


namespace NUMINAMATH_GPT_symmetry_implies_value_l2026_202615

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem symmetry_implies_value :
  (∀ (x : ℝ), ∃ (k : ℤ), ω * x - Real.pi / 3 = k * Real.pi + Real.pi / 2) →
  (∀ (x : ℝ), ∃ (k : ℤ), 2 * x + φ = k * Real.pi) →
  0 < φ → φ < Real.pi →
  ω = 2 →
  φ = Real.pi / 6 →
  g (Real.pi / 3) φ = -Real.sqrt 3 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  exact sorry

end NUMINAMATH_GPT_symmetry_implies_value_l2026_202615


namespace NUMINAMATH_GPT_groceries_delivered_l2026_202679

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end NUMINAMATH_GPT_groceries_delivered_l2026_202679


namespace NUMINAMATH_GPT_solution_interval_l2026_202606

theorem solution_interval (x : ℝ) : 
  (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) ∨ (7 < x) ↔ 
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0) := sorry

end NUMINAMATH_GPT_solution_interval_l2026_202606


namespace NUMINAMATH_GPT_car_not_sold_probability_l2026_202681

theorem car_not_sold_probability (a b : ℕ) (h : a = 5) (k : b = 6) : (b : ℚ) / (a + b : ℚ) = 6 / 11 :=
  by
    rw [h, k]
    norm_num

end NUMINAMATH_GPT_car_not_sold_probability_l2026_202681


namespace NUMINAMATH_GPT_inequality1_inequality2_l2026_202621

variable (a b c d : ℝ)

theorem inequality1 : 
  (a + c)^2 * (b + d)^2 ≥ 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) :=
  sorry

theorem inequality2 : 
  (a + c)^2 * (b + d)^2 ≥ 4 * b * c * (c * d + d * a + a * b) :=
  sorry

end NUMINAMATH_GPT_inequality1_inequality2_l2026_202621


namespace NUMINAMATH_GPT_Alissa_presents_equal_9_l2026_202636

def Ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0
def Alissa_presents := Ethan_presents - difference

theorem Alissa_presents_equal_9 : Alissa_presents = 9.0 := 
by sorry

end NUMINAMATH_GPT_Alissa_presents_equal_9_l2026_202636


namespace NUMINAMATH_GPT_range_of_m_if_p_range_of_m_if_p_and_q_l2026_202620

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  (3 - m > m - 1) ∧ (m - 1 > 0)

def proposition_q (m : ℝ) : Prop :=
  m^2 - 9 / 4 < 0

theorem range_of_m_if_p (m : ℝ) (hp : proposition_p m) : 1 < m ∧ m < 2 :=
  sorry

theorem range_of_m_if_p_and_q (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : 1 < m ∧ m < 3 / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_m_if_p_range_of_m_if_p_and_q_l2026_202620


namespace NUMINAMATH_GPT_plan1_maximizes_B_winning_probability_l2026_202635

open BigOperators

-- Definitions for the conditions
def prob_A_wins : ℚ := 3/4
def prob_B_wins : ℚ := 1/4

-- Plan 1 probabilities
def prob_B_win_2_0 : ℚ := prob_B_wins^2
def prob_B_win_2_1 : ℚ := (Nat.choose 2 1) * prob_B_wins * prob_A_wins * prob_B_wins
def prob_B_win_plan1 : ℚ := prob_B_win_2_0 + prob_B_win_2_1

-- Plan 2 probabilities
def prob_B_win_3_0 : ℚ := prob_B_wins^3
def prob_B_win_3_1 : ℚ := (Nat.choose 3 1) * prob_B_wins^2 * prob_A_wins * prob_B_wins
def prob_B_win_3_2 : ℚ := (Nat.choose 4 2) * prob_B_wins^2 * prob_A_wins^2 * prob_B_wins
def prob_B_win_plan2 : ℚ := prob_B_win_3_0 + prob_B_win_3_1 + prob_B_win_3_2

-- Theorem statement
theorem plan1_maximizes_B_winning_probability :
  prob_B_win_plan1 > prob_B_win_plan2 :=
by
  sorry

end NUMINAMATH_GPT_plan1_maximizes_B_winning_probability_l2026_202635


namespace NUMINAMATH_GPT_save_water_negate_l2026_202668

/-- If saving 30cm^3 of water is denoted as +30cm^3, then wasting 10cm^3 of water is denoted as -10cm^3. -/
theorem save_water_negate :
  (∀ (save_waste : ℤ → ℤ), save_waste 30 = 30 → save_waste (-10) = -10) :=
by
  sorry

end NUMINAMATH_GPT_save_water_negate_l2026_202668


namespace NUMINAMATH_GPT_problem_statement_l2026_202639

noncomputable def x : ℝ := sorry -- Let x be a real number satisfying the condition

theorem problem_statement (x_real_cond : x + 1/x = 3) : 
  (x^12 - 7*x^8 + 2*x^4) = 44387*x - 15088 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2026_202639


namespace NUMINAMATH_GPT_find_payment_y_l2026_202669

variable (X Y : Real)

axiom h1 : X + Y = 570
axiom h2 : X = 1.2 * Y

theorem find_payment_y : Y = 570 / 2.2 := by
  sorry

end NUMINAMATH_GPT_find_payment_y_l2026_202669


namespace NUMINAMATH_GPT_total_shaded_area_l2026_202695

theorem total_shaded_area (S T : ℝ) (h1 : 16 / S = 4) (h2 : S / T = 4) : 
    S^2 + 16 * T^2 = 32 := 
by {
    sorry
}

end NUMINAMATH_GPT_total_shaded_area_l2026_202695


namespace NUMINAMATH_GPT_desired_average_score_is_correct_l2026_202645

-- Conditions
def average_score_9_tests : ℕ := 82
def score_10th_test : ℕ := 92

-- Desired average score
def desired_average_score : ℕ := 83

-- Total score for 10 tests
def total_score_10_tests (avg9 : ℕ) (score10 : ℕ) : ℕ :=
  9 * avg9 + score10

-- Main theorem statement to prove
theorem desired_average_score_is_correct :
  total_score_10_tests average_score_9_tests score_10th_test / 10 = desired_average_score :=
by
  sorry

end NUMINAMATH_GPT_desired_average_score_is_correct_l2026_202645


namespace NUMINAMATH_GPT_part1_part2_l2026_202624

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a * sin A * sin B + b * cos A^2 = 4 / 3 * a)
variable (h2 : c^2 = a^2 + (1 / 4) * b^2)

theorem part1 : b = 4 / 3 * a := by sorry

theorem part2 : C = π / 3 := by sorry

end NUMINAMATH_GPT_part1_part2_l2026_202624


namespace NUMINAMATH_GPT_carl_highway_miles_l2026_202664

theorem carl_highway_miles
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (city_miles : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_cost : ℕ)
  (h1 : city_mpg = 30)
  (h2 : highway_mpg = 40)
  (h3 : city_miles = 60)
  (h4 : gas_cost_per_gallon = 3)
  (h5 : total_cost = 42)
  : (total_cost - (city_miles / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon * highway_mpg = 480 := 
by
  sorry

end NUMINAMATH_GPT_carl_highway_miles_l2026_202664


namespace NUMINAMATH_GPT_unique_solution_values_l2026_202694

theorem unique_solution_values (x y a : ℝ) :
  (∀ x y a, x^2 + y^2 + 2 * x ≤ 1 ∧ x - y + a = 0) → (a = -1 ∨ a = 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_unique_solution_values_l2026_202694


namespace NUMINAMATH_GPT_largest_possible_n_l2026_202659

theorem largest_possible_n (k : ℕ) (hk : k > 0) : ∃ n, n = 3 * k - 1 := 
  sorry

end NUMINAMATH_GPT_largest_possible_n_l2026_202659


namespace NUMINAMATH_GPT_equidistant_points_quadrants_l2026_202678

open Real

theorem equidistant_points_quadrants : 
  ∀ x y : ℝ, 
    (4 * x + 6 * y = 24) → (|x| = |y|) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y)) :=
by
  sorry

end NUMINAMATH_GPT_equidistant_points_quadrants_l2026_202678


namespace NUMINAMATH_GPT_inequality_holds_l2026_202617

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > c) : (a - b) * |c - b| > 0 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l2026_202617


namespace NUMINAMATH_GPT_gray_area_l2026_202603

-- Given conditions
def rect1_length : ℕ := 8
def rect1_width : ℕ := 10
def rect2_length : ℕ := 12
def rect2_width : ℕ := 9
def black_area : ℕ := 37

-- Define areas based on conditions
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width
def white_area : ℕ := area_rect1 - black_area

-- Theorem to prove the area of the gray part
theorem gray_area : area_rect2 - white_area = 65 :=
by
  sorry

end NUMINAMATH_GPT_gray_area_l2026_202603


namespace NUMINAMATH_GPT_problem_l2026_202648

theorem problem (f : ℕ → ℝ) 
  (h_def : ∀ x, f x = Real.cos (x * Real.pi / 3)) 
  (h_period : ∀ x, f (x + 6) = f x) : 
  (Finset.sum (Finset.range 2018) f) = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l2026_202648


namespace NUMINAMATH_GPT_find_k_value_l2026_202622

noncomputable def solve_for_k (k : ℚ) : Prop :=
  ∃ x : ℚ, (x = 1) ∧ (3 * x + (2 * k - 1) = x - 6 * (3 * k + 2))

theorem find_k_value : solve_for_k (-13 / 20) :=
  sorry

end NUMINAMATH_GPT_find_k_value_l2026_202622


namespace NUMINAMATH_GPT_min_fence_posts_needed_l2026_202634

-- Definitions for the problem conditions
def area_length : ℕ := 72
def regular_side : ℕ := 30
def sloped_side : ℕ := 33
def interval : ℕ := 15

-- The property we want to prove
theorem min_fence_posts_needed : 3 * ((sloped_side + interval - 1) / interval) + 3 * ((regular_side + interval - 1) / interval) = 6 := 
by
  sorry

end NUMINAMATH_GPT_min_fence_posts_needed_l2026_202634


namespace NUMINAMATH_GPT_triangle_area_l2026_202644

theorem triangle_area (X Y Z : ℝ) (r R : ℝ)
  (h1 : r = 7)
  (h2 : R = 25)
  (h3 : 2 * Real.cos Y = Real.cos X + Real.cos Z) :
  ∃ (p q r : ℕ), (p * Real.sqrt q / r = 133) ∧ (p + q + r = 135) :=
  sorry

end NUMINAMATH_GPT_triangle_area_l2026_202644


namespace NUMINAMATH_GPT_JuanitaDessertCost_l2026_202666

-- Define costs as constants
def brownieCost : ℝ := 2.50
def regularScoopCost : ℝ := 1.00
def premiumScoopCost : ℝ := 1.25
def deluxeScoopCost : ℝ := 1.50
def syrupCost : ℝ := 0.50
def nutsCost : ℝ := 1.50
def whippedCreamCost : ℝ := 0.75
def cherryCost : ℝ := 0.25

-- Define the total cost calculation
def totalCost : ℝ := brownieCost + regularScoopCost + premiumScoopCost +
                     deluxeScoopCost + syrupCost + syrupCost + nutsCost + whippedCreamCost + cherryCost

-- The proof problem: Prove that total cost equals $9.75
theorem JuanitaDessertCost : totalCost = 9.75 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_JuanitaDessertCost_l2026_202666


namespace NUMINAMATH_GPT_cone_volume_l2026_202655

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end NUMINAMATH_GPT_cone_volume_l2026_202655


namespace NUMINAMATH_GPT_inequality_proof_l2026_202650

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3) (x y z : ℝ) (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (hxyz_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l2026_202650


namespace NUMINAMATH_GPT_tom_initial_game_count_zero_l2026_202601

theorem tom_initial_game_count_zero
  (batman_game_cost superman_game_cost total_expenditure initial_game_count : ℝ)
  (h_batman_cost : batman_game_cost = 13.60)
  (h_superman_cost : superman_game_cost = 5.06)
  (h_total_expenditure : total_expenditure = 18.66)
  (h_initial_game_cost : initial_game_count = total_expenditure - (batman_game_cost + superman_game_cost)) :
  initial_game_count = 0 :=
by
  sorry

end NUMINAMATH_GPT_tom_initial_game_count_zero_l2026_202601


namespace NUMINAMATH_GPT_find_q_l2026_202629

theorem find_q (p q : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) (hp_congr : 5 * p ≡ 3 [MOD 4]) (hq_def : q = 13 * p + 2) : q = 41 := 
sorry

end NUMINAMATH_GPT_find_q_l2026_202629


namespace NUMINAMATH_GPT_jason_pears_count_l2026_202638

theorem jason_pears_count 
  (initial_pears : ℕ)
  (given_to_keith : ℕ)
  (received_from_mike : ℕ)
  (final_pears : ℕ)
  (h_initial : initial_pears = 46)
  (h_given : given_to_keith = 47)
  (h_received : received_from_mike = 12)
  (h_final : final_pears = 12) :
  initial_pears - given_to_keith + received_from_mike = final_pears :=
sorry

end NUMINAMATH_GPT_jason_pears_count_l2026_202638


namespace NUMINAMATH_GPT_parabola_x_intercepts_count_l2026_202646

theorem parabola_x_intercepts_count : 
  let equation := fun y : ℝ => -3 * y^2 + 2 * y + 3
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = equation y :=
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercepts_count_l2026_202646


namespace NUMINAMATH_GPT_old_man_coins_l2026_202683

theorem old_man_coins (x y : ℕ) (h : x ≠ y) (h_condition : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := 
sorry

end NUMINAMATH_GPT_old_man_coins_l2026_202683


namespace NUMINAMATH_GPT_total_cookies_l2026_202637

-- Define the number of bags and cookies per bag
def num_bags : Nat := 37
def cookies_per_bag : Nat := 19

-- The theorem stating the total number of cookies
theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_GPT_total_cookies_l2026_202637


namespace NUMINAMATH_GPT_tina_took_away_2_oranges_l2026_202619

-- Definition of the problem
def oranges_taken_away (x : ℕ) : Prop :=
  let original_oranges := 5
  let tangerines_left := 17 - 10 
  let oranges_left := original_oranges - x
  tangerines_left = oranges_left + 4 

-- The statement that needs to be proven
theorem tina_took_away_2_oranges : oranges_taken_away 2 :=
  sorry

end NUMINAMATH_GPT_tina_took_away_2_oranges_l2026_202619


namespace NUMINAMATH_GPT_combined_moment_l2026_202671

-- Definitions based on given conditions
variables (P Q Z : ℝ) -- Positions of the points and center of mass
variables (p q : ℝ) -- Masses of the points
variables (Mom_s : ℝ → ℝ) -- Moment function relative to axis s

-- Given:
-- 1. Positions P and Q with masses p and q respectively
-- 2. Combined point Z with total mass p + q
-- 3. Moments relative to the axis s: Mom_s P and Mom_s Q
-- To Prove: Moment of the combined point Z relative to axis s
-- is the sum of the moments of P and Q relative to the same axis

theorem combined_moment (hZ : Z = (P * p + Q * q) / (p + q)) :
  Mom_s Z = Mom_s P + Mom_s Q :=
sorry

end NUMINAMATH_GPT_combined_moment_l2026_202671


namespace NUMINAMATH_GPT_average_percentage_l2026_202686

theorem average_percentage (s1 s2 : ℕ) (a1 a2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : a1 = 70) (h3 : s2 = 10) (h4 : a2 = 90) (h5 : n = 25)
  : ((s1 * a1 + s2 * a2) / n : ℕ) = 78 :=
by
  -- We include sorry to skip the proof part.
  sorry

end NUMINAMATH_GPT_average_percentage_l2026_202686


namespace NUMINAMATH_GPT_simplify_expression_l2026_202658

theorem simplify_expression : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2026_202658


namespace NUMINAMATH_GPT_largest_result_l2026_202633

theorem largest_result (a b c : ℕ) (h1 : a = 0 / 100) (h2 : b = 0 * 100) (h3 : c = 100 - 0) : 
  c > a ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_largest_result_l2026_202633


namespace NUMINAMATH_GPT_kelly_grade_correct_l2026_202670

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end NUMINAMATH_GPT_kelly_grade_correct_l2026_202670


namespace NUMINAMATH_GPT_neutralization_reaction_l2026_202693

/-- When combining 2 moles of CH3COOH and 2 moles of NaOH, 2 moles of H2O are formed
    given the balanced chemical reaction CH3COOH + NaOH → CH3COONa + H2O 
    with a molar ratio of 1:1:1 (CH3COOH:NaOH:H2O). -/
theorem neutralization_reaction
  (mCH3COOH : ℕ) (mNaOH : ℕ) :
  (mCH3COOH = 2) → (mNaOH = 2) → (mCH3COOH = mNaOH) →
  ∃ mH2O : ℕ, mH2O = 2 :=
by intros; existsi 2; sorry

end NUMINAMATH_GPT_neutralization_reaction_l2026_202693


namespace NUMINAMATH_GPT_jenny_total_distance_seven_hops_l2026_202682

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem jenny_total_distance_seven_hops :
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  sum_geometric_series a r n = (14197 / 16384 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_jenny_total_distance_seven_hops_l2026_202682


namespace NUMINAMATH_GPT_value_of_f_2011_l2026_202602

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_2011 (h_even : ∀ x : ℝ, f x = f (-x))
                       (h_sym : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f (2 + x) = f (2 - x))
                       (h_def : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f x = 2^x) : 
  f 2011 = 1 / 2 := 
sorry

end NUMINAMATH_GPT_value_of_f_2011_l2026_202602


namespace NUMINAMATH_GPT_inequality_solution_sets_equivalence_l2026_202611

theorem inequality_solution_sets_equivalence
  (a b : ℝ)
  (h1 : (∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0)) :
  (∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ bx^2 - 5 * x + a > 0) :=
  sorry

end NUMINAMATH_GPT_inequality_solution_sets_equivalence_l2026_202611


namespace NUMINAMATH_GPT_brian_total_commission_l2026_202605

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end NUMINAMATH_GPT_brian_total_commission_l2026_202605


namespace NUMINAMATH_GPT_conference_duration_excluding_breaks_l2026_202656

-- Definitions based on the conditions
def total_hours : Nat := 14
def additional_minutes : Nat := 20
def break_minutes : Nat := 15

-- Total time including breaks
def total_time_minutes : Nat := total_hours * 60 + additional_minutes
-- Number of breaks
def number_of_breaks : Nat := total_hours
-- Total break time
def total_break_minutes : Nat := number_of_breaks * break_minutes

-- Proof statement
theorem conference_duration_excluding_breaks :
  total_time_minutes - total_break_minutes = 650 := by
  sorry

end NUMINAMATH_GPT_conference_duration_excluding_breaks_l2026_202656


namespace NUMINAMATH_GPT_reflected_circle_center_l2026_202667

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end NUMINAMATH_GPT_reflected_circle_center_l2026_202667


namespace NUMINAMATH_GPT_container_volume_ratio_l2026_202684

variables (A B C : ℝ)

theorem container_volume_ratio (h1 : (2 / 3) * A = (1 / 2) * B) (h2 : (1 / 2) * B = (3 / 5) * C) :
  A / C = 6 / 5 :=
sorry

end NUMINAMATH_GPT_container_volume_ratio_l2026_202684


namespace NUMINAMATH_GPT_find_ABC_sum_l2026_202697

theorem find_ABC_sum (A B C : ℤ) (h : ∀ x : ℤ, x = -3 ∨ x = 0 ∨ x = 4 → x^3 + A * x^2 + B * x + C = 0) : 
  A + B + C = -13 := 
by 
  sorry

end NUMINAMATH_GPT_find_ABC_sum_l2026_202697


namespace NUMINAMATH_GPT_largest_int_less_than_100_with_remainder_5_l2026_202625

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end NUMINAMATH_GPT_largest_int_less_than_100_with_remainder_5_l2026_202625


namespace NUMINAMATH_GPT_least_integer_to_multiple_of_3_l2026_202687

theorem least_integer_to_multiple_of_3 : ∃ n : ℕ, n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ m : ℕ, m > 0 → (527 + m) % 3 = 0 → m ≥ n :=
sorry

end NUMINAMATH_GPT_least_integer_to_multiple_of_3_l2026_202687


namespace NUMINAMATH_GPT_circle_area_in_square_centimeters_l2026_202688

theorem circle_area_in_square_centimeters (d_meters : ℤ) (h : d_meters = 8) :
  ∃ (A : ℤ), A = 160000 * Real.pi ∧ 
  A = π * (d_meters / 2) ^ 2 * 10000 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_in_square_centimeters_l2026_202688


namespace NUMINAMATH_GPT_unripe_oranges_after_days_l2026_202654

-- Definitions and Conditions
def sacks_per_day := 65
def days := 6

-- Statement to prove
theorem unripe_oranges_after_days : sacks_per_day * days = 390 := by
  sorry

end NUMINAMATH_GPT_unripe_oranges_after_days_l2026_202654


namespace NUMINAMATH_GPT_triangle_sides_inequality_triangle_sides_equality_condition_l2026_202657

theorem triangle_sides_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem triangle_sides_equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_triangle_sides_inequality_triangle_sides_equality_condition_l2026_202657


namespace NUMINAMATH_GPT_number_of_pens_l2026_202609

theorem number_of_pens (x y : ℝ) (h1 : 60 * (x + 2 * y) = 50 * (x + 3 * y)) (h2 : x = 3 * y) : 
  (60 * (x + 2 * y)) / x = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pens_l2026_202609


namespace NUMINAMATH_GPT_monotonic_increasing_implies_range_l2026_202604

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end NUMINAMATH_GPT_monotonic_increasing_implies_range_l2026_202604


namespace NUMINAMATH_GPT_arithmetic_sequence_6000th_term_l2026_202630

theorem arithmetic_sequence_6000th_term :
  ∀ (p r : ℕ), 
  (2 * p) = 2 * p → 
  (2 * p + 2 * r = 14) → 
  (14 + 2 * r = 4 * p - r) → 
  (2 * p + (6000 - 1) * 4 = 24006) :=
by 
  intros p r h h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_6000th_term_l2026_202630


namespace NUMINAMATH_GPT_f_2019_eq_2019_l2026_202692

def f : ℝ → ℝ := sorry

axiom f_pos : ∀ x, x > 0 → f x > 0
axiom f_one : f 1 = 1
axiom f_eq : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

theorem f_2019_eq_2019 : f 2019 = 2019 :=
by sorry

end NUMINAMATH_GPT_f_2019_eq_2019_l2026_202692


namespace NUMINAMATH_GPT_tangent_line_equation_inequality_range_l2026_202660

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  let x := Real.exp 1
  ∀ e : ℝ, e = Real.exp 1 → 
  ∀ y : ℝ, y = f (Real.exp 1) → 
  ∀ a b : ℝ, (y = a * Real.exp 1 + b) ∧ (a = 2) ∧ (b = -e) := sorry

theorem inequality_range (x : ℝ) (hx : x > 0) :
  (f x - 1/2 ≤ (3/2) * x^2 + a * x) → ∀ a : ℝ, a ≥ -2 := sorry

end NUMINAMATH_GPT_tangent_line_equation_inequality_range_l2026_202660


namespace NUMINAMATH_GPT_calc_expression_l2026_202641

theorem calc_expression : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l2026_202641


namespace NUMINAMATH_GPT_find_number_l2026_202665

theorem find_number (x : ℕ) (h : x + 15 = 96) : x = 81 := 
sorry

end NUMINAMATH_GPT_find_number_l2026_202665


namespace NUMINAMATH_GPT_jessica_has_three_dozens_of_red_marbles_l2026_202698

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_jessica_has_three_dozens_of_red_marbles_l2026_202698


namespace NUMINAMATH_GPT_proof_problem_l2026_202631

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem proof_problem
  (a : ℝ) (b : ℝ) (x : ℝ)
  (h₀ : 0 ≤ a)
  (h₁ : a ≤ 1 / 2)
  (h₂ : b = 1)
  (h₃ : 0 ≤ x) :
  (1 / f x) + (x / g x a b) ≥ 1 := by
    sorry

end NUMINAMATH_GPT_proof_problem_l2026_202631


namespace NUMINAMATH_GPT_debby_remaining_pictures_l2026_202696

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (deleted_pictures : ℕ)

def initial_pictures (zoo_pictures museum_pictures : ℕ) : ℕ :=
  zoo_pictures + museum_pictures

def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (initial_pictures zoo_pictures museum_pictures) - deleted_pictures

theorem debby_remaining_pictures :
  remaining_pictures 24 12 14 = 22 :=
by
  sorry

end NUMINAMATH_GPT_debby_remaining_pictures_l2026_202696


namespace NUMINAMATH_GPT_target_hit_probability_l2026_202680

theorem target_hit_probability (prob_A_hits : ℝ) (prob_B_hits : ℝ) (hA : prob_A_hits = 0.5) (hB : prob_B_hits = 0.6) :
  (1 - (1 - prob_A_hits) * (1 - prob_B_hits)) = 0.8 := 
by 
  sorry

end NUMINAMATH_GPT_target_hit_probability_l2026_202680


namespace NUMINAMATH_GPT_empty_seats_correct_l2026_202651

def children_count : ℕ := 52
def adult_count : ℕ := 29
def total_seats : ℕ := 95

theorem empty_seats_correct :
  total_seats - (children_count + adult_count) = 14 :=
by
  sorry

end NUMINAMATH_GPT_empty_seats_correct_l2026_202651


namespace NUMINAMATH_GPT_cone_base_circumference_l2026_202640

theorem cone_base_circumference 
  (r : ℝ) 
  (θ : ℝ) 
  (h₁ : r = 5) 
  (h₂ : θ = 225) : 
  (θ / 360 * 2 * Real.pi * r) = (25 * Real.pi / 4) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l2026_202640


namespace NUMINAMATH_GPT_percent_of_x_l2026_202608

-- The mathematical equivalent of the problem statement in Lean.
theorem percent_of_x (x : ℝ) (hx : 0 < x) : (x / 10 + x / 25) = 0.14 * x :=
by
  sorry

end NUMINAMATH_GPT_percent_of_x_l2026_202608


namespace NUMINAMATH_GPT_ellipse_a_plus_k_l2026_202618

theorem ellipse_a_plus_k (f1 f2 p : Real × Real) (a b h k : Real) :
  f1 = (2, 0) →
  f2 = (-2, 0) →
  p = (5, 3) →
  (∀ x y, ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) →
  a > 0 →
  b > 0 →
  h = 0 →
  k = 0 →
  a = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 →
  a + k = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ellipse_a_plus_k_l2026_202618


namespace NUMINAMATH_GPT_total_payment_mr_benson_made_l2026_202610

noncomputable def general_admission_ticket_cost : ℝ := 40
noncomputable def num_general_admission_tickets : ℕ := 10
noncomputable def num_vip_tickets : ℕ := 3
noncomputable def num_premium_tickets : ℕ := 2
noncomputable def vip_ticket_rate_increase : ℝ := 0.20
noncomputable def premium_ticket_rate_increase : ℝ := 0.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def threshold_tickets : ℕ := 10

noncomputable def vip_ticket_cost : ℝ := general_admission_ticket_cost * (1 + vip_ticket_rate_increase)
noncomputable def premium_ticket_cost : ℝ := general_admission_ticket_cost * (1 + premium_ticket_rate_increase)

noncomputable def total_general_admission_cost : ℝ := num_general_admission_tickets * general_admission_ticket_cost
noncomputable def total_vip_cost : ℝ := num_vip_tickets * vip_ticket_cost
noncomputable def total_premium_cost : ℝ := num_premium_tickets * premium_ticket_cost

noncomputable def total_tickets : ℕ := num_general_admission_tickets + num_vip_tickets + num_premium_tickets
noncomputable def tickets_exceeding_threshold : ℕ := if total_tickets > threshold_tickets then total_tickets - threshold_tickets else 0

noncomputable def discounted_vip_cost : ℝ := vip_ticket_cost * (1 - discount_rate)
noncomputable def discounted_premium_cost : ℝ := premium_ticket_cost * (1 - discount_rate)

noncomputable def total_discounted_vip_cost : ℝ :=  num_vip_tickets * discounted_vip_cost
noncomputable def total_discounted_premium_cost : ℝ := num_premium_tickets * discounted_premium_cost

noncomputable def total_cost_with_discounts : ℝ := total_general_admission_cost + total_discounted_vip_cost + total_discounted_premium_cost

theorem total_payment_mr_benson_made : total_cost_with_discounts = 650.80 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_payment_mr_benson_made_l2026_202610


namespace NUMINAMATH_GPT_find_f_2012_l2026_202616

-- Given a function f: ℤ → ℤ that satisfies the functional equation:
def functional_equation (f : ℤ → ℤ) := ∀ m n : ℤ, m + f (m + f (n + f m)) = n + f m

-- Given condition:
def f_6_is_6 (f : ℤ → ℤ) := f 6 = 6

-- We need to prove that f 2012 = -2000 under the given conditions.
theorem find_f_2012 (f : ℤ → ℤ) (hf : functional_equation f) (hf6 : f_6_is_6 f) : f 2012 = -2000 := sorry

end NUMINAMATH_GPT_find_f_2012_l2026_202616


namespace NUMINAMATH_GPT_additional_pots_produced_l2026_202653

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_additional_pots_produced_l2026_202653


namespace NUMINAMATH_GPT_polynomial_degree_is_14_l2026_202642

noncomputable def polynomial_degree (a b c d e f g h : ℝ) : ℕ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 then 14 else 0

theorem polynomial_degree_is_14 (a b c d e f g h : ℝ) (h_neq0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  polynomial_degree a b c d e f g h = 14 :=
by sorry

end NUMINAMATH_GPT_polynomial_degree_is_14_l2026_202642


namespace NUMINAMATH_GPT_cosine_identity_l2026_202690

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) :
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cosine_identity_l2026_202690


namespace NUMINAMATH_GPT_find_x_l2026_202647

theorem find_x (x : ℝ) (h : x^2 + 75 = (x - 20)^2) : x = 8.125 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2026_202647


namespace NUMINAMATH_GPT_suitable_M_unique_l2026_202652

noncomputable def is_suitable_M (M : ℝ) : Prop :=
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (1 + M ≤ a + M / (a * b)) ∨ 
  (1 + M ≤ b + M / (b * c)) ∨ 
  (1 + M ≤ c + M / (c * a))

theorem suitable_M_unique : is_suitable_M (1/2) ∧ 
  (∀ (M : ℝ), is_suitable_M M → M = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_suitable_M_unique_l2026_202652


namespace NUMINAMATH_GPT_min_value_expr_l2026_202675

open Real

theorem min_value_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ c, c = 4 * sqrt 3 - 6 ∧ ∀ (z w : ℝ), z = x ∧ w = y → (3 * z) / (3 * z + 2 * w) + w / (2 * z + w) ≥ c :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l2026_202675


namespace NUMINAMATH_GPT_blue_bead_probability_no_adjacent_l2026_202649

theorem blue_bead_probability_no_adjacent :
  let total_beads := 9
  let blue_beads := 5
  let green_beads := 3
  let red_bead := 1
  let total_permutations := Nat.factorial total_beads / (Nat.factorial blue_beads * Nat.factorial green_beads * Nat.factorial red_bead)
  let valid_arrangements := (Nat.factorial 4) / (Nat.factorial 3 * Nat.factorial 1)
  let no_adjacent_valid := 4
  let probability_no_adj := (no_adjacent_valid : ℚ) / total_permutations
  probability_no_adj = (1 : ℚ) / 126 := 
by
  sorry

end NUMINAMATH_GPT_blue_bead_probability_no_adjacent_l2026_202649


namespace NUMINAMATH_GPT_B_and_D_know_their_grades_l2026_202627

-- Define the students and their respective grades
inductive Grade : Type
| excellent : Grade
| good : Grade

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the information given in the problem regarding which student sees whose grade
def sees (s1 s2 : Student) : Prop :=
  (s1 = Student.A ∧ (s2 = Student.B ∨ s2 = Student.C)) ∨
  (s1 = Student.B ∧ s2 = Student.C) ∨
  (s1 = Student.D ∧ s2 = Student.A)

-- Define the condition that there are 2 excellent and 2 good grades
def grade_distribution (gA gB gC gD : Grade) : Prop :=
  gA ≠ gB → (gC = gA ∨ gC = gB) ∧ (gD = gA ∨ gD = gB) ∧
  (gA = Grade.excellent ∧ (gB = Grade.good ∨ gC = Grade.good ∨ gD = Grade.good)) ∧
  (gA = Grade.good ∧ (gB = Grade.excellent ∨ gC = Grade.excellent ∨ gD = Grade.excellent))

-- Student A's statement after seeing B and C's grades
def A_statement (gA gB gC : Grade) : Prop :=
  (gB = gA ∨ gC = gA) ∨ (gB ≠ gA ∧ gC ≠ gA)

-- Formal proof goal: Prove that B and D can know their own grades based on the information provided
theorem B_and_D_know_their_grades (gA gB gC gD : Grade)
  (h1 : grade_distribution gA gB gC gD)
  (h2 : A_statement gA gB gC)
  (h3 : sees Student.A Student.B)
  (h4 : sees Student.A Student.C)
  (h5 : sees Student.B Student.C)
  (h6 : sees Student.D Student.A) :
  (gB ≠ Grade.excellent ∨ gB ≠ Grade.good) ∧ (gD ≠ Grade.excellent ∨ gD ≠ Grade.good) :=
by sorry

end NUMINAMATH_GPT_B_and_D_know_their_grades_l2026_202627


namespace NUMINAMATH_GPT_proof_problem_l2026_202612

variable (a b c : ℝ)

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : ∀ x, abs (x + a) - abs (x - b) + c ≤ 10) :
  a + b + c = 10 ∧ 
  (∀ (h5 : a + b + c = 10), 
    (∃ a' b' c', a' = 11/3 ∧ b' = 8/3 ∧ c' = 11/3 ∧ 
                (∀ a'' b'' c'', a'' = a ∧ b'' = b ∧ c'' = c → 
                (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2) ≥ 8/3 ∧ 
                (1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) = 8 / 3 ))) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2026_202612


namespace NUMINAMATH_GPT_circle_radius_l2026_202626

theorem circle_radius 
  (x y : ℝ)
  (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = Real.sqrt 117 :=
by 
  sorry

end NUMINAMATH_GPT_circle_radius_l2026_202626


namespace NUMINAMATH_GPT_problem_l2026_202661

theorem problem (p q : Prop) (m : ℝ):
  (p = (m > 1)) →
  (q = (-2 ≤ m ∧ m ≤ 2)) →
  (¬q = (m < -2 ∨ m > 2)) →
  (¬(p ∧ q)) →
  (p ∨ q) →
  (¬q) →
  m > 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2026_202661


namespace NUMINAMATH_GPT_constant_term_expansion_l2026_202691

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion :
    ∀ x: ℂ, (x ≠ 0) → ∃ term: ℂ, 
    term = (-1 : ℂ) * binom 6 4 ∧ term = -15 := 
by
  intros x hx
  use (-1 : ℂ) * binom 6 4
  constructor
  · rfl
  · sorry

end NUMINAMATH_GPT_constant_term_expansion_l2026_202691


namespace NUMINAMATH_GPT_coffee_tea_soda_l2026_202677

theorem coffee_tea_soda (Pcoffee Ptea Psoda Pboth_no_soda : ℝ)
  (H1 : 0.9 = Pcoffee)
  (H2 : 0.8 = Ptea)
  (H3 : 0.7 = Psoda) :
  0.0 = Pboth_no_soda :=
  sorry

end NUMINAMATH_GPT_coffee_tea_soda_l2026_202677


namespace NUMINAMATH_GPT_erwan_spending_l2026_202672

def discount (price : ℕ) (percent : ℕ) : ℕ :=
  price - (price * percent / 100)

theorem erwan_spending (shoe_original_price : ℕ := 200) 
  (shoe_discount : ℕ := 30)
  (shirt_price : ℕ := 80)
  (num_shirts : ℕ := 2)
  (pants_price : ℕ := 150)
  (second_store_discount : ℕ := 20)
  (jacket_price : ℕ := 250)
  (tie_price : ℕ := 40)
  (hat_price : ℕ := 60)
  (watch_price : ℕ := 120)
  (wallet_price : ℕ := 49)
  (belt_price : ℕ := 35)
  (belt_discount : ℕ := 25)
  (scarf_price : ℕ := 45)
  (scarf_discount : ℕ := 10)
  (rewards_points_discount : ℕ := 5)
  (sales_tax : ℕ := 8)
  (gift_card : ℕ := 50)
  (shipping_fee : ℕ := 5)
  (num_shipping_stores : ℕ := 2) :
  ∃ total : ℕ,
    total = 85429 :=
by
  have first_store := discount shoe_original_price shoe_discount
  have second_store_total := pants_price + (shirt_price * num_shirts)
  have second_store := discount second_store_total second_store_discount
  have tie_half_price := tie_price / 2
  have hat_half_price := hat_price / 2
  have third_store := jacket_price + (tie_half_price + hat_half_price)
  have fourth_store := watch_price
  have fifth_store := discount belt_price belt_discount + discount scarf_price scarf_discount
  have subtotal := first_store + second_store + third_store + fourth_store + fifth_store
  have after_rewards_points := subtotal - (subtotal * rewards_points_discount / 100)
  have after_gift_card := after_rewards_points - gift_card
  have after_shipping_fees := after_gift_card + (shipping_fee * num_shipping_stores)
  have total := after_shipping_fees + (after_shipping_fees * sales_tax / 100)
  use total / 100 -- to match the monetary value in cents
  sorry

end NUMINAMATH_GPT_erwan_spending_l2026_202672


namespace NUMINAMATH_GPT_num_isosceles_right_triangles_in_ellipse_l2026_202689

theorem num_isosceles_right_triangles_in_ellipse
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t))
  :
  (∃ n : ℕ,
    (n = 3 ∧ a > Real.sqrt 3 * b) ∨
    (n = 1 ∧ (b < a ∧ a ≤ Real.sqrt 3 * b))
  ) :=
sorry

end NUMINAMATH_GPT_num_isosceles_right_triangles_in_ellipse_l2026_202689


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l2026_202614

open Set

theorem solve_system_of_inequalities : ∀ x : ℕ, (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1) → x ∈ ({0, 1, 2, 3} : Set ℕ) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l2026_202614


namespace NUMINAMATH_GPT_inverse_square_variation_l2026_202607

theorem inverse_square_variation (k : ℝ) (y x : ℝ) (h1: x = k / y^2) (h2: 0.25 = k / 36) : 
  x = 1 :=
by
  -- Here, you would provide further Lean code to complete the proof
  -- using the given hypothesis h1 and h2, along with some computation.
  sorry

end NUMINAMATH_GPT_inverse_square_variation_l2026_202607


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_range_of_f_in_interval_l2026_202673

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem smallest_positive_period_of_f (a : ℝ) (h : f a (π / 3) = 0) :
  ∃ T : ℝ, T = 2 * π ∧ (∀ x, f a (x + T) = f a x) :=
sorry

theorem range_of_f_in_interval (a : ℝ) (h : f a (π / 3) = 0) :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a x ∧ f a x ≤ 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_range_of_f_in_interval_l2026_202673
