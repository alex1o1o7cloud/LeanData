import Mathlib

namespace NUMINAMATH_GPT_fraction_students_say_dislike_actually_like_l978_97822

theorem fraction_students_say_dislike_actually_like (total_students : ℕ) (like_dancing_fraction : ℚ) 
  (like_dancing_say_dislike_fraction : ℚ) (dislike_dancing_say_dislike_fraction : ℚ) : 
  (∃ frac : ℚ, frac = 40.7 / 100) :=
by
  let total_students := (200 : ℕ)
  let like_dancing_fraction := (70 / 100 : ℚ)
  let like_dancing_say_dislike_fraction := (25 / 100 : ℚ)
  let dislike_dancing_say_dislike_fraction := (85 / 100 : ℚ)
  
  let total_like_dancing := total_students * like_dancing_fraction
  let total_dislike_dancing :=  total_students * (1 - like_dancing_fraction)
  let like_dancing_say_dislike := total_like_dancing * like_dancing_say_dislike_fraction
  let dislike_dancing_say_dislike := total_dislike_dancing * dislike_dancing_say_dislike_fraction
  let total_say_dislike := like_dancing_say_dislike + dislike_dancing_say_dislike
  let fraction_say_dislike_actually_like := like_dancing_say_dislike / total_say_dislike
  
  existsi fraction_say_dislike_actually_like
  sorry

end NUMINAMATH_GPT_fraction_students_say_dislike_actually_like_l978_97822


namespace NUMINAMATH_GPT_expectation_fair_coin_5_tosses_l978_97885

noncomputable def fairCoinExpectation (n : ℕ) : ℚ :=
  n * (1/2)

theorem expectation_fair_coin_5_tosses :
  fairCoinExpectation 5 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_expectation_fair_coin_5_tosses_l978_97885


namespace NUMINAMATH_GPT_Maria_bought_7_roses_l978_97877

theorem Maria_bought_7_roses
  (R : ℕ)
  (h1 : ∀ f : ℕ, 6 * f = 6 * f)
  (h2 : ∀ r : ℕ, ∃ d : ℕ, r = R ∧ d = 3)
  (h3 : 6 * R + 18 = 60) : R = 7 := by
  sorry

end NUMINAMATH_GPT_Maria_bought_7_roses_l978_97877


namespace NUMINAMATH_GPT_bucket_weight_l978_97894

theorem bucket_weight (c d : ℝ) (x y : ℝ) 
  (h1 : x + 3/4 * y = c) 
  (h2 : x + 1/3 * y = d) :
  x + 1/4 * y = (6 * d - c) / 5 := 
sorry

end NUMINAMATH_GPT_bucket_weight_l978_97894


namespace NUMINAMATH_GPT_range_of_k_for_real_roots_l978_97864

theorem range_of_k_for_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k <= 1 / 12 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_GPT_range_of_k_for_real_roots_l978_97864


namespace NUMINAMATH_GPT_isabel_homework_problems_l978_97892

theorem isabel_homework_problems (initial_problems finished_problems remaining_pages problems_per_page : ℕ) 
  (h1 : initial_problems = 72)
  (h2 : finished_problems = 32)
  (h3 : remaining_pages = 5)
  (h4 : initial_problems - finished_problems = 40)
  (h5 : 40 = remaining_pages * problems_per_page) : 
  problems_per_page = 8 := 
by sorry

end NUMINAMATH_GPT_isabel_homework_problems_l978_97892


namespace NUMINAMATH_GPT_first_player_wins_l978_97879

def initial_piles (p1 p2 : Nat) : Prop :=
  p1 = 33 ∧ p2 = 35

def winning_strategy (p1 p2 : Nat) : Prop :=
  ∃ moves : List (Nat × Nat), 
  (initial_piles p1 p2) →
  (∀ (p1' p2' : Nat), 
    (p1', p2') ∈ moves →
    p1' = 1 ∧ p2' = 1 ∨ p1' = 2 ∧ p2' = 1)

theorem first_player_wins : winning_strategy 33 35 :=
sorry

end NUMINAMATH_GPT_first_player_wins_l978_97879


namespace NUMINAMATH_GPT_area_square_15_cm_l978_97847

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area calculation for a square given the side length
def area_of_square (s : ℝ) : ℝ := s * s

-- The theorem statement translating the problem to Lean
theorem area_square_15_cm :
  area_of_square side_length = 225 :=
by
  -- We need to provide proof here, but 'sorry' is used to skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_area_square_15_cm_l978_97847


namespace NUMINAMATH_GPT_original_five_digit_number_l978_97806

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end NUMINAMATH_GPT_original_five_digit_number_l978_97806


namespace NUMINAMATH_GPT_tim_will_attend_game_probability_l978_97880

theorem tim_will_attend_game_probability :
  let P_rain := 0.60
  let P_sunny := 1 - P_rain
  let P_attends_given_rain := 0.25
  let P_attends_given_sunny := 0.70
  let P_rain_and_attends := P_rain * P_attends_given_rain
  let P_sunny_and_attends := P_sunny * P_attends_given_sunny
  (P_rain_and_attends + P_sunny_and_attends) = 0.43 :=
by
  sorry

end NUMINAMATH_GPT_tim_will_attend_game_probability_l978_97880


namespace NUMINAMATH_GPT_least_3_digit_number_l978_97859

variables (k S h t u : ℕ)

def is_3_digit_number (k : ℕ) : Prop := k ≥ 100 ∧ k < 1000

def digits_sum_eq (k h t u S : ℕ) : Prop :=
  k = 100 * h + 10 * t + u ∧ S = h + t + u

def difference_condition (h t : ℕ) : Prop :=
  t - h = 8

theorem least_3_digit_number (k S h t u : ℕ) :
  is_3_digit_number k →
  digits_sum_eq k h t u S →
  difference_condition h t →
  k * 3 < 200 →
  k = 19 * S :=
sorry

end NUMINAMATH_GPT_least_3_digit_number_l978_97859


namespace NUMINAMATH_GPT_find_last_number_l978_97849

theorem find_last_number (A B C D : ℕ) 
  (h1 : A + B + C = 18) 
  (h2 : B + C + D = 9) 
  (h3 : A + D = 13) 
  : D = 2 := by 
  sorry

end NUMINAMATH_GPT_find_last_number_l978_97849


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l978_97830

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l978_97830


namespace NUMINAMATH_GPT_sin_330_eq_neg_half_l978_97897

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_sin_330_eq_neg_half_l978_97897


namespace NUMINAMATH_GPT_complement_of_M_in_U_l978_97866

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_complement_of_M_in_U_l978_97866


namespace NUMINAMATH_GPT_arithmetic_progression_x_value_l978_97846

theorem arithmetic_progression_x_value :
  ∃ x : ℝ, (2 * x - 1) + ((5 * x + 6) - (3 * x + 4)) = (3 * x + 4) + ((3 * x + 4) - (2 * x - 1)) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_x_value_l978_97846


namespace NUMINAMATH_GPT_james_total_cost_l978_97875

def suit1 := 300
def suit2_pretail := 3 * suit1
def suit2 := suit2_pretail + 200
def total_cost := suit1 + suit2

theorem james_total_cost : total_cost = 1400 := by
  sorry

end NUMINAMATH_GPT_james_total_cost_l978_97875


namespace NUMINAMATH_GPT_find_sides_of_triangle_l978_97899

theorem find_sides_of_triangle (c : ℝ) (θ : ℝ) (h_ratio : ℝ) 
  (h_c : c = 2 * Real.sqrt 7)
  (h_theta : θ = Real.pi / 6) -- 30 degrees in radians
  (h_ratio_eq : ∃ k : ℝ, ∀ a b : ℝ, a = k ∧ b = h_ratio * k) :
  ∃ (a b : ℝ), a = 2 ∧ b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_sides_of_triangle_l978_97899


namespace NUMINAMATH_GPT_problem_l978_97882

theorem problem (a b a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ)
  (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) 
  (h4 : a₄ < a₅) (h5 : a₅ < a₆) (h6 : a₆ < a₇)
  (h7 : a₇ < a₈) (h8 : a₈ < a₉) (h9 : a₉ < a₁₀)
  (h10 : a₁₀ < a₁₁) (h11 : b₁ < b₂) (h12 : b₂ < b₃)
  (h13 : b₃ < b₄) (h14 : b₄ < b₅) (h15 : b₅ < b₆)
  (h16 : b₆ < b₇) (h17 : b₇ < b₈) (h18 : b₈ < b₉)
  (h19 : b₉ < b₁₀) (h20 : b₁₀ < b₁₁) 
  (h21 : a₁₀ + b₁₀ = a) (h22 : a₁₁ + b₁₁ = b) : 
  a = 1024 ∧ b = 2048 :=
sorry

end NUMINAMATH_GPT_problem_l978_97882


namespace NUMINAMATH_GPT_probability_of_specific_sequence_l978_97898

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem probability_of_specific_sequence :
  (1 - p)^7 * p^3 = sorry :=
by sorry

end NUMINAMATH_GPT_probability_of_specific_sequence_l978_97898


namespace NUMINAMATH_GPT_line_equation_under_transformation_l978_97834

noncomputable def T1_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

noncomputable def T2_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 3]
]

noncomputable def NM_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -2],
  ![3, 0]
]

theorem line_equation_under_transformation :
  ∀ x y : ℝ, (∃ x' y' : ℝ, NM_matrix.mulVec ![x, y] = ![x', y'] ∧ x' = y') → 3 * x + 2 * y = 0 :=
by sorry

end NUMINAMATH_GPT_line_equation_under_transformation_l978_97834


namespace NUMINAMATH_GPT_smallest_angle_of_cyclic_quadrilateral_l978_97860

theorem smallest_angle_of_cyclic_quadrilateral (angles : ℝ → ℝ) (a d : ℝ) :
  -- Conditions
  (∀ n : ℕ, angles n = a + n * d) ∧ 
  (angles 3 = 140) ∧
  (a + d + (a + 3 * d) = 180) →
  -- Conclusion
  (a = 40) :=
by sorry

end NUMINAMATH_GPT_smallest_angle_of_cyclic_quadrilateral_l978_97860


namespace NUMINAMATH_GPT_proof_problem_l978_97802

variable (α β : ℝ)

def interval_αβ : Prop := 
  α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ 
  β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)

def condition : Prop := α * Real.sin α - β * Real.sin β > 0

theorem proof_problem (h1 : interval_αβ α β) (h2 : condition α β) : α ^ 2 > β ^ 2 := 
sorry

end NUMINAMATH_GPT_proof_problem_l978_97802


namespace NUMINAMATH_GPT_monkeys_bananas_l978_97896

theorem monkeys_bananas (c₁ c₂ c₃ : ℕ) (h1 : ∀ (k₁ k₂ k₃ : ℕ), k₁ = c₁ → k₂ = c₂ → k₃ = c₃ → 4 * (k₁ / 3 + k₂ / 6 + k₃ / 18) = 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) ∧ 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) = k₁ / 6 + k₂ / 6 + k₃ / 6)
  (h2 : c₃ % 6 = 0) (h3 : 4 * (c₁ / 3 + c₂ / 6 + c₃ / 18) < 2 * (c₁ / 6 + c₂ / 3 + c₃ / 18 + 1)) :
  c₁ + c₂ + c₃ = 2352 :=
sorry

end NUMINAMATH_GPT_monkeys_bananas_l978_97896


namespace NUMINAMATH_GPT_horizontal_asymptote_exists_x_intercepts_are_roots_l978_97841

noncomputable def given_function (x : ℝ) : ℝ :=
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_exists :
  ∃ L : ℝ, ∀ x : ℝ, (∃ M : ℝ, M > 0 ∧ (∀ x > M, abs (given_function x - L) < 1)) ∧ L = 0 := 
sorry

theorem x_intercepts_are_roots :
  ∀ y, y = 0 ↔ ∃ x : ℝ, x ≠ 0 ∧ 15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5 = 0 :=
sorry

end NUMINAMATH_GPT_horizontal_asymptote_exists_x_intercepts_are_roots_l978_97841


namespace NUMINAMATH_GPT_bugs_ate_each_l978_97871

theorem bugs_ate_each : 
  ∀ (total_bugs total_flowers each_bug_flowers : ℕ), 
    total_bugs = 3 ∧ total_flowers = 6 ∧ each_bug_flowers = total_flowers / total_bugs -> each_bug_flowers = 2 := by
  sorry

end NUMINAMATH_GPT_bugs_ate_each_l978_97871


namespace NUMINAMATH_GPT_find_CB_l978_97821

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)

-- Given condition
-- D divides AB in the ratio 1:3 such that CA = a and CD = b

def D_divides_AB (A B D : V) : Prop := ∃ (k : ℝ), k = 1 / 4 ∧ A + k • (B - A) = D

theorem find_CB (CA CD : V) (A B D : V) (h1 : CA = A) (h2 : CD = B)
  (h3 : D_divides_AB A B D) : (B - A) = -3 • CA + 4 • CD :=
sorry

end NUMINAMATH_GPT_find_CB_l978_97821


namespace NUMINAMATH_GPT_lg_sum_eq_lg_double_diff_l978_97823

theorem lg_sum_eq_lg_double_diff (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_harmonic : 2 / y = 1 / x + 1 / z) : 
  Real.log (x + z) + Real.log (x - 2 * y + z) = 2 * Real.log (x - z) := 
by
  sorry

end NUMINAMATH_GPT_lg_sum_eq_lg_double_diff_l978_97823


namespace NUMINAMATH_GPT_correct_operations_result_l978_97815

theorem correct_operations_result (n : ℕ) 
  (h1 : n / 8 - 12 = 32) : (n * 8 + 12 = 2828) :=
sorry

end NUMINAMATH_GPT_correct_operations_result_l978_97815


namespace NUMINAMATH_GPT_range_subset_pos_iff_l978_97878

theorem range_subset_pos_iff (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_GPT_range_subset_pos_iff_l978_97878


namespace NUMINAMATH_GPT_wood_blocks_after_days_l978_97800

-- Defining the known conditions
def blocks_per_tree : Nat := 3
def trees_per_day : Nat := 2
def days : Nat := 5

-- Stating the theorem to prove the total number of blocks of wood after 5 days
theorem wood_blocks_after_days : blocks_per_tree * trees_per_day * days = 30 :=
by
  sorry

end NUMINAMATH_GPT_wood_blocks_after_days_l978_97800


namespace NUMINAMATH_GPT_speed_of_first_train_l978_97862

-- Definitions of the conditions
def ratio_speed (speed1 speed2 : ℝ) := speed1 / speed2 = 7 / 8
def speed_of_second_train := 400 / 4

-- The theorem we want to prove
theorem speed_of_first_train (speed2 := speed_of_second_train) (h : ratio_speed S1 speed2) :
  S1 = 87.5 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l978_97862


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l978_97889

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l978_97889


namespace NUMINAMATH_GPT_total_red_yellow_black_l978_97867

/-- Calculate the total number of red, yellow, and black shirts Gavin has,
given that he has 420 shirts in total, 85 of them are blue, and 157 are
green. -/
theorem total_red_yellow_black (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) :
  total_shirts = 420 → blue_shirts = 85 → green_shirts = 157 → 
  (total_shirts - (blue_shirts + green_shirts) = 178) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_red_yellow_black_l978_97867


namespace NUMINAMATH_GPT_mabel_tomatoes_l978_97842

theorem mabel_tomatoes (x : ℕ)
  (plant_1_bore : ℕ)
  (plant_2_bore : ℕ := x + 4)
  (total_first_two_plants : ℕ := x + plant_2_bore)
  (plant_3_bore : ℕ := 3 * total_first_two_plants)
  (plant_4_bore : ℕ := 3 * total_first_two_plants)
  (total_tomatoes : ℕ)
  (h1 : total_first_two_plants = 2 * x + 4)
  (h2 : plant_3_bore = 3 * (2 * x + 4))
  (h3 : plant_4_bore = 3 * (2 * x + 4))
  (h4 : total_tomatoes = x + plant_2_bore + plant_3_bore + plant_4_bore)
  (h5 : total_tomatoes = 140) :
   x = 8 :=
by
  sorry

end NUMINAMATH_GPT_mabel_tomatoes_l978_97842


namespace NUMINAMATH_GPT_alberto_vs_bjorn_distance_difference_l978_97863

noncomputable def alberto_distance (t : ℝ) : ℝ := (3.75 / 5) * t
noncomputable def bjorn_distance (t : ℝ) : ℝ := (3.4375 / 5) * t

theorem alberto_vs_bjorn_distance_difference :
  alberto_distance 5 - bjorn_distance 5 = 0.3125 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_alberto_vs_bjorn_distance_difference_l978_97863


namespace NUMINAMATH_GPT_total_men_employed_l978_97818

/--
A work which could be finished in 11 days was finished 3 days earlier 
after 10 more men joined. Prove that the total number of men employed 
to finish the work earlier is 37.
-/
theorem total_men_employed (x : ℕ) (h1 : 11 * x = 8 * (x + 10)) : x = 27 ∧ 27 + 10 = 37 := by
  sorry

end NUMINAMATH_GPT_total_men_employed_l978_97818


namespace NUMINAMATH_GPT_problem_statement_l978_97890

theorem problem_statement (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 4 / 5) : y - x = 500 / 9 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l978_97890


namespace NUMINAMATH_GPT_cauchy_schwarz_inequality_l978_97852

theorem cauchy_schwarz_inequality 
  (a b a1 b1 : ℝ) : ((a * a1 + b * b1) ^ 2 ≤ (a^2 + b^2) * (a1^2 + b1^2)) :=
 by sorry

end NUMINAMATH_GPT_cauchy_schwarz_inequality_l978_97852


namespace NUMINAMATH_GPT_angle_sum_acutes_l978_97850

theorem angle_sum_acutes (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_condition : |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0) : 
  α + β = π * 5/12 :=
by sorry

end NUMINAMATH_GPT_angle_sum_acutes_l978_97850


namespace NUMINAMATH_GPT_compute_n_l978_97858

theorem compute_n (avg1 avg2 avg3 avg4 avg5 : ℚ) (h1 : avg1 = 1234 ∨ avg2 = 1234 ∨ avg3 = 1234 ∨ avg4 = 1234 ∨ avg5 = 1234)
  (h2 : avg1 = 345 ∨ avg2 = 345 ∨ avg3 = 345 ∨ avg4 = 345 ∨ avg5 = 345)
  (h3 : avg1 = 128 ∨ avg2 = 128 ∨ avg3 = 128 ∨ avg4 = 128 ∨ avg5 = 128)
  (h4 : avg1 = 19 ∨ avg2 = 19 ∨ avg3 = 19 ∨ avg4 = 19 ∨ avg5 = 19)
  (h5 : avg1 = 9.5 ∨ avg2 = 9.5 ∨ avg3 = 9.5 ∨ avg4 = 9.5 ∨ avg5 = 9.5) :
  ∃ n : ℕ, n = 2014 :=
by
  sorry

end NUMINAMATH_GPT_compute_n_l978_97858


namespace NUMINAMATH_GPT_train_speed_kph_l978_97835

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kph_l978_97835


namespace NUMINAMATH_GPT_arithmetic_proof_l978_97873

def arithmetic_expression := 3889 + 12.952 - 47.95000000000027
def expected_result := 3854.002

theorem arithmetic_proof : arithmetic_expression = expected_result := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_proof_l978_97873


namespace NUMINAMATH_GPT_sqrt_inequality_l978_97826

theorem sqrt_inequality : (Real.sqrt 6 + Real.sqrt 7) > (2 * Real.sqrt 2 + Real.sqrt 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_inequality_l978_97826


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l978_97861

-- Let i be the imaginary unit
def i : ℂ := Complex.I

theorem sum_of_powers_of_i : (1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 + i^10) = i := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_i_l978_97861


namespace NUMINAMATH_GPT_power_of_two_plus_one_div_by_power_of_three_l978_97814

theorem power_of_two_plus_one_div_by_power_of_three (n : ℕ) : 3^(n + 1) ∣ (2^(3^n) + 1) :=
sorry

end NUMINAMATH_GPT_power_of_two_plus_one_div_by_power_of_three_l978_97814


namespace NUMINAMATH_GPT_exists_good_set_l978_97839

variable (M : Set ℕ) [DecidableEq M] [Fintype M]
variable (f : Finset ℕ → ℕ)

theorem exists_good_set :
  ∃ T : Finset ℕ, T.card = 10 ∧ (∀ k ∈ T, f (T.erase k) ≠ k) := by
  sorry

end NUMINAMATH_GPT_exists_good_set_l978_97839


namespace NUMINAMATH_GPT_abs_eq_zero_sum_is_neg_two_l978_97807

theorem abs_eq_zero_sum_is_neg_two (x y : ℝ) (h : |x - 1| + |y + 3| = 0) : x + y = -2 := 
by 
  sorry

end NUMINAMATH_GPT_abs_eq_zero_sum_is_neg_two_l978_97807


namespace NUMINAMATH_GPT_number_of_squares_l978_97856

def draws_88_lines (lines: ℕ) : Prop := lines = 88
def draws_triangles (triangles: ℕ) : Prop := triangles = 12
def draws_pentagons (pentagons: ℕ) : Prop := pentagons = 4

theorem number_of_squares (triangles pentagons sq_sides: ℕ) (h1: draws_88_lines (triangles * 3 + pentagons * 5 + sq_sides * 4))
    (h2: draws_triangles triangles) (h3: draws_pentagons pentagons) : sq_sides = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_squares_l978_97856


namespace NUMINAMATH_GPT_each_person_towel_day_l978_97848

def total_people (families : ℕ) (members_per_family : ℕ) : ℕ :=
  families * members_per_family

def total_towels (loads : ℕ) (towels_per_load : ℕ) : ℕ :=
  loads * towels_per_load

def towels_per_day (total_towels : ℕ) (days : ℕ) : ℕ :=
  total_towels / days

def towels_per_person_per_day (towels_per_day : ℕ) (total_people : ℕ) : ℕ :=
  towels_per_day / total_people

theorem each_person_towel_day
  (families : ℕ) (members_per_family : ℕ) (days : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h_family : families = 3) (h_members : members_per_family = 4) (h_days : days = 7)
  (h_loads : loads = 6) (h_towels_per_load : towels_per_load = 14) :
  towels_per_person_per_day (towels_per_day (total_towels loads towels_per_load) days) (total_people families members_per_family) = 1 :=
by {
  -- Import necessary assumptions
  sorry
}

end NUMINAMATH_GPT_each_person_towel_day_l978_97848


namespace NUMINAMATH_GPT_hyperbola_constants_l978_97811

theorem hyperbola_constants (h k a c b : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 2 ∧ c = 5 ∧ b = Real.sqrt 21 → 
  h + k + a + b = 0 + Real.sqrt 21 :=
by
  intro hka
  sorry

end NUMINAMATH_GPT_hyperbola_constants_l978_97811


namespace NUMINAMATH_GPT_solve_for_z_l978_97869

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
  sorry

end NUMINAMATH_GPT_solve_for_z_l978_97869


namespace NUMINAMATH_GPT_find_unit_vector_l978_97883

theorem find_unit_vector (a b : ℝ) : 
  a^2 + b^2 = 1 ∧ 3 * a + 4 * b = 0 →
  (a = 4 / 5 ∧ b = -3 / 5) ∨ (a = -4 / 5 ∧ b = 3 / 5) :=
by sorry

end NUMINAMATH_GPT_find_unit_vector_l978_97883


namespace NUMINAMATH_GPT_initial_students_count_l978_97888

theorem initial_students_count (n : ℕ) (W : ℝ) :
  (W = n * 28) →
  (W + 4 = (n + 1) * 27.2) →
  n = 29 :=
by
  intros hW hw_avg
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_students_count_l978_97888


namespace NUMINAMATH_GPT_value_of_a_ab_b_l978_97857

-- Define conditions
variables {a b : ℝ} (h1 : a * b = 1) (h2 : b = a + 2)

-- The proof problem
theorem value_of_a_ab_b : a - a * b - b = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_ab_b_l978_97857


namespace NUMINAMATH_GPT_b_profit_l978_97843

noncomputable def profit_share (x t : ℝ) : ℝ :=
  let total_profit := 31500
  let a_investment := 3 * x
  let a_period := 2 * t
  let b_investment := x
  let b_period := t
  let profit_ratio_a := a_investment * a_period
  let profit_ratio_b := b_investment * b_period
  let total_ratio := profit_ratio_a + profit_ratio_b
  let b_share := profit_ratio_b / total_ratio
  b_share * total_profit

theorem b_profit (x t : ℝ) : profit_share x t = 4500 :=
by
  sorry

end NUMINAMATH_GPT_b_profit_l978_97843


namespace NUMINAMATH_GPT_sally_pens_initial_count_l978_97854

theorem sally_pens_initial_count :
  ∃ P : ℕ, (P - (7 * 44)) / 2 = 17 ∧ P = 342 :=
by 
  sorry

end NUMINAMATH_GPT_sally_pens_initial_count_l978_97854


namespace NUMINAMATH_GPT_combinedTotalSandcastlesAndTowers_l978_97801

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end NUMINAMATH_GPT_combinedTotalSandcastlesAndTowers_l978_97801


namespace NUMINAMATH_GPT_find_tangent_c_l978_97825

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → (c = 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_tangent_c_l978_97825


namespace NUMINAMATH_GPT_james_browsers_l978_97874

def num_tabs_per_window := 10
def num_windows_per_browser := 3
def total_tabs := 60

theorem james_browsers : ∃ B : ℕ, (B * (num_windows_per_browser * num_tabs_per_window) = total_tabs) ∧ (B = 2) := sorry

end NUMINAMATH_GPT_james_browsers_l978_97874


namespace NUMINAMATH_GPT_grace_hours_pulling_weeds_l978_97853

variable (Charge_mowing : ℕ) (Charge_weeding : ℕ) (Charge_mulching : ℕ)
variable (H_m : ℕ) (H_u : ℕ) (E_s : ℕ)

theorem grace_hours_pulling_weeds 
  (Charge_mowing_eq : Charge_mowing = 6)
  (Charge_weeding_eq : Charge_weeding = 11)
  (Charge_mulching_eq : Charge_mulching = 9)
  (H_m_eq : H_m = 63)
  (H_u_eq : H_u = 10)
  (E_s_eq : E_s = 567) :
  ∃ W : ℕ, 6 * 63 + 11 * W + 9 * 10 = 567 ∧ W = 9 := by
  sorry

end NUMINAMATH_GPT_grace_hours_pulling_weeds_l978_97853


namespace NUMINAMATH_GPT_range_of_m_l978_97805

open Set

theorem range_of_m (m : ℝ) : 
  (∀ x, (m + 1 ≤ x ∧ x ≤ 2 * m - 1) → (-2 < x ∧ x ≤ 5)) → 
  m ∈ Iic (3 : ℝ) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l978_97805


namespace NUMINAMATH_GPT_rate_per_sqm_is_correct_l978_97812

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end NUMINAMATH_GPT_rate_per_sqm_is_correct_l978_97812


namespace NUMINAMATH_GPT_min_value_expr_l978_97851

theorem min_value_expr (a b : ℝ) (h : a - 2 * b + 8 = 0) : ∃ x : ℝ, x = 2^a + 1 / 4^b ∧ x = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l978_97851


namespace NUMINAMATH_GPT_natural_number_squares_l978_97872

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end NUMINAMATH_GPT_natural_number_squares_l978_97872


namespace NUMINAMATH_GPT_cos_300_eq_half_l978_97884

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_300_eq_half_l978_97884


namespace NUMINAMATH_GPT_curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l978_97827

-- Definitions for the conditions
def curve_C_polar (ρ θ : ℝ) := ρ = 4 * Real.sin θ
def line_l_parametric (x y t : ℝ) := 
  x = (Real.sqrt 3 / 2) * t ∧ 
  y = 1 + (1 / 2) * t

-- Theorem statements
theorem curve_C_cartesian_eq : ∀ x y : ℝ,
  (∃ (ρ θ : ℝ), curve_C_polar ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

theorem line_l_general_eq : ∀ x y t : ℝ,
  line_l_parametric x y t →
  x - (Real.sqrt 3) * y + Real.sqrt 3 = 0 :=
by sorry

theorem max_area_triangle_PAB : ∀ (P A B : ℝ × ℝ),
  (∃ (θ : ℝ), P = ⟨2 * Real.cos θ, 2 + 2 * Real.sin θ⟩ ∧
   (∃ t : ℝ, line_l_parametric A.1 A.2 t) ∧
   (∃ t' : ℝ, line_l_parametric B.1 B.2 t') ∧
   A ≠ B) →
  (1/2) * Real.sqrt 13 * (2 + Real.sqrt 3 / 2) = (4 * Real.sqrt 13 + Real.sqrt 39) / 4 :=
by sorry

end NUMINAMATH_GPT_curve_C_cartesian_eq_line_l_general_eq_max_area_triangle_PAB_l978_97827


namespace NUMINAMATH_GPT_min_abs_val_of_36_power_minus_5_power_l978_97817

theorem min_abs_val_of_36_power_minus_5_power :
  ∃ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| = 11 := sorry

end NUMINAMATH_GPT_min_abs_val_of_36_power_minus_5_power_l978_97817


namespace NUMINAMATH_GPT_total_candy_given_l978_97813

def candy_given_total (a b c : ℕ) : ℕ := a + b + c

def first_10_friends_candy (n : ℕ) := 10 * n

def next_7_friends_candy (n : ℕ) := 7 * (2 * n)

def remaining_friends_candy := 50

theorem total_candy_given (n : ℕ) (h1 : first_10_friends_candy 12 = 120)
  (h2 : next_7_friends_candy 12 = 168) (h3 : remaining_friends_candy = 50) :
  candy_given_total 120 168 50 = 338 := by
  sorry

end NUMINAMATH_GPT_total_candy_given_l978_97813


namespace NUMINAMATH_GPT_ones_digit_of_prime_in_sequence_l978_97895

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def valid_arithmetic_sequence (p1 p2 p3 p4: Nat) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4) ∧ (p3 = p2 + 4) ∧ (p4 = p3 + 4)

theorem ones_digit_of_prime_in_sequence (p1 p2 p3 p4 : Nat) (hp_seq : valid_arithmetic_sequence p1 p2 p3 p4) (hp1_gt_3 : p1 > 3) : 
  (p1 % 10) = 9 :=
sorry

end NUMINAMATH_GPT_ones_digit_of_prime_in_sequence_l978_97895


namespace NUMINAMATH_GPT_person_is_not_sane_l978_97809

-- Definitions
def Person : Type := sorry
def sane : Person → Prop := sorry
def human : Person → Prop := sorry
def vampire : Person → Prop := sorry
def declares (p : Person) (s : String) : Prop := sorry

-- Conditions
axiom transylvanian_declares_vampire (p : Person) : declares p "I am a vampire"
axiom sane_human_never_claims_vampire (p : Person) : sane p → human p → ¬ declares p "I am a vampire"
axiom sane_vampire_never_admits_vampire (p : Person) : sane p → vampire p → ¬ declares p "I am a vampire"
axiom insane_human_might_claim_vampire (p : Person) : ¬ sane p → human p → declares p "I am a vampire"
axiom insane_vampire_might_admit_vampire (p : Person) : ¬ sane p → vampire p → declares p "I am a vampire"

-- Proof statement
theorem person_is_not_sane (p : Person) : declares p "I am a vampire" → ¬ sane p :=
by
  intros h
  sorry

end NUMINAMATH_GPT_person_is_not_sane_l978_97809


namespace NUMINAMATH_GPT_john_remaining_money_l978_97840

theorem john_remaining_money (q : ℝ) : 
  let drink_cost := 5 * q
  let medium_pizza_cost := 3 * 2 * q
  let large_pizza_cost := 2 * 3 * q
  let dessert_cost := 4 * (1 / 2) * q
  let total_cost := drink_cost + medium_pizza_cost + large_pizza_cost + dessert_cost
  let initial_money := 60
  initial_money - total_cost = 60 - 19 * q :=
by
  sorry

end NUMINAMATH_GPT_john_remaining_money_l978_97840


namespace NUMINAMATH_GPT_arcsin_cos_arcsin_arccos_sin_arccos_l978_97870

-- Define the statement
theorem arcsin_cos_arcsin_arccos_sin_arccos (x : ℝ) 
  (h1 : -1 ≤ x) 
  (h2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) := 
sorry

end NUMINAMATH_GPT_arcsin_cos_arcsin_arccos_sin_arccos_l978_97870


namespace NUMINAMATH_GPT_max_value_of_expression_l978_97887

theorem max_value_of_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2 * b + 3 * c = 1) :
    (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) ≤ 7) :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l978_97887


namespace NUMINAMATH_GPT_parabola_line_dot_product_l978_97891

theorem parabola_line_dot_product (k x1 x2 y1 y2 : ℝ) 
  (h_line: ∀ x, y = k * x + 2)
  (h_parabola: ∀ x, y = (1 / 4) * x ^ 2) 
  (h_A: y1 = k * x1 + 2 ∧ y1 = (1 / 4) * x1 ^ 2)
  (h_B: y2 = k * x2 + 2 ∧ y2 = (1 / 4) * x2 ^ 2) :
  x1 * x2 + y1 * y2 = -4 := 
sorry

end NUMINAMATH_GPT_parabola_line_dot_product_l978_97891


namespace NUMINAMATH_GPT_min_area_of_B_l978_97829

noncomputable def setA := { p : ℝ × ℝ | abs (p.1 - 2) + abs (p.2 - 3) ≤ 1 }

noncomputable def setB (D E F : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0 ∧ D^2 + E^2 - 4 * F > 0 }

theorem min_area_of_B (D E F : ℝ) (h : setA ⊆ setB D E F) : 
  ∃ r : ℝ, (∀ p ∈ setB D E F, p.1^2 + p.2^2 ≤ r^2) ∧ (π * r^2 = 2 * π) :=
sorry

end NUMINAMATH_GPT_min_area_of_B_l978_97829


namespace NUMINAMATH_GPT_second_team_pieces_l978_97824

-- Definitions for the conditions
def total_pieces_required : ℕ := 500
def pieces_first_team : ℕ := 189
def pieces_third_team : ℕ := 180

-- The number of pieces the second team made
def pieces_second_team : ℕ := total_pieces_required - (pieces_first_team + pieces_third_team)

-- The theorem we are proving
theorem second_team_pieces : pieces_second_team = 131 := by
  unfold pieces_second_team
  norm_num
  sorry

end NUMINAMATH_GPT_second_team_pieces_l978_97824


namespace NUMINAMATH_GPT_total_matches_correct_total_points_earthlings_correct_total_players_is_square_l978_97844

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end NUMINAMATH_GPT_total_matches_correct_total_points_earthlings_correct_total_players_is_square_l978_97844


namespace NUMINAMATH_GPT_cost_of_6_bottle_caps_l978_97832

-- Define the cost of each bottle cap
def cost_per_bottle_cap : ℕ := 2

-- Define how many bottle caps we are buying
def number_of_bottle_caps : ℕ := 6

-- Define the total cost of the bottle caps
def total_cost : ℕ := 12

-- The proof statement to prove that the total cost is as expected
theorem cost_of_6_bottle_caps :
  cost_per_bottle_cap * number_of_bottle_caps = total_cost :=
by
  sorry

end NUMINAMATH_GPT_cost_of_6_bottle_caps_l978_97832


namespace NUMINAMATH_GPT_complete_the_square_l978_97868

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_the_square_l978_97868


namespace NUMINAMATH_GPT_convex_polygon_with_arith_prog_angles_l978_97803

theorem convex_polygon_with_arith_prog_angles 
  (n : ℕ) 
  (angles : Fin n → ℝ)
  (is_convex : ∀ i, angles i < 180)
  (arithmetic_progression : ∃ a d, d = 3 ∧ ∀ i, angles i = a + i * d)
  (largest_angle : ∃ i, angles i = 150)
  : n = 24 :=
sorry

end NUMINAMATH_GPT_convex_polygon_with_arith_prog_angles_l978_97803


namespace NUMINAMATH_GPT_find_unknown_number_l978_97804

-- Definitions

-- Declaring that we have an inserted number 'a' between 3 and unknown number 'b'
variable (a b : ℕ)

-- Conditions provided in the problem
def arithmetic_sequence_condition (a b : ℕ) : Prop := 
  a - 3 = b - a

def geometric_sequence_condition (a b : ℕ) : Prop :=
  (a - 6) / 3 = b / (a - 6)

-- The theorem statement equivalent to the problem
theorem find_unknown_number (h1 : arithmetic_sequence_condition a b) (h2 : geometric_sequence_condition a b) : b = 27 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l978_97804


namespace NUMINAMATH_GPT_comprehensive_survey_l978_97820

def suitable_for_census (s: String) : Prop := 
  s = "Surveying the heights of all classmates in the class"

theorem comprehensive_survey : suitable_for_census "Surveying the heights of all classmates in the class" :=
by
  sorry

end NUMINAMATH_GPT_comprehensive_survey_l978_97820


namespace NUMINAMATH_GPT_average_speed_round_trip_l978_97836

theorem average_speed_round_trip :
  ∀ (D : ℝ), 
  D > 0 → 
  let upstream_speed := 6 
  let downstream_speed := 5 
  (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 :=
by
  intro D hD
  let upstream_speed := 6
  let downstream_speed := 5
  have h : (2 * D) / ((D / upstream_speed) + (D / downstream_speed)) = 60 / 11 := sorry
  exact h

end NUMINAMATH_GPT_average_speed_round_trip_l978_97836


namespace NUMINAMATH_GPT_batman_game_cost_l978_97837

theorem batman_game_cost (total_spent superman_cost : ℝ) 
  (H1 : total_spent = 18.66) (H2 : superman_cost = 5.06) :
  total_spent - superman_cost = 13.60 :=
by
  sorry

end NUMINAMATH_GPT_batman_game_cost_l978_97837


namespace NUMINAMATH_GPT_parabola_directrix_l978_97816

theorem parabola_directrix (x y : ℝ) (h : y = 16 * x^2) : y = -1/64 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l978_97816


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l978_97833

-- Define atoms and their weights
def C_weight : ℕ := 12
def H_weight : ℕ := 1
def O_weight : ℕ := 16

-- Number of each atom in the compound and total molecular weight
def num_C : ℕ := 4
def num_O : ℕ := 1
def total_weight : ℕ := 65

-- Total mass of carbon and oxygen in the compound
def mass_C_O : ℕ := (num_C * C_weight) + (num_O * O_weight)

-- Mass and number of hydrogen atoms in the compound
def mass_H : ℕ := total_weight - mass_C_O
def num_H : ℕ := mass_H / H_weight

theorem hydrogen_atoms_in_compound : num_H = 1 := by
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l978_97833


namespace NUMINAMATH_GPT_choose_starters_with_twins_l978_97845

theorem choose_starters_with_twins :
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  total_ways - without_twins = 540 := 
by
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  exact Nat.sub_eq_of_eq_add sorry -- here we will need the exact proof steps which we skip

end NUMINAMATH_GPT_choose_starters_with_twins_l978_97845


namespace NUMINAMATH_GPT_original_denominator_l978_97881

theorem original_denominator (d : ℤ) : 
  (∀ n : ℤ, n = 3 → (n + 8) / (d + 8) = 1 / 3) → d = 25 :=
by
  intro h
  specialize h 3 rfl
  sorry

end NUMINAMATH_GPT_original_denominator_l978_97881


namespace NUMINAMATH_GPT_two_pow_start_digits_l978_97876

theorem two_pow_start_digits (A : ℕ) : 
  ∃ (m n : ℕ), 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1) :=
  sorry

end NUMINAMATH_GPT_two_pow_start_digits_l978_97876


namespace NUMINAMATH_GPT_rounds_on_sunday_l978_97828

theorem rounds_on_sunday (round_time total_time saturday_rounds : ℕ) (h1 : round_time = 30)
(h2 : total_time = 780) (h3 : saturday_rounds = 11) : 
(total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry

end NUMINAMATH_GPT_rounds_on_sunday_l978_97828


namespace NUMINAMATH_GPT_right_triangle_side_81_exists_arithmetic_progression_l978_97831

theorem right_triangle_side_81_exists_arithmetic_progression :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a - d)^2 + a^2 = (a + d)^2 ∧ (3*d = 81 ∨ 4*d = 81 ∨ 5*d = 81) :=
sorry

end NUMINAMATH_GPT_right_triangle_side_81_exists_arithmetic_progression_l978_97831


namespace NUMINAMATH_GPT_sum_sequence_six_l978_97855

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry

theorem sum_sequence_six :
  (∀ n, S n = 2 * a n + 1) → S 6 = 63 :=
by
  sorry

end NUMINAMATH_GPT_sum_sequence_six_l978_97855


namespace NUMINAMATH_GPT_pair_with_15_is_47_l978_97808

theorem pair_with_15_is_47 (numbers : Set ℕ) (k : ℕ) 
  (h : numbers = {49, 29, 9, 40, 22, 15, 53, 33, 13, 47}) 
  (pair_sum_eq_k : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → (a, b) ≠ (15, 15) → a + b = k) : 
  ∃ (k : ℕ), 15 + 47 = k := 
sorry

end NUMINAMATH_GPT_pair_with_15_is_47_l978_97808


namespace NUMINAMATH_GPT_volume_is_correct_l978_97893

noncomputable def volume_of_target_cube (V₁ : ℝ) (A₂ : ℝ) : ℝ :=
  if h₁ : V₁ = 8 then
    let s₁ := (8 : ℝ)^(1/3)
    let A₁ := 6 * s₁^2
    if h₂ : A₂ = 2 * A₁ then
      let s₂ := (A₂ / 6)^(1/2)
      let V₂ := s₂^3
      V₂
    else 0
  else 0

theorem volume_is_correct : volume_of_target_cube 8 48 = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_is_correct_l978_97893


namespace NUMINAMATH_GPT_count_sums_of_two_cubes_lt_400_l978_97810

theorem count_sums_of_two_cubes_lt_400 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, ∃ a b, 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 7 ∧ n = a^3 + b^3 ∧ (Odd a ∨ Odd b) ∧ n < 400) ∧
    s.card = 15 :=
by 
  sorry

end NUMINAMATH_GPT_count_sums_of_two_cubes_lt_400_l978_97810


namespace NUMINAMATH_GPT_find_k_unique_solution_l978_97886

theorem find_k_unique_solution (k : ℝ) (h: k ≠ 0) : (∀ x : ℝ, (x + 3) / (k * x - 2) = x → k = -3/4) :=
sorry

end NUMINAMATH_GPT_find_k_unique_solution_l978_97886


namespace NUMINAMATH_GPT_abigail_monthly_saving_l978_97838

-- Definitions based on the conditions
def total_saving := 48000
def months_in_year := 12

-- The statement to be proved
theorem abigail_monthly_saving : total_saving / months_in_year = 4000 :=
by sorry

end NUMINAMATH_GPT_abigail_monthly_saving_l978_97838


namespace NUMINAMATH_GPT_soccer_most_students_l978_97819

def sports := ["hockey", "basketball", "soccer", "volleyball", "badminton"]
def num_students (sport : String) : Nat :=
  match sport with
  | "hockey" => 30
  | "basketball" => 35
  | "soccer" => 50
  | "volleyball" => 20
  | "badminton" => 25
  | _ => 0

theorem soccer_most_students : ∀ sport ∈ sports, num_students "soccer" ≥ num_students sport := by
  sorry

end NUMINAMATH_GPT_soccer_most_students_l978_97819


namespace NUMINAMATH_GPT_hoseok_more_paper_than_minyoung_l978_97865

theorem hoseok_more_paper_than_minyoung : 
  ∀ (initial : ℕ) (minyoung_bought : ℕ) (hoseok_bought : ℕ), 
  initial = 150 →
  minyoung_bought = 32 →
  hoseok_bought = 49 →
  (initial + hoseok_bought) - (initial + minyoung_bought) = 17 :=
by
  intros initial minyoung_bought hoseok_bought h_initial h_min h_hos
  sorry

end NUMINAMATH_GPT_hoseok_more_paper_than_minyoung_l978_97865
