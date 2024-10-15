import Mathlib

namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2094_209419

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
noncomputable def S_n (n : ℕ) : ℝ := -n^2 + 4*n

theorem common_difference_of_arithmetic_sequence :
  (∀ n : ℕ, S n = S_n n) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = -2 :=
by
  intro h
  use -2
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2094_209419


namespace NUMINAMATH_GPT_Jack_can_form_rectangle_l2094_209457

theorem Jack_can_form_rectangle : 
  ∃ (a b : ℕ), 
  3 * a = 2016 ∧ 
  4 * a = 2016 ∧ 
  4 * b = 2016 ∧ 
  3 * b = 2016 ∧ 
  (503 * 4 + 3 * 9 = 2021) ∧ 
  (2 * 3 = 4) :=
by 
  sorry

end NUMINAMATH_GPT_Jack_can_form_rectangle_l2094_209457


namespace NUMINAMATH_GPT_ratio_pea_patch_to_radish_patch_l2094_209405

-- Definitions
def sixth_of_pea_patch : ℝ := 5
def whole_radish_patch : ℝ := 15

-- Theorem to prove
theorem ratio_pea_patch_to_radish_patch :
  (6 * sixth_of_pea_patch) / whole_radish_patch = 2 :=
by 
  -- skip the actual proof since it's not required
  sorry

end NUMINAMATH_GPT_ratio_pea_patch_to_radish_patch_l2094_209405


namespace NUMINAMATH_GPT_students_more_than_Yoongi_l2094_209449

theorem students_more_than_Yoongi (total_players : ℕ) (less_than_Yoongi : ℕ) (total_players_eq : total_players = 21) (less_than_eq : less_than_Yoongi = 11) : 
  ∃ more_than_Yoongi : ℕ, more_than_Yoongi = (total_players - 1 - less_than_Yoongi) ∧ more_than_Yoongi = 8 :=
by
  sorry

end NUMINAMATH_GPT_students_more_than_Yoongi_l2094_209449


namespace NUMINAMATH_GPT_fraction_calculation_l2094_209406

theorem fraction_calculation : 
  (1/2 - 1/3) / (3/7 * 2/8) = 14/9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2094_209406


namespace NUMINAMATH_GPT_max_frac_a_S_l2094_209485

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem max_frac_a_S (n : ℕ) (h : S n = 2^n - 1) : 
  let frac := (a n) / (a n * S n + a 6)
  ∃ N : ℕ, N > 0 ∧ (frac ≤ 1 / 15) := by
  sorry

end NUMINAMATH_GPT_max_frac_a_S_l2094_209485


namespace NUMINAMATH_GPT_largest_common_value_less_than_1000_l2094_209410

def arithmetic_sequence_1 (n : ℕ) : ℕ := 2 + 3 * n
def arithmetic_sequence_2 (m : ℕ) : ℕ := 4 + 8 * m

theorem largest_common_value_less_than_1000 :
  ∃ a n m : ℕ, a = arithmetic_sequence_1 n ∧ a = arithmetic_sequence_2 m ∧ a < 1000 ∧ a = 980 :=
by { sorry }

end NUMINAMATH_GPT_largest_common_value_less_than_1000_l2094_209410


namespace NUMINAMATH_GPT_VishalInvestedMoreThanTrishulBy10Percent_l2094_209433

variables (R T V : ℝ)

-- Given conditions
def RaghuInvests (R : ℝ) : Prop := R = 2500
def TrishulInvests (R T : ℝ) : Prop := T = 0.9 * R
def TotalInvestment (R T V : ℝ) : Prop := V + T + R = 7225
def PercentageInvestedMore (T V : ℝ) (P : ℝ) : Prop := P * T = V - T

-- Main theorem to prove
theorem VishalInvestedMoreThanTrishulBy10Percent (R T V : ℝ) (P : ℝ) :
  RaghuInvests R ∧ TrishulInvests R T ∧ TotalInvestment R T V → PercentageInvestedMore T V P → P = 0.1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_VishalInvestedMoreThanTrishulBy10Percent_l2094_209433


namespace NUMINAMATH_GPT_carpets_triple_overlap_area_l2094_209465

theorem carpets_triple_overlap_area {W H : ℕ} (hW : W = 10) (hH : H = 10) 
    {w1 h1 w2 h2 w3 h3 : ℕ} 
    (h1_w1 : w1 = 6) (h1_h1 : h1 = 8)
    (h2_w2 : w2 = 6) (h2_h2 : h2 = 6)
    (h3_w3 : w3 = 5) (h3_h3 : h3 = 7) :
    ∃ (area : ℕ), area = 6 := by
  sorry

end NUMINAMATH_GPT_carpets_triple_overlap_area_l2094_209465


namespace NUMINAMATH_GPT_sufficient_condition_l2094_209426

theorem sufficient_condition (a b : ℝ) : ab ≠ 0 → a ≠ 0 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_l2094_209426


namespace NUMINAMATH_GPT_cube_vertex_adjacency_l2094_209493

noncomputable def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem cube_vertex_adjacency :
  ∀ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ), 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8 ∧
  beautiful_face v1 v2 v3 v4 ∧ beautiful_face v5 v6 v7 v8 ∧
  beautiful_face v1 v3 v5 v7 ∧ beautiful_face v2 v4 v6 v8 ∧
  beautiful_face v1 v2 v5 v6 ∧ beautiful_face v3 v4 v7 v8 →
  (v6 = 6 → (v1 = 2 ∧ v2 = 3 ∧ v3 = 5) ∨ 
   (v1 = 3 ∧ v2 = 5 ∧ v3 = 7) ∨ 
   (v1 = 2 ∧ v2 = 3 ∧ v3 = 7)) :=
sorry

end NUMINAMATH_GPT_cube_vertex_adjacency_l2094_209493


namespace NUMINAMATH_GPT_sequence_general_term_l2094_209484

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 3 * a n - 2 * n ^ 2 + 4 * n + 4) :
  ∀ n, a n = 3^n + n^2 - n - 2 :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l2094_209484


namespace NUMINAMATH_GPT_problem_rect_ratio_l2094_209431

theorem problem_rect_ratio (W X Y Z U V R S : ℝ × ℝ) 
  (hYZ : Y = (0, 0))
  (hW : W = (0, 6))
  (hZ : Z = (7, 6))
  (hX : X = (7, 4))
  (hU : U = (5, 0))
  (hV : V = (4, 4))
  (hR : R = (5 / 3, 4))
  (hS : S = (0, 4))
  : (dist R S) / (dist X V) = 5 / 9 := 
sorry

end NUMINAMATH_GPT_problem_rect_ratio_l2094_209431


namespace NUMINAMATH_GPT_fill_in_the_blank_correct_option_l2094_209436

-- Assume each option is defined
def options := ["the other", "some", "another", "other"]

-- Define a helper function to validate the correct option
def is_correct_option (opt: String) : Prop :=
  opt = "another"

-- The main problem statement
theorem fill_in_the_blank_correct_option :
  (∀ opt, opt ∈ options → is_correct_option opt → opt = "another") :=
by
  intro opt h_option h_correct
  simp [is_correct_option] at h_correct
  exact h_correct

-- Test case to check the correct option
example : is_correct_option "another" :=
by
  simp [is_correct_option]

end NUMINAMATH_GPT_fill_in_the_blank_correct_option_l2094_209436


namespace NUMINAMATH_GPT_find_g_inv_l2094_209420

noncomputable def g (x : ℝ) : ℝ :=
  (x^7 - 1) / 4

noncomputable def g_inv_value : ℝ :=
  (51 / 32)^(1/7)

theorem find_g_inv (h : g (g_inv_value) = 19 / 128) : g_inv_value = (51 / 32)^(1/7) :=
by
  sorry

end NUMINAMATH_GPT_find_g_inv_l2094_209420


namespace NUMINAMATH_GPT_solve_for_x_l2094_209460

theorem solve_for_x :
  ∃ x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 4.5 :=
by
  use 4.5
  sorry

end NUMINAMATH_GPT_solve_for_x_l2094_209460


namespace NUMINAMATH_GPT_circle_sum_value_l2094_209468

-- Define the problem
theorem circle_sum_value (a b x : ℕ) (h1 : a = 35) (h2 : b = 47) : x = a + b :=
by
  -- Given conditions
  have ha : a = 35 := h1
  have hb : b = 47 := h2
  -- Prove that the value of x is the sum of a and b
  have h_sum : x = a + b := sorry
  -- Assert the value of x is 82 based on given a and b
  exact h_sum

end NUMINAMATH_GPT_circle_sum_value_l2094_209468


namespace NUMINAMATH_GPT_sum_integer_solutions_l2094_209464

theorem sum_integer_solutions (n : ℤ) (h1 : |n^2| < |n - 5|^2) (h2 : |n - 5|^2 < 16) : n = 2 := 
sorry

end NUMINAMATH_GPT_sum_integer_solutions_l2094_209464


namespace NUMINAMATH_GPT_eraser_ratio_l2094_209477

-- Define the variables and conditions
variables (c j g : ℕ)
variables (total : ℕ := 35)
variables (c_erasers : ℕ := 10)
variables (gabriel_erasers : ℕ := c_erasers / 2)
variables (julian_erasers : ℕ := c_erasers)

-- The proof statement
theorem eraser_ratio (hc : c_erasers = 10)
                      (h1 : c_erasers = 2 * gabriel_erasers)
                      (h2 : julian_erasers = c_erasers)
                      (h3 : c_erasers + gabriel_erasers + julian_erasers = total) :
                      julian_erasers / c_erasers = 1 :=
by
  sorry

end NUMINAMATH_GPT_eraser_ratio_l2094_209477


namespace NUMINAMATH_GPT_time_no_traffic_is_4_hours_l2094_209428

-- Definitions and conditions
def distance : ℕ := 200
def time_traffic : ℕ := 5

axiom traffic_speed_relation : ∃ (speed_traffic : ℕ), distance = speed_traffic * time_traffic
axiom speed_difference : ∀ (speed_traffic speed_no_traffic : ℕ), speed_no_traffic = speed_traffic + 10

-- Prove that the time when there's no traffic is 4 hours
theorem time_no_traffic_is_4_hours : ∀ (speed_traffic speed_no_traffic : ℕ), 
  distance = speed_no_traffic * (distance / speed_no_traffic) -> (distance / speed_no_traffic) = 4 :=
by
  intros speed_traffic speed_no_traffic h
  sorry

end NUMINAMATH_GPT_time_no_traffic_is_4_hours_l2094_209428


namespace NUMINAMATH_GPT_circle_inside_triangle_l2094_209458

-- Define the problem conditions
def triangle_sides : ℕ × ℕ × ℕ := (3, 4, 5)
def circle_area : ℚ := 25 / 8

-- Define the problem statement
theorem circle_inside_triangle (a b c : ℕ) (area : ℚ)
    (h1 : (a, b, c) = triangle_sides)
    (h2 : area = circle_area) :
    ∃ r R : ℚ, R < r ∧ 2 * r = a + b - c ∧ R^2 = area / π := sorry

end NUMINAMATH_GPT_circle_inside_triangle_l2094_209458


namespace NUMINAMATH_GPT_range_of_set_of_three_numbers_l2094_209445

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_set_of_three_numbers_l2094_209445


namespace NUMINAMATH_GPT_convert_base_7_to_base_10_l2094_209403

theorem convert_base_7_to_base_10 (n : ℕ) (h : n = 6 * 7^2 + 5 * 7^1 + 3 * 7^0) : n = 332 := by
  sorry

end NUMINAMATH_GPT_convert_base_7_to_base_10_l2094_209403


namespace NUMINAMATH_GPT_train_crosses_signal_pole_in_18_seconds_l2094_209459

-- Define the given conditions
def train_length := 300  -- meters
def platform_length := 450  -- meters
def time_to_cross_platform := 45  -- seconds

-- Define the question and the correct answer
def time_to_cross_signal_pole := 18  -- seconds (this is what we need to prove)

-- Define the total distance the train covers when crossing the platform
def total_distance_crossing_platform := train_length + platform_length  -- meters

-- Define the speed of the train
def train_speed := total_distance_crossing_platform / time_to_cross_platform  -- meters per second

theorem train_crosses_signal_pole_in_18_seconds :
  300 / train_speed = time_to_cross_signal_pole :=
by
  -- train_speed is defined directly in terms of the given conditions
  unfold train_speed total_distance_crossing_platform train_length platform_length time_to_cross_platform
  sorry

end NUMINAMATH_GPT_train_crosses_signal_pole_in_18_seconds_l2094_209459


namespace NUMINAMATH_GPT_garden_fencing_needed_l2094_209466

/-- Given a rectangular garden where the length is 300 yards and the length is twice the width,
prove that the total amount of fencing needed to enclose the garden is 900 yards. -/
theorem garden_fencing_needed :
  ∃ (W L P : ℝ), L = 300 ∧ L = 2 * W ∧ P = 2 * (L + W) ∧ P = 900 :=
by
  sorry

end NUMINAMATH_GPT_garden_fencing_needed_l2094_209466


namespace NUMINAMATH_GPT_total_sweaters_knit_l2094_209488

-- Definitions from condition a)
def monday_sweaters : ℕ := 8
def tuesday_sweaters : ℕ := monday_sweaters + 2
def wednesday_sweaters : ℕ := tuesday_sweaters - 4
def thursday_sweaters : ℕ := wednesday_sweaters
def friday_sweaters : ℕ := monday_sweaters / 2

-- Theorem statement
theorem total_sweaters_knit : 
  monday_sweaters + tuesday_sweaters + wednesday_sweaters + thursday_sweaters + friday_sweaters = 34 :=
  by
    sorry

end NUMINAMATH_GPT_total_sweaters_knit_l2094_209488


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2094_209455

-- First expression
theorem simplify_expr1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b ^ 2 :=
by
  sorry

-- Second expression
theorem simplify_expr2 (x : ℝ) : 
  ( ( (4 * x - 9) / (3 - x) - x + 3 ) / ( (x ^ 2 - 4) / (x - 3) ) ) = - (x / (x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2094_209455


namespace NUMINAMATH_GPT_total_white_roses_l2094_209423

-- Define the constants
def n_b : ℕ := 5
def n_t : ℕ := 7
def r_b : ℕ := 5
def r_t : ℕ := 12

-- State the theorem
theorem total_white_roses :
  n_t * r_t + n_b * r_b = 109 :=
by
  -- Automatic proof can be here; using sorry as placeholder
  sorry

end NUMINAMATH_GPT_total_white_roses_l2094_209423


namespace NUMINAMATH_GPT_total_fruit_count_l2094_209495

theorem total_fruit_count :
  let gerald_apple_bags := 5
  let gerald_orange_bags := 4
  let apples_per_gerald_bag := 30
  let oranges_per_gerald_bag := 25
  let pam_apple_bags := 6
  let pam_orange_bags := 4
  let sue_apple_bags := 2 * gerald_apple_bags
  let sue_orange_bags := gerald_orange_bags / 2
  let apples_per_sue_bag := apples_per_gerald_bag - 10
  let oranges_per_sue_bag := oranges_per_gerald_bag + 5
  
  let gerald_apples := gerald_apple_bags * apples_per_gerald_bag
  let gerald_oranges := gerald_orange_bags * oranges_per_gerald_bag
  
  let pam_apples := pam_apple_bags * (3 * apples_per_gerald_bag)
  let pam_oranges := pam_orange_bags * (2 * oranges_per_gerald_bag)
  
  let sue_apples := sue_apple_bags * apples_per_sue_bag
  let sue_oranges := sue_orange_bags * oranges_per_sue_bag

  let total_apples := gerald_apples + pam_apples + sue_apples
  let total_oranges := gerald_oranges + pam_oranges + sue_oranges
  total_apples + total_oranges = 1250 :=

by
  sorry

end NUMINAMATH_GPT_total_fruit_count_l2094_209495


namespace NUMINAMATH_GPT_largest_number_less_than_2_l2094_209452

theorem largest_number_less_than_2 (a b c : ℝ) (h_a : a = 0.8) (h_b : b = 1/2) (h_c : c = 0.5) : 
  a < 2 ∧ b < 2 ∧ c < 2 ∧ (∀ x, (x = a ∨ x = b ∨ x = c) → x < 2) → 
  a = 0.8 ∧ 
  (a > b ∧ a > c) ∧ 
  (a < 2) :=
by sorry

end NUMINAMATH_GPT_largest_number_less_than_2_l2094_209452


namespace NUMINAMATH_GPT_equation_solutions_l2094_209479

noncomputable def solve_equation (x : ℝ) : Prop :=
  x - 3 = 4 * (x - 3)^2

theorem equation_solutions :
  ∀ x : ℝ, solve_equation x ↔ x = 3 ∨ x = 3.25 :=
by sorry

end NUMINAMATH_GPT_equation_solutions_l2094_209479


namespace NUMINAMATH_GPT_evaluate_expression_l2094_209404

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a^a - a*(a-2)^a)^a = 1358954496 :=
by
  rw [h]  -- Substitute a with 4
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2094_209404


namespace NUMINAMATH_GPT_find_integers_l2094_209480

theorem find_integers (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) : 
  (a, b, c, d) = (1, 2, 3, 4) ∨ (a, b, c, d) = (1, 2, 4, 3) ∨ (a, b, c, d) = (1, 3, 2, 4) ∨ (a, b, c, d) = (1, 3, 4, 2) ∨ (a, b, c, d) = (1, 4, 2, 3) ∨ (a, b, c, d) = (1, 4, 3, 2) ∨ (a, b, c, d) = (2, 1, 3, 4) ∨ (a, b, c, d) = (2, 1, 4, 3) ∨ (a, b, c, d) = (2, 3, 1, 4) ∨ (a, b, c, d) = (2, 3, 4, 1) ∨ (a, b, c, d) = (2, 4, 1, 3) ∨ (a, b, c, d) = (2, 4, 3, 1) ∨ (a, b, c, d) = (3, 1, 2, 4) ∨ (a, b, c, d) = (3, 1, 4, 2) ∨ (a, b, c, d) = (3, 2, 1, 4) ∨ (a, b, c, d) = (3, 2, 4, 1) ∨ (a, b, c, d) = (3, 4, 1, 2) ∨ (a, b, c, d) = (3, 4, 2, 1) ∨ (a, b, c, d) = (4, 1, 2, 3) ∨ (a, b, c, d) = (4, 1, 3, 2) ∨ (a, b, c, d) = (4, 2, 1, 3) ∨ (a, b, c, d) = (4, 2, 3, 1) ∨ (a, b, c, d) = (4, 3, 1, 2) ∨ (a, b, c, d) = (4, 3, 2, 1) :=
sorry

end NUMINAMATH_GPT_find_integers_l2094_209480


namespace NUMINAMATH_GPT_correct_subsidy_equation_l2094_209415

-- Define the necessary variables and conditions
def sales_price (x : ℝ) := x  -- sales price of the mobile phone in yuan
def subsidy_rate : ℝ := 0.13  -- 13% subsidy rate
def number_of_phones : ℝ := 20  -- 20 units sold
def total_subsidy : ℝ := 2340  -- total subsidy provided

-- Lean theorem statement to prove the correct equation
theorem correct_subsidy_equation (x : ℝ) :
  number_of_phones * x * subsidy_rate = total_subsidy :=
by
  sorry -- proof to be completed

end NUMINAMATH_GPT_correct_subsidy_equation_l2094_209415


namespace NUMINAMATH_GPT_range_of_a_l2094_209413

theorem range_of_a (a : ℝ) (h_pos : a > 0)
  (p : ∀ x : ℝ, x^2 - 4 * a * x + 3 * a^2 ≤ 0)
  (q : ∀ x : ℝ, (x^2 - x - 6 < 0) ∧ (x^2 + 2 * x - 8 > 0)) :
  (a ∈ ((Set.Ioo 0 (2 / 3)) ∪ (Set.Ici 3))) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2094_209413


namespace NUMINAMATH_GPT_greatest_p_meets_conditions_l2094_209401

-- Define a four-digit number and its reversal being divisible by 63 and another condition of divisibility
def is_divisible_by (n m : ℕ) : Prop :=
  m % n = 0

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ a d => a * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def p := 9507

-- The main theorem we aim to prove.
theorem greatest_p_meets_conditions (p q : ℕ) 
  (h1 : is_four_digit p) 
  (h2 : is_four_digit q) 
  (h3 : reverse_digits p = q) 
  (h4 : is_divisible_by 63 p) 
  (h5 : is_divisible_by 63 q) 
  (h6 : is_divisible_by 9 p) : 
  p = 9507 :=
sorry

end NUMINAMATH_GPT_greatest_p_meets_conditions_l2094_209401


namespace NUMINAMATH_GPT_internal_angle_sine_l2094_209421

theorem internal_angle_sine (α : ℝ) (h1 : α > 0 ∧ α < 180) (h2 : Real.sin (α * (Real.pi / 180)) = 1 / 2) : α = 30 ∨ α = 150 :=
sorry

end NUMINAMATH_GPT_internal_angle_sine_l2094_209421


namespace NUMINAMATH_GPT_a_2023_le_1_l2094_209414

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, (a (n+1))^2 + a n * a (n+2) ≤ a n + a (n+2))

theorem a_2023_le_1 : a 2023 ≤ 1 := by
  sorry

end NUMINAMATH_GPT_a_2023_le_1_l2094_209414


namespace NUMINAMATH_GPT_optionA_optionC_l2094_209487

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem optionA : ∀ x : ℝ, f (x + 2) = f (-x + 2) := 
by sorry

theorem optionC : (∀ x : ℝ, x < 2 → f x > f (x + 0.01)) ∧ (∀ x : ℝ, x > 2 → f x < f (x - 0.01)) := 
by sorry

end NUMINAMATH_GPT_optionA_optionC_l2094_209487


namespace NUMINAMATH_GPT_problem_f_symmetric_l2094_209432

theorem problem_f_symmetric (f : ℝ → ℝ) (k : ℝ) (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b) (h_not_zero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = f x :=
sorry

end NUMINAMATH_GPT_problem_f_symmetric_l2094_209432


namespace NUMINAMATH_GPT_total_tiles_is_1352_l2094_209475

noncomputable def side_length_of_floor := 39

noncomputable def total_tiles_covering_floor (n : ℕ) : ℕ :=
  (n ^ 2) - ((n / 3) ^ 2)

theorem total_tiles_is_1352 :
  total_tiles_covering_floor side_length_of_floor = 1352 := by
  sorry

end NUMINAMATH_GPT_total_tiles_is_1352_l2094_209475


namespace NUMINAMATH_GPT_erica_earnings_l2094_209482

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end NUMINAMATH_GPT_erica_earnings_l2094_209482


namespace NUMINAMATH_GPT_john_total_money_l2094_209491

-- Variables representing the prices and quantities.
def chip_price : ℝ := 2
def corn_chip_price : ℝ := 1.5
def chips_quantity : ℕ := 15
def corn_chips_quantity : ℕ := 10

-- Hypothesis representing the total money John has.
theorem john_total_money : 
    (chips_quantity * chip_price + corn_chips_quantity * corn_chip_price) = 45 := by
  sorry

end NUMINAMATH_GPT_john_total_money_l2094_209491


namespace NUMINAMATH_GPT_count_two_digit_even_congruent_to_1_mod_4_l2094_209402

theorem count_two_digit_even_congruent_to_1_mod_4 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n % 4 = 1 ∧ 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0) ∧ S.card = 23 := 
sorry

end NUMINAMATH_GPT_count_two_digit_even_congruent_to_1_mod_4_l2094_209402


namespace NUMINAMATH_GPT_red_button_probability_l2094_209462

-- Definitions of the initial state
def initial_red_buttons : ℕ := 8
def initial_blue_buttons : ℕ := 12
def total_buttons := initial_red_buttons + initial_blue_buttons

-- Condition of removal and remaining buttons
def removed_buttons := total_buttons - (5 / 8 : ℚ) * total_buttons

-- Equal number of red and blue buttons removed
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

-- State after removal
def remaining_red_buttons := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons := initial_blue_buttons - removed_blue_buttons

-- Jars after removal
def jar_X := remaining_red_buttons + remaining_blue_buttons
def jar_Y := removed_red_buttons + removed_blue_buttons

-- Probability calculations
def probability_red_X : ℚ := remaining_red_buttons / jar_X
def probability_red_Y : ℚ := removed_red_buttons / jar_Y

-- Final probability
def final_probability : ℚ := probability_red_X * probability_red_Y

theorem red_button_probability :
  final_probability = 4 / 25 := 
  sorry

end NUMINAMATH_GPT_red_button_probability_l2094_209462


namespace NUMINAMATH_GPT_correct_option_is_B_l2094_209434

variable (f : ℝ → ℝ)
variable (h0 : f 0 = 2)
variable (h1 : ∀ x : ℝ, deriv f x > f x + 1)

theorem correct_option_is_B : 3 * Real.exp (1 : ℝ) < f 2 + 1 := sorry

end NUMINAMATH_GPT_correct_option_is_B_l2094_209434


namespace NUMINAMATH_GPT_proof_of_A_inter_complement_B_l2094_209442

variable (U : Set Nat) 
variable (A B : Set Nat)

theorem proof_of_A_inter_complement_B :
    (U = {1, 2, 3, 4}) →
    (B = {1, 2}) →
    (compl (A ∪ B) = {4}) →
    (A ∩ compl B = {3}) :=
by
  intros hU hB hCompl
  sorry

end NUMINAMATH_GPT_proof_of_A_inter_complement_B_l2094_209442


namespace NUMINAMATH_GPT_total_spots_l2094_209499

-- Define the variables
variables (R C G S B : ℕ)

-- State the problem conditions
def conditions : Prop :=
  R = 46 ∧
  C = R / 2 - 5 ∧
  G = 5 * C ∧
  S = 3 * R ∧
  B = 2 * (G + S)

-- State the proof problem
theorem total_spots : conditions R C G S B → G + C + S + B = 702 :=
by
  intro h
  obtain ⟨hR, hC, hG, hS, hB⟩ := h
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_total_spots_l2094_209499


namespace NUMINAMATH_GPT_simplify_expression_l2094_209440

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end NUMINAMATH_GPT_simplify_expression_l2094_209440


namespace NUMINAMATH_GPT_carina_total_coffee_l2094_209454

def number_of_ten_ounce_packages : ℕ := 4
def number_of_five_ounce_packages : ℕ := number_of_ten_ounce_packages + 2
def ounces_in_each_ten_ounce_package : ℕ := 10
def ounces_in_each_five_ounce_package : ℕ := 5

def total_coffee_ounces : ℕ := 
  (number_of_ten_ounce_packages * ounces_in_each_ten_ounce_package) +
  (number_of_five_ounce_packages * ounces_in_each_five_ounce_package)

theorem carina_total_coffee : total_coffee_ounces = 70 := by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_carina_total_coffee_l2094_209454


namespace NUMINAMATH_GPT_round_2748397_542_nearest_integer_l2094_209476

theorem round_2748397_542_nearest_integer :
  let n := 2748397.542
  let int_part := 2748397
  let decimal_part := 0.542
  (n.round = 2748398) :=
by
  sorry

end NUMINAMATH_GPT_round_2748397_542_nearest_integer_l2094_209476


namespace NUMINAMATH_GPT_find_a_l2094_209483

theorem find_a (a b c : ℝ) (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
                 (h2 : a * 15 * 7 = 1.5) : a = 6 :=
sorry

end NUMINAMATH_GPT_find_a_l2094_209483


namespace NUMINAMATH_GPT_ten_years_less_than_average_age_l2094_209416

theorem ten_years_less_than_average_age (L : ℕ) :
  (2 * L - 14) = 
    (2 * L - 4) - 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_ten_years_less_than_average_age_l2094_209416


namespace NUMINAMATH_GPT_correct_product_l2094_209490

theorem correct_product (a b : ℚ) (calc_incorrect : a = 52 ∧ b = 735)
                        (incorrect_product : a * b = 38220) :
  (0.52 * 7.35 = 3.822) :=
by
  sorry

end NUMINAMATH_GPT_correct_product_l2094_209490


namespace NUMINAMATH_GPT_bus_students_after_fifth_stop_l2094_209478

theorem bus_students_after_fifth_stop :
  let initial := 72
  let firstStop := (2 / 3 : ℚ) * initial
  let secondStop := (2 / 3 : ℚ) * firstStop
  let thirdStop := (2 / 3 : ℚ) * secondStop
  let fourthStop := (2 / 3 : ℚ) * thirdStop
  let fifthStop := fourthStop + 12
  fifthStop = 236 / 9 :=
by
  sorry

end NUMINAMATH_GPT_bus_students_after_fifth_stop_l2094_209478


namespace NUMINAMATH_GPT_find_a_10_l2094_209496

/-- 
a_n is an arithmetic sequence
-/
def a (n : ℕ) : ℝ := sorry

/-- 
Given conditions:
- Condition 1: a_2 + a_5 = 19
- Condition 2: S_5 = 40, where S_5 is the sum of the first five terms
-/
axiom condition1 : a 2 + a 5 = 19
axiom condition2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 40

noncomputable def a_10 : ℝ := a 10

theorem find_a_10 : a_10 = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_a_10_l2094_209496


namespace NUMINAMATH_GPT_longer_bus_ride_l2094_209417

theorem longer_bus_ride :
  let oscar := 0.75
  let charlie := 0.25
  oscar - charlie = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_longer_bus_ride_l2094_209417


namespace NUMINAMATH_GPT_product_of_repeating145_and_11_equals_1595_over_999_l2094_209498

-- Defining the repeating decimal as a fraction
def repeating145_as_fraction : ℚ :=
  145 / 999

-- Stating the main theorem
theorem product_of_repeating145_and_11_equals_1595_over_999 :
  11 * repeating145_as_fraction = 1595 / 999 :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating145_and_11_equals_1595_over_999_l2094_209498


namespace NUMINAMATH_GPT_simplify_fraction_l2094_209492

theorem simplify_fraction (a b : ℕ) (h : a = 2020) (h2 : b = 2018) :
  (2 ^ a - 2 ^ b) / (2 ^ a + 2 ^ b) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2094_209492


namespace NUMINAMATH_GPT_math_problem_l2094_209439

theorem math_problem (x : ℂ) (hx : x + 1/x = 3) : x^6 + 1/x^6 = 322 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l2094_209439


namespace NUMINAMATH_GPT_quadratic_value_at_3_l2094_209446

theorem quadratic_value_at_3 (a b c : ℝ) :
  (a * (-2)^2 + b * (-2) + c = -13 / 2) →
  (a * (-1)^2 + b * (-1) + c = -4) →
  (a * 0^2 + b * 0 + c = -2.5) →
  (a * 1^2 + b * 1 + c = -2) →
  (a * 2^2 + b * 2 + c = -2.5) →
  (a * 3^2 + b * 3 + c = -4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_value_at_3_l2094_209446


namespace NUMINAMATH_GPT_difference_is_minus_four_l2094_209437

def percentage_scoring_60 : ℝ := 0.15
def percentage_scoring_75 : ℝ := 0.25
def percentage_scoring_85 : ℝ := 0.40
def percentage_scoring_95 : ℝ := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85)

def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

def mean_score : ℝ :=
  (percentage_scoring_60 * score_60) +
  (percentage_scoring_75 * score_75) +
  (percentage_scoring_85 * score_85) +
  (percentage_scoring_95 * score_95)

def median_score : ℝ := score_85

def difference_mean_median : ℝ := mean_score - median_score

theorem difference_is_minus_four : difference_mean_median = -4 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_minus_four_l2094_209437


namespace NUMINAMATH_GPT_garden_width_l2094_209412

theorem garden_width (w : ℕ) (h1 : ∀ l : ℕ, l = w + 12 → l * w ≥ 120) : w = 6 := 
by
  sorry

end NUMINAMATH_GPT_garden_width_l2094_209412


namespace NUMINAMATH_GPT_sum_two_and_four_l2094_209425

theorem sum_two_and_four : 2 + 4 = 6 := by
  sorry

end NUMINAMATH_GPT_sum_two_and_four_l2094_209425


namespace NUMINAMATH_GPT_sum_of_interior_angles_l2094_209400

theorem sum_of_interior_angles (n : ℕ) (h₁ : 180 * (n - 2) = 2340) : 
  180 * ((n - 3) - 2) = 1800 := by
  -- Here, we'll solve the theorem using Lean's capabilities.
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l2094_209400


namespace NUMINAMATH_GPT_sum_first_11_even_numbers_is_132_l2094_209456

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_first_11_even_numbers_is_132 : sum_first_n_even_numbers 11 = 132 := 
  by
    sorry

end NUMINAMATH_GPT_sum_first_11_even_numbers_is_132_l2094_209456


namespace NUMINAMATH_GPT_price_sugar_salt_l2094_209409

/-- The price of two kilograms of sugar and five kilograms of salt is $5.50. If a kilogram of sugar 
    costs $1.50, then how much is the price of three kilograms of sugar and some kilograms of salt, 
    if the total price is $5? -/
theorem price_sugar_salt 
  (price_sugar_per_kg : ℝ)
  (price_total_2kg_sugar_5kg_salt : ℝ)
  (total_price : ℝ) :
  price_sugar_per_kg = 1.50 →
  price_total_2kg_sugar_5kg_salt = 5.50 →
  total_price = 5 →
  2 * price_sugar_per_kg + 5 * (price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5 = 5.50 →
  3 * price_sugar_per_kg + (total_price - 3 * price_sugar_per_kg) / ((price_total_2kg_sugar_5kg_salt - 2 * price_sugar_per_kg) / 5) = 1 →
  true :=
by
  sorry

end NUMINAMATH_GPT_price_sugar_salt_l2094_209409


namespace NUMINAMATH_GPT_solve_for_y_l2094_209474

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l2094_209474


namespace NUMINAMATH_GPT_x_intercept_perpendicular_l2094_209450

theorem x_intercept_perpendicular (k m x y : ℝ) (h1 : 4 * x - 3 * y = 12) (h2 : y = -3/4 * x + 3) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_perpendicular_l2094_209450


namespace NUMINAMATH_GPT_initial_number_of_fruits_l2094_209470

theorem initial_number_of_fruits (oranges apples limes : ℕ) (h_oranges : oranges = 50)
  (h_apples : apples = 72) (h_oranges_limes : oranges = 2 * limes) (h_apples_limes : apples = 3 * limes) :
  (oranges + apples + limes) * 2 = 288 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_fruits_l2094_209470


namespace NUMINAMATH_GPT_min_abs_A_l2094_209448

def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def A (a d : ℚ) (n : ℕ) : ℚ :=
  (arithmetic_sequence a d n) + (arithmetic_sequence a d (n + 1)) + 
  (arithmetic_sequence a d (n + 2)) + (arithmetic_sequence a d (n + 3)) + 
  (arithmetic_sequence a d (n + 4)) + (arithmetic_sequence a d (n + 5)) + 
  (arithmetic_sequence a d (n + 6))

theorem min_abs_A : (arithmetic_sequence 19 (-4/5) 26 = -1) ∧ 
                    (∀ n, 1 ≤ n) →
                    ∃ n : ℕ, |A 19 (-4/5) n| = 7/5 :=
by
  sorry

end NUMINAMATH_GPT_min_abs_A_l2094_209448


namespace NUMINAMATH_GPT_greatest_possible_sum_of_consecutive_integers_prod_lt_200_l2094_209467

theorem greatest_possible_sum_of_consecutive_integers_prod_lt_200 :
  ∃ n : ℤ, (n * (n + 1) < 200) ∧ ( ∀ m : ℤ, (m * (m + 1) < 200) → m ≤ n) ∧ (n + (n + 1) = 27) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_consecutive_integers_prod_lt_200_l2094_209467


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l2094_209408

variables (a b x m : ℝ)
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variables (h_x : x = a + 0.25 * a)
variables (h_m : m = b - 0.80 * b)
variables (h_ratio : m / x = 0.2)

theorem ratio_of_a_to_b (h_pos_a : 0 < a) (h_pos_b : 0 < b)
                        (h_x : x = a + 0.25 * a)
                        (h_m : m = b - 0.80 * b)
                        (h_ratio : m / x = 0.2) :
  a / b = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l2094_209408


namespace NUMINAMATH_GPT_total_insects_l2094_209435

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) (caterpillars_every_third_leaf : ℕ) :
  leaves = 84 →
  ladybugs_per_leaf = 139 →
  ants_per_leaf = 97 →
  caterpillars_every_third_leaf = 53 →
  (84 * 139) + (84 * 97) + (53 * (84 / 3)) = 21308 := 
by
  sorry

end NUMINAMATH_GPT_total_insects_l2094_209435


namespace NUMINAMATH_GPT_fill_tank_time_l2094_209489

-- Define the rates at which the pipes fill or empty the tank
def rateA : ℚ := 1 / 16
def rateB : ℚ := - (1 / 24)  -- Since pipe B empties the tank, it's negative.

-- Define the time after which pipe B is closed
def timeBClosed : ℚ := 21

-- Define the initial combined rate of both pipes
def combinedRate : ℚ := rateA + rateB

-- Define the proportion of the tank filled in the initial 21 minutes
def filledIn21Minutes : ℚ := combinedRate * timeBClosed

-- Define the remaining tank to be filled after pipe B is closed
def remainingTank : ℚ := 1 - filledIn21Minutes

-- Define the additional time required to fill the remaining part of the tank with only pipe A
def additionalTime : ℚ := remainingTank / rateA

-- Total time is the sum of the initial time and additional time
def totalTime : ℚ := timeBClosed + additionalTime

theorem fill_tank_time : totalTime = 30 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fill_tank_time_l2094_209489


namespace NUMINAMATH_GPT_square_area_given_equal_perimeters_l2094_209486

theorem square_area_given_equal_perimeters 
  (a b c : ℝ) (a_eq : a = 7.5) (b_eq : b = 9.5) (c_eq : c = 12) 
  (sq_perimeter_eq_tri : 4 * s = a + b + c) : 
  s^2 = 52.5625 :=
by
  sorry

end NUMINAMATH_GPT_square_area_given_equal_perimeters_l2094_209486


namespace NUMINAMATH_GPT_hyperbola_vertex_distance_l2094_209461

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0 →
  2 = 2 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_vertex_distance_l2094_209461


namespace NUMINAMATH_GPT_original_cost_price_of_car_l2094_209494

theorem original_cost_price_of_car (x : ℝ) (y : ℝ) (h1 : y = 0.87 * x) (h2 : 1.20 * y = 54000) :
  x = 54000 / 1.044 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_of_car_l2094_209494


namespace NUMINAMATH_GPT_not_subset_T_to_S_l2094_209407

def is_odd (x : ℤ) : Prop := ∃ n : ℤ, x = 2 * n + 1
def is_of_form_4k_plus_1 (y : ℤ) : Prop := ∃ k : ℤ, y = 4 * k + 1

theorem not_subset_T_to_S :
  ¬ (∀ y, is_of_form_4k_plus_1 y → is_odd y) :=
sorry

end NUMINAMATH_GPT_not_subset_T_to_S_l2094_209407


namespace NUMINAMATH_GPT_solve_inequality_l2094_209473

theorem solve_inequality : {x : ℝ | 3 * x ^ 2 - 7 * x - 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2094_209473


namespace NUMINAMATH_GPT_recurring_division_l2094_209497

def recurring_36_as_fraction : ℚ := 36 / 99
def recurring_12_as_fraction : ℚ := 12 / 99

theorem recurring_division :
  recurring_36_as_fraction / recurring_12_as_fraction = 3 := 
sorry

end NUMINAMATH_GPT_recurring_division_l2094_209497


namespace NUMINAMATH_GPT_maci_red_pens_l2094_209469

def cost_blue_pens (b : ℕ) (cost_blue : ℕ) : ℕ := b * cost_blue

def cost_red_pen (cost_blue : ℕ) : ℕ := 2 * cost_blue

def total_cost (cost_blue : ℕ) (n_blue : ℕ) (n_red : ℕ) : ℕ := 
  n_blue * cost_blue + n_red * (2 * cost_blue)

theorem maci_red_pens :
  ∀ (n_blue cost_blue n_red total : ℕ),
  n_blue = 10 →
  cost_blue = 10 →
  total = 400 →
  total_cost cost_blue n_blue n_red = total →
  n_red = 15 := 
by
  intros n_blue cost_blue n_red total h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_maci_red_pens_l2094_209469


namespace NUMINAMATH_GPT_quotient_of_N_div_3_l2094_209430

-- Define the number N
def N : ℕ := 7 * 12 + 4

-- Statement we need to prove
theorem quotient_of_N_div_3 : N / 3 = 29 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_N_div_3_l2094_209430


namespace NUMINAMATH_GPT_Elza_winning_strategy_l2094_209441

-- Define a hypothetical graph structure
noncomputable def cities := {i : ℕ // 1 ≤ i ∧ i ≤ 2013}
def connected (c1 c2 : cities) : Prop := sorry

theorem Elza_winning_strategy 
  (N : ℕ) 
  (roads : (cities × cities) → Prop) 
  (h1 : ∀ c1 c2, roads (c1, c2) → connected c1 c2)
  (h2 : N = 1006): 
  ∃ (strategy : cities → Prop), 
  (∃ c1 c2 : cities, (strategy c1 ∧ strategy c2)) ∧ connected c1 c2 :=
by 
  sorry

end NUMINAMATH_GPT_Elza_winning_strategy_l2094_209441


namespace NUMINAMATH_GPT_range_of_independent_variable_x_l2094_209451

noncomputable def range_of_x (x : ℝ) : Prop :=
  x > -2

theorem range_of_independent_variable_x (x : ℝ) :
  ∀ x, (x + 2 > 0) → range_of_x x :=
by
  intro x h
  unfold range_of_x
  linarith

end NUMINAMATH_GPT_range_of_independent_variable_x_l2094_209451


namespace NUMINAMATH_GPT_polynomial_transformation_exists_l2094_209429

theorem polynomial_transformation_exists (P : ℝ → ℝ → ℝ) (hP : ∀ x y, P (x - 1) (y - 2 * x + 1) = P x y) :
  ∃ Φ : ℝ → ℝ, ∀ x y, P x y = Φ (y - x^2) := by
  sorry

end NUMINAMATH_GPT_polynomial_transformation_exists_l2094_209429


namespace NUMINAMATH_GPT_mean_score_l2094_209411

variable (mean stddev : ℝ)

-- Conditions
axiom condition1 : 42 = mean - 5 * stddev
axiom condition2 : 67 = mean + 2.5 * stddev

theorem mean_score : mean = 58.67 := 
by 
  -- You would need to provide proof here
  sorry

end NUMINAMATH_GPT_mean_score_l2094_209411


namespace NUMINAMATH_GPT_condition_on_a_and_b_l2094_209481

theorem condition_on_a_and_b (a b p q : ℝ) 
    (h1 : (∀ x : ℝ, (x + a) * (x + b) = x^2 + p * x + q))
    (h2 : p > 0)
    (h3 : q < 0) :
    (a < 0 ∧ b > 0 ∧ b > -a) ∨ (a > 0 ∧ b < 0 ∧ a > -b) :=
by
  sorry

end NUMINAMATH_GPT_condition_on_a_and_b_l2094_209481


namespace NUMINAMATH_GPT_AB_ratio_CD_l2094_209463

variable (AB CD : ℝ)
variable (h : ℝ)
variable (O : Point)
variable (ABCD_isosceles : IsIsoscelesTrapezoid AB CD)
variable (areas_condition : List ℝ) 
-- where the list areas_condition represents: [S_OCD, S_OBC, S_OAB, S_ODA]

theorem AB_ratio_CD : 
  ABCD_isosceles ∧ areas_condition = [2, 3, 4, 5] → AB = 2 * CD :=
by
  sorry

end NUMINAMATH_GPT_AB_ratio_CD_l2094_209463


namespace NUMINAMATH_GPT_equation_of_line_l2094_209427

theorem equation_of_line (l : ℝ → ℝ) :
  (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x : ℝ, l x = (2 * l a / a) * x))
  ∨ (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x y : ℝ, 2 * x + y - 4 = 0)) := sorry

end NUMINAMATH_GPT_equation_of_line_l2094_209427


namespace NUMINAMATH_GPT_slope_of_intersection_points_l2094_209453

theorem slope_of_intersection_points {s x y : ℝ} 
  (h1 : 2 * x - 3 * y = 6 * s - 5) 
  (h2 : 3 * x + y = 9 * s + 4) : 
  ∃ m : ℝ, m = 3 ∧ (∀ s : ℝ, (∃ x y : ℝ, 2 * x - 3 * y = 6 * s - 5 ∧ 3 * x + y = 9 * s + 4) → y = m * x + (23/11)) := 
by
  sorry

end NUMINAMATH_GPT_slope_of_intersection_points_l2094_209453


namespace NUMINAMATH_GPT_total_students_mrs_mcgillicuddy_l2094_209438

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_students_mrs_mcgillicuddy_l2094_209438


namespace NUMINAMATH_GPT_log_inequality_l2094_209447

open Real

theorem log_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log (1 + sqrt (a * b)) ≤ (1 / 2) * (log (1 + a) + log (1 + b)) :=
sorry

end NUMINAMATH_GPT_log_inequality_l2094_209447


namespace NUMINAMATH_GPT_old_camera_model_cost_l2094_209472

theorem old_camera_model_cost (C new_model_cost discounted_lens_cost : ℝ)
  (h1 : new_model_cost = 1.30 * C)
  (h2 : discounted_lens_cost = 200)
  (h3 : new_model_cost + discounted_lens_cost = 5400)
  : C = 4000 := by
sorry

end NUMINAMATH_GPT_old_camera_model_cost_l2094_209472


namespace NUMINAMATH_GPT_shrimp_per_pound_l2094_209443

theorem shrimp_per_pound (shrimp_per_guest guests : ℕ) (cost_per_pound : ℝ) (total_spent : ℝ)
  (hshrimp_per_guest : shrimp_per_guest = 5) (hguests : guests = 40) (hcost_per_pound : cost_per_pound = 17.0) (htotal_spent : total_spent = 170.0) :
  let total_shrimp := shrimp_per_guest * guests
  let total_pounds := total_spent / cost_per_pound
  total_shrimp / total_pounds = 20 :=
by
  sorry

end NUMINAMATH_GPT_shrimp_per_pound_l2094_209443


namespace NUMINAMATH_GPT_length_of_wall_correct_l2094_209422

noncomputable def length_of_wall (s : ℝ) (w : ℝ) : ℝ :=
  let area_mirror := s * s
  let area_wall := 2 * area_mirror
  area_wall / w

theorem length_of_wall_correct : length_of_wall 18 32 = 20.25 :=
by
  -- This is the place for proof which is omitted deliberately
  sorry

end NUMINAMATH_GPT_length_of_wall_correct_l2094_209422


namespace NUMINAMATH_GPT_inequality_solution_l2094_209471

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) → a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2094_209471


namespace NUMINAMATH_GPT_find_x_satisfies_equation_l2094_209418

theorem find_x_satisfies_equation :
  let x : ℤ := -14
  ∃ x : ℤ, (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x)) :=
by
  let x := -14
  use x
  sorry

end NUMINAMATH_GPT_find_x_satisfies_equation_l2094_209418


namespace NUMINAMATH_GPT_number_division_l2094_209444

theorem number_division (n : ℕ) (h1 : 555 + 445 = 1000) (h2 : 555 - 445 = 110) 
  (h3 : n % 1000 = 80) (h4 : n / 1000 = 220) : n = 220080 :=
by {
  -- proof steps would go here
  sorry
}

end NUMINAMATH_GPT_number_division_l2094_209444


namespace NUMINAMATH_GPT_beach_trip_time_l2094_209424

noncomputable def totalTripTime (driveTime eachWay : ℝ) (beachTimeFactor : ℝ) : ℝ :=
  let totalDriveTime := eachWay * 2
  totalDriveTime + (totalDriveTime * beachTimeFactor)

theorem beach_trip_time :
  totalTripTime 2 2 2.5 = 14 := 
by
  sorry

end NUMINAMATH_GPT_beach_trip_time_l2094_209424
