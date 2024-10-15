import Mathlib

namespace NUMINAMATH_GPT_order_of_t_t2_neg_t_l788_78849

theorem order_of_t_t2_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t :=
by
  sorry

end NUMINAMATH_GPT_order_of_t_t2_neg_t_l788_78849


namespace NUMINAMATH_GPT_intersection_line_through_circles_l788_78817

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - 14 = 0

theorem intersection_line_through_circles : 
  (∀ x y : ℝ, circle1_equation x y → circle2_equation x y → x + y - 2 = 0) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_intersection_line_through_circles_l788_78817


namespace NUMINAMATH_GPT_mean_score_is_93_l788_78800

-- Define Jane's scores as a list
def scores : List ℕ := [98, 97, 92, 85, 93]

-- Define the mean of the scores
noncomputable def mean (lst : List ℕ) : ℚ := 
  (lst.foldl (· + ·) 0 : ℚ) / lst.length

-- The theorem to prove
theorem mean_score_is_93 : mean scores = 93 := by
  sorry

end NUMINAMATH_GPT_mean_score_is_93_l788_78800


namespace NUMINAMATH_GPT_game_ends_in_36_rounds_l788_78870

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

end NUMINAMATH_GPT_game_ends_in_36_rounds_l788_78870


namespace NUMINAMATH_GPT_age_of_youngest_child_l788_78891

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_age_of_youngest_child_l788_78891


namespace NUMINAMATH_GPT_product_mod_7_zero_l788_78807

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end NUMINAMATH_GPT_product_mod_7_zero_l788_78807


namespace NUMINAMATH_GPT_alcohol_quantity_l788_78805

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 4 / 3) (h2 : A / (W + 8) = 4 / 5) : A = 16 := 
by
  sorry

end NUMINAMATH_GPT_alcohol_quantity_l788_78805


namespace NUMINAMATH_GPT_radar_placement_and_coverage_area_l788_78882

noncomputable def max_distance (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (15 : ℝ) / Real.sin (Real.pi / n)

noncomputable def coverage_area (n : ℕ) (r : ℝ) (w : ℝ) : ℝ :=
  (480 : ℝ) * Real.pi / Real.tan (Real.pi / n)

theorem radar_placement_and_coverage_area 
  (n : ℕ) (r w : ℝ) (hn : n = 8) (hr : r = 17) (hw : w = 16) :
  max_distance n r w = (15 : ℝ) / Real.sin (Real.pi / 8) ∧
  coverage_area n r w = (480 : ℝ) * Real.pi / Real.tan (Real.pi / 8) :=
by
  sorry

end NUMINAMATH_GPT_radar_placement_and_coverage_area_l788_78882


namespace NUMINAMATH_GPT_correct_option_l788_78816

-- Definition of the conditions
def conditionA : Prop := (Real.sqrt ((-1 : ℝ)^2) = 1)
def conditionB : Prop := (Real.sqrt ((-1 : ℝ)^2) = -1)
def conditionC : Prop := (Real.sqrt (-(1^2) : ℝ) = 1)
def conditionD : Prop := (Real.sqrt (-(1^2) : ℝ) = -1)

-- Proving the correct condition
theorem correct_option : conditionA := by
  sorry

end NUMINAMATH_GPT_correct_option_l788_78816


namespace NUMINAMATH_GPT_bridget_apples_l788_78852

theorem bridget_apples (x : ℕ) (h1 : x - 2 ≥ 0) (h2 : (x - 2) / 3 = 0 → false)
    (h3 : (2 * (x - 2) / 3) - 5 = 6) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_bridget_apples_l788_78852


namespace NUMINAMATH_GPT_select_eight_genuine_dinars_l788_78878

theorem select_eight_genuine_dinars (coins : Fin 11 → ℝ) :
  (∃ (fake_coin : Option (Fin 11)), 
    ((∀ i j : Fin 11, i ≠ j → coins i = coins j) ∨
    (∀ (genuine_coins impostor_coins : Finset (Fin 11)), 
      genuine_coins ∪ impostor_coins = Finset.univ →
      impostor_coins.card = 1 →
      (∃ difference : ℝ, ∀ i ∈ genuine_coins, coins i = difference) ∧
      (∃ i ∈ impostor_coins, coins i ≠ difference)))) →
  (∃ (selected_coins : Finset (Fin 11)), selected_coins.card = 8 ∧
   (∀ i j : Fin 11, i ∈ selected_coins → j ∈ selected_coins → coins i = coins j)) :=
sorry

end NUMINAMATH_GPT_select_eight_genuine_dinars_l788_78878


namespace NUMINAMATH_GPT_trig_identity_proof_l788_78876

theorem trig_identity_proof :
  let sin240 := - (Real.sin (120 * Real.pi / 180))
  let tan240 := Real.tan (240 * Real.pi / 180)
  Real.sin (600 * Real.pi / 180) + tan240 = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l788_78876


namespace NUMINAMATH_GPT_expansion_three_times_expansion_six_times_l788_78856

-- Definition for the rule of expansion
def expand (a b : Nat) : Nat := a * b + a + b

-- Problem 1: Expansion with a = 1, b = 3 for 3 times results in 255.
theorem expansion_three_times : expand (expand (expand 1 3) 7) 31 = 255 := sorry

-- Problem 2: After 6 operations, the expanded number matches the given pattern.
theorem expansion_six_times (p q : ℕ) (hp : p > q) (hq : q > 0) : 
  ∃ m n, m = 8 ∧ n = 13 ∧ (expand (expand (expand (expand (expand (expand q (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) = (q + 1) ^ m * (p + 1) ^ n - 1 :=
sorry

end NUMINAMATH_GPT_expansion_three_times_expansion_six_times_l788_78856


namespace NUMINAMATH_GPT_pyramid_base_side_length_l788_78881

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l788_78881


namespace NUMINAMATH_GPT_solution_set_of_inequality_l788_78828

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_deriv_neg : ∀ x : ℝ, 0 < x → (x^2 + 1) * deriv f x + 2 * x * f x < 0)
  (h_f_neg1_zero : f (-1) = 0) :
  { x : ℝ | f x > 0 } = { x | x < -1 } ∪ { x | 0 < x ∧ x < 1 } := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l788_78828


namespace NUMINAMATH_GPT_min_sum_x_y_l788_78840

theorem min_sum_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y ≥ 9 :=
by sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y = 9 ↔ (x = 3 ∧ y = 6) :=
by sorry

end NUMINAMATH_GPT_min_sum_x_y_l788_78840


namespace NUMINAMATH_GPT_proof_sin_315_eq_neg_sqrt_2_div_2_l788_78873

noncomputable def sin_315_eq_neg_sqrt_2_div_2 : Prop :=
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2)

theorem proof_sin_315_eq_neg_sqrt_2_div_2 : sin_315_eq_neg_sqrt_2_div_2 := 
  by
    sorry

end NUMINAMATH_GPT_proof_sin_315_eq_neg_sqrt_2_div_2_l788_78873


namespace NUMINAMATH_GPT_reams_for_haley_correct_l788_78835

-- Definitions: 
-- total reams = 5
-- reams for sister = 3
-- reams for Haley = ?

def total_reams : Nat := 5
def reams_for_sister : Nat := 3
def reams_for_haley : Nat := total_reams - reams_for_sister

-- The proof problem: prove reams_for_haley = 2 given the conditions.
theorem reams_for_haley_correct : reams_for_haley = 2 := by 
  sorry

end NUMINAMATH_GPT_reams_for_haley_correct_l788_78835


namespace NUMINAMATH_GPT_integer_roots_p_l788_78813

theorem integer_roots_p (p x1 x2 : ℤ) (h1 : x1 * x2 = p + 4) (h2 : x1 + x2 = -p) : p = 8 ∨ p = -4 := 
sorry

end NUMINAMATH_GPT_integer_roots_p_l788_78813


namespace NUMINAMATH_GPT_find_starting_number_l788_78810

theorem find_starting_number (n : ℤ) (h1 : ∀ k : ℤ, n ≤ k ∧ k ≤ 38 → k % 4 = 0) (h2 : (n + 38) / 2 = 22) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_starting_number_l788_78810


namespace NUMINAMATH_GPT_average_weight_increase_l788_78804

theorem average_weight_increase 
  (n : ℕ) (old_weight new_weight : ℝ) (group_size := 8) 
  (old_weight := 70) (new_weight := 90) : 
  ((new_weight - old_weight) / group_size) = 2.5 := 
by sorry

end NUMINAMATH_GPT_average_weight_increase_l788_78804


namespace NUMINAMATH_GPT_equal_sum_sequence_a18_l788_78895

theorem equal_sum_sequence_a18
    (a : ℕ → ℕ)
    (h1 : a 1 = 2)
    (h2 : ∀ n, a n + a (n + 1) = 5) :
    a 18 = 3 :=
sorry

end NUMINAMATH_GPT_equal_sum_sequence_a18_l788_78895


namespace NUMINAMATH_GPT_tetrahedron_volume_is_zero_l788_78869

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  (1 / 6) * p * q * r

theorem tetrahedron_volume_is_zero (p q r : ℝ)
  (hpq : p^2 + q^2 = 36)
  (hqr : q^2 + r^2 = 64)
  (hrp : r^2 + p^2 = 100) :
  volume_of_tetrahedron p q r = 0 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_is_zero_l788_78869


namespace NUMINAMATH_GPT_meters_examined_l788_78853

theorem meters_examined (x : ℝ) (h1 : 0.07 / 100 * x = 2) : x = 2857 :=
by
  -- using the given setup and simplification
  sorry

end NUMINAMATH_GPT_meters_examined_l788_78853


namespace NUMINAMATH_GPT_quadratic_trinomial_constant_l788_78877

theorem quadratic_trinomial_constant (m : ℝ) (h : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
sorry

end NUMINAMATH_GPT_quadratic_trinomial_constant_l788_78877


namespace NUMINAMATH_GPT_math_problem_l788_78822

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end NUMINAMATH_GPT_math_problem_l788_78822


namespace NUMINAMATH_GPT_geometric_sequence_increasing_condition_l788_78844

theorem geometric_sequence_increasing_condition (a₁ a₂ a₄ : ℝ) (q : ℝ) (n : ℕ) (a : ℕ → ℝ):
  (∀ n, a n = a₁ * q^n) →
  (a₁ < a₂ ∧ a₂ < a₄) → 
  ¬ (∀ n, a n < a (n + 1)) → 
  (a₁ < a₂ ∧ a₂ < a₄) ∧ ¬ (∀ n, a n < a (n + 1)) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_condition_l788_78844


namespace NUMINAMATH_GPT_employees_excluding_manager_l788_78837

theorem employees_excluding_manager (average_salary average_increase manager_salary n : ℕ)
  (h_avg_salary : average_salary = 2400)
  (h_avg_increase : average_increase = 100)
  (h_manager_salary : manager_salary = 4900)
  (h_new_avg_salary : average_salary + average_increase = 2500)
  (h_total_salary : (n + 1) * (average_salary + average_increase) = n * average_salary + manager_salary) :
  n = 24 :=
by
  sorry

end NUMINAMATH_GPT_employees_excluding_manager_l788_78837


namespace NUMINAMATH_GPT_randy_initial_blocks_l788_78825

theorem randy_initial_blocks (used_blocks left_blocks total_blocks : ℕ) (h1 : used_blocks = 19) (h2 : left_blocks = 59) : total_blocks = used_blocks + left_blocks → total_blocks = 78 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_randy_initial_blocks_l788_78825


namespace NUMINAMATH_GPT_problem_solution_l788_78857

theorem problem_solution (a b c d x : ℚ) 
  (h1 : 2 * a + 2 = x) 
  (h2 : 3 * b + 3 = x) 
  (h3 : 4 * c + 4 = x) 
  (h4 : 5 * d + 5 = x) 
  (h5 : 2 * a + 3 * b + 4 * c + 5 * d + 6 = x) 
  : 2 * a + 3 * b + 4 * c + 5 * d = -10 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l788_78857


namespace NUMINAMATH_GPT_three_digit_number_is_504_l788_78890

theorem three_digit_number_is_504 (x : ℕ) [Decidable (x = 504)] :
  100 ≤ x ∧ x ≤ 999 →
  (x - 7) % 7 = 0 ∧
  (x - 8) % 8 = 0 ∧
  (x - 9) % 9 = 0 →
  x = 504 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_is_504_l788_78890


namespace NUMINAMATH_GPT_max_value_of_a_l788_78872

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) → a ≤ 5 :=
by sorry

end NUMINAMATH_GPT_max_value_of_a_l788_78872


namespace NUMINAMATH_GPT_half_angle_in_quadrant_l788_78809

theorem half_angle_in_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 90 < α ∧ α < k * 360 + 180) :
  ∃ n : ℤ, (n * 360 + 45 < α / 2 ∧ α / 2 < n * 360 + 90) ∨ (n * 360 + 225 < α / 2 ∧ α / 2 < n * 360 + 270) :=
by sorry

end NUMINAMATH_GPT_half_angle_in_quadrant_l788_78809


namespace NUMINAMATH_GPT_pyramid_volume_l788_78808

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

end NUMINAMATH_GPT_pyramid_volume_l788_78808


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l788_78818

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

end NUMINAMATH_GPT_ratio_of_men_to_women_l788_78818


namespace NUMINAMATH_GPT_min_cube_edge_division_l788_78892

theorem min_cube_edge_division (n : ℕ) (h : n^3 ≥ 1996) : n = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_cube_edge_division_l788_78892


namespace NUMINAMATH_GPT_age_of_eldest_child_l788_78829

theorem age_of_eldest_child (age_sum : ∀ (x : ℕ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 40) :
  ∃ x, x + 8 = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_of_eldest_child_l788_78829


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l788_78897

def M := {x : ℝ | 3 * x - x^2 > 0}
def N := {x : ℝ | x^2 - 4 * x + 3 > 0}
def I := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = I :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l788_78897


namespace NUMINAMATH_GPT_difference_of_cubes_not_div_by_twice_diff_l788_78899

theorem difference_of_cubes_not_div_by_twice_diff (a b : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_neq : a ≠ b) :
  ¬ (2 * (a - b)) ∣ ((a^3) - (b^3)) := 
sorry

end NUMINAMATH_GPT_difference_of_cubes_not_div_by_twice_diff_l788_78899


namespace NUMINAMATH_GPT_additional_people_required_l788_78834

-- Given condition: Four people can mow a lawn in 6 hours
def work_rate: ℕ := 4 * 6

-- New condition: Number of people needed to mow the lawn in 3 hours
def people_required_in_3_hours: ℕ := work_rate / 3

-- Statement: Number of additional people required
theorem additional_people_required : people_required_in_3_hours - 4 = 4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_additional_people_required_l788_78834


namespace NUMINAMATH_GPT_geometric_sum_equals_fraction_l788_78894

theorem geometric_sum_equals_fraction (n : ℕ) (a r : ℝ) 
  (h_a : a = 1) (h_r : r = 1 / 2) 
  (h_sum : a * (1 - r^n) / (1 - r) = 511 / 512) : 
  n = 9 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sum_equals_fraction_l788_78894


namespace NUMINAMATH_GPT_earnings_difference_l788_78803

def total_earnings : ℕ := 3875
def first_job_earnings : ℕ := 2125
def second_job_earnings := total_earnings - first_job_earnings

theorem earnings_difference : (first_job_earnings - second_job_earnings) = 375 := by
  sorry

end NUMINAMATH_GPT_earnings_difference_l788_78803


namespace NUMINAMATH_GPT_boxes_difference_l788_78833

theorem boxes_difference (white_balls red_balls balls_per_box : ℕ)
  (h_white : white_balls = 30)
  (h_red : red_balls = 18)
  (h_box : balls_per_box = 6) :
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_boxes_difference_l788_78833


namespace NUMINAMATH_GPT_pump_fill_time_without_leak_l788_78861

theorem pump_fill_time_without_leak
    (P : ℝ)
    (h1 : 2 + 1/7 = (15:ℝ)/7)
    (h2 : 1 / P - 1 / 30 = 7 / 15) :
  P = 2 := by
  sorry

end NUMINAMATH_GPT_pump_fill_time_without_leak_l788_78861


namespace NUMINAMATH_GPT_average_pages_per_book_l788_78883

theorem average_pages_per_book :
  let pages := [120, 150, 180, 210, 240]
  let num_books := 5
  let total_pages := pages.sum
  total_pages / num_books = 180 := by
  sorry

end NUMINAMATH_GPT_average_pages_per_book_l788_78883


namespace NUMINAMATH_GPT_correct_idiom_l788_78848

-- Define the conditions given in the problem
def context := "The vast majority of office clerks read a significant amount of materials"
def idiom_usage := "to say _ of additional materials"

-- Define the proof problem
theorem correct_idiom (context: String) (idiom_usage: String) : idiom_usage.replace "_ of additional materials" "nothing of newspapers and magazines" = "to say nothing of newspapers and magazines" :=
sorry

end NUMINAMATH_GPT_correct_idiom_l788_78848


namespace NUMINAMATH_GPT_cleaning_time_is_100_l788_78862

def time_hosing : ℕ := 10
def time_shampoo_per : ℕ := 15
def num_shampoos : ℕ := 3
def time_drying : ℕ := 20
def time_brushing : ℕ := 25

def total_time : ℕ :=
  time_hosing + (num_shampoos * time_shampoo_per) + time_drying + time_brushing

theorem cleaning_time_is_100 :
  total_time = 100 :=
by
  sorry

end NUMINAMATH_GPT_cleaning_time_is_100_l788_78862


namespace NUMINAMATH_GPT_small_bottles_initial_l788_78811

theorem small_bottles_initial
  (S : ℤ)
  (big_bottles_initial : ℤ := 15000)
  (sold_small_bottles_percentage : ℚ := 0.11)
  (sold_big_bottles_percentage : ℚ := 0.12)
  (remaining_bottles_in_storage : ℤ := 18540)
  (remaining_small_bottles : ℚ := 0.89 * S)
  (remaining_big_bottles : ℚ := 0.88 * big_bottles_initial)
  (h : remaining_small_bottles + remaining_big_bottles = remaining_bottles_in_storage)
  : S = 6000 :=
by
  sorry

end NUMINAMATH_GPT_small_bottles_initial_l788_78811


namespace NUMINAMATH_GPT_initial_pens_count_l788_78843

theorem initial_pens_count (P : ℕ) (h : 2 * (P + 22) - 19 = 75) : P = 25 :=
by
  sorry

end NUMINAMATH_GPT_initial_pens_count_l788_78843


namespace NUMINAMATH_GPT_intersection_A_B_l788_78864

def setA : Set ℝ := {x | 0 < x}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}
def intersectionAB : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = intersectionAB := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l788_78864


namespace NUMINAMATH_GPT_parallelogram_base_length_l788_78824

theorem parallelogram_base_length (area height : ℝ) (h_area : area = 108) (h_height : height = 9) : 
  ∃ base : ℝ, base = area / height ∧ base = 12 := 
  by sorry

end NUMINAMATH_GPT_parallelogram_base_length_l788_78824


namespace NUMINAMATH_GPT_greatest_possible_remainder_l788_78806

theorem greatest_possible_remainder (x : ℕ) (h: x % 7 ≠ 0) : (∃ r < 7, r = x % 7) ∧ x % 7 ≤ 6 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_remainder_l788_78806


namespace NUMINAMATH_GPT_quadratic_rewrite_as_square_of_binomial_plus_integer_l788_78821

theorem quadratic_rewrite_as_square_of_binomial_plus_integer :
    ∃ a b, ∀ x, x^2 + 16 * x + 72 = (x + a)^2 + b ∧ b = 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_as_square_of_binomial_plus_integer_l788_78821


namespace NUMINAMATH_GPT_fg_of_3_eq_79_l788_78859

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_79_l788_78859


namespace NUMINAMATH_GPT_find_correct_speed_l788_78812

-- Definitions for given conditions
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions as definitions
def condition1 (d t : ℝ) : Prop := distance_traveled 35 (t + (5 / 60)) = d
def condition2 (d t : ℝ) : Prop := distance_traveled 55 (t - (5 / 60)) = d

-- Statement to prove
theorem find_correct_speed (d t r : ℝ) (h1 : condition1 d t) (h2 : condition2 d t) :
  r = (d / t) ∧ r = 42.78 :=
by sorry

end NUMINAMATH_GPT_find_correct_speed_l788_78812


namespace NUMINAMATH_GPT_moe_share_of_pie_l788_78871

-- Definitions based on conditions
def leftover_pie : ℚ := 8 / 9
def num_people : ℚ := 3

-- Theorem to prove the amount of pie Moe took home
theorem moe_share_of_pie : (leftover_pie / num_people) = 8 / 27 := by
  sorry

end NUMINAMATH_GPT_moe_share_of_pie_l788_78871


namespace NUMINAMATH_GPT_max_value_fraction_l788_78820

theorem max_value_fraction (a b x y : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a^x = 3) (h4 : b^y = 3) (h5 : a + b = 2 * Real.sqrt 3) :
  1/x + 1/y ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_fraction_l788_78820


namespace NUMINAMATH_GPT_find_first_number_in_sequence_l788_78846

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

end NUMINAMATH_GPT_find_first_number_in_sequence_l788_78846


namespace NUMINAMATH_GPT_number_of_schools_l788_78838

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

end NUMINAMATH_GPT_number_of_schools_l788_78838


namespace NUMINAMATH_GPT_annual_income_before_tax_l788_78854

variable (I : ℝ) -- Define I as the annual income before tax

-- Conditions
def original_tax (I : ℝ) : ℝ := 0.42 * I
def new_tax (I : ℝ) : ℝ := 0.32 * I
def differential_savings (I : ℝ) : ℝ := original_tax I - new_tax I

-- Theorem: Given the conditions, the taxpayer's annual income before tax is $42,400
theorem annual_income_before_tax : differential_savings I = 4240 → I = 42400 := by
  sorry

end NUMINAMATH_GPT_annual_income_before_tax_l788_78854


namespace NUMINAMATH_GPT_min_m_value_arithmetic_seq_l788_78801

theorem min_m_value_arithmetic_seq :
  ∀ (a S : ℕ → ℚ) (m : ℕ),
  (∀ n : ℕ, a (n+2) = 5 ∧ a (n+6) = 21) →
  (∀ n : ℕ, S (n+1) = S n + 1 / a (n+1)) →
  (∀ n : ℕ, S (2 * n + 1) - S n ≤ m / 15) →
  ∀ n : ℕ, m = 5 :=
sorry

end NUMINAMATH_GPT_min_m_value_arithmetic_seq_l788_78801


namespace NUMINAMATH_GPT_isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l788_78855

-- Part 1 proof
theorem isosceles_triangle_sides_part1 (x : ℝ) (h1 : x + 2 * x + 2 * x = 20) : 
  x = 4 ∧ 2 * x = 8 :=
by
  sorry

-- Part 2 proof
theorem isosceles_triangle_sides_part2 (a b : ℝ) (h2 : a = 5) (h3 : 2 * b + a = 20) :
  b = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l788_78855


namespace NUMINAMATH_GPT_coordinates_of_M_l788_78858

theorem coordinates_of_M :
  -- Given the function f(x) = 2x^2 + 1
  let f : Real → Real := λ x => 2 * x^2 + 1
  -- And its derivative
  let f' : Real → Real := λ x => 4 * x
  -- The coordinates of point M where the instantaneous rate of change is -8 are (-2, 9)
  (∃ x0 : Real, f' x0 = -8 ∧ f x0 = y0 ∧ x0 = -2 ∧ y0 = 9) := by
    sorry

end NUMINAMATH_GPT_coordinates_of_M_l788_78858


namespace NUMINAMATH_GPT_quadrant_of_tan_and_cos_l788_78884

theorem quadrant_of_tan_and_cos (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ Q, (Q = 2) :=
by
  sorry


end NUMINAMATH_GPT_quadrant_of_tan_and_cos_l788_78884


namespace NUMINAMATH_GPT_number_of_8_tuples_l788_78842

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

end NUMINAMATH_GPT_number_of_8_tuples_l788_78842


namespace NUMINAMATH_GPT_tan_eleven_pi_over_three_l788_78826

theorem tan_eleven_pi_over_three : Real.tan (11 * Real.pi / 3) = -Real.sqrt 3 := 
    sorry

end NUMINAMATH_GPT_tan_eleven_pi_over_three_l788_78826


namespace NUMINAMATH_GPT_choir_grouping_l788_78830

theorem choir_grouping (sopranos altos tenors basses : ℕ)
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18)
  (ratio : ℕ) :
  ratio = 1 →
  ∃ G : ℕ, G ≤ 10 ∧ G ≤ 15 ∧ G ≤ 12 ∧ 2 * G ≤ 18 ∧ G = 9 :=
by sorry

end NUMINAMATH_GPT_choir_grouping_l788_78830


namespace NUMINAMATH_GPT_temperature_decrease_l788_78893

theorem temperature_decrease (rise_1_degC : ℝ) (decrease_2_degC : ℝ) 
  (h : rise_1_degC = 1) : decrease_2_degC = -2 :=
by 
  -- This is the statement with the condition and problem to be proven:
  sorry

end NUMINAMATH_GPT_temperature_decrease_l788_78893


namespace NUMINAMATH_GPT_scientific_notation_932700_l788_78839

theorem scientific_notation_932700 : 932700 = 9.327 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_932700_l788_78839


namespace NUMINAMATH_GPT_population_time_interval_l788_78887

theorem population_time_interval (T : ℕ) 
  (birth_rate : ℕ) (death_rate : ℕ) (net_increase_day : ℕ) (seconds_in_day : ℕ)
  (h_birth_rate : birth_rate = 8) 
  (h_death_rate : death_rate = 6) 
  (h_net_increase_day : net_increase_day = 86400)
  (h_seconds_in_day : seconds_in_day = 86400) : 
  T = 2 := sorry

end NUMINAMATH_GPT_population_time_interval_l788_78887


namespace NUMINAMATH_GPT_polygon_sides_l788_78867

theorem polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_l788_78867


namespace NUMINAMATH_GPT_f_zero_unique_l788_78879

theorem f_zero_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x + f (xy)) : f 0 = 0 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_f_zero_unique_l788_78879


namespace NUMINAMATH_GPT_solution_set_of_inequality_l788_78832

theorem solution_set_of_inequality (x : ℝ) : 
  (|x| * (1 - 2 * x) > 0) ↔ (x ∈ ((Set.Iio 0) ∪ (Set.Ioo 0 (1/2)))) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l788_78832


namespace NUMINAMATH_GPT_circle_area_isosceles_triangle_l788_78874

noncomputable def circle_area (a b c : ℝ) (is_isosceles : a = b ∧ (4 = a ∨ 4 = b) ∧ c = 3) : ℝ := sorry

theorem circle_area_isosceles_triangle :
  circle_area 4 4 3 ⟨rfl,Or.inl rfl, rfl⟩ = (64 / 13.75) * Real.pi := by
sorry

end NUMINAMATH_GPT_circle_area_isosceles_triangle_l788_78874


namespace NUMINAMATH_GPT_find_n_l788_78860

theorem find_n (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (h2 : b n = 0)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 :=
sorry

end NUMINAMATH_GPT_find_n_l788_78860


namespace NUMINAMATH_GPT_clever_value_points_l788_78827

def clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = (deriv f) x₀

theorem clever_value_points :
  (clever_value_point (fun x : ℝ => x^2)) ∧
  (clever_value_point (fun x : ℝ => Real.log x)) ∧
  (clever_value_point (fun x : ℝ => x + (1 / x))) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_clever_value_points_l788_78827


namespace NUMINAMATH_GPT_xyz_inequality_l788_78814

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l788_78814


namespace NUMINAMATH_GPT_probability_is_1_over_90_l788_78847

/-- Probability Calculation -/
noncomputable def probability_of_COLD :=
  (1 / (Nat.choose 5 3)) * (2 / 3) * (1 / (Nat.choose 4 2))

theorem probability_is_1_over_90 :
  probability_of_COLD = (1 / 90) :=
by
  sorry

end NUMINAMATH_GPT_probability_is_1_over_90_l788_78847


namespace NUMINAMATH_GPT_find_p_l788_78802

theorem find_p (p : ℤ)
  (h1 : ∀ (u v : ℤ), u > 0 → v > 0 → 5 * u ^ 2 - 5 * p * u + (66 * p - 1) = 0 ∧
    5 * v ^ 2 - 5 * p * v + (66 * p - 1) = 0) :
  p = 76 :=
sorry

end NUMINAMATH_GPT_find_p_l788_78802


namespace NUMINAMATH_GPT_custom_op_example_l788_78866

def custom_op (a b : ℕ) : ℕ := (a + 1) / b

theorem custom_op_example : custom_op 2 (custom_op 3 4) = 3 := 
by
  sorry

end NUMINAMATH_GPT_custom_op_example_l788_78866


namespace NUMINAMATH_GPT_no_valid_prime_angles_l788_78850

def is_prime (n : ℕ) : Prop := Prime n

theorem no_valid_prime_angles :
  ∀ (x : ℕ), (x < 30) ∧ is_prime x ∧ is_prime (3 * x) → False :=
by sorry

end NUMINAMATH_GPT_no_valid_prime_angles_l788_78850


namespace NUMINAMATH_GPT_find_f2_l788_78815

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1)
  (H2 : f 8 = 15) :
  f 2 = 3 := 
sorry

end NUMINAMATH_GPT_find_f2_l788_78815


namespace NUMINAMATH_GPT_person_A_takes_12_more_minutes_l788_78875

-- Define distances, speeds, times
variables (S : ℝ) (v_A v_B : ℝ) (t : ℝ)

-- Define conditions as hypotheses
def conditions (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t) : Prop :=
  (v_A * (t + 4/5) = 2/3 * S) ∧ (v_B * t = 2/3 * S) ∧ (v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S)

-- The proof problem statement
theorem person_A_takes_12_more_minutes
  (S : ℝ) (v_A v_B : ℝ) (t : ℝ)
  (h1 : t = 2/5) (h2 : v_A = (2/3) * S / (t + 4/5)) (h3 : v_B = (2/3) * S / t)
  (h4 : conditions S v_A v_B t h1 h2 h3) : (t + 4/5) + 6/5 = 96 / 60 + 12 / 60 :=
sorry

end NUMINAMATH_GPT_person_A_takes_12_more_minutes_l788_78875


namespace NUMINAMATH_GPT_part_cost_l788_78831

theorem part_cost (hours : ℕ) (hourly_rate total_paid : ℕ) 
  (h1 : hours = 2)
  (h2 : hourly_rate = 75)
  (h3 : total_paid = 300) : 
  total_paid - (hours * hourly_rate) = 150 := 
by
  sorry

end NUMINAMATH_GPT_part_cost_l788_78831


namespace NUMINAMATH_GPT_toothpicks_in_arithmetic_sequence_l788_78868

theorem toothpicks_in_arithmetic_sequence :
  let a1 := 5
  let d := 3
  let n := 15
  let a_n n := a1 + (n - 1) * d
  let sum_to_n n := n * (2 * a1 + (n - 1) * d) / 2
  sum_to_n n = 390 := by
  sorry

end NUMINAMATH_GPT_toothpicks_in_arithmetic_sequence_l788_78868


namespace NUMINAMATH_GPT_volume_of_cone_l788_78841

theorem volume_of_cone (d h : ℝ) (d_eq : d = 12) (h_eq : h = 9) : 
  (1 / 3) * π * (d / 2)^2 * h = 108 * π := 
by 
  rw [d_eq, h_eq] 
  sorry

end NUMINAMATH_GPT_volume_of_cone_l788_78841


namespace NUMINAMATH_GPT_question1_question2_question3_l788_78888

-- Question 1
theorem question1 (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2) :
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Question 2
theorem question2 (x m n: ℕ) (h : x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) :
  (m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7) :=
sorry

-- Question 3
theorem question3 : Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_question1_question2_question3_l788_78888


namespace NUMINAMATH_GPT_no_solution_intervals_l788_78885

theorem no_solution_intervals (a : ℝ) :
  (a < -13 ∨ a > 0) → ¬ ∃ x : ℝ, 6 * abs (x - 4 * a) + abs (x - a^2) + 5 * x - 3 * a = 0 :=
by sorry

end NUMINAMATH_GPT_no_solution_intervals_l788_78885


namespace NUMINAMATH_GPT_power_mod_8_l788_78880

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end NUMINAMATH_GPT_power_mod_8_l788_78880


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l788_78896

theorem simplify_and_evaluate_expression (x : ℤ) (hx : x = 3) : 
  (1 - (x / (x + 1))) / ((x^2 - 2 * x + 1) / (x^2 - 1)) = 1 / 2 := by
  rw [hx]
  -- Here we perform the necessary rewrites and simplifications as shown in the steps
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l788_78896


namespace NUMINAMATH_GPT_remaining_funds_correct_l788_78823

def david_initial_funds : ℝ := 1800
def emma_initial_funds : ℝ := 2400
def john_initial_funds : ℝ := 1200

def david_spent_percentage : ℝ := 0.60
def emma_spent_percentage : ℝ := 0.75
def john_spent_percentage : ℝ := 0.50

def david_remaining_funds : ℝ := david_initial_funds * (1 - david_spent_percentage)
def emma_spent : ℝ := emma_initial_funds * emma_spent_percentage
def emma_remaining_funds : ℝ := emma_spent - 800
def john_remaining_funds : ℝ := john_initial_funds * (1 - john_spent_percentage)

theorem remaining_funds_correct :
  david_remaining_funds = 720 ∧
  emma_remaining_funds = 1400 ∧
  john_remaining_funds = 600 :=
by
  sorry

end NUMINAMATH_GPT_remaining_funds_correct_l788_78823


namespace NUMINAMATH_GPT_ratio_diamond_brace_ring_l788_78865

theorem ratio_diamond_brace_ring
  (cost_ring : ℤ) (cost_car : ℤ) (total_worth : ℤ) (cost_diamond_brace : ℤ)
  (h1 : cost_ring = 4000) (h2 : cost_car = 2000) (h3 : total_worth = 14000)
  (h4 : cost_diamond_brace = total_worth - (cost_ring + cost_car)) :
  cost_diamond_brace / cost_ring = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_diamond_brace_ring_l788_78865


namespace NUMINAMATH_GPT_jasmine_percentage_after_adding_l788_78889

def initial_solution_volume : ℕ := 80
def initial_jasmine_percentage : ℝ := 0.10
def additional_jasmine_volume : ℕ := 5
def additional_water_volume : ℕ := 15

theorem jasmine_percentage_after_adding :
  let initial_jasmine_volume := initial_jasmine_percentage * initial_solution_volume
  let total_jasmine_volume := initial_jasmine_volume + additional_jasmine_volume
  let total_solution_volume := initial_solution_volume + additional_jasmine_volume + additional_water_volume
  let final_jasmine_percentage := (total_jasmine_volume / total_solution_volume) * 100
  final_jasmine_percentage = 13 := by
  sorry

end NUMINAMATH_GPT_jasmine_percentage_after_adding_l788_78889


namespace NUMINAMATH_GPT_find_ratio_l788_78863

theorem find_ratio (x y c d : ℝ) (h₁ : 4 * x - 2 * y = c) (h₂ : 5 * y - 10 * x = d) (h₃ : d ≠ 0) : c / d = 0 :=
sorry

end NUMINAMATH_GPT_find_ratio_l788_78863


namespace NUMINAMATH_GPT_sum_of_distances_l788_78851

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_1 = 1 / 9 * d_2) (h2 : d_1 + d_2 = 6) : d_1 + d_2 + 6 = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_l788_78851


namespace NUMINAMATH_GPT_ratio_of_areas_of_triangle_and_trapezoid_l788_78819

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

theorem ratio_of_areas_of_triangle_and_trapezoid :
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  (a_small / a_trapezoid) = (1 / 3) :=
by
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  have h : (a_small / a_trapezoid) = (1 / 3) := 
    by sorry  -- Here would be the proof steps, but we're skipping
  exact h

end NUMINAMATH_GPT_ratio_of_areas_of_triangle_and_trapezoid_l788_78819


namespace NUMINAMATH_GPT_correct_multiplication_l788_78898

theorem correct_multiplication (n : ℕ) (h₁ : 15 * n = 45) : 5 * n = 15 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_correct_multiplication_l788_78898


namespace NUMINAMATH_GPT_probability_of_no_adjacent_standing_is_123_over_1024_l788_78845

def total_outcomes : ℕ := 2 ^ 10

 -- Define the recursive sequence a_n
def a : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n

lemma a_10_val : a 10 = 123 := by
  sorry

def probability_no_adjacent_standing (n : ℕ): ℚ :=
  a n / total_outcomes

theorem probability_of_no_adjacent_standing_is_123_over_1024 :
  probability_no_adjacent_standing 10 = 123 / 1024 := by
  rw [probability_no_adjacent_standing, total_outcomes, a_10_val]
  norm_num

end NUMINAMATH_GPT_probability_of_no_adjacent_standing_is_123_over_1024_l788_78845


namespace NUMINAMATH_GPT_derivative_at_one_l788_78836

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_at_one : deriv f 1 = -1 := sorry

end NUMINAMATH_GPT_derivative_at_one_l788_78836


namespace NUMINAMATH_GPT_leoCurrentWeight_l788_78886

def currentWeightProblem (L K : Real) : Prop :=
  (L + 15 = 1.75 * K) ∧ (L + K = 250)

theorem leoCurrentWeight (L K : Real) (h : currentWeightProblem L K) : L = 154 :=
by
  sorry

end NUMINAMATH_GPT_leoCurrentWeight_l788_78886
