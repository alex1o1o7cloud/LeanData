import Mathlib

namespace NUMINAMATH_GPT_cosine_third_angle_of_triangle_l424_42419

theorem cosine_third_angle_of_triangle (X Y Z : ℝ)
  (sinX_eq : Real.sin X = 4/5)
  (cosY_eq : Real.cos Y = 12/13)
  (triangle_sum : X + Y + Z = Real.pi) :
  Real.cos Z = -16/65 :=
by
  -- proof will be filled in
  sorry

end NUMINAMATH_GPT_cosine_third_angle_of_triangle_l424_42419


namespace NUMINAMATH_GPT_literate_employees_l424_42472

theorem literate_employees (num_illiterate : ℕ) (wage_decrease_per_illiterate : ℕ)
  (total_average_salary_decrease : ℕ) : num_illiterate = 35 → 
                                        wage_decrease_per_illiterate = 25 →
                                        total_average_salary_decrease = 15 →
                                        ∃ L : ℕ, L = 23 :=
by {
  -- given: num_illiterate = 35
  -- given: wage_decrease_per_illiterate = 25
  -- given: total_average_salary_decrease = 15
  sorry
}

end NUMINAMATH_GPT_literate_employees_l424_42472


namespace NUMINAMATH_GPT_pi_bounds_l424_42412

theorem pi_bounds :
  3 < Real.pi ∧ Real.pi < 4 :=
by
  sorry

end NUMINAMATH_GPT_pi_bounds_l424_42412


namespace NUMINAMATH_GPT_find_k_l424_42451

theorem find_k (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 ↔ |k * x - 4| ≤ 2) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l424_42451


namespace NUMINAMATH_GPT_shane_gum_left_l424_42473

def elyse_initial_gum : ℕ := 100
def half (x : ℕ) := x / 2
def rick_gum : ℕ := half elyse_initial_gum
def shane_initial_gum : ℕ := half rick_gum
def chewed_gum : ℕ := 11

theorem shane_gum_left : shane_initial_gum - chewed_gum = 14 := by
  sorry

end NUMINAMATH_GPT_shane_gum_left_l424_42473


namespace NUMINAMATH_GPT_ball_distribution_l424_42413

theorem ball_distribution :
  ∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ x1 x2 x3, f x1 x2 x3 → x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) ∧
    (∃ (count : ℕ), (count = 15) ∧ (∀ x1 x2 x3, f x1 x2 x3 → count = 15)) :=
sorry

end NUMINAMATH_GPT_ball_distribution_l424_42413


namespace NUMINAMATH_GPT_apples_in_each_crate_l424_42421

theorem apples_in_each_crate
  (num_crates : ℕ) 
  (num_rotten : ℕ) 
  (num_boxes : ℕ) 
  (apples_per_box : ℕ) 
  (total_good_apples : ℕ) 
  (total_apples : ℕ)
  (h1 : num_crates = 12) 
  (h2 : num_rotten = 160) 
  (h3 : num_boxes = 100) 
  (h4 : apples_per_box = 20) 
  (h5 : total_good_apples = num_boxes * apples_per_box) 
  (h6 : total_apples = total_good_apples + num_rotten) : 
  total_apples / num_crates = 180 := 
by 
  sorry

end NUMINAMATH_GPT_apples_in_each_crate_l424_42421


namespace NUMINAMATH_GPT_number_of_cats_l424_42499

theorem number_of_cats 
  (n k : ℕ)
  (h1 : n * k = 999919)
  (h2 : k > n) :
  n = 991 :=
sorry

end NUMINAMATH_GPT_number_of_cats_l424_42499


namespace NUMINAMATH_GPT_principal_sum_l424_42408

noncomputable def diff_simple_compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
(P * ((1 + r / 100)^t) - P) - (P * r * t / 100)

theorem principal_sum (P : ℝ) (r : ℝ) (t : ℝ) (h : diff_simple_compound_interest P r t = 631) (hr : r = 10) (ht : t = 2) :
    P = 63100 := by
  sorry

end NUMINAMATH_GPT_principal_sum_l424_42408


namespace NUMINAMATH_GPT_log_inequality_l424_42404

theorem log_inequality {a x : ℝ} (h1 : 0 < x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1) : 
  abs (Real.logb a (1 - x)) > abs (Real.logb a (1 + x)) :=
sorry

end NUMINAMATH_GPT_log_inequality_l424_42404


namespace NUMINAMATH_GPT_tangent_intersects_x_axis_l424_42477

theorem tangent_intersects_x_axis (x0 x1 : ℝ) (hx : ∀ x : ℝ, x1 = x0 - 1) :
  x1 - x0 = -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_intersects_x_axis_l424_42477


namespace NUMINAMATH_GPT_find_sin_value_l424_42449

variable (x : ℝ)

theorem find_sin_value (h : Real.sin (x + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * Real.pi / 3 - x) = Real.sqrt 3 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_sin_value_l424_42449


namespace NUMINAMATH_GPT_intersection_M_N_l424_42465

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := 
sorry

end NUMINAMATH_GPT_intersection_M_N_l424_42465


namespace NUMINAMATH_GPT_rental_plans_count_l424_42443

-- Define the number of large buses, medium buses, and the total number of people.
def num_large_buses := 42
def num_medium_buses := 25
def total_people := 1511

-- State the theorem to prove that there are exactly 2 valid rental plans.
theorem rental_plans_count (x y : ℕ) :
  (num_large_buses * x + num_medium_buses * y = total_people) →
  (∃! (x y : ℕ), num_large_buses * x + num_medium_buses * y = total_people) :=
by
  sorry

end NUMINAMATH_GPT_rental_plans_count_l424_42443


namespace NUMINAMATH_GPT_common_roots_l424_42434

noncomputable def p (x a : ℝ) := x^3 + a * x^2 + 14 * x + 7
noncomputable def q (x b : ℝ) := x^3 + b * x^2 + 21 * x + 15

theorem common_roots (a b : ℝ) (r s : ℝ) (hr : r ≠ s)
  (hp : p r a = 0) (hp' : p s a = 0)
  (hq : q r b = 0) (hq' : q s b = 0) :
  a = 5 ∧ b = 4 :=
by sorry

end NUMINAMATH_GPT_common_roots_l424_42434


namespace NUMINAMATH_GPT_bruce_michael_total_goals_l424_42416

theorem bruce_michael_total_goals (bruce_goals : ℕ) (michael_goals : ℕ) 
  (h₁ : bruce_goals = 4) (h₂ : michael_goals = 3 * bruce_goals) : bruce_goals + michael_goals = 16 :=
by sorry

end NUMINAMATH_GPT_bruce_michael_total_goals_l424_42416


namespace NUMINAMATH_GPT_remitted_amount_is_correct_l424_42483

-- Define the constants and conditions of the problem
def total_sales : ℝ := 32500
def commission_rate1 : ℝ := 0.05
def commission_limit : ℝ := 10000
def commission_rate2 : ℝ := 0.04

-- Define the function to calculate the remitted amount
def remitted_amount (total_sales commission_rate1 commission_limit commission_rate2 : ℝ) : ℝ :=
  let commission1 := commission_rate1 * commission_limit
  let remaining_sales := total_sales - commission_limit
  let commission2 := commission_rate2 * remaining_sales
  total_sales - (commission1 + commission2)

-- Lean statement to prove the remitted amount
theorem remitted_amount_is_correct :
  remitted_amount total_sales commission_rate1 commission_limit commission_rate2 = 31100 :=
by
  sorry

end NUMINAMATH_GPT_remitted_amount_is_correct_l424_42483


namespace NUMINAMATH_GPT_textbook_weight_l424_42474

theorem textbook_weight
  (w : ℝ)
  (bookcase_limit : ℝ := 80)
  (hardcover_books : ℕ := 70)
  (hardcover_weight_per_book : ℝ := 0.5)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (knick_knack_weight : ℝ := 6)
  (over_limit : ℝ := 33)
  (total_items_weight : ℝ := bookcase_limit + over_limit)
  (hardcover_total_weight : ℝ := hardcover_books * hardcover_weight_per_book)
  (knick_knack_total_weight : ℝ := knick_knacks * knick_knack_weight)
  (remaining_weight : ℝ := total_items_weight - (hardcover_total_weight + knick_knack_total_weight)) :
  remaining_weight = textbooks * 2 :=
by
  sorry

end NUMINAMATH_GPT_textbook_weight_l424_42474


namespace NUMINAMATH_GPT_fraction_of_fliers_sent_out_l424_42415

-- Definitions based on the conditions
def total_fliers : ℕ := 2500
def fliers_next_day : ℕ := 1500

-- Defining the fraction sent in the morning as x
variable (x : ℚ)

-- The remaining fliers after morning
def remaining_fliers_morning := (1 - x) * total_fliers

-- The remaining fliers after afternoon
def remaining_fliers_afternoon := remaining_fliers_morning - (1/4) * remaining_fliers_morning

-- The theorem statement
theorem fraction_of_fliers_sent_out :
  remaining_fliers_afternoon = fliers_next_day → x = 1/5 :=
sorry

end NUMINAMATH_GPT_fraction_of_fliers_sent_out_l424_42415


namespace NUMINAMATH_GPT_system_solution_l424_42439

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l424_42439


namespace NUMINAMATH_GPT_length_of_first_platform_l424_42447

theorem length_of_first_platform 
  (train_length : ℕ) (first_time : ℕ) (second_platform_length : ℕ) (second_time : ℕ)
  (speed_first : ℕ) (speed_second : ℕ) :
  train_length = 230 → 
  first_time = 15 → 
  second_platform_length = 250 → 
  second_time = 20 → 
  speed_first = (train_length + L) / first_time →
  speed_second = (train_length + second_platform_length) / second_time →
  speed_first = speed_second →
  (L : ℕ) = 130 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_platform_l424_42447


namespace NUMINAMATH_GPT_midpoint_translation_l424_42450

theorem midpoint_translation (x1 y1 x2 y2 tx ty mx my : ℤ) 
  (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 5) (hy2 : y2 = -7)
  (htx : tx = 3) (hty : ty = -4)
  (hmx : mx = (x1 + x2) / 2 + tx) (hmy : my = (y1 + y2) / 2 + ty) : 
  mx = 6 ∧ my = -6 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_translation_l424_42450


namespace NUMINAMATH_GPT_sum_a4_a5_a6_l424_42435

section ArithmeticSequence

variable {a : ℕ → ℝ}

-- Condition 1: The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Condition 2: Given information
axiom a2_a8_eq_6 : a 2 + a 8 = 6

-- Question: Prove that a 4 + a 5 + a 6 = 9
theorem sum_a4_a5_a6 : is_arithmetic_sequence a → a 4 + a 5 + a 6 = 9 :=
by
  intro h_arith
  sorry

end ArithmeticSequence

end NUMINAMATH_GPT_sum_a4_a5_a6_l424_42435


namespace NUMINAMATH_GPT_interest_earned_after_4_years_l424_42410

noncomputable def calculate_total_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let A := P * (1 + r) ^ t
  A - P

theorem interest_earned_after_4_years :
  calculate_total_interest 2000 0.12 4 = 1147.04 :=
by
  sorry

end NUMINAMATH_GPT_interest_earned_after_4_years_l424_42410


namespace NUMINAMATH_GPT_parallel_lines_l424_42436

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l424_42436


namespace NUMINAMATH_GPT_unique_rectangles_l424_42442

theorem unique_rectangles (a b x y : ℝ) (h_dim : a < b) 
    (h_perimeter : 2 * (x + y) = a + b)
    (h_area : x * y = (a * b) / 2) : 
    (∃ x y : ℝ, (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 2) ∧ (x < a) ∧ (y < b)) → 
    (∃! z w : ℝ, (2 * (z + w) = a + b) ∧ (z * y = (a * b) / 2) ∧ (z < a) ∧ (w < b)) :=
sorry

end NUMINAMATH_GPT_unique_rectangles_l424_42442


namespace NUMINAMATH_GPT_sector_central_angle_l424_42406

noncomputable def sector_radius (r l : ℝ) : Prop :=
2 * r + l = 10

noncomputable def sector_area (r l : ℝ) : Prop :=
(1 / 2) * l * r = 4

noncomputable def central_angle (α r l : ℝ) : Prop :=
α = l / r

theorem sector_central_angle (r l α : ℝ) 
  (h1 : sector_radius r l) 
  (h2 : sector_area r l) 
  (h3 : central_angle α r l) : 
  α = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l424_42406


namespace NUMINAMATH_GPT_correct_propositions_l424_42467

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l424_42467


namespace NUMINAMATH_GPT_speed_in_still_water_l424_42425

namespace SwimmingProblem

variable (V_m V_s : ℝ)

-- Downstream condition
def downstream_condition : Prop := V_m + V_s = 18

-- Upstream condition
def upstream_condition : Prop := V_m - V_s = 13

-- The main theorem stating the problem
theorem speed_in_still_water (h_downstream : downstream_condition V_m V_s) 
                             (h_upstream : upstream_condition V_m V_s) :
    V_m = 15.5 :=
by
  sorry

end SwimmingProblem

end NUMINAMATH_GPT_speed_in_still_water_l424_42425


namespace NUMINAMATH_GPT_value_of_t_l424_42459

noncomputable def f (x t k : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem value_of_t (a b t k : ℝ) (h1 : 0 < t) (h2 : 0 < k) 
  (h3 : a + b = t) (h4 : a * b = k) (h5 : 2 * a = b - 2) (h6 : (-2)^2 = a * b) : 
  t = 5 := 
  sorry

end NUMINAMATH_GPT_value_of_t_l424_42459


namespace NUMINAMATH_GPT_distinct_four_digit_numbers_product_18_l424_42471

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end NUMINAMATH_GPT_distinct_four_digit_numbers_product_18_l424_42471


namespace NUMINAMATH_GPT_find_GQ_in_triangle_XYZ_l424_42426

noncomputable def GQ_in_triangle_XYZ_centroid : ℝ :=
  let XY := 13
  let XZ := 15
  let YZ := 24
  let centroid_ratio := 1 / 3
  let semi_perimeter := (XY + XZ + YZ) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - XY) * (semi_perimeter - XZ) * (semi_perimeter - YZ))
  let heightXR := (2 * area) / YZ
  (heightXR * centroid_ratio)

theorem find_GQ_in_triangle_XYZ :
  GQ_in_triangle_XYZ_centroid = 2.4 :=
sorry

end NUMINAMATH_GPT_find_GQ_in_triangle_XYZ_l424_42426


namespace NUMINAMATH_GPT_num_ways_to_buy_three_items_l424_42414

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end NUMINAMATH_GPT_num_ways_to_buy_three_items_l424_42414


namespace NUMINAMATH_GPT_product_of_roots_l424_42478

-- Define the quadratic function in terms of a, b, c
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the conditions
variables (a b c y : ℝ)

-- Given conditions from the problem
def condition_1 := ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2
def condition_2 := quadratic a b c y = 0
def condition_3 := quadratic a b c (4 * y) = 0

-- The statement to be proved
theorem product_of_roots (a b c y : ℝ) 
  (h1: ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2)
  (h2: quadratic a b c y = 0) 
  (h3: quadratic a b c (4 * y) = 0) :
  ∃ x1 x2, (quadratic a b c x = 0 → (x1 = y ∧ x2 = 4 * y) ∨ (x1 = 4 * y ∧ x2 = y)) ∧ x1 * x2 = 4 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l424_42478


namespace NUMINAMATH_GPT_intersection_point_l424_42486

def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 2 * t, 0, -1 + 3 * t)

def plane (x y z : ℝ) : Prop := x + 4 * y + 13 * z - 23 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (-1 - 2 * t) 0 (-1 + 3 * t) ∧ parametric_line t = (-3, 0, 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l424_42486


namespace NUMINAMATH_GPT_paper_clips_in_2_cases_l424_42454

variable (c b : ℕ)

theorem paper_clips_in_2_cases : 2 * (c * b) * 600 = (2 * c * b * 600) := by
  sorry

end NUMINAMATH_GPT_paper_clips_in_2_cases_l424_42454


namespace NUMINAMATH_GPT_perpendicular_lines_l424_42440

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l424_42440


namespace NUMINAMATH_GPT_line_points_satisfy_equation_l424_42433

theorem line_points_satisfy_equation (x_2 y_3 : ℝ) 
  (h_slope : ∃ k : ℝ, k = 2) 
  (h_P1 : ∃ P1 : ℝ × ℝ, P1 = (3, 5)) 
  (h_P2 : ∃ P2 : ℝ × ℝ, P2 = (x_2, 7)) 
  (h_P3 : ∃ P3 : ℝ × ℝ, P3 = (-1, y_3)) 
  (h_line : ∀ (x y : ℝ), y - 5 = 2 * (x - 3) ↔ 2 * x - y - 1 = 0) :
  x_2 = 4 ∧ y_3 = -3 :=
sorry

end NUMINAMATH_GPT_line_points_satisfy_equation_l424_42433


namespace NUMINAMATH_GPT_melissa_work_hours_l424_42438

variable (f : ℝ) (f_d : ℝ) (h_d : ℝ)

theorem melissa_work_hours (hf : f = 56) (hfd : f_d = 4) (hhd : h_d = 3) : 
  (f / f_d) * h_d = 42 := by
  sorry

end NUMINAMATH_GPT_melissa_work_hours_l424_42438


namespace NUMINAMATH_GPT_child_B_share_l424_42407

theorem child_B_share (total_money : ℕ) (ratio_A ratio_B ratio_C ratio_D ratio_E total_parts : ℕ) 
  (h1 : total_money = 12000)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 4)
  (h5 : ratio_D = 5)
  (h6 : ratio_E = 6)
  (h_total_parts : total_parts = ratio_A + ratio_B + ratio_C + ratio_D + ratio_E) :
  (total_money / total_parts) * ratio_B = 1800 :=
by
  sorry

end NUMINAMATH_GPT_child_B_share_l424_42407


namespace NUMINAMATH_GPT_number_of_paths_to_spell_BINGO_l424_42480

theorem number_of_paths_to_spell_BINGO : 
  ∃ (paths : ℕ), paths = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_paths_to_spell_BINGO_l424_42480


namespace NUMINAMATH_GPT_max_gold_coins_l424_42489

theorem max_gold_coins (k : ℤ) (h1 : ∃ k : ℤ, 15 * k + 3 < 120) : 
  ∃ n : ℤ, n = 15 * k + 3 ∧ n < 120 ∧ n = 108 :=
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l424_42489


namespace NUMINAMATH_GPT_min_value_fraction_l424_42492

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℝ, x = (a / (a + 2 * b) + b / (a + b)) ∧ x ≥ 1 - 1 / (2 * sqrt 2) ∧ x = 1 - 1 / (2 * sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l424_42492


namespace NUMINAMATH_GPT_number_of_books_l424_42469

-- Define the conditions
def ratio_books : ℕ := 7
def ratio_pens : ℕ := 3
def ratio_notebooks : ℕ := 2
def total_items : ℕ := 600

-- Define the theorem and the goal to prove
theorem number_of_books (sets : ℕ) (ratio_books : ℕ := 7) (total_items : ℕ := 600) : 
  sets = total_items / (7 + 3 + 2) → 
  sets * ratio_books = 350 :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_l424_42469


namespace NUMINAMATH_GPT_unique_triangle_exists_l424_42462

theorem unique_triangle_exists : 
  (¬ (∀ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = 3 → a + b > c)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 30 → ∃ (C : ℝ), C > 0)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 100 → ∃ (C : ℝ), C > 0)) ∧
  (∀ (b c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45 → ∃! (a c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45) :=
by sorry

end NUMINAMATH_GPT_unique_triangle_exists_l424_42462


namespace NUMINAMATH_GPT_find_c_l424_42482

-- Define conditions as Lean statements
theorem find_c :
  ∀ (c n : ℝ), 
  (n ^ 2 + 1 / 16 = 1 / 4) → 
  2 * n = c → 
  c < 0 → 
  c = - (Real.sqrt 3) / 2 :=
by
  intros c n h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_c_l424_42482


namespace NUMINAMATH_GPT_train_speed_problem_l424_42453

theorem train_speed_problem (l1 l2 : ℝ) (v2 : ℝ) (t : ℝ) (v1 : ℝ) :
  l1 = 120 → l2 = 280 → v2 = 30 → t = 19.99840012798976 →
  0.4 / (t / 3600) = v1 + v2 → v1 = 42 :=
by
  intros hl1 hl2 hv2 ht hrel
  rw [hl1, hl2, hv2, ht] at *
  sorry

end NUMINAMATH_GPT_train_speed_problem_l424_42453


namespace NUMINAMATH_GPT_chord_length_l424_42437

theorem chord_length
  (l_eq : ∀ (rho theta : ℝ), rho * (Real.sin theta - Real.cos theta) = 1)
  (gamma_eq : ∀ (rho : ℝ) (theta : ℝ), rho = 1) :
  ∃ AB : ℝ, AB = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l424_42437


namespace NUMINAMATH_GPT_tom_total_payment_l424_42497

theorem tom_total_payment :
  let apples_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let bananas_cost := 12 * 30
  let grapes_cost := 7 * 45
  let cherries_cost := 4 * 80
  apples_cost + mangoes_cost + oranges_cost + bananas_cost + grapes_cost + cherries_cost = 2250 :=
by
  sorry

end NUMINAMATH_GPT_tom_total_payment_l424_42497


namespace NUMINAMATH_GPT_g_at_1001_l424_42428

open Function

variable (g : ℝ → ℝ)

axiom g_property : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_at_1 : g 1 = 3

theorem g_at_1001 : g 1001 = -997 :=
by
  sorry

end NUMINAMATH_GPT_g_at_1001_l424_42428


namespace NUMINAMATH_GPT_largest_convex_ngon_with_integer_tangents_l424_42463

-- Definitions of conditions and the statement
def isConvex (n : ℕ) : Prop := n ≥ 3 -- Condition 1: n is at least 3
def isConvexPolygon (n : ℕ) : Prop := isConvex n -- Condition 2: the polygon is convex
def tanInteriorAnglesAreIntegers (n : ℕ) : Prop := true -- Placeholder for Condition 3

-- Statement to prove
theorem largest_convex_ngon_with_integer_tangents : 
  ∀ n : ℕ, isConvexPolygon n → tanInteriorAnglesAreIntegers n → n ≤ 8 :=
by
  intros n h_convex h_tangents
  sorry

end NUMINAMATH_GPT_largest_convex_ngon_with_integer_tangents_l424_42463


namespace NUMINAMATH_GPT_maximum_value_of_function_y_l424_42452

noncomputable def function_y (x : ℝ) : ℝ :=
  x * (3 - 2 * x)

theorem maximum_value_of_function_y : ∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ function_y x = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_function_y_l424_42452


namespace NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l424_42470

theorem base_angle_of_isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) (h_isosceles : A = B ∨ B = C ∨ A = C) (h_angle : A = 42 ∨ B = 42 ∨ C = 42) :
  A = 42 ∨ A = 69 ∨ B = 42 ∨ B = 69 ∨ C = 42 ∨ C = 69 :=
by
  sorry

end NUMINAMATH_GPT_base_angle_of_isosceles_triangle_l424_42470


namespace NUMINAMATH_GPT_four_digit_num_condition_l424_42401

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end NUMINAMATH_GPT_four_digit_num_condition_l424_42401


namespace NUMINAMATH_GPT_area_of_paper_l424_42409

-- Define the variables and conditions
variable (L W : ℝ)
variable (h1 : 2 * L + 4 * W = 34)
variable (h2 : 4 * L + 2 * W = 38)

-- Statement to prove
theorem area_of_paper : L * W = 35 := 
by
  sorry

end NUMINAMATH_GPT_area_of_paper_l424_42409


namespace NUMINAMATH_GPT_avg_height_correct_l424_42405

theorem avg_height_correct (h1 h2 h3 h4 : ℝ) (h_distinct: h1 ≠ h2 ∧ h2 ≠ h3 ∧ h3 ≠ h4 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h4)
  (h_tallest: h4 = 152) (h_shortest: h1 = 137) 
  (h4_largest: h4 > h3 ∧ h4 > h2 ∧ h4 > h1) (h1_smallest: h1 < h2 ∧ h1 < h3 ∧ h1 < h4) :
  ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg := 
sorry

end NUMINAMATH_GPT_avg_height_correct_l424_42405


namespace NUMINAMATH_GPT_least_tiles_needed_l424_42445

-- Define the conditions
def hallway_length_ft : ℕ := 18
def hallway_width_ft : ℕ := 6
def tile_side_in : ℕ := 6
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Translate conditions
def hallway_length_in := feet_to_inches hallway_length_ft
def hallway_width_in := feet_to_inches hallway_width_ft

-- Define the areas
def hallway_area : ℕ := hallway_length_in * hallway_width_in
def tile_area : ℕ := tile_side_in * tile_side_in

-- State the theorem to be proved
theorem least_tiles_needed :
  hallway_area / tile_area = 432 := 
sorry

end NUMINAMATH_GPT_least_tiles_needed_l424_42445


namespace NUMINAMATH_GPT_trisect_54_degree_angle_l424_42488

theorem trisect_54_degree_angle :
  ∃ (a1 a2 : ℝ), a1 = 18 ∧ a2 = 36 ∧ a1 + a2 + a2 = 54 :=
by sorry

end NUMINAMATH_GPT_trisect_54_degree_angle_l424_42488


namespace NUMINAMATH_GPT_angle_measure_of_E_l424_42403

theorem angle_measure_of_E (E F G H : ℝ) 
  (h1 : E = 3 * F) 
  (h2 : E = 4 * G) 
  (h3 : E = 6 * H) 
  (h_sum : E + F + G + H = 360) : 
  E = 206 := 
by 
  sorry

end NUMINAMATH_GPT_angle_measure_of_E_l424_42403


namespace NUMINAMATH_GPT_underachievers_l424_42455

-- Define the variables for the number of students in each group
variables (a b c : ℕ)

-- Given conditions as hypotheses
axiom total_students : a + b + c = 30
axiom top_achievers : a = 19
axiom average_students : c = 12

-- Prove the number of underachievers
theorem underachievers : b = 9 :=
by sorry

end NUMINAMATH_GPT_underachievers_l424_42455


namespace NUMINAMATH_GPT_total_apples_after_transactions_l424_42420

def initial_apples : ℕ := 65
def percentage_used : ℕ := 20
def apples_bought : ℕ := 15

theorem total_apples_after_transactions :
  (initial_apples * (1 - percentage_used / 100)) + apples_bought = 67 := 
by
  sorry

end NUMINAMATH_GPT_total_apples_after_transactions_l424_42420


namespace NUMINAMATH_GPT_conical_tank_volume_l424_42432

theorem conical_tank_volume
  (diameter : ℝ) (height : ℝ) (depth_linear : ∀ x : ℝ, 0 ≤ x ∧ x ≤ diameter / 2 → height - (height / (diameter / 2)) * x = 0) :
  diameter = 20 → height = 6 → (1 / 3) * Real.pi * (10 ^ 2) * height = 200 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_conical_tank_volume_l424_42432


namespace NUMINAMATH_GPT_problem_one_problem_two_l424_42424

-- Define the given vectors
def vector_oa : ℝ × ℝ := (-1, 3)
def vector_ob : ℝ × ℝ := (3, -1)
def vector_oc (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the subtraction of two 2D vectors
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Define the parallel condition (u and v are parallel if u = k*v for some scalar k)
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1  -- equivalent to u = k*v

-- Define the dot product in 2D
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Problem 1
theorem problem_one (m : ℝ) :
  is_parallel (vector_sub vector_ob vector_oa) (vector_oc m) ↔ m = -1 :=
by
-- Proof omitted
sorry

-- Problem 2
theorem problem_two (m : ℝ) :
  dot_product (vector_sub (vector_oc m) vector_oa) (vector_sub (vector_oc m) vector_ob) = 0 ↔
  m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_problem_one_problem_two_l424_42424


namespace NUMINAMATH_GPT_roots_equality_l424_42494

noncomputable def problem_statement (α β γ δ p q : ℝ) : Prop :=
(α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2 * p - 3 * q) ^ 2

theorem roots_equality (α β γ δ p q : ℝ)
  (h₁ : ∀ x, x^2 - 2 * p * x + 3 = 0 → (x = α ∨ x = β))
  (h₂ : ∀ x, x^2 - 3 * q * x + 4 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
sorry

end NUMINAMATH_GPT_roots_equality_l424_42494


namespace NUMINAMATH_GPT_parabola_focus_l424_42466

noncomputable def parabola_focus_coordinates (a : ℝ) : ℝ × ℝ :=
  if a ≠ 0 then (0, 1 / (4 * a)) else (0, 0)

theorem parabola_focus {x y : ℝ} (a : ℝ) (h : a = 2) (h_eq : y = a * x^2) :
  parabola_focus_coordinates a = (0, 1 / 8) :=
by sorry

end NUMINAMATH_GPT_parabola_focus_l424_42466


namespace NUMINAMATH_GPT_bruce_total_payment_l424_42431

def cost_of_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_of_mangoes (quantity rate : ℕ) : ℕ := quantity * rate

theorem bruce_total_payment : 
  cost_of_grapes 8 70 + cost_of_mangoes 11 55 = 1165 :=
by 
  sorry

end NUMINAMATH_GPT_bruce_total_payment_l424_42431


namespace NUMINAMATH_GPT_min_value_of_expression_l424_42441

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end NUMINAMATH_GPT_min_value_of_expression_l424_42441


namespace NUMINAMATH_GPT_range_of_a_l424_42458

variable (a : ℝ)

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a - 1) ^ x < (2 * a - 1) ^ y
def q (a : ℝ) : Prop := ∀ x : ℝ, 2 * a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (0 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l424_42458


namespace NUMINAMATH_GPT_integral_1_integral_2_integral_3_integral_4_integral_5_l424_42468
open Real

-- Integral 1
theorem integral_1 : ∫ (x : ℝ), sin x * cos x ^ 3 = -1 / 4 * cos x ^ 4 + C :=
by sorry

-- Integral 2
theorem integral_2 : ∫ (x : ℝ), 1 / ((1 + sqrt x) * sqrt x) = 2 * log (1 + sqrt x) + C :=
by sorry

-- Integral 3
theorem integral_3 : ∫ (x : ℝ), x ^ 2 * sqrt (x ^ 3 + 1) = 2 / 9 * (x ^ 3 + 1) ^ (3/2) + C :=
by sorry

-- Integral 4
theorem integral_4 : ∫ (x : ℝ), (exp (2 * x) - 3 * exp x) / exp x = exp x - 3 * x + C :=
by sorry

-- Integral 5
theorem integral_5 : ∫ (x : ℝ), (1 - x ^ 2) * exp x = - (x - 1) ^ 2 * exp x + C :=
by sorry

end NUMINAMATH_GPT_integral_1_integral_2_integral_3_integral_4_integral_5_l424_42468


namespace NUMINAMATH_GPT_number_of_friends_l424_42456

theorem number_of_friends (n : ℕ) (h1 : 100 % n = 0) (h2 : 100 % (n + 5) = 0) (h3 : 100 / n - 1 = 100 / (n + 5)) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l424_42456


namespace NUMINAMATH_GPT_problem_part_I_problem_part_II_l424_42476

-- Problem Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_part_I (x : ℝ) :
    (f (x + 3/2) ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 2) :=
  sorry

-- Problem Part II
theorem problem_part_II (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
    3*p + 2*q + r ≥ 9/4 :=
  sorry

end NUMINAMATH_GPT_problem_part_I_problem_part_II_l424_42476


namespace NUMINAMATH_GPT_part1_simplification_part2_inequality_l424_42464

-- Part 1: Prove the simplification of the algebraic expression
theorem part1_simplification (x : ℝ) (h₁ : x ≠ 3):
  (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2: Prove the solution set for the inequality system
theorem part2_inequality (x : ℝ) :
  (5 * x - 2 > 3 * (x + 1)) → (1/2 * x - 1 ≥ 7 - 3/2 * x) → x ≥ 4 :=
sorry

end NUMINAMATH_GPT_part1_simplification_part2_inequality_l424_42464


namespace NUMINAMATH_GPT_ratio_arithmetic_seq_a2019_a2017_eq_l424_42493

def ratio_arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n ≥ 1 → a (n+2) / a (n+1) - a (n+1) / a n = 2

theorem ratio_arithmetic_seq_a2019_a2017_eq (a : ℕ → ℝ) 
  (h : ratio_arithmetic_seq a) 
  (ha1 : a 1 = 1) 
  (ha2 : a 2 = 1) 
  (ha3 : a 3 = 3) : 
  a 2019 / a 2017 = 4 * 2017^2 - 1 :=
sorry

end NUMINAMATH_GPT_ratio_arithmetic_seq_a2019_a2017_eq_l424_42493


namespace NUMINAMATH_GPT_animals_total_sleep_in_one_week_l424_42429

-- Define the conditions
def cougar_sleep_per_night := 4 -- Cougar sleeps 4 hours per night
def zebra_extra_sleep := 2 -- Zebra sleeps 2 hours more than cougar

-- Calculate the sleep duration for the zebra
def zebra_sleep_per_night := cougar_sleep_per_night + zebra_extra_sleep

-- Total sleep duration per week
def week_nights := 7

-- Total weekly sleep durations
def cougar_weekly_sleep := cougar_sleep_per_night * week_nights
def zebra_weekly_sleep := zebra_sleep_per_night * week_nights

-- Total sleep time for both animals in one week
def total_weekly_sleep := cougar_weekly_sleep + zebra_weekly_sleep

-- The target theorem
theorem animals_total_sleep_in_one_week : total_weekly_sleep = 70 := by
  sorry

end NUMINAMATH_GPT_animals_total_sleep_in_one_week_l424_42429


namespace NUMINAMATH_GPT_divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l424_42475

noncomputable def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1)^n - x^n - 1

def P (x : ℝ) : ℝ := x^2 + x + 1

-- Prove Q(x, n) is divisible by P(x) if and only if n ≡ 1 or 5 (mod 6)
theorem divisibility_by_P (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

-- Prove Q(x, n) is divisible by P(x)^2 if and only if n ≡ 1 (mod 6)
theorem divisibility_by_P_squared (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 1 := 
sorry

-- Prove Q(x, n) is divisible by P(x)^3 if and only if n = 1
theorem divisibility_by_P_cubed (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^3 = 0 ↔ n = 1 := 
sorry

end NUMINAMATH_GPT_divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l424_42475


namespace NUMINAMATH_GPT_nellie_final_legos_l424_42423

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end NUMINAMATH_GPT_nellie_final_legos_l424_42423


namespace NUMINAMATH_GPT_smallest_positive_m_l424_42457

theorem smallest_positive_m (m : ℕ) : 
  (∃ n : ℤ, (10 * n * (n + 1) = 600) ∧ (m = 10 * (n + (n + 1)))) → (m = 170) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_positive_m_l424_42457


namespace NUMINAMATH_GPT_kanul_machinery_expense_l424_42417

theorem kanul_machinery_expense :
  let Total := 93750
  let RawMaterials := 35000
  let Cash := 0.20 * Total
  let Machinery := Total - (RawMaterials + Cash)
  Machinery = 40000 := by
sorry

end NUMINAMATH_GPT_kanul_machinery_expense_l424_42417


namespace NUMINAMATH_GPT_negation_of_p_l424_42479

-- Defining the proposition 'p'
def p : Prop := ∃ x : ℝ, x^3 > x

-- Stating the theorem
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^3 ≤ x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l424_42479


namespace NUMINAMATH_GPT_sym_axis_of_curve_eq_zero_b_plus_d_l424_42418

theorem sym_axis_of_curve_eq_zero_b_plus_d
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h_symm : ∀ x : ℝ, 2 * x = (a * ((a * x + b) / (c * x + d)) + b) / (c * ((a * x + b) / (c * x + d)) + d)) :
  b + d = 0 :=
sorry

end NUMINAMATH_GPT_sym_axis_of_curve_eq_zero_b_plus_d_l424_42418


namespace NUMINAMATH_GPT_concentration_of_spirit_in_vessel_a_l424_42487

theorem concentration_of_spirit_in_vessel_a :
  ∀ (x : ℝ), 
    (∀ (v1 v2 v3 : ℝ), v1 * (x / 100) + v2 * (30 / 100) + v3 * (10 / 100) = 15 * (26 / 100) →
      v1 + v2 + v3 = 15 →
      v1 = 4 → v2 = 5 → v3 = 6 →
      x = 45) :=
by
  intros x v1 v2 v3 h h_volume h_v1 h_v2 h_v3
  sorry

end NUMINAMATH_GPT_concentration_of_spirit_in_vessel_a_l424_42487


namespace NUMINAMATH_GPT_problem_D_l424_42444

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b c : V}

def is_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem problem_D (h₁ : is_parallel a b) (h₂ : is_parallel b c) (h₃ : b ≠ 0) : is_parallel a c :=
sorry

end NUMINAMATH_GPT_problem_D_l424_42444


namespace NUMINAMATH_GPT_expected_heads_64_coins_l424_42402

noncomputable def expected_heads (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_heads_64_coins : expected_heads 64 (15/16) = 60 := by
  sorry

end NUMINAMATH_GPT_expected_heads_64_coins_l424_42402


namespace NUMINAMATH_GPT_simplest_quadratic_radical_l424_42400
  
theorem simplest_quadratic_radical (A B C D: ℝ) 
  (hA : A = Real.sqrt 0.1) 
  (hB : B = Real.sqrt (-2)) 
  (hC : C = 3 * Real.sqrt 2) 
  (hD : D = -Real.sqrt 20) : C = 3 * Real.sqrt 2 :=
by
  have h1 : ∀ (x : ℝ), Real.sqrt x = Real.sqrt x := sorry
  sorry

end NUMINAMATH_GPT_simplest_quadratic_radical_l424_42400


namespace NUMINAMATH_GPT_carSpeedIs52mpg_l424_42484

noncomputable def carSpeed (fuelConsumptionKMPL : ℕ) -- 32 kilometers per liter
                           (gallonToLiter : ℝ)        -- 1 gallon = 3.8 liters
                           (fuelDecreaseGallons : ℝ)  -- 3.9 gallons
                           (timeHours : ℝ)            -- 5.7 hours
                           (kmToMiles : ℝ)            -- 1 mile = 1.6 kilometers
                           : ℝ :=
  let totalLiters := fuelDecreaseGallons * gallonToLiter
  let totalKilometers := totalLiters * fuelConsumptionKMPL
  let totalMiles := totalKilometers / kmToMiles
  totalMiles / timeHours

theorem carSpeedIs52mpg : carSpeed 32 3.8 3.9 5.7 1.6 = 52 := sorry

end NUMINAMATH_GPT_carSpeedIs52mpg_l424_42484


namespace NUMINAMATH_GPT_customers_left_l424_42498

theorem customers_left (x : ℕ) 
  (h1 : 47 - x + 20 = 26) : 
  x = 41 :=
sorry

end NUMINAMATH_GPT_customers_left_l424_42498


namespace NUMINAMATH_GPT_average_of_rest_of_class_l424_42427

def class_average (n : ℕ) (avg : ℕ) := n * avg
def sub_class_average (n : ℕ) (sub_avg : ℕ) := (n / 4) * sub_avg

theorem average_of_rest_of_class (n : ℕ) (h1 : class_average n 80 = 80 * n) (h2 : sub_class_average n 92 = (n / 4) * 92) :
  let A := 76
  A * (3 * n / 4) + (n / 4) * 92 = 80 * n := by
  sorry

end NUMINAMATH_GPT_average_of_rest_of_class_l424_42427


namespace NUMINAMATH_GPT_find_explicit_formula_range_of_k_l424_42448

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 - b * x + 4

theorem find_explicit_formula (h_extremum_at_2 : f a b 2 = -4 / 3 ∧ (3 * a * 4 - b = 0)) :
  ∃ a b, f a b x = (1 / 3) * x ^ 3 - 4 * x + 4 :=
sorry

theorem range_of_k (h_extremum_at_2 : f (1 / 3) 4 2 = -4 / 3) :
  ∃ k, -4 / 3 < k ∧ k < 8 / 3 :=
sorry

end NUMINAMATH_GPT_find_explicit_formula_range_of_k_l424_42448


namespace NUMINAMATH_GPT_cost_of_milkshake_l424_42491

theorem cost_of_milkshake
  (initial_money : ℝ)
  (remaining_after_cupcakes : ℝ)
  (remaining_after_sandwich : ℝ)
  (remaining_after_toy : ℝ)
  (final_remaining : ℝ)
  (money_spent_on_milkshake : ℝ) :
  initial_money = 20 →
  remaining_after_cupcakes = initial_money - (1 / 4) * initial_money →
  remaining_after_sandwich = remaining_after_cupcakes - 0.30 * remaining_after_cupcakes →
  remaining_after_toy = remaining_after_sandwich - (1 / 5) * remaining_after_sandwich →
  final_remaining = 3 →
  money_spent_on_milkshake = remaining_after_toy - final_remaining →
  money_spent_on_milkshake = 5.40 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_cost_of_milkshake_l424_42491


namespace NUMINAMATH_GPT_pet_store_cats_left_l424_42461

theorem pet_store_cats_left :
  let initial_siamese := 13.5
  let initial_house := 5.25
  let added_cats := 10.75
  let discount := 0.5
  let initial_total := initial_siamese + initial_house
  let new_total := initial_total + added_cats
  let final_total := new_total - discount
  final_total = 29 :=
by sorry

end NUMINAMATH_GPT_pet_store_cats_left_l424_42461


namespace NUMINAMATH_GPT_inequality_proof_l424_42422

variables {x y : ℝ}

theorem inequality_proof (hx_pos : x > 0) (hy_pos : y > 0) (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l424_42422


namespace NUMINAMATH_GPT_combi_sum_l424_42496

theorem combi_sum : (Nat.choose 8 2) + (Nat.choose 8 3) + (Nat.choose 9 2) = 120 :=
by
  sorry

end NUMINAMATH_GPT_combi_sum_l424_42496


namespace NUMINAMATH_GPT_teds_age_l424_42446

theorem teds_age (s t : ℕ) (h1 : t = 3 * s - 20) (h2 : t + s = 76) : t = 52 :=
by
  sorry

end NUMINAMATH_GPT_teds_age_l424_42446


namespace NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l424_42490

-- Define the constant probability of hitting the target in a single shot
def p_hit : ℚ := 2 / 3

-- Define the probability of missing the target in a single shot
def p_miss := 1 - p_hit

-- Define the probability of missing the target in all 3 shots
def p_miss_all_3 := p_miss ^ 3

-- Define the probability of hitting the target at least once in 3 shots
def p_hit_at_least_once := 1 - p_miss_all_3

-- Provide the theorem stating the solution
theorem probability_of_hitting_target_at_least_once :
  p_hit_at_least_once = 26 / 27 :=
by
  -- sorry is used to indicate the theorem needs to be proved
  sorry

end NUMINAMATH_GPT_probability_of_hitting_target_at_least_once_l424_42490


namespace NUMINAMATH_GPT_big_al_bananas_l424_42411

/-- Big Al ate 140 bananas from May 1 through May 6. Each day he ate five more bananas than on the previous day. On May 4, Big Al did not eat any bananas due to fasting. Prove that Big Al ate 38 bananas on May 6. -/
theorem big_al_bananas : 
  ∃ a : ℕ, (a + (a + 5) + (a + 10) + 0 + (a + 15) + (a + 20) = 140) ∧ ((a + 20) = 38) :=
by sorry

end NUMINAMATH_GPT_big_al_bananas_l424_42411


namespace NUMINAMATH_GPT_translation_theorem_l424_42495

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (θ : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

theorem translation_theorem
  (θ φ : ℝ)
  (hθ1 : |θ| < Real.pi / 2)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (hf : f θ 0 = 1 / 2)
  (hg : g θ φ 0 = 1 / 2) :
  φ = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_translation_theorem_l424_42495


namespace NUMINAMATH_GPT_real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l424_42485

def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ) * Complex.I

theorem real_part_0_or_3 (m : ℝ) : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) := sorry

theorem complex_part_not_0_or_3 (m : ℝ) : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) := sorry

theorem purely_imaginary_at_2 (m : ℝ) : (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) ↔ (m = 2) := sorry

theorem no_second_quadrant (m : ℝ) : ¬(m^2 - 5 * m + 6 < 0 ∧ m^2 - 3 * m > 0) := sorry

end NUMINAMATH_GPT_real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l424_42485


namespace NUMINAMATH_GPT_triangle_ineq_l424_42430

theorem triangle_ineq (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) < 5/2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_ineq_l424_42430


namespace NUMINAMATH_GPT_equal_ratios_l424_42460

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end NUMINAMATH_GPT_equal_ratios_l424_42460


namespace NUMINAMATH_GPT_find_person_age_l424_42481

theorem find_person_age : ∃ x : ℕ, 4 * (x + 4) - 4 * (x - 4) = x ∧ x = 32 := by
  sorry

end NUMINAMATH_GPT_find_person_age_l424_42481
