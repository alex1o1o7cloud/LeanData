import Mathlib

namespace NUMINAMATH_GPT_car_r_speed_l509_50991

theorem car_r_speed (v : ℝ) (h : 150 / v - 2 = 150 / (v + 10)) : v = 25 :=
sorry

end NUMINAMATH_GPT_car_r_speed_l509_50991


namespace NUMINAMATH_GPT_factorization_example_l509_50962

theorem factorization_example (x: ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorization_example_l509_50962


namespace NUMINAMATH_GPT_expected_value_abs_diff_HT_l509_50928

noncomputable def expected_abs_diff_HT : ℚ :=
  let F : ℕ → ℚ := sorry -- Recurrence relation omitted for brevity
  F 0

theorem expected_value_abs_diff_HT :
  expected_abs_diff_HT = 24 / 7 :=
sorry

end NUMINAMATH_GPT_expected_value_abs_diff_HT_l509_50928


namespace NUMINAMATH_GPT_pair_basis_of_plane_l509_50960

def vector_space := Type
variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (e1 e2 : V)
variable (h_basis : LinearIndependent ℝ ![e1, e2])
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0)

theorem pair_basis_of_plane
  (v1 v2 : V)
  (hv1 : v1 = e1 + e2)
  (hv2 : v2 = e1 - e2) :
  LinearIndependent ℝ ![v1, v2] :=
sorry

end NUMINAMATH_GPT_pair_basis_of_plane_l509_50960


namespace NUMINAMATH_GPT_smallest_number_l509_50956

theorem smallest_number (a b c d : ℤ) (h1 : a = -2) (h2 : b = 0) (h3 : c = -3) (h4 : d = 1) : 
  min (min a b) (min c d) = c :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_smallest_number_l509_50956


namespace NUMINAMATH_GPT_at_least_one_half_l509_50952

theorem at_least_one_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_half_l509_50952


namespace NUMINAMATH_GPT_complement_union_A_B_eq_neg2_0_l509_50969

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_eq_neg2_0_l509_50969


namespace NUMINAMATH_GPT_max_sum_seq_l509_50925

theorem max_sum_seq (a : ℕ → ℝ) (h1 : a 1 = 0)
  (h2 : abs (a 2) = abs (a 1 - 1)) 
  (h3 : abs (a 3) = abs (a 2 - 1)) 
  (h4 : abs (a 4) = abs (a 3 - 1)) 
  : ∃ M, (∀ (b : ℕ → ℝ), b 1 = 0 → abs (b 2) = abs (b 1 - 1) → abs (b 3) = abs (b 2 - 1) → abs (b 4) = abs (b 3 - 1) → (b 1 + b 2 + b 3 + b 4) ≤ M) 
    ∧ (a 1 + a 2 + a 3 + a 4 = M) :=
  sorry

end NUMINAMATH_GPT_max_sum_seq_l509_50925


namespace NUMINAMATH_GPT_bruce_age_multiple_of_son_l509_50977

structure Person :=
  (age : ℕ)

def bruce := Person.mk 36
def son := Person.mk 8
def multiple := 3

theorem bruce_age_multiple_of_son :
  ∃ (x : ℕ), bruce.age + x = multiple * (son.age + x) ∧ x = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_bruce_age_multiple_of_son_l509_50977


namespace NUMINAMATH_GPT_farmer_ducks_sold_l509_50906

theorem farmer_ducks_sold (D : ℕ) (earnings : ℕ) :
  (earnings = (10 * D) + (5 * 8)) →
  ((earnings / 2) * 2 = 60) →
  D = 2 := by
  sorry

end NUMINAMATH_GPT_farmer_ducks_sold_l509_50906


namespace NUMINAMATH_GPT_elle_practices_hours_l509_50978

variable (practice_time_weekday : ℕ) (days_weekday : ℕ) (multiplier_saturday : ℕ) (minutes_in_an_hour : ℕ) 
          (total_minutes_weekdays : ℕ) (total_minutes_saturday : ℕ) (total_minutes_week : ℕ) (total_hours : ℕ)

theorem elle_practices_hours :
  practice_time_weekday = 30 ∧
  days_weekday = 5 ∧
  multiplier_saturday = 3 ∧
  minutes_in_an_hour = 60 →
  total_minutes_weekdays = practice_time_weekday * days_weekday →
  total_minutes_saturday = practice_time_weekday * multiplier_saturday →
  total_minutes_week = total_minutes_weekdays + total_minutes_saturday →
  total_hours = total_minutes_week / minutes_in_an_hour →
  total_hours = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_elle_practices_hours_l509_50978


namespace NUMINAMATH_GPT_distance_to_place_l509_50973

theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) (D : ℝ) :
  rowing_speed = 5 ∧ current_speed = 1 ∧ total_time = 1 →
  D = 2.4 :=
by
  -- Rowing Parameters
  let V_d := rowing_speed + current_speed
  let V_u := rowing_speed - current_speed
  
  -- Time Variables
  let T_d := total_time / (V_d + V_u)
  let T_u := total_time - T_d

  -- Distance Calculations
  let D1 := V_d * T_d
  let D2 := V_u * T_u

  -- Prove D is the same distance both upstream and downstream
  sorry

end NUMINAMATH_GPT_distance_to_place_l509_50973


namespace NUMINAMATH_GPT_beam_reflection_problem_l509_50974

theorem beam_reflection_problem
  (A B D C : Point)
  (angle_CDA : ℝ)
  (total_path_length_max : ℝ)
  (equal_angle_reflections : ∀ (k : ℕ), angle_CDA * k ≤ 90)
  (path_length_constraint : ∀ (n : ℕ) (d : ℝ), 2 * n * d ≤ total_path_length_max)
  : angle_CDA = 5 ∧ total_path_length_max = 100 → ∃ (n : ℕ), n = 10 :=
sorry

end NUMINAMATH_GPT_beam_reflection_problem_l509_50974


namespace NUMINAMATH_GPT_solve_problem_l509_50934
noncomputable def is_solution (n : ℕ) : Prop :=
  ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) → (a + b + c ∣ a^2 + b^2 + c^2) → (a + b + c ∣ a^n + b^n + c^n)

theorem solve_problem : {n : ℕ // is_solution (3 * n - 1) ∧ is_solution (3 * n - 2)} :=
sorry

end NUMINAMATH_GPT_solve_problem_l509_50934


namespace NUMINAMATH_GPT_intersection_points_l509_50998

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 12 * x - 5
def parabola2 (x : ℝ) : ℝ := x ^ 2 - 2 * x + 3

theorem intersection_points :
  { p : ℝ × ℝ | p.snd = parabola1 p.fst ∧ p.snd = parabola2 p.fst } =
  { (1, -14), (4, -5) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l509_50998


namespace NUMINAMATH_GPT_min_expression_value_l509_50909

theorem min_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ x : ℝ, x = 5 ∧ ∀ y, (y = (b / (3 * a) + 3 / b)) → x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l509_50909


namespace NUMINAMATH_GPT_red_peppers_weight_l509_50972

theorem red_peppers_weight (total_weight green_weight : ℝ) (h1 : total_weight = 5.666666667) (h2 : green_weight = 2.8333333333333335) : 
  total_weight - green_weight = 2.8333333336666665 :=
by
  sorry

end NUMINAMATH_GPT_red_peppers_weight_l509_50972


namespace NUMINAMATH_GPT_aria_analysis_time_l509_50943

-- Definitions for the number of bones in each section
def skull_bones : ℕ := 29
def spine_bones : ℕ := 33
def thorax_bones : ℕ := 37
def upper_limb_bones : ℕ := 64
def lower_limb_bones : ℕ := 62

-- Definitions for the time spent per bone in each section (in minutes)
def time_per_skull_bone : ℕ := 15
def time_per_spine_bone : ℕ := 10
def time_per_thorax_bone : ℕ := 12
def time_per_upper_limb_bone : ℕ := 8
def time_per_lower_limb_bone : ℕ := 10

-- Definition for the total time needed in minutes
def total_time_in_minutes : ℕ :=
  (skull_bones * time_per_skull_bone) +
  (spine_bones * time_per_spine_bone) +
  (thorax_bones * time_per_thorax_bone) +
  (upper_limb_bones * time_per_upper_limb_bone) +
  (lower_limb_bones * time_per_lower_limb_bone)

-- Definition for the total time needed in hours
def total_time_in_hours : ℚ := total_time_in_minutes / 60

-- Theorem to prove the total time needed in hours is approximately 39.02
theorem aria_analysis_time : abs (total_time_in_hours - 39.02) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_aria_analysis_time_l509_50943


namespace NUMINAMATH_GPT_ribbon_length_ratio_l509_50971

theorem ribbon_length_ratio (original_length reduced_length : ℕ) (h1 : original_length = 55) (h2 : reduced_length = 35) : 
  (original_length / Nat.gcd original_length reduced_length) = 11 ∧
  (reduced_length / Nat.gcd original_length reduced_length) = 7 := 
  by
    sorry

end NUMINAMATH_GPT_ribbon_length_ratio_l509_50971


namespace NUMINAMATH_GPT_inverse_of_B_cubed_l509_50948

theorem inverse_of_B_cubed
  (B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3, -1],
    ![0, 5]
  ]) :
  (B_inv ^ 3) = ![
    ![27, -49],
    ![0, 125]
  ] := 
by
  sorry

end NUMINAMATH_GPT_inverse_of_B_cubed_l509_50948


namespace NUMINAMATH_GPT_area_ratio_of_similar_triangles_l509_50981

noncomputable def similarity_ratio := 3 / 5

theorem area_ratio_of_similar_triangles (k : ℝ) (h_sim : similarity_ratio = k) : (k^2 = 9 / 25) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_similar_triangles_l509_50981


namespace NUMINAMATH_GPT_value_of_expression_l509_50921

theorem value_of_expression :
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l509_50921


namespace NUMINAMATH_GPT_one_eighth_of_2_pow_44_eq_2_pow_x_l509_50986

theorem one_eighth_of_2_pow_44_eq_2_pow_x (x : ℕ) :
  (2^44 / 8 = 2^x) → x = 41 :=
by
  sorry

end NUMINAMATH_GPT_one_eighth_of_2_pow_44_eq_2_pow_x_l509_50986


namespace NUMINAMATH_GPT_coefficient_of_x9_in_expansion_l509_50945

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x9_in_expansion_l509_50945


namespace NUMINAMATH_GPT_scientists_from_usa_l509_50990

theorem scientists_from_usa (total_scientists : ℕ)
  (from_europe : ℕ)
  (from_canada : ℕ)
  (h1 : total_scientists = 70)
  (h2 : from_europe = total_scientists / 2)
  (h3 : from_canada = total_scientists / 5) :
  (total_scientists - from_europe - from_canada) = 21 :=
by
  sorry

end NUMINAMATH_GPT_scientists_from_usa_l509_50990


namespace NUMINAMATH_GPT_find_k_l509_50985

noncomputable def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_k
  (k : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = vec 3 1)
  (hb : b = vec 1 3)
  (hc : c = vec k (-2))
  (h_perp : dot_product (vec (a.1 - c.1) (a.2 - c.2)) (vec (a.1 - b.1) (a.2 - b.2)) = 0) :
  k = 0 :=
sorry

end NUMINAMATH_GPT_find_k_l509_50985


namespace NUMINAMATH_GPT_percentage_fertilizer_in_second_solution_l509_50926

theorem percentage_fertilizer_in_second_solution 
    (v1 v2 v3 : ℝ) 
    (p1 p2 p3 : ℝ) 
    (h1 : v1 = 20) 
    (h2 : v2 + v1 = 42) 
    (h3 : p1 = 74 / 100) 
    (h4 : p2 = 63 / 100) 
    (h5 : v3 = (63 * 42 - 74 * 20) / 22) 
    : p3 = (53 / 100) :=
by
  sorry

end NUMINAMATH_GPT_percentage_fertilizer_in_second_solution_l509_50926


namespace NUMINAMATH_GPT_find_large_number_l509_50946

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 50000) 
  (h2 : L = 13 * S + 317) : 
  L = 54140 := 
sorry

end NUMINAMATH_GPT_find_large_number_l509_50946


namespace NUMINAMATH_GPT_find_number_l509_50922

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l509_50922


namespace NUMINAMATH_GPT_trigonometric_expression_result_l509_50939

variable (α : ℝ)
variable (line_eq : ∀ x y : ℝ, 6 * x - 2 * y - 5 = 0)
variable (tan_alpha : Real.tan α = 3)

theorem trigonometric_expression_result :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_result_l509_50939


namespace NUMINAMATH_GPT_janine_total_pages_l509_50964

-- Define the conditions
def books_last_month : ℕ := 5
def books_this_month : ℕ := 2 * books_last_month
def books_per_page : ℕ := 10

-- Define the total number of pages she read in two months
def total_pages : ℕ :=
  let total_books := books_last_month + books_this_month
  total_books * books_per_page

-- State the theorem to be proven
theorem janine_total_pages : total_pages = 150 :=
by
  sorry

end NUMINAMATH_GPT_janine_total_pages_l509_50964


namespace NUMINAMATH_GPT_number_of_kids_l509_50965

theorem number_of_kids (A K : ℕ) (h1 : A + K = 13) (h2 : 7 * A = 28) : K = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_kids_l509_50965


namespace NUMINAMATH_GPT_price_per_pound_second_coffee_l509_50950

theorem price_per_pound_second_coffee
  (price_first : ℝ) (total_mix_weight : ℝ) (sell_price_per_pound : ℝ) (each_kind_weight : ℝ) 
  (total_sell_price : ℝ) (total_first_cost : ℝ) (total_second_cost : ℝ) (price_second : ℝ) :
  price_first = 2.15 →
  total_mix_weight = 18 →
  sell_price_per_pound = 2.30 →
  each_kind_weight = 9 →
  total_sell_price = total_mix_weight * sell_price_per_pound →
  total_first_cost = each_kind_weight * price_first →
  total_second_cost = total_sell_price - total_first_cost →
  price_second = total_second_cost / each_kind_weight →
  price_second = 2.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_price_per_pound_second_coffee_l509_50950


namespace NUMINAMATH_GPT_driver_travel_distance_per_week_l509_50957

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end NUMINAMATH_GPT_driver_travel_distance_per_week_l509_50957


namespace NUMINAMATH_GPT_lowest_test_score_dropped_l509_50941

theorem lowest_test_score_dropped (S L : ℕ)
  (h1 : S = 5 * 42) 
  (h2 : S - L = 4 * 48) : 
  L = 18 :=
by
  sorry

end NUMINAMATH_GPT_lowest_test_score_dropped_l509_50941


namespace NUMINAMATH_GPT_original_price_of_coffee_l509_50983

variable (P : ℝ)

theorem original_price_of_coffee :
  (4 * P - 2 * (1.5 * P) = 2) → P = 2 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_coffee_l509_50983


namespace NUMINAMATH_GPT_probability_of_neither_red_nor_purple_l509_50999

theorem probability_of_neither_red_nor_purple :
  let total_balls := 100
  let white_balls := 20
  let green_balls := 30
  let yellow_balls := 10
  let red_balls := 37
  let purple_balls := 3
  let neither_red_nor_purple_balls := white_balls + green_balls + yellow_balls
  (neither_red_nor_purple_balls : ℝ) / (total_balls : ℝ) = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_neither_red_nor_purple_l509_50999


namespace NUMINAMATH_GPT_speed_in_still_water_l509_50905

theorem speed_in_still_water (u d s : ℝ) (hu : u = 20) (hd : d = 60) (hs : s = (u + d) / 2) : s = 40 := 
by 
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l509_50905


namespace NUMINAMATH_GPT_value_of_x_add_y_not_integer_l509_50982

theorem value_of_x_add_y_not_integer (x y: ℝ) (h1: y = 3 * ⌊x⌋ + 4) (h2: y = 2 * ⌊x - 3⌋ + 7) (h3: ¬ ∃ n: ℤ, x = n): -8 < x + y ∧ x + y < -7 := 
sorry

end NUMINAMATH_GPT_value_of_x_add_y_not_integer_l509_50982


namespace NUMINAMATH_GPT_units_digit_of_n_l509_50932

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : m % 10 = 4) : n % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_n_l509_50932


namespace NUMINAMATH_GPT_peregrines_eat_30_percent_l509_50970

theorem peregrines_eat_30_percent (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (pigeons_left : ℕ) :
  initial_pigeons = 40 →
  chicks_per_pigeon = 6 →
  pigeons_left = 196 →
  (100 * (initial_pigeons * chicks_per_pigeon + initial_pigeons - pigeons_left)) / 
  (initial_pigeons * chicks_per_pigeon + initial_pigeons) = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_peregrines_eat_30_percent_l509_50970


namespace NUMINAMATH_GPT_johns_bakery_fraction_l509_50901

theorem johns_bakery_fraction :
  ∀ (M : ℝ), 
  (M / 4 + M / 3 + 6 + (24 - (M / 4 + M / 3 + 6)) = 24) →
  (24 : ℝ) = M →
  (4 + 8 + 6 = 18) →
  (24 - 18 = 6) →
  (6 / 24 = (1 / 6 : ℝ)) :=
by
  intros M h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_johns_bakery_fraction_l509_50901


namespace NUMINAMATH_GPT_solution_set_l509_50916

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

theorem solution_set :
  {x : ℝ | x + (x + 2) * f (x + 2) ≤ 5} = {x : ℝ | x ≤ 3 / 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_l509_50916


namespace NUMINAMATH_GPT_max_value_l509_50936

-- Definitions for the given conditions
def point_A := (3, 1)
def line_equation (m n : ℝ) := 3 * m + n + 1 = 0
def positive_product (m n : ℝ) := m * n > 0

-- The main statement to be proved
theorem max_value (m n : ℝ) (h1 : line_equation m n) (h2 : positive_product m n) : 
  (3 / m + 1 / n) ≤ -16 :=
sorry

end NUMINAMATH_GPT_max_value_l509_50936


namespace NUMINAMATH_GPT_find_angle_A_find_cos2C_minus_pi_over_6_l509_50910

noncomputable def triangle_area_formula (a b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * a * b * Real.sin C

noncomputable def given_area_formula (b c : ℝ) (S : ℝ) (a : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 6) * b * (b + c - a * Real.cos C)

noncomputable def angle_A (S b c a C : ℝ) (h : given_area_formula b c S a C) : ℝ :=
  Real.arcsin ((Real.sqrt 3 / 3) * (b + c - a * Real.cos C))

theorem find_angle_A (a b c S C : ℝ) (h : given_area_formula b c S a C) :
  angle_A S b c a C h = π / 3 :=
sorry

-- Part 2 related definitions
noncomputable def cos2C_minus_pi_over_6 (b c a C : ℝ) : ℝ :=
  let cos_C := (b^2 + c^2 - a^2) / (2 * b * c)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let cos_2C := 2 * cos_C^2 - 1
  let sin_2C := 2 * sin_C * cos_C
  cos_2C * (Real.sqrt 3 / 2) + sin_2C * (1 / 2)

theorem find_cos2C_minus_pi_over_6 (b c a C : ℝ) (hb : b = 1) (hc : c = 3) (ha : a = Real.sqrt 7) :
  cos2C_minus_pi_over_6 b c a C = - (4 * Real.sqrt 3 / 7) :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_cos2C_minus_pi_over_6_l509_50910


namespace NUMINAMATH_GPT_customer_can_receive_exact_change_l509_50951

theorem customer_can_receive_exact_change (k : ℕ) (hk : k ≤ 1000) :
  ∃ change : ℕ, change + k = 1000 ∧ change ≤ 1999 :=
by
  sorry

end NUMINAMATH_GPT_customer_can_receive_exact_change_l509_50951


namespace NUMINAMATH_GPT_compressor_stations_distances_l509_50966

theorem compressor_stations_distances 
    (x y z a : ℝ) 
    (h1 : x + y = 2 * z)
    (h2 : z + y = x + a)
    (h3 : x + z = 75)
    (h4 : 0 ≤ x)
    (h5 : 0 ≤ y)
    (h6 : 0 ≤ z)
    (h7 : 0 < a)
    (h8 : a < 100) :
  (a = 15 → x = 42 ∧ y = 24 ∧ z = 33) :=
by 
  intro ha_eq_15
  sorry

end NUMINAMATH_GPT_compressor_stations_distances_l509_50966


namespace NUMINAMATH_GPT_number_of_candidates_l509_50904

theorem number_of_candidates
  (n : ℕ)
  (h : n * (n - 1) = 132) : 
  n = 12 :=
sorry

end NUMINAMATH_GPT_number_of_candidates_l509_50904


namespace NUMINAMATH_GPT_find_ab_and_m_l509_50938

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_ab_and_m (a b m : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-1, -2))
  (h2 : ∀ (x : ℝ), (3 * a * x^2 + 2 * b * x) = -1/3 ↔ x = -1)
  (h3 : ∀ (x : ℝ), f a b x = a * x ^ 3 + b * x ^ 2)
  : (a = -13/3 ∧ b = -19/3) ∧ (0 < m ∧ m < 38/39) :=
sorry

end NUMINAMATH_GPT_find_ab_and_m_l509_50938


namespace NUMINAMATH_GPT_group_C_questions_l509_50953

theorem group_C_questions (a b c : ℕ) (total_questions : ℕ) (h1 : a + b + c = 100)
  (h2 : b = 23)
  (h3 : a ≥ (6 * (a + 2 * b + 3 * c)) / 10)
  (h4 : 2 * b ≤ (25 * (a + 2 * b + 3 * c)) / 100)
  (h5 : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c) :
  c = 1 :=
sorry

end NUMINAMATH_GPT_group_C_questions_l509_50953


namespace NUMINAMATH_GPT_quadratic_no_real_roots_range_l509_50912

theorem quadratic_no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 + 2 * x - k = 0)) ↔ k < -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_range_l509_50912


namespace NUMINAMATH_GPT_complex_imag_part_of_z_l509_50924

theorem complex_imag_part_of_z (z : ℂ) (h : z * (2 + ⅈ) = 3 - 6 * ⅈ) : z.im = -3 := by
  sorry

end NUMINAMATH_GPT_complex_imag_part_of_z_l509_50924


namespace NUMINAMATH_GPT_find_b_l509_50908

def h (x : ℝ) : ℝ := 5 * x + 6

theorem find_b : ∃ b : ℝ, h b = 0 ∧ b = -6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l509_50908


namespace NUMINAMATH_GPT_shanmukham_total_payment_l509_50911

noncomputable def total_price_shanmukham_pays : Real :=
  let itemA_price : Real := 6650
  let itemA_rebate : Real := 6 -- percentage
  let itemA_tax : Real := 10 -- percentage

  let itemB_price : Real := 8350
  let itemB_rebate : Real := 4 -- percentage
  let itemB_tax : Real := 12 -- percentage

  let itemC_price : Real := 9450
  let itemC_rebate : Real := 8 -- percentage
  let itemC_tax : Real := 15 -- percentage

  let final_price (price : Real) (rebate : Real) (tax : Real) : Real :=
    let rebate_amt := (rebate / 100) * price
    let price_after_rebate := price - rebate_amt
    let tax_amt := (tax / 100) * price_after_rebate
    price_after_rebate + tax_amt

  final_price itemA_price itemA_rebate itemA_tax +
  final_price itemB_price itemB_rebate itemB_tax +
  final_price itemC_price itemC_rebate itemC_tax

theorem shanmukham_total_payment :
  total_price_shanmukham_pays = 25852.12 := by
  sorry

end NUMINAMATH_GPT_shanmukham_total_payment_l509_50911


namespace NUMINAMATH_GPT_john_taking_pictures_years_l509_50979

-- Definitions based on the conditions
def pictures_per_day : ℕ := 10
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140
def days_per_year : ℕ := 365

-- Theorem statement
theorem john_taking_pictures_years : total_spent / cost_per_card * images_per_card / pictures_per_day / days_per_year = 3 :=
by
  sorry

end NUMINAMATH_GPT_john_taking_pictures_years_l509_50979


namespace NUMINAMATH_GPT_problem_statement_l509_50994

theorem problem_statement (a b c : ℝ) (h1 : a - b = 2) (h2 : b - c = -3) : a - c = -1 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l509_50994


namespace NUMINAMATH_GPT_hyperbola_dot_product_zero_l509_50903

theorem hyperbola_dot_product_zero
  (a b x y : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_hyperbola : (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_ecc : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 2) :
  let B := (-x, y)
  let C := (x, y)
  let A := (a, 0)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_dot_product_zero_l509_50903


namespace NUMINAMATH_GPT_train_length_l509_50975

noncomputable def convert_speed (v_kmh : ℝ) : ℝ :=
  v_kmh * (5 / 18)

def length_of_train (speed_mps : ℝ) (time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

theorem train_length (v_kmh : ℝ) (t_sec : ℝ) (length_m : ℝ) :
  v_kmh = 60 →
  t_sec = 45 →
  length_m = 750 →
  length_of_train (convert_speed v_kmh) t_sec = length_m :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_train_length_l509_50975


namespace NUMINAMATH_GPT_total_fish_count_l509_50914

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ)
  (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := 
  by 
    sorry

end NUMINAMATH_GPT_total_fish_count_l509_50914


namespace NUMINAMATH_GPT_distinct_positive_integers_exists_l509_50963

theorem distinct_positive_integers_exists 
(n : ℕ)
(a b : ℕ)
(h1 : a ≠ b)
(h2 : b % a = 0)
(h3 : a > 10^(2 * n - 1) ∧ a < 10^(2 * n))
(h4 : b > 10^(2 * n - 1) ∧ b < 10^(2 * n))
(h5 : ∀ x y : ℕ, a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < y ∧ x / 10^(n - 1) ≠ 0 ∧ y / 10^(n - 1) ≠ 0) :
a = (10^(2 * n) - 1) / 7 ∧ b = 6 * (10^(2 * n) - 1) / 7 := 
by
  sorry

end NUMINAMATH_GPT_distinct_positive_integers_exists_l509_50963


namespace NUMINAMATH_GPT_smallest_number_is_minus_three_l509_50935

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end NUMINAMATH_GPT_smallest_number_is_minus_three_l509_50935


namespace NUMINAMATH_GPT_total_distance_walked_l509_50927

-- Define the conditions
def home_to_school : ℕ := 750
def half_distance : ℕ := home_to_school / 2
def return_home : ℕ := half_distance
def home_to_school_again : ℕ := home_to_school

-- Define the theorem statement
theorem total_distance_walked : 
  half_distance + return_home + home_to_school_again = 1500 := by
  sorry

end NUMINAMATH_GPT_total_distance_walked_l509_50927


namespace NUMINAMATH_GPT_parabola_axis_l509_50997

section
variable (x y : ℝ)

-- Condition: Defines the given parabola equation.
def parabola_eq (x y : ℝ) : Prop := x = (1 / 4) * y^2

-- The Proof Problem: Prove that the axis of this parabola is x = -1/2.
theorem parabola_axis (h : parabola_eq x y) : x = - (1 / 2) := 
sorry
end

end NUMINAMATH_GPT_parabola_axis_l509_50997


namespace NUMINAMATH_GPT_gain_percentage_second_book_l509_50949

theorem gain_percentage_second_book (C1 C2 SP1 SP2 : ℝ) (H1 : C1 + C2 = 360) (H2 : C1 = 210) (H3 : SP1 = C1 - (15 / 100) * C1) (H4 : SP1 = SP2) (H5 : SP2 = C2 + (19 / 100) * C2) : 
  (19 : ℝ) = 19 := 
by
  sorry

end NUMINAMATH_GPT_gain_percentage_second_book_l509_50949


namespace NUMINAMATH_GPT_grogg_expected_value_l509_50995

theorem grogg_expected_value (n : ℕ) (p : ℝ) (h_n : 2 ≤ n) (h_p : 0 < p ∧ p < 1) :
  (p + n * p^n * (1 - p) = 1) ↔ (p = 1 / n^(1/n:ℝ)) :=
sorry

end NUMINAMATH_GPT_grogg_expected_value_l509_50995


namespace NUMINAMATH_GPT_simplify_expression_l509_50918

variable (a b : ℝ)

theorem simplify_expression : 
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l509_50918


namespace NUMINAMATH_GPT_blue_eyed_kitten_percentage_is_correct_l509_50968

def total_blue_eyed_kittens : ℕ := 5 + 6 + 4 + 7 + 3

def total_kittens : ℕ := 12 + 16 + 11 + 19 + 12

def percentage_blue_eyed_kittens (blue : ℕ) (total : ℕ) : ℚ := (blue : ℚ) / (total : ℚ) * 100

theorem blue_eyed_kitten_percentage_is_correct :
  percentage_blue_eyed_kittens total_blue_eyed_kittens total_kittens = 35.71 := sorry

end NUMINAMATH_GPT_blue_eyed_kitten_percentage_is_correct_l509_50968


namespace NUMINAMATH_GPT_least_sum_of_exponents_520_l509_50947

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_least_sum_of_exponents_520_l509_50947


namespace NUMINAMATH_GPT_sum_of_two_squares_l509_50993

theorem sum_of_two_squares (n : ℕ) (k m : ℤ) : 2 * n = k^2 + m^2 → ∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_two_squares_l509_50993


namespace NUMINAMATH_GPT_prob1_prob2_max_area_prob3_circle_diameter_l509_50954

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end NUMINAMATH_GPT_prob1_prob2_max_area_prob3_circle_diameter_l509_50954


namespace NUMINAMATH_GPT_unique_injective_f_solution_l509_50919

noncomputable def unique_injective_function (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))

theorem unique_injective_f_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))
  → (∀ x y : ℝ, f x = f y → x = y) -- injectivity condition
  → ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_unique_injective_f_solution_l509_50919


namespace NUMINAMATH_GPT_fixed_point_of_log_function_l509_50989

theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x y : ℝ, y = 2 + Real.logb a (x + 2) → y = 2 → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_log_function_l509_50989


namespace NUMINAMATH_GPT_train_seat_count_l509_50959

theorem train_seat_count (t : ℝ) (h1 : 0.20 * t = 0.2 * t)
  (h2 : 0.60 * t = 0.6 * t) (h3 : 30 + 0.20 * t + 0.60 * t = t) : t = 150 :=
by
  sorry

end NUMINAMATH_GPT_train_seat_count_l509_50959


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l509_50902

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (-16 ≤ a ∧ a ≤ 0) ↔ ∀ x : ℝ, ¬(x^2 + a * x - 4 * a < 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l509_50902


namespace NUMINAMATH_GPT_largest_n_for_crates_l509_50955

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end NUMINAMATH_GPT_largest_n_for_crates_l509_50955


namespace NUMINAMATH_GPT_slope_of_line_is_neg_one_l509_50907

theorem slope_of_line_is_neg_one (y : ℝ) (h : (y - 5) / (5 - (-3)) = -1) : y = -3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_is_neg_one_l509_50907


namespace NUMINAMATH_GPT_smallest_square_area_l509_50958

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) :
  ∃ s, s^2 = 81 ∧ (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ (a + c) ≤ s ∧ (b + d) ≤ s) :=
sorry

end NUMINAMATH_GPT_smallest_square_area_l509_50958


namespace NUMINAMATH_GPT_max_fans_theorem_l509_50992

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end NUMINAMATH_GPT_max_fans_theorem_l509_50992


namespace NUMINAMATH_GPT_sqrt_six_plus_s_cubed_l509_50900

theorem sqrt_six_plus_s_cubed (s : ℝ) : 
    Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) :=
sorry

end NUMINAMATH_GPT_sqrt_six_plus_s_cubed_l509_50900


namespace NUMINAMATH_GPT_line_circle_intersection_l509_50988

-- Define the line and circle in Lean
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 2

-- Define the proof about the intersection
theorem line_circle_intersection :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq x1 y1 ∧ circle_eq x1 y1) → (line_eq x2 y2 ∧ circle_eq x2 y2) → (x1 = x2 ∧ y1 = y2) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_circle_intersection_l509_50988


namespace NUMINAMATH_GPT_sum_of_acute_angles_l509_50967

theorem sum_of_acute_angles (α β : ℝ) (t : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 2 / t) (h_tanβ : Real.tan β = t / 15)
  (h_min : 10 * Real.tan α + 3 * Real.tan β = 4) :
  α + β = π / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_acute_angles_l509_50967


namespace NUMINAMATH_GPT_notebook_cost_l509_50920

theorem notebook_cost
  (students : ℕ)
  (majority_students : ℕ)
  (cost : ℕ)
  (notebooks : ℕ)
  (h1 : students = 36)
  (h2 : majority_students > 18)
  (h3 : notebooks > 1)
  (h4 : cost > notebooks)
  (h5 : majority_students * cost * notebooks = 2079) :
  cost = 11 :=
by
  sorry

end NUMINAMATH_GPT_notebook_cost_l509_50920


namespace NUMINAMATH_GPT_find_x_l509_50944

theorem find_x (x : ℝ) (h : 6 * x + 7 * x + 3 * x + 2 * x + 4 * x = 360) : 
  x = 180 / 11 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l509_50944


namespace NUMINAMATH_GPT_arithmetic_sequence_n_is_17_l509_50940

theorem arithmetic_sequence_n_is_17
  (a : ℕ → ℤ)  -- An arithmetic sequence a_n
  (h1 : a 1 = 5)  -- First term is 5
  (h5 : a 5 = -3)  -- Fifth term is -3
  (hn : a n = -27) : n = 17 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_is_17_l509_50940


namespace NUMINAMATH_GPT_distance_between_points_l509_50961

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 2 5 5 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l509_50961


namespace NUMINAMATH_GPT_gcd_2197_2209_l509_50984

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_2197_2209_l509_50984


namespace NUMINAMATH_GPT_one_fifty_percent_of_eighty_l509_50987

theorem one_fifty_percent_of_eighty : (150 / 100) * 80 = 120 :=
  by sorry

end NUMINAMATH_GPT_one_fifty_percent_of_eighty_l509_50987


namespace NUMINAMATH_GPT_pool_filling_time_l509_50929

theorem pool_filling_time (rate_jim rate_sue rate_tony : ℝ) (h1 : rate_jim = 1 / 30) (h2 : rate_sue = 1 / 45) (h3 : rate_tony = 1 / 90) : 
     1 / (rate_jim + rate_sue + rate_tony) = 15 := by
  sorry

end NUMINAMATH_GPT_pool_filling_time_l509_50929


namespace NUMINAMATH_GPT_total_mile_times_l509_50923

theorem total_mile_times (t_Tina t_Tony t_Tom t_Total : ℕ) 
  (h1 : t_Tina = 6) 
  (h2 : t_Tony = t_Tina / 2) 
  (h3 : t_Tom = t_Tina / 3) 
  (h4 : t_Total = t_Tina + t_Tony + t_Tom) : t_Total = 11 := 
sorry

end NUMINAMATH_GPT_total_mile_times_l509_50923


namespace NUMINAMATH_GPT_log_w_u_value_l509_50915

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_w_u_value (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu1 : u ≠ 1) (hv1 : v ≠ 1) (hw1 : w ≠ 1)
    (h1 : log u (v * w) + log v w = 5) (h2 : log v u + log w v = 3) : 
    log w u = 4 / 5 := 
sorry

end NUMINAMATH_GPT_log_w_u_value_l509_50915


namespace NUMINAMATH_GPT_ways_to_divide_day_l509_50913

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end NUMINAMATH_GPT_ways_to_divide_day_l509_50913


namespace NUMINAMATH_GPT_algebra_expression_value_l509_50931

theorem algebra_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 11) : 3 * x^2 + 9 * x + 12 = 30 := 
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l509_50931


namespace NUMINAMATH_GPT_product_of_roots_l509_50976

theorem product_of_roots : 
  ∀ (r1 r2 r3 : ℝ), (2 * r1 * r2 * r3 - 3 * (r1 * r2 + r2 * r3 + r3 * r1) - 15 * (r1 + r2 + r3) + 35 = 0) → 
  (r1 * r2 * r3 = -35 / 2) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l509_50976


namespace NUMINAMATH_GPT_max_n_value_l509_50917

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h_ineq : 1/(a - b) + 1/(b - c) ≥ n/(a - c)) : n ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_n_value_l509_50917


namespace NUMINAMATH_GPT_parallel_lines_slope_l509_50937

theorem parallel_lines_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + 2 * y - 1 = 0 → x = -2 * y + 1)
  (h2 : ∀ x y : ℝ, m * x - y = 0 → y = m * x) : 
  m = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l509_50937


namespace NUMINAMATH_GPT_speed_ratio_l509_50942

variable (d_A d_B : ℝ) (t_A t_B : ℝ)

-- Define the conditions
def condition1 : Prop := d_A = (1 + 1/5) * d_B
def condition2 : Prop := t_B = (1 - 1/11) * t_A

-- State the theorem that the speed ratio is 12:11
theorem speed_ratio (h1 : condition1 d_A d_B) (h2 : condition2 t_A t_B) :
  (d_A / t_A) / (d_B / t_B) = 12 / 11 :=
sorry

end NUMINAMATH_GPT_speed_ratio_l509_50942


namespace NUMINAMATH_GPT_fraction_div_subtract_l509_50930

theorem fraction_div_subtract : 
  (5 / 6 : ℚ) / (9 / 10) - (1 / 15) = 116 / 135 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_div_subtract_l509_50930


namespace NUMINAMATH_GPT_total_votes_election_l509_50933

theorem total_votes_election (total_votes fiona_votes elena_votes devin_votes : ℝ) 
  (Fiona_fraction : fiona_votes = (4/15) * total_votes)
  (Elena_fiona : elena_votes = fiona_votes + 15)
  (Devin_elena : devin_votes = 2 * elena_votes)
  (total_eq : total_votes = fiona_votes + elena_votes + devin_votes) :
  total_votes = 675 := 
sorry

end NUMINAMATH_GPT_total_votes_election_l509_50933


namespace NUMINAMATH_GPT_sin_cos_identity_l509_50980

theorem sin_cos_identity (α : ℝ) (h1 : Real.sin (α - Real.pi / 6) = 1 / 3) :
    Real.sin (2 * α - Real.pi / 6) + Real.cos (2 * α) = 7 / 9 :=
sorry

end NUMINAMATH_GPT_sin_cos_identity_l509_50980


namespace NUMINAMATH_GPT_num_unique_m_values_l509_50996

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end NUMINAMATH_GPT_num_unique_m_values_l509_50996
