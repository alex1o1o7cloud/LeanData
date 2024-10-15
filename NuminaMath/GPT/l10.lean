import Mathlib

namespace NUMINAMATH_GPT_consecutive_numbers_sum_l10_1044

theorem consecutive_numbers_sum (n : ℤ) (h1 : (n - 1) * n * (n + 1) = 210) (h2 : ∀ m, (m - 1) * m * (m + 1) = 210 → (m - 1)^2 + m^2 + (m + 1)^2 ≥ (n - 1)^2 + n^2 + (n + 1)^2) :
  (n - 1) + n = 11 :=
by 
  sorry

end NUMINAMATH_GPT_consecutive_numbers_sum_l10_1044


namespace NUMINAMATH_GPT_cassidy_total_grounding_days_l10_1047

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end NUMINAMATH_GPT_cassidy_total_grounding_days_l10_1047


namespace NUMINAMATH_GPT_red_marbles_initial_count_l10_1030

theorem red_marbles_initial_count (r g : ℕ) 
  (h1 : 3 * r = 5 * g)
  (h2 : 4 * (r - 18) = g + 27) :
  r = 29 :=
sorry

end NUMINAMATH_GPT_red_marbles_initial_count_l10_1030


namespace NUMINAMATH_GPT_find_r_l10_1024

theorem find_r (b r : ℝ) (h1 : b / (1 - r) = 18) (h2 : b * r^2 / (1 - r^2) = 6) : r = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l10_1024


namespace NUMINAMATH_GPT_range_of_a_l10_1005

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * a * x + a + 2 ≤ 0 → 1 ≤ x ∧ x ≤ 4) ↔ a ∈ Set.Ioo (-1 : ℝ) (18 / 7) ∨ a = 18 / 7 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l10_1005


namespace NUMINAMATH_GPT_sum_of_interior_edges_l10_1073

noncomputable def interior_edge_sum (outer_length : ℝ) (wood_width : ℝ) (frame_area : ℝ) : ℝ := 
  let outer_width := (frame_area + 3 * (outer_length - 2 * wood_width) * 4) / outer_length
  let inner_length := outer_length - 2 * wood_width
  let inner_width := outer_width - 2 * wood_width
  2 * inner_length + 2 * inner_width

theorem sum_of_interior_edges :
  interior_edge_sum 7 2 34 = 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_edges_l10_1073


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l10_1091

-- Define the percentages for various sectors
def microphotonics := 14
def home_electronics := 24
def food_additives := 15
def genetically_modified_microorganisms := 19
def industrial_lubricants := 8

-- The sum of the given percentages
def total_other_percentages := 
    microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

-- The remaining percentage for basic astrophysics
def basic_astrophysics_percentage := 100 - total_other_percentages

-- Number of degrees in a full circle
def full_circle_degrees := 360

-- Calculate the degrees representing basic astrophysics
def degrees_for_basic_astrophysics := (basic_astrophysics_percentage * full_circle_degrees) / 100

-- Theorem statement
theorem basic_astrophysics_degrees : degrees_for_basic_astrophysics = 72 := 
by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l10_1091


namespace NUMINAMATH_GPT_parity_equivalence_l10_1090

theorem parity_equivalence (p q : ℕ) :
  (Even (p^3 - q^3)) ↔ (Even (p + q)) :=
by
  sorry

end NUMINAMATH_GPT_parity_equivalence_l10_1090


namespace NUMINAMATH_GPT_functions_increase_faster_l10_1039

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- Restate the problem in Lean
theorem functions_increase_faster :
  (∀ (x : ℝ), deriv y₁ x = 100) ∧
  (∀ (x : ℝ), deriv y₂ x = 100) ∧
  (∀ (x : ℝ), deriv y₃ x = 99) ∧
  (100 > 99) :=
by
  sorry

end NUMINAMATH_GPT_functions_increase_faster_l10_1039


namespace NUMINAMATH_GPT_find_k_l10_1086

theorem find_k 
  (h : ∀ x, 2 * x ^ 2 + 14 * x + k = 0 → x = ((-14 + Real.sqrt 10) / 4) ∨ x = ((-14 - Real.sqrt 10) / 4)) :
  k = 93 / 4 :=
sorry

end NUMINAMATH_GPT_find_k_l10_1086


namespace NUMINAMATH_GPT_integral_cos_plus_one_l10_1072

theorem integral_cos_plus_one :
  ∫ x in - (Real.pi / 2).. (Real.pi / 2), (1 + Real.cos x) = Real.pi + 2 :=
by
  sorry

end NUMINAMATH_GPT_integral_cos_plus_one_l10_1072


namespace NUMINAMATH_GPT_find_two_digit_number_l10_1083

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end NUMINAMATH_GPT_find_two_digit_number_l10_1083


namespace NUMINAMATH_GPT_one_over_x_plus_one_over_y_eq_two_l10_1060

theorem one_over_x_plus_one_over_y_eq_two 
  (x y : ℝ)
  (h1 : 3^x = Real.sqrt 12)
  (h2 : 4^y = Real.sqrt 12) : 
  1 / x + 1 / y = 2 := 
by 
  sorry

end NUMINAMATH_GPT_one_over_x_plus_one_over_y_eq_two_l10_1060


namespace NUMINAMATH_GPT_peony_total_count_l10_1043

theorem peony_total_count (n : ℕ) (x : ℕ) (total_sample : ℕ) (single_sample : ℕ) (double_sample : ℕ) (thousand_sample : ℕ) (extra_thousand : ℕ)
    (h1 : thousand_sample > single_sample)
    (h2 : thousand_sample - single_sample = extra_thousand)
    (h3 : total_sample = single_sample + double_sample + thousand_sample)
    (h4 : total_sample = 12)
    (h5 : single_sample = 4)
    (h6 : double_sample = 2)
    (h7 : thousand_sample = 6)
    (h8 : extra_thousand = 30) :
    n = 180 :=
by 
  sorry

end NUMINAMATH_GPT_peony_total_count_l10_1043


namespace NUMINAMATH_GPT_squared_remainder_l10_1025

theorem squared_remainder (N : ℤ) (k : ℤ) :
  (N % 9 = 2 ∨ N % 9 = 7) → 
  (N^2 % 9 = 4) :=
by
  sorry

end NUMINAMATH_GPT_squared_remainder_l10_1025


namespace NUMINAMATH_GPT_find_b_l10_1071

-- Define the quadratic equation
def quadratic_eq (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x - 15

-- Prove that b = 49/8 given -8 is a solution to the quadratic equation
theorem find_b (b : ℝ) : quadratic_eq b (-8) = 0 -> b = 49 / 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_l10_1071


namespace NUMINAMATH_GPT_Bianca_pictures_distribution_l10_1056

theorem Bianca_pictures_distribution 
(pictures_total : ℕ) 
(pictures_in_one_album : ℕ) 
(albums_remaining : ℕ) 
(h1 : pictures_total = 33)
(h2 : pictures_in_one_album = 27)
(h3 : albums_remaining = 3)
: (pictures_total - pictures_in_one_album) / albums_remaining = 2 := 
by 
  sorry

end NUMINAMATH_GPT_Bianca_pictures_distribution_l10_1056


namespace NUMINAMATH_GPT_probability_first_die_l10_1021

theorem probability_first_die (n : ℕ) (n_pos : n = 4025) (m : ℕ) (m_pos : m = 2012) : 
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  (favorable_outcomes / total_outcomes : ℚ) = 1006 / 4025 :=
by
  have h_n : n = 4025 := n_pos
  have h_m : m = 2012 := m_pos
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  sorry

end NUMINAMATH_GPT_probability_first_die_l10_1021


namespace NUMINAMATH_GPT_store_owner_marked_price_l10_1069

theorem store_owner_marked_price (L M : ℝ) (h1 : M = (56 / 45) * L) : M / L = 124.44 / 100 :=
by
  sorry

end NUMINAMATH_GPT_store_owner_marked_price_l10_1069


namespace NUMINAMATH_GPT_number_multiply_increase_l10_1080

theorem number_multiply_increase (x : ℕ) (h : 25 * x = 25 + 375) : x = 16 := by
  sorry

end NUMINAMATH_GPT_number_multiply_increase_l10_1080


namespace NUMINAMATH_GPT_algebraic_expression_zero_l10_1064

theorem algebraic_expression_zero (a b : ℝ) (h : a^2 + 2 * a * b + b^2 = 0) : 
  a * (a + 4 * b) - (a + 2 * b) * (a - 2 * b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_zero_l10_1064


namespace NUMINAMATH_GPT_function_equivalence_l10_1074

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (-y) = -g y) ∧ (∀ x : ℝ, f x = g (1 - 2 * x^2) + 1010) :=
sorry

end NUMINAMATH_GPT_function_equivalence_l10_1074


namespace NUMINAMATH_GPT_cubic_identity_l10_1076

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end NUMINAMATH_GPT_cubic_identity_l10_1076


namespace NUMINAMATH_GPT_speed_ratio_of_runners_l10_1087

theorem speed_ratio_of_runners (v_A v_B : ℝ) (c : ℝ)
  (h1 : 0 < v_A ∧ 0 < v_B) -- They run at constant, but different speeds
  (h2 : (v_B / v_A) = (2 / 3)) -- Distance relationship from meeting points
  : v_B / v_A = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_of_runners_l10_1087


namespace NUMINAMATH_GPT_sum_lent_l10_1082

theorem sum_lent (P : ℝ) (r t : ℝ) (I : ℝ) (h1 : r = 6) (h2 : t = 6) (h3 : I = P - 672) (h4 : I = P * r * t / 100) :
  P = 1050 := by
  sorry

end NUMINAMATH_GPT_sum_lent_l10_1082


namespace NUMINAMATH_GPT_hypotenuse_is_2_sqrt_25_point_2_l10_1057

open Real

noncomputable def hypotenuse_length_of_right_triangle (ma mb : ℝ) (a b c : ℝ) : ℝ :=
  if h1 : ma = 6 ∧ mb = sqrt 27 then
    c
  else
    0

theorem hypotenuse_is_2_sqrt_25_point_2 :
  hypotenuse_length_of_right_triangle 6 (sqrt 27) a b (2 * sqrt 25.2) = 2 * sqrt 25.2 :=
by
  sorry -- proof to be filled

end NUMINAMATH_GPT_hypotenuse_is_2_sqrt_25_point_2_l10_1057


namespace NUMINAMATH_GPT_complement_of_M_in_U_is_1_4_l10_1061

-- Define U
def U : Set ℕ := {x | x < 5 ∧ x ≠ 0}

-- Define M
def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

-- The complement of M in U
def complement_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem complement_of_M_in_U_is_1_4 : complement_U_M = {1, 4} := 
by sorry

end NUMINAMATH_GPT_complement_of_M_in_U_is_1_4_l10_1061


namespace NUMINAMATH_GPT_second_quadrant_distance_l10_1063

theorem second_quadrant_distance 
    (m : ℝ) 
    (P : ℝ × ℝ)
    (hP1 : P = (m - 3, m + 2))
    (hP2 : (m + 2) > 0)
    (hP3 : (m - 3) < 0)
    (hDist : |(m + 2)| = 4) : P = (-1, 4) := 
by
  have h1 : m + 2 = 4 := sorry
  have h2 : m = 2 := sorry
  have h3 : P = (2 - 3, 2 + 2) := sorry
  have h4 : P = (-1, 4) := sorry
  exact h4

end NUMINAMATH_GPT_second_quadrant_distance_l10_1063


namespace NUMINAMATH_GPT_covered_area_of_strips_l10_1002

/-- Four rectangular strips of paper, each 16 cm long and 2 cm wide, overlap on a table. 
    We need to prove that the total area of the table surface covered by these strips is 112 cm². --/

theorem covered_area_of_strips (length width : ℝ) (number_of_strips : ℕ) (intersections : ℕ) 
    (area_of_strip : ℝ) (total_area_without_overlap : ℝ) (overlap_area : ℝ) 
    (actual_covered_area : ℝ) :
  length = 16 →
  width = 2 →
  number_of_strips = 4 →
  intersections = 4 →
  area_of_strip = length * width →
  total_area_without_overlap = number_of_strips * area_of_strip →
  overlap_area = intersections * (width * width) →
  actual_covered_area = total_area_without_overlap - overlap_area →
  actual_covered_area = 112 := 
by
  intros
  sorry

end NUMINAMATH_GPT_covered_area_of_strips_l10_1002


namespace NUMINAMATH_GPT_infinite_solutions_xyz_l10_1062

theorem infinite_solutions_xyz : ∀ k : ℕ, 
  (∃ n : ℕ, n > k ∧ ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008) →
  ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008 := 
sorry

end NUMINAMATH_GPT_infinite_solutions_xyz_l10_1062


namespace NUMINAMATH_GPT_distance_second_day_l10_1089

theorem distance_second_day 
  (total_distance : ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (r : ℚ)
  (hn : n = 6)
  (htotal : total_distance = 378)
  (hr : r = 1 / 2)
  (geo_sum : a1 * (1 - r^n) / (1 - r) = total_distance) :
  a1 * r = 96 :=
by
  sorry

end NUMINAMATH_GPT_distance_second_day_l10_1089


namespace NUMINAMATH_GPT_peanuts_in_box_l10_1034

theorem peanuts_in_box (original_peanuts added_peanuts total_peanuts : ℕ) (h1 : original_peanuts = 10) (h2 : added_peanuts = 8) (h3 : total_peanuts = original_peanuts + added_peanuts) : total_peanuts = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_peanuts_in_box_l10_1034


namespace NUMINAMATH_GPT_large_pizzas_sold_l10_1055

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end NUMINAMATH_GPT_large_pizzas_sold_l10_1055


namespace NUMINAMATH_GPT_probability_two_red_marbles_l10_1011

theorem probability_two_red_marbles
  (red_marbles : ℕ)
  (white_marbles : ℕ)
  (total_marbles : ℕ)
  (prob_first_red : ℚ)
  (prob_second_red_after_first_red : ℚ)
  (combined_probability : ℚ) :
  red_marbles = 5 →
  white_marbles = 7 →
  total_marbles = 12 →
  prob_first_red = 5 / 12 →
  prob_second_red_after_first_red = 4 / 11 →
  combined_probability = 5 / 33 →
  combined_probability = prob_first_red * prob_second_red_after_first_red := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_probability_two_red_marbles_l10_1011


namespace NUMINAMATH_GPT_probability_three_one_l10_1052

-- Definitions based on the conditions
def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4

-- Defining the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the total number of ways to draw 4 balls from 18
def total_ways_to_draw : ℕ := binom total_balls drawn_balls

-- Definition of the number of favorable ways to draw 3 black and 1 white ball
def favorable_black_white : ℕ := binom black_balls 3 * binom white_balls 1

-- Definition of the number of favorable ways to draw 1 black and 3 white balls
def favorable_white_black : ℕ := binom black_balls 1 * binom white_balls 3

-- Total favorable outcomes
def total_favorable_ways : ℕ := favorable_black_white + favorable_white_black

-- The probability of drawing 3 one color and 1 other color
def probability : ℚ := total_favorable_ways / total_ways_to_draw

-- Prove that the probability is 19/38
theorem probability_three_one :
  probability = 19 / 38 :=
sorry

end NUMINAMATH_GPT_probability_three_one_l10_1052


namespace NUMINAMATH_GPT_buying_beams_l10_1051

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end NUMINAMATH_GPT_buying_beams_l10_1051


namespace NUMINAMATH_GPT_mrs_santiago_more_roses_l10_1035

theorem mrs_santiago_more_roses :
  58 - 24 = 34 :=
by 
  sorry

end NUMINAMATH_GPT_mrs_santiago_more_roses_l10_1035


namespace NUMINAMATH_GPT_initial_investment_B_l10_1097
-- Import necessary Lean library

-- Define the necessary conditions and theorems
theorem initial_investment_B (x : ℝ) (profit_A : ℝ) (profit_total : ℝ)
  (initial_A : ℝ) (initial_A_after_8_months : ℝ) (profit_B : ℝ) 
  (initial_A_months : ℕ) (initial_A_after_8_months_months : ℕ) 
  (initial_B_months : ℕ) (initial_B_after_8_months_months : ℕ) : 
  initial_A = 3000 ∧ initial_A_after_8_months = 2000 ∧
  profit_A = 240 ∧ profit_total = 630 ∧ 
  profit_B = profit_total - profit_A ∧
  (initial_A * initial_A_months + initial_A_after_8_months * initial_A_after_8_months_months) /
  ((initial_B_months * x + initial_B_after_8_months_months * (x + 1000))) = 
  profit_A / profit_B →
  x = 4000 :=
by
  sorry

end NUMINAMATH_GPT_initial_investment_B_l10_1097


namespace NUMINAMATH_GPT_brady_june_hours_l10_1033

variable (x : ℕ) -- Number of hours worked every day in June

def hoursApril : ℕ := 6 * 30 -- Total hours in April
def hoursSeptember : ℕ := 8 * 30 -- Total hours in September
def hoursJune (x : ℕ) : ℕ := x * 30 -- Total hours in June
def totalHours (x : ℕ) : ℕ := hoursApril + hoursJune x + hoursSeptember -- Total hours over three months
def averageHours (x : ℕ) : ℕ := totalHours x / 3 -- Average hours per month

theorem brady_june_hours (h : averageHours x = 190) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_brady_june_hours_l10_1033


namespace NUMINAMATH_GPT_max_n_minus_m_l10_1003

/-- The function defined with given parameters. -/
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem max_n_minus_m (a b : ℝ) (h1 : -a / 2 = 1)
    (h2 : ∀ x, f x a b ≥ 2)
    (h3 : ∃ m n, (∀ x, f x a b ≤ 6 → m ≤ x ∧ x ≤ n) ∧ (n = 3 ∧ m = -1)) : 
    (∀ m n, (m ≤ n) → (n - m ≤ 4)) :=
by sorry

end NUMINAMATH_GPT_max_n_minus_m_l10_1003


namespace NUMINAMATH_GPT_compute_value_condition_l10_1085

theorem compute_value_condition (x : ℝ) (h : x + (1 / x) = 3) :
  (x - 2) ^ 2 + 25 / (x - 2) ^ 2 = -x + 5 := by
  sorry

end NUMINAMATH_GPT_compute_value_condition_l10_1085


namespace NUMINAMATH_GPT_range_of_a_l10_1008

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l10_1008


namespace NUMINAMATH_GPT_vanya_correct_answers_l10_1018

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_vanya_correct_answers_l10_1018


namespace NUMINAMATH_GPT_households_with_at_least_one_appliance_l10_1067

theorem households_with_at_least_one_appliance (total: ℕ) (color_tvs: ℕ) (refrigerators: ℕ) (both: ℕ) :
  total = 100 → color_tvs = 65 → refrigerators = 84 → both = 53 →
  (color_tvs + refrigerators - both) = 96 :=
by
  intros
  sorry

end NUMINAMATH_GPT_households_with_at_least_one_appliance_l10_1067


namespace NUMINAMATH_GPT_votes_cast_l10_1000

theorem votes_cast (V : ℝ) (candidate_votes : ℝ) (rival_margin : ℝ)
  (h1 : candidate_votes = 0.30 * V)
  (h2 : rival_margin = 4000)
  (h3 : 0.30 * V + (0.30 * V + rival_margin) = V) :
  V = 10000 := 
by 
  sorry

end NUMINAMATH_GPT_votes_cast_l10_1000


namespace NUMINAMATH_GPT_heaviest_lightest_difference_total_excess_weight_total_selling_price_l10_1019

-- Define deviations from standard weight and their counts
def deviations : List (ℚ × ℕ) := [(-3.5, 2), (-2, 4), (-1.5, 2), (0, 1), (1, 3), (2.5, 8)]

-- Define standard weight and price per kg
def standard_weight : ℚ := 18
def price_per_kg : ℚ := 1.8

-- Prove the three statements:
theorem heaviest_lightest_difference :
  (2.5 - (-3.5)) = 6 := by
  sorry

theorem total_excess_weight :
  (2 * -3.5 + 4 * -2 + 2 * -1.5 + 1 * 0 + 3 * 1 + 8 * 2.5) = 5 := by
  sorry

theorem total_selling_price :
  (standard_weight * 20 + 5) * price_per_kg = 657 := by
  sorry

end NUMINAMATH_GPT_heaviest_lightest_difference_total_excess_weight_total_selling_price_l10_1019


namespace NUMINAMATH_GPT_Nancy_more_pearl_beads_l10_1007

-- Define the problem conditions
def metal_beads_Nancy : ℕ := 40
def crystal_beads_Rose : ℕ := 20
def stone_beads_Rose : ℕ := crystal_beads_Rose * 2
def total_beads_needed : ℕ := 20 * 8
def total_Rose_beads : ℕ := crystal_beads_Rose + stone_beads_Rose
def pearl_beads_Nancy : ℕ := total_beads_needed - total_Rose_beads

-- State the theorem to prove
theorem Nancy_more_pearl_beads :
  pearl_beads_Nancy = metal_beads_Nancy + 60 :=
by
  -- We leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_Nancy_more_pearl_beads_l10_1007


namespace NUMINAMATH_GPT_area_triangle_MDA_l10_1014

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ := 
  let AM := r / 3
  let OM := (r ^ 2 - (AM ^ 2)).sqrt
  let AD := AM / 2
  let DM := AD / (1 / 2)
  1 / 2 * AD * DM

theorem area_triangle_MDA (r : ℝ) : area_of_triangle_MDA r = r ^ 2 / 36 := by
  sorry

end NUMINAMATH_GPT_area_triangle_MDA_l10_1014


namespace NUMINAMATH_GPT_green_socks_count_l10_1093

theorem green_socks_count: 
  ∀ (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) (red_socks : ℕ) (green_socks : ℕ),
  total_socks = 900 →
  white_socks = total_socks / 3 →
  blue_socks = total_socks / 4 →
  red_socks = total_socks / 5 →
  green_socks = total_socks - (white_socks + blue_socks + red_socks) →
  green_socks = 195 :=
by
  intros total_socks white_socks blue_socks red_socks green_socks
  sorry

end NUMINAMATH_GPT_green_socks_count_l10_1093


namespace NUMINAMATH_GPT_trapezoid_area_calc_l10_1096

noncomputable def isoscelesTrapezoidArea : ℝ :=
  let a := 1
  let b := 9
  let h := 2 * Real.sqrt 3
  0.5 * (a + b) * h

theorem trapezoid_area_calc : isoscelesTrapezoidArea = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_calc_l10_1096


namespace NUMINAMATH_GPT_number_of_packages_sold_l10_1026

noncomputable def supplier_charges (P : ℕ) : ℕ :=
  if P ≤ 10 then 25 * P
  else 250 + 20 * (P - 10)

theorem number_of_packages_sold
  (supplier_received : ℕ)
  (percent_to_X : ℕ)
  (percent_to_Y : ℕ)
  (percent_to_Z : ℕ)
  (per_package_price : ℕ)
  (discount_percent : ℕ)
  (discount_threshold : ℕ)
  (P : ℕ)
  (h_received : supplier_received = 1340)
  (h_to_X : percent_to_X = 15)
  (h_to_Y : percent_to_Y = 15)
  (h_to_Z : percent_to_Z = 70)
  (h_full_price : per_package_price = 25)
  (h_discount : discount_percent = 4 * per_package_price / 5)
  (h_threshold : discount_threshold = 10)
  (h_calculation : supplier_charges P = supplier_received) : P = 65 := 
sorry

end NUMINAMATH_GPT_number_of_packages_sold_l10_1026


namespace NUMINAMATH_GPT_original_paint_intensity_l10_1059

theorem original_paint_intensity 
  (P : ℝ)
  (H1 : 0 ≤ P ∧ P ≤ 100)
  (H2 : ∀ (unit : ℝ), unit = 100)
  (H3 : ∀ (replaced_fraction : ℝ), replaced_fraction = 1.5)
  (H4 : ∀ (new_intensity : ℝ), new_intensity = 30)
  (H5 : ∀ (solution_intensity : ℝ), solution_intensity = 0.25) :
  P = 15 := 
by
  sorry

end NUMINAMATH_GPT_original_paint_intensity_l10_1059


namespace NUMINAMATH_GPT_prime_p_squared_plus_71_divisors_l10_1032

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def num_distinct_divisors (n : ℕ) : ℕ :=
  (factors n).toFinset.card

theorem prime_p_squared_plus_71_divisors (p : ℕ) (hp : is_prime p) 
  (hdiv : num_distinct_divisors (p ^ 2 + 71) ≤ 10) : p = 2 ∨ p = 3 :=
sorry

end NUMINAMATH_GPT_prime_p_squared_plus_71_divisors_l10_1032


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l10_1077

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S3 : ℝ) 
  (h1 : a 1 = 1) (h2 : S3 = 3 / 4) 
  (h3 : S3 = a 1 + a 1 * q + a 1 * q^2) :
  q = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l10_1077


namespace NUMINAMATH_GPT_distance_between_first_and_last_is_140_l10_1058

-- Given conditions
def eightFlowers : ℕ := 8
def distanceFirstToFifth : ℕ := 80
def intervalsBetweenFirstAndFifth : ℕ := 4 -- 1 to 5 means 4 intervals
def intervalsBetweenFirstAndLast : ℕ := 7 -- 1 to 8 means 7 intervals
def distanceBetweenConsecutiveFlowers : ℕ := distanceFirstToFifth / intervalsBetweenFirstAndFifth
def totalDistanceFirstToLast : ℕ := distanceBetweenConsecutiveFlowers * intervalsBetweenFirstAndLast

-- Theorem to prove the question equals the correct answer
theorem distance_between_first_and_last_is_140 :
  totalDistanceFirstToLast = 140 := by
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_is_140_l10_1058


namespace NUMINAMATH_GPT_fraction_product_l10_1070

theorem fraction_product : 
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_l10_1070


namespace NUMINAMATH_GPT_area_ratio_of_similar_polygons_l10_1027

theorem area_ratio_of_similar_polygons (similarity_ratio: ℚ) (hratio: similarity_ratio = 1/5) : (similarity_ratio ^ 2 = 1/25) := 
by 
  sorry

end NUMINAMATH_GPT_area_ratio_of_similar_polygons_l10_1027


namespace NUMINAMATH_GPT_curve_is_segment_l10_1016

noncomputable def parametric_curve := {t : ℝ // 0 ≤ t ∧ t ≤ 5}

def x (t : parametric_curve) : ℝ := 3 * t.val ^ 2 + 2
def y (t : parametric_curve) : ℝ := t.val ^ 2 - 1

def line_equation (x y : ℝ) := x - 3 * y - 5 = 0

theorem curve_is_segment :
  ∀ (t : parametric_curve), line_equation (x t) (y t) ∧ 
  2 ≤ x t ∧ x t ≤ 77 :=
by
  sorry

end NUMINAMATH_GPT_curve_is_segment_l10_1016


namespace NUMINAMATH_GPT_value_of_a_l10_1079

theorem value_of_a (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x - 1 = 0) ∧ (∀ x y : ℝ, a * x^2 - 2 * x - 1 = 0 ∧ a * y^2 - 2 * y - 1 = 0 → x = y) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l10_1079


namespace NUMINAMATH_GPT_area_ratio_of_squares_l10_1053

theorem area_ratio_of_squares (R x y : ℝ) (hx : x^2 = (4/5) * R^2) (hy : y = R * Real.sqrt 2) :
  x^2 / y^2 = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l10_1053


namespace NUMINAMATH_GPT_sin_alpha_beta_gamma_values_l10_1075

open Real

theorem sin_alpha_beta_gamma_values (α β γ : ℝ)
  (h1 : sin α = sin (α + β + γ) + 1)
  (h2 : sin β = 3 * sin (α + β + γ) + 2)
  (h3 : sin γ = 5 * sin (α + β + γ) + 3) :
  sin α * sin β * sin γ = (3/64) ∨ sin α * sin β * sin γ = (1/8) :=
sorry

end NUMINAMATH_GPT_sin_alpha_beta_gamma_values_l10_1075


namespace NUMINAMATH_GPT_bird_families_difference_l10_1017

theorem bird_families_difference {initial_families flown_away : ℕ} (h1 : initial_families = 87) (h2 : flown_away = 7) :
  (initial_families - flown_away) - flown_away = 73 := by
sorry

end NUMINAMATH_GPT_bird_families_difference_l10_1017


namespace NUMINAMATH_GPT_mineral_sample_ages_l10_1040

/--
We have a mineral sample with digits {2, 2, 3, 3, 5, 9}.
Given the condition that the age must start with an odd number,
we need to prove that the total number of possible ages is 120.
-/
theorem mineral_sample_ages : 
  ∀ (l : List ℕ), l = [2, 2, 3, 3, 5, 9] → 
  (l.filter odd).length > 0 →
  ∃ n : ℕ, n = 120 :=
by
  intros l h_digits h_odd
  sorry

end NUMINAMATH_GPT_mineral_sample_ages_l10_1040


namespace NUMINAMATH_GPT_intersection_of_lines_l10_1036

theorem intersection_of_lines :
  ∃ (x y : ℚ), (8 * x - 3 * y = 24) ∧ (10 * x + 2 * y = 14) ∧ x = 45 / 23 ∧ y = -64 / 23 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l10_1036


namespace NUMINAMATH_GPT_negative_solution_range_l10_1098

theorem negative_solution_range (m : ℝ) : (∃ x : ℝ, 2 * x + 4 = m - x ∧ x < 0) → m < 4 := by
  sorry

end NUMINAMATH_GPT_negative_solution_range_l10_1098


namespace NUMINAMATH_GPT_max_f_value_l10_1046

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end NUMINAMATH_GPT_max_f_value_l10_1046


namespace NUMINAMATH_GPT_absolute_value_equation_solution_l10_1048

theorem absolute_value_equation_solution (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) ↔
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨ 
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨ 
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_equation_solution_l10_1048


namespace NUMINAMATH_GPT_balls_total_correct_l10_1088

-- Definitions based on the problem conditions
def red_balls_initial : ℕ := 16
def blue_balls : ℕ := 2 * red_balls_initial
def red_balls_lost : ℕ := 6
def red_balls_remaining : ℕ := red_balls_initial - red_balls_lost
def total_balls_after : ℕ := 74
def nonblue_red_balls_remaining : ℕ := red_balls_remaining + blue_balls

-- Goal: Find the number of yellow balls
def yellow_balls_bought : ℕ := total_balls_after - nonblue_red_balls_remaining

theorem balls_total_correct :
  yellow_balls_bought = 32 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_balls_total_correct_l10_1088


namespace NUMINAMATH_GPT_union_sets_l10_1013

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem union_sets : A ∪ B = { x | -1 < x ∧ x ≤ 4 } := 
by
   sorry

end NUMINAMATH_GPT_union_sets_l10_1013


namespace NUMINAMATH_GPT_satellite_modular_units_l10_1042

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end NUMINAMATH_GPT_satellite_modular_units_l10_1042


namespace NUMINAMATH_GPT_determine_constants_l10_1068

theorem determine_constants
  (C D : ℝ)
  (h1 : 3 * C + D = 7)
  (h2 : 4 * C - 2 * D = -15) :
  C = -0.1 ∧ D = 7.3 :=
by
  sorry

end NUMINAMATH_GPT_determine_constants_l10_1068


namespace NUMINAMATH_GPT_handshakes_at_event_l10_1050

theorem handshakes_at_event 
  (num_couples : ℕ) 
  (num_people : ℕ) 
  (num_handshakes_men : ℕ) 
  (num_handshakes_men_women : ℕ) 
  (total_handshakes : ℕ) 
  (cond1 : num_couples = 15) 
  (cond2 : num_people = 2 * num_couples) 
  (cond3 : num_handshakes_men = (num_couples * (num_couples - 1)) / 2) 
  (cond4 : num_handshakes_men_women = num_couples * (num_couples - 1)) 
  (cond5 : total_handshakes = num_handshakes_men + num_handshakes_men_women) : 
  total_handshakes = 315 := 
by sorry

end NUMINAMATH_GPT_handshakes_at_event_l10_1050


namespace NUMINAMATH_GPT_similar_rect_tiling_l10_1038

-- Define the dimensions of rectangles A and B
variables {a1 a2 b1 b2 : ℝ}

-- Define the tiling condition
def similar_tiled (a1 a2 b1 b2 : ℝ) : Prop := 
  -- A placeholder for the actual definition of similar tiling
  sorry

-- The main theorem to prove
theorem similar_rect_tiling (h : similar_tiled a1 a2 b1 b2) : similar_tiled b1 b2 a1 a2 :=
sorry

end NUMINAMATH_GPT_similar_rect_tiling_l10_1038


namespace NUMINAMATH_GPT_line_through_point_parallel_to_line_l10_1009

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_to_line_l10_1009


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l10_1020

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 + n^3 - n - 1) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l10_1020


namespace NUMINAMATH_GPT_license_plate_count_l10_1095

def num_license_plates : Nat :=
  let letters := 26 -- choices for each of the first two letters
  let primes := 4 -- choices for prime digits
  let composites := 4 -- choices for composite digits
  letters * letters * (primes * composites * 2)

theorem license_plate_count : num_license_plates = 21632 :=
  by
  sorry

end NUMINAMATH_GPT_license_plate_count_l10_1095


namespace NUMINAMATH_GPT_add_to_fraction_eq_l10_1012

theorem add_to_fraction_eq (n : ℕ) : (4 + n) / (7 + n) = 6 / 7 → n = 14 :=
by sorry

end NUMINAMATH_GPT_add_to_fraction_eq_l10_1012


namespace NUMINAMATH_GPT_smallest_w_l10_1010

theorem smallest_w (x y w : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 ^ x) ∣ (3125 * w)) (h4 : (3 ^ y) ∣ (3125 * w)) 
  (h5 : (5 ^ (x + y)) ∣ (3125 * w)) (h6 : (7 ^ (x - y)) ∣ (3125 * w))
  (h7 : (13 ^ 4) ∣ (3125 * w))
  (h8 : x + y ≤ 10) (h9 : x - y ≥ 2) :
  w = 33592336 :=
by
  sorry

end NUMINAMATH_GPT_smallest_w_l10_1010


namespace NUMINAMATH_GPT_generating_sets_Z2_l10_1045

theorem generating_sets_Z2 (a b : ℤ × ℤ) (h : Submodule.span ℤ ({a, b} : Set (ℤ × ℤ)) = ⊤) :
  let a₁ := a.1
  let a₂ := a.2
  let b₁ := b.1
  let b₂ := b.2
  a₁ * b₂ - a₂ * b₁ = 1 ∨ a₁ * b₂ - a₂ * b₁ = -1 := 
by
  sorry

end NUMINAMATH_GPT_generating_sets_Z2_l10_1045


namespace NUMINAMATH_GPT_product_of_xy_l10_1015

theorem product_of_xy (x y : ℝ) : 
  (1 / 5 * (x + y + 4 + 5 + 6) = 5) ∧ 
  (1 / 5 * ((x - 5) ^ 2 + (y - 5) ^ 2 + (4 - 5) ^ 2 + (5 - 5) ^ 2 + (6 - 5) ^ 2) = 2) 
  → x * y = 21 :=
by sorry

end NUMINAMATH_GPT_product_of_xy_l10_1015


namespace NUMINAMATH_GPT_compound_interest_correct_amount_l10_1028

-- Define constants and conditions
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def compound_interest (P R T : ℕ) : ℕ := P * ((1 + R / 100) ^ T - 1)

-- Given values and conditions
def P₁ : ℕ := 1750
def R₁ : ℕ := 8
def T₁ : ℕ := 3
def R₂ : ℕ := 10
def T₂ : ℕ := 2

def SI : ℕ := simple_interest P₁ R₁ T₁
def CI : ℕ := 2 * SI

def P₂ : ℕ := 4000

-- The statement to be proven
theorem compound_interest_correct_amount : 
  compound_interest P₂ R₂ T₂ = CI := 
by 
  sorry

end NUMINAMATH_GPT_compound_interest_correct_amount_l10_1028


namespace NUMINAMATH_GPT_ratio_of_length_to_perimeter_is_one_over_four_l10_1092

-- We define the conditions as given in the problem.
def room_length_1 : ℕ := 23 -- length of the rectangle in feet
def room_width_1 : ℕ := 15  -- width of the rectangle in feet
def room_width_2 : ℕ := 8   -- side of the square in feet

-- Total dimensions after including the square
def total_length : ℕ := room_length_1  -- total length remains the same
def total_width : ℕ := room_width_1 + room_width_2  -- width is sum of widths

-- Defining the perimeter
def perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width

-- Calculate the ratio
def length_to_perimeter_ratio (length perimeter : ℕ) : ℚ := length / perimeter

-- Theorem to prove the desired ratio is 1:4
theorem ratio_of_length_to_perimeter_is_one_over_four : 
  length_to_perimeter_ratio total_length (perimeter total_length total_width) = 1 / 4 :=
by
  -- Proof code would go here
  sorry

end NUMINAMATH_GPT_ratio_of_length_to_perimeter_is_one_over_four_l10_1092


namespace NUMINAMATH_GPT_james_total_matches_l10_1054

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end NUMINAMATH_GPT_james_total_matches_l10_1054


namespace NUMINAMATH_GPT_math_problem_l10_1031

theorem math_problem (x y : ℤ) (a b : ℤ) (h1 : x - 5 = 7 * a) (h2 : y + 7 = 7 * b) (h3 : (x ^ 2 + y ^ 3) % 11 = 0) : 
  ((y - x) / 13) = 13 :=
sorry

end NUMINAMATH_GPT_math_problem_l10_1031


namespace NUMINAMATH_GPT_min_value_arithmetic_sequence_l10_1066

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 2014 = 2) :
  (∃ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 ∧ a2 > 0 ∧ a2013 > 0 ∧ ∀ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 → (1/a2 + 1/a2013) ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_arithmetic_sequence_l10_1066


namespace NUMINAMATH_GPT_j_mod_2_not_zero_l10_1022

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 :=
sorry

end NUMINAMATH_GPT_j_mod_2_not_zero_l10_1022


namespace NUMINAMATH_GPT_coordinates_of_a_l10_1006

theorem coordinates_of_a
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (1, 2))
  (h1 : (a.1)^2 + (a.2)^2 = 5)
  (h2 : ∃ k : ℝ, a = (k, 2 * k))
  : a = (1, 2) ∨ a = (-1, -2) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_a_l10_1006


namespace NUMINAMATH_GPT_tickets_left_l10_1004

-- Define the number of tickets won by Dave
def tickets_won : ℕ := 14

-- Define the number of tickets lost by Dave
def tickets_lost : ℕ := 2

-- Define the number of tickets used to buy toys
def tickets_used : ℕ := 10

-- The theorem to prove that the number of tickets left is 2
theorem tickets_left : tickets_won - tickets_lost - tickets_used = 2 := by
  -- Initial computation of tickets left after losing some
  let tickets_after_lost := tickets_won - tickets_lost
  -- Computation of tickets left after using some
  let tickets_after_used := tickets_after_lost - tickets_used
  show tickets_after_used = 2
  sorry

end NUMINAMATH_GPT_tickets_left_l10_1004


namespace NUMINAMATH_GPT_geometric_sum_is_correct_l10_1084

theorem geometric_sum_is_correct : 
  let a := 1
  let r := 5
  let n := 6
  a * (r^n - 1) / (r - 1) = 3906 := by
  sorry

end NUMINAMATH_GPT_geometric_sum_is_correct_l10_1084


namespace NUMINAMATH_GPT_eval_expression_solve_inequalities_l10_1037

-- Problem 1: Evaluation of the expression equals sqrt(2)
theorem eval_expression : (1 - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + |Real.sqrt 2 - 1|) = Real.sqrt 2 := 
by sorry

-- Problem 2: Solution set of the inequality system
theorem solve_inequalities (x : ℝ) : 
  ((3 * x + 1) / 2 ≥ (4 * x + 3) / 3 ∧ 2 * x + 7 ≥ 5 * x - 17) ↔ (3 ≤ x ∧ x ≤ 8) :=
by sorry

end NUMINAMATH_GPT_eval_expression_solve_inequalities_l10_1037


namespace NUMINAMATH_GPT_algebraic_expression_value_l10_1081

theorem algebraic_expression_value (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (  ( ((x + 2)^2 * (x^2 - 2 * x + 4)^2) / ( (x^3 + 8)^2 ))^2
   * ( ((x - 2)^2 * (x^2 + 2 * x + 4)^2) / ( (x^3 - 8)^2 ))^2 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l10_1081


namespace NUMINAMATH_GPT_area_square_EFGH_equiv_144_l10_1065

theorem area_square_EFGH_equiv_144 (a b : ℝ) (h : a = 6) (hb : b = 6)
  (side_length_EFGH : ℝ) (hs : side_length_EFGH = a + 3 + 3) : side_length_EFGH ^ 2 = 144 :=
by
  -- Given conditions
  sorry

end NUMINAMATH_GPT_area_square_EFGH_equiv_144_l10_1065


namespace NUMINAMATH_GPT_unique_solution_n_l10_1099

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem unique_solution_n (h : ∀ n : ℕ, (n > 0) → n^3 = 8 * (sum_digits n)^3 + 6 * (sum_digits n) * n + 1 → n = 17) : 
  n = 17 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_n_l10_1099


namespace NUMINAMATH_GPT_commutativity_associativity_l10_1029

variables {α : Type*} (op : α → α → α)

-- Define conditions as hypotheses
axiom cond1 : ∀ a b c : α, op a (op b c) = op b (op c a)
axiom cond2 : ∀ a b c : α, op a b = op a c → b = c
axiom cond3 : ∀ a b c : α, op a c = op b c → a = b

-- Commutativity statement
theorem commutativity (a b : α) : op a b = op b a := sorry

-- Associativity statement
theorem associativity (a b c : α) : op (op a b) c = op a (op b c) := sorry

end NUMINAMATH_GPT_commutativity_associativity_l10_1029


namespace NUMINAMATH_GPT_gcd_lcm_product_135_l10_1078

theorem gcd_lcm_product_135 (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 135 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_135_l10_1078


namespace NUMINAMATH_GPT_find_a_l10_1094

theorem find_a (b c : ℤ) 
  (vertex_condition : ∀ (x : ℝ), x = -1 → (ax^2 + b*x + c) = -2)
  (point_condition : ∀ (x : ℝ), x = 0 → (a*x^2 + b*x + c) = -1) :
  ∃ (a : ℤ), a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l10_1094


namespace NUMINAMATH_GPT_log_property_l10_1049

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem log_property (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m + f n :=
by
  sorry

end NUMINAMATH_GPT_log_property_l10_1049


namespace NUMINAMATH_GPT_cyclist_time_to_climb_and_descend_hill_l10_1001

noncomputable def hill_length : ℝ := 400 -- hill length in meters
noncomputable def ascent_speed_kmh : ℝ := 7.2 -- ascent speed in km/h
noncomputable def ascent_speed_ms : ℝ := ascent_speed_kmh * 1000 / 3600 -- ascent speed converted in m/s
noncomputable def descent_speed_ms : ℝ := 2 * ascent_speed_ms -- descent speed in m/s

noncomputable def time_to_climb : ℝ := hill_length / ascent_speed_ms -- time to climb in seconds
noncomputable def time_to_descend : ℝ := hill_length / descent_speed_ms -- time to descend in seconds
noncomputable def total_time : ℝ := time_to_climb + time_to_descend -- total time in seconds

theorem cyclist_time_to_climb_and_descend_hill : total_time = 300 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_time_to_climb_and_descend_hill_l10_1001


namespace NUMINAMATH_GPT_find_x_l10_1023

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) : 
  let l5 := log_base 5 x
  let l6 := log_base 6 x
  let l7 := log_base 7 x
  let surface_area := 2 * (l5 * l6 + l5 * l7 + l6 * l7)
  let volume := l5 * l6 * l7 
  (surface_area = 2 * volume) → x = 210 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l10_1023


namespace NUMINAMATH_GPT_sum_of_exponents_l10_1041

theorem sum_of_exponents (n : ℕ) (h : n = 896) : 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 2^a + 2^b + 2^c = n ∧ a + b + c = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_exponents_l10_1041
