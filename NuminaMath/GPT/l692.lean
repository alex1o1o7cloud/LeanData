import Mathlib

namespace shaded_region_area_l692_69259

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l692_69259


namespace integer_sequence_existence_l692_69203

theorem integer_sequence_existence
  (n : ℕ) (a : ℕ → ℤ) (A B C : ℤ) 
  (h1 : (a 1 < A ∧ A < B ∧ B < a n) ∨ (a 1 > A ∧ A > B ∧ B > a n))
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n - 1 → (a (i + 1) - a i ≤ 1 ∨ a (i + 1) - a i ≥ -1))
  (h3 : A ≤ C ∧ C ≤ B ∨ A ≥ C ∧ C ≥ B) :
  ∃ i, 1 < i ∧ i < n ∧ a i = C := sorry

end integer_sequence_existence_l692_69203


namespace minimum_jellybeans_l692_69295

theorem minimum_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n = 164 :=
by sorry

end minimum_jellybeans_l692_69295


namespace range_of_a_minus_b_l692_69209

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) : 
  -3 < a - b ∧ a - b < 6 :=
by
  sorry

end range_of_a_minus_b_l692_69209


namespace union_of_subsets_l692_69291

open Set

variable (A B : Set ℕ)

theorem union_of_subsets (m : ℕ) (hA : A = {1, 3}) (hB : B = {1, 2, m}) (hSubset : A ⊆ B) :
    A ∪ B = {1, 2, 3} :=
  sorry

end union_of_subsets_l692_69291


namespace intersection_of_M_and_N_l692_69229

open Set

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
by {
  sorry
}

end intersection_of_M_and_N_l692_69229


namespace abs_sub_eq_abs_sub_l692_69264

theorem abs_sub_eq_abs_sub (a b : ℚ) : |a - b| = |b - a| :=
sorry

end abs_sub_eq_abs_sub_l692_69264


namespace value_of_g_neg3_l692_69218

def g (x : ℝ) : ℝ := x^3 - 2 * x

theorem value_of_g_neg3 : g (-3) = -21 := by
  sorry

end value_of_g_neg3_l692_69218


namespace sum_of_imaginary_parts_l692_69273

theorem sum_of_imaginary_parts (x y u v w z : ℝ) (h1 : y = 5) 
  (h2 : w = -x - u) (h3 : (x + y * I) + (u + v * I) + (w + z * I) = 4 * I) :
  v + z = -1 :=
by
  sorry

end sum_of_imaginary_parts_l692_69273


namespace total_distance_of_race_is_150_l692_69272

variable (D : ℝ)

-- Conditions
def A_covers_distance_in_45_seconds (D : ℝ) : Prop := ∃ A_speed, A_speed = D / 45
def B_covers_distance_in_60_seconds (D : ℝ) : Prop := ∃ B_speed, B_speed = D / 60
def A_beats_B_by_50_meters_in_60_seconds (D : ℝ) : Prop := (D / 45) * 60 = D + 50

theorem total_distance_of_race_is_150 :
  A_covers_distance_in_45_seconds D ∧ 
  B_covers_distance_in_60_seconds D ∧ 
  A_beats_B_by_50_meters_in_60_seconds D → 
  D = 150 :=
by
  sorry

end total_distance_of_race_is_150_l692_69272


namespace maximum_sum_set_l692_69274

def no_two_disjoint_subsets_have_equal_sums (S : Finset ℕ) : Prop :=
  ∀ (A B : Finset ℕ), A ≠ B ∧ A ∩ B = ∅ → (A.sum id) ≠ (B.sum id)

theorem maximum_sum_set (S : Finset ℕ) (h : ∀ x ∈ S, x ≤ 15) (h_subset_sum : no_two_disjoint_subsets_have_equal_sums S) : S.sum id = 61 :=
sorry

end maximum_sum_set_l692_69274


namespace cube_surface_area_l692_69255

theorem cube_surface_area (Q : ℝ) (a : ℝ) (H : (3 * a^2 * Real.sqrt 3) / 2 = Q) :
    (6 * (a * Real.sqrt 2) ^ 2) = (8 * Q * Real.sqrt 3) / 3 :=
by
  sorry

end cube_surface_area_l692_69255


namespace dark_lord_squads_l692_69206

def total_weight : ℕ := 1200
def orcs_per_squad : ℕ := 8
def capacity_per_orc : ℕ := 15
def squads_needed (w n c : ℕ) : ℕ := w / (n * c)

theorem dark_lord_squads :
  squads_needed total_weight orcs_per_squad capacity_per_orc = 10 :=
by sorry

end dark_lord_squads_l692_69206


namespace new_shoes_last_for_two_years_l692_69243

theorem new_shoes_last_for_two_years :
  let cost_repair := 11.50
  let cost_new := 28.00
  let increase_factor := 1.2173913043478261
  (cost_new / ((increase_factor) * cost_repair)) ≠ 0 :=
by
  sorry

end new_shoes_last_for_two_years_l692_69243


namespace Q3_x_coords_sum_eq_Q1_x_coords_sum_l692_69250

-- Define a 40-gon and its x-coordinates sum
def Q1_x_coords_sum : ℝ := 120

-- Statement to prove
theorem Q3_x_coords_sum_eq_Q1_x_coords_sum (Q1_x_coords_sum: ℝ) (h: Q1_x_coords_sum = 120) : 
  (Q3_x_coords_sum: ℝ) = Q1_x_coords_sum :=
sorry

end Q3_x_coords_sum_eq_Q1_x_coords_sum_l692_69250


namespace smallest_multiple_of_84_with_6_and_7_l692_69201

variable (N : Nat)

def is_multiple_of_84 (N : Nat) : Prop :=
  N % 84 = 0

def consists_of_6_and_7 (N : Nat) : Prop :=
  ∀ d ∈ N.digits 10, d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  ∃ N, is_multiple_of_84 N ∧ consists_of_6_and_7 N ∧ ∀ M, is_multiple_of_84 M ∧ consists_of_6_and_7 M → N ≤ M := 
sorry

end smallest_multiple_of_84_with_6_and_7_l692_69201


namespace car_travel_distance_l692_69270

theorem car_travel_distance (distance : ℝ) 
  (speed1 : ℝ := 80) 
  (speed2 : ℝ := 76.59574468085106) 
  (time_difference : ℝ := 2 / 3600) : 
  (distance / speed2 = distance / speed1 + time_difference) → 
  distance = 0.998177 :=
by
  -- assuming the above equation holds, we need to conclude the distance
  sorry

end car_travel_distance_l692_69270


namespace pq_difference_l692_69204

theorem pq_difference (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end pq_difference_l692_69204


namespace inscribed_circle_radius_eq_3_l692_69267

open Real

theorem inscribed_circle_radius_eq_3
  (a : ℝ) (A : ℝ) (p : ℝ) (r : ℝ)
  (h_eq_tri : ∀ (a : ℝ), A = (sqrt 3 / 4) * a^2)
  (h_perim : ∀ (a : ℝ), p = 3 * a)
  (h_area_perim : ∀ (a : ℝ), A = (3 / 2) * p) :
  r = 3 :=
by sorry

end inscribed_circle_radius_eq_3_l692_69267


namespace remainder_of_product_mod_17_l692_69294

theorem remainder_of_product_mod_17 :
  (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 :=
sorry

end remainder_of_product_mod_17_l692_69294


namespace part1_infinite_n_part2_no_solutions_l692_69210

-- Definitions for part (1)
theorem part1_infinite_n (n : ℕ) (x y z t : ℕ) :
  (∃ n, x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

-- Definitions for part (2)
theorem part2_no_solutions (n k m x y z t : ℕ) :
  n = 4 ^ k * (8 * m + 7) → ¬(x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

end part1_infinite_n_part2_no_solutions_l692_69210


namespace sufficient_but_not_necessary_l692_69254

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 3| - |x - 1| < 2) → x ≠ 1 ∧ ¬ (∀ x : ℝ, x ≠ 1 → |x - 3| - |x - 1| < 2) :=
by
  sorry

end sufficient_but_not_necessary_l692_69254


namespace unique_solution_for_all_y_l692_69233

theorem unique_solution_for_all_y (x : ℝ) (h : ∀ y : ℝ, 8 * x * y - 12 * y + 2 * x - 3 = 0) : x = 3 / 2 :=
sorry

end unique_solution_for_all_y_l692_69233


namespace vector_subtraction_proof_l692_69223

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (1, -6)
def scalar1 : ℝ := 2
def scalar2 : ℝ := 3

theorem vector_subtraction_proof :
  v1 - (scalar2 • (scalar1 • v2)) = (-3, 32) := by
  sorry

end vector_subtraction_proof_l692_69223


namespace possible_values_of_g_l692_69298

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
by
  sorry

end possible_values_of_g_l692_69298


namespace math_problem_l692_69258

noncomputable def condition1 (a b : ℤ) : Prop :=
  |2 + a| + |b - 3| = 0

noncomputable def condition2 (c d : ℝ) : Prop :=
  1 / c = -d

noncomputable def condition3 (e : ℤ) : Prop :=
  e = -5

theorem math_problem (a b e : ℤ) (c d : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 c d) 
  (h3 : condition3 e) : 
  -a^b + 1 / c - e + d = 13 :=
by
  sorry

end math_problem_l692_69258


namespace solve_system_correct_l692_69269

noncomputable def solve_system (a b c d e : ℝ) : Prop :=
  3 * a = (b + c + d) ^ 3 ∧ 
  3 * b = (c + d + e) ^ 3 ∧ 
  3 * c = (d + e + a) ^ 3 ∧ 
  3 * d = (e + a + b) ^ 3 ∧ 
  3 * e = (a + b + c) ^ 3

theorem solve_system_correct :
  ∀ (a b c d e : ℝ), solve_system a b c d e → 
    (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨ 
    (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨ 
    (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3) :=
by
  sorry

end solve_system_correct_l692_69269


namespace Jillian_largest_apartment_size_l692_69226

noncomputable def largest_apartment_size (budget rent_per_sqft: ℝ) : ℝ :=
  budget / rent_per_sqft

theorem Jillian_largest_apartment_size :
  largest_apartment_size 720 1.20 = 600 := 
by
  sorry

end Jillian_largest_apartment_size_l692_69226


namespace min_adults_at_amusement_park_l692_69281

def amusement_park_problem : Prop :=
  ∃ (x y z : ℕ), 
    x + y + z = 100 ∧
    3 * x + 2 * y + (3 / 10) * z = 100 ∧
    (∀ (x' : ℕ), x' < 2 → ¬(∃ (y' z' : ℕ), x' + y' + z' = 100 ∧ 3 * x' + 2 * y' + (3 / 10) * z' = 100))

theorem min_adults_at_amusement_park : amusement_park_problem := sorry

end min_adults_at_amusement_park_l692_69281


namespace A_takes_200_seconds_l692_69234

/-- 
  A can give B a start of 50 meters or 10 seconds in a kilometer race.
  How long does A take to complete the race?
-/
theorem A_takes_200_seconds (v_A : ℝ) (distance : ℝ) (start_meters : ℝ) (start_seconds : ℝ) :
  (start_meters = 50) ∧ (start_seconds = 10) ∧ (distance = 1000) ∧ 
  (v_A = start_meters / start_seconds) → distance / v_A = 200 :=
by
  sorry

end A_takes_200_seconds_l692_69234


namespace mod_calculation_l692_69288

theorem mod_calculation : (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end mod_calculation_l692_69288


namespace book_pages_l692_69216

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l692_69216


namespace probability_of_getting_a_prize_l692_69237

theorem probability_of_getting_a_prize {prizes blanks : ℕ} (h_prizes : prizes = 10) (h_blanks : blanks = 25) :
  (prizes / (prizes + blanks) : ℚ) = 2 / 7 :=
by
  sorry

end probability_of_getting_a_prize_l692_69237


namespace remaining_card_number_l692_69215

theorem remaining_card_number (A B C D E F G H : ℕ) (cards : Finset ℕ) 
  (hA : A + B = 10) 
  (hB : C - D = 1) 
  (hC : E * F = 24) 
  (hD : G / H = 3) 
  (hCards : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : A ∉ cards ∧ B ∉ cards ∧ C ∉ cards ∧ D ∉ cards ∧ E ∉ cards ∧ F ∉ cards ∧ G ∉ cards ∧ H ∉ cards) :
  7 ∈ cards := 
by
  sorry

end remaining_card_number_l692_69215


namespace people_owning_only_cats_and_dogs_l692_69275

theorem people_owning_only_cats_and_dogs 
  (total_people : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) 
  (total_snakes : ℕ) 
  (only_cats_and_dogs : ℕ) 
  (h1 : total_people = 89) 
  (h2 : only_dogs = 15) 
  (h3 : only_cats = 10) 
  (h4 : cats_dogs_snakes = 3) 
  (h5 : total_snakes = 59) 
  (h6 : total_people = only_dogs + only_cats + only_cats_and_dogs + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) : 
  only_cats_and_dogs = 5 := 
by 
  sorry

end people_owning_only_cats_and_dogs_l692_69275


namespace algebraic_expression_evaluation_l692_69285

theorem algebraic_expression_evaluation (x y : ℝ) : 
  3 * (x^2 - 2 * x * y + y^2) - 3 * (x^2 - 2 * x * y + y^2 - 1) = 3 :=
by
  sorry

end algebraic_expression_evaluation_l692_69285


namespace expression_evaluates_to_2023_l692_69271

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l692_69271


namespace katie_more_games_l692_69251

noncomputable def katie_games : ℕ := 57 + 39
noncomputable def friends_games : ℕ := 34
noncomputable def games_difference : ℕ := katie_games - friends_games

theorem katie_more_games : games_difference = 62 :=
by
  -- Proof omitted
  sorry

end katie_more_games_l692_69251


namespace floor_expression_equals_zero_l692_69240

theorem floor_expression_equals_zero
  (a b c : ℕ)
  (ha : a = 2010)
  (hb : b = 2007)
  (hc : c = 2008) :
  Int.floor ((a^3 : ℚ) / (b * c^2) - (c^3 : ℚ) / (b^2 * a)) = 0 := 
  sorry

end floor_expression_equals_zero_l692_69240


namespace triangle_side_lengths_l692_69211

theorem triangle_side_lengths (r : ℝ) (AC BC AB : ℝ) (y : ℝ) 
  (h1 : r = 3 * Real.sqrt 2)
  (h2 : AC = 5 * Real.sqrt y) 
  (h3 : BC = 13 * Real.sqrt y) 
  (h4 : AB = 10 * Real.sqrt y) : 
  r = 3 * Real.sqrt 2 → 
  (∃ (AC BC AB : ℝ), 
     AC = 5 * Real.sqrt (7) ∧ 
     BC = 13 * Real.sqrt (7) ∧ 
     AB = 10 * Real.sqrt (7)) :=
by
  sorry

end triangle_side_lengths_l692_69211


namespace linear_function_not_in_second_quadrant_l692_69236

theorem linear_function_not_in_second_quadrant (m : ℤ) (h1 : m + 4 > 0) (h2 : m + 2 ≤ 0) : 
  m = -3 ∨ m = -2 := 
sorry

end linear_function_not_in_second_quadrant_l692_69236


namespace solve_fractional_eq_l692_69231

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l692_69231


namespace extremum_values_l692_69239

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  (∀ x, f x ≤ 5) ∧ f (-1) = 5 ∧ (∀ x, f x ≥ -27) ∧ f 3 = -27 :=
by
  sorry

end extremum_values_l692_69239


namespace total_interest_proof_l692_69287

open Real

def initial_investment : ℝ := 10000
def interest_6_months : ℝ := 0.02 * initial_investment
def reinvested_amount_6_months : ℝ := initial_investment + interest_6_months
def interest_10_months : ℝ := 0.03 * reinvested_amount_6_months
def reinvested_amount_10_months : ℝ := reinvested_amount_6_months + interest_10_months
def interest_18_months : ℝ := 0.04 * reinvested_amount_10_months

def total_interest : ℝ := interest_6_months + interest_10_months + interest_18_months

theorem total_interest_proof : total_interest = 926.24 := by
    sorry

end total_interest_proof_l692_69287


namespace angle_PMN_is_60_l692_69207

-- Define given variables and their types
variable (P M N R Q : Prop)
variable (angle : Prop → Prop → Prop → ℝ)

-- Given conditions
variables (h1 : angle P Q R = 60)
variables (h2 : PM = MN)

-- The statement of what's to be proven
theorem angle_PMN_is_60 :
  angle P M N = 60 := sorry

end angle_PMN_is_60_l692_69207


namespace inequality_inverse_l692_69268

theorem inequality_inverse (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a) < (1 / b) :=
by
  sorry

end inequality_inverse_l692_69268


namespace time_after_1450_minutes_l692_69247

theorem time_after_1450_minutes (initial_time_in_minutes : ℕ := 360) (minutes_to_add : ℕ := 1450) : 
  (initial_time_in_minutes + minutes_to_add) % (24 * 60) = 370 :=
by
  -- Given (initial_time_in_minutes = 360 which is 6:00 a.m., minutes_to_add = 1450)
  -- Compute the time in minutes after 1450 minutes
  -- 24 hours = 1440 minutes, so (360 + 1450) % 1440 should equal 370
  sorry

end time_after_1450_minutes_l692_69247


namespace right_triangle_segments_l692_69282

open Real

theorem right_triangle_segments 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h_ab : a > b)
  (P Q : ℝ × ℝ) (P_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (Q_on_ellipse : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (Q_in_first_quad : Q.1 > 0 ∧ Q.2 > 0)
  (OQ_parallel_AP : ∃ k : ℝ, Q.1 = k * P.1 ∧ Q.2 = k * P.2)
  (M : ℝ × ℝ) (M_midpoint : M = ((P.1 + 0) / 2, (P.2 + 0) / 2))
  (R : ℝ × ℝ) (R_on_ellipse : R.1^2 / a^2 + R.2^2 / b^2 = 1)
  (OM_intersects_R : ∃ k : ℝ, R = (k * M.1, k * M.2))
: dist (0,0) Q ≠ 0 →
  dist (0,0) R ≠ 0 →
  dist (Q, R) ≠ 0 →
  dist (0,0) Q ^ 2 + dist (0,0) R ^ 2 = dist ((-a), (b)) ((a), (b)) ^ 2 :=
by
  sorry

end right_triangle_segments_l692_69282


namespace sunzi_classic_equation_l692_69263

theorem sunzi_classic_equation (x : ℕ) : 3 * (x - 2) = 2 * x + 9 :=
  sorry

end sunzi_classic_equation_l692_69263


namespace water_tank_full_capacity_l692_69241

theorem water_tank_full_capacity (x : ℝ) (h1 : x * (3/4) - x * (1/3) = 15) : x = 36 := 
by
  sorry

end water_tank_full_capacity_l692_69241


namespace option_C_is_neither_even_nor_odd_l692_69224

noncomputable def f_A (x : ℝ) : ℝ := x^2 + |x|
noncomputable def f_B (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 3^x
noncomputable def f_D (x : ℝ) : ℝ := 1/(x+1) + 1/(x-1)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

theorem option_C_is_neither_even_nor_odd : ¬ is_even f_C ∧ ¬ is_odd f_C :=
by
  sorry

end option_C_is_neither_even_nor_odd_l692_69224


namespace scientific_notation_of_87000000_l692_69208

theorem scientific_notation_of_87000000 :
  87000000 = 8.7 * 10^7 := 
sorry

end scientific_notation_of_87000000_l692_69208


namespace stream_current_speed_l692_69221

theorem stream_current_speed (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (1.5 * r + w) + 2 = 18 / (1.5 * r - w)) : w = 2.5 :=
by
  -- Translate the equations from the problem conditions directly.
  sorry

end stream_current_speed_l692_69221


namespace inequality_true_l692_69202

variables {a b : ℝ}
variables (c : ℝ)

theorem inequality_true (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 :=
by sorry

end inequality_true_l692_69202


namespace radius_of_circle_l692_69225

theorem radius_of_circle
  (r : ℝ)
  (h1 : ∀ x : ℝ, (x^2 + r = x) → (x^2 - x + r = 0) → ((-1)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
sorry

end radius_of_circle_l692_69225


namespace more_oaks_than_willows_l692_69261

theorem more_oaks_than_willows (total_trees willows : ℕ) (h1 : total_trees = 83) (h2 : willows = 36) :
  (total_trees - willows) - willows = 11 :=
by
  sorry

end more_oaks_than_willows_l692_69261


namespace polynomial_form_l692_69260

theorem polynomial_form (P : Polynomial ℝ) (hP : P ≠ 0)
    (h : ∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x)) :
    ∃ k : ℕ, k > 0 ∧ P = (X^2 + 1) ^ k :=
by sorry

end polynomial_form_l692_69260


namespace minimize_material_use_l692_69222

theorem minimize_material_use 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (total_area : x * y + (x^2 / 4) = 8) :
  (abs (x - 2.343) ≤ 0.001) ∧ (abs (y - 2.828) ≤ 0.001) :=
sorry

end minimize_material_use_l692_69222


namespace larger_integer_of_two_integers_diff_8_prod_120_l692_69286

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end larger_integer_of_two_integers_diff_8_prod_120_l692_69286


namespace find_age_of_30th_student_l692_69278

theorem find_age_of_30th_student :
  let avg1 := 23.5
  let n1 := 30
  let avg2 := 21.3
  let n2 := 9
  let avg3 := 19.7
  let n3 := 12
  let avg4 := 24.2
  let n4 := 7
  let avg5 := 35
  let n5 := 1
  let total_age_30 := n1 * avg1
  let total_age_9 := n2 * avg2
  let total_age_12 := n3 * avg3
  let total_age_7 := n4 * avg4
  let total_age_1 := n5 * avg5
  let total_age_29 := total_age_9 + total_age_12 + total_age_7 + total_age_1
  let age_30th := total_age_30 - total_age_29
  age_30th = 72.5 :=
by
  sorry

end find_age_of_30th_student_l692_69278


namespace range_of_largest_root_l692_69214

theorem range_of_largest_root :
  ∀ (a_2 a_1 a_0 : ℝ), 
  (|a_2| ≤ 1 ∧ |a_1| ≤ 1 ∧ |a_0| ≤ 1) ∧ (a_2 + a_1 + a_0 = 0) →
  (∃ s > 1, ∀ x > 0, x^3 + 3*a_2*x^2 + 5*a_1*x + a_0 = 0 → x ≤ s) ∧
  (s < 2) :=
by sorry

end range_of_largest_root_l692_69214


namespace round_24_6375_to_nearest_tenth_l692_69296

def round_to_nearest_tenth (n : ℚ) : ℚ :=
  let tenths := (n * 10).floor / 10
  let hundredths := (n * 100).floor % 10
  if hundredths < 5 then tenths else (tenths + 0.1)

theorem round_24_6375_to_nearest_tenth :
  round_to_nearest_tenth 24.6375 = 24.6 :=
by
  sorry

end round_24_6375_to_nearest_tenth_l692_69296


namespace cos_triple_angle_l692_69235

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end cos_triple_angle_l692_69235


namespace original_number_is_17_l692_69299

theorem original_number_is_17 (x : ℤ) (h : (x + 6) % 23 = 0) : x = 17 :=
sorry

end original_number_is_17_l692_69299


namespace integer_cube_less_than_triple_l692_69242

theorem integer_cube_less_than_triple (x : ℤ) : x^3 < 3 * x ↔ x = 0 :=
by 
  sorry

end integer_cube_less_than_triple_l692_69242


namespace necessary_but_not_sufficient_condition_l692_69297

theorem necessary_but_not_sufficient_condition (x : ℝ) : |x - 1| < 2 → -3 < x ∧ x < 3 :=
by
  sorry

end necessary_but_not_sufficient_condition_l692_69297


namespace gretchen_rachelle_ratio_l692_69249

-- Definitions of the conditions
def rachelle_pennies : ℕ := 180
def total_pennies : ℕ := 300
def rocky_pennies (gretchen_pennies : ℕ) : ℕ := gretchen_pennies / 3

-- The Lean 4 theorem statement
theorem gretchen_rachelle_ratio (gretchen_pennies : ℕ) 
    (h_total : rachelle_pennies + gretchen_pennies + rocky_pennies gretchen_pennies = total_pennies) :
    (gretchen_pennies : ℚ) / rachelle_pennies = 1 / 2 :=
sorry

end gretchen_rachelle_ratio_l692_69249


namespace no_solutions_Y_l692_69292

theorem no_solutions_Y (Y : ℕ) : 2 * Y + Y + 3 * Y = 14 ↔ false :=
by 
  sorry

end no_solutions_Y_l692_69292


namespace tennis_tournament_l692_69283

theorem tennis_tournament (n x : ℕ) 
    (p : ℕ := 4 * n) 
    (m : ℕ := (p * (p - 1)) / 2) 
    (r_women : ℕ := 3 * x) 
    (r_men : ℕ := 2 * x) 
    (total_wins : ℕ := r_women + r_men) 
    (h_matches : m = total_wins) 
    (h_ratio : r_women = 3 * x ∧ r_men = 2 * x ∧ 4 * n * (4 * n - 1) = 10 * x): 
    n = 4 :=
by
  sorry

end tennis_tournament_l692_69283


namespace product_of_sum_and_reciprocal_ge_four_l692_69280

theorem product_of_sum_and_reciprocal_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
sorry

end product_of_sum_and_reciprocal_ge_four_l692_69280


namespace a5_equals_2_l692_69248

variable {a : ℕ → ℝ}  -- a_n represents the nth term of the arithmetic sequence

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a 1 + n * d 

-- Given condition
axiom arithmetic_condition (h : is_arithmetic_sequence a) : a 1 + a 5 + a 9 = 6

-- The goal is to prove a_5 = 2
theorem a5_equals_2 (h : is_arithmetic_sequence a) (h_cond : a 1 + a 5 + a 9 = 6) : a 5 = 2 := 
by 
  sorry

end a5_equals_2_l692_69248


namespace singles_percentage_l692_69257

-- Definitions based on conditions
def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 7
def non_single_hits : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_single_hits

-- Theorem based on the proof problem
theorem singles_percentage :
  singles = 38 ∧ (singles / total_hits : ℚ) * 100 = 76 := 
  by
    sorry

end singles_percentage_l692_69257


namespace cristine_initial_lemons_l692_69244

theorem cristine_initial_lemons (L : ℕ) (h : (3 / 4 : ℚ) * L = 9) : L = 12 :=
sorry

end cristine_initial_lemons_l692_69244


namespace sum_of_all_possible_values_of_g10_l692_69205

noncomputable def g : ℕ → ℝ := sorry

axiom h1 : g 1 = 2
axiom h2 : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = 3 * (g m + g n)
axiom h3 : g 0 = 0

theorem sum_of_all_possible_values_of_g10 : g 10 = 59028 :=
by
  sorry

end sum_of_all_possible_values_of_g10_l692_69205


namespace find_missing_number_l692_69293

def average (l : List ℕ) : ℚ := l.sum / l.length

theorem find_missing_number : 
  ∃ x : ℕ, 
    average [744, 745, 747, 748, 749, 752, 752, 753, 755, x] = 750 :=
sorry

end find_missing_number_l692_69293


namespace total_population_l692_69289

theorem total_population (n : ℕ) (avg_population : ℕ) (h1 : n = 20) (h2 : avg_population = 4750) :
  n * avg_population = 95000 := by
  subst_vars
  sorry

end total_population_l692_69289


namespace rectangle_area_12_l692_69252

theorem rectangle_area_12
  (L W : ℝ)
  (h1 : L + W = 7)
  (h2 : L^2 + W^2 = 25) :
  L * W = 12 :=
by
  sorry

end rectangle_area_12_l692_69252


namespace proof_probability_second_science_given_first_arts_l692_69246

noncomputable def probability_second_science_given_first_arts : ℚ :=
  let total_questions := 5
  let science_questions := 3
  let arts_questions := 2

  -- Event A: drawing an arts question in the first draw.
  let P_A := arts_questions / total_questions

  -- Event AB: drawing an arts question in the first draw and a science question in the second draw.
  let P_AB := (arts_questions / total_questions) * (science_questions / (total_questions - 1))

  -- Conditional probability P(B|A): drawing a science question in the second draw given drawing an arts question in the first draw.
  P_AB / P_A

theorem proof_probability_second_science_given_first_arts :
  probability_second_science_given_first_arts = 3 / 4 :=
by
  -- Lean does not include the proof in the statement as required.
  sorry

end proof_probability_second_science_given_first_arts_l692_69246


namespace value_of_business_l692_69212

-- Defining the conditions
def owns_shares : ℚ := 2/3
def sold_fraction : ℚ := 3/4 
def sold_amount : ℝ := 75000 

-- The final proof statement
theorem value_of_business : 
  (owns_shares * sold_fraction) * value = sold_amount →
  value = 150000 :=
by
  sorry

end value_of_business_l692_69212


namespace factorization_correct_l692_69277

theorem factorization_correct :
  (∀ x : ℝ, x^2 - 6*x + 9 = (x - 3)^2) :=
by
  sorry

end factorization_correct_l692_69277


namespace find_f_13_l692_69266

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f (x + f x) = 3 * f x
axiom f_of_1 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
by
  have hf := f_property
  have hf1 := f_of_1
  sorry

end find_f_13_l692_69266


namespace magnitude_of_linear_combination_is_sqrt_65_l692_69265

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, 3 * m - 2)
noncomputable def perpendicular (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 = 0)

theorem magnitude_of_linear_combination_is_sqrt_65 (m : ℝ) 
  (h_perpendicular : perpendicular (vector_a m) (vector_b m)) : 
  ‖((2 : ℝ) • (vector_a 1) - (3 : ℝ) • (vector_b 1))‖ = Real.sqrt 65 := 
by
  sorry

end magnitude_of_linear_combination_is_sqrt_65_l692_69265


namespace alex_buys_15_pounds_of_rice_l692_69256

theorem alex_buys_15_pounds_of_rice (r b : ℝ) 
  (h1 : r + b = 30)
  (h2 : 75 * r + 35 * b = 1650) : 
  r = 15.0 := sorry

end alex_buys_15_pounds_of_rice_l692_69256


namespace greatest_mondays_in_45_days_l692_69238

-- Define the days in a week
def days_in_week : ℕ := 7

-- Define the total days being considered
def total_days : ℕ := 45

-- Calculate the complete weeks in the total days
def complete_weeks : ℕ := total_days / days_in_week

-- Calculate the extra days
def extra_days : ℕ := total_days % days_in_week

-- Define that the period starts on Monday (condition)
def starts_on_monday : Bool := true

-- Prove that the greatest number of Mondays in the first 45 days is 7
theorem greatest_mondays_in_45_days (h1 : days_in_week = 7) (h2 : total_days = 45) (h3 : starts_on_monday = true) : 
  (complete_weeks + if starts_on_monday && extra_days >= 1 then 1 else 0) = 7 := 
by
  sorry

end greatest_mondays_in_45_days_l692_69238


namespace greatest_divisor_450_90_l692_69290

open Nat

-- Define a condition for the set of divisors of given numbers which are less than a certain number.
def is_divisor (a : ℕ) (b : ℕ) : Prop := b % a = 0

def is_greatest_divisor (d : ℕ) (n : ℕ) (m : ℕ) (k : ℕ) : Prop :=
  is_divisor m d ∧ d < k ∧ ∀ (x : ℕ), x < k → is_divisor m x → x ≤ d

-- Define the proof problem.
theorem greatest_divisor_450_90 : is_greatest_divisor 18 450 90 30 := 
by
  sorry

end greatest_divisor_450_90_l692_69290


namespace markup_percentage_l692_69227

variable (W R : ℝ) -- W for Wholesale Cost, R for Retail Cost

-- Conditions:
-- 1. The sweater is sold at a 40% discount.
-- 2. When sold at a 40% discount, the merchant nets a 30% profit on the wholesale cost.
def discount_price (R : ℝ) : ℝ := 0.6 * R
def profit_price (W : ℝ) : ℝ := 1.3 * W

-- Hypotheses
axiom wholesale_cost_is_positive : W > 0
axiom discount_condition : discount_price R = profit_price W

-- Question: Prove that the percentage markup from wholesale to retail price is 116.67%.
theorem markup_percentage (W R : ℝ) 
  (wholesale_cost_is_positive : W > 0)
  (discount_condition : discount_price R = profit_price W) :
  ((R - W) / W * 100) = 116.67 := by
  sorry

end markup_percentage_l692_69227


namespace part1_l692_69230

theorem part1 (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 2 * b^2 = a^2 + c^2 :=
sorry

end part1_l692_69230


namespace evaluate_expression_l692_69262

noncomputable def log_4_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def log_8_16 : ℝ := Real.log 16 / Real.log 8

theorem evaluate_expression : Real.sqrt (log_4_8 * log_8_16) = Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l692_69262


namespace f_above_g_l692_69253

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

theorem f_above_g (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m > g x := 
sorry

end f_above_g_l692_69253


namespace selection_methods_count_l692_69279

-- Define the number of female students
def num_female_students : ℕ := 3

-- Define the number of male students
def num_male_students : ℕ := 2

-- Define the total number of different selection methods
def total_selection_methods : ℕ := num_female_students + num_male_students

-- Prove that the total number of different selection methods is 5
theorem selection_methods_count : total_selection_methods = 5 := by
  sorry

end selection_methods_count_l692_69279


namespace units_digit_product_l692_69217

theorem units_digit_product : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 := 
by
  sorry

end units_digit_product_l692_69217


namespace radius_condition_l692_69228

def X (x y : ℝ) : ℝ := 12 * x
def Y (x y : ℝ) : ℝ := 5 * y

def satisfies_condition (x y : ℝ) : Prop :=
  Real.sin (X x y + Y x y) = Real.sin (X x y) + Real.sin (Y x y)

def no_intersection (R : ℝ) : Prop :=
  ∀ (x y : ℝ), satisfies_condition x y → dist (0, 0) (x, y) ≥ R

theorem radius_condition :
  ∀ R : ℝ, (0 < R ∧ R < Real.pi / 15) →
  no_intersection R :=
sorry

end radius_condition_l692_69228


namespace sum_of_inner_segments_l692_69284

/-- Given the following conditions:
  1. The sum of the perimeters of the three quadrilaterals is 25 centimeters.
  2. The sum of the perimeters of the four triangles is 20 centimeters.
  3. The perimeter of triangle ABC is 19 centimeters.
Prove that AD + BE + CF = 13 centimeters. -/
theorem sum_of_inner_segments 
  (perimeter_quads : ℝ)
  (perimeter_tris : ℝ)
  (perimeter_ABC : ℝ)
  (hq : perimeter_quads = 25)
  (ht : perimeter_tris = 20)
  (hABC : perimeter_ABC = 19) 
  : AD + BE + CF = 13 :=
by
  sorry

end sum_of_inner_segments_l692_69284


namespace min_segments_required_l692_69276

noncomputable def min_segments (n : ℕ) : ℕ := (3 * n - 2 + 1) / 2

theorem min_segments_required (n : ℕ) (h : ∀ (A B : ℕ) (hA : A < n) (hB : B < n) (hAB : A ≠ B), 
  ∃ (C : ℕ), C < n ∧ (C ≠ A) ∧ (C ≠ B)) : 
  min_segments n = ⌈ (3 * n - 2 : ℝ) / 2 ⌉ := 
sorry

end min_segments_required_l692_69276


namespace divisible_by_9_l692_69200

theorem divisible_by_9 (k : ℕ) (h : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
sorry

end divisible_by_9_l692_69200


namespace initial_bananas_each_child_l692_69219

-- Define the variables and conditions.
def total_children : ℕ := 320
def absent_children : ℕ := 160
def present_children := total_children - absent_children
def extra_bananas : ℕ := 2

-- We are to prove the initial number of bananas each child was supposed to get.
theorem initial_bananas_each_child (B : ℕ) (x : ℕ) :
  B = total_children * x ∧ B = present_children * (x + extra_bananas) → x = 2 :=
by
  sorry

end initial_bananas_each_child_l692_69219


namespace total_wheels_l692_69220

-- Definitions of given conditions
def bicycles : ℕ := 50
def tricycles : ℕ := 20
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Theorem stating the total number of wheels for bicycles and tricycles combined
theorem total_wheels : bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160 :=
by
  sorry

end total_wheels_l692_69220


namespace find_common_ratio_l692_69232

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variable {a : ℕ → ℝ} {q : ℝ}

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : a 2 + a 4 = 20)
  (h3 : a 3 + a 5 = 40) : q = 2 :=
by
  sorry

end find_common_ratio_l692_69232


namespace problem_statement_l692_69213

noncomputable def a : ℝ := -0.5
noncomputable def b : ℝ := (1 + Real.sqrt 3) / 2

theorem problem_statement
  (h1 : a^2 = 9 / 36)
  (h2 : b^2 = (1 + Real.sqrt 3)^2 / 8)
  (h3 : a < 0)
  (h4 : b > 0) :
  ∃ (x y z : ℤ), (a - b)^2 = x * Real.sqrt y / z ∧ (x + y + z = 6) :=
sorry

end problem_statement_l692_69213


namespace remainder_product_l692_69245

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l692_69245
