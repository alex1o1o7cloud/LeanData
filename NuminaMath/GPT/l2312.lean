import Mathlib

namespace NUMINAMATH_GPT_determine_ts_l2312_231220

theorem determine_ts :
  ∃ t s : ℝ, 
  (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩) :=
by
  use 6, -9
  sorry

end NUMINAMATH_GPT_determine_ts_l2312_231220


namespace NUMINAMATH_GPT_females_watch_eq_seventy_five_l2312_231274

-- Definition of conditions
def males_watch : ℕ := 85
def females_dont_watch : ℕ := 120
def total_watch : ℕ := 160
def total_dont_watch : ℕ := 180

-- Definition of the proof problem
theorem females_watch_eq_seventy_five :
  total_watch - males_watch = 75 :=
by
  sorry

end NUMINAMATH_GPT_females_watch_eq_seventy_five_l2312_231274


namespace NUMINAMATH_GPT_parallel_slope_l2312_231234

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = -21) : 
    ∃ m : ℝ, m = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_slope_l2312_231234


namespace NUMINAMATH_GPT_binomial_probability_l2312_231254

theorem binomial_probability (n : ℕ) (p : ℝ) (h1 : (n * p = 300)) (h2 : (n * p * (1 - p) = 200)) :
    p = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_binomial_probability_l2312_231254


namespace NUMINAMATH_GPT_find_x_l2312_231278

theorem find_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y + 2) :
  x = (685 + 25 * Real.sqrt 745) / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2312_231278


namespace NUMINAMATH_GPT_heartsuit_example_l2312_231253

def heartsuit (x y: ℤ) : ℤ := 4 * x + 6 * y

theorem heartsuit_example : heartsuit 3 8 = 60 :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_example_l2312_231253


namespace NUMINAMATH_GPT_initial_number_of_men_l2312_231217

theorem initial_number_of_men
  (M : ℕ) (A : ℕ)
  (h1 : ∀ A_new : ℕ, A_new = A + 4)
  (h2 : ∀ total_age_increase : ℕ, total_age_increase = (2 * 52) - (36 + 32))
  (h3 : ∀ sum_age_men : ℕ, sum_age_men = M * A)
  (h4 : ∀ new_sum_age_men : ℕ, new_sum_age_men = sum_age_men + ((2 * 52) - (36 + 32))) :
  M = 9 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l2312_231217


namespace NUMINAMATH_GPT_chocolate_chip_cookies_count_l2312_231208

theorem chocolate_chip_cookies_count (h1 : 5 / 2 = 20 / (x : ℕ)) : x = 8 := 
by
  sorry -- Proof to be implemented

end NUMINAMATH_GPT_chocolate_chip_cookies_count_l2312_231208


namespace NUMINAMATH_GPT_students_answered_both_correctly_l2312_231267

theorem students_answered_both_correctly :
  ∀ (total_students set_problem function_problem both_incorrect x : ℕ),
    total_students = 50 → 
    set_problem = 40 →
    function_problem = 31 →
    both_incorrect = 4 →
    x = total_students - both_incorrect - (set_problem + function_problem - total_students) →
    x = 25 :=
by
  intros total_students set_problem function_problem both_incorrect x
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end NUMINAMATH_GPT_students_answered_both_correctly_l2312_231267


namespace NUMINAMATH_GPT_angie_total_taxes_l2312_231280

theorem angie_total_taxes:
  ∀ (salary : ℕ) (N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over : ℕ),
  salary = 80 →
  N_1 = 12 → T_1 = 8 → U_1 = 5 →
  N_2 = 15 → T_2 = 6 → U_2 = 7 →
  N_3 = 10 → T_3 = 9 → U_3 = 6 →
  N_4 = 14 → T_4 = 7 → U_4 = 4 →
  left_over = 18 →
  T_1 + T_2 + T_3 + T_4 = 30 :=
by
  intros salary N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over
  sorry

end NUMINAMATH_GPT_angie_total_taxes_l2312_231280


namespace NUMINAMATH_GPT_marathon_time_l2312_231231

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end NUMINAMATH_GPT_marathon_time_l2312_231231


namespace NUMINAMATH_GPT_quiz_competition_top_three_orders_l2312_231252

theorem quiz_competition_top_three_orders :
  let participants := 4
  let top_positions := 3
  let permutations := (Nat.factorial participants) / (Nat.factorial (participants - top_positions))
  permutations = 24 := 
by
  sorry

end NUMINAMATH_GPT_quiz_competition_top_three_orders_l2312_231252


namespace NUMINAMATH_GPT_pizza_cost_l2312_231275

theorem pizza_cost (soda_cost jeans_cost start_money quarters_left : ℝ) (quarters_value : ℝ) (total_left : ℝ) (pizza_cost : ℝ) :
  soda_cost = 1.50 → 
  jeans_cost = 11.50 → 
  start_money = 40 → 
  quarters_left = 97 → 
  quarters_value = 0.25 → 
  total_left = quarters_left * quarters_value → 
  pizza_cost = start_money - total_left - (soda_cost + jeans_cost) → 
  pizza_cost = 2.75 :=
by
  sorry

end NUMINAMATH_GPT_pizza_cost_l2312_231275


namespace NUMINAMATH_GPT_distance_B_to_center_l2312_231221

/-- Definitions for the geometrical scenario -/
structure NotchedCircleGeom where
  radius : ℝ
  A_pos : ℝ × ℝ
  B_pos : ℝ × ℝ
  C_pos : ℝ × ℝ
  AB_len : ℝ
  BC_len : ℝ
  angle_ABC_right : Prop
  
  -- Conditions derived from problem statement
  radius_eq_sqrt72 : radius = Real.sqrt 72
  AB_len_eq_8 : AB_len = 8
  BC_len_eq_3 : BC_len = 3
  angle_ABC_right_angle : angle_ABC_right
  
/-- Problem statement -/
theorem distance_B_to_center (geom : NotchedCircleGeom) :
  let x := geom.B_pos.1
  let y := geom.B_pos.2
  x^2 + y^2 = 50 :=
sorry

end NUMINAMATH_GPT_distance_B_to_center_l2312_231221


namespace NUMINAMATH_GPT_divisibility_of_powers_l2312_231243

theorem divisibility_of_powers (a b c d m : ℤ) (h_odd : m % 2 = 1)
  (h_sum_div : m ∣ (a + b + c + d))
  (h_sum_squares_div : m ∣ (a^2 + b^2 + c^2 + d^2)) : 
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d) :=
sorry

end NUMINAMATH_GPT_divisibility_of_powers_l2312_231243


namespace NUMINAMATH_GPT_crayons_per_child_l2312_231215

theorem crayons_per_child (children : ℕ) (total_crayons : ℕ) (h1 : children = 18) (h2 : total_crayons = 216) : 
    total_crayons / children = 12 := 
by
  sorry

end NUMINAMATH_GPT_crayons_per_child_l2312_231215


namespace NUMINAMATH_GPT_angle_sum_around_point_l2312_231249

theorem angle_sum_around_point (p q r s t : ℝ) (h : p + q + r + s + t = 360) : p = 360 - q - r - s - t :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_around_point_l2312_231249


namespace NUMINAMATH_GPT_analytical_expression_when_x_in_5_7_l2312_231269

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma symmetric_about_one (x : ℝ) : f (1 - x) = f (1 + x) := sorry
lemma values_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = x := sorry

theorem analytical_expression_when_x_in_5_7 (x : ℝ) (h : 5 < x ∧ x ≤ 7) :
  f x = 6 - x :=
sorry

end NUMINAMATH_GPT_analytical_expression_when_x_in_5_7_l2312_231269


namespace NUMINAMATH_GPT_find_seventh_term_l2312_231284

theorem find_seventh_term :
  ∃ r : ℚ, ∃ (a₁ a₇ a₁₀ : ℚ), 
    a₁ = 12 ∧ 
    a₁₀ = 78732 ∧ 
    a₇ = a₁ * r^6 ∧ 
    a₁₀ = a₁ * r^9 ∧ 
    a₇ = 8748 :=
by
  sorry

end NUMINAMATH_GPT_find_seventh_term_l2312_231284


namespace NUMINAMATH_GPT_shortest_side_of_triangle_l2312_231207

noncomputable def triangle_shortest_side_length (a b r : ℝ) (shortest : ℝ) : Prop :=
a = 8 ∧ b = 6 ∧ r = 4 ∧ shortest = 12

theorem shortest_side_of_triangle 
  (a b r shortest : ℝ) 
  (h : triangle_shortest_side_length a b r shortest) : shortest = 12 :=
sorry

end NUMINAMATH_GPT_shortest_side_of_triangle_l2312_231207


namespace NUMINAMATH_GPT_no_common_complex_roots_l2312_231296

theorem no_common_complex_roots (a b : ℚ) :
  ¬ ∃ α : ℂ, (α^5 - α - 1 = 0) ∧ (α^2 + a * α + b = 0) :=
sorry

end NUMINAMATH_GPT_no_common_complex_roots_l2312_231296


namespace NUMINAMATH_GPT_restaurant_cost_l2312_231229

theorem restaurant_cost (total_people kids adult_cost : ℕ)
  (h1 : total_people = 12)
  (h2 : kids = 7)
  (h3 : adult_cost = 3) :
  total_people - kids * adult_cost = 15 := by
  sorry

end NUMINAMATH_GPT_restaurant_cost_l2312_231229


namespace NUMINAMATH_GPT_cos_neg_75_eq_l2312_231200

noncomputable def cos_75_degrees : Real := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_neg_75_eq : Real.cos (-(75 * Real.pi / 180)) = cos_75_degrees := by
  sorry

end NUMINAMATH_GPT_cos_neg_75_eq_l2312_231200


namespace NUMINAMATH_GPT_inequality_proof_l2312_231240

theorem inequality_proof (b c : ℝ) (hb : 0 < b) (hc : 0 < c) :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ 
  (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2312_231240


namespace NUMINAMATH_GPT_helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l2312_231298

def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

theorem helicopter_A_highest_altitude :
  List.maximum heights_A = some 3.6 :=
by sorry

theorem helicopter_A_final_altitude :
  List.sum heights_A = 3.4 :=
by sorry

theorem helicopter_B_5th_performance :
  ∃ (x : ℝ), List.sum heights_B + x = 3.4 ∧ x = -0.2 :=
by sorry

end NUMINAMATH_GPT_helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l2312_231298


namespace NUMINAMATH_GPT_unique_solution_a_eq_4_l2312_231285

theorem unique_solution_a_eq_4 (a : ℝ) (h : ∀ x1 x2 : ℝ, (a * x1^2 + a * x1 + 1 = 0 ∧ a * x2^2 + a * x2 + 1 = 0) → x1 = x2) : a = 4 :=
sorry

end NUMINAMATH_GPT_unique_solution_a_eq_4_l2312_231285


namespace NUMINAMATH_GPT_equal_distribution_arithmetic_sequence_l2312_231233

theorem equal_distribution_arithmetic_sequence :
  ∃ a d : ℚ, (a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d)) ∧
  5 * a = 5 ∧
  a + 2 * d = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_equal_distribution_arithmetic_sequence_l2312_231233


namespace NUMINAMATH_GPT_smallest_k_for_Δk_un_zero_l2312_231273

def u (n : ℕ) : ℤ := n^3 - n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0     => u
  | (k+1) => λ n => Δ k u (n+1) - Δ k u n

theorem smallest_k_for_Δk_un_zero (u : ℕ → ℤ) (h : ∀ n, u n = n^3 - n) :
  ∀ n, Δ 4 u n = 0 ∧ (∀ k < 4, ∃ n, Δ k u n ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_Δk_un_zero_l2312_231273


namespace NUMINAMATH_GPT_least_possible_value_a2008_l2312_231225

theorem least_possible_value_a2008 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a n < a (n + 1)) 
  (h2 : ∀ i j k l, 1 ≤ i → i < j → j ≤ k → k < l → i + l = j + k → a i + a l > a j + a k)
  : a 2008 ≥ 2015029 :=
sorry

end NUMINAMATH_GPT_least_possible_value_a2008_l2312_231225


namespace NUMINAMATH_GPT_average_age_of_coaches_l2312_231203

theorem average_age_of_coaches 
  (total_members : ℕ) (average_age_members : ℕ)
  (num_girls : ℕ) (average_age_girls : ℕ)
  (num_boys : ℕ) (average_age_boys : ℕ)
  (num_coaches : ℕ) :
  total_members = 30 →
  average_age_members = 20 →
  num_girls = 10 →
  average_age_girls = 18 →
  num_boys = 15 →
  average_age_boys = 19 →
  num_coaches = 5 →
  (600 - (num_girls * average_age_girls) - (num_boys * average_age_boys)) / num_coaches = 27 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_age_of_coaches_l2312_231203


namespace NUMINAMATH_GPT_pine_trees_multiple_of_27_l2312_231255

noncomputable def numberOfPineTrees (n : ℕ) : ℕ := 27 * n

theorem pine_trees_multiple_of_27 (oak_trees : ℕ) (max_trees_per_row : ℕ) (rows_of_oak : ℕ) :
  oak_trees = 54 → max_trees_per_row = 27 → rows_of_oak = oak_trees / max_trees_per_row →
  ∃ n : ℕ, numberOfPineTrees n = 27 * n :=
by
  intros
  use (oak_trees - rows_of_oak * max_trees_per_row) / 27
  sorry

end NUMINAMATH_GPT_pine_trees_multiple_of_27_l2312_231255


namespace NUMINAMATH_GPT_john_coffees_per_day_l2312_231261

theorem john_coffees_per_day (x : ℕ)
  (h1 : ∀ p : ℕ, p = 2)
  (h2 : ∀ p : ℕ, p = p + p / 2)
  (h3 : ∀ n : ℕ, n = x / 2)
  (h4 : ∀ d : ℕ, 2 * x - 3 * (x / 2) = 2) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_john_coffees_per_day_l2312_231261


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2312_231270

variable (B S : ℝ)

def downstream_speed := 10
def upstream_speed := 4

theorem boat_speed_in_still_water :
  B + S = downstream_speed → 
  B - S = upstream_speed → 
  B = 7 :=
by
  intros h₁ h₂
  -- We would insert the proof steps here
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2312_231270


namespace NUMINAMATH_GPT_total_games_in_single_elimination_tournament_l2312_231210

def single_elimination_tournament_games (teams : ℕ) : ℕ :=
teams - 1

theorem total_games_in_single_elimination_tournament :
  single_elimination_tournament_games 23 = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_single_elimination_tournament_l2312_231210


namespace NUMINAMATH_GPT_distinct_integers_sum_l2312_231276

theorem distinct_integers_sum (n : ℕ) (h : n > 3) (a : Fin n → ℤ)
  (h1 : ∀ i, 1 ≤ a i) (h2 : ∀ i j, i < j → a i < a j) (h3 : ∀ i, a i ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
  k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ a i + a j = a k + a l ∧ a k + a l = a m :=
by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_l2312_231276


namespace NUMINAMATH_GPT_cos_seven_pi_six_eq_neg_sqrt_three_div_two_l2312_231235

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_cos_seven_pi_six_eq_neg_sqrt_three_div_two_l2312_231235


namespace NUMINAMATH_GPT_groupB_is_conditional_control_l2312_231258

-- Definitions based on conditions
def groupA_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea"}
def groupB_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea", "nitrate"}

-- The property that defines a conditional control in this context.
def conditional_control (control_sources : Set String) (experimental_sources : Set String) : Prop :=
  control_sources ≠ experimental_sources ∧ "urea" ∈ control_sources ∧ "nitrate" ∈ experimental_sources

-- Prove that Group B's experiment forms a conditional control
theorem groupB_is_conditional_control :
  ∃ nitrogen_sourcesA nitrogen_sourcesB, groupA_medium nitrogen_sourcesA ∧ groupB_medium nitrogen_sourcesB ∧
  conditional_control nitrogen_sourcesA nitrogen_sourcesB :=
by
  sorry

end NUMINAMATH_GPT_groupB_is_conditional_control_l2312_231258


namespace NUMINAMATH_GPT_find_a_l2312_231245

theorem find_a (a b c : ℤ) (h_vertex : ∀ x, (x - 2)*(x - 2) * a + 3 = a*x*x + b*x + c) 
  (h_point : (a*(3 - 2)*(3 -2) + 3 = 6)) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2312_231245


namespace NUMINAMATH_GPT_max_ab_perpendicular_l2312_231299

theorem max_ab_perpendicular (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + b = 3) : ab <= (9 / 8) := 
sorry

end NUMINAMATH_GPT_max_ab_perpendicular_l2312_231299


namespace NUMINAMATH_GPT_area_shaded_region_is_75_l2312_231289

-- Define the side length of the larger square
def side_length_large_square : ℝ := 10

-- Define the side length of the smaller square
def side_length_small_square : ℝ := 5

-- Define the area of the larger square
def area_large_square : ℝ := side_length_large_square ^ 2

-- Define the area of the smaller square
def area_small_square : ℝ := side_length_small_square ^ 2

-- Define the area of the shaded region
def area_shaded_region : ℝ := area_large_square - area_small_square

-- The theorem that states the area of the shaded region is 75 square units
theorem area_shaded_region_is_75 : area_shaded_region = 75 := by
  -- The proof will be filled in here when required
  sorry

end NUMINAMATH_GPT_area_shaded_region_is_75_l2312_231289


namespace NUMINAMATH_GPT_condition_of_inequality_l2312_231263

theorem condition_of_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2 * (x + y - 1)) : x = 1 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_condition_of_inequality_l2312_231263


namespace NUMINAMATH_GPT_trapezium_distance_l2312_231214

variable (a b h : ℝ)

theorem trapezium_distance (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b)
  (area_eq : 270 = 1/2 * (a + b) * h) (a_eq : a = 20) (b_eq : b = 16) : h = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_trapezium_distance_l2312_231214


namespace NUMINAMATH_GPT_fg_3_eq_7_l2312_231230

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 2) ^ 2

theorem fg_3_eq_7 : f (g 3) = 7 :=
by
  sorry

end NUMINAMATH_GPT_fg_3_eq_7_l2312_231230


namespace NUMINAMATH_GPT_find_x_plus_y_l2312_231266

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l2312_231266


namespace NUMINAMATH_GPT_remainder_x2023_plus_1_l2312_231292

noncomputable def remainder (a b : Polynomial ℂ) : Polynomial ℂ :=
a % b

theorem remainder_x2023_plus_1 :
  remainder (Polynomial.X ^ 2023 + 1) (Polynomial.X ^ 8 - Polynomial.X ^ 6 + Polynomial.X ^ 4 - Polynomial.X ^ 2 + 1) =
  - Polynomial.X ^ 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_x2023_plus_1_l2312_231292


namespace NUMINAMATH_GPT_frogs_count_l2312_231224

variables (Alex Brian Chris LeRoy Mike : Type) 

-- Definitions for the species
def toad (x : Type) : Prop := ∃ p : Prop, p -- Dummy definition for toads
def frog (x : Type) : Prop := ∃ p : Prop, ¬p -- Dummy definition for frogs

-- Conditions
axiom Alex_statement : (toad Alex) → (∃ x : ℕ, x = 3) ∧ (frog Alex) → (¬(∃ x : ℕ, x = 3))
axiom Brian_statement : (toad Brian) → (toad Mike) ∧ (frog Brian) → (frog Mike)
axiom Chris_statement : (toad Chris) → (toad LeRoy) ∧ (frog Chris) → (frog LeRoy)
axiom LeRoy_statement : (toad LeRoy) → (toad Chris) ∧ (frog LeRoy) → (frog Chris)
axiom Mike_statement : (toad Mike) → (∃ x : ℕ, x < 3) ∧ (frog Mike) → (¬(∃ x : ℕ, x < 3))

theorem frogs_count (total : ℕ) : total = 5 → 
  (∃ frog_count : ℕ, frog_count = 2) :=
by
  -- Leaving the proof as a sorry placeholder
  sorry

end NUMINAMATH_GPT_frogs_count_l2312_231224


namespace NUMINAMATH_GPT_total_investment_is_correct_l2312_231216

-- Define principal, rate, and number of years
def principal : ℝ := 8000
def rate : ℝ := 0.04
def years : ℕ := 10

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem total_investment_is_correct :
  compound_interest principal rate years = 11842 :=
by
  sorry

end NUMINAMATH_GPT_total_investment_is_correct_l2312_231216


namespace NUMINAMATH_GPT_element_with_36_36_percentage_is_O_l2312_231247

-- Define the chemical formula N2O and atomic masses
def chemical_formula : String := "N2O"
def atomic_mass_N : Float := 14.01
def atomic_mass_O : Float := 16.00

-- Define the molar mass of N2O
def molar_mass_N2O : Float := (2 * atomic_mass_N) + (1 * atomic_mass_O)

-- Mass of nitrogen in N2O
def mass_N_in_N2O : Float := 2 * atomic_mass_N

-- Mass of oxygen in N2O
def mass_O_in_N2O : Float := 1 * atomic_mass_O

-- Mass percentages
def mass_percentage_N : Float := (mass_N_in_N2O / molar_mass_N2O) * 100
def mass_percentage_O : Float := (mass_O_in_N2O / molar_mass_N2O) * 100

-- Prove that the element with a mass percentage of 36.36% is oxygen
theorem element_with_36_36_percentage_is_O : mass_percentage_O = 36.36 := sorry

end NUMINAMATH_GPT_element_with_36_36_percentage_is_O_l2312_231247


namespace NUMINAMATH_GPT_max_value_of_sum_l2312_231287

theorem max_value_of_sum (x y z : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (eq : x^2 + y^2 + z^2 + x + 2*y + 3*z = (13 : ℝ) / 4) : x + y + z ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_sum_l2312_231287


namespace NUMINAMATH_GPT_max_identifiable_cards_2013_l2312_231219

-- Define the number of cards
def num_cards : ℕ := 2013

-- Define the function that determines the maximum t for which the numbers can be found
def max_identifiable_cards (cards : ℕ) (select : ℕ) : ℕ :=
  if (cards = 2013) ∧ (select = 10) then 1986 else 0

-- The theorem to prove the property
theorem max_identifiable_cards_2013 :
  max_identifiable_cards 2013 10 = 1986 :=
sorry

end NUMINAMATH_GPT_max_identifiable_cards_2013_l2312_231219


namespace NUMINAMATH_GPT_problem_l2312_231272

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end NUMINAMATH_GPT_problem_l2312_231272


namespace NUMINAMATH_GPT_segment_radius_with_inscribed_equilateral_triangle_l2312_231213

theorem segment_radius_with_inscribed_equilateral_triangle (α h : ℝ) : 
  ∃ x : ℝ, x = (h / (Real.sin (α / 2))^2) * (Real.cos (α / 2) + Real.sqrt (1 + (1 / 3) * (Real.sin (α / 2))^2)) :=
sorry

end NUMINAMATH_GPT_segment_radius_with_inscribed_equilateral_triangle_l2312_231213


namespace NUMINAMATH_GPT_velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l2312_231222

noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)
noncomputable def v (A ω t : ℝ) : ℝ := deriv (x A ω) t
noncomputable def α (A ω t : ℝ) : ℝ := deriv (v A ω) t

theorem velocity_at_specific_time (A ω : ℝ) : 
  v A ω (2 * Real.pi / ω) = A * ω := 
sorry

theorem acceleration_at_specific_time (A ω : ℝ) :
  α A ω (2 * Real.pi / ω) = 0 :=
sorry

theorem acceleration_proportional_to_displacement (A ω t : ℝ) :
  α A ω t = -ω^2 * x A ω t :=
sorry

end NUMINAMATH_GPT_velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l2312_231222


namespace NUMINAMATH_GPT_triangle_height_dist_inequality_l2312_231209

variable {T : Type} [MetricSpace T] 

theorem triangle_height_dist_inequality {h_a h_b h_c l_a l_b l_c : ℝ} (h_a_pos : 0 < h_a) (h_b_pos : 0 < h_b) (h_c_pos : 0 < h_c) 
  (l_a_pos : 0 < l_a) (l_b_pos : 0 < l_b) (l_c_pos : 0 < l_c) :
  h_a / l_a + h_b / l_b + h_c / l_c >= 9 :=
sorry

end NUMINAMATH_GPT_triangle_height_dist_inequality_l2312_231209


namespace NUMINAMATH_GPT_intersection_A_B_l2312_231228

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end NUMINAMATH_GPT_intersection_A_B_l2312_231228


namespace NUMINAMATH_GPT_smallest_number_plus_3_divisible_by_18_70_100_21_l2312_231288

/-- 
The smallest number such that when increased by 3 is divisible by 18, 70, 100, and 21.
-/
theorem smallest_number_plus_3_divisible_by_18_70_100_21 : 
  ∃ n : ℕ, (∃ k : ℕ, n + 3 = k * 18) ∧ (∃ l : ℕ, n + 3 = l * 70) ∧ (∃ m : ℕ, n + 3 = m * 100) ∧ (∃ o : ℕ, n + 3 = o * 21) ∧ n = 6297 :=
sorry

end NUMINAMATH_GPT_smallest_number_plus_3_divisible_by_18_70_100_21_l2312_231288


namespace NUMINAMATH_GPT_smallest_palindrome_in_bases_2_and_4_l2312_231238

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let repr := n.digits base
  repr = repr.reverse

theorem smallest_palindrome_in_bases_2_and_4 (x : ℕ) :
  (x > 15) ∧ is_palindrome x 2 ∧ is_palindrome x 4 → x = 17 :=
by
  sorry

end NUMINAMATH_GPT_smallest_palindrome_in_bases_2_and_4_l2312_231238


namespace NUMINAMATH_GPT_angle_A_value_cos_A_minus_2x_value_l2312_231279

open Real

-- Let A, B, and C be the internal angles of triangle ABC.
variable {A B C x : ℝ}

-- Given conditions
axiom triangle_angles : A + B + C = π
axiom sinC_eq_2sinAminusB : sin C = 2 * sin (A - B)
axiom B_is_pi_over_6 : B = π / 6
axiom cosAplusx_is_neg_third : cos (A + x) = -1 / 3

-- Proof goals
theorem angle_A_value : A = π / 3 := by sorry

theorem cos_A_minus_2x_value : cos (A - 2 * x) = 7 / 9 := by sorry

end NUMINAMATH_GPT_angle_A_value_cos_A_minus_2x_value_l2312_231279


namespace NUMINAMATH_GPT_value_of_a_minus_2b_l2312_231212

theorem value_of_a_minus_2b 
  (a b : ℚ) 
  (h : ∀ y : ℚ, y > 0 → y ≠ 2 → y ≠ -3 → (a / (y-2) + b / (y+3) = (2 * y + 5) / ((y-2)*(y+3)))) 
  : a - 2 * b = 7 / 5 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_2b_l2312_231212


namespace NUMINAMATH_GPT_initial_money_l2312_231223

-- Let M represent the initial amount of money Mrs. Hilt had.
variable (M : ℕ)

-- Condition 1: Mrs. Hilt bought a pencil for 11 cents.
def pencil_cost : ℕ := 11

-- Condition 2: She had 4 cents left after buying the pencil.
def amount_left : ℕ := 4

-- Proof problem statement: Prove that M = 15 given the above conditions.
theorem initial_money (h : M = pencil_cost + amount_left) : M = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_money_l2312_231223


namespace NUMINAMATH_GPT_Carol_rectangle_length_l2312_231271

theorem Carol_rectangle_length :
  (∃ (L : ℕ), (L * 15 = 4 * 30) → L = 8) :=
by
  sorry

end NUMINAMATH_GPT_Carol_rectangle_length_l2312_231271


namespace NUMINAMATH_GPT_middle_part_proportional_l2312_231211

theorem middle_part_proportional (x : ℚ) (s : ℚ) (h : s = 120) 
    (proportional : (2 * x) + (1/2 * x) + (1/4 * x) = s) : 
    (1/2 * x) = 240/11 := 
by
  sorry

end NUMINAMATH_GPT_middle_part_proportional_l2312_231211


namespace NUMINAMATH_GPT_remainder_is_15_l2312_231290

-- Definitions based on conditions
def S : ℕ := 476
def L : ℕ := S + 2395
def quotient : ℕ := 6

-- The proof statement
theorem remainder_is_15 : ∃ R : ℕ, L = quotient * S + R ∧ R = 15 := by
  sorry

end NUMINAMATH_GPT_remainder_is_15_l2312_231290


namespace NUMINAMATH_GPT_payment_for_30_kilograms_l2312_231282

-- Define the price calculation based on quantity x
def payment_amount (x : ℕ) : ℕ :=
  if x ≤ 10 then 20 * x
  else 16 * x + 40

-- Prove that for x = 30, the payment amount y equals 520
theorem payment_for_30_kilograms : payment_amount 30 = 520 := by
  sorry

end NUMINAMATH_GPT_payment_for_30_kilograms_l2312_231282


namespace NUMINAMATH_GPT_ellipse_ratio_sum_l2312_231297

theorem ellipse_ratio_sum :
  (∃ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0) →
  (∃ a b : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0 → 
    (y = a * x ∨ y = b * x)) ∧ (a + b = 9)) :=
  sorry

end NUMINAMATH_GPT_ellipse_ratio_sum_l2312_231297


namespace NUMINAMATH_GPT_proof_problem_l2312_231204

def RealSets (A B : Set ℝ) : Set ℝ :=
let complementA := {x | -2 < x ∧ x < 3}
let unionAB := complementA ∪ B
unionAB

theorem proof_problem :
  let A := {x : ℝ | (x + 2) * (x - 3) ≥ 0}
  let B := {x : ℝ | x > 1}
  let complementA := {x : ℝ | -2 < x ∧ x < 3}
  let unionAB := complementA ∪ B
  unionAB = {x : ℝ | x > -2} :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2312_231204


namespace NUMINAMATH_GPT_find_m_range_l2312_231205

theorem find_m_range (m : ℝ) : 
  (∃ x : ℤ, 2 * (x : ℝ) - 1 ≤ 5 ∧ x - 1 ≥ m ∧ x ≤ 3) ∧ 
  (∃ y : ℤ, 2 * (y : ℝ) - 1 ≤ 5 ∧ y - 1 ≥ m ∧ y ≤ 3 ∧ x ≠ y) → 
  -1 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_GPT_find_m_range_l2312_231205


namespace NUMINAMATH_GPT_find_c1_minus_c2_l2312_231227

-- Define the conditions of the problem
variables (c1 c2 : ℝ)
variables (x y : ℝ)
variables (h1 : (2 : ℝ) * x + 3 * y = c1)
variables (h2 : (3 : ℝ) * x + 2 * y = c2)
variables (sol_x : x = 2)
variables (sol_y : y = 1)

-- Define the theorem to be proven
theorem find_c1_minus_c2 : c1 - c2 = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_c1_minus_c2_l2312_231227


namespace NUMINAMATH_GPT_sum_of_first_eight_terms_l2312_231202

theorem sum_of_first_eight_terms (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * (1 + 2 + 4 + 8) = 1) :
  a * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_sum_of_first_eight_terms_l2312_231202


namespace NUMINAMATH_GPT_five_points_distance_ratio_ge_two_sin_54_l2312_231294

theorem five_points_distance_ratio_ge_two_sin_54
  (points : Fin 5 → ℝ × ℝ)
  (distinct : Function.Injective points) :
  let distances := {d : ℝ | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (points i) (points j)}
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * Real.pi / 180) := by
  sorry

end NUMINAMATH_GPT_five_points_distance_ratio_ge_two_sin_54_l2312_231294


namespace NUMINAMATH_GPT_func4_same_domain_range_as_func1_l2312_231236

noncomputable def func1_domain : Set ℝ := {x | 0 < x}
noncomputable def func1_range : Set ℝ := {y | 0 < y}

noncomputable def func4_domain : Set ℝ := {x | 0 < x}
noncomputable def func4_range : Set ℝ := {y | 0 < y}

theorem func4_same_domain_range_as_func1 :
  (func4_domain = func1_domain) ∧ (func4_range = func1_range) :=
sorry

end NUMINAMATH_GPT_func4_same_domain_range_as_func1_l2312_231236


namespace NUMINAMATH_GPT_solve_eqn_l2312_231218

theorem solve_eqn (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ 6) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31 / 11 :=
by
sorry

end NUMINAMATH_GPT_solve_eqn_l2312_231218


namespace NUMINAMATH_GPT_jafaris_candy_l2312_231246

-- Define the conditions
variable (candy_total : Nat)
variable (taquon_candy : Nat)
variable (mack_candy : Nat)

-- Assume the conditions from the problem
axiom candy_total_def : candy_total = 418
axiom taquon_candy_def : taquon_candy = 171
axiom mack_candy_def : mack_candy = 171

-- Define the statement to be proved
theorem jafaris_candy : (candy_total - (taquon_candy + mack_candy)) = 76 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jafaris_candy_l2312_231246


namespace NUMINAMATH_GPT_total_wheels_in_garage_l2312_231283

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end NUMINAMATH_GPT_total_wheels_in_garage_l2312_231283


namespace NUMINAMATH_GPT_domain_of_function_l2312_231241

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2312_231241


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2312_231268

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (x : ℝ)
  (h_arith : arithmetic_seq a)
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l2312_231268


namespace NUMINAMATH_GPT_find_smallest_d_l2312_231250

theorem find_smallest_d (d : ℕ) : (5 + 6 + 2 + 4 + 8 + d) % 9 = 0 → d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_d_l2312_231250


namespace NUMINAMATH_GPT_sin_double_angle_l2312_231286

open Real

theorem sin_double_angle (α : ℝ) (h : tan α = -3/5) : sin (2 * α) = -15/17 :=
by
  -- We are skipping the proof here
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2312_231286


namespace NUMINAMATH_GPT_change_percentage_difference_l2312_231244

theorem change_percentage_difference 
  (initial_yes : ℚ) (initial_no : ℚ) (initial_undecided : ℚ)
  (final_yes : ℚ) (final_no : ℚ) (final_undecided : ℚ)
  (h_initial : initial_yes = 0.4 ∧ initial_no = 0.3 ∧ initial_undecided = 0.3)
  (h_final : final_yes = 0.6 ∧ final_no = 0.1 ∧ final_undecided = 0.3) :
  (final_yes - initial_yes + initial_no - final_no) = 0.2 := by
sorry

end NUMINAMATH_GPT_change_percentage_difference_l2312_231244


namespace NUMINAMATH_GPT_work_completion_time_l2312_231262

noncomputable def rate_b : ℝ := 1 / 24
noncomputable def rate_a : ℝ := 2 * rate_b
noncomputable def combined_rate : ℝ := rate_a + rate_b
noncomputable def completion_time : ℝ := 1 / combined_rate

theorem work_completion_time :
  completion_time = 8 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l2312_231262


namespace NUMINAMATH_GPT_min_value_of_ratio_l2312_231256

noncomputable def min_ratio (a b c d : ℕ) : ℝ :=
  let num := 1000 * a + 100 * b + 10 * c + d
  let denom := a + b + c + d
  (num : ℝ) / (denom : ℝ)

theorem min_value_of_ratio : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  min_ratio a b c d = 60.5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_ratio_l2312_231256


namespace NUMINAMATH_GPT_intersecting_lines_l2312_231295

theorem intersecting_lines (c d : ℝ) :
  (∀ x y, (x = (1/3) * y + c) ∧ (y = (1/3) * x + d) → x = 3 ∧ y = 6) →
  c + d = 6 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l2312_231295


namespace NUMINAMATH_GPT_ada_original_seat_l2312_231260

theorem ada_original_seat {positions : Fin 6 → Fin 6} 
  (Bea Ceci Dee Edie Fred Ada: Fin 6)
  (h1: Ada = 0)
  (h2: positions (Bea + 1) = Bea)
  (h3: positions (Ceci - 2) = Ceci)
  (h4: positions Dee = Edie ∧ positions Edie = Dee)
  (h5: positions Fred = Fred) :
  Ada = 1 → Bea = 1 → Ceci = 3 → Dee = 4 → Edie = 5 → Fred = 6 → Ada = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ada_original_seat_l2312_231260


namespace NUMINAMATH_GPT_problem_scores_ordering_l2312_231206

variable {J K L R : ℕ}

theorem problem_scores_ordering (h1 : J > K) (h2 : J > L) (h3 : J > R)
                                (h4 : L > min K R) (h5 : R > min K L)
                                (h6 : (J ≠ K) ∧ (J ≠ L) ∧ (J ≠ R) ∧ (K ≠ L) ∧ (K ≠ R) ∧ (L ≠ R)) :
                                K < L ∧ L < R :=
sorry

end NUMINAMATH_GPT_problem_scores_ordering_l2312_231206


namespace NUMINAMATH_GPT_problem1_no_solution_problem2_solution_l2312_231293

theorem problem1_no_solution (x : ℝ) 
  (h : (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1) : false :=
by
  -- The original equation turns out to have no solution
  sorry

theorem problem2_solution (x : ℝ) 
  (h : 1 - (x - 2)/(2 + x) = 16/(x^2 - 4)) : x = 6 :=
by
  -- The equation has a solution x = 6
  sorry

end NUMINAMATH_GPT_problem1_no_solution_problem2_solution_l2312_231293


namespace NUMINAMATH_GPT_exist_non_quadratic_residues_sum_l2312_231237

noncomputable section

def is_quadratic_residue_mod (p a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 ≡ a [ZMOD p]

theorem exist_non_quadratic_residues_sum {p : ℤ} (hp : p > 5) (hp_modeq : p ≡ 1 [ZMOD 4]) (a : ℤ) : 
  ∃ b c : ℤ, a = b + c ∧ ¬is_quadratic_residue_mod p b ∧ ¬is_quadratic_residue_mod p c :=
sorry

end NUMINAMATH_GPT_exist_non_quadratic_residues_sum_l2312_231237


namespace NUMINAMATH_GPT_boys_camp_total_l2312_231257

theorem boys_camp_total (T : ℕ) 
  (h1 : 0.20 * T = (0.20 : ℝ) * T) 
  (h2 : (0.30 : ℝ) * (0.20 * T) = (0.30 : ℝ) * (0.20 * T)) 
  (h3 : (0.70 : ℝ) * (0.20 * T) = 63) :
  T = 450 :=
by
  sorry

end NUMINAMATH_GPT_boys_camp_total_l2312_231257


namespace NUMINAMATH_GPT_water_tank_height_l2312_231248

theorem water_tank_height (r h : ℝ) (V : ℝ) (V_water : ℝ) (a b : ℕ) 
  (h_tank : h = 120) (r_tank : r = 20) (V_tank : V = (1/3) * π * r^2 * h) 
  (V_water_capacity : V_water = 0.4 * V) :
  a = 48 ∧ b = 2 ∧ V = 16000 * π ∧ V_water = 6400 * π ∧ 
  h_water = 48 * (2^(1/3) / 1) ∧ (a + b = 50) :=
by
  sorry

end NUMINAMATH_GPT_water_tank_height_l2312_231248


namespace NUMINAMATH_GPT_disloyal_bound_l2312_231239

variable {p n : ℕ}

/-- A number is disloyal if its GCD with n is not 1 -/
def isDisloyal (x : ℕ) (n : ℕ) := Nat.gcd x n ≠ 1

theorem disloyal_bound (p : ℕ) (n : ℕ) (hp : p.Prime) (hn : n % p^2 = 0) :
  (∃ D : Finset ℕ, (∀ x ∈ D, isDisloyal x n) ∧ D.card ≤ (n - 1) / p) :=
sorry

end NUMINAMATH_GPT_disloyal_bound_l2312_231239


namespace NUMINAMATH_GPT_find_constants_l2312_231242

theorem find_constants :
  ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 7) / (x^2 - 3 * x - 18) = P / (x - 6) + Q / (x + 3)) ∧
    P = 31 / 9 ∧ Q = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l2312_231242


namespace NUMINAMATH_GPT_partial_fraction_sum_zero_l2312_231201

theorem partial_fraction_sum_zero
    (A B C D E : ℝ)
    (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
        B * x * (x + 2) * (x + 3) * (x + 5) +
        C * x * (x + 1) * (x + 3) * (x + 5) +
        D * x * (x + 1) * (x + 2) * (x + 5) +
        E * x * (x + 1) * (x + 2) * (x + 3)) :
    A + B + C + D + E = 0 := by
    sorry

end NUMINAMATH_GPT_partial_fraction_sum_zero_l2312_231201


namespace NUMINAMATH_GPT_percentage_of_earrings_l2312_231291

theorem percentage_of_earrings (B M R : ℕ) (hB : B = 10) (hM : M = 2 * R) (hTotal : B + M + R = 70) : 
  (B * 100) / M = 25 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_earrings_l2312_231291


namespace NUMINAMATH_GPT_length_of_rectangular_garden_l2312_231259

theorem length_of_rectangular_garden (P B : ℝ) (h₁ : P = 1200) (h₂ : B = 240) :
  ∃ L : ℝ, P = 2 * (L + B) ∧ L = 360 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangular_garden_l2312_231259


namespace NUMINAMATH_GPT_integer_roots_if_q_positive_no_integer_roots_if_q_negative_l2312_231251

theorem integer_roots_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1) :=
sorry

theorem no_integer_roots_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬ ((∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1)) :=
sorry

end NUMINAMATH_GPT_integer_roots_if_q_positive_no_integer_roots_if_q_negative_l2312_231251


namespace NUMINAMATH_GPT_part1_part2_l2312_231277

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin x - 1/2 * Real.cos (2 * x) + a - 3/a + 1/2

theorem part1 (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 0 1 := sorry

theorem part2 (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 := sorry

end NUMINAMATH_GPT_part1_part2_l2312_231277


namespace NUMINAMATH_GPT_pages_of_shorter_book_is_10_l2312_231232

theorem pages_of_shorter_book_is_10
  (x : ℕ) 
  (h_diff : ∀ (y : ℕ), x = y - 10)
  (h_divide : (x + 10) / 2 = x) 
  : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_pages_of_shorter_book_is_10_l2312_231232


namespace NUMINAMATH_GPT_a_plus_b_eq_six_l2312_231264

theorem a_plus_b_eq_six (a b : ℤ) (k : ℝ) (h1 : k = a + Real.sqrt b)
  (h2 : ∀ k > 0, |Real.log k / Real.log 2 - Real.log (k + 6) / Real.log 2| = 1) :
  a + b = 6 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_six_l2312_231264


namespace NUMINAMATH_GPT_positive_real_solution_l2312_231226

theorem positive_real_solution (x : ℝ) (h : 0 < x)
  (h_eq : (1/3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 409 :=
sorry

end NUMINAMATH_GPT_positive_real_solution_l2312_231226


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2312_231281

variable (x y : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 1 ∧ y > 1) → (x + y > 2 ∧ x * y > 1) ∧
  ¬((x + y > 2 ∧ x * y > 1) → (x > 1 ∧ y > 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2312_231281


namespace NUMINAMATH_GPT_expression_subtracted_from_3_pow_k_l2312_231265

theorem expression_subtracted_from_3_pow_k (k : ℕ) (h : 15^k ∣ 759325) : 3^k - 0 = 1 :=
sorry

end NUMINAMATH_GPT_expression_subtracted_from_3_pow_k_l2312_231265
