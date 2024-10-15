import Mathlib

namespace NUMINAMATH_GPT_prove_inequality_l1442_144209

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end NUMINAMATH_GPT_prove_inequality_l1442_144209


namespace NUMINAMATH_GPT_solve_for_c_l1442_144290

theorem solve_for_c (c : ℚ) :
  (c - 35) / 14 = (2 * c + 9) / 49 →
  c = 1841 / 21 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_c_l1442_144290


namespace NUMINAMATH_GPT_characterize_functions_l1442_144240

open Function

noncomputable def f : ℚ → ℚ := sorry
noncomputable def g : ℚ → ℚ := sorry

axiom f_g_condition_1 : ∀ x y : ℚ, f (g (x) - g (y)) = f (g (x)) - y
axiom f_g_condition_2 : ∀ x y : ℚ, g (f (x) - f (y)) = g (f (x)) - y

theorem characterize_functions : 
  (∃ c : ℚ, ∀ x, f x = c * x) ∧ (∃ c : ℚ, ∀ x, g x = x / c) := 
sorry

end NUMINAMATH_GPT_characterize_functions_l1442_144240


namespace NUMINAMATH_GPT_a3_equals_neg7_l1442_144201

-- Definitions based on given conditions
noncomputable def a₁ := -11
noncomputable def d : ℤ := sorry -- this is derived but unknown presently
noncomputable def a(n : ℕ) : ℤ := a₁ + (n - 1) * d

axiom condition : a 4 + a 6 = -6

-- The proof problem statement
theorem a3_equals_neg7 : a 3 = -7 :=
by
  have h₁ : a₁ = -11 := rfl
  have h₂ : a 4 + a 6 = -6 := condition
  sorry

end NUMINAMATH_GPT_a3_equals_neg7_l1442_144201


namespace NUMINAMATH_GPT_angle_measure_l1442_144249

theorem angle_measure (x : ℝ) (h1 : x + 3 * x^2 + 10 = 90) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l1442_144249


namespace NUMINAMATH_GPT_range_of_m_for_inequality_l1442_144278

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, |x-1| + |x+m| > 3} = {m : ℝ | m < -4 ∨ m > 2} :=
sorry

end NUMINAMATH_GPT_range_of_m_for_inequality_l1442_144278


namespace NUMINAMATH_GPT_chess_team_boys_count_l1442_144228

theorem chess_team_boys_count : 
  ∃ (B G : ℕ), B + G = 30 ∧ (2 / 3 : ℚ) * G + B = 18 ∧ B = 6 := by
  sorry

end NUMINAMATH_GPT_chess_team_boys_count_l1442_144228


namespace NUMINAMATH_GPT_fill_tank_time_l1442_144284

theorem fill_tank_time (R L E : ℝ) (fill_time : ℝ) (leak_time : ℝ) (effective_rate : ℝ) : 
  (R = 1 / fill_time) → 
  (L = 1 / leak_time) →
  (E = R - L) →
  (fill_time = 10) →
  (leak_time = 110) →
  (E = 1 / effective_rate) →
  effective_rate = 11 :=
by
  sorry

end NUMINAMATH_GPT_fill_tank_time_l1442_144284


namespace NUMINAMATH_GPT_A_n_divisible_by_225_l1442_144207

theorem A_n_divisible_by_225 (n : ℕ) : 225 ∣ (16^n - 15 * n - 1) := by
  sorry

end NUMINAMATH_GPT_A_n_divisible_by_225_l1442_144207


namespace NUMINAMATH_GPT_find_k_l1442_144283

theorem find_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2023) : k = 2022 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_k_l1442_144283


namespace NUMINAMATH_GPT_functional_expression_y_x_maximize_profit_price_reduction_and_profit_l1442_144243

-- Define the conditions
variable (C_selling C_cost : ℝ := 80) (C_costComponent : ℝ := 30) (initialSales : ℝ := 600) 
variable (dec_price : ℝ := 2) (inc_sales : ℝ := 30)
variable (decrease x : ℝ)

-- Define and prove part 1: Functional expression between y and x
theorem functional_expression_y_x : (decrease : ℝ) → (15 * decrease + initialSales : ℝ) = (inc_sales / dec_price * decrease + initialSales) :=
by sorry

-- Define the function for weekly profit
def weekly_profit (x : ℝ) : ℝ := 
  let selling_price := C_selling - x
  let cost_price := C_costComponent
  let sales_volume := 15 * x + initialSales
  (selling_price - cost_price) * sales_volume

-- Prove the condition for maximizing weekly sales profit
theorem maximize_profit_price_reduction_and_profit : 
  (∀ x : ℤ, x % 2 = 0 → weekly_profit x ≤ 30360) ∧
  weekly_profit 4 = 30360 ∧ 
  weekly_profit 6 = 30360 :=
by sorry

end NUMINAMATH_GPT_functional_expression_y_x_maximize_profit_price_reduction_and_profit_l1442_144243


namespace NUMINAMATH_GPT_remainder_sum_div_8_l1442_144204

theorem remainder_sum_div_8 (n : ℤ) : (((8 - n) + (n + 5)) % 8) = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_sum_div_8_l1442_144204


namespace NUMINAMATH_GPT_polygon_sides_l1442_144293

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1442_144293


namespace NUMINAMATH_GPT_additional_cats_l1442_144245

theorem additional_cats {M R C : ℕ} (h1 : 20 * R = M) (h2 : 4 + 2 * C = 10) : C = 3 := 
  sorry

end NUMINAMATH_GPT_additional_cats_l1442_144245


namespace NUMINAMATH_GPT_depth_of_water_in_smaller_container_l1442_144244

theorem depth_of_water_in_smaller_container 
  (H_big : ℝ) (R_big : ℝ) (h_water : ℝ) 
  (H_small : ℝ) (R_small : ℝ) (expected_depth : ℝ) 
  (v_water_small : ℝ) 
  (v_water_big : ℝ) 
  (h_total_water : ℝ)
  (above_brim : ℝ) 
  (v_water_final : ℝ) : 

  H_big = 20 ∧ R_big = 6 ∧ h_water = 17 ∧ H_small = 18 ∧ R_small = 5 ∧ expected_depth = 2.88 ∧
  v_water_big = π * R_big^2 * H_big ∧ v_water_small = π * R_small^2 * H_small ∧ 
  h_total_water = π * R_big^2 * h_water ∧ above_brim = π * R_big^2 * (H_big - H_small) ∧ 
  v_water_final = above_brim →

  expected_depth = v_water_final / (π * R_small^2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_depth_of_water_in_smaller_container_l1442_144244


namespace NUMINAMATH_GPT_lollipops_initial_count_l1442_144272

theorem lollipops_initial_count (L : ℕ) (k : ℕ) 
  (h1 : L % 42 ≠ 0) 
  (h2 : (L + 22) % 42 = 0) : 
  L = 62 :=
by
  sorry

end NUMINAMATH_GPT_lollipops_initial_count_l1442_144272


namespace NUMINAMATH_GPT_triangle_side_length_b_l1442_144238

/-
In a triangle ABC with angles such that ∠C = 4∠A, and sides such that a = 35 and c = 64, prove that the length of side b is 140 * cos²(A).
-/
theorem triangle_side_length_b (A C : ℝ) (a c : ℝ) (hC : C = 4 * A) (ha : a = 35) (hc : c = 64) :
  ∃ (b : ℝ), b = 140 * (Real.cos A) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_b_l1442_144238


namespace NUMINAMATH_GPT_day_after_2_pow_20_is_friday_l1442_144226

-- Define the given conditions
def today_is_monday : ℕ := 0 -- Assuming Monday is represented by 0

-- Define the number of days after \(2^{20}\) days
def days_after : ℕ := 2^20

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the function to find the day of the week after a given number of days
def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % days_in_week

-- The theorem to prove
theorem day_after_2_pow_20_is_friday :
  day_of_week today_is_monday days_after = 5 := -- Friday is represented by 5 here
sorry

end NUMINAMATH_GPT_day_after_2_pow_20_is_friday_l1442_144226


namespace NUMINAMATH_GPT_problem_statement_l1442_144224

def a := 596
def b := 130
def c := 270

theorem problem_statement : a - b - c = a - (b + c) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1442_144224


namespace NUMINAMATH_GPT_sum_of_numbers_l1442_144213

-- Define the conditions
variables (a b : ℝ) (r d : ℝ)
def geometric_progression := a = 3 * r ∧ b = 3 * r^2
def arithmetic_progression := b = a + d ∧ 9 = b + d

-- Define the problem as proving the sum of a and b
theorem sum_of_numbers (h1 : geometric_progression a b r)
                       (h2 : arithmetic_progression a b d) : 
  a + b = 45 / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1442_144213


namespace NUMINAMATH_GPT_female_students_in_sample_l1442_144254

-- Definitions of the given conditions
def male_students : ℕ := 28
def female_students : ℕ := 21
def total_students : ℕ := male_students + female_students
def sample_size : ℕ := 14
def stratified_sampling_fraction : ℚ := (sample_size : ℚ) / (total_students : ℚ)
def female_sample_count : ℚ := stratified_sampling_fraction * (female_students : ℚ)

-- The theorem to prove
theorem female_students_in_sample : female_sample_count = 6 :=
by
  sorry

end NUMINAMATH_GPT_female_students_in_sample_l1442_144254


namespace NUMINAMATH_GPT_Annette_more_than_Sara_l1442_144271

variable (A C S : ℕ)

-- Define the given conditions as hypotheses
def Annette_Caitlin_weight : Prop := A + C = 95
def Caitlin_Sara_weight : Prop := C + S = 87

-- The theorem to prove: Annette weighs 8 pounds more than Sara
theorem Annette_more_than_Sara (h1 : Annette_Caitlin_weight A C)
                               (h2 : Caitlin_Sara_weight C S) :
  A - S = 8 := by
  sorry

end NUMINAMATH_GPT_Annette_more_than_Sara_l1442_144271


namespace NUMINAMATH_GPT_remainder_when_divided_by_95_l1442_144237

theorem remainder_when_divided_by_95 (x : ℤ) (h1 : x % 19 = 12) :
  x % 95 = 12 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_95_l1442_144237


namespace NUMINAMATH_GPT_minimum_groups_l1442_144250

theorem minimum_groups (total_players : ℕ) (max_per_group : ℕ)
  (h_total : total_players = 30)
  (h_max : max_per_group = 12) :
  ∃ x y, y ∣ total_players ∧ y ≤ max_per_group ∧ total_players / y = x ∧ x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_groups_l1442_144250


namespace NUMINAMATH_GPT_avg_minutes_eq_170_div_9_l1442_144267

-- Define the conditions
variables (s : ℕ) -- number of seventh graders
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2
def sixth_grade_minutes := 18
def seventh_grade_run_minutes := 20
def seventh_grade_stretching_minutes := 5
def eighth_grade_minutes := 12

-- Define the total activity minutes for each grade
def total_activity_minutes_sixth := sixth_grade_minutes * sixth_graders
def total_activity_minutes_seventh := (seventh_grade_run_minutes + seventh_grade_stretching_minutes) * seventh_graders
def total_activity_minutes_eighth := eighth_grade_minutes * eighth_graders

-- Calculate total activity minutes
def total_activity_minutes := total_activity_minutes_sixth + total_activity_minutes_seventh + total_activity_minutes_eighth

-- Calculate total number of students
def total_students := sixth_graders + seventh_graders + eighth_graders

-- Calculate average minutes per student
def average_minutes_per_student := total_activity_minutes / total_students

theorem avg_minutes_eq_170_div_9 : average_minutes_per_student s = 170 / 9 := by
  sorry

end NUMINAMATH_GPT_avg_minutes_eq_170_div_9_l1442_144267


namespace NUMINAMATH_GPT_fg_of_neg2_l1442_144241

def f (x : ℤ) : ℤ := x^2 + 4
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_of_neg2 : f (g (-2)) = 20 := by
  sorry

end NUMINAMATH_GPT_fg_of_neg2_l1442_144241


namespace NUMINAMATH_GPT_solve_for_q_l1442_144206

theorem solve_for_q (k l q : ℕ) (h1 : (2 : ℚ) / 3 = k / 45) (h2 : (2 : ℚ) / 3 = (k + l) / 75) (h3 : (2 : ℚ) / 3 = (q - l) / 105) : q = 90 :=
sorry

end NUMINAMATH_GPT_solve_for_q_l1442_144206


namespace NUMINAMATH_GPT_geometric_proportion_exists_l1442_144259

theorem geometric_proportion_exists (x y : ℝ) (h1 : x + (24 - x) = 24) 
  (h2 : y + (16 - y) = 16) (h3 : x^2 + y^2 + (16 - y)^2 + (24 - x)^2 = 580) : 
  (21 / 7 = 9 / 3) :=
  sorry

end NUMINAMATH_GPT_geometric_proportion_exists_l1442_144259


namespace NUMINAMATH_GPT_parabola_coefficients_sum_l1442_144236

theorem parabola_coefficients_sum :
  ∃ a b c : ℝ, 
  (∀ y : ℝ, (7 = -(6 ^ 2) * a + b * 6 + c)) ∧
  (5 = a * (-4) ^ 2 + b * (-4) + c) ∧
  (a + b + c = -42) := 
sorry

end NUMINAMATH_GPT_parabola_coefficients_sum_l1442_144236


namespace NUMINAMATH_GPT_dmitry_black_socks_l1442_144212

theorem dmitry_black_socks :
  let blue_socks := 10
  let initial_black_socks := 22
  let white_socks := 12
  let total_initial_socks := blue_socks + initial_black_socks + white_socks
  ∀ x : ℕ,
    let total_socks := total_initial_socks + x
    let black_socks := initial_black_socks + x
    (black_socks : ℚ) / (total_socks : ℚ) = 2 / 3 → x = 22 :=
by
  sorry

end NUMINAMATH_GPT_dmitry_black_socks_l1442_144212


namespace NUMINAMATH_GPT_jackson_total_calories_l1442_144296

def lettuce_calories : ℕ := 50
def carrots_calories : ℕ := 2 * lettuce_calories
def dressing_calories : ℕ := 210
def salad_calories : ℕ := lettuce_calories + carrots_calories + dressing_calories

def crust_calories : ℕ := 600
def pepperoni_calories : ℕ := crust_calories / 3
def cheese_calories : ℕ := 400
def pizza_calories : ℕ := crust_calories + pepperoni_calories + cheese_calories

def jackson_salad_fraction : ℚ := 1 / 4
def jackson_pizza_fraction : ℚ := 1 / 5

noncomputable def total_calories : ℚ := 
  jackson_salad_fraction * salad_calories + jackson_pizza_fraction * pizza_calories

theorem jackson_total_calories : total_calories = 330 := by
  sorry

end NUMINAMATH_GPT_jackson_total_calories_l1442_144296


namespace NUMINAMATH_GPT_unique_solution_k_values_l1442_144268

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, k * x ^ 2 - 3 * x + 2 = 0) ↔ (k = 0 ∨ k = 9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_k_values_l1442_144268


namespace NUMINAMATH_GPT_new_train_distance_l1442_144275

-- Define the given conditions
def distance_old : ℝ := 300
def percentage_increase : ℝ := 0.3

-- Define the target distance to prove
def distance_new : ℝ := distance_old + (percentage_increase * distance_old)

-- State the theorem
theorem new_train_distance : distance_new = 390 := by
  sorry

end NUMINAMATH_GPT_new_train_distance_l1442_144275


namespace NUMINAMATH_GPT_notebook_pen_ratio_l1442_144229

theorem notebook_pen_ratio (pen_cost notebook_total_cost : ℝ) (num_notebooks : ℕ)
  (h1 : pen_cost = 1.50) (h2 : notebook_total_cost = 18) (h3 : num_notebooks = 4) :
  (notebook_total_cost / num_notebooks) / pen_cost = 3 :=
by
  -- The steps to prove this would go here
  sorry

end NUMINAMATH_GPT_notebook_pen_ratio_l1442_144229


namespace NUMINAMATH_GPT_product_mod_32_l1442_144246

def product_of_all_odd_primes_less_than_32 : ℕ :=
  3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  (product_of_all_odd_primes_less_than_32) % 32 = 9 :=
sorry

end NUMINAMATH_GPT_product_mod_32_l1442_144246


namespace NUMINAMATH_GPT_leonine_cats_l1442_144219

theorem leonine_cats (n : ℕ) (h : n = (4 / 5 * n) + (4 / 5)) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_leonine_cats_l1442_144219


namespace NUMINAMATH_GPT_total_animals_l1442_144262

theorem total_animals (total_legs : ℕ) (number_of_sheep : ℕ)
  (legs_per_chicken : ℕ) (legs_per_sheep : ℕ)
  (H1 : total_legs = 60) 
  (H2 : number_of_sheep = 10)
  (H3 : legs_per_chicken = 2)
  (H4 : legs_per_sheep = 4) : 
  number_of_sheep + (total_legs - number_of_sheep * legs_per_sheep) / legs_per_chicken = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_animals_l1442_144262


namespace NUMINAMATH_GPT_lorraine_initial_brownies_l1442_144248

theorem lorraine_initial_brownies (B : ℝ) 
(h1: (0.375 * B - 1 = 5)) : B = 16 := 
sorry

end NUMINAMATH_GPT_lorraine_initial_brownies_l1442_144248


namespace NUMINAMATH_GPT_football_team_throwers_l1442_144295

theorem football_team_throwers {T N : ℕ} (h1 : 70 - T = N)
                                (h2 : 62 = T + (2 / 3 * N)) : 
                                T = 46 := 
by
  sorry

end NUMINAMATH_GPT_football_team_throwers_l1442_144295


namespace NUMINAMATH_GPT_find_value_l1442_144256

-- Defining the known conditions
def number : ℕ := 20
def half (n : ℕ) : ℕ := n / 2
def value_added (V : ℕ) : Prop := half number + V = 17

-- Proving that the value added to half the number is 7
theorem find_value : value_added 7 :=
by
  -- providing the proof for the theorem
  -- skipping the proof steps with sorry
  sorry

end NUMINAMATH_GPT_find_value_l1442_144256


namespace NUMINAMATH_GPT_danny_more_caps_l1442_144208

variable (found thrown_away : ℕ)

def bottle_caps_difference (found thrown_away : ℕ) : ℕ :=
  found - thrown_away

theorem danny_more_caps
  (h_found : found = 36)
  (h_thrown_away : thrown_away = 35) :
  bottle_caps_difference found thrown_away = 1 :=
by
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_danny_more_caps_l1442_144208


namespace NUMINAMATH_GPT_no_such_natural_numbers_exist_l1442_144269

theorem no_such_natural_numbers_exist :
  ¬ ∃ (x y : ℕ), ∃ (k m : ℕ), x^2 + x + 1 = y^k ∧ y^2 + y + 1 = x^m := 
by sorry

end NUMINAMATH_GPT_no_such_natural_numbers_exist_l1442_144269


namespace NUMINAMATH_GPT_solve_abs_eq_l1442_144261

theorem solve_abs_eq (x : ℝ) : |2*x - 6| = 3*x + 6 ↔ x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l1442_144261


namespace NUMINAMATH_GPT_find_number_divided_l1442_144239

theorem find_number_divided (n : ℕ) (h : n = 21 * 9 + 1) : n = 190 :=
by
  sorry

end NUMINAMATH_GPT_find_number_divided_l1442_144239


namespace NUMINAMATH_GPT_trisha_total_distance_l1442_144233

-- Define each segment of Trisha's walk in miles
def hotel_to_postcard : ℝ := 0.1111111111111111
def postcard_to_tshirt : ℝ := 0.2222222222222222
def tshirt_to_keychain : ℝ := 0.7777777777777778
def keychain_to_toy : ℝ := 0.5555555555555556
def meters_to_miles (m : ℝ) : ℝ := m * 0.000621371
def toy_to_bookstore : ℝ := meters_to_miles 400
def bookstore_to_hotel : ℝ := 0.6666666666666666

-- Sum of all distances
def total_distance : ℝ :=
  hotel_to_postcard +
  postcard_to_tshirt +
  tshirt_to_keychain +
  keychain_to_toy +
  toy_to_bookstore +
  bookstore_to_hotel

-- Proof statement
theorem trisha_total_distance : total_distance = 1.5818817333333333 := by
  sorry

end NUMINAMATH_GPT_trisha_total_distance_l1442_144233


namespace NUMINAMATH_GPT_find_integer_k_l1442_144221

noncomputable def P : ℤ → ℤ := sorry

theorem find_integer_k :
  P 1 = 2019 ∧ P 2019 = 1 ∧ ∃ k : ℤ, P k = k ∧ k = 1010 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_k_l1442_144221


namespace NUMINAMATH_GPT_percentage_profit_double_price_l1442_144274

theorem percentage_profit_double_price (C S1 S2 : ℝ) (h1 : S1 = 1.5 * C) (h2 : S2 = 2 * S1) : 
  ((S2 - C) / C) * 100 = 200 := by
  sorry

end NUMINAMATH_GPT_percentage_profit_double_price_l1442_144274


namespace NUMINAMATH_GPT_lloyd_house_of_cards_l1442_144205

theorem lloyd_house_of_cards 
  (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ)
  (h1 : decks = 24) (h2 : cards_per_deck = 78) (h3 : layers = 48) :
  ((decks * cards_per_deck) / layers) = 39 := 
  by
  sorry

end NUMINAMATH_GPT_lloyd_house_of_cards_l1442_144205


namespace NUMINAMATH_GPT_div_c_a_l1442_144215

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_div_c_a_l1442_144215


namespace NUMINAMATH_GPT_integral_right_angled_triangles_unique_l1442_144273

theorem integral_right_angled_triangles_unique : 
  ∀ a b c : ℤ, (a < b) ∧ (b < c) ∧ (a^2 + b^2 = c^2) ∧ (a * b = 4 * (a + b + c))
  ↔ (a = 10 ∧ b = 24 ∧ c = 26)
  ∨ (a = 12 ∧ b = 16 ∧ c = 20)
  ∨ (a = 9 ∧ b = 40 ∧ c = 41) :=
by {
  sorry
}

end NUMINAMATH_GPT_integral_right_angled_triangles_unique_l1442_144273


namespace NUMINAMATH_GPT_number_of_outliers_l1442_144264

def data_set : List ℕ := [4, 23, 27, 27, 35, 37, 37, 39, 47, 53]

def Q1 : ℕ := 27
def Q3 : ℕ := 39

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - (3 * IQR / 2)
def upper_threshold : ℕ := Q3 + (3 * IQR / 2)

def outliers (s : List ℕ) (low high : ℕ) : List ℕ :=
  s.filter (λ x => x < low ∨ x > high)

theorem number_of_outliers :
  outliers data_set lower_threshold upper_threshold = [4] :=
by
  sorry

end NUMINAMATH_GPT_number_of_outliers_l1442_144264


namespace NUMINAMATH_GPT_expression_value_l1442_144214

variable (m n : ℝ)

theorem expression_value (hm : 3 * m ^ 2 + 5 * m - 3 = 0)
                         (hn : 3 * n ^ 2 - 5 * n - 3 = 0)
                         (hneq : m * n ≠ 1) :
                         (1 / n ^ 2) + (m / n) - (5 / 3) * m = 25 / 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_value_l1442_144214


namespace NUMINAMATH_GPT_exists_two_positive_integers_dividing_3003_l1442_144220

theorem exists_two_positive_integers_dividing_3003 : 
  ∃ (m1 m2 : ℕ), m1 > 0 ∧ m2 > 0 ∧ m1 ≠ m2 ∧ (3003 % (m1^2 + 2) = 0) ∧ (3003 % (m2^2 + 2) = 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_two_positive_integers_dividing_3003_l1442_144220


namespace NUMINAMATH_GPT_homework_problem1_homework_problem2_l1442_144281

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end NUMINAMATH_GPT_homework_problem1_homework_problem2_l1442_144281


namespace NUMINAMATH_GPT_opposite_of_neg5_is_pos5_l1442_144247

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg5_is_pos5_l1442_144247


namespace NUMINAMATH_GPT_additional_plates_added_l1442_144286

def initial_plates : ℕ := 27
def added_plates : ℕ := 37
def total_plates : ℕ := 83

theorem additional_plates_added :
  total_plates - (initial_plates + added_plates) = 19 :=
by
  sorry

end NUMINAMATH_GPT_additional_plates_added_l1442_144286


namespace NUMINAMATH_GPT_seq_proof_l1442_144263

noncomputable def arithmetic_seq (a1 a2 : ℤ) : Prop :=
  ∃ (d : ℤ), a1 = -1 + d ∧ a2 = a1 + d ∧ -4 = a1 + 3 * d

noncomputable def geometric_seq (b : ℤ) : Prop :=
  b = 2 ∨ b = -2

theorem seq_proof (a1 a2 b : ℤ) 
  (h1 : arithmetic_seq a1 a2) 
  (h2 : geometric_seq b) : 
  (a2 + a1 : ℚ) / b = 5 / 2 ∨ (a2 + a1 : ℚ) / b = -5 / 2 := by
  sorry

end NUMINAMATH_GPT_seq_proof_l1442_144263


namespace NUMINAMATH_GPT_simultaneous_solution_exists_l1442_144230

-- Definitions required by the problem
def eqn1 (m x : ℝ) : ℝ := m * x + 2
def eqn2 (m x : ℝ) : ℝ := (3 * m - 2) * x + 5

-- Proof statement
theorem simultaneous_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = eqn1 m x ∧ y = eqn2 m x) ↔ (m ≠ 1) := 
sorry

end NUMINAMATH_GPT_simultaneous_solution_exists_l1442_144230


namespace NUMINAMATH_GPT_polynomial_divisibility_l1442_144292

theorem polynomial_divisibility (a : ℤ) (n : ℕ) (h_pos : 0 < n) : 
  (a ^ (2 * n + 1) + (a - 1) ^ (n + 2)) % (a ^ 2 - a + 1) = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1442_144292


namespace NUMINAMATH_GPT_vector_addition_correct_dot_product_correct_l1442_144277

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem vector_addition_correct :
  let a := (1, 2)
  let b := (3, 1)
  vector_add a b = (4, 3) := by
  sorry

theorem dot_product_correct :
  let a := (1, 2)
  let b := (3, 1)
  dot_product a b = 5 := by
  sorry

end NUMINAMATH_GPT_vector_addition_correct_dot_product_correct_l1442_144277


namespace NUMINAMATH_GPT_sam_total_coins_l1442_144270

theorem sam_total_coins (nickel_count : ℕ) (dime_count : ℕ) (total_value_cents : ℤ) (nickel_value : ℤ) (dime_value : ℤ)
  (h₁ : nickel_count = 12)
  (h₂ : total_value_cents = 240)
  (h₃ : nickel_value = 5)
  (h₄ : dime_value = 10)
  (h₅ : nickel_count * nickel_value + dime_count * dime_value = total_value_cents) :
  nickel_count + dime_count = 30 := 
  sorry

end NUMINAMATH_GPT_sam_total_coins_l1442_144270


namespace NUMINAMATH_GPT_Bryan_deposited_312_l1442_144253

-- Definitions based on conditions
def MarkDeposit : ℕ := 88
def TotalDeposit : ℕ := 400
def MaxBryanDeposit (MarkDeposit : ℕ) : ℕ := 5 * MarkDeposit 

def BryanDeposit (B : ℕ) : Prop := B < MaxBryanDeposit MarkDeposit ∧ MarkDeposit + B = TotalDeposit

theorem Bryan_deposited_312 : BryanDeposit 312 :=
by
   -- Proof steps go here
   sorry

end NUMINAMATH_GPT_Bryan_deposited_312_l1442_144253


namespace NUMINAMATH_GPT_unique_solution_exists_l1442_144282

theorem unique_solution_exists (ell : ℚ) (h : ell ≠ -2) : 
  (∃! x : ℚ, (x + 3) / (ell * x + 2) = x) ↔ ell = -1 / 12 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l1442_144282


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1442_144200

variable (a_1 q : ℚ) (S : ℕ → ℚ)

def geometric_sum (n : ℕ) : ℚ :=
  a_1 * (1 - q^n) / (1 - q)

def is_arithmetic_sequence (a b c : ℚ) : Prop :=
  2 * b = a + c

theorem common_ratio_of_geometric_sequence 
  (h1 : ∀ n, S n = geometric_sum a_1 q n)
  (h2 : ∀ n, is_arithmetic_sequence (S (n+2)) (S (n+1)) (S n)) : q = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1442_144200


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1442_144210

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a1 : a 1 = 2
axiom a2_a3_sum : a 2 + a 3 = 13

-- The theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) : a (4) + a (5) + a (6) = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1442_144210


namespace NUMINAMATH_GPT_sheets_of_paper_in_each_box_l1442_144216

theorem sheets_of_paper_in_each_box (E S : ℕ) (h1 : 2 * E + 40 = S) (h2 : 4 * (E - 40) = S) : S = 240 :=
by
  sorry

end NUMINAMATH_GPT_sheets_of_paper_in_each_box_l1442_144216


namespace NUMINAMATH_GPT_oranges_to_apples_equivalence_l1442_144242

theorem oranges_to_apples_equivalence :
  (forall (o l a : ℝ), 4 * o = 3 * l ∧ 5 * l = 7 * a -> 20 * o = 21 * a) :=
by
  intro o l a
  intro h
  sorry

end NUMINAMATH_GPT_oranges_to_apples_equivalence_l1442_144242


namespace NUMINAMATH_GPT_A_share_in_profit_l1442_144298

-- Define the investments and profits
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12200

-- Define the total investment
def total_investment : ℕ := A_investment + B_investment + C_investment

-- Define A's ratio in the investment
def A_ratio : ℚ := A_investment / total_investment

-- Define A's share in the profit
def A_share : ℚ := total_profit * A_ratio

-- The theorem to prove
theorem A_share_in_profit : A_share = 3660 := by
  sorry

end NUMINAMATH_GPT_A_share_in_profit_l1442_144298


namespace NUMINAMATH_GPT_new_average_l1442_144203

theorem new_average (n : ℕ) (a : ℕ) (multiplier : ℕ) (average : ℕ) :
  (n = 35) →
  (a = 25) →
  (multiplier = 5) →
  (average = 125) →
  ((n * a * multiplier) / n = average) :=
by
  intros hn ha hm havg
  rw [hn, ha, hm]
  norm_num
  sorry

end NUMINAMATH_GPT_new_average_l1442_144203


namespace NUMINAMATH_GPT_digits_conditions_l1442_144227

noncomputable def original_number : ℕ := 253
noncomputable def reversed_number : ℕ := 352

theorem digits_conditions (a b c : ℕ) : 
  a + b + c = 10 → 
  b = a + c → 
  (original_number = a * 100 + b * 10 + c) → 
  (reversed_number = c * 100 + b * 10 + a) → 
  reversed_number - original_number = 99 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_digits_conditions_l1442_144227


namespace NUMINAMATH_GPT_negation_of_proposition_l1442_144276

theorem negation_of_proposition:
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1442_144276


namespace NUMINAMATH_GPT_general_formula_l1442_144285

def a (n : ℕ) : ℕ :=
match n with
| 0 => 1
| k+1 => 2 * a k + 4

theorem general_formula (n : ℕ) : a (n+1) = 5 * 2^n - 4 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_l1442_144285


namespace NUMINAMATH_GPT_part_to_third_fraction_is_six_five_l1442_144279

noncomputable def ratio_of_part_to_third_fraction (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ℝ :=
  P / (N / 3)

theorem part_to_third_fraction_is_six_five (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ratio_of_part_to_third_fraction P N h1 h2 = 6 / 5 :=
  sorry

end NUMINAMATH_GPT_part_to_third_fraction_is_six_five_l1442_144279


namespace NUMINAMATH_GPT_missing_weights_l1442_144266

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end NUMINAMATH_GPT_missing_weights_l1442_144266


namespace NUMINAMATH_GPT_width_of_wide_flags_l1442_144232

def total_fabric : ℕ := 1000
def leftover_fabric : ℕ := 294
def num_square_flags : ℕ := 16
def square_flag_area : ℕ := 16
def num_tall_flags : ℕ := 10
def tall_flag_area : ℕ := 15
def num_wide_flags : ℕ := 20
def wide_flag_height : ℕ := 3

theorem width_of_wide_flags :
  (total_fabric - leftover_fabric - (num_square_flags * square_flag_area + num_tall_flags * tall_flag_area)) / num_wide_flags / wide_flag_height = 5 :=
by
  sorry

end NUMINAMATH_GPT_width_of_wide_flags_l1442_144232


namespace NUMINAMATH_GPT_man_days_to_complete_work_alone_l1442_144211

-- Defining the variables corresponding to the conditions
variable (M : ℕ)

-- Initial condition: The man can do the work alone in M days
def man_work_rate := 1 / (M : ℚ)
-- The son can do the work alone in 20 days
def son_work_rate := 1 / 20
-- Combined work rate when together
def combined_work_rate := 1 / 4

-- The main theorem we want to prove
theorem man_days_to_complete_work_alone
  (h : man_work_rate M + son_work_rate = combined_work_rate) :
  M = 5 := by
  sorry

end NUMINAMATH_GPT_man_days_to_complete_work_alone_l1442_144211


namespace NUMINAMATH_GPT_validate_option_B_l1442_144291

theorem validate_option_B (a b : ℝ) : 
  (2 * a + 3 * a^2 ≠ 5 * a^3) ∧ 
  ((-a^3)^2 = a^6) ∧ 
  (¬ (-4 * a^3 * b / (2 * a) = -2 * a^2)) ∧ 
  ((5 * a * b)^2 ≠ 10 * a^2 * b^2) := 
by
  sorry

end NUMINAMATH_GPT_validate_option_B_l1442_144291


namespace NUMINAMATH_GPT_no_such_integers_l1442_144217

theorem no_such_integers (x y : ℤ) : ¬ ∃ x y : ℤ, (x^4 + 6) % 13 = y^3 % 13 :=
sorry

end NUMINAMATH_GPT_no_such_integers_l1442_144217


namespace NUMINAMATH_GPT_solve_for_x_l1442_144280

theorem solve_for_x (x : ℝ) (h : (3 + 2 / x)^(1 / 3) = 2) : x = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1442_144280


namespace NUMINAMATH_GPT_tina_total_time_l1442_144287

-- Define constants for the problem conditions
def assignment_time : Nat := 20
def dinner_time : Nat := 17 * 60 + 30 -- 5:30 PM in minutes
def clean_time_per_key : Nat := 7
def total_keys : Nat := 30
def remaining_keys : Nat := total_keys - 1
def dry_time_per_key : Nat := 10
def break_time : Nat := 3
def keys_per_break : Nat := 5

-- Define a function to compute total cleaning time for remaining keys
def total_cleaning_time (keys : Nat) (clean_time : Nat) : Nat :=
  keys * clean_time

-- Define a function to compute total drying time for all keys
def total_drying_time (keys : Nat) (dry_time : Nat) : Nat :=
  keys * dry_time

-- Define a function to compute total break time
def total_break_time (keys : Nat) (keys_per_break : Nat) (break_time : Nat) : Nat :=
  (keys / keys_per_break) * break_time

-- Define a function to compute the total time including cleaning, drying, breaks, and assignment
def total_time (cleaning_time drying_time break_time assignment_time : Nat) : Nat :=
  cleaning_time + drying_time + break_time + assignment_time

-- The theorem to be proven
theorem tina_total_time : 
  total_time (total_cleaning_time remaining_keys clean_time_per_key) 
              (total_drying_time total_keys dry_time_per_key)
              (total_break_time total_keys keys_per_break break_time)
              assignment_time = 541 :=
by sorry

end NUMINAMATH_GPT_tina_total_time_l1442_144287


namespace NUMINAMATH_GPT_sqrt2_over_2_not_covered_by_rationals_l1442_144231

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end NUMINAMATH_GPT_sqrt2_over_2_not_covered_by_rationals_l1442_144231


namespace NUMINAMATH_GPT_twelve_star_three_eq_four_star_eight_eq_star_assoc_l1442_144299

def star (a b : ℕ) : ℕ := 10^a * 10^b

theorem twelve_star_three_eq : star 12 3 = 10^15 :=
by 
  -- Proof here
  sorry

theorem four_star_eight_eq : star 4 8 = 10^12 :=
by 
  -- Proof here
  sorry

theorem star_assoc (a b c : ℕ) : star (a + b) c = star a (b + c) :=
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_twelve_star_three_eq_four_star_eight_eq_star_assoc_l1442_144299


namespace NUMINAMATH_GPT_shoes_to_belts_ratio_l1442_144260

variable (hats : ℕ) (belts : ℕ) (shoes : ℕ)

theorem shoes_to_belts_ratio (hats_eq : hats = 5)
                            (belts_eq : belts = hats + 2)
                            (shoes_eq : shoes = 14) : 
  (shoes / (Nat.gcd shoes belts)) = 2 ∧ (belts / (Nat.gcd shoes belts)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_shoes_to_belts_ratio_l1442_144260


namespace NUMINAMATH_GPT_derivative_at_zero_l1442_144223

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- Theorem statement to prove
theorem derivative_at_zero : 
  deriv f 0 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l1442_144223


namespace NUMINAMATH_GPT_find_n_eq_130_l1442_144294

theorem find_n_eq_130 
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : 0 < n)
  (h2 : d1 < d2)
  (h3 : d2 < d3)
  (h4 : d3 < d4)
  (h5 : ∀ d, d ∣ n → d = d1 ∨ d = d2 ∨ d = d3 ∨ d = d4 ∨ d ∣ n → ¬(1 < d ∧ d < d1))
  (h6 : n = d1^2 + d2^2 + d3^2 + d4^2) : n = 130 := 
  sorry

end NUMINAMATH_GPT_find_n_eq_130_l1442_144294


namespace NUMINAMATH_GPT_g_decreasing_on_neg1_0_l1442_144257

noncomputable def f (x : ℝ) : ℝ := 8 + 2 * x - x^2 
noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_decreasing_on_neg1_0 : 
  ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y → g y < g x :=
sorry

end NUMINAMATH_GPT_g_decreasing_on_neg1_0_l1442_144257


namespace NUMINAMATH_GPT_opposite_of_neg_twelve_l1442_144222

def opposite (n : Int) : Int := -n

theorem opposite_of_neg_twelve : opposite (-12) = 12 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_twelve_l1442_144222


namespace NUMINAMATH_GPT_right_triangle_area_l1442_144202

theorem right_triangle_area (A B C : ℝ) (hA : A = 64) (hB : B = 49) (hC : C = 225) :
  let a := Real.sqrt A
  let b := Real.sqrt B
  let c := Real.sqrt C
  ∃ (area : ℝ), area = (1 / 2) * a * b ∧ area = 28 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1442_144202


namespace NUMINAMATH_GPT_rhombus_area_l1442_144252

-- Define the given conditions: diagonals and side length
def d1 : ℕ := 40
def d2 : ℕ := 18
def s : ℕ := 25

-- Prove that the area of the rhombus is 360 square units given the conditions
theorem rhombus_area :
  (d1 * d2) / 2 = 360 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1442_144252


namespace NUMINAMATH_GPT_weight_of_seventh_person_l1442_144251

noncomputable def weight_of_six_people : ℕ := 6 * 156
noncomputable def new_average_weight (x : ℕ) : Prop := (weight_of_six_people + x) / 7 = 151

theorem weight_of_seventh_person (x : ℕ) (h : new_average_weight x) : x = 121 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_seventh_person_l1442_144251


namespace NUMINAMATH_GPT_domain_f_monotonicity_f_inequality_solution_l1442_144255

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

variable {x : ℝ}

theorem domain_f : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 -> Set.Ioo (-1 : ℝ) 1 := sorry

theorem monotonicity_f : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f y < f x := sorry

theorem inequality_solution :
  {x : ℝ | f (2 * x - 1) < 0} = {x | x > 1 / 2 ∧ x < 1} := sorry

end NUMINAMATH_GPT_domain_f_monotonicity_f_inequality_solution_l1442_144255


namespace NUMINAMATH_GPT_three_power_not_square_l1442_144297

theorem three_power_not_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : ¬ ∃ k : ℕ, k * k = 3^m + 3^n + 1 := by 
  sorry

end NUMINAMATH_GPT_three_power_not_square_l1442_144297


namespace NUMINAMATH_GPT_find_original_number_l1442_144258

-- Defining the conditions as given in the problem
def original_number_condition (x : ℤ) : Prop :=
  3 * (3 * x - 6) = 141

-- Stating the main theorem to be proven
theorem find_original_number (x : ℤ) (h : original_number_condition x) : x = 17 :=
sorry

end NUMINAMATH_GPT_find_original_number_l1442_144258


namespace NUMINAMATH_GPT_computation_l1442_144288

def g (x : ℕ) : ℕ := 7 * x - 3

theorem computation : g (g (g (g 1))) = 1201 := by
  sorry

end NUMINAMATH_GPT_computation_l1442_144288


namespace NUMINAMATH_GPT_problem_statement_l1442_144265

noncomputable def general_term (a : ℕ → ℕ) (n : ℕ) : Prop :=
a n = n

noncomputable def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = (n * (n + 1)) / 2

noncomputable def b_def (S : ℕ → ℕ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (2 : ℚ) / (S n)

noncomputable def sum_b_first_n_terms (b : ℕ → ℚ) (T : ℕ → ℚ) : Prop :=
∀ n, T n = (4 * n) / (n + 1)

theorem problem_statement (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a n = 1 + (n - 1) * 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) - a n ≠ 0) →
  a 3 ^ 2 = a 1 * a 9 →
  general_term a 1 →
  sum_first_n_terms a S →
  b_def S b →
  sum_b_first_n_terms b T :=
by
  intro arithmetic_seq
  intro a_1_eq_1
  intro non_zero_diff
  intro geometric_seq
  intro gen_term_cond
  intro sum_terms_cond
  intro b_def_cond
  intro sum_b_terms_cond
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_problem_statement_l1442_144265


namespace NUMINAMATH_GPT_blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l1442_144234

variables (length magnitude : ℕ)
variable (price : ℝ)
variable (area : ℕ)

-- Definitions based on the conditions
def length_is_about_4 (length : ℕ) : Prop := length = 4
def price_is_about_9_50 (price : ℝ) : Prop := price = 9.50
def large_area_is_about_3 (area : ℕ) : Prop := area = 3
def small_area_is_about_1 (area : ℕ) : Prop := area = 1

-- Proof problem statements
theorem blackboard_length_is_meters : length_is_about_4 length → length = 4 := by sorry
theorem pencil_case_price_is_yuan : price_is_about_9_50 price → price = 9.50 := by sorry
theorem campus_area_is_hectares : large_area_is_about_3 area → area = 3 := by sorry
theorem fingernail_area_is_square_centimeters : small_area_is_about_1 area → area = 1 := by sorry

end NUMINAMATH_GPT_blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l1442_144234


namespace NUMINAMATH_GPT_divides_mn_minus_one_l1442_144235

theorem divides_mn_minus_one (m n p : ℕ) (hp : p.Prime) (h1 : m < n) (h2 : n < p) 
    (hm2 : p ∣ m^2 + 1) (hn2 : p ∣ n^2 + 1) : p ∣ m * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_divides_mn_minus_one_l1442_144235


namespace NUMINAMATH_GPT_totalGamesPlayed_l1442_144218

def numPlayers : ℕ := 30

def numGames (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem totalGamesPlayed :
  numGames numPlayers = 435 :=
by
  sorry

end NUMINAMATH_GPT_totalGamesPlayed_l1442_144218


namespace NUMINAMATH_GPT_area_bounded_by_curves_eq_l1442_144289

open Real

noncomputable def area_bounded_by_curves : ℝ :=
  1 / 2 * (∫ (φ : ℝ) in (π/4)..(π/2), (sqrt 2 * cos (φ - π / 4))^2) +
  1 / 2 * (∫ (φ : ℝ) in (π/2)..(3 * π / 4), (sqrt 2 * sin (φ - π / 4))^2)

theorem area_bounded_by_curves_eq : area_bounded_by_curves = (π + 2) / 4 :=
  sorry

end NUMINAMATH_GPT_area_bounded_by_curves_eq_l1442_144289


namespace NUMINAMATH_GPT_number_divided_by_189_l1442_144225

noncomputable def target_number : ℝ := 3486

theorem number_divided_by_189 :
  target_number / 189 = 18.444444444444443 :=
by
  sorry

end NUMINAMATH_GPT_number_divided_by_189_l1442_144225
