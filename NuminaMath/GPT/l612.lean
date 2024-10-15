import Mathlib

namespace NUMINAMATH_GPT_polynomial_addition_l612_61268

variable (x : ℝ)

def p := 3 * x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 2
def q := -3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 4

theorem polynomial_addition : p x + q x = -3 * x^3 + 2 * x^2 + 2 := by
  sorry

end NUMINAMATH_GPT_polynomial_addition_l612_61268


namespace NUMINAMATH_GPT_tomatoes_planted_each_kind_l612_61267

-- Definitions derived from Conditions
def total_rows : ℕ := 10
def spaces_per_row : ℕ := 15
def kinds_of_tomatoes : ℕ := 3
def kinds_of_cucumbers : ℕ := 5
def cucumbers_per_kind : ℕ := 4
def potatoes : ℕ := 30
def available_spaces : ℕ := 85

-- Theorem statement with the question and answer derived from the problem
theorem tomatoes_planted_each_kind : (kinds_of_tomatoes * (total_rows * spaces_per_row - Available_spaces - (kinds_of_cucumbers * cucumbers_per_kind + potatoes)) / kinds_of_tomatoes) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_tomatoes_planted_each_kind_l612_61267


namespace NUMINAMATH_GPT_range_f_compare_sizes_final_comparison_l612_61290

noncomputable def f (x : ℝ) := |2 * x - 1| + |x + 1|

theorem range_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ∈ Set.Ici (3 / 2)} :=
sorry

theorem compare_sizes (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
sorry

theorem final_comparison (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
by
  exact compare_sizes a ha

end NUMINAMATH_GPT_range_f_compare_sizes_final_comparison_l612_61290


namespace NUMINAMATH_GPT_find_multiple_l612_61262

-- Definitions and given conditions
def total_seats : ℤ := 387
def first_class_seats : ℤ := 77

-- The statement we need to prove
theorem find_multiple (m : ℤ) :
  (total_seats = first_class_seats + (m * first_class_seats + 2)) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l612_61262


namespace NUMINAMATH_GPT_solve_seating_problem_l612_61275

-- Define the conditions of the problem
def valid_seating_arrangements (n : ℕ) : Prop :=
  (∃ (x y : ℕ), x < y ∧ x + 1 < y ∧ y < n ∧ 
    (n ≥ 5 ∧ y - x - 1 > 0)) ∧
  (∃! (x' y' : ℕ), x' < y' ∧ x' + 1 < y' ∧ y' < n ∧ 
    (n ≥ 5 ∧ y' - x' - 1 > 0))

-- State the theorem
theorem solve_seating_problem : ∃ n : ℕ, valid_seating_arrangements n ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_seating_problem_l612_61275


namespace NUMINAMATH_GPT_length_of_bridge_is_correct_l612_61259

noncomputable def train_length : ℝ := 150
noncomputable def crossing_time : ℝ := 29.997600191984642
noncomputable def train_speed_kmph : ℝ := 36
noncomputable def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 149.97600191984642 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_correct_l612_61259


namespace NUMINAMATH_GPT_find_ab_exponent_l612_61296

theorem find_ab_exponent (a b : ℝ) 
  (h : |a - 2| + (b + 1 / 2)^2 = 0) : 
  a^2022 * b^2023 = -1 / 2 := 
sorry

end NUMINAMATH_GPT_find_ab_exponent_l612_61296


namespace NUMINAMATH_GPT_negation_of_existence_l612_61297

theorem negation_of_existence (h: ∃ x : ℝ, 0 < x ∧ (Real.log x + x - 1 ≤ 0)) :
  ¬ (∀ x : ℝ, 0 < x → ¬ (Real.log x + x - 1 ≤ 0)) :=
sorry

end NUMINAMATH_GPT_negation_of_existence_l612_61297


namespace NUMINAMATH_GPT_coeff_x2_in_expansion_l612_61236

theorem coeff_x2_in_expansion : 
  (2 : ℚ) - (1 / x) * ((1 + x)^6)^(2 : ℤ) = (10 : ℚ) :=
by sorry

end NUMINAMATH_GPT_coeff_x2_in_expansion_l612_61236


namespace NUMINAMATH_GPT_michael_pays_106_l612_61238

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def num_parrots : ℕ := 1
def num_fish : ℕ := 4

def cost_per_cat : ℕ := 13
def cost_per_dog : ℕ := 18
def cost_per_parrot : ℕ := 10
def cost_per_fish : ℕ := 4

def total_cost : ℕ :=
  (num_cats * cost_per_cat) +
  (num_dogs * cost_per_dog) +
  (num_parrots * cost_per_parrot) +
  (num_fish * cost_per_fish)

theorem michael_pays_106 : total_cost = 106 := by
  sorry

end NUMINAMATH_GPT_michael_pays_106_l612_61238


namespace NUMINAMATH_GPT_max_num_triangles_for_right_triangle_l612_61219

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_num_triangles_for_right_triangle_l612_61219


namespace NUMINAMATH_GPT_value_of_m_l612_61217

theorem value_of_m (m : ℝ) (h1 : m^2 - 2 * m - 1 = 2) (h2 : m ≠ 3) : m = -1 :=
sorry

end NUMINAMATH_GPT_value_of_m_l612_61217


namespace NUMINAMATH_GPT_green_block_weight_l612_61220

theorem green_block_weight
    (y : ℝ)
    (g : ℝ)
    (h1 : y = 0.6)
    (h2 : y = g + 0.2) :
    g = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_green_block_weight_l612_61220


namespace NUMINAMATH_GPT_johnson_oldest_child_age_l612_61295

/-- The average age of the three Johnson children is 10 years. 
    The two younger children are 6 years old and 8 years old. 
    Prove that the age of the oldest child is 16 years. -/
theorem johnson_oldest_child_age :
  ∃ x : ℕ, (6 + 8 + x) / 3 = 10 ∧ x = 16 :=
by
  sorry

end NUMINAMATH_GPT_johnson_oldest_child_age_l612_61295


namespace NUMINAMATH_GPT_larger_number_is_42_l612_61200

theorem larger_number_is_42 (x y : ℕ) (h1 : x + y = 77) (h2 : 5 * x = 6 * y) : x = 42 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_42_l612_61200


namespace NUMINAMATH_GPT_selling_price_conditions_met_l612_61253

-- Definitions based on the problem conditions
def initial_selling_price : ℝ := 50
def purchase_price : ℝ := 40
def initial_volume : ℝ := 500
def decrease_rate : ℝ := 10
def desired_profit : ℝ := 8000
def max_total_cost : ℝ := 10000

-- Definition for the selling price
def selling_price : ℝ := 80

-- Condition: Cost is below $10000 for the valid selling price
def valid_item_count (x : ℝ) : ℝ := initial_volume - decrease_rate * (x - initial_selling_price)

-- Cost calculation function
def total_cost (x : ℝ) : ℝ := purchase_price * valid_item_count x

-- Profit calculation function 
def profit (x : ℝ) : ℝ := (x - purchase_price) * valid_item_count x

-- Main theorem statement
theorem selling_price_conditions_met : 
  profit selling_price = desired_profit ∧ total_cost selling_price < max_total_cost :=
by
  sorry

end NUMINAMATH_GPT_selling_price_conditions_met_l612_61253


namespace NUMINAMATH_GPT_convert_BFACE_to_decimal_l612_61292

def hex_BFACE : ℕ := 11 * 16^4 + 15 * 16^3 + 10 * 16^2 + 12 * 16^1 + 14 * 16^0

theorem convert_BFACE_to_decimal : hex_BFACE = 785102 := by
  sorry

end NUMINAMATH_GPT_convert_BFACE_to_decimal_l612_61292


namespace NUMINAMATH_GPT_total_money_made_l612_61203

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end NUMINAMATH_GPT_total_money_made_l612_61203


namespace NUMINAMATH_GPT_xyz_product_neg4_l612_61223

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_xyz_product_neg4_l612_61223


namespace NUMINAMATH_GPT_findLineEquation_l612_61240

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to represent the hyperbola condition
def isOnHyperbola (pt : Point) : Prop :=
  pt.x ^ 2 - 4 * pt.y ^ 2 = 4

-- Define midpoint condition for points A and B
def isMidpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define points
def P : Point := ⟨8, 1⟩
def A : Point := sorry
def B : Point := sorry

-- Statement to prove
theorem findLineEquation :
  isOnHyperbola A ∧ isOnHyperbola B ∧ isMidpoint P A B →
  ∃ m b, (∀ pt : Point, pt.y = m * pt.x + b ↔ pt.x = 8 ∧ pt.y = 1) ∧ (m = 2) ∧ (b = -15) :=
by
  sorry

end NUMINAMATH_GPT_findLineEquation_l612_61240


namespace NUMINAMATH_GPT_girls_with_short_hair_count_l612_61222

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end NUMINAMATH_GPT_girls_with_short_hair_count_l612_61222


namespace NUMINAMATH_GPT_binom_coeff_divisibility_l612_61211

theorem binom_coeff_divisibility (p : ℕ) (hp : Prime p) : Nat.choose (2 * p) p - 2 ≡ 0 [MOD p^2] := 
sorry

end NUMINAMATH_GPT_binom_coeff_divisibility_l612_61211


namespace NUMINAMATH_GPT_top_and_bottom_area_each_l612_61255

def long_side_area : ℕ := 2 * 8 * 6
def short_side_area : ℕ := 2 * 5 * 6
def total_sides_area : ℕ := long_side_area + short_side_area
def total_needed_area : ℕ := 236
def top_and_bottom_area : ℕ := total_needed_area - total_sides_area

theorem top_and_bottom_area_each :
  top_and_bottom_area / 2 = 40 := by
  sorry

end NUMINAMATH_GPT_top_and_bottom_area_each_l612_61255


namespace NUMINAMATH_GPT_evie_collected_shells_for_6_days_l612_61224

theorem evie_collected_shells_for_6_days (d : ℕ) (h1 : 10 * d - 2 = 58) : d = 6 := by
  sorry

end NUMINAMATH_GPT_evie_collected_shells_for_6_days_l612_61224


namespace NUMINAMATH_GPT_six_cube_2d_faces_count_l612_61201

open BigOperators

theorem six_cube_2d_faces_count :
    let vertices := 64
    let edges_1d := 192
    let edges_2d := 240
    let small_cubes := 46656
    let faces_per_plane := 36
    let planes_count := 15 * 7^4
    faces_per_plane * planes_count = 1296150 := by
  sorry

end NUMINAMATH_GPT_six_cube_2d_faces_count_l612_61201


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_l612_61221

-- Define initial conditions
def initial_balls : ℕ := 6
def total_balls_after_white : ℕ := initial_balls + 1
def number_of_white_balls : ℕ := 1
def number_of_total_balls : ℕ := total_balls_after_white

-- Define the probability of drawing a white ball
def probability_of_white : ℚ := number_of_white_balls / number_of_total_balls

-- Statement to be proved
theorem probability_of_drawing_white_ball :
  probability_of_white = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_l612_61221


namespace NUMINAMATH_GPT_father_son_age_ratio_l612_61269

theorem father_son_age_ratio :
  ∃ S : ℕ, (45 = S + 15 * 2) ∧ (45 / S = 3) := 
sorry

end NUMINAMATH_GPT_father_son_age_ratio_l612_61269


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l612_61250

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l612_61250


namespace NUMINAMATH_GPT_distinct_real_roots_imply_sum_greater_than_two_l612_61213

noncomputable def function_f (x: ℝ) : ℝ := abs (Real.log x)

theorem distinct_real_roots_imply_sum_greater_than_two {k α β : ℝ} 
  (h₁ : function_f α = k) 
  (h₂ : function_f β = k) 
  (h₃ : α ≠ β) 
  (h4 : 0 < α ∧ α < 1)
  (h5 : 1 < β) :
  (1 / α) + (1 / β) > 2 :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_imply_sum_greater_than_two_l612_61213


namespace NUMINAMATH_GPT_addition_results_in_perfect_square_l612_61289

theorem addition_results_in_perfect_square : ∃ n: ℕ, n * n = 4440 + 49 :=
by
  sorry

end NUMINAMATH_GPT_addition_results_in_perfect_square_l612_61289


namespace NUMINAMATH_GPT_gallons_of_soup_l612_61288

def bowls_per_minute : ℕ := 5
def ounces_per_bowl : ℕ := 10
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem gallons_of_soup :
  (5 * 10 * 15 / 128) = 6 :=
by
  sorry

end NUMINAMATH_GPT_gallons_of_soup_l612_61288


namespace NUMINAMATH_GPT_molly_more_minutes_than_xanthia_l612_61285

-- Define the constants: reading speeds and book length
def xanthia_speed := 80  -- pages per hour
def molly_speed := 40    -- pages per hour
def book_length := 320   -- pages

-- Define the times taken to read the book in hours
def xanthia_time := book_length / xanthia_speed
def molly_time := book_length / molly_speed

-- Define the time difference in minutes
def time_difference_minutes := (molly_time - xanthia_time) * 60

theorem molly_more_minutes_than_xanthia : time_difference_minutes = 240 := 
by {
  -- Here the proof would go, but we'll leave it as a sorry for now.
  sorry
}

end NUMINAMATH_GPT_molly_more_minutes_than_xanthia_l612_61285


namespace NUMINAMATH_GPT_mixed_number_division_l612_61247

theorem mixed_number_division :
  (5 + 1 / 2 - (2 + 2 / 3)) / (1 + 1 / 5 + 3 + 1 / 4) = 0 + 170 / 267 := 
by
  sorry

end NUMINAMATH_GPT_mixed_number_division_l612_61247


namespace NUMINAMATH_GPT_benny_missed_games_l612_61246

def total_games : ℕ := 39
def attended_games : ℕ := 14
def missed_games : ℕ := total_games - attended_games

theorem benny_missed_games : missed_games = 25 := by
  sorry

end NUMINAMATH_GPT_benny_missed_games_l612_61246


namespace NUMINAMATH_GPT_missed_angle_l612_61216

def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem missed_angle :
  ∃ (n : ℕ), sum_interior_angles n = 3060 ∧ 3060 - 2997 = 63 :=
by {
  sorry
}

end NUMINAMATH_GPT_missed_angle_l612_61216


namespace NUMINAMATH_GPT_avg_class_weight_l612_61261

def num_students_A : ℕ := 24
def num_students_B : ℕ := 16
def avg_weight_A : ℕ := 40
def avg_weight_B : ℕ := 35

/-- Theorem: The average weight of the whole class is 38 kg --/
theorem avg_class_weight :
  (num_students_A * avg_weight_A + num_students_B * avg_weight_B) / (num_students_A + num_students_B) = 38 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_avg_class_weight_l612_61261


namespace NUMINAMATH_GPT_roots_difference_squared_quadratic_roots_property_l612_61251

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_roots_difference_squared_quadratic_roots_property_l612_61251


namespace NUMINAMATH_GPT_find_first_number_l612_61202

theorem find_first_number (x : ℤ) (k : ℤ) :
  (29 > 0) ∧ (x % 29 = 8) ∧ (1490 % 29 = 11) → x = 29 * k + 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_first_number_l612_61202


namespace NUMINAMATH_GPT_quadratic_solutions_l612_61277

-- Definition of the conditions given in the problem
def quadratic_axis_symmetry (b : ℝ) : Prop :=
  -b / 2 = 2

def equation_solutions (x b : ℝ) : Prop :=
  x^2 + b*x - 5 = 2*x - 13

-- The math proof problem statement in Lean 4
theorem quadratic_solutions (b : ℝ) (x1 x2 : ℝ) :
  quadratic_axis_symmetry b →
  equation_solutions x1 b →
  equation_solutions x2 b →
  (x1 = 2 ∧ x2 = 4) ∨ (x1 = 4 ∧ x2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solutions_l612_61277


namespace NUMINAMATH_GPT_remainder_polynomial_division_l612_61228

noncomputable def remainder_division : Polynomial ℝ := 
  (Polynomial.X ^ 4 + Polynomial.X ^ 3 - 4 * Polynomial.X + 1) % (Polynomial.X ^ 3 - 1)

theorem remainder_polynomial_division :
  remainder_division = -3 * Polynomial.X + 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_polynomial_division_l612_61228


namespace NUMINAMATH_GPT_three_digit_number_unchanged_upside_down_l612_61280

theorem three_digit_number_unchanged_upside_down (n : ℕ) :
  (n >= 100 ∧ n <= 999) ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 0 ∨ d = 8) ->
  n = 888 ∨ n = 808 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_unchanged_upside_down_l612_61280


namespace NUMINAMATH_GPT_product_of_solutions_l612_61282

theorem product_of_solutions (a b c x : ℝ) (h1 : -x^2 - 4 * x + 10 = 0) :
  x * (-4 - x) = -10 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l612_61282


namespace NUMINAMATH_GPT_four_digit_even_numbers_count_and_sum_l612_61229

variable (digits : Set ℕ) (used_once : ∀ d ∈ digits, d ≤ 6 ∧ d ≥ 1)

theorem four_digit_even_numbers_count_and_sum
  (hyp : digits = {1, 2, 3, 4, 5, 6}) :
  ∃ (N M : ℕ), 
    (N = 180 ∧ M = 680040) := 
sorry

end NUMINAMATH_GPT_four_digit_even_numbers_count_and_sum_l612_61229


namespace NUMINAMATH_GPT_correct_product_l612_61293

theorem correct_product : 0.125 * 5.12 = 0.64 := sorry

end NUMINAMATH_GPT_correct_product_l612_61293


namespace NUMINAMATH_GPT_max_value_of_expression_l612_61212

variable (a b c : ℝ)

theorem max_value_of_expression : 
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c := by
sorry

end NUMINAMATH_GPT_max_value_of_expression_l612_61212


namespace NUMINAMATH_GPT_seating_arrangement_l612_61241

theorem seating_arrangement (n : ℕ) (h1 : n * 9 + (100 - n) * 10 = 100) : n = 10 :=
by sorry

end NUMINAMATH_GPT_seating_arrangement_l612_61241


namespace NUMINAMATH_GPT_cuboid_surface_area_increase_l612_61225

variables (L W H : ℝ)
def SA_original (L W H : ℝ) : ℝ := 2 * (L * W + L * H + W * H)

def SA_new (L W H : ℝ) : ℝ := 2 * ((1.50 * L) * (1.70 * W) + (1.50 * L) * (1.80 * H) + (1.70 * W) * (1.80 * H))

theorem cuboid_surface_area_increase :
  (SA_new L W H - SA_original L W H) / SA_original L W H * 100 = 315.5 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_increase_l612_61225


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_and_5_l612_61214

theorem smallest_positive_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ (2 ∣ n) ∧ (5 ∣ n) ∧ n = 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_divisible_by_2_and_5_l612_61214


namespace NUMINAMATH_GPT_calculate_expression_l612_61294

theorem calculate_expression (y : ℝ) : (20 * y^3) * (7 * y^2) * (1 / (2 * y)^3) = 17.5 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l612_61294


namespace NUMINAMATH_GPT_amount_left_after_spending_l612_61235

-- Define the initial amount and percentage spent
def initial_amount : ℝ := 500
def percentage_spent : ℝ := 0.30

-- Define the proof statement that the amount left is 350
theorem amount_left_after_spending : 
  (initial_amount - (percentage_spent * initial_amount)) = 350 :=
by
  sorry

end NUMINAMATH_GPT_amount_left_after_spending_l612_61235


namespace NUMINAMATH_GPT_program_outputs_all_divisors_l612_61249

/--
  The function of the program is to output all divisors of \( n \), 
  given the initial conditions and operations in the program.
 -/
theorem program_outputs_all_divisors (n : ℕ) :
  ∀ I : ℕ, (1 ≤ I ∧ I ≤ n) → (∃ S : ℕ, (n % I = 0 ∧ S = I)) :=
by
  sorry

end NUMINAMATH_GPT_program_outputs_all_divisors_l612_61249


namespace NUMINAMATH_GPT_starting_number_divisible_by_3_count_l612_61281

-- Define a predicate for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define the main theorem
theorem starting_number_divisible_by_3_count : 
  ∃ n : ℕ, (∀ m, n ≤ m ∧ m ≤ 50 → divisible_by_3 m → ∃ s, (m = n + 3 * s) ∧ s < 13) ∧
           (∀ k : ℕ, (divisible_by_3 k) → n ≤ k ∧ k ≤ 50 → m = 12) :=
sorry

end NUMINAMATH_GPT_starting_number_divisible_by_3_count_l612_61281


namespace NUMINAMATH_GPT_school_total_students_l612_61232

theorem school_total_students (T G : ℕ) (h1 : 80 + G = T) (h2 : G = (80 * T) / 100) : T = 400 :=
by
  sorry

end NUMINAMATH_GPT_school_total_students_l612_61232


namespace NUMINAMATH_GPT_option_C_correct_l612_61209

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end NUMINAMATH_GPT_option_C_correct_l612_61209


namespace NUMINAMATH_GPT_problem_statement_l612_61205

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def beta : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := alpha^(500)
noncomputable def N : ℝ := alpha^(500) + beta^(500)
noncomputable def n : ℝ := N - 1
noncomputable def f : ℝ := x - n
noncomputable def one_minus_f : ℝ := 1 - f

theorem problem_statement : x * one_minus_f = 1 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_problem_statement_l612_61205


namespace NUMINAMATH_GPT_max_d_is_9_l612_61286

-- Define the 6-digit number of the form 8d8, 45e
def num (d e : ℕ) : ℕ :=
  800000 + 10000 * d + 800 + 450 + e

-- Define the conditions: the number is a multiple of 45, 0 ≤ d, e ≤ 9
def conditions (d e : ℕ) : Prop :=
  0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
  (num d e) % 45 = 0

-- Define the maximum value of d
noncomputable def max_d : ℕ :=
  9

-- The theorem statement to be proved
theorem max_d_is_9 :
  ∀ (d e : ℕ), conditions d e → d ≤ max_d :=
by
  sorry

end NUMINAMATH_GPT_max_d_is_9_l612_61286


namespace NUMINAMATH_GPT_count_multiples_of_30_l612_61256

theorem count_multiples_of_30 (a b n : ℕ) (h1 : a = 900) (h2 : b = 27000) 
    (h3 : ∃ n, 30 * n = a) (h4 : ∃ n, 30 * n = b) : 
    (b - a) / 30 + 1 = 871 := 
by
    sorry

end NUMINAMATH_GPT_count_multiples_of_30_l612_61256


namespace NUMINAMATH_GPT_ice_cream_cone_cost_is_5_l612_61254

noncomputable def cost_of_ice_cream_cone (x : ℝ) : Prop := 
  let total_cost_of_cones := 15 * x
  let total_cost_of_puddings := 5 * 2
  let extra_spent_on_cones := total_cost_of_cones - total_cost_of_puddings
  extra_spent_on_cones = 65

theorem ice_cream_cone_cost_is_5 : ∃ x : ℝ, cost_of_ice_cream_cone x ∧ x = 5 :=
by 
  use 5
  unfold cost_of_ice_cream_cone
  simp
  sorry

end NUMINAMATH_GPT_ice_cream_cone_cost_is_5_l612_61254


namespace NUMINAMATH_GPT_free_fall_height_and_last_second_distance_l612_61239

theorem free_fall_height_and_last_second_distance :
  let time := 11
  let initial_distance := 4.9
  let increment := 9.8
  let total_height := (initial_distance * time + increment * (time * (time - 1)) / 2)
  let last_second_distance := initial_distance + increment * (time - 1)
  total_height = 592.9 ∧ last_second_distance = 102.9 :=
by
  sorry

end NUMINAMATH_GPT_free_fall_height_and_last_second_distance_l612_61239


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l612_61271

-- Simplified and combined statements for clarity
theorem problem_1 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  f 3 + f (-1) = -3 := sorry

theorem problem_2 (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1)) : 
  ∀ x, f x = if x ≤ 0 then Real.logb (1/2) (-x + 1) else Real.logb (1/2) (x + 1) := sorry

theorem problem_3 (f : ℝ → ℝ) (h_cond : ∀ x ≤ 0, f x = Real.logb (1/2) (-x + 1))
  (h_cond_ev : ∀ x, f x = f (-x)) (a : ℝ) : 
  f (a - 1) < -1 ↔ a ∈ ((Set.Iio 0) ∪ (Set.Ioi 2)) := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l612_61271


namespace NUMINAMATH_GPT_train_speed_l612_61226

theorem train_speed :
  ∃ V : ℝ,
    (∃ L : ℝ, L = V * 18) ∧ 
    (∃ L : ℝ, L + 260 = V * 31) ∧ 
    V * 3.6 = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_l612_61226


namespace NUMINAMATH_GPT_parallelogram_height_l612_61283

theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 336) 
  (h_base : base = 14) 
  (h_formula : area = base * height) : 
  height = 24 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_height_l612_61283


namespace NUMINAMATH_GPT_elberta_money_l612_61243

theorem elberta_money (GrannySmith Anjou Elberta : ℝ)
  (h_granny : GrannySmith = 100)
  (h_anjou : Anjou = 1 / 4 * GrannySmith)
  (h_elberta : Elberta = Anjou + 5) : Elberta = 30 := by
  sorry

end NUMINAMATH_GPT_elberta_money_l612_61243


namespace NUMINAMATH_GPT_one_third_of_four_l612_61230

theorem one_third_of_four (h : 1/6 * 20 = 15) : 1/3 * 4 = 10 :=
sorry

end NUMINAMATH_GPT_one_third_of_four_l612_61230


namespace NUMINAMATH_GPT_length_of_bridge_l612_61233

noncomputable def train_length : ℝ := 155
noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_m_s * crossing_time_seconds

noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge : bridge_length = 220 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l612_61233


namespace NUMINAMATH_GPT_find_triplets_l612_61272

theorem find_triplets (a k m : ℕ) (hpos_a : 0 < a) (hpos_k : 0 < k) (hpos_m : 0 < m) (h_eq : k + a^k = m + 2 * a^m) :
  ∃ t : ℕ, 0 < t ∧ (a = 1 ∧ k = t + 1 ∧ m = t) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l612_61272


namespace NUMINAMATH_GPT_sin_double_angle_tangent_identity_l612_61231

theorem sin_double_angle_tangent_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.sin (2 * x) = 3 / 5 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_sin_double_angle_tangent_identity_l612_61231


namespace NUMINAMATH_GPT_number_of_f3_and_sum_of_f3_l612_61266

noncomputable def f : ℝ → ℝ := sorry
variable (a : ℝ)

theorem number_of_f3_and_sum_of_f3 (hf : ∀ x y : ℝ, f (f x - y) = f x + f (f y - f a) + x) :
  (∃! c : ℝ, f 3 = c) ∧ (∃ s : ℝ, (∀ c, f 3 = c → s = c) ∧ s = 3) :=
sorry

end NUMINAMATH_GPT_number_of_f3_and_sum_of_f3_l612_61266


namespace NUMINAMATH_GPT_triangle_is_equilateral_l612_61245

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)

-- Define a triangle's circumradius and inradius
structure TriangleProperties :=
  (circumradius : ℝ)
  (inradius : ℝ)
  (circumcenter_incenter_sq_distance : ℝ) -- OI^2 = circumradius^2 - 2*circumradius*inradius

noncomputable def circumcenter_incenter_coincide (T : Triangle) (P : TriangleProperties) : Prop :=
  P.circumcenter_incenter_sq_distance = 0

theorem triangle_is_equilateral
  (T : Triangle)
  (P : TriangleProperties)
  (hR : P.circumradius = 2 * P.inradius)
  (hOI : circumcenter_incenter_coincide T P) :
  ∃ (R r : ℝ), T = {A := 1 * r, B := 1 * r, C := 1 * r} :=
by sorry

end NUMINAMATH_GPT_triangle_is_equilateral_l612_61245


namespace NUMINAMATH_GPT_cost_price_l612_61257

theorem cost_price (MP SP C : ℝ) (h1 : MP = 74.21875)
  (h2 : SP = MP - 0.20 * MP)
  (h3 : SP = 1.25 * C) : C = 47.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_l612_61257


namespace NUMINAMATH_GPT_find_ratio_l612_61276

theorem find_ratio (f : ℝ → ℝ) (h : ∀ a b : ℝ, b^2 * f a = a^2 * f b) (h3 : f 3 ≠ 0) :
  (f 7 - f 3) / f 3 = 40 / 9 :=
sorry

end NUMINAMATH_GPT_find_ratio_l612_61276


namespace NUMINAMATH_GPT_total_votes_cast_correct_l612_61227

noncomputable def total_votes_cast : Nat :=
  let total_valid_votes : Nat := 1050
  let spoiled_votes : Nat := 325
  total_valid_votes + spoiled_votes

theorem total_votes_cast_correct :
  total_votes_cast = 1375 := by
  sorry

end NUMINAMATH_GPT_total_votes_cast_correct_l612_61227


namespace NUMINAMATH_GPT_cube_sphere_volume_ratio_l612_61252

theorem cube_sphere_volume_ratio (s : ℝ) (r : ℝ) (h : r = (Real.sqrt 3 * s) / 2):
  (s^3) / ((4 / 3) * Real.pi * r^3) = (2 * Real.sqrt 3) / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cube_sphere_volume_ratio_l612_61252


namespace NUMINAMATH_GPT_partial_fraction_sum_equals_251_l612_61242

theorem partial_fraction_sum_equals_251 (p q r A B C : ℝ) :
  (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧
  (∀ s : ℝ, (s ≠ p) ∧ (s ≠ q) ∧ (s ≠ r) →
  1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (p + q + r = 24) →
  (p * q + p * r + q * r = 151) →
  (p * q * r = 650) →
  (1 / A + 1 / B + 1 / C = 251) :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_sum_equals_251_l612_61242


namespace NUMINAMATH_GPT_proof_theorem_l612_61204

noncomputable def proof_problem : Prop :=
  let a := 6
  let b := 15
  let c := 7
  let lhs := a * b * c
  let rhs := (Real.sqrt ((a^2) + (2 * a) + (b^3) - (b^2) + (3 * b))) / (c^2 + c + 1) + 629.001
  lhs = rhs

theorem proof_theorem : proof_problem :=
  by
  sorry

end NUMINAMATH_GPT_proof_theorem_l612_61204


namespace NUMINAMATH_GPT_correct_option_is_D_l612_61287

noncomputable def option_A := 230
noncomputable def option_B := [251, 260]
noncomputable def option_B_average := 256
noncomputable def option_C := [21, 212, 256]
noncomputable def option_C_average := 163
noncomputable def option_D := [210, 240, 250]
noncomputable def option_D_average := 233

theorem correct_option_is_D :
  ∃ (correct_option : String), correct_option = "D" :=
  sorry

end NUMINAMATH_GPT_correct_option_is_D_l612_61287


namespace NUMINAMATH_GPT_arithmetic_evaluation_l612_61298

theorem arithmetic_evaluation :
  -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l612_61298


namespace NUMINAMATH_GPT_strawberries_weight_l612_61270

theorem strawberries_weight (total_weight apples_weight oranges_weight grapes_weight strawberries_weight : ℕ) 
  (h_total : total_weight = 10)
  (h_apples : apples_weight = 3)
  (h_oranges : oranges_weight = 1)
  (h_grapes : grapes_weight = 3) 
  (h_sum : total_weight = apples_weight + oranges_weight + grapes_weight + strawberries_weight) :
  strawberries_weight = 3 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_weight_l612_61270


namespace NUMINAMATH_GPT_length_of_plot_l612_61264

theorem length_of_plot (total_poles : ℕ) (distance : ℕ) (one_side : ℕ) (other_side : ℕ) 
  (poles_distance_condition : total_poles = 28) 
  (fencing_condition : distance = 10) 
  (side_condition : one_side = 50) 
  (rectangular_condition : total_poles = (2 * (one_side / distance) + 2 * (other_side / distance))) :
  other_side = 120 :=
by
  sorry

end NUMINAMATH_GPT_length_of_plot_l612_61264


namespace NUMINAMATH_GPT_laura_park_time_l612_61265

theorem laura_park_time
  (T : ℝ) -- Time spent at the park each trip in hours
  (walk_time : ℝ := 0.5) -- Time spent walking to and from the park each trip in hours
  (trips : ℕ := 6) -- Total number of trips
  (park_time_percentage : ℝ := 0.80) -- Percentage of total time spent at the park
  (total_park_time_eq : trips * T = park_time_percentage * (trips * (T + walk_time))) :
  T = 2 :=
by
  sorry

end NUMINAMATH_GPT_laura_park_time_l612_61265


namespace NUMINAMATH_GPT_mass_percentage_of_nitrogen_in_N2O5_l612_61263

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)

theorem mass_percentage_of_nitrogen_in_N2O5 : 
  (2 * atomic_mass_N / molar_mass_N2O5 * 100) = 25.94 := 
by 
  sorry

end NUMINAMATH_GPT_mass_percentage_of_nitrogen_in_N2O5_l612_61263


namespace NUMINAMATH_GPT_build_wall_30_persons_l612_61210

-- Defining the conditions
def work_rate (persons : ℕ) (days : ℕ) : ℚ := 1 / (persons * days)

-- Total work required to build the wall by 8 persons in 42 days
def total_work : ℚ := work_rate 8 42 * 8 * 42

-- Work rate for 30 persons
def combined_work_rate (persons : ℕ) : ℚ := persons * work_rate 8 42

-- Days required for 30 persons to complete the same work
def days_required (persons : ℕ) (work : ℚ) : ℚ := work / combined_work_rate persons

-- Expected result is 11.2 days for 30 persons
theorem build_wall_30_persons : days_required 30 total_work = 11.2 := 
by
  sorry

end NUMINAMATH_GPT_build_wall_30_persons_l612_61210


namespace NUMINAMATH_GPT_line_through_perpendicular_l612_61215

theorem line_through_perpendicular (x y : ℝ) :
  (∃ (k : ℝ), (2 * x - y + 3 = 0) ∧ k = - 1 / 2) →
  (∃ (a b c : ℝ), (a * (-1) + b * 1 + c = 0) ∧ a = 1 ∧ b = 2 ∧ c = -1) :=
by
  sorry

end NUMINAMATH_GPT_line_through_perpendicular_l612_61215


namespace NUMINAMATH_GPT_angle_sum_in_triangle_l612_61237

theorem angle_sum_in_triangle (A B C : ℝ) (h₁ : A + B = 90) (h₂ : A + B + C = 180) : C = 90 := by
  sorry

end NUMINAMATH_GPT_angle_sum_in_triangle_l612_61237


namespace NUMINAMATH_GPT_average_age_of_three_l612_61207

theorem average_age_of_three (Kimiko_age : ℕ) (Omi_age : ℕ) (Arlette_age : ℕ) 
  (h1 : Omi_age = 2 * Kimiko_age) 
  (h2 : Arlette_age = (3 * Kimiko_age) / 4) 
  (h3 : Kimiko_age = 28) : 
  (Kimiko_age + Omi_age + Arlette_age) / 3 = 35 := 
  by sorry

end NUMINAMATH_GPT_average_age_of_three_l612_61207


namespace NUMINAMATH_GPT_unique_solution_of_abc_l612_61248

theorem unique_solution_of_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_lt_ab_c : a < b) (h_lt_b_c: b < c) (h_eq_abc : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 :=
by {
  -- Proof skipped, only the statement is provided.
  sorry
}

end NUMINAMATH_GPT_unique_solution_of_abc_l612_61248


namespace NUMINAMATH_GPT_chord_length_is_correct_l612_61274

noncomputable def length_of_chord {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : Real :=
  2 * Real.sqrt 3

theorem chord_length_is_correct {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : 
 length_of_chord h_line h_curve = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_chord_length_is_correct_l612_61274


namespace NUMINAMATH_GPT_calculate_expression_l612_61206

theorem calculate_expression:
  500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l612_61206


namespace NUMINAMATH_GPT_polynomial_root_sum_l612_61279

theorem polynomial_root_sum 
  (c d : ℂ) 
  (h1 : c + d = 6) 
  (h2 : c * d = 10) 
  (h3 : c^2 - 6 * c + 10 = 0) 
  (h4 : d^2 - 6 * d + 10 = 0) : 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16156 := 
by sorry

end NUMINAMATH_GPT_polynomial_root_sum_l612_61279


namespace NUMINAMATH_GPT_number_of_people_eating_both_l612_61244

variable (A B C : Nat)

theorem number_of_people_eating_both (hA : A = 13) (hB : B = 19) (hC : C = B - A) : C = 6 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_people_eating_both_l612_61244


namespace NUMINAMATH_GPT_box_filling_rate_l612_61258

theorem box_filling_rate (l w h t : ℝ) (hl : l = 7) (hw : w = 6) (hh : h = 2) (ht : t = 21) : 
  (l * w * h) / t = 4 := by
  sorry

end NUMINAMATH_GPT_box_filling_rate_l612_61258


namespace NUMINAMATH_GPT_card_game_total_l612_61234

theorem card_game_total (C E O : ℝ) (h1 : E = (11 / 20) * C) (h2 : O = (9 / 20) * C) (h3 : E = O + 50) : C = 500 :=
sorry

end NUMINAMATH_GPT_card_game_total_l612_61234


namespace NUMINAMATH_GPT_geometric_sequence_general_formula_l612_61273

noncomputable def a_n (n : ℕ) : ℝ := 2^n

theorem geometric_sequence_general_formula :
  (∀ n : ℕ, 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) →
  (a_n 5 ^ 2 = a_n 10) →
  ∀ n : ℕ, a_n n = 2 ^ n := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_formula_l612_61273


namespace NUMINAMATH_GPT_not_perfect_cube_of_cond_l612_61278

open Int

theorem not_perfect_cube_of_cond (n : ℤ) (h₁ : 0 < n) (k : ℤ) 
  (h₂ : n^5 + n^3 + 2 * n^2 + 2 * n + 2 = k ^ 3) : 
  ¬ ∃ m : ℤ, 2 * n^2 + n + 2 = m ^ 3 :=
sorry

end NUMINAMATH_GPT_not_perfect_cube_of_cond_l612_61278


namespace NUMINAMATH_GPT_hawkeye_charged_4_times_l612_61260

variables (C B L S : ℝ) (N : ℕ)
def hawkeye_charging_problem : Prop :=
  C = 3.5 ∧ B = 20 ∧ L = 6 ∧ S = B - L ∧ N = (S / C) → N = 4 

theorem hawkeye_charged_4_times : hawkeye_charging_problem C B L S N :=
by {
  repeat { sorry }
}

end NUMINAMATH_GPT_hawkeye_charged_4_times_l612_61260


namespace NUMINAMATH_GPT_sector_area_l612_61218

theorem sector_area (alpha : ℝ) (r : ℝ) (h_alpha : alpha = Real.pi / 3) (h_r : r = 2) : 
  (1 / 2) * (alpha * r) * r = (2 * Real.pi) / 3 := 
by
  sorry

end NUMINAMATH_GPT_sector_area_l612_61218


namespace NUMINAMATH_GPT_john_flights_of_stairs_l612_61299

theorem john_flights_of_stairs (x : ℕ) : 
    let flight_height := 10
    let rope_height := flight_height / 2
    let ladder_height := rope_height + 10
    let total_height := 70
    10 * x + rope_height + ladder_height = total_height → x = 5 :=
by
    intro h
    sorry

end NUMINAMATH_GPT_john_flights_of_stairs_l612_61299


namespace NUMINAMATH_GPT_marcus_goal_points_value_l612_61208

-- Definitions based on conditions
def marcus_goals_first_type := 5
def marcus_goals_second_type := 10
def second_type_goal_points := 2
def team_total_points := 70
def marcus_percentage_points := 50

-- Theorem statement
theorem marcus_goal_points_value : 
  ∃ (x : ℕ), 5 * x + 10 * 2 = 35 ∧ 35 = 50 * team_total_points / 100 := 
sorry

end NUMINAMATH_GPT_marcus_goal_points_value_l612_61208


namespace NUMINAMATH_GPT_trip_length_is_440_l612_61284

noncomputable def total_trip_length (d : ℝ) : Prop :=
  55 * 0.02 * (d - 40) = d

theorem trip_length_is_440 :
  total_trip_length 440 :=
by
  sorry

end NUMINAMATH_GPT_trip_length_is_440_l612_61284


namespace NUMINAMATH_GPT_extra_flour_l612_61291

-- Define the conditions
def recipe_flour : ℝ := 7.0
def mary_flour : ℝ := 9.0

-- Prove the number of extra cups of flour Mary puts in
theorem extra_flour : mary_flour - recipe_flour = 2 :=
by
  sorry

end NUMINAMATH_GPT_extra_flour_l612_61291
