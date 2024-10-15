import Mathlib

namespace NUMINAMATH_GPT_matrix_inverse_problem_l2391_239174

theorem matrix_inverse_problem
  (x y z w : ℚ)
  (h1 : 2 * x + 3 * w = 1)
  (h2 : x * z = 15)
  (h3 : 4 * w = -8)
  (h4 : 4 * z = 5 * y) :
  x * y * z * w = -102.857 := by
    sorry

end NUMINAMATH_GPT_matrix_inverse_problem_l2391_239174


namespace NUMINAMATH_GPT_find_line_eq_show_point_on_circle_l2391_239111

noncomputable section

variables {x y x0 y0 : ℝ} (P Q : ℝ × ℝ) (h1 : y0 ≠ 0)
  (h2 : P = (x0, y0))
  (h3 : P.1^2/4 + P.2^2/3 = 1)
  (h4 : Q = (x0/4, y0/3))

theorem find_line_eq (M : ℝ × ℝ) (hM : ∀ (M : ℝ × ℝ), 
  ((P.1 - M.1) , (P.2 - M.2)) • (Q.1 , Q.2) = 0) :
  ∀ (x0 y0 : ℝ), y0 ≠ 0 → ∀ (x y : ℝ), 
  (x0 * x / 4 + y0 * y / 3 = 1) :=
by sorry
  
theorem show_point_on_circle (F S : ℝ × ℝ)
  (hF : F = (1, 0)) (hs : ∀ (x0 y0 : ℝ), y0 ≠ 0 → 
  S = (4, 0) ∧ ((S.1 - P.1) ^ 2 + (S.2 - P.2) ^ 2 = 36)) :
  ∀ (x y : ℝ), 
  (x - 1) ^ 2 + y ^ 2 = 36 := 
by sorry

end NUMINAMATH_GPT_find_line_eq_show_point_on_circle_l2391_239111


namespace NUMINAMATH_GPT_mod_exponent_problem_l2391_239105

theorem mod_exponent_problem : (11 ^ 2023) % 100 = 31 := by
  sorry

end NUMINAMATH_GPT_mod_exponent_problem_l2391_239105


namespace NUMINAMATH_GPT_sasha_remaining_questions_l2391_239140

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end NUMINAMATH_GPT_sasha_remaining_questions_l2391_239140


namespace NUMINAMATH_GPT_smallest_base_l2391_239198

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end NUMINAMATH_GPT_smallest_base_l2391_239198


namespace NUMINAMATH_GPT_stephen_total_distance_l2391_239128

noncomputable def total_distance : ℝ :=
let speed1 : ℝ := 16
let time1 : ℝ := 10 / 60
let distance1 : ℝ := speed1 * time1

let speed2 : ℝ := 12 - 2 -- headwind reduction
let time2 : ℝ := 20 / 60
let distance2 : ℝ := speed2 * time2

let speed3 : ℝ := 20 + 4 -- tailwind increase
let time3 : ℝ := 15 / 60
let distance3 : ℝ := speed3 * time3

distance1 + distance2 + distance3

theorem stephen_total_distance :
  total_distance = 12 :=
by sorry

end NUMINAMATH_GPT_stephen_total_distance_l2391_239128


namespace NUMINAMATH_GPT_slope_of_line_l2391_239137

theorem slope_of_line
  (m : ℝ)
  (b : ℝ)
  (h1 : b = 4)
  (h2 : ∀ x y : ℝ, y = m * x + b → (x = 199 ∧ y = 800) → True) :
  m = 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l2391_239137


namespace NUMINAMATH_GPT_avg_calculation_l2391_239107

-- Define averages
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 0 2) 0 = 7 / 9 :=
  by
    sorry

end NUMINAMATH_GPT_avg_calculation_l2391_239107


namespace NUMINAMATH_GPT_min_value_of_expression_l2391_239185

noncomputable def minValue (a : ℝ) : ℝ :=
  1 / (3 - 2 * a) + 2 / (a - 1)

theorem min_value_of_expression : ∀ a : ℝ, 1 < a ∧ a < 3 / 2 → (1 / (3 - 2 * a) + 2 / (a - 1)) ≥ 16 / 9 :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2391_239185


namespace NUMINAMATH_GPT_yeast_cells_at_10_30_l2391_239145

def yeast_population (initial_population : ℕ) (intervals : ℕ) (growth_rate : ℝ) (decay_rate : ℝ) : ℝ :=
  initial_population * (growth_rate * (1 - decay_rate)) ^ intervals

theorem yeast_cells_at_10_30 :
  yeast_population 50 6 3 0.10 = 52493 := by
  sorry

end NUMINAMATH_GPT_yeast_cells_at_10_30_l2391_239145


namespace NUMINAMATH_GPT_speed_boat_25_kmph_l2391_239180

noncomputable def speed_of_boat_in_still_water (V_s : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let V_d := distance / time
  V_d - V_s

theorem speed_boat_25_kmph (h_vs : V_s = 5) (h_time : time = 4) (h_distance : distance = 120) :
  speed_of_boat_in_still_water V_s time distance = 25 :=
by
  rw [h_vs, h_time, h_distance]
  unfold speed_of_boat_in_still_water
  simp
  norm_num

end NUMINAMATH_GPT_speed_boat_25_kmph_l2391_239180


namespace NUMINAMATH_GPT_rationalize_denominator_ABC_value_l2391_239192

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_ABC_value_l2391_239192


namespace NUMINAMATH_GPT_cos_C_max_ab_over_c_l2391_239160

theorem cos_C_max_ab_over_c
  (a b c S : ℝ) (A B C : ℝ)
  (h1 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : S = 0.5 * a * b * Real.sin C)
  : Real.cos C = 7 / 9 := 
sorry

end NUMINAMATH_GPT_cos_C_max_ab_over_c_l2391_239160


namespace NUMINAMATH_GPT_inequality_solution_l2391_239179

theorem inequality_solution (x : ℝ) :
    (x < 1 ∨ (3 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) ∨ x > 6) ↔
    ((x - 1) * (x - 3) * (x - 4) / ((x - 2) * (x - 5) * (x - 6)) > 0) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2391_239179


namespace NUMINAMATH_GPT_students_play_at_least_one_sport_l2391_239133

def B := 12
def C := 10
def S := 9
def Ba := 6

def B_and_C := 5
def B_and_S := 4
def B_and_Ba := 3
def C_and_S := 2
def C_and_Ba := 3
def S_and_Ba := 2

def B_and_C_and_S_and_Ba := 1

theorem students_play_at_least_one_sport : 
  B + C + S + Ba - B_and_C - B_and_S - B_and_Ba - C_and_S - C_and_Ba - S_and_Ba + B_and_C_and_S_and_Ba = 19 :=
by
  sorry

end NUMINAMATH_GPT_students_play_at_least_one_sport_l2391_239133


namespace NUMINAMATH_GPT_length_of_CD_l2391_239119

theorem length_of_CD
  (radius : ℝ)
  (length : ℝ)
  (total_volume : ℝ)
  (cylinder_volume : ℝ := π * radius^2 * length)
  (hemisphere_volume : ℝ := (2 * (2/3) * π * radius^3))
  (h1 : radius = 4)
  (h2 : total_volume = 432 * π)
  (h3 : total_volume = cylinder_volume + hemisphere_volume) :
  length = 22 := by
sorry

end NUMINAMATH_GPT_length_of_CD_l2391_239119


namespace NUMINAMATH_GPT_area_of_triangle_given_conditions_l2391_239194

noncomputable def area_triangle_ABC (a b B : ℝ) : ℝ :=
  0.5 * a * b * Real.sin B

theorem area_of_triangle_given_conditions :
  area_triangle_ABC 2 (Real.sqrt 3) (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_given_conditions_l2391_239194


namespace NUMINAMATH_GPT_no_nontrivial_sum_periodic_functions_l2391_239171

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end NUMINAMATH_GPT_no_nontrivial_sum_periodic_functions_l2391_239171


namespace NUMINAMATH_GPT_choir_members_l2391_239188

theorem choir_members (n k c : ℕ) (h1 : n = k^2 + 11) (h2 : n = c * (c + 5)) : n = 300 :=
sorry

end NUMINAMATH_GPT_choir_members_l2391_239188


namespace NUMINAMATH_GPT_christmas_tree_seller_l2391_239114

theorem christmas_tree_seller 
  (cost_spruce : ℕ := 220) 
  (cost_pine : ℕ := 250) 
  (cost_fir : ℕ := 330) 
  (total_revenue : ℕ := 36000) 
  (equal_trees: ℕ) 
  (h_costs : cost_spruce + cost_pine + cost_fir = 800) 
  (h_revenue : equal_trees * 800 = total_revenue):
  3 * equal_trees = 135 :=
sorry

end NUMINAMATH_GPT_christmas_tree_seller_l2391_239114


namespace NUMINAMATH_GPT_difference_of_coordinates_l2391_239142

-- Define point and its properties in Lean.
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property.
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Given points A and M
def A : Point := {x := 8, y := 0}
def M : Point := {x := 4, y := 1}

-- Assume B is a point with coordinates x and y
variable (B : Point)

-- The theorem to prove.
theorem difference_of_coordinates :
  is_midpoint M A B → B.x - B.y = -2 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_coordinates_l2391_239142


namespace NUMINAMATH_GPT_fewer_mpg_in_city_l2391_239148

def city_mpg := 14
def city_distance := 336
def highway_distance := 480

def tank_size := city_distance / city_mpg
def highway_mpg := highway_distance / tank_size
def fewer_mpg := highway_mpg - city_mpg

theorem fewer_mpg_in_city : fewer_mpg = 6 := by
  sorry

end NUMINAMATH_GPT_fewer_mpg_in_city_l2391_239148


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_l2391_239166

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_l2391_239166


namespace NUMINAMATH_GPT_minimal_overlap_facebook_instagram_l2391_239144

variable (P : ℝ → Prop)
variable [Nonempty (Set.Icc 0 1)]

theorem minimal_overlap_facebook_instagram :
  ∀ (f i : ℝ), f = 0.85 → i = 0.75 → ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ b = 0.6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimal_overlap_facebook_instagram_l2391_239144


namespace NUMINAMATH_GPT_prob_enter_A_and_exit_F_l2391_239182

-- Define the problem description
def entrances : ℕ := 2
def exits : ℕ := 3

-- Define the probabilities
def prob_enter_A : ℚ := 1 / entrances
def prob_exit_F : ℚ := 1 / exits

-- Statement that encapsulates the proof problem
theorem prob_enter_A_and_exit_F : prob_enter_A * prob_exit_F = 1 / 6 := 
by sorry

end NUMINAMATH_GPT_prob_enter_A_and_exit_F_l2391_239182


namespace NUMINAMATH_GPT_students_not_make_cut_l2391_239190

theorem students_not_make_cut (girls boys called_back : ℕ) 
  (h_girls : girls = 42) (h_boys : boys = 80)
  (h_called_back : called_back = 25) : 
  (girls + boys - called_back = 97) := by
  sorry

end NUMINAMATH_GPT_students_not_make_cut_l2391_239190


namespace NUMINAMATH_GPT_area_sum_four_smaller_circles_equals_area_of_large_circle_l2391_239109

theorem area_sum_four_smaller_circles_equals_area_of_large_circle (R : ℝ) :
  let radius_large := R
  let radius_small := R / 2
  let area_large := π * radius_large^2
  let area_small := π * radius_small^2
  let total_area_small := 4 * area_small
  area_large = total_area_small :=
by
  sorry

end NUMINAMATH_GPT_area_sum_four_smaller_circles_equals_area_of_large_circle_l2391_239109


namespace NUMINAMATH_GPT_product_of_numbers_l2391_239189

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : x * y = 200 :=
sorry

end NUMINAMATH_GPT_product_of_numbers_l2391_239189


namespace NUMINAMATH_GPT_product_of_numbers_l2391_239199

theorem product_of_numbers (a b : ℝ) 
  (h1 : a + b = 5 * (a - b))
  (h2 : a * b = 18 * (a - b)) : 
  a * b = 54 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l2391_239199


namespace NUMINAMATH_GPT_largest_square_area_l2391_239138

theorem largest_square_area (a b c : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_area_l2391_239138


namespace NUMINAMATH_GPT_fraction_not_exist_implies_x_neg_one_l2391_239117

theorem fraction_not_exist_implies_x_neg_one {x : ℝ} :
  ¬(∃ y : ℝ, y = 1 / (x + 1)) → x = -1 :=
by
  intro h
  have : x + 1 = 0 :=
    by
      contrapose! h
      exact ⟨1 / (x + 1), rfl⟩
  linarith

end NUMINAMATH_GPT_fraction_not_exist_implies_x_neg_one_l2391_239117


namespace NUMINAMATH_GPT_sector_area_eq_three_halves_l2391_239102

theorem sector_area_eq_three_halves (θ R S : ℝ) (hθ : θ = 3) (h₁ : 2 * R + θ * R = 5) :
  S = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_eq_three_halves_l2391_239102


namespace NUMINAMATH_GPT_math_problem_proof_l2391_239181

def ratio_area_BFD_square_ABCE (x : ℝ) (AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) : Prop :=
  let AE := (AF + FE)
  let area_square := (AE)^2
  let area_triangle_BFD := area_square - (1/2 * AF * (AE - FE) + 1/2 * (AE - FE) * FE + 1/2 * DE * CD)
  (area_triangle_BFD / area_square) = (1/16)
  
theorem math_problem_proof (x AF FE DE CD : ℝ) (h1 : AF = FE / 3) (h2 : CD = 3 * DE) (area_ratio : area_triangle_BFD / area_square = 1/16) : ratio_area_BFD_square_ABCE x AF FE DE CD h1 h2 :=
sorry

end NUMINAMATH_GPT_math_problem_proof_l2391_239181


namespace NUMINAMATH_GPT_speed_of_train_l2391_239193

theorem speed_of_train (length : ℝ) (time : ℝ) (conversion_factor : ℝ) (speed_kmh : ℝ) 
  (h1 : length = 240) (h2 : time = 16) (h3 : conversion_factor = 3.6) :
  speed_kmh = (length / time) * conversion_factor := 
sorry

end NUMINAMATH_GPT_speed_of_train_l2391_239193


namespace NUMINAMATH_GPT_book_price_l2391_239176

theorem book_price (n p : ℕ) (h : n * p = 104) (hn : 10 < n ∧ n < 60) : p = 2 ∨ p = 4 ∨ p = 8 :=
sorry

end NUMINAMATH_GPT_book_price_l2391_239176


namespace NUMINAMATH_GPT_find_b_find_perimeter_b_plus_c_l2391_239121

noncomputable def triangle_condition_1
  (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.cos B = (3 * c - b) * Real.cos A

noncomputable def triangle_condition_2
  (a b : ℝ) (C : ℝ) : Prop :=
  a * Real.sin C = 2 * Real.sqrt 2

noncomputable def triangle_condition_3
  (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2

noncomputable def given_a
  (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 2

theorem find_b
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b = 3 :=
sorry

theorem find_perimeter_b_plus_c
  (a b c A B C : ℝ)
  (h1 : triangle_condition_1 a b c A B C)
  (h2 : triangle_condition_2 a b B)
  (h3 : triangle_condition_3 a b c A)
  (h4 : given_a a) :
  b + c = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_b_find_perimeter_b_plus_c_l2391_239121


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2391_239127

theorem solution_set_of_inequality (x : ℝ) : x^2 < -2 * x + 15 ↔ -5 < x ∧ x < 3 := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2391_239127


namespace NUMINAMATH_GPT_determine_a_for_quadratic_l2391_239196

theorem determine_a_for_quadratic (a : ℝ) : 
  (∃ x : ℝ, 3 * x ^ (a - 1) - x = 5 ∧ a - 1 = 2) → a = 3 := 
sorry

end NUMINAMATH_GPT_determine_a_for_quadratic_l2391_239196


namespace NUMINAMATH_GPT_ratio_perimeters_l2391_239134

noncomputable def rectangle_length : ℝ := 3
noncomputable def rectangle_width : ℝ := 2
noncomputable def triangle_hypotenuse : ℝ := Real.sqrt ((rectangle_length / 2) ^ 2 + rectangle_width ^ 2)
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_width)
noncomputable def perimeter_rhombus : ℝ := 4 * triangle_hypotenuse

theorem ratio_perimeters (h1 : rectangle_length = 3) (h2 : rectangle_width = 2) :
  (perimeter_rectangle / perimeter_rhombus) = 1 :=
by
  /- proof would go here -/
  sorry

end NUMINAMATH_GPT_ratio_perimeters_l2391_239134


namespace NUMINAMATH_GPT_fraction_meaningful_l2391_239172

theorem fraction_meaningful (x : ℝ) : (x - 2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l2391_239172


namespace NUMINAMATH_GPT_minimum_n_value_l2391_239151

-- Define a multiple condition
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Given conditions
def conditions (n : ℕ) : Prop := 
  (n ≥ 8) ∧ is_multiple 4 n ∧ is_multiple 8 n

-- Lean theorem statement for the problem
theorem minimum_n_value (n : ℕ) (h : conditions n) : n = 8 :=
  sorry

end NUMINAMATH_GPT_minimum_n_value_l2391_239151


namespace NUMINAMATH_GPT_distance_between_trees_l2391_239173

theorem distance_between_trees 
  (rows columns : ℕ)
  (boundary_distance garden_length d : ℝ)
  (h_rows : rows = 10)
  (h_columns : columns = 12)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  (9 * d + 2 * boundary_distance = garden_length) → 
  d = 22 / 9 := 
by 
  intros h_eq
  sorry

end NUMINAMATH_GPT_distance_between_trees_l2391_239173


namespace NUMINAMATH_GPT_slope_of_parallel_line_l2391_239178

theorem slope_of_parallel_line (a b c : ℝ) (h : a = 3 ∧ b = -6 ∧ c = 12) :
  ∃ m : ℝ, (∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x - 2) ∧ m = 1/2 := 
sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l2391_239178


namespace NUMINAMATH_GPT_number_of_boys_l2391_239106

-- Definitions for the given conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := 20
def total_girls := 41
def happy_boys := 6
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

-- Define the total number of boys
def total_boys := total_children - total_girls

-- Proof statement
theorem number_of_boys : total_boys = 19 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_boys_l2391_239106


namespace NUMINAMATH_GPT_triangle_area_parallel_line_l2391_239141

/-- Given line passing through (8, 2) and parallel to y = -x + 1,
    the area of the triangle formed by this line and the coordinate axes is 50. -/
theorem triangle_area_parallel_line :
  ∃ k b : ℝ, k = -1 ∧ (8 * k + b = 2) ∧ (1/2 * 10 * 10 = 50) :=
sorry

end NUMINAMATH_GPT_triangle_area_parallel_line_l2391_239141


namespace NUMINAMATH_GPT_melt_brown_fabric_scientific_notation_l2391_239112

theorem melt_brown_fabric_scientific_notation :
  0.000156 = 1.56 * 10^(-4) :=
sorry

end NUMINAMATH_GPT_melt_brown_fabric_scientific_notation_l2391_239112


namespace NUMINAMATH_GPT_vehicles_sent_l2391_239123

theorem vehicles_sent (x y : ℕ) (h1 : x + y < 18) (h2 : y < 2 * x) (h3 : x + 4 < y) :
  x = 6 ∧ y = 11 := by
  sorry

end NUMINAMATH_GPT_vehicles_sent_l2391_239123


namespace NUMINAMATH_GPT_sum_modulo_9_l2391_239175

theorem sum_modulo_9 : 
  (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_modulo_9_l2391_239175


namespace NUMINAMATH_GPT_age_difference_l2391_239124

variables (O N A : ℕ)

theorem age_difference (avg_age_stable : 10 * A = 10 * A + 50 - O + N) :
  O - N = 50 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_age_difference_l2391_239124


namespace NUMINAMATH_GPT_average_visitors_on_sundays_is_correct_l2391_239103

noncomputable def average_visitors_sundays
  (num_sundays : ℕ) (num_non_sundays : ℕ) 
  (avg_non_sunday_visitors : ℕ) (avg_month_visitors : ℕ) : ℕ :=
  let total_month_days := num_sundays + num_non_sundays
  let total_visitors := avg_month_visitors * total_month_days
  let total_non_sunday_visitors := num_non_sundays * avg_non_sunday_visitors
  let total_sunday_visitors := total_visitors - total_non_sunday_visitors
  total_sunday_visitors / num_sundays

theorem average_visitors_on_sundays_is_correct :
  average_visitors_sundays 5 25 240 290 = 540 :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_on_sundays_is_correct_l2391_239103


namespace NUMINAMATH_GPT_difference_of_averages_l2391_239169

theorem difference_of_averages :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 16) / 3
  avg1 - avg2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_averages_l2391_239169


namespace NUMINAMATH_GPT_log_equation_l2391_239104

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation (x : ℝ) (h1 : x > 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^4) = 32) :
  (log_base_10 x)^4 - log_base_10 (x^4) = 4064 :=
by
  sorry

end NUMINAMATH_GPT_log_equation_l2391_239104


namespace NUMINAMATH_GPT_remainder_when_divided_l2391_239126

theorem remainder_when_divided (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_when_divided_l2391_239126


namespace NUMINAMATH_GPT_farmer_children_l2391_239164

theorem farmer_children (n : ℕ) 
  (h1 : 15 * n - 8 - 7 = 60) : n = 5 := 
by
  sorry

end NUMINAMATH_GPT_farmer_children_l2391_239164


namespace NUMINAMATH_GPT_no_solution_in_natural_numbers_l2391_239110

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end NUMINAMATH_GPT_no_solution_in_natural_numbers_l2391_239110


namespace NUMINAMATH_GPT_Natasha_speed_over_limit_l2391_239183

theorem Natasha_speed_over_limit (d : ℕ) (t : ℕ) (speed_limit : ℕ) 
    (h1 : d = 60) 
    (h2 : t = 1) 
    (h3 : speed_limit = 50) : (d / t - speed_limit = 10) :=
by
  -- Because d = 60, t = 1, and speed_limit = 50, we need to prove (60 / 1 - 50) = 10
  sorry

end NUMINAMATH_GPT_Natasha_speed_over_limit_l2391_239183


namespace NUMINAMATH_GPT_type_R_completion_time_l2391_239143

theorem type_R_completion_time :
  (∃ R : ℝ, (2 / R + 3 / 7 = 1 / 1.2068965517241381) ∧ abs (R - 5) < 0.01) :=
  sorry

end NUMINAMATH_GPT_type_R_completion_time_l2391_239143


namespace NUMINAMATH_GPT_sequence_general_formula_l2391_239165

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n : ℕ, a n = (2 / 3) * 3^n - n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l2391_239165


namespace NUMINAMATH_GPT_problem_1_problem_2_l2391_239186

theorem problem_1 
  : (∃ (m n : ℝ), m = -1 ∧ n = 1 ∧ ∀ (x : ℝ), |x + 1| + |2 * x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) :=
sorry

theorem problem_2 
  : (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → 
    ∃ (min_val : ℝ), min_val = 9 / 2 ∧ 
    ∀ (x : ℝ), x = (1 / a + 1 / b + 1 / c) → min_val ≤ x) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2391_239186


namespace NUMINAMATH_GPT_fraction_of_coins_in_decade_1800_through_1809_l2391_239131

theorem fraction_of_coins_in_decade_1800_through_1809 (total_coins : ℕ) (coins_in_decade : ℕ) (c : total_coins = 30) (d : coins_in_decade = 5) : coins_in_decade / (total_coins : ℚ) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_coins_in_decade_1800_through_1809_l2391_239131


namespace NUMINAMATH_GPT_simplify_expression_l2391_239122

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem simplify_expression (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = a + b) :
  (a / b) + (b / a) - (1 / (a * b)) = 1 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2391_239122


namespace NUMINAMATH_GPT_sqrt_expression_l2391_239139

theorem sqrt_expression (h : n < m ∧ m < 0) : 
  (Real.sqrt (m^2 + 2 * m * n + n^2) - Real.sqrt (m^2 - 2 * m * n + n^2)) = -2 * m := 
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_expression_l2391_239139


namespace NUMINAMATH_GPT_real_roots_exist_l2391_239170

theorem real_roots_exist (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_real_roots_exist_l2391_239170


namespace NUMINAMATH_GPT_range_for_k_solutions_when_k_eq_1_l2391_239136

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end NUMINAMATH_GPT_range_for_k_solutions_when_k_eq_1_l2391_239136


namespace NUMINAMATH_GPT_second_school_more_students_l2391_239108

theorem second_school_more_students (S1 S2 S3 : ℕ) 
  (hS3 : S3 = 200) 
  (hS1 : S1 = 2 * S2) 
  (h_total : S1 + S2 + S3 = 920) : 
  S2 - S3 = 40 :=
by
  sorry

end NUMINAMATH_GPT_second_school_more_students_l2391_239108


namespace NUMINAMATH_GPT_fraction_operations_l2391_239167

theorem fraction_operations :
  let a := 1 / 3
  let b := 1 / 4
  let c := 1 / 2
  (a + b = 7 / 12) ∧ ((7 / 12) / c = 7 / 6) := by
{
  sorry
}

end NUMINAMATH_GPT_fraction_operations_l2391_239167


namespace NUMINAMATH_GPT_abs_frac_lt_one_l2391_239159

theorem abs_frac_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |(x - y) / (1 - x * y)| < 1 :=
sorry

end NUMINAMATH_GPT_abs_frac_lt_one_l2391_239159


namespace NUMINAMATH_GPT_fermat_numbers_coprime_l2391_239100

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2 ^ 2 ^ (n - 1) + 1) (2 ^ 2 ^ (m - 1) + 1) = 1 :=
sorry

end NUMINAMATH_GPT_fermat_numbers_coprime_l2391_239100


namespace NUMINAMATH_GPT_cherry_sodas_in_cooler_l2391_239132

theorem cherry_sodas_in_cooler (C : ℕ) (h1 : (C + 2 * C = 24)) : C = 8 :=
sorry

end NUMINAMATH_GPT_cherry_sodas_in_cooler_l2391_239132


namespace NUMINAMATH_GPT_inequality_solution_intervals_l2391_239155

theorem inequality_solution_intervals (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := 
sorry

end NUMINAMATH_GPT_inequality_solution_intervals_l2391_239155


namespace NUMINAMATH_GPT_price_of_first_variety_of_oil_l2391_239149

theorem price_of_first_variety_of_oil 
  (P : ℕ) 
  (x : ℕ) 
  (cost_second_variety : ℕ) 
  (volume_second_variety : ℕ)
  (cost_mixture_per_liter : ℕ) 
  : x = 160 ∧ cost_second_variety = 60 ∧ volume_second_variety = 240 ∧ cost_mixture_per_liter = 52 → P = 40 :=
by
  sorry

end NUMINAMATH_GPT_price_of_first_variety_of_oil_l2391_239149


namespace NUMINAMATH_GPT_find_a_l2391_239113

-- Define sets A and B based on the given real number a
def A (a : ℝ) : Set ℝ := {a^2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}

-- Given condition
def condition (a : ℝ) : Prop := A a ∩ B a = {-3}

-- Prove that a = -2/3 is the solution satisfying the condition
theorem find_a : ∃ a : ℝ, condition a ∧ a = -2/3 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_find_a_l2391_239113


namespace NUMINAMATH_GPT_polynomial_sum_correct_l2391_239147

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - x - 5
def g (x : ℝ) : ℝ := -6 * x^3 - 7 * x^2 + 4 * x - 2
def h (x : ℝ) : ℝ := 2 * x^3 + 8 * x^2 + 6 * x + 3
def sum_polynomials (x : ℝ) : ℝ := -8 * x^3 + 3 * x^2 + 9 * x - 4

theorem polynomial_sum_correct (x : ℝ) : f x + g x + h x = sum_polynomials x :=
by sorry

end NUMINAMATH_GPT_polynomial_sum_correct_l2391_239147


namespace NUMINAMATH_GPT_highest_vs_lowest_temp_difference_l2391_239168

theorem highest_vs_lowest_temp_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 26) 
  (h_lowest : lowest_temp = 14) : 
  highest_temp - lowest_temp = 12 := 
by 
  sorry

end NUMINAMATH_GPT_highest_vs_lowest_temp_difference_l2391_239168


namespace NUMINAMATH_GPT_probability_of_roots_condition_l2391_239135

theorem probability_of_roots_condition :
  let k := 6 -- Lower bound of the interval
  let k' := 10 -- Upper bound of the interval
  let interval_length := k' - k
  let satisfying_interval_length := (22 / 3) - 6
  -- The probability that the roots of the quadratic equation satisfy x₁ ≤ 2x₂
  (satisfying_interval_length / interval_length) = (1 / 3) := by
    sorry

end NUMINAMATH_GPT_probability_of_roots_condition_l2391_239135


namespace NUMINAMATH_GPT_mario_oranges_l2391_239156

theorem mario_oranges (M L N T x : ℕ) 
  (H_L : L = 24) 
  (H_N : N = 96) 
  (H_T : T = 128) 
  (H_total : x + L + N = T) : 
  x = 8 :=
by
  rw [H_L, H_N, H_T] at H_total
  linarith

end NUMINAMATH_GPT_mario_oranges_l2391_239156


namespace NUMINAMATH_GPT_largest_percentage_increase_l2391_239152

def student_count (year: ℕ) : ℝ :=
  match year with
  | 2010 => 80
  | 2011 => 88
  | 2012 => 95
  | 2013 => 100
  | 2014 => 105
  | 2015 => 112
  | _    => 0  -- Because we only care about 2010-2015

noncomputable def percentage_increase (year1 year2 : ℕ) : ℝ :=
  ((student_count year2 - student_count year1) / student_count year1) * 100

theorem largest_percentage_increase :
  (∀ x y, percentage_increase 2010 2011 ≥ percentage_increase x y) :=
by sorry

end NUMINAMATH_GPT_largest_percentage_increase_l2391_239152


namespace NUMINAMATH_GPT_centroid_path_area_correct_l2391_239120

noncomputable def centroid_path_area (AB : ℝ) (A B C : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let R := AB / 2
  let radius_of_path := R / 3
  let area := Real.pi * radius_of_path ^ 2
  area

theorem centroid_path_area_correct (AB : ℝ) (A B C : ℝ × ℝ)
  (hAB : AB = 32)
  (hAB_diameter : (∃ O : ℝ × ℝ, dist O A = dist O B ∧ dist A B = 2 * dist O A))
  (hC_circle : ∃ O : ℝ × ℝ, dist O C = AB / 2 ∧ C ≠ A ∧ C ≠ B):
  centroid_path_area AB A B C (0, 0) = (256 / 9) * Real.pi := by
  sorry

end NUMINAMATH_GPT_centroid_path_area_correct_l2391_239120


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l2391_239115

theorem arithmetic_sequence_inequality 
  (a b c : ℝ) 
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : b - a = d)
  (h3 : c - b = d) :
  ¬ (a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l2391_239115


namespace NUMINAMATH_GPT_sum_of_angles_equal_360_l2391_239158

variables (A B C D F G : ℝ)

-- Given conditions.
def is_quadrilateral_interior_sum (A B C D : ℝ) : Prop := A + B + C + D = 360
def split_internal_angles (F G : ℝ) (C D : ℝ) : Prop := F + G = C + D

-- Proof problem statement.
theorem sum_of_angles_equal_360
  (h1 : is_quadrilateral_interior_sum A B C D)
  (h2 : split_internal_angles F G C D) :
  A + B + C + D + F + G = 360 :=
sorry

end NUMINAMATH_GPT_sum_of_angles_equal_360_l2391_239158


namespace NUMINAMATH_GPT_typist_salary_proof_l2391_239101

noncomputable def original_salary (x : ℝ) : Prop :=
  1.10 * x * 0.95 = 1045

theorem typist_salary_proof (x : ℝ) (H : original_salary x) : x = 1000 :=
sorry

end NUMINAMATH_GPT_typist_salary_proof_l2391_239101


namespace NUMINAMATH_GPT_polygon_with_largest_area_l2391_239118

noncomputable def area_of_polygon_A : ℝ := 6
noncomputable def area_of_polygon_B : ℝ := 4
noncomputable def area_of_polygon_C : ℝ := 4 + 2 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_D : ℝ := 3 + 3 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_E : ℝ := 7

theorem polygon_with_largest_area : 
  area_of_polygon_E > area_of_polygon_A ∧ 
  area_of_polygon_E > area_of_polygon_B ∧ 
  area_of_polygon_E > area_of_polygon_C ∧ 
  area_of_polygon_E > area_of_polygon_D :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_largest_area_l2391_239118


namespace NUMINAMATH_GPT_easter_eggs_problem_l2391_239197

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end NUMINAMATH_GPT_easter_eggs_problem_l2391_239197


namespace NUMINAMATH_GPT_even_expression_l2391_239146

theorem even_expression (m n : ℤ) (hm : Odd m) (hn : Odd n) : Even (m + 5 * n) :=
by
  sorry

end NUMINAMATH_GPT_even_expression_l2391_239146


namespace NUMINAMATH_GPT_length_of_uncovered_side_l2391_239157

variables (L W : ℝ)

-- Conditions
def area_eq_680 := (L * W = 680)
def fence_eq_178 := (2 * W + L = 178)

-- Theorem statement to prove the length of the uncovered side
theorem length_of_uncovered_side (h1 : area_eq_680 L W) (h2 : fence_eq_178 L W) : L = 170 := 
sorry

end NUMINAMATH_GPT_length_of_uncovered_side_l2391_239157


namespace NUMINAMATH_GPT_quadratic_roots_expression_l2391_239129

theorem quadratic_roots_expression (x1 x2 : ℝ) (h1 : x1^2 + x1 - 2023 = 0) (h2 : x2^2 + x2 - 2023 = 0) :
  x1^2 + 2*x1 + x2 = 2022 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_expression_l2391_239129


namespace NUMINAMATH_GPT_f_neg_2_f_monotonically_decreasing_l2391_239116

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂ - 4
axiom f_2 : f 2 = 0
axiom f_pos_2 (x : ℝ) : x > 2 → f x < 0

-- Statement to prove f(-2) = 8
theorem f_neg_2 : f (-2) = 8 := sorry

-- Statement to prove that f(x) is monotonically decreasing on ℝ
theorem f_monotonically_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ := sorry

end NUMINAMATH_GPT_f_neg_2_f_monotonically_decreasing_l2391_239116


namespace NUMINAMATH_GPT_initial_observations_count_l2391_239153

theorem initial_observations_count (S x n : ℕ) (h1 : S = 12 * n) (h2 : S + x = 11 * (n + 1)) (h3 : x = 5) : n = 6 :=
sorry

end NUMINAMATH_GPT_initial_observations_count_l2391_239153


namespace NUMINAMATH_GPT_number_of_common_tangents_l2391_239191

def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem number_of_common_tangents : ∃ n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, circleM x y → circleN x y → false) :=
by
  sorry

end NUMINAMATH_GPT_number_of_common_tangents_l2391_239191


namespace NUMINAMATH_GPT_param_A_valid_param_B_valid_l2391_239161

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Parameterization A
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)

-- Parameterization B
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)

-- Theorem to prove that parameterization A satisfies the line equation
theorem param_A_valid (t : ℝ) : line_eq (param_A t).1 (param_A t).2 := by
  sorry

-- Theorem to prove that parameterization B satisfies the line equation
theorem param_B_valid (t : ℝ) : line_eq (param_B t).1 (param_B t).2 := by
  sorry

end NUMINAMATH_GPT_param_A_valid_param_B_valid_l2391_239161


namespace NUMINAMATH_GPT_monkey_ladder_min_rungs_l2391_239125

/-- 
  Proof that the minimum number of rungs n that allows the monkey to climb 
  to the top of the ladder and return to the ground, given that the monkey 
  ascends 16 rungs or descends 9 rungs at a time, is 24. 
-/
theorem monkey_ladder_min_rungs (n : ℕ) (ascend descend : ℕ) 
  (h1 : ascend = 16) (h2 : descend = 9) 
  (h3 : (∃ x y : ℤ, 16 * x - 9 * y = n) ∧ 
        (∃ x' y' : ℤ, 16 * x' - 9 * y' = 0)) : 
  n = 24 :=
sorry

end NUMINAMATH_GPT_monkey_ladder_min_rungs_l2391_239125


namespace NUMINAMATH_GPT_houses_count_l2391_239162

theorem houses_count (n : ℕ) 
  (h1 : ∃ k : ℕ, k + 7 = 12)
  (h2 : ∃ m : ℕ, m + 25 = 30) :
  n = 32 :=
sorry

end NUMINAMATH_GPT_houses_count_l2391_239162


namespace NUMINAMATH_GPT_median_of_triangle_l2391_239184

variable (a b c : ℝ)

noncomputable def AM : ℝ :=
  (Real.sqrt (2 * b * b + 2 * c * c - a * a)) / 2

theorem median_of_triangle :
  abs (((b + c) / 2) - (a / 2)) < AM a b c ∧ 
  AM a b c < (b + c) / 2 := 
by
  sorry

end NUMINAMATH_GPT_median_of_triangle_l2391_239184


namespace NUMINAMATH_GPT_two_point_line_l2391_239187

theorem two_point_line (k b : ℝ) (h_k : k ≠ 0) :
  (∀ (x y : ℝ), (y = k * x + b → (x, y) = (0, 0) ∨ (x, y) = (1, 1))) →
  (∀ (x y : ℝ), (y = k * x + b → (x, y) ≠ (2, 0))) :=
by
  sorry

end NUMINAMATH_GPT_two_point_line_l2391_239187


namespace NUMINAMATH_GPT_sum_infinite_series_l2391_239195

theorem sum_infinite_series : 
  ∑' n : ℕ, (2 * (n + 1) + 3) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2) * ((n + 1) + 3)) = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_sum_infinite_series_l2391_239195


namespace NUMINAMATH_GPT_problem_range_of_k_l2391_239150

theorem problem_range_of_k (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → (0 < k ∧ k ≤ 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_problem_range_of_k_l2391_239150


namespace NUMINAMATH_GPT_simplify_fraction_l2391_239163

theorem simplify_fraction :
  10 * (15 / 8) * (-40 / 45) = -(50 / 3) :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l2391_239163


namespace NUMINAMATH_GPT_find_base_l2391_239130

noncomputable def base_satisfies_first_transaction (s : ℕ) : Prop :=
  5 * s^2 + 3 * s + 460 = s^3 + s^2 + 1

noncomputable def base_satisfies_second_transaction (s : ℕ) : Prop :=
  s^2 + 2 * s + 2 * s^2 + 6 * s = 5 * s^2

theorem find_base (s : ℕ) (h1 : base_satisfies_first_transaction s) (h2 : base_satisfies_second_transaction s) :
  s = 4 :=
sorry

end NUMINAMATH_GPT_find_base_l2391_239130


namespace NUMINAMATH_GPT_inequality_solution_set_correct_l2391_239177

noncomputable def inequality_solution_set (a b c x : ℝ) : Prop :=
  (a > c) → (b + c > 0) → ((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0)

theorem inequality_solution_set_correct (a b c : ℝ) :
  a > c → b + c > 0 → ∀ x, ((a > c) → (b + c > 0) → (((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0))) :=
by
  intros h1 h2 x
  sorry

end NUMINAMATH_GPT_inequality_solution_set_correct_l2391_239177


namespace NUMINAMATH_GPT_angle_C_is_65_deg_l2391_239154

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end NUMINAMATH_GPT_angle_C_is_65_deg_l2391_239154
