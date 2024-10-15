import Mathlib

namespace NUMINAMATH_GPT_banana_cost_l2272_227270

theorem banana_cost (pounds: ℕ) (rate: ℕ) (per_pounds: ℕ) : 
 (pounds = 18) → (rate = 3) → (per_pounds = 3) → 
  (pounds / per_pounds * rate = 18) := by
  intros
  sorry

end NUMINAMATH_GPT_banana_cost_l2272_227270


namespace NUMINAMATH_GPT_Dalton_saved_amount_l2272_227226

theorem Dalton_saved_amount (total_cost uncle_contribution additional_needed saved_from_allowance : ℕ) 
  (h_total_cost : total_cost = 7 + 12 + 4)
  (h_uncle_contribution : uncle_contribution = 13)
  (h_additional_needed : additional_needed = 4)
  (h_current_amount : total_cost - additional_needed = 19)
  (h_saved_amount : 19 - uncle_contribution = saved_from_allowance) :
  saved_from_allowance = 6 :=
sorry

end NUMINAMATH_GPT_Dalton_saved_amount_l2272_227226


namespace NUMINAMATH_GPT_find_a_l2272_227202

open Set

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {1, 2})
  (hB : B = {-a, a^2 + 3})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l2272_227202


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3s_l2272_227241

theorem instantaneous_velocity_at_3s (t s v : ℝ) (hs : s = t^3) (hts : t = 3*s) : v = 27 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3s_l2272_227241


namespace NUMINAMATH_GPT_john_friends_count_l2272_227255

-- Define the initial conditions
def initial_amount : ℚ := 7.10
def cost_of_sweets : ℚ := 1.05
def amount_per_friend : ℚ := 1.00
def remaining_amount : ℚ := 4.05

-- Define the intermediate values
def after_sweets : ℚ := initial_amount - cost_of_sweets
def given_away : ℚ := after_sweets - remaining_amount

-- Define the final proof statement
theorem john_friends_count : given_away / amount_per_friend = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_friends_count_l2272_227255


namespace NUMINAMATH_GPT_symmetrical_point_l2272_227299

-- Definition of symmetry with respect to the x-axis
def symmetrical (x y: ℝ) : ℝ × ℝ := (x, -y)

-- Coordinates of the original point A
def A : ℝ × ℝ := (-2, 3)

-- Coordinates of the symmetrical point
def symmetrical_A : ℝ × ℝ := symmetrical (-2) 3

-- The theorem we want to prove
theorem symmetrical_point :
  symmetrical_A = (-2, -3) :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_symmetrical_point_l2272_227299


namespace NUMINAMATH_GPT_simplify_expression_l2272_227235

variable (x : ℝ) (h : x ≠ 0)

theorem simplify_expression : (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2272_227235


namespace NUMINAMATH_GPT_value_of_f_at_log_l2272_227262

noncomputable def f : ℝ → ℝ := sorry -- We will define this below

-- Conditions as hypotheses
axiom odd_f : ∀ x : ℝ, f (-x) = - f (x)
axiom periodic_f : ∀ x : ℝ, f (x + 2) + f (x) = 0
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = 2^x - 1

-- Theorem statement
theorem value_of_f_at_log : f (Real.logb (1/8) 125) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_value_of_f_at_log_l2272_227262


namespace NUMINAMATH_GPT_path_to_tile_ratio_l2272_227208

theorem path_to_tile_ratio
  (t p : ℝ) 
  (tiles : ℕ := 400)
  (grid_size : ℕ := 20)
  (total_tile_area : ℝ := (tiles : ℝ) * t^2)
  (total_courtyard_area : ℝ := (grid_size * (t + 2 * p))^2) 
  (tile_area_fraction : ℝ := total_tile_area / total_courtyard_area) : 
  tile_area_fraction = 0.25 → 
  p / t = 0.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_path_to_tile_ratio_l2272_227208


namespace NUMINAMATH_GPT_product_of_three_numbers_l2272_227272

theorem product_of_three_numbers (a b c : ℝ) 
  (h₁ : a + b + c = 45)
  (h₂ : a = 2 * (b + c))
  (h₃ : c = 4 * b) : 
  a * b * c = 1080 := 
sorry

end NUMINAMATH_GPT_product_of_three_numbers_l2272_227272


namespace NUMINAMATH_GPT_find_A_l2272_227265

theorem find_A (A B : ℝ) 
  (h1 : A - 3 * B = 303.1)
  (h2 : 10 * B = A) : 
  A = 433 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l2272_227265


namespace NUMINAMATH_GPT_solve_y_l2272_227220

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_l2272_227220


namespace NUMINAMATH_GPT_split_coins_l2272_227263

theorem split_coins (p n d q : ℕ) (hp : p % 5 = 0) 
  (h_total : p + 5 * n + 10 * d + 25 * q = 10000) :
  ∃ (p1 n1 d1 q1 p2 n2 d2 q2 : ℕ),
    (p1 + 5 * n1 + 10 * d1 + 25 * q1 = 5000) ∧
    (p2 + 5 * n2 + 10 * d2 + 25 * q2 = 5000) ∧
    (p = p1 + p2) ∧ (n = n1 + n2) ∧ (d = d1 + d2) ∧ (q = q1 + q2) :=
sorry

end NUMINAMATH_GPT_split_coins_l2272_227263


namespace NUMINAMATH_GPT_slope_range_l2272_227287

theorem slope_range {A : ℝ × ℝ} (k : ℝ) : 
  A = (1, 1) → (0 < 1 - k ∧ 1 - k < 2) → -1 < k ∧ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_range_l2272_227287


namespace NUMINAMATH_GPT_total_precious_stones_is_305_l2272_227215

theorem total_precious_stones_is_305 :
  let agate := 25
  let olivine := agate + 5
  let sapphire := 2 * olivine
  let diamond := olivine + 11
  let amethyst := sapphire + diamond
  let ruby := diamond + 7
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 :=
by
  sorry

end NUMINAMATH_GPT_total_precious_stones_is_305_l2272_227215


namespace NUMINAMATH_GPT_quadratic_roots_diff_by_2_l2272_227214

theorem quadratic_roots_diff_by_2 (q : ℝ) (hq : 0 < q) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2 = 2 ∨ r2 - r1 = 2) ∧ r1 ^ 2 + (2 * q - 1) * r1 + q = 0 ∧ r2 ^ 2 + (2 * q - 1) * r2 + q = 0) ↔
  q = 1 + (Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_diff_by_2_l2272_227214


namespace NUMINAMATH_GPT_teddy_bear_cost_l2272_227204

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end NUMINAMATH_GPT_teddy_bear_cost_l2272_227204


namespace NUMINAMATH_GPT_tan_neg_210_eq_neg_sqrt_3_div_3_l2272_227210

theorem tan_neg_210_eq_neg_sqrt_3_div_3 : Real.tan (-210 * Real.pi / 180) = - (Real.sqrt 3 / 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_neg_210_eq_neg_sqrt_3_div_3_l2272_227210


namespace NUMINAMATH_GPT_find_a_l2272_227289

theorem find_a (a : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) 
    (h_curve : ∀ x, y x = x^4 + a * x^2 + 1)
    (h_derivative : ∀ x, y' x = (4 * x^3 + 2 * a * x))
    (h_tangent_slope : y' (-1) = 8) :
    a = -6 :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_find_a_l2272_227289


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2272_227230

theorem necessary_but_not_sufficient (x : ℝ) : 
  (0 < x ∧ x < 2) → (x^2 - x - 6 < 0) ∧ ¬ ((x^2 - x - 6 < 0) → (0 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2272_227230


namespace NUMINAMATH_GPT_mod_product_example_l2272_227251

theorem mod_product_example :
  ∃ m : ℤ, 256 * 738 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 ∧ m = 53 :=
by
  use 53
  sorry

end NUMINAMATH_GPT_mod_product_example_l2272_227251


namespace NUMINAMATH_GPT_cistern_wet_surface_area_l2272_227221

def cistern (length : ℕ) (width : ℕ) (water_height : ℝ) : ℝ :=
  (length * width : ℝ) + 2 * (water_height * length) + 2 * (water_height * width)

theorem cistern_wet_surface_area :
  cistern 7 5 1.40 = 68.6 :=
by
  sorry

end NUMINAMATH_GPT_cistern_wet_surface_area_l2272_227221


namespace NUMINAMATH_GPT_rainfall_on_thursday_l2272_227217

theorem rainfall_on_thursday
  (monday_am : ℝ := 2)
  (monday_pm : ℝ := 1)
  (tuesday_factor : ℝ := 2)
  (wednesday : ℝ := 0)
  (thursday : ℝ)
  (weekly_avg : ℝ := 4)
  (days_in_week : ℕ := 7)
  (total_weekly_rain : ℝ := days_in_week * weekly_avg) :
  2 * (monday_am + monday_pm + tuesday_factor * (monday_am + monday_pm) + thursday) 
    = total_weekly_rain
  → thursday = 5 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_on_thursday_l2272_227217


namespace NUMINAMATH_GPT_compare_neg_numbers_l2272_227271

theorem compare_neg_numbers : - 0.6 > - (2 / 3) := 
by sorry

end NUMINAMATH_GPT_compare_neg_numbers_l2272_227271


namespace NUMINAMATH_GPT_original_classes_l2272_227232

theorem original_classes (x : ℕ) (h1 : 280 % x = 0) (h2 : 585 % (x + 6) = 0) : x = 7 :=
sorry

end NUMINAMATH_GPT_original_classes_l2272_227232


namespace NUMINAMATH_GPT_find_fraction_l2272_227244

variable (x y z : ℂ) -- All complex numbers
variable (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) -- Non-zero conditions
variable (h2 : x + y + z = 10) -- Sum condition
variable (h3 : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) -- Given equation condition

theorem find_fraction 
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
    (h2 : x + y + z = 10)
    (h3 : 2 * ((x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2) = x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := 
sorry -- Proof yet to be completed

end NUMINAMATH_GPT_find_fraction_l2272_227244


namespace NUMINAMATH_GPT_typeB_lines_l2272_227290

noncomputable def isTypeBLine (line : Real → Real) : Prop :=
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧ (Real.sqrt ((P.1 + 5)^2 + P.2^2) - Real.sqrt ((P.1 - 5)^2 + P.2^2) = 6)

theorem typeB_lines :
  isTypeBLine (fun x => x + 1) ∧ isTypeBLine (fun x => 2) :=
by sorry

end NUMINAMATH_GPT_typeB_lines_l2272_227290


namespace NUMINAMATH_GPT_problem_statement_l2272_227246

variable {x y : ℝ}

theorem problem_statement (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : y - 2 / x ≠ 0) :
  (2 * x - 3 / y) / (3 * y - 2 / x) = (2 * x * y - 3) / (3 * x * y - 2) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2272_227246


namespace NUMINAMATH_GPT_max_popsicles_l2272_227288

theorem max_popsicles (budget : ℕ) (cost_single : ℕ) (popsicles_single : ℕ) (cost_box3 : ℕ) (popsicles_box3 : ℕ) (cost_box7 : ℕ) (popsicles_box7 : ℕ)
  (h_budget : budget = 10) (h_cost_single : cost_single = 1) (h_popsicles_single : popsicles_single = 1)
  (h_cost_box3 : cost_box3 = 3) (h_popsicles_box3 : popsicles_box3 = 3)
  (h_cost_box7 : cost_box7 = 4) (h_popsicles_box7 : popsicles_box7 = 7) :
  ∃ n, n = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_popsicles_l2272_227288


namespace NUMINAMATH_GPT_coordinates_A_B_l2272_227275

theorem coordinates_A_B : 
  (∃ x, 7 * x + 2 * 3 = 41) ∧ (∃ y, 7 * (-5) + 2 * y = 41) → 
  ((∃ x, x = 5) ∧ (∃ y, y = 38)) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_A_B_l2272_227275


namespace NUMINAMATH_GPT_problem_solution_l2272_227269

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2272_227269


namespace NUMINAMATH_GPT_parabola_circle_intersection_l2272_227286

theorem parabola_circle_intersection (a : ℝ) : 
  a ≤ Real.sqrt 2 + 1 / 4 → 
  ∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_circle_intersection_l2272_227286


namespace NUMINAMATH_GPT_compute_expression_l2272_227228

theorem compute_expression :
  (1 / 36) / ((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) + 
  (((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) / (1 / 36)) = -10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2272_227228


namespace NUMINAMATH_GPT_fraction_spent_on_fruits_l2272_227239

theorem fraction_spent_on_fruits (M : ℕ) (hM : M = 24) :
  (M - (M / 3 + M / 6) - 6) / M = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_fruits_l2272_227239


namespace NUMINAMATH_GPT_find_smaller_number_l2272_227242

def one_number_is_11_more_than_3times_another (x y : ℕ) : Prop :=
  y = 3 * x + 11

def their_sum_is_55 (x y : ℕ) : Prop :=
  x + y = 55

theorem find_smaller_number (x y : ℕ) (h1 : one_number_is_11_more_than_3times_another x y) (h2 : their_sum_is_55 x y) :
  x = 11 :=
by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2272_227242


namespace NUMINAMATH_GPT_y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l2272_227205

def y := 96 + 144 + 200 + 300 + 600 + 720 + 4800

theorem y_is_multiple_of_4 : y % 4 = 0 := 
by sorry

theorem y_is_not_multiple_of_8 : y % 8 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_16 : y % 16 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_32 : y % 32 ≠ 0 := 
by sorry

end NUMINAMATH_GPT_y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l2272_227205


namespace NUMINAMATH_GPT_tangent_lines_ln_l2272_227256

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end NUMINAMATH_GPT_tangent_lines_ln_l2272_227256


namespace NUMINAMATH_GPT_percentage_half_day_students_l2272_227216

theorem percentage_half_day_students
  (total_students : ℕ)
  (full_day_students : ℕ)
  (h_total : total_students = 80)
  (h_full_day : full_day_students = 60) :
  ((total_students - full_day_students) / total_students : ℚ) * 100 = 25 := 
by
  sorry

end NUMINAMATH_GPT_percentage_half_day_students_l2272_227216


namespace NUMINAMATH_GPT_problem_statement_l2272_227213

theorem problem_statement (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6) :
  (a * b / c) + (b * c / a) + (c * a / b) = 49 / 6 := 
by sorry

end NUMINAMATH_GPT_problem_statement_l2272_227213


namespace NUMINAMATH_GPT_min_x_y_l2272_227259

theorem min_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 2) :
  x + y ≥ 9 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_min_x_y_l2272_227259


namespace NUMINAMATH_GPT_polygon_sides_l2272_227250

theorem polygon_sides (n : ℕ) (h : n * (n - 3) / 2 = 20) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2272_227250


namespace NUMINAMATH_GPT_const_sequence_l2272_227266

theorem const_sequence (x y : ℝ) (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ n, a n - a (n + 1) = (a n ^ 2 - 1) / (a n + a (n - 1)))
  (h2 : ∀ n, a n = a (n + 1) → a n ^ 2 = 1 ∧ a n ≠ -a (n - 1))
  (h_init : a 1 = y ∧ a 0 = x)
  (hx : |x| = 1 ∧ y ≠ -x) :
  (∃ n0, ∀ n ≥ n0, a n = 1 ∨ a n = -1) := sorry

end NUMINAMATH_GPT_const_sequence_l2272_227266


namespace NUMINAMATH_GPT_distance_BF_l2272_227260

-- Given the focus F of the parabola y^2 = 4x
def focus_of_parabola : (ℝ × ℝ) := (1, 0)

-- Points A and B lie on the parabola y^2 = 4x
def point_A (x y : ℝ) := y^2 = 4 * x
def point_B (x y : ℝ) := y^2 = 4 * x

-- The line through F intersects the parabola at points A and B, and |AF| = 2
def distance_AF : ℝ := 2

-- Prove that |BF| = 2
theorem distance_BF : ∀ (A B F : ℝ × ℝ), 
  A = (1, F.2) → 
  B = (1, -F.2) → 
  F = (1, 0) → 
  |A.1 - F.1| + |A.2 - F.2| = distance_AF → 
  |B.1 - F.1| + |B.2 - F.2| = 2 :=
by
  intros A B F hA hB hF d_AF
  sorry

end NUMINAMATH_GPT_distance_BF_l2272_227260


namespace NUMINAMATH_GPT_quadratic_general_form_l2272_227293

theorem quadratic_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 → x^2 + x - 7 = 0 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_quadratic_general_form_l2272_227293


namespace NUMINAMATH_GPT_inverse_proportion_range_l2272_227224

theorem inverse_proportion_range (k : ℝ) (x : ℝ) :
  (∀ x : ℝ, (x < 0 -> (k - 1) / x > 0) ∧ (x > 0 -> (k - 1) / x < 0)) -> k < 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_range_l2272_227224


namespace NUMINAMATH_GPT_spadesuit_calculation_l2272_227268

def spadesuit (x y : ℝ) : ℝ := (x + 2 * y) ^ 2 * (x - y)

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 2 3) = 1046875 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_calculation_l2272_227268


namespace NUMINAMATH_GPT_time_for_grid_5x5_l2272_227264

-- Definition for the 3x7 grid conditions
def grid_3x7_minutes := 26
def grid_3x7_total_length := 4 * 7 + 8 * 3
def time_per_unit_length := grid_3x7_minutes / grid_3x7_total_length

-- Definition for the 5x5 grid total length
def grid_5x5_total_length := 6 * 5 + 6 * 5

-- Theorem stating that the time it takes to trace all lines of a 5x5 grid is 30 minutes
theorem time_for_grid_5x5 : (time_per_unit_length * grid_5x5_total_length) = 30 := by
  sorry

end NUMINAMATH_GPT_time_for_grid_5x5_l2272_227264


namespace NUMINAMATH_GPT_book_weight_l2272_227200

theorem book_weight (total_weight : ℕ) (num_books : ℕ) (each_book_weight : ℕ) 
  (h1 : total_weight = 42) (h2 : num_books = 14) :
  each_book_weight = total_weight / num_books :=
by
  sorry

end NUMINAMATH_GPT_book_weight_l2272_227200


namespace NUMINAMATH_GPT_circle_cut_by_parabolas_l2272_227233

theorem circle_cut_by_parabolas (n : ℕ) (h : n = 10) : 
  2 * n ^ 2 + 1 = 201 :=
by
  sorry

end NUMINAMATH_GPT_circle_cut_by_parabolas_l2272_227233


namespace NUMINAMATH_GPT_translation_correctness_l2272_227279

theorem translation_correctness :
  ( ∀ (x : ℝ), ((x + 4)^2 - 5) = ((x + 4)^2 - 5) ) :=
by
  sorry

end NUMINAMATH_GPT_translation_correctness_l2272_227279


namespace NUMINAMATH_GPT_aviana_brought_pieces_l2272_227278

variable (total_people : ℕ) (fraction_eat_pizza : ℚ) (pieces_per_person : ℕ) (remaining_pieces : ℕ)

theorem aviana_brought_pieces (h1 : total_people = 15) 
                             (h2 : fraction_eat_pizza = 3 / 5) 
                             (h3 : pieces_per_person = 4) 
                             (h4 : remaining_pieces = 14) :
                             ∃ (brought_pieces : ℕ), brought_pieces = 50 :=
by sorry

end NUMINAMATH_GPT_aviana_brought_pieces_l2272_227278


namespace NUMINAMATH_GPT_product_greater_than_constant_l2272_227243

noncomputable def f (x m : ℝ) := Real.log x - (m + 1) * x + (1 / 2) * m * x ^ 2
noncomputable def g (x m : ℝ) := Real.log x - (m + 1) * x

variables {x1 x2 m : ℝ} 
  (h1 : g x1 m = 0)
  (h2 : g x2 m = 0)
  (h3 : x2 > Real.exp 1 * x1)

theorem product_greater_than_constant :
  x1 * x2 > 2 / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_GPT_product_greater_than_constant_l2272_227243


namespace NUMINAMATH_GPT_operation_value_l2272_227280

def operation1 (y : ℤ) : ℤ := 8 - y
def operation2 (y : ℤ) : ℤ := y - 8

theorem operation_value : operation2 (operation1 15) = -15 := by
  sorry

end NUMINAMATH_GPT_operation_value_l2272_227280


namespace NUMINAMATH_GPT_find_a_and_b_l2272_227222

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2272_227222


namespace NUMINAMATH_GPT_original_speed_correct_l2272_227234

variables (t m s : ℝ)

noncomputable def original_speed (t m s : ℝ) : ℝ :=
  ((t * m + Real.sqrt (t^2 * m^2 + 4 * t * m * s)) / (2 * t))

theorem original_speed_correct (t m s : ℝ) (ht : 0 < t) : 
  original_speed t m s = (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t) :=
by
  sorry

end NUMINAMATH_GPT_original_speed_correct_l2272_227234


namespace NUMINAMATH_GPT_expand_and_simplify_product_l2272_227207

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := 4 * x^4 + 7 * x^2 + 16

theorem expand_and_simplify_product (x : ℝ) : initial_expr x = simplified_expr x := by
  -- We would provide the proof steps here
  sorry

end NUMINAMATH_GPT_expand_and_simplify_product_l2272_227207


namespace NUMINAMATH_GPT_edward_total_money_l2272_227223

-- define the amounts made and spent
def money_made_spring : ℕ := 2
def money_made_summer : ℕ := 27
def money_spent_supplies : ℕ := 5

-- total money left is calculated by adding what he made and subtracting the expenses
def total_money_end (m_spring m_summer m_supplies : ℕ) : ℕ :=
  m_spring + m_summer - m_supplies

-- the theorem to prove
theorem edward_total_money :
  total_money_end money_made_spring money_made_summer money_spent_supplies = 24 :=
by
  sorry

end NUMINAMATH_GPT_edward_total_money_l2272_227223


namespace NUMINAMATH_GPT_prod_sum_leq_four_l2272_227282

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end NUMINAMATH_GPT_prod_sum_leq_four_l2272_227282


namespace NUMINAMATH_GPT_abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l2272_227238

theorem abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one (x : ℝ) :
  |x| < 1 → x^3 < 1 ∧ (x^3 < 1 → |x| < 1 → False) :=
by
  sorry

end NUMINAMATH_GPT_abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l2272_227238


namespace NUMINAMATH_GPT_inequality_solution_set_l2272_227254

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : (x - 1) / x > 1 ↔ x < 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2272_227254


namespace NUMINAMATH_GPT_largest_number_among_list_l2272_227248

theorem largest_number_among_list :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  sorry

end NUMINAMATH_GPT_largest_number_among_list_l2272_227248


namespace NUMINAMATH_GPT_time_to_empty_l2272_227283

-- Definitions for the conditions
def rate_fill_no_leak (R : ℝ) := R = 1 / 2 -- Cistern fills in 2 hours without leak
def effective_fill_rate (R L : ℝ) := R - L = 1 / 4 -- Effective fill rate when leaking
def remember_fill_time_leak (R L : ℝ) := (R - L) * 4 = 1 -- 4 hours to fill with leak

-- Main theorem statement
theorem time_to_empty (R L : ℝ) (h1 : rate_fill_no_leak R) (h2 : effective_fill_rate R L)
  (h3 : remember_fill_time_leak R L) : (1 / L = 4) :=
by
  sorry

end NUMINAMATH_GPT_time_to_empty_l2272_227283


namespace NUMINAMATH_GPT_find_angles_l2272_227285

theorem find_angles (A B : ℝ) (h1 : A + B = 90) (h2 : A = 4 * B) : A = 72 ∧ B = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_angles_l2272_227285


namespace NUMINAMATH_GPT_auntie_em_parking_l2272_227245

theorem auntie_em_parking (total_spaces cars : ℕ) (probability_can_park : ℚ) :
  total_spaces = 20 →
  cars = 15 →
  probability_can_park = 232/323 :=
by
  sorry

end NUMINAMATH_GPT_auntie_em_parking_l2272_227245


namespace NUMINAMATH_GPT_inverse_function_evaluation_l2272_227274

theorem inverse_function_evaluation :
  ∀ (f : ℕ → ℕ) (f_inv : ℕ → ℕ),
    (∀ y, f_inv (f y) = y) ∧ (∀ x, f (f_inv x) = x) →
    f 4 = 7 →
    f 6 = 3 →
    f 3 = 6 →
    f_inv (f_inv 6 + f_inv 7) = 4 :=
by
  intros f f_inv hf hf1 hf2 hf3
  sorry

end NUMINAMATH_GPT_inverse_function_evaluation_l2272_227274


namespace NUMINAMATH_GPT_sum_of_remainders_l2272_227229

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l2272_227229


namespace NUMINAMATH_GPT_common_ratio_of_infinite_geometric_series_l2272_227212

noncomputable def first_term : ℝ := 500
noncomputable def series_sum : ℝ := 3125

theorem common_ratio_of_infinite_geometric_series (r : ℝ) (h₀ : first_term / (1 - r) = series_sum) : 
  r = 0.84 := 
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_infinite_geometric_series_l2272_227212


namespace NUMINAMATH_GPT_rectangles_with_perimeter_equals_area_l2272_227284

theorem rectangles_with_perimeter_equals_area (a b : ℕ) (h : 2 * (a + b) = a * b) : (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 4 ∧ b = 4) :=
  sorry

end NUMINAMATH_GPT_rectangles_with_perimeter_equals_area_l2272_227284


namespace NUMINAMATH_GPT_circle_represents_circle_iff_a_nonzero_l2272_227227

-- Define the equation given in the problem
def circleEquation (a x y : ℝ) : Prop :=
  a*x^2 + a*y^2 - 4*(a-1)*x + 4*y = 0

-- State the required theorem
theorem circle_represents_circle_iff_a_nonzero (a : ℝ) :
  (∃ c : ℝ, ∃ h k : ℝ, ∀ x y : ℝ, circleEquation a x y ↔ (x - h)^2 + (y - k)^2 = c)
  ↔ a ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_represents_circle_iff_a_nonzero_l2272_227227


namespace NUMINAMATH_GPT_mixing_ratios_indeterminacy_l2272_227240

theorem mixing_ratios_indeterminacy (x : ℝ) (a b : ℝ) (h1 : a + b = 50) (h2 : 0.40 * a + (x / 100) * b = 25) : False :=
sorry

end NUMINAMATH_GPT_mixing_ratios_indeterminacy_l2272_227240


namespace NUMINAMATH_GPT_compute_expression_l2272_227206

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2272_227206


namespace NUMINAMATH_GPT_alyssas_weekly_allowance_l2272_227231

-- Define the constants and parameters
def spent_on_movies (A : ℝ) := 0.5 * A
def spent_on_snacks (A : ℝ) := 0.2 * A
def saved_for_future (A : ℝ) := 0.25 * A

-- Define the remaining allowance after expenses
def remaining_allowance_after_expenses (A : ℝ) := A - spent_on_movies A - spent_on_snacks A - saved_for_future A

-- Define Alyssa's allowance given the conditions
theorem alyssas_weekly_allowance : ∀ (A : ℝ), 
  remaining_allowance_after_expenses A = 12 → 
  A = 240 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_alyssas_weekly_allowance_l2272_227231


namespace NUMINAMATH_GPT_joe_sold_50_cookies_l2272_227225

theorem joe_sold_50_cookies :
  ∀ (x : ℝ), (1.20 = 1 + 0.20 * 1) → (60 = 1.20 * x) → x = 50 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_joe_sold_50_cookies_l2272_227225


namespace NUMINAMATH_GPT_tires_sale_price_l2272_227296

variable (n : ℕ)
variable (t p_original p_sale : ℝ)

theorem tires_sale_price
  (h₁ : n = 4)
  (h₂ : t = 36)
  (h₃ : p_original = 84)
  (h₄ : p_sale = p_original - t / n) :
  p_sale = 75 := by
  sorry

end NUMINAMATH_GPT_tires_sale_price_l2272_227296


namespace NUMINAMATH_GPT_songs_distribution_l2272_227297

-- Define the sets involved
structure Girl := (Amy Beth Jo : Prop)
axiom no_song_liked_by_all : ∀ song : Girl, ¬(song.Amy ∧ song.Beth ∧ song.Jo)
axiom no_song_disliked_by_all : ∀ song : Girl, song.Amy ∨ song.Beth ∨ song.Jo
axiom pairwise_liked : ∀ song : Girl,
  (song.Amy ∧ song.Beth ∧ ¬song.Jo) ∨
  (song.Beth ∧ song.Jo ∧ ¬song.Amy) ∨
  (song.Jo ∧ song.Amy ∧ ¬song.Beth)

-- Define the theorem to prove that there are exactly 90 ways to distribute the songs
theorem songs_distribution : ∃ ways : ℕ, ways = 90 := sorry

end NUMINAMATH_GPT_songs_distribution_l2272_227297


namespace NUMINAMATH_GPT_total_packs_sold_l2272_227291

def packs_sold_village_1 : ℕ := 23
def packs_sold_village_2 : ℕ := 28

theorem total_packs_sold : packs_sold_village_1 + packs_sold_village_2 = 51 :=
by
  -- We acknowledge the correctness of the calculation.
  sorry

end NUMINAMATH_GPT_total_packs_sold_l2272_227291


namespace NUMINAMATH_GPT_crowdfunding_successful_l2272_227276

variable (highest_level second_level lowest_level total_amount : ℕ)
variable (x y z : ℕ)

noncomputable def crowdfunding_conditions (highest_level second_level lowest_level : ℕ) := 
  second_level = highest_level / 10 ∧ lowest_level = second_level / 10

noncomputable def total_raised (highest_level second_level lowest_level x y z : ℕ) :=
  highest_level * x + second_level * y + lowest_level * z

theorem crowdfunding_successful (h1 : highest_level = 5000) 
                                (h2 : crowdfunding_conditions highest_level second_level lowest_level) 
                                (h3 : total_amount = 12000) 
                                (h4 : y = 3) 
                                (h5 : z = 10) :
  total_raised highest_level second_level lowest_level x y z = total_amount → x = 2 := by
  sorry

end NUMINAMATH_GPT_crowdfunding_successful_l2272_227276


namespace NUMINAMATH_GPT_number_of_cows_consume_in_96_days_l2272_227281

-- Given conditions
def grass_growth_rate := 10 / 3
def consumption_by_70_cows_in_24_days := 70 * 24
def consumption_by_30_cows_in_60_days := 30 * 60
def total_grass_in_96_days := consumption_by_30_cows_in_60_days + 120

-- Problem statement
theorem number_of_cows_consume_in_96_days : 
  (x : ℕ) -> 96 * x = total_grass_in_96_days -> x = 20 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_number_of_cows_consume_in_96_days_l2272_227281


namespace NUMINAMATH_GPT_num_distinct_combinations_l2272_227257

-- Define the conditions
def num_dials : Nat := 4
def digits : List Nat := List.range 10  -- Digits from 0 to 9

-- Define what it means for a combination to have distinct digits
def distinct_digits (comb : List Nat) : Prop :=
  comb.length = num_dials ∧ comb.Nodup

-- The main statement for the theorem
theorem num_distinct_combinations : 
  ∃ (n : Nat), n = 5040 ∧ ∀ comb : List Nat, distinct_digits comb → comb.length = num_dials →
  (List.permutations digits).length = n :=
by
  sorry

end NUMINAMATH_GPT_num_distinct_combinations_l2272_227257


namespace NUMINAMATH_GPT_polygon_encloses_250_square_units_l2272_227218

def vertices : List (ℕ × ℕ) := [(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to calculate the area of the given polygon
  sorry

theorem polygon_encloses_250_square_units : polygon_area vertices = 250 := by
  -- Proof that the area of the polygon is 250 square units
  sorry

end NUMINAMATH_GPT_polygon_encloses_250_square_units_l2272_227218


namespace NUMINAMATH_GPT_prob_at_least_one_heart_spade_or_king_l2272_227277

theorem prob_at_least_one_heart_spade_or_king :
  let total_cards := 52
  let hearts := 13
  let spades := 13
  let kings := 4
  let unique_hsk := hearts + spades + 2  -- Two unique kings from other suits
  let prob_not_hsk := (total_cards - unique_hsk) / total_cards
  let prob_not_hsk_two_draws := prob_not_hsk * prob_not_hsk
  let prob_at_least_one_hsk := 1 - prob_not_hsk_two_draws
  prob_at_least_one_hsk = 133 / 169 :=
by sorry

end NUMINAMATH_GPT_prob_at_least_one_heart_spade_or_king_l2272_227277


namespace NUMINAMATH_GPT_sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l2272_227273

def is_sum_of_arithmetic_sequence (S : ℕ → ℚ) (a₁ d : ℚ) :=
  ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

theorem sum_has_minimum_term_then_d_positive
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_min : ∃ n : ℕ, ∀ m : ℕ, S n ≤ S m) :
  d > 0 :=
sorry

theorem Sn_positive_then_increasing_sequence
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_pos : ∀ n : ℕ, S n > 0) :
  (∀ n : ℕ, S n < S (n + 1)) :=
sorry

end NUMINAMATH_GPT_sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l2272_227273


namespace NUMINAMATH_GPT_largest_circle_area_rounded_to_nearest_int_l2272_227219

theorem largest_circle_area_rounded_to_nearest_int
  (x : Real)
  (hx : 3 * x^2 = 180) :
  let r := (16 * Real.sqrt 15) / (2 * Real.pi)
  let area_of_circle := Real.pi * r^2
  round (area_of_circle) = 306 :=
by
  sorry

end NUMINAMATH_GPT_largest_circle_area_rounded_to_nearest_int_l2272_227219


namespace NUMINAMATH_GPT_find_machines_l2272_227267

theorem find_machines (R : ℝ) : 
  (N : ℕ) -> 
  (H1 : N * R * 6 = 1) -> 
  (H2 : 4 * R * 12 = 1) -> 
  N = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_machines_l2272_227267


namespace NUMINAMATH_GPT_elephant_weight_l2272_227249

theorem elephant_weight :
  ∃ (w : ℕ), ∀ i : Fin 15, (i.val ≤ 13 → w + 2 * w = 15000) ∧ ((0:ℕ) < w → w = 5000) :=
by
  sorry

end NUMINAMATH_GPT_elephant_weight_l2272_227249


namespace NUMINAMATH_GPT_determine_a_l2272_227201

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem determine_a : (∃ a: ℝ, (∀ x: ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) ∧ ∀ x: ℝ, f x a ≤ 6 → -2 ≤ x ∧ x ≤ 3) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l2272_227201


namespace NUMINAMATH_GPT_distinct_exponentiation_values_l2272_227294

theorem distinct_exponentiation_values : 
  (∃ v1 v2 v3 v4 v5 : ℕ, 
    v1 = (3 : ℕ)^(3 : ℕ)^(3 : ℕ)^(3 : ℕ) ∧
    v2 = (3 : ℕ)^((3 : ℕ)^(3 : ℕ)^(3 : ℕ)) ∧
    v3 = (3 : ℕ)^(((3 : ℕ)^(3 : ℕ))^(3 : ℕ)) ∧
    v4 = ((3 : ℕ)^(3 : ℕ)^3) ∧
    v5 = ((3 : ℕ)^((3 : ℕ)^(3 : ℕ)^3)) ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5) := 
sorry

end NUMINAMATH_GPT_distinct_exponentiation_values_l2272_227294


namespace NUMINAMATH_GPT_solve_system_of_equations_l2272_227209

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ (x = 2) ∧ (y = 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2272_227209


namespace NUMINAMATH_GPT_hearts_total_shaded_area_l2272_227211

theorem hearts_total_shaded_area (A B C D : ℕ) (hA : A = 1) (hB : B = 4) (hC : C = 9) (hD : D = 16) :
  (D - C) + (B - A) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_hearts_total_shaded_area_l2272_227211


namespace NUMINAMATH_GPT_smallest_row_sum_greater_than_50_l2272_227295

noncomputable def sum_interior_pascal (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem smallest_row_sum_greater_than_50 : ∃ n, sum_interior_pascal n > 50 ∧ (∀ m, m < n → sum_interior_pascal m ≤ 50) ∧ sum_interior_pascal 7 = 62 ∧ (sum_interior_pascal 7) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_row_sum_greater_than_50_l2272_227295


namespace NUMINAMATH_GPT_probability_of_four_twos_in_five_rolls_l2272_227261

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end NUMINAMATH_GPT_probability_of_four_twos_in_five_rolls_l2272_227261


namespace NUMINAMATH_GPT_inverse_proportion_symmetric_l2272_227236

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_inverse_proportion_symmetric_l2272_227236


namespace NUMINAMATH_GPT_distance_of_canteen_from_each_camp_l2272_227258

noncomputable def distanceFromCanteen (distGtoRoad distBtoG : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (distGtoRoad ^ 2 + distBtoG ^ 2)
  hypotenuse / 2

theorem distance_of_canteen_from_each_camp :
  distanceFromCanteen 360 800 = 438.6 :=
by
  sorry -- The proof is omitted but must show that this statement is valid.

end NUMINAMATH_GPT_distance_of_canteen_from_each_camp_l2272_227258


namespace NUMINAMATH_GPT_evaluate_expression_l2272_227292

theorem evaluate_expression (a b : ℕ) (ha : a = 7) (hb : b = 5) : 3 * (a^3 + b^3) / (a^2 - a * b + b^2) = 36 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2272_227292


namespace NUMINAMATH_GPT_sequence_1_formula_sequence_2_formula_sequence_3_formula_l2272_227253

theorem sequence_1_formula (n : ℕ) (hn : n > 0) : 
  (∃ a : ℕ → ℚ, (a 1 = 1/2) ∧ (a 2 = 1/6) ∧ (a 3 = 1/12) ∧ (a 4 = 1/20) ∧ (∀ n, a n = 1/(n*(n+1)))) :=
by
  sorry

theorem sequence_2_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (∀ n, a n = 2^(n-1))) :=
by
  sorry

theorem sequence_3_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℚ, (a 1 = 4/5) ∧ (a 2 = 1/2) ∧ (a 3 = 4/11) ∧ (a 4 = 2/7) ∧ (∀ n, a n = 4/(3*n + 2))) :=
by
  sorry

end NUMINAMATH_GPT_sequence_1_formula_sequence_2_formula_sequence_3_formula_l2272_227253


namespace NUMINAMATH_GPT_capacity_of_smaller_bucket_l2272_227203

theorem capacity_of_smaller_bucket (x : ℕ) (h1 : x < 5) (h2 : 5 - x = 2) : x = 3 := by
  sorry

end NUMINAMATH_GPT_capacity_of_smaller_bucket_l2272_227203


namespace NUMINAMATH_GPT_largest_base_conversion_l2272_227237

theorem largest_base_conversion :
  let a := (3: ℕ)
  let b := (1 * 2^1 + 1 * 2^0: ℕ)
  let c := (3 * 8^0: ℕ)
  let d := (1 * 3^1 + 1 * 3^0: ℕ)
  max a (max b (max c d)) = d :=
by
  sorry

end NUMINAMATH_GPT_largest_base_conversion_l2272_227237


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2272_227247

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ ((1 / a < 1) → (a > 1 ∨ a < 0)) → 
  (∀ (P Q : Prop), (P → Q) → (Q → P ∨ False) → P ∧ ¬Q → False) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2272_227247


namespace NUMINAMATH_GPT_catch_up_distance_l2272_227252

def v_a : ℝ := 10 -- A's speed in kmph
def v_b : ℝ := 20 -- B's speed in kmph
def t : ℝ := 10 -- Time in hours when B starts after A

theorem catch_up_distance : v_b * t + v_a * t = 200 :=
by sorry

end NUMINAMATH_GPT_catch_up_distance_l2272_227252


namespace NUMINAMATH_GPT_larger_number_is_sixty_three_l2272_227298

theorem larger_number_is_sixty_three (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : y = 63 :=
  sorry

end NUMINAMATH_GPT_larger_number_is_sixty_three_l2272_227298
