import Mathlib

namespace NUMINAMATH_GPT_calculate_average_fish_caught_l1183_118361

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end NUMINAMATH_GPT_calculate_average_fish_caught_l1183_118361


namespace NUMINAMATH_GPT_plot_length_l1183_118391

theorem plot_length (b : ℕ) (cost_per_meter total_cost : ℕ)
  (h1 : cost_per_meter = 2650 / 100)  -- Since Lean works with integers, use 2650 instead of 26.50
  (h2 : total_cost = 5300)
  (h3 : 2 * (b + 16) + 2 * b = total_cost / cost_per_meter) :
  b + 16 = 58 :=
by
  -- Above theorem aims to prove the length of the plot is 58 meters, given the conditions.
  sorry

end NUMINAMATH_GPT_plot_length_l1183_118391


namespace NUMINAMATH_GPT_find_k_find_a_l1183_118384

noncomputable def f (a k : ℝ) (x : ℝ) := a ^ x + k * a ^ (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

theorem find_k (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : is_odd_function (f a k)) : k = -1 :=
sorry

theorem find_a (k : ℝ) (h₃ : k = -1) (h₄ : f 1 = 3 / 2) (h₅ : is_monotonic_increasing (f 2 k)) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_k_find_a_l1183_118384


namespace NUMINAMATH_GPT_cars_on_river_road_l1183_118315

-- Define the number of buses and cars
variables (B C : ℕ)

-- Given conditions
def ratio_condition : Prop := (B : ℚ) / C = 1 / 17
def fewer_buses_condition : Prop := B = C - 80

-- Problem statement
theorem cars_on_river_road (h_ratio : ratio_condition B C) (h_fewer : fewer_buses_condition B C) : C = 85 :=
by
  sorry

end NUMINAMATH_GPT_cars_on_river_road_l1183_118315


namespace NUMINAMATH_GPT_closest_perfect_square_to_1042_is_1024_l1183_118303

theorem closest_perfect_square_to_1042_is_1024 :
  ∀ n : ℕ, (n = 32 ∨ n = 33) → ((1042 - n^2 = 18) ↔ n = 32):=
by
  intros n hn
  cases hn
  case inl h32 => sorry
  case inr h33 => sorry

end NUMINAMATH_GPT_closest_perfect_square_to_1042_is_1024_l1183_118303


namespace NUMINAMATH_GPT_fraction_of_cookies_with_nuts_l1183_118327

theorem fraction_of_cookies_with_nuts
  (nuts_per_cookie : ℤ)
  (total_cookies : ℤ)
  (total_nuts : ℤ)
  (h1 : nuts_per_cookie = 2)
  (h2 : total_cookies = 60)
  (h3 : total_nuts = 72) :
  (total_nuts / nuts_per_cookie) / total_cookies = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_of_cookies_with_nuts_l1183_118327


namespace NUMINAMATH_GPT_find_second_offset_l1183_118308

theorem find_second_offset 
  (diagonal : ℝ) (offset1 : ℝ) (area_quad : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 20) (h2 : offset1 = 9) (h3 : area_quad = 150) :
  offset2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_second_offset_l1183_118308


namespace NUMINAMATH_GPT_compute_fraction_l1183_118348

theorem compute_fraction : 
  (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l1183_118348


namespace NUMINAMATH_GPT_money_made_l1183_118321

-- Define the conditions
def cost_per_bar := 4
def total_bars := 8
def bars_sold := total_bars - 3

-- We need to show that the money made is $20
theorem money_made :
  bars_sold * cost_per_bar = 20 := 
by
  sorry

end NUMINAMATH_GPT_money_made_l1183_118321


namespace NUMINAMATH_GPT_expression_evaluation_l1183_118341

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1183_118341


namespace NUMINAMATH_GPT_resulting_polygon_has_30_sides_l1183_118346

def polygon_sides : ℕ := 3 + 4 + 5 + 6 + 7 + 8 + 9 - 6 * 2

theorem resulting_polygon_has_30_sides : polygon_sides = 30 := by
  sorry

end NUMINAMATH_GPT_resulting_polygon_has_30_sides_l1183_118346


namespace NUMINAMATH_GPT_no_max_value_if_odd_and_symmetric_l1183_118330

variable (f : ℝ → ℝ)

-- Definitions:
def domain_is_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_symmetric_about_1_1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 - x) = 2 - f x

-- The theorem stating that under the given conditions there is no maximum value.
theorem no_max_value_if_odd_and_symmetric :
  domain_is_R f → is_odd_function f → is_symmetric_about_1_1 f → ¬∃ M : ℝ, ∀ x : ℝ, f x ≤ M := by
  sorry

end NUMINAMATH_GPT_no_max_value_if_odd_and_symmetric_l1183_118330


namespace NUMINAMATH_GPT_hypotenuse_is_18_point_8_l1183_118360

def hypotenuse_of_right_triangle (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2) * a * b = 24 ∧ a^2 + b^2 = c^2

theorem hypotenuse_is_18_point_8 (a b c : ℝ) (h : hypotenuse_of_right_triangle a b c) : c = 18.8 :=
  sorry

end NUMINAMATH_GPT_hypotenuse_is_18_point_8_l1183_118360


namespace NUMINAMATH_GPT_triangle_angle_B_l1183_118318

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end NUMINAMATH_GPT_triangle_angle_B_l1183_118318


namespace NUMINAMATH_GPT_minimize_sum_of_squares_at_mean_l1183_118399

-- Definitions of the conditions
def P1 (x1 : ℝ) : ℝ := x1
def P2 (x2 : ℝ) : ℝ := x2
def P3 (x3 : ℝ) : ℝ := x3
def P4 (x4 : ℝ) : ℝ := x4
def P5 (x5 : ℝ) : ℝ := x5

-- Definition of the function we want to minimize
def s (P : ℝ) (x1 x2 x3 x4 x5 : ℝ) : ℝ :=
  (P - x1)^2 + (P - x2)^2 + (P - x3)^2 + (P - x4)^2 + (P - x5)^2

-- Proof statement
theorem minimize_sum_of_squares_at_mean (x1 x2 x3 x4 x5 : ℝ) :
  ∃ P : ℝ, P = (x1 + x2 + x3 + x4 + x5) / 5 ∧ 
           ∀ x : ℝ, s P x1 x2 x3 x4 x5 ≤ s x x1 x2 x3 x4 x5 := 
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_at_mean_l1183_118399


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1183_118324

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_a4a5 : a 3 * a 4 = 24) :
  a 2 * a 5 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1183_118324


namespace NUMINAMATH_GPT_find_cube_difference_l1183_118345

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end NUMINAMATH_GPT_find_cube_difference_l1183_118345


namespace NUMINAMATH_GPT_complete_square_l1183_118388

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end NUMINAMATH_GPT_complete_square_l1183_118388


namespace NUMINAMATH_GPT_sum_values_l1183_118355

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 2) = -f x
axiom value_at_one : f 1 = 8

theorem sum_values :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end NUMINAMATH_GPT_sum_values_l1183_118355


namespace NUMINAMATH_GPT_ellie_needs_25ml_of_oil_l1183_118340

theorem ellie_needs_25ml_of_oil 
  (oil_per_wheel : ℕ) 
  (number_of_wheels : ℕ) 
  (other_parts_oil : ℕ) 
  (total_oil : ℕ)
  (h1 : oil_per_wheel = 10)
  (h2 : number_of_wheels = 2)
  (h3 : other_parts_oil = 5)
  (h4 : total_oil = oil_per_wheel * number_of_wheels + other_parts_oil) : 
  total_oil = 25 :=
  sorry

end NUMINAMATH_GPT_ellie_needs_25ml_of_oil_l1183_118340


namespace NUMINAMATH_GPT_complex_magnitude_pow_eight_l1183_118322

theorem complex_magnitude_pow_eight :
  (Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I))^8 = 7890481 / 390625 := 
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_pow_eight_l1183_118322


namespace NUMINAMATH_GPT_certain_person_current_age_l1183_118373

-- Define Sandys's current age and the certain person's current age
variable (S P : ℤ)

-- Conditions from the problem
def sandy_phone_bill_condition := 10 * S = 340
def sandy_age_relation := S + 2 = 3 * P

theorem certain_person_current_age (h1 : sandy_phone_bill_condition S) (h2 : sandy_age_relation S P) : P - 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_certain_person_current_age_l1183_118373


namespace NUMINAMATH_GPT_students_failed_l1183_118347

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end NUMINAMATH_GPT_students_failed_l1183_118347


namespace NUMINAMATH_GPT_unique_three_digit_base_g_l1183_118394

theorem unique_three_digit_base_g (g : ℤ) (h : ℤ) (a b c : ℤ) 
  (hg : g > 2) 
  (h_h : h = g + 1 ∨ h = g - 1) 
  (habc_g : a * g^2 + b * g + c = c * h^2 + b * h + a) : 
  a = (g + 1) / 2 ∧ b = (g - 1) / 2 ∧ c = (g - 1) / 2 :=
  sorry

end NUMINAMATH_GPT_unique_three_digit_base_g_l1183_118394


namespace NUMINAMATH_GPT_photographs_taken_l1183_118380

theorem photographs_taken (P : ℝ) (h : P + 0.80 * P = 180) : P = 100 :=
by sorry

end NUMINAMATH_GPT_photographs_taken_l1183_118380


namespace NUMINAMATH_GPT_degenerate_ellipse_b_value_l1183_118377

theorem degenerate_ellipse_b_value :
  ∃ b : ℝ, (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + b = 0 → x = -1 ∧ y = 3) ↔ b = 12 :=
by
  sorry

end NUMINAMATH_GPT_degenerate_ellipse_b_value_l1183_118377


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l1183_118302

theorem greatest_possible_value_of_x (x : ℕ) (H : Nat.lcm (Nat.lcm x 12) 18 = 108) : x ≤ 108 := sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l1183_118302


namespace NUMINAMATH_GPT_Amanda_family_paint_walls_l1183_118397

theorem Amanda_family_paint_walls :
  let num_people := 5
  let rooms_with_4_walls := 5
  let rooms_with_5_walls := 4
  let walls_per_room_4 := 4
  let walls_per_room_5 := 5
  let total_walls := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)
  total_walls / num_people = 8 :=
by
  -- We add a sorry to skip proof
  sorry

end NUMINAMATH_GPT_Amanda_family_paint_walls_l1183_118397


namespace NUMINAMATH_GPT_rectangle_area_in_inscribed_triangle_l1183_118379

theorem rectangle_area_in_inscribed_triangle (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  ∃ (y : ℝ), y = (b * (h - x)) / h ∧ (x * y) = (b * x * (h - x)) / h :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_in_inscribed_triangle_l1183_118379


namespace NUMINAMATH_GPT_option_a_is_correct_l1183_118323

variable (a b : ℝ)
variable (ha : a < 0)
variable (hb : b < 0)
variable (hab : a < b)

theorem option_a_is_correct : (a < abs (3 * a + 2 * b) / 5) ∧ (abs (3 * a + 2 * b) / 5 < b) :=
by
  sorry

end NUMINAMATH_GPT_option_a_is_correct_l1183_118323


namespace NUMINAMATH_GPT_range_of_x_l1183_118369

theorem range_of_x (a b c x : ℝ) (h1 : a^2 + 2 * b^2 + 3 * c^2 = 6) (h2 : a + 2 * b + 3 * c > |x + 1|) : -7 < x ∧ x < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1183_118369


namespace NUMINAMATH_GPT_algebraic_expression_value_l1183_118325

-- Define the given condition
def condition (a b : ℝ) : Prop := a + b - 2 = 0

-- State the theorem to prove the algebraic expression value
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : a^2 - b^2 + 4 * b = 4 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1183_118325


namespace NUMINAMATH_GPT_service_center_location_l1183_118311

def serviceCenterMilepost (x3 x10 : ℕ) (r : ℚ) : ℚ :=
  x3 + r * (x10 - x3)

theorem service_center_location :
  (serviceCenterMilepost 50 170 (2/3) : ℚ) = 130 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_service_center_location_l1183_118311


namespace NUMINAMATH_GPT_value_of_x_l1183_118331

theorem value_of_x (x : ℝ) (h : (10 - x)^2 = x^2 + 4) : x = 24 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1183_118331


namespace NUMINAMATH_GPT_james_drove_75_miles_l1183_118328

noncomputable def james_total_distance : ℝ :=
  let speed1 := 30  -- mph
  let time1 := 0.5  -- hours
  let speed2 := 2 * speed1
  let time2 := 2 * time1
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  distance1 + distance2

theorem james_drove_75_miles : james_total_distance = 75 := by 
  sorry

end NUMINAMATH_GPT_james_drove_75_miles_l1183_118328


namespace NUMINAMATH_GPT_cone_base_radius_l1183_118310

theorem cone_base_radius (R : ℝ) (theta : ℝ) (radius : ℝ) (hR : R = 30) (hTheta : theta = 120) :
    2 * Real.pi * radius = (theta / 360) * 2 * Real.pi * R → radius = 10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cone_base_radius_l1183_118310


namespace NUMINAMATH_GPT_problem_equiv_l1183_118326

theorem problem_equiv {a : ℤ} : (a^2 ≡ 9 [ZMOD 10]) ↔ (a ≡ 3 [ZMOD 10] ∨ a ≡ -3 [ZMOD 10] ∨ a ≡ 7 [ZMOD 10] ∨ a ≡ -7 [ZMOD 10]) :=
sorry

end NUMINAMATH_GPT_problem_equiv_l1183_118326


namespace NUMINAMATH_GPT_min_expr_value_l1183_118334

theorem min_expr_value (α β : ℝ) :
  ∃ (c : ℝ), c = 36 ∧ ((3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = c) :=
sorry

end NUMINAMATH_GPT_min_expr_value_l1183_118334


namespace NUMINAMATH_GPT_length_sixth_episode_l1183_118309

def length_first_episode : ℕ := 58
def length_second_episode : ℕ := 62
def length_third_episode : ℕ := 65
def length_fourth_episode : ℕ := 71
def length_fifth_episode : ℕ := 79
def total_viewing_time : ℕ := 450

theorem length_sixth_episode :
  length_first_episode + length_second_episode + length_third_episode + length_fourth_episode + length_fifth_episode + 115 = total_viewing_time := by
  sorry

end NUMINAMATH_GPT_length_sixth_episode_l1183_118309


namespace NUMINAMATH_GPT_megatech_basic_astrophysics_degrees_l1183_118337

def budget_allocation (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :=
  100 - (microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants)

noncomputable def degrees_for_astrophysics (percentage: ℕ) :=
  (percentage * 360) / 100

theorem megatech_basic_astrophysics_degrees (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  degrees_for_astrophysics (budget_allocation microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants) = 54 :=
by
  sorry

end NUMINAMATH_GPT_megatech_basic_astrophysics_degrees_l1183_118337


namespace NUMINAMATH_GPT_triangle_obtuse_of_eccentricities_l1183_118319

noncomputable def is_obtuse_triangle (a b m : ℝ) : Prop :=
  a^2 + b^2 - m^2 < 0

theorem triangle_obtuse_of_eccentricities (a b m : ℝ) (ha : a > 0) (hm : m > b) (hb : b > 0)
  (ecc_cond : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) > 1) :
  is_obtuse_triangle a b m := 
sorry

end NUMINAMATH_GPT_triangle_obtuse_of_eccentricities_l1183_118319


namespace NUMINAMATH_GPT_find_n_from_equation_l1183_118368

theorem find_n_from_equation :
  ∃ n : ℕ, (3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n) → n = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_equation_l1183_118368


namespace NUMINAMATH_GPT_boxes_with_nothing_l1183_118316

theorem boxes_with_nothing (h_total : 15 = total_boxes)
    (h_pencils : 9 = pencil_boxes)
    (h_pens : 5 = pen_boxes)
    (h_both_pens_and_pencils : 3 = both_pen_and_pencil_boxes)
    (h_markers : 4 = marker_boxes)
    (h_both_markers_and_pencils : 2 = both_marker_and_pencil_boxes)
    (h_no_markers_and_pens : no_marker_and_pen_boxes = 0)
    (h_no_all_three_items : no_all_three_items = 0) :
    ∃ (neither_boxes : ℕ), neither_boxes = 2 :=
by
  sorry

end NUMINAMATH_GPT_boxes_with_nothing_l1183_118316


namespace NUMINAMATH_GPT_total_players_ground_l1183_118365

-- Define the number of players for each type of sport
def c : ℕ := 10
def h : ℕ := 12
def f : ℕ := 16
def s : ℕ := 13

-- Statement of the problem to prove that the total number of players is 51
theorem total_players_ground : c + h + f + s = 51 :=
by
  -- proof will be added later
  sorry

end NUMINAMATH_GPT_total_players_ground_l1183_118365


namespace NUMINAMATH_GPT_domain_of_function_l1183_118352

def valid_domain (x : ℝ) : Prop :=
  x ≤ 3 ∧ x ≠ 0

theorem domain_of_function (x : ℝ) (h₀ : 3 - x ≥ 0) (h₁ : x ≠ 0) : valid_domain x :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1183_118352


namespace NUMINAMATH_GPT_problem_1_problem_2_l1183_118333

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1
theorem problem_1 (x : ℝ) : (∀ x, f x (-2) > 5) ↔ (x < -4 / 3 ∨ x > 2) :=
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) : (∀ x, f x a ≤ a * |x + 3|) → (a ≥ 1 / 2) :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1183_118333


namespace NUMINAMATH_GPT_arithmetic_sequence_a20_l1183_118320

theorem arithmetic_sequence_a20 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2)) → 
  (∃ a : ℕ → ℕ, a 20 = 39) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a20_l1183_118320


namespace NUMINAMATH_GPT_max_value_expression_l1183_118375

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (m : ℝ), (∀ x y : ℝ, x * y > 0 → 
  m ≥ (x / (x + y) + 2 * y / (x + 2 * y))) ∧ 
  m = 4 - 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_value_expression_l1183_118375


namespace NUMINAMATH_GPT_train_crossing_time_l1183_118364

-- Definitions of the given conditions
def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 175

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 14.25 := 
sorry

end NUMINAMATH_GPT_train_crossing_time_l1183_118364


namespace NUMINAMATH_GPT_harkamal_payment_l1183_118354

noncomputable def calculate_total_cost : ℝ :=
  let price_grapes := 8 * 70
  let price_mangoes := 9 * 45
  let price_apples := 5 * 30
  let price_strawberries := 3 * 100
  let price_oranges := 10 * 40
  let price_kiwis := 6 * 60
  let total_grapes_and_apples := price_grapes + price_apples
  let discount_grapes_and_apples := 0.10 * total_grapes_and_apples
  let total_oranges_and_kiwis := price_oranges + price_kiwis
  let discount_oranges_and_kiwis := 0.05 * total_oranges_and_kiwis
  let total_mangoes_and_strawberries := price_mangoes + price_strawberries
  let tax_mangoes_and_strawberries := 0.12 * total_mangoes_and_strawberries
  let total_amount := price_grapes + price_mangoes + price_apples + price_strawberries + price_oranges + price_kiwis
  total_amount - discount_grapes_and_apples - discount_oranges_and_kiwis + tax_mangoes_and_strawberries

theorem harkamal_payment : calculate_total_cost = 2150.6 :=
by
  sorry

end NUMINAMATH_GPT_harkamal_payment_l1183_118354


namespace NUMINAMATH_GPT_polygon_interior_equals_exterior_sum_eq_360_l1183_118386

theorem polygon_interior_equals_exterior_sum_eq_360 (n : ℕ) :
  (n - 2) * 180 = 360 → n = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polygon_interior_equals_exterior_sum_eq_360_l1183_118386


namespace NUMINAMATH_GPT_area_of_park_l1183_118396

variable (length breadth speed time perimeter area : ℕ)

axiom ratio_length_breadth : length = breadth / 4
axiom speed_kmh : speed = 12 * 1000 / 60 -- speed in m/min
axiom time_taken : time = 8 -- time in minutes
axiom perimeter_eq : perimeter = speed * time -- perimeter in meters
axiom length_breadth_relation : perimeter = 2 * (length + breadth)

theorem area_of_park : ∃ length breadth, (length = 160 ∧ breadth = 640 ∧ area = length * breadth ∧ area = 102400) :=
by
  sorry

end NUMINAMATH_GPT_area_of_park_l1183_118396


namespace NUMINAMATH_GPT_T_expansion_l1183_118395

def T (x : ℝ) : ℝ := (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1

theorem T_expansion (x : ℝ) : T x = (x - 1)^5 := by
  sorry

end NUMINAMATH_GPT_T_expansion_l1183_118395


namespace NUMINAMATH_GPT_number_of_people_l1183_118356

-- Conditions
def cost_oysters : ℤ := 3 * 15
def cost_shrimp : ℤ := 2 * 14
def cost_clams : ℤ := 2 * 135 / 10  -- Using integers for better precision
def total_cost : ℤ := cost_oysters + cost_shrimp + cost_clams
def amount_owed_each_person : ℤ := 25

-- Goal
theorem number_of_people (number_of_people : ℤ) : total_cost = number_of_people * amount_owed_each_person → number_of_people = 4 := by
  -- Proof to be completed here.
  sorry

end NUMINAMATH_GPT_number_of_people_l1183_118356


namespace NUMINAMATH_GPT_find_B_intersection_point_l1183_118378

theorem find_B_intersection_point (k1 k2 : ℝ) (hA1 : 1 ≠ 0) 
  (hA2 : k1 = -2) (hA3 : k2 = -2) : 
  (-1, 2) ∈ {p : ℝ × ℝ | ∃ k1 k2, p.2 = k1 * p.1 ∧ p.2 = k2 / p.1} :=
sorry

end NUMINAMATH_GPT_find_B_intersection_point_l1183_118378


namespace NUMINAMATH_GPT_dot_product_equilateral_l1183_118359

-- Define the conditions for the equilateral triangle ABC
variable {A B C : ℝ}

noncomputable def equilateral_triangle (A B C : ℝ) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ |A - B| = 1 ∧ |B - C| = 1 ∧ |C - A| = 1

-- Define the dot product of the vectors AB and BC
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_equilateral (A B C : ℝ) (h : equilateral_triangle A B C) : 
  dot_product (B - A, 0) (C - B, 0) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_dot_product_equilateral_l1183_118359


namespace NUMINAMATH_GPT_centroid_of_triangle_PQR_positions_l1183_118336

-- Define the basic setup
def square_side_length : ℕ := 12
def total_points : ℕ := 48

-- Define the centroid calculation condition
def centroid_positions_count : ℕ :=
  let side_segments := square_side_length
  let points_per_edge := total_points / 4
  let possible_positions_per_side := points_per_edge - 1
  (possible_positions_per_side * possible_positions_per_side)

/-- Proof statement: Proving the number of possible positions for the centroid of triangle PQR 
    formed by any three non-collinear points out of the 48 points on the perimeter of the square. --/
theorem centroid_of_triangle_PQR_positions : centroid_positions_count = 121 := 
  sorry

end NUMINAMATH_GPT_centroid_of_triangle_PQR_positions_l1183_118336


namespace NUMINAMATH_GPT_intersection_at_one_point_l1183_118349

theorem intersection_at_one_point (b : ℝ) :
  (∃ x₀ : ℝ, bx^2 + 7*x₀ + 4 = 0 ∧ (7)^2 - 4*b*4 = 0) →
  b = 49 / 16 :=
by
  sorry

end NUMINAMATH_GPT_intersection_at_one_point_l1183_118349


namespace NUMINAMATH_GPT_total_students_in_class_l1183_118344

-- Define the initial conditions
def num_students_in_row (a b: Nat) : Nat := a + 1 + b
def num_lines : Nat := 3
noncomputable def students_in_row : Nat := num_students_in_row 2 5 

-- Theorem to prove the total number of students in the class
theorem total_students_in_class : students_in_row * num_lines = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1183_118344


namespace NUMINAMATH_GPT_range_of_z_in_parallelogram_l1183_118382

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := {x := -1, y := 2}
def B : Point := {x := 3, y := 4}
def C : Point := {x := 4, y := -2}

-- Define the condition for point (x, y) to be inside the parallelogram (including boundary)
def isInsideParallelogram (p : Point) : Prop := sorry -- Placeholder for actual geometric condition

-- Statement of the problem
theorem range_of_z_in_parallelogram (p : Point) (h : isInsideParallelogram p) : 
  -14 ≤ 2 * p.x - 5 * p.y ∧ 2 * p.x - 5 * p.y ≤ 20 :=
sorry

end NUMINAMATH_GPT_range_of_z_in_parallelogram_l1183_118382


namespace NUMINAMATH_GPT_purple_valley_skirts_l1183_118362

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end NUMINAMATH_GPT_purple_valley_skirts_l1183_118362


namespace NUMINAMATH_GPT_find_x_l1183_118385

theorem find_x (c d : ℝ) (y z x : ℝ) 
  (h1 : y^2 = c * z^2) 
  (h2 : y = d / x)
  (h3 : y = 3) 
  (h4 : x = 4) 
  (h5 : z = 6) 
  (h6 : y = 2) 
  (h7 : z = 12) 
  : x = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1183_118385


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_common_ratio_l1183_118312

theorem geometric_arithmetic_sequence_common_ratio (a_1 a_2 a_3 q : ℝ) 
  (h1 : a_2 = a_1 * q) 
  (h2 : a_3 = a_1 * q^2)
  (h3 : 2 * a_3 = a_1 + a_2) : (q = 1) ∨ (q = -1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_common_ratio_l1183_118312


namespace NUMINAMATH_GPT_petroleum_crude_oil_problem_l1183_118357

variables (x y : ℝ)

theorem petroleum_crude_oil_problem (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 27.5) : y = 30 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_petroleum_crude_oil_problem_l1183_118357


namespace NUMINAMATH_GPT_evaluate_fg_of_8_l1183_118304

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem evaluate_fg_of_8 : f (g 8) = 211 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fg_of_8_l1183_118304


namespace NUMINAMATH_GPT_smallest_positive_value_l1183_118367

theorem smallest_positive_value (c d : ℤ) (h : c^2 > d^2) : 
  ∃ m > 0, m = (c^2 + d^2) / (c^2 - d^2) + (c^2 - d^2) / (c^2 + d^2) ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_value_l1183_118367


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1183_118392

theorem hyperbola_asymptote (x y : ℝ) : 
  (∀ x y : ℝ, (x^2 / 25 - y^2 / 16 = 1) → (y = (4 / 5) * x ∨ y = -(4 / 5) * x)) := 
by 
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1183_118392


namespace NUMINAMATH_GPT_exists_larger_integer_l1183_118370

theorem exists_larger_integer (a b : Nat) (h1 : b > a) (h2 : b - a = 5) (h3 : a * b = 88) :
  b = 11 :=
sorry

end NUMINAMATH_GPT_exists_larger_integer_l1183_118370


namespace NUMINAMATH_GPT_minimal_connections_correct_l1183_118383

-- Define a Lean structure to encapsulate the conditions
structure IslandsProblem where
  islands : ℕ
  towns : ℕ
  min_towns_per_island : ℕ
  condition_islands : islands = 13
  condition_towns : towns = 25
  condition_min_towns : min_towns_per_island = 1

-- Define a function to represent the minimal number of ferry connections
def minimalFerryConnections (p : IslandsProblem) : ℕ :=
  222

-- Define the statement to be proved
theorem minimal_connections_correct (p : IslandsProblem) : 
  p.islands = 13 → 
  p.towns = 25 → 
  p.min_towns_per_island = 1 → 
  minimalFerryConnections p = 222 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimal_connections_correct_l1183_118383


namespace NUMINAMATH_GPT_flower_beds_and_circular_path_fraction_l1183_118398

noncomputable def occupied_fraction 
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ) : ℝ :=
  let flower_bed_area := 2 * (1 / 2 : ℝ) * triangle_leg^2
  let circular_path_area := Real.pi * circle_radius ^ 2
  let occupied_area := flower_bed_area + circular_path_area
  occupied_area / (yard_length * yard_width)

theorem flower_beds_and_circular_path_fraction
  (yard_length : ℕ)
  (yard_width : ℕ)
  (side1 : ℕ)
  (side2 : ℕ)
  (triangle_leg : ℕ)
  (circle_radius : ℕ)
  (h1 : side1 = 20)
  (h2 : side2 = 30)
  (h3 : triangle_leg = (side2 - side1) / 2)
  (h4 : yard_length = 30)
  (h5 : yard_width = 5)
  (h6 : circle_radius = 2) :
  occupied_fraction yard_length yard_width side1 side2 triangle_leg circle_radius = (25 + 4 * Real.pi) / 150 :=
by sorry

end NUMINAMATH_GPT_flower_beds_and_circular_path_fraction_l1183_118398


namespace NUMINAMATH_GPT_minimum_value_frac_l1183_118305

theorem minimum_value_frac (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 2) :
  (p + q) / (p * q * r) ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_frac_l1183_118305


namespace NUMINAMATH_GPT_cos_two_pi_over_three_l1183_118376

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_two_pi_over_three_l1183_118376


namespace NUMINAMATH_GPT_expression_bounds_l1183_118301

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_expression_bounds_l1183_118301


namespace NUMINAMATH_GPT_no_solution_for_x_l1183_118313

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, (1 / (x - 4)) + (m / (x + 4)) ≠ ((m + 3) / (x^2 - 16))) ↔ (m = -1 ∨ m = 5 ∨ m = -1 / 3) :=
sorry

end NUMINAMATH_GPT_no_solution_for_x_l1183_118313


namespace NUMINAMATH_GPT_servings_correct_l1183_118342

-- Define the pieces of popcorn in a serving
def pieces_per_serving := 30

-- Define the pieces of popcorn Jared can eat
def jared_pieces := 90

-- Define the pieces of popcorn each friend can eat
def friend_pieces := 60

-- Define the number of friends
def friends := 3

-- Calculate total pieces eaten by friends
def total_friend_pieces := friends * friend_pieces

-- Calculate total pieces eaten by everyone
def total_pieces := jared_pieces + total_friend_pieces

-- Calculate the number of servings needed
def servings_needed := total_pieces / pieces_per_serving

theorem servings_correct : servings_needed = 9 :=
by
  sorry

end NUMINAMATH_GPT_servings_correct_l1183_118342


namespace NUMINAMATH_GPT_boy_completion_time_l1183_118390

theorem boy_completion_time (M W B : ℝ) (h1 : M + W + B = 1/3) (h2 : M = 1/6) (h3 : W = 1/18) : B = 1/9 :=
sorry

end NUMINAMATH_GPT_boy_completion_time_l1183_118390


namespace NUMINAMATH_GPT_number_of_valid_trapezoids_l1183_118351

noncomputable def calculate_number_of_trapezoids : ℕ :=
  let rows_1 := 7
  let rows_2 := 9
  let unit_spacing := 1
  let height := 2
  -- Here, we should encode the actual combinatorial calculation as per the problem solution
  -- but for the Lean 4 statement, we will provide the correct answer directly.
  361

theorem number_of_valid_trapezoids :
  calculate_number_of_trapezoids = 361 :=
sorry

end NUMINAMATH_GPT_number_of_valid_trapezoids_l1183_118351


namespace NUMINAMATH_GPT_num_pieces_l1183_118353

theorem num_pieces (total_length : ℝ) (piece_length : ℝ) 
  (h1: total_length = 253.75) (h2: piece_length = 0.425) :
  ⌊total_length / piece_length⌋ = 597 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_num_pieces_l1183_118353


namespace NUMINAMATH_GPT_fraction_multiplication_l1183_118314

theorem fraction_multiplication : (1 / 2) * (1 / 3) * (1 / 6) * 108 = 3 := by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1183_118314


namespace NUMINAMATH_GPT_two_point_seven_five_as_fraction_l1183_118358

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_two_point_seven_five_as_fraction_l1183_118358


namespace NUMINAMATH_GPT_S_40_value_l1183_118381

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom h1 : S 10 = 10
axiom h2 : S 30 = 70

theorem S_40_value : S 40 = 150 :=
by
  -- Conditions
  have h1 : S 10 = 10 := h1
  have h2 : S 30 = 70 := h2
  -- Start proof here
  sorry

end NUMINAMATH_GPT_S_40_value_l1183_118381


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1183_118389

theorem problem1 : (-3) - (-5) - 6 + (-4) = -8 := by sorry

theorem problem2 : ((1 / 9) + (1 / 6) - (1 / 2)) / (-1 / 18) = 4 := by sorry

theorem problem3 : -1^4 + abs (3 - 6) - 2 * (-2) ^ 2 = -6 := by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1183_118389


namespace NUMINAMATH_GPT_age_difference_proof_l1183_118329

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_proof_l1183_118329


namespace NUMINAMATH_GPT_Eunji_higher_than_Yoojung_l1183_118300

-- Define floors for Yoojung and Eunji
def Yoojung_floor: ℕ := 17
def Eunji_floor: ℕ := 25

-- Assert that Eunji lives on a higher floor than Yoojung
theorem Eunji_higher_than_Yoojung : Eunji_floor > Yoojung_floor :=
  by
    sorry

end NUMINAMATH_GPT_Eunji_higher_than_Yoojung_l1183_118300


namespace NUMINAMATH_GPT_beyonce_album_songs_l1183_118350

theorem beyonce_album_songs
  (singles : ℕ)
  (album1_songs album2_songs album3_songs total_songs : ℕ)
  (h1 : singles = 5)
  (h2 : album1_songs = 15)
  (h3 : album2_songs = 15)
  (h4 : total_songs = 55) :
  album3_songs = 20 :=
by
  sorry

end NUMINAMATH_GPT_beyonce_album_songs_l1183_118350


namespace NUMINAMATH_GPT_even_integers_diff_digits_200_to_800_l1183_118366

theorem even_integers_diff_digits_200_to_800 :
  ∃ n : ℕ, n = 131 ∧ (∀ x : ℕ, 200 ≤ x ∧ x < 800 ∧ (x % 2 = 0) ∧ (∀ i j : ℕ, i ≠ j → (x / 10^i % 10) ≠ (x / 10^j % 10)) ↔ x < n) :=
sorry

end NUMINAMATH_GPT_even_integers_diff_digits_200_to_800_l1183_118366


namespace NUMINAMATH_GPT_train_speed_l1183_118374

def length_of_train : ℝ := 160
def time_to_cross : ℝ := 18
def speed_in_kmh : ℝ := 32

theorem train_speed :
  (length_of_train / time_to_cross) * 3.6 = speed_in_kmh :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1183_118374


namespace NUMINAMATH_GPT_initial_amount_spent_l1183_118371

theorem initial_amount_spent
    (X : ℕ) -- initial amount of money to spend
    (sets_purchased : ℕ := 250) -- total sets purchased
    (sets_cost_20 : ℕ := 178) -- sets that cost $20 each
    (price_per_set : ℕ := 20) -- price of each set that cost $20
    (remaining_sets : ℕ := sets_purchased - sets_cost_20) -- remaining sets
    (spent_all : (X = sets_cost_20 * price_per_set + remaining_sets * 0)) -- spent all money, remaining sets assumed free to simplify as the exact price is not given or necessary
    : X = 3560 :=
    by
    sorry

end NUMINAMATH_GPT_initial_amount_spent_l1183_118371


namespace NUMINAMATH_GPT_average_production_last_5_days_l1183_118387

theorem average_production_last_5_days (tv_per_day_25 : ℕ) (total_tv_30 : ℕ) :
  tv_per_day_25 = 63 →
  total_tv_30 = 58 * 30 →
  (total_tv_30 - tv_per_day_25 * 25) / 5 = 33 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_average_production_last_5_days_l1183_118387


namespace NUMINAMATH_GPT_longest_possible_height_l1183_118343

theorem longest_possible_height (a b c : ℕ) (ha : a = 3 * c) (hb : b * 4 = 12 * c) (h_tri : a - c < b) (h_unequal : ¬(a = c)) :
  ∃ x : ℕ, (4 < x ∧ x < 6) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_longest_possible_height_l1183_118343


namespace NUMINAMATH_GPT_lemonade_glasses_l1183_118339

def lemons_total : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_glasses : lemons_total / lemons_per_glass = 9 := by
  sorry

end NUMINAMATH_GPT_lemonade_glasses_l1183_118339


namespace NUMINAMATH_GPT_angle_E_in_quadrilateral_l1183_118338

theorem angle_E_in_quadrilateral (E F G H : ℝ) 
  (h1 : E = 5 * H)
  (h2 : E = 4 * G)
  (h3 : E = (5/3) * F)
  (h_sum : E + F + G + H = 360) : 
  E = 131 := by 
  sorry

end NUMINAMATH_GPT_angle_E_in_quadrilateral_l1183_118338


namespace NUMINAMATH_GPT_matrix_system_solution_range_l1183_118306

theorem matrix_system_solution_range (m : ℝ) :
  (∃ x y: ℝ, 
    (m * x + y = m + 1) ∧ 
    (x + m * y = 2 * m)) ↔ m ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_matrix_system_solution_range_l1183_118306


namespace NUMINAMATH_GPT_chloromethane_formation_l1183_118372

variable (CH₄ Cl₂ CH₃Cl : Type)
variable (molesCH₄ molesCl₂ molesCH₃Cl : ℕ)

theorem chloromethane_formation 
  (h₁ : molesCH₄ = 3)
  (h₂ : molesCl₂ = 3)
  (reaction : CH₄ → Cl₂ → CH₃Cl)
  (one_to_one : ∀ (x y : ℕ), x = y → x = y): 
  molesCH₃Cl = 3 :=
by
  sorry

end NUMINAMATH_GPT_chloromethane_formation_l1183_118372


namespace NUMINAMATH_GPT_initial_notebooks_l1183_118363

variable (a n : ℕ)
variable (h1 : n = 13 * a + 8)
variable (h2 : n = 15 * a)

theorem initial_notebooks : n = 60 := by
  -- additional details within the proof
  sorry

end NUMINAMATH_GPT_initial_notebooks_l1183_118363


namespace NUMINAMATH_GPT_b_investment_l1183_118317

noncomputable def B_share := 880
noncomputable def A_share := 560
noncomputable def A_investment := 7000
noncomputable def C_investment := 18000
noncomputable def total_investment (B: ℝ) := A_investment + B + C_investment

theorem b_investment (B : ℝ) (P : ℝ)
    (h1 : 7000 / total_investment B * P = A_share)
    (h2 : B / total_investment B * P = B_share) : B = 8000 :=
by
  sorry

end NUMINAMATH_GPT_b_investment_l1183_118317


namespace NUMINAMATH_GPT_marble_probability_l1183_118335

theorem marble_probability :
  let total_ways := (Nat.choose 6 4)
  let favorable_ways := 
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1)
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_marble_probability_l1183_118335


namespace NUMINAMATH_GPT_symmetric_points_y_axis_l1183_118307

theorem symmetric_points_y_axis :
  ∀ (m n : ℝ), (m + 4 = 0) → (n = 3) → (m + n) ^ 2023 = -1 :=
by
  intros m n Hm Hn
  sorry

end NUMINAMATH_GPT_symmetric_points_y_axis_l1183_118307


namespace NUMINAMATH_GPT_common_root_of_two_equations_l1183_118332

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end NUMINAMATH_GPT_common_root_of_two_equations_l1183_118332


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1183_118393

theorem maximum_value_of_expression
  (a b c : ℝ)
  (h1 : 0 ≤ a)
  (h2 : 0 ≤ b)
  (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + 2 * c^2 = 1) :
  ab * Real.sqrt 3 + 3 * bc ≤ Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1183_118393
