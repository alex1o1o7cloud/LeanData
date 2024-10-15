import Mathlib

namespace NUMINAMATH_GPT_other_number_is_300_l1413_141376

theorem other_number_is_300 (A B : ℕ) (h1 : A = 231) (h2 : lcm A B = 2310) (h3 : gcd A B = 30) : B = 300 := by
  sorry

end NUMINAMATH_GPT_other_number_is_300_l1413_141376


namespace NUMINAMATH_GPT_double_inequality_pos_reals_equality_condition_l1413_141396

theorem double_inequality_pos_reals (x y z : ℝ) (x_pos: 0 < x) (y_pos: 0 < y) (z_pos: 0 < z):
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ (1 / 8) :=
  sorry

theorem equality_condition (x y z : ℝ) :
  ((1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) = (1 / 8)) ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
  sorry

end NUMINAMATH_GPT_double_inequality_pos_reals_equality_condition_l1413_141396


namespace NUMINAMATH_GPT_max_lateral_surface_area_of_pyramid_l1413_141350

theorem max_lateral_surface_area_of_pyramid (a h : ℝ) (r : ℝ) (h_eq : 2 * a^2 + h^2 = 4) (r_eq : r = 1) :
  ∃ (a : ℝ), (a = 1) :=
by
sorry

end NUMINAMATH_GPT_max_lateral_surface_area_of_pyramid_l1413_141350


namespace NUMINAMATH_GPT_unoccupied_cylinder_volume_l1413_141388

theorem unoccupied_cylinder_volume (r h : ℝ) (V_cylinder V_cone : ℝ) :
  r = 15 ∧ h = 30 ∧ V_cylinder = π * r^2 * h ∧ V_cone = (1/3) * π * r^2 * (r / 2) →
  V_cylinder - 2 * V_cone = 4500 * π :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_unoccupied_cylinder_volume_l1413_141388


namespace NUMINAMATH_GPT_parallel_lines_intersect_hyperbola_l1413_141372

noncomputable def point_A : (ℝ × ℝ) := (0, 14)
noncomputable def point_B : (ℝ × ℝ) := (0, 4)
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

theorem parallel_lines_intersect_hyperbola (k : ℝ)
  (x_K x_L x_M x_N : ℝ) 
  (hAK : hyperbola x_K = k * x_K + 14) (hAL : hyperbola x_L = k * x_L + 14)
  (hBM : hyperbola x_M = k * x_M + 4) (hBN : hyperbola x_N = k * x_N + 4)
  (vieta1 : x_K + x_L = -14 / k) (vieta2 : x_M + x_N = -4 / k) :
  (AL - AK) / (BN - BM) = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_intersect_hyperbola_l1413_141372


namespace NUMINAMATH_GPT_percentage_microphotonics_l1413_141344

noncomputable def percentage_home_electronics : ℝ := 24
noncomputable def percentage_food_additives : ℝ := 20
noncomputable def percentage_GMO : ℝ := 29
noncomputable def percentage_industrial_lubricants : ℝ := 8
noncomputable def angle_basic_astrophysics : ℝ := 18

theorem percentage_microphotonics : 
  ∀ (home_elec food_additives GMO industrial_lub angle_bas_astro : ℝ),
  home_elec = 24 →
  food_additives = 20 →
  GMO = 29 →
  industrial_lub = 8 →
  angle_bas_astro = 18 →
  (100 - (home_elec + food_additives + GMO + industrial_lub + ((angle_bas_astro / 360) * 100))) = 14 :=
by
  intros _ _ _ _ _
  sorry

end NUMINAMATH_GPT_percentage_microphotonics_l1413_141344


namespace NUMINAMATH_GPT_mean_of_added_numbers_l1413_141314

noncomputable def mean (a : List ℚ) : ℚ :=
  (a.sum) / (a.length)

theorem mean_of_added_numbers 
  (sum_eight_numbers : ℚ)
  (sum_eleven_numbers : ℚ)
  (x y z : ℚ)
  (h_eight : sum_eight_numbers = 8 * 72)
  (h_eleven : sum_eleven_numbers = 11 * 85)
  (h_sum_added : x + y + z = sum_eleven_numbers - sum_eight_numbers) :
  (x + y + z) / 3 = 119 + 2/3 := 
sorry

end NUMINAMATH_GPT_mean_of_added_numbers_l1413_141314


namespace NUMINAMATH_GPT_probability_correct_l1413_141359
noncomputable def probability_no_2_in_id : ℚ :=
  let total_ids := 5000
  let valid_ids := 2916
  valid_ids / total_ids

theorem probability_correct : probability_no_2_in_id = 729 / 1250 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l1413_141359


namespace NUMINAMATH_GPT_max_bees_in_largest_beehive_l1413_141360

def total_bees : ℕ := 2000000
def beehives : ℕ := 7
def min_ratio : ℚ := 0.7

theorem max_bees_in_largest_beehive (B_max : ℚ) : 
  (6 * (min_ratio * B_max) + B_max = total_bees) → 
  B_max <= 2000000 / 5.2 ∧ B_max.floor = 384615 :=
by
  sorry

end NUMINAMATH_GPT_max_bees_in_largest_beehive_l1413_141360


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1413_141370

-- Define the three given points
structure Point where
  x : ℝ
  y : ℝ

def p1 : Point := ⟨1, 3⟩
def p2 : Point := ⟨5, -1⟩
def p3 : Point := ⟨10, 3⟩

-- Define the statement that the distance between the foci of the ellipse they define is 2 * sqrt(4.25)
theorem distance_between_foci_of_ellipse : 
  ∃ (c : ℝ) (f : ℝ), f = 2 * Real.sqrt 4.25 ∧ 
  (∃ (ellipse : Point → Prop), ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1413_141370


namespace NUMINAMATH_GPT_least_even_p_l1413_141304

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem least_even_p 
  (p : ℕ) 
  (hp : 2 ∣ p) -- p is an even integer
  (h : is_square (300 * p)) -- 300 * p is the square of an integer
  : p = 3 := 
sorry

end NUMINAMATH_GPT_least_even_p_l1413_141304


namespace NUMINAMATH_GPT_reduced_price_l1413_141357

variable (original_price : ℝ) (final_amount : ℝ)

noncomputable def sales_tax (price : ℝ) : ℝ :=
  if price <= 2500 then price * 0.04
  else if price <= 4500 then 2500 * 0.04 + (price - 2500) * 0.07
  else 2500 * 0.04 + 2000 * 0.07 + (price - 4500) * 0.09

noncomputable def discount (price : ℝ) : ℝ :=
  if price <= 2000 then price * 0.02
  else if price <= 4000 then 2000 * 0.02 + (price - 2000) * 0.05
  else 2000 * 0.02 + 2000 * 0.05 + (price - 4000) * 0.10

theorem reduced_price (P : ℝ) (original_price := 5000) (final_amount := 2468) :
  P = original_price - discount original_price + sales_tax original_price → P = 2423 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_l1413_141357


namespace NUMINAMATH_GPT_soccer_games_per_month_l1413_141326

theorem soccer_games_per_month (total_games : ℕ) (months : ℕ) (h1 : total_games = 27) (h2 : months = 3) : total_games / months = 9 :=
by 
  sorry

end NUMINAMATH_GPT_soccer_games_per_month_l1413_141326


namespace NUMINAMATH_GPT_arrangement_count_l1413_141386

theorem arrangement_count (students : Fin 6) (teacher : Bool) :
  (teacher = true) ∧
  ∀ (A B : Fin 6), 
    A ≠ 0 ∧ B ≠ 5 →
    A ≠ B →
    (Sorry) = 960 := sorry

end NUMINAMATH_GPT_arrangement_count_l1413_141386


namespace NUMINAMATH_GPT_inequalities_hold_l1413_141339

variable {a b c x y z : ℝ}

theorem inequalities_hold 
  (h1 : x ≤ a)
  (h2 : y ≤ b)
  (h3 : z ≤ c) :
  x * y + y * z + z * x ≤ a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  x * y * z ≤ a * b * c :=
sorry

end NUMINAMATH_GPT_inequalities_hold_l1413_141339


namespace NUMINAMATH_GPT_y_relation_l1413_141395

noncomputable def f (x : ℝ) : ℝ := -2 * x + 5

theorem y_relation (x1 y1 y2 y3 : ℝ) (h1 : y1 = f x1) (h2 : y2 = f (x1 - 2)) (h3 : y3 = f (x1 + 3)) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end NUMINAMATH_GPT_y_relation_l1413_141395


namespace NUMINAMATH_GPT_units_digit_17_pow_2007_l1413_141361

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end NUMINAMATH_GPT_units_digit_17_pow_2007_l1413_141361


namespace NUMINAMATH_GPT_total_surface_area_of_pyramid_l1413_141340

noncomputable def base_length_ab : ℝ := 8 -- Length of side AB
noncomputable def base_length_ad : ℝ := 6 -- Length of side AD
noncomputable def height_pf : ℝ := 15 -- Perpendicular height from peak P to the base's center F

noncomputable def base_area : ℝ := base_length_ab * base_length_ad
noncomputable def fm_distance : ℝ := Real.sqrt ((base_length_ab / 2)^2 + (base_length_ad / 2)^2)
noncomputable def slant_height_pm : ℝ := Real.sqrt (height_pf^2 + fm_distance^2)

noncomputable def lateral_area_ab : ℝ := 2 * (0.5 * base_length_ab * slant_height_pm)
noncomputable def lateral_area_ad : ℝ := 2 * (0.5 * base_length_ad * slant_height_pm)
noncomputable def total_surface_area : ℝ := base_area + lateral_area_ab + lateral_area_ad

theorem total_surface_area_of_pyramid :
  total_surface_area = 48 + 55 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_pyramid_l1413_141340


namespace NUMINAMATH_GPT_drive_time_is_eleven_hours_l1413_141385

-- Define the distances and speed as constants
def distance_salt_lake_to_vegas : ℕ := 420
def distance_vegas_to_los_angeles : ℕ := 273
def average_speed : ℕ := 63

-- Calculate the total distance
def total_distance : ℕ := distance_salt_lake_to_vegas + distance_vegas_to_los_angeles

-- Calculate the total time required
def total_time : ℕ := total_distance / average_speed

-- Theorem stating Andy wants to complete the drive in 11 hours
theorem drive_time_is_eleven_hours : total_time = 11 := sorry

end NUMINAMATH_GPT_drive_time_is_eleven_hours_l1413_141385


namespace NUMINAMATH_GPT_percentage_of_ginger_is_correct_l1413_141334

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_ginger_is_correct_l1413_141334


namespace NUMINAMATH_GPT_problem1_calculation_l1413_141384

theorem problem1_calculation :
  (2 * Real.tan (Real.pi / 4) + (-1 / 2) ^ 0 + |Real.sqrt 3 - 1|) = 2 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_calculation_l1413_141384


namespace NUMINAMATH_GPT_alex_shirts_l1413_141391

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end NUMINAMATH_GPT_alex_shirts_l1413_141391


namespace NUMINAMATH_GPT_divisible_l1413_141355

def P (x : ℝ) : ℝ := 6 * x^3 + x^2 - 1
def Q (x : ℝ) : ℝ := 2 * x - 1

theorem divisible : ∃ R : ℝ → ℝ, ∀ x : ℝ, P x = Q x * R x :=
sorry

end NUMINAMATH_GPT_divisible_l1413_141355


namespace NUMINAMATH_GPT_compute_expression_l1413_141365

theorem compute_expression : 12 * (1 / 7) * 14 * 2 = 48 := 
sorry

end NUMINAMATH_GPT_compute_expression_l1413_141365


namespace NUMINAMATH_GPT_not_periodic_cos_add_cos_sqrt2_l1413_141311

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x * Real.sqrt 2)

theorem not_periodic_cos_add_cos_sqrt2 :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end NUMINAMATH_GPT_not_periodic_cos_add_cos_sqrt2_l1413_141311


namespace NUMINAMATH_GPT_perimeter_of_figure_composed_of_squares_l1413_141369

theorem perimeter_of_figure_composed_of_squares
  (n : ℕ)
  (side_length : ℝ)
  (square_perimeter : ℝ := 4 * side_length)
  (total_squares : ℕ := 7)
  (total_perimeter_if_independent : ℝ := square_perimeter * total_squares)
  (meet_at_vertices : ∀ i j : ℕ, i ≠ j → ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ¬(s1 = s2))
  : total_perimeter_if_independent = 28 :=
by sorry

end NUMINAMATH_GPT_perimeter_of_figure_composed_of_squares_l1413_141369


namespace NUMINAMATH_GPT_notepad_last_duration_l1413_141392

def note_duration (folds_per_paper : ℕ) (pieces_of_paper : ℕ) (notes_per_day : ℕ) : ℕ :=
  let note_size_papers_per_letter_paper := 2 ^ folds_per_paper
  let total_note_size_papers := pieces_of_paper * note_size_papers_per_letter_paper
  total_note_size_papers / notes_per_day

theorem notepad_last_duration :
  note_duration 3 5 10 = 4 := by
  sorry

end NUMINAMATH_GPT_notepad_last_duration_l1413_141392


namespace NUMINAMATH_GPT_average_of_data_set_l1413_141375

theorem average_of_data_set :
  (7 + 5 + (-2) + 5 + 10) / 5 = 5 :=
by sorry

end NUMINAMATH_GPT_average_of_data_set_l1413_141375


namespace NUMINAMATH_GPT_montoya_food_budget_l1413_141377

theorem montoya_food_budget (g t e : ℝ) (h1 : g = 0.6) (h2 : t = 0.8) : e = 0.2 :=
by sorry

end NUMINAMATH_GPT_montoya_food_budget_l1413_141377


namespace NUMINAMATH_GPT_sqrt_ab_equals_sqrt_2_l1413_141316

theorem sqrt_ab_equals_sqrt_2 
  (a b : ℝ)
  (h1 : a ^ 2 = 16 / 25)
  (h2 : b ^ 3 = 125 / 8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := 
by 
  -- proof will go here
  sorry

end NUMINAMATH_GPT_sqrt_ab_equals_sqrt_2_l1413_141316


namespace NUMINAMATH_GPT_triangle_area_l1413_141312

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) 
                      (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b) :
  ∃ A : ℝ, A = 2 * Real.sqrt 6 ∧
    ∃ (h : 0 ≤ A), A = (Real.sqrt (A * 12 * (12 - a) * (12 - b) * (12 - c))) :=
sorry

end NUMINAMATH_GPT_triangle_area_l1413_141312


namespace NUMINAMATH_GPT_problem_expression_value_l1413_141308

theorem problem_expression_value {a b c k1 k2 : ℂ} 
  (h_root : ∀ x, x^3 - k1 * x - k2 = 0 → x = a ∨ x = b ∨ x = c) 
  (h_condition : k1 + k2 ≠ 1)
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : a * b + b * c + c * a = -k1)
  (h_vieta3 : a * b * c = k2) :
  (1 + a)/(1 - a) + (1 + b)/(1 - b) + (1 + c)/(1 - c) = 
  (3 + k1 + 3 * k2)/(1 - k1 - k2) :=
by
  sorry

end NUMINAMATH_GPT_problem_expression_value_l1413_141308


namespace NUMINAMATH_GPT_factorize_expression_l1413_141301

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1413_141301


namespace NUMINAMATH_GPT_eval_expression_l1413_141302

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1413_141302


namespace NUMINAMATH_GPT_sara_total_spent_l1413_141325

def cost_of_tickets := 2 * 10.62
def cost_of_renting := 1.59
def cost_of_buying := 13.95
def total_spent := cost_of_tickets + cost_of_renting + cost_of_buying

theorem sara_total_spent : total_spent = 36.78 := by
  sorry

end NUMINAMATH_GPT_sara_total_spent_l1413_141325


namespace NUMINAMATH_GPT_equal_sum_seq_value_at_18_l1413_141398

-- Define what it means for a sequence to be an equal-sum sequence with a common sum
def equal_sum_seq (a : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_seq_value_at_18
  (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : equal_sum_seq a 5) :
  a 18 = 3 :=
sorry

end NUMINAMATH_GPT_equal_sum_seq_value_at_18_l1413_141398


namespace NUMINAMATH_GPT_maximize_det_l1413_141323

theorem maximize_det (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  (Matrix.det ![
    ![a, 1],
    ![1, b]
  ]) ≤ 0 :=
sorry

end NUMINAMATH_GPT_maximize_det_l1413_141323


namespace NUMINAMATH_GPT_car_return_speed_l1413_141303

noncomputable def round_trip_speed (d : ℝ) (r : ℝ) : ℝ :=
  let travel_time_to_B := d / 75
  let break_time := 1 / 2
  let travel_time_to_A := d / r
  let total_time := travel_time_to_B + travel_time_to_A + break_time
  let total_distance := 2 * d
  total_distance / total_time

theorem car_return_speed :
  let d := 150
  let avg_speed := 50
  round_trip_speed d 42.857 = avg_speed :=
by
  sorry

end NUMINAMATH_GPT_car_return_speed_l1413_141303


namespace NUMINAMATH_GPT_river_and_building_geometry_l1413_141373

open Real

theorem river_and_building_geometry (x y : ℝ) :
  (tan 60 * x = y) ∧ (tan 30 * (x + 30) = y) → x = 15 ∧ y = 15 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_river_and_building_geometry_l1413_141373


namespace NUMINAMATH_GPT_avg_price_of_pencil_l1413_141329

theorem avg_price_of_pencil 
  (total_pens : ℤ) (total_pencils : ℤ) (total_cost : ℤ)
  (avg_cost_pen : ℤ) (avg_cost_pencil : ℤ) :
  total_pens = 30 → 
  total_pencils = 75 → 
  total_cost = 690 → 
  avg_cost_pen = 18 → 
  (total_cost - total_pens * avg_cost_pen) / total_pencils = avg_cost_pencil → 
  avg_cost_pencil = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_avg_price_of_pencil_l1413_141329


namespace NUMINAMATH_GPT_count_solutions_l1413_141347

theorem count_solutions : 
  (∃ (n : ℕ), ∀ (x : ℕ), (x + 17) % 43 = 71 % 43 ∧ x < 150 → n = 4) := 
sorry

end NUMINAMATH_GPT_count_solutions_l1413_141347


namespace NUMINAMATH_GPT_find_m_plus_n_l1413_141393

noncomputable def overlapping_points (A B: ℝ × ℝ) (C D: ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let M_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let axis_slope := - 1 / k_AB
  let k_CD := (D.2 - C.2) / (D.1 - C.1)
  let M_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  k_CD = axis_slope ∧ (M_CD.2 - M_AB.2) = axis_slope * (M_CD.1 - M_AB.1)

theorem find_m_plus_n : 
  ∃ (m n: ℝ), overlapping_points (0, 2) (4, 0) (7, 3) (m, n) ∧ m + n = 34 / 5 :=
sorry

end NUMINAMATH_GPT_find_m_plus_n_l1413_141393


namespace NUMINAMATH_GPT_expansion_a0_value_l1413_141309

theorem expansion_a0_value :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), (∀ x : ℝ, (x+1)^5 = a_0 + a_1*(x-1) + a_2*(x-1)^2 + a_3*(x-1)^3 + a_4*(x-1)^4 + a_5*(x-1)^5) ∧ a_0 = 32 :=
  sorry

end NUMINAMATH_GPT_expansion_a0_value_l1413_141309


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l1413_141399

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ ((a + b + c) ^ 2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l1413_141399


namespace NUMINAMATH_GPT_largest_possible_average_l1413_141378

noncomputable def ten_test_scores (a b c d e f g h i j : ℤ) : ℤ :=
  a + b + c + d + e + f + g + h + i + j

theorem largest_possible_average
  (a b c d e f g h i j : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 100)
  (h2 : 0 ≤ b ∧ b ≤ 100)
  (h3 : 0 ≤ c ∧ c ≤ 100)
  (h4 : 0 ≤ d ∧ d ≤ 100)
  (h5 : 0 ≤ e ∧ e ≤ 100)
  (h6 : 0 ≤ f ∧ f ≤ 100)
  (h7 : 0 ≤ g ∧ g ≤ 100)
  (h8 : 0 ≤ h ∧ h ≤ 100)
  (h9 : 0 ≤ i ∧ i ≤ 100)
  (h10 : 0 ≤ j ∧ j ≤ 100)
  (h11 : a + b + c + d ≤ 190)
  (h12 : b + c + d + e ≤ 190)
  (h13 : c + d + e + f ≤ 190)
  (h14 : d + e + f + g ≤ 190)
  (h15 : e + f + g + h ≤ 190)
  (h16 : f + g + h + i ≤ 190)
  (h17 : g + h + i + j ≤ 190)
  : ((ten_test_scores a b c d e f g h i j : ℚ) / 10) ≤ 44.33 := sorry

end NUMINAMATH_GPT_largest_possible_average_l1413_141378


namespace NUMINAMATH_GPT_reach_14_from_458_l1413_141321

def double (n : ℕ) : ℕ :=
  n * 2

def erase_last_digit (n : ℕ) : ℕ :=
  n / 10

def can_reach (start target : ℕ) (ops : List (ℕ → ℕ)) : Prop :=
  ∃ seq : List (ℕ → ℕ), seq = ops ∧
    seq.foldl (fun acc f => f acc) start = target

-- The proof problem statement
theorem reach_14_from_458 : can_reach 458 14 [double, erase_last_digit, double, double, erase_last_digit, double, double, erase_last_digit] :=
  sorry

end NUMINAMATH_GPT_reach_14_from_458_l1413_141321


namespace NUMINAMATH_GPT_first_train_travels_more_l1413_141319

-- Define the conditions
def velocity_first_train := 50 -- speed of the first train in km/hr
def velocity_second_train := 40 -- speed of the second train in km/hr
def distance_between_P_and_Q := 900 -- distance between P and Q in km

-- Problem statement
theorem first_train_travels_more :
  ∃ t : ℝ, (velocity_first_train * t + velocity_second_train * t = distance_between_P_and_Q)
          → (velocity_first_train * t - velocity_second_train * t = 100) :=
by sorry

end NUMINAMATH_GPT_first_train_travels_more_l1413_141319


namespace NUMINAMATH_GPT_count_two_digit_integers_l1413_141315

def two_digit_integers_satisfying_condition : Nat :=
  let candidates := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]
  candidates.length

theorem count_two_digit_integers :
  two_digit_integers_satisfying_condition = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_two_digit_integers_l1413_141315


namespace NUMINAMATH_GPT_complex_equation_solution_l1413_141363

theorem complex_equation_solution (x : ℝ) (i : ℂ) (h_imag_unit : i * i = -1) (h_eq : (x + 2 * i) * (x - i) = 6 + 2 * i) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_equation_solution_l1413_141363


namespace NUMINAMATH_GPT_tory_needs_to_sell_more_packs_l1413_141366

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end NUMINAMATH_GPT_tory_needs_to_sell_more_packs_l1413_141366


namespace NUMINAMATH_GPT_set_difference_is_single_element_l1413_141379

-- Define the sets M and N based on the given conditions
def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}
def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

-- State the theorem that we need to prove
theorem set_difference_is_single_element : (N \ M) = {2003} :=
sorry

end NUMINAMATH_GPT_set_difference_is_single_element_l1413_141379


namespace NUMINAMATH_GPT_number_of_total_flowers_l1413_141397

theorem number_of_total_flowers :
  let n_pots := 141
  let flowers_per_pot := 71
  n_pots * flowers_per_pot = 10011 :=
by
  sorry

end NUMINAMATH_GPT_number_of_total_flowers_l1413_141397


namespace NUMINAMATH_GPT_translation_min_point_correct_l1413_141356

-- Define the original equation
def original_eq (x : ℝ) := |x| - 5

-- Define the translation function
def translate_point (p : ℝ × ℝ) (tx ty : ℝ) : ℝ × ℝ := (p.1 + tx, p.2 + ty)

-- Define the minimum point of the original equation
def original_min_point : ℝ × ℝ := (0, original_eq 0)

-- Translate the original minimum point three units right and four units up
def new_min_point := translate_point original_min_point 3 4

-- Prove that the new minimum point is (3, -1)
theorem translation_min_point_correct : new_min_point = (3, -1) :=
by
  sorry

end NUMINAMATH_GPT_translation_min_point_correct_l1413_141356


namespace NUMINAMATH_GPT_intersection_of_lines_l1413_141349

-- Define the first and second lines
def line1 (x : ℚ) : ℚ := 3 * x + 1
def line2 (x : ℚ) : ℚ := -7 * x - 5

-- Statement: Prove that the intersection of the lines given by
-- y = 3x + 1 and y + 5 = -7x is (-3/5, -4/5).

theorem intersection_of_lines :
  ∃ x y : ℚ, y = line1 x ∧ y = line2 x ∧ x = -3 / 5 ∧ y = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1413_141349


namespace NUMINAMATH_GPT_savings_calculation_l1413_141389

theorem savings_calculation (income expenditure savings : ℕ) (ratio_income ratio_expenditure : ℕ)
  (h_ratio : ratio_income = 10) (h_ratio2 : ratio_expenditure = 7) (h_income : income = 10000)
  (h_expenditure : 10 * expenditure = 7 * income) :
  savings = income - expenditure :=
by
  sorry

end NUMINAMATH_GPT_savings_calculation_l1413_141389


namespace NUMINAMATH_GPT_quincy_more_stuffed_animals_l1413_141331

theorem quincy_more_stuffed_animals (thor_sold jake_sold quincy_sold : ℕ) 
  (h1 : jake_sold = thor_sold + 10) 
  (h2 : quincy_sold = 10 * thor_sold) 
  (h3 : quincy_sold = 200) : 
  quincy_sold - jake_sold = 170 :=
by sorry

end NUMINAMATH_GPT_quincy_more_stuffed_animals_l1413_141331


namespace NUMINAMATH_GPT_quad_vertex_transform_l1413_141328

theorem quad_vertex_transform :
  ∀ (x y : ℝ) (h : y = -2 * x^2) (new_x new_y : ℝ) (h_translation : new_x = x + 3 ∧ new_y = y - 2),
  new_y = -2 * (new_x - 3)^2 + 2 :=
by
  intros x y h new_x new_y h_translation
  sorry

end NUMINAMATH_GPT_quad_vertex_transform_l1413_141328


namespace NUMINAMATH_GPT_cos_double_angle_l1413_141343

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1413_141343


namespace NUMINAMATH_GPT_Jungkook_has_the_largest_number_l1413_141320

theorem Jungkook_has_the_largest_number :
  let Yoongi := 4
  let Yuna := 5
  let Jungkook := 6 + 3
  Jungkook > Yoongi ∧ Jungkook > Yuna := by
    sorry

end NUMINAMATH_GPT_Jungkook_has_the_largest_number_l1413_141320


namespace NUMINAMATH_GPT_solution_set_of_gx_lt_0_l1413_141346

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := f_inv (1 - x) - f_inv (1 + x)

theorem solution_set_of_gx_lt_0 : { x : ℝ | g x < 0 } = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_gx_lt_0_l1413_141346


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1413_141313

theorem sum_of_squares_of_roots :
  let a := 5
  let b := -7
  let c := 2
  let x1 := (-b + (b^2 - 4*a*c)^(1/2)) / (2*a)
  let x2 := (-b - (b^2 - 4*a*c)^(1/2)) / (2*a)
  x1^2 + x2^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1413_141313


namespace NUMINAMATH_GPT_problem_statement_l1413_141310

theorem problem_statement :
  75 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -5/31 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1413_141310


namespace NUMINAMATH_GPT_larger_to_smaller_ratio_l1413_141341

theorem larger_to_smaller_ratio (x y : ℝ) (h1 : 0 < y) (h2 : y < x) (h3 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_larger_to_smaller_ratio_l1413_141341


namespace NUMINAMATH_GPT_positive_slope_of_asymptote_l1413_141307

-- Define the conditions
def is_hyperbola (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 5) ^ 2 + (y + 2) ^ 2)) = 3

-- Prove the positive slope of the asymptote of the given hyperbola
theorem positive_slope_of_asymptote :
  (∀ x y : ℝ, is_hyperbola x y) → abs (Real.sqrt 7 / 3) = Real.sqrt 7 / 3 :=
by
  intros h
  -- Proof to be provided (proof steps from the provided solution would be used here usually)
  sorry

end NUMINAMATH_GPT_positive_slope_of_asymptote_l1413_141307


namespace NUMINAMATH_GPT_total_number_of_animals_l1413_141383

theorem total_number_of_animals 
  (rabbits ducks chickens : ℕ)
  (h1 : chickens = 5 * ducks)
  (h2 : ducks = rabbits + 12)
  (h3 : rabbits = 4) : 
  chickens + ducks + rabbits = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l1413_141383


namespace NUMINAMATH_GPT_selection_ways_l1413_141353

def ways_to_select_president_and_secretary (n : Nat) : Nat :=
  n * (n - 1)

theorem selection_ways :
  ways_to_select_president_and_secretary 5 = 20 :=
by
  sorry

end NUMINAMATH_GPT_selection_ways_l1413_141353


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l1413_141382

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l1413_141382


namespace NUMINAMATH_GPT_number_of_sets_B_l1413_141380

def A : Set ℕ := {1, 2, 3}

theorem number_of_sets_B :
  ∃ B : Set ℕ, (A ∪ B = A ∧ 1 ∈ B ∧ (∃ n : ℕ, n = 4)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_sets_B_l1413_141380


namespace NUMINAMATH_GPT_ratio_ac_l1413_141354

variable {a b c d : ℝ}

-- Given the conditions
axiom ratio_ab : a / b = 5 / 4
axiom ratio_cd : c / d = 4 / 3
axiom ratio_db : d / b = 1 / 5

-- The statement to prove
theorem ratio_ac : a / c = 75 / 16 :=
  by sorry

end NUMINAMATH_GPT_ratio_ac_l1413_141354


namespace NUMINAMATH_GPT_selling_price_of_article_l1413_141371

theorem selling_price_of_article (CP : ℝ) (L_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 600) 
  (h2 : L_percent = 50) 
  : SP = 300 := 
by
  sorry

end NUMINAMATH_GPT_selling_price_of_article_l1413_141371


namespace NUMINAMATH_GPT_number_of_elements_in_set_l1413_141337

theorem number_of_elements_in_set 
  (S : ℝ) (n : ℝ) 
  (h_avg : S / n = 6.8) 
  (a : ℝ) (h_a : a = 6) 
  (h_new_avg : (S + 2 * a) / n = 9.2) : 
  n = 5 := 
  sorry

end NUMINAMATH_GPT_number_of_elements_in_set_l1413_141337


namespace NUMINAMATH_GPT_passes_through_point_l1413_141324

theorem passes_through_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end NUMINAMATH_GPT_passes_through_point_l1413_141324


namespace NUMINAMATH_GPT_white_clothing_probability_l1413_141335

theorem white_clothing_probability (total_athletes sample_size k_min k_max : ℕ) 
  (red_upper_bound white_upper_bound yellow_upper_bound sampled_start_interval : ℕ)
  (h_total : total_athletes = 600)
  (h_sample : sample_size = 50)
  (h_intervals : total_athletes / sample_size = 12)
  (h_group_start : sampled_start_interval = 4)
  (h_red_upper : red_upper_bound = 311)
  (h_white_upper : white_upper_bound = 496)
  (h_yellow_upper : yellow_upper_bound = 600)
  (h_k_min : k_min = 26)   -- Calculated from 312 <= 12k + 4
  (h_k_max : k_max = 41)  -- Calculated from 12k + 4 <= 496
  : (k_max - k_min + 1) / sample_size = 8 / 25 := 
by
  sorry

end NUMINAMATH_GPT_white_clothing_probability_l1413_141335


namespace NUMINAMATH_GPT_determine_a_l1413_141381

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem determine_a (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_a_l1413_141381


namespace NUMINAMATH_GPT_final_position_correct_l1413_141333

structure Position :=
(base : ℝ × ℝ)
(stem : ℝ × ℝ)

def initial_position : Position :=
{ base := (0, -1),
  stem := (1, 0) }

def reflect_x (p : Position) : Position :=
{ base := (p.base.1, -p.base.2),
  stem := (p.stem.1, -p.stem.2) }

def rotate_90_ccw (p : Position) : Position :=
{ base := (-p.base.2, p.base.1),
  stem := (-p.stem.2, p.stem.1) }

def half_turn (p : Position) : Position :=
{ base := (-p.base.1, -p.base.2),
  stem := (-p.stem.1, -p.stem.2) }

def reflect_y (p : Position) : Position :=
{ base := (-p.base.1, p.base.2),
  stem := (-p.stem.1, p.stem.2) }

def final_position : Position :=
reflect_y (half_turn (rotate_90_ccw (reflect_x initial_position)))

theorem final_position_correct : final_position = { base := (1, 0), stem := (0, 1) } :=
sorry

end NUMINAMATH_GPT_final_position_correct_l1413_141333


namespace NUMINAMATH_GPT_time_between_last_two_rings_l1413_141352

variable (n : ℕ) (x y : ℝ)

noncomputable def timeBetweenLastTwoRings : ℝ :=
  x + (n - 3) * y

theorem time_between_last_two_rings :
  timeBetweenLastTwoRings n x y = x + (n - 3) * y :=
by
  sorry

end NUMINAMATH_GPT_time_between_last_two_rings_l1413_141352


namespace NUMINAMATH_GPT_surface_area_of_cross_shape_with_five_unit_cubes_l1413_141364

noncomputable def unit_cube_surface_area : ℕ := 6
noncomputable def num_cubes : ℕ := 5
noncomputable def total_surface_area_iso_cubes : ℕ := num_cubes * unit_cube_surface_area
noncomputable def central_cube_exposed_faces : ℕ := 2
noncomputable def surrounding_cubes_exposed_faces : ℕ := 5
noncomputable def surrounding_cubes_count : ℕ := 4
noncomputable def cross_shape_surface_area : ℕ := 
  central_cube_exposed_faces + (surrounding_cubes_count * surrounding_cubes_exposed_faces)

theorem surface_area_of_cross_shape_with_five_unit_cubes : cross_shape_surface_area = 22 := 
by sorry

end NUMINAMATH_GPT_surface_area_of_cross_shape_with_five_unit_cubes_l1413_141364


namespace NUMINAMATH_GPT_part_1_conditions_part_2_min_value_l1413_141306

theorem part_1_conditions
  (a b x : ℝ)
  (h1: 2 * a * x^2 - 8 * x - 3 * a^2 < 0)
  (h2: ∀ x, -1 < x -> x < b)
  : a = 2 ∧ b = 3 := sorry

theorem part_2_min_value
  (a b x y : ℝ)
  (h1: x > 0)
  (h2: y > 0)
  (h3: a = 2)
  (h4: b = 3)
  (h5: (a / x) + (b / y) = 1)
  : ∃ min_val : ℝ, min_val = 3 * x + 2 * y ∧ min_val = 24 := sorry

end NUMINAMATH_GPT_part_1_conditions_part_2_min_value_l1413_141306


namespace NUMINAMATH_GPT_journey_speed_l1413_141367

theorem journey_speed (v : ℝ) 
  (h1 : 3 * v + 60 * 2 = 240)
  (h2 : 3 + 2 = 5) :
  v = 40 :=
by
  sorry

end NUMINAMATH_GPT_journey_speed_l1413_141367


namespace NUMINAMATH_GPT_supplement_of_complement_of_35_degree_angle_l1413_141338

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end NUMINAMATH_GPT_supplement_of_complement_of_35_degree_angle_l1413_141338


namespace NUMINAMATH_GPT_range_of_a_l1413_141317

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (e^x - a)^2 + x^2 - 2 * a * x + a^2 ≤ 1 / 2) ↔ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1413_141317


namespace NUMINAMATH_GPT_chrysler_building_floors_l1413_141305

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end NUMINAMATH_GPT_chrysler_building_floors_l1413_141305


namespace NUMINAMATH_GPT_quadratic_equation_m_condition_l1413_141387

theorem quadratic_equation_m_condition (m : ℝ) :
  (m + 1 ≠ 0) ↔ (m ≠ -1) :=
by sorry

end NUMINAMATH_GPT_quadratic_equation_m_condition_l1413_141387


namespace NUMINAMATH_GPT_day_of_week_proof_l1413_141300

/-- 
January 1, 1978, is a Sunday in the Gregorian calendar.
What day of the week is January 1, 2000, in the Gregorian calendar?
-/
def day_of_week_2000 := "Saturday"

theorem day_of_week_proof :
  let initial_year := 1978
  let target_year := 2000
  let initial_weekday := "Sunday"
  let years_between := target_year - initial_year -- 22 years
  let normal_days := years_between * 365 -- Normal days in these years
  let leap_years := 5 -- Number of leap years in the range
  let total_days := normal_days + leap_years -- Total days considering leap years
  let remainder_days := total_days % 7 -- days modulo 7
  initial_weekday = "Sunday" → remainder_days = 6 → 
  day_of_week_2000 = "Saturday" :=
by
  sorry

end NUMINAMATH_GPT_day_of_week_proof_l1413_141300


namespace NUMINAMATH_GPT_possible_values_of_n_l1413_141336

open Nat

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c (n : ℕ) : ℕ := a (b n)

noncomputable def T (n : ℕ) : ℕ := (Finset.range n).sum (λ i => c (i + 1))

theorem possible_values_of_n (n : ℕ) :
  T n < 2021 → n = 8 ∨ n = 9 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_n_l1413_141336


namespace NUMINAMATH_GPT_jeff_bought_from_chad_l1413_141318

/-
  Eric has 4 ninja throwing stars.
  Chad has twice as many ninja throwing stars as Eric.
  Jeff now has 6 ninja throwing stars.
  Together, they have 16 ninja throwing stars.
  How many ninja throwing stars did Jeff buy from Chad?
-/

def eric_stars : ℕ := 4
def chad_stars : ℕ := 2 * eric_stars
def jeff_stars : ℕ := 6
def total_stars : ℕ := 16

theorem jeff_bought_from_chad (bought : ℕ) :
  chad_stars - bought + jeff_stars + eric_stars = total_stars → bought = 2 :=
by
  sorry

end NUMINAMATH_GPT_jeff_bought_from_chad_l1413_141318


namespace NUMINAMATH_GPT_weekly_earnings_l1413_141342

-- Definition of the conditions
def hourly_rate : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

-- Theorem that conforms to the problem statement
theorem weekly_earnings : hourly_rate * hours_per_day * days_per_week = 640 := by
  sorry

end NUMINAMATH_GPT_weekly_earnings_l1413_141342


namespace NUMINAMATH_GPT_minimum_radius_third_sphere_l1413_141330

-- Definitions for the problem
def height_cone := 4
def base_radius_cone := 3
def cos_alpha := 4 / 5
def radius_identical_sphere := 4 / 3
def cos_beta := 1 -- since beta is maximized

-- Define the required minimum radius for the third sphere based on the given conditions
theorem minimum_radius_third_sphere :
  ∃ x : ℝ, x = 27 / 35 ∧
    (height_cone = 4) ∧ 
    (base_radius_cone = 3) ∧ 
    (cos_alpha = 4 / 5) ∧ 
    (radius_identical_sphere = 4 / 3) ∧ 
    (cos_beta = 1) :=
sorry

end NUMINAMATH_GPT_minimum_radius_third_sphere_l1413_141330


namespace NUMINAMATH_GPT_polynomial_zero_iff_divisibility_l1413_141374

theorem polynomial_zero_iff_divisibility (P : Polynomial ℤ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, P.eval (2^n) = n * k) ↔ P = 0 :=
by sorry

end NUMINAMATH_GPT_polynomial_zero_iff_divisibility_l1413_141374


namespace NUMINAMATH_GPT_increase_to_restore_l1413_141362

noncomputable def percentage_increase_to_restore (P : ℝ) : ℝ :=
  let reduced_price := 0.9 * P
  let restore_factor := P / reduced_price
  (restore_factor - 1) * 100

theorem increase_to_restore :
  percentage_increase_to_restore 100 = 100 / 9 :=
by
  sorry

end NUMINAMATH_GPT_increase_to_restore_l1413_141362


namespace NUMINAMATH_GPT_gcd_two_powers_l1413_141368

def m : ℕ := 2 ^ 1998 - 1
def n : ℕ := 2 ^ 1989 - 1

theorem gcd_two_powers :
  Nat.gcd (2 ^ 1998 - 1) (2 ^ 1989 - 1) = 511 := 
sorry

end NUMINAMATH_GPT_gcd_two_powers_l1413_141368


namespace NUMINAMATH_GPT_problem1_l1413_141332

theorem problem1 :
  (15 * (-3 / 4) + (-15) * (3 / 2) + 15 / 4) = -30 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1413_141332


namespace NUMINAMATH_GPT_solve_inequality_l1413_141348

theorem solve_inequality (x : ℝ) :
  (4 ≤ x^2 - 3 * x - 6 ∧ x^2 - 3 * x - 6 ≤ 2 * x + 8) ↔ (5 ≤ x ∧ x ≤ 7 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1413_141348


namespace NUMINAMATH_GPT_prove_values_of_a_and_b_prove_range_of_k_l1413_141327

variable {f : ℝ → ℝ}

-- (1) Prove values of a and b
theorem prove_values_of_a_and_b (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  (∀ x, f x = 2 * x - 1) := by
sorry

-- (2) Prove range of k
theorem prove_range_of_k (h_fx_2x_minus_1 : ∀ x : ℝ, f x = 2 * x - 1) :
  (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1 / 3 := by
sorry

end NUMINAMATH_GPT_prove_values_of_a_and_b_prove_range_of_k_l1413_141327


namespace NUMINAMATH_GPT_total_cost_of_tickets_l1413_141322

def number_of_adults := 2
def number_of_children := 3
def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6

theorem total_cost_of_tickets :
  let total_cost := number_of_adults * cost_of_adult_ticket + number_of_children * cost_of_child_ticket
  total_cost = 77 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_tickets_l1413_141322


namespace NUMINAMATH_GPT_unique_three_digit_numbers_count_l1413_141351

theorem unique_three_digit_numbers_count :
  ∃ l : List Nat, (∀ n ∈ l, 100 ≤ n ∧ n < 1000) ∧ 
    l = [230, 203, 302, 320] ∧ l.length = 4 := 
by
  sorry

end NUMINAMATH_GPT_unique_three_digit_numbers_count_l1413_141351


namespace NUMINAMATH_GPT_sum_midpoint_x_coords_l1413_141358

theorem sum_midpoint_x_coords (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
    (a + (a - 3)) / 2 + (a + c) / 2 + ((a - 3) + c) / 2 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_sum_midpoint_x_coords_l1413_141358


namespace NUMINAMATH_GPT_probability_of_graduate_degree_l1413_141390

-- Define the conditions as Lean statements
variable (k m : ℕ)
variable (G := 1 * k) 
variable (C := 2 * m) 
variable (N1 := 8 * k) -- from the ratio G:N = 1:8
variable (N2 := 3 * m) -- from the ratio C:N = 2:3

-- Least common multiple (LCM) of 8 and 3 is 24
-- Therefore, determine specific values for G, C, and N
-- Given these updates from solution steps we set:
def G_scaled : ℕ := 3
def C_scaled : ℕ := 16
def N_scaled : ℕ := 24

-- Total number of college graduates
def total_college_graduates : ℕ := G_scaled + C_scaled

-- Probability q of picking a college graduate with a graduate degree
def q : ℚ := G_scaled / total_college_graduates

-- Lean proof statement for equivalence
theorem probability_of_graduate_degree : 
  q = 3 / 19 := by
sorry

end NUMINAMATH_GPT_probability_of_graduate_degree_l1413_141390


namespace NUMINAMATH_GPT_smallest_mu_ineq_l1413_141394

theorem smallest_mu_ineq (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    a^2 + b^2 + c^2 + d^2 + 2 * a * d ≥ 2 * (a * b + b * c + c * d) := by {
    sorry
}

end NUMINAMATH_GPT_smallest_mu_ineq_l1413_141394


namespace NUMINAMATH_GPT_a_2016_value_l1413_141345

def S (n : ℕ) : ℕ := n^2 - 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_2016_value : a 2016 = 4031 := by
  sorry

end NUMINAMATH_GPT_a_2016_value_l1413_141345
