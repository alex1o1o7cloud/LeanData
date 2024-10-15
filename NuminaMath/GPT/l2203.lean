import Mathlib

namespace NUMINAMATH_GPT_regular_polygon_sides_l2203_220352

theorem regular_polygon_sides (n : ℕ) (h : n > 0) (h_exterior_angle : 360 / n = 10) : n = 36 :=
by sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2203_220352


namespace NUMINAMATH_GPT_gcd_of_180_and_450_l2203_220370

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end NUMINAMATH_GPT_gcd_of_180_and_450_l2203_220370


namespace NUMINAMATH_GPT_flask_forces_l2203_220322

theorem flask_forces (r : ℝ) (ρ g h_A h_B h_C V : ℝ) (A : ℝ) (FA FB FC : ℝ) (h1 : r = 2)
  (h2 : A = π * r^2)
  (h3 : V = A * h_A ∧ V = A * h_B ∧ V = A * h_C)
  (h4 : FC = ρ * g * h_C * A)
  (h5 : FA = ρ * g * h_A * A)
  (h6 : FB = ρ * g * h_B * A)
  (h7 : h_C > h_A ∧ h_A > h_B) : FC > FA ∧ FA > FB := 
sorry

end NUMINAMATH_GPT_flask_forces_l2203_220322


namespace NUMINAMATH_GPT_abc_zero_l2203_220399

theorem abc_zero {a b c : ℝ} 
(h1 : (a + b) * (b + c) * (c + a) = a * b * c)
(h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
a * b * c = 0 := 
by sorry

end NUMINAMATH_GPT_abc_zero_l2203_220399


namespace NUMINAMATH_GPT_geom_series_first_term_l2203_220307

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_geom_series_first_term_l2203_220307


namespace NUMINAMATH_GPT_sequence_b_n_l2203_220302

theorem sequence_b_n (b : ℕ → ℝ) (h₁ : b 1 = 2) (h₂ : ∀ n, (b (n + 1))^3 = 64 * (b n)^3) : 
    b 50 = 2 * 4^49 :=
sorry

end NUMINAMATH_GPT_sequence_b_n_l2203_220302


namespace NUMINAMATH_GPT_initial_amount_of_liquid_A_l2203_220356

theorem initial_amount_of_liquid_A (A B : ℝ) (initial_ratio : A = 4 * B) (removed_mixture : ℝ) (new_ratio : (A - (4/5) * removed_mixture) = (2 / 3) * ((B - (1/5) * removed_mixture) + removed_mixture)) :
  A = 16 := 
  sorry

end NUMINAMATH_GPT_initial_amount_of_liquid_A_l2203_220356


namespace NUMINAMATH_GPT_trapezoid_diagonal_intersection_l2203_220393

theorem trapezoid_diagonal_intersection (PQ RS PR : ℝ) (h1 : PQ = 3 * RS) (h2 : PR = 15) :
  ∃ RT : ℝ, RT = 15 / 4 :=
by
  have RT := 15 / 4
  use RT
  sorry

end NUMINAMATH_GPT_trapezoid_diagonal_intersection_l2203_220393


namespace NUMINAMATH_GPT_inequality_proof_l2203_220376

noncomputable def a : Real := (1 / 3) ^ Real.pi
noncomputable def b : Real := (1 / 3) ^ (1 / 2 : Real)
noncomputable def c : Real := Real.pi ^ (1 / 2 : Real)

theorem inequality_proof : a < b ∧ b < c :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_inequality_proof_l2203_220376


namespace NUMINAMATH_GPT_find_a_l2203_220340

theorem find_a (a : ℝ) : 
  (a + 3)^2 = (a + 1)^2 + (a + 2)^2 → a = 2 := 
by
  intro h
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_find_a_l2203_220340


namespace NUMINAMATH_GPT_people_got_on_at_third_stop_l2203_220369

theorem people_got_on_at_third_stop
  (initial : ℕ)
  (got_off_first : ℕ)
  (got_off_second : ℕ)
  (got_on_second : ℕ)
  (got_off_third : ℕ)
  (people_after_third : ℕ) :
  initial = 50 →
  got_off_first = 15 →
  got_off_second = 8 →
  got_on_second = 2 →
  got_off_third = 4 →
  people_after_third = 28 →
  ∃ got_on_third : ℕ, got_on_third = 3 :=
by
  sorry

end NUMINAMATH_GPT_people_got_on_at_third_stop_l2203_220369


namespace NUMINAMATH_GPT_batsman_average_is_18_l2203_220313
noncomputable def average_after_18_innings (score_18th: ℕ) (average_17th: ℕ) (innings: ℕ) : ℕ :=
  let total_runs_17 := average_17th * 17
  let total_runs_18 := total_runs_17 + score_18th
  total_runs_18 / innings

theorem batsman_average_is_18 {score_18th: ℕ} {average_17th: ℕ} {expected_average: ℕ} :
  score_18th = 1 → average_17th = 19 → expected_average = 18 →
  average_after_18_innings score_18th average_17th 18 = expected_average := by
  sorry

end NUMINAMATH_GPT_batsman_average_is_18_l2203_220313


namespace NUMINAMATH_GPT_oldest_child_age_l2203_220329

def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := 
  a + (n - 1) * d

theorem oldest_child_age (a : ℕ) (d : ℕ) (n : ℕ) 
  (average : (arithmeticProgression a d 1 + arithmeticProgression a d 2 + arithmeticProgression a d 3 + arithmeticProgression a d 4 + arithmeticProgression a d 5) / 5 = 10)
  (distinct : ∀ i j, i ≠ j → arithmeticProgression a d i ≠ arithmeticProgression a d j)
  (constant_difference : d = 3) :
  arithmeticProgression a d 5 = 16 :=
by
  sorry

end NUMINAMATH_GPT_oldest_child_age_l2203_220329


namespace NUMINAMATH_GPT_amoeba_doubling_time_l2203_220324

theorem amoeba_doubling_time (H1 : ∀ t : ℕ, t = 60 → 2^(t / 3) = 2^20) :
  ∀ t : ℕ, 2 * 2^(t / 3) = 2^20 → t = 57 :=
by
  intro t
  intro H2
  sorry

end NUMINAMATH_GPT_amoeba_doubling_time_l2203_220324


namespace NUMINAMATH_GPT_four_digit_numbers_using_0_and_9_l2203_220353

theorem four_digit_numbers_using_0_and_9 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d, d ∈ Nat.digits 10 n → (d = 0 ∨ d = 9)} = {9000, 9009, 9090, 9099, 9900, 9909, 9990, 9999} :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_using_0_and_9_l2203_220353


namespace NUMINAMATH_GPT_number_description_l2203_220391

theorem number_description :
  4 * 10000 + 3 * 1000 + 7 * 100 + 5 * 10 + 2 + 8 / 10 + 4 / 100 = 43752.84 :=
by
  sorry

end NUMINAMATH_GPT_number_description_l2203_220391


namespace NUMINAMATH_GPT_card_draw_probability_l2203_220342

theorem card_draw_probability :
  (13 / 52) * (13 / 51) * (13 / 50) = 2197 / 132600 :=
by
  sorry

end NUMINAMATH_GPT_card_draw_probability_l2203_220342


namespace NUMINAMATH_GPT_vehicles_count_l2203_220314

theorem vehicles_count (T : ℕ) : 
    2 * T + 3 * (2 * T) + (T / 2) + T = 180 → 
    T = 19 ∧ 2 * T = 38 ∧ 3 * (2 * T) = 114 ∧ (T / 2) = 9 := 
by 
    intros h
    sorry

end NUMINAMATH_GPT_vehicles_count_l2203_220314


namespace NUMINAMATH_GPT_series_proof_l2203_220395

noncomputable def series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / (b ^ (n + 1))

noncomputable def transformed_series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / ((a + 2 * b) ^ (n + 1))

theorem series_proof (a b : ℝ)
  (h1 : series_sum a b = 7)
  (h2 : a = 7 * (b - 1)) :
  transformed_series_sum a b = 7 * (b - 1) / (9 * b - 8) :=
by sorry

end NUMINAMATH_GPT_series_proof_l2203_220395


namespace NUMINAMATH_GPT_pure_imaginary_real_zero_l2203_220367

theorem pure_imaginary_real_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) (h : a * i = 0 + a * i) : a = 0 := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_real_zero_l2203_220367


namespace NUMINAMATH_GPT_evaluate_f_l2203_220355

def f (x : ℝ) : ℝ := x^2 + 4*x - 3

theorem evaluate_f (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_evaluate_f_l2203_220355


namespace NUMINAMATH_GPT_cookie_sheet_perimeter_l2203_220306

def width : ℕ := 10
def length : ℕ := 2

def perimeter (w l : ℕ) : ℕ := 2 * w + 2 * l

theorem cookie_sheet_perimeter : 
  perimeter width length = 24 := by
  sorry

end NUMINAMATH_GPT_cookie_sheet_perimeter_l2203_220306


namespace NUMINAMATH_GPT_probability_of_red_ball_l2203_220385

noncomputable def total_balls : Nat := 4 + 2
noncomputable def red_balls : Nat := 2

theorem probability_of_red_ball :
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_of_red_ball_l2203_220385


namespace NUMINAMATH_GPT_five_pow_sum_of_squares_l2203_220349

theorem five_pow_sum_of_squares (n : ℕ) : ∃ a b : ℕ, 5^n = a^2 + b^2 := 
sorry

end NUMINAMATH_GPT_five_pow_sum_of_squares_l2203_220349


namespace NUMINAMATH_GPT_ratio_of_perimeters_l2203_220388

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l2203_220388


namespace NUMINAMATH_GPT_certain_number_is_166_l2203_220371

theorem certain_number_is_166 :
  ∃ x : ℕ, x - 78 =  (4 - 30) + 114 ∧ x = 166 := by
  sorry

end NUMINAMATH_GPT_certain_number_is_166_l2203_220371


namespace NUMINAMATH_GPT_ratio_of_socks_l2203_220337

-- Conditions:
variable (B : ℕ) (W : ℕ) (L : ℕ)
-- B = number of black socks
-- W = initial number of white socks
-- L = number of white socks lost

-- Setting given conditions:
axiom hB : B = 6
axiom hL : L = W / 2
axiom hCond : W / 2 = B + 6

-- Prove the ratio of white socks to black socks is 4:1
theorem ratio_of_socks : B = 6 → W / 2 = B + 6 → (W / 2) + (W / 2) = 24 → (B : ℚ) / (W : ℚ) = 1 / 4 :=
by intros hB hCond hW
   sorry

end NUMINAMATH_GPT_ratio_of_socks_l2203_220337


namespace NUMINAMATH_GPT_solve_m_l2203_220312

theorem solve_m (m : ℝ) : 
  (m - 3) * x^2 - 3 * x + m^2 = 9 → m^2 - 9 = 0 → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_m_l2203_220312


namespace NUMINAMATH_GPT_area_of_parallelogram_l2203_220344

theorem area_of_parallelogram (b h : ℕ) (hb : b = 60) (hh : h = 16) : b * h = 960 := by
  -- Here goes the proof
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l2203_220344


namespace NUMINAMATH_GPT_solve_for_x_l2203_220390

theorem solve_for_x (x : ℤ) (h_eq : (7 * x - 5) / (x - 2) = 2 / (x - 2)) (h_cond : x ≠ 2) : x = 1 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2203_220390


namespace NUMINAMATH_GPT_print_shop_x_charge_l2203_220304

theorem print_shop_x_charge :
  ∃ (x : ℝ), 60 * x + 90 = 60 * 2.75 ∧ x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_print_shop_x_charge_l2203_220304


namespace NUMINAMATH_GPT_spike_crickets_hunted_morning_l2203_220346

def crickets_hunted_in_morning (C : ℕ) (total_daily_crickets : ℕ) : Prop :=
  4 * C = total_daily_crickets

theorem spike_crickets_hunted_morning (C : ℕ) (total_daily_crickets : ℕ) :
  total_daily_crickets = 20 → crickets_hunted_in_morning C total_daily_crickets → C = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_spike_crickets_hunted_morning_l2203_220346


namespace NUMINAMATH_GPT_lattice_points_in_region_l2203_220392

theorem lattice_points_in_region :
  ∃ n : ℕ, n = 12 ∧ 
  ( ∀ x y : ℤ, (y = x ∨ y = -x ∨ y = -x^2 + 4) → n = 12) :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_in_region_l2203_220392


namespace NUMINAMATH_GPT_ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l2203_220365

-- Define the ink length of a figure
def ink_length (n : ℕ) : ℕ := 5 * n

-- Part (a): Determine the ink length of Figure 4.
theorem ink_length_figure_4 : ink_length 4 = 20 := by
  sorry

-- Part (b): Determine the difference between the ink length of Figure 9 and the ink length of Figure 8.
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 5 := by
  sorry

-- Part (c): Determine the ink length of Figure 100.
theorem ink_length_figure_100 : ink_length 100 = 500 := by
  sorry

end NUMINAMATH_GPT_ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l2203_220365


namespace NUMINAMATH_GPT_ground_beef_total_cost_l2203_220317

-- Define the conditions
def price_per_kg : ℝ := 5.00
def quantity_in_kg : ℝ := 12

-- The total cost calculation
def total_cost (price_per_kg quantity_in_kg : ℝ) : ℝ := price_per_kg * quantity_in_kg

-- Theorem statement
theorem ground_beef_total_cost :
  total_cost price_per_kg quantity_in_kg = 60.00 :=
sorry

end NUMINAMATH_GPT_ground_beef_total_cost_l2203_220317


namespace NUMINAMATH_GPT_fraction_lost_l2203_220361

-- Definitions of the given conditions
def initial_pencils : ℕ := 30
def lost_pencils_initially : ℕ := 6
def current_pencils : ℕ := 16

-- Statement of the proof problem
theorem fraction_lost (initial_pencils lost_pencils_initially current_pencils : ℕ) :
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  (lost_remaining_pencils : ℚ) / remaining_pencils = 1 / 3 :=
by
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  sorry

end NUMINAMATH_GPT_fraction_lost_l2203_220361


namespace NUMINAMATH_GPT_lucy_total_journey_l2203_220398

-- Define the length of Lucy's journey
def lucy_journey (x : ℝ) : Prop :=
  (1 / 4) * x + 25 + (1 / 6) * x = x

-- State the theorem
theorem lucy_total_journey : ∃ x : ℝ, lucy_journey x ∧ x = 300 / 7 := by
  sorry

end NUMINAMATH_GPT_lucy_total_journey_l2203_220398


namespace NUMINAMATH_GPT_find_integer_triples_l2203_220308

theorem find_integer_triples (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ 
  (x = 668 ∧ y = 668 ∧ z = 667) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 667 ∧ y = 668 ∧ z = 668) :=
by sorry

end NUMINAMATH_GPT_find_integer_triples_l2203_220308


namespace NUMINAMATH_GPT_more_apples_than_pears_l2203_220377

-- Definitions based on conditions
def total_fruits : ℕ := 85
def apples : ℕ := 48

-- Statement to prove
theorem more_apples_than_pears : (apples - (total_fruits - apples)) = 11 := by
  -- proof steps
  sorry

end NUMINAMATH_GPT_more_apples_than_pears_l2203_220377


namespace NUMINAMATH_GPT_hawksbill_to_green_turtle_ratio_l2203_220374

theorem hawksbill_to_green_turtle_ratio (total_turtles : ℕ) (green_turtles : ℕ) (hawksbill_turtles : ℕ) (h1 : green_turtles = 800) (h2 : total_turtles = 3200) (h3 : hawksbill_turtles = total_turtles - green_turtles) :
  hawksbill_turtles / green_turtles = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_hawksbill_to_green_turtle_ratio_l2203_220374


namespace NUMINAMATH_GPT_largest_n_l2203_220338

theorem largest_n (x y z n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12 → n ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_l2203_220338


namespace NUMINAMATH_GPT_value_of_Z_4_3_l2203_220309

def Z (a b : ℤ) : ℤ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem value_of_Z_4_3 : Z 4 3 = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_Z_4_3_l2203_220309


namespace NUMINAMATH_GPT_probability_personA_not_personB_l2203_220350

theorem probability_personA_not_personB :
  let n := Nat.choose 5 3
  let m := Nat.choose 1 1 * Nat.choose 3 2
  (m / n : ℚ) = 3 / 10 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_personA_not_personB_l2203_220350


namespace NUMINAMATH_GPT_exists_integer_n_tangent_l2203_220394
open Real

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * (π / 180)

theorem exists_integer_n_tangent :
  ∃ (n : ℤ), -90 < (n : ℝ) ∧ (n : ℝ) < 90 ∧ tan (degree_to_radian (n : ℝ)) = tan (degree_to_radian 345) ∧ n = -15 :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_n_tangent_l2203_220394


namespace NUMINAMATH_GPT_find_five_digit_number_l2203_220300

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end NUMINAMATH_GPT_find_five_digit_number_l2203_220300


namespace NUMINAMATH_GPT_a_star_b_value_l2203_220343

theorem a_star_b_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) (h3 : b = 8) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 3 / 8 := by
sorry

end NUMINAMATH_GPT_a_star_b_value_l2203_220343


namespace NUMINAMATH_GPT_original_plan_months_l2203_220316

theorem original_plan_months (x : ℝ) (h : 1 / (x - 6) = 1.4 * (1 / x)) : x = 21 :=
by
  sorry

end NUMINAMATH_GPT_original_plan_months_l2203_220316


namespace NUMINAMATH_GPT_soccer_club_girls_l2203_220328

theorem soccer_club_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : (1 / 3 : ℚ) * G + B = 18) : 
  G = 18 := 
  by sorry

end NUMINAMATH_GPT_soccer_club_girls_l2203_220328


namespace NUMINAMATH_GPT_last_part_length_l2203_220379

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end NUMINAMATH_GPT_last_part_length_l2203_220379


namespace NUMINAMATH_GPT_problem_solution_l2203_220378

theorem problem_solution (n : ℤ) : 
  (1 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4) → (n = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_solution_l2203_220378


namespace NUMINAMATH_GPT_star_value_example_l2203_220318

def star (a b c : ℤ) : ℤ := (a + b + c) ^ 2

theorem star_value_example : star 3 (-5) 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_star_value_example_l2203_220318


namespace NUMINAMATH_GPT_projected_revenue_increase_is_20_percent_l2203_220320

noncomputable def projected_percentage_increase_of_revenue (R : ℝ) (actual_revenue : ℝ) (projected_revenue : ℝ) : ℝ :=
  (projected_revenue / R - 1) * 100

theorem projected_revenue_increase_is_20_percent (R : ℝ) (actual_revenue : ℝ) :
  actual_revenue = R * 0.75 →
  actual_revenue = (R * (1 + 20 / 100)) * 0.625 →
  projected_percentage_increase_of_revenue R ((R * (1 + 20 / 100))) = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_projected_revenue_increase_is_20_percent_l2203_220320


namespace NUMINAMATH_GPT_daisies_given_l2203_220310

theorem daisies_given (S : ℕ) (h : (5 + S) / 2 = 7) : S = 9 := by
  sorry

end NUMINAMATH_GPT_daisies_given_l2203_220310


namespace NUMINAMATH_GPT_complete_sets_characterization_l2203_220372

-- Definition of a complete set
def complete_set (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, (a + b ∈ A) → (a * b ∈ A)

-- Theorem stating that the complete sets of natural numbers are exactly
-- {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, ℕ.
theorem complete_sets_characterization :
  ∀ (A : Set ℕ), complete_set A ↔ (A = {1} ∨ A = {1, 2} ∨ A = {1, 2, 3} ∨ A = {1, 2, 3, 4} ∨ A = Set.univ) :=
sorry

end NUMINAMATH_GPT_complete_sets_characterization_l2203_220372


namespace NUMINAMATH_GPT_find_numbers_l2203_220360

theorem find_numbers :
  ∃ (x y z : ℕ), x = y + 75 ∧ 
                 (x * y = z + 1000) ∧
                 (z = 227 * y + 113) ∧
                 (x = 234) ∧ 
                 (y = 159) := by
  sorry

end NUMINAMATH_GPT_find_numbers_l2203_220360


namespace NUMINAMATH_GPT_min_value_of_abs_diff_l2203_220327
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem min_value_of_abs_diff (x1 x2 x : ℝ) (h1 : f x1 ≤ f x) (h2: f x ≤ f x2) : |x1 - x2| = π := by
  sorry

end NUMINAMATH_GPT_min_value_of_abs_diff_l2203_220327


namespace NUMINAMATH_GPT_integral_cos8_0_2pi_l2203_220396

noncomputable def definite_integral_cos8 (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.cos (x / 4)) ^ 8

theorem integral_cos8_0_2pi :
  definite_integral_cos8 0 (2 * Real.pi) = (35 * Real.pi) / 64 :=
by
  sorry

end NUMINAMATH_GPT_integral_cos8_0_2pi_l2203_220396


namespace NUMINAMATH_GPT_find_n_that_makes_vectors_collinear_l2203_220334

theorem find_n_that_makes_vectors_collinear (n : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, n)) (h_collinear : ∃ k : ℝ, 2 • a - b = k • b) : n = 9 :=
sorry

end NUMINAMATH_GPT_find_n_that_makes_vectors_collinear_l2203_220334


namespace NUMINAMATH_GPT_rectangle_and_square_problems_l2203_220326

theorem rectangle_and_square_problems :
  ∃ (length width : ℝ), 
    (length / width = 2) ∧ 
    (length * width = 50) ∧ 
    (length = 10) ∧
    (width = 5) ∧
    ∃ (side_length : ℝ), 
      (side_length ^ 2 = 50) ∧ 
      (side_length - width = 5 * (Real.sqrt 2 - 1)) := 
by
  sorry

end NUMINAMATH_GPT_rectangle_and_square_problems_l2203_220326


namespace NUMINAMATH_GPT_find_number_l2203_220387

theorem find_number (x : ℕ) (h : (9 * x) / 3 = 27) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2203_220387


namespace NUMINAMATH_GPT_min_value_x1_x2_l2203_220389

theorem min_value_x1_x2 (a x_1 x_2 : ℝ) (h_a_pos : 0 < a) (h_sol_set : x_1 + x_2 = 4 * a) (h_prod_set : x_1 * x_2 = 3 * a^2) : 
  x_1 + x_2 + a / (x_1 * x_2) = 4 * a + 1 / (3 * a) :=
sorry

end NUMINAMATH_GPT_min_value_x1_x2_l2203_220389


namespace NUMINAMATH_GPT_apple_juice_production_l2203_220347

noncomputable def apple_usage 
  (total_apples : ℝ) 
  (mixed_percentage : ℝ) 
  (juice_percentage : ℝ) 
  (sold_fresh_percentage : ℝ) : ℝ := 
  let mixed_apples := total_apples * mixed_percentage / 100
  let remainder_apples := total_apples - mixed_apples
  let juice_apples := remainder_apples * juice_percentage / 100
  juice_apples

theorem apple_juice_production :
  apple_usage 6 20 60 40 = 2.9 := 
by
  sorry

end NUMINAMATH_GPT_apple_juice_production_l2203_220347


namespace NUMINAMATH_GPT_exists_x_eq_28_l2203_220315

theorem exists_x_eq_28 : ∃ x : Int, 45 - (x - (37 - (15 - 16))) = 55 ↔ x = 28 := 
by
  sorry

end NUMINAMATH_GPT_exists_x_eq_28_l2203_220315


namespace NUMINAMATH_GPT_seashell_count_l2203_220321

theorem seashell_count (Sam Mary Lucy : Nat) (h1 : Sam = 18) (h2 : Mary = 47) (h3 : Lucy = 32) : 
  Sam + Mary + Lucy = 97 :=
by 
  sorry

end NUMINAMATH_GPT_seashell_count_l2203_220321


namespace NUMINAMATH_GPT_characters_per_day_l2203_220359

-- Definitions based on conditions
def chars_total_older : ℕ := 8000
def chars_total_younger : ℕ := 6000
def chars_per_day_diff : ℕ := 100

-- Define the main theorem
theorem characters_per_day (x : ℕ) :
  chars_total_older / x = chars_total_younger / (x - chars_per_day_diff) := 
sorry

end NUMINAMATH_GPT_characters_per_day_l2203_220359


namespace NUMINAMATH_GPT_apex_angle_of_quadrilateral_pyramid_l2203_220331

theorem apex_angle_of_quadrilateral_pyramid :
  ∃ (α : ℝ), α = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end NUMINAMATH_GPT_apex_angle_of_quadrilateral_pyramid_l2203_220331


namespace NUMINAMATH_GPT_jason_flame_time_l2203_220351

-- Define firing interval and flame duration
def firing_interval := 15
def flame_duration := 5

-- Define the function to calculate seconds per minute
def seconds_per_minute (interval : ℕ) (duration : ℕ) : ℕ :=
  (60 / interval) * duration

-- Theorem to state the problem
theorem jason_flame_time : 
  seconds_per_minute firing_interval flame_duration = 20 := 
by
  sorry

end NUMINAMATH_GPT_jason_flame_time_l2203_220351


namespace NUMINAMATH_GPT_rows_seating_l2203_220332

theorem rows_seating (x y : ℕ) (h : 7 * x + 6 * y = 52) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_rows_seating_l2203_220332


namespace NUMINAMATH_GPT_prob_yellow_is_3_over_5_required_red_balls_is_8_l2203_220330

-- Defining the initial conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 4
def yellow_balls : ℕ := 6

-- Part 1: Prove the probability of drawing a yellow ball is 3/5
theorem prob_yellow_is_3_over_5 :
  (yellow_balls : ℚ) / (total_balls : ℚ) = 3 / 5 := sorry

-- Part 2: Prove that adding 8 red balls makes the probability of drawing a red ball 2/3
theorem required_red_balls_is_8 (x : ℕ) :
  (red_balls + x : ℚ) / (total_balls + x : ℚ) = 2 / 3 → x = 8 := sorry

end NUMINAMATH_GPT_prob_yellow_is_3_over_5_required_red_balls_is_8_l2203_220330


namespace NUMINAMATH_GPT_unique_real_solution_l2203_220311

-- Define the variables
variables (x y : ℝ)

-- State the condition
def equation (x y : ℝ) : Prop :=
  (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * y

-- State the theorem
theorem unique_real_solution (y : ℝ) (h_y : 0 < y) : ∃! x : ℝ, equation x y :=
sorry

end NUMINAMATH_GPT_unique_real_solution_l2203_220311


namespace NUMINAMATH_GPT_math_problem_l2203_220305

theorem math_problem
  (x y z : ℤ)
  (hz : z ≠ 0)
  (eq1 : 2 * x - 3 * y - z = 0)
  (eq2 : x + 3 * y - 14 * z = 0) :
  (x^2 - x * y) / (y^2 + 2 * z^2) = 10 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l2203_220305


namespace NUMINAMATH_GPT_no_representation_of_form_eight_k_plus_3_or_5_l2203_220366

theorem no_representation_of_form_eight_k_plus_3_or_5 (k : ℤ) :
  ∀ x y : ℤ, (8 * k + 3 ≠ x^2 - 2 * y^2) ∧ (8 * k + 5 ≠ x^2 - 2 * y^2) :=
by sorry

end NUMINAMATH_GPT_no_representation_of_form_eight_k_plus_3_or_5_l2203_220366


namespace NUMINAMATH_GPT_product_evaluation_l2203_220368

noncomputable def product_term (n : ℕ) : ℚ :=
  1 - (1 / (n * n))

noncomputable def product_expression : ℚ :=
  10 * 71 * (product_term 2) * (product_term 3) * (product_term 4) * (product_term 5) *
  (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10)

theorem product_evaluation : product_expression = 71 := by
  sorry

end NUMINAMATH_GPT_product_evaluation_l2203_220368


namespace NUMINAMATH_GPT_slope_at_two_l2203_220373

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2
noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem slope_at_two (a b : ℝ) (h1 : f' 1 a b = 0) (h2 : f 1 a b = 10) :
  f' 2 4 (-11) = 17 :=
sorry

end NUMINAMATH_GPT_slope_at_two_l2203_220373


namespace NUMINAMATH_GPT_max_children_tickets_l2203_220301

theorem max_children_tickets 
  (total_budget : ℕ) (adult_ticket_cost : ℕ) 
  (child_ticket_cost_individual : ℕ) (child_ticket_cost_group : ℕ) (min_group_tickets : ℕ) 
  (remaining_budget : ℕ) :
  total_budget = 75 →
  adult_ticket_cost = 12 →
  child_ticket_cost_individual = 6 →
  child_ticket_cost_group = 4 →
  min_group_tickets = 5 →
  (remaining_budget = total_budget - adult_ticket_cost) →
  ∃ (n : ℕ), n = 15 ∧ n * child_ticket_cost_group ≤ remaining_budget :=
by
  intros h_total_budget h_adult_ticket_cost h_child_ticket_cost_individual h_child_ticket_cost_group h_min_group_tickets h_remaining_budget
  sorry

end NUMINAMATH_GPT_max_children_tickets_l2203_220301


namespace NUMINAMATH_GPT_parabola_range_proof_l2203_220348

noncomputable def parabola_range (a : ℝ) : Prop := 
  (-2 ≤ a ∧ a < 3) → 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19)

theorem parabola_range_proof (a : ℝ) (h : -2 ≤ a ∧ a < 3) : 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19) :=
sorry

end NUMINAMATH_GPT_parabola_range_proof_l2203_220348


namespace NUMINAMATH_GPT_intersection_M_N_l2203_220325

def M := { x : ℝ | |x| ≤ 1 }
def N := { x : ℝ | x^2 - x < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2203_220325


namespace NUMINAMATH_GPT_cost_of_two_sandwiches_l2203_220341

theorem cost_of_two_sandwiches (J S : ℝ) 
  (h1 : 5 * J = 10) 
  (h2 : S + J = 5) :
  2 * S = 6 := 
sorry

end NUMINAMATH_GPT_cost_of_two_sandwiches_l2203_220341


namespace NUMINAMATH_GPT_problem_statement_l2203_220323

-- Definitions for the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- The main statement that needs to be proved
theorem problem_statement (f : ℝ → ℝ) (h_odd : odd_function f) (h_monotone : monotone_decreasing f) : f (-1) > f 3 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2203_220323


namespace NUMINAMATH_GPT_jill_arrives_before_jack_l2203_220382

def pool_distance : ℝ := 2
def jill_speed : ℝ := 12
def jack_speed : ℝ := 4
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem jill_arrives_before_jack
    (d : ℝ) (v_jill : ℝ) (v_jack : ℝ) (convert : ℝ → ℝ)
    (h_d : d = pool_distance)
    (h_vj : v_jill = jill_speed)
    (h_vk : v_jack = jack_speed)
    (h_convert : convert = hours_to_minutes) :
  convert (d / v_jack) - convert (d / v_jill) = 20 := by
  sorry

end NUMINAMATH_GPT_jill_arrives_before_jack_l2203_220382


namespace NUMINAMATH_GPT_anne_distance_l2203_220335
  
theorem anne_distance (S T : ℕ) (H1 : S = 2) (H2 : T = 3) : S * T = 6 := by
  -- Given that speed S = 2 miles/hour and time T = 3 hours, we need to show the distance S * T = 6 miles.
  sorry

end NUMINAMATH_GPT_anne_distance_l2203_220335


namespace NUMINAMATH_GPT_cylinder_dimensions_l2203_220333

theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h d : ℝ)
  (h_d_eq : h = d) (r_sphere_val : r_sphere = 6) 
  (sphere_area_eq : 4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h) :
  h = 12 ∧ d = 12 :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_dimensions_l2203_220333


namespace NUMINAMATH_GPT_xiaohong_height_l2203_220303

theorem xiaohong_height 
  (father_height_cm : ℕ)
  (height_difference_dm : ℕ)
  (father_height : father_height_cm = 170)
  (height_difference : height_difference_dm = 4) :
  ∃ xiaohong_height_cm : ℕ, xiaohong_height_cm + height_difference_dm * 10 = father_height_cm :=
by
  use 130
  sorry

end NUMINAMATH_GPT_xiaohong_height_l2203_220303


namespace NUMINAMATH_GPT_product_decrease_l2203_220363

variable (a b : ℤ)

theorem product_decrease : (a - 3) * (b + 3) - a * b = 900 → a - b = 303 → a * b - (a + 3) * (b - 3) = 918 :=
by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_product_decrease_l2203_220363


namespace NUMINAMATH_GPT_total_numbers_l2203_220336

theorem total_numbers (N : ℕ) (sum_total : ℝ) (avg_total : ℝ) (avg1 : ℝ) (avg2 : ℝ) (avg3 : ℝ) :
  avg_total = 6.40 → avg1 = 6.2 → avg2 = 6.1 → avg3 = 6.9 →
  sum_total = 2 * avg1 + 2 * avg2 + 2 * avg3 →
  N = sum_total / avg_total →
  N = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_numbers_l2203_220336


namespace NUMINAMATH_GPT_tiffany_total_score_l2203_220357

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end NUMINAMATH_GPT_tiffany_total_score_l2203_220357


namespace NUMINAMATH_GPT_sphere_volume_in_cone_l2203_220339

theorem sphere_volume_in_cone :
  let d := 24
  let theta := 90
  let r := 24 * (Real.sqrt 2 - 1)
  let V := (4 / 3) * Real.pi * r^3
  ∃ (R : ℝ), r = R ∧ V = (4 / 3) * Real.pi * R^3 := by
  sorry

end NUMINAMATH_GPT_sphere_volume_in_cone_l2203_220339


namespace NUMINAMATH_GPT_number_of_2_dollar_socks_l2203_220383

-- Given conditions
def total_pairs (a b c : ℕ) := a + b + c = 15
def total_cost (a b c : ℕ) := 2 * a + 4 * b + 5 * c = 41
def min_each_pair (a b c : ℕ) := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- To be proved
theorem number_of_2_dollar_socks (a b c : ℕ) (h1 : total_pairs a b c) (h2 : total_cost a b c) (h3 : min_each_pair a b c) : 
  a = 11 := 
  sorry

end NUMINAMATH_GPT_number_of_2_dollar_socks_l2203_220383


namespace NUMINAMATH_GPT_table_tennis_matches_l2203_220381

theorem table_tennis_matches (n : ℕ) :
  ∃ x : ℕ, 3 * 2 - x + n * (n - 1) / 2 = 50 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_table_tennis_matches_l2203_220381


namespace NUMINAMATH_GPT_strawberry_unit_prices_l2203_220358

theorem strawberry_unit_prices (x y : ℝ) (h1 : x = 1.5 * y) (h2 : 2 * x - 2 * y = 10) : x = 15 ∧ y = 10 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_unit_prices_l2203_220358


namespace NUMINAMATH_GPT_difference_in_pencil_buyers_l2203_220384

theorem difference_in_pencil_buyers :
  ∀ (cost_per_pencil : ℕ) (total_cost_eighth_graders : ℕ) (total_cost_fifth_graders : ℕ), 
  cost_per_pencil = 13 →
  total_cost_eighth_graders = 234 →
  total_cost_fifth_graders = 325 →
  (total_cost_fifth_graders / cost_per_pencil) - (total_cost_eighth_graders / cost_per_pencil) = 7 :=
by
  intros cost_per_pencil total_cost_eighth_graders total_cost_fifth_graders 
         hcpe htc8 htc5
  sorry

end NUMINAMATH_GPT_difference_in_pencil_buyers_l2203_220384


namespace NUMINAMATH_GPT_boys_number_l2203_220397

variable (M W B : ℕ)

-- Conditions
axiom h1 : M = W
axiom h2 : W = B
axiom h3 : M * 8 = 120

theorem boys_number :
  B = 15 := by
  sorry

end NUMINAMATH_GPT_boys_number_l2203_220397


namespace NUMINAMATH_GPT_proportional_function_ratio_l2203_220345

-- Let k be a constant, and y = kx be a proportional function.
-- We know that f(1) = 3 and f(a) = b where b ≠ 0.
-- We want to prove that a / b = 1 / 3.

theorem proportional_function_ratio (a b k : ℝ) :
  (∀ x, x = 1 → k * x = 3) →
  (∀ x, x = a → k * x = b) →
  b ≠ 0 →
  a / b = 1 / 3 :=
by
  intros h1 h2 h3
  -- the proof will follow but is not required here
  sorry

end NUMINAMATH_GPT_proportional_function_ratio_l2203_220345


namespace NUMINAMATH_GPT_triangle_area_ratio_l2203_220380

theorem triangle_area_ratio (x y : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let A_area := (1/2) * (y/n) * (x/2)
  let B_area := (1/2) * (x/m) * (y/2)
  A_area / B_area = m / n := by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l2203_220380


namespace NUMINAMATH_GPT_find_q_minus_p_l2203_220354

theorem find_q_minus_p (p q : ℕ) (h1 : 0 < p) (h2 : 0 < q) 
  (h3 : 6 * q < 11 * p) (h4 : 9 * p < 5 * q) (h_min : ∀ r : ℕ, r > 0 → (6:ℚ)/11 < (p:ℚ)/r → (p:ℚ)/r < (5:ℚ)/9 → q ≤ r) :
  q - p = 9 :=
sorry

end NUMINAMATH_GPT_find_q_minus_p_l2203_220354


namespace NUMINAMATH_GPT_problem1_problem2_l2203_220375

-- Define the given angle
def given_angle (α : ℝ) : Prop := α = 2010

-- Define the theorem for the first problem
theorem problem1 (α : ℝ) (k : ℤ) (β : ℝ) (h₁ : given_angle α) 
  (h₂ : 0 ≤ β ∧ β < 360) (h₃ : α = k * 360 + β) : 
  -- Assert that α is in the third quadrant
  (190 ≤ β ∧ β < 270 → true) :=
sorry

-- Define the theorem for the second problem
theorem problem2 (α : ℝ) (θ : ℝ) (h₁ : given_angle α)
  (h₂ : -360 ≤ θ ∧ θ < 720)
  (h₃ : ∃ k : ℤ, θ = α + k * 360) : 
  θ = -150 ∨ θ = 210 ∨ θ = 570 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2203_220375


namespace NUMINAMATH_GPT_range_of_a_l2203_220386

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2203_220386


namespace NUMINAMATH_GPT_ratio_of_pieces_l2203_220364

theorem ratio_of_pieces (total_length shorter_piece longer_piece : ℕ) 
    (h1 : total_length = 6) (h2 : shorter_piece = 2)
    (h3 : longer_piece = total_length - shorter_piece) :
    ((longer_piece : ℚ) / (shorter_piece : ℚ)) = 2 :=
by
    sorry

end NUMINAMATH_GPT_ratio_of_pieces_l2203_220364


namespace NUMINAMATH_GPT_parallelepiped_diagonal_inequality_l2203_220319

theorem parallelepiped_diagonal_inequality 
  (a b c d : ℝ) 
  (h_d : d = Real.sqrt (a^2 + b^2 + c^2)) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_parallelepiped_diagonal_inequality_l2203_220319


namespace NUMINAMATH_GPT_largest_ordered_pair_exists_l2203_220362

-- Define the condition for ordered pairs (a, b)
def ordered_pair_condition (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ ∃ (k : ℤ), (a + b) * (a + b + 1) = k * a * b

-- Define the specific ordered pair to be checked
def specific_pair (a b : ℤ) : Prop :=
  a = 35 ∧ b = 90

-- The main statement to be proven
theorem largest_ordered_pair_exists : specific_pair 35 90 ∧ ordered_pair_condition 35 90 :=
by
  sorry

end NUMINAMATH_GPT_largest_ordered_pair_exists_l2203_220362
