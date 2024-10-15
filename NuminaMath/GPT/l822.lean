import Mathlib

namespace NUMINAMATH_GPT_fraction_sum_eq_l822_82254

theorem fraction_sum_eq : (7 / 10 : ℚ) + (3 / 100) + (9 / 1000) = 0.739 := sorry

end NUMINAMATH_GPT_fraction_sum_eq_l822_82254


namespace NUMINAMATH_GPT_sequence_formula_l822_82257

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
  (a 1 = 11) ∧
  (a 2 = 102) ∧
  (a 3 = 1003) ∧
  (a 4 = 10004) →
  ∀ n, a n = 10^n + n := by
  sorry

end NUMINAMATH_GPT_sequence_formula_l822_82257


namespace NUMINAMATH_GPT_slices_of_bread_left_l822_82225

variable (monday_to_friday_slices saturday_slices total_slices_used initial_slices slices_left: ℕ)

def sandwiches_monday_to_friday : ℕ := 5
def slices_per_sandwich : ℕ := 2
def sandwiches_saturday : ℕ := 2
def initial_slices_of_bread : ℕ := 22

theorem slices_of_bread_left :
  slices_left = initial_slices_of_bread - total_slices_used
  :=
by  sorry

end NUMINAMATH_GPT_slices_of_bread_left_l822_82225


namespace NUMINAMATH_GPT_minimum_value_l822_82282

open Real

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log (2^x) + log (8^y) = log 2) :
  ∃ (v : ℝ), v = 4 ∧ ∀ u, (∀ x y, x > 0 ∧ y > 0 → log (2^x) + log (8^y) = log 2 → x + 3*y = 1 → u = 4) := sorry

end NUMINAMATH_GPT_minimum_value_l822_82282


namespace NUMINAMATH_GPT_birds_in_tree_l822_82288

def initialBirds : Nat := 14
def additionalBirds : Nat := 21
def totalBirds := initialBirds + additionalBirds

theorem birds_in_tree : totalBirds = 35 := by
  sorry

end NUMINAMATH_GPT_birds_in_tree_l822_82288


namespace NUMINAMATH_GPT_length_AB_slope_one_OA_dot_OB_const_l822_82207

open Real

def parabola (x y : ℝ) : Prop := y * y = 4 * x
def line_through_focus (x y : ℝ) (k : ℝ) : Prop := x = k * y + 1
def line_slope_one (x y : ℝ) : Prop := y = x - 1

theorem length_AB_slope_one {x1 x2 y1 y2 : ℝ} (hA : parabola x1 y1) (hB : parabola x2 y2) 
  (hL : line_slope_one x1 y1) (hL' : line_slope_one x2 y2) : abs (x1 - x2) + abs (y1 - y2) = 8 := 
by
  sorry

theorem OA_dot_OB_const {x1 x2 y1 y2 : ℝ} {k : ℝ} (hA : parabola x1 y1)
  (hB : parabola x2 y2) (hL : line_through_focus x1 y1 k) (hL' : line_through_focus x2 y2 k) :
  x1 * x2 + y1 * y2 = -3 :=
by
  sorry

end NUMINAMATH_GPT_length_AB_slope_one_OA_dot_OB_const_l822_82207


namespace NUMINAMATH_GPT_rationalize_denominator_correct_l822_82228

noncomputable def rationalize_denominator : Prop :=
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_correct_l822_82228


namespace NUMINAMATH_GPT_integer_ratio_zero_l822_82249

theorem integer_ratio_zero
  (A B : ℤ)
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -1 → (A / (x - 3 : ℝ) + B / (x ^ 2 + 2 * x + 1) = (x ^ 3 - x ^ 2 + 3 * x + 1) / (x ^ 3 - x - 3))) :
  B / A = 0 :=
sorry

end NUMINAMATH_GPT_integer_ratio_zero_l822_82249


namespace NUMINAMATH_GPT_vertical_shirts_count_l822_82295

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end NUMINAMATH_GPT_vertical_shirts_count_l822_82295


namespace NUMINAMATH_GPT_combined_cost_increase_l822_82209

def original_bicycle_cost : ℝ := 200
def original_skates_cost : ℝ := 50
def bike_increase_percent : ℝ := 0.06
def skates_increase_percent : ℝ := 0.15

noncomputable def new_bicycle_cost : ℝ := original_bicycle_cost * (1 + bike_increase_percent)
noncomputable def new_skates_cost : ℝ := original_skates_cost * (1 + skates_increase_percent)
noncomputable def original_total_cost : ℝ := original_bicycle_cost + original_skates_cost
noncomputable def new_total_cost : ℝ := new_bicycle_cost + new_skates_cost
noncomputable def total_increase : ℝ := new_total_cost - original_total_cost
noncomputable def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem combined_cost_increase : percent_increase = 7.8 := by
  sorry

end NUMINAMATH_GPT_combined_cost_increase_l822_82209


namespace NUMINAMATH_GPT_fraction_simplification_l822_82226

theorem fraction_simplification :
  ( (5^1004)^4 - (5^1002)^4 ) / ( (5^1003)^4 - (5^1001)^4 ) = 25 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l822_82226


namespace NUMINAMATH_GPT_prime_factors_and_divisors_6440_l822_82252

theorem prime_factors_and_divisors_6440 :
  ∃ (a b c d : ℕ), 6440 = 2^a * 5^b * 7^c * 23^d ∧ a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧
  (a + 1) * (b + 1) * (c + 1) * (d + 1) = 32 :=
by 
  sorry

end NUMINAMATH_GPT_prime_factors_and_divisors_6440_l822_82252


namespace NUMINAMATH_GPT_marvelous_class_student_count_l822_82277

theorem marvelous_class_student_count (g : ℕ) (jb : ℕ) (jg : ℕ) (j_total : ℕ) (jl : ℕ) (init_jb : ℕ) : 
  jb = g + 3 →  -- Number of boys
  jg = 2 * g + 1 →  -- Jelly beans received by each girl
  init_jb = 726 →  -- Initial jelly beans
  jl = 4 →  -- Leftover jelly beans
  j_total = init_jb - jl →  -- Jelly beans distributed
  (jb * jb + g * jg = j_total) → -- Total jelly beans distributed equation
  2 * g + 1 + g + jb = 31 := -- Total number of students
by
  sorry

end NUMINAMATH_GPT_marvelous_class_student_count_l822_82277


namespace NUMINAMATH_GPT_value_before_decrease_l822_82244

theorem value_before_decrease
  (current_value decrease : ℤ)
  (current_value_equals : current_value = 1460)
  (decrease_equals : decrease = 12) :
  current_value + decrease = 1472 :=
by
  -- We assume the proof to follow here.
  sorry

end NUMINAMATH_GPT_value_before_decrease_l822_82244


namespace NUMINAMATH_GPT_Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l822_82298

theorem Mishas_fathers_speed (d : ℝ) (t : ℝ) (V : ℝ) 
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) :
  V = 1 :=
by
  sorry

theorem Mishas_fathers_speed_in_kmh (d : ℝ) (t : ℝ) (V : ℝ) (V_kmh : ℝ)
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) 
  (h4 : V_kmh = V * 60):
  V_kmh = 60 :=
by
  sorry

end NUMINAMATH_GPT_Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l822_82298


namespace NUMINAMATH_GPT_complete_the_square_3x2_9x_20_l822_82275

theorem complete_the_square_3x2_9x_20 : 
  ∃ (k : ℝ), (3:ℝ) * (x + ((-3)/2))^2 + k = 3 * x^2 + 9 * x + 20  :=
by
  -- Using exists
  use (53/4:ℝ)
  sorry

end NUMINAMATH_GPT_complete_the_square_3x2_9x_20_l822_82275


namespace NUMINAMATH_GPT_isosceles_triangle_area_l822_82237

theorem isosceles_triangle_area :
  ∀ (P Q R S : ℝ) (h1 : dist P Q = 26) (h2 : dist P R = 26) (h3 : dist Q R = 50),
  ∃ (area : ℝ), area = 25 * Real.sqrt 51 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l822_82237


namespace NUMINAMATH_GPT_minimum_value_fraction_l822_82285

theorem minimum_value_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1) :
  (1 / a) + (4 / b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l822_82285


namespace NUMINAMATH_GPT_impossible_digit_filling_l822_82273

theorem impossible_digit_filling (T : Fin 5 → Fin 8 → Fin 10) :
  (∀ d : Fin 10, (∃! r₁ r₂ r₃ r₄ : Fin 5, T r₁ = d ∧ T r₂ = d ∧ T r₃ = d ∧ T r₄ = d) ∧
                 (∃! c₁ c₂ c₃ c₄ : Fin 8, T c₁ = d ∧ T c₂ = d ∧ T c₃ = d ∧ T c₄ = d)) → False :=
by
  sorry

end NUMINAMATH_GPT_impossible_digit_filling_l822_82273


namespace NUMINAMATH_GPT_crossing_time_indeterminate_l822_82256

-- Define the lengths of the two trains.
def train_A_length : Nat := 120
def train_B_length : Nat := 150

-- Define the crossing time of the two trains when moving in the same direction.
def crossing_time_together : Nat := 135

-- Define a theorem to state that without additional information, the crossing time for a 150-meter train cannot be determined.
theorem crossing_time_indeterminate 
    (V120 V150 : Nat) 
    (H : V150 - V120 = 2) : 
    ∃ t, t > 0 -> t < 150 / V150 -> False :=
by 
    -- The proof is not provided.
    sorry

end NUMINAMATH_GPT_crossing_time_indeterminate_l822_82256


namespace NUMINAMATH_GPT_abs_neg_three_eq_three_l822_82283

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
sorry

end NUMINAMATH_GPT_abs_neg_three_eq_three_l822_82283


namespace NUMINAMATH_GPT_min_tables_42_l822_82247

def min_tables_needed (total_people : ℕ) (table_sizes : List ℕ) : ℕ :=
  sorry

theorem min_tables_42 :
  min_tables_needed 42 [4, 6, 8] = 6 :=
sorry

end NUMINAMATH_GPT_min_tables_42_l822_82247


namespace NUMINAMATH_GPT_exponentiation_division_l822_82214

variable {a : ℝ} (h1 : (a^2)^3 = a^6) (h2 : a^6 / a^2 = a^4)

theorem exponentiation_division : (a^2)^3 / a^2 = a^4 := 
by 
  sorry

end NUMINAMATH_GPT_exponentiation_division_l822_82214


namespace NUMINAMATH_GPT_seating_arrangements_l822_82265

theorem seating_arrangements (n : ℕ) (h_n : n = 6) (A B : Fin n) (h : A ≠ B) : 
  ∃ k : ℕ, k = 240 := 
by 
  sorry

end NUMINAMATH_GPT_seating_arrangements_l822_82265


namespace NUMINAMATH_GPT_mike_ride_distance_l822_82210

/-- 
Mike took a taxi to the airport and paid a starting amount plus $0.25 per mile. 
Annie took a different route to the airport and paid the same starting amount plus $5.00 in bridge toll fees plus $0.25 per mile. 
Each was charged exactly the same amount, and Annie's ride was 26 miles. 
Prove that Mike's ride was 46 miles given his starting amount was $2.50.
-/
theorem mike_ride_distance
  (S C A_miles : ℝ)                  -- S: starting amount, C: cost per mile, A_miles: Annie's ride distance
  (bridge_fee total_cost : ℝ)        -- bridge_fee: Annie's bridge toll fee, total_cost: total cost for both
  (M : ℝ)                            -- M: Mike's ride distance
  (hS : S = 2.5)
  (hC : C = 0.25)
  (hA_miles : A_miles = 26)
  (h_bridge_fee : bridge_fee = 5)
  (h_total_cost_equal : total_cost = S + bridge_fee + (C * A_miles))
  (h_total_cost_mike : total_cost = S + (C * M)) :
  M = 46 :=
by 
  sorry

end NUMINAMATH_GPT_mike_ride_distance_l822_82210


namespace NUMINAMATH_GPT_length_of_room_l822_82239

theorem length_of_room (L : ℝ) (w : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) (room_area : ℝ) :
  w = 12 →
  veranda_width = 2 →
  veranda_area = 144 →
  (L + 2 * veranda_width) * (w + 2 * veranda_width) - L * w = veranda_area →
  L = 20 :=
by
  intro h_w
  intro h_veranda_width
  intro h_veranda_area
  intro h_area_eq
  sorry

end NUMINAMATH_GPT_length_of_room_l822_82239


namespace NUMINAMATH_GPT_jake_has_more_balloons_l822_82211

-- Defining the given conditions as parameters
def initial_balloons_allan : ℕ := 2
def initial_balloons_jake : ℕ := 6
def additional_balloons_allan : ℕ := 3

-- Calculate total balloons each person has
def total_balloons_allan : ℕ := initial_balloons_allan + additional_balloons_allan
def total_balloons_jake : ℕ := initial_balloons_jake

-- Formalize the statement to be proved
theorem jake_has_more_balloons :
  total_balloons_jake - total_balloons_allan = 1 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_jake_has_more_balloons_l822_82211


namespace NUMINAMATH_GPT_product_xyz_l822_82248

variables (x y z : ℝ)

theorem product_xyz (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_xyz_l822_82248


namespace NUMINAMATH_GPT_part1_part2_l822_82235

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem part1 : f (5 * Real.pi / 6) = -2 := by
  sorry

variables {A : ℝ} (hA1 : A > 0) (hA2 : A ≤ Real.pi / 3) (hFA : f A = 8 / 5)

theorem part2 (h : A > 0 ∧ A ≤ Real.pi / 3 ∧ f A = 8 / 5) : f (A + Real.pi / 4) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l822_82235


namespace NUMINAMATH_GPT_find_x_l822_82262

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : x * floor x = 50) : x = 7.142857 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l822_82262


namespace NUMINAMATH_GPT_edward_money_l822_82286

theorem edward_money (initial_amount spent1 spent2 : ℕ) (h_initial : initial_amount = 34) (h_spent1 : spent1 = 9) (h_spent2 : spent2 = 8) :
  initial_amount - (spent1 + spent2) = 17 :=
by
  sorry

end NUMINAMATH_GPT_edward_money_l822_82286


namespace NUMINAMATH_GPT_integer_solution_l822_82272

theorem integer_solution (x : ℤ) (h : x^2 < 3 * x) : x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_GPT_integer_solution_l822_82272


namespace NUMINAMATH_GPT_length_of_common_chord_l822_82206

-- Problem conditions
variables (r : ℝ) (h : r = 15)

-- Statement to prove
theorem length_of_common_chord : 2 * (r / 2 * Real.sqrt 3) = 15 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_common_chord_l822_82206


namespace NUMINAMATH_GPT_elois_banana_bread_l822_82263

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end NUMINAMATH_GPT_elois_banana_bread_l822_82263


namespace NUMINAMATH_GPT_complex_multiplication_l822_82292

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  ((a + b * i) * (c + d * i)) = (-6 + 33 * i) :=
by
  have a := 3
  have b := -4
  have c := -6
  have d := 3
  sorry

end NUMINAMATH_GPT_complex_multiplication_l822_82292


namespace NUMINAMATH_GPT_increase_in_average_l822_82243

variable (A : ℝ)
variable (new_avg : ℝ := 44)
variable (score_12th_inning : ℝ := 55)
variable (total_runs_after_11 : ℝ := 11 * A)

theorem increase_in_average :
  ((total_runs_after_11 + score_12th_inning) / 12 - A = 1) :=
by
  sorry

end NUMINAMATH_GPT_increase_in_average_l822_82243


namespace NUMINAMATH_GPT_smallest_n_for_square_and_cube_l822_82236

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end NUMINAMATH_GPT_smallest_n_for_square_and_cube_l822_82236


namespace NUMINAMATH_GPT_greatest_radius_l822_82222

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end NUMINAMATH_GPT_greatest_radius_l822_82222


namespace NUMINAMATH_GPT_sphere_radius_equals_three_l822_82261

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_equals_three_l822_82261


namespace NUMINAMATH_GPT_mike_arcade_ratio_l822_82251

theorem mike_arcade_ratio :
  ∀ (weekly_pay food_cost hourly_rate play_minutes : ℕ),
    weekly_pay = 100 →
    food_cost = 10 →
    hourly_rate = 8 →
    play_minutes = 300 →
    (food_cost + (play_minutes / 60) * hourly_rate) / weekly_pay = 1 / 2 := 
by
  intros weekly_pay food_cost hourly_rate play_minutes h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_mike_arcade_ratio_l822_82251


namespace NUMINAMATH_GPT_journal_sessions_per_week_l822_82215

/-- Given that each student writes 4 pages in each session and will write 72 journal pages in 6 weeks, prove that there are 3 journal-writing sessions per week.
--/
theorem journal_sessions_per_week (pages_per_session : ℕ) (total_pages : ℕ) (weeks : ℕ) (sessions_per_week : ℕ) :
  pages_per_session = 4 →
  total_pages = 72 →
  weeks = 6 →
  total_pages = pages_per_session * sessions_per_week * weeks →
  sessions_per_week = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_journal_sessions_per_week_l822_82215


namespace NUMINAMATH_GPT_cube_surface_area_proof_l822_82242

-- Conditions
def prism_volume : ℕ := 10 * 5 * 20
def cube_volume : ℕ := 1000
def edge_length_of_cube : ℕ := 10
def cube_surface_area (s : ℕ) : ℕ := 6 * s * s

-- Theorem Statement
theorem cube_surface_area_proof : cube_volume = prism_volume → cube_surface_area edge_length_of_cube = 600 := 
by
  intros h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cube_surface_area_proof_l822_82242


namespace NUMINAMATH_GPT_equidistant_point_on_x_axis_l822_82205

theorem equidistant_point_on_x_axis (x : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 0)) (hB : B = (3, 5)) :
  (Real.sqrt ((x - (-3))^2)) = (Real.sqrt ((x - 3)^2 + 25)) →
  x = 25 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_equidistant_point_on_x_axis_l822_82205


namespace NUMINAMATH_GPT_larger_sphere_radius_l822_82266

theorem larger_sphere_radius (r : ℝ) (π : ℝ) (h : r^3 = 2) :
  r = 2^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_larger_sphere_radius_l822_82266


namespace NUMINAMATH_GPT_equation_of_trajectory_l822_82278

open Real

variable (P : ℝ → ℝ → Prop)
variable (C : ℝ → ℝ → Prop)
variable (L : ℝ → ℝ → Prop)

-- Definition of the fixed circle C
def fixed_circle (x y : ℝ) : Prop :=
  (x + 2) ^ 2 + y ^ 2 = 1

-- Definition of the fixed line L
def fixed_line (x y : ℝ) : Prop := 
  x = 1

noncomputable def moving_circle (P : ℝ → ℝ → Prop) (r : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ r > 0 ∧
  (∀ a b : ℝ, fixed_circle a b → ((x - a) ^ 2 + (y - b) ^ 2) = (r + 1) ^ 2) ∧
  (∀ a b : ℝ, fixed_line a b → (abs (x - a)) = (r + 1))

theorem equation_of_trajectory
  (P : ℝ → ℝ → Prop)
  (r : ℝ)
  (h : moving_circle P r) :
  ∀ x y : ℝ, P x y → y ^ 2 = -8 * x :=
by
  sorry

end NUMINAMATH_GPT_equation_of_trajectory_l822_82278


namespace NUMINAMATH_GPT_solve_for_m_l822_82201

theorem solve_for_m 
  (m : ℝ) 
  (h : (m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) : m = 6 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l822_82201


namespace NUMINAMATH_GPT_base_r_correct_l822_82250

theorem base_r_correct (r : ℕ) :
  (5 * r ^ 2 + 6 * r) + (4 * r ^ 2 + 2 * r) = r ^ 3 + r ^ 2 → r = 8 := 
by 
  sorry

end NUMINAMATH_GPT_base_r_correct_l822_82250


namespace NUMINAMATH_GPT_triangle_side_length_mod_l822_82212

theorem triangle_side_length_mod {a d x : ℕ} 
  (h_equilateral : ∃ (a : ℕ), 3 * a = 1 + d + x)
  (h_triangle : ∀ {a d x : ℕ}, 1 + d > x ∧ 1 + x > d ∧ d + x > 1)
  : d % 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_mod_l822_82212


namespace NUMINAMATH_GPT_misha_second_round_score_l822_82279

def misha_score_first_round (darts : ℕ) (score_per_dart_min : ℕ) : ℕ := 
  darts * score_per_dart_min

def misha_score_second_round (score_first : ℕ) (multiplier : ℕ) : ℕ := 
  score_first * multiplier

def misha_score_third_round (score_second : ℕ) (multiplier : ℚ) : ℚ := 
  score_second * multiplier

theorem misha_second_round_score (darts : ℕ) (score_per_dart_min : ℕ) (multiplier_second : ℕ) (multiplier_third : ℚ) 
  (h_darts : darts = 8) (h_score_per_dart_min : score_per_dart_min = 3) (h_multiplier_second : multiplier_second = 2) (h_multiplier_third : multiplier_third = 1.5) :
  misha_score_second_round (misha_score_first_round darts score_per_dart_min) multiplier_second = 48 :=
by sorry

end NUMINAMATH_GPT_misha_second_round_score_l822_82279


namespace NUMINAMATH_GPT__l822_82291

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : triangle_inequality 1 1 1 := 
by {
  -- Prove using the triangle inequality theorem that the sides form a triangle.
  -- This part is left as an exercise to the reader.
  sorry
}

end NUMINAMATH_GPT__l822_82291


namespace NUMINAMATH_GPT_sin_480_deg_l822_82255

theorem sin_480_deg : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_480_deg_l822_82255


namespace NUMINAMATH_GPT_quotient_of_poly_div_l822_82217

theorem quotient_of_poly_div :
  (10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6) / (5 * X^2 + 7) =
  2 * X^2 - X - (11 / 5) :=
sorry

end NUMINAMATH_GPT_quotient_of_poly_div_l822_82217


namespace NUMINAMATH_GPT_inscribed_square_side_length_l822_82238

-- Define a right triangle
structure RightTriangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)
  (is_right : PQ^2 + QR^2 = PR^2)

-- Define the triangle PQR
def trianglePQR : RightTriangle :=
  { PQ := 6, QR := 8, PR := 10, is_right := by norm_num }

-- Define the problem statement
theorem inscribed_square_side_length (t : ℝ) (h : RightTriangle) :
  t = 3 :=
  sorry

end NUMINAMATH_GPT_inscribed_square_side_length_l822_82238


namespace NUMINAMATH_GPT_students_in_canteen_l822_82274

-- Definitions for conditions
def total_students : ℕ := 40
def absent_fraction : ℚ := 1 / 10
def classroom_fraction : ℚ := 3 / 4

-- Lean 4 statement
theorem students_in_canteen :
  let absent_students := (absent_fraction * total_students)
  let present_students := (total_students - absent_students)
  let classroom_students := (classroom_fraction * present_students)
  let canteen_students := (present_students - classroom_students)
  canteen_students = 9 := by
    sorry

end NUMINAMATH_GPT_students_in_canteen_l822_82274


namespace NUMINAMATH_GPT_divisible_iff_condition_l822_82296

theorem divisible_iff_condition (a b : ℤ) : 
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) :=
  sorry

end NUMINAMATH_GPT_divisible_iff_condition_l822_82296


namespace NUMINAMATH_GPT_is_factor_l822_82271

-- Define the polynomial
def poly (x : ℝ) := x^4 + 4 * x^2 + 4

-- Define a candidate for being a factor
def factor_candidate (x : ℝ) := x^2 + 2

-- Proof problem: prove that factor_candidate is a factor of poly
theorem is_factor : ∀ x : ℝ, poly x = factor_candidate x * factor_candidate x := 
by
  intro x
  unfold poly factor_candidate
  sorry

end NUMINAMATH_GPT_is_factor_l822_82271


namespace NUMINAMATH_GPT_arcade_ticket_problem_l822_82287

-- Define all the conditions given in the problem
def initial_tickets : Nat := 13
def used_tickets : Nat := 8
def more_tickets_for_clothes : Nat := 10
def tickets_for_toys : Nat := 8
def tickets_for_clothes := tickets_for_toys + more_tickets_for_clothes

-- The proof statement (goal)
theorem arcade_ticket_problem : tickets_for_clothes = 18 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_arcade_ticket_problem_l822_82287


namespace NUMINAMATH_GPT_rationalize_denominator_l822_82240

theorem rationalize_denominator :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l822_82240


namespace NUMINAMATH_GPT_coordinates_equality_l822_82220

theorem coordinates_equality (a b : ℤ) 
  (h1 : b - 1 = 2) 
  (h2 : a + 3 = -1) : a + b = -1 :=
by 
  sorry

end NUMINAMATH_GPT_coordinates_equality_l822_82220


namespace NUMINAMATH_GPT_smallest_m_divisible_by_15_l822_82221

noncomputable def largest_prime_with_2023_digits : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ (largest_prime_with_2023_digits ^ 2 - m) % 15 = 0 ∧ m = 1 :=
  sorry

end NUMINAMATH_GPT_smallest_m_divisible_by_15_l822_82221


namespace NUMINAMATH_GPT_find_common_ratio_of_geometric_sequence_l822_82230

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem find_common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a n > a (n + 1))
  (h1 : a 1 * a 5 = 9)
  (h2 : a 2 + a 4 = 10) : 
  q = -1/3 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_of_geometric_sequence_l822_82230


namespace NUMINAMATH_GPT_proof_problem_l822_82234

def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem proof_problem (x : ℝ) :
  necessary_but_not_sufficient ((x+3)*(x-1) = 0) (x-1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l822_82234


namespace NUMINAMATH_GPT_eight_n_is_even_l822_82241

theorem eight_n_is_even (n : ℕ) (h : n = 7) : 8 * n = 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_eight_n_is_even_l822_82241


namespace NUMINAMATH_GPT_coloring_even_conditional_l822_82216

-- Define the problem parameters and constraints
def number_of_colorings (n : Nat) (even_red : Bool) (even_yellow : Bool) : Nat :=
  sorry  -- This function would contain the detailed computational logic.

-- Define the main theorem statement
theorem coloring_even_conditional (n : ℕ) (h1 : n > 0) : ∃ C : Nat, number_of_colorings n true true = C := 
by
  sorry  -- The proof would go here.


end NUMINAMATH_GPT_coloring_even_conditional_l822_82216


namespace NUMINAMATH_GPT_total_students_university_l822_82233

theorem total_students_university :
  ∀ (sample_size freshmen sophomores other_sample other_total total_students : ℕ),
  sample_size = 500 →
  freshmen = 200 →
  sophomores = 100 →
  other_sample = 200 →
  other_total = 3000 →
  total_students = (other_total * sample_size) / other_sample →
  total_students = 7500 :=
by
  intros sample_size freshmen sophomores other_sample other_total total_students
  sorry

end NUMINAMATH_GPT_total_students_university_l822_82233


namespace NUMINAMATH_GPT_always_meaningful_fraction_l822_82276

theorem always_meaningful_fraction {x : ℝ} : (∀ x, ∃ option : ℕ, 
  (option = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) ∨ 
  (option = 2 ∧ True) ∨ 
  (option = 3 ∧ x ≠ 0) ∨ 
  (option = 4 ∧ x ≠ 1)) → option = 2 :=
sorry

end NUMINAMATH_GPT_always_meaningful_fraction_l822_82276


namespace NUMINAMATH_GPT_range_of_a_l822_82281

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l822_82281


namespace NUMINAMATH_GPT_probability_two_dice_same_number_l822_82270

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end NUMINAMATH_GPT_probability_two_dice_same_number_l822_82270


namespace NUMINAMATH_GPT_arithmetic_sequence_y_l822_82208

theorem arithmetic_sequence_y :
  let a := 3^3
  let c := 3^5
  let y := (a + c) / 2
  y = 135 :=
by
  let a := 27
  let c := 243
  let y := (a + c) / 2
  show y = 135
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_y_l822_82208


namespace NUMINAMATH_GPT_ab_product_power_l822_82299

theorem ab_product_power (a b : ℤ) (n : ℕ) (h1 : (a * b)^n = 128 * 8) : n = 10 := by
  sorry

end NUMINAMATH_GPT_ab_product_power_l822_82299


namespace NUMINAMATH_GPT_investment_rate_l822_82264

theorem investment_rate (total_investment : ℝ) (invest1 : ℝ) (rate1 : ℝ) (invest2 : ℝ) (rate2 : ℝ) (desired_income : ℝ) (remaining_investment : ℝ) (remaining_rate : ℝ) : 
( total_investment = 12000 ∧ invest1 = 5000 ∧ rate1 = 0.06 ∧ invest2 = 4000 ∧ rate2 = 0.035 ∧ desired_income = 700 ∧ remaining_investment = 3000 ) → remaining_rate = 0.0867 :=
by
  sorry

end NUMINAMATH_GPT_investment_rate_l822_82264


namespace NUMINAMATH_GPT_smallest_n_mod_l822_82260

theorem smallest_n_mod :
  ∃ n : ℕ, (23 * n ≡ 5678 [MOD 11]) ∧ (∀ m : ℕ, (23 * m ≡ 5678 [MOD 11]) → (0 < n) ∧ (n ≤ m)) :=
  by
  sorry

end NUMINAMATH_GPT_smallest_n_mod_l822_82260


namespace NUMINAMATH_GPT_raft_people_with_life_jackets_l822_82268

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end NUMINAMATH_GPT_raft_people_with_life_jackets_l822_82268


namespace NUMINAMATH_GPT_existence_of_function_values_around_k_l822_82229

-- Define the function f(n, m) with the given properties
def is_valid_function (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n-1, m) + f (n+1, m) + f (n, m-1) + f (n, m+1)) / 4

-- Theorem to prove the existence of such a function
theorem existence_of_function :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f :=
sorry

-- Theorem to prove that for any k in ℤ, f(n, m) has values both greater and less than k
theorem values_around_k (k : ℤ) :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f ∧ (∃ n1 m1 n2 m2, f (n1, m1) > k ∧ f (n2, m2) < k) :=
sorry

end NUMINAMATH_GPT_existence_of_function_values_around_k_l822_82229


namespace NUMINAMATH_GPT_cone_base_circumference_l822_82245

theorem cone_base_circumference (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 240) :
  (2 / 3) * (2 * Real.pi * r) = 8 * Real.pi :=
by
  have circle_circumference : ℝ := 2 * Real.pi * r
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l822_82245


namespace NUMINAMATH_GPT_gold_bars_lost_l822_82202

-- Define the problem constants
def initial_bars : ℕ := 100
def friends : ℕ := 4
def bars_per_friend : ℕ := 20

-- Define the total distributed gold bars
def total_distributed : ℕ := friends * bars_per_friend

-- Define the number of lost gold bars
def lost_bars : ℕ := initial_bars - total_distributed

-- Theorem: Prove that the number of lost gold bars is 20
theorem gold_bars_lost : lost_bars = 20 := by
  sorry

end NUMINAMATH_GPT_gold_bars_lost_l822_82202


namespace NUMINAMATH_GPT_toys_produced_each_day_l822_82232

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_worked_per_week : ℕ)
  (same_number_toys_each_day : Prop) : 
  total_weekly_production = 4340 → days_worked_per_week = 2 → 
  same_number_toys_each_day →
  (total_weekly_production / days_worked_per_week = 2170) :=
by
  intros h_production h_days h_same_toys
  -- proof skipped
  sorry

end NUMINAMATH_GPT_toys_produced_each_day_l822_82232


namespace NUMINAMATH_GPT_trigonometric_identity_l822_82289

theorem trigonometric_identity : 
  (Real.cos (15 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) - Real.cos (75 * Real.pi / 180) * Real.sin (105 * Real.pi / 180))
  = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l822_82289


namespace NUMINAMATH_GPT_no_arithmetic_sequence_without_square_gt1_l822_82258

theorem no_arithmetic_sequence_without_square_gt1 (a d : ℕ) (h_d : d ≠ 0) :
  ¬(∀ n : ℕ, ∃ k : ℕ, k > 0 ∧ k ∈ {a + n * d | n : ℕ} ∧ ∀ m : ℕ, m > 1 → m * m ∣ k → false) := sorry

end NUMINAMATH_GPT_no_arithmetic_sequence_without_square_gt1_l822_82258


namespace NUMINAMATH_GPT_max_ages_within_two_std_dev_l822_82231

def average_age : ℕ := 30
def std_dev : ℕ := 12
def lower_limit : ℕ := average_age - 2 * std_dev
def upper_limit : ℕ := average_age + 2 * std_dev
def max_different_ages : ℕ := upper_limit - lower_limit + 1

theorem max_ages_within_two_std_dev
  (avg : ℕ) (std : ℕ) (h_avg : avg = average_age) (h_std : std = std_dev)
  : max_different_ages = 49 :=
by
  sorry

end NUMINAMATH_GPT_max_ages_within_two_std_dev_l822_82231


namespace NUMINAMATH_GPT_digit_pairs_for_divisibility_by_36_l822_82219

theorem digit_pairs_for_divisibility_by_36 (A B : ℕ) :
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧
  (∃ k4 k9 : ℕ, (10 * 5 + B = 4 * k4) ∧ (20 + A + B = 9 * k9)) ↔ 
  ((A = 5 ∧ B = 2) ∨ (A = 1 ∧ B = 6)) :=
by sorry

end NUMINAMATH_GPT_digit_pairs_for_divisibility_by_36_l822_82219


namespace NUMINAMATH_GPT_number_of_boys_in_first_group_l822_82213

-- Define the daily work ratios
variables (M B : ℝ) (h_ratio : M = 2 * B)

-- Define the number of boys in the first group
variable (x : ℝ)

-- Define the conditions provided by the problem
variables (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B))

-- State the theorem and include the correct answer
theorem number_of_boys_in_first_group (M B : ℝ) (h_ratio : M = 2 * B) (x : ℝ)
    (h1 : 5 * (12 * M + x * B) = 4 * (13 * M + 24 * B)) 
    : x = 16 := 
by 
    sorry

end NUMINAMATH_GPT_number_of_boys_in_first_group_l822_82213


namespace NUMINAMATH_GPT_morse_code_sequences_l822_82290

theorem morse_code_sequences : 
  let number_of_sequences := 
        (2 ^ 1) + (2 ^ 2) + (2 ^ 3) + (2 ^ 4) + (2 ^ 5)
  number_of_sequences = 62 :=
by
  sorry

end NUMINAMATH_GPT_morse_code_sequences_l822_82290


namespace NUMINAMATH_GPT_amount_of_solution_added_l822_82246

variable (x : ℝ)

-- Condition: The solution contains 90% alcohol
def solution_alcohol_amount (x : ℝ) : ℝ := 0.9 * x

-- Condition: Total volume of the new mixture after adding 16 liters of water
def total_volume (x : ℝ) : ℝ := x + 16

-- Condition: The percentage of alcohol in the new mixture is 54%
def new_mixture_alcohol_amount (x : ℝ) : ℝ := 0.54 * (total_volume x)

-- The proof goal: the amount of solution added is 24 liters
theorem amount_of_solution_added : new_mixture_alcohol_amount x = solution_alcohol_amount x → x = 24 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_solution_added_l822_82246


namespace NUMINAMATH_GPT_gcd_expression_l822_82297

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end NUMINAMATH_GPT_gcd_expression_l822_82297


namespace NUMINAMATH_GPT_exists_monochromatic_triangle_in_K6_l822_82204

/-- In a complete graph with 6 vertices where each edge is colored either red or blue,
    there exists a set of 3 vertices such that the edges joining them are all the same color. -/
theorem exists_monochromatic_triangle_in_K6 (color : Fin 6 → Fin 6 → Prop)
  (h : ∀ {i j : Fin 6}, i ≠ j → (color i j ∨ ¬ color i j)) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  ((color i j ∧ color j k ∧ color k i) ∨ (¬ color i j ∧ ¬ color j k ∧ ¬ color k i)) :=
by
  sorry

end NUMINAMATH_GPT_exists_monochromatic_triangle_in_K6_l822_82204


namespace NUMINAMATH_GPT_harry_total_cost_in_silver_l822_82227

def cost_of_spellbooks_in_gold := 5 * 5
def cost_of_potion_kits_in_silver := 3 * 20
def cost_of_owl_in_gold := 28
def gold_to_silver := 9

def cost_in_silver :=
  (cost_of_spellbooks_in_gold + cost_of_owl_in_gold) * gold_to_silver + cost_of_potion_kits_in_silver

theorem harry_total_cost_in_silver : cost_in_silver = 537 := by
  sorry

end NUMINAMATH_GPT_harry_total_cost_in_silver_l822_82227


namespace NUMINAMATH_GPT_final_price_of_jacket_l822_82224

noncomputable def original_price : ℝ := 240
noncomputable def initial_discount : ℝ := 0.6
noncomputable def additional_discount : ℝ := 0.25

theorem final_price_of_jacket :
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let final_price := price_after_initial_discount * (1 - additional_discount)
  final_price = 72 := 
by
  sorry

end NUMINAMATH_GPT_final_price_of_jacket_l822_82224


namespace NUMINAMATH_GPT_bracelet_ratio_l822_82269

-- Definition of the conditions
def initial_bingley_bracelets : ℕ := 5
def kelly_bracelets_given : ℕ := 16 / 4
def total_bracelets_after_receiving := initial_bingley_bracelets + kelly_bracelets_given
def bingley_remaining_bracelets : ℕ := 6
def bingley_bracelets_given := total_bracelets_after_receiving - bingley_remaining_bracelets

-- Lean 4 Statement
theorem bracelet_ratio : bingley_bracelets_given * 3 = total_bracelets_after_receiving := by
  sorry

end NUMINAMATH_GPT_bracelet_ratio_l822_82269


namespace NUMINAMATH_GPT_compute_difference_l822_82284

noncomputable def f (n : ℝ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_difference (r : ℝ) : f r - f (r - 1) = r * (r + 1) * (r + 2) := by
  sorry

end NUMINAMATH_GPT_compute_difference_l822_82284


namespace NUMINAMATH_GPT_final_sum_after_50_passes_l822_82223

theorem final_sum_after_50_passes
  (particip: ℕ) 
  (num_passes: particip = 50) 
  (init_disp: ℕ → ℤ) 
  (initial_condition : init_disp 0 = 1 ∧ init_disp 1 = 0 ∧ init_disp 2 = -1)
  (operations: Π (i : ℕ), 
    (init_disp 0 = 1 →
    init_disp 1 = 0 →
    (i % 2 = 0 → init_disp 2 = -1) →
    (i % 2 = 1 → init_disp 2 = 1))
  )
  : init_disp 0 + init_disp 1 + init_disp 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_final_sum_after_50_passes_l822_82223


namespace NUMINAMATH_GPT_range_of_a_l822_82218

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x > 1, f x = a^x) ∧ 
  (∀ x ≤ 1, f x = (4 - (a / 2)) * x + 2) → 
  4 ≤ a ∧ a < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l822_82218


namespace NUMINAMATH_GPT_hall_area_l822_82280

theorem hall_area 
  (L W : ℝ)
  (h1 : W = 1/2 * L)
  (h2 : L - W = 10) : 
  L * W = 200 := 
sorry

end NUMINAMATH_GPT_hall_area_l822_82280


namespace NUMINAMATH_GPT_table_relation_l822_82294

theorem table_relation (x y : ℕ) (hx : x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6) :
  (y = 3 ∧ x = 2) ∨ (y = 8 ∧ x = 3) ∨ (y = 15 ∧ x = 4) ∨ (y = 24 ∧ x = 5) ∨ (y = 35 ∧ x = 6) ↔ 
  y = x^2 - x + 2 :=
sorry

end NUMINAMATH_GPT_table_relation_l822_82294


namespace NUMINAMATH_GPT_find_y_l822_82200

theorem find_y (x y : ℝ) (h1 : x^2 = 2 * y - 6) (h2 : x = 7) : y = 55 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l822_82200


namespace NUMINAMATH_GPT_bananas_distribution_l822_82267

noncomputable def total_bananas : ℝ := 550.5
noncomputable def lydia_bananas : ℝ := 80.25
noncomputable def dawn_bananas : ℝ := lydia_bananas + 93
noncomputable def emily_bananas : ℝ := 198
noncomputable def donna_bananas : ℝ := emily_bananas / 2

theorem bananas_distribution :
  dawn_bananas = 173.25 ∧
  lydia_bananas = 80.25 ∧
  donna_bananas = 99 ∧
  emily_bananas = 198 ∧
  dawn_bananas + lydia_bananas + donna_bananas + emily_bananas = total_bananas :=
by
  sorry

end NUMINAMATH_GPT_bananas_distribution_l822_82267


namespace NUMINAMATH_GPT_gcd_7384_12873_l822_82259

theorem gcd_7384_12873 : Int.gcd 7384 12873 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_7384_12873_l822_82259


namespace NUMINAMATH_GPT_B_contribution_to_capital_l822_82293

theorem B_contribution_to_capital (A_capital : ℝ) (A_months : ℝ) (B_months : ℝ) (profit_ratio_A : ℝ) (profit_ratio_B : ℝ) (B_contribution : ℝ) :
  A_capital = 4500 →
  A_months = 12 →
  B_months = 5 →
  profit_ratio_A = 2 →
  profit_ratio_B = 3 →
  B_contribution = (4500 * 12 * 3) / (5 * 2) → 
  B_contribution = 16200 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_B_contribution_to_capital_l822_82293


namespace NUMINAMATH_GPT_swap_values_l822_82203

theorem swap_values : ∀ (a b : ℕ), a = 3 → b = 2 → 
  (∃ c : ℕ, c = b ∧ (b = a ∧ (a = c ∨ a = 2 ∧ b = 3))) :=
by
  sorry

end NUMINAMATH_GPT_swap_values_l822_82203


namespace NUMINAMATH_GPT_gasoline_reduction_l822_82253

theorem gasoline_reduction
  (P Q : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < Q)
  (price_increase_percent : ℝ := 0.25)
  (spending_increase_percent : ℝ := 0.05)
  (new_price : ℝ := P * (1 + price_increase_percent))
  (new_total_cost : ℝ := (P * Q) * (1 + spending_increase_percent)) :
  100 - (100 * (new_total_cost / new_price) / Q) = 16 :=
by
  sorry

end NUMINAMATH_GPT_gasoline_reduction_l822_82253
