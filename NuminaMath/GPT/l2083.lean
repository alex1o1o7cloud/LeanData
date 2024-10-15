import Mathlib

namespace NUMINAMATH_GPT_tiffany_math_homework_pages_l2083_208333

def math_problems (m : ℕ) : ℕ := 3 * m
def reading_problems : ℕ := 4 * 3
def total_problems (m : ℕ) : ℕ := math_problems m + reading_problems

theorem tiffany_math_homework_pages (m : ℕ) (h : total_problems m = 30) : m = 6 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_math_homework_pages_l2083_208333


namespace NUMINAMATH_GPT_distribute_places_l2083_208334

open Nat

theorem distribute_places (places schools : ℕ) (h_places : places = 7) (h_schools : schools = 3) : 
  ∃ n : ℕ, n = (Nat.choose (places - 1) (schools - 1)) ∧ n = 15 :=
by
  rw [h_places, h_schools]
  use 15
  , sorry

end NUMINAMATH_GPT_distribute_places_l2083_208334


namespace NUMINAMATH_GPT_find_x_l2083_208306

theorem find_x (x : ℝ) : (0.75 / x = 10 / 8) → (x = 0.6) := by
  sorry

end NUMINAMATH_GPT_find_x_l2083_208306


namespace NUMINAMATH_GPT_geom_seq_frac_l2083_208314

noncomputable def geom_seq_sum (a1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_frac (a1 q : ℕ) (hq : q > 1) (h_sum : a1 * (q ^ 3 + q ^ 6 + 1 + q + q ^ 2 + q ^ 5) = 20)
  (h_prod : a1 ^ 7 * q ^ (3 + 6) = 64) :
  geom_seq_sum a1 q 6 / geom_seq_sum a1 q 9 = 5 / 21 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_frac_l2083_208314


namespace NUMINAMATH_GPT_sum_single_digit_base_eq_21_imp_b_eq_7_l2083_208367

theorem sum_single_digit_base_eq_21_imp_b_eq_7 (b : ℕ) (h : (b - 1) * b / 2 = 2 * b + 1) : b = 7 :=
sorry

end NUMINAMATH_GPT_sum_single_digit_base_eq_21_imp_b_eq_7_l2083_208367


namespace NUMINAMATH_GPT_total_cost_of_long_distance_bill_l2083_208328

theorem total_cost_of_long_distance_bill
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (minutes_billed : ℝ := 28.08) :
  monthly_fee + cost_per_minute * minutes_billed = 12.02 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_long_distance_bill_l2083_208328


namespace NUMINAMATH_GPT_geometric_sequence_four_seven_prod_l2083_208384

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_four_seven_prod
    (a : ℕ → ℝ)
    (h_geom : is_geometric_sequence a)
    (h_roots : ∀ x, 3 * x^2 - 2 * x - 6 = 0 → (x = a 1 ∨ x = a 10)) :
  a 4 * a 7 = -2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_four_seven_prod_l2083_208384


namespace NUMINAMATH_GPT_julia_game_difference_l2083_208377

theorem julia_game_difference :
  let tag_monday := 28
  let hide_seek_monday := 15
  let tag_tuesday := 33
  let hide_seek_tuesday := 21
  let total_monday := tag_monday + hide_seek_monday
  let total_tuesday := tag_tuesday + hide_seek_tuesday
  let difference := total_tuesday - total_monday
  difference = 11 := by
  sorry

end NUMINAMATH_GPT_julia_game_difference_l2083_208377


namespace NUMINAMATH_GPT_heat_more_games_than_bulls_l2083_208341

theorem heat_more_games_than_bulls (H : ℕ) 
(h1 : 70 + H = 145) :
H - 70 = 5 :=
sorry

end NUMINAMATH_GPT_heat_more_games_than_bulls_l2083_208341


namespace NUMINAMATH_GPT_plane_equation_passing_through_point_and_parallel_l2083_208356

-- Define the point and the plane parameters
def point : ℝ × ℝ × ℝ := (2, 3, 1)
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

-- Main theorem statement
theorem plane_equation_passing_through_point_and_parallel :
  ∃ D : ℝ, plane 2 (-1) 3 D 2 3 1 ∧ plane 2 (-1) 3 D 0 0 0 :=
sorry

end NUMINAMATH_GPT_plane_equation_passing_through_point_and_parallel_l2083_208356


namespace NUMINAMATH_GPT_chord_ratio_l2083_208394

variable (XQ WQ YQ ZQ : ℝ)

theorem chord_ratio (h1 : XQ = 5) (h2 : WQ = 7) (h3 : XQ * YQ = WQ * ZQ) : YQ / ZQ = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_chord_ratio_l2083_208394


namespace NUMINAMATH_GPT_fewer_gallons_for_plants_correct_l2083_208382

-- Define the initial conditions
def initial_water : ℕ := 65
def water_per_car : ℕ := 7
def total_cars : ℕ := 2
def water_for_cars : ℕ := water_per_car * total_cars
def water_remaining_after_cars : ℕ := initial_water - water_for_cars
def water_for_plates_clothes : ℕ := 24
def water_remaining_before_plates_clothes : ℕ := water_for_plates_clothes * 2
def water_for_plants : ℕ := water_remaining_after_cars - water_remaining_before_plates_clothes

-- Define the query statement
def fewer_gallons_for_plants : Prop := water_per_car - water_for_plants = 4

-- Proof skeleton
theorem fewer_gallons_for_plants_correct : fewer_gallons_for_plants :=
by sorry

end NUMINAMATH_GPT_fewer_gallons_for_plants_correct_l2083_208382


namespace NUMINAMATH_GPT_tetrahedron_volume_from_pentagon_l2083_208343

noncomputable def volume_of_tetrahedron (side_length : ℝ) (diagonal_length : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem tetrahedron_volume_from_pentagon :
  ∀ (s : ℝ), s = 1 →
  volume_of_tetrahedron s ((1 + Real.sqrt 5) / 2) ((Real.sqrt 3) / 4) (Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)) =
  (1 + Real.sqrt 5) / 24 :=
by
  intros s hs
  rw [hs]
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_from_pentagon_l2083_208343


namespace NUMINAMATH_GPT_inequality_false_l2083_208322

variable {x y w : ℝ}

theorem inequality_false (hx : x > y) (hy : y > 0) (hw : w ≠ 0) : ¬(x^2 * w > y^2 * w) :=
by {
  sorry -- You could replace this "sorry" with a proper proof.
}

end NUMINAMATH_GPT_inequality_false_l2083_208322


namespace NUMINAMATH_GPT_six_to_2049_not_square_l2083_208325

theorem six_to_2049_not_square
  (h1: ∃ x: ℝ, 1^2048 = x^2)
  (h2: ∃ x: ℝ, 2^2050 = x^2)
  (h3: ¬∃ x: ℝ, 6^2049 = x^2)
  (h4: ∃ x: ℝ, 4^2051 = x^2)
  (h5: ∃ x: ℝ, 5^2052 = x^2):
  ¬∃ y: ℝ, y^2 = 6^2049 := 
by sorry

end NUMINAMATH_GPT_six_to_2049_not_square_l2083_208325


namespace NUMINAMATH_GPT_train_length_correct_l2083_208355

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end NUMINAMATH_GPT_train_length_correct_l2083_208355


namespace NUMINAMATH_GPT_sum_product_poly_roots_eq_l2083_208332

theorem sum_product_poly_roots_eq (b c : ℝ) 
  (h1 : -1 + 2 = -b) 
  (h2 : (-1) * 2 = c) : c + b = -3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_product_poly_roots_eq_l2083_208332


namespace NUMINAMATH_GPT_cole_round_trip_time_l2083_208397

theorem cole_round_trip_time :
  ∀ (speed_to_work speed_return : ℝ) (time_to_work_minutes : ℝ),
  speed_to_work = 75 ∧ speed_return = 105 ∧ time_to_work_minutes = 210 →
  (time_to_work_minutes / 60 + (speed_to_work * (time_to_work_minutes / 60)) / speed_return) = 6 := 
by
  sorry

end NUMINAMATH_GPT_cole_round_trip_time_l2083_208397


namespace NUMINAMATH_GPT_largest_among_a_b_c_d_l2083_208331

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_among_a_b_c_d : c = max a (max b (max c d)) := by
  sorry

end NUMINAMATH_GPT_largest_among_a_b_c_d_l2083_208331


namespace NUMINAMATH_GPT_find_n_l2083_208392

/-- Given: 
1. The second term in the expansion of (x + a)^n is binom n 1 * x^(n-1) * a = 210.
2. The third term in the expansion of (x + a)^n is binom n 2 * x^(n-2) * a^2 = 840.
3. The fourth term in the expansion of (x + a)^n is binom n 3 * x^(n-3) * a^3 = 2520.
We are to prove that n = 10. -/
theorem find_n (x a : ℕ) (n : ℕ)
  (h1 : Nat.choose n 1 * x^(n-1) * a = 210)
  (h2 : Nat.choose n 2 * x^(n-2) * a^2 = 840)
  (h3 : Nat.choose n 3 * x^(n-3) * a^3 = 2520) : 
  n = 10 := by sorry

end NUMINAMATH_GPT_find_n_l2083_208392


namespace NUMINAMATH_GPT_find_a1_in_arithmetic_sequence_l2083_208320

noncomputable def arithmetic_sequence_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem find_a1_in_arithmetic_sequence :
  ∀ (a₁ d : ℤ), d = -2 →
  (arithmetic_sequence_sum a₁ d 11 = arithmetic_sequence_sum a₁ d 10) →
  a₁ = 20 :=
by
  intro a₁ d hd hs
  sorry

end NUMINAMATH_GPT_find_a1_in_arithmetic_sequence_l2083_208320


namespace NUMINAMATH_GPT_brians_gas_usage_l2083_208301

theorem brians_gas_usage (miles_per_gallon : ℕ) (miles_traveled : ℕ) (gallons_used : ℕ) 
  (h1 : miles_per_gallon = 20) 
  (h2 : miles_traveled = 60) 
  (h3 : gallons_used = miles_traveled / miles_per_gallon) : 
  gallons_used = 3 := 
by 
  rw [h1, h2] at h3 
  exact h3

end NUMINAMATH_GPT_brians_gas_usage_l2083_208301


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l2083_208359

-- Define the condition of an equilateral triangle where each side is 7 cm
def side_length : ℕ := 7

def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the perimeter function for a triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- Statement to prove
theorem equilateral_triangle_perimeter : is_equilateral_triangle side_length side_length side_length → perimeter side_length side_length side_length = 21 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l2083_208359


namespace NUMINAMATH_GPT_minimum_value_condition_l2083_208321

theorem minimum_value_condition (x a : ℝ) (h1 : x > a) (h2 : ∀ y, y > a → x + 4 / (y - a) > 9) : a = 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_condition_l2083_208321


namespace NUMINAMATH_GPT_place_value_accuracy_l2083_208307

theorem place_value_accuracy (x : ℝ) (h : x = 3.20 * 10000) :
  ∃ p : ℕ, p = 100 ∧ (∃ k : ℤ, x / p = k) := by
  sorry

end NUMINAMATH_GPT_place_value_accuracy_l2083_208307


namespace NUMINAMATH_GPT_gabrielle_total_crates_l2083_208398

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end NUMINAMATH_GPT_gabrielle_total_crates_l2083_208398


namespace NUMINAMATH_GPT_fraction_eaten_correct_l2083_208317

def initial_nuts : Nat := 30
def nuts_left : Nat := 5
def eaten_nuts : Nat := initial_nuts - nuts_left
def fraction_eaten : Rat := eaten_nuts / initial_nuts

theorem fraction_eaten_correct : fraction_eaten = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_fraction_eaten_correct_l2083_208317


namespace NUMINAMATH_GPT_evaluate_g_at_6_l2083_208339

def g (x : ℝ) := 3 * x^4 - 19 * x^3 + 31 * x^2 - 27 * x - 72

theorem evaluate_g_at_6 : g 6 = 666 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_6_l2083_208339


namespace NUMINAMATH_GPT_ratio_of_squares_l2083_208399

noncomputable def right_triangle : Type := sorry -- Placeholder for the right triangle type

variables (a b c : ℕ)

-- Given lengths of the triangle sides
def triangle_sides (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2

-- Define x and y based on the conditions in the problem
def side_length_square_x (x : ℝ) : Prop :=
  0 < x ∧ x < 5 ∧ x < 12

def side_length_square_y (y : ℝ) : Prop :=
  0 < y ∧ y < 13

-- The main theorem to prove
theorem ratio_of_squares (x y : ℝ) :
  ∀ a b c, triangle_sides a b c →
  side_length_square_x x →
  side_length_square_y y →
  x / y = 1 :=
sorry

end NUMINAMATH_GPT_ratio_of_squares_l2083_208399


namespace NUMINAMATH_GPT_intersection_A_B_l2083_208336

def set_A : Set ℝ := {x | x > 0}
def set_B : Set ℝ := {x | x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {x | 0 < x ∧ x < 4} := sorry

end NUMINAMATH_GPT_intersection_A_B_l2083_208336


namespace NUMINAMATH_GPT_ending_number_of_SetB_l2083_208378

-- Definition of Set A
def SetA : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

-- Definition of Set B
def SetB_ends_at (n : ℕ) : Set ℕ := {i | 6 ≤ i ∧ i ≤ n}

-- The main theorem statement
theorem ending_number_of_SetB : ∃ n, SetA ∩ SetB_ends_at n = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 10 ∈ SetA ∩ SetB_ends_at n := 
sorry

end NUMINAMATH_GPT_ending_number_of_SetB_l2083_208378


namespace NUMINAMATH_GPT_John_l2083_208300

/-- Assume Grant scored 10 points higher on his math test than John.
John received a certain ratio of points as Hunter who scored 45 points on his math test.
Grant's test score was 100. -/
theorem John's_points_to_Hunter's_points_ratio 
  (Grant John Hunter : ℕ) 
  (h1 : Grant = John + 10)
  (h2 : Hunter = 45)
  (h_grant_score : Grant = 100) : 
  (John : ℚ) / (Hunter : ℚ) = 2 / 1 :=
sorry

end NUMINAMATH_GPT_John_l2083_208300


namespace NUMINAMATH_GPT_Brian_watch_animal_videos_l2083_208304

theorem Brian_watch_animal_videos :
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  total_time = 144 := by
{
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  have h1 : total_time = (4 + 8 + 24 + 36 + 72) := sorry
  exact h1
}

end NUMINAMATH_GPT_Brian_watch_animal_videos_l2083_208304


namespace NUMINAMATH_GPT_correct_option_l2083_208318

-- Define the operations as functions to be used in the Lean statement.
def optA : ℕ := 3 + 5 * 7 + 9
def optB : ℕ := 3 + 5 + 7 * 9
def optC : ℕ := 3 * 5 * 7 - 9
def optD : ℕ := 3 * 5 * 7 + 9
def optE : ℕ := 3 * 5 + 7 * 9

-- The theorem to prove that the correct option is (E).
theorem correct_option : optE = 78 ∧ optA ≠ 78 ∧ optB ≠ 78 ∧ optC ≠ 78 ∧ optD ≠ 78 := by {
  sorry
}

end NUMINAMATH_GPT_correct_option_l2083_208318


namespace NUMINAMATH_GPT_molecular_weight_of_moles_l2083_208324

-- Approximate atomic weights
def atomic_weight_N := 14.01
def atomic_weight_O := 16.00

-- Molecular weight of N2O3
def molecular_weight_N2O3 := (2 * atomic_weight_N) + (3 * atomic_weight_O)

-- Given the total molecular weight of some moles of N2O3
def total_molecular_weight : ℝ := 228

-- We aim to prove that the total molecular weight of some moles of N2O3 equals 228 g
theorem molecular_weight_of_moles (h: molecular_weight_N2O3 ≠ 0) :
  total_molecular_weight = 228 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_moles_l2083_208324


namespace NUMINAMATH_GPT_sin_alpha_beta_eq_l2083_208388

theorem sin_alpha_beta_eq 
  (α β : ℝ) 
  (h1 : π / 4 < α) (h2 : α < 3 * π / 4)
  (h3 : 0 < β) (h4 : β < π / 4)
  (h5: Real.sin (α + π / 4) = 3 / 5)
  (h6: Real.cos (π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 56 / 65 :=
sorry

end NUMINAMATH_GPT_sin_alpha_beta_eq_l2083_208388


namespace NUMINAMATH_GPT_fraction_zero_solution_l2083_208370

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end NUMINAMATH_GPT_fraction_zero_solution_l2083_208370


namespace NUMINAMATH_GPT_negate_proposition_l2083_208390

theorem negate_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) := 
sorry

end NUMINAMATH_GPT_negate_proposition_l2083_208390


namespace NUMINAMATH_GPT_initial_birds_correct_l2083_208310

def flown_away : ℝ := 8.0
def left_on_fence : ℝ := 4.0
def initial_birds : ℝ := flown_away + left_on_fence

theorem initial_birds_correct : initial_birds = 12.0 := by
  sorry

end NUMINAMATH_GPT_initial_birds_correct_l2083_208310


namespace NUMINAMATH_GPT_points_opposite_sides_l2083_208380

theorem points_opposite_sides (m : ℝ) : (-2 < m ∧ m < -1) ↔ ((2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0) := by
  sorry

end NUMINAMATH_GPT_points_opposite_sides_l2083_208380


namespace NUMINAMATH_GPT_triangle_perimeter_not_78_l2083_208346

theorem triangle_perimeter_not_78 (x : ℝ) (h1 : 11 < x) (h2 : x < 37) : 13 + 24 + x ≠ 78 :=
by
  -- Using the given conditions to show the perimeter is not 78
  intro h
  have h3 : 48 < 13 + 24 + x := by linarith
  have h4 : 13 + 24 + x < 74 := by linarith
  linarith

end NUMINAMATH_GPT_triangle_perimeter_not_78_l2083_208346


namespace NUMINAMATH_GPT_value_of_frac_l2083_208315

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_value_of_frac_l2083_208315


namespace NUMINAMATH_GPT_expression_evaluation_l2083_208364

theorem expression_evaluation (a b c d : ℝ) 
  (h₁ : a + b = 0) 
  (h₂ : c * d = 1) : 
  (a + b)^2 - 3 * (c * d)^4 = -3 := 
by
  -- Proof steps are omitted, as only the statement is required.
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2083_208364


namespace NUMINAMATH_GPT_evaluate_f_g_f_l2083_208347

-- Define f(x)
def f (x : ℝ) : ℝ := 4 * x + 4

-- Define g(x)
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem we're proving
theorem evaluate_f_g_f : f (g (f 3)) = 1360 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_g_f_l2083_208347


namespace NUMINAMATH_GPT_complement_of_intersection_l2083_208348

theorem complement_of_intersection (U M N : Set ℤ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 4}) (hN : N = {3, 4, 5}) :
   U \ (M ∩ N) = {1, 2, 3, 5} := by
   sorry

end NUMINAMATH_GPT_complement_of_intersection_l2083_208348


namespace NUMINAMATH_GPT_sum_of_exponents_l2083_208312

-- Definition of Like Terms
def like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 2

-- Theorem statement
theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_exponents_l2083_208312


namespace NUMINAMATH_GPT_number_of_plastic_bottles_l2083_208386

-- Define the weights of glass and plastic bottles
variables (G P : ℕ)

-- Define the number of plastic bottles in the second scenario
variable (x : ℕ)

-- Define the conditions
def condition_1 := 3 * G = 600
def condition_2 := G = P + 150
def condition_3 := 4 * G + x * P = 1050

-- Proof that x is equal to 5 given the conditions
theorem number_of_plastic_bottles (h1 : condition_1 G) (h2 : condition_2 G P) (h3 : condition_3 G P x) : x = 5 :=
sorry

end NUMINAMATH_GPT_number_of_plastic_bottles_l2083_208386


namespace NUMINAMATH_GPT_couch_cost_l2083_208337

theorem couch_cost
  (C : ℕ)  -- Cost of the couch
  (table_cost : ℕ := 100)
  (lamp_cost : ℕ := 50)
  (amount_paid : ℕ := 500)
  (amount_owed : ℕ := 400)
  (total_furniture_cost : ℕ := C + table_cost + lamp_cost)
  (remaining_amount_owed : total_furniture_cost - amount_paid = amount_owed) :
   C = 750 := 
sorry

end NUMINAMATH_GPT_couch_cost_l2083_208337


namespace NUMINAMATH_GPT_negation_of_exists_l2083_208361

theorem negation_of_exists : (¬ ∃ x_0 : ℝ, x_0 < 0 ∧ x_0^2 > 0) ↔ ∀ x : ℝ, x < 0 → x^2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_exists_l2083_208361


namespace NUMINAMATH_GPT_distance_between_trees_correct_l2083_208308

-- Define the given conditions
def yard_length : ℕ := 300
def tree_count : ℕ := 26
def interval_count : ℕ := tree_count - 1

-- Define the target distance between two consecutive trees
def target_distance : ℕ := 12

-- Prove that the distance between two consecutive trees is correct
theorem distance_between_trees_correct :
  yard_length / interval_count = target_distance := 
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_correct_l2083_208308


namespace NUMINAMATH_GPT_circle_equation_l2083_208395

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_l2083_208395


namespace NUMINAMATH_GPT_blaine_fish_caught_l2083_208360

theorem blaine_fish_caught (B : ℕ) (cond1 : B + 2 * B = 15) : B = 5 := by 
  sorry

end NUMINAMATH_GPT_blaine_fish_caught_l2083_208360


namespace NUMINAMATH_GPT_logan_list_count_l2083_208303

theorem logan_list_count : 
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    ∃ n, n = 871 ∧ 
        ∀ k, (k * 30 ≥ smallest_square_multiple ∧ k * 30 ≤ smallest_cube_multiple) ↔ (30 ≤ k ∧ k ≤ 900) :=
by
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    use 871
    sorry

end NUMINAMATH_GPT_logan_list_count_l2083_208303


namespace NUMINAMATH_GPT_right_triangle_no_k_values_l2083_208330

theorem right_triangle_no_k_values (k : ℕ) (h : k > 0) : 
  ¬ (∃ k, k > 0 ∧ ((17 > k ∧ 17^2 = 13^2 + k^2) ∨ (k > 17 ∧ k < 30 ∧ k^2 = 13^2 + 17^2))) :=
sorry

end NUMINAMATH_GPT_right_triangle_no_k_values_l2083_208330


namespace NUMINAMATH_GPT_least_positive_integer_reducible_fraction_l2083_208354

theorem least_positive_integer_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (∃ d : ℕ, d > 1 ∧ d ∣ (m - 10) ∧ d ∣ (9 * m + 11)) ↔ m ≥ n) ∧ n = 111 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_reducible_fraction_l2083_208354


namespace NUMINAMATH_GPT_tangent_value_range_l2083_208350

theorem tangent_value_range : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (π / 4) → 0 ≤ (Real.tan x) ∧ (Real.tan x) ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_value_range_l2083_208350


namespace NUMINAMATH_GPT_find_custom_operation_value_l2083_208372

noncomputable def custom_operation (a b : ℤ) : ℚ := (1 : ℚ)/a + (1 : ℚ)/b

theorem find_custom_operation_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) :
  custom_operation a b = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_find_custom_operation_value_l2083_208372


namespace NUMINAMATH_GPT_problem1_problem2_l2083_208396

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2083_208396


namespace NUMINAMATH_GPT_find_middle_part_value_l2083_208374

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end NUMINAMATH_GPT_find_middle_part_value_l2083_208374


namespace NUMINAMATH_GPT_employee_percentage_six_years_or_more_l2083_208335

theorem employee_percentage_six_years_or_more
  (x : ℕ)
  (total_employees : ℕ := 36 * x)
  (employees_6_or_more : ℕ := 8 * x) :
  (employees_6_or_more : ℚ) / (total_employees : ℚ) * 100 = 22.22 := 
sorry

end NUMINAMATH_GPT_employee_percentage_six_years_or_more_l2083_208335


namespace NUMINAMATH_GPT_boy_real_name_is_kolya_l2083_208366

variable (days_answers : Fin 6 → String)
variable (lies_on : Fin 6 → Bool)
variable (truth_days : List (Fin 6))

-- Define the conditions
def condition_truth_days : List (Fin 6) := [0, 1] -- Suppose Thursday is 0, Friday is 1.
def condition_lies_on (d : Fin 6) : Bool := d = 2 -- Suppose Tuesday is 2.

-- The sequence of answers
def condition_days_answers : Fin 6 → String := 
  fun d => match d with
    | 0 => "Kolya"
    | 1 => "Petya"
    | 2 => "Kolya"
    | 3 => "Petya"
    | 4 => "Vasya"
    | 5 => "Petya"
    | _ => "Unknown"

-- The proof problem statement
theorem boy_real_name_is_kolya : 
  ∀ (d : Fin 6), 
  (d ∈ condition_truth_days → condition_days_answers d = "Kolya") ∧
  (condition_lies_on d → condition_days_answers d ≠ "Vasya") ∧ 
  (¬(d ∈ condition_truth_days ∨ condition_lies_on d) → True) →
  "Kolya" = "Kolya" :=
by
  sorry

end NUMINAMATH_GPT_boy_real_name_is_kolya_l2083_208366


namespace NUMINAMATH_GPT_wrapping_paper_cost_l2083_208305
noncomputable def cost_per_roll (shirt_boxes XL_boxes: ℕ) (cost_total: ℝ) : ℝ :=
  let rolls_for_shirts := shirt_boxes / 5
  let rolls_for_xls := XL_boxes / 3
  let total_rolls := rolls_for_shirts + rolls_for_xls
  cost_total / total_rolls

theorem wrapping_paper_cost : cost_per_roll 20 12 32 = 4 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_cost_l2083_208305


namespace NUMINAMATH_GPT_age_twice_in_Y_years_l2083_208369

def present_age_of_son : ℕ := 24
def age_difference := 26
def present_age_of_man : ℕ := present_age_of_son + age_difference

theorem age_twice_in_Y_years : 
  ∃ (Y : ℕ), present_age_of_man + Y = 2 * (present_age_of_son + Y) → Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_twice_in_Y_years_l2083_208369


namespace NUMINAMATH_GPT_fourth_power_mod_7_is_0_l2083_208363

def fourth_smallest_prime := 7
def square_of_fourth_smallest_prime := fourth_smallest_prime ^ 2
def fourth_power_of_square := square_of_fourth_smallest_prime ^ 4

theorem fourth_power_mod_7_is_0 : 
  (fourth_power_of_square % 7) = 0 :=
by sorry

end NUMINAMATH_GPT_fourth_power_mod_7_is_0_l2083_208363


namespace NUMINAMATH_GPT_percentage_of_boys_l2083_208376

theorem percentage_of_boys (total_students boys girls : ℕ) (h_ratio : boys * 4 = girls * 3) (h_total : boys + girls = total_students) (h_total_students : total_students = 42) : (boys : ℚ) * 100 / total_students = 42.857 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_l2083_208376


namespace NUMINAMATH_GPT_other_root_l2083_208351

/-- Given the quadratic equation x^2 - 3x + k = 0 has one root as 1, 
    prove that the other root is 2. -/
theorem other_root (k : ℝ) (h : 1^2 - 3 * 1 + k = 0) : 
  2^2 - 3 * 2 + k = 0 := 
by 
  sorry

end NUMINAMATH_GPT_other_root_l2083_208351


namespace NUMINAMATH_GPT_pencils_multiple_of_10_l2083_208326

theorem pencils_multiple_of_10 (pens : ℕ) (students : ℕ) (pencils : ℕ) 
  (h_pens : pens = 1230) 
  (h_students : students = 10) 
  (h_max_distribute : ∀ s, s ≤ students → (∃ pens_per_student, pens = pens_per_student * s ∧ ∃ pencils_per_student, pencils = pencils_per_student * s)) :
  ∃ n, pencils = 10 * n :=
by
  sorry

end NUMINAMATH_GPT_pencils_multiple_of_10_l2083_208326


namespace NUMINAMATH_GPT_quadratic_roots_condition_l2083_208391

theorem quadratic_roots_condition (m : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1^2 - 3 * x1 + 2 * m = 0 ∧ x2^2 - 3 * x2 + 2 * m = 0) →
  m < 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l2083_208391


namespace NUMINAMATH_GPT_greatest_triangle_perimeter_l2083_208387

theorem greatest_triangle_perimeter :
  ∃ (x : ℕ), 3 < x ∧ x < 6 ∧ max (x + 4 * x + 17) (5 + 4 * 5 + 17) = 42 :=
by
  sorry

end NUMINAMATH_GPT_greatest_triangle_perimeter_l2083_208387


namespace NUMINAMATH_GPT_probability_correct_l2083_208313

-- Define the total number of bulbs, good quality bulbs, and inferior quality bulbs
def total_bulbs : ℕ := 6
def good_bulbs : ℕ := 4
def inferior_bulbs : ℕ := 2

-- Define the probability of drawing one good bulb and one inferior bulb with replacement
def probability_one_good_one_inferior : ℚ := (good_bulbs * inferior_bulbs * 2) / (total_bulbs ^ 2)

-- Theorem stating that the probability of drawing one good bulb and one inferior bulb is 4/9
theorem probability_correct : probability_one_good_one_inferior = 4 / 9 := 
by
  -- Proof is skipped here
  sorry

end NUMINAMATH_GPT_probability_correct_l2083_208313


namespace NUMINAMATH_GPT_largest_integral_x_l2083_208389

theorem largest_integral_x (x : ℤ) : 
  (1 / 4 : ℝ) < (x / 7) ∧ (x / 7) < (7 / 11 : ℝ) → x ≤ 4 := 
  sorry

end NUMINAMATH_GPT_largest_integral_x_l2083_208389


namespace NUMINAMATH_GPT_ara_final_height_is_59_l2083_208381

noncomputable def initial_shea_height : ℝ := 51.2
noncomputable def initial_ara_height : ℝ := initial_shea_height + 4
noncomputable def final_shea_height : ℝ := 64
noncomputable def shea_growth : ℝ := final_shea_height - initial_shea_height
noncomputable def ara_growth : ℝ := shea_growth / 3
noncomputable def final_ara_height : ℝ := initial_ara_height + ara_growth

theorem ara_final_height_is_59 :
  final_ara_height = 59 := by
  sorry

end NUMINAMATH_GPT_ara_final_height_is_59_l2083_208381


namespace NUMINAMATH_GPT_weights_equal_weights_equal_ints_weights_equal_rationals_l2083_208302

theorem weights_equal (w : Fin 13 → ℝ) (swap_n_weighs_balance : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℝ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_ints (w : Fin 13 → ℤ) (swap_n_weighs_balance_ints : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℤ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_rationals (w : Fin 13 → ℚ) (swap_n_weighs_balance_rationals : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℚ), ∀ (i : Fin 13), w i = m :=
by
  sorry

end NUMINAMATH_GPT_weights_equal_weights_equal_ints_weights_equal_rationals_l2083_208302


namespace NUMINAMATH_GPT_train_stoppage_time_l2083_208362

theorem train_stoppage_time
  (D : ℝ) -- Distance in kilometers
  (T_no_stop : ℝ := D / 300) -- Time without stoppages in hours
  (T_with_stop : ℝ := D / 200) -- Time with stoppages in hours
  (T_stop : ℝ := T_with_stop - T_no_stop) -- Time lost due to stoppages in hours
  (T_stop_minutes : ℝ := T_stop * 60) -- Time lost due to stoppages in minutes
  (stoppage_per_hour : ℝ := T_stop_minutes / (D / 300)) -- Time stopped per hour of travel
  : stoppage_per_hour = 30 := sorry

end NUMINAMATH_GPT_train_stoppage_time_l2083_208362


namespace NUMINAMATH_GPT_people_got_off_at_first_stop_l2083_208383

theorem people_got_off_at_first_stop 
  (X : ℕ)
  (h1 : 50 - X - 6 - 1 = 28) :
  X = 15 :=
by
  sorry

end NUMINAMATH_GPT_people_got_off_at_first_stop_l2083_208383


namespace NUMINAMATH_GPT_slope_angle_of_line_l2083_208338

theorem slope_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 4 * x + y - 1 = 0 ↔ y = m * x + 1) ∧ (m = -4) → 
  θ = Real.pi - Real.arctan 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_of_line_l2083_208338


namespace NUMINAMATH_GPT_tony_water_intake_l2083_208357

-- Define the constants and conditions
def water_yesterday : ℝ := 48
def percentage_less_yesterday : ℝ := 0.04
def percentage_more_day_before_yesterday : ℝ := 0.05

-- Define the key quantity to find
noncomputable def water_two_days_ago : ℝ := water_yesterday / (1.05 * (1 - percentage_less_yesterday))

-- The proof statement
theorem tony_water_intake :
  water_two_days_ago = 47.62 :=
by
  sorry

end NUMINAMATH_GPT_tony_water_intake_l2083_208357


namespace NUMINAMATH_GPT_gcd_175_100_65_l2083_208349

theorem gcd_175_100_65 : Nat.gcd (Nat.gcd 175 100) 65 = 5 :=
by
  sorry

end NUMINAMATH_GPT_gcd_175_100_65_l2083_208349


namespace NUMINAMATH_GPT_sally_spent_total_l2083_208393

section SallySpending

def peaches : ℝ := 12.32
def cherries : ℝ := 11.54
def total_spent : ℝ := peaches + cherries

theorem sally_spent_total :
  total_spent = 23.86 := by
  sorry

end SallySpending

end NUMINAMATH_GPT_sally_spent_total_l2083_208393


namespace NUMINAMATH_GPT_bathroom_square_footage_l2083_208327

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end NUMINAMATH_GPT_bathroom_square_footage_l2083_208327


namespace NUMINAMATH_GPT_equation_has_two_distinct_roots_l2083_208371

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end NUMINAMATH_GPT_equation_has_two_distinct_roots_l2083_208371


namespace NUMINAMATH_GPT_nine_fact_div_four_fact_eq_15120_l2083_208323

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end NUMINAMATH_GPT_nine_fact_div_four_fact_eq_15120_l2083_208323


namespace NUMINAMATH_GPT_cubic_identity_l2083_208319

theorem cubic_identity (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : xy + xz + yz = 30) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 648 :=
sorry

end NUMINAMATH_GPT_cubic_identity_l2083_208319


namespace NUMINAMATH_GPT_population_at_seven_years_l2083_208316

theorem population_at_seven_years (a x : ℕ) (y: ℝ) (h₀: a = 100) (h₁: x = 7) (h₂: y = a * Real.logb 2 (x + 1)):
  y = 300 :=
by
  -- We include the conditions in the theorem statement
  sorry

end NUMINAMATH_GPT_population_at_seven_years_l2083_208316


namespace NUMINAMATH_GPT_evaluate_256_pow_5_div_8_l2083_208379

theorem evaluate_256_pow_5_div_8 (h : 256 = 2^8) : 256^(5/8) = 32 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_256_pow_5_div_8_l2083_208379


namespace NUMINAMATH_GPT_no_prime_solutions_for_x2_plus_y3_eq_z4_l2083_208311

theorem no_prime_solutions_for_x2_plus_y3_eq_z4 :
  ¬ ∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := sorry

end NUMINAMATH_GPT_no_prime_solutions_for_x2_plus_y3_eq_z4_l2083_208311


namespace NUMINAMATH_GPT_fourth_intersection_point_exists_l2083_208345

noncomputable def find_fourth_intersection_point : Prop :=
  let points := [(4, 1/2), (-6, -1/3), (1/4, 8), (-2/3, -3)]
  ∃ (h k r : ℝ), 
  ∀ (x y : ℝ), (x, y) ∈ points → (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem fourth_intersection_point_exists :
  find_fourth_intersection_point :=
by
  sorry

end NUMINAMATH_GPT_fourth_intersection_point_exists_l2083_208345


namespace NUMINAMATH_GPT_proof_problem_l2083_208340

-- Definition for the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def probability (b : ℕ) : ℚ :=
  (binom (40 - b) 2 + binom (b - 1) 2 : ℚ) / 1225

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def minimum_b (b : ℕ) : Prop :=
  b = 11 ∧ probability 11 = 857 / 1225 ∧ is_coprime 857 1225 ∧ 857 + 1225 = 2082

-- Statement to prove
theorem proof_problem : ∃ b, minimum_b b := 
by
  -- Lean statement goes here
  sorry

end NUMINAMATH_GPT_proof_problem_l2083_208340


namespace NUMINAMATH_GPT_packs_sold_to_uncle_is_correct_l2083_208309

-- Define the conditions and constants
def total_packs_needed := 50
def packs_sold_to_grandmother := 12
def packs_sold_to_neighbor := 5
def packs_left_to_sell := 26

-- Calculate total packs sold so far
def total_packs_sold := total_packs_needed - packs_left_to_sell

-- Calculate total packs sold to grandmother and neighbor
def packs_sold_to_grandmother_and_neighbor := packs_sold_to_grandmother + packs_sold_to_neighbor

-- The pack sold to uncle
def packs_sold_to_uncle := total_packs_sold - packs_sold_to_grandmother_and_neighbor

-- Prove the packs sold to uncle
theorem packs_sold_to_uncle_is_correct : packs_sold_to_uncle = 7 := by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_packs_sold_to_uncle_is_correct_l2083_208309


namespace NUMINAMATH_GPT_simplify_polynomial_l2083_208368

theorem simplify_polynomial (y : ℝ) :
    (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) =
    2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2083_208368


namespace NUMINAMATH_GPT_inequality_proof_l2083_208385

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hxyz : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2083_208385


namespace NUMINAMATH_GPT_exists_smallest_positive_period_even_function_l2083_208329

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

noncomputable def functions : List (ℝ → ℝ) :=
  [
    (λ x => Real.sin (2 * x + Real.pi / 2)),
    (λ x => Real.cos (2 * x + Real.pi / 2)),
    (λ x => Real.sin (2 * x) + Real.cos (2 * x)),
    (λ x => Real.sin x + Real.cos x)
  ]

def smallest_positive_period_even_function : ℝ → Prop :=
  λ T => ∃ f ∈ functions, is_even_function f ∧ period f T ∧ T > 0

theorem exists_smallest_positive_period_even_function :
  smallest_positive_period_even_function Real.pi :=
sorry

end NUMINAMATH_GPT_exists_smallest_positive_period_even_function_l2083_208329


namespace NUMINAMATH_GPT_divisible_by_3_l2083_208375

theorem divisible_by_3 (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end NUMINAMATH_GPT_divisible_by_3_l2083_208375


namespace NUMINAMATH_GPT_coffee_ounces_per_cup_l2083_208358

theorem coffee_ounces_per_cup
  (persons : ℕ)
  (cups_per_person_per_day : ℕ)
  (cost_per_ounce : ℝ)
  (total_spent_per_week : ℝ)
  (total_cups_per_day : ℕ)
  (total_cups_per_week : ℕ)
  (total_ounces : ℝ)
  (ounces_per_cup : ℝ) :
  persons = 4 →
  cups_per_person_per_day = 2 →
  cost_per_ounce = 1.25 →
  total_spent_per_week = 35 →
  total_cups_per_day = persons * cups_per_person_per_day →
  total_cups_per_week = total_cups_per_day * 7 →
  total_ounces = total_spent_per_week / cost_per_ounce →
  ounces_per_cup = total_ounces / total_cups_per_week →
  ounces_per_cup = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_coffee_ounces_per_cup_l2083_208358


namespace NUMINAMATH_GPT_solve_abs_inequality_l2083_208342

theorem solve_abs_inequality (x : ℝ) : 
    (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ( -4 ≤ x ∧ x ≤ -1 ∨ 3 ≤ x ∧ x ≤ 6) := 
by
    sorry

end NUMINAMATH_GPT_solve_abs_inequality_l2083_208342


namespace NUMINAMATH_GPT_buns_per_pack_is_eight_l2083_208365

-- Declaring the conditions
def burgers_per_guest : ℕ := 3
def total_friends : ℕ := 10
def friends_no_meat : ℕ := 1
def friends_no_bread : ℕ := 1
def packs_of_buns : ℕ := 3

-- Derived values from the conditions
def effective_friends_for_burgers : ℕ := total_friends - friends_no_meat
def effective_friends_for_buns : ℕ := total_friends - friends_no_bread

-- Final computation to prove
def buns_per_pack : ℕ := 24 / packs_of_buns

-- Theorem statement
theorem buns_per_pack_is_eight : buns_per_pack = 8 := by
  -- use sorry as we are not providing the proof steps 
  sorry

end NUMINAMATH_GPT_buns_per_pack_is_eight_l2083_208365


namespace NUMINAMATH_GPT_john_additional_tax_l2083_208373

-- Define the old and new tax rates
def old_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 20 / 100
  else if income ≤ 1000000 then 100000 + (income - 500000) * 25 / 100
  else 225000 + (income - 1000000) * 30 / 100

def new_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 30 / 100
  else if income ≤ 1000000 then 150000 + (income - 500000) * 35 / 100
  else 325000 + (income - 1000000) * 40 / 100

-- Calculate the tax for rental income after deduction
def rental_income_tax (rental_income : ℕ) : ℕ :=
  let taxable_rental_income := rental_income - rental_income * 10 / 100
  taxable_rental_income * 40 / 100

-- Calculate the tax for investment income
def investment_income_tax (investment_income : ℕ) : ℕ :=
  investment_income * 25 / 100

-- Calculate the tax for self-employment income
def self_employment_income_tax (self_employment_income : ℕ) : ℕ :=
  self_employment_income * 15 / 100

-- Define the total additional tax John pays
def additional_tax_paid (old_main_income new_main_income rental_income investment_income self_employment_income : ℕ) : ℕ :=
  let old_tax_main := old_tax old_main_income
  let new_tax_main := new_tax new_main_income
  let rental_tax := rental_income_tax rental_income
  let investment_tax := investment_income_tax investment_income
  let self_employment_tax := self_employment_income_tax self_employment_income
  (new_tax_main - old_tax_main) + rental_tax + investment_tax + self_employment_tax

-- Prove John pays $352,250 more in taxes under the new system
theorem john_additional_tax (main_income_old main_income_new rental_income investment_income self_employment_income : ℕ) :
  main_income_old = 1000000 →
  main_income_new = 1500000 →
  rental_income = 100000 →
  investment_income = 50000 →
  self_employment_income = 25000 →
  additional_tax_paid main_income_old main_income_new rental_income investment_income self_employment_income = 352250 :=
by
  intros h_old h_new h_rental h_invest h_self
  rw [h_old, h_new, h_rental, h_invest, h_self]
  -- calculation steps are omitted
  sorry

end NUMINAMATH_GPT_john_additional_tax_l2083_208373


namespace NUMINAMATH_GPT_arithmetic_problem_l2083_208353

theorem arithmetic_problem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l2083_208353


namespace NUMINAMATH_GPT_largest_k_inequality_l2083_208344

theorem largest_k_inequality
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_pos : (a + b) * (b + c) * (c + a) > 0) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a ≥ 
  (1 / 2) * abs ((a^3 - b^3) / (a + b) + (b^3 - c^3) / (b + c) + (c^3 - a^3) / (c + a)) :=
by
  sorry

end NUMINAMATH_GPT_largest_k_inequality_l2083_208344


namespace NUMINAMATH_GPT_area_of_rectangle_l2083_208352

theorem area_of_rectangle (M N P Q R S X Y : Type) 
  (PQ : ℝ) (PX XY YQ : ℝ) (R_perpendicular_to_PQ S_perpendicular_to_PQ : Prop) 
  (R_through_M S_through_Q : Prop) 
  (segment_lengths : PQ = PX + XY + YQ) : PQ = 5 ∧ PX = 1 ∧ XY = 2 ∧ YQ = 2 
  → 2 * (1/2 * PQ * 2) = 10 :=
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2083_208352
