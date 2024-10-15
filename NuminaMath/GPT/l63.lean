import Mathlib

namespace NUMINAMATH_GPT_find_second_expression_l63_6399

theorem find_second_expression (a : ℕ) (x : ℕ) (h1 : (2 * a + 16 + x) / 2 = 69) (h2 : a = 26) : x = 70 := 
by
  sorry

end NUMINAMATH_GPT_find_second_expression_l63_6399


namespace NUMINAMATH_GPT_number_of_readers_who_read_both_l63_6367

theorem number_of_readers_who_read_both (S L B total : ℕ) (hS : S = 250) (hL : L = 550) (htotal : total = 650) (h : S + L - B = total) : B = 150 :=
by {
  /-
  Given:
  S = 250 (number of readers who read science fiction)
  L = 550 (number of readers who read literary works)
  total = 650 (total number of readers)
  h : S + L - B = total (relationship between sets)
  We need to prove: B = 150
  -/
  sorry
}

end NUMINAMATH_GPT_number_of_readers_who_read_both_l63_6367


namespace NUMINAMATH_GPT_greatest_prime_factor_f24_is_11_value_of_f12_l63_6321

def is_even (n : ℕ) : Prop := n % 2 = 0

def f (n : ℕ) : ℕ := (List.range' 2 ((n + 1) / 2)).map (λ x => 2 * x) |> List.prod

theorem greatest_prime_factor_f24_is_11 : 
  ¬ ∃ p, Prime p ∧ p ∣ f 24 ∧ p > 11 := 
  sorry

theorem value_of_f12 : f 12 = 46080 := 
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_f24_is_11_value_of_f12_l63_6321


namespace NUMINAMATH_GPT_find_a6_l63_6320

-- Define the arithmetic sequence properties
variables (a : ℕ → ℤ) (d : ℤ)

-- Define the initial conditions
axiom h1 : a 4 = 1
axiom h2 : a 7 = 16
axiom h_arith_seq : ∀ n, a (n + 1) - a n = d

-- Statement to prove
theorem find_a6 : a 6 = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_a6_l63_6320


namespace NUMINAMATH_GPT_solve_inequality_l63_6309

theorem solve_inequality (x : Real) : 
  (abs ((3 * x + 2) / (x - 2)) > 3) ↔ (x ∈ Set.Ioo (2 / 3) 2) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l63_6309


namespace NUMINAMATH_GPT_neg_sqrt_comparison_l63_6397

theorem neg_sqrt_comparison : -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end NUMINAMATH_GPT_neg_sqrt_comparison_l63_6397


namespace NUMINAMATH_GPT_initial_oak_trees_l63_6348

theorem initial_oak_trees (n : ℕ) (h : n - 2 = 7) : n = 9 := 
by
  sorry

end NUMINAMATH_GPT_initial_oak_trees_l63_6348


namespace NUMINAMATH_GPT_math_test_score_l63_6324

theorem math_test_score (K E M : ℕ) 
  (h₁ : (K + E) / 2 = 92) 
  (h₂ : (K + E + M) / 3 = 94) : 
  M = 98 := 
by 
  sorry

end NUMINAMATH_GPT_math_test_score_l63_6324


namespace NUMINAMATH_GPT_three_digit_integer_divisible_by_5_l63_6357

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end NUMINAMATH_GPT_three_digit_integer_divisible_by_5_l63_6357


namespace NUMINAMATH_GPT_smallest_in_sample_l63_6331

theorem smallest_in_sample:
  ∃ (m : ℕ) (δ : ℕ), m ≥ 0 ∧ δ > 0 ∧ δ * 5 = 80 ∧ 42 = δ * (42 / δ) + m ∧ m < δ ∧ (∀ i < 5, m + i * δ < 80) → m = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_in_sample_l63_6331


namespace NUMINAMATH_GPT_find_first_term_of_arithmetic_sequence_l63_6318

theorem find_first_term_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 2)
  (h_d : d = -1/2) : a 1 = 3 :=
sorry

end NUMINAMATH_GPT_find_first_term_of_arithmetic_sequence_l63_6318


namespace NUMINAMATH_GPT_find_circle_diameter_l63_6306

noncomputable def circle_diameter (AB CD : ℝ) (h_AB : AB = 16) (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) : ℝ :=
  2 * 10

theorem find_circle_diameter (AB CD : ℝ)
  (h_AB : AB = 16)
  (h_CD : CD = 4)
  (h_perp : ∃ M : ℝ → ℝ → Prop, M AB CD) :
  circle_diameter AB CD h_AB h_CD h_perp = 20 := 
  by sorry

end NUMINAMATH_GPT_find_circle_diameter_l63_6306


namespace NUMINAMATH_GPT_complex_sum_eighth_power_l63_6365

noncomputable def compute_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : ℂ :=
  ζ1^8 + ζ2^8 + ζ3^8

theorem complex_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : 
  compute_sum_eighth_power ζ1 ζ2 ζ3 h1 h2 h3 = 451.625 :=
sorry

end NUMINAMATH_GPT_complex_sum_eighth_power_l63_6365


namespace NUMINAMATH_GPT_birds_not_herons_are_geese_l63_6316

-- Define the given conditions
def percentage_geese : ℝ := 0.35
def percentage_swans : ℝ := 0.20
def percentage_herons : ℝ := 0.15
def percentage_ducks : ℝ := 0.30

-- Definition without herons
def percentage_non_herons : ℝ := 1 - percentage_herons

-- Theorem to prove
theorem birds_not_herons_are_geese :
  (percentage_geese / percentage_non_herons) * 100 = 41 :=
by
  sorry

end NUMINAMATH_GPT_birds_not_herons_are_geese_l63_6316


namespace NUMINAMATH_GPT_units_digit_base_9_l63_6307

theorem units_digit_base_9 (a b : ℕ) (h1 : a = 3 * 9 + 5) (h2 : b = 4 * 9 + 7) : 
  ((a + b) % 9) = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_base_9_l63_6307


namespace NUMINAMATH_GPT_range_f_x_le_neg_five_l63_6322

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x then 2^x - 3 else
if h : x < 0 then 3 - 2^(-x) else 0

theorem range_f_x_le_neg_five :
  ∀ x : ℝ, f x ≤ -5 ↔ x ≤ -3 :=
by sorry

end NUMINAMATH_GPT_range_f_x_le_neg_five_l63_6322


namespace NUMINAMATH_GPT_remainder_of_large_number_l63_6387

theorem remainder_of_large_number (N : ℕ) (hN : N = 123456789012): 
  N % 360 = 108 :=
by
  have h1 : N % 4 = 0 := by 
    sorry
  have h2 : N % 9 = 3 := by 
    sorry
  have h3 : N % 10 = 2 := by
    sorry
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_l63_6387


namespace NUMINAMATH_GPT_even_function_has_a_equal_2_l63_6390

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end NUMINAMATH_GPT_even_function_has_a_equal_2_l63_6390


namespace NUMINAMATH_GPT_frustum_radius_l63_6372

theorem frustum_radius (r : ℝ) (h1 : ∃ r1 r2, r1 = r 
                                  ∧ r2 = 3 * r 
                                  ∧ r1 * 2 * π * 3 = r2 * 2 * π
                                  ∧ (lateral_area = 84 * π)) (h2 : slant_height = 3) : 
  r = 7 :=
sorry

end NUMINAMATH_GPT_frustum_radius_l63_6372


namespace NUMINAMATH_GPT_quadratic_has_one_real_solution_l63_6344

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_real_solution_l63_6344


namespace NUMINAMATH_GPT_smaller_prime_is_x_l63_6329

theorem smaller_prime_is_x (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x = 17 :=
  sorry

end NUMINAMATH_GPT_smaller_prime_is_x_l63_6329


namespace NUMINAMATH_GPT_nancy_initial_files_correct_l63_6303

-- Definitions based on the problem conditions
def initial_files (deleted_files : ℕ) (folder_count : ℕ) (files_per_folder : ℕ) : ℕ :=
  (folder_count * files_per_folder) + deleted_files

-- The proof statement
theorem nancy_initial_files_correct :
  initial_files 31 7 7 = 80 :=
by
  sorry

end NUMINAMATH_GPT_nancy_initial_files_correct_l63_6303


namespace NUMINAMATH_GPT_volume_of_resulting_shape_l63_6350

-- Define the edge lengths
def edge_length (original : ℕ) (small : ℕ) := original = 5 ∧ small = 1

-- Define the volume of a cube
def volume (a : ℕ) : ℕ := a * a * a

-- State the proof problem
theorem volume_of_resulting_shape : ∀ (original small : ℕ), edge_length original small → 
  volume original - (5 * volume small) = 120 := by
  sorry

end NUMINAMATH_GPT_volume_of_resulting_shape_l63_6350


namespace NUMINAMATH_GPT_expected_number_of_digits_l63_6395

-- Define a noncomputable expected_digits function for an icosahedral die
noncomputable def expected_digits : ℝ :=
  let p1 := 9 / 20
  let p2 := 11 / 20
  (p1 * 1) + (p2 * 2)

theorem expected_number_of_digits :
  expected_digits = 1.55 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_expected_number_of_digits_l63_6395


namespace NUMINAMATH_GPT_john_average_speed_l63_6338

theorem john_average_speed:
  (∃ J : ℝ, Carla_speed = 35 ∧ Carla_time = 3 ∧ John_time = 3.5 ∧ J * John_time = Carla_speed * Carla_time) →
  (∃ J : ℝ, J = 30) :=
by
  -- Given Variables
  let Carla_speed : ℝ := 35
  let Carla_time : ℝ := 3
  let John_time : ℝ := 3.5
  -- Proof goal
  sorry

end NUMINAMATH_GPT_john_average_speed_l63_6338


namespace NUMINAMATH_GPT_initially_collected_oranges_l63_6358

-- Define the conditions from the problem
def oranges_eaten_by_father : ℕ := 2
def oranges_mildred_has_now : ℕ := 75

-- Define the proof problem (statement)
theorem initially_collected_oranges :
  (oranges_mildred_has_now + oranges_eaten_by_father = 77) :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_initially_collected_oranges_l63_6358


namespace NUMINAMATH_GPT_problem_inequality_l63_6334

theorem problem_inequality (a b c : ℝ) (h₀ : a + b + c = 0) (d : ℝ) (h₁ : d = max (|a|) (max (|b|) (|c|))) : 
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l63_6334


namespace NUMINAMATH_GPT_square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l63_6394

-- Define the problem conditions.
def square_grid (n : Nat) : Prop := true
def rectangle_grid (m n : Nat) : Prop := true

-- Define the grid size for square and rectangle.
def square_grid_21 := square_grid 21
def rectangle_grid_20_21 := rectangle_grid 20 21

-- Define the proof problem to find maximum moves.
theorem square_grid_21_max_moves : ∃ m : Nat, m = 3 :=
  sorry

theorem rectangle_grid_20_21_max_moves : ∃ m : Nat, m = 4 :=
  sorry

end NUMINAMATH_GPT_square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l63_6394


namespace NUMINAMATH_GPT_surface_area_circumscribed_sphere_l63_6312

theorem surface_area_circumscribed_sphere (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
    4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2) / 2)^2) = 50 * Real.pi :=
by
  rw [ha, hb, hc]
  -- prove the equality step-by-step
  sorry

end NUMINAMATH_GPT_surface_area_circumscribed_sphere_l63_6312


namespace NUMINAMATH_GPT_area_enclosed_by_S_l63_6313

open Complex

def five_presentable (v : ℂ) : Prop := abs v = 5

def S : Set ℂ := {u | ∃ v : ℂ, five_presentable v ∧ u = v - (1 / v)}

theorem area_enclosed_by_S : 
  ∃ (area : ℝ), area = 624 / 25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_S_l63_6313


namespace NUMINAMATH_GPT_lewis_speed_is_90_l63_6377

noncomputable def david_speed : ℝ := 50 -- mph
noncomputable def distance_chennai_hyderabad : ℝ := 350 -- miles
noncomputable def distance_meeting_point : ℝ := 250 -- miles

theorem lewis_speed_is_90 :
  ∃ L : ℝ, 
    (∀ t : ℝ, david_speed * t = distance_meeting_point) →
    (∀ t : ℝ, L * t = (distance_chennai_hyderabad + (distance_meeting_point - distance_chennai_hyderabad))) →
    L = 90 :=
by
  sorry

end NUMINAMATH_GPT_lewis_speed_is_90_l63_6377


namespace NUMINAMATH_GPT_james_daily_soda_consumption_l63_6392

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end NUMINAMATH_GPT_james_daily_soda_consumption_l63_6392


namespace NUMINAMATH_GPT_fundraising_exceeded_goal_l63_6319

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end NUMINAMATH_GPT_fundraising_exceeded_goal_l63_6319


namespace NUMINAMATH_GPT_initial_donuts_30_l63_6398

variable (x y : ℝ)
variable (p : ℝ := 0.30)

theorem initial_donuts_30 (h1 : y = 9) (h2 : y = p * x) : x = 30 := by
  sorry

end NUMINAMATH_GPT_initial_donuts_30_l63_6398


namespace NUMINAMATH_GPT_smallest_cubes_to_fill_box_l63_6300

theorem smallest_cubes_to_fill_box
  (L W D : ℕ)
  (hL : L = 30)
  (hW : W = 48)
  (hD : D = 12) :
  ∃ (n : ℕ), n = (L * W * D) / ((Nat.gcd (Nat.gcd L W) D) ^ 3) ∧ n = 80 := 
by
  sorry

end NUMINAMATH_GPT_smallest_cubes_to_fill_box_l63_6300


namespace NUMINAMATH_GPT_hexagonal_tiles_in_box_l63_6310

theorem hexagonal_tiles_in_box :
  ∃ a b c : ℕ, a + b + c = 35 ∧ 3 * a + 4 * b + 6 * c = 128 ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_hexagonal_tiles_in_box_l63_6310


namespace NUMINAMATH_GPT_probability_reach_origin_from_3_3_l63_6366

noncomputable def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x+1, 0 => 0
| 0, y+1 => 0
| x+1, y+1 => (1/3) * P x (y+1) + (1/3) * P (x+1) y + (1/3) * P x y

theorem probability_reach_origin_from_3_3 : P 3 3 = 1 / 27 := by
  sorry

end NUMINAMATH_GPT_probability_reach_origin_from_3_3_l63_6366


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l63_6305

theorem arithmetic_sequence_third_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  (S 5 = 10) ∧ (S n = n * (a 1 + a n) / 2) ∧ (a 5 = a 1 + 4 * d) ∧ 
  (∀ n, a n = a 1 + (n-1) * d) → (a 3 = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l63_6305


namespace NUMINAMATH_GPT_fixed_point_exists_l63_6362

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(a * (x + 1)) - 3

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  -- Sorry for skipping the proof
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l63_6362


namespace NUMINAMATH_GPT_boris_stopped_saving_in_may_2020_l63_6302

theorem boris_stopped_saving_in_may_2020 :
  ∀ (B V : ℕ) (start_date_B start_date_V stop_date : ℕ), 
    (∀ t, start_date_B + t ≤ stop_date → B = 200 * t) →
    (∀ t, start_date_V + t ≤ stop_date → V = 300 * t) → 
    V = 6 * B →
    stop_date = 17 → 
    B / 200 = 4 → 
    stop_date - B/200 = 2020 * 12 + 5 :=
by
  sorry

end NUMINAMATH_GPT_boris_stopped_saving_in_may_2020_l63_6302


namespace NUMINAMATH_GPT_greatest_integer_difference_l63_6382

theorem greatest_integer_difference (x y : ℤ) (hx : -6 < (x : ℝ)) (hx2 : (x : ℝ) < -2) (hy : 4 < (y : ℝ)) (hy2 : (y : ℝ) < 10) : 
  ∃ d : ℤ, d = y - x ∧ d = 14 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_difference_l63_6382


namespace NUMINAMATH_GPT_total_cost_price_l63_6361

theorem total_cost_price (C O B : ℝ) 
    (hC : 1.25 * C = 8340) 
    (hO : 1.30 * O = 4675) 
    (hB : 1.20 * B = 3600) : 
    C + O + B = 13268.15 := 
by 
    sorry

end NUMINAMATH_GPT_total_cost_price_l63_6361


namespace NUMINAMATH_GPT_barbara_typing_time_l63_6315

theorem barbara_typing_time:
  let original_speed := 212
  let speed_decrease := 40
  let document_length := 3440
  let new_speed := original_speed - speed_decrease
  (new_speed > 0) → 
  (document_length / new_speed = 20) :=
by
  intros
  sorry

end NUMINAMATH_GPT_barbara_typing_time_l63_6315


namespace NUMINAMATH_GPT_molecular_weight_BaSO4_l63_6323

-- Definitions for atomic weights of elements.
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_S : ℝ := 32.07
def atomic_weight_O : ℝ := 16.00

-- Defining the number of atoms in BaSO4
def num_Ba : ℕ := 1
def num_S : ℕ := 1
def num_O : ℕ := 4

-- Statement to be proved
theorem molecular_weight_BaSO4 :
  (num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O) = 233.40 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_BaSO4_l63_6323


namespace NUMINAMATH_GPT_simplify_expression_l63_6337

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l63_6337


namespace NUMINAMATH_GPT_functional_relationship_max_annual_profit_l63_6355

namespace FactoryProfit

-- Definitions of conditions
def fixed_annual_investment : ℕ := 100
def unit_investment : ℕ := 1
def sales_revenue (x : ℕ) : ℕ :=
  if x > 20 then 260 
  else 33 * x - x^2

def annual_profit (x : ℕ) : ℤ :=
  let revenue := sales_revenue x
  let total_investment := fixed_annual_investment + x
  revenue - total_investment

-- Statements to prove
theorem functional_relationship (x : ℕ) (hx : x > 0) :
  annual_profit x =
  if x ≤ 20 then
    (-x^2 : ℤ) + 32 * x - 100
  else
    160 - x :=
by sorry

theorem max_annual_profit : 
  ∃ x, annual_profit x = 144 ∧
  ∀ y, annual_profit y ≤ 144 :=
by sorry

end FactoryProfit

end NUMINAMATH_GPT_functional_relationship_max_annual_profit_l63_6355


namespace NUMINAMATH_GPT_negation_of_universal_l63_6363

theorem negation_of_universal :
  ¬ (∀ x : ℝ, 2 * x ^ 2 + x - 1 ≤ 0) ↔ ∃ x : ℝ, 2 * x ^ 2 + x - 1 > 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_universal_l63_6363


namespace NUMINAMATH_GPT_sin_2y_eq_37_40_l63_6384

variable (x y : ℝ)
variable (sin cos : ℝ → ℝ)

axiom sin_def : sin x = 2 * cos y - (5/2) * sin y
axiom cos_def : cos x = 2 * sin y - (5/2) * cos y

theorem sin_2y_eq_37_40 : sin (2 * y) = 37 / 40 := by
  sorry

end NUMINAMATH_GPT_sin_2y_eq_37_40_l63_6384


namespace NUMINAMATH_GPT_pq_sum_is_38_l63_6346

theorem pq_sum_is_38
  (p q : ℝ)
  (h_root : ∀ x, (2 * x^2) + (p * x) + q = 0 → x = 2 * Complex.I - 3 ∨ x = -2 * Complex.I - 3)
  (h_p_q : ∀ a b : ℂ, a + b = -p / 2 ∧ a * b = q / 2 → p = 12 ∧ q = 26) :
  p + q = 38 :=
sorry

end NUMINAMATH_GPT_pq_sum_is_38_l63_6346


namespace NUMINAMATH_GPT_cost_to_color_pattern_l63_6347

-- Define the basic properties of the squares
def square_side_length : ℕ := 4
def number_of_squares : ℕ := 4
def unit_cost (num_overlapping_squares : ℕ) : ℕ := num_overlapping_squares

-- Define the number of unit squares overlapping by different amounts
def unit_squares_overlapping_by_4 : ℕ := 1
def unit_squares_overlapping_by_3 : ℕ := 6
def unit_squares_overlapping_by_2 : ℕ := 12
def unit_squares_overlapping_by_1 : ℕ := 18

-- Calculate the total cost
def total_cost : ℕ :=
  unit_cost 4 * unit_squares_overlapping_by_4 +
  unit_cost 3 * unit_squares_overlapping_by_3 +
  unit_cost 2 * unit_squares_overlapping_by_2 +
  unit_cost 1 * unit_squares_overlapping_by_1

-- Statement to prove
theorem cost_to_color_pattern : total_cost = 64 := 
  sorry

end NUMINAMATH_GPT_cost_to_color_pattern_l63_6347


namespace NUMINAMATH_GPT_new_rectangle_dimensions_l63_6336

theorem new_rectangle_dimensions (l w : ℕ) (h_l : l = 12) (h_w : w = 10) :
  ∃ l' w' : ℕ, l' = l ∧ w' = w / 2 ∧ l' = 12 ∧ w' = 5 :=
by
  sorry

end NUMINAMATH_GPT_new_rectangle_dimensions_l63_6336


namespace NUMINAMATH_GPT_percentage_problem_l63_6373

theorem percentage_problem (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l63_6373


namespace NUMINAMATH_GPT_x_pow_4_minus_inv_x_pow_4_eq_727_l63_6360

theorem x_pow_4_minus_inv_x_pow_4_eq_727 (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end NUMINAMATH_GPT_x_pow_4_minus_inv_x_pow_4_eq_727_l63_6360


namespace NUMINAMATH_GPT_three_legged_reptiles_count_l63_6374

noncomputable def total_heads : ℕ := 300
noncomputable def total_legs : ℕ := 798

def number_of_three_legged_reptiles (b r m : ℕ) : Prop :=
  b + r + m = total_heads ∧
  2 * b + 3 * r + 4 * m = total_legs

theorem three_legged_reptiles_count (b r m : ℕ) (h : number_of_three_legged_reptiles b r m) :
  r = 102 :=
sorry

end NUMINAMATH_GPT_three_legged_reptiles_count_l63_6374


namespace NUMINAMATH_GPT_clock_angle_7_15_l63_6327

noncomputable def hour_angle_at (hour : ℕ) (minutes : ℕ) : ℝ :=
  hour * 30 + (minutes * 0.5)

noncomputable def minute_angle_at (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def small_angle (angle1 angle2 : ℝ) : ℝ :=
  let diff := abs (angle1 - angle2)
  if diff <= 180 then diff else 360 - diff

theorem clock_angle_7_15 : small_angle (hour_angle_at 7 15) (minute_angle_at 15) = 127.5 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_7_15_l63_6327


namespace NUMINAMATH_GPT_mixed_number_sum_l63_6354

theorem mixed_number_sum : 
  (4/5 + 9 * 4/5 + 99 * 4/5 + 999 * 4/5 + 9999 * 4/5 + 1 = 11111) := by
  sorry

end NUMINAMATH_GPT_mixed_number_sum_l63_6354


namespace NUMINAMATH_GPT_manager_wage_l63_6378

variable (M D C : ℝ)

def condition1 : Prop := D = M / 2
def condition2 : Prop := C = 1.25 * D
def condition3 : Prop := C = M - 3.1875

theorem manager_wage (h1 : condition1 M D) (h2 : condition2 D C) (h3 : condition3 M C) : M = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_manager_wage_l63_6378


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l63_6349

theorem quadratic_inequality_solution :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l63_6349


namespace NUMINAMATH_GPT_college_students_count_l63_6332

theorem college_students_count (girls boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
(h_ratio : ratio_boys = 6) (h_ratio_girls : ratio_girls = 5)
(h_girls : girls = 200)
(h_boys : boys = ratio_boys * (girls / ratio_girls)) :
  boys + girls = 440 := by
  sorry

end NUMINAMATH_GPT_college_students_count_l63_6332


namespace NUMINAMATH_GPT_k_at_27_l63_6385

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem k_at_27 (k : ℝ → ℝ)
    (hk_cubic : ∀ x, ∃ a b c, k x = a * x^3 + b * x^2 + c * x)
    (hk_at_0 : k 0 = 1)
    (hk_roots : ∀ a b c, (h a = 0) → (h b = 0) → (h c = 0) → 
                 ∃ (p q r: ℝ), k (p^3) = 0 ∧ k (q^3) = 0 ∧ k (r^3) = 0) :
    k 27 = -704 :=
sorry

end NUMINAMATH_GPT_k_at_27_l63_6385


namespace NUMINAMATH_GPT_tank_full_weight_l63_6339

theorem tank_full_weight (u v m n : ℝ) (h1 : m + 3 / 4 * n = u) (h2 : m + 1 / 3 * n = v) :
  m + n = 8 / 5 * u - 3 / 5 * v :=
sorry

end NUMINAMATH_GPT_tank_full_weight_l63_6339


namespace NUMINAMATH_GPT_monochromatic_triangle_in_K17_l63_6343

theorem monochromatic_triangle_in_K17 :
  ∀ (V : Type) (E : V → V → ℕ), (∀ v1 v2, 0 ≤ E v1 v2 ∧ E v1 v2 < 3) →
    (∃ (v1 v2 v3 : V), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E v1 v2 = E v2 v3 ∧ E v2 v3 = E v1 v3)) :=
by
  intro V E Hcl
  sorry

end NUMINAMATH_GPT_monochromatic_triangle_in_K17_l63_6343


namespace NUMINAMATH_GPT_regular_decagon_interior_angle_l63_6379

-- Define the number of sides in a regular decagon
def n : ℕ := 10

-- Define the formula for the sum of the interior angles of an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the measure of one interior angle of a regular decagon
def one_interior_angle_of_regular_polygon (sum_of_angles : ℕ) (n : ℕ) : ℕ :=
  sum_of_angles / n

-- Prove that the measure of one interior angle of a regular decagon is 144 degrees
theorem regular_decagon_interior_angle : one_interior_angle_of_regular_polygon (sum_of_interior_angles 10) 10 = 144 := by
  sorry

end NUMINAMATH_GPT_regular_decagon_interior_angle_l63_6379


namespace NUMINAMATH_GPT_focus_of_parabola_proof_l63_6388

noncomputable def focus_of_parabola (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

theorem focus_of_parabola_proof (a : ℝ) (h : a ≠ 0) :
  focus_of_parabola a h = (1 / (4 * a), 0) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_proof_l63_6388


namespace NUMINAMATH_GPT_base_number_is_4_l63_6326

theorem base_number_is_4 (some_number : ℕ) (h : 16^8 = some_number^16) : some_number = 4 :=
sorry

end NUMINAMATH_GPT_base_number_is_4_l63_6326


namespace NUMINAMATH_GPT_left_handed_rock_music_lovers_l63_6314

theorem left_handed_rock_music_lovers (total_club_members left_handed_members rock_music_lovers right_handed_dislike_rock: ℕ)
  (h1 : total_club_members = 25)
  (h2 : left_handed_members = 10)
  (h3 : rock_music_lovers = 18)
  (h4 : right_handed_dislike_rock = 3)
  (h5 : total_club_members = left_handed_members + (total_club_members - left_handed_members))
  : (∃ x : ℕ, x = 6 ∧ x + (left_handed_members - x) + (rock_music_lovers - x) + right_handed_dislike_rock = total_club_members) :=
sorry

end NUMINAMATH_GPT_left_handed_rock_music_lovers_l63_6314


namespace NUMINAMATH_GPT_broken_line_count_l63_6376

def num_right_moves : ℕ := 9
def num_up_moves : ℕ := 10
def total_moves : ℕ := num_right_moves + num_up_moves
def num_broken_lines : ℕ := Nat.choose total_moves num_right_moves

theorem broken_line_count : num_broken_lines = 92378 := by
  sorry

end NUMINAMATH_GPT_broken_line_count_l63_6376


namespace NUMINAMATH_GPT_quadratic_has_real_root_l63_6353

theorem quadratic_has_real_root {b : ℝ} :
  ∃ x : ℝ, x^2 + b*x + 25 = 0 ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l63_6353


namespace NUMINAMATH_GPT_sunland_more_plates_than_moonland_l63_6352

theorem sunland_more_plates_than_moonland :
  let sunland_plates := 26^5 * 10^2
  let moonland_plates := 26^3 * 10^3
  sunland_plates - moonland_plates = 1170561600 := by
  sorry

end NUMINAMATH_GPT_sunland_more_plates_than_moonland_l63_6352


namespace NUMINAMATH_GPT_sum_of_coefficients_is_2_l63_6386

noncomputable def polynomial_expansion_condition (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :=
  (x^2 + 1) * (x - 2)^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
                          a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + a_8 * (x - 1)^8 + 
                          a_9 * (x - 1)^9 + a_10 * (x - 1)^10 + a_11 * (x - 1)^11

theorem sum_of_coefficients_is_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :
  polynomial_expansion_condition 1 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  polynomial_expansion_condition 2 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_2_l63_6386


namespace NUMINAMATH_GPT_target_runs_is_282_l63_6304

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_segment : ℝ := 10
def run_rate_remaining_20_overs : ℝ := 12.5
def overs_second_segment : ℝ := 20

-- Define the calculation of runs in the first 10 overs
def runs_first_segment : ℝ := run_rate_first_10_overs * overs_first_segment

-- Define the calculation of runs in the remaining 20 overs
def runs_second_segment : ℝ := run_rate_remaining_20_overs * overs_second_segment

-- Define the target runs
def target_runs : ℝ := runs_first_segment + runs_second_segment

-- State the theorem
theorem target_runs_is_282 : target_runs = 282 :=
by
  -- This is where the proof would go, but it is omitted.
  sorry

end NUMINAMATH_GPT_target_runs_is_282_l63_6304


namespace NUMINAMATH_GPT_laundry_loads_needed_l63_6369

theorem laundry_loads_needed
  (families : ℕ) (people_per_family : ℕ)
  (towels_per_person_per_day : ℕ) (days : ℕ)
  (washing_machine_capacity : ℕ)
  (h_f : families = 7)
  (h_p : people_per_family = 6)
  (h_t : towels_per_person_per_day = 2)
  (h_d : days = 10)
  (h_w : washing_machine_capacity = 10) : 
  ((families * people_per_family * towels_per_person_per_day * days) / washing_machine_capacity) = 84 := 
by
  sorry

end NUMINAMATH_GPT_laundry_loads_needed_l63_6369


namespace NUMINAMATH_GPT_brenda_age_l63_6371

theorem brenda_age (A B J : ℕ) (h1 : A = 3 * B) (h2 : J = B + 10) (h3 : A = J) : B = 5 :=
sorry

end NUMINAMATH_GPT_brenda_age_l63_6371


namespace NUMINAMATH_GPT_number_of_freshmen_to_sample_l63_6317

-- Define parameters
def total_students : ℕ := 900
def sample_size : ℕ := 45
def freshmen_count : ℕ := 400
def sophomores_count : ℕ := 300
def juniors_count : ℕ := 200

-- Define the stratified sampling calculation
def stratified_sampling_calculation (group_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_size

-- Theorem stating that the number of freshmen to be sampled is 20
theorem number_of_freshmen_to_sample : stratified_sampling_calculation freshmen_count total_students sample_size = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_freshmen_to_sample_l63_6317


namespace NUMINAMATH_GPT_fraction_equality_l63_6301

theorem fraction_equality (a b c : ℝ) (hc : c ≠ 0) (h : a / c = b / c) : a = b := 
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l63_6301


namespace NUMINAMATH_GPT_geo_sequence_ratio_l63_6370

theorem geo_sequence_ratio
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (q : ℝ)
  (hq1 : q = 1 → S_8 = 8 * a_n 0 ∧ S_4 = 4 * a_n 0 ∧ S_8 = 2 * S_4)
  (hq2 : q ≠ 1 → S_8 = 2 * S_4 → false)
  (hS : ∀ n, S_n n = a_n 0 * (1 - q^n) / (1 - q))
  (h_condition : S_8 = 2 * S_4) :
  a_n 2 / a_n 0 = 1 := sorry

end NUMINAMATH_GPT_geo_sequence_ratio_l63_6370


namespace NUMINAMATH_GPT_sin_cos_identity_trig_identity_l63_6335

open Real

-- Problem I
theorem sin_cos_identity (α : ℝ) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 → 
  sin α * cos α = 3 / 10 := 
sorry

-- Problem II
theorem trig_identity : 
  (sqrt (1 - 2 * sin (10 * π / 180) * cos (10 * π / 180))) / 
  (cos (10 * π / 180) - sqrt (1 - cos (170 * π / 180)^2)) = 1 := 
sorry

end NUMINAMATH_GPT_sin_cos_identity_trig_identity_l63_6335


namespace NUMINAMATH_GPT_find_HCF_l63_6333

-- Given conditions
def LCM : ℕ := 750
def product_of_two_numbers : ℕ := 18750

-- Proof statement
theorem find_HCF (h : ℕ) (hpos : h > 0) :
  (LCM * h = product_of_two_numbers) → h = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_HCF_l63_6333


namespace NUMINAMATH_GPT_original_denominator_is_18_l63_6356

variable (d : ℕ)

theorem original_denominator_is_18
  (h1 : ∃ (d : ℕ), (3 + 7) / (d + 7) = 2 / 5) :
  d = 18 := 
sorry

end NUMINAMATH_GPT_original_denominator_is_18_l63_6356


namespace NUMINAMATH_GPT_triangle_properties_l63_6380

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
{
  sorry
}

end NUMINAMATH_GPT_triangle_properties_l63_6380


namespace NUMINAMATH_GPT_calculate_square_difference_l63_6308

theorem calculate_square_difference : 2023^2 - 2022^2 = 4045 := by
  sorry

end NUMINAMATH_GPT_calculate_square_difference_l63_6308


namespace NUMINAMATH_GPT_determinant_zero_l63_6340

noncomputable def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end NUMINAMATH_GPT_determinant_zero_l63_6340


namespace NUMINAMATH_GPT_Doug_lost_marbles_l63_6311

theorem Doug_lost_marbles (D E L : ℕ) 
    (h1 : E = D + 22) 
    (h2 : E = D - L + 30) 
    : L = 8 := by
  sorry

end NUMINAMATH_GPT_Doug_lost_marbles_l63_6311


namespace NUMINAMATH_GPT_maximum_volume_pyramid_is_one_sixteenth_l63_6389

open Real  -- Opening Real namespace for real number operations

noncomputable def maximum_volume_pyramid : ℝ :=
  let a := 1 -- side length of the equilateral triangle base
  let base_area := (sqrt 3 / 4) * (a * a) -- area of the equilateral triangle with side length 1
  let median := sqrt 3 / 2 * a -- median length of the triangle
  let height := 1 / 2 * median -- height of the pyramid
  let volume := 1 / 3 * base_area * height -- volume formula for a pyramid
  volume

theorem maximum_volume_pyramid_is_one_sixteenth :
  maximum_volume_pyramid = 1 / 16 :=
by
  simp [maximum_volume_pyramid] -- Simplify the volume definition
  sorry -- Proof omitted

end NUMINAMATH_GPT_maximum_volume_pyramid_is_one_sixteenth_l63_6389


namespace NUMINAMATH_GPT_face_value_of_shares_l63_6393

/-- A company pays a 12.5% dividend to its investors. -/
def div_rate := 0.125

/-- An investor gets a 25% return on their investment. -/
def roi_rate := 0.25

/-- The investor bought the shares at Rs. 20 each. -/
def purchase_price := 20

theorem face_value_of_shares (FV : ℝ) (div_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) 
  (h1 : purchase_price * roi_rate = div_rate * FV) : FV = 40 :=
by sorry

end NUMINAMATH_GPT_face_value_of_shares_l63_6393


namespace NUMINAMATH_GPT_line_segments_property_l63_6351

theorem line_segments_property (L : List (ℝ × ℝ)) :
  L.length = 50 →
  (∃ S : List (ℝ × ℝ), S.length = 8 ∧ ∃ x : ℝ, ∀ seg ∈ S, seg.fst ≤ x ∧ x ≤ seg.snd) ∨
  (∃ T : List (ℝ × ℝ), T.length = 8 ∧ ∀ seg1 ∈ T, ∀ seg2 ∈ T, seg1 ≠ seg2 → seg1.snd < seg2.fst ∨ seg2.snd < seg1.fst) :=
by
  -- Theorem proof placeholder
  sorry

end NUMINAMATH_GPT_line_segments_property_l63_6351


namespace NUMINAMATH_GPT_avg_of_7_consecutive_integers_l63_6383

theorem avg_of_7_consecutive_integers (a b : ℕ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 5 := 
  sorry

end NUMINAMATH_GPT_avg_of_7_consecutive_integers_l63_6383


namespace NUMINAMATH_GPT_sum_fourth_powers_const_l63_6375

-- Define the vertices of the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A (a : ℝ) : Point := {x := a, y := 0}
def B (a : ℝ) : Point := {x := 0, y := a}
def C (a : ℝ) : Point := {x := -a, y := 0}
def D (a : ℝ) : Point := {x := 0, y := -a}

-- Define distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Circle centered at origin
def on_circle (P : Point) (r : ℝ) : Prop :=
  P.x ^ 2 + P.y ^ 2 = r ^ 2

-- The main theorem
theorem sum_fourth_powers_const (a r : ℝ) (P : Point) (h : on_circle P r) :
  let AP_sq := dist_sq P (A a)
  let BP_sq := dist_sq P (B a)
  let CP_sq := dist_sq P (C a)
  let DP_sq := dist_sq P (D a)
  (AP_sq ^ 2 + BP_sq ^ 2 + CP_sq ^ 2 + DP_sq ^ 2) = 4 * (r^4 + a^4 + 4 * a^2 * r^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_fourth_powers_const_l63_6375


namespace NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l63_6391

noncomputable def f (x a : ℝ) : ℝ := x^3 - x + a
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1

def condition (a : ℝ) : Prop := a^2 - a = 0

theorem condition_neither_sufficient_nor_necessary
  (a : ℝ) :
  ¬(condition a → (∀ x : ℝ, f' x ≥ 0)) ∧ ¬((∀ x : ℝ, f' x ≥ 0) → condition a) :=
by
  sorry -- Proof is omitted as per the prompt

end NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l63_6391


namespace NUMINAMATH_GPT_gcd_108_450_l63_6364

theorem gcd_108_450 : Nat.gcd 108 450 = 18 :=
by
  sorry

end NUMINAMATH_GPT_gcd_108_450_l63_6364


namespace NUMINAMATH_GPT_find_radius_of_semicircle_l63_6345

-- Definitions for the rectangle and semi-circle
variable (L W : ℝ) -- Length and width of the rectangle
variable (r : ℝ) -- Radius of the semi-circle

-- Conditions given in the problem
def rectangle_perimeter : Prop := 2 * L + 2 * W = 216
def semicircle_diameter_eq_length : Prop := L = 2 * r 
def width_eq_twice_radius : Prop := W = 2 * r

-- Proof statement
theorem find_radius_of_semicircle
  (h_perimeter : rectangle_perimeter L W)
  (h_diameter : semicircle_diameter_eq_length L r)
  (h_width : width_eq_twice_radius W r) :
  r = 27 := by
  sorry

end NUMINAMATH_GPT_find_radius_of_semicircle_l63_6345


namespace NUMINAMATH_GPT_TimPrankCombinations_l63_6342

-- Definitions of the conditions in the problem
def MondayChoices : ℕ := 3
def TuesdayChoices : ℕ := 1
def WednesdayChoices : ℕ := 6
def ThursdayChoices : ℕ := 4
def FridayChoices : ℕ := 2

-- The main theorem to prove the total combinations
theorem TimPrankCombinations : 
  MondayChoices * TuesdayChoices * WednesdayChoices * ThursdayChoices * FridayChoices = 144 := 
by
  sorry

end NUMINAMATH_GPT_TimPrankCombinations_l63_6342


namespace NUMINAMATH_GPT_part1_part2_l63_6368

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part1 (h : 1 - a = -1) : a = 2 ∧ 
                                  (∀ x : ℝ, x < Real.log 2 → (Real.exp x - 2) < 0) ∧ 
                                  (∀ x : ℝ, x > Real.log 2 → (Real.exp x - 2) > 0) :=
by
  sorry

theorem part2 (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : f 2 x1 = f 2 x2) : 
  x1 + x2 < 2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l63_6368


namespace NUMINAMATH_GPT_find_b_given_a_l63_6330

-- Definitions based on the conditions
def varies_inversely (a b : ℝ) (k : ℝ) : Prop := a * b = k
def k_value : ℝ := 400

-- The proof statement
theorem find_b_given_a (a b : ℝ) (h1 : varies_inversely 800 0.5 k_value) (h2 : a = 3200) : b = 0.125 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_find_b_given_a_l63_6330


namespace NUMINAMATH_GPT_problem1_problem2_l63_6328

theorem problem1 : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : Real.sqrt (4 / 3) / Real.sqrt (7 / 3) * Real.sqrt (7 / 5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l63_6328


namespace NUMINAMATH_GPT_total_product_l63_6341

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12 
  else if n % 2 = 0 then 4 
  else 0 

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem total_product : total_points allie_rolls * total_points betty_rolls = 1120 := sorry

end NUMINAMATH_GPT_total_product_l63_6341


namespace NUMINAMATH_GPT_words_per_minute_after_break_l63_6325

variable (w : ℕ)

theorem words_per_minute_after_break (h : 10 * 5 - (w * 5) = 10) : w = 8 := by
  sorry

end NUMINAMATH_GPT_words_per_minute_after_break_l63_6325


namespace NUMINAMATH_GPT_part1_part1_eq_part2_tangent_part3_center_range_l63_6396

-- Define the conditions
def A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4
def circle_center_condition (x : ℝ) : ℝ := -x + 5
def radius : ℝ := 1

-- Part (1)
theorem part1 (x y : ℝ) (hx : y = line_l x) (hy : y = circle_center_condition x) :
  (x = 3 ∧ y = 2) :=
sorry

theorem part1_eq :
  ∃ C : ℝ × ℝ, C = (3, 2) ∧ ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 2) ^ 2 = 1 :=
sorry

-- Part (2)
theorem part2_tangent (x y : ℝ) (hx : y = 3) (hy : 3 * x + 4 * y - 12 = 0) :
  ∀ (a b : ℝ), a = 0 ∧ b = -3 / 4 :=
sorry

-- Part (3)
theorem part3_center_range (a : ℝ) (M : ℝ × ℝ) :
  (|2 * a - 4 - 3 / 2| ≤ 1) ->
  (9 / 4 ≤ a ∧ a ≤ 13 / 4) :=
sorry

end NUMINAMATH_GPT_part1_part1_eq_part2_tangent_part3_center_range_l63_6396


namespace NUMINAMATH_GPT_pages_revised_twice_theorem_l63_6381

noncomputable def pages_revised_twice (total_pages : ℕ) (cost_per_page : ℕ) (revision_cost_per_page : ℕ) 
                                      (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  let pages_revised_twice := (total_cost - (total_pages * cost_per_page + pages_revised_once * revision_cost_per_page)) 
                             / (revision_cost_per_page * 2)
  pages_revised_twice

theorem pages_revised_twice_theorem : 
  pages_revised_twice 100 10 5 30 1350 = 20 :=
by
  unfold pages_revised_twice
  norm_num

end NUMINAMATH_GPT_pages_revised_twice_theorem_l63_6381


namespace NUMINAMATH_GPT_equal_cost_at_20_minutes_l63_6359

/-- Define the cost functions for each telephone company -/
def united_cost (m : ℝ) : ℝ := 11 + 0.25 * m
def atlantic_cost (m : ℝ) : ℝ := 12 + 0.20 * m
def global_cost (m : ℝ) : ℝ := 13 + 0.15 * m

/-- Prove that at 20 minutes, the cost is the same for all three companies -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 ∧ atlantic_cost 20 = global_cost 20 :=
by
  sorry

end NUMINAMATH_GPT_equal_cost_at_20_minutes_l63_6359
