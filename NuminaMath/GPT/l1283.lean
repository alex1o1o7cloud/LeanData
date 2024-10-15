import Mathlib

namespace NUMINAMATH_GPT_find_n_with_divisors_conditions_l1283_128329

theorem find_n_with_divisors_conditions :
  ∃ n : ℕ, 
    (∀ d : ℕ, d ∣ n → d ∈ [1, n] ∧ 
    (∃ a b c : ℕ, a = 1 ∧ b = d / a ∧ c = d / b ∧ b = 7 * a ∧ d = 10 + b)) →
    n = 2891 :=
by
  sorry

end NUMINAMATH_GPT_find_n_with_divisors_conditions_l1283_128329


namespace NUMINAMATH_GPT_flight_duration_l1283_128392

theorem flight_duration (takeoff landing : Nat)
  (h m : Nat) (h_pos : 0 < m) (m_lt_60 : m < 60)
  (time_takeoff : takeoff = 9 * 60 + 27)
  (time_landing : landing = 11 * 60 + 56)
  (flight_duration : (landing - takeoff) = h * 60 + m) :
  h + m = 31 :=
sorry

end NUMINAMATH_GPT_flight_duration_l1283_128392


namespace NUMINAMATH_GPT_find_a_b_find_solution_set_l1283_128385

-- Conditions
variable {a b c x : ℝ}

-- Given inequality condition
def given_inequality (x : ℝ) (a b : ℝ) : Prop := a * x^2 + x + b > 0

-- Define the solution set
def solution_set (x : ℝ) (a b : ℝ) : Prop :=
  (x < -2 ∨ x > 1) ↔ given_inequality x a b

-- Part I: Prove values of a and b
theorem find_a_b
  (H : ∀ x, solution_set x a b) :
  a = 1 ∧ b = -2 := by sorry

-- Define the second inequality
def second_inequality (x : ℝ) (c : ℝ) : Prop := x^2 - (c - 2) * x - 2 * c < 0

-- Solution set for the second inequality
def second_solution_set (x : ℝ) (c : ℝ) : Prop :=
  (c = -2 → False) ∧
  (c > -2 → -2 < x ∧ x < c) ∧
  (c < -2 → c < x ∧ x < -2)

-- Part II: Prove the solution set
theorem find_solution_set
  (H : a = 1)
  (H1 : b = -2) :
  ∀ x, second_solution_set x c ↔ second_inequality x c := by sorry

end NUMINAMATH_GPT_find_a_b_find_solution_set_l1283_128385


namespace NUMINAMATH_GPT_exponential_function_value_l1283_128301

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_function_value :
  f (f 2) = 16 := by
  simp only [f]
  sorry

end NUMINAMATH_GPT_exponential_function_value_l1283_128301


namespace NUMINAMATH_GPT_heather_bicycled_distance_l1283_128355

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end NUMINAMATH_GPT_heather_bicycled_distance_l1283_128355


namespace NUMINAMATH_GPT_find_E_l1283_128358

theorem find_E (A H S M E : ℕ) (h1 : A ≠ 0) (h2 : H ≠ 0) (h3 : S ≠ 0) (h4 : M ≠ 0) (h5 : E ≠ 0) 
  (cond1 : A + H = E)
  (cond2 : S + M = E)
  (cond3 : E = (A * M - S * H) / (M - H)) : 
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end NUMINAMATH_GPT_find_E_l1283_128358


namespace NUMINAMATH_GPT_find_width_of_brick_l1283_128322

theorem find_width_of_brick (l h : ℝ) (SurfaceArea : ℝ) (w : ℝ) :
  l = 8 → h = 2 → SurfaceArea = 152 → 2*l*w + 2*l*h + 2*w*h = SurfaceArea → w = 6 :=
by
  intro l_value
  intro h_value
  intro SurfaceArea_value
  intro surface_area_equation
  sorry

end NUMINAMATH_GPT_find_width_of_brick_l1283_128322


namespace NUMINAMATH_GPT_find_m_l1283_128323

theorem find_m (m : ℤ) (a := (3, m)) (b := (1, -2)) (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1283_128323


namespace NUMINAMATH_GPT_perimeter_is_36_l1283_128369

-- Define an equilateral triangle with a given side length
def equilateral_triangle_perimeter (side_length : ℝ) : ℝ :=
  3 * side_length

-- Given: The base of the equilateral triangle is 12 m
def base_length : ℝ := 12

-- Theorem: The perimeter of the equilateral triangle is 36 m
theorem perimeter_is_36 : equilateral_triangle_perimeter base_length = 36 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_perimeter_is_36_l1283_128369


namespace NUMINAMATH_GPT_mowing_lawn_time_l1283_128391

theorem mowing_lawn_time (mary_time tom_time tom_solo_work : ℝ) 
  (mary_rate tom_rate : ℝ)
  (combined_rate remaining_lawn total_time : ℝ) :
  mary_time = 3 → 
  tom_time = 6 → 
  tom_solo_work = 3 → 
  mary_rate = 1 / mary_time → 
  tom_rate = 1 / tom_time → 
  combined_rate = mary_rate + tom_rate →
  remaining_lawn = 1 - (tom_solo_work * tom_rate) →
  total_time = tom_solo_work + (remaining_lawn / combined_rate) →
  total_time = 4 :=
by sorry

end NUMINAMATH_GPT_mowing_lawn_time_l1283_128391


namespace NUMINAMATH_GPT_ratio_twelfth_term_geometric_sequence_l1283_128338

theorem ratio_twelfth_term_geometric_sequence (G H : ℕ → ℝ) (n : ℕ) (a r b s : ℝ)
  (hG : ∀ n, G n = a * (r^n - 1) / (r - 1))
  (hH : ∀ n, H n = b * (s^n - 1) / (s - 1))
  (ratio_condition : ∀ n, G n / H n = (5 * n + 3) / (3 * n + 17)) :
  (a * r^11) / (b * s^11) = 2 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_twelfth_term_geometric_sequence_l1283_128338


namespace NUMINAMATH_GPT_wall_length_to_height_ratio_l1283_128365

theorem wall_length_to_height_ratio
  (W H L : ℝ)
  (V : ℝ)
  (h1 : H = 6 * W)
  (h2 : L * H * W = V)
  (h3 : V = 86436)
  (h4 : W = 6.999999999999999) :
  L / H = 7 :=
by
  sorry

end NUMINAMATH_GPT_wall_length_to_height_ratio_l1283_128365


namespace NUMINAMATH_GPT_polar_coordinates_to_rectangular_l1283_128367

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_coordinates_to_rectangular :
  polar_to_rectangular 10 (11 * Real.pi / 6) = (5 * Real.sqrt 3, -5) :=
by
  sorry

end NUMINAMATH_GPT_polar_coordinates_to_rectangular_l1283_128367


namespace NUMINAMATH_GPT_no_solutions_exist_l1283_128379

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end NUMINAMATH_GPT_no_solutions_exist_l1283_128379


namespace NUMINAMATH_GPT_power_function_not_pass_origin_l1283_128399

noncomputable def does_not_pass_through_origin (m : ℝ) : Prop :=
  ∀ x:ℝ, (m^2 - 3 * m + 3) * x^(m^2 - m - 2) ≠ 0

theorem power_function_not_pass_origin (m : ℝ) :
  does_not_pass_through_origin m ↔ (m = 1 ∨ m = 2) :=
sorry

end NUMINAMATH_GPT_power_function_not_pass_origin_l1283_128399


namespace NUMINAMATH_GPT_mashed_potatoes_suggestion_count_l1283_128370

def number_of_students_suggesting_bacon := 394
def extra_students_suggesting_mashed_potatoes := 63
def number_of_students_suggesting_mashed_potatoes := number_of_students_suggesting_bacon + extra_students_suggesting_mashed_potatoes

theorem mashed_potatoes_suggestion_count :
  number_of_students_suggesting_mashed_potatoes = 457 := by
  sorry

end NUMINAMATH_GPT_mashed_potatoes_suggestion_count_l1283_128370


namespace NUMINAMATH_GPT_combination_8_5_l1283_128310

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end NUMINAMATH_GPT_combination_8_5_l1283_128310


namespace NUMINAMATH_GPT_rotated_point_l1283_128393

def point := (ℝ × ℝ × ℝ)

def rotate_point (A P : point) (θ : ℝ) : point :=
  -- Function implementing the rotation (the full definition would normally be placed here)
  sorry

def A : point := (1, 1, 1)
def P : point := (1, 1, 0)

theorem rotated_point (θ : ℝ) (hθ : θ = 60) :
  rotate_point A P θ = (1/3, 4/3, 1/3) :=
sorry

end NUMINAMATH_GPT_rotated_point_l1283_128393


namespace NUMINAMATH_GPT_shortest_path_from_A_to_D_not_inside_circle_l1283_128307

noncomputable def shortest_path_length : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (18, 24)
  let O : ℝ × ℝ := (9, 12)
  let r : ℝ := 15
  15 * Real.pi

theorem shortest_path_from_A_to_D_not_inside_circle :
  let A := (0, 0)
  let D := (18, 24)
  let O := (9, 12)
  let r := 15
  shortest_path_length = 15 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_shortest_path_from_A_to_D_not_inside_circle_l1283_128307


namespace NUMINAMATH_GPT_function_zero_interval_l1283_128377

noncomputable def f (x : ℝ) : ℝ := 1 / 4^x - Real.log x / Real.log 4

theorem function_zero_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 := by
  sorry

end NUMINAMATH_GPT_function_zero_interval_l1283_128377


namespace NUMINAMATH_GPT_values_of_xyz_l1283_128384

theorem values_of_xyz (x y z : ℝ) (h1 : 2 * x - y + z = 14) (h2 : y = 2) (h3 : x + z = 3 * y + 5) : 
  x = 5 ∧ y = 2 ∧ z = 6 := 
by
  sorry

end NUMINAMATH_GPT_values_of_xyz_l1283_128384


namespace NUMINAMATH_GPT_estimate_total_observations_in_interval_l1283_128371

def total_observations : ℕ := 1000
def sample_size : ℕ := 50
def frequency_in_sample : ℝ := 0.12

theorem estimate_total_observations_in_interval : 
  frequency_in_sample * (total_observations : ℝ) = 120 :=
by
  -- conditions defined above
  -- use given frequency to estimate the total observations in the interval
  -- actual proof omitted
  sorry

end NUMINAMATH_GPT_estimate_total_observations_in_interval_l1283_128371


namespace NUMINAMATH_GPT_percentage_less_than_m_add_d_l1283_128357

def symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P m - x = P m + x

def within_one_stdev (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  P m - d = 0.68 ∧ P m + d = 0.68

theorem percentage_less_than_m_add_d 
  (P : ℝ → ℝ) (m d : ℝ) 
  (symm : symmetric_about_mean P m)
  (within_stdev : within_one_stdev P m d) : 
  ∃ f, f = 0.84 :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_m_add_d_l1283_128357


namespace NUMINAMATH_GPT_two_digit_number_reversed_l1283_128380

theorem two_digit_number_reversed :
  ∃ (x y : ℕ), (10 * x + y = 73) ∧ (10 * x + y = 2 * (10 * y + x) - 1) ∧ (x < 10) ∧ (y < 10) := 
by
  sorry

end NUMINAMATH_GPT_two_digit_number_reversed_l1283_128380


namespace NUMINAMATH_GPT_number_of_moles_of_H2O_l1283_128311

def reaction_stoichiometry (n_NaOH m_Cl2 : ℕ) : ℕ :=
  1  -- Moles of H2O produced according to the balanced equation with the given reactants

theorem number_of_moles_of_H2O 
  (n_NaOH : ℕ) (m_Cl2 : ℕ) 
  (h_NaOH : n_NaOH = 2) 
  (h_Cl2 : m_Cl2 = 1) :
  reaction_stoichiometry n_NaOH m_Cl2 = 1 :=
by
  rw [h_NaOH, h_Cl2]
  -- Would typically follow with the proof using the conditions and stoichiometric relation
  sorry  -- Proof step omitted

end NUMINAMATH_GPT_number_of_moles_of_H2O_l1283_128311


namespace NUMINAMATH_GPT_none_of_the_above_option_l1283_128398

-- Define integers m and n
variables (m n: ℕ)

-- Define P and R in terms of m and n
def P : ℕ := 2^m
def R : ℕ := 5^n

-- Define the statement to prove
theorem none_of_the_above_option : ∀ (m n : ℕ), 15^(m + n) ≠ P^(m + n) * R ∧ 15^(m + n) ≠ (3^m * 3^n * 5^m) ∧ 15^(m + n) ≠ (3^m * P^n) ∧ 15^(m + n) ≠ (2^m * 5^n * 5^m) :=
by sorry

end NUMINAMATH_GPT_none_of_the_above_option_l1283_128398


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_l1283_128305

open Real

variable (α : ℝ)

theorem cos_pi_minus_alpha (h1 : 0 < α ∧ α < π / 2) (h2 : sin α = 4 / 5) : cos (π - α) = -3 / 5 := by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_l1283_128305


namespace NUMINAMATH_GPT_common_ratio_l1283_128372

theorem common_ratio (a S r : ℝ) (h1 : S = a / (1 - r))
  (h2 : ar^5 / (1 - r) = S / 81) : r = 1 / 3 :=
sorry

end NUMINAMATH_GPT_common_ratio_l1283_128372


namespace NUMINAMATH_GPT_ways_to_distribute_books_into_bags_l1283_128330

theorem ways_to_distribute_books_into_bags : 
  let books := 5
  let bags := 4
  ∃ (ways : ℕ), ways = 41 := 
sorry

end NUMINAMATH_GPT_ways_to_distribute_books_into_bags_l1283_128330


namespace NUMINAMATH_GPT_chromium_percentage_l1283_128378

theorem chromium_percentage (x : ℝ) : 
  (15 * x / 100 + 35 * 8 / 100 = 50 * 8.6 / 100) → 
  x = 10 := 
sorry

end NUMINAMATH_GPT_chromium_percentage_l1283_128378


namespace NUMINAMATH_GPT_shaded_region_is_hyperbolas_l1283_128373

theorem shaded_region_is_hyperbolas (T : ℝ) (hT : T > 0) :
  (∃ (x y : ℝ), x * y = T / 4) ∧ (∃ (x y : ℝ), x * y = - (T / 4)) :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_is_hyperbolas_l1283_128373


namespace NUMINAMATH_GPT_number_of_valid_sequences_l1283_128345

-- Define the sequence property
def sequence_property (b : Fin 10 → Fin 10) : Prop :=
  ∀ i : Fin 10, 2 ≤ i → (∃ j : Fin 10, j < i ∧ (b j = b i + 1 ∨ b j = b i - 1 ∨ b j = b i + 2 ∨ b j = b i - 2))

-- Define the set of such sequences
def valid_sequences : Set (Fin 10 → Fin 10) := {b | sequence_property b}

-- Define the number of such sequences
def number_of_sequences : Fin 512 :=
  sorry -- Proof omitted for brevity

-- The final statement
theorem number_of_valid_sequences : number_of_sequences = 512 :=
  sorry  -- Skip proof

end NUMINAMATH_GPT_number_of_valid_sequences_l1283_128345


namespace NUMINAMATH_GPT_relationship_M_N_l1283_128347

def M : Set Int := {-1, 0, 1}
def N : Set Int := {x | ∃ a b : Int, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem relationship_M_N : N ⊆ M ∧ N ≠ M := by
  sorry

end NUMINAMATH_GPT_relationship_M_N_l1283_128347


namespace NUMINAMATH_GPT_machine_value_after_2_years_l1283_128351

section
def initial_value : ℝ := 1200
def depreciation_rate_year1 : ℝ := 0.10
def depreciation_rate_year2 : ℝ := 0.12
def repair_rate : ℝ := 0.03
def major_overhaul_rate : ℝ := 0.15

theorem machine_value_after_2_years :
  let value_after_repairs_2 := (initial_value * (1 - depreciation_rate_year1) + initial_value * repair_rate) * (1 - depreciation_rate_year2 + repair_rate)
  (value_after_repairs_2 * (1 - major_overhaul_rate)) = 863.23 := 
by
  -- proof here
  sorry
end

end NUMINAMATH_GPT_machine_value_after_2_years_l1283_128351


namespace NUMINAMATH_GPT_minimum_value_func1_minimum_value_func2_l1283_128303

-- Problem (1): 
theorem minimum_value_func1 (x : ℝ) (h : x > -1) : 
  (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

-- Problem (2): 
theorem minimum_value_func2 (x : ℝ) (h : x > 1) : 
  (x^2 + 8) / (x - 1) ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_func1_minimum_value_func2_l1283_128303


namespace NUMINAMATH_GPT_triangle_inequality_third_side_l1283_128344

theorem triangle_inequality_third_side (a : ℝ) (h1 : 3 + a > 7) (h2 : 7 + a > 3) (h3 : 3 + 7 > a) : 
  4 < a ∧ a < 10 :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_third_side_l1283_128344


namespace NUMINAMATH_GPT_sum_ab_eq_negative_two_l1283_128352

def f (x : ℝ) := x^3 + 3 * x^2 + 6 * x + 4

theorem sum_ab_eq_negative_two (a b : ℝ) (h1 : f a = 14) (h2 : f b = -14) : a + b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_ab_eq_negative_two_l1283_128352


namespace NUMINAMATH_GPT_at_least_one_ge_two_l1283_128353

theorem at_least_one_ge_two (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a + 1 / b ≥ 2 ∨ b + 1 / c ≥ 2 ∨ c + 1 / a ≥ 2 := 
sorry

end NUMINAMATH_GPT_at_least_one_ge_two_l1283_128353


namespace NUMINAMATH_GPT_initial_bottle_caps_l1283_128354

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end NUMINAMATH_GPT_initial_bottle_caps_l1283_128354


namespace NUMINAMATH_GPT_find_m_l1283_128362

open Set

def A (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 2}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem find_m (m : ℝ) :
  (A m ∩ B = ∅ ∧ A m ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1283_128362


namespace NUMINAMATH_GPT_tickets_needed_for_equal_distribution_l1283_128374

theorem tickets_needed_for_equal_distribution :
  ∃ k : ℕ, 865 + k ≡ 0 [MOD 9] ∧ k = 8 := sorry

end NUMINAMATH_GPT_tickets_needed_for_equal_distribution_l1283_128374


namespace NUMINAMATH_GPT_min_value_x3_y2_z_w2_l1283_128360

theorem min_value_x3_y2_z_w2 (x y z w : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w)
  (h : (1/x) + (1/y) + (1/z) + (1/w) = 8) : x^3 * y^2 * z * w^2 ≥ 1/432 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x3_y2_z_w2_l1283_128360


namespace NUMINAMATH_GPT_value_two_stddev_below_mean_l1283_128315

def mean : ℝ := 16.2
def standard_deviation : ℝ := 2.3

theorem value_two_stddev_below_mean : mean - 2 * standard_deviation = 11.6 :=
by
  sorry

end NUMINAMATH_GPT_value_two_stddev_below_mean_l1283_128315


namespace NUMINAMATH_GPT_sum_lent_is_10000_l1283_128337

theorem sum_lent_is_10000
  (P : ℝ)
  (r : ℝ := 0.075)
  (t : ℝ := 7)
  (I : ℝ := P - 4750) 
  (H1 : I = P * r * t) :
  P = 10000 :=
sorry

end NUMINAMATH_GPT_sum_lent_is_10000_l1283_128337


namespace NUMINAMATH_GPT_total_seashells_l1283_128390

theorem total_seashells :
  let initial_seashells : ℝ := 6.5
  let more_seashells : ℝ := 4.25
  initial_seashells + more_seashells = 10.75 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_l1283_128390


namespace NUMINAMATH_GPT_total_marbles_proof_l1283_128356

def red_marble_condition (b r : ℕ) : Prop :=
  r = b + (3 * b / 10)

def yellow_marble_condition (r y : ℕ) : Prop :=
  y = r + (5 * r / 10)

def total_marbles (b r y : ℕ) : ℕ :=
  r + b + y

theorem total_marbles_proof (b r y : ℕ)
  (h1 : red_marble_condition b r)
  (h2 : yellow_marble_condition r y) :
  total_marbles b r y = 425 * r / 130 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_marbles_proof_l1283_128356


namespace NUMINAMATH_GPT_average_temperature_l1283_128325

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end NUMINAMATH_GPT_average_temperature_l1283_128325


namespace NUMINAMATH_GPT_plane_speed_ratio_train_l1283_128340

def distance (speed time : ℝ) := speed * time

theorem plane_speed_ratio_train (x y z : ℝ)
  (h_train : distance x 20 = distance y 10)
  (h_wait_time : z > 5)
  (h_plane_meet_train : distance y (8/9) = distance x (z + 8/9)) :
  y = 10 * x :=
by {
  sorry
}

end NUMINAMATH_GPT_plane_speed_ratio_train_l1283_128340


namespace NUMINAMATH_GPT_total_distance_walked_l1283_128395

-- Define the conditions
def walking_rate : ℝ := 4
def time_before_break : ℝ := 2
def time_after_break : ℝ := 0.5

-- Define the required theorem
theorem total_distance_walked : 
  walking_rate * time_before_break + walking_rate * time_after_break = 10 := 
sorry

end NUMINAMATH_GPT_total_distance_walked_l1283_128395


namespace NUMINAMATH_GPT_find_p_q_r_l1283_128332

theorem find_p_q_r  (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
                    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p / q) - Real.sqrt r)
                    (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
                    (rel_prime : Nat.gcd p q = 1) : 
                    p + q + r = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_p_q_r_l1283_128332


namespace NUMINAMATH_GPT_aaron_and_carson_scoops_l1283_128383

def initial_savings (a c : ℕ) : Prop :=
  a = 150 ∧ c = 150

def total_savings (t a c : ℕ) : Prop :=
  t = a + c

def restaurant_expense (r t : ℕ) : Prop :=
  r = 3 * t / 4

def service_charge_inclusive (r sc : ℕ) : Prop :=
  r = sc * 115 / 100

def remaining_money (t r rm : ℕ) : Prop :=
  rm = t - r

def money_left (al cl : ℕ) : Prop :=
  al = 4 ∧ cl = 4

def ice_cream_scoop_cost (s : ℕ) : Prop :=
  s = 4

def total_scoops (rm ml s scoop_total : ℕ) : Prop :=
  scoop_total = (rm - (ml - 4 - 4)) / s

theorem aaron_and_carson_scoops :
  ∃ a c t r sc rm al cl s scoop_total, initial_savings a c ∧
  total_savings t a c ∧
  restaurant_expense r t ∧
  service_charge_inclusive r sc ∧
  remaining_money t r rm ∧
  money_left al cl ∧
  ice_cream_scoop_cost s ∧
  total_scoops rm (al + cl) s scoop_total ∧
  scoop_total = 16 :=
sorry

end NUMINAMATH_GPT_aaron_and_carson_scoops_l1283_128383


namespace NUMINAMATH_GPT_find_other_number_l1283_128397

theorem find_other_number
  (x y lcm hcf : ℕ)
  (h_lcm : Nat.lcm x y = lcm)
  (h_hcf : Nat.gcd x y = hcf)
  (h_x : x = 462)
  (h_lcm_value : lcm = 2310)
  (h_hcf_value : hcf = 30) :
  y = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1283_128397


namespace NUMINAMATH_GPT_longest_side_of_triangle_l1283_128387

-- Definitions of the conditions in a)
def side1 : ℝ := 9
def side2 (x : ℝ) : ℝ := x + 5
def side3 (x : ℝ) : ℝ := 2 * x + 3
def perimeter : ℝ := 40

-- Statement of the mathematically equivalent proof problem.
theorem longest_side_of_triangle (x : ℝ) (h : side1 + side2 x + side3 x = perimeter) : 
  max side1 (max (side2 x) (side3 x)) = side3 x := 
sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l1283_128387


namespace NUMINAMATH_GPT_abs_diff_x_y_l1283_128382

variables {x y : ℝ}

noncomputable def floor (z : ℝ) : ℤ := Int.floor z
noncomputable def fract (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y 
  (h1 : floor x + fract y = 3.7) 
  (h2 : fract x + floor y = 4.6) : 
  |x - y| = 1.1 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_x_y_l1283_128382


namespace NUMINAMATH_GPT_birds_on_fence_l1283_128309

def number_of_birds_on_fence : ℕ := 20

theorem birds_on_fence (x : ℕ) (h : 2 * x + 10 = 50) : x = number_of_birds_on_fence :=
by
  sorry

end NUMINAMATH_GPT_birds_on_fence_l1283_128309


namespace NUMINAMATH_GPT_inequality_proof_l1283_128335

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1) : 
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1283_128335


namespace NUMINAMATH_GPT_correct_equation_for_programmers_l1283_128364

theorem correct_equation_for_programmers (x : ℕ) 
  (hB : x > 0) 
  (programmer_b_speed : ℕ := x) 
  (programmer_a_speed : ℕ := 2 * x) 
  (data : ℕ := 2640) :
  (data / programmer_a_speed = data / programmer_b_speed - 120) :=
by
  -- sorry is used to skip the proof, focus on the statement
  sorry

end NUMINAMATH_GPT_correct_equation_for_programmers_l1283_128364


namespace NUMINAMATH_GPT_range_of_a_l1283_128361

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1/x) + a

theorem range_of_a (a : ℝ) (h : f a 0 = a^2) : (f a 0 = f a 0 -> 0 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1283_128361


namespace NUMINAMATH_GPT_custom_op_2006_l1283_128326

def custom_op (n : ℕ) : ℕ := 
  match n with 
  | 0 => 1
  | (n+1) => 2 + custom_op n

theorem custom_op_2006 : custom_op 2005 = 4011 :=
by {
  sorry
}

end NUMINAMATH_GPT_custom_op_2006_l1283_128326


namespace NUMINAMATH_GPT_profit_last_month_l1283_128376

variable (gas_expenses earnings_per_lawn lawns_mowed extra_income profit : ℤ)

def toms_profit (gas_expenses earnings_per_lawn lawns_mowed extra_income : ℤ) : ℤ :=
  (lawns_mowed * earnings_per_lawn + extra_income) - gas_expenses

theorem profit_last_month :
  toms_profit 17 12 3 10 = 29 :=
by
  rw [toms_profit]
  sorry

end NUMINAMATH_GPT_profit_last_month_l1283_128376


namespace NUMINAMATH_GPT_shaniqua_haircuts_l1283_128349

theorem shaniqua_haircuts
  (H : ℕ) -- number of haircuts
  (haircut_income : ℕ) (style_income : ℕ)
  (total_styles : ℕ) (total_income : ℕ)
  (haircut_income_eq : haircut_income = 12)
  (style_income_eq : style_income = 25)
  (total_styles_eq : total_styles = 5)
  (total_income_eq : total_income = 221)
  (income_from_styles : ℕ := total_styles * style_income)
  (income_from_haircuts : ℕ := total_income - income_from_styles) :
  H = income_from_haircuts / haircut_income :=
sorry

end NUMINAMATH_GPT_shaniqua_haircuts_l1283_128349


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l1283_128339

noncomputable def repeating_decimal := 7 + ((789 : ℚ) / (10^4 - 1))

theorem repeating_decimal_fraction :
  repeating_decimal = (365 : ℚ) / 85 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l1283_128339


namespace NUMINAMATH_GPT_angle_in_first_quadrant_l1283_128314

theorem angle_in_first_quadrant (x : ℝ) (h1 : Real.tan x > 0) (h2 : Real.sin x + Real.cos x > 0) : 
  0 < Real.sin x ∧ 0 < Real.cos x := 
by 
  sorry

end NUMINAMATH_GPT_angle_in_first_quadrant_l1283_128314


namespace NUMINAMATH_GPT_fraction_of_menu_safely_eaten_l1283_128320

-- Given conditions
def VegetarianDishes := 6
def GlutenContainingVegetarianDishes := 5
def TotalDishes := 3 * VegetarianDishes

-- Derived information
def GlutenFreeVegetarianDishes := VegetarianDishes - GlutenContainingVegetarianDishes

-- Question: What fraction of the menu can Sarah safely eat?
theorem fraction_of_menu_safely_eaten : 
  (GlutenFreeVegetarianDishes / TotalDishes) = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_menu_safely_eaten_l1283_128320


namespace NUMINAMATH_GPT_divisible_by_5_l1283_128334

theorem divisible_by_5 (x y : ℕ) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 5 ∣ x := sorry

end NUMINAMATH_GPT_divisible_by_5_l1283_128334


namespace NUMINAMATH_GPT_factor_by_resultant_l1283_128312

theorem factor_by_resultant (x f : ℤ) (h1 : x = 17) (h2 : (2 * x + 5) * f = 117) : f = 3 := 
by
  sorry

end NUMINAMATH_GPT_factor_by_resultant_l1283_128312


namespace NUMINAMATH_GPT_larger_integer_value_l1283_128331

theorem larger_integer_value (x y : ℕ) (h1 : (4 * x)^2 - 2 * x = 8100) (h2 : x + 10 = 2 * y) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_value_l1283_128331


namespace NUMINAMATH_GPT_find_units_digit_l1283_128363

theorem find_units_digit : 
  (7^1993 + 5^1993) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_units_digit_l1283_128363


namespace NUMINAMATH_GPT_sum_of_pills_in_larger_bottles_l1283_128317

-- Definitions based on the conditions
def supplements := 5
def pills_in_small_bottles := 2 * 30
def pills_per_day := 5
def days_used := 14
def pills_remaining := 350
def total_pills_before := pills_remaining + (pills_per_day * days_used)
def total_pills_in_large_bottles := total_pills_before - pills_in_small_bottles

-- The theorem statement that needs to be proven
theorem sum_of_pills_in_larger_bottles : total_pills_in_large_bottles = 360 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_of_pills_in_larger_bottles_l1283_128317


namespace NUMINAMATH_GPT_allowance_calculation_l1283_128313

theorem allowance_calculation (A : ℝ)
  (h1 : (3 / 5) * A + (1 / 3) * (2 / 5) * A + 0.40 = A)
  : A = 1.50 :=
sorry

end NUMINAMATH_GPT_allowance_calculation_l1283_128313


namespace NUMINAMATH_GPT_find_m_range_l1283_128319

-- Define the mathematical objects and conditions
def condition_p (m : ℝ) : Prop :=
  (|1 - m| / Real.sqrt 2) > 1

def condition_q (m : ℝ) : Prop :=
  m < 4

-- Define the proof problem
theorem find_m_range (p q : Prop) (m : ℝ) 
  (hp : ¬ p) (hq : q) (hpq : p ∨ q)
  (hP_imp : p → condition_p m)
  (hQ_imp : q → condition_q m) : 
  1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_find_m_range_l1283_128319


namespace NUMINAMATH_GPT_hyperbola_equation_l1283_128318

theorem hyperbola_equation (h1 : ∀ x y : ℝ, (x = 0 ∧ y = 0)) 
                           (h2 : ∀ a : ℝ, (2 * a = 4)) 
                           (h3 : ∀ c : ℝ, (c = 3)) : 
  ∃ b : ℝ, (b^2 = 5) ∧ (∀ x y : ℝ, (y^2 / 4) - (x^2 / b^2) = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1283_128318


namespace NUMINAMATH_GPT_zoo_visitors_l1283_128324

theorem zoo_visitors (visitors_friday : ℕ) 
  (h1 : 3 * visitors_friday = 3750) :
  visitors_friday = 1250 := 
sorry

end NUMINAMATH_GPT_zoo_visitors_l1283_128324


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1283_128316

theorem geometric_sequence_sum (a_1 : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  (∀ n, S (n+1) = a_1 * (1 - q^(n+1)) / (1 - q)) →
  S 4 / a_1 = 15 :=
by
  intros hq hsum
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1283_128316


namespace NUMINAMATH_GPT_snail_kite_eats_35_snails_l1283_128388

theorem snail_kite_eats_35_snails : 
  let day1 := 3
  let day2 := day1 + 2
  let day3 := day2 + 2
  let day4 := day3 + 2
  let day5 := day4 + 2
  day1 + day2 + day3 + day4 + day5 = 35 := 
by
  sorry

end NUMINAMATH_GPT_snail_kite_eats_35_snails_l1283_128388


namespace NUMINAMATH_GPT_simplify_expression_eval_at_2_l1283_128304

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (x^2 + a)^2 / ((a - b) * (a - c)) + (x^2 + b)^2 / ((b - a) * (b - c)) + (x^2 + c)^2 / ((c - a) * (c - b)) =
    x^4 + x^2 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

theorem eval_at_2 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
    (2^2 + a)^2 / ((a - b) * (a - c)) + (2^2 + b)^2 / ((b - a) * (b - c)) + (2^2 + c)^2 / ((c - a) * (c - b)) =
    16 + 4 * (a + b + c) + (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_GPT_simplify_expression_eval_at_2_l1283_128304


namespace NUMINAMATH_GPT_max_common_ratio_arithmetic_geometric_sequence_l1283_128386

open Nat

theorem max_common_ratio_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (k : ℕ) (q : ℝ) 
  (hk : k ≥ 2) (ha : ∀ n, a (n + 1) = a n + d)
  (hg : (a 1) * (a (2 * k)) = (a k) ^ 2) :
  q ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_common_ratio_arithmetic_geometric_sequence_l1283_128386


namespace NUMINAMATH_GPT_tom_fractions_l1283_128342

theorem tom_fractions (packages : ℕ) (cars_per_package : ℕ) (cars_left : ℕ) (nephews : ℕ) :
  packages = 10 → 
  cars_per_package = 5 → 
  cars_left = 30 → 
  nephews = 2 → 
  ∃ fraction_given : ℚ, fraction_given = 1/5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tom_fractions_l1283_128342


namespace NUMINAMATH_GPT_intersection_A_B_l1283_128343

def A : Set ℝ := { x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3) }
def B : Set ℝ := { x | ∃ y, y = Real.log x }

theorem intersection_A_B : A ∩ B = {x | x ∈ Set.Ici 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1283_128343


namespace NUMINAMATH_GPT_symmetric_point_origin_l1283_128359

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l1283_128359


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1283_128336

theorem sufficient_but_not_necessary (x: ℝ) (hx: 0 < x ∧ x < 1) : 0 < x^2 ∧ x^2 < 1 ∧ (∀ y, 0 < y^2 ∧ y^2 < 1 → (y > 0 ∧ y < 1 ∨ y < 0 ∧ y > -1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1283_128336


namespace NUMINAMATH_GPT_mason_savings_fraction_l1283_128306

theorem mason_savings_fraction (M p b : ℝ) (h : (1 / 4) * M = (2 / 5) * b * p) : 
  (M - b * p) / M = 3 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_mason_savings_fraction_l1283_128306


namespace NUMINAMATH_GPT_example_problem_l1283_128328

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end NUMINAMATH_GPT_example_problem_l1283_128328


namespace NUMINAMATH_GPT_B_pow_150_eq_I_l1283_128308

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end NUMINAMATH_GPT_B_pow_150_eq_I_l1283_128308


namespace NUMINAMATH_GPT_g_at_5_l1283_128300

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation :
  ∀ (x : ℝ), g x + 3 * g (2 - x) = 4 * x ^ 2 - 5 * x + 1

theorem g_at_5 : g 5 = -5 / 4 :=
by
  let h := functional_equation
  sorry

end NUMINAMATH_GPT_g_at_5_l1283_128300


namespace NUMINAMATH_GPT_intersection_complementA_setB_l1283_128396

noncomputable def setA : Set ℝ := { x | abs x > 1 }

noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

noncomputable def complementA : Set ℝ := { x | abs x ≤ 1 }

theorem intersection_complementA_setB : 
  (complementA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_complementA_setB_l1283_128396


namespace NUMINAMATH_GPT_debbie_total_tape_l1283_128368

def large_box_tape : ℕ := 4
def medium_box_tape : ℕ := 2
def small_box_tape : ℕ := 1
def label_tape : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5

def total_tape_used : ℕ := 
  (large_boxes_packed * (large_box_tape + label_tape)) +
  (medium_boxes_packed * (medium_box_tape + label_tape)) +
  (small_boxes_packed * (small_box_tape + label_tape))

theorem debbie_total_tape : total_tape_used = 44 := by
  sorry

end NUMINAMATH_GPT_debbie_total_tape_l1283_128368


namespace NUMINAMATH_GPT_running_time_constant_pace_l1283_128333

/-!
# Running Time Problem

We are given that the running pace is constant, it takes 30 minutes to run 5 miles,
and we need to find out how long it will take to run 2.5 miles.
-/

theorem running_time_constant_pace :
  ∀ (distance_to_store distance_to_cousin distance_run time_run : ℝ)
  (constant_pace : Prop),
  distance_to_store = 5 → time_run = 30 → distance_to_cousin = 2.5 →
  constant_pace → 
  time_run / distance_to_store * distance_to_cousin = 15 :=
by 
  intros distance_to_store distance_to_cousin distance_run time_run constant_pace 
         hds htr hdc hcp
  rw [hds, htr, hdc]
  exact sorry

end NUMINAMATH_GPT_running_time_constant_pace_l1283_128333


namespace NUMINAMATH_GPT_expression_is_integer_l1283_128327

theorem expression_is_integer (m : ℕ) (hm : 0 < m) :
  ∃ k : ℤ, k = (m^4 / 24 + m^3 / 4 + 11*m^2 / 24 + m / 4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_expression_is_integer_l1283_128327


namespace NUMINAMATH_GPT_Tahir_contribution_l1283_128381

theorem Tahir_contribution
  (headphone_cost : ℕ := 200)
  (kenji_yen : ℕ := 15000)
  (exchange_rate : ℕ := 100)
  (kenji_contribution : ℕ := kenji_yen / exchange_rate)
  (tahir_contribution : ℕ := headphone_cost - kenji_contribution) :
  tahir_contribution = 50 := 
  by sorry

end NUMINAMATH_GPT_Tahir_contribution_l1283_128381


namespace NUMINAMATH_GPT_base_conversion_arithmetic_l1283_128350

theorem base_conversion_arithmetic :
  let b5 := 2013
  let b3 := 11
  let b6 := 3124
  let b7 := 4321
  (b5₅ / b3₃ - b6₆ + b7₇ : ℝ) = 898.5 :=
by sorry

end NUMINAMATH_GPT_base_conversion_arithmetic_l1283_128350


namespace NUMINAMATH_GPT_five_hash_neg_one_l1283_128375

def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

theorem five_hash_neg_one : hash 5 (-1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_five_hash_neg_one_l1283_128375


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1283_128346

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_product (a_1 d : ℤ) :
  (a_n 4 a_1 d) + (a_n 7 a_1 d) = 2 →
  (a_n 5 a_1 d) * (a_n 6 a_1 d) = -3 →
  a_1 * (a_n 10 a_1 d) = -323 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1283_128346


namespace NUMINAMATH_GPT_time_ratio_l1283_128321

theorem time_ratio (distance : ℝ) (initial_time : ℝ) (new_speed : ℝ) :
  distance = 600 → initial_time = 5 → new_speed = 80 → (distance / new_speed) / initial_time = 1.5 :=
by
  intros hdist htime hspeed
  sorry

end NUMINAMATH_GPT_time_ratio_l1283_128321


namespace NUMINAMATH_GPT_sin_390_eq_half_l1283_128341

theorem sin_390_eq_half : Real.sin (390 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_390_eq_half_l1283_128341


namespace NUMINAMATH_GPT_min_reciprocal_sum_l1283_128394

theorem min_reciprocal_sum (a b x y : ℝ) (h1 : 8 * x - y - 4 ≤ 0) (h2 : x + y + 1 ≥ 0) (h3 : y - 4 * x ≤ 0) 
    (ha : a > 0) (hb : b > 0) (hz : a * x + b * y = 2) : 
    1 / a + 1 / b = 9 / 2 := 
    sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l1283_128394


namespace NUMINAMATH_GPT_ext_9_implication_l1283_128348

theorem ext_9_implication (a b : ℝ) (h1 : 3 + 2 * a + b = 0) (h2 : 1 + a + b + a^2 = 10) : (2 : ℝ)^3 + a * (2 : ℝ)^2 + b * (2 : ℝ) + a^2 - 1 = 17 := by
  sorry

end NUMINAMATH_GPT_ext_9_implication_l1283_128348


namespace NUMINAMATH_GPT_abs_z1_purely_imaginary_l1283_128366

noncomputable def z1 (a : ℝ) : Complex := ⟨a, 2⟩
def z2 : Complex := ⟨2, -1⟩

theorem abs_z1_purely_imaginary (a : ℝ) (ha : 2 * a - 2 = 0) : Complex.abs (z1 a) = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_z1_purely_imaginary_l1283_128366


namespace NUMINAMATH_GPT_compute_f_at_5_l1283_128389

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (10 ^ x) = x

theorem compute_f_at_5 : f 5 = Real.log 5 / Real.log 10 :=
by
  sorry

end NUMINAMATH_GPT_compute_f_at_5_l1283_128389


namespace NUMINAMATH_GPT_find_doodads_produced_in_four_hours_l1283_128302

theorem find_doodads_produced_in_four_hours :
  ∃ (n : ℕ),
    (∀ (workers hours widgets doodads : ℕ),
      (workers = 150 ∧ hours = 2 ∧ widgets = 800 ∧ doodads = 500) ∨
      (workers = 100 ∧ hours = 3 ∧ widgets = 750 ∧ doodads = 600) ∨
      (workers = 80  ∧ hours = 4 ∧ widgets = 480 ∧ doodads = n)
    ) → n = 640 :=
sorry

end NUMINAMATH_GPT_find_doodads_produced_in_four_hours_l1283_128302
