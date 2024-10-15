import Mathlib

namespace NUMINAMATH_GPT_cube_surface_area_l2313_231310

theorem cube_surface_area (v : ℝ) (h : v = 1000) : ∃ (s : ℝ), s^3 = v ∧ 6 * s^2 = 600 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2313_231310


namespace NUMINAMATH_GPT_matthew_total_time_l2313_231378

def assemble_time : ℝ := 1
def bake_time_normal : ℝ := 1.5
def decorate_time : ℝ := 1
def bake_time_double : ℝ := bake_time_normal * 2

theorem matthew_total_time :
  assemble_time + bake_time_double + decorate_time = 5 := 
by 
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_matthew_total_time_l2313_231378


namespace NUMINAMATH_GPT_edge_ratio_of_cubes_l2313_231305

theorem edge_ratio_of_cubes (a b : ℝ) (h : (a^3) / (b^3) = 64) : a / b = 4 :=
sorry

end NUMINAMATH_GPT_edge_ratio_of_cubes_l2313_231305


namespace NUMINAMATH_GPT_a_share_correct_l2313_231318

-- Investment periods for each individual in months
def investment_a := 12
def investment_b := 6
def investment_c := 4
def investment_d := 9
def investment_e := 7
def investment_f := 5

-- Investment multiplier for each individual
def multiplier_b := 2
def multiplier_c := 3
def multiplier_d := 4
def multiplier_e := 5
def multiplier_f := 6

-- Total annual gain
def total_gain := 38400

-- Calculate individual shares
def share_a (x : ℝ) := x * investment_a
def share_b (x : ℝ) := multiplier_b * x * investment_b
def share_c (x : ℝ) := multiplier_c * x * investment_c
def share_d (x : ℝ) := multiplier_d * x * investment_d
def share_e (x : ℝ) := multiplier_e * x * investment_e
def share_f (x : ℝ) := multiplier_f * x * investment_f

-- Calculate total investment
def total_investment (x : ℝ) :=
  share_a x + share_b x + share_c x + share_d x + share_e x + share_f x

-- Prove that a's share of the annual gain is Rs. 3360
theorem a_share_correct : 
  ∃ x : ℝ, (12 * x / total_investment x) * total_gain = 3360 := 
sorry

end NUMINAMATH_GPT_a_share_correct_l2313_231318


namespace NUMINAMATH_GPT_a5_value_l2313_231339

theorem a5_value (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a2 - a1 = 2)
  (h2 : a3 - a2 = 4)
  (h3 : a4 - a3 = 8)
  (h4 : a5 - a4 = 16) :
  a5 = 31 := by
  sorry

end NUMINAMATH_GPT_a5_value_l2313_231339


namespace NUMINAMATH_GPT_all_points_same_value_l2313_231396

theorem all_points_same_value {f : ℤ × ℤ → ℕ}
  (h : ∀ x y : ℤ, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ k : ℕ, ∀ x y : ℤ, f (x, y) = k :=
sorry

end NUMINAMATH_GPT_all_points_same_value_l2313_231396


namespace NUMINAMATH_GPT_nicky_speed_l2313_231323

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end NUMINAMATH_GPT_nicky_speed_l2313_231323


namespace NUMINAMATH_GPT_find_x_l2313_231359

variable (BrandA_millet : ℝ) (Mix_millet : ℝ) (Mix_ratio_A : ℝ) (Mix_ratio_B : ℝ)

axiom BrandA_contains_60_percent_millet : BrandA_millet = 0.60
axiom Mix_contains_50_percent_millet : Mix_millet = 0.50
axiom Mix_composition : Mix_ratio_A = 0.60 ∧ Mix_ratio_B = 0.40

theorem find_x (x : ℝ) :
  Mix_ratio_A * BrandA_millet + Mix_ratio_B * x = Mix_millet →
  x = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2313_231359


namespace NUMINAMATH_GPT_B_work_days_l2313_231385

theorem B_work_days (A B C : ℕ) (hA : A = 15) (hC : C = 30) (H : (5 / 15) + ((10 * (1 / C + 1 / B)) / (1 / C + 1 / B)) = 1) : B = 30 := by
  sorry

end NUMINAMATH_GPT_B_work_days_l2313_231385


namespace NUMINAMATH_GPT_find_expression_value_l2313_231312

theorem find_expression_value (a b : ℝ)
  (h1 : a^2 - a - 3 = 0)
  (h2 : b^2 - b - 3 = 0) :
  2 * a^3 + b^2 + 3 * a^2 - 11 * a - b + 5 = 23 :=
  sorry

end NUMINAMATH_GPT_find_expression_value_l2313_231312


namespace NUMINAMATH_GPT_line_properties_l2313_231330

theorem line_properties : ∃ m x_intercept, 
  (∀ (x y : ℝ), 4 * x + 7 * y = 28 → y = m * x + 4) ∧ 
  (∀ (x y : ℝ), y = 0 → 4 * x + 7 * y = 28 → x = x_intercept) ∧ 
  m = -4 / 7 ∧ 
  x_intercept = 7 :=
by 
  sorry

end NUMINAMATH_GPT_line_properties_l2313_231330


namespace NUMINAMATH_GPT_hot_dogs_served_for_dinner_l2313_231343

theorem hot_dogs_served_for_dinner
  (l t : ℕ) 
  (h_cond1 : l = 9) 
  (h_cond2 : t = 11) :
  ∃ d : ℕ, d = t - l ∧ d = 2 := by
  sorry

end NUMINAMATH_GPT_hot_dogs_served_for_dinner_l2313_231343


namespace NUMINAMATH_GPT_combined_weight_is_correct_l2313_231329

-- Frank and Gwen's candy weights
def frank_candy : ℕ := 10
def gwen_candy : ℕ := 7

-- The combined weight of candy
def combined_weight : ℕ := frank_candy + gwen_candy

-- Theorem that states the combined weight is 17 pounds
theorem combined_weight_is_correct : combined_weight = 17 :=
by
  -- proves that 10 + 7 = 17
  sorry

end NUMINAMATH_GPT_combined_weight_is_correct_l2313_231329


namespace NUMINAMATH_GPT_sphere_surface_area_l2313_231386

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (S : ℝ)
  (hV : V = 36 * π)
  (hvol : V = (4 / 3) * π * r^3) :
  S = 4 * π * r^2 :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l2313_231386


namespace NUMINAMATH_GPT_congruence_solutions_count_number_of_solutions_l2313_231321

theorem congruence_solutions_count (x : ℕ) (hx_pos : x > 0) (hx_lt : x < 200) :
  (x + 17) % 52 = 75 % 52 ↔ x = 6 ∨ x = 58 ∨ x = 110 ∨ x = 162 :=
by sorry

theorem number_of_solutions :
  (∃ x : ℕ, (0 < x ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52)) ∧
  (∃ x1 x2 x3 x4 : ℕ, x1 = 6 ∧ x2 = 58 ∧ x3 = 110 ∧ x4 = 162) ∧
  4 = 4 :=
by sorry

end NUMINAMATH_GPT_congruence_solutions_count_number_of_solutions_l2313_231321


namespace NUMINAMATH_GPT_monotonic_increasing_condition_l2313_231398

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → y a x₁ ≤ y a x₂) ↔ 
  (a = 0 ∨ a > 0) :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_condition_l2313_231398


namespace NUMINAMATH_GPT_prime_sum_exists_even_n_l2313_231313

theorem prime_sum_exists_even_n (n : ℕ) :
  (∃ a b c : ℤ, a + b + c = 0 ∧ Prime (a^n + b^n + c^n)) ↔ Even n := 
by
  sorry

end NUMINAMATH_GPT_prime_sum_exists_even_n_l2313_231313


namespace NUMINAMATH_GPT_faucet_fill_time_l2313_231300

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end NUMINAMATH_GPT_faucet_fill_time_l2313_231300


namespace NUMINAMATH_GPT_bob_same_color_probability_is_1_over_28_l2313_231340

def num_marriages : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

def david_marbles : ℕ := 3
def alice_marbles : ℕ := 3
def bob_marbles : ℕ := 3

def total_ways : ℕ := 1680
def favorable_ways : ℕ := 60
def probability_bob_same_color := favorable_ways / total_ways

theorem bob_same_color_probability_is_1_over_28 : probability_bob_same_color = (1 : ℚ) / 28 := by
  sorry

end NUMINAMATH_GPT_bob_same_color_probability_is_1_over_28_l2313_231340


namespace NUMINAMATH_GPT_problem_f_x_sum_neg_l2313_231316

open Function

-- Definitions for monotonic decreasing and odd properties of the function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isMonotonicallyDecreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f y ≤ f x

-- The main theorem to prove
theorem problem_f_x_sum_neg
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_monotone : isMonotonicallyDecreasing f)
  (x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ + x₂ > 0)
  (h₂ : x₂ + x₃ > 0)
  (h₃ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_f_x_sum_neg_l2313_231316


namespace NUMINAMATH_GPT_total_plate_combinations_l2313_231382

open Nat

def valid_letters := 24
def letter_positions := (choose 4 2)
def valid_digits := 10
def total_combinations := letter_positions * (valid_letters * valid_letters) * (valid_digits ^ 3)

theorem total_plate_combinations : total_combinations = 3456000 :=
  by
    -- Replace this sorry with steps to prove the theorem
    sorry

end NUMINAMATH_GPT_total_plate_combinations_l2313_231382


namespace NUMINAMATH_GPT_relationship_between_coefficients_l2313_231361

theorem relationship_between_coefficients
  (b c : ℝ)
  (h_discriminant : b^2 - 4 * c ≥ 0)
  (h_root_condition : ∃ x1 x2 : ℝ, x1^2 = -x2 ∧ x1 + x2 = -b ∧ x1 * x2 = c):
  b^3 - 3 * b * c - c^2 - c = 0 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_coefficients_l2313_231361


namespace NUMINAMATH_GPT_kolacky_bounds_l2313_231377

theorem kolacky_bounds (x y : ℕ) (h : 9 * x + 4 * y = 219) :
  294 ≤ 12 * x + 6 * y ∧ 12 * x + 6 * y ≤ 324 :=
sorry

end NUMINAMATH_GPT_kolacky_bounds_l2313_231377


namespace NUMINAMATH_GPT_cube_of_720_diamond_1001_l2313_231333

-- Define the operation \diamond
def diamond (a b : ℕ) : ℕ :=
  (Nat.factors (a * b)).toFinset.card

-- Define the specific numbers 720 and 1001
def n1 : ℕ := 720
def n2 : ℕ := 1001

-- Calculate the cubic of the result of diamond operation
def cube_of_diamond : ℕ := (diamond n1 n2) ^ 3

-- The statement to be proved
theorem cube_of_720_diamond_1001 : cube_of_diamond = 216 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_of_720_diamond_1001_l2313_231333


namespace NUMINAMATH_GPT_min_red_chips_l2313_231315

variable (w b r : ℕ)

theorem min_red_chips :
  (b ≥ w / 3) → (b ≤ r / 4) → (w + b ≥ 70) → r ≥ 72 :=
by
  sorry

end NUMINAMATH_GPT_min_red_chips_l2313_231315


namespace NUMINAMATH_GPT_points_per_round_l2313_231304

def total_points : ℕ := 78
def num_rounds : ℕ := 26

theorem points_per_round : total_points / num_rounds = 3 := by
  sorry

end NUMINAMATH_GPT_points_per_round_l2313_231304


namespace NUMINAMATH_GPT_distribute_seedlings_l2313_231381

noncomputable def box_contents : List ℕ := [28, 51, 135, 67, 123, 29, 56, 38, 79]

def total_seedlings (contents : List ℕ) : ℕ := contents.sum

def obtainable_by_sigmas (contents : List ℕ) (σs : List ℕ) : Prop :=
  ∃ groups : List (List ℕ),
    (groups.length = σs.length) ∧
    (∀ g ∈ groups, contents.contains g.sum) ∧
    (∀ g, g ∈ groups → g.sum ∈ σs)

theorem distribute_seedlings : 
  total_seedlings box_contents = 606 →
  obtainable_by_sigmas box_contents [202, 202, 202] ∧
  ∃ way1 way2 : List (List ℕ),
    (way1 ≠ way2) ∧
    (obtainable_by_sigmas box_contents [202, 202, 202]) :=
by
  sorry

end NUMINAMATH_GPT_distribute_seedlings_l2313_231381


namespace NUMINAMATH_GPT_max_2a_b_2c_l2313_231301

theorem max_2a_b_2c (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 2 * a + b + 2 * c ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_2a_b_2c_l2313_231301


namespace NUMINAMATH_GPT_probability_auntie_em_can_park_l2313_231370

/-- A parking lot has 20 spaces in a row. -/
def total_spaces : ℕ := 20

/-- Fifteen cars arrive, each requiring one parking space, and their drivers choose spaces at random from among the available spaces. -/
def cars : ℕ := 15

/-- Auntie Em's SUV requires 3 adjacent empty spaces. -/
def required_adjacent_spaces : ℕ := 3

/-- Calculate the probability that there are 3 consecutive empty spaces among the 5 remaining spaces after 15 cars are parked in 20 spaces.
Expected answer is (12501 / 15504) -/
theorem probability_auntie_em_can_park : 
    (1 - (↑(Nat.choose 15 5) / ↑(Nat.choose 20 5))) = (12501 / 15504) := 
sorry

end NUMINAMATH_GPT_probability_auntie_em_can_park_l2313_231370


namespace NUMINAMATH_GPT_rectangular_prism_volume_is_60_l2313_231309

def rectangularPrismVolume (a b c : ℕ) : ℕ := a * b * c 

theorem rectangular_prism_volume_is_60 (a b c : ℕ) 
  (h_ge_2 : a ≥ 2) (h_ge_2_b : b ≥ 2) (h_ge_2_c : c ≥ 2)
  (h_one_face : 2 * ((a-2)*(b-2) + (b-2)*(c-2) + (a-2)*(c-2)) = 24)
  (h_two_faces : 4 * ((a-2) + (b-2) + (c-2)) = 28) :
  rectangularPrismVolume a b c = 60 := 
  by sorry

end NUMINAMATH_GPT_rectangular_prism_volume_is_60_l2313_231309


namespace NUMINAMATH_GPT_simplify_expression_l2313_231326

theorem simplify_expression (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  1 - (1 / (1 + (a^2 / (1 - a^2)))) = a^2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2313_231326


namespace NUMINAMATH_GPT_ababab_divisible_by_13_l2313_231369

theorem ababab_divisible_by_13 (a b : ℕ) (ha: a < 10) (hb: b < 10) : 
  13 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) := 
by
  sorry

end NUMINAMATH_GPT_ababab_divisible_by_13_l2313_231369


namespace NUMINAMATH_GPT_complement_M_l2313_231337

open Set

-- Definitions and conditions
def U : Set ℝ := univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Theorem stating the complement of M with respect to the universal set U
theorem complement_M : compl M = {x | x < -2 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_M_l2313_231337


namespace NUMINAMATH_GPT_license_plate_increase_factor_l2313_231391

def old_plate_count : ℕ := 26^2 * 10^3
def new_plate_count : ℕ := 26^4 * 10^4
def increase_factor : ℕ := new_plate_count / old_plate_count

theorem license_plate_increase_factor : increase_factor = 2600 :=
by
  unfold increase_factor
  rw [old_plate_count, new_plate_count]
  norm_num
  sorry

end NUMINAMATH_GPT_license_plate_increase_factor_l2313_231391


namespace NUMINAMATH_GPT_percent_of_N_in_M_l2313_231399

theorem percent_of_N_in_M (N M : ℝ) (hM : M ≠ 0) : (N / M) * 100 = 100 * N / M :=
by
  sorry

end NUMINAMATH_GPT_percent_of_N_in_M_l2313_231399


namespace NUMINAMATH_GPT_least_distance_on_cone_l2313_231373

noncomputable def least_distance_fly_could_crawl_cone (R C : ℝ) (slant_height : ℝ) (start_dist vertex_dist : ℝ) : ℝ :=
  if start_dist = 150 ∧ vertex_dist = 450 ∧ R = 500 ∧ C = 800 * Real.pi ∧ slant_height = R ∧ 
     (500 * (8 * Real.pi / 5) = 800 * Real.pi) then 600 else 0

theorem least_distance_on_cone : least_distance_fly_could_crawl_cone 500 (800 * Real.pi) 500 150 450 = 600 :=
by
  sorry

end NUMINAMATH_GPT_least_distance_on_cone_l2313_231373


namespace NUMINAMATH_GPT_work_completion_in_16_days_l2313_231363

theorem work_completion_in_16_days (A B : ℕ) :
  (1 / A + 1 / B = 1 / 40) → (10 * (1 / A + 1 / B) = 1 / 4) →
  (12 * 1 / A = 3 / 4) → A = 16 :=
by
  intros h1 h2 h3
  -- Proof is omitted by "sorry".
  sorry

end NUMINAMATH_GPT_work_completion_in_16_days_l2313_231363


namespace NUMINAMATH_GPT_innings_played_l2313_231387

noncomputable def cricket_player_innings : Nat :=
  let average_runs := 32
  let increase_in_average := 6
  let next_innings_runs := 158
  let new_average := average_runs + increase_in_average
  let runs_before_next_innings (n : Nat) := average_runs * n
  let total_runs_after_next_innings (n : Nat) := runs_before_next_innings n + next_innings_runs
  let total_runs_with_new_average (n : Nat) := new_average * (n + 1)

  let n := (total_runs_after_next_innings 20) - (total_runs_with_new_average 20)
  
  n
     
theorem innings_played : cricket_player_innings = 20 := by
  sorry

end NUMINAMATH_GPT_innings_played_l2313_231387


namespace NUMINAMATH_GPT_range_of_m_l2313_231351

theorem range_of_m (a m : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  m * (a + 1/a) / Real.sqrt 2 > 1 → m ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2313_231351


namespace NUMINAMATH_GPT_range_of_m_l2313_231366

def quadratic_nonnegative (m : ℝ) : Prop :=
∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0

theorem range_of_m (m : ℝ) :
  quadratic_nonnegative m ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2313_231366


namespace NUMINAMATH_GPT_cost_per_minute_l2313_231390

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_minute_l2313_231390


namespace NUMINAMATH_GPT_count_students_in_meets_l2313_231335

theorem count_students_in_meets (A B : Finset ℕ) (hA : A.card = 13) (hB : B.card = 12) (hAB : (A ∩ B).card = 6) :
  (A ∪ B).card = 19 :=
by
  sorry

end NUMINAMATH_GPT_count_students_in_meets_l2313_231335


namespace NUMINAMATH_GPT_problem1_problem2_l2313_231332

-- Problem (1): Maximum value of (a + 1/a)(b + 1/b)
theorem problem1 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≤ 25 / 4 := 
sorry

-- Problem (2): Minimum value of u = (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3
theorem problem2 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2313_231332


namespace NUMINAMATH_GPT_average_speed_round_trip_l2313_231368

variable (D : ℝ) (u v : ℝ)
  
theorem average_speed_round_trip (h1 : u = 96) (h2 : v = 88) : 
  (2 * u * v) / (u + v) = 91.73913043 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l2313_231368


namespace NUMINAMATH_GPT_product_of_two_numbers_l2313_231364

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2313_231364


namespace NUMINAMATH_GPT_mark_weekly_reading_l2313_231355

-- Using the identified conditions
def daily_reading_hours : ℕ := 2
def additional_weekly_hours : ℕ := 4

-- Prove the total number of hours Mark wants to read per week is 18 hours
theorem mark_weekly_reading : (daily_reading_hours * 7 + additional_weekly_hours) = 18 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_mark_weekly_reading_l2313_231355


namespace NUMINAMATH_GPT_eagles_per_section_l2313_231346

theorem eagles_per_section (total_eagles sections : ℕ) (h1 : total_eagles = 18) (h2 : sections = 3) :
  total_eagles / sections = 6 := by
  sorry

end NUMINAMATH_GPT_eagles_per_section_l2313_231346


namespace NUMINAMATH_GPT_sides_of_triangle_expr_negative_l2313_231306

theorem sides_of_triangle_expr_negative (a b c : ℝ) 
(h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
(a - c)^2 - b^2 < 0 :=
sorry

end NUMINAMATH_GPT_sides_of_triangle_expr_negative_l2313_231306


namespace NUMINAMATH_GPT_tan_A_in_right_triangle_l2313_231311

theorem tan_A_in_right_triangle (AC : ℝ) (AB : ℝ) (BC : ℝ) (hAC : AC = Real.sqrt 20) (hAB : AB = 4) (h_right_triangle : AC^2 = AB^2 + BC^2) :
  Real.tan (Real.arcsin (AB / AC)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_A_in_right_triangle_l2313_231311


namespace NUMINAMATH_GPT_decimal_89_to_binary_l2313_231336

def decimal_to_binary (n : ℕ) : ℕ := sorry

theorem decimal_89_to_binary :
  decimal_to_binary 89 = 1011001 :=
sorry

end NUMINAMATH_GPT_decimal_89_to_binary_l2313_231336


namespace NUMINAMATH_GPT_total_molecular_weight_of_products_l2313_231348

/-- Problem Statement: Determine the total molecular weight of the products formed when
    8 moles of Copper(II) carbonate (CuCO3) react with 6 moles of Diphosphorus pentoxide (P4O10)
    to form Copper(II) phosphate (Cu3(PO4)2) and Carbon dioxide (CO2). -/
theorem total_molecular_weight_of_products 
  (moles_CuCO3 : ℕ) 
  (moles_P4O10 : ℕ)
  (atomic_weight_Cu : ℝ := 63.55)
  (atomic_weight_P : ℝ := 30.97)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_C : ℝ := 12.01)
  (molecular_weight_CuCO3 : ℝ := atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O)
  (molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O)
  (molecular_weight_Cu3PO4_2 : ℝ := (3 * atomic_weight_Cu) + (2 * atomic_weight_P) + (8 * atomic_weight_O))
  (moles_Cu3PO4_2_formed : ℝ := (8 : ℝ) / 3)
  (moles_CO2_formed : ℝ := 8)
  (total_molecular_weight_Cu3PO4_2 : ℝ := moles_Cu3PO4_2_formed * molecular_weight_Cu3PO4_2)
  (total_molecular_weight_CO2 : ℝ := moles_CO2_formed * molecular_weight_CO2) : 
  (total_molecular_weight_Cu3PO4_2 + total_molecular_weight_CO2) = 1368.45 := by
  sorry

end NUMINAMATH_GPT_total_molecular_weight_of_products_l2313_231348


namespace NUMINAMATH_GPT_exists_n_not_coprime_l2313_231362

theorem exists_n_not_coprime (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : q > p) (h3 : q - p > 1) :
  ∃ (n : ℕ), Nat.gcd (p + n) (q + n) ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_n_not_coprime_l2313_231362


namespace NUMINAMATH_GPT_pie_filling_cans_l2313_231331

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end NUMINAMATH_GPT_pie_filling_cans_l2313_231331


namespace NUMINAMATH_GPT_ordered_pair_solution_l2313_231389

theorem ordered_pair_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end NUMINAMATH_GPT_ordered_pair_solution_l2313_231389


namespace NUMINAMATH_GPT_anna_reading_time_l2313_231375

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end NUMINAMATH_GPT_anna_reading_time_l2313_231375


namespace NUMINAMATH_GPT_glass_original_water_l2313_231338

theorem glass_original_water 
  (O : ℝ)  -- Ounces of water originally in the glass
  (evap_per_day : ℝ)  -- Ounces of water evaporated per day
  (total_days : ℕ)    -- Total number of days evaporation occurs
  (percent_evaporated : ℝ)  -- Percentage of the original amount that evaporated
  (h1 : evap_per_day = 0.06)  -- 0.06 ounces of water evaporated each day
  (h2 : total_days = 20)  -- Evaporation occurred over a period of 20 days
  (h3 : percent_evaporated = 0.12)  -- 12% of the original amount evaporated during this period
  (h4 : evap_per_day * total_days = 1.2)  -- 0.06 ounces per day for 20 days total gives 1.2 ounces
  (h5 : percent_evaporated * O = evap_per_day * total_days) :  -- 1.2 ounces is 12% of the original amount
  O = 10 :=  -- Prove that the original amount is 10 ounces
sorry

end NUMINAMATH_GPT_glass_original_water_l2313_231338


namespace NUMINAMATH_GPT_divides_both_numerator_and_denominator_l2313_231308

theorem divides_both_numerator_and_denominator (x m : ℤ) :
  (x ∣ (5 * m + 6)) ∧ (x ∣ (8 * m + 7)) → (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13) :=
by
  sorry

end NUMINAMATH_GPT_divides_both_numerator_and_denominator_l2313_231308


namespace NUMINAMATH_GPT_min_sum_of_factors_l2313_231342

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l2313_231342


namespace NUMINAMATH_GPT_product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l2313_231345

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end NUMINAMATH_GPT_product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l2313_231345


namespace NUMINAMATH_GPT_minimum_value_inequality_l2313_231349

theorem minimum_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (Real.sqrt ((x^2 + 4 * y^2) * (2 * x^2 + 3 * y^2)) / (x * y)) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l2313_231349


namespace NUMINAMATH_GPT_value_of_a_l2313_231324

-- Conditions
def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

-- Theorem statement asserting the condition and the correct answer
theorem value_of_a (a : ℝ) : (A a ∩ B a).Nonempty → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2313_231324


namespace NUMINAMATH_GPT_volume_of_each_cube_is_correct_l2313_231350

def box_length : ℕ := 12
def box_width : ℕ := 16
def box_height : ℕ := 6
def total_volume : ℕ := 1152
def number_of_cubes : ℕ := 384

theorem volume_of_each_cube_is_correct :
  (total_volume / number_of_cubes = 3) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_each_cube_is_correct_l2313_231350


namespace NUMINAMATH_GPT_Mary_put_crayons_l2313_231302

def initial_crayons : ℕ := 7
def final_crayons : ℕ := 10
def added_crayons (i f : ℕ) : ℕ := f - i

theorem Mary_put_crayons :
  added_crayons initial_crayons final_crayons = 3 := 
by
  sorry

end NUMINAMATH_GPT_Mary_put_crayons_l2313_231302


namespace NUMINAMATH_GPT_e_exp_f_neg2_l2313_231388

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
axiom h_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_ln_pos : ∀ x : ℝ, x > 0 → f x = Real.log x

-- Theorem to prove:
theorem e_exp_f_neg2 : Real.exp (f (-2)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_e_exp_f_neg2_l2313_231388


namespace NUMINAMATH_GPT_min_val_of_a2_plus_b2_l2313_231383

variable (a b : ℝ)

def condition := 3 * a - 4 * b - 2 = 0

theorem min_val_of_a2_plus_b2 : condition a b → (∃ a b : ℝ, a^2 + b^2 = 4 / 25) := by 
  sorry

end NUMINAMATH_GPT_min_val_of_a2_plus_b2_l2313_231383


namespace NUMINAMATH_GPT_jake_sister_weight_ratio_l2313_231344

theorem jake_sister_weight_ratio (Jake_initial_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (sister_weight : ℕ) 
(h₁ : Jake_initial_weight = 156) 
(h₂ : total_weight = 224) 
(h₃ : weight_loss = 20) 
(h₄ : total_weight = Jake_initial_weight + sister_weight) :
(Jake_initial_weight - weight_loss) / sister_weight = 2 := by
  sorry

end NUMINAMATH_GPT_jake_sister_weight_ratio_l2313_231344


namespace NUMINAMATH_GPT_basketball_team_points_l2313_231334

theorem basketball_team_points (total_points : ℕ) (number_of_players : ℕ) (points_per_player : ℕ) 
  (h1 : total_points = 18) (h2 : number_of_players = 9) : points_per_player = 2 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_basketball_team_points_l2313_231334


namespace NUMINAMATH_GPT_sum_of_decimals_l2313_231379

theorem sum_of_decimals : 1.000 + 0.101 + 0.011 + 0.001 = 1.113 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l2313_231379


namespace NUMINAMATH_GPT_average_weight_of_girls_l2313_231372

theorem average_weight_of_girls (avg_weight_boys : ℕ) (num_boys : ℕ) (avg_weight_class : ℕ) (num_students : ℕ) :
  num_boys = 15 →
  avg_weight_boys = 48 →
  num_students = 25 →
  avg_weight_class = 45 →
  ( (avg_weight_class * num_students - avg_weight_boys * num_boys) / (num_students - num_boys) ) = 27 :=
by
  intros h_num_boys h_avg_weight_boys h_num_students h_avg_weight_class
  sorry

end NUMINAMATH_GPT_average_weight_of_girls_l2313_231372


namespace NUMINAMATH_GPT_full_time_and_year_l2313_231307

variable (Total F Y N FY : ℕ)

theorem full_time_and_year (h1 : Total = 130)
                            (h2 : F = 80)
                            (h3 : Y = 100)
                            (h4 : N = 20)
                            (h5 : Total = FY + (F - FY) + (Y - FY) + N) :
    FY = 90 := 
sorry

end NUMINAMATH_GPT_full_time_and_year_l2313_231307


namespace NUMINAMATH_GPT_ellipse_equation_and_slope_range_l2313_231314

theorem ellipse_equation_and_slope_range (a b : ℝ) (e : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧
  ∃! ℓ : ℝ × ℝ, (ℓ.2 = 1 ∧ ℓ.1 = -2) ∧
  ∀ x y : ℝ, x^2 + y^2 = b^2 → y = x + 2 →
  ((x - 0)^2 + (y - 0)^2 = b^2) ∧
  (
    (a^2 = (3 * b^2)) ∧ (b = Real.sqrt 2) ∧
    a > 0 ∧
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) ∧
    (-((Real.sqrt 2) / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < ((Real.sqrt 2) / 2))
  ) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_and_slope_range_l2313_231314


namespace NUMINAMATH_GPT_variance_of_scores_l2313_231357

def scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

def mean (xs : List ℕ) : ℚ := xs.sum / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end NUMINAMATH_GPT_variance_of_scores_l2313_231357


namespace NUMINAMATH_GPT_beads_per_necklace_l2313_231395

theorem beads_per_necklace (n : ℕ) (b : ℕ) (total_beads : ℕ) (total_necklaces : ℕ)
  (h1 : total_necklaces = 6) (h2 : total_beads = 18) (h3 : b * total_necklaces = total_beads) :
  b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_beads_per_necklace_l2313_231395


namespace NUMINAMATH_GPT_smallest_integer_quad_ineq_l2313_231394

-- Definition of the condition
def quad_ineq (n : ℤ) := n^2 - 14 * n + 45 > 0

-- Lean 4 statement of the math proof problem
theorem smallest_integer_quad_ineq : ∃ n : ℤ, quad_ineq n ∧ ∀ m : ℤ, quad_ineq m → n ≤ m :=
  by
    existsi 10
    sorry

end NUMINAMATH_GPT_smallest_integer_quad_ineq_l2313_231394


namespace NUMINAMATH_GPT_days_spent_on_Orbius5_l2313_231365

-- Define the conditions
def days_per_year : Nat := 250
def seasons_per_year : Nat := 5
def length_of_season : Nat := days_per_year / seasons_per_year
def seasons_stayed : Nat := 3

-- Theorem statement
theorem days_spent_on_Orbius5 : (length_of_season * seasons_stayed = 150) :=
by 
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_days_spent_on_Orbius5_l2313_231365


namespace NUMINAMATH_GPT_boys_under_six_ratio_l2313_231392

theorem boys_under_six_ratio (total_students : ℕ) (two_third_boys : (2/3 : ℚ) * total_students = 25) (boys_under_six : ℕ) (boys_under_six_eq : boys_under_six = 19) :
  boys_under_six / 25 = 19 / 25 :=
by
  sorry

end NUMINAMATH_GPT_boys_under_six_ratio_l2313_231392


namespace NUMINAMATH_GPT_interest_rate_eq_five_percent_l2313_231374

def total_sum : ℝ := 2665
def P2 : ℝ := 1332.5
def P1 : ℝ := total_sum - P2

theorem interest_rate_eq_five_percent :
  (3 * 0.03 * P1 = r * 0.03 * P2) → r = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_eq_five_percent_l2313_231374


namespace NUMINAMATH_GPT_simplify_expression_l2313_231347

theorem simplify_expression : 20 * (9 / 14) * (1 / 18) = 5 / 7 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2313_231347


namespace NUMINAMATH_GPT_sequence_le_zero_l2313_231320

noncomputable def sequence_property (N : ℕ) (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧ (a N = 0) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2)

theorem sequence_le_zero {N : ℕ} (a : ℕ → ℝ) (h : sequence_property N a) : 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 :=
sorry

end NUMINAMATH_GPT_sequence_le_zero_l2313_231320


namespace NUMINAMATH_GPT_min_x_plus_y_l2313_231356

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y = 9 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_l2313_231356


namespace NUMINAMATH_GPT_determine_x_l2313_231325

-- Definitions based on conditions
variables {x : ℝ}

-- Problem statement
theorem determine_x (h : (6 * x)^5 = (18 * x)^4) (hx : x ≠ 0) : x = 27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l2313_231325


namespace NUMINAMATH_GPT_number_zero_points_eq_three_l2313_231371

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - x^2

theorem number_zero_points_eq_three : ∃ x1 x2 x3 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (∀ y : ℝ, f y = 0 → (y = x1 ∨ y = x2 ∨ y = x3)) :=
sorry

end NUMINAMATH_GPT_number_zero_points_eq_three_l2313_231371


namespace NUMINAMATH_GPT_range_of_m_l2313_231317

open Set Real

theorem range_of_m (M N : Set ℝ) (m : ℝ) :
    (M = {x | x ≤ m}) →
    (N = {y | ∃ x : ℝ, y = 2^(-x)}) →
    (M ∩ N ≠ ∅) → m > 0 := by
  intros hM hN hMN
  sorry

end NUMINAMATH_GPT_range_of_m_l2313_231317


namespace NUMINAMATH_GPT_flagpole_height_l2313_231397

theorem flagpole_height (h : ℕ) (shadow_flagpole : ℕ) (height_building : ℕ) (shadow_building : ℕ) (similar_conditions : Prop) 
  (H1 : shadow_flagpole = 45) 
  (H2 : height_building = 24) 
  (H3 : shadow_building = 60) 
  (H4 : similar_conditions) 
  (H5 : h / 45 = 24 / 60) : h = 18 := 
by 
sorry

end NUMINAMATH_GPT_flagpole_height_l2313_231397


namespace NUMINAMATH_GPT_smallest_possible_value_expression_l2313_231354

open Real

noncomputable def min_expression_value (a b c : ℝ) : ℝ :=
  (a + b)^2 + (b - c)^2 + (c - a)^2 / a^2

theorem smallest_possible_value_expression :
  ∀ (a b c : ℝ), a > b → b > c → a + c = 2 * b → a ≠ 0 → min_expression_value a b c = 7 / 2 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_expression_l2313_231354


namespace NUMINAMATH_GPT_line_contains_point_iff_k_eq_neg1_l2313_231380

theorem line_contains_point_iff_k_eq_neg1 (k : ℝ) :
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ (2 - k * x = -4 * y)) ↔ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_line_contains_point_iff_k_eq_neg1_l2313_231380


namespace NUMINAMATH_GPT_library_books_new_releases_l2313_231384

theorem library_books_new_releases (P Q R S : Prop) 
  (h : ¬P) 
  (P_iff_Q : P ↔ Q)
  (Q_implies_R : Q → R)
  (S_iff_notP : S ↔ ¬P) : 
  Q ∧ S := by 
  sorry

end NUMINAMATH_GPT_library_books_new_releases_l2313_231384


namespace NUMINAMATH_GPT_jennie_speed_difference_l2313_231360

noncomputable def average_speed_difference : ℝ :=
  let distance := 200
  let time_heavy_traffic := 5
  let construction_delay := 0.5
  let rest_stops_heavy := 0.5
  let time_no_traffic := 4
  let rest_stops_no_traffic := 1 / 3
  let actual_driving_time_heavy := time_heavy_traffic - construction_delay - rest_stops_heavy
  let actual_driving_time_no := time_no_traffic - rest_stops_no_traffic
  let average_speed_heavy := distance / actual_driving_time_heavy
  let average_speed_no := distance / actual_driving_time_no
  average_speed_no - average_speed_heavy

theorem jennie_speed_difference :
  average_speed_difference = 4.5 :=
sorry

end NUMINAMATH_GPT_jennie_speed_difference_l2313_231360


namespace NUMINAMATH_GPT_arcsin_zero_l2313_231393

theorem arcsin_zero : Real.arcsin 0 = 0 := by
  sorry

end NUMINAMATH_GPT_arcsin_zero_l2313_231393


namespace NUMINAMATH_GPT_rectangular_solid_dimension_change_l2313_231367

theorem rectangular_solid_dimension_change (a b : ℝ) (h : 2 * a^2 + 4 * a * b = 0.6 * (6 * a^2)) : b = 0.4 * a :=
by sorry

end NUMINAMATH_GPT_rectangular_solid_dimension_change_l2313_231367


namespace NUMINAMATH_GPT_problem1_problem2_l2313_231328

/-- Problem 1: Prove the solution to the system of equations is x = 1/2 and y = 5 -/
theorem problem1 (x y : ℚ) (h1 : 2 * x - y = -4) (h2 : 4 * x - 5 * y = -23) : 
  x = 1 / 2 ∧ y = 5 := 
sorry

/-- Problem 2: Prove the value of the expression (x-3y)^{2} - (2x+y)(y-2x) when x = 2 and y = -1 is 40 -/
theorem problem2 (x y : ℚ) (h1 : x = 2) (h2 : y = -1) : 
  (x - 3 * y) ^ 2 - (2 * x + y) * (y - 2 * x) = 40 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2313_231328


namespace NUMINAMATH_GPT_lcm_of_numbers_with_ratio_and_hcf_l2313_231319

theorem lcm_of_numbers_with_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : Nat.gcd a b = 3) : Nat.lcm a b = 36 := 
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_with_ratio_and_hcf_l2313_231319


namespace NUMINAMATH_GPT_jacob_younger_than_michael_l2313_231376

variables (J M : ℕ)

theorem jacob_younger_than_michael (h1 : M + 9 = 2 * (J + 9)) (h2 : J = 5) : M - J = 14 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_jacob_younger_than_michael_l2313_231376


namespace NUMINAMATH_GPT_math_problem_l2313_231341

theorem math_problem
  (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x^2 + y^2 = 18) :
  x^2 + y^2 = 18 :=
sorry

end NUMINAMATH_GPT_math_problem_l2313_231341


namespace NUMINAMATH_GPT_fg_eq_neg7_l2313_231327

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem fg_eq_neg7 : f (g 2) = -7 :=
  by
    sorry

end NUMINAMATH_GPT_fg_eq_neg7_l2313_231327


namespace NUMINAMATH_GPT_last_student_score_is_61_l2313_231353

noncomputable def average_score_19_students := 82
noncomputable def average_score_20_students := 84
noncomputable def total_students := 20
noncomputable def oliver_multiplier := 2

theorem last_student_score_is_61 
  (total_score_19_students : ℝ := total_students - 1 * average_score_19_students)
  (total_score_20_students : ℝ := total_students * average_score_20_students)
  (oliver_score : ℝ := total_score_20_students - total_score_19_students)
  (last_student_score : ℝ := oliver_score / oliver_multiplier) :
  last_student_score = 61 :=
sorry

end NUMINAMATH_GPT_last_student_score_is_61_l2313_231353


namespace NUMINAMATH_GPT_total_brownies_correct_l2313_231352

def brownies_initial : Nat := 24
def father_ate : Nat := brownies_initial / 3
def remaining_after_father : Nat := brownies_initial - father_ate
def mooney_ate : Nat := remaining_after_father / 4
def remaining_after_mooney : Nat := remaining_after_father - mooney_ate
def benny_ate : Nat := (remaining_after_mooney * 2) / 5
def remaining_after_benny : Nat := remaining_after_mooney - benny_ate
def snoopy_ate : Nat := 3
def remaining_after_snoopy : Nat := remaining_after_benny - snoopy_ate
def new_batch : Nat := 24
def total_brownies : Nat := remaining_after_snoopy + new_batch

theorem total_brownies_correct : total_brownies = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_brownies_correct_l2313_231352


namespace NUMINAMATH_GPT_arccos_cos_8_eq_1_point_72_l2313_231303

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end NUMINAMATH_GPT_arccos_cos_8_eq_1_point_72_l2313_231303


namespace NUMINAMATH_GPT_larger_model_ratio_smaller_model_ratio_l2313_231358

-- Definitions for conditions
def statue_height := 305 -- The height of the actual statue in feet
def larger_model_height := 10 -- The height of the larger model in inches
def smaller_model_height := 5 -- The height of the smaller model in inches

-- The ratio calculation for larger model
theorem larger_model_ratio : 
  (statue_height : ℝ) / (larger_model_height : ℝ) = 30.5 := by
  sorry

-- The ratio calculation for smaller model
theorem smaller_model_ratio : 
  (statue_height : ℝ) / (smaller_model_height : ℝ) = 61 := by
  sorry

end NUMINAMATH_GPT_larger_model_ratio_smaller_model_ratio_l2313_231358


namespace NUMINAMATH_GPT_number_of_real_roots_l2313_231322

theorem number_of_real_roots :
  ∃ (roots_count : ℕ), roots_count = 2 ∧
  (∀ x : ℝ, x^2 - |2 * x - 1| - 4 = 0 → (x = -1 - Real.sqrt 6 ∨ x = 3)) :=
sorry

end NUMINAMATH_GPT_number_of_real_roots_l2313_231322
