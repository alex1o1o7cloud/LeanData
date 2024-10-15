import Mathlib

namespace NUMINAMATH_GPT_find_side_length_l1203_120320

theorem find_side_length (a b c : ℝ) (A : ℝ) 
  (h1 : Real.cos A = 7 / 8) 
  (h2 : c - a = 2) 
  (h3 : b = 3) : 
  a = 2 := by
  sorry

end NUMINAMATH_GPT_find_side_length_l1203_120320


namespace NUMINAMATH_GPT_trigonometric_identity1_trigonometric_identity2_l1203_120315

theorem trigonometric_identity1 (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin (Real.pi - θ) + Real.cos (θ - Real.pi)) / (Real.sin (θ + Real.pi) + Real.cos (θ + Real.pi)) = -1/3 :=
by
  sorry

theorem trigonometric_identity2 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4/5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity1_trigonometric_identity2_l1203_120315


namespace NUMINAMATH_GPT_Traci_trip_fraction_l1203_120356

theorem Traci_trip_fraction :
  let total_distance := 600
  let first_stop_distance := total_distance / 3
  let remaining_distance_after_first_stop := total_distance - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  (distance_between_stops / remaining_distance_after_first_stop) = 1 / 4 :=
by
  let total_distance := 600
  let first_stop_distance := 600 / 3
  let remaining_distance_after_first_stop := 600 - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  have h1 : total_distance = 600 := by exact rfl
  have h2 : first_stop_distance = 200 := by norm_num [first_stop_distance]
  have h3 : remaining_distance_after_first_stop = 400 := by norm_num [remaining_distance_after_first_stop]
  have h4 : distance_between_stops = 100 := by norm_num [distance_between_stops]
  show (distance_between_stops / remaining_distance_after_first_stop) = 1/4
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Traci_trip_fraction_l1203_120356


namespace NUMINAMATH_GPT_max_value_M_l1203_120330

open Real

theorem max_value_M :
  ∃ M : ℝ, ∀ x y z u : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < u ∧ z ≥ y ∧ (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) →
  M ≤ z / y ∧ M = 6 + 4 * sqrt 2 := 
  sorry

end NUMINAMATH_GPT_max_value_M_l1203_120330


namespace NUMINAMATH_GPT_rectangular_prism_volume_l1203_120300

theorem rectangular_prism_volume 
(l w h : ℝ) 
(h1 : l * w = 18) 
(h2 : w * h = 32) 
(h3 : l * h = 48) : 
l * w * h = 288 :=
sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l1203_120300


namespace NUMINAMATH_GPT_bisection_next_interval_l1203_120317

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- Define the intervals (1, 2) and (1.5, 2)
def interval_initial : Set ℝ := {x | 1 < x ∧ x < 2}
def interval_next : Set ℝ := {x | 1.5 < x ∧ x < 2}

-- State the theorem, with conditions
theorem bisection_next_interval 
  (root_in_interval_initial : ∃ x, f x = 0 ∧ x ∈ interval_initial)
  (f_1_negative : f 1 < 0)
  (f_2_positive : f 2 > 0)
  : ∃ x, f x = 0 ∧ x ∈ interval_next :=
sorry

end NUMINAMATH_GPT_bisection_next_interval_l1203_120317


namespace NUMINAMATH_GPT_brooke_sidney_ratio_l1203_120377

-- Definitions for the conditions
def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50
def brooke_total : ℕ := 438

-- Total jumping jacks by Sidney
def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

-- The ratio of Brooke’s jumping jacks to Sidney's total jumping jacks
def ratio := brooke_total / sidney_total

-- The proof goal
theorem brooke_sidney_ratio : ratio = 3 :=
by
  sorry

end NUMINAMATH_GPT_brooke_sidney_ratio_l1203_120377


namespace NUMINAMATH_GPT_train_relative_speed_l1203_120367

-- Definitions of given conditions
def initialDistance : ℝ := 13
def speedTrainA : ℝ := 37
def speedTrainB : ℝ := 43

-- Definition of the relative speed
def relativeSpeed : ℝ := speedTrainB - speedTrainA

-- Theorem to prove the relative speed
theorem train_relative_speed
  (h1 : initialDistance = 13)
  (h2 : speedTrainA = 37)
  (h3 : speedTrainB = 43) :
  relativeSpeed = 6 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_train_relative_speed_l1203_120367


namespace NUMINAMATH_GPT_circles_and_squares_intersection_l1203_120362

def circles_and_squares_intersection_count : Nat :=
  let radius := (1 : ℚ) / 8
  let square_side := (1 : ℚ) / 4
  let slope := (1 : ℚ) / 3
  let line (x : ℚ) : ℚ := slope * x
  let num_segments := 243
  let intersections_per_segment := 4
  num_segments * intersections_per_segment

theorem circles_and_squares_intersection : 
  circles_and_squares_intersection_count = 972 :=
by
  sorry

end NUMINAMATH_GPT_circles_and_squares_intersection_l1203_120362


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1203_120305

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + (6 / 10) / 9) : x = 11 / 30 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1203_120305


namespace NUMINAMATH_GPT_prob_no_rain_correct_l1203_120355

-- Define the probability of rain on each of the next five days
def prob_rain_each_day : ℚ := 1 / 2

-- Define the probability of no rain on a single day
def prob_no_rain_one_day : ℚ := 1 - prob_rain_each_day

-- Define the probability of no rain in any of the next five days
def prob_no_rain_five_days : ℚ := prob_no_rain_one_day ^ 5

-- Theorem statement
theorem prob_no_rain_correct : prob_no_rain_five_days = 1 / 32 := by
  sorry

end NUMINAMATH_GPT_prob_no_rain_correct_l1203_120355


namespace NUMINAMATH_GPT_time_to_watch_all_episodes_l1203_120350

theorem time_to_watch_all_episodes 
    (n_seasons : ℕ) (episodes_per_season : ℕ) (last_season_extra_episodes : ℕ) (hours_per_episode : ℚ)
    (h1 : n_seasons = 9)
    (h2 : episodes_per_season = 22)
    (h3 : last_season_extra_episodes = 4)
    (h4 : hours_per_episode = 0.5) :
    n_seasons * episodes_per_season + (episodes_per_season + last_season_extra_episodes) * hours_per_episode = 112 :=
by
  sorry

end NUMINAMATH_GPT_time_to_watch_all_episodes_l1203_120350


namespace NUMINAMATH_GPT_findWorkRateB_l1203_120336

-- Define the work rates of A and C given in the problem
def workRateA : ℚ := 1 / 8
def workRateC : ℚ := 1 / 16

-- Combined work rate when A, B, and C work together to complete the work in 4 days
def combinedWorkRate : ℚ := 1 / 4

-- Define the work rate of B that we need to prove
def workRateB : ℚ := 1 / 16

-- Theorem to prove that workRateB is equal to B's work rate given the conditions
theorem findWorkRateB : workRateA + workRateB + workRateC = combinedWorkRate :=
  by
  sorry

end NUMINAMATH_GPT_findWorkRateB_l1203_120336


namespace NUMINAMATH_GPT_squirrel_acorns_left_l1203_120301

noncomputable def acorns_per_winter_month (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) : ℕ :=
  let per_month := total_acorns / months
  let acorns_taken_per_month := acorns_taken_total / months
  per_month - acorns_taken_per_month

theorem squirrel_acorns_left (total_acorns : ℕ) (months : ℕ) (acorns_taken_total : ℕ) :
  total_acorns = 210 → months = 3 → acorns_taken_total = 30 → acorns_per_winter_month total_acorns months acorns_taken_total = 60 :=
by intros; sorry

end NUMINAMATH_GPT_squirrel_acorns_left_l1203_120301


namespace NUMINAMATH_GPT_sam_digits_memorized_l1203_120344

-- Definitions
def carlos_memorized (c : ℕ) := (c * 6 = 24)
def sam_memorized (s c : ℕ) := (s = c + 6)
def mina_memorized := 24

-- Theorem
theorem sam_digits_memorized (s c : ℕ) (h_c : carlos_memorized c) (h_s : sam_memorized s c) : s = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_sam_digits_memorized_l1203_120344


namespace NUMINAMATH_GPT_find_x0_l1203_120351

-- Define the given conditions
variable (p x_0 : ℝ) (P : ℝ × ℝ) (O : ℝ × ℝ)
variable (h_parabola : x_0^2 = 2 * p * 1)
variable (h_p_gt_zero : p > 0)
variable (h_point_P : P = (x_0, 1))
variable (h_origin : O = (0, 0))
variable (h_distance_condition : dist (x_0, 1) (0, 0) = dist (x_0, 1) (0, -p / 2))

-- The theorem we aim to prove
theorem find_x0 : x_0 = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_find_x0_l1203_120351


namespace NUMINAMATH_GPT_solve_system_l1203_120333

variables (a b c d : ℝ)

theorem solve_system :
  (a + c = -4) ∧
  (a * c + b + d = 6) ∧
  (a * d + b * c = -5) ∧
  (b * d = 2) →
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) :=
by
  intro h
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_solve_system_l1203_120333


namespace NUMINAMATH_GPT_cubic_expression_equals_two_l1203_120352

theorem cubic_expression_equals_two (x : ℝ) (h : 2 * x ^ 2 - 3 * x - 2022 = 0) :
  2 * x ^ 3 - x ^ 2 - 2025 * x - 2020 = 2 :=
sorry

end NUMINAMATH_GPT_cubic_expression_equals_two_l1203_120352


namespace NUMINAMATH_GPT_Bill_tossed_objects_l1203_120364

theorem Bill_tossed_objects (Ted_sticks Ted_rocks Bill_sticks Bill_rocks : ℕ)
  (h1 : Bill_sticks = Ted_sticks + 6)
  (h2 : Ted_rocks = 2 * Bill_rocks)
  (h3 : Ted_sticks = 10)
  (h4 : Ted_rocks = 10) :
  Bill_sticks + Bill_rocks = 21 :=
by
  sorry

end NUMINAMATH_GPT_Bill_tossed_objects_l1203_120364


namespace NUMINAMATH_GPT_participation_schemes_count_l1203_120316

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end NUMINAMATH_GPT_participation_schemes_count_l1203_120316


namespace NUMINAMATH_GPT_polynomial_division_result_q_neg1_r_1_sum_l1203_120384

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 5 * x^3 - 4 * x^2 + 2 * x + 1
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 + x
noncomputable def r (x : ℝ) : ℝ := 7 * x + 4

theorem polynomial_division_result : f (-1) = q (-1) * d (-1) + r (-1)
  ∧ f 1 = q 1 * d 1 + r 1 :=
by sorry

theorem q_neg1_r_1_sum : (q (-1) + r 1) = 13 :=
by sorry

end NUMINAMATH_GPT_polynomial_division_result_q_neg1_r_1_sum_l1203_120384


namespace NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l1203_120353

theorem isosceles_triangle_congruent_side_length
  (B : ℕ) (A : ℕ) (P : ℕ) (L : ℕ)
  (h₁ : B = 36) (h₂ : A = 108) (h₃ : P = 84) :
  L = 24 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l1203_120353


namespace NUMINAMATH_GPT_three_scientists_same_topic_l1203_120302

theorem three_scientists_same_topic
  (scientists : Finset ℕ)
  (h_size : scientists.card = 17)
  (topics : Finset ℕ)
  (h_topics : topics.card = 3)
  (communicates : ℕ → ℕ → ℕ)
  (h_communicate : ∀ a b : ℕ, a ≠ b → b ∈ scientists → communicates a b ∈ topics) :
  ∃ (a b c : ℕ), a ∈ scientists ∧ b ∈ scientists ∧ c ∈ scientists ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  communicates a b = communicates b c ∧ communicates b c = communicates a c := 
sorry

end NUMINAMATH_GPT_three_scientists_same_topic_l1203_120302


namespace NUMINAMATH_GPT_smallest_next_divisor_l1203_120310

theorem smallest_next_divisor (m : ℕ) (h_digit : 10000 ≤ m ∧ m < 100000) (h_odd : m % 2 = 1) (h_div : 437 ∣ m) :
  ∃ d : ℕ, 437 < d ∧ d ∣ m ∧ (∀ e : ℕ, 437 < e ∧ e < d → ¬ e ∣ m) ∧ d = 475 := 
sorry

end NUMINAMATH_GPT_smallest_next_divisor_l1203_120310


namespace NUMINAMATH_GPT_vertical_increase_is_100m_l1203_120325

theorem vertical_increase_is_100m 
  (a b x : ℝ)
  (hypotenuse : a = 100 * Real.sqrt 5)
  (slope_ratio : b = 2 * x)
  (pythagorean_thm : x^2 + b^2 = a^2) : 
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_vertical_increase_is_100m_l1203_120325


namespace NUMINAMATH_GPT_area_of_circle_l1203_120386

theorem area_of_circle (r : ℝ) (h : r = 3) : 
  (∀ A : ℝ, A = π * r^2) → A = 9 * π :=
by
  intro area_formula
  sorry

end NUMINAMATH_GPT_area_of_circle_l1203_120386


namespace NUMINAMATH_GPT_rope_length_third_post_l1203_120388

theorem rope_length_third_post (total first second fourth : ℕ) (h_total : total = 70) 
    (h_first : first = 24) (h_second : second = 20) (h_fourth : fourth = 12) : 
    (total - first - second - fourth) = 14 :=
by
  -- Proof is skipped, but we can state that the theorem should follow from the given conditions.
  sorry

end NUMINAMATH_GPT_rope_length_third_post_l1203_120388


namespace NUMINAMATH_GPT_pencils_before_buying_l1203_120341

theorem pencils_before_buying (x total bought : Nat) 
  (h1 : bought = 7) 
  (h2 : total = 10) 
  (h3 : total = x + bought) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_pencils_before_buying_l1203_120341


namespace NUMINAMATH_GPT_value_of_f_at_2_l1203_120319

def f (x : ℝ) : ℝ :=
  x^3 - x - 1

theorem value_of_f_at_2 : f 2 = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l1203_120319


namespace NUMINAMATH_GPT_volume_at_20_deg_l1203_120348

theorem volume_at_20_deg
  (ΔV_per_ΔT : ∀ ΔT : ℕ, ΔT = 5 → ∀ V : ℕ, V = 5)
  (initial_condition : ∀ V : ℕ, V = 40 ∧ ∀ T : ℕ, T = 40) :
  ∃ V : ℕ, V = 20 :=
by
  sorry

end NUMINAMATH_GPT_volume_at_20_deg_l1203_120348


namespace NUMINAMATH_GPT_solve_quadratic_completing_square_l1203_120326

theorem solve_quadratic_completing_square (x : ℝ) :
  x^2 - 4 * x + 3 = 0 → (x - 2)^2 = 1 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_completing_square_l1203_120326


namespace NUMINAMATH_GPT_num_students_play_cricket_l1203_120324

theorem num_students_play_cricket 
  (total_students : ℕ)
  (play_football : ℕ)
  (play_both : ℕ)
  (play_neither : ℕ)
  (C : ℕ) :
  total_students = 450 →
  play_football = 325 →
  play_both = 100 →
  play_neither = 50 →
  (total_students - play_neither = play_football + C - play_both) →
  C = 175 := by
  intros h0 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_num_students_play_cricket_l1203_120324


namespace NUMINAMATH_GPT_point_on_x_axis_l1203_120338

theorem point_on_x_axis (x : ℝ) (A : ℝ × ℝ) (h : A = (2 - x, x + 3)) (hy : A.snd = 0) : A = (5, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1203_120338


namespace NUMINAMATH_GPT_sqrt_expression_meaningful_domain_l1203_120375

theorem sqrt_expression_meaningful_domain {x : ℝ} (h : 3 - x ≥ 0) : x ≤ 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_meaningful_domain_l1203_120375


namespace NUMINAMATH_GPT_sin_transform_l1203_120335

theorem sin_transform (θ : ℝ) (h : Real.sin (θ - π / 12) = 3 / 4) :
  Real.sin (2 * θ + π / 3) = -1 / 8 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_sin_transform_l1203_120335


namespace NUMINAMATH_GPT_initial_girls_count_l1203_120328

-- Define the variables
variables (b g : ℕ)

-- Conditions
def condition1 := b = 3 * (g - 20)
def condition2 := 4 * (b - 60) = g - 20

-- Statement of the problem
theorem initial_girls_count
  (h1 : condition1 b g)
  (h2 : condition2 b g) : g = 460 / 11 := 
sorry

end NUMINAMATH_GPT_initial_girls_count_l1203_120328


namespace NUMINAMATH_GPT_sequence_first_last_four_equal_l1203_120323

theorem sequence_first_last_four_equal (S : List ℕ) (n : ℕ)
  (hS : S.length = n)
  (h_max : ∀ T : List ℕ, (∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                        (S.drop i).take 5 ≠ (S.drop j).take 5) → T.length ≤ n)
  (h_distinct : ∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                (S.drop i).take 5 ≠ (S.drop j).take 5) :
  (S.take 4 = S.drop (n-4)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_first_last_four_equal_l1203_120323


namespace NUMINAMATH_GPT_customers_remaining_l1203_120389

theorem customers_remaining (init : ℕ) (left : ℕ) (remaining : ℕ) :
  init = 21 → left = 9 → remaining = 12 → init - left = remaining :=
by sorry

end NUMINAMATH_GPT_customers_remaining_l1203_120389


namespace NUMINAMATH_GPT_emptying_tank_time_l1203_120343

theorem emptying_tank_time :
  let V := 30 * 12^3 -- volume of the tank in cubic inches
  let r_in := 3 -- rate of inlet pipe in cubic inches per minute
  let r_out1 := 12 -- rate of first outlet pipe in cubic inches per minute
  let r_out2 := 6 -- rate of second outlet pipe in cubic inches per minute
  let net_rate := r_out1 + r_out2 - r_in
  V / net_rate = 3456 := by
sorry

end NUMINAMATH_GPT_emptying_tank_time_l1203_120343


namespace NUMINAMATH_GPT_factory_sample_size_l1203_120397

noncomputable def sample_size (A B C : ℕ) (sample_A : ℕ) : ℕ :=
  let total_ratio := A + B + C
  let ratio_A := A / total_ratio
  sample_A / ratio_A

theorem factory_sample_size
  (A B C : ℕ) (h_ratio : A = 2 ∧ B = 3 ∧ C = 5)
  (sample_A : ℕ) (h_sample_A : sample_A = 16) :
  sample_size A B C sample_A = 80 :=
by
  simp [h_ratio, h_sample_A, sample_size]
  sorry

end NUMINAMATH_GPT_factory_sample_size_l1203_120397


namespace NUMINAMATH_GPT_g_one_third_value_l1203_120373

noncomputable def g : ℚ → ℚ := sorry

theorem g_one_third_value : (∀ (x : ℚ), x ≠ 0 → (4 * g (1 / x) + 3 * g x / x^2 = x^3)) → g (1 / 3) = 21 / 44 := by
  intro h
  sorry

end NUMINAMATH_GPT_g_one_third_value_l1203_120373


namespace NUMINAMATH_GPT_PQ_relationship_l1203_120354

-- Define the sets P and Q
def P := {x : ℝ | x >= 5}
def Q := {x : ℝ | 5 <= x ∧ x <= 7}

-- Statement to be proved
theorem PQ_relationship : Q ⊆ P ∧ Q ≠ P :=
by
  sorry

end NUMINAMATH_GPT_PQ_relationship_l1203_120354


namespace NUMINAMATH_GPT_largest_square_multiple_of_18_under_500_l1203_120361

theorem largest_square_multiple_of_18_under_500 : 
  ∃ n : ℕ, n * n < 500 ∧ n * n % 18 = 0 ∧ (∀ m : ℕ, m * m < 500 ∧ m * m % 18 = 0 → m * m ≤ n * n) → 
  n * n = 324 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_multiple_of_18_under_500_l1203_120361


namespace NUMINAMATH_GPT_neg_and_implication_l1203_120359

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end NUMINAMATH_GPT_neg_and_implication_l1203_120359


namespace NUMINAMATH_GPT_two_digit_multiples_of_6_and_9_l1203_120369

theorem two_digit_multiples_of_6_and_9 : ∃ n : ℕ, n = 5 ∧ (∀ k : ℤ, 10 ≤ k ∧ k < 100 ∧ (k % 6 = 0) ∧ (k % 9 = 0) → 
    k = 18 ∨ k = 36 ∨ k = 54 ∨ k = 72 ∨ k = 90) := 
sorry

end NUMINAMATH_GPT_two_digit_multiples_of_6_and_9_l1203_120369


namespace NUMINAMATH_GPT_no_solutions_l1203_120306

theorem no_solutions (x y : ℤ) (h : 8 * x + 3 * y^2 = 5) : False :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_l1203_120306


namespace NUMINAMATH_GPT_set_equivalence_l1203_120349

variable (M : Set ℕ)

theorem set_equivalence (h : M ∪ {1} = {1, 2, 3}) : M = {1, 2, 3} :=
sorry

end NUMINAMATH_GPT_set_equivalence_l1203_120349


namespace NUMINAMATH_GPT_blake_total_expenditure_l1203_120311

noncomputable def total_cost (rooms : ℕ) (primer_cost : ℝ) (paint_cost : ℝ) (primer_discount : ℝ) : ℝ :=
  let primer_needed := rooms
  let paint_needed := rooms
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := primer_needed * discounted_primer_cost
  let total_paint_cost := paint_needed * paint_cost
  total_primer_cost + total_paint_cost

theorem blake_total_expenditure :
  total_cost 5 30 25 0.20 = 245 := 
by
  sorry

end NUMINAMATH_GPT_blake_total_expenditure_l1203_120311


namespace NUMINAMATH_GPT_pool_filling_times_l1203_120383

theorem pool_filling_times:
  ∃ (x y z u : ℕ),
    (1/x + 1/y = 1/70) ∧
    (1/x + 1/z = 1/84) ∧
    (1/y + 1/z = 1/140) ∧
    (1/u = 1/x + 1/y + 1/z) ∧
    (x = 105) ∧
    (y = 210) ∧
    (z = 420) ∧
    (u = 60) := 
  sorry

end NUMINAMATH_GPT_pool_filling_times_l1203_120383


namespace NUMINAMATH_GPT_find_k_l1203_120357

theorem find_k (a b k : ℝ) (h1 : a ≠ b ∨ a = b)
    (h2 : a^2 - 12 * a + k + 2 = 0)
    (h3 : b^2 - 12 * b + k + 2 = 0)
    (h4 : 4^2 - 12 * 4 + k + 2 = 0) :
    k = 34 ∨ k = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1203_120357


namespace NUMINAMATH_GPT_fewer_bees_than_flowers_l1203_120370

theorem fewer_bees_than_flowers : 5 - 3 = 2 := by
  sorry

end NUMINAMATH_GPT_fewer_bees_than_flowers_l1203_120370


namespace NUMINAMATH_GPT_solve_for_m_l1203_120360

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 / (2^x + 1)) + m

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) ↔ m = -1 := by
sorry

end NUMINAMATH_GPT_solve_for_m_l1203_120360


namespace NUMINAMATH_GPT_remainder_of_sum_l1203_120392

theorem remainder_of_sum (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1203_120392


namespace NUMINAMATH_GPT_combined_platforms_length_is_correct_l1203_120391

noncomputable def combined_length_of_platforms (lengthA lengthB speedA_kmph speedB_kmph timeA_sec timeB_sec : ℝ) : ℝ :=
  let speedA := speedA_kmph * (1000 / 3600)
  let speedB := speedB_kmph * (1000 / 3600)
  let distanceA := speedA * timeA_sec
  let distanceB := speedB * timeB_sec
  let platformA := distanceA - lengthA
  let platformB := distanceB - lengthB
  platformA + platformB

theorem combined_platforms_length_is_correct :
  combined_length_of_platforms 650 450 115 108 30 25 = 608.32 := 
by 
  sorry

end NUMINAMATH_GPT_combined_platforms_length_is_correct_l1203_120391


namespace NUMINAMATH_GPT_cos_90_eq_0_l1203_120387

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end NUMINAMATH_GPT_cos_90_eq_0_l1203_120387


namespace NUMINAMATH_GPT_child_haircut_cost_l1203_120313

/-
Problem Statement:
- Women's haircuts cost $48.
- Tayzia and her two daughters get haircuts.
- Tayzia wants to give a 20% tip to the hair stylist, which amounts to $24.
Question: How much does a child's haircut cost?
-/

noncomputable def cost_of_child_haircut (C : ℝ) : Prop :=
  let women's_haircut := 48
  let tip := 24
  let total_cost_before_tip := women's_haircut + 2 * C
  total_cost_before_tip * 0.20 = tip ∧ total_cost_before_tip = 120 ∧ C = 36

theorem child_haircut_cost (C : ℝ) (h1 : cost_of_child_haircut C) : C = 36 :=
  by sorry

end NUMINAMATH_GPT_child_haircut_cost_l1203_120313


namespace NUMINAMATH_GPT_andy_more_candies_than_caleb_l1203_120312

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_andy_more_candies_than_caleb_l1203_120312


namespace NUMINAMATH_GPT_carrie_profit_l1203_120395

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end NUMINAMATH_GPT_carrie_profit_l1203_120395


namespace NUMINAMATH_GPT_total_cases_after_third_day_l1203_120363

-- Definitions for the conditions
def day1_cases : Nat := 2000
def day2_new_cases : Nat := 500
def day2_recoveries : Nat := 50
def day3_new_cases : Nat := 1500
def day3_recoveries : Nat := 200

-- Theorem stating the total number of cases after the third day
theorem total_cases_after_third_day : day1_cases + (day2_new_cases - day2_recoveries) + (day3_new_cases - day3_recoveries) = 3750 :=
by
  sorry

end NUMINAMATH_GPT_total_cases_after_third_day_l1203_120363


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1203_120327

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1203_120327


namespace NUMINAMATH_GPT_range_of_m_l1203_120393

theorem range_of_m (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + m - 1 = 0 ∧ x2^2 - 4 * x2 + m - 1 = 0 ∧ x1 ≠ x2) ∧ 
  (3 * (m - 1) - 4 > 2) →

  3 < m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1203_120393


namespace NUMINAMATH_GPT_sam_distinct_meals_count_l1203_120368

-- Definitions based on conditions
def main_dishes := ["Burger", "Pasta", "Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Chips", "Cookie", "Apple"]

-- Definition to exclude invalid combinations
def is_valid_combination (main : String) (beverage : String) : Bool :=
  if main = "Burger" && beverage = "Soda" then false else true

-- Number of valid combinations
def count_valid_meals : Nat :=
  main_dishes.length * beverages.length * snacks.length - snacks.length

theorem sam_distinct_meals_count : count_valid_meals = 15 := 
  sorry

end NUMINAMATH_GPT_sam_distinct_meals_count_l1203_120368


namespace NUMINAMATH_GPT_functions_are_even_l1203_120340

noncomputable def f_A (x : ℝ) : ℝ := -|x| + 2
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem functions_are_even :
  (∀ x : ℝ, f_A x = f_A (-x)) ∧
  (∀ x : ℝ, f_B x = f_B (-x)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_C x = f_C (-x)) :=
by
  sorry

end NUMINAMATH_GPT_functions_are_even_l1203_120340


namespace NUMINAMATH_GPT_problem_solution_l1203_120337

open Set

theorem problem_solution
    (a b : ℝ)
    (ineq : ∀ x : ℝ, 1 < x ∧ x < b → a * x^2 - 3 * x + 2 < 0)
    (f : ℝ → ℝ := λ x => (2 * a + b) * x - 1 / ((a - b) * (x - 1))) :
    a = 1 ∧ b = 2 ∧ (∀ x, 1 < x ∧ x < b → f x ≥ 8 ∧ (f x = 8 ↔ x = 3 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1203_120337


namespace NUMINAMATH_GPT_Jordan_income_l1203_120394

theorem Jordan_income (q A : ℝ) (h : A > 30000)
  (h1 : (q / 100 * 30000 + (q + 3) / 100 * (A - 30000) - 600) = (q + 0.5) / 100 * A) :
  A = 60000 :=
by
  sorry

end NUMINAMATH_GPT_Jordan_income_l1203_120394


namespace NUMINAMATH_GPT_star_wars_cost_l1203_120321

theorem star_wars_cost 
    (LK_cost LK_earn SW_earn: ℕ) 
    (half_profit: ℕ → ℕ)
    (h1: LK_cost = 10)
    (h2: LK_earn = 200)
    (h3: SW_earn = 405)
    (h4: LK_earn - LK_cost = half_profit SW_earn)
    (h5: half_profit SW_earn * 2 = SW_earn - (LK_earn - LK_cost)) :
    ∃ SW_cost : ℕ, SW_cost = 25 := 
by
  sorry

end NUMINAMATH_GPT_star_wars_cost_l1203_120321


namespace NUMINAMATH_GPT_original_three_numbers_are_arith_geo_seq_l1203_120371

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end NUMINAMATH_GPT_original_three_numbers_are_arith_geo_seq_l1203_120371


namespace NUMINAMATH_GPT_passengers_on_plane_l1203_120303

variables (P : ℕ) (fuel_per_mile : ℕ := 20) (fuel_per_person : ℕ := 3) (fuel_per_bag : ℕ := 2)
variables (num_crew : ℕ := 5) (bags_per_person : ℕ := 2) (trip_distance : ℕ := 400)
variables (total_fuel : ℕ := 106000)

def total_people := P + num_crew
def total_bags := bags_per_person * total_people
def total_fuel_per_mile := fuel_per_mile + fuel_per_person * P + fuel_per_bag * total_bags
def total_trip_fuel := trip_distance * total_fuel_per_mile

theorem passengers_on_plane : total_trip_fuel = total_fuel → P = 33 := 
by
  sorry

end NUMINAMATH_GPT_passengers_on_plane_l1203_120303


namespace NUMINAMATH_GPT_complex_multiplication_l1203_120345

theorem complex_multiplication {i : ℂ} (h : i^2 = -1) : i * (1 - i) = 1 + i := 
by 
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1203_120345


namespace NUMINAMATH_GPT_beret_count_l1203_120331

/-- James can make a beret from 3 spools of yarn. 
    He has 12 spools of red yarn, 15 spools of black yarn, and 6 spools of blue yarn.
    Prove that he can make 11 berets in total. -/
theorem beret_count (red_yarn : ℕ) (black_yarn : ℕ) (blue_yarn : ℕ) (spools_per_beret : ℕ) 
  (total_yarn : ℕ) (num_berets : ℕ) (h1 : red_yarn = 12) (h2 : black_yarn = 15) (h3 : blue_yarn = 6)
  (h4 : spools_per_beret = 3) (h5 : total_yarn = red_yarn + black_yarn + blue_yarn) 
  (h6 : num_berets = total_yarn / spools_per_beret) : 
  num_berets = 11 :=
by sorry

end NUMINAMATH_GPT_beret_count_l1203_120331


namespace NUMINAMATH_GPT_players_quit_l1203_120366

theorem players_quit (initial_players remaining_lives lives_per_player : ℕ) 
  (h1 : initial_players = 8) (h2 : remaining_lives = 15) (h3 : lives_per_player = 5) :
  initial_players - (remaining_lives / lives_per_player) = 5 :=
by
  -- A proof is required here
  sorry

end NUMINAMATH_GPT_players_quit_l1203_120366


namespace NUMINAMATH_GPT_josh_total_payment_with_tax_and_discount_l1203_120314

-- Definitions
def total_string_cheeses (pack1 : ℕ) (pack2 : ℕ) (pack3 : ℕ) : ℕ :=
  pack1 + pack2 + pack3

def total_cost_before_tax_and_discount (n : ℕ) (cost_per_cheese : ℚ) : ℚ :=
  n * cost_per_cheese

def discount_amount (cost : ℚ) (discount_rate : ℚ) : ℚ :=
  cost * discount_rate

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost - discount

def sales_tax_amount (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost * tax_rate

def total_cost (cost : ℚ) (tax : ℚ) : ℚ :=
  cost + tax

-- The statement
theorem josh_total_payment_with_tax_and_discount :
  let cost_per_cheese := 0.10
  let discount_rate := 0.05
  let tax_rate := 0.12
  total_cost (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                              (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate))
             (sales_tax_amount (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                                               (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate)) tax_rate) = 6.81 := 
  sorry

end NUMINAMATH_GPT_josh_total_payment_with_tax_and_discount_l1203_120314


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l1203_120322

variable {α β γ : ℝ}
variable (x y : ℕ)

theorem ratio_of_boys_to_girls (hα : α ≠ 1/2) (hprob : (x * β + y * γ) / (x + y) = 1/2) :
  (x : ℝ) / (y : ℝ) = (1/2 - γ) / (β - 1/2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l1203_120322


namespace NUMINAMATH_GPT_ways_to_divide_friends_l1203_120376

theorem ways_to_divide_friends : (4 ^ 8 = 65536) := by
  sorry

end NUMINAMATH_GPT_ways_to_divide_friends_l1203_120376


namespace NUMINAMATH_GPT_common_denominator_first_set_common_denominator_second_set_l1203_120318

theorem common_denominator_first_set (x y : ℕ) (h₁ : y ≠ 0) : Nat.lcm (3 * y) (2 * y^2) = 6 * y^2 :=
by sorry

theorem common_denominator_second_set (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : Nat.lcm (a^2 * b) (3 * a * b^2) = 3 * a^2 * b^2 :=
by sorry

end NUMINAMATH_GPT_common_denominator_first_set_common_denominator_second_set_l1203_120318


namespace NUMINAMATH_GPT_origin_movement_by_dilation_l1203_120380

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end NUMINAMATH_GPT_origin_movement_by_dilation_l1203_120380


namespace NUMINAMATH_GPT_last_three_digits_of_2_pow_15000_l1203_120304

-- We need to define the given condition as a hypothesis and then state the goal.
theorem last_three_digits_of_2_pow_15000 :
  (2 ^ 500 ≡ 1 [MOD 1250]) → (2 ^ 15000 ≡ 1 [MOD 1000]) := by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_2_pow_15000_l1203_120304


namespace NUMINAMATH_GPT_midpoint_coords_product_l1203_120339

def midpoint_prod (x1 y1 x2 y2 : ℤ) : ℤ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx * my

theorem midpoint_coords_product :
  midpoint_prod 4 (-7) (-8) 9 = -2 := by
  sorry

end NUMINAMATH_GPT_midpoint_coords_product_l1203_120339


namespace NUMINAMATH_GPT_alice_has_winning_strategy_l1203_120398

def alice_has_winning_strategy_condition (nums : List ℤ) : Prop :=
  nums.length = 17 ∧ ∀ x ∈ nums, ¬ (x % 17 = 0)

theorem alice_has_winning_strategy (nums : List ℤ) (H : alice_has_winning_strategy_condition nums) : ∃ (f : List ℤ → List ℤ), ∀ k, (f^[k] nums).sum % 17 = 0 :=
sorry

end NUMINAMATH_GPT_alice_has_winning_strategy_l1203_120398


namespace NUMINAMATH_GPT_angle_same_terminal_side_210_l1203_120346

theorem angle_same_terminal_side_210 (n : ℤ) : 
  ∃ k : ℤ, 210 = -510 + k * 360 ∧ 0 ≤ 210 ∧ 210 < 360 :=
by
  use 2
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_210_l1203_120346


namespace NUMINAMATH_GPT_x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l1203_120390

variable {x : ℝ}
variable {y : ℝ}

theorem x_gt_y_necessary_not_sufficient_for_x_gt_abs_y
  (hx : x > 0) :
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|) := by
  sorry

end NUMINAMATH_GPT_x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l1203_120390


namespace NUMINAMATH_GPT_total_books_together_l1203_120365

-- Given conditions
def SamBooks : Nat := 110
def JoanBooks : Nat := 102

-- Theorem to prove the total number of books they have together
theorem total_books_together : SamBooks + JoanBooks = 212 := 
by
  sorry

end NUMINAMATH_GPT_total_books_together_l1203_120365


namespace NUMINAMATH_GPT_min_value_inequality_l1203_120334

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l1203_120334


namespace NUMINAMATH_GPT_alpha_eq_one_l1203_120309

-- Definitions based on conditions from the problem statement.
variable (α : ℝ) 
variable (f : ℝ → ℝ)

-- The conditions defined as hypotheses
axiom functional_eq (x y : ℝ) : f (α * (x + y)) = f x + f y
axiom non_constant : ∃ x y : ℝ, f x ≠ 0

-- The statement to prove
theorem alpha_eq_one : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (α * (x + y)) = f x + f y) ∧ (∃ x y : ℝ, f x ≠ f y)) → α = 1 :=
by
  sorry

end NUMINAMATH_GPT_alpha_eq_one_l1203_120309


namespace NUMINAMATH_GPT_remaining_length_l1203_120385

variable (L₁ L₂: ℝ)
variable (H₁: L₁ = 0.41)
variable (H₂: L₂ = 0.33)

theorem remaining_length (L₁ L₂: ℝ) (H₁: L₁ = 0.41) (H₂: L₂ = 0.33) : L₁ - L₂ = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_remaining_length_l1203_120385


namespace NUMINAMATH_GPT_distinct_solutions_equation_number_of_solutions_a2019_l1203_120329

theorem distinct_solutions_equation (a : ℕ) (ha : a > 1) : 
  ∃ (x y : ℕ), (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / (a : ℚ)) ∧ x > 0 ∧ y > 0 ∧ (x ≠ y) ∧ 
  ∃ (x₁ y₁ x₂ y₂ : ℕ), (1 / (x₁ : ℚ) + 1 / (y₁ : ℚ) = 1 / (a : ℚ)) ∧
  (1 / (x₂ : ℚ) + 1 / (y₂ : ℚ) = 1 / (a : ℚ)) ∧
  x₁ ≠ y₁ ∧ x₂ ≠ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) := 
sorry

theorem number_of_solutions_a2019 :
  ∃ n, n = (3 * 3) := 
by {
  -- use 2019 = 3 * 673 and divisor count
  sorry 
}

end NUMINAMATH_GPT_distinct_solutions_equation_number_of_solutions_a2019_l1203_120329


namespace NUMINAMATH_GPT_impossible_to_form_palindrome_l1203_120358

-- Define the possible cards
inductive Card
| abc | bca | cab

-- Define the rule for palindrome formation
def canFormPalindrome (w : List Card) : Prop :=
  sorry  -- Placeholder for the actual formation rule

-- Define the theorem statement
theorem impossible_to_form_palindrome (w : List Card) :
  ¬canFormPalindrome w :=
sorry

end NUMINAMATH_GPT_impossible_to_form_palindrome_l1203_120358


namespace NUMINAMATH_GPT_horner_v3_value_l1203_120379

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end NUMINAMATH_GPT_horner_v3_value_l1203_120379


namespace NUMINAMATH_GPT_min_cost_example_l1203_120374

-- Define the numbers given in the problem
def num_students : Nat := 25
def num_vampire : Nat := 11
def num_pumpkin : Nat := 14
def pack_cost : Nat := 3
def individual_cost : Nat := 1
def pack_size : Nat := 5

-- Define the cost calculation function
def min_cost (num_v: Nat) (num_p: Nat) : Nat :=
  let num_v_packs := num_v / pack_size  -- number of packs needed for vampire bags
  let num_v_individual := num_v % pack_size  -- remaining vampire bags needed
  let num_v_cost := (num_v_packs * pack_cost) + (num_v_individual * individual_cost)
  let num_p_packs := num_p / pack_size  -- number of packs needed for pumpkin bags
  let num_p_individual := num_p % pack_size  -- remaining pumpkin bags needed
  let num_p_cost := (num_p_packs * pack_cost) + (num_p_individual * individual_cost)
  num_v_cost + num_p_cost

-- The statement to prove
theorem min_cost_example : min_cost num_vampire num_pumpkin = 17 :=
  by
  sorry

end NUMINAMATH_GPT_min_cost_example_l1203_120374


namespace NUMINAMATH_GPT_smallest_value_x_abs_eq_32_l1203_120382

theorem smallest_value_x_abs_eq_32 : ∃ x : ℚ, (x = -29 / 5) ∧ (|5 * x - 3| = 32) ∧ 
  (∀ y : ℚ, (|5 * y - 3| = 32) → (x ≤ y)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_x_abs_eq_32_l1203_120382


namespace NUMINAMATH_GPT_sum_of_roots_l1203_120396

theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hroots : ∀ x : ℝ, x^2 - p*x + 2*q = 0) :
  p + q = p :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_l1203_120396


namespace NUMINAMATH_GPT_initial_ripe_peaches_l1203_120342

theorem initial_ripe_peaches (P U R: ℕ) (H1: P = 18) (H2: 2 * 5 = 10) (H3: (U + 7) + U = 15 - 3) (H4: R + 10 = U + 7) : 
  R = 1 :=
by
  sorry

end NUMINAMATH_GPT_initial_ripe_peaches_l1203_120342


namespace NUMINAMATH_GPT_betty_age_l1203_120381

variable (C A B : ℝ)

-- conditions
def Carol_five_times_Alice := C = 5 * A
def Alice_twelve_years_younger_than_Carol := A = C - 12
def Carol_twice_as_old_as_Betty := C = 2 * B

-- goal
theorem betty_age (hc1 : Carol_five_times_Alice C A)
                  (hc2 : Alice_twelve_years_younger_than_Carol C A)
                  (hc3 : Carol_twice_as_old_as_Betty C B) : B = 7.5 := 
  by
  sorry

end NUMINAMATH_GPT_betty_age_l1203_120381


namespace NUMINAMATH_GPT_units_digit_of_product_l1203_120372

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : Nat) : Nat :=
  n % 10

def target_product : Nat :=
  factorial 1 * factorial 2 * factorial 3 * factorial 4

theorem units_digit_of_product : units_digit target_product = 8 :=
  by
    sorry

end NUMINAMATH_GPT_units_digit_of_product_l1203_120372


namespace NUMINAMATH_GPT_difference_in_pennies_l1203_120308

theorem difference_in_pennies (p : ℤ) : 
  let alice_nickels := 3 * p + 2
  let bob_nickels := 2 * p + 6
  let difference_nickels := alice_nickels - bob_nickels
  let difference_in_pennies := difference_nickels * 5
  difference_in_pennies = 5 * p - 20 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_pennies_l1203_120308


namespace NUMINAMATH_GPT_given_fraction_l1203_120332

variable (initial_cards : ℕ)
variable (cards_given_to_friend : ℕ)
variable (fraction_given_to_brother : ℚ)

noncomputable def fraction_given (initial_cards cards_given_to_friend : ℕ) (fraction_given_to_brother : ℚ) : Prop :=
  let cards_left := initial_cards / 2
  initial_cards - cards_left - cards_given_to_friend = fraction_given_to_brother * initial_cards

theorem given_fraction
  (h_initial : initial_cards = 16)
  (h_given_to_friend : cards_given_to_friend = 2)
  (h_fraction : fraction_given_to_brother = 3 / 8) :
  fraction_given initial_cards cards_given_to_friend fraction_given_to_brother :=
by
  sorry

end NUMINAMATH_GPT_given_fraction_l1203_120332


namespace NUMINAMATH_GPT_ana_wins_probability_l1203_120399

noncomputable def probability_ana_wins : ℚ := 
  let a := (1 / 2)^5
  let r := (1 / 2)^4
  a / (1 - r)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 30 :=
by
  sorry

end NUMINAMATH_GPT_ana_wins_probability_l1203_120399


namespace NUMINAMATH_GPT_train_speed_l1203_120347

noncomputable def train_length : ℝ := 1500
noncomputable def bridge_length : ℝ := 1200
noncomputable def crossing_time : ℝ := 30

theorem train_speed :
  (train_length + bridge_length) / crossing_time = 90 := by
  sorry

end NUMINAMATH_GPT_train_speed_l1203_120347


namespace NUMINAMATH_GPT_net_moles_nh3_after_reactions_l1203_120307

/-- Define the stoichiometry of the reactions and available amounts of reactants -/
def step1_reaction (nh4cl na2co3 : ℕ) : ℕ :=
  if nh4cl / 2 >= na2co3 then 
    2 * na2co3
  else 
    2 * (nh4cl / 2)

def step2_reaction (koh h3po4 : ℕ) : ℕ :=
  0  -- No NH3 produced in this step

theorem net_moles_nh3_after_reactions :
  let nh4cl := 3
  let na2co3 := 1
  let koh := 3
  let h3po4 := 1
  let nh3_after_step1 := step1_reaction nh4cl na2co3
  let nh3_after_step2 := step2_reaction koh h3po4
  nh3_after_step1 + nh3_after_step2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_net_moles_nh3_after_reactions_l1203_120307


namespace NUMINAMATH_GPT_distance_focus_parabola_to_line_l1203_120378

theorem distance_focus_parabola_to_line :
  let focus : ℝ × ℝ := (1, 0)
  let distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ := |A * p.1 + B * p.2 + C| / Real.sqrt (A^2 + B^2)
  distance focus 1 (-Real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_focus_parabola_to_line_l1203_120378
