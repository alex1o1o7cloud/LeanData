import Mathlib

namespace NUMINAMATH_GPT_number_of_distinct_digit_odd_numbers_l1865_186505

theorem number_of_distinct_digit_odd_numbers (a b c d : ℕ) :
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧
  a * 1000 + b * 100 + c * 10 + d ≤ 9999 ∧
  (a * 1000 + b * 100 + c * 10 + d) % 2 = 1 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0
  → ∃ (n : ℕ), n = 2240 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_distinct_digit_odd_numbers_l1865_186505


namespace NUMINAMATH_GPT_expected_sufferers_l1865_186594

theorem expected_sufferers 
  (fraction_condition : ℚ := 1 / 4)
  (sample_size : ℕ := 400) 
  (expected_number : ℕ := 100) : 
  fraction_condition * sample_size = expected_number := 
by 
  sorry

end NUMINAMATH_GPT_expected_sufferers_l1865_186594


namespace NUMINAMATH_GPT_scientist_prob_rain_l1865_186526

theorem scientist_prob_rain (x : ℝ) (p0 p1 : ℝ)
  (h0 : p0 + p1 = 1)
  (h1 : ∀ x : ℝ, x = (p0 * x^2 + p0 * (1 - x) * x + p1 * (1 - x) * x) / x + (1 - x) - x^2 / (x + 1))
  (h2 : (x + p0 / (x + 1) - x^2 / (x + 1)) = 0.2) :
  x = 1/9 := 
sorry

end NUMINAMATH_GPT_scientist_prob_rain_l1865_186526


namespace NUMINAMATH_GPT_choose_officers_from_six_l1865_186556

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end NUMINAMATH_GPT_choose_officers_from_six_l1865_186556


namespace NUMINAMATH_GPT_div_gt_sum_div_sq_l1865_186509

theorem div_gt_sum_div_sq (n d d' : ℕ) (h₁ : d' > d) (h₂ : d ∣ n) (h₃ : d' ∣ n) : 
  d' > d + d * d / n :=
by 
  sorry

end NUMINAMATH_GPT_div_gt_sum_div_sq_l1865_186509


namespace NUMINAMATH_GPT_scienceStudyTime_l1865_186578

def totalStudyTime : ℕ := 60
def mathStudyTime : ℕ := 35

theorem scienceStudyTime : totalStudyTime - mathStudyTime = 25 :=
by sorry

end NUMINAMATH_GPT_scienceStudyTime_l1865_186578


namespace NUMINAMATH_GPT_circle_area_from_points_l1865_186541

theorem circle_area_from_points (C D : ℝ × ℝ) (hC : C = (2, 3)) (hD : D = (8, 9)) : 
  ∃ A : ℝ, A = 18 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_from_points_l1865_186541


namespace NUMINAMATH_GPT_percent_daisies_l1865_186585

theorem percent_daisies 
    (total_flowers : ℕ)
    (yellow_flowers : ℕ)
    (yellow_tulips : ℕ)
    (blue_flowers : ℕ)
    (blue_daisies : ℕ)
    (h1 : 2 * yellow_tulips = yellow_flowers) 
    (h2 : 3 * blue_daisies = blue_flowers)
    (h3 : 10 * yellow_flowers = 7 * total_flowers) : 
    100 * (yellow_flowers / 2 + blue_daisies) = 45 * total_flowers :=
by
  sorry

end NUMINAMATH_GPT_percent_daisies_l1865_186585


namespace NUMINAMATH_GPT_change_received_l1865_186571

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end NUMINAMATH_GPT_change_received_l1865_186571


namespace NUMINAMATH_GPT_part_I_part_II_l1865_186598

noncomputable def M : Set ℝ := { x | |x + 1| + |x - 1| ≤ 2 }

theorem part_I : M = Set.Icc (-1 : ℝ) (1 : ℝ) := 
sorry

theorem part_II (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ (1/6)) (hz : |z| ≤ (1/9)) :
  |x + 2 * y - 3 * z| ≤ (5/3) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1865_186598


namespace NUMINAMATH_GPT_find_constant_t_l1865_186566

theorem find_constant_t :
  (exists t : ℚ,
  ∀ x : ℚ,
    (5 * x ^ 2 - 6 * x + 7) * (4 * x ^ 2 + t * x + 10) =
      20 * x ^ 4 - 48 * x ^ 3 + 114 * x ^ 2 - 102 * x + 70) :=
sorry

end NUMINAMATH_GPT_find_constant_t_l1865_186566


namespace NUMINAMATH_GPT_inequality_sum_l1865_186521

open Real
open BigOperators

theorem inequality_sum 
  (n : ℕ) 
  (h : n > 1) 
  (x : Fin n → ℝ)
  (hx1 : ∀ i, 0 < x i) 
  (hx2 : ∑ i, x i = 1) :
  ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end NUMINAMATH_GPT_inequality_sum_l1865_186521


namespace NUMINAMATH_GPT_determine_a_l1865_186555

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x ^ 2 + a else 2 ^ x

theorem determine_a (a : ℝ) (h1 : a > -1) (h2 : f a (f a (-1)) = 4) : a = 1 :=
sorry

end NUMINAMATH_GPT_determine_a_l1865_186555


namespace NUMINAMATH_GPT_power_computation_l1865_186512

theorem power_computation :
  16^10 * 8^6 / 4^22 = 16384 :=
by
  sorry

end NUMINAMATH_GPT_power_computation_l1865_186512


namespace NUMINAMATH_GPT_sin_log_infinite_zeros_in_01_l1865_186546

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end NUMINAMATH_GPT_sin_log_infinite_zeros_in_01_l1865_186546


namespace NUMINAMATH_GPT_surface_area_of_figure_l1865_186548

theorem surface_area_of_figure 
  (block_surface_area : ℕ) 
  (loss_per_block : ℕ) 
  (number_of_blocks : ℕ) 
  (effective_surface_area : ℕ)
  (total_surface_area : ℕ) 
  (h_block : block_surface_area = 18) 
  (h_loss : loss_per_block = 2) 
  (h_blocks : number_of_blocks = 4) 
  (h_effective : effective_surface_area = block_surface_area - loss_per_block) 
  (h_total : total_surface_area = number_of_blocks * effective_surface_area) : 
  total_surface_area = 64 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_figure_l1865_186548


namespace NUMINAMATH_GPT_turban_price_l1865_186549

theorem turban_price (T : ℝ) (total_salary : ℝ) (received_salary : ℝ)
  (cond1 : total_salary = 90 + T)
  (cond2 : received_salary = 65 + T)
  (cond3 : received_salary = (3 / 4) * total_salary) :
  T = 10 :=
by
  sorry

end NUMINAMATH_GPT_turban_price_l1865_186549


namespace NUMINAMATH_GPT_lightbulb_stops_on_friday_l1865_186501

theorem lightbulb_stops_on_friday
  (total_hours : ℕ) (daily_usage : ℕ) (start_day : ℕ) (stops_day : ℕ)
  (h_total_hours : total_hours = 24999)
  (h_daily_usage : daily_usage = 2)
  (h_start_day : start_day = 1) : 
  stops_day = 5 := by
  sorry

end NUMINAMATH_GPT_lightbulb_stops_on_friday_l1865_186501


namespace NUMINAMATH_GPT_determine_value_of_m_l1865_186520

theorem determine_value_of_m (m : ℤ) :
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 ↔ m = 11 := 
sorry

end NUMINAMATH_GPT_determine_value_of_m_l1865_186520


namespace NUMINAMATH_GPT_M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l1865_186587

-- Definition of the curve using parametric equations
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t, 2 * t^2 + 1)

-- Questions and proof statements
theorem M1_on_curve_C : ∃ t : ℝ, curve t = (0, 1) :=
by { 
  sorry 
}

theorem M2_not_on_curve_C : ¬ (∃ t : ℝ, curve t = (5, 4)) :=
by { 
  sorry 
}

theorem M3_on_curve_C_a_eq_9 (a : ℝ) : (∃ t : ℝ, curve t = (6, a)) → a = 9 :=
by { 
  sorry 
}

end NUMINAMATH_GPT_M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l1865_186587


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l1865_186558

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 6 * x + m * y - 1 = 0 ↔ 2 * x - y + 1 = 0) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l1865_186558


namespace NUMINAMATH_GPT_books_per_shelf_l1865_186542

theorem books_per_shelf :
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  remaining_books / shelves = 3 :=
by
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  have h1 : remaining_books = 12 := by simp [remaining_books]
  have h2 : remaining_books / shelves = 3 := by norm_num [remaining_books, shelves]
  exact h2

end NUMINAMATH_GPT_books_per_shelf_l1865_186542


namespace NUMINAMATH_GPT_sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l1865_186596

-- Problem (1)
theorem sector_angle_given_circumference_and_area :
  (∀ (r l : ℝ), 2 * r + l = 10 ∧ (1 / 2) * l * r = 4 → l / r = (1 / 2)) := by
  sorry

-- Problem (2)
theorem max_sector_area_given_circumference :
  (∀ (r l : ℝ), 2 * r + l = 40 → (r = 10 ∧ l = 20 ∧ (1 / 2) * l * r = 100 ∧ l / r = 2)) := by
  sorry

end NUMINAMATH_GPT_sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l1865_186596


namespace NUMINAMATH_GPT_light_intensity_after_glass_pieces_minimum_glass_pieces_l1865_186516

theorem light_intensity_after_glass_pieces (a : ℝ) (x : ℕ) : 
  (y : ℝ) = a * (0.9 ^ x) :=
sorry

theorem minimum_glass_pieces (a : ℝ) (x : ℕ) : 
  a * (0.9 ^ x) < a / 3 ↔ x ≥ 11 :=
sorry

end NUMINAMATH_GPT_light_intensity_after_glass_pieces_minimum_glass_pieces_l1865_186516


namespace NUMINAMATH_GPT_remaining_nap_time_is_three_hours_l1865_186562

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end NUMINAMATH_GPT_remaining_nap_time_is_three_hours_l1865_186562


namespace NUMINAMATH_GPT_simplify_expression_l1865_186504

variable (c d : ℝ)
variable (hc : 0 < c)
variable (hd : 0 < d)
variable (h : c^3 + d^3 = 3 * (c + d))

theorem simplify_expression : (c / d) + (d / c) - (3 / (c * d)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1865_186504


namespace NUMINAMATH_GPT_inequality_proof_l1865_186503

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_eq : a + b + c = 4 * (abc)^(1/3)) :
  2 * (ab + bc + ca) + 4 * min (a^2) (min (b^2) (c^2)) ≥ a^2 + b^2 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1865_186503


namespace NUMINAMATH_GPT_binom_30_3_l1865_186511

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end NUMINAMATH_GPT_binom_30_3_l1865_186511


namespace NUMINAMATH_GPT_ceil_floor_subtraction_l1865_186595

theorem ceil_floor_subtraction :
  ⌈(7:ℝ) / 3⌉ + ⌊- (7:ℝ) / 3⌋ - 3 = -3 := 
by
  sorry   -- Placeholder for the proof

end NUMINAMATH_GPT_ceil_floor_subtraction_l1865_186595


namespace NUMINAMATH_GPT_quadratic_eq_of_sum_and_product_l1865_186518

theorem quadratic_eq_of_sum_and_product (a b c : ℝ) (h_sum : -b / a = 4) (h_product : c / a = 3) :
    ∀ (x : ℝ), a * x^2 + b * x + c = a * x^2 - 4 * a * x + 3 * a :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_of_sum_and_product_l1865_186518


namespace NUMINAMATH_GPT_not_possible_to_tile_l1865_186500

theorem not_possible_to_tile 
    (m n : ℕ) (a b : ℕ)
    (h_m : m = 2018)
    (h_n : n = 2020)
    (h_a : a = 5)
    (h_b : b = 8) :
    ¬ ∃ k : ℕ, k * (a * b) = m * n := by
sorry

end NUMINAMATH_GPT_not_possible_to_tile_l1865_186500


namespace NUMINAMATH_GPT_part_I_part_II_l1865_186530

variable (α : ℝ)

-- The given conditions.
variable (h1 : π < α)
variable (h2 : α < (3 * π) / 2)
variable (h3 : Real.sin α = -4/5)

-- Part (I): Prove cos α = -3/5
theorem part_I : Real.cos α = -3/5 :=
sorry

-- Part (II): Prove sin 2α + 3 tan α = 24/25 + 4
theorem part_II : Real.sin (2 * α) + 3 * Real.tan α = 24/25 + 4 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1865_186530


namespace NUMINAMATH_GPT_problem_1_problem_2_l1865_186537

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem problem_1 (h₁ : ∀ x, x > 0 → x ≠ 1 → f x = x / Real.log x) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) ∧
  (∀ x, x > Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) :=
sorry

theorem problem_2 (h₁ : f x₁ = 1) (h₂ : f x₂ = 1) (h₃ : x₁ ≠ x₂) (h₄ : x₁ > 0) (h₅ : x₂ > 0):
  x₁ + x₂ > 2 * Real.exp 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1865_186537


namespace NUMINAMATH_GPT_triangle_incenter_equilateral_l1865_186568

theorem triangle_incenter_equilateral (a b c : ℝ) (h : (b + c) / a = (a + c) / b ∧ (a + c) / b = (a + b) / c) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_incenter_equilateral_l1865_186568


namespace NUMINAMATH_GPT_volume_of_water_flowing_per_minute_l1865_186531

variable (d w r : ℝ) (V : ℝ)

theorem volume_of_water_flowing_per_minute (h1 : d = 3) 
                                           (h2 : w = 32) 
                                           (h3 : r = 33.33) : 
  V = 3199.68 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_water_flowing_per_minute_l1865_186531


namespace NUMINAMATH_GPT_sum_of_squares_of_four_consecutive_even_numbers_l1865_186517

open Int

theorem sum_of_squares_of_four_consecutive_even_numbers (x y z w : ℤ) 
    (hx : x % 2 = 0) (hy : y = x + 2) (hz : z = x + 4) (hw : w = x + 6)
    : x + y + z + w = 36 → x^2 + y^2 + z^2 + w^2 = 344 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_four_consecutive_even_numbers_l1865_186517


namespace NUMINAMATH_GPT_remaining_cube_height_l1865_186532

/-- Given a cube with side length 2 units, where a corner is chopped off such that the cut runs
    through points on the three edges adjacent to a selected vertex, each at 1 unit distance
    from that vertex, the height of the remaining portion of the cube when the freshly cut face 
    is placed on a table is equal to (5 * sqrt 3) / 3. -/
theorem remaining_cube_height (s : ℝ) (h : ℝ) : 
    s = 2 → h = 1 → 
    ∃ height : ℝ, height = (5 * Real.sqrt 3) / 3 := 
by
    sorry

end NUMINAMATH_GPT_remaining_cube_height_l1865_186532


namespace NUMINAMATH_GPT_age_difference_l1865_186579

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 16) : (A + B) - (B + C) = 16 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1865_186579


namespace NUMINAMATH_GPT_quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l1865_186540

open Real

-- Mathematical translations of conditions and proofs
theorem quadratic_real_roots_range_of_m (m : ℝ) (h1 : ∃ x : ℝ, x^2 + 2 * x - (m - 2) = 0) :
  m ≥ 1 := by
  sorry

theorem quadratic_root_and_other_m (h1 : (1:ℝ) ^ 2 + 2 * 1 - (m - 2) = 0) :
  m = 3 ∧ ∃ x : ℝ, (x = -3) ∧ (x^2 + 2 * x - 3 = 0) := by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l1865_186540


namespace NUMINAMATH_GPT_prob_rain_both_days_l1865_186553

-- Declare the probabilities involved
def P_Monday : ℝ := 0.40
def P_Tuesday : ℝ := 0.30
def P_Tuesday_given_Monday : ℝ := 0.30

-- Prove the probability of it raining on both days
theorem prob_rain_both_days : P_Monday * P_Tuesday_given_Monday = 0.12 :=
by
  sorry

end NUMINAMATH_GPT_prob_rain_both_days_l1865_186553


namespace NUMINAMATH_GPT_harvest_unripe_oranges_l1865_186570

theorem harvest_unripe_oranges (R T D U: ℕ) (h1: R = 28) (h2: T = 2080) (h3: D = 26)
  (h4: T = D * (R + U)) :
  U = 52 :=
by
  sorry

end NUMINAMATH_GPT_harvest_unripe_oranges_l1865_186570


namespace NUMINAMATH_GPT_find_C_l1865_186597

theorem find_C (A B C : ℕ)
  (hA : A = 348)
  (hB : B = A + 173)
  (hC : C = B + 299) :
  C = 820 :=
sorry

end NUMINAMATH_GPT_find_C_l1865_186597


namespace NUMINAMATH_GPT_man_walking_time_l1865_186543

section TrainProblem

variables {T W : ℕ}

/-- Each day a man meets his wife at the train station after work,
    and then she drives him home. She always arrives exactly on time to pick him up.
    One day he catches an earlier train and arrives at the station an hour early.
    He immediately begins walking home along the same route the wife drives.
    Eventually, his wife sees him on her way to the station and drives him the rest of the way home.
    When they arrive home, the man notices that they arrived 30 minutes earlier than usual.
    How much time did the man spend walking? -/
theorem man_walking_time : 
    (∃ (T : ℕ), T > 30 ∧ (W = T - 30) ∧ (W + 30 = T)) → W = 30 :=
sorry

end TrainProblem

end NUMINAMATH_GPT_man_walking_time_l1865_186543


namespace NUMINAMATH_GPT_check_conditions_l1865_186564

noncomputable def f (x a b : ℝ) : ℝ := |x^2 - 2 * a * x + b|

theorem check_conditions (a b : ℝ) :
  ¬ (∀ x : ℝ, f x a b = f (-x) a b) ∧         -- f(x) is not necessarily an even function
  ¬ (∀ x : ℝ, (f 0 a b = f 2 a b → (f x a b = f (2 - x) a b))) ∧ -- No guaranteed symmetry about x=1
  (a^2 - b^2 ≤ 0 → ∀ x : ℝ, x ≥ a → ∀ y : ℝ, y ≥ x → f y a b ≥ f x a b) ∧ -- f(x) is increasing on [a, +∞) if a^2 - b^2 ≤ 0
  ¬ (∀ x : ℝ, f x a b ≤ |a^2 - b|)         -- f(x) does not necessarily have a max value of |a^2 - b|
:= sorry

end NUMINAMATH_GPT_check_conditions_l1865_186564


namespace NUMINAMATH_GPT_prob_neither_A_nor_B_l1865_186551

theorem prob_neither_A_nor_B
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ)
  (h1 : P_A = 0.25) (h2 : P_B = 0.30) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_prob_neither_A_nor_B_l1865_186551


namespace NUMINAMATH_GPT_b_distance_behind_proof_l1865_186572

-- Given conditions
def race_distance : ℕ := 1000
def a_time : ℕ := 40
def b_delay : ℕ := 10

def a_speed : ℕ := race_distance / a_time
def b_distance_behind : ℕ := a_speed * b_delay

theorem b_distance_behind_proof : b_distance_behind = 250 := by
  -- Prove that b_distance_behind = 250
  sorry

end NUMINAMATH_GPT_b_distance_behind_proof_l1865_186572


namespace NUMINAMATH_GPT_width_to_length_ratio_l1865_186577

variables {w l P : ℕ}

theorem width_to_length_ratio :
  l = 10 → P = 30 → P = 2 * (l + w) → (w : ℚ) / l = 1 / 2 :=
by
  intro h1 h2 h3
  -- Noncomputable definition for rational division
  -- (ℚ is used for exact rational division)
  sorry

#check width_to_length_ratio

end NUMINAMATH_GPT_width_to_length_ratio_l1865_186577


namespace NUMINAMATH_GPT_max_total_profit_max_avg_annual_profit_l1865_186508

noncomputable def total_profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 36
noncomputable def avg_annual_profit (x : ℕ) : ℝ := (total_profit x) / x

theorem max_total_profit : ∃ x : ℕ, total_profit x = 45 ∧ x = 9 :=
  by sorry

theorem max_avg_annual_profit : ∃ x : ℕ, avg_annual_profit x = 6 ∧ x = 6 :=
  by sorry

end NUMINAMATH_GPT_max_total_profit_max_avg_annual_profit_l1865_186508


namespace NUMINAMATH_GPT_solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l1865_186550

-- Problem 1: Solution Set of the Inequality
theorem solution_set_x2_minus_5x_plus_4 : 
  {x : ℝ | x^2 - 5 * x + 4 > 0} = {x : ℝ | x < 1 ∨ x > 4} :=
sorry

-- Problem 2: Range of Values for a
theorem range_of_a_if_x2_plus_ax_plus_4_gt_0 (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 4 > 0) :
  -4 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l1865_186550


namespace NUMINAMATH_GPT_paint_needed_l1865_186523

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end NUMINAMATH_GPT_paint_needed_l1865_186523


namespace NUMINAMATH_GPT_time_jack_first_half_l1865_186536

-- Define the conditions
def t_Jill : ℕ := 32
def t_2 : ℕ := 6
def t_Jack : ℕ := t_Jill - 7

-- Define the time Jack took for the first half
def t_1 : ℕ := t_Jack - t_2

-- State the theorem to prove
theorem time_jack_first_half : t_1 = 19 := by
  sorry

end NUMINAMATH_GPT_time_jack_first_half_l1865_186536


namespace NUMINAMATH_GPT_choose_students_l1865_186544

/-- There are 50 students in the class, including one class president and one vice-president. 
    We want to select 5 students to participate in an activity such that at least one of 
    the class president or vice-president is included. We assert that there are exactly 2 
    distinct methods for making this selection. -/
theorem choose_students (students : Finset ℕ) (class_president vice_president : ℕ) (students_card : students.card = 50)
  (students_ex : class_president ∈ students ∧ vice_president ∈ students) : 
  ∃ valid_methods : Finset (Finset ℕ), valid_methods.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_choose_students_l1865_186544


namespace NUMINAMATH_GPT_arrangement_condition_l1865_186522

theorem arrangement_condition (x y z : ℕ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (H1 : x ≤ y + z) 
  (H2 : y ≤ x + z) 
  (H3 : z ≤ x + y) : 
  ∃ (A : ℕ) (B : ℕ) (C : ℕ), 
    A = x ∧ B = y ∧ C = z ∧
    A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1 ∧
    (A ≤ B + C) ∧ (B ≤ A + C) ∧ (C ≤ A + B) :=
by
  sorry

end NUMINAMATH_GPT_arrangement_condition_l1865_186522


namespace NUMINAMATH_GPT_new_interest_rate_l1865_186507

theorem new_interest_rate 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (initial_rate : ℝ) 
  (time : ℝ) 
  (new_total_interest : ℝ)
  (principal : ℝ)
  (new_rate : ℝ) 
  (h1 : initial_interest = principal * initial_rate * time)
  (h2 : new_total_interest = initial_interest + additional_interest)
  (h3 : new_total_interest = principal * new_rate * time)
  (principal_val : principal = initial_interest / initial_rate) :
  new_rate = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_new_interest_rate_l1865_186507


namespace NUMINAMATH_GPT_part1_part2_l1865_186557

noncomputable def point_M (m : ℝ) : ℝ × ℝ := (2 * m + 1, m - 4)
def point_N : ℝ × ℝ := (5, 2)

theorem part1 (m : ℝ) (h : m - 4 = 2) : point_M m = (13, 2) := by
  sorry

theorem part2 (m : ℝ) (h : 2 * m + 1 = 3) : point_M m = (3, -3) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1865_186557


namespace NUMINAMATH_GPT_baron_munchausen_incorrect_l1865_186545

theorem baron_munchausen_incorrect : 
  ∀ (n : ℕ) (ab : ℕ), 10 ≤ n → n ≤ 99 → 0 ≤ ab → ab ≤ 99 
  → ¬ (∃ (m : ℕ), n * 100 + ab = m * m) := 
by
  intros n ab n_lower_bound n_upper_bound ab_lower_bound ab_upper_bound
  sorry

end NUMINAMATH_GPT_baron_munchausen_incorrect_l1865_186545


namespace NUMINAMATH_GPT_cost_per_top_l1865_186559
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end NUMINAMATH_GPT_cost_per_top_l1865_186559


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l1865_186554

-- Variables for the number of boys, girls, and teachers
variables (B G T : ℕ)

-- Conditions from the problem
def number_of_girls := G = 60
def number_of_teachers := T = (20 * B) / 100
def total_people := B + G + T = 114

-- Proving the ratio of boys to girls is 3:4 given the conditions
theorem ratio_of_boys_to_girls 
  (hG : number_of_girls G)
  (hT : number_of_teachers B T)
  (hTotal : total_people B G T) :
  B / 15 = 3 ∧ G / 15 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l1865_186554


namespace NUMINAMATH_GPT_min_neg_condition_l1865_186591

theorem min_neg_condition (a : ℝ) (x : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) → a < -7 :=
sorry

end NUMINAMATH_GPT_min_neg_condition_l1865_186591


namespace NUMINAMATH_GPT_find_a_l1865_186573

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.exp x

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 1 → (x - a) * (x - a + 2) ≤ 0) → a = 1 :=
by
  intro h
  sorry 

end NUMINAMATH_GPT_find_a_l1865_186573


namespace NUMINAMATH_GPT_two_positive_roots_condition_l1865_186583

theorem two_positive_roots_condition (a : ℝ) :
  (1 < a ∧ a ≤ 2) ∨ (a ≥ 10) ↔
  ∃ x1 x2 : ℝ, (1-a) * x1^2 + (a+2) * x1 - 4 = 0 ∧ 
               (1-a) * x2^2 + (a+2) * x2 - 4 = 0 ∧ 
               x1 > 0 ∧ x2 > 0 :=
sorry

end NUMINAMATH_GPT_two_positive_roots_condition_l1865_186583


namespace NUMINAMATH_GPT_find_third_number_l1865_186599

-- Definitions and conditions for the problem
def x : ℚ := 1.35
def third_number := 5
def proportion (a b c d : ℚ) := a * d = b * c 

-- Proposition to prove
theorem find_third_number : proportion 0.75 x third_number 9 := 
by
  -- It's advisable to split the proof steps here, but the proof itself is condensed.
  sorry

end NUMINAMATH_GPT_find_third_number_l1865_186599


namespace NUMINAMATH_GPT_factor_poly_l1865_186592

theorem factor_poly (x : ℤ) :
  (x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_factor_poly_l1865_186592


namespace NUMINAMATH_GPT_product_of_sums_of_squares_l1865_186552

theorem product_of_sums_of_squares (a b : ℤ) 
  (h1 : ∃ x1 y1 : ℤ, a = x1^2 + y1^2)
  (h2 : ∃ x2 y2 : ℤ, b = x2^2 + y2^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_sums_of_squares_l1865_186552


namespace NUMINAMATH_GPT_find_S10_value_l1865_186538

noncomputable def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, 4 * S n = n * (a n + a (n + 1))

theorem find_S10_value (a S : ℕ → ℕ) (h1 : a 4 = 7) (h2 : sequence_sum a S) :
  S 10 = 100 :=
sorry

end NUMINAMATH_GPT_find_S10_value_l1865_186538


namespace NUMINAMATH_GPT_exists_member_T_divisible_by_3_l1865_186528

-- Define the set T of all numbers which are the sum of the squares of four consecutive integers
def T := { x : ℤ | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 }

-- Theorem to prove that there exists a member in T which is divisible by 3
theorem exists_member_T_divisible_by_3 : ∃ x ∈ T, x % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_member_T_divisible_by_3_l1865_186528


namespace NUMINAMATH_GPT_Bert_sandwiches_left_l1865_186582

theorem Bert_sandwiches_left : (Bert:Type) → 
  (sandwiches_made : ℕ) → 
  sandwiches_made = 12 → 
  (sandwiches_eaten_day1 : ℕ) → 
  sandwiches_eaten_day1 = sandwiches_made / 2 → 
  (sandwiches_eaten_day2 : ℕ) → 
  sandwiches_eaten_day2 = sandwiches_eaten_day1 - 2 →
  (sandwiches_left : ℕ) → 
  sandwiches_left = sandwiches_made - (sandwiches_eaten_day1 + sandwiches_eaten_day2) → 
  sandwiches_left = 2 := 
  sorry

end NUMINAMATH_GPT_Bert_sandwiches_left_l1865_186582


namespace NUMINAMATH_GPT_matchstick_ratio_is_one_half_l1865_186524

def matchsticks_used (houses : ℕ) (matchsticks_per_house : ℕ) : ℕ :=
  houses * matchsticks_per_house

def ratio (a b : ℕ) : ℚ := a / b

def michael_original_matchsticks : ℕ := 600
def michael_houses : ℕ := 30
def matchsticks_per_house : ℕ := 10
def michael_used_matchsticks : ℕ := matchsticks_used michael_houses matchsticks_per_house

theorem matchstick_ratio_is_one_half :
  ratio michael_used_matchsticks michael_original_matchsticks = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_matchstick_ratio_is_one_half_l1865_186524


namespace NUMINAMATH_GPT_solve_log_sin_eq_l1865_186513

noncomputable def log_base (b : ℝ) (a : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem solve_log_sin_eq :
  ∀ x : ℝ, 
  (0 < Real.sin x ∧ Real.sin x < 1) →
  log_base (Real.sin x) 4 * log_base (Real.sin x ^ 2) 2 = 4 →
  ∃ k : ℤ, x = (-1)^k * (Real.pi / 4) + Real.pi * k := 
by
  sorry

end NUMINAMATH_GPT_solve_log_sin_eq_l1865_186513


namespace NUMINAMATH_GPT_leak_empties_tank_in_8_hours_l1865_186590

theorem leak_empties_tank_in_8_hours (capacity : ℕ) (inlet_rate_per_minute : ℕ) (time_with_inlet_open : ℕ) (time_without_inlet_open : ℕ) : 
  capacity = 8640 ∧ inlet_rate_per_minute = 6 ∧ time_with_inlet_open = 12 ∧ time_without_inlet_open = 8 := 
by 
  sorry

end NUMINAMATH_GPT_leak_empties_tank_in_8_hours_l1865_186590


namespace NUMINAMATH_GPT_cost_of_27_lilies_l1865_186580

theorem cost_of_27_lilies
  (cost_18 : ℕ)
  (price_ratio : ℕ → ℕ → Prop)
  (h_cost_18 : cost_18 = 30)
  (h_price_ratio : ∀ n m c : ℕ, price_ratio n m ↔ c = n * 5 / 3 ∧ m = c * 3 / 5) :
  ∃ c : ℕ, price_ratio 27 c ∧ c = 45 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_27_lilies_l1865_186580


namespace NUMINAMATH_GPT_inequality_proof_l1865_186586

open Real

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1865_186586


namespace NUMINAMATH_GPT_sally_money_l1865_186576

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end NUMINAMATH_GPT_sally_money_l1865_186576


namespace NUMINAMATH_GPT_average_score_of_class_l1865_186515

theorem average_score_of_class : 
  ∀ (total_students assigned_students make_up_students : ℕ)
    (assigned_avg_score make_up_avg_score : ℚ),
    total_students = 100 →
    assigned_students = 70 →
    make_up_students = total_students - assigned_students →
    assigned_avg_score = 60 →
    make_up_avg_score = 80 →
    (assigned_students * assigned_avg_score + make_up_students * make_up_avg_score) / total_students = 66 :=
by
  intro total_students assigned_students make_up_students assigned_avg_score make_up_avg_score
  intros h_total_students h_assigned_students h_make_up_students h_assigned_avg_score h_make_up_avg_score
  sorry

end NUMINAMATH_GPT_average_score_of_class_l1865_186515


namespace NUMINAMATH_GPT_last_digit_of_expression_l1865_186567

-- Conditions
def a : ℤ := 25
def b : ℤ := -3

-- Statement to be proved
theorem last_digit_of_expression :
  (a ^ 1999 + b ^ 2002) % 10 = 4 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_last_digit_of_expression_l1865_186567


namespace NUMINAMATH_GPT_min_value_inverse_sum_l1865_186534

theorem min_value_inverse_sum (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a + 2 * b = 1) : 
  ∃ (y : ℝ), y = 3 + 2 * Real.sqrt 2 ∧ (∀ x, x = (1 / a) + (1 / b) → y ≤ x) :=
sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l1865_186534


namespace NUMINAMATH_GPT_triangle_even_number_in_each_row_from_third_l1865_186525

/-- Each number in the (n+1)-th row of the triangle is the sum of three numbers 
  from the n-th row directly above this number and its immediate left and right neighbors.
  If such neighbors do not exist, they are considered as zeros.
  Prove that in each row of the triangle, starting from the third row,
  there is at least one even number. -/

theorem triangle_even_number_in_each_row_from_third (triangle : ℕ → ℕ → ℕ) :
  (∀ n i : ℕ, i > n → triangle n i = 0) →
  (∀ n i : ℕ, triangle (n+1) i = triangle n (i-1) + triangle n i + triangle n (i+1)) →
  ∀ n : ℕ, n ≥ 2 → ∃ i : ℕ, i ≤ n ∧ 2 ∣ triangle n i :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_even_number_in_each_row_from_third_l1865_186525


namespace NUMINAMATH_GPT_Q_gets_less_than_P_l1865_186502

theorem Q_gets_less_than_P (x : Real) (hx : x > 0) (hP : P = 1.25 * x): 
  Q = P * 0.8 := 
sorry

end NUMINAMATH_GPT_Q_gets_less_than_P_l1865_186502


namespace NUMINAMATH_GPT_simplify_polynomial_l1865_186527

theorem simplify_polynomial : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1865_186527


namespace NUMINAMATH_GPT_bike_price_l1865_186535

theorem bike_price (x : ℝ) (h1 : 0.1 * x = 150) : x = 1500 := 
by sorry

end NUMINAMATH_GPT_bike_price_l1865_186535


namespace NUMINAMATH_GPT_like_terms_exponents_product_l1865_186575

theorem like_terms_exponents_product (m n : ℤ) (a b : ℝ) 
  (h1 : 3 * a^m * b^2 = -1 * a^2 * b^(n+3)) : m * n = -2 :=
  sorry

end NUMINAMATH_GPT_like_terms_exponents_product_l1865_186575


namespace NUMINAMATH_GPT_cos_of_angle_in_third_quadrant_l1865_186593

theorem cos_of_angle_in_third_quadrant (A : ℝ) (hA : π < A ∧ A < 3 * π / 2) (h_sin : Real.sin A = -1 / 3) :
  Real.cos A = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_in_third_quadrant_l1865_186593


namespace NUMINAMATH_GPT_infinite_solutions_implies_a_eq_2_l1865_186514

theorem infinite_solutions_implies_a_eq_2 (a b : ℝ) (h : b = 1) :
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → a = 2 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_infinite_solutions_implies_a_eq_2_l1865_186514


namespace NUMINAMATH_GPT_polynomial_evaluation_l1865_186539

-- Define the polynomial p(x) and the conditions
noncomputable def p (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

-- Given conditions for p(1), p(2), p(3)
variables (a b c d : ℝ)
axiom h₁ : p 1 a b c d = 1993
axiom h₂ : p 2 a b c d = 3986
axiom h₃ : p 3 a b c d = 5979

-- The final proof statement
theorem polynomial_evaluation :
  (1 / 4) * (p 11 a b c d + p (-7) a b c d) = 5233 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1865_186539


namespace NUMINAMATH_GPT_sine_product_identity_l1865_186563

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end NUMINAMATH_GPT_sine_product_identity_l1865_186563


namespace NUMINAMATH_GPT_remaining_days_temperature_l1865_186533

theorem remaining_days_temperature :
  let avg_temp := 60
  let total_days := 7
  let temp_day1 := 40
  let temp_day2 := 40
  let temp_day3 := 40
  let temp_day4 := 80
  let temp_day5 := 80
  let total_temp := avg_temp * total_days
  let temp_first_five_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5
  total_temp - temp_first_five_days = 140 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_remaining_days_temperature_l1865_186533


namespace NUMINAMATH_GPT_age_problem_l1865_186560

theorem age_problem (S Sh K : ℕ) 
  (h1 : S / Sh = 4 / 3)
  (h2 : S / K = 4 / 2)
  (h3 : K + 10 = S)
  (h4 : S + 8 = 30) :
  S = 22 ∧ Sh = 17 ∧ K = 10 := 
sorry

end NUMINAMATH_GPT_age_problem_l1865_186560


namespace NUMINAMATH_GPT_height_of_fourth_person_l1865_186561

theorem height_of_fourth_person 
  (H : ℕ) 
  (h_avg : ((H) + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) :
  (H + 10 = 85) :=
by
  sorry

end NUMINAMATH_GPT_height_of_fourth_person_l1865_186561


namespace NUMINAMATH_GPT_increasing_function_range_l1865_186519

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 - 2 * x + Real.log x

theorem increasing_function_range (m : ℝ) : (∀ x > 0, m * x + (1 / x) - 2 ≥ 0) ↔ m ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_increasing_function_range_l1865_186519


namespace NUMINAMATH_GPT_seashells_after_giving_away_l1865_186581

-- Define the given conditions
def initial_seashells : ℕ := 79
def given_away_seashells : ℕ := 63

-- State the proof problem
theorem seashells_after_giving_away : (initial_seashells - given_away_seashells) = 16 :=
  by 
    sorry

end NUMINAMATH_GPT_seashells_after_giving_away_l1865_186581


namespace NUMINAMATH_GPT_distributive_property_example_l1865_186565

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = (3/4) * (-36) + (7/12) * (-36) - (5/9) * (-36) :=
by
  sorry

end NUMINAMATH_GPT_distributive_property_example_l1865_186565


namespace NUMINAMATH_GPT_sin_alpha_eq_sqrt_5_div_3_l1865_186589

variable (α : ℝ)

theorem sin_alpha_eq_sqrt_5_div_3
  (hα : 0 < α ∧ α < Real.pi)
  (h : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_eq_sqrt_5_div_3_l1865_186589


namespace NUMINAMATH_GPT_rhombus_difference_l1865_186569

theorem rhombus_difference (n : ℕ) (h : n > 3)
    (m : ℕ := 3 * (n - 1) * n / 2)
    (d : ℕ := 3 * (n - 3) * (n - 2) / 2) :
    m - d = 6 * n - 9 := by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_rhombus_difference_l1865_186569


namespace NUMINAMATH_GPT_n_minus_k_minus_l_square_number_l1865_186529

variable (n k l x : ℕ)

theorem n_minus_k_minus_l_square_number (h1 : x^2 < n)
                                        (h2 : n < (x + 1)^2)
                                        (h3 : n - k = x^2)
                                        (h4 : n + l = (x + 1)^2) :
  ∃ m : ℕ, n - k - l = m ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_n_minus_k_minus_l_square_number_l1865_186529


namespace NUMINAMATH_GPT_simplify_sum_of_polynomials_l1865_186547

-- Definitions of the given polynomials
def P (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15
def Q (x : ℝ) : ℝ := -5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9

-- Statement to prove that the sum of P and Q equals the simplified polynomial
theorem simplify_sum_of_polynomials (x : ℝ) : 
  P x + Q x = 2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := 
sorry

end NUMINAMATH_GPT_simplify_sum_of_polynomials_l1865_186547


namespace NUMINAMATH_GPT_prob_of_different_colors_l1865_186506

def total_balls_A : ℕ := 4 + 5 + 6
def total_balls_B : ℕ := 7 + 6 + 2

noncomputable def prob_same_color : ℚ :=
  (4 / ↑total_balls_A * 7 / ↑total_balls_B) +
  (5 / ↑total_balls_A * 6 / ↑total_balls_B) +
  (6 / ↑total_balls_A * 2 / ↑total_balls_B)

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_of_different_colors :
  prob_different_color = 31 / 45 :=
by
  sorry

end NUMINAMATH_GPT_prob_of_different_colors_l1865_186506


namespace NUMINAMATH_GPT_candy_cost_55_cents_l1865_186574

theorem candy_cost_55_cents
  (paid: ℕ) (change: ℕ) (num_coins: ℕ)
  (coin1 coin2 coin3 coin4: ℕ)
  (h1: paid = 100)
  (h2: num_coins = 4)
  (h3: coin1 = 25)
  (h4: coin2 = 10)
  (h5: coin3 = 10)
  (h6: coin4 = 0)
  (h7: change = coin1 + coin2 + coin3 + coin4) :
  paid - change = 55 :=
by
  -- The proof can be provided here.
  sorry

end NUMINAMATH_GPT_candy_cost_55_cents_l1865_186574


namespace NUMINAMATH_GPT_number_of_buses_l1865_186588

theorem number_of_buses (vans people_per_van buses people_per_bus extra_people_in_buses : ℝ) 
  (h_vans : vans = 6.0) 
  (h_people_per_van : people_per_van = 6.0) 
  (h_people_per_bus : people_per_bus = 18.0) 
  (h_extra_people_in_buses : extra_people_in_buses = 108.0) 
  (h_eq : people_per_bus * buses = vans * people_per_van + extra_people_in_buses) : 
  buses = 8.0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_buses_l1865_186588


namespace NUMINAMATH_GPT_pima_investment_value_l1865_186510

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end NUMINAMATH_GPT_pima_investment_value_l1865_186510


namespace NUMINAMATH_GPT_distance_traveled_by_second_hand_l1865_186584

def second_hand_length : ℝ := 8
def time_period_minutes : ℝ := 45
def rotations_per_minute : ℝ := 1

theorem distance_traveled_by_second_hand :
  let circumference := 2 * Real.pi * second_hand_length
  let rotations := time_period_minutes * rotations_per_minute
  let total_distance := rotations * circumference
  total_distance = 720 * Real.pi := by
  sorry

end NUMINAMATH_GPT_distance_traveled_by_second_hand_l1865_186584
