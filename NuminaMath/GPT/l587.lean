import Mathlib

namespace NUMINAMATH_GPT_david_age_l587_58707

theorem david_age (A B C D : ℕ)
  (h1 : A = B - 5)
  (h2 : B = C + 2)
  (h3 : D = C + 4)
  (h4 : A = 12) : D = 19 :=
sorry

end NUMINAMATH_GPT_david_age_l587_58707


namespace NUMINAMATH_GPT_westbound_cyclist_speed_increase_l587_58701

def eastbound_speed : ℕ := 18
def travel_time : ℕ := 6
def total_distance : ℕ := 246

theorem westbound_cyclist_speed_increase (x : ℕ) :
  eastbound_speed * travel_time + (eastbound_speed + x) * travel_time = total_distance →
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_westbound_cyclist_speed_increase_l587_58701


namespace NUMINAMATH_GPT_negative_solution_iff_sum_zero_l587_58730

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_solution_iff_sum_zero_l587_58730


namespace NUMINAMATH_GPT_separation_of_homologous_chromosomes_only_in_meiosis_l587_58760

-- We start by defining the conditions extracted from the problem.
def chromosome_replication (phase: String) : Prop :=  
  phase = "S phase"

def separation_of_homologous_chromosomes (process: String) : Prop := 
  process = "meiosis I"

def separation_of_chromatids (process: String) : Prop := 
  process = "mitosis anaphase" ∨ process = "meiosis II anaphase II"

def cytokinesis (end_phase: String) : Prop := 
  end_phase = "end mitosis" ∨ end_phase = "end meiosis"

-- Now, we state that the separation of homologous chromosomes does not occur during mitosis.
theorem separation_of_homologous_chromosomes_only_in_meiosis :
  ∀ (process: String), ¬ separation_of_homologous_chromosomes "mitosis" := 
sorry

end NUMINAMATH_GPT_separation_of_homologous_chromosomes_only_in_meiosis_l587_58760


namespace NUMINAMATH_GPT_accounting_major_students_count_l587_58736

theorem accounting_major_students_count (p q r s: ℕ) (h1: p * q * r * s = 1365) (h2: 1 < p) (h3: p < q) (h4: q < r) (h5: r < s):
  p = 3 :=
sorry

end NUMINAMATH_GPT_accounting_major_students_count_l587_58736


namespace NUMINAMATH_GPT_x_cubed_plus_y_cubed_l587_58779

theorem x_cubed_plus_y_cubed:
  ∀ (x y : ℝ), (x * (x ^ 4 + y ^ 4) = y ^ 5) → (x ^ 2 * (x + y) ≠ y ^ 3) → (x ^ 3 + y ^ 3 = 1) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_x_cubed_plus_y_cubed_l587_58779


namespace NUMINAMATH_GPT_triangle_is_isosceles_l587_58733

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC_sum : A + B + C = π) 
  (cos_rule : a * Real.cos B + b * Real.cos A = a) :
  a = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l587_58733


namespace NUMINAMATH_GPT_lucas_payment_l587_58759

noncomputable def payment (windows_per_floor : ℕ) (floors : ℕ) (days : ℕ) 
  (earn_per_window : ℝ) (delay_penalty : ℝ) (period : ℕ) : ℝ :=
  let total_windows := windows_per_floor * floors
  let earnings := total_windows * earn_per_window
  let penalty_periods := days / period
  let total_penalty := penalty_periods * delay_penalty
  earnings - total_penalty

theorem lucas_payment :
  payment 3 3 6 2 1 3 = 16 := by
  sorry

end NUMINAMATH_GPT_lucas_payment_l587_58759


namespace NUMINAMATH_GPT_compound_interest_eq_440_l587_58738

-- Define the conditions
variables (P R T SI CI : ℝ)
variables (H_SI : SI = P * R * T / 100)
variables (H_R : R = 20)
variables (H_T : T = 2)
variables (H_given : SI = 400)
variables (H_question : CI = P * (1 + R / 100)^T - P)

-- Define the goal to prove
theorem compound_interest_eq_440 : CI = 440 :=
by
  -- Conditions and the result should be proved here, but we'll use sorry to skip the proof step.
  sorry

end NUMINAMATH_GPT_compound_interest_eq_440_l587_58738


namespace NUMINAMATH_GPT_technicians_in_workshop_l587_58756

theorem technicians_in_workshop :
  (∃ T R: ℕ, T + R = 42 ∧ 8000 * 42 = 18000 * T + 6000 * R) → ∃ T: ℕ, T = 7 :=
by
  sorry

end NUMINAMATH_GPT_technicians_in_workshop_l587_58756


namespace NUMINAMATH_GPT_length_of_platform_is_280_l587_58783

-- Add conditions for speed, times and conversions
def speed_kmph : ℕ := 72
def time_platform : ℕ := 30
def time_man : ℕ := 16

-- Conversion from km/h to m/s
def speed_mps : ℤ := speed_kmph * 1000 / 3600

-- The length of the train when it crosses the man
def length_of_train : ℤ := speed_mps * time_man

-- The length of the platform
def length_of_platform : ℤ := (speed_mps * time_platform) - length_of_train

theorem length_of_platform_is_280 :
  length_of_platform = 280 := by
  sorry

end NUMINAMATH_GPT_length_of_platform_is_280_l587_58783


namespace NUMINAMATH_GPT_sum_even_probability_l587_58731

def probability_even_sum_of_wheels : ℚ :=
  let prob_wheel1_odd := 3 / 5
  let prob_wheel1_even := 2 / 5
  let prob_wheel2_odd := 2 / 3
  let prob_wheel2_even := 1 / 3
  (prob_wheel1_odd * prob_wheel2_odd) + (prob_wheel1_even * prob_wheel2_even)

theorem sum_even_probability :
  probability_even_sum_of_wheels = 8 / 15 :=
by
  -- Goal statement with calculations showed in the equivalent problem
  sorry

end NUMINAMATH_GPT_sum_even_probability_l587_58731


namespace NUMINAMATH_GPT_average_of_distinct_s_values_l587_58718

theorem average_of_distinct_s_values : 
  (1 + 5 + 2 + 4 + 3 + 3 + 4 + 2 + 5 + 1) / 3 = 7.33 :=
by
  sorry

end NUMINAMATH_GPT_average_of_distinct_s_values_l587_58718


namespace NUMINAMATH_GPT_angle_R_values_l587_58769

theorem angle_R_values (P Q : ℝ) (h1: 5 * Real.sin P + 2 * Real.cos Q = 5) (h2: 2 * Real.sin Q + 5 * Real.cos P = 3) : 
  ∃ R : ℝ, R = Real.arcsin (1/20) ∨ R = 180 - Real.arcsin (1/20) :=
by
  sorry

end NUMINAMATH_GPT_angle_R_values_l587_58769


namespace NUMINAMATH_GPT_caleb_grandfather_age_l587_58750

theorem caleb_grandfather_age :
  let yellow_candles := 27
  let red_candles := 14
  let blue_candles := 38
  yellow_candles + red_candles + blue_candles = 79 :=
by
  sorry

end NUMINAMATH_GPT_caleb_grandfather_age_l587_58750


namespace NUMINAMATH_GPT_polygon_interior_angles_540_implies_5_sides_l587_58767

theorem polygon_interior_angles_540_implies_5_sides (n : ℕ) :
  (n - 2) * 180 = 540 → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_angles_540_implies_5_sides_l587_58767


namespace NUMINAMATH_GPT_bert_ernie_ratio_l587_58751

theorem bert_ernie_ratio (berts_stamps ernies_stamps peggys_stamps : ℕ) 
  (h1 : peggys_stamps = 75) 
  (h2 : ernies_stamps = 3 * peggys_stamps) 
  (h3 : berts_stamps = peggys_stamps + 825) : 
  berts_stamps / ernies_stamps = 4 := 
by sorry

end NUMINAMATH_GPT_bert_ernie_ratio_l587_58751


namespace NUMINAMATH_GPT_expression_evaluation_l587_58799

theorem expression_evaluation (a : ℝ) (h : a = 9) : ( (a ^ (1 / 3)) / (a ^ (1 / 5)) ) = a^(2 / 15) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l587_58799


namespace NUMINAMATH_GPT_sin_four_thirds_pi_l587_58781

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_four_thirds_pi_l587_58781


namespace NUMINAMATH_GPT_side_length_of_square_l587_58743

theorem side_length_of_square (P : ℝ) (h1 : P = 12 / 25) : 
  P / 4 = 0.12 := 
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l587_58743


namespace NUMINAMATH_GPT_probability_interval_l587_58792

-- Define the probability distribution and conditions
def P (xi : ℕ) (c : ℚ) : ℚ := c / (xi * (xi + 1))

-- Given conditions
variables (c : ℚ)
axiom condition : P 1 c + P 2 c + P 3 c + P 4 c = 1

-- Define the interval probability
def interval_prob (c : ℚ) : ℚ := P 1 c + P 2 c

-- Prove that the computed probability matches the expected value
theorem probability_interval : interval_prob (5 / 4) = 5 / 6 :=
by
  -- skip proof
  sorry

end NUMINAMATH_GPT_probability_interval_l587_58792


namespace NUMINAMATH_GPT_sufficient_condition_l587_58724

theorem sufficient_condition (a b : ℝ) (h : b > a ∧ a > 0) : (a + 2) / (b + 2) > a / b :=
by sorry

end NUMINAMATH_GPT_sufficient_condition_l587_58724


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l587_58715

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l587_58715


namespace NUMINAMATH_GPT_not_partitionable_1_to_15_l587_58723

theorem not_partitionable_1_to_15 :
  ∀ (A B : Finset ℕ), (∀ x ∈ A, x ∈ Finset.range 16) →
    (∀ x ∈ B, x ∈ Finset.range 16) →
    A.card = 2 → B.card = 13 →
    A ∪ B = Finset.range 16 →
    ¬(A.sum id = B.prod id) :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_not_partitionable_1_to_15_l587_58723


namespace NUMINAMATH_GPT_incorrect_parallel_m_n_l587_58717

variables {l m n : Type} [LinearOrder m] [LinearOrder n] {α β : Type}

-- Assumptions for parallelism and orthogonality
def parallel (x y : Type) : Prop := sorry
def orthogonal (x y : Type) : Prop := sorry

-- Conditions
axiom parallel_m_l : parallel m l
axiom parallel_n_l : parallel n l
axiom orthogonal_m_α : orthogonal m α
axiom parallel_m_β : parallel m β
axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom orthogonal_m_β : orthogonal m β
axiom orthogonal_α_β : orthogonal α β

-- The theorem to prove
theorem incorrect_parallel_m_n : parallel m α ∧ parallel n α → ¬ parallel m n := sorry

end NUMINAMATH_GPT_incorrect_parallel_m_n_l587_58717


namespace NUMINAMATH_GPT_binomial_expansion_max_coefficient_l587_58794

theorem binomial_expansion_max_coefficient (n : ℕ) (h : n > 0) 
  (h_max_coefficient: ∀ m : ℕ, m ≠ 5 → (Nat.choose n m ≤ Nat.choose n 5)) : 
  n = 10 :=
sorry

end NUMINAMATH_GPT_binomial_expansion_max_coefficient_l587_58794


namespace NUMINAMATH_GPT_directrix_of_parabola_l587_58772

theorem directrix_of_parabola (x y : ℝ) : (y^2 = 8*x) → (x = -2) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l587_58772


namespace NUMINAMATH_GPT_problem1_problem2_l587_58749

-- Problem 1:
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

-- Problem 2:
theorem problem2 (α : ℝ) : 
  (Real.tan (2 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α + Real.pi) * Real.sin (-Real.pi + α)) = 1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l587_58749


namespace NUMINAMATH_GPT_time_to_school_l587_58777

theorem time_to_school (total_distance walk_speed run_speed distance_ran : ℕ) (h_total : total_distance = 1800)
    (h_walk_speed : walk_speed = 70) (h_run_speed : run_speed = 210) (h_distance_ran : distance_ran = 600) :
    total_distance / walk_speed + distance_ran / run_speed = 20 := by
  sorry

end NUMINAMATH_GPT_time_to_school_l587_58777


namespace NUMINAMATH_GPT_value_of_expression_l587_58763

theorem value_of_expression (m n : ℝ) (h : m + n = 4) : 2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 :=
  sorry

end NUMINAMATH_GPT_value_of_expression_l587_58763


namespace NUMINAMATH_GPT_arun_age_proof_l587_58729

theorem arun_age_proof {A G M : ℕ} 
  (h1 : (A - 6) / 18 = G)
  (h2 : G = M - 2)
  (h3 : M = 5) :
  A = 60 :=
by
  sorry

end NUMINAMATH_GPT_arun_age_proof_l587_58729


namespace NUMINAMATH_GPT_distance_to_fourth_side_l587_58752

theorem distance_to_fourth_side (s : ℕ) (d1 d2 d3 : ℕ) (x : ℕ) 
  (cond1 : d1 = 4) (cond2 : d2 = 7) (cond3 : d3 = 12)
  (h : d1 + d2 + d3 + x = s) : x = 9 ∨ x = 15 :=
  sorry

end NUMINAMATH_GPT_distance_to_fourth_side_l587_58752


namespace NUMINAMATH_GPT_eve_walked_distance_l587_58728

-- Defining the distances Eve ran and walked
def distance_ran : ℝ := 0.7
def distance_walked : ℝ := distance_ran - 0.1

-- Proving that the distance Eve walked is 0.6 mile
theorem eve_walked_distance : distance_walked = 0.6 := by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_eve_walked_distance_l587_58728


namespace NUMINAMATH_GPT_jim_taxi_total_charge_l587_58727

noncomputable def total_charge (initial_fee : ℝ) (per_mile_fee : ℝ) (mile_chunk : ℝ) (distance : ℝ) : ℝ :=
  initial_fee + (distance / mile_chunk) * per_mile_fee

theorem jim_taxi_total_charge :
  total_charge 2.35 0.35 (2/5) 3.6 = 5.50 :=
by
  sorry

end NUMINAMATH_GPT_jim_taxi_total_charge_l587_58727


namespace NUMINAMATH_GPT_parallelogram_area_l587_58719

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (base_condition : b = 8) (altitude_condition : h = 2 * b) : 
  A = 128 :=
by 
  sorry

end NUMINAMATH_GPT_parallelogram_area_l587_58719


namespace NUMINAMATH_GPT_r_exceeds_s_l587_58764

theorem r_exceeds_s (x y : ℚ) (h1 : x + 2 * y = 16 / 3) (h2 : 5 * x + 3 * y = 26) :
  x - y = 106 / 21 :=
sorry

end NUMINAMATH_GPT_r_exceeds_s_l587_58764


namespace NUMINAMATH_GPT_find_integers_l587_58702

theorem find_integers (a b m : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l587_58702


namespace NUMINAMATH_GPT_solve_log_eq_l587_58732

noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

theorem solve_log_eq :
  (∃ x : ℝ, log3 ((5 * x + 15) / (7 * x - 5)) + log3 ((7 * x - 5) / (2 * x - 3)) = 3 ∧ x = 96 / 49) :=
by
  sorry

end NUMINAMATH_GPT_solve_log_eq_l587_58732


namespace NUMINAMATH_GPT_smallest_points_to_guarantee_victory_l587_58782

noncomputable def pointsForWinning : ℕ := 5
noncomputable def pointsForSecond : ℕ := 3
noncomputable def pointsForThird : ℕ := 1

theorem smallest_points_to_guarantee_victory :
  ∀ (student_points : ℕ),
  (exists (x y z : ℕ), (x = pointsForWinning ∨ x = pointsForSecond ∨ x = pointsForThird) ∧
                         (y = pointsForWinning ∨ y = pointsForSecond ∨ y = pointsForThird) ∧
                         (z = pointsForWinning ∨ z = pointsForSecond ∨ z = pointsForThird) ∧
                         student_points = x + y + z) →
  (∃ (victory_points : ℕ), victory_points = 13) →
  (∀ other_points : ℕ, other_points < victory_points) :=
sorry

end NUMINAMATH_GPT_smallest_points_to_guarantee_victory_l587_58782


namespace NUMINAMATH_GPT_find_second_quadrant_point_l587_58739

def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem find_second_quadrant_point :
  (is_second_quadrant (2, 3) = false) ∧
  (is_second_quadrant (2, -3) = false) ∧
  (is_second_quadrant (-2, -3) = false) ∧
  (is_second_quadrant (-2, 3) = true) := 
sorry

end NUMINAMATH_GPT_find_second_quadrant_point_l587_58739


namespace NUMINAMATH_GPT_find_d_not_unique_solution_l587_58755

variable {x y k d : ℝ}

-- Definitions of the conditions
def eq1 (d : ℝ) (x y : ℝ) := 4 * (3 * x + 4 * y) = d
def eq2 (k : ℝ) (x y : ℝ) := k * x + 12 * y = 30

-- The theorem we need to prove
theorem find_d_not_unique_solution (h1: eq1 d x y) (h2: eq2 k x y) (h3 : ¬ ∃! (x y : ℝ), eq1 d x y ∧ eq2 k x y) : d = 40 := 
by
  sorry

end NUMINAMATH_GPT_find_d_not_unique_solution_l587_58755


namespace NUMINAMATH_GPT_double_theta_acute_l587_58734

theorem double_theta_acute (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end NUMINAMATH_GPT_double_theta_acute_l587_58734


namespace NUMINAMATH_GPT_problem_statement_l587_58796

def scientific_notation_correct (x : ℝ) : Prop :=
  x = 5.642 * 10 ^ 5

theorem problem_statement : scientific_notation_correct 564200 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l587_58796


namespace NUMINAMATH_GPT_each_baby_worms_per_day_l587_58775

variable (babies : Nat) (worms_papa : Nat) (worms_mama_caught : Nat) (worms_mama_stolen : Nat) (worms_needed : Nat)
variable (days : Nat)

theorem each_baby_worms_per_day 
  (h1 : babies = 6) 
  (h2 : worms_papa = 9) 
  (h3 : worms_mama_caught = 13) 
  (h4 : worms_mama_stolen = 2)
  (h5 : worms_needed = 34) 
  (h6 : days = 3) :
  (worms_papa + (worms_mama_caught - worms_mama_stolen) + worms_needed) / babies / days = 3 :=
by
  sorry

end NUMINAMATH_GPT_each_baby_worms_per_day_l587_58775


namespace NUMINAMATH_GPT_john_can_see_jane_for_45_minutes_l587_58757

theorem john_can_see_jane_for_45_minutes :
  ∀ (john_speed : ℝ) (jane_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ),
  john_speed = 7 →
  jane_speed = 3 →
  initial_distance = 1 →
  final_distance = 2 →
  (initial_distance / (john_speed - jane_speed) + final_distance / (john_speed - jane_speed)) * 60 = 45 :=
by
  intros john_speed jane_speed initial_distance final_distance
  sorry

end NUMINAMATH_GPT_john_can_see_jane_for_45_minutes_l587_58757


namespace NUMINAMATH_GPT_jaysons_moms_age_l587_58713

theorem jaysons_moms_age (jayson's_age dad's_age mom's_age : ℕ) 
  (h1 : jayson's_age = 10)
  (h2 : dad's_age = 4 * jayson's_age)
  (h3 : mom's_age = dad's_age - 2) :
  mom's_age - jayson's_age = 28 := 
by
  sorry

end NUMINAMATH_GPT_jaysons_moms_age_l587_58713


namespace NUMINAMATH_GPT_wall_length_proof_l587_58798

-- Define the conditions from the problem
def wall_height : ℝ := 100 -- Height in cm
def wall_thickness : ℝ := 5 -- Thickness in cm
def brick_length : ℝ := 25 -- Brick length in cm
def brick_width : ℝ := 11 -- Brick width in cm
def brick_height : ℝ := 6 -- Brick height in cm
def number_of_bricks : ℝ := 242.42424242424244

-- Calculate the volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Calculate the total volume of the bricks
def total_brick_volume : ℝ := brick_volume * number_of_bricks

-- Define the proof problem
theorem wall_length_proof : total_brick_volume = wall_height * wall_thickness * 800 :=
sorry

end NUMINAMATH_GPT_wall_length_proof_l587_58798


namespace NUMINAMATH_GPT_units_digit_of_3_pow_7_pow_6_l587_58780

theorem units_digit_of_3_pow_7_pow_6 :
  (3 ^ (7 ^ 6) % 10) = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_of_3_pow_7_pow_6_l587_58780


namespace NUMINAMATH_GPT_cab_time_l587_58774

theorem cab_time (d t : ℝ) (v : ℝ := d / t)
    (v1 : ℝ := (5 / 6) * v)
    (t1 : ℝ := d / v1)
    (v2 : ℝ := (2 / 3) * v)
    (t2 : ℝ := d / v2)
    (T : ℝ := t1 + t2)
    (delay : ℝ := 5) :
    let total_time := 2 * t + delay
    t * d ≠ 0 → T = total_time → t = 50 / 7 := by
    sorry

end NUMINAMATH_GPT_cab_time_l587_58774


namespace NUMINAMATH_GPT_divisible_by_6_l587_58787

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end NUMINAMATH_GPT_divisible_by_6_l587_58787


namespace NUMINAMATH_GPT_dice_sum_probability_l587_58758

def four_dice_probability_sum_to_remain_die : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 4 * 120
  favorable_outcomes / total_outcomes

theorem dice_sum_probability : four_dice_probability_sum_to_remain_die = 10 / 27 :=
  sorry

end NUMINAMATH_GPT_dice_sum_probability_l587_58758


namespace NUMINAMATH_GPT_find_x_l587_58754

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end NUMINAMATH_GPT_find_x_l587_58754


namespace NUMINAMATH_GPT_pq_sum_equals_4_l587_58765

theorem pq_sum_equals_4 (p q : ℝ) (h : (Polynomial.C 1 + Polynomial.C q * Polynomial.X + Polynomial.C p * Polynomial.X^2 + Polynomial.X^4).eval (2 + I) = 0) :
  p + q = 4 :=
sorry

end NUMINAMATH_GPT_pq_sum_equals_4_l587_58765


namespace NUMINAMATH_GPT_beta_angle_relationship_l587_58784

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end NUMINAMATH_GPT_beta_angle_relationship_l587_58784


namespace NUMINAMATH_GPT_Crimson_Valley_skirts_l587_58726

theorem Crimson_Valley_skirts
  (Azure_Valley_skirts : ℕ)
  (Seafoam_Valley_skirts : ℕ)
  (Purple_Valley_skirts : ℕ)
  (Crimson_Valley_skirts : ℕ)
  (h1 : Azure_Valley_skirts = 90)
  (h2 : Seafoam_Valley_skirts = (2/3 : ℚ) * Azure_Valley_skirts)
  (h3 : Purple_Valley_skirts = (1/4 : ℚ) * Seafoam_Valley_skirts)
  (h4 : Crimson_Valley_skirts = (1/3 : ℚ) * Purple_Valley_skirts)
  : Crimson_Valley_skirts = 5 := 
sorry

end NUMINAMATH_GPT_Crimson_Valley_skirts_l587_58726


namespace NUMINAMATH_GPT_sum_of_fraction_equiv_l587_58742

theorem sum_of_fraction_equiv : 
  let x := 3.714714714
  let num := 3711
  let denom := 999
  3711 + 999 = 4710 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_fraction_equiv_l587_58742


namespace NUMINAMATH_GPT_find_group_2018_l587_58766

theorem find_group_2018 :
  ∃ n : ℕ, 2 ≤ n ∧ 2018 ≤ 2 * n * (n + 1) ∧ 2018 > 2 * (n - 1) * n :=
by
  sorry

end NUMINAMATH_GPT_find_group_2018_l587_58766


namespace NUMINAMATH_GPT_complex_expr_simplify_l587_58720

noncomputable def complex_demo : Prop :=
  let i := Complex.I
  7 * (4 + 2 * i) - 2 * i * (7 + 3 * i) = (34 : ℂ)

theorem complex_expr_simplify : 
  complex_demo :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_complex_expr_simplify_l587_58720


namespace NUMINAMATH_GPT_max_value_of_expressions_l587_58722

theorem max_value_of_expressions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2 * a * b ∧ b > a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expressions_l587_58722


namespace NUMINAMATH_GPT_average_speed_with_stoppages_l587_58770

theorem average_speed_with_stoppages
    (D : ℝ) -- distance the train travels
    (T_no_stop : ℝ := D / 250) -- time taken to cover the distance without stoppages
    (T_with_stop : ℝ := 2 * T_no_stop) -- total time with stoppages
    : (D / T_with_stop) = 125 := 
by sorry

end NUMINAMATH_GPT_average_speed_with_stoppages_l587_58770


namespace NUMINAMATH_GPT_smallest_digits_to_append_l587_58712

theorem smallest_digits_to_append (n : ℕ) : ∃ d, d ≤ 4 ∧ ∃ k, (2014 * 10^d + k) % 2520 = 0 :=
  sorry

end NUMINAMATH_GPT_smallest_digits_to_append_l587_58712


namespace NUMINAMATH_GPT_find_expression_value_l587_58773

theorem find_expression_value 
  (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := 
by 
  sorry

end NUMINAMATH_GPT_find_expression_value_l587_58773


namespace NUMINAMATH_GPT_contrapositive_proposition_l587_58703

theorem contrapositive_proposition (x : ℝ) : (x > 10 → x > 1) ↔ (x ≤ 1 → x ≤ 10) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l587_58703


namespace NUMINAMATH_GPT_max_volume_range_of_a_x1_x2_inequality_l587_58795

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (a * x^2) - Real.exp 1 * x + a * x^2 - 1) / x

theorem max_volume (x : ℝ) (hx : 1 < x) :
  ∃ V : ℝ, V = (Real.pi / 3) * ((Real.log x)^2 / x) ∧ V = (4 * Real.pi / (3 * (Real.exp 2)^2)) :=
sorry

theorem range_of_a (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  0 < a ∧ a < (1/2) * (Real.exp 1) :=
sorry

theorem x1_x2_inequality (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  x1^2 + x2^2 > 2 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_max_volume_range_of_a_x1_x2_inequality_l587_58795


namespace NUMINAMATH_GPT_subset_M_union_N_l587_58771

theorem subset_M_union_N (M N P : Set ℝ) (f g : ℝ → ℝ)
  (hM : M = {x | f x = 0} ∧ M ≠ ∅)
  (hN : N = {x | g x = 0} ∧ N ≠ ∅)
  (hP : P = {x | f x * g x = 0} ∧ P ≠ ∅) :
  P ⊆ (M ∪ N) := 
sorry

end NUMINAMATH_GPT_subset_M_union_N_l587_58771


namespace NUMINAMATH_GPT_largest_n_exists_unique_k_l587_58708

theorem largest_n_exists_unique_k (n k : ℕ) :
  (∃! k, (8 : ℚ) / 15 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 7 / 13) →
  n ≤ 112 :=
sorry

end NUMINAMATH_GPT_largest_n_exists_unique_k_l587_58708


namespace NUMINAMATH_GPT_brenda_age_l587_58776

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_brenda_age_l587_58776


namespace NUMINAMATH_GPT_floor_ineq_l587_58790

theorem floor_ineq (x y : ℝ) : 
  Int.floor (2 * x) + Int.floor (2 * y) ≥ Int.floor x + Int.floor y + Int.floor (x + y) := 
sorry

end NUMINAMATH_GPT_floor_ineq_l587_58790


namespace NUMINAMATH_GPT_holiday_price_correct_l587_58746

-- Define the problem parameters
def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.10

-- Define the calculation for the first discount
def price_after_first_discount (original: ℝ) (rate: ℝ) : ℝ :=
  original * (1 - rate)

-- Define the calculation for the second discount
def price_after_second_discount (intermediate: ℝ) (rate: ℝ) : ℝ :=
  intermediate * (1 - rate)

-- The final Lean statement to prove
theorem holiday_price_correct : 
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 :=
by
  sorry

end NUMINAMATH_GPT_holiday_price_correct_l587_58746


namespace NUMINAMATH_GPT_circle_intersects_y_axis_with_constraints_l587_58710

theorem circle_intersects_y_axis_with_constraints {m n : ℝ} 
    (H1 : n = m ^ 2 + 2 * m + 2) 
    (H2 : abs m <= 2) : 
    1 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_GPT_circle_intersects_y_axis_with_constraints_l587_58710


namespace NUMINAMATH_GPT_sixth_ninth_grader_buddy_fraction_l587_58778

theorem sixth_ninth_grader_buddy_fraction
  (s n : ℕ)
  (h_fraction_pairs : n / 4 = s / 3)
  (h_buddy_pairing : (∀ i, i < n -> ∃ j, j < s) 
     ∧ (∀ j, j < s -> ∃ i, i < n) -- each sixth grader paired with one ninth grader and vice versa
  ) :
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_sixth_ninth_grader_buddy_fraction_l587_58778


namespace NUMINAMATH_GPT_expected_value_of_coins_is_95_5_l587_58721

-- Define the individual coin values in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50
def dollar_value : ℕ := 100

-- Expected value function with 1/2 probability 
def expected_value (coin_value : ℕ) : ℚ := (coin_value : ℚ) / 2

-- Calculate the total expected value of all coins flipped
noncomputable def total_expected_value : ℚ :=
  expected_value penny_value +
  expected_value nickel_value +
  expected_value dime_value +
  expected_value quarter_value +
  expected_value fifty_cent_value +
  expected_value dollar_value

-- Prove that the expected total value is 95.5
theorem expected_value_of_coins_is_95_5 :
  total_expected_value = 95.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_coins_is_95_5_l587_58721


namespace NUMINAMATH_GPT_frame_painting_ratio_l587_58789

theorem frame_painting_ratio :
  ∃ (x : ℝ), (20 + 2 * x) * (30 + 6 * x) = 1800 → 1 = 2 * (20 + 2 * x) / (30 + 6 * x) :=
by
  sorry

end NUMINAMATH_GPT_frame_painting_ratio_l587_58789


namespace NUMINAMATH_GPT_simplify_subtracted_terms_l587_58744

theorem simplify_subtracted_terms (r : ℝ) : 180 * r - 88 * r = 92 * r := 
by 
  sorry

end NUMINAMATH_GPT_simplify_subtracted_terms_l587_58744


namespace NUMINAMATH_GPT_problem1_problem2_l587_58785

-- problem (1): Prove that if a = 1 and (p ∨ q) is true, then the range of x is 1 < x < 3
def p (a x : ℝ) : Prop := x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : a = 1) (h₂ : p a x ∨ q x) : 
    1 < x ∧ x < 3 :=
sorry

-- problem (2): Prove that if p is a necessary but not sufficient condition for q,
-- then the range of a is 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l587_58785


namespace NUMINAMATH_GPT_equal_share_payments_l587_58714

theorem equal_share_payments (j n : ℝ) 
  (jack_payment : ℝ := 80) 
  (emma_payment : ℝ := 150) 
  (noah_payment : ℝ := 120)
  (liam_payment : ℝ := 200) 
  (total_cost := jack_payment + emma_payment + noah_payment + liam_payment) 
  (individual_share := total_cost / 4) 
  (jack_due := individual_share - jack_payment) 
  (emma_due := emma_payment - individual_share) 
  (noah_due := individual_share - noah_payment) 
  (liam_due := liam_payment - individual_share) 
  (j := jack_due) 
  (n := noah_due) : 
  j - n = 40 := 
by 
  sorry

end NUMINAMATH_GPT_equal_share_payments_l587_58714


namespace NUMINAMATH_GPT_inscribed_circle_radius_l587_58735

theorem inscribed_circle_radius
  (A p s : ℝ) (h1 : A = p) (h2 : s = p / 2) (r : ℝ) (h3 : A = r * s) :
  r = 2 :=
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l587_58735


namespace NUMINAMATH_GPT_ratio_bc_cd_l587_58705

-- Definitions based on given conditions.
variable (a b c d e : ℝ)
variable (h_ab : b - a = 5)
variable (h_ac : c - a = 11)
variable (h_de : e - d = 8)
variable (h_ae : e - a = 22)

-- The theorem to prove bc : cd = 2 : 1.
theorem ratio_bc_cd (h_ab : b - a = 5) (h_ac : c - a = 11) (h_de : e - d = 8) (h_ae : e - a = 22) :
  (c - b) / (d - c) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_bc_cd_l587_58705


namespace NUMINAMATH_GPT_log_sum_eq_two_l587_58711

theorem log_sum_eq_two : 
  ∀ (lg : ℝ → ℝ),
  (∀ x y : ℝ, lg (x * y) = lg x + lg y) →
  (∀ x y : ℝ, lg (x ^ y) = y * lg x) →
  lg 4 + 2 * lg 5 = 2 :=
by
  intros lg h1 h2
  sorry

end NUMINAMATH_GPT_log_sum_eq_two_l587_58711


namespace NUMINAMATH_GPT_trigonometric_identity_l587_58788

variable {θ u : ℝ} {n : ℤ}

-- Given condition
def cos_condition (θ u : ℝ) : Prop := 2 * Real.cos θ = u + (1 / u)

-- Theorem to prove
theorem trigonometric_identity (h : cos_condition θ u) : 2 * Real.cos (n * θ) = u^n + (1 / u^n) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l587_58788


namespace NUMINAMATH_GPT_percentage_of_work_day_in_meetings_is_25_l587_58725

-- Define the conditions
def workDayHours : ℕ := 9
def firstMeetingMinutes : ℕ := 45
def secondMeetingMinutes : ℕ := 2 * firstMeetingMinutes
def totalMeetingMinutes : ℕ := firstMeetingMinutes + secondMeetingMinutes
def workDayMinutes : ℕ := workDayHours * 60

-- Define the percentage calculation
def percentageOfWorkdaySpentInMeetings : ℕ := (totalMeetingMinutes * 100) / workDayMinutes

-- The theorem to be proven
theorem percentage_of_work_day_in_meetings_is_25 :
  percentageOfWorkdaySpentInMeetings = 25 :=
sorry

end NUMINAMATH_GPT_percentage_of_work_day_in_meetings_is_25_l587_58725


namespace NUMINAMATH_GPT_distinct_complex_numbers_no_solution_l587_58700

theorem distinct_complex_numbers_no_solution :
  ¬∃ (a b c d : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (a^3 - b * c * d = b^3 - c * d * a) ∧ 
  (b^3 - c * d * a = c^3 - d * a * b) ∧ 
  (c^3 - d * a * b = d^3 - a * b * c) := 
by {
  sorry
}

end NUMINAMATH_GPT_distinct_complex_numbers_no_solution_l587_58700


namespace NUMINAMATH_GPT_product_of_all_n_satisfying_quadratic_l587_58761

theorem product_of_all_n_satisfying_quadratic :
  (∃ n : ℕ, n^2 - 40 * n + 399 = 3) ∧
  (∀ p : ℕ, Prime p → ((∃ n : ℕ, n^2 - 40 * n + 399 = p) → p = 3)) →
  ∃ n1 n2 : ℕ, (n1^2 - 40 * n1 + 399 = 3) ∧ (n2^2 - 40 * n2 + 399 = 3) ∧ n1 ≠ n2 ∧ (n1 * n2 = 396) :=
by
  sorry

end NUMINAMATH_GPT_product_of_all_n_satisfying_quadratic_l587_58761


namespace NUMINAMATH_GPT_number_of_female_officers_l587_58741

theorem number_of_female_officers (total_on_duty : ℕ) (female_on_duty : ℕ) (percentage_on_duty : ℚ) : 
  total_on_duty = 500 → 
  female_on_duty = 250 → 
  percentage_on_duty = 1/4 → 
  (female_on_duty : ℚ) = percentage_on_duty * (total_on_duty / 2 : ℚ) →
  (total_on_duty : ℚ) = 4 * female_on_duty →
  total_on_duty = 1000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_number_of_female_officers_l587_58741


namespace NUMINAMATH_GPT_chi_square_test_l587_58748

-- Conditions
def n : ℕ := 100
def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25

-- Critical chi-square value for alpha = 0.001
def chi_square_critical : ℝ := 10.828

-- Calculated chi-square value
noncomputable def chi_square_value : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement to prove
theorem chi_square_test : chi_square_value > chi_square_critical :=
by sorry

end NUMINAMATH_GPT_chi_square_test_l587_58748


namespace NUMINAMATH_GPT_machine_part_masses_l587_58791

theorem machine_part_masses :
  ∃ (x y : ℝ), (y - 2 * x = 100) ∧ (875 / x - 900 / y = 3) ∧ (x = 175) ∧ (y = 450) :=
by {
  sorry
}

end NUMINAMATH_GPT_machine_part_masses_l587_58791


namespace NUMINAMATH_GPT_range_of_a_l587_58786

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → 1 ≤ a :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l587_58786


namespace NUMINAMATH_GPT_multiply_res_l587_58737

theorem multiply_res (
  h : 213 * 16 = 3408
) : 1.6 * 213 = 340.8 :=
sorry

end NUMINAMATH_GPT_multiply_res_l587_58737


namespace NUMINAMATH_GPT_sum_of_series_l587_58793

def series_sum : ℕ := 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))

theorem sum_of_series : series_sum = 2730 := by
  -- Expansion: 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))) = 2 + 2 * 4 + 2 * 4^2 + 2 * 4^3 + 2 * 4^4 + 2 * 4^5 
  -- Geometric series sum formula application: S = 2 + 2*4 + 2*4^2 + 2*4^3 + 2*4^4 + 2*4^5 = 2730
  sorry

end NUMINAMATH_GPT_sum_of_series_l587_58793


namespace NUMINAMATH_GPT_carpet_dimensions_l587_58753
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end NUMINAMATH_GPT_carpet_dimensions_l587_58753


namespace NUMINAMATH_GPT_max_value_x2y_l587_58797

theorem max_value_x2y : 
  ∃ (x y : ℕ), 
    7 * x + 4 * y = 140 ∧
    (∀ (x' y' : ℕ),
       7 * x' + 4 * y' = 140 → 
       x' ^ 2 * y' ≤ x ^ 2 * y) ∧
    x ^ 2 * y = 2016 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_x2y_l587_58797


namespace NUMINAMATH_GPT_roger_candies_left_l587_58740

theorem roger_candies_left (initial_candies : ℕ) (to_stephanie : ℕ) (to_john : ℕ) (to_emily : ℕ) : 
  initial_candies = 350 ∧ to_stephanie = 45 ∧ to_john = 25 ∧ to_emily = 18 → 
  initial_candies - (to_stephanie + to_john + to_emily) = 262 :=
by
  sorry

end NUMINAMATH_GPT_roger_candies_left_l587_58740


namespace NUMINAMATH_GPT_parabola_translation_correct_l587_58745

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Given vertex translation
def translated_vertex : ℝ × ℝ := (-2, -2)

-- Define the translated parabola equation
def translated_parabola (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2

-- The proof statement
theorem parabola_translation_correct :
  ∀ x, translated_parabola x = 3 * (x + 2)^2 - 2 := by
  sorry

end NUMINAMATH_GPT_parabola_translation_correct_l587_58745


namespace NUMINAMATH_GPT_inequality_true_l587_58704

theorem inequality_true (a b : ℝ) (h : a > b) : (2 * a - 1) > (2 * b - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_true_l587_58704


namespace NUMINAMATH_GPT_find_positive_integral_solution_l587_58762

theorem find_positive_integral_solution :
  ∃ n : ℕ, n > 0 ∧ (n - 1) * 101 = (n + 1) * 100 := by
sorry

end NUMINAMATH_GPT_find_positive_integral_solution_l587_58762


namespace NUMINAMATH_GPT_original_quadrilateral_area_l587_58768

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_original_quadrilateral_area_l587_58768


namespace NUMINAMATH_GPT_designed_height_correct_l587_58709
noncomputable def designed_height_of_lower_part (H : ℝ) (L : ℝ) : Prop :=
  H = 2 ∧ (H - L) / L = L / H

theorem designed_height_correct : ∃ L, designed_height_of_lower_part 2 L ∧ L = Real.sqrt 5 - 1 :=
by
  sorry

end NUMINAMATH_GPT_designed_height_correct_l587_58709


namespace NUMINAMATH_GPT_sticker_height_enlarged_l587_58716

theorem sticker_height_enlarged (orig_width orig_height new_width : ℝ)
    (h1 : orig_width = 3) (h2 : orig_height = 2) (h3 : new_width = 12) :
    new_width / orig_width * orig_height = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_sticker_height_enlarged_l587_58716


namespace NUMINAMATH_GPT_fixed_point_of_transformed_exponential_l587_58747

variable (a : ℝ)
variable (h_pos : 0 < a)
variable (h_ne_one : a ≠ 1)

theorem fixed_point_of_transformed_exponential :
    (∃ x y : ℝ, (y = a^(x-2) + 2) ∧ (y = x) ∧ (x = 2) ∧ (y = 3)) :=
by {
    sorry -- Proof goes here
}

end NUMINAMATH_GPT_fixed_point_of_transformed_exponential_l587_58747


namespace NUMINAMATH_GPT_max_value_of_y_l587_58706

noncomputable def maxY (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 10 * x + 60 * y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 60 * y) : 
  y ≤ 30 + 5 * Real.sqrt 37 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l587_58706
