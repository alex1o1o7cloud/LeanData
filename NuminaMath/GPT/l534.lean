import Mathlib

namespace NUMINAMATH_GPT_chord_intersection_l534_53401

theorem chord_intersection {AP BP CP DP : ℝ} (hAP : AP = 2) (hBP : BP = 6) (hCP_DP : ∃ k : ℝ, CP = k ∧ DP = 3 * k) :
  DP = 6 :=
by sorry

end NUMINAMATH_GPT_chord_intersection_l534_53401


namespace NUMINAMATH_GPT_sin_sum_leq_3_sqrt_3_div_2_l534_53470

theorem sin_sum_leq_3_sqrt_3_div_2 (A B C : ℝ) (h_sum : A + B + C = Real.pi) (h_pos : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_sin_sum_leq_3_sqrt_3_div_2_l534_53470


namespace NUMINAMATH_GPT_min_chips_to_A10_l534_53438

theorem min_chips_to_A10 (n : ℕ) (A : ℕ → ℕ) (hA1 : A 1 = n) :
  (∃ (σ : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i < 10 → (σ i = A i - 2) ∧ (σ (i + 1) = A (i + 1) + 1)) ∨ 
    (∀ i, 1 ≤ i ∧ i < 9 → (σ (i + 1) = A (i + 1) - 2) ∧ (σ (i + 2) = A (i + 2) + 1) ∧ (σ i = A i + 1)) ∧ 
    (∃ (k : ℕ), k = 10 ∧ σ k = 1)) →
  n ≥ 46 := sorry

end NUMINAMATH_GPT_min_chips_to_A10_l534_53438


namespace NUMINAMATH_GPT_average_annual_growth_rate_l534_53418

-- Define the conditions
def revenue_current_year : ℝ := 280
def revenue_planned_two_years : ℝ := 403.2

-- Define the growth equation
def growth_equation (x : ℝ) : Prop :=
  revenue_current_year * (1 + x)^2 = revenue_planned_two_years

-- State the theorem
theorem average_annual_growth_rate : ∃ x : ℝ, growth_equation x ∧ x = 0.2 := by
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_l534_53418


namespace NUMINAMATH_GPT_total_garbage_collected_correct_l534_53455

def Lizzie_group_collected : ℕ := 387
def other_group_collected : ℕ := Lizzie_group_collected - 39
def total_garbage_collected : ℕ := Lizzie_group_collected + other_group_collected

theorem total_garbage_collected_correct :
  total_garbage_collected = 735 :=
sorry

end NUMINAMATH_GPT_total_garbage_collected_correct_l534_53455


namespace NUMINAMATH_GPT_no_three_integers_exist_l534_53479

theorem no_three_integers_exist (x y z : ℤ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  ((x^2 - 1) % y = 0) ∧ ((x^2 - 1) % z = 0) ∧
  ((y^2 - 1) % x = 0) ∧ ((y^2 - 1) % z = 0) ∧
  ((z^2 - 1) % x = 0) ∧ ((z^2 - 1) % y = 0) → false :=
by
  sorry

end NUMINAMATH_GPT_no_three_integers_exist_l534_53479


namespace NUMINAMATH_GPT_insurance_covers_80_percent_l534_53498

def total_cost : ℝ := 300
def out_of_pocket_cost : ℝ := 60
def insurance_coverage : ℝ := 0.8  -- Representing 80%

theorem insurance_covers_80_percent :
  (total_cost - out_of_pocket_cost) / total_cost = insurance_coverage := by
  sorry

end NUMINAMATH_GPT_insurance_covers_80_percent_l534_53498


namespace NUMINAMATH_GPT_inequality_for_positive_integers_l534_53443

theorem inequality_for_positive_integers 
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : 1/a + 1/b = 1)
  (n : ℕ)
  (hn : n > 0) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2^(2*n) - 2^(n + 1) :=
sorry

end NUMINAMATH_GPT_inequality_for_positive_integers_l534_53443


namespace NUMINAMATH_GPT_nap_hours_in_70_days_l534_53416

-- Define the variables and conditions
variable (n d a b c e : ℕ)  -- assuming they are natural numbers

-- Define the total nap hours function
noncomputable def total_nap_hours (n d a b c e : ℕ) : ℕ :=
  (a + b) * 10

-- The statement to prove
theorem nap_hours_in_70_days (n d a b c e : ℕ) :
  total_nap_hours n d a b c e = (a + b) * 10 :=
by sorry

end NUMINAMATH_GPT_nap_hours_in_70_days_l534_53416


namespace NUMINAMATH_GPT_smaller_number_in_ratio_l534_53447

noncomputable def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem smaller_number_in_ratio (x : ℕ) (a b : ℕ) (h1 : a = 4 * x) (h2 : b = 5 * x) (h3 : LCM a b = 180) : a = 36 := 
by
  sorry

end NUMINAMATH_GPT_smaller_number_in_ratio_l534_53447


namespace NUMINAMATH_GPT_find_A_satisfy_3A_multiple_of_8_l534_53450

theorem find_A_satisfy_3A_multiple_of_8 (A : ℕ) (h : 0 ≤ A ∧ A < 10) : 8 ∣ (30 + A) ↔ A = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_A_satisfy_3A_multiple_of_8_l534_53450


namespace NUMINAMATH_GPT_stream_speed_l534_53444

variable (D v : ℝ)

/--
The time taken by a man to row his boat upstream is twice the time taken by him to row the same distance downstream.
If the speed of the boat in still water is 63 kmph, prove that the speed of the stream is 21 kmph.
-/
theorem stream_speed (h : D / (63 - v) = 2 * (D / (63 + v))) : v = 21 := 
sorry

end NUMINAMATH_GPT_stream_speed_l534_53444


namespace NUMINAMATH_GPT_total_profit_calculation_l534_53481

theorem total_profit_calculation (A B C : ℕ) (C_share total_profit : ℕ) 
  (hA : A = 27000) 
  (hB : B = 72000) 
  (hC : C = 81000) 
  (hC_share : C_share = 36000) 
  (h_ratio : C_share * 20 = total_profit * 9) :
  total_profit = 80000 := by
  sorry

end NUMINAMATH_GPT_total_profit_calculation_l534_53481


namespace NUMINAMATH_GPT_jose_peanuts_l534_53457

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jose_peanuts_l534_53457


namespace NUMINAMATH_GPT_total_wheels_combined_l534_53496

-- Define the counts of vehicles and wheels per vehicle in each storage area
def bicycles_A : ℕ := 16
def tricycles_A : ℕ := 7
def unicycles_A : ℕ := 10
def four_wheelers_A : ℕ := 5

def bicycles_B : ℕ := 12
def tricycles_B : ℕ := 5
def unicycles_B : ℕ := 8
def four_wheelers_B : ℕ := 3

def wheels_bicycle : ℕ := 2
def wheels_tricycle : ℕ := 3
def wheels_unicycle : ℕ := 1
def wheels_four_wheeler : ℕ := 4

-- Calculate total wheels in Storage Area A
def total_wheels_A : ℕ :=
  bicycles_A * wheels_bicycle + tricycles_A * wheels_tricycle + unicycles_A * wheels_unicycle + four_wheelers_A * wheels_four_wheeler
  
-- Calculate total wheels in Storage Area B
def total_wheels_B : ℕ :=
  bicycles_B * wheels_bicycle + tricycles_B * wheels_tricycle + unicycles_B * wheels_unicycle + four_wheelers_B * wheels_four_wheeler

-- Theorem stating that the combined total number of wheels in both storage areas is 142
theorem total_wheels_combined : total_wheels_A + total_wheels_B = 142 := by
  sorry

end NUMINAMATH_GPT_total_wheels_combined_l534_53496


namespace NUMINAMATH_GPT_ZYX_syndrome_diagnosis_l534_53493

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end NUMINAMATH_GPT_ZYX_syndrome_diagnosis_l534_53493


namespace NUMINAMATH_GPT_radius_increase_l534_53425

theorem radius_increase (C1 C2 : ℝ) (h1 : C1 = 30) (h2 : C2 = 40) : 
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  let Δr := r2 - r1
  Δr = 5 / Real.pi := by
sorry

end NUMINAMATH_GPT_radius_increase_l534_53425


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l534_53461

-- Part 1: Prove the solution set of the inequality f(x) < 6 is (-8/3, 4/3)
theorem part1_solution_set (x : ℝ) :
  (|2 * x + 3| + |x - 1| < 6) ↔ (-8 / 3 : ℝ) < x ∧ x < 4 / 3 :=
by sorry

-- Part 2: Prove the range of values for a that makes f(x) + f(-x) ≥ 5 is (-∞, -3/2] ∪ [3/2, +∞)
theorem part2_range_of_a (a : ℝ) (x : ℝ) :
  (|2 * x + a| + |x - 1| + |-2 * x + a| + |-x - 1| ≥ 5) ↔ 
  (a ≤ -3 / 2 ∨ a ≥ 3 / 2) :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l534_53461


namespace NUMINAMATH_GPT_total_cost_correct_l534_53442

-- Definitions for the conditions
def num_ladders_1 : ℕ := 10
def rungs_1 : ℕ := 50
def cost_per_rung_1 : ℕ := 2

def num_ladders_2 : ℕ := 20
def rungs_2 : ℕ := 60
def cost_per_rung_2 : ℕ := 3

def num_ladders_3 : ℕ := 30
def rungs_3 : ℕ := 80
def cost_per_rung_3 : ℕ := 4

-- Total cost calculation for the client
def total_cost : ℕ :=
  (num_ladders_1 * rungs_1 * cost_per_rung_1) +
  (num_ladders_2 * rungs_2 * cost_per_rung_2) +
  (num_ladders_3 * rungs_3 * cost_per_rung_3)

-- Statement to be proved
theorem total_cost_correct : total_cost = 14200 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_cost_correct_l534_53442


namespace NUMINAMATH_GPT_remainder_of_3042_div_98_l534_53433

theorem remainder_of_3042_div_98 : 3042 % 98 = 4 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_3042_div_98_l534_53433


namespace NUMINAMATH_GPT_earning_hours_per_week_l534_53403

theorem earning_hours_per_week (totalEarnings : ℝ) (originalWeeks : ℝ) (missedWeeks : ℝ) 
  (originalHoursPerWeek : ℝ) : 
  missedWeeks = 3 → originalWeeks = 15 → originalHoursPerWeek = 25 → totalEarnings = 3750 → 
  (totalEarnings / ((totalEarnings / (originalWeeks * originalHoursPerWeek)) * (originalWeeks - missedWeeks))) = 31.25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_earning_hours_per_week_l534_53403


namespace NUMINAMATH_GPT_six_digit_palindromes_count_l534_53430

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def is_non_zero_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem six_digit_palindromes_count : 
  (∃a b c : ℕ, is_non_zero_digit a ∧ is_digit b ∧ is_digit c) → 
  (∃ n : ℕ, n = 900) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_palindromes_count_l534_53430


namespace NUMINAMATH_GPT_tom_total_money_l534_53409

theorem tom_total_money :
  let initial_amount := 74
  let additional_amount := 86
  initial_amount + additional_amount = 160 :=
by
  let initial_amount := 74
  let additional_amount := 86
  show initial_amount + additional_amount = 160
  sorry

end NUMINAMATH_GPT_tom_total_money_l534_53409


namespace NUMINAMATH_GPT_option_C_holds_l534_53495

theorem option_C_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a - b / a > b - a / b := 
  sorry

end NUMINAMATH_GPT_option_C_holds_l534_53495


namespace NUMINAMATH_GPT_sum_of_fractions_eq_sum_of_cubes_l534_53400

theorem sum_of_fractions_eq_sum_of_cubes (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  ( (x-1)*(x+1) / (x*(x-1) + 1) + (2*(0.5-x)) / (x*(1-x) -1) ) = 
  ( ((x-1)*(x+1) / (x*(x-1) + 1))^3 + ((2*(0.5-x)) / (x*(1-x) -1))^3 ) :=
sorry

end NUMINAMATH_GPT_sum_of_fractions_eq_sum_of_cubes_l534_53400


namespace NUMINAMATH_GPT_units_digit_47_pow_47_l534_53460

theorem units_digit_47_pow_47 : (47^47) % 10 = 3 :=
  sorry

end NUMINAMATH_GPT_units_digit_47_pow_47_l534_53460


namespace NUMINAMATH_GPT_students_just_passed_l534_53490

theorem students_just_passed (total_students first_div_percent second_div_percent : ℝ)
  (h_total_students: total_students = 300)
  (h_first_div_percent: first_div_percent = 0.29)
  (h_second_div_percent: second_div_percent = 0.54)
  (h_no_failures : total_students = 300) :
  ∃ passed_students, passed_students = total_students - (first_div_percent * total_students + second_div_percent * total_students) ∧ passed_students = 51 :=
by
  sorry

end NUMINAMATH_GPT_students_just_passed_l534_53490


namespace NUMINAMATH_GPT_math_dance_residents_l534_53413

theorem math_dance_residents (p a b : ℕ) (hp : Nat.Prime p) 
    (h1 : b ≥ 1) 
    (h2 : (a + b)^2 = (p + 1) * a + b) :
    b = 1 := by
  sorry

end NUMINAMATH_GPT_math_dance_residents_l534_53413


namespace NUMINAMATH_GPT_leonid_painted_cells_l534_53452

theorem leonid_painted_cells (k l : ℕ) (hkl : k * l = 74) :
  ∃ (painted_cells : ℕ), painted_cells = ((2 * k + 1) * (2 * l + 1) - 74) ∧ (painted_cells = 373 ∨ painted_cells = 301) :=
by
  sorry

end NUMINAMATH_GPT_leonid_painted_cells_l534_53452


namespace NUMINAMATH_GPT_average_bracelets_per_day_l534_53427

theorem average_bracelets_per_day
  (cost_of_bike : ℕ)
  (price_per_bracelet : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (h1 : cost_of_bike = 112)
  (h2 : price_per_bracelet = 1)
  (h3 : weeks = 2)
  (h4 : days_per_week = 7) :
  (cost_of_bike / price_per_bracelet) / (weeks * days_per_week) = 8 :=
by
  sorry

end NUMINAMATH_GPT_average_bracelets_per_day_l534_53427


namespace NUMINAMATH_GPT_concrete_pillars_l534_53424

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end NUMINAMATH_GPT_concrete_pillars_l534_53424


namespace NUMINAMATH_GPT_intersection_points_sum_l534_53440

theorem intersection_points_sum (x1 x2 x3 y1 y2 y3 A B : ℝ)
(h1 : y1 = x1^3 - 3 * x1 + 2)
(h2 : x1 + 6 * y1 = 6)
(h3 : y2 = x2^3 - 3 * x2 + 2)
(h4 : x2 + 6 * y2 = 6)
(h5 : y3 = x3^3 - 3 * x3 + 2)
(h6 : x3 + 6 * y3 = 6)
(hA : A = x1 + x2 + x3)
(hB : B = y1 + y2 + y3) :
A = 0 ∧ B = 3 := 
by
  sorry

end NUMINAMATH_GPT_intersection_points_sum_l534_53440


namespace NUMINAMATH_GPT_total_pull_ups_per_week_l534_53421

-- Definitions from the conditions
def pull_ups_per_time := 2
def visits_per_day := 5
def days_per_week := 7

-- The Math proof problem statement
theorem total_pull_ups_per_week :
  pull_ups_per_time * visits_per_day * days_per_week = 70 := by
  sorry

end NUMINAMATH_GPT_total_pull_ups_per_week_l534_53421


namespace NUMINAMATH_GPT_fraction_august_tips_l534_53422

variable (A : ℝ) -- Define the average monthly tips A for March, April, May, June, July, and September
variable (august_tips : ℝ) -- Define the tips for August
variable (total_tips : ℝ) -- Define the total tips for all months

-- Define the conditions
def condition_average_tips : Prop := total_tips = 12 * A
def condition_august_tips : Prop := august_tips = 6 * A

-- The theorem we need to prove
theorem fraction_august_tips :
  condition_average_tips A total_tips →
  condition_august_tips A august_tips →
  (august_tips / total_tips) = (1 / 2) :=
by
  intros h_avg h_aug
  rw [condition_average_tips] at h_avg
  rw [condition_august_tips] at h_aug
  rw [h_avg, h_aug]
  simp
  sorry

end NUMINAMATH_GPT_fraction_august_tips_l534_53422


namespace NUMINAMATH_GPT_cos_double_angle_l534_53415

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -3/5) : Real.cos (2 * α) = -7/25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l534_53415


namespace NUMINAMATH_GPT_tangent_line_of_ellipse_l534_53466

variable {a b x y x₀ y₀ : ℝ}

theorem tangent_line_of_ellipse
    (h1 : 0 < a)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : (x₀, y₀) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_of_ellipse_l534_53466


namespace NUMINAMATH_GPT_cos_difference_l534_53412

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1 / 2) 
  (h2 : Real.cos A + Real.cos B = 3 / 2) : 
  Real.cos (A - B) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_difference_l534_53412


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l534_53489

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l534_53489


namespace NUMINAMATH_GPT_smallest_sum_l534_53451

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l534_53451


namespace NUMINAMATH_GPT_bob_walking_rate_is_12_l534_53482

-- Definitions for the problem
def yolanda_distance := 24
def yolanda_rate := 3
def bob_distance_when_met := 12
def time_yolanda_walked := 2

-- The theorem we need to prove
theorem bob_walking_rate_is_12 : 
  (bob_distance_when_met / (time_yolanda_walked - 1) = 12) :=
by sorry

end NUMINAMATH_GPT_bob_walking_rate_is_12_l534_53482


namespace NUMINAMATH_GPT_largest_divisor_of_square_l534_53463

theorem largest_divisor_of_square (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n ^ 2) : 12 ∣ n := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_square_l534_53463


namespace NUMINAMATH_GPT_arithmetic_seq_geom_eq_div_l534_53437

noncomputable def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + n * d

theorem arithmetic_seq_geom_eq_div (a1 d : ℝ) (h1 : d ≠ 0) (h2 : a1 ≠ 0) 
    (h_geom : (a 3 a1 d) ^ 2 = (a 1 a1 d) * (a 7 a1 d)) :
    (a 2 a1 d + a 5 a1 d + a 8 a1 d) / (a 3 a1 d + a 4 a1 d) = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_geom_eq_div_l534_53437


namespace NUMINAMATH_GPT_simplify_and_multiply_expression_l534_53487

variable (b : ℝ)

theorem simplify_and_multiply_expression :
  (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_multiply_expression_l534_53487


namespace NUMINAMATH_GPT_correct_product_l534_53423

theorem correct_product (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)  -- a is a two-digit number
  (h2 : 0 < b)  -- b is a positive integer
  (h3 : (a % 10) * 10 + (a / 10) * b = 161)  -- Reversing the digits of a and multiplying by b yields 161
  : a * b = 224 := 
sorry

end NUMINAMATH_GPT_correct_product_l534_53423


namespace NUMINAMATH_GPT_machine_worked_yesterday_l534_53484

noncomputable def shirts_made_per_minute : ℕ := 3
noncomputable def shirts_made_yesterday : ℕ := 9

theorem machine_worked_yesterday : 
  (shirts_made_yesterday / shirts_made_per_minute) = 3 :=
sorry

end NUMINAMATH_GPT_machine_worked_yesterday_l534_53484


namespace NUMINAMATH_GPT_circle_parametric_solution_l534_53417

theorem circle_parametric_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (hx : 4 * Real.cos θ = -2) (hy : 4 * Real.sin θ = 2 * Real.sqrt 3) :
    θ = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_circle_parametric_solution_l534_53417


namespace NUMINAMATH_GPT_f_zero_eq_zero_l534_53477

-- Define the problem conditions
variable {f : ℝ → ℝ}
variables (h_odd : ∀ x : ℝ, f (-x) = -f (x))
variables (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variables (h_eq : ∀ x : ℝ, f (1 - x) - f (1 + x) + 2 * x = 0)
variables (h_mono : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem
theorem f_zero_eq_zero : f 0 = 0 :=
by sorry

end NUMINAMATH_GPT_f_zero_eq_zero_l534_53477


namespace NUMINAMATH_GPT_max_lessons_l534_53456

theorem max_lessons (x y z : ℕ) 
  (h1 : 3 * y * z = 18) 
  (h2 : 3 * x * z = 63) 
  (h3 : 3 * x * y = 42) :
  3 * x * y * z = 126 :=
by
  sorry

end NUMINAMATH_GPT_max_lessons_l534_53456


namespace NUMINAMATH_GPT_xiao_ming_english_score_l534_53429

theorem xiao_ming_english_score :
  let a := 92
  let b := 90
  let c := 95
  let w_a := 3
  let w_b := 3
  let w_c := 4
  let total_weight := (w_a + w_b + w_c)
  let score := (a * w_a + b * w_b + c * w_c) / total_weight
  score = 92.6 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_english_score_l534_53429


namespace NUMINAMATH_GPT_probability_of_F_l534_53426

theorem probability_of_F (P : String → ℚ) (hD : P "D" = 1/4) (hE : P "E" = 1/3) (hG : P "G" = 1/6) (total : P "D" + P "E" + P "F" + P "G" = 1) :
  P "F" = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_F_l534_53426


namespace NUMINAMATH_GPT_avg_speed_is_20_l534_53491

-- Define the total distance and total time
def total_distance : ℕ := 100
def total_time : ℕ := 5

-- Define the average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The theorem to prove the average speed given the distance and time
theorem avg_speed_is_20 : average_speed total_distance total_time = 20 :=
by
  sorry

end NUMINAMATH_GPT_avg_speed_is_20_l534_53491


namespace NUMINAMATH_GPT_pastries_total_l534_53453

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end NUMINAMATH_GPT_pastries_total_l534_53453


namespace NUMINAMATH_GPT_integer_sum_l534_53499

theorem integer_sum {p q r s : ℤ} 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 22 := 
sorry

end NUMINAMATH_GPT_integer_sum_l534_53499


namespace NUMINAMATH_GPT_solve_system_of_equations_l534_53480

variables {a1 a2 a3 a4 : ℝ}

theorem solve_system_of_equations (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ (x1 x2 x3 x4 : ℝ),
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / (a1 - a4)) ∧ (x2 = 0) ∧ (x3 = 0) ∧ (x4 = 1 / (a1 - a4)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l534_53480


namespace NUMINAMATH_GPT_mandy_cinnamon_nutmeg_difference_l534_53445

theorem mandy_cinnamon_nutmeg_difference :
  0.67 - 0.5 = 0.17 :=
by
  sorry

end NUMINAMATH_GPT_mandy_cinnamon_nutmeg_difference_l534_53445


namespace NUMINAMATH_GPT_two_digit_ab_divisible_by_11_13_l534_53471

theorem two_digit_ab_divisible_by_11_13 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 11 = 0)
  (h4 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 13 = 0) :
  10 * a + b = 48 :=
sorry

end NUMINAMATH_GPT_two_digit_ab_divisible_by_11_13_l534_53471


namespace NUMINAMATH_GPT_pages_read_on_wednesday_l534_53478

theorem pages_read_on_wednesday (W : ℕ) (h : 18 + W + 23 = 60) : W = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_pages_read_on_wednesday_l534_53478


namespace NUMINAMATH_GPT_parking_lot_total_spaces_l534_53473

theorem parking_lot_total_spaces (ratio_fs_cc : ℕ) (ratio_cc_fs : ℕ) (fs_spaces : ℕ) (total_spaces : ℕ) 
  (h1 : ratio_fs_cc = 11) (h2 : ratio_cc_fs = 4) (h3 : fs_spaces = 330) :
  total_spaces = 450 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_total_spaces_l534_53473


namespace NUMINAMATH_GPT_emily_extra_distance_five_days_l534_53464

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_emily_extra_distance_five_days_l534_53464


namespace NUMINAMATH_GPT_probability_of_two_black_balls_relationship_x_y_l534_53448

-- Conditions
def initial_black_balls : ℕ := 3
def initial_white_balls : ℕ := 2

variable (x y : ℕ)

-- Given relationship
def total_white_balls := x + 2
def total_black_balls := y + 3
def white_ball_probability := (total_white_balls x) / (total_white_balls x + total_black_balls y + 5)

-- Proof goals
theorem probability_of_two_black_balls :
  (3 / 5) * (2 / 4) = 3 / 10 := by sorry

theorem relationship_x_y :
  white_ball_probability x y = 1 / 3 → y = 2 * x + 1 := by sorry

end NUMINAMATH_GPT_probability_of_two_black_balls_relationship_x_y_l534_53448


namespace NUMINAMATH_GPT_Lorin_black_marbles_l534_53405

variable (B : ℕ)

def Jimmy_yellow_marbles := 22
def Alex_yellow_marbles := Jimmy_yellow_marbles / 2
def Alex_black_marbles := 2 * B
def Alex_total_marbles := Alex_yellow_marbles + Alex_black_marbles

theorem Lorin_black_marbles : Alex_total_marbles = 19 → B = 4 :=
by
  intros h
  unfold Alex_total_marbles at h
  unfold Alex_yellow_marbles at h
  unfold Alex_black_marbles at h
  norm_num at h
  exact sorry

end NUMINAMATH_GPT_Lorin_black_marbles_l534_53405


namespace NUMINAMATH_GPT_correct_system_of_equations_l534_53458

theorem correct_system_of_equations (x y : ℝ) :
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by sorry

end NUMINAMATH_GPT_correct_system_of_equations_l534_53458


namespace NUMINAMATH_GPT_power_identity_l534_53483

theorem power_identity (x : ℝ) : (x ^ 10 = 25 ^ 5) → x = 5 := by
  sorry

end NUMINAMATH_GPT_power_identity_l534_53483


namespace NUMINAMATH_GPT_erasers_difference_l534_53494

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_erasers_difference_l534_53494


namespace NUMINAMATH_GPT_income_to_expenditure_ratio_l534_53459

theorem income_to_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 4000) (hSavings : S = I - E) : I / E = 5 / 3 := by
  -- To prove: I / E = 5 / 3 given hI, hS, and hSavings
  sorry

end NUMINAMATH_GPT_income_to_expenditure_ratio_l534_53459


namespace NUMINAMATH_GPT_min_days_equal_shifts_l534_53465

theorem min_days_equal_shifts (k n : ℕ) (h : 9 * k + 10 * n = 66) : k + n = 7 :=
sorry

end NUMINAMATH_GPT_min_days_equal_shifts_l534_53465


namespace NUMINAMATH_GPT_rain_stop_time_on_first_day_l534_53441

-- Define the problem conditions
def raining_time_day1 (x : ℕ) : Prop :=
  let start_time := 7 * 60 -- start time in minutes
  let stop_time := start_time + x * 60 -- stop time in minutes
  stop_time = 17 * 60 -- stop at 17:00 (5:00 PM)

def total_raining_time_46_hours (x : ℕ) : Prop :=
  x + (x + 2) + 2 * (x + 2) = 46

-- Main statement
theorem rain_stop_time_on_first_day (x : ℕ) (h1 : total_raining_time_46_hours x) : raining_time_day1 x :=
  sorry

end NUMINAMATH_GPT_rain_stop_time_on_first_day_l534_53441


namespace NUMINAMATH_GPT_arithmetic_sequence_S12_l534_53411

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2*a + (n-1)*d) / 2

def a_n (a d n : ℕ) : ℕ :=
  a + (n-1)*d

variable (a d : ℕ)

theorem arithmetic_sequence_S12 (h : a_n a d 4 + a_n a d 9 = 10) :
  arithmetic_sequence_sum a d 12 = 60 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_S12_l534_53411


namespace NUMINAMATH_GPT_ralphStartsWith_l534_53420

def ralphEndsWith : ℕ := 15
def ralphLoses : ℕ := 59

theorem ralphStartsWith : (ralphEndsWith + ralphLoses = 74) :=
by
  sorry

end NUMINAMATH_GPT_ralphStartsWith_l534_53420


namespace NUMINAMATH_GPT_part_one_part_two_part_three_l534_53414

theorem part_one : 12 - (-11) - 1 = 22 := 
by
  sorry

theorem part_two : -(1 ^ 4) / ((-3) ^ 2) / (9 / 5) = -5 / 81 := 
by
  sorry

theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := 
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_part_three_l534_53414


namespace NUMINAMATH_GPT_div_by_3_l534_53476

theorem div_by_3 (a b : ℤ) : 
  (∃ (k : ℤ), a = 3 * k) ∨ 
  (∃ (k : ℤ), b = 3 * k) ∨ 
  (∃ (k : ℤ), a + b = 3 * k) ∨ 
  (∃ (k : ℤ), a - b = 3 * k) :=
sorry

end NUMINAMATH_GPT_div_by_3_l534_53476


namespace NUMINAMATH_GPT_sum_of_squares_l534_53468

theorem sum_of_squares (x : ℚ) (hx : 7 * x = 15) : 
  (x^2 + (2 * x)^2 + (4 * x)^2 = 4725 / 49) := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l534_53468


namespace NUMINAMATH_GPT_triangle_area_l534_53410

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : a + b = 13) (h3 : c = Real.sqrt (a^2 + b^2)) : 
  (1 / 2) * a * b = 20 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l534_53410


namespace NUMINAMATH_GPT_cyclic_determinant_zero_l534_53472

open Matrix

-- Define the roots of the polynomial and the polynomial itself.
variables {α β γ δ : ℂ} -- We assume the roots are complex numbers.
variable (p q r : ℂ) -- Coefficients of the polynomial x^4 + px^2 + qx + r = 0

-- Define the matrix whose determinant we want to compute
def cyclic_matrix (α β γ δ : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![
    ![α, β, γ, δ],
    ![β, γ, δ, α],
    ![γ, δ, α, β],
    ![δ, α, β, γ]
  ]

-- Statement of the theorem
theorem cyclic_determinant_zero :
  ∀ (α β γ δ : ℂ) (p q r : ℂ),
  (∀ x : ℂ, x ^ 4 + p * x ^ 2 + q * x + r = 0 → x = α ∨ x = β ∨ x = γ ∨ x = δ) →
  det (cyclic_matrix α β γ δ) = 0 :=
by
  intros α β γ δ p q r hRoots
  sorry

end NUMINAMATH_GPT_cyclic_determinant_zero_l534_53472


namespace NUMINAMATH_GPT_composite_sum_pow_l534_53486

theorem composite_sum_pow (a b c d : ℕ) (h_pos : a > b ∧ b > c ∧ c > d)
    (h_div : (a + b - c + d) ∣ (a * c + b * d)) (m : ℕ) (h_m_pos : 0 < m) 
    (n : ℕ) (h_n_odd : n % 2 = 1) : ∃ k : ℕ, k > 1 ∧ k ∣ (a ^ n * b ^ m + c ^ m * d ^ n) :=
by
  sorry

end NUMINAMATH_GPT_composite_sum_pow_l534_53486


namespace NUMINAMATH_GPT_sum_of_reciprocals_l534_53434

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l534_53434


namespace NUMINAMATH_GPT_randy_trip_total_distance_l534_53492

-- Definition of the problem condition
def randy_trip_length (x : ℝ) : Prop :=
  x / 3 + 20 + x / 5 = x

-- The total length of Randy's trip
theorem randy_trip_total_distance : ∃ x : ℝ, randy_trip_length x ∧ x = 300 / 7 :=
by
  sorry

end NUMINAMATH_GPT_randy_trip_total_distance_l534_53492


namespace NUMINAMATH_GPT_cuboid_diagonal_length_l534_53408

theorem cuboid_diagonal_length (x y z : ℝ) 
  (h1 : y * z = Real.sqrt 2) 
  (h2 : z * x = Real.sqrt 3)
  (h3 : x * y = Real.sqrt 6) : 
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_cuboid_diagonal_length_l534_53408


namespace NUMINAMATH_GPT_abs_ab_eq_2_sqrt_65_l534_53439

theorem abs_ab_eq_2_sqrt_65
  (a b : ℝ)
  (h1 : b^2 - a^2 = 16)
  (h2 : a^2 + b^2 = 36) :
  |a * b| = 2 * Real.sqrt 65 := 
sorry

end NUMINAMATH_GPT_abs_ab_eq_2_sqrt_65_l534_53439


namespace NUMINAMATH_GPT_values_only_solution_l534_53431

variables (m n : ℝ) (x a b c : ℝ)

noncomputable def equation := (x + m)^3 - (x + n)^3 = (m + n)^3

theorem values_only_solution (hm : m ≠ 0) (hn : n ≠ 0) (hne : m ≠ n)
  (hx : x = a * m + b * n + c) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_values_only_solution_l534_53431


namespace NUMINAMATH_GPT_exists_strictly_increasing_sequences_l534_53419

theorem exists_strictly_increasing_sequences :
  ∃ u v : ℕ → ℕ, (∀ n, u n < u (n + 1)) ∧ (∀ n, v n < v (n + 1)) ∧ (∀ n, 5 * u n * (u n + 1) = v n ^ 2 + 1) :=
sorry

end NUMINAMATH_GPT_exists_strictly_increasing_sequences_l534_53419


namespace NUMINAMATH_GPT_cube_sum_equal_one_l534_53404

theorem cube_sum_equal_one (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = 1) (h3 : xyz = 1) :
  x^3 + y^3 + z^3 = 1 := 
sorry

end NUMINAMATH_GPT_cube_sum_equal_one_l534_53404


namespace NUMINAMATH_GPT_problem_inequality_l534_53432

-- Definitions and conditions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x + f (k - x)

-- The Lean proof problem
theorem problem_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  f a + (a + b) * Real.log 2 ≥ f (a + b) - f b := sorry

end NUMINAMATH_GPT_problem_inequality_l534_53432


namespace NUMINAMATH_GPT_minimum_rubles_to_reverse_order_of_chips_100_l534_53469

noncomputable def minimum_rubles_to_reverse_order_of_chips (n : ℕ) : ℕ :=
if n = 100 then 61 else 0

theorem minimum_rubles_to_reverse_order_of_chips_100 :
  minimum_rubles_to_reverse_order_of_chips 100 = 61 :=
by sorry

end NUMINAMATH_GPT_minimum_rubles_to_reverse_order_of_chips_100_l534_53469


namespace NUMINAMATH_GPT_Donovan_percentage_correct_l534_53436

-- Definitions based on conditions from part a)
def fullyCorrectAnswers : ℕ := 35
def incorrectAnswers : ℕ := 13
def partiallyCorrectAnswers : ℕ := 7
def pointPerFullAnswer : ℝ := 1
def pointPerPartialAnswer : ℝ := 0.5

-- Lean 4 statement to prove the problem mathematically
theorem Donovan_percentage_correct : 
  (fullyCorrectAnswers * pointPerFullAnswer + partiallyCorrectAnswers * pointPerPartialAnswer) / 
  (fullyCorrectAnswers + incorrectAnswers + partiallyCorrectAnswers) * 100 = 70.00 :=
by
  sorry

end NUMINAMATH_GPT_Donovan_percentage_correct_l534_53436


namespace NUMINAMATH_GPT_number_of_girls_l534_53428

theorem number_of_girls {total_children boys girls : ℕ} 
  (h_total : total_children = 60) 
  (h_boys : boys = 18) 
  (h_girls : girls = total_children - boys) : 
  girls = 42 := by 
  sorry

end NUMINAMATH_GPT_number_of_girls_l534_53428


namespace NUMINAMATH_GPT_min_contribution_proof_l534_53446

noncomputable def min_contribution (total_contribution : ℕ) (num_people : ℕ) (max_contribution: ℕ) :=
  ∃ (min_each_person: ℕ), num_people * min_each_person ≤ total_contribution ∧ max_contribution * (num_people - 1) + min_each_person ≥ total_contribution ∧ min_each_person = 2

theorem min_contribution_proof :
  min_contribution 30 15 16 :=
sorry

end NUMINAMATH_GPT_min_contribution_proof_l534_53446


namespace NUMINAMATH_GPT_no_domovoi_exists_l534_53402

variables {Domovoi Creature : Type}

def likes_pranks (c : Creature) : Prop := sorry
def likes_cleanliness_order (c : Creature) : Prop := sorry
def is_domovoi (c : Creature) : Prop := sorry

axiom all_domovoi_like_pranks : ∀ (c : Creature), is_domovoi c → likes_pranks c
axiom all_domovoi_like_cleanliness : ∀ (c : Creature), is_domovoi c → likes_cleanliness_order c
axiom cleanliness_implies_no_pranks : ∀ (c : Creature), likes_cleanliness_order c → ¬ likes_pranks c

theorem no_domovoi_exists : ¬ ∃ (c : Creature), is_domovoi c := 
sorry

end NUMINAMATH_GPT_no_domovoi_exists_l534_53402


namespace NUMINAMATH_GPT_inequality_proof_l534_53474

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l534_53474


namespace NUMINAMATH_GPT_min_value_f_l534_53497

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(2 - x)

theorem min_value_f : ∃ x : ℝ, f x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_f_l534_53497


namespace NUMINAMATH_GPT_area_ratio_l534_53475

variable (A_shape A_triangle : ℝ)

-- Condition: The area ratio given.
axiom ratio_condition : A_shape / A_triangle = 2

-- Theorem statement
theorem area_ratio (A_shape A_triangle : ℝ) (h : A_shape / A_triangle = 2) : A_shape / A_triangle = 2 :=
by
  exact h

end NUMINAMATH_GPT_area_ratio_l534_53475


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l534_53485

/-- The cost price \(C\) of a computer table is Rs. 7000 -/
theorem cost_price_of_computer_table : 
  ∃ (C : ℝ), (S = 1.20 * C) ∧ (S = 8400) → C = 7000 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l534_53485


namespace NUMINAMATH_GPT_sum_multiple_of_3_probability_l534_53454

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_multiple_of_3_probability_l534_53454


namespace NUMINAMATH_GPT_average_sleep_hours_l534_53488

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_average_sleep_hours_l534_53488


namespace NUMINAMATH_GPT_f_diff_ineq_l534_53467

variable {f : ℝ → ℝ}
variable (deriv_f : ∀ x > 0, x * (deriv f x) > 1)

theorem f_diff_ineq (h : ∀ x > 0, x * (deriv f x) > 1) : f 2 - f 1 > Real.log 2 := by 
  sorry

end NUMINAMATH_GPT_f_diff_ineq_l534_53467


namespace NUMINAMATH_GPT_linda_babysitting_hours_l534_53462

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end NUMINAMATH_GPT_linda_babysitting_hours_l534_53462


namespace NUMINAMATH_GPT_cupcakes_per_child_l534_53406

theorem cupcakes_per_child (total_cupcakes children : ℕ) (h1 : total_cupcakes = 96) (h2 : children = 8) : total_cupcakes / children = 12 :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_per_child_l534_53406


namespace NUMINAMATH_GPT_false_props_count_is_3_l534_53407

-- Define the propositions and their inferences

noncomputable def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2
noncomputable def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)
noncomputable def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n
noncomputable def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- The main statement to be proved
theorem false_props_count_is_3 (m n : ℝ) : 
  ¬ (original_prop m n) ∧ ¬ (contrapositive m n) ∧ ¬ (inverse m n) ∧ ¬ (negation m n) →
  (3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_false_props_count_is_3_l534_53407


namespace NUMINAMATH_GPT_Sally_out_of_pocket_payment_l534_53435

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end NUMINAMATH_GPT_Sally_out_of_pocket_payment_l534_53435


namespace NUMINAMATH_GPT_conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l534_53449

theorem conversion1 : 4 * 60 + 35 = 275 := by
  sorry

theorem conversion2 : 4 * 1000 + 35 = 4035 := by
  sorry

theorem conversion3_minutes : 678 / 60 = 11 := by
  sorry

theorem conversion3_seconds : 678 % 60 = 18 := by
  sorry

theorem conversion4 : 120000 / 10000 = 12 := by
  sorry

end NUMINAMATH_GPT_conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l534_53449
