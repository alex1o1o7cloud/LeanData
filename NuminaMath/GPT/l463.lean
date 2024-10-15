import Mathlib

namespace NUMINAMATH_GPT_perpendicular_condition_l463_46333

-- Condition definition
def is_perpendicular (a : ℝ) : Prop :=
  let line1_slope := -1
  let line2_slope := - (a / 2)
  (line1_slope * line2_slope = -1)

-- Statement of the theorem
theorem perpendicular_condition (a : ℝ) :
  is_perpendicular a ↔ a = -2 :=
sorry

end NUMINAMATH_GPT_perpendicular_condition_l463_46333


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l463_46314

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l463_46314


namespace NUMINAMATH_GPT_miles_round_trip_time_l463_46305

theorem miles_round_trip_time : 
  ∀ (d : ℝ), d = 57 →
  ∀ (t : ℝ), t = 40 →
  ∀ (x : ℝ), x = 4 →
  10 = ((2 * d * x) / t) * 2 := 
by
  intros d hd t ht x hx
  rw [hd, ht, hx]
  sorry

end NUMINAMATH_GPT_miles_round_trip_time_l463_46305


namespace NUMINAMATH_GPT_binary_110101_is_53_l463_46368

def binary_to_decimal (n : Nat) : Nat :=
  let digits := [1, 1, 0, 1, 0, 1]  -- Define binary digits from the problem statement
  digits.reverse.foldr (λ d (acc, pow) => (acc + d * (2^pow), pow + 1)) (0, 0) |>.fst

theorem binary_110101_is_53 : binary_to_decimal 110101 = 53 := by
  sorry

end NUMINAMATH_GPT_binary_110101_is_53_l463_46368


namespace NUMINAMATH_GPT_expression_c_is_negative_l463_46320

noncomputable def A : ℝ := -4.2
noncomputable def B : ℝ := 2.3
noncomputable def C : ℝ := -0.5
noncomputable def D : ℝ := 3.4
noncomputable def E : ℝ := -1.8

theorem expression_c_is_negative : D / B * C < 0 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_expression_c_is_negative_l463_46320


namespace NUMINAMATH_GPT_children_neither_blue_nor_red_is_20_l463_46396

-- Definitions
def num_children : ℕ := 45
def num_adults : ℕ := num_children / 3
def num_adults_blue : ℕ := num_adults / 3
def num_adults_red : ℕ := 4
def num_adults_other_colors : ℕ := num_adults - num_adults_blue - num_adults_red
def num_children_red : ℕ := 15
def num_remaining_children : ℕ := num_children - num_children_red
def num_children_other_colors : ℕ := num_remaining_children / 2
def num_children_blue : ℕ := 2 * num_adults_blue
def num_children_neither_blue_nor_red : ℕ := num_children - num_children_red - num_children_blue

-- Theorem statement
theorem children_neither_blue_nor_red_is_20 : num_children_neither_blue_nor_red = 20 :=
  by
  sorry

end NUMINAMATH_GPT_children_neither_blue_nor_red_is_20_l463_46396


namespace NUMINAMATH_GPT_percentage_calculation_l463_46315

theorem percentage_calculation :
  ( (2 / 3 * 2432 / 3 + 1 / 6 * 3225) / 450 * 100 ) = 239.54 := 
sorry

end NUMINAMATH_GPT_percentage_calculation_l463_46315


namespace NUMINAMATH_GPT_problem_solution_l463_46374

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l463_46374


namespace NUMINAMATH_GPT_oil_already_put_in_engine_l463_46370

def oil_per_cylinder : ℕ := 8
def cylinders : ℕ := 6
def additional_needed_oil : ℕ := 32

theorem oil_already_put_in_engine :
  (oil_per_cylinder * cylinders) - additional_needed_oil = 16 := by
  sorry

end NUMINAMATH_GPT_oil_already_put_in_engine_l463_46370


namespace NUMINAMATH_GPT_sample_var_interpretation_l463_46393

theorem sample_var_interpretation (squared_diffs : Fin 10 → ℝ) :
  (10 = 10) ∧ (∀ i, squared_diffs i = (i - 20)^2) →
  (∃ n: ℕ, n = 10 ∧ ∃ μ: ℝ, μ = 20) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sample_var_interpretation_l463_46393


namespace NUMINAMATH_GPT_lemonade_lemons_per_glass_l463_46345

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end NUMINAMATH_GPT_lemonade_lemons_per_glass_l463_46345


namespace NUMINAMATH_GPT_wire_length_between_poles_l463_46381

theorem wire_length_between_poles :
  let x_dist := 20
  let y_dist := (18 / 2) - 8
  (x_dist ^ 2 + y_dist ^ 2 = 401) :=
by
  sorry

end NUMINAMATH_GPT_wire_length_between_poles_l463_46381


namespace NUMINAMATH_GPT_rahul_matches_played_l463_46336

-- Define the conditions of the problem
variable (m : ℕ) -- number of matches Rahul has played so far
variable (runs_before : ℕ := 51 * m) -- total runs before today's match
variable (runs_today : ℕ := 69) -- runs scored today
variable (new_average : ℕ := 54) -- new batting average after today's match

-- The equation derived from the conditions
def batting_average_equation : Prop :=
  new_average * (m + 1) = runs_before + runs_today

-- The problem: prove that m = 5 given the conditions
theorem rahul_matches_played (h : batting_average_equation m) : m = 5 :=
  sorry

end NUMINAMATH_GPT_rahul_matches_played_l463_46336


namespace NUMINAMATH_GPT_simplify_frac_l463_46311

variable (m : ℝ)

theorem simplify_frac : m^2 ≠ 9 → (3 / (m^2 - 9) + m / (9 - m^2)) = - (1 / (m + 3)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_simplify_frac_l463_46311


namespace NUMINAMATH_GPT_greatest_visible_unit_cubes_from_corner_l463_46318

theorem greatest_visible_unit_cubes_from_corner
  (n : ℕ) (units : ℕ) 
  (cube_volume : ∀ x, x = 1000)
  (face_size : ∀ x, x = 10) :
  (units = 274) :=
by sorry

end NUMINAMATH_GPT_greatest_visible_unit_cubes_from_corner_l463_46318


namespace NUMINAMATH_GPT_opposite_of_2023_l463_46357

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l463_46357


namespace NUMINAMATH_GPT_largest_five_digit_number_with_product_l463_46322

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_number_with_product_l463_46322


namespace NUMINAMATH_GPT_multiple_of_Jills_age_l463_46373

theorem multiple_of_Jills_age (m : ℤ) : 
  ∀ (J R F : ℤ),
  J = 20 →
  F = 40 →
  R = m * J + 5 →
  (R + 15) - (J + 15) = (F + 15) - 30 →
  m = 2 :=
by
  intros J R F hJ hF hR hDiff
  sorry

end NUMINAMATH_GPT_multiple_of_Jills_age_l463_46373


namespace NUMINAMATH_GPT_probability_multiple_of_3_when_die_rolled_twice_l463_46395

theorem probability_multiple_of_3_when_die_rolled_twice :
  let total_outcomes := 36
  let favorable_outcomes := 12
  (12 / 36 : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_multiple_of_3_when_die_rolled_twice_l463_46395


namespace NUMINAMATH_GPT_math_competition_l463_46317

theorem math_competition :
  let Sammy_score := 20
  let Gab_score := 2 * Sammy_score
  let Cher_score := 2 * Gab_score
  let Total_score := Sammy_score + Gab_score + Cher_score
  let Opponent_score := 85
  Total_score - Opponent_score = 55 :=
by
  sorry

end NUMINAMATH_GPT_math_competition_l463_46317


namespace NUMINAMATH_GPT_division_of_fractions_l463_46356

theorem division_of_fractions :
  (5 / 6 : ℚ) / (11 / 12) = 10 / 11 := by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l463_46356


namespace NUMINAMATH_GPT_water_level_after_opening_valve_l463_46347

-- Define the initial conditions and final height to be proved
def initial_water_height_cm : ℝ := 40
def initial_oil_height_cm : ℝ := 40
def water_density : ℝ := 1000
def oil_density : ℝ := 700
def final_water_height_cm : ℝ := 34

-- The proof that the final height of water after equilibrium will be 34 cm
theorem water_level_after_opening_valve :
  ∀ (h_w h_o : ℝ),
  (water_density * h_w = oil_density * h_o) ∧ (h_w + h_o = initial_water_height_cm + initial_oil_height_cm) →
  h_w = final_water_height_cm :=
by
  -- Here goes the proof, skipped with sorry
  sorry

end NUMINAMATH_GPT_water_level_after_opening_valve_l463_46347


namespace NUMINAMATH_GPT_tangent_line_at_x_neg1_l463_46300

-- Definition of the curve.
def curve (x : ℝ) : ℝ := 2*x - x^3

-- Definition of the point of tangency.
def point_of_tangency_x : ℝ := -1

-- Definition of the point of tangency.
def point_of_tangency_y : ℝ := curve point_of_tangency_x

-- Definition of the derivative of the curve.
def derivative (x : ℝ) : ℝ := -3*x^2 + 2

-- Slope of the tangent at the point of tangency.
def slope_at_tangency : ℝ := derivative point_of_tangency_x

-- Equation of the tangent line function.
def tangent_line (x y : ℝ) := x + y + 2 = 0

theorem tangent_line_at_x_neg1 :
  tangent_line point_of_tangency_x point_of_tangency_y :=
by
  -- Here we will perform the proof, which is omitted for the purposes of this task.
  sorry

end NUMINAMATH_GPT_tangent_line_at_x_neg1_l463_46300


namespace NUMINAMATH_GPT_find_ctg_half_l463_46360

noncomputable def ctg (x : ℝ) := 1 / (Real.tan x)

theorem find_ctg_half
  (x : ℝ)
  (h : Real.sin x - Real.cos x = (1 + 2 * Real.sqrt 2) / 3) :
  ctg (x / 2) = Real.sqrt 2 / 2 ∨ ctg (x / 2) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_ctg_half_l463_46360


namespace NUMINAMATH_GPT_probability_not_sit_next_to_each_other_l463_46319

noncomputable def total_ways_to_choose_two_chairs_excluding_broken : ℕ := 28

noncomputable def unfavorable_outcomes : ℕ := 6

theorem probability_not_sit_next_to_each_other :
  (1 - (unfavorable_outcomes / total_ways_to_choose_two_chairs_excluding_broken) = (11 / 14)) :=
by sorry

end NUMINAMATH_GPT_probability_not_sit_next_to_each_other_l463_46319


namespace NUMINAMATH_GPT_kasun_family_children_count_l463_46337

theorem kasun_family_children_count 
    (m : ℝ) (x : ℕ) (y : ℝ)
    (h1 : (m + 50 + x * y + 10) / (3 + x) = 22)
    (h2 : (m + x * y + 10) / (2 + x) = 18) :
    x = 5 :=
by
  sorry

end NUMINAMATH_GPT_kasun_family_children_count_l463_46337


namespace NUMINAMATH_GPT_remainder_of_67_pow_67_plus_67_mod_68_l463_46348

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  -- Add the conditions and final proof step
  sorry

end NUMINAMATH_GPT_remainder_of_67_pow_67_plus_67_mod_68_l463_46348


namespace NUMINAMATH_GPT_A_should_shoot_air_l463_46372

-- Define the problem conditions
def hits_A : ℝ := 0.3
def hits_B : ℝ := 1
def hits_C : ℝ := 0.5

-- Define turns
inductive Turn
| A | B | C

-- Define the strategic choice
inductive Strategy
| aim_C | aim_B | shoot_air

-- Define the outcome structure
structure DuelOutcome where
  winner : Option Turn
  probability : ℝ

-- Noncomputable definition given the context of probabilistic reasoning
noncomputable def maximize_survival : Strategy := 
sorry

-- Main theorem to prove the optimal strategy
theorem A_should_shoot_air : maximize_survival = Strategy.shoot_air := 
sorry

end NUMINAMATH_GPT_A_should_shoot_air_l463_46372


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l463_46340

variable (S T I : ℝ)
variable (h : T = (5 / 4) * S)

theorem fraction_of_income_from_tips (h : T = (5 / 4) * S) (I : ℝ) (w : I = S + T) : (T / I) = 5 / 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l463_46340


namespace NUMINAMATH_GPT_pen_cost_l463_46377

theorem pen_cost
  (p q : ℕ)
  (h1 : 6 * p + 5 * q = 380)
  (h2 : 3 * p + 8 * q = 298) :
  p = 47 :=
sorry

end NUMINAMATH_GPT_pen_cost_l463_46377


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l463_46342

theorem arithmetic_sequence_formula (a : ℕ → ℕ) (d : ℕ) (h1 : d > 0) 
  (h2 : a 1 + a 4 + a 7 = 12) (h3 : a 1 * a 4 * a 7 = 28) :
  ∀ n, a n = n :=
sorry

theorem geometric_sequence_formula (b : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : b 1 = 16) (h2 : a 2 * b 2 = 4) :
  ∀ n, b n = 2^(n + 3) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = 2^(n + 3)) 
  (h3 : ∀ n, c n = a n * b n) :
  ∀ n, T n = 8 * (2^n * (n + 1) - 1) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_formula_sum_of_sequence_l463_46342


namespace NUMINAMATH_GPT_average_weight_of_remaining_carrots_l463_46327

noncomputable def total_weight_30_carrots : ℕ := 5940
noncomputable def total_weight_3_carrots : ℕ := 540
noncomputable def carrots_count_30 : ℕ := 30
noncomputable def carrots_count_3_removed : ℕ := 3
noncomputable def carrots_count_remaining : ℕ := 27
noncomputable def average_weight_of_removed_carrots : ℕ := 180

theorem average_weight_of_remaining_carrots :
  (total_weight_30_carrots - total_weight_3_carrots) / carrots_count_remaining = 200 :=
  by
  sorry

end NUMINAMATH_GPT_average_weight_of_remaining_carrots_l463_46327


namespace NUMINAMATH_GPT_intersection_A_B_l463_46306

def A := {x : ℝ | x > 3}
def B := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l463_46306


namespace NUMINAMATH_GPT_temperature_difference_on_day_xianning_l463_46383

theorem temperature_difference_on_day_xianning 
  (highest_temp : ℝ) (lowest_temp : ℝ) 
  (h_highest : highest_temp = 2) (h_lowest : lowest_temp = -3) : 
  highest_temp - lowest_temp = 5 := 
by
  sorry

end NUMINAMATH_GPT_temperature_difference_on_day_xianning_l463_46383


namespace NUMINAMATH_GPT_borrowing_methods_l463_46354

theorem borrowing_methods (A_has_3_books : True) (B_borrows_at_least_one_book : True) :
  (∃ (methods : ℕ), methods = 7) :=
by
  existsi 7
  sorry

end NUMINAMATH_GPT_borrowing_methods_l463_46354


namespace NUMINAMATH_GPT_container_marbles_volume_l463_46389

theorem container_marbles_volume {V₁ V₂ m₁ m₂ : ℕ} 
  (h₁ : V₁ = 24) (h₂ : m₁ = 75) (h₃ : V₂ = 72) :
  m₂ = 225 :=
by
  have proportion := (m₁ : ℚ) / V₁
  have proportion2 := (m₂ : ℚ) / V₂
  have h4 := proportion = proportion2
  sorry

end NUMINAMATH_GPT_container_marbles_volume_l463_46389


namespace NUMINAMATH_GPT_ratio_area_shaded_triangle_l463_46329

variables (PQ PX QR QY YR : ℝ)
variables {A : ℝ}

def midpoint_QR (QR QY YR : ℝ) : Prop := QR = QY + YR ∧ QY = YR

def fraction_PQ_PX (PQ PX : ℝ) : Prop := PX = (3 / 4) * PQ

noncomputable def area_square (PQ : ℝ) : ℝ := PQ * PQ

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem ratio_area_shaded_triangle
  (PQ PX QR QY YR : ℝ)
  (h_mid : midpoint_QR QR QY YR)
  (h_frac : fraction_PQ_PX PQ PX)
  (hQY_QR2 : QY = QR / 2)
  (hYR_QR2 : YR = QR / 2) :
  A = 5 / 16 :=
sorry

end NUMINAMATH_GPT_ratio_area_shaded_triangle_l463_46329


namespace NUMINAMATH_GPT_solve_for_x_l463_46387

theorem solve_for_x : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l463_46387


namespace NUMINAMATH_GPT_hula_hoop_radius_l463_46324

theorem hula_hoop_radius (d : ℝ) (hd : d = 14) : d / 2 = 7 :=
by
  rw [hd]
  norm_num

end NUMINAMATH_GPT_hula_hoop_radius_l463_46324


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l463_46338

-- Problem 1
theorem problem1 : -9 + 5 - 11 + 16 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : -9 + 5 - (-6) - 18 / (-3) = 8 :=
by
  sorry

-- Problem 3
theorem problem3 : -2^2 - ((-3) * (-4 / 3) - (-2)^3) = -16 :=
by
  sorry

-- Problem 4
theorem problem4 : (59 - (7 / 9 - 11 / 12 + 1 / 6) * (-6)^2) / (-7)^2 = 58 / 49 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l463_46338


namespace NUMINAMATH_GPT_sufficient_condition_l463_46394

theorem sufficient_condition (A B : Set α) (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
  by
    intro h1
    apply h
    exact h1

end NUMINAMATH_GPT_sufficient_condition_l463_46394


namespace NUMINAMATH_GPT_find_smallest_solution_l463_46359

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end NUMINAMATH_GPT_find_smallest_solution_l463_46359


namespace NUMINAMATH_GPT_trapezoid_perimeter_l463_46316

noncomputable def perimeter_trapezoid 
  (AB CD AD BC : ℝ) 
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : ℝ :=
AB + BC + CD + AD

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : perimeter_trapezoid AB CD AD BC h_AB_CD_parallel h_AD_perpendicular h_BC_perpendicular h_AB_eq h_CD_eq h_height = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l463_46316


namespace NUMINAMATH_GPT_f_fixed_point_l463_46355

-- Definitions and conditions based on the problem statement
def g (n : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

-- Helper functions for the repeated application of f
noncomputable def f_iter (n x : ℕ) : ℕ := 
    Nat.iterate f (x^2023) n

axiom g_bijective : Function.Bijective g
axiom f_repeated : ∀ x : ℕ, f_iter x x = x
axiom f_div_g : ∀ (x y : ℕ), x ∣ y → f x ∣ g y

-- Main theorem statement
theorem f_fixed_point : ∀ x : ℕ, f x = x := by
  sorry

end NUMINAMATH_GPT_f_fixed_point_l463_46355


namespace NUMINAMATH_GPT_total_days_on_jury_duty_l463_46367

-- Define the conditions
def jury_selection_days : ℕ := 2
def trial_duration_factor : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_per_day : ℕ := 24

-- Calculate the trial duration in days
def trial_days : ℕ := trial_duration_factor * jury_selection_days

-- Calculate the total deliberation time in days
def deliberation_total_hours : ℕ := deliberation_days * deliberation_hours_per_day
def deliberation_days_converted : ℕ := deliberation_total_hours / hours_per_day

-- Statement that John spends a total of 14 days on jury duty
theorem total_days_on_jury_duty : jury_selection_days + trial_days + deliberation_days_converted = 14 :=
sorry

end NUMINAMATH_GPT_total_days_on_jury_duty_l463_46367


namespace NUMINAMATH_GPT_cube_surface_area_l463_46349

theorem cube_surface_area (a : ℝ) : 
    let edge_length := 3 * a
    let face_area := edge_length^2
    let total_surface_area := 6 * face_area
    total_surface_area = 54 * a^2 := 
by sorry

end NUMINAMATH_GPT_cube_surface_area_l463_46349


namespace NUMINAMATH_GPT_abs_diff_eq_l463_46350

-- Define the conditions
variables (x y : ℝ)
axiom h1 : x + y = 30
axiom h2 : x * y = 162

-- Define the problem to prove
theorem abs_diff_eq : |x - y| = 6 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_GPT_abs_diff_eq_l463_46350


namespace NUMINAMATH_GPT_abc_inequality_l463_46343

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := 
sorry

end NUMINAMATH_GPT_abc_inequality_l463_46343


namespace NUMINAMATH_GPT_tan_angle_addition_l463_46326

theorem tan_angle_addition (x : Real) (h1 : Real.tan x = 3) (h2 : Real.tan (Real.pi / 3) = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by 
  sorry

end NUMINAMATH_GPT_tan_angle_addition_l463_46326


namespace NUMINAMATH_GPT_f_2_values_l463_46321

theorem f_2_values (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, |f x - f y| = |x - y|)
  (hf1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 :=
sorry

end NUMINAMATH_GPT_f_2_values_l463_46321


namespace NUMINAMATH_GPT_job_positions_growth_rate_l463_46302

theorem job_positions_growth_rate (x : ℝ) :
  1501 * (1 + x) ^ 2 = 1815 := sorry

end NUMINAMATH_GPT_job_positions_growth_rate_l463_46302


namespace NUMINAMATH_GPT_derivative_at_zero_l463_46375

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2 / 3) * x

theorem derivative_at_zero : HasDerivAt f (2 / 3) 0 := sorry

end NUMINAMATH_GPT_derivative_at_zero_l463_46375


namespace NUMINAMATH_GPT_cricket_target_run_l463_46376

theorem cricket_target_run (run_rate1 run_rate2 : ℝ) (overs1 overs2 : ℕ) (T : ℝ) 
  (h1 : run_rate1 = 3.2) (h2 : overs1 = 10) (h3 : run_rate2 = 25) (h4 : overs2 = 10) :
  T = (run_rate1 * overs1) + (run_rate2 * overs2) → T = 282 :=
by
  sorry

end NUMINAMATH_GPT_cricket_target_run_l463_46376


namespace NUMINAMATH_GPT_total_listening_days_l463_46312

theorem total_listening_days (x y z t : ℕ) (h1 : x = 8) (h2 : y = 12) (h3 : z = 30) (h4 : t = 2) :
  (x + y + z) * t = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_listening_days_l463_46312


namespace NUMINAMATH_GPT_problem_statement_l463_46390

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- defining conditions
axiom a1_4_7 : a 1 + a 4 + a 7 = 39
axiom a2_5_8 : a 2 + a 5 + a 8 = 33
axiom is_arithmetic : arithmetic_seq a d

theorem problem_statement : a 5 + a 8 + a 11 = 15 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l463_46390


namespace NUMINAMATH_GPT_poly_a_c_sum_l463_46398

theorem poly_a_c_sum {a b c d : ℝ} (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + a * x + b)
  (hg : ∀ x, g x = x^2 + c * x + d)
  (hv_f_root_g : g (-a / 2) = 0)
  (hv_g_root_f : f (-c / 2) = 0)
  (f_min : ∀ x, f x ≥ -25)
  (g_min : ∀ x, g x ≥ -25)
  (f_g_intersect : f 50 = -25 ∧ g 50 = -25) : a + c = -101 :=
by
  sorry

end NUMINAMATH_GPT_poly_a_c_sum_l463_46398


namespace NUMINAMATH_GPT_complex_number_sum_equals_one_l463_46365

variable {a b c d : ℝ}
variable {ω : ℂ}

theorem complex_number_sum_equals_one
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω : ω^4 = 1) 
  (hω_ne : ω ≠ 1)
  (h_eq : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω)
  : (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 :=
by sorry

end NUMINAMATH_GPT_complex_number_sum_equals_one_l463_46365


namespace NUMINAMATH_GPT_tangent_circles_radii_l463_46313

noncomputable def radii_of_tangent_circles (R r : ℝ) (h : R > r) : Set ℝ :=
  { x | x = (R * r) / ((Real.sqrt R + Real.sqrt r)^2) ∨ x = (R * r) / ((Real.sqrt R - Real.sqrt r)^2) }

theorem tangent_circles_radii (R r : ℝ) (h : R > r) :
  ∃ x, x ∈ radii_of_tangent_circles R r h := sorry

end NUMINAMATH_GPT_tangent_circles_radii_l463_46313


namespace NUMINAMATH_GPT_lateral_area_of_given_cone_l463_46362

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_lateral_area_of_given_cone_l463_46362


namespace NUMINAMATH_GPT_sin_cos_theta_l463_46388

-- Define the problem conditions and the question as a Lean statement
theorem sin_cos_theta (θ : ℝ) (h : Real.tan (θ + Real.pi / 2) = 2) : Real.sin θ * Real.cos θ = -2 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_cos_theta_l463_46388


namespace NUMINAMATH_GPT_find_first_divisor_l463_46335

theorem find_first_divisor (x : ℕ) (k m : ℕ) (h₁ : 282 = k * x + 3) (h₂ : 282 = 9 * m + 3) : x = 31 :=
sorry

end NUMINAMATH_GPT_find_first_divisor_l463_46335


namespace NUMINAMATH_GPT_zero_knights_l463_46382

noncomputable def knights_count (n : ℕ) : ℕ := sorry

theorem zero_knights (n : ℕ) (half_lairs : n ≥ 205) :
  knights_count 410 = 0 :=
sorry

end NUMINAMATH_GPT_zero_knights_l463_46382


namespace NUMINAMATH_GPT_total_score_l463_46384

theorem total_score (score_cap : ℝ) (score_val : ℝ) (score_imp : ℝ) (wt_cap : ℝ) (wt_val : ℝ) (wt_imp : ℝ) (total_weight : ℝ) :
  score_cap = 8 → score_val = 9 → score_imp = 7 → wt_cap = 5 → wt_val = 3 → wt_imp = 2 → total_weight = 10 →
  ((score_cap * (wt_cap / total_weight)) + (score_val * (wt_val / total_weight)) + (score_imp * (wt_imp / total_weight))) = 8.1 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_score_l463_46384


namespace NUMINAMATH_GPT_Adam_marbles_l463_46344

variable (Adam Greg : Nat)

theorem Adam_marbles (h1 : Greg = 43) (h2 : Greg = Adam + 14) : Adam = 29 := 
by
  sorry

end NUMINAMATH_GPT_Adam_marbles_l463_46344


namespace NUMINAMATH_GPT_single_discount_equivalence_l463_46399

noncomputable def original_price : ℝ := 50
noncomputable def discount1 : ℝ := 0.15
noncomputable def discount2 : ℝ := 0.10
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)
noncomputable def effective_discount_price := 
  apply_discount (apply_discount original_price discount1) discount2
noncomputable def effective_discount :=
  (original_price - effective_discount_price) / original_price

theorem single_discount_equivalence :
  effective_discount = 0.235 := by
  sorry

end NUMINAMATH_GPT_single_discount_equivalence_l463_46399


namespace NUMINAMATH_GPT_triangle_right_angle_AB_solution_l463_46358

theorem triangle_right_angle_AB_solution (AC BC AB : ℝ) (hAC : AC = 6) (hBC : BC = 8) :
  (AC^2 + BC^2 = AB^2 ∨ AB^2 + AC^2 = BC^2) ↔ (AB = 10 ∨ AB = 2 * Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_triangle_right_angle_AB_solution_l463_46358


namespace NUMINAMATH_GPT_positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l463_46346

theorem positive_even_integers_less_than_1000_not_divisible_by_3_or_11 :
  ∃ n : ℕ, n = 108 ∧
    (∀ m : ℕ, 0 < m → 2 ∣ m → m < 1000 → (¬ (3 ∣ m) ∧ ¬ (11 ∣ m) ↔ m ≤ n)) :=
sorry

end NUMINAMATH_GPT_positive_even_integers_less_than_1000_not_divisible_by_3_or_11_l463_46346


namespace NUMINAMATH_GPT_maximum_gold_coins_l463_46341

theorem maximum_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n ≤ 146 :=
by
  sorry

end NUMINAMATH_GPT_maximum_gold_coins_l463_46341


namespace NUMINAMATH_GPT_value_of_a6_in_arithmetic_sequence_l463_46353

/-- In the arithmetic sequence {a_n}, if a_2 and a_{10} are the two roots of the equation
    x^2 + 12x - 8 = 0, prove that the value of a_6 is -6. -/
theorem value_of_a6_in_arithmetic_sequence :
  ∃ a_2 a_10 : ℤ, (a_2 + a_10 = -12 ∧
  (2: ℤ) * ((a_2 + a_10) / (2 * 1)) = a_2 + a_10 ) → 
  ∃ a_6: ℤ, a_6 = -6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a6_in_arithmetic_sequence_l463_46353


namespace NUMINAMATH_GPT_sample_size_9_l463_46379

variable (X : Nat)

theorem sample_size_9 (h : 36 % X = 0 ∧ 36 % (X + 1) ≠ 0) : X = 9 := 
sorry

end NUMINAMATH_GPT_sample_size_9_l463_46379


namespace NUMINAMATH_GPT_sin_cos_expr1_sin_cos_expr2_l463_46332

variable {x : ℝ}
variable (hx : Real.tan x = 2)

theorem sin_cos_expr1 : (2 / 3) * (Real.sin x)^2 + (1 / 4) * (Real.cos x)^2 = 7 / 12 := by
  sorry

theorem sin_cos_expr2 : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_cos_expr1_sin_cos_expr2_l463_46332


namespace NUMINAMATH_GPT_range_of_a_l463_46363

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the conditions: f has a unique zero point x₀ and x₀ < 0
def unique_zero_point (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0

-- The theorem we need to prove
theorem range_of_a (a : ℝ) : unique_zero_point a → a > 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l463_46363


namespace NUMINAMATH_GPT_expected_total_rainfall_10_days_l463_46334

theorem expected_total_rainfall_10_days :
  let P_sun := 0.5
  let P_rain3 := 0.3
  let P_rain6 := 0.2
  let daily_rain := (P_sun * 0) + (P_rain3 * 3) + (P_rain6 * 6)
  daily_rain * 10 = 21 :=
by
  sorry

end NUMINAMATH_GPT_expected_total_rainfall_10_days_l463_46334


namespace NUMINAMATH_GPT_sum_remainder_product_remainder_l463_46369

open Nat

-- Define the modulus conditions
variables (x y z : ℕ)
def condition1 : Prop := x % 15 = 11
def condition2 : Prop := y % 15 = 13
def condition3 : Prop := z % 15 = 14

-- Proof statement for the sum remainder
theorem sum_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x + y + z) % 15 = 8 :=
by
  sorry

-- Proof statement for the product remainder
theorem product_remainder (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 z) : (x * y * z) % 15 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_product_remainder_l463_46369


namespace NUMINAMATH_GPT_max_checkers_on_chessboard_l463_46323

theorem max_checkers_on_chessboard : 
  ∃ (w b : ℕ), (∀ r c : ℕ, r < 8 ∧ c < 8 → w = 2 * b) ∧ (8 * (w + b) = 48) ∧ (w + b) * 8 ≤ 64 :=
by sorry

end NUMINAMATH_GPT_max_checkers_on_chessboard_l463_46323


namespace NUMINAMATH_GPT_problem1_problem2_l463_46352

-- Problem 1: Prove that (a/(a - b)) + (b/(b - a)) = 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2: Prove that (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c
theorem problem2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l463_46352


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l463_46380

open Set Real

noncomputable def quadratic_inequality (x : ℝ) : Prop := -9 * x^2 + 6 * x + 8 > 0

theorem solve_quadratic_inequality :
  {x : ℝ | -9 * x^2 + 6 * x + 8 > 0} = {x : ℝ | -2/3 < x ∧ x < 4/3} :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l463_46380


namespace NUMINAMATH_GPT_pelican_count_in_shark_bite_cove_l463_46371

theorem pelican_count_in_shark_bite_cove
  (num_sharks_pelican_bay : ℕ)
  (num_pelicans_shark_bite_cove : ℕ)
  (num_pelicans_moved : ℕ) :
  num_sharks_pelican_bay = 60 →
  num_sharks_pelican_bay = 2 * num_pelicans_shark_bite_cove →
  num_pelicans_moved = num_pelicans_shark_bite_cove / 3 →
  num_pelicans_shark_bite_cove - num_pelicans_moved = 20 :=
by
  sorry

end NUMINAMATH_GPT_pelican_count_in_shark_bite_cove_l463_46371


namespace NUMINAMATH_GPT_total_suitcases_correct_l463_46386

-- Conditions as definitions
def num_siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def num_parents : Nat := 2
def suitcases_per_parent : Nat := 3

-- Total suitcases calculation
def total_suitcases :=
  (num_siblings * suitcases_per_sibling) + (num_parents * suitcases_per_parent)

-- Statement to prove
theorem total_suitcases_correct : total_suitcases = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_suitcases_correct_l463_46386


namespace NUMINAMATH_GPT_ryan_final_tokens_l463_46366

-- Conditions
def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 2 / 3
def candy_crush_fraction : ℚ := 1 / 2
def skiball_tokens : ℕ := 7
def friend_borrowed_tokens : ℕ := 5
def friend_returned_tokens : ℕ := 8
def laser_tag_tokens : ℕ := 3
def parents_purchase_factor : ℕ := 10

-- Final Answer
theorem ryan_final_tokens : initial_tokens - 24  - 6 - skiball_tokens + friend_returned_tokens + (parents_purchase_factor * skiball_tokens) - laser_tag_tokens = 75 :=
by sorry

end NUMINAMATH_GPT_ryan_final_tokens_l463_46366


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_20_l463_46301

open BigOperators

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 1 - a 0)

theorem arithmetic_sequence_sum_20 {a : ℕ → ℤ} (h_arith : is_arithmetic_sequence a)
    (h1 : a 0 + a 1 + a 2 = -24)
    (h18 : a 17 + a 18 + a 19 = 78) :
    ∑ i in Finset.range 20, a i = 180 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_20_l463_46301


namespace NUMINAMATH_GPT_non_neg_solutions_l463_46328

theorem non_neg_solutions (x y z : ℕ) :
  (x^3 = 2 * y^2 - z) →
  (y^3 = 2 * z^2 - x) →
  (z^3 = 2 * x^2 - y) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_non_neg_solutions_l463_46328


namespace NUMINAMATH_GPT_ratio_a_to_c_l463_46385

variables {a b c d : ℚ}

theorem ratio_a_to_c
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 3 / 10) :
  a / c = 25 / 12 :=
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l463_46385


namespace NUMINAMATH_GPT_intersecting_lines_a_b_sum_zero_l463_46331

theorem intersecting_lines_a_b_sum_zero
    (a b : ℝ)
    (h₁ : ∀ z : ℝ × ℝ, z = (3, -3) → z.1 = (1 / 3) * z.2 + a)
    (h₂ : ∀ z : ℝ × ℝ, z = (3, -3) → z.2 = (1 / 3) * z.1 + b)
    :
    a + b = 0 := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_a_b_sum_zero_l463_46331


namespace NUMINAMATH_GPT_drying_time_short_haired_dog_l463_46330

theorem drying_time_short_haired_dog (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : 6 * x + 9 * (2 * x) = 240) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_drying_time_short_haired_dog_l463_46330


namespace NUMINAMATH_GPT_check_numbers_has_property_P_l463_46325

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem check_numbers_has_property_P :
  has_property_P 1 ∧ has_property_P 5 ∧ has_property_P 2014 ∧ ¬has_property_P 2013 :=
by
  sorry

end NUMINAMATH_GPT_check_numbers_has_property_P_l463_46325


namespace NUMINAMATH_GPT_math_problem_l463_46391

variables {R : Type*} [Ring R] (x y z : R)

theorem math_problem (h : x * y + y * z + z * x = 0) : 
  3 * x * y * z + x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_math_problem_l463_46391


namespace NUMINAMATH_GPT_z_in_fourth_quadrant_l463_46361

noncomputable def z : ℂ := (3 * Complex.I - 2) / (Complex.I - 1) * Complex.I

theorem z_in_fourth_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_GPT_z_in_fourth_quadrant_l463_46361


namespace NUMINAMATH_GPT_christopher_age_l463_46308

theorem christopher_age (G C : ℕ) (h1 : C = 2 * G) (h2 : C - 9 = 5 * (G - 9)) : C = 24 := 
by
  sorry

end NUMINAMATH_GPT_christopher_age_l463_46308


namespace NUMINAMATH_GPT_probability_of_not_shorter_than_one_meter_l463_46303

noncomputable def probability_of_event_A : ℝ := 
  let length_of_rope : ℝ := 3
  let event_A_probability : ℝ := 1 / 3
  event_A_probability

theorem probability_of_not_shorter_than_one_meter (l : ℝ) (h_l : l = 3) : 
    probability_of_event_A = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_of_not_shorter_than_one_meter_l463_46303


namespace NUMINAMATH_GPT_number_of_drawings_on_first_page_l463_46339

-- Let D be the number of drawings on the first page.
variable (D : ℕ)

-- Conditions:
-- 1. D is the number of drawings on the first page.
-- 2. The number of drawings increases by 5 after every page.
-- 3. The total number of drawings in the first five pages is 75.

theorem number_of_drawings_on_first_page (h : D + (D + 5) + (D + 10) + (D + 15) + (D + 20) = 75) :
    D = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_drawings_on_first_page_l463_46339


namespace NUMINAMATH_GPT_rectangle_segments_sum_l463_46364

theorem rectangle_segments_sum :
  let EF := 6
  let FG := 8
  let n := 210
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length (k : ℕ) : ℝ := diagonal_length * (n - k) / n
  let sum_segments := 2 * (Finset.sum (Finset.range 210) segment_length) - diagonal_length
  sum_segments = 2080 := by
  sorry

end NUMINAMATH_GPT_rectangle_segments_sum_l463_46364


namespace NUMINAMATH_GPT_correct_structure_l463_46310

-- Definitions for the conditions regarding flowchart structures
def loop_contains_conditional : Prop := ∀ (loop : Prop), ∃ (conditional : Prop), conditional ∧ loop
def unique_flowchart_for_boiling_water : Prop := ∀ (flowcharts : Prop), ∃! (boiling_process : Prop), flowcharts ∧ boiling_process
def conditional_does_not_contain_sequential : Prop := ∀ (conditional : Prop), ∃ (sequential : Prop), ¬ (conditional ∧ sequential)
def conditional_must_contain_loop : Prop := ∀ (conditional : Prop), ∃ (loop : Prop), conditional ∧ loop

-- The proof statement
theorem correct_structure (A B C D : Prop) (hA : A = loop_contains_conditional) 
  (hB : B = unique_flowchart_for_boiling_water) 
  (hC : C = conditional_does_not_contain_sequential) 
  (hD : D = conditional_must_contain_loop) : 
  A = loop_contains_conditional ∧ ¬ B ∧ ¬ C ∧ ¬ D :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_structure_l463_46310


namespace NUMINAMATH_GPT_cheryl_used_material_l463_46392

theorem cheryl_used_material 
  (a b c l : ℚ) 
  (ha : a = 3 / 8) 
  (hb : b = 1 / 3) 
  (hl : l = 15 / 40) 
  (Hc: c = a + b): 
  (c - l = 1 / 3) := 
by 
  -- proof will be deferred to Lean's syntax for user to fill in.
  sorry

end NUMINAMATH_GPT_cheryl_used_material_l463_46392


namespace NUMINAMATH_GPT_first_digit_base9_650_l463_46397

theorem first_digit_base9_650 : ∃ d : ℕ, 
  d = 8 ∧ (∃ k : ℕ, 650 = d * 9^2 + k ∧ k < 9^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_first_digit_base9_650_l463_46397


namespace NUMINAMATH_GPT_odd_function_behavior_on_interval_l463_46351

theorem odd_function_behavior_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 4 → f x₁ < f x₂)
  (h_max : ∀ x, 1 ≤ x → x ≤ 4 → f x ≤ 5) :
  (∀ x, -4 ≤ x → x ≤ -1 → f (-4) ≤ f x ∧ f x ≤ f (-1)) ∧ f (-4) = -5 :=
sorry

end NUMINAMATH_GPT_odd_function_behavior_on_interval_l463_46351


namespace NUMINAMATH_GPT_problem_1_problem_2_l463_46304

noncomputable def f (x k : ℝ) : ℝ := (2 * k * x) / (x * x + 6 * k)

theorem problem_1 (k m : ℝ) (hk : k > 0)
  (hsol : ∀ x, (f x k) > m ↔ x < -3 ∨ x > -2) :
  ∀ x, 5 * m * x ^ 2 + k * x + 3 > 0 ↔ -1 < x ∧ x < 3 / 2 :=
sorry

theorem problem_2 (k : ℝ) (hk : k > 0)
  (hsol : ∃ (x : ℝ), x > 3 ∧ (f x k) > 1) :
  k > 6 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l463_46304


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l463_46378

variables (a r : ℝ) (h1 : a * r ^ 2 = 12 / 5) (h2 : a * r ^ 6 = 48)

theorem geometric_sequence_fifth_term : a * r ^ 4 = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l463_46378


namespace NUMINAMATH_GPT_find_a_l463_46307

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (a - 1) * x

theorem find_a {a : ℝ} : 
  (∀ x : ℝ, 0 < x → f x a ≤ x^2 * Real.exp x - Real.log x - 4 * x - 1) → 
  a ≤ -2 :=
sorry

end NUMINAMATH_GPT_find_a_l463_46307


namespace NUMINAMATH_GPT_min_value_of_quadratic_l463_46309

theorem min_value_of_quadratic (m : ℝ) (x : ℝ) (hx1 : 3 ≤ x) (hx2 : x < 4) (h : x^2 - 4 * x ≥ m) : 
  m ≤ -3 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l463_46309
