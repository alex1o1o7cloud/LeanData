import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1123_112388

theorem simplify_expression :
  (∃ (x : Real), x = 3 * (Real.sqrt 3 + Real.sqrt 7) / (4 * Real.sqrt (3 + Real.sqrt 5)) ∧ 
    x = Real.sqrt (224 - 22 * Real.sqrt 105) / 8) := sorry

end NUMINAMATH_GPT_simplify_expression_l1123_112388


namespace NUMINAMATH_GPT_speed_of_man_l1123_112327

theorem speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ)
  (relative_speed_km_h : ℝ)
  (h_train_length : train_length = 440)
  (h_train_speed : train_speed_kmph = 60)
  (h_time : time_seconds = 24)
  (h_relative_speed : relative_speed_km_h = (train_length / time_seconds) * 3.6):
  (relative_speed_km_h - train_speed_kmph) = 6 :=
by sorry

end NUMINAMATH_GPT_speed_of_man_l1123_112327


namespace NUMINAMATH_GPT_three_digit_sum_seven_l1123_112335

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_three_digit_sum_seven_l1123_112335


namespace NUMINAMATH_GPT_C_gets_more_than_D_l1123_112334

-- Define the conditions
def proportion_B := 3
def share_B : ℕ := 3000
def proportion_C := 5
def proportion_D := 4

-- Define the parts based on B's share
def part_value := share_B / proportion_B

-- Define the shares based on the proportions
def share_C := proportion_C * part_value
def share_D := proportion_D * part_value

-- Prove the final statement about the difference
theorem C_gets_more_than_D : share_C - share_D = 1000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_C_gets_more_than_D_l1123_112334


namespace NUMINAMATH_GPT_min_distance_squared_l1123_112333

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end NUMINAMATH_GPT_min_distance_squared_l1123_112333


namespace NUMINAMATH_GPT_speed_of_first_boy_l1123_112320

theorem speed_of_first_boy (x : ℝ) (h1 : 7.5 > 0) (h2 : 16 > 0) (h3 : 32 > 0) (h4 : 32 = 16 * (x - 7.5)) : x = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_first_boy_l1123_112320


namespace NUMINAMATH_GPT_quadrilateral_area_l1123_112322

theorem quadrilateral_area (a b c d : ℝ) (horizontally_vertically_apart : a = b + 1 ∧ b = c + 1 ∧ c = d + 1 ∧ d = a + 1) : 
  area_of_quadrilateral = 6 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_l1123_112322


namespace NUMINAMATH_GPT_sunil_total_amount_l1123_112314

noncomputable def principal (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  CI / ((1 + R / 100) ^ T - 1)

noncomputable def total_amount (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  let P := principal CI R T
  P + CI

theorem sunil_total_amount (CI : ℝ) (R : ℝ) (T : ℕ) :
  CI = 420 → R = 10 → T = 2 → total_amount CI R T = 2420 := by
  intros hCI hR hT
  rw [hCI, hR, hT]
  sorry

end NUMINAMATH_GPT_sunil_total_amount_l1123_112314


namespace NUMINAMATH_GPT_todd_initial_gum_l1123_112323

theorem todd_initial_gum (x : ℝ)
(h1 : 150 = 0.25 * x)
(h2 : x + 150 = 890) :
x = 712 :=
by
  -- Here "by" is used to denote the beginning of proof block
  sorry -- Proof will be filled in later.

end NUMINAMATH_GPT_todd_initial_gum_l1123_112323


namespace NUMINAMATH_GPT_greatest_value_of_x_is_20_l1123_112363

noncomputable def greatest_multiple_of_4 (x : ℕ) : Prop :=
  (x % 4 = 0 ∧ x^2 < 500 ∧ ∀ y : ℕ, (y % 4 = 0 ∧ y^2 < 500) → y ≤ x)

theorem greatest_value_of_x_is_20 : greatest_multiple_of_4 20 :=
  by 
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_is_20_l1123_112363


namespace NUMINAMATH_GPT_average_population_increase_l1123_112349

-- Conditions
def population_2000 : ℕ := 450000
def population_2005 : ℕ := 467000
def years : ℕ := 5

-- Theorem statement
theorem average_population_increase :
  (population_2005 - population_2000) / years = 3400 := by
  sorry

end NUMINAMATH_GPT_average_population_increase_l1123_112349


namespace NUMINAMATH_GPT_solution_set_f_2_minus_x_l1123_112311

def f (x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_f_2_minus_x (a b : ℝ) (h_even : b - 2 * a = 0)
  (h_mono : 0 < a) :
  {x : ℝ | f (2 - x) a b > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_2_minus_x_l1123_112311


namespace NUMINAMATH_GPT_root_difference_l1123_112300

theorem root_difference (p : ℝ) (r s : ℝ) :
  (r + s = p) ∧ (r * s = (p^2 - 1) / 4) ∧ (r ≥ s) → r - s = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_root_difference_l1123_112300


namespace NUMINAMATH_GPT_problem_conditions_l1123_112377

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l1123_112377


namespace NUMINAMATH_GPT_intersection_subset_complement_l1123_112370

open Set

variable (U A B : Set ℕ)

theorem intersection_subset_complement (U : Set ℕ) (A B : Set ℕ) 
  (hU: U = {1, 2, 3, 4, 5, 6}) 
  (hA: A = {1, 3, 5}) 
  (hB: B = {2, 4, 5}) : 
  A ∩ (U \ B) = {1, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_subset_complement_l1123_112370


namespace NUMINAMATH_GPT_sum_of_sequence_eq_six_seventeenth_l1123_112396

noncomputable def cn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.cos (n * Real.arctan (2 / 3))
noncomputable def dn (n : ℕ) : ℝ := (Real.sqrt 13) ^ n * Real.sin (n * Real.arctan (2 / 3))

theorem sum_of_sequence_eq_six_seventeenth : 
  (∑' n : ℕ, (cn n * dn n / 8^n)) = 6/17 := sorry

end NUMINAMATH_GPT_sum_of_sequence_eq_six_seventeenth_l1123_112396


namespace NUMINAMATH_GPT_extremum_of_f_l1123_112353

def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

theorem extremum_of_f :
  ∃ (xmin xmax : ℝ) (x1 y1 : ℝ), f x1 y1 = xmin ∧ (x1 = 11 ∧ y1 = 2) ∧
  ∃ (xmax : ℝ) (x2 y2 : ℝ), f x2 y2 = xmax ∧ (x2 = 1 ∧ y2 = 4) ∧
  xmin = 10 ∧ xmax = 570 := 
by
  sorry

end NUMINAMATH_GPT_extremum_of_f_l1123_112353


namespace NUMINAMATH_GPT_sarahs_score_l1123_112336

theorem sarahs_score (g s : ℕ) (h1 : s = g + 60) (h2 : s + g = 260) : s = 160 :=
sorry

end NUMINAMATH_GPT_sarahs_score_l1123_112336


namespace NUMINAMATH_GPT_no_integral_roots_l1123_112398

theorem no_integral_roots :
  ¬(∃ (x : ℤ), 5 * x^2 + 3 = 40) ∧
  ¬(∃ (x : ℤ), (3 * x - 2)^3 = (x - 2)^3 - 27) ∧
  ¬(∃ (x : ℤ), x^2 - 4 = 3 * x - 4) :=
by sorry

end NUMINAMATH_GPT_no_integral_roots_l1123_112398


namespace NUMINAMATH_GPT_find_principal_sum_l1123_112341

theorem find_principal_sum (P : ℝ) (r : ℝ) (A2 : ℝ) (A3 : ℝ) : 
  (A2 = 7000) → (A3 = 9261) → 
  (A2 = P * (1 + r)^2) → (A3 = P * (1 + r)^3) → 
  P = 4000 :=
by
  intro hA2 hA3 hA2_eq hA3_eq
  -- here, we assume the proof steps leading to P = 4000
  sorry

end NUMINAMATH_GPT_find_principal_sum_l1123_112341


namespace NUMINAMATH_GPT_probability_of_experts_winning_l1123_112319

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end NUMINAMATH_GPT_probability_of_experts_winning_l1123_112319


namespace NUMINAMATH_GPT_min_value_reciprocal_l1123_112385

theorem min_value_reciprocal (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  3 ≤ (1/a) + (1/b) + (1/c) :=
sorry

end NUMINAMATH_GPT_min_value_reciprocal_l1123_112385


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1123_112389

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ),
    ρ = 15 →
    θ = 5 * Real.pi / 6 →
    φ = Real.pi / 3 →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    z = ρ * Real.cos φ →
    x = -45 / 4 ∧ y = -15 * Real.sqrt 3 / 4 ∧ z = 7.5 := 
by
  intro ρ θ φ x y z
  intro hρ hθ hφ hx hy hz
  rw [hρ, hθ, hφ] at *
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1123_112389


namespace NUMINAMATH_GPT_send_messages_ways_l1123_112376

theorem send_messages_ways : (3^4 = 81) :=
by
  sorry

end NUMINAMATH_GPT_send_messages_ways_l1123_112376


namespace NUMINAMATH_GPT_no_such_natural_number_exists_l1123_112306

theorem no_such_natural_number_exists :
  ¬ ∃ (n : ℕ), (∃ (m k : ℤ), 2 * n - 5 = 9 * m ∧ n - 2 = 15 * k) :=
by
  sorry

end NUMINAMATH_GPT_no_such_natural_number_exists_l1123_112306


namespace NUMINAMATH_GPT_amount_of_flour_already_put_in_l1123_112351

theorem amount_of_flour_already_put_in 
  (total_flour_needed : ℕ) (flour_remaining : ℕ) (x : ℕ) 
  (h1 : total_flour_needed = 9) 
  (h2 : flour_remaining = 7) 
  (h3 : total_flour_needed - flour_remaining = x) : 
  x = 2 := 
sorry

end NUMINAMATH_GPT_amount_of_flour_already_put_in_l1123_112351


namespace NUMINAMATH_GPT_johanna_loses_half_turtles_l1123_112305

theorem johanna_loses_half_turtles
  (owen_turtles_initial : ℕ)
  (johanna_turtles_fewer : ℕ)
  (owen_turtles_after_month : ℕ)
  (owen_turtles_final : ℕ)
  (johanna_donates_rest_to_owen : ℚ → ℚ)
  (x : ℚ)
  (hx1 : owen_turtles_initial = 21)
  (hx2 : johanna_turtles_fewer = 5)
  (hx3 : owen_turtles_after_month = owen_turtles_initial * 2)
  (hx4 : owen_turtles_final = owen_turtles_after_month + johanna_donates_rest_to_owen (1 - x))
  (hx5 : owen_turtles_final = 50) :
  x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_johanna_loses_half_turtles_l1123_112305


namespace NUMINAMATH_GPT_james_vegetable_consumption_l1123_112393

def vegetable_consumption_weekdays (asparagus broccoli cauliflower spinach : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + spinach

def vegetable_consumption_weekend (asparagus broccoli cauliflower other_veg : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + other_veg

def total_vegetable_consumption (
  wd_asparagus wd_broccoli wd_cauliflower wd_spinach : ℚ)
  (sat_asparagus sat_broccoli sat_cauliflower sat_other : ℚ)
  (sun_asparagus sun_broccoli sun_cauliflower sun_other : ℚ) : ℚ :=
  5 * vegetable_consumption_weekdays wd_asparagus wd_broccoli wd_cauliflower wd_spinach +
  vegetable_consumption_weekend sat_asparagus sat_broccoli sat_cauliflower sat_other +
  vegetable_consumption_weekend sun_asparagus sun_broccoli sun_cauliflower sun_other

theorem james_vegetable_consumption :
  total_vegetable_consumption 0.5 0.75 0.875 0.5 0.3 0.4 0.6 1 0.3 0.4 0.6 0.5 = 17.225 :=
sorry

end NUMINAMATH_GPT_james_vegetable_consumption_l1123_112393


namespace NUMINAMATH_GPT_sum_first_12_terms_geom_seq_l1123_112365

def geometric_sequence_periodic (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem sum_first_12_terms_geom_seq :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 2 ∧
    a 3 = 4 ∧
    geometric_sequence_periodic a 8 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_12_terms_geom_seq_l1123_112365


namespace NUMINAMATH_GPT_mod_product_l1123_112367

theorem mod_product :
  (105 * 86 * 97) % 25 = 10 :=
by
  sorry

end NUMINAMATH_GPT_mod_product_l1123_112367


namespace NUMINAMATH_GPT_cycle_original_cost_l1123_112344

theorem cycle_original_cost (SP : ℝ) (gain : ℝ) (CP : ℝ) (h₁ : SP = 2000) (h₂ : gain = 1) (h₃ : SP = CP * (1 + gain)) : CP = 1000 :=
by
  sorry

end NUMINAMATH_GPT_cycle_original_cost_l1123_112344


namespace NUMINAMATH_GPT_find_third_number_l1123_112387

-- Given conditions
variable (A B C : ℕ)
variable (LCM HCF : ℕ)
variable (h1 : A = 36)
variable (h2 : B = 44)
variable (h3 : LCM = 792)
variable (h4 : HCF = 12)
variable (h5 : A * B * C = LCM * HCF)

-- Desired proof
theorem find_third_number : C = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l1123_112387


namespace NUMINAMATH_GPT_infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l1123_112302

noncomputable def grid_size := 10
noncomputable def initial_infected_count_1 := 9
noncomputable def initial_infected_count_2 := 10

def condition (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n → 
  infected + steps * (infected / 2) < grid_size * grid_size

def can_infect_entire_grid (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n ∧ (
  ∃ t : ℕ, infected + t * (infected / 2) = grid_size * grid_size)

theorem infection_does_not_spread_with_9_cells :
  ¬ can_infect_entire_grid initial_infected_count_1 :=
by
  sorry

theorem minimum_infected_cells_needed :
  condition initial_infected_count_2 :=
by
  sorry

end NUMINAMATH_GPT_infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l1123_112302


namespace NUMINAMATH_GPT_tangent_slope_of_cubic_l1123_112368

theorem tangent_slope_of_cubic (P : ℝ × ℝ) (tangent_at_P : ℝ) (h1 : P.snd = P.fst ^ 3)
  (h2 : tangent_at_P = 3) : P = (1,1) ∨ P = (-1,-1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_slope_of_cubic_l1123_112368


namespace NUMINAMATH_GPT_sequences_get_arbitrarily_close_l1123_112340

noncomputable def a_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^n
noncomputable def b_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^(n + 1)

theorem sequences_get_arbitrarily_close (n : ℕ) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b_n n - a_n n| < ε :=
sorry

end NUMINAMATH_GPT_sequences_get_arbitrarily_close_l1123_112340


namespace NUMINAMATH_GPT_identical_digits_time_l1123_112337

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end NUMINAMATH_GPT_identical_digits_time_l1123_112337


namespace NUMINAMATH_GPT_sum_of_coefficients_l1123_112325

noncomputable def u : ℕ → ℕ
| 0       => 5
| (n + 1) => u n + (3 + 4 * (n - 1))

theorem sum_of_coefficients :
  (2 + -3 + 6) = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_coefficients_l1123_112325


namespace NUMINAMATH_GPT_arithmetic_sequence_a2015_l1123_112330

theorem arithmetic_sequence_a2015 :
  (∀ n : ℕ, n > 0 → (∃ a_n a_n1 : ℝ,
    a_n1 = a_n + 2 ∧ a_n + a_n1 = 4 * n - 58))
  → (∃ a_2015 : ℝ, a_2015 = 4000) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2015_l1123_112330


namespace NUMINAMATH_GPT_ab_minus_a_plus_b_eq_two_l1123_112307

theorem ab_minus_a_plus_b_eq_two
  (a b : ℝ)
  (h1 : a + 1 ≠ 0)
  (h2 : b - 1 ≠ 0)
  (h3 : a + (1 / (a + 1)) = b + (1 / (b - 1)) - 2)
  (h4 : a - b + 2 ≠ 0)
: ab - a + b = 2 :=
sorry

end NUMINAMATH_GPT_ab_minus_a_plus_b_eq_two_l1123_112307


namespace NUMINAMATH_GPT_even_odd_product_zero_l1123_112347

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_product_zero (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : is_even f) (hg : is_odd g) : ∀ x, f (-x) * g (-x) + f x * g x = 0 :=
by
  intro x
  have h₁ := hf x
  have h₂ := hg x
  sorry

end NUMINAMATH_GPT_even_odd_product_zero_l1123_112347


namespace NUMINAMATH_GPT_problem_statement_l1123_112392

def g (x : ℝ) : ℝ := x ^ 3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem problem_statement : f (g 3) = 53 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1123_112392


namespace NUMINAMATH_GPT_exist_c_l1123_112358

theorem exist_c (p : ℕ) (r : ℤ) (a b : ℤ) [Fact (Nat.Prime p)]
  (hp1 : r^7 ≡ 1 [ZMOD p])
  (hp2 : r + 1 - a^2 ≡ 0 [ZMOD p])
  (hp3 : r^2 + 1 - b^2 ≡ 0 [ZMOD p]) :
  ∃ c : ℤ, (r^3 + 1 - c^2) ≡ 0 [ZMOD p] :=
by
  sorry

end NUMINAMATH_GPT_exist_c_l1123_112358


namespace NUMINAMATH_GPT_chord_length_intercepted_by_line_on_circle_l1123_112352

theorem chord_length_intercepted_by_line_on_circle :
  ∀ (ρ θ : ℝ), (ρ = 4) →
  (ρ * Real.sin (θ + (Real.pi / 4)) = 2) →
  (4 * Real.sqrt (16 - (2 ^ 2)) = 4 * Real.sqrt 3) :=
by
  intros ρ θ hρ hline_eq
  sorry

end NUMINAMATH_GPT_chord_length_intercepted_by_line_on_circle_l1123_112352


namespace NUMINAMATH_GPT_quadrilateral_area_l1123_112310

theorem quadrilateral_area (c d : ℤ) (h1 : 0 < d) (h2 : d < c) (h3 : 2 * ((c : ℝ) ^ 2 - (d : ℝ) ^ 2) = 18) : 
  c + d = 9 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1123_112310


namespace NUMINAMATH_GPT_perpendicular_vector_l1123_112364

-- Vectors a and b are given
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

-- Defining the vector addition and scalar multiplication for our context
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (m : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (m * v.1, m * v.2)

-- The vector a + m * b
def a_plus_m_b (m : ℝ) : ℝ × ℝ := vector_add a (scalar_mul m b)

-- The dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The statement that a is perpendicular to (a + m * b) when m = 5
theorem perpendicular_vector : dot_product a (a_plus_m_b 5) = 0 :=
sorry

end NUMINAMATH_GPT_perpendicular_vector_l1123_112364


namespace NUMINAMATH_GPT_no_valid_sum_seventeen_l1123_112380

def std_die (n : ℕ) : Prop := n ∈ [1, 2, 3, 4, 5, 6]

def valid_dice (a b c d : ℕ) : Prop := std_die a ∧ std_die b ∧ std_die c ∧ std_die d

def sum_dice (a b c d : ℕ) : ℕ := a + b + c + d

def prod_dice (a b c d : ℕ) : ℕ := a * b * c * d

theorem no_valid_sum_seventeen (a b c d : ℕ) (h_valid : valid_dice a b c d) (h_prod : prod_dice a b c d = 360) : sum_dice a b c d ≠ 17 :=
sorry

end NUMINAMATH_GPT_no_valid_sum_seventeen_l1123_112380


namespace NUMINAMATH_GPT_minimize_cost_l1123_112342

-- Define the unit prices of the soccer balls.
def price_A := 50
def price_B := 80

-- Define the condition for the total number of balls and cost function.
def total_balls := 80
def cost (a : ℕ) : ℕ := price_A * a + price_B * (total_balls - a)
def valid_a (a : ℕ) : Prop := 30 ≤ a ∧ a ≤ (3 * (total_balls - a))

-- Prove the number of brand A soccer balls to minimize the total cost.
theorem minimize_cost : ∃ a : ℕ, valid_a a ∧ ∀ b : ℕ, valid_a b → cost a ≤ cost b :=
sorry

end NUMINAMATH_GPT_minimize_cost_l1123_112342


namespace NUMINAMATH_GPT_latte_cost_l1123_112326

theorem latte_cost :
  ∃ (latte_cost : ℝ), 
    2 * 2.25 + 3.50 + 0.50 + 2 * 2.50 + 3.50 + 2 * latte_cost = 25.00 ∧ 
    latte_cost = 4.00 :=
by
  use 4.00
  simp
  sorry

end NUMINAMATH_GPT_latte_cost_l1123_112326


namespace NUMINAMATH_GPT_calculate_three_times_neg_two_l1123_112372

-- Define the multiplication of a positive and a negative number resulting in a negative number
def multiply_positive_negative (a b : Int) (ha : a > 0) (hb : b < 0) : Int :=
  a * b

-- Define the absolute value multiplication
def absolute_value_multiplication (a b : Int) : Int :=
  abs a * abs b

-- The theorem that verifies the calculation
theorem calculate_three_times_neg_two : 3 * (-2) = -6 :=
by
  -- Using the given conditions to conclude the result
  sorry

end NUMINAMATH_GPT_calculate_three_times_neg_two_l1123_112372


namespace NUMINAMATH_GPT_last_digit_inverse_power_two_l1123_112378

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_last_digit_inverse_power_two_l1123_112378


namespace NUMINAMATH_GPT_determine_x_l1123_112348

theorem determine_x (x : ℕ) 
  (hx1 : x % 6 = 0) 
  (hx2 : x^2 > 196) 
  (hx3 : x < 30) : 
  x = 18 ∨ x = 24 := 
sorry

end NUMINAMATH_GPT_determine_x_l1123_112348


namespace NUMINAMATH_GPT_find_x0_l1123_112359

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 1
else if x < 0 then -x^2 + 1
else 0

theorem find_x0 :
  ∃ x0 : ℝ, f x0 = 1/2 ∧ x0 = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l1123_112359


namespace NUMINAMATH_GPT_pie_eating_contest_l1123_112386

theorem pie_eating_contest :
  let a := 5 / 6
  let b := 7 / 8
  let c := 2 / 3
  let max_pie := max a (max b c)
  let min_pie := min a (min b c)
  max_pie - min_pie = 5 / 24 :=
by
  sorry

end NUMINAMATH_GPT_pie_eating_contest_l1123_112386


namespace NUMINAMATH_GPT_opposite_of_expression_l1123_112355

theorem opposite_of_expression : 
  let expr := 1 - (3 : ℝ)^(1/3)
  (-1 + (3 : ℝ)^(1/3)) = (3 : ℝ)^(1/3) - 1 :=
by 
  let expr := 1 - (3 : ℝ)^(1/3)
  sorry

end NUMINAMATH_GPT_opposite_of_expression_l1123_112355


namespace NUMINAMATH_GPT_determine_valid_m_l1123_112331

-- The function given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

-- The range of values for m
def valid_m (m : ℝ) : Prop := -1/4 ≤ m ∧ m ≤ 0

-- The condition that f is increasing on (-∞, 2)
def increasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < a → f x ≤ f y

-- The main statement we want to prove
theorem determine_valid_m (m : ℝ) :
  increasing_on_interval (f m) 2 ↔ valid_m m :=
sorry

end NUMINAMATH_GPT_determine_valid_m_l1123_112331


namespace NUMINAMATH_GPT_mushroom_problem_l1123_112329

variables (x1 x2 x3 x4 : ℕ)

theorem mushroom_problem
  (h1 : x1 + x2 = 6)
  (h2 : x1 + x3 = 7)
  (h3 : x2 + x3 = 9)
  (h4 : x2 + x4 = 11)
  (h5 : x3 + x4 = 12)
  (h6 : x1 + x4 = 9) :
  x1 = 2 ∧ x2 = 4 ∧ x3 = 5 ∧ x4 = 7 := 
  by
    sorry

end NUMINAMATH_GPT_mushroom_problem_l1123_112329


namespace NUMINAMATH_GPT_pencils_sold_l1123_112354

theorem pencils_sold (C S : ℝ) (n : ℝ) 
  (h1 : 12 * C = n * S) (h2 : S = 1.5 * C) : n = 8 := by
  sorry

end NUMINAMATH_GPT_pencils_sold_l1123_112354


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1123_112357

theorem sum_of_three_numbers (a b c : ℕ)
    (h1 : a + b = 35)
    (h2 : b + c = 40)
    (h3 : c + a = 45) :
    a + b + c = 60 := 
  by sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1123_112357


namespace NUMINAMATH_GPT_relationship_of_arithmetic_progression_l1123_112316

theorem relationship_of_arithmetic_progression (x y z d : ℝ) (h1 : x + (y - z) + d = y + (z - x))
    (h2 : y + (z - x) + d = z + (x - y))
    (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
    x = y + d / 2 ∧ z = y + d := by
  sorry

end NUMINAMATH_GPT_relationship_of_arithmetic_progression_l1123_112316


namespace NUMINAMATH_GPT_best_sampling_method_l1123_112391

theorem best_sampling_method :
  let elderly := 27
  let middle_aged := 54
  let young := 81
  let total_population := elderly + middle_aged + young
  let sample_size := 36
  let sampling_methods := ["simple random sampling", "systematic sampling", "stratified sampling"]
  stratified_sampling
:=
by
  sorry

end NUMINAMATH_GPT_best_sampling_method_l1123_112391


namespace NUMINAMATH_GPT_pencils_per_child_l1123_112395

-- Define the conditions
def totalPencils : ℕ := 18
def numberOfChildren : ℕ := 9

-- The proof problem
theorem pencils_per_child : totalPencils / numberOfChildren = 2 := 
by
  sorry

end NUMINAMATH_GPT_pencils_per_child_l1123_112395


namespace NUMINAMATH_GPT_seating_arrangement_l1123_112339

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem seating_arrangement : 
  let republicans := 6
  let democrats := 4
  (factorial (republicans - 1)) * (binom republicans democrats) * (factorial democrats) = 43200 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1123_112339


namespace NUMINAMATH_GPT_ken_paid_20_l1123_112379

section
variable (pound_price : ℤ) (pounds_bought : ℤ) (change_received : ℤ)
variable (total_cost : ℤ) (amount_paid : ℤ)

-- Conditions
def price_per_pound := 7  -- A pound of steak costs $7
def pounds_bought_value := 2  -- Ken bought 2 pounds of steak
def change_received_value := 6  -- Ken received $6 back after paying

-- Intermediate Calculations
def total_cost_of_steak := pounds_bought_value * price_per_pound  -- Total cost of steak
def amount_paid_calculated := total_cost_of_steak + change_received_value  -- Amount paid based on total cost and change received

-- Problem Statement
theorem ken_paid_20 : (total_cost_of_steak = total_cost) ∧ (amount_paid_calculated = amount_paid) -> amount_paid = 20 :=
by
  intros h
  sorry
end

end NUMINAMATH_GPT_ken_paid_20_l1123_112379


namespace NUMINAMATH_GPT_median_length_of_pieces_is_198_l1123_112360

   -- Define the conditions
   variables (A B C D E : ℕ)
   variables (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E)
   variables (avg_length : (A + B + C + D + E) = 640)
   variables (h_A_max : A ≤ 110)

   -- Statement of the problem (proof stub)
   theorem median_length_of_pieces_is_198 :
     C = 198 :=
   by
   sorry
   
end NUMINAMATH_GPT_median_length_of_pieces_is_198_l1123_112360


namespace NUMINAMATH_GPT_part1_part2_l1123_112332

noncomputable def f (x a : ℝ) : ℝ := |(x - a)| + |(x + 2)|

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≤ 7) : -4 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 * a + 1) : a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1123_112332


namespace NUMINAMATH_GPT_circle_radius_order_l1123_112346

theorem circle_radius_order 
  (rA: ℝ) (rA_condition: rA = 2)
  (CB: ℝ) (CB_condition: CB = 10 * Real.pi)
  (AC: ℝ) (AC_condition: AC = 16 * Real.pi) :
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  rA < rC ∧ rC < rB :=
by 
  sorry

end NUMINAMATH_GPT_circle_radius_order_l1123_112346


namespace NUMINAMATH_GPT_mark_total_theater_spending_l1123_112345

def week1_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week2_cost : ℝ := (2.5 * 6 - 0.1 * (2.5 * 6)) + 3
def week3_cost : ℝ := 4 * 4 + 3
def week4_cost : ℝ := (3 * 5 - 0.2 * (3 * 5)) + 3
def week5_cost : ℝ := (2 * (3.5 * 6 - 0.1 * (3.5 * 6))) + 6
def week6_cost : ℝ := 2 * 7 + 3

def total_cost : ℝ := week1_cost + week2_cost + week3_cost + week4_cost + week5_cost + week6_cost

theorem mark_total_theater_spending : total_cost = 126.30 := sorry

end NUMINAMATH_GPT_mark_total_theater_spending_l1123_112345


namespace NUMINAMATH_GPT_rajas_income_l1123_112301

theorem rajas_income (I : ℝ) 
  (h1 : 0.60 * I + 0.10 * I + 0.10 * I + 5000 = I) : I = 25000 :=
by
  sorry

end NUMINAMATH_GPT_rajas_income_l1123_112301


namespace NUMINAMATH_GPT_algebraic_expression_value_l1123_112356

-- Define the conditions
variables (x y : ℝ)
-- Condition 1: x - y = 5
def cond1 : Prop := x - y = 5
-- Condition 2: xy = -3
def cond2 : Prop := x * y = -3

-- Define the statement to be proved
theorem algebraic_expression_value :
  cond1 x y → cond2 x y → x^2 * y - x * y^2 = -15 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1123_112356


namespace NUMINAMATH_GPT_range_of_m_l1123_112313

theorem range_of_m (x m : ℝ) (h1 : 2 * x - m ≤ 3) (h2 : -5 < x) (h3 : x < 4) :
  ∃ m, ∀ (x : ℝ), (-5 < x ∧ x < 4) → (2 * x - m ≤ 3) ↔ (m ≥ 5) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1123_112313


namespace NUMINAMATH_GPT_bicycles_in_garage_l1123_112304

theorem bicycles_in_garage 
  (B : ℕ) 
  (h1 : 4 * 3 = 12) 
  (h2 : 7 * 1 = 7) 
  (h3 : 2 * B + 12 + 7 = 25) : 
  B = 3 := 
by
  sorry

end NUMINAMATH_GPT_bicycles_in_garage_l1123_112304


namespace NUMINAMATH_GPT_problem1_problem2_l1123_112390

variable (a b : ℝ)

theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  1/a + 1/(b+1) ≥ 4/5 := by
  sorry

theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  4/(a*b) + a/b ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1123_112390


namespace NUMINAMATH_GPT_monthly_expenses_last_month_l1123_112362

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end NUMINAMATH_GPT_monthly_expenses_last_month_l1123_112362


namespace NUMINAMATH_GPT_no_integer_b_two_distinct_roots_l1123_112381

theorem no_integer_b_two_distinct_roots :
  ∀ b : ℤ, ¬ ∃ x y : ℤ, x ≠ y ∧ (x^4 + 4 * x^3 + b * x^2 + 16 * x + 8 = 0) ∧ (y^4 + 4 * y^3 + b * y^2 + 16 * y + 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_b_two_distinct_roots_l1123_112381


namespace NUMINAMATH_GPT_candy_mixture_l1123_112361

theorem candy_mixture (x : ℝ) (h1 : x * 3 + 64 * 2 = (x + 64) * 2.2) : x + 64 = 80 :=
by sorry

end NUMINAMATH_GPT_candy_mixture_l1123_112361


namespace NUMINAMATH_GPT_evaluate_expression_at_neg3_l1123_112383

theorem evaluate_expression_at_neg3 : (5 + (-3) * (5 + (-3)) - 5^2) / ((-3) - 5 + (-3)^2) = -26 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_neg3_l1123_112383


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1123_112309

theorem perfect_square_trinomial (x : ℝ) : 
  let a := x
  let b := 1 / 2
  2 * a * b = x :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1123_112309


namespace NUMINAMATH_GPT_tan_eq_sin3x_solutions_l1123_112369

open Real

theorem tan_eq_sin3x_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ tan x = sin (3 * x)) ∧ s.card = 6 :=
sorry

end NUMINAMATH_GPT_tan_eq_sin3x_solutions_l1123_112369


namespace NUMINAMATH_GPT_increased_speed_l1123_112324

theorem increased_speed
  (d : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) 
  (h1 : d = 2) 
  (h2 : s1 = 2) 
  (h3 : t1 = 1)
  (h4 : t2 = 2 / 3)
  (h5 : s1 * t1 = d)
  (h6 : s2 * t2 = d) :
  s2 - s1 = 1 := 
sorry

end NUMINAMATH_GPT_increased_speed_l1123_112324


namespace NUMINAMATH_GPT_gcd_a_b_eq_one_l1123_112338

def a : ℕ := 123^2 + 235^2 + 347^2
def b : ℕ := 122^2 + 234^2 + 348^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_a_b_eq_one_l1123_112338


namespace NUMINAMATH_GPT_center_of_circle_l1123_112384

theorem center_of_circle : ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → (1, 1) = (1, 1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_center_of_circle_l1123_112384


namespace NUMINAMATH_GPT_greatest_possible_sum_l1123_112303

noncomputable def eight_products_sum_max : ℕ :=
  let a := 3
  let b := 4
  let c := 5
  let d := 8
  let e := 6
  let f := 7
  7 * (c + d) * (e + f)

theorem greatest_possible_sum (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) :
  eight_products_sum_max = 1183 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_sum_l1123_112303


namespace NUMINAMATH_GPT_group_size_l1123_112371

theorem group_size (n : ℕ) (T : ℕ) (h1 : T = 14 * n) (h2 : T + 32 = 16 * (n + 1)) : n = 8 :=
by
  -- We skip the proof steps
  sorry

end NUMINAMATH_GPT_group_size_l1123_112371


namespace NUMINAMATH_GPT_relationship_m_n_l1123_112382

variables {a b : ℝ}

theorem relationship_m_n (h1 : |a| ≠ |b|) (m : ℝ) (n : ℝ)
  (hm : m = (|a| - |b|) / |a - b|)
  (hn : n = (|a| + |b|) / |a + b|) :
  m ≤ n :=
by sorry

end NUMINAMATH_GPT_relationship_m_n_l1123_112382


namespace NUMINAMATH_GPT_polar_equation_is_circle_of_radius_five_l1123_112374

theorem polar_equation_is_circle_of_radius_five :
  ∀ θ : ℝ, (3 * Real.sin θ + 4 * Real.cos θ) ^ 2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_polar_equation_is_circle_of_radius_five_l1123_112374


namespace NUMINAMATH_GPT_total_students_in_class_l1123_112373

-- No need for noncomputable def here as we're dealing with basic arithmetic

theorem total_students_in_class (jellybeans_total jellybeans_left boys_girls_diff : ℕ)
  (girls boys students : ℕ) :
  jellybeans_total = 450 →
  jellybeans_left = 10 →
  boys_girls_diff = 3 →
  boys = girls + boys_girls_diff →
  students = girls + boys →
  (girls * girls) + (boys * boys) = jellybeans_total - jellybeans_left →
  students = 29 := 
by
  intro h_total h_left h_diff h_boys h_students h_distribution
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1123_112373


namespace NUMINAMATH_GPT_total_valid_votes_l1123_112350

theorem total_valid_votes (V : ℕ) (h1 : 0.70 * (V: ℝ) - 0.30 * (V: ℝ) = 184) : V = 460 :=
by sorry

end NUMINAMATH_GPT_total_valid_votes_l1123_112350


namespace NUMINAMATH_GPT_tallest_is_Justina_l1123_112312

variable (H G I J K : ℝ)

axiom height_conditions1 : H < G
axiom height_conditions2 : G < J
axiom height_conditions3 : K < I
axiom height_conditions4 : I < G

theorem tallest_is_Justina : J > G ∧ J > H ∧ J > I ∧ J > K :=
by
  sorry

end NUMINAMATH_GPT_tallest_is_Justina_l1123_112312


namespace NUMINAMATH_GPT_price_after_discount_l1123_112394

-- Define the original price and discount
def original_price : ℕ := 76
def discount : ℕ := 25

-- The main proof statement
theorem price_after_discount : original_price - discount = 51 := by
  sorry

end NUMINAMATH_GPT_price_after_discount_l1123_112394


namespace NUMINAMATH_GPT_number_of_triangles_l1123_112318

open Nat

/-- Each side of a square is divided into 8 equal parts, and using the divisions
as vertices (not including the vertices of the square), the number of different 
triangles that can be obtained is 3136. -/
theorem number_of_triangles (n : ℕ := 7) :
  (n * 4).choose 3 - 4 * n.choose 3 = 3136 := 
sorry

end NUMINAMATH_GPT_number_of_triangles_l1123_112318


namespace NUMINAMATH_GPT_interest_earned_l1123_112328

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) := P * (1 + r) ^ t

theorem interest_earned :
  let P := 2000
  let r := 0.05
  let t := 5
  let A := compound_interest P r t
  A - P = 552.56 :=
by
  sorry

end NUMINAMATH_GPT_interest_earned_l1123_112328


namespace NUMINAMATH_GPT_margie_change_l1123_112343

theorem margie_change (n_sold n_cost n_paid : ℕ) (h1 : n_sold = 3) (h2 : n_cost = 50) (h3 : n_paid = 500) : 
  n_paid - (n_sold * n_cost) = 350 := by
  sorry

end NUMINAMATH_GPT_margie_change_l1123_112343


namespace NUMINAMATH_GPT_straight_line_cannot_intersect_all_segments_l1123_112308

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end NUMINAMATH_GPT_straight_line_cannot_intersect_all_segments_l1123_112308


namespace NUMINAMATH_GPT_simplify_fraction_l1123_112315

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end NUMINAMATH_GPT_simplify_fraction_l1123_112315


namespace NUMINAMATH_GPT_parabola_properties_l1123_112375

theorem parabola_properties (m : ℝ) :
  (∀ P : ℝ × ℝ, P = (m, 1) ∧ (P.1 ^ 2 = 4 * P.2) →
    ((∃ y : ℝ, y = -1) ∧ (dist P (0, 1) = 2))) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l1123_112375


namespace NUMINAMATH_GPT_gcd_of_polynomial_l1123_112397

theorem gcd_of_polynomial (a : ℤ) (h : 720 ∣ a) : Int.gcd (a^2 + 8*a + 18) (a + 6) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_polynomial_l1123_112397


namespace NUMINAMATH_GPT_number_of_lines_with_negative_reciprocal_intercepts_l1123_112317

-- Define the point (-2, 4)
def point : ℝ × ℝ := (-2, 4)

-- Define the condition that intercepts are negative reciprocals
def are_negative_reciprocals (a b : ℝ) : Prop :=
  a * b = -1

-- Define the proof problem: 
-- Number of lines through point (-2, 4) with intercepts negative reciprocals of each other
theorem number_of_lines_with_negative_reciprocal_intercepts :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ (a b : ℝ), are_negative_reciprocals a b →
  (∃ m k : ℝ, (k * (-2) + m = 4) ∧ ((m ⁻¹ = a ∧ k = b) ∨ (k = a ∧ m ⁻¹ = b))) :=
sorry

end NUMINAMATH_GPT_number_of_lines_with_negative_reciprocal_intercepts_l1123_112317


namespace NUMINAMATH_GPT_lockers_remaining_open_l1123_112321

-- Define the number of lockers and students
def num_lockers : ℕ := 1000

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to count perfect squares up to a given number
def count_perfect_squares_up_to (n : ℕ) : ℕ :=
  Nat.sqrt n

-- Theorem statement
theorem lockers_remaining_open : 
  count_perfect_squares_up_to num_lockers = 31 :=
by
  -- Proof left out because it's not necessary to provide
  sorry

end NUMINAMATH_GPT_lockers_remaining_open_l1123_112321


namespace NUMINAMATH_GPT_simplify_expression_l1123_112366

theorem simplify_expression (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1123_112366


namespace NUMINAMATH_GPT_least_xy_l1123_112399

noncomputable def condition (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 1 / 7)

theorem least_xy (x y : ℕ) (h : condition x y) : x * y = 98 :=
sorry

end NUMINAMATH_GPT_least_xy_l1123_112399
