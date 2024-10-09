import Mathlib

namespace common_point_of_arithmetic_progression_lines_l1984_198422

theorem common_point_of_arithmetic_progression_lines 
  (a d : ℝ) 
  (h₁ : a ≠ 0)
  (h_d_ne_zero : d ≠ 0) 
  (h₃ : ∀ (x y : ℝ), (x = -1 ∧ y = 1) ↔ (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = (a-2*d))) :
  (∀ (x y : ℝ), (a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = a-2*d) → x = -1 ∧ y = 1) :=
by 
  sorry

end common_point_of_arithmetic_progression_lines_l1984_198422


namespace radius_ratio_ge_sqrt2plus1_l1984_198446

theorem radius_ratio_ge_sqrt2plus1 (r R a h : ℝ) (h1 : 2 * a ≠ 0) (h2 : h ≠ 0) 
  (hr : r = a * h / (a + Real.sqrt (a ^ 2 + h ^ 2)))
  (hR : R = (2 * a ^ 2 + h ^ 2) / (2 * h)) : 
  R / r ≥ 1 + Real.sqrt 2 := 
sorry

end radius_ratio_ge_sqrt2plus1_l1984_198446


namespace fourth_person_knight_l1984_198466

-- Let P1, P2, P3, and P4 be the statements made by the four people respectively.
def P1 := ∀ x y z w : Prop, x = y ∧ y = z ∧ z = w ∧ w = ¬w
def P2 := ∃! x y z w : Prop, x = true
def P3 := ∀ x y z w : Prop, (x = true ∧ y = true ∧ z = false) ∨ (x = true ∧ y = false ∧ z = true) ∨ (x = false ∧ y = true ∧ z = true)
def P4 := ∀ x : Prop, x = true → x = true

-- Now let's express the requirement of proving that the fourth person is a knight
theorem fourth_person_knight : P4 := by
  sorry

end fourth_person_knight_l1984_198466


namespace ratio_ashley_mary_l1984_198438

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary_l1984_198438


namespace kyle_delivers_daily_papers_l1984_198481

theorem kyle_delivers_daily_papers (x : ℕ) (h : 6 * x + (x - 10) + 30 = 720) : x = 100 :=
by
  sorry

end kyle_delivers_daily_papers_l1984_198481


namespace parallel_resistors_l1984_198410
noncomputable def resistance_R (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors :
  resistance_R 5 7 3 9 = 315 / 248 :=
by
  sorry

end parallel_resistors_l1984_198410


namespace triangle_coordinates_sum_l1984_198482

noncomputable def coordinates_of_triangle_A (p q : ℚ) : Prop :=
  let B := (12, 19)
  let C := (23, 20)
  let area := ((B.1 * C.2 + C.1 * q + p * B.2) - (B.2 * C.1 + C.2 * p + q * B.1)) / 2 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let median_slope := (q - M.2) / (p - M.1)
  area = 60 ∧ median_slope = 3 

theorem triangle_coordinates_sum (p q : ℚ) 
(h : coordinates_of_triangle_A p q) : p + q = 52 := 
sorry

end triangle_coordinates_sum_l1984_198482


namespace painted_sphere_area_proportionality_l1984_198472

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end painted_sphere_area_proportionality_l1984_198472


namespace smallest_number_of_cubes_filling_box_l1984_198428
open Nat

theorem smallest_number_of_cubes_filling_box (L W D : ℕ) (hL : L = 27) (hW : W = 15) (hD : D = 6) :
  let gcd := 3
  let cubes_along_length := L / gcd
  let cubes_along_width := W / gcd
  let cubes_along_depth := D / gcd
  cubes_along_length * cubes_along_width * cubes_along_depth = 90 :=
by
  sorry

end smallest_number_of_cubes_filling_box_l1984_198428


namespace find_f_1_0_plus_f_2_0_general_form_F_l1984_198474

variable {F : ℝ → ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ a, F a a = a
axiom cond2 : ∀ (k a b : ℝ), F (k * a) (k * b) = k * F a b
axiom cond3 : ∀ (a1 a2 b1 b2 : ℝ), F (a1 + a2) (b1 + b2) = F a1 b1 + F a2 b2
axiom cond4 : ∀ (a b : ℝ), F a b = F b ((a + b) / 2)

-- Proof problem
theorem find_f_1_0_plus_f_2_0 : F 1 0 + F 2 0 = 0 :=
sorry

theorem general_form_F : ∀ (x y : ℝ), F x y = y :=
sorry

end find_f_1_0_plus_f_2_0_general_form_F_l1984_198474


namespace not_divisible_by_121_l1984_198497

theorem not_divisible_by_121 (n : ℤ) : ¬ ∃ t : ℤ, (n^2 + 3*n + 5) = 121 * t ∧ (n^2 - 3*n + 5) = 121 * t := sorry

end not_divisible_by_121_l1984_198497


namespace f_monotonic_m_range_l1984_198459

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonic {x : ℝ} (h : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
  Monotone f :=
sorry

theorem m_range {x : ℝ} (h : x ∈ Set.Ioo 0 (Real.pi / 2)) {m : ℝ} (hm : f x ≥ m * x^2) :
  m ≤ 0 :=
sorry

end f_monotonic_m_range_l1984_198459


namespace wrapping_paper_area_l1984_198407

theorem wrapping_paper_area (length width : ℕ) (h1 : width = 6) (h2 : 2 * (length + width) = 28) : length * width = 48 :=
by
  sorry

end wrapping_paper_area_l1984_198407


namespace initial_production_rate_l1984_198448

theorem initial_production_rate 
  (x : ℝ)
  (h1 : 60 <= (60 * x) / 30 - 60 + 1800)
  (h2 : 60 <= 120)
  (h3 : 30 = (120 / (60 / x + 1))) : x = 20 := by
  sorry

end initial_production_rate_l1984_198448


namespace propane_tank_and_burner_cost_l1984_198431

theorem propane_tank_and_burner_cost
(Total_money: ℝ)
(Sheet_cost: ℝ)
(Rope_cost: ℝ)
(Helium_cost_per_oz: ℝ)
(Lift_per_oz: ℝ)
(Max_height: ℝ)
(ht: Total_money = 200)
(hs: Sheet_cost = 42)
(hr: Rope_cost = 18)
(hh: Helium_cost_per_oz = 1.50)
(hlo: Lift_per_oz = 113)
(hm: Max_height = 9492)
:
(Total_money - (Sheet_cost + Rope_cost) 
 - (Max_height / Lift_per_oz * Helium_cost_per_oz) 
 = 14) :=
by
  sorry

end propane_tank_and_burner_cost_l1984_198431


namespace least_three_digit_multiple_l1984_198492

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l1984_198492


namespace find_a_l1984_198480

theorem find_a (a : ℝ) (h1 : 0 < a)
  (c1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (c2 : ∀ x y : ℝ, x^2 + y^2 + 2 * a * y - 6 = 0)
  (h_chord : (2 * Real.sqrt 3) = 2 * Real.sqrt 3) :
  a = 1 := 
sorry

end find_a_l1984_198480


namespace find_a1_l1984_198402

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  a 6 = 9 →
  a 3 = 3 * a 2 →
  a 1 = -1 :=
by
  sorry

end find_a1_l1984_198402


namespace root_in_interval_l1984_198473

noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a b x = 0 :=
by {
  sorry
}

end root_in_interval_l1984_198473


namespace boat_speed_in_still_water_l1984_198450

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by sorry

end boat_speed_in_still_water_l1984_198450


namespace unit_prices_minimize_cost_l1984_198437

theorem unit_prices (x y : ℕ) (h1 : x + 2 * y = 40) (h2 : 2 * x + 3 * y = 70) :
  x = 20 ∧ y = 10 :=
by {
  sorry -- proof would go here
}

theorem minimize_cost (total_pieces : ℕ) (cost_A cost_B : ℕ) 
  (total_cost : ℕ → ℕ)
  (h3 : total_pieces = 60) 
  (h4 : ∀ m, cost_A * m + cost_B * (total_pieces - m) = total_cost m) 
  (h5 : ∀ m, cost_A * m + cost_B * (total_pieces - m) ≥ 800) 
  (h6 : ∀ m, m ≥ (total_pieces - m) / 2) :
  total_cost 20 = 800 :=
by {
  sorry -- proof would go here
}

end unit_prices_minimize_cost_l1984_198437


namespace car_rental_cost_per_mile_l1984_198444

def daily_rental_rate := 29.0
def total_amount_paid := 46.12
def miles_driven := 214.0

theorem car_rental_cost_per_mile : 
  (total_amount_paid - daily_rental_rate) / miles_driven = 0.08 := 
by
  sorry

end car_rental_cost_per_mile_l1984_198444


namespace hyperbola_condition_l1984_198433

noncomputable def hyperbola_eccentricity_difference (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e_2pi_over_3 := Real.sqrt 3 + 1
  let e_pi_over_3 := (Real.sqrt 3) / 3 + 1
  e_2pi_over_3 - e_pi_over_3

theorem hyperbola_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  hyperbola_eccentricity_difference a b h1 h2 = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_condition_l1984_198433


namespace problem1_problem2_l1984_198413

def A := { x : ℝ | -2 < x ∧ x ≤ 4 }
def B := { x : ℝ | 2 - x < 1 }
def U := ℝ
def complement_B := { x : ℝ | x ≤ 1 }

theorem problem1 : { x : ℝ | 1 < x ∧ x ≤ 4 } = { x : ℝ | x ∈ A ∧ x ∈ B } := 
by sorry

theorem problem2 : { x : ℝ | x ≤ 4 } = { x : ℝ | x ∈ A ∨ x ∈ complement_B } := 
by sorry

end problem1_problem2_l1984_198413


namespace sqrt_23_range_l1984_198485

theorem sqrt_23_range : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end sqrt_23_range_l1984_198485


namespace ratio_of_part_to_whole_l1984_198447

theorem ratio_of_part_to_whole (N : ℝ) (P : ℝ) (h1 : (1/4) * (2/5) * N = 17) (h2 : 0.40 * N = 204) :
  P = (2/5) * N → P / N = 2 / 5 :=
by
  intro h3
  sorry

end ratio_of_part_to_whole_l1984_198447


namespace isosceles_triangle_side_length_l1984_198426

theorem isosceles_triangle_side_length (base : ℝ) (area : ℝ) (congruent_side : ℝ) 
  (h_base : base = 30) (h_area : area = 60) : congruent_side = Real.sqrt 241 :=
by 
  sorry

end isosceles_triangle_side_length_l1984_198426


namespace find_x1_l1984_198401

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/4) 
  : x1 = 3/4 := 
sorry

end find_x1_l1984_198401


namespace fraction_calculation_l1984_198457

theorem fraction_calculation :
  (1 / 4) * (1 / 3) * (1 / 6) * 144 + (1 / 2) = (5 / 2) :=
by
  sorry

end fraction_calculation_l1984_198457


namespace new_cost_percentage_l1984_198441

variables (t c a x : ℝ) (n : ℕ)

def original_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * c * (a * x) ^ n

def new_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * (2 * c) * ((2 * a) * x) ^ (n + 2)

theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  new_cost t c a x n = 2^(n+1) * original_cost t c a x n * x^2 :=
by
  sorry

end new_cost_percentage_l1984_198441


namespace average_speed_of_horse_l1984_198468

/-- Definitions of the conditions given in the problem. --/
def pony_speed : ℕ := 20
def pony_head_start_hours : ℕ := 3
def horse_chase_hours : ℕ := 4

-- Define a proof problem for the average speed of the horse.
theorem average_speed_of_horse : (pony_head_start_hours * pony_speed + horse_chase_hours * pony_speed) / horse_chase_hours = 35 := by
  -- Setting up the necessary distances
  let pony_head_start_distance := pony_head_start_hours * pony_speed
  let pony_additional_distance := horse_chase_hours * pony_speed
  let total_pony_distance := pony_head_start_distance + pony_additional_distance
  -- Asserting the average speed of the horse
  let horse_average_speed := total_pony_distance / horse_chase_hours
  show horse_average_speed = 35
  sorry

end average_speed_of_horse_l1984_198468


namespace sum_of_reciprocals_l1984_198464

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : (1 / x + 1 / y) = (1 / 3) :=
by
  sorry

end sum_of_reciprocals_l1984_198464


namespace problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l1984_198454

-- Problem 1: Monotonicity of f(x) = 1 - 3x on ℝ
theorem problem1_monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → (1 - 3 * x1) > (1 - 3 * x2) :=
by
  -- Proof (skipped)
  sorry

-- Problem 2: Monotonicity of g(x) = 1/x + 2 on (0, ∞) and (-∞, 0)
theorem problem2_monotonic_decreasing_pos : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

theorem problem2_monotonic_decreasing_neg : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

end problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l1984_198454


namespace sum_of_solutions_l1984_198484

theorem sum_of_solutions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (∃ x₁ x₂ : ℝ, (3 * x₁ + 2) * (x₁ - 4) = 0 ∧ (3 * x₂ + 2) * (x₂ - 4) = 0 ∧
  x₁ ≠ 1 ∧ x₁ ≠ -1 ∧ x₂ ≠ 1 ∧ x₂ ≠ -1 ∧ x₁ + x₂ = 10 / 3) :=
sorry

end sum_of_solutions_l1984_198484


namespace original_number_not_800_l1984_198416

theorem original_number_not_800 (x : ℕ) (h : 10 * x = x + 720) : x ≠ 800 :=
by {
  sorry
}

end original_number_not_800_l1984_198416


namespace solve_for_y_l1984_198477

theorem solve_for_y (y : ℝ) (h : (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1) : y = 4.5 :=
by
  -- Proof goes here
  sorry

end solve_for_y_l1984_198477


namespace therapist_charge_difference_l1984_198430

theorem therapist_charge_difference :
  ∃ F A : ℝ, F + 4 * A = 350 ∧ F + A = 161 ∧ F - A = 35 :=
by {
  -- Placeholder for the actual proof.
  sorry
}

end therapist_charge_difference_l1984_198430


namespace junior_score_l1984_198483

theorem junior_score (total_students : ℕ) (juniors_percentage : ℝ) (seniors_percentage : ℝ)
  (class_average : ℝ) (senior_average : ℝ) (juniors_same_score : Prop) 
  (h1 : juniors_percentage = 0.2) (h2 : seniors_percentage = 0.8)
  (h3 : class_average = 85) (h4 : senior_average = 84) : 
  ∃ junior_score : ℝ, juniors_same_score → junior_score = 89 :=
by
  sorry

end junior_score_l1984_198483


namespace cylinder_lateral_surface_area_l1984_198490
noncomputable def lateralSurfaceArea (S : ℝ) : ℝ :=
  let l := Real.sqrt S
  let d := l
  let r := d / 2
  let h := l
  2 * Real.pi * r * h

theorem cylinder_lateral_surface_area (S : ℝ) (hS : S ≥ 0) : 
  lateralSurfaceArea S = Real.pi * S := by
  sorry

end cylinder_lateral_surface_area_l1984_198490


namespace solve_system_a_solve_system_b_l1984_198494

-- For problem (a):
theorem solve_system_a (x y : ℝ) :
  (x + y + x * y = 5) ∧ (x * y * (x + y) = 6) → 
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := 
by
  sorry

-- For problem (b):
theorem solve_system_b (x y : ℝ) :
  (x^3 + y^3 + 2 * x * y = 4) ∧ (x^2 - x * y + y^2 = 1) → 
  (x = 1 ∧ y = 1) := 
by
  sorry

end solve_system_a_solve_system_b_l1984_198494


namespace no_sum_of_three_squares_l1984_198418

theorem no_sum_of_three_squares (n : ℤ) (h : n % 8 = 7) : 
  ¬ ∃ a b c : ℤ, a^2 + b^2 + c^2 = n :=
by 
sorry

end no_sum_of_three_squares_l1984_198418


namespace percentage_more_likely_to_lose_both_l1984_198475

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l1984_198475


namespace problem_solution_l1984_198427

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.rpow 3 (1 / 3)
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem problem_solution : c < a ∧ a < b := 
by
  sorry

end problem_solution_l1984_198427


namespace d_not_unique_minimum_l1984_198404

noncomputable def d (n : ℕ) (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (Finset.min' (Finset.univ.image (λ i => abs (x i - t))) sorry + 
  Finset.max' (Finset.univ.image (λ i => abs (x i - t))) sorry) / 2

theorem d_not_unique_minimum (n : ℕ) (x : Fin n → ℝ) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ d n x t1 = d n x t2 := sorry

end d_not_unique_minimum_l1984_198404


namespace tomatoes_left_after_yesterday_correct_l1984_198476

def farmer_initial_tomatoes := 160
def tomatoes_picked_yesterday := 56
def tomatoes_left_after_yesterday : ℕ := farmer_initial_tomatoes - tomatoes_picked_yesterday

theorem tomatoes_left_after_yesterday_correct :
  tomatoes_left_after_yesterday = 104 :=
by
  unfold tomatoes_left_after_yesterday
  -- Proof goes here
  sorry

end tomatoes_left_after_yesterday_correct_l1984_198476


namespace inequality_cubed_l1984_198400

theorem inequality_cubed (a b : ℝ) (h : a < b ∧ b < 0) : a^3 ≤ b^3 :=
sorry

end inequality_cubed_l1984_198400


namespace compare_fractions_l1984_198429

theorem compare_fractions (a b m : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end compare_fractions_l1984_198429


namespace sara_total_payment_l1984_198412

structure DecorationCosts where
  balloons: ℝ
  tablecloths: ℝ
  streamers: ℝ
  banners: ℝ
  confetti: ℝ
  change_received: ℝ

noncomputable def total_cost (c : DecorationCosts) : ℝ :=
  c.balloons + c.tablecloths + c.streamers + c.banners + c.confetti

noncomputable def amount_given (c : DecorationCosts) : ℝ :=
  total_cost c + c.change_received

theorem sara_total_payment : 
  ∀ (costs : DecorationCosts), 
    costs = ⟨3.50, 18.25, 9.10, 14.65, 7.40, 6.38⟩ →
    amount_given costs = 59.28 :=
by
  intros
  sorry

end sara_total_payment_l1984_198412


namespace find_a_l1984_198470

variable (a b c : ℤ)

theorem find_a (h1 : a + b = 2) (h2 : b + c = 0) (h3 : |c| = 1) : a = 3 ∨ a = 1 := 
sorry

end find_a_l1984_198470


namespace common_ratio_solution_l1984_198436

-- Define the problem condition
def geometric_sum_condition (a1 : ℝ) (q : ℝ) : Prop :=
  (a1 * (1 - q^3)) / (1 - q) = 3 * a1

-- Define the theorem we want to prove
theorem common_ratio_solution (a1 : ℝ) (q : ℝ) (h : geometric_sum_condition a1 q) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_solution_l1984_198436


namespace find_k_l1984_198489

theorem find_k (k : ℝ) 
  (h1 : ∀ (r s : ℝ), r + s = -k ∧ r * s = 8 → (r + 3) + (s + 3) = k) : 
  k = 3 :=
by
  sorry

end find_k_l1984_198489


namespace value_of_x_squared_plus_y_squared_l1984_198425

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : 
  x^2 + y^2 = 4 :=
  sorry

end value_of_x_squared_plus_y_squared_l1984_198425


namespace find_room_length_l1984_198456

variable (width : ℝ) (cost rate : ℝ) (length : ℝ)

theorem find_room_length (h_width : width = 4.75)
  (h_cost : cost = 34200)
  (h_rate : rate = 900)
  (h_area : cost / rate = length * width) :
  length = 8 :=
sorry

end find_room_length_l1984_198456


namespace sum_of_digits_second_smallest_mult_of_lcm_l1984_198414

theorem sum_of_digits_second_smallest_mult_of_lcm :
  let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
  let M := 2 * lcm12345678
  (Nat.digits 10 M).sum = 15 := by
    -- Definitions from the problem statement
    let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
    let M := 2 * lcm12345678
    sorry

end sum_of_digits_second_smallest_mult_of_lcm_l1984_198414


namespace last_two_digits_7_pow_2018_l1984_198495

theorem last_two_digits_7_pow_2018 : 
  (7 ^ 2018) % 100 = 49 := 
sorry

end last_two_digits_7_pow_2018_l1984_198495


namespace rectangle_area_constant_l1984_198478

theorem rectangle_area_constant (d : ℝ) (x : ℝ)
  (length width : ℝ)
  (h_length : length = 5 * x)
  (h_width : width = 4 * x)
  (h_diagonal : d = Real.sqrt (length ^ 2 + width ^ 2)) :
  (exists k : ℝ, k = 20 / 41 ∧ (length * width = k * d ^ 2)) :=
by
  use 20 / 41
  sorry

end rectangle_area_constant_l1984_198478


namespace eventually_constant_sequence_a_floor_l1984_198493

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_b (n : ℕ) : ℝ := sorry

axiom base_conditions : 
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (∀ n, sequence_a (n + 1) * sequence_b n = 1 + sequence_a n + sequence_a n * sequence_b n) ∧
  (∀ n, sequence_b (n + 1) * sequence_a n = 1 + sequence_b n + sequence_a n * sequence_b n)

theorem eventually_constant_sequence_a_floor:
  (∃ N, ∀ n ≥ N, 4 < sequence_a n ∧ sequence_a n < 5) →
  (∃ N, ∀ n ≥ N, Int.floor (sequence_a n) = 4) :=
sorry

end eventually_constant_sequence_a_floor_l1984_198493


namespace rice_flour_weights_l1984_198423

variables (r f : ℝ)

theorem rice_flour_weights :
  (8 * r + 6 * f = 550) ∧ (4 * r + 7 * f = 375) → (r = 50) ∧ (f = 25) :=
by
  intro h
  sorry

end rice_flour_weights_l1984_198423


namespace discount_difference_l1984_198435

def original_amount : ℚ := 20000
def single_discount_rate : ℚ := 0.30
def first_discount_rate : ℚ := 0.25
def second_discount_rate : ℚ := 0.05

theorem discount_difference :
  (original_amount * (1 - single_discount_rate)) - (original_amount * (1 - first_discount_rate) * (1 - second_discount_rate)) = 250 := by
  sorry

end discount_difference_l1984_198435


namespace cheryl_material_used_l1984_198449

theorem cheryl_material_used :
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  used = (52 / 247 : ℚ) :=
by
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  have : used = (52 / 247 : ℚ) := sorry
  exact this

end cheryl_material_used_l1984_198449


namespace product_of_variables_l1984_198499

theorem product_of_variables (a b c d : ℚ)
  (h1 : 4 * a + 5 * b + 7 * c + 9 * d = 56)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d) :
  a * b * c * d = 58653 / 10716361 := 
sorry

end product_of_variables_l1984_198499


namespace Jacob_eats_more_calories_than_planned_l1984_198455

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l1984_198455


namespace regression_line_l1984_198496

theorem regression_line (x y : ℝ) (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 6.5)
  (h_point : (x1, y1) = (2, 3)) :
  (y - y1) = m * (x - x1) ↔ y = 6.5 * x - 10 :=
by
  sorry

end regression_line_l1984_198496


namespace positive_real_solution_unique_l1984_198486

theorem positive_real_solution_unique :
  (∃! x : ℝ, 0 < x ∧ x^12 + 5 * x^11 - 3 * x^10 + 2000 * x^9 - 1500 * x^8 = 0) :=
sorry

end positive_real_solution_unique_l1984_198486


namespace arithmetic_sequence_example_l1984_198443

theorem arithmetic_sequence_example (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h2 : a 2 = 2) (h14 : a 14 = 18) : a 8 = 10 :=
by
  sorry

end arithmetic_sequence_example_l1984_198443


namespace rosy_fish_is_twelve_l1984_198453

/-- Let lilly_fish be the number of fish Lilly has. -/
def lilly_fish : ℕ := 10

/-- Let total_fish be the total number of fish Lilly and Rosy have together. -/
def total_fish : ℕ := 22

/-- Prove that the number of fish Rosy has is equal to 12. -/
theorem rosy_fish_is_twelve : (total_fish - lilly_fish) = 12 :=
by sorry

end rosy_fish_is_twelve_l1984_198453


namespace percentage_managers_decrease_l1984_198421

theorem percentage_managers_decrease
  (employees : ℕ)
  (initial_percentage : ℝ)
  (managers_leave : ℝ)
  (new_percentage : ℝ)
  (h1 : employees = 200)
  (h2 : initial_percentage = 99)
  (h3 : managers_leave = 100)
  (h4 : new_percentage = 98) :
  ((initial_percentage / 100 * employees - managers_leave) / (employees - managers_leave) * 100 = new_percentage) :=
by
  -- To be proven
  sorry

end percentage_managers_decrease_l1984_198421


namespace symmetric_circle_l1984_198462

theorem symmetric_circle (x y : ℝ) :
  let C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }
  let L := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  ∃ C' : ℝ × ℝ → Prop, (∀ p, C' p ↔ (p.1)^2 + (p.2)^2 = 1) :=
sorry

end symmetric_circle_l1984_198462


namespace most_likely_number_of_cars_l1984_198451

theorem most_likely_number_of_cars 
    (cars_in_first_10_seconds : ℕ := 6) 
    (time_for_first_10_seconds : ℕ := 10) 
    (total_time_seconds : ℕ := 165) 
    (constant_speed : Prop := true) : 
    ∃ (num_cars : ℕ), num_cars = 100 :=
by
  sorry

end most_likely_number_of_cars_l1984_198451


namespace usage_difference_correct_l1984_198440

def computerUsageLastWeek : ℕ := 91

def computerUsageThisWeek : ℕ :=
  let first4days := 4 * 8
  let last3days := 3 * 10
  first4days + last3days

def computerUsageFollowingWeek : ℕ :=
  let weekdays := 5 * (5 + 3)
  let weekends := 2 * 12
  weekdays + weekends

def differenceThisWeek : ℕ := computerUsageLastWeek - computerUsageThisWeek
def differenceFollowingWeek : ℕ := computerUsageLastWeek - computerUsageFollowingWeek

theorem usage_difference_correct :
  differenceThisWeek = 29 ∧ differenceFollowingWeek = 27 := by
  sorry

end usage_difference_correct_l1984_198440


namespace total_weight_puffy_muffy_l1984_198463

def scruffy_weight : ℕ := 12
def muffy_weight : ℕ := scruffy_weight - 3
def puffy_weight : ℕ := muffy_weight + 5

theorem total_weight_puffy_muffy : puffy_weight + muffy_weight = 23 := 
by
  sorry

end total_weight_puffy_muffy_l1984_198463


namespace debra_probability_theorem_l1984_198460

-- Define event for Debra's coin flipping game starting with "HTT"
def debra_coin_game_event : Prop := 
  let heads_probability : ℝ := 0.5
  let tails_probability : ℝ := 0.5
  let initial_prob : ℝ := heads_probability * tails_probability * tails_probability
  let Q : ℝ := 1 / 3  -- the computed probability of getting HH after HTT
  let final_probability : ℝ := initial_prob * Q
  final_probability = 1 / 24

-- The theorem statement
theorem debra_probability_theorem :
  debra_coin_game_event := 
by
  sorry

end debra_probability_theorem_l1984_198460


namespace ned_shirts_problem_l1984_198479

theorem ned_shirts_problem
  (long_sleeve_shirts : ℕ)
  (total_shirts_washed : ℕ)
  (total_shirts_had : ℕ)
  (h1 : long_sleeve_shirts = 21)
  (h2 : total_shirts_washed = 29)
  (h3 : total_shirts_had = total_shirts_washed + 1) :
  ∃ short_sleeve_shirts : ℕ, short_sleeve_shirts = total_shirts_had - total_shirts_washed - 1 :=
by
  sorry

end ned_shirts_problem_l1984_198479


namespace plant_species_numbering_impossible_l1984_198424

theorem plant_species_numbering_impossible :
  ∀ (n m : ℕ), 2 ≤ n ∨ n ≤ 20000 ∧ 2 ≤ m ∨ m ≤ 20000 ∧ n ≠ m → 
  ∃ x y : ℕ, 2 ≤ x ∨ x ≤ 20000 ∧ 2 ≤ y ∨ y ≤ 20000 ∧ x ≠ y ∧
  (∀ k : ℕ, gcd x k = gcd n k ∧ gcd y k = gcd m k) :=
  by sorry

end plant_species_numbering_impossible_l1984_198424


namespace math_problem_l1984_198420

-- Define the mixed numbers as fractions
def mixed_3_1_5 := 16 / 5 -- 3 + 1/5 = 16/5
def mixed_4_1_2 := 9 / 2  -- 4 + 1/2 = 9/2
def mixed_2_3_4 := 11 / 4 -- 2 + 3/4 = 11/4
def mixed_1_2_3 := 5 / 3  -- 1 + 2/3 = 5/3

-- Define the main expression
def main_expr := 53 * (mixed_3_1_5 - mixed_4_1_2) / (mixed_2_3_4 + mixed_1_2_3)

-- Define the expected answer in its fractional form
def expected_result := -78 / 5

-- The theorem to prove the main expression equals the expected mixed number
theorem math_problem : main_expr = expected_result :=
by sorry

end math_problem_l1984_198420


namespace find_a_l1984_198409

theorem find_a (a : ℝ) 
  (h1 : a < 0)
  (h2 : a < 1/3)
  (h3 : -2 * a + (1 - 3 * a) = 6) : 
  a = -1 := 
by 
  sorry

end find_a_l1984_198409


namespace imaginary_part_of_z_l1984_198458

theorem imaginary_part_of_z (z : ℂ) (h : (z / (1 - I)) = (3 + I)) : z.im = -2 :=
sorry

end imaginary_part_of_z_l1984_198458


namespace problem_lean_statement_l1984_198411

theorem problem_lean_statement (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 * x ^ 2 + 5 * x + 3)
  (h2 : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 0 :=
by sorry

end problem_lean_statement_l1984_198411


namespace equal_roots_h_l1984_198406

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + (h / 3) = 0) -> h = 4 :=
by 
  sorry

end equal_roots_h_l1984_198406


namespace solve_for_x_l1984_198417

def star (a b : ℝ) : ℝ := 3 * a - b

theorem solve_for_x :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solve_for_x_l1984_198417


namespace min_value_expr_l1984_198471

noncomputable def min_value (a b c : ℝ) := 4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c)

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  min_value a b c ≥ 8 / Real.sqrt 3 :=
by
  sorry

end min_value_expr_l1984_198471


namespace negation_proposition_real_l1984_198461

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l1984_198461


namespace frank_initial_candy_l1984_198405

theorem frank_initial_candy (n : ℕ) (h1 : n = 21) (h2 : 2 > 0) :
  2 * n = 42 :=
by
  --* Use the hypotheses to establish the required proof
  sorry

end frank_initial_candy_l1984_198405


namespace least_value_sum_l1984_198403

theorem least_value_sum (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z = 92 :=
sorry

end least_value_sum_l1984_198403


namespace total_peanuts_is_388_l1984_198452

def peanuts_total (jose kenya marcos : ℕ) : ℕ :=
  jose + kenya + marcos

theorem total_peanuts_is_388 :
  ∀ (jose kenya marcos : ℕ),
    (jose = 85) →
    (kenya = jose + 48) →
    (marcos = kenya + 37) →
    peanuts_total jose kenya marcos = 388 := 
by
  intros jose kenya marcos h_jose h_kenya h_marcos
  sorry

end total_peanuts_is_388_l1984_198452


namespace maria_savings_l1984_198469

variable (S : ℝ) -- Define S as a real number (amount saved initially)

-- Conditions
def bike_cost : ℝ := 600
def additional_money : ℝ := 250 + 230

-- Theorem statement
theorem maria_savings : S + additional_money = bike_cost → S = 120 :=
by
  intro h -- Assume the hypothesis (condition)
  sorry -- Proof will go here

end maria_savings_l1984_198469


namespace valid_values_l1984_198498

noncomputable def is_defined (x : ℝ) : Prop := 
  (x^2 - 4*x + 3 > 0) ∧ (5 - x^2 > 0)

theorem valid_values (x : ℝ) : 
  is_defined x ↔ (-Real.sqrt 5 < x ∧ x < 1) ∨ (3 < x ∧ x < Real.sqrt 5) := by
  sorry

end valid_values_l1984_198498


namespace solve_inequality_inequality_proof_l1984_198408

-- Problem 1: Solve the inequality |2x+1| - |x-4| > 2
theorem solve_inequality (x : ℝ) :
  (|2 * x + 1| - |x - 4| > 2) ↔ (x < -7 ∨ x > (5/3)) :=
sorry

-- Problem 2: Prove the inequality given a > 0 and b > 0
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
sorry

end solve_inequality_inequality_proof_l1984_198408


namespace length_of_train_l1984_198432

theorem length_of_train :
  ∀ (L : ℝ) (V : ℝ),
  (∀ t p : ℝ, t = 14 → p = 535.7142857142857 → V = L / t) →
  (∀ t p : ℝ, t = 39 → p = 535.7142857142857 → V = (L + p) / t) →
  L = 300 :=
by
  sorry

end length_of_train_l1984_198432


namespace cos_alpha_minus_beta_l1984_198467

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_cos_add : Real.cos (α + β) = -5 / 13)
  (h_tan_sum : Real.tan α + Real.tan β = 3) :
  Real.cos (α - β) = 1 :=
by
  sorry

end cos_alpha_minus_beta_l1984_198467


namespace cayli_combinations_l1984_198415

theorem cayli_combinations (art_choices sports_choices music_choices : ℕ)
  (h1 : art_choices = 2)
  (h2 : sports_choices = 3)
  (h3 : music_choices = 4) :
  art_choices * sports_choices * music_choices = 24 := by
  sorry

end cayli_combinations_l1984_198415


namespace infinite_product_value_l1984_198491

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, 9^(1/(3^n))

theorem infinite_product_value : infinite_product = 27 := 
  by sorry

end infinite_product_value_l1984_198491


namespace increasing_function_iff_l1984_198419

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else (3 - a) * x + (1 / 2) * a

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end increasing_function_iff_l1984_198419


namespace sin_240_eq_neg_sqrt3_div_2_l1984_198445

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l1984_198445


namespace distance_between_parallel_lines_eq_2_l1984_198439

def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_parallel_lines_eq_2 :
  let A := 3
  let B := -4
  let c1 := 2
  let c2 := -8
  let d := (|c1 - c2| / Real.sqrt (A^2 + B^2))
  d = 2 :=
by
  sorry

end distance_between_parallel_lines_eq_2_l1984_198439


namespace lines_connecting_intersections_l1984_198465

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem lines_connecting_intersections (n : ℕ) (h : n ≥ 2) :
  let N := binomial n 2
  binomial N 2 = (n * n * (n - 1) * (n - 1) - 2 * n * (n - 1)) / 8 :=
by {
  sorry
}

end lines_connecting_intersections_l1984_198465


namespace total_books_l1984_198442

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l1984_198442


namespace second_hand_degrees_per_minute_l1984_198434

theorem second_hand_degrees_per_minute (clock_gains_5_minutes_per_hour : true) :
  (360 / 60 = 6) := 
by
  sorry

end second_hand_degrees_per_minute_l1984_198434


namespace student_average_grade_l1984_198487

noncomputable def average_grade_two_years : ℝ :=
  let year1_courses := 6
  let year1_average_grade := 100
  let year1_total_points := year1_courses * year1_average_grade

  let year2_courses := 5
  let year2_average_grade := 40
  let year2_total_points := year2_courses * year2_average_grade

  let total_courses := year1_courses + year2_courses
  let total_points := year1_total_points + year2_total_points

  total_points / total_courses

theorem student_average_grade : average_grade_two_years = 72.7 :=
by
  sorry

end student_average_grade_l1984_198487


namespace sqrt_720_simplified_l1984_198488

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l1984_198488
