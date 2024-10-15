import Mathlib

namespace NUMINAMATH_GPT_stratified_sampling_young_employees_l1233_123361

-- Given conditions
def total_young : Nat := 350
def total_middle_aged : Nat := 500
def total_elderly : Nat := 150
def total_employees : Nat := total_young + total_middle_aged + total_elderly
def representatives_to_select : Nat := 20
def sampling_ratio : Rat := representatives_to_select / (total_employees : Rat)

-- Proof goal
theorem stratified_sampling_young_employees :
  (total_young : Rat) * sampling_ratio = 7 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_young_employees_l1233_123361


namespace NUMINAMATH_GPT_probability_bc_seated_next_l1233_123388

theorem probability_bc_seated_next {P : ℝ} : 
  P = 2 / 3 :=
sorry

end NUMINAMATH_GPT_probability_bc_seated_next_l1233_123388


namespace NUMINAMATH_GPT_find_a_l1233_123335

theorem find_a
  (r1 r2 r3 : ℕ)
  (hr1 : r1 > 2) (hr2 : r2 > 2) (hr3 : r3 > 2)
  (a b c : ℤ)
  (hr : (Polynomial.X - Polynomial.C (r1 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r2 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r3 : ℤ)) = 
         Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
  (h : a + b + c + 1 = -2009) :
  a = -58 := sorry

end NUMINAMATH_GPT_find_a_l1233_123335


namespace NUMINAMATH_GPT_range_of_a_l1233_123380

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, 0 < x → 0 ≤ 2 * x^2 - a * x + a) ↔ 0 < a ∧ a ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1233_123380


namespace NUMINAMATH_GPT_lower_limit_of_range_l1233_123381

theorem lower_limit_of_range (A : Set ℕ) (range_A : ℕ) (h1 : ∀ n ∈ A, Prime n∧ n ≤ 36) (h2 : range_A = 14)
  (h3 : ∃ x, x ∈ A ∧ ¬(∃ y, y ∈ A ∧ y > x)) (h4 : ∃ x, x ∈ A ∧ x = 31): 
  ∃ m, m ∈ A ∧ m = 17 := 
sorry

end NUMINAMATH_GPT_lower_limit_of_range_l1233_123381


namespace NUMINAMATH_GPT_prob_two_girls_l1233_123372

variable (Pboy Pgirl : ℝ)

-- Conditions
def prob_boy : Prop := Pboy = 1 / 2
def prob_girl : Prop := Pgirl = 1 / 2

-- The theorem to be proven
theorem prob_two_girls (h₁ : prob_boy Pboy) (h₂ : prob_girl Pgirl) : (Pgirl * Pgirl) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_prob_two_girls_l1233_123372


namespace NUMINAMATH_GPT_ratio_of_areas_l1233_123356

variable (s : ℝ)
def side_length_square := s
def side_length_longer_rect := 1.2 * s
def side_length_shorter_rect := 0.7 * s
def area_square := s^2
def area_rect := (1.2 * s) * (0.7 * s)

theorem ratio_of_areas (h1 : s > 0) :
  area_rect s / area_square s = 21 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1233_123356


namespace NUMINAMATH_GPT_mean_of_second_set_l1233_123367

theorem mean_of_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 90) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_second_set_l1233_123367


namespace NUMINAMATH_GPT_angle_D_in_pentagon_l1233_123315

theorem angle_D_in_pentagon (A B C D E : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : D = E) (h4 : A + 40 = D) 
  (h5 : A + B + C + D + E = 540) : D = 132 :=
by
  -- Add proof here if needed
  sorry

end NUMINAMATH_GPT_angle_D_in_pentagon_l1233_123315


namespace NUMINAMATH_GPT_probability_heads_l1233_123369

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end NUMINAMATH_GPT_probability_heads_l1233_123369


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1233_123397

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ ((m * 2 + 1 * m * (m - 1)) = 0) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1233_123397


namespace NUMINAMATH_GPT_form_a_set_l1233_123383

def is_definitive (description: String) : Prop :=
  match description with
  | "comparatively small numbers" => False
  | "non-negative even numbers not greater than 10" => True
  | "all triangles" => True
  | "points in the Cartesian coordinate plane with an x-coordinate of zero" => True
  | "tall male students" => False
  | "students under 17 years old in a certain class" => True
  | _ => False

theorem form_a_set :
  is_definitive "comparatively small numbers" = False ∧
  is_definitive "non-negative even numbers not greater than 10" = True ∧
  is_definitive "all triangles" = True ∧
  is_definitive "points in the Cartesian coordinate plane with an x-coordinate of zero" = True ∧
  is_definitive "tall male students" = False ∧
  is_definitive "students under 17 years old in a certain class" = True :=
by
  repeat { split };
  exact sorry

end NUMINAMATH_GPT_form_a_set_l1233_123383


namespace NUMINAMATH_GPT_find_radius_l1233_123318

-- Definition of the conditions
def area_of_sector : ℝ := 10 -- The area of the sector in square centimeters
def arc_length : ℝ := 4     -- The arc length of the sector in centimeters

-- The radius of the circle we want to prove
def radius (r : ℝ) : Prop :=
  (r * 4) / 2 = 10

-- The theorem to be proved
theorem find_radius : ∃ r : ℝ, radius r :=
by
  use 5
  unfold radius
  norm_num

end NUMINAMATH_GPT_find_radius_l1233_123318


namespace NUMINAMATH_GPT_printers_finish_tasks_l1233_123364

theorem printers_finish_tasks :
  ∀ (start_time_1 finish_half_time_1 start_time_2 : ℕ) (half_task_duration full_task_duration second_task_duration : ℕ),
    start_time_1 = 9 * 60 ∧
    finish_half_time_1 = 12 * 60 + 30 ∧
    half_task_duration = finish_half_time_1 - start_time_1 ∧
    full_task_duration = 2 * half_task_duration ∧
    start_time_2 = 13 * 60 ∧
    second_task_duration = 2 * 60 ∧
    start_time_1 + full_task_duration = 4 * 60 ∧
    start_time_2 + second_task_duration = 15 * 60 →
  max (start_time_1 + full_task_duration) (start_time_2 + second_task_duration) = 16 * 60 := 
by
  intros start_time_1 finish_half_time_1 start_time_2 half_task_duration full_task_duration second_task_duration
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  sorry

end NUMINAMATH_GPT_printers_finish_tasks_l1233_123364


namespace NUMINAMATH_GPT_no_solution_abs_eq_2_l1233_123340

theorem no_solution_abs_eq_2 (x : ℝ) :
  |x - 5| = |x + 3| + 2 → false :=
by sorry

end NUMINAMATH_GPT_no_solution_abs_eq_2_l1233_123340


namespace NUMINAMATH_GPT_find_constant_l1233_123351

variable (constant : ℝ)

theorem find_constant (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 1 - 2 * t)
  (h2 : y = constant * t - 2)
  (h3 : x = y) : constant = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_l1233_123351


namespace NUMINAMATH_GPT_find_some_number_l1233_123370

theorem find_some_number (a : ℕ) (h1 : a = 105) (h2 : a^3 = some_number * 35 * 45 * 35) : some_number = 1 := by
  sorry

end NUMINAMATH_GPT_find_some_number_l1233_123370


namespace NUMINAMATH_GPT_candidates_count_l1233_123339

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 := 
sorry

end NUMINAMATH_GPT_candidates_count_l1233_123339


namespace NUMINAMATH_GPT_terminal_sides_positions_l1233_123362

def in_third_quadrant (θ : ℝ) (k : ℤ) : Prop :=
  (180 + k * 360 : ℝ) < θ ∧ θ < (270 + k * 360 : ℝ)

theorem terminal_sides_positions (θ : ℝ) (k : ℤ) :
  in_third_quadrant θ k →
  ((2 * θ > 360 + 2 * k * 360 ∧ 2 * θ < 540 + 2 * k * 360) ∨
   (90 + k * 180 < θ / 2 ∧ θ / 2 < 135 + k * 180) ∨
   (2 * θ = 360 + 2 * k * 360) ∨ (2 * θ = 540 + 2 * k * 360) ∨ 
   (θ / 2 = 90 + k * 180) ∨ (θ / 2 = 135 + k * 180)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_terminal_sides_positions_l1233_123362


namespace NUMINAMATH_GPT_first_instance_height_35_l1233_123379
noncomputable def projectile_height (t : ℝ) : ℝ := -5 * t^2 + 30 * t

theorem first_instance_height_35 {t : ℝ} (h : projectile_height t = 35) :
  t = 3 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_first_instance_height_35_l1233_123379


namespace NUMINAMATH_GPT_range_of_a_l1233_123391

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Icc (-1 : ℝ) 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1233_123391


namespace NUMINAMATH_GPT_three_primes_sum_odd_l1233_123344

theorem three_primes_sum_odd (primes : Finset ℕ) (h_prime : ∀ p ∈ primes, Prime p) :
  primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} →
  (Nat.choose 9 3 / Nat.choose 10 3 : ℚ) = 7 / 10 := by
  -- Let the set of first ten prime numbers.
  -- As per condition, primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  -- Then show that the probability calculation yields 7/10
  sorry

end NUMINAMATH_GPT_three_primes_sum_odd_l1233_123344


namespace NUMINAMATH_GPT_ratio_of_areas_l1233_123320

noncomputable def large_square_side : ℝ := 4
noncomputable def large_square_area : ℝ := large_square_side ^ 2
noncomputable def inscribed_square_side : ℝ := 1  -- As it fits in the definition from the problem description
noncomputable def inscribed_square_area : ℝ := inscribed_square_side ^ 2

theorem ratio_of_areas :
  (inscribed_square_area / large_square_area) = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1233_123320


namespace NUMINAMATH_GPT_eval_expr1_eval_expr2_l1233_123323

theorem eval_expr1 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := 
by
  sorry

theorem eval_expr2 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^2 + b^2) / (a + b) = 5.8 :=
by
  sorry

end NUMINAMATH_GPT_eval_expr1_eval_expr2_l1233_123323


namespace NUMINAMATH_GPT_paint_replacement_fractions_l1233_123371

variables {r b g : ℚ}

/-- Given the initial and replacement intensities and the final intensities of red, blue,
and green paints respectively, prove the fractions of the original amounts of each paint color
that were replaced. -/
theorem paint_replacement_fractions :
  (0.6 * (1 - r) + 0.3 * r = 0.4) ∧
  (0.4 * (1 - b) + 0.15 * b = 0.25) ∧
  (0.25 * (1 - g) + 0.1 * g = 0.18) →
  (r = 2/3) ∧ (b = 3/5) ∧ (g = 7/15) :=
by
  sorry

end NUMINAMATH_GPT_paint_replacement_fractions_l1233_123371


namespace NUMINAMATH_GPT_sum_of_coefficients_l1233_123373

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 : ℤ)
  (h : (1 - 2 * X)^5 = a + a1 * X + a2 * X^2 + a3 * X^3 + a4 * X^4 + a5 * X^5) :
  a1 + a2 + a3 + a4 + a5 = -2 :=
by {
  -- the proof steps would go here
  sorry
}

end NUMINAMATH_GPT_sum_of_coefficients_l1233_123373


namespace NUMINAMATH_GPT_smallest_possible_N_l1233_123321

theorem smallest_possible_N {p q r s t : ℕ} (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (ht: 0 < t) 
  (sum_eq: p + q + r + s + t = 3015) :
  ∃ N, N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ N = 1508 := 
sorry

end NUMINAMATH_GPT_smallest_possible_N_l1233_123321


namespace NUMINAMATH_GPT_discount_correct_l1233_123398

variable {a : ℝ} (discount_percent : ℝ) (profit_percent : ℝ → ℝ)

noncomputable def calc_discount : ℝ :=
  discount_percent

theorem discount_correct :
  (discount_percent / 100) = (33 + 1 / 3) / 100 →
  profit_percent (discount_percent / 100) = (3 / 2) * (discount_percent / 100) →
  a * (1 - discount_percent / 100) * (1 + profit_percent (discount_percent / 100)) = a →
  discount_percent = 33 + 1 / 3 :=
by sorry

end NUMINAMATH_GPT_discount_correct_l1233_123398


namespace NUMINAMATH_GPT_part1_part2_l1233_123346

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (1 - a) * x + (1 - a)

theorem part1 (x : ℝ) : f x 4 ≥ 7 ↔ x ≥ 5 ∨ x ≤ -2 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, -1 < x → f x a ≥ 0) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1233_123346


namespace NUMINAMATH_GPT_total_muffins_correct_l1233_123399

-- Define the conditions
def boys_count := 3
def muffins_per_boy := 12
def girls_count := 2
def muffins_per_girl := 20

-- Define the question and answer
def total_muffins_for_sale : Nat :=
  boys_count * muffins_per_boy + girls_count * muffins_per_girl

theorem total_muffins_correct :
  total_muffins_for_sale = 76 := by
  sorry

end NUMINAMATH_GPT_total_muffins_correct_l1233_123399


namespace NUMINAMATH_GPT_marie_messages_days_l1233_123301

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end NUMINAMATH_GPT_marie_messages_days_l1233_123301


namespace NUMINAMATH_GPT_relationship_between_p_and_q_l1233_123312

variable (x y : ℝ)

def p := x * y ≥ 0
def q := |x + y| = |x| + |y|

theorem relationship_between_p_and_q : (p x y ↔ q x y) :=
sorry

end NUMINAMATH_GPT_relationship_between_p_and_q_l1233_123312


namespace NUMINAMATH_GPT_value_of_a_b_squared_l1233_123347

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a - b = Real.sqrt 2
axiom h2 : a * b = 4

theorem value_of_a_b_squared : (a + b)^2 = 18 := by
   sorry

end NUMINAMATH_GPT_value_of_a_b_squared_l1233_123347


namespace NUMINAMATH_GPT_equation_equivalence_l1233_123359

theorem equation_equivalence (p q : ℝ) (hp₀ : p ≠ 0) (hp₅ : p ≠ 5) (hq₀ : q ≠ 0) (hq₇ : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) → p = 9 * q / (q - 15) :=
by
  sorry

end NUMINAMATH_GPT_equation_equivalence_l1233_123359


namespace NUMINAMATH_GPT_paint_rate_5_l1233_123309
noncomputable def rate_per_sq_meter (L : ℝ) (total_cost : ℝ) (B : ℝ) : ℝ :=
  let Area := L * B
  total_cost / Area

theorem paint_rate_5 : 
  ∀ (L B total_cost rate : ℝ),
    L = 19.595917942265423 →
    total_cost = 640 →
    L = 3 * B →
    rate = rate_per_sq_meter L total_cost B →
    rate = 5 :=
by
  intros L B total_cost rate hL hC hR hRate
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_paint_rate_5_l1233_123309


namespace NUMINAMATH_GPT_find_angle_x_l1233_123331

noncomputable def angle_x (angle_ABC angle_ACB angle_CDE : ℝ) : ℝ :=
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  180 - angle_AED

theorem find_angle_x (angle_ABC angle_ACB angle_CDE : ℝ) :
  angle_ABC = 70 → angle_ACB = 90 → angle_CDE = 42 → angle_x angle_ABC angle_ACB angle_CDE = 158 :=
by
  intros hABC hACB hCDE
  simp [angle_x, hABC, hACB, hCDE]
  sorry

end NUMINAMATH_GPT_find_angle_x_l1233_123331


namespace NUMINAMATH_GPT_basketball_team_initial_players_l1233_123341

theorem basketball_team_initial_players
  (n : ℕ)
  (h_average_initial : Real := 190)
  (height_nikolai : Real := 197)
  (height_peter : Real := 181)
  (h_average_new : Real := 188)
  (total_height_initial : Real := h_average_initial * n)
  (total_height_new : Real := total_height_initial - (height_nikolai - height_peter))
  (avg_height_new_calculated : Real := total_height_new / n) :
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_initial_players_l1233_123341


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1233_123395

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1233_123395


namespace NUMINAMATH_GPT_dylan_trip_time_l1233_123354

def total_time_of_trip (d1 d2 d3 v1 v2 v3 b : ℕ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let time_riding := t1 + t2 + t3
  let time_breaks := b * 25 / 60
  time_riding + time_breaks

theorem dylan_trip_time :
  total_time_of_trip 400 150 700 50 40 60 3 = 24.67 :=
by
  unfold total_time_of_trip
  sorry

end NUMINAMATH_GPT_dylan_trip_time_l1233_123354


namespace NUMINAMATH_GPT_charles_travel_time_l1233_123334

theorem charles_travel_time (D S T : ℕ) (hD : D = 6) (hS : S = 3) : T = D / S → T = 2 :=
by
  intros h
  rw [hD, hS] at h
  simp at h
  exact h

end NUMINAMATH_GPT_charles_travel_time_l1233_123334


namespace NUMINAMATH_GPT_max_zeros_in_product_l1233_123368

theorem max_zeros_in_product (a b c : ℕ) (h_sum : a + b + c = 1003) : ∃ N, N = 7 ∧ ∀ p : ℕ, (a * b * c = p) → (∃ k, p = 10^k ∧ k ≤ N) ∧ (∀ k, p = 10^k → k ≤ 7) :=
by
  sorry

end NUMINAMATH_GPT_max_zeros_in_product_l1233_123368


namespace NUMINAMATH_GPT_two_point_questions_count_l1233_123314

theorem two_point_questions_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
sorry

end NUMINAMATH_GPT_two_point_questions_count_l1233_123314


namespace NUMINAMATH_GPT_tom_won_whack_a_mole_l1233_123389

variable (W : ℕ)  -- let W be the number of tickets Tom won playing 'whack a mole'
variable (won_skee_ball : ℕ := 25)  -- Tom won 25 tickets playing 'skee ball'
variable (spent_on_hat : ℕ := 7)  -- Tom spent 7 tickets on a hat
variable (tickets_left : ℕ := 50)  -- Tom has 50 tickets left

theorem tom_won_whack_a_mole :
  W + 25 + 50 = 57 →
  W = 7 :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_tom_won_whack_a_mole_l1233_123389


namespace NUMINAMATH_GPT_problem_statement_l1233_123313

theorem problem_statement :
  (2 * 3 * 4) * (1/2 + 1/3 + 1/4) = 26 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1233_123313


namespace NUMINAMATH_GPT_intersection_A_B_l1233_123307

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1233_123307


namespace NUMINAMATH_GPT_subtraction_example_l1233_123349

theorem subtraction_example : 2 - 3 = -1 := 
by {
  -- We need to prove that 2 - 3 = -1
  -- The proof is to be filled here
  sorry
}

end NUMINAMATH_GPT_subtraction_example_l1233_123349


namespace NUMINAMATH_GPT_minimum_value_expression_l1233_123360

variable (a b c k : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a = k ∧ b = k ∧ c = k)

theorem minimum_value_expression : 
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1233_123360


namespace NUMINAMATH_GPT_company_p_percentage_increase_l1233_123386

theorem company_p_percentage_increase :
  (460 - 400.00000000000006) / 400.00000000000006 * 100 = 15 := 
by
  sorry

end NUMINAMATH_GPT_company_p_percentage_increase_l1233_123386


namespace NUMINAMATH_GPT_simplify_expression_l1233_123394

variable (a : ℝ) (ha : a ≠ 0)

theorem simplify_expression : (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1233_123394


namespace NUMINAMATH_GPT_distance_missouri_to_new_york_by_car_l1233_123363

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end NUMINAMATH_GPT_distance_missouri_to_new_york_by_car_l1233_123363


namespace NUMINAMATH_GPT_ice_cube_count_l1233_123375

theorem ice_cube_count (cubes_per_tray : ℕ) (tray_count : ℕ) (H1: cubes_per_tray = 9) (H2: tray_count = 8) :
  cubes_per_tray * tray_count = 72 :=
by
  sorry

end NUMINAMATH_GPT_ice_cube_count_l1233_123375


namespace NUMINAMATH_GPT_commute_time_x_l1233_123343

theorem commute_time_x (d : ℝ) (walk_speed : ℝ) (train_speed : ℝ) (extra_time : ℝ) (diff_time : ℝ) :
  d = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  diff_time = 10 →
  (diff_time : ℝ) * 60 = (d / walk_speed - (d / train_speed + extra_time / 60)) * 60 →
  extra_time = 15.5 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_x_l1233_123343


namespace NUMINAMATH_GPT_sum_of_numbers_l1233_123352

theorem sum_of_numbers (x : ℝ) (h : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) : 
  x + 2 * x + 4 * x = 105 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1233_123352


namespace NUMINAMATH_GPT_smallest_mu_exists_l1233_123384

theorem smallest_mu_exists (a b c d : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :
  ∃ μ : ℝ, μ = (3 / 2) - (3 / (4 * Real.sqrt 2)) ∧ 
    (a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + μ * b^2 * c + c^2 * d) :=
by
  sorry

end NUMINAMATH_GPT_smallest_mu_exists_l1233_123384


namespace NUMINAMATH_GPT_original_book_pages_l1233_123390

theorem original_book_pages (n k : ℕ) (h1 : (n * (n + 1)) / 2 - (2 * k + 1) = 4979)
: n = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_book_pages_l1233_123390


namespace NUMINAMATH_GPT_line_intersects_circle_and_focus_condition_l1233_123355

variables {x y k : ℝ}

/-- The line l intersects the circle x^2 + y^2 + 2x - 4y + 1 = 0 at points A and B. If the midpoint of the chord AB is the focus of the parabola x^2 = 4y, then prove that the equation of the line l is x - y + 1 = 0. -/
theorem line_intersects_circle_and_focus_condition :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, l x = y) ∧
  (∀ A B : ℝ × ℝ, ∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (0, 1)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧
  x^2 = 4*y ) → 
  (∀ x y : ℝ, x - y + 1 = 0) :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_and_focus_condition_l1233_123355


namespace NUMINAMATH_GPT_Matilda_age_is_35_l1233_123329

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end NUMINAMATH_GPT_Matilda_age_is_35_l1233_123329


namespace NUMINAMATH_GPT_find_number_l1233_123325

theorem find_number (x : ℝ) (h : 0.30 * x - 70 = 20) : x = 300 :=
sorry

end NUMINAMATH_GPT_find_number_l1233_123325


namespace NUMINAMATH_GPT_f_odd_f_decreasing_f_max_min_l1233_123348

noncomputable def f : ℝ → ℝ := sorry

lemma f_add (x y : ℝ) : f (x + y) = f x + f y := sorry
lemma f_neg1 : f (-1) = 2 := sorry
lemma f_positive_less_than_zero {x : ℝ} (hx : x > 0) : f x < 0 := sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_decreasing : ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 := sorry

theorem f_max_min : ∀ (f_max f_min : ℝ),
  f_max = f (-2) ∧ f_min = f 4 ∧
  f (-2) = 4 ∧ f 4 = -8 := sorry

end NUMINAMATH_GPT_f_odd_f_decreasing_f_max_min_l1233_123348


namespace NUMINAMATH_GPT_number_of_correct_calculations_is_one_l1233_123374

/- Given conditions -/
def cond1 (a : ℝ) : Prop := a^2 * a^2 = 2 * a^2
def cond2 (a b : ℝ) : Prop := (a - b)^2 = a^2 - b^2
def cond3 (a : ℝ) : Prop := a^2 + a^3 = a^5
def cond4 (a b : ℝ) : Prop := (-2 * a^2 * b^3)^3 = -6 * a^6 * b^3
def cond5 (a : ℝ) : Prop := (-a^3)^2 / a = a^5

/- Statement to prove the number of correct calculations is 1 -/
theorem number_of_correct_calculations_is_one :
  (¬ (cond1 a)) ∧ (¬ (cond2 a b)) ∧ (¬ (cond3 a)) ∧ (¬ (cond4 a b)) ∧ (cond5 a) → 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_calculations_is_one_l1233_123374


namespace NUMINAMATH_GPT_pythagorean_diagonal_l1233_123317

variable (m : ℕ) (h_m : m ≥ 3)

theorem pythagorean_diagonal (h : (2 * m)^2 + a^2 = (a + 2)^2) :
  (a + 2) = m^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_diagonal_l1233_123317


namespace NUMINAMATH_GPT_product_of_last_two_digits_l1233_123300

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 11) (h2 : ∃ (n : ℕ), 10 * A + B = 6 * n) : A * B = 24 :=
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l1233_123300


namespace NUMINAMATH_GPT_girls_friends_count_l1233_123385

variable (days_in_week : ℕ)
variable (total_friends : ℕ)
variable (boys : ℕ)

axiom H1 : days_in_week = 7
axiom H2 : total_friends = 2 * days_in_week
axiom H3 : boys = 11

theorem girls_friends_count : total_friends - boys = 3 :=
by sorry

end NUMINAMATH_GPT_girls_friends_count_l1233_123385


namespace NUMINAMATH_GPT_cost_per_load_is_25_cents_l1233_123302

def washes_per_bottle := 80
def price_per_bottle_on_sale := 20
def bottles := 2
def total_cost := bottles * price_per_bottle_on_sale -- 2 * 20 = 40
def total_loads := bottles * washes_per_bottle -- 2 * 80 = 160
def cost_per_load_in_dollars := total_cost / total_loads -- 40 / 160 = 0.25
def cost_per_load_in_cents := cost_per_load_in_dollars * 100

theorem cost_per_load_is_25_cents :
  cost_per_load_in_cents = 25 :=
by 
  sorry

end NUMINAMATH_GPT_cost_per_load_is_25_cents_l1233_123302


namespace NUMINAMATH_GPT_quadratic_function_range_l1233_123342

theorem quadratic_function_range (a b c : ℝ) (x y : ℝ) :
  (∀ x, x = -4 → y = a * (-4)^2 + b * (-4) + c → y = 3) ∧
  (∀ x, x = -3 → y = a * (-3)^2 + b * (-3) + c → y = -2) ∧
  (∀ x, x = -2 → y = a * (-2)^2 + b * (-2) + c → y = -5) ∧
  (∀ x, x = -1 → y = a * (-1)^2 + b * (-1) + c → y = -6) ∧
  (∀ x, x = 0 → y = a * 0^2 + b * 0 + c → y = -5) →
  (∀ x, x < -2 → y > -5) :=
sorry

end NUMINAMATH_GPT_quadratic_function_range_l1233_123342


namespace NUMINAMATH_GPT_max_value_g_l1233_123338

def g : ℕ → ℕ
| n => if n < 7 then n + 8 else g (n - 3)

theorem max_value_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 14 := by
  sorry

end NUMINAMATH_GPT_max_value_g_l1233_123338


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1233_123336

def set_M : Set ℝ := {x | -1 < x}
def set_N : Set ℝ := {x | x * (x + 2) ≤ 0}

theorem intersection_of_M_and_N : (set_M ∩ set_N) = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1233_123336


namespace NUMINAMATH_GPT_inequality_am_gm_l1233_123311

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
by sorry

end NUMINAMATH_GPT_inequality_am_gm_l1233_123311


namespace NUMINAMATH_GPT_point_inside_circle_range_l1233_123382

theorem point_inside_circle_range (a : ℝ) : ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) :=
  by
  sorry

end NUMINAMATH_GPT_point_inside_circle_range_l1233_123382


namespace NUMINAMATH_GPT_edge_length_of_prism_l1233_123345

-- Definitions based on conditions
def rectangular_prism_edges : ℕ := 12
def total_edge_length : ℕ := 72

-- Proof problem statement
theorem edge_length_of_prism (num_edges : ℕ) (total_length : ℕ) (h1 : num_edges = rectangular_prism_edges) (h2 : total_length = total_edge_length) : 
  (total_length / num_edges) = 6 :=
by {
  -- The proof is omitted here as instructed
  sorry
}

end NUMINAMATH_GPT_edge_length_of_prism_l1233_123345


namespace NUMINAMATH_GPT_line_through_points_l1233_123393

theorem line_through_points (m b : ℝ)
  (h_slope : m = (-1 - 3) / (-3 - 1))
  (h_point : 3 = m * 1 + b) :
  m + b = 3 :=
sorry

end NUMINAMATH_GPT_line_through_points_l1233_123393


namespace NUMINAMATH_GPT_total_swordfish_catch_l1233_123304

-- Definitions
def S_c : ℝ := 5 - 2
def S_m : ℝ := S_c - 1
def S_a : ℝ := 2 * S_m

def W_s : ℕ := 3  -- Number of sunny days
def W_r : ℕ := 2  -- Number of rainy days

-- Sunny and rainy day adjustments
def Shelly_sunny_catch : ℝ := S_c + 0.20 * S_c
def Sam_sunny_catch : ℝ := S_m + 0.20 * S_m
def Sara_sunny_catch : ℝ := S_a + 0.20 * S_a

def Shelly_rainy_catch : ℝ := S_c - 0.10 * S_c
def Sam_rainy_catch : ℝ := S_m - 0.10 * S_m
def Sara_rainy_catch : ℝ := S_a - 0.10 * S_a

-- Total catch calculations
def Shelly_total_catch : ℝ := W_s * Shelly_sunny_catch + W_r * Shelly_rainy_catch
def Sam_total_catch : ℝ := W_s * Sam_sunny_catch + W_r * Sam_rainy_catch
def Sara_total_catch : ℝ := W_s * Sara_sunny_catch + W_r * Sara_rainy_catch

def Total_catch : ℝ := Shelly_total_catch + Sam_total_catch + Sara_total_catch

-- Proof statement
theorem total_swordfish_catch : ⌊Total_catch⌋ = 48 := 
  by sorry

end NUMINAMATH_GPT_total_swordfish_catch_l1233_123304


namespace NUMINAMATH_GPT_pictures_total_l1233_123366

theorem pictures_total (peter_pics : ℕ) (quincy_extra_pics : ℕ) (randy_pics : ℕ) (quincy_pics : ℕ) (total_pics : ℕ) 
  (h1 : peter_pics = 8)
  (h2 : quincy_extra_pics = 20)
  (h3 : randy_pics = 5)
  (h4 : quincy_pics = peter_pics + quincy_extra_pics)
  (h5 : total_pics = randy_pics + peter_pics + quincy_pics) :
  total_pics = 41 :=
by sorry

end NUMINAMATH_GPT_pictures_total_l1233_123366


namespace NUMINAMATH_GPT_valid_number_of_apples_l1233_123350

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end NUMINAMATH_GPT_valid_number_of_apples_l1233_123350


namespace NUMINAMATH_GPT_Ed_cats_l1233_123376

variable (C F : ℕ)

theorem Ed_cats 
  (h1 : F = 2 * (C + 2))
  (h2 : 2 + C + F = 15) : 
  C = 3 := by 
  sorry

end NUMINAMATH_GPT_Ed_cats_l1233_123376


namespace NUMINAMATH_GPT_problem_find_f_l1233_123378

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_find_f {k : ℝ} :
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) →
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) →
  (∀ x : ℝ, 0 < x → f x = k * x) :=
by
  intro h1 h2
  apply sorry

end NUMINAMATH_GPT_problem_find_f_l1233_123378


namespace NUMINAMATH_GPT_total_amount_l1233_123310

def mark_dollars : ℚ := 5 / 8
def carolyn_dollars : ℚ := 2 / 5
def total_dollars : ℚ := mark_dollars + carolyn_dollars

theorem total_amount : total_dollars = 1.025 := by
  sorry

end NUMINAMATH_GPT_total_amount_l1233_123310


namespace NUMINAMATH_GPT_time_spent_giving_bath_l1233_123332

theorem time_spent_giving_bath
  (total_time : ℕ)
  (walk_time : ℕ)
  (bath_time blowdry_time : ℕ)
  (walk_distance walk_speed : ℤ)
  (walk_distance_eq : walk_distance = 3)
  (walk_speed_eq : walk_speed = 6)
  (total_time_eq : total_time = 60)
  (walk_time_eq : walk_time = (walk_distance * 60 / walk_speed))
  (half_blowdry_time : blowdry_time = bath_time / 2)
  (time_eq : bath_time + blowdry_time = total_time - walk_time)
  : bath_time = 20 := by
  sorry

end NUMINAMATH_GPT_time_spent_giving_bath_l1233_123332


namespace NUMINAMATH_GPT_option_a_option_b_l1233_123358

theorem option_a (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  -- Proof goes here
  sorry

theorem option_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : a * b ≤ (a + b)^2 / 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_option_a_option_b_l1233_123358


namespace NUMINAMATH_GPT_students_doing_hula_hoops_l1233_123337

def number_of_students_jumping_rope : ℕ := 7
def number_of_students_doing_hula_hoops : ℕ := 5 * number_of_students_jumping_rope

theorem students_doing_hula_hoops : number_of_students_doing_hula_hoops = 35 :=
by
  sorry

end NUMINAMATH_GPT_students_doing_hula_hoops_l1233_123337


namespace NUMINAMATH_GPT_determine_abc_l1233_123316

-- Definitions
def parabola_equation (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition (a b c : ℝ) : Prop :=
  ∀ y, parabola_equation a b c y = a * (y + 6)^2 + 3

def point_condition (a b c : ℝ) : Prop :=
  parabola_equation a b c (-6) = 3 ∧ parabola_equation a b c (-4) = 2

-- Proposition to prove
theorem determine_abc : 
  ∃ a b c : ℝ, vertex_condition a b c ∧ point_condition a b c
  ∧ (a + b + c = -25/4) :=
sorry

end NUMINAMATH_GPT_determine_abc_l1233_123316


namespace NUMINAMATH_GPT_min_x_plus_2y_l1233_123319

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_x_plus_2y_l1233_123319


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1233_123353

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) - a * x

theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
  (∀ x < 0, f a x ≥ f a (x + 1)) ∧ (∀ x > 0, f a x ≤ f a (x + 1)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.sin x - Real.cos x + 2 - a * x) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1233_123353


namespace NUMINAMATH_GPT_tree_F_height_l1233_123328

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tree_F_height_l1233_123328


namespace NUMINAMATH_GPT_intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l1233_123392

-- Definitions based on the conditions
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x - a * y + 2 = 0
def perpendicular (a : ℝ) : Prop := a = 0
def parallel (a : ℝ) : Prop := a = 1 ∨ a = -1

-- Theorem 1: Intersection point when a = 0 is (-2, 2)
theorem intersection_point_zero_a_0 :
  ∀ x y : ℝ, l₁ 0 x y → l₂ 0 x y → (x, y) = (-2, 2) := 
by
  sorry

-- Theorem 2: Line l₁ always passes through (0, 2)
theorem l₁_passes_through_0_2 :
  ∀ a : ℝ, l₁ a 0 2 := 
by
  sorry

-- Theorem 3: l₁ is perpendicular to l₂ implies a = 0
theorem l₁_perpendicular_l₂ :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ∀ m n, (a * m + (n / a) = 0)) → (a = 0) :=
by
  sorry

-- Theorem 4: l₁ is parallel to l₂ implies a = 1 or a = -1
theorem l₁_parallel_l₂ :
  ∀ a : ℝ, parallel a → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_zero_a_0_l₁_passes_through_0_2_l₁_perpendicular_l₂_l₁_parallel_l₂_l1233_123392


namespace NUMINAMATH_GPT_solution_exists_for_100_100_l1233_123357

def exists_positive_integers_sum_of_cubes (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a^3 + b^3 + c^3 + d^3 = x

theorem solution_exists_for_100_100 : exists_positive_integers_sum_of_cubes (100 ^ 100) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_for_100_100_l1233_123357


namespace NUMINAMATH_GPT_ratio_of_weights_l1233_123324

noncomputable def tyler_weight (sam_weight : ℝ) : ℝ := sam_weight + 25
noncomputable def ratio_of_peter_to_tyler (peter_weight tyler_weight : ℝ) : ℝ := peter_weight / tyler_weight

theorem ratio_of_weights (sam_weight : ℝ) (peter_weight : ℝ) (h_sam : sam_weight = 105) (h_peter : peter_weight = 65) :
  ratio_of_peter_to_tyler peter_weight (tyler_weight sam_weight) = 0.5 := by
  -- We use the conditions to derive the information
  sorry

end NUMINAMATH_GPT_ratio_of_weights_l1233_123324


namespace NUMINAMATH_GPT_rectangle_height_l1233_123365

-- Define the given right-angled triangle with its legs and hypotenuse
variables {a b c d : ℝ}

-- Define the conditions: Right-angled triangle with legs a, b and hypotenuse c
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the height of the inscribed rectangle is d
def height_of_rectangle (a b d : ℝ) : Prop :=
  d = a + b

-- The problem statement: Prove that the height of the rectangle is the sum of the heights of the squares
theorem rectangle_height (a b c d : ℝ) (ht : right_angled_triangle a b c) : height_of_rectangle a b d :=
by
  sorry

end NUMINAMATH_GPT_rectangle_height_l1233_123365


namespace NUMINAMATH_GPT_base8_subtraction_correct_l1233_123303

noncomputable def base8_subtraction (x y : Nat) : Nat :=
  if y > x then 0 else x - y

theorem base8_subtraction_correct :
  base8_subtraction 546 321 - 105 = 120 :=
by
  -- Given the condition that all arithmetic is in base 8
  sorry

end NUMINAMATH_GPT_base8_subtraction_correct_l1233_123303


namespace NUMINAMATH_GPT_find_m_l1233_123305

theorem find_m (m : ℤ) (x y : ℤ) (h1 : x = 1) (h2 : y = m) (h3 : 3 * x - 4 * y = 7) : m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1233_123305


namespace NUMINAMATH_GPT_distance_between_centers_eq_l1233_123387

theorem distance_between_centers_eq (r1 r2 : ℝ) : ∃ d : ℝ, (d = r1 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_distance_between_centers_eq_l1233_123387


namespace NUMINAMATH_GPT_simplify_sqrt_product_l1233_123377

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) : 
  (Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_product_l1233_123377


namespace NUMINAMATH_GPT_solve_system_of_equations_l1233_123333

variable {x : Fin 15 → ℤ}

theorem solve_system_of_equations (h : ∀ i : Fin 15, 1 - x i * x ((i + 1) % 15) = 0) :
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) :=
by
  -- Here we put the proof, but it's omitted for now.
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1233_123333


namespace NUMINAMATH_GPT_value_of_5y_l1233_123308

-- Define positive integers
variables {x y z : ℕ}

-- Define the conditions
def conditions (x y z : ℕ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (5 * y = 6 * z) ∧ (x + y + z = 26)

-- The theorem statement
theorem value_of_5y (x y z : ℕ) (h : conditions x y z) : 5 * y = 30 :=
by
  -- proof skipped (proof goes here)
  sorry

end NUMINAMATH_GPT_value_of_5y_l1233_123308


namespace NUMINAMATH_GPT_cadence_worked_longer_by_5_months_l1233_123396

-- Definitions
def months_old_company : ℕ := 36

def salary_old_company : ℕ := 5000

def salary_new_company : ℕ := 6000

def total_earnings : ℕ := 426000

-- Prove that Cadence worked 5 months longer at her new company
theorem cadence_worked_longer_by_5_months :
  ∃ x : ℕ, 
  total_earnings = salary_old_company * months_old_company + 
                  salary_new_company * (months_old_company + x)
  ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_cadence_worked_longer_by_5_months_l1233_123396


namespace NUMINAMATH_GPT_bike_riders_count_l1233_123306

variables (B H : ℕ)

theorem bike_riders_count
  (h₁ : H = B + 178)
  (h₂ : H + B = 676) :
  B = 249 :=
sorry

end NUMINAMATH_GPT_bike_riders_count_l1233_123306


namespace NUMINAMATH_GPT_travel_time_correct_l1233_123326

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_correct_l1233_123326


namespace NUMINAMATH_GPT_construct_triangle_from_medians_l1233_123330

theorem construct_triangle_from_medians
    (s_a s_b s_c : ℝ)
    (h1 : s_a + s_b > s_c)
    (h2 : s_a + s_c > s_b)
    (h3 : s_b + s_c > s_a) :
    ∃ (a b c : ℝ), 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∃ (median_a median_b median_c : ℝ), 
        median_a = s_a ∧ 
        median_b = s_b ∧ 
        median_c = s_c) :=
sorry

end NUMINAMATH_GPT_construct_triangle_from_medians_l1233_123330


namespace NUMINAMATH_GPT_probability_of_distance_less_than_8000_l1233_123322

-- Define distances between cities

noncomputable def distances : List (String × String × ℕ) :=
  [("Bangkok", "Cape Town", 6300),
   ("Bangkok", "Honolulu", 7609),
   ("Bangkok", "London", 5944),
   ("Bangkok", "Tokyo", 2870),
   ("Cape Town", "Honolulu", 11535),
   ("Cape Town", "London", 5989),
   ("Cape Town", "Tokyo", 13400),
   ("Honolulu", "London", 7240),
   ("Honolulu", "Tokyo", 3805),
   ("London", "Tokyo", 5950)]

-- Define the total number of pairs and the pairs with distances less than 8000 miles

noncomputable def total_pairs : ℕ := 10
noncomputable def pairs_less_than_8000 : ℕ := 7

-- Define the statement of the probability being 7/10
theorem probability_of_distance_less_than_8000 :
  pairs_less_than_8000 / total_pairs = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_distance_less_than_8000_l1233_123322


namespace NUMINAMATH_GPT_books_not_sold_l1233_123327

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end NUMINAMATH_GPT_books_not_sold_l1233_123327
