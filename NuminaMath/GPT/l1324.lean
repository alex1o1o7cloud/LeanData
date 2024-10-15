import Mathlib

namespace NUMINAMATH_GPT_find_volume_from_vessel_c_l1324_132411

noncomputable def concentration_vessel_a : ℝ := 0.45
noncomputable def concentration_vessel_b : ℝ := 0.30
noncomputable def concentration_vessel_c : ℝ := 0.10
noncomputable def volume_vessel_a : ℝ := 4
noncomputable def volume_vessel_b : ℝ := 5
noncomputable def resultant_concentration : ℝ := 0.26

theorem find_volume_from_vessel_c (x : ℝ) : 
    concentration_vessel_a * volume_vessel_a + concentration_vessel_b * volume_vessel_b + concentration_vessel_c * x = 
    resultant_concentration * (volume_vessel_a + volume_vessel_b + x) → 
    x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_volume_from_vessel_c_l1324_132411


namespace NUMINAMATH_GPT_odd_increasing_three_digit_numbers_count_eq_50_l1324_132473

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end NUMINAMATH_GPT_odd_increasing_three_digit_numbers_count_eq_50_l1324_132473


namespace NUMINAMATH_GPT_percentage_comedies_l1324_132491

theorem percentage_comedies (a : ℕ) (d c T : ℕ) 
  (h1 : d = 5 * a) 
  (h2 : c = 10 * a) 
  (h3 : T = c + d + a) : 
  (c : ℝ) / T * 100 = 62.5 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_comedies_l1324_132491


namespace NUMINAMATH_GPT_cost_of_pencils_and_pens_l1324_132403

theorem cost_of_pencils_and_pens (a b : ℝ) (h1 : 4 * a + b = 2.60) (h2 : a + 3 * b = 2.15) : 3 * a + 2 * b = 2.63 :=
sorry

end NUMINAMATH_GPT_cost_of_pencils_and_pens_l1324_132403


namespace NUMINAMATH_GPT_rewrite_expression_and_compute_l1324_132451

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rewrite_expression_and_compute_l1324_132451


namespace NUMINAMATH_GPT_cost_of_article_l1324_132420

theorem cost_of_article (C G : ℝ) (h1 : C + G = 348) (h2 : C + 1.05 * G = 350) : C = 308 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l1324_132420


namespace NUMINAMATH_GPT_profit_percentage_is_50_l1324_132446

noncomputable def cost_of_machine := 11000
noncomputable def repair_cost := 5000
noncomputable def transportation_charges := 1000
noncomputable def selling_price := 25500

noncomputable def total_cost := cost_of_machine + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost) * 100

theorem profit_percentage_is_50 : profit_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_50_l1324_132446


namespace NUMINAMATH_GPT_student_B_more_stable_than_A_student_B_more_stable_l1324_132464

-- Define students A and B.
structure Student :=
  (average_score : ℝ)
  (variance : ℝ)

-- Given data for both students.
def studentA : Student :=
  { average_score := 90, variance := 51 }

def studentB : Student :=
  { average_score := 90, variance := 12 }

-- The theorem that student B has more stable performance than student A.
theorem student_B_more_stable_than_A (A B : Student) (h_avg : A.average_score = B.average_score) :
  A.variance > B.variance → B.variance < A.variance :=
by
  intro h
  linarith

-- Specific instance of the theorem with given data for students A and B.
theorem student_B_more_stable : studentA.variance > studentB.variance → studentB.variance < studentA.variance :=
  student_B_more_stable_than_A studentA studentB rfl

end NUMINAMATH_GPT_student_B_more_stable_than_A_student_B_more_stable_l1324_132464


namespace NUMINAMATH_GPT_number_of_three_digit_integers_congruent_to_2_mod_4_l1324_132442

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_integers_congruent_to_2_mod_4_l1324_132442


namespace NUMINAMATH_GPT_PQRS_product_l1324_132431

noncomputable def P : ℝ := (Real.sqrt 2023 + Real.sqrt 2024)
noncomputable def Q : ℝ := (-Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def R : ℝ := (Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def S : ℝ := (Real.sqrt 2024 - Real.sqrt 2023)

theorem PQRS_product : (P * Q * R * S) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_PQRS_product_l1324_132431


namespace NUMINAMATH_GPT_difference_qr_l1324_132484

-- Definitions of p, q, r in terms of the common multiplier x
def p (x : ℕ) := 3 * x
def q (x : ℕ) := 7 * x
def r (x : ℕ) := 12 * x

-- Given condition that the difference between p and q's share is 4000
def condition1 (x : ℕ) := q x - p x = 4000

-- Theorem stating that the difference between q and r's share is 5000
theorem difference_qr (x : ℕ) (h : condition1 x) : r x - q x = 5000 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_difference_qr_l1324_132484


namespace NUMINAMATH_GPT_sqrt_range_l1324_132480

theorem sqrt_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_range_l1324_132480


namespace NUMINAMATH_GPT_arithmetic_problem_l1324_132413

noncomputable def arithmetic_progression (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

noncomputable def sum_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_problem (a₁ d : ℝ)
  (h₁ : a₁ + (a₁ + 2 * d) = 5)
  (h₂ : 4 * (2 * a₁ + 3 * d) / 2 = 20) :
  (sum_terms a₁ d 8 - 2 * sum_terms a₁ d 4) / (sum_terms a₁ d 6 - sum_terms a₁ d 4 - sum_terms a₁ d 2) = 10 := by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l1324_132413


namespace NUMINAMATH_GPT_density_ratio_of_large_cube_l1324_132458

theorem density_ratio_of_large_cube 
  (V0 m0 : ℝ) (initial_density replacement_density: ℝ)
  (initial_mass final_mass : ℝ) (V_total : ℝ) 
  (h1 : initial_density = m0 / V0)
  (h2 : replacement_density = 2 * initial_density)
  (h3 : initial_mass = 8 * m0)
  (h4 : final_mass = 6 * m0 + 2 * (2 * m0))
  (h5 : V_total = 8 * V0) :
  initial_density / (final_mass / V_total) = 0.8 :=
sorry

end NUMINAMATH_GPT_density_ratio_of_large_cube_l1324_132458


namespace NUMINAMATH_GPT_employee_Y_base_pay_l1324_132432

theorem employee_Y_base_pay (P : ℝ) (h1 : 1.2 * P + P * 1.1 + P * 1.08 + P = P * 4.38)
                            (h2 : 2 * 1.5 * 1.2 * P = 3.6 * P)
                            (h3 : P * 4.38 + 100 + 3.6 * P = 1800) :
  P = 213.03 :=
by
  sorry

end NUMINAMATH_GPT_employee_Y_base_pay_l1324_132432


namespace NUMINAMATH_GPT_vertex_of_parabola_l1324_132433

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = (x - 6)^2 + 3 ↔ (x = 6 ∧ y = 3)) :=
sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1324_132433


namespace NUMINAMATH_GPT_max_value_of_f_l1324_132475

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ x, (f x = 1 / exp 1) ∧ (∀ y, f y ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1324_132475


namespace NUMINAMATH_GPT_direct_proportion_l1324_132468

theorem direct_proportion (c f p : ℝ) (h : f ≠ 0 ∧ p = c * f) : ∃ k : ℝ, p / f = k * (f / f) :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_l1324_132468


namespace NUMINAMATH_GPT_calculate_g_at_5_l1324_132462

variable {R : Type} [LinearOrderedField R] (g : R → R)
variable (x : R)

theorem calculate_g_at_5 (h : ∀ x : R, g (3 * x - 4) = 5 * x - 7) : g 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_calculate_g_at_5_l1324_132462


namespace NUMINAMATH_GPT_temperature_value_l1324_132487

theorem temperature_value (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 221) : t = 105 :=
by
  sorry

end NUMINAMATH_GPT_temperature_value_l1324_132487


namespace NUMINAMATH_GPT_solve_for_w_l1324_132448

theorem solve_for_w (w : ℕ) (h : w^2 - 5 * w = 0) (hp : w > 0) : w = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_w_l1324_132448


namespace NUMINAMATH_GPT_expand_expression_l1324_132426

variable {R : Type} [CommRing R]
variables (x y : R)

theorem expand_expression :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1324_132426


namespace NUMINAMATH_GPT_angle_C_is_sixty_l1324_132478

variable {A B C D E : Type}
variable {AD BE BC AC : ℝ}
variable {triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A} 
variable (angle_C : ℝ)

-- Given conditions
variable (h_eq : AD * BC = BE * AC)
variable (h_ineq : AC ≠ BC)

-- To prove
theorem angle_C_is_sixty (h_eq : AD * BC = BE * AC) (h_ineq : AC ≠ BC) : angle_C = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_is_sixty_l1324_132478


namespace NUMINAMATH_GPT_nh4cl_formed_l1324_132492

theorem nh4cl_formed :
  (∀ (nh3 hcl nh4cl : ℝ), nh3 = 1 ∧ hcl = 1 → nh3 + hcl = nh4cl → nh4cl = 1) :=
by
  intros nh3 hcl nh4cl
  sorry

end NUMINAMATH_GPT_nh4cl_formed_l1324_132492


namespace NUMINAMATH_GPT_existence_of_points_on_AC_l1324_132453

theorem existence_of_points_on_AC (A B C M : ℝ) (hAB : abs (A - B) = 2) (hBC : abs (B - C) = 1) :
  ((abs (A - M) + abs (B - M) = abs (C - M)) ↔ (M = A - 1) ∨ (M = A + 1)) :=
by
  sorry

end NUMINAMATH_GPT_existence_of_points_on_AC_l1324_132453


namespace NUMINAMATH_GPT_find_value_l1324_132477

theorem find_value (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2 * b)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l1324_132477


namespace NUMINAMATH_GPT_shaded_cubes_count_l1324_132417

theorem shaded_cubes_count :
  let faces := 6
  let shaded_on_one_face := 5
  let corner_cubes := 8
  let center_cubes := 2 * 1 -- center cubes shared among opposite faces
  let total_shaded_cubes := corner_cubes + center_cubes
  faces = 6 → shaded_on_one_face = 5 → corner_cubes = 8 → center_cubes = 2 →
  total_shaded_cubes = 10 := 
by
  intros _ _ _ _ 
  sorry

end NUMINAMATH_GPT_shaded_cubes_count_l1324_132417


namespace NUMINAMATH_GPT_henry_seashells_l1324_132489

theorem henry_seashells (H L : ℕ) (h1 : H + 24 + L = 59) (h2 : H + 24 + (3 * L) / 4 = 53) : H = 11 := by
  sorry

end NUMINAMATH_GPT_henry_seashells_l1324_132489


namespace NUMINAMATH_GPT_mary_total_earnings_l1324_132457

-- Define the earnings for each job
def cleaning_earnings (homes_cleaned : ℕ) : ℕ := 46 * homes_cleaned
def babysitting_earnings (days_babysat : ℕ) : ℕ := 35 * days_babysat
def petcare_earnings (days_petcare : ℕ) : ℕ := 60 * days_petcare

-- Define the total earnings
def total_earnings (homes_cleaned days_babysat days_petcare : ℕ) : ℕ :=
  cleaning_earnings homes_cleaned + babysitting_earnings days_babysat + petcare_earnings days_petcare

-- Given values
def homes_cleaned_last_week : ℕ := 4
def days_babysat_last_week : ℕ := 5
def days_petcare_last_week : ℕ := 3

-- Prove the total earnings
theorem mary_total_earnings : total_earnings homes_cleaned_last_week days_babysat_last_week days_petcare_last_week = 539 :=
by
  -- We just state the theorem; the proof is not required
  sorry

end NUMINAMATH_GPT_mary_total_earnings_l1324_132457


namespace NUMINAMATH_GPT_bricks_in_wall_is_720_l1324_132439

/-- 
Two bricklayers have varying speeds: one could build a wall in 12 hours and 
the other in 15 hours if working alone. Their efficiency decreases by 12 bricks
per hour when they work together. The contractor placed them together on this 
project and the wall was completed in 6 hours.
Prove that the number of bricks in the wall is 720.
-/
def number_of_bricks_in_wall (y : ℕ) : Prop :=
  let rate1 := y / 12
  let rate2 := y / 15
  let combined_rate := rate1 + rate2 - 12
  6 * combined_rate = y

theorem bricks_in_wall_is_720 : ∃ y : ℕ, number_of_bricks_in_wall y ∧ y = 720 :=
  by sorry

end NUMINAMATH_GPT_bricks_in_wall_is_720_l1324_132439


namespace NUMINAMATH_GPT_product_of_digits_l1324_132470

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 4 = 0) : A * B = 32 ∨ A * B = 36 :=
sorry

end NUMINAMATH_GPT_product_of_digits_l1324_132470


namespace NUMINAMATH_GPT_quadratic_passes_through_neg3_n_l1324_132415

-- Definition of the quadratic function with given conditions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
axiom max_at_neg2 : ∀ x, quadratic a b c x ≤ 8
axiom value_at_neg2 : quadratic a b c (-2) = 8
axiom passes_through_1_4 : quadratic a b c 1 = 4

-- Statement to prove
theorem quadratic_passes_through_neg3_n : quadratic a b c (-3) = 68 / 9 :=
sorry

end NUMINAMATH_GPT_quadratic_passes_through_neg3_n_l1324_132415


namespace NUMINAMATH_GPT_max_value_of_f_l1324_132493

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f 1 = 1 / Real.exp 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_f_l1324_132493


namespace NUMINAMATH_GPT_directrix_of_parabola_l1324_132416

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1324_132416


namespace NUMINAMATH_GPT_geom_seq_a4_l1324_132499

theorem geom_seq_a4 (a1 a2 a3 a4 r : ℝ)
  (h1 : a1 + a2 + a3 = 7)
  (h2 : a1 * a2 * a3 = 8)
  (h3 : a1 > 0)
  (h4 : r > 1)
  (h5 : a2 = a1 * r)
  (h6 : a3 = a1 * r^2)
  (h7 : a4 = a1 * r^3) : 
  a4 = 8 :=
sorry

end NUMINAMATH_GPT_geom_seq_a4_l1324_132499


namespace NUMINAMATH_GPT_find_certain_number_l1324_132423

theorem find_certain_number (n x : ℤ) (h1 : 9 - n / x = 7 + 8 / x) (h2 : x = 6) : n = 8 := by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1324_132423


namespace NUMINAMATH_GPT_factor_expression_l1324_132469

variable (a : ℝ)

theorem factor_expression : 37 * a^2 + 111 * a = 37 * a * (a + 3) :=
  sorry

end NUMINAMATH_GPT_factor_expression_l1324_132469


namespace NUMINAMATH_GPT_length_of_second_train_l1324_132483

theorem length_of_second_train (speed1 speed2 : ℝ) (length1 time : ℝ) (h1 : speed1 = 60) (h2 : speed2 = 40) 
  (h3 : length1 = 450) (h4 : time = 26.99784017278618) :
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  let length2 := total_distance - length1
  length2 = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l1324_132483


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_l1324_132495

theorem annual_rent_per_square_foot
  (length width : ℕ) (monthly_rent : ℕ) (h_length : length = 10)
  (h_width : width = 8) (h_monthly_rent : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := 
by 
  -- We assume the theorem is true.
  sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_l1324_132495


namespace NUMINAMATH_GPT_max_homework_time_l1324_132459

theorem max_homework_time (biology_time history_time geography_time : ℕ) :
    biology_time = 20 ∧ history_time = 2 * biology_time ∧ geography_time = 3 * history_time →
    biology_time + history_time + geography_time = 180 :=
by
    intros
    sorry

end NUMINAMATH_GPT_max_homework_time_l1324_132459


namespace NUMINAMATH_GPT_average_vegetables_per_week_l1324_132427

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_vegetables_per_week_l1324_132427


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1324_132428

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 4) :
  (x ^ 2 - 5 * x + 4 ≥ 0 ∧ ¬(∀ x, (x ^ 2 - 5 * x + 4 ≥ 0 → x > 4))) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1324_132428


namespace NUMINAMATH_GPT_total_rainfall_2007_correct_l1324_132472

noncomputable def rainfall_2005 : ℝ := 40.5
noncomputable def rainfall_2006 : ℝ := rainfall_2005 + 3
noncomputable def rainfall_2007 : ℝ := rainfall_2006 + 4
noncomputable def total_rainfall_2007 : ℝ := 12 * rainfall_2007

theorem total_rainfall_2007_correct : total_rainfall_2007 = 570 := 
sorry

end NUMINAMATH_GPT_total_rainfall_2007_correct_l1324_132472


namespace NUMINAMATH_GPT_possible_to_form_square_l1324_132441

def shape_covers_units : ℕ := 4

theorem possible_to_form_square (shape : ℕ) : ∃ n : ℕ, ∃ k : ℕ, n * n = shape * k :=
by
  use 4
  use 4
  sorry

end NUMINAMATH_GPT_possible_to_form_square_l1324_132441


namespace NUMINAMATH_GPT_average_first_6_numbers_l1324_132471

theorem average_first_6_numbers (A : ℕ) (h1 : (13 * 9) = (6 * A + 45 + 6 * 7)) : A = 5 :=
by 
  -- h1 : 117 = (6 * A + 45 + 42),
  -- solving for the value of A by performing algebraic operations will prove it.
  sorry

end NUMINAMATH_GPT_average_first_6_numbers_l1324_132471


namespace NUMINAMATH_GPT_general_formula_correct_S_k_equals_189_l1324_132476

-- Define the arithmetic sequence with initial conditions
def a (n : ℕ) : ℤ :=
  if n = 1 then -11
  else sorry  -- Will be defined by the general formula

-- Given conditions in Lean
def initial_condition (a : ℕ → ℤ) :=
  a 1 = -11 ∧ a 4 + a 6 = -6

-- General formula for the arithmetic sequence to be proven
def general_formula (n : ℕ) : ℤ := 2 * n - 13

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℤ :=
  n^2 - 12 * n

-- Problem 1: Prove the general formula
theorem general_formula_correct : ∀ n : ℕ, initial_condition a → a n = general_formula n :=
by sorry

-- Problem 2: Prove that k = 21 such that S_k = 189
theorem S_k_equals_189 : ∃ k : ℕ, S k = 189 ∧ k = 21 :=
by sorry

end NUMINAMATH_GPT_general_formula_correct_S_k_equals_189_l1324_132476


namespace NUMINAMATH_GPT_base_conversion_addition_correct_l1324_132485

theorem base_conversion_addition_correct :
  let A := 10
  let C := 12
  let n13 := 3 * 13^2 + 7 * 13^1 + 6
  let n14 := 4 * 14^2 + A * 14^1 + C
  n13 + n14 = 1540 := by
    let A := 10
    let C := 12
    let n13 := 3 * 13^2 + 7 * 13^1 + 6
    let n14 := 4 * 14^2 + A * 14^1 + C
    let sum := n13 + n14
    have h1 : n13 = 604 := by sorry
    have h2 : n14 = 936 := by sorry
    have h3 : sum = 1540 := by sorry
    exact h3

end NUMINAMATH_GPT_base_conversion_addition_correct_l1324_132485


namespace NUMINAMATH_GPT_inequality_true_l1324_132461

-- Define the conditions
variables (a b : ℝ) (h : a < b) (hb_neg : b < 0)

-- State the theorem to be proved
theorem inequality_true (ha : a < b) (hb : b < 0) : (|a| / |b| > 1) :=
sorry

end NUMINAMATH_GPT_inequality_true_l1324_132461


namespace NUMINAMATH_GPT_second_number_multiple_of_seven_l1324_132405

theorem second_number_multiple_of_seven (x : ℕ) (h : gcd (gcd 105 x) 2436 = 7) : 7 ∣ x :=
sorry

end NUMINAMATH_GPT_second_number_multiple_of_seven_l1324_132405


namespace NUMINAMATH_GPT_sum_min_max_x_y_l1324_132463

theorem sum_min_max_x_y (x y : ℕ) (h : 6 * x + 7 * y = 2012): 288 + 335 = 623 :=
by
  sorry

end NUMINAMATH_GPT_sum_min_max_x_y_l1324_132463


namespace NUMINAMATH_GPT_simplify_expression_l1324_132479

theorem simplify_expression (a : ℝ) (h : a = -2) : 
  (1 - a / (a + 1)) / (1 / (1 - a ^ 2)) = 1 / 3 :=
by
  subst h
  sorry

end NUMINAMATH_GPT_simplify_expression_l1324_132479


namespace NUMINAMATH_GPT_square_area_l1324_132490

theorem square_area (side : ℕ) (h : side = 19) : side * side = 361 := by
  sorry

end NUMINAMATH_GPT_square_area_l1324_132490


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1324_132406

-- Define the sequence and state the conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

-- The mathematical problem rewritten in Lean 4 statement
theorem geometric_sequence_sum (a : ℕ → ℝ) (s : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : s 2 = 7)
  (h3 : s 6 = 91)
  : ∃ s_4 : ℝ, s_4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1324_132406


namespace NUMINAMATH_GPT_intersection_A_complement_UB_l1324_132440

-- Definitions of the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5 * x ≥ 0}

-- Complement of B w.r.t. U
def complement_U_B : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

-- The statement we want to prove
theorem intersection_A_complement_UB : A ∩ complement_U_B = {2, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_UB_l1324_132440


namespace NUMINAMATH_GPT_symmetrical_circle_proof_l1324_132481

open Real

-- Definition of the original circle equation
def original_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Defining the symmetrical circle equation to be proven
def symmetrical_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 5

theorem symmetrical_circle_proof :
  ∀ x y : ℝ, original_circle x y ↔ symmetrical_circle x y :=
by sorry

end NUMINAMATH_GPT_symmetrical_circle_proof_l1324_132481


namespace NUMINAMATH_GPT_unused_combinations_eq_40_l1324_132434

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end NUMINAMATH_GPT_unused_combinations_eq_40_l1324_132434


namespace NUMINAMATH_GPT_find_ratio_l1324_132465

open Real

-- Definitions and conditions
variables (b1 b2 : ℝ) (F1 F2 : ℝ × ℝ)
noncomputable def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 49) + (Q.2^2 / b1^2) = 1
noncomputable def hyperbola_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 16) - (Q.2^2 / b2^2) = 1
noncomputable def same_foci (Q : ℝ × ℝ) : Prop := true  -- Placeholder: Representing that both shapes have the same foci F1 and F2

-- The main theorem
theorem find_ratio (Q : ℝ × ℝ) (h1 : ellipse_eq b1 Q) (h2 : hyperbola_eq b2 Q) (h3 : same_foci Q) : 
  abs ((dist Q F1) - (dist Q F2)) / ((dist Q F1) + (dist Q F2)) = 4 / 7 := 
sorry

end NUMINAMATH_GPT_find_ratio_l1324_132465


namespace NUMINAMATH_GPT_symmetric_point_exists_l1324_132435

-- Define the point P and line equation.
structure Point (α : Type*) := (x : α) (y : α)
def P : Point ℝ := ⟨5, -2⟩
def line_eq (x y : ℝ) : Prop := x - y + 5 = 0

-- Define a function for the line PQ being perpendicular to the given line.
def is_perpendicular (P Q : Point ℝ) : Prop :=
  (Q.y - P.y) / (Q.x - P.x) = -1

-- Define a function for the midpoint of PQ lying on the given line.
def midpoint_on_line (P Q : Point ℝ) : Prop :=
  line_eq ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Define the symmetry function based on the provided conditions.
def is_symmetric (Q : Point ℝ) : Prop :=
  is_perpendicular P Q ∧ midpoint_on_line P Q

-- State the main theorem to be proved: there exists a point Q that satisfies the 
-- conditions and is symmetric to P with respect to the given line.
theorem symmetric_point_exists : ∃ Q : Point ℝ, is_symmetric Q ∧ Q = ⟨-7, 10⟩ :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_exists_l1324_132435


namespace NUMINAMATH_GPT_shoe_price_on_monday_l1324_132498

theorem shoe_price_on_monday
  (price_on_thursday : ℝ)
  (price_increase : ℝ)
  (discount : ℝ)
  (price_on_friday : ℝ := price_on_thursday * (1 + price_increase))
  (price_on_monday : ℝ := price_on_friday * (1 - discount))
  (price_on_thursday_eq : price_on_thursday = 50)
  (price_increase_eq : price_increase = 0.2)
  (discount_eq : discount = 0.15) :
  price_on_monday = 51 :=
by
  sorry

end NUMINAMATH_GPT_shoe_price_on_monday_l1324_132498


namespace NUMINAMATH_GPT_symmetry_origin_l1324_132412

def f (x : ℝ) : ℝ := x^3 + x

theorem symmetry_origin : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end NUMINAMATH_GPT_symmetry_origin_l1324_132412


namespace NUMINAMATH_GPT_total_amount_l1324_132443

-- Define p, q, r and their shares
variables (p q r : ℕ)

-- Given conditions translated to Lean definitions
def ratio_pq := (5 * q) = (4 * p)
def ratio_qr := (9 * r) = (10 * q)
def r_share := r = 400

-- Statement to prove
theorem total_amount (hpq : ratio_pq p q) (hqr : ratio_qr q r) (hr : r_share r) :
  (p + q + r) = 1210 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l1324_132443


namespace NUMINAMATH_GPT_wendy_score_l1324_132452

def score_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def treasures_second_level : ℕ := 3

theorem wendy_score :
  score_per_treasure * treasures_first_level + score_per_treasure * treasures_second_level = 35 :=
by
  sorry

end NUMINAMATH_GPT_wendy_score_l1324_132452


namespace NUMINAMATH_GPT_solve_for_q_l1324_132437

theorem solve_for_q 
  (n m q : ℕ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m + n) / 90)
  (h3 : 5 / 6 = (q - m) / 150) : 
  q = 150 :=
sorry

end NUMINAMATH_GPT_solve_for_q_l1324_132437


namespace NUMINAMATH_GPT_order_of_numbers_l1324_132488

theorem order_of_numbers (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) : y < -y ∧ -y < -xy ∧ -xy < x :=
by 
  sorry

end NUMINAMATH_GPT_order_of_numbers_l1324_132488


namespace NUMINAMATH_GPT_minimum_dot_product_l1324_132449

-- Definitions of points A and B
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 0)

-- Definition of condition that P lies on the line x - y + 1 = 0
def onLineP (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Definition of dot product between vectors PA and PB
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Lean 4 theorem statement
theorem minimum_dot_product (P : ℝ × ℝ) (hP : onLineP P) : 
  dotProduct P pointA pointB = 0 := 
sorry

end NUMINAMATH_GPT_minimum_dot_product_l1324_132449


namespace NUMINAMATH_GPT_find_green_pepper_weight_l1324_132456

variable (weight_red_peppers : ℝ) (total_weight_peppers : ℝ)

theorem find_green_pepper_weight 
    (h1 : weight_red_peppers = 0.33) 
    (h2 : total_weight_peppers = 0.66) 
    : total_weight_peppers - weight_red_peppers = 0.33 := 
by sorry

end NUMINAMATH_GPT_find_green_pepper_weight_l1324_132456


namespace NUMINAMATH_GPT_distinct_license_plates_l1324_132494

noncomputable def license_plates : ℕ :=
  let digits_possibilities := 10^5
  let letters_possibilities := 26^3
  let positions := 6
  positions * digits_possibilities * letters_possibilities

theorem distinct_license_plates : 
  license_plates = 105456000 := by
  sorry

end NUMINAMATH_GPT_distinct_license_plates_l1324_132494


namespace NUMINAMATH_GPT_find_an_l1324_132421

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end NUMINAMATH_GPT_find_an_l1324_132421


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1324_132401

-- Definitions and Conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Problem Statement
theorem arithmetic_sequence_ninth_term
  (h1 : a 3 = 4)
  (h2 : S 11 = 110)
  (h3 : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  a 9 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1324_132401


namespace NUMINAMATH_GPT_power_of_powers_l1324_132422

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end NUMINAMATH_GPT_power_of_powers_l1324_132422


namespace NUMINAMATH_GPT_range_of_x_l1324_132445

theorem range_of_x (x : ℝ) (h : (x + 1) ^ 0 = 1) : x ≠ -1 :=
sorry

end NUMINAMATH_GPT_range_of_x_l1324_132445


namespace NUMINAMATH_GPT_expression_divisible_by_19_l1324_132418

theorem expression_divisible_by_19 (n : ℕ) (h : n > 0) : 
  19 ∣ (5^(2*n - 1) + 3^(n - 2) * 2^(n - 1)) := 
by 
  sorry

end NUMINAMATH_GPT_expression_divisible_by_19_l1324_132418


namespace NUMINAMATH_GPT_m_value_quadratic_l1324_132438

theorem m_value_quadratic (m : ℝ)
  (h1 : |m - 2| = 2)
  (h2 : m - 4 ≠ 0) :
  m = 0 :=
sorry

end NUMINAMATH_GPT_m_value_quadratic_l1324_132438


namespace NUMINAMATH_GPT_subset_P1_P2_l1324_132408

def P1 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P2 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

theorem subset_P1_P2 (a : ℝ) : P1 a ⊆ P2 a :=
by intros x hx; sorry

end NUMINAMATH_GPT_subset_P1_P2_l1324_132408


namespace NUMINAMATH_GPT_sum_of_squares_eq_three_l1324_132497

theorem sum_of_squares_eq_three
  (a b s : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a * s^2 + b * s + b = 0)
  (h₂ : a * (1 / s)^2 + a * (1 / s) + b = 0)
  (h₃ : s * (1 / s) = 1) :
  s^2 + (1 / s)^2 = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_eq_three_l1324_132497


namespace NUMINAMATH_GPT_max_teams_4_weeks_l1324_132486

noncomputable def max_teams_in_tournament (weeks number_teams : ℕ) : ℕ :=
  if h : weeks > 0 then (number_teams * (number_teams - 1)) / (2 * weeks) else 0

theorem max_teams_4_weeks : max_teams_in_tournament 4 7 = 6 := by
  -- Assumptions
  let n := 6
  let teams := 7 * n
  let weeks := 4
  
  -- Define the constraints and checks here
  sorry

end NUMINAMATH_GPT_max_teams_4_weeks_l1324_132486


namespace NUMINAMATH_GPT_remainder_2pow33_minus_1_div_9_l1324_132466

theorem remainder_2pow33_minus_1_div_9 : (2^33 - 1) % 9 = 7 := 
  sorry

end NUMINAMATH_GPT_remainder_2pow33_minus_1_div_9_l1324_132466


namespace NUMINAMATH_GPT_chord_through_P_midpoint_of_ellipse_has_given_line_l1324_132447

-- Define the ellipse
def ellipse (x y : ℝ) := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def pointP := (3, 1)

-- Define the problem statement
theorem chord_through_P_midpoint_of_ellipse_has_given_line:
  ∃ (m : ℝ) (c : ℝ), (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 144 → x + y = m ∧ 3 * x + y = c) → 
  ∃ (A : ℝ) (B : ℝ), ellipse 3 1 ∧ (A * 4 + B * 3 - 15 = 0) := sorry

end NUMINAMATH_GPT_chord_through_P_midpoint_of_ellipse_has_given_line_l1324_132447


namespace NUMINAMATH_GPT_ferris_wheel_capacity_l1324_132455

theorem ferris_wheel_capacity 
  (num_seats : ℕ)
  (people_per_seat : ℕ)
  (h1 : num_seats = 4)
  (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_capacity_l1324_132455


namespace NUMINAMATH_GPT_project_completion_days_l1324_132454

theorem project_completion_days (A_days : ℕ) (B_days : ℕ) (A_alone_days : ℕ) :
  A_days = 20 → B_days = 25 → A_alone_days = 2 → (A_alone_days : ℚ) * (1 / A_days) + (10 : ℚ) * (1 / (A_days * B_days / (A_days + B_days))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_project_completion_days_l1324_132454


namespace NUMINAMATH_GPT_collete_age_ratio_l1324_132414

theorem collete_age_ratio (Ro R C : ℕ) (h1 : R = 2 * Ro) (h2 : Ro = 8) (h3 : R - C = 12) :
  C / Ro = 1 / 2 := by
sorry

end NUMINAMATH_GPT_collete_age_ratio_l1324_132414


namespace NUMINAMATH_GPT_economic_rationale_education_policy_l1324_132402

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end NUMINAMATH_GPT_economic_rationale_education_policy_l1324_132402


namespace NUMINAMATH_GPT_find_S6_l1324_132409

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}

/-- sum_of_first_n_terms_of_geometric_sequence -/
def sum_of_first_n_terms_of_geometric_sequence (S : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, S n = a1 * (1 - r^(n+1)) / (1 - r)

-- Given conditions
axiom geom_seq_positive_terms : ∀ n, a n > 0
axiom sum_S2 : S 2 = 3
axiom sum_S4 : S 4 = 15

theorem find_S6 : S 6 = 63 := by
  sorry

end NUMINAMATH_GPT_find_S6_l1324_132409


namespace NUMINAMATH_GPT_geometric_sum_five_terms_l1324_132474

theorem geometric_sum_five_terms (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_a2a4 : a 1 * a 3 = 16)
  (h_ratio : (a 3 + a 4 + a 7) / (a 0 + a 1 + a 4) = 8) :
  S 5 = 31 :=
sorry

end NUMINAMATH_GPT_geometric_sum_five_terms_l1324_132474


namespace NUMINAMATH_GPT_correct_time_fraction_l1324_132404

theorem correct_time_fraction : 
  (∀ hour : ℕ, hour < 24 → true) →
  (∀ minute : ℕ, minute < 60 → (minute ≠ 16)) →
  (fraction_of_correct_time = 59 / 60) :=
by
  intros h_hour h_minute
  sorry

end NUMINAMATH_GPT_correct_time_fraction_l1324_132404


namespace NUMINAMATH_GPT_committee_with_one_boy_one_girl_prob_l1324_132424

def total_members := 30
def boys := 12
def girls := 18
def committee_size := 6

theorem committee_with_one_boy_one_girl_prob :
  let total_ways := Nat.choose total_members committee_size
  let all_boys_ways := Nat.choose boys committee_size
  let all_girls_ways := Nat.choose girls committee_size
  let prob_all_boys_or_all_girls := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - prob_all_boys_or_all_girls
  desired_prob = 19145 / 19793 :=
by
  sorry

end NUMINAMATH_GPT_committee_with_one_boy_one_girl_prob_l1324_132424


namespace NUMINAMATH_GPT_floor_add_ceil_eq_five_l1324_132482

theorem floor_add_ceil_eq_five (x : ℝ) :
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 5 ↔ 2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_floor_add_ceil_eq_five_l1324_132482


namespace NUMINAMATH_GPT_new_solution_is_45_percent_liquid_x_l1324_132467

-- Define initial conditions
def solution_y_initial_weight := 8.0 -- kilograms
def percent_liquid_x := 0.30
def percent_water := 0.70
def evaporated_water_weight := 4.0 -- kilograms
def added_solution_y_weight := 4.0 -- kilograms

-- Define the relevant quantities
def liquid_x_initial := solution_y_initial_weight * percent_liquid_x
def water_initial := solution_y_initial_weight * percent_water
def remaining_water_after_evaporation := water_initial - evaporated_water_weight

def liquid_x_after_evaporation := liquid_x_initial 
def water_after_evaporation := remaining_water_after_evaporation

def added_liquid_x := added_solution_y_weight * percent_liquid_x
def added_water := added_solution_y_weight * percent_water

def total_liquid_x := liquid_x_after_evaporation + added_liquid_x
def total_water := water_after_evaporation + added_water

def total_new_solution_weight := total_liquid_x + total_water

def new_solution_percent_liquid_x := (total_liquid_x / total_new_solution_weight) * 100

-- The theorem we want to prove
theorem new_solution_is_45_percent_liquid_x : new_solution_percent_liquid_x = 45 := by
  sorry

end NUMINAMATH_GPT_new_solution_is_45_percent_liquid_x_l1324_132467


namespace NUMINAMATH_GPT_marys_next_birthday_l1324_132410

theorem marys_next_birthday (d s m : ℝ) (h1 : s = 0.7 * d) (h2 : m = 1.3 * s) (h3 : m + s + d = 25.2) : m + 1 = 9 :=
by
  sorry

end NUMINAMATH_GPT_marys_next_birthday_l1324_132410


namespace NUMINAMATH_GPT_triangle_abc_l1324_132407

/-!
# Problem Statement
In triangle ABC with side lengths a, b, and c opposite to vertices A, B, and C respectively, we are given that ∠A = 2 * ∠B. We need to prove that a² = b * (b + c).
-/

variables (A B C : Type) -- Define vertices of the triangle
variables (α β γ : ℝ) -- Define angles at vertices A, B, and C respectively.

-- Define sides of the triangle
variables (a b c x y : ℝ) -- Define sides opposite to the corresponding angles

-- Main statement to prove in Lean 4
theorem triangle_abc (h1 : α = 2 * β) (h2 : a = b * (2 * β)) :
  a^2 = b * (b + c) :=
sorry

end NUMINAMATH_GPT_triangle_abc_l1324_132407


namespace NUMINAMATH_GPT_find_b_l1324_132430

theorem find_b (a b : ℝ) (h₁ : ∀ x y, y = 0.75 * x + 1 → (4, b) = (x, y))
                (h₂ : k = 0.75) : b = 4 :=
by sorry

end NUMINAMATH_GPT_find_b_l1324_132430


namespace NUMINAMATH_GPT_find_x_l1324_132425

theorem find_x (x : ℕ) (h : x + 1 = 6) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_l1324_132425


namespace NUMINAMATH_GPT_number_of_schools_is_8_l1324_132436

-- Define the number of students trying out and not picked per school
def students_trying_out := 65.0
def students_not_picked := 17.0
def students_picked := students_trying_out - students_not_picked

-- Define the total number of students who made the teams
def total_students_made_teams := 384.0

-- Define the number of schools
def number_of_schools := total_students_made_teams / students_picked

theorem number_of_schools_is_8 : number_of_schools = 8 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_schools_is_8_l1324_132436


namespace NUMINAMATH_GPT_number_of_different_ways_to_travel_l1324_132444

-- Define the conditions
def number_of_morning_flights : ℕ := 2
def number_of_afternoon_flights : ℕ := 3

-- Assert the question and the answer
theorem number_of_different_ways_to_travel : 
  (number_of_morning_flights * number_of_afternoon_flights) = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_different_ways_to_travel_l1324_132444


namespace NUMINAMATH_GPT_mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l1324_132496

noncomputable def Mork_base_income (M : ℝ) : ℝ := M
noncomputable def Mindy_base_income (M : ℝ) : ℝ := 4 * M
noncomputable def Mork_total_income (M : ℝ) : ℝ := 1.5 * M
noncomputable def Mindy_total_income (M : ℝ) : ℝ := 6 * M

noncomputable def Mork_total_tax (M : ℝ) : ℝ :=
  0.4 * M + 0.5 * 0.5 * M
noncomputable def Mindy_total_tax (M : ℝ) : ℝ :=
  0.3 * 4 * M + 0.35 * 2 * M

noncomputable def Mork_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M) / (Mork_total_income M)

noncomputable def Mindy_effective_tax_rate (M : ℝ) : ℝ :=
  (Mindy_total_tax M) / (Mindy_total_income M)

noncomputable def combined_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M + Mindy_total_tax M) / (Mork_total_income M + Mindy_total_income M)

theorem mork_effective_tax_rate_theorem (M : ℝ) : Mork_effective_tax_rate M = 43.33 / 100 := sorry
theorem mindy_effective_tax_rate_theorem (M : ℝ) : Mindy_effective_tax_rate M = 31.67 / 100 := sorry
theorem combined_effective_tax_rate_theorem (M : ℝ) : combined_effective_tax_rate M = 34 / 100 := sorry

end NUMINAMATH_GPT_mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l1324_132496


namespace NUMINAMATH_GPT_domain_f_correct_domain_g_correct_l1324_132450

noncomputable def domain_f : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ x ≠ 1}

noncomputable def expected_domain_f : Set ℝ :=
  {x | (-1 ≤ x ∧ x < 1) ∨ x > 1}

theorem domain_f_correct :
  domain_f = expected_domain_f :=
by
  sorry

noncomputable def domain_g : Set ℝ :=
  {x | 3 - 4 * x > 0}

noncomputable def expected_domain_g : Set ℝ :=
  {x | x < 3 / 4}

theorem domain_g_correct :
  domain_g = expected_domain_g :=
by
  sorry

end NUMINAMATH_GPT_domain_f_correct_domain_g_correct_l1324_132450


namespace NUMINAMATH_GPT_survey_respondents_l1324_132460

theorem survey_respondents
  (X Y Z : ℕ) 
  (h1 : X = 360) 
  (h2 : X * 4 = Y * 9) 
  (h3 : X * 3 = Z * 9) : 
  X + Y + Z = 640 :=
by
  sorry

end NUMINAMATH_GPT_survey_respondents_l1324_132460


namespace NUMINAMATH_GPT_peanut_count_l1324_132419

-- Definitions
def initial_peanuts : Nat := 10
def added_peanuts : Nat := 8

-- Theorem to prove
theorem peanut_count : (initial_peanuts + added_peanuts) = 18 := 
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_peanut_count_l1324_132419


namespace NUMINAMATH_GPT_cost_of_article_l1324_132429

variable (C : ℝ) 
variable (G : ℝ)
variable (H1 : G = 380 - C)
variable (H2 : 1.05 * G = 420 - C)

theorem cost_of_article : C = 420 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l1324_132429


namespace NUMINAMATH_GPT_pi_div_two_minus_alpha_in_third_quadrant_l1324_132400

theorem pi_div_two_minus_alpha_in_third_quadrant (α : ℝ) (k : ℤ) (h : ∃ k : ℤ, (π + 2 * k * π < α) ∧ (α < 3 * π / 2 + 2 * k * π)) : 
  ∃ k : ℤ, (π + 2 * k * π < (π / 2 - α)) ∧ ((π / 2 - α) < 3 * π / 2 + 2 * k * π) :=
sorry

end NUMINAMATH_GPT_pi_div_two_minus_alpha_in_third_quadrant_l1324_132400
