import Mathlib

namespace NUMINAMATH_GPT_find_a_minus_b_l34_3482

theorem find_a_minus_b (a b : ℝ)
  (h1 : 6 = a * 3 + b)
  (h2 : 26 = a * 7 + b) :
  a - b = 14 := 
sorry

end NUMINAMATH_GPT_find_a_minus_b_l34_3482


namespace NUMINAMATH_GPT_dow_jones_morning_value_l34_3461

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end NUMINAMATH_GPT_dow_jones_morning_value_l34_3461


namespace NUMINAMATH_GPT_masha_comb_teeth_count_l34_3463

theorem masha_comb_teeth_count (katya_teeth : ℕ) (masha_to_katya_ratio : ℕ) 
  (katya_teeth_eq : katya_teeth = 11) 
  (masha_to_katya_ratio_eq : masha_to_katya_ratio = 5) : 
  ∃ masha_teeth : ℕ, masha_teeth = 53 :=
by
  have katya_segments := 2 * katya_teeth - 1
  have masha_segments := masha_to_katya_ratio * katya_segments
  let masha_teeth := (masha_segments + 1) / 2
  use masha_teeth
  have masha_teeth_eq := (2 * masha_teeth - 1 = 105)
  sorry

end NUMINAMATH_GPT_masha_comb_teeth_count_l34_3463


namespace NUMINAMATH_GPT_NumberOfStudentsEnrolledOnlyInEnglish_l34_3442

-- Definition of the problem's variables and conditions
variables (TotalStudents BothEnglishAndGerman TotalGerman OnlyEnglish OnlyGerman : ℕ)
variables (h1 : TotalStudents = 52)
variables (h2 : BothEnglishAndGerman = 12)
variables (h3 : TotalGerman = 22)
variables (h4 : TotalStudents = OnlyEnglish + OnlyGerman + BothEnglishAndGerman)
variables (h5 : OnlyGerman = TotalGerman - BothEnglishAndGerman)

-- Theorem to prove the number of students enrolled only in English
theorem NumberOfStudentsEnrolledOnlyInEnglish : OnlyEnglish = 30 :=
by
  -- Insert the necessary proof steps here to derive the number of students enrolled only in English from the given conditions
  sorry

end NUMINAMATH_GPT_NumberOfStudentsEnrolledOnlyInEnglish_l34_3442


namespace NUMINAMATH_GPT_square_b_perimeter_l34_3494

/-- Square A has an area of 121 square centimeters. Square B has a certain perimeter.
  If square B is placed within square A and a random point is chosen within square A,
  the probability that the point is not within square B is 0.8677685950413223.
  Prove the perimeter of square B is 16 centimeters. -/
theorem square_b_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) 
  (h1 : area_A = 121)
  (h2 : prob = 0.8677685950413223)
  (h3 : ∃ (a b : ℝ), area_A = a * a ∧ a * a - b * b = prob * area_A) :
  perimeter_B = 16 :=
sorry

end NUMINAMATH_GPT_square_b_perimeter_l34_3494


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l34_3408

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l34_3408


namespace NUMINAMATH_GPT_roots_cubic_sum_l34_3473

theorem roots_cubic_sum :
  (∃ x1 x2 x3 x4 : ℂ, (x1^4 + 5*x1^3 + 6*x1^2 + 5*x1 + 1 = 0) ∧
                       (x2^4 + 5*x2^3 + 6*x2^2 + 5*x2 + 1 = 0) ∧
                       (x3^4 + 5*x3^3 + 6*x3^2 + 5*x3 + 1 = 0) ∧
                       (x4^4 + 5*x4^3 + 6*x4^2 + 5*x4 + 1 = 0)) →
  (x1^3 + x2^3 + x3^3 + x4^3 = -54) :=
sorry

end NUMINAMATH_GPT_roots_cubic_sum_l34_3473


namespace NUMINAMATH_GPT_tammy_total_miles_l34_3417

noncomputable def miles_per_hour : ℝ := 1.527777778
noncomputable def hours_driven : ℝ := 36.0
noncomputable def total_miles := miles_per_hour * hours_driven

theorem tammy_total_miles : abs (total_miles - 55.0) < 1e-5 :=
by
  sorry

end NUMINAMATH_GPT_tammy_total_miles_l34_3417


namespace NUMINAMATH_GPT_curve_symmetry_l34_3416

theorem curve_symmetry :
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧
  ∀ (ρ θ' : ℝ), ρ = 4 * Real.sin (θ' - Real.pi / 3) ↔ ρ = 4 * Real.sin ((θ - θ') - Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_curve_symmetry_l34_3416


namespace NUMINAMATH_GPT_calculate_series_l34_3481

theorem calculate_series : 20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 200 := 
by
  sorry

end NUMINAMATH_GPT_calculate_series_l34_3481


namespace NUMINAMATH_GPT_driver_net_hourly_rate_l34_3484

theorem driver_net_hourly_rate
  (hours : ℝ) (speed : ℝ) (efficiency : ℝ) (cost_per_gallon : ℝ) (compensation_rate : ℝ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : efficiency = 25)
  (h4 : cost_per_gallon = 2.50)
  (h5 : compensation_rate = 0.60)
  :
  ((compensation_rate * (speed * hours) - (cost_per_gallon * (speed * hours / efficiency))) / hours) = 25 :=
sorry

end NUMINAMATH_GPT_driver_net_hourly_rate_l34_3484


namespace NUMINAMATH_GPT_bicycle_speed_l34_3458

theorem bicycle_speed (d1 d2 v1 v_avg : ℝ)
  (h1 : d1 = 300) 
  (h2 : d1 + d2 = 450) 
  (h3 : v1 = 20) 
  (h4 : v_avg = 18) : 
  (d2 / ((d1 / v1) + d2 / (d2 * v_avg / 450)) = 15) :=
by 
  sorry

end NUMINAMATH_GPT_bicycle_speed_l34_3458


namespace NUMINAMATH_GPT_geometric_series_cubes_sum_l34_3421

theorem geometric_series_cubes_sum (b s : ℝ) (h : -1 < s ∧ s < 1) :
  ∑' n : ℕ, (b * s^n)^3 = b^3 / (1 - s^3) := 
sorry

end NUMINAMATH_GPT_geometric_series_cubes_sum_l34_3421


namespace NUMINAMATH_GPT_remainder_when_dividing_928927_by_6_l34_3450

theorem remainder_when_dividing_928927_by_6 :
  928927 % 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_928927_by_6_l34_3450


namespace NUMINAMATH_GPT_range_of_a_l34_3425

noncomputable def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 < a ∨ -1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l34_3425


namespace NUMINAMATH_GPT_possible_sets_B_l34_3407

def A : Set ℤ := {-1}

def isB (B : Set ℤ) : Prop :=
  A ∪ B = {-1, 3}

theorem possible_sets_B : ∀ B : Set ℤ, isB B → B = {3} ∨ B = {-1, 3} :=
by
  intros B hB
  sorry

end NUMINAMATH_GPT_possible_sets_B_l34_3407


namespace NUMINAMATH_GPT_krishan_money_l34_3467

variable {R G K : ℕ}

theorem krishan_money 
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (hR : R = 588)
  : K = 3468 :=
by
  sorry

end NUMINAMATH_GPT_krishan_money_l34_3467


namespace NUMINAMATH_GPT_isosceles_triangle_area_l34_3437

-- Definitions
def isosceles_triangle (b h : ℝ) : Prop :=
∃ a : ℝ, a * b / 2 = a * h

def square_of_area_one (a : ℝ) : Prop :=
a = 1

def centroids_coincide (g_triangle g_square : ℝ × ℝ) : Prop :=
g_triangle = g_square

-- The statement of the problem
theorem isosceles_triangle_area
  (b h : ℝ)
  (s : ℝ)
  (triangle_centroid : ℝ × ℝ)
  (square_centroid : ℝ × ℝ)
  (H1 : isosceles_triangle b h)
  (H2 : square_of_area_one s)
  (H3 : centroids_coincide triangle_centroid square_centroid)
  : b * h / 2 = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l34_3437


namespace NUMINAMATH_GPT_toys_produced_in_week_l34_3468

-- Define the number of working days in a week
def working_days_in_week : ℕ := 4

-- Define the number of toys produced per day
def toys_produced_per_day : ℕ := 1375

-- The statement to be proved
theorem toys_produced_in_week :
  working_days_in_week * toys_produced_per_day = 5500 :=
by
  sorry

end NUMINAMATH_GPT_toys_produced_in_week_l34_3468


namespace NUMINAMATH_GPT_contrapositive_l34_3433

theorem contrapositive (x y : ℝ) : (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_contrapositive_l34_3433


namespace NUMINAMATH_GPT_monthly_fee_for_second_plan_l34_3423

theorem monthly_fee_for_second_plan 
  (monthly_fee_first_plan : ℝ) 
  (rate_first_plan : ℝ) 
  (rate_second_plan : ℝ) 
  (minutes : ℕ) 
  (monthly_fee_second_plan : ℝ) :
  monthly_fee_first_plan = 22 -> 
  rate_first_plan = 0.13 -> 
  rate_second_plan = 0.18 -> 
  minutes = 280 -> 
  (22 + 0.13 * 280 = monthly_fee_second_plan + 0.18 * 280) -> 
  monthly_fee_second_plan = 8 := 
by
  intros h_fee_first_plan h_rate_first_plan h_rate_second_plan h_minutes h_equal_costs
  sorry

end NUMINAMATH_GPT_monthly_fee_for_second_plan_l34_3423


namespace NUMINAMATH_GPT_value_of_expression_l34_3434

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x - 4)^2 = 100 :=
by
  -- Given the hypothesis h: x = -2
  -- Need to show: (3 * x - 4)^2 = 100
  sorry

end NUMINAMATH_GPT_value_of_expression_l34_3434


namespace NUMINAMATH_GPT_intersection_equals_l34_3471

def A : Set ℝ := {x | x < 1}

def B : Set ℝ := {x | x^2 + x ≤ 6}

theorem intersection_equals : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_equals_l34_3471


namespace NUMINAMATH_GPT_deepak_current_age_l34_3403

variable (A D : ℕ)

def ratio_condition : Prop := A * 5 = D * 2
def arun_future_age (A : ℕ) : Prop := A + 10 = 30

theorem deepak_current_age (h1 : ratio_condition A D) (h2 : arun_future_age A) : D = 50 := sorry

end NUMINAMATH_GPT_deepak_current_age_l34_3403


namespace NUMINAMATH_GPT_geometric_sequence_sum_terms_l34_3464

noncomputable def geometric_sequence (a_1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_sum_terms :
  ∀ (a_1 q : ℕ), a_1 = 3 → 
  (geometric_sequence 3 q 1 + geometric_sequence 3 q 2 + geometric_sequence 3 q 3 = 21) →
  (q > 0) →
  (geometric_sequence 3 q 3 + geometric_sequence 3 q 4 + geometric_sequence 3 q 5 = 84) :=
by
  intros a_1 q h1 hsum hqpos
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_terms_l34_3464


namespace NUMINAMATH_GPT_max_possible_intersections_l34_3443

theorem max_possible_intersections : 
  let num_x := 12
  let num_y := 6
  let intersections := (num_x * (num_x - 1) / 2) * (num_y * (num_y - 1) / 2)
  intersections = 990 := 
by 
  sorry

end NUMINAMATH_GPT_max_possible_intersections_l34_3443


namespace NUMINAMATH_GPT_behavior_of_g_l34_3476

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 4 * x ^ 2 + 5

theorem behavior_of_g :
  (∀ x, (∃ M, x ≥ M → g x < 0)) ∧ (∀ x, (∃ N, x ≤ N → g x > 0)) :=
by
  sorry

end NUMINAMATH_GPT_behavior_of_g_l34_3476


namespace NUMINAMATH_GPT_rectangle_MQ_l34_3492

theorem rectangle_MQ :
  ∀ (PQ QR PM MQ : ℝ),
    PQ = 4 →
    QR = 10 →
    PM = MQ →
    MQ = 2 * Real.sqrt 10 → 
    0 < MQ
:= by
  intros PQ QR PM MQ h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_rectangle_MQ_l34_3492


namespace NUMINAMATH_GPT_download_time_ratio_l34_3474

-- Define the conditions of the problem
def mac_download_time : ℕ := 10
def audio_glitches : ℕ := 2 * 4
def video_glitches : ℕ := 6
def time_with_glitches : ℕ := audio_glitches + video_glitches
def time_without_glitches : ℕ := 2 * time_with_glitches
def total_time : ℕ := 82

-- Define the Windows download time as a variable
def windows_download_time : ℕ := total_time - (mac_download_time + time_with_glitches + time_without_glitches)

-- Prove the required ratio
theorem download_time_ratio : 
  (windows_download_time / mac_download_time = 3) :=
by
  -- Perform a straightforward calculation as defined in the conditions and solution steps
  sorry

end NUMINAMATH_GPT_download_time_ratio_l34_3474


namespace NUMINAMATH_GPT_root_quadratic_expression_value_l34_3401

theorem root_quadratic_expression_value (m : ℝ) (h : m^2 - m - 3 = 0) : 2023 - m^2 + m = 2020 := 
by 
  sorry

end NUMINAMATH_GPT_root_quadratic_expression_value_l34_3401


namespace NUMINAMATH_GPT_invested_sum_l34_3497

theorem invested_sum (P r : ℝ) 
  (peter_total : P + 3 * P * r = 815) 
  (david_total : P + 4 * P * r = 870) 
  : P = 650 := 
by
  sorry

end NUMINAMATH_GPT_invested_sum_l34_3497


namespace NUMINAMATH_GPT_sum_of_ages_l34_3431

-- Definitions for Robert's and Maria's current ages
variables (R M : ℕ)

-- Conditions based on the problem statement
theorem sum_of_ages
  (h1 : R = M + 8)
  (h2 : R + 5 = 3 * (M - 3)) :
  R + M = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l34_3431


namespace NUMINAMATH_GPT_ursula_purchases_total_cost_l34_3415

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end NUMINAMATH_GPT_ursula_purchases_total_cost_l34_3415


namespace NUMINAMATH_GPT_abc_sum_l34_3496

theorem abc_sum (f : ℝ → ℝ) (a b c : ℝ) :
  f (x - 2) = 2 * x^2 - 5 * x + 3 → f x = a * x^2 + b * x + c → a + b + c = 6 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_abc_sum_l34_3496


namespace NUMINAMATH_GPT_pages_revised_twice_l34_3435

theorem pages_revised_twice
  (x : ℕ)
  (h1 : ∀ x, x > 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h2 : ∀ x, x < 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h3 : 1000 + 100 + 10 * 30 = 1400) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_pages_revised_twice_l34_3435


namespace NUMINAMATH_GPT_find_ab_for_equation_l34_3479

theorem find_ab_for_equation (a b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (∃ x, x = 12 - x1 - x2) ∧ (a * x1^2 - 24 * x1 + b) / (x1^2 - 1) = x1
  ∧ (a * x2^2 - 24 * x2 + b) / (x2^2 - 1) = x2) ∧ (a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819) := sorry

end NUMINAMATH_GPT_find_ab_for_equation_l34_3479


namespace NUMINAMATH_GPT_complex_solution_l34_3448

theorem complex_solution (z : ℂ) (h : z^2 = -5 - 12 * Complex.I) :
  z = 2 - 3 * Complex.I ∨ z = -2 + 3 * Complex.I := 
sorry

end NUMINAMATH_GPT_complex_solution_l34_3448


namespace NUMINAMATH_GPT_largest_distance_between_spheres_l34_3462

theorem largest_distance_between_spheres :
  let O1 := (3, -14, 8)
  let O2 := (-9, 5, -12)
  let d := Real.sqrt ((3 + 9)^2 + (-14 - 5)^2 + (8 + 12)^2)
  let r1 := 24
  let r2 := 50
  r1 + d + r2 = Real.sqrt 905 + 74 :=
by
  intro O1 O2 d r1 r2
  sorry

end NUMINAMATH_GPT_largest_distance_between_spheres_l34_3462


namespace NUMINAMATH_GPT_b_should_pay_l34_3428

-- Definitions for the number of horses and their duration in months
def horses_of_a := 12
def months_of_a := 8

def horses_of_b := 16
def months_of_b := 9

def horses_of_c := 18
def months_of_c := 6

-- Total rent
def total_rent := 870

-- Shares in horse-months for each person
def share_of_a := horses_of_a * months_of_a
def share_of_b := horses_of_b * months_of_b
def share_of_c := horses_of_c * months_of_c

-- Total share in horse-months
def total_share := share_of_a + share_of_b + share_of_c

-- Fraction for b
def fraction_for_b := share_of_b / total_share

-- Amount b should pay
def amount_for_b := total_rent * fraction_for_b

-- Theorem to verify the amount b should pay
theorem b_should_pay : amount_for_b = 360 := by
  -- The steps of the proof would go here
  sorry

end NUMINAMATH_GPT_b_should_pay_l34_3428


namespace NUMINAMATH_GPT_candy_bag_division_l34_3454

theorem candy_bag_division (total_candy bags_candy : ℕ) (h1 : total_candy = 42) (h2 : bags_candy = 21) : 
  total_candy / bags_candy = 2 := 
by
  sorry

end NUMINAMATH_GPT_candy_bag_division_l34_3454


namespace NUMINAMATH_GPT_square_of_1008_l34_3456

theorem square_of_1008 : 1008^2 = 1016064 := 
by sorry

end NUMINAMATH_GPT_square_of_1008_l34_3456


namespace NUMINAMATH_GPT_power_mod_eq_one_l34_3489

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end NUMINAMATH_GPT_power_mod_eq_one_l34_3489


namespace NUMINAMATH_GPT_max_two_digit_times_max_one_digit_is_three_digit_l34_3453

def max_two_digit : ℕ := 99
def max_one_digit : ℕ := 9
def product := max_two_digit * max_one_digit

theorem max_two_digit_times_max_one_digit_is_three_digit :
  100 ≤ product ∧ product < 1000 :=
by
  -- Prove that the product is a three-digit number
  sorry

end NUMINAMATH_GPT_max_two_digit_times_max_one_digit_is_three_digit_l34_3453


namespace NUMINAMATH_GPT_evaluate_g_at_neg2_l34_3409

def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem evaluate_g_at_neg2 : g (-2) = -16 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg2_l34_3409


namespace NUMINAMATH_GPT_inequality_square_l34_3447

theorem inequality_square (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_square_l34_3447


namespace NUMINAMATH_GPT_max_val_a_l34_3449

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3 * x + 2)

theorem max_val_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 1, f a x ≥ 0) : a ≤ 1 := sorry

end NUMINAMATH_GPT_max_val_a_l34_3449


namespace NUMINAMATH_GPT_Emily_candies_l34_3455

theorem Emily_candies (jennifer_candies emily_candies bob_candies : ℕ) 
    (h1: jennifer_candies = 2 * emily_candies)
    (h2: jennifer_candies = 3 * bob_candies)
    (h3: bob_candies = 4) : emily_candies = 6 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_Emily_candies_l34_3455


namespace NUMINAMATH_GPT_system_of_inequalities_solutions_l34_3420

theorem system_of_inequalities_solutions (x : ℤ) :
  (3 * x - 2 ≥ 2 * x - 5) ∧ ((x / 2 - (x - 2) / 3 < 1 / 2)) →
  (x = -3 ∨ x = -2) :=
by sorry

end NUMINAMATH_GPT_system_of_inequalities_solutions_l34_3420


namespace NUMINAMATH_GPT_solve_inequality_l34_3487

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∨ a = 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ False)) ∧
  (0 < a ∧ a < 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a^2 < x ∧ x < a)) ∧
  (a < 0 ∨ a > 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a < x ∧ x < a^2)) :=
  by
    sorry

end NUMINAMATH_GPT_solve_inequality_l34_3487


namespace NUMINAMATH_GPT_problem_statement_l34_3406

-- Mathematical Definitions
def num_students : ℕ := 6
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 3

def event_A : Prop := ∃ (boyA : ℕ), boyA < num_boys
def event_B : Prop := ∃ (girlB : ℕ), girlB < num_girls

def C (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select 3 out of 6 students
def total_ways : ℕ := C num_students num_selected

-- Probability of event A
def P_A : ℚ := C (num_students - 1) (num_selected - 1) / total_ways

-- Probability of events A and B
def P_AB : ℚ := C (num_students - 2) (num_selected - 2) / total_ways

-- Conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem problem_statement : P_B_given_A = 2 / 5 := sorry

end NUMINAMATH_GPT_problem_statement_l34_3406


namespace NUMINAMATH_GPT_probability_samantha_in_sam_not_l34_3445

noncomputable def probability_in_picture_but_not (time_samantha : ℕ) (lap_samantha : ℕ) (time_sam : ℕ) (lap_sam : ℕ) : ℚ :=
  let seconds_raced := 900
  let samantha_laps := seconds_raced / time_samantha
  let sam_laps := seconds_raced / time_sam
  let start_line_samantha := (samantha_laps - (samantha_laps % 1)) * time_samantha + ((samantha_laps % 1) * lap_samantha)
  let start_line_sam := (sam_laps - (sam_laps % 1)) * time_sam + ((sam_laps % 1) * lap_sam)
  let in_picture_duration := 80
  let overlapping_time := 30
  overlapping_time / in_picture_duration

theorem probability_samantha_in_sam_not : probability_in_picture_but_not 120 60 75 25 = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_probability_samantha_in_sam_not_l34_3445


namespace NUMINAMATH_GPT_area_square_A_32_l34_3475

-- Define the areas of the squares in Figure B and Figure A and their relationship with the triangle areas
def identical_isosceles_triangles_with_squares (area_square_B : ℝ) (area_triangle_B : ℝ) (area_square_A : ℝ) (area_triangle_A : ℝ) :=
  area_triangle_B = (area_square_B / 2) * 4 ∧
  area_square_A / area_triangle_A = 4 / 9

theorem area_square_A_32 {area_square_B : ℝ} (h : area_square_B = 36) :
  identical_isosceles_triangles_with_squares area_square_B 72 32 72 :=
by
  sorry

end NUMINAMATH_GPT_area_square_A_32_l34_3475


namespace NUMINAMATH_GPT_Q_coordinates_l34_3426

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def P : Point := ⟨0, 3⟩
def R : Point := ⟨5, 0⟩

def isRectangle (A B C D : Point) : Prop :=
  -- replace this with the actual implementation of rectangle properties
  sorry

theorem Q_coordinates :
  ∃ Q : Point, isRectangle O P Q R ∧ Q.x = 5 ∧ Q.y = 3 :=
by
  -- replace this with the actual proof
  sorry

end NUMINAMATH_GPT_Q_coordinates_l34_3426


namespace NUMINAMATH_GPT_harry_basketball_points_l34_3405

theorem harry_basketball_points :
  ∃ (x y : ℕ), 
    (x < 15) ∧ 
    (y < 15) ∧ 
    (62 + x) % 11 = 0 ∧ 
    (62 + x + y) % 12 = 0 ∧ 
    (x * y = 24) :=
by
  sorry

end NUMINAMATH_GPT_harry_basketball_points_l34_3405


namespace NUMINAMATH_GPT_find_a_l34_3477

theorem find_a (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end NUMINAMATH_GPT_find_a_l34_3477


namespace NUMINAMATH_GPT_polynomial_evaluation_l34_3414

theorem polynomial_evaluation (p : Polynomial ℚ) 
  (hdeg : p.degree = 7)
  (h : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / 2^(n + 1)) : 
  p.eval 0 = 255 / 2^28 := 
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l34_3414


namespace NUMINAMATH_GPT_inequality_solution_l34_3486

theorem inequality_solution (y : ℝ) : 
  (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ (7 ≤ y ∧ y ≤ 11 ∨ -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l34_3486


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_divisible_by_120_l34_3400

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_divisible_by_120_l34_3400


namespace NUMINAMATH_GPT_compelling_quadruples_l34_3438
   
   def isCompellingQuadruple (a b c d : ℕ) : Prop :=
     1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d < b + c 

   def compellingQuadruplesCount (count : ℕ) : Prop :=
     count = 80
   
   theorem compelling_quadruples :
     ∃ count, compellingQuadruplesCount count :=
   by
     use 80
     sorry
   
end NUMINAMATH_GPT_compelling_quadruples_l34_3438


namespace NUMINAMATH_GPT_new_cost_relation_l34_3440

def original_cost (k t b : ℝ) : ℝ :=
  k * (t * b)^4

def new_cost (k t b : ℝ) : ℝ :=
  k * ((2 * b) * (0.75 * t))^4

theorem new_cost_relation (k t b : ℝ) (C : ℝ) 
  (hC : C = original_cost k t b) :
  new_cost k t b = 25.63 * C := sorry

end NUMINAMATH_GPT_new_cost_relation_l34_3440


namespace NUMINAMATH_GPT_quadratic_inequality_condition_l34_3439

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_condition_l34_3439


namespace NUMINAMATH_GPT_absolute_difference_distance_l34_3404

/-- Renaldo drove 15 kilometers, Ernesto drove 7 kilometers more than one-third of Renaldo's distance, 
Marcos drove -5 kilometers. Prove that the absolute difference between the total distances driven by 
Renaldo and Ernesto combined, and the distance driven by Marcos is 22 kilometers. -/
theorem absolute_difference_distance :
  let renaldo_distance := 15
  let ernesto_distance := 7 + (1 / 3) * renaldo_distance
  let marcos_distance := -5
  abs ((renaldo_distance + ernesto_distance) - marcos_distance) = 22 := by
  sorry

end NUMINAMATH_GPT_absolute_difference_distance_l34_3404


namespace NUMINAMATH_GPT_four_inv_mod_35_l34_3499

theorem four_inv_mod_35 : ∃ x : ℕ, 4 * x ≡ 1 [MOD 35] ∧ x = 9 := 
by 
  use 9
  sorry

end NUMINAMATH_GPT_four_inv_mod_35_l34_3499


namespace NUMINAMATH_GPT_valid_schedule_count_l34_3493

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end NUMINAMATH_GPT_valid_schedule_count_l34_3493


namespace NUMINAMATH_GPT_walking_rate_ratio_l34_3410

theorem walking_rate_ratio (R R' : ℝ) (usual_time early_time : ℝ) (H1 : usual_time = 42) (H2 : early_time = 36) 
(H3 : R * usual_time = R' * early_time) : (R' / R = 7 / 6) :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_walking_rate_ratio_l34_3410


namespace NUMINAMATH_GPT_inequality_sin_values_l34_3465

theorem inequality_sin_values :
  let a := Real.sin (-5)
  let b := Real.sin 3
  let c := Real.sin 5
  a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_inequality_sin_values_l34_3465


namespace NUMINAMATH_GPT_y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l34_3411

noncomputable def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_is_multiple_of_3 : y % 3 = 0 :=
sorry

theorem y_is_multiple_of_9 : y % 9 = 0 :=
sorry

theorem y_is_multiple_of_27 : y % 27 = 0 :=
sorry

theorem y_is_multiple_of_81 : y % 81 = 0 :=
sorry

end NUMINAMATH_GPT_y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l34_3411


namespace NUMINAMATH_GPT_math_problem_l34_3418

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := n * (n + 1)

-- Definitions for the terms a_n, b_n, c_n, and the sum T_n
def a_n (n : ℕ) (h : n ≠ 0) : ℕ := if n = 1 then 2 else 2 * n
def b_n (n : ℕ) (h : n ≠ 0) : ℕ := 2 * (3^n + 1)
def c_n (n : ℕ) (h : n ≠ 0) : ℕ := a_n n h * b_n n h / 4
def T (n : ℕ) (h : 0 < n) : ℕ := 
  (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2

-- Main theorem to establish the solution
theorem math_problem (n : ℕ) (h : n ≠ 0) : 
  S n = n * (n + 1) →
  a_n n h = 2 * n ∧ 
  b_n n h = 2 * (3^n + 1) ∧ 
  T n (Nat.pos_of_ne_zero h) = (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2 := 
by
  intros hS
  sorry

end NUMINAMATH_GPT_math_problem_l34_3418


namespace NUMINAMATH_GPT_sin_cos_values_trigonometric_expression_value_l34_3460

-- Define the conditions
variables (α : ℝ)
def point_on_terminal_side (x y : ℝ) (r : ℝ) : Prop :=
  (x = 3) ∧ (y = 4) ∧ (r = 5)

-- Define the problem statements
theorem sin_cos_values (x y r : ℝ) (h: point_on_terminal_side x y r) : 
  (Real.sin α = 4 / 5) ∧ (Real.cos α = 3 / 5) :=
sorry

theorem trigonometric_expression_value (h1: Real.sin α = 4 / 5) (h2: Real.cos α = 3 / 5) :
  (2 * Real.cos (π / 2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11 / 8 :=
sorry

end NUMINAMATH_GPT_sin_cos_values_trigonometric_expression_value_l34_3460


namespace NUMINAMATH_GPT_capacity_of_second_bucket_l34_3429

theorem capacity_of_second_bucket (c1 : ∃ (tank_capacity : ℕ), tank_capacity = 12 * 49) (c2 : ∃ (bucket_count : ℕ), bucket_count = 84) :
  ∃ (bucket_capacity : ℕ), bucket_capacity = 7 :=
by
  -- Extract the total capacity of the tank from condition 1
  obtain ⟨tank_capacity, htank⟩ := c1
  -- Extract the number of buckets from condition 2
  obtain ⟨bucket_count, hcount⟩ := c2
  -- Use the given relations to calculate the capacity of each bucket
  use tank_capacity / bucket_count
  -- Provide the necessary calculations
  sorry

end NUMINAMATH_GPT_capacity_of_second_bucket_l34_3429


namespace NUMINAMATH_GPT_rhombus_diagonals_perpendicular_l34_3491

section circumscribed_quadrilateral

variables {a b c d : ℝ}

-- Definition of a tangential quadrilateral satisfying Pitot's theorem.
def tangential_quadrilateral (a b c d : ℝ) :=
  a + c = b + d

-- Defining a rhombus in terms of its sides
def rhombus (a b c d : ℝ) :=
  a = b ∧ b = c ∧ c = d

-- The theorem we want to prove
theorem rhombus_diagonals_perpendicular
  (h : tangential_quadrilateral a b c d)
  (hr : rhombus a b c d) : 
  true := sorry

end circumscribed_quadrilateral

end NUMINAMATH_GPT_rhombus_diagonals_perpendicular_l34_3491


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l34_3427

theorem arithmetic_expression_evaluation :
  1325 + (180 / 60) * 3 - 225 = 1109 :=
by
  sorry -- To be filled with the proof steps

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l34_3427


namespace NUMINAMATH_GPT_evaluate_f_at_7_l34_3459

theorem evaluate_f_at_7 :
  (∃ f : ℕ → ℕ, (∀ x, f (2 * x + 1) = x ^ 2 - 2 * x) ∧ f 7 = 3) :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_f_at_7_l34_3459


namespace NUMINAMATH_GPT_find_missing_employee_l34_3480

-- Definitions based on the problem context
def employee_numbers : List Nat := List.range (52)
def sample_size := 4

-- The given conditions, stating that these employees are in the sample
def in_sample (x : Nat) : Prop := x = 6 ∨ x = 32 ∨ x = 45 ∨ x = 19

-- Define systematic sampling method condition
def systematic_sample (nums : List Nat) (size interval : Nat) : Prop :=
  nums = List.map (fun i => 6 + i * interval % 52) (List.range size)

-- The employees in the sample must include 6
def start_num := 6
def interval := 13
def expected_sample := [6, 19, 32, 45]

-- The Lean theorem we need to prove
theorem find_missing_employee :
  systematic_sample expected_sample sample_size interval ∧
  in_sample 6 ∧ in_sample 32 ∧ in_sample 45 →
  in_sample 19 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_employee_l34_3480


namespace NUMINAMATH_GPT_remainder_when_7n_divided_by_5_l34_3495

theorem remainder_when_7n_divided_by_5 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
  sorry

end NUMINAMATH_GPT_remainder_when_7n_divided_by_5_l34_3495


namespace NUMINAMATH_GPT_find_range_a_l34_3422

theorem find_range_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, (1 ≤ x ∧ x ≤ 2) → (2 ≤ y ∧ y ≤ 3) → (xy ≤ a*x^2 + 2*y^2)) ↔ (-1/2 ≤ a) :=
sorry

end NUMINAMATH_GPT_find_range_a_l34_3422


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l34_3413

variables {a : ℕ → ℝ}
-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the monotonically increasing condition
def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

-- Define the specific statement
theorem necessary_and_sufficient_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 < a 3 ↔ is_monotonically_increasing a) :=
by sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l34_3413


namespace NUMINAMATH_GPT_find_a2015_l34_3498

def seq (a : ℕ → ℕ) :=
  (a 1 = 1) ∧
  (a 2 = 4) ∧
  (a 3 = 9) ∧
  (∀ n, 4 ≤ n → a n = a (n-1) + a (n-2) - a (n-3))

theorem find_a2015 (a : ℕ → ℕ) (h_seq : seq a) : a 2015 = 8057 :=
sorry

end NUMINAMATH_GPT_find_a2015_l34_3498


namespace NUMINAMATH_GPT_total_songs_bought_l34_3472

def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7

theorem total_songs_bought :
  (country_albums + pop_albums) * songs_per_album = 70 := by
  sorry

end NUMINAMATH_GPT_total_songs_bought_l34_3472


namespace NUMINAMATH_GPT_solution_set_l34_3485

theorem solution_set {x : ℝ} :
  abs ((7 - x) / 4) < 3 ∧ 0 ≤ x ↔ 0 ≤ x ∧ x < 19 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l34_3485


namespace NUMINAMATH_GPT_minimum_value_of_f_l34_3478

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ y, (∀ x, f x ≥ y) ∧ y = 3 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l34_3478


namespace NUMINAMATH_GPT_jasons_shelves_l34_3432

theorem jasons_shelves (total_books : ℕ) (number_of_shelves : ℕ) (h_total_books : total_books = 315) (h_number_of_shelves : number_of_shelves = 7) : (total_books / number_of_shelves) = 45 := 
by
  sorry

end NUMINAMATH_GPT_jasons_shelves_l34_3432


namespace NUMINAMATH_GPT_find_q_l34_3436

theorem find_q (q x : ℝ) (h1 : x = 2) (h2 : q * x - 3 = 11) : q = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l34_3436


namespace NUMINAMATH_GPT_opposite_of_neg_2022_eq_2022_l34_3424

-- Define what it means to find the opposite of a number
def opposite (n : Int) : Int := -n

-- State the theorem that needs to be proved
theorem opposite_of_neg_2022_eq_2022 : opposite (-2022) = 2022 :=
by
  -- Proof would go here but we skip it with sorry
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2022_eq_2022_l34_3424


namespace NUMINAMATH_GPT_even_function_on_neg_interval_l34_3488

theorem even_function_on_neg_interval
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_incr : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → f x₁ ≤ f x₂)
  (h_min : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → 0 ≤ f x) :
  (∀ x : ℝ, -3 ≤ x → x ≤ -1 → 0 ≤ f x) ∧ (∀ x₁ x₂ : ℝ, -3 ≤ x₁ → x₁ < x₂ → x₂ ≤ -1 → f x₁ ≥ f x₂) :=
sorry

end NUMINAMATH_GPT_even_function_on_neg_interval_l34_3488


namespace NUMINAMATH_GPT_num_female_students_l34_3402

theorem num_female_students (F : ℕ) (h1: 8 * 85 + F * 92 = (8 + F) * 90) : F = 20 := 
by
  sorry

end NUMINAMATH_GPT_num_female_students_l34_3402


namespace NUMINAMATH_GPT_distinct_rectangles_l34_3430

theorem distinct_rectangles :
  ∃! (l w : ℝ), l * w = 100 ∧ l + w = 24 :=
sorry

end NUMINAMATH_GPT_distinct_rectangles_l34_3430


namespace NUMINAMATH_GPT_all_are_multiples_of_3_l34_3490

theorem all_are_multiples_of_3 :
  (123 % 3 = 0) ∧
  (234 % 3 = 0) ∧
  (345 % 3 = 0) ∧
  (456 % 3 = 0) ∧
  (567 % 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_all_are_multiples_of_3_l34_3490


namespace NUMINAMATH_GPT_sum_of_inradii_eq_height_l34_3419

variables (a b c h b1 a1 : ℝ)
variables (r r1 r2 : ℝ)

-- Assume CH is the height of the right-angled triangle ABC from the vertex of the right angle.
-- r, r1, r2 are the radii of the incircles of triangles ABC, AHC, and BHC respectively.
-- Given definitions:
-- BC = a
-- AC = b
-- AB = c
-- AH = b1
-- BH = a1
-- CH = h

-- Formulas for the radii of the respective triangles:
-- r : radius of incircle of triangle ABC = (a + b - h) / 2
-- r1 : radius of incircle of triangle AHC = (h + b1 - b) / 2
-- r2 : radius of incircle of triangle BHC = (h + a1 - a) / 2

theorem sum_of_inradii_eq_height 
  (H₁ : r = (a + b - h) / 2)
  (H₂ : r1 = (h + b1 - b) / 2) 
  (H₃ : r2 = (h + a1 - a) / 2) 
  (H₄ : b1 = b - h) 
  (H₅ : a1 = a - h) : 
  r + r1 + r2 = h :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inradii_eq_height_l34_3419


namespace NUMINAMATH_GPT_number_of_students_at_table_l34_3466

theorem number_of_students_at_table :
  ∃ (n : ℕ), n ∣ 119 ∧ (n = 7 ∨ n = 17) :=
sorry

end NUMINAMATH_GPT_number_of_students_at_table_l34_3466


namespace NUMINAMATH_GPT_linear_term_zero_implies_sum_zero_l34_3470

-- Define the condition that the product does not have a linear term
def no_linear_term (x a b : ℝ) : Prop :=
  (x + a) * (x + b) = x^2 + (a + b) * x + a * b

-- Given the condition, we need to prove that a + b = 0
theorem linear_term_zero_implies_sum_zero {a b : ℝ} (h : ∀ x : ℝ, no_linear_term x a b) : a + b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_linear_term_zero_implies_sum_zero_l34_3470


namespace NUMINAMATH_GPT_solution_system_l34_3457

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end NUMINAMATH_GPT_solution_system_l34_3457


namespace NUMINAMATH_GPT_seashells_count_l34_3483

theorem seashells_count {s : ℕ} (h : s + 6 = 25) : s = 19 :=
by
  sorry

end NUMINAMATH_GPT_seashells_count_l34_3483


namespace NUMINAMATH_GPT_functional_equation_solution_l34_3412

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), (∀ (m n : ℤ), f (m + f (f n)) = -f (f (m + 1)) - n) → (∀ (p : ℤ), f p = 1 - p) :=
by
  intro f h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l34_3412


namespace NUMINAMATH_GPT_circle_x_intercept_l34_3452

theorem circle_x_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (k1 : y1 = 2) (h2 : x2 = 11) (k2 : y2 = 8) :
  ∃ x : ℝ, (x ≠ 3) ∧ ((x - 7) ^ 2 + (0 - 5) ^ 2 = 25) ∧ (x = 7) :=
by
  sorry

end NUMINAMATH_GPT_circle_x_intercept_l34_3452


namespace NUMINAMATH_GPT_minimum_ticket_cost_l34_3444

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end NUMINAMATH_GPT_minimum_ticket_cost_l34_3444


namespace NUMINAMATH_GPT_not_inequality_l34_3469

theorem not_inequality (x : ℝ) : ¬ (x^2 + 2*x - 3 < 0) :=
sorry

end NUMINAMATH_GPT_not_inequality_l34_3469


namespace NUMINAMATH_GPT_taxi_ride_cost_l34_3451

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l34_3451


namespace NUMINAMATH_GPT_last_number_is_two_l34_3446

theorem last_number_is_two (A B C D : ℝ)
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) :
  D = 2 :=
sorry

end NUMINAMATH_GPT_last_number_is_two_l34_3446


namespace NUMINAMATH_GPT_UF_opponent_score_l34_3441

theorem UF_opponent_score 
  (total_points : ℕ)
  (games_played : ℕ)
  (previous_points_avg : ℕ)
  (championship_score : ℕ)
  (opponent_score : ℕ)
  (total_points_condition : total_points = 720)
  (games_played_condition : games_played = 24)
  (previous_points_avg_condition : previous_points_avg = total_points / games_played)
  (championship_score_condition : championship_score = previous_points_avg / 2 - 2)
  (loss_by_condition : opponent_score = championship_score - 2) :
  opponent_score = 11 :=
by
  sorry

end NUMINAMATH_GPT_UF_opponent_score_l34_3441
