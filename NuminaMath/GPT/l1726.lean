import Mathlib

namespace NUMINAMATH_GPT_original_number_l1726_172684

theorem original_number (x : ℝ) (h : x * 1.20 = 1080) : x = 900 :=
sorry

end NUMINAMATH_GPT_original_number_l1726_172684


namespace NUMINAMATH_GPT_cost_of_headphones_l1726_172671

-- Define the constants for the problem
def bus_ticket_cost : ℕ := 11
def drinks_and_snacks_cost : ℕ := 3
def wifi_cost_per_hour : ℕ := 2
def trip_hours : ℕ := 3
def earnings_per_hour : ℕ := 12
def total_earnings := earnings_per_hour * trip_hours
def total_expenses_without_headphones := bus_ticket_cost + drinks_and_snacks_cost + (wifi_cost_per_hour * trip_hours)

-- Prove the cost of headphones, H, is $16 
theorem cost_of_headphones : total_earnings = total_expenses_without_headphones + 16 := by
  -- setup the goal
  sorry

end NUMINAMATH_GPT_cost_of_headphones_l1726_172671


namespace NUMINAMATH_GPT_f_zero_f_pos_f_decreasing_solve_inequality_l1726_172661

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul_add (m n : ℝ) : f m * f n = f (m + n)
axiom f_pos_neg (x : ℝ) : x < 0 → 1 < f x

theorem f_zero : f 0 = 1 :=
sorry

theorem f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1 :=
sorry

theorem f_decreasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem solve_inequality (a x : ℝ) :
  f (x^2 - 3 * a * x + 1) * f (-3 * x + 6 * a + 1) ≥ 1 ↔
  (a > 1/3 ∧ 2 ≤ x ∧ x ≤ 3 * a + 1) ∨
  (a = 1/3 ∧ x = 2) ∨
  (a < 1/3 ∧ 3 * a + 1 ≤ x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_GPT_f_zero_f_pos_f_decreasing_solve_inequality_l1726_172661


namespace NUMINAMATH_GPT_marks_in_english_l1726_172633

theorem marks_in_english :
  let m := 35             -- Marks in Mathematics
  let p := 52             -- Marks in Physics
  let c := 47             -- Marks in Chemistry
  let b := 55             -- Marks in Biology
  let n := 5              -- Number of subjects
  let avg := 46.8         -- Average marks
  let total_marks := avg * n
  total_marks - (m + p + c + b) = 45 := sorry

end NUMINAMATH_GPT_marks_in_english_l1726_172633


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1726_172698

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1726_172698


namespace NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l1726_172607

theorem remainder_of_3_pow_2023_mod_7 : (3^2023) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l1726_172607


namespace NUMINAMATH_GPT_weight_lift_equality_l1726_172648

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end NUMINAMATH_GPT_weight_lift_equality_l1726_172648


namespace NUMINAMATH_GPT_grade_assignment_ways_l1726_172660

theorem grade_assignment_ways : (4 ^ 12) = 16777216 :=
by
  -- mathematical proof
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l1726_172660


namespace NUMINAMATH_GPT_product_of_roots_l1726_172651

theorem product_of_roots (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
    (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by sorry

end NUMINAMATH_GPT_product_of_roots_l1726_172651


namespace NUMINAMATH_GPT_sum_real_imag_parts_l1726_172639

open Complex

theorem sum_real_imag_parts (z : ℂ) (i : ℂ) (i_property : i * i = -1) (z_eq : z * i = -1 + i) :
  (z.re + z.im = 2) :=
  sorry

end NUMINAMATH_GPT_sum_real_imag_parts_l1726_172639


namespace NUMINAMATH_GPT_object_distance_traveled_l1726_172635

theorem object_distance_traveled
  (t : ℕ) (v_mph : ℝ) (mile_to_feet : ℕ)
  (h_t : t = 2)
  (h_v : v_mph = 68.18181818181819)
  (h_mile : mile_to_feet = 5280) :
  ∃ d : ℝ, d = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_object_distance_traveled_l1726_172635


namespace NUMINAMATH_GPT_find_p5_l1726_172690

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end NUMINAMATH_GPT_find_p5_l1726_172690


namespace NUMINAMATH_GPT_number_of_students_per_normal_class_l1726_172680

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_per_normal_class_l1726_172680


namespace NUMINAMATH_GPT_prop_sufficient_not_necessary_l1726_172649

-- Let p and q be simple propositions.
variables (p q : Prop)

-- Define the statement to be proved: 
-- "either p or q is false" is a sufficient but not necessary condition 
-- for "not p is true".
theorem prop_sufficient_not_necessary (hpq : ¬(p ∧ q)) : ¬ p :=
sorry

end NUMINAMATH_GPT_prop_sufficient_not_necessary_l1726_172649


namespace NUMINAMATH_GPT_ln_1_2_over_6_gt_e_l1726_172669

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end NUMINAMATH_GPT_ln_1_2_over_6_gt_e_l1726_172669


namespace NUMINAMATH_GPT_increase_in_license_plates_l1726_172619

/-- The number of old license plates and new license plates in MiraVille. -/
def old_license_plates : ℕ := 26^2 * 10^3
def new_license_plates : ℕ := 26^2 * 10^4

/-- The ratio of the number of new license plates to the number of old license plates is 10. -/
theorem increase_in_license_plates : new_license_plates / old_license_plates = 10 := by
  unfold old_license_plates new_license_plates
  sorry

end NUMINAMATH_GPT_increase_in_license_plates_l1726_172619


namespace NUMINAMATH_GPT_force_of_water_on_lock_wall_l1726_172621

noncomputable def force_on_the_wall (l h γ g : ℝ) : ℝ :=
  γ * g * l * (h^2 / 2)

theorem force_of_water_on_lock_wall :
  force_on_the_wall 20 5 1000 9.81 = 2.45 * 10^6 := by
  sorry

end NUMINAMATH_GPT_force_of_water_on_lock_wall_l1726_172621


namespace NUMINAMATH_GPT_range_of_ab_l1726_172683

noncomputable def range_ab : Set ℝ := 
  { x | 4 ≤ x ∧ x ≤ 112 / 9 }

theorem range_of_ab (a b : ℝ) 
  (q : ℝ) (h1 : q ∈ (Set.Icc (1/3) 2)) 
  (h2 : ∃ m : ℝ, ∃ nq : ℕ, 
    (m * q ^ nq) * m ^ (2 - nq) = 1 ∧ 
    (m + m * q ^ nq) = a ∧ 
    (m * q + m * q ^ 2) = b):
  ab = (q + 1/q + q^2 + 1/q^2) → 
  (ab ∈ range_ab) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_ab_l1726_172683


namespace NUMINAMATH_GPT_value_of_f_at_1_over_16_l1726_172676

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem value_of_f_at_1_over_16 (α : ℝ) (h : f 4 α = 2) : f (1 / 16) α = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_1_over_16_l1726_172676


namespace NUMINAMATH_GPT_simon_project_score_l1726_172672

-- Define the initial conditions
def num_students_before : Nat := 20
def num_students_total : Nat := 21
def avg_before : ℕ := 86
def avg_after : ℕ := 88

-- Calculate total score before Simon's addition
def total_score_before : ℕ := num_students_before * avg_before

-- Calculate total score after Simon's addition
def total_score_after : ℕ := num_students_total * avg_after

-- Definition to represent Simon's score
def simon_score : ℕ := total_score_after - total_score_before

-- Theorem that we need to prove
theorem simon_project_score : simon_score = 128 :=
by
  sorry

end NUMINAMATH_GPT_simon_project_score_l1726_172672


namespace NUMINAMATH_GPT_taxi_fare_l1726_172622

theorem taxi_fare (fare : ℕ → ℝ) (distance : ℕ) :
  (∀ d, d > 10 → fare d = 20 + (d - 10) * (140 / 70)) →
  fare 80 = 160 →
  fare 100 = 200 :=
by
  intros h_fare h_fare_80
  show fare 100 = 200
  sorry

end NUMINAMATH_GPT_taxi_fare_l1726_172622


namespace NUMINAMATH_GPT_baked_by_brier_correct_l1726_172659

def baked_by_macadams : ℕ := 20
def baked_by_flannery : ℕ := 17
def total_baked : ℕ := 55

def baked_by_brier : ℕ := total_baked - (baked_by_macadams + baked_by_flannery)

-- Theorem statement
theorem baked_by_brier_correct : baked_by_brier = 18 := 
by
  -- proof will go here 
  sorry

end NUMINAMATH_GPT_baked_by_brier_correct_l1726_172659


namespace NUMINAMATH_GPT_sin_product_identity_sin_cos_fraction_identity_l1726_172650

-- First Proof Problem: Proving that the product of sines equals the given value
theorem sin_product_identity :
  (Real.sin (Real.pi * 6 / 180) * 
   Real.sin (Real.pi * 42 / 180) * 
   Real.sin (Real.pi * 66 / 180) * 
   Real.sin (Real.pi * 78 / 180)) = 
  (Real.sqrt 5 - 1) / 32 := 
by 
  sorry

-- Second Proof Problem: Given sin alpha and alpha in the second quadrant, proving the given fraction value
theorem sin_cos_fraction_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + Real.pi / 4)) / 
  (Real.sin (2 * α) + Real.cos (2 * α) + 1) = 
  -Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_sin_product_identity_sin_cos_fraction_identity_l1726_172650


namespace NUMINAMATH_GPT_greatest_sum_of_other_two_roots_l1726_172675

noncomputable def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^3 - k * x^2 + 20 * x - 15

theorem greatest_sum_of_other_two_roots (k x1 x2 : ℝ) (h : polynomial 3 k = 0) (hx : x1 * x2 = 5)
  (h_prod_sum : 3 * x1 + 3 * x2 + x1 * x2 = 20) : x1 + x2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_sum_of_other_two_roots_l1726_172675


namespace NUMINAMATH_GPT_retirement_total_l1726_172603

/-- A company retirement plan allows an employee to retire when their age plus years of employment total a specific number.
A female employee was hired in 1990 on her 32nd birthday. She could first be eligible to retire under this provision in 2009. -/
def required_total_age_years_of_employment : ℕ :=
  let hire_year := 1990
  let retirement_year := 2009
  let age_when_hired := 32
  let years_of_employment := retirement_year - hire_year
  let age_at_retirement := age_when_hired + years_of_employment
  age_at_retirement + years_of_employment

theorem retirement_total :
  required_total_age_years_of_employment = 70 :=
by
  sorry

end NUMINAMATH_GPT_retirement_total_l1726_172603


namespace NUMINAMATH_GPT_highest_lowest_difference_l1726_172632

variable (x1 x2 x3 x4 x5 x_max x_min : ℝ)

theorem highest_lowest_difference (h1 : x1 + x2 + x3 + x4 + x5 - x_max = 37.84)
                                  (h2 : x1 + x2 + x3 + x4 + x5 - x_min = 38.64):
                                  x_max - x_min = 0.8 := 
by
  sorry

end NUMINAMATH_GPT_highest_lowest_difference_l1726_172632


namespace NUMINAMATH_GPT_compute_expression_l1726_172617

theorem compute_expression : 1005^2 - 995^2 - 1003^2 + 997^2 = 8000 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1726_172617


namespace NUMINAMATH_GPT_range_of_m_l1726_172646

def G (x y : ℤ) : ℤ :=
  if x ≥ y then x - y
  else y - x

theorem range_of_m (m : ℤ) :
  (∀ x, 0 < x → G x 1 > 4 → G (-1) x ≤ m) ↔ 9 ≤ m ∧ m < 10 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1726_172646


namespace NUMINAMATH_GPT_gcd_polynomials_l1726_172613

noncomputable def b : ℤ := sorry -- since b is given as an odd multiple of 997

theorem gcd_polynomials (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_polynomials_l1726_172613


namespace NUMINAMATH_GPT_walking_speed_l1726_172682

-- Define the constants and variables
def speed_there := 25 -- speed from village to post-office in kmph
def total_time := 5.8 -- total round trip time in hours
def distance := 20.0 -- distance to the post-office in km
 
-- Define the theorem that needs to be proved
theorem walking_speed :
  ∃ (speed_back : ℝ), speed_back = 4 := 
by
  sorry

end NUMINAMATH_GPT_walking_speed_l1726_172682


namespace NUMINAMATH_GPT_sum_pattern_l1726_172687

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_pattern_l1726_172687


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1726_172699

-- System (1)
theorem system1_solution (x y : ℝ) (h1 : x + y = 1) (h2 : 3 * x + y = 5) : x = 2 ∧ y = -1 := sorry

-- System (2)
theorem system2_solution (x y : ℝ) (h1 : 3 * (x - 1) + 4 * y = 1) (h2 : 2 * x + 3 * (y + 1) = 2) : x = 16 ∧ y = -11 := sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1726_172699


namespace NUMINAMATH_GPT_premium_percentage_on_shares_l1726_172679

theorem premium_percentage_on_shares
    (investment : ℕ)
    (share_price : ℕ)
    (premium_percentage : ℕ)
    (dividend_percentage : ℕ)
    (total_dividend : ℕ)
    (number_of_shares : ℕ)
    (investment_eq : investment = number_of_shares * (share_price + premium_percentage))
    (dividend_eq : total_dividend = number_of_shares * (share_price * dividend_percentage / 100))
    (investment_val : investment = 14400)
    (share_price_val : share_price = 100)
    (dividend_percentage_val : dividend_percentage = 5)
    (total_dividend_val : total_dividend = 600)
    (number_of_shares_val : number_of_shares = 600 / 5) :
    premium_percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_premium_percentage_on_shares_l1726_172679


namespace NUMINAMATH_GPT_total_columns_l1726_172694

variables (N L : ℕ)

theorem total_columns (h1 : L > 1500) (h2 : L = 30 * (N - 70)) : N = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_columns_l1726_172694


namespace NUMINAMATH_GPT_balance_balls_l1726_172681

variable {R Y B W : ℕ}

theorem balance_balls (h1 : 4 * R = 8 * B) 
                      (h2 : 3 * Y = 9 * B) 
                      (h3 : 5 * B = 3 * W) : 
    (2 * R + 4 * Y + 3 * W) = 21 * B :=
by 
  sorry

end NUMINAMATH_GPT_balance_balls_l1726_172681


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_99_l1726_172658

theorem last_three_digits_of_7_pow_99 : (7 ^ 99) % 1000 = 573 := 
by sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_99_l1726_172658


namespace NUMINAMATH_GPT_lateral_surface_area_of_rotated_square_l1726_172656

noncomputable def lateralSurfaceAreaOfRotatedSquare (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length * side_length

theorem lateral_surface_area_of_rotated_square :
  lateralSurfaceAreaOfRotatedSquare 1 = 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_rotated_square_l1726_172656


namespace NUMINAMATH_GPT_general_formula_of_geometric_seq_term_in_arithmetic_seq_l1726_172689

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Condition: Geometric sequence {a_n} with a_1 = 2 and a_4 = 16
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q

-- General formula for the sequence {a_n}
theorem general_formula_of_geometric_seq 
  (ha : geometric_seq a) (h1 : a 1 = 2) (h4 : a 4 = 16) :
  ∀ n, a n = 2^n :=
sorry

-- Condition: Arithmetic sequence {b_n} with b_3 = a_3 and b_5 = a_5
def arithmetic_seq (b : ℕ → ℝ) := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d

-- Check if a_9 is a term in the sequence {b_n} and find its term number
theorem term_in_arithmetic_seq 
  (ha : geometric_seq a) (hb : arithmetic_seq b)
  (h1 : a 1 = 2) (h4 : a 4 = 16)
  (hb3 : b 3 = a 3) (hb5 : b 5 = a 5) :
  ∃ n, b n = a 9 ∧ n = 45 :=
sorry

end NUMINAMATH_GPT_general_formula_of_geometric_seq_term_in_arithmetic_seq_l1726_172689


namespace NUMINAMATH_GPT_speed_in_still_water_l1726_172692

variable (upstream downstream : ℝ)

-- Conditions
def upstream_speed : Prop := upstream = 26
def downstream_speed : Prop := downstream = 40

-- Question and correct answer
theorem speed_in_still_water (h1 : upstream_speed upstream) (h2 : downstream_speed downstream) :
  (upstream + downstream) / 2 = 33 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1726_172692


namespace NUMINAMATH_GPT_inequality_solution_l1726_172686

theorem inequality_solution (x : ℝ) :
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1726_172686


namespace NUMINAMATH_GPT_Matt_jumped_for_10_minutes_l1726_172604

def Matt_skips_per_second : ℕ := 3

def total_skips : ℕ := 1800

def minutes_jumped (m : ℕ) : Prop :=
  m * (Matt_skips_per_second * 60) = total_skips

theorem Matt_jumped_for_10_minutes : minutes_jumped 10 :=
by
  sorry

end NUMINAMATH_GPT_Matt_jumped_for_10_minutes_l1726_172604


namespace NUMINAMATH_GPT_largest_p_q_sum_l1726_172611

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end NUMINAMATH_GPT_largest_p_q_sum_l1726_172611


namespace NUMINAMATH_GPT_average_weight_l1726_172636

/-- 
Given the following conditions:
1. (A + B) / 2 = 40
2. (B + C) / 2 = 41
3. B = 27
Prove that the average weight of a, b, and c is 45 kg.
-/
theorem average_weight (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27): 
  (A + B + C) / 3 = 45 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_l1726_172636


namespace NUMINAMATH_GPT_least_number_subtracted_to_divisible_by_10_l1726_172642

def least_subtract_to_divisible_by_10 (n : ℕ) : ℕ :=
  let last_digit := n % 10
  10 - last_digit

theorem least_number_subtracted_to_divisible_by_10 (n : ℕ) : (n = 427751) → ((n - least_subtract_to_divisible_by_10 n) % 10 = 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_least_number_subtracted_to_divisible_by_10_l1726_172642


namespace NUMINAMATH_GPT_number_of_quadratic_PQ_equal_to_PR_l1726_172670

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, Q = λ x => a * x^2 + b * x + c

theorem number_of_quadratic_PQ_equal_to_PR :
  let possible_Qx_fwds := 4^4
  let non_quadratic_cases := 6
  possible_Qx_fwds - non_quadratic_cases = 250 :=
by
  sorry

end NUMINAMATH_GPT_number_of_quadratic_PQ_equal_to_PR_l1726_172670


namespace NUMINAMATH_GPT_total_distance_is_105_km_l1726_172627

-- Define the boat's speed in still water
def boat_speed_still_water : ℝ := 50

-- Define the current speeds for each hour
def current_speed_first_hour : ℝ := 10
def current_speed_second_hour : ℝ := 20
def current_speed_third_hour : ℝ := 15

-- Calculate the effective speeds for each hour
def effective_speed_first_hour := boat_speed_still_water - current_speed_first_hour
def effective_speed_second_hour := boat_speed_still_water - current_speed_second_hour
def effective_speed_third_hour := boat_speed_still_water - current_speed_third_hour

-- Calculate the distance traveled in each hour
def distance_first_hour := effective_speed_first_hour * 1
def distance_second_hour := effective_speed_second_hour * 1
def distance_third_hour := effective_speed_third_hour * 1

-- Define the total distance
def total_distance_traveled := distance_first_hour + distance_second_hour + distance_third_hour

-- Prove that the total distance traveled is 105 km
theorem total_distance_is_105_km : total_distance_traveled = 105 := by
  sorry

end NUMINAMATH_GPT_total_distance_is_105_km_l1726_172627


namespace NUMINAMATH_GPT_contractor_realized_work_done_after_20_days_l1726_172630

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end NUMINAMATH_GPT_contractor_realized_work_done_after_20_days_l1726_172630


namespace NUMINAMATH_GPT_solve_fractional_equation_l1726_172674

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (2 * x) / (x - 3) = 1 ↔ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1726_172674


namespace NUMINAMATH_GPT_no_solution_5x_plus_2_eq_17y_l1726_172623

theorem no_solution_5x_plus_2_eq_17y :
  ¬∃ (x y : ℕ), 5^x + 2 = 17^y :=
sorry

end NUMINAMATH_GPT_no_solution_5x_plus_2_eq_17y_l1726_172623


namespace NUMINAMATH_GPT_tangent_points_sum_constant_l1726_172605

theorem tangent_points_sum_constant 
  (a : ℝ) (x1 y1 x2 y2 : ℝ)
  (hC1 : x1^2 = 4 * y1)
  (hC2 : x2^2 = 4 * y2)
  (hT1 : y1 - (-2) = (1/2)*x1*(x1 - a))
  (hT2 : y2 - (-2) = (1/2)*x2*(x2 - a)) :
  x1 * x2 + y1 * y2 = -4 :=
sorry

end NUMINAMATH_GPT_tangent_points_sum_constant_l1726_172605


namespace NUMINAMATH_GPT_profit_percentage_is_twenty_percent_l1726_172624

def selling_price : ℕ := 900
def profit : ℕ := 150
def cost_price : ℕ := selling_price - profit
def profit_percentage : ℕ := (profit * 100) / cost_price

theorem profit_percentage_is_twenty_percent : profit_percentage = 20 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_twenty_percent_l1726_172624


namespace NUMINAMATH_GPT_candies_count_l1726_172654

theorem candies_count (x : ℚ) (h : x + 3 * x + 12 * x + 72 * x = 468) : x = 117 / 22 :=
by
  sorry

end NUMINAMATH_GPT_candies_count_l1726_172654


namespace NUMINAMATH_GPT_auditorium_rows_l1726_172666

noncomputable def rows_in_auditorium : Nat :=
  let class1 := 30
  let class2 := 26
  let condition1 := ∃ row : Nat, row < class1 ∧ ∀ students_per_row : Nat, students_per_row ≤ row 
  let condition2 := ∃ empty_rows : Nat, empty_rows ≥ 3 ∧ ∀ students : Nat, students = class2 - empty_rows
  29

theorem auditorium_rows (n : Nat) (class1 : Nat) (class2 : Nat) (c1 : class1 ≥ n) (c2 : class2 ≤ n - 3)
  : n = 29 :=
by
  sorry

end NUMINAMATH_GPT_auditorium_rows_l1726_172666


namespace NUMINAMATH_GPT_symmetric_line_eq_l1726_172610

theorem symmetric_line_eq : ∀ (x y : ℝ), (x - 2*y - 1 = 0) ↔ (2*x - y + 1 = 0) :=
by sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1726_172610


namespace NUMINAMATH_GPT_discriminant_zero_no_harmonic_progression_l1726_172647

theorem discriminant_zero_no_harmonic_progression (a b c : ℝ) 
    (h_disc : b^2 = 24 * a * c) : 
    ¬ (2 * (1 / b) = (1 / a) + (1 / c)) := 
sorry

end NUMINAMATH_GPT_discriminant_zero_no_harmonic_progression_l1726_172647


namespace NUMINAMATH_GPT_scientific_notation_3050000_l1726_172652

def scientific_notation (n : ℕ) : String :=
  "3.05 × 10^6"

theorem scientific_notation_3050000 :
  scientific_notation 3050000 = "3.05 × 10^6" :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_3050000_l1726_172652


namespace NUMINAMATH_GPT_closest_total_population_of_cities_l1726_172640

theorem closest_total_population_of_cities 
    (n_cities : ℕ) (avg_population_lower avg_population_upper : ℕ)
    (h_lower : avg_population_lower = 3800) (h_upper : avg_population_upper = 4200) :
  (25:ℕ) * (4000:ℕ) = 100000 :=
by
  sorry

end NUMINAMATH_GPT_closest_total_population_of_cities_l1726_172640


namespace NUMINAMATH_GPT_triangle_strike_interval_l1726_172663

/-- Jacob strikes the cymbals every 7 beats and the triangle every t beats.
    Given both are struck at the same time every 14 beats, this proves t = 2. -/
theorem triangle_strike_interval :
  ∃ t : ℕ, t ≠ 7 ∧ (∀ n : ℕ, (7 * n % t = 0) → ∃ k : ℕ, 7 * n = 14 * k) ∧ t = 2 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_triangle_strike_interval_l1726_172663


namespace NUMINAMATH_GPT_unique_function_satisfying_condition_l1726_172628

theorem unique_function_satisfying_condition (k : ℕ) (hk : 0 < k) :
  ∀ f : ℕ → ℕ, (∀ m n : ℕ, 0 < m → 0 < n → f m + f n ∣ (m + n) ^ k) →
  ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
by
  sorry

end NUMINAMATH_GPT_unique_function_satisfying_condition_l1726_172628


namespace NUMINAMATH_GPT_water_consumption_and_bill_34_7_l1726_172688

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end NUMINAMATH_GPT_water_consumption_and_bill_34_7_l1726_172688


namespace NUMINAMATH_GPT_airplane_speed_l1726_172608

noncomputable def distance : ℝ := 378.6   -- Distance in km
noncomputable def time : ℝ := 693.5       -- Time in seconds

noncomputable def altitude : ℝ := 10      -- Altitude in km
noncomputable def earth_radius : ℝ := 6370 -- Earth's radius in km

noncomputable def speed : ℝ := distance / time * 3600  -- Speed in km/h
noncomputable def adjusted_speed : ℝ := speed * (earth_radius + altitude) / earth_radius

noncomputable def min_distance : ℝ := 378.6 - 0.03     -- Minimum possible distance in km
noncomputable def max_distance : ℝ := 378.6 + 0.03     -- Maximum possible distance in km
noncomputable def min_time : ℝ := 693.5 - 1.5          -- Minimum possible time in s
noncomputable def max_time : ℝ := 693.5 + 1.5          -- Maximum possible time in s

noncomputable def max_speed : ℝ := max_distance / min_time * 3600 -- Max speed with uncertainty
noncomputable def min_speed : ℝ := min_distance / max_time * 3600 -- Min speed with uncertainty

theorem airplane_speed :
  1960 < adjusted_speed ∧ adjusted_speed < 1970 :=
by
  sorry

end NUMINAMATH_GPT_airplane_speed_l1726_172608


namespace NUMINAMATH_GPT_pencils_to_sell_for_profit_l1726_172620

theorem pencils_to_sell_for_profit 
    (total_pencils : ℕ) 
    (buy_price sell_price : ℝ) 
    (desired_profit : ℝ) 
    (h_total_pencils : total_pencils = 2000) 
    (h_buy_price : buy_price = 0.15) 
    (h_sell_price : sell_price = 0.30) 
    (h_desired_profit : desired_profit = 150) :
    total_pencils * buy_price + desired_profit = total_pencils * sell_price → total_pencils = 1500 :=
by
    sorry

end NUMINAMATH_GPT_pencils_to_sell_for_profit_l1726_172620


namespace NUMINAMATH_GPT_perfect_square_l1726_172653

variables {n x k ℓ : ℕ}

theorem perfect_square (h1 : x^2 < n) (h2 : n < (x + 1)^2)
  (h3 : k = n - x^2) (h4 : ℓ = (x + 1)^2 - n) :
  ∃ m : ℕ, n - k * ℓ = m^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_l1726_172653


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l1726_172638

variable {n : ℕ}

def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := n * (2 + (2 + (n - 1) * (3 - 1))) / 2 -- sum of an arithmetic sequence
def T (n : ℕ) : ℕ := (3 * n - 4) * 2 ^ (n + 1) + 8

theorem arithmetic_geometric_sequence_problem :
  (a 1 = 2) ∧ (b 1 = 2) ∧ (a 4 + b 4 = 27) ∧ (S 4 - b 4 = 10) →
  (∀ n, T n = (3 * n - 4) * 2 ^ (n + 1) + 8) := sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_problem_l1726_172638


namespace NUMINAMATH_GPT_product_of_values_l1726_172615

-- Given definitions: N as a real number and R as a real constant
variables (N R : ℝ)

-- Condition
def condition : Prop := N - 5 / N = R

-- The proof statement
theorem product_of_values (h : condition N R) : ∀ (N1 N2 : ℝ), ((N1 - 5 / N1 = R) ∧ (N2 - 5 / N2 = R)) → (N1 * N2 = -5) :=
by sorry

end NUMINAMATH_GPT_product_of_values_l1726_172615


namespace NUMINAMATH_GPT_infinite_danish_numbers_l1726_172637

-- Definitions translated from problem conditions
def is_danish (n : ℕ) : Prop :=
  ∃ k, n = 3 * k ∨ n = 2 * 4 ^ k

theorem infinite_danish_numbers :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, is_danish n ∧ is_danish (2^n + n) := sorry

end NUMINAMATH_GPT_infinite_danish_numbers_l1726_172637


namespace NUMINAMATH_GPT_yeast_population_at_1_20_pm_l1726_172618

def yeast_population (initial : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  initial * rate^time

theorem yeast_population_at_1_20_pm : 
  yeast_population 50 3 4 = 4050 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_yeast_population_at_1_20_pm_l1726_172618


namespace NUMINAMATH_GPT_ads_minutes_l1726_172600

-- Definitions and conditions
def videos_per_day : Nat := 2
def minutes_per_video : Nat := 7
def total_time_on_youtube : Nat := 17

-- The theorem to prove
theorem ads_minutes : (total_time_on_youtube - (videos_per_day * minutes_per_video)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ads_minutes_l1726_172600


namespace NUMINAMATH_GPT_smallest_n_divisible_31_l1726_172616

theorem smallest_n_divisible_31 (n : ℕ) : 31 ∣ (5 ^ n + n) → n = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_31_l1726_172616


namespace NUMINAMATH_GPT_base12_mod_9_remainder_l1726_172668

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 7 * 12^2 + 3 * 12^1 + 2 * 12^0

theorem base12_mod_9_remainder : (base12_to_base10 1732) % 9 = 2 := by
  sorry

end NUMINAMATH_GPT_base12_mod_9_remainder_l1726_172668


namespace NUMINAMATH_GPT_instantaneous_speed_at_4_l1726_172629

def motion_equation (t : ℝ) : ℝ := t^2 - 2 * t + 5

theorem instantaneous_speed_at_4 :
  (deriv motion_equation 4) = 6 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_speed_at_4_l1726_172629


namespace NUMINAMATH_GPT_set_representation_l1726_172697

def A (x : ℝ) := -3 < x ∧ x < 1
def B (x : ℝ) := x ≤ -1
def C (x : ℝ) := -2 < x ∧ x ≤ 2

theorem set_representation :
  (∀ x, A x ↔ (A x ∧ (B x ∨ C x))) ∧
  (∀ x, A x ↔ (A x ∨ (B x ∧ C x))) ∧
  (∀ x, A x ↔ ((A x ∧ B x) ∨ (A x ∧ C x))) :=
by
  sorry

end NUMINAMATH_GPT_set_representation_l1726_172697


namespace NUMINAMATH_GPT_number_of_dolls_of_jane_l1726_172602

-- Given conditions
def total_dolls (J D : ℕ) := J + D = 32
def jill_has_more (J D : ℕ) := D = J + 6

-- Statement to prove
theorem number_of_dolls_of_jane (J D : ℕ) (h1 : total_dolls J D) (h2 : jill_has_more J D) : J = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dolls_of_jane_l1726_172602


namespace NUMINAMATH_GPT_remainder_mod_29_l1726_172693

-- Definitions of the given conditions
def N (k : ℕ) := 899 * k + 63

-- The proof statement to be proved
theorem remainder_mod_29 (k : ℕ) : (N k) % 29 = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_mod_29_l1726_172693


namespace NUMINAMATH_GPT_total_boys_in_groups_l1726_172662

-- Definitions of number of groups
def total_groups : ℕ := 35
def groups_with_1_boy : ℕ := 10
def groups_with_at_least_2_boys : ℕ := 19
def groups_with_3_boys_twice_groups_with_3_girls (groups_with_3_boys groups_with_3_girls : ℕ) : Prop :=
  groups_with_3_boys = 2 * groups_with_3_girls

theorem total_boys_in_groups :
  ∃ (groups_with_3_girls groups_with_3_boys groups_with_1_girl_2_boys : ℕ),
    groups_with_1_boy + groups_with_at_least_2_boys + groups_with_3_girls = total_groups
    ∧ groups_with_3_boys_twice_groups_with_3_girls groups_with_3_boys groups_with_3_girls
    ∧ groups_with_1_girl_2_boys + groups_with_3_boys = groups_with_at_least_2_boys
    ∧ (groups_with_1_boy * 1 + groups_with_1_girl_2_boys * 2 + groups_with_3_boys * 3) = 60 :=
sorry

end NUMINAMATH_GPT_total_boys_in_groups_l1726_172662


namespace NUMINAMATH_GPT_bottle_caps_total_l1726_172677

-- Mathematical conditions
def x : ℕ := 18
def y : ℕ := 63

-- Statement to prove
theorem bottle_caps_total : x + y = 81 :=
by
  -- The proof is skipped as indicated by 'sorry'
  sorry

end NUMINAMATH_GPT_bottle_caps_total_l1726_172677


namespace NUMINAMATH_GPT_volume_to_surface_area_ratio_l1726_172609

-- Define the shape as described in the problem
structure Shape :=
(center_cube : ℕ)  -- Center cube
(surrounding_cubes : ℕ)  -- Surrounding cubes
(unit_volume : ℕ)  -- Volume of each unit cube
(unit_face_area : ℕ)  -- Surface area of each face of the unit cube

-- Conditions and definitions
def is_special_shape (s : Shape) : Prop :=
  s.center_cube = 1 ∧ s.surrounding_cubes = 7 ∧ s.unit_volume = 1 ∧ s.unit_face_area = 1

-- Theorem statement
theorem volume_to_surface_area_ratio (s : Shape) (h : is_special_shape s) : (s.center_cube + s.surrounding_cubes) * s.unit_volume / (s.surrounding_cubes * 5 * s.unit_face_area) = 8 / 35 :=
by
  sorry

end NUMINAMATH_GPT_volume_to_surface_area_ratio_l1726_172609


namespace NUMINAMATH_GPT_wonderland_cities_l1726_172667

theorem wonderland_cities (V E B : ℕ) (hE : E = 45) (hB : B = 42) (h_connected : connected_graph) (h_simple : simple_graph) (h_bridges : count_bridges = 42) : V = 45 :=
sorry

end NUMINAMATH_GPT_wonderland_cities_l1726_172667


namespace NUMINAMATH_GPT_range_of_a_l1726_172685

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a^x₁ = x₁ ∧ a^x₂ = x₂

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : has_two_distinct_real_roots a) : 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1726_172685


namespace NUMINAMATH_GPT_jack_handing_in_amount_l1726_172614

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end NUMINAMATH_GPT_jack_handing_in_amount_l1726_172614


namespace NUMINAMATH_GPT_number_exceeds_its_part_by_20_l1726_172606

theorem number_exceeds_its_part_by_20 (x : ℝ) (h : x = (3/8) * x + 20) : x = 32 :=
sorry

end NUMINAMATH_GPT_number_exceeds_its_part_by_20_l1726_172606


namespace NUMINAMATH_GPT_assisted_work_time_l1726_172665

theorem assisted_work_time (a b c : ℝ) (ha : a = 1 / 11) (hb : b = 1 / 20) (hc : c = 1 / 55) :
  (1 / ((a + b) + (a + c) / 2)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_assisted_work_time_l1726_172665


namespace NUMINAMATH_GPT_sum_first_8_terms_of_geom_seq_l1726_172655

-- Definitions: the sequence a_n, common ratio q, and the fact that specific terms form an arithmetic sequence.
def geom_seq (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) := ∀ n, a n = a1 * q^(n-1)
def arith_seq (b c d : ℕ) := 2 * b + (c - 2 * b) = d

-- Conditions
variables {a : ℕ → ℕ} {a1 : ℕ} {q : ℕ}
variables (h1 : geom_seq a a1 q) (h2 : q = 2)
variables (h3 : arith_seq (2 * a 4) (a 6) 48)

-- Goal: sum of the first 8 terms of the sequence equals 255
def sum_geometric_sequence (a1 : ℕ) (q : ℕ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem sum_first_8_terms_of_geom_seq : 
  sum_geometric_sequence a1 q 8 = 255 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_8_terms_of_geom_seq_l1726_172655


namespace NUMINAMATH_GPT_possible_value_of_b_l1726_172678

-- Definition of the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Condition for the linear function to pass through the second, third, and fourth quadrants
def passes_second_third_fourth_quadrants (b : ℝ) : Prop :=
  b < 0

-- Lean 4 statement expressing the problem
theorem possible_value_of_b (b : ℝ) (h : passes_second_third_fourth_quadrants b) : b = -1 :=
  sorry

end NUMINAMATH_GPT_possible_value_of_b_l1726_172678


namespace NUMINAMATH_GPT_gcd_6724_13104_l1726_172631

theorem gcd_6724_13104 : Int.gcd 6724 13104 = 8 := 
sorry

end NUMINAMATH_GPT_gcd_6724_13104_l1726_172631


namespace NUMINAMATH_GPT_sports_parade_children_l1726_172601

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end NUMINAMATH_GPT_sports_parade_children_l1726_172601


namespace NUMINAMATH_GPT_oxygen_part_weight_l1726_172634

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O
def given_molecular_weight : ℝ := 108

theorem oxygen_part_weight : molecular_weight_N2O = 44.02 → atomic_weight_O = 16.00 := by
  sorry

end NUMINAMATH_GPT_oxygen_part_weight_l1726_172634


namespace NUMINAMATH_GPT_average_hidden_primes_l1726_172657

theorem average_hidden_primes (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : 44 + x = 59 + y ∧ 59 + y = 38 + z) :
  (x + y + z) / 3 = 14 := 
by
  sorry

end NUMINAMATH_GPT_average_hidden_primes_l1726_172657


namespace NUMINAMATH_GPT_value_of_expression_l1726_172691

theorem value_of_expression (p q r s : ℝ) (h : -27 * p + 9 * q - 3 * r + s = -7) : 
  4 * p - 2 * q + r - s = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1726_172691


namespace NUMINAMATH_GPT_max_value_of_cubes_l1726_172695

theorem max_value_of_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 + ab + ac + ad + bc + bd + cd = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_max_value_of_cubes_l1726_172695


namespace NUMINAMATH_GPT_number_of_carbon_atoms_l1726_172625

/-- A proof to determine the number of carbon atoms in a compound given specific conditions
-/
theorem number_of_carbon_atoms
  (H_atoms : ℕ) (O_atoms : ℕ) (C_weight : ℕ) (H_weight : ℕ) (O_weight : ℕ) (Molecular_weight : ℕ) :
  H_atoms = 6 →
  O_atoms = 1 →
  C_weight = 12 →
  H_weight = 1 →
  O_weight = 16 →
  Molecular_weight = 58 →
  (Molecular_weight - (H_atoms * H_weight + O_atoms * O_weight)) / C_weight = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_number_of_carbon_atoms_l1726_172625


namespace NUMINAMATH_GPT_oliver_used_fraction_l1726_172612

variable (x : ℚ)

/--
Oliver had 135 stickers. He used a fraction x of his stickers, gave 2/5 of the remaining to his friend, and kept the remaining 54 stickers. Prove that he used 1/3 of his stickers.
-/
theorem oliver_used_fraction (h : 135 - (135 * x) - (2 / 5) * (135 - 135 * x) = 54) : 
  x = 1 / 3 := 
sorry

end NUMINAMATH_GPT_oliver_used_fraction_l1726_172612


namespace NUMINAMATH_GPT_simplify_nested_fourth_roots_l1726_172673

variable (M : ℝ)
variable (hM : M > 1)

theorem simplify_nested_fourth_roots : 
  (M^(1/4) * (M^(1/4) * (M^(1/4) * M)^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end NUMINAMATH_GPT_simplify_nested_fourth_roots_l1726_172673


namespace NUMINAMATH_GPT_cos_shifted_eq_l1726_172645

noncomputable def cos_shifted (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) : Real :=
  Real.cos (theta + Real.pi / 4)

theorem cos_shifted_eq (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos_shifted theta h1 h2 = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end NUMINAMATH_GPT_cos_shifted_eq_l1726_172645


namespace NUMINAMATH_GPT_tank_fraction_l1726_172643

theorem tank_fraction (x : ℚ) (h₁ : 48 * x + 8 = 48 * (9 / 10)) : x = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_tank_fraction_l1726_172643


namespace NUMINAMATH_GPT_right_triangle_side_length_l1726_172696

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 12) 
  (h_right : a^2 + b^2 = c^2) : 
  b = Real.sqrt 119 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_side_length_l1726_172696


namespace NUMINAMATH_GPT_area_of_triangle_AMN_l1726_172626

theorem area_of_triangle_AMN
  (α : ℝ) -- Angle at vertex A
  (S : ℝ) -- Area of triangle ABC
  (area_AMN_eq : ∀ (α : ℝ) (S : ℝ), ∃ (area_AMN : ℝ), area_AMN = S * (Real.cos α)^2) :
  ∃ area_AMN, area_AMN = S * (Real.cos α)^2 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_AMN_l1726_172626


namespace NUMINAMATH_GPT_depth_of_first_hole_l1726_172644

-- Conditions as definitions in Lean 4
def number_of_workers_first_hole : Nat := 45
def hours_worked_first_hole : Nat := 8

def number_of_workers_second_hole : Nat := 110  -- 45 existing workers + 65 extra workers
def hours_worked_second_hole : Nat := 6
def depth_second_hole : Nat := 55

-- The key assumption that work done (W) is proportional to the depth of the hole (D)
theorem depth_of_first_hole :
  let work_first_hole := number_of_workers_first_hole * hours_worked_first_hole
  let work_second_hole := number_of_workers_second_hole * hours_worked_second_hole
  let depth_first_hole := (work_first_hole * depth_second_hole) / work_second_hole
  depth_first_hole = 30 := sorry

end NUMINAMATH_GPT_depth_of_first_hole_l1726_172644


namespace NUMINAMATH_GPT_monotonically_increasing_sequence_b_bounds_l1726_172664

theorem monotonically_increasing_sequence_b_bounds (b : ℝ) :
  (∀ n : ℕ, 0 < n → (n + 1)^2 + b * (n + 1) > n^2 + b * n) ↔ b > -3 :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_sequence_b_bounds_l1726_172664


namespace NUMINAMATH_GPT_complement_A_in_U_l1726_172641

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} :=
by 
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l1726_172641
