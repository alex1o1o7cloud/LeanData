import Mathlib

namespace NUMINAMATH_GPT_quadratic_inequality_real_solutions_l761_76199

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_real_solutions_l761_76199


namespace NUMINAMATH_GPT_factor_expression_l761_76129

variable (x : ℤ)

theorem factor_expression : 63 * x - 21 = 21 * (3 * x - 1) := 
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l761_76129


namespace NUMINAMATH_GPT_minimize_fraction_l761_76187

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) ↔ (∀ m : ℕ, 0 < m → (n / 3 + 27 / n) ≤ (m / 3 + 27 / m)) :=
by
  sorry

end NUMINAMATH_GPT_minimize_fraction_l761_76187


namespace NUMINAMATH_GPT_distance_is_30_l761_76188

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end NUMINAMATH_GPT_distance_is_30_l761_76188


namespace NUMINAMATH_GPT_Emily_sixth_quiz_score_l761_76140

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end NUMINAMATH_GPT_Emily_sixth_quiz_score_l761_76140


namespace NUMINAMATH_GPT_cylinder_volume_l761_76178

theorem cylinder_volume (h : ℝ) (H1 : π * h ^ 2 = 4 * π) : (π * (h / 2) ^ 2 * h) = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l761_76178


namespace NUMINAMATH_GPT_ratio_IM_IN_l761_76165

noncomputable def compute_ratio (IA IB IC ID : ℕ) (M N : ℕ) : ℚ :=
  (IA * IC : ℚ) / (IB * ID : ℚ)

theorem ratio_IM_IN (IA IB IC ID : ℕ) (hIA : IA = 12) (hIB : IB = 16) (hIC : IC = 14) (hID : ID = 11) :
  compute_ratio IA IB IC ID = 21 / 22 := by
  rw [hIA, hIB, hIC, hID]
  sorry

end NUMINAMATH_GPT_ratio_IM_IN_l761_76165


namespace NUMINAMATH_GPT_alice_burgers_each_day_l761_76116

theorem alice_burgers_each_day (cost_per_burger : ℕ) (total_spent : ℕ) (days_in_june : ℕ) 
  (h1 : cost_per_burger = 13) (h2 : total_spent = 1560) (h3 : days_in_june = 30) :
  (total_spent / cost_per_burger) / days_in_june = 4 := by
  sorry

end NUMINAMATH_GPT_alice_burgers_each_day_l761_76116


namespace NUMINAMATH_GPT_no_perfect_square_in_range_l761_76122

theorem no_perfect_square_in_range :
  ¬∃ (x : ℕ), 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ (n : ℕ), x = n * n :=
by
  sorry

end NUMINAMATH_GPT_no_perfect_square_in_range_l761_76122


namespace NUMINAMATH_GPT_number_of_squares_centered_at_60_45_l761_76148

noncomputable def number_of_squares_centered_at (cx : ℕ) (cy : ℕ) : ℕ :=
  let aligned_with_axes := 45
  let not_aligned_with_axes := 2025
  aligned_with_axes + not_aligned_with_axes

theorem number_of_squares_centered_at_60_45 : number_of_squares_centered_at 60 45 = 2070 := 
  sorry

end NUMINAMATH_GPT_number_of_squares_centered_at_60_45_l761_76148


namespace NUMINAMATH_GPT_consecutive_even_product_6digit_l761_76160

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_product_6digit_l761_76160


namespace NUMINAMATH_GPT_cody_discount_l761_76102

theorem cody_discount (initial_cost tax_rate cody_paid total_paid price_before_discount discount: ℝ) 
  (h1 : initial_cost = 40)
  (h2 : tax_rate = 0.05)
  (h3 : cody_paid = 17)
  (h4 : total_paid = 2 * cody_paid)
  (h5 : price_before_discount = initial_cost * (1 + tax_rate))
  (h6 : discount = price_before_discount - total_paid) :
  discount = 8 := by
  sorry

end NUMINAMATH_GPT_cody_discount_l761_76102


namespace NUMINAMATH_GPT_subset_implies_range_l761_76132

open Set

-- Definitions based on the problem statement
def A : Set ℝ := { x : ℝ | x < 5 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x < a }

-- Theorem statement
theorem subset_implies_range (a : ℝ) (h : A ⊆ B a) : a ≥ 5 :=
sorry

end NUMINAMATH_GPT_subset_implies_range_l761_76132


namespace NUMINAMATH_GPT_min_toys_to_add_l761_76175

theorem min_toys_to_add (T x : ℕ) (h1 : T % 12 = 3) (h2 : T % 18 = 3) :
  ((T + x) % 7 = 0) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_toys_to_add_l761_76175


namespace NUMINAMATH_GPT_sarah_likes_digits_l761_76112

theorem sarah_likes_digits : ∀ n : ℕ, n % 8 = 0 → (n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sarah_likes_digits_l761_76112


namespace NUMINAMATH_GPT_find_q_l761_76177

theorem find_q (q : ℤ) (h1 : lcm (lcm 12 16) (lcm 18 q) = 144) : q = 1 := sorry

end NUMINAMATH_GPT_find_q_l761_76177


namespace NUMINAMATH_GPT_cooking_oil_remaining_l761_76100

theorem cooking_oil_remaining (initial_weight : ℝ) (fraction_used : ℝ) (remaining_weight : ℝ) :
  initial_weight = 5 → fraction_used = 4 / 5 → remaining_weight = 21 / 5 → initial_weight * (1 - fraction_used) ≠ remaining_weight → initial_weight * (1 - fraction_used) = 1 :=
by 
  intros h_initial_weight h_fraction_used h_remaining_weight h_contradiction
  sorry

end NUMINAMATH_GPT_cooking_oil_remaining_l761_76100


namespace NUMINAMATH_GPT_central_angle_of_sector_l761_76141

open Real

theorem central_angle_of_sector (l S : ℝ) (α R : ℝ) (hl : l = 4) (hS : S = 4) (h1 : l = α * R) (h2 : S = 1/2 * α * R^2) : 
  α = 2 :=
by
  -- Proof will be supplied here
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l761_76141


namespace NUMINAMATH_GPT_cuboid_dimensions_sum_l761_76180

theorem cuboid_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 45) 
  (h2 : B * C = 80) 
  (h3 : C * A = 180) : 
  A + B + C = 145 / 9 :=
sorry

end NUMINAMATH_GPT_cuboid_dimensions_sum_l761_76180


namespace NUMINAMATH_GPT_problem_l761_76194

noncomputable def f (x : ℝ) (a b : ℝ) := (b - 2^x) / (2^(x+1) + a)

theorem problem (a b k : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) →
  (f 0 a b = 0) → (f (-1) a b = -f 1 a b) → 
  a = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, x < y → f x a b > f y a b) ∧ 
  (∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0 → k < 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_l761_76194


namespace NUMINAMATH_GPT_number_of_students_in_all_events_l761_76173

variable (T A B : ℕ)

-- Defining given conditions
-- Total number of students in the class
def total_students : ℕ := 45
-- Number of students participating in the Soccer event
def soccer_students : ℕ := 39
-- Number of students participating in the Basketball event
def basketball_students : ℕ := 28

-- Main theorem to prove
theorem number_of_students_in_all_events
  (h_total : T = total_students)
  (h_soccer : A = soccer_students)
  (h_basketball : B = basketball_students) :
  ∃ x : ℕ, x = A + B - T := sorry

end NUMINAMATH_GPT_number_of_students_in_all_events_l761_76173


namespace NUMINAMATH_GPT_oil_price_reduction_l761_76109

theorem oil_price_reduction (P P_r : ℝ) (h1 : P_r = 24.3) (h2 : 1080 / P - 1080 / P_r = 8) : 
  ((P - P_r) / P) * 100 = 18.02 := by
  sorry

end NUMINAMATH_GPT_oil_price_reduction_l761_76109


namespace NUMINAMATH_GPT_rubber_boat_lost_time_l761_76159

theorem rubber_boat_lost_time (a b : ℝ) (x : ℝ) (h : (5 - x) * (a - b) + (6 - x) * b = a + b) : x = 4 :=
  sorry

end NUMINAMATH_GPT_rubber_boat_lost_time_l761_76159


namespace NUMINAMATH_GPT_find_abc_and_sqrt_l761_76162

theorem find_abc_and_sqrt (a b c : ℤ) (h1 : 3 * a - 2 * b - 1 = 9) (h2 : a + 2 * b = -8) (h3 : c = Int.floor (2 + Real.sqrt 7)) :
  a = 2 ∧ b = -2 ∧ c = 4 ∧ (Real.sqrt (a - b + c) = 2 * Real.sqrt 2 ∨ Real.sqrt (a - b + c) = -2 * Real.sqrt 2) :=
by
  -- proof details go here
  sorry

end NUMINAMATH_GPT_find_abc_and_sqrt_l761_76162


namespace NUMINAMATH_GPT_negation_proposition_l761_76101

theorem negation_proposition : 
  ¬ (∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_proposition_l761_76101


namespace NUMINAMATH_GPT_correct_ordering_of_powers_l761_76182

theorem correct_ordering_of_powers :
  (6 ^ 8) < (3 ^ 15) ∧ (3 ^ 15) < (8 ^ 10) :=
by
  -- Define the expressions for each power
  let a := (8 : ℕ) ^ 10
  let b := (3 : ℕ) ^ 15
  let c := (6 : ℕ) ^ 8
  
  -- To utilize the values directly in inequalities
  have h1 : (c < b) := sorry -- Proof that 6^8 < 3^15
  have h2 : (b < a) := sorry -- Proof that 3^15 < 8^10

  exact ⟨h1, h2⟩ -- Conjunction of h1 and h2 to show 6^8 < 3^15 < 8^10

end NUMINAMATH_GPT_correct_ordering_of_powers_l761_76182


namespace NUMINAMATH_GPT_solve_for_x_l761_76151

theorem solve_for_x (x : ℝ) (h : 3 * x + 1 = -(5 - 2 * x)) : x = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l761_76151


namespace NUMINAMATH_GPT_rahul_share_of_payment_l761_76142

def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

theorem rahul_share_of_payment : (work_rate_rahul / (work_rate_rahul + work_rate_rajesh)) * total_payment = 60 := by
  sorry

end NUMINAMATH_GPT_rahul_share_of_payment_l761_76142


namespace NUMINAMATH_GPT_minute_hand_moves_180_degrees_l761_76133

noncomputable def minute_hand_angle_6_15_to_6_45 : ℝ :=
  let degrees_per_hour := 360
  let hours_period := 0.5
  degrees_per_hour * hours_period

theorem minute_hand_moves_180_degrees :
  minute_hand_angle_6_15_to_6_45 = 180 :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_moves_180_degrees_l761_76133


namespace NUMINAMATH_GPT_total_husk_is_30_bags_l761_76119

-- Define the total number of cows and the number of days.
def numCows : ℕ := 30
def numDays : ℕ := 30

-- Define the rate of consumption: one cow eats one bag in 30 days.
def consumptionRate (cows : ℕ) (days : ℕ) : ℕ := cows / days

-- Define the total amount of husk consumed in 30 days by 30 cows.
def totalHusk (cows : ℕ) (days : ℕ) (rate : ℕ) : ℕ := cows * rate

-- State the problem in a theorem.
theorem total_husk_is_30_bags : totalHusk numCows numDays 1 = 30 := by
  sorry

end NUMINAMATH_GPT_total_husk_is_30_bags_l761_76119


namespace NUMINAMATH_GPT_rest_area_location_l761_76104

theorem rest_area_location :
  ∃ (rest_area : ℝ), rest_area = 35 + (95 - 35) / 2 :=
by
  -- Here we set the variables for the conditions
  let fifth_exit := 35
  let seventh_exit := 95
  let rest_area := 35 + (95 - 35) / 2
  use rest_area
  sorry

end NUMINAMATH_GPT_rest_area_location_l761_76104


namespace NUMINAMATH_GPT_number_of_perfect_numbers_l761_76192

-- Define the concept of a perfect number
def perfect_number (a b : ℕ) : ℕ := (a + b)^2

-- Define the proposition we want to prove
theorem number_of_perfect_numbers : ∃ n : ℕ, n = 15 ∧ 
  ∀ p, ∃ a b : ℕ, p = perfect_number a b ∧ p < 200 :=
sorry

end NUMINAMATH_GPT_number_of_perfect_numbers_l761_76192


namespace NUMINAMATH_GPT_ball_reaches_height_less_than_2_after_6_bounces_l761_76154

theorem ball_reaches_height_less_than_2_after_6_bounces :
  ∃ (k : ℕ), 16 * (2/3) ^ k < 2 ∧ ∀ (m : ℕ), m < k → 16 * (2/3) ^ m ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_ball_reaches_height_less_than_2_after_6_bounces_l761_76154


namespace NUMINAMATH_GPT_line_through_three_points_l761_76105

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def p1 : Point := { x := 1, y := -1 }
def p2 : Point := { x := 3, y := 3 }
def p3 : Point := { x := 2, y := 1 }

-- The line that passes through the points
def line_eq (m b : ℝ) (p : Point) : Prop :=
  p.y = m * p.x + b

-- The condition of passing through the three points
def passes_three_points (m b : ℝ) : Prop :=
  line_eq m b p1 ∧ line_eq m b p2 ∧ line_eq m b p3

-- The statement to prove
theorem line_through_three_points (m b : ℝ) (h : passes_three_points m b) : m + b = -1 :=
  sorry

end NUMINAMATH_GPT_line_through_three_points_l761_76105


namespace NUMINAMATH_GPT_factor_expression_l761_76172

theorem factor_expression (x y a b : ℝ) : 
  ∃ f : ℝ, 3 * x * (a - b) - 9 * y * (b - a) = f * (x + 3 * y) ∧ f = 3 * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l761_76172


namespace NUMINAMATH_GPT_larger_number_is_1671_l761_76156

variable (L S : ℕ)

noncomputable def problem_conditions :=
  L - S = 1395 ∧ L = 6 * S + 15

theorem larger_number_is_1671 (h : problem_conditions L S) : L = 1671 := by
  sorry

end NUMINAMATH_GPT_larger_number_is_1671_l761_76156


namespace NUMINAMATH_GPT_compare_neg_fractions_l761_76150

theorem compare_neg_fractions : (-3 / 5) < (-1 / 3) := 
by {
  sorry
}

end NUMINAMATH_GPT_compare_neg_fractions_l761_76150


namespace NUMINAMATH_GPT_OH_squared_correct_l761_76185

noncomputable def OH_squared (O H : Point) (a b c R : ℝ) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem OH_squared_correct :
  ∀ (O H : Point) (a b c : ℝ) (R : ℝ),
    R = 7 →
    a^2 + b^2 + c^2 = 29 →
    OH_squared O H a b c R = 412 := by
  intros O H a b c R hR habc
  simp [OH_squared, hR, habc]
  sorry

end NUMINAMATH_GPT_OH_squared_correct_l761_76185


namespace NUMINAMATH_GPT_total_roses_in_a_week_l761_76176

theorem total_roses_in_a_week : 
  let day1 := 24 
  let day2 := day1 + 6
  let day3 := day2 + 6
  let day4 := day3 + 6
  let day5 := day4 + 6
  let day6 := day5 + 6
  let day7 := day6 + 6
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) = 294 :=
by
  sorry

end NUMINAMATH_GPT_total_roses_in_a_week_l761_76176


namespace NUMINAMATH_GPT_jackson_earnings_l761_76169

def hourly_rate_usd : ℝ := 5
def hourly_rate_gbp : ℝ := 3
def hourly_rate_jpy : ℝ := 400

def hours_vacuuming : ℝ := 2
def sessions_vacuuming : ℝ := 2

def hours_washing_dishes : ℝ := 0.5
def hours_cleaning_bathroom := hours_washing_dishes * 3

def exchange_rate_gbp_to_usd : ℝ := 1.35
def exchange_rate_jpy_to_usd : ℝ := 0.009

def earnings_in_usd : ℝ := (hours_vacuuming * sessions_vacuuming * hourly_rate_usd)
def earnings_in_gbp : ℝ := (hours_washing_dishes * hourly_rate_gbp)
def earnings_in_jpy : ℝ := (hours_cleaning_bathroom * hourly_rate_jpy)

def converted_gbp_to_usd : ℝ := earnings_in_gbp * exchange_rate_gbp_to_usd
def converted_jpy_to_usd : ℝ := earnings_in_jpy * exchange_rate_jpy_to_usd

def total_earnings_usd : ℝ := earnings_in_usd + converted_gbp_to_usd + converted_jpy_to_usd

theorem jackson_earnings : total_earnings_usd = 27.425 := by
  sorry

end NUMINAMATH_GPT_jackson_earnings_l761_76169


namespace NUMINAMATH_GPT_inequality_proof_l761_76181

theorem inequality_proof (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (x * y / Real.sqrt (x * y + y * z) + y * z / Real.sqrt (y * z + z * x) + z * x / Real.sqrt (z * x + x * y)) 
  ≤ (Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l761_76181


namespace NUMINAMATH_GPT_evaluate_expression_at_x_zero_l761_76138

theorem evaluate_expression_at_x_zero (x : ℕ) (h1 : x < 3) (h2 : x ≠ 1) (h3 : x ≠ 2) : ((3 / (x - 1) - x - 1) / (x - 2) / (x^2 - 2 * x + 1)) = 2 :=
by
  -- Here we need to provide our proof, though for now it’s indicated by sorry
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_zero_l761_76138


namespace NUMINAMATH_GPT_log_ordering_l761_76183

theorem log_ordering {x a b c : ℝ} (h1 : 1 < x) (h2 : x < 10) (ha : a = Real.log x^2) (hb : b = Real.log (Real.log x)) (hc : c = (Real.log x)^2) :
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_log_ordering_l761_76183


namespace NUMINAMATH_GPT_simplify_fraction_l761_76179

theorem simplify_fraction : (8 / (5 * 42) = 4 / 105) :=
by
    sorry

end NUMINAMATH_GPT_simplify_fraction_l761_76179


namespace NUMINAMATH_GPT_tom_savings_by_having_insurance_l761_76163

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end NUMINAMATH_GPT_tom_savings_by_having_insurance_l761_76163


namespace NUMINAMATH_GPT_find_n_l761_76193

-- Define the original and new parabola conditions
def original_parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
noncomputable def new_parabola (x n : ℝ) : ℝ := (x - n + 2)^2 - 1

-- Define the conditions for points A and B lying on the new parabola
def point_A (n : ℝ) : Prop := ∃ y₁ : ℝ, new_parabola 2 n = y₁
def point_B (n : ℝ) : Prop := ∃ y₂ : ℝ, new_parabola 4 n = y₂

-- Define the condition that y1 > y2
def points_condition (n : ℝ) : Prop := ∃ y₁ y₂ : ℝ, new_parabola 2 n = y₁ ∧ new_parabola 4 n = y₂ ∧ y₁ > y₂

-- Prove that n = 6 is the necessary value given the conditions
theorem find_n : ∀ n, (0 < n) → point_A n ∧ point_B n ∧ points_condition n → n = 6 :=
  by
    sorry

end NUMINAMATH_GPT_find_n_l761_76193


namespace NUMINAMATH_GPT_integer_pairs_satisfy_equation_l761_76145

theorem integer_pairs_satisfy_equation :
  ∃ (S : Finset (ℤ × ℤ)), S.card = 5 ∧ ∀ (m n : ℤ), (m, n) ∈ S ↔ m^2 + n = m * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_satisfy_equation_l761_76145


namespace NUMINAMATH_GPT_find_piglets_l761_76161

theorem find_piglets (chickens piglets goats sick_animals : ℕ) 
  (h1 : chickens = 26) 
  (h2 : goats = 34) 
  (h3 : sick_animals = 50) 
  (h4 : (chickens + piglets + goats) / 2 = sick_animals) : piglets = 40 := 
by
  sorry

end NUMINAMATH_GPT_find_piglets_l761_76161


namespace NUMINAMATH_GPT_gcf_lcm_360_210_l761_76149

theorem gcf_lcm_360_210 :
  let factorization_360 : ℕ × ℕ × ℕ × ℕ := (3, 2, 1, 0) -- Prime exponents for 2, 3, 5, 7
  let factorization_210 : ℕ × ℕ × ℕ × ℕ := (1, 1, 1, 1) -- Prime exponents for 2, 3, 5, 7
  gcd (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 30 ∧
  lcm (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 2520 :=
by {
  let factorization_360 := (3, 2, 1, 0)
  let factorization_210 := (1, 1, 1, 1)
  sorry
}

end NUMINAMATH_GPT_gcf_lcm_360_210_l761_76149


namespace NUMINAMATH_GPT_determine_function_l761_76117

noncomputable def functional_solution (f : ℝ → ℝ) : Prop := 
  ∃ (C₁ C₂ : ℝ), ∀ (x : ℝ), 0 < x → f x = C₁ * x + C₂ / x 

theorem determine_function (f : ℝ → ℝ) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + 1 / x) * f y = f (x * y) + f (y / x)) →
  functional_solution f :=
sorry

end NUMINAMATH_GPT_determine_function_l761_76117


namespace NUMINAMATH_GPT_apply_f_2019_times_l761_76153

noncomputable def f (x : ℝ) : ℝ := (1 - x^3) ^ (-1/3 : ℝ)

theorem apply_f_2019_times (x : ℝ) (n : ℕ) (h : n = 2019) (hx : x = 2018) : 
  (f^[n]) x = 2018 :=
by
  sorry

end NUMINAMATH_GPT_apply_f_2019_times_l761_76153


namespace NUMINAMATH_GPT_greatest_integer_n_l761_76186

theorem greatest_integer_n (n : ℤ) : n^2 - 9 * n + 20 ≤ 0 → n ≤ 5 := sorry

end NUMINAMATH_GPT_greatest_integer_n_l761_76186


namespace NUMINAMATH_GPT_true_propositions_l761_76134

theorem true_propositions :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3 : ℚ) * x^2 + (1/2 : ℚ) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_true_propositions_l761_76134


namespace NUMINAMATH_GPT_range_of_a_l761_76146

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * x - 1) / (x - 1) < 0 ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l761_76146


namespace NUMINAMATH_GPT_exists_naturals_l761_76125

def sum_of_digits (a : ℕ) : ℕ := sorry

theorem exists_naturals (R : ℕ) (hR : R > 0) :
  ∃ n : ℕ, n > 0 ∧ (sum_of_digits (n^2)) / (sum_of_digits n) = R :=
by
  sorry

end NUMINAMATH_GPT_exists_naturals_l761_76125


namespace NUMINAMATH_GPT_parabola_properties_l761_76108

theorem parabola_properties 
  (p : ℝ) (h_pos : 0 < p) (m : ℝ) 
  (A B : ℝ × ℝ)
  (h_AB_on_parabola : ∀ (P : ℝ × ℝ), P = A ∨ P = B → (P.snd)^2 = 2 * p * P.fst) 
  (h_line_intersection : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.fst = m * P.snd + 3)
  (h_dot_product : (A.fst * B.fst + A.snd * B.snd) = 6)
  : (exists C : ℝ × ℝ, C = (-3, 0)) ∧
    (∃ k1 k2 : ℝ, 
        k1 = A.snd / (A.fst + 3) ∧ 
        k2 = B.snd / (B.fst + 3) ∧ 
        (1 / k1^2 + 1 / k2^2 - 2 * m^2) = 24) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l761_76108


namespace NUMINAMATH_GPT_combined_rocket_height_l761_76164

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  sorry

end NUMINAMATH_GPT_combined_rocket_height_l761_76164


namespace NUMINAMATH_GPT_seating_sessions_l761_76147

theorem seating_sessions (num_parents num_pupils morning_parents afternoon_parents morning_pupils mid_day_pupils evening_pupils session_capacity total_sessions : ℕ) 
  (h1 : num_parents = 61)
  (h2 : num_pupils = 177)
  (h3 : session_capacity = 44)
  (h4 : morning_parents = 35)
  (h5 : afternoon_parents = 26)
  (h6 : morning_pupils = 65)
  (h7 : mid_day_pupils = 57)
  (h8 : evening_pupils = 55)
  (h9 : total_sessions = 8) :
  ∃ (parent_sessions pupil_sessions : ℕ), 
    parent_sessions + pupil_sessions = total_sessions ∧
    parent_sessions = (morning_parents + session_capacity - 1) / session_capacity + (afternoon_parents + session_capacity - 1) / session_capacity ∧
    pupil_sessions = (morning_pupils + session_capacity - 1) / session_capacity + (mid_day_pupils + session_capacity - 1) / session_capacity + (evening_pupils + session_capacity - 1) / session_capacity := 
by
  sorry

end NUMINAMATH_GPT_seating_sessions_l761_76147


namespace NUMINAMATH_GPT_sqrt_sum_bounds_l761_76157

theorem sqrt_sum_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
    4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2 - b)^2) + 
                   Real.sqrt (b^2 + (2 - c)^2) + 
                   Real.sqrt (c^2 + (2 - d)^2) + 
                   Real.sqrt (d^2 + (2 - a)^2) ∧
    Real.sqrt (a^2 + (2 - b)^2) + 
    Real.sqrt (b^2 + (2 - c)^2) + 
    Real.sqrt (c^2 + (2 - d)^2) + 
    Real.sqrt (d^2 + (2 - a)^2) ≤ 8 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_bounds_l761_76157


namespace NUMINAMATH_GPT_new_concentration_of_mixture_l761_76111

theorem new_concentration_of_mixture :
  let v1 := 2
  let c1 := 0.25
  let v2 := 6
  let c2 := 0.40
  let V := 10
  let alcohol_amount_v1 := v1 * c1
  let alcohol_amount_v2 := v2 * c2
  let total_alcohol := alcohol_amount_v1 + alcohol_amount_v2
  let new_concentration := (total_alcohol / V) * 100
  new_concentration = 29 := 
by
  sorry

end NUMINAMATH_GPT_new_concentration_of_mixture_l761_76111


namespace NUMINAMATH_GPT_distance_midpoints_eq_2_5_l761_76136

theorem distance_midpoints_eq_2_5 (A B C : ℝ) (hAB : A < B) (hBC : B < C) (hAC_len : C - A = 5) :
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    (M2 - M1 = 2.5) :=
by
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    sorry

end NUMINAMATH_GPT_distance_midpoints_eq_2_5_l761_76136


namespace NUMINAMATH_GPT_quadratic_function_series_sum_l761_76143

open Real

noncomputable def P (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 7

theorem quadratic_function_series_sum :
  (∀ (x : ℝ), 0 < x ∧ x < 1 →
    (∑' n, P n * x^n) = (16 * x^2 - 11 * x + 7) / (1 - x)^3) :=
sorry

end NUMINAMATH_GPT_quadratic_function_series_sum_l761_76143


namespace NUMINAMATH_GPT_shared_candy_equally_l761_76152

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end NUMINAMATH_GPT_shared_candy_equally_l761_76152


namespace NUMINAMATH_GPT_daily_rental_cost_l761_76198

theorem daily_rental_cost (rental_fee_per_day : ℝ) (mileage_rate : ℝ) (budget : ℝ) (max_miles : ℝ) 
  (h1 : mileage_rate = 0.20) 
  (h2 : budget = 88.0) 
  (h3 : max_miles = 190.0) :
  rental_fee_per_day = 50.0 := 
by
  sorry

end NUMINAMATH_GPT_daily_rental_cost_l761_76198


namespace NUMINAMATH_GPT_smallest_distance_AB_ge_2_l761_76120

noncomputable def A (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9
noncomputable def B (x y : ℝ) : Prop := y^2 = -8 * x

theorem smallest_distance_AB_ge_2 :
  ∀ (x1 y1 x2 y2 : ℝ), A x1 y1 → B x2 y2 → dist (x1, y1) (x2, y2) ≥ 2 := by
  sorry

end NUMINAMATH_GPT_smallest_distance_AB_ge_2_l761_76120


namespace NUMINAMATH_GPT_ellipse_k_range_l761_76158

theorem ellipse_k_range
  (k : ℝ)
  (h1 : k - 4 > 0)
  (h2 : 10 - k > 0)
  (h3 : k - 4 > 10 - k) :
  7 < k ∧ k < 10 :=
sorry

end NUMINAMATH_GPT_ellipse_k_range_l761_76158


namespace NUMINAMATH_GPT_eugene_cards_in_deck_l761_76126

theorem eugene_cards_in_deck 
  (cards_used_per_card : ℕ)
  (boxes_used : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_leftover : ℕ)
  (total_toothpicks_used : ℕ)
  (cards_used : ℕ)
  (total_cards_in_deck : ℕ)
  (h1 : cards_used_per_card = 75)
  (h2 : boxes_used = 6)
  (h3 : toothpicks_per_box = 450)
  (h4 : cards_leftover = 16)
  (h5 : total_toothpicks_used = boxes_used * toothpicks_per_box)
  (h6 : cards_used = total_toothpicks_used / cards_used_per_card)
  (h7 : total_cards_in_deck = cards_used + cards_leftover) :
  total_cards_in_deck = 52 :=
by 
  sorry

end NUMINAMATH_GPT_eugene_cards_in_deck_l761_76126


namespace NUMINAMATH_GPT_coconut_grove_nut_yield_l761_76144

theorem coconut_grove_nut_yield (x : ℕ) (Y : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * Y = 3 * x * 100)
  (h2 : x = 8) : Y = 180 := 
by
  sorry

end NUMINAMATH_GPT_coconut_grove_nut_yield_l761_76144


namespace NUMINAMATH_GPT_functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l761_76174

-- Conditions for y1
def cost_price : ℕ := 60
def selling_price_first_10_days : ℕ := 80
def y1 : ℕ → ℕ := fun x => x * x - 8 * x + 56
def items_sold_day4 : ℕ := 40
def items_sold_day6 : ℕ := 44

-- Conditions for y2
def selling_price_post_10_days : ℕ := 100
def y2 : ℕ → ℕ := fun x => 2 * x + 8
def gross_profit_condition : ℕ := 1120

-- 1) Prove functional relationship of y1.
theorem functional_relationship_y1 (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) : 
  y1 x = x * x - 8 * x + 56 := 
by
  sorry

-- 2) Prove value of x for daily gross profit $1120 on any day within first 10 days.
theorem daily_gross_profit_1120_first_10_days (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) (gp : (selling_price_first_10_days - cost_price) * y1 x = gross_profit_condition) : 
  x = 8 := 
by
  sorry

-- 3) Prove total gross profit W and range for 26 < x ≤ 31.
theorem total_gross_profit_W (x : ℕ) (h : 26 < x ∧ x ≤ 31) : 
  (100 - (cost_price - 2 * (y2 x - 60))) * (y2 x) = 8 * x * x - 96 * x - 512 := 
by
  sorry

end NUMINAMATH_GPT_functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l761_76174


namespace NUMINAMATH_GPT_find_f2_plus_fneg2_l761_76115

def f (x a: ℝ) := (x + a)^3

theorem find_f2_plus_fneg2 (a : ℝ)
  (h_cond : ∀ x : ℝ, f (1 + x) a = -f (1 - x) a) :
  f 2 (-1) + f (-2) (-1) = -26 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_plus_fneg2_l761_76115


namespace NUMINAMATH_GPT_number_of_three_digit_numbers_with_123_exactly_once_l761_76118

theorem number_of_three_digit_numbers_with_123_exactly_once : 
  (∃ (l : List ℕ), l = [1, 2, 3] ∧ l.permutations.length = 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_three_digit_numbers_with_123_exactly_once_l761_76118


namespace NUMINAMATH_GPT_probability_10_products_expected_value_of_products_l761_76190

open ProbabilityTheory

/-- Probability calculations for worker assessment. -/
noncomputable def worker_assessment_probability (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p^9 * (10 - 9*p)

/-- Expected value of total products produced and debugged by Worker A -/
noncomputable def expected_products (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  20 - 10*p - 10*p^9 + 10*p^10

/-- Theorem 1: Prove that the probability that Worker A ends the assessment by producing only 10 products is p^9(10 - 9p). -/
theorem probability_10_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  worker_assessment_probability p h = p^9 * (10 - 9*p) := by
  sorry

/-- Theorem 2: Prove the expected value E(X) of the total number of products produced and debugged by Worker A is 20 - 10p - 10p^9 + 10p^{10}. -/
theorem expected_value_of_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  expected_products p h = 20 - 10*p - 10*p^9 + 10*p^10 := by
  sorry

end NUMINAMATH_GPT_probability_10_products_expected_value_of_products_l761_76190


namespace NUMINAMATH_GPT_number_of_rhombuses_l761_76155

-- Definition: A grid with 25 small equilateral triangles arranged in a larger triangular pattern
def equilateral_grid (n : ℕ) : Prop :=
  n = 25

-- Theorem: Proving the number of rhombuses that can be formed from the grid
theorem number_of_rhombuses (n : ℕ) (h : equilateral_grid n) : ℕ :=
  30 

-- Main proof statement
example (n : ℕ) (h : equilateral_grid n) : number_of_rhombuses n h = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rhombuses_l761_76155


namespace NUMINAMATH_GPT_find_m_n_l761_76103

theorem find_m_n (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (h1 : m * n ∣ 3 ^ m + 1) (h2 : m * n ∣ 3 ^ n + 1) : 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l761_76103


namespace NUMINAMATH_GPT_trees_died_proof_l761_76170

def treesDied (original : Nat) (remaining : Nat) : Nat := original - remaining

theorem trees_died_proof : treesDied 20 4 = 16 := by
  -- Here we put the steps needed to prove the theorem, which is essentially 20 - 4 = 16.
  sorry

end NUMINAMATH_GPT_trees_died_proof_l761_76170


namespace NUMINAMATH_GPT_inequality_solution_set_l761_76189

theorem inequality_solution_set (x : ℝ) : (|x - 1| + 2 * x > 4) ↔ (x > 3) := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l761_76189


namespace NUMINAMATH_GPT_z_is_1_2_decades_younger_than_x_l761_76107

variable (x y z w : ℕ) -- Assume ages as natural numbers

def age_equivalence_1 : Prop := x + y = y + z + 12
def age_equivalence_2 : Prop := x + y + w = y + z + w + 12

theorem z_is_1_2_decades_younger_than_x (h1 : age_equivalence_1 x y z) (h2 : age_equivalence_2 x y z w) :
  z = x - 12 := by
  sorry

end NUMINAMATH_GPT_z_is_1_2_decades_younger_than_x_l761_76107


namespace NUMINAMATH_GPT_max_y_difference_l761_76114

theorem max_y_difference : (∃ x, (5 - 2 * x^2 + 2 * x^3 = 1 + x^2 + x^3)) ∧ 
                           (∀ y1 y2, y1 = 5 - 2 * (2^2) + 2 * (2^3) ∧ y2 = 5 - 2 * (1/2)^2 + 2 * (1/2)^3 → 
                           (y1 - y2 = 11.625)) := sorry

end NUMINAMATH_GPT_max_y_difference_l761_76114


namespace NUMINAMATH_GPT_width_of_playground_is_250_l761_76171

noncomputable def total_area_km2 : ℝ := 0.6
def num_playgrounds : ℕ := 8
def length_of_playground_m : ℝ := 300

theorem width_of_playground_is_250 :
  let total_area_m2 := total_area_km2 * 1000000
  let area_of_one_playground := total_area_m2 / num_playgrounds
  let width_of_playground := area_of_one_playground / length_of_playground_m
  width_of_playground = 250 := by
  sorry

end NUMINAMATH_GPT_width_of_playground_is_250_l761_76171


namespace NUMINAMATH_GPT_find_a_l761_76139

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the main hypothesis: (ai / (1 - i)) = (-1 + i)
def hypothesis (a : ℂ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Now, we state the theorem we need to prove
theorem find_a (a : ℝ) (ha : hypothesis a) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l761_76139


namespace NUMINAMATH_GPT_smallest_possible_c_l761_76167

theorem smallest_possible_c 
  (a b c : ℕ) (hp : a > 0 ∧ b > 0 ∧ c > 0) 
  (hg : b^2 = a * c) 
  (ha : 2 * c = a + b) : 
  c = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_c_l761_76167


namespace NUMINAMATH_GPT_solution_set_of_fractional_inequality_l761_76137

theorem solution_set_of_fractional_inequality :
  {x : ℝ | (x + 1) / (x - 3) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_fractional_inequality_l761_76137


namespace NUMINAMATH_GPT_distance_A_beats_B_l761_76127

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem distance_A_beats_B :
  let distance_A := 5 -- km
  let time_A := 10 / 60 -- hours (10 minutes)
  let time_B := 14 / 60 -- hours (14 minutes)
  let speed_A := speed distance_A time_A
  let speed_B := speed distance_A time_B
  let distance_A_in_time_B := speed_A * time_B
  distance_A_in_time_B - distance_A = 2 := -- km
by
  sorry

end NUMINAMATH_GPT_distance_A_beats_B_l761_76127


namespace NUMINAMATH_GPT_greatest_product_sum_2006_l761_76131

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end NUMINAMATH_GPT_greatest_product_sum_2006_l761_76131


namespace NUMINAMATH_GPT_gain_percent_is_100_l761_76195

variable {C S : ℝ}

-- Given conditions
axiom h1 : 50 * C = 25 * S
axiom h2 : S = 2 * C

-- Prove the gain percent is 100%
theorem gain_percent_is_100 (h1 : 50 * C = 25 * S) (h2 : S = 2 * C) : (S - C) / C * 100 = 100 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_100_l761_76195


namespace NUMINAMATH_GPT_tangent_line_at_1_l761_76130

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_1 :
  let x := (1 : ℝ)
  let y := (f 1)
  ∃ m b : ℝ, (∀ x, y - m * (x - 1) + b = 0)
  ∧ (m = -2)
  ∧ (b = -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_l761_76130


namespace NUMINAMATH_GPT_decrease_travel_time_l761_76110

variable (distance : ℕ) (initial_speed : ℕ) (speed_increase : ℕ)

def original_travel_time (distance initial_speed : ℕ) : ℕ :=
  distance / initial_speed

def new_travel_time (distance new_speed : ℕ) : ℕ :=
  distance / new_speed

theorem decrease_travel_time (h₁ : distance = 600) (h₂ : initial_speed = 50) (h₃ : speed_increase = 25) :
  original_travel_time distance initial_speed - new_travel_time distance (initial_speed + speed_increase) = 4 :=
by
  sorry

end NUMINAMATH_GPT_decrease_travel_time_l761_76110


namespace NUMINAMATH_GPT_ninety_percent_greater_than_eighty_percent_l761_76135

-- Define the constants involved in the problem
def ninety_percent (n : ℕ) : ℝ := 0.90 * n
def eighty_percent (n : ℕ) : ℝ := 0.80 * n

-- Define the problem statement
theorem ninety_percent_greater_than_eighty_percent :
  ninety_percent 40 - eighty_percent 30 = 12 :=
by
  sorry

end NUMINAMATH_GPT_ninety_percent_greater_than_eighty_percent_l761_76135


namespace NUMINAMATH_GPT_average_after_31st_inning_l761_76113

-- Define the conditions as Lean definitions
def initial_average (A : ℝ) := A

def total_runs_before_31st_inning (A : ℝ) := 30 * A

def score_in_31st_inning := 105

def new_average (A : ℝ) := A + 3

def total_runs_after_31st_inning (A : ℝ) := total_runs_before_31st_inning A + score_in_31st_inning

-- Define the statement to prove the batsman's average after the 31st inning is 15
theorem average_after_31st_inning (A : ℝ) : total_runs_after_31st_inning A = 31 * (new_average A) → new_average A = 15 := by
  sorry

end NUMINAMATH_GPT_average_after_31st_inning_l761_76113


namespace NUMINAMATH_GPT_solve_for_x_l761_76191

theorem solve_for_x (x : ℕ) : (1 : ℚ) / 2 = x / 8 → x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l761_76191


namespace NUMINAMATH_GPT_intersection_complement_l761_76123

open Set

noncomputable def U : Set ℝ := {-1, 0, 1, 4}
def A : Set ℝ := {-1, 1}
def B : Set ℝ := {1, 4}
def C_U_B : Set ℝ := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l761_76123


namespace NUMINAMATH_GPT_sqrt_inequality_l761_76166

theorem sqrt_inequality : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by sorry

end NUMINAMATH_GPT_sqrt_inequality_l761_76166


namespace NUMINAMATH_GPT_total_cost_of_books_l761_76184

def book_cost (num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook : ℕ) : ℕ :=
  (num_mathbooks * cost_mathbook) + (num_artbooks * cost_artbook) + (num_sciencebooks * cost_sciencebook)

theorem total_cost_of_books :
  let num_mathbooks := 2
  let num_artbooks := 3
  let num_sciencebooks := 6
  let cost_mathbook := 3
  let cost_artbook := 2
  let cost_sciencebook := 3
  book_cost num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_books_l761_76184


namespace NUMINAMATH_GPT_domain_of_function_l761_76124

theorem domain_of_function : 
  {x : ℝ | x ≠ 1 ∧ x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l761_76124


namespace NUMINAMATH_GPT_combination_8_5_l761_76197

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end NUMINAMATH_GPT_combination_8_5_l761_76197


namespace NUMINAMATH_GPT_false_statement_l761_76121

noncomputable def heartsuit (x y : ℝ) := abs (x - y)
noncomputable def diamondsuit (z w : ℝ) := (z + w) ^ 2

theorem false_statement : ∃ (x y : ℝ), (heartsuit x y) ^ 2 ≠ diamondsuit x y := by
  sorry

end NUMINAMATH_GPT_false_statement_l761_76121


namespace NUMINAMATH_GPT_jessica_current_age_l761_76196

-- Define the conditions
def jessicaOlderThanClaire (jessica claire : ℕ) : Prop :=
  jessica = claire + 6

def claireAgeInTwoYears (claire : ℕ) : Prop :=
  claire + 2 = 20

-- State the theorem to prove
theorem jessica_current_age : ∃ jessica claire : ℕ, 
  jessicaOlderThanClaire jessica claire ∧ claireAgeInTwoYears claire ∧ jessica = 24 := 
sorry

end NUMINAMATH_GPT_jessica_current_age_l761_76196


namespace NUMINAMATH_GPT_least_possible_mn_correct_l761_76106

def least_possible_mn (m n : ℕ) : ℕ :=
  m + n

theorem least_possible_mn_correct (m n : ℕ) :
  (Nat.gcd (m + n) 210 = 1) →
  (n^n ∣ m^m) →
  ¬(n ∣ m) →
  least_possible_mn m n = 407 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_mn_correct_l761_76106


namespace NUMINAMATH_GPT_sin_C_value_area_of_triangle_l761_76168

open Real
open Classical

variable {A B C a b c : ℝ}

-- Given conditions
axiom h1 : b = sqrt 2
axiom h2 : c = 1
axiom h3 : cos B = 3 / 4

-- Proof statements
theorem sin_C_value : sin C = sqrt 14 / 8 := sorry

theorem area_of_triangle : 1 / 2 * b * c * sin (B + C) = sqrt 7 / 4 := sorry

end NUMINAMATH_GPT_sin_C_value_area_of_triangle_l761_76168


namespace NUMINAMATH_GPT_contradiction_proof_l761_76128

theorem contradiction_proof (a b c : ℝ) (h : ¬ (a > 0 ∨ b > 0 ∨ c > 0)) : false :=
by
  sorry

end NUMINAMATH_GPT_contradiction_proof_l761_76128
