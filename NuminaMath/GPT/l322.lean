import Mathlib

namespace NUMINAMATH_GPT_angle_sum_around_point_l322_32254

theorem angle_sum_around_point (y : ℝ) (h1 : 150 + y + y = 360) : y = 105 :=
by sorry

end NUMINAMATH_GPT_angle_sum_around_point_l322_32254


namespace NUMINAMATH_GPT_center_of_circle_l322_32291

theorem center_of_circle :
  ∀ (x y : ℝ), (x^2 - 8 * x + y^2 - 4 * y = 16) → (x, y) = (4, 2) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l322_32291


namespace NUMINAMATH_GPT_girl_weaves_on_tenth_day_l322_32234

theorem girl_weaves_on_tenth_day 
  (a1 d : ℝ)
  (h1 : 7 * a1 + 21 * d = 28)
  (h2 : a1 + d + a1 + 4 * d + a1 + 7 * d = 15) :
  a1 + 9 * d = 10 :=
by sorry

end NUMINAMATH_GPT_girl_weaves_on_tenth_day_l322_32234


namespace NUMINAMATH_GPT_sphere_volume_equals_surface_area_l322_32210

theorem sphere_volume_equals_surface_area (r : ℝ) (hr : r = 3) :
  (4 / 3) * π * r^3 = 4 * π * r^2 := by
  sorry

end NUMINAMATH_GPT_sphere_volume_equals_surface_area_l322_32210


namespace NUMINAMATH_GPT_katrina_tax_deduction_l322_32200

variable (hourlyWage : ℚ) (taxRate : ℚ)

def wageInCents (wage : ℚ) : ℚ := wage * 100
def taxInCents (wageInCents : ℚ) (rate : ℚ) : ℚ := wageInCents * rate / 100

theorem katrina_tax_deduction : 
  hourlyWage = 25 ∧ taxRate = 2.5 → taxInCents (wageInCents hourlyWage) taxRate = 62.5 := 
by 
  sorry

end NUMINAMATH_GPT_katrina_tax_deduction_l322_32200


namespace NUMINAMATH_GPT_time_to_reach_ship_l322_32218

-- Define the conditions
def rate_of_descent := 30 -- feet per minute
def depth_to_ship := 2400 -- feet

-- Define the proof statement
theorem time_to_reach_ship : (depth_to_ship / rate_of_descent) = 80 :=
by
  -- The proof will be inserted here in practice
  sorry

end NUMINAMATH_GPT_time_to_reach_ship_l322_32218


namespace NUMINAMATH_GPT_no_valid_n_for_conditions_l322_32213

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end NUMINAMATH_GPT_no_valid_n_for_conditions_l322_32213


namespace NUMINAMATH_GPT_square_of_other_leg_l322_32244

variable {R : Type} [CommRing R]

theorem square_of_other_leg (a b c : R) (h1 : a^2 + b^2 = c^2) (h2 : c = a + 2) : b^2 = 4 * a + 4 :=
by
  sorry

end NUMINAMATH_GPT_square_of_other_leg_l322_32244


namespace NUMINAMATH_GPT_Flora_initial_daily_milk_l322_32235

def total_gallons : ℕ := 105
def total_weeks : ℕ := 3
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def extra_gallons_daily : ℕ := 2

theorem Flora_initial_daily_milk : 
  (total_gallons / total_days) = 5 := by
  sorry

end NUMINAMATH_GPT_Flora_initial_daily_milk_l322_32235


namespace NUMINAMATH_GPT_max_sum_11xy_3x_2012yz_l322_32271

theorem max_sum_11xy_3x_2012yz (x y z : ℕ) (h : x + y + z = 1000) : 
  11 * x * y + 3 * x + 2012 * y * z ≤ 503000000 :=
sorry

end NUMINAMATH_GPT_max_sum_11xy_3x_2012yz_l322_32271


namespace NUMINAMATH_GPT_value_of_x_l322_32251

theorem value_of_x (x : ℤ) (h : 3 * x = (26 - x) + 26) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l322_32251


namespace NUMINAMATH_GPT_min_value_of_expression_l322_32236

open Real

theorem min_value_of_expression (x y z : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 0 < z) (h₃ : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l322_32236


namespace NUMINAMATH_GPT_senior_citizen_tickets_l322_32295

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end NUMINAMATH_GPT_senior_citizen_tickets_l322_32295


namespace NUMINAMATH_GPT_rachel_bella_total_distance_l322_32246

theorem rachel_bella_total_distance:
  ∀ (distance_land distance_sea total_distance: ℕ), 
  distance_land = 451 → 
  distance_sea = 150 → 
  total_distance = distance_land + distance_sea → 
  total_distance = 601 := 
by 
  intros distance_land distance_sea total_distance h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_rachel_bella_total_distance_l322_32246


namespace NUMINAMATH_GPT_range_of_x_l322_32253

theorem range_of_x (S : ℕ → ℕ) (a : ℕ → ℕ) (x : ℕ) :
  (∀ n, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  S 0 = 0 →
  a 1 = x →
  (∀ n, a n ≤ a (n + 1)) →
  2 < x ∧ x < 3 := 
sorry

end NUMINAMATH_GPT_range_of_x_l322_32253


namespace NUMINAMATH_GPT_domain_lg_sqrt_l322_32292

def domain_of_function (x : ℝ) : Prop :=
  1 - x > 0 ∧ x + 2 > 0

theorem domain_lg_sqrt (x : ℝ) : 
  domain_of_function x ↔ -2 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_domain_lg_sqrt_l322_32292


namespace NUMINAMATH_GPT_total_adults_across_all_three_buses_l322_32261

def total_passengers : Nat := 450
def bus_A_passengers : Nat := 120
def bus_B_passengers : Nat := 210
def bus_C_passengers : Nat := 120
def children_ratio_A : ℚ := 1/3
def children_ratio_B : ℚ := 2/5
def children_ratio_C : ℚ := 3/8

theorem total_adults_across_all_three_buses :
  let children_A := bus_A_passengers * children_ratio_A
  let children_B := bus_B_passengers * children_ratio_B
  let children_C := bus_C_passengers * children_ratio_C
  let adults_A := bus_A_passengers - children_A
  let adults_B := bus_B_passengers - children_B
  let adults_C := bus_C_passengers - children_C
  (adults_A + adults_B + adults_C) = 281 := by {
    -- The proof steps will go here
    sorry
}

end NUMINAMATH_GPT_total_adults_across_all_three_buses_l322_32261


namespace NUMINAMATH_GPT_percentage_of_sikhs_is_10_l322_32290

-- Definitions based on the conditions
def total_boys : ℕ := 850
def percent_muslims : ℕ := 34
def percent_hindus : ℕ := 28
def other_community_boys : ℕ := 238

-- The problem statement to prove
theorem percentage_of_sikhs_is_10 :
  ((total_boys - ((percent_muslims * total_boys / 100) + (percent_hindus * total_boys / 100) + other_community_boys))
  * 100 / total_boys) = 10 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_sikhs_is_10_l322_32290


namespace NUMINAMATH_GPT_anna_pizza_fraction_l322_32281

theorem anna_pizza_fraction :
  let total_slices := 16
  let anna_eats := 2
  let shared_slices := 1
  let anna_share := shared_slices / 3
  let fraction_alone := anna_eats / total_slices
  let fraction_shared := anna_share / total_slices
  fraction_alone + fraction_shared = 7 / 48 :=
by
  sorry

end NUMINAMATH_GPT_anna_pizza_fraction_l322_32281


namespace NUMINAMATH_GPT_distance_amanda_to_kimberly_l322_32245

-- Define the given conditions
def amanda_speed : ℝ := 2 -- miles per hour
def amanda_time : ℝ := 3 -- hours

-- Prove that the distance is 6 miles
theorem distance_amanda_to_kimberly : amanda_speed * amanda_time = 6 := by
  sorry

end NUMINAMATH_GPT_distance_amanda_to_kimberly_l322_32245


namespace NUMINAMATH_GPT_container_volume_ratio_l322_32264

theorem container_volume_ratio (A B : ℕ) 
  (h1 : (3 / 4 : ℚ) * A = (5 / 8 : ℚ) * B) :
  (A : ℚ) / B = 5 / 6 :=
by
  admit
-- sorry

end NUMINAMATH_GPT_container_volume_ratio_l322_32264


namespace NUMINAMATH_GPT_calorie_allowance_correct_l322_32267

-- Definitions based on the problem's conditions
def daily_calorie_allowance : ℕ := 2000
def weekly_calorie_allowance : ℕ := 10500
def days_in_week : ℕ := 7

-- The statement to be proven
theorem calorie_allowance_correct :
  daily_calorie_allowance * days_in_week = weekly_calorie_allowance :=
by
  sorry

end NUMINAMATH_GPT_calorie_allowance_correct_l322_32267


namespace NUMINAMATH_GPT_incorrect_statement_count_l322_32284

theorem incorrect_statement_count :
  let statements := ["Every number has a square root",
                     "The square root of a number must be positive",
                     "The square root of a^2 is a",
                     "The square root of (π - 4)^2 is π - 4",
                     "A square root cannot be negative"]
  let incorrect := [statements.get! 0, statements.get! 1, statements.get! 2, statements.get! 3]
  incorrect.length = 4 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_count_l322_32284


namespace NUMINAMATH_GPT_average_check_l322_32220

variable (a b c d e f g x : ℕ)

def sum_natural (l : List ℕ) : ℕ := l.foldr (λ x y => x + y) 0

theorem average_check (h1 : a = 54) (h2 : b = 55) (h3 : c = 57) (h4 : d = 58) (h5 : e = 59) (h6 : f = 63) (h7 : g = 65) (h8 : x = 65) (avg : 60 * 8 = 480) :
    sum_natural [a, b, c, d, e, f, g, x] = 480 :=
by
  sorry

end NUMINAMATH_GPT_average_check_l322_32220


namespace NUMINAMATH_GPT_find_value_l322_32239

noncomputable def roots_of_equation (a b c : ℝ) : Prop :=
  10 * a^3 + 502 * a + 3010 = 0 ∧
  10 * b^3 + 502 * b + 3010 = 0 ∧
  10 * c^3 + 502 * c + 3010 = 0

theorem find_value (a b c : ℝ)
  (h : roots_of_equation a b c) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l322_32239


namespace NUMINAMATH_GPT_reciprocals_sum_l322_32202

theorem reciprocals_sum (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) : 
  (1 / a) + (1 / b) = 6 := 
sorry

end NUMINAMATH_GPT_reciprocals_sum_l322_32202


namespace NUMINAMATH_GPT_min_inv_sum_four_l322_32286

theorem min_inv_sum_four (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  4 ≤ (1 / a + 1 / b) := 
sorry

end NUMINAMATH_GPT_min_inv_sum_four_l322_32286


namespace NUMINAMATH_GPT_partition_impossible_l322_32216

def sum_of_list (l : List Int) : Int := l.foldl (· + ·) 0

theorem partition_impossible
  (l : List Int)
  (h : l = [-7, -4, -2, 3, 5, 9, 10, 18, 21, 33])
  (total_sum : Int := sum_of_list l)
  (target_diff : Int := 9) :
  ¬∃ (l1 l2 : List Int), 
    (l1 ++ l2 = l ∧ 
     sum_of_list l1 - sum_of_list l2 = target_diff ∧
     total_sum  = 86) := 
sorry

end NUMINAMATH_GPT_partition_impossible_l322_32216


namespace NUMINAMATH_GPT_find_principal_sum_l322_32211

theorem find_principal_sum
  (R : ℝ) (P : ℝ)
  (H1 : 0 < R)
  (H2 : 8 * 10 * P / 100 = 150) :
  P = 187.50 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_sum_l322_32211


namespace NUMINAMATH_GPT_subtraction_result_l322_32266

theorem subtraction_result: (3.75 - 1.4 = 2.35) :=
by
  sorry

end NUMINAMATH_GPT_subtraction_result_l322_32266


namespace NUMINAMATH_GPT_inequality_solution_l322_32258

theorem inequality_solution {x : ℝ} (h : 2 * x + 1 > x + 2) : x > 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l322_32258


namespace NUMINAMATH_GPT_total_students_l322_32206

-- Definition of the problem conditions
def ratio_boys_girls : ℕ := 8
def ratio_girls : ℕ := 5
def number_girls : ℕ := 160

-- The main theorem statement
theorem total_students (b g : ℕ) (h1 : b * ratio_girls = g * ratio_boys_girls) (h2 : g = number_girls) :
  b + g = 416 :=
sorry

end NUMINAMATH_GPT_total_students_l322_32206


namespace NUMINAMATH_GPT_ming_estimate_less_l322_32205

theorem ming_estimate_less (x y δ : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : δ > 0) : 
  (x + δ) - (y + 2 * δ) < x - y :=
by 
  sorry

end NUMINAMATH_GPT_ming_estimate_less_l322_32205


namespace NUMINAMATH_GPT_kataleya_total_amount_paid_l322_32265

/-- A store offers a $2 discount for every $10 purchase on any item in the store.
Kataleya went to the store and bought 400 peaches sold at forty cents each.
Prove that the total amount of money she paid at the store for the fruits is $128. -/
theorem kataleya_total_amount_paid : 
  let price_per_peach : ℝ := 0.40
  let number_of_peaches : ℝ := 400 
  let total_cost : ℝ := number_of_peaches * price_per_peach
  let discount_per_10_dollars : ℝ := 2
  let number_of_discounts := total_cost / 10
  let total_discount := number_of_discounts * discount_per_10_dollars
  let amount_paid := total_cost - total_discount
  amount_paid = 128 :=
by
  sorry

end NUMINAMATH_GPT_kataleya_total_amount_paid_l322_32265


namespace NUMINAMATH_GPT_infinite_points_on_line_with_positive_rational_coordinates_l322_32237

theorem infinite_points_on_line_with_positive_rational_coordinates :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 + p.2 = 4 ∧ 0 < p.1 ∧ 0 < p.2) ∧ S.Infinite :=
sorry

end NUMINAMATH_GPT_infinite_points_on_line_with_positive_rational_coordinates_l322_32237


namespace NUMINAMATH_GPT_altitude_identity_l322_32227

variable {a b c d : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def right_angle_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2

def altitude_property (a b c d : ℝ) : Prop :=
  a * b = c * d

theorem altitude_identity (a b c d : ℝ) (h1: right_angle_triangle a b c) (h2: altitude_property a b c d) :
  1 / a^2 + 1 / b^2 = 1 / d^2 :=
sorry

end NUMINAMATH_GPT_altitude_identity_l322_32227


namespace NUMINAMATH_GPT_intersection_expression_value_l322_32294

theorem intersection_expression_value
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁ * y₁ = 1)
  (h₂ : x₂ * y₂ = 1)
  (h₃ : x₁ = -x₂)
  (h₄ : y₁ = -y₂) :
  x₁ * y₂ + x₂ * y₁ = -2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_expression_value_l322_32294


namespace NUMINAMATH_GPT_kafelnikov_served_in_first_game_l322_32263

theorem kafelnikov_served_in_first_game (games : ℕ) (kafelnikov_wins : ℕ) (becker_wins : ℕ)
  (server_victories : ℕ) (x y : ℕ) 
  (h1 : kafelnikov_wins = 6)
  (h2 : becker_wins = 3)
  (h3 : server_victories = 5)
  (h4 : games = 9)
  (h5 : kafelnikov_wins + becker_wins = games)
  (h6 : (5 - x) + y = 5) 
  (h7 : x + y = 6):
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_kafelnikov_served_in_first_game_l322_32263


namespace NUMINAMATH_GPT_final_hair_length_l322_32226

-- Define the initial conditions and the expected final result.
def initial_hair_length : ℕ := 14
def hair_growth (x : ℕ) : ℕ := x
def hair_cut : ℕ := 20

-- Prove that the final hair length is x - 6.
theorem final_hair_length (x : ℕ) : initial_hair_length + hair_growth x - hair_cut = x - 6 :=
by
  sorry

end NUMINAMATH_GPT_final_hair_length_l322_32226


namespace NUMINAMATH_GPT_line_through_points_l322_32285

theorem line_through_points (x1 y1 x2 y2 : ℕ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (3, 8)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = 2 := 
by
  sorry

end NUMINAMATH_GPT_line_through_points_l322_32285


namespace NUMINAMATH_GPT_find_value_of_a_l322_32270

-- Define variables and constants
variable (a : ℚ)
variable (b : ℚ := 3 * a)
variable (c : ℚ := 4 * b)
variable (d : ℚ := 6 * c)
variable (total : ℚ := 186)

-- State the theorem
theorem find_value_of_a (h : a + b + c + d = total) : a = 93 / 44 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l322_32270


namespace NUMINAMATH_GPT_product_calculation_l322_32219

theorem product_calculation :
  12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end NUMINAMATH_GPT_product_calculation_l322_32219


namespace NUMINAMATH_GPT_jane_earnings_l322_32299

def age_of_child (jane_start_age : ℕ) (child_factor : ℕ) : ℕ :=
  jane_start_age / child_factor

def babysit_rate (age : ℕ) : ℕ :=
  if age < 2 then 5
  else if age <= 5 then 7
  else 8

def amount_earned (hours rate : ℕ) : ℕ := 
  hours * rate

def total_earnings (earnings : List ℕ) : ℕ :=
  earnings.foldl (·+·) 0

theorem jane_earnings
  (jane_start_age : ℕ := 18)
  (child_A_hours : ℕ := 50)
  (child_B_hours : ℕ := 90)
  (child_C_hours : ℕ := 130)
  (child_D_hours : ℕ := 70) :
  let child_A_age := age_of_child jane_start_age 2
  let child_B_age := child_A_age - 2
  let child_C_age := child_B_age + 3
  let child_D_age := child_C_age
  let earnings_A := amount_earned child_A_hours (babysit_rate child_A_age)
  let earnings_B := amount_earned child_B_hours (babysit_rate child_B_age)
  let earnings_C := amount_earned child_C_hours (babysit_rate child_C_age)
  let earnings_D := amount_earned child_D_hours (babysit_rate child_D_age)
  total_earnings [earnings_A, earnings_B, earnings_C, earnings_D] = 2720 :=
by
  sorry

end NUMINAMATH_GPT_jane_earnings_l322_32299


namespace NUMINAMATH_GPT_find_m_l322_32203

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the addition of vectors
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the main theorem without proof
theorem find_m (m : ℝ) : dot_product (vec_add vec_a (vec_b m)) vec_a = 0 ↔ m = -7/2 := by
  sorry

end NUMINAMATH_GPT_find_m_l322_32203


namespace NUMINAMATH_GPT_monthly_average_growth_rate_eq_l322_32277

theorem monthly_average_growth_rate_eq (x : ℝ) :
  16 * (1 + x)^2 = 25 :=
sorry

end NUMINAMATH_GPT_monthly_average_growth_rate_eq_l322_32277


namespace NUMINAMATH_GPT_intersection_A_B_l322_32241

def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}
def inter : Set ℝ := {x | 3 < x ∧ x < 4}

theorem intersection_A_B : A ∩ B = inter := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l322_32241


namespace NUMINAMATH_GPT_coin_toss_min_n_l322_32222

theorem coin_toss_min_n (n : ℕ) :
  (1 : ℝ) - (1 / (2 : ℝ)) ^ n ≥ 15 / 16 → n ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_coin_toss_min_n_l322_32222


namespace NUMINAMATH_GPT_triangle_subsegment_length_l322_32259

theorem triangle_subsegment_length (DF DE EF DG GF : ℚ)
  (h_ratio : ∃ x : ℚ, DF = 3 * x ∧ DE = 4 * x ∧ EF = 5 * x)
  (h_EF_len : EF = 20)
  (h_angle_bisector : DG + GF = DE ∧ DG / GF = DE / DF) :
  DF < DE ∧ DE < EF →
  min DG GF = 48 / 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_subsegment_length_l322_32259


namespace NUMINAMATH_GPT_john_salary_april_l322_32232

theorem john_salary_april 
  (initial_salary : ℤ)
  (raise_percentage : ℤ)
  (cut_percentage : ℤ)
  (bonus : ℤ)
  (february_salary : ℤ)
  (march_salary : ℤ)
  : initial_salary = 3000 →
    raise_percentage = 10 →
    cut_percentage = 15 →
    bonus = 500 →
    february_salary = initial_salary + (initial_salary * raise_percentage / 100) →
    march_salary = february_salary - (february_salary * cut_percentage / 100) →
    march_salary + bonus = 3305 :=
by
  intros
  sorry

end NUMINAMATH_GPT_john_salary_april_l322_32232


namespace NUMINAMATH_GPT_find_number_l322_32278

theorem find_number (number : ℝ) : 469138 * number = 4690910862 → number = 10000.1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l322_32278


namespace NUMINAMATH_GPT_overall_average_score_l322_32274

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / (scores.length)

theorem overall_average_score :
  let male_scores_avg := 82
  let female_scores_avg := 92
  let num_male_students := 8
  let num_female_students := 32
  let total_students := num_male_students + num_female_students
  let combined_scores_total := num_male_students * male_scores_avg + num_female_students * female_scores_avg
  average_score ([combined_scores_total]) / total_students = 90 :=
by 
  sorry

end NUMINAMATH_GPT_overall_average_score_l322_32274


namespace NUMINAMATH_GPT_a_n3_l322_32289

def right_angled_triangle_array (a : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, 1 ≤ j ∧ j ≤ i →
    (j = 1 → a i j = 1 / 4 + (i - 1) / 4) ∧
    (i ≥ 3 → (1 < j → a i j = a i 1 * (1 / 2)^(j - 1)))

theorem a_n3 (a : ℕ → ℕ → ℚ) (n : ℕ) (h : right_angled_triangle_array a) : a n 3 = n / 16 :=
sorry

end NUMINAMATH_GPT_a_n3_l322_32289


namespace NUMINAMATH_GPT_average_minutes_run_per_day_l322_32238

theorem average_minutes_run_per_day (e : ℕ)
  (sixth_grade_avg : ℕ := 16)
  (seventh_grade_avg : ℕ := 18)
  (eighth_grade_avg : ℕ := 12)
  (sixth_graders : ℕ := 3 * e)
  (seventh_graders : ℕ := 2 * e)
  (eighth_graders : ℕ := e) :
  ((sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders)
   / (sixth_graders + seventh_graders + eighth_graders) : ℕ) = 16 := 
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_per_day_l322_32238


namespace NUMINAMATH_GPT_youtube_dislikes_l322_32280

def initial_dislikes (likes : ℕ) : ℕ := (likes / 2) + 100

def new_dislikes (initial : ℕ) : ℕ := initial + 1000

theorem youtube_dislikes
  (likes : ℕ)
  (h_likes : likes = 3000) :
  new_dislikes (initial_dislikes likes) = 2600 :=
by
  sorry

end NUMINAMATH_GPT_youtube_dislikes_l322_32280


namespace NUMINAMATH_GPT_cakes_donated_l322_32298
-- Import necessary libraries for arithmetic operations and proofs

-- Define the conditions and required proof in Lean
theorem cakes_donated (c : ℕ) (h : 8 * c + 4 * c + 2 * c = 140) : c = 10 :=
by
  sorry

end NUMINAMATH_GPT_cakes_donated_l322_32298


namespace NUMINAMATH_GPT_eval_at_5_l322_32231

def g (x : ℝ) : ℝ := 3 * x^4 - 8 * x^3 + 15 * x^2 - 10 * x - 75

theorem eval_at_5 : g 5 = 1125 := by
  sorry

end NUMINAMATH_GPT_eval_at_5_l322_32231


namespace NUMINAMATH_GPT_max_self_intersection_points_13_max_self_intersection_points_1950_l322_32283

def max_self_intersection_points (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n - 3) / 2 else n * (n - 4) / 2 + 1

theorem max_self_intersection_points_13 : max_self_intersection_points 13 = 65 :=
by sorry

theorem max_self_intersection_points_1950 : max_self_intersection_points 1950 = 1897851 :=
by sorry

end NUMINAMATH_GPT_max_self_intersection_points_13_max_self_intersection_points_1950_l322_32283


namespace NUMINAMATH_GPT_find_missing_figure_l322_32224

theorem find_missing_figure (x : ℝ) (h : 0.003 * x = 0.15) : x = 50 :=
sorry

end NUMINAMATH_GPT_find_missing_figure_l322_32224


namespace NUMINAMATH_GPT_vector_calc_l322_32276

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end NUMINAMATH_GPT_vector_calc_l322_32276


namespace NUMINAMATH_GPT_sum_of_ages_l322_32229

theorem sum_of_ages (rose_age mother_age : ℕ) (rose_age_eq : rose_age = 25) (mother_age_eq : mother_age = 75) : 
  rose_age + mother_age = 100 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l322_32229


namespace NUMINAMATH_GPT_correct_propositions_count_l322_32214

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_propositions_count_l322_32214


namespace NUMINAMATH_GPT_max_sum_of_factors_l322_32268

theorem max_sum_of_factors (h k : ℕ) (h_even : Even h) (prod_eq : h * k = 24) : h + k ≤ 14 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l322_32268


namespace NUMINAMATH_GPT_loggers_count_l322_32217

theorem loggers_count 
  (cut_rate : ℕ) 
  (forest_width : ℕ) 
  (forest_height : ℕ) 
  (tree_density : ℕ) 
  (days_per_month : ℕ) 
  (months : ℕ) 
  (total_loggers : ℕ)
  (total_trees : ℕ := forest_width * forest_height * tree_density) 
  (total_days : ℕ := days_per_month * months)
  (trees_cut_down_per_logger : ℕ := cut_rate * total_days) 
  (expected_loggers : ℕ := total_trees / trees_cut_down_per_logger) 
  (h1: cut_rate = 6)
  (h2: forest_width = 4)
  (h3: forest_height = 6)
  (h4: tree_density = 600)
  (h5: days_per_month = 30)
  (h6: months = 10)
  (h7: total_loggers = expected_loggers)
: total_loggers = 8 := 
by {
    sorry
}

end NUMINAMATH_GPT_loggers_count_l322_32217


namespace NUMINAMATH_GPT_eggs_per_hen_l322_32212

theorem eggs_per_hen (total_chickens : ℕ) (num_roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ) :
  total_chickens = 440 →
  num_roosters = 39 →
  non_laying_hens = 15 →
  total_eggs = 1158 →
  (total_eggs / (total_chickens - num_roosters - non_laying_hens) = 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_eggs_per_hen_l322_32212


namespace NUMINAMATH_GPT_decreasing_function_solution_set_l322_32243

theorem decreasing_function_solution_set {f : ℝ → ℝ} (h : ∀ x y, x < y → f y < f x) :
  {x : ℝ | f 2 < f (2*x + 1)} = {x : ℝ | x < 1/2} :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_solution_set_l322_32243


namespace NUMINAMATH_GPT_probability_not_rel_prime_50_l322_32228

theorem probability_not_rel_prime_50 : 
  let n := 50;
  let non_rel_primes_count := n - Nat.totient 50;
  let total_count := n;
  let probability := (non_rel_primes_count : ℚ) / (total_count : ℚ);
  probability = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_rel_prime_50_l322_32228


namespace NUMINAMATH_GPT_inequality_holds_for_n_ge_0_l322_32233

theorem inequality_holds_for_n_ge_0
  (n : ℤ)
  (h : n ≥ 0)
  (a b c x y z : ℝ)
  (Habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (Hxyz : 0 < x ∧ 0 < y ∧ 0 < z)
  (Hmax : max a (max b (max c (max x (max y z)))) = a)
  (Hsum : a + b + c = x + y + z)
  (Hprod : a * b * c = x * y * z) : a^n + b^n + c^n ≥ x^n + y^n + z^n := 
sorry

end NUMINAMATH_GPT_inequality_holds_for_n_ge_0_l322_32233


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l322_32250

def atomic_weight (count : ℕ) (atomic_mass : ℝ) : ℝ :=
  count * atomic_mass

def molecular_weight (C_atom_count H_atom_count O_atom_count : ℕ)
  (C_atomic_weight H_atomic_weight O_atomic_weight : ℝ) : ℝ :=
  (atomic_weight C_atom_count C_atomic_weight) +
  (atomic_weight H_atom_count H_atomic_weight) +
  (atomic_weight O_atom_count O_atomic_weight)

theorem molecular_weight_of_compound :
  molecular_weight 3 6 1 12.01 1.008 16.00 = 58.078 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l322_32250


namespace NUMINAMATH_GPT_completing_square_correctness_l322_32269

theorem completing_square_correctness :
  (2 * x^2 - 4 * x - 7 = 0) ->
  ((x - 1)^2 = 9 / 2) :=
sorry

end NUMINAMATH_GPT_completing_square_correctness_l322_32269


namespace NUMINAMATH_GPT_max_value_of_s_l322_32201

theorem max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 10)
  (h2 : p * q + p * r + p * s + q * r + q * s + r * s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_s_l322_32201


namespace NUMINAMATH_GPT_number_of_positive_real_solutions_l322_32215

noncomputable def p (x : ℝ) : ℝ := x^12 + 5 * x^11 + 20 * x^10 + 1300 * x^9 - 1105 * x^8

theorem number_of_positive_real_solutions : ∃! x : ℝ, 0 < x ∧ p x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_positive_real_solutions_l322_32215


namespace NUMINAMATH_GPT_jake_total_distance_l322_32207

noncomputable def jake_rate : ℝ := 4 -- Jake's walking rate in miles per hour
noncomputable def total_time : ℝ := 2 -- Jake's total walking time in hours
noncomputable def break_time : ℝ := 0.5 -- Jake's break time in hours

theorem jake_total_distance :
  jake_rate * (total_time - break_time) = 6 :=
by
  sorry

end NUMINAMATH_GPT_jake_total_distance_l322_32207


namespace NUMINAMATH_GPT_least_integer_in_ratio_1_3_5_l322_32249

theorem least_integer_in_ratio_1_3_5 (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 90) (h_ratio : a * 3 = b ∧ a * 5 = c) : a = 10 :=
sorry

end NUMINAMATH_GPT_least_integer_in_ratio_1_3_5_l322_32249


namespace NUMINAMATH_GPT_parallelogram_area_correct_l322_32287

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_correct (base height : ℝ) (h_base : base = 30) (h_height : height = 12) : parallelogram_area base height = 360 :=
by
  rw [h_base, h_height]
  simp [parallelogram_area]
  sorry

end NUMINAMATH_GPT_parallelogram_area_correct_l322_32287


namespace NUMINAMATH_GPT_savings_account_after_8_weeks_l322_32272

noncomputable def initial_amount : ℕ := 43
noncomputable def weekly_allowance : ℕ := 10
noncomputable def comic_book_cost : ℕ := 3
noncomputable def saved_per_week : ℕ := weekly_allowance - comic_book_cost
noncomputable def weeks : ℕ := 8
noncomputable def savings_in_8_weeks : ℕ := saved_per_week * weeks
noncomputable def total_piggy_bank_after_8_weeks : ℕ := initial_amount + savings_in_8_weeks

theorem savings_account_after_8_weeks : total_piggy_bank_after_8_weeks = 99 :=
by
  have h1 : saved_per_week = 7 := rfl
  have h2 : savings_in_8_weeks = 56 := rfl
  have h3 : total_piggy_bank_after_8_weeks = 99 := rfl
  exact h3

end NUMINAMATH_GPT_savings_account_after_8_weeks_l322_32272


namespace NUMINAMATH_GPT_find_A_for_club_suit_l322_32230

def club_suit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

theorem find_A_for_club_suit :
  ∃ A : ℝ, club_suit A 3 = 73 ∧ A = 50 / 3 :=
sorry

end NUMINAMATH_GPT_find_A_for_club_suit_l322_32230


namespace NUMINAMATH_GPT_Drew_older_than_Maya_by_5_l322_32255

variable (Maya Drew Peter John Jacob : ℕ)
variable (h1 : John = 30)
variable (h2 : John = 2 * Maya)
variable (h3 : Jacob = 11)
variable (h4 : Jacob + 2 = (Peter + 2) / 2)
variable (h5 : Peter = Drew + 4)

theorem Drew_older_than_Maya_by_5 : Drew = Maya + 5 :=
by
  have Maya_age : Maya = 30 / 2 := by sorry
  have Jacob_age_in_2_years : Jacob + 2 = 13 := by sorry
  have Peter_age_in_2_years : Peter + 2 = 2 * 13 := by sorry
  have Peter_age : Peter = 26 - 2 := by sorry
  have Drew_age : Drew = Peter - 4 := by sorry
  have Drew_older_than_Maya : Drew = Maya + 5 := by sorry
  exact Drew_older_than_Maya

end NUMINAMATH_GPT_Drew_older_than_Maya_by_5_l322_32255


namespace NUMINAMATH_GPT_smallest_possible_n_l322_32257

theorem smallest_possible_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 60) : n = 16 :=
sorry

end NUMINAMATH_GPT_smallest_possible_n_l322_32257


namespace NUMINAMATH_GPT_time_to_pass_jogger_l322_32221

noncomputable def jogger_speed_kmh : ℕ := 9
noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_length : ℕ := 130
noncomputable def jogger_ahead_distance : ℕ := 240
noncomputable def train_speed_kmh : ℕ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover : ℕ := jogger_ahead_distance + train_length
noncomputable def time_taken_to_pass : ℝ := total_distance_to_cover / relative_speed

theorem time_to_pass_jogger : time_taken_to_pass = 37 := sorry

end NUMINAMATH_GPT_time_to_pass_jogger_l322_32221


namespace NUMINAMATH_GPT_dot_product_a_b_l322_32279

-- Definitions for unit vectors e1 and e2 with given conditions
variables (e1 e2 : ℝ × ℝ)
variables (h_norm_e1 : e1.1^2 + e1.2^2 = 1) -- e1 is a unit vector
variables (h_norm_e2 : e2.1^2 + e2.2^2 = 1) -- e2 is a unit vector
variables (h_angle : e1.1 * e2.1 + e1.2 * e2.2 = -1 / 2) -- angle between e1 and e2 is 120 degrees

-- Definitions for vectors a and b
def a : ℝ × ℝ := (e1.1 + e2.1, e1.2 + e2.2)
def b : ℝ × ℝ := (e1.1 - 3 * e2.1, e1.2 - 3 * e2.2)

-- Theorem to prove
theorem dot_product_a_b : (a e1 e2) • (b e1 e2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l322_32279


namespace NUMINAMATH_GPT_bob_km_per_gallon_l322_32247

-- Define the total distance Bob can drive.
def total_distance : ℕ := 100

-- Define the total amount of gas in gallons Bob's car uses.
def total_gas : ℕ := 10

-- Define the expected kilometers per gallon
def expected_km_per_gallon : ℕ := 10

-- Define the statement we want to prove
theorem bob_km_per_gallon : total_distance / total_gas = expected_km_per_gallon :=
by 
  sorry

end NUMINAMATH_GPT_bob_km_per_gallon_l322_32247


namespace NUMINAMATH_GPT_problem_statement_l322_32252

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (∃ φn : ℕ, φn = Nat.totient n ∧ p ∣ φn ∧ (∀ a : ℕ, Nat.gcd a n = 1 → n ∣ a ^ (φn / p) - 1)) ↔ 
  (∃ q1 q2 : ℕ, q1 ≠ q2 ∧ Nat.Prime q1 ∧ Nat.Prime q2 ∧ q1 ≡ 1 [MOD p] ∧ q2 ≡ 1 [MOD p] ∧ q1 ∣ n ∧ q2 ∣ n ∨ 
  (∃ q : ℕ, Nat.Prime q ∧ q ≡ 1 [MOD p] ∧ q ∣ n ∧ p ^ 2 ∣ n)) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l322_32252


namespace NUMINAMATH_GPT_island_not_Maya_l322_32297

variable (A B : Prop)
variable (IslandMaya : Prop)
variable (Liar : Prop → Prop)
variable (TruthTeller : Prop → Prop)

-- A's statement: "We are both liars, and this island is called Maya."
axiom A_statement : Liar A ∧ Liar B ∧ IslandMaya

-- B's statement: "At least one of us is a liar, and this island is not called Maya."
axiom B_statement : (Liar A ∨ Liar B) ∧ ¬IslandMaya

theorem island_not_Maya : ¬IslandMaya := by
  sorry

end NUMINAMATH_GPT_island_not_Maya_l322_32297


namespace NUMINAMATH_GPT_part1_part2_l322_32223

theorem part1 (x : ℝ) (m : ℝ) :
  (∃ x, x^2 - 2*(m-1)*x + m^2 = 0) → (m ≤ 1 / 2) := 
  sorry

theorem part2 (x1 x2 : ℝ) (m : ℝ) :
  (x1^2 - 2*(m-1)*x1 + m^2 = 0) ∧ (x2^2 - 2*(m-1)*x2 + m^2 = 0) ∧ 
  (x1^2 + x2^2 = 8 - 3*x1*x2) → (m = -2 / 5) := 
  sorry

end NUMINAMATH_GPT_part1_part2_l322_32223


namespace NUMINAMATH_GPT_polygon_vertices_product_at_least_2014_l322_32208

theorem polygon_vertices_product_at_least_2014 :
  ∀ (vertices : Fin 90 → ℕ), 
    (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 90) → 
    ∃ i, (vertices i) * (vertices ((i + 1) % 90)) ≥ 2014 :=
sorry

end NUMINAMATH_GPT_polygon_vertices_product_at_least_2014_l322_32208


namespace NUMINAMATH_GPT_range_of_x_l322_32256

-- Let p and q be propositions regarding the range of x:
def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Main theorem statement
theorem range_of_x 
  (h1 : ∀ x : ℝ, p x ∨ q x)
  (h2 : ∀ x : ℝ, ¬ q x) :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 4) := by
  sorry

end NUMINAMATH_GPT_range_of_x_l322_32256


namespace NUMINAMATH_GPT_tire_cost_l322_32288

theorem tire_cost (total_cost : ℕ) (number_of_tires : ℕ) (cost_per_tire : ℕ) 
    (h1 : total_cost = 240) 
    (h2 : number_of_tires = 4)
    (h3 : cost_per_tire = total_cost / number_of_tires) : 
    cost_per_tire = 60 :=
sorry

end NUMINAMATH_GPT_tire_cost_l322_32288


namespace NUMINAMATH_GPT_speed_with_current_l322_32204

theorem speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h1 : current_speed = 2.8) 
  (h2 : against_current_speed = 9.4) 
  (h3 : against_current_speed = v - current_speed) 
  : (v + current_speed) = 15 := by
  sorry

end NUMINAMATH_GPT_speed_with_current_l322_32204


namespace NUMINAMATH_GPT_bob_weight_l322_32275

theorem bob_weight (j b : ℝ) (h1 : j + b = 220) (h2 : b - 2 * j = b / 3) : b = 165 :=
  sorry

end NUMINAMATH_GPT_bob_weight_l322_32275


namespace NUMINAMATH_GPT_systematic_sampling_interval_l322_32273

-- Definitions based on conditions
def population_size : ℕ := 1000
def sample_size : ℕ := 40

-- Theorem statement 
theorem systematic_sampling_interval :
  population_size / sample_size = 25 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l322_32273


namespace NUMINAMATH_GPT_smallest_pretty_num_l322_32260

-- Define the notion of a pretty number
def is_pretty (n : ℕ) : Prop :=
  ∃ d1 d2 : ℕ, (1 ≤ d1 ∧ d1 ≤ n) ∧ (1 ≤ d2 ∧ d2 ≤ n) ∧ d2 - d1 ∣ n ∧ (1 < d1)

-- Define the statement to prove that 160400 is the smallest pretty number greater than 401 that is a multiple of 401
theorem smallest_pretty_num (n : ℕ) (hn1 : n > 401) (hn2 : n % 401 = 0) : n = 160400 :=
  sorry

end NUMINAMATH_GPT_smallest_pretty_num_l322_32260


namespace NUMINAMATH_GPT_Danny_shorts_washed_l322_32240

-- Define the given conditions
def Cally_white_shirts : ℕ := 10
def Cally_colored_shirts : ℕ := 5
def Cally_shorts : ℕ := 7
def Cally_pants : ℕ := 6

def Danny_white_shirts : ℕ := 6
def Danny_colored_shirts : ℕ := 8
def Danny_pants : ℕ := 6

def total_clothes_washed : ℕ := 58

-- Calculate total clothes washed by Cally
def total_cally_clothes : ℕ := 
  Cally_white_shirts + Cally_colored_shirts + Cally_shorts + Cally_pants

-- Calculate total clothes washed by Danny (excluding shorts)
def total_danny_clothes_excl_shorts : ℕ := 
  Danny_white_shirts + Danny_colored_shirts + Danny_pants

-- Define the statement to be proven
theorem Danny_shorts_washed : 
  total_clothes_washed - (total_cally_clothes + total_danny_clothes_excl_shorts) = 10 := by
  sorry

end NUMINAMATH_GPT_Danny_shorts_washed_l322_32240


namespace NUMINAMATH_GPT_period_of_f_l322_32293

noncomputable def f (x : ℝ) : ℝ := (Real.tan (x/3)) + (Real.sin x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

end NUMINAMATH_GPT_period_of_f_l322_32293


namespace NUMINAMATH_GPT_find_positive_m_has_exactly_single_solution_l322_32296

theorem find_positive_m_has_exactly_single_solution :
  ∃ m : ℝ, 0 < m ∧ (∀ x : ℝ, 16 * x^2 + m * x + 4 = 0 → x = 16) :=
sorry

end NUMINAMATH_GPT_find_positive_m_has_exactly_single_solution_l322_32296


namespace NUMINAMATH_GPT_marys_final_amount_l322_32248

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + simple_interest P r t

theorem marys_final_amount 
  (P : ℝ := 200)
  (A_after_2_years : ℝ := 260)
  (t1 : ℝ := 2)
  (t2 : ℝ := 6)
  (r : ℝ := (A_after_2_years - P) / (P * t1)) :
  final_amount P r t2 = 380 := 
by
  sorry

end NUMINAMATH_GPT_marys_final_amount_l322_32248


namespace NUMINAMATH_GPT_emerson_distance_l322_32225

theorem emerson_distance (d1 : ℕ) : 
  (d1 + 15 + 18 = 39) → d1 = 6 := 
by
  intro h
  have h1 : 33 = 39 - d1 := sorry -- Steps to manipulate equation to find d1
  sorry

end NUMINAMATH_GPT_emerson_distance_l322_32225


namespace NUMINAMATH_GPT_find_y_l322_32242

-- Define vectors as tuples
def vector_1 : ℝ × ℝ := (3, 4)
def vector_2 (y : ℝ) : ℝ × ℝ := (y, -5)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem we want to prove
theorem find_y (y : ℝ) :
  orthogonal vector_1 (vector_2 y) → y = (20 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_y_l322_32242


namespace NUMINAMATH_GPT_rectangle_ratio_l322_32262

-- Define the width of the rectangle
def width : ℕ := 7

-- Define the area of the rectangle
def area : ℕ := 196

-- Define that the length is a multiple of the width
def length_is_multiple_of_width (l w : ℕ) : Prop := ∃ k : ℕ, l = k * w

-- Define that the ratio of the length to the width is 4:1
def ratio_is_4_to_1 (l w : ℕ) : Prop := l / w = 4

theorem rectangle_ratio (l w : ℕ) (h1 : w = width) (h2 : area = l * w) (h3 : length_is_multiple_of_width l w) : ratio_is_4_to_1 l w :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l322_32262


namespace NUMINAMATH_GPT_measure_of_angle_B_l322_32209

theorem measure_of_angle_B 
  (A B C: ℝ)
  (a b c: ℝ)
  (h1: A + B + C = π)
  (h2: B / A = C / B)
  (h3: b^2 - a^2 = a * c) : B = 2 * π / 7 :=
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_l322_32209


namespace NUMINAMATH_GPT_Blair_17th_turn_l322_32282

/-
  Jo begins counting by saying "5". Blair then continues the sequence, each time saying a number that is 2 more than the last number Jo said. Jo increments by 1 each turn after Blair. They alternate turns.
  Prove that Blair says the number 55 on her 17th turn.
-/

def Jo_initial := 5
def increment_Jo := 1
def increment_Blair := 2

noncomputable def blair_sequence (n : ℕ) : ℕ :=
  Jo_initial + increment_Blair + (n - 1) * (increment_Jo + increment_Blair)

theorem Blair_17th_turn : blair_sequence 17 = 55 := by
    sorry

end NUMINAMATH_GPT_Blair_17th_turn_l322_32282
