import Mathlib

namespace NUMINAMATH_GPT_abc_not_8_l1433_143361

theorem abc_not_8 (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 :=
sorry

end NUMINAMATH_GPT_abc_not_8_l1433_143361


namespace NUMINAMATH_GPT_min_chips_to_color_all_cells_l1433_143372

def min_chips_needed (n : ℕ) : ℕ := n

theorem min_chips_to_color_all_cells (n : ℕ) :
  min_chips_needed n = n :=
sorry

end NUMINAMATH_GPT_min_chips_to_color_all_cells_l1433_143372


namespace NUMINAMATH_GPT_function_cannot_be_decreasing_if_f1_lt_f2_l1433_143338

variable (f : ℝ → ℝ)

theorem function_cannot_be_decreasing_if_f1_lt_f2
  (h : f 1 < f 2) : ¬ (∀ x y, x < y → f y < f x) :=
by
  sorry

end NUMINAMATH_GPT_function_cannot_be_decreasing_if_f1_lt_f2_l1433_143338


namespace NUMINAMATH_GPT_age_double_condition_l1433_143387

theorem age_double_condition (S M X : ℕ) (h1 : S = 44) (h2 : M = S + 46) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_double_condition_l1433_143387


namespace NUMINAMATH_GPT_third_number_is_507_l1433_143388

theorem third_number_is_507 (x : ℕ) 
  (h1 : (55 + 48 + x + 2 + 684 + 42) / 6 = 223) : 
  x = 507 := by
  sorry

end NUMINAMATH_GPT_third_number_is_507_l1433_143388


namespace NUMINAMATH_GPT_range_of_m_l1433_143300

noncomputable def inequality_has_solutions (x m : ℝ) :=
  |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, inequality_has_solutions x m) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1433_143300


namespace NUMINAMATH_GPT_students_more_than_rabbits_l1433_143335

-- Definitions of conditions
def classrooms : ℕ := 5
def students_per_classroom : ℕ := 22
def rabbits_per_classroom : ℕ := 2

-- Statement of the theorem
theorem students_more_than_rabbits :
  classrooms * students_per_classroom - classrooms * rabbits_per_classroom = 100 := 
  by
    sorry

end NUMINAMATH_GPT_students_more_than_rabbits_l1433_143335


namespace NUMINAMATH_GPT_expand_product_l1433_143397

def poly1 (x : ℝ) := 4 * x + 2
def poly2 (x : ℝ) := 3 * x - 1
def poly3 (x : ℝ) := x + 6

theorem expand_product (x : ℝ) :
  (poly1 x) * (poly2 x) * (poly3 x) = 12 * x^3 + 74 * x^2 + 10 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1433_143397


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1433_143399

theorem system1_solution :
  ∃ (x y : ℝ), 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

theorem system2_solution :
  ∃ (x y : ℝ), 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_system1_solution_system2_solution_l1433_143399


namespace NUMINAMATH_GPT_circumcircle_eqn_l1433_143353

variables (D E F : ℝ)

def point_A := (4, 0)
def point_B := (0, 3)
def point_C := (0, 0)

-- Define the system of equations for the circumcircle
def system : Prop :=
  (16 + 4*D + F = 0) ∧
  (9 + 3*E + F = 0) ∧
  (F = 0)

theorem circumcircle_eqn : system D E F → (D = -4 ∧ E = -3 ∧ F = 0) :=
sorry -- Proof omitted

end NUMINAMATH_GPT_circumcircle_eqn_l1433_143353


namespace NUMINAMATH_GPT_expression_value_l1433_143390

theorem expression_value :
  2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 :=
by sorry

end NUMINAMATH_GPT_expression_value_l1433_143390


namespace NUMINAMATH_GPT_roots_of_quadratic_l1433_143302

theorem roots_of_quadratic (b c : ℝ) (h1 : 1 + -2 = -b) (h2 : 1 * -2 = c) : b = 1 ∧ c = -2 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1433_143302


namespace NUMINAMATH_GPT_speed_of_ship_with_two_sails_l1433_143330

noncomputable def nautical_mile : ℝ := 1.15
noncomputable def land_miles_traveled : ℝ := 345
noncomputable def time_with_one_sail : ℝ := 4
noncomputable def time_with_two_sails : ℝ := 4
noncomputable def speed_with_one_sail : ℝ := 25

theorem speed_of_ship_with_two_sails :
  ∃ S : ℝ, 
    (S * time_with_two_sails + speed_with_one_sail * time_with_one_sail = land_miles_traveled / nautical_mile) → 
    S = 50  :=
by
  sorry

end NUMINAMATH_GPT_speed_of_ship_with_two_sails_l1433_143330


namespace NUMINAMATH_GPT_walking_total_distance_l1433_143371

theorem walking_total_distance :
  let t1 := 1    -- first hour on level ground
  let t2 := 0.5  -- next 0.5 hour on level ground
  let t3 := 0.75 -- 45 minutes uphill
  let t4 := 0.5  -- 30 minutes uphill
  let t5 := 0.5  -- 30 minutes downhill
  let t6 := 0.25 -- 15 minutes downhill
  let t7 := 1.5  -- 1.5 hours on level ground
  let t8 := 0.75 -- 45 minutes on level ground
  let s1 := 4    -- speed for t1 (4 km/hr)
  let s2 := 5    -- speed for t2 (5 km/hr)
  let s3 := 3    -- speed for t3 (3 km/hr)
  let s4 := 2    -- speed for t4 (2 km/hr)
  let s5 := 6    -- speed for t5 (6 km/hr)
  let s6 := 7    -- speed for t6 (7 km/hr)
  let s7 := 4    -- speed for t7 (4 km/hr)
  let s8 := 6    -- speed for t8 (6 km/hr)
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5 + s6 * t6 + s7 * t7 + s8 * t8 = 25 :=
by sorry

end NUMINAMATH_GPT_walking_total_distance_l1433_143371


namespace NUMINAMATH_GPT_min_value_l1433_143318

noncomputable def f (x : ℝ) : ℝ := 2017 * x + Real.sin (x / 2018) + (2019 ^ x - 1) / (2019 ^ x + 1)

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : f (2 * a) + f (b - 4) = 0) :
  2 * a + b = 4 → (1 / a + 2 / b) = 2 :=
by sorry

end NUMINAMATH_GPT_min_value_l1433_143318


namespace NUMINAMATH_GPT_next_leap_year_visible_after_2017_l1433_143332

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def stromquist_visible (start_year interval next_leap : ℕ) : Prop :=
  ∃ k : ℕ, next_leap = start_year + k * interval ∧ is_leap_year next_leap

theorem next_leap_year_visible_after_2017 :
  stromquist_visible 2017 61 2444 :=
  sorry

end NUMINAMATH_GPT_next_leap_year_visible_after_2017_l1433_143332


namespace NUMINAMATH_GPT_find_y_l1433_143379

theorem find_y (y : ℤ) (h : (15 + 26 + y) / 3 = 23) : y = 28 :=
by sorry

end NUMINAMATH_GPT_find_y_l1433_143379


namespace NUMINAMATH_GPT_find_ratio_of_arithmetic_sequences_l1433_143315

variable {a_n b_n : ℕ → ℕ}
variable {A_n B_n : ℕ → ℝ}

def arithmetic_sums (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℝ) : Prop :=
  ∀ n, A_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 8 - a_n 7))) / 2 ∧
         B_n n = (n * (2 * b_n 1 + (n - 1) * (b_n 8 - b_n 7))) / 2

theorem find_ratio_of_arithmetic_sequences 
    (h : ∀ n, A_n n / B_n n = (5 * n - 3) / (n + 9)) :
    ∃ r : ℝ, r = 3 := by
  sorry

end NUMINAMATH_GPT_find_ratio_of_arithmetic_sequences_l1433_143315


namespace NUMINAMATH_GPT_water_remaining_l1433_143357

variable (initial_amount : ℝ) (leaked_amount : ℝ)

theorem water_remaining (h1 : initial_amount = 0.75)
                       (h2 : leaked_amount = 0.25) :
  initial_amount - leaked_amount = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_water_remaining_l1433_143357


namespace NUMINAMATH_GPT_min_x_div_y_l1433_143367

theorem min_x_div_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) : ∃c: ℝ, c = 1 ∧ ∀(a: ℝ), x = a → y = 1 → a/y ≥ c :=
by
  sorry

end NUMINAMATH_GPT_min_x_div_y_l1433_143367


namespace NUMINAMATH_GPT_solve_phi_l1433_143341

theorem solve_phi (n : ℕ) : 
  (∃ (x y z : ℕ), 5 * x + 2 * y + z = 10 * n) → 
  (∃ (φ : ℕ), φ = 5 * n^2 + 4 * n + 1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_phi_l1433_143341


namespace NUMINAMATH_GPT_find_a7_coefficient_l1433_143347

theorem find_a7_coefficient (a_7 : ℤ) : 
    (∀ x : ℤ, (x+1)^5 * (2*x-1)^3 = a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) → a_7 = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_coefficient_l1433_143347


namespace NUMINAMATH_GPT_raghu_investment_l1433_143380

theorem raghu_investment (R : ℝ) 
  (h1 : ∀ T : ℝ, T = 0.9 * R) 
  (h2 : ∀ V : ℝ, V = 0.99 * R) 
  (h3 : R + 0.9 * R + 0.99 * R = 6069) : 
  R = 2100 := 
by
  sorry

end NUMINAMATH_GPT_raghu_investment_l1433_143380


namespace NUMINAMATH_GPT_economical_speed_l1433_143358

variable (a k : ℝ)
variable (ha : 0 < a) (hk : 0 < k)

theorem economical_speed (v : ℝ) : 
  v = (a / (2 * k))^(1/3) :=
sorry

end NUMINAMATH_GPT_economical_speed_l1433_143358


namespace NUMINAMATH_GPT_line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l1433_143355

-- Definitions for the first condition
def P : ℝ × ℝ := (3, 2)
def passes_through_P (l : ℝ → ℝ) := l P.1 = P.2
def equal_intercepts (l : ℝ → ℝ) := ∃ a : ℝ, l a = 0 ∧ l (-a) = 0

-- Equation 1: Line passing through P with equal intercepts
theorem line_through_P_with_equal_intercepts :
  (∃ l : ℝ → ℝ, passes_through_P l ∧ equal_intercepts l ∧ 
   (∀ x y : ℝ, l x = y ↔ (2 * x - 3 * y = 0) ∨ (x + y - 5 = 0))) :=
sorry

-- Definitions for the second condition
def A : ℝ × ℝ := (-1, -3)
def passes_through_A (l : ℝ → ℝ) := l A.1 = A.2
def inclination_90 (l : ℝ → ℝ) := ∀ x : ℝ, l x = l 0

-- Equation 2: Line passing through A with inclination 90°
theorem line_through_A_with_inclination_90 :
  (∃ l : ℝ → ℝ, passes_through_A l ∧ inclination_90 l ∧ 
   (∀ x y : ℝ, l x = y ↔ (x + 1 = 0))) :=
sorry

end NUMINAMATH_GPT_line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l1433_143355


namespace NUMINAMATH_GPT_problem_solution_l1433_143378

theorem problem_solution :
  ∀ (x y z : ℤ),
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = 40 :=
by
  intros x y z h1 h2 hx
  rw [hx] at h1 h2
  -- Here you could continue solving but we'll use sorry to indicate the end as no proof is requested.
  sorry

end NUMINAMATH_GPT_problem_solution_l1433_143378


namespace NUMINAMATH_GPT_sum_of_squares_l1433_143331

theorem sum_of_squares (x y : ℤ) (h : ∃ k : ℤ, (x^2 + y^2) = 5 * k) : 
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_l1433_143331


namespace NUMINAMATH_GPT_equilateral_triangle_ratio_correct_l1433_143324

noncomputable def equilateral_triangle_ratio : ℝ :=
  let side_length := 12
  let perimeter := 3 * side_length
  let altitude := side_length * (Real.sqrt 3 / 2)
  let area := (side_length * altitude) / 2
  area / perimeter

theorem equilateral_triangle_ratio_correct :
  equilateral_triangle_ratio = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_ratio_correct_l1433_143324


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1433_143327

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1433_143327


namespace NUMINAMATH_GPT_youtube_dislikes_l1433_143376

theorem youtube_dislikes (x y : ℕ) 
  (h1 : x = 3 * y) 
  (h2 : x = 100 + 2 * y) 
  (h_y_increased : ∃ y' : ℕ, y' = 3 * y) :
  y' = 300 := by
  sorry

end NUMINAMATH_GPT_youtube_dislikes_l1433_143376


namespace NUMINAMATH_GPT_bus_driver_limit_of_hours_l1433_143349

theorem bus_driver_limit_of_hours (r o T H L : ℝ)
  (h_reg_rate : r = 16)
  (h_ot_rate : o = 1.75 * r)
  (h_total_comp : T = 752)
  (h_hours_worked : H = 44)
  (h_equation : r * L + o * (H - L) = T) :
  L = 40 :=
  sorry

end NUMINAMATH_GPT_bus_driver_limit_of_hours_l1433_143349


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1433_143322

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (0 < x ∧ x < (1 / 2)) ∧ (f (1 / 2) - f x) > 0 :=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1433_143322


namespace NUMINAMATH_GPT_digits_in_8_20_3_30_base_12_l1433_143325

def digits_in_base (n b : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + Nat.log b n

theorem digits_in_8_20_3_30_base_12 : digits_in_base (8^20 * 3^30) 12 = 31 :=
by
  sorry

end NUMINAMATH_GPT_digits_in_8_20_3_30_base_12_l1433_143325


namespace NUMINAMATH_GPT_range_of_a_l1433_143398

/-- Given a fixed point A(a, 3) is outside the circle x^2 + y^2 - 2ax - 3y + a^2 + a = 0,
we want to show that the range of values for a is (0, 9/4). -/
theorem range_of_a (a : ℝ) :
  (∃ (A : ℝ × ℝ), A = (a, 3) ∧ ¬(∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0))
  ↔ (0 < a ∧ a < 9/4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1433_143398


namespace NUMINAMATH_GPT__l1433_143364

noncomputable def angle_ACB_is_45_degrees (A B C D E F : Type) [LinearOrderedField A]
  (angle : A → A → A → A) (AB AC : A) (h1 : AB = 3 * AC)
  (BAE ACD : A) (h2 : BAE = ACD)
  (BCA : A) (h3 : BAE = 2 * BCA)
  (CF FE : A) (h4 : CF = FE)
  (is_isosceles : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a = b → b = c → a = c)
  (triangle_sum : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a + b + c = 180) :
  ∃ (angle_ACB : A), angle_ACB = 45 := 
by
  -- Here we assume we have the appropriate conditions from geometry
  -- Then you'd prove the theorem based on given hypotheses
  sorry

end NUMINAMATH_GPT__l1433_143364


namespace NUMINAMATH_GPT_Arman_hours_worked_l1433_143394

/--
  Given:
  - LastWeekHours = 35
  - LastWeekRate = 10 (in dollars per hour)
  - IncreaseRate = 0.5 (in dollars per hour)
  - TotalEarnings = 770 (in dollars)
  Prove that:
  - ThisWeekHours = 40
-/
theorem Arman_hours_worked (LastWeekHours : ℕ) (LastWeekRate : ℕ) (IncreaseRate : ℕ) (TotalEarnings : ℕ)
  (h1 : LastWeekHours = 35)
  (h2 : LastWeekRate = 10)
  (h3 : IncreaseRate = 1/2)  -- because 0.5 as a fraction is 1/2
  (h4 : TotalEarnings = 770)
  : ∃ ThisWeekHours : ℕ, ThisWeekHours = 40 :=
by
  sorry

end NUMINAMATH_GPT_Arman_hours_worked_l1433_143394


namespace NUMINAMATH_GPT_max_absolute_difference_l1433_143307

theorem max_absolute_difference (a b c d e : ℤ) (p : ℤ) :
  0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ 100 ∧ p = (a + b + c + d + e) / 5 →
  (|p - c| ≤ 40) :=
by
  sorry

end NUMINAMATH_GPT_max_absolute_difference_l1433_143307


namespace NUMINAMATH_GPT_convert_to_dms_convert_to_decimal_degrees_l1433_143365

-- Problem 1: Conversion of 24.29 degrees to degrees, minutes, and seconds 
theorem convert_to_dms (d : ℝ) (h : d = 24.29) : 
  (∃ deg min sec, d = deg + min / 60 + sec / 3600 ∧ deg = 24 ∧ min = 17 ∧ sec = 24) :=
by
  sorry

-- Problem 2: Conversion of 36 degrees 40 minutes 30 seconds to decimal degrees
theorem convert_to_decimal_degrees (deg min sec : ℝ) (h : deg = 36 ∧ min = 40 ∧ sec = 30) : 
  (deg + min / 60 + sec / 3600) = 36.66 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_dms_convert_to_decimal_degrees_l1433_143365


namespace NUMINAMATH_GPT_find_radii_l1433_143383

-- Definitions based on the problem conditions
def tangent_lengths (TP T'Q r r' PQ: ℝ) : Prop :=
  TP = 6 ∧ T'Q = 10 ∧ PQ = 16 ∧ r < r'

-- The main theorem to prove the radii are 15 and 5
theorem find_radii (TP T'Q r r' PQ: ℝ) 
  (h : tangent_lengths TP T'Q r r' PQ) :
  r = 15 ∧ r' = 5 :=
sorry

end NUMINAMATH_GPT_find_radii_l1433_143383


namespace NUMINAMATH_GPT_milk_packet_volume_l1433_143316

theorem milk_packet_volume :
  ∃ (m : ℕ), (150 * m = 1250 * 30) ∧ m = 250 :=
by
  sorry

end NUMINAMATH_GPT_milk_packet_volume_l1433_143316


namespace NUMINAMATH_GPT_geometric_series_ratio_half_l1433_143350

theorem geometric_series_ratio_half (a r S : ℝ) (hS : S = a / (1 - r)) 
  (h_ratio : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_ratio_half_l1433_143350


namespace NUMINAMATH_GPT_largest_value_x_y_l1433_143366

theorem largest_value_x_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 11 / 4 :=
sorry

end NUMINAMATH_GPT_largest_value_x_y_l1433_143366


namespace NUMINAMATH_GPT_smallest_a_no_inverse_mod_72_90_l1433_143310

theorem smallest_a_no_inverse_mod_72_90 :
  ∃ (a : ℕ), a > 0 ∧ ∀ b : ℕ, (b > 0 → gcd b 72 > 1 ∧ gcd b 90 > 1 → b ≥ a) ∧ gcd a 72 > 1 ∧ gcd a 90 > 1 ∧ a = 6 :=
by sorry

end NUMINAMATH_GPT_smallest_a_no_inverse_mod_72_90_l1433_143310


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l1433_143305

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  a'' + b'' = 3 * S + 24 := 
by
  let a' := a + 4
  let b' := b + 4
  let a'' := 3 * a'
  let b'' := 3 * b'
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l1433_143305


namespace NUMINAMATH_GPT_no_integer_solutions_l1433_143356

theorem no_integer_solutions (x y z : ℤ) : ¬ (x^2 + y^2 = 3 * z^2) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l1433_143356


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1433_143314

-- Definitions of the conditions
def a : ℝ := 1
def b (k : ℝ) : ℝ := -3 * k
def c : ℝ := -2

-- Definition of the discriminant function
def discriminant (k : ℝ) : ℝ := (b k) ^ 2 - 4 * a * c

-- Logical statement to be proved
theorem quadratic_has_two_distinct_real_roots (k : ℝ) : discriminant k > 0 :=
by
  unfold discriminant
  unfold b a c
  simp
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1433_143314


namespace NUMINAMATH_GPT_solve_for_x_l1433_143320

theorem solve_for_x :
  ∀ x : ℝ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) → x = -80 / 19 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1433_143320


namespace NUMINAMATH_GPT_equation_D_has_two_distinct_real_roots_l1433_143337

def quadratic_has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem equation_D_has_two_distinct_real_roots : quadratic_has_two_distinct_real_roots 1 2 (-1) :=
by {
  sorry
}

end NUMINAMATH_GPT_equation_D_has_two_distinct_real_roots_l1433_143337


namespace NUMINAMATH_GPT_cost_of_socks_l1433_143333

theorem cost_of_socks (x : ℝ) : 
  let initial_amount := 20
  let hat_cost := 7 
  let final_amount := 5
  let socks_pairs := 4
  let remaining_amount := initial_amount - hat_cost
  remaining_amount - socks_pairs * x = final_amount 
  -> x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_socks_l1433_143333


namespace NUMINAMATH_GPT_a_n_values_l1433_143323

noncomputable def a : ℕ → ℕ := sorry
noncomputable def S : ℕ → ℕ := sorry

axiom Sn_property (n : ℕ) (hn : n > 0) : S n = 2 * (a n) - n

theorem a_n_values : a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7 ∧ ∀ n : ℕ, n > 0 → a n = 2^n - 1 := 
by sorry

end NUMINAMATH_GPT_a_n_values_l1433_143323


namespace NUMINAMATH_GPT_number_of_outliers_l1433_143309

def data_set : List ℕ := [10, 24, 36, 36, 42, 45, 45, 46, 58, 64]
def Q1 : ℕ := 36
def Q3 : ℕ := 46
def IQR : ℕ := Q3 - Q1
def low_threshold : ℕ := Q1 - 15
def high_threshold : ℕ := Q3 + 15
def outliers : List ℕ := data_set.filter (λ x => x < low_threshold ∨ x > high_threshold)

theorem number_of_outliers : outliers.length = 3 :=
  by
    -- Proof would go here
    sorry

end NUMINAMATH_GPT_number_of_outliers_l1433_143309


namespace NUMINAMATH_GPT_part1_part2_l1433_143359

-- Define the parabola C as y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l with slope k passing through point P(-2, 1)
def line (x y k : ℝ) : Prop := y - 1 = k * (x + 2)

-- Part 1: Prove the range of k for which line l intersects parabola C at two points is -1 < k < -1/2
theorem part1 (k : ℝ) : 
  (∃ x y, parabola x y ∧ line x y k) ∧ (∃ u v, parabola u v ∧ u ≠ x ∧ line u v k) ↔ -1 < k ∧ k < -1/2 := sorry

-- Part 2: Prove the equations of line l when it intersects parabola C at only one point are y = 1, y = -x - 1, and y = -1/2 x
theorem part2 (k : ℝ) : 
  (∃! x y, parabola x y ∧ line x y k) ↔ (k = 0 ∨ k = -1 ∨ k = -1/2) := sorry

end NUMINAMATH_GPT_part1_part2_l1433_143359


namespace NUMINAMATH_GPT_radius_of_smaller_base_of_truncated_cone_l1433_143351

theorem radius_of_smaller_base_of_truncated_cone 
  (r1 r2 r3 : ℕ) (touching : 2 * r1 = r2 ∧ r1 + r3 = r2 * 2):
  (∀ (R : ℕ), R = 6) :=
sorry

end NUMINAMATH_GPT_radius_of_smaller_base_of_truncated_cone_l1433_143351


namespace NUMINAMATH_GPT_expression_value_l1433_143343

theorem expression_value :
  ( (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) ) = 1 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1433_143343


namespace NUMINAMATH_GPT_find_k_values_l1433_143328

/-- 
Prove that the values of k such that the positive difference between the 
roots of 3x^2 + 5x + k = 0 equals the sum of the squares of the roots 
are exactly (70 + 10sqrt(33))/8 and (70 - 10sqrt(33))/8.
-/
theorem find_k_values (k : ℝ) :
  (∀ (a b : ℝ), (3 * a^2 + 5 * a + k = 0 ∧ 3 * b^2 + 5 * b + k = 0 ∧ |a - b| = a^2 + b^2))
  ↔ (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end NUMINAMATH_GPT_find_k_values_l1433_143328


namespace NUMINAMATH_GPT_sam_age_two_years_ago_l1433_143344

variables (S J : ℕ)
variables (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9))

theorem sam_age_two_years_ago : S - 2 = 7 := by
  sorry

end NUMINAMATH_GPT_sam_age_two_years_ago_l1433_143344


namespace NUMINAMATH_GPT_percent_time_in_meetings_l1433_143374

theorem percent_time_in_meetings
  (work_day_minutes : ℕ := 8 * 60)
  (first_meeting_minutes : ℕ := 30)
  (second_meeting_minutes : ℕ := 3 * 30) :
  (first_meeting_minutes + second_meeting_minutes) / work_day_minutes * 100 = 25 :=
by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_percent_time_in_meetings_l1433_143374


namespace NUMINAMATH_GPT_angle_in_fourth_quadrant_l1433_143389

theorem angle_in_fourth_quadrant (θ : ℝ) (h : θ = -1445) : (θ % 360) > 270 ∧ (θ % 360) < 360 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_fourth_quadrant_l1433_143389


namespace NUMINAMATH_GPT_incorrect_statement_implies_m_eq_zero_l1433_143326

theorem incorrect_statement_implies_m_eq_zero
  (m : ℝ)
  (y : ℝ → ℝ)
  (h : ∀ x, y x = m * x + 4 * m - 2)
  (intersects_y_axis_at : y 0 = -2) :
  m = 0 :=
sorry

end NUMINAMATH_GPT_incorrect_statement_implies_m_eq_zero_l1433_143326


namespace NUMINAMATH_GPT_main_theorem_l1433_143385

variable (a : ℝ)

def M : Set ℝ := {x | x > 1 / 2 ∧ x < 1} ∪ {x | x > 1}
def N : Set ℝ := {x | x > 0 ∧ x ≤ 1 / 2}

theorem main_theorem : M ∩ N = ∅ :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l1433_143385


namespace NUMINAMATH_GPT_clothing_factory_exceeded_tasks_l1433_143393

theorem clothing_factory_exceeded_tasks :
  let first_half := (2 : ℚ) / 3
  let second_half := (3 : ℚ) / 5
  first_half + second_half - 1 = (4 : ℚ) / 15 :=
by
  sorry

end NUMINAMATH_GPT_clothing_factory_exceeded_tasks_l1433_143393


namespace NUMINAMATH_GPT_points_earned_l1433_143363

-- Define the given conditions
def points_per_enemy := 5
def total_enemies := 8
def enemies_remaining := 6

-- Calculate the number of enemies defeated
def enemies_defeated := total_enemies - enemies_remaining

-- Calculate the points earned based on the enemies defeated
theorem points_earned : enemies_defeated * points_per_enemy = 10 := by
  -- Insert mathematical operations
  sorry

end NUMINAMATH_GPT_points_earned_l1433_143363


namespace NUMINAMATH_GPT_number_of_aluminum_atoms_l1433_143381

def molecular_weight (n : ℕ) : ℝ :=
  n * 26.98 + 30.97 + 4 * 16.0

theorem number_of_aluminum_atoms (n : ℕ) (h : molecular_weight n = 122) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_aluminum_atoms_l1433_143381


namespace NUMINAMATH_GPT_speed_conversion_l1433_143306

theorem speed_conversion (speed_kmph : ℝ) (h : speed_kmph = 18) : speed_kmph * (1000 / 3600) = 5 := by
  sorry

end NUMINAMATH_GPT_speed_conversion_l1433_143306


namespace NUMINAMATH_GPT_compute_P_2_4_8_l1433_143384

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry

axiom homogeneity (x y z k : ℝ) : P (k * x) (k * y) (k * z) = (k ^ 4) * P x y z

axiom symmetry (a b c : ℝ) : P a b c = P b c a

axiom zero_cond (a b : ℝ) : P a a b = 0

axiom initial_cond : P 1 2 3 = 1

theorem compute_P_2_4_8 : P 2 4 8 = 56 := sorry

end NUMINAMATH_GPT_compute_P_2_4_8_l1433_143384


namespace NUMINAMATH_GPT_max_non_cyclic_handshakes_l1433_143392

theorem max_non_cyclic_handshakes (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end NUMINAMATH_GPT_max_non_cyclic_handshakes_l1433_143392


namespace NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l1433_143312

theorem cos_double_angle_of_parallel_vectors
  (α : ℝ)
  (a : ℝ × ℝ := (1/3, Real.tan α))
  (b : ℝ × ℝ := (Real.cos α, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  Real.cos (2 * α) = 1 / 9 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l1433_143312


namespace NUMINAMATH_GPT_probability_larry_wins_l1433_143348

noncomputable def P_larry_wins_game : ℝ :=
  let p_hit := (1 : ℝ) / 3
  let p_miss := (2 : ℝ) / 3
  let r := p_miss^3
  (p_hit / (1 - r))

theorem probability_larry_wins :
  P_larry_wins_game = 9 / 19 :=
by
  -- Proof is omitted, but the outline and logic are given in the problem statement
  sorry

end NUMINAMATH_GPT_probability_larry_wins_l1433_143348


namespace NUMINAMATH_GPT_parabola_directrix_l1433_143301

theorem parabola_directrix (y x : ℝ) : y^2 = -8 * x → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1433_143301


namespace NUMINAMATH_GPT_university_diploma_percentage_l1433_143354

-- Define the conditions
variables (P N JD ND : ℝ)
-- P: total population assumed as 100% for simplicity
-- N: percentage of people with university diploma
-- JD: percentage of people who have the job of their choice
-- ND: percentage of people who do not have a university diploma but have the job of their choice
variables (A : ℝ) -- A: University diploma percentage of those who do not have the job of their choice
variable (total_diploma : ℝ)
axiom country_Z_conditions : 
  (P = 100) ∧ (ND = 18) ∧ (JD = 40) ∧ (A = 25)

-- Define the proof problem
theorem university_diploma_percentage :
  (N = ND + (JD - ND) + (total_diploma * (P - JD * (P / JD) / P))) →
  N = 37 :=
by
  sorry

end NUMINAMATH_GPT_university_diploma_percentage_l1433_143354


namespace NUMINAMATH_GPT_number_of_diet_soda_l1433_143336

variable (d r : ℕ)

-- Define the conditions of the problem
def condition1 : Prop := r = d + 79
def condition2 : Prop := r = 83

-- State the theorem we want to prove
theorem number_of_diet_soda (h1 : condition1 d r) (h2 : condition2 r) : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_diet_soda_l1433_143336


namespace NUMINAMATH_GPT_abs_h_of_roots_sum_squares_eq_34_l1433_143317

theorem abs_h_of_roots_sum_squares_eq_34 
  (h : ℝ)
  (h_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0)) 
  (sum_of_squares_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0) → r^2 + s^2 = 34) :
  |h| = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_h_of_roots_sum_squares_eq_34_l1433_143317


namespace NUMINAMATH_GPT_indefinite_integral_solution_l1433_143360

open Real

theorem indefinite_integral_solution (c : ℝ) : 
  ∫ x, (1 - cos x) / (x - sin x) ^ 2 = - 1 / (x - sin x) + c := 
sorry

end NUMINAMATH_GPT_indefinite_integral_solution_l1433_143360


namespace NUMINAMATH_GPT_nancy_carrots_next_day_l1433_143313

-- Definitions based on conditions
def carrots_picked_on_first_day : Nat := 12
def carrots_thrown_out : Nat := 2
def total_carrots_after_two_days : Nat := 31

-- Problem statement
theorem nancy_carrots_next_day :
  let carrots_left_after_first_day := carrots_picked_on_first_day - carrots_thrown_out
  let carrots_picked_next_day := total_carrots_after_two_days - carrots_left_after_first_day
  carrots_picked_next_day = 21 :=
by
  sorry

end NUMINAMATH_GPT_nancy_carrots_next_day_l1433_143313


namespace NUMINAMATH_GPT_equation_of_l2_l1433_143321

-- Define the initial line equation
def l1 (x : ℝ) : ℝ := -2 * x - 2

-- Define the transformed line equation after translation
def l2 (x : ℝ) : ℝ := l1 (x + 1) + 2

-- Statement to prove
theorem equation_of_l2 : ∀ x, l2 x = -2 * x - 2 := by
  sorry

end NUMINAMATH_GPT_equation_of_l2_l1433_143321


namespace NUMINAMATH_GPT_certain_fraction_exists_l1433_143340

theorem certain_fraction_exists (a b : ℚ) (h : a / b = 3 / 4) :
  (a / b) / (1 / 5) = (3 / 4) / (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_certain_fraction_exists_l1433_143340


namespace NUMINAMATH_GPT_correct_statement_3_l1433_143329

-- Definitions
def acute_angles (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_less_than_90 (θ : ℝ) : Prop := θ < 90
def angles_in_first_quadrant (θ : ℝ) : Prop := ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90

-- Sets
def M := {θ | acute_angles θ}
def N := {θ | angles_less_than_90 θ}
def P := {θ | angles_in_first_quadrant θ}

-- Proof statement
theorem correct_statement_3 : M ⊆ P := sorry

end NUMINAMATH_GPT_correct_statement_3_l1433_143329


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1433_143304

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1433_143304


namespace NUMINAMATH_GPT_abs_nonneg_position_l1433_143386

theorem abs_nonneg_position (a : ℝ) : 0 ≤ |a| ∧ |a| ≥ 0 → (exists x : ℝ, x = |a| ∧ x ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_abs_nonneg_position_l1433_143386


namespace NUMINAMATH_GPT_max_soap_boxes_in_carton_l1433_143395

-- Define the measurements of the carton
def L_carton := 25
def W_carton := 42
def H_carton := 60

-- Define the measurements of the soap box
def L_soap_box := 7
def W_soap_box := 12
def H_soap_box := 5

-- Calculate the volume of the carton
def V_carton := L_carton * W_carton * H_carton

-- Calculate the volume of the soap box
def V_soap_box := L_soap_box * W_soap_box * H_soap_box

-- Define the number of soap boxes that can fit in the carton
def number_of_soap_boxes := V_carton / V_soap_box

-- Prove that the number of soap boxes that can fit in the carton is 150
theorem max_soap_boxes_in_carton : number_of_soap_boxes = 150 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_max_soap_boxes_in_carton_l1433_143395


namespace NUMINAMATH_GPT_base_six_four_digit_odd_final_l1433_143373

theorem base_six_four_digit_odd_final :
  ∃ b : ℕ, (b^4 > 285 ∧ 285 ≥ b^3 ∧ (285 % b) % 2 = 1) :=
by 
  use 6
  sorry

end NUMINAMATH_GPT_base_six_four_digit_odd_final_l1433_143373


namespace NUMINAMATH_GPT_max_possible_value_l1433_143345

-- Define the expressions and the conditions
def expr1 := 10 * 10
def expr2 := 10 / 10
def expr3 := expr1 + 10
def expr4 := expr3 - expr2

-- Define our main statement that asserts the maximum value is 109
theorem max_possible_value: expr4 = 109 := by
  sorry

end NUMINAMATH_GPT_max_possible_value_l1433_143345


namespace NUMINAMATH_GPT_fg_of_3_l1433_143396

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 3 * x

theorem fg_of_3 : f (g 3) = -2 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_l1433_143396


namespace NUMINAMATH_GPT_smallest_integer_with_20_divisors_l1433_143311

theorem smallest_integer_with_20_divisors : ∃ n : ℕ, 
  (0 < n) ∧ 
  (∀ m : ℕ, (0 < m ∧ ∃ k : ℕ, m = n * k) ↔ (∃ d : ℕ, d.succ * (20 / d.succ) = 20)) ∧ 
  n = 240 := 
sorry

end NUMINAMATH_GPT_smallest_integer_with_20_divisors_l1433_143311


namespace NUMINAMATH_GPT_range_of_a_l1433_143319

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Lean statement for the problem
theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : -1 < a ∧ a ≤ 1 := 
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_range_of_a_l1433_143319


namespace NUMINAMATH_GPT_distance_between_incenter_and_circumcenter_of_right_triangle_l1433_143368

theorem distance_between_incenter_and_circumcenter_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (right_triangle : a^2 + b^2 = c^2) :
    ∃ (IO : ℝ), IO = Real.sqrt 5 :=
by
  rw [h1, h2, h3] at right_triangle
  have h_sum : 6^2 + 8^2 = 10^2 := by sorry
  exact ⟨Real.sqrt 5, by sorry⟩

end NUMINAMATH_GPT_distance_between_incenter_and_circumcenter_of_right_triangle_l1433_143368


namespace NUMINAMATH_GPT_altitude_length_l1433_143370

noncomputable def length_of_altitude (l w : ℝ) : ℝ :=
  2 * l * w / Real.sqrt (l ^ 2 + w ^ 2)

theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  ∃ h : ℝ, h = length_of_altitude l w := by
  sorry

end NUMINAMATH_GPT_altitude_length_l1433_143370


namespace NUMINAMATH_GPT_fraction_of_l1433_143339

theorem fraction_of (a b : ℚ) (h_a : a = 3/4) (h_b : b = 1/6) : b / a = 2/9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_l1433_143339


namespace NUMINAMATH_GPT_sam_initial_nickels_l1433_143346

variable (n_now n_given n_initial : Nat)

theorem sam_initial_nickels (h_now : n_now = 63) (h_given : n_given = 39) (h_relation : n_now = n_initial + n_given) : n_initial = 24 :=
by
  sorry

end NUMINAMATH_GPT_sam_initial_nickels_l1433_143346


namespace NUMINAMATH_GPT_possible_ways_to_choose_gates_l1433_143377

theorem possible_ways_to_choose_gates : 
  ∃! (ways : ℕ), ways = 20 := 
by
  sorry

end NUMINAMATH_GPT_possible_ways_to_choose_gates_l1433_143377


namespace NUMINAMATH_GPT_factorization_of_expression_l1433_143369

theorem factorization_of_expression (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l1433_143369


namespace NUMINAMATH_GPT_area_bounded_region_l1433_143382

theorem area_bounded_region : 
  (∃ x y : ℝ, x^2 + y^2 = 2 * abs (x - y) + 2 * abs (x + y)) →
  (bounded_area : ℝ) = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_bounded_region_l1433_143382


namespace NUMINAMATH_GPT_base_of_parallelogram_l1433_143303

theorem base_of_parallelogram (area height base : ℝ) 
  (h_area : area = 320)
  (h_height : height = 16) :
  base = area / height :=
by 
  rw [h_area, h_height]
  norm_num
  sorry

end NUMINAMATH_GPT_base_of_parallelogram_l1433_143303


namespace NUMINAMATH_GPT_opposite_neg_two_is_two_l1433_143342

theorem opposite_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_neg_two_is_two_l1433_143342


namespace NUMINAMATH_GPT_correct_calculation_l1433_143391

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1433_143391


namespace NUMINAMATH_GPT_sum_common_divisors_60_18_l1433_143308

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_common_divisors_60_18_l1433_143308


namespace NUMINAMATH_GPT_sufficiency_not_necessity_l1433_143352

def l1 : Type := sorry
def l2 : Type := sorry

def skew_lines (l1 l2 : Type) : Prop := sorry
def do_not_intersect (l1 l2 : Type) : Prop := sorry

theorem sufficiency_not_necessity (p q : Prop) 
  (hp : p = skew_lines l1 l2)
  (hq : q = do_not_intersect l1 l2) :
  (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficiency_not_necessity_l1433_143352


namespace NUMINAMATH_GPT_perimeter_of_figure_l1433_143362

variable (x y : ℝ)
variable (lengths : Set ℝ)
variable (perpendicular_adjacent : Prop)
variable (area : ℝ)

-- Conditions
def condition_1 : Prop := ∀ l ∈ lengths, l = x ∨ l = y
def condition_2 : Prop := perpendicular_adjacent
def condition_3 : Prop := area = 252
def condition_4 : Prop := x = 2 * y

-- Problem statement
theorem perimeter_of_figure
  (h1 : condition_1 x y lengths)
  (h2 : condition_2 perpendicular_adjacent)
  (h3 : condition_3 area)
  (h4 : condition_4 x y) :
  ∃ perimeter : ℝ, perimeter = 96 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_l1433_143362


namespace NUMINAMATH_GPT_sum_of_three_distinct_integers_product_625_l1433_143375

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_distinct_integers_product_625_l1433_143375


namespace NUMINAMATH_GPT_find_digits_l1433_143334

theorem find_digits (a b c d : ℕ) 
  (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9)
  (h₂ : 0 ≤ c ∧ c ≤ 9)
  (h₃ : 0 ≤ d ∧ d ≤ 9)
  (h₄ : (10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) :
  1000 * a + 100 * b + 10 * c + d = 2315 :=
by
  sorry

end NUMINAMATH_GPT_find_digits_l1433_143334
