import Mathlib

namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1249_124910

theorem quadratic_inequality_solution (a b c : ℝ) (h : a < 0) 
  (h_sol : ∀ x, ax^2 + bx + c > 0 ↔ x > -2 ∧ x < 1) :
  ∀ x, ax^2 + (a + b) * x + c - a < 0 ↔ x < -3 ∨ x > 1 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1249_124910


namespace NUMINAMATH_GPT_tetrahedron_condition_proof_l1249_124935

/-- Define the conditions for the necessary and sufficient condition for each k -/
def tetrahedron_condition (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => a < Real.sqrt 3
  | 2 => Real.sqrt (2 - Real.sqrt 3) < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => a < Real.sqrt 3
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False -- not applicable for other values of k

/-- Prove that the condition is valid for given a and k -/
theorem tetrahedron_condition_proof (a : ℝ) (k : ℕ) : tetrahedron_condition a k := 
  by
  sorry

end NUMINAMATH_GPT_tetrahedron_condition_proof_l1249_124935


namespace NUMINAMATH_GPT_rearrange_cards_l1249_124928

theorem rearrange_cards :
  (∀ (arrangement : List ℕ), arrangement = [3, 1, 2, 4, 5, 6] ∨ arrangement = [1, 2, 4, 5, 6, 3] →
  (∀ card, card ∈ arrangement → List.erase arrangement card = [1, 2, 4, 5, 6] ∨
                                        List.erase arrangement card = [3, 1, 2, 4, 5]) →
  List.length arrangement = 6) →
  (∃ n, n = 10) :=
by
  sorry

end NUMINAMATH_GPT_rearrange_cards_l1249_124928


namespace NUMINAMATH_GPT_misha_is_older_l1249_124912

-- Definitions for the conditions
def tanya_age_19_months_ago : ℕ := 16
def months_ago_for_tanya : ℕ := 19
def misha_age_in_16_months : ℕ := 19
def months_ahead_for_misha : ℕ := 16

-- Convert months to years and residual months
def months_to_years_months (m : ℕ) : ℕ × ℕ := (m / 12, m % 12)

-- Computation for Tanya's current age
def tanya_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ago_for_tanya
  (tanya_age_19_months_ago + years, months)

-- Computation for Misha's current age
def misha_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ahead_for_misha
  (misha_age_in_16_months - years, months)

-- Proof statement
theorem misha_is_older : misha_age_now > tanya_age_now := by
  sorry

end NUMINAMATH_GPT_misha_is_older_l1249_124912


namespace NUMINAMATH_GPT_largest_value_l1249_124966

theorem largest_value :
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  -- conditions
  let A := 3 + 1 + 4
  let B := 3 * 1 + 4
  let C := 3 + 1 * 4
  let D := 3 * 1 * 4
  let E := 3 + 0 * 1 + 4
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_largest_value_l1249_124966


namespace NUMINAMATH_GPT_evaluate_g_at_neg2_l1249_124917

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 35 * x^2 - 28 * x - 84

theorem evaluate_g_at_neg2 : g (-2) = 320 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg2_l1249_124917


namespace NUMINAMATH_GPT_midpoint_P_AB_l1249_124977

structure Point := (x : ℝ) (y : ℝ)

def segment_midpoint (P A B : Point) : Prop := P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

variables {A D C E P B : Point}
variables (h1 : A.x = D.x ∧ A.y = D.y)
variables (h2 : D.x = C.x ∧ D.y = C.y)
variables (h3 : D.x = P.x ∧ D.y = P.y ∧ P.x = E.x ∧ P.y = E.y)
variables (h4 : B.x = E.x ∧ B.y = E.y)
variables (h5 : A.x = C.x ∧ A.y = C.y)
variables (angle_ADC : ∀ x y : ℝ, (x - A.x)^2 + (y - A.y)^2 = (x - D.x)^2 + (y - D.y)^2 → (x - C.x)^2 + (y - C.y)^2 = (x - D.x)^2 + (y - D.y)^2)
variables (angle_DPE : ∀ x y : ℝ, (x - D.x)^2 + (y - P.y)^2 = (x - P.x)^2 + (y - E.y)^2 → (x - E.x)^2 + (y - E.y)^2 = (x - P.x)^2 + (y - E.y)^2)
variables (angle_BEC : ∀ x y : ℝ, (x - B.x)^2 + (y - E.y)^2 = (x - E.x)^2 + (y - C.y)^2 → (x - B.x)^2 + (y - C.y)^2 = (x - E.x)^2 + (y - C.y)^2)

theorem midpoint_P_AB : segment_midpoint P A B := 
sorry

end NUMINAMATH_GPT_midpoint_P_AB_l1249_124977


namespace NUMINAMATH_GPT_arrange_order_l1249_124927

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := Real.cos 2

theorem arrange_order : c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_arrange_order_l1249_124927


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1249_124999

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) : x^2 + y^2 = 109 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1249_124999


namespace NUMINAMATH_GPT_total_weight_of_8_bags_total_sales_amount_of_qualified_products_l1249_124931

-- Definitions
def deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def standard_weight_per_bag : ℤ := 450
def threshold : ℤ := 4
def price_per_bag : ℤ := 3

-- Part 1: Total weight of the 8 bags of laundry detergent
theorem total_weight_of_8_bags : 
  8 * standard_weight_per_bag + deviations.sum = 3598 := 
by
  sorry

-- Part 2: Total sales amount of qualified products
theorem total_sales_amount_of_qualified_products : 
  price_per_bag * (deviations.filter (fun x => abs x ≤ threshold)).length = 18 := 
by
  sorry

end NUMINAMATH_GPT_total_weight_of_8_bags_total_sales_amount_of_qualified_products_l1249_124931


namespace NUMINAMATH_GPT_parametric_to_standard_l1249_124941

theorem parametric_to_standard (theta : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = 1 + 2 * Real.cos theta)
  (h2 : y = -2 + 2 * Real.sin theta) :
  (x - 1)^2 + (y + 2)^2 = 4 :=
sorry

end NUMINAMATH_GPT_parametric_to_standard_l1249_124941


namespace NUMINAMATH_GPT_equations_of_motion_l1249_124947

-- Initial conditions and setup
def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 45

-- Questions:
-- 1. Equations of motion for point M
-- 2. Equation of the trajectory of point M
-- 3. Velocity of point M

theorem equations_of_motion (t : ℝ) :
  let xM := 45 * (1 + Real.cos (omega * t))
  let yM := 45 * Real.sin (omega * t)
  xM = 45 * (1 + Real.cos (omega * t)) ∧
  yM = 45 * Real.sin (omega * t) ∧
  ((yM / 45) ^ 2 + ((xM - 45) / 45) ^ 2 = 1) ∧
  let vMx := -450 * Real.sin (omega * t)
  let vMy := 450 * Real.cos (omega * t)
  (vMx = -450 * Real.sin (omega * t)) ∧
  (vMy = 450 * Real.cos (omega * t)) :=
by
  sorry

end NUMINAMATH_GPT_equations_of_motion_l1249_124947


namespace NUMINAMATH_GPT_trig_expression_value_l1249_124957

open Real

theorem trig_expression_value (x : ℝ) (h : tan (π - x) = -2) : 
  4 * sin x ^ 2 - 3 * sin x * cos x - 5 * cos x ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_trig_expression_value_l1249_124957


namespace NUMINAMATH_GPT_initial_amount_of_money_l1249_124956

-- Define the costs and purchased quantities
def cost_tshirt : ℕ := 8
def cost_keychain_set : ℕ := 2
def cost_bag : ℕ := 10
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

-- Define derived quantities
def sets_of_keychains_bought : ℕ := keychains_bought / 3

-- Define the total costs
def total_cost_tshirts : ℕ := tshirts_bought * cost_tshirt
def total_cost_bags : ℕ := bags_bought * cost_bag
def total_cost_keychains : ℕ := sets_of_keychains_bought * cost_keychain_set

-- Define the initial amount of money
def total_initial_amount : ℕ := total_cost_tshirts + total_cost_bags + total_cost_keychains

-- The theorem proving the initial amount Timothy had
theorem initial_amount_of_money : total_initial_amount = 50 := by
  -- The proof is not required, so we use sorry to skip it
  sorry

end NUMINAMATH_GPT_initial_amount_of_money_l1249_124956


namespace NUMINAMATH_GPT_range_of_x_in_function_l1249_124985

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end NUMINAMATH_GPT_range_of_x_in_function_l1249_124985


namespace NUMINAMATH_GPT_sum_of_numbers_l1249_124967

open Function

theorem sum_of_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : b = 8) 
  (h4 : (a + b + c) / 3 = a + 7) 
  (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 63 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1249_124967


namespace NUMINAMATH_GPT_completing_square_solution_l1249_124903

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_completing_square_solution_l1249_124903


namespace NUMINAMATH_GPT_factor_of_increase_l1249_124951

noncomputable def sum_arithmetic_progression (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem factor_of_increase (a1 d n : ℕ) (h1 : a1 > 0) (h2 : (sum_arithmetic_progression a1 (3 * d) n = 2 * sum_arithmetic_progression a1 d n)) :
  sum_arithmetic_progression a1 (4 * d) n = (5 / 2) * sum_arithmetic_progression a1 d n :=
sorry

end NUMINAMATH_GPT_factor_of_increase_l1249_124951


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l1249_124920

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (x^2 - 2*x + 3 = (x-1)^2 + 2) :=
by intro x; sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l1249_124920


namespace NUMINAMATH_GPT_find_c_l1249_124948

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end NUMINAMATH_GPT_find_c_l1249_124948


namespace NUMINAMATH_GPT_rent_of_first_apartment_l1249_124962

theorem rent_of_first_apartment (R : ℝ) :
  let cost1 := R + 260 + (31 * 20 * 0.58)
  let cost2 := 900 + 200 + (21 * 20 * 0.58)
  (cost1 - cost2 = 76) → R = 800 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rent_of_first_apartment_l1249_124962


namespace NUMINAMATH_GPT_exists_x_y_not_divisible_by_3_l1249_124933

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (h_pos : 0 < k) :
  ∃ x y : ℤ, (x^2 + 2 * y^2 = 3^k) ∧ (x % 3 ≠ 0) ∧ (y % 3 ≠ 0) := 
sorry

end NUMINAMATH_GPT_exists_x_y_not_divisible_by_3_l1249_124933


namespace NUMINAMATH_GPT_max_k_divides_expression_l1249_124916

theorem max_k_divides_expression : ∃ k, (∀ n : ℕ, n > 0 → 2^k ∣ (3^(2*n + 3) + 40*n - 27)) ∧ k = 6 :=
sorry

end NUMINAMATH_GPT_max_k_divides_expression_l1249_124916


namespace NUMINAMATH_GPT_slope_of_regression_line_l1249_124914

variable (h : ℝ)
variable (t1 T1 t2 T2 t3 T3 : ℝ)

-- Given conditions.
axiom t2_is_equally_spaced : t2 = t1 + h
axiom t3_is_equally_spaced : t3 = t1 + 2 * h

theorem slope_of_regression_line :
  t2 = t1 + h →
  t3 = t1 + 2 * h →
  (T3 - T1) / (t3 - t1) = (T3 - T1) / ((t1 + 2 * h) - t1) := 
by
  sorry

end NUMINAMATH_GPT_slope_of_regression_line_l1249_124914


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1249_124923

theorem sum_of_squares_of_roots (x₁ x₂ : ℚ) (h : 6 * x₁^2 - 9 * x₁ + 5 = 0 ∧ 6 * x₂^2 - 9 * x₂ + 5 = 0 ∧ x₁ ≠ x₂) : x₁^2 + x₂^2 = 7 / 12 :=
by
  -- Since we are only required to write the statement, we leave the proof as sorry
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1249_124923


namespace NUMINAMATH_GPT_max_ab_condition_l1249_124955

-- Define the circles and the tangency condition
def circle1 (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}
def circle2 (b : ℝ) : Set (ℝ × ℝ) := {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}
def internally_tangent (a b : ℝ) : Prop := (a + b) ^ 2 = 1

-- Define the maximum value condition
def max_ab (a b : ℝ) : ℝ := a * b

-- Main theorem
theorem max_ab_condition {a b : ℝ} (h_tangent : internally_tangent a b) : max_ab a b ≤ 1 / 4 :=
by
  -- Proof steps are not necessary, so we use sorry to end the proof.
  sorry

end NUMINAMATH_GPT_max_ab_condition_l1249_124955


namespace NUMINAMATH_GPT_part1_part2_l1249_124952

theorem part1 (m : ℝ) (P : ℝ × ℝ) : (P = (3*m - 6, m + 1)) → (P.1 = 0) → (P = (0, 3)) :=
by
  sorry

theorem part2 (m : ℝ) (A P : ℝ × ℝ) : A = (1, -2) → (P = (3*m - 6, m + 1)) → (P.2 = A.2) → (P = (-15, -2)) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1249_124952


namespace NUMINAMATH_GPT_smallest_xy_l1249_124974

theorem smallest_xy :
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / (3 * y) = 1 / 6) ∧ (∀ (x' y' : ℕ), (0 < x') ∧ (0 < y') ∧ (1 / x' + 1 / (3 * y') = 1 / 6) → x' * y' ≥ x * y) ∧ x * y = 48 :=
sorry

end NUMINAMATH_GPT_smallest_xy_l1249_124974


namespace NUMINAMATH_GPT_anna_coaching_days_l1249_124978

/-- The total number of days from January 1 to September 4 in a non-leap year -/
def total_days_in_non_leap_year_up_to_sept4 : ℕ :=
  let days_in_january := 31
  let days_in_february := 28
  let days_in_march := 31
  let days_in_april := 30
  let days_in_may := 31
  let days_in_june := 30
  let days_in_july := 31
  let days_in_august := 31
  let days_up_to_sept4 := 4
  days_in_january + days_in_february + days_in_march + days_in_april +
  days_in_may + days_in_june + days_in_july + days_in_august + days_up_to_sept4

theorem anna_coaching_days : total_days_in_non_leap_year_up_to_sept4 = 247 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_anna_coaching_days_l1249_124978


namespace NUMINAMATH_GPT_original_selling_price_l1249_124953

theorem original_selling_price (P : ℝ) (h : 0.7 * P = 560) : P = 800 :=
by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1249_124953


namespace NUMINAMATH_GPT_calc_expr_l1249_124959

theorem calc_expr : (3^5 * 6^3 + 3^3) = 52515 := by
  sorry

end NUMINAMATH_GPT_calc_expr_l1249_124959


namespace NUMINAMATH_GPT_average_visitors_per_day_l1249_124996

theorem average_visitors_per_day (avg_sunday : ℕ) (avg_other_day : ℕ) (days_in_month : ℕ) (starts_on_sunday : Bool) :
  avg_sunday = 570 →
  avg_other_day = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  (5 * avg_sunday + 25 * avg_other_day) / days_in_month = 295 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l1249_124996


namespace NUMINAMATH_GPT_price_of_turban_correct_l1249_124921

noncomputable def initial_yearly_salary : ℝ := 90
noncomputable def initial_monthly_salary : ℝ := initial_yearly_salary / 12
noncomputable def raise : ℝ := 0.05 * initial_monthly_salary

noncomputable def first_3_months_salary : ℝ := 3 * initial_monthly_salary
noncomputable def second_3_months_salary : ℝ := 3 * (initial_monthly_salary + raise)
noncomputable def third_3_months_salary : ℝ := 3 * (initial_monthly_salary + 2 * raise)

noncomputable def total_cash_salary : ℝ := first_3_months_salary + second_3_months_salary + third_3_months_salary
noncomputable def actual_cash_received : ℝ := 80
noncomputable def price_of_turban : ℝ := actual_cash_received - total_cash_salary

theorem price_of_turban_correct : price_of_turban = 9.125 :=
by
  sorry

end NUMINAMATH_GPT_price_of_turban_correct_l1249_124921


namespace NUMINAMATH_GPT_compute_K_l1249_124925

theorem compute_K (P Q T N K : ℕ) (x y z : ℕ) 
  (hP : P * x + Q * y = z) 
  (hT : T * x + N * y = z)
  (hK : K * x = z)
  (h_unique : P > 0 ∧ Q > 0 ∧ T > 0 ∧ N > 0 ∧ K > 0) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end NUMINAMATH_GPT_compute_K_l1249_124925


namespace NUMINAMATH_GPT_smallest_points_to_exceed_mean_l1249_124997

theorem smallest_points_to_exceed_mean (X y : ℕ) (h_scores : 24 + 17 + 25 = 66) 
  (h_mean_9_gt_mean_6 : X / 6 < (X + 66) / 9) (h_mean_10_gt_22 : (X + 66 + y) / 10 > 22) 
  : y ≥ 24 := by
  sorry

end NUMINAMATH_GPT_smallest_points_to_exceed_mean_l1249_124997


namespace NUMINAMATH_GPT_smallest_integer_k_distinct_real_roots_l1249_124981

theorem smallest_integer_k_distinct_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, x^2 - x + 2 - k = 0 → x ≠ 0) ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_k_distinct_real_roots_l1249_124981


namespace NUMINAMATH_GPT_one_cow_eating_one_bag_in_12_days_l1249_124939

def average_days_to_eat_one_bag (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) : ℕ :=
  total_days / (total_bags / number_of_cows)

theorem one_cow_eating_one_bag_in_12_days (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) (h_total_bags : total_bags = 50) (h_total_days : total_days = 20) (h_number_of_cows : number_of_cows = 30) : 
  average_days_to_eat_one_bag total_bags total_days number_of_cows = 12 := by
  sorry

end NUMINAMATH_GPT_one_cow_eating_one_bag_in_12_days_l1249_124939


namespace NUMINAMATH_GPT_max_take_home_pay_at_5000_dollars_l1249_124988

noncomputable def income_tax (x : ℕ) : ℕ :=
  if x ≤ 5000 then x * 5 / 100
  else 250 + 10 * ((x - 5000 / 1000) - 5) ^ 2

noncomputable def take_home_pay (y : ℕ) : ℕ :=
  y - income_tax y

theorem max_take_home_pay_at_5000_dollars : ∀ y : ℕ, take_home_pay y ≤ take_home_pay 5000 := by
  sorry

end NUMINAMATH_GPT_max_take_home_pay_at_5000_dollars_l1249_124988


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1249_124902

-- Part (a)
theorem part_a (x y : ℕ) (h : (2 * x + 11 * y) = 3 * x + 4 * y) : x = 7 * y := by
  sorry

-- Part (b)
theorem part_b (u v : ℚ) : ∃ (x y : ℚ), (x + y) / 2 = (u.num * v.den + v.num * u.den) / (2 * u.den * v.den) := by
  sorry

-- Part (c)
theorem part_c (u v : ℚ) (h : u < v) : ∀ (m : ℚ), (m.num = u.num + v.num) ∧ (m.den = u.den + v.den) → u < m ∧ m < v := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1249_124902


namespace NUMINAMATH_GPT_secretary_longest_time_l1249_124984

theorem secretary_longest_time (h_ratio : ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y = 2 * x ∧ z = 3 * x ∧ (5 * x = 40)) :
  5 * x = 40 := sorry

end NUMINAMATH_GPT_secretary_longest_time_l1249_124984


namespace NUMINAMATH_GPT_puppy_weight_l1249_124940

theorem puppy_weight (a b c : ℕ) 
  (h1 : a + b + c = 24) 
  (h2 : a + c = 2 * b) 
  (h3 : a + b = c) : 
  a = 4 :=
sorry

end NUMINAMATH_GPT_puppy_weight_l1249_124940


namespace NUMINAMATH_GPT_div_power_sub_one_l1249_124904

theorem div_power_sub_one : 11 * 31 * 61 ∣ 20^15 - 1 := 
by
  sorry

end NUMINAMATH_GPT_div_power_sub_one_l1249_124904


namespace NUMINAMATH_GPT_longer_diagonal_of_rhombus_l1249_124932

theorem longer_diagonal_of_rhombus {a b d1 : ℕ} (h1 : a = b) (h2 : a = 65) (h3 : d1 = 60) : 
  ∃ d2, (d2^2) = (2 * (a^2) - (d1^2)) ∧ d2 = 110 :=
by
  sorry

end NUMINAMATH_GPT_longer_diagonal_of_rhombus_l1249_124932


namespace NUMINAMATH_GPT_bus_stop_time_l1249_124943

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) (h1: speed_without_stoppages = 48) (h2: speed_with_stoppages = 24) :
  ∃ (minutes_stopped_per_hour : ℝ), minutes_stopped_per_hour = 30 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l1249_124943


namespace NUMINAMATH_GPT_initial_pencils_correct_l1249_124911

variable (pencils_taken remaining_pencils initial_pencils : ℕ)

def initial_number_of_pencils (pencils_taken remaining_pencils : ℕ) : ℕ :=
  pencils_taken + remaining_pencils

theorem initial_pencils_correct (h₁ : pencils_taken = 22) (h₂ : remaining_pencils = 12) :
  initial_number_of_pencils pencils_taken remaining_pencils = 34 := by
  rw [h₁, h₂]
  rfl

end NUMINAMATH_GPT_initial_pencils_correct_l1249_124911


namespace NUMINAMATH_GPT_parabola_addition_l1249_124964

def f (a b c x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c
def g (a b c x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

theorem parabola_addition (a b c x : ℝ) : 
  (f a b c x + g a b c x) = (2 * a * x^2 + 2 * c - 7 * b) :=
by
  sorry

end NUMINAMATH_GPT_parabola_addition_l1249_124964


namespace NUMINAMATH_GPT_number_of_zeros_f_l1249_124913

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^2 - x - 1

-- The theorem statement that proves the function has exactly two zeros
theorem number_of_zeros_f : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_f_l1249_124913


namespace NUMINAMATH_GPT_isosceles_right_triangle_quotient_l1249_124938

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_isosceles_right_triangle_quotient_l1249_124938


namespace NUMINAMATH_GPT_difference_in_interest_rates_l1249_124992

-- Definitions
def Principal : ℝ := 2300
def Time : ℝ := 3
def ExtraInterest : ℝ := 69

-- The difference in rates
theorem difference_in_interest_rates (R dR : ℝ) (h : (Principal * (R + dR) * Time) / 100 =
    (Principal * R * Time) / 100 + ExtraInterest) : dR = 1 :=
  sorry

end NUMINAMATH_GPT_difference_in_interest_rates_l1249_124992


namespace NUMINAMATH_GPT_parabola_focus_l1249_124900

theorem parabola_focus (x y : ℝ) : (y = x^2 / 8) → (y = x^2 / 8) ∧ (∃ p, p = (0, 2)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1249_124900


namespace NUMINAMATH_GPT_find_b_l1249_124906

theorem find_b (a b c : ℚ) (h : (3 * x^2 - 4 * x + 2) * (a * x^2 + b * x + c) = 9 * x^4 - 10 * x^3 + 5 * x^2 - 8 * x + 4)
  (ha : a = 3) : b = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1249_124906


namespace NUMINAMATH_GPT_percentage_difference_height_l1249_124972

-- Define the heights of persons B, A, and C
variables (H_B H_A H_C : ℝ)

-- Condition: Person A's height is 30% less than person B's height
def person_A_height : Prop := H_A = 0.70 * H_B

-- Condition: Person C's height is 20% more than person A's height
def person_C_height : Prop := H_C = 1.20 * H_A

-- The proof problem: Prove that the percentage difference between H_B and H_C is 16%
theorem percentage_difference_height (h1 : person_A_height H_B H_A) (h2 : person_C_height H_A H_C) :
  ((H_B - H_C) / H_B) * 100 = 16 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_height_l1249_124972


namespace NUMINAMATH_GPT_floor_div_add_floor_div_succ_eq_l1249_124989

theorem floor_div_add_floor_div_succ_eq (n : ℤ) : 
  (⌊(n : ℝ)/2⌋ + ⌊(n + 1 : ℝ)/2⌋ : ℤ) = n := 
sorry

end NUMINAMATH_GPT_floor_div_add_floor_div_succ_eq_l1249_124989


namespace NUMINAMATH_GPT_stateA_selection_percentage_l1249_124944

theorem stateA_selection_percentage :
  ∀ (P : ℕ), (∀ (n : ℕ), n = 8000) → (7 * 8000 / 100 = P * 8000 / 100 + 80) → P = 6 := by
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_stateA_selection_percentage_l1249_124944


namespace NUMINAMATH_GPT_inverse_function_l1249_124965

noncomputable def f (x : ℝ) := 3 - 7 * x + x^2

noncomputable def g (x : ℝ) := (7 + Real.sqrt (37 + 4 * x)) / 2

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_inverse_function_l1249_124965


namespace NUMINAMATH_GPT_volume_decreases_by_sixteen_point_sixty_seven_percent_l1249_124924

variable {P V k : ℝ}

-- Stating the conditions
def inverse_proportionality (P V k : ℝ) : Prop :=
  P * V = k

def increased_pressure (P : ℝ) : ℝ :=
  1.2 * P

-- Theorem statement to prove the volume decrease percentage
theorem volume_decreases_by_sixteen_point_sixty_seven_percent (P V k : ℝ)
  (h1 : inverse_proportionality P V k)
  (h2 : P' = increased_pressure P) :
  V' = V / 1.2 ∧ (100 * (V - V') / V) = 16.67 :=
by
  sorry

end NUMINAMATH_GPT_volume_decreases_by_sixteen_point_sixty_seven_percent_l1249_124924


namespace NUMINAMATH_GPT_inequality_proof_l1249_124907

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1249_124907


namespace NUMINAMATH_GPT_seventh_diagram_shaded_triangles_l1249_124980

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * fact n

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- The main theorem stating the relationship between the number of shaded sub-triangles and the factorial/Fibonacci sequence
theorem seventh_diagram_shaded_triangles :
  ∃ k : ℕ, (k : ℚ) = (fib 7 : ℚ) / (fact 7 : ℚ) ∧ k = 13 := sorry

end NUMINAMATH_GPT_seventh_diagram_shaded_triangles_l1249_124980


namespace NUMINAMATH_GPT_wind_speed_l1249_124936

theorem wind_speed (w : ℝ) (h : 420 / (253 + w) = 350 / (253 - w)) : w = 23 :=
by
  sorry

end NUMINAMATH_GPT_wind_speed_l1249_124936


namespace NUMINAMATH_GPT_max_students_can_participate_l1249_124909

theorem max_students_can_participate (max_funds rent cost_per_student : ℕ) (h_max_funds : max_funds = 800) (h_rent : rent = 300) (h_cost_per_student : cost_per_student = 15) :
  ∃ x : ℕ, x ≤ (max_funds - rent) / cost_per_student ∧ x = 33 :=
by
  sorry

end NUMINAMATH_GPT_max_students_can_participate_l1249_124909


namespace NUMINAMATH_GPT_rob_final_value_in_euros_l1249_124970

noncomputable def initial_value_in_usd : ℝ := 
  (7 * 0.25) + (3 * 0.10) + (5 * 0.05) + (12 * 0.01) + (3 * 0.50) + (2 * 1.00)

noncomputable def value_after_losing_coins : ℝ := 
  (6 * 0.25) + (2 * 0.10) + (4 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_first_exchange : ℝ :=
  (6 * 0.25) + (4 * 0.10) + (1 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_second_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (11 * 0.01) + (1 * 0.50) + (1 * 1.00)

noncomputable def value_after_third_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def final_value_in_usd : ℝ := 
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def exchange_rate_usd_to_eur : ℝ := 0.85

noncomputable def final_value_in_eur : ℝ :=
  final_value_in_usd * exchange_rate_usd_to_eur

theorem rob_final_value_in_euros : final_value_in_eur = 2.9835 := by
  sorry

end NUMINAMATH_GPT_rob_final_value_in_euros_l1249_124970


namespace NUMINAMATH_GPT_geometric_series_sum_l1249_124963

theorem geometric_series_sum : 
  ∑' n : ℕ, (1 / 4) * (1 / 2)^n = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1249_124963


namespace NUMINAMATH_GPT_youngest_age_is_20_l1249_124986

-- Definitions of the ages
def siblings_ages (y : ℕ) : List ℕ := [y, y+2, y+7, y+11]

-- Condition of the problem: average age is 25
def average_age_25 (y : ℕ) : Prop := (siblings_ages y).sum = 100

-- The statement to be proven
theorem youngest_age_is_20 (y : ℕ) (h : average_age_25 y) : y = 20 :=
  sorry

end NUMINAMATH_GPT_youngest_age_is_20_l1249_124986


namespace NUMINAMATH_GPT_veronica_max_area_l1249_124930

noncomputable def max_area_garden : ℝ :=
  let l := 105
  let w := 420 - 2 * l
  l * w

theorem veronica_max_area : ∃ (A : ℝ), max_area_garden = 22050 :=
by
  use 22050
  show max_area_garden = 22050
  sorry

end NUMINAMATH_GPT_veronica_max_area_l1249_124930


namespace NUMINAMATH_GPT_initial_fund_is_890_l1249_124901

-- Given Conditions
def initial_fund (n : ℕ) : ℝ := 60 * n - 10
def bonus_given (n : ℕ) : ℝ := 50 * n
def remaining_fund (initial : ℝ) (bonus : ℝ) : ℝ := initial - bonus

-- Proof problem: Prove that the initial amount equals $890 under the given constraints
theorem initial_fund_is_890 :
  ∃ n : ℕ, 
    initial_fund n = 890 ∧ 
    initial_fund n - bonus_given n = 140 :=
by
  sorry

end NUMINAMATH_GPT_initial_fund_is_890_l1249_124901


namespace NUMINAMATH_GPT_harrison_croissant_expenditure_l1249_124990

-- Define the conditions
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def weeks_in_year : ℕ := 52

-- Define the total cost of croissants in a year
def total_cost (cost_regular cost_almond : ℝ) (weeks : ℕ) : ℝ :=
  (weeks * cost_regular) + (weeks * cost_almond)

-- State the proof problem
theorem harrison_croissant_expenditure :
  total_cost cost_regular_croissant cost_almond_croissant weeks_in_year = 468.00 :=
by
  sorry

end NUMINAMATH_GPT_harrison_croissant_expenditure_l1249_124990


namespace NUMINAMATH_GPT_mod_sum_example_l1249_124998

theorem mod_sum_example :
  (9^5 + 8^4 + 7^6) % 5 = 4 :=
by sorry

end NUMINAMATH_GPT_mod_sum_example_l1249_124998


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_is_purely_imaginary_l1249_124958

noncomputable def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ (b : ℝ), z = ⟨0, b⟩

theorem necessary_but_not_sufficient_condition_is_purely_imaginary (a b : ℝ) (h_imaginary : is_purely_imaginary (⟨a, b⟩)) : 
  (a = 0) ∧ (b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_is_purely_imaginary_l1249_124958


namespace NUMINAMATH_GPT_linear_regression_change_l1249_124994

theorem linear_regression_change (x : ℝ) :
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  y2 - y1 = -1.5 := by
  -- y1 = 2 - 1.5 * x
  -- y2 = 2 - 1.5 * x - 1.5
  -- Δ y = y2 - y1
  sorry

end NUMINAMATH_GPT_linear_regression_change_l1249_124994


namespace NUMINAMATH_GPT_valid_vector_parameterizations_of_line_l1249_124946

theorem valid_vector_parameterizations_of_line (t : ℝ) :
  (∃ t : ℝ, (∃ x y : ℝ, (x = 1 + t ∧ y = t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = -t ∧ y = -1 - t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = 2 + 0.5 * t ∧ y = 1 + 0.5 * t ∧ y = x - 1))) :=
by sorry

end NUMINAMATH_GPT_valid_vector_parameterizations_of_line_l1249_124946


namespace NUMINAMATH_GPT_dining_bill_split_l1249_124919

theorem dining_bill_split (original_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (total_bill_with_tip : ℝ) (amount_per_person : ℝ)
  (h1 : original_bill = 139.00)
  (h2 : num_people = 3)
  (h3 : tip_percent = 0.10)
  (h4 : total_bill_with_tip = original_bill + (tip_percent * original_bill))
  (h5 : amount_per_person = total_bill_with_tip / num_people) :
  amount_per_person = 50.97 :=
by 
  sorry

end NUMINAMATH_GPT_dining_bill_split_l1249_124919


namespace NUMINAMATH_GPT_smallest_x_for_M_squared_l1249_124975

theorem smallest_x_for_M_squared (M x : ℤ) (h1 : 540 = 2^2 * 3^3 * 5) (h2 : 540 * x = M^2) (h3 : x > 0) : x = 15 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_M_squared_l1249_124975


namespace NUMINAMATH_GPT_lcm_gcd_product_eq_product_12_15_l1249_124949

theorem lcm_gcd_product_eq_product_12_15 :
  lcm 12 15 * gcd 12 15 = 12 * 15 :=
sorry

end NUMINAMATH_GPT_lcm_gcd_product_eq_product_12_15_l1249_124949


namespace NUMINAMATH_GPT_find_num_carbon_atoms_l1249_124976

def num_carbon_atoms (nH nO mH mC mO mol_weight : ℕ) : ℕ :=
  (mol_weight - (nH * mH + nO * mO)) / mC

theorem find_num_carbon_atoms :
  num_carbon_atoms 2 3 1 12 16 62 = 1 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_find_num_carbon_atoms_l1249_124976


namespace NUMINAMATH_GPT_line_perpendicular_to_plane_l1249_124926

open Classical

-- Define the context of lines and planes.
variables {Line : Type} {Plane : Type}

-- Define the perpendicular and parallel relations.
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Declare the distinct lines and non-overlapping planes.
variable {m n : Line}
variable {α β : Plane}

-- State the theorem.
theorem line_perpendicular_to_plane (h1 : parallel m n) (h2 : perpendicular n β) : perpendicular m β :=
sorry

end NUMINAMATH_GPT_line_perpendicular_to_plane_l1249_124926


namespace NUMINAMATH_GPT_sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l1249_124950

theorem sum_of_squares_multiple_of_five :
  ( (-1)^2 + 0^2 + 1^2 + 2^2 + 3^2 ) % 5 = 0 :=
by
  sorry

theorem sum_of_consecutive_squares_multiple_of_five 
  (n : ℤ) :
  ((n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l1249_124950


namespace NUMINAMATH_GPT_preceding_integer_l1249_124961

def bin_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

theorem preceding_integer : bin_to_nat [true, true, false, false, false] - 1 = bin_to_nat [true, false, true, true, true] := by
  sorry

end NUMINAMATH_GPT_preceding_integer_l1249_124961


namespace NUMINAMATH_GPT_find_A_in_terms_of_B_and_C_l1249_124979

theorem find_A_in_terms_of_B_and_C 
  (A B C : ℝ) (hB : B ≠ 0) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = A * x - 2 * B^2)
  (hg : ∀ x, g x = B * x + C * x^2)
  (hfg : f (g 1) = 4 * B^2)
  : A = 6 * B * B / (B + C) :=
by
  sorry

end NUMINAMATH_GPT_find_A_in_terms_of_B_and_C_l1249_124979


namespace NUMINAMATH_GPT_polynomial_has_one_positive_real_solution_l1249_124995

-- Define the polynomial
def f (x : ℝ) : ℝ := x ^ 10 + 4 * x ^ 9 + 7 * x ^ 8 + 2023 * x ^ 7 - 2024 * x ^ 6

-- The proof problem statement
theorem polynomial_has_one_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ f x = 0 := by
  sorry

end NUMINAMATH_GPT_polynomial_has_one_positive_real_solution_l1249_124995


namespace NUMINAMATH_GPT_no_real_solutions_l1249_124934

theorem no_real_solutions :
  ∀ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) ≠ 1 / 8) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1249_124934


namespace NUMINAMATH_GPT_no_odd_integers_satisfy_equation_l1249_124954

theorem no_odd_integers_satisfy_equation :
  ¬ ∃ (x y z : ℤ), (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (z % 2 ≠ 0) ∧ 
  (x + y)^2 + (x + z)^2 = (y + z)^2 :=
by
  sorry

end NUMINAMATH_GPT_no_odd_integers_satisfy_equation_l1249_124954


namespace NUMINAMATH_GPT_base7_to_base10_321_is_162_l1249_124945

-- Define the conversion process from a base-7 number to base-10
def convert_base7_to_base10 (n: ℕ) : ℕ :=
  3 * 7^2 + 2 * 7^1 + 1 * 7^0

theorem base7_to_base10_321_is_162 :
  convert_base7_to_base10 321 = 162 :=
by
  sorry

end NUMINAMATH_GPT_base7_to_base10_321_is_162_l1249_124945


namespace NUMINAMATH_GPT_size_ratio_l1249_124929

variable {A B C : ℝ} -- Declaring that A, B, and C are real numbers (their sizes)
variable (h1 : A = 3 * B) -- A is three times the size of B
variable (h2 : B = (1 / 2) * C) -- B is half the size of C

theorem size_ratio (h1 : A = 3 * B) (h2 : B = (1 / 2) * C) : A / C = 1.5 :=
by
  sorry -- Proof goes here, to be completed

end NUMINAMATH_GPT_size_ratio_l1249_124929


namespace NUMINAMATH_GPT_quadratic_fixed_points_l1249_124915

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
  a * x^2 + (3 * a - 1) * x - (10 * a + 3)

theorem quadratic_fixed_points (a : ℝ) (h : a ≠ 0) :
  quadratic_function a 2 = -5 ∧ quadratic_function a (-5) = 2 :=
by sorry

end NUMINAMATH_GPT_quadratic_fixed_points_l1249_124915


namespace NUMINAMATH_GPT_difference_is_four_l1249_124993

def chickens_in_coop := 14
def chickens_in_run := 2 * chickens_in_coop
def chickens_free_ranging := 52
def difference := 2 * chickens_in_run - chickens_free_ranging

theorem difference_is_four : difference = 4 := by
  sorry

end NUMINAMATH_GPT_difference_is_four_l1249_124993


namespace NUMINAMATH_GPT_non_negative_integers_abs_less_than_3_l1249_124973

theorem non_negative_integers_abs_less_than_3 :
  { x : ℕ | x < 3 } = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_non_negative_integers_abs_less_than_3_l1249_124973


namespace NUMINAMATH_GPT_minimize_cost_l1249_124969

noncomputable def shipping_cost (x : ℝ) : ℝ := 5 * x
noncomputable def storage_cost (x : ℝ) : ℝ := 20 / x
noncomputable def total_cost (x : ℝ) : ℝ := shipping_cost x + storage_cost x

theorem minimize_cost : ∃ x : ℝ, x = 2 ∧ total_cost x = 20 :=
by
  use 2
  unfold total_cost
  unfold shipping_cost
  unfold storage_cost
  sorry

end NUMINAMATH_GPT_minimize_cost_l1249_124969


namespace NUMINAMATH_GPT_standard_eq_circle_C_equation_line_AB_l1249_124983

-- Define the center of circle C and the line l
def center_C : ℝ × ℝ := (2, 1)
def line_l (x y : ℝ) : Prop := x = 3

-- Define the standard equation of circle C
def eq_circle_C (x y : ℝ) : Prop :=
  (x - center_C.1)^2 + (y - center_C.2)^2 = 1

-- Equation of circle O
def eq_circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the condition that circle C intersects with circle O at points A and B
def intersects (x y : ℝ) : Prop :=
  eq_circle_C x y ∧ eq_circle_O x y

-- Define the equation of line AB in general form
def eq_line_AB (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Prove the standard equation of circle C is (x-2)^2 + (y-1)^2 = 1
theorem standard_eq_circle_C:
  eq_circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

-- Prove that the equation of line AB is 2x + y - 4 = 0, given the intersection points A and B
theorem equation_line_AB (x y : ℝ) (h : intersects x y) :
  eq_line_AB x y :=
sorry

end NUMINAMATH_GPT_standard_eq_circle_C_equation_line_AB_l1249_124983


namespace NUMINAMATH_GPT_percent_of_a_is_4b_l1249_124937

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end NUMINAMATH_GPT_percent_of_a_is_4b_l1249_124937


namespace NUMINAMATH_GPT_max_minute_hands_l1249_124960

theorem max_minute_hands (m n : ℕ) (h1 : m * n = 27) : m + n ≤ 28 :=
by sorry

end NUMINAMATH_GPT_max_minute_hands_l1249_124960


namespace NUMINAMATH_GPT_exist_functions_fg_neq_f1f1_g1g1_l1249_124922

-- Part (a)
theorem exist_functions_fg :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, (f ∘ g) x = (g ∘ f) x) ∧ 
    (∀ x, (f ∘ f) x = (g ∘ g) x) ∧ 
    (∀ x, f x ≠ g x) := 
sorry

-- Part (b)
theorem neq_f1f1_g1g1 
  (f1 g1 : ℝ → ℝ)
  (H_comm : ∀ x, (f1 ∘ g1) x = (g1 ∘ f1) x)
  (H_neq: ∀ x, f1 x ≠ g1 x) :
  ∀ x, (f1 ∘ f1) x ≠ (g1 ∘ g1) x :=
sorry

end NUMINAMATH_GPT_exist_functions_fg_neq_f1f1_g1g1_l1249_124922


namespace NUMINAMATH_GPT_g_at_2_eq_9_l1249_124905

def g (x : ℝ) : ℝ := x^2 + 3 * x - 1

theorem g_at_2_eq_9 : g 2 = 9 := by
  sorry

end NUMINAMATH_GPT_g_at_2_eq_9_l1249_124905


namespace NUMINAMATH_GPT_fractional_equation_no_solution_l1249_124987

theorem fractional_equation_no_solution (a : ℝ) :
  (¬ ∃ x, x ≠ 1 ∧ x ≠ 0 ∧ ((x - a) / (x - 1) - 3 / x = 1)) → (a = 1 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_fractional_equation_no_solution_l1249_124987


namespace NUMINAMATH_GPT_max_difference_second_largest_second_smallest_l1249_124908

theorem max_difference_second_largest_second_smallest :
  ∀ (a b c d e f g h : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h ∧
  a + b + c = 27 ∧
  a + b + c + d + e + f + g + h = 152 ∧
  f + g + h = 87 →
  g - b = 26 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_max_difference_second_largest_second_smallest_l1249_124908


namespace NUMINAMATH_GPT_union_complement_inter_l1249_124942

noncomputable def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | -1 ≤ x ∧ x < 5 }

def C_U_M : Set ℝ := U \ M
def M_inter_N : Set ℝ := { x | x ≥ 2 ∧ x < 5 }

theorem union_complement_inter (C_U_M M_inter_N : Set ℝ) :
  C_U_M ∪ M_inter_N = { x | x < 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_complement_inter_l1249_124942


namespace NUMINAMATH_GPT_vertex_of_parabola_l1249_124918

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- The theorem stating the vertex of the parabola
theorem vertex_of_parabola : ∃ h k : ℝ, (h, k) = (2, -5) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1249_124918


namespace NUMINAMATH_GPT_ratio_area_of_circle_to_triangle_l1249_124982

theorem ratio_area_of_circle_to_triangle
  (h r b : ℝ)
  (h_triangle : ∃ a, a = b + r ∧ a^2 + b^2 = h^2) :
  (∃ A s : ℝ, s = b + (r + h) / 2 ∧ A = r * s ∧ (∃ circle_area triangle_area : ℝ, circle_area = π * r^2 ∧ triangle_area = 2 * A ∧ circle_area / triangle_area = 2 * π * r / (2 * b + r + h))) :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_of_circle_to_triangle_l1249_124982


namespace NUMINAMATH_GPT_intersection_single_point_l1249_124991

def A (x y : ℝ) := x^2 + y^2 = 4
def B (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y - 4)^2 = r^2

theorem intersection_single_point (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, A p.1 p.2 ∧ B p.1 p.2 r) → r = 3 :=
by
  apply sorry -- Proof goes here

end NUMINAMATH_GPT_intersection_single_point_l1249_124991


namespace NUMINAMATH_GPT_yeast_population_at_130pm_l1249_124971

noncomputable def yeast_population (initial_population : ℕ) (time_increments : ℕ) (growth_factor : ℕ) : ℕ :=
  initial_population * growth_factor ^ time_increments

theorem yeast_population_at_130pm : yeast_population 30 3 3 = 810 :=
by
  sorry

end NUMINAMATH_GPT_yeast_population_at_130pm_l1249_124971


namespace NUMINAMATH_GPT_maximum_area_rectangle_l1249_124968

-- Define the conditions
def length (x : ℝ) := x
def width (x : ℝ) := 2 * x
def perimeter (x : ℝ) := 2 * (length x + width x)

-- The proof statement
theorem maximum_area_rectangle (h : perimeter x = 40) : 2 * (length x) * (width x) = 800 / 9 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_rectangle_l1249_124968
