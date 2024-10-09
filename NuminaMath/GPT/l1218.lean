import Mathlib

namespace mode_of_data_set_l1218_121827

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l1218_121827


namespace tom_annual_car_leasing_cost_l1218_121860

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end tom_annual_car_leasing_cost_l1218_121860


namespace alpha_cubic_expression_l1218_121869

theorem alpha_cubic_expression (α : ℝ) (hα : α^2 - 8 * α - 5 = 0) : α^3 - 7 * α^2 - 13 * α + 6 = 11 :=
sorry

end alpha_cubic_expression_l1218_121869


namespace purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l1218_121890

def z (m : ℝ) : Complex := Complex.mk (2 * m^2 - 3 * m - 2) (m^2 - 3 * m + 2)

theorem purely_imaginary_implies_m_eq_neg_half (m : ℝ) : 
  (z m).re = 0 ↔ m = -1 / 2 := sorry

theorem simplify_z_squared_over_z_add_5_plus_2i (z_zero : ℂ) :
  z 0 = ⟨-2, 2⟩ →
  (z 0)^2 / (z 0 + Complex.mk 5 2) = ⟨-32 / 25, -24 / 25⟩ := sorry

end purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l1218_121890


namespace smallest_sum_of_pairwise_distinct_squares_l1218_121862

theorem smallest_sum_of_pairwise_distinct_squares :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = z^2 ∧ c + a = y^2 ∧ a + b + c = 55 :=
sorry

end smallest_sum_of_pairwise_distinct_squares_l1218_121862


namespace second_term_arithmetic_sequence_l1218_121859

theorem second_term_arithmetic_sequence (a d : ℝ) (h : a + (a + 2 * d) = 10) : 
  a + d = 5 :=
by
  sorry

end second_term_arithmetic_sequence_l1218_121859


namespace arithmetic_sequence_term_2011_is_671st_l1218_121896

theorem arithmetic_sequence_term_2011_is_671st:
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → (3 * n - 2 = 2011) → n = 671 :=
by 
  intros a1 d n ha1 hd h_eq;
  sorry

end arithmetic_sequence_term_2011_is_671st_l1218_121896


namespace gray_region_area_l1218_121844

theorem gray_region_area (r : ℝ) : 
  let inner_circle_radius := r
  let outer_circle_radius := r + 3
  let inner_circle_area := Real.pi * (r ^ 2)
  let outer_circle_area := Real.pi * ((r + 3) ^ 2)
  let gray_region_area := outer_circle_area - inner_circle_area
  gray_region_area = 6 * Real.pi * r + 9 * Real.pi := 
by
  sorry

end gray_region_area_l1218_121844


namespace angle_in_third_quadrant_l1218_121854

/-- 
Given that the terminal side of angle α is in the third quadrant,
prove that the terminal side of α/3 cannot be in the second quadrant.
-/
theorem angle_in_third_quadrant (α : ℝ) (k : ℤ)
  (h : π + 2 * k * π < α ∧ α < 3 / 2 * π + 2 * k * π) :
  ¬ (π / 2 < α / 3 ∧ α / 3 < π) :=
sorry

end angle_in_third_quadrant_l1218_121854


namespace solve_inequality_1_find_range_of_a_l1218_121892

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end solve_inequality_1_find_range_of_a_l1218_121892


namespace interval_satisfaction_l1218_121812

theorem interval_satisfaction (a : ℝ) :
  (4 ≤ a / (3 * a - 6)) ∧ (a / (3 * a - 6) > 12) → a < 72 / 35 := 
by
  sorry

end interval_satisfaction_l1218_121812


namespace rolling_a_6_on_10th_is_random_event_l1218_121813

-- Definition of what it means for an event to be "random"
def is_random_event (event : ℕ → Prop) : Prop := 
  ∃ n : ℕ, event n

-- Condition: A die roll outcome for getting a 6
def die_roll_getting_6 (roll : ℕ) : Prop := 
  roll = 6

-- The main theorem to state the problem and the conclusion
theorem rolling_a_6_on_10th_is_random_event (event : ℕ → Prop) 
  (h : ∀ n, event n = die_roll_getting_6 n) : 
  is_random_event (event) := 
  sorry

end rolling_a_6_on_10th_is_random_event_l1218_121813


namespace minutkin_bedtime_l1218_121882

def time_minutkin_goes_to_bed 
    (morning_time : ℕ) 
    (morning_turns : ℕ) 
    (night_turns : ℕ) 
    (morning_hours : ℕ) 
    (morning_minutes : ℕ)
    (hours_per_turn : ℕ) 
    (minutes_per_turn : ℕ) : Nat := 
    ((morning_hours * 60 + morning_minutes) - (night_turns * hours_per_turn * 60 + night_turns * minutes_per_turn)) % 1440 

theorem minutkin_bedtime : 
    time_minutkin_goes_to_bed 9 9 11 8 30 1 12 = 1290 :=
    sorry

end minutkin_bedtime_l1218_121882


namespace max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l1218_121817

variable {m x x0 : ℝ}

def proposition_p (m : ℝ) : Prop := ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m
def proposition_q (m : ℝ) : Prop := ∃ x0 : ℝ, x0 ^ 2 - m * x0 + 1 = 0

theorem max_val_of_m_if_p_true (h : proposition_p m) : m ≤ Real.sqrt 2 := by
  sorry

theorem range_of_m_if_one_prop_true_one_false (hp : proposition_p m) (hq : ¬ proposition_q m) : (-2 < m ∧ m ≤ Real.sqrt 2) ∨ (2 ≤ m) := by
  sorry

theorem range_of_m_if_one_prop_false_one_true (hp : ¬ proposition_p m) (hq : proposition_q m) : (m ≥ 2) := by
  sorry

end max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l1218_121817


namespace triangle_inequality_problem_l1218_121826

-- Define the problem statement: Given the specified conditions, prove the interval length and sum
theorem triangle_inequality_problem :
  ∀ (A B C D : Type) (AB AC BC BD CD AD AO : ℝ),
  AB = 12 ∧ CD = 4 →
  (∃ x : ℝ, (4 < x ∧ x < 24) ∧ (AC = x ∧ m = 4 ∧ n = 24 ∧ m + n = 28)) :=
by
  intro A B C D AB AC BC BD CD AD AO h
  sorry

end triangle_inequality_problem_l1218_121826


namespace only_positive_integer_x_l1218_121880

theorem only_positive_integer_x (x : ℕ) (k : ℕ) (h1 : 2 * x + 1 = k^2) (h2 : x > 0) :
  ¬ (∃ y : ℕ, (y >= 2 * x + 2 ∧ y <= 3 * x + 2 ∧ ∃ m : ℕ, y = m^2)) → x = 4 := 
by sorry

end only_positive_integer_x_l1218_121880


namespace mike_total_work_time_l1218_121876

theorem mike_total_work_time :
  let wash_time := 10
  let oil_change_time := 15
  let tire_change_time := 30
  let paint_time := 45
  let engine_service_time := 60

  let num_wash := 9
  let num_oil_change := 6
  let num_tire_change := 2
  let num_paint := 4
  let num_engine_service := 3
  
  let total_minutes := 
        num_wash * wash_time +
        num_oil_change * oil_change_time +
        num_tire_change * tire_change_time +
        num_paint * paint_time +
        num_engine_service * engine_service_time

  let total_hours := total_minutes / 60

  total_hours = 10 :=
  by
    -- Definitions of times per task
    let wash_time := 10
    let oil_change_time := 15
    let tire_change_time := 30
    let paint_time := 45
    let engine_service_time := 60

    -- Definitions of number of tasks performed
    let num_wash := 9
    let num_oil_change := 6
    let num_tire_change := 2
    let num_paint := 4
    let num_engine_service := 3

    -- Calculate total minutes
    let total_minutes := 
      num_wash * wash_time +
      num_oil_change * oil_change_time +
      num_tire_change * tire_change_time +
      num_paint * paint_time +
      num_engine_service * engine_service_time
    
    -- Calculate total hours
    let total_hours := total_minutes / 60

    -- Required equality to prove
    have : total_hours = 10 := sorry
    exact this

end mike_total_work_time_l1218_121876


namespace quadratic_distinct_roots_l1218_121811

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_distinct_roots (k : ℝ) :
  (k ≠ 0) ∧ (1 > k) ↔ has_two_distinct_real_roots k (-6) 9 :=
by
  sorry

end quadratic_distinct_roots_l1218_121811


namespace certain_number_plus_two_l1218_121891

theorem certain_number_plus_two (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end certain_number_plus_two_l1218_121891


namespace compute_expression_l1218_121883

theorem compute_expression : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end compute_expression_l1218_121883


namespace number_of_triangles_l1218_121868

open Nat

-- Define the number of combinations
def comb : Nat → Nat → Nat
  | n, k => if k > n then 0 else n.choose k

-- The given conditions
def points_on_OA := 5
def points_on_OB := 6
def point_O := 1
def total_points := points_on_OA + points_on_OB + point_O -- should equal 12

-- Lean proof problem statement
theorem number_of_triangles : comb total_points 3 - comb points_on_OA 3 - comb points_on_OB 3 = 165 := by
  sorry

end number_of_triangles_l1218_121868


namespace arithmetic_sequence_common_difference_l1218_121802

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 1 = 13) (h4 : a_n 4 = 1) : 
  ∃ d : ℤ, d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l1218_121802


namespace blocks_to_get_home_l1218_121855

-- Definitions based on conditions provided
def blocks_to_park := 4
def blocks_to_school := 7
def trips_per_day := 3
def total_daily_blocks := 66

-- The proof statement for the number of blocks Ray walks to get back home
theorem blocks_to_get_home 
  (h1: blocks_to_park = 4)
  (h2: blocks_to_school = 7)
  (h3: trips_per_day = 3)
  (h4: total_daily_blocks = 66) : 
  (total_daily_blocks / trips_per_day - (blocks_to_park + blocks_to_school) = 11) :=
by
  sorry

end blocks_to_get_home_l1218_121855


namespace product_of_fraction_l1218_121832

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l1218_121832


namespace sqrt_inequality_l1218_121846

theorem sqrt_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (habc : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := 
sorry

end sqrt_inequality_l1218_121846


namespace S2016_value_l1218_121897

theorem S2016_value (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -2016)
  (h2 : ∀ n, S (n+1) = S n + a (n+1))
  (h3 : ∀ n, a (n+1) = a n + d)
  (h4 : (S 2015) / 2015 - (S 2012) / 2012 = 3) : S 2016 = -2016 := 
sorry

end S2016_value_l1218_121897


namespace lcm_of_5_6_8_9_l1218_121829

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l1218_121829


namespace total_people_wearing_hats_l1218_121805

variable (total_adults : ℕ) (total_children : ℕ)
variable (half_adults : ℕ) (women : ℕ) (men : ℕ)
variable (women_with_hats : ℕ) (men_with_hats : ℕ)
variable (children_with_hats : ℕ)
variable (total_with_hats : ℕ)

-- Given conditions
def conditions : Prop :=
  total_adults = 1800 ∧
  total_children = 200 ∧
  half_adults = total_adults / 2 ∧
  women = half_adults ∧
  men = half_adults ∧
  women_with_hats = (25 * women) / 100 ∧
  men_with_hats = (12 * men) / 100 ∧
  children_with_hats = (10 * total_children) / 100 ∧
  total_with_hats = women_with_hats + men_with_hats + children_with_hats

-- Proof goal
theorem total_people_wearing_hats : conditions total_adults total_children half_adults women men women_with_hats men_with_hats children_with_hats total_with_hats → total_with_hats = 353 :=
by
  intros h
  sorry

end total_people_wearing_hats_l1218_121805


namespace smallest_f_for_perfect_square_l1218_121865

theorem smallest_f_for_perfect_square (f : ℕ) (h₁: 3150 = 2 * 3 * 5^2 * 7) (h₂: ∃ m : ℕ, 3150 * f = m^2) :
  f = 14 :=
sorry

end smallest_f_for_perfect_square_l1218_121865


namespace min_sum_xy_l1218_121867

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y + x * y = 3) : x + y ≥ 2 :=
by
  sorry

end min_sum_xy_l1218_121867


namespace sam_total_cents_l1218_121888

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10
def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25
def nickels_to_cents (nickels : ℕ) : ℕ := nickels * 5
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

noncomputable def total_cents (initial_dimes dad_dimes mom_dimes grandma_dollars sister_quarters_initial : ℕ)
                             (initial_quarters dad_quarters mom_quarters grandma_transform sister_quarters_donation : ℕ)
                             (initial_nickels dad_nickels mom_nickels grandma_conversion sister_nickels_donation : ℕ) : ℕ :=
  dimes_to_cents initial_dimes +
  quarters_to_cents initial_quarters +
  nickels_to_cents initial_nickels +
  dimes_to_cents dad_dimes +
  quarters_to_cents dad_quarters -
  nickels_to_cents mom_nickels -
  dimes_to_cents mom_dimes +
  dollars_to_cents grandma_dollars +
  quarters_to_cents sister_quarters_donation +
  nickels_to_cents sister_nickels_donation

theorem sam_total_cents :
  total_cents 9 7 2 3 4 5 2 0 0 3 2 1 = 735 := 
  by exact sorry

end sam_total_cents_l1218_121888


namespace average_salary_of_managers_l1218_121800

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end average_salary_of_managers_l1218_121800


namespace max_s_value_l1218_121861

theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3)
  (h : ((r - 2) * 180 / r : ℚ) / ((s - 2) * 180 / s) = 60 / 59) :
  s = 117 :=
by
  sorry

end max_s_value_l1218_121861


namespace four_nabla_seven_l1218_121843

-- Define the operation ∇
def nabla (a b : ℤ) : ℚ :=
  (a + b) / (1 + a * b)

theorem four_nabla_seven :
  nabla 4 7 = 11 / 29 :=
by
  sorry

end four_nabla_seven_l1218_121843


namespace max_x_add_2y_l1218_121830

theorem max_x_add_2y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + 2 * y ≤ 4 :=
sorry

end max_x_add_2y_l1218_121830


namespace find_number_l1218_121801

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end find_number_l1218_121801


namespace squares_with_center_35_65_l1218_121849

theorem squares_with_center_35_65 : 
  (∃ (n : ℕ), n = 1190 ∧ ∀ (x y : ℕ), x ≠ y → (x, y) = (35, 65)) :=
sorry

end squares_with_center_35_65_l1218_121849


namespace expression_is_integer_if_k_eq_2_l1218_121877

def binom (n k : ℕ) := n.factorial / (k.factorial * (n-k).factorial)

theorem expression_is_integer_if_k_eq_2 
  (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : k = 2) : 
  ∃ (m : ℕ), m = (n - 3 * k + 2) * binom n k / (k + 2) := sorry

end expression_is_integer_if_k_eq_2_l1218_121877


namespace find_Q_div_P_l1218_121806

variable (P Q : ℚ)
variable (h_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
  P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))

theorem find_Q_div_P : Q / P = -6 / 13 := by
  sorry

end find_Q_div_P_l1218_121806


namespace annika_total_kilometers_east_l1218_121864

def annika_constant_rate : ℝ := 10 -- 10 minutes per kilometer
def distance_hiked_initially : ℝ := 2.5 -- 2.5 kilometers
def total_time_to_return : ℝ := 35 -- 35 minutes

theorem annika_total_kilometers_east :
  (total_time_to_return - (distance_hiked_initially * annika_constant_rate)) / annika_constant_rate + distance_hiked_initially = 3.5 := by
  sorry

end annika_total_kilometers_east_l1218_121864


namespace prove_fraction_identity_l1218_121852

theorem prove_fraction_identity 
  (x y z : ℝ)
  (h1 : (x * z) / (x + y) + (y * z) / (y + z) + (x * y) / (z + x) = -18)
  (h2 : (z * y) / (x + y) + (z * x) / (y + z) + (y * x) / (z + x) = 20) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 20.5 := 
by
  sorry

end prove_fraction_identity_l1218_121852


namespace solution_interval_l1218_121873

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x - x^(1 / 3)

theorem solution_interval (x₀ : ℝ) 
  (h_solution : (1 / 2)^x₀ = x₀^(1 / 3)) : x₀ ∈ Set.Ioo (1 / 3) (1 / 2) :=
by
  sorry

end solution_interval_l1218_121873


namespace part1_part2_case1_part2_case2_part2_case3_l1218_121841

namespace InequalityProof

variable {a x : ℝ}

def f (a x : ℝ) := a * x^2 + x - a

theorem part1 (h : a = 1) : (x > 1 ∨ x < -2) → f a x > 1 :=
by sorry

theorem part2_case1 (h1 : a < 0) (h2 : a < -1/2) : (- (a + 1) / a) < x ∧ x < 1 → f a x > 1 :=
by sorry

theorem part2_case2 (h1 : a < 0) (h2 : a = -1/2) : x ≠ 1 → f a x > 1 :=
by sorry

theorem part2_case3 (h1 : a < 0) (h2 : 0 > a) (h3 : a > -1/2) : 1 < x ∧ x < - (a + 1) / a → f a x > 1 :=
by sorry

end InequalityProof

end part1_part2_case1_part2_case2_part2_case3_l1218_121841


namespace Taran_original_number_is_12_l1218_121881

open Nat

theorem Taran_original_number_is_12 (x : ℕ)
  (h1 : (5 * x) + 5 - 5 = 73 ∨ (5 * x) + 5 - 6 = 73 ∨ (5 * x) + 6 - 5 = 73 ∨ (5 * x) + 6 - 6 = 73 ∨ 
       (6 * x) + 5 - 5 = 73 ∨ (6 * x) + 5 - 6 = 73 ∨ (6 * x) + 6 - 5 = 73 ∨ (6 * x) + 6 - 6 = 73) : x = 12 := by
  sorry

end Taran_original_number_is_12_l1218_121881


namespace find_number_1920_find_number_60_l1218_121815

theorem find_number_1920 : 320 * 6 = 1920 :=
by sorry

theorem find_number_60 : (1920 / 7 = 60) :=
by sorry

end find_number_1920_find_number_60_l1218_121815


namespace product_is_zero_l1218_121895

theorem product_is_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := 
by
  sorry

end product_is_zero_l1218_121895


namespace number_of_one_dollar_coins_l1218_121885

theorem number_of_one_dollar_coins (t : ℕ) :
  (∃ k : ℕ, 3 * k = t) → ∃ k : ℕ, k = t / 3 :=
by
  sorry

end number_of_one_dollar_coins_l1218_121885


namespace count_total_kids_in_lawrence_l1218_121810

namespace LawrenceCountyKids

/-- Number of kids who went to camp from Lawrence county -/
def kids_went_to_camp : ℕ := 610769

/-- Number of kids who stayed home -/
def kids_stayed_home : ℕ := 590796

/-- Total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := 1201565

/-- Proof statement -/
theorem count_total_kids_in_lawrence :
  kids_went_to_camp + kids_stayed_home = total_kids_in_county :=
sorry

end LawrenceCountyKids

end count_total_kids_in_lawrence_l1218_121810


namespace traveling_distance_l1218_121835

/-- Let D be the total distance from the dormitory to the city in kilometers.
Given the following conditions:
1. The student traveled 1/3 of the way by foot.
2. The student traveled 3/5 of the way by bus.
3. The remaining portion of the journey was covered by car, which equals 2 kilometers.
We need to prove that the total distance D is 30 kilometers. -/ 
theorem traveling_distance (D : ℕ) 
  (h1 : (1 / 3 : ℚ) * D + (3 / 5 : ℚ) * D + 2 = D) : D = 30 := 
sorry

end traveling_distance_l1218_121835


namespace inequality_div_l1218_121866

theorem inequality_div (m n : ℝ) (h : m > n) : (m / 5) > (n / 5) :=
sorry

end inequality_div_l1218_121866


namespace fixed_point_l1218_121851

theorem fixed_point (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 :=
by
  sorry

end fixed_point_l1218_121851


namespace max_min_diff_c_l1218_121837

theorem max_min_diff_c {a b c : ℝ} 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 15) : 
  (∃ c_max c_min, 
    (∀ a b c, a + b + c = 3 ∧ a^2 + b^2 + c^2 = 15 → c_min ≤ c ∧ c ≤ c_max) ∧ 
    c_max - c_min = 16 / 3) :=
sorry

end max_min_diff_c_l1218_121837


namespace greatest_fraction_lt_17_l1218_121833

theorem greatest_fraction_lt_17 :
  ∃ (x : ℚ), x = 15 / 4 ∧ x^2 < 17 ∧ ∀ y : ℚ, y < 4 → y^2 < 17 → y ≤ 15 / 4 := 
by
  use 15 / 4
  sorry

end greatest_fraction_lt_17_l1218_121833


namespace power_equality_l1218_121878

theorem power_equality : 
  ( (11 : ℝ) ^ (1 / 5) / (11 : ℝ) ^ (1 / 7) ) = (11 : ℝ) ^ (2 / 35) := 
by sorry

end power_equality_l1218_121878


namespace fraction_equation_solution_l1218_121831

theorem fraction_equation_solution (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := 
by
  sorry

end fraction_equation_solution_l1218_121831


namespace triangle_interior_angle_ge_60_l1218_121871

theorem triangle_interior_angle_ge_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < 60) (h3 : B < 60) (h4 : C < 60) : false := 
by
  sorry

end triangle_interior_angle_ge_60_l1218_121871


namespace max_min_sundays_in_month_l1218_121848

def week_days : ℕ := 7
def min_month_days : ℕ := 28
def months_days (d : ℕ) : Prop := d = 28 ∨ d = 30 ∨ d = 31

theorem max_min_sundays_in_month (d : ℕ) (h1 : months_days d) :
  4 ≤ (d / week_days) + ite (d % week_days > 0) 1 0 ∧ (d / week_days) + ite (d % week_days > 0) 1 0 ≤ 5 :=
by
  sorry

end max_min_sundays_in_month_l1218_121848


namespace smallest_n_good_sequence_2014_l1218_121834

-- Define the concept of a "good sequence"
def good_sequence (a : ℕ → ℝ) : Prop :=
  a 0 > 0 ∧
  ∀ i, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

-- Define the smallest n such that a good sequence reaches 2014 at a_n
theorem smallest_n_good_sequence_2014 :
  ∃ (n : ℕ), (∀ a, good_sequence a → a n = 2014) ∧
  ∀ (m : ℕ), m < n → ∀ a, good_sequence a → a m ≠ 2014 :=
sorry

end smallest_n_good_sequence_2014_l1218_121834


namespace perfect_square_461_l1218_121886

theorem perfect_square_461 (x : ℤ) (y : ℤ) (hx : 5 ∣ x) (hy : 5 ∣ y) 
  (h : x^2 + 461 = y^2) : x^2 = 52900 :=
  sorry

end perfect_square_461_l1218_121886


namespace janek_favorite_number_l1218_121808

theorem janek_favorite_number (S : Set ℕ) (n : ℕ) :
  S = {6, 8, 16, 22, 32} →
  n / 2 ∈ S →
  (n + 6) ∈ S →
  (n - 10) ∈ S →
  2 * n ∈ S →
  n = 16 := by
  sorry

end janek_favorite_number_l1218_121808


namespace upstream_speed_proof_l1218_121828

-- Definitions based on the conditions in the problem
def speed_in_still_water : ℝ := 25
def speed_downstream : ℝ := 35

-- The speed of the man rowing upstream
def speed_upstream : ℝ := speed_in_still_water - (speed_downstream - speed_in_still_water)

theorem upstream_speed_proof : speed_upstream = 15 := by
  -- Proof is omitted by using sorry
  sorry

end upstream_speed_proof_l1218_121828


namespace value_of_a_l1218_121870

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0.5 → 1 - a / 2^x > 0) → a = Real.sqrt 2 :=
by
  sorry

end value_of_a_l1218_121870


namespace oranges_taken_from_basket_l1218_121898

-- Define the original number of oranges and the number left after taking some out.
def original_oranges : ℕ := 8
def oranges_left : ℕ := 3

-- Prove that the number of oranges taken from the basket equals 5.
theorem oranges_taken_from_basket : original_oranges - oranges_left = 5 := by
  sorry

end oranges_taken_from_basket_l1218_121898


namespace lindas_daughters_and_granddaughters_no_daughters_l1218_121809

def number_of_people_with_no_daughters (total_daughters total_descendants daughters_with_5_daughters : ℕ) : ℕ :=
  total_descendants - (5 * daughters_with_5_daughters - total_daughters + daughters_with_5_daughters)

theorem lindas_daughters_and_granddaughters_no_daughters
  (total_daughters : ℕ)
  (total_descendants : ℕ)
  (daughters_with_5_daughters : ℕ)
  (H1 : total_daughters = 8)
  (H2 : total_descendants = 43)
  (H3 : 5 * daughters_with_5_daughters = 35)
  : number_of_people_with_no_daughters total_daughters total_descendants daughters_with_5_daughters = 36 :=
by
  -- Code to check the proof goes here.
  sorry

end lindas_daughters_and_granddaughters_no_daughters_l1218_121809


namespace range_of_m_l1218_121804

def set_A : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : Set ℝ := { x : ℝ | (2 * m - 1) ≤ x ∧ x ≤ (2 * m + 1) }

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ (-1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l1218_121804


namespace remainder_25197629_mod_4_l1218_121899

theorem remainder_25197629_mod_4 : 25197629 % 4 = 1 := by
  sorry

end remainder_25197629_mod_4_l1218_121899


namespace part1_solution_set_part2_range_a_l1218_121803

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l1218_121803


namespace largest_divisor_of_m_square_minus_n_square_l1218_121836

theorem largest_divisor_of_m_square_minus_n_square (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k : ℤ, k = 8 ∧ ∀ a b : ℤ, a % 2 = 1 → b % 2 = 1 → a > b → 8 ∣ (a^2 - b^2) := 
by
  sorry

end largest_divisor_of_m_square_minus_n_square_l1218_121836


namespace inequality_solution_l1218_121872

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) * (x + 1) / (x - 2)

theorem inequality_solution :
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | 2 < x ∧ x ≤ 8/3 } ∪ { x : ℝ | 4 ≤ x } :=
by sorry

end inequality_solution_l1218_121872


namespace person_walking_speed_on_escalator_l1218_121816

theorem person_walking_speed_on_escalator 
  (v : ℝ) 
  (escalator_speed : ℝ := 15) 
  (escalator_length : ℝ := 180) 
  (time_taken : ℝ := 10)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) : 
  v = 3 := 
by 
  -- The proof steps will be filled in if required
  sorry

end person_walking_speed_on_escalator_l1218_121816


namespace hyperbola_k_range_l1218_121838

theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k + 2) - y^2 / (5 - k) = 1)) → (-2 < k ∧ k < 5) :=
by
  sorry

end hyperbola_k_range_l1218_121838


namespace factor_polynomial_l1218_121814

theorem factor_polynomial :
  ∀ (x : ℤ), 9 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 5 * x^2 = (x^2 + 4) * (9 * x^2 + 22 * x + 342) :=
by
  intro x
  sorry

end factor_polynomial_l1218_121814


namespace sum_a1_to_a12_l1218_121874

variable {a : ℕ → ℕ}

axiom geom_seq (n : ℕ) : a n * a (n + 1) * a (n + 2) = 8
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2

theorem sum_a1_to_a12 : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_a1_to_a12_l1218_121874


namespace citizen_income_l1218_121820

theorem citizen_income (total_tax : ℝ) (income : ℝ) :
  total_tax = 15000 →
  (income ≤ 20000 → total_tax = income * 0.10) ∧
  (20000 < income ∧ income ≤ 50000 → total_tax = (20000 * 0.10) + ((income - 20000) * 0.15)) ∧
  (50000 < income ∧ income ≤ 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + ((income - 50000) * 0.20)) ∧
  (income > 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + (40000 * 0.20) + ((income - 90000) * 0.25)) →
  income = 92000 :=
by
  sorry

end citizen_income_l1218_121820


namespace exam_papers_count_l1218_121845

theorem exam_papers_count (F x : ℝ) :
  (∀ n : ℕ, n = 5) →    -- condition 1: equivalence of n to proportions count
  (6 * x + 7 * x + 8 * x + 9 * x + 10 * x = 40 * x) →    -- condition 2: sum of proportions
  (40 * x = 0.60 * n * F) →   -- condition 3: student obtained 60% of total marks
  (7 * x > 0.50 * F ∧ 8 * x > 0.50 * F ∧ 9 * x > 0.50 * F ∧ 10 * x > 0.50 * F ∧ 6 * x ≤ 0.50 * F) →  -- condition 4: more than 50% in 4 papers
  ∃ n : ℕ, n = 5 :=    -- prove: number of papers is 5
sorry

end exam_papers_count_l1218_121845


namespace int_solve_ineq_l1218_121884

theorem int_solve_ineq (x : ℤ) : (x + 3)^3 ≤ 8 ↔ x ≤ -1 :=
by sorry

end int_solve_ineq_l1218_121884


namespace James_beat_record_by_72_l1218_121842

-- Define the conditions as given in the problem
def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def conversions : ℕ := 6
def points_per_conversion : ℕ := 2
def old_record : ℕ := 300

-- Define the necessary calculations based on the conditions
def points_from_touchdowns_per_game : ℕ := touchdowns_per_game * points_per_touchdown
def points_from_touchdowns_in_season : ℕ := games_in_season * points_from_touchdowns_per_game
def points_from_conversions : ℕ := conversions * points_per_conversion
def total_points_in_season : ℕ := points_from_touchdowns_in_season + points_from_conversions
def points_above_old_record : ℕ := total_points_in_season - old_record

-- State the proof problem
theorem James_beat_record_by_72 : points_above_old_record = 72 :=
by
  sorry

end James_beat_record_by_72_l1218_121842


namespace find_positive_number_l1218_121889

theorem find_positive_number (x : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = (49 / 216) * (1 / x)) : x = 24.5 :=
by
  sorry

end find_positive_number_l1218_121889


namespace investment_percentage_l1218_121824

theorem investment_percentage (x : ℝ) :
  (4000 * (x / 100) + 3500 * 0.04 + 2500 * 0.064 = 500) ↔ (x = 5) :=
by
  sorry

end investment_percentage_l1218_121824


namespace power_of_two_last_digit_product_divisible_by_6_l1218_121856

theorem power_of_two_last_digit_product_divisible_by_6 (n : Nat) (h : 3 < n) :
  ∃ d m : Nat, (2^n = 10 * m + d) ∧ (m * d) % 6 = 0 :=
by
  sorry

end power_of_two_last_digit_product_divisible_by_6_l1218_121856


namespace tony_lottery_winning_l1218_121863

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end tony_lottery_winning_l1218_121863


namespace luncheon_tables_needed_l1218_121807

theorem luncheon_tables_needed (invited : ℕ) (no_show : ℕ) (people_per_table : ℕ) (people_attended : ℕ) (tables_needed : ℕ) :
  invited = 47 →
  no_show = 7 →
  people_per_table = 5 →
  people_attended = invited - no_show →
  tables_needed = people_attended / people_per_table →
  tables_needed = 8 := by {
  -- Proof here
  sorry
}

end luncheon_tables_needed_l1218_121807


namespace surface_area_of_box_l1218_121823

variable {l w h : ℝ}

def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * h + w * h + l * w)

theorem surface_area_of_box (l w h : ℝ) : surfaceArea l w h = 2 * (l * h + w * h + l * w) :=
by
  sorry

end surface_area_of_box_l1218_121823


namespace cost_per_first_30_kg_is_10_l1218_121857

-- Definitions of the constants based on the conditions
def cost_per_33_kg (p q : ℝ) : Prop := 30 * p + 3 * q = 360
def cost_per_36_kg (p q : ℝ) : Prop := 30 * p + 6 * q = 420
def cost_per_25_kg (p : ℝ) : Prop := 25 * p = 250

-- The statement we want to prove
theorem cost_per_first_30_kg_is_10 (p q : ℝ) 
  (h1 : cost_per_33_kg p q)
  (h2 : cost_per_36_kg p q)
  (h3 : cost_per_25_kg p) : 
  p = 10 :=
sorry

end cost_per_first_30_kg_is_10_l1218_121857


namespace girls_more_than_boys_l1218_121879

variables (B G : ℕ)
def ratio_condition : Prop := 3 * G = 4 * B
def total_students_condition : Prop := B + G = 49

theorem girls_more_than_boys
  (h1 : ratio_condition B G)
  (h2 : total_students_condition B G) :
  G = B + 7 :=
sorry

end girls_more_than_boys_l1218_121879


namespace part1_proof_part2_proof_part3_proof_part4_proof_l1218_121858

variable {A B C : Type}
variables {a b c : ℝ}  -- Sides of the triangle
variables {h_a h_b h_c r r_a r_b r_c : ℝ}  -- Altitudes, inradius, and exradii of \triangle ABC

-- Part 1: Proving the sum of altitudes related to sides and inradius
theorem part1_proof : h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 2: Proving the sum of reciprocals of altitudes related to the reciprocal of inradius and exradii
theorem part2_proof : (1 / h_a) + (1 / h_b) + (1 / h_c) = 1 / r ∧ 1 / r = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

-- Part 3: Combining results of parts 1 and 2 to prove product of sums
theorem part3_proof : (h_a + h_b + h_c) * ((1 / h_a) + (1 / h_b) + (1 / h_c)) = (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 4: Final geometric identity
theorem part4_proof : (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 := sorry

end part1_proof_part2_proof_part3_proof_part4_proof_l1218_121858


namespace arrow_hits_apple_l1218_121818

noncomputable def time_to_hit (L V0 : ℝ) (α β : ℝ) : ℝ :=
  (L / V0) * (Real.sin β / Real.sin (α + β))

theorem arrow_hits_apple (g : ℝ) (L V0 : ℝ) (α β : ℝ) (h : (L / V0) * (Real.sin β / Real.sin (α + β)) = 3 / 4) 
  : time_to_hit L V0 α β = 3 / 4 := 
  by
  sorry

end arrow_hits_apple_l1218_121818


namespace division_remainder_l1218_121847

theorem division_remainder : 
  ∀ (Dividend Divisor Quotient Remainder : ℕ), 
  Dividend = 760 → 
  Divisor = 36 → 
  Quotient = 21 → 
  Dividend = (Divisor * Quotient) + Remainder → 
  Remainder = 4 := 
by 
  intros Dividend Divisor Quotient Remainder h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : 760 = 36 * 21 + Remainder := h4
  linarith

end division_remainder_l1218_121847


namespace trajectory_equation_equation_of_line_l1218_121821

-- Define the parabola and the trajectory
def parabola (x y : ℝ) := y^2 = 16 * x
def trajectory (x y : ℝ) := y^2 = 4 * x

-- Define the properties of the point P and the line l
def is_midpoint (P A B : ℝ × ℝ) :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through_point (x y k : ℝ) := 
  k * x + y = 1

-- Proof problem (Ⅰ): trajectory of the midpoints of segments perpendicular to the x-axis from points on the parabola
theorem trajectory_equation : ∀ (M : ℝ × ℝ), 
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ is_midpoint M P (P.1, 0)) → 
  trajectory M.1 M.2 :=
sorry

-- Proof problem (Ⅱ): equation of line l
theorem equation_of_line : ∀ (A B P : ℝ × ℝ), 
  trajectory A.1 A.2 → trajectory B.1 B.2 → 
  P = (3,2) → is_midpoint P A B → 
  ∃ k, line_through_point (A.1 - B.1) (A.2 - B.2) k :=
sorry

end trajectory_equation_equation_of_line_l1218_121821


namespace minimize_distance_AP_BP_l1218_121819

theorem minimize_distance_AP_BP :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -1 ∧
    ∀ P' : ℝ × ℝ, P'.1 = 0 → 
      (dist (3, 2) P + dist (1, -2) P) ≤ (dist (3, 2) P' + dist (1, -2) P') := by
sorry

end minimize_distance_AP_BP_l1218_121819


namespace total_charge_for_trip_l1218_121822

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end total_charge_for_trip_l1218_121822


namespace selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l1218_121839

section ProofProblems

-- Definitions and constants
def num_males := 6
def num_females := 4
def total_athletes := 10
def num_selections := 5
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- 1. Number of selection methods for 3 males and 2 females
theorem selection_3m2f : binom 6 3 * binom 4 2 = 120 := by sorry

-- 2. Number of selection methods with at least one captain
theorem selection_at_least_one_captain :
  2 * binom 8 4 + binom 8 3 = 196 := by sorry

-- 3. Number of selection methods with at least one female athlete
theorem selection_at_least_one_female :
  binom 10 5 - binom 6 5 = 246 := by sorry

-- 4. Number of selection methods with both a captain and at least one female athlete
theorem selection_captain_and_female :
  binom 9 4 + binom 8 4 - binom 5 4 = 191 := by sorry

end ProofProblems

end selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l1218_121839


namespace arc_length_l1218_121887

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 10) (h_α : α = 2 * Real.pi / 3) : 
  r * α = 20 * Real.pi / 3 := 
by {
sorry
}

end arc_length_l1218_121887


namespace train_passes_jogger_in_36_seconds_l1218_121850

/-- A jogger runs at 9 km/h, 240m ahead of a train moving at 45 km/h.
The train is 120m long. Prove the train passes the jogger in 36 seconds. -/
theorem train_passes_jogger_in_36_seconds
  (distance_ahead : ℝ)
  (jogger_speed_km_hr train_speed_km_hr train_length_m : ℝ)
  (jogger_speed_m_s train_speed_m_s relative_speed_m_s distance_to_cover time_to_pass : ℝ)
  (h1 : distance_ahead = 240)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_speed_km_hr = 45)
  (h4 : train_length_m = 120)
  (h5 : jogger_speed_m_s = jogger_speed_km_hr * 1000 / 3600)
  (h6 : train_speed_m_s = train_speed_km_hr * 1000 / 3600)
  (h7 : relative_speed_m_s = train_speed_m_s - jogger_speed_m_s)
  (h8 : distance_to_cover = distance_ahead + train_length_m)
  (h9 : time_to_pass = distance_to_cover / relative_speed_m_s) :
  time_to_pass = 36 := 
sorry

end train_passes_jogger_in_36_seconds_l1218_121850


namespace reciprocal_inequality_of_negatives_l1218_121875

variable (a b : ℝ)

/-- Given that a < b < 0, prove that 1/a > 1/b. -/
theorem reciprocal_inequality_of_negatives (h1 : a < b) (h2 : b < 0) : (1/a) > (1/b) :=
sorry

end reciprocal_inequality_of_negatives_l1218_121875


namespace johns_uncommon_cards_l1218_121893

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards_l1218_121893


namespace sum_of_digits_base2_345_l1218_121840

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end sum_of_digits_base2_345_l1218_121840


namespace triangle_obtuse_l1218_121853

theorem triangle_obtuse
  (A B : ℝ) 
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (h : Real.cos A > Real.sin B) : 
  π / 2 < π - (A + B) ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l1218_121853


namespace repeating_decimal_to_fraction_l1218_121825

theorem repeating_decimal_to_fraction : 
∀ (x : ℝ), x = 4 + (0.0036 / (1 - 0.01)) → x = 144/33 :=
by
  intro x hx
  -- This is a placeholder where the conversion proof would go.
  sorry

end repeating_decimal_to_fraction_l1218_121825


namespace distance_between_trees_l1218_121894

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end distance_between_trees_l1218_121894
