import Mathlib

namespace NUMINAMATH_CALUDE_specialist_time_calculation_l2157_215758

theorem specialist_time_calculation (days_in_hospital : ℕ) (bed_charge_per_day : ℕ) 
  (specialist_charge_per_hour : ℕ) (ambulance_charge : ℕ) (total_bill : ℕ) : 
  days_in_hospital = 3 →
  bed_charge_per_day = 900 →
  specialist_charge_per_hour = 250 →
  ambulance_charge = 1800 →
  total_bill = 4625 →
  (total_bill - (days_in_hospital * bed_charge_per_day + ambulance_charge)) / 
    (2 * (specialist_charge_per_hour / 60)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_specialist_time_calculation_l2157_215758


namespace NUMINAMATH_CALUDE_expression_simplification_l2157_215784

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 7) :
  (2 / (x - 3) - 1 / (x + 3)) / ((x^2 + 9*x) / (x^2 - 9)) = Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2157_215784


namespace NUMINAMATH_CALUDE_original_decimal_l2157_215763

theorem original_decimal (x : ℝ) : (100 * x = x + 29.7) → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l2157_215763


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2157_215791

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first nine terms of a geometric series with first term 1/3 and common ratio 2/3 is 19171/19683 -/
theorem geometric_series_sum :
  geometricSum (1/3) (2/3) 9 = 19171/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2157_215791


namespace NUMINAMATH_CALUDE_self_employed_tax_calculation_l2157_215789

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * tax_rate

/-- Theorem: The tax amount for a self-employed citizen with a gross income of 350,000.00 rubles and a tax rate of 6% is 21,000.00 rubles --/
theorem self_employed_tax_calculation :
  let gross_income : ℝ := 350000.00
  let tax_rate : ℝ := 0.06
  calculate_tax_amount gross_income tax_rate = 21000.00 := by
  sorry

#eval calculate_tax_amount 350000.00 0.06

end NUMINAMATH_CALUDE_self_employed_tax_calculation_l2157_215789


namespace NUMINAMATH_CALUDE_junk_mail_calculation_l2157_215720

theorem junk_mail_calculation (blocks : ℕ) (houses_per_block : ℕ) (mail_per_house : ℕ)
  (h1 : blocks = 16)
  (h2 : houses_per_block = 17)
  (h3 : mail_per_house = 4) :
  blocks * houses_per_block * mail_per_house = 1088 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_calculation_l2157_215720


namespace NUMINAMATH_CALUDE_count_non_negative_rationals_l2157_215754

def rational_list : List ℚ := [-15, 5 + 1/3, -23/100, 0, 76/10, 2, -1/3, 314/100]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_non_negative_rationals_l2157_215754


namespace NUMINAMATH_CALUDE_orange_ratio_l2157_215713

theorem orange_ratio (good_oranges bad_oranges : ℕ) 
  (h1 : good_oranges = 24) 
  (h2 : bad_oranges = 8) : 
  (good_oranges : ℚ) / bad_oranges = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_ratio_l2157_215713


namespace NUMINAMATH_CALUDE_altitude_scientific_notation_l2157_215714

/-- The altitude of a medium-high orbit satellite in China's Beidou satellite navigation system -/
def altitude : ℝ := 21500000

/-- The scientific notation representation of the altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the altitude is equal to its scientific notation representation -/
theorem altitude_scientific_notation : altitude = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_altitude_scientific_notation_l2157_215714


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2157_215740

/-- A quadratic expression ax^2 + bx + c is a perfect square trinomial if there exists a real number k such that ax^2 + bx + c = (kx + r)^2 for some real r. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (k * x + r)^2

/-- If 4x^2 + mx + 9 is a perfect square trinomial, then m = 12 or m = -12. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquareTrinomial 4 m 9 → m = 12 ∨ m = -12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2157_215740


namespace NUMINAMATH_CALUDE_log_equation_holds_l2157_215770

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) + Real.log 7 / Real.log 10 = Real.log 7 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2157_215770


namespace NUMINAMATH_CALUDE_tangent_implies_a_equals_two_l2157_215753

noncomputable section

-- Define the line and curve equations
def line (x : ℝ) : ℝ := x + 1
def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    line x = curve a x ∧ 
    (deriv (curve a)) x = (deriv line) x

-- Theorem statement
theorem tangent_implies_a_equals_two :
  ∀ a : ℝ, is_tangent a → a = 2 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_implies_a_equals_two_l2157_215753


namespace NUMINAMATH_CALUDE_product_586645_9999_l2157_215767

theorem product_586645_9999 : 586645 * 9999 = 5865885355 := by
  sorry

end NUMINAMATH_CALUDE_product_586645_9999_l2157_215767


namespace NUMINAMATH_CALUDE_braden_final_amount_l2157_215706

/-- Calculates the final amount in Braden's money box after winning a bet -/
def final_amount (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_final_amount :
  let initial_amount : ℕ := 400
  let bet_multiplier : ℕ := 2
  final_amount initial_amount bet_multiplier = 1200 := by sorry

end NUMINAMATH_CALUDE_braden_final_amount_l2157_215706


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2157_215798

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x + 3)^2 - 15,
    prove that when 4px^2 + 4qx + 4r is written in the form m(x - h)^2 + k, then h = -3 -/
theorem quadratic_transformation (p q r : ℝ) 
  (h : ∀ x, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) :
  ∃ (m k : ℝ), ∀ x, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - (-3))^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2157_215798


namespace NUMINAMATH_CALUDE_trajectory_and_tangent_lines_l2157_215795

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the projection line
def projection_line (x : ℝ) : Prop := x = 3

-- Define the point P
def point_P (M N : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 = M.1 + N.1 - 0 ∧ P.2 = M.2 + N.2 - 0

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (1, 4)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem trajectory_and_tangent_lines :
  ∀ (M N P : ℝ × ℝ),
    ellipse M.1 M.2 →
    projection_line N.1 →
    point_P M N P →
    (∀ (x y : ℝ), P = (x, y) → trajectory_E x y) ∧
    (∃ (x y : ℝ), (x, y) = point_A ∧ 
      (tangent_line_1 x ∨ tangent_line_2 x y) ∧
      (∀ (t : ℝ), trajectory_E (x + t) (y + t) → t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_tangent_lines_l2157_215795


namespace NUMINAMATH_CALUDE_intersection_condition_l2157_215718

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x + 1 < 3}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : M ∩ N a = N a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2157_215718


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2157_215752

theorem max_sum_of_pairwise_sums (a b c d e : ℝ) 
  (h : (a + b) + (a + c) + (b + c) + (d + e) = 1096) :
  (a + d) + (a + e) + (b + d) + (b + e) + (c + d) + (c + e) ≤ 4384 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2157_215752


namespace NUMINAMATH_CALUDE_quadratic_sum_l2157_215775

theorem quadratic_sum (x y : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : x * y = -15) :
  4 * x^2 + 4 * y^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2157_215775


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_l2157_215719

theorem max_sum_with_lcm_gcd (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 140) 
  (h_gcd : Nat.gcd a b = 5) : 
  a + b ≤ 145 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_l2157_215719


namespace NUMINAMATH_CALUDE_evelyn_lost_bottle_caps_l2157_215704

/-- The number of bottle caps Evelyn lost -/
def bottle_caps_lost (initial : ℝ) (final : ℝ) : ℝ :=
  initial - final

/-- Proof that Evelyn lost 18.0 bottle caps -/
theorem evelyn_lost_bottle_caps :
  bottle_caps_lost 63.0 45 = 18.0 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_lost_bottle_caps_l2157_215704


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l2157_215797

-- Part 1: System of equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 13 ∧ 2 * x + 3 * y = -8 ∧ x = 11 ∧ y = -10 := by sorry

-- Part 2: System of inequalities
theorem solve_system_inequalities :
  ∀ y : ℝ, ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2 ∧ 2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l2157_215797


namespace NUMINAMATH_CALUDE_marks_initial_trees_l2157_215785

theorem marks_initial_trees (total_after_planting : ℕ) (trees_to_plant : ℕ) : 
  total_after_planting = 25 → trees_to_plant = 12 → total_after_planting - trees_to_plant = 13 := by
  sorry

end NUMINAMATH_CALUDE_marks_initial_trees_l2157_215785


namespace NUMINAMATH_CALUDE_bottle_cap_count_l2157_215724

/-- Represents the number of bottle caps in one ounce -/
def caps_per_ounce : ℕ := 7

/-- Represents the weight of the bottle cap collection in pounds -/
def collection_weight_pounds : ℕ := 18

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

/-- Calculates the total number of bottle caps in the collection -/
def total_caps : ℕ := collection_weight_pounds * ounces_per_pound * caps_per_ounce

theorem bottle_cap_count : total_caps = 2016 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l2157_215724


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2157_215732

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  (m + 2) * (m - 2) + 3 * m * (m + 2) = 0

/-- The condition m = 1/2 -/
def condition (m : ℝ) : Prop := m = 1/2

/-- The statement that m = 1/2 is sufficient but not necessary for perpendicularity -/
theorem sufficient_not_necessary :
  (∀ m : ℝ, condition m → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → condition m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2157_215732


namespace NUMINAMATH_CALUDE_johnny_distance_l2157_215792

/-- The distance between Q and Y in kilometers -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in kilometers per hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in kilometers per hour -/
def johnny_rate : ℝ := 4

/-- The time difference in hours between when Matthew and Johnny start walking -/
def time_difference : ℝ := 1

/-- The theorem stating that Johnny walked 24 km when they met -/
theorem johnny_distance : ℝ := by
  sorry

end NUMINAMATH_CALUDE_johnny_distance_l2157_215792


namespace NUMINAMATH_CALUDE_probability_of_sum_three_is_one_over_216_l2157_215799

def standard_die := Finset.range 6

def roll_sum (a b c : ℕ) : ℕ := a + b + c

def probability_of_sum_three : ℚ :=
  (Finset.filter (λ (abc : ℕ × ℕ × ℕ) => roll_sum abc.1 abc.2.1 abc.2.2 = 3) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ)

theorem probability_of_sum_three_is_one_over_216 :
  probability_of_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_three_is_one_over_216_l2157_215799


namespace NUMINAMATH_CALUDE_simplify_expression_l2157_215739

theorem simplify_expression (y : ℝ) : 
  3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2) = 0 * y^2 + 0 * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2157_215739


namespace NUMINAMATH_CALUDE_pear_arrangement_l2157_215772

theorem pear_arrangement (n : ℕ) (weights : Fin (2*n+2) → ℝ) :
  ∃ (perm : Fin (2*n+2) ≃ Fin (2*n+2)),
    ∀ i : Fin (2*n+2), |weights (perm i) - weights (perm (i+1))| ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_pear_arrangement_l2157_215772


namespace NUMINAMATH_CALUDE_freight_yard_washing_machines_l2157_215737

/-- Proves that the total number of washing machines removed is 30,000 given the conditions of the freight yard problem. -/
theorem freight_yard_washing_machines 
  (num_containers : ℕ) 
  (crates_per_container : ℕ) 
  (boxes_per_crate : ℕ) 
  (machines_per_box : ℕ) 
  (machines_removed_per_box : ℕ) 
  (h1 : num_containers = 50)
  (h2 : crates_per_container = 20)
  (h3 : boxes_per_crate = 10)
  (h4 : machines_per_box = 8)
  (h5 : machines_removed_per_box = 3) : 
  num_containers * crates_per_container * boxes_per_crate * machines_removed_per_box = 30000 := by
  sorry

#check freight_yard_washing_machines

end NUMINAMATH_CALUDE_freight_yard_washing_machines_l2157_215737


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2157_215736

/-- The time it takes for a pipe to fill a tank without a leak, given:
    - With the leak, it takes 10 hours to fill the tank.
    - The leak can empty the full tank in 10 hours. -/
theorem pipe_fill_time (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) 
    (h1 : fill_time_with_leak = 10)
    (h2 : leak_empty_time = 10) : 
  ∃ T : ℝ, T = 5 ∧ (1 / T - 1 / leak_empty_time = 1 / fill_time_with_leak) := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2157_215736


namespace NUMINAMATH_CALUDE_consecutive_missing_factors_l2157_215794

theorem consecutive_missing_factors (n : ℕ) (h1 : n > 30) : 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 → (k ≠ 16 ∧ k ≠ 17 → n % k = 0)) →
  (∃ (m : ℕ), m ≥ 1 ∧ m < 30 ∧ n % m ≠ 0 ∧ n % (m + 1) ≠ 0) →
  (∀ (j : ℕ), j ≥ 1 ∧ j < 30 ∧ n % j ≠ 0 ∧ n % (j + 1) ≠ 0 → j = 16) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_missing_factors_l2157_215794


namespace NUMINAMATH_CALUDE_garden_fence_area_l2157_215773

/-- Given an L-shaped fence and two straight fence sections of 13m and 14m,
    prove that it's possible to create a rectangular area of at least 200 m². -/
theorem garden_fence_area (length : ℝ) (width : ℝ) : 
  length = 13 → width = 17 → length * width ≥ 200 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_area_l2157_215773


namespace NUMINAMATH_CALUDE_system_solution_l2157_215757

def system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ,
  system_equations x₁ x₂ x₃ x₄ x₅ →
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
   (x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3) ∨
   (x₁ = -1/3 ∧ x₂ = -1/3 ∧ x₃ = -1/3 ∧ x₄ = -1/3 ∧ x₅ = -1/3)) :=
by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2157_215757


namespace NUMINAMATH_CALUDE_profit_margin_in_terms_of_retail_price_l2157_215710

/-- Given a profit margin P, production cost C, retail price P_R, and constants k and c,
    prove that P can be expressed in terms of P_R. -/
theorem profit_margin_in_terms_of_retail_price
  (P C P_R k c : ℝ) (hP : P = k * C) (hP_R : P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
sorry

end NUMINAMATH_CALUDE_profit_margin_in_terms_of_retail_price_l2157_215710


namespace NUMINAMATH_CALUDE_jim_driven_distance_l2157_215726

theorem jim_driven_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 →
  remaining = 432 →
  driven = total_journey - remaining →
  driven = 768 := by
sorry

end NUMINAMATH_CALUDE_jim_driven_distance_l2157_215726


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l2157_215712

/-- Proves that the percentage markup on a video recorder's wholesale cost is 20%,
    given the wholesale cost, employee discount, and final price paid by the employee. -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount_percent : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount_percent = 10)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discounted_price := retail_price * (1 - employee_discount_percent / 100)
  markup_percentage = 20 :=
sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l2157_215712


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l2157_215777

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ Odd n := by sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l2157_215777


namespace NUMINAMATH_CALUDE_interest_difference_l2157_215786

theorem interest_difference (principal rate time : ℝ) : 
  principal = 300 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 204 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2157_215786


namespace NUMINAMATH_CALUDE_girls_on_same_team_probability_l2157_215793

/-- The probability of all three girls being on the same team when five boys and three girls
    are randomly divided into two four-person teams is 1/7. -/
theorem girls_on_same_team_probability :
  let total_children : ℕ := 8
  let num_boys : ℕ := 5
  let num_girls : ℕ := 3
  let team_size : ℕ := 4
  let total_ways : ℕ := (Nat.choose total_children team_size) / 2
  let favorable_ways : ℕ := Nat.choose num_boys 1
  ↑favorable_ways / ↑total_ways = 1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_girls_on_same_team_probability_l2157_215793


namespace NUMINAMATH_CALUDE_chess_game_probabilities_l2157_215782

-- Define the probabilities
def prob_draw : ℚ := 1/2
def prob_B_win : ℚ := 1/3

-- Define the statements to be proven
def prob_A_win : ℚ := 1 - prob_draw - prob_B_win
def prob_A_not_lose : ℚ := prob_draw + prob_A_win
def prob_B_lose : ℚ := prob_A_win
def prob_B_not_lose : ℚ := prob_draw + prob_B_win

-- Theorem to prove the statements
theorem chess_game_probabilities :
  (prob_A_win = 1/6) ∧
  (prob_A_not_lose = 2/3) ∧
  (prob_B_lose = 1/6) ∧
  (prob_B_not_lose = 5/6) :=
by sorry

end NUMINAMATH_CALUDE_chess_game_probabilities_l2157_215782


namespace NUMINAMATH_CALUDE_candy_duration_l2157_215741

theorem candy_duration (neighbors_candy : ℝ) (sister_candy : ℝ) (daily_consumption : ℝ) :
  neighbors_candy = 11.0 →
  sister_candy = 5.0 →
  daily_consumption = 8.0 →
  (neighbors_candy + sister_candy) / daily_consumption = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_candy_duration_l2157_215741


namespace NUMINAMATH_CALUDE_circle_radius_l2157_215751

/-- Given a circle with center (0,k) where k > 5, which is tangent to the lines y=2x, y=-2x, and y=5,
    the radius of the circle is (k-5)/√5. -/
theorem circle_radius (k : ℝ) (h : k > 5) : ∃ r : ℝ,
  r > 0 ∧
  r = (k - 5) / Real.sqrt 5 ∧
  (∀ x y : ℝ, (x = 0 ∧ y = k) → (x^2 + (y - k)^2 = r^2)) ∧
  (∃ x y : ℝ, y = 2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x y : ℝ, y = -2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x : ℝ, x^2 + (5 - k)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2157_215751


namespace NUMINAMATH_CALUDE_abes_age_l2157_215742

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 35 → present_age = 21 :=
by sorry

end NUMINAMATH_CALUDE_abes_age_l2157_215742


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2157_215729

theorem geometric_sequence_seventh_term
  (a : ℝ) (r : ℝ)
  (positive_sequence : ∀ n : ℕ, a * r ^ (n - 1) > 0)
  (fourth_term : a * r^3 = 16)
  (tenth_term : a * r^9 = 2) :
  a * r^6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2157_215729


namespace NUMINAMATH_CALUDE_total_lost_or_given_equals_sum_l2157_215769

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  given_to_friends : ℕ
  lost : ℕ
  total_lost_or_given : ℕ

/-- Theorem stating that the total number of crayons lost or given away
    is equal to the sum of crayons given to friends and crayons lost --/
theorem total_lost_or_given_equals_sum (c : CrayonCounts)
  (h1 : c.given_to_friends = 52)
  (h2 : c.lost = 535)
  (h3 : c.total_lost_or_given = 587) :
  c.total_lost_or_given = c.given_to_friends + c.lost := by
  sorry

#check total_lost_or_given_equals_sum

end NUMINAMATH_CALUDE_total_lost_or_given_equals_sum_l2157_215769


namespace NUMINAMATH_CALUDE_percent_equation_solution_l2157_215755

theorem percent_equation_solution :
  ∃ x : ℝ, (0.75 / 100) * x = 0.06 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_percent_equation_solution_l2157_215755


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2157_215778

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_sum : i + i^3 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2157_215778


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2157_215733

/-- 
Given a boat traveling downstream in a stream, this theorem proves that 
the speed of the boat in still water is 5 km/hr, based on the given conditions.
-/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 100)
  (h3 : downstream_time = 10)
  (h4 : downstream_distance = (boat_speed + stream_speed) * downstream_time) :
  boat_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2157_215733


namespace NUMINAMATH_CALUDE_max_value_of_a_l2157_215760

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x + 1) / x

def g (x : ℝ) : ℝ := Real.exp x - Real.log x + 2*x^2 + 1

theorem max_value_of_a (h : ∀ x > 0, x * f x a ≤ g x) :
  a ≤ Real.exp 1 + 1 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_a_l2157_215760


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2157_215743

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f ≥ 441/10 ∧
  ∃ a' b' c' d' e' f' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' = 441/10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2157_215743


namespace NUMINAMATH_CALUDE_cubic_equation_value_l2157_215707

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2005 = -2004 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l2157_215707


namespace NUMINAMATH_CALUDE_rectangle_area_l2157_215788

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (v1 v2 v3 v4 : ℝ × ℝ) : 
  v1 = (-8, 1) → v2 = (1, 1) → v3 = (1, -7) → v4 = (-8, -7) →
  let width := |v2.1 - v1.1|
  let height := |v2.2 - v3.2|
  width * height = 72 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2157_215788


namespace NUMINAMATH_CALUDE_negation_of_implication_l2157_215716

theorem negation_of_implication :
  ¬(x = 1 → x^2 = 1) ↔ (x = 1 → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2157_215716


namespace NUMINAMATH_CALUDE_spheres_radius_in_cone_l2157_215745

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three spheres in a cone --/
structure SpheresInCone where
  cone : Cone
  sphere : Sphere
  spheresTangent : Bool
  spheresTangentToBase : Bool
  spheresNotTangentToSides : Bool

/-- The theorem statement --/
theorem spheres_radius_in_cone (config : SpheresInCone) : 
  config.cone.baseRadius = 6 ∧ 
  config.cone.height = 15 ∧ 
  config.spheresTangent ∧ 
  config.spheresTangentToBase ∧ 
  config.spheresNotTangentToSides →
  config.sphere.radius = 27 - 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_spheres_radius_in_cone_l2157_215745


namespace NUMINAMATH_CALUDE_cost_increase_when_b_doubled_l2157_215749

theorem cost_increase_when_b_doubled (t : ℝ) (b : ℝ) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  new_cost = 16 * original_cost :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_when_b_doubled_l2157_215749


namespace NUMINAMATH_CALUDE_one_ball_in_last_box_l2157_215702

/-- The number of boxes and balls -/
def n : ℕ := 100

/-- The probability of a ball landing in a specific box -/
def p : ℚ := 1 / n

/-- The probability of exactly one ball landing in the last box -/
def prob_one_in_last : ℚ := ((n - 1 : ℚ) / n) ^ (n - 1)

/-- Theorem stating the probability of exactly one ball in the last box -/
theorem one_ball_in_last_box : 
  prob_one_in_last = ((n - 1 : ℚ) / n) ^ (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_one_ball_in_last_box_l2157_215702


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2157_215721

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔ 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2157_215721


namespace NUMINAMATH_CALUDE_min_amount_spent_l2157_215715

/-- Represents the price of a volleyball in yuan -/
def volleyball_price : ℝ := 80

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 100

/-- Represents the total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- Represents the minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 25

/-- Theorem stating the minimum amount spent on purchasing the balls -/
theorem min_amount_spent :
  let x := min_soccer_balls
  let y := total_balls - x
  x * soccer_ball_price + y * volleyball_price = 4500 ∧
  x ≥ y ∧
  500 / soccer_ball_price = 400 / volleyball_price ∧
  soccer_ball_price = volleyball_price + 20 := by
  sorry


end NUMINAMATH_CALUDE_min_amount_spent_l2157_215715


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2157_215783

theorem ratio_x_to_y (x y : ℚ) (h : (10 * x - 3 * y) / (13 * x - 2 * y) = 3 / 5) :
  x / y = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2157_215783


namespace NUMINAMATH_CALUDE_solution_set_f_geq_6_range_of_a_for_nonempty_solution_l2157_215756

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

-- Theorem for the first part of the problem
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_nonempty_solution :
  ∀ a : ℝ, (∃ x : ℝ, f x < a + x) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_6_range_of_a_for_nonempty_solution_l2157_215756


namespace NUMINAMATH_CALUDE_xiaojun_father_age_relation_l2157_215734

/-- 
Given:
- Xiaojun is currently 5 years old
- Xiaojun's father is currently 31 years old

Prove that after 8 years, Xiaojun's father's age will be 3 times Xiaojun's age.
-/
theorem xiaojun_father_age_relation (xiaojun_age : ℕ) (father_age : ℕ) :
  xiaojun_age = 5 →
  father_age = 31 →
  ∃ (years : ℕ), years = 8 ∧ (father_age + years) = 3 * (xiaojun_age + years) :=
by sorry

end NUMINAMATH_CALUDE_xiaojun_father_age_relation_l2157_215734


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l2157_215779

/-- The number of convex quadrilaterals formed by 15 points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (h : n = 15) : 
  Nat.choose n 4 = 1365 :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l2157_215779


namespace NUMINAMATH_CALUDE_range_of_a_l2157_215701

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2157_215701


namespace NUMINAMATH_CALUDE_sales_amount_is_194_l2157_215759

/-- Represents the sales data for a stationery store --/
structure SalesData where
  eraser_price : ℝ
  regular_price : ℝ
  short_price : ℝ
  eraser_sold : ℕ
  regular_sold : ℕ
  short_sold : ℕ

/-- Calculates the total sales amount --/
def total_sales (data : SalesData) : ℝ :=
  data.eraser_price * data.eraser_sold +
  data.regular_price * data.regular_sold +
  data.short_price * data.short_sold

/-- Theorem stating that the total sales amount is $194 --/
theorem sales_amount_is_194 (data : SalesData) 
  (h1 : data.eraser_price = 0.8)
  (h2 : data.regular_price = 0.5)
  (h3 : data.short_price = 0.4)
  (h4 : data.eraser_sold = 200)
  (h5 : data.regular_sold = 40)
  (h6 : data.short_sold = 35) :
  total_sales data = 194 := by
  sorry

end NUMINAMATH_CALUDE_sales_amount_is_194_l2157_215759


namespace NUMINAMATH_CALUDE_identical_differences_exist_l2157_215796

theorem identical_differences_exist (a : Fin 20 → ℕ) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bounded : ∀ i, a i ≤ 70) : 
  ∃ (i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 20), 
    i₁ < j₁ ∧ i₂ < j₂ ∧ i₃ < j₃ ∧ i₄ < j₄ ∧ 
    (i₁ ≠ i₂ ∨ j₁ ≠ j₂) ∧ (i₁ ≠ i₃ ∨ j₁ ≠ j₃) ∧ (i₁ ≠ i₄ ∨ j₁ ≠ j₄) ∧
    (i₂ ≠ i₃ ∨ j₂ ≠ j₃) ∧ (i₂ ≠ i₄ ∨ j₂ ≠ j₄) ∧ (i₃ ≠ i₄ ∨ j₃ ≠ j₄) ∧
    a j₁ - a i₁ = a j₂ - a i₂ ∧ 
    a j₁ - a i₁ = a j₃ - a i₃ ∧ 
    a j₁ - a i₁ = a j₄ - a i₄ :=
by sorry

end NUMINAMATH_CALUDE_identical_differences_exist_l2157_215796


namespace NUMINAMATH_CALUDE_different_color_probability_l2157_215700

/-- The probability of drawing two chips of different colors from a bag containing 
    7 red chips and 4 green chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips : ℕ := 7 + 4
  let red_chips : ℕ := 7
  let green_chips : ℕ := 4
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  prob_red * prob_green + prob_green * prob_red = 56 / 121 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l2157_215700


namespace NUMINAMATH_CALUDE_semicircles_in_triangle_l2157_215735

/-- Represents a semicircle in the triangle -/
structure Semicircle where
  radius : ℝ
  touches_triangle : Bool
  touches_other_semicircles : Bool
  diameter_on_triangle_side : Bool

/-- Represents the equilateral triangle with semicircles -/
structure TriangleWithSemicircles where
  side_length : ℝ
  semicircles : List Semicircle
  is_equilateral : Bool

/-- The main theorem statement -/
theorem semicircles_in_triangle (t : TriangleWithSemicircles) :
  t.semicircles.length = 3 ∧
  t.semicircles.all (λ s => s.radius = 1 ∧ s.touches_triangle ∧ s.touches_other_semicircles ∧ s.diameter_on_triangle_side) ∧
  t.is_equilateral
  →
  t.side_length = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_semicircles_in_triangle_l2157_215735


namespace NUMINAMATH_CALUDE_harry_milk_bottles_l2157_215774

theorem harry_milk_bottles (initial_bottles : ℕ) (jason_bought : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 35)
  (h2 : jason_bought = 5)
  (h3 : remaining_bottles = 24) :
  initial_bottles - jason_bought - remaining_bottles = 6 := by
  sorry

end NUMINAMATH_CALUDE_harry_milk_bottles_l2157_215774


namespace NUMINAMATH_CALUDE_tan_equality_225_l2157_215762

theorem tan_equality_225 (m : ℤ) :
  -180 < m ∧ m < 180 →
  (Real.tan (m * π / 180) = Real.tan (225 * π / 180) ↔ m = 45 ∨ m = -135) := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_225_l2157_215762


namespace NUMINAMATH_CALUDE_candy_division_ways_l2157_215711

def divide_candies (total : ℕ) (min_per_person : ℕ) : ℕ :=
  total - 2 * min_per_person + 1

theorem candy_division_ways :
  divide_candies 8 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_ways_l2157_215711


namespace NUMINAMATH_CALUDE_roots_of_unity_quadratic_equation_l2157_215728

theorem roots_of_unity_quadratic_equation :
  ∃! (S : Finset ℂ),
    (∀ z ∈ S, (Complex.abs z = 1) ∧
      (∃ a : ℤ, z ^ 2 + a * z + 1 = 0 ∧
        -2 ≤ a ∧ a ≤ 2 ∧
        ∃ k : ℤ, a = k * Real.cos (k * π / 6))) ∧
    Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_unity_quadratic_equation_l2157_215728


namespace NUMINAMATH_CALUDE_min_value_expression_l2157_215723

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2157_215723


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_range_l2157_215764

theorem sqrt_x_minus_5_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_range_l2157_215764


namespace NUMINAMATH_CALUDE_seventh_root_ratio_l2157_215725

theorem seventh_root_ratio (x : ℝ) (hx : x > 0) :
  (x ^ (1/2)) / (x ^ (1/4)) = x ^ (1/4) :=
sorry

end NUMINAMATH_CALUDE_seventh_root_ratio_l2157_215725


namespace NUMINAMATH_CALUDE_exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l2157_215765

-- Statement 1
theorem exists_real_cube_less_than_one : ∃ x : ℝ, x^3 < 1 := by sorry

-- Statement 2
theorem no_rational_square_root_of_two : ¬ ∃ x : ℚ, x^2 = 2 := by sorry

-- Statement 3
theorem not_all_natural_cube_greater_than_square : 
  ¬ ∀ x : ℕ, x^3 > x^2 := by sorry

-- Statement 4
theorem all_real_square_plus_one_positive : 
  ∀ x : ℝ, x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l2157_215765


namespace NUMINAMATH_CALUDE_exist_similar_numbers_l2157_215790

/-- A function that generates a number by repeating a given 3-digit number n times -/
def repeatDigits (d : Nat) (n : Nat) : Nat :=
  (d * (Nat.pow 10 (3 * n) - 1)) / 999

/-- Theorem stating the existence of three similar 1995-digit numbers with the required property -/
theorem exist_similar_numbers : ∃ (A B C : Nat),
  (A = repeatDigits 459 665) ∧
  (B = repeatDigits 495 665) ∧
  (C = repeatDigits 954 665) ∧
  (A + B = C) ∧
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_exist_similar_numbers_l2157_215790


namespace NUMINAMATH_CALUDE_botanist_flower_distribution_l2157_215771

theorem botanist_flower_distribution (total_flowers : ℕ) (num_bouquets : ℕ) (additional_flowers : ℕ) : 
  total_flowers = 601 →
  num_bouquets = 8 →
  additional_flowers = 7 →
  (total_flowers + additional_flowers) % num_bouquets = 0 ∧
  (total_flowers + additional_flowers - 1) % num_bouquets ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_botanist_flower_distribution_l2157_215771


namespace NUMINAMATH_CALUDE_pie_not_crust_percentage_l2157_215750

/-- Given a pie weighing 200 grams with a crust of 50 grams,
    prove that 75% of the pie is not crust. -/
theorem pie_not_crust_percentage :
  let total_weight : ℝ := 200
  let crust_weight : ℝ := 50
  let non_crust_weight : ℝ := total_weight - crust_weight
  let non_crust_percentage : ℝ := (non_crust_weight / total_weight) * 100
  non_crust_percentage = 75 := by
  sorry


end NUMINAMATH_CALUDE_pie_not_crust_percentage_l2157_215750


namespace NUMINAMATH_CALUDE_club_size_l2157_215709

/-- The number of committees in the club -/
def num_committees : ℕ := 5

/-- A member of the club -/
structure Member where
  committees : Finset (Fin num_committees)
  mem_two_committees : committees.card = 2

/-- The club -/
structure Club where
  members : Finset Member
  unique_pair_member : ∀ (c1 c2 : Fin num_committees), c1 ≠ c2 → 
    (members.filter (λ m => c1 ∈ m.committees ∧ c2 ∈ m.committees)).card = 1

theorem club_size (c : Club) : c.members.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l2157_215709


namespace NUMINAMATH_CALUDE_inequality_proof_l2157_215746

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 + 8*b*c))/a + (Real.sqrt (b^2 + 8*a*c))/b + (Real.sqrt (c^2 + 8*a*b))/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2157_215746


namespace NUMINAMATH_CALUDE_daycare_peas_preference_l2157_215731

theorem daycare_peas_preference (total : ℕ) (peas carrots corn : ℕ) : 
  total > 0 ∧
  carrots = 9 ∧
  corn = 5 ∧
  corn = (25 : ℕ) * total / 100 ∧
  total = peas + carrots + corn →
  peas = 6 := by
  sorry

end NUMINAMATH_CALUDE_daycare_peas_preference_l2157_215731


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2157_215787

theorem right_triangle_perimeter (base height : ℝ) (h_base : base = 4) (h_height : height = 3) :
  let hypotenuse := Real.sqrt (base^2 + height^2)
  base + height + hypotenuse = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2157_215787


namespace NUMINAMATH_CALUDE_problem_solution_l2157_215744

theorem problem_solution : 
  ((-0.125 : ℝ)^2023 * 8^2024 = -8) ∧ 
  (((-27 : ℝ)^(1/3 : ℝ) + (5^2 : ℝ)^(1/2 : ℝ) - 2/3 * ((9/4 : ℝ)^(1/2 : ℝ))) = 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2157_215744


namespace NUMINAMATH_CALUDE_max_areas_for_n_eq_one_l2157_215747

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas for n = 1 -/
theorem max_areas_for_n_eq_one :
  ∀ (disk : DividedDisk),
    disk.n = 1 →
    max_areas disk = 15 :=
  sorry

end NUMINAMATH_CALUDE_max_areas_for_n_eq_one_l2157_215747


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l2157_215727

def a (n : ℕ) : ℚ := (4*n - 3) / (2*n + 1)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l2157_215727


namespace NUMINAMATH_CALUDE_veronica_cherry_pie_l2157_215703

/-- Given that:
  - There are 80 cherries in one pound
  - It takes 10 minutes to pit 20 cherries
  - It takes Veronica 2 hours to pit all the cherries
  Prove that Veronica needs 3 pounds of cherries for her pie. -/
theorem veronica_cherry_pie (cherries_per_pound : ℕ) (pit_time : ℕ) (pit_amount : ℕ) (total_time : ℕ) :
  cherries_per_pound = 80 →
  pit_time = 10 →
  pit_amount = 20 →
  total_time = 120 →
  (total_time / pit_time) * pit_amount / cherries_per_pound = 3 :=
by sorry

end NUMINAMATH_CALUDE_veronica_cherry_pie_l2157_215703


namespace NUMINAMATH_CALUDE_remainder_problem_l2157_215705

theorem remainder_problem (x : ℤ) : x % 95 = 31 → x % 19 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2157_215705


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2157_215738

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_standard_equation :
  ∀ (x y : ℝ → ℝ),
  (∃ (t : ℝ), x t = -3 ∧ y t = 2 * Real.sqrt 7) →
  (∃ (t : ℝ), x t = 6 * Real.sqrt 2 ∧ y t = -7) →
  (∃ (t : ℝ), x t = 2 ∧ y t = 2 * Real.sqrt 3) →
  (∀ (t : ℝ), (x t)^2 / 4 - (y t)^2 / 3 = 1 ↔ ∃ (k : ℝ), k * ((x t)^2 / 4 - (y t)^2 / 3) = k) →
  ∀ (t : ℝ), (y t)^2 / 9 - (x t)^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2157_215738


namespace NUMINAMATH_CALUDE_area_under_curve_l2157_215766

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the bounds
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem area_under_curve : 
  (∫ x in a..b, f x) = 4 := by sorry

end NUMINAMATH_CALUDE_area_under_curve_l2157_215766


namespace NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l2157_215781

/-- Represents a conic section defined by the equation ax² + y² = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Predicate to determine if a conic section is an ellipse or hyperbola -/
def is_ellipse_or_hyperbola (conic : ConicSection) : Prop :=
  sorry

/-- Theorem stating that c ≠ 0 is necessary but not sufficient for
    ax² + y² = c to represent an ellipse or hyperbola -/
theorem c_neq_zero_necessary_not_sufficient :
  (∀ conic : ConicSection, is_ellipse_or_hyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬is_ellipse_or_hyperbola conic) :=
sorry

end NUMINAMATH_CALUDE_c_neq_zero_necessary_not_sufficient_l2157_215781


namespace NUMINAMATH_CALUDE_symmetry_across_x_axis_l2157_215776

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetry_across_x_axis :
  let P : Point := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_across_x_axis_l2157_215776


namespace NUMINAMATH_CALUDE_equal_tuesdays_fridays_count_l2157_215722

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Counts the number of occurrences of a specific weekday in a month -/
def countWeekday (startDay : Weekday) (targetDay : Weekday) : Nat :=
  sorry

/-- Checks if the number of Tuesdays equals the number of Fridays for a given start day -/
def hasSameTuesdaysAndFridays (startDay : Weekday) : Bool :=
  countWeekday startDay Weekday.Tuesday = countWeekday startDay Weekday.Friday

/-- The set of all possible start days that result in equal Tuesdays and Fridays -/
def validStartDays : Finset Weekday :=
  sorry

theorem equal_tuesdays_fridays_count :
  Finset.card validStartDays = 4 := by sorry

end NUMINAMATH_CALUDE_equal_tuesdays_fridays_count_l2157_215722


namespace NUMINAMATH_CALUDE_rabbit_speed_l2157_215708

/-- Proves that given a dog running at 24 miles per hour chasing a rabbit with a 0.6-mile head start,
    if it takes the dog 4 minutes to catch up to the rabbit, then the rabbit's speed is 15 miles per hour. -/
theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  dog_speed = 24 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  ∃ (rabbit_speed : ℝ),
    rabbit_speed * catch_up_time = dog_speed * catch_up_time - head_start ∧
    rabbit_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l2157_215708


namespace NUMINAMATH_CALUDE_animath_interns_pigeonhole_l2157_215768

theorem animath_interns_pigeonhole (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_animath_interns_pigeonhole_l2157_215768


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2157_215748

theorem cloth_cost_price (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) :
  meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 35 →
  (selling_price - meters * profit_per_meter) / meters = 70 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2157_215748


namespace NUMINAMATH_CALUDE_arthur_actual_weight_l2157_215717

/-- The weight shown on the scales when weighing King Arthur -/
def arthur_scale : ℕ := 19

/-- The weight shown on the scales when weighing the royal horse -/
def horse_scale : ℕ := 101

/-- The weight shown on the scales when weighing King Arthur and the horse together -/
def combined_scale : ℕ := 114

/-- The actual weight of King Arthur -/
def arthur_weight : ℕ := 13

/-- The consistent error of the scales -/
def scale_error : ℕ := 6

theorem arthur_actual_weight :
  arthur_weight + scale_error = arthur_scale ∧
  arthur_weight + (horse_scale - scale_error) + scale_error = combined_scale :=
sorry

end NUMINAMATH_CALUDE_arthur_actual_weight_l2157_215717


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l2157_215730

/-- Piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a^x

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l2157_215730


namespace NUMINAMATH_CALUDE_min_students_is_minimum_l2157_215780

/-- The minimum number of students in the circle -/
def min_students : ℕ := 37

/-- Congcong's numbers are congruent modulo the number of students -/
axiom congcong_congruence : 25 ≡ 99 [ZMOD min_students]

/-- Mingming's numbers are congruent modulo the number of students -/
axiom mingming_congruence : 8 ≡ 119 [ZMOD min_students]

/-- The number of students is the minimum positive integer satisfying both congruences -/
theorem min_students_is_minimum :
  ∀ m : ℕ, m > 0 → (25 ≡ 99 [ZMOD m] ∧ 8 ≡ 119 [ZMOD m]) → m ≥ min_students :=
by sorry

end NUMINAMATH_CALUDE_min_students_is_minimum_l2157_215780


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2157_215761

/-- Given a geometric sequence {a_n} with sum S_n = 3^(n-1) + t for all n ≥ 1,
    prove that t + a_3 = 17/3 -/
theorem geometric_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (t : ℚ) 
  (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^(n-1) + t)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1))
  (h3 : ∀ n m : ℕ, n ≥ 1 → m ≥ 1 → a (n+1) / a n = a (m+1) / a m) :
  t + a 3 = 17/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2157_215761
