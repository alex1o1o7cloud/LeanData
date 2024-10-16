import Mathlib

namespace NUMINAMATH_CALUDE_james_muffins_count_l1959_195922

def arthur_muffins : ℝ := 115.0
def baking_ratio : ℝ := 12.0

theorem james_muffins_count : 
  arthur_muffins / baking_ratio = 9.5833 := by sorry

end NUMINAMATH_CALUDE_james_muffins_count_l1959_195922


namespace NUMINAMATH_CALUDE_weight_difference_l1959_195966

-- Define the weights as real numbers
variable (W_A W_B W_C W_D W_E : ℝ)

-- Define the conditions
def condition1 : Prop := (W_A + W_B + W_C) / 3 = 80
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 82
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 81
def condition4 : Prop := W_A = 95
def condition5 : Prop := W_E > W_D

-- Theorem statement
theorem weight_difference (h1 : condition1 W_A W_B W_C)
                          (h2 : condition2 W_A W_B W_C W_D)
                          (h3 : condition3 W_B W_C W_D W_E)
                          (h4 : condition4 W_A)
                          (h5 : condition5 W_D W_E) : 
  W_E - W_D = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l1959_195966


namespace NUMINAMATH_CALUDE_correct_subtraction_l1959_195991

theorem correct_subtraction (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l1959_195991


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1959_195935

theorem polynomial_factorization (m : ℤ) : 
  (∃ (A B C D E F : ℤ), 
    (A * x + B * y + C) * (D * x + E * y + F) = x^2 + 4*x*y + 2*x + m*y + m^2 - 16) ↔ 
  (m = 5 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1959_195935


namespace NUMINAMATH_CALUDE_charlotte_boots_cost_l1959_195937

/-- Calculates the amount Charlotte needs to bring to buy discounted boots -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (discount_rate * original_price)

/-- Proves that Charlotte needs to bring $72 for the boots -/
theorem charlotte_boots_cost : discounted_price 90 0.2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_charlotte_boots_cost_l1959_195937


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1959_195962

-- Define the piecewise function g
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then c * x + 4
  else if x ≥ -3 then x - 6
  else 3 * x - d

-- Theorem statement
theorem continuous_piecewise_function_sum (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -7/3 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1959_195962


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l1959_195969

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 6

/-- The weight of the first alloy in kg -/
def weight_first : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 7.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first * weight_first + chromium_percentage_second * weight_second) / (weight_first + weight_second) = chromium_percentage_result :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l1959_195969


namespace NUMINAMATH_CALUDE_golden_ratio_expression_l1959_195973

theorem golden_ratio_expression (S : ℝ) (h : S^2 + S - 1 = 0) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_expression_l1959_195973


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1959_195968

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1959_195968


namespace NUMINAMATH_CALUDE_min_sum_squares_l1959_195908

theorem min_sum_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3*a*b*c = 8) :
  ∃ (m : ℝ), (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3*x*y*z = 8 → x^2 + y^2 + z^2 ≥ m) ∧
             (a^2 + b^2 + c^2 ≥ m) ∧
             (m = 4) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1959_195908


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1959_195910

theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) / 
  (Real.cos (20 * π / 180) - Real.sqrt (1 - Real.cos (160 * π / 180) ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1959_195910


namespace NUMINAMATH_CALUDE_min_gumballs_for_given_machine_l1959_195996

/-- Represents the number of gumballs of each color -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- The minimum number of gumballs needed to guarantee at least 4 of the same color -/
def minGumballs (machine : GumballMachine) : ℕ := sorry

/-- Theorem stating the minimum number of gumballs needed for the given machine -/
theorem min_gumballs_for_given_machine :
  let machine : GumballMachine := ⟨8, 10, 6⟩
  minGumballs machine = 10 := by sorry

end NUMINAMATH_CALUDE_min_gumballs_for_given_machine_l1959_195996


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_range_l1959_195929

/-- An isosceles triangle with perimeter 16 -/
structure IsoscelesTriangle where
  x : ℝ  -- base length
  y : ℝ  -- leg length
  perimeter_eq : x + 2*y = 16
  leg_eq : y = -1/2 * x + 8

/-- The range of the base length x in an isosceles triangle -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) : 0 < t.x ∧ t.x < 8 := by
  sorry

#check isosceles_triangle_base_range

end NUMINAMATH_CALUDE_isosceles_triangle_base_range_l1959_195929


namespace NUMINAMATH_CALUDE_partner_calculation_l1959_195936

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l1959_195936


namespace NUMINAMATH_CALUDE_janes_coins_l1959_195948

theorem janes_coins (q d : ℕ) : 
  q + d = 30 → 
  (10 * q + 25 * d) - (25 * q + 10 * d) = 150 →
  25 * q + 10 * d = 450 :=
by sorry

end NUMINAMATH_CALUDE_janes_coins_l1959_195948


namespace NUMINAMATH_CALUDE_cashier_bills_l1959_195911

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 126) 
  (h2 : total_value = 840) : ∃ (five_dollar_bills ten_dollar_bills : ℕ), 
  five_dollar_bills + ten_dollar_bills = total_bills ∧ 
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧ 
  five_dollar_bills = 84 := by
sorry

end NUMINAMATH_CALUDE_cashier_bills_l1959_195911


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1959_195954

/-- A quadratic function passing through three specific points has a coefficient of 8/5 for its x² term. -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c → 
    ((x = -3 ∧ y = 2) ∨ (x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = -6))) → 
  a = 8/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1959_195954


namespace NUMINAMATH_CALUDE_range_of_a_l1959_195906

theorem range_of_a (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1959_195906


namespace NUMINAMATH_CALUDE_min_sum_at_6_l1959_195987

/-- Arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -11
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * n - 24)

/-- Theorem stating that S_n reaches its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → S seq 6 ≤ S seq n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l1959_195987


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l1959_195946

/-- Represents the savings from Coupon A (20% discount) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B ($50 flat discount) -/
def savingsB : ℝ := 50

/-- Represents the savings from Coupon C (30% discount on amount over $200) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 200)

/-- The minimum price where Coupon A saves at least as much as Coupons B and C -/
def minPrice : ℝ := 250

/-- The maximum price where Coupon A saves at least as much as Coupons B and C -/
def maxPrice : ℝ := 600

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 200 →
  (savingsA price ≥ savingsB ∧ savingsA price ≥ savingsC price) →
  minPrice ≤ price ∧ price ≤ maxPrice →
  maxPrice - minPrice = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l1959_195946


namespace NUMINAMATH_CALUDE_unique_function_identity_l1959_195904

theorem unique_function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x ≠ 0, f x = x^2 * f (1/x))
  (h2 : ∀ x y, f (x + y) = f x + f y)
  (h3 : f 1 = 1) :
  ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_function_identity_l1959_195904


namespace NUMINAMATH_CALUDE_original_fraction_value_l1959_195967

theorem original_fraction_value (x : ℚ) : 
  (x + 1) / (x + 8) = 11 / 17 → x / (x + 7) = 71 / 113 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_value_l1959_195967


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1959_195956

theorem no_solution_for_equation :
  ¬ ∃ (x : ℝ), (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1959_195956


namespace NUMINAMATH_CALUDE_people_eating_both_veg_and_nonveg_l1959_195963

theorem people_eating_both_veg_and_nonveg (veg_only : ℕ) (nonveg_only : ℕ) (total_veg : ℕ) 
  (h1 : veg_only = 15)
  (h2 : nonveg_only = 8)
  (h3 : total_veg = 26) :
  total_veg - veg_only = 11 := by
  sorry

#check people_eating_both_veg_and_nonveg

end NUMINAMATH_CALUDE_people_eating_both_veg_and_nonveg_l1959_195963


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1959_195953

/-- The parabola -/
def parabola (x : ℝ) : ℝ := x^2

/-- The line parallel to the tangent line -/
def parallel_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

/-- The proposed tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- Theorem: The tangent line to the parabola y = x^2 that is parallel to 2x - y + 4 = 0 
    has the equation 2x - y - 1 = 0 -/
theorem tangent_line_equation : 
  ∃ (x₀ y₀ : ℝ), 
    y₀ = parabola x₀ ∧ 
    tangent_line x₀ y₀ ∧
    ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), y = k*x + (k*2 - 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1959_195953


namespace NUMINAMATH_CALUDE_parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l1959_195993

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the distance between a point and a line
def distance_point_line (p : Point) (l : Line) : ℝ := sorry

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define when two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the given lines and point
def line1 : Line := λ x y _ ↦ 3 * x + 4 * y - 2 = 0
def line2 : Line := λ x y _ ↦ x + 3 * y - 5 = 0
def P : Point := (-1, 0)

-- Theorem for the first part
theorem parallel_lines_at_distance_1 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x + 4 * y + 3 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x + 4 * y - 7 = z) ∧
    parallel l1 line1 ∧
    parallel l2 line1 ∧
    (∀ p, distance_point_line p l1 = 1) ∧
    (∀ p, distance_point_line p l2 = 1) := sorry

-- Theorem for the second part
theorem perpendicular_lines_at_distance_sqrt2 :
  ∃ (l1 l2 : Line),
    (∀ x y z, l1 x y z ↔ 3 * x - y + 9 = z) ∧
    (∀ x y z, l2 x y z ↔ 3 * x - y - 3 = z) ∧
    perpendicular l1 line2 ∧
    perpendicular l2 line2 ∧
    distance_point_line P l1 = Real.sqrt 2 ∧
    distance_point_line P l2 = Real.sqrt 2 := sorry

end NUMINAMATH_CALUDE_parallel_lines_at_distance_1_perpendicular_lines_at_distance_sqrt2_l1959_195993


namespace NUMINAMATH_CALUDE_age_ratio_five_years_ago_l1959_195988

/-- Represents the ages of Lucy and Lovely -/
structure Ages where
  lucy : ℕ
  lovely : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  a.lucy = 50 ∧
  ∃ x : ℚ, (a.lucy - 5 : ℚ) = x * (a.lovely - 5 : ℚ) ∧
  (a.lucy + 10 : ℚ) = 2 * (a.lovely + 10 : ℚ)

/-- The theorem statement -/
theorem age_ratio_five_years_ago (a : Ages) :
  problem_conditions a →
  (a.lucy - 5 : ℚ) / (a.lovely - 5 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_five_years_ago_l1959_195988


namespace NUMINAMATH_CALUDE_system_solution_l1959_195939

theorem system_solution : ∃! (x y : ℚ), 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -241/29 ∧ y = -32/29 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1959_195939


namespace NUMINAMATH_CALUDE_first_day_exceeding_threshold_l1959_195984

-- Define the growth function for the bacteria colony
def bacteriaCount (n : ℕ) : ℕ := 4 * 3^n

-- Define the threshold
def threshold : ℕ := 200

-- Theorem statement
theorem first_day_exceeding_threshold :
  (∃ n : ℕ, bacteriaCount n > threshold) ∧
  (∀ k : ℕ, k < 4 → bacteriaCount k ≤ threshold) ∧
  (bacteriaCount 4 > threshold) := by
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_threshold_l1959_195984


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_3063_l1959_195907

theorem smallest_prime_factor_of_3063 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3063 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3063 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_3063_l1959_195907


namespace NUMINAMATH_CALUDE_sophomore_count_l1959_195972

theorem sophomore_count (n : ℕ) : 
  n > 1000 → -- Ensure n is large enough to accommodate all students
  (60 : ℚ) / n = (27 : ℚ) / 450 →
  n = 450 + 250 + 300 :=
by
  sorry

end NUMINAMATH_CALUDE_sophomore_count_l1959_195972


namespace NUMINAMATH_CALUDE_divisible_by_seven_l1959_195925

def repeated_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

theorem divisible_by_seven : ∃ k : Nat,
  (repeated_digit 8 50 * 10 + 5) * 10^50 + repeated_digit 9 50 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l1959_195925


namespace NUMINAMATH_CALUDE_fraction_equality_l1959_195978

theorem fraction_equality (a b c : ℝ) :
  (3 * a^2 + 3 * b^2 - 5 * c^2 + 6 * a * b) / (4 * a^2 + 4 * c^2 - 6 * b^2 + 8 * a * c) =
  ((a + b + Real.sqrt (5 * c^2)) * (a + b - Real.sqrt (5 * c^2))) /
  ((2 * (a + c) + Real.sqrt (6 * b^2)) * (2 * (a + c) - Real.sqrt (6 * b^2))) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1959_195978


namespace NUMINAMATH_CALUDE_equation_solution_l1959_195949

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 63 / 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1959_195949


namespace NUMINAMATH_CALUDE_juan_tire_count_l1959_195915

/-- The number of tires on the vehicles Juan saw --/
def total_tires (cars trucks bicycles tricycles : ℕ) : ℕ :=
  4 * (cars + trucks) + 2 * bicycles + 3 * tricycles

/-- Theorem stating the total number of tires Juan saw --/
theorem juan_tire_count : total_tires 15 8 3 1 = 101 := by
  sorry

end NUMINAMATH_CALUDE_juan_tire_count_l1959_195915


namespace NUMINAMATH_CALUDE_max_candy_pieces_l1959_195997

theorem max_candy_pieces (n : ℕ) (mean : ℕ) (min_pieces : ℕ) :
  n = 25 →
  mean = 7 →
  min_pieces = 2 →
  ∃ (max_pieces : ℕ),
    max_pieces = n * mean - (n - 1) * min_pieces ∧
    max_pieces = 127 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l1959_195997


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l1959_195940

/-- A rectangular parallelepiped with length and width twice the height and sum of edge lengths 100 cm has surface area 400 cm² -/
theorem rectangular_parallelepiped_surface_area 
  (h : ℝ) 
  (sum_edges : 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100) : 
  2 * (2 * h) * (2 * h) + 2 * (2 * h) * h + 2 * (2 * h) * h = 400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l1959_195940


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l1959_195971

/-- The equation of a circle with center (1, -1) tangent to the line x + y - √6 = 0 --/
theorem circle_equation_with_tangent_line :
  ∃ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 3 ∧
  (x + y - Real.sqrt 6 = 0 → 
    ∃ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ + 1)^2 = 3 ∧ x₀ + y₀ - Real.sqrt 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l1959_195971


namespace NUMINAMATH_CALUDE_restaurant_bill_solution_l1959_195992

/-- Represents the restaurant bill problem -/
def restaurant_bill_problem (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ children : ℕ, 
    adults * meal_cost + children * meal_cost = total_bill

/-- Theorem stating the solution to the restaurant bill problem -/
theorem restaurant_bill_solution :
  restaurant_bill_problem 2 8 56 → ∃ children : ℕ, children = 5 :=
by
  sorry

#check restaurant_bill_solution

end NUMINAMATH_CALUDE_restaurant_bill_solution_l1959_195992


namespace NUMINAMATH_CALUDE_poem_line_growth_l1959_195977

/-- The number of months required to reach a target number of lines in a poem, 
    given the initial number of lines and the number of lines added per month. -/
def months_to_reach_target (initial_lines : ℕ) (lines_per_month : ℕ) (target_lines : ℕ) : ℕ :=
  (target_lines - initial_lines) / lines_per_month

theorem poem_line_growth : months_to_reach_target 24 3 90 = 22 := by
  sorry

end NUMINAMATH_CALUDE_poem_line_growth_l1959_195977


namespace NUMINAMATH_CALUDE_squirrel_difference_l1959_195960

def scotland_squirrels : ℕ := 120000
def scotland_percentage : ℚ := 3/4

theorem squirrel_difference : 
  scotland_squirrels - (scotland_squirrels / scotland_percentage - scotland_squirrels) = 80000 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_difference_l1959_195960


namespace NUMINAMATH_CALUDE_min_moves_ten_elements_l1959_195958

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single move in the circular arrangement -/
def Move (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is sorted in ascending order -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of moves required to sort the arrangement -/
def MinMoves (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: The minimum number of moves to sort 10 distinct elements in a circle is 8 -/
theorem min_moves_ten_elements :
  ∀ (arr : CircularArrangement 10), MinMoves 10 arr = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_ten_elements_l1959_195958


namespace NUMINAMATH_CALUDE_parabola_c_value_l1959_195998

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_c_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 6 →
  Parabola b c 5 = 10 →
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1959_195998


namespace NUMINAMATH_CALUDE_coffee_brew_efficiency_l1959_195930

theorem coffee_brew_efficiency (total_lbs : ℕ) (cups_per_day : ℕ) (total_days : ℕ) 
  (h1 : total_lbs = 3)
  (h2 : cups_per_day = 3)
  (h3 : total_days = 40) :
  (cups_per_day * total_days) / total_lbs = 40 := by
  sorry

#check coffee_brew_efficiency

end NUMINAMATH_CALUDE_coffee_brew_efficiency_l1959_195930


namespace NUMINAMATH_CALUDE_f_minimum_at_neg_two_l1959_195980

/-- The function f(x) = |x+1| + |x+2| + |x+3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem f_minimum_at_neg_two :
  f (-2) = 2 ∧ ∀ x : ℝ, f x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_at_neg_two_l1959_195980


namespace NUMINAMATH_CALUDE_remainder_50_power_50_mod_7_l1959_195919

theorem remainder_50_power_50_mod_7 : 50^50 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_50_power_50_mod_7_l1959_195919


namespace NUMINAMATH_CALUDE_random_events_identification_l1959_195912

-- Define the types of events
inductive EventType
  | Random
  | Impossible
  | Certain

-- Define the events
structure Event :=
  (description : String)
  (eventType : EventType)

-- Define the function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  e.eventType = EventType.Random

-- Define the events
def coinEvent : Event :=
  { description := "Picking a 10 cent coin from a pocket with 50 cent, 10 cent, and 1 yuan coins"
  , eventType := EventType.Random }

def waterEvent : Event :=
  { description := "Water boiling at 90°C under standard atmospheric pressure"
  , eventType := EventType.Impossible }

def shooterEvent : Event :=
  { description := "A shooter hitting the 10-ring in one shot"
  , eventType := EventType.Random }

def diceEvent : Event :=
  { description := "Rolling two dice and the sum not exceeding 12"
  , eventType := EventType.Certain }

-- Theorem to prove
theorem random_events_identification :
  (isRandomEvent coinEvent ∧ isRandomEvent shooterEvent) ∧
  (¬isRandomEvent waterEvent ∧ ¬isRandomEvent diceEvent) :=
sorry

end NUMINAMATH_CALUDE_random_events_identification_l1959_195912


namespace NUMINAMATH_CALUDE_bexy_bicycle_speed_l1959_195959

/-- Bexy's round trip problem -/
theorem bexy_bicycle_speed :
  -- Bexy's walking distance and time
  let bexy_walk_distance : ℝ := 5
  let bexy_walk_time : ℝ := 1

  -- Ben's total round trip time in hours
  let ben_total_time : ℝ := 160 / 60

  -- Ben's speed relative to Bexy's average speed
  let ben_speed_ratio : ℝ := 1 / 2

  -- Bexy's bicycle speed
  ∃ bexy_bike_speed : ℝ,
    -- Ben's walking time is twice Bexy's
    let ben_walk_time : ℝ := 2 * bexy_walk_time

    -- Ben's biking time
    let ben_bike_time : ℝ := ben_total_time - ben_walk_time

    -- Ben's biking speed
    let ben_bike_speed : ℝ := bexy_bike_speed * ben_speed_ratio

    -- Distance traveled equals speed times time
    ben_bike_speed * ben_bike_time = bexy_walk_distance ∧
    bexy_bike_speed = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bexy_bicycle_speed_l1959_195959


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_l1959_195989

def curve (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

def bisection_point : ℝ × ℝ := (3, -1)

def chord_equation (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

theorem chord_bisected_by_point (m b : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    curve x₁ y₁ ∧ 
    curve x₂ y₂ ∧ 
    chord_equation m b x₁ y₁ ∧ 
    chord_equation m b x₂ y₂ ∧ 
    ((x₁ + x₂)/2, (y₁ + y₂)/2) = bisection_point) →
  chord_equation (-3/4) (11/4) 3 (-1) ∧ 
  (∀ x y : ℝ, chord_equation (-3/4) (11/4) x y ↔ 3*x + 4*y - 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_l1959_195989


namespace NUMINAMATH_CALUDE_x_intercept_distance_l1959_195913

/-- Given two lines with slopes 4 and -2 intersecting at (8, 20),
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →  -- Equation of line1
  (∀ x, line2 x = -2 * x + 36) →  -- Equation of line2
  line1 8 = 20 →  -- Intersection point
  line2 8 = 20 →  -- Intersection point
  |((36 : ℝ) / 2) - (12 / 4)| = 15 := by
  sorry


end NUMINAMATH_CALUDE_x_intercept_distance_l1959_195913


namespace NUMINAMATH_CALUDE_tank_emptying_equivalence_l1959_195901

/-- Represents the work capacity of pumps emptying a tank -/
def tank_emptying_work (pumps : ℕ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  pumps * hours_per_day * days

theorem tank_emptying_equivalence (d : ℝ) (h : d > 0) :
  let original_work := tank_emptying_work 3 8 2
  let new_work := tank_emptying_work 6 (8 / d) d
  original_work = new_work :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_equivalence_l1959_195901


namespace NUMINAMATH_CALUDE_sum_properties_l1959_195986

theorem sum_properties (a b : ℤ) (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  Even (a + b) ∧ (4 ∣ (a + b)) ∧ ¬(∀ (a b : ℤ), 4 ∣ a → 8 ∣ b → 8 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_sum_properties_l1959_195986


namespace NUMINAMATH_CALUDE_game_download_time_l1959_195917

/-- Calculates the remaining download time for a game -/
theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 ∧ downloaded = 310 ∧ speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end NUMINAMATH_CALUDE_game_download_time_l1959_195917


namespace NUMINAMATH_CALUDE_larger_number_problem_l1959_195964

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 35 → max x y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1959_195964


namespace NUMINAMATH_CALUDE_transformed_circle_center_l1959_195905

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

def circle_center : ℝ × ℝ := (4, -3)

theorem transformed_circle_center :
  let reflected := reflect_x circle_center
  let translated_right := translate reflected 5 0
  let final_position := translate translated_right 0 3
  final_position = (9, 6) := by sorry

end NUMINAMATH_CALUDE_transformed_circle_center_l1959_195905


namespace NUMINAMATH_CALUDE_triangular_array_theorem_l1959_195943

/-- Represents the elements of a triangular array -/
def a (i j : ℕ) : ℚ :=
  sorry

/-- The common ratio for geometric sequences in rows -/
def common_ratio : ℚ := 1 / 2

/-- The common difference for the arithmetic sequence in the first column -/
def common_diff : ℚ := 1 / 4

theorem triangular_array_theorem (n : ℕ) (h : n > 0) :
  ∀ (i j : ℕ), i ≥ j → i > 0 → j > 0 →
  (∀ k, k > 0 → a k 1 - a (k-1) 1 = common_diff) →
  (∀ k l, k > 2 → l > 0 → a k (l+1) / a k l = common_ratio) →
  a n 3 = n / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_theorem_l1959_195943


namespace NUMINAMATH_CALUDE_no_quinary_country_46_airlines_l1959_195965

/-- A quinary country is a country where each city is connected by air lines with exactly five other cities. -/
structure QuinaryCountry where
  cities : ℕ
  airLines : ℕ
  isQuinary : airLines = (cities * 5) / 2

/-- Theorem: There cannot exist a quinary country with exactly 46 air lines. -/
theorem no_quinary_country_46_airlines : ¬ ∃ (q : QuinaryCountry), q.airLines = 46 := by
  sorry

end NUMINAMATH_CALUDE_no_quinary_country_46_airlines_l1959_195965


namespace NUMINAMATH_CALUDE_fraction_equality_l1959_195944

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (1 / a + 1 / b = 1 / 2) → (a * b / (a + b) = 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1959_195944


namespace NUMINAMATH_CALUDE_shelving_orders_eq_1280_l1959_195961

/-- The number of books in total -/
def total_books : ℕ := 10

/-- The label of the book that has already been shelved -/
def shelved_book : ℕ := 9

/-- Calculate the number of different possible orders for shelving the remaining books -/
def shelving_orders : ℕ :=
  (Finset.range (total_books - 1)).sum (fun k =>
    (Nat.choose (total_books - 2) k) * (k + 2))

/-- Theorem stating that the number of different possible orders for shelving the remaining books is 1280 -/
theorem shelving_orders_eq_1280 : shelving_orders = 1280 := by
  sorry

end NUMINAMATH_CALUDE_shelving_orders_eq_1280_l1959_195961


namespace NUMINAMATH_CALUDE_prob_not_all_same_five_8sided_dice_l1959_195955

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability that not all k n-sided dice show the same number -/
def prob_not_all_same (n k : ℕ) : ℚ :=
  1 - (n : ℚ) / (n ^ k : ℚ)

/-- Theorem: The probability of not all five 8-sided dice showing the same number is 4095/4096 -/
theorem prob_not_all_same_five_8sided_dice :
  prob_not_all_same n k = 4095 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_five_8sided_dice_l1959_195955


namespace NUMINAMATH_CALUDE_resource_sum_theorem_l1959_195900

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The amount of mineral X in base 6 -/
def mineral_x : List Nat := [2, 3, 4, 1]

/-- The amount of mineral Y in base 6 -/
def mineral_y : List Nat := [4, 1, 2, 3]

/-- The amount of water in base 6 -/
def water : List Nat := [4, 1, 2]

theorem resource_sum_theorem :
  base6_to_base10 mineral_x + base6_to_base10 mineral_y + base6_to_base10 water = 868 := by
  sorry

end NUMINAMATH_CALUDE_resource_sum_theorem_l1959_195900


namespace NUMINAMATH_CALUDE_schedule_arrangements_l1959_195918

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Calculates the number of arrangements for scheduling subjects with given constraints -/
def num_arrangements : ℕ :=
  5 * 4 * (Finset.range 4).prod (λ i => i + 1)

/-- Theorem stating the number of different arrangements -/
theorem schedule_arrangements :
  num_arrangements = 480 := by sorry

end NUMINAMATH_CALUDE_schedule_arrangements_l1959_195918


namespace NUMINAMATH_CALUDE_unique_contributions_exist_l1959_195914

/-- Represents the contributions of five friends to a project -/
structure Contributions where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (c : Contributions) : Prop :=
  c.A = 1.1 * c.B ∧
  c.C = 0.8 * c.A ∧
  c.D = 2 * c.B ∧
  c.E = c.D - 200 ∧
  c.A + c.B + c.C + c.D + c.E = 1500

/-- Theorem stating that there exists a unique set of contributions satisfying the conditions -/
theorem unique_contributions_exist : ∃! c : Contributions, satisfies_conditions c :=
sorry

end NUMINAMATH_CALUDE_unique_contributions_exist_l1959_195914


namespace NUMINAMATH_CALUDE_bobby_candy_count_l1959_195957

/-- The number of candy pieces Bobby ate initially -/
def initial_candy : ℕ := 26

/-- The number of additional candy pieces Bobby ate -/
def additional_candy : ℕ := 17

/-- The total number of candy pieces Bobby ate -/
def total_candy : ℕ := initial_candy + additional_candy

theorem bobby_candy_count : total_candy = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l1959_195957


namespace NUMINAMATH_CALUDE_x_value_in_equation_l1959_195950

theorem x_value_in_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 24 * x * y = 2 * x^3 + 3 * x^2 * y^2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_equation_l1959_195950


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1959_195928

theorem sum_of_x_values (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt ((x - 2)^2) = 8 ↔ x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1959_195928


namespace NUMINAMATH_CALUDE_horner_method_equals_direct_evaluation_l1959_195976

/-- Horner's method for evaluating a polynomial --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^4 - 3x^2 + 7x - 2 --/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 - 3*x^2 + 7*x - 2

/-- Coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [-2, 7, 0, -3, 2, 1]

theorem horner_method_equals_direct_evaluation :
  horner coeffs 2 = f 2 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_direct_evaluation_l1959_195976


namespace NUMINAMATH_CALUDE_x_value_proof_l1959_195902

theorem x_value_proof (x y z : ℝ) (h1 : x = y) (h2 : y = 2 * z) (h3 : x * y * z = 256) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1959_195902


namespace NUMINAMATH_CALUDE_root_ratio_sum_zero_l1959_195951

theorem root_ratio_sum_zero (a b n p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) 
  (h_roots : ∃ (x y : ℝ), x / y = a / b ∧ p * x^2 + n * x + n = 0 ∧ p * y^2 + n * y + n = 0) :
  Real.sqrt (a / b) + Real.sqrt (b / a) + Real.sqrt (n / p) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_ratio_sum_zero_l1959_195951


namespace NUMINAMATH_CALUDE_smallest_b_value_l1959_195974

theorem smallest_b_value (a b c : ℚ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)  -- arithmetic sequence condition
  (h4 : c^2 = a * b)    -- geometric sequence condition
  : b ≥ (1/2 : ℚ) ∧ ∃ (a' b' c' : ℚ), 
    a' < b' ∧ b' < c' ∧ 
    2 * b' = a' + c' ∧ 
    c'^2 = a' * b' ∧ 
    b' = (1/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1959_195974


namespace NUMINAMATH_CALUDE_distribution_difference_l1959_195933

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
sorry

end NUMINAMATH_CALUDE_distribution_difference_l1959_195933


namespace NUMINAMATH_CALUDE_function_composition_theorem_l1959_195934

theorem function_composition_theorem (a b : ℤ) :
  (∃ f g : ℤ → ℤ, ∀ x : ℤ, f (g x) = x + a ∧ g (f x) = x + b) ↔ (a = b ∨ a = -b) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_theorem_l1959_195934


namespace NUMINAMATH_CALUDE_x_value_proof_l1959_195945

theorem x_value_proof (x : ℕ) 
  (h1 : x > 0) 
  (h2 : (x * x) / 100 = 16) 
  (h3 : 4 ∣ x) : 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l1959_195945


namespace NUMINAMATH_CALUDE_gcd_7979_3713_l1959_195990

theorem gcd_7979_3713 : Nat.gcd 7979 3713 = 79 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7979_3713_l1959_195990


namespace NUMINAMATH_CALUDE_symmetric_function_trigonometric_identity_l1959_195903

theorem symmetric_function_trigonometric_identity (θ : ℝ) :
  (∀ x : ℝ, x^2 + (Real.sin θ - Real.cos θ) * x + Real.sin θ = 
            (-x)^2 + (Real.sin θ - Real.cos θ) * (-x) + Real.sin θ) →
  2 * Real.sin θ * Real.cos θ + Real.cos (2 * θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_trigonometric_identity_l1959_195903


namespace NUMINAMATH_CALUDE_stratified_sampling_result_count_l1959_195970

def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_sample_size : ℕ := 60

def proportional_allocation (total : ℕ) (part : ℕ) (sample : ℕ) : ℕ :=
  (part * sample) / total

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem stratified_sampling_result_count :
  (binomial_coefficient junior_students (proportional_allocation (junior_students + senior_students) junior_students total_sample_size)) *
  (binomial_coefficient senior_students (proportional_allocation (junior_students + senior_students) senior_students total_sample_size)) =
  (binomial_coefficient 400 40) * (binomial_coefficient 200 20) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_count_l1959_195970


namespace NUMINAMATH_CALUDE_group_work_problem_l1959_195994

theorem group_work_problem (n : ℕ) (W_total : ℝ) : 
  (n : ℝ) * (W_total / 55) = ((n : ℝ) - 15) * (W_total / 60) → n = 165 := by
  sorry

end NUMINAMATH_CALUDE_group_work_problem_l1959_195994


namespace NUMINAMATH_CALUDE_square_difference_l1959_195920

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : 
  (x - y)^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1959_195920


namespace NUMINAMATH_CALUDE_eleventh_term_is_110_div_7_l1959_195926

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first six terms is 30
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 30
  -- Seventh term is 10
  seventh_term : a + 6*d = 10

/-- The eleventh term of the specific arithmetic sequence is 110/7 -/
theorem eleventh_term_is_110_div_7 (seq : ArithmeticSequence) :
  seq.a + 10*seq.d = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_110_div_7_l1959_195926


namespace NUMINAMATH_CALUDE_gala_dinner_seating_l1959_195975

/-- The number of couples to be seated -/
def num_couples : ℕ := 6

/-- The total number of people to be seated -/
def total_people : ℕ := 2 * num_couples

/-- The number of ways to arrange the husbands -/
def husband_arrangements : ℕ := (total_people - 1).factorial

/-- The number of equivalent arrangements due to rotation and reflection -/
def equivalent_arrangements : ℕ := 2 * total_people

/-- The number of unique seating arrangements -/
def unique_arrangements : ℕ := husband_arrangements / equivalent_arrangements

theorem gala_dinner_seating :
  unique_arrangements = 5760 :=
sorry

end NUMINAMATH_CALUDE_gala_dinner_seating_l1959_195975


namespace NUMINAMATH_CALUDE_unique_root_condition_l1959_195942

theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  a = (Real.log 6 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1959_195942


namespace NUMINAMATH_CALUDE_chessboard_clearable_l1959_195938

/-- Represents the number of chips on a chessboard -/
def Chessboard := Fin 8 → Fin 8 → ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | remove_column : Fin 8 → Operation
  | double_row : Fin 8 → Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.remove_column j => fun i k => if k = j then (board i k).pred else board i k
  | Operation.double_row i => fun k j => if k = i then 2 * (board k j) else board k j

/-- Checks if the board is cleared (all cells are zero) -/
def is_cleared (board : Chessboard) : Prop :=
  ∀ i j, board i j = 0

theorem chessboard_clearable (initial_board : Chessboard) :
  ∃ (ops : List Operation), is_cleared (ops.foldl apply_operation initial_board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_clearable_l1959_195938


namespace NUMINAMATH_CALUDE_constant_c_value_l1959_195995

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l1959_195995


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocals_l1959_195923

theorem min_sum_of_reciprocals (x y z : ℕ+) (h : (1 : ℚ) / x + 4 / y + 9 / z = 1) :
  36 ≤ (x : ℚ) + y + z :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocals_l1959_195923


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_formula_l1959_195982

/-- The sum of an arithmetic sequence of consecutive integers -/
def arithmeticSequenceSum (k : ℕ) : ℕ :=
  let firstTerm := (k - 1)^2 + 1
  let numTerms := 2 * k
  numTerms * (2 * firstTerm + (numTerms - 1)) / 2

theorem arithmetic_sequence_sum_formula (k : ℕ) :
  arithmeticSequenceSum k = 2 * k^3 + k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_formula_l1959_195982


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l1959_195921

theorem shopkeeper_loss_percentage
  (profit_rate : ℝ)
  (theft_rate : ℝ)
  (h_profit : profit_rate = 0.1)
  (h_theft : theft_rate = 0.2) :
  let selling_price := 1 + profit_rate
  let remaining_goods := 1 - theft_rate
  let cost_price_remaining := remaining_goods
  let selling_price_remaining := selling_price * remaining_goods
  let loss := theft_rate
  loss / cost_price_remaining = 0.25 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l1959_195921


namespace NUMINAMATH_CALUDE_revenue_growth_exists_l1959_195941

/-- Represents the revenue growth rate in a supermarket over three months -/
def revenue_growth_equation (x : ℝ) : Prop :=
  let january_revenue : ℝ := 90
  let total_revenue : ℝ := 144
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = total_revenue

/-- Theorem stating that the revenue growth equation holds for some growth rate x -/
theorem revenue_growth_exists : ∃ x : ℝ, revenue_growth_equation x := by
  sorry

end NUMINAMATH_CALUDE_revenue_growth_exists_l1959_195941


namespace NUMINAMATH_CALUDE_birds_on_fence_l1959_195979

/-- The number of additional birds that joined the fence -/
def additional_birds : ℕ := sorry

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 2

/-- The number of storks that joined the fence -/
def joined_storks : ℕ := 4

theorem birds_on_fence :
  additional_birds = 5 :=
by
  have h1 : initial_birds + additional_birds = joined_storks + 3 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1959_195979


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l1959_195909

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l1959_195909


namespace NUMINAMATH_CALUDE_jacks_burgers_l1959_195952

/-- Given Jack's barbecue sauce recipe and usage, prove how many burgers he can make. -/
theorem jacks_burgers :
  -- Total sauce
  let total_sauce : ℚ := 3 + 1 + 1

  -- Sauce per burger
  let sauce_per_burger : ℚ := 1 / 4

  -- Sauce per pulled pork sandwich
  let sauce_per_pps : ℚ := 1 / 6

  -- Number of pulled pork sandwiches
  let num_pps : ℕ := 18

  -- Sauce used for pulled pork sandwiches
  let sauce_for_pps : ℚ := sauce_per_pps * num_pps

  -- Remaining sauce for burgers
  let remaining_sauce : ℚ := total_sauce - sauce_for_pps

  -- Number of burgers Jack can make
  ↑(remaining_sauce / sauce_per_burger).floor = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_jacks_burgers_l1959_195952


namespace NUMINAMATH_CALUDE_f_properties_l1959_195981

-- Define the function f(x) = (x-2)(x+4)
def f (x : ℝ) : ℝ := (x - 2) * (x + 4)

-- Theorem statement
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x > f y) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y → f x < f y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -9) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = -9) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l1959_195981


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1959_195927

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  ((Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6/11) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1959_195927


namespace NUMINAMATH_CALUDE_smallest_square_perimeter_l1959_195947

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.sideLength

/-- Represents three concentric squares -/
structure ConcentricSquares where
  largest : Square
  middle : Square
  smallest : Square
  distanceBetweenSides : ℝ

/-- The theorem stating the perimeter of the smallest square in the given configuration -/
theorem smallest_square_perimeter (cs : ConcentricSquares)
    (h1 : cs.largest.sideLength = 22)
    (h2 : cs.distanceBetweenSides = 3)
    (h3 : cs.middle.sideLength = cs.largest.sideLength - 2 * cs.distanceBetweenSides)
    (h4 : cs.smallest.sideLength = cs.middle.sideLength - 2 * cs.distanceBetweenSides) :
    cs.smallest.perimeter = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_perimeter_l1959_195947


namespace NUMINAMATH_CALUDE_decreasing_order_l1959_195932

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the even property of f(x-1)
axiom f_even : ∀ x : ℝ, f (-x - 1) = f (x - 1)

-- Define the decreasing property of f on [-1,+∞)
axiom f_decreasing : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ ≥ -1 → x₂ ≥ -1 → 
  (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log (7/2) / Real.log (1/2))
noncomputable def b : ℝ := f (Real.log (7/2) / Real.log (1/3))
noncomputable def c : ℝ := f (Real.log (3/2) / Real.log 2)

-- The theorem to prove
theorem decreasing_order : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_decreasing_order_l1959_195932


namespace NUMINAMATH_CALUDE_like_terms_exponents_l1959_195999

theorem like_terms_exponents (m n : ℤ) : 
  (∃ (x y : ℝ), 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) → m = 4 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l1959_195999


namespace NUMINAMATH_CALUDE_zoo_guide_count_l1959_195916

/-- Represents the number of children spoken to by a guide on a specific day -/
structure DailyGuideCount where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- Represents the count of guides for each language -/
structure GuideCount where
  total : ℕ
  english : ℕ
  french : ℕ

def weekend_count (guides : GuideCount) (friday saturday sunday : DailyGuideCount) : ℕ :=
  let spanish_guides := guides.total - guides.english - guides.french
  let friday_total := guides.english * friday.english + guides.french * friday.french + spanish_guides * friday.spanish
  let saturday_total := guides.english * saturday.english + guides.french * saturday.french + spanish_guides * saturday.spanish
  let sunday_total := guides.english * sunday.english + guides.french * sunday.french + spanish_guides * sunday.spanish
  friday_total + saturday_total + sunday_total

theorem zoo_guide_count :
  let guides : GuideCount := { total := 22, english := 10, french := 6 }
  let friday : DailyGuideCount := { english := 20, french := 25, spanish := 30 }
  let saturday : DailyGuideCount := { english := 22, french := 24, spanish := 32 }
  let sunday : DailyGuideCount := { english := 24, french := 23, spanish := 35 }
  weekend_count guides friday saturday sunday = 1674 := by
  sorry


end NUMINAMATH_CALUDE_zoo_guide_count_l1959_195916


namespace NUMINAMATH_CALUDE_product_of_roots_l1959_195924

theorem product_of_roots (t : ℝ) : (∀ t, t^2 = 49) → (t * (-t) = -49) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1959_195924


namespace NUMINAMATH_CALUDE_megans_books_l1959_195985

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end NUMINAMATH_CALUDE_megans_books_l1959_195985


namespace NUMINAMATH_CALUDE_reflection_matrix_correct_l1959_195931

/-- Reflection matrix over the line y = x -/
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1; 1, 0]

/-- A point in 2D space -/
def Point := Fin 2 → ℝ

/-- Reflect a point over the line y = x -/
def reflect (p : Point) : Point :=
  λ i => p (if i = 0 then 1 else 0)

theorem reflection_matrix_correct :
  ∀ (p : Point), reflection_matrix.mulVec p = reflect p :=
by sorry

end NUMINAMATH_CALUDE_reflection_matrix_correct_l1959_195931


namespace NUMINAMATH_CALUDE_overtime_earnings_calculation_l1959_195983

/-- Calculates the additional earnings per overtime shift -/
def overtime_earnings (regular_days : ℕ) (daily_wage : ℕ) (overtime_shifts : ℕ) (total_earnings : ℕ) : ℕ :=
  let regular_earnings := regular_days * daily_wage
  let total_overtime_earnings := total_earnings - regular_earnings
  total_overtime_earnings / overtime_shifts

/-- Proves that the additional earnings per overtime shift is $15 -/
theorem overtime_earnings_calculation :
  overtime_earnings 5 30 3 195 = 15 := by
  sorry

end NUMINAMATH_CALUDE_overtime_earnings_calculation_l1959_195983
