import Mathlib

namespace NUMINAMATH_CALUDE_frisbee_sales_minimum_receipts_l2560_256060

theorem frisbee_sales_minimum_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 8 →
  3 * x + 4 * y ≥ 200 :=
by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_minimum_receipts_l2560_256060


namespace NUMINAMATH_CALUDE_box_minus_two_zero_three_l2560_256051

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_minus_two_zero_three : box (-2) 0 3 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_box_minus_two_zero_three_l2560_256051


namespace NUMINAMATH_CALUDE_solution_sets_l2560_256064

def solution_set_1 (a b : ℝ) : Set ℝ := {x | a * x - b > 0}
def solution_set_2 (a b : ℝ) : Set ℝ := {x | (a * x + b) / (x - 2) > 0}

theorem solution_sets (a b : ℝ) :
  solution_set_1 a b = Set.Ioi 1 →
  solution_set_2 a b = Set.Iic (-1) ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l2560_256064


namespace NUMINAMATH_CALUDE_mikes_pumpkins_l2560_256065

theorem mikes_pumpkins (sandy_pumpkins : ℕ) (total_pumpkins : ℕ) (mike_pumpkins : ℕ) : 
  sandy_pumpkins = 51 → total_pumpkins = 74 → mike_pumpkins = total_pumpkins - sandy_pumpkins → mike_pumpkins = 23 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pumpkins_l2560_256065


namespace NUMINAMATH_CALUDE_first_hour_rate_is_25_l2560_256023

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℕ
  additionalHourRate : ℕ

/-- Represents the rental details for Ashwin -/
structure RentalDetails where
  totalCost : ℕ
  totalHours : ℕ

/-- Theorem stating that given the rental conditions, the first hour rate was $25 -/
theorem first_hour_rate_is_25 (rental : RentalCost) (details : RentalDetails) :
  rental.additionalHourRate = 10 ∧
  details.totalCost = 125 ∧
  details.totalHours = 11 →
  rental.firstHourRate = 25 := by
  sorry

#check first_hour_rate_is_25

end NUMINAMATH_CALUDE_first_hour_rate_is_25_l2560_256023


namespace NUMINAMATH_CALUDE_sequence_decreasing_l2560_256093

/-- Given real numbers a and b such that b > a > 1, define the sequence x_n as follows:
    x_n = 2^n * (b^(1/2^n) - a^(1/2^n))
    This theorem states that the sequence is decreasing. -/
theorem sequence_decreasing (a b : ℝ) (h1 : b > a) (h2 : a > 1) :
  ∀ n : ℕ, (2^n * (b^(1/(2^n)) - a^(1/(2^n)))) > (2^(n+1) * (b^(1/(2^(n+1))) - a^(1/(2^(n+1))))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l2560_256093


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2560_256038

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a line intersects a circle at exactly one point -/
def intersectsAtOnePoint (l : ParametricLine) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem line_circle_intersection (m : ℝ) :
  let l : ParametricLine := { x := λ t => 3 * t, y := λ t => 4 * t + m }
  let c : Circle := { center := (1, 0), radius := 1 }
  intersectsAtOnePoint l c → m = 1/3 ∨ m = -3 := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l2560_256038


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2560_256035

theorem fraction_evaluation : (3^4 - 3^3) / (3^(-2) + 3^(-1)) = 121.5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2560_256035


namespace NUMINAMATH_CALUDE_ceiling_squared_fraction_l2560_256058

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_squared_fraction_l2560_256058


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2560_256054

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2560_256054


namespace NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l2560_256073

/-- Proves that the sale price for making a 55% profit is $2792, given the conditions. -/
theorem sale_price_for_55_percent_profit 
  (equal_profit_loss : ∀ (cp sp_profit : ℝ), sp_profit - cp = cp - 448)
  (profit_amount : ∀ (cp : ℝ), 0.55 * cp = 992) :
  ∃ (cp : ℝ), cp + 992 = 2792 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l2560_256073


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l2560_256076

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 13)
  (h2 : z + x = 14)
  (h3 : x + y = 15) :
  Real.sqrt (x * y * z * (x + y + z)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l2560_256076


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2560_256045

def base_seven_to_ten (n : List Nat) : Nat :=
  List.foldr (λ (digit : Nat) (acc : Nat) => 7 * acc + digit) 0 n

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [2, 3, 4, 5, 6] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2560_256045


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2560_256037

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 11 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2560_256037


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2560_256098

/-- The area of a circle with diameter endpoints at (1, 3) and (8, 6) is 58π/4 square units. -/
theorem circle_area_from_diameter_endpoints :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (8, 6)
  let diameter_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 58 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l2560_256098


namespace NUMINAMATH_CALUDE_unique_x_for_all_y_l2560_256026

theorem unique_x_for_all_y : ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 18 * y + x - 2 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_x_for_all_y_l2560_256026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2560_256015

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 14) :
  (∀ n : ℕ, a n = 3 * n - 1) ∧
  (∃ n : ℕ, n * (a 1 + a n) / 2 = 155 ∧ n = 10) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2560_256015


namespace NUMINAMATH_CALUDE_ivan_tsarevich_revival_l2560_256027

/-- Represents the scenario of Wolf, Ivan Tsarevich, and the Raven --/
structure RevivalScenario where
  initialDistance : ℝ
  wolfSpeed : ℝ
  waterNeeded : ℝ
  springFlowRate : ℝ
  ravenSpeed : ℝ
  ravenWaterLossRate : ℝ

/-- Determines if Ivan Tsarevich can be revived after the given time --/
def canRevive (scenario : RevivalScenario) (time : ℝ) : Prop :=
  let waterCollectionTime := scenario.waterNeeded / scenario.springFlowRate
  let wolfDistance := scenario.wolfSpeed * waterCollectionTime
  let remainingDistance := scenario.initialDistance - wolfDistance
  let meetingTime := remainingDistance / (scenario.ravenSpeed + scenario.wolfSpeed)
  let totalTime := waterCollectionTime + meetingTime
  let waterLost := scenario.ravenWaterLossRate * meetingTime
  totalTime ≤ time ∧ scenario.waterNeeded - waterLost > 0

/-- The main theorem stating that Ivan Tsarevich can be revived after 4 hours --/
theorem ivan_tsarevich_revival (scenario : RevivalScenario)
  (h1 : scenario.initialDistance = 20)
  (h2 : scenario.wolfSpeed = 3)
  (h3 : scenario.waterNeeded = 1)
  (h4 : scenario.springFlowRate = 0.5)
  (h5 : scenario.ravenSpeed = 6)
  (h6 : scenario.ravenWaterLossRate = 0.25) :
  canRevive scenario 4 := by
  sorry

end NUMINAMATH_CALUDE_ivan_tsarevich_revival_l2560_256027


namespace NUMINAMATH_CALUDE_base_is_seven_l2560_256044

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

/-- The transaction equation holds for the given base -/
def transaction_holds (s : Nat) : Prop :=
  to_base_10 [3, 2, 5] s + to_base_10 [3, 5, 4] s = to_base_10 [0, 0, 1, 1] s

theorem base_is_seven :
  ∃ s : Nat, s > 1 ∧ transaction_holds s ∧ s = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_is_seven_l2560_256044


namespace NUMINAMATH_CALUDE_student_response_change_difference_l2560_256070

/-- Represents the percentages of student responses --/
structure ResponsePercentages :=
  (yes : ℝ)
  (no : ℝ)
  (undecided : ℝ)

/-- The problem statement --/
theorem student_response_change_difference 
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 40)
  (h_initial_undecided : initial.undecided = 20)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 30)
  (h_final_undecided : final.undecided = 10) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 40 :=
sorry

end NUMINAMATH_CALUDE_student_response_change_difference_l2560_256070


namespace NUMINAMATH_CALUDE_propositions_truth_l2560_256068

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem propositions_truth : 
  (∀ a : ℝ, a^2 ≥ 0) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ ∧ f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2560_256068


namespace NUMINAMATH_CALUDE_car_profit_percent_l2560_256021

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (car_cost repair_cost taxes insurance selling_price : ℝ) :
  car_cost = 36400 →
  repair_cost = 8000 →
  taxes = 4500 →
  insurance = 2500 →
  selling_price = 68400 →
  let total_cost := car_cost + repair_cost + taxes + insurance
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  abs (profit_percent - 33.07) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_car_profit_percent_l2560_256021


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2560_256020

theorem quadratic_inequality_solution_set (x : ℝ) : 
  -x^2 + 2*x + 3 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2560_256020


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2560_256042

theorem triangle_angle_sum (A B : Real) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2)
  (hsinA : Real.sin A = Real.sqrt 5 / 5) (hsinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = π/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2560_256042


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_equality_condition_l2560_256053

theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) = b / (5 * c) ∧ b / (5 * c) = c / (7 * a)) ↔
  (a / (3 * b) = 1 / Real.rpow 105 (1/3) ∧
   b / (5 * c) = 1 / Real.rpow 105 (1/3) ∧
   c / (7 * a) = 1 / Real.rpow 105 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_equality_condition_l2560_256053


namespace NUMINAMATH_CALUDE_speed_ratio_in_race_l2560_256047

/-- In a race, contestant A has a head start and wins. This theorem proves the ratio of their speeds. -/
theorem speed_ratio_in_race (total_distance : ℝ) (head_start : ℝ) (win_margin : ℝ)
  (h1 : total_distance = 500)
  (h2 : head_start = 300)
  (h3 : win_margin = 100)
  : (total_distance - head_start) / (total_distance - win_margin) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_in_race_l2560_256047


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2560_256084

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = Set.Ioo a (1/a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2560_256084


namespace NUMINAMATH_CALUDE_units_digit_base_6_product_l2560_256096

theorem units_digit_base_6_product (a b : ℕ) (ha : a = 312) (hb : b = 67) :
  (a * b) % 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_base_6_product_l2560_256096


namespace NUMINAMATH_CALUDE_range_of_expression_l2560_256085

theorem range_of_expression (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, y = x^2 + 2*b*x + 1 → y ≠ 2*a*(x + b)) →
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) →
    1 < (a - Real.cos θ)^2 + (b - Real.sin θ)^2 ∧
    (a - Real.cos θ)^2 + (b - Real.sin θ)^2 < 4 := by
  sorry


end NUMINAMATH_CALUDE_range_of_expression_l2560_256085


namespace NUMINAMATH_CALUDE_compound_animals_l2560_256019

theorem compound_animals (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = dogs - dogs / 5 →
  frogs = 2 * dogs →
  cats + dogs + frogs = 304 →
  frogs = 160 := by sorry

end NUMINAMATH_CALUDE_compound_animals_l2560_256019


namespace NUMINAMATH_CALUDE_S_value_l2560_256069

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_value : S = 2 * Real.sqrt 23 - 2 := by sorry

end NUMINAMATH_CALUDE_S_value_l2560_256069


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2560_256063

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2560_256063


namespace NUMINAMATH_CALUDE_merchant_loss_l2560_256030

/-- The total loss incurred by a merchant on a counterfeit transaction -/
def total_loss (purchase_cost : ℕ) (additional_price : ℕ) : ℕ :=
  purchase_cost + additional_price

/-- Theorem stating that under the given conditions, the total loss is 92 yuan -/
theorem merchant_loss :
  let purchase_cost : ℕ := 80
  let additional_price : ℕ := 12
  total_loss purchase_cost additional_price = 92 := by
  sorry

#check merchant_loss

end NUMINAMATH_CALUDE_merchant_loss_l2560_256030


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l2560_256055

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l2560_256055


namespace NUMINAMATH_CALUDE_average_weight_increase_l2560_256004

/-- Proves that replacing a person weighing 68 kg with a person weighing 95.5 kg
    in a group of 5 people increases the average weight by 5.5 kg -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 5 * initial_average
  let new_total := initial_total - 68 + 95.5
  let new_average := new_total / 5
  new_average - initial_average = 5.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2560_256004


namespace NUMINAMATH_CALUDE_min_days_to_plant_100_trees_l2560_256010

def trees_planted (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem min_days_to_plant_100_trees :
  (∃ n : ℕ, trees_planted n ≥ 100) ∧
  (∀ n : ℕ, trees_planted n ≥ 100 → n ≥ 6) ∧
  trees_planted 6 ≥ 100 :=
sorry

end NUMINAMATH_CALUDE_min_days_to_plant_100_trees_l2560_256010


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2560_256017

/-- The probability of finding treasure and no traps on a single island -/
def p_treasure : ℚ := 1/5

/-- The probability of finding traps and no treasure on a single island -/
def p_traps : ℚ := 1/10

/-- The probability of finding neither treasure nor traps on a single island -/
def p_neither : ℚ := 7/10

/-- The total number of islands -/
def total_islands : ℕ := 8

/-- The number of islands with treasure we want to find -/
def treasure_islands : ℕ := 4

theorem pirate_treasure_probability :
  (Nat.choose total_islands treasure_islands : ℚ) *
  p_treasure ^ treasure_islands *
  p_neither ^ (total_islands - treasure_islands) =
  33614 / 1250000 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_probability_l2560_256017


namespace NUMINAMATH_CALUDE_insect_legs_count_l2560_256074

theorem insect_legs_count (num_insects : ℕ) (legs_per_insect : ℕ) : 
  num_insects = 5 → legs_per_insect = 6 → num_insects * legs_per_insect = 30 := by
  sorry

end NUMINAMATH_CALUDE_insect_legs_count_l2560_256074


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2560_256082

theorem sphere_surface_area_ratio (r1 r2 : ℝ) (h1 : r1 = 40) (h2 : r2 = 10) :
  (4 * π * r1^2) / (4 * π * r2^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2560_256082


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2560_256024

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The center of a circle passing through three given points -/
def circleCenterThroughThreePoints (A B C : Point) : Point :=
  sorry

/-- The three given points -/
def A : Point := ⟨2, 2⟩
def B : Point := ⟨6, 2⟩
def C : Point := ⟨4, 5⟩

/-- Theorem stating that the center of the circle passing through A, B, and C is (4, 17/6) -/
theorem circle_center_coordinates : 
  let center := circleCenterThroughThreePoints A B C
  center.x = 4 ∧ center.y = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2560_256024


namespace NUMINAMATH_CALUDE_special_triangle_sides_l2560_256006

/-- A triangle with sides a, b, and c satisfying specific conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = 18
  sum_eq_double_c : a + b = 2 * c
  b_eq_double_a : b = 2 * a

/-- Theorem stating the unique side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 4 ∧ t.b = 8 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l2560_256006


namespace NUMINAMATH_CALUDE_y_values_l2560_256078

theorem y_values (x : ℝ) (h : x^2 + 6 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 135 / 7 ∨ y = 216 / 13 := by sorry

end NUMINAMATH_CALUDE_y_values_l2560_256078


namespace NUMINAMATH_CALUDE_max_x_value_l2560_256072

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |2*x - a|

-- State the theorem
theorem max_x_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ 1/m + 4/n) →
  (∃ x : ℝ, ∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) ∧
  (∀ x : ℝ, (∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) → |x| ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_max_x_value_l2560_256072


namespace NUMINAMATH_CALUDE_cistern_leak_time_l2560_256011

/-- Proves that if a cistern can be filled by pipe A in 16 hours and both pipes A and B together fill the cistern in 80.00000000000001 hours, then pipe B alone can leak out the full cistern in 80 hours. -/
theorem cistern_leak_time (fill_time_A : ℝ) (fill_time_both : ℝ) (leak_time_B : ℝ) : 
  fill_time_A = 16 →
  fill_time_both = 80.00000000000001 →
  (1 / fill_time_A) - (1 / leak_time_B) = 1 / fill_time_both →
  leak_time_B = 80 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l2560_256011


namespace NUMINAMATH_CALUDE_tv_sets_b_is_30_l2560_256005

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop a -/
def tv_sets_a : ℕ := 20

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: The number of TV sets in shop b is 30 -/
theorem tv_sets_b_is_30 : 
  num_shops * average_tv_sets - (tv_sets_a + tv_sets_c + tv_sets_d + tv_sets_e) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tv_sets_b_is_30_l2560_256005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l2560_256012

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subset : (a 2 / a 1) = (a 5 / a 2)

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_sequence_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l2560_256012


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2560_256036

theorem lcm_factor_proof (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ A ≥ B ∧ Nat.gcd A B = 30 ∧ A = 450 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ Nat.lcm A B = 30 * x * y ∧ (x = 15 ∨ y = 15) :=
by
  sorry

#check lcm_factor_proof

end NUMINAMATH_CALUDE_lcm_factor_proof_l2560_256036


namespace NUMINAMATH_CALUDE_bombardier_solution_l2560_256056

/-- Represents the number of bombs thrown by each bombardier -/
structure BombardierShots where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of the bombardier problem -/
def satisfiesConditions (shots : BombardierShots) : Prop :=
  (shots.first + shots.second = shots.third + 26) ∧
  (shots.second + shots.third = shots.first + shots.second + 38) ∧
  (shots.first + shots.third = shots.second + 24)

/-- Theorem stating the solution to the bombardier problem -/
theorem bombardier_solution :
  ∃ (shots : BombardierShots), satisfiesConditions shots ∧
    shots.first = 25 ∧ shots.second = 64 ∧ shots.third = 63 := by
  sorry

end NUMINAMATH_CALUDE_bombardier_solution_l2560_256056


namespace NUMINAMATH_CALUDE_first_digit_base_9_l2560_256034

def base_3_digits : List Nat := [2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1]

def y : Nat := (List.reverse base_3_digits).enum.foldl (fun acc (i, digit) => acc + digit * (3 ^ i)) 0

theorem first_digit_base_9 : ∃ (k : Nat), 4 * (9 ^ k) ≤ y ∧ y < 5 * (9 ^ k) ∧ (∀ m, m > k → y < 4 * (9 ^ m)) :=
  sorry

end NUMINAMATH_CALUDE_first_digit_base_9_l2560_256034


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2560_256046

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2560_256046


namespace NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l2560_256050

def isPeriodicSequence (x : ℕ → ℤ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

def satisfiesRecurrence (x : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → x (n + 1) = 2 * x n + 3 * x (n - 1)

def isConstant (x : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, x m = x n

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℕ → ℤ, satisfiesRecurrence x ∧ isPeriodicSequence x ∧ ¬isConstant x := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l2560_256050


namespace NUMINAMATH_CALUDE_greatest_fourth_term_of_arithmetic_sequence_l2560_256013

theorem greatest_fourth_term_of_arithmetic_sequence 
  (a : ℕ) 
  (d : ℕ) 
  (sum_eq_65 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65) 
  (a_positive : a > 0) :
  ∀ (b : ℕ) (e : ℕ), 
    b > 0 → 
    b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 → 
    b + 3*e ≤ a + 3*d :=
by sorry

end NUMINAMATH_CALUDE_greatest_fourth_term_of_arithmetic_sequence_l2560_256013


namespace NUMINAMATH_CALUDE_Q_representation_exists_zero_polynomial_l2560_256033

variable (x₁ x₂ x₃ x₄ : ℝ)

def Q (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

def P₁ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ + x₂ - x₃ - x₄
def P₂ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + x₃ - x₄
def P₃ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ - x₃ + x₄
def P₄ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 0

theorem Q_representation (x₁ x₂ x₃ x₄ : ℝ) :
  Q x₁ x₂ x₃ x₄ = (P₁ x₁ x₂ x₃ x₄)^2 + (P₂ x₁ x₂ x₃ x₄)^2 + (P₃ x₁ x₂ x₃ x₄)^2 + (P₄ x₁ x₂ x₃ x₄)^2 :=
sorry

theorem exists_zero_polynomial (f g h k : ℝ → ℝ → ℝ → ℝ → ℝ) 
  (hQ : ∀ x₁ x₂ x₃ x₄, Q x₁ x₂ x₃ x₄ = (f x₁ x₂ x₃ x₄)^2 + (g x₁ x₂ x₃ x₄)^2 + (h x₁ x₂ x₃ x₄)^2 + (k x₁ x₂ x₃ x₄)^2) :
  (f = λ _ _ _ _ => 0) ∨ (g = λ _ _ _ _ => 0) ∨ (h = λ _ _ _ _ => 0) ∨ (k = λ _ _ _ _ => 0) :=
sorry

end NUMINAMATH_CALUDE_Q_representation_exists_zero_polynomial_l2560_256033


namespace NUMINAMATH_CALUDE_equation_describes_parabola_l2560_256003

-- Define the equation
def equation (x y : ℝ) : Prop := |y + 5| = Real.sqrt ((x - 2)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, eq x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem equation_describes_parabola : describes_parabola equation := by sorry

end NUMINAMATH_CALUDE_equation_describes_parabola_l2560_256003


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2560_256018

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 6*x₁ + 5 = 2*x₁ - 8) ∧ 
                (x₂^2 - 6*x₂ + 5 = 2*x₂ - 8) ∧ 
                (x₁ + x₂ = 8)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2560_256018


namespace NUMINAMATH_CALUDE_cubic_equation_no_negative_roots_l2560_256001

theorem cubic_equation_no_negative_roots :
  ∀ x : ℝ, x < 0 → x^3 - 9*x^2 + 23*x - 15 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_no_negative_roots_l2560_256001


namespace NUMINAMATH_CALUDE_mollys_age_l2560_256048

theorem mollys_age (sandy_age molly_age : ℕ) : 
  sandy_age = 42 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age = 54 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l2560_256048


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l2560_256091

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z + 2)^2 - 8*I = c * I) → 
  z = -2*I :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l2560_256091


namespace NUMINAMATH_CALUDE_quadratic_rotate_translate_l2560_256081

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Rotation of a function by 180 degrees around the origin -/
def Rotate180 (f : ℝ → ℝ) : ℝ → ℝ := fun x ↦ -f x

/-- Translation of a function upwards by d units -/
def TranslateUp (f : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ f x + d

/-- The theorem stating the result of rotating a quadratic function 180 degrees
    around the origin and then translating it upwards -/
theorem quadratic_rotate_translate (a b c d : ℝ) :
  (TranslateUp (Rotate180 (QuadraticFunction a b c)) d) =
  QuadraticFunction (-a) (-b) (-c + d) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rotate_translate_l2560_256081


namespace NUMINAMATH_CALUDE_sqrt_product_equals_240_l2560_256000

theorem sqrt_product_equals_240 : Real.sqrt 128 * Real.sqrt 50 * (27 ^ (1/3 : ℝ)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_240_l2560_256000


namespace NUMINAMATH_CALUDE_complex_division_result_l2560_256008

theorem complex_division_result : (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2560_256008


namespace NUMINAMATH_CALUDE_consecutive_sum_equals_fourteen_l2560_256077

theorem consecutive_sum_equals_fourteen (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 14 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_equals_fourteen_l2560_256077


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l2560_256022

theorem zoo_animal_difference (parrots : ℕ) (snakes : ℕ) (monkeys : ℕ) (elephants : ℕ) (zebras : ℕ) : 
  parrots = 8 → 
  snakes = 3 * parrots → 
  monkeys = 2 * snakes → 
  elephants = (parrots + snakes) / 2 → 
  zebras = elephants - 3 → 
  monkeys - zebras = 35 := by
sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l2560_256022


namespace NUMINAMATH_CALUDE_students_using_green_l2560_256080

theorem students_using_green (total : ℕ) (both : ℕ) (red : ℕ) : 
  total = 70 → both = 38 → red = 56 → 
  total = (total - both) + red → 
  (total - both) = 52 := by sorry

end NUMINAMATH_CALUDE_students_using_green_l2560_256080


namespace NUMINAMATH_CALUDE_equal_expressions_l2560_256002

theorem equal_expressions (x y z : ℤ) :
  x + 2 * y * z = (x + y) * (x + 2 * z) ↔ x + y + 2 * z = 1 ∨ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l2560_256002


namespace NUMINAMATH_CALUDE_math_competition_nonparticipants_l2560_256075

theorem math_competition_nonparticipants (total_students : ℕ) 
  (h1 : total_students = 39) 
  (h2 : ∃ participants : ℕ, participants = total_students / 3) : 
  ∃ nonparticipants : ℕ, nonparticipants = 26 ∧ nonparticipants = total_students - (total_students / 3) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_nonparticipants_l2560_256075


namespace NUMINAMATH_CALUDE_special_ellipse_major_twice_minor_l2560_256083

/-- An ellipse where one focus and two vertices form an equilateral triangle -/
structure SpecialEllipse where
  -- Major axis length
  a : ℝ
  -- Minor axis length
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Constraint that one focus and two vertices form an equilateral triangle
  equilateral_triangle : c = a / 2
  -- Standard ellipse equation
  ellipse_equation : a^2 = b^2 + c^2

/-- The major axis is twice the minor axis in a special ellipse -/
theorem special_ellipse_major_twice_minor (e : SpecialEllipse) : e.a = 2 * e.b := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_major_twice_minor_l2560_256083


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2560_256062

def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2560_256062


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l2560_256029

-- Define the circle
def Circle (O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = 1}

-- Define the ellipse
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem min_dot_product_on_ellipse (O A B P : ℝ × ℝ) :
  O = (0, 0) →  -- Assume the circle is centered at the origin
  A ∈ Circle O →
  B ∈ Circle O →
  (A.1 = -B.1 ∧ A.2 = -B.2) →  -- AB is a diameter
  P ∈ Ellipse →
  ∀ Q ∈ Ellipse, dotProduct (Q.1 - A.1, Q.2 - A.2) (Q.1 - B.1, Q.2 - B.2) ≥ 2 ∧
  ∃ R ∈ Ellipse, dotProduct (R.1 - A.1, R.2 - A.2) (R.1 - B.1, R.2 - B.2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l2560_256029


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2560_256094

theorem trigonometric_identity :
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) +
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) +
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2560_256094


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2560_256043

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2560_256043


namespace NUMINAMATH_CALUDE_function_difference_implies_a_range_l2560_256014

open Real

theorem function_difference_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → 
    (a * log x₁ + x₁^2) - (a * log x₂ + x₂^2) > 2) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_function_difference_implies_a_range_l2560_256014


namespace NUMINAMATH_CALUDE_three_card_draw_probability_l2560_256079

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (spades : Nat)
  (hearts : Nat)
  (diamonds : Nat)

/-- Calculates the probability of drawing a specific card from the deck -/
def drawProbability (deck : Deck) (targetCards : Nat) : Rat :=
  targetCards / deck.total

/-- Theorem: Probability of drawing a spade, then a heart, then a diamond -/
theorem three_card_draw_probability (deck : Deck) 
  (h1 : deck.total = 52)
  (h2 : deck.spades = 13)
  (h3 : deck.hearts = 13)
  (h4 : deck.diamonds = 13) :
  (drawProbability deck deck.spades) * 
  (drawProbability ⟨deck.total - 1, deck.spades - 1, deck.hearts, deck.diamonds⟩ deck.hearts) * 
  (drawProbability ⟨deck.total - 2, deck.spades - 1, deck.hearts - 1, deck.diamonds⟩ deck.diamonds) = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_draw_probability_l2560_256079


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l2560_256016

/-- Given a hyperbola with equation x² - y²/3 = 1, prove its focal length is 4 and eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (a b c : ℝ),
    (a = 1 ∧ b^2 = 3) ∧
    (c^2 = a^2 + b^2) ∧
    (2 * c = 4) ∧
    (c / a = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l2560_256016


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l2560_256057

theorem modulus_of_complex_expression : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l2560_256057


namespace NUMINAMATH_CALUDE_simplify_fraction_l2560_256028

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2560_256028


namespace NUMINAMATH_CALUDE_expression_evaluation_l2560_256066

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  (x - 2*y)^2 - (x - 3*y)*(x + 3*y) - 4*y^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2560_256066


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l2560_256059

/-- A line in a 3D space --/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it empty

/-- Two lines are parallel --/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines form equal angles with a third line --/
def equal_angles (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- A line is perpendicular to another line --/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Main theorem: Exactly two of the given propositions about parallel lines are false --/
theorem parallel_lines_theorem :
  ∃ (prop1 prop2 prop3 : Prop),
    prop1 = (∀ l1 l2 l3 : Line3D, equal_angles l1 l2 l3 → parallel l1 l2) ∧
    prop2 = (∀ l1 l2 l3 : Line3D, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2) ∧
    prop3 = (∀ l1 l2 l3 : Line3D, parallel l1 l3 → parallel l2 l3 → parallel l1 l2) ∧
    (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l2560_256059


namespace NUMINAMATH_CALUDE_num_friends_is_four_l2560_256090

/-- The number of friends who volunteered with James to plant flowers -/
def num_friends : ℕ :=
  let total_flowers : ℕ := 200
  let days : ℕ := 2
  let james_flowers_per_day : ℕ := 20
  (total_flowers - james_flowers_per_day * days) / (james_flowers_per_day * days)

theorem num_friends_is_four : num_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_friends_is_four_l2560_256090


namespace NUMINAMATH_CALUDE_second_cube_volume_is_64_l2560_256049

-- Define the volume of the first cube
def first_cube_volume : ℝ := 8

-- Define the relationship between the surface areas of the two cubes
def surface_area_ratio : ℝ := 4

-- Theorem statement
theorem second_cube_volume_is_64 :
  let first_side := first_cube_volume ^ (1/3 : ℝ)
  let first_surface_area := 6 * first_side^2
  let second_surface_area := surface_area_ratio * first_surface_area
  let second_side := (second_surface_area / 6) ^ (1/2 : ℝ)
  second_side^3 = 64 := by sorry

end NUMINAMATH_CALUDE_second_cube_volume_is_64_l2560_256049


namespace NUMINAMATH_CALUDE_sum_of_cube_edges_l2560_256087

-- Define a cube with edge length 15
def cube_edge_length : ℝ := 15

-- Define the number of edges in a cube
def cube_num_edges : ℕ := 12

-- Theorem: The sum of all edge lengths in the cube is 180
theorem sum_of_cube_edges :
  cube_edge_length * cube_num_edges = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_edges_l2560_256087


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l2560_256052

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if a tuple represents a valid three-digit number -/
def isValidThreeDigitNumber (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- Converts a three-digit number tuple to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  let (a, b, c) := n
  100 * a + 10 * b + c

/-- Generates all permutations of a three-digit number -/
def permutations (n : ThreeDigitNumber) : List ThreeDigitNumber :=
  let (a, b, c) := n
  [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]

/-- Calculates the average of the permutations of a three-digit number -/
def averageOfPermutations (n : ThreeDigitNumber) : Nat :=
  (List.sum (List.map toNumber (permutations n))) / 6

/-- Checks if a three-digit number satisfies the given condition -/
def satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  isValidThreeDigitNumber n ∧ averageOfPermutations n = toNumber n

/-- The set of three-digit numbers that satisfy the condition -/
def solutionSet : Set Nat :=
  {370, 407, 481, 518, 592, 629}

/-- The main theorem to be proved -/
theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  satisfiesCondition n ↔ toNumber n ∈ solutionSet := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l2560_256052


namespace NUMINAMATH_CALUDE_game_ends_in_53_rounds_l2560_256032

/-- Represents the state of the game at any given round -/
structure GameState :=
  (A B C D : ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  ⟨16, 15, 14, 13⟩

/-- Function to update the game state after one round -/
def update_state (state : GameState) : GameState :=
  sorry

/-- Predicate to check if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.A = 0 ∨ state.B = 0 ∨ state.C = 0 ∨ state.D = 0

/-- The number of rounds the game lasts -/
def game_duration : ℕ := 53

theorem game_ends_in_53_rounds :
  ∃ (final_state : GameState),
    (game_duration.iterate update_state initial_state = final_state) ∧
    game_ended final_state ∧
    ∀ (n : ℕ), n < game_duration →
      ¬game_ended (n.iterate update_state initial_state) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_53_rounds_l2560_256032


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2560_256099

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2560_256099


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2560_256009

theorem equilateral_triangle_side_length 
  (circular_radius : ℝ) 
  (circular_speed : ℝ) 
  (triangular_speed : ℝ) 
  (h1 : circular_radius = 60) 
  (h2 : circular_speed = 6) 
  (h3 : triangular_speed = 5) :
  ∃ x : ℝ, 
    (3 * x = triangular_speed * ((2 * Real.pi * circular_radius) / circular_speed)) ∧ 
    x = 100 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2560_256009


namespace NUMINAMATH_CALUDE_chantel_final_bracelet_count_l2560_256039

/-- The number of bracelets Chantel has at the end -/
def final_bracelet_count : ℕ :=
  let first_week_production := 7 * 4
  let after_first_giveaway := first_week_production - 8
  let second_period_production := 10 * 5
  let before_second_giveaway := after_first_giveaway + second_period_production
  before_second_giveaway - 12

/-- Theorem stating that Chantel ends up with 58 bracelets -/
theorem chantel_final_bracelet_count : final_bracelet_count = 58 := by
  sorry

end NUMINAMATH_CALUDE_chantel_final_bracelet_count_l2560_256039


namespace NUMINAMATH_CALUDE_quadratic_roots_sign_l2560_256092

theorem quadratic_roots_sign (a b c : ℝ) :
  (a * c < 0) ↔ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sign_l2560_256092


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l2560_256095

-- Define the conditions
def α (x : ℝ) : Prop := x^2 = 4
def β (x : ℝ) : Prop := x = 2

-- State the theorem
theorem alpha_necessary_not_sufficient :
  (∀ x, β x → α x) ∧ (∃ x, α x ∧ ¬β x) := by sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l2560_256095


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2560_256088

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line l passing through a vertex (0, b) and a focus (c, 0),
    if the distance from the center to l is b/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1 / Real.sqrt ((1 / c^2) + (1 / b^2)) = b / 4) →
  c / a = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2560_256088


namespace NUMINAMATH_CALUDE_soldier_rearrangement_l2560_256061

theorem soldier_rearrangement (n : Nat) (h : n = 20 ∨ n = 21) :
  ∃ (d : ℝ), d = 10 * Real.sqrt 2 ∧
  (∀ (rearrangement : Fin n × Fin n → Fin n × Fin n),
    (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
      Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                 (j.val - (rearrangement (i, j)).2.val)^2) ≥ d) →
    (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) ∧
  (∀ (d' : ℝ), d' > d →
    ¬∃ (rearrangement : Fin n × Fin n → Fin n × Fin n),
      (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
        Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                   (j.val - (rearrangement (i, j)).2.val)^2) ≥ d') ∧
      (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) :=
by sorry

end NUMINAMATH_CALUDE_soldier_rearrangement_l2560_256061


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l2560_256089

/-- Given two polynomials g and f, where g has three distinct roots that are also roots of f,
    prove that f(1) = -1333 -/
theorem polynomial_root_problem (a b c : ℝ) : 
  let g := fun x : ℝ => x^3 + a*x^2 + x + 8
  let f := fun x : ℝ => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ x : ℝ, g x = 0 → f x = 0) →
  f 1 = -1333 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l2560_256089


namespace NUMINAMATH_CALUDE_prob_three_ones_in_four_rolls_eq_5_324_l2560_256097

/-- A fair, regular six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a fair die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of rolling a 1 exactly three times in four rolls of a fair die -/
def prob_three_ones_in_four_rolls : ℚ :=
  (choose 4 3 : ℚ) * (prob {0})^3 * (1 - prob {0})

theorem prob_three_ones_in_four_rolls_eq_5_324 :
  prob_three_ones_in_four_rolls = 5 / 324 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_ones_in_four_rolls_eq_5_324_l2560_256097


namespace NUMINAMATH_CALUDE_factorization_equality_minimum_value_expression_minimum_value_at_one_l2560_256040

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 := by sorry

-- Problem 2
theorem minimum_value_expression (n : ℝ) :
  (n^2 - 2*n - 3) * (n^2 - 2*n + 5) + 17 ≥ 1 := by sorry

theorem minimum_value_at_one :
  (1^2 - 2*1 - 3) * (1^2 - 2*1 + 5) + 17 = 1 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_minimum_value_expression_minimum_value_at_one_l2560_256040


namespace NUMINAMATH_CALUDE_quadratic_vertex_condition_l2560_256086

theorem quadratic_vertex_condition (a b c x₀ y₀ : ℝ) (h_a : a ≠ 0) :
  (∀ m n : ℝ, n = a * m^2 + b * m + c → a * (y₀ - n) ≤ 0) →
  y₀ = a * x₀^2 + b * x₀ + c →
  2 * a * x₀ + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_condition_l2560_256086


namespace NUMINAMATH_CALUDE_job_selection_probability_l2560_256071

theorem job_selection_probability 
  (jamie_prob : ℚ) 
  (tom_prob : ℚ) 
  (h1 : jamie_prob = 2/3) 
  (h2 : tom_prob = 5/7) : 
  jamie_prob * tom_prob = 10/21 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l2560_256071


namespace NUMINAMATH_CALUDE_garlic_cloves_remaining_l2560_256067

theorem garlic_cloves_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 237 → used = 184 → remaining = initial - used → remaining = 53 := by
sorry

end NUMINAMATH_CALUDE_garlic_cloves_remaining_l2560_256067


namespace NUMINAMATH_CALUDE_sum_of_coordinates_equals_eight_l2560_256031

def point_C : ℝ × ℝ := (3, 4)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def point_D : ℝ × ℝ := reflect_over_y_axis point_C

theorem sum_of_coordinates_equals_eight :
  point_C.1 + point_C.2 + point_D.1 + point_D.2 = 8 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_equals_eight_l2560_256031


namespace NUMINAMATH_CALUDE_distance_to_chord_equals_half_chord_l2560_256041

-- Define the circle and points
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the function to calculate distance from a point to a line segment
def distancePointToSegment (p : Point) (a b : Point) : ℝ := sorry

-- Define the theorem
theorem distance_to_chord_equals_half_chord (O A B C D E : Point) (circle : Circle) :
  O = circle.center →
  distance A E = 2 * circle.radius →
  (∀ p ∈ [A, B, C, E], distance O p = circle.radius) →
  distancePointToSegment O A B = (distance C D) / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_chord_equals_half_chord_l2560_256041


namespace NUMINAMATH_CALUDE_four_even_cards_different_suits_count_l2560_256007

/-- Represents a standard playing card suit -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Represents an even-numbered card (including face cards) -/
inductive EvenCard
| two
| four
| six
| eight
| ten
| queen

/-- The number of suits in a standard deck -/
def number_of_suits : Nat := 4

/-- The number of even-numbered cards in each suit -/
def even_cards_per_suit : Nat := 6

/-- A function to calculate the number of ways to choose 4 cards from a standard deck
    under the given conditions -/
def choose_four_even_cards_different_suits : Nat :=
  number_of_suits * even_cards_per_suit ^ 4

/-- The theorem stating that the number of ways to choose 4 cards from a standard deck,
    where all four cards are of different suits, each card is even-numbered,
    and the order doesn't matter, is equal to 1296 -/
theorem four_even_cards_different_suits_count :
  choose_four_even_cards_different_suits = 1296 := by
  sorry


end NUMINAMATH_CALUDE_four_even_cards_different_suits_count_l2560_256007


namespace NUMINAMATH_CALUDE_black_lambs_count_l2560_256025

theorem black_lambs_count (total : ℕ) (white : ℕ) (h1 : total = 6048) (h2 : white = 193) :
  total - white = 5855 := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_l2560_256025
