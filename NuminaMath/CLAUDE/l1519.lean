import Mathlib

namespace NUMINAMATH_CALUDE_walter_chores_l1519_151900

/-- The number of days Walter worked -/
def total_days : ℕ := 10

/-- Walter's earnings for a regular day -/
def regular_pay : ℕ := 3

/-- Walter's earnings for an exceptional day -/
def exceptional_pay : ℕ := 5

/-- Walter's total earnings -/
def total_earnings : ℕ := 36

/-- The number of days Walter did chores exceptionally well -/
def exceptional_days : ℕ := 3

/-- The number of days Walter did regular chores -/
def regular_days : ℕ := total_days - exceptional_days

theorem walter_chores :
  regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
  regular_days + exceptional_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l1519_151900


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1519_151981

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₄ = 60 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1519_151981


namespace NUMINAMATH_CALUDE_plastic_bottles_count_l1519_151907

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The total weight of the second scenario in grams -/
def total_weight : ℕ := 1050

/-- The number of glass bottles in the second scenario -/
def num_glass_bottles : ℕ := 4

theorem plastic_bottles_count :
  ∃ (x : ℕ), 
    3 * glass_bottle_weight = 600 ∧
    glass_bottle_weight = plastic_bottle_weight + 150 ∧
    4 * glass_bottle_weight + x * plastic_bottle_weight = total_weight ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plastic_bottles_count_l1519_151907


namespace NUMINAMATH_CALUDE_daughter_least_intelligent_l1519_151949

-- Define the types for people and intelligence levels
inductive Person : Type
| Father : Person
| Sister : Person
| Son : Person
| Daughter : Person

inductive IntelligenceLevel : Type
| Least : IntelligenceLevel
| Smartest : IntelligenceLevel

-- Define the properties
def isTwin (p1 p2 : Person) : Prop := sorry

def sex (p : Person) : Bool := sorry

def age (p : Person) : ℕ := sorry

def intelligenceLevel (p : Person) : IntelligenceLevel := sorry

-- Define the theorem
theorem daughter_least_intelligent 
  (h1 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        (∃ p3 : Person, isTwin p1 p3 ∧ sex p3 ≠ sex p2))
  (h2 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        age p1 = age p2)
  : intelligenceLevel Person.Daughter = IntelligenceLevel.Least := by
  sorry

end NUMINAMATH_CALUDE_daughter_least_intelligent_l1519_151949


namespace NUMINAMATH_CALUDE_sqrt_equality_problem_l1519_151920

theorem sqrt_equality_problem : ∃ (a x : ℝ), 
  x > 0 ∧ 
  Real.sqrt x = 2 * a - 3 ∧ 
  Real.sqrt x = 5 - a ∧ 
  a = -2 ∧ 
  x = 49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_problem_l1519_151920


namespace NUMINAMATH_CALUDE_choose_captains_l1519_151969

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_captains_l1519_151969


namespace NUMINAMATH_CALUDE_evans_earnings_l1519_151909

/-- Proves that Evan earned $21 given the conditions of the problem -/
theorem evans_earnings (markese_earnings : ℕ) (total_earnings : ℕ) (earnings_difference : ℕ)
  (h1 : markese_earnings = 16)
  (h2 : total_earnings = 37)
  (h3 : markese_earnings + earnings_difference = total_earnings)
  (h4 : earnings_difference = 5) : 
  total_earnings - markese_earnings = 21 := by
  sorry

end NUMINAMATH_CALUDE_evans_earnings_l1519_151909


namespace NUMINAMATH_CALUDE_function_is_negation_l1519_151927

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x

/-- The main theorem stating that g(x) = -x for all x -/
theorem function_is_negation (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -x :=
sorry

end NUMINAMATH_CALUDE_function_is_negation_l1519_151927


namespace NUMINAMATH_CALUDE_oil_leaked_before_work_l1519_151945

def total_oil_leaked : ℕ := 11687
def oil_leaked_during_work : ℕ := 5165

theorem oil_leaked_before_work (total : ℕ) (during_work : ℕ) 
  (h1 : total = total_oil_leaked) 
  (h2 : during_work = oil_leaked_during_work) : 
  total - during_work = 6522 := by
  sorry

end NUMINAMATH_CALUDE_oil_leaked_before_work_l1519_151945


namespace NUMINAMATH_CALUDE_parallel_planes_sum_l1519_151937

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_sum (x y : ℝ) : 
  let n₁ : Fin 3 → ℝ := ![x, 1, -2]
  let n₂ : Fin 3 → ℝ := ![-1, y, 1/2]
  (∃ (k : ℝ), ∀ i, n₁ i = k * n₂ i) →
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_sum_l1519_151937


namespace NUMINAMATH_CALUDE_minimum_value_problem_l1519_151999

theorem minimum_value_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧
  (∀ (c d : ℝ), c ≠ 0 → d ≠ 0 →
    c^2 + d^2 + 2 / c^2 + d / c + 1 / d^2 ≥ x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2) ∧
  x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2 = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l1519_151999


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1519_151946

/-- Given a hyperbola C: (y^2 / a^2) - (x^2 / b^2) = 1 with a > 0 and b > 0,
    whose asymptotes intersect with the circle x^2 + (y - 2)^2 = 1,
    the eccentricity e of C satisfies 1 < e < 2√3/3. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | a * x = b * y ∨ a * x = -b * y}
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - 2)^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p ∈ asymptotes, p ∈ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1519_151946


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1519_151972

theorem inequality_system_solution_range (k : ℝ) : 
  (∃! x : ℤ, (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k + 7)*x + 7*k < 0)) ↔ 
  (k ∈ Set.Icc (-5 : ℝ) 3 ∪ Set.Ioc 4 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1519_151972


namespace NUMINAMATH_CALUDE_merged_class_size_and_rank_l1519_151906

/-- Represents a group of students with known positions from left and right -/
structure StudentGroup where
  leftPos : Nat
  rightPos : Nat

/-- Calculates the total number of students in a group -/
def groupSize (g : StudentGroup) : Nat :=
  g.leftPos + g.rightPos - 1

theorem merged_class_size_and_rank (groupA groupB groupC : StudentGroup)
  (hA : groupA = ⟨8, 13⟩)
  (hB : groupB = ⟨12, 10⟩)
  (hC : groupC = ⟨7, 6⟩) :
  let totalStudents := groupSize groupA + groupSize groupB + groupSize groupC
  let rankFromLeft := groupSize groupA + groupB.leftPos
  totalStudents = 53 ∧ rankFromLeft = 32 := by
  sorry

end NUMINAMATH_CALUDE_merged_class_size_and_rank_l1519_151906


namespace NUMINAMATH_CALUDE_constant_ratio_sum_theorem_l1519_151980

theorem constant_ratio_sum_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_constant_ratio : ∃ k : ℝ, 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) :
  (∃ k : ℝ, k = -1 ∧ 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) ∧
  x₁ + x₂ + x₃ + x₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_sum_theorem_l1519_151980


namespace NUMINAMATH_CALUDE_cars_produced_in_europe_l1519_151988

def cars_north_america : ℕ := 3884
def total_cars : ℕ := 6755

theorem cars_produced_in_europe : 
  total_cars - cars_north_america = 2871 := by sorry

end NUMINAMATH_CALUDE_cars_produced_in_europe_l1519_151988


namespace NUMINAMATH_CALUDE_rowing_time_with_current_l1519_151992

/-- The time to cover a distance with, against, and without current -/
structure RowingTimes where
  with_current : ℚ
  against_current : ℚ
  no_current : ℚ

/-- The conditions of the rowing problem -/
def rowing_conditions (t : RowingTimes) : Prop :=
  t.against_current = 60 / 7 ∧ t.no_current = t.with_current - 7

/-- The theorem stating the time to cover the distance with the current -/
theorem rowing_time_with_current (t : RowingTimes) :
  rowing_conditions t → t.with_current = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_with_current_l1519_151992


namespace NUMINAMATH_CALUDE_prob_not_adjacent_seven_chairs_l1519_151903

/-- The number of chairs in the row -/
def n : ℕ := 7

/-- The number of ways two people can sit next to each other in a row of n chairs -/
def adjacent_seating (n : ℕ) : ℕ := n - 1

/-- The total number of ways two people can choose seats from n chairs -/
def total_seating (n : ℕ) : ℕ := n.choose 2

/-- The probability that Mary and James don't sit next to each other
    when randomly choosing seats in a row of n chairs -/
def prob_not_adjacent (n : ℕ) : ℚ :=
  1 - (adjacent_seating n : ℚ) / (total_seating n : ℚ)

theorem prob_not_adjacent_seven_chairs :
  prob_not_adjacent n = 5/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_seven_chairs_l1519_151903


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l1519_151993

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
   (x - 1)^2 + (y - 1)^2 = 1 ∧ 
   ∀ (x' y' : ℝ), (m + 1) * x' + (n + 1) * y' - 2 = 0 → (x' - 1)^2 + (y' - 1)^2 ≥ 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l1519_151993


namespace NUMINAMATH_CALUDE_cos_25_minus_alpha_equals_one_third_l1519_151926

theorem cos_25_minus_alpha_equals_one_third 
  (h : Real.sin (65 * π / 180 + α) = 1 / 3) : 
  Real.cos (25 * π / 180 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_25_minus_alpha_equals_one_third_l1519_151926


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l1519_151987

theorem max_sum_squared_distances (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  (∃ (max_val : ℝ), max_val = 15 + 24 * (1.5 / Real.sqrt (1.5^2 + 1)) - 16 * (1 / Real.sqrt (1.5^2 + 1)) ∧
   ∀ (w : ℂ), Complex.abs (w - (3 - 3*I)) = 4 →
     Complex.abs (w - (2 + I))^2 + Complex.abs (w - (6 - 2*I))^2 ≤ max_val) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l1519_151987


namespace NUMINAMATH_CALUDE_probability_is_203_225_l1519_151960

/-- The probability that 3 boys born in June 1990 have different birthdays -/
def probability_different_birthdays : ℚ :=
  (30 : ℚ) * 29 * 28 / (30 * 30 * 30)

theorem probability_is_203_225 : probability_different_birthdays = 203 / 225 := by
  sorry

#eval probability_different_birthdays

end NUMINAMATH_CALUDE_probability_is_203_225_l1519_151960


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1519_151913

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 80 ∧ 
  G = 100 ∧ 
  H = I ∧ 
  J = 2 * H + 20 ∧ 
  F + G + H + I + J = 540 →
  max F (max G (max H (max I J))) = 190 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l1519_151913


namespace NUMINAMATH_CALUDE_magic_square_x_value_l1519_151950

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℤ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧ 
    (entries 0 j + entries 1 j + entries 2 j = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 2 + entries 1 1 + entries 2 0 = entries 0 0 + entries 0 1 + entries 0 2)

/-- The main theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.entries 0 0 = x)
  (h2 : ms.entries 0 1 = 21)
  (h3 : ms.entries 0 2 = 70)
  (h4 : ms.entries 1 0 = 7) :
  x = 133 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l1519_151950


namespace NUMINAMATH_CALUDE_sams_work_hours_sams_september_february_hours_l1519_151911

/-- Calculates the number of hours Sam worked from September to February -/
theorem sams_work_hours (earnings_mar_aug : ℝ) (hours_mar_aug : ℝ) (console_cost : ℝ) (car_repair_cost : ℝ) (remaining_hours : ℝ) : ℝ :=
  let hourly_rate := earnings_mar_aug / hours_mar_aug
  let remaining_earnings := console_cost - (earnings_mar_aug - car_repair_cost)
  let total_hours_needed := remaining_earnings / hourly_rate
  total_hours_needed - remaining_hours

/-- Proves that Sam worked 8 hours from September to February -/
theorem sams_september_february_hours : sams_work_hours 460 23 600 340 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sams_work_hours_sams_september_february_hours_l1519_151911


namespace NUMINAMATH_CALUDE_not_always_true_false_and_implies_true_or_l1519_151916

theorem not_always_true_false_and_implies_true_or : 
  ¬ ∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_always_true_false_and_implies_true_or_l1519_151916


namespace NUMINAMATH_CALUDE_automobile_finance_credit_l1519_151917

/-- The problem statement as a theorem --/
theorem automobile_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_company_fraction : ℝ) : 
  total_credit = 416.6666666666667 →
  auto_credit_percentage = 0.36 →
  finance_company_fraction = 0.5 →
  finance_company_fraction * (auto_credit_percentage * total_credit) = 75 := by
sorry

end NUMINAMATH_CALUDE_automobile_finance_credit_l1519_151917


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1519_151967

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ 2 → x ≠ -2 → x ≠ 0 → x = 1 →
  (3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1519_151967


namespace NUMINAMATH_CALUDE_expression_simplification_l1519_151958

theorem expression_simplification :
  (4 * 6) / (12 * 14) * (3 * 5 * 7 * 9) / (4 * 6 * 8) * 7 = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1519_151958


namespace NUMINAMATH_CALUDE_jessica_money_difference_l1519_151976

/-- Proves that Jessica has 90 dollars more than Rodney given the stated conditions. -/
theorem jessica_money_difference (jessica_money : ℕ) (lily_money : ℕ) (ian_money : ℕ) (rodney_money : ℕ) :
  jessica_money = 150 ∧
  jessica_money = lily_money + 30 ∧
  lily_money = 3 * ian_money ∧
  ian_money + 20 = rodney_money →
  jessica_money - rodney_money = 90 :=
by sorry

end NUMINAMATH_CALUDE_jessica_money_difference_l1519_151976


namespace NUMINAMATH_CALUDE_certain_number_proof_l1519_151998

theorem certain_number_proof : ∃ n : ℕ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1519_151998


namespace NUMINAMATH_CALUDE_max_x_plus_y_max_x_plus_y_achieved_l1519_151966

theorem max_x_plus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  x + y ≤ 1 / Real.sqrt 2 :=
by sorry

theorem max_x_plus_y_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 3 * (x^2 + y^2) = x - y ∧ x + y > 1 / Real.sqrt 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_x_plus_y_max_x_plus_y_achieved_l1519_151966


namespace NUMINAMATH_CALUDE_max_value_expression_l1519_151918

theorem max_value_expression : 
  ∃ (M : ℝ), M = 27 ∧ 
  ∀ (x y : ℝ), 
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) * 
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1519_151918


namespace NUMINAMATH_CALUDE_garden_topsoil_cost_is_112_l1519_151932

/-- The cost of topsoil for a rectangular garden -/
def garden_topsoil_cost (length width depth price_per_cubic_foot : ℝ) : ℝ :=
  length * width * depth * price_per_cubic_foot

/-- Theorem: The cost of topsoil for the given garden is $112 -/
theorem garden_topsoil_cost_is_112 :
  garden_topsoil_cost 8 4 0.5 7 = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_topsoil_cost_is_112_l1519_151932


namespace NUMINAMATH_CALUDE_min_value_of_function_l1519_151923

theorem min_value_of_function :
  ∀ x : ℝ, x^2 + 1 / (x^2 + 1) + 3 ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1519_151923


namespace NUMINAMATH_CALUDE_pet_store_cats_l1519_151971

theorem pet_store_cats (siamese : ℝ) (house : ℝ) (added : ℝ) : 
  siamese = 13.0 → house = 5.0 → added = 10.0 → 
  siamese + house + added = 28.0 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1519_151971


namespace NUMINAMATH_CALUDE_money_loses_exchange_value_valid_money_properties_l1519_151924

/-- Represents an individual on an island -/
structure Individual where
  name : String

/-- Represents money found on the island -/
structure Money where
  amount : ℕ

/-- Represents the state of being on a deserted island -/
structure DesertedIsland where
  inhabitants : List Individual

/-- Function to determine if money has value as a medium of exchange -/
def hasExchangeValue (island : DesertedIsland) (money : Money) : Prop :=
  island.inhabitants.length > 1

/-- Theorem stating that money loses its exchange value on a deserted island with only one inhabitant -/
theorem money_loses_exchange_value 
  (crusoe : Individual) 
  (island : DesertedIsland) 
  (money : Money) 
  (h1 : island.inhabitants = [crusoe]) : 
  ¬(hasExchangeValue island money) := by
  sorry

/-- Properties required for an item to be considered money -/
structure MoneyProperties where
  durability : Prop
  portability : Prop
  divisibility : Prop
  acceptability : Prop
  uniformity : Prop
  limitedSupply : Prop

/-- Function to determine if an item can be considered money -/
def isValidMoney (item : MoneyProperties) : Prop :=
  item.durability ∧ 
  item.portability ∧ 
  item.divisibility ∧ 
  item.acceptability ∧ 
  item.uniformity ∧ 
  item.limitedSupply

/-- Theorem stating that an item must possess all required properties to be considered valid money -/
theorem valid_money_properties (item : MoneyProperties) :
  isValidMoney item ↔ 
    (item.durability ∧ 
     item.portability ∧ 
     item.divisibility ∧ 
     item.acceptability ∧ 
     item.uniformity ∧ 
     item.limitedSupply) := by
  sorry

end NUMINAMATH_CALUDE_money_loses_exchange_value_valid_money_properties_l1519_151924


namespace NUMINAMATH_CALUDE_max_two_digit_times_max_one_digit_is_three_digit_l1519_151928

theorem max_two_digit_times_max_one_digit_is_three_digit : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 99 * 9 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_times_max_one_digit_is_three_digit_l1519_151928


namespace NUMINAMATH_CALUDE_sum_formula_l1519_151991

/-- Given a sequence {a_n}, S_n is the sum of the first n terms and satisfies S_n = 2a_n - 2^n -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  2 * a n - 2^n

/-- Theorem stating that S_n = n * 2^n -/
theorem sum_formula (a : ℕ → ℝ) (n : ℕ) : S a n = n * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_formula_l1519_151991


namespace NUMINAMATH_CALUDE_minimum_value_sum_reciprocals_l1519_151929

theorem minimum_value_sum_reciprocals (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_three : a + b + c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_sum_reciprocals_l1519_151929


namespace NUMINAMATH_CALUDE_smallest_positive_a_l1519_151901

theorem smallest_positive_a (a : ℝ) : 
  a > 0 ∧ 
  (⌊2016 * a⌋ : ℤ) - (⌈a⌉ : ℤ) + 1 = 2016 ∧ 
  ∀ b : ℝ, b > 0 → (⌊2016 * b⌋ : ℤ) - (⌈b⌉ : ℤ) + 1 = 2016 → a ≤ b → 
  a = 2017 / 2016 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l1519_151901


namespace NUMINAMATH_CALUDE_train_length_and_speed_l1519_151943

/-- Proves the length and speed of a train given its crossing times over two platforms. -/
theorem train_length_and_speed 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : platform1_length = 90) 
  (h2 : platform2_length = 120) 
  (h3 : time1 = 12) 
  (h4 : time2 = 15) 
  (h5 : train_speed * time1 = train_length + platform1_length) 
  (h6 : train_speed * time2 = train_length + platform2_length) : 
  train_length = 30 ∧ train_speed = 10 := by
  sorry

#check train_length_and_speed

end NUMINAMATH_CALUDE_train_length_and_speed_l1519_151943


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1519_151974

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 6 ≤ 8 ∧ x - 7 < 2 * (x - 3)}
  S = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1519_151974


namespace NUMINAMATH_CALUDE_smallest_satisfying_integer_l1519_151953

/-- Checks if at least one of three consecutive integers is divisible by a given number -/
def atLeastOneDivisible (n : ℕ) (d : ℕ) : Prop :=
  d ∣ n ∨ d ∣ (n + 1) ∨ d ∣ (n + 2)

/-- The main theorem stating that 98 is the smallest positive integer satisfying the given conditions -/
theorem smallest_satisfying_integer : ∀ k : ℕ, k < 98 →
  ¬(atLeastOneDivisible k (2^2) ∧
    atLeastOneDivisible k (3^2) ∧
    atLeastOneDivisible k (5^2) ∧
    atLeastOneDivisible k (7^2)) ∧
  (atLeastOneDivisible 98 (2^2) ∧
   atLeastOneDivisible 98 (3^2) ∧
   atLeastOneDivisible 98 (5^2) ∧
   atLeastOneDivisible 98 (7^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_integer_l1519_151953


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1519_151912

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h1 : m1 = Real.sqrt 52) (h2 : m2 = Real.sqrt 73) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 = c^2 ∧
  m1^2 = (2 * b^2 + 2 * c^2 - a^2) / 4 ∧
  m2^2 = (2 * a^2 + 2 * c^2 - b^2) / 4 ∧
  c = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1519_151912


namespace NUMINAMATH_CALUDE_hot_dog_stand_sales_time_l1519_151961

/-- 
Given a hot dog stand that sells 10 hot dogs per hour at $2 each,
prove that it takes 10 hours to reach $200 in sales.
-/
theorem hot_dog_stand_sales_time : 
  let hot_dogs_per_hour : ℕ := 10
  let price_per_hot_dog : ℚ := 2
  let sales_goal : ℚ := 200
  let sales_per_hour : ℚ := hot_dogs_per_hour * price_per_hot_dog
  let hours_needed : ℚ := sales_goal / sales_per_hour
  hours_needed = 10 := by sorry

end NUMINAMATH_CALUDE_hot_dog_stand_sales_time_l1519_151961


namespace NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l1519_151978

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem angle_not_sharing_terminal_side :
  ¬(same_terminal_side 680 (-750)) ∧
  (same_terminal_side 330 (-750)) ∧
  (same_terminal_side (-30) (-750)) ∧
  (same_terminal_side (-1110) (-750)) :=
sorry

end NUMINAMATH_CALUDE_angle_not_sharing_terminal_side_l1519_151978


namespace NUMINAMATH_CALUDE_correct_graph_representation_l1519_151959

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem (m n : Car) : Prop :=
  m.speed > 0 ∧
  n.speed = 2 * m.speed ∧
  m.distance = n.distance ∧
  m.distance = m.speed * m.time ∧
  n.distance = n.speed * n.time

/-- The theorem to prove -/
theorem correct_graph_representation (m n : Car) 
  (h : problem m n) : n.speed = 2 * m.speed ∧ n.time = m.time / 2 := by
  sorry


end NUMINAMATH_CALUDE_correct_graph_representation_l1519_151959


namespace NUMINAMATH_CALUDE_second_point_y_coordinate_l1519_151934

/-- Given two points on a line, prove the y-coordinate of the second point -/
theorem second_point_y_coordinate
  (m n k : ℝ)
  (h1 : m = 2 * n + 3)  -- First point (m, n) satisfies line equation
  (h2 : m + 2 = 2 * (n + k) + 3)  -- Second point (m + 2, n + k) satisfies line equation
  (h3 : k = 1)  -- Given condition
  : n + k = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_second_point_y_coordinate_l1519_151934


namespace NUMINAMATH_CALUDE_mothers_age_l1519_151910

theorem mothers_age (certain_age : ℕ) (mothers_age : ℕ) : 
  mothers_age = 3 * certain_age → 
  certain_age + mothers_age = 40 → 
  mothers_age = 30 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l1519_151910


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l1519_151938

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l1519_151938


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1519_151982

/-- For the function f(x) = x + sin(x) + 1, f(x) + f(-x) = 2 for all real x -/
theorem f_sum_symmetric (x : ℝ) : let f : ℝ → ℝ := λ x ↦ x + Real.sin x + 1
  f x + f (-x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1519_151982


namespace NUMINAMATH_CALUDE_de_morgans_laws_l1519_151925

universe u

theorem de_morgans_laws {U : Type u} (A B : Set U) :
  (Set.compl (A ∪ B) = Set.compl A ∩ Set.compl B) ∧
  (Set.compl (A ∩ B) = Set.compl A ∪ Set.compl B) := by
  sorry

end NUMINAMATH_CALUDE_de_morgans_laws_l1519_151925


namespace NUMINAMATH_CALUDE_range_of_x_l1519_151970

theorem range_of_x (x : ℝ) (h1 : 1 / x ≤ 4) (h2 : 1 / x ≥ -2) : x ≥ 1 / 4 ∨ x ≤ -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1519_151970


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1519_151902

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 4) :
  (1 / x + 1 / y) ≥ 1 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 4 ∧ 1 / x + 1 / y = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1519_151902


namespace NUMINAMATH_CALUDE_triangle_classification_l1519_151930

/-- Triangle classification based on side lengths --/
def TriangleType (a b c : ℝ) : Type :=
  { type : String // 
    type = "acute" ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 ∨
    type = "right" ∧ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∨
    type = "obtuse" ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) }

theorem triangle_classification :
  ∃ (t1 : TriangleType 4 6 8) (t2 : TriangleType 10 24 26) (t3 : TriangleType 10 12 14),
    t1.val = "obtuse" ∧ t2.val = "right" ∧ t3.val = "acute" := by
  sorry


end NUMINAMATH_CALUDE_triangle_classification_l1519_151930


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1519_151979

theorem intersection_point_of_lines (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 1) ↔ (x = 25/31 ∧ y = -22/31) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1519_151979


namespace NUMINAMATH_CALUDE_x_142_equals_1995_unique_l1519_151952

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := 
  if p x = 2 then 1 else sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n * p (x n) / q (x n)

theorem x_142_equals_1995_unique : 
  (x 142 = 1995) ∧ (∀ n : ℕ, n ≠ 142 → x n ≠ 1995) := by sorry

end NUMINAMATH_CALUDE_x_142_equals_1995_unique_l1519_151952


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1519_151933

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2*x + 1

-- Define the property of having a non-empty solution set
def has_nonempty_solution_set (a : ℝ) : Prop :=
  ∃ x, f a x < 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∃ a, has_nonempty_solution_set a ∧ a ≥ 2) ∧
  (∃ a, ¬has_nonempty_solution_set a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1519_151933


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l1519_151951

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define the complement of A with respect to U
def complement_A : Finset Nat := {2}

-- Define set A based on its complement
def A : Finset Nat := U \ complement_A

-- Theorem statement
theorem number_of_proper_subsets_of_A : 
  Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l1519_151951


namespace NUMINAMATH_CALUDE_inequality_proof_l1519_151922

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ ((a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1519_151922


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1519_151997

theorem train_speed_calculation (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 145 ∧ bridge_length = 230 ∧ crossing_time = 30 →
  ((train_length + bridge_length) / crossing_time) * 3.6 = 45 := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1519_151997


namespace NUMINAMATH_CALUDE_parking_probability_l1519_151915

/-- Represents a parking lot -/
structure ParkingLot where
  totalSpaces : ℕ
  occupiedSpaces : ℕ

/-- Calculates the probability of finding a specified number of adjacent empty spaces -/
def probabilityOfAdjacentEmptySpaces (lot : ParkingLot) (requiredSpaces : ℕ) : ℚ :=
  sorry

theorem parking_probability (lot : ParkingLot) :
  lot.totalSpaces = 20 →
  lot.occupiedSpaces = 14 →
  probabilityOfAdjacentEmptySpaces lot 3 = 19/25 :=
by sorry

end NUMINAMATH_CALUDE_parking_probability_l1519_151915


namespace NUMINAMATH_CALUDE_at_least_one_hit_probability_l1519_151914

theorem at_least_one_hit_probability 
  (prob_A prob_B prob_C : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.5) 
  (h_C : prob_C = 0.4) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_probability_l1519_151914


namespace NUMINAMATH_CALUDE_cubic_factorization_l1519_151905

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1519_151905


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1519_151962

theorem power_fraction_simplification :
  (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1519_151962


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l1519_151957

def simplified_terms_count (n : ℕ) : ℕ :=
  (n / 2 + 1)^2

theorem simplified_expression_terms (n : ℕ) (h : n = 2008) :
  simplified_terms_count n = 1010025 :=
by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l1519_151957


namespace NUMINAMATH_CALUDE_bus_problem_l1519_151942

/-- Proves that given a person walking to a bus stand, if they miss the bus by 10 minutes
    when walking at 4 km/h, and arrive 5 minutes early when walking at speed v,
    then v = 5 km/h. -/
theorem bus_problem (distance : ℝ) (speed1 speed2 : ℝ) : 
  distance = 5 →
  speed1 = 4 →
  distance / speed1 - distance / speed2 = 1/4 →
  speed2 = 5 := by
  sorry

#check bus_problem

end NUMINAMATH_CALUDE_bus_problem_l1519_151942


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1519_151977

theorem coefficient_x_squared_in_expansion : 
  let expansion := (X - 2 / X) ^ 4
  ∃ a b c d e : ℤ, 
    expansion = a * X^4 + b * X^3 + c * X^2 + d * X + e * X^0 ∧ 
    c = -8
  := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1519_151977


namespace NUMINAMATH_CALUDE_baby_panda_eats_50_pounds_l1519_151935

/-- The amount of bamboo (in pounds) an adult panda eats per day -/
def adult_panda_daily : ℕ := 138

/-- The total amount of bamboo (in pounds) eaten by both adult and baby pandas in a week -/
def total_weekly : ℕ := 1316

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of bamboo (in pounds) a baby panda eats per day -/
def baby_panda_daily : ℕ := (total_weekly - adult_panda_daily * days_per_week) / days_per_week

theorem baby_panda_eats_50_pounds : baby_panda_daily = 50 := by
  sorry

end NUMINAMATH_CALUDE_baby_panda_eats_50_pounds_l1519_151935


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1519_151986

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1519_151986


namespace NUMINAMATH_CALUDE_max_product_ab_l1519_151931

theorem max_product_ab (a b : ℝ) : (∀ x : ℝ, Real.exp x ≥ a * x + b) → a * b ≤ Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_ab_l1519_151931


namespace NUMINAMATH_CALUDE_book_area_l1519_151995

/-- The area of a rectangular book with length 5 inches and width 10 inches is 50 square inches. -/
theorem book_area : 
  let length : ℝ := 5
  let width : ℝ := 10
  let area := length * width
  area = 50 := by sorry

end NUMINAMATH_CALUDE_book_area_l1519_151995


namespace NUMINAMATH_CALUDE_incorrect_statement_C_l1519_151944

theorem incorrect_statement_C : ¬ (∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_C_l1519_151944


namespace NUMINAMATH_CALUDE_line_OF_equation_l1519_151963

/-- Given a triangle ABC with vertices A(0,a), B(b,0), C(c,0), and a point P(0,p) on line segment AO
    (not an endpoint), where a, b, c, and p are non-zero real numbers, prove that the equation of
    line OF is (1/c - 1/b)x + (1/p - 1/a)y = 0, where F is the intersection of lines CP and AB. -/
theorem line_OF_equation (a b c p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0)
    (hp_between : 0 < p ∧ p < a) : 
    ∃ (x y : ℝ), (1 / c - 1 / b) * x + (1 / p - 1 / a) * y = 0 ↔ 
    (∃ (t : ℝ), x = t * c ∧ y = t * p) ∧ (∃ (s : ℝ), x = s * b ∧ y = s * a) := by
  sorry

end NUMINAMATH_CALUDE_line_OF_equation_l1519_151963


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1519_151984

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1 / a^2 > 1 / b^2) ∧
  (∃ a b : ℝ, 1 / a^2 > 1 / b^2 ∧ ¬(b > a ∧ a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1519_151984


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1519_151985

def is_solution (x y w : ℕ) : Prop :=
  2^x * 3^y - 5^x * 7^w = 1

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 0, 0), (3, 0, 1), (1, 1, 0), (2, 2, 1)}

theorem diophantine_equation_solutions :
  ∀ x y w : ℕ, is_solution x y w ↔ (x, y, w) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1519_151985


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1519_151968

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 4*m + 1 = 0 → 2023 - m^2 + 4*m = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1519_151968


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1519_151956

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : ℕ, Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1519_151956


namespace NUMINAMATH_CALUDE_teacher_count_l1519_151940

theorem teacher_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) :
  total = 3000 →
  sample_size = 150 →
  students_in_sample = 140 →
  (total - (total * students_in_sample / sample_size) : ℕ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_teacher_count_l1519_151940


namespace NUMINAMATH_CALUDE_probability_red_white_green_is_correct_l1519_151954

def total_marbles : ℕ := 100
def blue_marbles : ℕ := 15
def red_marbles : ℕ := 35
def white_marbles : ℕ := 30
def green_marbles : ℕ := 10
def yellow_marbles : ℕ := total_marbles - (blue_marbles + red_marbles + white_marbles + green_marbles)

def probability_red_white_green : ℚ := (red_marbles + white_marbles + green_marbles : ℚ) / total_marbles

theorem probability_red_white_green_is_correct : probability_red_white_green = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_white_green_is_correct_l1519_151954


namespace NUMINAMATH_CALUDE_average_age_increase_l1519_151939

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 19 →
  student_avg_age = 20 →
  teacher_age = 40 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_age_increase_l1519_151939


namespace NUMINAMATH_CALUDE_circle_tangent_perpendicular_l1519_151983

-- Define the types for our geometric objects
variable (Point Circle Line : Type)

-- Define the necessary operations and relations
variable (radius : Circle → ℝ)
variable (intersect : Circle → Circle → Set Point)
variable (tangent_point : Circle → Line → Point)
variable (line_through : Point → Point → Line)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem circle_tangent_perpendicular 
  (Γ Γ' : Circle) 
  (A B C D : Point) 
  (t : Line) :
  radius Γ = radius Γ' →
  A ∈ intersect Γ Γ' →
  B ∈ intersect Γ Γ' →
  C = tangent_point Γ t →
  D = tangent_point Γ' t →
  perpendicular (line_through A C) (line_through B D) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_perpendicular_l1519_151983


namespace NUMINAMATH_CALUDE_max_c_value_l1519_151965

noncomputable section

def is_valid_solution (a b c x y z : ℝ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  a^x + b^y + c^z = 4 ∧
  x * a^x + y * b^y + z * c^z = 6 ∧
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9

theorem max_c_value (a b c x y z : ℝ) (h : is_valid_solution a b c x y z) :
  c ≤ Real.rpow 4 (1/3) :=
sorry

end

end NUMINAMATH_CALUDE_max_c_value_l1519_151965


namespace NUMINAMATH_CALUDE_time_left_before_movie_l1519_151994

def movie_time_minutes : ℕ := 2 * 60

def homework_time : ℕ := 30

def room_cleaning_time : ℕ := homework_time / 2

def dog_walking_time : ℕ := homework_time + 5

def trash_taking_time : ℕ := homework_time / 6

def total_chore_time : ℕ := homework_time + room_cleaning_time + dog_walking_time + trash_taking_time

theorem time_left_before_movie : movie_time_minutes - total_chore_time = 35 := by
  sorry

end NUMINAMATH_CALUDE_time_left_before_movie_l1519_151994


namespace NUMINAMATH_CALUDE_problem_solution_l1519_151921

theorem problem_solution : 
  let x : ℚ := 5
  let intermediate : ℚ := x * 12 / (180 / 3)
  intermediate + 80 = 81 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1519_151921


namespace NUMINAMATH_CALUDE_cartesian_angle_theorem_l1519_151919

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  -- The x-coordinate of the point on the terminal side
  x : ℝ
  -- The y-coordinate of the point on the terminal side
  y : ℝ
  -- The initial side is the non-negative half of the x-axis
  initial_side_positive_x : x > 0

/-- The theorem statement for the given problem -/
theorem cartesian_angle_theorem (α : CartesianAngle) 
  (h1 : α.x = 2) (h2 : α.y = 4) : 
  Real.tan (Real.arctan (α.y / α.x)) = 2 ∧ 
  (2 * Real.sin (Real.pi - Real.arctan (α.y / α.x)) + 
   2 * Real.cos (Real.arctan (α.y / α.x) / 2) ^ 2 - 1) / 
  (Real.sqrt 2 * Real.sin (Real.arctan (α.y / α.x) + Real.pi / 4)) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cartesian_angle_theorem_l1519_151919


namespace NUMINAMATH_CALUDE_sector_central_angle_l1519_151989

/-- Given a sector with radius 6 and area 6π, its central angle measure in degrees is 60. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) : 
  radius = 6 → area = 6 * Real.pi → angle = (area * 360) / (Real.pi * radius ^ 2) → angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1519_151989


namespace NUMINAMATH_CALUDE_train_passing_time_l1519_151996

/-- Calculates the time taken for a train to pass a man moving in the opposite direction. -/
theorem train_passing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 200 →
  train_speed = 80 →
  man_speed = 10 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 8 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l1519_151996


namespace NUMINAMATH_CALUDE_factorial_ratio_l1519_151948

theorem factorial_ratio : Nat.factorial 11 / Nat.factorial 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1519_151948


namespace NUMINAMATH_CALUDE_inverse_89_mod_91_l1519_151973

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_91_l1519_151973


namespace NUMINAMATH_CALUDE_intersection_length_l1519_151975

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the circle C in polar form
def circle_C (ρ θ : ℝ) : Prop := ρ^2 + 2*ρ*(Real.sin θ) = 3

-- Define the intersection points
def intersection_points (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ → Prop) : Set (ℝ × ℝ) :=
  {p | ∃ t, l t = p ∧ ∃ ρ θ, C ρ θ ∧ p.1 = ρ * (Real.cos θ) ∧ p.2 = ρ * (Real.sin θ)}

-- Theorem statement
theorem intersection_length :
  let points := intersection_points line_l circle_C
  ∃ M N : ℝ × ℝ, M ∈ points ∧ N ∈ points ∧ M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_length_l1519_151975


namespace NUMINAMATH_CALUDE_remaining_area_after_rectangle_removal_l1519_151941

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  area : ℝ

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  area : ℝ

/-- Represents the geometric figure described in the problem -/
structure GeometricFigure where
  XYZ : Triangle
  ABCD : Rectangle
  isXYZEquilateral : Prop
  smallestTriangleCount : ℕ
  smallestTriangleArea : ℝ

/-- The theorem to be proved -/
theorem remaining_area_after_rectangle_removal
  (figure : GeometricFigure)
  (h1 : figure.XYZ.area = 80)
  (h2 : figure.ABCD.area = 28)
  (h3 : figure.isXYZEquilateral)
  (h4 : figure.smallestTriangleCount = 9)
  (h5 : figure.smallestTriangleArea = 2) :
  figure.XYZ.area - figure.ABCD.area = 52 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_rectangle_removal_l1519_151941


namespace NUMINAMATH_CALUDE_binomial_12_11_l1519_151955

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l1519_151955


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l1519_151908

/-- Given a parallelogram with consecutive side lengths 10, 5y+3, 12, and 4x-1, prove that x + y = 91/20 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (4 * x - 1 = 10) →   -- First pair of opposite sides
  (5 * y + 3 = 12) →   -- Second pair of opposite sides
  x + y = 91/20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l1519_151908


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l1519_151936

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wallDimensions : Dimensions :=
  { length := 800, width := 22.5, height := 600 }

/-- The known dimensions of a brick in centimeters (height is unknown) -/
def brickDimensions (h : ℝ) : Dimensions :=
  { length := 50, width := 11.25, height := h }

/-- The number of bricks needed to build the wall -/
def numberOfBricks : ℕ := 3200

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧
    (volume wallDimensions = ↑numberOfBricks * volume (brickDimensions h)) := by
  sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l1519_151936


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1519_151904

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3 > 0) →
  (x - 3) * (3 * x + 7) = 11 * x - 4 →
  x = (13 + Real.sqrt 373) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1519_151904


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1519_151947

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) 
  (perimeter_relation : 4 * s = 2 * π * r) : 
  (s^2) / (π * r^2) = π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1519_151947


namespace NUMINAMATH_CALUDE_matrix_commutation_result_l1519_151990

theorem matrix_commutation_result (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = -3) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_result_l1519_151990


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l1519_151964

theorem unique_prime_perfect_square :
  ∀ p : ℕ, Prime p → (∃ q : ℕ, 5^p + 4*p^4 = q^2) → p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l1519_151964
