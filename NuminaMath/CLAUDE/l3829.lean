import Mathlib

namespace NUMINAMATH_CALUDE_four_numbers_product_equality_l3829_382976

theorem four_numbers_product_equality (p : ℝ) (hp : p ≥ 1) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a : ℝ) > p ∧ (b : ℝ) > p ∧ (c : ℝ) > p ∧ (d : ℝ) > p ∧
    (a : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (b : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (c : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (d : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (a * b : ℕ) = c * d :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_product_equality_l3829_382976


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_middle_same_is_90000_l3829_382964

/-- Counts the number of six-digit numbers where only the middle two digits are the same -/
def count_six_digit_numbers_middle_same : ℕ :=
  -- First digit: 9 choices (1-9)
  9 * 
  -- Second digit: 10 choices (0-9)
  10 * 
  -- Third digit: 10 choices (0-9)
  10 * 
  -- Fourth digit: 1 choice (same as third)
  1 * 
  -- Fifth digit: 10 choices (0-9)
  10 * 
  -- Sixth digit: 10 choices (0-9)
  10

/-- Theorem stating that the count of six-digit numbers with only middle digits the same is 90000 -/
theorem count_six_digit_numbers_middle_same_is_90000 :
  count_six_digit_numbers_middle_same = 90000 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_middle_same_is_90000_l3829_382964


namespace NUMINAMATH_CALUDE_carmen_total_sales_l3829_382917

/-- Represents the sales to a house -/
structure HouseSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ
  samoasPrice : ℚ
  thinMintsPrice : ℚ
  fudgeDelightsPrice : ℚ
  sugarCookiesPrice : ℚ

/-- Calculates the total sales for a house -/
def houseSalesTotal (sales : HouseSales) : ℚ :=
  sales.samoas * sales.samoasPrice +
  sales.thinMints * sales.thinMintsPrice +
  sales.fudgeDelights * sales.fudgeDelightsPrice +
  sales.sugarCookies * sales.sugarCookiesPrice

/-- Represents Carmen's total sales -/
def carmenSales : List HouseSales :=
  [
    { samoas := 3, thinMints := 0, fudgeDelights := 0, sugarCookies := 0,
      samoasPrice := 4, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 2, fudgeDelights := 1, sugarCookies := 0,
      samoasPrice := 0, thinMintsPrice := 7/2, fudgeDelightsPrice := 5, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 0, fudgeDelights := 0, sugarCookies := 9,
      samoasPrice := 0, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 2 }
  ]

theorem carmen_total_sales :
  (carmenSales.map houseSalesTotal).sum = 42 := by
  sorry

end NUMINAMATH_CALUDE_carmen_total_sales_l3829_382917


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3829_382991

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3829_382991


namespace NUMINAMATH_CALUDE_negative_a_squared_times_b_over_a_squared_l3829_382924

theorem negative_a_squared_times_b_over_a_squared (a b : ℝ) (h : a ≠ 0) :
  ((-a)^2 * b) / (a^2) = b := by sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_b_over_a_squared_l3829_382924


namespace NUMINAMATH_CALUDE_triangle_most_stable_triangular_structures_sturdy_l3829_382984

-- Define a structure
structure Shape :=
  (stability : ℝ)

-- Define a triangle
def Triangle : Shape :=
  { stability := 1 }

-- Define other shapes (for comparison)
def Square : Shape :=
  { stability := 0.8 }

def Pentagon : Shape :=
  { stability := 0.9 }

-- Theorem: Triangles have the highest stability
theorem triangle_most_stable :
  ∀ s : Shape, Triangle.stability ≥ s.stability :=
sorry

-- Theorem: Structures using triangles are sturdy
theorem triangular_structures_sturdy (structure_stability : Shape → ℝ) :
  structure_stability Triangle = 1 →
  ∀ s : Shape, structure_stability Triangle ≥ structure_stability s :=
sorry

end NUMINAMATH_CALUDE_triangle_most_stable_triangular_structures_sturdy_l3829_382984


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3829_382968

def is_clock_equivalent (t : ℕ) : Prop :=
  24 ∣ (t^2 - t)

theorem least_clock_equivalent_hour : 
  ∀ t : ℕ, t > 5 → t < 9 → ¬(is_clock_equivalent t) ∧ is_clock_equivalent 9 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3829_382968


namespace NUMINAMATH_CALUDE_calculate_ampersand_composition_l3829_382952

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 10 - x
def ampersand_left (x : ℝ) : ℝ := x - 10

-- State the theorem
theorem calculate_ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_calculate_ampersand_composition_l3829_382952


namespace NUMINAMATH_CALUDE_store_profit_analysis_l3829_382973

/-- Represents the relationship between sales volume and selling price -/
def sales_volume (x : ℝ) : ℝ := -x + 120

/-- Represents the profit function -/
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - 60)

/-- The cost price per item -/
def cost_price : ℝ := 60

/-- The maximum allowed profit percentage -/
def max_profit_percentage : ℝ := 0.45

theorem store_profit_analysis 
  (h1 : ∀ x, x ≥ cost_price)  -- Selling price not lower than cost price
  (h2 : ∀ x, profit x ≤ max_profit_percentage * cost_price * (x - cost_price))  -- Profit not exceeding 45%
  : 
  (∃ max_profit_price : ℝ, 
    max_profit_price = 87 ∧ 
    profit max_profit_price = 891 ∧ 
    ∀ x, profit x ≤ profit max_profit_price) ∧ 
  (∀ x, profit x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 110) := by
  sorry


end NUMINAMATH_CALUDE_store_profit_analysis_l3829_382973


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3829_382940

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3829_382940


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l3829_382997

theorem drum_capacity_ratio (CX CY : ℝ) 
  (h1 : CX > 0) 
  (h2 : CY > 0) 
  (h3 : (1/2 * CX + 1/3 * CY) / CY = 7/12) : 
  CY / CX = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l3829_382997


namespace NUMINAMATH_CALUDE_consecutive_good_numbers_l3829_382936

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for a number being "good" (not divisible by the sum of its digits) -/
def is_good (n : ℕ) : Prop := ¬(sum_of_digits n ∣ n)

/-- Main theorem -/
theorem consecutive_good_numbers (n : ℕ) (hn : n > 0) :
  ∃ (start : ℕ), ∀ (i : ℕ), i < n → is_good (start + i) := by sorry

end NUMINAMATH_CALUDE_consecutive_good_numbers_l3829_382936


namespace NUMINAMATH_CALUDE_fraction_equality_l3829_382955

theorem fraction_equality (a b : ℚ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3829_382955


namespace NUMINAMATH_CALUDE_cleaner_flow_rate_l3829_382907

/-- Represents the rate of cleaner flow through a pipe over time --/
structure CleanerFlow where
  initial_rate : ℝ
  middle_rate : ℝ
  final_rate : ℝ
  total_time : ℝ
  first_change_time : ℝ
  second_change_time : ℝ
  total_amount : ℝ

/-- The cleaner flow satisfies the problem conditions --/
def satisfies_conditions (flow : CleanerFlow) : Prop :=
  flow.initial_rate = 2 ∧
  flow.final_rate = 4 ∧
  flow.total_time = 30 ∧
  flow.first_change_time = 15 ∧
  flow.second_change_time = 25 ∧
  flow.total_amount = 80 ∧
  flow.initial_rate * flow.first_change_time +
  flow.middle_rate * (flow.second_change_time - flow.first_change_time) +
  flow.final_rate * (flow.total_time - flow.second_change_time) = flow.total_amount

theorem cleaner_flow_rate (flow : CleanerFlow) :
  satisfies_conditions flow → flow.middle_rate = 3 := by
  sorry


end NUMINAMATH_CALUDE_cleaner_flow_rate_l3829_382907


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3829_382930

/-- The equation |y-3| = √((x+4)² + 4y²) represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), |y - 3| = Real.sqrt ((x + 4)^2 + 4*y^2) →
  ∃ (A B C D E : ℝ), A ≠ 0 ∧ C ≠ 0 ∧ A * C < 0 ∧
    A * y^2 + B * y + C * x^2 + D * x + E = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3829_382930


namespace NUMINAMATH_CALUDE_horner_operations_count_l3829_382970

/-- Represents a polynomial of degree 6 with a constant term -/
structure Polynomial6 where
  coeffs : Fin 7 → ℝ
  constant_term : coeffs 0 ≠ 0

/-- Counts the number of operations in Horner's method for a polynomial of degree 6 -/
def horner_operations (p : Polynomial6) : ℕ :=
  6 + 6

theorem horner_operations_count (p : Polynomial6) :
  horner_operations p = 12 := by
  sorry

#check horner_operations_count

end NUMINAMATH_CALUDE_horner_operations_count_l3829_382970


namespace NUMINAMATH_CALUDE_concert_attendance_l3829_382911

theorem concert_attendance (num_buses : ℕ) (students_per_bus : ℕ) (students_in_minivan : ℕ) :
  num_buses = 12 →
  students_per_bus = 38 →
  students_in_minivan = 5 →
  num_buses * students_per_bus + students_in_minivan = 461 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3829_382911


namespace NUMINAMATH_CALUDE_sarah_cookies_count_l3829_382932

/-- The number of cookies Sarah took -/
def cookies_sarah_took (total_cookies : ℕ) (num_neighbors : ℕ) (cookies_per_neighbor : ℕ) (cookies_left : ℕ) : ℕ :=
  total_cookies - cookies_left - (num_neighbors - 1) * cookies_per_neighbor

theorem sarah_cookies_count :
  cookies_sarah_took 150 15 10 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cookies_count_l3829_382932


namespace NUMINAMATH_CALUDE_exists_special_sequence_l3829_382951

/-- A sequence of positive integers satisfying the required properties -/
def SpecialSequence : Type :=
  ℕ → ℕ+

/-- The property that a number has no square factors other than 1 -/
def HasNoSquareFactors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- The main theorem stating the existence of the special sequence -/
theorem exists_special_sequence :
  ∃ (seq : SpecialSequence),
    (∀ i j : ℕ, i < j → seq i < seq j) ∧
    (∀ i j : ℕ, i ≠ j → HasNoSquareFactors ((seq i).val + (seq j).val)) := by
  sorry


end NUMINAMATH_CALUDE_exists_special_sequence_l3829_382951


namespace NUMINAMATH_CALUDE_expression_simplification_l3829_382958

theorem expression_simplification (y : ℝ) :
  3 * y - 7 * y^2 + 15 - (2 + 6 * y - 7 * y^2) = -3 * y + 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3829_382958


namespace NUMINAMATH_CALUDE_reverse_difference_for_253_l3829_382905

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  h_range := n.o_range
  t_range := n.t_range
  o_range := n.h_range

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem reverse_difference_for_253 (n : ThreeDigitNumber) 
    (h_253 : n.toNat = 253)
    (h_sum : n.sumOfDigits = 10)
    (h_middle : n.tens = n.hundreds + n.ones) :
    (n.reverse.toNat - n.toNat) = 99 := by
  sorry

#check reverse_difference_for_253

end NUMINAMATH_CALUDE_reverse_difference_for_253_l3829_382905


namespace NUMINAMATH_CALUDE_function_properties_l3829_382922

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ)
  (h1 : is_odd (λ x ↦ f (x + 2)))
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Ici 2 → x₂ ∈ Set.Ici 2 → x₁ < x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0) :
  (∀ x y, x < y → f x < f y) ∧
  {x : ℝ | f x < 0} = Set.Iio 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3829_382922


namespace NUMINAMATH_CALUDE_testicular_cell_properties_l3829_382965

-- Define the possible bases
inductive Base
| A
| C
| T

-- Define the possible cell cycle periods
inductive Period
| Interphase
| EarlyMitosis
| LateMitosis
| EarlyMeiosis1
| LateMeiosis1
| EarlyMeiosis2
| LateMeiosis2

-- Define the structure of a testicular cell
structure TesticularCell where
  nucleotideTypes : Finset (List Base)
  lowestStabilityPeriod : Period
  dnaSeperationPeriod : Period

-- Define the theorem
theorem testicular_cell_properties : ∃ (cell : TesticularCell),
  (cell.nucleotideTypes.card = 3) ∧
  (cell.lowestStabilityPeriod = Period.Interphase) ∧
  (cell.dnaSeperationPeriod = Period.LateMeiosis1 ∨ cell.dnaSeperationPeriod = Period.LateMeiosis2) :=
by
  sorry

end NUMINAMATH_CALUDE_testicular_cell_properties_l3829_382965


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3829_382945

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3829_382945


namespace NUMINAMATH_CALUDE_acute_triangle_selection_l3829_382902

/-- A point on a circle, with a color attribute -/
structure ColoredPoint where
  point : ℝ × ℝ
  color : Nat

/-- Represents a circle with colored points -/
structure ColoredCircle where
  center : ℝ × ℝ
  radius : ℝ
  points : List ColoredPoint

/-- Checks if three points form an acute or right-angled triangle -/
def isAcuteOrRightTriangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Checks if a ColoredCircle has at least one point of each color (assuming colors are 1, 2, 3) -/
def hasAllColors (circle : ColoredCircle) : Prop := sorry

/-- The main theorem to be proved -/
theorem acute_triangle_selection (circle : ColoredCircle) 
  (h : hasAllColors circle) : 
  ∃ (p1 p2 p3 : ColoredPoint), 
    p1 ∈ circle.points ∧ 
    p2 ∈ circle.points ∧ 
    p3 ∈ circle.points ∧ 
    p1.color ≠ p2.color ∧ 
    p2.color ≠ p3.color ∧ 
    p1.color ≠ p3.color ∧ 
    isAcuteOrRightTriangle p1.point p2.point p3.point := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_selection_l3829_382902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3829_382975

/-- 
Given an arithmetic sequence with:
- First term a₁ = -5
- Last term aₙ = 40
- Common difference d = 3

Prove that the sequence has 16 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ),
  (a 0 = -5) →  -- First term
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference
  (∃ k, a k = 40) →  -- Last term
  (∃ n, n = 16 ∧ a (n - 1) = 40) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3829_382975


namespace NUMINAMATH_CALUDE_machine_production_time_l3829_382981

theorem machine_production_time (T : ℝ) : 
  T > 0 ∧ 
  (1 / T + 1 / 30 = 1 / 12) → 
  T = 45 := by
sorry

end NUMINAMATH_CALUDE_machine_production_time_l3829_382981


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l3829_382938

/-- Calculates the number of widgets Nina can purchase given her budget and widget price information. -/
def widgets_nina_can_buy (budget : ℕ) (reduced_price_widgets : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price := (budget + reduced_price_widgets * price_reduction) / reduced_price_widgets
  budget / original_price

/-- Proves that Nina can buy 6 widgets given the problem conditions. -/
theorem nina_widget_purchase :
  widgets_nina_can_buy 48 8 2 = 6 := by
  sorry

#eval widgets_nina_can_buy 48 8 2

end NUMINAMATH_CALUDE_nina_widget_purchase_l3829_382938


namespace NUMINAMATH_CALUDE_plain_lemonade_price_calculation_l3829_382921

/-- The price of a glass of plain lemonade -/
def plain_lemonade_price : ℚ := 3 / 4

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_sold : ℕ := 36

/-- The amount made from strawberry lemonade -/
def strawberry_lemonade_sales : ℕ := 16

/-- The difference between plain and strawberry lemonade sales -/
def sales_difference : ℕ := 11

theorem plain_lemonade_price_calculation :
  plain_lemonade_price * plain_lemonade_sold = 
  (strawberry_lemonade_sales + sales_difference : ℚ) := by sorry

end NUMINAMATH_CALUDE_plain_lemonade_price_calculation_l3829_382921


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3829_382919

theorem mod_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -300 ≡ n [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3829_382919


namespace NUMINAMATH_CALUDE_grandfather_age_proof_l3829_382913

/-- The age of Xiaoming's grandfather -/
def grandfather_age : ℕ := 79

/-- The result after processing the grandfather's age -/
def processed_age (age : ℕ) : ℕ :=
  ((age - 15) / 4 - 6) * 10

theorem grandfather_age_proof :
  processed_age grandfather_age = 100 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_age_proof_l3829_382913


namespace NUMINAMATH_CALUDE_percent_exceeding_speed_limit_l3829_382962

theorem percent_exceeding_speed_limit 
  (total_motorists : ℕ) 
  (h_total_positive : total_motorists > 0)
  (percent_ticketed : ℝ) 
  (h_percent_ticketed : percent_ticketed = 10)
  (percent_unticketed_speeders : ℝ) 
  (h_percent_unticketed : percent_unticketed_speeders = 50) : 
  (percent_ticketed * total_motorists / 100 + 
   percent_ticketed * total_motorists / 100) / total_motorists * 100 = 20 := by
  sorry

#check percent_exceeding_speed_limit

end NUMINAMATH_CALUDE_percent_exceeding_speed_limit_l3829_382962


namespace NUMINAMATH_CALUDE_cubic_monotonicity_l3829_382910

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem cubic_monotonicity 
  (a b c d : ℝ) 
  (h1 : f a b c d 0 = -4)
  (h2 : f' a b c 0 = 12)
  (h3 : f a b c d 2 = 0)
  (h4 : f' a b c 2 = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2 ∧
  (∀ x < x₁, f' a b c x > 0) ∧
  (∀ x ∈ Set.Ioo x₁ x₂, f' a b c x < 0) ∧
  (∀ x > x₂, f' a b c x > 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonicity_l3829_382910


namespace NUMINAMATH_CALUDE_ad_agency_client_distribution_l3829_382967

/-- Given an advertising agency with 180 clients, where:
    - 115 use television
    - 110 use radio
    - 130 use magazines
    - 85 use television and magazines
    - 75 use television and radio
    - 80 use all three
    This theorem proves that 95 clients use radio and magazines. -/
theorem ad_agency_client_distribution (total : ℕ) (T R M TM TR TRM : ℕ) 
  (h_total : total = 180)
  (h_T : T = 115)
  (h_R : R = 110)
  (h_M : M = 130)
  (h_TM : TM = 85)
  (h_TR : TR = 75)
  (h_TRM : TRM = 80)
  : total = T + R + M - TR - TM - (T + R + M - TR - TM - total + TRM) + TRM := by
  sorry

end NUMINAMATH_CALUDE_ad_agency_client_distribution_l3829_382967


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3829_382946

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (purchase_price_ratio : ℝ) 
  (discount_ratio : ℝ) 
  (profit_ratio : ℝ) 
  (marked_price_ratio : ℝ) 
  (h1 : purchase_price_ratio = 0.7) 
  (h2 : discount_ratio = 0.25) 
  (h3 : profit_ratio = 0.3) 
  (h4 : marked_price_ratio * (1 - discount_ratio) * list_price - 
        purchase_price_ratio * list_price = 
        profit_ratio * marked_price_ratio * (1 - discount_ratio) * list_price) :
  marked_price_ratio = 1.33 := by
  sorry

#check merchant_pricing_strategy

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3829_382946


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_conditions_l3829_382901

theorem integer_pairs_satisfying_conditions :
  ∀ m n : ℤ, 
    m^2 = n^5 + n^4 + 1 ∧ 
    (m - 7*n) ∣ (m - 4*n) → 
    ((m = -1 ∧ n = 0) ∨ (m = 1 ∧ n = 0)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_conditions_l3829_382901


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_min_value_is_three_l3829_382912

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_min_value_is_three_l3829_382912


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_l3829_382986

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ |x - 1| ≥ 3) →
  a < -4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_l3829_382986


namespace NUMINAMATH_CALUDE_cow_count_is_twenty_l3829_382950

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: In a group where the total number of legs is 40 more than twice
    the number of heads, the number of cows is 20 -/
theorem cow_count_is_twenty (count : AnimalCount) 
    (h : totalLegs count = 2 * totalHeads count + 40) : 
    count.cows = 20 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_twenty_l3829_382950


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3829_382914

/-- A normally distributed random variable -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (X : NormalRV) (f : ℝ → Prop) : ℝ := sorry

theorem normal_distribution_probability 
  (X : NormalRV)
  (h1 : P X (λ x => x > 5) = 0.2)
  (h2 : P X (λ x => x < -1) = 0.2) :
  P X (λ x => 2 < x ∧ x < 5) = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3829_382914


namespace NUMINAMATH_CALUDE_D_value_l3829_382949

/-- The determinant of a matrix with elements |i-j| -/
def D (n : ℕ) : ℚ :=
  let M : Matrix (Fin n) (Fin n) ℚ := λ i j => |i.val - j.val|
  M.det

/-- Theorem stating the value of the determinant D_n -/
theorem D_value (n : ℕ) (h : n > 0) : D n = (-1)^(n-1) * (n-1) * 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_D_value_l3829_382949


namespace NUMINAMATH_CALUDE_problem_solution_l3829_382941

theorem problem_solution : (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3829_382941


namespace NUMINAMATH_CALUDE_cubic_polynomial_with_arithmetic_progression_roots_l3829_382956

/-- A cubic polynomial with coefficients in ℂ -/
structure CubicPolynomial where
  a : ℂ
  b : ℂ
  c : ℂ
  d : ℂ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_in_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), (r - d) * (r) * (r + d) = -p.d ∧
                (r - d) + r + (r + d) = p.b ∧
                (r - d) * r + (r - d) * (r + d) + r * (r + d) = p.c

/-- The roots of a cubic polynomial are not all real -/
def roots_not_all_real (p : CubicPolynomial) : Prop :=
  ∃ (r : ℂ), r.im ≠ 0 ∧ (r^3 + p.a * r^2 + p.b * r + p.c = 0)

theorem cubic_polynomial_with_arithmetic_progression_roots (a : ℝ) :
  let p := CubicPolynomial.mk 1 (-9) 42 a
  roots_in_arithmetic_progression p ∧ roots_not_all_real p → a = -72 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_with_arithmetic_progression_roots_l3829_382956


namespace NUMINAMATH_CALUDE_short_bar_length_l3829_382957

theorem short_bar_length (total_length long_short_diff : ℝ) 
  (h1 : total_length = 950)
  (h2 : long_short_diff = 150) :
  let short_bar := (total_length - long_short_diff) / 2
  short_bar = 400 := by
sorry

end NUMINAMATH_CALUDE_short_bar_length_l3829_382957


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3829_382985

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis, 
    prove that its semi-minor axis has length √7 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -4))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3829_382985


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l3829_382983

/-- Given two points A and B on a plane, and a point C such that BC = 1/2 * AB,
    this theorem proves that C has specific coordinates. -/
theorem extended_segment_coordinates (A B C : ℝ × ℝ) : 
  A = (2, -2) → 
  B = (14, 4) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (20, 7) := by
sorry


end NUMINAMATH_CALUDE_extended_segment_coordinates_l3829_382983


namespace NUMINAMATH_CALUDE_intersection_M_N_l3829_382994

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3829_382994


namespace NUMINAMATH_CALUDE_inequality_reciprocal_l3829_382935

theorem inequality_reciprocal (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
sorry

end NUMINAMATH_CALUDE_inequality_reciprocal_l3829_382935


namespace NUMINAMATH_CALUDE_abes_age_l3829_382937

theorem abes_age (present_age : ℕ) : 
  (present_age + (present_age - 7) = 31) → present_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_l3829_382937


namespace NUMINAMATH_CALUDE_special_sequence_bijective_l3829_382988

/-- A sequence of integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinite positive values
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinite negative values
  (∀ n : ℕ+, ∀ i j, i ≠ j → i ≤ n → j ≤ n → a i % n ≠ a j % n)  -- Distinct modulo n

/-- The theorem stating that every integer appears exactly once in the sequence -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_bijective_l3829_382988


namespace NUMINAMATH_CALUDE_factor_polynomial_l3829_382974

theorem factor_polynomial (x y : ℝ) : 
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3829_382974


namespace NUMINAMATH_CALUDE_equal_arcs_equal_chords_l3829_382927

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc where
  circle : Circle
  start_angle : ℝ
  end_angle : ℝ

/-- Represents a chord of a circle -/
structure Chord where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Function to calculate the length of an arc -/
def arcLength (arc : Arc) : ℝ := sorry

/-- Function to calculate the length of a chord -/
def chordLength (chord : Chord) : ℝ := sorry

/-- Theorem: In a circle, equal arcs correspond to equal chords -/
theorem equal_arcs_equal_chords (c : Circle) (arc1 arc2 : Arc) (chord1 chord2 : Chord) :
  arc1.circle = c → arc2.circle = c →
  chord1.circle = c → chord2.circle = c →
  arcLength arc1 = arcLength arc2 →
  chord1.endpoint1 = (c.center.1 + c.radius * Real.cos arc1.start_angle,
                      c.center.2 + c.radius * Real.sin arc1.start_angle) →
  chord1.endpoint2 = (c.center.1 + c.radius * Real.cos arc1.end_angle,
                      c.center.2 + c.radius * Real.sin arc1.end_angle) →
  chord2.endpoint1 = (c.center.1 + c.radius * Real.cos arc2.start_angle,
                      c.center.2 + c.radius * Real.sin arc2.start_angle) →
  chord2.endpoint2 = (c.center.1 + c.radius * Real.cos arc2.end_angle,
                      c.center.2 + c.radius * Real.sin arc2.end_angle) →
  chordLength chord1 = chordLength chord2 := by sorry

end NUMINAMATH_CALUDE_equal_arcs_equal_chords_l3829_382927


namespace NUMINAMATH_CALUDE_liter_equals_cubic_decimeter_l3829_382989

-- Define the conversion factor between liters and cubic decimeters
def liter_to_cubic_decimeter : ℝ := 1

-- Theorem statement
theorem liter_equals_cubic_decimeter :
  1.5 * liter_to_cubic_decimeter = 1.5 := by sorry

end NUMINAMATH_CALUDE_liter_equals_cubic_decimeter_l3829_382989


namespace NUMINAMATH_CALUDE_original_price_per_acre_l3829_382929

/-- Proves that the original price per acre was $140 --/
theorem original_price_per_acre 
  (total_area : ℕ)
  (sold_area : ℕ)
  (selling_price : ℕ)
  (profit : ℕ)
  (h1 : total_area = 200)
  (h2 : sold_area = total_area / 2)
  (h3 : selling_price = 200)
  (h4 : profit = 6000)
  : (selling_price * sold_area - profit) / sold_area = 140 := by
  sorry

end NUMINAMATH_CALUDE_original_price_per_acre_l3829_382929


namespace NUMINAMATH_CALUDE_base_b_square_l3829_382990

theorem base_b_square (b : ℕ) : 
  (2 * b + 4)^2 = 5 * b^2 + 5 * b + 4 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l3829_382990


namespace NUMINAMATH_CALUDE_max_children_spell_names_l3829_382918

/-- Represents the available letters in the bag -/
def LetterBag : Finset Char := {'A', 'A', 'A', 'A', 'B', 'B', 'D', 'I', 'I', 'M', 'M', 'N', 'N', 'N', 'Y', 'Y'}

/-- Represents the names of the children -/
inductive Child
| Anna
| Vanya
| Dani
| Dima

/-- Returns the set of letters needed to spell a child's name -/
def lettersNeeded (c : Child) : Finset Char :=
  match c with
  | Child.Anna => {'A', 'N', 'N', 'A'}
  | Child.Vanya => {'V', 'A', 'N', 'Y'}
  | Child.Dani => {'D', 'A', 'N', 'Y'}
  | Child.Dima => {'D', 'I', 'M', 'A'}

/-- Theorem stating the maximum number of children who can spell their names -/
theorem max_children_spell_names :
  ∃ (S : Finset Child), (∀ c ∈ S, lettersNeeded c ⊆ LetterBag) ∧ 
                        (∀ T : Finset Child, (∀ c ∈ T, lettersNeeded c ⊆ LetterBag) → T.card ≤ S.card) ∧
                        S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_children_spell_names_l3829_382918


namespace NUMINAMATH_CALUDE_ellipse_point_properties_l3829_382904

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-4, 0)
def right_focus : ℝ × ℝ := (4, 0)

-- Define the angle between PF₁ and PF₂
def angle_PF1F2 (P : ℝ × ℝ) : ℝ := 60

-- Theorem statement
theorem ellipse_point_properties (P : ℝ × ℝ) 
  (h_on_ellipse : is_on_ellipse P.1 P.2) 
  (h_angle : angle_PF1F2 P = 60) :
  (∃ (S : ℝ), S = 3 * Real.sqrt 3 ∧ 
    S = (1/2) * Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) *
              Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) *
              Real.sin (angle_PF1F2 P * π / 180)) ∧
  (P.1 = 5 * Real.sqrt 13 / 4 ∨ P.1 = -5 * Real.sqrt 13 / 4) ∧
  (P.2 = 4 * Real.sqrt 3 / 4 ∨ P.2 = -4 * Real.sqrt 3 / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_properties_l3829_382904


namespace NUMINAMATH_CALUDE_perpendicular_planes_line_parallel_l3829_382942

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  perpendicular_plane_plane : Plane → Plane → Prop
  perpendicular_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop

variable {S : Space3D}

/-- The main theorem -/
theorem perpendicular_planes_line_parallel 
  (α β : S.Plane) (m : S.Line)
  (h1 : S.perpendicular_plane_plane α β)
  (h2 : S.perpendicular_line_plane m β)
  (h3 : ¬ S.line_in_plane m α) :
  S.parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_line_parallel_l3829_382942


namespace NUMINAMATH_CALUDE_gold_copper_ratio_l3829_382909

/-- Proves that the ratio of gold to copper in an alloy that is 17 times as heavy as water is 4:1,
    given that gold is 19 times as heavy as water and copper is 9 times as heavy as water. -/
theorem gold_copper_ratio (g c : ℝ) 
  (h1 : g > 0) 
  (h2 : c > 0) 
  (h_gold : 19 * g = 17 * (g + c)) 
  (h_copper : 9 * c = 17 * (g + c) - 19 * g) : 
  g / c = 4 := by
sorry

end NUMINAMATH_CALUDE_gold_copper_ratio_l3829_382909


namespace NUMINAMATH_CALUDE_reliable_plumbing_hourly_charge_l3829_382948

/-- Paul's Plumbing visit charge -/
def paul_visit : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def paul_hourly : ℕ := 35

/-- Reliable Plumbing visit charge -/
def reliable_visit : ℕ := 75

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Reliable Plumbing's hourly labor charge -/
def reliable_hourly : ℕ := 30

theorem reliable_plumbing_hourly_charge :
  paul_visit + labor_hours * paul_hourly = reliable_visit + labor_hours * reliable_hourly :=
by sorry

end NUMINAMATH_CALUDE_reliable_plumbing_hourly_charge_l3829_382948


namespace NUMINAMATH_CALUDE_bluetooth_module_stock_l3829_382993

theorem bluetooth_module_stock (total_modules : ℕ) (total_cost : ℚ)
  (expensive_cost cheap_cost : ℚ) :
  total_modules = 11 →
  total_cost = 45 →
  expensive_cost = 10 →
  cheap_cost = 7/2 →
  ∃ (expensive_count cheap_count : ℕ),
    expensive_count + cheap_count = total_modules ∧
    expensive_count * expensive_cost + cheap_count * cheap_cost = total_cost ∧
    cheap_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_bluetooth_module_stock_l3829_382993


namespace NUMINAMATH_CALUDE_strawberries_per_basket_is_15_l3829_382947

/-- The number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket (kimberly_amount : ℕ) (brother_baskets : ℕ) (parents_amount : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / 4) / brother_baskets

/-- Theorem stating the number of strawberries in each basket picked by Kimberly's brother -/
theorem strawberries_per_basket_is_15 
  (kimberly_amount : ℕ) 
  (brother_baskets : ℕ) 
  (parents_amount : ℕ) 
  (total_amount : ℕ) 
  (h1 : kimberly_amount = 8 * (brother_baskets * strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount))
  (h2 : parents_amount = kimberly_amount - 93)
  (h3 : brother_baskets = 3)
  (h4 : total_amount = 4 * 168)
  : strawberries_per_basket kimberly_amount brother_baskets parents_amount total_amount = 15 :=
sorry


end NUMINAMATH_CALUDE_strawberries_per_basket_is_15_l3829_382947


namespace NUMINAMATH_CALUDE_pavan_journey_l3829_382999

theorem pavan_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  speed1 = 30 →
  speed2 = 25 →
  ∃ (total_distance : ℝ),
    total_distance / (2 * speed1) + total_distance / (2 * speed2) = total_time ∧
    total_distance = 300 :=
by sorry

end NUMINAMATH_CALUDE_pavan_journey_l3829_382999


namespace NUMINAMATH_CALUDE_fruit_selection_problem_l3829_382972

/-- The number of ways to choose n items from k groups with at least m items from each group -/
def choose_with_minimum (n k m : ℕ) : ℕ :=
  (n - k * m + k - 1).choose (k - 1)

/-- The problem statement -/
theorem fruit_selection_problem :
  choose_with_minimum 15 4 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_selection_problem_l3829_382972


namespace NUMINAMATH_CALUDE_students_pets_difference_l3829_382931

theorem students_pets_difference (num_classrooms : ℕ) (students_per_class : ℕ) (pets_per_class : ℕ)
  (h1 : num_classrooms = 5)
  (h2 : students_per_class = 20)
  (h3 : pets_per_class = 3) :
  num_classrooms * students_per_class - num_classrooms * pets_per_class = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_pets_difference_l3829_382931


namespace NUMINAMATH_CALUDE_virus_spread_l3829_382926

def infection_rate : ℕ → ℕ
  | 0 => 1
  | n + 1 => infection_rate n * 9

theorem virus_spread (x : ℕ) :
  (∃ n : ℕ, infection_rate n = 81) →
  (∀ n : ℕ, infection_rate (n + 1) = infection_rate n * 9) →
  infection_rate 2 = 81 →
  infection_rate 3 > 700 :=
by sorry

#check virus_spread

end NUMINAMATH_CALUDE_virus_spread_l3829_382926


namespace NUMINAMATH_CALUDE_cyclist_meeting_oncoming_buses_l3829_382906

/-- The time interval between a cyclist meeting oncoming buses, given constant speeds and specific time intervals -/
theorem cyclist_meeting_oncoming_buses 
  (overtake_interval : ℝ) 
  (bus_interval : ℝ) 
  (h1 : overtake_interval > 0)
  (h2 : bus_interval > 0)
  (h3 : bus_interval = overtake_interval / 2) :
  overtake_interval / 2 = bus_interval := by
sorry

end NUMINAMATH_CALUDE_cyclist_meeting_oncoming_buses_l3829_382906


namespace NUMINAMATH_CALUDE_bus_ride_difference_l3829_382939

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l3829_382939


namespace NUMINAMATH_CALUDE_union_of_specific_sets_l3829_382933

theorem union_of_specific_sets :
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {2, 4}
  A ∪ B = {0, 1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_specific_sets_l3829_382933


namespace NUMINAMATH_CALUDE_mindmaster_secret_codes_l3829_382943

/-- The number of different colors available for pegs -/
def num_colors : ℕ := 8

/-- The number of slots in the code -/
def num_slots : ℕ := 4

/-- The total number of options for each slot (colors + empty) -/
def options_per_slot : ℕ := num_colors + 1

/-- The number of possible secret codes in the Mindmaster variation -/
theorem mindmaster_secret_codes :
  (options_per_slot ^ num_slots) - 1 = 6560 := by sorry

end NUMINAMATH_CALUDE_mindmaster_secret_codes_l3829_382943


namespace NUMINAMATH_CALUDE_diophantine_approximation_l3829_382979

theorem diophantine_approximation (α : ℝ) (C : ℝ) (h_α : α > 0) (h_C : C > 1) :
  ∃ (x : ℕ) (y : ℤ), (x : ℝ) < C ∧ |x * α - y| ≤ 1 / C := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l3829_382979


namespace NUMINAMATH_CALUDE_f_properties_l3829_382969

def f (x : ℝ) := |x - 2|

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3829_382969


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3829_382925

/-- Given a geometric sequence where the first term is 512 and the 8th term is 2,
    prove that the 6th term is 16. -/
theorem geometric_sequence_sixth_term
  (a : ℝ) -- First term
  (r : ℝ) -- Common ratio
  (h1 : a = 512) -- First term is 512
  (h2 : a * r^7 = 2) -- 8th term is 2
  : a * r^5 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3829_382925


namespace NUMINAMATH_CALUDE_fraction_equality_l3829_382908

theorem fraction_equality : (2 - 4 + 8 - 16 + 32 + 64) / (4 - 8 + 16 - 32 + 64 + 128) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3829_382908


namespace NUMINAMATH_CALUDE_miniature_cars_per_package_l3829_382996

theorem miniature_cars_per_package (total_packages : ℕ) (fraction_given_away : ℚ) (cars_left : ℕ) : 
  total_packages = 10 → 
  fraction_given_away = 2/5 → 
  cars_left = 30 → 
  ∃ (cars_per_package : ℕ), 
    cars_per_package = 5 ∧ 
    (total_packages * cars_per_package) * (1 - fraction_given_away) = cars_left :=
by
  sorry

end NUMINAMATH_CALUDE_miniature_cars_per_package_l3829_382996


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3829_382900

theorem sqrt_product_simplification (q : ℝ) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^2) * Real.sqrt (14 * q^3) = 4 * q^3 * Real.sqrt 105 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3829_382900


namespace NUMINAMATH_CALUDE_find_m_l3829_382977

theorem find_m : ∃ m : ℝ, (15 : ℝ)^(4*m) = (1/15 : ℝ)^(m-30) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3829_382977


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3829_382960

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 25 →
  c = b + 7 →
  a ≤ b ∧ b ≤ c →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3829_382960


namespace NUMINAMATH_CALUDE_sum_of_altitudes_is_23_and_one_seventh_l3829_382959

/-- A triangle formed by the line 18x + 9y = 108 and the coordinate axes -/
structure Triangle where
  -- The line equation
  line_eq : ℝ → ℝ → Prop := fun x y => 18 * x + 9 * y = 108
  -- The triangle is formed with coordinate axes
  forms_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ line_eq a 0 ∧ line_eq 0 b

/-- The sum of the lengths of the altitudes of the triangle -/
def sum_of_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the sum of the altitudes is 23 1/7 -/
theorem sum_of_altitudes_is_23_and_one_seventh (t : Triangle) :
  sum_of_altitudes t = 23 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_is_23_and_one_seventh_l3829_382959


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3829_382971

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a sampling method -/
def selectionProbability (method : SamplingMethod) (N : ℕ) (n : ℕ) : ℝ :=
  sorry

theorem equal_selection_probability (N : ℕ) (n : ℕ) :
  ∀ (m₁ m₂ : SamplingMethod), selectionProbability m₁ N n = selectionProbability m₂ N n :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l3829_382971


namespace NUMINAMATH_CALUDE_pears_transport_l3829_382978

/-- Prove that given 8 tons of apples and the amount of pears being 7 times the amount of apples,
    the total amount of pears transported is 56 tons. -/
theorem pears_transport (apple_tons : ℕ) (pear_multiplier : ℕ) : 
  apple_tons = 8 → pear_multiplier = 7 → apple_tons * pear_multiplier = 56 := by
  sorry

end NUMINAMATH_CALUDE_pears_transport_l3829_382978


namespace NUMINAMATH_CALUDE_share_multiple_l3829_382954

theorem share_multiple (total : ℚ) (c_share : ℚ) (x : ℚ) : 
  total = 585 →
  c_share = 260 →
  ∃ (a_share b_share : ℚ),
    a_share + b_share + c_share = total ∧
    x * a_share = 6 * b_share ∧
    x * a_share = 3 * c_share →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l3829_382954


namespace NUMINAMATH_CALUDE_magnitude_of_b_l3829_382987

def a : ℝ × ℝ := (2, 1)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h1 : a.fst * b.fst + a.snd * b.snd = 10)
  (h2 : (a.fst + 2 * b.fst)^2 + (a.snd + 2 * b.snd)^2 = 50) :
  Real.sqrt (b.fst^2 + b.snd^2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l3829_382987


namespace NUMINAMATH_CALUDE_negation_equivalence_l3829_382980

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3829_382980


namespace NUMINAMATH_CALUDE_n_value_l3829_382995

theorem n_value : ∃ (n : ℤ), (1/6 : ℚ) < (n : ℚ)/24 ∧ (n : ℚ)/24 < (1/4 : ℚ) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_n_value_l3829_382995


namespace NUMINAMATH_CALUDE_total_cost_is_180_l3829_382916

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let palm_fern_cost : ℝ := 15.00
  let creeping_jenny_cost : ℝ := 4.00
  let geranium_cost : ℝ := 3.50
  let plants_per_pot : ℕ := 1 + 4 + 4
  let cost_per_pot : ℝ := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost
  let corners : ℕ := 4
  corners * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_180_l3829_382916


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3829_382923

/-- Given an inverse proportion function f(x) = k/x where k ≠ 0 and 1 ≤ x ≤ 3,
    if the difference between the maximum and minimum values of f(x) is 4,
    then k = ±6 -/
theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ k / x
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ f 1 ∧ f x ≥ f 3) →
  f 1 - f 3 = 4 →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3829_382923


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3829_382961

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The number of possible license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count * vowel_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3829_382961


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3829_382998

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (7 * bowling_ball_weight = 4 * canoe_weight) →
    (3 * canoe_weight = 84) →
    bowling_ball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3829_382998


namespace NUMINAMATH_CALUDE_unique_c_for_unique_solution_l3829_382966

/-- The quadratic equation in x with parameter b -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 3*b + 1/b)*x + c = 0

/-- The statement to be proved -/
theorem unique_c_for_unique_solution :
  ∃! c : ℝ, c ≠ 0 ∧
    ∃! b : ℝ, b > 0 ∧
      (∃! x : ℝ, quadratic b c x) ∧
      c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_unique_solution_l3829_382966


namespace NUMINAMATH_CALUDE_overlap_area_is_75_l3829_382915

-- Define a 30-60-90 triangle
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = hypotenuse / 2
  longLeg_eq : longLeg = shortLeg * Real.sqrt 3

-- Define the overlapping configuration
def overlapArea (t : Triangle30_60_90) : ℝ :=
  t.longLeg * t.longLeg

-- Theorem statement
theorem overlap_area_is_75 (t : Triangle30_60_90) :
  overlapArea t = 75 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_75_l3829_382915


namespace NUMINAMATH_CALUDE_hiking_trip_up_rate_l3829_382944

/-- Represents the hiking trip parameters -/
structure HikingTrip where
  upRate : ℝ  -- Rate of ascent in miles per day
  downRate : ℝ  -- Rate of descent in miles per day
  upTime : ℝ  -- Time taken for ascent in days
  downTime : ℝ  -- Time taken for descent in days
  downDistance : ℝ  -- Distance of the descent route in miles

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.upTime = trip.downTime ∧  -- Same time for each route
  trip.downRate = 1.5 * trip.upRate ∧  -- Down rate is 1.5 times up rate
  trip.upTime = 2 ∧  -- 2 days to go up
  trip.downDistance = 9  -- 9 miles down

theorem hiking_trip_up_rate (trip : HikingTrip) 
  (h : validHikingTrip trip) : trip.upRate = 3 := by
  sorry

end NUMINAMATH_CALUDE_hiking_trip_up_rate_l3829_382944


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3829_382953

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3829_382953


namespace NUMINAMATH_CALUDE_skips_per_second_l3829_382934

def minutes_jumped : ℕ := 10
def total_skips : ℕ := 1800

def seconds_jumped : ℕ := minutes_jumped * 60

theorem skips_per_second : total_skips / seconds_jumped = 3 := by
  sorry

end NUMINAMATH_CALUDE_skips_per_second_l3829_382934


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3829_382963

theorem fraction_meaningful (x : ℝ) : (1 : ℝ) / (x - 4) ≠ 0 ↔ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3829_382963


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3829_382982

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 3*x + 28

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -4 ∨ x ≥ 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≤ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3829_382982


namespace NUMINAMATH_CALUDE_peter_double_harriet_age_l3829_382903

def mother_age : ℕ := 60
def harriet_age : ℕ := 13

def peter_age : ℕ := mother_age / 2

def years_until_double (x : ℕ) : Prop :=
  peter_age + x = 2 * (harriet_age + x)

theorem peter_double_harriet_age :
  ∃ x : ℕ, years_until_double x ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_peter_double_harriet_age_l3829_382903


namespace NUMINAMATH_CALUDE_minimum_spend_equal_fruits_l3829_382920

/-- Represents a fruit set with apples, oranges, and cost -/
structure FruitSet where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- Calculates the total cost of buying multiple fruit sets -/
def totalCost (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.cost * quantity

/-- Calculates the total number of apples in multiple fruit sets -/
def totalApples (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.apples * quantity

/-- Calculates the total number of oranges in multiple fruit sets -/
def totalOranges (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.oranges * quantity

theorem minimum_spend_equal_fruits : 
  let set1 : FruitSet := ⟨3, 15, 360⟩
  let set2 : FruitSet := ⟨20, 5, 500⟩
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    totalApples set1 x + totalApples set2 y = totalOranges set1 x + totalOranges set2 y ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ 
       totalApples set1 a + totalApples set2 b = totalOranges set1 a + totalOranges set2 b) →
      totalCost set1 x + totalCost set2 y ≤ totalCost set1 a + totalCost set2 b ∧
    totalCost set1 x + totalCost set2 y = 3800 :=
by
  sorry


end NUMINAMATH_CALUDE_minimum_spend_equal_fruits_l3829_382920


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3829_382928

theorem quadratic_root_problem (m : ℝ) :
  (2 : ℝ)^2 + 2 + m = 0 → ∃ (x : ℝ), x^2 + x + m = 0 ∧ x ≠ 2 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3829_382928


namespace NUMINAMATH_CALUDE_water_conservation_function_correct_water_conserved_third_year_correct_l3829_382992

/-- Represents the water conservation model for a city's tree planting program. -/
structure WaterConservationModel where
  /-- The number of trees planted annually (in millions) -/
  annual_trees : ℕ
  /-- The initial water conservation in 2009 (in million cubic meters) -/
  initial_conservation : ℕ
  /-- The water conservation by 2015 (in billion cubic meters) -/
  final_conservation : ℚ
  /-- The year considered as the first year -/
  start_year : ℕ
  /-- The year when the forest city construction will be completed -/
  end_year : ℕ

/-- The water conservation function for the city's tree planting program. -/
def water_conservation_function (model : WaterConservationModel) (x : ℚ) : ℚ :=
  (4/3) * x + (5/3)

/-- Theorem stating that the given water conservation function is correct for the model. -/
theorem water_conservation_function_correct (model : WaterConservationModel) 
  (h1 : model.annual_trees = 500)
  (h2 : model.initial_conservation = 300)
  (h3 : model.final_conservation = 11/10)
  (h4 : model.start_year = 2009)
  (h5 : model.end_year = 2015) :
  ∀ x : ℚ, 1 ≤ x ∧ x ≤ 7 →
    water_conservation_function model x = 
      (model.final_conservation - (model.initial_conservation / 1000)) / (model.end_year - model.start_year) * x + 
      (model.initial_conservation / 1000) := by
  sorry

/-- Theorem stating that the water conserved in the third year (2011) is correct. -/
theorem water_conserved_third_year_correct (model : WaterConservationModel) :
  water_conservation_function model 3 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_water_conservation_function_correct_water_conserved_third_year_correct_l3829_382992
