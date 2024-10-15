import Mathlib

namespace NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l1001_100103

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 500) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l1001_100103


namespace NUMINAMATH_CALUDE_amy_haircut_l1001_100113

/-- Given an initial hair length and the amount cut off, calculates the final hair length -/
def final_hair_length (initial_length cut_off : ℕ) : ℕ :=
  initial_length - cut_off

/-- Proves that given an initial hair length of 11 inches and cutting off 4 inches, 
    the resulting hair length is 7 inches -/
theorem amy_haircut : final_hair_length 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_haircut_l1001_100113


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l1001_100178

/-- Represents a rectangular garden with strawberry plants -/
structure StrawberryGarden where
  length : ℝ
  width : ℝ
  plants_per_sqft : ℝ
  strawberries_per_plant : ℝ

/-- Calculates the expected total number of strawberries in the garden -/
def total_strawberries (garden : StrawberryGarden) : ℝ :=
  garden.length * garden.width * garden.plants_per_sqft * garden.strawberries_per_plant

/-- Theorem stating the expected number of strawberries in Carrie's garden -/
theorem carries_strawberry_harvest :
  let garden : StrawberryGarden := {
    length := 10,
    width := 15,
    plants_per_sqft := 5,
    strawberries_per_plant := 12
  }
  total_strawberries garden = 9000 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l1001_100178


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_seven_l1001_100107

theorem sum_of_solutions_is_seven : 
  let f (x : ℝ) := |x^2 - 8*x + 12|
  let g (x : ℝ) := 35/4 - x
  ∃ (a b : ℝ), (f a = g a) ∧ (f b = g b) ∧ (a + b = 7) ∧ 
    (∀ (x : ℝ), (f x = g x) → (x = a ∨ x = b)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_seven_l1001_100107


namespace NUMINAMATH_CALUDE_johns_age_l1001_100116

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l1001_100116


namespace NUMINAMATH_CALUDE_alfred_christmas_shopping_goal_l1001_100156

def christmas_shopping_goal (initial_amount : ℕ) (monthly_savings : ℕ) (months : ℕ) : ℕ :=
  initial_amount + monthly_savings * months

theorem alfred_christmas_shopping_goal :
  christmas_shopping_goal 100 75 12 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_alfred_christmas_shopping_goal_l1001_100156


namespace NUMINAMATH_CALUDE_product_sequence_sum_l1001_100164

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 3 = 16 → b = a - 1 → a + b = 95 := by sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l1001_100164


namespace NUMINAMATH_CALUDE_function_composition_equality_l1001_100131

theorem function_composition_equality (b : ℚ) : 
  let p : ℚ → ℚ := λ x => 3 * x - 5
  let q : ℚ → ℚ := λ x => 4 * x - b
  p (q 3) = 9 → b = 22 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1001_100131


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1001_100181

theorem cubic_roots_sum (m : ℤ) (a b c : ℤ) :
  (∀ x : ℤ, x^3 - 2015*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 100 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1001_100181


namespace NUMINAMATH_CALUDE_eccentricity_ratio_range_l1001_100106

theorem eccentricity_ratio_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e₁ := Real.sqrt (a^2 - b^2) / a
  let e₂ := Real.sqrt (a^2 - b^2) / b
  (e₁ * e₂ < 1) →
  (∃ x, x > Real.sqrt 2 ∧ x < (1 + Real.sqrt 5) / 2 ∧ e₂ / e₁ = x) ∧
  (∀ y, y ≤ Real.sqrt 2 ∨ y ≥ (1 + Real.sqrt 5) / 2 → e₂ / e₁ ≠ y) :=
by sorry


end NUMINAMATH_CALUDE_eccentricity_ratio_range_l1001_100106


namespace NUMINAMATH_CALUDE_fraction_zero_value_l1001_100138

theorem fraction_zero_value (x : ℝ) : 
  (|x| - 3) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_value_l1001_100138


namespace NUMINAMATH_CALUDE_expansion_properties_l1001_100101

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Sum of odd-indexed binomial coefficients for (a + b)^n -/
def sumOddCoeffs (n : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (x + 1/√x)^n -/
def constantTerm (n : ℕ) : ℕ := sorry

/-- Theorem about the expansion of (x + 1/√x)^9 -/
theorem expansion_properties :
  (sumOddCoeffs 9 = 256) ∧ (constantTerm 9 = 84) := by sorry

end NUMINAMATH_CALUDE_expansion_properties_l1001_100101


namespace NUMINAMATH_CALUDE_eliza_walking_distance_l1001_100159

/-- Proves that Eliza walked 4.5 kilometers given the conditions of the problem -/
theorem eliza_walking_distance :
  ∀ (total_time : ℝ) (rollerblade_speed : ℝ) (walk_speed : ℝ) (distance : ℝ),
    total_time = 1.5 →  -- 90 minutes converted to hours
    rollerblade_speed = 12 →
    walk_speed = 4 →
    (distance / rollerblade_speed) + (distance / walk_speed) = total_time →
    distance = 4.5 := by
  sorry

#check eliza_walking_distance

end NUMINAMATH_CALUDE_eliza_walking_distance_l1001_100159


namespace NUMINAMATH_CALUDE_seven_division_theorem_l1001_100186

/-- Given a natural number n, returns the sum of its digits. -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of digits in n. -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number consists only of the digit 7. -/
def all_sevens (n : ℕ) : Prop := sorry

theorem seven_division_theorem (N : ℕ) :
  digit_sum N = 2021 →
  ∃ q : ℕ, N = 7 * q ∧ all_sevens q →
  num_digits q = 503 := by sorry

end NUMINAMATH_CALUDE_seven_division_theorem_l1001_100186


namespace NUMINAMATH_CALUDE_tom_payment_l1001_100168

/-- The total amount Tom paid to the shopkeeper -/
def total_amount (apple_quantity apple_rate mango_quantity mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1145 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l1001_100168


namespace NUMINAMATH_CALUDE_calculate_expression_l1001_100189

theorem calculate_expression : (-2022)^0 - 2 * Real.tan (π/4) + |-2| + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1001_100189


namespace NUMINAMATH_CALUDE_modified_ohara_triple_49_64_l1001_100119

/-- Definition of a Modified O'Hara triple -/
def isModifiedOHaraTriple (a b x : ℕ+) : Prop :=
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = (x.val : ℝ)^2

/-- Theorem: If (49, 64, x) is a Modified O'Hara triple, then x = √113 -/
theorem modified_ohara_triple_49_64 (x : ℕ+) :
  isModifiedOHaraTriple 49 64 x → x.val = Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_49_64_l1001_100119


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l1001_100102

theorem quadratic_point_m_value (a m : ℝ) :
  a > 0 →
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l1001_100102


namespace NUMINAMATH_CALUDE_lower_variance_more_stable_l1001_100126

/-- Represents a set of data -/
structure DataSet where
  variance : ℝ

/-- Defines the stability relation between two data sets -/
def more_stable (a b : DataSet) : Prop := a.variance < b.variance

/-- Theorem stating that a data set with lower variance is more stable -/
theorem lower_variance_more_stable (A B : DataSet) 
  (hA : A.variance = 0.01) (hB : B.variance = 0.1) : 
  more_stable A B := by
  sorry

#check lower_variance_more_stable

end NUMINAMATH_CALUDE_lower_variance_more_stable_l1001_100126


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l1001_100160

def blue_plates : ℕ := 6
def red_plates : ℕ := 3
def green_plates : ℕ := 2
def yellow_plates : ℕ := 2

def total_plates : ℕ := blue_plates + red_plates + green_plates + yellow_plates

def circular_arrangements (n : ℕ) (k : List ℕ) : ℕ :=
  Nat.factorial (n - 1) / (k.map Nat.factorial).prod

theorem plate_arrangement_theorem :
  let total_arrangements := circular_arrangements total_plates [blue_plates, red_plates, green_plates, yellow_plates]
  let green_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, 1, yellow_plates]
  let yellow_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, green_plates, 1]
  let both_adjacent := circular_arrangements (total_plates - 2) [blue_plates, red_plates, 1, 1]
  total_arrangements - green_adjacent - yellow_adjacent + both_adjacent = 50400 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l1001_100160


namespace NUMINAMATH_CALUDE_stationery_profit_theorem_l1001_100142

/-- Profit function for a stationery item --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 600 * x - 8000

/-- Daily sales volume function --/
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

/-- Purchase price of the stationery item --/
def purchase_price : ℝ := 20

/-- Theorem stating the properties of the profit function and its maximum --/
theorem stationery_profit_theorem :
  (∀ x, profit_function x = (x - purchase_price) * sales_volume x) ∧
  (∃ x_max, ∀ x, profit_function x ≤ profit_function x_max ∧ x_max = 30) ∧
  (∃ x_constrained, 
    sales_volume x_constrained ≥ 120 ∧
    (∀ x, sales_volume x ≥ 120 → profit_function x ≤ profit_function x_constrained) ∧
    x_constrained = 28 ∧
    profit_function x_constrained = 960) :=
by sorry

end NUMINAMATH_CALUDE_stationery_profit_theorem_l1001_100142


namespace NUMINAMATH_CALUDE_lcm_24_36_42_l1001_100122

theorem lcm_24_36_42 : Nat.lcm 24 (Nat.lcm 36 42) = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_42_l1001_100122


namespace NUMINAMATH_CALUDE_tensor_inequality_implies_a_range_l1001_100193

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem tensor_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → tensor (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_tensor_inequality_implies_a_range_l1001_100193


namespace NUMINAMATH_CALUDE_parabola_parameter_is_two_l1001_100111

/-- Proves that given a hyperbola and a parabola with specific properties, 
    the parameter of the parabola is 2. -/
theorem parabola_parameter_is_two 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (eccentricity : Real.sqrt (1 + b^2 / a^2) = 2)
  (parabola : ∀ x y, y^2 = 2 * p * x)
  (triangle_area : 1/4 * p^2 * b / a = Real.sqrt 3) :
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_is_two_l1001_100111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1001_100197

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) 
    (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1001_100197


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l1001_100175

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l1001_100175


namespace NUMINAMATH_CALUDE_consecutive_product_problem_l1001_100198

theorem consecutive_product_problem :
  let n : ℕ := 77
  let product := n * (n + 1) * (n + 2)
  (product ≥ 100000 ∧ product < 1000000) ∧  -- six-digit number
  (product / 10000 = 47) ∧                  -- left-hand digits are '47'
  (product % 100 = 74)                      -- right-hand digits are '74'
  :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_problem_l1001_100198


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l1001_100196

theorem smallest_n_for_candy (n : ℕ) : (∀ k : ℕ, k > 0 ∧ k < n → ¬(10 ∣ 25*k ∧ 18 ∣ 25*k ∧ 20 ∣ 25*k)) ∧ 
                                       (10 ∣ 25*n ∧ 18 ∣ 25*n ∧ 20 ∣ 25*n) → 
                                       n = 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l1001_100196


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1001_100185

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1001_100185


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1001_100144

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1001_100144


namespace NUMINAMATH_CALUDE_solution_set_intersection_condition_l1001_100163

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem for part (1)
theorem solution_set (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part (2)
theorem intersection_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m y = g x) ↔ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_intersection_condition_l1001_100163


namespace NUMINAMATH_CALUDE_speakers_cost_calculation_l1001_100128

/-- The amount spent on speakers, given the total amount spent on car parts and the amount spent on new tires. -/
def amount_spent_on_speakers (total_spent : ℚ) (tires_cost : ℚ) : ℚ :=
  total_spent - tires_cost

/-- Theorem stating that the amount spent on speakers is $118.54, given the total spent and the cost of tires. -/
theorem speakers_cost_calculation (total_spent tires_cost : ℚ) 
  (h1 : total_spent = 224.87)
  (h2 : tires_cost = 106.33) : 
  amount_spent_on_speakers total_spent tires_cost = 118.54 := by
  sorry

#eval amount_spent_on_speakers 224.87 106.33

end NUMINAMATH_CALUDE_speakers_cost_calculation_l1001_100128


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1001_100117

/-- The number of days Amanda needs to sell tickets -/
def days_to_sell : ℕ := 3

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := 80

/-- The number of tickets sold on day 1 -/
def day1_sales : ℕ := 20

/-- The number of tickets sold on day 2 -/
def day2_sales : ℕ := 32

/-- The number of tickets sold on day 3 -/
def day3_sales : ℕ := 28

/-- Theorem stating that Amanda needs 3 days to sell all tickets -/
theorem amanda_ticket_sales : 
  days_to_sell = 3 ∧ 
  total_tickets = day1_sales + day2_sales + day3_sales := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1001_100117


namespace NUMINAMATH_CALUDE_smallest_number_above_threshold_l1001_100169

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let above_threshold := numbers.filter (λ x => x > threshold)
  above_threshold.minimum? = some 1.2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_above_threshold_l1001_100169


namespace NUMINAMATH_CALUDE_complement_of_A_in_S_l1001_100136

def S : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

theorem complement_of_A_in_S :
  (S \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_S_l1001_100136


namespace NUMINAMATH_CALUDE_christmas_decorations_l1001_100172

theorem christmas_decorations (boxes : ℕ) (used : ℕ) (given_away : ℕ) : 
  boxes = 4 → used = 35 → given_away = 25 → (used + given_away) / boxes = 15 := by
  sorry

end NUMINAMATH_CALUDE_christmas_decorations_l1001_100172


namespace NUMINAMATH_CALUDE_fermats_little_theorem_distinct_colorings_l1001_100171

theorem fermats_little_theorem (p : ℕ) (n : ℤ) (hp : Nat.Prime p) :
  (↑n ^ p - n : ℤ) % ↑p = 0 := by
  sorry

theorem distinct_colorings (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℕ, (n ^ p - n : ℕ) / p + n = k := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_distinct_colorings_l1001_100171


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1001_100109

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) →
  Real.sqrt (1 + (b/a)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1001_100109


namespace NUMINAMATH_CALUDE_barge_length_is_125_steps_l1001_100151

/-- Represents the scenario of Jake walking along a barge on a river -/
structure BargeProblem where
  -- Length of Jake's step upstream
  step_length : ℝ
  -- Length the barge moves while Jake takes one step
  barge_speed : ℝ
  -- Length of the barge
  barge_length : ℝ
  -- Jake walks faster than the barge
  jake_faster : barge_speed < step_length
  -- 300 steps downstream from back to front
  downstream_eq : 300 * (1.5 * step_length) = barge_length + 300 * barge_speed
  -- 60 steps upstream from front to back
  upstream_eq : 60 * step_length = barge_length - 60 * barge_speed

/-- The length of the barge is 125 times Jake's upstream step length -/
theorem barge_length_is_125_steps (p : BargeProblem) : p.barge_length = 125 * p.step_length := by
  sorry


end NUMINAMATH_CALUDE_barge_length_is_125_steps_l1001_100151


namespace NUMINAMATH_CALUDE_harriet_miles_run_l1001_100145

theorem harriet_miles_run (total_miles : ℝ) (katarina_miles : ℝ) (adriana_miles : ℝ) 
  (h1 : total_miles = 285)
  (h2 : katarina_miles = 51)
  (h3 : adriana_miles = 74)
  (h4 : ∃ (x : ℝ), x * 3 + katarina_miles + adriana_miles = total_miles) :
  ∃ (harriet_miles : ℝ), harriet_miles = 53.33 ∧ 
    harriet_miles * 3 + katarina_miles + adriana_miles = total_miles :=
by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_run_l1001_100145


namespace NUMINAMATH_CALUDE_complex_square_equality_l1001_100112

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 15 + 8 * Complex.I →
  c + d * Complex.I = 4 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1001_100112


namespace NUMINAMATH_CALUDE_monotone_decreasing_cubic_l1001_100139

/-- A function f is monotonically decreasing on an open interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem monotone_decreasing_cubic (a : ℝ) :
  MonotonicallyDecreasing (fun x => x^3 - a*x^2 + 1) 0 2 → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_cubic_l1001_100139


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1001_100100

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (3 - i)
  Complex.im z = -1/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1001_100100


namespace NUMINAMATH_CALUDE_sqrt_65_bound_l1001_100174

theorem sqrt_65_bound (n : ℕ+) (h : (n : ℝ) < Real.sqrt 65 ∧ Real.sqrt 65 < (n : ℝ) + 1) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bound_l1001_100174


namespace NUMINAMATH_CALUDE_symmetry_axis_l1001_100190

-- Define a function g with the given symmetry property
def g : ℝ → ℝ := sorry

-- State the symmetry property of g
axiom g_symmetry : ∀ x : ℝ, g x = g (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) g := by sorry

end NUMINAMATH_CALUDE_symmetry_axis_l1001_100190


namespace NUMINAMATH_CALUDE_probability_allison_wins_l1001_100166

structure Cube where
  faces : List ℕ
  valid : faces.length = 6

def allison_cube : Cube := ⟨List.replicate 6 5, rfl⟩
def brian_cube : Cube := ⟨[1, 2, 3, 4, 5, 6], rfl⟩
def noah_cube : Cube := ⟨[2, 2, 2, 6, 6, 6], rfl⟩

def prob_roll_less_than (n : ℕ) (c : Cube) : ℚ :=
  (c.faces.filter (· < n)).length / c.faces.length

theorem probability_allison_wins : 
  prob_roll_less_than 5 brian_cube * prob_roll_less_than 5 noah_cube = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_allison_wins_l1001_100166


namespace NUMINAMATH_CALUDE_benny_spent_34_dollars_l1001_100182

/-- Calculates the amount spent on baseball gear given the initial amount and the amount left over. -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Proves that Benny spent 34 dollars on baseball gear. -/
theorem benny_spent_34_dollars (initial : ℕ) (left_over : ℕ) 
    (h1 : initial = 67) (h2 : left_over = 33) : 
    amount_spent initial left_over = 34 := by
  sorry

#eval amount_spent 67 33

end NUMINAMATH_CALUDE_benny_spent_34_dollars_l1001_100182


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1001_100154

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l1001_100154


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1001_100130

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is approximately 74.094 g/mol -/
theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 74.094| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1001_100130


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_l1001_100195

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem f_increasing_on_negative (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f m x < f m y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_l1001_100195


namespace NUMINAMATH_CALUDE_exists_good_pair_for_all_constructed_pair_is_good_l1001_100188

/-- A pair of natural numbers (m, n) is good if mn and (m+1)(n+1) are perfect squares -/
def is_good_pair (m n : ℕ) : Prop :=
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2

/-- For every natural number m, there exists a good pair (m, n) with n > m -/
theorem exists_good_pair_for_all (m : ℕ) : ∃ n : ℕ, n > m ∧ is_good_pair m n := by
  sorry

/-- The constructed pair (m, m(4m + 3)²) is good for any natural number m -/
theorem constructed_pair_is_good (m : ℕ) : is_good_pair m (m * (4 * m + 3) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_good_pair_for_all_constructed_pair_is_good_l1001_100188


namespace NUMINAMATH_CALUDE_ending_number_proof_l1001_100115

theorem ending_number_proof (start : ℕ) (multiples : ℚ) (end_number : ℕ) : 
  start = 81 → 
  multiples = 93.33333333333333 → 
  end_number = (start + 3 * (multiples.floor - 1)) → 
  end_number = 357 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l1001_100115


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1001_100149

/-- A three-digit number is represented by its digits a, b, and c. -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The product of the digits of a three-digit number. -/
def digit_product (a b c : ℕ) : ℕ := a * b * c

/-- Predicate for a valid three-digit number. -/
def is_valid_three_digit (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- The main theorem: 175 is the only three-digit number that is 5 times the product of its digits. -/
theorem unique_three_digit_number :
  ∀ a b c : ℕ,
    is_valid_three_digit a b c →
    (three_digit_number a b c = 5 * digit_product a b c) →
    (a = 1 ∧ b = 7 ∧ c = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1001_100149


namespace NUMINAMATH_CALUDE_inequality_preservation_l1001_100118

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3)*a - 1 > (1/3)*b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1001_100118


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1001_100108

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  (m - n) / n * 100

/-- Theorem: When a shopkeeper sells 10 articles at the cost price of 12 articles, the profit percentage is 20% -/
theorem shopkeeper_profit : profit_percentage 10 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1001_100108


namespace NUMINAMATH_CALUDE_probability_three_black_balls_l1001_100192

-- Define the number of white and black balls
def white_balls : ℕ := 4
def black_balls : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_all_black : ℚ :=
  (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

-- State the theorem
theorem probability_three_black_balls :
  probability_all_black = 14 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_balls_l1001_100192


namespace NUMINAMATH_CALUDE_b_95_mod_49_l1001_100176

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- The remainder of b_95 when divided by 49 is 36 -/
theorem b_95_mod_49 : b 95 % 49 = 36 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l1001_100176


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1001_100184

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : ∃ (x : ℝ),
  (x + 3) * (24 / 60) = 7.2 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1001_100184


namespace NUMINAMATH_CALUDE_expression_evaluation_l1001_100127

theorem expression_evaluation : (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1001_100127


namespace NUMINAMATH_CALUDE_binomial_sum_equals_higher_binomial_l1001_100161

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem
theorem binomial_sum_equals_higher_binomial :
  binomial 6 3 + binomial 6 2 = binomial 7 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_higher_binomial_l1001_100161


namespace NUMINAMATH_CALUDE_base_7_addition_l1001_100162

/-- Addition in base 7 --/
def add_base_7 (a b : Nat) : Nat :=
  sorry

/-- Conversion from base 10 to base 7 --/
def to_base_7 (n : Nat) : Nat :=
  sorry

/-- Conversion from base 7 to base 10 --/
def from_base_7 (n : Nat) : Nat :=
  sorry

theorem base_7_addition :
  add_base_7 (from_base_7 25) (from_base_7 54) = from_base_7 112 :=
by sorry

end NUMINAMATH_CALUDE_base_7_addition_l1001_100162


namespace NUMINAMATH_CALUDE_shopkeeper_weight_problem_l1001_100147

theorem shopkeeper_weight_problem (actual_weight : ℝ) (profit_percentage : ℝ) :
  actual_weight = 800 →
  profit_percentage = 25 →
  ∃ standard_weight : ℝ,
    standard_weight = 1000 ∧
    (standard_weight - actual_weight) / actual_weight * 100 = profit_percentage :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_weight_problem_l1001_100147


namespace NUMINAMATH_CALUDE_inverse_function_implies_a_value_l1001_100124

def f (a : ℝ) (x : ℝ) : ℝ := a - 2 * x

theorem inverse_function_implies_a_value (a : ℝ) :
  (∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g (-3) = 3) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_implies_a_value_l1001_100124


namespace NUMINAMATH_CALUDE_similar_triangles_proportion_l1001_100157

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_proportion 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {X, Y, Z})
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_ZY : dist Z Y = 32) :
  dist X Y = 16 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_proportion_l1001_100157


namespace NUMINAMATH_CALUDE_dog_groom_time_l1001_100120

/-- The time (in hours) it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The total time (in hours) it takes to groom 5 dogs and 3 cats -/
def total_groom_time : ℝ := 14

/-- The number of dogs groomed -/
def num_dogs : ℕ := 5

/-- The number of cats groomed -/
def num_cats : ℕ := 3

/-- Theorem stating that the time to groom a dog is 2.5 hours -/
theorem dog_groom_time : 
  ∃ (dog_time : ℝ), 
    dog_time * num_dogs + cat_groom_time * num_cats = total_groom_time ∧ 
    dog_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_dog_groom_time_l1001_100120


namespace NUMINAMATH_CALUDE_equation_solution_l1001_100165

theorem equation_solution : ∃! x : ℝ, (x + 1)^63 + (x + 1)^62*(x - 1) + (x + 1)^61*(x - 1)^2 + (x - 1)^63 = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1001_100165


namespace NUMINAMATH_CALUDE_train_crossing_time_l1001_100167

/-- Calculates the time taken for a train to cross a signal post -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 350 → 
  train_speed_kmph = 72 → 
  (train_length / (train_speed_kmph * 1000 / 3600)) = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1001_100167


namespace NUMINAMATH_CALUDE_e_sequence_property_l1001_100129

/-- Definition of an E-sequence -/
def is_e_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → |a (k + 1) - a k| = 1

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → a k < a (k + 1)

theorem e_sequence_property (a : ℕ → ℤ) :
  is_e_sequence a 2000 →
  a 1 = 13 →
  (is_increasing a 2000 ↔ a 2000 = 2012) :=
by sorry

end NUMINAMATH_CALUDE_e_sequence_property_l1001_100129


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1001_100104

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) (h2 : x - y = 5/24) : x^2 - y^2 = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1001_100104


namespace NUMINAMATH_CALUDE_divisible_by_35_l1001_100134

theorem divisible_by_35 (n : ℕ) : ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_35_l1001_100134


namespace NUMINAMATH_CALUDE_fourth_person_height_l1001_100125

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ = h₁ + 2 →                 -- Difference between 1st and 2nd is 2 inches
  h₃ = h₂ + 2 →                 -- Difference between 2nd and 3rd is 2 inches
  h₄ = h₃ + 6 →                 -- Difference between 3rd and 4th is 6 inches
  (h₁ + h₂ + h₃ + h₄) / 4 = 78  -- Average height is 78 inches
  → h₄ = 84 :=                  -- Fourth person's height is 84 inches
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1001_100125


namespace NUMINAMATH_CALUDE_commercial_reduction_percentage_l1001_100194

theorem commercial_reduction_percentage 
  (original_length : ℝ) 
  (shortened_length : ℝ) 
  (h1 : original_length = 30) 
  (h2 : shortened_length = 21) : 
  (original_length - shortened_length) / original_length * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_commercial_reduction_percentage_l1001_100194


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1001_100187

theorem average_of_remaining_numbers 
  (n : ℕ) 
  (total_avg : ℚ) 
  (subset_sum : ℚ) 
  (h1 : n = 5) 
  (h2 : total_avg = 20) 
  (h3 : subset_sum = 48) : 
  ((n : ℚ) * total_avg - subset_sum) / ((n : ℚ) - 3) = 26 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1001_100187


namespace NUMINAMATH_CALUDE_pizza_order_problem_l1001_100110

theorem pizza_order_problem (slices_per_pizza : ℕ) (james_fraction : ℚ) (james_slices : ℕ) :
  slices_per_pizza = 6 →
  james_fraction = 2 / 3 →
  james_slices = 8 →
  (james_slices : ℚ) / james_fraction / slices_per_pizza = 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_order_problem_l1001_100110


namespace NUMINAMATH_CALUDE_investment_in_bank_a_l1001_100191

def total_investment : ℝ := 1500
def bank_a_rate : ℝ := 0.04
def bank_b_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1740.54

theorem investment_in_bank_a (x : ℝ) :
  x * (1 + bank_a_rate) ^ years + (total_investment - x) * (1 + bank_b_rate) ^ years = final_amount →
  x = 695 := by
sorry

end NUMINAMATH_CALUDE_investment_in_bank_a_l1001_100191


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l1001_100143

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = x^2 * (y^2 + y - 6) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l1001_100143


namespace NUMINAMATH_CALUDE_no_other_products_of_three_primes_l1001_100133

/-- The reverse of a natural number -/
def reverse (n : ℕ) : ℕ := sorry

/-- Predicate for a number being the product of exactly three distinct primes -/
def isProductOfThreeDistinctPrimes (n : ℕ) : Prop := sorry

theorem no_other_products_of_three_primes : 
  let original := 2017
  let reversed := 7102
  -- 7102 is the reverse of 2017
  reverse original = reversed →
  -- 7102 is the product of three distinct primes p, q, and r
  ∃ (p q r : ℕ), isProductOfThreeDistinctPrimes reversed ∧ 
                 reversed = p * q * r ∧ 
                 p ≠ q ∧ p ≠ r ∧ q ≠ r →
  -- There are no other positive integers that are products of three distinct primes 
  -- summing to the same value as p + q + r
  ¬∃ (n : ℕ), n ≠ reversed ∧ 
              isProductOfThreeDistinctPrimes n ∧
              (∃ (p1 p2 p3 : ℕ), n = p1 * p2 * p3 ∧ 
                                 p1 + p2 + p3 = p + q + r ∧
                                 p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
by sorry

end NUMINAMATH_CALUDE_no_other_products_of_three_primes_l1001_100133


namespace NUMINAMATH_CALUDE_paint_needed_for_columns_l1001_100150

-- Define constants
def num_columns : ℕ := 20
def column_height : ℝ := 20
def column_diameter : ℝ := 10
def paint_coverage : ℝ := 350

-- Theorem statement
theorem paint_needed_for_columns :
  ∃ (gallons : ℕ),
    gallons * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) ∧
    ∀ (g : ℕ), g * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) → g ≥ gallons :=
by sorry

end NUMINAMATH_CALUDE_paint_needed_for_columns_l1001_100150


namespace NUMINAMATH_CALUDE_inequality_implies_sum_l1001_100152

/-- Given that (x-a)(x-b)/(x-c) ≤ 0 if and only if x < -6 or |x-30| ≤ 2, and a < b,
    prove that a + 2b + 3c = 74 -/
theorem inequality_implies_sum (a b c : ℝ) :
  (∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  a < b →
  a + 2*b + 3*c = 74 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_l1001_100152


namespace NUMINAMATH_CALUDE_natasha_hill_climbing_l1001_100135

/-- Natasha's hill climbing problem -/
theorem natasha_hill_climbing
  (time_up : ℝ)
  (time_down : ℝ)
  (avg_speed_total : ℝ)
  (h_time_up : time_up = 4)
  (h_time_down : time_down = 2)
  (h_avg_speed_total : avg_speed_total = 1.5) :
  let distance := avg_speed_total * (time_up + time_down) / 2
  let avg_speed_up := distance / time_up
  avg_speed_up = 1.125 := by
sorry

end NUMINAMATH_CALUDE_natasha_hill_climbing_l1001_100135


namespace NUMINAMATH_CALUDE_vector_dot_product_l1001_100180

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (1, -3) ∧ a - b = (3, 7) → a • b = -12 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_l1001_100180


namespace NUMINAMATH_CALUDE_unique_line_configuration_l1001_100177

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The specific configuration described in the problem -/
def problem_config : LineConfiguration :=
  { n := 8,  -- This is what we want to prove
    total_intersections := 16,
    triple_intersections := 6 }

/-- Theorem stating that the problem configuration is the only valid one -/
theorem unique_line_configuration :
  ∀ (config : LineConfiguration),
    (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) →  -- every pair of lines intersects
    (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n → 
      i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
      ¬∃ (p : ℕ), p < config.total_intersections) →  -- no four lines pass through a single point
    config.total_intersections = 16 →
    config.triple_intersections = 6 →
    config = problem_config :=
by sorry

end NUMINAMATH_CALUDE_unique_line_configuration_l1001_100177


namespace NUMINAMATH_CALUDE_midpoint_pentagon_perimeter_l1001_100114

/-- A convex pentagon in a 2D plane. -/
structure ConvexPentagon where
  -- We don't need to define the specific properties of a convex pentagon for this problem

/-- The sum of all diagonals of a convex pentagon. -/
def sum_of_diagonals (p : ConvexPentagon) : ℝ := sorry

/-- The pentagon formed by connecting the midpoints of the sides of a convex pentagon. -/
def midpoint_pentagon (p : ConvexPentagon) : ConvexPentagon := sorry

/-- The perimeter of a pentagon. -/
def perimeter (p : ConvexPentagon) : ℝ := sorry

/-- 
Theorem: The perimeter of the pentagon formed by connecting the midpoints 
of the sides of a convex pentagon is equal to half the sum of all diagonals 
of the original pentagon.
-/
theorem midpoint_pentagon_perimeter (p : ConvexPentagon) : 
  perimeter (midpoint_pentagon p) = (1/2) * sum_of_diagonals p := by
  sorry

end NUMINAMATH_CALUDE_midpoint_pentagon_perimeter_l1001_100114


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1001_100199

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1001_100199


namespace NUMINAMATH_CALUDE_max_t_value_l1001_100148

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  r ≤ 17 →
  t ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l1001_100148


namespace NUMINAMATH_CALUDE_farmer_crops_after_pest_destruction_l1001_100132

-- Define the constants
def corn_cobs_per_row : ℕ := 9
def potatoes_per_row : ℕ := 30
def corn_rows : ℕ := 10
def potato_rows : ℕ := 5
def pest_destruction_ratio : ℚ := 1/2

-- Define the theorem
theorem farmer_crops_after_pest_destruction :
  (corn_rows * corn_cobs_per_row + potato_rows * potatoes_per_row) * pest_destruction_ratio = 120 := by
  sorry

end NUMINAMATH_CALUDE_farmer_crops_after_pest_destruction_l1001_100132


namespace NUMINAMATH_CALUDE_decimal_representation_theorem_l1001_100173

theorem decimal_representation_theorem (n m : ℕ) (h1 : n > m) (h2 : m ≥ 1) 
  (h3 : ∃ k : ℕ, ∃ p : ℕ, 0 < p ∧ p < n ∧ 
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 ≥ 143 ∧
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 < 144) :
  n > 125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_theorem_l1001_100173


namespace NUMINAMATH_CALUDE_first_pay_cut_percentage_l1001_100153

theorem first_pay_cut_percentage 
  (overall_decrease : Real) 
  (second_cut : Real) 
  (third_cut : Real) 
  (h1 : overall_decrease = 27.325)
  (h2 : second_cut = 10)
  (h3 : third_cut = 15) : 
  ∃ (first_cut : Real), 
    first_cut = 5 ∧ 
    (1 - overall_decrease / 100) = 
    (1 - first_cut / 100) * (1 - second_cut / 100) * (1 - third_cut / 100) := by
  sorry


end NUMINAMATH_CALUDE_first_pay_cut_percentage_l1001_100153


namespace NUMINAMATH_CALUDE_fraction2012_is_16_45_l1001_100121

/-- Represents a fraction in the sequence -/
structure Fraction :=
  (numerator : Nat)
  (denominator : Nat)
  (h1 : numerator ≤ denominator / 2)
  (h2 : numerator > 0)
  (h3 : denominator > 0)

/-- The sequence of fractions not exceeding 1/2 -/
def fractionSequence : Nat → Fraction := sorry

/-- The 2012th fraction in the sequence -/
def fraction2012 : Fraction := fractionSequence 2012

/-- Theorem stating that the 2012th fraction is 16/45 -/
theorem fraction2012_is_16_45 :
  fraction2012.numerator = 16 ∧ fraction2012.denominator = 45 := by sorry

end NUMINAMATH_CALUDE_fraction2012_is_16_45_l1001_100121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1001_100170

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_sum : a 3 + a 5 = 8) :
  a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l1001_100170


namespace NUMINAMATH_CALUDE_constant_function_shifted_l1001_100179

-- Define g as a function from real numbers to real numbers
def g : ℝ → ℝ := fun _ ↦ -3

-- Theorem statement
theorem constant_function_shifted (x : ℝ) : g (x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_shifted_l1001_100179


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1001_100141

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - k = 0 ∧ x = 1) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1001_100141


namespace NUMINAMATH_CALUDE_ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l1001_100140

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 144 + y^2 / 169 = 1
def ellipse2 (x y m : ℝ) : Prop := x^2 / m^2 + y^2 / (m^2 + 1) = 1
def ellipse3 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1
def ellipse4 (x y m : ℝ) : Prop := x^2 / (m - 5) + y^2 / (m + 4) = 1

-- Define focal points
def focal_points (a b : ℝ) : Set (ℝ × ℝ) := {(-a, 0), (a, 0)} ∪ {(0, -b), (0, b)}

-- Theorem statements
theorem ellipse1_focal_points :
  ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 5 ∧ 
  ∀ (x y : ℝ), ellipse1 x y → (x, y) ∈ f := sorry

theorem ellipse2_focal_points :
  ∀ (m : ℝ), ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 1 ∧ 
  ∀ (x y : ℝ), ellipse2 x y m → (x, y) ∈ f := sorry

theorem ellipses_different_focal_points :
  ∀ (m : ℝ), m > 0 →
  ¬∃ (f : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), ellipse3 x y → (x, y) ∈ f) ∧
    (∀ (x y : ℝ), ellipse4 x y m → (x, y) ∈ f) := sorry

end NUMINAMATH_CALUDE_ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l1001_100140


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l1001_100123

theorem set_equality_implies_sum (m n : ℝ) : 
  let P : Set ℝ := {m / n, 1}
  let Q : Set ℝ := {n, 0}
  P = Q → m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l1001_100123


namespace NUMINAMATH_CALUDE_correct_purchase_ways_l1001_100183

def num_cookie_types : ℕ := 7
def num_cupcake_types : ℕ := 4
def total_items : ℕ := 4

def purchase_ways : ℕ := sorry

theorem correct_purchase_ways : purchase_ways = 4054 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_ways_l1001_100183


namespace NUMINAMATH_CALUDE_eldest_age_l1001_100105

theorem eldest_age (a b c d : ℕ) : 
  (∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 8 * x ∧ d = 9 * x) →  -- ages are in ratio 5:7:8:9
  (a - 10) + (b - 10) + (c - 10) + (d - 10) = 107 →             -- sum of ages 10 years ago
  d = 45                                                        -- present age of eldest
  := by sorry

end NUMINAMATH_CALUDE_eldest_age_l1001_100105


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1001_100146

/-- Given a geometric sequence {a_n} where a_3 + a_7 = 5, 
    prove that a_2a_4 + 2a_4a_6 + a_6a_8 = 25 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum : a 3 + a 7 = 5) :
    a 2 * a 4 + 2 * a 4 * a 6 + a 6 * a 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1001_100146


namespace NUMINAMATH_CALUDE_coefficient_of_negative_six_xy_l1001_100158

/-- The coefficient of a monomial is the numeric factor that multiplies the variable parts. -/
def coefficient (m : ℤ) (x : String) (y : String) : ℤ := m

theorem coefficient_of_negative_six_xy :
  coefficient (-6) "x" "y" = -6 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_negative_six_xy_l1001_100158


namespace NUMINAMATH_CALUDE_amy_garden_seeds_l1001_100137

theorem amy_garden_seeds (initial_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : initial_seeds = 101)
  (h2 : big_garden_seeds = 47)
  (h3 : small_gardens = 9) :
  (initial_seeds - big_garden_seeds) / small_gardens = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_garden_seeds_l1001_100137


namespace NUMINAMATH_CALUDE_middle_term_is_36_l1001_100155

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℝ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val →
    a j - a i = a k - a j)

/-- The theorem stating that the middle term of the arithmetic sequence is 36 -/
theorem middle_term_is_36 (seq : ArithmeticSequence)
  (h1 : seq.a 0 = 11)
  (h2 : seq.a 6 = 61) :
  seq.a 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_is_36_l1001_100155
