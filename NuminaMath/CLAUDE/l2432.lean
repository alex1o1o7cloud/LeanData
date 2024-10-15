import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l2432_243290

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator : ℚ := 5 + x * (5 + x) - 5^2
  let denominator : ℚ := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2432_243290


namespace NUMINAMATH_CALUDE_construction_equation_correct_l2432_243240

/-- Represents a construction project with a work stoppage -/
structure ConstructionProject where
  totalLength : ℝ
  originalDailyRate : ℝ
  workStoppageDays : ℝ
  increasedDailyRate : ℝ

/-- The equation correctly represents the construction project situation -/
theorem construction_equation_correct (project : ConstructionProject) 
  (h1 : project.totalLength = 2000)
  (h2 : project.workStoppageDays = 3)
  (h3 : project.increasedDailyRate = project.originalDailyRate + 40) :
  project.totalLength / project.originalDailyRate - 
  project.totalLength / project.increasedDailyRate = 
  project.workStoppageDays := by
sorry

end NUMINAMATH_CALUDE_construction_equation_correct_l2432_243240


namespace NUMINAMATH_CALUDE_expand_quadratic_l2432_243293

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end NUMINAMATH_CALUDE_expand_quadratic_l2432_243293


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2432_243281

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    aₙ₊₁ = r * aₙ for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) :
  a 1 * a 2 * a 3 = -8 → a 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2432_243281


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l2432_243266

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l2432_243266


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2432_243285

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let x₁ := to_base_10 [3, 5, 2] 8
  let y₁ := to_base_10 [3, 1] 4
  let x₂ := to_base_10 [2, 3, 1] 5
  let y₂ := to_base_10 [3, 2] 3
  (x₁ : ℚ) / y₁ + (x₂ : ℚ) / y₂ = 28.67 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2432_243285


namespace NUMINAMATH_CALUDE_tangerines_left_l2432_243291

def total_tangerines : ℕ := 27
def eaten_tangerines : ℕ := 18

theorem tangerines_left : total_tangerines - eaten_tangerines = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l2432_243291


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l2432_243299

/-- Calculates the profit John makes from selling popcorn bags -/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := (total_bags : ℚ) * discounted_price
  let total_revenue : ℚ := (adult_bags : ℚ) * adult_price + (child_bags : ℚ) * child_price
  let profit : ℚ := total_revenue - total_cost
  profit = 112 :=
by
  sorry


end NUMINAMATH_CALUDE_john_popcorn_profit_l2432_243299


namespace NUMINAMATH_CALUDE_set_A_at_most_one_element_l2432_243282

theorem set_A_at_most_one_element (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a ≥ 9/8 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_set_A_at_most_one_element_l2432_243282


namespace NUMINAMATH_CALUDE_periodic_function_l2432_243260

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end NUMINAMATH_CALUDE_periodic_function_l2432_243260


namespace NUMINAMATH_CALUDE_equal_chord_length_l2432_243217

/-- Given a circle C and two lines l1 and l2, prove that they intercept chords of equal length on C -/
theorem equal_chord_length (r d : ℝ) (h : r > 0) :
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let l1 := {p : ℝ × ℝ | 2 * p.1 + 3 * p.2 + 1 = 0}
  let l2 := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}
  let chord_length (l : Set (ℝ × ℝ)) := 
    Real.sqrt (4 * r^2 - 4 * (1 / (2^2 + 3^2)))
  chord_length l1 = d → chord_length l2 = d :=
by sorry

end NUMINAMATH_CALUDE_equal_chord_length_l2432_243217


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l2432_243264

/-- Represents the investment amounts of Vishal, Trishul, and Raghu -/
structure Investments where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The conditions of the investment problem -/
def InvestmentConditions (i : Investments) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.raghu = 2300 ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investments) 
  (h : InvestmentConditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l2432_243264


namespace NUMINAMATH_CALUDE_evaluate_expression_l2432_243295

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 + 13) = -763 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2432_243295


namespace NUMINAMATH_CALUDE_smallest_base_for_80_l2432_243222

theorem smallest_base_for_80 :
  ∀ b : ℕ, b ≥ 5 → b^2 ≤ 80 ∧ 80 < b^3 →
  ∀ c : ℕ, c < 5 → ¬(c^2 ≤ 80 ∧ 80 < c^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_80_l2432_243222


namespace NUMINAMATH_CALUDE_scientific_notation_1500_l2432_243284

theorem scientific_notation_1500 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500 = a * (10 : ℝ) ^ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1500_l2432_243284


namespace NUMINAMATH_CALUDE_taxi_seating_arrangements_l2432_243258

theorem taxi_seating_arrangements :
  let n : ℕ := 6  -- total number of people
  let m : ℕ := 4  -- maximum capacity of each taxi
  let k : ℕ := 2  -- number of taxis
  Nat.choose n m * 2 + (Nat.choose n (n / k)) = 50 :=
by sorry

end NUMINAMATH_CALUDE_taxi_seating_arrangements_l2432_243258


namespace NUMINAMATH_CALUDE_albert_number_puzzle_l2432_243261

theorem albert_number_puzzle (n : ℕ) : 
  (1 : ℚ) / n + (1 : ℚ) / 2 = (1 : ℚ) / 3 + (2 : ℚ) / (n + 1) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_albert_number_puzzle_l2432_243261


namespace NUMINAMATH_CALUDE_jackson_chairs_l2432_243230

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (tables_with_4_seats tables_with_6_seats : ℕ) : ℕ :=
  tables_with_4_seats * 4 + tables_with_6_seats * 6

/-- Proof that Jackson needs to buy 96 chairs -/
theorem jackson_chairs : total_chairs 6 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jackson_chairs_l2432_243230


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2432_243288

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2432_243288


namespace NUMINAMATH_CALUDE_expression_value_l2432_243253

theorem expression_value (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2432_243253


namespace NUMINAMATH_CALUDE_gcd_sum_lcm_eq_gcd_l2432_243280

theorem gcd_sum_lcm_eq_gcd (a b : ℤ) : Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_lcm_eq_gcd_l2432_243280


namespace NUMINAMATH_CALUDE_sin_50_cos_80_cos_160_l2432_243205

theorem sin_50_cos_80_cos_160 :
  Real.sin (50 * π / 180) * Real.cos (80 * π / 180) * Real.cos (160 * π / 180) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_sin_50_cos_80_cos_160_l2432_243205


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2432_243216

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > 0 → a * b ≥ 0) ↔ (∃ (a b : ℝ), a > 0 ∧ a * b < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2432_243216


namespace NUMINAMATH_CALUDE_original_time_calculation_l2432_243212

theorem original_time_calculation (original_speed : ℝ) (original_time : ℝ) 
  (h1 : original_speed > 0) (h2 : original_time > 0) : 
  (original_time / 0.8 = original_time + 10) → original_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_time_calculation_l2432_243212


namespace NUMINAMATH_CALUDE_bread_lasts_three_days_l2432_243250

/-- Represents the number of days bread will last for a household -/
def days_bread_lasts (
  household_members : ℕ)
  (breakfast_slices_per_member : ℕ)
  (snack_slices_per_member : ℕ)
  (slices_per_loaf : ℕ)
  (number_of_loaves : ℕ) : ℕ :=
  let total_slices := number_of_loaves * slices_per_loaf
  let daily_consumption := household_members * (breakfast_slices_per_member + snack_slices_per_member)
  total_slices / daily_consumption

/-- Theorem stating that 5 loaves of bread will last 3 days for a family of 4 -/
theorem bread_lasts_three_days :
  days_bread_lasts 4 3 2 12 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bread_lasts_three_days_l2432_243250


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2432_243267

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2432_243267


namespace NUMINAMATH_CALUDE_work_completion_time_l2432_243208

theorem work_completion_time (b_alone : ℝ) (together_time : ℝ) (b_remaining : ℝ) 
  (h1 : b_alone = 28)
  (h2 : together_time = 3)
  (h3 : b_remaining = 21) :
  ∃ a_alone : ℝ, 
    a_alone = 21 ∧ 
    together_time * (1 / a_alone + 1 / b_alone) + b_remaining * (1 / b_alone) = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2432_243208


namespace NUMINAMATH_CALUDE_expression_value_l2432_243247

theorem expression_value (x y : ℝ) (h : x - y = 1) : 3*x - 3*y + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2432_243247


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2432_243234

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 4 and a₄ = 2, a₆ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2432_243234


namespace NUMINAMATH_CALUDE_purchase_in_fourth_month_l2432_243225

/-- Represents the financial state of the family --/
structure FamilyFinance where
  monthlyIncome : ℕ
  monthlyExpenses : ℕ
  initialSavings : ℕ
  furnitureCost : ℕ

/-- Calculates the month when the family can make the purchase --/
def purchaseMonth (finance : FamilyFinance) : ℕ :=
  let monthlySavings := finance.monthlyIncome - finance.monthlyExpenses
  let additionalRequired := finance.furnitureCost - finance.initialSavings
  (additionalRequired + monthlySavings - 1) / monthlySavings + 1

/-- The main theorem stating that the family can make the purchase in the 4th month --/
theorem purchase_in_fourth_month (finance : FamilyFinance) 
  (h1 : finance.monthlyIncome = 150000)
  (h2 : finance.monthlyExpenses = 115000)
  (h3 : finance.initialSavings = 45000)
  (h4 : finance.furnitureCost = 127000) :
  purchaseMonth finance = 4 := by
  sorry

#eval purchaseMonth { 
  monthlyIncome := 150000, 
  monthlyExpenses := 115000, 
  initialSavings := 45000, 
  furnitureCost := 127000 
}

end NUMINAMATH_CALUDE_purchase_in_fourth_month_l2432_243225


namespace NUMINAMATH_CALUDE_ruby_height_l2432_243297

/-- Given the heights of various people, prove Ruby's height --/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

#check ruby_height

end NUMINAMATH_CALUDE_ruby_height_l2432_243297


namespace NUMINAMATH_CALUDE_solution_difference_l2432_243206

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 26 * p - 130) →
  ((q - 5) * (q + 5) = 26 * q - 130) →
  p ≠ q →
  p > q →
  p - q = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2432_243206


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l2432_243283

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l2432_243283


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2432_243241

theorem solution_set_inequality (x : ℝ) : 
  x * |x + 2| < 0 ↔ x < -2 ∨ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2432_243241


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l2432_243203

/-- Represents a rectangular roof -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular roof with length 4 times the width and area 784 sq ft,
    the difference between length and width is 42 ft -/
theorem roof_length_width_difference 
  (roof : RectangularRoof)
  (h1 : roof.length = 4 * roof.width)
  (h2 : roof.area = 784)
  (h3 : roof.area = roof.length * roof.width) :
  roof.length - roof.width = 42 := by
  sorry


end NUMINAMATH_CALUDE_roof_length_width_difference_l2432_243203


namespace NUMINAMATH_CALUDE_positive_solution_sum_l2432_243259

theorem positive_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 93 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 123 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_sum_l2432_243259


namespace NUMINAMATH_CALUDE_product_xyz_equals_2898_l2432_243271

theorem product_xyz_equals_2898 (x y z : ℝ) 
  (eq1 : -3*x + 4*y - z = 28)
  (eq2 : 3*x - 2*y + z = 8)
  (eq3 : x + y - z = 2) :
  x * y * z = 2898 := by sorry

end NUMINAMATH_CALUDE_product_xyz_equals_2898_l2432_243271


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2432_243244

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P(x₀, y₀) on its right branch such that the difference between
    its distances to the left and right foci is 8, and the product of its
    distances to the two asymptotes is 16/5, prove that the eccentricity of
    the hyperbola is √5/2. -/
theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0)
    (heq : x₀^2 / a^2 - y₀^2 / b^2 = 1)
    (hright : x₀ > 0)
    (hfoci : 2 * a = 8)
    (hasymptotes : (b * x₀ - a * y₀) * (b * x₀ + a * y₀) / (a^2 + b^2) = 16/5) :
    let c := Real.sqrt (a^2 + b^2)
    c / a = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2432_243244


namespace NUMINAMATH_CALUDE_unspent_portion_after_transfer_l2432_243269

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  sorry

/-- Theorem stating the unspent portion of the platinum card's limit after transfer -/
theorem unspent_portion_after_transfer
  (gold : CreditCard)
  (platinum : CreditCard)
  (h1 : platinum.limit = 2 * gold.limit)
  (h2 : ∃ X : ℝ, gold.balance = X * gold.limit)
  (h3 : platinum.balance = (1 / 7) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (12 - 7 * (gold.balance / gold.limit)) / 14 :=
  sorry

end NUMINAMATH_CALUDE_unspent_portion_after_transfer_l2432_243269


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l2432_243233

/-- Represents a 4x4x4 cube composed of smaller cubes -/
structure Cube4x4x4 where
  total_cubes : Nat
  face_size : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in a 4x4x4 cube -/
def count_shaded_cubes (cube : Cube4x4x4) : Nat :=
  sorry

/-- Theorem stating that 24 cubes are shaded on at least one face -/
theorem shaded_cubes_count (cube : Cube4x4x4) 
  (h1 : cube.total_cubes = 64)
  (h2 : cube.face_size = 4)
  (h3 : cube.shaded_per_face = 8) : 
  count_shaded_cubes cube = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l2432_243233


namespace NUMINAMATH_CALUDE_range_of_m_l2432_243232

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ m > 1/3

def q (m : ℝ) : Prop := ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 0 < m ∧ m < 15

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1/3 ≤ m ∧ m < 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2432_243232


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l2432_243270

/-- Represents a 3-digit number abc where a, b, c are single digits --/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_positive : a ≥ 1
  sum_constraint : a + b + c = 8

/-- Represents the odometer readings for Denise's trip --/
structure OdometerReadings where
  initial : ThreeDigitNumber
  final : ThreeDigitNumber
  final_swap : final.a = initial.b ∧ final.b = initial.a ∧ final.c = initial.c

/-- Represents Denise's trip --/
structure Trip where
  readings : OdometerReadings
  hours : ℕ
  hours_positive : hours > 0
  speed : ℕ
  speed_eq : speed = 48
  distance_constraint : 90 * (readings.initial.b - readings.initial.a) = hours * speed

theorem odometer_sum_squares (t : Trip) : 
  t.readings.initial.a ^ 2 + t.readings.initial.b ^ 2 + t.readings.initial.c ^ 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l2432_243270


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2432_243236

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 2340) → (180 * ((n - 3) - 2) = 1800) := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2432_243236


namespace NUMINAMATH_CALUDE_test_total_points_l2432_243201

theorem test_total_points (total_questions : ℕ) (four_point_questions : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10) :
  let two_point_questions := total_questions - four_point_questions
  total_questions * 2 + four_point_questions * 2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_test_total_points_l2432_243201


namespace NUMINAMATH_CALUDE_atop_difference_l2432_243294

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y + x - y

-- Theorem statement
theorem atop_difference : (atop 7 4) - (atop 4 7) = 6 := by
  sorry

end NUMINAMATH_CALUDE_atop_difference_l2432_243294


namespace NUMINAMATH_CALUDE_complex_magnitude_l2432_243287

/-- Given a complex number z = (3+i)/(1+2i), prove that its magnitude |z| is equal to √2 -/
theorem complex_magnitude (z : ℂ) : z = (3 + I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2432_243287


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2432_243262

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + 5 * x^2 + 16) - (x^6 + 4 * x^5 - 2 * x^3 + 3 * x^2 + 18) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 + 2 * x^2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2432_243262


namespace NUMINAMATH_CALUDE_green_blue_difference_l2432_243278

/-- Represents the number of beads of each color in Sue's necklace -/
structure BeadCount where
  purple : Nat
  blue : Nat
  green : Nat

/-- The conditions of Sue's necklace -/
def sueNecklace : BeadCount where
  purple := 7
  blue := 2 * 7
  green := 46 - (7 + 2 * 7)

theorem green_blue_difference :
  sueNecklace.green - sueNecklace.blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l2432_243278


namespace NUMINAMATH_CALUDE_inequality_proof_l2432_243279

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 ∧
  ((a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3 / 2 ↔
   a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2432_243279


namespace NUMINAMATH_CALUDE_cricket_matches_played_l2432_243289

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

theorem cricket_matches_played 
  (rahul ankit : CricketPlayer)
  (h1 : batting_average rahul = 46)
  (h2 : batting_average ankit = 52)
  (h3 : batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54)
  (h4 : ∃ x : ℕ, 
        batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54 ∧
        batting_average {matches_played := ankit.matches_played + 1, 
                         total_runs := ankit.total_runs + x} = 54) :
  rahul.matches_played = 3 ∧ ankit.matches_played = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_matches_played_l2432_243289


namespace NUMINAMATH_CALUDE_mean_equality_problem_l2432_243223

theorem mean_equality_problem (z : ℝ) : 
  (7 + 11 + 5 + 9) / 4 = (15 + z) / 2 → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l2432_243223


namespace NUMINAMATH_CALUDE_square_area_is_36_l2432_243272

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareInPlane where
  -- Define the y-coordinates of the vertices
  y1 : ℝ := 0
  y2 : ℝ := 3
  y3 : ℝ := 0
  y4 : ℝ := -3

/-- The area of the square -/
def squareArea (s : SquareInPlane) : ℝ := 36

/-- Theorem: The area of the square with given y-coordinates is 36 -/
theorem square_area_is_36 (s : SquareInPlane) : squareArea s = 36 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_36_l2432_243272


namespace NUMINAMATH_CALUDE_equal_area_line_coeff_sum_l2432_243238

/-- A region formed by eight unit circles packed in the first quadrant --/
def R : Set (ℝ × ℝ) :=
  sorry

/-- A line with slope 3 that divides R into two equal areas --/
def l : Set (ℝ × ℝ) :=
  sorry

/-- The line l expressed in the form ax = by + c --/
def line_equation (a b c : ℕ) : Prop :=
  ∀ x y, (x, y) ∈ l ↔ a * x = b * y + c

/-- The coefficients a, b, and c are positive integers with gcd 1 --/
def coeff_constraints (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.gcd a (Nat.gcd b c) = 1

theorem equal_area_line_coeff_sum :
  ∃ a b c : ℕ,
    line_equation a b c ∧
    coeff_constraints a b c ∧
    a^2 + b^2 + c^2 = 65 :=
sorry

end NUMINAMATH_CALUDE_equal_area_line_coeff_sum_l2432_243238


namespace NUMINAMATH_CALUDE_allie_toys_count_l2432_243263

/-- Proves that given a set of toys with specified values, the total number of toys is correct -/
theorem allie_toys_count (total_worth : ℕ) (special_toy_worth : ℕ) (regular_toy_worth : ℕ) :
  total_worth = 52 →
  special_toy_worth = 12 →
  regular_toy_worth = 5 →
  ∃ (n : ℕ), n * regular_toy_worth + special_toy_worth = total_worth ∧ n + 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_allie_toys_count_l2432_243263


namespace NUMINAMATH_CALUDE_problem_solution_l2432_243211

theorem problem_solution (x y z : ℝ) 
  (h1 : 3 = 0.15 * x)
  (h2 : 3 = 0.25 * y)
  (h3 : z = 0.30 * y) :
  x - y + z = 11.6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2432_243211


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l2432_243273

theorem max_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 = 2}
  ∀ p ∈ circle, ∀ q ∈ line, 
    ∃ r ∈ circle, Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2) = 3 ∧
    ∀ s ∈ circle, Real.sqrt ((s.1 - q.1)^2 + (s.2 - q.2)^2) ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l2432_243273


namespace NUMINAMATH_CALUDE_inverse_function_property_l2432_243227

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 3 = 1 → f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2432_243227


namespace NUMINAMATH_CALUDE_basketball_team_points_l2432_243274

theorem basketball_team_points (x : ℚ) (y : ℕ) : 
  (1 / 3 : ℚ) * x + (1 / 5 : ℚ) * x + 18 + y = x → 
  y ≤ 21 → 
  y = 15 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_points_l2432_243274


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2432_243245

/-- The eccentricity of the ellipse x²/25 + y²/16 = 1 is 3/5 -/
theorem ellipse_eccentricity :
  let e : ℝ := Real.sqrt (1 - 16 / 25)
  e = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2432_243245


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l2432_243242

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (faces : Fin 4)
  (vertices : Fin 4)
  (edges : Fin 6)

/-- A stripe configuration on a tetrahedron -/
def StripeConfiguration := Tetrahedron → Fin 3

/-- Predicate for a continuous stripe encircling the tetrahedron -/
def IsContinuousStripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def TotalConfigurations : ℕ := 3^4

/-- The number of stripe configurations that result in a continuous stripe -/
def ContinuousStripeConfigurations : ℕ := 2^4

/-- The probability of a continuous stripe encircling the tetrahedron -/
def ProbabilityContinuousStripe : ℚ :=
  ContinuousStripeConfigurations / TotalConfigurations

theorem continuous_stripe_probability :
  ProbabilityContinuousStripe = 16 / 81 :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l2432_243242


namespace NUMINAMATH_CALUDE_cincinnati_to_nyc_distance_l2432_243204

/-- The total distance between Cincinnati and New York City -/
def total_distance (day1 day2 day3 remaining : ℕ) : ℕ :=
  day1 + day2 + day3 + remaining

/-- The distance walked on the second day -/
def day2_distance (day1 : ℕ) : ℕ :=
  day1 / 2 - 6

theorem cincinnati_to_nyc_distance :
  total_distance 20 (day2_distance 20) 10 36 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cincinnati_to_nyc_distance_l2432_243204


namespace NUMINAMATH_CALUDE_characterize_M_l2432_243249

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem characterize_M : M = {0, 1/2, 1/3} := by sorry

end NUMINAMATH_CALUDE_characterize_M_l2432_243249


namespace NUMINAMATH_CALUDE_deck_width_l2432_243298

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 := by sorry

end NUMINAMATH_CALUDE_deck_width_l2432_243298


namespace NUMINAMATH_CALUDE_circles_intersect_l2432_243257

/-- The circles x^2 + y^2 + 4x - 4y - 8 = 0 and x^2 + y^2 - 2x + 4y + 1 = 0 intersect. -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 + 4*x - 4*y - 8 = 0) ∧ (x^2 + y^2 - 2*x + 4*y + 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l2432_243257


namespace NUMINAMATH_CALUDE_cersei_cousin_fraction_l2432_243292

def initial_candies : ℕ := 50
def given_to_siblings : ℕ := 5 + 5
def eaten_by_cersei : ℕ := 12
def left_after_eating : ℕ := 18

theorem cersei_cousin_fraction :
  let remaining_after_siblings := initial_candies - given_to_siblings
  let given_to_cousin := remaining_after_siblings - (left_after_eating + eaten_by_cersei)
  (given_to_cousin : ℚ) / remaining_after_siblings = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_cersei_cousin_fraction_l2432_243292


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l2432_243209

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l2432_243209


namespace NUMINAMATH_CALUDE_cistern_filling_time_l2432_243224

/-- Given a cistern that can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 30/7 hours, prove that the time it takes
    for the other tap alone to fill the cistern is 3 hours. -/
theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 30 / 7 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l2432_243224


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2432_243277

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 4 / y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2432_243277


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2432_243228

/-- The number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2432_243228


namespace NUMINAMATH_CALUDE_xyz_value_l2432_243252

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2432_243252


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l2432_243256

/-- Represents an octagon with given width and height -/
structure Octagon where
  width : ℕ
  height : ℕ

/-- Calculates the area of the octagon -/
def octagonArea (o : Octagon) : ℕ :=
  -- The actual calculation is not provided, as it should be part of the proof
  sorry

/-- Theorem stating that an octagon with width 5 and height 8 has an area of 30 square units -/
theorem octagon_area_theorem (o : Octagon) (h1 : o.width = 5) (h2 : o.height = 8) : 
  octagonArea o = 30 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l2432_243256


namespace NUMINAMATH_CALUDE_linear_equation_and_absolute_value_l2432_243221

theorem linear_equation_and_absolute_value (m a : ℝ) :
  (∀ x, (m^2 - 9) * x^2 - (m - 3) * x + 6 = 0 → (m^2 - 9 = 0 ∧ m - 3 ≠ 0)) →
  |a| ≤ |m| →
  |a + m| + |a - m| = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_and_absolute_value_l2432_243221


namespace NUMINAMATH_CALUDE_multiplication_equation_l2432_243246

theorem multiplication_equation (m : ℕ) : 72519 * m = 724827405 → m = 9999 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_l2432_243246


namespace NUMINAMATH_CALUDE_total_beads_count_l2432_243251

/-- Represents the number of beads of each color in Sue's necklace --/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ
  red : ℕ

/-- Defines the conditions for Sue's necklace --/
def necklace_conditions (counts : BeadCounts) : Prop :=
  counts.purple = 7 ∧
  counts.blue = 2 * counts.purple ∧
  counts.green = counts.blue + 11 ∧
  counts.red = counts.green / 2 ∧
  (counts.purple + counts.blue + counts.green + counts.red) % 2 = 0

/-- The theorem to be proved --/
theorem total_beads_count (counts : BeadCounts) 
  (h : necklace_conditions counts) : 
  counts.purple + counts.blue + counts.green + counts.red = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_count_l2432_243251


namespace NUMINAMATH_CALUDE_circular_motion_angle_l2432_243235

theorem circular_motion_angle (θ : Real) : 
  (0 < θ) ∧ (θ < π) ∧                        -- 0 < θ < π
  (π < 2*θ) ∧ (2*θ < 3*π/2) ∧                -- Reaches third quadrant in 2 minutes
  (∃ (n : ℤ), 14*θ = n * (2*π)) →            -- Returns to original position in 14 minutes
  (θ = 4*π/7) ∨ (θ = 5*π/7) := by
sorry

end NUMINAMATH_CALUDE_circular_motion_angle_l2432_243235


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2432_243219

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 1 →                     -- first term condition
  d ≠ 0 →                       -- common difference not zero
  (a 2)^2 = a 1 * a 5 →         -- geometric sequence condition
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2432_243219


namespace NUMINAMATH_CALUDE_second_class_average_mark_l2432_243248

theorem second_class_average_mark (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg_total : ℝ) 
  (h1 : students1 = 22)
  (h2 : students2 = 28)
  (h3 : avg1 = 40)
  (h4 : avg_total = 51.2) :
  (avg_total * (students1 + students2) - avg1 * students1) / students2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_class_average_mark_l2432_243248


namespace NUMINAMATH_CALUDE_polygon_sides_l2432_243200

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (360 : ℝ) / (180 * (n - 2)) = 2 / 9 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2432_243200


namespace NUMINAMATH_CALUDE_problem_solution_l2432_243226

theorem problem_solution (a b c : ℝ) 
  (eq : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h1 : b = 15)
  (h2 : c = 5)
  (h3 : 2 = Real.sqrt ((a + 2) * (15 + 3)) / (5 + 1)) :
  a = 6 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2432_243226


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2432_243276

def M : Set ℝ := {x | x^2 = 2}
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_values (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2432_243276


namespace NUMINAMATH_CALUDE_zeros_of_g_l2432_243218

-- Define the power function f
def f : ℝ → ℝ := fun x => x^3

-- Define the function g
def g : ℝ → ℝ := fun x => f x - x

-- State the theorem
theorem zeros_of_g :
  (f 2 = 8) →
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_g_l2432_243218


namespace NUMINAMATH_CALUDE_fourth_member_income_l2432_243214

/-- Proves that in a family of 4 members with a given average income and known incomes of 3 members, the income of the fourth member is as calculated. -/
theorem fourth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 : ℕ)
  (h1 : family_size = 4)
  (h2 : average_income = 10000)
  (h3 : income1 = 15000)
  (h4 : income2 = 6000)
  (h5 : income3 = 11000) :
  (family_size * average_income) - (income1 + income2 + income3) = 8000 := by
  sorry

#eval (4 * 10000) - (15000 + 6000 + 11000)

end NUMINAMATH_CALUDE_fourth_member_income_l2432_243214


namespace NUMINAMATH_CALUDE_max_taxiing_time_l2432_243231

/-- The function representing the distance traveled by the plane after landing -/
def y (t : ℝ) : ℝ := 60 * t - 2 * t^2

/-- The maximum time the plane uses for taxiing -/
def s : ℝ := 15

theorem max_taxiing_time :
  ∀ t : ℝ, y t ≤ y s :=
by sorry

end NUMINAMATH_CALUDE_max_taxiing_time_l2432_243231


namespace NUMINAMATH_CALUDE_equation_solution_l2432_243202

theorem equation_solution (x y z : ℚ) 
  (eq1 : x - 4*y - 2*z = 0) 
  (eq2 : 3*x + 2*y - z = 0) 
  (z_neq_zero : z ≠ 0) : 
  (x^2 - 5*x*y) / (2*y^2 + z^2) = 164/147 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2432_243202


namespace NUMINAMATH_CALUDE_midpoint_tetrahedron_volume_ratio_l2432_243213

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices, just that it's a regular tetrahedron
  is_regular : Bool

/-- The tetrahedron formed by connecting the midpoints of the edges of a regular tetrahedron -/
def midpoint_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  { is_regular := true }  -- The midpoint tetrahedron is also regular

/-- The volume of a tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ :=
  sorry  -- We don't need to define this explicitly for the theorem

/-- 
  The ratio of the volume of the midpoint tetrahedron to the volume of the original tetrahedron
  is 1/8
-/
theorem midpoint_tetrahedron_volume_ratio (t : RegularTetrahedron) :
  volume (midpoint_tetrahedron t) / volume t = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_tetrahedron_volume_ratio_l2432_243213


namespace NUMINAMATH_CALUDE_combination_equality_solutions_l2432_243220

theorem combination_equality_solutions (x : ℕ) : 
  (Nat.choose 25 (2*x) = Nat.choose 25 (x + 4)) ↔ (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_solutions_l2432_243220


namespace NUMINAMATH_CALUDE_major_axis_length_is_6_l2432_243254

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The ellipse is tangent to the line x = 4 -/
  tangent_x_4 : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinates of the foci -/
  foci_y1 : ℝ
  foci_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 6 -/
theorem major_axis_length_is_6 (e : Ellipse) 
  (h1 : e.tangent_y_axis = true) 
  (h2 : e.tangent_x_4 = true)
  (h3 : e.foci_x = 3)
  (h4 : e.foci_y1 = 1 + Real.sqrt 3)
  (h5 : e.foci_y2 = 1 - Real.sqrt 3) : 
  majorAxisLength e = 6 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_is_6_l2432_243254


namespace NUMINAMATH_CALUDE_two_from_four_combination_l2432_243210

theorem two_from_four_combination : Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_two_from_four_combination_l2432_243210


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_21_l2432_243243

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem F_of_4_f_of_5_equals_21 : F 4 (f 5) = 21 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_21_l2432_243243


namespace NUMINAMATH_CALUDE_remainder_problem_l2432_243268

theorem remainder_problem : (2^300 + 300) % (2^150 + 2^75 + 1) = 298 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2432_243268


namespace NUMINAMATH_CALUDE_survey_size_l2432_243286

-- Define the problem parameters
def percent_independent : ℚ := 752 / 1000
def percent_no_companionship : ℚ := 621 / 1000
def misinformed_students : ℕ := 41

-- Define the theorem
theorem survey_size :
  ∃ (total_students : ℕ),
    (total_students > 0) ∧
    (↑misinformed_students : ℚ) / (percent_independent * percent_no_companionship * ↑total_students) = 1 ∧
    total_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_survey_size_l2432_243286


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2432_243239

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.2) * (1 - 0.15) = 231.2 → P = 340 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2432_243239


namespace NUMINAMATH_CALUDE_rectangle_area_l2432_243265

/-- Given a rectangle divided into 18 congruent squares, where the length is three times
    the width and the diagonal of one small square is 5 cm, the area of the entire
    rectangular region is 112.5 square cm. -/
theorem rectangle_area (n m : ℕ) (s : ℝ) : 
  n * m = 18 →
  n = 2 * m →
  s^2 + s^2 = 5^2 →
  (n * s) * (m * s) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2432_243265


namespace NUMINAMATH_CALUDE_car_speed_calculation_l2432_243207

/-- Proves that a car's speed is 104 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (h1 : fuel_efficiency = 64) -- km per liter
  (h2 : fuel_consumed = 3.9) -- gallons
  (h3 : time = 5.7) -- hours
  (h4 : (1 : ℝ) / 3.8 = 1 / 3.8) -- 1 gallon = 3.8 liters
  (h5 : (1 : ℝ) / 1.6 = 1 / 1.6) -- 1 mile = 1.6 kilometers
  : (fuel_efficiency * fuel_consumed * 3.8) / (time * 1.6) = 104 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l2432_243207


namespace NUMINAMATH_CALUDE_finite_squared_nilpotent_matrices_l2432_243229

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    the set of all possible A^2 matrices is finite. -/
theorem finite_squared_nilpotent_matrices 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  Set.Finite {B : Matrix (Fin 3) (Fin 3) ℝ | ∃ (A : Matrix (Fin 3) (Fin 3) ℝ), A ^ 4 = 0 ∧ B = A ^ 2} :=
sorry

end NUMINAMATH_CALUDE_finite_squared_nilpotent_matrices_l2432_243229


namespace NUMINAMATH_CALUDE_final_cat_count_l2432_243215

def initial_siamese : ℝ := 13.5
def initial_house : ℝ := 5.25
def cats_added : ℝ := 10.75
def cats_discounted : ℝ := 0.5

theorem final_cat_count :
  initial_siamese + initial_house + cats_added - cats_discounted = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_cat_count_l2432_243215


namespace NUMINAMATH_CALUDE_sum_15_terms_eq_56_25_l2432_243296

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  -- The 11th term is 5.25
  a11 : ℝ
  a11_eq : a11 = 5.25
  -- The 7th term is 3.25
  a7 : ℝ
  a7_eq : a7 = 3.25

/-- The sum of the first 15 terms of the arithmetic progression -/
def sum_15_terms (ap : ArithmeticProgression) : ℝ :=
  -- Definition of the sum (to be proved)
  56.25

/-- Theorem stating that the sum of the first 15 terms is 56.25 -/
theorem sum_15_terms_eq_56_25 (ap : ArithmeticProgression) :
  sum_15_terms ap = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_terms_eq_56_25_l2432_243296


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2432_243255

/-- Given a line passing through points (0, -4) and (4, 4), prove that the product of its slope and y-intercept equals -8. -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
  (∀ x y : ℝ, y = m * x + b → (x = 0 ∧ y = -4) ∨ (x = 4 ∧ y = 4)) → 
  m * b = -8 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2432_243255


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2432_243237

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x > 2) ↔ (∃ x : ℝ, x ≥ 1 ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2432_243237


namespace NUMINAMATH_CALUDE_min_value_theorem_l2432_243275

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), (3 / a + 2 / b) ≥ x := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2432_243275
