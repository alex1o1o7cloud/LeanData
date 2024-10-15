import Mathlib

namespace NUMINAMATH_CALUDE_race_completion_time_l2081_208114

/-- Given a 1000-meter race where runner A beats runner B by either 60 meters or 10 seconds,
    this theorem proves that runner A completes the race in 156.67 seconds. -/
theorem race_completion_time :
  ∀ (speed_A speed_B : ℝ),
  speed_A > 0 ∧ speed_B > 0 →
  1000 / speed_A = 940 / speed_B →
  1000 / speed_A = (1000 / speed_B) - 10 →
  1000 / speed_A = 156.67 :=
by sorry

end NUMINAMATH_CALUDE_race_completion_time_l2081_208114


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2081_208177

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 1) →
  (a 2 + a 3 + a 4 = 2) →
  (a 5 + a 6 + a 7 = 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2081_208177


namespace NUMINAMATH_CALUDE_cone_volume_l2081_208164

/-- The volume of a cone with lateral surface area 2√3π and central angle √3π is π. -/
theorem cone_volume (r l : ℝ) (h_angle : 2 * π * r / l = Real.sqrt 3 * π)
  (h_area : π * r * l = 2 * Real.sqrt 3 * π) : 
  (1/3) * π * r^2 * Real.sqrt (l^2 - r^2) = π :=
sorry

end NUMINAMATH_CALUDE_cone_volume_l2081_208164


namespace NUMINAMATH_CALUDE_path_area_and_cost_l2081_208161

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 959 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1918 := by
  sorry

#eval path_area 75 55 3.5
#eval construction_cost (path_area 75 55 3.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l2081_208161


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2081_208157

theorem scientific_notation_equality : 122254 = 1.22254 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2081_208157


namespace NUMINAMATH_CALUDE_problem_statement_l2081_208175

theorem problem_statement (a b : ℕ) : 
  a = 105 → a^3 = 21 * 35 * 45 * b → b = 105 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2081_208175


namespace NUMINAMATH_CALUDE_intersection_S_T_l2081_208195

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4*x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2081_208195


namespace NUMINAMATH_CALUDE_peanut_butter_jars_l2081_208141

/-- Given 2032 ounces of peanut butter distributed equally among jars of 16, 28, 40, and 52 ounces,
    the total number of jars is 60. -/
theorem peanut_butter_jars :
  let total_ounces : ℕ := 2032
  let jar_sizes : List ℕ := [16, 28, 40, 52]
  let num_sizes : ℕ := jar_sizes.length
  ∃ (x : ℕ),
    (x * (jar_sizes.sum)) = total_ounces ∧
    (num_sizes * x) = 60
  := by sorry

end NUMINAMATH_CALUDE_peanut_butter_jars_l2081_208141


namespace NUMINAMATH_CALUDE_pieces_per_package_calculation_l2081_208113

/-- Given the number of gum packages, candy packages, and total pieces,
    calculate the number of pieces per package. -/
def pieces_per_package (gum_packages : ℕ) (candy_packages : ℕ) (total_pieces : ℕ) : ℚ :=
  total_pieces / (gum_packages + candy_packages)

/-- Theorem stating that with 28 gum packages, 14 candy packages, and 7 total pieces,
    the number of pieces per package is 1/6. -/
theorem pieces_per_package_calculation :
  pieces_per_package 28 14 7 = 1/6 := by
  sorry

#eval pieces_per_package 28 14 7

end NUMINAMATH_CALUDE_pieces_per_package_calculation_l2081_208113


namespace NUMINAMATH_CALUDE_P_50_is_identity_l2081_208181

def P : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -4, -3]

theorem P_50_is_identity : P ^ 50 = 1 := by sorry

end NUMINAMATH_CALUDE_P_50_is_identity_l2081_208181


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_l2081_208132

def numbers : List Nat := [18, 24, 36]

def C : Nat := numbers.foldl Nat.gcd 0

def D : Nat := numbers.foldl Nat.lcm 1

theorem sum_gcd_lcm : C + D = 78 := by sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_l2081_208132


namespace NUMINAMATH_CALUDE_set_operation_result_l2081_208176

def set_operation (M N : Set Int) : Set Int :=
  {x | ∃ y z, y ∈ N ∧ z ∈ M ∧ x = y - z}

theorem set_operation_result :
  let M : Set Int := {0, 1, 2}
  let N : Set Int := {-2, -3}
  set_operation M N = {-2, -3, -4, -5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l2081_208176


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_value_condition_l2081_208168

-- Part 1: Inequality solution
theorem inequality_solution (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ x ≤ 1 ∨ x > 2 :=
sorry

-- Part 2: Minimum value
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_minimum_value_condition_l2081_208168


namespace NUMINAMATH_CALUDE_inequality_solution_l2081_208153

open Set Real

def inequality_holds (x a : ℝ) : Prop :=
  (a + 2) * x - (1 + 2 * a) * (x^2)^(1/3) - 6 * x^(1/3) + a^2 + 4 * a - 5 > 0

theorem inequality_solution :
  ∀ x : ℝ, (∃ a ∈ Icc (-2) 1, inequality_holds x a) ↔ 
  x ∈ Iio (-1) ∪ Ioo (-1) 0 ∪ Ioi 8 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2081_208153


namespace NUMINAMATH_CALUDE_alice_paid_48_percent_of_srp_l2081_208129

-- Define the suggested retail price (SRP)
def suggested_retail_price : ℝ := 100

-- Define the marked price (MP) as 80% of SRP
def marked_price : ℝ := 0.8 * suggested_retail_price

-- Define Alice's purchase price as 60% of MP
def alice_price : ℝ := 0.6 * marked_price

-- Theorem to prove
theorem alice_paid_48_percent_of_srp :
  alice_price / suggested_retail_price = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_48_percent_of_srp_l2081_208129


namespace NUMINAMATH_CALUDE_petes_bottle_return_l2081_208197

/-- Represents the number of bottles Pete needs to return to the store -/
def bottles_to_return (total_owed : ℚ) (cash_in_wallet : ℚ) (cash_in_pockets : ℚ) (bottle_return_rate : ℚ) : ℕ :=
  sorry

/-- The theorem stating the number of bottles Pete needs to return -/
theorem petes_bottle_return : 
  bottles_to_return 90 40 40 (1/2) = 20 := by sorry

end NUMINAMATH_CALUDE_petes_bottle_return_l2081_208197


namespace NUMINAMATH_CALUDE_range_of_x₁_l2081_208179

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the condition given in the problem
def Condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1

-- Theorem statement
theorem range_of_x₁ (h_increasing : IsIncreasing f) (h_condition : Condition f) :
  ∀ x₁, (∃ x₂, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) ↔ x₁ > 1 :=
by sorry


end NUMINAMATH_CALUDE_range_of_x₁_l2081_208179


namespace NUMINAMATH_CALUDE_max_value_2q_minus_r_l2081_208106

theorem max_value_2q_minus_r :
  ∀ q r : ℕ+, 
  965 = 22 * q + r → 
  ∀ q' r' : ℕ+, 
  965 = 22 * q' + r' → 
  2 * q - r ≤ 67 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2q_minus_r_l2081_208106


namespace NUMINAMATH_CALUDE_two_digit_number_square_equals_cube_of_digit_sum_l2081_208196

theorem two_digit_number_square_equals_cube_of_digit_sum :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_square_equals_cube_of_digit_sum_l2081_208196


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2081_208144

/-- Proves that the ratio of A's speed to B's speed is 2:1 in a race where A gives B a head start -/
theorem race_speed_ratio (race_length : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : race_length = 142)
  (h2 : head_start = 71)
  (h3 : race_length / speed_A = (race_length - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

#check race_speed_ratio

end NUMINAMATH_CALUDE_race_speed_ratio_l2081_208144


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2081_208140

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_64_over_9_l2081_208140


namespace NUMINAMATH_CALUDE_function_inequality_l2081_208103

theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : Real.exp (a + 1) = a + 4) (h2 : Real.log (b + 3) = b) :
  let f := fun x => Real.exp x + (a - b) * x
  f (2/3) < f 0 ∧ f 0 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2081_208103


namespace NUMINAMATH_CALUDE_minimum_speed_to_clear_building_l2081_208187

/-- The minimum speed required for a stone to clear a building -/
theorem minimum_speed_to_clear_building 
  (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) 
  (h_α : 0 < α ∧ α < π / 2) : 
  ∃ (v₀ : ℝ), v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧ 
  (∀ (v : ℝ), v > v₀ → 
    ∃ (trajectory : ℝ → ℝ), 
      (∀ x, trajectory x ≤ H + Real.tan α * (l - x)) ∧
      (∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ l ∧ 
        trajectory x₁ = H ∧ trajectory x₂ = H + Real.tan α * (l - x₂))) :=
sorry

end NUMINAMATH_CALUDE_minimum_speed_to_clear_building_l2081_208187


namespace NUMINAMATH_CALUDE_four_three_seating_chart_l2081_208149

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpret (pair : ℕ × ℕ) : SeatingChart :=
  { columns := pair.1, rows := pair.2 }

/-- States that (4,3) represents 4 columns and 3 rows -/
theorem four_three_seating_chart :
  let chart := interpret (4, 3)
  chart.columns = 4 ∧ chart.rows = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_three_seating_chart_l2081_208149


namespace NUMINAMATH_CALUDE_store_transaction_result_l2081_208131

/-- Represents the result of a store's transaction -/
inductive TransactionResult
  | BreakEven
  | Profit (amount : ℝ)
  | Loss (amount : ℝ)

/-- Calculates the result of a store's transaction given the selling price and profit/loss percentages -/
def calculateTransactionResult (sellingPrice : ℝ) (profit1 : ℝ) (loss2 : ℝ) : TransactionResult :=
  sorry

theorem store_transaction_result :
  let sellingPrice : ℝ := 80
  let profit1 : ℝ := 60
  let loss2 : ℝ := 20
  calculateTransactionResult sellingPrice profit1 loss2 = TransactionResult.Profit 10 :=
sorry

end NUMINAMATH_CALUDE_store_transaction_result_l2081_208131


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l2081_208138

-- First inequality
theorem inequality_one (x : ℝ) : 
  (|1 - (2*x - 1)/3| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 5) :=
sorry

-- Second inequality
theorem inequality_two (x : ℝ) :
  ((2 - x)*(x + 3) < 2 - x) ↔ (x > 2 ∨ x < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l2081_208138


namespace NUMINAMATH_CALUDE_max_c_value_l2081_208111

theorem max_c_value (a b : ℝ) (h : a + 2*b = 2) : 
  ∃ c_max : ℝ, c_max = 3 ∧ 
  (∀ c : ℝ, (3:ℝ)^a + (9:ℝ)^b ≥ c^2 - c → c ≤ c_max) ∧
  ((3:ℝ)^a + (9:ℝ)^b ≥ c_max^2 - c_max) :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l2081_208111


namespace NUMINAMATH_CALUDE_orange_book_pages_l2081_208110

/-- Proves that the number of pages in each orange book is 510, given the specified conditions --/
theorem orange_book_pages : ℕ → Prop :=
  fun (x : ℕ) =>
    let purple_pages_per_book : ℕ := 230
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let extra_orange_pages : ℕ := 890
    (purple_pages_per_book * purple_books_read + extra_orange_pages = orange_books_read * x) →
    x = 510

/-- The proof of the theorem --/
lemma prove_orange_book_pages : orange_book_pages 510 := by
  sorry

end NUMINAMATH_CALUDE_orange_book_pages_l2081_208110


namespace NUMINAMATH_CALUDE_quadratic_sum_l2081_208133

/-- Given a quadratic function f(x) = -3x^2 - 27x + 81, prove that when 
    rewritten in the form a(x+b)^2 + c, the sum a + b + c equals 143.25 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3*x^2 - 27*x + 81) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = 143.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2081_208133


namespace NUMINAMATH_CALUDE_boosters_club_average_sales_l2081_208159

/-- The average monthly sales for the Boosters Club candy sales --/
theorem boosters_club_average_sales :
  let sales : List ℕ := [90, 50, 70, 110, 80]
  let total_sales : ℕ := sales.sum
  let num_months : ℕ := sales.length
  (total_sales : ℚ) / num_months = 80 := by sorry

end NUMINAMATH_CALUDE_boosters_club_average_sales_l2081_208159


namespace NUMINAMATH_CALUDE_subtraction_decimal_l2081_208108

theorem subtraction_decimal : (3.56 : ℝ) - (1.89 : ℝ) = 1.67 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l2081_208108


namespace NUMINAMATH_CALUDE_problem_statement_l2081_208123

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Define the intervals
def I : Set ℝ := Set.Icc 0 2
def J : Set ℝ := Set.Icc (1/2) 2

-- State the theorem
theorem problem_statement :
  (∃ M : ℤ, (M = 4 ∧ ∀ N : ℤ, N > M → ¬∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ g x₁ - g x₂ ≥ N)) ∧
  (∃ a : ℝ, (a = 1 ∧ ∀ s t : ℝ, s ∈ J → t ∈ J → f a s ≥ g t) ∧
            ∀ b : ℝ, b < a → ∃ s t : ℝ, s ∈ J ∧ t ∈ J ∧ f b s < g t) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statement_l2081_208123


namespace NUMINAMATH_CALUDE_additional_miles_with_bakery_stop_l2081_208188

/-- The additional miles driven with a bakery stop compared to without -/
theorem additional_miles_with_bakery_stop
  (apartment_to_bakery : ℕ)
  (bakery_to_grandma : ℕ)
  (grandma_to_apartment : ℕ)
  (h1 : apartment_to_bakery = 9)
  (h2 : bakery_to_grandma = 24)
  (h3 : grandma_to_apartment = 27) :
  (apartment_to_bakery + bakery_to_grandma + grandma_to_apartment) -
  (2 * grandma_to_apartment) = 6 :=
by sorry

end NUMINAMATH_CALUDE_additional_miles_with_bakery_stop_l2081_208188


namespace NUMINAMATH_CALUDE_parents_gift_ratio_equal_l2081_208170

/-- Represents the spending on Christmas gifts -/
structure ChristmasGifts where
  sibling_cost : ℕ  -- Cost per sibling's gift
  num_siblings : ℕ  -- Number of siblings
  total_spent : ℕ  -- Total amount spent on all gifts
  parent_cost : ℕ  -- Cost per parent's gift

/-- Theorem stating that the ratio of gift values for Mia's parents is 1:1 -/
theorem parents_gift_ratio_equal (gifts : ChristmasGifts)
  (h1 : gifts.sibling_cost = 30)
  (h2 : gifts.num_siblings = 3)
  (h3 : gifts.total_spent = 150)
  (h4 : gifts.parent_cost = 30) :
  gifts.parent_cost / gifts.parent_cost = 1 := by
  sorry

#check parents_gift_ratio_equal

end NUMINAMATH_CALUDE_parents_gift_ratio_equal_l2081_208170


namespace NUMINAMATH_CALUDE_distance_to_origin_l2081_208145

/-- The distance from point (5, -12) to the origin in the Cartesian coordinate system is 13. -/
theorem distance_to_origin : Real.sqrt (5^2 + (-12)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2081_208145


namespace NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2081_208169

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

/-- Proof that Tim grew 44 cantaloupes -/
theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2081_208169


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2081_208186

theorem hyperbola_sum (F₁ F₂ : ℝ × ℝ) (h k a b : ℝ) :
  F₁ = (-2, 0) →
  F₂ = (2, 0) →
  a > 0 →
  b > 0 →
  (∀ P : ℝ × ℝ, |dist P F₁ - dist P F₂| = 2 ↔ 
    (P.1 - h)^2 / a^2 - (P.2 - k)^2 / b^2 = 1) →
  h + k + a + b = 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2081_208186


namespace NUMINAMATH_CALUDE_airplane_flight_problem_l2081_208143

/-- Airplane flight problem -/
theorem airplane_flight_problem 
  (wind_speed : ℝ) 
  (time_with_wind : ℝ) 
  (time_against_wind : ℝ) 
  (h1 : wind_speed = 24)
  (h2 : time_with_wind = 2.8)
  (h3 : time_against_wind = 3) :
  ∃ (airplane_speed : ℝ) (distance : ℝ),
    airplane_speed = 696 ∧ 
    distance = 2016 ∧
    time_with_wind * (airplane_speed + wind_speed) = distance ∧
    time_against_wind * (airplane_speed - wind_speed) = distance :=
by
  sorry


end NUMINAMATH_CALUDE_airplane_flight_problem_l2081_208143


namespace NUMINAMATH_CALUDE_solution_exists_l2081_208182

-- Define the functions f and g
def f (x : ℝ) := x^2 + 10
def g (x : ℝ) := x^2 - 6

-- State the theorem
theorem solution_exists (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 14) :
  a = Real.sqrt 8 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_exists_l2081_208182


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2081_208198

theorem line_not_in_second_quadrant (α : Real) (h : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ x / Real.cos α + y / Real.sin α = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2081_208198


namespace NUMINAMATH_CALUDE_common_roots_solution_l2081_208112

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧
    p^3 + c*p^2 + 7*p + 4 = 0 ∧
    p^3 + d*p^2 + 10*p + 6 = 0 ∧
    q^3 + c*q^2 + 7*q + 4 = 0 ∧
    q^3 + d*q^2 + 10*q + 6 = 0

/-- The theorem stating the unique solution for c and d -/
theorem common_roots_solution :
  ∀ c d : ℝ, has_two_common_roots c d → c = -5 ∧ d = -6 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_solution_l2081_208112


namespace NUMINAMATH_CALUDE_tan_neg_390_degrees_l2081_208127

theorem tan_neg_390_degrees : Real.tan ((-390 : ℝ) * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_390_degrees_l2081_208127


namespace NUMINAMATH_CALUDE_exists_valid_configuration_l2081_208119

/-- A configuration of 9 numbers placed in circles -/
def Configuration := Fin 9 → Nat

/-- The 6 lines connecting the circles -/
def Lines := Fin 6 → Fin 3 → Fin 9

/-- Check if a configuration is valid -/
def is_valid_configuration (config : Configuration) (lines : Lines) : Prop :=
  (∀ i : Fin 9, config i ∈ Finset.range 10 \ {0}) ∧  -- Numbers are from 1 to 9
  (∃ i : Fin 9, config i = 6) ∧                      -- 6 is included
  (∀ i j : Fin 9, i ≠ j → config i ≠ config j) ∧     -- All numbers are different
  (∀ l : Fin 6, (config (lines l 0) + config (lines l 1) + config (lines l 2) = 23))  -- Sum on each line is 23

theorem exists_valid_configuration (lines : Lines) : 
  ∃ (config : Configuration), is_valid_configuration config lines :=
sorry

end NUMINAMATH_CALUDE_exists_valid_configuration_l2081_208119


namespace NUMINAMATH_CALUDE_at_op_difference_l2081_208174

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - y * x - 3 * x + 2 * y

-- State the theorem
theorem at_op_difference : at_op 9 5 - at_op 5 9 = -20 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l2081_208174


namespace NUMINAMATH_CALUDE_range_of_expression_l2081_208102

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- State the theorem
theorem range_of_expression (t : AcuteTriangle) 
  (h : t.b * Real.cos t.A - t.a * Real.cos t.B = t.a) :
  2 < Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 ∧ 
  Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 < Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2081_208102


namespace NUMINAMATH_CALUDE_original_number_is_509_l2081_208199

theorem original_number_is_509 (subtracted_number : ℕ) : 
  (509 - subtracted_number) % 9 = 0 →
  subtracted_number ≥ 5 →
  ∀ n < subtracted_number, (509 - n) % 9 ≠ 0 →
  509 = 509 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_is_509_l2081_208199


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l2081_208178

def ellipse (h k a b : ℝ) := fun (x y : ℝ) ↦ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_parameter_sum :
  ∃ h k a b : ℝ,
    (∀ x y : ℝ, ellipse h k a b x y ↔ 
      Real.sqrt ((x - 0)^2 + (y - 0)^2) + Real.sqrt ((x - 6)^2 + (y - 0)^2) = 10) ∧
    h + k + a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l2081_208178


namespace NUMINAMATH_CALUDE_total_food_items_is_149_l2081_208136

/-- Represents the eating habits of a person -/
structure EatingHabits where
  croissants : ℕ
  cakes : ℕ
  pizzas : ℕ

/-- Calculates the total food items consumed by a person -/
def totalFoodItems (habits : EatingHabits) : ℕ :=
  habits.croissants + habits.cakes + habits.pizzas

/-- The eating habits of Jorge -/
def jorge : EatingHabits :=
  { croissants := 7, cakes := 18, pizzas := 30 }

/-- The eating habits of Giuliana -/
def giuliana : EatingHabits :=
  { croissants := 5, cakes := 14, pizzas := 25 }

/-- The eating habits of Matteo -/
def matteo : EatingHabits :=
  { croissants := 6, cakes := 16, pizzas := 28 }

/-- Theorem stating that the total food items consumed by Jorge, Giuliana, and Matteo is 149 -/
theorem total_food_items_is_149 :
  totalFoodItems jorge + totalFoodItems giuliana + totalFoodItems matteo = 149 := by
  sorry

end NUMINAMATH_CALUDE_total_food_items_is_149_l2081_208136


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l2081_208189

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem four_digit_number_problem (N : ℕ) 
  (h1 : is_four_digit N) 
  (h2 : (70000 + N) - (10 * N + 7) = 53208) : 
  N = 1865 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l2081_208189


namespace NUMINAMATH_CALUDE_descent_time_calculation_l2081_208117

theorem descent_time_calculation (climb_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  climb_time = 4 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    avg_speed_total = (2 * avg_speed_climb * climb_time) / (climb_time + descent_time) :=
by sorry

end NUMINAMATH_CALUDE_descent_time_calculation_l2081_208117


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2081_208124

/-- A circle with a diameter of 10 units -/
def Circle := {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2) ≤ 25}

/-- A line at distance d from the origin -/
def Line (d : ℝ) := {p : ℝ × ℝ | p.2 = d}

/-- The line is tangent to the circle if and only if the distance is 5 -/
theorem line_tangent_to_circle (d : ℝ) : 
  (∃ (p : ℝ × ℝ), p ∈ Circle ∩ Line d ∧ 
    ∀ (q : ℝ × ℝ), q ∈ Circle ∩ Line d → q = p) ↔ 
  d = 5 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2081_208124


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2081_208116

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∀ x' y' : ℝ, y' = m * x' + 1 ∧ x'^2 + 4 * y'^2 = 1 → x = x' ∧ y = y') →
  m^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2081_208116


namespace NUMINAMATH_CALUDE_average_weight_increase_l2081_208118

/-- Proves that replacing a person weighing 47 kg with a person weighing 68 kg in a group of 6 people increases the average weight by 3.5 kg -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 47 →
  new_weight = 68 →
  (new_weight - old_weight) / initial_count = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2081_208118


namespace NUMINAMATH_CALUDE_inequality_proof_l2081_208166

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + a) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2081_208166


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l2081_208125

/-- 
Given a man's rate (speed in still water) and his speed against the stream,
this theorem proves his speed with the stream.
-/
theorem mans_speed_with_stream 
  (rate : ℝ) 
  (speed_against : ℝ) 
  (h1 : rate = 2) 
  (h2 : speed_against = 6) : 
  rate + (speed_against - rate) = 6 := by
sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l2081_208125


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l2081_208151

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (k : ℕ), 
    (1000 : ℚ) / 137 = 7 + (n : ℚ) / (10 ^ k) ∧ 
    0 < n ∧ 
    n < 10 ^ k ∧ 
    n % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l2081_208151


namespace NUMINAMATH_CALUDE_kim_sweater_count_l2081_208165

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := 34

/-- The maximum number of sweaters Kim can knit in a day -/
def max_daily_sweaters : ℕ := 10

theorem kim_sweater_count :
  monday_sweaters ≤ max_daily_sweaters ∧
  monday_sweaters +
  (monday_sweaters + 2) +
  ((monday_sweaters + 2) - 4) +
  ((monday_sweaters + 2) - 4) +
  (monday_sweaters / 2) = total_sweaters :=
by sorry

end NUMINAMATH_CALUDE_kim_sweater_count_l2081_208165


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2081_208156

/-- A geometric sequence with a negative common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : a 3 - 4 = a 2) :
  a 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2081_208156


namespace NUMINAMATH_CALUDE_range_of_linear_function_l2081_208192

theorem range_of_linear_function (c : ℝ) (h : c ≠ 0) :
  let g : ℝ → ℝ := λ x ↦ c * x + 2
  let domain := Set.Icc (-1 : ℝ) 2
  let range := Set.image g domain
  range = if c > 0 
    then Set.Icc (-c + 2) (2 * c + 2)
    else Set.Icc (2 * c + 2) (-c + 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l2081_208192


namespace NUMINAMATH_CALUDE_chapters_per_book_l2081_208163

theorem chapters_per_book (total_books : ℕ) (total_chapters : ℕ) (h1 : total_books = 4) (h2 : total_chapters = 68) :
  total_chapters / total_books = 17 := by
  sorry

end NUMINAMATH_CALUDE_chapters_per_book_l2081_208163


namespace NUMINAMATH_CALUDE_inverse_functions_l2081_208135

-- Define the types of functions
def LinearDecreasing : Type := ℝ → ℝ
def PiecewiseConstant : Type := ℝ → ℝ
def VerticalLine : Type := ℝ → ℝ
def Semicircle : Type := ℝ → ℝ
def ModifiedPolynomial : Type := ℝ → ℝ

-- Define the property of having an inverse
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- State the theorem
theorem inverse_functions 
  (F : LinearDecreasing) 
  (G : PiecewiseConstant) 
  (H : VerticalLine) 
  (I : Semicircle) 
  (J : ModifiedPolynomial) : 
  HasInverse F ∧ HasInverse G ∧ ¬HasInverse H ∧ ¬HasInverse I ∧ ¬HasInverse J := by
  sorry

end NUMINAMATH_CALUDE_inverse_functions_l2081_208135


namespace NUMINAMATH_CALUDE_circle_equation_l2081_208173

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its distance from the y-axis equals its radius -/
def tangentToYAxis (c : Circle) : Prop :=
  |c.center.1| = c.radius

theorem circle_equation (c : Circle) (h : tangentToYAxis c) (h2 : c.center = (-2, 3)) :
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2081_208173


namespace NUMINAMATH_CALUDE_equal_area_segment_property_l2081_208193

/-- A trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  area_ratio : (b + (b + 75)) / (b + 75 + (b + 150)) = 1 / 2  -- Midpoint segment divides areas in ratio 1:2

/-- The length of the segment parallel to bases dividing the trapezoid into equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  let x : ℝ := sorry
  x

/-- The main theorem -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(2812.5 + 112.5 * equal_area_segment t) / 100⌋ = ⌊(equal_area_segment t)^2 / 100⌋ := by
  sorry

end NUMINAMATH_CALUDE_equal_area_segment_property_l2081_208193


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2081_208142

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2081_208142


namespace NUMINAMATH_CALUDE_triangle_area_l2081_208150

/-- The area of a triangle with base 3 meters and height 4 meters is 6 square meters. -/
theorem triangle_area : 
  let base : ℝ := 3
  let height : ℝ := 4
  let area : ℝ := (base * height) / 2
  area = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2081_208150


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_l2081_208155

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
by
  have h1 : points_per_enemy = 5 := by sorry
  have h2 : total_enemies = 8 := by sorry
  have h3 : points_earned = 10 := by sorry
  
  -- Define the number of enemies defeated
  let enemies_defeated := points_earned / points_per_enemy
  
  -- Calculate enemies left undefeated
  let enemies_left := total_enemies - enemies_defeated
  
  exact enemies_left

end NUMINAMATH_CALUDE_enemies_left_undefeated_l2081_208155


namespace NUMINAMATH_CALUDE_leftover_snacks_problem_l2081_208104

/-- Calculates the number of leftover snacks when feeding goats with dietary restrictions --/
def leftover_snacks (total_goats : ℕ) (restricted_goats : ℕ) (baby_carrots : ℕ) (cherry_tomatoes : ℕ) : ℕ :=
  let unrestricted_goats := total_goats - restricted_goats
  let tomatoes_per_restricted_goat := cherry_tomatoes / restricted_goats
  let leftover_tomatoes := cherry_tomatoes % restricted_goats
  let carrots_per_unrestricted_goat := baby_carrots / unrestricted_goats
  let leftover_carrots := baby_carrots % unrestricted_goats
  leftover_tomatoes + leftover_carrots

/-- Theorem stating that given the problem conditions, 6 snacks will be left over --/
theorem leftover_snacks_problem :
  leftover_snacks 9 3 124 56 = 6 := by
  sorry

end NUMINAMATH_CALUDE_leftover_snacks_problem_l2081_208104


namespace NUMINAMATH_CALUDE_student_count_l2081_208115

theorem student_count (n : ℕ) (rank_top rank_bottom : ℕ) 
  (h1 : rank_top = 75)
  (h2 : rank_bottom = 75)
  (h3 : n = rank_top + rank_bottom - 1) :
  n = 149 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2081_208115


namespace NUMINAMATH_CALUDE_two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l2081_208139

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Our specific quadratic equation x^2 - 2x - 3m^2 = 0 -/
def our_equation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -2, c := -3*m^2 }

theorem two_distinct_roots_for_all_m (m : ℝ) :
  has_two_distinct_real_roots (our_equation m) := by
  sorry

theorem m_value_when_root_sum_condition (m : ℝ) (α β : ℝ)
  (h1 : α + β = 2)
  (h2 : α + 2*β = 5)
  (h3 : α * β = -(-3*m^2)) :
  m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l2081_208139


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l2081_208172

theorem sqrt_fraction_equality : 
  Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35)) = 256 / Real.sqrt 2049 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l2081_208172


namespace NUMINAMATH_CALUDE_max_profit_l2081_208183

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

-- Define the sales volume function
def P (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def y (x : ℝ) : ℝ := 26 - 4 / (x + 1) - x

-- State the theorem
theorem max_profit (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → y x ≤ (if a ≥ 1 then 23 else 26 - 4 / (a + 1) - a)) ∧
  (if a ≥ 1 
   then y 1 = 23 
   else y a = 26 - 4 / (a + 1) - a) :=
sorry

end

end NUMINAMATH_CALUDE_max_profit_l2081_208183


namespace NUMINAMATH_CALUDE_range_of_H_l2081_208147

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H : 
  Set.range H = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_H_l2081_208147


namespace NUMINAMATH_CALUDE_two_solutions_exist_l2081_208109

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ := fun x =>
  if x < -1 then -2 * x
  else if x < 3 then 2 * x + 1
  else -2 * x + 16

-- Define the property we want to prove
def satisfies_equation (x : ℝ) : Prop := g (g x) = 4

-- Theorem statement
theorem two_solutions_exist :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ satisfies_equation x :=
sorry

end NUMINAMATH_CALUDE_two_solutions_exist_l2081_208109


namespace NUMINAMATH_CALUDE_am_gm_inequality_l2081_208122

theorem am_gm_inequality (a b : ℝ) (h : a * b > 0) : a / b + b / a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l2081_208122


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l2081_208107

theorem cosine_sine_sum_equals_half : 
  Real.cos (36 * π / 180) * Real.cos (96 * π / 180) + 
  Real.sin (36 * π / 180) * Real.sin (84 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l2081_208107


namespace NUMINAMATH_CALUDE_square_1849_product_l2081_208171

theorem square_1849_product (x : ℤ) (h : x^2 = 1849) : (x + 2) * (x - 2) = 1845 := by
  sorry

end NUMINAMATH_CALUDE_square_1849_product_l2081_208171


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_fraction_l2081_208128

theorem first_nonzero_digit_of_fraction (n : ℕ) (h : n = 1029) : 
  ∃ (k : ℕ) (d : ℕ), 
    0 < d ∧ d < 10 ∧
    (↑k : ℚ) < (1 : ℚ) / n ∧
    (1 : ℚ) / n < ((↑k + 1) : ℚ) / 10 ∧
    d = 9 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_fraction_l2081_208128


namespace NUMINAMATH_CALUDE_speed_difference_proof_l2081_208121

/-- Proves the speed difference between two vehicles given their travel conditions -/
theorem speed_difference_proof (base_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  base_speed = 44 →
  time = 4 →
  total_distance = 384 →
  ∃ (speed_diff : ℝ),
    speed_diff > 0 ∧
    total_distance = base_speed * time + (base_speed + speed_diff) * time ∧
    speed_diff = 8 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_proof_l2081_208121


namespace NUMINAMATH_CALUDE_y_intercept_of_perpendicular_line_l2081_208160

-- Define line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define a point on a line given its slope and a point it passes through
def point_on_line (m x₀ y₀ y : ℝ) (x : ℝ) : Prop := y - y₀ = m * (x - x₀)

-- Theorem statement
theorem y_intercept_of_perpendicular_line :
  ∃ (m : ℝ), 
    (∀ x y, line_l x y → y = (1/2) * x + (1/2)) →
    perpendicular (1/2) m →
    point_on_line m (-1) 0 0 0 →
    ∃ y, point_on_line m 0 y 0 0 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_perpendicular_line_l2081_208160


namespace NUMINAMATH_CALUDE_fraction_of_a_equal_to_quarter_of_b_l2081_208158

theorem fraction_of_a_equal_to_quarter_of_b : ∀ (a b x : ℚ), 
  a + b = 1210 →
  b = 484 →
  x * a = (1/4) * b →
  x = 1/6 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_a_equal_to_quarter_of_b_l2081_208158


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2081_208120

-- Define the sets U, A, and B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (U \ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2081_208120


namespace NUMINAMATH_CALUDE_gcd_36745_59858_l2081_208126

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_36745_59858_l2081_208126


namespace NUMINAMATH_CALUDE_speed_conversion_l2081_208190

-- Define the conversion factors
def km_to_m : ℚ := 1000
def hour_to_sec : ℚ := 3600

-- Define the given speed in km/h
def speed_kmh : ℚ := 72

-- Define the conversion function
def kmh_to_ms (speed : ℚ) : ℚ :=
  speed * km_to_m / hour_to_sec

-- Theorem statement
theorem speed_conversion :
  kmh_to_ms speed_kmh = 20 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l2081_208190


namespace NUMINAMATH_CALUDE_rice_mixture_cost_problem_l2081_208162

/-- The cost of the second variety of rice per kg -/
def second_variety_cost : ℝ := 12.50

/-- The cost of the first variety of rice per kg -/
def first_variety_cost : ℝ := 5

/-- The cost of the mixture per kg -/
def mixture_cost : ℝ := 7.50

/-- The ratio of the two varieties of rice -/
def rice_ratio : ℝ := 0.5

theorem rice_mixture_cost_problem :
  first_variety_cost * 1 + second_variety_cost * rice_ratio = mixture_cost * (1 + rice_ratio) :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_problem_l2081_208162


namespace NUMINAMATH_CALUDE_perfect_game_score_l2081_208185

/-- Given that a perfect score is 21 points, prove that the total points
    after 3 perfect games is equal to 63. -/
theorem perfect_game_score (perfect_score : ℕ) (num_games : ℕ) :
  perfect_score = 21 → num_games = 3 → perfect_score * num_games = 63 := by
  sorry

end NUMINAMATH_CALUDE_perfect_game_score_l2081_208185


namespace NUMINAMATH_CALUDE_x_fifth_plus_72x_l2081_208184

theorem x_fifth_plus_72x (x : ℝ) (h : x^2 + 6*x = 12) : x^5 + 72*x = 2808*x - 4320 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_72x_l2081_208184


namespace NUMINAMATH_CALUDE_cubic_factorization_l2081_208191

theorem cubic_factorization (m : ℝ) : m^3 - 9*m = m*(m+3)*(m-3) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2081_208191


namespace NUMINAMATH_CALUDE_log_product_equality_l2081_208180

theorem log_product_equality : 
  ∀ (x : ℝ), x > 0 → 
  (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
  (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
  (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
  (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
  (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
  (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) = 
  1 + Real.log 5 / Real.log 3 := by
sorry

end NUMINAMATH_CALUDE_log_product_equality_l2081_208180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_five_l2081_208137

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def sum_arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_first_five
  (a d : ℤ)
  (h1 : arithmetic_sequence a d 6 = 10)
  (h2 : arithmetic_sequence a d 7 = 15)
  (h3 : arithmetic_sequence a d 8 = 20) :
  sum_arithmetic_sequence a d 5 = -25 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_five_l2081_208137


namespace NUMINAMATH_CALUDE_E_and_G_complementary_l2081_208167

/-- The sample space of selecting 3 products from 100 products. -/
def Ω : Type := Unit

/-- The probability measure on the sample space. -/
def P : Ω → ℝ := sorry

/-- The event that all 3 selected products are non-defective. -/
def E : Set Ω := sorry

/-- The event that all 3 selected products are defective. -/
def F : Set Ω := sorry

/-- The event that at least one of the 3 selected products is defective. -/
def G : Set Ω := sorry

/-- The total number of products. -/
def total_products : ℕ := 100

/-- The number of defective products. -/
def defective_products : ℕ := 5

/-- The number of products selected. -/
def selected_products : ℕ := 3

theorem E_and_G_complementary :
  E ∪ G = Set.univ ∧ E ∩ G = ∅ :=
sorry

end NUMINAMATH_CALUDE_E_and_G_complementary_l2081_208167


namespace NUMINAMATH_CALUDE_bridge_length_l2081_208105

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2081_208105


namespace NUMINAMATH_CALUDE_worker_loading_time_l2081_208154

/-- The time taken by two workers to load a truck together -/
def combined_time : ℝ := 3.428571428571429

/-- The time taken by the second worker to load the truck alone -/
def second_worker_time : ℝ := 8

/-- The time taken by the first worker to load the truck alone -/
def first_worker_time : ℝ := 1.142857142857143

/-- Theorem stating the relationship between the workers' loading times -/
theorem worker_loading_time :
  (1 / combined_time) = (1 / first_worker_time) + (1 / second_worker_time) :=
by sorry

end NUMINAMATH_CALUDE_worker_loading_time_l2081_208154


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2081_208148

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → x^2 + y^2 + z^2 ≥ m ∧ a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2081_208148


namespace NUMINAMATH_CALUDE_andrews_age_l2081_208152

theorem andrews_age :
  ∀ (a g : ℝ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l2081_208152


namespace NUMINAMATH_CALUDE_system_solution_l2081_208130

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + 3 * y + 14 ≤ 0 ∧
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

-- Theorem stating that the solution to the system is (-2, -4)
theorem system_solution :
  ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (-2, -4) := by
  sorry


end NUMINAMATH_CALUDE_system_solution_l2081_208130


namespace NUMINAMATH_CALUDE_red_card_count_l2081_208194

theorem red_card_count (red_credit blue_credit total_cards total_credit : ℕ) 
  (h1 : red_credit = 3)
  (h2 : blue_credit = 5)
  (h3 : total_cards = 20)
  (h4 : total_credit = 84) :
  ∃ (red_cards blue_cards : ℕ),
    red_cards + blue_cards = total_cards ∧
    red_credit * red_cards + blue_credit * blue_cards = total_credit ∧
    red_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_card_count_l2081_208194


namespace NUMINAMATH_CALUDE_smallest_n_for_real_root_l2081_208101

/-- A polynomial with coefficients in [100, 101] -/
def PolynomialInRange (P : Polynomial ℝ) : Prop :=
  ∀ i, (100 : ℝ) ≤ P.coeff i ∧ P.coeff i ≤ 101

/-- The existence of a polynomial with a real root -/
def ExistsPolynomialWithRealRoot (n : ℕ) : Prop :=
  ∃ (P : Polynomial ℝ), PolynomialInRange P ∧ P.degree = 2*n ∧ ∃ x : ℝ, P.eval x = 0

/-- The main theorem stating that 100 is the smallest n for which a polynomial
    with coefficients in [100, 101] can have a real root -/
theorem smallest_n_for_real_root :
  (ExistsPolynomialWithRealRoot 100) ∧
  (∀ m : ℕ, m < 100 → ¬(ExistsPolynomialWithRealRoot m)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_real_root_l2081_208101


namespace NUMINAMATH_CALUDE_wire_cutting_l2081_208146

theorem wire_cutting (x : ℝ) :
  let total_length := Real.sqrt 600 + 12 * x
  let A := (Real.sqrt 600 + 15 * x - 9 * x^2) / 2
  let B := (Real.sqrt 600 + 9 * x - 9 * x^2) / 2
  let C := 9 * x^2
  (A = B + 3 * x) ∧
  (C = (A - B)^2) ∧
  (A + B + C = total_length) :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l2081_208146


namespace NUMINAMATH_CALUDE_solve_system_l2081_208100

theorem solve_system (x y : ℝ) (eq1 : x + y = 15) (eq2 : x - y = 5) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2081_208100


namespace NUMINAMATH_CALUDE_quadratic_solution_l2081_208134

theorem quadratic_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 11 * x - 20 = 0) : x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2081_208134
