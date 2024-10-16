import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l3547_354781

/-- Given that when x = 30, the value of ax³ + bx - 7 is 9,
    prove that the value of ax³ + bx + 2 when x = -30 is -14 -/
theorem expression_value (a b : ℝ) : 
  (30^3 * a + 30 * b - 7 = 9) → 
  ((-30)^3 * a + (-30) * b + 2 = -14) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3547_354781


namespace NUMINAMATH_CALUDE_point_groups_theorem_l3547_354720

theorem point_groups_theorem (n₁ n₂ : ℕ) : 
  n₁ + n₂ = 28 → 
  (n₁ * (n₁ - 1)) / 2 - (n₂ * (n₂ - 1)) / 2 = 81 → 
  (n₁ = 17 ∧ n₂ = 11) ∨ (n₁ = 11 ∧ n₂ = 17) := by
  sorry

end NUMINAMATH_CALUDE_point_groups_theorem_l3547_354720


namespace NUMINAMATH_CALUDE_triangle_side_count_l3547_354760

theorem triangle_side_count : ∃! n : ℕ, 
  n = (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x + 5 > 8 ∧ 8 + 5 > x
  ) (Finset.range 100)).card ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_count_l3547_354760


namespace NUMINAMATH_CALUDE_expression_decrease_l3547_354702

theorem expression_decrease (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  (x' * y' ^ 2) / (x * y ^ 2) = 0.216 := by sorry

end NUMINAMATH_CALUDE_expression_decrease_l3547_354702


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l3547_354731

/-- Given two nonconstant geometric sequences with different common ratios,
    if 3(a₃ - b₃) = 4(a₂ - b₂), then the sum of their common ratios is 4/3 -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ p r : ℝ) 
    (h1 : k ≠ 0)
    (h2 : p ≠ 1)
    (h3 : r ≠ 1)
    (h4 : p ≠ r)
    (h5 : a₂ = k * p)
    (h6 : a₃ = k * p^2)
    (h7 : b₂ = k * r)
    (h8 : b₃ = k * r^2)
    (h9 : 3 * (a₃ - b₃) = 4 * (a₂ - b₂)) :
  p + r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l3547_354731


namespace NUMINAMATH_CALUDE_floor_of_4_7_l3547_354714

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l3547_354714


namespace NUMINAMATH_CALUDE_circular_fortress_volume_l3547_354732

theorem circular_fortress_volume : 
  let base_circumference : ℝ := 48
  let height : ℝ := 11
  let π : ℝ := 3
  let radius := base_circumference / (2 * π)
  let volume := π * radius^2 * height
  volume = 2112 := by
  sorry

end NUMINAMATH_CALUDE_circular_fortress_volume_l3547_354732


namespace NUMINAMATH_CALUDE_remainder_theorem_l3547_354700

-- Define the polynomial p(x)
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^5 + B * x^3 + C * x + 4

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (p A B C 3 = 11) → (p A B C (-3) = -3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3547_354700


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3547_354788

theorem line_tangent_to_circle (m : ℝ) :
  (∀ x y : ℝ, x + y - m = 0 ∧ x^2 + y^2 = 2 → (∀ ε > 0, ∃ x' y' : ℝ, x' + y' - m = 0 ∧ x'^2 + y'^2 < 2 ∧ (x' - x)^2 + (y' - y)^2 < ε)) ↔
  (m > 2 ∨ m < -2) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3547_354788


namespace NUMINAMATH_CALUDE_max_median_soda_sales_l3547_354763

/-- Represents the soda sales data for a weekend -/
structure SodaSales where
  totalCans : ℕ
  totalCustomers : ℕ
  minCansPerCustomer : ℕ

/-- Calculates the maximum possible median number of cans bought per customer -/
def maxPossibleMedian (sales : SodaSales) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem max_median_soda_sales (sales : SodaSales)
  (h1 : sales.totalCans = 300)
  (h2 : sales.totalCustomers = 120)
  (h3 : sales.minCansPerCustomer = 2) :
  maxPossibleMedian sales = 3 :=
  sorry

end NUMINAMATH_CALUDE_max_median_soda_sales_l3547_354763


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l3547_354743

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 45) → (n * exterior_angle = 360) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l3547_354743


namespace NUMINAMATH_CALUDE_triangle_lines_correct_l3547_354758

/-- Triangle with vertices A(-5,0), B(3,-3), and C(0,2) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle : Triangle := { A := (-5, 0), B := (3, -3), C := (0, 2) }

/-- The equation of the line containing side BC -/
def line_BC : LineEquation := { a := 5, b := 3, c := -6 }

/-- The equation of the line containing the altitude from A to side BC -/
def altitude_A : LineEquation := { a := 5, b := 2, c := 25 }

theorem triangle_lines_correct (t : Triangle) (bc : LineEquation) (alt : LineEquation) :
  t = triangle → bc = line_BC → alt = altitude_A := by sorry

end NUMINAMATH_CALUDE_triangle_lines_correct_l3547_354758


namespace NUMINAMATH_CALUDE_fifteen_switches_connections_l3547_354755

/-- The number of unique connections in a network of switches -/
def uniqueConnections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 15 switches, where each switch connects to 
    exactly 4 other switches, the total number of unique connections is 30. -/
theorem fifteen_switches_connections : 
  uniqueConnections 15 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_switches_connections_l3547_354755


namespace NUMINAMATH_CALUDE_percentage_of_360_l3547_354717

theorem percentage_of_360 : (33 + 1 / 3 : ℚ) / 100 * 360 = 120 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_l3547_354717


namespace NUMINAMATH_CALUDE_eight_operations_proof_l3547_354747

theorem eight_operations_proof :
  (((8 : ℝ) / 8) * (8 / 8) = 1) ∧
  ((8 : ℝ) / 8 + 8 / 8 = 2) := by
  sorry

end NUMINAMATH_CALUDE_eight_operations_proof_l3547_354747


namespace NUMINAMATH_CALUDE_quadratic_roots_d_value_l3547_354762

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_d_value_l3547_354762


namespace NUMINAMATH_CALUDE_farm_tax_land_percentage_l3547_354738

theorem farm_tax_land_percentage 
  (total_tax : ℝ) 
  (individual_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_tax = 480) :
  individual_tax / total_tax = 0.125 := by
sorry

end NUMINAMATH_CALUDE_farm_tax_land_percentage_l3547_354738


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l3547_354759

/-- The binary representation of the number --/
def binary_num : List Bool := [true, true, false, true, true, true, false, false, true, false, true, true]

/-- Convert a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Get the last three digits of a binary number --/
def last_three_digits (binary : List Bool) : List Bool :=
  binary.reverse.take 3

theorem remainder_of_binary_div_8 :
  binary_to_decimal (last_three_digits binary_num) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l3547_354759


namespace NUMINAMATH_CALUDE_speed_of_car_C_l3547_354722

/-- Proves that given the conditions of the problem, the speed of car C is 26 km/h --/
theorem speed_of_car_C (v_A v_B : ℝ) (t_A t_B t_C : ℝ) :
  v_A = 24 →
  v_B = 20 →
  t_A = 5 / 60 →
  t_B = 10 / 60 →
  t_C = 12 / 60 →
  v_A * t_A = v_B * t_B →
  ∃ (v_C : ℝ), v_C * t_C = v_A * t_A ∧ v_C = 26 :=
by sorry

#check speed_of_car_C

end NUMINAMATH_CALUDE_speed_of_car_C_l3547_354722


namespace NUMINAMATH_CALUDE_farmer_problem_l3547_354736

theorem farmer_problem (total_cost : ℕ) (rabbit_cost chicken_cost : ℕ) 
  (h_total : total_cost = 1125)
  (h_rabbit : rabbit_cost = 30)
  (h_chicken : chicken_cost = 45) :
  ∃! (r c : ℕ), 
    r > 0 ∧ c > 0 ∧ 
    r * rabbit_cost + c * chicken_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_problem_l3547_354736


namespace NUMINAMATH_CALUDE_kody_half_age_of_mohamed_l3547_354739

def years_ago (mohamed_current_age kody_current_age : ℕ) : ℕ :=
  let x : ℕ := 4
  x

theorem kody_half_age_of_mohamed (mohamed_current_age kody_current_age : ℕ)
  (h1 : mohamed_current_age = 2 * 30)
  (h2 : kody_current_age = 32)
  (h3 : ∃ x : ℕ, kody_current_age - x = (mohamed_current_age - x) / 2) :
  years_ago mohamed_current_age kody_current_age = 4 := by
sorry

end NUMINAMATH_CALUDE_kody_half_age_of_mohamed_l3547_354739


namespace NUMINAMATH_CALUDE_store_transaction_result_l3547_354785

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

end NUMINAMATH_CALUDE_store_transaction_result_l3547_354785


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l3547_354706

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (φ : ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|)
  (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l3547_354706


namespace NUMINAMATH_CALUDE_ice_cream_sales_ratio_l3547_354769

/-- Ice cream sales problem -/
theorem ice_cream_sales_ratio (tuesday_sales wednesday_sales : ℕ) : 
  tuesday_sales = 12000 →
  wednesday_sales = 36000 - tuesday_sales →
  (wednesday_sales : ℚ) / tuesday_sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_ratio_l3547_354769


namespace NUMINAMATH_CALUDE_subtraction_problem_l3547_354787

theorem subtraction_problem (A B : ℕ) : 
  (A ≥ 10 ∧ A ≤ 99) → 
  (B ≥ 10 ∧ B ≤ 99) → 
  A = 23 - 8 → 
  B + 7 = 18 → 
  A - B = 4 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3547_354787


namespace NUMINAMATH_CALUDE_kendra_toy_purchase_l3547_354709

/-- The price of a wooden toy -/
def toy_price : ℕ := 20

/-- The price of a hat -/
def hat_price : ℕ := 10

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount of money Kendra started with -/
def initial_money : ℕ := 100

/-- The amount of change Kendra received -/
def change_received : ℕ := 30

/-- The number of wooden toys Kendra bought -/
def toys_bought : ℕ := 2

theorem kendra_toy_purchase :
  toy_price * toys_bought + hat_price * hats_bought = initial_money - change_received :=
by sorry

end NUMINAMATH_CALUDE_kendra_toy_purchase_l3547_354709


namespace NUMINAMATH_CALUDE_distinct_digit_numbers_count_l3547_354723

/-- A function that counts the number of integers between 1000 and 9999 with four distinct digits -/
def count_distinct_digit_numbers : ℕ :=
  9 * 9 * 8 * 7

/-- The theorem stating that the count of integers between 1000 and 9999 with four distinct digits is 4536 -/
theorem distinct_digit_numbers_count :
  count_distinct_digit_numbers = 4536 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_numbers_count_l3547_354723


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3547_354752

theorem fraction_multiplication : (1 : ℚ) / 2 * 3 / 5 * 7 / 11 = 21 / 110 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3547_354752


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l3547_354766

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  324 > 300 ∧
  324 % 25 = 24 ∧
  ∀ m : ℕ, m > 300 ∧ m % 25 = 24 → m ≥ 324 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l3547_354766


namespace NUMINAMATH_CALUDE_fence_painting_combinations_l3547_354774

def number_of_colors : ℕ := 5
def number_of_tools : ℕ := 4

theorem fence_painting_combinations :
  number_of_colors * number_of_tools = 20 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_combinations_l3547_354774


namespace NUMINAMATH_CALUDE_positive_integer_solutions_l3547_354729

theorem positive_integer_solutions :
  ∀ x y z : ℕ+,
    x < y →
    2 * (x + 1) * (y + 1) - 1 = x * y * z →
    ((x = 1 ∧ y = 3 ∧ z = 5) ∨ (x = 3 ∧ y = 7 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l3547_354729


namespace NUMINAMATH_CALUDE_smallest_integer_y_l3547_354767

theorem smallest_integer_y : ∃ y : ℤ, (1 : ℚ) / 4 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 : ℚ) / 4 < (z : ℚ) / 7 ∧ (z : ℚ) / 7 < 2 / 3 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l3547_354767


namespace NUMINAMATH_CALUDE_roxy_daily_consumption_l3547_354728

/-- Represents the daily water consumption of the siblings --/
structure WaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Represents the total weekly water consumption of the siblings --/
def weekly_total (wc : WaterConsumption) : ℕ :=
  7 * (wc.theo + wc.mason + wc.roxy)

/-- Theorem stating that given the conditions, Roxy drinks 9 cups of water daily --/
theorem roxy_daily_consumption (wc : WaterConsumption) :
  wc.theo = 8 → wc.mason = 7 → weekly_total wc = 168 → wc.roxy = 9 := by
  sorry

#check roxy_daily_consumption

end NUMINAMATH_CALUDE_roxy_daily_consumption_l3547_354728


namespace NUMINAMATH_CALUDE_array_exists_iff_even_l3547_354754

/-- A type representing the possible entries in the array -/
inductive Entry
  | neg : Entry
  | zero : Entry
  | pos : Entry

/-- Definition of a valid array -/
def ValidArray (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j : Fin n), arr i j ∈ [Entry.neg, Entry.zero, Entry.pos]

/-- Definition of row sum -/
def RowSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (i : Fin n) : ℤ :=
  (Finset.univ.sum fun j => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- Definition of column sum -/
def ColSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (j : Fin n) : ℤ :=
  (Finset.univ.sum fun i => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- All sums are different -/
def AllSumsDifferent (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j i' j' : Fin n), 
    (RowSum n arr i = RowSum n arr i' → i = i') ∧
    (ColSum n arr j = ColSum n arr j' → j = j') ∧
    (RowSum n arr i ≠ ColSum n arr j)

/-- Main theorem: The array with described properties exists if and only if n is even -/
theorem array_exists_iff_even (n : ℕ) :
  (∃ (arr : Matrix (Fin n) (Fin n) Entry), 
    ValidArray n arr ∧ AllSumsDifferent n arr) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_array_exists_iff_even_l3547_354754


namespace NUMINAMATH_CALUDE_length_MN_l3547_354751

/-- The length of MN where M and N are points on two lines and S is their midpoint -/
theorem length_MN (M N S : ℝ × ℝ) : 
  S = (10, 8) →
  (∃ x₁, M = (x₁, 14 * x₁ / 9)) →
  (∃ x₂, N = (x₂, 5 * x₂ / 12)) →
  S.1 = (M.1 + N.1) / 2 →
  S.2 = (M.2 + N.2) / 2 →
  ∃ length, length = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_length_MN_l3547_354751


namespace NUMINAMATH_CALUDE_problem_solution_l3547_354718

theorem problem_solution (a b : ℝ) 
  (h1 : a^2 + 2*b = 0) 
  (h2 : |a^2 - 2*b| = 8) : 
  b + 2023 = 2021 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3547_354718


namespace NUMINAMATH_CALUDE_addition_is_unique_solution_l3547_354726

-- Define the possible operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 7 (-7) = 0 ∧ 
  (op = Operation.Add ∨ op = Operation.Sub ∨ op = Operation.Mul ∨ op = Operation.Div) :=
by sorry

end NUMINAMATH_CALUDE_addition_is_unique_solution_l3547_354726


namespace NUMINAMATH_CALUDE_A_3_2_l3547_354715

def A : Nat → Nat → Nat
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l3547_354715


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3547_354794

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3547_354794


namespace NUMINAMATH_CALUDE_exists_right_triangle_different_colors_l3547_354737

-- Define the color type
inductive Color
| Blue
| Green
| Red

-- Define the plane as a type
def Plane := ℝ × ℝ

-- Define a coloring function
def coloring : Plane → Color := sorry

-- Define the existence of at least one point of each color
axiom exists_blue : ∃ p : Plane, coloring p = Color.Blue
axiom exists_green : ∃ p : Plane, coloring p = Color.Green
axiom exists_red : ∃ p : Plane, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Plane) : Prop := sorry

-- Theorem statement
theorem exists_right_triangle_different_colors :
  ∃ p q r : Plane, 
    is_right_triangle p q r ∧ 
    coloring p ≠ coloring q ∧ 
    coloring q ≠ coloring r ∧ 
    coloring r ≠ coloring p :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_different_colors_l3547_354737


namespace NUMINAMATH_CALUDE_consumer_installment_credit_l3547_354775

theorem consumer_installment_credit (total_credit : ℝ) : 
  (0.36 * total_credit = 3 * 57) → total_credit = 475 := by
  sorry

end NUMINAMATH_CALUDE_consumer_installment_credit_l3547_354775


namespace NUMINAMATH_CALUDE_geometric_series_equation_l3547_354725

theorem geometric_series_equation (x : ℝ) : x = 9 →
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equation_l3547_354725


namespace NUMINAMATH_CALUDE_parabola_properties_l3547_354713

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hc : c > 1) 
  (h_point : parabola a b c 2 = 0) 
  (h_symmetry : -b / (2 * a) = 1/2) :
  abc < 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = a ∧ parabola a b c x₂ = a) ∧
  a < -1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3547_354713


namespace NUMINAMATH_CALUDE_sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l3547_354740

/-- Given an angle θ, prove that sin²θ + cos²θ = 1 -/
theorem sin_sq_plus_cos_sq_eq_one (θ : Real) : (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

/-- Given s = sin θ and c = cos θ for some angle θ, prove that s² + c² = 1 -/
theorem s_sq_plus_c_sq_eq_one (s c : Real) (h : ∃ θ : Real, s = Real.sin θ ∧ c = Real.cos θ) : s^2 + c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_sq_plus_cos_sq_eq_one_s_sq_plus_c_sq_eq_one_l3547_354740


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_l3547_354786

def numbers : List Nat := [18, 24, 36]

def C : Nat := numbers.foldl Nat.gcd 0

def D : Nat := numbers.foldl Nat.lcm 1

theorem sum_gcd_lcm : C + D = 78 := by sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_l3547_354786


namespace NUMINAMATH_CALUDE_p_squared_plus_18_composite_l3547_354790

theorem p_squared_plus_18_composite (p : ℕ) (hp : Prime p) : ¬ Prime (p^2 + 18) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_18_composite_l3547_354790


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3547_354765

theorem rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3547_354765


namespace NUMINAMATH_CALUDE_min_average_of_four_integers_l3547_354711

theorem min_average_of_four_integers (a b c d : ℕ+) 
  (ha : a = 3 * b)
  (hc : c = b + 2)
  (hd : d ≥ 2) :
  (a + b + c + d : ℚ) / 4 ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_average_of_four_integers_l3547_354711


namespace NUMINAMATH_CALUDE_graphs_with_inverses_l3547_354793

-- Define the types of graphs
inductive GraphType
| Linear
| Parabola
| DisconnectedLinear
| Semicircle
| Cubic

-- Define a function to check if a graph has an inverse
def has_inverse (g : GraphType) : Prop :=
  match g with
  | GraphType.Linear => true
  | GraphType.Parabola => false
  | GraphType.DisconnectedLinear => true
  | GraphType.Semicircle => false
  | GraphType.Cubic => false

-- Define the specific graphs given in the problem
def graph_A : GraphType := GraphType.Linear
def graph_B : GraphType := GraphType.Parabola
def graph_C : GraphType := GraphType.DisconnectedLinear
def graph_D : GraphType := GraphType.Semicircle
def graph_E : GraphType := GraphType.Cubic

-- Theorem stating which graphs have inverses
theorem graphs_with_inverses :
  (has_inverse graph_A ∧ has_inverse graph_C) ∧
  (¬has_inverse graph_B ∧ ¬has_inverse graph_D ∧ ¬has_inverse graph_E) :=
by sorry

end NUMINAMATH_CALUDE_graphs_with_inverses_l3547_354793


namespace NUMINAMATH_CALUDE_min_y_value_l3547_354727

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 64*y) : 
  ∃ (y_min : ℝ), y_min = 32 - 2 * Real.sqrt 281 ∧ 
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 64*y' → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_value_l3547_354727


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3547_354796

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : ∃ r, (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r) : 
  (a 3) / (a 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3547_354796


namespace NUMINAMATH_CALUDE_unifying_sqrt_plus_m_range_l3547_354724

/-- A function is unifying on [a,b] if it's monotonic and maps [a,b] onto itself --/
def IsUnifying (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧ 
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The theorem stating the range of m for which f(x) = √(x+1) + m is a unifying function --/
theorem unifying_sqrt_plus_m_range :
  ∃ a b : ℝ, a < b ∧ 
  (∃ m : ℝ, IsUnifying (fun x ↦ Real.sqrt (x + 1) + m) a b) ↔ 
  m ∈ Set.Ioo (-5/4) (-1) ∪ {-1} :=
sorry

end NUMINAMATH_CALUDE_unifying_sqrt_plus_m_range_l3547_354724


namespace NUMINAMATH_CALUDE_probability_system_l3547_354716

/-- Given a probability system with parameters p and q, prove that the probabilities x, y, and z satisfy specific relations. -/
theorem probability_system (p q x y z : ℝ) : 
  z = p * y + q * x → 
  x = p + q * x^2 → 
  y = q + p * y^2 → 
  x ≠ y → 
  p + q = 1 → 
  0 ≤ p ∧ p ≤ 1 → 
  0 ≤ q ∧ q ≤ 1 → 
  0 ≤ x ∧ x ≤ 1 → 
  0 ≤ y ∧ y ≤ 1 → 
  0 ≤ z ∧ z ≤ 1 → 
  x = 1 ∧ y = q / p ∧ z = 2 * q :=
by sorry

end NUMINAMATH_CALUDE_probability_system_l3547_354716


namespace NUMINAMATH_CALUDE_total_spent_on_games_l3547_354750

def batman_game_cost : ℚ := 13.60
def superman_game_cost : ℚ := 5.06

theorem total_spent_on_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_games_l3547_354750


namespace NUMINAMATH_CALUDE_train_length_problem_l3547_354782

/-- The length of two trains passing each other on parallel tracks --/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 50 * (5/18) →
  slower_speed = 36 * (5/18) →
  passing_time = 36 →
  ∃ (train_length : ℝ), train_length = 70 ∧ 
    2 * train_length = (faster_speed - slower_speed) * passing_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_problem_l3547_354782


namespace NUMINAMATH_CALUDE_inequality_implication_l3547_354712

theorem inequality_implication (a b : ℝ) (h : a > b) : -6*a < -6*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3547_354712


namespace NUMINAMATH_CALUDE_bricks_for_wall_l3547_354744

/-- Calculates the number of bricks needed to build a wall -/
def bricks_needed (wall_length wall_height wall_thickness brick_length brick_width brick_height : ℕ) : ℕ :=
  let wall_volume := wall_length * wall_height * wall_thickness
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume

/-- Theorem stating the number of bricks needed for the given wall and brick dimensions -/
theorem bricks_for_wall : bricks_needed 800 600 2 5 11 6 = 2910 := by
  sorry

end NUMINAMATH_CALUDE_bricks_for_wall_l3547_354744


namespace NUMINAMATH_CALUDE_order_relation_abc_l3547_354708

/-- Prove that given a = (4 - ln 4) / e^2, b = ln 2 / 2, and c = 1/e, we have b < a < c -/
theorem order_relation_abc :
  let a : ℝ := (4 - Real.log 4) / Real.exp 2
  let b : ℝ := Real.log 2 / 2
  let c : ℝ := 1 / Real.exp 1
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_relation_abc_l3547_354708


namespace NUMINAMATH_CALUDE_three_hundredth_term_of_sequence_l3547_354733

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem three_hundredth_term_of_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 8) (h₂ : a₂ = -8) :
  geometric_sequence a₁ (a₂ / a₁) 300 = -8 := by
  sorry

end NUMINAMATH_CALUDE_three_hundredth_term_of_sequence_l3547_354733


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3547_354721

/-- The area of the union of a square with side length 12 and a circle with radius 12
    centered at one of the square's vertices is equal to 144 + 108π. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let quarter_circle_area : ℝ := circle_area / 4
  let union_area : ℝ := square_area + circle_area - quarter_circle_area
  union_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3547_354721


namespace NUMINAMATH_CALUDE_train_passing_platform_l3547_354761

/-- Given a train of length 240 meters passing a pole in 24 seconds,
    prove that it takes 89 seconds to pass a platform of length 650 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 240)
  (h2 : pole_passing_time = 24)
  (h3 : platform_length = 650) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 89 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3547_354761


namespace NUMINAMATH_CALUDE_perimeter_difference_l3547_354772

/-- Calculates the perimeter of a rectangle given its width and height. -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- Calculates the perimeter of a cross-shaped figure composed of 5 unit squares. -/
def cross_perimeter : ℕ := 8

/-- Theorem stating the difference between the perimeters of a 4x3 rectangle and a cross-shaped figure. -/
theorem perimeter_difference : 
  (rectangle_perimeter 4 3) - cross_perimeter = 6 := by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l3547_354772


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l3547_354780

theorem pencil_eraser_cost_problem : ∃ (p e : ℕ), 
  13 * p + 3 * e = 100 ∧ 
  p > e ∧ 
  p + e = 10 := by
sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l3547_354780


namespace NUMINAMATH_CALUDE_right_triangle_count_l3547_354741

/-- Count of right triangles with integer leg lengths a and b, hypotenuse b+2, and b < 50 -/
theorem right_triangle_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a * a + b * b = (b + 2) * (b + 2) ∧ 
    0 < a ∧ 
    0 < b ∧ 
    b < 50
  ) (Finset.product (Finset.range 200) (Finset.range 50))).card = 7 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_count_l3547_354741


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_condition_l3547_354773

theorem quadratic_no_real_roots_condition (m x : ℝ) : 
  (∀ x, x^2 - 2*x + m ≠ 0) → m ≥ 0 ∧ 
  ∃ m₀ ≥ 0, ∃ x₀, x₀^2 - 2*x₀ + m₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_condition_l3547_354773


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l3547_354798

def camp_cedar (num_boys : ℕ) (girl_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

theorem camp_cedar_counselors :
  camp_cedar 40 3 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselors_l3547_354798


namespace NUMINAMATH_CALUDE_sector_angle_l3547_354703

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) : 
  (1/2 * θ * r^2 = 1) → (2*r + θ*r = 4) → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3547_354703


namespace NUMINAMATH_CALUDE_maintenance_check_interval_l3547_354748

theorem maintenance_check_interval (original : ℝ) (new : ℝ) : 
  new = 1.5 * original → new = 45 → original = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_interval_l3547_354748


namespace NUMINAMATH_CALUDE_function_simplification_and_sum_l3547_354779

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x - 3)

theorem function_simplification_and_sum :
  ∃ (A B C D : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A * x^2 + B * x + C) ∧
    (∀ x : ℝ, f x = A * x^2 + B * x + C ↔ x ≠ D) ∧
    A + B + C + D = 24 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_and_sum_l3547_354779


namespace NUMINAMATH_CALUDE_round_4995000_to_million_l3547_354791

/-- Round a natural number to the nearest million -/
def round_to_million (n : ℕ) : ℕ :=
  if n % 1000000 ≥ 500000 then
    ((n + 500000) / 1000000) * 1000000
  else
    (n / 1000000) * 1000000

/-- Theorem: Rounding 4995000 to the nearest million equals 5000000 -/
theorem round_4995000_to_million :
  round_to_million 4995000 = 5000000 := by
  sorry

end NUMINAMATH_CALUDE_round_4995000_to_million_l3547_354791


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3547_354734

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  c = 12 →
  C = π / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3547_354734


namespace NUMINAMATH_CALUDE_cookies_sold_first_village_l3547_354795

/-- Given the total number of packs sold and the number sold in the second village,
    calculate the number of packs sold in the first village. -/
theorem cookies_sold_first_village 
  (total_packs : ℕ) 
  (second_village_packs : ℕ) 
  (h1 : total_packs = 51) 
  (h2 : second_village_packs = 28) : 
  total_packs - second_village_packs = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_first_village_l3547_354795


namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l3547_354757

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, f (-x) = f x) 
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) : 
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 0) := by
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l3547_354757


namespace NUMINAMATH_CALUDE_infinitely_many_coprimes_in_arithmetic_sequence_l3547_354764

theorem infinitely_many_coprimes_in_arithmetic_sequence 
  (a b m : ℕ+) (h : Nat.Coprime a b) :
  ∃ (s : Set ℕ), Set.Infinite s ∧ ∀ k ∈ s, Nat.Coprime (a + k * b) m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprimes_in_arithmetic_sequence_l3547_354764


namespace NUMINAMATH_CALUDE_min_value_of_f_l3547_354730

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3547_354730


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3547_354756

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3547_354756


namespace NUMINAMATH_CALUDE_unique_scores_count_l3547_354735

/-- Represents the number of baskets made by the player -/
def total_baskets : ℕ := 7

/-- Represents the possible point values for each basket -/
inductive BasketType
| two_point : BasketType
| three_point : BasketType

/-- Calculates the total score given a list of basket types -/
def calculate_score (baskets : List BasketType) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketType.two_point => 2
    | BasketType.three_point => 3) 0

/-- Generates all possible combinations of basket types -/
def generate_combinations : List (List BasketType) :=
  sorry

/-- Theorem stating that the number of unique possible scores is 8 -/
theorem unique_scores_count :
  (generate_combinations.map calculate_score).toFinset.card = 8 := by sorry

end NUMINAMATH_CALUDE_unique_scores_count_l3547_354735


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3547_354704

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the circle
structure Circle where
  radius : ℝ

-- Define the problem conditions
def tangent_and_midpoint (rect : Rectangle) (circ : Circle) : Prop :=
  -- Circle is tangent to sides EF and EH at their midpoints
  -- and passes through the midpoint of side FG
  True

-- Theorem statement
theorem rectangle_area_theorem (rect : Rectangle) (circ : Circle) :
  tangent_and_midpoint rect circ →
  rect.width * rect.height = 4 * circ.radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3547_354704


namespace NUMINAMATH_CALUDE_forest_logging_time_l3547_354771

/-- Represents a logging team with its characteristics -/
structure LoggingTeam where
  loggers : ℕ
  daysPerWeek : ℕ
  treesPerLoggerPerDay : ℕ

/-- Calculates the number of months needed to cut down all trees in the forest -/
def monthsToLogForest (forestWidth : ℕ) (forestLength : ℕ) (treesPerSquareMile : ℕ) 
  (teams : List LoggingTeam) (daysPerMonth : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that it takes 5 months to log the entire forest -/
theorem forest_logging_time : 
  let forestWidth := 4
  let forestLength := 6
  let treesPerSquareMile := 600
  let teamA := LoggingTeam.mk 6 5 5
  let teamB := LoggingTeam.mk 8 4 6
  let teamC := LoggingTeam.mk 10 3 8
  let teamD := LoggingTeam.mk 12 2 10
  let teams := [teamA, teamB, teamC, teamD]
  let daysPerMonth := 30
  monthsToLogForest forestWidth forestLength treesPerSquareMile teams daysPerMonth = 5 :=
  sorry

end NUMINAMATH_CALUDE_forest_logging_time_l3547_354771


namespace NUMINAMATH_CALUDE_certain_number_proof_l3547_354701

theorem certain_number_proof (k : ℕ) (n : ℕ) : 
  (6^k - k^6 = 1) → (18^k ∣ n) → (k = 0 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3547_354701


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l3547_354778

theorem cryptarithm_solution :
  ∃! (C H U K T R I G N S : ℕ),
    C < 10 ∧ H < 10 ∧ U < 10 ∧ K < 10 ∧ T < 10 ∧ R < 10 ∧ I < 10 ∧ G < 10 ∧ N < 10 ∧ S < 10 ∧
    T ≠ 0 ∧
    C ≠ H ∧ C ≠ U ∧ C ≠ K ∧ C ≠ T ∧ C ≠ R ∧ C ≠ I ∧ C ≠ G ∧ C ≠ N ∧ C ≠ S ∧
    H ≠ U ∧ H ≠ K ∧ H ≠ T ∧ H ≠ R ∧ H ≠ I ∧ H ≠ G ∧ H ≠ N ∧ H ≠ S ∧
    U ≠ K ∧ U ≠ T ∧ U ≠ R ∧ U ≠ I ∧ U ≠ G ∧ U ≠ N ∧ U ≠ S ∧
    K ≠ T ∧ K ≠ R ∧ K ≠ I ∧ K ≠ G ∧ K ≠ N ∧ K ≠ S ∧
    T ≠ R ∧ T ≠ I ∧ T ≠ G ∧ T ≠ N ∧ T ≠ S ∧
    R ≠ I ∧ R ≠ G ∧ R ≠ N ∧ R ≠ S ∧
    I ≠ G ∧ I ≠ N ∧ I ≠ S ∧
    G ≠ N ∧ G ≠ S ∧
    N ≠ S ∧
    100000*C + 10000*H + 1000*U + 100*C + 10*K +
    100000*T + 10000*R + 1000*I + 100*G + 10*G +
    100000*T + 10000*U + 1000*R + 100*N + 10*S =
    100000*T + 10000*R + 1000*I + 100*C + 10*K + S ∧
    C = 9 ∧ H = 3 ∧ U = 5 ∧ K = 4 ∧ T = 1 ∧ R = 2 ∧ I = 0 ∧ G = 6 ∧ N = 8 ∧ S = 7 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l3547_354778


namespace NUMINAMATH_CALUDE_percentage_difference_l3547_354770

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 83.33333333333333 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3547_354770


namespace NUMINAMATH_CALUDE_inverse_square_relation_l3547_354745

/-- Given that x varies inversely as the square of y, and y = 2 when x = 1,
    prove that x = 1/9 when y = 6 -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
    (y = 6) → (x = 1/9) := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l3547_354745


namespace NUMINAMATH_CALUDE_lulu_cupcakes_count_l3547_354710

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lulu baked 16 mini cupcakes -/
theorem lulu_cupcakes_count (p : Pastries) 
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lola_pies = 8)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lulu_cupcakes = 16 := by
  sorry

end NUMINAMATH_CALUDE_lulu_cupcakes_count_l3547_354710


namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l3547_354753

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 2*a = 0) :
  a + b ≥ (10/9)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l3547_354753


namespace NUMINAMATH_CALUDE_expression_simplification_l3547_354783

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hd : 2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2 ≠ 0) : 
  (Real.sqrt 3 * (a - b^2) + Real.sqrt 3 * b * (8 * b^3)^(1/3)) / 
  Real.sqrt (2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2) * 
  (Real.sqrt (2 * a) - Real.sqrt (2 * c)) / 
  (Real.sqrt (3 / a) - Real.sqrt (3 / c)) = 
  -Real.sqrt (a * c) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3547_354783


namespace NUMINAMATH_CALUDE_power_multiplication_problem_solution_l3547_354776

theorem power_multiplication (n : ℕ) : n * (n ^ n) = n ^ (n + 1) := by sorry

theorem problem_solution : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  apply power_multiplication

end NUMINAMATH_CALUDE_power_multiplication_problem_solution_l3547_354776


namespace NUMINAMATH_CALUDE_unique_modulo_representation_l3547_354707

theorem unique_modulo_representation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulo_representation_l3547_354707


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l3547_354742

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 7
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 1

-- State the theorem
theorem polynomial_sum_equality :
  ∀ x : ℝ, p x + q x + r x + s x = -2 * x^2 + 9 * x - 11 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l3547_354742


namespace NUMINAMATH_CALUDE_at_least_one_nonzero_l3547_354705

theorem at_least_one_nonzero (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) ↔ a^2 + b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_nonzero_l3547_354705


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l3547_354749

/-- Calculates the total distance traveled when walking and running at different rates for different durations. -/
theorem total_distance_walked_and_run 
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 45 →
  walking_rate = 4 →
  running_time = 30 →
  running_rate = 10 →
  (walking_time / 60) * walking_rate + (running_time / 60) * running_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l3547_354749


namespace NUMINAMATH_CALUDE_product_of_numbers_l3547_354789

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 220) : x * y = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3547_354789


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3547_354797

theorem absolute_value_equality (x : ℝ) (h : x > 0) :
  |x + Real.sqrt ((x + 1)^2)| = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3547_354797


namespace NUMINAMATH_CALUDE_problem_solution_l3547_354777

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 2)
  (eq2 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 24)
  (eq3 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 246)
  (eq4 : 25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ + 144*x₈ = 1234) :
  36*x₁ + 49*x₂ + 64*x₃ + 81*x₄ + 100*x₅ + 121*x₆ + 144*x₇ + 169*x₈ = 1594 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3547_354777


namespace NUMINAMATH_CALUDE_animal_population_canada_animal_population_l3547_354799

/-- The combined population of moose, beavers, caribou, wolves, grizzly bears, and mountain lions in Canada, given the specified ratios and human population. -/
theorem animal_population (human_population : ℝ) : ℝ :=
  let beaver_population := human_population / 19
  let moose_population := beaver_population / 2
  let caribou_population := 3/2 * moose_population
  let wolf_population := 4 * caribou_population
  let grizzly_population := wolf_population / 3
  let mountain_lion_population := grizzly_population / 2
  moose_population + beaver_population + caribou_population + wolf_population + grizzly_population + mountain_lion_population

/-- Theorem stating that the combined animal population in Canada is 13.5 million, given a human population of 38 million. -/
theorem canada_animal_population :
  animal_population 38 = 13.5 := by sorry

end NUMINAMATH_CALUDE_animal_population_canada_animal_population_l3547_354799


namespace NUMINAMATH_CALUDE_brick_width_calculation_l3547_354719

/-- Proves that given a courtyard of 25 meters by 15 meters, to be paved with 18750 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_length = 0.2 →
  total_bricks = 18750 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * 100) * (courtyard_width * 100) = 
      total_bricks * brick_length * 100 * brick_width * 100 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l3547_354719


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3547_354746

/-- Proves that the sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees (θ : Real) : θ = π / 6 → Real.sin θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3547_354746


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l3547_354792

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmeticSequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l3547_354792


namespace NUMINAMATH_CALUDE_special_natural_numbers_l3547_354768

theorem special_natural_numbers : 
  {x : ℕ | ∃ (y z : ℤ), x = 2 * y^2 - 1 ∧ x^2 = 2 * z^2 - 1} = {1, 7} := by
  sorry

end NUMINAMATH_CALUDE_special_natural_numbers_l3547_354768


namespace NUMINAMATH_CALUDE_simplify_expression_l3547_354784

theorem simplify_expression : (9 * 10^8) / (3 * 10^3) = 300000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3547_354784
