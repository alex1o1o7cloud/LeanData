import Mathlib

namespace NUMINAMATH_CALUDE_sara_balloons_l1708_170872

theorem sara_balloons (tom_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : tom_balloons = 9)
  (h2 : total_balloons = 17) :
  total_balloons - tom_balloons = 8 := by sorry

end NUMINAMATH_CALUDE_sara_balloons_l1708_170872


namespace NUMINAMATH_CALUDE_bank_transfer_balance_l1708_170880

theorem bank_transfer_balance (initial_balance first_transfer second_transfer service_charge_rate : ℝ) 
  (h1 : initial_balance = 400)
  (h2 : first_transfer = 90)
  (h3 : second_transfer = 60)
  (h4 : service_charge_rate = 0.02)
  : initial_balance - (first_transfer + first_transfer * service_charge_rate + second_transfer * service_charge_rate) = 307 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_balance_l1708_170880


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1708_170859

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z + 1) = -3 + 2 * Complex.I) → z = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1708_170859


namespace NUMINAMATH_CALUDE_total_bottles_proof_l1708_170839

/-- Represents the total number of bottles -/
def total_bottles : ℕ := 180

/-- Represents the number of bottles containing only cider -/
def cider_bottles : ℕ := 40

/-- Represents the number of bottles containing only beer -/
def beer_bottles : ℕ := 80

/-- Represents the number of bottles given to the first house -/
def first_house_bottles : ℕ := 90

/-- Proves that the total number of bottles is 180 given the problem conditions -/
theorem total_bottles_proof :
  total_bottles = cider_bottles + beer_bottles + (2 * first_house_bottles - cider_bottles - beer_bottles) :=
by sorry

end NUMINAMATH_CALUDE_total_bottles_proof_l1708_170839


namespace NUMINAMATH_CALUDE_sales_and_profit_l1708_170831

theorem sales_and_profit (x : ℤ) (y : ℝ) : 
  (8 ≤ x ∧ x ≤ 15) →
  (y = -5 * (x : ℝ) + 150) →
  (y = 105 ↔ x = 9) →
  (y = 95 ↔ x = 11) →
  (y = 85 ↔ x = 13) →
  (∃ (x : ℤ), 8 ≤ x ∧ x ≤ 15 ∧ (x - 8) * (-5 * x + 150) = 425 ↔ x = 13) :=
by sorry

end NUMINAMATH_CALUDE_sales_and_profit_l1708_170831


namespace NUMINAMATH_CALUDE_fuel_station_theorem_l1708_170891

/-- Represents the fuel station problem --/
def fuel_station_problem (service_cost : ℚ) (fuel_cost_per_liter : ℚ) 
  (num_minivans : ℕ) (num_trucks : ℕ) (total_cost : ℚ) (minivan_tank : ℚ) : Prop :=
  let total_service_cost := (num_minivans + num_trucks : ℚ) * service_cost
  let total_fuel_cost := total_cost - total_service_cost
  let minivan_fuel_cost := (num_minivans : ℚ) * minivan_tank * fuel_cost_per_liter
  let truck_fuel_cost := total_fuel_cost - minivan_fuel_cost
  let truck_fuel_liters := truck_fuel_cost / fuel_cost_per_liter
  let truck_tank := truck_fuel_liters / (num_trucks : ℚ)
  let percentage_increase := (truck_tank - minivan_tank) / minivan_tank * 100
  percentage_increase = 120

/-- The main theorem to be proved --/
theorem fuel_station_theorem : 
  fuel_station_problem 2.2 0.7 4 2 395.4 65 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_theorem_l1708_170891


namespace NUMINAMATH_CALUDE_sliced_meat_cost_l1708_170883

/-- Given a 4-pack of sliced meat costing $40.00 with a 30% rush delivery fee,
    the cost per type of sliced meat is $13.00. -/
theorem sliced_meat_cost (pack_cost : ℝ) (num_types : ℕ) (rush_fee_percent : ℝ) :
  pack_cost = 40 →
  num_types = 4 →
  rush_fee_percent = 0.3 →
  (pack_cost + pack_cost * rush_fee_percent) / num_types = 13 := by
  sorry

end NUMINAMATH_CALUDE_sliced_meat_cost_l1708_170883


namespace NUMINAMATH_CALUDE_unique_arrangements_zoo_animals_l1708_170874

def num_elephants : ℕ := 4
def num_rabbits : ℕ := 3
def num_parrots : ℕ := 5

def total_animals : ℕ := num_elephants + num_rabbits + num_parrots

theorem unique_arrangements_zoo_animals :
  (Nat.factorial 3) * (Nat.factorial num_elephants) * (Nat.factorial num_rabbits) * (Nat.factorial num_parrots) = 103680 :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangements_zoo_animals_l1708_170874


namespace NUMINAMATH_CALUDE_f_value_at_2_l1708_170829

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

-- State the theorem
theorem f_value_at_2 (a b c : ℝ) : f a b c (-2) = 10 → f a b c 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1708_170829


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_subset_condition_l1708_170875

-- Define set A
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

-- Define set B as a function of m
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

-- Statement 1
theorem intersection_when_m_is_one : 
  A ∩ B 1 = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Statement 2
theorem subset_condition : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_subset_condition_l1708_170875


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1708_170846

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -(122 / 121) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1708_170846


namespace NUMINAMATH_CALUDE_binomial_18_6_l1708_170895

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l1708_170895


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1708_170815

-- Define the angle in degrees
def angle : ℝ := 330

-- State the theorem
theorem sin_330_degrees : Real.sin (angle * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1708_170815


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1708_170823

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 16 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 16 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1708_170823


namespace NUMINAMATH_CALUDE_urn_probability_l1708_170886

/-- Represents the contents of the urn -/
structure UrnContents :=
  (red : ℕ)
  (blue : ℕ)

/-- The operation of drawing a ball and adding another of the same color -/
def draw_and_add (contents : UrnContents) : UrnContents → ℕ → ℝ
  | contents, n => sorry

/-- The probability of having a specific urn content after n operations -/
def prob_after_operations (initial : UrnContents) (final : UrnContents) (n : ℕ) : ℝ :=
  sorry

/-- The probability of removing a specific color ball given the urn contents -/
def prob_remove_color (contents : UrnContents) (remove_red : Bool) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability :
  let initial := UrnContents.mk 2 1
  let final := UrnContents.mk 4 4
  let operations := 6
  (prob_after_operations initial (UrnContents.mk 5 4) operations *
   prob_remove_color (UrnContents.mk 5 4) true) = 5/63 :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_l1708_170886


namespace NUMINAMATH_CALUDE_initial_distance_between_students_l1708_170855

theorem initial_distance_between_students
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ)
  (h1 : speed1 = 1.6)
  (h2 : speed2 = 1.9)
  (h3 : time = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : time > 0) :
  speed1 * time + speed2 * time = 350 := by
sorry

end NUMINAMATH_CALUDE_initial_distance_between_students_l1708_170855


namespace NUMINAMATH_CALUDE_distance_from_origin_l1708_170814

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 5)^2 + (y - 8)^2 = 13^2 →
  x > 5 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (370 + 20 * Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1708_170814


namespace NUMINAMATH_CALUDE_drawing_set_prices_and_quantity_l1708_170807

/-- Represents the cost and selling prices of drawing tool sets from two brands -/
structure DrawingSetPrices where
  costA : ℝ
  costB : ℝ
  sellA : ℝ
  sellB : ℝ

/-- Theorem stating the properties of the drawing set prices and minimum purchase quantity -/
theorem drawing_set_prices_and_quantity (p : DrawingSetPrices)
  (h1 : p.costA = p.costB + 2.5)
  (h2 : 200 / p.costA = 2 * (75 / p.costB))
  (h3 : p.sellA = 13)
  (h4 : p.sellB = 9.5) :
  p.costA = 10 ∧ p.costB = 7.5 ∧
  (∀ a : ℕ, (p.sellA - p.costA) * a + (p.sellB - p.costB) * (2 * a + 4) > 120 → a ≥ 17) :=
sorry

end NUMINAMATH_CALUDE_drawing_set_prices_and_quantity_l1708_170807


namespace NUMINAMATH_CALUDE_farm_tree_sub_branches_l1708_170812

/-- Proves that the number of sub-branches per branch is 40, given the conditions from the farm tree problem -/
theorem farm_tree_sub_branches :
  let branches_per_tree : ℕ := 10
  let leaves_per_sub_branch : ℕ := 60
  let total_trees : ℕ := 4
  let total_leaves : ℕ := 96000
  ∃ (sub_branches_per_branch : ℕ),
    sub_branches_per_branch = 40 ∧
    total_leaves = total_trees * branches_per_tree * leaves_per_sub_branch * sub_branches_per_branch :=
by sorry

end NUMINAMATH_CALUDE_farm_tree_sub_branches_l1708_170812


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l1708_170802

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l1708_170802


namespace NUMINAMATH_CALUDE_system_real_solutions_l1708_170827

/-- The system of equations has real solutions if and only if p ≤ 0, q ≥ 0, and p^2 - 4q ≥ 0 -/
theorem system_real_solutions (p q : ℝ) :
  (∃ (x y z : ℝ), (Real.sqrt x + Real.sqrt y = z) ∧
                   (2 * x + 2 * y + p = 0) ∧
                   (z^4 + p * z^2 + q = 0)) ↔
  (p ≤ 0 ∧ q ≥ 0 ∧ p^2 - 4*q ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_system_real_solutions_l1708_170827


namespace NUMINAMATH_CALUDE_angle_U_measure_l1708_170817

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Measure of angle F
  I : ℝ  -- Measure of angle I
  U : ℝ  -- Measure of angle U
  G : ℝ  -- Measure of angle G
  E : ℝ  -- Measure of angle E
  R : ℝ  -- Measure of angle R

/-- The theorem stating the property of angle U in the given hexagon -/
theorem angle_U_measure (FIGURE : Hexagon) 
  (h1 : FIGURE.F = FIGURE.I ∧ FIGURE.I = FIGURE.U)  -- ∠F ≅ ∠I ≅ ∠U
  (h2 : FIGURE.G + FIGURE.E = 180)  -- ∠G is supplementary to ∠E
  (h3 : FIGURE.R = 2 * FIGURE.U)  -- ∠R = 2∠U
  : FIGURE.U = 108 := by
  sorry

#check angle_U_measure

end NUMINAMATH_CALUDE_angle_U_measure_l1708_170817


namespace NUMINAMATH_CALUDE_pie_slices_today_l1708_170857

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem pie_slices_today : total_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_today_l1708_170857


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1708_170899

theorem consecutive_integers_average (x y : ℝ) : 
  (∃ (a b : ℝ), a = x + 2 ∧ b = x + 4 ∧ y = (x + a + b) / 3) →
  (x + 3 + (x + 4) + (x + 5) + (x + 6)) / 4 = x + 4.5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1708_170899


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l1708_170848

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 8) ∧
  (x₂^2 + 4*x₂ + 3 = 8) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l1708_170848


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1708_170890

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 2 ∧ b = 4 ∧ c = 4 →  -- Two sides are 4, one side is 2
  a + b + c = 10 :=        -- The perimeter is 10
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1708_170890


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_l1708_170811

theorem r_fourth_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_l1708_170811


namespace NUMINAMATH_CALUDE_count_triples_product_million_l1708_170878

theorem count_triples_product_million : 
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => triple.1 * triple.2.1 * triple.2.2 = 10^6) (Finset.product (Finset.range (10^6 + 1)) (Finset.product (Finset.range (10^6 + 1)) (Finset.range (10^6 + 1))))).card = 784 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_product_million_l1708_170878


namespace NUMINAMATH_CALUDE_transformation_matrix_correct_l1708_170873

/-- The transformation matrix M -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

/-- Rotation matrix for 90 degrees counterclockwise -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- Scaling matrix with factor 2 -/
def S : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

theorem transformation_matrix_correct :
  M = S * R :=
sorry

end NUMINAMATH_CALUDE_transformation_matrix_correct_l1708_170873


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l1708_170893

theorem sum_of_fractions_geq_three (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + a^2) / (1 + a*b) + (1 + b^2) / (1 + b*c) + (1 + c^2) / (1 + c*a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_three_l1708_170893


namespace NUMINAMATH_CALUDE_equation_solution_l1708_170833

theorem equation_solution :
  ∃ x : ℝ, 38 + 2 * x^3 = 1250 ∧ x = (606 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1708_170833


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l1708_170862

theorem reciprocal_of_two : (2⁻¹ : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l1708_170862


namespace NUMINAMATH_CALUDE_solve_equation_l1708_170851

theorem solve_equation (y z : ℝ) (h1 : y = -2.6) (h2 : z = 4.3) :
  ∃ x : ℝ, 5 * x - 2 * y + 3.7 * z = 1.45 ∧ x = -3.932 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1708_170851


namespace NUMINAMATH_CALUDE_digit_sum_l1708_170806

theorem digit_sum (a b : ℕ) : 
  a < 10 → b < 10 → (32 * a + 300) * (10 * b + 4) = 1486 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l1708_170806


namespace NUMINAMATH_CALUDE_sale_price_calculation_l1708_170858

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price with tax is approximately 677.61 -/
theorem sale_price_calculation :
  let costPrice := 526.50
  let profitRate := 0.17
  let taxRate := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.61) < 0.01 := by
  sorry

#eval salePriceWithTax 526.50 0.17 0.10

end NUMINAMATH_CALUDE_sale_price_calculation_l1708_170858


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1708_170803

theorem solve_exponential_equation :
  ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ∧ x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1708_170803


namespace NUMINAMATH_CALUDE_minutes_to_year_l1708_170887

/-- Proves that 525,600 minutes is equivalent to 365 days (1 year) --/
theorem minutes_to_year (minutes_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : 
  minutes_per_hour = 60 → hours_per_day = 24 → days_per_year = 365 →
  525600 / (minutes_per_hour * hours_per_day) = days_per_year := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_year_l1708_170887


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_product_l1708_170865

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2012 * 7 is 13 -/
theorem sum_of_digits_of_power_product : ∃ n : ℕ, 
  (n = 2^2010 * 5^2012 * 7) ∧ 
  (List.sum (Nat.digits 10 n) = 13) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_product_l1708_170865


namespace NUMINAMATH_CALUDE_smallest_n_for_pencil_paradox_l1708_170808

theorem smallest_n_for_pencil_paradox : ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c d : ℕ, 
      6 * a + 10 * b = m ∧ 
      6 * c + 10 * d = m + 2 ∧ 
      7 * a + 12 * b > 7 * c + 12 * d)) ∧
  (∃ a b c d : ℕ, 
    6 * a + 10 * b = n ∧ 
    6 * c + 10 * d = n + 2 ∧ 
    7 * a + 12 * b > 7 * c + 12 * d) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_pencil_paradox_l1708_170808


namespace NUMINAMATH_CALUDE_imoProblem1995_l1708_170838

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Checks if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

theorem imoProblem1995 (n : ℕ) (h_n : n > 3) :
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ (i j k : Fin n), i < j → j < k → ¬areCollinear (A i) (A j) (A k)) ∧
    (∀ (i j k : Fin n), i < j → j < k → 
      triangleArea (A i) (A j) (A k) = r i + r j + r k)) ↔ 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_imoProblem1995_l1708_170838


namespace NUMINAMATH_CALUDE_integral_equals_ln_80_over_23_l1708_170800

open Real MeasureTheory

theorem integral_equals_ln_80_over_23 :
  ∫ x in (1 : ℝ)..2, (9 * x + 4) / (x^5 + 3 * x^2 + x) = Real.log (80 / 23) := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ln_80_over_23_l1708_170800


namespace NUMINAMATH_CALUDE_fruit_display_problem_l1708_170870

theorem fruit_display_problem (bananas oranges apples : ℕ) 
  (h1 : apples = 2 * oranges)
  (h2 : oranges = 2 * bananas)
  (h3 : bananas + oranges + apples = 35) :
  bananas = 5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_display_problem_l1708_170870


namespace NUMINAMATH_CALUDE_office_printing_calculation_l1708_170818

/-- Calculate the number of one-page documents printed per day -/
def documents_per_day (packs : ℕ) (sheets_per_pack : ℕ) (days : ℕ) : ℕ :=
  (packs * sheets_per_pack) / days

theorem office_printing_calculation :
  documents_per_day 2 240 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_office_printing_calculation_l1708_170818


namespace NUMINAMATH_CALUDE_log_identity_l1708_170894

theorem log_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hcb : c > b) 
  (h_pythagorean : a^2 + b^2 = c^2) : 
  Real.log a / Real.log (c + b) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
sorry

end NUMINAMATH_CALUDE_log_identity_l1708_170894


namespace NUMINAMATH_CALUDE_salary_comparison_l1708_170835

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  total_students : ℕ
  graduating_students : ℕ
  dropout_salary : ℝ
  high_salary : ℝ
  mid_salary : ℝ
  low_salary : ℝ
  default_salary : ℝ
  high_salary_ratio : ℝ
  mid_salary_ratio : ℝ
  low_salary_ratio : ℝ

/-- Represents Fyodor's salary growth --/
structure SalaryGrowth where
  initial_salary : ℝ
  yearly_increase : ℝ
  years : ℕ

/-- Calculates the expected salary based on the given distribution --/
def expected_salary (d : SalaryDistribution) : ℝ :=
  let graduate_prob := d.graduating_students / d.total_students
  let default_salary_ratio := 1 - d.high_salary_ratio - d.mid_salary_ratio - d.low_salary_ratio
  graduate_prob * (d.high_salary_ratio * d.high_salary + 
                   d.mid_salary_ratio * d.mid_salary + 
                   d.low_salary_ratio * d.low_salary + 
                   default_salary_ratio * d.default_salary) +
  (1 - graduate_prob) * d.dropout_salary

/-- Calculates Fyodor's salary after a given number of years --/
def fyodor_salary (g : SalaryGrowth) : ℝ :=
  g.initial_salary + g.yearly_increase * g.years

/-- The main theorem to prove --/
theorem salary_comparison 
  (d : SalaryDistribution)
  (g : SalaryGrowth)
  (h1 : d.total_students = 300)
  (h2 : d.graduating_students = 270)
  (h3 : d.dropout_salary = 25000)
  (h4 : d.high_salary = 60000)
  (h5 : d.mid_salary = 80000)
  (h6 : d.low_salary = 25000)
  (h7 : d.default_salary = 40000)
  (h8 : d.high_salary_ratio = 1/5)
  (h9 : d.mid_salary_ratio = 1/10)
  (h10 : d.low_salary_ratio = 1/20)
  (h11 : g.initial_salary = 25000)
  (h12 : g.yearly_increase = 3000)
  (h13 : g.years = 4)
  : expected_salary d = 39625 ∧ expected_salary d - fyodor_salary g = 2625 := by
  sorry


end NUMINAMATH_CALUDE_salary_comparison_l1708_170835


namespace NUMINAMATH_CALUDE_zero_exponent_is_one_l1708_170822

theorem zero_exponent_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_is_one_l1708_170822


namespace NUMINAMATH_CALUDE_egg_collection_ratio_l1708_170845

/-- 
Given:
- Benjamin collects 6 dozen eggs
- Trisha collects 4 dozen less than Benjamin
- The total eggs collected by all three is 26 dozen

Prove that the ratio of Carla's eggs to Benjamin's eggs is 3:1
-/
theorem egg_collection_ratio : 
  let benjamin_eggs : ℕ := 6
  let trisha_eggs : ℕ := benjamin_eggs - 4
  let total_eggs : ℕ := 26
  let carla_eggs : ℕ := total_eggs - benjamin_eggs - trisha_eggs
  (carla_eggs : ℚ) / benjamin_eggs = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_egg_collection_ratio_l1708_170845


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1708_170832

theorem triangle_angle_proof (a b c : ℝ) (S : ℝ) (C : ℝ) :
  a > 0 → b > 0 → c > 0 → S > 0 →
  0 < C → C < π →
  S = (1/2) * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 4 * Real.sqrt 3 * S →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1708_170832


namespace NUMINAMATH_CALUDE_histogram_approximates_density_curve_l1708_170888

/-- Represents a sample frequency distribution histogram --/
structure SampleHistogram where
  sampleSize : ℕ
  groupInterval : ℝ
  distribution : ℝ → ℝ

/-- Represents a population density curve --/
def PopulationDensityCurve := ℝ → ℝ

/-- Measures the difference between a histogram and a density curve --/
def difference (h : SampleHistogram) (p : PopulationDensityCurve) : ℝ := sorry

theorem histogram_approximates_density_curve
  (h : ℕ → SampleHistogram)
  (p : PopulationDensityCurve)
  (hsize : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).sampleSize > 1 / ε)
  (hinterval : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).groupInterval < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, difference (h n) p < ε :=
sorry

end NUMINAMATH_CALUDE_histogram_approximates_density_curve_l1708_170888


namespace NUMINAMATH_CALUDE_function_composition_value_l1708_170810

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := 4 * x - 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_value (a b : ℝ) :
  (∀ x : ℝ, h a b x = (x - 14) / 2) → a - 2 * b = 101 / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_value_l1708_170810


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l1708_170801

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l1708_170801


namespace NUMINAMATH_CALUDE_oil_percentage_in_dressing_q_l1708_170869

/-- Represents the composition of a salad dressing -/
structure Dressing where
  vinegar : ℝ
  oil : ℝ

/-- Represents the mixture of two dressings -/
structure Mixture where
  dressing_p : Dressing
  dressing_q : Dressing
  p_ratio : ℝ
  q_ratio : ℝ
  vinegar : ℝ

/-- Theorem stating that given the conditions of the problem, 
    the oil percentage in dressing Q is 90% -/
theorem oil_percentage_in_dressing_q 
  (p : Dressing)
  (q : Dressing)
  (mix : Mixture)
  (h1 : p.vinegar = 0.3)
  (h2 : p.oil = 0.7)
  (h3 : q.vinegar = 0.1)
  (h4 : mix.dressing_p = p)
  (h5 : mix.dressing_q = q)
  (h6 : mix.p_ratio = 0.1)
  (h7 : mix.q_ratio = 0.9)
  (h8 : mix.vinegar = 0.12)
  : q.oil = 0.9 := by
  sorry

#check oil_percentage_in_dressing_q

end NUMINAMATH_CALUDE_oil_percentage_in_dressing_q_l1708_170869


namespace NUMINAMATH_CALUDE_dot_product_AB_AC_t_value_l1708_170876

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1: Dot product of AB and AC
theorem dot_product_AB_AC : dot_product AB AC = 2 := by sorry

-- Theorem 2: Value of t
theorem t_value : ∃ t : ℝ, t = -3 ∧ dot_product (AB.1 - t * OC.1, AB.2 - t * OC.2) OB = 0 := by sorry

end NUMINAMATH_CALUDE_dot_product_AB_AC_t_value_l1708_170876


namespace NUMINAMATH_CALUDE_clarence_oranges_left_l1708_170884

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def total_oranges : ℕ := initial_oranges + received_oranges

theorem clarence_oranges_left : (total_oranges / 2 : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_left_l1708_170884


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1708_170879

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1708_170879


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1708_170867

theorem fixed_point_parabola (s : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1708_170867


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_ferry_tourists_sum_proof_l1708_170863

theorem ferry_tourists_sum : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun n a d s =>
    n = 10 ∧ a = 120 ∧ d = 2 →
    s = n * (2 * a - (n - 1) * d) / 2 →
    s = 1110

-- The proof is omitted
theorem ferry_tourists_sum_proof : ferry_tourists_sum 10 120 2 1110 := by sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_ferry_tourists_sum_proof_l1708_170863


namespace NUMINAMATH_CALUDE_valid_parameterization_l1708_170840

/-- A structure representing a vector parameterization of a line -/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- Predicate to check if a vector parameterization represents the line y = 2x - 7 -/
def IsValidParam (p : VectorParam) : Prop :=
  p.y₀ = 2 * p.x₀ - 7 ∧ ∃ (k : ℝ), p.dx = k * 1 ∧ p.dy = k * 2

/-- Theorem stating the conditions for a valid parameterization of y = 2x - 7 -/
theorem valid_parameterization (p : VectorParam) :
  IsValidParam p ↔ 
  ∀ (t : ℝ), (p.y₀ + t * p.dy) = 2 * (p.x₀ + t * p.dx) - 7 :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1708_170840


namespace NUMINAMATH_CALUDE_goldfish_count_l1708_170882

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := 44

/-- Theorem stating that the number of goldfish is 8 -/
theorem goldfish_count : num_goldfish = 8 ∧ 
  num_angelfish = num_goldfish + 4 ∧ 
  num_guppies = 2 * num_angelfish ∧ 
  total_fish = num_goldfish + num_angelfish + num_guppies :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_l1708_170882


namespace NUMINAMATH_CALUDE_odd_function_properties_l1708_170828

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem odd_function_properties (a b : ℝ) 
  (h_odd : ∀ x, f a b x = -f a b (-x)) :
  (a = 1 ∧ b = 1) ∧
  (∀ t : ℝ, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1708_170828


namespace NUMINAMATH_CALUDE_bobs_money_l1708_170834

theorem bobs_money (X : ℝ) :
  X > 0 →
  let day1_remainder := X / 2
  let day2_remainder := day1_remainder * 4 / 5
  let day3_remainder := day2_remainder * 5 / 8
  day3_remainder = 20 →
  X = 80 :=
by sorry

end NUMINAMATH_CALUDE_bobs_money_l1708_170834


namespace NUMINAMATH_CALUDE_quadrilateral_angles_not_always_form_triangle_l1708_170809

theorem quadrilateral_angles_not_always_form_triangle : ∃ (α β γ δ : ℝ),
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧
  α + β + γ + δ = 360 ∧
  ¬(α + β > γ ∧ β + γ > α ∧ γ + α > β) ∧
  ¬(α + β > δ ∧ β + δ > α ∧ δ + α > β) ∧
  ¬(α + γ > δ ∧ γ + δ > α ∧ δ + α > γ) ∧
  ¬(β + γ > δ ∧ γ + δ > β ∧ δ + β > γ) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angles_not_always_form_triangle_l1708_170809


namespace NUMINAMATH_CALUDE_largest_base5_three_digit_to_base10_l1708_170854

-- Define a function to convert a base-5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the largest three-digit number in base 5
def largestBase5ThreeDigit : List Nat := [4, 4, 4]

-- Theorem statement
theorem largest_base5_three_digit_to_base10 :
  base5ToBase10 largestBase5ThreeDigit = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_three_digit_to_base10_l1708_170854


namespace NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l1708_170871

/-- Represents the seating arrangement around a round table -/
structure SeatingArrangement where
  knights : ℕ
  samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights between knights for the given arrangement -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.knights = 40)
  (h2 : arrangement.samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights ⟨40, 10, 7⟩

end NUMINAMATH_CALUDE_max_knights_between_knights_theorem_l1708_170871


namespace NUMINAMATH_CALUDE_equation_equivalence_l1708_170837

theorem equation_equivalence (x : ℝ) : 
  (1 - (x + 3) / 6 = x / 2) ↔ (6 - x - 3 = 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1708_170837


namespace NUMINAMATH_CALUDE_soda_price_calculation_l1708_170830

/-- Calculates the price of soda cans with applicable discounts -/
theorem soda_price_calculation (regular_price : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (cans : ℕ) : 
  regular_price = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  cans = 75 →
  let discounted_price := regular_price * (1 - case_discount)
  let bulk_discounted_price := discounted_price * (1 - bulk_discount)
  let total_price := bulk_discounted_price * cans
  total_price = 9.405 := by
  sorry

#check soda_price_calculation

end NUMINAMATH_CALUDE_soda_price_calculation_l1708_170830


namespace NUMINAMATH_CALUDE_circle_tangency_l1708_170850

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (t : ℝ) : 
  externally_tangent (0, 0) (t, 0) 2 1 → t = 3 ∨ t = -3 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l1708_170850


namespace NUMINAMATH_CALUDE_problem_statement_l1708_170826

theorem problem_statement (m n : ℤ) (h : m * n = m + 3) : 
  3 * m - 3 * (m * n) + 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1708_170826


namespace NUMINAMATH_CALUDE_tour_group_dish_choices_l1708_170866

/-- Represents the number of people in the tour group -/
def total_people : ℕ := 92

/-- Represents the number of different dish combinations -/
def dish_combinations : ℕ := 9

/-- Represents the minimum number of people who must choose the same combination -/
def min_same_choice : ℕ := total_people / dish_combinations + 1

theorem tour_group_dish_choices :
  ∃ (combination : Fin dish_combinations),
    (Finset.filter (λ person : Fin total_people =>
      person.val % dish_combinations = combination.val) (Finset.univ : Finset (Fin total_people))).card
    ≥ min_same_choice :=
sorry

end NUMINAMATH_CALUDE_tour_group_dish_choices_l1708_170866


namespace NUMINAMATH_CALUDE_michaels_weight_loss_goal_l1708_170843

/-- The total weight Michael wants to lose by June -/
def total_weight_loss (march_loss april_loss may_loss : ℕ) : ℕ :=
  march_loss + april_loss + may_loss

/-- Proof that Michael's total weight loss goal is 10 pounds -/
theorem michaels_weight_loss_goal :
  ∃ (march_loss april_loss may_loss : ℕ),
    march_loss = 3 ∧
    april_loss = 4 ∧
    may_loss = 3 ∧
    total_weight_loss march_loss april_loss may_loss = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_weight_loss_goal_l1708_170843


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1708_170889

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

-- Theorem to prove
theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1708_170889


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l1708_170853

theorem unique_triplet_solution :
  ∀ x y z : ℕ,
    (1 + x / (y + z : ℚ))^2 + (1 + y / (z + x : ℚ))^2 + (1 + z / (x + y : ℚ))^2 = 27/4
    ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l1708_170853


namespace NUMINAMATH_CALUDE_balls_after_1500_steps_l1708_170804

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateBallPlacement (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a BoxState -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1500 steps
    is equal to the sum of digits of 1500 in base-4 -/
theorem balls_after_1500_steps :
  countBalls (simulateBallPlacement 1500) = sumDigits (toBase4 1500) :=
sorry

end NUMINAMATH_CALUDE_balls_after_1500_steps_l1708_170804


namespace NUMINAMATH_CALUDE_golf_course_distance_l1708_170813

/-- Represents a golf shot with distance and wind conditions -/
structure GolfShot where
  distance : ℝ
  windSpeed : ℝ
  windDirection : String

/-- Calculates the total distance to the hole given three golf shots -/
def distanceToHole (shot1 shot2 shot3 : GolfShot) (slopeEffect : ℝ) : ℝ :=
  shot1.distance + (shot2.distance - slopeEffect)

theorem golf_course_distance :
  let shot1 : GolfShot := { distance := 180, windSpeed := 10, windDirection := "tailwind" }
  let shot2 : GolfShot := { distance := 90, windSpeed := 7, windDirection := "crosswind" }
  let shot3 : GolfShot := { distance := 0, windSpeed := 5, windDirection := "headwind" }
  let slopeEffect : ℝ := 20
  distanceToHole shot1 shot2 shot3 slopeEffect = 270 := by
  sorry

end NUMINAMATH_CALUDE_golf_course_distance_l1708_170813


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_ending_in_seven_l1708_170825

theorem sum_of_arithmetic_sequence_ending_in_seven : 
  ∀ (a : ℕ) (d : ℕ) (n : ℕ),
    a = 107 → d = 10 → n = 40 →
    (a + (n - 1) * d = 497) →
    (n * (a + (a + (n - 1) * d))) / 2 = 12080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_ending_in_seven_l1708_170825


namespace NUMINAMATH_CALUDE_grover_profit_l1708_170841

def number_of_boxes : ℕ := 3
def masks_per_box : ℕ := 20
def total_cost : ℚ := 15
def selling_price_per_mask : ℚ := 1/2

def total_masks : ℕ := number_of_boxes * masks_per_box
def total_revenue : ℚ := (total_masks : ℚ) * selling_price_per_mask
def profit : ℚ := total_revenue - total_cost

theorem grover_profit : profit = 15 := by
  sorry

end NUMINAMATH_CALUDE_grover_profit_l1708_170841


namespace NUMINAMATH_CALUDE_greater_than_reciprocal_reciprocal_comparison_l1708_170897

theorem greater_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x > 1 / x

theorem reciprocal_comparison : 
  ¬ greater_than_reciprocal (-3/2) ∧
  ¬ greater_than_reciprocal (-1) ∧
  ¬ greater_than_reciprocal (1/3) ∧
  greater_than_reciprocal 2 ∧
  greater_than_reciprocal 3 := by
sorry

end NUMINAMATH_CALUDE_greater_than_reciprocal_reciprocal_comparison_l1708_170897


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l1708_170847

/-- The area of wrapping paper required to wrap a rectangular box -/
def wrapping_paper_area (w : ℝ) (h : ℝ) : ℝ :=
  4 * (w + h)^2

/-- Theorem: The area of a square sheet of wrapping paper required to wrap a rectangular box
    with dimensions 2w × w × h, such that the corners of the paper meet at the center of the
    top of the box, is equal to 4(w + h)^2. -/
theorem wrapping_paper_area_theorem (w : ℝ) (h : ℝ) 
    (hw : w > 0) (hh : h > 0) : 
    wrapping_paper_area w h = 4 * (w + h)^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l1708_170847


namespace NUMINAMATH_CALUDE_coprime_and_indivisible_l1708_170868

theorem coprime_and_indivisible (n : ℕ) (h1 : n > 3) (h2 : Odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ Nat.gcd (a * b * (a + b)) n = 1 ∧ ¬(n ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_and_indivisible_l1708_170868


namespace NUMINAMATH_CALUDE_order_of_differences_l1708_170864

theorem order_of_differences (a b c : ℝ) : 
  a = Real.sqrt 3 - Real.sqrt 2 →
  b = Real.sqrt 6 - Real.sqrt 5 →
  c = Real.sqrt 7 - Real.sqrt 6 →
  a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_order_of_differences_l1708_170864


namespace NUMINAMATH_CALUDE_calculation_proof_l1708_170842

theorem calculation_proof : 2456 + 144 / 12 * 5 - 256 = 2260 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1708_170842


namespace NUMINAMATH_CALUDE_quadratic_range_l1708_170892

def f (x : ℝ) := x^2 - 4*x + 1

theorem quadratic_range : 
  ∀ y ∈ Set.Icc (-2 : ℝ) 6, ∃ x ∈ Set.Icc 3 5, f x = y ∧
  ∀ x ∈ Set.Icc 3 5, f x ∈ Set.Icc (-2 : ℝ) 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_l1708_170892


namespace NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l1708_170856

theorem night_day_crew_loading_ratio :
  ∀ (D N B : ℚ),
  N = (2/3) * D →                     -- Night crew has 2/3 as many workers as day crew
  (2/3) * B = D * (B / D) →           -- Day crew loaded 2/3 of all boxes
  (1/3) * B = N * (B / N) →           -- Night crew loaded 1/3 of all boxes
  (B / N) / (B / D) = 3/4              -- Ratio of boxes loaded by each night worker to each day worker
:= by sorry

end NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l1708_170856


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l1708_170824

/-- Two concentric circles with radii 1 and 3 -/
def inner_radius : ℝ := 1
def outer_radius : ℝ := 3

/-- Radius of circles tangent to both concentric circles -/
def tangent_circle_radius : ℝ := 1

/-- Maximum number of non-overlapping tangent circles -/
def max_tangent_circles : ℕ := 6

/-- Theorem stating the radius of tangent circles and the maximum number of such circles -/
theorem tangent_circles_theorem :
  (tangent_circle_radius = 1) ∧
  (max_tangent_circles = 6) := by
  sorry

#check tangent_circles_theorem

end NUMINAMATH_CALUDE_tangent_circles_theorem_l1708_170824


namespace NUMINAMATH_CALUDE_gecko_eggs_hatched_l1708_170860

/-- The number of eggs that actually hatch from a gecko's yearly egg-laying, given the total number of eggs, infertility rate, and calcification issue rate. -/
theorem gecko_eggs_hatched (total_eggs : ℕ) (infertility_rate : ℚ) (calcification_rate : ℚ) : 
  total_eggs = 30 →
  infertility_rate = 1/5 →
  calcification_rate = 1/3 →
  (total_eggs : ℚ) * (1 - infertility_rate) * (1 - calcification_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eggs_hatched_l1708_170860


namespace NUMINAMATH_CALUDE_circle_area_difference_l1708_170836

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 12
  let r2 : ℝ := d2 / 2
  π * r1^2 - π * r2^2 = 864 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1708_170836


namespace NUMINAMATH_CALUDE_f_13_equals_223_l1708_170852

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_223_l1708_170852


namespace NUMINAMATH_CALUDE_worksheet_problems_l1708_170881

theorem worksheet_problems (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  total_worksheets = 17 →
  graded_worksheets = 8 →
  problems_left = 63 →
  (total_worksheets - graded_worksheets) * (problems_left / (total_worksheets - graded_worksheets)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_worksheet_problems_l1708_170881


namespace NUMINAMATH_CALUDE_function_symmetry_l1708_170816

/-- Given a function f(x) = a*sin(x) - b*cos(x) where f(x) takes an extreme value when x = π/4,
    prove that y = f(3π/4 - x) is an odd function and its graph is symmetric about (π, 0) -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x - b * Real.cos x
  (∃ (extreme : ℝ), f (π/4) = extreme ∧ ∀ x, f x ≤ extreme) →
  let y : ℝ → ℝ := λ x ↦ f (3*π/4 - x)
  (∀ x, y (-x) = -y x) ∧  -- odd function
  (∀ x, y (2*π - x) = -y x)  -- symmetry about (π, 0)
:= by sorry

end NUMINAMATH_CALUDE_function_symmetry_l1708_170816


namespace NUMINAMATH_CALUDE_index_card_area_l1708_170898

theorem index_card_area (width height : ℝ) (h1 : width = 5) (h2 : height = 8) : 
  ((width - 2) * height = 24 → width * (height - 2) = 30) ∧ 
  ((width * (height - 2) = 24 → (width - 2) * height = 30)) :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l1708_170898


namespace NUMINAMATH_CALUDE_product_five_reciprocal_squares_sum_l1708_170896

theorem product_five_reciprocal_squares_sum (a b : ℕ) (h : a * b = 5) :
  (1 : ℝ) / (a^2 : ℝ) + (1 : ℝ) / (b^2 : ℝ) = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_product_five_reciprocal_squares_sum_l1708_170896


namespace NUMINAMATH_CALUDE_paint_usage_l1708_170861

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  second_week_fraction = 1 / 5 →
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 104 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_l1708_170861


namespace NUMINAMATH_CALUDE_product_bcd_value_l1708_170877

theorem product_bcd_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : c * d * e = 500)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 1) :
  b * c * d = 65 := by
  sorry

end NUMINAMATH_CALUDE_product_bcd_value_l1708_170877


namespace NUMINAMATH_CALUDE_quiz_average_change_l1708_170844

theorem quiz_average_change (total_students : ℕ) (dropped_score : ℝ) (new_average : ℝ) :
  total_students = 16 →
  dropped_score = 55 →
  new_average = 63 →
  (((total_students : ℝ) * new_average + dropped_score) / (total_students : ℝ)) = 62.5 :=
by sorry

end NUMINAMATH_CALUDE_quiz_average_change_l1708_170844


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1708_170885

theorem smallest_four_digit_multiple_of_18 :
  ∃ n : ℕ, n = 1008 ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧
  n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1708_170885


namespace NUMINAMATH_CALUDE_sheets_exceed_500_at_step_31_l1708_170805

def sheets_after_steps (initial_sheets : ℕ) (steps : ℕ) : ℕ :=
  initial_sheets + steps * (steps + 1) / 2

theorem sheets_exceed_500_at_step_31 :
  sheets_after_steps 10 31 > 500 ∧ sheets_after_steps 10 30 ≤ 500 := by
  sorry

end NUMINAMATH_CALUDE_sheets_exceed_500_at_step_31_l1708_170805


namespace NUMINAMATH_CALUDE_triangle_area_rational_l1708_170820

/-- A point on the unit circle with rational coordinates -/
structure RationalUnitCirclePoint where
  x : ℚ
  y : ℚ
  on_circle : x^2 + y^2 = 1

/-- The area of a triangle with vertices on the unit circle is rational -/
theorem triangle_area_rational (p₁ p₂ p₃ : RationalUnitCirclePoint) :
  ∃ a : ℚ, a = (1/2) * |p₁.x * (p₂.y - p₃.y) + p₂.x * (p₃.y - p₁.y) + p₃.x * (p₁.y - p₂.y)| :=
sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l1708_170820


namespace NUMINAMATH_CALUDE_expression_simplification_l1708_170849

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  2 * (a^2 * b + a * b^2) - 3 * (a^2 * b + 1) - 2 * a * b^2 - 2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1708_170849


namespace NUMINAMATH_CALUDE_clinton_belts_l1708_170819

/-- Proves that Clinton has 7 belts given the conditions -/
theorem clinton_belts :
  ∀ (shoes belts : ℕ),
  shoes = 14 →
  shoes = 2 * belts →
  belts = 7 := by
  sorry

end NUMINAMATH_CALUDE_clinton_belts_l1708_170819


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_is_two_l1708_170821

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0  -- Arithmetic property
  h_sum : ∀ n, S n = n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)  -- Sum formula

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem arithmetic_seq_common_diff_is_two (seq : ArithmeticSequence) 
    (h : seq.S 2020 / 2020 - seq.S 20 / 20 = 2000) : 
    seq.a 1 - seq.a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_is_two_l1708_170821
