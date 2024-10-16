import Mathlib

namespace NUMINAMATH_CALUDE_x_value_theorem_l1179_117929

theorem x_value_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧ 
    (∀ r : ℕ, Prime r → r ∣ x ↔ r = 3 ∨ r = p ∨ r = q)) →
  x = 480 ∨ x = 2016 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1179_117929


namespace NUMINAMATH_CALUDE_circle_equation_l1179_117920

/-- Given a real number a, prove that the equation a²x² + (a+2)y² + 4x + 8y + 5a = 0
    represents a circle with center (-2, -4) and radius 5 if and only if a = -1 -/
theorem circle_equation (a : ℝ) :
  (∃ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0) ∧
  (∀ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0 ↔
    (x + 2)^2 + (y + 4)^2 = 25) ↔
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1179_117920


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l1179_117998

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l1179_117998


namespace NUMINAMATH_CALUDE_vector_addition_l1179_117923

/-- Given two 2D vectors a and b, prove that 2b + 3a equals (6,1) -/
theorem vector_addition (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (0, -1)) :
  2 • b + 3 • a = (6, 1) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l1179_117923


namespace NUMINAMATH_CALUDE_permutation_combination_inequality_l1179_117918

theorem permutation_combination_inequality (n : ℕ+) :
  (n.val.factorial / (n.val - 2).factorial)^2 > 6 * (n.val.choose 4) ↔ n.val ∈ ({2, 3, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_permutation_combination_inequality_l1179_117918


namespace NUMINAMATH_CALUDE_intersection_M_N_l1179_117996

-- Define set M
def M : Set ℕ := {y | y < 6}

-- Define set N
def N : Set ℕ := {2, 3, 6}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1179_117996


namespace NUMINAMATH_CALUDE_line_through_points_l1179_117937

/-- 
A line in a rectangular coordinate system is defined by the equation x = 5y + 5.
This line passes through two points (m, n) and (m + 2, n + p).
The theorem proves that under these conditions, p must equal 2/5.
-/
theorem line_through_points (m n : ℝ) : 
  (m = 5 * n + 5) → 
  (m + 2 = 5 * (n + p) + 5) → 
  p = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1179_117937


namespace NUMINAMATH_CALUDE_curve_C_and_point_Q_existence_l1179_117936

noncomputable section

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the curve C
def curve_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the fixed point (0, 1/2)
def fixed_point : ℝ × ℝ := (0, 1/2)

-- Define the point Q
def Q : ℝ × ℝ := (0, 6)

-- State the theorem
theorem curve_C_and_point_Q_existence :
  ∀ (P : ℝ × ℝ),
  (∃ (center : ℝ × ℝ), (center.1 - F.1)^2 + (center.2 - F.2)^2 = (center.1 - P.1)^2 + (center.2 - P.2)^2 ∧
                       ∃ (T : ℝ × ℝ), T ∈ circle_O ∧ (center.1 - T.1)^2 + (center.2 - T.2)^2 = (F.1 - P.1)^2 / 4 + (F.2 - P.2)^2 / 4) →
  P ∈ curve_C ∧
  ∀ (M N : ℝ × ℝ), M ∈ curve_C → N ∈ curve_C →
    (N.2 - M.2) * fixed_point.1 = (N.1 - M.1) * (fixed_point.2 - M.2) + M.1 * (N.2 - M.2) →
    (M.2 - Q.2) / (M.1 - Q.1) + (N.2 - Q.2) / (N.1 - Q.1) = 0 :=
by sorry

end

end NUMINAMATH_CALUDE_curve_C_and_point_Q_existence_l1179_117936


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1179_117921

/-- A right triangle with hypotenuse 13 and one leg 5 -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : hypotenuse^2 = leg1^2 + leg2^2
  hypotenuse_is_13 : hypotenuse = 13
  leg1_is_5 : leg1 = 5

/-- Properties of the specific right triangle -/
theorem right_triangle_properties (t : RightTriangle) :
  t.leg2 = 12 ∧
  (1/2 : ℝ) * t.leg1 * t.leg2 = 30 ∧
  t.leg1 + t.leg2 + t.hypotenuse = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1179_117921


namespace NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l1179_117905

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_size_scientific_notation :
  toScientificNotation 0.0000012 = ScientificNotation.mk 1.2 (-6) sorry := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l1179_117905


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1179_117909

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1179_117909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1179_117939

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 15 = 48 →
  a 3 + 3 * a 8 + a 13 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1179_117939


namespace NUMINAMATH_CALUDE_line_chart_most_appropriate_for_temperature_over_time_l1179_117976

-- Define the types of charts
inductive ChartType
| PieChart
| LineChart
| BarChart

-- Define the properties of the data
structure DataProperties where
  isTemperature : Bool
  isOverTime : Bool
  needsChangeObservation : Bool

-- Define the function to determine the most appropriate chart type
def mostAppropriateChart (props : DataProperties) : ChartType :=
  if props.isTemperature ∧ props.isOverTime ∧ props.needsChangeObservation then
    ChartType.LineChart
  else
    ChartType.BarChart  -- Default to BarChart for other cases

-- Theorem statement
theorem line_chart_most_appropriate_for_temperature_over_time 
  (props : DataProperties) 
  (h1 : props.isTemperature = true) 
  (h2 : props.isOverTime = true) 
  (h3 : props.needsChangeObservation = true) : 
  mostAppropriateChart props = ChartType.LineChart := by
  sorry


end NUMINAMATH_CALUDE_line_chart_most_appropriate_for_temperature_over_time_l1179_117976


namespace NUMINAMATH_CALUDE_equation_roots_l1179_117950

theorem equation_roots :
  let f (x : ℝ) := x^2 - 2*x - 2/x + 1/x^2 - 13
  ∃ (a b c d : ℝ),
    (a = (5 + Real.sqrt 21) / 2) ∧
    (b = (5 - Real.sqrt 21) / 2) ∧
    (c = (-3 + Real.sqrt 5) / 2) ∧
    (d = (-3 - Real.sqrt 5) / 2) ∧
    (∀ x : ℝ, x ≠ 0 → (f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1179_117950


namespace NUMINAMATH_CALUDE_car_travel_time_l1179_117947

/-- Given a truck and car with specific conditions, prove the car's travel time --/
theorem car_travel_time (truck_distance : ℝ) (truck_time : ℝ) (speed_difference : ℝ) (distance_difference : ℝ) :
  truck_distance = 296 →
  truck_time = 8 →
  speed_difference = 18 →
  distance_difference = 6.5 →
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := truck_distance + distance_difference
  car_distance / car_speed = 5.5 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_l1179_117947


namespace NUMINAMATH_CALUDE_cell_growth_proof_l1179_117928

/-- The time interval between cell divisions in minutes -/
def division_interval : ℕ := 20

/-- The total time elapsed in minutes -/
def total_time : ℕ := 3 * 60 + 20

/-- The number of cells after one division -/
def cells_after_division : ℕ := 2

/-- The number of cells after a given number of divisions -/
def cells_after_divisions (n : ℕ) : ℕ := cells_after_division ^ n

theorem cell_growth_proof :
  cells_after_divisions (total_time / division_interval) = 1024 :=
by sorry

end NUMINAMATH_CALUDE_cell_growth_proof_l1179_117928


namespace NUMINAMATH_CALUDE_vehicle_speeds_theorem_l1179_117932

/-- Represents the speeds of two vehicles traveling in opposite directions -/
structure VehicleSpeeds where
  slow : ℝ
  fast : ℝ
  speed_diff : fast = slow + 8

/-- Proves that given the conditions, the speeds of the vehicles are 44 and 52 mph -/
theorem vehicle_speeds_theorem (v : VehicleSpeeds) 
  (h : 4 * (v.slow + v.fast) = 384) : 
  v.slow = 44 ∧ v.fast = 52 := by
  sorry

#check vehicle_speeds_theorem

end NUMINAMATH_CALUDE_vehicle_speeds_theorem_l1179_117932


namespace NUMINAMATH_CALUDE_machine_comparison_l1179_117946

def machine_A : List ℕ := [0, 2, 1, 0, 3, 0, 2, 1, 2, 4]
def machine_B : List ℕ := [2, 1, 1, 2, 1, 0, 2, 1, 3, 2]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_comparison :
  average machine_A = average machine_B ∧
  variance machine_B < variance machine_A :=
sorry

end NUMINAMATH_CALUDE_machine_comparison_l1179_117946


namespace NUMINAMATH_CALUDE_air_conditioner_sales_theorem_l1179_117911

/-- Represents the shopping mall's air conditioner purchases and sales --/
structure AirConditionerSales where
  first_purchase_total : ℝ
  first_purchase_unit_price : ℝ
  second_purchase_total : ℝ
  second_purchase_unit_price : ℝ
  selling_price_increase : ℝ
  profit_rate : ℝ
  discount_rate : ℝ

/-- Theorem about the unit cost of the first purchase and maximum discounted units --/
theorem air_conditioner_sales_theorem (sale : AirConditionerSales)
  (h1 : sale.first_purchase_total = 24000)
  (h2 : sale.first_purchase_unit_price = 3000)
  (h3 : sale.second_purchase_total = 52000)
  (h4 : sale.second_purchase_unit_price = sale.first_purchase_unit_price + 200)
  (h5 : sale.selling_price_increase = 200)
  (h6 : sale.profit_rate = 0.22)
  (h7 : sale.discount_rate = 0.95) :
  ∃ (first_unit_cost max_discounted : ℝ),
    first_unit_cost = 2400 ∧
    max_discounted = 8 ∧
    (sale.first_purchase_total / first_unit_cost) * 2 = sale.second_purchase_total / sale.second_purchase_unit_price ∧
    sale.first_purchase_unit_price * (sale.first_purchase_total / first_unit_cost) +
    (sale.first_purchase_unit_price + sale.selling_price_increase) * sale.discount_rate * max_discounted +
    (sale.first_purchase_unit_price + sale.selling_price_increase) * ((sale.second_purchase_total / sale.second_purchase_unit_price) - max_discounted) ≥
    (sale.first_purchase_total + sale.second_purchase_total) * (1 + sale.profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_sales_theorem_l1179_117911


namespace NUMINAMATH_CALUDE_factors_of_product_l1179_117989

/-- A natural number with exactly three factors is the square of a prime. -/
def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The number of factors of n^k where n is a prime square. -/
def num_factors_prime_square_pow (n k : ℕ) : ℕ :=
  2 * k + 1

/-- The main theorem -/
theorem factors_of_product (a b c : ℕ) 
  (ha : is_prime_square a) 
  (hb : is_prime_square b)
  (hc : is_prime_square c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (num_factors_prime_square_pow a 3) * 
  (num_factors_prime_square_pow b 4) * 
  (num_factors_prime_square_pow c 5) = 693 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_product_l1179_117989


namespace NUMINAMATH_CALUDE_probability_same_number_four_dice_l1179_117927

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being rolled -/
def numberOfDice : ℕ := 4

/-- The probability of all dice showing the same number -/
def probabilitySameNumber : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem probability_same_number_four_dice :
  probabilitySameNumber = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_number_four_dice_l1179_117927


namespace NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l1179_117933

/-- A characteristic associated with a polynomial -/
def A_P (P : Polynomial ℝ) : Type :=
  sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from A_P -/
theorem degree_not_determined_by_A_P :
  ∃ (P₁ P₂ : Polynomial ℝ), A_P P₁ = A_P P₂ ∧ Polynomial.degree P₁ ≠ Polynomial.degree P₂ :=
sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_A_P_l1179_117933


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l1179_117902

theorem quadratic_inequality_solution_condition (k : ℝ) : 
  (k > 0) → 
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ 
  (k > 0 ∧ k < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l1179_117902


namespace NUMINAMATH_CALUDE_complex_subtraction_l1179_117966

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 2 + 3*I) (h₂ : z₂ = 3 + I) : 
  z₁ - z₂ = -1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1179_117966


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1179_117959

/-- The probability that at least one of three independent events occurs, 
    given that each event has a probability of 1/3. -/
theorem prob_at_least_one_of_three (p : ℝ) (h_p : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l1179_117959


namespace NUMINAMATH_CALUDE_elevator_exit_probability_l1179_117925

/-- The number of floors where people can exit the elevator -/
def num_floors : ℕ := 9

/-- The probability that two people exit the elevator on different floors -/
def prob_different_floors : ℚ := 8 / 9

theorem elevator_exit_probability :
  (num_floors : ℚ) * (num_floors - 1) / (num_floors * num_floors) = prob_different_floors := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_probability_l1179_117925


namespace NUMINAMATH_CALUDE_middle_school_sample_size_l1179_117908

/-- Represents the number of schools to be sampled in a stratified sampling scenario -/
def stratified_sample (total : ℕ) (category : ℕ) (sample_size : ℕ) : ℕ :=
  (category * sample_size) / total

/-- Theorem stating the correct number of middle schools to be sampled -/
theorem middle_school_sample_size :
  let total_schools : ℕ := 700
  let middle_schools : ℕ := 200
  let sample_size : ℕ := 70
  stratified_sample total_schools middle_schools sample_size = 20 := by
  sorry


end NUMINAMATH_CALUDE_middle_school_sample_size_l1179_117908


namespace NUMINAMATH_CALUDE_series_solution_l1179_117992

-- Define the series
def S (x y : ℝ) : ℝ := 1 + 2*x*y + 3*(x*y)^2 + 4*(x*y)^3 + 5*(x*y)^4 + 6*(x*y)^5 + 7*(x*y)^6 + 8*(x*y)^7

-- State the theorem
theorem series_solution :
  ∃ (x y : ℝ), S x y = 16 ∧ x = 3/4 ∧ (y = 1 ∨ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_series_solution_l1179_117992


namespace NUMINAMATH_CALUDE_festival_sunny_days_l1179_117975

def probability_exactly_two_sunny (n : ℕ) (p : ℝ) : ℝ :=
  (n.choose 2 : ℝ) * (1 - p)^2 * p^(n - 2)

theorem festival_sunny_days :
  probability_exactly_two_sunny 5 0.6 = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_festival_sunny_days_l1179_117975


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_one_l1179_117948

theorem x_squared_plus_y_squared_equals_one
  (x y : ℝ)
  (h1 : (x^2 + y^2 + 1) * (x^2 + y^2 + 3) = 8)
  (h2 : x^2 + y^2 ≥ 0) :
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_one_l1179_117948


namespace NUMINAMATH_CALUDE_locus_of_P_perpendicular_line_through_focus_l1179_117971

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M on the ellipse
def point_M (x y : ℝ) : Prop := ellipse_C x y

-- Define the point N as the foot of the perpendicular from M to x-axis
def point_N (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point P
def point_P (x y : ℝ) (mx my : ℝ) : Prop :=
  point_M mx my ∧ (x - mx)^2 + y^2 = 2 * my^2

-- Define the point Q
def point_Q (y : ℝ) : ℝ × ℝ := (-3, y)

-- Theorem 1: The locus of P is a circle
theorem locus_of_P (x y : ℝ) :
  (∃ mx my, point_P x y mx my) → x^2 + y^2 = 2 :=
sorry

-- Theorem 2: Line through P perpendicular to OQ passes through left focus
theorem perpendicular_line_through_focus (x y qy : ℝ) (mx my : ℝ) :
  point_P x y mx my →
  (x * (-3 - x) + y * (qy - y) = 1) →
  (∃ t : ℝ, x + t * (qy - y) = -1 ∧ y - t * (-3 - x) = 0) :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_perpendicular_line_through_focus_l1179_117971


namespace NUMINAMATH_CALUDE_real_roots_condition_l1179_117995

theorem real_roots_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 5*x*y + 2*x + 9 = 0) ↔ (x ≤ -0.6 ∨ x ≥ 0.92) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l1179_117995


namespace NUMINAMATH_CALUDE_cube_not_always_positive_l1179_117919

theorem cube_not_always_positive : 
  ¬(∀ x : ℝ, x^3 > 0) := by sorry

end NUMINAMATH_CALUDE_cube_not_always_positive_l1179_117919


namespace NUMINAMATH_CALUDE_total_notebooks_purchased_l1179_117953

def john_purchases : List Nat := [2, 4, 6, 8, 10]
def wife_purchases : List Nat := [3, 7, 5, 9, 11]

theorem total_notebooks_purchased : 
  (List.sum john_purchases) + (List.sum wife_purchases) = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_purchased_l1179_117953


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1179_117997

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2 / (n * (a n + 3))

-- Define the sum S_n
def S (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_proof :
  (a 3 = 5) ∧ (a 17 = 3 * a 6) ∧
  (∀ n : ℕ, n > 0 → b n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → S n = n / (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l1179_117997


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l1179_117949

theorem existence_of_non_divisible_pair (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l1179_117949


namespace NUMINAMATH_CALUDE_f_equals_mod_l1179_117926

def g (n : ℤ) : ℕ :=
  if n ≥ 1 then 1 else 0

def f : ℕ → ℕ → ℕ
  | 0, m => 0
  | n+1, m => ((1 - g m + g m * g (m-1-(f n m))) * (1 + f n m)) % m

theorem f_equals_mod (n m : ℕ) : f n m = n % m := by
  sorry

end NUMINAMATH_CALUDE_f_equals_mod_l1179_117926


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l1179_117954

/-- Represents the position of the cat or mouse -/
inductive Position
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The number of moves in the problem -/
def totalMoves : ℕ := 315

/-- The length of the cat's movement cycle -/
def catCycleLength : ℕ := 4

/-- The length of the mouse's movement cycle -/
def mouseCycleLength : ℕ := 8

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : Position :=
  match moves % catCycleLength with
  | 0 => Position.TopLeft
  | 1 => Position.TopRight
  | 2 => Position.BottomRight
  | 3 => Position.BottomLeft
  | _ => Position.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : Position :=
  match moves % mouseCycleLength with
  | 0 => Position.TopMiddle
  | 1 => Position.TopRight
  | 2 => Position.RightMiddle
  | 3 => Position.BottomRight
  | 4 => Position.BottomMiddle
  | 5 => Position.BottomLeft
  | 6 => Position.LeftMiddle
  | 7 => Position.TopLeft
  | _ => Position.TopMiddle  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions : 
  catPosition totalMoves = Position.BottomRight ∧ 
  mousePosition totalMoves = Position.RightMiddle := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l1179_117954


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1179_117963

theorem complex_modulus_problem (i z : ℂ) (h1 : i^2 = -1) (h2 : i * z = (1 - 2*i)^2) : 
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1179_117963


namespace NUMINAMATH_CALUDE_range_of_a_l1179_117935

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  (a ≤ -2 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1179_117935


namespace NUMINAMATH_CALUDE_intersection_points_count_l1179_117903

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  nonconcurrent : Bool  -- Represents that no three diagonals are concurrent

/-- The number of intersection points of diagonals inside a convex polygon -/
def intersectionPoints (p : ConvexPolygon n) : ℕ := sorry

/-- Theorem: The number of intersection points of diagonals inside a convex n-gon
    where no three diagonals are concurrent is equal to (n choose 4) -/
theorem intersection_points_count (n : ℕ) (p : ConvexPolygon n) 
    (h : p.nonconcurrent = true) : 
  intersectionPoints p = Nat.choose n 4 := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1179_117903


namespace NUMINAMATH_CALUDE_brayden_gavin_touchdowns_l1179_117988

theorem brayden_gavin_touchdowns :
  let touchdown_points : ℕ := 7
  let cole_freddy_touchdowns : ℕ := 9
  let point_difference : ℕ := 14
  let brayden_gavin_touchdowns : ℕ := 7

  touchdown_points * cole_freddy_touchdowns = 
  touchdown_points * brayden_gavin_touchdowns + point_difference :=
by
  sorry

end NUMINAMATH_CALUDE_brayden_gavin_touchdowns_l1179_117988


namespace NUMINAMATH_CALUDE_solutions_count_l1179_117981

/-- The number of solutions to the Diophantine equation 3x + 5y = 805 where x and y are positive integers -/
def num_solutions : ℕ :=
  (Finset.filter (fun t : ℕ => 265 - 5 * t > 0 ∧ 2 + 3 * t > 0) (Finset.range 53)).card

theorem solutions_count : num_solutions = 53 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_l1179_117981


namespace NUMINAMATH_CALUDE_actual_tax_raise_expectation_l1179_117965

-- Define the population
def Population := ℝ

-- Define the fraction of liars and economists
def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 0.9

-- Define the affirmative answer percentages
def taxes_raised : ℝ := 0.4
def money_supply_increased : ℝ := 0.3
def bonds_issued : ℝ := 0.5
def reserves_spent : ℝ := 0

-- Define the theorem
theorem actual_tax_raise_expectation :
  let total_affirmative := taxes_raised + money_supply_increased + bonds_issued + reserves_spent
  fraction_liars * 3 + fraction_economists = total_affirmative →
  taxes_raised - fraction_liars = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_actual_tax_raise_expectation_l1179_117965


namespace NUMINAMATH_CALUDE_middle_segment_length_l1179_117934

theorem middle_segment_length (a b c : ℝ) (ha : a = 1) (hb : b = 3) 
  (hc : c * c = a * b) : c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_middle_segment_length_l1179_117934


namespace NUMINAMATH_CALUDE_train_length_l1179_117978

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 150 → time = 12 → ∃ length : ℝ, abs (length - 500.04) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1179_117978


namespace NUMINAMATH_CALUDE_energy_drink_cost_l1179_117952

theorem energy_drink_cost (cupcakes : Nat) (cupcake_price : ℚ) 
  (cookies : Nat) (cookie_price : ℚ) (basketballs : Nat) 
  (basketball_price : ℚ) (energy_drinks : Nat) :
  cupcakes = 50 →
  cupcake_price = 2 →
  cookies = 40 →
  cookie_price = 1/2 →
  basketballs = 2 →
  basketball_price = 40 →
  energy_drinks = 20 →
  (cupcakes * cupcake_price + cookies * cookie_price - basketballs * basketball_price) / energy_drinks = 2 := by
sorry


end NUMINAMATH_CALUDE_energy_drink_cost_l1179_117952


namespace NUMINAMATH_CALUDE_decimal_34_to_binary_l1179_117951

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_34_to_binary :
  decimal_to_binary 34 = [1, 0, 0, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_34_to_binary_l1179_117951


namespace NUMINAMATH_CALUDE_darryl_honeydews_l1179_117985

/-- Represents the problem of determining the initial number of honeydews --/
def honeydew_problem (initial_cantaloupes : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (cantaloupe_price : ℕ) (honeydew_price : ℕ)
  (total_revenue : ℕ) : Prop :=
  ∃ (initial_honeydews : ℕ),
    -- Revenue calculation
    (initial_cantaloupes - dropped_cantaloupes - final_cantaloupes) * cantaloupe_price +
    (initial_honeydews - rotten_honeydews - final_honeydews) * honeydew_price = total_revenue

theorem darryl_honeydews :
  honeydew_problem 30 8 9 2 3 2 3 85 →
  ∃ (initial_honeydews : ℕ), initial_honeydews = 27 :=
sorry

end NUMINAMATH_CALUDE_darryl_honeydews_l1179_117985


namespace NUMINAMATH_CALUDE_cadastral_value_calculation_l1179_117943

/-- Calculates the cadastral value of a land plot given the tax amount and tax rate -/
theorem cadastral_value_calculation (tax_amount : ℝ) (tax_rate : ℝ) :
  tax_amount = 4500 →
  tax_rate = 0.003 →
  tax_amount = tax_rate * 1500000 := by
  sorry

#check cadastral_value_calculation

end NUMINAMATH_CALUDE_cadastral_value_calculation_l1179_117943


namespace NUMINAMATH_CALUDE_complex_fourth_power_l1179_117968

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l1179_117968


namespace NUMINAMATH_CALUDE_shortest_routes_equals_45_l1179_117958

/-- Calculates the number of shortest paths between two points on a grid. -/
def gridPaths (x y : ℕ) : ℕ := sorry

/-- The number of shortest routes from A(0,0) to C(x_C, y_C) passing through B(x_B, y_B) -/
def shortestRoutes (x_B y_B x_C y_C : ℕ) : ℕ :=
  (gridPaths x_B y_B) * (gridPaths (x_C - x_B) (y_C - y_B))

theorem shortest_routes_equals_45 (x_B y_B x_C y_C : ℕ) : 
  shortestRoutes x_B y_B x_C y_C = 45 := by sorry

end NUMINAMATH_CALUDE_shortest_routes_equals_45_l1179_117958


namespace NUMINAMATH_CALUDE_infinitely_many_rational_pairs_sum_equals_product_l1179_117940

theorem infinitely_many_rational_pairs_sum_equals_product :
  ∃ f : ℚ → ℚ × ℚ, Function.Injective f ∧ ∀ z, (f z).1 + (f z).2 = (f z).1 * (f z).2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_rational_pairs_sum_equals_product_l1179_117940


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1179_117904

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1179_117904


namespace NUMINAMATH_CALUDE_cube_equality_condition_l1179_117906

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  (edge_length : n > 3)

/-- The number of unit cubes with exactly two faces painted -/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 4)

/-- The number of unit cubes with no faces painted -/
def no_faces_painted (c : Cube n) : ℕ := (n - 2)^3

/-- Theorem stating the equality condition for n = 5 -/
theorem cube_equality_condition (n : ℕ) (c : Cube n) :
  two_faces_painted c = no_faces_painted c ↔ n = 5 :=
sorry

end NUMINAMATH_CALUDE_cube_equality_condition_l1179_117906


namespace NUMINAMATH_CALUDE_pages_difference_l1179_117913

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day (when not resting) -/
def pages_per_day_B : ℕ := 13

/-- The total number of days -/
def total_days : ℕ := 7

/-- The number of days in Person B's reading cycle -/
def cycle_days : ℕ := 3

/-- The number of days Person B reads in a cycle -/
def reading_days_per_cycle : ℕ := 2

/-- Calculate the number of pages read by Person A -/
def pages_read_A : ℕ := total_days * pages_per_day_A

/-- Calculate the number of full cycles in the total days -/
def full_cycles : ℕ := total_days / cycle_days

/-- Calculate the number of days Person B reads -/
def reading_days_B : ℕ := full_cycles * reading_days_per_cycle + (total_days % cycle_days)

/-- Calculate the number of pages read by Person B -/
def pages_read_B : ℕ := reading_days_B * pages_per_day_B

/-- The theorem to prove -/
theorem pages_difference : pages_read_B - pages_read_A = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l1179_117913


namespace NUMINAMATH_CALUDE_min_value_x_l1179_117991

theorem min_value_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : ∀ (a b : ℝ), a > 0 → b > 0 → 1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2)
  (h2 : ∀ (a b : ℝ), a > 0 → b > 0 → 4 * a + b * (1 - a) = 0) :
  x ≥ 1 ∧ ∀ (y : ℝ), y > 0 → y < 1 → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / a^2 + 16 / b^2 < 1 + y / 2 - y^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_l1179_117991


namespace NUMINAMATH_CALUDE_lcm_1540_2310_l1179_117999

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1540_2310_l1179_117999


namespace NUMINAMATH_CALUDE_monkey_fruit_ratio_l1179_117984

theorem monkey_fruit_ratio (a b x y z : ℝ) : 
  a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
  x = 1.4 * a →
  y + 0.25 * y = 1.25 * y →
  b = 2 * z →
  a + b = x + y →
  a + b = z + 1.4 * a →
  a / b = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_monkey_fruit_ratio_l1179_117984


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l1179_117979

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4*x^3 - 1

-- Theorem statement
theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  f' P.1 = 3 →  -- The slope of the tangent line at P is 3
  f P.1 = P.2 → -- P lies on the curve f(x)
  P = (1, 0) := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l1179_117979


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l1179_117900

theorem double_acute_angle_range (α : Real) (h : 0 < α ∧ α < π / 2) : 
  0 < 2 * α ∧ 2 * α < π := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l1179_117900


namespace NUMINAMATH_CALUDE_xy_minus_two_equals_negative_one_l1179_117980

theorem xy_minus_two_equals_negative_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y - 2 = -1 := by
sorry

end NUMINAMATH_CALUDE_xy_minus_two_equals_negative_one_l1179_117980


namespace NUMINAMATH_CALUDE_log_cutting_l1179_117901

theorem log_cutting (fallen_pieces fixed_pieces : ℕ) 
  (h1 : fallen_pieces = 10)
  (h2 : fixed_pieces = 2) :
  fallen_pieces + fixed_pieces - 1 = 11 := by
sorry

end NUMINAMATH_CALUDE_log_cutting_l1179_117901


namespace NUMINAMATH_CALUDE_equation_solution_l1179_117977

theorem equation_solution :
  ∃ y : ℝ, (5 : ℝ)^(2*y) * (25 : ℝ)^y = (625 : ℝ)^3 ∧ y = 3 :=
by
  -- Define 25 and 625 in terms of 5
  have h1 : (25 : ℝ) = (5 : ℝ)^2 := by sorry
  have h2 : (625 : ℝ) = (5 : ℝ)^4 := by sorry

  -- Prove the existence of y
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1179_117977


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1179_117983

open Set Real

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_intersection_A_B :
  (Aᶜ ∪ Bᶜ) = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1179_117983


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1179_117972

theorem water_tank_capacity (c : ℚ) : 
  (1 / 5 : ℚ) * c + 5 = (2 / 7 : ℚ) * c → c = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1179_117972


namespace NUMINAMATH_CALUDE_min_sum_xyz_l1179_117944

theorem min_sum_xyz (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) :
  ∀ (a b c : ℤ), (a - 10) * (b - 5) * (c - 2) = 1000 → x + y + z ≤ a + b + c :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xyz_l1179_117944


namespace NUMINAMATH_CALUDE_quadrilateral_max_area_and_angles_l1179_117930

/-- A quadrilateral with two sides of length 3 and two sides of length 4 -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (side1_eq_3 : side1 = 3)
  (side2_eq_4 : side2 = 4)
  (side3_eq_3 : side3 = 3)
  (side4_eq_4 : side4 = 4)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The angles of a quadrilateral -/
def angles (q : Quadrilateral) : Fin 4 → ℝ := sorry

/-- The sum of two opposite angles in a quadrilateral -/
def opposite_angles_sum (q : Quadrilateral) : ℝ := 
  angles q 0 + angles q 2

theorem quadrilateral_max_area_and_angles (q : Quadrilateral) : 
  (∀ q' : Quadrilateral, area q' ≤ area q) → 
  (area q = 12 ∧ opposite_angles_sum q = 180) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_max_area_and_angles_l1179_117930


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1179_117916

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1179_117916


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1179_117960

/-- Given a right triangle PQR with legs of length 9 and 12, prove that a square inscribed
    with one side on the hypotenuse and vertices on the other two sides has side length 45/8 -/
theorem inscribed_square_side_length (P Q R : ℝ × ℝ) 
  (right_angle_P : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  (leg_PQ : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 9^2)
  (leg_PR : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 12^2)
  (square : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)
  (inscribed : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (A.1 - Q.1) * (R.1 - Q.1) + (A.2 - Q.2) * (R.2 - Q.2) = 0 ∧
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)) ∧
    (∃ u : ℝ, 0 < u ∧ u < 1 ∧ D = (u * P.1 + (1 - u) * Q.1, u * P.2 + (1 - u) * Q.2)) ∧
    (∃ v : ℝ, 0 < v ∧ v < 1 ∧ C = (v * P.1 + (1 - v) * R.1, v * P.2 + (1 - v) * R.2)))
  : ∃ (A B C D : ℝ × ℝ), square A B C D ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (45/8)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1179_117960


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l1179_117955

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 9
def circle3 (x y : ℝ) : Prop := (x - 5)^2 + (y - 2)^2 = 16

-- Define the intersection points
def intersection (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem intersection_distance_squared :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection x1 y1 ∧
    intersection x2 y2 ∧
    circle3 x1 y1 ∧
    circle3 x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 224 / 9 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l1179_117955


namespace NUMINAMATH_CALUDE_debra_accusation_l1179_117962

/-- Represents the number of cookies in various states -/
structure CookieCount where
  initial : ℕ
  louSeniorEaten : ℕ
  louieJuniorTaken : ℕ
  remaining : ℕ

/-- The cookie scenario as described in the problem -/
def cookieScenario : CookieCount where
  initial := 22
  louSeniorEaten := 4
  louieJuniorTaken := 7
  remaining := 11

/-- Theorem stating the portion of cookies Debra accuses Lou Senior of eating -/
theorem debra_accusation (c : CookieCount) (h1 : c = cookieScenario) :
  c.louSeniorEaten = 4 ∧ c.initial = 22 := by sorry

end NUMINAMATH_CALUDE_debra_accusation_l1179_117962


namespace NUMINAMATH_CALUDE_average_age_combined_l1179_117938

theorem average_age_combined (n_students : Nat) (n_parents : Nat)
  (avg_age_students : ℚ) (avg_age_parents : ℚ)
  (h1 : n_students = 50)
  (h2 : n_parents = 75)
  (h3 : avg_age_students = 10)
  (h4 : avg_age_parents = 40) :
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l1179_117938


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l1179_117990

theorem modulo_eleven_residue : (312 + 6 * 47 + 8 * 154 + 5 * 22) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l1179_117990


namespace NUMINAMATH_CALUDE_quadrilateral_area_relation_l1179_117987

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define the intersection point of diagonals
def O : ℝ × ℝ := sorry

-- Define a point P inside triangle AOB
variable (P : ℝ × ℝ)

-- Assume P is inside triangle AOB
axiom P_inside_AOB : sorry

-- Define the area function for triangles
def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_area_relation :
  area P C D - area P A B = area P A C + area P B D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_relation_l1179_117987


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1179_117924

-- Define the curve
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - k) + y^2 / (1 + k) = 1

-- Define the conditions for an ellipse
def is_ellipse (k : ℝ) : Prop :=
  1 - k > 0 ∧ 1 + k > 0 ∧ 1 - k ≠ 1 + k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ ((-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1179_117924


namespace NUMINAMATH_CALUDE_class_visual_conditions_most_suitable_l1179_117970

/-- Represents a survey method --/
inductive SurveyMethod
  | EnergyLamps
  | ClassVisualConditions
  | ProvinceInternetUsage
  | CanalFishTypes

/-- Defines what constitutes a comprehensive investigation --/
def isComprehensive (method : SurveyMethod) : Prop :=
  match method with
  | .ClassVisualConditions => true
  | _ => false

/-- Theorem stating that understanding the visual conditions of Class 803 
    is the most suitable method for a comprehensive investigation --/
theorem class_visual_conditions_most_suitable :
  ∀ (method : SurveyMethod), 
    isComprehensive method → method = SurveyMethod.ClassVisualConditions :=
by sorry

end NUMINAMATH_CALUDE_class_visual_conditions_most_suitable_l1179_117970


namespace NUMINAMATH_CALUDE_complex_equation_product_l1179_117993

theorem complex_equation_product (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l1179_117993


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1179_117922

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -5/3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1179_117922


namespace NUMINAMATH_CALUDE_jordana_age_proof_l1179_117910

/-- Jennifer's age in ten years -/
def jennifer_future_age : ℕ := 30

/-- Number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_relative_age : ℕ := 3

/-- Calculate Jordana's current age -/
def jordana_current_age : ℕ :=
  jennifer_future_age * jordana_relative_age - years_ahead

theorem jordana_age_proof :
  jordana_current_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_jordana_age_proof_l1179_117910


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1179_117907

theorem fraction_sum_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1179_117907


namespace NUMINAMATH_CALUDE_no_rational_roots_odd_coeff_l1179_117945

theorem no_rational_roots_odd_coeff (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Int.gcd p q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_rational_roots_odd_coeff_l1179_117945


namespace NUMINAMATH_CALUDE_largest_marble_count_l1179_117931

theorem largest_marble_count : ∃ n : ℕ, n < 400 ∧ 
  n % 3 = 1 ∧ n % 7 = 2 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m < 400 → m % 3 = 1 → m % 7 = 2 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_marble_count_l1179_117931


namespace NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l1179_117917

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_six_million_scientific_notation :
  toScientificNotation 2600000 = ScientificNotation.mk 2.6 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l1179_117917


namespace NUMINAMATH_CALUDE_sqrt_19_bounds_l1179_117986

theorem sqrt_19_bounds : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  have h1 : 16 < 19 := by sorry
  have h2 : 19 < 25 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_19_bounds_l1179_117986


namespace NUMINAMATH_CALUDE_root_range_implies_k_range_l1179_117942

theorem root_range_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_k_range_l1179_117942


namespace NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l1179_117973

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = 3 ∨ m = -3) →  -- |m| = 3
  ((a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4) := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l1179_117973


namespace NUMINAMATH_CALUDE_kids_to_adult_meals_ratio_l1179_117964

theorem kids_to_adult_meals_ratio 
  (kids_meals : ℕ) 
  (total_meals : ℕ) 
  (h1 : kids_meals = 8) 
  (h2 : total_meals = 12) : 
  (kids_meals : ℚ) / ((total_meals - kids_meals) : ℚ) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_kids_to_adult_meals_ratio_l1179_117964


namespace NUMINAMATH_CALUDE_fred_car_wash_earnings_l1179_117982

/-- Fred's earnings from washing the family car -/
def car_wash_earnings (weekly_allowance : ℕ) (final_amount : ℕ) : ℕ :=
  final_amount - weekly_allowance / 2

/-- Proof that Fred earned $6 from washing the family car -/
theorem fred_car_wash_earnings :
  car_wash_earnings 16 14 = 6 :=
by sorry

end NUMINAMATH_CALUDE_fred_car_wash_earnings_l1179_117982


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1179_117941

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 * 7 / 9 = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1179_117941


namespace NUMINAMATH_CALUDE_certain_number_proof_l1179_117969

theorem certain_number_proof (x : ℝ) : (((x + 10) * 2) / 2) - 2 = 88 / 2 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1179_117969


namespace NUMINAMATH_CALUDE_sequence_properties_l1179_117915

def S (n : ℕ) : ℚ := (1/2) * n^2 + (1/2) * n

def a (n : ℕ) : ℚ := n

def b (n : ℕ) : ℚ := a n * 2^(n-1)

def T (n : ℕ) : ℚ := (n-1) * 2^n + 1

theorem sequence_properties (n : ℕ) :
  (∀ k, k ≤ n → S k = (1/2) * k^2 + (1/2) * k) →
  (∀ k, k ≤ n → a k = k) ∧
  (T n = (n-1) * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1179_117915


namespace NUMINAMATH_CALUDE_sara_salad_cost_l1179_117967

/-- The cost of Sara's lunch items -/
structure LunchCost where
  hotdog : ℝ
  total : ℝ

/-- Calculates the cost of the salad given the total lunch cost and hotdog cost -/
def salad_cost (lunch : LunchCost) : ℝ :=
  lunch.total - lunch.hotdog

/-- Theorem stating that Sara's salad cost $5.10 -/
theorem sara_salad_cost :
  let lunch : LunchCost := { hotdog := 5.36, total := 10.46 }
  salad_cost lunch = 5.10 := by
  sorry

end NUMINAMATH_CALUDE_sara_salad_cost_l1179_117967


namespace NUMINAMATH_CALUDE_congruence_solution_l1179_117957

theorem congruence_solution (n : ℕ) (b : ℕ) (h1 : n ≥ 2) (h2 : b < n) :
  (∃ y : ℤ, (10 * y + 3) % 15 = 7 % 15) →
  (∃ y : ℤ, y % 3 = 2 % 3) :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1179_117957


namespace NUMINAMATH_CALUDE_bracket_two_equals_twelve_l1179_117912

def bracket (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem bracket_two_equals_twelve : bracket 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bracket_two_equals_twelve_l1179_117912


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1179_117961

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1179_117961


namespace NUMINAMATH_CALUDE_gcd_cube_plus_sixteen_and_plus_four_l1179_117994

theorem gcd_cube_plus_sixteen_and_plus_four (n : ℕ) (h : n > 2^4) :
  Nat.gcd (n^3 + 4^2) (n + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_sixteen_and_plus_four_l1179_117994


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1179_117914

theorem functional_equation_solution (f : ℝ → ℝ) (h_continuous : Continuous f) 
  (h_equation : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x)) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1179_117914


namespace NUMINAMATH_CALUDE_not_perfect_square_l1179_117956

theorem not_perfect_square (n : ℕ) : ∀ m : ℕ, 4 * n^2 + 4 * n + 4 ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1179_117956


namespace NUMINAMATH_CALUDE_victors_allowance_l1179_117974

theorem victors_allowance (initial_amount final_amount allowance : ℕ) :
  initial_amount = 10 →
  final_amount = 18 →
  allowance = final_amount - initial_amount →
  allowance = 8 := by
  sorry

end NUMINAMATH_CALUDE_victors_allowance_l1179_117974
