import Mathlib

namespace NUMINAMATH_CALUDE_total_cookies_and_brownies_l3538_353847

theorem total_cookies_and_brownies :
  let cookie_bags : ℕ := 272
  let cookies_per_bag : ℕ := 45
  let brownie_bags : ℕ := 158
  let brownies_per_bag : ℕ := 32
  cookie_bags * cookies_per_bag + brownie_bags * brownies_per_bag = 17296 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_and_brownies_l3538_353847


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_eight_thirds_l3538_353828

theorem greatest_integer_less_than_negative_eight_thirds :
  Int.floor (-8/3 : ℚ) = -3 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_eight_thirds_l3538_353828


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3538_353804

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ d, ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  2 * a 6 + 2 * a 8 = (a 7) ^ 2 → a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3538_353804


namespace NUMINAMATH_CALUDE_hyperbola_focus_m_value_l3538_353874

/-- Given a hyperbola with equation 3mx^2 - my^2 = 3 and one focus at (0, 2), prove that m = -1 -/
theorem hyperbola_focus_m_value (m : ℝ) : 
  (∃ (x y : ℝ), 3 * m * x^2 - m * y^2 = 3) →  -- Hyperbola equation
  (∃ (a b : ℝ), a^2 / (3/m) + b^2 / (1/m) = 1) →  -- Standard form of hyperbola
  (2 : ℝ)^2 = (3/m) + (1/m) →  -- Focus property
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_m_value_l3538_353874


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l3538_353839

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l3538_353839


namespace NUMINAMATH_CALUDE_pennsylvania_quarters_l3538_353879

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  ⌊total * state_fraction * penn_fraction⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_pennsylvania_quarters_l3538_353879


namespace NUMINAMATH_CALUDE_positive_sum_reciprocal_inequality_l3538_353823

theorem positive_sum_reciprocal_inequality (p : ℝ) (hp : p > 0) :
  p + 1/p > 2 ↔ p ≠ 1 := by sorry

end NUMINAMATH_CALUDE_positive_sum_reciprocal_inequality_l3538_353823


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3538_353848

def A : Set ℝ := {x | x^2 - 4*x - 5 > 0}
def B : Set ℝ := {x | 4 - x^2 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -2 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3538_353848


namespace NUMINAMATH_CALUDE_min_abs_z_l3538_353894

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - (3 + 2*I)) = 7) :
  ∃ (z_min : ℂ), ∀ (w : ℂ), Complex.abs (w - 2*I) + Complex.abs (w - (3 + 2*I)) = 7 →
    Complex.abs z_min ≤ Complex.abs w ∧ Complex.abs z_min = 2 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l3538_353894


namespace NUMINAMATH_CALUDE_remaining_digits_average_l3538_353858

theorem remaining_digits_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 20 →
  subset = 14 →
  total_avg = 500 →
  subset_avg = 390 →
  let remaining := total - subset
  let remaining_sum := total * total_avg - subset * subset_avg
  remaining_sum / remaining = 756.67 := by
  sorry

end NUMINAMATH_CALUDE_remaining_digits_average_l3538_353858


namespace NUMINAMATH_CALUDE_johns_socks_theorem_l3538_353815

/-- The number of pairs of matched socks John initially had -/
def initial_pairs : ℕ := 9

/-- The number of individual socks John loses -/
def lost_socks : ℕ := 5

/-- The greatest number of pairs of matched socks John can have left after losing socks -/
def remaining_pairs : ℕ := 7

theorem johns_socks_theorem :
  (2 * initial_pairs - lost_socks ≥ 2 * remaining_pairs) ∧
  (2 * (initial_pairs - 1) - lost_socks < 2 * remaining_pairs) := by
  sorry

end NUMINAMATH_CALUDE_johns_socks_theorem_l3538_353815


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3538_353830

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 + 3 * Complex.I) : z = 3 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3538_353830


namespace NUMINAMATH_CALUDE_component_scrap_probability_l3538_353813

theorem component_scrap_probability 
  (pass_first : ℝ) 
  (pass_second : ℝ) 
  (h1 : pass_first = 0.8) 
  (h2 : pass_second = 0.9) : 
  (1 - pass_first) * (1 - pass_second) = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_component_scrap_probability_l3538_353813


namespace NUMINAMATH_CALUDE_percentage_increase_l3538_353811

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 350 → final = 525 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3538_353811


namespace NUMINAMATH_CALUDE_probability_one_of_each_l3538_353806

/-- The number of forks in the drawer -/
def num_forks : ℕ := 7

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 8

/-- The number of knives in the drawer -/
def num_knives : ℕ := 5

/-- The total number of pieces of silverware -/
def total_pieces : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be selected -/
def num_selected : ℕ := 3

/-- The probability of selecting one fork, one spoon, and one knife -/
theorem probability_one_of_each : 
  (num_forks * num_spoons * num_knives : ℚ) / (Nat.choose total_pieces num_selected) = 14 / 57 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l3538_353806


namespace NUMINAMATH_CALUDE_nonagon_configuration_count_l3538_353896

structure NonagonConfiguration where
  vertices : Fin 9 → Fin 11
  center : Fin 11
  midpoint : Fin 11
  all_different : ∀ i j, i ≠ j → 
    (vertices i ≠ vertices j) ∧ 
    (vertices i ≠ center) ∧ 
    (vertices i ≠ midpoint) ∧ 
    (center ≠ midpoint)
  equal_sums : ∀ i : Fin 9, 
    (vertices i : ℕ) + (midpoint : ℕ) + (center : ℕ) = 
    (vertices 0 : ℕ) + (midpoint : ℕ) + (center : ℕ)

def count_valid_configurations : ℕ := sorry

theorem nonagon_configuration_count :
  count_valid_configurations = 10321920 := by sorry

end NUMINAMATH_CALUDE_nonagon_configuration_count_l3538_353896


namespace NUMINAMATH_CALUDE_family_income_increase_l3538_353840

theorem family_income_increase (I : ℝ) (S M F G : ℝ) : 
  I > 0 →
  S = 0.05 * I →
  M = 0.15 * I →
  F = 0.25 * I →
  G = I - S - M - F →
  (2 * G - G) / I = 0.55 := by
sorry

end NUMINAMATH_CALUDE_family_income_increase_l3538_353840


namespace NUMINAMATH_CALUDE_gcd_324_243_135_l3538_353816

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_243_135_l3538_353816


namespace NUMINAMATH_CALUDE_simplify_expression_l3538_353853

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y^3)^2 = 9 * x^4 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3538_353853


namespace NUMINAMATH_CALUDE_scientific_notation_425000_l3538_353850

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_425000 :
  toScientificNotation 425000 = ScientificNotation.mk 4.25 5 sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_425000_l3538_353850


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3538_353814

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3538_353814


namespace NUMINAMATH_CALUDE_proposition_implication_l3538_353877

theorem proposition_implication (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 10 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧ 
  (∃ x, 1 - m ≤ x ∧ x ≤ 1 + m ∧ (x < -2 ∨ x > 10)) ∧
  (m > 0) →
  m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l3538_353877


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3538_353852

theorem min_value_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → b ≤ x) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3538_353852


namespace NUMINAMATH_CALUDE_activities_equally_popular_l3538_353810

def dodgeball : Rat := 10 / 25
def artWorkshop : Rat := 12 / 30
def movieScreening : Rat := 18 / 45
def quizBowl : Rat := 16 / 40

theorem activities_equally_popular :
  dodgeball = artWorkshop ∧
  artWorkshop = movieScreening ∧
  movieScreening = quizBowl := by
  sorry

end NUMINAMATH_CALUDE_activities_equally_popular_l3538_353810


namespace NUMINAMATH_CALUDE_product_of_exponents_l3538_353895

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 54 → 
  2^5 + 7^s = 375 → 
  p * r * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l3538_353895


namespace NUMINAMATH_CALUDE_inequality_proof_l3538_353801

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3538_353801


namespace NUMINAMATH_CALUDE_point_movement_l3538_353851

/-- A point in the 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Move a point up by a given number of units -/
def moveUp (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

/-- Move a point left by a given number of units -/
def moveLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

theorem point_movement :
  let A : Point2D := { x := 1, y := -2 }
  let B : Point2D := moveLeft (moveUp A 3) 2
  B.x = -1 ∧ B.y = 1 := by sorry

end NUMINAMATH_CALUDE_point_movement_l3538_353851


namespace NUMINAMATH_CALUDE_min_value_theorem_l3538_353864

-- Define the equation
def equation (x y : ℝ) : Prop := y^2 - 2*x + 4 = 0

-- Define the expression to minimize
def expression (x y : ℝ) : ℝ := x^2 + y^2 + 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min : ℝ), min = -8 ∧
  (∀ (x y : ℝ), equation x y → expression x y ≥ min) ∧
  (∃ (x y : ℝ), equation x y ∧ expression x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3538_353864


namespace NUMINAMATH_CALUDE_students_per_bus_l3538_353859

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l3538_353859


namespace NUMINAMATH_CALUDE_parabola_sum_l3538_353825

/-- Represents a parabola of the form x = dy^2 + ey + f -/
structure Parabola where
  d : ℚ
  e : ℚ
  f : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.xCoord (p : Parabola) (y : ℚ) : ℚ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_sum (p : Parabola) :
  p.xCoord (-6) = 7 →  -- vertex condition
  p.xCoord (-3) = 2 →  -- point condition
  p.d + p.e + p.f = -182/9 := by
  sorry

#eval (-5/9 : ℚ) + (-20/3 : ℚ) + (-13 : ℚ)  -- Should evaluate to -182/9

end NUMINAMATH_CALUDE_parabola_sum_l3538_353825


namespace NUMINAMATH_CALUDE_product_of_x_values_l3538_353835

theorem product_of_x_values (x : ℝ) : 
  (|10 / x - 4| = 3) → 
  (∃ y : ℝ, (|10 / y - 4| = 3) ∧ x * y = 100 / 7) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l3538_353835


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3538_353872

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 5 = 11 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3538_353872


namespace NUMINAMATH_CALUDE_solution_difference_l3538_353867

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 25 * r - 125) →
  ((s - 5) * (s + 5) = 25 * s - 125) →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3538_353867


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3538_353857

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 30) (h2 : n2 = 50) (h3 : avg1 = 40) (h4 : avg2 = 90) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3538_353857


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l3538_353875

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 3 units right and 5 units up,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 22 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 2*(x-3)^2 - (x-3) + 7 + 5 = a*x^2 + b*x + c) → 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l3538_353875


namespace NUMINAMATH_CALUDE_triangle_area_is_24_l3538_353846

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (0, 6)
def vertex3 : ℝ × ℝ := (8, 10)

-- Define the triangle area calculation function
def triangleArea (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- State the theorem
theorem triangle_area_is_24 :
  triangleArea vertex1 vertex2 vertex3 = 24 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_24_l3538_353846


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sin_squared_fifteen_l3538_353871

theorem sqrt_three_minus_sin_squared_fifteen (π : Real) :
  (Real.sqrt 3) / 2 - Real.sqrt 3 * (Real.sin (π / 12))^2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sin_squared_fifteen_l3538_353871


namespace NUMINAMATH_CALUDE_y_intercepts_equal_negative_two_l3538_353824

-- Define the equations
def equation1 (x y : ℝ) : Prop := 2 * x - 3 * y = 6
def equation2 (x y : ℝ) : Prop := x + 4 * y = -8

-- Define y-intercept
def is_y_intercept (y : ℝ) (eq : ℝ → ℝ → Prop) : Prop := eq 0 y

-- Theorem statement
theorem y_intercepts_equal_negative_two :
  (is_y_intercept (-2) equation1) ∧ (is_y_intercept (-2) equation2) :=
sorry

end NUMINAMATH_CALUDE_y_intercepts_equal_negative_two_l3538_353824


namespace NUMINAMATH_CALUDE_container_capacity_sum_l3538_353837

/-- Represents the capacity and fill levels of a container -/
structure Container where
  capacity : ℝ
  initial_fill : ℝ
  final_fill : ℝ
  added_water : ℝ

/-- Calculates the total capacity of three containers -/
def total_capacity (a b c : Container) : ℝ :=
  a.capacity + b.capacity + c.capacity

/-- The problem statement -/
theorem container_capacity_sum : 
  ∃ (a b c : Container),
    a.initial_fill = 0.3 * a.capacity ∧
    a.final_fill = 0.75 * a.capacity ∧
    a.added_water = 36 ∧
    b.initial_fill = 0.4 * b.capacity ∧
    b.final_fill = 0.7 * b.capacity ∧
    b.added_water = 20 ∧
    c.initial_fill = 0.5 * c.capacity ∧
    c.final_fill = 2/3 * c.capacity ∧
    c.added_water = 12 ∧
    total_capacity a b c = 218.6666666666667 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_sum_l3538_353837


namespace NUMINAMATH_CALUDE_bucket_fill_time_l3538_353849

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that the time taken to fill the bucket completely is 150 seconds. -/
theorem bucket_fill_time (fill_rate : ℝ) (h : fill_rate * (2/3) = 1/100) :
  (1 / fill_rate) = 150 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l3538_353849


namespace NUMINAMATH_CALUDE_selling_price_optimal_l3538_353854

/-- Represents the selling price of toy A in yuan -/
def selling_price : ℝ := 65

/-- Represents the purchase price of toy A in yuan -/
def purchase_price : ℝ := 60

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.4

/-- Represents the daily profit target in yuan -/
def profit_target : ℝ := 2500

/-- Calculates the number of units sold per day based on the selling price -/
def units_sold (x : ℝ) : ℝ := 1800 - 20 * x

/-- Calculates the profit per unit based on the selling price -/
def profit_per_unit (x : ℝ) : ℝ := x - purchase_price

/-- Calculates the total daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

/-- Theorem stating that the selling price of 65 yuan results in the target profit
    while satisfying the profit margin constraint -/
theorem selling_price_optimal :
  daily_profit selling_price = profit_target ∧
  profit_per_unit selling_price / selling_price ≤ max_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_selling_price_optimal_l3538_353854


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_product_l3538_353808

theorem max_value_of_sin_cos_product (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.sin (x + α) * Real.cos (x + α)) →
  (∀ x, f x ≤ f 1) →
  ∃ k : ℤ, α = Real.pi / 4 + k * Real.pi / 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_product_l3538_353808


namespace NUMINAMATH_CALUDE_cos_difference_l3538_353855

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 3/2) : 
  Real.cos (A - B) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_l3538_353855


namespace NUMINAMATH_CALUDE_property_price_calculation_l3538_353876

/-- Calculate the total price of a property given the price per square foot and the sizes of the house and barn. -/
theorem property_price_calculation
  (price_per_sq_ft : ℕ)
  (house_size : ℕ)
  (barn_size : ℕ)
  (h1 : price_per_sq_ft = 98)
  (h2 : house_size = 2400)
  (h3 : barn_size = 1000) :
  price_per_sq_ft * (house_size + barn_size) = 333200 := by
  sorry

#eval 98 * (2400 + 1000) -- Sanity check

end NUMINAMATH_CALUDE_property_price_calculation_l3538_353876


namespace NUMINAMATH_CALUDE_tommys_tomato_profit_l3538_353829

/-- Represents the problem of calculating Tommy's profit from selling tomatoes --/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20  -- kg
  let num_crates : ℕ := 3
  let purchase_cost : ℕ := 330  -- $
  let selling_price : ℕ := 6    -- $ per kg
  let rotten_tomatoes : ℕ := 3  -- kg
  
  let total_tomatoes : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_tomatoes - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - purchase_cost
  
  profit = 12 := by sorry

end NUMINAMATH_CALUDE_tommys_tomato_profit_l3538_353829


namespace NUMINAMATH_CALUDE_binary_to_hex_conversion_l3538_353831

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : Nat) : String :=
  let rec aux (m : Nat) (acc : String) : String :=
    if m = 0 then
      if acc.isEmpty then "0" else acc
    else
      let digit := m % 16
      let hex_digit := if digit < 10 then 
        Char.toString (Char.ofNat (digit + 48))
      else
        Char.toString (Char.ofNat (digit + 55))
      aux (m / 16) (hex_digit ++ acc)
  aux n ""

/-- The binary number 1011101₂ -/
def binary_number : List Bool := [true, false, true, true, true, false, true]

theorem binary_to_hex_conversion :
  (binary_to_decimal binary_number = 93) ∧
  (decimal_to_hex 93 = "5D") := by
  sorry

end NUMINAMATH_CALUDE_binary_to_hex_conversion_l3538_353831


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_36_2310_l3538_353861

theorem gcd_lcm_sum_36_2310 : Nat.gcd 36 2310 + Nat.lcm 36 2310 = 13866 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_36_2310_l3538_353861


namespace NUMINAMATH_CALUDE_distribute_five_projects_three_teams_l3538_353842

/-- The number of ways to distribute n distinct projects among k teams,
    where each team must receive at least one project. -/
def distribute_projects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 projects among 3 teams results in 60 arrangements -/
theorem distribute_five_projects_three_teams :
  distribute_projects 5 3 = 60 := by sorry

end NUMINAMATH_CALUDE_distribute_five_projects_three_teams_l3538_353842


namespace NUMINAMATH_CALUDE_sum_of_obtuse_angles_l3538_353820

open Real

theorem sum_of_obtuse_angles (α β : Real) : 
  π < α ∧ α < 2*π → 
  π < β ∧ β < 2*π → 
  sin α = sqrt 5 / 5 → 
  cos β = -(3 * sqrt 10) / 10 → 
  α + β = 7 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_obtuse_angles_l3538_353820


namespace NUMINAMATH_CALUDE_joe_cars_count_l3538_353886

/-- Proves that Joe will have 62 cars after getting 12 more cars, given he initially had 50 cars. -/
theorem joe_cars_count (initial_cars : ℕ) (additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l3538_353886


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l3538_353862

theorem number_subtraction_problem (x : ℝ) : 0.60 * x - 40 = 50 ↔ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l3538_353862


namespace NUMINAMATH_CALUDE_cook_carrots_problem_l3538_353888

theorem cook_carrots_problem (initial_carrots : ℕ) 
  (fraction_used_before_lunch : ℚ) (carrots_not_used : ℕ) : 
  initial_carrots = 300 →
  fraction_used_before_lunch = 2/5 →
  carrots_not_used = 72 →
  (initial_carrots - fraction_used_before_lunch * initial_carrots - carrots_not_used) / 
  (initial_carrots - fraction_used_before_lunch * initial_carrots) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cook_carrots_problem_l3538_353888


namespace NUMINAMATH_CALUDE_divisor_problem_l3538_353899

theorem divisor_problem (d : ℕ+) (n : ℕ) (h1 : n % d = 3) (h2 : (2 * n) % d = 2) : d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3538_353899


namespace NUMINAMATH_CALUDE_stamp_collection_percentage_l3538_353897

theorem stamp_collection_percentage (total : ℕ) (chinese_percent : ℚ) (japanese_count : ℕ) : 
  total = 100 →
  chinese_percent = 35 / 100 →
  japanese_count = 45 →
  (total - (chinese_percent * total).floor - japanese_count) / total * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_percentage_l3538_353897


namespace NUMINAMATH_CALUDE_cone_height_l3538_353887

/-- A cone with volume 8000π cubic inches and a vertical cross section with a 90-degree vertex angle has a height of 20 × ∛6 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : ℝ) :
  V = 8000 * Real.pi ∧ θ = Real.pi / 2 →
  h = 20 * (6 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l3538_353887


namespace NUMINAMATH_CALUDE_amount_ratio_l3538_353856

/-- Given three amounts a, b, and c in rupees, prove that the ratio of a to b is 3:1 -/
theorem amount_ratio (a b c : ℕ) : 
  a + b + c = 645 →
  b = c + 25 →
  b = 134 →
  a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l3538_353856


namespace NUMINAMATH_CALUDE_exponential_equation_and_inequality_l3538_353882

/-- Given a > 0 and a ≠ 1, this theorem proves the conditions for equality and inequality
    between a^(3x+1) and a^(-2x) -/
theorem exponential_equation_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(3*x + 1) = a^(-2*x) ↔ x = 1/5) ∧
  (∀ x : ℝ, (a > 1 → (a^(3*x + 1) < a^(-2*x) ↔ x < 1/5)) ∧
            (a < 1 → (a^(3*x + 1) < a^(-2*x) ↔ x > 1/5))) :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_and_inequality_l3538_353882


namespace NUMINAMATH_CALUDE_matrix_vector_product_plus_vector_l3538_353865

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -5, 6]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![5; -2]
def w : Matrix (Fin 2) (Fin 1) ℝ := !![1; -1]

theorem matrix_vector_product_plus_vector :
  A * v + w = !![25; -38] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_product_plus_vector_l3538_353865


namespace NUMINAMATH_CALUDE_find_b_l3538_353826

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0}
def set2 (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3*p.1 + b}

-- State the theorem
theorem find_b : ∃ b : ℝ, set1 ⊂ set2 b → b = 2 := by sorry

end NUMINAMATH_CALUDE_find_b_l3538_353826


namespace NUMINAMATH_CALUDE_a_5_of_1034_is_5_l3538_353802

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Factorial base representation -/
def factorial_base_rep (n : ℕ) : List ℕ :=
  sorry

/-- The 5th coefficient in the factorial base representation -/
def a_5 (n : ℕ) : ℕ :=
  match factorial_base_rep n with
  | b₁ :: b₂ :: b₃ :: b₄ :: b₅ :: _ => b₅
  | _ => 0  -- Default case if the list is too short

/-- Theorem stating that the 5th coefficient of 1034 in factorial base is 5 -/
theorem a_5_of_1034_is_5 : a_5 1034 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_5_of_1034_is_5_l3538_353802


namespace NUMINAMATH_CALUDE_no_rain_probability_l3538_353893

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l3538_353893


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3538_353812

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3538_353812


namespace NUMINAMATH_CALUDE_remaining_wire_length_l3538_353845

-- Define the length of the iron wire
def wire_length (a b : ℝ) : ℝ := 5 * a + 4 * b

-- Define the perimeter of the rectangle
def rectangle_perimeter (a b : ℝ) : ℝ := 2 * (a + b)

-- Theorem statement
theorem remaining_wire_length (a b : ℝ) :
  wire_length a b - rectangle_perimeter a b = 3 * a + 2 * b := by
  sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l3538_353845


namespace NUMINAMATH_CALUDE_race_head_start_l3538_353880

/-- Proves that the head start in a race is equal to the difference in distances covered by two runners with different speeds in a given time. -/
theorem race_head_start (cristina_speed nicky_speed : ℝ) (race_time : ℝ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : cristina_speed = 4)
  (h3 : nicky_speed = 3)
  (h4 : race_time = 36) :
  cristina_speed * race_time - nicky_speed * race_time = 36 := by
  sorry

#check race_head_start

end NUMINAMATH_CALUDE_race_head_start_l3538_353880


namespace NUMINAMATH_CALUDE_bridget_bakery_profit_l3538_353819

/-- Calculates the profit for Bridget's bakery given the specified conditions. -/
def bakery_profit (total_loaves : ℕ) (morning_price afternoon_price late_price : ℚ)
  (operational_cost production_cost : ℚ) : ℚ :=
  let morning_sales := (2 : ℚ) / 5 * total_loaves
  let afternoon_sales := (1 : ℚ) / 2 * (total_loaves - morning_sales)
  let late_sales := (2 : ℚ) / 3 * (total_loaves - morning_sales - afternoon_sales)
  
  let revenue := morning_sales * morning_price + 
                 afternoon_sales * afternoon_price + 
                 late_sales * late_price
  
  let cost := (total_loaves : ℚ) * production_cost + operational_cost
  
  revenue - cost

/-- Theorem stating that under the given conditions, Bridget's bakery profit is $53. -/
theorem bridget_bakery_profit :
  bakery_profit 60 3 (3/2) 2 10 1 = 53 := by
  sorry

#eval bakery_profit 60 3 (3/2) 2 10 1

end NUMINAMATH_CALUDE_bridget_bakery_profit_l3538_353819


namespace NUMINAMATH_CALUDE_number_of_students_in_line_l3538_353822

/-- The number of students in a line with specific conditions -/
theorem number_of_students_in_line :
  ∀ (n : ℕ),
  (∃ (eunjung_position yoojung_position : ℕ),
    eunjung_position = 5 ∧
    yoojung_position = n ∧
    yoojung_position - eunjung_position = 9) →
  n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_line_l3538_353822


namespace NUMINAMATH_CALUDE_food_expense_percentage_l3538_353843

/-- Represents the percentage of income spent on various expenses --/
structure IncomeDistribution where
  food : ℝ
  education : ℝ
  rent : ℝ
  remaining : ℝ

/-- Proves that the percentage of income spent on food is 50% --/
theorem food_expense_percentage (d : IncomeDistribution) : d.food = 50 :=
  by
  have h1 : d.education = 15 := sorry
  have h2 : d.rent = 50 * (100 - d.food - d.education) / 100 := sorry
  have h3 : d.remaining = 17.5 := sorry
  have h4 : d.food + d.education + d.rent + d.remaining = 100 := sorry
  sorry

#check food_expense_percentage

end NUMINAMATH_CALUDE_food_expense_percentage_l3538_353843


namespace NUMINAMATH_CALUDE_eight_vases_needed_l3538_353827

/-- Represents the number of flowers of each type -/
structure FlowerCounts where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Represents the capacity of a vase for each flower type -/
structure VaseCapacity where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Calculates the minimum number of vases needed -/
def minVasesNeeded (flowers : FlowerCounts) (capacity : VaseCapacity) : ℕ :=
  sorry

/-- Theorem stating that 8 vases are needed for the given flower counts -/
theorem eight_vases_needed :
  let flowers := FlowerCounts.mk 20 15 5
  let capacity := VaseCapacity.mk 6 8 4
  minVasesNeeded flowers capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_vases_needed_l3538_353827


namespace NUMINAMATH_CALUDE_certain_number_proof_l3538_353866

theorem certain_number_proof (x : ℤ) : x + 34 - 53 = 28 ↔ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3538_353866


namespace NUMINAMATH_CALUDE_gallop_waddle_length_difference_l3538_353890

/-- The number of waddles Percy takes between consecutive lampposts -/
def percy_waddles : ℕ := 36

/-- The number of gallops Zelda takes between consecutive lampposts -/
def zelda_gallops : ℕ := 15

/-- The number of the last lamppost -/
def last_lamppost : ℕ := 31

/-- The distance in feet from the first to the last lamppost -/
def total_distance : ℕ := 3720

/-- The difference between Zelda's gallop length and Percy's waddle length -/
def gallop_waddle_difference : ℚ := 31 / 15

theorem gallop_waddle_length_difference :
  let percy_waddle_length : ℚ := total_distance / (percy_waddles * (last_lamppost - 1))
  let zelda_gallop_length : ℚ := total_distance / (zelda_gallops * (last_lamppost - 1))
  zelda_gallop_length - percy_waddle_length = gallop_waddle_difference := by
  sorry

end NUMINAMATH_CALUDE_gallop_waddle_length_difference_l3538_353890


namespace NUMINAMATH_CALUDE_expression_evaluation_l3538_353863

theorem expression_evaluation (x y z : ℝ) :
  (x + (y - z)) - ((x + z) - y) = 2*y - 2*z := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3538_353863


namespace NUMINAMATH_CALUDE_ten_integer_segments_l3538_353832

/-- Right triangle DEF with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def num_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- Our specific right triangle -/
def triangle : RightTriangle :=
  { de := 18, ef := 24 }

theorem ten_integer_segments : num_integer_segments triangle = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_integer_segments_l3538_353832


namespace NUMINAMATH_CALUDE_fraction_subtraction_problem_l3538_353805

theorem fraction_subtraction_problem : (1/2 : ℚ) + 5/6 - 2/3 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_problem_l3538_353805


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l3538_353817

theorem grocery_store_bottles : 
  157 + 126 + 87 + 52 + 64 = 486 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l3538_353817


namespace NUMINAMATH_CALUDE_solution_set_l3538_353834

/-- A function representing the quadratic expression inside the absolute value -/
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + 3*a + b

/-- The condition for the inequality to have exactly one solution -/
def has_unique_solution (a b : ℝ) : Prop :=
  ∃! x, |f a b x| ≤ 4

/-- The theorem stating the solution set -/
theorem solution_set :
  ∀ a : ℝ, has_unique_solution a (a^2 - 3*a + 4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l3538_353834


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l3538_353898

theorem onion_harvest_weight (initial_bags : ℕ) (trips : ℕ) (bag_weight : ℕ) : 
  initial_bags = 10 → trips = 20 → bag_weight = 50 →
  (initial_bags * ((2 ^ trips) - 1)) * bag_weight = 524287500 := by
  sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l3538_353898


namespace NUMINAMATH_CALUDE_fraction_difference_l3538_353841

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : 1 / a - 1 / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l3538_353841


namespace NUMINAMATH_CALUDE_dan_marbles_l3538_353881

theorem dan_marbles (initial_marbles given_marbles : ℕ) 
  (h1 : initial_marbles = 64)
  (h2 : given_marbles = 14) :
  initial_marbles - given_marbles = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l3538_353881


namespace NUMINAMATH_CALUDE_incorrect_number_correction_l3538_353844

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 46)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 50) :
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = actual_num - incorrect_num ∧ 
    actual_num = 65 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_number_correction_l3538_353844


namespace NUMINAMATH_CALUDE_student_trip_cost_is_1925_l3538_353833

/-- Calculates the amount each student needs for a trip given fundraising conditions -/
def student_trip_cost (num_students : ℕ) 
                      (misc_expenses : ℕ) 
                      (day1_raised : ℕ) 
                      (day2_raised : ℕ) 
                      (day3_raised : ℕ) 
                      (additional_days : ℕ) 
                      (additional_per_student : ℕ) : ℕ :=
  let first_three_days := day1_raised + day2_raised + day3_raised
  let next_days_total := (first_three_days / 2) * additional_days
  let total_raised := first_three_days + next_days_total
  let total_needed := total_raised + misc_expenses + (num_students * additional_per_student)
  total_needed / num_students

/-- Theorem stating that given the specific conditions, each student needs $1925 for the trip -/
theorem student_trip_cost_is_1925 : 
  student_trip_cost 6 3000 600 900 400 4 475 = 1925 := by
  sorry

#eval student_trip_cost 6 3000 600 900 400 4 475

end NUMINAMATH_CALUDE_student_trip_cost_is_1925_l3538_353833


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l3538_353818

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (hx : x > 0) : ScientificNotation :=
  sorry

theorem moon_radius_scientific_notation :
  toScientificNotation 1738000 (by norm_num) =
    ScientificNotation.mk 1.738 6 (by norm_num) (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l3538_353818


namespace NUMINAMATH_CALUDE_election_majority_l3538_353800

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 700 →
  winning_percentage = 84 / 100 →
  (winning_percentage * total_votes : ℚ).floor - 
  ((1 - winning_percentage) * total_votes : ℚ).floor = 476 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l3538_353800


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l3538_353884

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l3538_353884


namespace NUMINAMATH_CALUDE_number_equation_solution_l3538_353892

theorem number_equation_solution : 
  ∃ x : ℝ, (5020 - (1004 / x) = 4970) ∧ (x = 20.08) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3538_353892


namespace NUMINAMATH_CALUDE_impossible_sums_l3538_353860

-- Define the coin values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the set of possible coin values
def coin_values : Set ℕ := {penny, nickel, dime, quarter}

-- Define a function to check if a sum is possible with 5 coins
def is_possible_sum (sum : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = sum

-- Theorem statement
theorem impossible_sums : ¬(is_possible_sum 22) ∧ ¬(is_possible_sum 48) :=
sorry

end NUMINAMATH_CALUDE_impossible_sums_l3538_353860


namespace NUMINAMATH_CALUDE_unique_group_size_l3538_353809

theorem unique_group_size (n : ℕ) (k : ℕ) : 
  (∀ (i j : Fin n), i ≠ j → ∃! (call : Bool), call) →
  (∀ (subset : Finset (Fin n)), subset.card = n - 2 → 
    (subset.sum (λ i => (subset.filter (λ j => j ≠ i)).card) / 2) = 3^k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_group_size_l3538_353809


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3538_353836

theorem rectangle_perimeter (L W : ℝ) : 
  (L - 4 = W + 3) → 
  ((L - 4) * (W + 3) = L * W) → 
  (2 * L + 2 * W = 50) := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3538_353836


namespace NUMINAMATH_CALUDE_soccer_team_winning_percentage_l3538_353869

/-- Calculates the winning percentage of a soccer team -/
def winning_percentage (games_played : ℕ) (games_won : ℕ) : ℚ :=
  (games_won : ℚ) / (games_played : ℚ) * 100

/-- Theorem stating that a team with 280 games played and 182 wins has a 65% winning percentage -/
theorem soccer_team_winning_percentage :
  let games_played : ℕ := 280
  let games_won : ℕ := 182
  winning_percentage games_played games_won = 65 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_winning_percentage_l3538_353869


namespace NUMINAMATH_CALUDE_sara_marbles_l3538_353838

def marbles_problem (initial_marbles : ℕ) (remaining_marbles : ℕ) : Prop :=
  initial_marbles - remaining_marbles = 7

theorem sara_marbles : marbles_problem 10 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l3538_353838


namespace NUMINAMATH_CALUDE_height_diameter_ratio_l3538_353889

/-- A sphere inscribed in a right circular cylinder with equal diameters -/
structure InscribedSphere where
  r : ℝ  -- radius of the sphere and cylinder
  h : ℝ  -- height of the cylinder
  volume_ratio : h * r^2 = 8/3 * r^3  -- cylinder volume is twice sphere volume

/-- The ratio of cylinder height to sphere diameter is 4/3 -/
theorem height_diameter_ratio (s : InscribedSphere) : s.h / (2 * s.r) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_height_diameter_ratio_l3538_353889


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3538_353891

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) 
  (h1 : n = 20) 
  (h2 : total_games = 380) : 
  total_games = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3538_353891


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l3538_353807

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if three circles are congruent -/
def areCongruent (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- Checks if a circle is tangent to all sides of a rectangle -/
def isTangentToRectangle (c : Circle) (r : Rectangle) : Prop :=
  sorry

/-- Checks if a circle passes through two points -/
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_area_is_72 
  (ABCD : Rectangle) (P Q R : Point) (circleP circleQ circleR : Circle) :
  circleP.center = P →
  circleQ.center = Q →
  circleR.center = R →
  areCongruent circleP circleQ circleR →
  isTangentToRectangle circleP ABCD →
  isTangentToRectangle circleQ ABCD →
  isTangentToRectangle circleR ABCD →
  circleQ.radius = 3 →
  passesThrough circleQ P R →
  rectangleArea ABCD = 72 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l3538_353807


namespace NUMINAMATH_CALUDE_condition_analysis_l3538_353878

theorem condition_analysis (x y : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 0 → (x - 1) * (y - 2) = 0) ∧ 
  (∃ x y : ℝ, (x - 1) * (y - 2) = 0 ∧ (x - 1)^2 + (y - 2)^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l3538_353878


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3538_353868

theorem product_expansion_sum (a b c d e : ℝ) :
  (∀ x : ℝ, (5*x^3 - 3*x^2 + x - 8)*(8 - 3*x) = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  16*a + 8*b + 4*c + 2*d + e = 44 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3538_353868


namespace NUMINAMATH_CALUDE_trivia_game_base_points_l3538_353803

/-- Calculates the base points per round for a trivia game -/
theorem trivia_game_base_points
  (total_rounds : ℕ)
  (total_score : ℕ)
  (bonus_points : ℕ)
  (penalty_points : ℕ)
  (h1 : total_rounds = 5)
  (h2 : total_score = 370)
  (h3 : bonus_points = 50)
  (h4 : penalty_points = 30) :
  (total_score + bonus_points - penalty_points) / total_rounds = 78 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_base_points_l3538_353803


namespace NUMINAMATH_CALUDE_calculator_game_result_l3538_353885

/-- The number of participants in the game -/
def num_participants : ℕ := 60

/-- The operation applied to the first calculator -/
def op1 (n : ℕ) (x : ℤ) : ℤ := x ^ 3 ^ n

/-- The operation applied to the second calculator -/
def op2 (n : ℕ) (x : ℤ) : ℤ := x ^ (2 ^ n)

/-- The operation applied to the third calculator -/
def op3 (n : ℕ) (x : ℤ) : ℤ := (-1) ^ n * x

/-- The final sum of the numbers on the calculators after one complete round -/
def final_sum : ℤ := op1 num_participants 2 + op2 num_participants 0 + op3 num_participants (-1)

theorem calculator_game_result : final_sum = 2 ^ (3 ^ 60) + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_result_l3538_353885


namespace NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l3538_353821

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ (x, y) = P) ∧
  (∀ x y, 3*x - 4*y + 8 = 0 ↔ (∃ t, (x, y) = (t*3 + P.1, t*4 + P.2))) ∧
  (∀ x y, 4*x + 3*y - 6 = 0 ↔ (∃ t, (x, y) = (t*4 + P.1, -t*3 + P.2))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l3538_353821


namespace NUMINAMATH_CALUDE_hairstylist_normal_haircut_price_l3538_353883

theorem hairstylist_normal_haircut_price :
  let normal_price : ℝ := x
  let special_price : ℝ := 6
  let trendy_price : ℝ := 8
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let weekly_earnings : ℝ := 413
  (normal_price * (normal_per_day * days_per_week : ℝ) +
   special_price * (special_per_day * days_per_week : ℝ) +
   trendy_price * (trendy_per_day * days_per_week : ℝ) = weekly_earnings) →
  normal_price = 5 := by
sorry

end NUMINAMATH_CALUDE_hairstylist_normal_haircut_price_l3538_353883


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3538_353870

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.log (abs x)

theorem solution_set_of_inequality :
  {x : ℝ | f (x + 1) > f (2 * x - 1)} = {x : ℝ | 0 < x ∧ x < 1/2 ∨ 1/2 < x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3538_353870


namespace NUMINAMATH_CALUDE_no_natural_solution_l3538_353873

theorem no_natural_solution :
  ¬∃ (a b c : ℕ), (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l3538_353873
