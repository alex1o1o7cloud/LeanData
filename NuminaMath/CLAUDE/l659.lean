import Mathlib

namespace NUMINAMATH_CALUDE_vikki_take_home_pay_l659_65987

def weekly_pay_calculation (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  total_earnings - total_deductions

theorem vikki_take_home_pay :
  weekly_pay_calculation 42 10 (20/100) (5/100) 5 = 310 := by
  sorry

end NUMINAMATH_CALUDE_vikki_take_home_pay_l659_65987


namespace NUMINAMATH_CALUDE_sum_equals_220_l659_65966

theorem sum_equals_220 : 145 + 33 + 29 + 13 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_220_l659_65966


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l659_65978

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 84) = 24 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l659_65978


namespace NUMINAMATH_CALUDE_value_of_a_l659_65917

theorem value_of_a : ∀ a : ℕ, 
  (a * (9^3) = 3 * (15^5)) → 
  (a = 5^5) → 
  (a = 3125) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l659_65917


namespace NUMINAMATH_CALUDE_equation_solution_l659_65965

theorem equation_solution : 
  ∃ (x : ℝ), (4 * x - 5) / (5 * x - 10) = 3 / 4 ∧ x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l659_65965


namespace NUMINAMATH_CALUDE_abc_sum_l659_65992

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c + 10 = 51)
  (eq2 : b * c + a + 10 = 51)
  (eq3 : a * c + b + 10 = 51) :
  a + b + c = 41 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l659_65992


namespace NUMINAMATH_CALUDE_exponential_inequality_and_unique_a_l659_65927

open Real

theorem exponential_inequality_and_unique_a :
  (∀ x > -1, exp x > (x + 1)^2 / 2) ∧
  (∃! a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1) ∧
  (∃ a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1 ∧ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_and_unique_a_l659_65927


namespace NUMINAMATH_CALUDE_max_tank_volume_l659_65961

/-- A rectangular parallelepiped tank with the given properties -/
structure Tank where
  a : Real  -- length of the base
  b : Real  -- width of the base
  h : Real  -- height of the tank
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  side_area_condition : a * h ≥ a * b ∧ b * h ≥ a * b

/-- The theorem stating the maximum volume of the tank -/
theorem max_tank_volume (tank : Tank) (h_val : tank.h = 1.5) :
  (∀ t : Tank, t.h = 1.5 → t.a * t.b * t.h ≤ tank.a * tank.b * tank.h) →
  tank.a * tank.b * tank.h = 3.375 := by
  sorry

end NUMINAMATH_CALUDE_max_tank_volume_l659_65961


namespace NUMINAMATH_CALUDE_common_area_of_30_60_90_triangles_l659_65920

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ

/-- The area common to two congruent 30-60-90 triangles with coinciding shorter legs -/
def commonArea (t : Triangle30_60_90) : ℝ := t.shortLeg ^ 2

/-- Theorem: The area common to two congruent 30-60-90 triangles with hypotenuse 16 and coinciding shorter legs is 64 square units -/
theorem common_area_of_30_60_90_triangles :
  ∀ t : Triangle30_60_90,
  t.hypotenuse = 16 →
  t.shortLeg = t.hypotenuse / 2 →
  t.longLeg = t.shortLeg * Real.sqrt 3 →
  commonArea t = 64 := by
  sorry

end NUMINAMATH_CALUDE_common_area_of_30_60_90_triangles_l659_65920


namespace NUMINAMATH_CALUDE_total_books_count_l659_65934

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l659_65934


namespace NUMINAMATH_CALUDE_estate_value_l659_65903

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter1 : ℝ
  daughter2 : ℝ
  son : ℝ
  husband : ℝ
  gardener : ℝ

/-- The estate distribution satisfies the given conditions --/
def validDistribution (e : EstateDistribution) : Prop :=
  -- The two daughters and son receive 3/5 of the estate
  e.daughter1 + e.daughter2 + e.son = 3/5 * e.total
  -- The daughters and son share in the ratio of 5:3:2
  ∧ e.daughter1 = 5/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.daughter2 = 3/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.son = 2/10 * (e.daughter1 + e.daughter2 + e.son)
  -- The husband gets three times as much as the son
  ∧ e.husband = 3 * e.son
  -- The gardener receives $600
  ∧ e.gardener = 600
  -- The total estate is the sum of all shares
  ∧ e.total = e.daughter1 + e.daughter2 + e.son + e.husband + e.gardener

/-- The estate value is $15000 --/
theorem estate_value (e : EstateDistribution) (h : validDistribution e) : e.total = 15000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l659_65903


namespace NUMINAMATH_CALUDE_bird_multiple_l659_65906

theorem bird_multiple : ∃ x : ℝ, x * 20 + 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bird_multiple_l659_65906


namespace NUMINAMATH_CALUDE_range_of_a_l659_65919

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ a ∧ x < 2 ↔ x < 2) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l659_65919


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l659_65948

theorem radical_conjugate_sum_product (a b : ℝ) :
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 →
  a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l659_65948


namespace NUMINAMATH_CALUDE_min_value_of_a_l659_65984

theorem min_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∃ x, f x ≤ 0) →
  (∀ x, f x = Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x) →
  a ≥ 1 - 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l659_65984


namespace NUMINAMATH_CALUDE_sophies_purchase_amount_l659_65910

/-- Calculates the total amount Sophie spends on her purchase --/
def sophies_purchase (cupcake_price : ℚ) (doughnut_price : ℚ) (pie_price : ℚ)
  (cookie_price : ℚ) (chocolate_price : ℚ) (soda_price : ℚ) (gum_price : ℚ)
  (chips_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let subtotal := 5 * cupcake_price + 6 * doughnut_price + 4 * pie_price +
    15 * cookie_price + 8 * chocolate_price + 12 * soda_price +
    3 * gum_price + 10 * chips_price
  let discounted_total := subtotal * (1 - discount_rate)
  let tax_amount := discounted_total * tax_rate
  discounted_total + tax_amount

/-- Theorem stating that Sophie's total purchase amount is $69.45 --/
theorem sophies_purchase_amount :
  sophies_purchase 2 1 2 (6/10) (3/2) (6/5) (4/5) (11/10) (1/10) (6/100) = (6945/100) := by
  sorry

end NUMINAMATH_CALUDE_sophies_purchase_amount_l659_65910


namespace NUMINAMATH_CALUDE_H2O_formation_l659_65947

-- Define the molecules and their molar quantities
def HCl_moles : ℚ := 2
def CaCO3_moles : ℚ := 1

-- Define the balanced equation coefficients
def HCl_coeff : ℚ := 2
def CaCO3_coeff : ℚ := 1
def H2O_coeff : ℚ := 1

-- Define the function to calculate the amount of H2O formed
def H2O_formed (HCl : ℚ) (CaCO3 : ℚ) : ℚ :=
  min (HCl / HCl_coeff) (CaCO3 / CaCO3_coeff) * H2O_coeff

-- State the theorem
theorem H2O_formation :
  H2O_formed HCl_moles CaCO3_moles = 1 := by
  sorry

end NUMINAMATH_CALUDE_H2O_formation_l659_65947


namespace NUMINAMATH_CALUDE_arithmetic_events_classification_l659_65931

/-- Represents the sign of a number -/
inductive Sign
| Positive
| Negative

/-- Represents the result of an arithmetic operation -/
inductive Result
| Positive
| Negative

/-- Represents an arithmetic event -/
structure ArithmeticEvent :=
  (operation : String)
  (sign1 : Sign)
  (sign2 : Sign)
  (result : Result)

/-- Defines the four events described in the problem -/
def events : List ArithmeticEvent :=
  [ ⟨"Addition", Sign.Positive, Sign.Negative, Result.Negative⟩
  , ⟨"Subtraction", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Multiplication", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Division", Sign.Positive, Sign.Negative, Result.Negative⟩ ]

/-- Predicate to determine if an event is certain -/
def isCertain (e : ArithmeticEvent) : Prop :=
  e.operation = "Division" ∧ 
  e.sign1 ≠ e.sign2 ∧ 
  e.result = Result.Negative

/-- Predicate to determine if an event is random -/
def isRandom (e : ArithmeticEvent) : Prop :=
  (e.operation = "Addition" ∨ e.operation = "Subtraction") ∧
  e.sign1 ≠ e.sign2

theorem arithmetic_events_classification :
  ∃ (certain : ArithmeticEvent) (random1 random2 : ArithmeticEvent),
    certain ∈ events ∧
    random1 ∈ events ∧
    random2 ∈ events ∧
    isCertain certain ∧
    isRandom random1 ∧
    isRandom random2 ∧
    random1 ≠ random2 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_events_classification_l659_65931


namespace NUMINAMATH_CALUDE_set_containment_implies_a_bound_l659_65963

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- State the theorem
theorem set_containment_implies_a_bound (a : ℝ) :
  A ⊆ B a → a ≤ -2 := by
  sorry

-- The range of a is implicitly (-∞, -2] because a ≤ -2

end NUMINAMATH_CALUDE_set_containment_implies_a_bound_l659_65963


namespace NUMINAMATH_CALUDE_forest_width_is_correct_l659_65942

/-- The width of a forest in miles -/
def forest_width : ℝ := 6

/-- The length of the forest in miles -/
def forest_length : ℝ := 4

/-- The number of trees per square mile -/
def trees_per_square_mile : ℕ := 600

/-- The number of trees one logger can cut per day -/
def trees_per_logger_per_day : ℕ := 6

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of loggers working -/
def number_of_loggers : ℕ := 8

/-- The number of months it takes to cut down all trees -/
def months_to_cut_all_trees : ℕ := 10

theorem forest_width_is_correct : 
  forest_width * forest_length * trees_per_square_mile = 
  (trees_per_logger_per_day * days_per_month * number_of_loggers * months_to_cut_all_trees : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_forest_width_is_correct_l659_65942


namespace NUMINAMATH_CALUDE_rectangle_area_l659_65907

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = y^2 → 3 * w^2 = (3 * y^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l659_65907


namespace NUMINAMATH_CALUDE_units_digit_of_7_19_l659_65940

theorem units_digit_of_7_19 : (7^19) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_19_l659_65940


namespace NUMINAMATH_CALUDE_pet_store_dogs_l659_65999

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 28 := by
sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l659_65999


namespace NUMINAMATH_CALUDE_real_roots_quadratic_l659_65951

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 16*k = 0) ↔ k ≤ 0 ∨ k ≥ 64 := by
sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_l659_65951


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l659_65980

theorem cubic_root_equation_solution :
  ∃! x : ℝ, 2.61 * (9 - Real.sqrt (x + 1))^(1/3) + (7 + Real.sqrt (x + 1))^(1/3) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l659_65980


namespace NUMINAMATH_CALUDE_circle_division_sum_integer_l659_65973

theorem circle_division_sum_integer :
  ∃ (a b c d e : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / d : ℚ) + (d / e : ℚ) + (e / a : ℚ) = n :=
sorry

end NUMINAMATH_CALUDE_circle_division_sum_integer_l659_65973


namespace NUMINAMATH_CALUDE_park_outer_boundary_diameter_l659_65923

/-- Represents a circular park with concentric features -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  grassy_area_width : ℝ
  walking_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.grassy_area_width + park.walking_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 52 feet -/
theorem park_outer_boundary_diameter :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    grassy_area_width := 4,
    walking_path_width := 6
  }
  outer_boundary_diameter park = 52 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_boundary_diameter_l659_65923


namespace NUMINAMATH_CALUDE_product_of_constrained_values_l659_65968

theorem product_of_constrained_values (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_of_constrained_values_l659_65968


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l659_65950

/-- The slope of a chord on an ellipse, given its midpoint -/
theorem ellipse_chord_slope (x₁ x₂ y₁ y₂ : ℝ) :
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₁ - y₂) / (x₁ - x₂) = -9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l659_65950


namespace NUMINAMATH_CALUDE_area_36_implies_a_plus_b_6_l659_65990

/-- A quadrilateral with vertices defined by a positive integer a -/
structure Quadrilateral (a : ℕ+) where
  P : ℝ × ℝ := (a, a)
  Q : ℝ × ℝ := (a, -a)
  R : ℝ × ℝ := (-a, -a)
  S : ℝ × ℝ := (-a, a)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral a) : ℝ := sorry

/-- Theorem: If the area of quadrilateral PQRS is 36, then a+b = 6 -/
theorem area_36_implies_a_plus_b_6 (a b : ℕ+) (q : Quadrilateral a) :
  area q = 36 → a + b = 6 := by sorry

end NUMINAMATH_CALUDE_area_36_implies_a_plus_b_6_l659_65990


namespace NUMINAMATH_CALUDE_largest_area_polygon_E_l659_65998

/-- Represents a polygon composed of unit squares and right triangles -/
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

/-- Calculates the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

/-- The given polygons -/
def polygonA : Polygon := ⟨3, 2⟩
def polygonB : Polygon := ⟨6, 0⟩
def polygonC : Polygon := ⟨4, 3⟩
def polygonD : Polygon := ⟨5, 1⟩
def polygonE : Polygon := ⟨7, 0⟩

theorem largest_area_polygon_E :
  ∀ p ∈ [polygonA, polygonB, polygonC, polygonD, polygonE],
    areaOfPolygon p ≤ areaOfPolygon polygonE :=
by sorry

end NUMINAMATH_CALUDE_largest_area_polygon_E_l659_65998


namespace NUMINAMATH_CALUDE_equation_solution_l659_65960

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l659_65960


namespace NUMINAMATH_CALUDE_Z_in_second_quadrant_l659_65901

-- Define the complex number Z
def Z : ℂ := Complex.I * (1 + Complex.I)

-- Theorem statement
theorem Z_in_second_quadrant : 
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_Z_in_second_quadrant_l659_65901


namespace NUMINAMATH_CALUDE_joan_gave_sam_seashells_l659_65941

/-- The number of seashells Joan gave to Sam -/
def seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) : ℕ :=
  initial_seashells - remaining_seashells

/-- Theorem: Joan gave Sam 43 seashells -/
theorem joan_gave_sam_seashells (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70)
  (h2 : remaining_seashells = 27) :
  seashells_given_to_sam initial_seashells remaining_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_sam_seashells_l659_65941


namespace NUMINAMATH_CALUDE_even_function_property_l659_65994

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end NUMINAMATH_CALUDE_even_function_property_l659_65994


namespace NUMINAMATH_CALUDE_number_pyramid_result_l659_65937

theorem number_pyramid_result : 123456 * 9 + 7 = 1111111 := by
  sorry

end NUMINAMATH_CALUDE_number_pyramid_result_l659_65937


namespace NUMINAMATH_CALUDE_percent_profit_problem_l659_65954

/-- Given that the cost price of 60 articles equals the selling price of 50 articles,
    prove that the percent profit is 20%. -/
theorem percent_profit_problem (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_problem_l659_65954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l659_65959

/-- An arithmetic sequence with the given properties has a common difference of 1/3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (h1 : a 3 + a 5 = 2)  -- First condition
  (h2 : a 7 + a 10 + a 13 = 9)  -- Second condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : ∃ d : ℚ, d = 1/3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l659_65959


namespace NUMINAMATH_CALUDE_symmetric_lines_b_value_l659_65953

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two lines are symmetric with respect to a given point -/
def are_symmetric (l1 l2 : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), l1.a * x + l1.b * y + l1.c = 0 →
    ∃ (x' y' : ℝ), l2.a * x' + l2.b * y' + l2.c = 0 ∧
      p.x = (x + x') / 2 ∧ p.y = (y + y') / 2

/-- The main theorem stating that given the conditions, b must equal 2 -/
theorem symmetric_lines_b_value :
  ∀ (a b : ℝ),
  let l1 : Line := ⟨1, 2, -3⟩
  let l2 : Line := ⟨a, 4, b⟩
  let p : Point := ⟨1, 0⟩
  are_symmetric l1 l2 p → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_b_value_l659_65953


namespace NUMINAMATH_CALUDE_magical_tree_properties_l659_65991

structure FruitTree :=
  (bananas : Nat)
  (oranges : Nat)

inductive PickAction
  | PickOneBanana
  | PickOneOrange
  | PickTwoBananas
  | PickTwoOranges
  | PickBananaAndOrange

def applyAction (tree : FruitTree) (action : PickAction) : FruitTree :=
  match action with
  | PickAction.PickOneBanana => tree
  | PickAction.PickOneOrange => tree
  | PickAction.PickTwoBananas => 
      if tree.bananas ≥ 2 then { bananas := tree.bananas - 2, oranges := tree.oranges + 1 }
      else tree
  | PickAction.PickTwoOranges => 
      if tree.oranges ≥ 2 then { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree
  | PickAction.PickBananaAndOrange =>
      if tree.bananas ≥ 1 && tree.oranges ≥ 1 then
        { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree

def initialTree : FruitTree := { bananas := 15, oranges := 20 }

theorem magical_tree_properties :
  -- 1. It's possible to reach a state with exactly one fruit
  (∃ (actions : List PickAction), (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1) ∧
  -- 2. If there's only one fruit left, it must be a banana
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1 →
    (actions.foldl applyAction initialTree).bananas = 1) ∧
  -- 3. It's impossible to reach a state with no fruits
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_magical_tree_properties_l659_65991


namespace NUMINAMATH_CALUDE_k_range_k_values_circle_origin_l659_65921

-- Define the line and hyperbola equations
def line (k x : ℝ) : ℝ := k * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ < 0 ∧ x₂ > 0 ∧
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = line k x₁ ∧ y₂ = line k x₂

-- Theorem for the range of k
theorem k_range :
  ∀ k : ℝ, intersection_points k ↔ -Real.sqrt 3 < k ∧ k < Real.sqrt 3 :=
sorry

-- Define the condition for the circle passing through the origin
def circle_through_origin (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points k ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the values of k when the circle passes through the origin
theorem k_values_circle_origin :
  ∀ k : ℝ, circle_through_origin k ↔ k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_k_range_k_values_circle_origin_l659_65921


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l659_65974

-- Define the number of black and yellow balls
def num_black_balls : ℕ := 4
def num_yellow_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := num_black_balls + num_yellow_balls

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l659_65974


namespace NUMINAMATH_CALUDE_problem_solution_l659_65938

theorem problem_solution (a b c : ℚ) 
  (sum_condition : a + b + c = 200)
  (equal_condition : a + 10 = b - 10 ∧ b - 10 = 10 * c) : 
  b = 2210 / 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l659_65938


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l659_65936

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles is 180°
  b = 4 * a →              -- ratio condition
  c = 7 * a →              -- ratio condition
  a = 15 ∧ b = 60 ∧ c = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l659_65936


namespace NUMINAMATH_CALUDE_extreme_value_and_tangent_line_l659_65985

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem extreme_value_and_tangent_line 
  (a : ℝ) 
  (h1 : f' a 3 = 0) -- f(x) has an extreme value at x = 3
  (h2 : f a 1 = 16) -- Point A(1,16) is on f(x)
  : 
  (∀ x, f a x = 2 * x^3 - 12 * x^2 + 18 * x + 8) ∧ 
  (f' a 1 = 0) := by 
  sorry

#check extreme_value_and_tangent_line

end NUMINAMATH_CALUDE_extreme_value_and_tangent_line_l659_65985


namespace NUMINAMATH_CALUDE_power_three_inverse_exponent_l659_65933

theorem power_three_inverse_exponent (x y : ℕ) : 
  (2^x : ℕ) ∣ 900 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 900) ∧ 
  (5^y : ℕ) ∣ 900 ∧ 
  ∀ l > y, ¬((5^l : ℕ) ∣ 900) → 
  (1/3 : ℚ)^(2*(y - x)) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_three_inverse_exponent_l659_65933


namespace NUMINAMATH_CALUDE_zoo_trip_attendance_l659_65944

/-- The number of buses available for the zoo trip -/
def num_buses : ℕ := 3

/-- The number of people that would go in each bus if evenly distributed -/
def people_per_bus : ℕ := 73

/-- The total number of people going to the zoo -/
def total_people : ℕ := num_buses * people_per_bus

theorem zoo_trip_attendance : total_people = 219 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_attendance_l659_65944


namespace NUMINAMATH_CALUDE_product_xy_in_parallelogram_l659_65946

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram is 18√3 -/
theorem product_xy_in_parallelogram (p : Parallelogram) 
    (h1 : p.EF = 42)
    (h2 : p.FG = fun y ↦ 4 * y^2 + 1)
    (h3 : p.GH = fun x ↦ 3 * x + 6)
    (h4 : p.HE = 28) :
    ∃ x y, p.GH x = p.EF ∧ p.FG y = p.HE ∧ x * y = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_in_parallelogram_l659_65946


namespace NUMINAMATH_CALUDE_total_triangles_is_28_l659_65967

/-- Represents a triangular arrangement of equilateral triangles -/
structure TriangularArrangement where
  rows : ℕ
  -- Each row n contains n unit triangles
  unit_triangles_in_row : (n : ℕ) → n ≤ rows → ℕ
  unit_triangles_in_row_eq : ∀ n h, unit_triangles_in_row n h = n

/-- Counts the total number of equilateral triangles in the arrangement -/
def count_all_triangles (arrangement : TriangularArrangement) : ℕ :=
  sorry

/-- The main theorem: In a triangular arrangement with 6 rows, 
    the total number of equilateral triangles is 28 -/
theorem total_triangles_is_28 :
  ∀ (arrangement : TriangularArrangement),
  arrangement.rows = 6 →
  count_all_triangles arrangement = 28 :=
sorry

end NUMINAMATH_CALUDE_total_triangles_is_28_l659_65967


namespace NUMINAMATH_CALUDE_power_multiplication_l659_65911

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l659_65911


namespace NUMINAMATH_CALUDE_impossible_arrangement_l659_65962

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of digits used in the grid -/
def Digits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Checks if a row contains three different digits -/
def row_valid (g : Grid) (i : Fin 3) : Prop :=
  (Finset.card {g i 0, g i 1, g i 2}) = 3

/-- Checks if a column contains three different digits -/
def col_valid (g : Grid) (j : Fin 3) : Prop :=
  (Finset.card {g 0 j, g 1 j, g 2 j}) = 3

/-- Checks if the main diagonal contains three different digits -/
def main_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 0, g 1 1, g 2 2}) = 3

/-- Checks if the anti-diagonal contains three different digits -/
def anti_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 2, g 1 1, g 2 0}) = 3

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 3, row_valid g i) ∧
  (∀ j : Fin 3, col_valid g j) ∧
  main_diag_valid g ∧
  anti_diag_valid g

theorem impossible_arrangement : ¬∃ (g : Grid), valid_grid g := by
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l659_65962


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_150_l659_65932

theorem remainder_98_power_50_mod_150 : 98^50 ≡ 74 [ZMOD 150] := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_150_l659_65932


namespace NUMINAMATH_CALUDE_f_monotonicity_and_range_l659_65970

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 5

theorem f_monotonicity_and_range :
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ z ∈ Set.Icc 0 1, 5 ≤ f z ∧ f z ≤ Real.exp 1 + 5) ∧
  (f 0 = 5) ∧
  (f 1 = Real.exp 1 + 5) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_range_l659_65970


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l659_65945

def A : Set Int := {x | |x| < 3}
def B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l659_65945


namespace NUMINAMATH_CALUDE_floor_equation_solution_l659_65995

theorem floor_equation_solution (x : ℚ) : 
  (⌊20 * x + 23⌋ = 20 + 23 * x) ↔ 
  (∃ k : ℕ, k ≤ 7 ∧ x = (23 - k : ℚ) / 23) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l659_65995


namespace NUMINAMATH_CALUDE_school_event_water_drinkers_l659_65979

/-- Proves that 60 students chose water given the conditions of the school event -/
theorem school_event_water_drinkers (total : ℕ) (juice_percent soda_percent : ℚ) 
  (soda_count : ℕ) : 
  juice_percent = 1/2 →
  soda_percent = 3/10 →
  soda_count = 90 →
  total = soda_count / soda_percent →
  (1 - juice_percent - soda_percent) * total = 60 :=
by
  sorry

#check school_event_water_drinkers

end NUMINAMATH_CALUDE_school_event_water_drinkers_l659_65979


namespace NUMINAMATH_CALUDE_cube_difference_l659_65909

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : 
  a^3 - b^3 = 124 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l659_65909


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l659_65949

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l659_65949


namespace NUMINAMATH_CALUDE_trapezoid_in_square_l659_65958

theorem trapezoid_in_square (s : ℝ) (x : ℝ) : 
  s = 2 → -- Side length of the square
  (1/3) * s^2 = (1/2) * (s + x) * (s/2) → -- Area of trapezoid is 1/3 of square's area
  x = 2/3 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_in_square_l659_65958


namespace NUMINAMATH_CALUDE_quadrangular_prism_angles_l659_65902

/-- A quadrangular prism with specific geometric properties -/
structure QuadrangularPrism where
  -- Base angles
  angleASB : ℝ
  angleDCS : ℝ
  -- Dihedral angle between SAD and SBC
  dihedralAngle : ℝ

/-- The theorem stating the possible angle measures in the quadrangular prism -/
theorem quadrangular_prism_angles (prism : QuadrangularPrism)
  (h1 : prism.angleASB = π/6)  -- 30°
  (h2 : prism.angleDCS = π/4)  -- 45°
  (h3 : prism.dihedralAngle = π/3)  -- 60°
  : (∃ (angleBSC angleASD : ℝ),
      (angleBSC = π/2 ∧ angleASD = π - Real.arccos (Real.sqrt 3 / 2)) ∨
      (angleBSC = Real.arccos (2 * Real.sqrt 2 / 3) ∧ 
       angleASD = Real.arccos (5 * Real.sqrt 3 / 9))) := by
  sorry


end NUMINAMATH_CALUDE_quadrangular_prism_angles_l659_65902


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l659_65988

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l659_65988


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l659_65922

theorem largest_non_representable_integer : 
  (∀ n > 97, ∃ a b : ℕ, n = 8 * a + 15 * b) ∧ 
  (¬ ∃ a b : ℕ, 97 = 8 * a + 15 * b) := by
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l659_65922


namespace NUMINAMATH_CALUDE_difference_of_half_and_third_l659_65929

theorem difference_of_half_and_third : 1/2 - 1/3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_half_and_third_l659_65929


namespace NUMINAMATH_CALUDE_fraction_undefined_l659_65955

theorem fraction_undefined (x : ℝ) : (3 * x - 1) / (x + 3) = 0 / 0 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_l659_65955


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l659_65935

/-- The length of the tangent segment from the origin to the circle passing through 
    the points (2,3), (4,6), and (3,9) is 3√5. -/
theorem tangent_length_to_circle : 
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (4, 6)
  let C : ℝ × ℝ := (3, 9)
  let O : ℝ × ℝ := (0, 0)
  ∃ (circle : Set (ℝ × ℝ)) (T : ℝ × ℝ),
    A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧
    T ∈ circle ∧
    (∀ P ∈ circle, dist O P ≥ dist O T) ∧
    dist O T = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l659_65935


namespace NUMINAMATH_CALUDE_sin_power_sum_l659_65993

theorem sin_power_sum (φ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < φ) (h2 : φ < π / 2) 
  (h3 : x + 1 / x = 2 * Real.sin φ) 
  (h4 : n > 0) : 
  x^n + 1 / x^n = 2 * Real.sin (n * φ) := by
  sorry

end NUMINAMATH_CALUDE_sin_power_sum_l659_65993


namespace NUMINAMATH_CALUDE_unicity_of_inverse_l659_65916

variable {G : Type*} [Group G]

theorem unicity_of_inverse (x y z : G) (h1 : 1 = x * y) (h2 : 1 = z * x) :
  y = z ∧ y = x⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_unicity_of_inverse_l659_65916


namespace NUMINAMATH_CALUDE_player_arrangement_count_l659_65982

/-- The number of ways to arrange players from three teams -/
def arrange_players (n : ℕ) : ℕ :=
  (n.factorial) * (n.factorial) * (n.factorial) * (n.factorial)

/-- Theorem: The number of ways to arrange 9 players from 3 teams is 1296 -/
theorem player_arrangement_count :
  arrange_players 3 = 1296 := by
  sorry

#eval arrange_players 3

end NUMINAMATH_CALUDE_player_arrangement_count_l659_65982


namespace NUMINAMATH_CALUDE_chord_length_squared_l659_65908

/-- Two circles with given properties and a line through their intersection point --/
structure TwoCirclesWithLine where
  /-- Radius of the first circle --/
  r₁ : ℝ
  /-- Radius of the second circle --/
  r₂ : ℝ
  /-- Distance between the centers of the circles --/
  d : ℝ
  /-- Length of the chord QP (equal to PR) --/
  x : ℝ

/-- Theorem stating the square of the chord length in the given configuration --/
theorem chord_length_squared (c : TwoCirclesWithLine)
  (h₁ : c.r₁ = 5)
  (h₂ : c.r₂ = 10)
  (h₃ : c.d = 16)
  (h₄ : c.x > 0) :
  c.x^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l659_65908


namespace NUMINAMATH_CALUDE_eleanor_cookies_l659_65997

theorem eleanor_cookies (N : ℕ) : 
  N % 13 = 5 → N % 8 = 3 → N < 150 → N = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_eleanor_cookies_l659_65997


namespace NUMINAMATH_CALUDE_lottery_winner_prize_l659_65900

def lottery_prize (num_tickets : ℕ) (first_ticket_price : ℕ) (price_increase : ℕ) (profit : ℕ) : ℕ :=
  let total_revenue := (num_tickets * (2 * first_ticket_price + (num_tickets - 1) * price_increase)) / 2
  total_revenue - profit

theorem lottery_winner_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winner_prize_l659_65900


namespace NUMINAMATH_CALUDE_triangle_side_sum_l659_65928

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (3 * b - a) * Real.cos C = c * Real.cos A →
  c^2 = a * b →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 2 →
  a + b = Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l659_65928


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l659_65956

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / (b ^ 2) → x + y = 2 * c → 
  min x y = (2 * a * c) / (a + b ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l659_65956


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l659_65957

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2006
  ∃ k : ℕ, k = 34 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n ∧
    n = 2 * 17 * 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l659_65957


namespace NUMINAMATH_CALUDE_line_l1_equation_range_of_b_l659_65930

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 4 = 0

-- Define the midpoint P of the chord intercepted by line l1
def midpoint_P : ℝ × ℝ := (5, 3)

-- Define line l2
def line_l2 (x y b : ℝ) : Prop := x + y + b = 0

-- Theorem for the equation of line l1
theorem line_l1_equation : 
  ∃ (l1 : ℝ → ℝ → Prop), 
  (∀ x y, l1 x y ↔ 2*x + y - 13 = 0) ∧ 
  (∀ x y, l1 x y → circle_C x y) ∧
  (l1 (midpoint_P.1) (midpoint_P.2)) := 
sorry

-- Theorem for the range of b
theorem range_of_b :
  ∀ b : ℝ, (∃ x y, circle_C x y ∧ line_l2 x y b) ↔ 
  (-3 * Real.sqrt 2 - 5 < b ∧ b < 3 * Real.sqrt 2 - 5) := 
sorry

end NUMINAMATH_CALUDE_line_l1_equation_range_of_b_l659_65930


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_four_l659_65918

theorem smallest_addition_for_multiple_of_four : 
  (∃ n : ℕ+, 4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) ∧ 
  (∀ n : ℕ+, (4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) → n = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_four_l659_65918


namespace NUMINAMATH_CALUDE_angle_sum_l659_65924

theorem angle_sum (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 4 * Real.cos (2 * a) + 3 * Real.cos (2 * b) = 1) :
  a + b = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_l659_65924


namespace NUMINAMATH_CALUDE_min_sum_squares_distances_l659_65952

/-- An isosceles right triangle with leg length a -/
structure IsoscelesRightTriangle (a : ℝ) :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (legs_length : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = a ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = a)
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

/-- The sum of squares of distances from a point to the vertices of the triangle -/
def sum_of_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) (point : ℝ × ℝ) : ℝ :=
  (point.1 - triangle.A.1)^2 + (point.2 - triangle.A.2)^2 +
  (point.1 - triangle.B.1)^2 + (point.2 - triangle.B.2)^2 +
  (point.1 - triangle.C.1)^2 + (point.2 - triangle.C.2)^2

/-- The theorem stating the minimum point and value -/
theorem min_sum_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) :
  ∃ (min_point : ℝ × ℝ),
    (∀ (point : ℝ × ℝ), sum_of_squares_distances a triangle min_point ≤ sum_of_squares_distances a triangle point) ∧
    min_point = (a/3, a/3) ∧
    sum_of_squares_distances a triangle min_point = (4*a^2)/3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_distances_l659_65952


namespace NUMINAMATH_CALUDE_complex_power_difference_l659_65996

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^2048 - 1/x^2048 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l659_65996


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l659_65986

-- Define the ellipse
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  foci_correct : foci1 = (1, 2) ∧ foci2 = (1, 6)
  point_on_ellipse : point = (7, 4)
  equation : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x, y) ∈ {p : ℝ × ℝ | ∃ (t : ℝ), p.1 = h + a * Real.cos t ∧ p.2 = k + b * Real.sin t}

theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 10 := by
  sorry

#check ellipse_a_plus_k

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l659_65986


namespace NUMINAMATH_CALUDE_apples_eaten_by_dog_l659_65964

theorem apples_eaten_by_dog (apples_on_tree : ℕ) (apples_on_ground : ℕ) (apples_remaining : ℕ) : 
  apples_on_tree = 5 → apples_on_ground = 8 → apples_remaining = 10 →
  apples_on_tree + apples_on_ground - apples_remaining = 3 := by
sorry

end NUMINAMATH_CALUDE_apples_eaten_by_dog_l659_65964


namespace NUMINAMATH_CALUDE_max_stickers_purchasable_l659_65926

theorem max_stickers_purchasable (budget : ℚ) (unit_cost : ℚ) : 
  budget = 10 → unit_cost = 3/4 → 
  (∃ (n : ℕ), n * unit_cost ≤ budget ∧ 
    ∀ (m : ℕ), m * unit_cost ≤ budget → m ≤ n) → 
  (∃ (max_stickers : ℕ), max_stickers = 13) :=
by sorry

end NUMINAMATH_CALUDE_max_stickers_purchasable_l659_65926


namespace NUMINAMATH_CALUDE_last_day_of_second_quarter_in_common_year_l659_65915

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Definition of a common year -/
def isCommonYear (daysInYear : Nat) : Prop :=
  daysInYear = 365

/-- Definition of the last day of the second quarter -/
def isLastDayOfSecondQuarter (d : DayInMonth) : Prop :=
  d.month = 6 ∧ d.day = 30

/-- Theorem: In a common year, the last day of the second quarter is June 30 -/
theorem last_day_of_second_quarter_in_common_year 
  (daysInYear : Nat) 
  (h : isCommonYear daysInYear) :
  ∃ d : DayInMonth, isLastDayOfSecondQuarter d :=
by
  sorry

end NUMINAMATH_CALUDE_last_day_of_second_quarter_in_common_year_l659_65915


namespace NUMINAMATH_CALUDE_fraction_greater_than_one_implication_false_l659_65977

theorem fraction_greater_than_one_implication_false : 
  ¬(∀ a b : ℝ, a / b > 1 → a > b) := by sorry

end NUMINAMATH_CALUDE_fraction_greater_than_one_implication_false_l659_65977


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l659_65912

theorem circle_diameter_ratio (C D : Real) :
  -- Circle C is inside circle D
  C < D →
  -- Diameter of circle D is 20 cm
  D = 10 →
  -- Ratio of shaded area to area of circle C is 7:1
  (π * D^2 - π * C^2) / (π * C^2) = 7 →
  -- The diameter of circle C is 5√5 cm
  2 * C = 5 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l659_65912


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_two_l659_65981

theorem sqrt_difference_equals_two_sqrt_two :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_two_l659_65981


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l659_65914

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  let cards_after_maria := initial_cards - (initial_cards + 1) / 2
  let cards_after_peter := cards_after_maria - 1
  let cards_paul_added := final_cards - cards_after_peter
  cards_paul_added = 12

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l659_65914


namespace NUMINAMATH_CALUDE_constant_term_expansion_l659_65925

/-- The constant term in the expansion of (2x + 1/x - 1)^5 is -161 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (2*x + 1/x - 1)^5
  ∃ g : ℝ → ℝ, ∀ x ≠ 0, f x = g x + (-161) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l659_65925


namespace NUMINAMATH_CALUDE_remainder_divisibility_l659_65976

theorem remainder_divisibility (x : ℤ) (h : x % 66 = 14) : x % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l659_65976


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l659_65989

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l659_65989


namespace NUMINAMATH_CALUDE_compound_inequality_l659_65939

theorem compound_inequality (x : ℝ) : 
  x > -1/2 → (3 - 1/(3*x + 4) < 5 ∧ 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_compound_inequality_l659_65939


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l659_65975

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 3) (h2 : b = -1) :
  (4 * a^2 * b - 5 * b^2) - 3 * (a^2 * b - 2 * b^2) = -8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l659_65975


namespace NUMINAMATH_CALUDE_sets_in_borel_sigma_algebra_l659_65983

-- Define the type for infinite sequences of real numbers
def RealSequence := ℕ → ℝ

-- Define the Borel σ-algebra on ℝ^∞
def BorelSigmaAlgebra : Set (Set RealSequence) := sorry

-- Define the limsup of a sequence
def limsup (x : RealSequence) : ℝ := sorry

-- Define the limit of a sequence
def limit (x : RealSequence) : Option ℝ := sorry

-- Theorem statement
theorem sets_in_borel_sigma_algebra (a : ℝ) :
  {x : RealSequence | limsup x ≤ a} ∈ BorelSigmaAlgebra ∧
  {x : RealSequence | ∃ (l : ℝ), limit x = some l ∧ l > a} ∈ BorelSigmaAlgebra :=
sorry

end NUMINAMATH_CALUDE_sets_in_borel_sigma_algebra_l659_65983


namespace NUMINAMATH_CALUDE_hummus_servings_thomas_hummus_servings_l659_65969

/-- Calculates the number of servings of hummus Thomas is making -/
theorem hummus_servings (recipe_cup : ℕ) (can_ounces : ℕ) (cup_ounces : ℕ) (cans_bought : ℕ) : ℕ :=
  let total_ounces := can_ounces * cans_bought
  let servings := total_ounces / cup_ounces
  servings

/-- Proves that Thomas is making 21 servings of hummus -/
theorem thomas_hummus_servings :
  hummus_servings 1 16 6 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_hummus_servings_thomas_hummus_servings_l659_65969


namespace NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_minus_sqrt_two_l659_65972

theorem abs_a_plus_b_equals_three_minus_sqrt_two 
  (a b : ℝ) (h : Real.sqrt (2*a + 6) + |b - Real.sqrt 2| = 0) : 
  |a + b| = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_minus_sqrt_two_l659_65972


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l659_65904

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem third_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 12 →
  arithmetic_sequence a₁ d 22 = 15 →
  arithmetic_sequence a₁ d 3 = -42 :=
by sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l659_65904


namespace NUMINAMATH_CALUDE_plum_difference_l659_65971

def sharon_plums : ℕ := 7
def allan_plums : ℕ := 10

theorem plum_difference : allan_plums - sharon_plums = 3 := by
  sorry

end NUMINAMATH_CALUDE_plum_difference_l659_65971


namespace NUMINAMATH_CALUDE_stock_price_change_l659_65913

/-- Calculates the net percentage change in stock price over three years -/
def netPercentageChange (year1Change : Real) (year2Change : Real) (year3Change : Real) : Real :=
  let price1 := 1 + year1Change
  let price2 := price1 * (1 + year2Change)
  let price3 := price2 * (1 + year3Change)
  (price3 - 1) * 100

/-- Theorem stating the net percentage change for the given scenario -/
theorem stock_price_change : 
  ∀ (ε : Real), ε > 0 → 
  |netPercentageChange (-0.08) 0.10 0.06 - 7.272| < ε :=
sorry

end NUMINAMATH_CALUDE_stock_price_change_l659_65913


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l659_65943

theorem right_triangle_angle_sum (A B C : ℝ) : 
  A = 20 → C = 90 → A + B + C = 180 → B = 70 := by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l659_65943


namespace NUMINAMATH_CALUDE_train_length_l659_65905

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 216) (h2 : time = 6) :
  speed * (5 / 18) * time = 360 :=
sorry

end NUMINAMATH_CALUDE_train_length_l659_65905
