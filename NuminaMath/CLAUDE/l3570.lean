import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_equation_l3570_357049

variables (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 - x + 5

def g (x : ℝ) : ℝ := -x^4 + 7*x^2 + x - 6

theorem polynomial_equation :
  f x + g x = 4*x^2 + x - 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l3570_357049


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3570_357026

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3570_357026


namespace NUMINAMATH_CALUDE_sequence_value_proof_l3570_357037

def sequence_sum (n : ℕ) : ℤ := n^2 - 9*n

def sequence_term (n : ℕ) : ℤ := sequence_sum n - sequence_sum (n-1)

theorem sequence_value_proof (k : ℕ) (h : 5 < sequence_term k ∧ sequence_term k < 8) :
  sequence_term k = 6 := by sorry

end NUMINAMATH_CALUDE_sequence_value_proof_l3570_357037


namespace NUMINAMATH_CALUDE_max_notebooks_with_10_dollars_l3570_357024

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | single
  | pack4
  | pack7

/-- Returns the cost of a pack given its size -/
def packCost (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 3
  | PackSize.pack7 => 5

/-- Returns the number of notebooks in a pack given its size -/
def packNotebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.single => 1
  | PackSize.pack4 => 4
  | PackSize.pack7 => 7

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * packCost PackSize.single +
  p.pack4 * packCost PackSize.pack4 +
  p.pack7 * packCost PackSize.pack7

/-- Calculates the total number of notebooks in a purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * packNotebooks PackSize.single +
  p.pack4 * packNotebooks PackSize.pack4 +
  p.pack7 * packNotebooks PackSize.pack7

/-- The maximum number of notebooks that can be purchased with $10 is 14 -/
theorem max_notebooks_with_10_dollars :
  (∀ p : Purchase, totalCost p ≤ 10 → totalNotebooks p ≤ 14) ∧
  (∃ p : Purchase, totalCost p ≤ 10 ∧ totalNotebooks p = 14) :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_with_10_dollars_l3570_357024


namespace NUMINAMATH_CALUDE_line_circle_intersection_condition_l3570_357091

/-- The line y = x + b intersects the circle x^2 + y^2 = 1 -/
def line_intersects_circle (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = x + b ∧ x^2 + y^2 = 1

/-- The condition 0 < b < 1 is necessary but not sufficient for the intersection -/
theorem line_circle_intersection_condition (b : ℝ) :
  line_intersects_circle b → 0 < b ∧ b < 1 ∧
  ¬(∀ b : ℝ, 0 < b ∧ b < 1 → line_intersects_circle b) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_condition_l3570_357091


namespace NUMINAMATH_CALUDE_kelly_apples_l3570_357038

def initial_apples : ℕ := 56
def picked_apples : ℝ := 105.0
def total_apples : ℕ := 161

theorem kelly_apples : 
  initial_apples + picked_apples = total_apples :=
by sorry

end NUMINAMATH_CALUDE_kelly_apples_l3570_357038


namespace NUMINAMATH_CALUDE_multiple_of_nine_three_odd_l3570_357089

theorem multiple_of_nine_three_odd (n : ℕ) :
  (∀ m : ℕ, 9 ∣ m → 3 ∣ m) →
  Odd n →
  9 ∣ n →
  3 ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_three_odd_l3570_357089


namespace NUMINAMATH_CALUDE_remaining_two_average_l3570_357061

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) :
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 4.60 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.8 →
  (n₅ + n₆) / 2 = 6.6 :=
by sorry

end NUMINAMATH_CALUDE_remaining_two_average_l3570_357061


namespace NUMINAMATH_CALUDE_unique_consecutive_triangle_with_double_angle_l3570_357029

/-- Represents a triangle with side lengths (a, a+1, a+2) -/
structure ConsecutiveTriangle where
  a : ℕ
  a_pos : a > 0

/-- Calculates the cosine of an angle in a ConsecutiveTriangle using the law of cosines -/
def cos_angle (t : ConsecutiveTriangle) (side : Fin 3) : ℚ :=
  match side with
  | 0 => (t.a^2 + 6*t.a + 5) / (2*t.a^2 + 6*t.a + 4)
  | 1 => ((t.a + 1) * (t.a + 3)) / (2*t.a*(t.a + 2))
  | 2 => ((t.a - 1) * (t.a - 3)) / (2*t.a*(t.a + 1))

/-- Checks if one angle is twice another in a ConsecutiveTriangle -/
def has_double_angle (t : ConsecutiveTriangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ cos_angle t j = 2 * (cos_angle t i)^2 - 1

/-- The main theorem stating that there's a unique ConsecutiveTriangle with a double angle -/
theorem unique_consecutive_triangle_with_double_angle :
  ∃! (t : ConsecutiveTriangle), has_double_angle t :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_triangle_with_double_angle_l3570_357029


namespace NUMINAMATH_CALUDE_inequality_proof_l3570_357034

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3570_357034


namespace NUMINAMATH_CALUDE_gcd_equality_l3570_357090

theorem gcd_equality (a b c : ℕ+) (h : Nat.gcd (a^2 - 1) (Nat.gcd (b^2 - 1) (c^2 - 1)) = 1) :
  Nat.gcd (a*b + c) (Nat.gcd (b*c + a) (c*a + b)) = Nat.gcd a (Nat.gcd b c) :=
sorry

end NUMINAMATH_CALUDE_gcd_equality_l3570_357090


namespace NUMINAMATH_CALUDE_elvin_first_month_bill_l3570_357017

/-- Represents Elvin's monthly telephone bill structure -/
structure PhoneBill where
  fixed_charge : ℝ
  call_charge : ℝ

/-- Calculates the total bill given a PhoneBill -/
def total_bill (bill : PhoneBill) : ℝ :=
  bill.fixed_charge + bill.call_charge

theorem elvin_first_month_bill :
  ∀ (bill1 bill2 : PhoneBill),
    total_bill bill1 = 52 →
    total_bill bill2 = 76 →
    bill2.call_charge = 2 * bill1.call_charge →
    bill1.fixed_charge = bill2.fixed_charge →
    total_bill bill1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_elvin_first_month_bill_l3570_357017


namespace NUMINAMATH_CALUDE_cos_135_degrees_l3570_357004

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l3570_357004


namespace NUMINAMATH_CALUDE_log_simplification_l3570_357096

theorem log_simplification (a b c d x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : x > 0) (h6 : y > 0) :
  Real.log (2 * a / (3 * b)) + Real.log (3 * b / (4 * c)) + Real.log (4 * c / (5 * d)) - Real.log (10 * a * y / (3 * d * x)) = Real.log (3 * x / (25 * y)) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l3570_357096


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3570_357057

theorem tangent_line_to_ln_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3570_357057


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3570_357077

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3570_357077


namespace NUMINAMATH_CALUDE_contrapositive_example_l3570_357068

theorem contrapositive_example : 
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3570_357068


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3570_357010

theorem gcd_power_two_minus_one : Nat.gcd (2^1025 - 1) (2^1056 - 1) = 2^31 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3570_357010


namespace NUMINAMATH_CALUDE_lattice_triangle_properties_l3570_357067

/-- A lattice point in the xy-plane -/
structure LatticePoint where
  x : Int
  y : Int

/-- A triangle with vertices at lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Count of lattice points on a side (excluding endpoints) -/
def latticePointsOnSide (P Q : LatticePoint) : Nat :=
  sorry

/-- Area of a triangle with lattice point vertices -/
def triangleArea (t : LatticeTriangle) : Int :=
  sorry

theorem lattice_triangle_properties (t : LatticeTriangle) :
  (latticePointsOnSide t.A t.B % 2 = 1 ∧ latticePointsOnSide t.A t.C % 2 = 1 →
    latticePointsOnSide t.B t.C % 2 = 1) ∧
  (latticePointsOnSide t.A t.B = 3 ∧ latticePointsOnSide t.A t.C = 3 →
    ∃ k : Int, triangleArea t = 8 * k) :=
  sorry

end NUMINAMATH_CALUDE_lattice_triangle_properties_l3570_357067


namespace NUMINAMATH_CALUDE_juan_milk_needed_l3570_357075

/-- The number of cookies that can be baked with one half-gallon of milk -/
def cookies_per_half_gallon : ℕ := 48

/-- The number of cookies Juan wants to bake -/
def cookies_to_bake : ℕ := 40

/-- The amount of milk needed for baking, in half-gallons -/
def milk_needed : ℕ := 1

theorem juan_milk_needed :
  cookies_to_bake ≤ cookies_per_half_gallon → milk_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_juan_milk_needed_l3570_357075


namespace NUMINAMATH_CALUDE_min_subset_size_for_sum_l3570_357099

theorem min_subset_size_for_sum (n : ℕ+) :
  let M := Finset.range (2 * n)
  ∃ k : ℕ+, (∀ A : Finset ℕ, A ⊆ M → A.card = k →
    ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1) ∧
  (∀ k' : ℕ+, k' < k →
    ∃ A : Finset ℕ, A ⊆ M ∧ A.card = k' ∧
    ∀ a b c d : ℕ, a ∈ A → b ∈ A → c ∈ A → d ∈ A →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a + b + c + d ≠ 4 * n + 1) ∧
  k = n + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_subset_size_for_sum_l3570_357099


namespace NUMINAMATH_CALUDE_range_of_a_l3570_357018

theorem range_of_a (x a : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → (x - a) * (x - a - 4) > 0) ∧ 
  (∃ x, (x - a) * (x - a - 4) > 0 ∧ (x < -2 ∨ x > 1)) →
  a < -6 ∨ a > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3570_357018


namespace NUMINAMATH_CALUDE_range_of_a_l3570_357019

/-- Given a ≥ -2, prove that if C ⊆ B, then a ∈ [1/2, 3] -/
theorem range_of_a (a : ℝ) (ha : a ≥ -2) :
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2 * x + 3}
  let C := {t : ℝ | ∃ x ∈ A, t = x^2}
  C ⊆ B → a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3570_357019


namespace NUMINAMATH_CALUDE_unique_a_l3570_357055

theorem unique_a : ∃! a : ℝ, (∃ m : ℤ, a + 2/3 = m) ∧ (∃ n : ℤ, 1/a - 3/4 = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_l3570_357055


namespace NUMINAMATH_CALUDE_salary_increase_after_five_years_l3570_357083

theorem salary_increase_after_five_years (annual_raise : Real) 
  (h1 : annual_raise = 0.15) : 
  (1 + annual_raise)^5 > 2 := by
  sorry

#check salary_increase_after_five_years

end NUMINAMATH_CALUDE_salary_increase_after_five_years_l3570_357083


namespace NUMINAMATH_CALUDE_polar_equation_is_parabola_l3570_357074

theorem polar_equation_is_parabola :
  ∀ (r θ x y : ℝ),
  (r = 2 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ (a b : ℝ), x^2 = a * y + b :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_parabola_l3570_357074


namespace NUMINAMATH_CALUDE_max_value_of_linear_function_max_value_achieved_l3570_357097

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- State the theorem
theorem max_value_of_linear_function :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ 3 :=
by
  sorry

-- State that the maximum is achieved
theorem max_value_achieved :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_linear_function_max_value_achieved_l3570_357097


namespace NUMINAMATH_CALUDE_smallest_polygon_with_lighting_property_l3570_357084

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a polygon -/
def isInside (p : Point) (poly : Polygon n) : Prop := sorry

/-- Predicate to check if a point on a side of the polygon is lightened by a bulb -/
def isLightened (p : Point) (side : Fin n) (poly : Polygon n) (bulb : Point) : Prop := sorry

/-- Predicate to check if a polygon satisfies the lighting property -/
def hasLightingProperty (poly : Polygon n) : Prop :=
  ∃ bulb : Point, isInside bulb poly ∧
    ∀ side : Fin n, ∃ p : Point, ¬isLightened p side poly bulb

/-- Predicate to check if two bulbs light up the whole perimeter -/
def lightsWholePerimeter (poly : Polygon n) (bulb1 bulb2 : Point) : Prop :=
  ∀ side : Fin n, ∀ p : Point, isLightened p side poly bulb1 ∨ isLightened p side poly bulb2

theorem smallest_polygon_with_lighting_property :
  (∀ n < 6, ¬∃ poly : Polygon n, hasLightingProperty poly) ∧
  (∃ poly : Polygon 6, hasLightingProperty poly) ∧
  (∀ poly : Polygon 6, ∃ bulb1 bulb2 : Point, lightsWholePerimeter poly bulb1 bulb2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_polygon_with_lighting_property_l3570_357084


namespace NUMINAMATH_CALUDE_rectangles_in_five_by_five_grid_l3570_357087

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square array of dots -/
def numRectangles (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_five_by_five_grid :
  numRectangles gridSize = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_five_by_five_grid_l3570_357087


namespace NUMINAMATH_CALUDE_sandys_change_l3570_357047

/-- Represents the cost of a drink order -/
structure DrinkOrder where
  cappuccino : ℕ
  icedTea : ℕ
  cafeLatte : ℕ
  espresso : ℕ

/-- Calculates the total cost of a drink order -/
def totalCost (order : DrinkOrder) : ℚ :=
  2 * order.cappuccino + 3 * order.icedTea + 1.5 * order.cafeLatte + 1 * order.espresso

/-- Calculates the change received from a given payment -/
def changeReceived (payment : ℚ) (order : DrinkOrder) : ℚ :=
  payment - totalCost order

/-- Sandy's specific drink order -/
def sandysOrder : DrinkOrder :=
  { cappuccino := 3
  , icedTea := 2
  , cafeLatte := 2
  , espresso := 2 }

/-- Theorem stating that Sandy receives $3 in change -/
theorem sandys_change :
  changeReceived 20 sandysOrder = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_change_l3570_357047


namespace NUMINAMATH_CALUDE_total_students_l3570_357069

theorem total_students (A B C : ℕ) : 
  B = A - 8 →
  C = 5 * B →
  B = 25 →
  A + B + C = 183 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3570_357069


namespace NUMINAMATH_CALUDE_benny_payment_l3570_357082

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people in the group -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_payment : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_benny_payment_l3570_357082


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3570_357073

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3570_357073


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3570_357053

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3570_357053


namespace NUMINAMATH_CALUDE_irrational_pi_only_l3570_357065

theorem irrational_pi_only (a b c d : ℝ) : 
  a = 1 / 7 → b = Real.pi → c = -1 → d = 0 → 
  (¬ Irrational a ∧ Irrational b ∧ ¬ Irrational c ∧ ¬ Irrational d) := by
  sorry

end NUMINAMATH_CALUDE_irrational_pi_only_l3570_357065


namespace NUMINAMATH_CALUDE_total_unique_eagle_types_l3570_357041

/-- The number of unique types of eagles across all sections -/
def uniqueEagleTypes (sectionA sectionB sectionC sectionD sectionE : ℝ)
  (overlapAB overlapBC overlapCD overlapDE overlapACE : ℝ) : ℝ :=
  sectionA + sectionB + sectionC + sectionD + sectionE - 
  (overlapAB + overlapBC + overlapCD + overlapDE - overlapACE)

/-- Theorem stating the total number of unique eagle types -/
theorem total_unique_eagle_types :
  uniqueEagleTypes 12.5 8.3 10.7 14.2 17.1 3.5 2.1 3.7 4.4 1.5 = 51.6 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_eagle_types_l3570_357041


namespace NUMINAMATH_CALUDE_book_selection_ways_l3570_357078

def num_books : ℕ := 5
def num_students : ℕ := 2

theorem book_selection_ways :
  (num_books ^ num_students : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l3570_357078


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3570_357095

universe u

def U : Set (Fin 6) := {1, 2, 3, 4, 5, 6}
def P : Set (Fin 6) := {1, 2, 3, 4}
def Q : Set (Fin 6) := {3, 4, 5, 6}

theorem intersection_complement_equality :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3570_357095


namespace NUMINAMATH_CALUDE_scientific_notation_1742000_l3570_357025

theorem scientific_notation_1742000 :
  ∃ (a : ℝ) (n : ℤ), 1742000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  use 1.742, 6
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1742000_l3570_357025


namespace NUMINAMATH_CALUDE_principal_is_2500_l3570_357044

/-- Given a simple interest, interest rate, and time period, calculates the principal amount. -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 2500. -/
theorem principal_is_2500 :
  let simple_interest : ℚ := 1000
  let rate : ℚ := 10
  let time : ℚ := 4
  calculate_principal simple_interest rate time = 2500 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_2500_l3570_357044


namespace NUMINAMATH_CALUDE_order_of_exponentials_l3570_357062

theorem order_of_exponentials : 
  let a : ℝ := (2 : ℝ) ^ (1/5 : ℝ)
  let b : ℝ := (2/5 : ℝ) ^ (1/5 : ℝ)
  let c : ℝ := (2/5 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l3570_357062


namespace NUMINAMATH_CALUDE_island_inhabitants_l3570_357042

theorem island_inhabitants (total : Nat) (blue_eyed : Nat) (brown_eyed : Nat) : 
  total = 100 →
  blue_eyed + brown_eyed = total →
  (blue_eyed * brown_eyed * 2 > (total * (total - 1)) / 2) →
  (∀ (x : Nat), x ≤ blue_eyed → x ≤ brown_eyed → x * (total - x) ≤ blue_eyed * brown_eyed) →
  46 ≤ brown_eyed ∧ brown_eyed ≤ 54 := by
  sorry

end NUMINAMATH_CALUDE_island_inhabitants_l3570_357042


namespace NUMINAMATH_CALUDE_simplify_expression_l3570_357052

theorem simplify_expression : 8 * (15 / 11) * (-25 / 40) = -15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3570_357052


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3570_357071

/-- The line equation y = 2mx + 2 intersects the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if m^2 = 3/16 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, (2 * p.1^2 + 8 * p.2^2 = 8) ∧ (p.2 = 2 * m * p.1 + 2)) ↔ m^2 = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3570_357071


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l3570_357093

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x ≥ m ∧ ∃ (y : ℝ), f 1 y = m ∧ m = 1 :=
sorry

-- Part 2: Range of a for f(x) ≥ a when x ∈ [-1, +∞)
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ -1 → f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l3570_357093


namespace NUMINAMATH_CALUDE_given_circles_are_externally_tangent_l3570_357007

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- Definition of the given circles --/
def givenCircles : TwoCircles where
  c1 := fun (x, y) ↦ x^2 + y^2 - 4*x - 6*y + 9 = 0
  c2 := fun (x, y) ↦ x^2 + y^2 + 12*x + 6*y - 19 = 0

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii --/
def areExternallyTangent (circles : TwoCircles) : Prop :=
  ∃ (x1 y1 x2 y2 r1 r2 : ℝ),
    (∀ (x y : ℝ), circles.c1 (x, y) ↔ (x - x1)^2 + (y - y1)^2 = r1^2) ∧
    (∀ (x y : ℝ), circles.c2 (x, y) ↔ (x - x2)^2 + (y - y2)^2 = r2^2) ∧
    (x2 - x1)^2 + (y2 - y1)^2 = (r1 + r2)^2

/-- Theorem stating that the given circles are externally tangent --/
theorem given_circles_are_externally_tangent :
  areExternallyTangent givenCircles := by sorry

end NUMINAMATH_CALUDE_given_circles_are_externally_tangent_l3570_357007


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3570_357035

-- Define the direction vectors
def v1 (a : ℝ) : Fin 3 → ℝ := ![2*a, 3, 2]
def v2 : Fin 3 → ℝ := ![2, 3, 2]

-- Define the condition for parallel lines
def are_parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, v1 a i = k * v2 i

-- Theorem statement
theorem parallel_lines_imply_a_equals_one :
  are_parallel 1 → ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l3570_357035


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3570_357080

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3570_357080


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l3570_357045

theorem probability_of_y_selection 
  (p_x : ℝ) 
  (p_both : ℝ) 
  (h1 : p_x = 1 / 7)
  (h2 : p_both = 0.05714285714285714) :
  p_both / p_x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l3570_357045


namespace NUMINAMATH_CALUDE_even_number_power_of_two_l3570_357050

theorem even_number_power_of_two (A : ℕ) :
  A % 2 = 0 →
  (∀ P : ℕ, Nat.Prime P → P ∣ A → (P - 1) ∣ (A - 1)) →
  ∃ k : ℕ, A = 2^k :=
sorry

end NUMINAMATH_CALUDE_even_number_power_of_two_l3570_357050


namespace NUMINAMATH_CALUDE_additional_birds_l3570_357015

theorem additional_birds (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312) :
  final_birds - initial_birds = 81 := by
  sorry

end NUMINAMATH_CALUDE_additional_birds_l3570_357015


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3570_357059

theorem ping_pong_rackets_sold (total_amount : ℝ) (average_price : ℝ) (h1 : total_amount = 490) (h2 : average_price = 9.8) :
  total_amount / average_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l3570_357059


namespace NUMINAMATH_CALUDE_base12_divisibility_rule_l3570_357046

/-- 
Represents a number in base-12 as a list of digits, 
where each digit is between 0 and 11 (inclusive).
--/
def Base12Number := List Nat

/-- Converts a Base12Number to its decimal representation. --/
def toDecimal (n : Base12Number) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (12 ^ i)) 0

/-- Calculates the sum of digits in a Base12Number. --/
def digitSum (n : Base12Number) : Nat :=
  n.sum

theorem base12_divisibility_rule (n : Base12Number) :
  11 ∣ digitSum n → 11 ∣ toDecimal n := by sorry

end NUMINAMATH_CALUDE_base12_divisibility_rule_l3570_357046


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l3570_357094

/-- Represents the value of one billion -/
def billion : ℝ := 10^9

/-- The given amount in billions -/
def amount : ℝ := 4.15

theorem billion_scientific_notation : 
  amount * billion = 4.15 * 10^9 := by sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l3570_357094


namespace NUMINAMATH_CALUDE_pie_chart_percentage_central_angle_relation_l3570_357081

/-- Represents a part of a pie chart -/
structure PieChartPart where
  percentage : ℝ
  centralAngle : ℝ

/-- Theorem stating the relationship between percentage and central angle in a pie chart -/
theorem pie_chart_percentage_central_angle_relation (part : PieChartPart) :
  part.percentage = part.centralAngle / 360 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_percentage_central_angle_relation_l3570_357081


namespace NUMINAMATH_CALUDE_log2_derivative_l3570_357079

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_log2_derivative_l3570_357079


namespace NUMINAMATH_CALUDE_beaver_home_fraction_l3570_357070

theorem beaver_home_fraction (total_beavers : ℕ) (swim_percentage : ℚ) :
  total_beavers = 4 →
  swim_percentage = 3/4 →
  (1 : ℚ) - swim_percentage = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_beaver_home_fraction_l3570_357070


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3570_357031

theorem opposite_of_negative_fraction :
  -(-(1 / 2024)) = 1 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3570_357031


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l3570_357016

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |g(x)| = 15 for x ∈ {0, 1, 2, 4, 5, 6} -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g x| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 75 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l3570_357016


namespace NUMINAMATH_CALUDE_markup_is_ten_l3570_357064

/-- Calculates the markup given shop price, tax rate, and profit -/
def calculate_markup (shop_price : ℝ) (tax_rate : ℝ) (profit : ℝ) : ℝ :=
  shop_price - (shop_price * (1 - tax_rate) - profit)

theorem markup_is_ten :
  let shop_price : ℝ := 90
  let tax_rate : ℝ := 0.1
  let profit : ℝ := 1
  calculate_markup shop_price tax_rate profit = 10 := by
sorry

end NUMINAMATH_CALUDE_markup_is_ten_l3570_357064


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3570_357020

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A line passing through a point and intersecting an ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  intersectionPoints : Fin 2 → ℝ × ℝ

/-- The problem statement -/
theorem ellipse_and_line_properties
  (E : Ellipse)
  (l : IntersectingLine E)
  (h₁ : E.a^2 - E.b^2 = 1) -- Condition for foci at (-1,0) and (1,0)
  (h₂ : (l.intersectionPoints 0).1 + (l.intersectionPoints 1).1 +
        ((l.intersectionPoints 0).1 + 1)^2 + (l.intersectionPoints 0).2^2 +
        ((l.intersectionPoints 1).1 + 1)^2 + (l.intersectionPoints 1).2^2 = 16) -- Perimeter condition
  (h₃ : (l.intersectionPoints 0).1 * (l.intersectionPoints 1).1 +
        (l.intersectionPoints 0).2 * (l.intersectionPoints 1).2 = 0) -- Perpendicularity condition
  : (E.a = Real.sqrt 3 ∧ E.b = Real.sqrt 2) ∧
    (l.k = Real.sqrt 2 ∨ l.k = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3570_357020


namespace NUMINAMATH_CALUDE_subset_implies_a_squared_equals_two_l3570_357056

/-- Given sets A and B, where A = {0, 2, 3} and B = {2, a² + 1}, and B is a subset of A,
    prove that a² = 2, where a is a real number. -/
theorem subset_implies_a_squared_equals_two (a : ℝ) : 
  let A : Set ℝ := {0, 2, 3}
  let B : Set ℝ := {2, a^2 + 1}
  B ⊆ A → a^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_squared_equals_two_l3570_357056


namespace NUMINAMATH_CALUDE_tan_theta_value_l3570_357092

theorem tan_theta_value (θ : Real) 
  (h1 : Real.cos (θ / 2) = 4 / 5) 
  (h2 : Real.sin θ < 0) : 
  Real.tan θ = -24 / 7 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3570_357092


namespace NUMINAMATH_CALUDE_franks_boxes_l3570_357009

/-- The number of boxes Frank filled with toys -/
def filled_boxes : ℕ := 8

/-- The number of boxes Frank has left empty -/
def empty_boxes : ℕ := 5

/-- The total number of boxes Frank had initially -/
def total_boxes : ℕ := filled_boxes + empty_boxes

theorem franks_boxes : total_boxes = 13 := by
  sorry

end NUMINAMATH_CALUDE_franks_boxes_l3570_357009


namespace NUMINAMATH_CALUDE_store_price_reduction_l3570_357008

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 →
  first_reduction < 100 →
  (original_price * (1 - first_reduction / 100) * (1 - 0.14) = 0.774 * original_price) →
  first_reduction = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_price_reduction_l3570_357008


namespace NUMINAMATH_CALUDE_max_value_of_a_l3570_357054

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_six : a^2 + b^2 + c^2 = 6) : 
  ∀ x : ℝ, x ≤ 2 ∧ (∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 6 ∧ a₀ = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3570_357054


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_square_l3570_357063

theorem square_sum_and_reciprocal_square (x : ℝ) (h : x + 2/x = 6) :
  x^2 + 4/x^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_square_l3570_357063


namespace NUMINAMATH_CALUDE_average_transformation_l3570_357021

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l3570_357021


namespace NUMINAMATH_CALUDE_unique_triple_l3570_357060

theorem unique_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 1 ∧ (x, y, z) = (2, 2, 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l3570_357060


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l3570_357022

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (final_avg : ℚ) : 
  initial_avg = 17 →
  incorrect_num = 26 →
  correct_num = 56 →
  final_avg = 20 →
  (∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg + (correct_num - incorrect_num) ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l3570_357022


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l3570_357088

-- Define the dimensions of the yard
def yard_length : ℝ := 30
def yard_width : ℝ := 6

-- Define the lengths of the parallel sides of the trapezoidal remainder
def trapezoid_long_side : ℝ := 30
def trapezoid_short_side : ℝ := 20

-- Define the fraction we want to prove
def target_fraction : ℚ := 5/36

-- Theorem statement
theorem flower_bed_fraction :
  let yard_area := yard_length * yard_width
  let triangle_leg := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := triangle_leg^2 / 2
  let flower_beds_area := 2 * triangle_area
  flower_beds_area / yard_area = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l3570_357088


namespace NUMINAMATH_CALUDE_money_division_l3570_357051

theorem money_division (a b c : ℚ) : 
  a = (1/2) * b ∧ b = (1/2) * c ∧ c = 232 → a + b + c = 406 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3570_357051


namespace NUMINAMATH_CALUDE_people_who_got_off_l3570_357085

theorem people_who_got_off (initial_people : ℕ) (people_left : ℕ) : 
  initial_people = 48 → people_left = 31 → initial_people - people_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_l3570_357085


namespace NUMINAMATH_CALUDE_derivative_zero_neither_necessary_nor_sufficient_l3570_357023

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (a - ε) (a + ε), f x ≤ f a ∨ f x ≥ f a

-- Define the statement to be proven
theorem derivative_zero_neither_necessary_nor_sufficient :
  ¬(∀ f : ℝ → ℝ, ∀ a : ℝ, (has_extremum f a ↔ HasDerivAt f 0 a)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_neither_necessary_nor_sufficient_l3570_357023


namespace NUMINAMATH_CALUDE_polynomial_equality_l3570_357027

theorem polynomial_equality (m n : ℤ) : 
  (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m*x + n) → 
  (m = 4 ∧ n = -32) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3570_357027


namespace NUMINAMATH_CALUDE_factorization_equality_l3570_357072

theorem factorization_equality (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equality_l3570_357072


namespace NUMINAMATH_CALUDE_next_year_with_digit_sum_five_l3570_357001

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem next_year_with_digit_sum_five : 
  ∀ y : ℕ, y > 2021 ∧ y < 2030 → sum_of_digits y ≠ 5 ∧ sum_of_digits 2030 = 5 :=
by sorry

end NUMINAMATH_CALUDE_next_year_with_digit_sum_five_l3570_357001


namespace NUMINAMATH_CALUDE_tangent_circles_count_l3570_357011

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- Checks if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

/-- Counts the number of circles tangent to all three given lines -/
def count_tangent_circles (l1 l2 l3 : Line) : Nat :=
  sorry

/-- The main theorem stating the possible values for the number of tangent circles -/
theorem tangent_circles_count (l1 l2 l3 : Line) :
  l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 →
  (count_tangent_circles l1 l2 l3 = 0 ∨
   count_tangent_circles l1 l2 l3 = 2 ∨
   count_tangent_circles l1 l2 l3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l3570_357011


namespace NUMINAMATH_CALUDE_inequality_solution_l3570_357039

theorem inequality_solution (x : ℝ) : 
  (|(7 - 2*x) / 4| < 3) ↔ (-5/2 < x ∧ x < 19/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3570_357039


namespace NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3570_357043

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let num_bins : ℕ := 5
  let prob_6_7_4_4_4 := (Nat.choose num_bins 2 * Nat.choose total_balls 6 * Nat.choose 19 7 * 
                         Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
                        (num_bins ^ total_balls : ℚ)
  let prob_5_5_5_5_5 := (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * 
                         Nat.choose 10 5 * Nat.choose 5 5) / 
                        (num_bins ^ total_balls : ℚ)
  prob_6_7_4_4_4 / prob_5_5_5_5_5 = 
    (10 * Nat.choose total_balls 6 * Nat.choose 19 7 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
    (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * Nat.choose 10 5 * Nat.choose 5 5)
  := by sorry

end NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l3570_357043


namespace NUMINAMATH_CALUDE_positive_sum_inequalities_l3570_357002

theorem positive_sum_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 4) : 
  a^2 + b^2/4 + c^2/9 ≥ 8/7 ∧ 
  1/(a+c) + 1/(a+b) + 1/(b+c) ≥ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_inequalities_l3570_357002


namespace NUMINAMATH_CALUDE_simplify_fourth_root_exponent_sum_l3570_357030

theorem simplify_fourth_root_exponent_sum (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ) (n m : ℕ), (48 * a^5 * b^8 * c^14)^(1/4) = k * b^n * c^m ∧ n + m = 5 :=
sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_exponent_sum_l3570_357030


namespace NUMINAMATH_CALUDE_age_difference_l3570_357048

theorem age_difference (a b c d : ℤ) 
  (total_ab_cd : a + b = c + d + 20)
  (total_bd_ac : b + d = a + c + 10) :
  d = a - 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3570_357048


namespace NUMINAMATH_CALUDE_circle_equation_l3570_357005

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point :=
  ⟨1, 0⟩

/-- The line passing through the focus with slope angle 30° -/
def Line : Set Point :=
  {p : Point | p.y = (Real.sqrt 3 / 3) * (p.x - 1)}

/-- Intersection points of the parabola and the line -/
def intersectionPoints : Set Point :=
  Parabola ∩ Line

/-- The circle with AB as diameter -/
def Circle (A B : Point) : Set Point :=
  {p : Point | (p.x - (A.x + B.x) / 2)^2 + (p.y - (A.y + B.y) / 2)^2 = ((A.x - B.x)^2 + (A.y - B.y)^2) / 4}

theorem circle_equation (A B : Point) 
  (hA : A ∈ intersectionPoints) (hB : B ∈ intersectionPoints) (hAB : A ≠ B) :
  Circle A B = {p : Point | (p.x - 7)^2 + (p.y - 2 * Real.sqrt 3)^2 = 64} :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3570_357005


namespace NUMINAMATH_CALUDE_weight_of_b_l3570_357013

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l3570_357013


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_functions_l3570_357036

/-- A function that satisfies the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = x^2) ∧ (Set.range f = Set.Icc 1 4)

/-- There exist infinitely many functions satisfying the given conditions -/
theorem infinitely_many_satisfying_functions :
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ f ∈ S, satisfying_function f :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_functions_l3570_357036


namespace NUMINAMATH_CALUDE_star_equality_l3570_357058

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (5,4) ★ (1,1) = (x,y) ★ (4,3), then x = 8 -/
theorem star_equality (x y : ℤ) :
  star 5 4 1 1 = star x y 4 3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_l3570_357058


namespace NUMINAMATH_CALUDE_z_coordinate_at_x_7_l3570_357028

/-- A line in 3D space passing through two points -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Get the z-coordinate of a point on the line given its x-coordinate -/
def get_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

theorem z_coordinate_at_x_7 (line : Line3D) 
  (h1 : line.point1 = (1, 4, 3)) 
  (h2 : line.point2 = (4, 3, 0)) : 
  get_z_coordinate line 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_z_coordinate_at_x_7_l3570_357028


namespace NUMINAMATH_CALUDE_max_value_constraint_l3570_357012

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = 5 ∧ ∀ (a b : ℝ), a^2 + b^2 + a*b = 1 → 3*a - 2*b ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3570_357012


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3570_357032

theorem quadratic_function_property (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -1/3 * x₁^2 + 5 →
  y₂ = -1/3 * x₂^2 + 5 →
  0 < x₁ →
  x₁ < x₂ →
  y₂ < y₁ ∧ y₁ < 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3570_357032


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3570_357086

noncomputable def f (x a : ℝ) : ℝ := (1 + Real.cos (2 * x)) * 1 + 1 * (Real.sqrt 3 * Real.sin (2 * x) + a)

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3570_357086


namespace NUMINAMATH_CALUDE_point_n_from_m_l3570_357040

/-- Given two points M and N in a 2D plane, prove that N can be obtained
    from M by moving 4 units upward. -/
theorem point_n_from_m (M N : ℝ × ℝ) : 
  M = (-1, -1) → N = (-1, 3) → N.2 - M.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_n_from_m_l3570_357040


namespace NUMINAMATH_CALUDE_solve_for_y_l3570_357076

theorem solve_for_y (x y : ℝ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3570_357076


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3570_357003

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a ≤ 0 ∧ a^2 + a ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3570_357003


namespace NUMINAMATH_CALUDE_correct_proposition_l3570_357006

-- Define proposition p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem statement
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l3570_357006


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3570_357066

theorem digit_sum_problem (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- digits are less than 10
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- digits are different
  (c + a = 10) →  -- condition from right column
  (b + c + 1 = 10) →  -- condition from middle column
  (a + d + 1 = 11) →  -- condition from left column
  (a + b + c + d = 19) :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3570_357066


namespace NUMINAMATH_CALUDE_water_speed_calculation_l3570_357033

/-- 
Given a person who can swim at 4 km/h in still water and takes 7 hours to swim 14 km against a current,
prove that the speed of the water is 2 km/h.
-/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) :
  still_water_speed = 4 →
  distance = 14 →
  time = 7 →
  ∃ (water_speed : ℝ), water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l3570_357033


namespace NUMINAMATH_CALUDE_zack_traveled_18_countries_l3570_357098

/-- The number of countries George traveled to -/
def george_countries : ℕ := 6

/-- The number of countries Joseph traveled to -/
def joseph_countries : ℕ := george_countries / 2

/-- The number of countries Patrick traveled to -/
def patrick_countries : ℕ := 3 * joseph_countries

/-- The number of countries Zack traveled to -/
def zack_countries : ℕ := 2 * patrick_countries

/-- Proof that Zack traveled to 18 countries -/
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_18_countries_l3570_357098


namespace NUMINAMATH_CALUDE_water_spilled_is_eight_quarts_l3570_357014

/-- Represents the water supply problem from the shipwreck scenario -/
structure WaterSupply where
  initial_people : ℕ
  initial_days : ℕ
  spill_day : ℕ
  quart_per_person_per_day : ℕ

/-- The amount of water spilled in the shipwreck scenario -/
def water_spilled (ws : WaterSupply) : ℕ :=
  ws.initial_people + 7

/-- Theorem stating that the amount of water spilled is 8 quarts -/
theorem water_spilled_is_eight_quarts (ws : WaterSupply) 
  (h1 : ws.initial_days = 13)
  (h2 : ws.quart_per_person_per_day = 1)
  (h3 : ws.spill_day = 5)
  (h4 : ws.initial_people > 0)
  : water_spilled ws = 8 := by
  sorry

#check water_spilled_is_eight_quarts

end NUMINAMATH_CALUDE_water_spilled_is_eight_quarts_l3570_357014


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3570_357000

/-- The smallest natural number divisible by 21 with exactly 105 distinct divisors -/
def smallest_number_with_properties : ℕ := 254016

/-- The number of distinct divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  (smallest_number_with_properties % 21 = 0) ∧
  (num_divisors smallest_number_with_properties = 105) ∧
  (∀ m : ℕ, m < smallest_number_with_properties →
    ¬(m % 21 = 0 ∧ num_divisors m = 105)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3570_357000
