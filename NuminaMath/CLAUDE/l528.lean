import Mathlib

namespace NUMINAMATH_CALUDE_water_added_to_reach_new_ratio_l528_52884

-- Define the initial mixture volume
def initial_volume : ℝ := 80

-- Define the initial ratio of milk to water
def initial_milk_ratio : ℝ := 7
def initial_water_ratio : ℝ := 3

-- Define the amount of water evaporated
def evaporated_water : ℝ := 8

-- Define the new ratio of milk to water
def new_milk_ratio : ℝ := 5
def new_water_ratio : ℝ := 4

-- Theorem to prove
theorem water_added_to_reach_new_ratio :
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let water_after_evaporation := initial_water - evaporated_water
  let x := (((new_water_ratio / new_milk_ratio) * initial_milk) - water_after_evaporation)
  x = 28.8 := by sorry

end NUMINAMATH_CALUDE_water_added_to_reach_new_ratio_l528_52884


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l528_52838

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g 0| = 10 ∧ |g 1| = 10 ∧ |g 3| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 70 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l528_52838


namespace NUMINAMATH_CALUDE_symmetry_classification_l528_52859

-- Define the shape type
inductive Shape
  | Parallelogram
  | Rectangle
  | RightTrapezoid
  | Square
  | EquilateralTriangle
  | LineSegment

-- Define properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.EquilateralTriangle => True
  | Shape.LineSegment => True
  | _ => False

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.Rectangle => True
  | Shape.Square => True
  | Shape.LineSegment => True
  | _ => False

-- Theorem statement
theorem symmetry_classification (s : Shape) :
  (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔
  (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.LineSegment) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_classification_l528_52859


namespace NUMINAMATH_CALUDE_unique_P_value_l528_52833

theorem unique_P_value (x y P : ℤ) : 
  x > 0 → y > 0 → x + y = P → 3 * x + 5 * y = 13 → P = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_P_value_l528_52833


namespace NUMINAMATH_CALUDE_total_marbles_is_76_l528_52875

/-- The total number of marbles in an arrangement where 9 rows have 8 marbles each and 1 row has 4 marbles -/
def total_marbles : ℕ := 
  let rows_with_eight := 9
  let marbles_per_row_eight := 8
  let rows_with_four := 1
  let marbles_per_row_four := 4
  rows_with_eight * marbles_per_row_eight + rows_with_four * marbles_per_row_four

/-- Theorem stating that the total number of marbles is 76 -/
theorem total_marbles_is_76 : total_marbles = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_76_l528_52875


namespace NUMINAMATH_CALUDE_find_y_l528_52889

theorem find_y (n x y : ℝ) : 
  (100 + 200 + n + x) / 4 = 250 ∧ 
  (n + 150 + 100 + x + y) / 5 = 200 → 
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_l528_52889


namespace NUMINAMATH_CALUDE_P_has_negative_and_positive_roots_l528_52853

def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 9

theorem P_has_negative_and_positive_roots :
  (∃ (a : ℝ), a < 0 ∧ P a = 0) ∧ (∃ (b : ℝ), b > 0 ∧ P b = 0) := by sorry

end NUMINAMATH_CALUDE_P_has_negative_and_positive_roots_l528_52853


namespace NUMINAMATH_CALUDE_rainfall_water_level_rise_l528_52816

/-- Given 15 liters of rainfall per square meter, the rise in water level in a pool is 1.5 cm. -/
theorem rainfall_water_level_rise :
  let rainfall_per_sqm : ℝ := 15  -- liters per square meter
  let liters_to_cubic_cm : ℝ := 1000  -- 1 liter = 1000 cm³
  let sqm_to_sqcm : ℝ := 10000  -- 1 m² = 10000 cm²
  rainfall_per_sqm * liters_to_cubic_cm / sqm_to_sqcm = 1.5  -- cm
  := by sorry

end NUMINAMATH_CALUDE_rainfall_water_level_rise_l528_52816


namespace NUMINAMATH_CALUDE_camp_participants_equality_l528_52844

structure CampParticipants where
  mathOrange : ℕ
  mathPurple : ℕ
  physicsOrange : ℕ
  physicsPurple : ℕ

theorem camp_participants_equality (p : CampParticipants) 
  (h : p.physicsOrange = p.mathPurple) : 
  p.mathOrange + p.mathPurple = p.mathOrange + p.physicsOrange :=
by
  sorry

#check camp_participants_equality

end NUMINAMATH_CALUDE_camp_participants_equality_l528_52844


namespace NUMINAMATH_CALUDE_equation_solution_l528_52823

theorem equation_solution : ∃ b : ℝ, ∀ a : ℝ, (-6) * a^2 = 3 * (4*a + b) ∧ (a = 1 → b = -6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l528_52823


namespace NUMINAMATH_CALUDE_total_lunch_is_fifteen_l528_52899

/-- The total amount spent on lunch given the conditions -/
def total_lunch_amount (your_amount : ℕ) (friend_amount : ℕ) : ℕ :=
  your_amount + friend_amount

/-- Theorem: The total amount spent on lunch is $15 -/
theorem total_lunch_is_fifteen :
  ∃ (your_amount : ℕ),
    (your_amount + 1 = 8) →
    (total_lunch_amount your_amount 8 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_total_lunch_is_fifteen_l528_52899


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l528_52804

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1| + |a * x - 3 * a|

-- Part 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} :=
sorry

-- Part 2: Range of a when solution set is ℝ
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 5) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l528_52804


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l528_52885

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (ρ : ℝ) (h : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l528_52885


namespace NUMINAMATH_CALUDE_rhombus_area_l528_52895

/-- The area of a rhombus with side length 4 cm and an acute angle of 45° is 8 cm². -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → acute_angle = π / 4 → 
  (side_length * side_length * Real.sin acute_angle) = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l528_52895


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_dishonest_dealer_profit_result_l528_52812

/-- Calculates the overall percent profit for a dishonest dealer selling two products. -/
theorem dishonest_dealer_profit (weight_A weight_B : ℝ) (cost_A cost_B : ℝ) 
  (discount_A discount_B : ℝ) (purchase_A purchase_B : ℝ) : ℝ :=
  let actual_weight_A := weight_A / 1000 * purchase_A
  let actual_weight_B := weight_B / 1000 * purchase_B
  let cost_price_A := actual_weight_A * cost_A
  let cost_price_B := actual_weight_B * cost_B
  let total_cost_price := cost_price_A + cost_price_B
  let selling_price_A := purchase_A * cost_A
  let selling_price_B := purchase_B * cost_B
  let discounted_price_A := selling_price_A * (1 - discount_A)
  let discounted_price_B := selling_price_B * (1 - discount_B)
  let total_selling_price := discounted_price_A + discounted_price_B
  let profit := total_selling_price - total_cost_price
  let percent_profit := profit / total_cost_price * 100
  percent_profit

/-- The overall percent profit is approximately 30.99% -/
theorem dishonest_dealer_profit_result : 
  ∃ ε > 0, |dishonest_dealer_profit 700 750 60 80 0.05 0.03 6 12 - 30.99| < ε :=
sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_dishonest_dealer_profit_result_l528_52812


namespace NUMINAMATH_CALUDE_plane_equation_from_perpendicular_foot_l528_52854

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (c : PlaneCoefficients) : Prop :=
  c.A * p.x + c.B * p.y + c.C * p.z + c.D = 0

/-- Check if a vector is perpendicular to a plane -/
def vectorPerpendicularToPlane (v : Point3D) (c : PlaneCoefficients) : Prop :=
  ∃ (k : ℝ), v.x = k * c.A ∧ v.y = k * c.B ∧ v.z = k * c.C

/-- The main theorem -/
theorem plane_equation_from_perpendicular_foot : 
  ∃ (c : PlaneCoefficients),
    c.A = 4 ∧ c.B = -3 ∧ c.C = 1 ∧ c.D = -52 ∧
    pointOnPlane ⟨8, -6, 2⟩ c ∧
    vectorPerpendicularToPlane ⟨8, -6, 2⟩ c ∧
    c.A > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs c.A) (Int.natAbs c.B)) (Int.natAbs c.C)) (Int.natAbs c.D) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_from_perpendicular_foot_l528_52854


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l528_52842

/-- Given an escalator moving upwards at a certain rate with a specified length,
    prove that a person walking on it at a certain rate will take a specific time
    to cover the entire length. -/
theorem escalator_walking_rate
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 196)
  (h3 : time_taken = 14)
  : ∃ (walking_rate : ℝ),
    escalator_length = (walking_rate + escalator_speed) * time_taken ∧
    walking_rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l528_52842


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_a_eq_one_l528_52849

theorem pure_imaginary_iff_a_eq_one (a : ℝ) :
  (∃ b : ℝ, Complex.mk (a^2 - 1) (a + 1) = Complex.I * b) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_a_eq_one_l528_52849


namespace NUMINAMATH_CALUDE_fourth_power_sum_of_cubic_roots_l528_52821

theorem fourth_power_sum_of_cubic_roots (a b c : ℝ) : 
  (a^3 - 3*a + 1 = 0) → 
  (b^3 - 3*b + 1 = 0) → 
  (c^3 - 3*c + 1 = 0) → 
  a^4 + b^4 + c^4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_of_cubic_roots_l528_52821


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l528_52890

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l528_52890


namespace NUMINAMATH_CALUDE_shifted_function_sum_l528_52850

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 6)

-- Define a, b, and c
def a : ℝ := 3
def b : ℝ := 38
def c : ℝ := 115

theorem shifted_function_sum (x : ℝ) : g x = a * x^2 + b * x + c ∧ a + b + c = 156 := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_sum_l528_52850


namespace NUMINAMATH_CALUDE_total_profit_is_5400_l528_52876

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 5400 given the specified conditions -/
theorem total_profit_is_5400 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 4500)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 3000) :
  total_profit ps = 5400 :=
sorry

end NUMINAMATH_CALUDE_total_profit_is_5400_l528_52876


namespace NUMINAMATH_CALUDE_distinct_remainders_l528_52871

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2^(sequence_a n) + sequence_a n

theorem distinct_remainders (n m : ℕ) (hn : n < 243) (hm : m < 243) (hnm : n ≠ m) :
  sequence_a n % 243 ≠ sequence_a m % 243 := by
  sorry

end NUMINAMATH_CALUDE_distinct_remainders_l528_52871


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l528_52882

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (a b : ℝ) :
  let P := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualX rotated.1 rotated.2
  final = (5, -4) →
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l528_52882


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l528_52809

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Define the distances |PF₁| and |PF₂|
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_area :
  hyperbola P.1 P.2 →
  PF₁ / PF₂ = 3 / 4 →
  (1/2) * ‖F₁ - F₂‖ * ‖P - (F₁ + F₂)/2‖ = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l528_52809


namespace NUMINAMATH_CALUDE_expression_value_l528_52881

theorem expression_value (a b c k : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hk : k = 2) :
  (a - (b - k * c)) - ((a - b) - k * c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l528_52881


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_existence_condition_l528_52829

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} :=
sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ (-7 < a ∧ a < -1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_existence_condition_l528_52829


namespace NUMINAMATH_CALUDE_line_circle_intersection_l528_52879

-- Define the line l and circle C
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Theorem statement
theorem line_circle_intersection :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  (∃ (x y : ℝ), line_l k x y ∧ circle_C x y) →
  (∀ (x y : ℝ), line_l k x y → x = 1 ∧ y = 1) ∧
  (∃ (chord_length : ℝ), chord_length = Real.sqrt 8 ∧ 
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≥ chord_length^2) ∧
  (∃ (max_chord : ℝ), max_chord = 4 ∧
    ∀ (x y : ℝ), line_l k x y ∧ circle_C x y → 
      (x - A.1)^2 + (y - A.2)^2 ≤ max_chord^2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l528_52879


namespace NUMINAMATH_CALUDE_distributions_without_zhoubi_l528_52825

/-- Represents the number of books -/
def num_books : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 3

/-- Represents the total number of distribution methods -/
def total_distributions : ℕ := 36

/-- Represents the number of distribution methods where student A receives "Zhoubi Suanjing" -/
def distributions_with_zhoubi : ℕ := 12

/-- Theorem stating the number of distribution methods where A does not receive "Zhoubi Suanjing" -/
theorem distributions_without_zhoubi :
  total_distributions - distributions_with_zhoubi = 24 :=
by sorry

end NUMINAMATH_CALUDE_distributions_without_zhoubi_l528_52825


namespace NUMINAMATH_CALUDE_last_two_digits_squares_l528_52830

theorem last_two_digits_squares (a b : ℕ) :
  (50 ∣ (a + b) ∨ 50 ∣ (a - b)) → a^2 ≡ b^2 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_squares_l528_52830


namespace NUMINAMATH_CALUDE_power_inequality_l528_52865

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l528_52865


namespace NUMINAMATH_CALUDE_smallest_n_value_l528_52896

def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  c = 710 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ p, p > 0 ∧ ¬(10 ∣ p) ∧ 
    a.factorial * b.factorial * c.factorial ≠ p * (10 ^ k)) →
  n = 500 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l528_52896


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l528_52873

/-- Calculates the discount percentage on a shirt given the original prices,
    total paid, and discount on the jacket. -/
theorem shirt_discount_percentage
  (jacket_price : ℝ)
  (shirt_price : ℝ)
  (total_paid : ℝ)
  (jacket_discount : ℝ)
  (h1 : jacket_price = 100)
  (h2 : shirt_price = 60)
  (h3 : total_paid = 110)
  (h4 : jacket_discount = 0.3)
  : (1 - (total_paid - jacket_price * (1 - jacket_discount)) / shirt_price) * 100 = 100 / 3 := by
  sorry

#eval (1 - (110 - 100 * (1 - 0.3)) / 60) * 100

end NUMINAMATH_CALUDE_shirt_discount_percentage_l528_52873


namespace NUMINAMATH_CALUDE_correct_calculation_l528_52831

theorem correct_calculation (a b : ℝ) : 6 * a^2 * b - b * a^2 = 5 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l528_52831


namespace NUMINAMATH_CALUDE_intersection_A_B_l528_52851

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l528_52851


namespace NUMINAMATH_CALUDE_triangle_area_in_rectangle_l528_52806

/-- Given a rectangle with dimensions 30 cm by 28 cm containing four congruent right-angled triangles,
    where the hypotenuse of each triangle forms part of the rectangle's perimeter,
    the total area of the four triangles is 56 cm². -/
theorem triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a + 2 * b = 30 →
  2 * b = 28 →
  4 * (1/2 * a * b) = 56 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_in_rectangle_l528_52806


namespace NUMINAMATH_CALUDE_percentage_problem_l528_52866

theorem percentage_problem (x : ℝ) (a : ℝ) (h1 : (x / 100) * 170 = 85) (h2 : a = 170) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l528_52866


namespace NUMINAMATH_CALUDE_final_state_only_beads_l528_52803

/-- Represents the types of items in the exchange system -/
inductive Item
  | Gold
  | Pearl
  | Bead

/-- Represents the state of items in the exchange system -/
structure ItemState :=
  (gold : ℕ)
  (pearl : ℕ)
  (bead : ℕ)

/-- Represents an exchange rule -/
structure ExchangeRule :=
  (input1 : Item)
  (input2 : Item)
  (output : Item)

/-- Applies an exchange rule to the current state -/
def applyExchange (state : ItemState) (rule : ExchangeRule) : ItemState :=
  sorry

/-- Checks if an exchange rule can be applied to the current state -/
def canApplyExchange (state : ItemState) (rule : ExchangeRule) : Prop :=
  sorry

/-- Represents the exchange system with initial state and rules -/
structure ExchangeSystem :=
  (initialState : ItemState)
  (rules : List ExchangeRule)

/-- Defines the final state after all possible exchanges -/
def finalState (system : ExchangeSystem) : ItemState :=
  sorry

/-- Theorem: The final state after all exchanges will only have beads -/
theorem final_state_only_beads (system : ExchangeSystem) :
  system.initialState = ItemState.mk 24 26 25 →
  system.rules = [
    ExchangeRule.mk Item.Gold Item.Pearl Item.Bead,
    ExchangeRule.mk Item.Gold Item.Bead Item.Pearl,
    ExchangeRule.mk Item.Pearl Item.Bead Item.Gold
  ] →
  ∃ n : ℕ, finalState system = ItemState.mk 0 0 n :=
sorry

end NUMINAMATH_CALUDE_final_state_only_beads_l528_52803


namespace NUMINAMATH_CALUDE_fourth_sample_is_twenty_l528_52878

/-- Represents a systematic sampling scheme. -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Generates the nth sample number in a systematic sampling scheme. -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * s.interval

/-- The theorem stating that 20 is the fourth sample number. -/
theorem fourth_sample_is_twenty
  (total_students : ℕ)
  (h_total : total_students = 56)
  (sample : SystematicSample)
  (h_population : sample.population = total_students)
  (h_sample_size : sample.sample_size = 4)
  (h_first_sample : sample.first_sample = 6)
  (h_interval : sample.interval = total_students / sample.sample_size)
  (h_third_sample : nth_sample sample 3 = 34)
  (h_fourth_sample : nth_sample sample 4 = 48) :
  nth_sample sample 2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_twenty_l528_52878


namespace NUMINAMATH_CALUDE_cos_negative_120_degrees_l528_52893

theorem cos_negative_120_degrees : Real.cos (-(120 * Real.pi / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_120_degrees_l528_52893


namespace NUMINAMATH_CALUDE_water_bottle_cost_l528_52818

/-- Proves that the cost of a water bottle is $2 given the conditions of Adam's shopping trip. -/
theorem water_bottle_cost (num_sandwiches : ℕ) (sandwich_price total_cost : ℚ) : 
  num_sandwiches = 3 →
  sandwich_price = 3 →
  total_cost = 11 →
  total_cost - (num_sandwiches : ℚ) * sandwich_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l528_52818


namespace NUMINAMATH_CALUDE_inequality_solution_set_l528_52852

theorem inequality_solution_set :
  ∀ x : ℝ, (7 / 30 + |x - 7 / 60| < 11 / 20) ↔ (-1 / 5 < x ∧ x < 13 / 30) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l528_52852


namespace NUMINAMATH_CALUDE_function_period_l528_52880

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) = f (2 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (5 + x) = f (5 - x)

-- Define the period
def is_period (T : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_period (f : ℝ → ℝ) 
  (h1 : condition1 f) (h2 : condition2 f) : 
  (∃ T : ℝ, T > 0 ∧ is_period T f ∧ ∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) ∧
  (∀ T : ℝ, T > 0 ∧ is_period T f ∧ (∀ S : ℝ, S > 0 ∧ is_period S f → T ≤ S) → T = 6) :=
sorry

end NUMINAMATH_CALUDE_function_period_l528_52880


namespace NUMINAMATH_CALUDE_tony_age_in_six_years_l528_52815

/-- Given Jacob's current age and Tony's age relative to Jacob's, 
    calculate Tony's age after a certain number of years. -/
def tony_future_age (jacob_age : ℕ) (years_passed : ℕ) : ℕ :=
  (jacob_age / 2) + years_passed

/-- Theorem: Tony will be 18 years old in 6 years -/
theorem tony_age_in_six_years :
  tony_future_age 24 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tony_age_in_six_years_l528_52815


namespace NUMINAMATH_CALUDE_regular_icosahedron_edges_l528_52892

/-- A regular icosahedron is a convex polyhedron with 20 faces, each of which is an equilateral triangle. -/
structure RegularIcosahedron :=
  (faces : Nat)
  (face_shape : String)
  (is_convex : Bool)
  (h_faces : faces = 20)
  (h_face_shape : face_shape = "equilateral triangle")
  (h_convex : is_convex = true)

/-- The number of edges in a regular icosahedron -/
def num_edges (i : RegularIcosahedron) : Nat := 30

/-- Theorem: A regular icosahedron has 30 edges -/
theorem regular_icosahedron_edges (i : RegularIcosahedron) : num_edges i = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_icosahedron_edges_l528_52892


namespace NUMINAMATH_CALUDE_unique_f_exists_l528_52869

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function f(n) to be proved unique -/
def f (n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem unique_f_exists (n : ℕ) (h1 : n > 1) (h2 : n ≠ 10) :
  ∃! fn : ℕ, fn ≥ 2 ∧ ∀ k : ℕ, 0 < k → k < fn →
    sum_of_digits k + sum_of_digits (fn - k) = n :=
sorry

end NUMINAMATH_CALUDE_unique_f_exists_l528_52869


namespace NUMINAMATH_CALUDE_no_four_consecutive_lucky_numbers_l528_52840

/-- A function that checks if a number is lucky -/
def is_lucky (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n ≤ 9999999 ∧ 
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    a ≠ 0 ∧ (n % (a * b * c * d * e * f * g) = 0)

/-- Theorem stating that four consecutive lucky numbers do not exist -/
theorem no_four_consecutive_lucky_numbers :
  ¬ ∃ (n : ℕ), is_lucky n ∧ is_lucky (n + 1) ∧ is_lucky (n + 2) ∧ is_lucky (n + 3) :=
sorry

end NUMINAMATH_CALUDE_no_four_consecutive_lucky_numbers_l528_52840


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l528_52872

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
    is 156 and the distance from one corner to the farthest corner is 25, then its total surface area is 896. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 156)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + c * a) = 896 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l528_52872


namespace NUMINAMATH_CALUDE_marbles_lost_l528_52802

theorem marbles_lost (initial_marbles remaining_marbles : ℕ) 
  (h1 : initial_marbles = 16) 
  (h2 : remaining_marbles = 9) : 
  initial_marbles - remaining_marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l528_52802


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l528_52886

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - 2*x^5 - 3*x^4 + 4*x^3 + 5*x^2 - x - 2
def g (x : ℝ) : ℝ := (x-3)*(x^2-1)

-- Define the remainder
def r (x : ℝ) : ℝ := 18*x^2 + x - 17

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l528_52886


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l528_52891

/-- The perimeter of a hexagon with side length 7 inches is 42 inches. -/
theorem hexagon_perimeter : 
  ∀ (hexagon_side_length : ℝ), 
  hexagon_side_length = 7 → 
  6 * hexagon_side_length = 42 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l528_52891


namespace NUMINAMATH_CALUDE_tree_planting_optimization_l528_52817

/-- Tree planting activity optimization problem -/
theorem tree_planting_optimization (total_families : ℕ) 
  (silver_poplars : ℕ) (purple_plums : ℕ) 
  (silver_poplar_time : ℚ) (purple_plum_time : ℚ) :
  total_families = 65 →
  silver_poplars = 150 →
  purple_plums = 160 →
  silver_poplar_time = 2/5 →
  purple_plum_time = 3/5 →
  ∃ (group_a_families : ℕ) (duration : ℚ),
    group_a_families = 25 ∧
    duration = 12/5 ∧
    group_a_families ≤ total_families ∧
    (group_a_families : ℚ) * silver_poplars * silver_poplar_time = 
      (total_families - group_a_families : ℚ) * purple_plums * purple_plum_time ∧
    duration = (group_a_families : ℚ) * silver_poplars * silver_poplar_time / group_a_families ∧
    ∀ (other_group_a : ℕ) (other_duration : ℚ),
      other_group_a ≤ total_families →
      (other_group_a : ℚ) * silver_poplars * silver_poplar_time = 
        (total_families - other_group_a : ℚ) * purple_plums * purple_plum_time →
      other_duration = (other_group_a : ℚ) * silver_poplars * silver_poplar_time / other_group_a →
      duration ≤ other_duration :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_optimization_l528_52817


namespace NUMINAMATH_CALUDE_difference_in_circumferences_l528_52824

/-- The difference in circumferences of two concentric circles -/
theorem difference_in_circumferences 
  (inner_diameter : ℝ) 
  (track_width : ℝ) 
  (h1 : inner_diameter = 50) 
  (h2 : track_width = 15) : 
  (inner_diameter + 2 * track_width) * π - inner_diameter * π = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_difference_in_circumferences_l528_52824


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l528_52898

theorem percentage_of_percentage (x : ℝ) (h : x ≠ 0) :
  (60 / 100) * (30 / 100) * x = (18 / 100) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l528_52898


namespace NUMINAMATH_CALUDE_largest_non_sum_of_100_composites_l528_52836

/-- A number is composite if it's the product of two integers greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number can be expressed as the sum of 100 composite numbers -/
def IsSumOf100Composites (n : ℕ) : Prop :=
  ∃ (f : Fin 100 → ℕ), (∀ i, IsComposite (f i)) ∧ n = (Finset.univ.sum f)

/-- 403 is the largest integer that cannot be expressed as the sum of 100 composites -/
theorem largest_non_sum_of_100_composites :
  (¬ IsSumOf100Composites 403) ∧ (∀ n > 403, IsSumOf100Composites n) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_100_composites_l528_52836


namespace NUMINAMATH_CALUDE_total_sword_weight_l528_52857

/-- The number of squads the Dark Lord has -/
def num_squads : ℕ := 10

/-- The number of orcs in each squad -/
def orcs_per_squad : ℕ := 8

/-- The weight of swords each orc carries (in pounds) -/
def sword_weight_per_orc : ℕ := 15

/-- Theorem stating the total weight of swords to be transported -/
theorem total_sword_weight : 
  num_squads * orcs_per_squad * sword_weight_per_orc = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_sword_weight_l528_52857


namespace NUMINAMATH_CALUDE_max_sum_xy_l528_52843

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1 ∧ (x ≠ 0 ∨ y ≠ 0) ∧ x^2 + y^2 ≠ 2

-- State the theorem
theorem max_sum_xy :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → x + y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ x + y = max) :=
sorry

end NUMINAMATH_CALUDE_max_sum_xy_l528_52843


namespace NUMINAMATH_CALUDE_robins_camera_pictures_l528_52801

theorem robins_camera_pictures :
  ∀ (phone_pics camera_pics total_pics albums pics_per_album : ℕ),
    phone_pics = 35 →
    albums = 5 →
    pics_per_album = 8 →
    total_pics = albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    camera_pics = 5 := by
  sorry

end NUMINAMATH_CALUDE_robins_camera_pictures_l528_52801


namespace NUMINAMATH_CALUDE_trigonometric_identities_l528_52811

theorem trigonometric_identities (α : Real) 
  (h : 3 * Real.sin α - 2 * Real.cos α = 0) : 
  (((Real.cos α - Real.sin α) / (Real.cos α + Real.sin α)) + 
   ((Real.cos α + Real.sin α) / (Real.cos α - Real.sin α)) = 6) ∧ 
  ((Real.sin α)^2 - 2 * Real.sin α * Real.cos α + 4 * (Real.cos α)^2 = 28/13) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l528_52811


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l528_52819

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 1224 → n + (n + 1) = -69 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l528_52819


namespace NUMINAMATH_CALUDE_prob_reroll_two_dice_l528_52862

-- Define a die as a natural number between 1 and 6
def Die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define a roll of four dice
def FourDiceRoll := Die × Die × Die × Die

-- Function to calculate the sum of four dice
def diceSum (roll : FourDiceRoll) : ℕ := roll.1 + roll.2.1 + roll.2.2.1 + roll.2.2.2

-- Function to determine if a roll is a win (sum is 9)
def isWin (roll : FourDiceRoll) : Prop := diceSum roll = 9

-- Function to calculate the probability of winning by rerolling all four dice
def probWinRerollAll : ℚ := 56 / 1296

-- Function to calculate the probability of winning by rerolling two dice
def probWinRerollTwo (keptSum : ℕ) : ℚ :=
  if keptSum ≤ 7 then (9 - keptSum - 1) / 36
  else (13 - (9 - keptSum)) / 36

-- Theorem: The probability of Jason choosing to reroll exactly two dice is 1/18
theorem prob_reroll_two_dice : 
  (∀ roll : FourDiceRoll, ∃ (keptSum : ℕ), 
    (keptSum ≤ 8 ∧ probWinRerollTwo keptSum > probWinRerollAll) ∨
    (keptSum > 8 ∧ probWinRerollAll ≥ probWinRerollTwo keptSum)) →
  (2 : ℚ) / 36 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_reroll_two_dice_l528_52862


namespace NUMINAMATH_CALUDE_mushroom_count_l528_52814

theorem mushroom_count :
  ∀ n m : ℕ,
  n ≤ 70 →
  m = (52 * n) / 100 →
  ∃ x : ℕ,
  x ≤ 3 ∧
  2 * (m - x) = n - 3 →
  n = 25 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_count_l528_52814


namespace NUMINAMATH_CALUDE_product_repeating_nine_and_nine_l528_52855

/-- The repeating decimal 0.999... -/
def repeating_decimal_nine : ℚ := 1

theorem product_repeating_nine_and_nine :
  repeating_decimal_nine * 9 = 9 := by sorry

end NUMINAMATH_CALUDE_product_repeating_nine_and_nine_l528_52855


namespace NUMINAMATH_CALUDE_wolf_prize_laureates_l528_52835

theorem wolf_prize_laureates (total_scientists : ℕ) 
                              (both_wolf_and_nobel : ℕ) 
                              (total_nobel : ℕ) 
                              (h1 : total_scientists = 50)
                              (h2 : both_wolf_and_nobel = 16)
                              (h3 : total_nobel = 27)
                              (h4 : total_nobel - both_wolf_and_nobel = 
                                    (total_scientists - wolf_laureates - (total_nobel - both_wolf_and_nobel)) + 3) :
  wolf_laureates = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_wolf_prize_laureates_l528_52835


namespace NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l528_52868

def is_special_fraction (a b : ℕ+) : Prop := a.val + b.val = 18

def sum_of_special_fractions (n : ℤ) : Prop :=
  ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
    is_special_fraction a₁ b₁ ∧ 
    is_special_fraction a₂ b₂ ∧ 
    n = (a₁.val : ℤ) * b₂.val + (a₂.val : ℤ) * b₁.val

theorem count_distinct_sums_of_special_fractions : 
  ∃! (s : Finset ℤ), 
    (∀ n, n ∈ s ↔ sum_of_special_fractions n) ∧ 
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l528_52868


namespace NUMINAMATH_CALUDE_video_subscription_duration_l528_52874

theorem video_subscription_duration (monthly_cost : ℚ) (total_paid : ℚ) : 
  monthly_cost = 14 →
  total_paid = 84 →
  (total_paid / (monthly_cost / 2)) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_video_subscription_duration_l528_52874


namespace NUMINAMATH_CALUDE_a_less_than_2_necessary_not_sufficient_l528_52820

theorem a_less_than_2_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < 2*a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 2*a) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_2_necessary_not_sufficient_l528_52820


namespace NUMINAMATH_CALUDE_garment_costs_l528_52845

/-- The cost of garment A -/
def cost_A : ℝ := 300

/-- The cost of garment B -/
def cost_B : ℝ := 200

/-- The total cost of garments A and B -/
def total_cost : ℝ := 500

/-- The profit margin for garment A -/
def profit_margin_A : ℝ := 0.3

/-- The profit margin for garment B -/
def profit_margin_B : ℝ := 0.2

/-- The total profit -/
def total_profit : ℝ := 130

/-- Theorem: Given the conditions, the costs of garments A and B are 300 yuan and 200 yuan respectively -/
theorem garment_costs : 
  cost_A + cost_B = total_cost ∧ 
  profit_margin_A * cost_A + profit_margin_B * cost_B = total_profit := by
  sorry

end NUMINAMATH_CALUDE_garment_costs_l528_52845


namespace NUMINAMATH_CALUDE_least_possible_value_of_x_l528_52827

theorem least_possible_value_of_x : 
  ∃ (x y z : ℤ), 
    (∃ k : ℤ, x = 2 * k) ∧ 
    (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) ∧ 
    y - x > 5 ∧ 
    (∀ w : ℤ, w - x ≥ 9 → w ≥ z) ∧ 
    (∀ v : ℤ, (∃ j : ℤ, v = 2 * j) → v ≥ x) → 
    x = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_of_x_l528_52827


namespace NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l528_52822

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (3 * n) / 8 → 
  hats = (5 * n) / 6 → 
  both ≥ gloves + hats - n → 
  both ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l528_52822


namespace NUMINAMATH_CALUDE_two_equidistant_points_l528_52813

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and a point on the line -/
structure Line where
  normal : ℝ × ℝ
  point : ℝ × ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleTangentConfig where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d1 : ℝ  -- distance from circle center to tangent1
  d2 : ℝ  -- distance from circle center to tangent2

/-- Predicate to check if a point is equidistant from a circle and two lines -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l1 l2 : Line) : Prop := sorry

/-- Main theorem: There are exactly two points equidistant from the circle and both tangents -/
theorem two_equidistant_points (config : CircleTangentConfig) 
  (h1 : config.d1 ≠ config.d2)
  (h2 : config.d1 > config.circle.radius)
  (h3 : config.d2 > config.circle.radius)
  (h4 : config.tangent1.normal = config.tangent2.normal) :  -- parallel tangents
  ∃! (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    isEquidistant p1 config.circle config.tangent1 config.tangent2 ∧ 
    isEquidistant p2 config.circle config.tangent1 config.tangent2 := by
  sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l528_52813


namespace NUMINAMATH_CALUDE_smaller_number_problem_l528_52800

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 14 → min x y = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l528_52800


namespace NUMINAMATH_CALUDE_statement_C_is_false_l528_52861

def f (x : ℝ) := 3 - 4*x - 2*x^2

theorem statement_C_is_false :
  ¬(∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f x ≤ max ∧ f x ≥ min) ∧ 
    (max = -3) ∧ (min = -13) ∧
    (∃ x₁ ∈ Set.Icc 1 2, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc 1 2, f x₂ = min)) :=
by
  sorry


end NUMINAMATH_CALUDE_statement_C_is_false_l528_52861


namespace NUMINAMATH_CALUDE_ryan_bus_trips_l528_52846

/-- Represents Ryan's commuting schedule and times --/
structure CommuteSchedule where
  bike_time : ℕ
  bus_time : ℕ
  friend_time : ℕ
  bike_frequency : ℕ
  friend_frequency : ℕ
  total_time : ℕ

/-- Calculates the number of bus trips given a CommuteSchedule --/
def calculate_bus_trips (schedule : CommuteSchedule) : ℕ :=
  (schedule.total_time - 
   (schedule.bike_time * schedule.bike_frequency + 
    schedule.friend_time * schedule.friend_frequency)) / 
  schedule.bus_time

/-- Ryan's actual commute schedule --/
def ryan_schedule : CommuteSchedule :=
  { bike_time := 30
  , bus_time := 40
  , friend_time := 10
  , bike_frequency := 1
  , friend_frequency := 1
  , total_time := 160 }

/-- Theorem stating that Ryan takes the bus 3 times a week --/
theorem ryan_bus_trips : calculate_bus_trips ryan_schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_ryan_bus_trips_l528_52846


namespace NUMINAMATH_CALUDE_distance_to_place_l528_52867

/-- The distance to the place given the man's rowing speed, river speed, and total time -/
theorem distance_to_place (mans_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  mans_speed = 4 →
  river_speed = 2 →
  total_time = 1.5 →
  (1 / (mans_speed + river_speed) + 1 / (mans_speed - river_speed)) * total_time = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l528_52867


namespace NUMINAMATH_CALUDE_cookie_count_indeterminate_l528_52847

theorem cookie_count_indeterminate (total_bananas : ℕ) (num_boxes : ℕ) (bananas_per_box : ℕ) 
  (h1 : total_bananas = 40)
  (h2 : num_boxes = 8)
  (h3 : bananas_per_box = 5)
  (h4 : total_bananas = num_boxes * bananas_per_box) :
  ¬ ∃ (cookie_count : ℕ), ∀ (n : ℕ), cookie_count = n :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_count_indeterminate_l528_52847


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_numerator_increase_proof_l528_52887

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (resulting_fraction : ℚ) : ℚ :=
  let numerator_increase := 
    (resulting_fraction * (1 - denominator_decrease / 100) / original_fraction - 1) * 100
  numerator_increase

#check numerator_increase_percentage (3/4) 8 (15/16) = 15

theorem numerator_increase_proof :
  numerator_increase_percentage (3/4) 8 (15/16) = 15 := by sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_numerator_increase_proof_l528_52887


namespace NUMINAMATH_CALUDE_inequality_system_solution_l528_52807

/-- Given that the solution set of the inequality system {x + 1 > 2x - 2, x < a} is x < 3,
    prove that the range of values for a is a ≥ 3. -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 1 > 2*x - 2 ∧ x < a) ↔ x < 3) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l528_52807


namespace NUMINAMATH_CALUDE_original_price_l528_52856

/-- Given an article with price changes and final price, calculate the original price -/
theorem original_price (q r : ℚ) : 
  (∃ (x : ℚ), x * (1 + q / 100) * (1 - r / 100) = 2) →
  (∃ (x : ℚ), x = 200 / (100 + q - r - q * r / 100)) :=
by sorry

end NUMINAMATH_CALUDE_original_price_l528_52856


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l528_52858

theorem floor_sqrt_eight_count : 
  (Finset.range 81 \ Finset.range 64).card = 17 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l528_52858


namespace NUMINAMATH_CALUDE_salad_price_proof_l528_52888

/-- Proves that the price of each salad is $2.50 given the problem conditions --/
theorem salad_price_proof (hot_dog_price : ℝ) (hot_dog_count : ℕ) (salad_count : ℕ) 
  (payment : ℝ) (change : ℝ) :
  hot_dog_price = 1.5 →
  hot_dog_count = 5 →
  salad_count = 3 →
  payment = 20 →
  change = 5 →
  (payment - change - hot_dog_price * hot_dog_count) / salad_count = 2.5 := by
sorry

#eval (20 - 5 - 1.5 * 5) / 3

end NUMINAMATH_CALUDE_salad_price_proof_l528_52888


namespace NUMINAMATH_CALUDE_determine_c_l528_52841

/-- Given integers a and b, where there exist unique x, y, z satisfying the LCM conditions,
    the value of c can be uniquely determined. -/
theorem determine_c (a b : ℕ) 
  (h_exists : ∃! (x y z : ℕ), a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ ∃ c, c = Nat.lcm x y) :
  ∃! c, ∀ (x y z : ℕ), 
    (a = Nat.lcm y z ∧ b = Nat.lcm x z ∧ c = Nat.lcm x y) → 
    (∀ (x' y' z' : ℕ), a = Nat.lcm y' z' ∧ b = Nat.lcm x' z' → (x = x' ∧ y = y' ∧ z = z')) :=
by sorry


end NUMINAMATH_CALUDE_determine_c_l528_52841


namespace NUMINAMATH_CALUDE_remainder_of_n_l528_52808

theorem remainder_of_n (n : ℕ) (h1 : n^3 % 7 = 3) (h2 : n^4 % 7 = 2) : n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l528_52808


namespace NUMINAMATH_CALUDE_seating_arrangement_l528_52877

theorem seating_arrangement (total_people : ℕ) (row_sizes : List ℕ) : 
  total_people = 65 →
  (∀ x ∈ row_sizes, x = 7 ∨ x = 8 ∨ x = 9) →
  (List.sum row_sizes = total_people) →
  (List.count 9 row_sizes = 1) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_l528_52877


namespace NUMINAMATH_CALUDE_regular_ngon_inscribed_circle_l528_52897

theorem regular_ngon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (n : ℝ) / 2 * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_ngon_inscribed_circle_l528_52897


namespace NUMINAMATH_CALUDE_equation_equality_l528_52894

theorem equation_equality (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) 
  (hac : a + c = 10) : 
  (10 * b + a) * (10 * b + c) = 100 * b * (b + 1) + a * c := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l528_52894


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l528_52839

-- Define the polynomials h(x) and k(x)
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + x + 15
def k (q s : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + s

-- State the theorem
theorem polynomial_root_relation (p q s : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →
  (∀ x : ℝ, h p x = 0 → k q s x = 0) →
  k q s 1 = -16048 :=
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l528_52839


namespace NUMINAMATH_CALUDE_quadratic_inequality_l528_52810

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of f ≥ 0
def solution_set (a b c : ℝ) : Set ℝ := {x | x ≤ -3 ∨ x ≥ 4}

-- Define the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x ≥ 0) :
  a > 0 ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l528_52810


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l528_52848

def A : Set ℝ := {x | 2 * x - x^2 > 0}
def B : Set ℝ := {x | x > 1}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l528_52848


namespace NUMINAMATH_CALUDE_spaceship_travel_distance_l528_52863

def earth_to_x : ℝ := 0.5
def x_to_y : ℝ := 0.1
def y_to_earth : ℝ := 0.1

theorem spaceship_travel_distance :
  earth_to_x + x_to_y + y_to_earth = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_travel_distance_l528_52863


namespace NUMINAMATH_CALUDE_problem_solution_l528_52834

def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - m^4 ≤ 0

theorem problem_solution (m : ℝ) :
  (∀ x, q x m → p x) → (m ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ x, ¬(q x m) → ¬(p x)) → (m ≥ 3 ∨ m ≤ -3) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m))) → (m ≥ 3 ∨ m ≤ -3) :=
by sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l528_52834


namespace NUMINAMATH_CALUDE_inverse_not_in_M_exponential_in_M_logarithmic_in_M_l528_52828

-- Define set M
def M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Problem 1
theorem inverse_not_in_M :
  ¬ M (fun x => 1 / x) := by sorry

-- Problem 2
theorem exponential_in_M (k b : ℝ) :
  M (fun x => k * 2^x + b) ↔ (k = 0 ∧ b = 0) ∨ (k ≠ 0 ∧ (2 * k + b) / k > 0) := by sorry

-- Problem 3
theorem logarithmic_in_M :
  ∀ a : ℝ, M (fun x => Real.log (a / (x^2 + 2))) ↔ 
    (a ≥ 3/2 ∧ a ≤ 6 ∧ a ≠ 3) := by sorry

end NUMINAMATH_CALUDE_inverse_not_in_M_exponential_in_M_logarithmic_in_M_l528_52828


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l528_52860

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l528_52860


namespace NUMINAMATH_CALUDE_bonus_distribution_l528_52826

theorem bonus_distribution (total_amount : ℕ) (total_notes : ℕ) 
  (h1 : total_amount = 160) 
  (h2 : total_notes = 25) : 
  ∃ (x y z : ℕ), 
    x + y + z = total_notes ∧ 
    2*x + 5*y + 10*z = total_amount ∧ 
    y = z ∧ 
    x = 5 ∧ y = 10 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_bonus_distribution_l528_52826


namespace NUMINAMATH_CALUDE_book_pages_calculation_l528_52883

theorem book_pages_calculation (num_books : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_side = 4 → 
  sides_per_sheet = 2 → 
  num_sheets = 150 → 
  (num_sheets * pages_per_side * sides_per_sheet) / num_books = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l528_52883


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l528_52837

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) (h : i * i = -1) :
  Complex.im (i / (1 + i)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l528_52837


namespace NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_l528_52864

/-- The maximum area of an equilateral triangle inscribed in a 13x14 rectangle --/
theorem max_area_inscribed_equilateral_triangle :
  ∃ (A : ℝ),
    A = (183 : ℝ) * Real.sqrt 3 ∧
    ∀ (s : ℝ),
      0 ≤ s →
      s ≤ 13 →
      s * Real.sqrt 3 / 2 ≤ 14 →
      s^2 * Real.sqrt 3 / 4 ≤ A :=
by sorry

#eval (183 : Nat) + 3 + 0

end NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_l528_52864


namespace NUMINAMATH_CALUDE_f_increasing_decreasing_l528_52805

-- Define the function f(x) = x^3 - x^2 - x
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

-- Theorem statement
theorem f_increasing_decreasing :
  (∀ x < -1/3, f' x > 0) ∧
  (∀ x ∈ Set.Ioo (-1/3) 1, f' x < 0) ∧
  (∀ x > 1, f' x > 0) ∧
  (f 1 = -1) ∧
  (f' 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_decreasing_l528_52805


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l528_52870

theorem initial_mean_calculation (n : ℕ) (M initial_wrong corrected_value new_mean : ℝ) :
  n = 50 ∧
  initial_wrong = 23 ∧
  corrected_value = 30 ∧
  new_mean = 36.5 ∧
  (n : ℝ) * new_mean = (n : ℝ) * M + (corrected_value - initial_wrong) →
  M = 36.36 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l528_52870


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l528_52832

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a + b + c = 19) : 
  a*b + b*c + a*c = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l528_52832
