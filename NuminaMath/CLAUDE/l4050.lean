import Mathlib

namespace NUMINAMATH_CALUDE_coefficient_x5_is_11_l4050_405070

/-- The coefficient of x^5 in the expansion of ((x^2 + x - 1)^5) -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in ((x^2 + x - 1)^5) is 11 -/
theorem coefficient_x5_is_11 : coefficient_x5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_is_11_l4050_405070


namespace NUMINAMATH_CALUDE_cubes_fill_box_l4050_405083

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the size of a cube -/
def CubeSize : ℕ := 2

/-- Calculate the number of cubes that can fit along a given dimension -/
def cubesAlongDimension (dimension : ℕ) : ℕ :=
  dimension / CubeSize

/-- Calculate the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) : ℕ :=
  (cubesAlongDimension box.length) * (cubesAlongDimension box.width) * (cubesAlongDimension box.height)

/-- Calculate the volume of the box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculate the volume occupied by the cubes -/
def cubesVolume (box : BoxDimensions) : ℕ :=
  totalCubes box * (CubeSize * CubeSize * CubeSize)

/-- The main theorem: The volume occupied by cubes is equal to the box volume -/
theorem cubes_fill_box (box : BoxDimensions) 
  (h1 : box.length = 8) (h2 : box.width = 6) (h3 : box.height = 12) : 
  cubesVolume box = boxVolume box := by
  sorry

#check cubes_fill_box

end NUMINAMATH_CALUDE_cubes_fill_box_l4050_405083


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_midpoint_l4050_405005

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- Check if a point is on the altitude from A to BC -/
def isOnAltitude (t : Triangle) (d : Point) : Prop :=
  (t.B.y - t.C.y) * (d.x - t.A.x) = (t.C.x - t.B.x) * (d.y - t.A.y)

theorem isosceles_triangle_altitude_midpoint (t : Triangle) (d : Point) :
  t.A = Point.mk 5 7 →
  t.B = Point.mk (-1) 3 →
  d = Point.mk 1 5 →
  isIsosceles t →
  isOnAltitude t d →
  t.C = Point.mk 3 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_altitude_midpoint_l4050_405005


namespace NUMINAMATH_CALUDE_sin_330_degrees_l4050_405063

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l4050_405063


namespace NUMINAMATH_CALUDE_book_chunk_sheets_l4050_405040

/-- Checks if two numbers have the same digits (possibly in different order) -/
def sameDigits (a b : Nat) : Bool := sorry

/-- Finds the smallest even number greater than n composed of the same digits as n -/
def smallestEvenWithSameDigits (n : Nat) : Nat := sorry

theorem book_chunk_sheets (first_page last_page : Nat) : 
  first_page = 163 →
  last_page = smallestEvenWithSameDigits first_page →
  (last_page - first_page + 1) / 2 = 77 := by sorry

end NUMINAMATH_CALUDE_book_chunk_sheets_l4050_405040


namespace NUMINAMATH_CALUDE_equidistant_point_on_leg_l4050_405027

/-- 
Given a right triangle with legs 240 and 320 rods, and hypotenuse 400 rods,
prove that the point on the longer leg equidistant from the other two vertices
is 95 rods from the right angle.
-/
theorem equidistant_point_on_leg (a b c x : ℝ) : 
  a = 240 → b = 320 → c = 400 → 
  a^2 + b^2 = c^2 →
  x^2 + a^2 = (b - x)^2 + b^2 →
  x = 95 := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_on_leg_l4050_405027


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l4050_405058

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l4050_405058


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l4050_405068

def a (m : ℝ) : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (-2, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_proof (m : ℝ) :
  parallel ((a m).1 - b.1, (a m).2 - b.2) b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l4050_405068


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4050_405075

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 15*x^2 + 12*x - 20
  let g : ℝ → ℝ := λ x => x - 2
  (f 2) = 16 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4050_405075


namespace NUMINAMATH_CALUDE_freddy_travel_time_l4050_405081

/-- Represents the travel details of a person -/
structure TravelDetails where
  start : String
  destination : String
  distance : ℝ
  time : ℝ

/-- Given travel conditions for Eddy and Freddy -/
def travel_conditions : Prop :=
  ∃ (eddy freddy : TravelDetails),
    eddy.start = "A" ∧
    eddy.destination = "B" ∧
    freddy.start = "A" ∧
    freddy.destination = "C" ∧
    eddy.distance = 540 ∧
    freddy.distance = 300 ∧
    eddy.time = 3 ∧
    (eddy.distance / eddy.time) / (freddy.distance / freddy.time) = 2.4

/-- Theorem: Freddy's travel time is 4 hours -/
theorem freddy_travel_time : travel_conditions → ∃ (freddy : TravelDetails), freddy.time = 4 := by
  sorry


end NUMINAMATH_CALUDE_freddy_travel_time_l4050_405081


namespace NUMINAMATH_CALUDE_sphere_division_l4050_405007

/-- The maximum number of parts into which the surface of a sphere can be divided by n great circles -/
def max_sphere_parts (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_sphere_parts gives the correct number of maximum parts -/
theorem sphere_division (n : ℕ) :
  max_sphere_parts n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_sphere_division_l4050_405007


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l4050_405093

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line parallel to y-axis passing through a given point
def LineParallelToYAxis (p : Point2D) := {q : Point2D | q.x = p.x}

theorem line_through_point_parallel_to_y_axis 
  (A : Point2D) 
  (h : A.x = -3 ∧ A.y = 1) 
  (P : Point2D) 
  (h_on_line : P ∈ LineParallelToYAxis A) : 
  P.x = -3 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_y_axis_l4050_405093


namespace NUMINAMATH_CALUDE_zero_point_location_l4050_405034

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

-- State the theorem
theorem zero_point_location (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (0 < a) → (a < 2) →
  (f a x₁ = 0) → (f a x₂ = 0) → (f a x₃ = 0) →
  (x₁ < x₂) → (x₂ < x₃) →
  (0 < x₂) ∧ (x₂ < 1) := by
  sorry

end NUMINAMATH_CALUDE_zero_point_location_l4050_405034


namespace NUMINAMATH_CALUDE_base8_536_equals_base7_1054_l4050_405076

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 536 in base 8 is equal to 1054 in base 7 -/
theorem base8_536_equals_base7_1054 : 
  base10_to_base7 (base8_to_base10 536) = 1054 := by sorry

end NUMINAMATH_CALUDE_base8_536_equals_base7_1054_l4050_405076


namespace NUMINAMATH_CALUDE_camel_zebra_ratio_l4050_405037

/-- Proves that the ratio of camels to zebras is 1:2 given the specified conditions -/
theorem camel_zebra_ratio :
  ∀ (zebras camels monkeys giraffes : ℕ),
    zebras = 12 →
    monkeys = 4 * camels →
    giraffes = 2 →
    monkeys = giraffes + 22 →
    (camels : ℚ) / zebras = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_camel_zebra_ratio_l4050_405037


namespace NUMINAMATH_CALUDE_field_area_is_500_l4050_405048

/-- Represents the area of a field divided into two parts -/
structure FieldArea where
  small : ℝ
  large : ℝ

/-- Calculates the total area of the field -/
def total_area (f : FieldArea) : ℝ := f.small + f.large

/-- Theorem: The total area of the field is 500 hectares -/
theorem field_area_is_500 (f : FieldArea) 
  (h1 : f.small = 225)
  (h2 : f.large - f.small = (1/5) * ((f.small + f.large) / 2)) :
  total_area f = 500 := by
  sorry

end NUMINAMATH_CALUDE_field_area_is_500_l4050_405048


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l4050_405053

theorem max_value_of_sum_and_powers (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 2 →
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℝ), 
    x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 →
    x + y + z + w = 2 →
    x + y^2 + z^3 + w^4 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l4050_405053


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l4050_405025

def original_mean : ℝ := 45
def num_observations : ℕ := 100
def incorrect_observations : List ℝ := [32, 12, 25]
def correct_observations : List ℝ := [67, 52, 85]

theorem corrected_mean_calculation :
  let original_sum := original_mean * num_observations
  let incorrect_sum := incorrect_observations.sum
  let correct_sum := correct_observations.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_sum := original_sum + adjustment
  let corrected_mean := corrected_sum / num_observations
  corrected_mean = 46.35 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l4050_405025


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4050_405098

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4050_405098


namespace NUMINAMATH_CALUDE_probability_of_letter_in_probability_l4050_405001

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in the word 'PROBABILITY' -/
def unique_letters : ℕ := 9

/-- The probability of randomly selecting a letter from the alphabet
    that appears in the word 'PROBABILITY' -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_letter_in_probability :
  probability = 9 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_probability_l4050_405001


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l4050_405039

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l4050_405039


namespace NUMINAMATH_CALUDE_luke_mowing_money_l4050_405052

/-- The amount of money Luke made mowing lawns -/
def mowing_money : ℝ := sorry

/-- The amount of money Luke made weed eating -/
def weed_eating_money : ℝ := 18

/-- The amount Luke spends per week -/
def weekly_spending : ℝ := 3

/-- The number of weeks the money lasts -/
def weeks_lasted : ℝ := 9

/-- Theorem stating that Luke made $9 mowing lawns -/
theorem luke_mowing_money : mowing_money = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_mowing_money_l4050_405052


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_is_18_l4050_405018

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Theorem: The y-coordinate of vertex C in the given pentagon is 18 -/
theorem y_coordinate_of_C_is_18 (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 6))
  (h3 : p.D = (6, 6))
  (h4 : p.E = (6, 0))
  (h5 : hasVerticalSymmetry p)
  (h6 : pentagonArea p = 72)
  : p.C.2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_is_18_l4050_405018


namespace NUMINAMATH_CALUDE_infinite_solutions_in_interval_l4050_405054

theorem infinite_solutions_in_interval (x : Real) (h : x ∈ Set.Icc 0 (2 * Real.pi)) :
  Real.cos ((Real.pi / 2) * Real.cos x + (Real.pi / 2) * Real.sin x) =
  Real.sin ((Real.pi / 2) * Real.cos x - (Real.pi / 2) * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_in_interval_l4050_405054


namespace NUMINAMATH_CALUDE_fifteen_factorial_trailing_zeros_l4050_405069

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Base 18 expressed as 2 · 3² -/
def base18 : ℕ := 2 * 3^2

/-- The main theorem -/
theorem fifteen_factorial_trailing_zeros :
  trailingZeros (factorial 15) base18 = 3 := by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_trailing_zeros_l4050_405069


namespace NUMINAMATH_CALUDE_dimes_percentage_l4050_405088

theorem dimes_percentage (num_nickels num_dimes : ℕ) 
  (nickel_value dime_value : ℕ) : 
  num_nickels = 40 → 
  num_dimes = 30 → 
  nickel_value = 5 → 
  dime_value = 10 → 
  (num_dimes * dime_value : ℚ) / 
  (num_nickels * nickel_value + num_dimes * dime_value) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_dimes_percentage_l4050_405088


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l4050_405032

/-- Atomic weight of Potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Potassium atoms in the compound -/
def num_K : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound -/
def molecular_weight : ℝ := num_K * atomic_weight_K + num_Br * atomic_weight_Br + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 167.00 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l4050_405032


namespace NUMINAMATH_CALUDE_math_competition_problem_l4050_405035

theorem math_competition_problem (n : ℕ) (S : Finset (Finset (Fin 6))) :
  (∀ (i j : Fin 6), i ≠ j → (S.filter (λ s => i ∈ s ∧ j ∈ s)).card > (2 * S.card) / 5) →
  (∀ s ∈ S, s.card ≤ 5) →
  (S.filter (λ s => s.card = 5)).card ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_math_competition_problem_l4050_405035


namespace NUMINAMATH_CALUDE_triangle_altitude_l4050_405045

/-- Given a triangle with area 720 square feet and base 40 feet, its altitude is 36 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 720 → base = 40 → area = (1/2) * base * altitude → altitude = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l4050_405045


namespace NUMINAMATH_CALUDE_min_students_in_both_clubs_l4050_405031

theorem min_students_in_both_clubs
  (total_students : ℕ)
  (club1_students : ℕ)
  (club2_students : ℕ)
  (h1 : total_students = 33)
  (h2 : club1_students ≥ 24)
  (h3 : club2_students ≥ 24) :
  ∃ (intersection : ℕ), intersection ≥ 15 ∧
    intersection ≤ min club1_students club2_students ∧
    intersection ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_both_clubs_l4050_405031


namespace NUMINAMATH_CALUDE_polynomial_multiple_divisible_by_three_l4050_405065

theorem polynomial_multiple_divisible_by_three 
  {R : Type*} [CommRing R] [Nontrivial R] :
  ∀ (P : Polynomial R), P ≠ 0 → 
  ∃ (Q : Polynomial R), Q ≠ 0 ∧ 
  ∀ (i : ℕ), (P * Q).coeff i ≠ 0 → i % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiple_divisible_by_three_l4050_405065


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l4050_405080

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_line_at_one (x : ℝ) :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := 2
  (fun x => m * (x - p.1) + p.2) = (fun x => 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l4050_405080


namespace NUMINAMATH_CALUDE_no_multiples_of_5005_l4050_405041

theorem no_multiples_of_5005 : ¬∃ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 49 ∧ 
  ∃ (k : ℕ+), 5005 * k = 10^j - 10^i := by
  sorry

end NUMINAMATH_CALUDE_no_multiples_of_5005_l4050_405041


namespace NUMINAMATH_CALUDE_exists_fourth_root_of_3_to_20_l4050_405030

theorem exists_fourth_root_of_3_to_20 : ∃ n : ℕ, n^4 = 3^20 ∧ n = 243 := by sorry

end NUMINAMATH_CALUDE_exists_fourth_root_of_3_to_20_l4050_405030


namespace NUMINAMATH_CALUDE_average_snack_sales_theorem_l4050_405051

/-- Represents the sales data for snacks per 6 movie tickets sold -/
structure SnackSales where
  crackers_quantity : ℕ
  crackers_price : ℚ
  beverage_quantity : ℕ
  beverage_price : ℚ
  chocolate_quantity : ℕ
  chocolate_price : ℚ

/-- Calculates the average snack sales per movie ticket -/
def average_snack_sales_per_ticket (sales : SnackSales) : ℚ :=
  let total_sales := sales.crackers_quantity * sales.crackers_price +
                     sales.beverage_quantity * sales.beverage_price +
                     sales.chocolate_quantity * sales.chocolate_price
  total_sales / 6

/-- The main theorem stating the average snack sales per movie ticket -/
theorem average_snack_sales_theorem (sales : SnackSales) 
  (h1 : sales.crackers_quantity = 3)
  (h2 : sales.crackers_price = 9/4)
  (h3 : sales.beverage_quantity = 4)
  (h4 : sales.beverage_price = 3/2)
  (h5 : sales.chocolate_quantity = 4)
  (h6 : sales.chocolate_price = 1) :
  average_snack_sales_per_ticket sales = 279/100 := by
  sorry

end NUMINAMATH_CALUDE_average_snack_sales_theorem_l4050_405051


namespace NUMINAMATH_CALUDE_inequality_proof_l4050_405013

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧
  ∀ k > 3/4, ∃ a' b' c' d' : ℝ, a' ≥ -1 ∧ b' ≥ -1 ∧ c' ≥ -1 ∧ d' ≥ -1 ∧
    a'^3 + b'^3 + c'^3 + d'^3 + 1 < k * (a' + b' + c' + d') :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4050_405013


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l4050_405092

/-- Proves that the given trigonometric expression evaluates to √3 --/
theorem trig_expression_equals_sqrt_three :
  (Real.cos (350 * π / 180) - 2 * Real.sin (160 * π / 180)) / Real.sin (-190 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l4050_405092


namespace NUMINAMATH_CALUDE_f_properties_l4050_405016

noncomputable def f (x : ℝ) := 4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x - 2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

def has_max_at (f : ℝ → ℝ) (M : ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = M ∧ ∀ x, f x ≤ M

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (is_smallest_positive_period f Real.pi) ∧
  (∀ k : ℤ, has_max_at f 4 (k * Real.pi + Real.pi / 6)) ∧
  (∀ k : ℤ, is_increasing_on f (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l4050_405016


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l4050_405029

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at F is 90 degrees (right angle)
  sorry

def angle_E_is_45_deg (t : Triangle) : Prop :=
  -- The angle at E is 45 degrees
  sorry

def side_DF_length (t : Triangle) : ℝ :=
  -- The length of side DF
  8

-- Define the incircle radius
def incircle_radius (t : Triangle) : ℝ :=
  -- The radius of the incircle
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_E_is_45_deg t)
  (h3 : side_DF_length t = 8) :
  incircle_radius t = 8 - 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l4050_405029


namespace NUMINAMATH_CALUDE_bisector_quadrilateral_is_square_l4050_405085

/-- A rectangle that is not a square -/
structure NonSquareRectangle where
  length : ℝ
  width : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  not_square : length ≠ width

/-- The quadrilateral formed by the intersection of angle bisectors -/
structure BisectorQuadrilateral (r : NonSquareRectangle) where
  vertices : Fin 4 → ℝ × ℝ

/-- Theorem: The quadrilateral formed by the intersection of angle bisectors in a non-square rectangle is a square -/
theorem bisector_quadrilateral_is_square (r : NonSquareRectangle) (q : BisectorQuadrilateral r) :
  IsSquare q.vertices := by sorry

end NUMINAMATH_CALUDE_bisector_quadrilateral_is_square_l4050_405085


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_l4050_405036

/-- The minimum distance between points on y = e^x and y = ln(x) is √2 -/
theorem min_distance_exp_ln (P Q : ℝ × ℝ) :
  (∃ x : ℝ, P = (x, Real.exp x)) →
  (∃ y : ℝ, Q = (y, Real.log y)) →
  ∀ P' Q' : ℝ × ℝ,
  (∃ x' : ℝ, P' = (x', Real.exp x')) →
  (∃ y' : ℝ, Q' = (y', Real.log y')) →
  Real.sqrt 2 ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_l4050_405036


namespace NUMINAMATH_CALUDE_beach_towel_usage_per_person_per_day_l4050_405047

theorem beach_towel_usage_per_person_per_day :
  let num_families : ℕ := 3
  let people_per_family : ℕ := 4
  let total_days : ℕ := 7
  let towels_per_load : ℕ := 14
  let total_loads : ℕ := 6
  let total_people : ℕ := num_families * people_per_family
  let total_towels : ℕ := towels_per_load * total_loads
  let towels_per_day : ℕ := total_towels / total_days
  towels_per_day / total_people = 1 :=
by sorry

end NUMINAMATH_CALUDE_beach_towel_usage_per_person_per_day_l4050_405047


namespace NUMINAMATH_CALUDE_barbara_typing_speed_reduction_l4050_405091

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Barbara's typing speed has reduced by 40 words per minute. -/
theorem barbara_typing_speed_reduction :
  typing_speed_reduction 212 3440 20 = 40 := by
  sorry

#eval typing_speed_reduction 212 3440 20

end NUMINAMATH_CALUDE_barbara_typing_speed_reduction_l4050_405091


namespace NUMINAMATH_CALUDE_sum_of_X_and_Y_is_12_l4050_405084

/-- Converts a single-digit number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := n

/-- Converts a two-digit number from base 6 to base 10 -/
def twoDigitBase6ToBase10 (tens : ℕ) (ones : ℕ) : ℕ := 
  6 * tens + ones

theorem sum_of_X_and_Y_is_12 (X Y : ℕ) : 
  (X < 6 ∧ Y < 6) →  -- Ensure X and Y are single digits in base 6
  twoDigitBase6ToBase10 1 3 + twoDigitBase6ToBase10 X Y = 
  twoDigitBase6ToBase10 2 0 + twoDigitBase6ToBase10 5 2 →
  X + Y = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_X_and_Y_is_12_l4050_405084


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4050_405057

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4050_405057


namespace NUMINAMATH_CALUDE_second_month_sale_l4050_405024

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale ∧
    sale_month2 = 7927 :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l4050_405024


namespace NUMINAMATH_CALUDE_any_nonzero_rational_to_zero_power_is_one_l4050_405060

theorem any_nonzero_rational_to_zero_power_is_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_rational_to_zero_power_is_one_l4050_405060


namespace NUMINAMATH_CALUDE_min_value_theorem_l4050_405079

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : -2 * m - n + 2 = 0) :
  2 / m + 1 / n ≥ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4050_405079


namespace NUMINAMATH_CALUDE_choose_three_from_seven_l4050_405074

/-- The number of ways to choose 3 distinct people from a group of 7 to fill 3 distinct positions -/
def ways_to_choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that choosing 3 distinct people from a group of 7 to fill 3 distinct positions can be done in 210 ways -/
theorem choose_three_from_seven :
  ways_to_choose_officers 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_seven_l4050_405074


namespace NUMINAMATH_CALUDE_derivative_of_f_l4050_405028

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  HasDerivAt f (2*x*(Real.cos x) - x^2*(Real.sin x)) x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l4050_405028


namespace NUMINAMATH_CALUDE_max_value_implies_a_l4050_405073

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 32 ∧ ∀ (x : ℝ), a * x * (x - 2)^2 ≤ M) →
  a = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l4050_405073


namespace NUMINAMATH_CALUDE_dividend_proof_l4050_405056

theorem dividend_proof (y : ℝ) (x : ℝ) (h : y > 3) :
  (x = (3 * y + 5) * (2 * y - 1) + (5 * y - 13)) →
  (x = 6 * y^2 + 12 * y - 18) := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l4050_405056


namespace NUMINAMATH_CALUDE_no_largest_non_expressible_l4050_405087

-- Define a function to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the property of being expressible as the sum of a multiple of 36 and a non-square
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ b > 0 ∧ ¬(is_square b)

-- Theorem statement
theorem no_largest_non_expressible :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬(is_expressible m) :=
sorry

end NUMINAMATH_CALUDE_no_largest_non_expressible_l4050_405087


namespace NUMINAMATH_CALUDE_sum_of_squares_is_four_l4050_405019

/-- Represents a rectangle ABCD with an inscribed ellipse K and a point P on K. -/
structure RectangleWithEllipse where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Angle parameter for point P on the ellipse -/
  θ : ℝ
  /-- AB = 2 -/
  h_ab : ab = 2
  /-- AD < √2 -/
  h_ad : ad < Real.sqrt 2

/-- The sum of squares of AM and LB is always 4 -/
theorem sum_of_squares_is_four (rect : RectangleWithEllipse) :
  let x_M := (Real.sqrt 2 * (Real.cos rect.θ - 1)) / (Real.sqrt 2 - Real.sin rect.θ) + 1
  let x_L := (Real.sqrt 2 * (1 + Real.cos rect.θ)) / (Real.sqrt 2 - Real.sin rect.θ) - 1
  (1 + x_M)^2 + (1 - x_L)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_four_l4050_405019


namespace NUMINAMATH_CALUDE_judy_spending_l4050_405050

-- Define the prices and quantities
def carrot_price : ℚ := 1
def carrot_quantity : ℕ := 8
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_regular_price : ℚ := 4
def pineapple_quantity : ℕ := 3
def flour_price : ℚ := 5
def flour_quantity : ℕ := 3
def ice_cream_price : ℚ := 7
def ice_cream_quantity : ℕ := 2

-- Define the discount conditions
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 30

-- Calculate the total before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  (pineapple_regular_price / 2) * pineapple_quantity +
  flour_price * flour_quantity +
  ice_cream_price * ice_cream_quantity

-- Apply discounts
def final_total : ℚ :=
  let discounted_total := 
    if total_before_discounts > discount_threshold
    then total_before_discounts * (1 - discount_rate)
    else total_before_discounts
  if discounted_total ≥ coupon_threshold
  then discounted_total - coupon_value
  else discounted_total

-- Theorem statement
theorem judy_spending : final_total = 44.5 := by sorry

end NUMINAMATH_CALUDE_judy_spending_l4050_405050


namespace NUMINAMATH_CALUDE_max_value_on_circle_l4050_405009

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 9) :
  ∃ (M : ℝ), M = 9 ∧ ∀ (a b : ℝ), a^2 + b^2 = 9 → 3 * |a| + 2 * |b| ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l4050_405009


namespace NUMINAMATH_CALUDE_equation_equivalence_l4050_405090

theorem equation_equivalence : ∀ x : ℝ, x * (x + 2) = 5 ↔ x^2 + 2*x - 5 = 0 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4050_405090


namespace NUMINAMATH_CALUDE_square_sum_of_roots_l4050_405071

theorem square_sum_of_roots (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_roots_l4050_405071


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l4050_405094

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 3, 4, 5, 6, 6, 5]

def total_faces : ℕ := 24

theorem hidden_faces_sum :
  let total_dots := 4 * standard_die_sum
  let visible_dots := visible_faces.sum
  let hidden_faces := total_faces - visible_faces.length
  hidden_faces = 15 ∧ total_dots - visible_dots = 51 := by sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l4050_405094


namespace NUMINAMATH_CALUDE_special_rectangle_ratio_l4050_405078

/-- A rectangle with the property that the square of the ratio of its short side to its long side
    is equal to the ratio of its long side to its diagonal. -/
structure SpecialRectangle where
  short : ℝ
  long : ℝ
  diagonal : ℝ
  short_positive : 0 < short
  long_positive : 0 < long
  diagonal_positive : 0 < diagonal
  pythagorean : diagonal^2 = short^2 + long^2
  special_property : (short / long)^2 = long / diagonal

/-- The ratio of the short side to the long side in a SpecialRectangle is (√5 - 1) / 3. -/
theorem special_rectangle_ratio (r : SpecialRectangle) : 
  r.short / r.long = (Real.sqrt 5 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_ratio_l4050_405078


namespace NUMINAMATH_CALUDE_integer_part_of_M_l4050_405044

theorem integer_part_of_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  4 < Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) < 5 :=
by sorry

end NUMINAMATH_CALUDE_integer_part_of_M_l4050_405044


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l4050_405000

theorem not_divides_power_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l4050_405000


namespace NUMINAMATH_CALUDE_problem_solution_l4050_405021

theorem problem_solution (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4050_405021


namespace NUMINAMATH_CALUDE_cylinder_different_views_l4050_405064

/-- Represents a geometric body --/
inductive GeometricBody
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

/-- Represents the dimensions of a view --/
structure ViewDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Returns true if all three views have the same dimensions --/
def sameViewDimensions (top front left : ViewDimensions) : Prop :=
  top.length = front.length ∧
  front.height = left.height ∧
  left.width = top.width

/-- Returns the three orthogonal views of a geometric body --/
def getViews (body : GeometricBody) : (ViewDimensions × ViewDimensions × ViewDimensions) :=
  sorry

theorem cylinder_different_views :
  ∀ (body : GeometricBody),
    (∃ (top front left : ViewDimensions),
      getViews body = (top, front, left) ∧
      ¬(sameViewDimensions top front left)) ↔
    body = GeometricBody.Cylinder :=
  sorry

end NUMINAMATH_CALUDE_cylinder_different_views_l4050_405064


namespace NUMINAMATH_CALUDE_platform_length_l4050_405043

/-- Given a train of length 450 meters that crosses a platform in 60 seconds
    and a signal pole in 30 seconds, prove that the length of the platform is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 450)
  (h2 : platform_crossing_time = 60)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 450 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l4050_405043


namespace NUMINAMATH_CALUDE_lose_sector_area_l4050_405072

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the LOSE sector is 150π sq cm. -/
theorem lose_sector_area (radius : ℝ) (win_prob : ℝ) (lose_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  lose_area = 150 * Real.pi → 
  lose_area = (1 - win_prob) * Real.pi * radius^2 := by
sorry

end NUMINAMATH_CALUDE_lose_sector_area_l4050_405072


namespace NUMINAMATH_CALUDE_population_ratio_l4050_405014

/-- Given three cities X, Y, and Z, where the population of X is 5 times that of Y,
    and the population of Y is twice that of Z, prove that the ratio of the
    population of X to Z is 10:1 -/
theorem population_ratio (x y z : ℕ) (hxy : x = 5 * y) (hyz : y = 2 * z) :
  x / z = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l4050_405014


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4050_405020

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 → 
  exterior_angle = 18 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4050_405020


namespace NUMINAMATH_CALUDE_hyperbola_ratio_l4050_405026

theorem hyperbola_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (3^2 / a^2 - (3 * Real.sqrt 2)^2 / b^2 = 1) →
  (Real.tan (45 * π / 360) = b / a) →
  (a / b = Real.sqrt 2 + 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_ratio_l4050_405026


namespace NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l4050_405006

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 12*x + a = 0 ∧
    y^3 - 12*y + a = 0 ∧
    z^3 - 12*z + a = 0) ↔
  -16 < a ∧ a < 16 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l4050_405006


namespace NUMINAMATH_CALUDE_farmer_sheep_problem_l4050_405049

theorem farmer_sheep_problem (total : ℕ) 
  (h1 : total % 3 = 0)  -- First son's share is whole
  (h2 : total % 5 = 0)  -- Second son's share is whole
  (h3 : total % 6 = 0)  -- Third son's share is whole
  (h4 : total % 8 = 0)  -- Daughter's share is whole
  (h5 : total - (total / 3 + total / 5 + total / 6 + total / 8) = 12)  -- Charity's share
  : total = 68 := by
  sorry

end NUMINAMATH_CALUDE_farmer_sheep_problem_l4050_405049


namespace NUMINAMATH_CALUDE_largest_possible_b_value_l4050_405099

theorem largest_possible_b_value (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →  -- c is the smallest odd prime number
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ (x ≠ c) → (a * x * c ≠ 360)) →
  (b = 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_b_value_l4050_405099


namespace NUMINAMATH_CALUDE_fish_population_estimate_l4050_405022

/-- Represents the data from a single round of fish catching --/
structure RoundData where
  caught : Nat
  tagged : Nat

/-- Represents the data from the fish population study --/
structure FishStudy where
  round1 : RoundData
  round2 : RoundData
  round3 : RoundData

/-- The Lincoln-Petersen estimator function --/
def lincolnPetersen (c1 c2 r2 : Nat) : Nat :=
  (c1 * c2) / r2

/-- Theorem stating that the estimated fish population is 800 --/
theorem fish_population_estimate (study : FishStudy)
    (h1 : study.round1 = { caught := 30, tagged := 0 })
    (h2 : study.round2 = { caught := 80, tagged := 6 })
    (h3 : study.round3 = { caught := 100, tagged := 10 }) :
    lincolnPetersen study.round2.caught study.round3.caught study.round3.tagged = 800 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l4050_405022


namespace NUMINAMATH_CALUDE_polyhedron_sum_l4050_405095

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  P : ℕ  -- number of pentagonal faces meeting at each vertex
  faces_sum : faces = triangular_faces + pentagonal_faces
  faces_32 : faces = 32
  vertex_relation : vertices * (T + P - 2) = 60
  face_relation : 5 * vertices * T + 3 * vertices * P = 480

/-- The sum of P, T, and V for the specific polyhedron is 34 -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.vertices = 34 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l4050_405095


namespace NUMINAMATH_CALUDE_max_students_distribution_l4050_405003

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1001) (h2 : pencils = 910) :
  (∃ (students pen_per_student pencil_per_student : ℕ),
    students * pen_per_student = pens ∧
    students * pencil_per_student = pencils ∧
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) ↔
  students = Nat.gcd pens pencils :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l4050_405003


namespace NUMINAMATH_CALUDE_number_not_divisible_by_8_and_digit_product_l4050_405066

def numbers : List Nat := [1616, 1728, 1834, 1944, 2056]

def is_divisible_by_8 (n : Nat) : Bool :=
  n % 8 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem number_not_divisible_by_8_and_digit_product :
  ∃ n ∈ numbers, ¬is_divisible_by_8 n ∧ units_digit n * tens_digit n = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_not_divisible_by_8_and_digit_product_l4050_405066


namespace NUMINAMATH_CALUDE_books_remaining_l4050_405033

theorem books_remaining (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l4050_405033


namespace NUMINAMATH_CALUDE_shortest_paths_count_l4050_405004

/-- The number of shortest paths on a chess board from (0,0) to (m,n) -/
def numShortestPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of shortest paths from (0,0) to (m,n) on a chess board,
    where movement is restricted to coordinate axis directions and
    direction changes only at integer coordinates, is equal to (m+n choose m) -/
theorem shortest_paths_count (m n : ℕ) :
  numShortestPaths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_shortest_paths_count_l4050_405004


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l4050_405096

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 42) 
  (h2 : num_bookshelves = 21) :
  total_books / num_bookshelves = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l4050_405096


namespace NUMINAMATH_CALUDE_sqrt_sum_bounds_l4050_405086

theorem sqrt_sum_bounds :
  let m := Real.sqrt 4 + Real.sqrt 3
  3 < m ∧ m < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_bounds_l4050_405086


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l4050_405002

theorem no_linear_term_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + a) * (x - 1/2) = x^2 + b + c * x) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l4050_405002


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achieved_l4050_405011

theorem min_value_of_reciprocal_sum (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  4/x + 1/y ≥ 9 := by
  sorry

theorem min_value_achieved (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4/x₀ + 1/y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achieved_l4050_405011


namespace NUMINAMATH_CALUDE_expansion_theorem_l4050_405055

theorem expansion_theorem (x : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 2) / (Nat.choose n 4) = 3 / 14) →
  (n = 10 ∧ 
   ∃ m : ℕ, m = 8 ∧ 
   (Nat.choose n m) = 45 ∧ 
   20 - 2 * m - (1/2) * m = 0) :=
by sorry

end NUMINAMATH_CALUDE_expansion_theorem_l4050_405055


namespace NUMINAMATH_CALUDE_triangle_sine_ratio_l4050_405042

theorem triangle_sine_ratio (A B C : ℝ) (h1 : 0 < A ∧ A < π)
                                       (h2 : 0 < B ∧ B < π)
                                       (h3 : 0 < C ∧ C < π)
                                       (h4 : A + B + C = π)
                                       (h5 : Real.sin A / Real.sin B = 6/5)
                                       (h6 : Real.sin B / Real.sin C = 5/4) :
  Real.sin B = 5 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_ratio_l4050_405042


namespace NUMINAMATH_CALUDE_initial_customers_count_l4050_405082

/-- The number of customers who left the restaurant -/
def customers_left : ℕ := 11

/-- The number of customers who remained in the restaurant -/
def customers_remained : ℕ := 3

/-- The initial number of customers in the restaurant -/
def initial_customers : ℕ := customers_left + customers_remained

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_count_l4050_405082


namespace NUMINAMATH_CALUDE_factor_expression_l4050_405046

theorem factor_expression (y : ℝ) : 6*y*(y+2) + 15*(y+2) + 12 = 3*(2*y+5)*(y+2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4050_405046


namespace NUMINAMATH_CALUDE_ratio_first_to_last_l4050_405017

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a x c d b : ℚ)
  (is_arithmetic : ∃ (diff : ℚ), x = a + diff ∧ c = x + diff ∧ d = c + diff ∧ b = d + diff)
  (fourth_term : d = 3 * x)
  (fifth_term : b = 4 * x)

/-- The ratio of the first term to the last term is -1/4 -/
theorem ratio_first_to_last (seq : ArithmeticSequence) : seq.a / seq.b = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_to_last_l4050_405017


namespace NUMINAMATH_CALUDE_partnership_capital_share_l4050_405038

theorem partnership_capital_share 
  (total_capital : ℝ) 
  (total_profit : ℝ) 
  (a_profit_share : ℝ) 
  (b_capital_share : ℝ) 
  (c_capital_share : ℝ) 
  (h1 : b_capital_share = (1 / 4 : ℝ) * total_capital) 
  (h2 : c_capital_share = (1 / 5 : ℝ) * total_capital) 
  (h3 : a_profit_share = (800 : ℝ)) 
  (h4 : total_profit = (2400 : ℝ)) 
  (h5 : a_profit_share / total_profit = (1 / 3 : ℝ)) :
  ∃ (a_capital_share : ℝ), 
    a_capital_share = (1 / 3 : ℝ) * total_capital ∧ 
    a_capital_share + b_capital_share + c_capital_share ≤ total_capital := by
  sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l4050_405038


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l4050_405089

theorem prop_a_necessary_not_sufficient :
  ¬(∀ x y : ℤ, (x ≠ 1000 ∨ y ≠ 1002) ↔ (x + y ≠ 2002)) ∧
  (∀ x y : ℤ, (x + y ≠ 2002) → (x ≠ 1000 ∨ y ≠ 1002)) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l4050_405089


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l4050_405077

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 9 * X^4 + 8 * X^3 - 12 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 3 * X^2 + 2 * X + 5
  let quotient : Polynomial ℚ := 3 * X^2 - 2 * X + 2
  (dividend.div divisor) = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l4050_405077


namespace NUMINAMATH_CALUDE_circle_area_difference_l4050_405008

theorem circle_area_difference (L W : ℝ) (h1 : L > 0) (h2 : W > 0) 
  (h3 : L * π = 704) (h4 : W * π = 396) : 
  (π * (L / 2)^2 - π * (W / 2)^2) = (L^2 - W^2) / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l4050_405008


namespace NUMINAMATH_CALUDE_probability_of_dime_l4050_405097

theorem probability_of_dime (quarter_value nickel_value penny_value dime_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value total_dime_value : ℚ)
  (h1 : quarter_value = 25/100)
  (h2 : nickel_value = 5/100)
  (h3 : penny_value = 1/100)
  (h4 : dime_value = 10/100)
  (h5 : total_quarter_value = 15)
  (h6 : total_nickel_value = 5)
  (h7 : total_penny_value = 2)
  (h8 : total_dime_value = 12) :
  (total_dime_value / dime_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value) + 
   (total_dime_value / dime_value)) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_dime_l4050_405097


namespace NUMINAMATH_CALUDE_train_length_calculation_l4050_405015

/-- Calculates the length of a train given its speed, the time it takes to pass a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 50 →
  (train_speed * passing_time) - bridge_length = 485 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l4050_405015


namespace NUMINAMATH_CALUDE_complex_simplification_l4050_405067

theorem complex_simplification :
  let i : ℂ := Complex.I
  (1 + i)^2 / (2 - 3*i) = 6/5 - 4/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l4050_405067


namespace NUMINAMATH_CALUDE_equivalent_propositions_l4050_405062

theorem equivalent_propositions (x y : ℝ) :
  (x > 1 ∧ y < -3 → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) := by
sorry

end NUMINAMATH_CALUDE_equivalent_propositions_l4050_405062


namespace NUMINAMATH_CALUDE_sams_basketball_score_l4050_405061

theorem sams_basketball_score (total : ℕ) (friend_score : ℕ) (sam_score : ℕ) :
  total = 87 →
  friend_score = 12 →
  total = sam_score + friend_score →
  sam_score = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_basketball_score_l4050_405061


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l4050_405012

theorem increasing_order_x_z_y (x : ℝ) (h : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l4050_405012


namespace NUMINAMATH_CALUDE_total_students_l4050_405059

theorem total_students (middle_school : ℕ) (elementary_school : ℕ) (high_school : ℕ) : 
  middle_school = 50 → 
  elementary_school = 4 * middle_school - 3 → 
  high_school = 2 * elementary_school → 
  elementary_school + middle_school + high_school = 641 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l4050_405059


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l4050_405010

-- Statement ②
theorem sufficient_not_necessary :
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) :=
sorry

-- Statement ③
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
sorry

-- Statement ④
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l4050_405010


namespace NUMINAMATH_CALUDE_thank_you_cards_percentage_l4050_405023

/-- The percentage of students who gave thank you cards to Ms. Jones -/
def percentage_thank_you_cards (
  total_students : ℕ)
  (gift_card_value : ℚ)
  (total_gift_card_amount : ℚ)
  (gift_card_fraction : ℚ) : ℚ :=
  (total_gift_card_amount / gift_card_value / gift_card_fraction) / total_students * 100

/-- Theorem stating that 30% of Ms. Jones' class gave her thank you cards -/
theorem thank_you_cards_percentage :
  percentage_thank_you_cards 50 10 50 (1/3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_thank_you_cards_percentage_l4050_405023
