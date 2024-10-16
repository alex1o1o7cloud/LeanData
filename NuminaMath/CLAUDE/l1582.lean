import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1582_158281

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 22 + 5 + y) / 5 = 12 → y = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1582_158281


namespace NUMINAMATH_CALUDE_area_of_special_points_triangle_l1582_158280

/-- A triangle with side lengths 18, 24, and 30 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_sides : a = 18 ∧ b = 24 ∧ c = 30

/-- The incenter of a triangle -/
def incenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : RightTriangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of the triangle formed by the incenter, circumcenter, and centroid of a 18-24-30 right triangle is 6 -/
theorem area_of_special_points_triangle (t : RightTriangle) : 
  triangleArea (incenter t) (circumcenter t) (centroid t) = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_special_points_triangle_l1582_158280


namespace NUMINAMATH_CALUDE_post_office_problem_l1582_158252

/-- Proves that given the conditions from the post office problem, each month has 30 days. -/
theorem post_office_problem (letters_per_day : ℕ) (packages_per_day : ℕ) 
  (total_mail : ℕ) (num_months : ℕ) :
  letters_per_day = 60 →
  packages_per_day = 20 →
  total_mail = 14400 →
  num_months = 6 →
  (total_mail / (letters_per_day + packages_per_day)) / num_months = 30 := by
  sorry

end NUMINAMATH_CALUDE_post_office_problem_l1582_158252


namespace NUMINAMATH_CALUDE_third_bouquet_carnations_l1582_158292

/-- Represents a bouquet of carnations -/
structure Bouquet where
  carnations : ℕ

/-- Represents a collection of three bouquets -/
structure ThreeBouquets where
  first : Bouquet
  second : Bouquet
  third : Bouquet

/-- Calculates the average number of carnations in three bouquets -/
def averageCarnations (b : ThreeBouquets) : ℚ :=
  (b.first.carnations + b.second.carnations + b.third.carnations) / 3

/-- Theorem: Given the conditions, the third bouquet must have 13 carnations -/
theorem third_bouquet_carnations (b : ThreeBouquets) 
    (h1 : b.first.carnations = 9)
    (h2 : b.second.carnations = 14)
    (h3 : averageCarnations b = 12) :
    b.third.carnations = 13 := by
  sorry

#check third_bouquet_carnations

end NUMINAMATH_CALUDE_third_bouquet_carnations_l1582_158292


namespace NUMINAMATH_CALUDE_distance_and_intersection_l1582_158278

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 3)

theorem distance_and_intersection :
  let distance := Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2)
  let slope := (point2.2 - point1.2) / (point2.1 - point1.1)
  let y_intercept := point1.2 - slope * point1.1
  let line := fun x => slope * x + y_intercept
  (distance = 4 * Real.sqrt 5) ∧
  (line (-1) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_distance_and_intersection_l1582_158278


namespace NUMINAMATH_CALUDE_simplify_expression_l1582_158223

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1582_158223


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l1582_158218

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l1582_158218


namespace NUMINAMATH_CALUDE_tromino_coverage_l1582_158209

/-- Represents a tromino (L-shaped piece formed from three squares) --/
structure Tromino

/-- Represents a chessboard --/
structure Chessboard (n : ℕ) where
  size : n ≥ 7
  odd : Odd n

/-- Counts the number of black squares on the chessboard --/
def black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

/-- Counts the minimum number of trominoes required to cover all black squares --/
def min_trominoes (n : ℕ) : ℕ := (n + 1)^2 / 4

/-- Theorem stating the minimum number of trominoes required to cover all black squares --/
theorem tromino_coverage (n : ℕ) (board : Chessboard n) :
  min_trominoes n = black_squares n := by sorry

end NUMINAMATH_CALUDE_tromino_coverage_l1582_158209


namespace NUMINAMATH_CALUDE_bisection_method_condition_l1582_158238

/-- A continuous function on a closed interval -/
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

/-- The bisection method is applicable on an interval -/
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

/-- Theorem: For the bisection method to be applicable on an interval [a, b],
    the function f must satisfy f(a) · f(b) < 0 -/
theorem bisection_method_condition (f : ℝ → ℝ) (a b : ℝ) :
  BisectionApplicable f a b → f a * f b < 0 := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_condition_l1582_158238


namespace NUMINAMATH_CALUDE_new_person_weight_l1582_158253

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 100 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1582_158253


namespace NUMINAMATH_CALUDE_fifth_month_sale_l1582_158207

def sale_1 : ℕ := 5420
def sale_2 : ℕ := 5660
def sale_3 : ℕ := 6200
def sale_4 : ℕ := 6350
def sale_6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sale_5 : ℕ),
    sale_5 = num_months * average_sale - (sale_1 + sale_2 + sale_3 + sale_4 + sale_6) ∧
    sale_5 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l1582_158207


namespace NUMINAMATH_CALUDE_area_calculation_l1582_158294

/-- The lower boundary function of the region -/
def lower_bound (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_calculation : area_of_region = 12.875 := by sorry

end NUMINAMATH_CALUDE_area_calculation_l1582_158294


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1582_158202

theorem x_one_minus_f_equals_one :
  let α : ℝ := 3 + 2 * Real.sqrt 2
  let x : ℝ := α ^ 50
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1582_158202


namespace NUMINAMATH_CALUDE_pascal_burger_ratio_l1582_158245

/-- The mass of fats in grams in a Pascal Burger -/
def mass_fats : ℕ := 32

/-- The mass of carbohydrates in grams in a Pascal Burger -/
def mass_carbs : ℕ := 48

/-- The ratio of fats to carbohydrates in a Pascal Burger -/
def fats_to_carbs_ratio : Rat := mass_fats / mass_carbs

theorem pascal_burger_ratio :
  fats_to_carbs_ratio = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_pascal_burger_ratio_l1582_158245


namespace NUMINAMATH_CALUDE_tan_difference_l1582_158234

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end NUMINAMATH_CALUDE_tan_difference_l1582_158234


namespace NUMINAMATH_CALUDE_car_braking_distance_l1582_158211

def braking_sequence (n : ℕ) : ℕ :=
  max (50 - 10 * n) 0

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map braking_sequence |>.sum

theorem car_braking_distance :
  ∃ n : ℕ, total_distance n = 150 ∧ braking_sequence n = 0 :=
sorry

end NUMINAMATH_CALUDE_car_braking_distance_l1582_158211


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1582_158222

def numbers : List ℝ := [12, 18, 25, 33, 40]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 25.6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l1582_158222


namespace NUMINAMATH_CALUDE_highest_number_on_paper_l1582_158240

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.010416666666666666 → n = 96 := by
  sorry

end NUMINAMATH_CALUDE_highest_number_on_paper_l1582_158240


namespace NUMINAMATH_CALUDE_earl_bird_optimal_speed_l1582_158259

/-- Represents the problem of finding the optimal speed for Mr. Earl E. Bird --/
def optimal_speed (distance : ℝ) (late_speed early_speed : ℝ) (late_time early_time : ℝ) : Prop :=
  let exact_time := distance / 70
  (distance / late_speed = exact_time + late_time) ∧
  (distance / early_speed = exact_time - early_time) ∧
  (late_speed < early_speed) ∧
  (70 = early_speed)

/-- Theorem stating that 70 mph is the optimal speed for Mr. Earl E. Bird --/
theorem earl_bird_optimal_speed :
  ∃ (distance : ℝ),
    optimal_speed distance 50 70 (1/12) (1/12) :=
sorry

end NUMINAMATH_CALUDE_earl_bird_optimal_speed_l1582_158259


namespace NUMINAMATH_CALUDE_floor_length_is_20_l1582_158237

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- Theorem stating the length of the floor under given conditions. -/
theorem floor_length_is_20 (floor : RectangularFloor)
  (h1 : floor.length = 3 * floor.breadth)
  (h2 : floor.paintingCost = 400)
  (h3 : floor.paintingRate = 3)
  (h4 : floor.paintingCost / floor.paintingRate = floor.length * floor.breadth) :
  floor.length = 20 := by
  sorry


end NUMINAMATH_CALUDE_floor_length_is_20_l1582_158237


namespace NUMINAMATH_CALUDE_reduced_rate_fraction_is_nine_fourteenths_l1582_158282

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℚ :=
  let total_hours_per_week : ℕ := 7 * 24
  let weekday_reduced_hours : ℕ := 5 * 12
  let weekend_reduced_hours : ℕ := 2 * 24
  let total_reduced_hours : ℕ := weekday_reduced_hours + weekend_reduced_hours
  ↑total_reduced_hours / ↑total_hours_per_week

/-- Proof that the reduced rate fraction is 9/14 -/
theorem reduced_rate_fraction_is_nine_fourteenths :
  reduced_rate_fraction = 9 / 14 :=
by sorry

end NUMINAMATH_CALUDE_reduced_rate_fraction_is_nine_fourteenths_l1582_158282


namespace NUMINAMATH_CALUDE_function_is_identity_or_reflection_l1582_158263

-- Define the function f
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- State the theorem
theorem function_is_identity_or_reflection (a b : ℝ) :
  (∀ x : ℝ, f a b (f a b x) = x) →
  ((a = 1 ∧ b = 0) ∨ ∃ c : ℝ, a = -1 ∧ ∀ x : ℝ, f a b x = -x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_is_identity_or_reflection_l1582_158263


namespace NUMINAMATH_CALUDE_system_no_solution_l1582_158229

-- Define the system of equations
def system (n : ℝ) (x y z : ℝ) : Prop :=
  2*n*x + 3*y = 2 ∧ 3*n*y + 4*z = 3 ∧ 4*x + 2*n*z = 4

-- Theorem statement
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, ¬ system n x y z) ↔ n = -Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_system_no_solution_l1582_158229


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1582_158214

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^12 + 8*x^11 + 18*x^10 + 2048*x^9 - 1638*x^8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1582_158214


namespace NUMINAMATH_CALUDE_museum_visitors_l1582_158286

theorem museum_visitors (V : ℕ) : 
  (∃ E : ℕ, 
    (E + 150 = V) ∧ 
    (E = (3 * V) / 4)) → 
  V = 600 := by
sorry

end NUMINAMATH_CALUDE_museum_visitors_l1582_158286


namespace NUMINAMATH_CALUDE_diego_apple_capacity_l1582_158293

/-- The maximum weight of apples Diego can buy given his carrying capacity and other fruit weights -/
theorem diego_apple_capacity (capacity : ℝ) (watermelon grapes oranges bananas : ℝ) 
  (h_capacity : capacity = 50) 
  (h_watermelon : watermelon = 1.5)
  (h_grapes : grapes = 2.75)
  (h_oranges : oranges = 3.5)
  (h_bananas : bananas = 2.7) :
  capacity - (watermelon + grapes + oranges + bananas) = 39.55 := by
  sorry

#check diego_apple_capacity

end NUMINAMATH_CALUDE_diego_apple_capacity_l1582_158293


namespace NUMINAMATH_CALUDE_hydrangea_spend_1989_to_2021_l1582_158233

/-- The amount spent on hydrangeas from a start year to an end year -/
def hydrangeaSpend (startYear endYear : ℕ) (pricePerPlant : ℚ) : ℚ :=
  (endYear - startYear + 1 : ℕ) * pricePerPlant

/-- Theorem stating the total spend on hydrangeas from 1989 to 2021 -/
theorem hydrangea_spend_1989_to_2021 :
  hydrangeaSpend 1989 2021 20 = 640 := by
  sorry

end NUMINAMATH_CALUDE_hydrangea_spend_1989_to_2021_l1582_158233


namespace NUMINAMATH_CALUDE_existence_of_abcd_l1582_158262

theorem existence_of_abcd (n : ℕ) (h : n > 1) : 
  ∃ (a b c d : ℕ), (a + b = c + d) ∧ (a * b - c * d = 4 * n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abcd_l1582_158262


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1582_158298

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1582_158298


namespace NUMINAMATH_CALUDE_exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l1582_158255

/-- Represents the equation x|x| + px + q = 0 --/
def abs_equation (x p q : ℝ) : Prop :=
  x * abs x + p * x + q = 0

/-- There exists a case where p^2 - 4q < 0 and the equation has real roots --/
theorem exists_roots_when_discriminant_negative :
  ∃ (p q : ℝ), p^2 - 4*q < 0 ∧ (∃ x : ℝ, abs_equation x p q) :=
sorry

/-- There exists a case where p < 0, q > 0, and the equation does not have exactly three real roots --/
theorem not_always_three_roots_when_p_neg_q_pos :
  ∃ (p q : ℝ), p < 0 ∧ q > 0 ∧ ¬(∃! (x y z : ℝ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    abs_equation x p q ∧ abs_equation y p q ∧ abs_equation z p q) :=
sorry

end NUMINAMATH_CALUDE_exists_roots_when_discriminant_negative_not_always_three_roots_when_p_neg_q_pos_l1582_158255


namespace NUMINAMATH_CALUDE_ngon_division_formula_l1582_158251

/-- The number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at a single point. -/
def ngon_division (n : ℕ) : ℕ :=
  (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1

/-- Theorem stating that the number of parts into which the diagonals
    of an n-gon divide it, given that no three diagonals intersect at
    a single point, is equal to the formula derived. -/
theorem ngon_division_formula (n : ℕ) (h : n ≥ 3) :
  ngon_division n = (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ngon_division_formula_l1582_158251


namespace NUMINAMATH_CALUDE_x_values_l1582_158220

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1582_158220


namespace NUMINAMATH_CALUDE_polygon_area_is_six_l1582_158246

/-- The vertices of the polygon -/
def vertices : List (ℤ × ℤ) := [
  (0, 0), (0, 2), (1, 2), (2, 3), (2, 2), (3, 2), (3, 0), (2, 0), (2, 1), (1, 0)
]

/-- Calculate the area of a polygon given its vertices using the Shoelace formula -/
def polygonArea (vs : List (ℤ × ℤ)) : ℚ :=
  let pairs := vs.zip (vs.rotate 1)
  let sum := pairs.foldl (fun acc (p, q) => acc + (p.1 * q.2 - p.2 * q.1)) 0
  (sum.natAbs : ℚ) / 2

/-- The theorem stating that the area of the given polygon is 6 square units -/
theorem polygon_area_is_six :
  polygonArea vertices = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_area_is_six_l1582_158246


namespace NUMINAMATH_CALUDE_scientific_notation_448000_l1582_158254

theorem scientific_notation_448000 : 448000 = 4.48 * (10 : ℝ) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_448000_l1582_158254


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1582_158215

theorem triangle_side_calculation (a b : ℝ) (A B : Real) (hpos : 0 < a) :
  a = Real.sqrt 2 →
  B = 60 * π / 180 →
  A = 45 * π / 180 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1582_158215


namespace NUMINAMATH_CALUDE_cylinder_volume_with_inscribed_sphere_l1582_158258

/-- The volume of a cylinder with an inscribed sphere (tangent to top, bottom, and side) is 2π, 
    given that the volume of the inscribed sphere is 4π/3. -/
theorem cylinder_volume_with_inscribed_sphere (r : ℝ) (h : ℝ) :
  (4 / 3 * π * r^3 = 4 * π / 3) →
  (π * r^2 * h = 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_with_inscribed_sphere_l1582_158258


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1582_158277

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (EF : ℝ)
  (GH : ℝ)
  (EG : ℝ)
  (FH : ℝ)
  (is_trapezoid : EF ≠ GH)

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + t.EG + t.FH

/-- Theorem: The perimeter of the given trapezoid is 38 units -/
theorem trapezoid_perimeter :
  ∃ (t : Trapezoid), t.EF = 10 ∧ t.GH = 14 ∧ t.EG = 7 ∧ t.FH = 7 ∧ perimeter t = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1582_158277


namespace NUMINAMATH_CALUDE_smallest_fraction_l1582_158241

theorem smallest_fraction : 
  let a := 7 / 15
  let b := 5 / 11
  let c := 16 / 33
  let d := 49 / 101
  let e := 89 / 183
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_l1582_158241


namespace NUMINAMATH_CALUDE_min_value_expression_l1582_158232

open Real

theorem min_value_expression (α β : ℝ) (h : α + β = π / 2) :
  (∀ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 65) ∧
  (∃ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 = 65) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1582_158232


namespace NUMINAMATH_CALUDE_range_of_a_l1582_158225

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) ∪ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1582_158225


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l1582_158248

/-- Proves that given a 12-mile trip divided into three equal parts, each taking 15 minutes,
    with speeds of 16 mph and 12 mph for the first two parts respectively,
    the speed for the last part must be 16 mph. -/
theorem bicycle_trip_speed (total_distance : ℝ) (part_time : ℝ) (speed1 speed2 : ℝ) :
  total_distance = 12 →
  part_time = 0.25 →
  speed1 = 16 →
  speed2 = 12 →
  (speed1 * part_time + speed2 * part_time + 4) = total_distance →
  4 / part_time = 16 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_trip_speed_l1582_158248


namespace NUMINAMATH_CALUDE_rabbit_chicken_puzzle_l1582_158208

theorem rabbit_chicken_puzzle (total_animals : ℕ) (rabbit_count : ℕ) : 
  total_animals = 40 →
  4 * rabbit_count = 10 * 2 * (total_animals - rabbit_count) + 8 →
  rabbit_count = 33 := by
sorry

end NUMINAMATH_CALUDE_rabbit_chicken_puzzle_l1582_158208


namespace NUMINAMATH_CALUDE_complex_addition_subtraction_l1582_158201

theorem complex_addition_subtraction : 
  (1 : ℂ) * (5 - 6 * I) + (-2 - 2 * I) - (3 + 3 * I) = -11 * I := by sorry

end NUMINAMATH_CALUDE_complex_addition_subtraction_l1582_158201


namespace NUMINAMATH_CALUDE_dice_probability_relationship_l1582_158256

/-- The probability that the sum of two fair dice does not exceed 5 -/
def p₁ : ℚ := 5/18

/-- The probability that the sum of two fair dice is greater than 5 -/
def p₂ : ℚ := 11/18

/-- The probability that the sum of two fair dice is an even number -/
def p₃ : ℚ := 1/2

/-- Theorem stating the relationship between p₁, p₂, and p₃ -/
theorem dice_probability_relationship : p₁ < p₃ ∧ p₃ < p₂ := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_relationship_l1582_158256


namespace NUMINAMATH_CALUDE_smallest_additional_airplanes_lucas_airplanes_arrangement_l1582_158247

theorem smallest_additional_airplanes (current_airplanes : ℕ) (row_size : ℕ) : ℕ :=
  let next_multiple := (current_airplanes + row_size - 1) / row_size * row_size
  next_multiple - current_airplanes

theorem lucas_airplanes_arrangement :
  smallest_additional_airplanes 37 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_additional_airplanes_lucas_airplanes_arrangement_l1582_158247


namespace NUMINAMATH_CALUDE_junior_score_l1582_158267

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  0.2 * n * junior_score + 0.8 * n * 84 = n * 85 →
  junior_score = 89 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l1582_158267


namespace NUMINAMATH_CALUDE_product_of_binomials_l1582_158283

theorem product_of_binomials (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binomials_l1582_158283


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l1582_158275

theorem gcd_sum_and_sum_of_squares (a b : ℤ) : 
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2) = 1 ∨ Int.gcd (a + b) (a^2 + b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l1582_158275


namespace NUMINAMATH_CALUDE_min_value_of_function_l1582_158250

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x + 5) * (x + 2) / (x + 1) ≥ 9 ∧
  (x + 5) * (x + 2) / (x + 1) = 9 ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1582_158250


namespace NUMINAMATH_CALUDE_base_b_divisibility_l1582_158244

theorem base_b_divisibility (b : ℤ) : b ∈ ({3, 4, 5, 6, 8} : Set ℤ) →
  (b * (2 * b^2 - b - 1)) % 4 ≠ 0 ↔ b = 3 ∨ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_b_divisibility_l1582_158244


namespace NUMINAMATH_CALUDE_optimal_purchase_max_profit_l1582_158210

/-- Represents the types of multimedia --/
inductive MultimediaType
| A
| B

/-- Represents the cost and price of each type of multimedia --/
def cost_price (t : MultimediaType) : ℝ × ℝ :=
  match t with
  | MultimediaType.A => (3, 3.3)
  | MultimediaType.B => (2.4, 2.8)

/-- The total number of sets to purchase --/
def total_sets : ℕ := 50

/-- The total cost in million yuan --/
def total_cost : ℝ := 132

/-- Theorem for part 1 of the problem --/
theorem optimal_purchase :
  ∃ (a b : ℕ),
    a + b = total_sets ∧
    a * (cost_price MultimediaType.A).1 + b * (cost_price MultimediaType.B).1 = total_cost ∧
    a = 20 ∧ b = 30 := by sorry

/-- Function to calculate profit --/
def profit (a : ℕ) : ℝ :=
  let b := total_sets - a
  a * ((cost_price MultimediaType.A).2 - (cost_price MultimediaType.A).1) +
  b * ((cost_price MultimediaType.B).2 - (cost_price MultimediaType.B).1)

/-- Theorem for part 2 of the problem --/
theorem max_profit :
  ∃ (a : ℕ),
    10 < a ∧ a < 20 ∧
    (∀ m, 10 < m → m < 20 → profit m ≤ profit a) ∧
    a = 11 ∧ profit a = 18.9 := by sorry

end NUMINAMATH_CALUDE_optimal_purchase_max_profit_l1582_158210


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1582_158226

theorem system_of_inequalities (x : ℝ) :
  3 * (x + 1) > 5 * x + 4 ∧ (x - 1) / 2 ≤ (2 * x - 1) / 3 → -1 ≤ x ∧ x < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1582_158226


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l1582_158206

def a : ℝ × ℝ := (3, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem orthogonal_vectors (x : ℝ) : 
  (a - b x) • a = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l1582_158206


namespace NUMINAMATH_CALUDE_remainder_theorem_l1582_158269

theorem remainder_theorem (n : ℤ) (h : n % 9 = 3) : (5 * n - 12) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1582_158269


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1582_158265

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 2) / (x^2 - 5 * x + 6) ≤ 3 ↔ 
  (5/3 < x ∧ x ≤ 2) ∨ (3 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1582_158265


namespace NUMINAMATH_CALUDE_nonnegative_integer_representation_l1582_158289

theorem nonnegative_integer_representation (n : ℕ) : 
  ∃ (a b c : ℕ+), n = a^2 + b^2 - c^2 ∧ a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_representation_l1582_158289


namespace NUMINAMATH_CALUDE_probability_one_of_three_cars_wins_l1582_158291

/-- The probability that one of three specific cars wins a race -/
theorem probability_one_of_three_cars_wins (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) :
  total_cars = 12 →
  prob_x = 1 / 6 →
  prob_y = 1 / 10 →
  prob_z = 1 / 8 →
  prob_x + prob_y + prob_z = 47 / 120 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_three_cars_wins_l1582_158291


namespace NUMINAMATH_CALUDE_ahmed_min_grade_l1582_158205

/-- The number of assignments excluding the final one -/
def num_assignments : ℕ := 9

/-- Ahmed's current average score -/
def ahmed_average : ℕ := 91

/-- Emily's current average score -/
def emily_average : ℕ := 92

/-- Sarah's current average score -/
def sarah_average : ℕ := 94

/-- The minimum passing score -/
def min_score : ℕ := 70

/-- The maximum possible score -/
def max_score : ℕ := 100

/-- Emily's score on the final assignment -/
def emily_final : ℕ := 90

/-- Function to calculate the total score -/
def total_score (average : ℕ) (final : ℕ) : ℕ :=
  average * num_assignments + final

/-- Theorem stating the minimum grade Ahmed needs -/
theorem ahmed_min_grade :
  ∀ x : ℕ, 
    (x ≤ max_score) →
    (total_score ahmed_average x > total_score emily_average emily_final) →
    (total_score ahmed_average x > total_score sarah_average min_score) →
    (∀ y : ℕ, y < x → (total_score ahmed_average y ≤ total_score emily_average emily_final ∨
                       total_score ahmed_average y ≤ total_score sarah_average min_score)) →
    x = 98 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_min_grade_l1582_158205


namespace NUMINAMATH_CALUDE_max_m_value_inequality_proof_l1582_158295

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Theorem for part I
theorem max_m_value (m : ℝ) : 
  (∀ x, f x ≥ |m - 1|) → m ≤ 2 :=
sorry

-- Theorem for part II
theorem inequality_proof (a b : ℝ) :
  a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_proof_l1582_158295


namespace NUMINAMATH_CALUDE_circle_areas_equal_l1582_158287

theorem circle_areas_equal (x y : Real) 
  (hx : 2 * Real.pi * x = 10 * Real.pi) 
  (hy : y / 2 = 2.5) : 
  Real.pi * x^2 = Real.pi * y^2 := by
sorry

end NUMINAMATH_CALUDE_circle_areas_equal_l1582_158287


namespace NUMINAMATH_CALUDE_problem_solution_l1582_158231

theorem problem_solution : 
  (∃ x : ℚ, x - 2/11 = -1/3 ∧ x = -5/33) ∧ 
  (-2 - (-1/3 + 1/2) = -13/6) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1582_158231


namespace NUMINAMATH_CALUDE_q_zero_at_two_two_l1582_158274

def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*y^3 + b₈*x^4 + b₉*y^4

theorem q_zero_at_two_two 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) 
  (h₀ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0)
  (h₁ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0)
  (h₂ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0)
  (h₃ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0)
  (h₄ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0)
  (h₅ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0)
  (h₆ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) (-1) = 0)
  (h₇ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0)
  (h₈ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_q_zero_at_two_two_l1582_158274


namespace NUMINAMATH_CALUDE_margie_change_l1582_158270

/-- Calculates the change received after a purchase -/
def change_received (banana_price : ℚ) (orange_price : ℚ) (num_bananas : ℕ) (num_oranges : ℕ) (paid_amount : ℚ) : ℚ :=
  let total_cost := banana_price * num_bananas + orange_price * num_oranges
  paid_amount - total_cost

/-- Proves that Margie received $7.60 in change -/
theorem margie_change : 
  let banana_price : ℚ := 30/100
  let orange_price : ℚ := 60/100
  let num_bananas : ℕ := 4
  let num_oranges : ℕ := 2
  let paid_amount : ℚ := 10
  change_received banana_price orange_price num_bananas num_oranges paid_amount = 76/10 := by
  sorry

#eval change_received (30/100) (60/100) 4 2 10

end NUMINAMATH_CALUDE_margie_change_l1582_158270


namespace NUMINAMATH_CALUDE_irrationality_of_cube_plus_sqrt_two_l1582_158230

theorem irrationality_of_cube_plus_sqrt_two (t : ℝ) :
  (∃ (r : ℚ), t + Real.sqrt 2 = r) → ¬ (∃ (s : ℚ), t^3 + Real.sqrt 2 = s) := by
sorry

end NUMINAMATH_CALUDE_irrationality_of_cube_plus_sqrt_two_l1582_158230


namespace NUMINAMATH_CALUDE_factorization_proof_l1582_158279

theorem factorization_proof (a b : ℝ) : -a^3 + 12*a^2*b - 36*a*b^2 = -a*(a-6*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1582_158279


namespace NUMINAMATH_CALUDE_birdhouse_distance_l1582_158227

/-- Proves that the birdhouse distance is 1200 feet given the problem conditions --/
theorem birdhouse_distance (car_distance : ℝ) (car_speed_mph : ℝ) 
  (lawn_chair_distance_multiplier : ℝ) (lawn_chair_time_multiplier : ℝ)
  (birdhouse_distance_multiplier : ℝ) (birdhouse_speed_percentage : ℝ) :
  car_distance = 200 →
  car_speed_mph = 80 →
  lawn_chair_distance_multiplier = 2 →
  lawn_chair_time_multiplier = 1.5 →
  birdhouse_distance_multiplier = 3 →
  birdhouse_speed_percentage = 0.6 →
  (birdhouse_distance_multiplier * lawn_chair_distance_multiplier * car_distance) = 1200 := by
  sorry

#check birdhouse_distance

end NUMINAMATH_CALUDE_birdhouse_distance_l1582_158227


namespace NUMINAMATH_CALUDE_max_cos_sum_l1582_158228

theorem max_cos_sum (x y : ℝ) 
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (max : ℝ), max = 1 + (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ 
    ∀ (x' y' : ℝ), 
      Real.sin y' + Real.sin x' + Real.cos (3 * x') = 0 →
      Real.sin (2 * y') - Real.sin (2 * x') = Real.cos (4 * x') + Real.cos (2 * x') →
      Real.cos y' + Real.cos x' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_cos_sum_l1582_158228


namespace NUMINAMATH_CALUDE_correct_mark_l1582_158264

theorem correct_mark (num_pupils : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ) : 
  num_pupils = 44 →
  wrong_mark = 67 →
  (wrong_mark - correct_mark : ℚ) = num_pupils / 2 →
  correct_mark = 45 := by
sorry

end NUMINAMATH_CALUDE_correct_mark_l1582_158264


namespace NUMINAMATH_CALUDE_wendys_brother_candy_prove_wendys_brother_candy_l1582_158219

/-- Wendy's candy problem -/
theorem wendys_brother_candy : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (wendys_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ) (brothers_pieces : ℕ) =>
    wendys_boxes * pieces_per_box + brothers_pieces = total_pieces →
    wendys_boxes = 2 →
    pieces_per_box = 3 →
    total_pieces = 12 →
    brothers_pieces = 6

/-- Proof of Wendy's candy problem -/
theorem prove_wendys_brother_candy : wendys_brother_candy 2 3 12 6 := by
  sorry

end NUMINAMATH_CALUDE_wendys_brother_candy_prove_wendys_brother_candy_l1582_158219


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l1582_158257

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2/(1-2*m) + y^2/(m+2) = 1 ∧ (1-2*m)*(m+2) < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m ≤ -2 ∨ m ≥ 1 :=
sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -2 ∨ m > 1/2 :=
sorry

-- Theorem for the range of m when "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -2 < m ∧ m ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l1582_158257


namespace NUMINAMATH_CALUDE_height_difference_is_half_l1582_158203

/-- A circle tangent to the parabola y = x^2 + 1 at two points -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_condition : (a^2 + ((a^2 + 1) - b)^2 = r^2) ∧ 
                      ((-a)^2 + (((-a)^2 + 1) - b)^2 = r^2)
  /-- The circle's center is on the y-axis -/
  center_on_y_axis : b > 0

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ :=
  c.b - (c.a^2 + 1)

/-- Theorem: The height difference is always 1/2 -/
theorem height_difference_is_half (c : TangentCircle) : 
  height_difference c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_half_l1582_158203


namespace NUMINAMATH_CALUDE_expression_is_equation_l1582_158296

-- Define what an equation is
def is_equation (e : Prop) : Prop :=
  ∃ (lhs rhs : ℝ → ℝ → ℝ), e = (∀ r y, lhs r y = rhs r y)

-- Define the expression we want to prove is an equation
def expression (r y : ℝ) : ℝ := 3 * r + y

-- Theorem statement
theorem expression_is_equation :
  is_equation (∀ r y : ℝ, expression r y = 5) :=
sorry

end NUMINAMATH_CALUDE_expression_is_equation_l1582_158296


namespace NUMINAMATH_CALUDE_polygonal_chain_existence_l1582_158216

-- Define a type for points in a plane
def Point := ℝ × ℝ

-- Define a type for lines in a plane
def Line := Point → Point → Prop

-- Define a type for a polygonal chain
def PolygonalChain (n : ℕ) := Fin (n + 1) → Point

-- Define the property of n lines in a plane
def LinesInPlane (n : ℕ) (lines : Fin n → Line) : Prop :=
  -- No two lines are parallel
  ∀ i j, i ≠ j → ¬ (∀ p q, lines i p q ↔ lines j p q) ∧
  -- No three lines intersect at a single point
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ ∃ p, (lines i p p ∧ lines j p p ∧ lines k p p)

-- Define the property of a non-self-intersecting polygonal chain
def NonSelfIntersecting (chain : PolygonalChain n) : Prop :=
  ∀ i j k l, i < j → j < k → k < l → 
    ¬ (∃ p, (chain i = p ∧ chain j = p) ∨ (chain k = p ∧ chain l = p))

-- Define the property that each line contains exactly one segment of the chain
def EachLineOneSegment (n : ℕ) (lines : Fin n → Line) (chain : PolygonalChain n) : Prop :=
  ∀ i, ∃! j, lines i (chain j) (chain (j + 1))

-- The main theorem
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h : LinesInPlane n lines) :
  ∃ chain : PolygonalChain n, NonSelfIntersecting chain ∧ EachLineOneSegment n lines chain :=
sorry

end NUMINAMATH_CALUDE_polygonal_chain_existence_l1582_158216


namespace NUMINAMATH_CALUDE_candy_soda_price_before_increase_l1582_158268

/-- Proves that the total price of a candy box and a soda can before a price increase is 16 pounds, given their initial prices and percentage increases. -/
theorem candy_soda_price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10) 
  (h2 : soda_price = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_price + soda_price = 16 := by
  sorry

#check candy_soda_price_before_increase

end NUMINAMATH_CALUDE_candy_soda_price_before_increase_l1582_158268


namespace NUMINAMATH_CALUDE_storage_volume_calculation_l1582_158266

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ := 27 * yards

/-- Calculates the total volume in cubic feet -/
def total_volume (initial_yards : ℝ) (additional_feet : ℝ) : ℝ :=
  cubic_yards_to_cubic_feet initial_yards + additional_feet

/-- Theorem: The total volume is 180 cubic feet -/
theorem storage_volume_calculation :
  total_volume 5 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_storage_volume_calculation_l1582_158266


namespace NUMINAMATH_CALUDE_brother_age_in_five_years_l1582_158299

/-- Given the ages of Nick and his siblings, prove the brother's age in 5 years -/
theorem brother_age_in_five_years
  (nick_age : ℕ)
  (sister_age_diff : ℕ)
  (h_nick_age : nick_age = 13)
  (h_sister_age_diff : sister_age_diff = 6)
  (brother_age : ℕ)
  (h_brother_age : brother_age = (nick_age + (nick_age + sister_age_diff)) / 2) :
  brother_age + 5 = 21 := by
sorry


end NUMINAMATH_CALUDE_brother_age_in_five_years_l1582_158299


namespace NUMINAMATH_CALUDE_jackson_souvenirs_count_l1582_158212

theorem jackson_souvenirs_count :
  let hermit_crabs : ℕ := 120
  let shells_per_crab : ℕ := 8
  let starfish_per_shell : ℕ := 5
  let sand_dollars_per_starfish : ℕ := 3
  let sand_dollars_per_coral : ℕ := 4

  let total_shells : ℕ := hermit_crabs * shells_per_crab
  let total_starfish : ℕ := total_shells * starfish_per_shell
  let total_sand_dollars : ℕ := total_starfish * sand_dollars_per_starfish
  let total_coral : ℕ := total_sand_dollars / sand_dollars_per_coral

  let total_souvenirs : ℕ := hermit_crabs + total_shells + total_starfish + total_sand_dollars + total_coral

  total_souvenirs = 22880 :=
by
  sorry

end NUMINAMATH_CALUDE_jackson_souvenirs_count_l1582_158212


namespace NUMINAMATH_CALUDE_valid_triangulations_are_4_7_19_l1582_158276

/-- Represents a triangulation of a triangle. -/
structure Triangulation where
  num_triangles : ℕ
  sides_per_vertex : ℕ

/-- Predicate to check if a triangulation is valid according to the problem conditions. -/
def is_valid_triangulation (t : Triangulation) : Prop :=
  t.num_triangles ≤ 19 ∧
  t.num_triangles > 0 ∧
  t.sides_per_vertex > 2

/-- The set of all valid triangulations. -/
def valid_triangulations : Set Triangulation :=
  {t : Triangulation | is_valid_triangulation t}

/-- Theorem stating that the only valid triangulations have 4, 7, or 19 triangles. -/
theorem valid_triangulations_are_4_7_19 :
  ∀ t ∈ valid_triangulations, t.num_triangles = 4 ∨ t.num_triangles = 7 ∨ t.num_triangles = 19 := by
  sorry

end NUMINAMATH_CALUDE_valid_triangulations_are_4_7_19_l1582_158276


namespace NUMINAMATH_CALUDE_max_value_theorem_l1582_158213

theorem max_value_theorem (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 200) :
  ∃ (max_value : ℝ), max_value = 30000 ∧ 
  ∀ (x y z w : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → 
  x + y + z + w = 200 → 2*x*y + 3*y*z + 4*z*w ≤ max_value :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1582_158213


namespace NUMINAMATH_CALUDE_circle_trajectory_l1582_158221

/-- Circle 1 with center (a/2, -1) and radius sqrt((a/2)^2 + 2) -/
def circle1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a/2)^2 + (p.2 + 1)^2 = (a/2)^2 + 2}

/-- Circle 2 with center (0, 0) and radius 1 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line y = x - 1 -/
def symmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Point C(-a, a) -/
def pointC (a : ℝ) : ℝ × ℝ := (-a, a)

/-- Circle P passing through point C(-a, a) and tangent to y-axis -/
def circleP (a : ℝ) (center : ℝ × ℝ) : Prop :=
  (center.1 + a)^2 + (center.2 - a)^2 = center.1^2

/-- Trajectory of the center of circle P -/
def trajectoryP : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 4*p.1 - 4*p.2 + 8 = 0}

theorem circle_trajectory :
  ∃ (a : ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ symmetryLine → (p ∈ circle1 a ↔ p ∈ circle2)) ∧
    (a = 2) ∧
    (∀ (center : ℝ × ℝ), circleP a center → center ∈ trajectoryP) :=
  sorry

end NUMINAMATH_CALUDE_circle_trajectory_l1582_158221


namespace NUMINAMATH_CALUDE_vector_arithmetic_l1582_158239

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the expected result. -/
theorem vector_arithmetic (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_arithmetic_l1582_158239


namespace NUMINAMATH_CALUDE_relationship_abc_l1582_158217

theorem relationship_abc :
  let a := Real.tan (135 * π / 180)
  let b := Real.cos (Real.cos 0)
  let c := (fun x : ℝ => (x^2 + 1/2)^0) 0
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1582_158217


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1582_158242

theorem fraction_subtraction (x : ℝ) (hx : x ≠ 0) : 1 / x - 2 / (3 * x) = 1 / (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1582_158242


namespace NUMINAMATH_CALUDE_no_integer_solution_for_dog_nails_l1582_158260

/-- Represents the number of nails Cassie needs to cut for all her pets -/
def total_nails : ℕ := 113

/-- Represents the number of dogs Cassie has -/
def num_dogs : ℕ := 4

/-- Represents the number of parrots Cassie has -/
def num_parrots : ℕ := 8

/-- Represents the number of nails each parrot has -/
def nails_per_parrot : ℕ := 8

/-- Represents the number of feet each dog has -/
def feet_per_dog : ℕ := 4

/-- Theorem stating that there is no integer solution for the number of nails per dog foot -/
theorem no_integer_solution_for_dog_nails :
  ¬ ∃ (nails_per_dog_foot : ℕ), 
    num_dogs * feet_per_dog * nails_per_dog_foot + num_parrots * nails_per_parrot = total_nails :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_dog_nails_l1582_158260


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l1582_158236

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l1582_158236


namespace NUMINAMATH_CALUDE_sequence_is_geometric_progression_l1582_158273

theorem sequence_is_geometric_progression (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n = (1 : ℚ) / 3 * (a n - 1)) :
  a 1 = -(1 : ℚ) / 2 ∧ 
  a 2 = (1 : ℚ) / 4 ∧ 
  (∀ n : ℕ+, n > 1 → a n / a (n - 1) = -(1 : ℚ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_geometric_progression_l1582_158273


namespace NUMINAMATH_CALUDE_f_comparison_l1582_158288

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f (-x) = f x)
variable (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

-- State the theorem
theorem f_comparison (a : ℝ) : f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_comparison_l1582_158288


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1582_158200

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 ≤ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1582_158200


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1582_158243

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1582_158243


namespace NUMINAMATH_CALUDE_problem_statement_l1582_158297

theorem problem_statement (x y : ℤ) (hx : x = 1) (hy : y = 630) : 2019 * x - 3 * y - 9 = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1582_158297


namespace NUMINAMATH_CALUDE_equation_solution_l1582_158271

theorem equation_solution (x : ℝ) : (4 * x + 2) / (5 * x - 5) = 3 / 4 → x = -23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1582_158271


namespace NUMINAMATH_CALUDE_min_value_of_f_l1582_158224

def f (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1582_158224


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_diagonals_l1582_158284

/-- A convex quadrilateral with side lengths a, b, c, d in sequence, inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  R : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  cyclic : a^2 + b^2 + c^2 + d^2 = 8 * R^2

/-- The diagonals of a quadrilateral are perpendicular -/
def has_perpendicular_diagonals (q : CyclicQuadrilateral) : Prop :=
  ∃ (A B C D : ℝ × ℝ), 
    (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

/-- 
If a convex quadrilateral ABCD with side lengths a, b, c, d in sequence, 
inscribed in a circle with radius R, satisfies a^2 + b^2 + c^2 + d^2 = 8R^2, 
then the diagonals of the quadrilateral are perpendicular.
-/
theorem cyclic_quadrilateral_perpendicular_diagonals (q : CyclicQuadrilateral) :
  has_perpendicular_diagonals q :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_diagonals_l1582_158284


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1582_158290

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_fraction_simplification :
  (1 - i) / (1 + i) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1582_158290


namespace NUMINAMATH_CALUDE_shift_left_one_unit_l1582_158285

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

theorem shift_left_one_unit (p : Parabola) :
  p.a = 2 ∧ p.b = 0 ∧ p.c = -1 →
  let p_shifted := shift_horizontal p 1
  p_shifted.a = 2 ∧ p_shifted.b = 4 ∧ p_shifted.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_shift_left_one_unit_l1582_158285


namespace NUMINAMATH_CALUDE_locus_of_P_l1582_158204

def M : ℝ × ℝ := (0, 5)
def N : ℝ × ℝ := (0, -5)

def perimeter : ℝ := 36

def is_on_locus (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.1^2 / 144 + P.2^2 / 169 = 1)

theorem locus_of_P (P : ℝ × ℝ) : 
  (dist M P + dist N P + dist M N = perimeter) → is_on_locus P :=
by sorry


end NUMINAMATH_CALUDE_locus_of_P_l1582_158204


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1582_158235

theorem complex_exponential_sum (α β γ : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) + Complex.exp (Complex.I * γ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) + Complex.exp (-Complex.I * γ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1582_158235


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1582_158272

theorem sum_of_three_numbers (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 252 → 
  a*b + b*c + c*a = 116 → 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1582_158272


namespace NUMINAMATH_CALUDE_optimal_cylinder_ratio_in_sphere_l1582_158261

/-- Theorem: Optimal cylinder ratio in a sphere -/
theorem optimal_cylinder_ratio_in_sphere
  (S : ℝ) (V : ℝ) (h_S : S > 0) (h_V : V > 0) :
  ∃ (r h : ℝ),
    r > 0 ∧ h > 0 ∧
    π * r^2 * h = V ∧
    r^2 + (h/2)^2 ≤ S^2 ∧
    (∀ (r' h' : ℝ), r' > 0 → h' > 0 →
      π * r'^2 * h' = V →
      r'^2 + (h'/2)^2 ≤ S^2 →
      2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') ∧
    h / r = 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_cylinder_ratio_in_sphere_l1582_158261


namespace NUMINAMATH_CALUDE_existence_of_factors_l1582_158249

theorem existence_of_factors : ∃ (a b c d : ℕ), 
  (10 ≤ a ∧ a < 100) ∧ 
  (10 ≤ b ∧ b < 100) ∧ 
  (10 ≤ c ∧ c < 100) ∧ 
  (10 ≤ d ∧ d < 100) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * a * b * c * d = 2016000 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_factors_l1582_158249
