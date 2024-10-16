import Mathlib

namespace NUMINAMATH_CALUDE_mean_of_other_two_l3720_372072

def numbers : List ℤ := [2179, 2231, 2307, 2375, 2419, 2433]

def sum_of_all : ℤ := numbers.sum

def mean_of_four : ℤ := 2323

def sum_of_four : ℤ := 4 * mean_of_four

theorem mean_of_other_two (h : sum_of_four = 4 * mean_of_four) :
  (sum_of_all - sum_of_four) / 2 = 2321 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_other_two_l3720_372072


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l3720_372025

def number_of_male_actors : ℕ := 4
def number_of_female_actors : ℕ := 5

def arrangement_count (m n : ℕ) : ℕ := sorry

theorem photo_lineup_arrangements :
  arrangement_count number_of_male_actors number_of_female_actors =
    (arrangement_count number_of_female_actors number_of_female_actors) *
    (arrangement_count (number_of_female_actors + 1) number_of_male_actors) -
    2 * (arrangement_count (number_of_female_actors - 1) (number_of_female_actors - 1)) *
    (arrangement_count number_of_female_actors number_of_male_actors) :=
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l3720_372025


namespace NUMINAMATH_CALUDE_cupcakes_per_event_l3720_372042

theorem cupcakes_per_event (total_cupcakes : ℕ) (num_events : ℕ) 
  (h1 : total_cupcakes = 768) 
  (h2 : num_events = 8) :
  total_cupcakes / num_events = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_event_l3720_372042


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_207_l3720_372029

theorem sum_of_last_two_digits_of_9_pow_207 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 9^207 ≡ 10*a + b [ZMOD 100] ∧ a + b = 15 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_207_l3720_372029


namespace NUMINAMATH_CALUDE_base_seven_addition_l3720_372082

/-- Given a base 7 addition problem 5XY₇ + 52₇ = 62X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 2 * 7 + X → X + Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_addition_l3720_372082


namespace NUMINAMATH_CALUDE_population_ratio_problem_l3720_372014

theorem population_ratio_problem (s v : ℝ) 
  (h1 : 0.94 * s = 1.14 * v) : s / v = 57 / 47 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_problem_l3720_372014


namespace NUMINAMATH_CALUDE_max_frisbee_receipts_l3720_372078

/-- Represents the total receipts from frisbee sales -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Proves that the maximum total receipts from frisbee sales is $204 -/
theorem max_frisbee_receipts :
  ∀ x y : ℕ,
  x + y = 60 →
  y ≥ 24 →
  total_receipts x y ≤ 204 :=
by
  sorry

#eval total_receipts 36 24  -- Should output 204

end NUMINAMATH_CALUDE_max_frisbee_receipts_l3720_372078


namespace NUMINAMATH_CALUDE_mrs_hilt_total_miles_l3720_372052

/-- The total miles run by Mrs. Hilt in a week -/
def total_miles (monday wednesday friday : ℕ) : ℕ := monday + wednesday + friday

/-- Theorem: Mrs. Hilt's total miles run in the week is 12 -/
theorem mrs_hilt_total_miles : total_miles 3 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_miles_l3720_372052


namespace NUMINAMATH_CALUDE_trapezoid_determines_plane_l3720_372087

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space --/
structure Trapezoid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  parallel_sides : (A.x - B.x) * (C.y - D.y) = (A.y - B.y) * (C.x - D.x) ∧
                   (A.x - D.x) * (B.y - C.y) = (A.y - D.y) * (B.x - C.x)

/-- A plane in 3D space --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine if a point lies on a plane --/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid uniquely determines a plane --/
theorem trapezoid_determines_plane (t : Trapezoid) :
  ∃! plane : Plane, point_on_plane t.A plane ∧
                    point_on_plane t.B plane ∧
                    point_on_plane t.C plane ∧
                    point_on_plane t.D plane :=
sorry

end NUMINAMATH_CALUDE_trapezoid_determines_plane_l3720_372087


namespace NUMINAMATH_CALUDE_set_membership_implies_x_values_l3720_372085

theorem set_membership_implies_x_values (A : Set ℝ) (x : ℝ) :
  A = {2, 4, x^2 - x} → 6 ∈ A → x = 3 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_x_values_l3720_372085


namespace NUMINAMATH_CALUDE_sum_of_squares_l3720_372050

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3720_372050


namespace NUMINAMATH_CALUDE_water_price_solution_l3720_372074

/-- Represents the problem of calculating water price per gallon -/
def water_price_problem (gallons_per_inch : ℝ) (monday_rain : ℝ) (tuesday_rain : ℝ) (total_revenue : ℝ) : Prop :=
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  let price_per_gallon := total_revenue / total_gallons
  price_per_gallon = 1.20

/-- The main theorem stating the solution to the water pricing problem -/
theorem water_price_solution :
  water_price_problem 15 4 3 126 := by
  sorry

#check water_price_solution

end NUMINAMATH_CALUDE_water_price_solution_l3720_372074


namespace NUMINAMATH_CALUDE_point_quadrant_transformation_l3720_372015

def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_quadrant_transformation (x y : ℝ) :
  third_quadrant x y → fourth_quadrant (-x) (y-1) := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_transformation_l3720_372015


namespace NUMINAMATH_CALUDE_books_read_in_may_l3720_372000

theorem books_read_in_may (may june july total : ℕ) 
  (h1 : june = 6)
  (h2 : july = 10)
  (h3 : total = 18)
  (h4 : may + june + july = total) :
  may = 2 := by
sorry

end NUMINAMATH_CALUDE_books_read_in_may_l3720_372000


namespace NUMINAMATH_CALUDE_largest_n_is_69_l3720_372090

/-- Represents a three-digit number in a given base --/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a three-digit number from a given base to base 10 --/
def to_base_10 (base : ℕ) (num : ThreeDigitNumber base) : ℕ :=
  num.hundreds * base^2 + num.tens * base + num.ones

theorem largest_n_is_69 :
  ∀ (n : ℕ) (base_5 : ThreeDigitNumber 5) (base_9 : ThreeDigitNumber 9),
    n > 0 →
    to_base_10 5 base_5 = n →
    to_base_10 9 base_9 = n →
    base_5.hundreds = base_9.ones →
    base_5.tens = base_9.tens →
    base_5.ones = base_9.hundreds →
    n ≤ 69 :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_69_l3720_372090


namespace NUMINAMATH_CALUDE_a_range_l3720_372020

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - (1/2) * x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4*x

theorem a_range (a : ℝ) :
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0) →
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0 ∧ g a x_0 - (1/2) * x_0^2 + 2*a > 0) →
  a ∈ Set.Ioo (-4/ℯ + 1/ℯ^2) 0 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3720_372020


namespace NUMINAMATH_CALUDE_diamond_expression_result_l3720_372001

/-- The diamond operation defined as a ⋄ b = a - 1/b -/
def diamond (a b : ℚ) : ℚ := a - 1 / b

/-- Theorem stating the result of the given expression -/
theorem diamond_expression_result :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  sorry

end NUMINAMATH_CALUDE_diamond_expression_result_l3720_372001


namespace NUMINAMATH_CALUDE_root_of_polynomial_l3720_372088

theorem root_of_polynomial (a₁ a₂ a₃ a₄ a₅ b : ℤ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
  a₄ ≠ a₅ →
  a₁ + a₂ + a₃ + a₄ + a₅ = 9 →
  (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l3720_372088


namespace NUMINAMATH_CALUDE_pyramid_height_l3720_372031

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let center_to_corner := side * Real.sqrt 2 / 2
  Real.sqrt (apex_to_vertex ^ 2 - center_to_corner ^ 2) = 4 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_l3720_372031


namespace NUMINAMATH_CALUDE_dot_product_calculation_l3720_372005

theorem dot_product_calculation (a b : ℝ × ℝ) : 
  a = (2, 1) → a - 2 • b = (1, 1) → a • b = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_calculation_l3720_372005


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3720_372084

theorem inequality_equivalence (x : ℝ) : 
  (2 < (x - 1)⁻¹ ∧ (x - 1)⁻¹ < 3 ∧ x ≠ 1) ↔ (4/3 < x ∧ x < 3/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3720_372084


namespace NUMINAMATH_CALUDE_second_place_prize_l3720_372099

theorem second_place_prize (total_prize : ℕ) (num_winners : ℕ) (first_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) :
  total_prize = 800 →
  num_winners = 18 →
  first_prize = 200 →
  third_prize = 120 →
  other_prize = 22 →
  (num_winners - 3) * other_prize + first_prize + third_prize + 150 = total_prize :=
by sorry

end NUMINAMATH_CALUDE_second_place_prize_l3720_372099


namespace NUMINAMATH_CALUDE_expression_evaluation_l3720_372044

theorem expression_evaluation :
  let a : ℕ := 3
  let b : ℕ := 2
  let c : ℕ := 1
  ((a^2 + b*c) + (a*b + c))^2 - ((a^2 + b*c) - (a*b + c))^2 = 308 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3720_372044


namespace NUMINAMATH_CALUDE_median_length_l3720_372073

/-- Triangle ABC with given side lengths and median BM --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  BM : ℝ
  h_AB : AB = 5
  h_BC : BC = 12
  h_AC : AC = 13
  h_BM : ∃ m : ℝ, BM = m * Real.sqrt 2

/-- The value of m in the equation BM = m√2 is 13/2 --/
theorem median_length (t : Triangle) : ∃ m : ℝ, t.BM = m * Real.sqrt 2 ∧ m = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_length_l3720_372073


namespace NUMINAMATH_CALUDE_torus_division_theorem_l3720_372045

/-- Represents a torus surface -/
structure TorusSurface where
  -- Add necessary fields here

/-- Represents a path on the torus surface -/
structure PathOnTorus where
  -- Add necessary fields here

/-- Represents the outer equator of the torus -/
def outerEquator : PathOnTorus :=
  sorry

/-- Represents a helical line on the torus -/
def helicalLine : PathOnTorus :=
  sorry

/-- Counts the number of regions a torus surface is divided into when cut along given paths -/
def countRegions (surface : TorusSurface) (path1 path2 : PathOnTorus) : ℕ :=
  sorry

/-- Theorem stating that cutting a torus along its outer equator and a helical line divides it into 3 parts -/
theorem torus_division_theorem (surface : TorusSurface) :
  countRegions surface outerEquator helicalLine = 3 :=
sorry

end NUMINAMATH_CALUDE_torus_division_theorem_l3720_372045


namespace NUMINAMATH_CALUDE_cosine_sum_l3720_372007

theorem cosine_sum (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧
  -Real.pi/2 < β ∧ β < 0 ∧
  Real.cos (Real.pi/4 + α) = 1/3 ∧
  Real.cos (Real.pi/4 - β/2) = Real.sqrt 3/3 →
  Real.cos (α + β/2) = 5 * Real.sqrt 3/9 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_l3720_372007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l3720_372043

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l3720_372043


namespace NUMINAMATH_CALUDE_family_d_members_l3720_372081

/-- Represents the number of members in each family -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The initial number of members in each family -/
def initial : FamilyMembers :=
  { a := 7
    b := 8
    c := 10
    d := 13  -- This is what we want to prove
    e := 6
    f := 10 }

/-- The number of families -/
def numFamilies : ℕ := 6

/-- The number of members who left each family -/
def membersLeft : ℕ := 1

/-- The average number of members after some left -/
def newAverage : ℕ := 8

/-- Theorem: The initial number of members in family d is 13 -/
theorem family_d_members : initial.d = 13 := by sorry

end NUMINAMATH_CALUDE_family_d_members_l3720_372081


namespace NUMINAMATH_CALUDE_triangle_properties_l3720_372036

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h1 : a = 3
  h2 : b = 4
  h3 : B = π/2 + A

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) : Real.tan t.B = -4/3 ∧ t.c = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3720_372036


namespace NUMINAMATH_CALUDE_count_possible_sums_l3720_372076

/-- The set of integers from 1 to 150 -/
def S : Finset ℕ := Finset.range 150

/-- The size of subset C -/
def k : ℕ := 80

/-- The minimum possible sum of k elements from S -/
def min_sum : ℕ := k * (k + 1) / 2

/-- The maximum possible sum of k elements from S -/
def max_sum : ℕ := (Finset.sum S id - (150 - k) * (150 - k + 1) / 2)

/-- The number of possible values for the sum of k elements from S -/
def num_possible_sums : ℕ := max_sum - min_sum + 1

theorem count_possible_sums :
  num_possible_sums = 6844 := by sorry

end NUMINAMATH_CALUDE_count_possible_sums_l3720_372076


namespace NUMINAMATH_CALUDE_M_value_l3720_372059

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4
  | (n + 2) => (2*n + 2)^2 + (2*n + 4)^2 - M n

theorem M_value : M 75 = 22800 := by
  sorry

end NUMINAMATH_CALUDE_M_value_l3720_372059


namespace NUMINAMATH_CALUDE_f_derivative_f_at_one_f_equality_l3720_372022

/-- A function f satisfying f'(x) = 4x^3 for all x and f(1) = -1 -/
def f : ℝ → ℝ :=
  sorry

theorem f_derivative (x : ℝ) : deriv f x = 4 * x^3 :=
  sorry

theorem f_at_one : f 1 = -1 :=
  sorry

theorem f_equality (x : ℝ) : f x = x^4 - 2 :=
  sorry

end NUMINAMATH_CALUDE_f_derivative_f_at_one_f_equality_l3720_372022


namespace NUMINAMATH_CALUDE_parabola_chord_constant_sum_l3720_372023

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = 2x^2 -/
def parabola (p : Point) : Prop :=
  p.y = 2 * p.x^2

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: For the parabola y = 2x^2, if there exists a constant c such that
    for any chord AB passing through the point (0,c), the value
    t = 1/AC^2 + 1/BC^2 is constant, then c = 1/4 and t = 8 -/
theorem parabola_chord_constant_sum (c : ℝ) :
  (∃ t : ℝ, ∀ A B : Point,
    parabola A ∧ parabola B ∧
    (∃ m : ℝ, A.y = m * A.x + c ∧ B.y = m * B.x + c) →
    1 / distanceSquared A ⟨0, c⟩ + 1 / distanceSquared B ⟨0, c⟩ = t) →
  c = 1/4 ∧ t = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_constant_sum_l3720_372023


namespace NUMINAMATH_CALUDE_min_wins_to_advance_exactly_ten_wins_l3720_372056

def football_advancement (total_matches win_matches loss_matches : ℕ) : Prop :=
  let draw_matches := total_matches - win_matches - loss_matches
  3 * win_matches + draw_matches ≥ 33

theorem min_wins_to_advance :
  ∀ win_matches : ℕ,
    football_advancement 15 win_matches 2 →
    win_matches ≥ 10 :=
by
  sorry

theorem exactly_ten_wins :
  football_advancement 15 10 2 ∧
  ∀ win_matches : ℕ, win_matches < 10 → ¬(football_advancement 15 win_matches 2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_wins_to_advance_exactly_ten_wins_l3720_372056


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3720_372063

/-- Represents the number of fish of a particular species in the pond -/
structure FishPopulation where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Represents the number of tagged fish caught in the second round -/
structure TaggedCatch where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Calculates the estimated population of a species based on the initial tagging and second catch -/
def estimatePopulation (initialTagged : ℕ) (secondCatchTotal : ℕ) (taggedInSecondCatch : ℕ) : ℕ :=
  (initialTagged * secondCatchTotal) / taggedInSecondCatch

/-- Theorem stating the estimated fish population given the initial tagging and second catch data -/
theorem fish_population_estimate 
  (initialTagged : ℕ) 
  (secondCatchTotal : ℕ) 
  (taggedCatch : TaggedCatch) : 
  initialTagged = 40 →
  secondCatchTotal = 180 →
  taggedCatch.speciesA = 3 →
  taggedCatch.speciesB = 5 →
  taggedCatch.speciesC = 2 →
  let estimatedPopulation := FishPopulation.mk
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesA)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesB)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesC)
  estimatedPopulation.speciesA = 2400 ∧ 
  estimatedPopulation.speciesB = 1440 ∧ 
  estimatedPopulation.speciesC = 3600 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l3720_372063


namespace NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_irrational_l3720_372010

theorem sqrt2_plus_sqrt3_irrational : Irrational (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_plus_sqrt3_irrational_l3720_372010


namespace NUMINAMATH_CALUDE_remainder_3n_div_7_l3720_372054

theorem remainder_3n_div_7 (n : Int) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3n_div_7_l3720_372054


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3720_372033

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- State that AB is perpendicular to x-axis
def AB_perpendicular_to_x : Prop := sorry

-- Define the perimeter of triangle AF₁B
def perimeter_AF1B : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  AB_perpendicular_to_x →
  perimeter_AF1B = 24 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3720_372033


namespace NUMINAMATH_CALUDE_max_yellow_balls_l3720_372086

/-- Represents the total number of balls -/
def n : ℕ := 91

/-- Represents the number of yellow balls in the first 70 picked -/
def initial_yellow : ℕ := 63

/-- Represents the total number of balls initially picked -/
def initial_total : ℕ := 70

/-- Represents the number of yellow balls in each subsequent batch of 7 -/
def batch_yellow : ℕ := 5

/-- Represents the total number of balls in each subsequent batch -/
def batch_total : ℕ := 7

/-- The minimum percentage of yellow balls required -/
def min_percentage : ℚ := 85 / 100

theorem max_yellow_balls :
  n = initial_total + batch_total * ((n - initial_total) / batch_total) ∧
  (initial_yellow + batch_yellow * ((n - initial_total) / batch_total)) / n ≥ min_percentage ∧
  ∀ m : ℕ, m > n →
    (initial_yellow + batch_yellow * ((m - initial_total) / batch_total)) / m < min_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_yellow_balls_l3720_372086


namespace NUMINAMATH_CALUDE_solution_set_equality_l3720_372062

theorem solution_set_equality (m : ℝ) : 
  (Set.Iio m = {x : ℝ | 2 * x + 1 < 5}) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3720_372062


namespace NUMINAMATH_CALUDE_ice_cream_profit_l3720_372003

/-- Proves the number of ice cream cones needed to be sold for a specific profit -/
theorem ice_cream_profit (cone_price : ℚ) (expense_ratio : ℚ) (target_profit : ℚ) :
  cone_price = 5 →
  expense_ratio = 4/5 →
  target_profit = 200 →
  (target_profit / (1 - expense_ratio)) / cone_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_profit_l3720_372003


namespace NUMINAMATH_CALUDE_quiz_probability_l3720_372006

theorem quiz_probability (n : ℕ) : 
  (1 : ℚ) / 3 * (1 / 2) ^ n = 1 / 12 → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l3720_372006


namespace NUMINAMATH_CALUDE_power_division_equality_l3720_372096

theorem power_division_equality (a b : ℝ) : (a^2 * b)^3 / ((-a * b)^2) = a^4 * b := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l3720_372096


namespace NUMINAMATH_CALUDE_sqrt_2023_bound_l3720_372098

theorem sqrt_2023_bound (n : ℤ) : n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_bound_l3720_372098


namespace NUMINAMATH_CALUDE_scaled_model_height_l3720_372065

/-- Represents a cylindrical monument --/
structure CylindricalMonument where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- Represents a scaled model of the monument --/
structure ScaledModel where
  volume : ℝ
  height : ℝ

/-- Theorem stating the relationship between the original monument and its scaled model --/
theorem scaled_model_height 
  (monument : CylindricalMonument) 
  (model : ScaledModel) : 
  monument.height = 100 ∧ 
  monument.baseRadius = 20 ∧ 
  monument.volume = 125600 ∧ 
  model.volume = 1.256 → 
  model.height = 1 := by
  sorry


end NUMINAMATH_CALUDE_scaled_model_height_l3720_372065


namespace NUMINAMATH_CALUDE_contractor_engagement_l3720_372080

/-- Contractor engagement problem -/
theorem contractor_engagement
  (daily_wage : ℝ)
  (daily_fine : ℝ)
  (total_payment : ℝ)
  (absent_days : ℕ)
  (h1 : daily_wage = 25)
  (h2 : daily_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10) :
  ∃ (worked_days : ℕ) (total_days : ℕ),
    worked_days * daily_wage - absent_days * daily_fine = total_payment ∧
    total_days = worked_days + absent_days ∧
    total_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_l3720_372080


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l3720_372075

/-- A function is a quadratic radical if it can be expressed as the square root of a non-negative real-valued expression. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(a² + 1) is a quadratic radical. -/
theorem sqrt_a_squared_plus_one_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (a^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_one_is_quadratic_radical_l3720_372075


namespace NUMINAMATH_CALUDE_pirate_coins_l3720_372097

/-- Represents the number of pirates --/
def num_pirates : ℕ := 15

/-- Calculates the number of coins remaining after the k-th pirate takes their share --/
def coins_after (k : ℕ) (initial_coins : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_coins

/-- Checks if a given number of initial coins results in each pirate receiving a whole number of coins --/
def valid_distribution (initial_coins : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ num_pirates → (coins_after k initial_coins - coins_after (k+1) initial_coins).isInt

/-- The statement to be proved --/
theorem pirate_coins :
  ∃ initial_coins : ℕ,
    valid_distribution initial_coins ∧
    (∀ n : ℕ, n < initial_coins → ¬valid_distribution n) ∧
    coins_after (num_pirates - 1) initial_coins = 1001 := by
  sorry


end NUMINAMATH_CALUDE_pirate_coins_l3720_372097


namespace NUMINAMATH_CALUDE_age_difference_proof_l3720_372011

/-- Given two persons with an age difference of 16 years, where the elder is currently 30 years old,
    this theorem proves that 6 years ago, the elder person was three times as old as the younger one. -/
theorem age_difference_proof :
  ∀ (younger_age elder_age : ℕ) (years_ago : ℕ),
    elder_age = 30 →
    elder_age = younger_age + 16 →
    elder_age - years_ago = 3 * (younger_age - years_ago) →
    years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3720_372011


namespace NUMINAMATH_CALUDE_sufficient_condition_l3720_372048

theorem sufficient_condition (x y : ℝ) : x > 3 ∧ y > 3 → x + y > 6 ∧ x * y > 9 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l3720_372048


namespace NUMINAMATH_CALUDE_crayons_lost_l3720_372017

/-- Given that Paul gave away 52 crayons and lost or gave away a total of 587 crayons,
    prove that the number of crayons he lost is 535. -/
theorem crayons_lost (crayons_given_away : ℕ) (total_lost_or_given_away : ℕ)
    (h1 : crayons_given_away = 52)
    (h2 : total_lost_or_given_away = 587) :
    total_lost_or_given_away - crayons_given_away = 535 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_l3720_372017


namespace NUMINAMATH_CALUDE_lg_24_in_terms_of_a_b_l3720_372077

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_24_in_terms_of_a_b (a b : ℝ) (h1 : lg 6 = a) (h2 : lg 12 = b) :
  lg 24 = 2 * b - a := by
  sorry

end NUMINAMATH_CALUDE_lg_24_in_terms_of_a_b_l3720_372077


namespace NUMINAMATH_CALUDE_one_point_zero_six_million_scientific_notation_l3720_372034

theorem one_point_zero_six_million_scientific_notation :
  (1.06 : ℝ) * (1000000 : ℝ) = (1.06 : ℝ) * (10 ^ 6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_one_point_zero_six_million_scientific_notation_l3720_372034


namespace NUMINAMATH_CALUDE_tangent_line_proof_l3720_372070

noncomputable def f (x : ℝ) : ℝ := -(1/2) * x + Real.log x

theorem tangent_line_proof :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  f x₀ = (1/2) * x₀ - 1 ∧
  deriv f x₀ = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l3720_372070


namespace NUMINAMATH_CALUDE_adults_on_bus_l3720_372068

theorem adults_on_bus (total_passengers : ℕ) (children_fraction : ℚ) : 
  total_passengers = 360 → children_fraction = 3/7 → 
  (total_passengers : ℚ) * (1 - children_fraction) = 205 := by
sorry

end NUMINAMATH_CALUDE_adults_on_bus_l3720_372068


namespace NUMINAMATH_CALUDE_andrews_age_proof_l3720_372037

/-- Andrew's age in years -/
def andrew_age : ℝ := 7.875

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℝ := 9 * andrew_age

/-- Age difference between Andrew and his grandfather at Andrew's birth -/
def age_difference : ℝ := 63

theorem andrews_age_proof : 
  grandfather_age - andrew_age = age_difference ∧ 
  grandfather_age = 9 * andrew_age ∧ 
  andrew_age = 7.875 := by sorry

end NUMINAMATH_CALUDE_andrews_age_proof_l3720_372037


namespace NUMINAMATH_CALUDE_simplify_fraction_l3720_372024

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  (1 - 1 / x) / ((1 - x^2) / x) = -1 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3720_372024


namespace NUMINAMATH_CALUDE_officer_average_salary_l3720_372095

/-- Proves that the average salary of officers is 420 Rs/month given the specified conditions -/
theorem officer_average_salary
  (total_employees : ℕ)
  (officers : ℕ)
  (non_officers : ℕ)
  (average_salary : ℚ)
  (non_officer_salary : ℚ)
  (h1 : total_employees = officers + non_officers)
  (h2 : total_employees = 465)
  (h3 : officers = 15)
  (h4 : non_officers = 450)
  (h5 : average_salary = 120)
  (h6 : non_officer_salary = 110) :
  (total_employees * average_salary - non_officers * non_officer_salary) / officers = 420 :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l3720_372095


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l3720_372026

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l3720_372026


namespace NUMINAMATH_CALUDE_factors_of_60_l3720_372069

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_60_l3720_372069


namespace NUMINAMATH_CALUDE_det_A_equals_l3720_372071

-- Define the matrix as a function of y
def A (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y^2 + 1, 2*y, 2*y;
     2*y, y^2 + 3, 2*y;
     2*y, 2*y, y^2 + 5]

-- State the theorem
theorem det_A_equals (y : ℝ) : 
  Matrix.det (A y) = y^6 + y^4 + 35*y^2 + 15 - 32*y := by
  sorry

end NUMINAMATH_CALUDE_det_A_equals_l3720_372071


namespace NUMINAMATH_CALUDE_no_return_after_12_jumps_all_return_after_13_jumps_l3720_372004

/-- Represents a point on a circle -/
structure CirclePoint where
  position : ℕ

/-- The number of points on the circle -/
def n : ℕ := 12

/-- The jump function that moves a point to the next clockwise midpoint -/
def jump (p : CirclePoint) : CirclePoint :=
  ⟨(p.position + 1) % n⟩

/-- Applies the jump function k times -/
def jumpK (p : CirclePoint) (k : ℕ) : CirclePoint :=
  match k with
  | 0 => p
  | k + 1 => jump (jumpK p k)

theorem no_return_after_12_jumps :
  ∀ p : CirclePoint, jumpK p 12 ≠ p :=
sorry

theorem all_return_after_13_jumps :
  ∀ p : CirclePoint, jumpK p 13 = p :=
sorry

end NUMINAMATH_CALUDE_no_return_after_12_jumps_all_return_after_13_jumps_l3720_372004


namespace NUMINAMATH_CALUDE_olivias_initial_amount_l3720_372046

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount Olivia spent at the supermarket -/
def supermarket_expense : ℕ := 31

/-- The amount Olivia spent at the showroom -/
def showroom_expense : ℕ := 49

/-- The amount Olivia had left after spending -/
def remaining_amount : ℕ := 26

/-- Theorem stating that Olivia's initial amount was $106 -/
theorem olivias_initial_amount : 
  initial_amount = supermarket_expense + showroom_expense + remaining_amount := by sorry

end NUMINAMATH_CALUDE_olivias_initial_amount_l3720_372046


namespace NUMINAMATH_CALUDE_problem_solution_l3720_372066

theorem problem_solution : 48 / (7 - 3/4 + 1/8) = 128/17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3720_372066


namespace NUMINAMATH_CALUDE_range_of_fraction_l3720_372094

theorem range_of_fraction (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 ∧ ∃ (a' b' : ℝ), b' > a' ∧ a' > 0 ∧ a' * b' = 2 ∧ (a'^2 + b'^2) / (a' - b') = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3720_372094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3720_372035

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_ratio 
  (a x b : ℝ) 
  (h1 : ∃ d, arithmetic_sequence a d 1 = a ∧ 
             arithmetic_sequence a d 2 = x ∧ 
             arithmetic_sequence a d 3 = b ∧ 
             arithmetic_sequence a d 4 = 2*x) :
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3720_372035


namespace NUMINAMATH_CALUDE_algebraic_expression_solution_l3720_372061

theorem algebraic_expression_solution (m : ℚ) : 
  (5 * (2 - 1) + 3 * m * 2 = -7) → 
  (∃ x : ℚ, 5 * (x - 1) + 3 * m * x = -1 ∧ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_solution_l3720_372061


namespace NUMINAMATH_CALUDE_jamal_shelving_problem_l3720_372093

/-- The number of books Jamal still has to shelve after working through different sections of the library. -/
def books_to_shelve (initial : ℕ) (history : ℕ) (fiction : ℕ) (children : ℕ) (misplaced : ℕ) : ℕ :=
  initial - history - fiction - children + misplaced

/-- Theorem stating that Jamal has 16 books left to shelve given the specific numbers from the problem. -/
theorem jamal_shelving_problem :
  books_to_shelve 51 12 19 8 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jamal_shelving_problem_l3720_372093


namespace NUMINAMATH_CALUDE_min_overlap_blue_eyes_lunch_box_l3720_372039

theorem min_overlap_blue_eyes_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 18) :
  blue_eyes + lunch_box - total_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_blue_eyes_lunch_box_l3720_372039


namespace NUMINAMATH_CALUDE_calcium_iodide_weight_l3720_372058

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem calcium_iodide_weight : total_weight_CaI2 = 1469.4 := by
  sorry

end NUMINAMATH_CALUDE_calcium_iodide_weight_l3720_372058


namespace NUMINAMATH_CALUDE_mikaela_tiled_walls_l3720_372027

/-- Calculates the number of walls tiled instead of painted given the initial number of paint containers, 
    total number of walls, containers used for the ceiling, and containers left over. -/
def walls_tiled (initial_containers : ℕ) (total_walls : ℕ) (ceiling_containers : ℕ) (leftover_containers : ℕ) : ℕ :=
  total_walls - (initial_containers - ceiling_containers - leftover_containers) / (initial_containers / total_walls)

/-- Proves that given 16 containers of paint initially, 4 equally-sized walls, 1 container used for the ceiling, 
    and 3 containers left over, the number of walls tiled instead of painted is 1. -/
theorem mikaela_tiled_walls :
  walls_tiled 16 4 1 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_tiled_walls_l3720_372027


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l3720_372018

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.length)

/-- Theorem: Optimal dimensions for a 600 sq ft garden with length twice the width -/
theorem optimal_garden_dimensions :
  ∃ (g : RectangularGarden),
    area g = 600 ∧
    g.length = 2 * g.width ∧
    g.width = 10 * Real.sqrt 3 ∧
    g.length = 20 * Real.sqrt 3 ∧
    ∀ (h : RectangularGarden),
      area h = 600 → h.length = 2 * h.width → perimeter h ≥ perimeter g :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l3720_372018


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3720_372089

/-- Definition of a repeating decimal with a single digit repeating -/
def repeating_decimal (d : ℕ) : ℚ := (d : ℚ) / 9

/-- The problem statement -/
theorem repeating_decimal_sum : 
  repeating_decimal 6 + repeating_decimal 2 - repeating_decimal 4 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3720_372089


namespace NUMINAMATH_CALUDE_candy_fraction_of_earnings_l3720_372079

/-- Proves that the fraction of earnings spent on candy is 1/6 -/
theorem candy_fraction_of_earnings : 
  ∀ (candy_bar_price lollipop_price driveway_charge : ℚ)
    (candy_bars lollipops driveways : ℕ),
  candy_bar_price = 3/4 →
  lollipop_price = 1/4 →
  driveway_charge = 3/2 →
  candy_bars = 2 →
  lollipops = 4 →
  driveways = 10 →
  (candy_bar_price * candy_bars + lollipop_price * lollipops) / 
  (driveway_charge * driveways) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_candy_fraction_of_earnings_l3720_372079


namespace NUMINAMATH_CALUDE_survey_support_l3720_372021

theorem survey_support (N A B N_o : ℕ) (h1 : N = 198) (h2 : A = 149) (h3 : B = 119) (h4 : N_o = 29) :
  A + B - (N - N_o) = 99 :=
by sorry

end NUMINAMATH_CALUDE_survey_support_l3720_372021


namespace NUMINAMATH_CALUDE_halfway_fraction_l3720_372040

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3720_372040


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3720_372047

theorem trigonometric_identity : 
  Real.cos (π / 12) * Real.cos (5 * π / 12) + Real.cos (π / 8)^2 - 1/2 = (Real.sqrt 2 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3720_372047


namespace NUMINAMATH_CALUDE_min_omega_value_l3720_372053

/-- Given that ω > 0 and the graph of y = 2cos(ωx + π/5) - 1 overlaps with itself
    after shifting right by 5π/4 units, prove that the minimum value of ω is 8/5. -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, 2 * Real.cos (ω * x + π / 5) - 1 = 2 * Real.cos (ω * (x + 5 * π / 4) + π / 5) - 1) :
  ω ≥ 8 / 5 ∧ ∀ ω' > 0, (∀ x : ℝ, 2 * Real.cos (ω' * x + π / 5) - 1 = 2 * Real.cos (ω' * (x + 5 * π / 4) + π / 5) - 1) → ω' ≥ ω :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3720_372053


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l3720_372060

theorem consecutive_product_not_power (x a n : ℕ) : 
  a ≥ 2 → n ≥ 2 → (x - 1) * x * (x + 1) ≠ a^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l3720_372060


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l3720_372028

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l3720_372028


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l3720_372008

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  let purple_tulips : ℕ := 2150
  let pink_daisies : ℕ := 3500
  let blue_irises : ℕ := 2973
  let orange_marigolds : ℕ := 4234
  let lavender_orchids : ℕ := 350
  let orchid_pots : ℕ := 5
  let sunflower_boxes : ℕ := 7
  let sunflowers_per_box : ℕ := 120
  let sunflowers_last_box : ℕ := 95
  let violet_lily_pairs : ℕ := 13

  red_roses + yellow_carnations + white_roses + purple_tulips + 
  pink_daisies + blue_irises + orange_marigolds + lavender_orchids + 
  (sunflower_boxes - 1) * sunflowers_per_box + sunflowers_last_box + 
  2 * violet_lily_pairs = 21332 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l3720_372008


namespace NUMINAMATH_CALUDE_line_through_points_l3720_372019

/-- Given a line with equation x = 3y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3720_372019


namespace NUMINAMATH_CALUDE_find_a_and_b_l3720_372038

theorem find_a_and_b (a b d : ℤ) : 
  (∃ x : ℝ, Real.sqrt (x - a) + Real.sqrt (x + b) = 7 ∧ x = 12) →
  (∃ x : ℝ, Real.sqrt (x + a) + Real.sqrt (x + d) = 7 ∧ x = 13) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_find_a_and_b_l3720_372038


namespace NUMINAMATH_CALUDE_sum_mod_twelve_l3720_372009

theorem sum_mod_twelve : (2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_twelve_l3720_372009


namespace NUMINAMATH_CALUDE_area_of_region_l3720_372032

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 77 ∧ 
   A = Real.pi * (((x + 8)^2 + (y - 3)^2) / 4) ∧
   x^2 + y^2 - 8 = 6*y - 16*x + 4) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l3720_372032


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3720_372092

def original_price : ℝ := 100
def discount_percentage : ℝ := 25

theorem sale_price_calculation :
  let discount_amount := (discount_percentage / 100) * original_price
  let sale_price := original_price - discount_amount
  sale_price = 75 := by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3720_372092


namespace NUMINAMATH_CALUDE_cube_volume_from_paper_area_l3720_372067

/-- Given a rectangular piece of paper with length 48 inches and width 72 inches
    that covers exactly the surface area of a cube, prove that the volume of the cube
    is 8 cubic feet, where 1 foot is 12 inches. -/
theorem cube_volume_from_paper_area (paper_length : ℝ) (paper_width : ℝ) 
    (inches_per_foot : ℝ) (h1 : paper_length = 48) (h2 : paper_width = 72) 
    (h3 : inches_per_foot = 12) : 
    let paper_area := paper_length * paper_width
    let cube_side_length := Real.sqrt (paper_area / 6)
    let cube_side_length_feet := cube_side_length / inches_per_foot
    cube_side_length_feet ^ 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_paper_area_l3720_372067


namespace NUMINAMATH_CALUDE_cars_without_features_l3720_372055

theorem cars_without_features (total : ℕ) (air_bag : ℕ) (power_windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : air_bag = 45)
  (h3 : power_windows = 30)
  (h4 : both = 12) :
  total - (air_bag + power_windows - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_features_l3720_372055


namespace NUMINAMATH_CALUDE_visitors_to_both_countries_l3720_372030

theorem visitors_to_both_countries 
  (total : ℕ) 
  (iceland : ℕ) 
  (norway : ℕ) 
  (neither : ℕ) 
  (h1 : total = 60) 
  (h2 : iceland = 35) 
  (h3 : norway = 23) 
  (h4 : neither = 33) : 
  ∃ (both : ℕ), both = 31 ∧ 
    total = iceland + norway - both + neither :=
by sorry

end NUMINAMATH_CALUDE_visitors_to_both_countries_l3720_372030


namespace NUMINAMATH_CALUDE_parabola_satisfies_conditions_l3720_372049

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 8*x - 8*y + 16 = 0

-- Define the conditions
def passes_through_point (eq : ℝ → ℝ → Prop) : Prop :=
  eq 2 8

def focus_y_coordinate (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ x, eq x 4 ∧ (∀ y, eq x y → (y - 4)^2 ≤ (8 - 4)^2)

def axis_parallel_to_x (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y → eq x (-y + 8)

def vertex_on_y_axis (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ y, eq 0 y

def coefficients_are_integers (a b c d e f : ℤ) : Prop :=
  ∀ x y : ℝ, (a:ℝ)*x^2 + (b:ℝ)*x*y + (c:ℝ)*y^2 + (d:ℝ)*x + (e:ℝ)*y + (f:ℝ) = 0 ↔ parabola_equation x y

def c_is_positive (c : ℤ) : Prop :=
  c > 0

def gcd_is_one (a b c d e f : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs)) (d.natAbs)) (e.natAbs)) (f.natAbs) = 1

-- State the theorem
theorem parabola_satisfies_conditions :
  ∃ a b c d e f : ℤ,
    passes_through_point parabola_equation ∧
    focus_y_coordinate parabola_equation ∧
    axis_parallel_to_x parabola_equation ∧
    vertex_on_y_axis parabola_equation ∧
    coefficients_are_integers a b c d e f ∧
    c_is_positive c ∧
    gcd_is_one a b c d e f :=
  sorry

end NUMINAMATH_CALUDE_parabola_satisfies_conditions_l3720_372049


namespace NUMINAMATH_CALUDE_critical_point_theorem_l3720_372012

def sequence_property (x : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0) ∧
  (8 * x 2 - 7 * x 1) * (x 1)^7 = 8 ∧
  (∀ k ≥ 2, (x (k+1)) * (x (k-1)) - (x k)^2 = ((x (k-1))^8 - (x k)^8) / ((x k)^7 * (x (k-1))^7))

def monotonically_decreasing (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n+1) ≤ x n

def not_monotonic (x : ℕ → ℝ) : Prop :=
  ∃ m n, m < n ∧ x m < x n

theorem critical_point_theorem (x : ℕ → ℝ) (h : sequence_property x) :
  ∃ a : ℝ, a = 8^(1/8) ∧
    ((x 1 > a → monotonically_decreasing x) ∧
     (0 < x 1 ∧ x 1 < a → not_monotonic x)) :=
sorry

end NUMINAMATH_CALUDE_critical_point_theorem_l3720_372012


namespace NUMINAMATH_CALUDE_cubic_minus_x_factorization_l3720_372013

theorem cubic_minus_x_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_x_factorization_l3720_372013


namespace NUMINAMATH_CALUDE_reading_assignment_valid_l3720_372002

/-- Represents the reading assignment for Alice, Bob, and Chandra -/
structure ReadingAssignment where
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  total_pages : ℕ

/-- Calculates the time spent reading for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Proves that the given reading assignment satisfies the conditions -/
theorem reading_assignment_valid (ra : ReadingAssignment) 
  (h_alice : ra.alice_pages = 416)
  (h_bob : ra.bob_pages = 208)
  (h_chandra : ra.chandra_pages = 276)
  (h_alice_speed : ra.alice_speed = 18)
  (h_bob_speed : ra.bob_speed = 36)
  (h_chandra_speed : ra.chandra_speed = 27)
  (h_total : ra.total_pages = 900) : 
  ra.alice_pages + ra.bob_pages + ra.chandra_pages = ra.total_pages ∧
  reading_time ra.alice_pages ra.alice_speed = reading_time ra.bob_pages ra.bob_speed ∧
  reading_time ra.bob_pages ra.bob_speed = reading_time ra.chandra_pages ra.chandra_speed :=
by sorry


end NUMINAMATH_CALUDE_reading_assignment_valid_l3720_372002


namespace NUMINAMATH_CALUDE_circle_circumference_l3720_372064

/-- The circumference of a circle with radius 36 is 72π -/
theorem circle_circumference (π : ℝ) (h : π > 0) : ∃ (k : ℝ), k * π = 2 * π * 36 ∧ k = 72 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l3720_372064


namespace NUMINAMATH_CALUDE_accurate_to_hundreds_place_l3720_372091

/-- Represents a number with a specified precision --/
structure PreciseNumber where
  value : ℝ
  precision : ℕ

/-- Defines what it means for a number to be accurate to a certain place value --/
def accurate_to (n : PreciseNumber) (place : ℕ) : Prop :=
  ∃ (m : ℤ), n.value = (m : ℝ) * (10 : ℝ) ^ place ∧ n.precision = place

/-- The statement to be proved --/
theorem accurate_to_hundreds_place :
  let n : PreciseNumber := ⟨4.0 * 10^3, 2⟩
  accurate_to n 2 :=
sorry

end NUMINAMATH_CALUDE_accurate_to_hundreds_place_l3720_372091


namespace NUMINAMATH_CALUDE_fold_sequence_counts_l3720_372041

/-- Represents the possible shapes after folding -/
inductive Shape
  | Square
  | IsoscelesTriangle
  | Rectangle (k : ℕ)

/-- Represents a sequence of folds -/
def FoldSequence := List Shape

/-- Counts the number of possible folding sequences -/
def countFoldSequences (n : ℕ) : ℕ :=
  sorry

theorem fold_sequence_counts :
  (countFoldSequences 3 = 5) ∧
  (countFoldSequences 6 = 24) ∧
  (countFoldSequences 9 = 149) := by
  sorry

end NUMINAMATH_CALUDE_fold_sequence_counts_l3720_372041


namespace NUMINAMATH_CALUDE_boat_speed_theorem_l3720_372051

/-- Represents the speed of a boat in different conditions --/
structure BoatSpeed where
  stillWater : ℝ  -- Speed in still water
  upstream : ℝ    -- Speed against the stream
  downstream : ℝ  -- Speed with the stream

/-- 
Given a man's rowing speed in still water is 4 km/h and 
his speed against the stream is 4 km/h, 
his speed with the stream is also 4 km/h.
-/
theorem boat_speed_theorem (speed : BoatSpeed) 
  (h1 : speed.stillWater = 4)
  (h2 : speed.upstream = 4) :
  speed.downstream = 4 := by
  sorry

#check boat_speed_theorem

end NUMINAMATH_CALUDE_boat_speed_theorem_l3720_372051


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3720_372083

theorem polynomial_factorization (a : ℝ) : 
  (a^2 - 4*a + 2) * (a^2 - 4*a + 6) + 4 = (a - 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3720_372083


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l3720_372016

-- Define what an O'Hara triple is
def is_ohara_triple (c d y : ℕ) : Prop :=
  c > 0 ∧ d > 0 ∧ y > 0 ∧ (Real.sqrt c + Real.sqrt d = y)

-- State the theorem
theorem ohara_triple_49_16 :
  ∀ y : ℕ, is_ohara_triple 49 16 y → y = 11 :=
by sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l3720_372016


namespace NUMINAMATH_CALUDE_gcd_630_945_l3720_372057

theorem gcd_630_945 : Nat.gcd 630 945 = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_630_945_l3720_372057
