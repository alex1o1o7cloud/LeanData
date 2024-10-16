import Mathlib

namespace NUMINAMATH_CALUDE_banana_price_is_60_cents_l1921_192108

def apple_price : ℚ := 0.70
def total_cost : ℚ := 5.60
def total_fruits : ℕ := 9

theorem banana_price_is_60_cents :
  ∃ (num_apples num_bananas : ℕ) (banana_price : ℚ),
    num_apples + num_bananas = total_fruits ∧
    num_apples * apple_price + num_bananas * banana_price = total_cost ∧
    banana_price = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_banana_price_is_60_cents_l1921_192108


namespace NUMINAMATH_CALUDE_organization_growth_l1921_192188

/-- Represents the number of people in the organization after a given number of years. -/
def people_count (initial_total : ℕ) (leaders : ℕ) (years : ℕ) : ℕ :=
  leaders + (initial_total - leaders) * (3^years)

/-- Theorem stating the number of people in the organization after 5 years. -/
theorem organization_growth :
  people_count 15 5 5 = 2435 := by
  sorry

#eval people_count 15 5 5

end NUMINAMATH_CALUDE_organization_growth_l1921_192188


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1921_192152

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis -/
def symmetricPointXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: The symmetric point of P(-2, 3) with respect to x-axis is (-2, -3) -/
theorem symmetric_point_x_axis :
  let P : Point2D := { x := -2, y := 3 }
  symmetricPointXAxis P = { x := -2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1921_192152


namespace NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1921_192199

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_eight_factorial_l1921_192199


namespace NUMINAMATH_CALUDE_locus_of_tangent_points_l1921_192116

/-- Given a parabola y^2 = 2px and a constant k, prove that the locus of points P(x, y) 
    from which tangents can be drawn to the parabola with slopes m1 and m2 satisfying 
    m1 * m2^2 + m1^2 * m2 = k, is the parabola x^2 = (p / (2k)) * y -/
theorem locus_of_tangent_points (p k : ℝ) (hp : p > 0) (hk : k ≠ 0) :
  ∀ x y m1 m2 : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    y1^2 = 2 * p * x1 ∧ 
    y2^2 = 2 * p * x2 ∧
    m1 = p / y1 ∧
    m2 = p / y2 ∧
    2 * y = y1 + y2 ∧
    x^2 = x1 * x2 ∧
    m1 * m2^2 + m1^2 * m2 = k) →
  x^2 = (p / (2 * k)) * y := by
  sorry

end NUMINAMATH_CALUDE_locus_of_tangent_points_l1921_192116


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1921_192150

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ
  averageIncrease : ℚ

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.innings

/-- Theorem stating the batsman's average after the 17th inning -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.innings = 17)
  (h2 : b.lastInningRuns = 90)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with
    innings := b.innings - 1,
    totalRuns := b.totalRuns - b.lastInningRuns
  } + b.averageIncrease) :
  calculateAverage b = 42 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1921_192150


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1921_192137

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1921_192137


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1921_192164

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1921_192164


namespace NUMINAMATH_CALUDE_correct_seat_ratio_l1921_192144

/-- The ratio of coach class seats to first-class seats in an airplane -/
def seat_ratio (total_seats first_class_seats : ℕ) : ℚ × ℚ :=
  let coach_seats := total_seats - first_class_seats
  (coach_seats, first_class_seats)

/-- Theorem stating the correct ratio of coach to first-class seats -/
theorem correct_seat_ratio :
  seat_ratio 387 77 = (310, 77) := by
  sorry

#eval seat_ratio 387 77

end NUMINAMATH_CALUDE_correct_seat_ratio_l1921_192144


namespace NUMINAMATH_CALUDE_equivalent_discount_l1921_192151

/-- Proves that a single discount of 23.5% on $1200 results in the same final price
    as successive discounts of 15% and 10%. -/
theorem equivalent_discount (original_price : ℝ) (discount1 discount2 single_discount : ℝ) :
  original_price = 1200 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - single_discount) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l1921_192151


namespace NUMINAMATH_CALUDE_total_panels_eq_600_l1921_192181

/-- The number of houses in the neighborhood -/
def num_houses : ℕ := 10

/-- The number of double windows downstairs in each house -/
def num_double_windows : ℕ := 6

/-- The number of glass panels in each double window -/
def panels_per_double_window : ℕ := 4

/-- The number of single windows upstairs in each house -/
def num_single_windows : ℕ := 8

/-- The number of glass panels in each single window -/
def panels_per_single_window : ℕ := 3

/-- The number of bay windows in each house -/
def num_bay_windows : ℕ := 2

/-- The number of glass panels in each bay window -/
def panels_per_bay_window : ℕ := 6

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := num_houses * (
  num_double_windows * panels_per_double_window +
  num_single_windows * panels_per_single_window +
  num_bay_windows * panels_per_bay_window
)

theorem total_panels_eq_600 : total_panels = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_panels_eq_600_l1921_192181


namespace NUMINAMATH_CALUDE_smallest_multiple_l1921_192157

theorem smallest_multiple (x : ℕ+) : (∀ y : ℕ+, 720 * y.val % 1250 = 0 → x ≤ y) ∧ 720 * x.val % 1250 = 0 ↔ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1921_192157


namespace NUMINAMATH_CALUDE_dalmatian_spots_l1921_192179

theorem dalmatian_spots (bill_spots phil_spots : ℕ) : 
  bill_spots = 39 → 
  bill_spots = 2 * phil_spots - 1 → 
  bill_spots + phil_spots = 59 := by
sorry

end NUMINAMATH_CALUDE_dalmatian_spots_l1921_192179


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1921_192166

def C : Set Nat := {51, 53, 54, 55, 57}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ 
    (∀ (m : Nat), m ∈ C → 
      (∃ (p : Nat), Nat.Prime p ∧ p ∣ n) → 
      (∃ (q : Nat), Nat.Prime q ∧ q ∣ m ∧ p ≤ q)) ∧
    n = 54 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1921_192166


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1921_192174

theorem polynomial_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1921_192174


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1921_192105

theorem square_of_negative_product (b : ℝ) : (-3 * b)^2 = 9 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1921_192105


namespace NUMINAMATH_CALUDE_star_op_equation_has_two_distinct_real_roots_l1921_192111

/-- Custom operation ※ -/
def star_op (a b : ℝ) : ℝ := a^2 * b + a * b - 1

/-- Theorem stating that x※1 = 0 has two distinct real roots -/
theorem star_op_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star_op x 1 = 0 ∧ star_op y 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equation_has_two_distinct_real_roots_l1921_192111


namespace NUMINAMATH_CALUDE_sin_product_equals_cos_over_eight_l1921_192146

theorem sin_product_equals_cos_over_eight :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  Real.cos (10 * π / 180) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_cos_over_eight_l1921_192146


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l1921_192186

theorem smallest_seating_arrangement (M : ℕ+) : (∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y) → M ≥ 3 :=
by sorry

theorem three_satisfies_seating_arrangement : ∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y :=
by sorry

theorem smallest_M_is_three : (∀ M : ℕ+, M < 3 → ¬(∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y)) ∧
                              (∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l1921_192186


namespace NUMINAMATH_CALUDE_temperature_function_properties_l1921_192155

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- Define the theorem
theorem temperature_function_properties :
  -- Conditions
  (T (-4) = 8) ∧
  (T 0 = 60) ∧
  (T 1 = 58) ∧
  (deriv T (-4) = deriv T 4) ∧
  -- Conclusions
  (∀ t ∈ Set.Icc (-2) 2, T t ≤ 62) ∧
  (T (-1) = 62) ∧
  (T 2 = 62) :=
by sorry

end NUMINAMATH_CALUDE_temperature_function_properties_l1921_192155


namespace NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_two_l1921_192106

theorem product_of_fractions_and_powers_of_two : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * 
  (1 / 1024 : ℚ) * 2048 * (1 / 4096 : ℚ) * 8192 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_two_l1921_192106


namespace NUMINAMATH_CALUDE_polynomial_B_value_l1921_192180

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 144

def roots : List ℤ := [3, 3, 2, 2, 1, 1]

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r ∈ roots, polynomial r A B C D = 0) →
  B = -126 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l1921_192180


namespace NUMINAMATH_CALUDE_constant_discount_increase_l1921_192176

/-- Represents the discount percentage for a given number of pizzas -/
def discount (n : ℕ) : ℚ :=
  match n with
  | 1 => 0
  | 2 => 4/100
  | 3 => 8/100
  | _ => 0  -- Default case, not used in this problem

/-- The theorem states that the discount increase is constant -/
theorem constant_discount_increase :
  ∃ (r : ℚ), (discount 2 - discount 1 = r) ∧ (discount 3 - discount 2 = r) ∧ (r = 4/100) := by
  sorry

#check constant_discount_increase

end NUMINAMATH_CALUDE_constant_discount_increase_l1921_192176


namespace NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1921_192134

/-- Represents a triangle with consecutive natural number side lengths and an inscribed circle radius of 4 -/
structure SpecialTriangle where
  n : ℕ
  side_a : ℕ := n - 1
  side_b : ℕ := n
  side_c : ℕ := n + 1
  inradius : ℝ := 4

/-- The radius of the circumcircle of a SpecialTriangle is 65/8 -/
theorem circumradius_of_special_triangle (t : SpecialTriangle) : 
  (t.side_a : ℝ) * t.side_b * t.side_c / (4 * t.inradius * (t.side_a + t.side_b + t.side_c) / 2) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_of_special_triangle_l1921_192134


namespace NUMINAMATH_CALUDE_fraction_problem_l1921_192168

theorem fraction_problem (x : ℚ) : 
  (5 / 6 : ℚ) * 576 = x * 576 + 300 → x = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1921_192168


namespace NUMINAMATH_CALUDE_max_rectangles_equals_black_squares_l1921_192154

/-- Represents a figure that can be cut into squares and rectangles -/
structure Figure where
  shape : Set (ℕ × ℕ)  -- Set of coordinates representing the shape

/-- Counts the number of black squares when coloring the middle diagonal -/
def count_black_squares (f : Figure) : ℕ :=
  sorry

/-- Represents the specific figure given in the problem -/
def given_figure : Figure :=
  { shape := sorry }

/-- The maximum number of 1×2 rectangles that can be obtained -/
def max_rectangles (f : Figure) : ℕ :=
  sorry

theorem max_rectangles_equals_black_squares :
  max_rectangles given_figure = count_black_squares given_figure ∧
  count_black_squares given_figure = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_equals_black_squares_l1921_192154


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1921_192145

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  is_pure_imaginary z → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1921_192145


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1921_192162

/-- Given points A, B, and C in a 2D plane, with specific conditions on their coordinates and the lines connecting them, prove that the sum of certain coordinate values is 1. -/
theorem point_coordinate_sum (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 5)
  let B : ℝ × ℝ := (2, 2 - b)
  let C : ℝ × ℝ := (4, 2)
  (A.2 = B.2) →  -- AB is parallel to x-axis
  (A.1 = C.1) →  -- AC is parallel to y-axis
  a + b = 1 := by
sorry


end NUMINAMATH_CALUDE_point_coordinate_sum_l1921_192162


namespace NUMINAMATH_CALUDE_square_root_of_1024_l1921_192163

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l1921_192163


namespace NUMINAMATH_CALUDE_initial_birds_count_l1921_192196

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that joined the fence -/
def joined_birds : ℕ := 4

/-- The total number of birds on the fence after joining -/
def total_birds : ℕ := 5

/-- Theorem stating that the initial number of birds is 1 -/
theorem initial_birds_count : initial_birds = 1 :=
  by sorry

end NUMINAMATH_CALUDE_initial_birds_count_l1921_192196


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l1921_192178

/-- Given a set of observations with incorrect recordings, calculate the corrected mean. -/
theorem corrected_mean_calculation 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value1 incorrect_value2 correct_value1 correct_value2 : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value1 = 23 →
  incorrect_value2 = 55 →
  correct_value1 = 34 →
  correct_value2 = 45 →
  let original_sum := n * original_mean
  let adjusted_sum := original_sum - incorrect_value1 - incorrect_value2 + correct_value1 + correct_value2
  let new_mean := adjusted_sum / n
  new_mean = 36.02 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l1921_192178


namespace NUMINAMATH_CALUDE_octopus_gloves_bracelets_arrangements_l1921_192175

/-- The number of arms an octopus has -/
def num_arms : ℕ := 8

/-- The total number of items (gloves and bracelets) -/
def total_items : ℕ := 2 * num_arms

/-- The number of valid arrangements for putting on gloves and bracelets -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_arms)

/-- Theorem stating the correct number of valid arrangements -/
theorem octopus_gloves_bracelets_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_arms) :=
by sorry

end NUMINAMATH_CALUDE_octopus_gloves_bracelets_arrangements_l1921_192175


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l1921_192109

theorem sphere_radius_from_surface_area :
  ∀ (r : ℝ), (4 : ℝ) * Real.pi * r^2 = (4 : ℝ) * Real.pi → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l1921_192109


namespace NUMINAMATH_CALUDE_shoes_theorem_l1921_192153

/-- Given an initial number of shoe pairs and a number of lost individual shoes,
    calculate the maximum number of complete pairs remaining. -/
def maxRemainingPairs (initialPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  initialPairs - lostShoes

/-- Theorem: Given 26 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 17. -/
theorem shoes_theorem :
  maxRemainingPairs 26 9 = 17 := by
  sorry

#eval maxRemainingPairs 26 9

end NUMINAMATH_CALUDE_shoes_theorem_l1921_192153


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1921_192195

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1921_192195


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_form_l1921_192136

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number -/
def given_number : ℝ := 0.0000077

/-- The scientific notation representation of the given number -/
def scientific_form : ScientificNotation :=
  { coefficient := 7.7
    exponent := -6
    coeff_range := by sorry }

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_form :
  given_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_form_l1921_192136


namespace NUMINAMATH_CALUDE_age_ratio_l1921_192104

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions on the ages -/
def validAges (a : Ages) : Prop :=
  a.roy = a.julia + 8 ∧
  a.roy + 2 = 3 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 96

/-- The theorem to be proved -/
theorem age_ratio (a : Ages) (h : validAges a) :
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1921_192104


namespace NUMINAMATH_CALUDE_average_temperature_twthf_l1921_192147

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday is 46 degrees -/
theorem average_temperature_twthf (temp_mon : ℝ) (temp_fri : ℝ) (avg_mtwth : ℝ) :
  temp_mon = 43 →
  temp_fri = 35 →
  avg_mtwth = 48 →
  let temp_twth : ℝ := (4 * avg_mtwth - temp_mon) / 3
  let avg_twthf : ℝ := (3 * temp_twth + temp_fri) / 4
  ∀ ε > 0, |avg_twthf - 46| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_twthf_l1921_192147


namespace NUMINAMATH_CALUDE_purple_pairs_coincide_l1921_192143

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  yellow : ℕ
  green : ℕ
  purple : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  yellow_yellow : ℕ
  green_green : ℕ
  yellow_purple : ℕ
  purple_purple : ℕ

/-- The main theorem to prove -/
theorem purple_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.yellow = 4)
  (h2 : counts.green = 6)
  (h3 : counts.purple = 10)
  (h4 : pairs.yellow_yellow = 3)
  (h5 : pairs.green_green = 4)
  (h6 : pairs.yellow_purple = 3) :
  pairs.purple_purple = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_pairs_coincide_l1921_192143


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l1921_192184

theorem difference_c_minus_a (a b c d k : ℝ) : 
  (a + b) / 2 = 45 →
  (b + c) / 2 = 50 →
  (a + c + d) / 3 = 60 →
  a^2 + b^2 + c^2 + d^2 = k →
  c - a = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l1921_192184


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1921_192120

theorem basketball_team_selection (n : ℕ) (k : ℕ) (twins : ℕ) :
  n = 12 → k = 5 → twins = 2 →
  (Nat.choose n k) - (Nat.choose (n - twins) k) = 540 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1921_192120


namespace NUMINAMATH_CALUDE_discount_reduction_l1921_192130

/-- Proves that applying a 30% discount followed by a 20% discount
    results in a total reduction of 44% from the original price. -/
theorem discount_reduction (P : ℝ) (P_pos : P > 0) :
  let first_discount := 0.3
  let second_discount := 0.2
  let price_after_first := P * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_reduction := (P - price_after_second) / P
  total_reduction = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_discount_reduction_l1921_192130


namespace NUMINAMATH_CALUDE_difference_largest_smallest_n_l1921_192183

-- Define a convex n-gon
def ConvexNGon (n : ℕ) := n ≥ 3

-- Define an odd prime number
def OddPrime (p : ℕ) := Nat.Prime p ∧ p % 2 = 1

-- Define the condition that all interior angles are odd primes
def AllAnglesOddPrime (n : ℕ) (angles : Fin n → ℕ) :=
  ∀ i, OddPrime (angles i)

-- Define the sum of interior angles of an n-gon
def InteriorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the condition that the sum of angles equals the interior angle sum
def AnglesSumToInteriorSum (n : ℕ) (angles : Fin n → ℕ) :=
  (Finset.univ.sum angles) = InteriorAngleSum n

-- Main theorem
theorem difference_largest_smallest_n :
  ∃ (n_min n_max : ℕ),
    (ConvexNGon n_min ∧
     ∃ angles_min, AllAnglesOddPrime n_min angles_min ∧ AnglesSumToInteriorSum n_min angles_min) ∧
    (ConvexNGon n_max ∧
     ∃ angles_max, AllAnglesOddPrime n_max angles_max ∧ AnglesSumToInteriorSum n_max angles_max) ∧
    (∀ n, ConvexNGon n → 
      (∃ angles, AllAnglesOddPrime n angles ∧ AnglesSumToInteriorSum n angles) →
      n_min ≤ n ∧ n ≤ n_max) ∧
    n_max - n_min = 356 :=
sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_n_l1921_192183


namespace NUMINAMATH_CALUDE_car_distance_proof_l1921_192170

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 80 →
  (initial_time * 3 / 2) * speed = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1921_192170


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l1921_192140

-- Define the function f(x) = x³ - 2x² + 3x + 1
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 3

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l1921_192140


namespace NUMINAMATH_CALUDE_cheese_weight_l1921_192113

/-- Represents the weight of two pieces of cheese -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- The function that represents taking a bite from the larger piece -/
def take_bite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- The theorem stating the original weight of the cheese -/
theorem cheese_weight (initial : CheesePair) :
  (take_bite (take_bite (take_bite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
sorry

end NUMINAMATH_CALUDE_cheese_weight_l1921_192113


namespace NUMINAMATH_CALUDE_range_of_a_l1921_192107

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x^2 + (a+2)*x + 1) * ((3-2*a)*x^2 + 5*x + (3-2*a)) ≥ 0) →
  a ∈ Set.Icc (-4 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1921_192107


namespace NUMINAMATH_CALUDE_common_chord_length_is_2sqrt5_l1921_192193

/-- Circle C1 with equation x^2 + y^2 + 2x + 8y - 8 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C2 with equation x^2 + y^2 - 4x - 4y - 2 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

/-- The circles C1 and C2 intersect -/
axiom circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y

/-- The length of the common chord of two intersecting circles -/
def common_chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the common chord of C1 and C2 is 2√5 -/
theorem common_chord_length_is_2sqrt5 :
  common_chord_length C1 C2 = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2sqrt5_l1921_192193


namespace NUMINAMATH_CALUDE_common_tangents_theorem_l1921_192190

/-- Represents the relative position of two circles -/
inductive CirclePosition
  | Outside
  | TouchingExternally
  | Intersecting
  | TouchingInternally
  | Inside
  | Identical
  | OnePoint
  | TwoDistinctPoints
  | TwoCoincidingPoints

/-- Represents the number of common tangents -/
inductive TangentCount
  | Zero
  | One
  | Two
  | Three
  | Four
  | Infinite

/-- Function to determine the number of common tangents based on circle position -/
def commonTangents (position : CirclePosition) : TangentCount :=
  match position with
  | CirclePosition.Outside => TangentCount.Four
  | CirclePosition.TouchingExternally => TangentCount.Three
  | CirclePosition.Intersecting => TangentCount.Two
  | CirclePosition.TouchingInternally => TangentCount.One
  | CirclePosition.Inside => TangentCount.Zero
  | CirclePosition.Identical => TangentCount.Infinite
  | CirclePosition.OnePoint => TangentCount.Two  -- Assuming the point is outside the circle
  | CirclePosition.TwoDistinctPoints => TangentCount.One
  | CirclePosition.TwoCoincidingPoints => TangentCount.Infinite

/-- Theorem stating that the number of common tangents depends on the relative position of circles -/
theorem common_tangents_theorem (position : CirclePosition) :
  (commonTangents position = TangentCount.Zero) ∨
  (commonTangents position = TangentCount.One) ∨
  (commonTangents position = TangentCount.Two) ∨
  (commonTangents position = TangentCount.Three) ∨
  (commonTangents position = TangentCount.Four) ∨
  (commonTangents position = TangentCount.Infinite) :=
by sorry

end NUMINAMATH_CALUDE_common_tangents_theorem_l1921_192190


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1921_192160

/-- The distance between the foci of a hyperbola with equation x^2/32 - y^2/8 = 1 is 4√10 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ),
  x^2 / 32 - y^2 / 8 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ),
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (4 * Real.sqrt 10)^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1921_192160


namespace NUMINAMATH_CALUDE_sqrt_ceil_floor_sum_l1921_192122

theorem sqrt_ceil_floor_sum : 
  ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌊Real.sqrt 350⌋ = 39 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ceil_floor_sum_l1921_192122


namespace NUMINAMATH_CALUDE_intersections_divisible_by_three_l1921_192129

/-- The number of intersections between segments connecting points on parallel lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n + 1) * n / 4

/-- Theorem stating that the number of intersections is divisible by 3 -/
theorem intersections_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, num_intersections n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_intersections_divisible_by_three_l1921_192129


namespace NUMINAMATH_CALUDE_all_naturals_reachable_l1921_192159

def triple_plus_one (x : ℕ) : ℕ := 3 * x + 1

def floor_half (x : ℕ) : ℕ := x / 2

def reachable (n : ℕ) : Prop :=
  ∃ (seq : List (ℕ → ℕ)), seq.foldl (λ acc f => f acc) 1 = n ∧
    ∀ f ∈ seq, f = triple_plus_one ∨ f = floor_half

theorem all_naturals_reachable : ∀ n : ℕ, reachable n := by
  sorry

end NUMINAMATH_CALUDE_all_naturals_reachable_l1921_192159


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l1921_192132

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 2|

-- Theorem 1: Solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem 2: Range of a for non-empty solution set
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_for_nonempty_solution_l1921_192132


namespace NUMINAMATH_CALUDE_coefficient_of_3x_squared_l1921_192110

/-- Definition of a coefficient in a monomial term -/
def coefficient (term : ℝ → ℝ) : ℝ :=
  term 1

/-- The term 3x^2 -/
def term (x : ℝ) : ℝ := 3 * x^2

/-- Theorem: The coefficient of 3x^2 is 3 -/
theorem coefficient_of_3x_squared :
  coefficient term = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_3x_squared_l1921_192110


namespace NUMINAMATH_CALUDE_divide_100_by_0_25_l1921_192148

theorem divide_100_by_0_25 : (100 : ℝ) / 0.25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_divide_100_by_0_25_l1921_192148


namespace NUMINAMATH_CALUDE_jellybeans_in_larger_box_l1921_192103

/-- Given a box with jellybeans and another box with tripled dimensions, 
    calculate the number of jellybeans in the larger box. -/
theorem jellybeans_in_larger_box 
  (small_box_jellybeans : ℕ) 
  (scale_factor : ℕ) 
  (h1 : small_box_jellybeans = 150) 
  (h2 : scale_factor = 3) : 
  (scale_factor ^ 3 : ℕ) * small_box_jellybeans = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_in_larger_box_l1921_192103


namespace NUMINAMATH_CALUDE_book_set_cost_l1921_192191

/-- The cost of a book set given lawn mowing parameters -/
theorem book_set_cost 
  (charge_rate : ℚ)
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (lawns_mowed : ℕ)
  (additional_area : ℕ)
  (h1 : charge_rate = 1 / 10)
  (h2 : lawn_length = 20)
  (h3 : lawn_width = 15)
  (h4 : lawns_mowed = 3)
  (h5 : additional_area = 600) :
  (lawn_length * lawn_width * lawns_mowed + additional_area) * charge_rate = 150 := by
  sorry

#check book_set_cost

end NUMINAMATH_CALUDE_book_set_cost_l1921_192191


namespace NUMINAMATH_CALUDE_coefficient_x6y4_in_expansion_l1921_192125

theorem coefficient_x6y4_in_expansion : ∀ x y : ℝ,
  (Nat.choose 10 4 : ℝ) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6y4_in_expansion_l1921_192125


namespace NUMINAMATH_CALUDE_exact_one_root_at_most_one_root_l1921_192197

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 + 2*x + 1 = 0

-- Define the set of roots
def root_set (a : ℝ) : Set ℝ := {x | quadratic_equation a x}

-- Statement 1: A contains exactly one element iff a = 1 or a = 0
theorem exact_one_root (a : ℝ) : 
  (∃! x, x ∈ root_set a) ↔ (a = 1 ∨ a = 0) :=
sorry

-- Statement 2: A contains at most one element iff a ∈ {0} ∪ [1, +∞)
theorem at_most_one_root (a : ℝ) :
  (∀ x y, x ∈ root_set a → y ∈ root_set a → x = y) ↔ (a = 0 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_exact_one_root_at_most_one_root_l1921_192197


namespace NUMINAMATH_CALUDE_playground_boys_count_l1921_192189

theorem playground_boys_count (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 63 → girls = 28 → boys = total - girls → boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_playground_boys_count_l1921_192189


namespace NUMINAMATH_CALUDE_average_weight_abc_l1921_192165

/-- Given the average weight of a and b is 40 kg, the average weight of b and c is 44 kg,
    and the weight of b is 33 kg, prove that the average weight of a, b, and c is 45 kg. -/
theorem average_weight_abc (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 44)
  (h3 : b = 33) :
  (a + b + c) / 3 = 45 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_abc_l1921_192165


namespace NUMINAMATH_CALUDE_pyramid_volume_l1921_192198

/-- Represents a pyramid with a triangular base --/
structure TriangularPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_side3 : ℝ
  lateral_angle : ℝ

/-- Calculates the volume of a triangular pyramid --/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that a pyramid with the given properties has a volume of 6 --/
theorem pyramid_volume :
  ∀ (p : TriangularPyramid),
    p.base_side1 = 6 ∧
    p.base_side2 = 5 ∧
    p.base_side3 = 5 ∧
    p.lateral_angle = π / 4 →
    volume p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1921_192198


namespace NUMINAMATH_CALUDE_exponent_addition_l1921_192102

theorem exponent_addition (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l1921_192102


namespace NUMINAMATH_CALUDE_inequality_proof_l1921_192169

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b = 3 + b - a) : (3 / b) + (1 / a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1921_192169


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_10080_l1921_192100

/-- Given that 10080 = 2^4 * 3^2 * 5 * 7, this function counts the number of positive integer factors of 10080 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1), (7, 1)]
  -- Function implementation
  sorry

/-- The number of positive integer factors of 10080 that are perfect squares is 6. -/
theorem perfect_square_factors_of_10080 : count_perfect_square_factors = 6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_10080_l1921_192100


namespace NUMINAMATH_CALUDE_rectangle_side_sum_l1921_192139

theorem rectangle_side_sum (x y : ℝ) : 
  (2 * x + 4 = 10) → (8 * y - 2 = 10) → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_sum_l1921_192139


namespace NUMINAMATH_CALUDE_f_geq_1_solution_set_g_max_value_l1921_192112

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_1_solution_set (x : ℝ) :
  f x ≥ 1 ↔ x ≥ 1 := by sorry

theorem g_max_value :
  ∃ x₀ : ℝ, ∀ x : ℝ, g x ≤ g x₀ ∧ g x₀ = 5/4 := by sorry

end NUMINAMATH_CALUDE_f_geq_1_solution_set_g_max_value_l1921_192112


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l1921_192133

/-- The number of digits to the right of the decimal point when a fraction is expressed as a decimal. -/
def decimal_digits (n : ℚ) : ℕ := sorry

/-- The fraction we're considering -/
def fraction : ℚ := 3^6 / (6^4 * 625)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of our fraction is 4 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l1921_192133


namespace NUMINAMATH_CALUDE_valid_sequences_count_l1921_192187

/-- Represents the colors of the houses -/
inductive Color
  | Orange
  | Red
  | Blue
  | Yellow
  | Green
  | Purple

/-- A sequence of colored houses -/
def HouseSequence := List Color

/-- Checks if a color appears before another in a sequence -/
def appearsBefore (c1 c2 : Color) (seq : HouseSequence) : Prop :=
  ∃ i j, i < j ∧ seq.getD i c1 = c1 ∧ seq.getD j c2 = c2

/-- Checks if two colors are adjacent in a sequence -/
def areAdjacent (c1 c2 : Color) (seq : HouseSequence) : Prop :=
  ∃ i, (seq.getD i c1 = c1 ∧ seq.getD (i+1) c2 = c2) ∨ 
       (seq.getD i c2 = c2 ∧ seq.getD (i+1) c1 = c1)

/-- Checks if a sequence is valid according to the given conditions -/
def isValidSequence (seq : HouseSequence) : Prop :=
  seq.length = 6 ∧ 
  seq.Nodup ∧
  appearsBefore Color.Orange Color.Red seq ∧
  appearsBefore Color.Blue Color.Yellow seq ∧
  areAdjacent Color.Red Color.Green seq ∧
  ¬(areAdjacent Color.Blue Color.Yellow seq) ∧
  ¬(areAdjacent Color.Blue Color.Red seq)

/-- The main theorem to be proved -/
theorem valid_sequences_count :
  ∃! (validSeqs : List HouseSequence), 
    (∀ seq, seq ∈ validSeqs ↔ isValidSequence seq) ∧ 
    validSeqs.length = 3 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l1921_192187


namespace NUMINAMATH_CALUDE_disease_test_probability_l1921_192158

theorem disease_test_probability (p_disease : ℝ) (p_positive_given_disease : ℝ) (p_positive_given_no_disease : ℝ) :
  p_disease = 1 / 300 →
  p_positive_given_disease = 1 →
  p_positive_given_no_disease = 0.03 →
  (p_disease * p_positive_given_disease) / 
  (p_disease * p_positive_given_disease + (1 - p_disease) * p_positive_given_no_disease) = 100 / 997 := by
  sorry

end NUMINAMATH_CALUDE_disease_test_probability_l1921_192158


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1921_192142

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧ 
  white = 22 ∧ 
  green = 18 ∧ 
  red = 5 ∧ 
  purple = 7 ∧ 
  prob = 4/5 ∧ 
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 8 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1921_192142


namespace NUMINAMATH_CALUDE_phd_total_time_l1921_192131

def phd_timeline (acclimation_time : ℝ) (basics_time : ℝ) (research_factor : ℝ) (dissertation_factor : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_factor)
  let dissertation_time := acclimation_time * dissertation_factor
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_total_time :
  phd_timeline 1 2 0.75 0.5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_phd_total_time_l1921_192131


namespace NUMINAMATH_CALUDE_rectangle_existence_uniqueness_l1921_192123

theorem rectangle_existence_uniqueness 
  (a b : ℝ) 
  (h_ab : 0 < a ∧ a < b) : 
  ∃! (x y : ℝ), 
    x < a ∧ 
    y < b ∧ 
    2 * (x + y) = a + b ∧ 
    x * y = a * b / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_uniqueness_l1921_192123


namespace NUMINAMATH_CALUDE_combined_price_increase_percentage_l1921_192182

def skateboard_initial_price : ℝ := 120
def knee_pads_initial_price : ℝ := 30
def skateboard_increase_percent : ℝ := 8
def knee_pads_increase_percent : ℝ := 15

theorem combined_price_increase_percentage :
  let skateboard_new_price := skateboard_initial_price * (1 + skateboard_increase_percent / 100)
  let knee_pads_new_price := knee_pads_initial_price * (1 + knee_pads_increase_percent / 100)
  let initial_total := skateboard_initial_price + knee_pads_initial_price
  let new_total := skateboard_new_price + knee_pads_new_price
  (new_total - initial_total) / initial_total * 100 = 9.4 := by sorry

end NUMINAMATH_CALUDE_combined_price_increase_percentage_l1921_192182


namespace NUMINAMATH_CALUDE_fourDigitPermutationsFromSixIs360_l1921_192117

/-- The number of permutations of 4 digits chosen from a set of 6 digits -/
def fourDigitPermutationsFromSix : ℕ :=
  6 * 5 * 4 * 3

/-- Theorem stating that the number of four-digit numbers without repeating digits
    from the set {1, 2, 3, 4, 5, 6} is equal to 360 -/
theorem fourDigitPermutationsFromSixIs360 : fourDigitPermutationsFromSix = 360 := by
  sorry


end NUMINAMATH_CALUDE_fourDigitPermutationsFromSixIs360_l1921_192117


namespace NUMINAMATH_CALUDE_mother_daughter_ages_l1921_192115

/-- Given a mother and daughter where:
    1. The mother is 27 years older than her daughter.
    2. A year ago, the mother was twice as old as her daughter.
    Prove that the mother is 55 years old and the daughter is 28 years old. -/
theorem mother_daughter_ages (mother_age daughter_age : ℕ) 
  (h1 : mother_age = daughter_age + 27)
  (h2 : mother_age - 1 = 2 * (daughter_age - 1)) :
  mother_age = 55 ∧ daughter_age = 28 := by
sorry

end NUMINAMATH_CALUDE_mother_daughter_ages_l1921_192115


namespace NUMINAMATH_CALUDE_canoe_production_sum_l1921_192149

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

def sum_geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^n - 1)) / (r - 1)

theorem canoe_production_sum :
  let a := 10
  let r := 3
  let n := 4
  sum_geometric_sequence a r n = 400 := by
sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l1921_192149


namespace NUMINAMATH_CALUDE_semicircle_radius_l1921_192173

/-- The radius of a semi-circle with perimeter 198 cm is 198 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 198) : 
  perimeter / (Real.pi + 2) = 198 / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l1921_192173


namespace NUMINAMATH_CALUDE_problem_statement_l1921_192114

theorem problem_statement (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) : 
  (1/3) * x^8 * y^9 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1921_192114


namespace NUMINAMATH_CALUDE_retail_price_l1921_192118

/-- The retail price of a product given its cost price and percentage increase -/
theorem retail_price (a : ℝ) (percent_increase : ℝ) (h : percent_increase = 30) :
  a + (percent_increase / 100) * a = 1.3 * a := by
  sorry

end NUMINAMATH_CALUDE_retail_price_l1921_192118


namespace NUMINAMATH_CALUDE_volume_integrals_l1921_192172

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x^3)

theorem volume_integrals (π : ℝ) (h₁ : π > 0) :
  (∫ (x : ℝ) in Set.Ioi 0, π * (f x)^2) = π / 6 ∧
  (∫ (x : ℝ) in Set.Icc 0 (Real.rpow 3 (1/3)), π * x^2 * (1 - 3*x^3) * Real.exp (-x^3)) = π * (Real.exp (-1/3) - 2/3) :=
sorry

end NUMINAMATH_CALUDE_volume_integrals_l1921_192172


namespace NUMINAMATH_CALUDE_alice_painted_cuboids_l1921_192121

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 36

theorem alice_painted_cuboids :
  num_cuboids * faces_per_cuboid = total_faces_painted :=
by sorry

end NUMINAMATH_CALUDE_alice_painted_cuboids_l1921_192121


namespace NUMINAMATH_CALUDE_onesDigitOfComplexExpression_l1921_192127

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The given complex expression -/
def complexExpression : ℕ :=
  onesDigit ((73 ^ 1253) * (44 ^ 987) + (47 ^ 123) / (39 ^ 654) * (86 ^ 1484) - (32 ^ 1987) % 10)

/-- Theorem stating that the ones digit of the complex expression is 2 -/
theorem onesDigitOfComplexExpression : complexExpression = 2 := by
  sorry

end NUMINAMATH_CALUDE_onesDigitOfComplexExpression_l1921_192127


namespace NUMINAMATH_CALUDE_west_notation_l1921_192119

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function that represents the notation for walking in a given direction
def walkNotation (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_notation (d : ℝ) :
  walkNotation Direction.East d = d →
  walkNotation Direction.West d = -d :=
by sorry

end NUMINAMATH_CALUDE_west_notation_l1921_192119


namespace NUMINAMATH_CALUDE_rotation_of_A_to_B_l1921_192101

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_of_A_to_B :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let B : ℝ × ℝ := rotate90CCW A.1 A.2
  B = (-1, Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_rotation_of_A_to_B_l1921_192101


namespace NUMINAMATH_CALUDE_rsa_factorization_l1921_192167

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    400000001 = p * q ∧ 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p = 20201 ∧ 
    q = 19801 := by
  sorry

end NUMINAMATH_CALUDE_rsa_factorization_l1921_192167


namespace NUMINAMATH_CALUDE_positive_number_square_plus_twice_l1921_192171

theorem positive_number_square_plus_twice : ∃ n : ℝ, n > 0 ∧ n^2 + 2*n = 210 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_plus_twice_l1921_192171


namespace NUMINAMATH_CALUDE_unique_polynomial_with_given_value_l1921_192128

/-- A polynomial with natural number coefficients less than 10 -/
def PolynomialWithSmallCoeffs (p : Polynomial ℕ) : Prop :=
  ∀ i, (p.coeff i) < 10

theorem unique_polynomial_with_given_value :
  ∀ p : Polynomial ℕ,
  PolynomialWithSmallCoeffs p →
  p.eval 10 = 1248 →
  p = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 4 + Polynomial.monomial 0 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_polynomial_with_given_value_l1921_192128


namespace NUMINAMATH_CALUDE_prob_no_match_three_picks_correct_l1921_192161

/-- The probability of not having a matching pair after 3 picks from 3 pairs of socks -/
def prob_no_match_three_picks : ℚ := 2 / 5

/-- The number of pairs of socks -/
def num_pairs : ℕ := 3

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The probability of picking a non-matching sock on the second draw -/
def prob_second_draw : ℚ := 4 / 5

/-- The probability of picking a non-matching sock on the third draw -/
def prob_third_draw : ℚ := 1 / 2

theorem prob_no_match_three_picks_correct :
  prob_no_match_three_picks = prob_second_draw * prob_third_draw :=
sorry

end NUMINAMATH_CALUDE_prob_no_match_three_picks_correct_l1921_192161


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1921_192141

/-- Given sets A and B, where A ⊆ B, prove that a can only be 0, 1, or 1/2 -/
theorem possible_values_of_a (a : ℝ) :
  let A := {x : ℝ | a * x - 1 = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  A ⊆ B → (a = 0 ∨ a = 1 ∨ a = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_possible_values_of_a_l1921_192141


namespace NUMINAMATH_CALUDE_carols_spending_contradiction_l1921_192156

theorem carols_spending_contradiction (savings : ℝ) (tv_fraction : ℝ) 
  (h1 : savings > 0)
  (h2 : 0 < tv_fraction)
  (h3 : tv_fraction < 1/4)
  (h4 : 1/4 * savings + tv_fraction * savings = 0.25 * savings) : False :=
sorry

end NUMINAMATH_CALUDE_carols_spending_contradiction_l1921_192156


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l1921_192194

-- Define the equation of motion
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l1921_192194


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1921_192138

/-- Given a cubic polynomial x³ - 2x² + 5x - 8 with roots p, q, r,
    and another cubic polynomial x³ + ux² + vx + w with roots p+q, q+r, r+p,
    prove that w = 34 -/
theorem cubic_roots_problem (p q r u v w : ℝ) : 
  (∀ x, x^3 - 2*x^2 + 5*x - 8 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ x, x^3 + u*x^2 + v*x + w = 0 ↔ x = p+q ∨ x = q+r ∨ x = r+p) →
  w = 34 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1921_192138


namespace NUMINAMATH_CALUDE_tourist_travel_speeds_l1921_192192

theorem tourist_travel_speeds (total_distance : ℝ) (car_fraction : ℚ) (speed_difference : ℝ) (time_difference : ℝ) :
  total_distance = 160 ∧
  car_fraction = 5/8 ∧
  speed_difference = 20 ∧
  time_difference = 1/4 →
  (∃ (car_speed boat_speed : ℝ),
    (car_speed = 80 ∧ boat_speed = 60) ∨
    (car_speed = 100 ∧ boat_speed = 80)) ∧
    (car_speed - boat_speed = speed_difference) ∧
    (total_distance * car_fraction / car_speed = 
     total_distance * (1 - car_fraction) / boat_speed + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_tourist_travel_speeds_l1921_192192


namespace NUMINAMATH_CALUDE_compare_fractions_l1921_192135

theorem compare_fractions (a : ℝ) :
  (a = 0 → 1 / (1 - a) = 1 + a) ∧
  (0 < a ∧ a < 1 → 1 / (1 - a) > 1 + a) ∧
  (a > 1 → 1 / (1 - a) < 1 + a) := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l1921_192135


namespace NUMINAMATH_CALUDE_james_travel_distance_l1921_192126

/-- Calculates the total distance traveled during a road trip with multiple legs -/
def total_distance (speeds : List ℝ) (durations : List ℝ) : ℝ :=
  (List.zip speeds durations).map (fun (s, t) => s * t) |>.sum

/-- Theorem: James' total travel distance is 995.0 miles -/
theorem james_travel_distance : 
  let speeds : List ℝ := [80.0, 65.0, 75.0, 70.0]
  let durations : List ℝ := [2.0, 4.0, 3.0, 5.0]
  total_distance speeds durations = 995.0 := by
  sorry


end NUMINAMATH_CALUDE_james_travel_distance_l1921_192126


namespace NUMINAMATH_CALUDE_catherine_stationery_l1921_192185

theorem catherine_stationery (initial_pens initial_pencils pens_given pencils_given remaining_pens remaining_pencils : ℕ) :
  initial_pens = initial_pencils →
  pens_given = 36 →
  pencils_given = 16 →
  remaining_pens = 36 →
  remaining_pencils = 28 →
  initial_pens - pens_given = remaining_pens →
  initial_pencils - pencils_given = remaining_pencils →
  initial_pens = 72 ∧ initial_pencils = 72 := by
sorry

end NUMINAMATH_CALUDE_catherine_stationery_l1921_192185


namespace NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_squares_l1921_192124

/-- In a right triangle with sides a and b, hypotenuse c, and altitude x drawn on the hypotenuse,
    the following equation holds: 1/x² = 1/a² + 1/b² -/
theorem right_triangle_altitude_reciprocal_squares 
  (a b c x : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_altitude : a * b = c * x) : 
  1 / x^2 = 1 / a^2 + 1 / b^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_squares_l1921_192124


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1921_192177

theorem units_digit_of_7_power_75_plus_6 : 
  (7^75 + 6) % 10 = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_75_plus_6_l1921_192177
