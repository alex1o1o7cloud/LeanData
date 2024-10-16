import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_m_n_l3249_324993

theorem existence_of_m_n (p : Nat) (hp : p.Prime) (hp10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3249_324993


namespace NUMINAMATH_CALUDE_article_cost_l3249_324980

/-- Represents the cost of an article in rupees -/
def cost : ℝ := 60

/-- Represents the selling price of the article in rupees -/
def selling_price : ℝ := cost * 1.25

/-- Represents the new cost if bought at 20% less -/
def new_cost : ℝ := cost * 0.8

/-- Represents the new selling price if sold for Rs. 12.60 less -/
def new_selling_price : ℝ := selling_price - 12.60

theorem article_cost :
  (selling_price = cost * 1.25) ∧
  (new_selling_price = new_cost * 1.3) ∧
  (cost = 60) := by sorry

end NUMINAMATH_CALUDE_article_cost_l3249_324980


namespace NUMINAMATH_CALUDE_weekend_reading_l3249_324986

/-- The number of pages Bekah needs to read for history class -/
def total_pages : ℕ := 408

/-- The number of days left to finish reading -/
def days_left : ℕ := 5

/-- The number of pages Bekah needs to read each day for the remaining days -/
def pages_per_day : ℕ := 59

/-- The number of pages Bekah read over the weekend -/
def pages_read_weekend : ℕ := total_pages - (days_left * pages_per_day)

theorem weekend_reading :
  pages_read_weekend = 113 := by sorry

end NUMINAMATH_CALUDE_weekend_reading_l3249_324986


namespace NUMINAMATH_CALUDE_big_bottles_sold_percentage_l3249_324945

/-- Proves that the percentage of big bottles sold is 14% given the initial quantities,
    the percentage of small bottles sold, and the total remaining bottles. -/
theorem big_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (small_sold_percentage : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 6000)
  (h2 : initial_big = 15000)
  (h3 : small_sold_percentage = 12 / 100)
  (h4 : total_remaining = 18180)
  : (initial_big - (total_remaining - (initial_small - small_sold_percentage * initial_small))) / initial_big = 14 / 100 :=
by sorry

end NUMINAMATH_CALUDE_big_bottles_sold_percentage_l3249_324945


namespace NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3249_324995

/-- Two circles of radius r are externally tangent to each other and internally tangent to the ellipse x² + 4y² = 5. -/
theorem circle_radius_in_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3249_324995


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3249_324972

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ) ^ x = 12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3249_324972


namespace NUMINAMATH_CALUDE_sequence_sum_l3249_324990

theorem sequence_sum (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0) 
  (h1 : a 1 = 2) (h2 : a 2 = 3) (h3 : a 3 = 4) (h5 : a 5 = 6) :
  ∃ (a_val t : ℝ), a_val > 0 ∧ t > 0 ∧ a_val = a 5 ∧ t = a_val^2 - 1 ∧ a_val + t = 41 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3249_324990


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3249_324913

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - x = 0}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3249_324913


namespace NUMINAMATH_CALUDE_negative_greater_than_reciprocal_is_proper_fraction_l3249_324921

theorem negative_greater_than_reciprocal_is_proper_fraction (a : ℝ) :
  a < 0 ∧ a > 1 / a → -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_negative_greater_than_reciprocal_is_proper_fraction_l3249_324921


namespace NUMINAMATH_CALUDE_toms_cat_surgery_savings_l3249_324970

/-- Calculates the savings made by having insurance for a pet's surgery --/
def calculate_insurance_savings (
  insurance_duration : ℕ
  ) (insurance_monthly_cost : ℝ
  ) (procedure_cost : ℝ
  ) (insurance_coverage_percentage : ℝ
  ) : ℝ :=
  let total_insurance_cost := insurance_duration * insurance_monthly_cost
  let out_of_pocket_cost := procedure_cost * (1 - insurance_coverage_percentage)
  let total_cost_with_insurance := out_of_pocket_cost + total_insurance_cost
  procedure_cost - total_cost_with_insurance

/-- Theorem stating that the savings made by having insurance for Tom's cat surgery is $3520 --/
theorem toms_cat_surgery_savings :
  calculate_insurance_savings 24 20 5000 0.8 = 3520 := by
  sorry

end NUMINAMATH_CALUDE_toms_cat_surgery_savings_l3249_324970


namespace NUMINAMATH_CALUDE_alternating_digit_sum_2017_l3249_324962

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Alternating sum of digit sums from 1 to n -/
def alternating_digit_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun i => (-1)^i.succ * (digit_sum (i + 1) : ℤ))

/-- The alternating sum of digit sums for integers from 1 to 2017 is equal to 1009 -/
theorem alternating_digit_sum_2017 : alternating_digit_sum 2017 = 1009 := by sorry

end NUMINAMATH_CALUDE_alternating_digit_sum_2017_l3249_324962


namespace NUMINAMATH_CALUDE_pond_water_after_50_days_l3249_324942

/-- Calculates the remaining water in a pond after a given number of days, considering evaporation. -/
def remaining_water (initial_water : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_water - evaporation_rate * days

/-- Theorem stating that a pond with 500 gallons of water, losing 1 gallon per day, will have 450 gallons after 50 days. -/
theorem pond_water_after_50_days :
  remaining_water 500 1 50 = 450 := by
  sorry

#eval remaining_water 500 1 50

end NUMINAMATH_CALUDE_pond_water_after_50_days_l3249_324942


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3249_324977

-- Define the number in billions
def number_in_billions : ℝ := 8.36

-- Define the scientific notation
def scientific_notation : ℝ := 8.36 * (10 ^ 9)

-- Theorem statement
theorem billion_to_scientific_notation :
  (number_in_billions * 10^9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3249_324977


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l3249_324924

/-- Theorem: Rectangle Width Decrease
Given a rectangle where:
- The length increases by 20%
- The area increases by 4%
Then the width must decrease by 40/3% (approximately 13.33%) -/
theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.2 * L) (h2 : L' * W' = 1.04 * L * W) :
  W' = (1 - 40 / 300) * W :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l3249_324924


namespace NUMINAMATH_CALUDE_unique_pair_cube_prime_l3249_324901

theorem unique_pair_cube_prime : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 
  ∃ (p : ℕ), Prime p ∧ (x * y^3) / (x + y) = p^3 ∧ 
  x = 2 ∧ y = 14 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_cube_prime_l3249_324901


namespace NUMINAMATH_CALUDE_pascal_triangle_24th_row_20th_number_l3249_324933

theorem pascal_triangle_24th_row_20th_number : 
  (Nat.choose 24 19) = 42504 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_24th_row_20th_number_l3249_324933


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3249_324959

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  (a 1 + a 3 = 8) ∧
  (a 5 + a 7 = 4)

/-- The sum of specific terms in the geometric sequence equals 3 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3249_324959


namespace NUMINAMATH_CALUDE_horner_first_coefficient_l3249_324909

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

-- Define Horner's method for a 5th degree polynomial
def horner_method (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem horner_first_coefficient (x : ℝ) :
  ∃ (a₁ : ℝ), horner_method 0.5 4 0 (-3) a₁ (-1) x = f x ∧ a₁ = 1 :=
sorry

end NUMINAMATH_CALUDE_horner_first_coefficient_l3249_324909


namespace NUMINAMATH_CALUDE_cookie_pie_ratio_is_seven_fourths_l3249_324908

/-- The ratio of students preferring cookies to those preferring pie -/
def cookie_pie_ratio (total_students : ℕ) (cookie_preference : ℕ) (pie_preference : ℕ) : ℚ :=
  cookie_preference / pie_preference

theorem cookie_pie_ratio_is_seven_fourths :
  cookie_pie_ratio 800 280 160 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_pie_ratio_is_seven_fourths_l3249_324908


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3249_324979

theorem tic_tac_toe_tie_probability (ben_win_prob tom_win_prob tie_prob : ℚ) : 
  ben_win_prob = 1/4 → tom_win_prob = 2/5 → tie_prob = 1 - (ben_win_prob + tom_win_prob) → 
  tie_prob = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3249_324979


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3249_324920

theorem base_conversion_problem : ∃ (n A B : ℕ), 
  n > 0 ∧
  n = 8 * A + B ∧
  n = 6 * B + A ∧
  A < 8 ∧
  B < 6 ∧
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3249_324920


namespace NUMINAMATH_CALUDE_card_arrangement_probability_l3249_324931

theorem card_arrangement_probability : 
  let total_arrangements : ℕ := 24
  let favorable_arrangements : ℕ := 2
  let probability : ℚ := favorable_arrangements / total_arrangements
  probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_card_arrangement_probability_l3249_324931


namespace NUMINAMATH_CALUDE_resulting_temperature_correct_l3249_324943

/-- The resulting temperature when rising from 5°C to t°C -/
def resulting_temperature (t : ℝ) : ℝ := 5 + t

/-- Theorem stating that the resulting temperature is correct -/
theorem resulting_temperature_correct (t : ℝ) : 
  resulting_temperature t = 5 + t := by sorry

end NUMINAMATH_CALUDE_resulting_temperature_correct_l3249_324943


namespace NUMINAMATH_CALUDE_sum_of_abs_sum_and_diff_lt_two_l3249_324982

theorem sum_of_abs_sum_and_diff_lt_two (a b : ℝ) : 
  (|a| < 1) → (|b| < 1) → (|a + b| + |a - b| < 2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_sum_and_diff_lt_two_l3249_324982


namespace NUMINAMATH_CALUDE_lindas_coins_value_l3249_324946

theorem lindas_coins_value :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 150 →
  5 * n + 10 * d + 25 * q = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_value_l3249_324946


namespace NUMINAMATH_CALUDE_cost_of_600_pages_l3249_324989

-- Define the cost per 5 pages in cents
def cost_per_5_pages : ℕ := 10

-- Define the number of pages to be copied
def pages_to_copy : ℕ := 600

-- Theorem to prove the cost of copying 600 pages
theorem cost_of_600_pages : 
  (pages_to_copy / 5) * cost_per_5_pages = 1200 :=
by
  sorry

#check cost_of_600_pages

end NUMINAMATH_CALUDE_cost_of_600_pages_l3249_324989


namespace NUMINAMATH_CALUDE_quotient_problem_l3249_324966

theorem quotient_problem (q d1 d2 : ℝ) 
  (h1 : q = 6 * d1)  -- quotient is 6 times the dividend
  (h2 : q = 15 * d2) -- quotient is 15 times the divisor
  (h3 : d1 / d2 = q) -- definition of quotient
  : q = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3249_324966


namespace NUMINAMATH_CALUDE_standard_deviation_shift_l3249_324900

-- Define the standard deviation function for a list of real numbers
def standardDeviation (xs : List ℝ) : ℝ := sorry

-- Define a function to add a constant to each element of a list
def addConstant (xs : List ℝ) (c : ℝ) : List ℝ := sorry

-- Theorem statement
theorem standard_deviation_shift (a b c : ℝ) :
  standardDeviation [a + 2, b + 2, c + 2] = 2 →
  standardDeviation [a, b, c] = 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_shift_l3249_324900


namespace NUMINAMATH_CALUDE_external_equilaterals_centers_theorem_l3249_324923

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (base : Point)
  (apex : Point)

/-- Returns the center of an equilateral triangle -/
def centerOfEquilateral (t : EquilateralTriangle) : Point := sorry

/-- Returns the centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

/-- Constructs equilateral triangles on the sides of a given triangle -/
def constructExternalEquilaterals (t : Triangle) : 
  (EquilateralTriangle × EquilateralTriangle × EquilateralTriangle) := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop := sorry

theorem external_equilaterals_centers_theorem (t : Triangle) :
  let (eqAB, eqBC, eqCA) := constructExternalEquilaterals t
  let centerAB := centerOfEquilateral eqAB
  let centerBC := centerOfEquilateral eqBC
  let centerCA := centerOfEquilateral eqCA
  isEquilateral centerAB centerBC centerCA ∧
  centroid (Triangle.mk centerAB centerBC centerCA) = centroid t := by sorry

end NUMINAMATH_CALUDE_external_equilaterals_centers_theorem_l3249_324923


namespace NUMINAMATH_CALUDE_inequality_range_l3249_324941

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4*a) ↔ -1 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3249_324941


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_complex_l3249_324916

theorem imaginary_part_of_pure_imaginary_complex (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_complex_l3249_324916


namespace NUMINAMATH_CALUDE_uniqueRootIff_l3249_324936

/-- A function that represents the quadratic equation ax^2 + (a-3)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Predicate that determines if the graph of f(a, x) intersects the x-axis at only one point --/
def hasUniqueRoot (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f(a, x) has a unique root if and only if a is 0, 1, or 9 --/
theorem uniqueRootIff (a : ℝ) : hasUniqueRoot a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by
  sorry

end NUMINAMATH_CALUDE_uniqueRootIff_l3249_324936


namespace NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l3249_324940

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 108) → 
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l3249_324940


namespace NUMINAMATH_CALUDE_octahedron_sum_theorem_l3249_324930

/-- Represents an octahedron with numbered faces -/
structure NumberedOctahedron where
  lowest_number : ℕ
  face_count : ℕ
  is_consecutive : Bool
  opposite_faces_diff : ℕ

/-- The sum of numbers on an octahedron with the given properties -/
def octahedron_sum (o : NumberedOctahedron) : ℕ :=
  8 * o.lowest_number + 28

/-- Theorem stating the sum of numbers on the octahedron -/
theorem octahedron_sum_theorem (o : NumberedOctahedron) :
  o.face_count = 8 ∧ 
  o.is_consecutive = true ∧ 
  o.opposite_faces_diff = 2 →
  octahedron_sum o = 8 * o.lowest_number + 28 :=
by
  sorry

#check octahedron_sum_theorem

end NUMINAMATH_CALUDE_octahedron_sum_theorem_l3249_324930


namespace NUMINAMATH_CALUDE_expression_simplification_l3249_324926

theorem expression_simplification :
  (Real.sqrt 2 * 2^(1/2 : ℝ) * 2) + (18 / 3 * 2) - (8^(1/2 : ℝ) * 4) = 16 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3249_324926


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l3249_324927

/-- Represents a coloring of an infinite grid -/
def GridColoring := ℤ → ℤ → Bool

/-- Represents a move of the (m, n)-condylure -/
structure CondylureMove (m n : ℕ+) where
  horizontal : ℤ
  vertical : ℤ
  move_valid : (horizontal.natAbs = m ∧ vertical = 0) ∨ (horizontal = 0 ∧ vertical.natAbs = n)

/-- Theorem stating that for any positive m and n, there exists a grid coloring
    such that the (m, n)-condylure always lands on a different colored cell -/
theorem exists_valid_coloring (m n : ℕ+) :
  ∃ (coloring : GridColoring),
    ∀ (x y : ℤ) (move : CondylureMove m n),
      coloring (x + move.horizontal) (y + move.vertical) ≠ coloring x y :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l3249_324927


namespace NUMINAMATH_CALUDE_intersection_condition_area_condition_l3249_324978

/-- The hyperbola C: x² - y² = 1 -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line L: y = kx - 1 -/
def L (k x y : ℝ) : Prop := y = k * x - 1

/-- L intersects C at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- The area of triangle AOB is √2 -/
def triangle_area_sqrt_2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
    (1/2 : ℝ) * |x₁ - x₂| = Real.sqrt 2

theorem intersection_condition (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 2 < k ∧ k < -1 :=
sorry

theorem area_condition (k : ℝ) :
  triangle_area_sqrt_2 k ↔ k = 0 ∨ k = Real.sqrt 6 / 2 ∨ k = -Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_area_condition_l3249_324978


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3249_324948

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_distance : ℝ) (second_part_speed : ℝ) : 
  total_distance = 850 ∧ 
  first_part_distance = 400 ∧ 
  first_part_speed = 20 ∧
  second_part_distance = 450 ∧
  second_part_speed = 15 →
  (total_distance / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed))) = 17 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3249_324948


namespace NUMINAMATH_CALUDE_negative_difference_equals_reversed_difference_l3249_324938

theorem negative_difference_equals_reversed_difference (a b : ℝ) : 
  -(a - b) = b - a := by sorry

end NUMINAMATH_CALUDE_negative_difference_equals_reversed_difference_l3249_324938


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l3249_324919

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℝ :=
  if y ≥ 0 then Real.sqrt y else -Real.sqrt (-y)

-- Theorem statement
theorem inverse_sum_equals_negative_six :
  f_inv 9 + f_inv (-81) = -6 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_six_l3249_324919


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_ending_in_3_l3249_324910

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def hasUnitsDigitOf3 (n : ℕ) : Prop := n % 10 = 3

def firstEightPrimesEndingIn3 : List ℕ := 
  [3, 13, 23, 43, 53, 73, 83, 103]

theorem sum_first_eight_primes_ending_in_3 :
  (∀ n ∈ firstEightPrimesEndingIn3, isPrime n ∧ hasUnitsDigitOf3 n) →
  (∀ p : ℕ, isPrime p → hasUnitsDigitOf3 p → 
    p ∉ firstEightPrimesEndingIn3 → 
    p > (List.maximum firstEightPrimesEndingIn3).getD 0) →
  List.sum firstEightPrimesEndingIn3 = 394 := by
sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_ending_in_3_l3249_324910


namespace NUMINAMATH_CALUDE_gis_main_functions_l3249_324914

/-- Represents the main functions of geographic information technology -/
inductive GISFunction
  | input : GISFunction
  | manage : GISFunction
  | analyze : GISFunction
  | express : GISFunction

/-- Represents the type of data handled by geographic information technology -/
def GeospatialData : Type := Unit

/-- The set of main functions of geographic information technology -/
def mainFunctions : Set GISFunction :=
  {GISFunction.input, GISFunction.manage, GISFunction.analyze, GISFunction.express}

/-- States that the main functions of geographic information technology
    are to input, manage, analyze, and express geospatial data -/
theorem gis_main_functions :
  ∀ f : GISFunction, f ∈ mainFunctions →
  ∃ (d : GeospatialData), (f = GISFunction.input ∨ f = GISFunction.manage ∨
                           f = GISFunction.analyze ∨ f = GISFunction.express) :=
sorry

end NUMINAMATH_CALUDE_gis_main_functions_l3249_324914


namespace NUMINAMATH_CALUDE_photo_rectangle_perimeters_l3249_324994

/-- Represents a photograph with length and width -/
structure Photo where
  length : ℝ
  width : ℝ

/-- Represents a rectangle composed of photographs -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The problem statement -/
theorem photo_rectangle_perimeters 
  (photo : Photo)
  (rect1 rect2 rect3 : Rectangle)
  (h1 : 2 * (photo.length + photo.width) = 20)
  (h2 : 2 * (rect2.length + rect2.width) = 56)
  (h3 : rect1.length = 2 * photo.length ∧ rect1.width = 2 * photo.width)
  (h4 : rect2.length = photo.length ∧ rect2.width = 4 * photo.width)
  (h5 : rect3.length = 4 * photo.length ∧ rect3.width = photo.width) :
  2 * (rect1.length + rect1.width) = 40 ∧ 2 * (rect3.length + rect3.width) = 44 := by
  sorry

end NUMINAMATH_CALUDE_photo_rectangle_perimeters_l3249_324994


namespace NUMINAMATH_CALUDE_students_AD_combined_prove_students_AD_combined_l3249_324985

/-- The number of students in classes A and B combined -/
def students_AB : ℕ := 83

/-- The number of students in classes B and C combined -/
def students_BC : ℕ := 86

/-- The number of students in classes C and D combined -/
def students_CD : ℕ := 88

/-- Theorem stating that the number of students in classes A and D combined is 85 -/
theorem students_AD_combined : ℕ := 85

/-- Proof of the theorem -/
theorem prove_students_AD_combined : students_AD_combined = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_AD_combined_prove_students_AD_combined_l3249_324985


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3249_324963

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 / (x - 6) = 3 / (x + 1)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x = -22 :=
sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3249_324963


namespace NUMINAMATH_CALUDE_original_item_is_mirror_l3249_324937

-- Define the code language as a function
def code (x : String) : String :=
  match x with
  | "item" => "pencil"
  | "pencil" => "mirror"
  | "mirror" => "board"
  | _ => x

-- Define the useful item to write on paper
def useful_item : String := "pencil"

-- Define the coded useful item
def coded_useful_item : String := "2"

-- Theorem to prove
theorem original_item_is_mirror :
  (code useful_item = coded_useful_item) → 
  (∃ x, code x = useful_item ∧ code (code x) = coded_useful_item) →
  (∃ y, code y = useful_item ∧ y = "mirror") :=
by sorry

end NUMINAMATH_CALUDE_original_item_is_mirror_l3249_324937


namespace NUMINAMATH_CALUDE_dodecagon_areas_l3249_324992

/-- A regular dodecagon with side length 1 cm -/
structure RegularDodecagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- An equilateral triangle within the dodecagon -/
structure EquilateralTriangle where
  area : ℝ
  is_one_cm_squared : area = 1

/-- A square within the dodecagon -/
structure Square where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- A regular hexagon within the dodecagon -/
structure RegularHexagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- The decomposition of the dodecagon -/
structure DodecagonDecomposition where
  triangles : Finset EquilateralTriangle
  squares : Finset Square
  hexagon : RegularHexagon
  triangle_count : triangles.card = 6
  square_count : squares.card = 6

theorem dodecagon_areas 
  (d : RegularDodecagon) 
  (decomp : DodecagonDecomposition) : 
  /- 1. The area of the hexagon is 6 cm² -/
  decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6 = 6 ∧ 
  /- 2. The area of the figure formed by removing 12 equilateral triangles is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12) = 6 ∧
  /- 3. The area of the figure formed by removing 2 regular hexagons is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (2 * (decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_areas_l3249_324992


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3249_324944

/-- The complex number 2/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant : 
  let z : ℂ := 2 / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3249_324944


namespace NUMINAMATH_CALUDE_final_water_level_approx_34cm_l3249_324911

/-- Represents the properties of a liquid in a cylindrical vessel -/
structure Liquid where
  density : ℝ
  initial_height : ℝ

/-- Represents a system of two connected cylindrical vessels with different liquids -/
structure ConnectedVessels where
  water : Liquid
  oil : Liquid

/-- Calculates the final water level in the first vessel after opening the valve -/
def final_water_level (vessels : ConnectedVessels) : ℝ :=
  sorry

/-- The theorem states that given the initial conditions, the final water level
    will be approximately 34 cm -/
theorem final_water_level_approx_34cm (vessels : ConnectedVessels)
  (h_water_density : vessels.water.density = 1000)
  (h_oil_density : vessels.oil.density = 700)
  (h_initial_height : vessels.water.initial_height = 40 ∧ vessels.oil.initial_height = 40) :
  ∃ ε > 0, |final_water_level vessels - 34| < ε :=
sorry

end NUMINAMATH_CALUDE_final_water_level_approx_34cm_l3249_324911


namespace NUMINAMATH_CALUDE_five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l3249_324915

/-- The maximum number of intersection points for n lines in a plane,
    where no three lines intersect at one point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_triple_intersections : Bool

/-- Theorem stating the impossibility of 5 lines with 11 intersections -/
theorem five_lines_eleven_intersections_impossible :
  ∀ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true →
    config.num_intersections ≠ 11 :=
sorry

/-- Theorem stating the possibility of 5 lines with 9 intersections -/
theorem five_lines_nine_intersections_possible :
  ∃ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true ∧
    config.num_intersections = 9 :=
sorry

end NUMINAMATH_CALUDE_five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l3249_324915


namespace NUMINAMATH_CALUDE_sara_quarters_l3249_324928

/-- The number of quarters Sara has after receiving more from her dad -/
def total_quarters (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Theorem stating that Sara now has 70 quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l3249_324928


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3249_324949

/-- The sum of an infinite geometric series with first term 1 and common ratio 2/3 is 3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 2/3
  let S := ∑' n, a * r^n
  S = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3249_324949


namespace NUMINAMATH_CALUDE_cubic_inequality_range_l3249_324904

theorem cubic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ (3 * 2^(1/3)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_range_l3249_324904


namespace NUMINAMATH_CALUDE_handshake_count_l3249_324996

theorem handshake_count (n : Nat) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3249_324996


namespace NUMINAMATH_CALUDE_yujeong_drank_most_l3249_324976

/-- Represents the amount of water drunk by each person in liters -/
structure WaterConsumption where
  yujeong : ℚ
  eunji : ℚ
  yuna : ℚ

/-- Determines who drank the most water -/
def drankMost (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

theorem yujeong_drank_most (consumption : WaterConsumption) 
  (h1 : consumption.yujeong = 7/10)
  (h2 : consumption.eunji = 1/2)
  (h3 : consumption.yuna = 6/10) :
  drankMost consumption = "Yujeong" :=
by
  sorry

#eval drankMost ⟨7/10, 1/2, 6/10⟩

end NUMINAMATH_CALUDE_yujeong_drank_most_l3249_324976


namespace NUMINAMATH_CALUDE_oxford_high_school_population_l3249_324934

/-- The total number of people in Oxford High School -/
def total_people (teachers : ℕ) (principal : ℕ) (classes : ℕ) (students_per_class : ℕ) : ℕ :=
  teachers + principal + (classes * students_per_class)

/-- Theorem: The total number of people in Oxford High School is 349 -/
theorem oxford_high_school_population :
  total_people 48 1 15 20 = 349 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_l3249_324934


namespace NUMINAMATH_CALUDE_tens_digit_of_8_power_1701_l3249_324939

theorem tens_digit_of_8_power_1701 : ∃ n : ℕ, 8^1701 ≡ n [ZMOD 100] ∧ n < 100 ∧ (n / 10 : ℕ) = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_power_1701_l3249_324939


namespace NUMINAMATH_CALUDE_farmer_boso_animals_l3249_324932

theorem farmer_boso_animals (a b : ℕ) (h1 : 5 * b = b^(a-5)) (h2 : b = 5) (h3 : a = 7) : ∃ (L : ℕ), L = 3 ∧ 
  (4 * (5 * b) + 2 * (5 * a + 7) + 6 * b^(a-5) = 100 * L + 10 * L + L + 1) :=
sorry

end NUMINAMATH_CALUDE_farmer_boso_animals_l3249_324932


namespace NUMINAMATH_CALUDE_angle_between_AO₂_and_CO₁_is_45_degrees_l3249_324952

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter H
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the angle between two lines given by two points each
def angle_between_lines (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem angle_between_AO₂_and_CO₁_is_45_degrees 
  (ABC : Triangle) 
  (acute_angled : sorry) -- Condition: ABC is acute-angled
  (angle_B_30 : sorry) -- Condition: ∠B = 30°
  : 
  let H := orthocenter ABC
  let O₁ := incenter ABC.A ABC.B H
  let O₂ := incenter ABC.C ABC.B H
  angle_between_lines ABC.A O₂ ABC.C O₁ = 45 := by sorry

end NUMINAMATH_CALUDE_angle_between_AO₂_and_CO₁_is_45_degrees_l3249_324952


namespace NUMINAMATH_CALUDE_equation_solution_l3249_324917

theorem equation_solution : ∃ r : ℚ, 23 - 5 = 3 * r + 2 ∧ r = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3249_324917


namespace NUMINAMATH_CALUDE_circle_inscribed_line_intersection_l3249_324912

-- Define the angle
variable (angle : Angle)

-- Define the circles
variable (ω Ω : Circle)

-- Define the line
variable (l : Line)

-- Define the points
variable (A B C D E F : Point)

-- Define the inscribed property
def inscribed (c : Circle) (α : Angle) : Prop := sorry

-- Define the intersection property
def intersects (l : Line) (c : Circle) (p q : Point) : Prop := sorry

-- Define the order of points on a line
def ordered_on_line (l : Line) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop := sorry

-- Define the equality of line segments
def segment_eq (p₁ p₂ q₁ q₂ : Point) : Prop := sorry

theorem circle_inscribed_line_intersection 
  (h₁ : inscribed ω angle)
  (h₂ : inscribed Ω angle)
  (h₃ : intersects l angle A F)
  (h₄ : intersects l ω B C)
  (h₅ : intersects l Ω D E)
  (h₆ : ordered_on_line l A B C D E F)
  (h₇ : segment_eq B C D E) :
  segment_eq A B E F := by sorry

end NUMINAMATH_CALUDE_circle_inscribed_line_intersection_l3249_324912


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3249_324905

theorem inequality_equivalence (y : ℝ) : 
  (7 / 30 + |y - 19 / 60| < 17 / 30) ↔ (-1 / 60 < y ∧ y < 13 / 20) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3249_324905


namespace NUMINAMATH_CALUDE_sum_of_max_min_on_interval_l3249_324954

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem sum_of_max_min_on_interval :
  let a : ℝ := 0
  let b : ℝ := 3
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max + min = -10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_on_interval_l3249_324954


namespace NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l3249_324999

/-- The difference between the longest and shortest diagonals of a regular n-gon equals its side length if and only if n = 9 -/
theorem regular_ngon_diagonal_difference (n : ℕ) (h : n ≥ 3) :
  let R : ℝ := 1  -- Assume unit circle for simplicity
  let side_length := 2 * Real.sin (Real.pi / n)
  let shortest_diagonal := 2 * Real.sin (2 * Real.pi / n)
  let longest_diagonal := if n % 2 = 0 then 2 else 2 * Real.cos (Real.pi / (2 * n))
  longest_diagonal - shortest_diagonal = side_length ↔ n = 9 := by
sorry


end NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l3249_324999


namespace NUMINAMATH_CALUDE_solve_for_y_l3249_324956

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3249_324956


namespace NUMINAMATH_CALUDE_flour_to_add_l3249_324953

/-- The total number of cups of flour required by the recipe. -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added. -/
def flour_added : ℕ := 2

/-- The number of cups of flour Mary still needs to add. -/
def flour_needed : ℕ := total_flour - flour_added

theorem flour_to_add : flour_needed = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l3249_324953


namespace NUMINAMATH_CALUDE_liz_scored_three_three_pointers_l3249_324967

/-- Represents the basketball game scenario described in the problem -/
structure BasketballGame where
  initial_deficit : ℕ
  free_throws : ℕ
  jump_shots : ℕ
  opponent_points : ℕ
  final_deficit : ℕ

/-- Calculates the number of three-pointers Liz scored -/
def three_pointers (game : BasketballGame) : ℕ :=
  let points_needed := game.initial_deficit - game.final_deficit + game.opponent_points
  let points_from_other_shots := game.free_throws + 2 * game.jump_shots
  (points_needed - points_from_other_shots) / 3

/-- Theorem stating that Liz scored 3 three-pointers -/
theorem liz_scored_three_three_pointers :
  let game := BasketballGame.mk 20 5 4 10 8
  three_pointers game = 3 := by sorry

end NUMINAMATH_CALUDE_liz_scored_three_three_pointers_l3249_324967


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3249_324922

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_from_diameter (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 0)
  let N : ℝ × ℝ := (0, 4)
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
      radius^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ∧
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3249_324922


namespace NUMINAMATH_CALUDE_arrangement_plans_count_l3249_324987

/-- The number of ways to arrange teachers into classes -/
def arrangement_count (n m : ℕ) : ℕ :=
  -- n: total number of teachers
  -- m: number of classes
  sorry

/-- Xiao Li must be in class one -/
def xiao_li_in_class_one : Prop :=
  sorry

/-- Each class must have at least one teacher -/
def at_least_one_teacher_per_class : Prop :=
  sorry

/-- The main theorem stating the number of arrangement plans -/
theorem arrangement_plans_count :
  arrangement_count 5 3 = 50 ∧ xiao_li_in_class_one ∧ at_least_one_teacher_per_class :=
sorry

end NUMINAMATH_CALUDE_arrangement_plans_count_l3249_324987


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l3249_324950

def is_1962_digit (n : ℕ) : Prop := 10^1961 ≤ n ∧ n < 10^1962

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9 
  (n : ℕ) 
  (h1 : is_1962_digit n) 
  (h2 : n % 9 = 0) : 
  let a := sum_of_digits n
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_1962_digit_number_div_by_9_l3249_324950


namespace NUMINAMATH_CALUDE_min_trapezium_perimeter_min_trapezium_perimeter_achievable_l3249_324971

/-- A right-angled isosceles triangle with hypotenuse √2 cm -/
structure RightIsoscelesTriangle where
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = Real.sqrt 2

/-- A trapezium formed by copies of the triangle -/
structure Trapezium where
  perimeter : ℝ
  formed_by_triangles : ℕ → RightIsoscelesTriangle

/-- The theorem stating the minimum perimeter of the trapezium -/
theorem min_trapezium_perimeter (t : Trapezium) : 
  t.perimeter ≥ 4 + 2 * Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum perimeter is achievable -/
theorem min_trapezium_perimeter_achievable : 
  ∃ t : Trapezium, t.perimeter = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_trapezium_perimeter_min_trapezium_perimeter_achievable_l3249_324971


namespace NUMINAMATH_CALUDE_buttons_multiple_l3249_324984

theorem buttons_multiple (sue_buttons kendra_buttons mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : kendra_buttons = 2 * sue_buttons)
  (h3 : ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4)
  (h4 : mari_buttons = 64) : 
  ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4 ∧ m = 5 :=
by sorry

end NUMINAMATH_CALUDE_buttons_multiple_l3249_324984


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l3249_324983

/-- Given a cubic function f(x) = x^3 + ax^2 + bx + a^2 with an extreme value of 10 at x = 1,
    prove that f(2) = 18. -/
theorem extreme_value_cubic (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧
  f 1 = 10 →
  f 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l3249_324983


namespace NUMINAMATH_CALUDE_alice_basic_salary_l3249_324968

/-- Calculates the monthly basic salary given total sales, commission rate, and savings. -/
def calculate_basic_salary (total_sales : ℝ) (commission_rate : ℝ) (savings : ℝ) : ℝ :=
  let total_earnings := savings * 10
  let commission := total_sales * commission_rate
  total_earnings - commission

/-- Proves that given the specified conditions, Alice's monthly basic salary is $240. -/
theorem alice_basic_salary :
  let total_sales : ℝ := 2500
  let commission_rate : ℝ := 0.02
  let savings : ℝ := 29
  calculate_basic_salary total_sales commission_rate savings = 240 := by
  sorry

#eval calculate_basic_salary 2500 0.02 29

end NUMINAMATH_CALUDE_alice_basic_salary_l3249_324968


namespace NUMINAMATH_CALUDE_anne_wandering_l3249_324969

/-- Anne's wandering problem -/
theorem anne_wandering (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2.0 → time = 1.5 → distance = speed * time → distance = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_anne_wandering_l3249_324969


namespace NUMINAMATH_CALUDE_f_min_value_g_leq_f_range_l3249_324955

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), m = 3 ∧ ∀ x, f x ≥ m := by sorry

-- Theorem for the range of a where g(a) ≤ f(x) for all x
theorem g_leq_f_range : ∀ a : ℝ, (∀ x : ℝ, g a ≤ f x) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_f_min_value_g_leq_f_range_l3249_324955


namespace NUMINAMATH_CALUDE_cube_ratio_equals_64_l3249_324918

theorem cube_ratio_equals_64 : (88888 / 22222)^3 = 64 := by
  have h : 88888 / 22222 = 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_cube_ratio_equals_64_l3249_324918


namespace NUMINAMATH_CALUDE_probability_of_dime_l3249_324903

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 800
  | Coin.Nickel => 700
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  coinCount Coin.Dime / totalCoins = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_dime_l3249_324903


namespace NUMINAMATH_CALUDE_min_product_under_constraints_l3249_324973

theorem min_product_under_constraints (x y w : ℝ) : 
  x > 0 → y > 0 → w > 0 →
  x + y + w = 1 →
  x ≤ 2*y ∧ x ≤ 2*w ∧ y ≤ 2*x ∧ y ≤ 2*w ∧ w ≤ 2*x ∧ w ≤ 2*y →
  x*y*w ≥ 2*(2*Real.sqrt 3 - 3)/27 := by
sorry

end NUMINAMATH_CALUDE_min_product_under_constraints_l3249_324973


namespace NUMINAMATH_CALUDE_function_characterization_l3249_324929

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (|c| ≤ 1) ∧ (∀ x : ℝ, f x = c * x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3249_324929


namespace NUMINAMATH_CALUDE_f_min_value_inequality_solution_l3249_324998

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 5|

-- Theorem 1: The minimum value of f(x) is 6
theorem f_min_value : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem 2: Solution to the inequality when m = 6
theorem inequality_solution :
  ∀ x : ℝ, (|x - 3| - 2*x ≤ 4) ↔ (x ≥ -1/3) := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_solution_l3249_324998


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3249_324935

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 1, a^2 + 4}
  A ∩ B = {3} → a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3249_324935


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l3249_324974

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem b_proper_subset_of_a_iff_a_in_range (a : ℝ) :
  B a ⊂ A ↔ (0 ≤ a ∧ a ≤ 1) ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_b_proper_subset_of_a_iff_a_in_range_l3249_324974


namespace NUMINAMATH_CALUDE_eds_pets_l3249_324981

/-- The number of pets Ed has -/
def total_pets (dogs cats : ℕ) : ℕ :=
  let fish := 2 * (dogs + cats)
  let birds := dogs * cats
  dogs + cats + fish + birds

/-- Theorem stating the total number of Ed's pets -/
theorem eds_pets : total_pets 2 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eds_pets_l3249_324981


namespace NUMINAMATH_CALUDE_triangle_property_l3249_324907

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (a, Real.sqrt 3 * b)
  let n : ℝ × ℝ := (Real.cos (π / 2 - B), Real.cos (π - A))
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  c = 3 →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area formula
  A = π / 3 ∧ a = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3249_324907


namespace NUMINAMATH_CALUDE_geometric_progressions_existence_l3249_324961

theorem geometric_progressions_existence :
  (∃ a r : ℚ, 
    (∀ k : ℕ, k < 4 → 200 ≤ a * r^k ∧ a * r^k ≤ 1200) ∧
    (∀ k : ℕ, k < 4 → ∃ n : ℕ, a * r^k = n)) ∧
  (∃ b s : ℚ, 
    (∀ k : ℕ, k < 6 → 200 ≤ b * s^k ∧ b * s^k ≤ 1200) ∧
    (∀ k : ℕ, k < 6 → ∃ n : ℕ, b * s^k = n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progressions_existence_l3249_324961


namespace NUMINAMATH_CALUDE_files_remaining_l3249_324965

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 4)
  (h2 : video_files = 21)
  (h3 : deleted_files = 23) :
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3249_324965


namespace NUMINAMATH_CALUDE_transformed_line_equation_l3249_324951

/-- Given a line and a scaling transformation, prove the equation of the transformed line -/
theorem transformed_line_equation (x y x' y' : ℝ) :
  (x - 2 * y = 2) →  -- Original line equation
  (x' = x) →         -- Scaling transformation for x
  (y' = 2 * y) →     -- Scaling transformation for y
  (x' - y' - 2 = 0)  -- Resulting line equation
:= by sorry

end NUMINAMATH_CALUDE_transformed_line_equation_l3249_324951


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3249_324947

/-- The maximum value of x+y given x^2 + y^2 = 100 and xy = 36 is 2√43 -/
theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3249_324947


namespace NUMINAMATH_CALUDE_quadratic_equations_same_roots_l3249_324906

/-- Two quadratic equations have the same roots if and only if their coefficients are proportional -/
theorem quadratic_equations_same_roots (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) :
  (∀ x, a₁ * x^2 + b₁ * x + c₁ = 0 ↔ a₂ * x^2 + b₂ * x + c₂ = 0) ↔
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_same_roots_l3249_324906


namespace NUMINAMATH_CALUDE_facebook_group_members_l3249_324975

/-- Proves that the original number of members in a Facebook group was 150 -/
theorem facebook_group_members : 
  ∀ (original_members removed_members remaining_messages_per_week messages_per_member_per_day : ℕ),
  removed_members = 20 →
  messages_per_member_per_day = 50 →
  remaining_messages_per_week = 45500 →
  original_members = 
    (remaining_messages_per_week / (messages_per_member_per_day * 7)) + removed_members →
  original_members = 150 := by
sorry

end NUMINAMATH_CALUDE_facebook_group_members_l3249_324975


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3249_324957

theorem discount_percentage_proof (num_people : ℕ) (savings_per_person : ℝ) (final_price : ℝ) :
  num_people = 3 →
  savings_per_person = 4 →
  final_price = 48 →
  let total_savings := num_people * savings_per_person
  let original_price := final_price + total_savings
  let discount_percentage := (total_savings / original_price) * 100
  discount_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3249_324957


namespace NUMINAMATH_CALUDE_price_difference_shirt_sweater_l3249_324925

theorem price_difference_shirt_sweater : 
  ∀ (shirt_price sweater_price : ℝ),
    shirt_price = 36.46 →
    shirt_price < sweater_price →
    shirt_price + sweater_price = 80.34 →
    sweater_price - shirt_price = 7.42 := by
sorry

end NUMINAMATH_CALUDE_price_difference_shirt_sweater_l3249_324925


namespace NUMINAMATH_CALUDE_prime_composite_inequality_composite_inequality_exists_l3249_324997

theorem prime_composite_inequality (n : ℕ) :
  (∀ (a : Fin n → ℕ), Function.Injective a →
    ∃ (i j : Fin n), i ≠ j ∧ (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ↔
  Nat.Prime (2 * n - 1) :=
by sorry

theorem composite_inequality_exists (n : ℕ) :
  ¬(Nat.Prime (2 * n - 1)) →
  ∃ (a : Fin n → ℕ), Function.Injective a ∧
    ∀ (i j : Fin n), (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_composite_inequality_exists_l3249_324997


namespace NUMINAMATH_CALUDE_translation_theorem_l3249_324960

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally and vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -2, y := 3 }
  let A' : Point := translate (translate A 0 (-3)) 4 0
  A'.x = 2 ∧ A'.y = 0 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3249_324960


namespace NUMINAMATH_CALUDE_power_of_128_four_sevenths_l3249_324902

theorem power_of_128_four_sevenths : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_four_sevenths_l3249_324902


namespace NUMINAMATH_CALUDE_prob_all_players_odd_sum_l3249_324991

/-- The number of tiles --/
def n : ℕ := 12

/-- The number of odd tiles --/
def odd_tiles : ℕ := n / 2

/-- The number of even tiles --/
def even_tiles : ℕ := n / 2

/-- The number of tiles each player selects --/
def tiles_per_player : ℕ := 4

/-- The number of players --/
def num_players : ℕ := 3

/-- The probability of all players getting an odd sum --/
def prob_all_odd_sum : ℚ := 800 / 963

/-- Theorem stating the probability of all players getting an odd sum --/
theorem prob_all_players_odd_sum :
  let total_distributions := Nat.choose n tiles_per_player * 
                             Nat.choose (n - tiles_per_player) tiles_per_player * 
                             Nat.choose (n - 2 * tiles_per_player) tiles_per_player
  let odd_sum_distributions := (Nat.choose odd_tiles 3 * Nat.choose even_tiles 1)^num_players / 
                               Nat.factorial num_players
  (odd_sum_distributions : ℚ) / total_distributions = prob_all_odd_sum := by
  sorry

end NUMINAMATH_CALUDE_prob_all_players_odd_sum_l3249_324991


namespace NUMINAMATH_CALUDE_total_spent_calculation_l3249_324964

-- Define currency exchange rates
def gbp_to_usd : ℝ := 1.38
def eur_to_usd : ℝ := 1.12
def jpy_to_usd : ℝ := 0.0089

-- Define purchases
def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost_gbp : ℝ := 85.62
def tires_quantity : ℕ := 4
def printer_cables_cost_eur : ℝ := 12.54
def printer_cables_quantity : ℕ := 2
def blank_cds_cost_jpy : ℝ := 9800

-- Define sales tax rate
def sales_tax_rate : ℝ := 0.0825

-- Theorem statement
theorem total_spent_calculation :
  let usd_taxable := speakers_cost + cd_player_cost
  let usd_tax := usd_taxable * sales_tax_rate
  let usd_with_tax := usd_taxable + usd_tax
  let tires_usd := (tires_cost_gbp * tires_quantity) * gbp_to_usd
  let cables_usd := (printer_cables_cost_eur * printer_cables_quantity) * eur_to_usd
  let cds_usd := blank_cds_cost_jpy * jpy_to_usd
  usd_with_tax + tires_usd + cables_usd + cds_usd = 886.04 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l3249_324964


namespace NUMINAMATH_CALUDE_factor_into_sqrt_l3249_324958

theorem factor_into_sqrt (a b : ℝ) (h : a < b) :
  (a - b) * Real.sqrt (-1 / (a - b)) = -Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factor_into_sqrt_l3249_324958


namespace NUMINAMATH_CALUDE_comic_book_frames_per_page_l3249_324988

/-- Given a comic book with a total number of frames and pages, 
    calculate the number of frames per page. -/
def frames_per_page (total_frames : ℕ) (total_pages : ℕ) : ℕ :=
  total_frames / total_pages

/-- Theorem stating that for a comic book with 143 frames and 13 pages, 
    the number of frames per page is 11. -/
theorem comic_book_frames_per_page :
  frames_per_page 143 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_frames_per_page_l3249_324988
