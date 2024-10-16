import Mathlib

namespace NUMINAMATH_CALUDE_expected_hits_greater_than_half_l1201_120155

/-- The expected number of hit targets is always greater than or equal to half the number of boys/targets. -/
theorem expected_hits_greater_than_half (n : ℕ) (hn : n > 0) :
  n * (1 - (1 - 1 / n)^n) ≥ n / 2 := by
  sorry

#check expected_hits_greater_than_half

end NUMINAMATH_CALUDE_expected_hits_greater_than_half_l1201_120155


namespace NUMINAMATH_CALUDE_angle_U_measure_l1201_120169

-- Define the hexagon and its angles
structure Hexagon :=
  (F I G U R E : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.F + h.I + h.G + h.U + h.R + h.E = 720

def angles_congruent (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U

def angles_supplementary (h : Hexagon) : Prop :=
  h.G + h.R = 180 ∧ h.E + h.U = 180

-- Theorem statement
theorem angle_U_measure (h : Hexagon) 
  (valid : is_valid_hexagon h) 
  (congruent : angles_congruent h)
  (supplementary : angles_supplementary h) : 
  h.U = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_U_measure_l1201_120169


namespace NUMINAMATH_CALUDE_inverse_proportion_k_condition_l1201_120110

/-- Theorem: For an inverse proportion function y = (k-1)/x, given two points
    A(x₁, y₁) and B(x₂, y₂) on its graph where 0 < x₁ < x₂ and y₁ < y₂, 
    the value of k must be less than 1. -/
theorem inverse_proportion_k_condition 
  (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (k - 1) / x₁) (h5 : y₂ = (k - 1) / x₂) : 
  k < 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_condition_l1201_120110


namespace NUMINAMATH_CALUDE_pie_not_crust_percentage_l1201_120196

/-- Given a pie weighing 200 grams with 50 grams of crust, 
    the percentage of the pie that is not crust is 75%. -/
theorem pie_not_crust_percentage 
  (total_weight : ℝ) 
  (crust_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pie_not_crust_percentage_l1201_120196


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1201_120122

theorem remainder_divisibility (N : ℕ) (h : N % 125 = 40) : N % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1201_120122


namespace NUMINAMATH_CALUDE_isosceles_diagonal_implies_two_equal_among_four_l1201_120128

-- Define a convex n-gon
structure ConvexNGon where
  n : ℕ
  sides : Fin n → ℝ
  is_convex : Bool
  n_gt_4 : n > 4

-- Define the isosceles triangle property
def isosceles_diagonal_property (polygon : ConvexNGon) : Prop :=
  ∀ (i j : Fin polygon.n), i ≠ j → 
    ∃ (k : Fin polygon.n), k ≠ i ∧ k ≠ j ∧ 
      (polygon.sides i = polygon.sides k ∨ polygon.sides j = polygon.sides k)

-- Define the property of having at least two equal sides among any four
def two_equal_among_four (polygon : ConvexNGon) : Prop :=
  ∀ (i j k l : Fin polygon.n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i →
    polygon.sides i = polygon.sides j ∨ polygon.sides i = polygon.sides k ∨
    polygon.sides i = polygon.sides l ∨ polygon.sides j = polygon.sides k ∨
    polygon.sides j = polygon.sides l ∨ polygon.sides k = polygon.sides l

-- The theorem to be proved
theorem isosceles_diagonal_implies_two_equal_among_four 
  (polygon : ConvexNGon) (h : isosceles_diagonal_property polygon) :
  two_equal_among_four polygon := by
  sorry

end NUMINAMATH_CALUDE_isosceles_diagonal_implies_two_equal_among_four_l1201_120128


namespace NUMINAMATH_CALUDE_thirteen_sided_polygon_property_n_sided_polygon_property_l1201_120151

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

-- Define a line type
structure Line :=
  (a b c : ℝ)

-- Function to check if a line contains a side of a polygon
def line_contains_side (l : Line) (p : Polygon) (i : Fin p.sides) : Prop :=
  -- Implementation details omitted
  sorry

-- Function to count how many sides of a polygon a line contains
def count_sides_on_line (l : Line) (p : Polygon) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem for 13-sided polygons
theorem thirteen_sided_polygon_property :
  ∀ (p : Polygon), p.sides = 13 →
  ∃ (l : Line), ∃ (i : Fin p.sides),
    line_contains_side l p i ∧
    count_sides_on_line l p = 1 :=
sorry

-- Theorem for n-sided polygons where n > 13
theorem n_sided_polygon_property :
  ∀ (n : ℕ), n > 13 →
  ∃ (p : Polygon), p.sides = n ∧
  ∀ (l : Line), ∀ (i : Fin p.sides),
    line_contains_side l p i →
    count_sides_on_line l p ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_thirteen_sided_polygon_property_n_sided_polygon_property_l1201_120151


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficients_l1201_120187

/-- Given a cubic polynomial with real coefficients that has 2 - 3i as a root,
    prove that its coefficients are a = -3, b = 1, and c = -39. -/
theorem cubic_polynomial_coefficients 
  (p : ℂ → ℂ) 
  (h1 : ∀ x, p x = x^3 + a*x^2 + b*x - c) 
  (h2 : p (2 - 3*I) = 0) 
  (a b c : ℝ) :
  a = -3 ∧ b = 1 ∧ c = -39 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficients_l1201_120187


namespace NUMINAMATH_CALUDE_total_legs_equals_1564_l1201_120145

/-- Calculates the total number of legs of all animals owned by Mark -/
def totalLegs (numKangaroos : ℕ) : ℕ :=
  let numGoats := 3 * numKangaroos
  let numSpiders := 2 * numGoats
  let numBirds := numSpiders / 2
  let kangarooLegs := 2 * numKangaroos
  let goatLegs := 4 * numGoats
  let spiderLegs := 8 * numSpiders
  let birdLegs := 2 * numBirds
  kangarooLegs + goatLegs + spiderLegs + birdLegs

/-- Theorem stating that the total number of legs of all Mark's animals is 1564 -/
theorem total_legs_equals_1564 : totalLegs 23 = 1564 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_equals_1564_l1201_120145


namespace NUMINAMATH_CALUDE_fifteen_sided_figure_area_l1201_120186

/-- A fifteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FifteenSidedFigure where
  full_squares : ℕ
  small_triangles : ℕ
  h_full_squares : full_squares = 10
  h_small_triangles : small_triangles = 10

/-- The area of the fifteen-sided figure is 15 cm² -/
theorem fifteen_sided_figure_area (fig : FifteenSidedFigure) : 
  (fig.full_squares : ℝ) + (fig.small_triangles : ℝ) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_figure_area_l1201_120186


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1201_120160

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) : 
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution : 
  ∃ (k : ℕ), k = 3 ∧ (427398 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (427398 - m) % 15 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1201_120160


namespace NUMINAMATH_CALUDE_at_least_one_zero_l1201_120152

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_zero_l1201_120152


namespace NUMINAMATH_CALUDE_fraction_equality_l1201_120185

theorem fraction_equality (a b : ℝ) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1201_120185


namespace NUMINAMATH_CALUDE_zach_savings_l1201_120137

/-- Represents the financial situation of Zach saving for a bike --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  lawnMowingPay : ℕ
  babysittingRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach has already saved --/
def amountSaved (s : BikeSavings) : ℕ :=
  s.bikeCost - (s.weeklyAllowance + s.lawnMowingPay + s.babysittingRate * s.babysittingHours) - s.additionalNeeded

/-- Theorem stating that for Zach's specific situation, he has already saved $65 --/
theorem zach_savings : 
  let s : BikeSavings := {
    bikeCost := 100,
    weeklyAllowance := 5,
    lawnMowingPay := 10,
    babysittingRate := 7,
    babysittingHours := 2,
    additionalNeeded := 6
  }
  amountSaved s = 65 := by sorry

end NUMINAMATH_CALUDE_zach_savings_l1201_120137


namespace NUMINAMATH_CALUDE_folded_rectangle_theorem_l1201_120130

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle with vertices A, B, C, D -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents the folded state of the rectangle -/
structure FoldedRectangle :=
  (rect : Rectangle)
  (E : Point)
  (F : Point)

/-- Given a folded rectangle, returns the measure of Angle 1 in degrees -/
def angle1 (fr : FoldedRectangle) : ℝ := sorry

/-- Given a folded rectangle, returns the measure of Angle 2 in degrees -/
def angle2 (fr : FoldedRectangle) : ℝ := sorry

/-- Predicate to check if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def areCongruentTriangles (ABC DEF : Point × Point × Point) : Prop := sorry

theorem folded_rectangle_theorem (fr : FoldedRectangle) :
  isOnSegment fr.rect.A fr.E fr.rect.B ∧
  areCongruentTriangles (fr.rect.D, fr.rect.C, fr.F) (fr.rect.D, fr.E, fr.F) ∧
  angle1 fr = 22 →
  angle2 fr = 44 :=
sorry

end NUMINAMATH_CALUDE_folded_rectangle_theorem_l1201_120130


namespace NUMINAMATH_CALUDE_somu_age_problem_l1201_120162

theorem somu_age_problem (somu_age father_age : ℕ) : 
  (somu_age = father_age / 3) →
  (somu_age - 10 = (father_age - 10) / 5) →
  somu_age = 20 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l1201_120162


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l1201_120129

theorem range_of_m_for_quadratic_equation (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo (-1) 0 ∧ x₂ ∈ Set.Ioi 3 ∧
    x₁^2 - 2*m*x₁ + m - 3 = 0 ∧ x₂^2 - 2*m*x₂ + m - 3 = 0) →
  m ∈ Set.Ioo (6/5) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l1201_120129


namespace NUMINAMATH_CALUDE_ryan_learning_time_l1201_120126

/-- Represents the time Ryan spends on learning languages in hours -/
structure LearningTime where
  total : ℝ
  english : ℝ
  chinese : ℝ

/-- Theorem: Given Ryan's total learning time and English learning time, 
    prove that his Chinese learning time is the difference -/
theorem ryan_learning_time (rt : LearningTime) 
  (h1 : rt.total = 3) 
  (h2 : rt.english = 2) 
  (h3 : rt.total = rt.english + rt.chinese) : 
  rt.chinese = 1 := by
sorry

end NUMINAMATH_CALUDE_ryan_learning_time_l1201_120126


namespace NUMINAMATH_CALUDE_accurate_counting_requires_shaking_l1201_120113

/-- Represents a yeast cell -/
structure YeastCell where
  id : ℕ

/-- Represents a culture fluid containing yeast cells -/
structure CultureFluid where
  cells : List YeastCell

/-- Represents a test tube containing culture fluid -/
structure TestTube where
  fluid : CultureFluid

/-- Represents a hemocytometer for counting cells -/
structure Hemocytometer where
  volume : ℝ
  count : CultureFluid → ℕ

/-- Yeast is a unicellular fungus -/
axiom yeast_is_unicellular : ∀ (y : YeastCell), true

/-- Yeast is a facultative anaerobe -/
axiom yeast_is_facultative_anaerobe : ∀ (y : YeastCell), true

/-- Yeast distribution in culture fluid is uneven -/
axiom yeast_distribution_uneven : ∀ (cf : CultureFluid), true

/-- A hemocytometer is used for counting yeast cells -/
axiom hemocytometer_used : ∃ (h : Hemocytometer), true

/-- Shaking the test tube before sampling leads to accurate counting -/
theorem accurate_counting_requires_shaking (tt : TestTube) (h : Hemocytometer) :
  (∀ (sample : CultureFluid), h.count sample = h.count tt.fluid) ↔ 
  (∃ (shaken_tt : TestTube), shaken_tt.fluid = tt.fluid ∧ 
    ∀ (sample : CultureFluid), h.count sample = h.count shaken_tt.fluid) :=
sorry

end NUMINAMATH_CALUDE_accurate_counting_requires_shaking_l1201_120113


namespace NUMINAMATH_CALUDE_product_14_sum_9_l1201_120171

theorem product_14_sum_9 :
  ∀ a b : ℕ, 
    1 ≤ a ∧ a ≤ 10 →
    1 ≤ b ∧ b ≤ 10 →
    a * b = 14 →
    a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_14_sum_9_l1201_120171


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l1201_120180

theorem ring_toss_earnings (daily_earnings : ℕ) (days : ℕ) (total_earnings : ℕ) : 
  daily_earnings = 33 → days = 5 → total_earnings = daily_earnings * days → total_earnings = 165 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l1201_120180


namespace NUMINAMATH_CALUDE_expression_simplification_l1201_120156

theorem expression_simplification :
  Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt 3 - 1| + Real.sqrt 3 = -13 / 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1201_120156


namespace NUMINAMATH_CALUDE_school_year_work_hours_jacqueline_work_hours_l1201_120131

/-- Calculates the required weekly work hours during school year given summer work details and school year target. -/
theorem school_year_work_hours 
  (summer_hours_per_week : ℕ) 
  (summer_weeks : ℕ) 
  (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) 
  (school_year_target : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := school_year_target / hourly_wage
  total_school_year_hours / school_year_weeks

/-- Proves that given the specific conditions, Jacqueline needs to work 15 hours per week during the school year. -/
theorem jacqueline_work_hours : 
  school_year_work_hours 60 8 6000 40 7500 = 15 := by
  sorry

end NUMINAMATH_CALUDE_school_year_work_hours_jacqueline_work_hours_l1201_120131


namespace NUMINAMATH_CALUDE_james_out_of_pocket_l1201_120172

def initial_purchase : ℝ := 3000
def returned_tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost_increase : ℝ := 0.2
def sold_bike_price_ratio : ℝ := 0.8
def toaster_cost : ℝ := 100

theorem james_out_of_pocket :
  let remaining_after_returns := initial_purchase - returned_tv_cost - returned_bike_cost
  let sold_bike_cost := returned_bike_cost * (1 + sold_bike_cost_increase)
  let sold_bike_price := sold_bike_cost * sold_bike_price_ratio
  let final_amount := remaining_after_returns - sold_bike_price + toaster_cost
  final_amount = 1420 := by sorry

end NUMINAMATH_CALUDE_james_out_of_pocket_l1201_120172


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1201_120158

theorem triangle_angle_problem (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1201_120158


namespace NUMINAMATH_CALUDE_smallest_integer_l1201_120192

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  b ≥ 60 ∧ ∀ c : ℕ, c < 60 → Nat.lcm a c / Nat.gcd a c ≠ 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l1201_120192


namespace NUMINAMATH_CALUDE_A_intersect_B_l1201_120179

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1201_120179


namespace NUMINAMATH_CALUDE_rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l1201_120195

/-- The value of x for a rectangle with specific properties -/
def rectangle_x : ℝ := 1.924

/-- The length of the rectangle -/
def length (x : ℝ) : ℝ := 5 * x

/-- The width of the rectangle -/
def width (x : ℝ) : ℝ := 2 * x + 3

/-- The area of the rectangle -/
def area (x : ℝ) : ℝ := length x * width x

/-- The perimeter of the rectangle -/
def perimeter (x : ℝ) : ℝ := 2 * (length x + width x)

/-- Theorem stating that rectangle_x satisfies the given conditions -/
theorem rectangle_x_satisfies_conditions :
  area rectangle_x = 2 * perimeter rectangle_x ∧
  length rectangle_x > 0 ∧
  width rectangle_x > 0 := by
  sorry

/-- Theorem stating that rectangle_x is the unique solution -/
theorem rectangle_x_unique_solution :
  ∀ y : ℝ, (area y = 2 * perimeter y ∧ length y > 0 ∧ width y > 0) → y = rectangle_x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l1201_120195


namespace NUMINAMATH_CALUDE_quadratic_form_h_l1201_120109

theorem quadratic_form_h (a k h : ℝ) : 
  (∀ x, 8 * x^2 + 12 * x + 7 = a * (x - h)^2 + k) → h = -3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_l1201_120109


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1201_120175

theorem problem_1 : 2023^2 - 2024 * 2022 = 1 := by sorry

theorem problem_2 (a b c : ℝ) : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1201_120175


namespace NUMINAMATH_CALUDE_sum_of_evens_1_to_101_l1201_120157

theorem sum_of_evens_1_to_101 : 
  (Finset.range 51).sum (fun i => 2 * i) = 2550 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_1_to_101_l1201_120157


namespace NUMINAMATH_CALUDE_root_existence_and_bounds_l1201_120121

theorem root_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x₁ x₂ : ℝ),
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 ≤ x₁ ∧ x₁ ≤ 2 * a / 3) ∧
    (-2 * b / 3 ≤ x₂ ∧ x₂ ≤ -b / 3) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_and_bounds_l1201_120121


namespace NUMINAMATH_CALUDE_square_area_ratio_l1201_120143

theorem square_area_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (perimeter_ratio : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1201_120143


namespace NUMINAMATH_CALUDE_lineups_count_is_sixty_l1201_120108

/-- The number of ways to arrange r items out of n items -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible lineups for 3 games with 3 athletes selected from 5 -/
def lineups_count : ℕ := permutations 5 3

theorem lineups_count_is_sixty : lineups_count = 60 := by sorry

end NUMINAMATH_CALUDE_lineups_count_is_sixty_l1201_120108


namespace NUMINAMATH_CALUDE_seeds_sown_count_l1201_120133

/-- The number of seeds that germinated -/
def seeds_germinated : ℕ := 970

/-- The frequency of normal seed germination -/
def germination_rate : ℚ := 97/100

/-- The total number of seeds sown -/
def total_seeds : ℕ := 1000

/-- Theorem stating that the total number of seeds sown is 1000 -/
theorem seeds_sown_count : 
  (seeds_germinated : ℚ) / germination_rate = total_seeds := by sorry

end NUMINAMATH_CALUDE_seeds_sown_count_l1201_120133


namespace NUMINAMATH_CALUDE_base_k_equivalence_l1201_120114

/-- 
Given a natural number k, this function converts a number from base k to decimal.
The input is a list of digits in reverse order (least significant digit first).
-/
def baseKToDecimal (k : ℕ) (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * k^i) 0

/-- 
This theorem states that if 26 in decimal is equal to 32 in base-k, then k must be 8.
-/
theorem base_k_equivalence :
  ∀ k : ℕ, k > 1 → baseKToDecimal k [2, 3] = 26 → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_k_equivalence_l1201_120114


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l1201_120139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- The x-coordinate of the directrix for the parabola y^2 = 8x -/
def directrix_x : ℝ := -2

/-- Distance between two x-coordinates -/
def distance_x (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem parabola_point_to_directrix_distance
  (P : Point)
  (h1 : P ∈ Parabola)
  (h2 : distance_x P.x 0 = 4) :
  distance_x P.x directrix_x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l1201_120139


namespace NUMINAMATH_CALUDE_prob_3_red_in_5_draws_eq_8_81_l1201_120176

/-- The probability of drawing a red ball from the bag -/
def prob_red : ℚ := 1 / 3

/-- The probability of drawing a white ball from the bag -/
def prob_white : ℚ := 2 / 3

/-- The number of ways to choose 2 red balls from 4 draws -/
def ways_to_choose_2_from_4 : ℕ := 6

/-- The probability of drawing exactly 3 red balls in 5 draws, with the last draw being red -/
def prob_3_red_in_5_draws : ℚ :=
  ways_to_choose_2_from_4 * prob_red^2 * prob_white^2 * prob_red

theorem prob_3_red_in_5_draws_eq_8_81 : 
  prob_3_red_in_5_draws = 8 / 81 := by sorry

end NUMINAMATH_CALUDE_prob_3_red_in_5_draws_eq_8_81_l1201_120176


namespace NUMINAMATH_CALUDE_carton_length_is_30_inches_l1201_120141

/-- Proves that the length of a carton is 30 inches given specific dimensions and constraints -/
theorem carton_length_is_30_inches 
  (carton_width : ℕ) 
  (carton_height : ℕ)
  (soap_length : ℕ) 
  (soap_width : ℕ) 
  (soap_height : ℕ)
  (max_soap_boxes : ℕ)
  (h1 : carton_width = 42)
  (h2 : carton_height = 60)
  (h3 : soap_length = 7)
  (h4 : soap_width = 6)
  (h5 : soap_height = 5)
  (h6 : max_soap_boxes = 360) :
  ∃ (carton_length : ℕ), carton_length = 30 ∧ 
    carton_length * carton_width * carton_height = 
    max_soap_boxes * soap_length * soap_width * soap_height :=
by
  sorry

end NUMINAMATH_CALUDE_carton_length_is_30_inches_l1201_120141


namespace NUMINAMATH_CALUDE_f_composition_result_l1201_120127

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_composition_result :
  f (f (f (f (1 + 2*I)))) = 503521 + 420000*I :=
by sorry

end NUMINAMATH_CALUDE_f_composition_result_l1201_120127


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1201_120119

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 32) : 
  (a 2 / a 1 = 2) ∨ (a 2 / a 1 = -2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1201_120119


namespace NUMINAMATH_CALUDE_difference_of_squares_l1201_120149

theorem difference_of_squares (n : ℕ) : 
  (n = 105 → ∃! k : ℕ, k = 4 ∧ ∃ s : Finset (ℕ × ℕ), s.card = k ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ x^2 - y^2 = n) ∧
  (n = 106 → ¬∃ (x y : ℕ), x^2 - y^2 = n) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1201_120149


namespace NUMINAMATH_CALUDE_annulus_area_dead_grass_area_l1201_120188

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (r₁ r₂ : ℝ) (h : 0 < r₁ ∧ r₁ < r₂) : 
  π * (r₂^2 - r₁^2) = π * (r₂ + r₁) * (r₂ - r₁) :=
by sorry

/-- The area of dead grass caused by a walking man with a sombrero -/
theorem dead_grass_area (r_walk r_sombrero : ℝ) 
  (h_walk : r_walk = 5)
  (h_sombrero : r_sombrero = 3) : 
  π * ((r_walk + r_sombrero)^2 - (r_walk - r_sombrero)^2) = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_annulus_area_dead_grass_area_l1201_120188


namespace NUMINAMATH_CALUDE_min_value_C_over_D_l1201_120102

theorem min_value_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^2 + 1/x^2 = C) (hD : x + 1/x = D) (hCpos : C > 0) (hDpos : D > 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = C / D → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_C_over_D_l1201_120102


namespace NUMINAMATH_CALUDE_expression_evaluation_l1201_120153

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1201_120153


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l1201_120199

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on AB and AC respectively
def D (triangle : Triangle) : ℝ × ℝ := sorry
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the angle bisector AT
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the intersection point F of AT and DE
def F (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths
def AD (triangle : Triangle) : ℝ := 2
def DB (triangle : Triangle) : ℝ := 6
def AE (triangle : Triangle) : ℝ := 5
def EC (triangle : Triangle) : ℝ := 3

-- Define the ratio AF/AT
def AF_AT_ratio (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (triangle : Triangle) :
  AF_AT_ratio triangle = 2/5 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l1201_120199


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l1201_120135

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x + 1| ≥ 0) ↔ (∃ x : ℝ, |x + 1| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l1201_120135


namespace NUMINAMATH_CALUDE_matrix_commute_special_case_l1201_120183

open Matrix

theorem matrix_commute_special_case 
  (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![10, 3; -2, 5]) : 
  D * C = C * D := by
sorry

end NUMINAMATH_CALUDE_matrix_commute_special_case_l1201_120183


namespace NUMINAMATH_CALUDE_line_segment_proportion_l1201_120167

theorem line_segment_proportion (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a / c = c / b → a = 4 → b = 9 → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l1201_120167


namespace NUMINAMATH_CALUDE_smallest_with_four_odd_eight_even_divisors_l1201_120178

/-- Count of positive odd integer divisors of n -/
def oddDivisorCount (n : ℕ+) : ℕ := sorry

/-- Count of positive even integer divisors of n -/
def evenDivisorCount (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having exactly four positive odd integer divisors and eight positive even integer divisors -/
def hasFourOddEightEvenDivisors (n : ℕ+) : Prop :=
  oddDivisorCount n = 4 ∧ evenDivisorCount n = 8

theorem smallest_with_four_odd_eight_even_divisors :
  ∃ (n : ℕ+), hasFourOddEightEvenDivisors n ∧
  ∀ (m : ℕ+), hasFourOddEightEvenDivisors m → n ≤ m :=
by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_with_four_odd_eight_even_divisors_l1201_120178


namespace NUMINAMATH_CALUDE_max_value_cubic_function_l1201_120189

theorem max_value_cubic_function :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^3 - 3*x^2 + 2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cubic_function_l1201_120189


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1201_120132

theorem min_value_of_expression (x : ℝ) (h : x > 2) :
  x + 2 / (x - 2) ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

theorem lower_bound_achievable :
  ∃ x > 2, x + 2 / (x - 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l1201_120132


namespace NUMINAMATH_CALUDE_probability_of_different_colors_l1201_120164

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 4

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 5

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + green_marbles + white_marbles + blue_marbles

/-- The number of ways to choose 2 marbles from the total number of marbles -/
def total_ways : ℕ := total_marbles.choose 2

/-- The number of ways to choose 2 marbles of different colors -/
def different_color_ways : ℕ :=
  red_marbles * green_marbles +
  red_marbles * white_marbles +
  red_marbles * blue_marbles +
  green_marbles * white_marbles +
  green_marbles * blue_marbles +
  white_marbles * blue_marbles

/-- The probability of drawing two marbles of different colors -/
def probability : ℚ := different_color_ways / total_ways

theorem probability_of_different_colors :
  probability = 83 / 105 :=
sorry

end NUMINAMATH_CALUDE_probability_of_different_colors_l1201_120164


namespace NUMINAMATH_CALUDE_moving_circle_locus_l1201_120100

-- Define the fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the locus of the center of the moving circle
def locus (x y : ℝ) : Prop := (x + 2)^2 - (y^2 / 13^2) = 1 ∧ x < -1

-- State the theorem
theorem moving_circle_locus :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), circle_M x' y' → (x - x')^2 + (y - y')^2 = r^2) ∧
    (∀ (x' y' : ℝ), circle_N x' y' → (x - x')^2 + (y - y')^2 = r^2)) →
  locus x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_locus_l1201_120100


namespace NUMINAMATH_CALUDE_right_triangle_cos_a_l1201_120115

theorem right_triangle_cos_a (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) : 
  B = π/2 → 3 * Real.tan A = 4 * Real.sin A → Real.cos A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_a_l1201_120115


namespace NUMINAMATH_CALUDE_change_calculation_l1201_120184

/-- Calculates the change in USD given the cost per cup in Euros, payment in Euros, and USD/Euro conversion rate -/
def calculate_change_usd (cost_per_cup_eur : ℝ) (payment_eur : ℝ) (usd_per_eur : ℝ) : ℝ :=
  (payment_eur - cost_per_cup_eur) * usd_per_eur

/-- Proves that the change received is 0.4956 USD given the specified conditions -/
theorem change_calculation :
  let cost_per_cup_eur : ℝ := 0.58
  let payment_eur : ℝ := 1
  let usd_per_eur : ℝ := 1.18
  calculate_change_usd cost_per_cup_eur payment_eur usd_per_eur = 0.4956 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l1201_120184


namespace NUMINAMATH_CALUDE_unique_c_value_l1201_120190

theorem unique_c_value (c : ℝ) : c + ⌊c⌋ = 23.2 → c = 11.7 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l1201_120190


namespace NUMINAMATH_CALUDE_linear_function_properties_l1201_120118

/-- A linear function y = ax + b satisfying specific conditions -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_properties (a b : ℝ) :
  (LinearFunction a b 1 = 1 ∧ LinearFunction a b 2 = -5) →
  (a = -6 ∧ b = 7 ∧
   LinearFunction a b 0 = 7 ∧
   ∀ x, LinearFunction a b x > 0 ↔ x < 7/6) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1201_120118


namespace NUMINAMATH_CALUDE_problem_solution_l1201_120148

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  ∃ m : ℝ,
    m > 0 ∧
    (∀ x : ℝ, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1) ∧
    -- Part 1: The value of m is 1
    m = 1 ∧
    -- Part 2: Maximum value of 2a√(1+b²) is 2√2
    (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 * m →
      2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 * m ∧
      2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1201_120148


namespace NUMINAMATH_CALUDE_one_cut_divides_two_squares_equally_l1201_120124

-- Define a square
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Function to check if a line passes through a point
def line_passes_through_point (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
               l.point1.2 + t * (l.point2.2 - l.point1.2))

-- Function to check if a line divides a square into two equal parts
def line_divides_square_equally (l : Line) (s : Square) : Prop :=
  line_passes_through_point l s.center

-- Theorem statement
theorem one_cut_divides_two_squares_equally 
  (s1 s2 : Square) (l : Line) : 
  line_passes_through_point l s1.center → 
  line_passes_through_point l s2.center → 
  line_divides_square_equally l s1 ∧ line_divides_square_equally l s2 :=
sorry

end NUMINAMATH_CALUDE_one_cut_divides_two_squares_equally_l1201_120124


namespace NUMINAMATH_CALUDE_range_of_z_l1201_120198

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l1201_120198


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l1201_120159

theorem quadratic_function_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a * x - b * x^2 ≤ 1) → a ≤ 2 * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l1201_120159


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1201_120150

theorem perfect_squares_between_50_and_200 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 200) (Finset.range 15)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_200_l1201_120150


namespace NUMINAMATH_CALUDE_multiply_by_three_l1201_120191

theorem multiply_by_three (x : ℤ) : x + 14 = 56 → 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_three_l1201_120191


namespace NUMINAMATH_CALUDE_households_with_appliances_l1201_120194

theorem households_with_appliances 
  (total : ℕ) 
  (tv : ℕ) 
  (fridge : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : tv = 65) 
  (h3 : fridge = 84) 
  (h4 : both = 53) : 
  tv + fridge - both = 96 := by
  sorry

end NUMINAMATH_CALUDE_households_with_appliances_l1201_120194


namespace NUMINAMATH_CALUDE_sample_size_is_twenty_l1201_120136

/-- Represents the total number of employees in the company -/
def total_employees : ℕ := 1000

/-- Represents the number of middle-aged workers in the company -/
def middle_aged_workers : ℕ := 350

/-- Represents the number of middle-aged workers in the sample -/
def sample_middle_aged : ℕ := 7

/-- Theorem stating that the sample size is 20 given the conditions -/
theorem sample_size_is_twenty :
  ∃ (sample_size : ℕ),
    (sample_middle_aged : ℚ) / sample_size = middle_aged_workers / total_employees ∧
    sample_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_twenty_l1201_120136


namespace NUMINAMATH_CALUDE_expression_evaluation_l1201_120173

theorem expression_evaluation : 3^4 - 4 * 3^3 + 6 * 3^2 - 4 * 3 + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1201_120173


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1201_120123

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h_f1 : f 1 = 1) 
  (h_f2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1201_120123


namespace NUMINAMATH_CALUDE_largest_value_between_2_and_3_l1201_120112

theorem largest_value_between_2_and_3 (x : ℝ) (h : 2 < x ∧ x < 3) :
  x^2 ≥ x ∧ x^2 ≥ 3*x ∧ x^2 ≥ Real.sqrt x ∧ x^2 ≥ 1/x :=
by sorry

end NUMINAMATH_CALUDE_largest_value_between_2_and_3_l1201_120112


namespace NUMINAMATH_CALUDE_male_to_female_ratio_l1201_120181

theorem male_to_female_ratio (total_students : ℕ) (female_students : ℕ) : 
  total_students = 52 → female_students = 13 → 
  (total_students - female_students : ℚ) / female_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_male_to_female_ratio_l1201_120181


namespace NUMINAMATH_CALUDE_jill_earnings_l1201_120107

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := tip_rate * total_orders
  wage_earnings + tip_earnings

/-- Theorem stating that Jill's earnings for the week are $240.00 --/
theorem jill_earnings :
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jill_earnings_l1201_120107


namespace NUMINAMATH_CALUDE_betty_height_l1201_120134

theorem betty_height (dog_height : ℕ) (carter_height : ℕ) (betty_height_inches : ℕ) :
  dog_height = 24 →
  carter_height = 2 * dog_height →
  betty_height_inches = carter_height - 12 →
  betty_height_inches / 12 = 3 :=
by sorry

end NUMINAMATH_CALUDE_betty_height_l1201_120134


namespace NUMINAMATH_CALUDE_increasing_condition_l1201_120193

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

-- State the theorem
theorem increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ↔ a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_condition_l1201_120193


namespace NUMINAMATH_CALUDE_trapezoid_height_l1201_120144

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  let height := a * b / (b - a)
  ∃ (x y : ℝ), 
    (x^2 + y^2 = a^2 + b^2) ∧ 
    ((b - a)^2 = x^2 + y^2 - x*y*Real.sqrt 2) ∧
    (x * y * Real.sqrt 2 = 2 * (b - a) * height) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l1201_120144


namespace NUMINAMATH_CALUDE_jimmys_father_emails_l1201_120161

/-- The number of emails Jimmy's father received per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails : ℕ := 5

/-- The total number of days in April -/
def total_days : ℕ := 30

/-- The day Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := 15

/-- The total number of emails Jimmy's father received in April -/
def total_emails : ℕ := 675

theorem jimmys_father_emails :
  initial_emails_per_day * subscription_day +
  (initial_emails_per_day + additional_emails) * (total_days - subscription_day) =
  total_emails :=
by
  sorry

#check jimmys_father_emails

end NUMINAMATH_CALUDE_jimmys_father_emails_l1201_120161


namespace NUMINAMATH_CALUDE_representable_integers_l1201_120163

/-- Represents an arithmetic expression using only the digit 2 and basic operations -/
inductive Expr
  | two : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.two => 2
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Counts the number of 2's used in an expression -/
def count_twos : Expr → ℕ
  | Expr.two => 1
  | Expr.add e1 e2 => count_twos e1 + count_twos e2
  | Expr.sub e1 e2 => count_twos e1 + count_twos e2
  | Expr.mul e1 e2 => count_twos e1 + count_twos e2
  | Expr.div e1 e2 => count_twos e1 + count_twos e2

theorem representable_integers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2019 →
  ∃ e : Expr, eval e = n ∧ count_twos e ≤ 17 :=
sorry

end NUMINAMATH_CALUDE_representable_integers_l1201_120163


namespace NUMINAMATH_CALUDE_west_3km_is_negative_3_l1201_120182

/-- Represents the direction of travel -/
inductive Direction
  | East
  | West

/-- Represents a distance traveled with direction -/
structure Travel where
  distance : ℝ
  direction : Direction

/-- Converts a Travel to a signed real number -/
def Travel.toSignedDistance (t : Travel) : ℝ :=
  match t.direction with
  | Direction.East => t.distance
  | Direction.West => -t.distance

theorem west_3km_is_negative_3 :
  let eastward_2km : Travel := ⟨2, Direction.East⟩
  let westward_3km : Travel := ⟨3, Direction.West⟩
  eastward_2km.toSignedDistance = 2 →
  westward_3km.toSignedDistance = -3 :=
by sorry

end NUMINAMATH_CALUDE_west_3km_is_negative_3_l1201_120182


namespace NUMINAMATH_CALUDE_book_price_decrease_l1201_120147

theorem book_price_decrease (P : ℝ) (x : ℝ) : 
  P - ((P - (x / 100) * P) * 1.2) = 10.000000000000014 → 
  x = 50 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_price_decrease_l1201_120147


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l1201_120174

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 / 8) * Real.sqrt 3 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l1201_120174


namespace NUMINAMATH_CALUDE_xyz_value_l1201_120154

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1201_120154


namespace NUMINAMATH_CALUDE_willies_cream_calculation_l1201_120170

/-- The amount of whipped cream Willie needs in total (in lbs) -/
def total_cream : ℕ := 300

/-- The amount of cream Willie needs to buy (in lbs) -/
def cream_to_buy : ℕ := 151

/-- The amount of cream Willie got from his farm (in lbs) -/
def cream_from_farm : ℕ := total_cream - cream_to_buy

theorem willies_cream_calculation :
  cream_from_farm = 149 := by
  sorry

end NUMINAMATH_CALUDE_willies_cream_calculation_l1201_120170


namespace NUMINAMATH_CALUDE_quadruple_batch_cans_l1201_120165

/-- Represents the number of cans for each ingredient in a normal batch of chili --/
structure NormalBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Defines a normal batch of chili according to Carla's recipe --/
def carla_normal_batch : NormalBatch where
  chilis := 1
  beans := 2
  tomatoes := 3  -- 50% more than beans, so 2 * 1.5 = 3

/-- Calculates the total number of cans for a given batch size --/
def total_cans (batch : NormalBatch) (multiplier : ℕ) : ℕ :=
  multiplier * (batch.chilis + batch.beans + batch.tomatoes)

/-- Theorem: A quadruple batch of Carla's chili requires 24 cans of food --/
theorem quadruple_batch_cans : 
  total_cans carla_normal_batch 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadruple_batch_cans_l1201_120165


namespace NUMINAMATH_CALUDE_cello_practice_time_l1201_120177

/-- Calculates the remaining practice time in minutes for a cellist -/
theorem cello_practice_time (total_hours : ℝ) (daily_minutes : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ daily_minutes = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (daily_minutes * practice_days : ℕ) = 278 := by
  sorry

#check cello_practice_time

end NUMINAMATH_CALUDE_cello_practice_time_l1201_120177


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l1201_120166

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

theorem intersection_complement_M_and_N : 
  (Set.univ \ M) ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l1201_120166


namespace NUMINAMATH_CALUDE_like_terms_mn_value_l1201_120106

theorem like_terms_mn_value (n m : ℕ) :
  (∃ (a b : ℝ) (x y : ℝ), a * x^n * y^3 = b * x^3 * y^m) →
  m^n = 27 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_mn_value_l1201_120106


namespace NUMINAMATH_CALUDE_max_value_interval_condition_l1201_120125

/-- The function f(x) = (1/3)x^3 - x has a maximum value on the interval (2m, 1-m) if and only if m ∈ [-1, -1/2). -/
theorem max_value_interval_condition (m : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Ioo (2*m) (1-m) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (2*m) (1-m) → 
      (1/3 * x^3 - x) ≥ (1/3 * y^3 - y))) ↔ 
  m ∈ Set.Icc (-1) (-1/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_interval_condition_l1201_120125


namespace NUMINAMATH_CALUDE_parallelogram_area_l1201_120117

/-- Proves that a parallelogram with base 16 cm and where 2 times the sum of its base and height is 56, has an area of 192 square centimeters. -/
theorem parallelogram_area (b h : ℝ) : 
  b = 16 → 2 * (b + h) = 56 → b * h = 192 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1201_120117


namespace NUMINAMATH_CALUDE_speed_calculation_l1201_120103

-- Define the distance in meters
def distance_meters : ℝ := 375.03

-- Define the time in seconds
def time_seconds : ℝ := 25

-- Define the conversion factor from m/s to km/h
def mps_to_kmph : ℝ := 3.6

-- Theorem to prove
theorem speed_calculation :
  let speed_mps := distance_meters / time_seconds
  let speed_kmph := speed_mps * mps_to_kmph
  ∃ ε > 0, |speed_kmph - 54.009| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l1201_120103


namespace NUMINAMATH_CALUDE_solve_equation_l1201_120138

theorem solve_equation (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1201_120138


namespace NUMINAMATH_CALUDE_total_balloons_l1201_120142

/-- Represents the number of balloons of each color -/
structure BalloonCounts where
  gold : ℕ
  silver : ℕ
  black : ℕ
  blue : ℕ
  red : ℕ

/-- The conditions of the balloon problem -/
def balloon_problem (b : BalloonCounts) : Prop :=
  b.gold = 141 ∧
  b.silver = 2 * b.gold ∧
  b.black = 150 ∧
  b.blue = b.silver / 2 ∧
  b.red = b.blue / 3

/-- The theorem stating the total number of balloons -/
theorem total_balloons (b : BalloonCounts) 
  (h : balloon_problem b) : 
  b.gold + b.silver + b.black + b.blue + b.red = 761 := by
  sorry

#check total_balloons

end NUMINAMATH_CALUDE_total_balloons_l1201_120142


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1201_120140

-- Define a right-angled triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  h_right_angle : angleA + angleB = π / 2
  h_sides : c^2 = a^2 + b^2
  h_not_equal : a ≠ b

-- State the theorem
theorem right_triangle_properties (t : RightTriangle) :
  (Real.tan t.angleA * Real.tan t.angleB ≠ 1) ∧
  (Real.sin t.angleA = t.a / t.c) ∧
  (t.c^2 - t.a^2 = t.b^2) ∧
  (t.c = t.b / Real.cos t.angleA) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l1201_120140


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1201_120105

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_roots : a 3 + a 15 = 6 ∧ a 3 * a 15 = 8) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1201_120105


namespace NUMINAMATH_CALUDE_square_area_doubling_l1201_120111

theorem square_area_doubling (s : ℝ) (h : s > 0) :
  (2 * s) ^ 2 = 4 * s ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_doubling_l1201_120111


namespace NUMINAMATH_CALUDE_apples_theorem_l1201_120168

def apples_problem (initial_apples : ℕ) (ricki_removes : ℕ) (days : ℕ) : Prop :=
  let samson_removes := 2 * ricki_removes
  let bindi_removes := 3 * samson_removes
  let total_daily_removal := ricki_removes + samson_removes + bindi_removes
  let total_weekly_removal := total_daily_removal * days
  total_weekly_removal = initial_apples + 2150

theorem apples_theorem : apples_problem 1000 50 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_theorem_l1201_120168


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l1201_120104

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 + 2*x - m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 2^2 - 4*(1)*(-m)

-- Theorem statement
theorem no_real_roots_condition (m : ℝ) :
  (∀ x, quadratic_equation x m ≠ 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l1201_120104


namespace NUMINAMATH_CALUDE_jose_pool_charge_ratio_l1201_120101

/-- Represents the daily revenue from Jose's swimming pool --/
def daily_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  8 * kids_charge + 10 * adults_charge

/-- Represents the weekly revenue from Jose's swimming pool --/
def weekly_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  7 * daily_revenue kids_charge adults_charge

/-- Theorem stating the ratio of adult to kid charge in Jose's swimming pool --/
theorem jose_pool_charge_ratio :
  ∃ (adults_charge : ℚ),
    weekly_revenue 3 adults_charge = 588 ∧
    adults_charge / 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_jose_pool_charge_ratio_l1201_120101


namespace NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l1201_120146

theorem quadratic_equality_implies_coefficient (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l1201_120146


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l1201_120116

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l1201_120116


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1201_120197

theorem complex_fraction_equals_i : (1 + Complex.I * Real.sqrt 3) / (Real.sqrt 3 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1201_120197


namespace NUMINAMATH_CALUDE_non_constant_polynomial_not_always_palindrome_l1201_120120

/-- A number is a palindrome if it reads the same from left to right as it reads from right to left in base 10 -/
def is_palindrome (n : ℤ) : Prop := sorry

/-- The theorem states that for any non-constant polynomial with integer coefficients, 
    there exists a positive integer n such that p(n) is not a palindrome number -/
theorem non_constant_polynomial_not_always_palindrome 
  (p : Polynomial ℤ) (h : ¬ (p.degree = 0)) : 
  ∃ (n : ℕ), ¬ is_palindrome (p.eval n) := by sorry

end NUMINAMATH_CALUDE_non_constant_polynomial_not_always_palindrome_l1201_120120
