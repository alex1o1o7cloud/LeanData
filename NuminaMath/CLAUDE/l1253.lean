import Mathlib

namespace NUMINAMATH_CALUDE_same_terminal_side_l1253_125352

theorem same_terminal_side : ∀ θ : Real,
  θ ≥ 0 ∧ θ < 2 * Real.pi →
  (θ = 2 * Real.pi / 3) ↔ ∃ k : Int, θ = -4 * Real.pi / 3 + 2 * Real.pi * k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1253_125352


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1253_125320

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I)
  (h2 : Complex.abs z = Real.sqrt 2)
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1253_125320


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l1253_125369

theorem sum_of_odd_numbers (n : ℕ) (sum_of_first_n_odds : ℕ → ℕ) 
  (h1 : ∀ k, sum_of_first_n_odds k = k^2)
  (h2 : sum_of_first_n_odds 100 = 10000)
  (h3 : sum_of_first_n_odds 50 = 2500) :
  sum_of_first_n_odds 100 - sum_of_first_n_odds 50 = 7500 := by
  sorry

#check sum_of_odd_numbers

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l1253_125369


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1253_125354

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1253_125354


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1253_125321

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 1846 [ZMOD 26] → n ≤ m) ∧ 
  (5 * n ≡ 1846 [ZMOD 26]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1253_125321


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l1253_125377

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |5 - x| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l1253_125377


namespace NUMINAMATH_CALUDE_linear_function_properties_l1253_125329

/-- A linear function of the form y = mx + 4m - 2 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m - 2

theorem linear_function_properties :
  ∃ m : ℝ, 
    (∃ y : ℝ, y ≠ -2 ∧ linear_function m 0 = y) ∧ 
    (let f := linear_function (1/3);
     ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ f x₁ > 0 ∧ x₂ < 0 ∧ f x₂ < 0 ∧ x₃ > 0 ∧ f x₃ < 0) ∧
    (linear_function (1/2) 0 = 0) ∧
    (∀ m : ℝ, linear_function m (-4) = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1253_125329


namespace NUMINAMATH_CALUDE_line_equation_proof_l1253_125370

/-- 
Given a line mx + (n/2)y - 1 = 0 with a y-intercept of -1 and an angle of inclination 
twice that of the line √3x - y - 3√3 = 0, prove that m = -√3 and n = -2.
-/
theorem line_equation_proof (m n : ℝ) : 
  (∀ x y, m * x + (n / 2) * y - 1 = 0) →  -- Line equation
  (0 + (n / 2) * (-1) - 1 = 0) →  -- y-intercept is -1
  (Real.arctan m = 2 * Real.arctan (Real.sqrt 3)) →  -- Angle of inclination relation
  (m = -Real.sqrt 3 ∧ n = -2) := by
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1253_125370


namespace NUMINAMATH_CALUDE_right_triangle_sides_exist_l1253_125351

/-- A right triangle with perimeter k and incircle radius ρ --/
structure RightTriangle (k ρ : ℝ) where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = k
  incircle_eq : a * b = 2 * ρ * (k / 2)
  pythagorean : a^2 + b^2 = c^2

/-- The side lengths of a right triangle satisfy the given conditions --/
theorem right_triangle_sides_exist (k ρ : ℝ) (hk : k > 0) (hρ : ρ > 0) :
  ∃ (t : RightTriangle k ρ), t.a > 0 ∧ t.b > 0 ∧ t.c > 0 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_exist_l1253_125351


namespace NUMINAMATH_CALUDE_equation_real_roots_a_range_l1253_125307

theorem equation_real_roots_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 2 - 2^(-|x-2|) = 2 + a) → -1 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_real_roots_a_range_l1253_125307


namespace NUMINAMATH_CALUDE_age_problem_l1253_125391

theorem age_problem (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 40) : x = 10 ∧ y = 30 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1253_125391


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1253_125367

theorem inequality_solution_set :
  {x : ℝ | (4 : ℝ) / (x + 1) ≤ 1} = Set.Iic (-1) ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1253_125367


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1253_125363

theorem smallest_lcm_with_gcd_5 (p q : ℕ) : 
  1000 ≤ p ∧ p < 10000 ∧ 
  1000 ≤ q ∧ q < 10000 ∧ 
  Nat.gcd p q = 5 →
  201000 ≤ Nat.lcm p q ∧ 
  ∃ (p' q' : ℕ), 1000 ≤ p' ∧ p' < 10000 ∧ 
                 1000 ≤ q' ∧ q' < 10000 ∧ 
                 Nat.gcd p' q' = 5 ∧
                 Nat.lcm p' q' = 201000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1253_125363


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1253_125309

/-- The lateral area of a cylinder with base diameter and height both equal to 4 cm is 16π cm². -/
theorem cylinder_lateral_area (π : ℝ) (h : π > 0) : 
  let d : ℝ := 4 -- diameter
  let h : ℝ := 4 -- height
  let r : ℝ := d / 2 -- radius
  let lateral_area : ℝ := 2 * π * r * h
  lateral_area = 16 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1253_125309


namespace NUMINAMATH_CALUDE_tan_theta_value_l1253_125319

theorem tan_theta_value (θ : Real) (a : Real) 
  (h1 : (4, a) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ})
  (h2 : Real.sin (θ - π) = 3/5) : 
  Real.tan θ = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1253_125319


namespace NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1253_125334

theorem sum_nonnegative_implies_one_nonnegative (a b : ℝ) : 
  a + b ≥ 0 → (a ≥ 0 ∨ b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_nonnegative_implies_one_nonnegative_l1253_125334


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1253_125304

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1253_125304


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1253_125333

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Represents a number in base 5 -/
structure Base5 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 5

theorem base5_division_theorem (a b : Base5) :
  let a_base10 := base5ToBase10 a.digits
  let b_base10 := base5ToBase10 b.digits
  let quotient_base10 := a_base10 / b_base10
  let quotient_base5 := Base5.mk (base10ToBase5 quotient_base10) sorry
  a = Base5.mk [1, 3, 2, 4] sorry ∧ 
  b = Base5.mk [1, 2] sorry → 
  quotient_base5 = Base5.mk [1, 1, 0] sorry := by
  sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1253_125333


namespace NUMINAMATH_CALUDE_count_satisfying_integers_is_five_l1253_125362

/-- The count of positive integers n satisfying (n + 1050) / 90 = ⌊√n⌋ -/
def count_satisfying_integers : ℕ := 5

/-- Predicate defining when a positive integer satisfies the equation -/
def satisfies_equation (n : ℕ+) : Prop :=
  (n + 1050) / 90 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly 5 positive integers satisfy the equation -/
theorem count_satisfying_integers_is_five :
  (∃! (S : Finset ℕ+), S.card = count_satisfying_integers ∧ 
    ∀ n, n ∈ S ↔ satisfies_equation n) :=
by sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_is_five_l1253_125362


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1253_125357

theorem triangle_equilateral (a b c : ℝ) 
  (triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h : a^2 + b^2 + c^2 = a*b + b*c + c*a) : 
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l1253_125357


namespace NUMINAMATH_CALUDE_smallest_integer_square_75_more_than_double_l1253_125371

theorem smallest_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_75_more_than_double_l1253_125371


namespace NUMINAMATH_CALUDE_average_speed_to_destination_l1253_125324

/-- Proves that given a round trip with a total one-way distance of 150 km,
    a return speed of 30 km/hr, and an average speed for the whole journey of 37.5 km/hr,
    the average speed while traveling to the place is 50 km/hr. -/
theorem average_speed_to_destination (v : ℝ) : 
  (150 : ℝ) / v + (150 : ℝ) / 30 = 300 / (37.5 : ℝ) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_to_destination_l1253_125324


namespace NUMINAMATH_CALUDE_find_M_l1253_125348

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2024 + 2025 + 2026) / M ∧ M = 552 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1253_125348


namespace NUMINAMATH_CALUDE_expression_equals_159_l1253_125341

def numerator : List ℕ := [12, 24, 36, 48, 60]
def denominator : List ℕ := [6, 18, 30, 42, 54]

def term (x : ℕ) : ℕ := x^4 + 375

def expression : ℚ :=
  (numerator.map term).prod / (denominator.map term).prod

theorem expression_equals_159 : expression = 159 := by sorry

end NUMINAMATH_CALUDE_expression_equals_159_l1253_125341


namespace NUMINAMATH_CALUDE_sideEdgeLength_of_rightTriangularPyramid_l1253_125339

/-- A right triangular pyramid with three mutually perpendicular side edges of equal length -/
structure RightTriangularPyramid where
  sideEdgeLength : ℝ
  mutuallyPerpendicular : Bool
  equalLength : Bool

/-- The circumscribed sphere of a RightTriangularPyramid -/
def circumscribedSphere (pyramid : RightTriangularPyramid) : ℝ → Prop :=
  fun surfaceArea => surfaceArea = 4 * Real.pi

/-- Theorem: The length of a side edge of a right triangular pyramid is 2√3/3 -/
theorem sideEdgeLength_of_rightTriangularPyramid (pyramid : RightTriangularPyramid) 
  (h1 : pyramid.mutuallyPerpendicular = true)
  (h2 : pyramid.equalLength = true)
  (h3 : circumscribedSphere pyramid (4 * Real.pi)) :
  pyramid.sideEdgeLength = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sideEdgeLength_of_rightTriangularPyramid_l1253_125339


namespace NUMINAMATH_CALUDE_total_orchestra_members_l1253_125368

/-- Represents the number of boys in the orchestra -/
def boys : ℕ := sorry

/-- Represents the number of girls in the orchestra -/
def girls : ℕ := sorry

/-- The number of girls is twice the number of boys -/
axiom girls_twice_boys : girls = 2 * boys

/-- If 24 girls are transferred, the number of boys will be twice the number of girls -/
axiom boys_twice_remaining_girls : boys = 2 * (girls - 24)

/-- The total number of boys and girls in the orchestra is 48 -/
theorem total_orchestra_members : boys + girls = 48 := by sorry

end NUMINAMATH_CALUDE_total_orchestra_members_l1253_125368


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l1253_125318

theorem units_digit_of_fraction : ∃ n : ℕ, n % 10 = 4 ∧ (30 * 31 * 32 * 33 * 34) / 400 = n := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l1253_125318


namespace NUMINAMATH_CALUDE_initial_sales_tax_percentage_l1253_125392

/-- Proves that the initial sales tax percentage is 3.5% given the conditions -/
theorem initial_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (tax_difference : ℝ) 
  (h1 : market_price = 7800)
  (h2 : new_tax_rate = 10 / 3)
  (h3 : tax_difference = 13) :
  ∃ (x : ℝ), x = 3.5 ∧ market_price * (x / 100 - new_tax_rate / 100) = tax_difference :=
sorry

end NUMINAMATH_CALUDE_initial_sales_tax_percentage_l1253_125392


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1253_125360

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > -1) (hab : a + b = 1) :
  1 / a + 1 / (b + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1253_125360


namespace NUMINAMATH_CALUDE_box_third_dimension_l1253_125300

/-- Proves that the third dimension of a rectangular box is 6 cm, given specific conditions -/
theorem box_third_dimension (num_cubes : ℕ) (cube_volume : ℝ) (length width : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 9 →
  width = 12 →
  (num_cubes : ℝ) * cube_volume = length * width * 6 :=
by sorry

end NUMINAMATH_CALUDE_box_third_dimension_l1253_125300


namespace NUMINAMATH_CALUDE_faculty_size_l1253_125316

/-- The number of second year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The percentage of second year students in the total student body -/
def second_year_percentage : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 674

theorem faculty_size : 
  ∃ (second_year_students : ℕ), 
    second_year_students = numeric_methods + automatic_control - both_subjects ∧
    (second_year_students : ℚ) / total_students = second_year_percentage :=
by sorry

end NUMINAMATH_CALUDE_faculty_size_l1253_125316


namespace NUMINAMATH_CALUDE_total_books_l1253_125322

def books_per_shelf : ℕ := 15
def mystery_shelves : ℕ := 8
def picture_shelves : ℕ := 4
def biography_shelves : ℕ := 3
def scifi_shelves : ℕ := 5

theorem total_books : 
  books_per_shelf * (mystery_shelves + picture_shelves + biography_shelves + scifi_shelves) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1253_125322


namespace NUMINAMATH_CALUDE_rex_cards_left_is_150_l1253_125373

/-- The number of Pokemon cards Rex has left after dividing his cards among himself and his siblings -/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := 2 * nicolesCards
  let combinedTotal := nicolesCards + cindysCards
  let rexCards := combinedTotal / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left given the initial conditions -/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_is_150_l1253_125373


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1253_125335

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the focal points
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem hyperbola_intersection_theorem 
  (k m : ℝ) 
  (A B : ℝ × ℝ) 
  (h_focal_length : Real.sqrt 16 = 4)
  (h_imaginary_axis : Real.sqrt 4 = 2)
  (h_m_nonzero : m ≠ 0)
  (h_distinct : A ≠ B)
  (h_on_hyperbola_A : hyperbola A.1 A.2)
  (h_on_hyperbola_B : hyperbola B.1 B.2)
  (h_on_line_A : line k m A.1 A.2)
  (h_on_line_B : line k m B.1 B.2)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3)
  (h_passes_F₂ : line k m F₂.1 F₂.2) :
  -- 1. Eccentricity
  (2 * Real.sqrt 3 / 3 = Real.sqrt (1 - 1 / 3)) ∧
  -- 2. Equation of line l
  ((k = 1 ∧ m = -2) ∨ (k = -1 ∧ m = 2) ∨ (k = 0 ∧ m ≠ 0)) ∧
  -- 3. Range of m
  (k ≠ 0 → m ∈ Set.Icc (-1/4) 0 ∪ Set.Ioi 4) ∧
  (k = 0 → m ∈ Set.univ \ {0}) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1253_125335


namespace NUMINAMATH_CALUDE_last_bead_color_l1253_125330

def bead_colors := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def necklace_length : Nat := 85

theorem last_bead_color (h : necklace_length = 85) :
  bead_colors[(necklace_length - 1) % bead_colors.length] = "yellow" := by
  sorry

end NUMINAMATH_CALUDE_last_bead_color_l1253_125330


namespace NUMINAMATH_CALUDE_triangle_side_b_l1253_125325

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.B = π / 6 → t.a = Real.sqrt 3 → t.c = 1 → t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l1253_125325


namespace NUMINAMATH_CALUDE_largest_number_l1253_125305

theorem largest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = -2) (hd : d = 1) :
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1253_125305


namespace NUMINAMATH_CALUDE_school_athletes_equation_l1253_125359

/-- 
Given a school with x athletes divided into y groups, prove that the following system of equations holds:
7y = x - 3
8y = x + 5
-/
theorem school_athletes_equation (x y : ℕ) 
  (h1 : 7 * y = x - 3)  -- If there are 7 people in each group, there will be 3 people left over
  (h2 : 8 * y = x + 5)  -- If there are 8 people in each group, there will be a shortage of 5 people
  : 7 * y = x - 3 ∧ 8 * y = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_school_athletes_equation_l1253_125359


namespace NUMINAMATH_CALUDE_survey_respondents_l1253_125347

/-- Represents the number of people preferring each brand in a survey. -/
structure SurveyPreferences where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (prefs : SurveyPreferences) : ℕ :=
  prefs.x + prefs.y + prefs.z

/-- Theorem stating the total number of respondents in the survey. -/
theorem survey_respondents : ∃ (prefs : SurveyPreferences), 
  prefs.x = 360 ∧ 
  prefs.x * 4 = prefs.y * 9 ∧ 
  prefs.x * 3 = prefs.z * 9 ∧ 
  totalRespondents prefs = 640 := by
  sorry


end NUMINAMATH_CALUDE_survey_respondents_l1253_125347


namespace NUMINAMATH_CALUDE_solve_for_b_l1253_125393

theorem solve_for_b (b : ℝ) : 
  4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → b = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1253_125393


namespace NUMINAMATH_CALUDE_solution_set_equality_l1253_125342

open Set

def S : Set ℝ := {x | |x + 1| + |x - 4| ≥ 7}

theorem solution_set_equality : S = Iic (-2) ∪ Ici 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1253_125342


namespace NUMINAMATH_CALUDE_equation_solutions_l1253_125346

theorem equation_solutions :
  (∀ x : ℚ, (1/2 * x - 2 = 4 + 1/3 * x) ↔ (x = 36)) ∧
  (∀ x : ℚ, ((x - 1) / 4 - 2 = (2 * x - 3) / 6) ↔ (x = -21)) ∧
  (∀ x : ℚ, (1/3 * (x - 1/2 * (x - 1)) = 2/3 * (x - 1/2)) ↔ (x = 1)) ∧
  (∀ x : ℚ, (x / (7/10) - (17/100 - 1/5 * x) / (3/100) = 1) ↔ (x = 14/17)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1253_125346


namespace NUMINAMATH_CALUDE_fraction_zero_iff_x_zero_l1253_125372

theorem fraction_zero_iff_x_zero (x : ℝ) (h : x ≠ -2) :
  2 * x / (x + 2) = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_fraction_zero_iff_x_zero_l1253_125372


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l1253_125301

theorem polygon_angle_sum (n : ℕ) : (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l1253_125301


namespace NUMINAMATH_CALUDE_cans_recycled_from_64_l1253_125376

def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 4 then 0
  else (initial_cans / 4) + recycle_cans (initial_cans / 4)

theorem cans_recycled_from_64 :
  recycle_cans 64 = 21 :=
sorry

end NUMINAMATH_CALUDE_cans_recycled_from_64_l1253_125376


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1253_125343

def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x ≥ 1, monotonically_increasing (f a) 1 x) →
  (a ≤ -2 ∧ ∃ b, b ≤ 0 ∧ b > -2 ∧ ∀ x ≥ 1, monotonically_increasing (f b) 1 x) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1253_125343


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1253_125396

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1253_125396


namespace NUMINAMATH_CALUDE_total_players_l1253_125381

theorem total_players (outdoor : ℕ) (indoor : ℕ) (both : ℕ)
  (h1 : outdoor = 350)
  (h2 : indoor = 110)
  (h3 : both = 60) :
  outdoor + indoor - both = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l1253_125381


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l1253_125382

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies that can be kept --/
def max_kept_candies (cd : CandyDistribution) : ℕ := 
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- The theorem stating the maximum number of candies that can be kept --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l1253_125382


namespace NUMINAMATH_CALUDE_number_of_friends_l1253_125365

/-- Given that Mary, Sam, Keith, and Alyssa each have 6 baseball cards,
    prove that the number of friends is 4. -/
theorem number_of_friends : ℕ :=
  let mary_cards := 6
  let sam_cards := 6
  let keith_cards := 6
  let alyssa_cards := 6
  4

#check number_of_friends

end NUMINAMATH_CALUDE_number_of_friends_l1253_125365


namespace NUMINAMATH_CALUDE_sales_tax_difference_l1253_125337

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) 
  (h1 : price = 50)
  (h2 : tax_rate1 = 0.0725)
  (h3 : tax_rate2 = 0.0675) : 
  (tax_rate1 - tax_rate2) * price = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l1253_125337


namespace NUMINAMATH_CALUDE_range_of_m_l1253_125358

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∃ m : ℝ, m < -1 ∨ m > 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1253_125358


namespace NUMINAMATH_CALUDE_badminton_players_count_l1253_125383

theorem badminton_players_count (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_tennis : tennis = 18)
  (h_neither : neither = 5)
  (h_both : both = 3) :
  total = tennis + (total - tennis - neither) - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_count_l1253_125383


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1253_125397

/-- Given a car that uses 6.5 gallons of gasoline to travel 130 kilometers,
    prove that its fuel efficiency is 20 kilometers per gallon. -/
theorem car_fuel_efficiency :
  ∀ (distance : ℝ) (fuel : ℝ),
    distance = 130 →
    fuel = 6.5 →
    distance / fuel = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1253_125397


namespace NUMINAMATH_CALUDE_nina_savings_time_l1253_125395

theorem nina_savings_time (video_game_cost : ℝ) (headset_cost : ℝ) (sales_tax_rate : ℝ) 
  (weekly_allowance : ℝ) (savings_rate : ℝ) :
  video_game_cost = 50 →
  headset_cost = 70 →
  sales_tax_rate = 0.12 →
  weekly_allowance = 10 →
  savings_rate = 0.40 →
  ⌈(((video_game_cost + headset_cost) * (1 + sales_tax_rate)) / 
    (weekly_allowance * savings_rate))⌉ = 34 := by
  sorry

end NUMINAMATH_CALUDE_nina_savings_time_l1253_125395


namespace NUMINAMATH_CALUDE_penumbra_ring_area_l1253_125313

/-- The area of a ring formed between two concentric circles --/
theorem penumbra_ring_area (r_umbra : ℝ) (r_penumbra : ℝ) (h1 : r_umbra = 40) (h2 : r_penumbra = 3 * r_umbra) :
  let a_ring := π * r_penumbra^2 - π * r_umbra^2
  a_ring = 12800 * π := by sorry

end NUMINAMATH_CALUDE_penumbra_ring_area_l1253_125313


namespace NUMINAMATH_CALUDE_inequalities_hold_l1253_125345

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a*b + b*c + c*a) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^2*y^2*z^2 < a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l1253_125345


namespace NUMINAMATH_CALUDE_pages_per_chapter_book_chapters_calculation_l1253_125353

theorem pages_per_chapter 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) : Nat :=
  let total_chapters_read := days_to_finish * chapters_per_day
  total_chapters_read / total_chapters

theorem book_chapters_calculation 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) :
  total_chapters = 2 →
  days_to_finish = 664 →
  chapters_per_day = 332 →
  pages_per_chapter total_chapters days_to_finish chapters_per_day = 110224 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_book_chapters_calculation_l1253_125353


namespace NUMINAMATH_CALUDE_correct_calculation_l1253_125379

theorem correct_calculation (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1253_125379


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l1253_125384

theorem complex_root_magnitude (z : ℂ) : z^2 + z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l1253_125384


namespace NUMINAMATH_CALUDE_min_colors_regular_ngon_l1253_125306

/-- 
Represents a coloring of sides and diagonals in a regular n-gon.
The coloring is valid if any two segments sharing a common point have different colors.
-/
def ValidColoring (n : ℕ) := 
  { coloring : (Fin n × Fin n) → ℕ // 
    ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → 
    coloring (i, j) ≠ coloring (i, k) ∧ 
    coloring (i, j) ≠ coloring (j, k) ∧ 
    coloring (i, k) ≠ coloring (j, k) }

/-- 
The minimum number of colors needed for a valid coloring of a regular n-gon 
is equal to n.
-/
theorem min_colors_regular_ngon (n : ℕ) (h : n ≥ 3) : 
  (∃ (c : ValidColoring n), ∀ (i j : Fin n), c.val (i, j) < n) ∧ 
  (∀ (c : ValidColoring n) (m : ℕ), (∀ (i j : Fin n), c.val (i, j) < m) → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_colors_regular_ngon_l1253_125306


namespace NUMINAMATH_CALUDE_complex_modulus_l1253_125344

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1253_125344


namespace NUMINAMATH_CALUDE_tetrahedron_volume_not_determined_by_face_areas_l1253_125350

/-- A tetrahedron with four faces --/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas --/
theorem tetrahedron_volume_not_determined_by_face_areas :
  ∃ (t1 t2 : Tetrahedron), (∀ i : Fin 4, t1.faces i = t2.faces i) ∧ t1.volume ≠ t2.volume :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_not_determined_by_face_areas_l1253_125350


namespace NUMINAMATH_CALUDE_target_hit_probability_l1253_125356

theorem target_hit_probability (total_groups : ℕ) (hit_groups : ℕ) : 
  total_groups = 20 → hit_groups = 5 → 
  (hit_groups : ℚ) / total_groups = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1253_125356


namespace NUMINAMATH_CALUDE_line_properties_l1253_125388

-- Define the line
def line_equation (x : ℝ) : ℝ := -4 * x - 12

-- Theorem statement
theorem line_properties :
  (∀ x, line_equation x = -4 * x - 12) →
  (line_equation (-3) = 0) →
  (line_equation 0 = -12) ∧
  (line_equation 2 = -20) := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l1253_125388


namespace NUMINAMATH_CALUDE_f_properties_l1253_125314

/-- The function f(x) = x^3 - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- Theorem stating the range of a and the fixed point property -/
theorem f_properties (a : ℝ) (x₀ : ℝ) 
  (ha : a > 0)
  (hf : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y)
  (hx₀ : x₀ ≥ 1)
  (hfx₀ : f a x₀ ≥ 1)
  (hffx₀ : f a (f a x₀) = x₀) :
  (0 < a ∧ a ≤ 3) ∧ f a x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1253_125314


namespace NUMINAMATH_CALUDE_square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l1253_125326

theorem square_of_arithmetic_mean_le_arithmetic_mean_of_squares
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b) / 2) ^ 2 ≤ (a ^ 2 + b ^ 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l1253_125326


namespace NUMINAMATH_CALUDE_number_multiplication_l1253_125394

theorem number_multiplication (x : ℤ) : 50 = x + 26 → 9 * x = 216 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l1253_125394


namespace NUMINAMATH_CALUDE_apples_for_pies_l1253_125340

/-- Calculates the number of apples needed to make a given number of pies. -/
def apples_needed (apples_per_pie : ℝ) (num_pies : ℕ) : ℝ :=
  apples_per_pie * (num_pies : ℝ)

/-- Theorem stating that 504 apples are needed to make 126 pies,
    given that it takes 4.0 apples to make 1.0 pie. -/
theorem apples_for_pies :
  apples_needed 4.0 126 = 504 := by
  sorry

end NUMINAMATH_CALUDE_apples_for_pies_l1253_125340


namespace NUMINAMATH_CALUDE_elizabeth_stickers_l1253_125323

/-- Calculates the total number of stickers used on water bottles -/
def total_stickers (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (stickers_per_bottle : ℕ) : ℕ :=
  (initial_bottles - lost_bottles - stolen_bottles) * stickers_per_bottle

/-- Theorem: Given Elizabeth's specific situation, she uses 21 stickers in total -/
theorem elizabeth_stickers : 
  total_stickers 10 2 1 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_stickers_l1253_125323


namespace NUMINAMATH_CALUDE_unique_solution_l1253_125331

/-- Represents a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ → ℕ
| 0 => n % base
| k + 1 => baseRepresentation (n / base) base k

/-- The equation to be solved --/
def equationHolds (x : ℕ) : Prop :=
  baseRepresentation 2016 x 3 * x^3 +
  baseRepresentation 2016 x 2 * x^2 +
  baseRepresentation 2016 x 1 * x +
  baseRepresentation 2016 x 0 = x^3 + 2*x + 342

theorem unique_solution :
  ∃! x : ℕ, x > 0 ∧ equationHolds x ∧ x = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1253_125331


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l1253_125349

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => 
  p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 21 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m < 21, g m ≠ 4) ∧ g 21 = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l1253_125349


namespace NUMINAMATH_CALUDE_pi_fourth_in_range_of_g_l1253_125328

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_g : ∃ (x : ℝ), g x = π / 4 := by sorry

end NUMINAMATH_CALUDE_pi_fourth_in_range_of_g_l1253_125328


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l1253_125386

/-- Given an initial angle of 60 degrees that is rotated 520 degrees clockwise,
    the resulting acute angle is 100 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 520 →
  (initial_angle + rotation) % 360 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l1253_125386


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1253_125308

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1253_125308


namespace NUMINAMATH_CALUDE_final_result_proof_l1253_125361

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 2976) :
  (chosen_number / 12) - 240 = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l1253_125361


namespace NUMINAMATH_CALUDE_population_doubling_time_l1253_125375

/-- The number of years required for a population to double given birth and death rates -/
theorem population_doubling_time (birth_rate death_rate : ℚ) : 
  birth_rate = 39.4 ∧ death_rate = 19.4 → 
  (70 : ℚ) / ((birth_rate - death_rate) / 10) = 35 := by
  sorry

end NUMINAMATH_CALUDE_population_doubling_time_l1253_125375


namespace NUMINAMATH_CALUDE_function_property_l1253_125398

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∃ a : ℝ, ∀ x : ℝ, f (1 + x) = a + f x)
  (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  ∃ a : ℝ, (∀ x : ℝ, f (1 + x) = a + f x) ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1253_125398


namespace NUMINAMATH_CALUDE_a_8_equals_3_l1253_125311

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmeticSequence (b : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d

theorem a_8_equals_3
  (a b : Sequence)
  (h1 : a 1 = 3)
  (h2 : IsArithmeticSequence b)
  (h3 : ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_3_l1253_125311


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1253_125310

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1253_125310


namespace NUMINAMATH_CALUDE_composition_of_odd_functions_l1253_125355

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem composition_of_odd_functions (f : ℝ → ℝ) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by sorry

end NUMINAMATH_CALUDE_composition_of_odd_functions_l1253_125355


namespace NUMINAMATH_CALUDE_final_week_study_hours_l1253_125399

def study_hours : List ℕ := [8, 10, 9, 11, 10, 7]
def total_weeks : ℕ := 7
def required_average : ℕ := 9

theorem final_week_study_hours :
  ∃ (x : ℕ), 
    (List.sum study_hours + x) / total_weeks = required_average ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_week_study_hours_l1253_125399


namespace NUMINAMATH_CALUDE_inequality_solution_l1253_125385

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1/2 ↔ x ∈ Set.Ioc (-8) (-2) ∪ Set.Icc 6 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1253_125385


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_tenth_squared_l1253_125338

theorem decimal_equivalent_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_tenth_squared_l1253_125338


namespace NUMINAMATH_CALUDE_power_product_equality_l1253_125317

theorem power_product_equality (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1253_125317


namespace NUMINAMATH_CALUDE_secret_spread_day_secret_spread_saturday_unique_day_for_3280_l1253_125389

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2 + 1

theorem secret_spread_day : ∃ (d : ℕ), secret_spread d = 3280 :=
  sorry

theorem secret_spread_saturday : secret_spread 7 = 3280 :=
  sorry

theorem unique_day_for_3280 : ∀ (d : ℕ), secret_spread d = 3280 → d = 7 :=
  sorry

end NUMINAMATH_CALUDE_secret_spread_day_secret_spread_saturday_unique_day_for_3280_l1253_125389


namespace NUMINAMATH_CALUDE_pet_store_snakes_l1253_125327

/-- The number of snakes in a pet store -/
theorem pet_store_snakes (num_cages : ℕ) (snakes_per_cage : ℕ) : 
  num_cages = 2 → snakes_per_cage = 2 → num_cages * snakes_per_cage = 4 := by
  sorry

#check pet_store_snakes

end NUMINAMATH_CALUDE_pet_store_snakes_l1253_125327


namespace NUMINAMATH_CALUDE_candies_remaining_l1253_125332

/-- Calculates the number of candies remaining after Carlos ate all yellow candies -/
theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow = 3 * red - 20)
  (h3 : blue = yellow / 2) :
  red + blue = 90 := by
  sorry

#check candies_remaining

end NUMINAMATH_CALUDE_candies_remaining_l1253_125332


namespace NUMINAMATH_CALUDE_f_properties_l1253_125312

noncomputable def f (x : ℝ) : ℝ := x - Real.log x - 1

theorem f_properties :
  (∀ x > 0, f x ≥ 0) ∧
  (∀ p : ℝ, (∀ x ≥ 1, f (1/x) ≥ (Real.log x)^2 / (p + Real.log x)) ↔ p ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1253_125312


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1253_125366

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = Complex.mk (-11/13) (29/13) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1253_125366


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_14_l1253_125303

theorem binomial_coefficient_21_14 : Nat.choose 21 14 = 116280 :=
by
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_14_l1253_125303


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l1253_125364

/-- The number of teams in the conference -/
def num_teams : ℕ := 12

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

/-- Theorem stating the total number of games in a season -/
theorem high_school_twelve_games :
  total_games = 204 :=
sorry

end NUMINAMATH_CALUDE_high_school_twelve_games_l1253_125364


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1253_125390

def A : Set ℕ := {0, 1}
def B : Set ℕ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1253_125390


namespace NUMINAMATH_CALUDE_bathroom_break_duration_l1253_125374

theorem bathroom_break_duration
  (total_distance : ℝ)
  (driving_speed : ℝ)
  (lunch_break : ℝ)
  (num_bathroom_breaks : ℕ)
  (total_trip_time : ℝ)
  (h1 : total_distance = 480)
  (h2 : driving_speed = 60)
  (h3 : lunch_break = 0.5)
  (h4 : num_bathroom_breaks = 2)
  (h5 : total_trip_time = 9) :
  (total_trip_time - total_distance / driving_speed - lunch_break) / num_bathroom_breaks = 0.25 := by
  sorry

#check bathroom_break_duration

end NUMINAMATH_CALUDE_bathroom_break_duration_l1253_125374


namespace NUMINAMATH_CALUDE_clown_mobiles_count_l1253_125315

theorem clown_mobiles_count (clowns_per_mobile : ℕ) (total_clowns : ℕ) (mobiles_count : ℕ) : 
  clowns_per_mobile = 28 → 
  total_clowns = 140 → 
  mobiles_count * clowns_per_mobile = total_clowns →
  mobiles_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_clown_mobiles_count_l1253_125315


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l1253_125380

theorem candidate_vote_difference :
  let total_votes : ℝ := 25000.000000000007
  let candidate_percentage : ℝ := 0.4
  let rival_percentage : ℝ := 1 - candidate_percentage
  let candidate_votes : ℝ := total_votes * candidate_percentage
  let rival_votes : ℝ := total_votes * rival_percentage
  let vote_difference : ℝ := rival_votes - candidate_votes
  ∃ (ε : ℝ), ε > 0 ∧ |vote_difference - 5000| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l1253_125380


namespace NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_digit_sum_not_9_twelve_cubed_base_10_sum_of_digits_1728_base_10_largest_base_for_12_cubed_digit_sum_not_9_l1253_125336

/-- The sum of digits of a natural number in a given base -/
def sum_of_digits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- The representation of a natural number in a given base -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem largest_base_for_12_cubed_digit_sum :
  ∀ b : ℕ, b > 8 → sum_of_digits (12^3) b = 3^2 := by sorry

theorem base_8_digit_sum_not_9 :
  sum_of_digits (12^3) 8 ≠ 3^2 := by sorry

theorem twelve_cubed_base_10 :
  12^3 = 1728 := by sorry

theorem sum_of_digits_1728_base_10 :
  sum_of_digits 1728 10 = 3^2 := by sorry

/-- 8 is the largest base b such that the sum of the base-b digits of 12^3 is not equal to 3^2 -/
theorem largest_base_for_12_cubed_digit_sum_not_9 :
  ∀ b : ℕ, b > 8 → sum_of_digits (12^3) b = 3^2 ∧
  sum_of_digits (12^3) 8 ≠ 3^2 := by sorry

end NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_digit_sum_not_9_twelve_cubed_base_10_sum_of_digits_1728_base_10_largest_base_for_12_cubed_digit_sum_not_9_l1253_125336


namespace NUMINAMATH_CALUDE_jellybean_problem_minimum_jellybean_count_l1253_125387

theorem jellybean_problem (n : ℕ) : 
  (n ≥ 150) ∧ (n % 17 = 15) → n ≥ 151 :=
by sorry

theorem minimum_jellybean_count : 
  ∃ (n : ℕ), n ≥ 150 ∧ n % 17 = 15 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_minimum_jellybean_count_l1253_125387


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1253_125302

theorem complex_equation_solution (x y : ℕ+) 
  (h : (x - Complex.I * y) ^ 2 = 15 - 20 * Complex.I) : 
  x - Complex.I * y = 5 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1253_125302


namespace NUMINAMATH_CALUDE_inequality_relation_l1253_125378

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l1253_125378
