import Mathlib

namespace NUMINAMATH_CALUDE_function_is_periodic_l821_82150

-- Define the function f and the constant a
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- State the theorem
theorem function_is_periodic
  (h1 : ∀ x, f x ≠ 0)
  (h2 : a > 0)
  (h3 : ∀ x, f (x - a) = 1 / f x) :
  ∀ x, f x = f (x + 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_function_is_periodic_l821_82150


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l821_82165

theorem unique_pair_satisfying_conditions :
  ∃! (n p : ℕ+), 
    (Nat.Prime p.val) ∧ 
    (-↑n : ℤ) ≤ 2 * ↑p ∧
    (↑p - 1 : ℤ) ^ n.val + 1 ∣ ↑n ^ (p.val - 1) ∧
    n = 3 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_conditions_l821_82165


namespace NUMINAMATH_CALUDE_color_tv_cost_price_l821_82154

/-- The cost price of a color TV satisfying the given conditions -/
def cost_price : ℝ := 3000

/-- The selling price before discount -/
def selling_price (cost : ℝ) : ℝ := cost * 1.4

/-- The discounted price -/
def discounted_price (price : ℝ) : ℝ := price * 0.8

/-- The profit is the difference between the discounted price and the cost price -/
def profit (cost : ℝ) : ℝ := discounted_price (selling_price cost) - cost

theorem color_tv_cost_price : 
  profit cost_price = 360 :=
sorry

end NUMINAMATH_CALUDE_color_tv_cost_price_l821_82154


namespace NUMINAMATH_CALUDE_carlos_summer_reading_l821_82158

/-- The number of books Carlos read in summer vacation --/
def total_books : ℕ := 100

/-- The number of books Carlos read in July --/
def july_books : ℕ := 28

/-- The number of books Carlos read in August --/
def august_books : ℕ := 30

/-- The number of books Carlos read in June --/
def june_books : ℕ := total_books - (july_books + august_books)

theorem carlos_summer_reading : june_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_carlos_summer_reading_l821_82158


namespace NUMINAMATH_CALUDE_passing_percentage_is_36_percent_l821_82155

/-- The passing percentage for an engineering exam --/
def passing_percentage (failed_marks : ℕ) (scored_marks : ℕ) (max_marks : ℕ) : ℚ :=
  ((scored_marks + failed_marks : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 36% --/
theorem passing_percentage_is_36_percent :
  passing_percentage 14 130 400 = 36 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_36_percent_l821_82155


namespace NUMINAMATH_CALUDE_alpha_in_third_quadrant_l821_82157

theorem alpha_in_third_quadrant (α : Real) 
  (h1 : Real.tan (α - 3 * Real.pi) > 0) 
  (h2 : Real.sin (-α + Real.pi) < 0) : 
  Real.pi < α ∧ α < 3 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_in_third_quadrant_l821_82157


namespace NUMINAMATH_CALUDE_product_sum_equality_l821_82119

/-- Given a base b, this function converts a number from base b to base 10 -/
def baseToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b -/
def decimalToBase (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem: If (13)(15)(17) = 4652 in base b, then 13 + 15 + 17 = 51 in base b -/
theorem product_sum_equality (b : ℕ) (h : b > 1) :
  (baseToDecimal 13 b * baseToDecimal 15 b * baseToDecimal 17 b = baseToDecimal 4652 b) →
  (decimalToBase (baseToDecimal 13 b + baseToDecimal 15 b + baseToDecimal 17 b) b = 51) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_equality_l821_82119


namespace NUMINAMATH_CALUDE_two_zeros_read_in_2006_06_l821_82104

-- Define a function to count the number of zeros read in a number
def countZerosRead (n : ℝ) : ℕ := sorry

-- Define the given numbers
def num1 : ℝ := 200.06
def num2 : ℝ := 20.06
def num3 : ℝ := 2006.06

-- Theorem statement
theorem two_zeros_read_in_2006_06 :
  (countZerosRead num1 < 2) ∧
  (countZerosRead num2 < 2) ∧
  (countZerosRead num3 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_read_in_2006_06_l821_82104


namespace NUMINAMATH_CALUDE_smallest_sum_for_equation_l821_82138

theorem smallest_sum_for_equation : ∃ (a b : ℕ+), 
  (2^10 * 7^4 : ℕ) = a^(b:ℕ) ∧ 
  (∀ (c d : ℕ+), (2^10 * 7^4 : ℕ) = c^(d:ℕ) → a + b ≤ c + d) ∧
  a + b = 1570 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_for_equation_l821_82138


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l821_82170

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 5) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l821_82170


namespace NUMINAMATH_CALUDE_unique_positive_solution_l821_82110

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x^8 + 8*x^7 + 28*x^6 + 2023*x^5 - 1807*x^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l821_82110


namespace NUMINAMATH_CALUDE_factors_of_48_l821_82184

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by sorry

end NUMINAMATH_CALUDE_factors_of_48_l821_82184


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l821_82176

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℕ) (h₁ : a₁ = 4) (h₂ : a₂ = 13) (h₃ : a₃ = 22) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 130 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l821_82176


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l821_82153

/-- Given a quadratic function f(x) = ax² + 2, prove that if its tangent line
    at x = 1 is perpendicular to the line 2x - y + 1 = 0, then a = -1/4. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x
  let tangent_slope : ℝ := f' 1
  let perpendicular_line_slope : ℝ := 2
  tangent_slope * perpendicular_line_slope = -1 → a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l821_82153


namespace NUMINAMATH_CALUDE_equivalence_complex_inequality_l821_82190

theorem equivalence_complex_inequality (a b c d : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ z : ℂ, Complex.abs (z - a) + Complex.abs (z - b) ≥ 
    Complex.abs (z - c) + Complex.abs (z - d)) ↔
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ 
    c = t • a + (1 - t) • b ∧ 
    d = (1 - t) • a + t • b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_complex_inequality_l821_82190


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l821_82181

/-- Represents an arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem largest_n_for_product_1764 :
  ∀ u v : ℕ,
  u ≥ 1 → v ≥ 1 → u ≤ v →
  ∃ n : ℕ, n ≥ 1 ∧
    (arithmeticSequence 3 u n) * (arithmeticSequence 3 v n) = 1764 →
  ∀ m : ℕ, m > 40 →
    (arithmeticSequence 3 u m) * (arithmeticSequence 3 v m) ≠ 1764 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l821_82181


namespace NUMINAMATH_CALUDE_flower_arrangement_problem_l821_82114

/-- Represents the flower arrangement problem --/
theorem flower_arrangement_problem 
  (initial_roses : ℕ) 
  (initial_daisies : ℕ) 
  (thrown_roses : ℕ) 
  (thrown_daisies : ℕ) 
  (final_roses : ℕ) 
  (final_daisies : ℕ) 
  (time_constraint : ℕ) :
  initial_roses = 21 →
  initial_daisies = 17 →
  thrown_roses = 34 →
  thrown_daisies = 25 →
  final_roses = 15 →
  final_daisies = 10 →
  time_constraint = 2 →
  (thrown_roses + thrown_daisies) - 
  ((thrown_roses - initial_roses + final_roses) + 
   (thrown_daisies - initial_daisies + final_daisies)) = 13 :=
by sorry


end NUMINAMATH_CALUDE_flower_arrangement_problem_l821_82114


namespace NUMINAMATH_CALUDE_deaf_students_count_l821_82149

/-- Represents a school for deaf and blind students. -/
structure DeafBlindSchool where
  total_students : ℕ
  deaf_students : ℕ
  blind_students : ℕ
  deaf_triple_blind : deaf_students = 3 * blind_students
  total_sum : total_students = deaf_students + blind_students

/-- Theorem: In a school with 240 total students, where the number of deaf students
    is three times the number of blind students, the number of deaf students is 180. -/
theorem deaf_students_count (school : DeafBlindSchool) 
  (h_total : school.total_students = 240) : school.deaf_students = 180 := by
  sorry

end NUMINAMATH_CALUDE_deaf_students_count_l821_82149


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l821_82112

/-- The height of a tree after n years, given its initial height and growth rate -/
def tree_height (initial_height : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_rate ^ n

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h1 : ∃ initial_height : ℝ, tree_height initial_height 3 5 = 243)
  (h2 : ∀ n : ℕ, tree_height initial_height 3 (n + 1) = 3 * tree_height initial_height 3 n) :
  tree_height initial_height 3 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l821_82112


namespace NUMINAMATH_CALUDE_mk_97_check_one_l821_82189

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 0 ∧ 4 * (x^2 - x) = 0 := by sorry

end NUMINAMATH_CALUDE_mk_97_check_one_l821_82189


namespace NUMINAMATH_CALUDE_triangle_area_l821_82107

/-- Given a triangle ABC where:
    - The side opposite to angle B has length 2
    - The side opposite to angle C has length 2√3
    - Angle C measures 2π/3 radians
    Prove that the area of the triangle is 3 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 3 →
  C = 2 * π / 3 →
  (1/2) * b * c * Real.sin A = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l821_82107


namespace NUMINAMATH_CALUDE_extended_volume_of_specific_box_l821_82124

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The main theorem -/
theorem extended_volume_of_specific_box :
  let box : Box := { length := 2, width := 3, height := 6 }
  extendedVolume box = (324 + 37 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_volume_of_specific_box_l821_82124


namespace NUMINAMATH_CALUDE_inequality_solution_l821_82133

theorem inequality_solution :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2 ∧
    a = 0 ∧ b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l821_82133


namespace NUMINAMATH_CALUDE_max_silver_tokens_l821_82106

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rates for the two booths --/
structure ExchangeRates where
  redToSilver : TokenCount → TokenCount
  blueToSilver : TokenCount → TokenCount

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 60, blue := 90, silver := 0 }

/-- The exchange rates for the two booths --/
def boothRates : ExchangeRates :=
  { redToSilver := λ tc => { red := tc.red - 3, blue := tc.blue + 2, silver := tc.silver + 1 },
    blueToSilver := λ tc => { red := tc.red + 1, blue := tc.blue - 4, silver := tc.silver + 2 } }

/-- Determines if further exchanges are possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (finalTokens : TokenCount),
    (¬canExchange finalTokens) ∧
    (finalTokens.silver = 101) ∧
    (∃ (exchanges : List (TokenCount → TokenCount)),
      exchanges.foldl (λ acc f => f acc) initialTokens = finalTokens) :=
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l821_82106


namespace NUMINAMATH_CALUDE_asterisk_replacement_l821_82121

theorem asterisk_replacement : (54 / 18) * (54 / 162) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l821_82121


namespace NUMINAMATH_CALUDE_right_triangle_area_l821_82103

/-- Given a right triangle with circumscribed circle radius R and inscribed circle radius r,
    prove that its area is r(2R + r). -/
theorem right_triangle_area (R r : ℝ) (h_positive_R : R > 0) (h_positive_r : r > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c = 2 * R ∧
    r = (a + b - c) / 2 ∧
    a^2 + b^2 = c^2 ∧
    (1/2) * a * b = r * (2 * R + r) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l821_82103


namespace NUMINAMATH_CALUDE_min_correct_answers_to_advance_l821_82123

/-- Given a math competition with the following conditions:
  * There are 25 questions in total
  * Each correct answer is worth 4 points
  * Each incorrect or unanswered question results in -1 point
  * A minimum of 60 points is required to advance
  This theorem proves that the minimum number of correctly answered questions
  to advance is 17. -/
theorem min_correct_answers_to_advance (total_questions : ℕ) (correct_points : ℤ) 
  (incorrect_points : ℤ) (min_points_to_advance : ℤ) :
  total_questions = 25 →
  correct_points = 4 →
  incorrect_points = -1 →
  min_points_to_advance = 60 →
  ∃ (min_correct : ℕ), 
    min_correct = 17 ∧ 
    (min_correct : ℤ) * correct_points + (total_questions - min_correct) * incorrect_points ≥ min_points_to_advance ∧
    ∀ (x : ℕ), x < min_correct → 
      (x : ℤ) * correct_points + (total_questions - x) * incorrect_points < min_points_to_advance :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_advance_l821_82123


namespace NUMINAMATH_CALUDE_jarry_secretary_or_treasurer_prob_l821_82109

/-- A club with 10 members, including Jarry -/
structure Club where
  members : Finset Nat
  jarry : Nat
  total_members : members.card = 10
  jarry_in_club : jarry ∈ members

/-- The probability of Jarry being either secretary or treasurer -/
def probability_jarry_secretary_or_treasurer (club : Club) : ℚ :=
  19 / 90

/-- Theorem stating the probability of Jarry being secretary or treasurer -/
theorem jarry_secretary_or_treasurer_prob (club : Club) :
  probability_jarry_secretary_or_treasurer club = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_jarry_secretary_or_treasurer_prob_l821_82109


namespace NUMINAMATH_CALUDE_bus_driver_compensation_theorem_l821_82111

/-- Represents the compensation structure and work hours of a bus driver -/
structure BusDriverCompensation where
  regular_rate : ℝ
  overtime_rate : ℝ
  total_compensation : ℝ
  total_hours : ℝ
  regular_hours_limit : ℝ

/-- Calculates the overtime rate based on the regular rate -/
def overtime_rate (regular_rate : ℝ) : ℝ :=
  regular_rate * 1.75

/-- Theorem stating the conditions and the result to be proved -/
theorem bus_driver_compensation_theorem (driver : BusDriverCompensation) :
  driver.regular_rate = 16 ∧
  driver.overtime_rate = overtime_rate driver.regular_rate ∧
  driver.total_compensation = 920 ∧
  driver.total_hours = 50 →
  driver.regular_hours_limit = 40 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_theorem_l821_82111


namespace NUMINAMATH_CALUDE_equal_cake_division_l821_82163

theorem equal_cake_division (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_division_l821_82163


namespace NUMINAMATH_CALUDE_vector_expression_l821_82166

/-- Given vectors a, b, and c in ℝ², prove that c = 2a + b -/
theorem vector_expression (a b c : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : a + b = (0, 3)) 
  (h3 : c = (1, 5)) : 
  c = 2 • a + b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l821_82166


namespace NUMINAMATH_CALUDE_example_is_fractional_equation_l821_82198

/-- Definition of a fractional equation -/
def is_fractional_equation (eq : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ) (h : ℝ → ℝ), 
    (∀ y, f y ≠ 0 ∧ g y ≠ 0) ∧ 
    eq ↔ (h x / f x - 3 / g x = 1) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ f x = a * x + b) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ g x = c * x + d)

/-- The equation (x / (2x - 1)) - (3 / (2x + 1)) = 1 is a fractional equation -/
theorem example_is_fractional_equation : 
  is_fractional_equation (∃ x : ℝ, x / (2 * x - 1) - 3 / (2 * x + 1) = 1) :=
sorry

end NUMINAMATH_CALUDE_example_is_fractional_equation_l821_82198


namespace NUMINAMATH_CALUDE_bc_length_l821_82137

/-- A right triangle with specific properties -/
structure RightTriangle where
  -- The lengths of the sides
  ab : ℝ
  ac : ℝ
  bc : ℝ
  -- The median from A to BC
  median : ℝ
  -- Conditions
  ab_eq : ab = 3
  ac_eq : ac = 4
  median_eq_bc : median = bc
  pythagorean : bc^2 = ab^2 + ac^2

/-- The length of BC in the specific right triangle is 5 -/
theorem bc_length (t : RightTriangle) : t.bc = 5 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l821_82137


namespace NUMINAMATH_CALUDE_inequality_solution_length_l821_82152

theorem inequality_solution_length (a b : ℝ) : 
  (∀ x, (a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) ↔ ((a - 5) / 3 ≤ x ∧ x ≤ (b - 8) / 3)) →
  ((b - 8) / 3 - (a - 5) / 3 = 18) →
  b - a = 57 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l821_82152


namespace NUMINAMATH_CALUDE_intercepts_sum_l821_82193

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the specific parabola y = 3x^2 - 9x + 4 -/
def parabola : QuadraticFunction :=
  { a := 3, b := -9, c := 4 }

/-- The y-intercept of the parabola -/
def y_intercept : Point :=
  { x := 0, y := parabola.c }

/-- Theorem stating that the sum of the y-intercept's y-coordinate and the x-coordinates of the two x-intercepts equals 19/3 -/
theorem intercepts_sum (e f : ℝ) 
  (h1 : parabola.a * e^2 + parabola.b * e + parabola.c = 0)
  (h2 : parabola.a * f^2 + parabola.b * f + parabola.c = 0)
  (h3 : e ≠ f) : 
  y_intercept.y + e + f = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_intercepts_sum_l821_82193


namespace NUMINAMATH_CALUDE_wild_animal_picture_difference_l821_82175

/-- The number of wild animal pictures Ralph has -/
def ralph_wild_animals : ℕ := 58

/-- The number of wild animal pictures Derrick has -/
def derrick_wild_animals : ℕ := 76

/-- Theorem stating the difference in wild animal pictures between Derrick and Ralph -/
theorem wild_animal_picture_difference :
  derrick_wild_animals - ralph_wild_animals = 18 := by sorry

end NUMINAMATH_CALUDE_wild_animal_picture_difference_l821_82175


namespace NUMINAMATH_CALUDE_square_perimeters_product_l821_82159

theorem square_perimeters_product (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 85)
  (h2 : x ^ 2 - y ^ 2 = 45)
  : (4 * x) * (4 * y) = 32 * Real.sqrt 325 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_product_l821_82159


namespace NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l821_82142

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

end NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l821_82142


namespace NUMINAMATH_CALUDE_correct_mark_proof_l821_82185

/-- Proves that given a class of 20 pupils, if entering 73 instead of the correct mark
    increases the class average by 0.5, then the correct mark should have been 63. -/
theorem correct_mark_proof (n : ℕ) (wrong_mark correct_mark : ℝ) 
    (h1 : n = 20)
    (h2 : wrong_mark = 73)
    (h3 : (wrong_mark - correct_mark) / n = 0.5) :
  correct_mark = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_proof_l821_82185


namespace NUMINAMATH_CALUDE_triangle_properties_l821_82102

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with angles A, B, C and opposite sides a, b, c
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- B is an obtuse angle
  π/2 < B ∧ B < π ∧
  -- √3a = 2b sin A
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  -- 1. B = 2π/3
  B = 2 * π / 3 ∧
  -- 2. If the area is 15√3/4 and b = 7, then a + c = 8
  (1/2 * a * c * Real.sin B = 15 * Real.sqrt 3 / 4 ∧ b = 7 → a + c = 8) ∧
  -- 3. If b = 6, the maximum area is 3√3
  (b = 6 → ∀ (a' c' : ℝ), 1/2 * a' * c' * Real.sin B ≤ 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l821_82102


namespace NUMINAMATH_CALUDE_modulus_of_z_values_of_a_and_b_l821_82177

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)

-- Theorem for the modulus of z
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

-- Theorem for the values of a and b
theorem values_of_a_and_b :
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + i → a = -3 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_values_of_a_and_b_l821_82177


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l821_82147

theorem divisibility_implies_equality (a b : ℕ+) (h : (a * b : ℕ) ∣ (a ^ 2 + b ^ 2 : ℕ)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l821_82147


namespace NUMINAMATH_CALUDE_eleventh_flip_probability_l821_82122

def is_fair_coin (coin : Type) : Prop := sorry

def probability_of_tails (coin : Type) : ℚ := sorry

def previous_flips_heads (coin : Type) (n : ℕ) : Prop := sorry

theorem eleventh_flip_probability (coin : Type) 
  (h_fair : is_fair_coin coin)
  (h_previous : previous_flips_heads coin 10) :
  probability_of_tails coin = 1/2 := by sorry

end NUMINAMATH_CALUDE_eleventh_flip_probability_l821_82122


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l821_82173

theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l821_82173


namespace NUMINAMATH_CALUDE_odd_function_implies_a_zero_l821_82101

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- State the theorem
theorem odd_function_implies_a_zero :
  (∀ x, f a x = -f a (-x)) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_zero_l821_82101


namespace NUMINAMATH_CALUDE_periodic_trig_function_l821_82192

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx - β), where α, β, a, and b are non-zero real numbers,
    if f(2016) = -1, then f(2017) = 1 -/
theorem periodic_trig_function (α β a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x - β)
  f 2016 = -1 → f 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l821_82192


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_two_l821_82197

theorem divisibility_by_power_of_two (n : ℕ) (h : n > 0) :
  ∃ x : ℤ, (2^n : ℤ) ∣ (x^2 - 17) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_two_l821_82197


namespace NUMINAMATH_CALUDE_prime_divisibility_special_primes_characterization_l821_82191

theorem prime_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → 
  (p + q^2) ∣ (p^2 + q) → (p + q^2) ∣ (p*q - 1) := by
  sorry

-- Part b
def special_primes : Set ℕ := {p | Nat.Prime p ∧ (p + 121) ∣ (p^2 + 11)}

-- The theorem states that the set of special primes is equal to {101, 323, 1211}
theorem special_primes_characterization : 
  special_primes = {101, 323, 1211} := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_special_primes_characterization_l821_82191


namespace NUMINAMATH_CALUDE_negative_sum_positive_product_l821_82140

theorem negative_sum_positive_product (a b : ℝ) : 
  a + b < 0 → ab > 0 → a < 0 ∧ b < 0 := by sorry

end NUMINAMATH_CALUDE_negative_sum_positive_product_l821_82140


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l821_82148

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 4 ∨ x = (-8 - Real.sqrt 16) / 4) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l821_82148


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l821_82167

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 50 = 20) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l821_82167


namespace NUMINAMATH_CALUDE_nonSimilar500PointedStars_l821_82105

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStars (n : ℕ) : ℕ :=
  (n.totient - 2) / 2

/-- Theorem: The number of non-similar regular 500-pointed stars is 99 -/
theorem nonSimilar500PointedStars : nonSimilarStars 500 = 99 := by
  sorry

#eval nonSimilarStars 500  -- This should evaluate to 99

end NUMINAMATH_CALUDE_nonSimilar500PointedStars_l821_82105


namespace NUMINAMATH_CALUDE_intersection_M_N_l821_82161

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l821_82161


namespace NUMINAMATH_CALUDE_crates_needed_is_fifteen_l821_82127

/-- Calculates the number of crates needed to load items in a warehouse --/
def calculate_crates (crate_capacity : ℕ) (nail_bags : ℕ) (nail_weight : ℕ) 
  (hammer_bags : ℕ) (hammer_weight : ℕ) (plank_bags : ℕ) (plank_weight : ℕ) 
  (left_out_weight : ℕ) : ℕ :=
  let total_weight := nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight
  let loadable_weight := total_weight - left_out_weight
  (loadable_weight + crate_capacity - 1) / crate_capacity

/-- Theorem stating that given the problem conditions, 15 crates are needed --/
theorem crates_needed_is_fifteen :
  calculate_crates 20 4 5 12 5 10 30 80 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crates_needed_is_fifteen_l821_82127


namespace NUMINAMATH_CALUDE_age_difference_l821_82145

/-- Given that Sachin is 14 years old and the ratio of Sachin's age to Rahul's age is 7:9,
    prove that the difference between Rahul's age and Sachin's age is 4 years. -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 14 → 
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l821_82145


namespace NUMINAMATH_CALUDE_det_A_equals_two_l821_82116

theorem det_A_equals_two (a d : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![a, -2; 1, d] →
  A + 2 * A⁻¹ = 0 →
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_det_A_equals_two_l821_82116


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l821_82135

theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ)
  (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : total_boys = 22)
  (h5 : total_girls = 38)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = total_boys + total_girls)
  (h9 : sad_children ≥ sad_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l821_82135


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_difference_l821_82194

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Tangent points divide sides
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  t_d : ℝ
  -- Conditions
  side_sum : a + b = t_a + t_b
  side_sum' : b + c = t_b + t_c
  side_sum'' : c + d = t_c + t_d
  side_sum''' : d + a = t_d + t_a

/-- The main theorem -/
theorem inscribed_circle_tangent_difference 
  (q : CyclicQuadrilateralWithInscribedCircle)
  (h1 : q.a = 70)
  (h2 : q.b = 90)
  (h3 : q.c = 130)
  (h4 : q.d = 110) :
  |q.t_c - (q.c - q.t_c)| = 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_difference_l821_82194


namespace NUMINAMATH_CALUDE_parabola_equation_l821_82196

/-- The equation of a parabola with focus (2, 0) and directrix x + 2 = 0 -/
theorem parabola_equation :
  ∀ (x y : ℝ),
    (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y) →
    (∀ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y →
      (P.1 - 2)^2 + P.2^2 = (P.1 + 2)^2) ↔
    y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l821_82196


namespace NUMINAMATH_CALUDE_snail_speed_ratio_l821_82187

-- Define the speeds and times
def speed_snail1 : ℝ := 2
def time_snail1 : ℝ := 20
def time_snail3 : ℝ := 2

-- Define the relationship between snail speeds
def speed_snail3 (speed_snail2 : ℝ) : ℝ := 5 * speed_snail2

-- Define the race distance
def race_distance : ℝ := speed_snail1 * time_snail1

-- Theorem statement
theorem snail_speed_ratio :
  ∃ (speed_snail2 : ℝ),
    speed_snail3 speed_snail2 * time_snail3 = race_distance ∧
    speed_snail2 / speed_snail1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_snail_speed_ratio_l821_82187


namespace NUMINAMATH_CALUDE_count_solutions_3x_plus_5y_equals_501_l821_82195

theorem count_solutions_3x_plus_5y_equals_501 :
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 5 * p.2 = 501 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 168) (Finset.range 101))).card = 34 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_3x_plus_5y_equals_501_l821_82195


namespace NUMINAMATH_CALUDE_always_odd_expression_l821_82113

theorem always_odd_expression (o n : ℕ) (ho : Odd o) (hn : n > 0) :
  Odd (o^3 + n^2 * o^2) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_expression_l821_82113


namespace NUMINAMATH_CALUDE_max_det_bound_l821_82162

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ) 
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  |M.det| ≤ 327680 * 2^16 := by
  sorry

end NUMINAMATH_CALUDE_max_det_bound_l821_82162


namespace NUMINAMATH_CALUDE_quadratic_minimum_point_l821_82168

/-- The x-coordinate of the minimum point of a quadratic function f(x) = x^2 - 2px + 4q,
    where p and q are positive real numbers, is p. -/
theorem quadratic_minimum_point (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - 2*p*x + 4*q
  (∀ x, f p ≤ f x) ∧ (∃ x, f p < f x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_point_l821_82168


namespace NUMINAMATH_CALUDE_paper_strip_division_l821_82174

theorem paper_strip_division (total_fraction : ℚ) (num_books : ℕ) : 
  total_fraction = 5/8 ∧ num_books = 5 → 
  total_fraction / num_books = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_division_l821_82174


namespace NUMINAMATH_CALUDE_number_minus_six_l821_82134

theorem number_minus_six : ∃ x : ℚ, x / 5 = 2 ∧ x - 6 = 4 := by sorry

end NUMINAMATH_CALUDE_number_minus_six_l821_82134


namespace NUMINAMATH_CALUDE_no_real_solutions_l821_82179

theorem no_real_solutions (a b c : ℝ) : ¬ ∃ x y z : ℝ, 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l821_82179


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l821_82160

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l821_82160


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l821_82172

def jungkook_erasers : ℕ := 6

def jimin_erasers (j : ℕ) : ℕ := j + 4

def seokjin_erasers (j : ℕ) : ℕ := j - 3

theorem jungkook_has_fewest_erasers :
  ∀ (j s : ℕ), 
    j = jimin_erasers jungkook_erasers →
    s = seokjin_erasers j →
    jungkook_erasers ≤ j ∧ jungkook_erasers ≤ s :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l821_82172


namespace NUMINAMATH_CALUDE_speed_time_relationship_l821_82169

-- Define the initial speed and time
variable (x y : ℝ)

-- Define the percentage increases/decreases
variable (a b : ℝ)

-- Condition: x and y are positive (speed and time can't be negative or zero)
variable (hx : x > 0)
variable (hy : y > 0)

-- Condition: a and b are percentages (between 0 and 100)
variable (ha : 0 ≤ a ∧ a ≤ 100)
variable (hb : 0 ≤ b ∧ b ≤ 100)

-- Theorem stating the relationship between a and b
theorem speed_time_relationship : 
  x * y = x * (1 + a / 100) * (y * (1 - b / 100)) → 
  b = (100 * a) / (100 + a) := by
sorry

end NUMINAMATH_CALUDE_speed_time_relationship_l821_82169


namespace NUMINAMATH_CALUDE_workers_contribution_problem_l821_82151

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The original total contribution in paise (100 paise = 1 rupee) -/
def original_total : ℕ := 30000000  -- 3 lacs = 300,000 rupees = 30,000,000 paise

/-- The new total contribution if each worker contributed 50 rupees extra, in paise -/
def new_total : ℕ := 36000000  -- 3.60 lacs = 360,000 rupees = 36,000,000 paise

/-- The extra contribution per worker in paise -/
def extra_contribution : ℕ := 5000  -- 50 rupees = 5,000 paise

theorem workers_contribution_problem :
  (original_total / number_of_workers : ℚ) * number_of_workers = original_total ∧
  ((original_total / number_of_workers : ℚ) + extra_contribution) * number_of_workers = new_total :=
sorry

end NUMINAMATH_CALUDE_workers_contribution_problem_l821_82151


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_5_million_l821_82188

theorem scientific_notation_of_56_5_million :
  56500000 = 5.65 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_5_million_l821_82188


namespace NUMINAMATH_CALUDE_unique_f_exists_l821_82143

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function f(n) to be proved unique -/
def f (n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem unique_f_exists (n : ℕ) (h1 : n > 1) (h2 : n ≠ 10) :
  ∃! fn : ℕ, fn ≥ 2 ∧ ∀ k : ℕ, 0 < k → k < fn →
    sum_of_digits k + sum_of_digits (fn - k) = n :=
sorry

end NUMINAMATH_CALUDE_unique_f_exists_l821_82143


namespace NUMINAMATH_CALUDE_parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l821_82120

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l: y = k(x+1) passing through M(-1, 0)
def line (k x y : ℝ) : Prop := y = k*(x+1)

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point M on the x-axis where the directrix intersects
def M : ℝ × ℝ := (-1, 0)

-- Define the relationship between AM and AF
def AM_AF_relation (A : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  (x₁ + 1)^2 + (y₁)^2 = (25/16) * ((x₁ - 1)^2 + y₁^2)

-- Define the perpendicularity condition for QA and QB
def perpendicular_condition (Q A B : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := Q
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₀ - y₁) * (y₀ - y₂) = -(x₀ - x₁) * (x₀ - x₂)

theorem parabola_intersection_and_perpendicularity (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line k A.1 A.2 ∧ 
    line k B.1 B.2 ∧ 
    AM_AF_relation A) →
  k = 3/4 ∨ k = -3/4 :=
sorry

theorem perpendicular_intersection_range (k : ℝ) :
  (∃ Q A B : ℝ × ℝ,
    parabola Q.1 Q.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    perpendicular_condition Q A B) ↔
  (k > 0 ∧ k < Real.sqrt 5 / 5) ∨ (k < 0 ∧ k > -Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l821_82120


namespace NUMINAMATH_CALUDE_walkway_area_is_416_l821_82186

/-- Represents a garden with flower beds and walkways -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℝ
  bed_height : ℝ
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h1 : g.rows = 4)
  (h2 : g.columns = 3)
  (h3 : g.bed_width = 8)
  (h4 : g.bed_height = 3)
  (h5 : g.walkway_width = 2) : 
  walkway_area g = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l821_82186


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_4_l821_82144

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability that the sum of two dice is greater than 4 -/
theorem prob_sum_greater_than_4 : 
  (total_outcomes - outcomes_sum_4_or_less : ℚ) / total_outcomes = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_4_l821_82144


namespace NUMINAMATH_CALUDE_sequence_prime_properties_l821_82130

/-- The sequence a(n) = 3^(2^n) + 1 for n ≥ 1 -/
def a (n : ℕ) : ℕ := 3^(2^n) + 1

/-- The set of primes that do not divide any term of the sequence -/
def nondividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, n ≥ 1 → ¬(p ∣ a n)}

/-- The set of primes that divide at least one term of the sequence -/
def dividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ n, n ≥ 1 ∧ p ∣ a n}

theorem sequence_prime_properties :
  (Set.Infinite nondividing_primes) ∧ (Set.Infinite dividing_primes) := by
  sorry

end NUMINAMATH_CALUDE_sequence_prime_properties_l821_82130


namespace NUMINAMATH_CALUDE_evaluate_expression_l821_82118

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l821_82118


namespace NUMINAMATH_CALUDE_garrett_granola_purchase_l821_82139

/-- Represents the cost of Garrett's granola bar purchase -/
def total_cost (oatmeal_count : ℕ) (oatmeal_price : ℚ) (peanut_count : ℕ) (peanut_price : ℚ) : ℚ :=
  oatmeal_count * oatmeal_price + peanut_count * peanut_price

/-- Proves that Garrett's total granola bar purchase cost is $19.50 -/
theorem garrett_granola_purchase :
  total_cost 6 1.25 8 1.50 = 19.50 := by
  sorry

end NUMINAMATH_CALUDE_garrett_granola_purchase_l821_82139


namespace NUMINAMATH_CALUDE_f_surjective_l821_82128

/-- Sequence of prime numbers -/
def prime_seq : ℕ → ℕ 
  | 0 => 2
  | n + 1 => sorry

/-- The function f as defined in the problem -/
noncomputable def f (k n : ℕ+) : ℕ := sorry

/-- The main theorem to be proved -/
theorem f_surjective : ∀ M : ℕ+, ∃ k n : ℕ+, f k n = M := by sorry

end NUMINAMATH_CALUDE_f_surjective_l821_82128


namespace NUMINAMATH_CALUDE_laura_debt_l821_82115

/-- Calculates the total amount owed after one year given a principal amount,
    an annual interest rate, and assuming simple interest. -/
def totalAmountOwed (principal : ℝ) (interestRate : ℝ) : ℝ :=
  principal * (1 + interestRate)

/-- Proves that given a principal of $35 and an interest rate of 9%,
    the total amount owed after one year is $38.15. -/
theorem laura_debt : totalAmountOwed 35 0.09 = 38.15 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_l821_82115


namespace NUMINAMATH_CALUDE_sequence_ratio_l821_82100

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n - 2,
    prove that the ratio a_8 / a_6 = 4. -/
theorem sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 2) : 
  a 8 / a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l821_82100


namespace NUMINAMATH_CALUDE_sum_difference_arithmetic_sequences_l821_82182

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_difference (seq1 seq2 : List ℕ) : ℕ :=
  (seq1.zip seq2).map (λ (a, b) => a - b) |>.sum

theorem sum_difference_arithmetic_sequences : 
  let seq1 := arithmetic_sequence 2101 1 123
  let seq2 := arithmetic_sequence 401 1 123
  sum_difference seq1 seq2 = 209100 := by
sorry

#eval sum_difference (arithmetic_sequence 2101 1 123) (arithmetic_sequence 401 1 123)

end NUMINAMATH_CALUDE_sum_difference_arithmetic_sequences_l821_82182


namespace NUMINAMATH_CALUDE_diagonal_crosses_820_cubes_l821_82183

/-- The number of unit cubes crossed by an internal diagonal in a rectangular solid. -/
def cubesCrossedByDiagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem stating that the number of cubes crossed by the diagonal in a 200 × 330 × 360 solid is 820. -/
theorem diagonal_crosses_820_cubes :
  cubesCrossedByDiagonal 200 330 360 = 820 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_crosses_820_cubes_l821_82183


namespace NUMINAMATH_CALUDE_at_least_two_unusual_numbers_l821_82171

/-- A hundred-digit number is unusual if its cube ends with itself but its square does not. -/
def IsUnusual (n : ℕ) : Prop :=
  n ^ 3 % 10^100 = n % 10^100 ∧ n ^ 2 % 10^100 ≠ n % 10^100

/-- There are at least two hundred-digit unusual numbers. -/
theorem at_least_two_unusual_numbers : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10^99 ≤ n₁ ∧ n₁ < 10^100 ∧
  10^99 ≤ n₂ ∧ n₂ < 10^100 ∧
  IsUnusual n₁ ∧ IsUnusual n₂ := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_unusual_numbers_l821_82171


namespace NUMINAMATH_CALUDE_building_windows_l821_82178

/-- The number of windows already installed -/
def installed_windows : ℕ := 6

/-- The time it takes to install one window (in hours) -/
def hours_per_window : ℕ := 5

/-- The time it will take to install the remaining windows (in hours) -/
def remaining_hours : ℕ := 20

/-- The total number of windows needed for the building -/
def total_windows : ℕ := installed_windows + remaining_hours / hours_per_window

theorem building_windows : total_windows = 10 := by
  sorry

end NUMINAMATH_CALUDE_building_windows_l821_82178


namespace NUMINAMATH_CALUDE_square_perimeter_l821_82108

/-- Given two squares I and II, where the diagonal of I is a+b and the area of II is twice the area of I, 
    the perimeter of II is 4(a+b) -/
theorem square_perimeter (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 2 * area_I
  let side_II := Real.sqrt area_II
  side_II * 4 = 4 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l821_82108


namespace NUMINAMATH_CALUDE_local_taxes_in_cents_l821_82117

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- The local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The amount of local taxes paid in cents per hour is 60 -/
theorem local_taxes_in_cents : 
  (hourly_wage * tax_rate * cents_per_dollar : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_local_taxes_in_cents_l821_82117


namespace NUMINAMATH_CALUDE_trig_identity_l821_82125

theorem trig_identity : 
  (Real.tan (7.5 * π / 180) * Real.tan (15 * π / 180)) / 
    (Real.tan (15 * π / 180) - Real.tan (7.5 * π / 180)) + 
  Real.sqrt 3 * (Real.sin (7.5 * π / 180)^2 - Real.cos (7.5 * π / 180)^2) = 
  -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l821_82125


namespace NUMINAMATH_CALUDE_estate_division_l821_82164

theorem estate_division (E : ℝ) 
  (h1 : ∃ (x : ℝ), 6 * x = 2/3 * E)  -- Two sons and daughter receive 2/3 of estate in 3:2:1 ratio
  (h2 : ∃ (x : ℝ), 3 * x = E - (9 * x + 750))  -- Wife's share is 3x, where x is daughter's share
  (h3 : 750 ≤ E)  -- Butler's share is $750
  : E = 7500 := by
  sorry

end NUMINAMATH_CALUDE_estate_division_l821_82164


namespace NUMINAMATH_CALUDE_dice_sum_pigeonhole_l821_82136

/-- A type representing a fair six-sided die -/
def Die := Fin 6

/-- The sum of four dice rolls -/
def DiceSum := Nat

/-- The minimum number of throws required to guarantee a repeated sum -/
def MinThrows : Nat := 22

/-- The number of possible distinct sums when rolling four dice -/
def DistinctSums : Nat := 21

theorem dice_sum_pigeonhole :
  MinThrows = DistinctSums + 1 ∧
  ∀ n : Nat, n < MinThrows → 
    ∃ f : Fin n → DiceSum,
      ∀ i j : Fin n, i ≠ j → f i ≠ f j :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_pigeonhole_l821_82136


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l821_82132

theorem pascal_triangle_interior_sum (row_6_sum : ℕ) (row_8_sum : ℕ) : 
  row_6_sum = 30 → row_8_sum = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l821_82132


namespace NUMINAMATH_CALUDE_mark_distance_before_turning_l821_82199

/-- Proves that Mark walked 7.5 miles before turning around -/
theorem mark_distance_before_turning (chris_speed : ℝ) (school_distance : ℝ) 
  (mark_extra_time : ℝ) (h1 : chris_speed = 3) (h2 : school_distance = 9) 
  (h3 : mark_extra_time = 2) : 
  let chris_time := school_distance / chris_speed
  let mark_time := chris_time + mark_extra_time
  let mark_total_distance := chris_speed * mark_time
  mark_total_distance / 2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_mark_distance_before_turning_l821_82199


namespace NUMINAMATH_CALUDE_problem_solution_l821_82146

theorem problem_solution (x : ℚ) : x = (1 / x) * (-x) - 3 * x + 4 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l821_82146


namespace NUMINAMATH_CALUDE_sequence_nonpositive_l821_82180

theorem sequence_nonpositive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h_convex : ∀ k : ℕ, 1 ≤ k ∧ k < n → a (k-1) - 2*a k + a (k+1) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_nonpositive_l821_82180


namespace NUMINAMATH_CALUDE_complement_subset_relation_l821_82131

open Set

theorem complement_subset_relation (P Q : Set ℝ) : 
  (P = {x : ℝ | 0 < x ∧ x < 1}) → 
  (Q = {x : ℝ | x^2 + x - 2 ≤ 0}) → 
  ((compl Q) ⊆ (compl P)) :=
by
  sorry

end NUMINAMATH_CALUDE_complement_subset_relation_l821_82131


namespace NUMINAMATH_CALUDE_distance_to_place_l821_82141

/-- The distance to the place given the man's rowing speed, river speed, and total time -/
theorem distance_to_place (mans_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  mans_speed = 4 →
  river_speed = 2 →
  total_time = 1.5 →
  (1 / (mans_speed + river_speed) + 1 / (mans_speed - river_speed)) * total_time = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_place_l821_82141


namespace NUMINAMATH_CALUDE_unknown_number_multiplication_l821_82129

theorem unknown_number_multiplication (x : ℤ) : 
  55 = x + 45 - 62 → 7 * x = 504 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_multiplication_l821_82129


namespace NUMINAMATH_CALUDE_square_ratio_problem_l821_82126

theorem square_ratio_problem :
  ∀ (s1 s2 : ℝ),
  (s1^2 / s2^2 = 75 / 128) →
  ∃ (a b c : ℕ),
  (s1 / s2 = (a : ℝ) * Real.sqrt b / c) ∧
  a = 5 ∧ b = 6 ∧ c = 16 ∧
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l821_82126


namespace NUMINAMATH_CALUDE_product_equals_four_l821_82156

theorem product_equals_four : 16 * 0.5 * 4 * 0.125 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_four_l821_82156
