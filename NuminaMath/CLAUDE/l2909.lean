import Mathlib

namespace NUMINAMATH_CALUDE_min_solutions_in_interval_l2909_290914

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_solutions_in_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period f 3) 
  (h_root : f 2 = 0) : 
  ∃ (a b c d : ℝ), 0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 6 ∧ 
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 :=
sorry

end NUMINAMATH_CALUDE_min_solutions_in_interval_l2909_290914


namespace NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2909_290949

/-- The number of days Sheila can place cinnamon balls -/
def days : ℕ := 10

/-- The total number of cinnamon balls Sheila bought -/
def total_balls : ℕ := 50

/-- The number of family members Sheila placed a cinnamon ball for every day -/
def family_members : ℕ := total_balls / days

theorem sheila_cinnamon_balls : family_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2909_290949


namespace NUMINAMATH_CALUDE_tank_capacity_l2909_290981

/-- The capacity of a tank given outlet and inlet pipe rates -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (combined_time : ℝ)
  (h1 : outlet_time = 8)
  (h2 : inlet_rate = 8)
  (h3 : combined_time = 12) :
  ∃ (capacity : ℝ), capacity = 11520 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / combined_time :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_l2909_290981


namespace NUMINAMATH_CALUDE_bill_difference_l2909_290916

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 5)
  (h2 : joe_tip = 10)
  (h3 : mike_percent = 20)
  (h4 : joe_percent = 25)
  (h5 : mike_tip = mike_percent / 100 * mike_bill)
  (h6 : joe_tip = joe_percent / 100 * joe_bill) :
  |mike_bill - joe_bill| = 15 :=
sorry

end NUMINAMATH_CALUDE_bill_difference_l2909_290916


namespace NUMINAMATH_CALUDE_g_uniqueness_l2909_290920

/-- The functional equation for g -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y

theorem g_uniqueness (g : ℝ → ℝ) (h1 : g 1 = 1) (h2 : FunctionalEquation g) :
    ∀ x : ℝ, g x = 4^x - 3^x := by
  sorry

end NUMINAMATH_CALUDE_g_uniqueness_l2909_290920


namespace NUMINAMATH_CALUDE_cube_roots_less_than_12_l2909_290992

theorem cube_roots_less_than_12 : 
  (Finset.range 1728).card = 1727 :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_less_than_12_l2909_290992


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_horse_ratio_l2909_290989

/-- Represents the Stewart farm with sheep and horses -/
structure StewartFarm where
  sheep : ℕ
  horses : ℕ
  horseFoodPerHorse : ℕ
  totalHorseFood : ℕ

/-- The ratio between two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the simplified ratio between two natural numbers -/
def simplifiedRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem: The ratio of sheep to horses on the Stewart farm is 5:7 -/
theorem stewart_farm_sheep_horse_ratio (farm : StewartFarm)
    (h1 : farm.sheep = 40)
    (h2 : farm.horseFoodPerHorse = 230)
    (h3 : farm.totalHorseFood = 12880)
    (h4 : farm.horses * farm.horseFoodPerHorse = farm.totalHorseFood) :
    simplifiedRatio farm.sheep farm.horses = { numerator := 5, denominator := 7 } := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_horse_ratio_l2909_290989


namespace NUMINAMATH_CALUDE_speed_ratio_is_one_third_l2909_290913

/-- The problem setup for two moving objects A and B -/
structure MovementProblem where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B
  initialDistance : ℝ  -- Initial distance of B from O

/-- The conditions of the problem -/
def satisfiesConditions (p : MovementProblem) : Prop :=
  p.initialDistance = 300 ∧
  p.vA = |p.initialDistance - p.vB| ∧
  7 * p.vA = |p.initialDistance - 7 * p.vB|

/-- The theorem to be proved -/
theorem speed_ratio_is_one_third (p : MovementProblem) 
  (h : satisfiesConditions p) : p.vA / p.vB = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_one_third_l2909_290913


namespace NUMINAMATH_CALUDE_partner_count_l2909_290927

theorem partner_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 := by
  sorry

end NUMINAMATH_CALUDE_partner_count_l2909_290927


namespace NUMINAMATH_CALUDE_quadratic_roots_implications_l2909_290911

theorem quadratic_roots_implications (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧
    (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2)) →
  (a ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo 3 4) ∧
  (∀ a' ∈ Set.Ioo 3 4, a'^3 > a'^2 - a' + 1) ∧
  (∀ a' ∈ Set.Ioo (-2 : ℝ) (-1), a'^3 < a'^2 - a' + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_implications_l2909_290911


namespace NUMINAMATH_CALUDE_volume_ratio_of_rotated_triangle_l2909_290931

/-- Given a right-angled triangle with perpendicular sides of lengths a and b,
    the ratio of the volume of the solid formed by rotating around side a
    to the volume of the solid formed by rotating around side b is b : a. -/
theorem volume_ratio_of_rotated_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / 3 * π * b^2 * a) / (1 / 3 * π * a^2 * b) = b / a :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_of_rotated_triangle_l2909_290931


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l2909_290985

theorem cubic_equation_properties (k : ℝ) :
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                (x ≠ y ∨ y ≠ z ∨ x ≠ z)) ∧
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                x + y + z = -2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l2909_290985


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l2909_290960

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  3 * x^2 + 6 * x - 4 = 0 ↔ x = (-3 + Real.sqrt 21) / 3 ∨ x = (-3 - Real.sqrt 21) / 3 := by
  sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (2 * x + 1) = 4 * x + 2 ↔ x = -1/2 ∨ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l2909_290960


namespace NUMINAMATH_CALUDE_simplify_expression_l2909_290993

theorem simplify_expression (x : ℝ) (h : x > 3) :
  3 * |3 - x| - |x^2 - 6*x + 10| + |x^2 - 2*x + 1| = 7*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2909_290993


namespace NUMINAMATH_CALUDE_johns_hats_cost_l2909_290934

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_in_week * cost_per_hat

theorem johns_hats_cost : total_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l2909_290934


namespace NUMINAMATH_CALUDE_triple_a_award_distribution_l2909_290937

theorem triple_a_award_distribution (n : Nat) (k : Nat) (h1 : n = 10) (h2 : k = 7) :
  (Nat.choose (n - k + k - 1) (n - k)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_triple_a_award_distribution_l2909_290937


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2909_290901

theorem unknown_number_proof (X : ℕ) : 
  1000 + X + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2909_290901


namespace NUMINAMATH_CALUDE_probability_of_selecting_letter_from_word_l2909_290987

/-- The number of characters in the extended alphabet -/
def alphabet_size : ℕ := 30

/-- The word from which we're checking letters -/
def word : String := "MATHEMATICS"

/-- The number of unique letters in the word -/
def unique_letters : ℕ := (word.toList.eraseDups).length

/-- The probability of selecting a letter from the word -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_selecting_letter_from_word :
  probability = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_letter_from_word_l2909_290987


namespace NUMINAMATH_CALUDE_median_equation_altitude_equation_l2909_290998

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

-- Theorem for the median equation
theorem median_equation : 
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 5*x + y = 20) ∧
    is_line_equation a b c A ∧
    is_line_equation a b c ((B.1 + C.1) / 2, (B.2 + C.2) / 2) :=
sorry

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (a b c : ℝ),
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 3*x + 2*y = 12) ∧
    is_line_equation a b c A ∧
    (∀ (p : ℝ × ℝ), is_line_equation (B.2 - C.2) (C.1 - B.1) 0 p → 
      (p.2 - A.2) * (p.1 - A.1) = -(a * (p.1 - A.1) + b * (p.2 - A.2))^2 / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_median_equation_altitude_equation_l2909_290998


namespace NUMINAMATH_CALUDE_all_hyperprimes_l2909_290955

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def isSegmentPrime (n : ℕ) : Prop :=
  ∀ start len : ℕ, len > 0 → start + len ≤ (Nat.digits 10 n).length →
    isPrime (Nat.digits 10 n |> List.take len |> List.drop start |> List.foldl (· * 10 + ·) 0)

def isHyperprime (n : ℕ) : Prop := n > 0 ∧ isSegmentPrime n

theorem all_hyperprimes :
  {n : ℕ | isHyperprime n} = {2, 3, 5, 7, 23, 37, 53, 73, 373} := by sorry

end NUMINAMATH_CALUDE_all_hyperprimes_l2909_290955


namespace NUMINAMATH_CALUDE_product_of_base9_digits_9876_l2909_290932

/-- Converts a base 10 number to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 9 representation of 9876₁₀ is 192 --/
theorem product_of_base9_digits_9876 :
  productOfList (toBase9 9876) = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base9_digits_9876_l2909_290932


namespace NUMINAMATH_CALUDE_students_not_participating_l2909_290979

/-- Given a class with the following properties:
  * There are 15 students in total
  * 7 students participate in mathematical modeling
  * 9 students participate in computer programming
  * 3 students participate in both activities
  This theorem proves that 2 students do not participate in either activity. -/
theorem students_not_participating (total : ℕ) (modeling : ℕ) (programming : ℕ) (both : ℕ) :
  total = 15 →
  modeling = 7 →
  programming = 9 →
  both = 3 →
  total - (modeling + programming - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_not_participating_l2909_290979


namespace NUMINAMATH_CALUDE_center_is_one_l2909_290977

/-- A 3x3 table of positive real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)
  (all_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)

/-- The conditions for the table -/
def TableConditions (t : Table) : Prop :=
  t.a * t.b * t.c = 1 ∧
  t.d * t.e * t.f = 1 ∧
  t.g * t.h * t.i = 1 ∧
  t.a * t.d * t.g = 1 ∧
  t.b * t.e * t.h = 1 ∧
  t.c * t.f * t.i = 1 ∧
  t.a * t.b * t.d * t.e = 2 ∧
  t.b * t.c * t.e * t.f = 2 ∧
  t.d * t.e * t.g * t.h = 2 ∧
  t.e * t.f * t.h * t.i = 2

/-- The theorem stating that the center cell must be 1 -/
theorem center_is_one (t : Table) (h : TableConditions t) : t.e = 1 := by
  sorry


end NUMINAMATH_CALUDE_center_is_one_l2909_290977


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l2909_290953

/-- A polynomial satisfying the given conditions -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x, P (x^2 + 1) = P x^2 + 1)

/-- Theorem stating that the identity function is the only polynomial satisfying the conditions -/
theorem unique_satisfying_polynomial :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P → (∀ x, P x = x) :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l2909_290953


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l2909_290952

theorem root_condition_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x > 1 ∧ y < 1) →
  m > 5/2 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l2909_290952


namespace NUMINAMATH_CALUDE_books_returned_percentage_l2909_290972

/-- Calculates the percentage of loaned books that were returned -/
def percentage_returned (initial_books : ℕ) (loaned_books : ℕ) (final_books : ℕ) : ℚ :=
  ((final_books - (initial_books - loaned_books)) : ℚ) / (loaned_books : ℚ) * 100

/-- Theorem stating that the percentage of returned books is 65% -/
theorem books_returned_percentage 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 20)
  (h3 : final_books = 68) : 
  percentage_returned initial_books loaned_books final_books = 65 := by
  sorry

#eval percentage_returned 75 20 68

end NUMINAMATH_CALUDE_books_returned_percentage_l2909_290972


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l2909_290996

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f' x = 4 → (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l2909_290996


namespace NUMINAMATH_CALUDE_lucky_larry_calculation_l2909_290958

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 16 ∧ b = 2 ∧ c = 3 ∧ d = 12 → 
  (a / (b / (c * (d / e))) = a / b / c * d / e) → 
  e = 9 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_calculation_l2909_290958


namespace NUMINAMATH_CALUDE_gcd_102_238_l2909_290951

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l2909_290951


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l2909_290940

theorem absolute_value_equation_roots : ∃ (x y : ℝ), 
  (x^2 - 3*|x| - 10 = 0) ∧ 
  (y^2 - 3*|y| - 10 = 0) ∧ 
  (x + y = 0) ∧ 
  (x * y = -25) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l2909_290940


namespace NUMINAMATH_CALUDE_movie_duration_l2909_290975

theorem movie_duration (screens : ℕ) (open_hours : ℕ) (total_movies : ℕ) 
  (h1 : screens = 6) 
  (h2 : open_hours = 8) 
  (h3 : total_movies = 24) : 
  (screens * open_hours) / total_movies = 2 := by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l2909_290975


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2909_290941

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 800 / y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 10) →  -- 0 < y ≤ 10
  (∃ (a' b' c' : ℕ), a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ 
    100 * a' + 10 * b' + c' = 800 / y ∧ 
    a' + b' + c' = 8 ∧
    ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 → 
      100 * x + 10 * y + z = 800 / y → x + y + z ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2909_290941


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2909_290944

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 4 ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2909_290944


namespace NUMINAMATH_CALUDE_range_of_f_l2909_290924

def f (x : ℝ) : ℝ := x^2 - 3*x

def domain : Set ℝ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2909_290924


namespace NUMINAMATH_CALUDE_correct_subtraction_l2909_290971

-- Define the polynomials
def original_poly (x : ℝ) := 2*x^2 - x + 3
def mistaken_poly (x : ℝ) := x^2 + 14*x - 6

-- Theorem statement
theorem correct_subtraction :
  ∀ x : ℝ, original_poly x - mistaken_poly x = -29*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2909_290971


namespace NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l2909_290902

theorem smallest_multiple_of_42_and_56_not_18 : 
  ∃ (n : ℕ), n > 0 ∧ 42 ∣ n ∧ 56 ∣ n ∧ ¬(18 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 42 ∣ m → 56 ∣ m → ¬(18 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l2909_290902


namespace NUMINAMATH_CALUDE_fraction_square_equals_twentyfive_l2909_290904

theorem fraction_square_equals_twentyfive : (123456^2 : ℚ) / (24691^2 : ℚ) = 25 := by sorry

end NUMINAMATH_CALUDE_fraction_square_equals_twentyfive_l2909_290904


namespace NUMINAMATH_CALUDE_exponential_fraction_simplification_l2909_290995

theorem exponential_fraction_simplification :
  (3^1008 + 3^1006) / (3^1008 - 3^1006) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fraction_simplification_l2909_290995


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2909_290957

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4 * i) / (1 + 2 * i) = 11 / 5 - 2 / 5 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2909_290957


namespace NUMINAMATH_CALUDE_ab_power_is_negative_eight_l2909_290910

theorem ab_power_is_negative_eight (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_is_negative_eight_l2909_290910


namespace NUMINAMATH_CALUDE_simplify_expression_l2909_290982

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2909_290982


namespace NUMINAMATH_CALUDE_quadratic_roots_modulus_l2909_290963

theorem quadratic_roots_modulus (a : ℝ) : 
  (∀ x : ℂ, (a * x^2 + x + 1 = 0) → Complex.abs x < 1) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_modulus_l2909_290963


namespace NUMINAMATH_CALUDE_logical_conditions_l2909_290933

-- Define a proposition type to represent logical statements
variable (A B : Prop)

-- Define sufficient condition
def is_sufficient_condition (A B : Prop) : Prop :=
  A → B

-- Define necessary condition
def is_necessary_condition (A B : Prop) : Prop :=
  B → A

-- Define necessary and sufficient condition
def is_necessary_and_sufficient_condition (A B : Prop) : Prop :=
  (A → B) ∧ (B → A)

-- Theorem statement
theorem logical_conditions :
  (is_sufficient_condition A B ↔ (A → B)) ∧
  (is_necessary_condition A B ↔ (B → A)) ∧
  (is_necessary_and_sufficient_condition A B ↔ ((A → B) ∧ (B → A))) :=
by sorry

end NUMINAMATH_CALUDE_logical_conditions_l2909_290933


namespace NUMINAMATH_CALUDE_max_distance_theorem_l2909_290965

def vector_a : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, y)
def vector_b : ℝ × ℝ := (1, 2)

theorem max_distance_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = Real.sqrt 5 + 1 ∧
    ∀ (a : ℝ × ℝ), a = vector_a (x, y) →
      ‖a - vector_b‖ ≤ max_val ∧
      ∃ (a' : ℝ × ℝ), a' = vector_a (x', y') ∧ x'^2 + y'^2 = 1 ∧ ‖a' - vector_b‖ = max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l2909_290965


namespace NUMINAMATH_CALUDE_stratified_sampling_city_B_l2909_290962

theorem stratified_sampling_city_B (total_points : ℕ) (city_B_points : ℕ) (sample_size : ℕ) :
  total_points = 450 →
  city_B_points = 150 →
  sample_size = 90 →
  (city_B_points : ℚ) / (total_points : ℚ) * (sample_size : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_city_B_l2909_290962


namespace NUMINAMATH_CALUDE_prob_sum_15_three_dice_l2909_290961

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The sum we're looking for -/
def targetSum : ℕ := 15

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces^numDice

/-- The number of favorable outcomes (sum of 15) -/
def favorableOutcomes : ℕ := 7

/-- Theorem: The probability of rolling a sum of 15 with three standard 6-faced dice is 7/72 -/
theorem prob_sum_15_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 72 := by sorry

end NUMINAMATH_CALUDE_prob_sum_15_three_dice_l2909_290961


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l2909_290946

theorem smallest_k_with_remainder_one (k : ℕ) : k > 1 ∧ 
  k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 → k ≥ 400 := by
  sorry

theorem k_400_satisfies_conditions : 
  400 > 1 ∧ 400 % 19 = 1 ∧ 400 % 7 = 1 ∧ 400 % 3 = 1 := by
  sorry

theorem smallest_k_is_400 : 
  ∃! k : ℕ, k > 1 ∧ k % 19 = 1 ∧ k % 7 = 1 ∧ k % 3 = 1 ∧ 
  ∀ m : ℕ, (m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1) → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_k_400_satisfies_conditions_smallest_k_is_400_l2909_290946


namespace NUMINAMATH_CALUDE_total_chair_cost_l2909_290970

/-- Represents the cost calculation for chairs in a room -/
structure RoomChairs where
  count : Nat
  price : Nat

/-- Calculates the total cost for a set of room chairs -/
def totalCost (rc : RoomChairs) : Nat :=
  rc.count * rc.price

/-- Theorem: The total cost of chairs for the entire house is $2045 -/
theorem total_chair_cost (livingRoom kitchen diningRoom patio : RoomChairs)
    (h1 : livingRoom = ⟨3, 75⟩)
    (h2 : kitchen = ⟨6, 50⟩)
    (h3 : diningRoom = ⟨8, 100⟩)
    (h4 : patio = ⟨12, 60⟩) :
    totalCost livingRoom + totalCost kitchen + totalCost diningRoom + totalCost patio = 2045 := by
  sorry

#eval totalCost ⟨3, 75⟩ + totalCost ⟨6, 50⟩ + totalCost ⟨8, 100⟩ + totalCost ⟨12, 60⟩

end NUMINAMATH_CALUDE_total_chair_cost_l2909_290970


namespace NUMINAMATH_CALUDE_equation_solution_l2909_290956

theorem equation_solution :
  ∃! x : ℚ, x ≠ 0 ∧ x ≠ 2 ∧ (2 * x) / (x - 2) - 2 = 1 / (x * (x - 2)) ∧ x = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2909_290956


namespace NUMINAMATH_CALUDE_combined_travel_time_l2909_290922

/-- Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer
    than the car to cover the same distance, the combined time for both to reach station B
    is 11 hours. -/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 →
  train_time = car_time + 2 →
  car_time + train_time = 11 := by
sorry

end NUMINAMATH_CALUDE_combined_travel_time_l2909_290922


namespace NUMINAMATH_CALUDE_hospital_staff_count_l2909_290999

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h_total : total = 280)
  (h_ratio : doctor_ratio = 5 ∧ nurse_ratio = 9) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 180 :=
by sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l2909_290999


namespace NUMINAMATH_CALUDE_nh3_moles_produced_l2909_290928

structure Reaction where
  reactants : List (String × ℚ)
  products : List (String × ℚ)

def initial_moles : List (String × ℚ) := [
  ("NH4Cl", 3),
  ("KOH", 3),
  ("Na2CO3", 1),
  ("H3PO4", 1)
]

def reaction1 : Reaction := {
  reactants := [("NH4Cl", 2), ("Na2CO3", 1)],
  products := [("NH3", 2), ("CO2", 1), ("NaCl", 2), ("H2O", 1)]
}

def reaction2 : Reaction := {
  reactants := [("KOH", 2), ("H3PO4", 1)],
  products := [("K2HPO4", 1), ("H2O", 2)]
}

def limiting_reactant (reaction : Reaction) (available : List (String × ℚ)) : String :=
  sorry

def moles_produced (reaction : Reaction) (product : String) (limiting : String) : ℚ :=
  sorry

theorem nh3_moles_produced : 
  moles_produced reaction1 "NH3" (limiting_reactant reaction1 initial_moles) = 2 :=
sorry

end NUMINAMATH_CALUDE_nh3_moles_produced_l2909_290928


namespace NUMINAMATH_CALUDE_intersection_condition_l2909_290900

theorem intersection_condition (m : ℤ) : 
  let A : Set ℤ := {0, m}
  let B : Set ℤ := {n : ℤ | n^2 - 3*n < 0}
  (A ∩ B).Nonempty → m = 1 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_condition_l2909_290900


namespace NUMINAMATH_CALUDE_total_balloons_count_l2909_290997

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := 9

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have -/
def total_balloons : ℕ := tom_balloons + sara_balloons

theorem total_balloons_count : total_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l2909_290997


namespace NUMINAMATH_CALUDE_willow_count_l2909_290950

theorem willow_count (total : ℕ) (diff : ℕ) : 
  total = 83 →
  diff = 11 →
  ∃ (willows oaks : ℕ),
    willows + oaks = total ∧
    oaks = willows + diff ∧
    willows = 36 := by
  sorry

end NUMINAMATH_CALUDE_willow_count_l2909_290950


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l2909_290923

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.8 * first_sale_price
  final_price / original_price = 0.64 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l2909_290923


namespace NUMINAMATH_CALUDE_teacher_raise_percentage_l2909_290968

def former_salary : ℕ := 45000
def num_kids : ℕ := 9
def payment_per_kid : ℕ := 6000

def total_new_salary : ℕ := num_kids * payment_per_kid

def raise_amount : ℕ := total_new_salary - former_salary

def raise_percentage : ℚ := (raise_amount : ℚ) / (former_salary : ℚ) * 100

theorem teacher_raise_percentage :
  raise_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_teacher_raise_percentage_l2909_290968


namespace NUMINAMATH_CALUDE_snowfall_probability_l2909_290994

theorem snowfall_probability (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.3) :
  (1 - p_A) * (1 - p_B) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_probability_l2909_290994


namespace NUMINAMATH_CALUDE_combined_salary_proof_l2909_290930

/-- The combined salary of two people A and B -/
def combinedSalary (salaryA salaryB : ℝ) : ℝ := salaryA + salaryB

/-- The savings of a person given their salary and spending percentage -/
def savings (salary spendingPercentage : ℝ) : ℝ := salary * (1 - spendingPercentage)

theorem combined_salary_proof (salaryA salaryB : ℝ) 
  (hSpendA : savings salaryA 0.8 = savings salaryB 0.85)
  (hSalaryB : salaryB = 8000) :
  combinedSalary salaryA salaryB = 14000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salary_proof_l2909_290930


namespace NUMINAMATH_CALUDE_student_selection_count_l2909_290939

theorem student_selection_count (n m k : ℕ) (hn : n = 60) (hm : m = 2) (hk : k = 5) :
  (Nat.choose n k - Nat.choose (n - m) k : ℕ) =
  (Nat.choose m 1 * Nat.choose (n - 1) (k - 1) -
   Nat.choose m 2 * Nat.choose (n - m) (k - 2) : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_student_selection_count_l2909_290939


namespace NUMINAMATH_CALUDE_pentagon_area_increase_l2909_290905

/-- The increase in area when expanding a convex pentagon's boundary --/
theorem pentagon_area_increase (P s : ℝ) (h : P > 0) (h' : s > 0) :
  let increase := s * P + π * s^2
  increase = s * P + π * s^2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_increase_l2909_290905


namespace NUMINAMATH_CALUDE_sequence_product_l2909_290988

theorem sequence_product (n a : ℕ) :
  ∃ u v : ℕ, n / (n + a) = (u / (u + a)) * (v / (v + a)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_product_l2909_290988


namespace NUMINAMATH_CALUDE_train_length_is_200_l2909_290935

/-- The length of a train that crosses a 200-meter bridge in 10 seconds
    and passes a lamp post on the bridge in 5 seconds. -/
def train_length : ℝ := 200

/-- The length of the bridge in meters. -/
def bridge_length : ℝ := 200

/-- The time taken to cross the bridge in seconds. -/
def bridge_crossing_time : ℝ := 10

/-- The time taken to pass the lamp post in seconds. -/
def lamppost_passing_time : ℝ := 5

/-- Theorem stating that the train length is 200 meters given the conditions. -/
theorem train_length_is_200 :
  train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_200_l2909_290935


namespace NUMINAMATH_CALUDE_mark_payment_l2909_290903

def bread_cost : ℚ := 21/5
def cheese_cost : ℚ := 41/20
def nickel_value : ℚ := 1/20
def dime_value : ℚ := 1/10
def quarter_value : ℚ := 1/4
def num_nickels : ℕ := 8

theorem mark_payment (total_cost change payment : ℚ) :
  total_cost = bread_cost + cheese_cost →
  change = num_nickels * nickel_value + dime_value + quarter_value →
  payment = total_cost + change →
  payment = 7 := by sorry

end NUMINAMATH_CALUDE_mark_payment_l2909_290903


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bound_one_half_is_greatest_bound_l2909_290966

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_ratio_bound (t : Triangle) :
  (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 > 1 / 2 :=
sorry

theorem one_half_is_greatest_bound :
  ∀ ε > 0, ∃ t : Triangle, (t.a ^ 2 + t.b ^ 2) / t.c ^ 2 < 1 / 2 + ε :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bound_one_half_is_greatest_bound_l2909_290966


namespace NUMINAMATH_CALUDE_line_l_properties_l2909_290976

/-- Given a line l: (m-1)x + 2my + 2 = 0, where m is a real number -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x + 2 * m * y + 2 = 0

theorem line_l_properties (m : ℝ) :
  /- 1. Line l passes through the point (2, -1) -/
  line_l m 2 (-1) ∧
  /- 2. If the slope of l is non-positive and y-intercept is non-negative, then m ≤ 0 -/
  ((∀ x y : ℝ, line_l m x y → (1 - m) / (2 * m) ≤ 0 ∧ -1 / m ≥ 0) → m ≤ 0) ∧
  /- 3. When the x-intercept equals the y-intercept, m = -1 -/
  (∃ a : ℝ, a ≠ 0 ∧ line_l m a 0 ∧ line_l m 0 (-a)) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l2909_290976


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l2909_290936

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) : (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l2909_290936


namespace NUMINAMATH_CALUDE_elevator_exit_theorem_l2909_290983

/-- The number of ways 9 passengers can exit an elevator in groups of 2, 3, and 4 at any of 10 floors -/
def elevator_exit_ways : ℕ :=
  Nat.factorial 10 / Nat.factorial 4

/-- Theorem stating that the number of ways 9 passengers can exit an elevator
    in groups of 2, 3, and 4 at any of 10 floors is equal to 10! / 4! -/
theorem elevator_exit_theorem :
  elevator_exit_ways = Nat.factorial 10 / Nat.factorial 4 := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_theorem_l2909_290983


namespace NUMINAMATH_CALUDE_least_positive_period_is_30_l2909_290906

/-- A function satisfying the given condition -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬IsPeriod f q

theorem least_positive_period_is_30 :
  ∀ f : ℝ → ℝ, PeriodicFunction f → IsLeastPositivePeriod f 30 :=
sorry

end NUMINAMATH_CALUDE_least_positive_period_is_30_l2909_290906


namespace NUMINAMATH_CALUDE_intersection_range_value_range_on_curve_l2909_290980

-- Define the line l
def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | ∃ t, x = -2 + t * Real.cos α ∧ y = t * Real.sin α}

-- Define the curve C
def curve_C : Set (Real × Real) :=
  {(x, y) | (x - 2)^2 + y^2 = 4}

-- Theorem for part (I)
theorem intersection_range (α : Real) :
  (∃ p, p ∈ line_l α ∧ p ∈ curve_C) ↔ 
  (0 ≤ α ∧ α ≤ Real.pi/6) ∨ (5*Real.pi/6 ≤ α ∧ α ≤ Real.pi) :=
sorry

-- Theorem for part (II)
theorem value_range_on_curve :
  ∀ (x y : Real), (x, y) ∈ curve_C → -2 ≤ x + Real.sqrt 3 * y ∧ x + Real.sqrt 3 * y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_value_range_on_curve_l2909_290980


namespace NUMINAMATH_CALUDE_decimal_sum_and_product_l2909_290943

theorem decimal_sum_and_product :
  let sum := 0.5 + 0.03 + 0.007
  sum = 0.537 ∧ 3 * sum = 1.611 :=
by sorry

end NUMINAMATH_CALUDE_decimal_sum_and_product_l2909_290943


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2909_290945

theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (aₙ : ℝ) (n : ℕ) :
  a₁ = 2.5 →
  d = 4 →
  aₙ = 46.5 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2909_290945


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2909_290954

/-- Proves that the eccentricity of a hyperbola with specific properties is 2√3/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x : ℝ) ↦ b / a * x
  let F := (c, 0)
  let A := (c, b^2 / a)
  let B := (c, b * c / a)
  hyperbola c (b^2 / a) ∧ 
  A.1 = (F.1 + B.1) / 2 ∧ 
  A.2 = (F.2 + B.2) / 2 →
  c / a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2909_290954


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2909_290947

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)
variable (intersection_line : Plane → Plane → Line)

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perp_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line)
  (h1 : perpendicular_to_plane l1 p)
  (h2 : perpendicular_to_plane l2 p) :
  parallel l1 l2 :=
sorry

-- Theorem 2: In perpendicular planes, a line not perpendicular to the intersection
-- is not perpendicular to the other plane
theorem line_not_perp_to_intersection_not_perp_to_other_plane
  (p1 p2 : Plane) (l : Line)
  (h1 : planes_perpendicular p1 p2)
  (h2 : in_plane l p1)
  (h3 : ¬ perpendicular l (intersection_line p1 p2)) :
  ¬ perpendicular_to_plane l p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_line_not_perp_to_intersection_not_perp_to_other_plane_l2909_290947


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2909_290978

def U : Set ℤ := {x | -4 < x ∧ x < 4}
def A : Set ℤ := {-1, 0, 2, 3}
def B : Set ℤ := {-2, 0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {-3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2909_290978


namespace NUMINAMATH_CALUDE_cubic_root_from_quadratic_l2909_290925

theorem cubic_root_from_quadratic : ∀ r : ℝ, 
  (r^2 = r + 2) → (r^3 = 3*r + 2) ∧ (3 * 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_from_quadratic_l2909_290925


namespace NUMINAMATH_CALUDE_problem_statement_l2909_290929

theorem problem_statement :
  (∀ x : ℝ, (x + 8) * (x + 11) < (x + 9) * (x + 10)) ∧
  (Real.sqrt 5 - 2 > Real.sqrt 6 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2909_290929


namespace NUMINAMATH_CALUDE_statement_2_statement_3_l2909_290921

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 2
theorem statement_2 (m n : Line) (α : Plane) :
  parallel m α → perpendicular n α → perpendicular_lines n m :=
sorry

-- Statement 3
theorem statement_3 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_statement_2_statement_3_l2909_290921


namespace NUMINAMATH_CALUDE_complex_number_equality_l2909_290908

theorem complex_number_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2909_290908


namespace NUMINAMATH_CALUDE_blue_eyed_percentage_is_correct_l2909_290986

def cat_kittens : List (ℕ × ℕ) := [(5, 7), (6, 8), (4, 6), (7, 9), (3, 5)]

def total_blue_eyed : ℕ := (cat_kittens.map Prod.fst).sum

def total_kittens : ℕ := (cat_kittens.map (λ p => p.fst + p.snd)).sum

def blue_eyed_percentage : ℚ := (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100

theorem blue_eyed_percentage_is_correct : 
  blue_eyed_percentage = 125/3 := by sorry

end NUMINAMATH_CALUDE_blue_eyed_percentage_is_correct_l2909_290986


namespace NUMINAMATH_CALUDE_job_completion_time_l2909_290926

theorem job_completion_time (a_time b_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) : 
  a_time = 15 →
  combined_time = 8 →
  combined_work = 0.9333333333333333 →
  combined_work = combined_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2909_290926


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l2909_290984

/-- The function f(x) = x^4 - x --/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  P.1 = 1 ∧ P.2 = 0 ↔
    f P.1 = P.2 ∧ f' P.1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l2909_290984


namespace NUMINAMATH_CALUDE_red_rows_in_specific_grid_l2909_290912

/-- Represents the grid coloring problem -/
structure GridColoring where
  total_rows : ℕ
  squares_per_row : ℕ
  blue_rows : ℕ
  green_squares : ℕ
  red_squares_per_row : ℕ

/-- Calculates the number of red rows in the grid -/
def red_rows (g : GridColoring) : ℕ :=
  let total_squares := g.total_rows * g.squares_per_row
  let blue_squares := g.blue_rows * g.squares_per_row
  let red_squares := total_squares - blue_squares - g.green_squares
  red_squares / g.red_squares_per_row

/-- Theorem stating the number of red rows in the specific problem -/
theorem red_rows_in_specific_grid :
  let g : GridColoring := {
    total_rows := 10,
    squares_per_row := 15,
    blue_rows := 4,
    green_squares := 66,
    red_squares_per_row := 6
  }
  red_rows g = 4 := by sorry

end NUMINAMATH_CALUDE_red_rows_in_specific_grid_l2909_290912


namespace NUMINAMATH_CALUDE_rs_length_l2909_290919

/-- Triangle ABC with altitude CH, points R and S on CH -/
structure TriangleWithAltitude where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point H on the altitude CH -/
  H : ℝ × ℝ
  /-- Point R on CH, tangent point of inscribed circle in ACH -/
  R : ℝ × ℝ
  /-- Point S on CH, tangent point of inscribed circle in BCH -/
  S : ℝ × ℝ
  /-- CH is an altitude of triangle ABC -/
  altitude : (C.1 - H.1) * (B.1 - A.1) + (C.2 - H.2) * (B.2 - A.2) = 0
  /-- AB = 13 -/
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13
  /-- AC = 12 -/
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 12
  /-- BC = 5 -/
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5
  /-- R is on CH -/
  r_on_ch : ∃ t : ℝ, R = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  /-- S is on CH -/
  s_on_ch : ∃ t : ℝ, S = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))

/-- The main theorem to prove -/
theorem rs_length (t : TriangleWithAltitude) : 
  Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_rs_length_l2909_290919


namespace NUMINAMATH_CALUDE_square_of_a_l2909_290938

theorem square_of_a (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b ≤ c) (h3 : c < d)
  (h4 : a * d = b * c)
  (h5 : Real.sqrt d - Real.sqrt a ≤ 1) :
  ∃ (n : ℕ), a = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_a_l2909_290938


namespace NUMINAMATH_CALUDE_sin_five_pi_thirds_l2909_290973

theorem sin_five_pi_thirds : Real.sin (5 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_thirds_l2909_290973


namespace NUMINAMATH_CALUDE_length_PQ_is_sqrt_82_l2909_290967

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop :=
  y = 2*x

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = x - 1

-- Define the point M
def point_M : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem length_PQ_is_sqrt_82 :
  ∀ (P Q : ℝ × ℝ),
  circle_C (-2) 1 →  -- Point A on circle C
  circle_C 5 0 →     -- Point B on circle C
  (∃ (cx cy : ℝ), circle_C cx cy ∧ line_center cx cy) →  -- Center of C on y = 2x
  line_m P.1 P.2 →   -- P is on line m
  line_m Q.1 Q.2 →   -- Q is on line m
  circle_C P.1 P.2 → -- P is on circle C
  circle_C Q.1 Q.2 → -- Q is on circle C
  line_m point_M.1 point_M.2 →  -- M is on line m
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_length_PQ_is_sqrt_82_l2909_290967


namespace NUMINAMATH_CALUDE_region_area_l2909_290948

/-- The region in the plane defined by |x + 2y| + |x - 2y| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 + 2*p.2| + |p.1 - 2*p.2| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem region_area : area Region = 9 := by
  sorry

end NUMINAMATH_CALUDE_region_area_l2909_290948


namespace NUMINAMATH_CALUDE_probability_red_ball_is_four_fifths_l2909_290917

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The experiment setup -/
def experiment : List Container :=
  [{ red := 5, green := 5 },  -- Container A
   { red := 7, green := 3 },  -- Container B
   { red := 7, green := 3 }]  -- Container C

/-- The probability of selecting a red ball in the described experiment -/
def probability_red_ball : ℚ :=
  (experiment.map (fun c => c.red / (c.red + c.green))).sum / experiment.length

theorem probability_red_ball_is_four_fifths :
  probability_red_ball = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_is_four_fifths_l2909_290917


namespace NUMINAMATH_CALUDE_adams_initial_money_l2909_290909

/-- Adam's initial money problem -/
theorem adams_initial_money :
  ∀ (x : ℤ), (x - 2 + 5 = 8) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_adams_initial_money_l2909_290909


namespace NUMINAMATH_CALUDE_polynomial_ascending_powers_x_l2909_290964

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^3 - 5*x*y^2 - 7*y^3 + 8*x^2*y

-- Define a function to extract the degree of x in a term
def degree_x (term : ℝ → ℝ → ℝ) : ℕ :=
  sorry  -- Implementation details omitted

-- Define the ascending order of terms with respect to x
def ascending_order_x (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  degree_x term1 ≤ degree_x term2

-- State the theorem
theorem polynomial_ascending_powers_x :
  ∃ (term1 term2 term3 term4 : ℝ → ℝ → ℝ),
    (∀ x y, p x y = term1 x y + term2 x y + term3 x y + term4 x y) ∧
    (ascending_order_x term1 term2) ∧
    (ascending_order_x term2 term3) ∧
    (ascending_order_x term3 term4) ∧
    (∀ x y, term1 x y = -7*y^3) ∧
    (∀ x y, term2 x y = -5*x*y^2) ∧
    (∀ x y, term3 x y = 8*x^2*y) ∧
    (∀ x y, term4 x y = x^3) :=
  sorry

end NUMINAMATH_CALUDE_polynomial_ascending_powers_x_l2909_290964


namespace NUMINAMATH_CALUDE_contrapositive_equality_l2909_290959

theorem contrapositive_equality (a b : ℝ) : 
  (¬(|a| = |b|) → ¬(a = -b)) ↔ (a = -b → |a| = |b|) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equality_l2909_290959


namespace NUMINAMATH_CALUDE_initial_men_count_l2909_290942

theorem initial_men_count (initial_days : ℝ) (additional_men : ℕ) (final_days : ℝ) :
  initial_days = 18 →
  additional_men = 450 →
  final_days = 13.090909090909092 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2909_290942


namespace NUMINAMATH_CALUDE_primitive_roots_existence_l2909_290915

theorem primitive_roots_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x : Nat, IsPrimitiveRoot x p ∧ IsPrimitiveRoot (4 * x) p :=
sorry

end NUMINAMATH_CALUDE_primitive_roots_existence_l2909_290915


namespace NUMINAMATH_CALUDE_major_axis_length_l2909_290969

def ellipse_equation (x y : ℝ) : Prop := y^2 / 25 + x^2 / 15 = 1

theorem major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  max a b = 5 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_l2909_290969


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2909_290918

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2909_290918


namespace NUMINAMATH_CALUDE_function_properties_l2909_290974

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

/-- The main theorem stating the properties of the function -/
theorem function_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2909_290974


namespace NUMINAMATH_CALUDE_rectangle_equal_angles_l2909_290990

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ  -- width
  b : ℝ  -- height
  pos_a : 0 < a
  pos_b : 0 < b

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Theorem: The set of points P between parallel lines AB and CD of a rectangle
    such that ∠APB = ∠CPD is the line y = b/2 -/
theorem rectangle_equal_angles (rect : Rectangle) :
  ∀ P : Point,
    0 ≤ P.y ∧ P.y ≤ rect.b →
    (angle ⟨0, 0⟩ P ⟨rect.a, 0⟩ = angle ⟨rect.a, rect.b⟩ P ⟨0, rect.b⟩) ↔
    P.y = rect.b / 2 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_equal_angles_l2909_290990


namespace NUMINAMATH_CALUDE_perpendicular_edges_count_l2909_290907

/-- A cube is a three-dimensional shape with 6 square faces -/
structure Cube where
  -- Add necessary fields here

/-- An edge of a cube -/
structure Edge (c : Cube) where
  -- Add necessary fields here

/-- Predicate to check if two edges are perpendicular -/
def perpendicular (c : Cube) (e1 e2 : Edge c) : Prop :=
  sorry

theorem perpendicular_edges_count (c : Cube) (e : Edge c) :
  (∃ (s : Finset (Edge c)), s.card = 8 ∧ ∀ e' ∈ s, perpendicular c e e') ∧
  ¬∃ (s : Finset (Edge c)), s.card > 8 ∧ ∀ e' ∈ s, perpendicular c e e' :=
sorry

end NUMINAMATH_CALUDE_perpendicular_edges_count_l2909_290907


namespace NUMINAMATH_CALUDE_rooms_already_painted_l2909_290991

/-- Given a painting job with the following parameters:
  * total_rooms: The total number of rooms to be painted
  * hours_per_room: The number of hours it takes to paint one room
  * remaining_hours: The number of hours left to complete the job
  This theorem proves that the number of rooms already painted is equal to
  the total number of rooms minus the number of rooms that can be painted
  in the remaining time. -/
theorem rooms_already_painted
  (total_rooms : ℕ)
  (hours_per_room : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_rooms = 10)
  (h2 : hours_per_room = 8)
  (h3 : remaining_hours = 16) :
  total_rooms - (remaining_hours / hours_per_room) = 8 := by
  sorry

end NUMINAMATH_CALUDE_rooms_already_painted_l2909_290991
