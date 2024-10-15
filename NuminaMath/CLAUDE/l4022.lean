import Mathlib

namespace NUMINAMATH_CALUDE_average_words_per_page_l4022_402265

/-- Proves that for a book with given specifications, the average number of words per page is 1250 --/
theorem average_words_per_page
  (sheets : ℕ)
  (total_words : ℕ)
  (pages_per_sheet : ℕ)
  (h1 : sheets = 12)
  (h2 : total_words = 240000)
  (h3 : pages_per_sheet = 16) :
  total_words / (sheets * pages_per_sheet) = 1250 :=
by sorry

end NUMINAMATH_CALUDE_average_words_per_page_l4022_402265


namespace NUMINAMATH_CALUDE_prime_sum_squares_divisibility_l4022_402267

theorem prime_sum_squares_divisibility (p : ℕ) (h1 : Nat.Prime p) 
  (h2 : ∃ k : ℕ, 3 * p + 10 = (k^2 + (k+1)^2 + (k+2)^2 + (k+3)^2 + (k+4)^2 + (k+5)^2)) :
  36 ∣ (p - 7) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_divisibility_l4022_402267


namespace NUMINAMATH_CALUDE_threeTangentLines_l4022_402251

/-- Represents a circle in the 2D plane --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The first circle: x^2 + y^2 + 4x - 4y + 7 = 0 --/
def circle1 : Circle := { a := 1, b := 1, c := 4, d := -4, e := 7 }

/-- The second circle: x^2 + y^2 - 4x - 10y + 13 = 0 --/
def circle2 : Circle := { a := 1, b := 1, c := -4, d := -10, e := 13 }

/-- Count the number of lines tangent to both circles --/
def countTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 lines tangent to both circles --/
theorem threeTangentLines : countTangentLines circle1 circle2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_threeTangentLines_l4022_402251


namespace NUMINAMATH_CALUDE_badge_exchange_l4022_402233

theorem badge_exchange (t : ℕ) (v : ℕ) : 
  v = t + 5 →
  (v - (24 * v) / 100 + (20 * t) / 100) + 1 = (t - (20 * t) / 100 + (24 * v) / 100) →
  t = 45 ∧ v = 50 :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l4022_402233


namespace NUMINAMATH_CALUDE_efqs_equals_qrst_l4022_402295

/-- Assigns a value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- Calculates the product of values assigned to a list of characters -/
def list_product (s : List Char) : ℕ :=
  s.map letter_value |>.foldl (·*·) 1

/-- Checks if a list of characters contains distinct elements -/
def distinct_chars (s : List Char) : Prop :=
  s.toFinset.card = s.length

theorem efqs_equals_qrst : ∃ (e f q s : Char), 
  distinct_chars ['E', 'F', 'Q', 'S'] ∧
  list_product ['E', 'F', 'Q', 'S'] = list_product ['Q', 'R', 'S', 'T'] :=
by sorry

end NUMINAMATH_CALUDE_efqs_equals_qrst_l4022_402295


namespace NUMINAMATH_CALUDE_mobile_purchase_price_l4022_402297

def grinder_price : ℝ := 15000
def grinder_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def total_profit : ℝ := 200

theorem mobile_purchase_price (mobile_price : ℝ) : 
  (grinder_price * (1 - grinder_loss_percent) + mobile_price * (1 + mobile_profit_percent)) - 
  (grinder_price + mobile_price) = total_profit → 
  mobile_price = 8000 := by
sorry

end NUMINAMATH_CALUDE_mobile_purchase_price_l4022_402297


namespace NUMINAMATH_CALUDE_halfway_point_fractions_l4022_402202

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/12) (hb : b = 13/12) :
  (a + b) / 2 = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_fractions_l4022_402202


namespace NUMINAMATH_CALUDE_equation_solutions_l4022_402212

def solutions : Set (ℤ × ℤ) := {(-13,-2), (-4,-1), (-1,0), (2,3), (3,6), (4,15), (6,-21), (7,-12), (8,-9), (11,-6), (14,-5), (23,-4)}

theorem equation_solutions :
  ∀ (x y : ℤ), (x * y + 3 * x - 5 * y = -3) ↔ (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4022_402212


namespace NUMINAMATH_CALUDE_student_grade_problem_l4022_402271

theorem student_grade_problem (courses_last_year : ℕ) (courses_year_before : ℕ) 
  (avg_grade_year_before : ℚ) (avg_grade_two_years : ℚ) :
  courses_last_year = 6 →
  courses_year_before = 5 →
  avg_grade_year_before = 40 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + 
   courses_last_year * (592 : ℚ) / 6) / (courses_year_before + courses_last_year) = 
  avg_grade_two_years :=
by sorry

end NUMINAMATH_CALUDE_student_grade_problem_l4022_402271


namespace NUMINAMATH_CALUDE_marbles_fraction_l4022_402260

theorem marbles_fraction (total_marbles : ℕ) (marbles_taken : ℕ) :
  total_marbles = 100 →
  marbles_taken = 11 →
  (marbles_taken : ℚ) / (total_marbles : ℚ) = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_marbles_fraction_l4022_402260


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l4022_402254

/-- Theorem: For a hyperbola passing through the point (4, √3) with asymptotes y = ± (1/2)x, 
    its standard equation is x²/4 - y² = 1 -/
theorem hyperbola_standard_equation 
  (passes_through : (4 : ℝ)^2 / 4 - 3 = 1) 
  (asymptotes : ∀ (x y : ℝ), y = (1/2) * x ∨ y = -(1/2) * x) :
  ∀ (x y : ℝ), x^2 / 4 - y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l4022_402254


namespace NUMINAMATH_CALUDE_rectangle_in_circle_l4022_402223

/-- A rectangle with sides 7 cm and 24 cm is inscribed in a circle. -/
theorem rectangle_in_circle (a b r : ℝ) (h1 : a = 7) (h2 : b = 24) 
  (h3 : a^2 + b^2 = (2*r)^2) : 
  (2 * π * r = 25 * π) ∧ (a * b = 168) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_l4022_402223


namespace NUMINAMATH_CALUDE_rectangle_area_l4022_402298

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (width length : ℝ), 
    width > 0 ∧ 
    length = 2 * width ∧ 
    x^2 = width^2 + length^2 ∧ 
    width * length = (2/5) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4022_402298


namespace NUMINAMATH_CALUDE_complex_product_theorem_l4022_402229

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 3)
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) :
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l4022_402229


namespace NUMINAMATH_CALUDE_roots_relation_l4022_402249

-- Define the polynomials f and g
def f (x : ℝ) := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ r : ℝ, f r = 0 → g (r^2) b c d = 0) →
  b = -2 ∧ c = 1 ∧ d = -12 :=
sorry

end NUMINAMATH_CALUDE_roots_relation_l4022_402249


namespace NUMINAMATH_CALUDE_odd_integer_dividing_power_plus_one_l4022_402276

theorem odd_integer_dividing_power_plus_one (n : ℕ) : 
  n ≥ 1 → 
  Odd n → 
  (n ∣ 3^n + 1) → 
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_dividing_power_plus_one_l4022_402276


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l4022_402278

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 24 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 21 ∧
      (team_size : ℚ) * team_average_age = 
        captain_age + (captain_age + wicket_keeper_age_diff) + 
        ((team_size - 2) : ℚ) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l4022_402278


namespace NUMINAMATH_CALUDE_limit_f_at_origin_l4022_402222

/-- The function f(x, y) = (x^2 + y^2)^2 x^2 y^2 -/
def f (x y : ℝ) : ℝ := (x^2 + y^2)^2 * x^2 * y^2

/-- The limit of f(x, y) as x and y approach 0 is 1 -/
theorem limit_f_at_origin :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 → |f x y - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_f_at_origin_l4022_402222


namespace NUMINAMATH_CALUDE_fifth_term_value_l4022_402285

/-- A geometric sequence with positive terms satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  first_second_sum : a 1 + 2 * a 2 = 4
  fourth_squared : a 4 ^ 2 = 4 * a 3 * a 7

/-- The fifth term of the geometric sequence is 1/8 -/
theorem fifth_term_value (seq : GeometricSequence) : seq.a 5 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l4022_402285


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l4022_402203

/-- Given a set of observations with known properties, calculate the incorrect value. -/
theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 34) : 
  ∃ (incorrect_value : ℝ), 
    incorrect_value = n * new_mean - (n - 1) * original_mean - correct_value + n * (new_mean - original_mean) :=
by
  sorry

#check incorrect_observation_value

end NUMINAMATH_CALUDE_incorrect_observation_value_l4022_402203


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l4022_402258

theorem greatest_power_under_500 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 500 → (∀ x y : ℕ, x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) → a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l4022_402258


namespace NUMINAMATH_CALUDE_max_average_raise_l4022_402248

theorem max_average_raise (R S C A : ℝ) : 
  0.05 < R ∧ R < 0.10 →
  0.07 < S ∧ S < 0.12 →
  0.04 < C ∧ C < 0.09 →
  0.06 < A ∧ A < 0.15 →
  (R + S + C + A) / 4 ≤ 0.085 →
  ∃ (R' S' C' A' : ℝ),
    0.05 < R' ∧ R' < 0.10 ∧
    0.07 < S' ∧ S' < 0.12 ∧
    0.04 < C' ∧ C' < 0.09 ∧
    0.06 < A' ∧ A' < 0.15 ∧
    (R' + S' + C' + A') / 4 = 0.085 :=
by sorry

end NUMINAMATH_CALUDE_max_average_raise_l4022_402248


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_div_by_five_l4022_402255

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of four consecutive primes starting from the nth prime -/
def sumFourConsecutivePrimes (n : ℕ) : ℕ :=
  nthPrime n + nthPrime (n + 1) + nthPrime (n + 2) + nthPrime (n + 3)

/-- The main theorem -/
theorem smallest_sum_four_consecutive_primes_div_by_five :
  ∀ n : ℕ, sumFourConsecutivePrimes n % 5 = 0 → sumFourConsecutivePrimes n ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_primes_div_by_five_l4022_402255


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l4022_402210

theorem square_perimeter_problem (perimeter_A perimeter_B : ℝ) 
  (h1 : perimeter_A = 20)
  (h2 : perimeter_B = 40)
  (h3 : ∀ (side_A side_B : ℝ), 
    perimeter_A = 4 * side_A → 
    perimeter_B = 4 * side_B → 
    ∃ (perimeter_C : ℝ), perimeter_C = 4 * (side_A + side_B)) :
  ∃ (perimeter_C : ℝ), perimeter_C = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l4022_402210


namespace NUMINAMATH_CALUDE_smallest_n_is_83_l4022_402293

def candy_problem (money : ℕ) : Prop :=
  ∃ (r g b : ℕ),
    money = 18 * r ∧
    money = 20 * g ∧
    money = 22 * b ∧
    money = 24 * 83 ∧
    ∀ (n : ℕ), n < 83 → money ≠ 24 * n

theorem smallest_n_is_83 :
  ∃ (money : ℕ), candy_problem money :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_83_l4022_402293


namespace NUMINAMATH_CALUDE_bob_candies_l4022_402207

/-- Given that Jennifer bought twice as many candies as Emily, Jennifer bought three times as many
    candies as Bob, and Emily bought 6 candies, prove that Bob bought 4 candies. -/
theorem bob_candies (emily_candies : ℕ) (jennifer_candies : ℕ) (bob_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : emily_candies = 6) :
  bob_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_candies_l4022_402207


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l4022_402242

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x + 3

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + a

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x + 1  -- We use b = 1 as it's not relevant for finding a

-- Theorem statement
theorem tangent_line_implies_a_value :
  ∀ a : ℝ, (curve_derivative a 1 = 1) → a = -3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l4022_402242


namespace NUMINAMATH_CALUDE_part1_part2_l4022_402280

-- Define the quadratic expression
def quadratic (a x : ℝ) : ℝ := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- Part 1
theorem part1 : 
  ∀ x : ℝ, quadratic (-2) x < 0 ↔ x ≠ -1 :=
sorry

-- Part 2
theorem part2 : 
  (∀ x : ℝ, quadratic a x < 0) ↔ a ∈ Set.Ioc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l4022_402280


namespace NUMINAMATH_CALUDE_mixture_composition_l4022_402264

/-- Represents the composition of a solution --/
structure Solution :=
  (a : ℝ)  -- Percentage of chemical A
  (b : ℝ)  -- Percentage of chemical B
  (sum_to_100 : a + b = 100)

/-- The problem statement --/
theorem mixture_composition 
  (X : Solution)
  (Y : Solution)
  (Z : Solution)
  (h_X : X.a = 40)
  (h_Y : Y.a = 50)
  (h_Z : Z.a = 30)
  : ∃ (x y z : ℝ),
    x + y + z = 100 ∧
    x * X.a / 100 + y * Y.a / 100 + z * Z.a / 100 = 46 ∧
    x = 40 ∧ y = 60 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l4022_402264


namespace NUMINAMATH_CALUDE_wall_bricks_count_l4022_402227

def wall_problem (initial_courses : ℕ) (bricks_per_course : ℕ) (added_courses : ℕ) : ℕ :=
  let total_courses := initial_courses + added_courses
  let total_bricks := total_courses * bricks_per_course
  let removed_bricks := bricks_per_course / 2
  total_bricks - removed_bricks

theorem wall_bricks_count :
  wall_problem 3 400 2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l4022_402227


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l4022_402246

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 78 →
  fourth_test_score = 86 →
  (3 * initial_average + fourth_test_score) / 4 - initial_average = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l4022_402246


namespace NUMINAMATH_CALUDE_equation_roots_l4022_402263

theorem equation_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -x₂) ∧ 
  (x₁ = Real.sqrt 2 ∨ x₁ = -Real.sqrt 2) ∧
  (x₂ = Real.sqrt 2 ∨ x₂ = -Real.sqrt 2) ∧
  (x₃ = 1/2) ∧
  (2 * x₁^5 - x₁^4 - 2 * x₁^3 + x₁^2 - 4 * x₁ + 2 = 0) ∧
  (2 * x₂^5 - x₂^4 - 2 * x₂^3 + x₂^2 - 4 * x₂ + 2 = 0) ∧
  (2 * x₃^5 - x₃^4 - 2 * x₃^3 + x₃^2 - 4 * x₃ + 2 = 0) := by
  sorry

#check equation_roots

end NUMINAMATH_CALUDE_equation_roots_l4022_402263


namespace NUMINAMATH_CALUDE_range_of_m_l4022_402228

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4022_402228


namespace NUMINAMATH_CALUDE_orthogonal_circles_on_radical_axis_l4022_402282

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  c1.radius^2 = (x1 - x2)^2 + (y1 - y2)^2 - c2.radius^2

-- Define the radical axis
def on_radical_axis (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  let (x, y) := p
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x - x1)^2 + (y - y1)^2 - c1.radius^2 = (x - x2)^2 + (y - y2)^2 - c2.radius^2

-- Main theorem
theorem orthogonal_circles_on_radical_axis (S1 S2 : Circle) (O : ℝ × ℝ) :
  (∃ r : ℝ, r > 0 ∧ is_orthogonal ⟨O, r⟩ S1 ∧ is_orthogonal ⟨O, r⟩ S2) ↔
  (on_radical_axis O S1 S2 ∧ O ≠ S1.center ∧ O ≠ S2.center) :=
sorry

end NUMINAMATH_CALUDE_orthogonal_circles_on_radical_axis_l4022_402282


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l4022_402262

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l4022_402262


namespace NUMINAMATH_CALUDE_product_divisibility_l4022_402220

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (b + c) * (b + a) = k * (a + c)) ∧
  (∃ k : ℤ, (c + a) * (c + b) = k * (a + b)) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l4022_402220


namespace NUMINAMATH_CALUDE_fencing_requirement_l4022_402284

/-- A rectangular field with specific properties. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- The theorem stating the fencing requirement for the given field. -/
theorem fencing_requirement (field : RectangularField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 80)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  2 * field.width + field.uncovered_side = 97 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l4022_402284


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_leq_4_l4022_402214

/-- The quadratic function f(x) = x^2 + 4x + a has a real root implies a ≤ 4 -/
theorem quadratic_root_implies_a_leq_4 (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + a = 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_leq_4_l4022_402214


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l4022_402259

theorem norm_scalar_multiple (v : ℝ × ℝ) :
  ‖v‖ = 7 → ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l4022_402259


namespace NUMINAMATH_CALUDE_fourth_roll_three_prob_l4022_402257

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_three_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the event of rolling three threes in a row
def three_threes_event : Prop := True

-- Theorem statement
theorem fourth_roll_three_prob :
  three_threes_event →
  (die_selection_prob * fair_die_prob^3 * fair_die_prob +
   die_selection_prob * biased_die_three_prob^3 * biased_die_three_prob) /
  (die_selection_prob * fair_die_prob^3 +
   die_selection_prob * biased_die_three_prob^3) = 41 / 84 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_three_prob_l4022_402257


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4022_402289

/-- A quadratic equation that is divisible by (x - 1) and has a constant term of 2 -/
def quadratic_equation (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem quadratic_equation_properties : 
  (∃ (q : ℝ → ℝ), ∀ x, quadratic_equation x = (x - 1) * q x) ∧ 
  (quadratic_equation 0 = 2) := by
  sorry

#check quadratic_equation_properties

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4022_402289


namespace NUMINAMATH_CALUDE_value_of_3a_plus_6b_l4022_402243

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3a_plus_6b_l4022_402243


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l4022_402281

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The distance between intersections of a function with a horizontal line -/
def IntersectionDistance (f : ℝ → ℝ) (y : ℝ) : ℝ → ℝ → ℝ :=
  λ x₁ x₂ => |x₂ - x₁|

theorem monotone_increasing_interval
  (ω φ : ℝ)
  (f : ℝ → ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hf : f = λ x => 2 * Real.sin (ω * x + φ))
  (heven : EvenFunction f)
  (hmin : ∃ x₁ x₂, f x₁ = 2 ∧ f x₂ = 2 ∧ 
    ∀ y₁ y₂, f y₁ = 2 → f y₂ = 2 → 
    IntersectionDistance f 2 y₁ y₂ ≥ IntersectionDistance f 2 x₁ x₂ ∧
    IntersectionDistance f 2 x₁ x₂ = π) :
  StrictMonoOn f (Set.Ioo (-π/2) (-π/4)) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l4022_402281


namespace NUMINAMATH_CALUDE_linear_function_theorem_l4022_402279

theorem linear_function_theorem (k b : ℝ) :
  (∃ (x y : ℝ), y = k * x + b ∧ x = 0 ∧ y = -2) →
  (1/2 * |2/k| * 2 = 3) →
  ((k = 2/3 ∧ b = -2) ∨ (k = -2/3 ∧ b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l4022_402279


namespace NUMINAMATH_CALUDE_min_A_mats_l4022_402294

/-- Represents the purchase and sale of bamboo mats -/
structure BambooMatSale where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  sale_price_A : ℝ
  sale_price_B : ℝ

/-- The conditions of the bamboo mat sale problem -/
def bamboo_mat_conditions (s : BambooMatSale) : Prop :=
  10 * s.purchase_price_A + 15 * s.purchase_price_B = 3600 ∧
  25 * s.purchase_price_A + 30 * s.purchase_price_B = 8100 ∧
  s.sale_price_A = 260 ∧
  s.sale_price_B = 180

/-- The profit calculation for a given number of mats A -/
def profit (s : BambooMatSale) (num_A : ℝ) : ℝ :=
  (s.sale_price_A - s.purchase_price_A) * num_A +
  (s.sale_price_B - s.purchase_price_B) * (60 - num_A)

/-- The main theorem stating the minimum number of A mats to purchase -/
theorem min_A_mats (s : BambooMatSale) 
  (h : bamboo_mat_conditions s) : 
  ∃ (n : ℕ), n = 40 ∧ 
  (∀ (m : ℕ), m ≥ 40 → profit s m ≥ 4400) ∧
  (∀ (m : ℕ), m < 40 → profit s m < 4400) := by
  sorry

end NUMINAMATH_CALUDE_min_A_mats_l4022_402294


namespace NUMINAMATH_CALUDE_orange_count_after_changes_l4022_402230

/-- The number of oranges in a bin after removing some and adding new ones. -/
def oranges_in_bin (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that starting with 50 oranges, removing 40, and adding 24 results in 34 oranges. -/
theorem orange_count_after_changes : oranges_in_bin 50 40 24 = 34 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_after_changes_l4022_402230


namespace NUMINAMATH_CALUDE_right_triangle_check_l4022_402283

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem right_triangle_check :
  ¬ is_right_triangle 1 3 4 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 1 (Real.sqrt 3) ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_l4022_402283


namespace NUMINAMATH_CALUDE_g_range_l4022_402213

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 5 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 9) / (Real.cos x - 1)

theorem g_range (x : ℝ) (h : Real.cos x ≠ 1) : 
  6 ≤ g x ∧ g x < 12 := by sorry

end NUMINAMATH_CALUDE_g_range_l4022_402213


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4022_402247

/-- Given a geometric sequence {a_n} with a_1 = 1, a_2 = 2, and a_3 = 4, prove that a_6 = 32 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n) : 
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4022_402247


namespace NUMINAMATH_CALUDE_gpa_probability_at_least_3_6_l4022_402252

/-- Represents the possible grades a student can receive. -/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value. -/
def gradeToPoints (g : Grade) : ℕ :=
  match g with
  | Grade.A => 4
  | Grade.B => 3
  | Grade.C => 2
  | Grade.D => 1

/-- Calculates the GPA given a list of grades. -/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 5

/-- Represents the probability distribution of grades for a class. -/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades. -/
def englishProb : GradeProbability :=
  { probA := 1/4, probB := 1/3, probC := 5/12, probD := 0 }

/-- The probability distribution for History grades. -/
def historyProb : GradeProbability :=
  { probA := 1/3, probB := 1/4, probC := 5/12, probD := 0 }

/-- Theorem stating the probability of achieving a GPA of at least 3.6. -/
theorem gpa_probability_at_least_3_6 :
  let allGrades := [Grade.A, Grade.A, Grade.A] -- Math, Science, Art
  let probAtLeast3_6 := (
    englishProb.probA * historyProb.probA +
    englishProb.probA * historyProb.probB +
    englishProb.probB * historyProb.probA +
    englishProb.probB * historyProb.probB
  )
  probAtLeast3_6 = 49/144 := by sorry

end NUMINAMATH_CALUDE_gpa_probability_at_least_3_6_l4022_402252


namespace NUMINAMATH_CALUDE_expression_evaluation_l4022_402208

theorem expression_evaluation : 
  -14 - (-2)^3 * (1/4) - 16 * ((1/2) - (1/4) + (3/8)) = -22 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4022_402208


namespace NUMINAMATH_CALUDE_cubic_value_in_set_l4022_402206

theorem cubic_value_in_set (A : Set ℝ) (a : ℝ) 
  (h1 : 5 ∈ A) 
  (h2 : a^2 + 2*a + 4 ∈ A) 
  (h3 : 7 ∈ A) : 
  a^3 = 1 ∨ a^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_cubic_value_in_set_l4022_402206


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4022_402277

theorem complex_modulus_problem (z : ℂ) (h : (1 - 2*I)*z = 5*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4022_402277


namespace NUMINAMATH_CALUDE_sangita_flying_hours_l4022_402288

/-- Calculates the required flying hours per month to meet a pilot certification goal -/
def required_hours_per_month (total_required : ℕ) (day_hours : ℕ) (night_hours : ℕ) (cross_country_hours : ℕ) (months : ℕ) : ℕ :=
  (total_required - (day_hours + night_hours + cross_country_hours)) / months

/-- Proves that Sangita needs to fly 220 hours per month to meet her goal -/
theorem sangita_flying_hours : 
  required_hours_per_month 1500 50 9 121 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sangita_flying_hours_l4022_402288


namespace NUMINAMATH_CALUDE_peter_banana_purchase_l4022_402274

def problem (initial_amount : ℕ) 
            (potato_price potato_quantity : ℕ)
            (tomato_price tomato_quantity : ℕ)
            (cucumber_price cucumber_quantity : ℕ)
            (banana_price : ℕ)
            (remaining_amount : ℕ) : Prop :=
  let potato_cost := potato_price * potato_quantity
  let tomato_cost := tomato_price * tomato_quantity
  let cucumber_cost := cucumber_price * cucumber_quantity
  let total_cost := potato_cost + tomato_cost + cucumber_cost
  let banana_cost := initial_amount - remaining_amount - total_cost
  banana_cost / banana_price = 14

theorem peter_banana_purchase :
  problem 500 2 6 3 9 4 5 5 426 := by
  sorry

end NUMINAMATH_CALUDE_peter_banana_purchase_l4022_402274


namespace NUMINAMATH_CALUDE_side_altitude_inequality_l4022_402269

/-- Triangle ABC with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  hₐ : ℝ
  hb : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_hₐ : 0 < hₐ
  pos_hb : 0 < hb

/-- Theorem: In a triangle, a ≥ b if and only if a + hₐ ≥ b + hb -/
theorem side_altitude_inequality (t : Triangle) : t.a ≥ t.b ↔ t.a + t.hₐ ≥ t.b + t.hb := by
  sorry

end NUMINAMATH_CALUDE_side_altitude_inequality_l4022_402269


namespace NUMINAMATH_CALUDE_new_student_weight_l4022_402286

theorem new_student_weight (n : ℕ) (w_avg_initial w_avg_final w_new : ℝ) :
  n = 29 →
  w_avg_initial = 28 →
  w_avg_final = 27.2 →
  (n : ℝ) * w_avg_initial = ((n : ℝ) + 1) * w_avg_final - w_new →
  w_new = 4 := by sorry

end NUMINAMATH_CALUDE_new_student_weight_l4022_402286


namespace NUMINAMATH_CALUDE_pipe_fill_time_pipe_B_fill_time_l4022_402287

/-- Given two pipes A and B that can fill a tank, this theorem proves the time it takes for pipe B to fill the tank. -/
theorem pipe_fill_time (fill_time_A : ℝ) (fill_time_both : ℝ) (fill_amount : ℝ) : ℝ :=
  let fill_rate_A := 1 / fill_time_A
  let fill_rate_both := fill_amount / fill_time_both
  let fill_rate_B := fill_rate_both - fill_rate_A
  1 / fill_rate_B

/-- The main theorem that proves the time it takes for pipe B to fill the tank under the given conditions. -/
theorem pipe_B_fill_time : pipe_fill_time 16 12.000000000000002 (5/4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_pipe_B_fill_time_l4022_402287


namespace NUMINAMATH_CALUDE_decimal_has_three_digits_l4022_402218

-- Define the decimal number
def decimal : ℚ := 0.049

-- Theorem stating that the decimal has 3 digits after the decimal point
theorem decimal_has_three_digits : 
  (decimal * 1000).num % 1000 ≠ 0 ∧ (decimal * 100).num % 100 = 0 :=
sorry

end NUMINAMATH_CALUDE_decimal_has_three_digits_l4022_402218


namespace NUMINAMATH_CALUDE_milk_cartons_per_stack_l4022_402234

theorem milk_cartons_per_stack (total_cartons : ℕ) (num_stacks : ℕ) 
  (h1 : total_cartons = 799)
  (h2 : num_stacks = 133)
  (h3 : total_cartons % num_stacks = 0) :
  total_cartons / num_stacks = 6 := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_per_stack_l4022_402234


namespace NUMINAMATH_CALUDE_keanu_refills_l4022_402236

/-- Calculates the number of refills needed for a round trip given the tank capacity, fuel consumption rate, and one-way distance. -/
def refills_needed (tank_capacity : ℚ) (consumption_per_40_miles : ℚ) (one_way_distance : ℚ) : ℚ :=
  let consumption_per_mile := consumption_per_40_miles / 40
  let round_trip_distance := one_way_distance * 2
  let total_consumption := round_trip_distance * consumption_per_mile
  (total_consumption / tank_capacity).ceil

/-- Theorem stating that for the given conditions, 14 refills are needed. -/
theorem keanu_refills :
  refills_needed 8 8 280 = 14 := by
  sorry

end NUMINAMATH_CALUDE_keanu_refills_l4022_402236


namespace NUMINAMATH_CALUDE_complex_equality_l4022_402226

theorem complex_equality (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 * Complex.I)
  Complex.re z = Complex.im z → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l4022_402226


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l4022_402237

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 ≥ 0 → d2 ≥ 0 → (d1 + d2) / 2 = (d1 / 1 + d2 / 1) / 2 := by
  sorry

/-- The average speed of a car traveling 10 km in the first hour and 60 km in the second hour is 35 km/h -/
theorem car_average_speed : 
  let d1 : ℝ := 10  -- Distance traveled in the first hour
  let d2 : ℝ := 60  -- Distance traveled in the second hour
  (d1 + d2) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l4022_402237


namespace NUMINAMATH_CALUDE_second_discount_percentage_l4022_402219

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  initial_price = 200 →
  first_discount = 10 →
  final_price = 171 →
  (initial_price * (1 - first_discount / 100) * (1 - (initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100))) = final_price) ∧
  ((initial_price * (1 - first_discount / 100) - final_price) / (initial_price * (1 - first_discount / 100)) * 100 = 5) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l4022_402219


namespace NUMINAMATH_CALUDE_point_outside_circle_l4022_402240

theorem point_outside_circle (r d : ℝ) (hr : r = 2) (hd : d = 3) :
  d > r :=
by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l4022_402240


namespace NUMINAMATH_CALUDE_maria_coin_count_l4022_402235

/-- Represents the number of stacks for each coin type -/
structure CoinStacks where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Represents the number of coins in each stack for each coin type -/
structure CoinsPerStack where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins given the number of stacks and coins per stack -/
def totalCoins (stacks : CoinStacks) (perStack : CoinsPerStack) : ℕ :=
  stacks.pennies * perStack.pennies +
  stacks.nickels * perStack.nickels +
  stacks.dimes * perStack.dimes

theorem maria_coin_count :
  let stacks : CoinStacks := { pennies := 3, nickels := 5, dimes := 7 }
  let perStack : CoinsPerStack := { pennies := 10, nickels := 8, dimes := 4 }
  totalCoins stacks perStack = 98 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_count_l4022_402235


namespace NUMINAMATH_CALUDE_tank_length_calculation_l4022_402201

/-- Calculates the length of a tank given its dimensions and plastering costs. -/
theorem tank_length_calculation (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.75)
  (h_total_cost : total_cost = 558) :
  ∃ length : ℝ, length = 25 ∧ 
  total_cost = (2 * (length * depth) + 2 * (width * depth) + (length * width)) * cost_per_sqm :=
by sorry

end NUMINAMATH_CALUDE_tank_length_calculation_l4022_402201


namespace NUMINAMATH_CALUDE_candy_cost_l4022_402250

def amount_given : ℚ := 1
def change_received : ℚ := 0.46

theorem candy_cost : amount_given - change_received = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l4022_402250


namespace NUMINAMATH_CALUDE_percentage_problem_l4022_402216

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = 4 / 5 * 25 + 6 → P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4022_402216


namespace NUMINAMATH_CALUDE_cleaning_fluid_purchase_l4022_402221

theorem cleaning_fluid_purchase :
  ∃ (x y : ℕ), 
    30 * x + 20 * y = 160 ∧ 
    x + y = 7 ∧
    ∀ (a b : ℕ), 30 * a + 20 * b = 160 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_cleaning_fluid_purchase_l4022_402221


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l4022_402239

theorem coefficient_x_squared (x : ℝ) : 
  let expansion := (1 + 1/x + 1/x^2) * (1 + x^2)^5
  ∃ a b c d e : ℝ, expansion = a*x^2 + b*x + c + d/x + e/x^2 ∧ a = 15 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l4022_402239


namespace NUMINAMATH_CALUDE_equivalent_operations_l4022_402290

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/3) = x * (15/12) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l4022_402290


namespace NUMINAMATH_CALUDE_model_b_sample_size_l4022_402245

/-- Calculates the number of items to be sampled from a stratum in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (stratumSize : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (stratumSize * totalSampleSize) / totalPopulation

theorem model_b_sample_size :
  let totalProduction : ℕ := 9200
  let modelBProduction : ℕ := 6000
  let totalSampleSize : ℕ := 46
  stratifiedSampleSize totalProduction modelBProduction totalSampleSize = 30 := by
  sorry

end NUMINAMATH_CALUDE_model_b_sample_size_l4022_402245


namespace NUMINAMATH_CALUDE_pipe_filling_time_l4022_402211

/-- Proves that Pipe A takes 20 minutes to fill the tank alone given the conditions -/
theorem pipe_filling_time (t : ℝ) : 
  t > 0 →  -- Pipe A fills the tank in t minutes (t must be positive)
  (t / 4 > 0) →  -- Pipe B fills the tank in t/4 minutes (t/4 must be positive)
  (1 / t + 1 / (t / 4) = 1 / 4) →  -- When both pipes are open, it takes 4 minutes to fill the tank
  t = 20 := by
sorry


end NUMINAMATH_CALUDE_pipe_filling_time_l4022_402211


namespace NUMINAMATH_CALUDE_parallelogram45_diag_product_l4022_402272

/-- A parallelogram with one angle of 45° -/
structure Parallelogram45 where
  a : ℝ
  b : ℝ
  d₁ : ℝ
  d₂ : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_diag₁ : d₁^2 = a^2 + b^2 + Real.sqrt 2 * a * b
  h_diag₂ : d₂^2 = a^2 + b^2 - Real.sqrt 2 * a * b

/-- The product of squared diagonals equals the sum of fourth powers of sides -/
theorem parallelogram45_diag_product (p : Parallelogram45) :
    p.d₁^2 * p.d₂^2 = p.a^4 + p.b^4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram45_diag_product_l4022_402272


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l4022_402215

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l4022_402215


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l4022_402209

theorem fourth_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 23) (h₂ : r₂ = 35) (h₃ : r₃ = Real.sqrt 1754) :
  π * r₃^2 = π * r₁^2 + π * r₂^2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l4022_402209


namespace NUMINAMATH_CALUDE_jenny_ate_65_squares_l4022_402225

-- Define the number of chocolate squares Mike ate
def mike_squares : ℕ := 20

-- Define the number of chocolate squares Jenny ate
def jenny_squares : ℕ := 3 * mike_squares + 5

-- Theorem to prove
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_squares_l4022_402225


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l4022_402232

/-- The probability of a randomly chosen point within a circle of radius 5 being closer to the center than to the boundary, given an inner concentric circle of radius 2 -/
theorem probability_closer_to_center (outer_radius inner_radius : ℝ) : 
  outer_radius = 5 → 
  inner_radius = 2 → 
  (π * inner_radius^2) / (π * outer_radius^2) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l4022_402232


namespace NUMINAMATH_CALUDE_equation_proof_l4022_402241

theorem equation_proof : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l4022_402241


namespace NUMINAMATH_CALUDE_range_of_k_prove_k_range_l4022_402261

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x | x < k}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (A ∪ B k = B k) → k > 2 := by
  sorry

-- The range of k
def k_range : Set ℝ := {k | k > 2}

-- Theorem to prove the range of k
theorem prove_k_range :
  ∀ k, (A ∪ B k = B k) ↔ k ∈ k_range := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_prove_k_range_l4022_402261


namespace NUMINAMATH_CALUDE_least_value_quadratic_inequality_l4022_402200

theorem least_value_quadratic_inequality :
  ∃ (x : ℝ), x = 4 ∧ (∀ y : ℝ, -y^2 + 9*y - 20 ≤ 0 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_inequality_l4022_402200


namespace NUMINAMATH_CALUDE_milk_price_problem_l4022_402231

theorem milk_price_problem (initial_cost initial_bottles subsequent_cost : ℝ) : 
  initial_cost = 108 →
  subsequent_cost = 90 →
  ∃ (price : ℝ), 
    initial_bottles * price = initial_cost ∧
    (initial_bottles + 1) * (price * 0.25) = subsequent_cost →
    price = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_problem_l4022_402231


namespace NUMINAMATH_CALUDE_total_selling_price_is_correct_l4022_402238

def cycle_price : ℕ := 2000
def scooter_price : ℕ := 25000
def bike_price : ℕ := 60000

def cycle_loss_percent : ℚ := 10 / 100
def scooter_loss_percent : ℚ := 15 / 100
def bike_loss_percent : ℚ := 5 / 100

def selling_price (price : ℕ) (loss_percent : ℚ) : ℚ :=
  price - (price * loss_percent)

def total_selling_price : ℚ :=
  selling_price cycle_price cycle_loss_percent +
  selling_price scooter_price scooter_loss_percent +
  selling_price bike_price bike_loss_percent

theorem total_selling_price_is_correct :
  total_selling_price = 80050 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_is_correct_l4022_402238


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4022_402275

theorem complex_modulus_problem (z : ℂ) :
  (1 + Complex.I) * z = 1 - 2 * Complex.I^3 →
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4022_402275


namespace NUMINAMATH_CALUDE_inequality_and_equalities_l4022_402205

theorem inequality_and_equalities : 
  ((-3)^2 ≠ -3^2) ∧ 
  (|-5| = -(-5)) ∧ 
  (-Real.sqrt 4 = -2) ∧ 
  ((-1)^3 = -1^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equalities_l4022_402205


namespace NUMINAMATH_CALUDE_reflect_x_coordinates_l4022_402291

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across x-axis operation
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem reflect_x_coordinates (x y : ℝ) :
  reflect_x (x, y) = (x, -y) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_coordinates_l4022_402291


namespace NUMINAMATH_CALUDE_cubic_equation_root_difference_l4022_402270

theorem cubic_equation_root_difference (a b c : ℚ) : 
  ∃ (p q r : ℚ), p^3 + a*p^2 + b*p + c = 0 ∧ 
                  q^3 + a*q^2 + b*q + c = 0 ∧ 
                  r^3 + a*r^2 + b*r + c = 0 ∧ 
                  (q - p = 2014 ∨ r - q = 2014 ∨ r - p = 2014) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_difference_l4022_402270


namespace NUMINAMATH_CALUDE_min_distance_point_to_curve_l4022_402273

theorem min_distance_point_to_curve (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  let P : Prod Real Real := (1 + Real.cos α, Real.sin α)
  let C : Set (Prod Real Real) := {Q : Prod Real Real | Q.1 + Q.2 = 9}
  (∃ (d : Real), d = 4 * Real.sqrt 2 - 1 ∧
    ∀ Q ∈ C, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_curve_l4022_402273


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l4022_402299

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2011 + Real.sqrt 2010 →
  Q = -Real.sqrt 2011 - Real.sqrt 2010 →
  R = Real.sqrt 2011 - Real.sqrt 2010 →
  S = Real.sqrt 2010 - Real.sqrt 2011 →
  P * Q * R * S = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l4022_402299


namespace NUMINAMATH_CALUDE_min_value_theorem_l4022_402292

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∀ x, 2 * a + b + c ≥ x → x ≤ 2 * Real.sqrt 3 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4022_402292


namespace NUMINAMATH_CALUDE_complex_equation_result_l4022_402204

theorem complex_equation_result (x y : ℝ) (i : ℂ) 
  (h1 : x * i + 2 = y - i) 
  (h2 : i^2 = -1) : 
  x - y = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_result_l4022_402204


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l4022_402217

def Circle (center : ℝ × ℝ) (radius : ℝ) := { p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

theorem intersection_distance_squared :
  let circle1 := Circle (5, 0) 5
  let circle2 := Circle (0, 5) 5
  ∀ C D : ℝ × ℝ, C ∈ circle1 ∧ C ∈ circle2 ∧ D ∈ circle1 ∧ D ∈ circle2 ∧ C ≠ D →
  distance_squared C D = 50 := by
sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l4022_402217


namespace NUMINAMATH_CALUDE_scientific_notation_34_million_l4022_402256

theorem scientific_notation_34_million : 
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_34_million_l4022_402256


namespace NUMINAMATH_CALUDE_intercepted_segment_length_l4022_402266

-- Define the polar equations
def line_equation (p θ : ℝ) : Prop := p * Real.cos θ = 1
def circle_equation (p θ : ℝ) : Prop := p = 4 * Real.cos θ

-- Define the theorem
theorem intercepted_segment_length :
  ∃ (p₁ θ₁ p₂ θ₂ : ℝ),
    line_equation p₁ θ₁ ∧
    line_equation p₂ θ₂ ∧
    circle_equation p₁ θ₁ ∧
    circle_equation p₂ θ₂ ∧
    (p₁ * Real.cos θ₁ - p₂ * Real.cos θ₂)^2 + (p₁ * Real.sin θ₁ - p₂ * Real.sin θ₂)^2 = 12 :=
sorry

end NUMINAMATH_CALUDE_intercepted_segment_length_l4022_402266


namespace NUMINAMATH_CALUDE_square_area_thirteen_l4022_402224

/-- The area of a square with vertices at (1, 1), (-2, 3), (-1, 8), and (2, 4) is 13 square units. -/
theorem square_area_thirteen : 
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 3)
  let R : ℝ × ℝ := (-1, 8)
  let S : ℝ × ℝ := (2, 4)
  let square_area := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  square_area = 13 := by sorry

end NUMINAMATH_CALUDE_square_area_thirteen_l4022_402224


namespace NUMINAMATH_CALUDE_equation_solution_l4022_402244

theorem equation_solution : ∀ x : ℝ, 4 * x^2 - (x - 1)^2 = 0 ↔ x = -1 ∨ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4022_402244


namespace NUMINAMATH_CALUDE_peach_ripeness_difference_l4022_402268

def bowl_of_peaches (total_peaches initial_ripe ripening_rate days_passed peaches_eaten : ℕ) : ℕ :=
  let ripe_peaches := initial_ripe + ripening_rate * days_passed - peaches_eaten
  let unripe_peaches := total_peaches - ripe_peaches
  ripe_peaches - unripe_peaches

theorem peach_ripeness_difference :
  bowl_of_peaches 18 4 2 5 3 = 4 := by
  sorry

#eval bowl_of_peaches 18 4 2 5 3

end NUMINAMATH_CALUDE_peach_ripeness_difference_l4022_402268


namespace NUMINAMATH_CALUDE_marc_total_spent_l4022_402296

/-- Calculates the total amount Marc spent on his purchases --/
def total_spent (model_car_price : ℝ) (paint_price : ℝ) (paintbrush_price : ℝ)
                (display_case_price : ℝ) (model_car_discount : ℝ) (paint_coupon : ℝ)
                (gift_card : ℝ) (first_tax_rate : ℝ) (second_tax_rate : ℝ) : ℝ :=
  let model_cars_cost := 5 * model_car_price * (1 - model_car_discount)
  let paint_cost := 5 * paint_price - paint_coupon
  let paintbrushes_cost := 7 * paintbrush_price
  let first_subtotal := model_cars_cost + paint_cost + paintbrushes_cost - gift_card
  let first_transaction := first_subtotal * (1 + first_tax_rate)
  let display_cases_cost := 3 * display_case_price
  let second_transaction := display_cases_cost * (1 + second_tax_rate)
  first_transaction + second_transaction

/-- Theorem stating that Marc's total spent is $187.02 --/
theorem marc_total_spent :
  total_spent 20 10 2 15 0.1 5 20 0.08 0.06 = 187.02 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spent_l4022_402296


namespace NUMINAMATH_CALUDE_monotonic_increasing_sufficient_not_necessary_l4022_402253

-- Define a monotonically increasing function on ℝ
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Define the existence of x₁ < x₂ such that f(x₁) < f(x₂)
def ExistsStrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem stating that monotonic increasing is sufficient but not necessary
-- for the existence of strictly increasing points
theorem monotonic_increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (MonotonicIncreasing f → ExistsStrictlyIncreasing f) ∧
  ¬(ExistsStrictlyIncreasing f → MonotonicIncreasing f) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sufficient_not_necessary_l4022_402253
