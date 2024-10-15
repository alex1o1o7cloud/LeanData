import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_given_max_product_l3118_311846

theorem min_sum_given_max_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, a * b * x + y ≤ 8) → a + b ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_max_product_l3118_311846


namespace NUMINAMATH_CALUDE_volleyball_tournament_teams_l3118_311818

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  teams : ℕ
  no_win_fraction : ℚ
  single_round : Bool

/-- Theorem: In a single round volleyball tournament where 20% of teams did not win a single game,
    the total number of teams must be 5. -/
theorem volleyball_tournament_teams
  (t : VolleyballTournament)
  (h1 : t.no_win_fraction = 1/5)
  (h2 : t.single_round = true)
  : t.teams = 5 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_teams_l3118_311818


namespace NUMINAMATH_CALUDE_school_boys_count_l3118_311879

theorem school_boys_count :
  ∀ (total boys girls : ℕ),
  total = 900 →
  boys + girls = total →
  girls * total = boys * boys →
  boys = 810 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l3118_311879


namespace NUMINAMATH_CALUDE_prob_same_color_is_one_twentieth_l3118_311827

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of girls selecting marbles -/
def number_of_girls : ℕ := 3

/-- Calculates the probability of all girls selecting the same colored marble -/
def prob_same_color : ℚ :=
  2 * (marbles_per_color.factorial / (marbles_per_color + number_of_girls).factorial)

/-- Theorem stating that the probability of all girls selecting the same colored marble is 1/20 -/
theorem prob_same_color_is_one_twentieth : prob_same_color = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_one_twentieth_l3118_311827


namespace NUMINAMATH_CALUDE_geometric_sequence_and_curve_max_l3118_311888

/-- Given real numbers a, b, c, and d forming a geometric sequence, 
    if the curve y = 3x - x^3 has a local maximum at x = b with the value c, 
    then ad = 2 -/
theorem geometric_sequence_and_curve_max (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →                 -- local maximum condition
  (3 * b - b^3 = c) →                                    -- value at local maximum
  a * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_curve_max_l3118_311888


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3118_311898

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 < 4) ↔ (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3118_311898


namespace NUMINAMATH_CALUDE_sheep_price_is_30_l3118_311815

/-- Represents the farm animals and their sale --/
structure FarmSale where
  goats : ℕ
  sheep : ℕ
  goat_price : ℕ
  sheep_price : ℕ
  goats_sold_ratio : ℚ
  sheep_sold_ratio : ℚ
  total_sale : ℕ

/-- The conditions of the farm sale problem --/
def farm_conditions (s : FarmSale) : Prop :=
  s.goats * 7 = s.sheep * 5 ∧
  s.goats + s.sheep = 360 ∧
  s.goats_sold_ratio = 1/2 ∧
  s.sheep_sold_ratio = 2/3 ∧
  s.goat_price = 40 ∧
  s.total_sale = 7200

/-- The theorem stating that the sheep price is $30 --/
theorem sheep_price_is_30 (s : FarmSale) (h : farm_conditions s) : s.sheep_price = 30 := by
  sorry


end NUMINAMATH_CALUDE_sheep_price_is_30_l3118_311815


namespace NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3118_311848

/-- Theorem: For an ellipse with center (3, -5), semi-major axis length 7, and semi-minor axis length 4,
    the sum of its center coordinates and axis lengths is 9. -/
theorem ellipse_sum_coordinates_and_axes :
  ∀ (h k a b : ℝ),
    h = 3 →
    k = -5 →
    a = 7 →
    b = 4 →
    h + k + a + b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_coordinates_and_axes_l3118_311848


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3118_311843

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_third : a 3 = 30)
  (h_ninth : a 9 = 60) :
  a 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l3118_311843


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l3118_311838

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_expression_equals_one :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 5 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l3118_311838


namespace NUMINAMATH_CALUDE_original_advertisers_from_university_a_l3118_311803

/-- Represents the fraction of advertisers from University A -/
def fractionFromUniversityA : ℚ := 3/4

/-- Represents the total number of original network advertisers -/
def totalOriginalAdvertisers : ℕ := 20

/-- Represents the percentage of computer advertisers from University A -/
def percentageFromUniversityA : ℚ := 75/100

theorem original_advertisers_from_university_a :
  (↑⌊(percentageFromUniversityA * totalOriginalAdvertisers)⌋ : ℚ) / totalOriginalAdvertisers = fractionFromUniversityA :=
sorry

end NUMINAMATH_CALUDE_original_advertisers_from_university_a_l3118_311803


namespace NUMINAMATH_CALUDE_product_rule_l3118_311835

theorem product_rule (b a : ℤ) (h : 0 ≤ a ∧ a < 10) : 
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := by
  sorry

end NUMINAMATH_CALUDE_product_rule_l3118_311835


namespace NUMINAMATH_CALUDE_triangle_similarity_fc_value_l3118_311852

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 13.875 -/
theorem triangle_similarity_fc_value (DC CB AD AB ED : ℝ) : 
  DC = 10 →
  CB = 9 →
  AB = (1/3) * AD →
  ED = (3/4) * AD →
  ∃ (FC : ℝ), FC = 13.875 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_fc_value_l3118_311852


namespace NUMINAMATH_CALUDE_cake_pieces_count_l3118_311886

/-- Given 50 friends and 3 pieces of cake per friend, prove that the total number of cake pieces is 150. -/
theorem cake_pieces_count (num_friends : ℕ) (pieces_per_friend : ℕ) : 
  num_friends = 50 → pieces_per_friend = 3 → num_friends * pieces_per_friend = 150 := by
  sorry


end NUMINAMATH_CALUDE_cake_pieces_count_l3118_311886


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3118_311849

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3118_311849


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3118_311842

/-- The number of ways to assign grades to students. -/
def assignGrades (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 different grades to 15 students is 4^15. -/
theorem grade_assignment_count :
  assignGrades 15 4 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3118_311842


namespace NUMINAMATH_CALUDE_pencils_given_l3118_311836

theorem pencils_given (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 9)
  (h2 : total_pencils = 65) :
  total_pencils - initial_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_l3118_311836


namespace NUMINAMATH_CALUDE_approximate_solution_exists_l3118_311870

def f (x : ℝ) := 2 * x^3 + 3 * x - 3

theorem approximate_solution_exists :
  (f 0.625 < 0) →
  (f 0.75 > 0) →
  (f 0.6875 < 0) →
  ∃ x : ℝ, x ∈ Set.Icc 0.6 0.8 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_approximate_solution_exists_l3118_311870


namespace NUMINAMATH_CALUDE_domain_range_sum_l3118_311820

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x

-- Define the theorem
theorem domain_range_sum (m n : ℝ) : 
  (∀ x, m ≤ x ∧ x ≤ n → 2*m ≤ f x ∧ f x ≤ 2*n) →
  (∀ y, 2*m ≤ y ∧ y ≤ 2*n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y) →
  m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_domain_range_sum_l3118_311820


namespace NUMINAMATH_CALUDE_f_2009_is_zero_l3118_311847

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2009_is_zero (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_odd_function (fun x ↦ f (x - 1))) : 
  f 2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_is_zero_l3118_311847


namespace NUMINAMATH_CALUDE_tamtam_yellow_shells_l3118_311806

/-- The number of shells Tamtam collected of each color --/
structure ShellCollection where
  total : ℕ
  purple : ℕ
  pink : ℕ
  blue : ℕ
  orange : ℕ

/-- Calculates the number of yellow shells in a collection --/
def yellowShells (s : ShellCollection) : ℕ :=
  s.total - (s.purple + s.pink + s.blue + s.orange)

/-- Tamtam's shell collection --/
def tamtamShells : ShellCollection :=
  { total := 65
    purple := 13
    pink := 8
    blue := 12
    orange := 14 }

/-- Theorem stating that Tamtam collected 18 yellow shells --/
theorem tamtam_yellow_shells : yellowShells tamtamShells = 18 := by
  sorry

end NUMINAMATH_CALUDE_tamtam_yellow_shells_l3118_311806


namespace NUMINAMATH_CALUDE_little_red_height_calculation_l3118_311800

/-- Little Ming's height in meters -/
def little_ming_height : ℝ := 1.3

/-- The difference in height between Little Ming and Little Red in meters -/
def height_difference : ℝ := 0.2

/-- Little Red's height in meters -/
def little_red_height : ℝ := little_ming_height - height_difference

theorem little_red_height_calculation :
  little_red_height = 1.1 := by sorry

end NUMINAMATH_CALUDE_little_red_height_calculation_l3118_311800


namespace NUMINAMATH_CALUDE_amount_of_b_l3118_311813

theorem amount_of_b (a b : ℚ) : 
  a + b = 1210 → 
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b → 
  b = 484 := by
sorry

end NUMINAMATH_CALUDE_amount_of_b_l3118_311813


namespace NUMINAMATH_CALUDE_like_terms_exponent_l3118_311830

theorem like_terms_exponent (x y : ℝ) (m : ℤ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^3 * y^(m+3) = b * x^3 * y^5) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l3118_311830


namespace NUMINAMATH_CALUDE_triangle_property_l3118_311862

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively
  (h1 : A + B + C = π)  -- Sum of angles in a triangle
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive side lengths
  (h3 : b < c)  -- Given condition

-- Define the existence of points E and F
def points_exist (t : Triangle) : Prop :=
  ∃ E F : ℝ, 
    E > 0 ∧ F > 0 ∧
    E ≤ t.c ∧ F ≤ t.b ∧
    E = F ∧
    ∃ D : ℝ, D > 0 ∧ D < t.a ∧
    (t.A / 2 = Real.arctan (D / E) + Real.arctan (D / F))

-- Theorem statement
theorem triangle_property (t : Triangle) (h : points_exist t) :
  t.A / 2 ≤ t.B ∧ (t.a * t.c) / (t.b + t.c) = t.c * (t.a / (t.b + t.c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l3118_311862


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_and_half_l3118_311811

theorem reciprocal_of_negative_three_and_half (x : ℚ) :
  x = -3.5 → (1 / x) = -2/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_and_half_l3118_311811


namespace NUMINAMATH_CALUDE_guaranteed_pairs_l3118_311883

/-- A color of a candy -/
inductive Color
| Black
| White

/-- A position in the 7x7 grid -/
structure Position where
  x : Fin 7
  y : Fin 7

/-- A configuration of the candy box -/
def Configuration := Position → Color

/-- Two positions are adjacent if they are side-by-side or diagonal -/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x = p2.x ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y = p2.y) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y = p2.y) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val + 1 = p2.x.val ∧ p1.y.val = p2.y.val + 1) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val + 1 = p2.y.val) ∨
  (p1.x.val = p2.x.val + 1 ∧ p1.y.val = p2.y.val + 1)

/-- A pair of adjacent positions with the same color -/
structure ColoredPair (config : Configuration) where
  p1 : Position
  p2 : Position
  adj : adjacent p1 p2
  same_color : config p1 = config p2

/-- The main theorem: there always exists a set of at least 16 pairs of adjacent cells with the same color -/
theorem guaranteed_pairs (config : Configuration) : 
  ∃ (pairs : Finset (ColoredPair config)), pairs.card ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_pairs_l3118_311883


namespace NUMINAMATH_CALUDE_f_value_theorem_l3118_311814

-- Define the polynomial equation
def polynomial_equation (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a1*x^3 + a2*x^2 + a3*x + a4 = (x+1)^4 + b1*(x+1)^3 + b2*(x+1)^2 + b3*(x+1) + b4

-- Define the mapping f
def f (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : ℝ := b1 - b2 + b3 - b4

-- Theorem statement
theorem f_value_theorem :
  ∀ b1 b2 b3 b4 : ℝ, polynomial_equation 2 0 1 6 b1 b2 b3 b4 → f 2 0 1 6 b1 b2 b3 b4 = -3 :=
by sorry

end NUMINAMATH_CALUDE_f_value_theorem_l3118_311814


namespace NUMINAMATH_CALUDE_inequality_theorem_l3118_311837

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3118_311837


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3118_311816

/-- An isosceles triangle with perimeter 7 and one side length 3 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles triangle condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3118_311816


namespace NUMINAMATH_CALUDE_problem_solution_l3118_311825

def problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (∀ x, g_inv (g x) = x) ∧
  (∀ y, g (g_inv y) = y) ∧
  g 4 = 6 ∧
  g 6 = 2 ∧
  g 3 = 7 ∧
  g_inv (g_inv 6 + g_inv 7) = 3

theorem problem_solution :
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ), problem g g_inv :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3118_311825


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l3118_311851

theorem ratio_sum_problem (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ratio : b = 2*a ∧ c = 4*a ∧ d = 5*a) 
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 2540) : 
  a + b + c + d = 12 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l3118_311851


namespace NUMINAMATH_CALUDE_fraction_value_l3118_311891

theorem fraction_value (m n : ℝ) (h : (m - 8)^2 + |n + 6| = 0) : n / m = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3118_311891


namespace NUMINAMATH_CALUDE_hyperbola_chord_length_l3118_311850

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of the hyperbola -/
structure Chord where
  A : Point
  B : Point

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- The foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_chord_length 
  (h : Hyperbola) 
  (c : Chord) 
  (F1 F2 : Point) :
  (foci h = (F1, F2)) →
  (on_hyperbola h c.A) →
  (on_hyperbola h c.B) →
  (distance F1 c.A = 0 ∨ distance F1 c.B = 0) →
  (distance c.A F2 + distance c.B F2 = 2 * distance c.A c.B) →
  distance c.A c.B = 4 * h.a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_chord_length_l3118_311850


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3118_311890

def is_geometric_sequence_with_ratio_2 (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

theorem condition_necessary_not_sufficient :
  (∀ a : ℕ → ℝ, is_geometric_sequence_with_ratio_2 a → condition a) ∧
  (∃ a : ℕ → ℝ, condition a ∧ ¬is_geometric_sequence_with_ratio_2 a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3118_311890


namespace NUMINAMATH_CALUDE_problem_statement_l3118_311840

-- Define the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

-- State the theorem
theorem problem_statement (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 11 = 4 * a 7 →
  arithmetic_sequence b →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3118_311840


namespace NUMINAMATH_CALUDE_discount_rate_for_given_profit_l3118_311877

/-- Given a product with cost price, marked price, and desired profit percentage,
    calculate the discount rate needed to achieve the desired profit. -/
def calculate_discount_rate (cost_price marked_price profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  selling_price / marked_price

theorem discount_rate_for_given_profit :
  let cost_price : ℚ := 200
  let marked_price : ℚ := 300
  let profit_percentage : ℚ := 20
  calculate_discount_rate cost_price marked_price profit_percentage = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_discount_rate_for_given_profit_l3118_311877


namespace NUMINAMATH_CALUDE_combined_flock_size_is_300_l3118_311826

/-- Calculates the combined flock size after a given number of years -/
def combinedFlockSize (initialSize birthRate deathRate years additionalFlockSize : ℕ) : ℕ :=
  initialSize + (birthRate - deathRate) * years + additionalFlockSize

/-- Theorem: The combined flock size after 5 years is 300 ducks -/
theorem combined_flock_size_is_300 :
  combinedFlockSize 100 30 20 5 150 = 300 := by
  sorry

#eval combinedFlockSize 100 30 20 5 150

end NUMINAMATH_CALUDE_combined_flock_size_is_300_l3118_311826


namespace NUMINAMATH_CALUDE_square_sum_equals_nineteen_l3118_311893

theorem square_sum_equals_nineteen (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x^2 + y^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_nineteen_l3118_311893


namespace NUMINAMATH_CALUDE_c_minus_d_value_l3118_311808

theorem c_minus_d_value (c d : ℝ) 
  (eq1 : 2020 * c + 2024 * d = 2030)
  (eq2 : 2022 * c + 2026 * d = 2032) : 
  c - d = -4 := by
sorry

end NUMINAMATH_CALUDE_c_minus_d_value_l3118_311808


namespace NUMINAMATH_CALUDE_park_road_perimeter_l3118_311884

/-- Given a square park with a road inside, proves that the perimeter of the outer edge of the road is 600 meters -/
theorem park_road_perimeter (side_length : ℝ) : 
  side_length > 0 →
  side_length^2 - (side_length - 6)^2 = 1764 →
  4 * side_length = 600 := by
sorry

end NUMINAMATH_CALUDE_park_road_perimeter_l3118_311884


namespace NUMINAMATH_CALUDE_solution_positivity_l3118_311869

theorem solution_positivity (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m * x - 1 = 2 * x) ↔ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_positivity_l3118_311869


namespace NUMINAMATH_CALUDE_inequality_proof_l3118_311868

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13)
  (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3118_311868


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3118_311810

theorem polygon_interior_angles_sum (n : ℕ) (h : n = 9) :
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3118_311810


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l3118_311856

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8)*x + 20 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l3118_311856


namespace NUMINAMATH_CALUDE_carol_stereo_savings_l3118_311824

theorem carol_stereo_savings : 
  ∀ (stereo_fraction : ℚ),
  (stereo_fraction + (1/3) * stereo_fraction = 1/4) →
  stereo_fraction = 3/16 := by
sorry

end NUMINAMATH_CALUDE_carol_stereo_savings_l3118_311824


namespace NUMINAMATH_CALUDE_number_120_more_than_third_l3118_311863

theorem number_120_more_than_third : ∃ x : ℚ, x = (1/3) * x + 120 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_120_more_than_third_l3118_311863


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l3118_311892

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7) 
  (h2 : river_current = 1) 
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l3118_311892


namespace NUMINAMATH_CALUDE_bus_departure_interval_l3118_311855

/-- Represents the number of minutes between 6:00 AM and 7:00 AM -/
def total_minutes : ℕ := 60

/-- Represents the number of bus departures between 6:00 AM and 7:00 AM -/
def num_departures : ℕ := 11

/-- Calculates the interval between consecutive bus departures -/
def interval (total : ℕ) (departures : ℕ) : ℚ :=
  (total : ℚ) / ((departures - 1) : ℚ)

/-- Proves that the interval between consecutive bus departures is 6 minutes -/
theorem bus_departure_interval :
  interval total_minutes num_departures = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_departure_interval_l3118_311855


namespace NUMINAMATH_CALUDE_most_trailing_zeros_l3118_311829

-- Define a function to count trailing zeros
def countTrailingZeros (n : ℕ) : ℕ := sorry

-- Define the arithmetic expressions
def expr1 : ℕ := 300 + 60
def expr2 : ℕ := 22 * 5
def expr3 : ℕ := 25 * 4
def expr4 : ℕ := 400 / 8

-- Theorem statement
theorem most_trailing_zeros :
  countTrailingZeros expr3 ≥ countTrailingZeros expr1 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr2 ∧
  countTrailingZeros expr3 ≥ countTrailingZeros expr4 :=
by sorry

end NUMINAMATH_CALUDE_most_trailing_zeros_l3118_311829


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3118_311821

/-- An isosceles, obtuse triangle with one angle 75% larger than a right angle has smallest angles of 11.25°. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.75 * 90 →  -- largest angle is 75% larger than right angle
  a = 11.25 :=
by
  sorry

#check isosceles_obtuse_triangle_smallest_angle

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3118_311821


namespace NUMINAMATH_CALUDE_modulus_of_2_minus_i_l3118_311885

theorem modulus_of_2_minus_i : 
  let z : ℂ := 2 - I
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_2_minus_i_l3118_311885


namespace NUMINAMATH_CALUDE_otherSideHeadsProbabilityIsCorrect_l3118_311841

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | DoubleHeads
  | DoubleTails

/-- Represents the possible outcomes of a coin flip -/
inductive FlipResult
  | Heads
  | Tails

/-- The probability of selecting each coin -/
def coinProbability : Coin → ℚ
  | Coin.Normal => 1/3
  | Coin.DoubleHeads => 1/3
  | Coin.DoubleTails => 1/3

/-- The probability of getting heads given a specific coin -/
def headsGivenCoin : Coin → ℚ
  | Coin.Normal => 1/2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

/-- The probability that the other side is heads given that heads was observed -/
def otherSideHeadsProbability : ℚ := by sorry

theorem otherSideHeadsProbabilityIsCorrect :
  otherSideHeadsProbability = 2/3 := by sorry

end NUMINAMATH_CALUDE_otherSideHeadsProbabilityIsCorrect_l3118_311841


namespace NUMINAMATH_CALUDE_parallel_vectors_l3118_311833

theorem parallel_vectors (a b : ℝ × ℝ) :
  a = (-1, 3) →
  b.1 = 2 →
  (a.1 * b.2 = a.2 * b.1) →
  b.2 = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3118_311833


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l3118_311859

/-- The amount of money Zachary received from selling his video games -/
def zachary_earnings : ℕ := 40 * 5

/-- The amount of money Jason received from selling his video games -/
def jason_earnings : ℕ := zachary_earnings + (zachary_earnings * 30 / 100)

/-- The amount of money Ryan received from selling his video games -/
def ryan_earnings : ℕ := jason_earnings + 50

/-- The amount of money Emily received from selling her video games -/
def emily_earnings : ℕ := ryan_earnings - (ryan_earnings * 20 / 100)

/-- The amount of money Lily received from selling her video games -/
def lily_earnings : ℕ := emily_earnings + 70

/-- The total amount of money received by all five friends -/
def total_earnings : ℕ := zachary_earnings + jason_earnings + ryan_earnings + emily_earnings + lily_earnings

theorem video_game_sales_earnings : total_earnings = 1336 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l3118_311859


namespace NUMINAMATH_CALUDE_consecutive_pair_sum_divisible_by_five_l3118_311899

theorem consecutive_pair_sum_divisible_by_five (n : ℕ) : 
  n < 1500 → 
  (n + (n + 1)) % 5 = 0 → 
  (57 + 58) % 5 = 0 → 
  57 = n := by
sorry

end NUMINAMATH_CALUDE_consecutive_pair_sum_divisible_by_five_l3118_311899


namespace NUMINAMATH_CALUDE_initial_water_was_six_cups_l3118_311834

/-- Represents the water consumption during a hike --/
structure HikeWaterConsumption where
  total_distance : ℝ
  total_time : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_three_miles_rate : ℝ

/-- Calculates the initial amount of water in the canteen --/
def initial_water (h : HikeWaterConsumption) : ℝ :=
  h.remaining_water + h.leak_rate * h.total_time + 
  h.last_mile_consumption + h.first_three_miles_rate * (h.total_distance - 1)

/-- Theorem stating that the initial amount of water in the canteen was 6 cups --/
theorem initial_water_was_six_cups (h : HikeWaterConsumption) 
  (h_distance : h.total_distance = 4)
  (h_time : h.total_time = 2)
  (h_remaining : h.remaining_water = 1)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 1)
  (h_first_three : h.first_three_miles_rate = 2/3) :
  initial_water h = 6 := by
  sorry


end NUMINAMATH_CALUDE_initial_water_was_six_cups_l3118_311834


namespace NUMINAMATH_CALUDE_negation_of_implication_negation_of_greater_than_one_l3118_311887

theorem negation_of_implication (p q : Prop) :
  ¬(p → q) ↔ (p ∧ ¬q) := by sorry

theorem negation_of_greater_than_one :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_negation_of_greater_than_one_l3118_311887


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3118_311881

theorem square_difference_theorem (N : ℕ+) : 
  (∃ x : ℤ, 2^(N : ℕ) - 2 * (N : ℤ) = x^2) ↔ N = 1 ∨ N = 2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3118_311881


namespace NUMINAMATH_CALUDE_expand_expression_l3118_311831

theorem expand_expression (x : ℝ) : 2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3118_311831


namespace NUMINAMATH_CALUDE_digit_150_of_1_13_l3118_311823

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur

theorem digit_150_of_1_13 : decimal_rep_1_13 150 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_1_13_l3118_311823


namespace NUMINAMATH_CALUDE_basketball_probability_l3118_311832

-- Define the success rate
def success_rate : ℚ := 1/2

-- Define the total number of shots
def total_shots : ℕ := 10

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 3

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem statement
theorem basketball_probability :
  probability total_shots successful_shots success_rate = 15/128 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l3118_311832


namespace NUMINAMATH_CALUDE_zach_needs_six_dollars_l3118_311817

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_money babysit_rate babysit_hours savings : ℕ) : ℕ :=
  let total_earned := allowance + lawn_money + babysit_rate * babysit_hours
  let total_available := savings + total_earned
  if total_available ≥ bike_cost then 0
  else bike_cost - total_available

/-- Theorem stating how much more money Zach needs to earn -/
theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 2 65 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_dollars_l3118_311817


namespace NUMINAMATH_CALUDE_union_equals_B_implies_B_is_real_l3118_311867

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the theorem
theorem union_equals_B_implies_B_is_real (B : Set ℝ) (h : A ∪ B = B) : B = Set.univ :=
sorry

end NUMINAMATH_CALUDE_union_equals_B_implies_B_is_real_l3118_311867


namespace NUMINAMATH_CALUDE_games_comparison_l3118_311858

/-- Given Henry's and Neil's initial game counts and the number of games Henry gave to Neil,
    calculate how many times more games Henry has than Neil after the transfer. -/
theorem games_comparison (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  games_given = 5 →
  (henry_initial - games_given) / (neil_initial + games_given) = 4 := by
sorry

end NUMINAMATH_CALUDE_games_comparison_l3118_311858


namespace NUMINAMATH_CALUDE_chessboard_game_outcomes_l3118_311871

/-- Represents the outcome of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents the starting position of the piece -/
inductive StartPosition
  | Corner
  | AdjacentToCorner

/-- Defines the game on an n × n chessboard -/
def chessboardGame (n : ℕ) (startPos : StartPosition) : GameOutcome :=
  match n, startPos with
  | n, StartPosition.Corner =>
      if n % 2 = 0 then
        GameOutcome.FirstPlayerWins
      else
        GameOutcome.SecondPlayerWins
  | _, StartPosition.AdjacentToCorner => GameOutcome.FirstPlayerWins

/-- Theorem stating the game outcomes -/
theorem chessboard_game_outcomes :
  (∀ n : ℕ, n > 1 →
    (n % 2 = 0 → chessboardGame n StartPosition.Corner = GameOutcome.FirstPlayerWins) ∧
    (n % 2 = 1 → chessboardGame n StartPosition.Corner = GameOutcome.SecondPlayerWins)) ∧
  (∀ n : ℕ, n > 1 → chessboardGame n StartPosition.AdjacentToCorner = GameOutcome.FirstPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_chessboard_game_outcomes_l3118_311871


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3118_311839

-- Define the GCF (Greatest Common Factor) function
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the LCM (Least Common Multiple) function
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 15 21) (LCM 10 14) = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3118_311839


namespace NUMINAMATH_CALUDE_farmer_brown_chickens_l3118_311880

/-- Given the number of sheep, total legs, and legs per animal, calculate the number of chickens -/
def calculate_chickens (num_sheep : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (sheep_legs : ℕ) : ℕ :=
  (total_legs - num_sheep * sheep_legs) / chicken_legs

/-- Theorem stating that under the given conditions, the number of chickens is 7 -/
theorem farmer_brown_chickens :
  let num_sheep : ℕ := 5
  let total_legs : ℕ := 34
  let chicken_legs : ℕ := 2
  let sheep_legs : ℕ := 4
  calculate_chickens num_sheep total_legs chicken_legs sheep_legs = 7 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_chickens_l3118_311880


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3118_311876

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given sets of lengths
def set_A : (ℝ × ℝ × ℝ) := (3, 4, 8)
def set_B : (ℝ × ℝ × ℝ) := (2, 5, 2)
def set_C : (ℝ × ℝ × ℝ) := (3, 5, 6)
def set_D : (ℝ × ℝ × ℝ) := (5, 6, 11)

-- Theorem stating that only set_C can form a triangle
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  (can_form_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3118_311876


namespace NUMINAMATH_CALUDE_bottle_caps_problem_l3118_311819

/-- The number of bottle caps left in a jar after removing some. -/
def bottle_caps_left (original : ℕ) (removed : ℕ) : ℕ :=
  original - removed

/-- Theorem stating that 40 bottle caps are left when 47 are removed from 87. -/
theorem bottle_caps_problem :
  bottle_caps_left 87 47 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_problem_l3118_311819


namespace NUMINAMATH_CALUDE_cos_240_degrees_l3118_311844

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l3118_311844


namespace NUMINAMATH_CALUDE_candy_purchase_calculation_l3118_311860

/-- Calculates the change and discounted price per pack for a candy purchase. -/
theorem candy_purchase_calculation (packs : ℕ) (regular_price discount payment : ℚ) 
  (h_packs : packs = 3)
  (h_regular_price : regular_price = 12)
  (h_discount : discount = 15 / 100)
  (h_payment : payment = 20) :
  let discounted_total := regular_price * (1 - discount)
  let change := payment - discounted_total
  let price_per_pack := discounted_total / packs
  change = 980 / 100 ∧ price_per_pack = 340 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candy_purchase_calculation_l3118_311860


namespace NUMINAMATH_CALUDE_biker_journey_time_l3118_311845

/-- Given a biker's journey between two towns, prove the time taken for the first half. -/
theorem biker_journey_time (total_distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ) (second_half_time : ℝ) :
  total_distance = 140 →
  initial_speed = 14 →
  speed_increase = 2 →
  second_half_time = 7/3 →
  (total_distance / 2) / initial_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_time_l3118_311845


namespace NUMINAMATH_CALUDE_exists_abc_for_all_n_l3118_311857

def interval (k : ℕ) := Set.Ioo (k^2 : ℝ) (k^2 + k + 3 * Real.sqrt 3)

theorem exists_abc_for_all_n :
  ∀ (n : ℕ), ∃ (a b c : ℝ),
    (∃ (k₁ : ℕ), a ∈ interval k₁) ∧
    (∃ (k₂ : ℕ), b ∈ interval k₂) ∧
    (∃ (k₃ : ℕ), c ∈ interval k₃) ∧
    (n : ℝ) = a * b / c :=
by
  sorry


end NUMINAMATH_CALUDE_exists_abc_for_all_n_l3118_311857


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3118_311878

theorem complex_fraction_simplification :
  (5 - 3 * Complex.I) / (2 - 3 * Complex.I) = -19/5 - 9/5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3118_311878


namespace NUMINAMATH_CALUDE_mystery_number_proof_l3118_311864

theorem mystery_number_proof (mystery : ℕ) : mystery * 24 = 173 * 240 → mystery = 1730 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l3118_311864


namespace NUMINAMATH_CALUDE_system_equation_solution_l3118_311866

theorem system_equation_solution (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l3118_311866


namespace NUMINAMATH_CALUDE_unique_solution_base_6_l3118_311894

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

def decimal_to_base_6 (n : ℕ) : ℕ := 
  (n / 36) * 100 + ((n / 6) % 6) * 10 + (n % 6)

theorem unique_solution_base_6 :
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A < 6 ∧ B < 6 ∧ C < 6 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    base_6_to_decimal (100 * A + 10 * B + C) + base_6_to_decimal (10 * B + C) = 
      base_6_to_decimal (100 * A + 10 * C + A) ∧
    A = 3 ∧ B = 1 ∧ C = 2 ∧
    decimal_to_base_6 (A + B + C) = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_base_6_l3118_311894


namespace NUMINAMATH_CALUDE_certain_number_problem_l3118_311865

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.30 * 30 + 21 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3118_311865


namespace NUMINAMATH_CALUDE_sequence_problem_l3118_311861

/-- Given S_n = n^2 - 1 for all natural numbers n, prove that a_2016 = 4031 where a_n = S_n - S_(n-1) for n ≥ 2 -/
theorem sequence_problem (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
    (h1 : ∀ n, S n = n^2 - 1)
    (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 2016 = 4031 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3118_311861


namespace NUMINAMATH_CALUDE_mod_sixteen_equivalence_l3118_311809

theorem mod_sixteen_equivalence : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 15 ∧ m ≡ 12345 [ZMOD 16] ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_sixteen_equivalence_l3118_311809


namespace NUMINAMATH_CALUDE_pears_left_l3118_311896

theorem pears_left (keith_pears : ℕ) (total_pears : ℕ) : 
  keith_pears = 62 →
  total_pears = 186 →
  total_pears = keith_pears + 2 * keith_pears →
  140 = total_pears - (total_pears / 4) := by
  sorry

#check pears_left

end NUMINAMATH_CALUDE_pears_left_l3118_311896


namespace NUMINAMATH_CALUDE_area_of_triangle_APO_l3118_311875

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Area of a triangle given three points -/
def triangleArea (P Q R : Point) : ℝ := sorry

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Check if a point is on a line segment between two other points -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Check if a line bisects another line segment -/
def bisectsSegment (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem area_of_triangle_APO (ABCD : Parallelogram) (P Q O : Point) (k : ℝ) :
  parallelogramArea ABCD = k →
  bisectsSegment ABCD.D P ABCD.C O →
  bisectsSegment ABCD.A Q ABCD.B O →
  onSegment ABCD.A P ABCD.B →
  onSegment ABCD.C Q ABCD.D →
  triangleArea ABCD.A P O = k / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_APO_l3118_311875


namespace NUMINAMATH_CALUDE_plants_cost_theorem_l3118_311822

/-- Calculates the final cost of plants given the original price, discount rate, tax rate, and delivery surcharge. -/
def finalCost (originalPrice discountRate taxRate deliverySurcharge : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let withTax := discountedPrice * (1 + taxRate)
  withTax + deliverySurcharge

/-- Theorem stating that the final cost of the plants is $440.71 given the specified conditions. -/
theorem plants_cost_theorem :
  finalCost 467 0.15 0.08 12 = 440.71 := by
  sorry

#eval finalCost 467 0.15 0.08 12

end NUMINAMATH_CALUDE_plants_cost_theorem_l3118_311822


namespace NUMINAMATH_CALUDE_product_increase_theorem_l3118_311872

theorem product_increase_theorem :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) =
    13 * (a * b * c * d * e * f * g) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l3118_311872


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3118_311804

theorem sum_reciprocals (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a/b + b/c + c/a = 100) : 
  b/a + c/b + a/c = -101 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3118_311804


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_with_given_lengths_l3118_311807

/-- A quadrilateral that can be inscribed in a circle -/
structure InscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  diagonal : ℝ

/-- Theorem: A quadrilateral with side lengths 15, 36, 48, 27, and diagonal 54
    can be inscribed in a circle with diameter 54 -/
theorem inscribed_quadrilateral_with_given_lengths :
  ∃ (q : InscribedQuadrilateral),
    q.a = 15 ∧
    q.b = 36 ∧
    q.c = 48 ∧
    q.d = 27 ∧
    q.diagonal = 54 ∧
    (∃ (r : ℝ), r = 54 ∧ r = q.diagonal) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_with_given_lengths_l3118_311807


namespace NUMINAMATH_CALUDE_matrix_transformation_l3118_311828

/-- Given a 2x2 matrix M with eigenvector [1, 1] corresponding to eigenvalue 8,
    prove that the transformation of point (-1, 2) by M results in (-2, 4) -/
theorem matrix_transformation (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 4, b]
  (M.mulVec ![1, 1] = ![8, 8]) →
  (M.mulVec ![-1, 2] = ![-2, 4]) := by
sorry

end NUMINAMATH_CALUDE_matrix_transformation_l3118_311828


namespace NUMINAMATH_CALUDE_expression_simplification_l3118_311889

theorem expression_simplification (x : ℝ) : 
  (1 + Real.sin (2 * x) - Real.cos (2 * x)) / (1 + Real.sin (2 * x) + Real.cos (2 * x)) = Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3118_311889


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l3118_311882

-- Define the vectors a and b as functions of x
def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b (x : ℝ) : ℝ × ℝ := (x + 2, x - 4)

-- Define the perpendicularity condition
def perpendicular (x : ℝ) : Prop :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0

-- Theorem statement
theorem perpendicular_necessary_not_sufficient :
  (∀ x : ℝ, x = 2 → perpendicular x) ∧
  (∃ x : ℝ, perpendicular x ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l3118_311882


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l3118_311874

/-- The maximum number of cubes that can fit in a rectangular box -/
def max_cubes (box_length box_width box_height cube_volume : ℕ) : ℕ :=
  (box_length * box_width * box_height) / cube_volume

/-- Theorem stating the maximum number of 43 cm³ cubes in a 13x17x22 cm box -/
theorem max_cubes_in_box : max_cubes 13 17 22 43 = 114 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l3118_311874


namespace NUMINAMATH_CALUDE_total_sum_calculation_l3118_311895

theorem total_sum_calculation (share_a share_b share_c : ℝ) : 
  3 * share_a = 4 * share_b ∧ 
  3 * share_a = 7 * share_c ∧ 
  share_c = 83.99999999999999 → 
  share_a + share_b + share_c = 426.9999999999999 := by
sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l3118_311895


namespace NUMINAMATH_CALUDE_max_first_term_is_16_l3118_311873

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, n > 0 → a n > 0) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The maximum possible value of the first term in the special sequence is 16 -/
theorem max_first_term_is_16 (a : ℕ → ℝ) (h : SpecialSequence a) : 
  ∃ (M : ℝ), M = 16 ∧ a 1 ≤ M ∧ ∀ (b : ℕ → ℝ), SpecialSequence b → b 1 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_first_term_is_16_l3118_311873


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l3118_311801

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies peanut_pies : ℕ) : 
  total_pies = 48 →
  chocolate_pies ≥ total_pies / 2 →
  marshmallow_pies ≥ 2 * total_pies / 3 →
  cayenne_pies ≥ 3 * total_pies / 5 →
  peanut_pies ≥ total_pies / 8 →
  ∃ (pies_without_ingredients : ℕ), 
    pies_without_ingredients ≤ 16 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + peanut_pies ≥ total_pies :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l3118_311801


namespace NUMINAMATH_CALUDE_arun_weight_upper_bound_l3118_311854

-- Define the weight range according to Arun's opinion
def arun_lower_bound : ℝ := 66
def arun_upper_bound : ℝ := 72

-- Define the weight range according to Arun's brother's opinion
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70

-- Define the average weight
def average_weight : ℝ := 68

-- Define mother's upper bound (to be proven)
def mother_upper_bound : ℝ := 70

-- Theorem statement
theorem arun_weight_upper_bound :
  ∀ w : ℝ,
  (w > arun_lower_bound ∧ w < arun_upper_bound) →
  (w > brother_lower_bound ∧ w < brother_upper_bound) →
  (w ≤ mother_upper_bound) →
  (∃ w_min w_max : ℝ, 
    w_min > max arun_lower_bound brother_lower_bound ∧
    w_max < min arun_upper_bound brother_upper_bound ∧
    (w_min + w_max) / 2 = average_weight) →
  mother_upper_bound = 70 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_upper_bound_l3118_311854


namespace NUMINAMATH_CALUDE_scott_total_earnings_l3118_311853

/-- 
Proves that the total money Scott made from selling smoothies and cakes is $156, 
given the prices and quantities of items sold.
-/
theorem scott_total_earnings : 
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let smoothies_sold : ℕ := 40
  let cakes_sold : ℕ := 18
  
  smoothie_price * smoothies_sold + cake_price * cakes_sold = 156 := by
  sorry

end NUMINAMATH_CALUDE_scott_total_earnings_l3118_311853


namespace NUMINAMATH_CALUDE_telephone_bill_proof_l3118_311805

theorem telephone_bill_proof (F C : ℝ) : 
  F + C = 40 →
  F + 2*C = 76 →
  F + C = 40 := by
sorry

end NUMINAMATH_CALUDE_telephone_bill_proof_l3118_311805


namespace NUMINAMATH_CALUDE_correct_ball_arrangements_l3118_311802

/-- The number of ways to arrange 9 balls with 2 red, 3 yellow, and 4 white balls -/
def ballArrangements : ℕ := 2520

/-- The total number of balls -/
def totalBalls : ℕ := 9

/-- The number of red balls -/
def redBalls : ℕ := 2

/-- The number of yellow balls -/
def yellowBalls : ℕ := 3

/-- The number of white balls -/
def whiteBalls : ℕ := 4

theorem correct_ball_arrangements :
  ballArrangements = Nat.factorial totalBalls / (Nat.factorial redBalls * Nat.factorial yellowBalls * Nat.factorial whiteBalls) :=
by sorry

end NUMINAMATH_CALUDE_correct_ball_arrangements_l3118_311802


namespace NUMINAMATH_CALUDE_homeless_donation_distribution_l3118_311812

theorem homeless_donation_distribution (total spent second_set third_set first_set : ℚ) : 
  total = 900 ∧ second_set = 260 ∧ third_set = 315 ∧ 
  total = first_set + second_set + third_set →
  first_set = 325 := by sorry

end NUMINAMATH_CALUDE_homeless_donation_distribution_l3118_311812


namespace NUMINAMATH_CALUDE_second_volume_pages_l3118_311897

/-- Calculates the number of digits used to number pages up to n --/
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + (n - 9) * 2
  else 189 + (n - 99) * 3

/-- Represents the properties of the two volumes --/
structure TwoVolumes :=
  (first : ℕ)
  (second : ℕ)
  (total_digits : ℕ)
  (page_difference : ℕ)

/-- The main theorem about the number of pages in the second volume --/
theorem second_volume_pages (v : TwoVolumes) 
  (h1 : v.total_digits = 888)
  (h2 : v.second = v.first + v.page_difference)
  (h3 : v.page_difference = 8)
  (h4 : digits_used v.first + digits_used v.second = v.total_digits) :
  v.second = 170 := by
  sorry

#check second_volume_pages

end NUMINAMATH_CALUDE_second_volume_pages_l3118_311897
