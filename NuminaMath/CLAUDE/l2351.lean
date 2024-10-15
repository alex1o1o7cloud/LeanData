import Mathlib

namespace NUMINAMATH_CALUDE_circle_M_equation_l2351_235112

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define the point M
def point_M : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem circle_M_equation :
  (∃ (x y : ℝ), line_equation x y ∧ (x, y) = point_M) ∧
  circle_equation 3 0 ∧
  circle_equation 0 1 →
  ∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 1)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2351_235112


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_is_finite_set_of_points_l2351_235195

def goldfish_cost (n : ℕ) : ℚ := 20 * n + 10

def valid_purchase (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∃ (S : Set (ℕ × ℚ)),
    Finite S ∧
    (∀ p ∈ S, valid_purchase p.1 ∧ p.2 = goldfish_cost p.1) ∧
    (∀ n, valid_purchase n → (n, goldfish_cost n) ∈ S) :=
  sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_is_finite_set_of_points_l2351_235195


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2351_235116

/-- 
Given a rhombus with area 90 cm² and one diagonal of length 12 cm,
prove that the length of the other diagonal is 15 cm.
-/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 90 → d2 = 12 → area = (d1 * d2) / 2 → d1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2351_235116


namespace NUMINAMATH_CALUDE_count_eight_digit_cyclic_fixed_points_l2351_235134

def is_eight_digit (n : ℕ) : Prop := 10^7 ≤ n ∧ n < 10^8

def last_digit (n : ℕ) : ℕ := n % 10

def cyclic_permutation (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  (n % 10) * 10^(d-1) + n / 10

def iterative_permutation (n : ℕ) (k : ℕ) : ℕ :=
  Nat.iterate cyclic_permutation k n

theorem count_eight_digit_cyclic_fixed_points :
  (∃ (S : Finset ℕ), (∀ a ∈ S, is_eight_digit a ∧ last_digit a ≠ 0 ∧
    iterative_permutation a 4 = a) ∧ S.card = 9^4) := by sorry

end NUMINAMATH_CALUDE_count_eight_digit_cyclic_fixed_points_l2351_235134


namespace NUMINAMATH_CALUDE_robin_afternoon_bottles_l2351_235154

/-- The number of bottles Robin drank in the morning -/
def morning_bottles : ℕ := 7

/-- The total number of bottles Robin drank -/
def total_bottles : ℕ := 14

/-- The number of bottles Robin drank in the afternoon -/
def afternoon_bottles : ℕ := total_bottles - morning_bottles

theorem robin_afternoon_bottles : afternoon_bottles = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_afternoon_bottles_l2351_235154


namespace NUMINAMATH_CALUDE_max_consecutive_non_palindromic_l2351_235181

/-- A year is palindromic if it reads the same backward and forward -/
def isPalindromic (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999 ∧ 
  (year / 1000 = year % 10) ∧ ((year / 100) % 10 = (year / 10) % 10)

/-- The maximum number of consecutive non-palindromic years between 1000 and 9999 -/
def maxConsecutiveNonPalindromic : ℕ := 109

theorem max_consecutive_non_palindromic :
  ∀ (start : ℕ) (len : ℕ),
    start ≥ 1000 → start + len ≤ 9999 →
    (∀ y : ℕ, y ≥ start ∧ y < start + len → ¬isPalindromic y) →
    len ≤ maxConsecutiveNonPalindromic :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_non_palindromic_l2351_235181


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2351_235119

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2351_235119


namespace NUMINAMATH_CALUDE_product_of_roots_zero_l2351_235171

theorem product_of_roots_zero (x₁ x₂ : ℝ) : 
  ((-x₁^2 + 3*x₁ = 0) ∧ (-x₂^2 + 3*x₂ = 0)) → x₁ * x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_zero_l2351_235171


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l2351_235130

def initial_cupcakes : ℕ := 38
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 8

theorem todd_ate_cupcakes : 
  initial_cupcakes - packages * cupcakes_per_package = 14 := by
  sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l2351_235130


namespace NUMINAMATH_CALUDE_integer_sum_of_powers_l2351_235144

theorem integer_sum_of_powers (a b c : ℤ) 
  (h : (a - b)^10 + (a - c)^10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_powers_l2351_235144


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l2351_235128

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the height difference in inches between two people -/
def height_difference (height1_feet height1_inches height2_feet height2_inches : ℕ) : ℕ :=
  (height_to_inches height1_feet height1_inches) - (height_to_inches height2_feet height2_inches)

theorem vlad_sister_height_difference :
  height_difference 6 3 2 10 = 41 := by
  sorry

end NUMINAMATH_CALUDE_vlad_sister_height_difference_l2351_235128


namespace NUMINAMATH_CALUDE_carpet_transformation_possible_l2351_235162

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a cut piece of a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the state of the carpet after being cut -/
structure DamagedCarpet where
  original : Rectangle
  cutOut : CutPiece

/-- Function to check if a transformation from a damaged carpet to a new rectangle is possible -/
def canTransform (damaged : DamagedCarpet) (new : Rectangle) : Prop :=
  damaged.original.width * damaged.original.height - 
  damaged.cutOut.width * damaged.cutOut.height = 
  new.width * new.height

/-- The main theorem to prove -/
theorem carpet_transformation_possible : 
  ∃ (damaged : DamagedCarpet) (new : Rectangle),
    damaged.original = ⟨9, 12⟩ ∧ 
    damaged.cutOut = ⟨1, 8⟩ ∧
    new = ⟨10, 10⟩ ∧
    canTransform damaged new :=
sorry

end NUMINAMATH_CALUDE_carpet_transformation_possible_l2351_235162


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2351_235184

/-- Given that five identical bowling balls weigh the same as four identical canoes,
    and two canoes weigh 80 pounds, prove that one bowling ball weighs 32 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℕ),
    5 * bowling_ball_weight = 4 * canoe_weight →
    2 * canoe_weight = 80 →
    bowling_ball_weight = 32 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2351_235184


namespace NUMINAMATH_CALUDE_integer_divisibility_in_range_l2351_235100

theorem integer_divisibility_in_range (n : ℕ+) : 
  ∃ (a b c : ℤ), 
    (n : ℤ)^2 < a ∧ a < b ∧ b < c ∧ c < (n : ℤ)^2 + n + 3 * Real.sqrt n ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c) % a = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_in_range_l2351_235100


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2351_235152

theorem inequality_solution_set (x : ℝ) : 
  (8 - x^2 > 2*x) ↔ (-4 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2351_235152


namespace NUMINAMATH_CALUDE_min_value_of_max_function_l2351_235141

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ∃ (t : ℝ), t = 4 ∧ ∀ (s : ℝ), s ≥ max (x^2) (4 / (y * (x - y))) → s ≥ t :=
sorry

end NUMINAMATH_CALUDE_min_value_of_max_function_l2351_235141


namespace NUMINAMATH_CALUDE_no_valid_solutions_l2351_235153

theorem no_valid_solutions : ¬∃ (a b : ℝ), ∀ x, (a * x + b)^2 = 4 * x^2 + 4 * x + 4 := by sorry

end NUMINAMATH_CALUDE_no_valid_solutions_l2351_235153


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l2351_235156

/-- Given a polynomial g(x) that satisfies g(x + 1) - g(x) = 8x + 6 for all x,
    prove that the leading coefficient of g(x) is 4. -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) : 
  (∀ x, g (x + 1) - g x = 8 * x + 6) → 
  ∃ a b c : ℝ, (∀ x, g x = 4 * x^2 + a * x + b) ∧ c = 4 ∧ c * x^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l2351_235156


namespace NUMINAMATH_CALUDE_sqrt_40000_l2351_235166

theorem sqrt_40000 : Real.sqrt 40000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40000_l2351_235166


namespace NUMINAMATH_CALUDE_triangle_area_sum_properties_l2351_235196

/-- Represents a rectangular prism with dimensions 1, 2, and 3 -/
structure RectangularPrism where
  length : ℝ := 1
  width : ℝ := 2
  height : ℝ := 3

/-- Represents the sum of areas of all triangles with vertices on the prism -/
def triangleAreaSum (prism : RectangularPrism) : ℝ := sorry

/-- Theorem stating the properties of the triangle area sum -/
theorem triangle_area_sum_properties (prism : RectangularPrism) :
  ∃ (m a n : ℤ), 
    triangleAreaSum prism = m + a * Real.sqrt n ∧ 
    m + n + a = 49 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_sum_properties_l2351_235196


namespace NUMINAMATH_CALUDE_simplify_tan_cot_expression_l2351_235109

theorem simplify_tan_cot_expression :
  let tan_60 := Real.sqrt 3
  let cot_60 := 1 / Real.sqrt 3
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_cot_expression_l2351_235109


namespace NUMINAMATH_CALUDE_balloon_distribution_l2351_235193

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 22 → white = 40 → green = 70 → chartreuse = 90 → friends = 10 →
  (red + white + green + chartreuse) % friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2351_235193


namespace NUMINAMATH_CALUDE_median_sum_bounds_l2351_235118

/-- The sum of the medians of a triangle is less than its perimeter and greater than its semiperimeter -/
theorem median_sum_bounds (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hma : ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (hmb : mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (hmc : mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) :
  (a + b + c) / 2 < ma + mb + mc ∧ ma + mb + mc < a + b + c := by
  sorry

end NUMINAMATH_CALUDE_median_sum_bounds_l2351_235118


namespace NUMINAMATH_CALUDE_total_spending_is_638_l2351_235135

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 given the conditions -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_638_l2351_235135


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2351_235131

-- Problem 1
theorem simplify_expression_1 (m n : ℝ) : 
  (2*m + n)^2 - (4*m + 3*n)*(m - n) = 8*m*n + 4*n^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) 
  (h1 : x ≠ 3) (h2 : 2*x^2 - 5*x - 3 ≠ 0) : 
  ((2*x + 1)*(3*x - 4) / (2*x^2 - 5*x - 3) - 1) / ((4*x^2 - 1) / (x - 3)) = 1 / (2*x + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2351_235131


namespace NUMINAMATH_CALUDE_overlap_length_l2351_235140

/-- Given information about overlapping segments, prove the length of each overlap --/
theorem overlap_length (total_length : ℝ) (measured_length : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : measured_length = 83)
  (h3 : num_overlaps = 6) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = measured_length + num_overlaps * x :=
by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l2351_235140


namespace NUMINAMATH_CALUDE_vector_projection_l2351_235101

/-- Given vectors m and n, prove that the projection of m onto n is 8√13/13 -/
theorem vector_projection (m n : ℝ × ℝ) : m = (1, 2) → n = (2, 3) → 
  (m.1 * n.1 + m.2 * n.2) / Real.sqrt (n.1^2 + n.2^2) = 8 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2351_235101


namespace NUMINAMATH_CALUDE_qing_dynasty_problem_l2351_235169

/-- Represents the price of animals in ancient Chinese currency (taels) --/
structure AnimalPrices where
  horse : ℝ
  cattle : ℝ

/-- Represents a combination of horses and cattle --/
structure AnimalCombination where
  horses : ℕ
  cattle : ℕ

/-- Calculates the total cost of a combination of animals given their prices --/
def totalCost (prices : AnimalPrices) (combo : AnimalCombination) : ℝ :=
  prices.horse * combo.horses + prices.cattle * combo.cattle

/-- The theorem representing the original problem --/
theorem qing_dynasty_problem (prices : AnimalPrices) : 
  totalCost prices ⟨4, 6⟩ = 48 ∧ 
  totalCost prices ⟨2, 5⟩ = 38 ↔ 
  4 * prices.horse + 6 * prices.cattle = 48 ∧
  2 * prices.horse + 5 * prices.cattle = 38 := by
  sorry


end NUMINAMATH_CALUDE_qing_dynasty_problem_l2351_235169


namespace NUMINAMATH_CALUDE_derangement_of_five_l2351_235145

/-- Calculates the number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (k + 1) * (derangement (k + 1) + derangement k)

/-- The number of derangements for 5 elements is 44 -/
theorem derangement_of_five : derangement 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_derangement_of_five_l2351_235145


namespace NUMINAMATH_CALUDE_range_of_a_l2351_235106

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 5) / (x^2 - a) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (3 ∈ M a) ∧ (5 ∉ M a) ↔ (1 ≤ a ∧ a < 5/3) ∨ (9 < a ∧ a ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2351_235106


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2351_235187

theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 ∧ diagonal = 17 ∧ length^2 + width^2 = diagonal^2 →
  2 * (length + width) = 46 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2351_235187


namespace NUMINAMATH_CALUDE_nuts_division_l2351_235183

theorem nuts_division (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) 
  (h1 : bags = 65) 
  (h2 : nuts_per_bag = 15) 
  (h3 : students = 13) : 
  (bags * nuts_per_bag) / students = 75 := by
  sorry

#check nuts_division

end NUMINAMATH_CALUDE_nuts_division_l2351_235183


namespace NUMINAMATH_CALUDE_average_speed_first_part_is_35_l2351_235137

-- Define the total trip duration in hours
def total_trip_duration : ℝ := 24

-- Define the average speed for the entire trip in miles per hour
def average_speed_entire_trip : ℝ := 50

-- Define the duration of the first part of the trip in hours
def first_part_duration : ℝ := 4

-- Define the speed for the remaining part of the trip in miles per hour
def remaining_part_speed : ℝ := 53

-- Define the average speed for the first part of the trip
def average_speed_first_part : ℝ := 35

-- Theorem statement
theorem average_speed_first_part_is_35 :
  total_trip_duration * average_speed_entire_trip =
  first_part_duration * average_speed_first_part +
  (total_trip_duration - first_part_duration) * remaining_part_speed :=
by sorry

end NUMINAMATH_CALUDE_average_speed_first_part_is_35_l2351_235137


namespace NUMINAMATH_CALUDE_soccer_league_games_l2351_235127

/-- The number of games played in a soccer league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league of 12 teams, where each team plays 4 games with every other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l2351_235127


namespace NUMINAMATH_CALUDE_point_is_centroid_l2351_235110

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC in a real inner product space, if P is any point in the space and G satisfies PG = 1/3(PA + PB + PC), then G is the centroid of triangle ABC -/
theorem point_is_centroid (A B C P G : V) :
  (G - P) = (1 / 3 : ℝ) • ((A - P) + (B - P) + (C - P)) →
  G = (1 / 3 : ℝ) • (A + B + C) :=
sorry

end NUMINAMATH_CALUDE_point_is_centroid_l2351_235110


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2351_235102

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (Even x) → (x + 2)^2 - x^2 = 84 → x + (x + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2351_235102


namespace NUMINAMATH_CALUDE_babysitter_hours_l2351_235186

/-- The number of hours Milly hires the babysitter -/
def hours : ℕ := sorry

/-- The hourly rate of the current babysitter -/
def current_rate : ℕ := 16

/-- The hourly rate of the new babysitter -/
def new_rate : ℕ := 12

/-- The extra charge per scream for the new babysitter -/
def scream_charge : ℕ := 3

/-- The number of times the kids scream per babysitting gig -/
def scream_count : ℕ := 2

/-- The amount saved by switching to the new babysitter -/
def savings : ℕ := 18

theorem babysitter_hours : 
  current_rate * hours = new_rate * hours + scream_charge * scream_count + savings :=
by sorry

end NUMINAMATH_CALUDE_babysitter_hours_l2351_235186


namespace NUMINAMATH_CALUDE_probability_between_C_and_E_l2351_235164

/-- Given points A, B, C, D, E on a line segment AB, prove that the probability
    of a randomly selected point on AB being between C and E is 1/24. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < E ∧ E < D ∧ D < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  B - E = 2 * (E - C) →            -- BE = 2CE
  (E - C) / (B - A) = 1 / 24 := by
    sorry

end NUMINAMATH_CALUDE_probability_between_C_and_E_l2351_235164


namespace NUMINAMATH_CALUDE_square_areas_product_equality_l2351_235126

theorem square_areas_product_equality (α : Real) : 
  (Real.cos α)^4 * (Real.sin α)^4 = ((Real.cos α)^2 * (Real.sin α)^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_product_equality_l2351_235126


namespace NUMINAMATH_CALUDE_inequality_solution_l2351_235148

theorem inequality_solution :
  let f (x : ℝ) := x^3 - 3*x - 3/x + 1/x^3 + 5
  ∀ x : ℝ, (202 * Real.sqrt (f x) ≤ 0) ↔
    (x = (-1 - Real.sqrt 21 + Real.sqrt (2 * Real.sqrt 21 + 6)) / 4 ∨
     x = (-1 - Real.sqrt 21 - Real.sqrt (2 * Real.sqrt 21 + 6)) / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2351_235148


namespace NUMINAMATH_CALUDE_simplify_sum_of_roots_l2351_235172

theorem simplify_sum_of_roots : 
  Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) + Real.sqrt (1 + 2 + 3 + 4 + 5) + 2 = 
  Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 + Real.sqrt 15 + 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sum_of_roots_l2351_235172


namespace NUMINAMATH_CALUDE_min_questions_correct_l2351_235167

/-- Represents a company with N people, where one person knows everyone but is known by no one. -/
structure Company (N : ℕ) where
  -- The number of people in the company is at least 2
  people_count : N ≥ 2
  -- The function that determines if person i knows person j
  knows : Fin N → Fin N → Bool
  -- There exists a person who knows everyone else but is known by no one
  exists_z : ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z)

/-- The minimum number of questions needed to identify the person Z -/
def min_questions (N : ℕ) (c : Company N) : ℕ := N - 1

/-- Theorem stating that the minimum number of questions needed is N - 1 -/
theorem min_questions_correct (N : ℕ) (c : Company N) :
  ∀ strategy : (Fin N → Fin N → Bool) → Fin N,
  (∀ knows : Fin N → Fin N → Bool, 
   ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z) →
   ∃ questions : Finset (Fin N × Fin N),
     questions.card ≥ min_questions N c ∧
     strategy knows = z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_questions_correct_l2351_235167


namespace NUMINAMATH_CALUDE_investment_change_l2351_235188

/-- Proves that an investment of $200 with a 20% loss followed by a 25% gain results in 0% change --/
theorem investment_change (initial_investment : ℝ) (first_year_loss_percent : ℝ) (second_year_gain_percent : ℝ) :
  initial_investment = 200 →
  first_year_loss_percent = 20 →
  second_year_gain_percent = 25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let final_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  final_amount = initial_investment := by
  sorry

#check investment_change

end NUMINAMATH_CALUDE_investment_change_l2351_235188


namespace NUMINAMATH_CALUDE_systematic_sample_count_l2351_235151

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def count_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) / (s.total_population / s.sample_size))

/-- Theorem stating that for the given parameters, 13 individuals are sampled from the interval -/
theorem systematic_sample_count (s : SystematicSample) 
  (h1 : s.total_population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 461)
  (h4 : s.interval_end = 720) :
  count_in_interval s = 13 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_count_l2351_235151


namespace NUMINAMATH_CALUDE_chips_ratio_l2351_235139

-- Define the total number of bags
def total_bags : ℕ := 3

-- Define the number of bags eaten for dinner
def dinner_bags : ℕ := 1

-- Define the number of bags eaten after dinner
def after_dinner_bags : ℕ := total_bags - dinner_bags

-- Theorem to prove
theorem chips_ratio :
  (after_dinner_bags : ℚ) / (dinner_bags : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_chips_ratio_l2351_235139


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2351_235180

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2351_235180


namespace NUMINAMATH_CALUDE_three_correct_letters_probability_l2351_235117

/-- The number of people and letters --/
def n : ℕ := 5

/-- The number of people who receive the correct letter --/
def k : ℕ := 3

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people --/
def prob_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.factorial (n - k)) / Nat.factorial n

theorem three_correct_letters_probability :
  prob_correct_letters n k = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_three_correct_letters_probability_l2351_235117


namespace NUMINAMATH_CALUDE_b_oxen_count_main_theorem_l2351_235111

/-- Represents the number of oxen and months for each person --/
structure Grazing :=
  (oxen : ℕ)
  (months : ℕ)

/-- Calculates the total grazing cost --/
def total_cost (a b c : Grazing) (cost_per_ox_month : ℚ) : ℚ :=
  (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months : ℚ) * cost_per_ox_month

/-- Theorem: Given the conditions, b put 12 oxen for grazing --/
theorem b_oxen_count (total_rent : ℚ) (c_rent : ℚ) : ℕ :=
  let a : Grazing := ⟨10, 7⟩
  let b : Grazing := ⟨12, 5⟩  -- We claim b put 12 oxen
  let c : Grazing := ⟨15, 3⟩
  let cost_per_ox_month : ℚ := c_rent / (c.oxen * c.months)
  have h1 : total_cost a b c cost_per_ox_month = total_rent := by sorry
  have h2 : c.oxen * c.months * cost_per_ox_month = c_rent := by sorry
  b.oxen

/-- The main theorem stating that given the conditions, b put 12 oxen for grazing --/
theorem main_theorem : b_oxen_count 140 36 = 12 := by sorry

end NUMINAMATH_CALUDE_b_oxen_count_main_theorem_l2351_235111


namespace NUMINAMATH_CALUDE_tan_equality_in_range_l2351_235157

theorem tan_equality_in_range (m : ℤ) :
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1230 * π / 180) →
  m = -30 := by sorry

end NUMINAMATH_CALUDE_tan_equality_in_range_l2351_235157


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2351_235113

-- Problem 1
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2351_235113


namespace NUMINAMATH_CALUDE_smallest_number_l2351_235190

theorem smallest_number : 
  -5 < -Real.pi ∧ -5 < -Real.sqrt 3 ∧ -5 < 0 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2351_235190


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2351_235179

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 2) : 
  x + y ≥ 9/2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 2 ∧ x + y = 9/2 :=
by
  sorry

#check min_value_of_sum

end NUMINAMATH_CALUDE_min_value_of_sum_l2351_235179


namespace NUMINAMATH_CALUDE_z_shaped_area_l2351_235173

/-- The area of a Z-shaped region formed by subtracting two squares from a rectangle -/
theorem z_shaped_area (rectangle_length rectangle_width square1_side square2_side : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : square1_side = 2)
  (h4 : square2_side = 1) :
  rectangle_length * rectangle_width - (square1_side^2 + square2_side^2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_z_shaped_area_l2351_235173


namespace NUMINAMATH_CALUDE_specific_hexahedron_volume_l2351_235120

/-- A regular hexahedron with specific dimensions -/
structure RegularHexahedron where
  -- Base edge length
  ab : ℝ
  -- Top edge length
  a₁b₁ : ℝ
  -- Height
  aa₁ : ℝ
  -- Regularity conditions
  ab_positive : 0 < ab
  a₁b₁_positive : 0 < a₁b₁
  aa₁_positive : 0 < aa₁

/-- The volume of a regular hexahedron -/
def volume (h : RegularHexahedron) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem stating the volume of the specific hexahedron -/
theorem specific_hexahedron_volume :
  ∃ (h : RegularHexahedron),
    h.ab = 2 ∧
    h.a₁b₁ = 3 ∧
    h.aa₁ = Real.sqrt 10 ∧
    volume h = (57 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexahedron_volume_l2351_235120


namespace NUMINAMATH_CALUDE_dataset_mode_is_five_l2351_235108

def dataset : List ℕ := [0, 1, 2, 3, 3, 5, 5, 5]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode_is_five : mode dataset = 5 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_is_five_l2351_235108


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2351_235138

def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x > 1}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ici (2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2351_235138


namespace NUMINAMATH_CALUDE_reduced_journey_time_l2351_235103

/-- Calculates the reduced time of a journey when speed is increased -/
theorem reduced_journey_time 
  (original_time : ℝ) 
  (original_speed : ℝ) 
  (new_speed : ℝ) 
  (h1 : original_time = 50) 
  (h2 : original_speed = 48) 
  (h3 : new_speed = 60) : 
  (original_time * original_speed) / new_speed = 40 := by
  sorry

end NUMINAMATH_CALUDE_reduced_journey_time_l2351_235103


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_radius_l2351_235170

/-- Given a quadrilateral ABCD inscribed in a circle with diagonals intersecting at M,
    where AB = a, CD = b, and ∠AMB = α, the radius R of the circle is as follows. -/
theorem inscribed_quadrilateral_radius 
  (a b : ℝ) (α : ℝ) (ha : a > 0) (hb : b > 0) (hα : 0 < α ∧ α < π) :
  ∃ (R : ℝ), R = (Real.sqrt (a^2 + b^2 + 2*a*b*(Real.cos α))) / (2 * Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_radius_l2351_235170


namespace NUMINAMATH_CALUDE_shooter_prob_below_8_l2351_235175

-- Define the probabilities
def prob_10_ring : ℝ := 0.20
def prob_9_ring : ℝ := 0.30
def prob_8_ring : ℝ := 0.10

-- Define the probability of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10_ring + prob_9_ring + prob_8_ring)

-- Theorem statement
theorem shooter_prob_below_8 : prob_below_8 = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_shooter_prob_below_8_l2351_235175


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l2351_235178

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (perpendicular : P → L → Prop) -- Plane is perpendicular to a line
variable (parallel_planes : P → P → Prop) -- Two planes are parallel
variable (parallel_line_plane : L → P → Prop) -- A line is parallel to a plane
variable (line_in_plane : L → P → Prop) -- A line is in a plane

-- Theorem 1: If two planes are both perpendicular to the same line, then these two planes are parallel
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : perpendicular p1 l) 
  (h2 : perpendicular p2 l) : 
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: If two planes are parallel to each other, then a line in one of the planes is parallel to the other plane
theorem parallel_planes_imply_line_parallel 
  (p1 p2 : P) (l : L) 
  (h1 : parallel_planes p1 p2) 
  (h2 : line_in_plane l p1) : 
  parallel_line_plane l p2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_parallel_planes_imply_line_parallel_l2351_235178


namespace NUMINAMATH_CALUDE_abc_mod_five_l2351_235159

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 0 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 3 →
  (a*b*c) % 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_five_l2351_235159


namespace NUMINAMATH_CALUDE_complex_simplification_l2351_235165

theorem complex_simplification :
  (4 - 2*Complex.I) - (7 - 2*Complex.I) + (6 - 3*Complex.I) = 3 - 3*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2351_235165


namespace NUMINAMATH_CALUDE_A_intersect_B_l2351_235168

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| ≥ 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2351_235168


namespace NUMINAMATH_CALUDE_fraction_simplification_l2351_235177

theorem fraction_simplification (a b c : ℝ) (h : 2*a - 3*c - 4 + b ≠ 0) :
  (6*a^2 - 2*b^2 + 6*c^2 + a*b - 13*a*c - 4*b*c - 18*a - 5*b + 17*c + 12) / 
  (4*a^2 - b^2 + 9*c^2 - 12*a*c - 16*a + 24*c + 16) = 
  (3*a - 2*c - 3 + 2*b) / (2*a - 3*c - 4 + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2351_235177


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l2351_235161

/-- Given a natural number, returns the number of its digits -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number formed by reversing its digits -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number with an odd number of digits, 
    the difference between the number and its reverse is divisible by 99 -/
theorem difference_divisible_by_99 (n : ℕ) (h : Odd (numDigits n)) :
  99 ∣ (n - reverseDigits n) := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_99_l2351_235161


namespace NUMINAMATH_CALUDE_unique_right_triangle_area_twice_perimeter_l2351_235185

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  hyp : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- The area of a right triangle is equal to twice its perimeter -/
def areaEqualsTwicePerimeter (t : RightTriangle) : Prop :=
  (t.a * t.b : ℕ) = 4 * (t.a + t.b + t.c)

/-- There exists exactly one right triangle with integer leg lengths
    where the area is equal to twice the perimeter -/
theorem unique_right_triangle_area_twice_perimeter :
  ∃! t : RightTriangle, areaEqualsTwicePerimeter t := by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_area_twice_perimeter_l2351_235185


namespace NUMINAMATH_CALUDE_class_size_l2351_235149

def is_valid_total_score (n : ℕ) : Prop :=
  n ≥ 4460 ∧ n < 4470 ∧ n % 100 = 64 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem class_size (total_score : ℕ) (h1 : is_valid_total_score total_score) 
  (h2 : (total_score : ℚ) / 72 = 62) : 
  ∃ (num_students : ℕ), (num_students : ℚ) = total_score / 72 := by
sorry

end NUMINAMATH_CALUDE_class_size_l2351_235149


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2351_235155

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2351_235155


namespace NUMINAMATH_CALUDE_min_value_expression_l2351_235121

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 ∧
  ((6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) = 3 ↔ 3 * x = y ∧ y = 3 * z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2351_235121


namespace NUMINAMATH_CALUDE_complete_sets_l2351_235132

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_sets :
  ∀ A : Set ℕ, A.Nonempty →
    (is_complete A ↔ 
      A = {1} ∨ 
      A = {1, 2} ∨ 
      A = {1, 2, 3, 4} ∨ 
      A = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_complete_sets_l2351_235132


namespace NUMINAMATH_CALUDE_cost_price_equals_selling_price_l2351_235125

/-- The number of articles whose selling price equals the cost price of 20 articles -/
def x : ℚ :=
  16

/-- The profit percentage -/
def profit_percentage : ℚ :=
  25 / 100

theorem cost_price_equals_selling_price (C : ℚ) (h : C > 0) :
  20 * C = x * C * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_equals_selling_price_l2351_235125


namespace NUMINAMATH_CALUDE_bottles_used_first_game_l2351_235198

theorem bottles_used_first_game 
  (total_bottles : ℕ)
  (bottles_left : ℕ)
  (bottles_used_second : ℕ)
  (h1 : total_bottles = 200)
  (h2 : bottles_left = 20)
  (h3 : bottles_used_second = 110) :
  total_bottles - bottles_left - bottles_used_second = 70 :=
by sorry

end NUMINAMATH_CALUDE_bottles_used_first_game_l2351_235198


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2351_235124

theorem quadratic_inequality_solution_range (t : ℝ) :
  (∃ c : ℝ, c ≤ 1 ∧ c^2 - 3*c + t ≤ 0) → t ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2351_235124


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2351_235146

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : Float :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | "O" => 16.00
  | "C" => 12.01
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms for each element in the compound -/
def atom_count (element : String) : Nat :=
  match element with
  | "N" => 2
  | "H" => 6
  | "Br" => 1
  | "O" => 1
  | "C" => 3
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : Float :=
  ["N", "H", "Br", "O", "C"].map (fun e => (atomic_weight e) * (atom_count e).toFloat)
    |> List.sum

/-- Theorem: The molecular weight of the compound is approximately 166.01 amu -/
theorem compound_molecular_weight :
  (molecular_weight - 166.01).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2351_235146


namespace NUMINAMATH_CALUDE_book_selling_price_total_selling_price_is_595_l2351_235174

/-- Calculates the total selling price of two books given the following conditions:
    - Total cost of two books is 600
    - First book is sold at a loss of 15%
    - Second book is sold at a gain of 19%
    - Cost of the book sold at a loss is 350
-/
theorem book_selling_price (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage : ℝ) (loss_book_cost : ℝ) : ℝ :=
  let selling_price_loss_book := loss_book_cost * (1 - loss_percentage / 100)
  let gain_book_cost := total_cost - loss_book_cost
  let selling_price_gain_book := gain_book_cost * (1 + gain_percentage / 100)
  selling_price_loss_book + selling_price_gain_book

theorem total_selling_price_is_595 :
  book_selling_price 600 15 19 350 = 595 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_total_selling_price_is_595_l2351_235174


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l2351_235136

theorem smallest_side_of_triangle (x : ℝ) : 
  10 + (3*x + 6) + (x + 5) = 60 →
  10 ≤ 3*x + 6 ∧ 10 ≤ x + 5 →
  10 = min 10 (min (3*x + 6) (x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l2351_235136


namespace NUMINAMATH_CALUDE_prime_divisibility_l2351_235114

theorem prime_divisibility (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (4 * n + 1)) :
  (4 * n + 1) ∣ (n^(2*n) - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2351_235114


namespace NUMINAMATH_CALUDE_library_books_theorem_l2351_235143

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new edition
variable (is_new_edition : Book → Prop)

-- Theorem stating that if not all books are new editions,
-- then there exists a book that is not a new edition and not all books are new editions
theorem library_books_theorem (h : ¬ ∀ b : Book, is_new_edition b) :
  (∃ b : Book, ¬ is_new_edition b) ∧ (¬ ∀ b : Book, is_new_edition b) :=
by sorry

end NUMINAMATH_CALUDE_library_books_theorem_l2351_235143


namespace NUMINAMATH_CALUDE_competition_participants_count_l2351_235129

/-- Represents the math competition scenario -/
structure Competition where
  fullScore : ℕ
  initialGoldThreshold : ℕ
  initialSilverLowerThreshold : ℕ
  initialSilverUpperThreshold : ℕ
  changedGoldThreshold : ℕ
  changedSilverLowerThreshold : ℕ
  changedSilverUpperThreshold : ℕ
  initialGoldCount : ℕ
  initialSilverCount : ℕ
  nonMedalCount : ℕ
  changedGoldCount : ℕ
  changedSilverCount : ℕ
  changedGoldAverage : ℕ
  changedSilverAverage : ℕ

/-- The theorem to be proved -/
theorem competition_participants_count (c : Competition) 
  (h1 : c.fullScore = 120)
  (h2 : c.initialGoldThreshold = 100)
  (h3 : c.initialSilverLowerThreshold = 80)
  (h4 : c.initialSilverUpperThreshold = 99)
  (h5 : c.changedGoldThreshold = 90)
  (h6 : c.changedSilverLowerThreshold = 70)
  (h7 : c.changedSilverUpperThreshold = 89)
  (h8 : c.initialSilverCount = c.initialGoldCount + 8)
  (h9 : c.nonMedalCount = c.initialGoldCount + c.initialSilverCount + 9)
  (h10 : c.changedGoldCount = c.initialGoldCount + 5)
  (h11 : c.changedSilverCount = c.initialSilverCount + 5)
  (h12 : c.changedGoldCount * c.changedGoldAverage = c.changedSilverCount * c.changedSilverAverage)
  (h13 : c.changedGoldAverage = 95)
  (h14 : c.changedSilverAverage = 75) :
  c.initialGoldCount + c.initialSilverCount + c.nonMedalCount = 125 :=
sorry


end NUMINAMATH_CALUDE_competition_participants_count_l2351_235129


namespace NUMINAMATH_CALUDE_motorcycle_price_l2351_235104

theorem motorcycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 400 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 2000 := by
sorry

end NUMINAMATH_CALUDE_motorcycle_price_l2351_235104


namespace NUMINAMATH_CALUDE_infinitely_many_square_sum_square_no_zero_l2351_235158

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Check if a number contains no zero digits -/
def no_zero_digits (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem infinitely_many_square_sum_square_no_zero :
  ∃ f : ℕ → ℕ, 
    (∀ m : ℕ, ∃ k : ℕ, f m = k^2) ∧ 
    (∀ m : ℕ, ∃ l : ℕ, S (f m) = l^2) ∧
    (∀ m : ℕ, no_zero_digits (f m)) ∧
    Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_square_sum_square_no_zero_l2351_235158


namespace NUMINAMATH_CALUDE_equation_solution_l2351_235123

theorem equation_solution : 
  ∃ x : ℝ, (7 + 3.5 * x = 2.1 * x - 25) ∧ (x = -32 / 1.4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2351_235123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2351_235189

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 14 = 2) :
  a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2351_235189


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2351_235122

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the condition for F being outside the circle with diameter CD
def F_outside_circle (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ > 0

theorem parabola_line_intersection (a : ℝ) :
  a < 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ a ∧ line x₂ y₂ a ∧
    F_outside_circle x₁ y₁ x₂ y₂) ↔
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2351_235122


namespace NUMINAMATH_CALUDE_chicken_coop_max_area_l2351_235133

/-- The maximum area of a rectangular chicken coop with one side against a wall --/
theorem chicken_coop_max_area :
  let wall_length : ℝ := 15
  let fence_length : ℝ := 24
  let area (x : ℝ) : ℝ := x * (fence_length - x) / 2
  let max_area : ℝ := 72
  ∀ x, 0 < x ∧ x ≤ wall_length → area x ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_chicken_coop_max_area_l2351_235133


namespace NUMINAMATH_CALUDE_digit_count_8_pow_12_times_5_pow_18_l2351_235194

theorem digit_count_8_pow_12_times_5_pow_18 : 
  (Nat.log 10 (8^12 * 5^18) + 1 : ℕ) = 24 := by sorry

end NUMINAMATH_CALUDE_digit_count_8_pow_12_times_5_pow_18_l2351_235194


namespace NUMINAMATH_CALUDE_problem_1_l2351_235142

theorem problem_1 (x y z : ℝ) : -x * y^2 * z^3 * (-x^2 * y)^3 = x^7 * y^5 * z^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2351_235142


namespace NUMINAMATH_CALUDE_line_equation_l2351_235192

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 5 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection points
def intersection (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola x y ∧ line k x y

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection k x₁ y₁ ∧
    intersection k x₂ y₂ ∧
    (x₁ + x₂) / 2 = -2/3

theorem line_equation :
  ∀ k : ℝ, midpoint_condition k → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l2351_235192


namespace NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2351_235191

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) : 
  a = 1/8 → b = 1/4 → c = a / b → c = 1/2 := by sorry

end NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2351_235191


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2351_235107

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2351_235107


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2351_235199

/-- An arithmetic sequence with first term -1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  2 * n - 17

/-- The sum of the first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℤ :=
  n^2 - 6*n

theorem arithmetic_sequence_properties :
  ∀ n : ℕ,
  (n > 0) →
  (arithmetic_sequence n = 2 * n - 17) ∧
  (sequence_sum n = n^2 - 6*n) ∧
  (∀ t : ℝ, (∀ k : ℕ, k > 0 → sequence_sum k > t) ↔ t < -6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2351_235199


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_pentagon_l2351_235115

/-- The measure of the largest angle in a pentagon with specific angle conditions -/
theorem largest_angle_in_specific_pentagon : 
  ∀ (A B C D E x : ℝ),
  -- Pentagon conditions
  A + B + C + D + E = 540 →
  -- Specific angle conditions
  A = 70 →
  B = 90 →
  C = D →
  E = 3 * x - 10 →
  C = x →
  -- Conclusion
  max A (max B (max C (max D E))) = 224 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_specific_pentagon_l2351_235115


namespace NUMINAMATH_CALUDE_g_determinant_identity_g_1002_1004_minus_1003_squared_l2351_235197

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

-- Define the sequence G
def G : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => G (n + 1) - G n

-- State the theorem
theorem g_determinant_identity (n : ℕ) :
  A ^ n = !![G (n + 1), G n; G n, G (n - 1)] →
  G n * G (n + 2) - G (n + 1) ^ 2 = 1 := by
  sorry

-- The specific case for n = 1003
theorem g_1002_1004_minus_1003_squared :
  G 1002 * G 1004 - G 1003 ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_determinant_identity_g_1002_1004_minus_1003_squared_l2351_235197


namespace NUMINAMATH_CALUDE_min_difference_is_1747_l2351_235150

/-- Represents a valid digit assignment for the problem -/
structure DigitAssignment where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                  d ≠ e ∧ d ≠ f ∧
                  e ≠ f
  all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0
  sum_constraint : 1000 * a + 100 * b + 10 * c + d + 10 * e + f = 2010

/-- The main theorem stating the minimum difference -/
theorem min_difference_is_1747 : 
  ∀ (assign : DigitAssignment), 
    1000 * assign.a + 100 * assign.b + 10 * assign.c + assign.d - (10 * assign.e + assign.f) = 1747 := by
  sorry

end NUMINAMATH_CALUDE_min_difference_is_1747_l2351_235150


namespace NUMINAMATH_CALUDE_percentage_less_l2351_235163

theorem percentage_less (x y : ℝ) (h : x = 5 * y) : (x - y) / x * 100 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_less_l2351_235163


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l2351_235105

def carnival_spending (total_budget food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  total_budget - total_spent

theorem carnival_spending_theorem :
  carnival_spending 100 20 = 40 :=
by sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l2351_235105


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_four_l2351_235176

theorem max_value_implies_a_equals_four :
  ∀ a b c : ℕ,
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (∀ x y z : ℕ, 
    x ∈ ({1, 2, 4} : Set ℕ) →
    y ∈ ({1, 2, 4} : Set ℕ) →
    z ∈ ({1, 2, 4} : Set ℕ) →
    x ≠ y → y ≠ z → x ≠ z →
    (a / 2 : ℚ) / (b / c : ℚ) ≥ (x / 2 : ℚ) / (y / z : ℚ)) →
  (a / 2 : ℚ) / (b / c : ℚ) = 4 →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_four_l2351_235176


namespace NUMINAMATH_CALUDE_village_population_l2351_235147

theorem village_population (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 3312) → P = 4600 := by sorry

end NUMINAMATH_CALUDE_village_population_l2351_235147


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2351_235182

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2351_235182


namespace NUMINAMATH_CALUDE_total_games_played_l2351_235160

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem total_games_played (first_100_win_rate : ℝ) (remaining_win_rate : ℝ) (total_win_rate : ℝ) 
  (h1 : first_100_win_rate = 0.85)
  (h2 : remaining_win_rate = 0.5)
  (h3 : total_win_rate = 0.7) :
  ∃ (total_games : ℕ), 
    total_games = 175 ∧ 
    (first_100_win_rate * 100 + remaining_win_rate * (total_games - 100 : ℝ)) / total_games = total_win_rate :=
by sorry

end NUMINAMATH_CALUDE_total_games_played_l2351_235160
