import Mathlib

namespace absolute_value_inequality_l2440_244078

theorem absolute_value_inequality (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 4 ↔ x ∈ Set.Icc 1 3 ∪ Set.Icc 7 9 := by
  sorry

end absolute_value_inequality_l2440_244078


namespace exists_cube_with_2014_prime_points_l2440_244025

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Predicate to check if a number is prime -/
def isPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a point is in the first octant -/
def isFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Predicate to check if a point has all prime coordinates -/
def isPrimePoint (p : Point3D) : Prop :=
  isPrime p.x ∧ isPrime p.y ∧ isPrime p.z

/-- Definition of a cube in 3D space -/
structure Cube where
  corner : Point3D
  edgeLength : ℤ

/-- Predicate to check if a point is inside a cube -/
def isInsideCube (p : Point3D) (c : Cube) : Prop :=
  c.corner.x ≤ p.x ∧ p.x < c.corner.x + c.edgeLength ∧
  c.corner.y ≤ p.y ∧ p.y < c.corner.y + c.edgeLength ∧
  c.corner.z ≤ p.z ∧ p.z < c.corner.z + c.edgeLength

/-- The main theorem to be proved -/
theorem exists_cube_with_2014_prime_points :
  ∃ (c : Cube), c.edgeLength = 2014 ∧
    isFirstOctant c.corner ∧
    (∃ (points : Finset Point3D),
      points.card = 2014 ∧
      (∀ p ∈ points, isPrimePoint p ∧ isInsideCube p c) ∧
      (∀ p : Point3D, isPrimePoint p ∧ isInsideCube p c → p ∈ points)) :=
sorry

end exists_cube_with_2014_prime_points_l2440_244025


namespace intersection_M_P_l2440_244080

-- Define the sets M and P
def M (a : ℝ) : Set ℝ := {x | x > a ∧ a^2 - 12*a + 20 < 0}
def P : Set ℝ := {x | x ≤ 10}

-- Theorem statement
theorem intersection_M_P (a : ℝ) : M a ∩ P = {x | a < x ∧ x ≤ 10} :=
by sorry

end intersection_M_P_l2440_244080


namespace other_sales_percentage_l2440_244030

/-- Represents the sales distribution of the Dreamy Bookstore for April -/
structure SalesDistribution where
  notebooks : ℝ
  bookmarks : ℝ
  other : ℝ

/-- The sales distribution for the Dreamy Bookstore in April -/
def april_sales : SalesDistribution where
  notebooks := 45
  bookmarks := 25
  other := 100 - (45 + 25)

/-- Theorem stating that the percentage of sales that were neither notebooks nor bookmarks is 30% -/
theorem other_sales_percentage (s : SalesDistribution) 
  (h1 : s.notebooks = 45)
  (h2 : s.bookmarks = 25)
  (h3 : s.notebooks + s.bookmarks + s.other = 100) :
  s.other = 30 := by
  sorry

#eval april_sales.other

end other_sales_percentage_l2440_244030


namespace min_value_theorem_l2440_244065

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ y / x + 4 / y = 8 := by
  sorry

end min_value_theorem_l2440_244065


namespace prob_ace_jack_queen_is_8_over_16575_l2440_244058

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of Aces in a standard deck. -/
def NumAces : ℕ := 4

/-- The number of Jacks in a standard deck. -/
def NumJacks : ℕ := 4

/-- The number of Queens in a standard deck. -/
def NumQueens : ℕ := 4

/-- The probability of drawing an Ace, then a Jack, then a Queen from a standard deck without replacement. -/
def probAceJackQueen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_ace_jack_queen_is_8_over_16575 :
  probAceJackQueen = 8 / 16575 := by
  sorry

end prob_ace_jack_queen_is_8_over_16575_l2440_244058


namespace adqr_is_cyclic_l2440_244004

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point lies on a line segment between two other points -/
def point_on_segment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def segments_equal (A B C D : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem adqr_is_cyclic 
  (A B C D P Q R T : Point)
  (h_convex : is_convex ⟨A, B, C, D⟩)
  (h_equal1 : segments_equal A P P T)
  (h_equal2 : segments_equal P T T D)
  (h_equal3 : segments_equal Q B B C)
  (h_equal4 : segments_equal B C C R)
  (h_on_AB1 : point_on_segment A P B)
  (h_on_AB2 : point_on_segment A Q B)
  (h_on_CD1 : point_on_segment C R D)
  (h_on_CD2 : point_on_segment C T D)
  (h_bctp_cyclic : is_cyclic ⟨B, C, T, P⟩) :
  is_cyclic ⟨A, D, Q, R⟩ :=
sorry

end adqr_is_cyclic_l2440_244004


namespace lastTwoDigits_7_2012_l2440_244057

/-- The last two digits of 7^n, for any natural number n -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The pattern of last two digits repeats every 4 exponents -/
axiom lastTwoDigitsPattern (k : ℕ) :
  (lastTwoDigits (4*k - 2) = 49) ∧
  (lastTwoDigits (4*k - 1) = 43) ∧
  (lastTwoDigits (4*k) = 1) ∧
  (lastTwoDigits (4*k + 1) = 7)

theorem lastTwoDigits_7_2012 :
  lastTwoDigits 2012 = 1 := by sorry

end lastTwoDigits_7_2012_l2440_244057


namespace quadratic_equation_roots_l2440_244013

theorem quadratic_equation_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (x₁^2 + 4*x₁ + 3 = 0) ∧ 
  (x₂^2 + 4*x₂ + 3 = 0) :=
by sorry

end quadratic_equation_roots_l2440_244013


namespace min_value_expression_l2440_244039

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 6 * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = 6 * Real.sqrt 3 ↔ 
    a^2 = 1/6 ∧ b = -1/(2*a) ∧ c = 2*a) :=
sorry

end min_value_expression_l2440_244039


namespace triangle_inequality_sum_l2440_244076

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  c / (a + b) + a / (b + c) + b / (c + a) > 1 := by
  sorry

end triangle_inequality_sum_l2440_244076


namespace smallest_right_triangle_area_l2440_244086

/-- The smallest area of a right triangle with sides 6 and 8 -/
theorem smallest_right_triangle_area :
  let sides : Finset ℝ := {6, 8}
  ∃ (a b c : ℝ), a ∈ sides ∧ b ∈ sides ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (∀ (x y z : ℝ), x ∈ sides → y ∈ sides → z > 0 → x^2 + y^2 = z^2 →
      (1/2) * a * b ≤ (1/2) * x * y) ∧
    (1/2) * a * b = 6 * Real.sqrt 7 :=
by sorry

end smallest_right_triangle_area_l2440_244086


namespace zongzi_survey_measure_l2440_244017

-- Define the types of statistical measures
inductive StatMeasure
| Variance
| Mean
| Median
| Mode

-- Define a function that determines the most appropriate measure
def most_appropriate_measure (survey_goal : String) (data_type : String) : StatMeasure :=
  if survey_goal = "determine most preferred" && data_type = "categorical" then
    StatMeasure.Mode
  else
    StatMeasure.Mean  -- Default to mean for other cases

-- Theorem statement
theorem zongzi_survey_measure :
  most_appropriate_measure "determine most preferred" "categorical" = StatMeasure.Mode :=
by sorry

end zongzi_survey_measure_l2440_244017


namespace digit2021_is_one_l2440_244046

/-- The sequence of digits formed by concatenating natural numbers starting from 1 -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2021st digit in the sequence -/
def digit2021 : ℕ := digitSequence 2021

theorem digit2021_is_one : digit2021 = 1 := by
  sorry

end digit2021_is_one_l2440_244046


namespace unique_function_solution_l2440_244060

/-- The functional equation f(x + f(y)) = x + y + k has exactly one solution. -/
theorem unique_function_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, ∀ x y : ℝ, f (x + f y) = x + y + k :=
by sorry

end unique_function_solution_l2440_244060


namespace student_sums_correct_l2440_244099

theorem student_sums_correct (total : ℕ) (correct : ℕ) (wrong : ℕ) : 
  total = 54 → 
  wrong = 2 * correct → 
  total = correct + wrong → 
  correct = 18 := by
sorry

end student_sums_correct_l2440_244099


namespace semicircle_in_right_triangle_l2440_244005

/-- Given a right-angled triangle with an inscribed semicircle, where:
    - The semicircle has radius r
    - The shorter edges of the triangle are tangent to the semicircle and have lengths a and b
    - The diameter of the semicircle lies on the hypotenuse of the triangle
    Then: 1/r = 1/a + 1/b -/
theorem semicircle_in_right_triangle (r a b : ℝ) 
    (hr : r > 0) (ha : a > 0) (hb : b > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_tangent : ∃ p q : ℝ × ℝ, 
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2*r)^2 ∧
      (p.1 - 0)^2 + (p.2 - 0)^2 = a^2 ∧
      (q.1 - 0)^2 + (q.2 - 0)^2 = b^2) :
  1/r = 1/a + 1/b := by
    sorry

end semicircle_in_right_triangle_l2440_244005


namespace intersection_probability_in_decagon_l2440_244015

/-- A regular decagon is a 10-sided polygon -/
def RegularDecagon : ℕ := 10

/-- The number of diagonals in a regular decagon -/
def NumDiagonals : ℕ := (RegularDecagon.choose 2) - RegularDecagon

/-- The number of ways to choose two diagonals -/
def WaysToChooseTwoDiagonals : ℕ := NumDiagonals.choose 2

/-- The number of convex quadrilaterals that can be formed in a regular decagon -/
def NumConvexQuadrilaterals : ℕ := RegularDecagon.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the decagon -/
def ProbabilityIntersectionInside : ℚ := NumConvexQuadrilaterals / WaysToChooseTwoDiagonals

theorem intersection_probability_in_decagon :
  ProbabilityIntersectionInside = 42 / 119 := by sorry

end intersection_probability_in_decagon_l2440_244015


namespace bees_count_l2440_244045

theorem bees_count (first_day_count : ℕ) (second_day_count : ℕ) : 
  (second_day_count = 3 * first_day_count) → 
  (second_day_count = 432) → 
  (first_day_count = 144) := by
sorry

end bees_count_l2440_244045


namespace sum_square_gt_four_times_adjacent_products_l2440_244061

theorem sum_square_gt_four_times_adjacent_products 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end sum_square_gt_four_times_adjacent_products_l2440_244061


namespace movie_channels_cost_12_l2440_244081

def basic_cable_cost : ℝ := 15
def total_cost : ℝ := 36

def movie_channel_cost : ℝ → Prop :=
  λ m => m > 0 ∧ 
         m + (m - 3) + basic_cable_cost = total_cost

theorem movie_channels_cost_12 : 
  movie_channel_cost 12 := by sorry

end movie_channels_cost_12_l2440_244081


namespace apple_distribution_exists_and_unique_l2440_244048

/-- Represents the last names of the children -/
inductive LastName
| Smith
| Brown
| Jones
| Robinson

/-- Represents a child with their name and number of apples -/
structure Child where
  firstName : String
  lastName : LastName
  apples : Nat

/-- The problem statement -/
theorem apple_distribution_exists_and_unique :
  ∃! (distribution : List Child),
    (distribution.length = 8) ∧
    (distribution.map (λ c => c.apples)).sum = 32 ∧
    (∃ ann ∈ distribution, ann.firstName = "Ann" ∧ ann.apples = 1) ∧
    (∃ mary ∈ distribution, mary.firstName = "Mary" ∧ mary.apples = 2) ∧
    (∃ jane ∈ distribution, jane.firstName = "Jane" ∧ jane.apples = 3) ∧
    (∃ kate ∈ distribution, kate.firstName = "Kate" ∧ kate.apples = 4) ∧
    (∃ ned ∈ distribution, ned.firstName = "Ned" ∧ ned.lastName = LastName.Smith ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Smith ∧ sister.apples = ned.apples) ∧
    (∃ tom ∈ distribution, tom.firstName = "Tom" ∧ tom.lastName = LastName.Brown ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Brown ∧ tom.apples = 2 * sister.apples) ∧
    (∃ bill ∈ distribution, bill.firstName = "Bill" ∧ bill.lastName = LastName.Jones ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Jones ∧ bill.apples = 3 * sister.apples) ∧
    (∃ jack ∈ distribution, jack.firstName = "Jack" ∧ jack.lastName = LastName.Robinson ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Robinson ∧ jack.apples = 4 * sister.apples) :=
by sorry

end apple_distribution_exists_and_unique_l2440_244048


namespace roses_remaining_proof_l2440_244036

def dozen : ℕ := 12

def initial_roses : ℕ := 3 * dozen

def roses_given_away : ℕ := initial_roses / 2

def roses_in_vase : ℕ := initial_roses - roses_given_away

def wilted_roses : ℕ := roses_in_vase / 3

def remaining_roses : ℕ := roses_in_vase - wilted_roses

theorem roses_remaining_proof :
  remaining_roses = 12 := by sorry

end roses_remaining_proof_l2440_244036


namespace chinese_vs_english_spanish_difference_l2440_244074

def hours_english : ℕ := 6
def hours_chinese : ℕ := 7
def hours_spanish : ℕ := 4
def hours_french : ℕ := 5

theorem chinese_vs_english_spanish_difference :
  Int.natAbs (hours_chinese - (hours_english + hours_spanish)) = 3 := by
  sorry

end chinese_vs_english_spanish_difference_l2440_244074


namespace multiplication_division_equality_l2440_244092

theorem multiplication_division_equality : 15 * (1 / 5) * 40 / 4 = 30 := by sorry

end multiplication_division_equality_l2440_244092


namespace range_of_piecewise_function_l2440_244003

/-- Given two linear functions f and g, and a piecewise function r,
    prove that the range of r is [a/2 + b, c + d] -/
theorem range_of_piecewise_function
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (r : ℝ → ℝ)
  (ha : a < 0)
  (hc : c > 0)
  (hf : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (hg : ∀ x, 0 ≤ x ∧ x ≤ 1 → g x = c * x + d)
  (hr : ∀ x, 0 ≤ x ∧ x ≤ 1 → r x = if x ≤ 0.5 then f x else g x) :
  Set.range r = Set.Icc (a / 2 + b) (c + d) :=
sorry

end range_of_piecewise_function_l2440_244003


namespace complex_symmetry_product_l2440_244020

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 2) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -5 := by
sorry

end complex_symmetry_product_l2440_244020


namespace petes_number_l2440_244049

theorem petes_number : ∃ x : ℚ, 3 * (3 * x - 5) = 96 ∧ x = 111 / 9 := by
  sorry

end petes_number_l2440_244049


namespace cubic_equation_solutions_l2440_244033

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x^3 - 3*x^2*(Real.sqrt 3) + 9*x - 3*(Real.sqrt 3)) + (x - Real.sqrt 3)^2 = 0 ↔ 
  x = Real.sqrt 3 ∨ x = -1 + Real.sqrt 3 := by
sorry

end cubic_equation_solutions_l2440_244033


namespace divisor_property_l2440_244009

theorem divisor_property (k : ℕ) : 
  (15 ^ k) ∣ 759325 → 3 ^ k - 0 = 1 := by
  sorry

end divisor_property_l2440_244009


namespace inequality_solution_set_l2440_244052

theorem inequality_solution_set (x : ℝ) : 
  (∃ y, y > 1 ∧ y < x) ↔ (x^2 - x) * (Real.exp x - 1) > 0 := by
sorry

end inequality_solution_set_l2440_244052


namespace perpendicular_implies_perpendicular_lines_l2440_244082

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_implies_perpendicular_lines 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  parallel α β → perpendicularLines l m :=
by sorry

end perpendicular_implies_perpendicular_lines_l2440_244082


namespace geometric_progression_arcsin_least_t_l2440_244062

theorem geometric_progression_arcsin_least_t : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π / 2 → 
    ∃ (r : ℝ), r > 0 ∧
    (Real.arcsin (Real.sin α) = α) ∧
    (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
    (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
    (Real.arcsin (Real.sin (t * α)) = r^3 * α)) ∧
  (∀ (t' : ℝ), t' > 0 → 
    (∀ (α : ℝ), 0 < α → α < π / 2 → 
      ∃ (r : ℝ), r > 0 ∧
      (Real.arcsin (Real.sin α) = α) ∧
      (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
      (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
      (Real.arcsin (Real.sin (t' * α)) = r^3 * α)) →
    t ≤ t') ∧
  t = 16 * Real.sqrt 6 / 3 := by
sorry

end geometric_progression_arcsin_least_t_l2440_244062


namespace sequence_properties_and_sum_l2440_244026

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_properties_and_sum (a b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 1 = 2 →
  (∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, b (n + 1) = r * b n) →
  a 2 + b 3 = 7 →
  a 4 + b 5 = 21 →
  (∀ n : ℕ, c n = a n / b n) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, S n = 6 - (n + 3) / 2^(n - 1)) :=
by sorry

end sequence_properties_and_sum_l2440_244026


namespace line_plane_perpendicular_l2440_244038

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (different : Line → Line → Prop)
variable (non_coincident : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : different m n) 
  (h2 : non_coincident α β) 
  (h3 : perpendicular m α) 
  (h4 : parallel m β) : 
  plane_perpendicular α β :=
sorry

end line_plane_perpendicular_l2440_244038


namespace congruence_from_equal_sides_equal_sides_from_congruence_l2440_244068

/-- Triangle type -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Congruence relation between triangles -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side of a triangle -/
def side_length (t : Triangle) (i : Fin 3) : ℝ := sorry

/-- Two triangles have equal corresponding sides -/
def equal_sides (t1 t2 : Triangle) : Prop :=
  ∀ i : Fin 3, side_length t1 i = side_length t2 i

theorem congruence_from_equal_sides (t1 t2 : Triangle) :
  equal_sides t1 t2 → congruent t1 t2 := by sorry

theorem equal_sides_from_congruence (t1 t2 : Triangle) :
  congruent t1 t2 → equal_sides t1 t2 := by sorry

end congruence_from_equal_sides_equal_sides_from_congruence_l2440_244068


namespace rainy_days_pigeonhole_l2440_244089

theorem rainy_days_pigeonhole (n : ℕ) (m : ℕ) (h : n > 2 * m) :
  ∃ (x : ℕ), x ≤ m ∧ (∃ (S : Finset ℕ), S.card ≥ 3 ∧ ∀ i ∈ S, i < n ∧ x = i % (m + 1)) :=
by
  sorry

#check rainy_days_pigeonhole 64 30

end rainy_days_pigeonhole_l2440_244089


namespace special_hexagon_perimeter_l2440_244031

/-- A hexagon that shares three sides with a rectangle and has the other three sides
    each equal to one of the rectangle's dimensions. -/
structure SpecialHexagon where
  rect_side1 : ℕ
  rect_side2 : ℕ

/-- The perimeter of the special hexagon. -/
def perimeter (h : SpecialHexagon) : ℕ :=
  2 * h.rect_side1 + 2 * h.rect_side2 + h.rect_side1 + h.rect_side2

/-- Theorem stating that the perimeter of the special hexagon with sides 7 and 5 is 36. -/
theorem special_hexagon_perimeter :
  ∃ (h : SpecialHexagon), h.rect_side1 = 7 ∧ h.rect_side2 = 5 ∧ perimeter h = 36 := by
  sorry

end special_hexagon_perimeter_l2440_244031


namespace next_skipped_perfect_square_l2440_244034

theorem next_skipped_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m^2 = n) ∧
  (∀ y : ℕ, y > x ∧ y < n → ¬∃ m : ℕ, m^2 = y) ∧
  (∃ m : ℕ, m^2 = x + 4 * Real.sqrt x + 4) :=
sorry

end next_skipped_perfect_square_l2440_244034


namespace solutions_are_correct_l2440_244059

def solutions : Set ℂ := {
  (16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  (16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  Complex.I * 2^(2/3),
  -Complex.I * 2^(2/3)
}

theorem solutions_are_correct : {z : ℂ | z^6 = -16} = solutions := by
  sorry

end solutions_are_correct_l2440_244059


namespace original_triangle_area_l2440_244070

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (4 * side)^2 * (original_area / side^2)) →
  new_area = 64 →
  original_area = 4 := by
sorry

end original_triangle_area_l2440_244070


namespace valid_arrangement_iff_even_l2440_244042

/-- A valid grid arrangement for the problem -/
def ValidArrangement (n : ℕ) (grid : Fin n → Fin n → ℕ) : Prop :=
  (∀ i j, grid i j ∈ Finset.range (n^2)) ∧
  (∀ k : Fin (n^2 - 1), ∃ i j i' j', 
    grid i j = k ∧ grid i' j' = k + 1 ∧ 
    ((i = i' ∧ j.val + 1 = j'.val) ∨ 
     (j = j' ∧ i.val + 1 = i'.val))) ∧
  (∀ i j i' j', grid i j % n = grid i' j' % n → 
    (i ≠ i' ∧ j ≠ j'))

/-- The main theorem stating that a valid arrangement exists if and only if n is even -/
theorem valid_arrangement_iff_even (n : ℕ) (h : n > 1) :
  (∃ grid, ValidArrangement n grid) ↔ Even n :=
sorry

end valid_arrangement_iff_even_l2440_244042


namespace pepper_remaining_l2440_244011

theorem pepper_remaining (initial_amount used_amount : ℝ) 
  (h1 : initial_amount = 0.25)
  (h2 : used_amount = 0.16) : 
  initial_amount - used_amount = 0.09 := by
  sorry

end pepper_remaining_l2440_244011


namespace problem_solution_l2440_244001

theorem problem_solution :
  (∀ (a : ℝ), a ≠ 0 → (a^2)^3 / (-a)^2 = a^4) ∧
  (∀ (a b : ℝ), (a+2*b)*(a+b) - 3*a*(a+b) = -2*a^2 + 2*b^2) :=
by sorry

end problem_solution_l2440_244001


namespace union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l2440_244097

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∪ B = {x | 2 < x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem 2: (ℝ \ A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 3
theorem intersection_A_C_nonempty_implies_a_gt_3 (a : ℝ) : (A ∩ C a).Nonempty → a > 3 := by sorry

end union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l2440_244097


namespace arctan_sum_equals_pi_over_four_l2440_244019

theorem arctan_sum_equals_pi_over_four :
  ∃ n : ℕ+, 
    Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4 ∧
    n = 56 := by
  sorry

end arctan_sum_equals_pi_over_four_l2440_244019


namespace pizza_combinations_l2440_244053

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 3

/-- The maximum number of unique pizzas that can be made -/
def max_pizzas : ℕ := Nat.choose n k

theorem pizza_combinations :
  max_pizzas = 56 := by sorry

end pizza_combinations_l2440_244053


namespace gcf_of_60_180_150_l2440_244035

theorem gcf_of_60_180_150 : Nat.gcd 60 (Nat.gcd 180 150) = 30 := by
  sorry

end gcf_of_60_180_150_l2440_244035


namespace trajectory_and_line_m_l2440_244008

/-- The distance ratio condition for point P -/
def distance_ratio (x y : ℝ) : Prop :=
  (((x - 3 * Real.sqrt 3)^2 + y^2).sqrt) / (|x - 4 * Real.sqrt 3|) = Real.sqrt 3 / 2

/-- The equation of the ellipse -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

/-- The equation of line m -/
def on_line_m (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

/-- The midpoint condition -/
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2

theorem trajectory_and_line_m :
  (∀ x y : ℝ, distance_ratio x y ↔ on_ellipse x y) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse x₁ y₁ ∧ on_ellipse x₂ y₂ ∧ is_midpoint x₁ y₁ x₂ y₂ →
    on_line_m x₁ y₁ ∧ on_line_m x₂ y₂) :=
sorry

end trajectory_and_line_m_l2440_244008


namespace isosceles_triangle_perimeter_l2440_244094

-- Define an isosceles triangle with sides 4, 8, and 8
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = 4 ∧ b = 8 ∧ c = 8 ∧ b = c

-- Define the perimeter of a triangle
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isosceles_triangle a b c → triangle_perimeter a b c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l2440_244094


namespace floor_composition_identity_l2440_244016

open Real

theorem floor_composition_identity (α : ℝ) (n : ℕ) (h : α > 1) :
  let β := 1 / α
  let fₐ (x : ℝ) := ⌊α * x + 1/2⌋
  let fᵦ (x : ℝ) := ⌊β * x + 1/2⌋
  fᵦ (fₐ n) = n := by
  sorry

end floor_composition_identity_l2440_244016


namespace max_x_minus_y_l2440_244067

theorem max_x_minus_y (x y z : ℝ) (sum_eq : x + y + z = 2) (prod_eq : x*y + y*z + z*x = 1) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧ ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → |a - b| ≤ max :=
by sorry

end max_x_minus_y_l2440_244067


namespace total_value_is_18_60_l2440_244051

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a half-dollar coin in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dollar coin in dollars -/
def dollar_coin_value : ℚ := 1.00

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 25

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 15

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 12

/-- The number of half-dollar coins Tom found -/
def num_half_dollars : ℕ := 7

/-- The number of dollar coins Tom found -/
def num_dollar_coins : ℕ := 3

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 375

/-- The total value of the coins Tom found -/
def total_value : ℚ :=
  num_quarters * quarter_value +
  num_dimes * dime_value +
  num_nickels * nickel_value +
  num_half_dollars * half_dollar_value +
  num_dollar_coins * dollar_coin_value +
  num_pennies * penny_value

theorem total_value_is_18_60 : total_value = 18.60 := by
  sorry

end total_value_is_18_60_l2440_244051


namespace jelly_bean_ratio_l2440_244083

theorem jelly_bean_ratio : 
  ∀ (total_jelly_beans red_jelly_beans coconut_flavored_red_jelly_beans : ℕ),
    total_jelly_beans = 4000 →
    coconut_flavored_red_jelly_beans = 750 →
    4 * coconut_flavored_red_jelly_beans = red_jelly_beans →
    3 * total_jelly_beans = 4 * red_jelly_beans :=
by
  sorry

end jelly_bean_ratio_l2440_244083


namespace statement_d_is_incorrect_l2440_244063

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- State the theorem
theorem statement_d_is_incorrect
  (α β : Plane) (l m n : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (h_perp_planes : perp α β)
  (h_perp_m_α : perpLine m α)
  (h_perp_n_β : perpLine n β) :
  ¬ (∀ m n, perpLines m n) :=
sorry

end statement_d_is_incorrect_l2440_244063


namespace no_three_common_tangents_l2440_244073

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles -/
def commonTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem: It's impossible for two circles in the same plane to have exactly 3 common tangents -/
theorem no_three_common_tangents (c1 c2 : Circle) : 
  commonTangents c1 c2 ≠ 3 := by
  sorry

end no_three_common_tangents_l2440_244073


namespace shaded_area_is_7pi_l2440_244047

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles in the problem -/
structure CircleConfiguration where
  smallCircles : List Circle
  largeCircle : Circle
  allIntersectAtTangency : Bool

/-- Calculates the area of the shaded region given a circle configuration -/
def shadedArea (config : CircleConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the shaded area for the given configuration -/
theorem shaded_area_is_7pi (config : CircleConfiguration)
  (h1 : config.smallCircles.length = 13)
  (h2 : ∀ c ∈ config.smallCircles, c.radius = 1)
  (h3 : config.allIntersectAtTangency = true) :
  shadedArea config = 7 * Real.pi :=
  sorry

end shaded_area_is_7pi_l2440_244047


namespace geometric_sequence_ratio_l2440_244014

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a4 : a 4 = 1) 
  (h_a7 : a 7 = 8) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_ratio_l2440_244014


namespace divisibility_property_l2440_244010

theorem divisibility_property (n p q : ℕ) : 
  n > 0 → 
  Prime p → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end divisibility_property_l2440_244010


namespace chessboard_cover_l2440_244032

def coverWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => coverWays (k + 2) + coverWays (k + 1)

theorem chessboard_cover : coverWays 9 = 55 := by
  sorry

end chessboard_cover_l2440_244032


namespace cistern_emptying_l2440_244090

/-- Represents the fraction of a cistern emptied in a given time -/
def fractionEmptied (time : ℚ) : ℚ :=
  if time = 8 then 1/3
  else if time = 16 then 2/3
  else 0

theorem cistern_emptying (t : ℚ) :
  fractionEmptied 8 = 1/3 →
  fractionEmptied 16 = 2 * fractionEmptied 8 :=
by sorry

end cistern_emptying_l2440_244090


namespace schur_like_inequality_l2440_244037

theorem schur_like_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) ≥ a + b + c :=
sorry

end schur_like_inequality_l2440_244037


namespace eight_stairs_climb_ways_l2440_244024

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => climbStairs (m + 2) + climbStairs (m + 1)

-- Theorem stating that there are 4 ways to climb 8 stairs
theorem eight_stairs_climb_ways : climbStairs 8 = 4 := by
  sorry

end eight_stairs_climb_ways_l2440_244024


namespace p_has_four_digits_l2440_244098

-- Define p as given in the problem
def p : ℚ := 125 * 243 * 16 / 405

-- Function to count the number of digits in a rational number
def count_digits (q : ℚ) : ℕ := sorry

-- Theorem stating that p has 4 digits
theorem p_has_four_digits : count_digits p = 4 := by sorry

end p_has_four_digits_l2440_244098


namespace weight_of_replaced_person_l2440_244088

theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the group
  (w_new : ℝ) -- weight of the new person
  (w_avg_increase : ℝ) -- increase in average weight
  (h1 : n = 12) -- there are 12 people initially
  (h2 : w_new = 106) -- the new person weighs 106 kg
  (h3 : w_avg_increase = 4) -- average weight increases by 4 kg
  : ∃ w_old : ℝ, w_old = 58 ∧ n * w_avg_increase = w_new - w_old :=
by sorry

end weight_of_replaced_person_l2440_244088


namespace quadratic_point_m_l2440_244069

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_m (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end quadratic_point_m_l2440_244069


namespace second_half_speed_l2440_244087

def total_distance : ℝ := 336
def total_time : ℝ := 15
def first_half_speed : ℝ := 21

theorem second_half_speed : ℝ := by
  have h1 : total_distance / 2 = first_half_speed * (total_time / 2) := by sorry
  have h2 : total_distance / 2 = 24 * (total_time - total_time / 2) := by sorry
  exact 24

end second_half_speed_l2440_244087


namespace line_intersection_range_l2440_244085

/-- The line y = e^x + b has at most one common point with both f(x) = e^x and g(x) = ln(x) 
    if and only if b is in the closed interval [-2, 0] -/
theorem line_intersection_range (b : ℝ) : 
  (∀ x : ℝ, (∃! y : ℝ, y = Real.exp x + b ∧ (y = Real.exp x ∨ y = Real.log x)) ∨
            (∀ y : ℝ, y ≠ Real.exp x + b ∨ (y ≠ Real.exp x ∧ y ≠ Real.log x))) ↔ 
  b ∈ Set.Icc (-2) 0 := by
  sorry

end line_intersection_range_l2440_244085


namespace part_one_part_two_part_three_l2440_244043

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the universal set U
def U : Set ℝ := A ∪ B 3

-- Part 1
theorem part_one : A ∩ (U \ B 3) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : A ∩ B m = ∅ → m ≤ -2 := by sorry

-- Part 3
theorem part_three : A ∩ B m = A → m ≥ 4 := by sorry

end part_one_part_two_part_three_l2440_244043


namespace max_value_theorem_l2440_244056

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  ∃ (z : ℝ), z = x^2 + 2*x*y + 3*y^2 ∧ z ≤ 132 + 48 * Real.sqrt 3 ∧
  ∃ (a b : ℝ), a^2 - 2*a*b + 3*b^2 = 12 ∧ a > 0 ∧ b > 0 ∧
  a^2 + 2*a*b + 3*b^2 = 132 + 48 * Real.sqrt 3 :=
sorry

end max_value_theorem_l2440_244056


namespace line_passes_through_center_l2440_244095

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3*x + 2*y = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (2, -3)

/-- Theorem stating that the line passes through the center of the circle -/
theorem line_passes_through_center : 
  line_equation center.1 center.2 ∧ circle_equation center.1 center.2 := by
  sorry

end line_passes_through_center_l2440_244095


namespace eliminate_denominators_l2440_244027

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  1 + 2 / (x - 1) = (x - 5) / (x - 3)

-- Define the result after eliminating denominators
def eliminated_denominators (x : ℝ) : Prop :=
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1)

-- Theorem stating that eliminating denominators in the original equation
-- results in the specified equation
theorem eliminate_denominators (x : ℝ) :
  original_equation x → eliminated_denominators x :=
by
  sorry

end eliminate_denominators_l2440_244027


namespace solution_implies_q_value_l2440_244084

theorem solution_implies_q_value (q : ℚ) (h : 2 * q - 3 = 11) : q = 7 := by
  sorry

end solution_implies_q_value_l2440_244084


namespace savings_calculation_l2440_244000

/-- The amount saved per month in dollars -/
def monthly_savings : ℕ := 3000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total amount saved after one year -/
def total_savings : ℕ := monthly_savings * months_in_year

theorem savings_calculation : total_savings = 36000 := by
  sorry

end savings_calculation_l2440_244000


namespace tripled_base_and_exponent_l2440_244018

theorem tripled_base_and_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ (2 * b) → x = 3 * Real.sqrt 3 * a := by
  sorry

end tripled_base_and_exponent_l2440_244018


namespace expression_equals_one_l2440_244041

theorem expression_equals_one :
  |Real.sqrt 3 - 2| + (-1/2)⁻¹ + (2023 - Real.pi)^0 + 3 * Real.tan (30 * π / 180) = 1 := by
  sorry

end expression_equals_one_l2440_244041


namespace cylinder_lateral_surface_area_l2440_244096

/-- Given a cylinder with base area S and lateral surface that unfolds into a square,
    prove that its lateral surface area is 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  circumference * height = 4 * Real.pi * S := by
  sorry

end cylinder_lateral_surface_area_l2440_244096


namespace gcd_4034_10085_base5_l2440_244064

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem gcd_4034_10085_base5 :
  let g := Nat.gcd 4034 10085
  isValidBase5 (toBase5 g) ∧ toBase5 g = [2, 3, 0, 1, 3] := by
  sorry

end gcd_4034_10085_base5_l2440_244064


namespace solution_satisfies_conditions_l2440_244007

noncomputable def y (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) + (1 - k*x)^(1/k))
  else
    Real.cosh x

noncomputable def z (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) - (1 - k*x)^(1/k))
  else
    Real.sinh x

theorem solution_satisfies_conditions (k : ℝ) :
  (∀ x, (deriv (y k)) x = (z k x) * ((y k x) + (z k x))^k) ∧
  (∀ x, (deriv (z k)) x = (y k x) * ((y k x) + (z k x))^k) ∧
  y k 0 = 1 ∧
  z k 0 = 0 := by
  sorry

end solution_satisfies_conditions_l2440_244007


namespace abs_value_of_z_l2440_244093

theorem abs_value_of_z (z : ℂ) (h : z = Complex.I * (1 - Complex.I)) : Complex.abs z = Real.sqrt 2 := by
  sorry

end abs_value_of_z_l2440_244093


namespace cole_total_students_l2440_244023

/-- The number of students in Ms. Cole's math classes -/
structure ColeMathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- The conditions for Ms. Cole's math classes -/
def cole_math_class_conditions (c : ColeMathClasses) : Prop :=
  c.sixth_level = 40 ∧
  c.fourth_level = 4 * c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level

/-- The theorem stating the total number of students Ms. Cole teaches -/
theorem cole_total_students (c : ColeMathClasses) 
  (h : cole_math_class_conditions c) : 
  c.sixth_level + c.fourth_level + c.seventh_level = 520 := by
  sorry


end cole_total_students_l2440_244023


namespace no_positive_integer_sequence_exists_positive_irrational_sequence_l2440_244072

-- Part 1: Non-existence of positive integer sequence
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) :=
sorry

-- Part 2: Existence of positive irrational number sequence
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end no_positive_integer_sequence_exists_positive_irrational_sequence_l2440_244072


namespace circle_point_inequality_l2440_244066

theorem circle_point_inequality (m n c : ℝ) : 
  (∀ m n, m^2 + (n - 2)^2 = 1 → m + n + c ≥ 1) → c ≥ Real.sqrt 2 - 1 := by
  sorry

end circle_point_inequality_l2440_244066


namespace triangle_abc_proof_l2440_244054

open Real

theorem triangle_abc_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  -- Given conditions
  (0 < A) → (A < 2 * π / 3) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (m = (a / 2, c / 2)) →
  (n = (cos C, cos A)) →
  (n.1 * m.1 + n.2 * m.2 = b * cos B) →
  (cos ((A - C) / 2) = sqrt 3 * sin A) →
  (m.1 * m.1 + m.2 * m.2 = 5) →
  -- Conclusions
  (B = π / 3) ∧
  (1 / 2 * a * b * sin C = 2 * sqrt 3) :=
by sorry

end triangle_abc_proof_l2440_244054


namespace f_lipschitz_implies_m_bounded_l2440_244050

theorem f_lipschitz_implies_m_bounded (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ [-2, 2] → x₂ ∈ [-2, 2] →
    |((fun x => Real.exp (m * x) + x^4 - m * x) x₁) -
     ((fun x => Real.exp (m * x) + x^4 - m * x) x₂)| ≤ Real.exp 4 + 11) →
  m ∈ [-2, 2] := by
sorry

end f_lipschitz_implies_m_bounded_l2440_244050


namespace sin_210_degrees_l2440_244071

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l2440_244071


namespace inequality_proof_l2440_244091

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (a + b) / 2 := by
  sorry

end inequality_proof_l2440_244091


namespace volleyball_team_score_l2440_244006

/-- Volleyball team scoring problem -/
theorem volleyball_team_score (lizzie_score : ℕ) (team_total : ℕ) : 
  lizzie_score = 4 →
  team_total = 50 →
  17 = team_total - (lizzie_score + (lizzie_score + 3) + 2 * (lizzie_score + (lizzie_score + 3))) :=
by
  sorry

end volleyball_team_score_l2440_244006


namespace integral_3x_plus_sin_x_l2440_244044

theorem integral_3x_plus_sin_x (x : Real) :
  ∫ x in (0)..(π/2), (3 * x + Real.sin x) = (3/8) * π^2 + 1 := by
  sorry

end integral_3x_plus_sin_x_l2440_244044


namespace invitation_ways_l2440_244022

def number_of_teachers : ℕ := 10
def teachers_to_invite : ℕ := 6

def ways_to_invite (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem invitation_ways : 
  ways_to_invite number_of_teachers teachers_to_invite - 
  ways_to_invite (number_of_teachers - 2) (teachers_to_invite - 2) = 140 :=
by
  sorry

end invitation_ways_l2440_244022


namespace smallest_solution_is_negative_85_l2440_244028

def floor_equation (x : ℤ) : Prop :=
  Int.floor (x / 2) + Int.floor (x / 3) + Int.floor (x / 7) = x

theorem smallest_solution_is_negative_85 :
  (∀ y < -85, ¬ floor_equation y) ∧ floor_equation (-85) := by
  sorry

end smallest_solution_is_negative_85_l2440_244028


namespace ellipse_foci_distance_l2440_244012

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x^2 + 4y^2 = 16 -/
def Ellipse (P : Point) : Prop :=
  P.x^2 + 4 * P.y^2 = 16

/-- Represents the distance between two points -/
def distance (P Q : Point) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2)^(1/2)

/-- Theorem: For a point P on the ellipse x^2 + 4y^2 = 16 with foci F1 and F2,
    if the distance from P to F1 is 7, then the distance from P to F2 is 1 -/
theorem ellipse_foci_distance (P F1 F2 : Point) :
  Ellipse P →
  distance P F1 = 7 →
  distance P F2 = 1 :=
sorry

end ellipse_foci_distance_l2440_244012


namespace complement_A_B_l2440_244029

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_A_B : (A \ B) = {0, 2, 6, 10} := by sorry

end complement_A_B_l2440_244029


namespace final_price_after_discounts_l2440_244021

def original_price : ℝ := 250
def first_discount : ℝ := 0.60
def second_discount : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount) * (1 - second_discount)) = 75 := by
sorry

end final_price_after_discounts_l2440_244021


namespace area_of_region_l2440_244055

-- Define the lower bound function
def lower_bound (x : ℝ) : ℝ := |x - 4|

-- Define the upper bound function
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

-- Theorem statement
theorem area_of_region : MeasureTheory.volume region = 10 := by
  sorry

end area_of_region_l2440_244055


namespace smallest_valid_number_l2440_244075

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ),
    n = 5 * 10^k + m ∧
    m * 10 + 5 = (5 * 10^k + m) / 4

theorem smallest_valid_number :
  ∃ (n : ℕ),
    is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → n ≤ m ∧
    n = 512820
  := by sorry

end smallest_valid_number_l2440_244075


namespace cousin_future_age_l2440_244077

/-- Given the ages of Nick and his relatives, prove the cousin's future age. -/
theorem cousin_future_age (nick_age : ℕ) (sister_age_diff : ℕ) (cousin_age_diff : ℕ) :
  nick_age = 13 →
  sister_age_diff = 6 →
  cousin_age_diff = 3 →
  let sister_age := nick_age + sister_age_diff
  let brother_age := (nick_age + sister_age) / 2
  let cousin_age := brother_age - cousin_age_diff
  cousin_age + (2 * brother_age - cousin_age) = 32 := by
  sorry

end cousin_future_age_l2440_244077


namespace intersection_of_A_and_B_l2440_244040

def A : Set ℝ := {x : ℝ | -x^2 + x + 6 > 0}
def B : Set ℝ := {x : ℝ | x^2 + 2*x - 8 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l2440_244040


namespace article_cost_price_l2440_244079

theorem article_cost_price (profit_percent : ℝ) (discount_percent : ℝ) (price_reduction : ℝ) (new_profit_percent : ℝ) :
  profit_percent = 25 →
  discount_percent = 20 →
  price_reduction = 8.40 →
  new_profit_percent = 30 →
  ∃ (cost : ℝ), 
    cost > 0 ∧
    (cost + profit_percent / 100 * cost) - price_reduction = 
    (cost * (1 - discount_percent / 100)) * (1 + new_profit_percent / 100) ∧
    cost = 40 := by
  sorry

end article_cost_price_l2440_244079


namespace functional_equation_solution_l2440_244002

theorem functional_equation_solution (f : ℕ → ℝ) 
  (h : ∀ x y : ℕ, f (x + y) + f (x - y) = f (3 * x)) : 
  ∀ x : ℕ, f x = 0 := by sorry

end functional_equation_solution_l2440_244002
