import Mathlib

namespace NUMINAMATH_CALUDE_count_parallelograms_l460_46049

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram with vertices P, Q, R, S -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram using the shoelace formula -/
def area (p : Parallelogram) : ℚ :=
  (1 / 2 : ℚ) * |p.Q.x * p.S.y - p.S.x * p.Q.y|

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is on the line y = mx -/
def isOnLine (p : Point) (m : ℤ) : Prop :=
  p.y = m * p.x

/-- The main theorem to be proved -/
theorem count_parallelograms :
  let validParallelogram (p : Parallelogram) : Prop :=
    p.P = ⟨0, 0⟩ ∧
    isFirstQuadrant p.Q ∧
    isFirstQuadrant p.R ∧
    isFirstQuadrant p.S ∧
    isOnLine p.Q 2 ∧
    isOnLine p.S 3 ∧
    area p = 2000000
  (parallelograms : Finset Parallelogram) →
  (∀ p ∈ parallelograms, validParallelogram p) →
  parallelograms.card = 196 :=
sorry

end NUMINAMATH_CALUDE_count_parallelograms_l460_46049


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l460_46035

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_statements
    (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
    (¬ (q > 1 → IncreasingSequence a)) ∧
    (¬ (IncreasingSequence a → q > 1)) ∧
    (¬ (q ≤ 1 → ¬IncreasingSequence a)) ∧
    (¬ (¬IncreasingSequence a → q ≤ 1)) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l460_46035


namespace NUMINAMATH_CALUDE_exponential_dominance_l460_46050

theorem exponential_dominance (k : ℝ) (hk : k > 0) :
  ∃ x₀ : ℝ, ∀ x ≥ x₀, (2 : ℝ) ^ ((2 : ℝ) ^ x) > ((2 : ℝ) ^ x) ^ k :=
sorry

end NUMINAMATH_CALUDE_exponential_dominance_l460_46050


namespace NUMINAMATH_CALUDE_point_q_coordinates_l460_46076

/-- Given two points P and Q in a 2D Cartesian coordinate system,
    prove that Q has coordinates (1, -3) under the given conditions. -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D space
  (h1 : P = (1, 2))  -- P has coordinates (1, 2)
  (h2 : (Q.2 : ℝ) < 0)  -- Q is below the x-axis
  (h3 : P.1 = Q.1)  -- PQ is parallel to the y-axis
  (h4 : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5)  -- PQ = 5
  : Q = (1, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l460_46076


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l460_46055

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin (-1, a) (b, 2) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l460_46055


namespace NUMINAMATH_CALUDE_birthday_money_ratio_l460_46088

theorem birthday_money_ratio : 
  ∀ (total_money video_game_cost goggles_cost money_left : ℚ),
    total_money = 100 →
    video_game_cost = total_money / 4 →
    money_left = 60 →
    goggles_cost = total_money - video_game_cost - money_left →
    goggles_cost / (total_money - video_game_cost) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_ratio_l460_46088


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l460_46034

theorem fraction_equality_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 56 / Q → P + Q = 126 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l460_46034


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l460_46013

theorem quadratic_roots_problem (a b p q : ℝ) : 
  p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧
  p^2 - a*p + b = 0 ∧
  q^2 - a*q + b = 0 ∧
  a^2 - p*a - q = 0 ∧
  b^2 - p*b - q = 0 →
  a = 1 ∧ b = -2 ∧ p = -1 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l460_46013


namespace NUMINAMATH_CALUDE_total_blood_cells_l460_46041

/-- The total number of blood cells in two samples is 7,341, given that the first sample contains 4,221 blood cells and the second sample contains 3,120 blood cells. -/
theorem total_blood_cells (sample1 : Nat) (sample2 : Nat)
  (h1 : sample1 = 4221)
  (h2 : sample2 = 3120) :
  sample1 + sample2 = 7341 := by
  sorry

end NUMINAMATH_CALUDE_total_blood_cells_l460_46041


namespace NUMINAMATH_CALUDE_proportional_increase_l460_46054

/-- Given the equation 3x - 2y = 7, this theorem proves that y increases proportionally to x
    and determines the proportionality coefficient. -/
theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) :
  ∃ (k b : ℝ), y = k * x + b ∧ k = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_increase_l460_46054


namespace NUMINAMATH_CALUDE_sqrt_588_simplification_l460_46090

theorem sqrt_588_simplification : Real.sqrt 588 = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_588_simplification_l460_46090


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l460_46089

theorem polynomial_division_theorem (x : ℚ) :
  (3 * x + 1) * (2 * x^3 + x^2 - 7/3 * x + 20/9) + 31/27 = 
  6 * x^4 + 5 * x^3 - 4 * x^2 + x + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l460_46089


namespace NUMINAMATH_CALUDE_reciprocal_sum_l460_46015

theorem reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / (x * y) = 5) (h3 : (x + y) / 5 = 0.6) :
  1 / x + 1 / y = 15 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l460_46015


namespace NUMINAMATH_CALUDE_like_terms_exponents_l460_46005

theorem like_terms_exponents (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), -4 * a^(x-y) * b^4 = k * a^2 * b^(x+y)) → 
  (x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l460_46005


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_is_two_81ths_l460_46098

/-- A regular tetrahedron with stripes painted on its faces -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : ℕ
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : ℕ
  /-- The total number of possible stripe combinations -/
  total_combinations : ℕ
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : ℕ
  /-- Constraint: num_faces is 4 for a tetrahedron -/
  face_constraint : num_faces = 4
  /-- Constraint: orientations_per_face is 3 -/
  orientation_constraint : orientations_per_face = 3
  /-- Constraint: total_combinations is orientations_per_face^num_faces -/
  combination_constraint : total_combinations = orientations_per_face ^ num_faces
  /-- Constraint: favorable_outcomes is 2 -/
  outcome_constraint : favorable_outcomes = 2

/-- The probability of having a continuous stripe connecting all vertices -/
def continuous_stripe_probability (t : StripedTetrahedron) : ℚ :=
  t.favorable_outcomes / t.total_combinations

/-- Theorem: The probability of a continuous stripe is 2/81 -/
theorem continuous_stripe_probability_is_two_81ths (t : StripedTetrahedron) :
  continuous_stripe_probability t = 2 / 81 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_is_two_81ths_l460_46098


namespace NUMINAMATH_CALUDE_area_relationship_l460_46019

/-- Two congruent isosceles right-angled triangles with inscribed squares -/
structure TriangleWithSquare where
  /-- The side length of the triangle -/
  side : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The inscribed square's side is less than the triangle's side -/
  h_square_fits : square_side < side

/-- The theorem stating the relationship between the areas of squares P and R -/
theorem area_relationship (t : TriangleWithSquare) (h_area_p : t.square_side ^ 2 = 45) :
  ∃ (r : ℝ), r ^ 2 = 40 ∧ ∃ (t' : TriangleWithSquare), t'.square_side ^ 2 = r ^ 2 :=
sorry

end NUMINAMATH_CALUDE_area_relationship_l460_46019


namespace NUMINAMATH_CALUDE_value_of_a_l460_46063

theorem value_of_a (U : Set ℝ) (A : Set ℝ) (a : ℝ) : 
  U = {2, 3, a^2 - a - 1} →
  A = {2, 3} →
  U \ A = {1} →
  a = -1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l460_46063


namespace NUMINAMATH_CALUDE_double_beavers_half_time_beavers_build_dam_l460_46051

/-- Represents the time (in hours) it takes a given number of beavers to build a dam -/
def build_time (num_beavers : ℕ) : ℝ := 
  if num_beavers = 18 then 8 else 
  if num_beavers = 36 then 4 else 0

/-- The proposition that doubling the number of beavers halves the build time -/
theorem double_beavers_half_time : 
  build_time 36 = (build_time 18) / 2 := by
sorry

/-- The main theorem stating that 36 beavers can build the dam in 4 hours -/
theorem beavers_build_dam : 
  build_time 36 = 4 := by
sorry

end NUMINAMATH_CALUDE_double_beavers_half_time_beavers_build_dam_l460_46051


namespace NUMINAMATH_CALUDE_bacteria_count_theorem_l460_46030

/-- The number of bacteria after growth, given the original count and increase. -/
def bacteria_after_growth (original : ℕ) (increase : ℕ) : ℕ :=
  original + increase

/-- Theorem stating that the number of bacteria after growth is 8917,
    given the original count of 600 and an increase of 8317. -/
theorem bacteria_count_theorem :
  bacteria_after_growth 600 8317 = 8917 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_theorem_l460_46030


namespace NUMINAMATH_CALUDE_ace_in_top_probability_l460_46033

/-- A standard deck of cards --/
def standard_deck : ℕ := 52

/-- The number of top cards we're considering --/
def top_cards : ℕ := 3

/-- The probability of the Ace of Spades being among the top cards --/
def prob_ace_in_top : ℚ := 3 / 52

theorem ace_in_top_probability :
  prob_ace_in_top = top_cards / standard_deck :=
by sorry

end NUMINAMATH_CALUDE_ace_in_top_probability_l460_46033


namespace NUMINAMATH_CALUDE_inscribed_circle_areas_l460_46028

-- Define the square with its diagonal
def square_diagonal : ℝ := 40

-- Define the theorem
theorem inscribed_circle_areas :
  let square_side := square_diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_radius := square_side / 2
  let circle_area := π * circle_radius ^ 2
  square_area = 800 ∧ circle_area = 200 * π := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_areas_l460_46028


namespace NUMINAMATH_CALUDE_triangle_expression_value_l460_46039

theorem triangle_expression_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_expression_value_l460_46039


namespace NUMINAMATH_CALUDE_sum_of_fractions_l460_46010

theorem sum_of_fractions : 
  (1 : ℚ) / 2 + 1 / 6 + 1 / 12 + 1 / 20 + 1 / 30 + 1 / 42 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l460_46010


namespace NUMINAMATH_CALUDE_polynomial_equality_l460_46075

theorem polynomial_equality (m n : ℝ) : 
  (∀ x : ℝ, (x + 1) * (2 * x - 3) = 2 * x^2 + m * x + n) → 
  m = -1 ∧ n = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l460_46075


namespace NUMINAMATH_CALUDE_simplify_fraction_l460_46021

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l460_46021


namespace NUMINAMATH_CALUDE_cubic_function_property_l460_46052

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem: If f(a) = a^3 and f(b) = b^3, then c = 16 -/
theorem cubic_function_property (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : f a b c a = a^3) (h2 : f a b c b = b^3) : c = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l460_46052


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l460_46077

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 4 * y + 2

-- State the theorem
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l460_46077


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l460_46032

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  (f 0 = 0) ∧ (f 2 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l460_46032


namespace NUMINAMATH_CALUDE_sin_405_degrees_l460_46097

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l460_46097


namespace NUMINAMATH_CALUDE_clothes_expenditure_fraction_l460_46084

def salary : ℝ := 190000

theorem clothes_expenditure_fraction 
  (food_fraction : ℝ) 
  (rent_fraction : ℝ) 
  (remaining : ℝ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : remaining = 19000)
  (h4 : ∃ (clothes_fraction : ℝ), 
    salary * (1 - food_fraction - rent_fraction - clothes_fraction) = remaining) :
  ∃ (clothes_fraction : ℝ), clothes_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_clothes_expenditure_fraction_l460_46084


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l460_46056

theorem quadratic_inequality_range (m : ℝ) :
  (¬∃ x : ℝ, x^2 - 2*x + m ≤ 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l460_46056


namespace NUMINAMATH_CALUDE_dvd_cost_is_six_l460_46061

/-- Represents the DVD production and sales scenario --/
structure DVDProduction where
  movieCost : ℕ
  dailySales : ℕ
  daysPerWeek : ℕ
  weeks : ℕ
  profit : ℕ
  sellingPriceFactor : ℚ

/-- Calculates the production cost of a single DVD --/
def calculateDVDCost (p : DVDProduction) : ℚ :=
  let totalSales := p.dailySales * p.daysPerWeek * p.weeks
  let revenue := p.profit + p.movieCost
  let costPerDVD := revenue / (totalSales * (p.sellingPriceFactor - 1))
  costPerDVD

/-- Theorem stating that the DVD production cost is $6 --/
theorem dvd_cost_is_six (p : DVDProduction) 
  (h1 : p.movieCost = 2000)
  (h2 : p.dailySales = 500)
  (h3 : p.daysPerWeek = 5)
  (h4 : p.weeks = 20)
  (h5 : p.profit = 448000)
  (h6 : p.sellingPriceFactor = 5/2) :
  calculateDVDCost p = 6 := by
  sorry

#eval calculateDVDCost {
  movieCost := 2000,
  dailySales := 500,
  daysPerWeek := 5,
  weeks := 20,
  profit := 448000,
  sellingPriceFactor := 5/2
}

end NUMINAMATH_CALUDE_dvd_cost_is_six_l460_46061


namespace NUMINAMATH_CALUDE_birthday_spending_l460_46044

theorem birthday_spending (initial_amount remaining_amount : ℕ) : 
  initial_amount = 7 → remaining_amount = 5 → initial_amount - remaining_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_birthday_spending_l460_46044


namespace NUMINAMATH_CALUDE_power_zero_equivalence_l460_46047

theorem power_zero_equivalence (x : ℝ) (h : x ≠ 0) : x^0 = 1/(x^0) := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equivalence_l460_46047


namespace NUMINAMATH_CALUDE_max_sum_of_proportional_integers_l460_46042

theorem max_sum_of_proportional_integers (x y z : ℤ) : 
  (x : ℚ) / 5 = 6 / (y : ℚ) → 
  (x : ℚ) / 5 = (z : ℚ) / 2 → 
  (∃ (a b c : ℤ), x = a ∧ y = b ∧ z = c) →
  (∀ (x' y' z' : ℤ), (x' : ℚ) / 5 = 6 / (y' : ℚ) → (x' : ℚ) / 5 = (z' : ℚ) / 2 → x + y + z ≥ x' + y' + z') →
  x + y + z = 43 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_proportional_integers_l460_46042


namespace NUMINAMATH_CALUDE_percentage_difference_l460_46037

theorem percentage_difference : (62 / 100 * 150) - (20 / 100 * 250) = 43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l460_46037


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l460_46093

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l460_46093


namespace NUMINAMATH_CALUDE_range_of_a_l460_46048

-- Define the function f
def f (x : ℝ) : ℝ := x * (abs x + 4)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a^2) + f a < 0) → (-1 < a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l460_46048


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l460_46066

theorem largest_n_for_unique_k : ∃ (k : ℤ),
  (8 : ℚ) / 15 < (112 : ℚ) / (112 + k) ∧ (112 : ℚ) / (112 + k) < 7 / 13 ∧
  ∀ (m : ℕ) (k' : ℤ), m > 112 →
    ((8 : ℚ) / 15 < (m : ℚ) / (m + k') ∧ (m : ℚ) / (m + k') < 7 / 13 →
     ∃ (k'' : ℤ), k'' ≠ k' ∧ (8 : ℚ) / 15 < (m : ℚ) / (m + k'') ∧ (m : ℚ) / (m + k'') < 7 / 13) :=
by sorry

#check largest_n_for_unique_k

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l460_46066


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l460_46012

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_6 = 21, prove that a_8 = 63 -/
theorem geometric_sequence_a8 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a6 : a 6 = 21) : a 8 = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l460_46012


namespace NUMINAMATH_CALUDE_equation_has_six_roots_l460_46023

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 1)^3 / (x^2 * (x - 1)^2)

def is_root (x : ℝ) : Prop := f x = f Real.pi

theorem equation_has_six_roots :
  ∃ (r1 r2 r3 r4 r5 r6 : ℝ),
    r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
    r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
    r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
    r4 ≠ r5 ∧ r4 ≠ r6 ∧
    r5 ≠ r6 ∧
    is_root r1 ∧ is_root r2 ∧ is_root r3 ∧ is_root r4 ∧ is_root r5 ∧ is_root r6 ∧
    ∀ x : ℝ, is_root x → (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4 ∨ x = r5 ∨ x = r6) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_six_roots_l460_46023


namespace NUMINAMATH_CALUDE_odd_periodic_monotone_increasing_l460_46014

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y < b → f x < f y

theorem odd_periodic_monotone_increasing (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 4)
  (h_monotone : monotone_increasing_on f 0 2) :
  f 3 < 0 ∧ 0 < f 1 := by sorry

end NUMINAMATH_CALUDE_odd_periodic_monotone_increasing_l460_46014


namespace NUMINAMATH_CALUDE_triangle_inequality_l460_46022

theorem triangle_inequality (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b →
  ¬(a = 3 ∧ b = 5 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l460_46022


namespace NUMINAMATH_CALUDE_polynomial_independent_of_x_l460_46003

-- Define the polynomial
def polynomial (x y a b : ℝ) : ℝ := 9*x^3 + y^2 + a*x - b*x^3 + x + 5

-- State the theorem
theorem polynomial_independent_of_x (y a b : ℝ) :
  (∀ x₁ x₂ : ℝ, polynomial x₁ y a b = polynomial x₂ y a b) →
  a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independent_of_x_l460_46003


namespace NUMINAMATH_CALUDE_correct_calculation_l460_46065

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l460_46065


namespace NUMINAMATH_CALUDE_interest_rate_first_part_l460_46036

/-- Given a total sum and a second part, calculates the first part. -/
def firstPart (total second : ℝ) : ℝ := total - second

/-- Calculates simple interest. -/
def simpleInterest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Theorem stating the interest rate for the first part is 3% per annum. -/
theorem interest_rate_first_part (total second : ℝ) (h1 : total = 2704) (h2 : second = 1664) :
  let first := firstPart total second
  let rate2 := 0.05
  let time1 := 8
  let time2 := 3
  simpleInterest first ((3 : ℝ) / 100) time1 = simpleInterest second rate2 time2 := by
  sorry

#check interest_rate_first_part

end NUMINAMATH_CALUDE_interest_rate_first_part_l460_46036


namespace NUMINAMATH_CALUDE_guppies_count_l460_46087

/-- The number of Goldfish -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Swordtails -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

theorem guppies_count :
  (num_goldfish : ℚ) * goldfish_food +
  (num_swordtails : ℚ) * swordtail_food +
  (num_guppies : ℚ) * guppy_food = total_food :=
by sorry

end NUMINAMATH_CALUDE_guppies_count_l460_46087


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l460_46046

/-- The eccentricity of a hyperbola with specific intersection properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let Γ := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let c := Real.sqrt (a^2 + b^2)
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∀ A B : ℝ × ℝ,
    A ∈ Γ → B ∈ Γ →
    (∃ t : ℝ, A = F₂ + t • (B - F₂)) →
    ‖A - F₁‖ = ‖F₁ - F₂‖ →
    ‖B - F₂‖ = 2 * ‖A - F₂‖ →
    c / a = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l460_46046


namespace NUMINAMATH_CALUDE_incenter_is_intersection_of_angle_bisectors_l460_46009

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance from a point to a line segment -/
noncomputable def distanceToSide (P : Point) (side : Point × Point) : ℝ := sorry

/-- The angle bisector of an angle in a triangle -/
noncomputable def angleBisector (vertex : Point) (side1 : Point) (side2 : Point) : Point × Point := sorry

/-- The intersection point of two lines -/
noncomputable def lineIntersection (line1 : Point × Point) (line2 : Point × Point) : Point := sorry

theorem incenter_is_intersection_of_angle_bisectors (T : Triangle) :
  ∃ (P : Point),
    (∀ (side : Point × Point), 
      side ∈ [(T.A, T.B), (T.B, T.C), (T.C, T.A)] → 
      distanceToSide P side = distanceToSide P (T.A, T.B)) ↔
    (P = lineIntersection 
      (angleBisector T.A T.B T.C) 
      (angleBisector T.B T.C T.A)) :=
by sorry

end NUMINAMATH_CALUDE_incenter_is_intersection_of_angle_bisectors_l460_46009


namespace NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l460_46070

theorem same_root_implies_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l460_46070


namespace NUMINAMATH_CALUDE_trumpet_section_fraction_l460_46080

/-- The fraction of students in the trumpet section -/
def trumpet_fraction : ℝ := sorry

/-- The fraction of students in the trombone section -/
def trombone_fraction : ℝ := 0.12

/-- The fraction of students in either the trumpet or trombone section -/
def trumpet_or_trombone_fraction : ℝ := 0.63

theorem trumpet_section_fraction :
  trumpet_fraction = 0.51 :=
by
  sorry

end NUMINAMATH_CALUDE_trumpet_section_fraction_l460_46080


namespace NUMINAMATH_CALUDE_alice_bob_earnings_l460_46017

/-- Given the working hours and hourly rates of Alice and Bob, prove that the value of t that makes their earnings equal is 7.8 -/
theorem alice_bob_earnings (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_earnings_l460_46017


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l460_46086

def a : Fin 3 → ℝ := ![1, 5, 2]
def b : Fin 3 → ℝ := ![-1, 1, -1]
def c : Fin 3 → ℝ := ![1, 1, 1]

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l460_46086


namespace NUMINAMATH_CALUDE_students_liking_new_menu_l460_46069

theorem students_liking_new_menu (total_students : ℕ) (disliking_students : ℕ) 
  (h1 : total_students = 400) 
  (h2 : disliking_students = 165) : 
  total_students - disliking_students = 235 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_new_menu_l460_46069


namespace NUMINAMATH_CALUDE_pictures_on_sixth_day_l460_46031

def artists_group1 : ℕ := 6
def artists_group2 : ℕ := 8
def days_interval1 : ℕ := 2
def days_interval2 : ℕ := 3
def days_observed : ℕ := 5
def pictures_in_5_days : ℕ := 30

theorem pictures_on_sixth_day :
  let total_6_days := artists_group1 * (6 / days_interval1) + artists_group2 * (6 / days_interval2)
  (total_6_days - pictures_in_5_days : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_sixth_day_l460_46031


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l460_46027

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_defined_l460_46027


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l460_46004

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l460_46004


namespace NUMINAMATH_CALUDE_product_parity_probabilities_l460_46024

/-- The probability that the product of two arbitrary natural numbers is even -/
def prob_even_product : ℚ := 3/4

/-- The probability that the product of two arbitrary natural numbers is odd -/
def prob_odd_product : ℚ := 1/4

theorem product_parity_probabilities :
  (prob_even_product + prob_odd_product = 1) ∧
  (prob_even_product = 3/4) ∧
  (prob_odd_product = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_product_parity_probabilities_l460_46024


namespace NUMINAMATH_CALUDE_max_cube_path_length_l460_46082

/-- Represents a cube with edges of a given length -/
structure Cube where
  edgeLength : ℝ
  edgeCount : ℕ

/-- Represents a path on the cube -/
structure CubePath where
  length : ℝ
  edgeCount : ℕ

/-- The maximum path length on a cube without retracing -/
def maxPathLength (c : Cube) : ℝ := sorry

theorem max_cube_path_length 
  (c : Cube) 
  (h1 : c.edgeLength = 3)
  (h2 : c.edgeCount = 12) :
  maxPathLength c = 24 := by sorry

end NUMINAMATH_CALUDE_max_cube_path_length_l460_46082


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l460_46074

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 85 ∧ Nat.gcd 36 n = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l460_46074


namespace NUMINAMATH_CALUDE_birds_in_tree_l460_46091

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) : 
  initial_birds = initial_birds := by sorry

end NUMINAMATH_CALUDE_birds_in_tree_l460_46091


namespace NUMINAMATH_CALUDE_gcd_360_1260_l460_46062

theorem gcd_360_1260 : Nat.gcd 360 1260 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_1260_l460_46062


namespace NUMINAMATH_CALUDE_stating_exist_same_arrangement_l460_46006

/-- The size of the grid -/
def grid_size : Nat := 25

/-- The size of the sub-squares we're considering -/
def square_size : Nat := 3

/-- The number of possible 3x3 squares in a 25x25 grid -/
def num_squares : Nat := (grid_size - square_size + 1) ^ 2

/-- The number of possible arrangements of plus signs in a 3x3 square -/
def num_arrangements : Nat := 2 ^ (square_size ^ 2)

/-- 
Theorem stating that there exist at least two 3x3 squares 
with the same arrangement of plus signs in a 25x25 grid 
-/
theorem exist_same_arrangement : num_squares > num_arrangements := by sorry

end NUMINAMATH_CALUDE_stating_exist_same_arrangement_l460_46006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l460_46008

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 2 * (a 1 + 6 * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l460_46008


namespace NUMINAMATH_CALUDE_mia_average_first_four_days_l460_46083

theorem mia_average_first_four_days 
  (total_distance : ℝ) 
  (race_days : ℕ) 
  (jesse_avg_first_three : ℝ) 
  (jesse_day_four : ℝ) 
  (combined_avg_last_three : ℝ) 
  (h1 : total_distance = 30)
  (h2 : race_days = 7)
  (h3 : jesse_avg_first_three = 2/3)
  (h4 : jesse_day_four = 10)
  (h5 : combined_avg_last_three = 6) :
  ∃ mia_avg_first_four : ℝ,
    mia_avg_first_four = 3 ∧
    mia_avg_first_four * 4 + combined_avg_last_three * 3 = total_distance ∧
    jesse_avg_first_three * 3 + jesse_day_four + combined_avg_last_three * 3 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_mia_average_first_four_days_l460_46083


namespace NUMINAMATH_CALUDE_expression_factorization_l460_46016

theorem expression_factorization (x : ℝ) :
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l460_46016


namespace NUMINAMATH_CALUDE_cube_sum_problem_l460_46072

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l460_46072


namespace NUMINAMATH_CALUDE_salem_women_count_l460_46026

/-- Proves the number of women in Salem after population change -/
theorem salem_women_count (leesburg_population : ℕ) (salem_multiplier : ℕ) (people_moving_out : ℕ) :
  leesburg_population = 58940 →
  salem_multiplier = 15 →
  people_moving_out = 130000 →
  (salem_multiplier * leesburg_population - people_moving_out) / 2 = 377050 := by
sorry

end NUMINAMATH_CALUDE_salem_women_count_l460_46026


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l460_46002

theorem arithmetic_progression_of_primes (p d : ℕ) : 
  p ≠ 3 →
  Prime (p - d) →
  Prime p →
  Prime (p + d) →
  ∃ k : ℕ, d = 6 * k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l460_46002


namespace NUMINAMATH_CALUDE_booboo_arrangements_l460_46043

def word_arrangements (n : ℕ) (r₁ : ℕ) (r₂ : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r₁ * Nat.factorial r₂)

theorem booboo_arrangements :
  word_arrangements 6 2 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_booboo_arrangements_l460_46043


namespace NUMINAMATH_CALUDE_min_value_theorem_l460_46001

theorem min_value_theorem (a₁ a₂ : ℝ) 
  (h : (3 / (3 + 2 * Real.sin a₁)) + (2 / (4 - Real.sin (2 * a₂))) = 1) :
  ∃ (m : ℝ), m = π / 4 ∧ ∀ (x : ℝ), |4 * π - a₁ + a₂| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l460_46001


namespace NUMINAMATH_CALUDE_wall_bricks_proof_l460_46094

/-- The time (in hours) it takes Alice to build the wall alone -/
def alice_time : ℝ := 8

/-- The time (in hours) it takes Bob to build the wall alone -/
def bob_time : ℝ := 12

/-- The decrease in productivity (in bricks per hour) when Alice and Bob work together -/
def productivity_decrease : ℝ := 15

/-- The time (in hours) it takes Alice and Bob to build the wall together -/
def combined_time : ℝ := 6

/-- The number of bricks in the wall -/
def wall_bricks : ℝ := 360

theorem wall_bricks_proof :
  let alice_rate := wall_bricks / alice_time
  let bob_rate := wall_bricks / bob_time
  let combined_rate := alice_rate + bob_rate - productivity_decrease
  combined_rate * combined_time = wall_bricks := by
  sorry

#check wall_bricks_proof

end NUMINAMATH_CALUDE_wall_bricks_proof_l460_46094


namespace NUMINAMATH_CALUDE_share_ratio_l460_46081

/-- Given a total amount divided among three people (a, b, c), prove the ratio of a's share to the sum of b's and c's shares -/
theorem share_ratio (total a b c : ℚ) : 
  total = 100 →
  a = 20 →
  b = (3 / 5) * (a + c) →
  total = a + b + c →
  a / (b + c) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l460_46081


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l460_46067

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem 1: A line is perpendicular to a plane iff it's perpendicular to every line in the plane
theorem perpendicular_to_plane_iff_perpendicular_to_all_lines 
  (l : Line) (p : Plane) :
  perpendicular_to_plane l p ↔ 
  ∀ (m : Line), in_plane m p → perpendicular l m :=
sorry

-- Theorem 2: If a is parallel to b, and l is perpendicular to a, then l is perpendicular to b
theorem perpendicular_parallel_transitive 
  (a b l : Line) :
  parallel a b → perpendicular l a → perpendicular l b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l460_46067


namespace NUMINAMATH_CALUDE_triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l460_46085

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := -2 * a * b - b + 1

-- Theorem statements
theorem triangle_3_7 : triangle 3 7 = -48 := by sorry

theorem triangle_3_neg4 : triangle 3 (-4) = 29 := by sorry

theorem triangle_neg4_3 : triangle (-4) 3 = 22 := by sorry

theorem triangle_not_commutative : ∃ a b : ℚ, triangle a b ≠ triangle b a := by sorry

end NUMINAMATH_CALUDE_triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l460_46085


namespace NUMINAMATH_CALUDE_min_value_7x_5y_l460_46092

theorem min_value_7x_5y (x y : ℕ) 
  (h1 : ∃ k : ℤ, x + 2*y = 5*k)
  (h2 : ∃ m : ℤ, x + y = 3*m)
  (h3 : 2*x + y ≥ 99) :
  7*x + 5*y ≥ 366 := by
sorry

end NUMINAMATH_CALUDE_min_value_7x_5y_l460_46092


namespace NUMINAMATH_CALUDE_final_amount_is_16_l460_46045

def purchase1 : ℚ := 215/100
def purchase2 : ℚ := 475/100
def purchase3 : ℚ := 1060/100
def discount_rate : ℚ := 1/10

def total_before_discount : ℚ := purchase1 + purchase2 + purchase3
def discounted_total : ℚ := total_before_discount * (1 - discount_rate)

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem final_amount_is_16 :
  round_to_nearest_dollar discounted_total = 16 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_is_16_l460_46045


namespace NUMINAMATH_CALUDE_round_trip_no_car_percentage_l460_46059

theorem round_trip_no_car_percentage
  (total_round_trip : ℝ)
  (round_trip_with_car : ℝ)
  (h1 : round_trip_with_car = 25)
  (h2 : total_round_trip = 62.5) :
  total_round_trip - round_trip_with_car = 37.5 := by
sorry

end NUMINAMATH_CALUDE_round_trip_no_car_percentage_l460_46059


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l460_46071

/-- Given a hyperbola and a line passing through its right focus, 
    prove the equations of the asymptotes -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (C : ℝ × ℝ → Prop) 
  (l : ℝ × ℝ → Prop) 
  (F : ℝ × ℝ) :
  (C = λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1) →
  (l = λ (x, y) ↦ x + 3*y - 2*b = 0) →
  (∃ c, F = (c, 0) ∧ l F) →
  (∃ f : ℝ → ℝ, f x = Real.sqrt 3 / 3 * x ∧ 
   ∀ (x y : ℝ), (C (x, y) → (y = f x ∨ y = -f x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l460_46071


namespace NUMINAMATH_CALUDE_friends_games_l460_46057

theorem friends_games (katie_games : ℕ) (difference : ℕ) : 
  katie_games = 81 → difference = 22 → katie_games - difference = 59 := by
  sorry

end NUMINAMATH_CALUDE_friends_games_l460_46057


namespace NUMINAMATH_CALUDE_graph_not_in_third_quadrant_l460_46079

def f (x : ℝ) : ℝ := -x + 2

theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_graph_not_in_third_quadrant_l460_46079


namespace NUMINAMATH_CALUDE_binomial_9_choose_3_l460_46040

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_3_l460_46040


namespace NUMINAMATH_CALUDE_cosine_domain_range_minimum_l460_46060

open Real

theorem cosine_domain_range_minimum (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = cos x) →
  (∀ x ∈ Set.Icc a b, -1/2 ≤ f x ∧ f x ≤ 1) →
  (∃ x ∈ Set.Icc a b, f x = -1/2) →
  (∃ x ∈ Set.Icc a b, f x = 1) →
  b - a ≥ 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_cosine_domain_range_minimum_l460_46060


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l460_46018

theorem parabola_point_comparison :
  ∀ (y₁ y₂ : ℝ),
  y₁ = (-5)^2 + 2*(-5) + 3 →
  y₂ = 2^2 + 2*2 + 3 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l460_46018


namespace NUMINAMATH_CALUDE_cookie_cost_l460_46020

/-- Proves that the cost of each cookie is $15, given the conditions of Andrew's cookie purchases in May. -/
theorem cookie_cost (cookies_per_day : ℕ) (days_in_may : ℕ) (total_spent : ℕ) : 
  cookies_per_day = 3 →
  days_in_may = 31 →
  total_spent = 1395 →
  (total_spent : ℚ) / (cookies_per_day * days_in_may : ℚ) = 15 := by
  sorry


end NUMINAMATH_CALUDE_cookie_cost_l460_46020


namespace NUMINAMATH_CALUDE_sqrt_3a_plus_2b_l460_46099

theorem sqrt_3a_plus_2b (a b : ℝ) 
  (h1 : (2*a + 3)^2 = 3^2) 
  (h2 : (5*a + 2*b - 1)^2 = 4^2) : 
  (3*a + 2*b)^2 = 4^2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_3a_plus_2b_l460_46099


namespace NUMINAMATH_CALUDE_parabola_equation_theorem_l460_46078

/-- A parabola with vertex at the origin, x-axis as the axis of symmetry, 
    and passing through the point (4, -2) -/
structure Parabola where
  -- The parabola passes through (4, -2)
  passes_through : (4 : ℝ)^2 + (-2 : ℝ)^2 ≠ 0

/-- The equation of the parabola is either y^2 = x or x^2 = -8y -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = x) ∨ (∀ x y : ℝ, x^2 = -8*y)

/-- Theorem stating that the parabola satisfies one of the two equations -/
theorem parabola_equation_theorem (p : Parabola) : parabola_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_theorem_l460_46078


namespace NUMINAMATH_CALUDE_quadratic_equation_b_value_l460_46025

theorem quadratic_equation_b_value 
  (b : ℝ) 
  (h1 : 2 * (5 : ℝ)^2 + b * 5 - 65 = 0) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_b_value_l460_46025


namespace NUMINAMATH_CALUDE_specific_building_height_l460_46053

/-- Calculates the height of a building with specific floor heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_height_l460_46053


namespace NUMINAMATH_CALUDE_circuit_probability_l460_46007

/-- The probability that a circuit with two independently controlled switches
    connected in parallel can operate normally. -/
theorem circuit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  p1 * (1 - p2) + (1 - p1) * p2 + p1 * p2 = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_circuit_probability_l460_46007


namespace NUMINAMATH_CALUDE_circle_area_equality_l460_46096

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) (h₃ : r₃ = 20) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l460_46096


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l460_46000

/-- A line passing through two points (2, 9) and (5, 17) intersects the y-axis at (0, 11/3) -/
theorem line_intersection_y_axis : 
  ∃ (m b : ℚ), 
    (9 = m * 2 + b) ∧ 
    (17 = m * 5 + b) ∧ 
    (11/3 = b) := by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l460_46000


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l460_46073

def salvadore_earnings : ℕ := 1956

def santo_earnings (salvadore : ℕ) : ℕ := salvadore / 2

def maria_earnings (santo : ℕ) : ℕ := santo * 3

def pedro_earnings (santo maria : ℕ) : ℕ := santo + maria

theorem total_earnings_theorem (salvadore : ℕ) 
  (h_salvadore : salvadore = salvadore_earnings) : 
  salvadore + 
  santo_earnings salvadore + 
  maria_earnings (santo_earnings salvadore) + 
  pedro_earnings (santo_earnings salvadore) (maria_earnings (santo_earnings salvadore)) = 9780 := by
  sorry


end NUMINAMATH_CALUDE_total_earnings_theorem_l460_46073


namespace NUMINAMATH_CALUDE_max_value_of_z_l460_46029

theorem max_value_of_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = x + y / 2 ∧ z ≤ 7 ∧ ∀ (w : ℝ), w = x + y / 2 → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l460_46029


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_three_l460_46095

theorem three_digit_multiples_of_three : 
  (Finset.filter (fun c => (100 + 10 * c + 7) % 3 = 0) (Finset.range 10)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_three_l460_46095


namespace NUMINAMATH_CALUDE_max_value_of_expression_l460_46011

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^10 + x^8 - 6*x^6 + 27*x^4 + 64) ≤ 1/8.38 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l460_46011


namespace NUMINAMATH_CALUDE_coffee_maker_price_l460_46058

theorem coffee_maker_price (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) : 
  sale_price = 70 → discount = 20 → original_price = sale_price + discount → original_price = 90 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l460_46058


namespace NUMINAMATH_CALUDE_polyhedron_edge_vertex_relation_l460_46064

/-- Represents a polyhedron with its vertex and edge properties -/
structure Polyhedron where
  /-- p k is the number of vertices where k edges meet -/
  p : ℕ → ℕ
  /-- a is the total number of edges -/
  a : ℕ

/-- The sum of k * p k for all k ≥ 3 equals twice the total number of edges -/
theorem polyhedron_edge_vertex_relation (P : Polyhedron) :
  2 * P.a = ∑' k, k * P.p k := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_vertex_relation_l460_46064


namespace NUMINAMATH_CALUDE_complex_equation_solution_l460_46038

theorem complex_equation_solution (a : ℝ) (ha : a ≥ 0) :
  let S := {z : ℂ | z^2 + 2 * Complex.abs z = a}
  S = {z : ℂ | z = -(1 - Real.sqrt (1 + a)) ∨ z = (1 - Real.sqrt (1 + a))} ∪
      (if 0 ≤ a ∧ a ≤ 1 then
        {z : ℂ | z = Complex.I * (1 + Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 + Real.sqrt (1 - a))) ∨
                 z = Complex.I * (1 - Real.sqrt (1 - a)) ∨
                 z = Complex.I * (-(1 - Real.sqrt (1 - a)))}
      else ∅) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l460_46038


namespace NUMINAMATH_CALUDE_negation_of_tan_gt_sin_l460_46068

open Real

theorem negation_of_tan_gt_sin :
  (¬ (∀ x, -π/2 < x ∧ x < π/2 → tan x > sin x)) ↔
  (∃ x, -π/2 < x ∧ x < π/2 ∧ tan x ≤ sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_tan_gt_sin_l460_46068
