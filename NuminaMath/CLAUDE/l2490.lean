import Mathlib

namespace NUMINAMATH_CALUDE_line_through_points_l2490_249082

/-- Given a line y = cx + d passing through the points (3, -3) and (6, 9), prove that c + d = -11 -/
theorem line_through_points (c d : ℝ) : 
  (-3 : ℝ) = c * 3 + d → 
  9 = c * 6 + d → 
  c + d = -11 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2490_249082


namespace NUMINAMATH_CALUDE_complex_function_from_real_part_l2490_249070

open Complex

/-- Given that u(x, y) = x^2 - y^2 + 2x is the real part of a differentiable complex function f(z),
    prove that f(z) = z^2 + 2z + c for some constant c. -/
theorem complex_function_from_real_part
  (f : ℂ → ℂ)
  (h_diff : Differentiable ℂ f)
  (h_real_part : ∀ z : ℂ, (f z).re = z.re^2 - z.im^2 + 2*z.re) :
  ∃ c : ℂ, ∀ z : ℂ, f z = z^2 + 2*z + c :=
sorry

end NUMINAMATH_CALUDE_complex_function_from_real_part_l2490_249070


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2490_249058

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 - 4 * x - 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 3)^2 = 3 * x * (x - 3)
  let sol1 : Set ℝ := {(2 + Real.sqrt 6) / 2, (2 - Real.sqrt 6) / 2}
  let sol2 : Set ℝ := {3, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) := by
  sorry

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2490_249058


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l2490_249063

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  cube_edge_length : Real

/-- Calculates the exposed surface area of a cube sculpture -/
def exposed_surface_area (sculpture : CubeSculpture) : Real :=
  let top_area := sculpture.top_layer * (5 * sculpture.cube_edge_length ^ 2)
  let middle_area := 4 * sculpture.middle_layer * sculpture.cube_edge_length ^ 2
  let bottom_area := sculpture.bottom_layer * sculpture.cube_edge_length ^ 2
  top_area + middle_area + bottom_area

/-- The main theorem stating that the exposed surface area of the specific sculpture is 35 square meters -/
theorem sculpture_surface_area :
  let sculpture : CubeSculpture := {
    top_layer := 1,
    middle_layer := 6,
    bottom_layer := 12,
    cube_edge_length := 1
  }
  exposed_surface_area sculpture = 35 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l2490_249063


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_l2490_249045

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -3]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular a (fun i => a i + b m i) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_l2490_249045


namespace NUMINAMATH_CALUDE_escalator_problem_solution_l2490_249077

/-- The time taken to descend an escalator under different conditions -/
def EscalatorProblem (s : ℝ) : Prop :=
  let t_standing := (3/2 : ℝ)  -- Time taken when standing on moving escalator
  let t_running_stationary := (1 : ℝ)  -- Time taken when running on stationary escalator
  let v_escalator := s / t_standing  -- Speed of escalator
  let v_running := s / t_running_stationary  -- Speed of running
  let v_combined := v_escalator + v_running  -- Combined speed
  let t_running_moving := s / v_combined  -- Time taken when running on moving escalator
  t_running_moving = (3/5 : ℝ)

/-- Theorem stating the solution to the escalator problem -/
theorem escalator_problem_solution :
  ∀ s : ℝ, s > 0 → EscalatorProblem s :=
by
  sorry

end NUMINAMATH_CALUDE_escalator_problem_solution_l2490_249077


namespace NUMINAMATH_CALUDE_range_of_m_l2490_249067

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (B m).Nonempty ∧ B m ⊆ A → 2 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2490_249067


namespace NUMINAMATH_CALUDE_count_valid_words_l2490_249072

/-- The number of letters in each word -/
def word_length : ℕ := 4

/-- The number of available letters -/
def alphabet_size : ℕ := 5

/-- The number of letters that must be included -/
def required_letters : ℕ := 2

/-- The number of 4-letter words that can be formed using the letters A, B, C, D, and E, 
    with repetition allowed, and including both A and E at least once -/
def valid_words : ℕ := alphabet_size^word_length - 2*(alphabet_size-1)^word_length + (alphabet_size-2)^word_length

theorem count_valid_words : valid_words = 194 := by sorry

end NUMINAMATH_CALUDE_count_valid_words_l2490_249072


namespace NUMINAMATH_CALUDE_fathers_age_fathers_age_is_52_l2490_249044

theorem fathers_age (sons_age_5_years_ago : ℕ) (years_passed : ℕ) : ℕ :=
  let sons_current_age := sons_age_5_years_ago + years_passed
  2 * sons_current_age

theorem fathers_age_is_52 : fathers_age 21 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_fathers_age_is_52_l2490_249044


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l2490_249002

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Given three mutually externally tangent circles with radii 1, 2, and 3,
    returns the triangle formed by their points of tangency -/
def tangentTriangle (c1 c2 c3 : Circle) : Triangle := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Three mutually externally tangent circles with radii 1, 2, and 3 -/
def circle1 : Circle := { center := (0, 0), radius := 1 }
def circle2 : Circle := { center := (3, 0), radius := 2 }
def circle3 : Circle := { center := (0, 4), radius := 3 }

theorem tangent_triangle_area :
  triangleArea (tangentTriangle circle1 circle2 circle3) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l2490_249002


namespace NUMINAMATH_CALUDE_purple_shoes_count_l2490_249078

theorem purple_shoes_count (total : ℕ) (blue : ℕ) (h1 : total = 1250) (h2 : blue = 540) :
  let remaining := total - blue
  let purple := remaining / 2
  purple = 355 := by
sorry

end NUMINAMATH_CALUDE_purple_shoes_count_l2490_249078


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2490_249094

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 129.6 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2490_249094


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l2490_249027

/-- Given a large semicircle with radius 12, a circle with radius 6 inside it, 
    and a smaller semicircle, all pairwise tangent to each other, 
    the radius of the smaller semicircle is 4. -/
theorem small_semicircle_radius (r : ℝ) 
  (h1 : r > 0) -- radius of smaller semicircle is positive
  (h2 : 12 > 0) -- radius of larger semicircle is positive
  (h3 : 6 > 0)  -- radius of circle is positive
  (h4 : r < 12) -- radius of smaller semicircle is less than larger semicircle
  (h5 : r + 6 < 12) -- sum of radii of smaller semicircle and circle is less than larger semicircle
  : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l2490_249027


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l2490_249059

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a / Real.exp x

theorem tangent_parallel_implies_a_equals_e (a : ℝ) :
  f_derivative a 1 = 0 → a = Real.exp 1 := by sorry

theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), f a x = Real.log a ∧ 
  ∀ (y : ℝ), f a y ≤ f a x := by sorry

theorem no_extreme_values_when_a_nonpositive (a : ℝ) (h : a ≤ 0) :
  ∀ (x : ℝ), ∃ (y : ℝ), f a y > f a x := by sorry

end

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l2490_249059


namespace NUMINAMATH_CALUDE_special_function_max_l2490_249073

open Real

/-- A continuous function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x y, f (x + y) * f (x - y) = f x ^ 2 - f y ^ 2) ∧
  (∀ x, f (x + 2 * π) = f x) ∧
  (∀ a, 0 < a → a < 2 * π → ∃ x, f (x + a) ≠ f x)

/-- The main theorem to be proved -/
theorem special_function_max (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, |f (π / 2)| ≥ f x :=
sorry

end NUMINAMATH_CALUDE_special_function_max_l2490_249073


namespace NUMINAMATH_CALUDE_share_difference_l2490_249035

theorem share_difference (total amount_a amount_b amount_c : ℕ) : 
  total = 120 →
  amount_b = 20 →
  amount_a + amount_b + amount_c = total →
  amount_a = amount_c - 20 →
  amount_a - amount_b = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_share_difference_l2490_249035


namespace NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l2490_249053

/-- A parallelogram in a 2D plane -/
structure Parallelogram :=
  (P Q R S : ℝ × ℝ)

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Calculate the area of a parallelogram -/
def area_parallelogram (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a triangle -/
def area_triangle (t : Triangle) : ℝ :=
  sorry

/-- Check if a triangle is inside a parallelogram -/
def is_inside (t : Triangle) (p : Parallelogram) : Prop :=
  sorry

/-- Theorem: The area of a triangle inside a parallelogram is at most half the area of the parallelogram -/
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : Triangle) 
  (h : is_inside t p) : area_triangle t ≤ (1/2) * area_parallelogram p :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l2490_249053


namespace NUMINAMATH_CALUDE_diagonal_pigeonhole_l2490_249023

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths in a regular n-gon -/
def distinct_lengths (n : ℕ) : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n (n : ℕ) : ℕ := distinct_lengths n + 1

theorem diagonal_pigeonhole :
  smallest_n n = 1008 :=
sorry

end NUMINAMATH_CALUDE_diagonal_pigeonhole_l2490_249023


namespace NUMINAMATH_CALUDE_min_green_tiles_l2490_249061

/-- Represents the colors of tiles --/
inductive Color
  | Red
  | Orange
  | Yellow
  | Green
  | Blue
  | Indigo

/-- Represents the number of tiles for each color --/
structure TileCount where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ
  blue : ℕ
  indigo : ℕ

/-- The total number of tiles --/
def total_tiles : ℕ := 100

/-- Checks if the tile count satisfies all constraints --/
def satisfies_constraints (tc : TileCount) : Prop :=
  tc.red + tc.orange + tc.yellow + tc.green + tc.blue + tc.indigo = total_tiles ∧
  tc.indigo ≥ tc.red + tc.orange + tc.yellow + tc.green + tc.blue ∧
  tc.blue ≥ tc.red + tc.orange + tc.yellow + tc.green ∧
  tc.green ≥ tc.red + tc.orange + tc.yellow ∧
  tc.yellow ≥ tc.red + tc.orange ∧
  tc.orange ≥ tc.red

/-- Checks if one tile count is preferred over another according to the client's preferences --/
def is_preferred (tc1 tc2 : TileCount) : Prop :=
  tc1.red > tc2.red ∨
  (tc1.red = tc2.red ∧ tc1.orange > tc2.orange) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow > tc2.yellow) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green > tc2.green) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue > tc2.blue) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue = tc2.blue ∧ tc1.indigo > tc2.indigo)

/-- The theorem to be proved --/
theorem min_green_tiles :
  ∃ (optimal : TileCount),
    satisfies_constraints optimal ∧
    optimal.green = 13 ∧
    ∀ (tc : TileCount), satisfies_constraints tc → ¬is_preferred tc optimal :=
by sorry

end NUMINAMATH_CALUDE_min_green_tiles_l2490_249061


namespace NUMINAMATH_CALUDE_tenth_grader_average_score_l2490_249090

/-- Represents a chess tournament between 9th and 10th graders -/
structure ChessTournament where
  ninth_graders : ℕ
  tenth_graders : ℕ
  tournament_points : ℕ

/-- The number of 10th graders is 10 times the number of 9th graders -/
axiom tenth_grader_count (t : ChessTournament) : t.tenth_graders = 10 * t.ninth_graders

/-- Each player plays every other player exactly once -/
axiom total_games (t : ChessTournament) : t.tournament_points = (t.ninth_graders + t.tenth_graders) * (t.ninth_graders + t.tenth_graders - 1) / 2

/-- The average score of a 10th grader is 10 points -/
theorem tenth_grader_average_score (t : ChessTournament) :
  t.tournament_points / t.tenth_graders = 10 :=
sorry

end NUMINAMATH_CALUDE_tenth_grader_average_score_l2490_249090


namespace NUMINAMATH_CALUDE_proposition_d_is_true_l2490_249057

theorem proposition_d_is_true (a b : ℝ) : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_d_is_true_l2490_249057


namespace NUMINAMATH_CALUDE_cube_of_product_l2490_249008

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l2490_249008


namespace NUMINAMATH_CALUDE_prime_cube_minus_one_divisibility_l2490_249036

theorem prime_cube_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  30 ∣ (p^3 - 1) ↔ p ≡ 1 [MOD 15] := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_minus_one_divisibility_l2490_249036


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2490_249095

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := 
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := 
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 18 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 9 = 0 ∧
    n = 5643 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2490_249095


namespace NUMINAMATH_CALUDE_range_of_3x_plus_2y_l2490_249096

theorem range_of_3x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 4) : 
  2 ≤ 3*x + 2*y ∧ 3*x + 2*y ≤ 9.5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_3x_plus_2y_l2490_249096


namespace NUMINAMATH_CALUDE_powers_of_two_start_with_any_digits_l2490_249089

theorem powers_of_two_start_with_any_digits (A m : ℕ) : 
  ∃ n : ℕ+, (10 ^ m * A : ℝ) < (2 : ℝ) ^ (n : ℝ) ∧ (2 : ℝ) ^ (n : ℝ) < (10 ^ m * (A + 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_powers_of_two_start_with_any_digits_l2490_249089


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_divisible_by_16_l2490_249007

theorem sum_consecutive_odd_integers_divisible_by_16 :
  let start := 2101
  let count := 15
  let sequence := List.range count |>.map (fun i => start + 2 * i)
  sequence.sum % 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_divisible_by_16_l2490_249007


namespace NUMINAMATH_CALUDE_two_heroes_two_villains_l2490_249081

/-- Represents the type of an inhabitant -/
inductive InhabitantType
| Hero
| Villain

/-- Represents an inhabitant on the island -/
structure Inhabitant where
  type : InhabitantType

/-- Represents the table with four inhabitants -/
structure Table where
  inhabitants : Fin 4 → Inhabitant

/-- Defines what it means for an inhabitant to tell the truth -/
def tellsTruth (i : Inhabitant) : Prop :=
  i.type = InhabitantType.Hero

/-- Defines what an inhabitant says about themselves -/
def claimsSelfHero (i : Inhabitant) : Prop :=
  true

/-- Defines what an inhabitant says about the person on their right -/
def claimsRightVillain (t : Table) (pos : Fin 4) : Prop :=
  true

/-- The main theorem stating that the only valid configuration is 2 Heroes and 2 Villains alternating -/
theorem two_heroes_two_villains (t : Table) :
  (∀ (pos : Fin 4), claimsSelfHero (t.inhabitants pos)) →
  (∀ (pos : Fin 4), claimsRightVillain t pos) →
  (∃ (pos : Fin 4),
    tellsTruth (t.inhabitants pos) ∧
    ¬tellsTruth (t.inhabitants (pos + 1)) ∧
    tellsTruth (t.inhabitants (pos + 2)) ∧
    ¬tellsTruth (t.inhabitants (pos + 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_two_heroes_two_villains_l2490_249081


namespace NUMINAMATH_CALUDE_youngest_child_age_l2490_249069

/-- Given 5 children born at intervals of 2 years each, 
    if the sum of their ages is 55 years, 
    then the age of the youngest child is 7 years. -/
theorem youngest_child_age 
  (n : ℕ) 
  (h1 : n = 5) 
  (interval : ℕ) 
  (h2 : interval = 2) 
  (total_age : ℕ) 
  (h3 : total_age = 55) 
  (youngest_age : ℕ) 
  (h4 : youngest_age * n + (n * (n - 1) / 2) * interval = total_age) : 
  youngest_age = 7 := by
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2490_249069


namespace NUMINAMATH_CALUDE_pentagon_area_ratio_l2490_249055

-- Define the pentagon PQRST
structure Pentagon :=
  (P Q R S T : ℝ × ℝ)

-- Define the properties of the pentagon
def ConvexPentagon (p : Pentagon) : Prop :=
  sorry -- Definition of convex pentagon

def ParallelLines (A B C D : ℝ × ℝ) : Prop :=
  sorry -- Definition of parallel lines

def AngleMeasure (A B C : ℝ × ℝ) : ℝ :=
  sorry -- Definition of angle measure

def Distance (A B : ℝ × ℝ) : ℝ :=
  sorry -- Definition of distance between two points

def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  sorry -- Definition of triangle area

-- Theorem statement
theorem pentagon_area_ratio (p : Pentagon) 
  (h_convex : ConvexPentagon p)
  (h_parallel1 : ParallelLines p.P p.Q p.S p.T)
  (h_parallel2 : ParallelLines p.Q p.R p.P p.S)
  (h_parallel3 : ParallelLines p.P p.T p.R p.S)
  (h_angle : AngleMeasure p.P p.Q p.R = 120)
  (h_pq : Distance p.P p.Q = 4)
  (h_qr : Distance p.Q p.R = 6)
  (h_rs : Distance p.R p.S = 18) :
  TriangleArea p.P p.Q p.R / TriangleArea p.Q p.T p.S = 16 / 81 :=
sorry

end NUMINAMATH_CALUDE_pentagon_area_ratio_l2490_249055


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2490_249054

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 1 ∧ 
  k = 2 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 4 ∧ 
  b * b = c * c - a * a → 
  h + k + a + b = 7 + Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2490_249054


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l2490_249062

/-- An even function that is monotonically increasing on the non-negative reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem even_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_even_increasing : EvenIncreasingFunction f) 
  (h_f_1 : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l2490_249062


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l2490_249019

theorem simplify_fraction_1 (a b : ℝ) (h : a ≠ b) :
  (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_l2490_249019


namespace NUMINAMATH_CALUDE_handicraft_sale_properties_l2490_249087

/-- Represents the daily profit function for a handicraft item sale --/
def daily_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

/-- Represents the daily sales volume function --/
def daily_sales (x : ℝ) : ℝ :=
  50 + 5 * (100 - x)

/-- Theorem stating the properties of the handicraft item sale --/
theorem handicraft_sale_properties :
  let cost : ℝ := 50
  let base_price : ℝ := 100
  let base_sales : ℝ := 50
  ∀ x : ℝ, cost ≤ x ∧ x ≤ base_price →
    (daily_profit x = (x - cost) * daily_sales x) ∧
    (∃ max_profit max_price, 
      max_profit = daily_profit max_price ∧
      max_price = 80 ∧ 
      max_profit = 4500 ∧
      ∀ y, cost ≤ y ∧ y ≤ base_price → daily_profit y ≤ max_profit) ∧
    (∃ min_total_cost,
      min_total_cost = 5000 ∧
      ∀ z, cost ≤ z ∧ z ≤ base_price →
        daily_profit z ≥ 4000 → cost * daily_sales z ≥ min_total_cost) := by
  sorry


end NUMINAMATH_CALUDE_handicraft_sale_properties_l2490_249087


namespace NUMINAMATH_CALUDE_inverse_f_128_l2490_249024

/-- Given a function f: ℝ → ℝ satisfying f(4) = 2 and f(2x) = 2f(x) for all x,
    prove that f⁻¹(128) = 256 -/
theorem inverse_f_128 (f : ℝ → ℝ) (h1 : f 4 = 2) (h2 : ∀ x, f (2 * x) = 2 * f x) :
  f⁻¹ 128 = 256 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_128_l2490_249024


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2490_249001

-- Define the universe set U
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Finset ℕ := {2, 4, 6}

-- Define set B
def B : Finset ℕ := {1, 3, 5, 7}

-- Theorem statement
theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2490_249001


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l2490_249079

theorem factor_difference_of_squares (y : ℝ) :
  25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l2490_249079


namespace NUMINAMATH_CALUDE_tv_price_difference_l2490_249085

def budget : ℕ := 1000
def initial_discount : ℕ := 100
def additional_discount_percent : ℕ := 20

theorem tv_price_difference : 
  let price_after_initial_discount := budget - initial_discount
  let additional_discount := price_after_initial_discount * additional_discount_percent / 100
  let final_price := price_after_initial_discount - additional_discount
  budget - final_price = 280 := by sorry

end NUMINAMATH_CALUDE_tv_price_difference_l2490_249085


namespace NUMINAMATH_CALUDE_halfway_fraction_l2490_249012

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2490_249012


namespace NUMINAMATH_CALUDE_gardener_work_theorem_l2490_249000

/-- Represents the outcome of the gardener's work. -/
structure GardenerOutcome where
  diligentDays : ℕ
  shirkingDays : ℕ

/-- Calculates the pretzel balance based on the gardener's work outcome. -/
def pretzelBalance (outcome : GardenerOutcome) : ℤ :=
  (3 * outcome.diligentDays) - outcome.shirkingDays

theorem gardener_work_theorem :
  ∃ (outcome : GardenerOutcome),
    outcome.diligentDays + outcome.shirkingDays = 26 ∧
    pretzelBalance outcome = 62 ∧
    outcome.diligentDays = 22 ∧
    outcome.shirkingDays = 4 := by
  sorry

#check gardener_work_theorem

end NUMINAMATH_CALUDE_gardener_work_theorem_l2490_249000


namespace NUMINAMATH_CALUDE_min_product_equal_sum_l2490_249098

theorem min_product_equal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ ∧ a₀ * b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_product_equal_sum_l2490_249098


namespace NUMINAMATH_CALUDE_expression_value_l2490_249013

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -3) : -a - b^3 + a*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2490_249013


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2490_249051

theorem cube_edge_ratio (v₁ v₂ v₃ v₄ : ℝ) (h : v₁ / v₂ = 216 / 64 ∧ v₂ / v₃ = 64 / 27 ∧ v₃ / v₄ = 27 / 1) :
  ∃ (e₁ e₂ e₃ e₄ : ℝ), v₁ = e₁^3 ∧ v₂ = e₂^3 ∧ v₃ = e₃^3 ∧ v₄ = e₄^3 ∧ 
  e₁ / e₂ = 6 / 4 ∧ e₂ / e₃ = 4 / 3 ∧ e₃ / e₄ = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2490_249051


namespace NUMINAMATH_CALUDE_odd_prime_product_probability_l2490_249005

/-- A standard die with six faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of odd prime numbers on a standard die. -/
def OddPrimeOnDie : Finset ℕ := {3, 5}

/-- The number of times the die is rolled. -/
def NumRolls : ℕ := 8

/-- The probability of rolling an odd prime on a single roll of a standard die. -/
def SingleRollProbability : ℚ := (OddPrimeOnDie.card : ℚ) / (StandardDie.card : ℚ)

theorem odd_prime_product_probability :
  (SingleRollProbability ^ NumRolls : ℚ) = 1 / 6561 :=
sorry

end NUMINAMATH_CALUDE_odd_prime_product_probability_l2490_249005


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_l2490_249016

/-- The general term of the sequence -/
def a (n : ℕ) : ℚ := n + n^2 / (n^2 + 1)

/-- The first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3/2) ∧ 
  (a 2 = 14/5) ∧ 
  (a 3 = 39/10) ∧ 
  (a 4 = 84/17) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_l2490_249016


namespace NUMINAMATH_CALUDE_potato_ratio_l2490_249048

theorem potato_ratio (total : ℕ) (wedges : ℕ) (chip_wedge_diff : ℕ) :
  total = 67 →
  wedges = 13 →
  chip_wedge_diff = 436 →
  let remaining := total - wedges
  let fries := remaining / 2
  let chips := remaining / 2
  fries = chips := by sorry

end NUMINAMATH_CALUDE_potato_ratio_l2490_249048


namespace NUMINAMATH_CALUDE_divisible_by_10101_l2490_249039

/-- Given a two-digit number, returns the six-digit number formed by repeating it three times -/
def f (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + 10 * n

/-- Theorem: For any two-digit number n, f(n) is divisible by 10101 -/
theorem divisible_by_10101 (n : ℕ) (h : 10 ≤ n ∧ n < 100) : 
  ∃ k : ℕ, f n = 10101 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_10101_l2490_249039


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2490_249088

/-- Perimeter of triangle PF₁F₂ for a specific ellipse -/
theorem ellipse_triangle_perimeter :
  ∀ (a b c : ℝ) (P F₁ F₂ : ℝ × ℝ),
  -- Ellipse equation
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | x^2 / a^2 + y^2 / 2 = 1} → x^2 / a^2 + y^2 / 2 = 1) →
  -- F₁ and F₂ are foci
  F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0 →
  -- P is on the ellipse
  P ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 2 = 1} →
  -- F₁ is symmetric to y = -x at P
  P.1 = -F₁.2 ∧ P.2 = -F₁.1 →
  -- Perimeter of triangle PF₁F₂
  dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2490_249088


namespace NUMINAMATH_CALUDE_base8_addition_subtraction_l2490_249026

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (x : ℕ) : ℕ :=
  let ones := x % 10
  let eights := x / 10
  8 * eights + ones

/-- Converts a base 10 number to base 8 --/
def base10ToBase8 (x : ℕ) : ℕ :=
  let quotient := x / 8
  let remainder := x % 8
  10 * quotient + remainder

theorem base8_addition_subtraction :
  base10ToBase8 ((base8ToBase10 10 + base8ToBase10 26) - base8ToBase10 13) = 23 := by
  sorry

end NUMINAMATH_CALUDE_base8_addition_subtraction_l2490_249026


namespace NUMINAMATH_CALUDE_defective_components_probability_l2490_249049

-- Define the probability function
def probability (p q r : ℕ) : ℚ :=
  let total_components := p + q
  let numerator := q * (Nat.descFactorial (r-1) (q-1)) * (Nat.descFactorial p (r-q)) +
                   p * (Nat.descFactorial (r-1) (p-1)) * (Nat.descFactorial q (r-p))
  let denominator := Nat.descFactorial total_components r
  ↑numerator / ↑denominator

-- State the theorem
theorem defective_components_probability (p q r : ℕ) 
  (h1 : q < p) (h2 : p < r) (h3 : r < p + q) :
  probability p q r = (↑q * Nat.descFactorial (r-1) (q-1) * Nat.descFactorial p (r-q) + 
                       ↑p * Nat.descFactorial (r-1) (p-1) * Nat.descFactorial q (r-p)) / 
                      Nat.descFactorial (p+q) r :=
by
  sorry


end NUMINAMATH_CALUDE_defective_components_probability_l2490_249049


namespace NUMINAMATH_CALUDE_concentric_circles_circumference_difference_l2490_249018

/-- The difference in circumferences of two concentric circles -/
theorem concentric_circles_circumference_difference 
  (inner_diameter : ℝ) 
  (distance_between_circles : ℝ) : 
  inner_diameter = 100 → 
  distance_between_circles = 15 → 
  (inner_diameter + 2 * distance_between_circles) * π - inner_diameter * π = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_circumference_difference_l2490_249018


namespace NUMINAMATH_CALUDE_gym_member_ratio_l2490_249042

theorem gym_member_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_gym_member_ratio_l2490_249042


namespace NUMINAMATH_CALUDE_triangle_area_l2490_249006

/-- The area of a triangle with base 15 and height p is equal to 15p/2 -/
theorem triangle_area (p : ℝ) : 
  (1/2 : ℝ) * 15 * p = 15 * p / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2490_249006


namespace NUMINAMATH_CALUDE_participation_schemes_l2490_249075

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of people to be selected -/
def selected_people : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 3

/-- The number of special people (A and B) -/
def special_people : ℕ := 2

/-- Calculates the number of permutations of r items from n -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem participation_schemes : 
  permutations total_people selected_people - 
  permutations (total_people - special_people) selected_people = 54 := by
sorry

end NUMINAMATH_CALUDE_participation_schemes_l2490_249075


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l2490_249025

theorem function_inequality_implies_m_bound (f g : ℝ → ℝ) (m : ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l2490_249025


namespace NUMINAMATH_CALUDE_only_c_is_perfect_square_l2490_249015

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

def number_a : ℕ := 4^4 * 5^5 * 6^6
def number_b : ℕ := 4^4 * 5^6 * 6^5
def number_c : ℕ := 4^5 * 5^4 * 6^6
def number_d : ℕ := 4^6 * 5^4 * 6^5
def number_e : ℕ := 4^6 * 5^5 * 6^4

theorem only_c_is_perfect_square :
  ¬(is_perfect_square number_a) ∧
  ¬(is_perfect_square number_b) ∧
  is_perfect_square number_c ∧
  ¬(is_perfect_square number_d) ∧
  ¬(is_perfect_square number_e) :=
sorry

end NUMINAMATH_CALUDE_only_c_is_perfect_square_l2490_249015


namespace NUMINAMATH_CALUDE_zoo_pictures_l2490_249003

/-- Represents the number of pictures Debby took at the zoo -/
def Z : ℕ := sorry

/-- The total number of pictures Debby initially took -/
def total_initial : ℕ := Z + 12

/-- The number of pictures Debby deleted -/
def deleted : ℕ := 14

/-- The number of pictures Debby has remaining -/
def remaining : ℕ := 22

theorem zoo_pictures : Z = 24 :=
  sorry

end NUMINAMATH_CALUDE_zoo_pictures_l2490_249003


namespace NUMINAMATH_CALUDE_four_digit_sum_mod_1000_l2490_249056

def four_digit_sum : ℕ := sorry

theorem four_digit_sum_mod_1000 : four_digit_sum % 1000 = 320 := by sorry

end NUMINAMATH_CALUDE_four_digit_sum_mod_1000_l2490_249056


namespace NUMINAMATH_CALUDE_physics_marks_l2490_249029

theorem physics_marks (P C M : ℝ) 
  (h1 : (P + C + M) / 3 = 85)
  (h2 : (P + M) / 2 = 90)
  (h3 : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l2490_249029


namespace NUMINAMATH_CALUDE_proportional_difference_theorem_l2490_249076

theorem proportional_difference_theorem (x y z k₁ k₂ : ℝ) 
  (h1 : y - z = k₁ * x)
  (h2 : z - x = k₂ * y)
  (h3 : k₁ ≠ k₂)
  (h4 : z = 3 * (x - y))
  (h5 : x ≠ 0)
  (h6 : y ≠ 0) :
  (k₁ + 3) * (k₂ + 3) = 8 := by
sorry

end NUMINAMATH_CALUDE_proportional_difference_theorem_l2490_249076


namespace NUMINAMATH_CALUDE_vector_sum_and_scalar_mult_l2490_249074

/-- Prove that the sum of the vector (3, -2, 5) and 2 times the vector (-1, 4, -3) is equal to the vector (1, 6, -1). -/
theorem vector_sum_and_scalar_mult :
  let v₁ : Fin 3 → ℝ := ![3, -2, 5]
  let v₂ : Fin 3 → ℝ := ![-1, 4, -3]
  v₁ + 2 • v₂ = ![1, 6, -1] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_and_scalar_mult_l2490_249074


namespace NUMINAMATH_CALUDE_eight_div_repeating_third_eq_24_l2490_249066

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by 0.333... --/
def result : ℚ := 8 / repeating_third

/-- Theorem stating that 8 divided by 0.333... equals 24 --/
theorem eight_div_repeating_third_eq_24 : result = 24 := by sorry

end NUMINAMATH_CALUDE_eight_div_repeating_third_eq_24_l2490_249066


namespace NUMINAMATH_CALUDE_stock_price_decrease_l2490_249004

theorem stock_price_decrease (x : ℝ) (h : x > 0) :
  let increase_factor := 1.3
  let decrease_factor := 1 - 1 / increase_factor
  x = (1 - decrease_factor) * (increase_factor * x) :=
by sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l2490_249004


namespace NUMINAMATH_CALUDE_books_borrowed_by_lunchtime_correct_books_borrowed_l2490_249043

theorem books_borrowed_by_lunchtime 
  (initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) : ℕ :=
  let books_borrowed_lunchtime := 
    initial_books + books_added - books_borrowed_evening - books_remaining
  books_borrowed_lunchtime

#check @books_borrowed_by_lunchtime

theorem correct_books_borrowed (
  initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) 
  (h1 : initial_books = 100) 
  (h2 : books_added = 40) 
  (h3 : books_borrowed_evening = 30) 
  (h4 : books_remaining = 60) :
  books_borrowed_by_lunchtime initial_books books_added books_borrowed_evening books_remaining = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_borrowed_by_lunchtime_correct_books_borrowed_l2490_249043


namespace NUMINAMATH_CALUDE_third_smallest_prime_cubed_to_fourth_l2490_249093

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem third_smallest_prime_cubed_to_fourth : (nthPrime 3) ^ 3 ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_third_smallest_prime_cubed_to_fourth_l2490_249093


namespace NUMINAMATH_CALUDE_polygon_sides_l2490_249033

theorem polygon_sides (n : ℕ) (h1 : n > 2) : 
  (140 + 145 * (n - 1) = 180 * (n - 2)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2490_249033


namespace NUMINAMATH_CALUDE_gcd_difference_is_perfect_square_l2490_249092

theorem gcd_difference_is_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * (y - x) = k * k := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_is_perfect_square_l2490_249092


namespace NUMINAMATH_CALUDE_fraction_value_l2490_249034

theorem fraction_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  (a - b) / (a + b) = -7 ∨ (a - b) / (a + b) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2490_249034


namespace NUMINAMATH_CALUDE_circle_rolling_in_triangle_l2490_249060

theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) (h4 : r = 2) :
  let k := (a + b + c - 6 * r) / (a + b + c)
  (k * a + k * b + k * c) = 220 / 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_rolling_in_triangle_l2490_249060


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2490_249031

theorem complex_equation_solution (z : ℂ) (i : ℂ) :
  i * i = -1 →
  i * z = 2 + 4 * i →
  z = 4 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2490_249031


namespace NUMINAMATH_CALUDE_smallest_number_l2490_249099

def numbers : List ℤ := [0, -2, 1, 5]

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

#check smallest_number

end NUMINAMATH_CALUDE_smallest_number_l2490_249099


namespace NUMINAMATH_CALUDE_complex_number_fourth_quadrant_l2490_249030

theorem complex_number_fourth_quadrant (z : ℂ) : 
  (z.re > 0) →  -- z is in the fourth quadrant (real part positive)
  (z.im < 0) →  -- z is in the fourth quadrant (imaginary part negative)
  (z.re + z.im = 7) →  -- sum of real and imaginary parts is 7
  (Complex.abs z = 13) →  -- magnitude of z is 13
  z = Complex.mk 12 (-5) :=  -- z equals 12 - 5i
by sorry

end NUMINAMATH_CALUDE_complex_number_fourth_quadrant_l2490_249030


namespace NUMINAMATH_CALUDE_otimes_identity_l2490_249086

-- Define the new operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^3

-- Theorem statement
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 + k^6 + 6*k^7 + k^9 := by
  sorry

end NUMINAMATH_CALUDE_otimes_identity_l2490_249086


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2490_249047

theorem max_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  ∃ M : ℝ, M = 20 ∧ ∀ x y z w : ℝ, x^2 + y^2 + z^2 + w^2 = 5 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2490_249047


namespace NUMINAMATH_CALUDE_hcf_of_numbers_l2490_249040

def number1 : ℕ := 210
def number2 : ℕ := 330
def lcm_value : ℕ := 2310

theorem hcf_of_numbers : Nat.gcd number1 number2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_numbers_l2490_249040


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2490_249050

/-- The four smallest prime numbers -/
def smallest_primes : Finset Nat := {2, 3, 5, 7}

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is divisible by all numbers in a finset if it's divisible by their product -/
def divisible_by_all (n : Nat) (s : Finset Nat) : Prop :=
  ∀ m ∈ s, n % m = 0

theorem smallest_four_digit_divisible_by_smallest_primes :
  ∃ (n : Nat), is_four_digit n ∧ 
               divisible_by_all n smallest_primes ∧
               ∀ (m : Nat), is_four_digit m ∧ divisible_by_all m smallest_primes → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2490_249050


namespace NUMINAMATH_CALUDE_wood_cutting_problem_l2490_249091

theorem wood_cutting_problem : Nat.gcd 90 72 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_problem_l2490_249091


namespace NUMINAMATH_CALUDE_framed_painting_perimeter_l2490_249065

/-- The perimeter of a framed rectangular painting -/
theorem framed_painting_perimeter
  (height : ℕ) -- Height of the painting
  (width : ℕ) -- Width of the painting
  (frame_width : ℕ) -- Width of the frame
  (h1 : height = 12)
  (h2 : width = 15)
  (h3 : frame_width = 3) :
  2 * (height + 2 * frame_width + width + 2 * frame_width) = 78 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_perimeter_l2490_249065


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2490_249041

theorem solution_set_of_inequality (x : ℝ) :
  (((3 * x + 1) / (1 - 2 * x) ≥ 0) ↔ (-1/3 ≤ x ∧ x < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2490_249041


namespace NUMINAMATH_CALUDE_factorization_equality_l2490_249017

theorem factorization_equality (y : ℝ) : 49 - 16 * y^2 + 8 * y = (7 - 4 * y) * (7 + 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2490_249017


namespace NUMINAMATH_CALUDE_cubic_is_closed_log_not_closed_sqrt_closed_condition_l2490_249022

-- Define a closed function
def is_closed_function (f : ℝ → ℝ) : Prop :=
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y))

-- Theorem for the cubic function
theorem cubic_is_closed : is_closed_function (fun x => -x^3) :=
sorry

-- Theorem for the logarithmic function
theorem log_not_closed : ¬ is_closed_function (fun x => 2*x - Real.log x) :=
sorry

-- Theorem for the square root function
theorem sqrt_closed_condition (k : ℝ) : 
  is_closed_function (fun x => k + Real.sqrt (x + 2)) ↔ -9/4 < k ∧ k ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_is_closed_log_not_closed_sqrt_closed_condition_l2490_249022


namespace NUMINAMATH_CALUDE_inequality_proof_l2490_249028

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2490_249028


namespace NUMINAMATH_CALUDE_not_q_is_false_l2490_249038

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_q_is_false_l2490_249038


namespace NUMINAMATH_CALUDE_sample_is_extracurricular_homework_l2490_249010

/-- Represents a student in the survey -/
structure Student where
  id : Nat
  hasExtracurricularHomework : Bool

/-- Represents the survey conducted by the middle school -/
structure Survey where
  totalPopulation : Finset Student
  selectedSample : Finset Student
  sampleSize : Nat

/-- Definition of a valid survey -/
def validSurvey (s : Survey) : Prop :=
  s.totalPopulation.card = 1800 ∧
  s.selectedSample.card = 300 ∧
  s.selectedSample ⊆ s.totalPopulation ∧
  s.sampleSize = s.selectedSample.card

/-- Definition of the sample in the survey -/
def sampleDefinition (s : Survey) : Finset Student :=
  s.selectedSample.filter (λ student => student.hasExtracurricularHomework)

/-- Theorem stating that the sample is the extracurricular homework of 300 students -/
theorem sample_is_extracurricular_homework (s : Survey) (h : validSurvey s) :
  sampleDefinition s = s.selectedSample :=
sorry


end NUMINAMATH_CALUDE_sample_is_extracurricular_homework_l2490_249010


namespace NUMINAMATH_CALUDE_no_negative_exponents_l2490_249009

theorem no_negative_exponents (a b c d : ℤ) 
  (h : (4 : ℝ)^a + (4 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d + 1) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l2490_249009


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2490_249097

-- Define the quadratic polynomial
def p (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value (a b c : ℝ) (ha : a > 0) (h1 : p a b c 1 = 4) (h2 : p a b c 2 = 15) :
  (∃ (x : ℝ), ∀ (y : ℝ), p a b c y ≤ p a b c x) ∧
  (∀ (x : ℝ), p a b c x ≤ 4) ∧
  p a b c 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2490_249097


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2490_249021

-- Define the variables x and y as real numbers
variable (x y : ℝ)

-- State the theorem
theorem express_y_in_terms_of_x (h : 2 * x + y = 1) : y = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2490_249021


namespace NUMINAMATH_CALUDE_calories_consumed_l2490_249084

/-- Given a package of candy with 3 servings of 120 calories each,
    prove that eating half the package results in consuming 180 calories. -/
theorem calories_consumed (servings : ℕ) (calories_per_serving : ℕ) (portion_eaten : ℚ) : 
  servings = 3 → 
  calories_per_serving = 120 → 
  portion_eaten = 1/2 →
  (↑servings * ↑calories_per_serving : ℚ) * portion_eaten = 180 := by
sorry

end NUMINAMATH_CALUDE_calories_consumed_l2490_249084


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2490_249052

-- Define the universal set U
def U : Set ℕ := {x | 1 < x ∧ x < 5}

-- Define set A
def A : Set ℕ := {2, 3}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2490_249052


namespace NUMINAMATH_CALUDE_line_m_equation_l2490_249014

-- Define the plane
def Plane := ℝ × ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in the plane
def Point := Plane

-- Define the given lines
def ℓ : Line := { a := 2, b := -5, c := 0 }
def m : Line := { a := 5, b := 2, c := 0 }

-- Define the given points
def Q : Point := (3, -2)
def Q'' : Point := (-2, 3)

-- Define the reflection operation
def reflect (p : Point) (L : Line) : Point := sorry

-- State the theorem
theorem line_m_equation :
  ∃ (Q' : Point),
    reflect Q m = Q' ∧
    reflect Q' ℓ = Q'' ∧
    m.a = 5 ∧ m.b = 2 ∧ m.c = 0 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l2490_249014


namespace NUMINAMATH_CALUDE_total_snake_owners_l2490_249071

/- Define the total number of pet owners -/
def total_pet_owners : ℕ := 120

/- Define the number of people owning specific combinations of pets -/
def only_dogs : ℕ := 25
def only_cats : ℕ := 18
def only_birds : ℕ := 12
def only_snakes : ℕ := 15
def only_hamsters : ℕ := 7
def cats_and_dogs : ℕ := 8
def dogs_and_birds : ℕ := 5
def cats_and_birds : ℕ := 6
def cats_and_snakes : ℕ := 7
def dogs_and_snakes : ℕ := 10
def dogs_and_hamsters : ℕ := 4
def cats_and_hamsters : ℕ := 3
def birds_and_hamsters : ℕ := 5
def birds_and_snakes : ℕ := 2
def snakes_and_hamsters : ℕ := 3
def cats_dogs_birds : ℕ := 3
def cats_dogs_snakes : ℕ := 4
def cats_snakes_hamsters : ℕ := 2
def all_pets : ℕ := 1

/- Theorem stating the total number of snake owners -/
theorem total_snake_owners : 
  only_snakes + cats_and_snakes + dogs_and_snakes + birds_and_snakes + 
  snakes_and_hamsters + cats_dogs_snakes + cats_snakes_hamsters + all_pets = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_l2490_249071


namespace NUMINAMATH_CALUDE_alpha_value_l2490_249032

-- Define the triangle and point S
variable (P Q R S : Point)

-- Define the angles
variable (α β γ δ : ℝ)

-- Define the conditions
variable (triangle_PQR : Triangle P Q R)
variable (S_interior : InteriorPoint S triangle_PQR)
variable (QSP_bisected : AngleBisector S Q (Angle P S Q))
variable (delta_exterior : ExteriorAngle Q triangle_PQR δ)

-- Given angle values
variable (beta_value : β = 100)
variable (gamma_value : γ = 30)
variable (delta_value : δ = 150)

-- Theorem statement
theorem alpha_value : α = 215 := by sorry

end NUMINAMATH_CALUDE_alpha_value_l2490_249032


namespace NUMINAMATH_CALUDE_addition_subtraction_problem_l2490_249068

theorem addition_subtraction_problem : (0.45 + 52.7) - 0.25 = 52.9 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_problem_l2490_249068


namespace NUMINAMATH_CALUDE_platform_length_l2490_249011

/-- Calculates the length of a platform given a train's speed and crossing times. -/
theorem platform_length (train_speed_kmph : ℝ) (time_cross_platform : ℝ) (time_cross_man : ℝ) : 
  train_speed_kmph = 72 ∧ 
  time_cross_platform = 30 ∧ 
  time_cross_man = 16 → 
  (train_speed_kmph * 1000 / 3600 * time_cross_platform) - 
  (train_speed_kmph * 1000 / 3600 * time_cross_man) = 280 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2490_249011


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2490_249064

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2490_249064


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2490_249083

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2490_249083


namespace NUMINAMATH_CALUDE_ratio_q_p_l2490_249080

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 4
def drawn_slips : ℕ := 4

def p : ℚ := (distinct_numbers * 1) / Nat.choose total_slips drawn_slips
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 162 := by
  sorry

end NUMINAMATH_CALUDE_ratio_q_p_l2490_249080


namespace NUMINAMATH_CALUDE_days_A_worked_alone_l2490_249046

/-- Represents the number of days it takes for A and B to finish the work together -/
def total_days_together : ℝ := 40

/-- Represents the number of days it takes for A to finish the work alone -/
def total_days_A : ℝ := 28

/-- Represents the number of days A and B worked together before B left -/
def days_worked_together : ℝ := 10

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

theorem days_A_worked_alone :
  let remaining_work := total_work - (days_worked_together / total_days_together)
  let days_A_alone := remaining_work * total_days_A
  days_A_alone = 21 := by
sorry

end NUMINAMATH_CALUDE_days_A_worked_alone_l2490_249046


namespace NUMINAMATH_CALUDE_smallest_natural_numbers_satisfying_equation_l2490_249037

theorem smallest_natural_numbers_satisfying_equation :
  ∃ (A B : ℕ+),
    (360 : ℝ) / ((A : ℝ) * (A : ℝ) * (A : ℝ) / (B : ℝ)) = 5 ∧
    ∀ (A' B' : ℕ+),
      (360 : ℝ) / ((A' : ℝ) * (A' : ℝ) * (A' : ℝ) / (B' : ℝ)) = 5 →
      (A ≤ A' ∧ B ≤ B') ∧
    A = 6 ∧
    B = 3 ∧
    A + B = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_numbers_satisfying_equation_l2490_249037


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l2490_249020

theorem sqrt_sum_difference : Real.sqrt 50 + Real.sqrt 32 - Real.sqrt 2 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l2490_249020
