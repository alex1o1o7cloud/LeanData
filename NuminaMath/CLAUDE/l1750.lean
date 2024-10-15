import Mathlib

namespace NUMINAMATH_CALUDE_total_pumpkin_pies_l1750_175097

theorem total_pumpkin_pies (pinky helen emily jake : ℕ)
  (h1 : pinky = 147)
  (h2 : helen = 56)
  (h3 : emily = 89)
  (h4 : jake = 122) :
  pinky + helen + emily + jake = 414 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkin_pies_l1750_175097


namespace NUMINAMATH_CALUDE_percent_of_y_l1750_175057

theorem percent_of_y (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) / y = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1750_175057


namespace NUMINAMATH_CALUDE_dogs_per_box_l1750_175066

theorem dogs_per_box (total_boxes : ℕ) (total_dogs : ℕ) (dogs_per_box : ℕ) :
  total_boxes = 7 →
  total_dogs = 28 →
  total_dogs = total_boxes * dogs_per_box →
  dogs_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_dogs_per_box_l1750_175066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l1750_175065

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The 5th term is the geometric mean of the 3rd and 11th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 3 * a 11

theorem arithmetic_sequence_with_geometric_mean 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_mean_condition a) : 
  a 1 = -1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l1750_175065


namespace NUMINAMATH_CALUDE_base_prime_repr_225_l1750_175034

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Theorem: The base prime representation of 225 is [2, 2, 0] -/
theorem base_prime_repr_225 : 
  base_prime_repr 225 = [2, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base_prime_repr_225_l1750_175034


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1750_175083

theorem simplify_trig_expression (α : ℝ) :
  (1 - Real.cos (2 * α) + Real.sin (2 * α)) / (1 + Real.cos (2 * α) + Real.sin (2 * α)) = Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1750_175083


namespace NUMINAMATH_CALUDE_leah_chocolates_l1750_175099

theorem leah_chocolates (leah_chocolates max_chocolates : ℕ) : 
  leah_chocolates = max_chocolates + 8 →
  max_chocolates = leah_chocolates / 3 →
  leah_chocolates = 12 := by
sorry

end NUMINAMATH_CALUDE_leah_chocolates_l1750_175099


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1750_175068

/-- The equation of a line perpendicular to 2x+y-5=0 and passing through (2,3) is x-2y+4=0 -/
theorem perpendicular_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m : ℝ), (2 : ℝ) * x + y - 5 = 0 ↔ y = -2 * x + m) →
  (∃ (k : ℝ), k * (x - 2) + 3 = y ∧ k * 2 = -1) →
  x - 2 * y + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1750_175068


namespace NUMINAMATH_CALUDE_matrix_product_equals_C_l1750_175094

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, -1, 2; 1, 0, 5; 4, 1, -2]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![2, -3, 4; -1, 5, -2; 0, 2, 7]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![7, -10, 28; 2, 7, 39; 7, -11, 0]

theorem matrix_product_equals_C : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_equals_C_l1750_175094


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l1750_175084

def sequence_sum (n : ℕ) : ℤ := n^2 - 1

def sequence_term (n : ℕ) : ℤ :=
  if n = 1 then 0
  else 2 * n - 2

theorem sum_of_specific_terms : 
  sequence_term 1 + sequence_term 3 + sequence_term 5 + sequence_term 7 + sequence_term 9 = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l1750_175084


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1750_175000

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_10 : a + b + c + d = 10) :
  (1/a + 9/b + 25/c + 49/d) ≥ 25.6 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 0 < d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 10 ∧
    1/a₀ + 9/b₀ + 25/c₀ + 49/d₀ = 25.6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1750_175000


namespace NUMINAMATH_CALUDE_arccos_cos_three_l1750_175048

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l1750_175048


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l1750_175004

/-- Given the following conditions about wildflowers:
  - Total wildflowers picked: 44
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
Prove that there are 4 more flowers containing red than containing white. -/
theorem wildflower_color_difference 
  (total : ℕ) 
  (yellow_white : ℕ) 
  (red_yellow : ℕ) 
  (red_white : ℕ) 
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l1750_175004


namespace NUMINAMATH_CALUDE_find_x_value_l1750_175016

theorem find_x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((3 * x) / 7) = x) : x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l1750_175016


namespace NUMINAMATH_CALUDE_max_ab_line_tangent_circle_l1750_175073

/-- The maximum value of ab when a line is tangent to a circle -/
theorem max_ab_line_tangent_circle (a b : ℝ) : 
  -- Line equation: x + 2y = 0
  -- Circle equation: (x-a)² + (y-b)² = 5
  -- Line is tangent to circle
  (∃ x y : ℝ, x + 2*y = 0 ∧ (x-a)^2 + (y-b)^2 = 5 ∧ 
    ∀ x' y' : ℝ, x' + 2*y' = 0 → (x'-a)^2 + (y'-b)^2 ≥ 5) →
  -- Center of circle is above the line
  a + 2*b > 0 →
  -- The maximum value of ab is 25/8
  a * b ≤ 25/8 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_line_tangent_circle_l1750_175073


namespace NUMINAMATH_CALUDE_smaller_circles_radius_l1750_175029

/-- Configuration of circles -/
structure CircleConfiguration where
  centralRadius : ℝ
  smallerRadius : ℝ
  numSmallerCircles : ℕ

/-- Defines a valid configuration of circles -/
def isValidConfiguration (config : CircleConfiguration) : Prop :=
  config.centralRadius = 1 ∧
  config.numSmallerCircles = 6 ∧
  -- Each smaller circle touches two others and the central circle
  -- (This condition is implicit in the geometry of the problem)
  True

/-- Theorem stating the radius of smaller circles in the given configuration -/
theorem smaller_circles_radius (config : CircleConfiguration)
  (h : isValidConfiguration config) :
  config.smallerRadius = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circles_radius_l1750_175029


namespace NUMINAMATH_CALUDE_x_value_proof_l1750_175024

theorem x_value_proof (x : ℕ) : 
  (Nat.lcm x 18 - Nat.gcd x 18 = 120) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1750_175024


namespace NUMINAMATH_CALUDE_rectangle_divided_by_line_l1750_175006

/-- 
Given a rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2),
if a line passing through the origin (0, 0) divides the rectangle into two identical quadrilaterals
and has a slope of 1/3, then x = 5.
-/
theorem rectangle_divided_by_line (x : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    -- l is a line passing through the origin
    (0, 0) ∈ l ∧
    -- l divides the rectangle into two identical quadrilaterals
    (∃ m : ℝ × ℝ, m ∈ l ∧ m.1 = (1 + x) / 2 ∧ m.2 = 1) ∧
    -- The slope of l is 1/3
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (q.2 - p.2) / (q.1 - p.1) = 1/3)) →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_divided_by_line_l1750_175006


namespace NUMINAMATH_CALUDE_total_green_marbles_l1750_175085

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have -/
def total_green : ℕ := sara_green + tom_green

theorem total_green_marbles : total_green = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_green_marbles_l1750_175085


namespace NUMINAMATH_CALUDE_basketball_free_throw_percentage_l1750_175019

theorem basketball_free_throw_percentage 
  (p : ℝ) 
  (h : 0 ≤ p ∧ p ≤ 1) 
  (h_prob : (1 - p)^2 + 2*p*(1 - p) = 16/25) : 
  p = 3/5 := by sorry

end NUMINAMATH_CALUDE_basketball_free_throw_percentage_l1750_175019


namespace NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l1750_175059

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_multiple_of_1_to_10 :
  ∃ (n : ℕ), n > 0 ∧ is_multiple_of_all n ∧ ∀ m : ℕ, m > 0 → is_multiple_of_all m → n ≤ m :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_1_to_10_l1750_175059


namespace NUMINAMATH_CALUDE_propositions_truth_l1750_175096

theorem propositions_truth :
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a*b < 0) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ ¬(a^2 < a*b ∧ a*b < b^2)) ∧
  (∃ c a b : ℝ, c > a ∧ a > b ∧ b > 0 ∧ ¬(a/(c-a) < b/(c-b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a/b > (a+c)/(b+c)) :=
by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l1750_175096


namespace NUMINAMATH_CALUDE_cube_sum_in_interval_l1750_175036

theorem cube_sum_in_interval (n : ℕ) : ∃ k x y : ℕ,
  (n : ℝ) - 4 * Real.sqrt (n : ℝ) ≤ k ∧
  k ≤ (n : ℝ) + 4 * Real.sqrt (n : ℝ) ∧
  k = x^3 + y^3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_in_interval_l1750_175036


namespace NUMINAMATH_CALUDE_marbles_selection_theorem_l1750_175028

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def ordinary_marbles : ℕ := total_marbles - special_marbles

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marbles_selection_theorem :
  (choose_marbles special_marbles 2 * choose_marbles ordinary_marbles 3) +
  (choose_marbles special_marbles 3 * choose_marbles ordinary_marbles 2) +
  (choose_marbles special_marbles 4 * choose_marbles ordinary_marbles 1) +
  (choose_marbles special_marbles 5 * choose_marbles ordinary_marbles 0) = 2121 :=
by sorry

end NUMINAMATH_CALUDE_marbles_selection_theorem_l1750_175028


namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l1750_175042

theorem fraction_equals_decimal : (8 : ℚ) / (4 * 25) = 0.08 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l1750_175042


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l1750_175082

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
structure GeometricProgression where
  firstTerm : ℚ
  commonRatio : ℚ

/-- The n-th term of a geometric progression. -/
def nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.firstTerm * gp.commonRatio ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    nthTerm gp 2 = 37 + 1/3 ∧
    nthTerm gp 6 = 2 + 1/3 ∧
    gp.firstTerm = 224/3 ∧
    gp.commonRatio = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l1750_175082


namespace NUMINAMATH_CALUDE_rectangle_area_l1750_175037

theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l = 3 * w ∧ x^2 = l^2 + w^2 ∧ l * w = (3 * x^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1750_175037


namespace NUMINAMATH_CALUDE_worker_distance_at_explosion_l1750_175053

/-- The time in seconds when the bomb explodes -/
def bomb_time : ℝ := 45

/-- The speed of the worker in yards per second -/
def worker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- The distance run by the worker after t seconds, in feet -/
def worker_distance (t : ℝ) : ℝ := worker_speed * yards_to_feet * t

/-- The distance traveled by sound after the bomb explodes, in feet -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - bomb_time)

/-- The time when the worker hears the explosion -/
noncomputable def explosion_time : ℝ := 
  (sound_speed * bomb_time) / (sound_speed - worker_speed * yards_to_feet)

/-- The theorem stating that the worker runs approximately 275 yards when he hears the explosion -/
theorem worker_distance_at_explosion : 
  ∃ ε > 0, abs (worker_distance explosion_time / yards_to_feet - 275) < ε :=
sorry

end NUMINAMATH_CALUDE_worker_distance_at_explosion_l1750_175053


namespace NUMINAMATH_CALUDE_reflection_of_point_2_5_l1750_175063

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of point (2, 5) across the x-axis is (2, -5) -/
theorem reflection_of_point_2_5 :
  let p := Point.mk 2 5
  reflectAcrossXAxis p = Point.mk 2 (-5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_2_5_l1750_175063


namespace NUMINAMATH_CALUDE_no_complete_non_self_intersecting_path_l1750_175072

/-- Represents the surface of a Rubik's cube -/
structure RubiksCubeSurface where
  squares : Nat
  diagonals : Nat
  vertices : Nat

/-- The surface of a standard Rubik's cube -/
def standardRubiksCube : RubiksCubeSurface :=
  { squares := 54
  , diagonals := 54
  , vertices := 56 }

/-- A path on the surface of a Rubik's cube -/
structure DiagonalPath (surface : RubiksCubeSurface) where
  length : Nat
  is_non_self_intersecting : Bool

/-- Theorem stating the impossibility of creating a non-self-intersecting path
    using all diagonals on the surface of a standard Rubik's cube -/
theorem no_complete_non_self_intersecting_path 
  (surface : RubiksCubeSurface) 
  (h_surface : surface = standardRubiksCube) :
  ¬∃ (path : DiagonalPath surface), 
    path.length = surface.diagonals ∧ 
    path.is_non_self_intersecting = true := by
  sorry


end NUMINAMATH_CALUDE_no_complete_non_self_intersecting_path_l1750_175072


namespace NUMINAMATH_CALUDE_cos_36_degrees_l1750_175032

theorem cos_36_degrees (h : Real.sin (108 * π / 180) = 3 * Real.sin (36 * π / 180) - 4 * (Real.sin (36 * π / 180))^3) :
  Real.cos (36 * π / 180) = (1 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l1750_175032


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1750_175088

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 10.0 → 
  7 * s = 5 * l → 
  s + l = 24.0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1750_175088


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_of_20_factorial_l1750_175058

-- Define n as 20!
def n : ℕ := (List.range 20).foldl (· * ·) 1

-- Define the property of k being the greatest integer for which 2^k divides n
def is_greatest_power_of_two_factor (k : ℕ) : Prop :=
  2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k

-- Theorem statement
theorem greatest_power_of_two_factor_of_20_factorial :
  is_greatest_power_of_two_factor 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_of_20_factorial_l1750_175058


namespace NUMINAMATH_CALUDE_chris_pears_equal_lily_apples_l1750_175008

/-- Represents the number of fruits in the box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ
  apples_twice_pears : apples = 2 * pears

/-- Represents the distribution of fruits between Chris and Lily -/
structure FruitDistribution where
  box : FruitBox
  chris_apples : ℕ
  chris_pears : ℕ
  lily_apples : ℕ
  lily_pears : ℕ
  total_distributed : chris_apples + chris_pears + lily_apples + lily_pears = box.apples + box.pears
  chris_twice_lily : chris_apples + chris_pears = 2 * (lily_apples + lily_pears)

/-- Theorem stating that Chris took as many pears as Lily took apples -/
theorem chris_pears_equal_lily_apples (dist : FruitDistribution) : 
  dist.chris_pears = dist.lily_apples := by sorry

end NUMINAMATH_CALUDE_chris_pears_equal_lily_apples_l1750_175008


namespace NUMINAMATH_CALUDE_shaded_percentage_is_75_percent_l1750_175017

/-- Represents a square grid composed of smaller squares -/
structure Grid where
  side_length : ℕ
  small_squares : ℕ
  shaded_squares : ℕ

/-- Calculates the percentage of shaded squares in the grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (g.shaded_squares : ℚ) / (g.small_squares : ℚ) * 100

/-- Theorem stating that the percentage of shaded squares is 75% -/
theorem shaded_percentage_is_75_percent (g : Grid) 
  (h1 : g.side_length = 8)
  (h2 : g.small_squares = g.side_length * g.side_length)
  (h3 : g.shaded_squares = 48) : 
  shaded_percentage g = 75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_is_75_percent_l1750_175017


namespace NUMINAMATH_CALUDE_constant_value_l1750_175092

theorem constant_value (t : ℝ) (constant : ℝ) :
  let x := constant - 2 * t
  let y := 2 * t - 2
  (t = 0.75 → x = y) →
  constant = 1 := by sorry

end NUMINAMATH_CALUDE_constant_value_l1750_175092


namespace NUMINAMATH_CALUDE_red_card_events_l1750_175095

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, ¬(P d ∧ Q d)

-- Define opposite events
def opposite_events (P Q : Distribution → Prop) : Prop :=
  ∀ d : Distribution, P d ↔ ¬Q d

-- Theorem statement
theorem red_card_events :
  (mutually_exclusive A_gets_red B_gets_red) ∧
  ¬(opposite_events A_gets_red B_gets_red) := by
  sorry

end NUMINAMATH_CALUDE_red_card_events_l1750_175095


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1750_175021

/-- The ages of two people A and B satisfy certain conditions. -/
structure AgeRatio where
  a : ℕ  -- Current age of A
  b : ℕ  -- Current age of B
  past_future_ratio : a - 4 = b + 4  -- Ratio 1:1 for A's past and B's future
  future_past_ratio : a + 4 = 5 * (b - 4)  -- Ratio 5:1 for A's future and B's past

/-- The ratio of current ages of A and B is 2:1 -/
theorem age_ratio_is_two_to_one (ages : AgeRatio) : 
  2 * ages.b = ages.a := by sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1750_175021


namespace NUMINAMATH_CALUDE_trig_expression_max_value_trig_expression_max_achievable_l1750_175023

theorem trig_expression_max_value (A B C : Real) :
  (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 ≤ 1 :=
sorry

theorem trig_expression_max_achievable :
  ∃ (A B C : Real), (Real.sin A)^2 * (Real.cos B)^2 + (Real.sin B)^2 * (Real.cos C)^2 + (Real.sin C)^2 * (Real.cos A)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trig_expression_max_value_trig_expression_max_achievable_l1750_175023


namespace NUMINAMATH_CALUDE_sequence_properties_l1750_175011

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define a property for isolated points in a graph
def HasIsolatedPoints (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Theorem statement
theorem sequence_properties :
  (∃ (s : Sequence), True) ∧
  (∀ (s : Sequence), HasIsolatedPoints s) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1750_175011


namespace NUMINAMATH_CALUDE_triangle_area_l1750_175044

theorem triangle_area (A B C : Real) (a b c : Real) :
  (b = c * (2 * Real.sin A + Real.cos A)) →
  (a = Real.sqrt 2) →
  (B = 3 * Real.pi / 4) →
  (∃ (S : Real), S = (1 / 2) * a * c * Real.sin B ∧ S = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1750_175044


namespace NUMINAMATH_CALUDE_intersection_range_l1750_175009

/-- The curve y = 1 + √(4 - x²) intersects with the line y = k(x + 2) + 5 at two points
    if and only if k is in the range [-1, -3/4) --/
theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (1 + Real.sqrt (4 - x₁^2) = k * (x₁ + 2) + 5) ∧
    (1 + Real.sqrt (4 - x₂^2) = k * (x₂ + 2) + 5)) ↔ 
  (k ≥ -1 ∧ k < -3/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1750_175009


namespace NUMINAMATH_CALUDE_find_subtracted_number_l1750_175005

theorem find_subtracted_number (x N : ℝ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 := by
  sorry

end NUMINAMATH_CALUDE_find_subtracted_number_l1750_175005


namespace NUMINAMATH_CALUDE_cube_diagonal_pairs_60_degrees_l1750_175055

/-- A regular hexahedron (cube) -/
structure Cube where
  /-- Number of faces in a cube -/
  faces : ℕ
  /-- Number of diagonals per face -/
  diagonals_per_face : ℕ
  /-- Total number of face diagonals -/
  total_diagonals : ℕ
  /-- Total number of possible diagonal pairs -/
  total_pairs : ℕ
  /-- Number of diagonal pairs that don't form a 60° angle -/
  non_60_pairs : ℕ

/-- The number of pairs of face diagonals in a cube that form a 60° angle -/
def pairs_forming_60_degrees (c : Cube) : ℕ :=
  c.total_pairs - c.non_60_pairs

/-- Theorem stating that in a regular hexahedron (cube), 
    the number of pairs of face diagonals that form a 60° angle is 48 -/
theorem cube_diagonal_pairs_60_degrees (c : Cube) 
  (h1 : c.faces = 6)
  (h2 : c.diagonals_per_face = 2)
  (h3 : c.total_diagonals = 12)
  (h4 : c.total_pairs = 66)
  (h5 : c.non_60_pairs = 18) :
  pairs_forming_60_degrees c = 48 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_pairs_60_degrees_l1750_175055


namespace NUMINAMATH_CALUDE_division_problem_l1750_175014

theorem division_problem (dividend divisor : ℕ) : 
  (dividend / divisor = 3) → 
  (dividend % divisor = 20) → 
  (dividend + divisor + 3 + 20 = 303) → 
  (divisor = 65 ∧ dividend = 215) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1750_175014


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1750_175015

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x - 1
def q (x : ℝ) : ℝ := 2 * x^3 - x^2 + 5 * x - 4

theorem coefficient_of_x_squared :
  (∃ a b c d e : ℝ, ∀ x, p x * q x = a * x^5 + b * x^4 + c * x^3 - 31 * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1750_175015


namespace NUMINAMATH_CALUDE_mixture_ratio_proof_l1750_175003

theorem mixture_ratio_proof (p q : ℝ) : 
  p + q = 35 →
  p / (q + 13) = 5 / 7 →
  p / q = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_proof_l1750_175003


namespace NUMINAMATH_CALUDE_passing_mark_is_160_l1750_175070

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total : ℕ
  passing : ℕ

/-- The condition that a candidate scoring 40% fails by 40 marks -/
def condition1 (e : Exam) : Prop :=
  (40 * e.total) / 100 = e.passing - 40

/-- The condition that a candidate scoring 60% passes by 20 marks -/
def condition2 (e : Exam) : Prop :=
  (60 * e.total) / 100 = e.passing + 20

/-- Theorem stating that given the conditions, the passing mark is 160 -/
theorem passing_mark_is_160 (e : Exam) 
  (h1 : condition1 e) (h2 : condition2 e) : e.passing = 160 := by
  sorry


end NUMINAMATH_CALUDE_passing_mark_is_160_l1750_175070


namespace NUMINAMATH_CALUDE_annie_figurines_count_l1750_175076

def number_of_tvs : ℕ := 5
def cost_per_tv : ℕ := 50
def total_spent : ℕ := 260
def cost_per_figurine : ℕ := 1

theorem annie_figurines_count :
  (total_spent - number_of_tvs * cost_per_tv) / cost_per_figurine = 10 := by
  sorry

end NUMINAMATH_CALUDE_annie_figurines_count_l1750_175076


namespace NUMINAMATH_CALUDE_jill_trips_to_fill_tank_l1750_175038

/-- Represents the water fetching problem with Jack and Jill -/
def WaterFetchingProblem (tank_capacity : ℕ) (bucket_capacity : ℕ) (jack_buckets : ℕ) 
  (jill_buckets : ℕ) (jack_trips : ℕ) (jill_trips : ℕ) (leak_rate : ℕ) : Prop :=
  ∃ (jill_total_trips : ℕ),
    -- The tank capacity is 600 gallons
    tank_capacity = 600 ∧
    -- Each bucket holds 5 gallons
    bucket_capacity = 5 ∧
    -- Jack carries 2 buckets per trip
    jack_buckets = 2 ∧
    -- Jill carries 1 bucket per trip
    jill_buckets = 1 ∧
    -- Jack makes 3 trips for every 2 trips Jill makes
    jack_trips = 3 ∧
    jill_trips = 2 ∧
    -- The tank leaks 2 gallons every time both return
    leak_rate = 2 ∧
    -- The number of trips Jill makes is 20
    jill_total_trips = 20 ∧
    -- The tank is filled after Jill's trips
    jill_total_trips * jill_trips * (jack_buckets * bucket_capacity * jack_trips + 
      jill_buckets * bucket_capacity * jill_trips - leak_rate) / (jack_trips + jill_trips) ≥ tank_capacity

/-- Theorem stating that given the conditions, Jill will make 20 trips before the tank is filled -/
theorem jill_trips_to_fill_tank : 
  WaterFetchingProblem 600 5 2 1 3 2 2 := by sorry

end NUMINAMATH_CALUDE_jill_trips_to_fill_tank_l1750_175038


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l1750_175089

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  y_eq_x_plus_3 : y = x + 3
  product_eq_2541 : x * y * z = 2541

/-- The sum of the dimensions of a box. -/
def sum_dimensions (d : BoxDimensions) : ℕ := d.x + d.y + d.z

/-- Theorem stating the minimum sum of dimensions for the given conditions. -/
theorem min_sum_dimensions :
  ∀ d : BoxDimensions, sum_dimensions d ≥ 38 := by sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l1750_175089


namespace NUMINAMATH_CALUDE_zero_in_interval_implies_alpha_range_l1750_175087

theorem zero_in_interval_implies_alpha_range (α : ℝ) :
  (∃ x ∈ Set.Icc 0 1, x^2 + 2*α*x + 1 = 0) → α ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_implies_alpha_range_l1750_175087


namespace NUMINAMATH_CALUDE_problem_statement_l1750_175078

theorem problem_statement : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1750_175078


namespace NUMINAMATH_CALUDE_quadratic_residue_prime_power_l1750_175020

theorem quadratic_residue_prime_power (p : Nat) (a : Nat) (k : Nat) :
  Nat.Prime p →
  Odd p →
  (∃ y : Nat, y^2 ≡ a [MOD p]) →
  ∃ z : Nat, z^2 ≡ a [MOD p^k] :=
sorry

end NUMINAMATH_CALUDE_quadratic_residue_prime_power_l1750_175020


namespace NUMINAMATH_CALUDE_printer_time_calculation_l1750_175049

/-- Given a printer that prints 23 pages per minute, prove that it takes 15 minutes to print 345 pages. -/
theorem printer_time_calculation (print_rate : ℕ) (total_pages : ℕ) (time : ℕ) : 
  print_rate = 23 → total_pages = 345 → time = total_pages / print_rate → time = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_calculation_l1750_175049


namespace NUMINAMATH_CALUDE_two_pipes_fill_time_l1750_175012

/-- Given two pipes filling a tank, where one pipe is 3 times as fast as the other,
    and the slower pipe can fill the tank in 160 minutes,
    prove that both pipes together can fill the tank in 40 minutes. -/
theorem two_pipes_fill_time (slow_pipe_time : ℝ) (fast_pipe_time : ℝ) : 
  slow_pipe_time = 160 →
  fast_pipe_time = slow_pipe_time / 3 →
  (1 / fast_pipe_time + 1 / slow_pipe_time)⁻¹ = 40 :=
by sorry

end NUMINAMATH_CALUDE_two_pipes_fill_time_l1750_175012


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l1750_175051

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l1750_175051


namespace NUMINAMATH_CALUDE_jason_attended_twelve_games_l1750_175033

def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (missed : ℕ) : ℕ :=
  planned_this_month + planned_last_month - missed

theorem jason_attended_twelve_games :
  games_attended 11 17 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jason_attended_twelve_games_l1750_175033


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1750_175041

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1750_175041


namespace NUMINAMATH_CALUDE_square_diagonals_perpendicular_l1750_175052

structure Rhombus where
  diagonals_perpendicular : Bool

structure Square extends Rhombus

theorem square_diagonals_perpendicular (rhombus_property : Rhombus → Bool)
    (square_is_rhombus : Square → Rhombus)
    (h1 : ∀ r : Rhombus, rhombus_property r = r.diagonals_perpendicular)
    (h2 : ∀ s : Square, rhombus_property (square_is_rhombus s) = true) :
  ∀ s : Square, s.diagonals_perpendicular = true := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_perpendicular_l1750_175052


namespace NUMINAMATH_CALUDE_number_equation_l1750_175027

theorem number_equation : ∃ x : ℚ, (5 + 4/9) / 7 = 5 * x ∧ x = 49/315 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1750_175027


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1750_175025

/-- Given a geometric sequence {a_n} with sum S_n, prove that if S_3 = 39 and a_2 = 9,
    then the common ratio q satisfies q^2 - (10/3)q + 1 = 0 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : S 3 = 39) 
  (h2 : a 2 = 9) 
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) 
  (h4 : ∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) 
  : q^2 - (10/3) * q + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1750_175025


namespace NUMINAMATH_CALUDE_probability_selecting_A_and_B_l1750_175064

theorem probability_selecting_A_and_B : 
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let total_combinations := Nat.choose total_students selected_students
  let favorable_combinations := Nat.choose (total_students - 2) (selected_students - 2)
  (favorable_combinations : ℚ) / total_combinations = 3 / 10 :=
sorry

end NUMINAMATH_CALUDE_probability_selecting_A_and_B_l1750_175064


namespace NUMINAMATH_CALUDE_circle_radius_l1750_175039

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 1 = 0) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1750_175039


namespace NUMINAMATH_CALUDE_initial_papers_count_l1750_175002

/-- The number of papers Charles initially bought -/
def initial_papers : ℕ := sorry

/-- The number of pictures Charles drew today -/
def pictures_today : ℕ := 6

/-- The number of pictures Charles drew before work yesterday -/
def pictures_before_work : ℕ := 6

/-- The number of pictures Charles drew after work yesterday -/
def pictures_after_work : ℕ := 6

/-- The number of papers Charles has left -/
def papers_left : ℕ := 2

/-- Theorem stating that the initial number of papers is equal to the sum of papers used for pictures and papers left -/
theorem initial_papers_count : 
  initial_papers = pictures_today + pictures_before_work + pictures_after_work + papers_left :=
by sorry

end NUMINAMATH_CALUDE_initial_papers_count_l1750_175002


namespace NUMINAMATH_CALUDE_max_b_in_box_l1750_175090

/-- Given a rectangular box with volume 360 cubic units and integer dimensions a, b, and c 
    where a > b > c > 2, the maximum value of b is 10. -/
theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  a > b →
  b > c →
  c > 2 →
  b ≤ 10 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ a' > b' ∧ b' > c' ∧ c' > 2 ∧ b' = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l1750_175090


namespace NUMINAMATH_CALUDE_simplify_expression_l1750_175079

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1750_175079


namespace NUMINAMATH_CALUDE_car_travel_time_l1750_175093

theorem car_travel_time (speed_x speed_y distance_after_y : ℝ) 
  (hx : speed_x = 35)
  (hy : speed_y = 70)
  (hd : distance_after_y = 42)
  (h_same_distance : ∀ t : ℝ, speed_x * (t + (distance_after_y / speed_x)) = speed_y * t) :
  (distance_after_y / speed_x) * 60 = 72 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l1750_175093


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_is_34_l1750_175026

/-- Represents a three-dimensional shape --/
structure Shape3D where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- A cube --/
def cube : Shape3D :=
  { faces := 6, edges := 12, vertices := 8 }

/-- Adds a pyramid to one face of a given shape --/
def addPyramid (shape : Shape3D) : Shape3D :=
  { faces := shape.faces + 3,  -- One face is covered, 4 new faces added
    edges := shape.edges + 4,  -- 4 new edges from apex to base
    vertices := shape.vertices + 1 }  -- 1 new vertex (apex)

/-- Calculates the sum of faces, edges, and vertices --/
def sumComponents (shape : Shape3D) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Theorem: The maximum sum of exterior faces, vertices, and edges
    of a shape formed by adding a pyramid to one face of a cube is 34 --/
theorem cube_pyramid_sum_is_34 :
  sumComponents (addPyramid cube) = 34 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_sum_is_34_l1750_175026


namespace NUMINAMATH_CALUDE_apples_per_hour_l1750_175031

/-- Proves that eating the same number of apples every hour for 3 hours,
    totaling 15 apples, results in 5 apples per hour. -/
theorem apples_per_hour 
  (total_hours : ℕ) 
  (total_apples : ℕ) 
  (h1 : total_hours = 3) 
  (h2 : total_apples = 15) : 
  total_apples / total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_hour_l1750_175031


namespace NUMINAMATH_CALUDE_unique_integral_root_l1750_175050

theorem unique_integral_root :
  ∃! (x : ℤ), x - 8 / (x - 4 : ℚ) = 2 - 8 / (x - 4 : ℚ) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integral_root_l1750_175050


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1750_175098

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 12) → 
    ∃ b : ℝ, y₁ = m * x₁ + b :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1750_175098


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1750_175091

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) : a > 0 → b > 0 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2 ∧ b * x + a * y = 0 ∧ y^2 = b^2) →  -- Chord condition
  c^2 = a^2 * (1 + (c^2 / a^2 - 1)) →  -- Semi-latus rectum condition
  Real.sqrt ((c^2 / a^2) - 1) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1750_175091


namespace NUMINAMATH_CALUDE_paper_area_problem_l1750_175062

theorem paper_area_problem (x : ℕ) : 
  (2 * 11 * 11 = 2 * x * 11 + 100) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_area_problem_l1750_175062


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1750_175056

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l1750_175056


namespace NUMINAMATH_CALUDE_curve_intersects_median_unique_point_l1750_175061

/-- Given non-collinear points A, B, C with complex coordinates, 
    prove that the curve intersects the median of triangle ABC at a unique point. -/
theorem curve_intersects_median_unique_point 
  (a b c : ℝ) 
  (h_non_collinear : a + c ≠ 2*b) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
               (1/2 + Complex.I * b) * 2 * (Real.cos t)^2 * (Real.sin t)^2 + 
               (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    (p.re = 1/2 ∧ p.im = (a + 2*b + c) / 4) := by
  sorry


end NUMINAMATH_CALUDE_curve_intersects_median_unique_point_l1750_175061


namespace NUMINAMATH_CALUDE_river_speed_proof_l1750_175086

-- Define the problem parameters
def distance : ℝ := 200
def timeInterval : ℝ := 4
def speedA : ℝ := 36
def speedB : ℝ := 64

-- Define the river current speed as a variable
def riverSpeed : ℝ := 14

-- Theorem statement
theorem river_speed_proof :
  -- First meeting time
  let firstMeetTime : ℝ := distance / (speedA + speedB)
  -- Total time
  let totalTime : ℝ := firstMeetTime + timeInterval
  -- Total distance covered
  let totalDistance : ℝ := 3 * distance
  -- Equation for boat A's journey
  totalDistance = (speedA + riverSpeed + speedA - riverSpeed) * totalTime →
  -- Conclusion
  riverSpeed = 14 := by
  sorry

end NUMINAMATH_CALUDE_river_speed_proof_l1750_175086


namespace NUMINAMATH_CALUDE_equation_solution_l1750_175022

theorem equation_solution (x : ℝ) : (25 : ℝ) / 75 = (x / 75) ^ 3 → x = 75 / (3 : ℝ) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1750_175022


namespace NUMINAMATH_CALUDE_range_of_c_l1750_175035

def p (c : ℝ) := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > c

theorem range_of_c :
  ∀ c : ℝ, c > 0 →
  ((p c ∨ q c) ∧ ¬(p c ∧ q c)) →
  (c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l1750_175035


namespace NUMINAMATH_CALUDE_no_additional_omelets_l1750_175010

-- Define the number of eggs per omelet type
def eggs_plain : ℕ := 3
def eggs_cheese : ℕ := 4
def eggs_vegetable : ℕ := 5

-- Define the total number of eggs
def total_eggs : ℕ := 36

-- Define the number of omelets already requested
def plain_omelets : ℕ := 4
def cheese_omelets : ℕ := 2
def vegetable_omelets : ℕ := 3

-- Calculate the number of eggs used for requested omelets
def used_eggs : ℕ := plain_omelets * eggs_plain + cheese_omelets * eggs_cheese + vegetable_omelets * eggs_vegetable

-- Define the remaining eggs
def remaining_eggs : ℕ := total_eggs - used_eggs

-- Theorem: No additional omelets can be made
theorem no_additional_omelets :
  remaining_eggs < eggs_plain ∧ remaining_eggs < eggs_cheese ∧ remaining_eggs < eggs_vegetable :=
by sorry

end NUMINAMATH_CALUDE_no_additional_omelets_l1750_175010


namespace NUMINAMATH_CALUDE_bennetts_brothers_l1750_175060

/-- Given that Aaron has four brothers and Bennett's number of brothers is two less than twice
    the number of Aaron's brothers, prove that Bennett has 6 brothers. -/
theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) 
    (h1 : aaron_brothers = 4)
    (h2 : bennett_brothers = 2 * aaron_brothers - 2) : 
  bennett_brothers = 6 := by
  sorry

end NUMINAMATH_CALUDE_bennetts_brothers_l1750_175060


namespace NUMINAMATH_CALUDE_system_solution_condition_l1750_175081

/-- The system of equations has at least one solution if and only if -|a| ≤ b ≤ √2|a| -/
theorem system_solution_condition (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ x + |y| = b) ↔ -|a| ≤ b ∧ b ≤ Real.sqrt 2 * |a| :=
by sorry

end NUMINAMATH_CALUDE_system_solution_condition_l1750_175081


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiples_l1750_175046

theorem smallest_number_divisible_by_multiples (n : ℕ) : n = 200 ↔ 
  (∀ m : ℕ, m < n → ¬(15 ∣ (m - 20) ∧ 30 ∣ (m - 20) ∧ 45 ∣ (m - 20) ∧ 60 ∣ (m - 20))) ∧
  (15 ∣ (n - 20) ∧ 30 ∣ (n - 20) ∧ 45 ∣ (n - 20) ∧ 60 ∣ (n - 20)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiples_l1750_175046


namespace NUMINAMATH_CALUDE_lcm_gcd_ratio_540_360_l1750_175075

theorem lcm_gcd_ratio_540_360 : Nat.lcm 540 360 / Nat.gcd 540 360 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_ratio_540_360_l1750_175075


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l1750_175045

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  (initial_amount / 2) / 2 = 51 → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l1750_175045


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1750_175071

/-- The system of equations has exactly one solution if and only if 
    (3 - √5)/2 < t < (3 + √5)/2 -/
theorem unique_solution_condition (t : ℝ) : 
  (∃! x y z v : ℝ, x + y + z + v = 0 ∧ 
    (x*y + y*z + z*v) + t*(x*z + x*v + y*v) = 0) ↔ 
  ((3 - Real.sqrt 5) / 2 < t ∧ t < (3 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1750_175071


namespace NUMINAMATH_CALUDE_candy_mixture_price_l1750_175069

/-- Given two types of candy mixed to produce a mixture with known total weight and value,
    prove that the price of the second candy is $4.30 per pound. -/
theorem candy_mixture_price (x : ℝ) :
  x > 0 ∧
  x + 6.25 = 10 ∧
  3.5 * x + 6.25 * 4.3 = 4 * 10 →
  4.3 = (4 * 10 - 3.5 * x) / 6.25 :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l1750_175069


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1750_175030

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ∧
  (N % 11 = 10) ∧
  (∀ M : ℕ, M > 0 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 → M ≥ N) ∧
  N = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l1750_175030


namespace NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5280_l1750_175047

theorem smallest_two_digit_factor_of_5280 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 5280 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 5280 → min x y ≥ 66) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5280_l1750_175047


namespace NUMINAMATH_CALUDE_rational_solution_product_l1750_175043

theorem rational_solution_product : ∃ (k₁ k₂ : ℕ+), 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₁.val = 0) ∧ 
  (∃ (x : ℚ), 3 * x^2 + 17 * x + k₂.val = 0) ∧ 
  (∀ (k : ℕ+), (∃ (x : ℚ), 3 * x^2 + 17 * x + k.val = 0) → k = k₁ ∨ k = k₂) ∧
  k₁.val * k₂.val = 336 := by
sorry

end NUMINAMATH_CALUDE_rational_solution_product_l1750_175043


namespace NUMINAMATH_CALUDE_least_six_digit_divisible_by_198_l1750_175077

theorem least_six_digit_divisible_by_198 : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧  -- 6-digit number condition
  n % 198 = 0 ∧                 -- divisibility condition
  ∀ m : ℕ, (m ≥ 100000 ∧ m < 1000000) ∧ m % 198 = 0 → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_six_digit_divisible_by_198_l1750_175077


namespace NUMINAMATH_CALUDE_multiply_monomials_l1750_175001

theorem multiply_monomials (x : ℝ) : 2 * x * (5 * x^2) = 10 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_monomials_l1750_175001


namespace NUMINAMATH_CALUDE_min_value_product_l1750_175018

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by sorry

end NUMINAMATH_CALUDE_min_value_product_l1750_175018


namespace NUMINAMATH_CALUDE_area_ratio_bound_for_special_triangles_l1750_175013

/-- Given two right-angled triangles where the incircle radius of the first equals
    the circumcircle radius of the second, prove that the ratio of their areas
    is at least 3 + 2√2 -/
theorem area_ratio_bound_for_special_triangles (S S' r : ℝ) :
  (∃ (a b c a' b' c' : ℝ),
    -- First triangle is right-angled
    a^2 + b^2 = c^2 ∧
    -- Second triangle is right-angled
    a'^2 + b'^2 = c'^2 ∧
    -- Incircle radius of first triangle equals circumcircle radius of second
    r = c' / 2 ∧
    -- Area formulas
    S = r^2 * (a/r + b/r + c/r - π) / 2 ∧
    S' = a' * b' / 2) →
  S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_bound_for_special_triangles_l1750_175013


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l1750_175040

/-- A circle with diameter endpoints (3,2) and (11,8) intersects the x-axis at x = 7 -/
theorem circle_x_axis_intersection :
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (11, 8)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius : ℝ := ((p1.1 - center.1)^2 + (p1.2 - center.2)^2).sqrt
  ∃ x : ℝ, x ≠ p1.1 ∧ (x - center.1)^2 + center.2^2 = radius^2 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l1750_175040


namespace NUMINAMATH_CALUDE_cab_driver_income_l1750_175054

theorem cab_driver_income (income_day1 income_day2 income_day3 income_day4 : ℕ)
  (average_income : ℕ) (total_days : ℕ) :
  income_day1 = 200 →
  income_day2 = 150 →
  income_day3 = 750 →
  income_day4 = 400 →
  average_income = 400 →
  total_days = 5 →
  (income_day1 + income_day2 + income_day3 + income_day4 + 
    (average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4))) / total_days = average_income →
  average_income * total_days - (income_day1 + income_day2 + income_day3 + income_day4) = 500 :=
by sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1750_175054


namespace NUMINAMATH_CALUDE_percentage_increase_l1750_175080

theorem percentage_increase (w : ℝ) (P : ℝ) : 
  w = 80 →
  (w + P / 100 * w) - (w - 25 / 100 * w) = 30 →
  P = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l1750_175080


namespace NUMINAMATH_CALUDE_track_circumference_l1750_175007

/-- The circumference of a circular track given specific meeting conditions of two travelers -/
theorem track_circumference : 
  ∀ (circumference : ℝ) 
    (speed_A speed_B : ℝ) 
    (first_meeting second_meeting : ℝ),
  speed_A > 0 →
  speed_B > 0 →
  first_meeting = 150 →
  second_meeting = circumference - 90 →
  first_meeting / (circumference / 2 - first_meeting) = 
    (circumference / 2 + 90) / (circumference - 90) →
  circumference = 720 := by
sorry

end NUMINAMATH_CALUDE_track_circumference_l1750_175007


namespace NUMINAMATH_CALUDE_point_C_satisfies_condition_l1750_175067

/-- Given points A(-2, 1) and B(1, 4) in the plane, prove that C(-1, 2) satisfies AC = (1/2)CB -/
theorem point_C_satisfies_condition :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (-1, 2)
  (C.1 - A.1, C.2 - A.2) = (1/2 : ℝ) • (B.1 - C.1, B.2 - C.2) := by
  sorry

#check point_C_satisfies_condition

end NUMINAMATH_CALUDE_point_C_satisfies_condition_l1750_175067


namespace NUMINAMATH_CALUDE_phyllis_marble_count_l1750_175074

/-- The number of groups of marbles in Phyllis's collection -/
def num_groups : ℕ := 32

/-- The number of marbles in each group -/
def marbles_per_group : ℕ := 2

/-- The total number of marbles in Phyllis's collection -/
def total_marbles : ℕ := num_groups * marbles_per_group

theorem phyllis_marble_count : total_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_phyllis_marble_count_l1750_175074
