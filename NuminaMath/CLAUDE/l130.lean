import Mathlib

namespace NUMINAMATH_CALUDE_jessie_initial_weight_l130_13033

/-- Represents Jessie's weight change after jogging --/
structure WeightChange where
  lost : ℕ      -- Weight lost in kilograms
  current : ℕ   -- Current weight in kilograms

/-- Calculates the initial weight before jogging --/
def initial_weight (w : WeightChange) : ℕ :=
  w.lost + w.current

/-- Theorem stating Jessie's initial weight was 192 kg --/
theorem jessie_initial_weight :
  let w : WeightChange := { lost := 126, current := 66 }
  initial_weight w = 192 := by
  sorry

end NUMINAMATH_CALUDE_jessie_initial_weight_l130_13033


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l130_13068

theorem candidate_vote_percentage
  (total_votes : ℝ)
  (vote_difference : ℝ)
  (h_total : total_votes = 25000.000000000007)
  (h_diff : vote_difference = 5000) :
  let candidate_percentage := (total_votes - vote_difference) / (2 * total_votes) * 100
  candidate_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l130_13068


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l130_13079

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles A, B, and C in degrees

-- Define the condition for option D
def angle_ratio (t : Triangle) : Prop :=
  ∃ (k : Real), t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

-- Theorem: If the angles of a triangle satisfy the given ratio, it's not necessarily a right triangle
theorem not_necessarily_right_triangle (t : Triangle) : 
  angle_ratio t → ¬ (t.A = 90 ∨ t.B = 90 ∨ t.C = 90) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l130_13079


namespace NUMINAMATH_CALUDE_primitive_root_extension_l130_13003

theorem primitive_root_extension (p : ℕ) (x : ℤ) (h_p : Nat.Prime p) (h_p_odd : p % 2 = 1)
  (h_primitive_root_p2 : IsPrimitiveRoot x (p^2)) :
  ∀ α : ℕ, α ≥ 2 → IsPrimitiveRoot x (p^α) :=
by sorry

end NUMINAMATH_CALUDE_primitive_root_extension_l130_13003


namespace NUMINAMATH_CALUDE_paper_tearing_theorem_l130_13031

/-- Represents the number of parts after n tears -/
def num_parts (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the number of parts is always odd and can never be 100 -/
theorem paper_tearing_theorem :
  ∀ n : ℕ, ∃ k : ℕ, num_parts n = 2 * k + 1 ∧ num_parts n ≠ 100 :=
sorry

end NUMINAMATH_CALUDE_paper_tearing_theorem_l130_13031


namespace NUMINAMATH_CALUDE_least_period_is_twelve_l130_13090

/-- A function satisfying the given condition -/
def SatisfiesCondition (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) + g (x - 2) = g x

/-- The period of a function -/
def IsPeriod (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g (x + p) = g x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (g : ℝ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ IsPeriod g q ∧ ∀ p, 0 < p ∧ p < q → ¬IsPeriod g p

theorem least_period_is_twelve :
  ∀ g : ℝ → ℝ, SatisfiesCondition g → IsLeastPositivePeriod g 12 := by
  sorry

end NUMINAMATH_CALUDE_least_period_is_twelve_l130_13090


namespace NUMINAMATH_CALUDE_sphere_chords_theorem_l130_13082

/-- Represents a sphere with a point inside and three perpendicular chords -/
structure SphereWithChords where
  R : ℝ  -- radius of the sphere
  a : ℝ  -- distance of point A from the center
  h : 0 < R ∧ 0 ≤ a ∧ a < R  -- constraints on R and a

/-- The sum of squares of three mutually perpendicular chords through a point in a sphere -/
def sum_of_squares_chords (s : SphereWithChords) : ℝ := 12 * s.R^2 - 4 * s.a^2

/-- The sum of squares of the segments of three mutually perpendicular chords created by a point in a sphere -/
def sum_of_squares_segments (s : SphereWithChords) : ℝ := 6 * s.R^2 - 2 * s.a^2

/-- Theorem stating the properties of chords in a sphere -/
theorem sphere_chords_theorem (s : SphereWithChords) :
  (sum_of_squares_chords s = 12 * s.R^2 - 4 * s.a^2) ∧
  (sum_of_squares_segments s = 6 * s.R^2 - 2 * s.a^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_chords_theorem_l130_13082


namespace NUMINAMATH_CALUDE_transformation_confluence_l130_13019

/-- Represents a word in the alphabet {a, b} --/
inductive Word
| empty : Word
| cons_a : ℕ → Word → Word
| cons_b : ℕ → Word → Word

/-- Represents a transformation rule --/
structure TransformRule where
  k : ℕ
  l : ℕ
  k' : ℕ
  l' : ℕ
  h_k : k ≥ 1
  h_l : l ≥ 1
  h_k' : k' ≥ 1
  h_l' : l' ≥ 1

/-- Applies a transformation rule to a word --/
def applyRule (rule : TransformRule) (w : Word) : Option Word :=
  sorry

/-- Checks if a word is terminal with respect to a rule --/
def isTerminal (rule : TransformRule) (w : Word) : Prop :=
  applyRule rule w = none

/-- Represents a sequence of transformations --/
def TransformSequence := List (TransformRule × Word)

/-- Applies a sequence of transformations to a word --/
def applySequence (seq : TransformSequence) (w : Word) : Word :=
  sorry

theorem transformation_confluence (rule : TransformRule) (w : Word) :
  ∀ (seq1 seq2 : TransformSequence),
    isTerminal rule (applySequence seq1 w) →
    isTerminal rule (applySequence seq2 w) →
    applySequence seq1 w = applySequence seq2 w :=
  sorry

end NUMINAMATH_CALUDE_transformation_confluence_l130_13019


namespace NUMINAMATH_CALUDE_min_sum_distances_on_BC_l130_13083

/-- Four distinct points on a line -/
structure FourPoints where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  ordered : A < B ∧ B < C ∧ C < D

/-- Sum of distances from a point X to A, B, C, and D -/
def sumOfDistances (fp : FourPoints) (X : ℝ) : ℝ :=
  |X - fp.A| + |X - fp.B| + |X - fp.C| + |X - fp.D|

/-- The point that minimizes the sum of distances is on the segment BC -/
theorem min_sum_distances_on_BC (fp : FourPoints) :
  ∃ (X : ℝ), fp.B ≤ X ∧ X ≤ fp.C ∧
  ∀ (Y : ℝ), sumOfDistances fp X ≤ sumOfDistances fp Y :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_on_BC_l130_13083


namespace NUMINAMATH_CALUDE_quotient_change_l130_13021

theorem quotient_change (A B : ℝ) (h : A / B = 0.514) : 
  (10 * A) / (B / 100) = 514 := by
sorry

end NUMINAMATH_CALUDE_quotient_change_l130_13021


namespace NUMINAMATH_CALUDE_unique_base_sum_l130_13047

def sum_single_digits (b : ℕ) : ℕ := 
  if b % 2 = 0 then
    b * (b - 1) / 2
  else
    (b^2 - 1) / 2

theorem unique_base_sum : 
  ∃! b : ℕ, b > 0 ∧ sum_single_digits b = 2 * b + 8 :=
sorry

end NUMINAMATH_CALUDE_unique_base_sum_l130_13047


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l130_13085

theorem sum_sqrt_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x / (1 - x)) + Real.sqrt (y / (1 - y)) + Real.sqrt (z / (1 - z)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l130_13085


namespace NUMINAMATH_CALUDE_parabola_theorem_l130_13092

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Line passing through (1,0) -/
structure Line where
  k : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = k*(x-1)

/-- Point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : para.eq x y

/-- Circle passing through three points -/
def circle_passes_through (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x3-x1)*(x2-x1) + (y3-y1)*(y2-y1) = 0

/-- Main theorem -/
theorem parabola_theorem (para : Parabola) :
  (∀ l : Line, ∀ P Q : ParabolaPoint para,
    l.eq P.x P.y ∧ l.eq Q.x Q.y →
    circle_passes_through P.x P.y Q.x Q.y 0 0) →
  para.p = 1/2 ∧
  (∀ R : ℝ × ℝ,
    (∃ P Q : ParabolaPoint para,
      R.1 = P.x + Q.x - 1/4 ∧
      R.2 = P.y + Q.y) →
    R.2^2 = R.1 - 7/4) :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l130_13092


namespace NUMINAMATH_CALUDE_modulo_equivalence_problem_l130_13063

theorem modulo_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 15478 [MOD 15] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_problem_l130_13063


namespace NUMINAMATH_CALUDE_fraction_of_silver_knights_with_shields_l130_13029

theorem fraction_of_silver_knights_with_shields :
  ∀ (total_knights : ℕ) (silver_knights : ℕ) (golden_knights : ℕ) (knights_with_shields : ℕ)
    (silver_knights_with_shields : ℕ) (golden_knights_with_shields : ℕ),
  total_knights > 0 →
  silver_knights + golden_knights = total_knights →
  silver_knights = (3 * total_knights) / 8 →
  knights_with_shields = total_knights / 4 →
  silver_knights_with_shields + golden_knights_with_shields = knights_with_shields →
  silver_knights_with_shields * golden_knights = 3 * golden_knights_with_shields * silver_knights →
  silver_knights_with_shields * 7 = silver_knights * 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_silver_knights_with_shields_l130_13029


namespace NUMINAMATH_CALUDE_spelling_bee_probability_l130_13095

/-- The probability of selecting all girls in a spelling bee competition -/
theorem spelling_bee_probability (total : ℕ) (girls : ℕ) (selected : ℕ) 
  (h_total : total = 8) 
  (h_girls : girls = 5)
  (h_selected : selected = 3) :
  (Nat.choose girls selected : ℚ) / (Nat.choose total selected) = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_probability_l130_13095


namespace NUMINAMATH_CALUDE_largest_special_square_l130_13081

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem largest_special_square : 
  (is_perfect_square 1681) ∧ 
  (1681 % 10 ≠ 0) ∧ 
  (is_perfect_square (remove_last_two_digits 1681)) ∧ 
  (∀ m : ℕ, m > 1681 → 
    ¬(is_perfect_square m ∧ 
      m % 10 ≠ 0 ∧ 
      is_perfect_square (remove_last_two_digits m))) :=
by sorry

end NUMINAMATH_CALUDE_largest_special_square_l130_13081


namespace NUMINAMATH_CALUDE_craig_seashells_l130_13018

theorem craig_seashells : ∃ (c : ℕ), c = 54 ∧ c > 0 ∧ ∃ (b : ℕ), 
  (c : ℚ) / (b : ℚ) = 9 / 7 ∧ b = c - 12 := by
  sorry

end NUMINAMATH_CALUDE_craig_seashells_l130_13018


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l130_13027

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem parallel_vectors_imply_x_half :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + 2 • b x = k • (2 • a - 2 • b x)) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l130_13027


namespace NUMINAMATH_CALUDE_birth_probability_l130_13036

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := p^n

-- Define the probability of having 3 of one gender and 2 of the other
def prob_three_two : ℚ := Nat.choose n 3 * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_four_one : ℚ := 2 * Nat.choose n 1 * p^n

theorem birth_probability :
  prob_three_two > prob_all_same ∧
  prob_four_one > prob_all_same ∧
  prob_three_two = prob_four_one :=
by sorry

end NUMINAMATH_CALUDE_birth_probability_l130_13036


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l130_13026

theorem pet_shop_dogs (birds : ℕ) (snakes : ℕ) (spider : ℕ) (total_legs : ℕ) :
  birds = 3 → snakes = 4 → spider = 1 → total_legs = 34 →
  ∃ dogs : ℕ, dogs = 5 ∧ total_legs = birds * 2 + dogs * 4 + spider * 8 :=
by sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l130_13026


namespace NUMINAMATH_CALUDE_probability_not_rain_l130_13077

theorem probability_not_rain (p : ℚ) (h : p = 3 / 10) : 1 - p = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_rain_l130_13077


namespace NUMINAMATH_CALUDE_train_length_l130_13088

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), abs (length_m - 140.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l130_13088


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l130_13032

theorem jerrys_average_increase :
  ∀ (original_average new_average : ℚ),
  original_average = 94 →
  (3 * original_average + 102) / 4 = new_average →
  new_average - original_average = 2 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l130_13032


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l130_13014

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, -2) and (-4, 10) is 6. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -2
  let x₂ : ℝ := -4
  let y₂ : ℝ := 10
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 6 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l130_13014


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l130_13059

theorem quadratic_form_equivalence (b m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 56 = (x + m)^2 + 20) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l130_13059


namespace NUMINAMATH_CALUDE_soccer_lineup_count_l130_13094

theorem soccer_lineup_count (n : ℕ) (h : n = 18) : 
  n * (n - 1) * (Nat.choose (n - 2) 9) = 3501120 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_count_l130_13094


namespace NUMINAMATH_CALUDE_saree_discount_problem_l130_13099

/-- Proves that the first discount percentage is 20% given the conditions of the saree pricing problem --/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 350 →
  final_price = 266 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l130_13099


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_in_range_l130_13089

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_iff_m_in_range (m : ℝ) :
  (A m ∩ B m).Nonempty ↔ 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_in_range_l130_13089


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l130_13039

/-- Given a hyperbola with equation x²/m² - y² = 4 where m > 0 and focal distance 8,
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let focal_distance : ℝ := 8
  let a : ℝ := m * 2
  let b : ℝ := 2
  let c : ℝ := focal_distance / 2
  let eccentricity : ℝ := c / a
  eccentricity = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l130_13039


namespace NUMINAMATH_CALUDE_race_length_is_1000_l130_13030

/-- The length of a race given Aubrey's and Violet's positions -/
def race_length (violet_distance_covered : ℕ) (violet_distance_to_finish : ℕ) : ℕ :=
  violet_distance_covered + violet_distance_to_finish

/-- Theorem stating that the race length is 1000 meters -/
theorem race_length_is_1000 :
  race_length 721 279 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l130_13030


namespace NUMINAMATH_CALUDE_parabola_c_value_l130_13058

/-- A parabola passing through two specific points has a determined c-value. -/
theorem parabola_c_value (b c : ℝ) : 
  (2 = 1^2 + b*1 + c) ∧ (2 = 5^2 + b*5 + c) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l130_13058


namespace NUMINAMATH_CALUDE_minimum_guests_l130_13008

theorem minimum_guests (total_food : ℝ) (max_individual_food : ℝ) (h1 : total_food = 520) (h2 : max_individual_food = 1.5) :
  ∃ n : ℕ, n * max_individual_food ≥ total_food ∧ ∀ m : ℕ, m * max_individual_food ≥ total_food → m ≥ n ∧ n = 347 :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_l130_13008


namespace NUMINAMATH_CALUDE_max_terms_of_arithmetic_sequence_l130_13057

/-- An arithmetic sequence with common difference 4 and real-valued terms -/
def ArithmeticSequence (a₁ : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1) * 4

/-- The sum of terms from the second to the nth term -/
def SumOfRemainingTerms (a₁ : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (a₁ + 2 * n)

/-- The condition that the square of the first term plus the sum of remaining terms does not exceed 100 -/
def SequenceCondition (a₁ : ℝ) (n : ℕ) : Prop :=
  a₁^2 + SumOfRemainingTerms a₁ n ≤ 100

theorem max_terms_of_arithmetic_sequence :
  ∀ a₁ : ℝ, ∀ n : ℕ, SequenceCondition a₁ n → n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_terms_of_arithmetic_sequence_l130_13057


namespace NUMINAMATH_CALUDE_problem_solution_l130_13016

theorem problem_solution : (69842 * 69842 - 30158 * 30158) / (69842 - 30158) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l130_13016


namespace NUMINAMATH_CALUDE_function_max_min_condition_l130_13097

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

-- State the theorem
theorem function_max_min_condition (a : ℝ) : 
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a > 2 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_condition_l130_13097


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l130_13017

theorem min_value_trig_expression (α β : Real) :
  ∃ (min : Real),
    (∀ (α' β' : Real), (3 * Real.cos α' + 4 * Real.sin β' - 7)^2 + (3 * Real.sin α' + 4 * Real.cos β' - 12)^2 ≥ min) ∧
    ((3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = min) ∧
    min = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l130_13017


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l130_13050

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∃ z w : ℝ, Real.sin z + Real.sin w = 1/3 ∧ 
    Real.sin w + (Real.cos z)^2 = 19/12) ∧
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin b + (Real.cos a)^2 ≤ 19/12) ∧
  (∃ u v : ℝ, Real.sin u + Real.sin v = 1/3 ∧ 
    Real.sin v + (Real.cos u)^2 = -2/3) ∧
  (∀ c d : ℝ, Real.sin c + Real.sin d = 1/3 → 
    Real.sin d + (Real.cos c)^2 ≥ -2/3) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l130_13050


namespace NUMINAMATH_CALUDE_ab_value_l130_13020

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l130_13020


namespace NUMINAMATH_CALUDE_plate_color_probability_l130_13070

/-- The probability of selecting two plates of the same color -/
def same_color_probability (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let same_color_combinations := (red.choose 2) + (blue.choose 2) + (yellow.choose 2)
  let total_combinations := total.choose 2
  same_color_combinations / total_combinations

/-- Theorem: The probability of selecting two plates of the same color
    given 7 red plates, 5 blue plates, and 3 yellow plates is 34/105 -/
theorem plate_color_probability :
  same_color_probability 7 5 3 = 34 / 105 := by
  sorry

end NUMINAMATH_CALUDE_plate_color_probability_l130_13070


namespace NUMINAMATH_CALUDE_max_value_implies_a_l130_13015

def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 3/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l130_13015


namespace NUMINAMATH_CALUDE_circle_M_equation_l130_13060

/-- Circle M passing through two points with center on a line -/
def circle_M (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), 
    (a - b - 4 = 0) ∧ 
    (((-1) - a)^2 + ((-4) - b)^2 = (x - a)^2 + (y - b)^2) ∧
    ((6 - a)^2 + (3 - b)^2 = (x - a)^2 + (y - b)^2)

/-- Theorem: The equation of circle M is (x-3)^2 + (y+1)^2 = 25 -/
theorem circle_M_equation : 
  ∀ x y : ℝ, circle_M x y ↔ (x - 3)^2 + (y + 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_M_equation_l130_13060


namespace NUMINAMATH_CALUDE_min_value_circle_l130_13054

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (min : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 7 - 4*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_circle_l130_13054


namespace NUMINAMATH_CALUDE_tony_distance_behind_l130_13010

-- Define the slope length
def slope_length : ℝ := 700

-- Define the meeting point distance from the top
def meeting_point : ℝ := 70

-- Define Maria's and Tony's uphill speeds as variables
variable (maria_uphill_speed tony_uphill_speed : ℝ)

-- Define the theorem
theorem tony_distance_behind (maria_uphill_speed tony_uphill_speed : ℝ) 
  (h_positive : maria_uphill_speed > 0 ∧ tony_uphill_speed > 0) :
  let maria_total_distance := slope_length + slope_length / 2
  let tony_total_distance := maria_total_distance * (tony_uphill_speed / maria_uphill_speed)
  let distance_behind := maria_total_distance - tony_total_distance
  2 * distance_behind = 300 := by sorry

end NUMINAMATH_CALUDE_tony_distance_behind_l130_13010


namespace NUMINAMATH_CALUDE_cos_210_degrees_l130_13075

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l130_13075


namespace NUMINAMATH_CALUDE_optimal_strategy_and_expected_red_balls_l130_13062

-- Define the contents of A's box
structure BoxA where
  red : ℕ
  white : ℕ
  sum_eq_four : red + white = 4

-- Define the contents of B's box
def BoxB : Finset (Fin 4) := {0, 1, 2, 3}

-- Define the probability of winning for A given their box contents
def win_probability (box : BoxA) : ℚ :=
  (box.red * box.white * 2) / (12 * 6)

-- Define the expected number of red balls drawn
def expected_red_balls (box : BoxA) : ℚ :=
  (box.red * 2 / 6) + (2 / 4)

-- Theorem statement
theorem optimal_strategy_and_expected_red_balls :
  ∃ (box : BoxA),
    (∀ (other : BoxA), win_probability box ≥ win_probability other) ∧
    (box.red = 2 ∧ box.white = 2) ∧
    (expected_red_balls box = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_optimal_strategy_and_expected_red_balls_l130_13062


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_values_l130_13056

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem unique_solution_implies_a_values (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_values_l130_13056


namespace NUMINAMATH_CALUDE_at_least_two_primes_in_base_n_1002_l130_13007

def base_n_1002 (n : ℕ) : ℕ := n^3 + 2

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem at_least_two_primes_in_base_n_1002 : 
  ∃ n1 n2 : ℕ, n1 ≥ 2 ∧ n2 ≥ 2 ∧ n1 ≠ n2 ∧ 
  is_prime (base_n_1002 n1) ∧ is_prime (base_n_1002 n2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_primes_in_base_n_1002_l130_13007


namespace NUMINAMATH_CALUDE_total_seashells_is_fifty_l130_13061

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem: The total number of seashells found by Tim and Sally is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_fifty_l130_13061


namespace NUMINAMATH_CALUDE_square_difference_dollar_l130_13046

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem square_difference_dollar (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_dollar_l130_13046


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l130_13093

theorem least_positive_integer_with_given_remainders : ∃ N : ℕ+,
  (N : ℤ) ≡ 3 [ZMOD 4] ∧
  (N : ℤ) ≡ 4 [ZMOD 5] ∧
  (N : ℤ) ≡ 5 [ZMOD 6] ∧
  (N : ℤ) ≡ 6 [ZMOD 7] ∧
  (N : ℤ) ≡ 10 [ZMOD 11] ∧
  (∀ m : ℕ+, m < N →
    ¬((m : ℤ) ≡ 3 [ZMOD 4] ∧
      (m : ℤ) ≡ 4 [ZMOD 5] ∧
      (m : ℤ) ≡ 5 [ZMOD 6] ∧
      (m : ℤ) ≡ 6 [ZMOD 7] ∧
      (m : ℤ) ≡ 10 [ZMOD 11])) ∧
  N = 4619 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l130_13093


namespace NUMINAMATH_CALUDE_distance_to_parabola_directrix_l130_13023

/-- The distance from a point to the directrix of a parabola -/
def distance_to_directrix (a : ℝ) (P : ℝ × ℝ) : ℝ :=
  |P.1 + a|

/-- The parabola equation -/
def is_parabola (x y : ℝ) (a : ℝ) : Prop :=
  y^2 = -4*a*x

theorem distance_to_parabola_directrix :
  ∃ (a : ℝ), 
    is_parabola (-2) 4 a ∧ 
    distance_to_directrix a (-2, 4) = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_to_parabola_directrix_l130_13023


namespace NUMINAMATH_CALUDE_paths_7_6_grid_l130_13025

/-- The number of paths on a grid from bottom left to top right -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of paths on a 7 by 6 grid is 1716 -/
theorem paths_7_6_grid : grid_paths 7 6 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_paths_7_6_grid_l130_13025


namespace NUMINAMATH_CALUDE_greatest_sum_of_valid_pair_l130_13040

/-- Two integers that differ by 2 and have a product less than 500 -/
def ValidPair (n m : ℤ) : Prop :=
  m = n + 2 ∧ n * m < 500

/-- The sum of a valid pair of integers -/
def PairSum (n m : ℤ) : ℤ := n + m

/-- Theorem: The greatest possible sum of two integers that differ by 2 
    and whose product is less than 500 is 44 -/
theorem greatest_sum_of_valid_pair : 
  (∃ (n m : ℤ), ValidPair n m ∧ 
    ∀ (k l : ℤ), ValidPair k l → PairSum k l ≤ PairSum n m) ∧
  (∀ (n m : ℤ), ValidPair n m → PairSum n m ≤ 44) := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_of_valid_pair_l130_13040


namespace NUMINAMATH_CALUDE_quadratic_factorization_l130_13002

theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l130_13002


namespace NUMINAMATH_CALUDE_decimal_representation_digits_l130_13012

theorem decimal_representation_digits (n : ℕ) (d : ℕ) (h : n / d = 7^3 / (14^2 * 125)) : 
  (∃ k : ℕ, n / d = k / 1000 ∧ k < 1000 ∧ k ≥ 100) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_digits_l130_13012


namespace NUMINAMATH_CALUDE_min_d_value_l130_13049

theorem min_d_value (a b d : ℕ+) (h1 : a < b) (h2 : b < d + 1)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
       p.2 = |p.1 - a| + |p.1 - b| + |p.1 - (d + 1)|) :
  (∀ d' : ℕ+, d' ≥ d → 
    ∃ a' b' : ℕ+, a' < b' ∧ b' < d' + 1 ∧
    ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
    p.2 = |p.1 - a'| + |p.1 - b'| + |p.1 - (d' + 1)|) →
  d = 2020 := by
sorry

end NUMINAMATH_CALUDE_min_d_value_l130_13049


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l130_13074

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l130_13074


namespace NUMINAMATH_CALUDE_evaluate_expression_l130_13096

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l130_13096


namespace NUMINAMATH_CALUDE_min_value_of_expression_l130_13055

theorem min_value_of_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y + 1)^2 ≥ 5 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b + 1)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l130_13055


namespace NUMINAMATH_CALUDE_folded_blankets_theorem_l130_13034

/-- The thickness of a stack of folded blankets -/
def folded_blankets_thickness (initial_thickness : ℕ) (num_blankets : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * initial_thickness * (2 ^ num_folds)

/-- Theorem: The thickness of n blankets, each initially 3 inches thick and folded 4 times, is 48n inches -/
theorem folded_blankets_theorem (n : ℕ) :
  folded_blankets_thickness 3 n 4 = 48 * n := by
  sorry

end NUMINAMATH_CALUDE_folded_blankets_theorem_l130_13034


namespace NUMINAMATH_CALUDE_stating_num_distributions_eq_16_l130_13071

/-- Represents the number of classes -/
def num_classes : ℕ := 4

/-- Represents the number of "Outstanding Class" spots -/
def num_outstanding_class : ℕ := 4

/-- Represents the number of "Outstanding Group Branch" spots -/
def num_outstanding_group : ℕ := 1

/-- Represents the total number of spots to be distributed -/
def total_spots : ℕ := num_outstanding_class + num_outstanding_group

/-- 
  Theorem stating that the number of ways to distribute the spots among classes,
  with each class receiving at least one spot, is equal to 16
-/
theorem num_distributions_eq_16 : 
  (Finset.univ.filter (fun f : Fin num_classes → Fin (total_spots + 1) => 
    (∀ i, f i > 0) ∧ (Finset.sum Finset.univ f = total_spots))).card = 16 := by
  sorry


end NUMINAMATH_CALUDE_stating_num_distributions_eq_16_l130_13071


namespace NUMINAMATH_CALUDE_mandy_med_school_acceptances_l130_13000

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 42)
  (h2 : applied_fraction = 1 / 3)
  (h3 : accepted_fraction = 1 / 2)
  : ℕ :=
  by
    sorry

#check mandy_med_school_acceptances

end NUMINAMATH_CALUDE_mandy_med_school_acceptances_l130_13000


namespace NUMINAMATH_CALUDE_sum_of_solutions_l130_13028

theorem sum_of_solutions (x : ℝ) : 
  (5 * x^2 - 3 * x - 2 = 0) → 
  (∃ y : ℝ, 5 * y^2 - 3 * y - 2 = 0 ∧ x + y = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l130_13028


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l130_13048

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l130_13048


namespace NUMINAMATH_CALUDE_question_differentiates_inhabitants_l130_13013

-- Define the types of inhabitants
inductive InhabitantType
  | TruthTeller
  | Liar

-- Define the possible answers
inductive Answer
  | Yes
  | No

-- Function to determine how an inhabitant would answer the question
def answer_question (inhabitant_type : InhabitantType) : Answer :=
  match inhabitant_type with
  | InhabitantType.TruthTeller => Answer.No
  | InhabitantType.Liar => Answer.Yes

-- Theorem stating that the question can differentiate between truth-tellers and liars
theorem question_differentiates_inhabitants :
  ∀ (t : InhabitantType),
    (t = InhabitantType.TruthTeller ↔ answer_question t = Answer.No) ∧
    (t = InhabitantType.Liar ↔ answer_question t = Answer.Yes) :=
by sorry

end NUMINAMATH_CALUDE_question_differentiates_inhabitants_l130_13013


namespace NUMINAMATH_CALUDE_percentage_difference_l130_13005

theorem percentage_difference : (0.6 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l130_13005


namespace NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l130_13004

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_geq_four_l130_13004


namespace NUMINAMATH_CALUDE_first_platform_length_l130_13009

/-- Given a train and two platforms, calculates the length of the first platform. -/
theorem first_platform_length 
  (train_length : ℝ) 
  (first_platform_time : ℝ) 
  (second_platform_length : ℝ) 
  (second_platform_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_time = 15)
  (h3 : second_platform_length = 500)
  (h4 : second_platform_time = 20) :
  ∃ L : ℝ, (L + train_length) / first_platform_time = 
           (second_platform_length + train_length) / second_platform_time ∧ 
           L = 350 :=
by sorry

end NUMINAMATH_CALUDE_first_platform_length_l130_13009


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l130_13064

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔
    x ∈ Set.Ioi 3 ∪ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l130_13064


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l130_13065

theorem reciprocal_of_negative_fraction (a b : ℚ) (h : b ≠ 0) :
  ((-a) / b)⁻¹ = -(b / a) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l130_13065


namespace NUMINAMATH_CALUDE_cookfire_logs_added_l130_13076

/-- The number of logs added to a cookfire each hour, given the initial number of logs,
    burn rate, duration, and final number of logs. -/
def logsAddedPerHour (initialLogs burnRate duration finalLogs : ℕ) : ℕ :=
  let logsAfterBurning := initialLogs - burnRate * duration
  (finalLogs - logsAfterBurning + burnRate * (duration - 1)) / duration

theorem cookfire_logs_added (x : ℕ) :
  logsAddedPerHour 6 3 3 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_cookfire_logs_added_l130_13076


namespace NUMINAMATH_CALUDE_last_two_digits_theorem_l130_13084

theorem last_two_digits_theorem (n : ℕ) (h : Odd n) :
  (2^(2*n) * (2^(2*n + 1) - 1)) % 100 = 28 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_theorem_l130_13084


namespace NUMINAMATH_CALUDE_difference_of_squares_simplification_l130_13035

theorem difference_of_squares_simplification : (365^2 - 349^2) / 16 = 714 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_simplification_l130_13035


namespace NUMINAMATH_CALUDE_complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l130_13044

/-- Representation of a complex number in a complex base -n+i --/
structure ComplexBaseRepresentation (n : ℕ+) where
  coeffs : Fin 4 → Fin 257
  nonzero_lead : coeffs 3 ≠ 0

/-- The value represented by a ComplexBaseRepresentation --/
def value (n : ℕ+) (rep : ComplexBaseRepresentation n) : ℂ :=
  (rep.coeffs 3 : ℂ) * (-n + Complex.I)^3 +
  (rep.coeffs 2 : ℂ) * (-n + Complex.I)^2 +
  (rep.coeffs 1 : ℂ) * (-n + Complex.I) +
  (rep.coeffs 0 : ℂ)

/-- Theorem stating the existence and uniqueness of the representation --/
theorem complex_base_representation_exists_unique (n : ℕ+) (z : ℂ) 
  (h : ∃ (r s : ℤ), z = r + s * Complex.I) :
  ∃! (rep : ComplexBaseRepresentation n), value n rep = z :=
sorry

/-- Theorem stating that for base -4+i, there exist integers representable in four digits --/
theorem integer_representable_in_base_neg4_plus_i :
  ∃ (k : ℤ) (rep : ComplexBaseRepresentation 4),
    value 4 rep = k ∧ k = (value 4 rep).re :=
sorry

end NUMINAMATH_CALUDE_complex_base_representation_exists_unique_integer_representable_in_base_neg4_plus_i_l130_13044


namespace NUMINAMATH_CALUDE_M_equals_N_l130_13006

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l130_13006


namespace NUMINAMATH_CALUDE_angle_measure_l130_13037

theorem angle_measure : 
  ∃ x : ℝ, (x = 2 * (90 - x) - 60) ∧ (x = 40) := by sorry

end NUMINAMATH_CALUDE_angle_measure_l130_13037


namespace NUMINAMATH_CALUDE_remainder_theorem_l130_13098

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x - 2) * p x + 12) →
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x + 2) * p x + 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l130_13098


namespace NUMINAMATH_CALUDE_sum_of_integers_with_product_5_4_l130_13024

theorem sum_of_integers_with_product_5_4 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_product_5_4_l130_13024


namespace NUMINAMATH_CALUDE_area_of_isosceles_right_triangle_l130_13041

/-- Given a square ABCD and an isosceles right triangle CMN where:
  - The area of square ABCD is 4 square inches
  - MN = NC
  - x is the length of BN
Prove that the area of triangle CMN is (2 - 2x + 0.5x^2)√2 square inches -/
theorem area_of_isosceles_right_triangle (x : ℝ) :
  let abcd_area : ℝ := 4
  let cmn_is_isosceles_right : Prop := true
  let mn_eq_nc : Prop := true
  let bn_length : ℝ := x
  let cmn_area : ℝ := (2 - 2*x + 0.5*x^2) * Real.sqrt 2
  abcd_area = 4 → cmn_is_isosceles_right → mn_eq_nc → cmn_area = (2 - 2*x + 0.5*x^2) * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_area_of_isosceles_right_triangle_l130_13041


namespace NUMINAMATH_CALUDE_rook_tour_existence_l130_13091

/-- A rook move on an m × n board. -/
inductive RookMove
  | up : RookMove
  | right : RookMove
  | down : RookMove
  | left : RookMove

/-- A valid sequence of rook moves on an m × n board. -/
def ValidMoveSequence (m n : ℕ) : List RookMove → Prop :=
  sorry

/-- A sequence of moves visits all squares exactly once and returns to start. -/
def VisitsAllSquaresOnce (m n : ℕ) (moves : List RookMove) : Prop :=
  sorry

theorem rook_tour_existence (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ moves : List RookMove, ValidMoveSequence m n moves ∧ VisitsAllSquaresOnce m n moves) ↔
  (Even m ∧ Even n) :=
sorry

end NUMINAMATH_CALUDE_rook_tour_existence_l130_13091


namespace NUMINAMATH_CALUDE_roller_coaster_cost_proof_l130_13042

/-- The cost of a ride on the Ferris wheel in tickets -/
def ferris_wheel_cost : ℚ := 2

/-- The discount in tickets for going on multiple rides -/
def multiple_ride_discount : ℚ := 1

/-- The value of the newspaper coupon in tickets -/
def newspaper_coupon : ℚ := 1

/-- The total number of tickets Zach needed to buy for both rides -/
def total_tickets_bought : ℚ := 7

/-- The cost of a ride on the roller coaster in tickets -/
def roller_coaster_cost : ℚ := 7

theorem roller_coaster_cost_proof :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = total_tickets_bought :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_proof_l130_13042


namespace NUMINAMATH_CALUDE_pyramid_volume_approx_l130_13067

/-- Represents a pyramid with a square base and a vertex -/
structure Pyramid where
  baseArea : ℝ
  triangleABEArea : ℝ
  triangleCDEArea : ℝ

/-- Calculates the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

/-- The theorem stating the volume of the given pyramid -/
theorem pyramid_volume_approx (p : Pyramid) 
  (h1 : p.baseArea = 324)
  (h2 : p.triangleABEArea = 180)
  (h3 : p.triangleCDEArea = 126) : 
  ∃ (ε : ℝ), ε > 0 ∧ |pyramidVolume p - 2125.76| < ε := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_approx_l130_13067


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l130_13045

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_first_quadrant :
  (2 - Complex.I) * z = 1 + Complex.I →
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l130_13045


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l130_13078

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set_size := 2 * n
  let special_num := 1 + 1 / n
  let regular_num := 1
  let sum := (set_size - 1) * regular_num + special_num
  sum / set_size = 1 + 1 / (2 * n^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l130_13078


namespace NUMINAMATH_CALUDE_catenary_properties_l130_13086

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, f a b x = f a b (-x) ↔ a = b) ∧
  (∀ x, f a b x = -f a b (-x) ↔ a = -b) ∧
  (a * b < 0 → ∀ x y, x < y → f a b x < f a b y ∨ ∀ x y, x < y → f a b x > f a b y) ∧
  (a * b > 0 → ∃ x, (∀ y, f a b y ≥ f a b x) ∨ (∀ y, f a b y ≤ f a b x)) :=
sorry

end NUMINAMATH_CALUDE_catenary_properties_l130_13086


namespace NUMINAMATH_CALUDE_no_solution_range_l130_13038

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + a + 1| + |x + a^2 - 2| ≥ 3) ↔ 
  (a ≤ -2 ∨ (0 ≤ a ∧ a ≤ 1) ∨ 3 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_no_solution_range_l130_13038


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l130_13051

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l130_13051


namespace NUMINAMATH_CALUDE_wages_comparison_l130_13022

theorem wages_comparison (erica robin charles : ℝ) 
  (h1 : robin = 1.3 * erica) 
  (h2 : charles = 1.23076923076923077 * robin) : 
  charles = 1.6 * erica := by
sorry

end NUMINAMATH_CALUDE_wages_comparison_l130_13022


namespace NUMINAMATH_CALUDE_fraction_product_cube_l130_13069

theorem fraction_product_cube (x : ℝ) (hx : x ≠ 0) : 
  (8 / 9)^3 * (x / 3)^3 * (3 / x)^3 = 512 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_cube_l130_13069


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l130_13001

open Complex

theorem fourteenth_root_of_unity : ∃ (n : ℕ) (h : n ≤ 13),
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) =
  Complex.exp (2 * Real.pi * (n : ℝ) * Complex.I / 14) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l130_13001


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l130_13073

theorem min_value_trig_expression (α : ℝ) : 
  9 / (Real.sin α)^2 + 1 / (Real.cos α)^2 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l130_13073


namespace NUMINAMATH_CALUDE_power_two_gt_sum_powers_l130_13072

theorem power_two_gt_sum_powers (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_power_two_gt_sum_powers_l130_13072


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l130_13087

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), 
    prove that a - b = 3 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l130_13087


namespace NUMINAMATH_CALUDE_parabola_vertex_l130_13080

/-- 
Given a parabola y = -x^2 + px + q where the solution to y ≤ 0 is (-∞, -4] ∪ [6, ∞),
prove that the vertex of the parabola is (1, 25).
-/
theorem parabola_vertex (p q : ℝ) : 
  (∀ x, -x^2 + p*x + q ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ x y, x = 1 ∧ y = 25 ∧ ∀ t, -t^2 + p*t + q ≤ y := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l130_13080


namespace NUMINAMATH_CALUDE_additional_monthly_income_l130_13053

/-- Given a shoe company's current monthly sales and desired annual income,
    calculate the additional monthly income required to reach the annual goal. -/
theorem additional_monthly_income
  (current_monthly_sales : ℕ)
  (desired_annual_income : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : desired_annual_income = 60000) :
  (desired_annual_income - current_monthly_sales * 12) / 12 = 1000 :=
by sorry

end NUMINAMATH_CALUDE_additional_monthly_income_l130_13053


namespace NUMINAMATH_CALUDE_ninth_term_is_negative_256_l130_13011

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℤ, a (n + 1) = a n * q
  prod_condition : a 2 * a 5 = -32
  sum_condition : a 3 + a 4 = 4

/-- The theorem stating that a₉ = -256 for the given geometric sequence -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_negative_256_l130_13011


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l130_13066

/-- If a, b, and c form a geometric sequence of real numbers, then ax^2 + bx + c has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic (a b c : ℝ) 
  (h_geo : b^2 = a*c) (h_pos : a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l130_13066


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l130_13052

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) : 
  radius = 14 → pencil_length = 0.5 → 
  (2 * radius * 12) / pencil_length = 56 := by
  sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l130_13052


namespace NUMINAMATH_CALUDE_C_power_50_l130_13043

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299 : ℤ), -100; 800, 251] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l130_13043
