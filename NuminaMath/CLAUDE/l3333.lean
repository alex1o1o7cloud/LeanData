import Mathlib

namespace NUMINAMATH_CALUDE_inequalities_theorem_l3333_333397

variables (a b c x y z : ℝ)

def M : ℝ := a * x + b * y + c * z
def N : ℝ := a * z + b * y + c * x
def P : ℝ := a * y + b * z + c * x
def Q : ℝ := a * z + b * x + c * y

theorem inequalities_theorem (h1 : a > b) (h2 : b > c) (h3 : x > y) (h4 : y > z) :
  M a b c x y z > P a b c x y z ∧ 
  P a b c x y z > N a b c x y z ∧ 
  M a b c x y z > Q a b c x y z ∧ 
  Q a b c x y z > N a b c x y z :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3333_333397


namespace NUMINAMATH_CALUDE_john_needs_72_strings_l3333_333339

/-- Calculates the total number of strings needed for restringing instruments --/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string * strings_per_8string

theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_72_strings_l3333_333339


namespace NUMINAMATH_CALUDE_basketball_team_size_l3333_333313

theorem basketball_team_size 
  (total_score : ℕ) 
  (min_score : ℕ) 
  (max_score : ℕ) 
  (h1 : total_score = 100) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) :
  ∃ (team_size : ℕ), 
    team_size * min_score ≤ total_score ∧ 
    total_score ≤ (team_size - 1) * min_score + max_score ∧
    team_size = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_size_l3333_333313


namespace NUMINAMATH_CALUDE_loggerhead_turtle_eggs_per_nest_l3333_333351

/-- The average number of eggs per nest for loggerhead turtles -/
def average_eggs_per_nest (total_eggs : ℕ) (total_nests : ℕ) : ℚ :=
  total_eggs / total_nests

/-- Theorem: The average number of eggs per nest is 150 -/
theorem loggerhead_turtle_eggs_per_nest :
  average_eggs_per_nest 3000000 20000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_loggerhead_turtle_eggs_per_nest_l3333_333351


namespace NUMINAMATH_CALUDE_sin_3phi_from_exponential_l3333_333345

theorem sin_3phi_from_exponential (φ : ℝ) :
  Complex.exp (Complex.I * φ) = (1 + Complex.I * Real.sqrt 8) / 3 →
  Real.sin (3 * φ) = -5 * Real.sqrt 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_3phi_from_exponential_l3333_333345


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positives_l3333_333394

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_distinct_ab : a ≠ b) (h_distinct_ac : a ≠ c) (h_distinct_bc : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positives_l3333_333394


namespace NUMINAMATH_CALUDE_kelly_peanuts_weight_l3333_333338

/-- Given the total weight of snacks and the weight of raisins, 
    calculate the weight of peanuts Kelly bought. -/
theorem kelly_peanuts_weight 
  (total_snacks : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : total_snacks = 0.5) 
  (h2 : raisins_weight = 0.4) : 
  total_snacks - raisins_weight = 0.1 := by
  sorry

#check kelly_peanuts_weight

end NUMINAMATH_CALUDE_kelly_peanuts_weight_l3333_333338


namespace NUMINAMATH_CALUDE_unique_solutions_for_exponential_equation_l3333_333372

theorem unique_solutions_for_exponential_equation :
  ∀ x n : ℕ+, 3 * 2^(x : ℕ) + 4 = (n : ℕ)^2 ↔ (x = 2 ∧ n = 4) ∨ (x = 5 ∧ n = 10) ∨ (x = 6 ∧ n = 14) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_for_exponential_equation_l3333_333372


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3333_333300

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 36,
    diagonal_length := 48,
    longer_base := 60
  }
  area t = 1105.92 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3333_333300


namespace NUMINAMATH_CALUDE_total_chairs_is_59_l3333_333337

/-- The number of chairs in the office canteen -/
def total_chairs : ℕ :=
  let round_tables : ℕ := 3
  let rectangular_tables : ℕ := 4
  let square_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let chairs_per_square_table : ℕ := 4
  let extra_chairs : ℕ := 5
  (round_tables * chairs_per_round_table) +
  (rectangular_tables * chairs_per_rectangular_table) +
  (square_tables * chairs_per_square_table) +
  extra_chairs

/-- Theorem stating that the total number of chairs in the office canteen is 59 -/
theorem total_chairs_is_59 : total_chairs = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_is_59_l3333_333337


namespace NUMINAMATH_CALUDE_thursday_loaves_l3333_333370

def bakery_sequence : List ℕ := [5, 11, 10, 14, 19, 25]

def alternating_differences (seq : List ℕ) : List ℕ :=
  List.zipWith (λ a b => b - a) seq (seq.tail)

theorem thursday_loaves :
  let seq := bakery_sequence
  let diffs := alternating_differences seq
  (seq[1] = 11 ∧
   diffs[0] = diffs[2] + 1 ∧
   diffs[1] = diffs[3] - 1 ∧
   diffs[2] = diffs[4] + 1) →
  seq[1] = 11 := by sorry

end NUMINAMATH_CALUDE_thursday_loaves_l3333_333370


namespace NUMINAMATH_CALUDE_distance_between_chord_endpoints_l3333_333307

/-- In a circle with radius R, given two mutually perpendicular chords MN and PQ,
    where NQ = m, the distance between points M and P is √(4R² - m²). -/
theorem distance_between_chord_endpoints (R m : ℝ) (R_pos : R > 0) (m_pos : m > 0) :
  ∃ (M P : ℝ × ℝ),
    (∃ (N Q : ℝ × ℝ),
      (∀ (X : ℝ × ℝ), (X.1 - 0)^2 + (X.2 - 0)^2 = R^2 → 
        ((M.1 - N.1) * (P.1 - Q.1) + (M.2 - N.2) * (P.2 - Q.2) = 0) ∧
        ((N.1 - Q.1)^2 + (N.2 - Q.2)^2 = m^2)) →
      ((M.1 - P.1)^2 + (M.2 - P.2)^2 = 4 * R^2 - m^2)) :=
sorry

end NUMINAMATH_CALUDE_distance_between_chord_endpoints_l3333_333307


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3333_333399

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3333_333399


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3333_333325

def f (x : ℝ) := x^4 - 8*x^2 + 3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = min) ∧
    max = 3 ∧ min = -13 := by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3333_333325


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3333_333324

/-- The number of ways to distribute n distinct objects among k recipients -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a distribution satisfies the given conditions -/
def validDistribution (d : List ℕ) : Prop :=
  d.length = 3 ∧ d.sum = 8 ∧ d.all (· > 0) ∧ d.Nodup

theorem chocolate_distribution :
  sumOfDigits (distribute 8 3) = 24 :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3333_333324


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l3333_333328

/-- The trajectory of point P given the symmetry of points A and B and the product of slopes condition -/
theorem trajectory_of_point_P (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (1, -1)
  let P : ℝ × ℝ := (x, y)
  let slope_AP := (y - A.2) / (x - A.1)
  let slope_BP := (y - B.2) / (x - B.1)
  x ≠ 1 ∧ x ≠ -1 →
  slope_AP * slope_BP = 1/3 →
  3 * y^2 - x^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l3333_333328


namespace NUMINAMATH_CALUDE_baby_nexus_monograms_l3333_333322

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters to exclude (X and one other) --/
def excluded_letters : ℕ := 2

/-- The number of letters to choose for the monogram (first and middle initials) --/
def letters_to_choose : ℕ := 2

/-- Calculates the number of possible monograms for baby Nexus --/
def monogram_count : ℕ :=
  Nat.choose (alphabet_size - excluded_letters) letters_to_choose

theorem baby_nexus_monograms :
  monogram_count = 253 := by
  sorry

end NUMINAMATH_CALUDE_baby_nexus_monograms_l3333_333322


namespace NUMINAMATH_CALUDE_real_roots_imply_m_value_l3333_333311

theorem real_roots_imply_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 := by
sorry

end NUMINAMATH_CALUDE_real_roots_imply_m_value_l3333_333311


namespace NUMINAMATH_CALUDE_simplify_fraction_l3333_333309

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3333_333309


namespace NUMINAMATH_CALUDE_N_composite_and_three_factors_l3333_333381

def N (n : ℕ) : ℤ := n^4 - 90*n^2 - 91*n - 90

theorem N_composite_and_three_factors (n : ℕ) (h : n > 10) :
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N n = x * y * z) :=
sorry

end NUMINAMATH_CALUDE_N_composite_and_three_factors_l3333_333381


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l3333_333348

/-- The degree measure of an interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

theorem exterior_angle_measure :
  let square_angle : ℚ := 90
  let heptagon_angle : ℚ := interior_angle 7
  let exterior_angle : ℚ := 360 - heptagon_angle - square_angle
  exterior_angle = 990 / 7 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l3333_333348


namespace NUMINAMATH_CALUDE_narration_per_disc_l3333_333367

/-- Represents the duration of the narration in minutes -/
def narration_duration : ℕ := 6 * 60 + 45

/-- Represents the capacity of each disc in minutes -/
def disc_capacity : ℕ := 75

/-- Calculates the minimum number of discs needed -/
def min_discs : ℕ := (narration_duration + disc_capacity - 1) / disc_capacity

/-- Theorem stating the duration of narration on each disc -/
theorem narration_per_disc :
  (narration_duration : ℚ) / min_discs = 67.5 := by sorry

end NUMINAMATH_CALUDE_narration_per_disc_l3333_333367


namespace NUMINAMATH_CALUDE_area_of_three_semicircle_intersection_l3333_333320

/-- The area of intersection of three semicircles forming a square -/
theorem area_of_three_semicircle_intersection (r : ℝ) (h : r = 2) : 
  let square_side := 2 * r
  let square_area := square_side ^ 2
  square_area = 16 := by sorry

end NUMINAMATH_CALUDE_area_of_three_semicircle_intersection_l3333_333320


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l3333_333343

theorem smallest_advantageous_discount : ∃ (n : ℕ), n = 29 ∧ 
  (∀ (x : ℝ), x > 0 → 
    (1 - n / 100) * x < (1 - 0.12) * (1 - 0.18) * x ∧
    (1 - n / 100) * x < (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∧
    (1 - n / 100) * x < (1 - 0.20) * (1 - 0.10) * x) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (x : ℝ), x > 0 ∧
      ((1 - m / 100) * x ≥ (1 - 0.12) * (1 - 0.18) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.20) * (1 - 0.10) * x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l3333_333343


namespace NUMINAMATH_CALUDE_book_selection_problem_l3333_333350

theorem book_selection_problem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 4) :
  (Nat.choose (n - 1) k) = (Nat.choose (n - 1) (m - 1)) :=
sorry

end NUMINAMATH_CALUDE_book_selection_problem_l3333_333350


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3333_333336

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p ∧ p ∣ (2^14 + 7^12) ∧ ∀ q, Prime q → q ∣ (2^14 + 7^12) → p ≤ q → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3333_333336


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3333_333384

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_mono : is_monotonically_increasing a)
  (h_prod : a 1 * a 9 = 64)
  (h_sum : a 3 + a 7 = 20) :
  a 11 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3333_333384


namespace NUMINAMATH_CALUDE_expression_evaluation_l3333_333310

theorem expression_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x^2 - 2*x) - 1 / (x^2 - 4*x + 4)) / (2 / (x^2 - 2*x))
  expr = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3333_333310


namespace NUMINAMATH_CALUDE_angle_A_is_45_degrees_l3333_333391

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- The sum of angles in a triangle is 180°
  A + B + C = Real.pi ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- State the theorem
theorem angle_A_is_45_degrees :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = Real.pi / 3 →
  A = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_is_45_degrees_l3333_333391


namespace NUMINAMATH_CALUDE_find_genuine_coins_l3333_333377

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := Fin 7 → Coin

/-- A weighing function that compares two sets of coins -/
def weigh (coins : CoinSet) (left right : List (Fin 7)) : WeighResult :=
  sorry

/-- Checks if a given set of coins contains exactly 3 genuine coins -/
def isValidResult (coins : CoinSet) (result : List (Fin 7)) : Prop :=
  result.length = 3 ∧ ∀ i ∈ result.toFinset, coins i = Coin.Genuine

/-- The main theorem stating that it's possible to find 3 genuine coins in two weighings -/
theorem find_genuine_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 7), i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit)
  (h2 : ∀ (i : Fin 7), coins i ≠ Coin.Counterfeit → coins i = Coin.Genuine)
  : ∃ (w1 w2 : List (Fin 7) × List (Fin 7)) (result : List (Fin 7)),
    isValidResult coins result ∧ 
    (∀ (c1 c2 : CoinSet), 
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c1 i = Coin.Genuine) →
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c2 i = Coin.Genuine) →
      weigh c1 w1.1 w1.2 = weigh c2 w1.1 w1.2 →
      weigh c1 w2.1 w2.2 = weigh c2 w2.1 w2.2 →
      (∀ (i : Fin 7), i ∈ result → c1 i = Coin.Genuine)) :=
sorry

end NUMINAMATH_CALUDE_find_genuine_coins_l3333_333377


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l3333_333317

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n+2 => validSequences (n+1) + validSequences n

/-- The length of the sequence -/
def sequenceLength : ℕ := 12

/-- The total number of possible sequences -/
def totalSequences : ℕ := 2^sequenceLength

theorem probability_no_consecutive_ones :
  (validSequences sequenceLength : ℚ) / totalSequences = 377 / 4096 := by
  sorry

#eval validSequences sequenceLength + totalSequences

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l3333_333317


namespace NUMINAMATH_CALUDE_vessel_base_length_l3333_333364

/-- The length of the base of a vessel given specific conditions -/
theorem vessel_base_length : ∀ (breadth rise cube_edge : ℝ),
  breadth = 30 →
  rise = 15 →
  cube_edge = 30 →
  (cube_edge ^ 3) = breadth * rise * 60 :=
by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l3333_333364


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3333_333344

/-- Represents the types of figures in the pattern -/
inductive Figure
| Triangle
| Square

/-- Represents a point on the line ℓ -/
structure PointOnLine where
  position : ℝ

/-- Represents the infinite pattern on line ℓ -/
def Pattern := ℕ → Figure

/-- Represents the possible rigid motion transformations -/
inductive RigidMotion
| Rotation (center : PointOnLine) (angle : ℝ)
| Translation (distance : ℝ)
| ReflectionAcrossL
| ReflectionPerpendicular (point : PointOnLine)

/-- Defines the alternating pattern of triangles and squares -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Figure.Triangle else Figure.Square

/-- Checks if a rigid motion preserves the pattern -/
def preservesPattern (motion : RigidMotion) (pattern : Pattern) : Prop :=
  ∀ n, pattern n = pattern (n + 1)  -- This is a simplification; actual preservation would be more complex

/-- The main theorem stating that only translation preserves the pattern -/
theorem only_translation_preserves_pattern :
  ∀ motion : RigidMotion,
    preservesPattern motion alternatingPattern ↔ ∃ d, motion = RigidMotion.Translation d :=
sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3333_333344


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l3333_333326

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 14 → d^2 = l^2 + w^2 + h^2 → w^2 = 122 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l3333_333326


namespace NUMINAMATH_CALUDE_largest_base_5_to_base_7_l3333_333306

/-- The largest four-digit number in base-5 -/
def m : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Conversion of a natural number to its base-7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  sorry

theorem largest_base_5_to_base_7 :
  to_base_7 m = [1, 5, 5, 1] :=
sorry

end NUMINAMATH_CALUDE_largest_base_5_to_base_7_l3333_333306


namespace NUMINAMATH_CALUDE_b_value_in_discriminant_l3333_333386

/-- For a quadratic equation ax^2 + bx + c = 0, 
    the discriminant is defined as b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 2x - 3 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem b_value_in_discriminant :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x ↔ a*x^2 + b*x + c = 0) ∧
    b = -2 :=
sorry

end NUMINAMATH_CALUDE_b_value_in_discriminant_l3333_333386


namespace NUMINAMATH_CALUDE_towels_folded_in_one_hour_l3333_333303

/-- Represents the number of towels a person can fold in one hour --/
def towels_per_hour (
  jane_rate : ℕ → ℕ
) (
  kyla_rate : ℕ → ℕ
) (
  anthony_rate : ℕ → ℕ
) (
  david_rate : ℕ → ℕ
) : ℕ :=
  jane_rate 60 + kyla_rate 60 + anthony_rate 60 + david_rate 60

/-- Jane's folding rate: 5 towels in 5 minutes, 3-minute break after every 5 minutes --/
def jane_rate (minutes : ℕ) : ℕ :=
  (minutes / 8) * 5

/-- Kyla's folding rate: 12 towels in 10 minutes for first 30 minutes, then 6 towels in 10 minutes --/
def kyla_rate (minutes : ℕ) : ℕ :=
  min 36 (minutes / 10 * 12) + max 0 ((minutes - 30) / 10 * 6)

/-- Anthony's folding rate: 14 towels in 20 minutes, 10-minute break after 40 minutes --/
def anthony_rate (minutes : ℕ) : ℕ :=
  (min minutes 40) / 20 * 14

/-- David's folding rate: 4 towels in 15 minutes, speed increases by 1 towel per 15 minutes for every 3 sets --/
def david_rate (minutes : ℕ) : ℕ :=
  (minutes / 15) * 4 + (minutes / 45)

theorem towels_folded_in_one_hour :
  towels_per_hour jane_rate kyla_rate anthony_rate david_rate = 134 := by
  sorry

end NUMINAMATH_CALUDE_towels_folded_in_one_hour_l3333_333303


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3333_333398

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^2 + 8 = (x - 1)*(x^5 + x^4 + x^3 + x^2 + 3*x + 3) + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3333_333398


namespace NUMINAMATH_CALUDE_total_football_games_l3333_333314

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- Theorem: The total number of football games Joan went to is 13 -/
theorem total_football_games : games_this_year + games_last_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l3333_333314


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l3333_333332

theorem triangle_angle_ratio (A B C : ℝ) (x : ℝ) : 
  B = x * A →
  C = A + 12 →
  A = 24 →
  A + B + C = 180 →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l3333_333332


namespace NUMINAMATH_CALUDE_exponent_problem_l3333_333375

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 1/4) :
  x^(2*m - n) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l3333_333375


namespace NUMINAMATH_CALUDE_sum_f_negative_l3333_333369

/-- The function f(x) = 2x³ + 4x -/
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

/-- Theorem: Given f(x) = 2x³ + 4x and a + b < 0, b + c < 0, c + a < 0, then f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) :
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3333_333369


namespace NUMINAMATH_CALUDE_area_of_second_square_l3333_333373

/-- A right isosceles triangle with two inscribed squares -/
structure RightIsoscelesTriangleWithSquares where
  -- The side length of the triangle
  b : ℝ
  -- The side length of the first inscribed square (ADEF)
  a₁ : ℝ
  -- The side length of the second inscribed square (GHIJ)
  a : ℝ
  -- The first square is inscribed in the triangle
  h_a₁_inscribed : a₁ = b / 2
  -- The second square is inscribed in the triangle
  h_a_inscribed : a = (2 * b ^ 2) / (3 * b * Real.sqrt 2)

/-- The theorem statement -/
theorem area_of_second_square (t : RightIsoscelesTriangleWithSquares) 
    (h_area_first : t.a₁ ^ 2 = 2250) : 
    t.a ^ 2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_area_of_second_square_l3333_333373


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l3333_333387

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ width : ℝ, width = 40 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_spacing :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l3333_333387


namespace NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l3333_333368

theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l3333_333368


namespace NUMINAMATH_CALUDE_tan_equality_in_range_l3333_333388

theorem tan_equality_in_range : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (286 * π / 180) ∧ n = -74 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_in_range_l3333_333388


namespace NUMINAMATH_CALUDE_caravan_hens_count_l3333_333304

/-- A caravan with hens, goats, camels, and keepers. -/
structure Caravan where
  hens : ℕ
  goats : ℕ
  camels : ℕ
  keepers : ℕ

/-- Calculate the total number of feet in the caravan. -/
def totalFeet (c : Caravan) : ℕ :=
  2 * c.hens + 4 * c.goats + 4 * c.camels + 2 * c.keepers

/-- Calculate the total number of heads in the caravan. -/
def totalHeads (c : Caravan) : ℕ :=
  c.hens + c.goats + c.camels + c.keepers

/-- The main theorem stating the number of hens in the caravan. -/
theorem caravan_hens_count : ∃ (c : Caravan), 
  c.goats = 45 ∧ 
  c.camels = 8 ∧ 
  c.keepers = 15 ∧ 
  totalFeet c = totalHeads c + 224 ∧ 
  c.hens = 50 := by
  sorry


end NUMINAMATH_CALUDE_caravan_hens_count_l3333_333304


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3333_333308

theorem smallest_marble_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 10 = 0 ∧ 
  (n - 10) % 7 = 0 ∧
  n = 143 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → (m + 7) % 10 = 0 → (m - 10) % 7 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3333_333308


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3333_333329

/-- Calculates the lowest possible sale price of a jersey as a percentage of its list price -/
theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_sale_discount : ℝ) :
  list_price = 80 ∧ 
  max_regular_discount = 0.5 ∧ 
  summer_sale_discount = 0.2 →
  (list_price * (1 - max_regular_discount) - list_price * summer_sale_discount) / list_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3333_333329


namespace NUMINAMATH_CALUDE_remaining_episodes_l3333_333323

theorem remaining_episodes (seasons : Nat) (episodes_per_season : Nat) 
  (watched_fraction : Rat) (h1 : seasons = 12) (h2 : episodes_per_season = 20) 
  (h3 : watched_fraction = 1/3) : 
  seasons * episodes_per_season - (seasons * episodes_per_season * watched_fraction).floor = 160 := by
  sorry

end NUMINAMATH_CALUDE_remaining_episodes_l3333_333323


namespace NUMINAMATH_CALUDE_soap_cost_per_pound_l3333_333305

theorem soap_cost_per_pound 
  (num_bars : ℕ) 
  (weight_per_bar : ℝ) 
  (total_cost : ℝ) 
  (h1 : num_bars = 20)
  (h2 : weight_per_bar = 1.5)
  (h3 : total_cost = 15) : 
  total_cost / (num_bars * weight_per_bar) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_soap_cost_per_pound_l3333_333305


namespace NUMINAMATH_CALUDE_weight_increase_percentage_shyam_weight_increase_percentage_l3333_333356

theorem weight_increase_percentage (ram_ratio : ℝ) (shyam_ratio : ℝ) 
  (ram_increase : ℝ) (total_weight : ℝ) (total_increase : ℝ) : ℝ :=
  let original_total := total_weight / (1 + total_increase / 100)
  let x := original_total / (ram_ratio + shyam_ratio)
  let ram_original := ram_ratio * x
  let shyam_original := shyam_ratio * x
  let ram_new := ram_original * (1 + ram_increase / 100)
  let shyam_new := total_weight - ram_new
  (shyam_new - shyam_original) / shyam_original * 100

/-- Given the weights of Ram and Shyam in a 7:5 ratio, Ram's weight increased by 10%,
    and the total weight after increase is 82.8 kg with a 15% total increase,
    prove that Shyam's weight increase percentage is 22%. -/
theorem shyam_weight_increase_percentage :
  weight_increase_percentage 7 5 10 82.8 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_weight_increase_percentage_shyam_weight_increase_percentage_l3333_333356


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3333_333393

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + k^2 - 1 = 0) ↔ k ∈ Set.Icc (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3333_333393


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3333_333363

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_set (s : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ s = {x | a ≤ x ∧ x ≤ b ∧ is_odd x ∧ ∀ y, a ≤ y ∧ y < x → is_odd y}

def median (s : Set ℤ) : ℤ := sorry

theorem smallest_integer_in_set (s : Set ℤ) :
  is_consecutive_odd_set s ∧ median s = 153 ∧ (∃ x ∈ s, ∀ y ∈ s, y ≤ x) ∧ 167 ∈ s →
  (∃ z ∈ s, ∀ w ∈ s, z ≤ w) ∧ 139 ∈ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3333_333363


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l3333_333321

/-- Given a rectangular solid with adjacent face areas √2, √3, and √6,
    the surface area of its circumscribed sphere is 6π. -/
theorem circumscribed_sphere_area (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  4 * Real.pi * ((Real.sqrt 6) / 2)^2 = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_circumscribed_sphere_area_l3333_333321


namespace NUMINAMATH_CALUDE_red_cars_count_l3333_333327

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 28 := by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l3333_333327


namespace NUMINAMATH_CALUDE_product_18396_9999_l3333_333365

theorem product_18396_9999 : 18396 * 9999 = 183962604 := by sorry

end NUMINAMATH_CALUDE_product_18396_9999_l3333_333365


namespace NUMINAMATH_CALUDE_root_equation_q_value_l3333_333358

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by sorry

end NUMINAMATH_CALUDE_root_equation_q_value_l3333_333358


namespace NUMINAMATH_CALUDE_optimal_portfolio_l3333_333380

/-- Represents an investment project with maximum profit and loss percentages -/
structure Project where
  max_profit : Real
  max_loss : Real

/-- Represents an investment portfolio with amounts invested in two projects -/
structure Portfolio where
  amount_a : Real
  amount_b : Real

def project_a : Project := { max_profit := 1.0, max_loss := 0.3 }
def project_b : Project := { max_profit := 0.5, max_loss := 0.1 }

def total_investment_limit : Real := 100000
def max_allowed_loss : Real := 18000

def portfolio_loss (p : Portfolio) : Real :=
  p.amount_a * project_a.max_loss + p.amount_b * project_b.max_loss

def portfolio_profit (p : Portfolio) : Real :=
  p.amount_a * project_a.max_profit + p.amount_b * project_b.max_profit

def is_valid_portfolio (p : Portfolio) : Prop :=
  p.amount_a ≥ 0 ∧ p.amount_b ≥ 0 ∧
  p.amount_a + p.amount_b ≤ total_investment_limit ∧
  portfolio_loss p ≤ max_allowed_loss

theorem optimal_portfolio :
  ∃ (p : Portfolio), is_valid_portfolio p ∧
    ∀ (q : Portfolio), is_valid_portfolio q → portfolio_profit q ≤ portfolio_profit p :=
  sorry

end NUMINAMATH_CALUDE_optimal_portfolio_l3333_333380


namespace NUMINAMATH_CALUDE_only_thirteen_fourths_between_three_and_four_l3333_333340

theorem only_thirteen_fourths_between_three_and_four :
  let numbers : List ℚ := [5/2, 11/4, 11/5, 13/4, 13/5]
  ∀ x ∈ numbers, (3 < x ∧ x < 4) ↔ x = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_only_thirteen_fourths_between_three_and_four_l3333_333340


namespace NUMINAMATH_CALUDE_highway_problem_l3333_333374

-- Define the speeds and distances
def yi_initial_speed : ℝ := 60
def speed_reduction_jia : ℝ := 0.4
def speed_reduction_yi : ℝ := 0.25
def time_jia_to_bing : ℝ := 9
def extra_distance_yi : ℝ := 50

-- Define the theorem
theorem highway_problem :
  ∃ (jia_initial_speed : ℝ) (distance_AD : ℝ),
    jia_initial_speed = 125 ∧ distance_AD = 1880 := by
  sorry


end NUMINAMATH_CALUDE_highway_problem_l3333_333374


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l3333_333376

/-- Calculate the revenue of a tanning salon for a month --/
theorem tanning_salon_revenue :
  let first_visit_charge : ℚ := 10
  let subsequent_visit_charge : ℚ := 8
  let discount_rate : ℚ := 0.1
  let premium_service_charge : ℚ := 15
  let premium_service_rate : ℚ := 0.2
  let total_customers : ℕ := 150
  let second_visit_customers : ℕ := 40
  let third_visit_customers : ℕ := 15
  let fourth_visit_customers : ℕ := 5

  let first_visit_revenue : ℚ := 
    (premium_service_rate * total_customers.cast) * premium_service_charge +
    ((1 - premium_service_rate) * total_customers.cast) * first_visit_charge
  let second_visit_revenue : ℚ := second_visit_customers.cast * subsequent_visit_charge
  let discounted_visit_charge : ℚ := subsequent_visit_charge * (1 - discount_rate)
  let third_visit_revenue : ℚ := third_visit_customers.cast * discounted_visit_charge
  let fourth_visit_revenue : ℚ := fourth_visit_customers.cast * discounted_visit_charge

  let total_revenue : ℚ := 
    first_visit_revenue + second_visit_revenue + third_visit_revenue + fourth_visit_revenue

  total_revenue = 2114 := by sorry

end NUMINAMATH_CALUDE_tanning_salon_revenue_l3333_333376


namespace NUMINAMATH_CALUDE_inequality_properties_l3333_333355

theorem inequality_properties (m n : ℝ) : 
  (∀ a : ℝ, a > 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_properties_l3333_333355


namespace NUMINAMATH_CALUDE_number_equation_l3333_333353

theorem number_equation : ∃ x : ℝ, x * (37 - 15) - 25 = 327 :=
by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3333_333353


namespace NUMINAMATH_CALUDE_q_minimized_at_2_l3333_333312

/-- The quadratic function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- The value of x that minimizes q -/
def minimizing_x : ℝ := 2

theorem q_minimized_at_2 :
  ∀ x : ℝ, q x ≥ q minimizing_x :=
sorry

end NUMINAMATH_CALUDE_q_minimized_at_2_l3333_333312


namespace NUMINAMATH_CALUDE_min_value_expression_l3333_333316

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) :
  (4 / (x + 3 * y)) + (1 / (x - y)) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3333_333316


namespace NUMINAMATH_CALUDE_paul_weed_eating_money_l3333_333362

/-- The amount of money Paul made weed eating -/
def weed_eating_money (mowing_money weekly_spending weeks_lasted : ℕ) : ℕ :=
  weekly_spending * weeks_lasted - mowing_money

/-- Theorem stating that Paul made $28 weed eating -/
theorem paul_weed_eating_money :
  weed_eating_money 44 9 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_weed_eating_money_l3333_333362


namespace NUMINAMATH_CALUDE_tournament_rankings_l3333_333382

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(no_ties : Bool)

/-- Represents the final ranking of teams -/
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)
(fifth : Team)
(sixth : Team)

/-- Counts the number of possible ranking sequences for the given tournament -/
def countPossibleRankings (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) 
  (h1 : t.saturday_matches = [Match.mk Team.A Team.B, Match.mk Team.C Team.D, Match.mk Team.E Team.F])
  (h2 : t.no_ties = true) : 
  countPossibleRankings t = 288 :=
sorry

end NUMINAMATH_CALUDE_tournament_rankings_l3333_333382


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_k_l3333_333315

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := by sorry

-- Theorem 2: Range of k for |k - 1| < g(x)
theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |k - 1| < g x) → -3 < k ∧ k < 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_k_l3333_333315


namespace NUMINAMATH_CALUDE_card_sum_theorem_l3333_333335

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l3333_333335


namespace NUMINAMATH_CALUDE_spurs_basketball_distribution_l3333_333360

theorem spurs_basketball_distribution (num_players : ℕ) (total_basketballs : ℕ) 
  (h1 : num_players = 22) 
  (h2 : total_basketballs = 242) : 
  total_basketballs / num_players = 11 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketball_distribution_l3333_333360


namespace NUMINAMATH_CALUDE_sequence_difference_l3333_333333

theorem sequence_difference (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, n > 0 → S n = n^2 + 2*n) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 4 - a 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3333_333333


namespace NUMINAMATH_CALUDE_zoo_animal_count_l3333_333378

theorem zoo_animal_count (initial_count : ℕ) (gorillas_sent : ℕ) (hippo_adopted : ℕ) 
  (rhinos_taken : ℕ) (final_count : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l3333_333378


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l3333_333359

theorem gcd_special_numbers : Nat.gcd 33333333 777777777 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l3333_333359


namespace NUMINAMATH_CALUDE_system_has_solution_l3333_333396

/-- Given a system of equations {sin x + a = b x, cos x = b} where a and b are real numbers,
    and the equation sin x + a = b x has exactly two solutions,
    prove that the system has at least one solution. -/
theorem system_has_solution (a b : ℝ) 
    (h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
         (∀ x, Real.sin x + a = b * x ↔ x = x₁ ∨ x = x₂)) :
  ∃ x, Real.sin x + a = b * x ∧ Real.cos x = b := by
  sorry

end NUMINAMATH_CALUDE_system_has_solution_l3333_333396


namespace NUMINAMATH_CALUDE_belle_weekly_treat_cost_l3333_333331

/-- The cost to feed Belle treats for a week -/
def weekly_treat_cost (dog_biscuits_per_day : ℕ) (rawhide_bones_per_day : ℕ) 
  (dog_biscuit_cost : ℚ) (rawhide_bone_cost : ℚ) (days_per_week : ℕ) : ℚ :=
  (dog_biscuits_per_day * dog_biscuit_cost + rawhide_bones_per_day * rawhide_bone_cost) * days_per_week

/-- Proof that Belle's weekly treat cost is $21.00 -/
theorem belle_weekly_treat_cost :
  weekly_treat_cost 4 2 0.25 1 7 = 21 := by
  sorry

#eval weekly_treat_cost 4 2 (1/4) 1 7

end NUMINAMATH_CALUDE_belle_weekly_treat_cost_l3333_333331


namespace NUMINAMATH_CALUDE_maya_shoe_probability_l3333_333318

/-- Represents the number of pairs for each shoe color --/
structure ShoePairs where
  black : Nat
  brown : Nat
  grey : Nat
  white : Nat

/-- Calculates the probability of picking two shoes of the same color,
    one left and one right, given a distribution of shoe pairs --/
def samePairColorProbability (pairs : ShoePairs) : Rat :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.grey + pairs.white)
  let numerator := pairs.black * pairs.black + pairs.brown * pairs.brown +
                   pairs.grey * pairs.grey + pairs.white * pairs.white
  numerator / (totalShoes * (totalShoes - 1))

/-- Maya's shoe collection --/
def mayasShoes : ShoePairs := ⟨8, 4, 3, 1⟩

theorem maya_shoe_probability :
  samePairColorProbability mayasShoes = 45 / 248 := by
  sorry

end NUMINAMATH_CALUDE_maya_shoe_probability_l3333_333318


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3333_333379

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((627 + y) % 510 = 0 ∧ (627 + y) % 4590 = 0 ∧ (627 + y) % 105 = 0)) ∧
  ((627 + x) % 510 = 0 ∧ (627 + x) % 4590 = 0 ∧ (627 + x) % 105 = 0) ∧
  x = 31503 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3333_333379


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3333_333352

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define set A
def SetA (x y m : ℝ) : Prop := (m + 3) * x + (m - 2) * y - 1 - 2 * m = 0

-- Define set B (tangent lines to the circle)
def SetB (x y : ℝ) : Prop := ∃ (a b : ℝ), Circle a b ∧ (x - a) * a + (y - b) * b = 0

-- Define the intersection set
def IntersectionSet (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_of_sets :
  ∀ (x y : ℝ), (∃ (m : ℝ), SetA x y m) ∧ SetB x y ↔ IntersectionSet x y :=
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3333_333352


namespace NUMINAMATH_CALUDE_floor_of_e_eq_two_l3333_333383

/-- The floor of Euler's number is 2 -/
theorem floor_of_e_eq_two : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_eq_two_l3333_333383


namespace NUMINAMATH_CALUDE_range_of_f_l3333_333390

-- Define the function f
def f : ℝ → ℝ := λ x => x^2 - 10*x - 4

-- State the theorem
theorem range_of_f :
  ∀ t : ℝ, t ∈ Set.Ioo 0 8 → ∃ y : ℝ, y ∈ Set.Icc (-29) (-4) ∧ y = f t ∧
  ∀ z : ℝ, z ∈ Set.Icc (-29) (-4) → ∃ s : ℝ, s ∈ Set.Ioo 0 8 ∧ z = f s :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3333_333390


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3333_333392

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3333_333392


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l3333_333385

/-- A geometric progression with positive terms where each term is the average of the next two terms plus 2 has a common ratio of 1. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2)) / 2 + 2) →  -- Condition on terms
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l3333_333385


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3333_333349

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship between major and minor axes
def major_axis_ratio : ℝ := 1.75

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis := 2 * cylinder_radius
  let major_axis := major_axis_ratio * minor_axis
  major_axis = 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3333_333349


namespace NUMINAMATH_CALUDE_island_ratio_l3333_333395

theorem island_ratio (centipedes humans sheep : ℕ) : 
  centipedes = 2 * humans →
  centipedes = 100 →
  sheep + humans = 75 →
  sheep.gcd humans = 25 →
  (sheep / 25 : ℚ) / (humans / 25 : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_island_ratio_l3333_333395


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l3333_333334

theorem stratified_sampling_size (high_school_students junior_high_students : ℕ) 
  (high_school_sample : ℕ) (total_sample : ℕ) : 
  high_school_students = 3500 →
  junior_high_students = 1500 →
  high_school_sample = 70 →
  (high_school_sample : ℚ) / high_school_students = 
    (total_sample : ℚ) / (high_school_students + junior_high_students) →
  total_sample = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l3333_333334


namespace NUMINAMATH_CALUDE_power_division_equals_integer_l3333_333366

theorem power_division_equals_integer : 3^18 / 27^2 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_integer_l3333_333366


namespace NUMINAMATH_CALUDE_mortdecai_charity_donation_l3333_333354

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs --/
def collection_days : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects per day --/
def collected_dozens_per_day : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def market_delivery : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mall_delivery : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pie_dozens : ℕ := 4

/-- Theorem stating that Mortdecai donates 48 eggs to charity --/
theorem mortdecai_charity_donation : 
  (collection_days * collected_dozens_per_day - (market_delivery + mall_delivery) - pie_dozens) * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_mortdecai_charity_donation_l3333_333354


namespace NUMINAMATH_CALUDE_calculate_expression_l3333_333342

theorem calculate_expression : 3000 * (3000 ^ 2999) * 2 ^ 3000 = 3000 ^ 3000 * 2 ^ 3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3333_333342


namespace NUMINAMATH_CALUDE_correct_categorization_l3333_333357

-- Define the teams
def IntegerTeam : Set ℝ := {0, -8}
def FractionTeam : Set ℝ := {1/7, 0.505}
def IrrationalTeam : Set ℝ := {Real.sqrt 13, Real.pi}

-- Define the properties for each team
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n
def isFraction (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def isIrrational (x : ℝ) : Prop := ¬(isInteger x ∨ isFraction x)

-- Theorem to prove the correct categorization
theorem correct_categorization :
  (∀ x ∈ IntegerTeam, isInteger x) ∧
  (∀ x ∈ FractionTeam, isFraction x) ∧
  (∀ x ∈ IrrationalTeam, isIrrational x) :=
  sorry

end NUMINAMATH_CALUDE_correct_categorization_l3333_333357


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3333_333371

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + (a+1)^2 = 0) ↔ (a ∈ Set.Icc (-2) (-2/3) ∧ a ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3333_333371


namespace NUMINAMATH_CALUDE_distance_to_soccer_is_12_l3333_333330

-- Define the distances and costs
def distance_to_grocery : ℝ := 8
def distance_to_school : ℝ := 6
def miles_per_gallon : ℝ := 25
def cost_per_gallon : ℝ := 2.5
def total_gas_cost : ℝ := 5

-- Define the unknown distance to soccer practice
def distance_to_soccer : ℝ → ℝ := λ x => x

-- Define the total distance driven
def total_distance (x : ℝ) : ℝ :=
  distance_to_grocery + distance_to_school + distance_to_soccer x + 2 * distance_to_soccer x

-- Theorem stating that the distance to soccer practice is 12 miles
theorem distance_to_soccer_is_12 :
  ∃ x : ℝ, distance_to_soccer x = 12 ∧ 
    total_distance x = (total_gas_cost / cost_per_gallon) * miles_per_gallon := by
  sorry

end NUMINAMATH_CALUDE_distance_to_soccer_is_12_l3333_333330


namespace NUMINAMATH_CALUDE_curve_is_ellipse_l3333_333302

/-- Given real numbers a and b where ab ≠ 0, the curve bx² + ay² = ab represents an ellipse. -/
theorem curve_is_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A^2 + y^2 / B^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_l3333_333302


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3333_333361

/-- Represents the number of villages in each category -/
structure VillageCategories where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample sizes for each category -/
structure SampleSizes where
  first : ℕ
  secondAndThird : ℕ

/-- Checks if the sampling is stratified -/
def isStratifiedSampling (vc : VillageCategories) (ss : SampleSizes) : Prop :=
  (ss.first : ℚ) / vc.first = (ss.first + ss.secondAndThird : ℚ) / vc.total

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (vc : VillageCategories) (ss : SampleSizes) :
  vc.total = 300 →
  vc.first = 60 →
  vc.second = 100 →
  vc.third = vc.total - vc.first - vc.second →
  ss.first = 3 →
  isStratifiedSampling vc ss →
  ss.secondAndThird = 12 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3333_333361


namespace NUMINAMATH_CALUDE_building_has_at_least_43_floors_l3333_333341

/-- Represents a building with apartments -/
structure Building where
  apartments_per_floor : ℕ
  kolya_floor : ℕ
  kolya_apartment : ℕ
  vasya_floor : ℕ
  vasya_apartment : ℕ

/-- The specific building described in the problem -/
def problem_building : Building :=
  { apartments_per_floor := 4
  , kolya_floor := 5
  , kolya_apartment := 83
  , vasya_floor := 3
  , vasya_apartment := 169
  }

/-- Calculates the minimum number of floors in the building -/
def min_floors (b : Building) : ℕ :=
  ((b.vasya_apartment - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that the building has at least 43 floors -/
theorem building_has_at_least_43_floors :
  min_floors problem_building ≥ 43 := by
  sorry


end NUMINAMATH_CALUDE_building_has_at_least_43_floors_l3333_333341


namespace NUMINAMATH_CALUDE_vector_norm_sum_l3333_333346

theorem vector_norm_sum (a b : ℝ × ℝ) :
  let m := (2 * a.1 + b.1, 2 * a.2 + b.2) / 2
  m = (-1, 5) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_vector_norm_sum_l3333_333346


namespace NUMINAMATH_CALUDE_eighth_prime_is_19_l3333_333319

/-- Natural numbers are non-negative integers -/
def NaturalNumber (n : ℕ) : Prop := True

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d > 0 → d < p → p % d ≠ 0

/-- The nth prime number -/
def NthPrime (n : ℕ) : ℕ :=
  sorry

theorem eighth_prime_is_19 : NthPrime 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_eighth_prime_is_19_l3333_333319


namespace NUMINAMATH_CALUDE_probability_heart_then_club_l3333_333347

theorem probability_heart_then_club (total_cards : Nat) (hearts : Nat) (clubs : Nat) :
  total_cards = 52 →
  hearts = 13 →
  clubs = 13 →
  (hearts : ℚ) / total_cards * clubs / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_club_l3333_333347


namespace NUMINAMATH_CALUDE_function_characterization_l3333_333389

theorem function_characterization (f : ℕ → ℕ) 
  (h_increasing : ∀ x y : ℕ, x ≤ y → f x ≤ f y)
  (h_square1 : ∀ n : ℕ, ∃ k : ℕ, f n + n + 1 = k^2)
  (h_square2 : ∀ n : ℕ, ∃ k : ℕ, f (f n) - f n = k^2) :
  ∀ x : ℕ, f x = x^2 + x :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3333_333389


namespace NUMINAMATH_CALUDE_painting_area_is_1836_l3333_333301

/-- The area of a rectangular painting within a frame -/
def painting_area (frame_width outer_length outer_width : ℝ) : ℝ :=
  (outer_length - 2 * frame_width) * (outer_width - 2 * frame_width)

/-- Theorem: The area of the painting is 1836 cm² -/
theorem painting_area_is_1836 :
  painting_area 8 70 50 = 1836 := by
  sorry

end NUMINAMATH_CALUDE_painting_area_is_1836_l3333_333301
