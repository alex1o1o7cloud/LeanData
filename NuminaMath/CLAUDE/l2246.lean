import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_octagon_area_inscribed_octagon_area_is_1400_l2246_224608

/-- The area of an inscribed octagon in a square -/
theorem inscribed_octagon_area (square_perimeter : ℝ) (h1 : square_perimeter = 160) : ℝ :=
  let square_side := square_perimeter / 4
  let triangle_leg := square_side / 4
  let triangle_area := (1 / 2) * triangle_leg * triangle_leg
  let total_triangle_area := 4 * triangle_area
  let square_area := square_side * square_side
  square_area - total_triangle_area

/-- The area of the inscribed octagon is 1400 square centimeters -/
theorem inscribed_octagon_area_is_1400 (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  inscribed_octagon_area square_perimeter h1 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_inscribed_octagon_area_is_1400_l2246_224608


namespace NUMINAMATH_CALUDE_average_first_five_multiples_of_five_l2246_224676

/-- The average of the first 5 multiples of 5 is 15 -/
theorem average_first_five_multiples_of_five : 
  (List.sum (List.map (· * 5) (List.range 5))) / 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_first_five_multiples_of_five_l2246_224676


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_12_825_l2246_224647

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α
  pos : 0 < side

/-- Represents the configuration of three squares aligned on their bottom edges -/
structure SquareConfiguration (α : Type*) [LinearOrderedField α] where
  small : Square α
  medium : Square α
  large : Square α
  alignment : small.side + medium.side + large.side > 0

/-- Calculates the area of the quadrilateral formed in the square configuration -/
noncomputable def quadrilateralArea {α : Type*} [LinearOrderedField α] (config : SquareConfiguration α) : α :=
  sorry

/-- Theorem stating that the area of the quadrilateral in the given configuration is 12.825 -/
theorem quadrilateral_area_is_12_825 :
  let config : SquareConfiguration ℝ := {
    small := { side := 3, pos := by norm_num },
    medium := { side := 5, pos := by norm_num },
    large := { side := 7, pos := by norm_num },
    alignment := by norm_num
  }
  quadrilateralArea config = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_12_825_l2246_224647


namespace NUMINAMATH_CALUDE_intersection_point_correct_l2246_224604

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The first line: y = 3x + 4 -/
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 4

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The point through which the perpendicular line passes -/
def point : (ℚ × ℚ) := (3, 2)

/-- The perpendicular line passing through (3, 2) -/
def line₂ (x y : ℚ) : Prop := y - point.2 = m₂ * (x - point.1)

/-- The intersection point of the two lines -/
def intersection_point : (ℚ × ℚ) := (-3/10, 31/10)

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l2246_224604


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2246_224609

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2246_224609


namespace NUMINAMATH_CALUDE_derived_series_divergence_l2246_224644

/-- Given a divergent series with positive nonincreasing terms, 
    the derived series also diverges. -/
theorem derived_series_divergence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_nonincr : ∀ n, a (n + 1) ≤ a n) 
  (h_diverge : ¬ Summable a) :
  ¬ Summable (fun n ↦ a n / (1 + n * a n)) := by
  sorry

end NUMINAMATH_CALUDE_derived_series_divergence_l2246_224644


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l2246_224686

theorem nested_sqrt_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l2246_224686


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_50_l2246_224610

/-- Represents a triangle with specific side lengths -/
structure Triangle where
  left_side : ℝ
  right_side : ℝ
  base : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.left_side + t.right_side + t.base

/-- Theorem: The perimeter of a triangle with given conditions is 50 cm -/
theorem triangle_perimeter_is_50 :
  ∀ t : Triangle,
    t.left_side = 12 →
    t.right_side = t.left_side + 2 →
    t.base = 24 →
    perimeter t = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_50_l2246_224610


namespace NUMINAMATH_CALUDE_exists_valid_number_l2246_224698

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, (n / 10^i) % 10 ≠ 0)

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem exists_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ (n + reverse_number n) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_number_l2246_224698


namespace NUMINAMATH_CALUDE_track_length_satisfies_conditions_l2246_224670

/-- The length of a circular track satisfying the given conditions -/
def track_length : ℝ := 766.67

/-- Two runners on a circular track -/
structure Runners :=
  (track_length : ℝ)
  (initial_separation : ℝ)
  (first_meeting_distance : ℝ)
  (second_meeting_distance : ℝ)

/-- The conditions of the problem -/
def problem_conditions (r : Runners) : Prop :=
  r.initial_separation = 0.75 * r.track_length ∧
  r.first_meeting_distance = 120 ∧
  r.second_meeting_distance = 180

/-- The theorem stating that the track length satisfies the problem conditions -/
theorem track_length_satisfies_conditions :
  ∃ (r : Runners), r.track_length = track_length ∧ problem_conditions r :=
sorry

end NUMINAMATH_CALUDE_track_length_satisfies_conditions_l2246_224670


namespace NUMINAMATH_CALUDE_positive_expression_l2246_224628

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  0 < b + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2246_224628


namespace NUMINAMATH_CALUDE_factor_of_quadratic_l2246_224656

theorem factor_of_quadratic (m x : ℤ) : 
  (∃ k : ℤ, (m - x) * k = m^2 - 5*m - 24) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_of_quadratic_l2246_224656


namespace NUMINAMATH_CALUDE_logic_propositions_l2246_224662

-- Define the propositions
def corresponding_angles_equal (l₁ l₂ : Line) : Prop := sorry
def lines_parallel (l₁ l₂ : Line) : Prop := sorry

-- Define the sine function and angle measure
def sin : ℝ → ℝ := sorry
def degree : ℝ → ℝ := sorry

-- Define the theorem
theorem logic_propositions :
  -- 1. Contrapositive
  (∀ l₁ l₂ : Line, (corresponding_angles_equal l₁ l₂ → lines_parallel l₁ l₂) ↔ 
    (¬lines_parallel l₁ l₂ → ¬corresponding_angles_equal l₁ l₂)) ∧
  -- 2. Necessary but not sufficient condition
  (∀ α : ℝ, sin α = 1/2 → degree α = 30) ∧
  (∃ β : ℝ, sin β = 1/2 ∧ degree β ≠ 30) ∧
  -- 3. Falsity of conjunction
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- 4. Negation of existence
  (¬(∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_logic_propositions_l2246_224662


namespace NUMINAMATH_CALUDE_fred_newspaper_earnings_l2246_224607

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings (total_earnings washing_earnings : ℕ) : ℕ :=
  total_earnings - washing_earnings

theorem fred_newspaper_earnings :
  newspaper_earnings 90 74 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fred_newspaper_earnings_l2246_224607


namespace NUMINAMATH_CALUDE_inequality_proof_l2246_224614

theorem inequality_proof (a₁ a₂ a₃ S : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1)
  (hS : S = a₁ + a₂ + a₃)
  (hₐ₁ : a₁^2 / (a₁ - 1) > S)
  (hₐ₂ : a₂^2 / (a₂ - 1) > S)
  (hₐ₃ : a₃^2 / (a₃ - 1) > S) :
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2246_224614


namespace NUMINAMATH_CALUDE_min_value_of_b_in_geometric_sequence_l2246_224627

theorem min_value_of_b_in_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, (a = b / r ∧ c = b * r) ∨ (a = b * r ∧ c = b / r)) →  -- geometric sequence condition
  ((a = 1 ∧ c = 4) ∨ (a = 4 ∧ c = 1) ∨ (a = 1 ∧ b = 4) ∨ (a = 4 ∧ b = 1) ∨ (b = 1 ∧ c = 4) ∨ (b = 4 ∧ c = 1)) →  -- 1 and 4 are in the sequence
  b ≥ -2 ∧ ∃ b₀ : ℝ, b₀ = -2 ∧ 
    (∃ r : ℝ, (b₀ = b₀ / r ∧ 4 = b₀ * r) ∨ (1 = b₀ * r ∧ 4 = b₀ / r)) ∧
    ((1 = 1 ∧ 4 = 4) ∨ (1 = 4 ∧ 4 = 1) ∨ (1 = 1 ∧ b₀ = 4) ∨ (1 = 4 ∧ b₀ = 1) ∨ (b₀ = 1 ∧ 4 = 4) ∨ (b₀ = 4 ∧ 4 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_b_in_geometric_sequence_l2246_224627


namespace NUMINAMATH_CALUDE_largest_intersection_point_l2246_224663

-- Define the polynomial P(x)
def P (x b : ℝ) : ℝ := x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3

-- Define the line L(x)
def L (x c d : ℝ) : ℝ := c*x - d

-- Theorem statement
theorem largest_intersection_point (b c d : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, P x b = L x c d ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max ∧ x_max = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_largest_intersection_point_l2246_224663


namespace NUMINAMATH_CALUDE_valid_numbers_l2246_224640

def is_valid_number (n : ℕ) : Prop :=
  500 < n ∧ n < 2500 ∧ n % 180 = 0 ∧ n % 75 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 900 ∨ n = 1800 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2246_224640


namespace NUMINAMATH_CALUDE_always_close_piece_l2246_224650

-- Define the grid structure
structure Grid :=
  (points : Set (ℤ × ℤ))
  (adjacent : (ℤ × ℤ) → Set (ℤ × ℤ))
  (initial : ℤ × ℤ)

-- Define the grid distance
def gridDistance (g : Grid) (p : ℤ × ℤ) : ℕ :=
  sorry

-- Define the marking function
def mark (n : ℕ) : ℚ :=
  1 / 2^n

-- Define the sum of markings for pieces
def pieceSum (g : Grid) (pieces : Set (ℤ × ℤ)) : ℚ :=
  sorry

-- Define the sum of markings for points with grid distance ≥ 7
def distantSum (g : Grid) : ℚ :=
  sorry

-- Main theorem
theorem always_close_piece (g : Grid) (pieces : Set (ℤ × ℤ)) :
  pieceSum g pieces > distantSum g :=
sorry

end NUMINAMATH_CALUDE_always_close_piece_l2246_224650


namespace NUMINAMATH_CALUDE_one_and_one_third_problem_l2246_224615

theorem one_and_one_third_problem : ∃ x : ℚ, (4/3) * x = 36 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_problem_l2246_224615


namespace NUMINAMATH_CALUDE_onion_harvest_scientific_notation_l2246_224623

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem onion_harvest_scientific_notation :
  toScientificNotation 325000000 = ScientificNotation.mk 3.25 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_onion_harvest_scientific_notation_l2246_224623


namespace NUMINAMATH_CALUDE_minkyung_height_l2246_224693

/-- Proves that Minkyung's height is 1.69 meters given the heights of Haeun and Nayeon relative to others -/
theorem minkyung_height :
  let haeun_height : ℝ := 1.56
  let nayeon_shorter_than_haeun : ℝ := 0.14
  let nayeon_shorter_than_minkyung : ℝ := 0.27
  let nayeon_height : ℝ := haeun_height - nayeon_shorter_than_haeun
  let minkyung_height : ℝ := nayeon_height + nayeon_shorter_than_minkyung
  minkyung_height = 1.69 := by sorry

end NUMINAMATH_CALUDE_minkyung_height_l2246_224693


namespace NUMINAMATH_CALUDE_sqrt_twenty_minus_sqrt_five_l2246_224630

theorem sqrt_twenty_minus_sqrt_five : Real.sqrt 20 - Real.sqrt 5 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twenty_minus_sqrt_five_l2246_224630


namespace NUMINAMATH_CALUDE_tile_size_calculation_l2246_224637

theorem tile_size_calculation (length width : ℝ) (num_tiles : ℕ) (h1 : length = 2) (h2 : width = 12) (h3 : num_tiles = 6) :
  (length * width) / num_tiles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tile_size_calculation_l2246_224637


namespace NUMINAMATH_CALUDE_bombardment_percentage_l2246_224658

/-- Proves that the percentage of people who died by bombardment is 10% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4500)
  (h2 : final_population = 3240)
  (h3 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population)))) :
  ∃ x : ℝ, x = 10 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population))) :=
by sorry

#check bombardment_percentage

end NUMINAMATH_CALUDE_bombardment_percentage_l2246_224658


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2246_224671

/-- Represents the quarterly scores of a team --/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Check if a sequence of four numbers is arithmetic --/
def isArithmetic (s : QuarterlyScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3

/-- Check if a sequence of four numbers is geometric --/
def isGeometric (s : QuarterlyScores) : Prop :=
  s.q1 > 0 ∧ s.q2 % s.q1 = 0 ∧ s.q3 % s.q2 = 0 ∧ s.q4 % s.q3 = 0 ∧
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3

/-- Sum of all quarterly scores --/
def totalScore (s : QuarterlyScores) : ℕ :=
  s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_scores :
  ∃ (raiders wildcats : QuarterlyScores),
    -- Tied at halftime
    raiders.q1 + raiders.q2 = wildcats.q1 + wildcats.q2 ∧
    -- Raiders' scores form an arithmetic sequence
    isArithmetic raiders ∧
    -- Wildcats' scores form a geometric sequence
    isGeometric wildcats ∧
    -- Fourth quarter combined score is half of total combined score
    raiders.q4 + wildcats.q4 = (totalScore raiders + totalScore wildcats) / 2 ∧
    -- Neither team scored more than 100 points
    totalScore raiders ≤ 100 ∧ totalScore wildcats ≤ 100 ∧
    -- First quarter total is one of the given options
    (raiders.q1 + wildcats.q1 = 10 ∨
     raiders.q1 + wildcats.q1 = 15 ∨
     raiders.q1 + wildcats.q1 = 20 ∨
     raiders.q1 + wildcats.q1 = 9 ∨
     raiders.q1 + wildcats.q1 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l2246_224671


namespace NUMINAMATH_CALUDE_b_85_mod_50_l2246_224657

/-- The sequence b_n is defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The 85th term of the sequence b_n is congruent to 36 modulo 50 -/
theorem b_85_mod_50 : b 85 ≡ 36 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_b_85_mod_50_l2246_224657


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2246_224690

/-- A geometric sequence with first term a and common ratio r -/
def GeometricSequence (a r : ℝ) : ℕ → ℝ :=
  fun n => a * r^(n - 1)

/-- An increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem geometric_sequence_increasing_condition (a r : ℝ) :
  (IsIncreasing (GeometricSequence a r) → 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5) ∧
  (∃ a r : ℝ, 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5 ∧
    ¬IsIncreasing (GeometricSequence a r)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l2246_224690


namespace NUMINAMATH_CALUDE_equation_solution_l2246_224606

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2246_224606


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2246_224666

/-- Given a right triangle, prove that if rotation about one leg produces a cone
    of volume 1620π cm³ and rotation about the other leg produces a cone
    of volume 3240π cm³, then the length of the hypotenuse is √507 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / 3 * π * a * b^2 = 1620 * π) →
  (1 / 3 * π * b * a^2 = 3240 * π) →
  Real.sqrt (a^2 + b^2) = Real.sqrt 507 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2246_224666


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2246_224673

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) -- Points in 2D plane
  (h_right : (R.1 - S.1) * (Q.1 - S.1) + (R.2 - S.2) * (Q.2 - S.2) = 0) -- Right angle at S
  (h_cos : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 3/5) -- cos R = 3/5
  (h_rs : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 10) -- RS = 10
  : Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 8 := by -- QS = 8
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2246_224673


namespace NUMINAMATH_CALUDE_four_heads_before_three_tails_l2246_224696

/-- The probability of encountering 4 consecutive heads before 3 consecutive tails in repeated fair coin flips -/
def q : ℚ := 16/23

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1/2)

theorem four_heads_before_three_tails (fair_coin : (ℚ → Prop) → Prop) : q = 16/23 := by
  sorry

end NUMINAMATH_CALUDE_four_heads_before_three_tails_l2246_224696


namespace NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l2246_224699

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (c = a ∧ b ≠ c)

-- Statement of the theorem
theorem quadratic_roots_and_isosceles_triangle :
  (∀ k : ℝ, discriminant k > 0) ∧
  (∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧
    is_isosceles x y 4) → (k = 3 ∨ k = 4)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l2246_224699


namespace NUMINAMATH_CALUDE_sum_of_digits_equation_l2246_224629

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_equation : 
  ∃ x : ℕ, x + S x = 2001 ∧ x = 1977 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equation_l2246_224629


namespace NUMINAMATH_CALUDE_sequence_property_l2246_224678

def sequence_condition (a : ℕ → ℝ) (m r : ℝ) : Prop :=
  a 1 = m ∧
  (∀ k : ℕ, a (2*k) = 2 * a (2*k - 1)) ∧
  (∀ k : ℕ, a (2*k + 1) = a (2*k) + r) ∧
  (∀ n : ℕ, n > 0 → a (n + 2) = a n)

theorem sequence_property (a : ℕ → ℝ) (m r : ℝ) 
  (h : sequence_condition a m r) : m + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2246_224678


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l2246_224695

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 125]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l2246_224695


namespace NUMINAMATH_CALUDE_yogurt_refund_calculation_l2246_224692

theorem yogurt_refund_calculation (total_packs : ℕ) (expired_percentage : ℚ) (cost_per_pack : ℕ) :
  total_packs = 80 →
  expired_percentage = 40/100 →
  cost_per_pack = 12 →
  (total_packs : ℚ) * expired_percentage * cost_per_pack = 384 :=
by sorry

end NUMINAMATH_CALUDE_yogurt_refund_calculation_l2246_224692


namespace NUMINAMATH_CALUDE_optimal_sampling_for_surveys_l2246_224617

-- Define the survey types
inductive SurveyType
  | Income
  | ArtStudents

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define the structure of the population for each survey
structure Population where
  survey_type : SurveyType
  total_size : Nat
  strata : Option (List Nat)

-- Define the sample size for each survey
structure SampleSize where
  survey_type : SurveyType
  size : Nat

-- Define the optimal sampling method function
def optimal_sampling_method (pop : Population) (sample : SampleSize) : SamplingMethod :=
  match pop.survey_type, pop.total_size, pop.strata, sample.size with
  | SurveyType.Income, _, some strata, 100 => SamplingMethod.Stratified
  | SurveyType.ArtStudents, 5, none, 3 => SamplingMethod.SimpleRandom
  | _, _, _, _ => SamplingMethod.SimpleRandom  -- Default case

-- Theorem statement
theorem optimal_sampling_for_surveys :
  let income_survey : Population := {
    survey_type := SurveyType.Income,
    total_size := 420,
    strata := some [125, 200, 95]
  }
  let art_survey : Population := {
    survey_type := SurveyType.ArtStudents,
    total_size := 5,
    strata := none
  }
  let income_sample : SampleSize := {
    survey_type := SurveyType.Income,
    size := 100
  }
  let art_sample : SampleSize := {
    survey_type := SurveyType.ArtStudents,
    size := 3
  }
  (optimal_sampling_method income_survey income_sample = SamplingMethod.Stratified) ∧
  (optimal_sampling_method art_survey art_sample = SamplingMethod.SimpleRandom) :=
by sorry

end NUMINAMATH_CALUDE_optimal_sampling_for_surveys_l2246_224617


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2246_224619

/-- Given a quadratic equation x^2 - 4x + m = 0 with one root x₁ = 1, prove that the other root x₂ = 3 -/
theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₂ : ℝ, ∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = 1 ∨ x = x₂)) → 
  (∃ x₂ : ℝ, x₂ = 3 ∧ ∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = 1 ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2246_224619


namespace NUMINAMATH_CALUDE_yvonne_success_probability_l2246_224616

theorem yvonne_success_probability 
  (p_xavier : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_yvonne_not_zelda : ℝ) 
  (h1 : p_xavier = 1/5)
  (h2 : p_zelda = 5/8)
  (h3 : p_xavier_yvonne_not_zelda = 0.0375) :
  ∃ p_yvonne : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_yvonne_not_zelda ∧ 
    p_yvonne = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_yvonne_success_probability_l2246_224616


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l2246_224669

-- Define a type for colors
inductive Color
| Black
| White

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  p1 : Point
  p2 : Point
  p3 : Point
  eq_sides : distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1

-- Theorem statement
theorem equilateral_triangle_exists :
  ∃ (t : EquilateralTriangle),
    (distance t.p1 t.p2 = 1 ∨ distance t.p1 t.p2 = Real.sqrt 3) ∧
    (coloring t.p1 = coloring t.p2 ∧ coloring t.p2 = coloring t.p3) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l2246_224669


namespace NUMINAMATH_CALUDE_trig_inequality_l2246_224641

theorem trig_inequality (θ : Real) (h : π < θ ∧ θ < 5 * π / 4) :
  Real.cos θ < Real.sin θ ∧ Real.sin θ < Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l2246_224641


namespace NUMINAMATH_CALUDE_quadratic_roots_l2246_224611

theorem quadratic_roots (a b c : ℝ) (h : (b^3)^2 - 4*(a^3)*(c^3) > 0) :
  (b^5)^2 - 4*(a^5)*(c^5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2246_224611


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2246_224653

/-- In a plane rectangular coordinate system, the coordinates of a point
    with respect to the origin are equal to its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let A : ℝ × ℝ := (x, y)
  A = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2246_224653


namespace NUMINAMATH_CALUDE_quadratic_properties_l2246_224683

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties (a b c m : ℝ) (h_a : a ≠ 0) :
  quadratic a b c (-2) = m ∧
  quadratic a b c (-1) = 1 ∧
  quadratic a b c 0 = -1 ∧
  quadratic a b c 1 = 1 ∧
  quadratic a b c 2 = 7 →
  (∀ x, quadratic a b c x = quadratic a b c (-x)) ∧  -- Symmetry axis at x = 0
  quadratic a b c 0 = -1 ∧                           -- Vertex at (0, -1)
  m = 7 ∧
  a > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2246_224683


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_h_eq_two_l2246_224651

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (c : ℝ), l1.direction = c • l2.direction

/-- The first line parameterized by s -/
def line1 (h : ℝ) : Line3D :=
  { point := (1, 0, 4),
    direction := (2, -1, h) }

/-- The second line parameterized by t -/
def line2 : Line3D :=
  { point := (0, 0, -6),
    direction := (3, 1, -2) }

/-- The main theorem stating the condition for coplanarity -/
theorem lines_coplanar_iff_h_eq_two :
  ∀ h : ℝ, are_coplanar (line1 h) line2 ↔ h = 2 := by
  sorry


end NUMINAMATH_CALUDE_lines_coplanar_iff_h_eq_two_l2246_224651


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2246_224655

theorem min_value_quadratic :
  ∀ x : ℝ, 3 * x^2 + 6 * x + 1487 ≥ 1484 ∧
  ∃ x : ℝ, 3 * x^2 + 6 * x + 1487 = 1484 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2246_224655


namespace NUMINAMATH_CALUDE_max_cfriendly_diff_l2246_224602

-- Define c-friendly function
def CFriendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  c > 1 ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → |f x - f y| ≤ c * |x - y|

-- State the theorem
theorem max_cfriendly_diff (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) :
  CFriendly c f →
  x ∈ Set.Icc 0 1 →
  y ∈ Set.Icc 0 1 →
  |f x - f y| ≤ (c + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_cfriendly_diff_l2246_224602


namespace NUMINAMATH_CALUDE_student_count_l2246_224638

/-- Represents the number of students in a class -/
def n : ℕ := sorry

/-- Average mark for Subject A before exclusion -/
def avg_a : ℚ := 80

/-- Average mark for Subject B before exclusion -/
def avg_b : ℚ := 85

/-- Average mark for Subject C before exclusion -/
def avg_c : ℚ := 75

/-- Number of excluded students -/
def excluded : ℕ := 5

/-- Average mark of excluded students for Subject A -/
def excluded_avg_a : ℚ := 20

/-- Average mark of excluded students for Subject B -/
def excluded_avg_b : ℚ := 25

/-- Average mark of excluded students for Subject C -/
def excluded_avg_c : ℚ := 15

/-- New average mark for Subject A after exclusion -/
def new_avg_a : ℚ := 90

/-- New average mark for Subject B after exclusion -/
def new_avg_b : ℚ := 95

/-- New average mark for Subject C after exclusion -/
def new_avg_c : ℚ := 85

theorem student_count : n = 35 := by sorry

end NUMINAMATH_CALUDE_student_count_l2246_224638


namespace NUMINAMATH_CALUDE_cat_whiskers_ratio_l2246_224659

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem stating the relationship between the cats' whiskers and their ratio -/
theorem cat_whiskers_ratio (c : CatWhiskers) : 
  c.juniper = 12 →
  c.buffy = 40 →
  c.puffy = 3 * c.juniper →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  (Ratio.mk c.puffy c.scruffy) = (Ratio.mk 1 2) := by
  sorry


end NUMINAMATH_CALUDE_cat_whiskers_ratio_l2246_224659


namespace NUMINAMATH_CALUDE_sum_of_quadratic_and_linear_l2246_224674

/-- Given a quadratic function q and a linear function p satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem sum_of_quadratic_and_linear 
  (q : ℝ → ℝ) 
  (p : ℝ → ℝ) 
  (hq_quad : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
  (hq_zeros : q 1 = 0 ∧ q 3 = 0)
  (hq_value : q 4 = 8)
  (hp_linear : ∃ m b : ℝ, ∀ x, p x = m * x + b)
  (hp_value : p 5 = 15) :
  ∀ x, p x + q x = (8/3) * x^2 - (29/3) * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_and_linear_l2246_224674


namespace NUMINAMATH_CALUDE_league_games_l2246_224636

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 → 
  total_games = 1900 → 
  total_games = (num_teams * (num_teams - 1) * games_per_matchup) / 2 → 
  games_per_matchup = 10 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l2246_224636


namespace NUMINAMATH_CALUDE_expression_evaluation_l2246_224634

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  2 * a^2 - 3 * b^2 + a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2246_224634


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_D_to_D_l2246_224649

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_is_8 : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_D_to_D_l2246_224649


namespace NUMINAMATH_CALUDE_overall_average_speed_l2246_224685

theorem overall_average_speed
  (car_time : Real) (car_speed : Real) (horse_time : Real) (horse_speed : Real)
  (h1 : car_time = 45 / 60)
  (h2 : car_speed = 20)
  (h3 : horse_time = 30 / 60)
  (h4 : horse_speed = 6)
  : (car_speed * car_time + horse_speed * horse_time) / (car_time + horse_time) = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_overall_average_speed_l2246_224685


namespace NUMINAMATH_CALUDE_units_digit_F_500_l2246_224665

-- Define the modified Fermat number function
def F (n : ℕ) : ℕ := 2^(2^(2*n)) + 1

-- Theorem statement
theorem units_digit_F_500 : F 500 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_F_500_l2246_224665


namespace NUMINAMATH_CALUDE_triangle_example_1_triangle_example_2_l2246_224642

-- Define the new operation ▲
def triangle (m n : ℤ) : ℤ := m - n + m * n

-- Theorem statements
theorem triangle_example_1 : triangle 3 (-4) = -5 := by sorry

theorem triangle_example_2 : triangle (-6) (triangle 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_example_1_triangle_example_2_l2246_224642


namespace NUMINAMATH_CALUDE_sequence_with_2018_distinct_elements_l2246_224652

theorem sequence_with_2018_distinct_elements :
  ∃ a : ℝ, ∃ (x : ℕ → ℝ), 
    (x 1 = a) ∧ 
    (∀ n : ℕ, x (n + 1) = (1 / 2) * (x n - 1 / x n)) ∧
    (∃ m : ℕ, m ≤ 2018 ∧ x m = 0) ∧
    (∀ i j : ℕ, i < j ∧ j ≤ 2018 → x i ≠ x j) ∧
    (∀ k : ℕ, k > 2018 → x k = 0) :=
by sorry

end NUMINAMATH_CALUDE_sequence_with_2018_distinct_elements_l2246_224652


namespace NUMINAMATH_CALUDE_heather_blocks_l2246_224648

/-- Given that Heather starts with 86 blocks and shares 41 blocks,
    prove that she ends up with 45 blocks. -/
theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (final_blocks : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_blocks = 41)
  (h3 : final_blocks = initial_blocks - shared_blocks) :
  final_blocks = 45 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_l2246_224648


namespace NUMINAMATH_CALUDE_sunflower_seed_distribution_l2246_224620

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) :
  total_seeds = 54 →
  num_cans = 9 →
  total_seeds = num_cans * seeds_per_can →
  seeds_per_can = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_distribution_l2246_224620


namespace NUMINAMATH_CALUDE_team_scoring_problem_l2246_224601

theorem team_scoring_problem (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) 
  (h1 : player1_score = 20)
  (h2 : player2_score = player1_score / 2)
  (h3 : ∃ X : ℕ, player3_score = X * player2_score)
  (h4 : (player1_score + player2_score + player3_score) / 3 = 30) :
  ∃ X : ℕ, player3_score = X * player2_score ∧ X = 6 := by
  sorry

end NUMINAMATH_CALUDE_team_scoring_problem_l2246_224601


namespace NUMINAMATH_CALUDE_clock_coincidences_l2246_224697

/-- Represents a clock with minute and hour hands -/
structure Clock :=
  (minuteRotations : ℕ) -- Number of full rotations of minute hand in 12 hours
  (hourRotations : ℕ)   -- Number of full rotations of hour hand in 12 hours

/-- The standard 12-hour clock -/
def standardClock : Clock :=
  { minuteRotations := 12,
    hourRotations := 1 }

/-- Number of coincidences between minute and hour hands in 12 hours -/
def coincidences (c : Clock) : ℕ :=
  c.minuteRotations - c.hourRotations

/-- Interval between coincidences in minutes -/
def coincidenceInterval (c : Clock) : ℚ :=
  (12 * 60) / (coincidences c)

theorem clock_coincidences (c : Clock) :
  c = standardClock →
  coincidences c = 11 ∧
  coincidenceInterval c = 65 + 5/11 :=
sorry

end NUMINAMATH_CALUDE_clock_coincidences_l2246_224697


namespace NUMINAMATH_CALUDE_square_puzzle_l2246_224667

/-- Given a square with side length n satisfying the equation n^2 + 20 = (n + 1)^2 - 9,
    prove that the total number of small squares is 216. -/
theorem square_puzzle (n : ℕ) (h : n^2 + 20 = (n + 1)^2 - 9) : n^2 + 20 = 216 := by
  sorry

end NUMINAMATH_CALUDE_square_puzzle_l2246_224667


namespace NUMINAMATH_CALUDE_conference_beverages_l2246_224624

theorem conference_beverages (total participants : ℕ) (coffee_drinkers : ℕ) (tea_drinkers : ℕ) (both_drinkers : ℕ) :
  total = 30 →
  coffee_drinkers = 15 →
  tea_drinkers = 18 →
  both_drinkers = 8 →
  total - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
sorry

end NUMINAMATH_CALUDE_conference_beverages_l2246_224624


namespace NUMINAMATH_CALUDE_abs_neg_one_eq_one_l2246_224689

theorem abs_neg_one_eq_one : |(-1 : ℚ)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_eq_one_l2246_224689


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l2246_224691

/-- The volume of a cube with side area 64 cm² is 512 cm³ -/
theorem cube_volume_from_side_area :
  ∀ (side_length : ℝ),
  side_length ^ 2 = 64 →
  side_length ^ 3 = 512 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l2246_224691


namespace NUMINAMATH_CALUDE_dianes_gambling_problem_l2246_224633

/-- Diane's gambling problem -/
theorem dianes_gambling_problem 
  (x y a b : ℝ) 
  (h1 : x * a = 65)
  (h2 : y * b = 150)
  (h3 : x * a - y * b = -50) :
  y * b - x * a = 50 := by
  sorry

end NUMINAMATH_CALUDE_dianes_gambling_problem_l2246_224633


namespace NUMINAMATH_CALUDE_triangle_area_l2246_224646

/-- Given a triangle ABC where sin A = 3/5 and the dot product of vectors AB and AC is 8,
    prove that the area of the triangle is 3. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let sinA : ℝ := 3/5
  let dotProduct : ℝ := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dotProduct = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
         Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * sinA = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2246_224646


namespace NUMINAMATH_CALUDE_score_difference_l2246_224654

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℝ
  physics : ℝ
  chemistry : ℝ

/-- The problem statement -/
theorem score_difference (s : Scores) 
  (h1 : s.math + s.physics = 20)
  (h2 : (s.math + s.chemistry) / 2 = 20)
  (h3 : s.chemistry > s.physics) :
  s.chemistry - s.physics = 20 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l2246_224654


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l2246_224622

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under given conditions. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sink_depth : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sink_depth water_density = 60 := by
sorry


end NUMINAMATH_CALUDE_mass_of_man_on_boat_l2246_224622


namespace NUMINAMATH_CALUDE_original_number_proof_l2246_224684

theorem original_number_proof (x : ℝ) 
  (h1 : x * 74 = 19832) 
  (h2 : x / 100 * 0.74 = 1.9832) : 
  x = 268 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2246_224684


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2246_224672

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -1 ∧ b₀ = -1 ∧ c₀ = -1 ∧ d₀ = -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2246_224672


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2246_224680

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  2 * X^5 + 11 * X^4 - 48 * X^3 - 60 * X^2 + 20 * X + 50 =
  (X^3 + 7 * X^2 + 4) * q + (-27 * X^3 - 68 * X^2 + 32 * X + 50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2246_224680


namespace NUMINAMATH_CALUDE_fruit_drink_composition_l2246_224613

-- Define the composition of the fruit drink
def orange_percent : ℝ := 25
def watermelon_percent : ℝ := 40
def grape_ounces : ℝ := 70

-- Define the total volume of the drink
def total_volume : ℝ := 200

-- Theorem statement
theorem fruit_drink_composition :
  orange_percent + watermelon_percent + (grape_ounces / total_volume * 100) = 100 ∧
  grape_ounces / (grape_ounces / total_volume * 100) * 100 = total_volume :=
by sorry

end NUMINAMATH_CALUDE_fruit_drink_composition_l2246_224613


namespace NUMINAMATH_CALUDE_biology_class_size_l2246_224687

theorem biology_class_size :
  ∀ (S : ℕ), 
    (S : ℝ) * 0.8 * 0.25 = 8 →
    S = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_biology_class_size_l2246_224687


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2246_224603

theorem cubic_expression_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2246_224603


namespace NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_B_complement_union_A_B_l2246_224618

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < -2 ∨ x > 5}
def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

-- Theorem statements
theorem complement_A : Aᶜ = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := by sorry

theorem complement_B : Bᶜ = {x : ℝ | x < 4 ∨ x > 6} := by sorry

theorem intersection_A_B : A ∩ B = {x : ℝ | 5 < x ∧ x ≤ 6} := by sorry

theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_B_intersection_A_B_complement_union_A_B_l2246_224618


namespace NUMINAMATH_CALUDE_combined_age_is_28_l2246_224631

/-- Represents the ages of Michael and his brothers -/
structure FamilyAges where
  michael : ℕ
  younger_brother : ℕ
  older_brother : ℕ

/-- Defines the conditions for the ages of Michael and his brothers -/
def valid_ages (ages : FamilyAges) : Prop :=
  ages.younger_brother = 5 ∧
  ages.older_brother = 3 * ages.younger_brother ∧
  ages.older_brother = 2 * (ages.michael - 1) + 1

/-- Theorem stating that the combined age of Michael and his brothers is 28 years -/
theorem combined_age_is_28 (ages : FamilyAges) (h : valid_ages ages) :
  ages.michael + ages.younger_brother + ages.older_brother = 28 := by
  sorry


end NUMINAMATH_CALUDE_combined_age_is_28_l2246_224631


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2246_224681

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2246_224681


namespace NUMINAMATH_CALUDE_fraction_problem_l2246_224661

theorem fraction_problem : ∃ x : ℚ, 
  x * (3/4 : ℚ) = (1/6 : ℚ) ∧ x - (1/12 : ℚ) = (5/36 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2246_224661


namespace NUMINAMATH_CALUDE_hours_worked_on_second_job_l2246_224682

/-- Calculates the number of hours worked on the second job given the total earnings and other job details -/
theorem hours_worked_on_second_job
  (hourly_rate_1 hourly_rate_2 hourly_rate_3 : ℚ)
  (hours_1 hours_3 : ℚ)
  (days : ℚ)
  (total_earnings : ℚ)
  (h1 : hourly_rate_1 = 7)
  (h2 : hourly_rate_2 = 10)
  (h3 : hourly_rate_3 = 12)
  (h4 : hours_1 = 3)
  (h5 : hours_3 = 4)
  (h6 : days = 5)
  (h7 : total_earnings = 445)
  : ∃ hours_2 : ℚ, hours_2 = 2 ∧ 
    days * (hourly_rate_1 * hours_1 + hourly_rate_2 * hours_2 + hourly_rate_3 * hours_3) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hours_worked_on_second_job_l2246_224682


namespace NUMINAMATH_CALUDE_joans_kittens_l2246_224632

/-- Represents the number of kittens Joan gave to her friends -/
def kittens_given_away (initial_kittens current_kittens : ℕ) : ℕ :=
  initial_kittens - current_kittens

/-- Proves that Joan gave away 2 kittens -/
theorem joans_kittens : kittens_given_away 8 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l2246_224632


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2246_224668

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) :
  (60 / 360 * (2 * Real.pi * C) = 40 / 360 * (2 * Real.pi * D)) →
  (Real.pi * C^2) / (Real.pi * D^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2246_224668


namespace NUMINAMATH_CALUDE_exponent_rule_l2246_224612

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l2246_224612


namespace NUMINAMATH_CALUDE_epipen_insurance_coverage_l2246_224664

/-- Calculates the insurance coverage percentage for EpiPens -/
theorem epipen_insurance_coverage 
  (frequency : ℕ) -- Number of EpiPens per year
  (cost : ℝ) -- Cost of each EpiPen in dollars
  (annual_payment : ℝ) -- John's annual payment in dollars
  (h1 : frequency = 2) -- John gets 2 EpiPens per year
  (h2 : cost = 500) -- Each EpiPen costs $500
  (h3 : annual_payment = 250) -- John pays $250 per year
  : (1 - annual_payment / (frequency * cost)) * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_epipen_insurance_coverage_l2246_224664


namespace NUMINAMATH_CALUDE_min_squares_covering_sqrt63_l2246_224677

theorem min_squares_covering_sqrt63 :
  ∀ n : ℕ, n ≥ 2 → (4 * n - 4 ≥ Real.sqrt 63 ↔ n ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_covering_sqrt63_l2246_224677


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l2246_224625

/-- Represents a point on the perimeter of the block area -/
structure Point where
  distance : ℝ  -- Distance from the starting point A
  mk_point_valid : 0 ≤ distance ∧ distance < 24

/-- Represents a walker -/
structure Walker where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The scenario of the two walkers -/
def scenario : Prop :=
  ∃ (jane hector : Walker) (meeting_point : Point),
    jane.speed = 2 * hector.speed ∧
    jane.direction ≠ hector.direction ∧
    (jane.direction = true → meeting_point.distance = 16) ∧
    (jane.direction = false → meeting_point.distance = 8)

/-- The theorem to be proved -/
theorem meeting_point_theorem :
  scenario → 
  ∃ (meeting_point : Point), 
    (meeting_point.distance = 8 ∨ meeting_point.distance = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l2246_224625


namespace NUMINAMATH_CALUDE_kitchen_guest_bath_living_area_l2246_224679

/-- Calculates the area of the kitchen, guest bath, and living area given the areas of other rooms and rent information -/
theorem kitchen_guest_bath_living_area 
  (master_bath_area : ℝ) 
  (guest_bedroom_area : ℝ) 
  (num_guest_bedrooms : ℕ) 
  (total_rent : ℝ) 
  (cost_per_sqft : ℝ) 
  (h1 : master_bath_area = 500) 
  (h2 : guest_bedroom_area = 200) 
  (h3 : num_guest_bedrooms = 2) 
  (h4 : total_rent = 3000) 
  (h5 : cost_per_sqft = 2) : 
  ℝ := by
  sorry

#check kitchen_guest_bath_living_area

end NUMINAMATH_CALUDE_kitchen_guest_bath_living_area_l2246_224679


namespace NUMINAMATH_CALUDE_det_related_matrix_l2246_224635

/-- Given a 2x2 matrix with determinant 4, prove that the determinant of a related matrix is 12 -/
theorem det_related_matrix (a b c d : ℝ) (h : a * d - b * c = 4) :
  a * (7 * c + 3 * d) - c * (7 * a + 3 * b) = 12 := by
  sorry


end NUMINAMATH_CALUDE_det_related_matrix_l2246_224635


namespace NUMINAMATH_CALUDE_patio_surrounded_by_bushes_l2246_224660

/-- The side length of the square patio in feet -/
def patio_side_length : ℝ := 20

/-- The spacing between rose bushes in feet -/
def bush_spacing : ℝ := 2

/-- The number of rose bushes needed to surround the patio -/
def num_bushes : ℕ := 40

/-- Theorem stating that the number of rose bushes needed to surround the square patio is 40 -/
theorem patio_surrounded_by_bushes :
  (4 * patio_side_length) / bush_spacing = num_bushes := by sorry

end NUMINAMATH_CALUDE_patio_surrounded_by_bushes_l2246_224660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2246_224688

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2246_224688


namespace NUMINAMATH_CALUDE_sculpture_cost_in_yen_l2246_224626

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Japanese yen -/
def usd_to_jpy : ℚ := 110

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 136

/-- Theorem stating the cost of the sculpture in Japanese yen -/
theorem sculpture_cost_in_yen : 
  (sculpture_cost_nad / usd_to_nad) * usd_to_jpy = 1870 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_yen_l2246_224626


namespace NUMINAMATH_CALUDE_function_periodicity_l2246_224639

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define the existence of c
def existsC (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ f (c / 2) = 0

-- Define periodicity
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f x

-- Theorem statement
theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : functionalEquation f) 
  (h2 : existsC f) :
  ∃ T : ℝ, T > 0 ∧ isPeriodic f T :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l2246_224639


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2246_224675

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 12 * Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2246_224675


namespace NUMINAMATH_CALUDE_sum_terms_increase_l2246_224694

def sum_terms (k : ℕ) : ℕ := 2^(k-1) + 1

theorem sum_terms_increase (k : ℕ) (h : k ≥ 2) : 
  sum_terms (k+1) - sum_terms k = 2^(k-1) := by
  sorry

end NUMINAMATH_CALUDE_sum_terms_increase_l2246_224694


namespace NUMINAMATH_CALUDE_xy_sum_of_squares_l2246_224645

theorem xy_sum_of_squares (x y : ℝ) : 
  x * y = 12 → 
  x^2 * y + x * y^2 + x + y = 120 → 
  x^2 + y^2 = 10344 / 169 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_of_squares_l2246_224645


namespace NUMINAMATH_CALUDE_inequality_reversal_l2246_224605

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(1 - x > 1 - y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l2246_224605


namespace NUMINAMATH_CALUDE_intersection_points_of_cubic_l2246_224621

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_of_cubic (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ + c = 0 ∧ f x₂ + c = 0 ∧
    ∀ x, f x + c = 0 → x = x₁ ∨ x = x₂) ↔ c = -2 ∨ c = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_of_cubic_l2246_224621


namespace NUMINAMATH_CALUDE_ball_path_on_5x2_table_l2246_224600

/-- A rectangular table with integer dimensions -/
structure RectTable where
  length : ℕ
  width : ℕ

/-- The path of a ball on a rectangular table -/
def BallPath (table : RectTable) :=
  { bounces : ℕ // bounces ≤ table.length + table.width }

/-- Theorem: A ball on a 5x2 table reaches the opposite corner in 5 bounces -/
theorem ball_path_on_5x2_table :
  ∀ (table : RectTable),
    table.length = 5 →
    table.width = 2 →
    ∃ (path : BallPath table),
      path.val = 5 ∧
      (∀ (other_path : BallPath table), other_path.val ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_ball_path_on_5x2_table_l2246_224600


namespace NUMINAMATH_CALUDE_pills_in_week_l2246_224643

/-- Calculates the number of pills taken in a week given the interval between pills in hours -/
def pills_per_week (hours_between_pills : ℕ) : ℕ :=
  let hours_per_day : ℕ := 24
  let days_per_week : ℕ := 7
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: A person who takes a pill every 6 hours will take 28 pills in a week -/
theorem pills_in_week : pills_per_week 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_pills_in_week_l2246_224643
