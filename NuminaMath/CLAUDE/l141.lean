import Mathlib

namespace NUMINAMATH_CALUDE_jake_bitcoin_factor_l141_14147

theorem jake_bitcoin_factor (initial_fortune : ℕ) (first_donation : ℕ) (second_donation : ℕ) (final_amount : ℕ) :
  initial_fortune = 80 ∧
  first_donation = 20 ∧
  second_donation = 10 ∧
  final_amount = 80 →
  ∃ (factor : ℚ), 
    factor = 3 ∧
    final_amount = (((initial_fortune - first_donation) / 2) * factor).floor - second_donation :=
by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_factor_l141_14147


namespace NUMINAMATH_CALUDE_even_weeks_count_l141_14172

/-- Represents a day in a month --/
structure Day where
  number : ℕ
  month : ℕ
  deriving Repr

/-- Represents a week in a calendar --/
structure Week where
  days : List Day
  deriving Repr

/-- Determines if a week is even based on the sum of its day numbers --/
def isEvenWeek (w : Week) : Bool :=
  (w.days.map (λ d => d.number)).sum % 2 == 0

/-- Generates the 52 weeks starting from the first Monday of January --/
def generateWeeks : List Week :=
  sorry

/-- Counts the number of even weeks in a list of weeks --/
def countEvenWeeks (weeks : List Week) : ℕ :=
  (weeks.filter isEvenWeek).length

/-- Theorem stating that the number of even weeks in the 52-week period is 30 --/
theorem even_weeks_count :
  countEvenWeeks generateWeeks = 30 := by
  sorry

end NUMINAMATH_CALUDE_even_weeks_count_l141_14172


namespace NUMINAMATH_CALUDE_f_inequality_and_range_l141_14184

noncomputable def f (x : ℝ) := 1 - Real.exp (-x)

theorem f_inequality_and_range :
  (∀ x > -1, f x ≥ x / (x + 1)) ∧
  (Set.Icc (0 : ℝ) (1/2) = {a | ∀ x ≥ 0, f x ≤ x / (a * x + 1)}) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_and_range_l141_14184


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l141_14106

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l141_14106


namespace NUMINAMATH_CALUDE_special_sets_exist_l141_14175

/-- Two infinite sets of non-negative integers with special properties -/
structure SpecialSets where
  A : Set ℕ
  B : Set ℕ
  infinite_A : Set.Infinite A
  infinite_B : Set.Infinite B
  unique_rep : ∀ n : ℕ, ∃! (a : ℕ) (b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b
  multiples : ∃ k : ℕ, k > 1 ∧ (∀ x ∈ A, ∃ m : ℕ, x = m * k) ∨ (∀ x ∈ B, ∃ m : ℕ, x = m * k)

/-- The existence of special sets with the required properties -/
theorem special_sets_exist : ∃ S : SpecialSets, True := by
  sorry

end NUMINAMATH_CALUDE_special_sets_exist_l141_14175


namespace NUMINAMATH_CALUDE_a_3_equals_zero_l141_14181

theorem a_3_equals_zero (a : ℕ → ℝ) (h : ∀ n, a n = Real.sin (n * π / 3)) : a 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_zero_l141_14181


namespace NUMINAMATH_CALUDE_major_preference_stronger_than_gender_l141_14164

/-- Represents the observed K^2 value for gender preference --/
def k1 : ℝ := 1.010

/-- Represents the observed K^2 value for major preference --/
def k2 : ℝ := 9.090

/-- Theorem stating that the observed K^2 value for major preference is greater than the observed K^2 value for gender preference --/
theorem major_preference_stronger_than_gender : k2 > k1 := by sorry

end NUMINAMATH_CALUDE_major_preference_stronger_than_gender_l141_14164


namespace NUMINAMATH_CALUDE_doll_collection_increase_l141_14100

theorem doll_collection_increase (initial_count : ℕ) : 
  (initial_count : ℚ) * (1 + 1/4) = initial_count + 2 → 
  initial_count + 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l141_14100


namespace NUMINAMATH_CALUDE_octal_to_decimal_23456_l141_14161

/-- Converts a base-8 digit to its base-10 equivalent --/
def octal_to_decimal (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- The base-10 equivalent of 23456 in base-8 --/
def base_10_equivalent : ℕ :=
  octal_to_decimal 6 0 +
  octal_to_decimal 5 1 +
  octal_to_decimal 4 2 +
  octal_to_decimal 3 3 +
  octal_to_decimal 2 4

/-- Theorem: The base-10 equivalent of 23456 in base-8 is 5934 --/
theorem octal_to_decimal_23456 : base_10_equivalent = 5934 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_23456_l141_14161


namespace NUMINAMATH_CALUDE_total_coins_l141_14182

theorem total_coins (quarters_piles dimes_piles nickels_piles pennies_piles : ℕ)
  (quarters_per_pile dimes_per_pile nickels_per_pile pennies_per_pile : ℕ)
  (h1 : quarters_piles = 5)
  (h2 : dimes_piles = 5)
  (h3 : nickels_piles = 3)
  (h4 : pennies_piles = 4)
  (h5 : quarters_per_pile = 3)
  (h6 : dimes_per_pile = 3)
  (h7 : nickels_per_pile = 4)
  (h8 : pennies_per_pile = 5) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 62 :=
by sorry

end NUMINAMATH_CALUDE_total_coins_l141_14182


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l141_14183

/-- Given a quadratic equation mx^2 + nx - (m+n) = 0, prove that:
    1. The equation has two real roots.
    2. If n = 1 and the product of the roots is greater than 1, then -1/2 < m < 0. -/
theorem quadratic_equation_properties (m n : ℝ) :
  let f : ℝ → ℝ := λ x => m * x^2 + n * x - (m + n)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (n = 1 → (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ * x₂ > 1 → -1/2 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l141_14183


namespace NUMINAMATH_CALUDE_unique_products_count_l141_14152

def bag_A : Finset ℕ := {1, 3, 5, 7}
def bag_B : Finset ℕ := {2, 4, 6, 8}

theorem unique_products_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 * p.2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_products_count_l141_14152


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l141_14132

/-- Given seven points on a circle -/
def num_points : ℕ := 7

/-- Number of chords that can be formed from seven points -/
def total_chords : ℕ := num_points.choose 2

/-- Number of chords selected -/
def selected_chords : ℕ := 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (num_points.choose selected_chords) / (total_chords.choose selected_chords)

/-- Theorem: The probability of forming a convex pentagon by randomly selecting
    five chords from seven points on a circle is 1/969 -/
theorem convex_pentagon_probability : probability = 1 / 969 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l141_14132


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l141_14162

theorem polynomial_root_sum (b c d e : ℝ) : 
  (∀ x : ℝ, 2*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 4 ∨ x = -3 ∨ x = 5 ∨ x = ((-b-c-d)/2)) →
  (b + c + d) / 2 = 151 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l141_14162


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_minimum_l141_14167

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / (2 * x) + (3 / 2) * x + 1

theorem tangent_perpendicular_and_minimum (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((a / x) - 1 / (2 * x^2) + 3 / 2) x) →
  (HasDerivAt (f a) 0 1) →
  a = -1 ∧
  ∀ x > 0, f (-1) x ≥ 3 ∧ f (-1) 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_minimum_l141_14167


namespace NUMINAMATH_CALUDE_hexagon_area_in_triangle_l141_14102

/-- The area of a regular hexagon inscribed in a square, which is inscribed in a circle, 
    which is in turn inscribed in a triangle with side length 6 cm, is 27√3 cm². -/
theorem hexagon_area_in_triangle (s : ℝ) (h : s = 6) : 
  let r := s / 2 * Real.sqrt 3 / 3
  let square_side := 2 * r
  let hexagon_side := r
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side ^ 2
  hexagon_area = 27 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_in_triangle_l141_14102


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l141_14149

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n ↦ (a n)^2)) ∧
  (∀ k : ℝ, k ≠ 0 → is_geometric_sequence (fun n ↦ k * a n)) ∧
  (is_geometric_sequence (fun n ↦ 1 / (a n))) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l141_14149


namespace NUMINAMATH_CALUDE_remaining_average_l141_14143

theorem remaining_average (total : ℝ) (avg1 avg2 : ℝ) :
  total = 6 * 5.40 ∧
  avg1 = 5.2 ∧
  avg2 = 5.80 →
  (total - 2 * avg1 - 2 * avg2) / 2 = 5.20 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l141_14143


namespace NUMINAMATH_CALUDE_calculation_correction_l141_14124

theorem calculation_correction (x : ℝ) (h : 63 / x = 9) : 36 - x = 29 := by
  sorry

end NUMINAMATH_CALUDE_calculation_correction_l141_14124


namespace NUMINAMATH_CALUDE_bug_visits_29_tiles_l141_14151

/-- Represents a rectangular floor --/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor --/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

/-- The specific floor in the problem --/
def problemFloor : RectangularFloor :=
  { width := 11, length := 19 }

/-- Theorem stating that a bug walking diagonally across the problem floor visits 29 tiles --/
theorem bug_visits_29_tiles : tilesVisited problemFloor = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_visits_29_tiles_l141_14151


namespace NUMINAMATH_CALUDE_same_height_antonio_maria_l141_14135

-- Define the type for height comparisons
inductive HeightComparison
  | Taller : HeightComparison
  | Shorter : HeightComparison
  | Same : HeightComparison

-- Define the siblings
inductive Sibling
  | Luiza : Sibling
  | Maria : Sibling
  | Antonio : Sibling
  | Julio : Sibling

-- Define the height comparison function
def compareHeight : Sibling → Sibling → HeightComparison := sorry

-- State the theorem
theorem same_height_antonio_maria :
  (compareHeight Sibling.Luiza Sibling.Antonio = HeightComparison.Taller) →
  (compareHeight Sibling.Antonio Sibling.Julio = HeightComparison.Taller) →
  (compareHeight Sibling.Maria Sibling.Luiza = HeightComparison.Shorter) →
  (compareHeight Sibling.Julio Sibling.Maria = HeightComparison.Shorter) →
  (compareHeight Sibling.Antonio Sibling.Maria = HeightComparison.Same) :=
by
  sorry

end NUMINAMATH_CALUDE_same_height_antonio_maria_l141_14135


namespace NUMINAMATH_CALUDE_acid_concentration_theorem_l141_14129

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) : Prop :=
  let water1 := (acid1 / 0.05) - acid1
  let water2 := water - water1
  let conc2 := acid2 / (acid2 + water2)
  conc2 = 70 / 300 →
  let total_water := water1 + water2
  (acid3 / (acid3 + total_water)) * 100 = 10.5

theorem acid_concentration_theorem :
  acid_concentration_problem 10 20 30 255.714 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_theorem_l141_14129


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l141_14150

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence satisfying the given conditions -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l141_14150


namespace NUMINAMATH_CALUDE_erasers_in_box_l141_14169

/-- The number of erasers left in the box after a series of operations -/
def erasers_left (initial : ℕ) (taken : ℕ) (added : ℕ) : ℕ :=
  let remaining := initial - taken
  let half_taken := remaining / 2
  remaining - half_taken + added

/-- Theorem stating the number of erasers left in the box -/
theorem erasers_in_box : erasers_left 320 67 30 = 157 := by
  sorry

end NUMINAMATH_CALUDE_erasers_in_box_l141_14169


namespace NUMINAMATH_CALUDE_two_A_minus_four_B_y_value_when_independent_of_x_l141_14178

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y - 2 * x
def B (x y : ℝ) : ℝ := x^2 - x * y + 1

-- Theorem 1: 2A - 4B = 10xy - 4x - 4
theorem two_A_minus_four_B (x y : ℝ) :
  2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4 := by sorry

-- Theorem 2: When 2A - 4B is independent of x, y = 2/5
theorem y_value_when_independent_of_x (y : ℝ) :
  (∀ x : ℝ, 2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4) →
  y = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_two_A_minus_four_B_y_value_when_independent_of_x_l141_14178


namespace NUMINAMATH_CALUDE_fifth_match_goals_is_five_l141_14188

/-- The number of goals scored in the fifth match -/
def fifth_match_goals : ℕ := 5

/-- The total number of matches played -/
def total_matches : ℕ := 5

/-- The increase in average goals after the fifth match -/
def average_increase : ℚ := 1/5

/-- The total number of goals in all matches -/
def total_goals : ℕ := 21

/-- Theorem stating that the number of goals scored in the fifth match is 5 -/
theorem fifth_match_goals_is_five :
  fifth_match_goals = 5 ∧
  (fifth_match_goals : ℚ) + (total_goals - fifth_match_goals) = total_goals ∧
  (total_goals : ℚ) / total_matches = 
    ((total_goals - fifth_match_goals) : ℚ) / (total_matches - 1) + average_increase :=
by sorry

end NUMINAMATH_CALUDE_fifth_match_goals_is_five_l141_14188


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l141_14138

theorem fraction_sum_simplification :
  8 / 24 - 5 / 72 + 3 / 8 = 23 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l141_14138


namespace NUMINAMATH_CALUDE_triangle_segment_inequality_l141_14134

/-- Represents a configuration of points in space -/
structure PointConfiguration where
  n : ℕ
  K : ℕ
  T : ℕ
  h_n_ge_2 : n ≥ 2
  h_K_gt_1 : K > 1
  h_no_four_coplanar : True  -- This is a placeholder for the condition

/-- The main theorem -/
theorem triangle_segment_inequality (config : PointConfiguration) :
  9 * (config.T ^ 2) < 2 * (config.K ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_inequality_l141_14134


namespace NUMINAMATH_CALUDE_garden_fence_length_l141_14130

/-- The total length of a fence surrounding a sector-shaped garden -/
def fence_length (radius : ℝ) (central_angle : ℝ) : ℝ :=
  radius * central_angle + 2 * radius

/-- Proof that the fence length for a garden with radius 30m and central angle 120° is 20π + 60m -/
theorem garden_fence_length :
  fence_length 30 (2 * Real.pi / 3) = 20 * Real.pi + 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l141_14130


namespace NUMINAMATH_CALUDE_sleep_variance_proof_l141_14185

def sleep_data : List ℝ := [6, 6, 7, 6, 7, 8, 9]

theorem sleep_variance_proof :
  let n : ℕ := sleep_data.length
  let mean : ℝ := (sleep_data.sum) / n
  let variance : ℝ := (sleep_data.map (λ x => (x - mean)^2)).sum / n
  mean = 7 → variance = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_sleep_variance_proof_l141_14185


namespace NUMINAMATH_CALUDE_bd_value_l141_14153

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.C = dist t.B t.C

def onLine (A B D : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), D.1 = A.1 + k * (B.1 - A.1) ∧ D.2 = A.2 + k * (B.2 - A.2)

def between (A B D : ℝ × ℝ) : Prop :=
  onLine A B D ∧ dist A B + dist B D = dist A D

-- State the theorem
theorem bd_value (t : Triangle) :
  isIsosceles t →
  dist t.A t.C = 10 →
  dist t.B t.C = 10 →
  dist t.A t.B = 4 →
  onLine t.A t.B t.D →
  between t.A t.B t.D →
  dist t.C t.D = 12 →
  dist t.B t.D = 4 * Real.sqrt 3 - 2 := by
  sorry


end NUMINAMATH_CALUDE_bd_value_l141_14153


namespace NUMINAMATH_CALUDE_high_school_total_students_l141_14155

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The high school satisfies the given conditions. -/
def satisfies_conditions (hs : HighSchool) : Prop :=
  hs.senior_students = 600 ∧
  hs.sample_size = 90 ∧
  hs.freshman_sample = 27 ∧
  hs.sophomore_sample = 33

/-- Theorem stating that a high school satisfying the given conditions has 1800 total students. -/
theorem high_school_total_students (hs : HighSchool) 
  (h : satisfies_conditions hs) : hs.total_students = 1800 := by
  sorry

end NUMINAMATH_CALUDE_high_school_total_students_l141_14155


namespace NUMINAMATH_CALUDE_exist_three_naturals_with_prime_sum_and_product_l141_14189

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Theorem statement
theorem exist_three_naturals_with_prime_sum_and_product :
  ∃ a b c : ℕ, isPrime (a + b + c) ∧ isPrime (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_exist_three_naturals_with_prime_sum_and_product_l141_14189


namespace NUMINAMATH_CALUDE_triangle_properties_l141_14176

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (8, 2)

-- Define the median from A to BC
def median_A_BC (x y : ℝ) : Prop :=
  y = 4

-- Define the perpendicular bisector of AC
def perp_bisector_AC (x y : ℝ) : Prop :=
  y = 4 * x - 13

-- Theorem statement
theorem triangle_properties :
  (∀ x y, median_A_BC x y ↔ y = 4) ∧
  (∀ x y, perp_bisector_AC x y ↔ y = 4 * x - 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l141_14176


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l141_14103

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l141_14103


namespace NUMINAMATH_CALUDE_pencil_distribution_l141_14170

/-- Given:
  - total_pencils: The total number of pencils
  - original_classes: The original number of classes
  - remaining_pencils: The number of pencils remaining after distribution
  - pencil_difference: The difference in pencils per class compared to the original plan
  This theorem proves that the actual number of classes is 11
-/
theorem pencil_distribution
  (total_pencils : ℕ)
  (original_classes : ℕ)
  (remaining_pencils : ℕ)
  (pencil_difference : ℕ)
  (h1 : total_pencils = 172)
  (h2 : original_classes = 4)
  (h3 : remaining_pencils = 7)
  (h4 : pencil_difference = 28)
  : ∃ (actual_classes : ℕ),
    actual_classes > original_classes ∧
    (total_pencils - remaining_pencils) / actual_classes + pencil_difference = total_pencils / original_classes ∧
    actual_classes = 11 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l141_14170


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l141_14117

/-- For a geometric sequence with positive terms and common ratio q where q^2 = 4,
    the ratio (a_3 + a_4) / (a_4 + a_5) equals 1/2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- common ratio is q
  q^2 = 4 →
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l141_14117


namespace NUMINAMATH_CALUDE_quadratic_sum_l141_14179

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (-6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 261) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l141_14179


namespace NUMINAMATH_CALUDE_tyler_erasers_count_l141_14187

def tyler_problem (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_price : ℕ) (remaining_money : ℕ) : ℕ := 
  let money_after_scissors := initial_money - scissors_count * scissors_price
  let money_spent_on_erasers := money_after_scissors - remaining_money
  money_spent_on_erasers / eraser_price

theorem tyler_erasers_count : 
  tyler_problem 100 8 5 4 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tyler_erasers_count_l141_14187


namespace NUMINAMATH_CALUDE_integer_part_inequality_l141_14119

theorem integer_part_inequality (m n : ℕ+) : 
  (∀ (α β : ℝ), ⌊(m+n)*α⌋ + ⌊(m+n)*β⌋ ≥ ⌊m*α⌋ + ⌊m*β⌋ + ⌊n*(α+β)⌋) ↔ m = n :=
sorry

end NUMINAMATH_CALUDE_integer_part_inequality_l141_14119


namespace NUMINAMATH_CALUDE_maria_fair_money_l141_14154

/-- The amount of money Maria had when she left the fair -/
def money_left : ℕ := 16

/-- The difference between the money Maria had when she got to the fair and when she left -/
def money_difference : ℕ := 71

/-- The amount of money Maria had when she got to the fair -/
def money_start : ℕ := money_left + money_difference

theorem maria_fair_money : money_start = 87 := by
  sorry

end NUMINAMATH_CALUDE_maria_fair_money_l141_14154


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l141_14127

/-- Represents a tetrahedron with vertices P, Q, R, and S. -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths. -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3√2. -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (15/4) * Real.sqrt 2
  }
  tetrahedronVolume t = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l141_14127


namespace NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l141_14105

-- Define the grades
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def is_satisfactory (g : Grade) : Prop :=
  g = Grade.B ∨ g = Grade.C ∨ g = Grade.D

-- Define the number of students for each grade
def num_students (g : Grade) : ℕ :=
  match g with
  | Grade.A => 8
  | Grade.B => 6
  | Grade.C => 5
  | Grade.D => 4
  | Grade.F => 7

-- Define the total number of students
def total_students : ℕ :=
  num_students Grade.A + num_students Grade.B + num_students Grade.C +
  num_students Grade.D + num_students Grade.F

-- Define the number of students with satisfactory grades
def satisfactory_students : ℕ :=
  num_students Grade.B + num_students Grade.C + num_students Grade.D

-- Theorem to prove
theorem fraction_of_satisfactory_grades :
  (satisfactory_students : ℚ) / total_students = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l141_14105


namespace NUMINAMATH_CALUDE_student_count_l141_14136

theorem student_count (bags : ℕ) (nuts_per_bag : ℕ) (nuts_per_student : ℕ) : 
  bags = 65 → nuts_per_bag = 15 → nuts_per_student = 75 → 
  (bags * nuts_per_bag) / nuts_per_student = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l141_14136


namespace NUMINAMATH_CALUDE_cosine_sine_difference_equals_sine_double_angle_l141_14194

theorem cosine_sine_difference_equals_sine_double_angle (α : ℝ) :
  (Real.cos (π / 4 - α))^2 - (Real.sin (π / 4 - α))^2 = Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_difference_equals_sine_double_angle_l141_14194


namespace NUMINAMATH_CALUDE_total_interest_calculation_l141_14186

/-- Given a principal amount and an interest rate, calculates the total interest after 10 years
    when the principal is trebled after 5 years and the initial 10-year simple interest is 1200. -/
theorem total_interest_calculation (P R : ℝ) : 
  (P * R * 10) / 100 = 1200 → 
  (P * R * 5) / 100 + (3 * P * R * 5) / 100 = 3000 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l141_14186


namespace NUMINAMATH_CALUDE_work_efficiency_l141_14122

/-- Given a person who takes x days to complete a task, and Tanya who is 25% more efficient
    and takes 12 days to complete the same task, prove that x is equal to 15 days. -/
theorem work_efficiency (x : ℝ) : 
  (∃ (person : ℝ → ℝ) (tanya : ℝ → ℝ), 
    (∀ t, tanya t = 0.75 * person t) ∧ 
    (tanya 12 = person x)) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_efficiency_l141_14122


namespace NUMINAMATH_CALUDE_sqrt_88_plus_42sqrt3_form_l141_14157

theorem sqrt_88_plus_42sqrt3_form : ∃ (a b c : ℤ), 
  (Real.sqrt (88 + 42 * Real.sqrt 3) = a + b * Real.sqrt c) ∧ 
  (∀ (k : ℕ), k > 1 → ¬(∃ (m : ℕ), c = k^2 * m)) ∧
  (a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88_plus_42sqrt3_form_l141_14157


namespace NUMINAMATH_CALUDE_savings_calculation_l141_14107

/-- Calculates the amount saved given sales, basic salary, commission rate, and savings rate -/
def calculate_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let total_earnings := basic_salary + sales * commission_rate
  total_earnings * savings_rate

/-- Proves that given the specified conditions, the amount saved is $29 -/
theorem savings_calculation :
  let sales := 2500
  let basic_salary := 240
  let commission_rate := 0.02
  let savings_rate := 0.10
  calculate_savings sales basic_salary commission_rate savings_rate = 29 := by
sorry

#eval calculate_savings 2500 240 0.02 0.10

end NUMINAMATH_CALUDE_savings_calculation_l141_14107


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l141_14128

theorem smallest_right_triangle_area :
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * Real.sqrt (c^2 - a^2)
  area = (7 * Real.sqrt 15) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l141_14128


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l141_14110

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 8920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l141_14110


namespace NUMINAMATH_CALUDE_expression_simplification_l141_14145

theorem expression_simplification (x : ℝ) : 
  (x^3 - 2)^2 + (x^2 + 2*x)^2 = x^6 + x^4 + 4*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l141_14145


namespace NUMINAMATH_CALUDE_a_power_value_l141_14125

theorem a_power_value (a n : ℝ) (h : a^(2*n) = 3) : 2*a^(6*n) - 1 = 53 := by
  sorry

end NUMINAMATH_CALUDE_a_power_value_l141_14125


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l141_14191

theorem sqrt_equality_condition (x : ℝ) : 
  Real.sqrt ((x + 1)^2 + (x - 1)^2) = (x + 1) - (x - 1) ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l141_14191


namespace NUMINAMATH_CALUDE_perimeter_3x3_grid_l141_14190

/-- The perimeter of a square grid of unit squares -/
def grid_perimeter (rows columns : ℕ) : ℕ :=
  2 * (rows + columns)

/-- Theorem: The perimeter of a 3x3 grid of unit squares is 18 -/
theorem perimeter_3x3_grid : grid_perimeter 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_3x3_grid_l141_14190


namespace NUMINAMATH_CALUDE_factor_expression_l141_14116

theorem factor_expression (x : ℝ) : 75*x + 45 = 15*(5*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l141_14116


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_16_min_value_achieved_l141_14158

theorem min_value_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 → a * b ≤ x * y := by
  sorry

theorem min_value_is_16 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  a * b ≥ 16 := by
  sorry

theorem min_value_achieved (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x * y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_16_min_value_achieved_l141_14158


namespace NUMINAMATH_CALUDE_final_x_value_l141_14199

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ
  s : ℕ

/-- Updates the state for one iteration -/
def update_state (st : State) : State :=
  { x := st.x + 3, s := st.s + st.x^2 }

/-- Checks if the termination condition is met -/
def terminate? (st : State) : Bool :=
  st.s ≥ 1000

/-- Runs the program until termination -/
def run_program : ℕ → State → State
  | 0, st => st
  | n + 1, st => if terminate? st then st else run_program n (update_state st)

/-- The initial state of the program -/
def initial_state : State :=
  { x := 4, s := 0 }

theorem final_x_value :
  (run_program 1000 initial_state).x = 22 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l141_14199


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l141_14101

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l141_14101


namespace NUMINAMATH_CALUDE_min_difference_unit_complex_l141_14126

theorem min_difference_unit_complex (z w : ℂ) 
  (hz : Complex.abs z = 1) 
  (hw : Complex.abs w = 1) 
  (h_sum : 1 ≤ Complex.abs (z + w) ∧ Complex.abs (z + w) ≤ Real.sqrt 2) : 
  Real.sqrt 2 ≤ Complex.abs (z - w) := by
  sorry

end NUMINAMATH_CALUDE_min_difference_unit_complex_l141_14126


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_property_l141_14196

/-- Given an ellipse and a circle, prove a property of tangents from a point on the ellipse to the circle -/
theorem ellipse_circle_tangent_property
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (P : ℝ × ℝ) (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (hP_not_vertex : P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧ P ≠ (0, b) ∧ P ≠ (0, -b))
  (A B : ℝ × ℝ)
  (hA : A.1^2 + A.2^2 = b^2)
  (hB : B.1^2 + B.2^2 = b^2)
  (hPA : P.1 * A.1 + P.2 * A.2 = b^2)
  (hPB : P.1 * B.1 + P.2 * B.2 = b^2)
  (M : ℝ × ℝ) (hM : M.2 = 0 ∧ M.1 * P.1 = b^2)
  (N : ℝ × ℝ) (hN : N.1 = 0 ∧ N.2 * P.2 = b^2) :
  a^2 / (N.2^2) + b^2 / (M.1^2) = a^2 / b^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_property_l141_14196


namespace NUMINAMATH_CALUDE_rectangle_ratio_l141_14198

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l141_14198


namespace NUMINAMATH_CALUDE_function_inequality_l141_14159

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := (1/3)^x - x^2

-- State the theorem
theorem function_inequality (x₀ x₁ x₂ m : ℝ) 
  (h1 : f x₀ = m) 
  (h2 : x₁ ∈ Set.Ioo 0 x₀) 
  (h3 : x₂ ∈ Set.Ioi x₀) : 
  f x₁ > m ∧ f x₂ < m := by
  sorry

end

end NUMINAMATH_CALUDE_function_inequality_l141_14159


namespace NUMINAMATH_CALUDE_stating_final_number_lower_bound_l141_14139

/-- 
Given a sequence of n ones, we repeatedly replace two numbers a and b 
with (a+b)/4 for n-1 steps. This function represents the final number 
after these operations.
-/
noncomputable def final_number (n : ℕ) : ℝ :=
  sorry

/-- 
Theorem stating that the final number after n-1 steps of the described 
operation, starting with n ones, is greater than or equal to 1/n.
-/
theorem final_number_lower_bound (n : ℕ) (h : n > 0) : 
  final_number n ≥ 1 / n :=
sorry

end NUMINAMATH_CALUDE_stating_final_number_lower_bound_l141_14139


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l141_14144

theorem wendy_trip_miles (total_miles second_day_miles first_day_miles third_day_miles : ℕ) :
  total_miles = 493 →
  first_day_miles = 125 →
  third_day_miles = 145 →
  second_day_miles = total_miles - first_day_miles - third_day_miles →
  second_day_miles = 223 := by
sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l141_14144


namespace NUMINAMATH_CALUDE_acute_angle_tangent_implies_a_equals_one_l141_14168

/-- The curve C: y = x^3 - 2ax^2 + 2ax -/
def C (a : ℤ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℤ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

theorem acute_angle_tangent_implies_a_equals_one (a : ℤ) :
  (∀ x : ℝ, C_derivative a x > 0) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_tangent_implies_a_equals_one_l141_14168


namespace NUMINAMATH_CALUDE_prime_sum_30_l141_14121

theorem prime_sum_30 (A B C : ℕ) : 
  Prime A ∧ Prime B ∧ Prime C ∧
  A < 20 ∧ B < 20 ∧ C < 20 ∧
  A + B + C = 30 →
  (A = 2 ∧ B = 11 ∧ C = 17) ∨
  (A = 2 ∧ B = 17 ∧ C = 11) ∨
  (A = 11 ∧ B = 2 ∧ C = 17) ∨
  (A = 11 ∧ B = 17 ∧ C = 2) ∨
  (A = 17 ∧ B = 2 ∧ C = 11) ∨
  (A = 17 ∧ B = 11 ∧ C = 2) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_30_l141_14121


namespace NUMINAMATH_CALUDE_gcd_problem_l141_14177

/-- The operation * represents the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: The GCD of ((12 * 16) * (18 * 12)) is 2 -/
theorem gcd_problem : gcd_op (gcd_op (gcd_op 12 16) (gcd_op 18 12)) 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l141_14177


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l141_14109

theorem quadratic_root_problem (b : ℝ) :
  ((-2 : ℝ)^2 + b * (-2) = 0) → (0^2 + b * 0 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l141_14109


namespace NUMINAMATH_CALUDE_cylinder_ellipse_major_axis_length_l141_14197

/-- Represents the properties of an ellipse formed by intersecting a right circular cylinder with a plane -/
structure CylinderEllipse where
  cylinder_radius : ℝ
  major_axis_ratio : ℝ

/-- Calculates the length of the major axis of the ellipse -/
def major_axis_length (e : CylinderEllipse) : ℝ :=
  2 * e.cylinder_radius * (1 + e.major_axis_ratio)

/-- Theorem stating the length of the major axis for the given conditions -/
theorem cylinder_ellipse_major_axis_length :
  let e : CylinderEllipse := { cylinder_radius := 3, major_axis_ratio := 0.4 }
  major_axis_length e = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_major_axis_length_l141_14197


namespace NUMINAMATH_CALUDE_pr_equals_21_l141_14141

/-- Triangle PQR with given side lengths -/
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℕ

/-- The triangle inequality theorem holds for the given triangle -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.PQ + t.PR > t.QR ∧ t.QR + t.PQ > t.PR ∧ t.PR + t.QR > t.PQ

/-- The theorem stating that PR = 21 satisfies the conditions -/
theorem pr_equals_21 (t : Triangle) 
  (h1 : t.PQ = 7) 
  (h2 : t.QR = 20) 
  (h3 : t.PR = 21) : 
  satisfies_triangle_inequality t :=
sorry

end NUMINAMATH_CALUDE_pr_equals_21_l141_14141


namespace NUMINAMATH_CALUDE_cos_sin_difference_equals_sqrt3_over_2_l141_14163

theorem cos_sin_difference_equals_sqrt3_over_2 :
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) -
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_difference_equals_sqrt3_over_2_l141_14163


namespace NUMINAMATH_CALUDE_not_a_gt_b_l141_14171

theorem not_a_gt_b (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : A / (1 / 5) = B * (1 / 4)) : A ≤ B := by
  sorry

end NUMINAMATH_CALUDE_not_a_gt_b_l141_14171


namespace NUMINAMATH_CALUDE_happy_valley_farm_arrangement_l141_14193

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_farm_arrangement :
  arrange_animals 5 3 4 = 103680 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_farm_arrangement_l141_14193


namespace NUMINAMATH_CALUDE_dividend_calculation_l141_14156

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l141_14156


namespace NUMINAMATH_CALUDE_smallest_consecutive_sum_divisible_by_17_l141_14165

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if two natural numbers are consecutive -/
def areConsecutive (a b : ℕ) : Prop := b = a + 1

/-- Check if there exist smaller consecutive numbers satisfying the condition -/
def existSmallerPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), x < a ∧ areConsecutive x y ∧ 
    (sumOfDigits x % 17 = 0) ∧ (sumOfDigits y % 17 = 0)

theorem smallest_consecutive_sum_divisible_by_17 :
  areConsecutive 8899 8900 ∧
  (sumOfDigits 8899 % 17 = 0) ∧
  (sumOfDigits 8900 % 17 = 0) ∧
  ¬(existSmallerPair 8899 8900) :=
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sum_divisible_by_17_l141_14165


namespace NUMINAMATH_CALUDE_negation_of_proposition_l141_14180

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l141_14180


namespace NUMINAMATH_CALUDE_total_pencils_l141_14174

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  monday = 35 ∧
  tuesday = 42 ∧
  wednesday = 3 * tuesday ∧
  thursday = wednesday / 2 ∧
  friday = 2 * monday

theorem total_pencils :
  ∀ monday tuesday wednesday thursday friday : ℕ,
    pencils_problem monday tuesday wednesday thursday friday →
    monday + tuesday + wednesday + thursday + friday = 336 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l141_14174


namespace NUMINAMATH_CALUDE_flagpole_break_height_l141_14160

/-- Proves that a 6-meter flagpole breaking and touching the ground 2 meters away
    from its base breaks at a height of 3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 6 →                            -- Total height of flagpole
  x > 0 →                            -- Breaking point is above ground
  x < h →                            -- Breaking point is below the top
  x^2 + 2^2 = (h - x)^2 →            -- Pythagorean theorem
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l141_14160


namespace NUMINAMATH_CALUDE_quadratic_inequality_sufficient_not_necessary_l141_14120

theorem quadratic_inequality_sufficient_not_necessary :
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ ¬(x^2 - 2*x < 0)) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sufficient_not_necessary_l141_14120


namespace NUMINAMATH_CALUDE_chimney_bricks_proof_l141_14114

/-- The number of hours it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- The number of hours it takes Brandon to build the chimney alone -/
def brandon_time : ℝ := 12

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 15

/-- The number of hours it takes Brenda and Brandon to build the chimney together -/
def combined_time : ℝ := 6

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 360

theorem chimney_bricks_proof : 
  combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_decrease) = chimney_bricks := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_proof_l141_14114


namespace NUMINAMATH_CALUDE_sum_fractions_equals_eight_l141_14140

/-- Given real numbers a, b, and c satisfying specific conditions, 
    prove that b/(a + b) + c/(b + c) + a/(c + a) = 8 -/
theorem sum_fractions_equals_eight 
  (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_equals_eight_l141_14140


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l141_14146

theorem rectangular_prism_volume (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66)
  : a * b * c = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l141_14146


namespace NUMINAMATH_CALUDE_angle_bisector_length_squared_l141_14118

/-- Given a triangle with sides a, b, and c, fa is the length of the angle bisector of angle α,
    and u and v are the lengths of the segments into which fa divides side a. -/
theorem angle_bisector_length_squared (a b c fa u v : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ fa > 0 ∧ u > 0 ∧ v > 0)
  (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b)
  (h_segments : u + v = a)
  (h_ratio : u / v = c / b) :
  fa^2 = b * c - u * v := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_squared_l141_14118


namespace NUMINAMATH_CALUDE_unique_number_l141_14112

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Odd n ∧ 
  n % 9 = 0 ∧ 
  is_perfect_square (digit_product n) :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l141_14112


namespace NUMINAMATH_CALUDE_shaded_area_between_squares_l141_14173

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h_large : large_side = 9)
  (h_small : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_squares_l141_14173


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l141_14108

theorem cubic_equation_solution (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 25 * y^3) 
  (h3 : a - b = y) : 
  b = -(1 - Real.sqrt 33) / 2 * y ∨ b = -(1 + Real.sqrt 33) / 2 * y := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l141_14108


namespace NUMINAMATH_CALUDE_farmers_market_spending_l141_14192

/-- Given Sandi's initial amount and Gillian's total spending, prove that Gillian spent $150 more than three times Sandi's spending. -/
theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ)
  (h1 : sandi_initial = 600)
  (h2 : gillian_total = 1050) :
  gillian_total - 3 * (sandi_initial / 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_farmers_market_spending_l141_14192


namespace NUMINAMATH_CALUDE_count_special_numbers_is_360_l141_14133

/-- A function that counts 4-digit numbers beginning with 2 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits : Finset ℕ := Finset.range 10
  let non_two_digits : Finset ℕ := digits.erase 2

  let case1 := 3 * non_two_digits.card * (non_two_digits.card - 1)
  let case2 := 3 * non_two_digits.card * (non_two_digits.card - 1)

  case1 + case2

/-- Theorem stating that the count of special numbers is 360 -/
theorem count_special_numbers_is_360 : count_special_numbers = 360 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_360_l141_14133


namespace NUMINAMATH_CALUDE_dog_speed_is_400_l141_14195

-- Define the constants from the problem
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def fetch_time_minutes : ℕ := 9
def feet_per_yard : ℕ := 3

-- Define the dog's speed as a function
def dog_speed : ℚ :=
  (football_fields * yards_per_field * feet_per_yard) / fetch_time_minutes

-- Theorem statement
theorem dog_speed_is_400 : dog_speed = 400 := by
  sorry

end NUMINAMATH_CALUDE_dog_speed_is_400_l141_14195


namespace NUMINAMATH_CALUDE_horizontal_cut_length_l141_14166

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  area : ℝ

/-- A horizontal cut in an isosceles triangle -/
structure HorizontalCut where
  triangle : IsoscelesTriangle
  trapezoidArea : ℝ
  cutLength : ℝ

/-- The main theorem -/
theorem horizontal_cut_length 
  (triangle : IsoscelesTriangle)
  (cut : HorizontalCut)
  (h1 : triangle.area = 144)
  (h2 : triangle.height = 24)
  (h3 : cut.triangle = triangle)
  (h4 : cut.trapezoidArea = 108) :
  cut.cutLength = 6 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_cut_length_l141_14166


namespace NUMINAMATH_CALUDE_three_does_not_divide_31_l141_14137

theorem three_does_not_divide_31 : ¬ ∃ q : ℤ, 31 = 3 * q := by
  sorry

end NUMINAMATH_CALUDE_three_does_not_divide_31_l141_14137


namespace NUMINAMATH_CALUDE_linear_equation_solution_l141_14104

/-- Given that (a - 3)x^(|a| - 2) + 6 = 0 is a linear equation in terms of x,
    prove that the solution is x = 1 -/
theorem linear_equation_solution (a : ℝ) :
  (∀ x, ∃ k m, (a - 3) * x^(|a| - 2) + 6 = k * x + m) →
  ∃! x, (a - 3) * x^(|a| - 2) + 6 = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l141_14104


namespace NUMINAMATH_CALUDE_joan_gave_two_balloons_l141_14142

/-- The number of blue balloons Joan gave to Jessica --/
def balloons_given_to_jessica (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

/-- Proof that Joan gave 2 balloons to Jessica --/
theorem joan_gave_two_balloons : 
  balloons_given_to_jessica 9 5 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_two_balloons_l141_14142


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l141_14148

theorem missing_digit_divisible_by_six (n : ℕ) (h1 : n ≥ 100 ∧ n < 1000) 
  (h2 : ∃ d : ℕ, d < 10 ∧ n = 500 + 10 * d + 2) (h3 : n % 6 = 0) : 
  ∃ d : ℕ, d = 2 ∧ n = 500 + 10 * d + 2 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l141_14148


namespace NUMINAMATH_CALUDE_prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l141_14115

/-- The probability that three students do not dine in the same cafeteria when randomly choosing between two cafeterias -/
theorem prob_not_same_cafeteria : ℚ :=
  let num_cafeterias : ℕ := 2
  let num_students : ℕ := 3
  let total_choices : ℕ := num_cafeterias ^ num_students
  let same_cafeteria_choices : ℕ := num_cafeterias
  let diff_cafeteria_choices : ℕ := total_choices - same_cafeteria_choices
  (diff_cafeteria_choices : ℚ) / total_choices

theorem prob_not_same_cafeteria_is_three_fourths :
  prob_not_same_cafeteria = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_prob_not_same_cafeteria_prob_not_same_cafeteria_is_three_fourths_l141_14115


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l141_14113

theorem scientific_notation_proof : 
  (55000000 : ℝ) = 5.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l141_14113


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l141_14131

theorem min_value_of_sum_of_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (a / c) = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l141_14131


namespace NUMINAMATH_CALUDE_vector_equation_solution_l141_14123

/-- Given vectors a and b, if 3a - 2b + c = 0, then c = (-23, -12) -/
theorem vector_equation_solution (a b c : ℝ × ℝ) :
  a = (5, 2) →
  b = (-4, -3) →
  3 • a - 2 • b + c = (0, 0) →
  c = (-23, -12) := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l141_14123


namespace NUMINAMATH_CALUDE_total_pebbles_is_50_l141_14111

/-- Represents the number of pebbles of each color and the total --/
structure PebbleCounts where
  white : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ

/-- Defines the conditions of the pebble problem --/
def pebble_problem (p : PebbleCounts) : Prop :=
  p.white = 20 ∧
  p.red = p.white / 2 ∧
  p.blue = p.red / 3 ∧
  p.green = p.blue + 5 ∧
  p.red = p.total / 5 ∧
  p.total = p.white + p.red + p.blue + p.green

/-- Theorem stating that the total number of pebbles is 50 --/
theorem total_pebbles_is_50 :
  ∃ p : PebbleCounts, pebble_problem p ∧ p.total = 50 :=
by sorry

end NUMINAMATH_CALUDE_total_pebbles_is_50_l141_14111
