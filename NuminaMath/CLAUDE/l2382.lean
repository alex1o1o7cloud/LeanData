import Mathlib

namespace NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l2382_238235

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial_sum : ℕ := (List.range 20).foldl (fun acc i => acc + factorial ((i + 1) * 5)) 0

theorem last_two_digits_of_factorial_sum :
  last_two_digits factorial_sum = 20 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_factorial_sum_l2382_238235


namespace NUMINAMATH_CALUDE_parallel_line_with_chord_l2382_238294

/-- Given a line parallel to 3x + 3y + 5 = 0 and intercepted by the circle x² + y² = 20
    with a chord length of 6√2, prove that the equation of the line is x + y ± 2 = 0 -/
theorem parallel_line_with_chord (a b c : ℝ) : 
  (∃ k : ℝ, a = 3 * k ∧ b = 3 * k) → -- Line is parallel to 3x + 3y + 5 = 0
  (∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 ≤ 20) → -- Line intersects the circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    x₁^2 + y₁^2 = 20 ∧
    x₂^2 + y₂^2 = 20 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 72) → -- Chord length is 6√2
  ∃ s : ℝ, (s = 1 ∨ s = -1) ∧ a * x + b * y + c = 0 ↔ x + y + 2 * s = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_with_chord_l2382_238294


namespace NUMINAMATH_CALUDE_characterize_nat_function_l2382_238245

/-- A function from natural numbers to natural numbers -/
def NatFunction := ℕ → ℕ

/-- Predicate that checks if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Theorem statement -/
theorem characterize_nat_function (f : NatFunction) :
  (∀ m n : ℕ, IsPerfectSquare (f n + 2 * m * n + f m)) →
  ∃ ℓ : ℕ, ∀ n : ℕ, f n = (n + 2 * ℓ)^2 - 2 * ℓ^2 :=
by sorry

end NUMINAMATH_CALUDE_characterize_nat_function_l2382_238245


namespace NUMINAMATH_CALUDE_investment_problem_l2382_238272

/-- Proves that given the conditions of the investment problem, the initial sum invested was $900 -/
theorem investment_problem (P : ℝ) : 
  P > 0 → 
  (P * (4.5 / 100) * 7) - (P * (4 / 100) * 7) = 31.5 → 
  P = 900 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l2382_238272


namespace NUMINAMATH_CALUDE_point_on_line_l2382_238256

/-- Given a line passing through points (0, 2) and (-10, 0),
    prove that the point (25, 7) lies on this line. -/
theorem point_on_line : ∀ (x y : ℝ),
  (x = 25 ∧ y = 7) →
  (y - 2) * 10 = (x - 0) * 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2382_238256


namespace NUMINAMATH_CALUDE_set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l2382_238212

-- Set 1
def set1 : Set ℤ := {x | x.natAbs ≤ 2}
def set1_alt : Set ℤ := {-2, -1, 0, 1, 2}

theorem set1_equality : set1 = set1_alt := by sorry

-- Set 2
def set2 : Set ℕ := {x | x > 0 ∧ x % 3 = 0 ∧ x < 10}
def set2_alt : Set ℕ := {3, 6, 9}

theorem set2_equality : set2 = set2_alt := by sorry

-- Set 3
def set3 : Set ℤ := {x | x = Int.natAbs x ∧ x < 5}
def set3_alt : Set ℤ := {0, 1, 2, 3, 4}

theorem set3_equality : set3 = set3_alt := by sorry

-- Set 4
def set4 : Set (ℕ+ × ℕ+) := {p | p.1 + p.2 = 6}
def set4_alt : Set (ℕ+ × ℕ+) := {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

theorem set4_equality : set4 = set4_alt := by sorry

-- Set 5
def set5 : Set ℤ := {-3, -1, 1, 3, 5}
def set5_alt : Set ℤ := {x | ∃ k : ℤ, x = 2*k - 1 ∧ -1 ≤ k ∧ k ≤ 3}

theorem set5_equality : set5 = set5_alt := by sorry

end NUMINAMATH_CALUDE_set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l2382_238212


namespace NUMINAMATH_CALUDE_green_fish_count_l2382_238290

theorem green_fish_count (total : ℕ) (blue : ℕ) (orange : ℕ) (green : ℕ) : 
  total = 80 →
  blue = total / 2 →
  orange = blue - 15 →
  total = blue + orange + green →
  green = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_fish_count_l2382_238290


namespace NUMINAMATH_CALUDE_original_sugar_percentage_l2382_238283

/-- Given a solution where one fourth is replaced by a 42% sugar solution,
    resulting in an 18% sugar solution, prove that the original solution
    must have been 10% sugar. -/
theorem original_sugar_percentage
  (original : ℝ)
  (replaced : ℝ := 1/4)
  (second_solution : ℝ := 42)
  (final_solution : ℝ := 18)
  (h : (1 - replaced) * original + replaced * second_solution = final_solution) :
  original = 10 :=
sorry

end NUMINAMATH_CALUDE_original_sugar_percentage_l2382_238283


namespace NUMINAMATH_CALUDE_average_weight_increase_l2382_238271

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 70 →
  new_weight = 94 →
  (new_weight - replaced_weight) / initial_count = 3 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2382_238271


namespace NUMINAMATH_CALUDE_a_10_value_l2382_238229

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 9)
  (h_13 : a 13 = -3) :
  a 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_a_10_value_l2382_238229


namespace NUMINAMATH_CALUDE_smallest_m_for_candies_l2382_238280

theorem smallest_m_for_candies : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬(10 ∣ 15*k ∧ 18 ∣ 15*k ∧ 20 ∣ 15*k)) ∧
  (10 ∣ 15*m ∧ 18 ∣ 15*m ∧ 20 ∣ 15*m) ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_candies_l2382_238280


namespace NUMINAMATH_CALUDE_b_amount_l2382_238211

theorem b_amount (a b : ℚ) 
  (h1 : a + b = 2530)
  (h2 : (3/5) * a = (2/7) * b) : 
  b = 1714 := by sorry

end NUMINAMATH_CALUDE_b_amount_l2382_238211


namespace NUMINAMATH_CALUDE_all_defective_impossible_l2382_238216

structure ProductSet where
  total : ℕ
  defective : ℕ
  drawn : ℕ
  h_total : total = 10
  h_defective : defective = 2
  h_drawn : drawn = 3
  h_defective_lt_total : defective < total

def all_defective (s : ProductSet) : Prop :=
  ∀ (i : Fin s.drawn), i.val < s.defective

theorem all_defective_impossible (s : ProductSet) : ¬ (all_defective s) := by
  sorry

end NUMINAMATH_CALUDE_all_defective_impossible_l2382_238216


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2382_238237

theorem quadratic_factorization (x : ℝ) : x^2 - 3*x - 4 = (x + 1)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2382_238237


namespace NUMINAMATH_CALUDE_line_parameterization_l2382_238261

/-- Given a line y = 5x - 7 parameterized as (x, y) = (s, -3) + t(3, m),
    prove that s = 4/5 and m = 8 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t x y : ℝ, x = s + 3*t ∧ y = -3 + m*t → y = 5*x - 7) →
  s = 4/5 ∧ m = 8 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2382_238261


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2382_238225

theorem nested_fraction_evaluation : 
  let f (x : ℝ) := (x + 2) / (x - 2)
  let g (x : ℝ) := (f x + 2) / (f x - 2)
  g 1 = 1/5 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2382_238225


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2382_238266

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the arithmetic sequence equals 120. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

/-- The main theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the difference between a_7 and one-third of a_5 is 16. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : SumCondition a) : 
    a 7 - (1/3) * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2382_238266


namespace NUMINAMATH_CALUDE_handshake_partition_handshake_same_neighbors_l2382_238215

open Set

structure HandshakeGraph (α : Type*) [Fintype α] where
  edges : Set (α × α)
  symm : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges
  handshake_property : ∀ a b c d, (a, b) ∈ edges → (b, c) ∈ edges → (c, d) ∈ edges →
    (a, c) ∈ edges ∨ (a, d) ∈ edges ∨ (b, d) ∈ edges

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem handshake_partition (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (X Y : Set (Fin n)), X.Nonempty ∧ Y.Nonempty ∧ X ∪ Y = univ ∧ X ∩ Y = ∅ ∧
  (∀ x y, x ∈ X → y ∈ Y → ((x, y) ∈ G.edges ↔ ∀ a ∈ X, ∀ b ∈ Y, (a, b) ∈ G.edges)) :=
sorry

theorem handshake_same_neighbors (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (A B : Fin n), A ≠ B ∧
  {x | x ≠ A ∧ x ≠ B ∧ (A, x) ∈ G.edges} = {x | x ≠ A ∧ x ≠ B ∧ (B, x) ∈ G.edges} :=
sorry

end NUMINAMATH_CALUDE_handshake_partition_handshake_same_neighbors_l2382_238215


namespace NUMINAMATH_CALUDE_eight_amp_two_l2382_238242

/-- Custom binary operation & -/
def amp (a b : ℤ) : ℤ := (a + b) * (a - b) + a * b

/-- Theorem: 8 & 2 = 76 -/
theorem eight_amp_two : amp 8 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_eight_amp_two_l2382_238242


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2382_238232

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^r in the expansion of (x + 1/(2x))^n -/
def coeff (n r : ℕ) : ℚ := (binomial n r) * (1 / 2 ^ r)

theorem binomial_expansion_properties (n : ℕ) (h : n ≥ 2) :
  (2 * coeff n 1 = coeff n 0 + coeff n 2 ↔ n = 8) ∧
  (n = 8 → coeff n 2 = 7) := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2382_238232


namespace NUMINAMATH_CALUDE_soccer_team_starters_l2382_238289

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quadruplets_in_lineup : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 3 →
  (Nat.choose quadruplets quadruplets_in_lineup) * (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l2382_238289


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2382_238233

theorem original_denominator_problem (d : ℕ) : 
  (3 : ℚ) / d ≠ 0 → 
  (6 : ℚ) / (d + 3) = 1 / 3 → 
  d = 15 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2382_238233


namespace NUMINAMATH_CALUDE_g_of_5_l2382_238220

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- State the theorem
theorem g_of_5 (a b c : ℝ) : g a b c (-5) = -3 → g a b c 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l2382_238220


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2382_238282

theorem largest_prime_divisor_to_test (n : ℕ) (h : 1000 ≤ n ∧ n ≤ 1100) :
  (∀ p : ℕ, p.Prime → p ≤ 31 → ¬(p ∣ n)) → n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2382_238282


namespace NUMINAMATH_CALUDE_h_of_2_equals_2_l2382_238265

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^31 - 1)

-- Theorem statement
theorem h_of_2_equals_2 : h 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_h_of_2_equals_2_l2382_238265


namespace NUMINAMATH_CALUDE_shoveling_time_bounds_l2382_238284

/-- Represents the snow shoveling scenario -/
structure SnowShoveling where
  initialRate : ℕ  -- Initial shoveling rate in cubic yards per hour
  rateDecrease : ℕ  -- Rate decrease per hour in cubic yards
  driveWidth : ℕ  -- Driveway width in yards
  driveLength : ℕ  -- Driveway length in yards
  snowDepth : ℕ  -- Snow depth in yards

/-- Calculates the time taken to shovel the driveway clean -/
def shovelingTime (s : SnowShoveling) : ℕ :=
  sorry

/-- Theorem stating that it takes at least 9 hours and less than 10 hours to clear the driveway -/
theorem shoveling_time_bounds (s : SnowShoveling) 
  (h1 : s.initialRate = 30)
  (h2 : s.rateDecrease = 2)
  (h3 : s.driveWidth = 4)
  (h4 : s.driveLength = 10)
  (h5 : s.snowDepth = 5) :
  9 ≤ shovelingTime s ∧ shovelingTime s < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_shoveling_time_bounds_l2382_238284


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2382_238243

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence where the 4th term is 8 and the 10th term is 2, the 7th term is 1. -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_4th : a 4 = 8)
  (h_10th : a 10 = 2)
  : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2382_238243


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2382_238208

theorem real_part_of_complex_fraction : 
  (5 * Complex.I / (1 + 2 * Complex.I)).re = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l2382_238208


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l2382_238238

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 4) : 
  (triangles + circles : ℚ) / total = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l2382_238238


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2382_238270

/-- Represents the number of students in each grade and the total sample size -/
structure SchoolSample where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from each grade -/
def stratifiedSample (school : SchoolSample) : ℕ × ℕ × ℕ :=
  let sample_fraction := school.sample_size / school.total_students
  let first := school.first_grade * sample_fraction
  let second := school.second_grade * sample_fraction
  let third := school.third_grade * sample_fraction
  (first, second, third)

/-- Theorem stating the correct stratified sample for the given school -/
theorem correct_stratified_sample 
  (school : SchoolSample) 
  (h1 : school.total_students = 900)
  (h2 : school.first_grade = 300)
  (h3 : school.second_grade = 200)
  (h4 : school.third_grade = 400)
  (h5 : school.sample_size = 45) :
  stratifiedSample school = (15, 10, 20) := by
  sorry

#eval stratifiedSample { 
  total_students := 900, 
  first_grade := 300, 
  second_grade := 200, 
  third_grade := 400, 
  sample_size := 45 
}

end NUMINAMATH_CALUDE_correct_stratified_sample_l2382_238270


namespace NUMINAMATH_CALUDE_lcm_of_1540_and_660_l2382_238278

theorem lcm_of_1540_and_660 : Nat.lcm 1540 660 = 4620 := by sorry

end NUMINAMATH_CALUDE_lcm_of_1540_and_660_l2382_238278


namespace NUMINAMATH_CALUDE_share_ratio_problem_l2382_238251

theorem share_ratio_problem (total : ℝ) (share_A : ℝ) (ratio_B_C : ℚ) 
  (h_total : total = 116000)
  (h_share_A : share_A = 29491.525423728814)
  (h_ratio_B_C : ratio_B_C = 5/6) :
  ∃ (share_B : ℝ), share_A / share_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_problem_l2382_238251


namespace NUMINAMATH_CALUDE_statements_correctness_l2382_238263

-- Define the statements
def statement_A (l : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, l = {(x, y) | x + y = c} ∧ (-2, -3) ∈ l ∧ c = -5

def statement_B (m : ℝ) : Prop :=
  (1, 3) ∈ {(x, y) | 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0}

def statement_C (θ : ℝ) : Prop :=
  ∀ x y : ℝ, y - 1 = Real.tan θ * (x - 1) ↔ (x, y) ∈ {(x, y) | y - 1 = Real.tan θ * (x - 1)}

def statement_D (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x y : ℝ, (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁) ↔
    (x, y) ∈ {(x, y) | (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁)}

-- Theorem stating which statements are correct and incorrect
theorem statements_correctness :
  (∃ l : Set (ℝ × ℝ), ¬statement_A l) ∧
  (∀ m : ℝ, statement_B m) ∧
  (∃ θ : ℝ, ¬statement_C θ) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, statement_D x₁ y₁ x₂ y₂) := by
  sorry


end NUMINAMATH_CALUDE_statements_correctness_l2382_238263


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_seven_largest_n_is_99996_l2382_238286

theorem largest_n_multiple_of_seven (n : ℕ) : n < 100000 →
  (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0 →
  n ≤ 99996 :=
by sorry

theorem largest_n_is_99996 :
  (9 * (99996 - 3)^6 - 99996^3 + 16 * 99996 - 27) % 7 = 0 ∧
  99996 < 100000 ∧
  ∀ m : ℕ, m < 100000 →
    (9 * (m - 3)^6 - m^3 + 16 * m - 27) % 7 = 0 →
    m ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_seven_largest_n_is_99996_l2382_238286


namespace NUMINAMATH_CALUDE_inequality_proof_l2382_238207

theorem inequality_proof (a b c : ℝ) (h : a * b < 0) :
  a^2 + b^2 + c^2 > 2*a*b + 2*b*c + 2*c*a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2382_238207


namespace NUMINAMATH_CALUDE_fastest_student_requires_comprehensive_survey_l2382_238246

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  requiredMethod : SurveyMethod

/-- Define the four survey scenarios -/
def viewershipSurvey : SurveyScenario :=
  { description := "Investigating the viewership rate of the Spring Festival Gala"
    requiredMethod := SurveyMethod.Sample }

def colorantSurvey : SurveyScenario :=
  { description := "Investigating whether the colorant content of a certain food in the market meets national standards"
    requiredMethod := SurveyMethod.Sample }

def shoeSoleSurvey : SurveyScenario :=
  { description := "Testing the number of times the shoe soles produced by a shoe factory can withstand bending"
    requiredMethod := SurveyMethod.Sample }

def fastestStudentSurvey : SurveyScenario :=
  { description := "Selecting the fastest student in short-distance running at a certain school to participate in the city-wide competition"
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem stating that selecting the fastest student requires a comprehensive survey -/
theorem fastest_student_requires_comprehensive_survey :
  fastestStudentSurvey.requiredMethod = SurveyMethod.Comprehensive ∧
  viewershipSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  colorantSurvey.requiredMethod ≠ SurveyMethod.Comprehensive ∧
  shoeSoleSurvey.requiredMethod ≠ SurveyMethod.Comprehensive :=
sorry

end NUMINAMATH_CALUDE_fastest_student_requires_comprehensive_survey_l2382_238246


namespace NUMINAMATH_CALUDE_margo_walking_distance_l2382_238222

/-- The total distance walked by Margo given her walking times and average speed -/
theorem margo_walking_distance 
  (time_to_friend : ℝ) 
  (time_from_friend : ℝ) 
  (average_speed : ℝ) 
  (h1 : time_to_friend = 15)
  (h2 : time_from_friend = 10)
  (h3 : average_speed = 3.6) : 
  (time_to_friend + time_from_friend) / 60 * average_speed = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_margo_walking_distance_l2382_238222


namespace NUMINAMATH_CALUDE_minimum_students_with_girl_percentage_l2382_238276

theorem minimum_students_with_girl_percentage (n : ℕ) (g : ℕ) : n > 0 → g > 0 → (25 : ℚ) / 100 < (g : ℚ) / n → (g : ℚ) / n < (30 : ℚ) / 100 → n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_minimum_students_with_girl_percentage_l2382_238276


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2382_238214

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
def condition : Prop := z * (1 + 2*Complex.I) = Complex.abs (-3 + 4*Complex.I)

-- State the theorem
theorem imaginary_part_of_z (h : condition z) : Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2382_238214


namespace NUMINAMATH_CALUDE_cubic_polynomials_inequality_l2382_238204

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Finset ℝ := sorry

/-- Check if all roots of a polynomial are positive -/
def all_roots_positive (p : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, r > 0

/-- Given two cubic polynomials, check if the roots of one are reciprocals of the other -/
def roots_are_reciprocals (p q : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, (1 / r) ∈ roots q

theorem cubic_polynomials_inequality (p q : CubicPolynomial) 
  (h_positive : all_roots_positive p)
  (h_reciprocal : roots_are_reciprocals p q) :
  p.a * q.a > 9 ∧ p.b * q.b > 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_inequality_l2382_238204


namespace NUMINAMATH_CALUDE_fuji_to_total_ratio_l2382_238205

/-- Represents an apple orchard with Fuji and Gala trees -/
structure AppleOrchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The ratio of pure Fuji trees to all trees in the orchard is 39:52 -/
theorem fuji_to_total_ratio (orchard : AppleOrchard) :
  orchard.crossPollinated = (orchard.totalTrees : ℚ) * (1/10) ∧
  orchard.pureFuji + orchard.crossPollinated = 221 ∧
  orchard.pureGala = 39 →
  (orchard.pureFuji : ℚ) / orchard.totalTrees = 39 / 52 :=
by sorry

end NUMINAMATH_CALUDE_fuji_to_total_ratio_l2382_238205


namespace NUMINAMATH_CALUDE_point_on_ellipse_l2382_238217

/-- The coordinates of a point P on an ellipse satisfying specific conditions -/
theorem point_on_ellipse (x y : ℝ) : 
  x > 0 → -- P is on the right side of y-axis
  x^2 / 5 + y^2 / 4 = 1 → -- P is on the ellipse
  (1/2) * 2 * |y| = 1 → -- Area of triangle PF₁F₂ is 1
  (x = Real.sqrt 15 / 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_point_on_ellipse_l2382_238217


namespace NUMINAMATH_CALUDE_water_level_drop_l2382_238231

/-- The water level drop in a cylindrical container when two spheres are removed -/
theorem water_level_drop (container_radius : ℝ) (sphere_diameter : ℝ) : 
  container_radius = 5 →
  sphere_diameter = 5 →
  (π * container_radius^2 * (5/3)) = (2 * (4/3) * π * (sphere_diameter/2)^3) :=
by sorry

end NUMINAMATH_CALUDE_water_level_drop_l2382_238231


namespace NUMINAMATH_CALUDE_quadratic_counterexample_l2382_238293

theorem quadratic_counterexample :
  ∃ m : ℝ, m < -2 ∧ ∀ x : ℝ, x^2 + m*x + 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_counterexample_l2382_238293


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2382_238206

theorem quadratic_equation_properties (m : ℝ) :
  let f (x : ℝ) := x^2 - (2*m - 1)*x - 3*m^2 + m
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₂/x₁ + x₁/x₂ = -5/2 → (m = 1 ∨ m = 2/5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2382_238206


namespace NUMINAMATH_CALUDE_canoe_travel_time_l2382_238200

/-- Given two villages A and B connected by a river with current velocity v_r,
    and a canoe with velocity v in still water, prove that if the time to travel
    from A to B is 3 times the time to travel from B to A, and v = 2*v_r, then
    the time to travel from B to A without paddles is 3 times longer than with paddles. -/
theorem canoe_travel_time (v v_r : ℝ) (S : ℝ) (h1 : v > 0) (h2 : v_r > 0) (h3 : S > 0) :
  (S / (v + v_r) = 3 * S / (v - v_r)) → (v = 2 * v_r) → (S / v_r = 3 * S / (v - v_r)) := by
  sorry

end NUMINAMATH_CALUDE_canoe_travel_time_l2382_238200


namespace NUMINAMATH_CALUDE_twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l2382_238285

theorem twenty_two_percent_of_300 : ℝ → Prop :=
  fun result => (22 / 100 : ℝ) * 300 = result

theorem prove_twenty_two_percent_of_300 : twenty_two_percent_of_300 66 := by
  sorry

end NUMINAMATH_CALUDE_twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l2382_238285


namespace NUMINAMATH_CALUDE_summer_camp_group_size_l2382_238210

/-- The number of children in Mrs. Generous' summer camp group -/
def num_children : ℕ := 31

/-- The number of jelly beans Mrs. Generous brought -/
def total_jelly_beans : ℕ := 500

/-- The number of jelly beans left after distribution -/
def leftover_jelly_beans : ℕ := 10

/-- The difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 3

theorem summer_camp_group_size :
  ∃ (girls boys : ℕ),
    girls + boys = num_children ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_group_size_l2382_238210


namespace NUMINAMATH_CALUDE_largest_expression_l2382_238267

theorem largest_expression : 
  let a := 3 + 1 + 4
  let b := 3 * 1 + 4
  let c := 3 + 1 * 4
  let d := 3 * 1 * 4
  let e := 3 + 0 * 1 + 4
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2382_238267


namespace NUMINAMATH_CALUDE_total_discount_percentage_l2382_238209

theorem total_discount_percentage (initial_discount subsequent_discount : ℝ) : 
  initial_discount = 0.25 → 
  subsequent_discount = 0.35 → 
  1 - (1 - initial_discount) * (1 - subsequent_discount) = 0.5125 := by
sorry

end NUMINAMATH_CALUDE_total_discount_percentage_l2382_238209


namespace NUMINAMATH_CALUDE_segment_construction_l2382_238213

/-- Given positive real numbers a, b, c, d, and e, there exists a real number x
    such that x = (a * b * c) / (d * e). -/
theorem segment_construction (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hd : d > 0) (he : e > 0) : ∃ x : ℝ, x = (a * b * c) / (d * e) := by
  sorry

end NUMINAMATH_CALUDE_segment_construction_l2382_238213


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2382_238236

theorem unique_solution_condition (h : ℝ) (h_neq_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2382_238236


namespace NUMINAMATH_CALUDE_cubic_sum_geq_product_sum_l2382_238258

theorem cubic_sum_geq_product_sum {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  a^3 + b^3 + c^3 ≥ a*b + b*c + c*a ∧ 
  (a^3 + b^3 + c^3 = a*b + b*c + c*a ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_geq_product_sum_l2382_238258


namespace NUMINAMATH_CALUDE_cinema_seat_removal_l2382_238295

/-- The number of seats that should be removed from a cinema with
    total_seats arranged in rows of seats_per_row, given expected_attendees,
    to minimize unoccupied seats while ensuring full rows. -/
def seats_to_remove (total_seats seats_per_row expected_attendees : ℕ) : ℕ :=
  total_seats - (((expected_attendees + seats_per_row - 1) / seats_per_row) * seats_per_row)

/-- Theorem stating that for the given cinema setup, 88 seats should be removed. -/
theorem cinema_seat_removal :
  seats_to_remove 240 8 150 = 88 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seat_removal_l2382_238295


namespace NUMINAMATH_CALUDE_eight_b_value_l2382_238277

theorem eight_b_value (a b : ℚ) 
  (eq1 : 4 * a + 3 * b = 5)
  (eq2 : a = b - 3) :
  8 * b = 136 / 7 := by
sorry

end NUMINAMATH_CALUDE_eight_b_value_l2382_238277


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l2382_238202

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

/-- The equation of the ellipse we want to prove -/
def target_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The foci of an ellipse with equation ax^2 + by^2 = c -/
def foci (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = (1/a - 1/b) * c ∧ y = 0}

theorem ellipse_equation_proof :
  (∀ x y, given_ellipse x y ↔ 3 * x^2 + 8 * y^2 = 24) →
  (target_ellipse 3 2) →
  (foci 3 8 24 = foci (1/15) (1/10) 1) →
  ∀ x y, target_ellipse x y ↔ x^2 / 15 + y^2 / 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l2382_238202


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2382_238291

theorem candy_bar_cost 
  (num_soft_drinks : ℕ) 
  (cost_per_soft_drink : ℕ) 
  (num_candy_bars : ℕ) 
  (total_spent : ℕ) 
  (h1 : num_soft_drinks = 2)
  (h2 : cost_per_soft_drink = 4)
  (h3 : num_candy_bars = 5)
  (h4 : total_spent = 28) :
  (total_spent - num_soft_drinks * cost_per_soft_drink) / num_candy_bars = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2382_238291


namespace NUMINAMATH_CALUDE_negation_equivalence_l2382_238254

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2382_238254


namespace NUMINAMATH_CALUDE_binomial_8_2_l2382_238260

theorem binomial_8_2 : Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_2_l2382_238260


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2382_238240

/-- The repeating decimal 0.51246246246... -/
def repeating_decimal : ℚ := 
  51246 / 100000 + (246 / 100000) * (1 / (1 - 1 / 1000))

/-- The fraction representation -/
def fraction : ℚ := 511734 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2382_238240


namespace NUMINAMATH_CALUDE_three_solutions_implies_a_gt_one_l2382_238224

/-- The equation has three different real solutions -/
def has_three_solutions (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (1 / (x + 2) = a * |x|) ∧
    (1 / (y + 2) = a * |y|) ∧
    (1 / (z + 2) = a * |z|)

/-- If the equation has three different real solutions, then a > 1 -/
theorem three_solutions_implies_a_gt_one :
  ∀ a : ℝ, has_three_solutions a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_implies_a_gt_one_l2382_238224


namespace NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l2382_238247

/-- Two isosceles triangles with equal vertical angles and areas in ratio 16:36 have heights in ratio 2:3 -/
theorem isosceles_triangles_height_ratio (b₁ b₂ h₁ h₂ : ℝ) (area₁ area₂ : ℝ) :
  b₁ > 0 → b₂ > 0 → h₁ > 0 → h₂ > 0 →
  area₁ = (b₁ * h₁) / 2 →
  area₂ = (b₂ * h₂) / 2 →
  area₁ / area₂ = 16 / 36 →
  b₁ / b₂ = h₁ / h₂ →
  h₁ / h₂ = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l2382_238247


namespace NUMINAMATH_CALUDE_main_theorem_l2382_238268

/-- The logarithm function with base 2 -/
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

/-- The main function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log2 (x + a)

/-- The companion function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * log2 (4*x + a)

/-- The difference function F(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

theorem main_theorem (a : ℝ) (h : a > 0) :
  (∀ x, f a x < -1 ↔ -a < x ∧ x < 1/2 - a) ∧
  (∀ x ∈ Set.Ioo 0 2, f a x < g a x ↔ 0 < a ∧ a ≤ 1) ∧
  (∃ M, M = 1 - (1/2) * log2 3 ∧ 
    ∀ x ∈ Set.Ioo 0 2, |F 1 x| ≤ M ∧
    ∃ x₀ ∈ Set.Ioo 0 2, |F 1 x₀| = M) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l2382_238268


namespace NUMINAMATH_CALUDE_prime_sum_difference_l2382_238264

theorem prime_sum_difference (m n p : ℕ) 
  (hm : Nat.Prime m) (hn : Nat.Prime n) (hp : Nat.Prime p)
  (h_pos : 0 < p ∧ 0 < n ∧ 0 < m)
  (h_order : m > n ∧ n > p)
  (h_sum : m + n + p = 74)
  (h_diff : m - n - p = 44) :
  m = 59 ∧ n = 13 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_difference_l2382_238264


namespace NUMINAMATH_CALUDE_parabola_translation_l2382_238273

/-- Represents a parabola in the form y = -(x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { h := p.h + dx, k := p.k + dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0
  let translated := translate original 1 2
  translated = Parabola.mk 2 2 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2382_238273


namespace NUMINAMATH_CALUDE_quadratic_square_form_sum_l2382_238297

theorem quadratic_square_form_sum (x : ℝ) :
  ∃ (a b c : ℤ), a > 0 ∧
  (25 * x^2 + 30 * x - 35 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 52 := by sorry

end NUMINAMATH_CALUDE_quadratic_square_form_sum_l2382_238297


namespace NUMINAMATH_CALUDE_thursday_coffee_consumption_l2382_238201

/-- Represents the professor's coffee consumption model -/
structure CoffeeModel where
  k : ℝ
  coffee : ℝ → ℝ → ℝ
  wednesday_meetings : ℝ
  wednesday_sleep : ℝ
  wednesday_coffee : ℝ
  thursday_meetings : ℝ
  thursday_sleep : ℝ

/-- Theorem stating the professor's coffee consumption on Thursday -/
theorem thursday_coffee_consumption (model : CoffeeModel) 
  (h1 : model.coffee m h = model.k * m / h)
  (h2 : model.wednesday_coffee = model.coffee model.wednesday_meetings model.wednesday_sleep)
  (h3 : model.wednesday_meetings = 3)
  (h4 : model.wednesday_sleep = 8)
  (h5 : model.wednesday_coffee = 3)
  (h6 : model.thursday_meetings = 5)
  (h7 : model.thursday_sleep = 10) :
  model.coffee model.thursday_meetings model.thursday_sleep = 4 := by
  sorry

end NUMINAMATH_CALUDE_thursday_coffee_consumption_l2382_238201


namespace NUMINAMATH_CALUDE_peggy_final_doll_count_l2382_238255

/-- Calculates the final number of dolls Peggy has after a series of events --/
def finalDollCount (initialDolls : ℕ) (grandmotherGift : ℕ) : ℕ :=
  let birthdayGift := grandmotherGift / 2
  let afterBirthday := initialDolls + grandmotherGift + birthdayGift
  let afterSpringCleaning := afterBirthday - (afterBirthday / 10)
  let easterGift := birthdayGift / 3
  let afterEaster := afterSpringCleaning + easterGift
  let afterExchange := afterEaster - 1
  let christmasGift := easterGift + (easterGift / 5)
  let afterChristmas := afterExchange + christmasGift
  afterChristmas - 3

/-- Theorem stating that Peggy ends up with 50 dolls --/
theorem peggy_final_doll_count :
  finalDollCount 6 28 = 50 := by
  sorry

end NUMINAMATH_CALUDE_peggy_final_doll_count_l2382_238255


namespace NUMINAMATH_CALUDE_ratio_s5_s8_l2382_238227

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_ratio : (4 * (2 * a 1 + 3 * (a 2 - a 1))) / (6 * (2 * a 1 + 5 * (a 2 - a 1))) = -2/3

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The main theorem -/
theorem ratio_s5_s8 (seq : ArithmeticSequence) : 
  (sum_n seq 5) / (sum_n seq 8) = 1 / 40.8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_s5_s8_l2382_238227


namespace NUMINAMATH_CALUDE_printer_problem_l2382_238228

/-- Given a total of 42 pages, where every 7th page is crumpled and every 3rd page is blurred,
    the number of pages that are neither crumpled nor blurred is 24. -/
theorem printer_problem (total_pages : Nat) (crumple_interval : Nat) (blur_interval : Nat)
    (h1 : total_pages = 42)
    (h2 : crumple_interval = 7)
    (h3 : blur_interval = 3) :
    total_pages - (total_pages / crumple_interval + total_pages / blur_interval - total_pages / (crumple_interval * blur_interval)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_printer_problem_l2382_238228


namespace NUMINAMATH_CALUDE_not_on_line_l2382_238219

-- Define the quadratic function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x)
def g (a b c x x_1 x_2 : ℝ) : ℝ := f a b c (x - x_1) + f a b c (x - x_2)

theorem not_on_line (a b c x_1 x_2 : ℝ) 
  (h1 : ∃ x_1 x_2, f a b c x_1 = 0 ∧ f a b c x_2 = 0) -- f has two zeros
  (h2 : f a b c 1 = 2 * a) -- f(1) = 2a
  (h3 : a > c) -- a > c
  (h4 : ∀ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 ≤ 2 / a) -- max of g(x) in [0,1] is 2/a
  (h5 : ∃ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 = 2 / a) -- max of g(x) in [0,1] is achieved
  : a + b ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_not_on_line_l2382_238219


namespace NUMINAMATH_CALUDE_problem_statement_l2382_238287

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 4*b ≤ x + 4*y) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/a^2 + 2/b^2 ≤ 1/x^2 + 2/y^2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ a + 4*b = x + 4*y ∧ a + 4*b = 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 1/a^2 + 2/b^2 = 1/x^2 + 2/y^2 ∧ 1/a^2 + 2/b^2 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2382_238287


namespace NUMINAMATH_CALUDE_number_of_B_l2382_238234

/-- Given that the number of A is x and the number of B is a less than half of A,
    prove that the number of B is equal to (1/2)x - a. -/
theorem number_of_B (x a : ℝ) (hA : x ≥ 0) (hB : x ≥ 2 * a) :
  (1/2 : ℝ) * x - a = (1/2 : ℝ) * x - a :=
by sorry

end NUMINAMATH_CALUDE_number_of_B_l2382_238234


namespace NUMINAMATH_CALUDE_function_translation_l2382_238274

/-- Given a function f(x) = 3 * sin(2x + π/3), prove that translating it right by π/6 units
    and then downwards by 1 unit results in the function g(x) = 3 * sin(2x) - 1 -/
theorem function_translation (x : ℝ) :
  let f := λ x : ℝ => 3 * Real.sin (2 * x + π / 3)
  let g := λ x : ℝ => 3 * Real.sin (2 * x) - 1
  f (x - π / 6) - 1 = g x := by
  sorry

end NUMINAMATH_CALUDE_function_translation_l2382_238274


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2382_238239

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2382_238239


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l2382_238221

/-- Represents the price reduction and sales model of a shopping mall -/
structure MallSales where
  initialCost : ℝ
  initialPrice : ℝ
  initialSales : ℝ
  salesIncrease : ℝ
  priceReduction : ℝ

/-- Calculates the daily profit based on the given sales model -/
def dailyProfit (m : MallSales) : ℝ :=
  let newSales := m.initialSales + m.salesIncrease * m.priceReduction
  let newProfit := m.initialPrice - m.initialCost - m.priceReduction
  newSales * newProfit

/-- Theorem stating that a price reduction of 30 yuan results in a daily profit of 3600 yuan -/
theorem optimal_price_reduction (m : MallSales) 
  (h1 : m.initialCost = 220)
  (h2 : m.initialPrice = 280)
  (h3 : m.initialSales = 30)
  (h4 : m.salesIncrease = 3)
  : dailyProfit { m with priceReduction := 30 } = 3600 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_l2382_238221


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2382_238252

/-- The line l with equation (a-2)y = (3a-1)x - 1 does not pass through the second quadrant
    if and only if a ∈ [2, +∞) -/
theorem line_not_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → ¬(x < 0 ∧ y > 0)) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2382_238252


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l2382_238218

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l2382_238218


namespace NUMINAMATH_CALUDE_inequality_multiplication_l2382_238248

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l2382_238248


namespace NUMINAMATH_CALUDE_constant_value_l2382_238259

/-- A function satisfying the given conditions -/
def f (c : ℝ) : ℝ → ℝ :=
  fun x ↦ sorry

/-- The theorem stating the problem conditions and conclusion -/
theorem constant_value (c : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x + 3 * f (c - x) = x) →
  f 2 = 2 →
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l2382_238259


namespace NUMINAMATH_CALUDE_no_rational_solutions_l2382_238262

theorem no_rational_solutions (n : ℕ) (x y : ℚ) : (x + Real.sqrt 3 * y) ^ n ≠ Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_l2382_238262


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l2382_238298

theorem smallest_sum_consecutive_integers (n : ℕ) : 
  (∀ k < n, k * (k + 1) ≤ 420) → n * (n + 1) > 420 → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l2382_238298


namespace NUMINAMATH_CALUDE_calculator_result_l2382_238223

def calculator_operation (n : ℕ) : ℕ :=
  let doubled := n * 2
  let swapped := (doubled % 10) * 10 + (doubled / 10)
  swapped + 2

def is_valid_input (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 49

theorem calculator_result :
  (∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 44) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 43) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 42) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 41) :=
sorry

end NUMINAMATH_CALUDE_calculator_result_l2382_238223


namespace NUMINAMATH_CALUDE_function_is_even_l2382_238249

/-- A function satisfying certain properties is even -/
theorem function_is_even (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (1 + x) = -f x)
  (h3 : ¬ ∀ x y, f x = f y) : 
  ∀ x, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_function_is_even_l2382_238249


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2382_238269

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : Point × Point := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Check if three points are on the same circle -/
def on_same_circle (p1 p2 p3 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (center radius : Point) (p1 p2 : Point) : Prop := sorry

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  is_on_hyperbola h p ∧
  on_same_circle f1 f2 p ∧
  circle_tangent_to_segment origin f1 p f2 →
  eccentricity h = (3 + 6 * Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2382_238269


namespace NUMINAMATH_CALUDE_fraction_equality_l2382_238296

theorem fraction_equality : (8 : ℚ) / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2382_238296


namespace NUMINAMATH_CALUDE_marys_max_earnings_l2382_238203

/-- Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  overtimeRateIncrease : ℚ

/-- Calculate Mary's maximum weekly earnings --/
def calculateMaxEarnings (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularRate * schedule.regularHours
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := schedule.regularRate * (1 + schedule.overtimeRateIncrease)
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Mary's specific work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 40
  , regularHours := 20
  , regularRate := 8
  , overtimeRateIncrease := 1/4 }

/-- Theorem: Mary's maximum weekly earnings are $360 --/
theorem marys_max_earnings :
  calculateMaxEarnings marysSchedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_marys_max_earnings_l2382_238203


namespace NUMINAMATH_CALUDE_sugar_water_ratio_l2382_238281

theorem sugar_water_ratio (total_cups sugar_cups : ℕ) : 
  total_cups = 84 → sugar_cups = 28 → 
  ∃ (a b : ℕ), a = 1 ∧ b = 2 ∧ sugar_cups * b = (total_cups - sugar_cups) * a :=
by sorry

end NUMINAMATH_CALUDE_sugar_water_ratio_l2382_238281


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_triples_l2382_238288

def reciprocal_sum_one (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 3, 6), (2, 6, 3), (3, 2, 6), (3, 6, 2), (6, 2, 3), (6, 3, 2),
   (2, 4, 4), (4, 2, 4), (4, 4, 2), (3, 3, 3)}

theorem reciprocal_sum_one_triples :
  ∀ (a b c : ℕ+), reciprocal_sum_one a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_triples_l2382_238288


namespace NUMINAMATH_CALUDE_negation_of_difference_l2382_238257

theorem negation_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end NUMINAMATH_CALUDE_negation_of_difference_l2382_238257


namespace NUMINAMATH_CALUDE_sunset_time_correct_l2382_238244

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of daylight -/
structure DaylightLength where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight length -/
def calculate_sunset (sunrise : Time) (daylight : DaylightLength) : Time :=
  sorry

theorem sunset_time_correct (sunrise : Time) (daylight : DaylightLength) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 56 →
  let sunset := calculate_sunset sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 39 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_correct_l2382_238244


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2382_238292

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b - c = 0) : (a^3 + b^3 + c^3) / (a * b * c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2382_238292


namespace NUMINAMATH_CALUDE_correct_comparison_l2382_238230

theorem correct_comparison :
  (-5/6 : ℚ) < -4/5 ∧
  ¬(-(-21) < -21) ∧
  ¬(-(abs (-21/2)) > 26/3) ∧
  ¬(-(abs (-23/3)) > -(-23/3)) :=
by sorry

end NUMINAMATH_CALUDE_correct_comparison_l2382_238230


namespace NUMINAMATH_CALUDE_smallest_possible_d_l2382_238275

theorem smallest_possible_d (c d : ℝ) : 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (2/c + 2/d ≤ 2) → 
  d ≥ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l2382_238275


namespace NUMINAMATH_CALUDE_sector_central_angle_l2382_238253

/-- Given a circular sector with arc length 4 and area 4, prove that its central angle in radians is 2. -/
theorem sector_central_angle (arc_length area : ℝ) (h1 : arc_length = 4) (h2 : area = 4) :
  let r := 2 * area / arc_length
  2 * area / (r ^ 2) = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2382_238253


namespace NUMINAMATH_CALUDE_stratified_sample_ninth_grade_l2382_238241

/-- Represents the number of students in each grade and the sample size for 7th grade -/
structure SchoolData where
  total : ℕ
  seventh : ℕ
  eighth : ℕ
  ninth : ℕ
  sample_seventh : ℕ

/-- Calculates the sample size for 9th grade using stratified sampling -/
def stratified_sample (data : SchoolData) : ℕ :=
  (data.sample_seventh * data.ninth) / data.seventh

/-- Theorem stating that the stratified sample for 9th grade is 224 given the school data -/
theorem stratified_sample_ninth_grade 
  (data : SchoolData) 
  (h1 : data.total = 1700)
  (h2 : data.seventh = 600)
  (h3 : data.eighth = 540)
  (h4 : data.ninth = 560)
  (h5 : data.sample_seventh = 240) :
  stratified_sample data = 224 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_ninth_grade_l2382_238241


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2382_238250

theorem sum_of_A_and_B : ∀ (A B : ℚ), 3/7 = 6/A ∧ 6/A = B/21 → A + B = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2382_238250


namespace NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l2382_238226

-- Define the types for our transformations
def CentralSymmetry (center : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry
def Translation (vector : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define composition of transformations
def Compose (f g : (ℝ × ℝ) → (ℝ × ℝ)) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Theorem 1: Composition of two central symmetries is a translation
theorem composition_central_symmetries_is_translation 
  (c1 c2 : ℝ × ℝ) : 
  ∃ (v : ℝ × ℝ), Compose (CentralSymmetry c2) (CentralSymmetry c1) = Translation v := by sorry

-- Theorem 2: Composition of translation and central symmetry (both orders) is a central symmetry
theorem composition_translation_central_symmetry_is_central_symmetry 
  (v : ℝ × ℝ) (c : ℝ × ℝ) : 
  (∃ (c1 : ℝ × ℝ), Compose (Translation v) (CentralSymmetry c) = CentralSymmetry c1) ∧
  (∃ (c2 : ℝ × ℝ), Compose (CentralSymmetry c) (Translation v) = CentralSymmetry c2) := by sorry

end NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l2382_238226


namespace NUMINAMATH_CALUDE_find_k_value_l2382_238279

/-- Given two functions f and g, prove that if f(5) - g(5) = 12, then k = -53/5 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 8)
  (hg : ∀ x, g x = x^2 - k * x + 3)
  (h_diff : f 5 - g 5 = 12) : 
  k = -53/5 := by sorry

end NUMINAMATH_CALUDE_find_k_value_l2382_238279


namespace NUMINAMATH_CALUDE_gabriel_diabetes_capsules_l2382_238299

theorem gabriel_diabetes_capsules (forgot_days took_days : ℕ) 
  (h1 : forgot_days = 3) 
  (h2 : took_days = 28) : 
  forgot_days + took_days = 31 := by
  sorry

end NUMINAMATH_CALUDE_gabriel_diabetes_capsules_l2382_238299
