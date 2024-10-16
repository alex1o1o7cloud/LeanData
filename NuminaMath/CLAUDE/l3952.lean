import Mathlib

namespace NUMINAMATH_CALUDE_intersection_lines_l3952_395261

-- Define the fixed points M₁ and M₂
def M₁ : ℝ × ℝ := (26, 1)
def M₂ : ℝ × ℝ := (2, 1)

-- Define the point P
def P : ℝ × ℝ := (-2, 3)

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - M₁.1)^2 + (y - M₁.2)^2) / ((x - M₂.1)^2 + (y - M₂.2)^2)) = 25

-- Define the trajectory of M
def trajectory (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define the chord length condition
def chord_length (l : ℝ → ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
    y₁ = l x₁ ∧ y₂ = l x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64

-- Theorem statement
theorem intersection_lines :
  ∀ (l : ℝ → ℝ),
    (∀ x, l x = -2 ∨ l x = (-5/12) * x + 23/6) ↔
    (∀ M, distance_ratio M → trajectory M.1 M.2) ∧
    chord_length l ∧
    l P.1 = P.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_lines_l3952_395261


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3952_395221

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 4 * y = 26) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 1076 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3952_395221


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l3952_395242

theorem smallest_solution_absolute_value_equation :
  let x : ℝ := (-3 - Real.sqrt 17) / 2
  (∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y) ∧ (x * |x| = 3 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l3952_395242


namespace NUMINAMATH_CALUDE_pentagonal_prism_diagonals_l3952_395293

/-- A regular pentagonal prism -/
structure RegularPentagonalPrism where
  /-- The number of vertices on each base -/
  base_vertices : ℕ
  /-- The total number of vertices -/
  total_vertices : ℕ
  /-- The number of base vertices is 5 -/
  base_is_pentagon : base_vertices = 5
  /-- The total number of vertices is twice the number of base vertices -/
  total_is_double_base : total_vertices = 2 * base_vertices

/-- A diagonal in a regular pentagonal prism -/
def is_diagonal (prism : RegularPentagonalPrism) (v1 v2 : ℕ) : Prop :=
  v1 ≠ v2 ∧ 
  v1 < prism.total_vertices ∧ 
  v2 < prism.total_vertices ∧
  (v1 < prism.base_vertices ↔ v2 ≥ prism.base_vertices)

/-- The total number of diagonals in a regular pentagonal prism -/
def total_diagonals (prism : RegularPentagonalPrism) : ℕ :=
  (prism.base_vertices * prism.base_vertices)

theorem pentagonal_prism_diagonals (prism : RegularPentagonalPrism) : 
  total_diagonals prism = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_diagonals_l3952_395293


namespace NUMINAMATH_CALUDE_inequality_proof_l3952_395264

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + y - 1)^2 / z + (y + z - 1)^2 / x + (z + x - 1)^2 / y ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3952_395264


namespace NUMINAMATH_CALUDE_smallest_palindrome_base3_l3952_395226

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_palindrome_base3 :
  ∀ n : ℕ,
  isPalindrome n 3 ∧ numDigits n 3 = 5 →
  (∃ b : ℕ, b ≠ 3 ∧ isPalindrome (convertBase n 3 b) b ∧ numDigits (convertBase n 3 b) b = 3) →
  n ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base3_l3952_395226


namespace NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l3952_395223

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l3952_395223


namespace NUMINAMATH_CALUDE_test_problem_value_l3952_395239

theorem test_problem_value (total_points total_problems four_point_problems : ℕ)
  (h1 : total_points = 100)
  (h2 : total_problems = 30)
  (h3 : four_point_problems = 10)
  (h4 : four_point_problems < total_problems) :
  (total_points - 4 * four_point_problems) / (total_problems - four_point_problems) = 3 :=
by sorry

end NUMINAMATH_CALUDE_test_problem_value_l3952_395239


namespace NUMINAMATH_CALUDE_set_equality_l3952_395278

def A : Set ℝ := {x : ℝ | |x| < 3}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem set_equality : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3952_395278


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3952_395266

theorem geometric_sequence_problem (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  q > 0 ∧ 
  (∀ n, a (n + 1) = a n * q) ∧ 
  (∀ n, a n > 0) ∧
  (a 1 = 1 / q^2) ∧
  (S 5 = S 2 + 2) →
  q = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3952_395266


namespace NUMINAMATH_CALUDE_chips_after_steps_chips_after_25_steps_l3952_395253

/-- Represents the state of trays with chips -/
def TrayState := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : TrayState :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Counts the number of true values in a list of booleans -/
def countTrueValues (l : List Bool) : Nat :=
  l.filter id |>.length

/-- The number of chips after n steps is equal to the number of 1s in the binary representation of n -/
theorem chips_after_steps (n : Nat) : 
  countTrueValues (toBinary n) = countTrueValues (toBinary n) := by sorry

/-- The number of chips after 25 steps is equal to the number of 1s in the binary representation of 25 -/
theorem chips_after_25_steps : 
  countTrueValues (toBinary 25) = 3 := by sorry

end NUMINAMATH_CALUDE_chips_after_steps_chips_after_25_steps_l3952_395253


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l3952_395270

/-- The number of emails Jack received in different parts of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Given information about Jack's email count -/
def jack_emails : EmailCount where
  morning := 5
  afternoon := 13 - 5
  evening := 72

/-- Theorem stating that Jack received 8 emails in the afternoon -/
theorem jack_afternoon_emails :
  jack_emails.afternoon = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l3952_395270


namespace NUMINAMATH_CALUDE_smallest_layer_sugar_l3952_395292

/-- Represents a three-layer cake with sugar requirements -/
structure ThreeLayerCake where
  smallest_layer : ℝ
  second_layer : ℝ
  third_layer : ℝ
  second_is_twice_first : second_layer = 2 * smallest_layer
  third_is_thrice_second : third_layer = 3 * second_layer
  third_layer_sugar : third_layer = 12

/-- Proves that the smallest layer of the cake requires 2 cups of sugar -/
theorem smallest_layer_sugar (cake : ThreeLayerCake) : cake.smallest_layer = 2 := by
  sorry

#check smallest_layer_sugar

end NUMINAMATH_CALUDE_smallest_layer_sugar_l3952_395292


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3952_395263

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x) ↔ 
  (k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3952_395263


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3952_395236

theorem coefficient_x3y5_in_expansion : 
  (Finset.range 9).sum (fun k => 
    if k = 3 then (Nat.choose 8 k : ℕ) 
    else 0) = 56 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l3952_395236


namespace NUMINAMATH_CALUDE_max_value_on_curve_l3952_395240

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2)

-- Define a point P on the curve C
def P (x y : ℝ) : Prop := ∃ (ρ θ : ℝ), C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem max_value_on_curve :
  ∀ (x y : ℝ), P x y → (∀ (x' y' : ℝ), P x' y' → 3 * x + 4 * y ≤ 3 * x' + 4 * y') →
  3 * x + 4 * y = Real.sqrt 145 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l3952_395240


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l3952_395238

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_two_times_sqrt_three_eq_sqrt_six : 
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l3952_395238


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l3952_395285

/-- The height of a right circular cylinder inscribed in a hemisphere --/
theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7)
  (h_inscribed : r_cylinder ≤ r_hemisphere) :
  Real.sqrt (r_hemisphere^2 - r_cylinder^2) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l3952_395285


namespace NUMINAMATH_CALUDE_prob_different_classes_correct_expected_value_class1_correct_l3952_395225

/-- Represents the number of classes in the first year -/
def num_classes : ℕ := 8

/-- Represents the total number of students selected for the community service group -/
def total_selected : ℕ := 10

/-- Represents the number of students selected from Class 1 -/
def class1_selected : ℕ := 3

/-- Represents the number of students selected from each of the other classes -/
def other_classes_selected : ℕ := 1

/-- Represents the number of students randomly selected for the activity -/
def activity_selected : ℕ := 3

/-- Probability of selecting 3 students from different classes -/
def prob_different_classes : ℚ := 49/60

/-- Expected value of the number of students selected from Class 1 -/
def expected_value_class1 : ℚ := 43/40

/-- Theorem stating the probability of selecting 3 students from different classes -/
theorem prob_different_classes_correct :
  let total_ways := Nat.choose total_selected activity_selected
  let ways_with_one_from_class1 := Nat.choose class1_selected 1 * Nat.choose (total_selected - class1_selected) 2
  let ways_with_none_from_class1 := Nat.choose class1_selected 0 * Nat.choose (total_selected - class1_selected) 3
  (ways_with_one_from_class1 + ways_with_none_from_class1) / total_ways = prob_different_classes :=
sorry

/-- Theorem stating the expected value of the number of students selected from Class 1 -/
theorem expected_value_class1_correct :
  let p0 := (7 : ℚ) / 24
  let p1 := (21 : ℚ) / 40
  let p2 := (7 : ℚ) / 40
  let p3 := (1 : ℚ) / 120
  0 * p0 + 1 * p1 + 2 * p2 + 3 * p3 = expected_value_class1 :=
sorry

end NUMINAMATH_CALUDE_prob_different_classes_correct_expected_value_class1_correct_l3952_395225


namespace NUMINAMATH_CALUDE_prime_sum_to_square_l3952_395233

theorem prime_sum_to_square (a b : ℕ) : 
  let P := (Nat.lcm a b / (a + 1)) + (Nat.lcm a b / (b + 1))
  Prime P → ∃ n : ℕ, 4 * P + 5 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_to_square_l3952_395233


namespace NUMINAMATH_CALUDE_next_number_is_two_l3952_395202

-- Define the sequence pattern
def sequence_pattern (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 
  let peak := n + 1
  let cycle_length := 2 * peak - 1
  let position := (m + 1) % cycle_length
  if position < peak then position + 1
  else 2 * peak - position - 1

-- Define the specific sequence from the problem
def given_sequence : List ℕ := [1, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 6, 5, 3, 1, 2, 3, 4, 5, 6, 7, 6, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 3, 1]

-- Theorem to prove
theorem next_number_is_two : 
  ∃ (n : ℕ), sequence_pattern n (given_sequence.length) = 2 :=
by sorry

end NUMINAMATH_CALUDE_next_number_is_two_l3952_395202


namespace NUMINAMATH_CALUDE_outfit_count_l3952_395276

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 3

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The number of tie options (including no tie). -/
def tie_options : ℕ := num_ties + 1

/-- The number of belt options (including no belt). -/
def belt_options : ℕ := num_belts + 1

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * belt_options

theorem outfit_count : total_outfits = 432 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l3952_395276


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3952_395244

theorem polynomial_product_expansion (x : ℝ) :
  (2 * x^3 - 3 * x^2 + 4) * (3 * x^2 + x + 1) =
  6 * x^5 - 7 * x^4 - x^3 + 9 * x^2 + 4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3952_395244


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3952_395237

theorem simplify_square_roots : 
  (Real.sqrt 507 / Real.sqrt 48) - (Real.sqrt 175 / Real.sqrt 112) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3952_395237


namespace NUMINAMATH_CALUDE_prank_combinations_l3952_395210

/-- The number of choices for each day of the prank --/
def prank_choices : List Nat := [1, 2, 6, 3, 1]

/-- The total number of combinations for the prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem prank_combinations :
  total_combinations = 36 := by sorry

end NUMINAMATH_CALUDE_prank_combinations_l3952_395210


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3952_395274

theorem quadratic_equation_real_roots (a : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + (a - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3952_395274


namespace NUMINAMATH_CALUDE_baseball_hits_theorem_l3952_395288

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem baseball_hits_theorem :
  singles = 35 ∧ (singles : ℚ) / total_hits * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_baseball_hits_theorem_l3952_395288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l3952_395260

/-- An arithmetic sequence with the given properties has the general term formula a_n = 2n -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 2 →  -- a_1 = 2
  (a 2 * a 8 = (a 4)^2) →  -- (a_2, a_4, a_8) forms a geometric sequence
  (∀ n, a n = 2 * n) :=  -- general term formula
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l3952_395260


namespace NUMINAMATH_CALUDE_square_root_problem_l3952_395267

theorem square_root_problem (x : ℝ) : Real.sqrt x - (Real.sqrt 625 / Real.sqrt 25) = 12 → x = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3952_395267


namespace NUMINAMATH_CALUDE_picture_distribution_l3952_395275

theorem picture_distribution (total : ℕ) (main_album : ℕ) (other_albums : ℕ) 
  (h1 : total = 33) 
  (h2 : main_album = 27) 
  (h3 : other_albums = 3) :
  (total - main_album) / other_albums = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_picture_distribution_l3952_395275


namespace NUMINAMATH_CALUDE_lcm_of_4_6_10_18_l3952_395208

theorem lcm_of_4_6_10_18 : Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 10 18)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_10_18_l3952_395208


namespace NUMINAMATH_CALUDE_division_with_same_remainder_l3952_395291

theorem division_with_same_remainder (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℤ, 200 = k * x + 2) :
  ∀ n : ℤ, ∃ k : ℤ, 200 = k * x + 2 ∧ n ≠ k → ∃ m : ℤ, n * x + 2 = m * x + (n * x + 2) % x ∧ (n * x + 2) % x = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_with_same_remainder_l3952_395291


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3952_395203

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) * (π / 3) = 60 * π :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3952_395203


namespace NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l3952_395211

-- Proposition B
theorem proposition_b (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (m + 1) / (n + 1) < m / n := by sorry

-- Proposition C
theorem proposition_c (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Proposition D
theorem proposition_d (a b : ℝ) (h1 : a ≥ b) (h2 : b > -1) :
  a / (a + 1) ≥ b / (b + 1) := by sorry

end NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l3952_395211


namespace NUMINAMATH_CALUDE_anthony_pencils_l3952_395222

/-- The number of pencils Anthony has after giving some to Kathryn -/
def pencils_remaining (initial : Float) (given : Float) : Float :=
  initial - given

/-- Theorem: Anthony has 47.0 pencils after giving some to Kathryn -/
theorem anthony_pencils :
  pencils_remaining 56.0 9.0 = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l3952_395222


namespace NUMINAMATH_CALUDE_irrationality_of_32_minus_sqrt3_l3952_395287

theorem irrationality_of_32_minus_sqrt3 : Irrational (32 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_irrationality_of_32_minus_sqrt3_l3952_395287


namespace NUMINAMATH_CALUDE_first_group_size_correct_l3952_395229

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 24

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l3952_395229


namespace NUMINAMATH_CALUDE_total_age_problem_l3952_395257

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 12 →
  a + b + c = 32 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l3952_395257


namespace NUMINAMATH_CALUDE_total_fish_count_l3952_395289

/-- The number of fish owned by each person -/
def lilly_fish : ℕ := 10
def rosy_fish : ℕ := 11
def alex_fish : ℕ := 15
def jamie_fish : ℕ := 8
def sam_fish : ℕ := 20

/-- Theorem stating that the total number of fish is 64 -/
theorem total_fish_count : 
  lilly_fish + rosy_fish + alex_fish + jamie_fish + sam_fish = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3952_395289


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l3952_395282

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (1 + x) = f (1 - x)
  min_value : ∃ x₀, ∀ x, f x ≥ f x₀ ∧ f x₀ = -1
  zero_at_zero : f 0 = 0

/-- The theorem stating that a quadratic function with the given properties
    is equal to x^2 - 2x -/
theorem quadratic_function_unique (qf : QuadraticFunction) :
  ∀ x, qf.f x = x^2 - 2*x := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_unique_l3952_395282


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3952_395245

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3952_395245


namespace NUMINAMATH_CALUDE_subtraction_equality_l3952_395286

theorem subtraction_equality : 8888888888888 - 4444444444444 = 4444444444444 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equality_l3952_395286


namespace NUMINAMATH_CALUDE_minimum_tents_l3952_395219

theorem minimum_tents (Y : ℕ) : (∃ X : ℕ, 
  X > 0 ∧ 
  10 * (X - 1) < (3 : ℚ) / 2 * Y ∧ (3 : ℚ) / 2 * Y < 10 * X ∧
  10 * (X + 2) < (8 : ℚ) / 5 * Y ∧ (8 : ℚ) / 5 * Y < 10 * (X + 3)) →
  Y ≥ 213 :=
by sorry

end NUMINAMATH_CALUDE_minimum_tents_l3952_395219


namespace NUMINAMATH_CALUDE_seventieth_number_with_remainder_five_seventieth_number_is_557_l3952_395227

theorem seventieth_number_with_remainder_five : ℕ → Prop :=
  fun n => ∃ k : ℕ, n = 8 * k + 5 ∧ n > 0

theorem seventieth_number_is_557 :
  ∃! n : ℕ, seventieth_number_with_remainder_five n ∧ (∃ m : ℕ, m = 70 ∧
    (∀ k < n, seventieth_number_with_remainder_five k →
      (∃ i : ℕ, i < m ∧ (∀ j < k, seventieth_number_with_remainder_five j → ∃ l : ℕ, l < i)))) ∧
  n = 557 :=
by sorry

end NUMINAMATH_CALUDE_seventieth_number_with_remainder_five_seventieth_number_is_557_l3952_395227


namespace NUMINAMATH_CALUDE_arrangement_count_is_150_l3952_395209

/-- The number of ways to arrange volunteers among events --/
def arrange_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The number of arrangements for 5 volunteers and 3 events --/
def arrangement_count : ℕ := arrange_volunteers 5 3

/-- Theorem: The number of arrangements for 5 volunteers and 3 events,
    such that each event has at least one participant, is 150 --/
theorem arrangement_count_is_150 : arrangement_count = 150 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_150_l3952_395209


namespace NUMINAMATH_CALUDE_solution_approximation_l3952_395297

/-- A linear function f. -/
noncomputable def f (x : ℝ) : ℝ := x

/-- The equation to be solved. -/
def equation (x : ℝ) : Prop :=
  f (x * 0.004) / 0.03 = 9.237333333333334

/-- The theorem stating that the solution to the equation is approximately 69.3. -/
theorem solution_approximation :
  ∃ x : ℝ, equation x ∧ abs (x - 69.3) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_solution_approximation_l3952_395297


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_bounds_l3952_395290

theorem geometric_sequence_sum_bounds (a : ℕ → ℚ) (S : ℕ → ℚ) (A B : ℚ) :
  (∀ n : ℕ, a n = 4/3 * (-1/3)^n) →
  (∀ n : ℕ, S (n+1) = (4/3 * (1 - (-1/3)^(n+1))) / (1 + 1/3)) →
  (∀ n : ℕ, n > 0 → A ≤ S n - 1 / S n ∧ S n - 1 / S n ≤ B) →
  59/72 ≤ B - A :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_bounds_l3952_395290


namespace NUMINAMATH_CALUDE_max_value_of_f_l3952_395280

def f (x : ℝ) : ℝ := -3 * x^2 + 18

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3952_395280


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3952_395204

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  M = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l3952_395204


namespace NUMINAMATH_CALUDE_min_volume_pyramid_l3952_395283

/-- A pyramid with a regular triangular base -/
structure Pyramid where
  base_side_length : ℝ
  apex_angle : ℝ

/-- The volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := sorry

/-- The constraint on the apex angle -/
def apex_angle_constraint (p : Pyramid) : Prop :=
  p.apex_angle ≤ 2 * Real.arcsin (1/3)

theorem min_volume_pyramid :
  ∃ (p : Pyramid),
    p.base_side_length = 6 ∧
    apex_angle_constraint p ∧
    (∀ (q : Pyramid),
      q.base_side_length = 6 →
      apex_angle_constraint q →
      volume p ≤ volume q) ∧
    volume p = 5 * Real.sqrt 23 :=
sorry

end NUMINAMATH_CALUDE_min_volume_pyramid_l3952_395283


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l3952_395268

/-- The function f(x) = (3 - x^2)e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotonic_increasing_interval (x : ℝ) : 
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l3952_395268


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l3952_395250

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l3952_395250


namespace NUMINAMATH_CALUDE_equation_proof_l3952_395201

theorem equation_proof : 10 * 6 - (9 - 3) * 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3952_395201


namespace NUMINAMATH_CALUDE_abc_inequality_l3952_395214

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  9 * a * b * c ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c < 1/4 + 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3952_395214


namespace NUMINAMATH_CALUDE_weight_of_a_l3952_395273

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 50 →
  (a + b + c + d) / 4 = 53 →
  (b + c + d + e) / 4 = 51 →
  e = d + 3 →
  a = 73 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l3952_395273


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3952_395249

theorem min_value_of_exponential_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  (9 : ℝ)^x + 3^y ≥ 54 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 6 ∧ (9 : ℝ)^x + 3^y = 54 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3952_395249


namespace NUMINAMATH_CALUDE_strictly_increasing_quadratic_function_l3952_395269

theorem strictly_increasing_quadratic_function (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → (x^2 - a*x) < (y^2 - a*y)) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_quadratic_function_l3952_395269


namespace NUMINAMATH_CALUDE_f_eval_neg_one_l3952_395234

-- Define the polynomials f and g
def f (p q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 200*x + r
def g (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20

-- State the theorem
theorem f_eval_neg_one (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g p x = 0 ∧ g p y = 0 ∧ g p z = 0) →
  (∀ x : ℝ, g p x = 0 → f p q r x = 0) →
  f p q r (-1) = -6319 :=
by sorry

end NUMINAMATH_CALUDE_f_eval_neg_one_l3952_395234


namespace NUMINAMATH_CALUDE_linear_function_b_values_l3952_395258

theorem linear_function_b_values (k b : ℝ) :
  (∀ x, -3 ≤ x ∧ x ≤ 1 → -1 ≤ k * x + b ∧ k * x + b ≤ 8) →
  b = 5/4 ∨ b = 23/4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_b_values_l3952_395258


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3952_395272

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧ 
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧ 
  A ≠ B

-- Theorem statement
theorem hyperbola_theorem 
  (center : ℝ × ℝ) 
  (right_focus : ℝ × ℝ) 
  (right_vertex : ℝ × ℝ) 
  (A B : ℝ × ℝ) :
  center = (0, 0) →
  right_focus = (2, 0) →
  right_vertex = (Real.sqrt 3, 0) →
  intersection_points A B →
  (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3952_395272


namespace NUMINAMATH_CALUDE_gcd_10_factorial_12_factorial_l3952_395230

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_10_factorial_12_factorial : Nat.gcd (factorial 10) (factorial 12) = factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10_factorial_12_factorial_l3952_395230


namespace NUMINAMATH_CALUDE_book_sale_amount_l3952_395235

/-- Calculates the total amount received from selling books given the following conditions:
  * A fraction of the books were sold
  * A certain number of books remained unsold
  * Each sold book was sold at a fixed price
-/
def totalAmountReceived (fractionSold : Rat) (remainingBooks : Nat) (pricePerBook : Rat) : Rat :=
  let totalBooks := remainingBooks / (1 - fractionSold)
  let soldBooks := totalBooks * fractionSold
  soldBooks * pricePerBook

/-- Proves that given the specific conditions of the book sale, 
    the total amount received is $255 -/
theorem book_sale_amount : 
  totalAmountReceived (2/3) 30 (21/5) = 255 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_amount_l3952_395235


namespace NUMINAMATH_CALUDE_inequalities_proof_l3952_395246

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (a^2 + b^2 ≥ 9/5) ∧ (a^3*b + 4*a*b^3 ≤ 81/16) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3952_395246


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3952_395256

/-- Given a hyperbola with equation x²/64 - y²/36 = 1, 
    its asymptotes have equations y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 
  (x^2 / 64 - y^2 / 36 = 1) →
  (∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3952_395256


namespace NUMINAMATH_CALUDE_prize_distribution_l3952_395279

theorem prize_distribution (total_winners : ℕ) (min_award : ℝ) (max_award : ℝ) : 
  total_winners = 20 →
  min_award = 20 →
  max_award = 340 →
  (∃ (prize : ℝ), 
    prize > 0 ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℝ), min_award ≤ award ∧ award ≤ max_award) ∧
    (2/5 * prize = 3/5 * total_winners * max_award) ∧
    prize = 10200) :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l3952_395279


namespace NUMINAMATH_CALUDE_binary_sum_equality_l3952_395271

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem binary_sum_equality : 
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true, false]  -- 1110₂
  let b4 := [true, false, true, true, true]  -- 10111₂
  let b5 := [true, true, false, false, false]  -- 11000₂
  let sum := [true, true, true, false, false, false, true, false]  -- 11100010₂
  binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + 
  binary_to_nat b4 + binary_to_nat b5 = binary_to_nat sum := by
  sorry

#eval binary_to_nat [true, true, true, false, false, false, true, false]  -- Should output 226

end NUMINAMATH_CALUDE_binary_sum_equality_l3952_395271


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l3952_395217

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M on circle C
def point_M (x₀ y₀ : ℝ) : Prop := circle_C x₀ y₀

-- Define the vector ON
def vector_ON (y₀ : ℝ) : ℝ × ℝ := (0, y₀)

-- Define the vector OQ as the sum of OM and ON
def vector_OQ (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, 2 * y₀)

-- State the theorem
theorem trajectory_of_Q (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, point_M x₀ y₀ ∧ vector_OQ x₀ y₀ = (x, y)) →
  x^2/4 + y^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l3952_395217


namespace NUMINAMATH_CALUDE_intersection_point_unique_l3952_395254

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / 3 ∧ (x - 3) / 2 = (z + 3) / 2

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (5, 2, -1)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line_equation p.1 p.2.1 p.2.2 ∧ 
    plane_equation p.1 p.2.1 p.2.2 ∧
    p = intersection_point :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l3952_395254


namespace NUMINAMATH_CALUDE_sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l3952_395232

theorem sqrt_seven_irrational :
  ∀ (a b : ℚ), a^2 ≠ 7 * b^2 :=
sorry

theorem rational_numbers :
  ∃ (q₁ q₂ q₃ : ℚ),
    (q₁ : ℝ) = 3.14159265 ∧
    (q₂ : ℝ) = Real.sqrt 36 ∧
    (q₃ : ℝ) = 4.1 :=
sorry

theorem sqrt_seven_is_irrational :
  Irrational (Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_sqrt_seven_irrational_rational_numbers_sqrt_seven_is_irrational_l3952_395232


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l3952_395205

/-- The volume of a rectangular solid with specific face areas and sum of dimensions -/
theorem rectangular_solid_volume
  (a b c : ℝ)
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (bottom_area : c * a = 6)
  (sum_dimensions : a + b + c = 11)
  : a * b * c = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l3952_395205


namespace NUMINAMATH_CALUDE_school_play_tickets_l3952_395243

theorem school_play_tickets (student_price adult_price adult_count total : ℕ) 
  (h1 : student_price = 6)
  (h2 : adult_price = 8)
  (h3 : adult_count = 12)
  (h4 : total = 216) :
  ∃ student_count : ℕ, student_count * student_price + adult_count * adult_price = total ∧ student_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l3952_395243


namespace NUMINAMATH_CALUDE_comparison_theorem_l3952_395248

theorem comparison_theorem :
  (2 * Real.sqrt 3 < 3 * Real.sqrt 2) ∧
  ((Real.sqrt 10 - 1) / 3 > 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l3952_395248


namespace NUMINAMATH_CALUDE_rectangle_width_three_l3952_395265

/-- A rectangle with length twice its width and area equal to perimeter has width 3. -/
theorem rectangle_width_three (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 6 * w) → w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_three_l3952_395265


namespace NUMINAMATH_CALUDE_flute_cost_calculation_l3952_395284

/-- The cost of Jason's purchases at the music store -/
def total_spent : ℝ := 158.35

/-- The cost of the music tool -/
def music_tool_cost : ℝ := 8.89

/-- The cost of the song book -/
def song_book_cost : ℝ := 7

/-- The cost of the flute -/
def flute_cost : ℝ := total_spent - (music_tool_cost + song_book_cost)

theorem flute_cost_calculation : flute_cost = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_calculation_l3952_395284


namespace NUMINAMATH_CALUDE_pure_imaginary_sixth_power_l3952_395241

theorem pure_imaginary_sixth_power (a : ℝ) (z : ℂ) :
  z = a + (a + 1) * Complex.I →
  z.im ≠ 0 →
  z.re = 0 →
  z^6 = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_sixth_power_l3952_395241


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3952_395259

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 8 * (x - y)) : x / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3952_395259


namespace NUMINAMATH_CALUDE_diane_honey_harvest_l3952_395277

/-- The total amount of honey harvested over three years -/
def total_honey_harvest (year1 : ℕ) (increase_year2 : ℕ) (increase_year3 : ℕ) : ℕ :=
  year1 + (year1 + increase_year2) + (year1 + increase_year2 + increase_year3)

/-- Theorem stating the total honey harvest over three years -/
theorem diane_honey_harvest :
  total_honey_harvest 2479 6085 7890 = 27497 := by
  sorry

end NUMINAMATH_CALUDE_diane_honey_harvest_l3952_395277


namespace NUMINAMATH_CALUDE_max_uncovered_sections_theorem_l3952_395281

/-- Represents a carpet with a given length -/
structure Carpet where
  length : ℝ

/-- Represents a corridor with a given length -/
structure Corridor where
  length : ℝ

/-- Represents a carpet arrangement in a corridor -/
structure CarpetArrangement where
  carpets : List Carpet
  corridor : Corridor

/-- Calculates the maximum number of uncovered sections in a carpet arrangement -/
def maxUncoveredSections (arrangement : CarpetArrangement) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of uncovered sections -/
theorem max_uncovered_sections_theorem 
  (arrangement : CarpetArrangement)
  (h1 : arrangement.carpets.length = 20)
  (h2 : (arrangement.carpets.map Carpet.length).sum = 1000)
  (h3 : arrangement.corridor.length = 100) :
  maxUncoveredSections arrangement = 11 :=
sorry

end NUMINAMATH_CALUDE_max_uncovered_sections_theorem_l3952_395281


namespace NUMINAMATH_CALUDE_twins_age_problem_l3952_395228

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 15 → age = 7 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l3952_395228


namespace NUMINAMATH_CALUDE_race_head_start_l3952_395299

theorem race_head_start (Va Vb L H : ℝ) :
  Va = (22 / 19) * Vb →
  L / Va = (L - H) / Vb →
  H = (3 / 22) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l3952_395299


namespace NUMINAMATH_CALUDE_integer_midpoint_exists_l3952_395207

def Point := ℤ × ℤ

theorem integer_midpoint_exists (P : Fin 5 → Point) :
  ∃ i j : Fin 5, i ≠ j ∧ 
    let (xi, yi) := P i
    let (xj, yj) := P j
    (xi + xj) % 2 = 0 ∧ (yi + yj) % 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_midpoint_exists_l3952_395207


namespace NUMINAMATH_CALUDE_right_triangle_area_l3952_395213

theorem right_triangle_area (a b : ℝ) (h1 : a = 25) (h2 : b = 20) :
  (1 / 2 : ℝ) * a * b = 250 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3952_395213


namespace NUMINAMATH_CALUDE_region_contains_point_c_l3952_395251

def point_in_region (x y : ℝ) : Prop :=
  x + 2*y - 1 > 0 ∧ x - y + 3 < 0

theorem region_contains_point_c :
  point_in_region 0 4 ∧
  ¬point_in_region (-4) 1 ∧
  ¬point_in_region 2 2 ∧
  ¬point_in_region (-2) 1 := by
  sorry

#check region_contains_point_c

end NUMINAMATH_CALUDE_region_contains_point_c_l3952_395251


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3952_395212

/-- Given an infinitely decreasing geometric progression with sum S and terms a₁, a₂, a₃, ...,
    prove that S / (S - a₁) = a₁ / a₂ -/
theorem geometric_progression_ratio (S a₁ a₂ : ℝ) (a : ℕ → ℝ) :
  (∀ n, a n = a₁ * (a₂ / a₁) ^ (n - 1)) →  -- Geometric progression definition
  (a₂ / a₁ < 1) →                          -- Decreasing condition
  (S = ∑' n, a n) →                        -- S is the sum of the progression
  S / (S - a₁) = a₁ / a₂ := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3952_395212


namespace NUMINAMATH_CALUDE_remaining_gift_card_value_l3952_395294

def bestBuyCardValue : ℕ := 500
def walmartCardValue : ℕ := 200

def initialBestBuyCards : ℕ := 6
def initialWalmartCards : ℕ := 9

def sentBestBuyCards : ℕ := 1
def sentWalmartCards : ℕ := 2

theorem remaining_gift_card_value :
  (initialBestBuyCards - sentBestBuyCards) * bestBuyCardValue +
  (initialWalmartCards - sentWalmartCards) * walmartCardValue = 3900 := by
  sorry

end NUMINAMATH_CALUDE_remaining_gift_card_value_l3952_395294


namespace NUMINAMATH_CALUDE_large_cheese_block_volume_l3952_395206

/-- The volume of a large cheese block is 32 cubic feet -/
theorem large_cheese_block_volume :
  ∀ (w d l : ℝ),
  w * d * l = 4 →
  (2 * w) * (2 * d) * (2 * l) = 32 :=
by sorry

end NUMINAMATH_CALUDE_large_cheese_block_volume_l3952_395206


namespace NUMINAMATH_CALUDE_skateboard_ramp_speed_increase_l3952_395295

/-- Calculates the additional speed required to reach the top of a skateboard ramp -/
theorem skateboard_ramp_speed_increase 
  (ramp_height : ℝ) 
  (ramp_incline : ℝ) 
  (speed_without_wind : ℝ) 
  (trial_speeds : List ℝ) 
  (wind_resistance_min : ℝ) 
  (wind_resistance_max : ℝ) : 
  ramp_height = 50 → 
  ramp_incline = 30 → 
  speed_without_wind = 40 → 
  trial_speeds = [36, 34, 38] → 
  wind_resistance_min = 3 → 
  wind_resistance_max = 5 → 
  (List.sum trial_speeds / trial_speeds.length + 
   (wind_resistance_min + wind_resistance_max) / 2 + 
   speed_without_wind) - 
  (List.sum trial_speeds / trial_speeds.length) = 8 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_ramp_speed_increase_l3952_395295


namespace NUMINAMATH_CALUDE_divisibility_condition_l3952_395218

theorem divisibility_condition (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ (x + y^3) % (x^2 + y^2) = 0 →
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨
  (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3952_395218


namespace NUMINAMATH_CALUDE_faucet_fill_time_l3952_395255

-- Define the constants from the problem
def tub_size_1 : ℝ := 200  -- Size of the first tub in gallons
def tub_size_2 : ℝ := 50   -- Size of the second tub in gallons
def faucets_1 : ℝ := 4     -- Number of faucets for the first tub
def faucets_2 : ℝ := 8     -- Number of faucets for the second tub
def time_1 : ℝ := 12       -- Time to fill the first tub in minutes

-- Define the theorem
theorem faucet_fill_time :
  ∃ (rate : ℝ),
    (rate * faucets_1 * time_1 = tub_size_1) ∧
    (rate * faucets_2 * (90 / 60) = tub_size_2) :=
by sorry

end NUMINAMATH_CALUDE_faucet_fill_time_l3952_395255


namespace NUMINAMATH_CALUDE_convex_polyhedron_inequalities_l3952_395216

/-- Represents a convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  faces_at_least_three_edges : 2 * edges ≥ 3 * faces

/-- The inequalities for convex polyhedrons. -/
theorem convex_polyhedron_inequalities (p : ConvexPolyhedron) :
  (3 * p.vertices ≥ 6 + p.faces) ∧ (3 * p.edges ≥ 6 + p.faces) := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_inequalities_l3952_395216


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3952_395220

theorem solution_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x = 4 ∧ a * x - 3 = 4 * x + 1) → a = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3952_395220


namespace NUMINAMATH_CALUDE_first_book_length_l3952_395224

theorem first_book_length :
  ∀ (book1 book2 total_pages daily_pages days : ℕ),
    book2 = 100 →
    days = 14 →
    daily_pages = 20 →
    total_pages = daily_pages * days →
    book1 + book2 = total_pages →
    book1 = 180 := by
sorry

end NUMINAMATH_CALUDE_first_book_length_l3952_395224


namespace NUMINAMATH_CALUDE_largest_common_difference_and_terms_l3952_395247

def is_decreasing_arithmetic_progression (a b c : ℤ) : Prop :=
  ∃ d : ℤ, d < 0 ∧ b = a + d ∧ c = a + 2*d

def has_two_roots (a b c : ℤ) : Prop :=
  b^2 - 4*a*c ≥ 0

theorem largest_common_difference_and_terms 
  (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : is_decreasing_arithmetic_progression a b c)
  (h3 : has_two_roots (2*a) (2*b) c)
  (h4 : has_two_roots (2*a) c (2*b))
  (h5 : has_two_roots (2*b) (2*a) c)
  (h6 : has_two_roots (2*b) c (2*a))
  (h7 : has_two_roots c (2*a) (2*b))
  (h8 : has_two_roots c (2*b) (2*a)) :
  ∃ d : ℤ, d = -5 ∧ a = 4 ∧ b = -1 ∧ c = -6 ∧ 
  ∀ d' : ℤ, (∃ a' b' c' : ℤ, 
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧
    is_decreasing_arithmetic_progression a' b' c' ∧
    has_two_roots (2*a') (2*b') c' ∧
    has_two_roots (2*a') c' (2*b') ∧
    has_two_roots (2*b') (2*a') c' ∧
    has_two_roots (2*b') c' (2*a') ∧
    has_two_roots c' (2*a') (2*b') ∧
    has_two_roots c' (2*b') (2*a') ∧
    d' < 0) → d' ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_common_difference_and_terms_l3952_395247


namespace NUMINAMATH_CALUDE_roots_sum_abs_l3952_395298

theorem roots_sum_abs (d e f n : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = (x - d) * (x - e) * (x - f)) →
  abs d + abs e + abs f = 98 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_abs_l3952_395298


namespace NUMINAMATH_CALUDE_watch_dealer_profit_l3952_395296

theorem watch_dealer_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ m : ℕ, d = 3 * m) →
  (10 * n - 30 = 100) →
  (∀ k : ℕ, k < n → ¬(10 * k - 30 = 100)) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_watch_dealer_profit_l3952_395296


namespace NUMINAMATH_CALUDE_vector_BA_complex_l3952_395215

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector BA is their difference. -/
theorem vector_BA_complex (OA OB : ℂ) (h1 : OA = 2 - 3*I) (h2 : OB = -3 + 2*I) :
  OA - OB = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_vector_BA_complex_l3952_395215


namespace NUMINAMATH_CALUDE_finite_ring_power_equality_l3952_395200

theorem finite_ring_power_equality (A : Type) [Ring A] [Fintype A] :
  ∃ (m p : ℕ), m > p ∧ p ≥ 1 ∧ ∀ (a : A), a^m = a^p := by
  sorry

end NUMINAMATH_CALUDE_finite_ring_power_equality_l3952_395200


namespace NUMINAMATH_CALUDE_equal_distribution_of_treats_l3952_395262

def cookies : ℕ := 20
def cupcakes : ℕ := 25
def brownies : ℕ := 35
def students : ℕ := 20

theorem equal_distribution_of_treats :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_treats_l3952_395262


namespace NUMINAMATH_CALUDE_motorcycles_in_parking_lot_l3952_395252

theorem motorcycles_in_parking_lot :
  let total_wheels : ℕ := 117
  let num_cars : ℕ := 19
  let wheels_per_car : ℕ := 5
  let wheels_per_motorcycle : ℕ := 2
  let num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle
  num_motorcycles = 11 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_parking_lot_l3952_395252


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l3952_395231

theorem sum_of_powers_of_two (n : ℕ) : 
  (1 : ℚ) / 2^10 + 1 / 2^9 + 1 / 2^8 = n / 2^10 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l3952_395231
