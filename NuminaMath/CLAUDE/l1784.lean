import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_m_l1784_178454

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m) / (x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x ≤ 1/3) ∨ (1/2 ≤ x ∧ x < 6)} := by sorry

-- Theorem for the range of m when A ∪ C = C
theorem range_of_m (m : ℝ) :
  (A ∪ C m = C m) → (-3 ≤ m ∧ m ≤ -1) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_m_l1784_178454


namespace NUMINAMATH_CALUDE_smallest_k_with_multiple_sequences_l1784_178429

/-- A sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧
  (∀ n, a (n + 1) ≥ a n) ∧
  (∀ n > 2, a n = a (n - 1) + a (n - 2))

/-- The existence of at least two distinct valid sequences with a₉ = k -/
def HasMultipleSequences (k : ℕ) : Prop :=
  ∃ a b : ℕ → ℕ, ValidSequence a ∧ ValidSequence b ∧ a ≠ b ∧ a 9 = k ∧ b 9 = k

/-- 748 is the smallest k for which multiple valid sequences exist -/
theorem smallest_k_with_multiple_sequences :
  HasMultipleSequences 748 ∧ ∀ k < 748, ¬HasMultipleSequences k :=
sorry

end NUMINAMATH_CALUDE_smallest_k_with_multiple_sequences_l1784_178429


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_eq_168_l1784_178448

/-- The number of positive three-digit integers less than 700 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let total_three_digit_integers := 700 - 100
  let integers_without_repeated_digits := 6 * 9 * 8
  total_three_digit_integers - integers_without_repeated_digits

theorem count_integers_with_repeated_digits_eq_168 :
  count_integers_with_repeated_digits = 168 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_eq_168_l1784_178448


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1784_178405

theorem right_rectangular_prism_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 30)
  (h2 : face_area2 = 45)
  (h3 : face_area3 = 75) :
  ∃ (x y z : ℝ), 
    x * y = face_area1 ∧ 
    x * z = face_area2 ∧ 
    y * z = face_area3 ∧ 
    x * y * z = 150 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l1784_178405


namespace NUMINAMATH_CALUDE_mike_baseball_cards_l1784_178490

theorem mike_baseball_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 64) 
  (h2 : new_cards = 18) : 
  initial_cards + new_cards = 82 := by
  sorry

end NUMINAMATH_CALUDE_mike_baseball_cards_l1784_178490


namespace NUMINAMATH_CALUDE_baseball_league_games_l1784_178433

/-- The number of teams in the baseball league -/
def num_teams : ℕ := 9

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

theorem baseball_league_games :
  total_games = 144 :=
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l1784_178433


namespace NUMINAMATH_CALUDE_gcd_difference_square_l1784_178408

theorem gcd_difference_square (x y z : ℕ+) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * (y.val - x.val) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_square_l1784_178408


namespace NUMINAMATH_CALUDE_divisors_equidistant_from_third_l1784_178417

theorem divisors_equidistant_from_third (n : ℕ) : 
  (∃ (a b : ℕ), a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
   (n : ℚ) / 3 - (a : ℚ) = (b : ℚ) - (n : ℚ) / 3) → 
  ∃ (k : ℕ), n = 6 * k :=
sorry

end NUMINAMATH_CALUDE_divisors_equidistant_from_third_l1784_178417


namespace NUMINAMATH_CALUDE_count_special_sequences_l1784_178479

def sequence_length : ℕ := 15

-- Define a function that counts sequences with all ones consecutive
def count_all_ones_consecutive (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2 - 1

-- Define a function that counts sequences with all zeros consecutive
def count_all_zeros_consecutive (n : ℕ) : ℕ :=
  count_all_ones_consecutive n

-- Define a function that counts sequences with both all zeros and all ones consecutive
def count_both_consecutive : ℕ := 2

-- Theorem statement
theorem count_special_sequences :
  count_all_ones_consecutive sequence_length +
  count_all_zeros_consecutive sequence_length -
  count_both_consecutive = 268 := by
  sorry

end NUMINAMATH_CALUDE_count_special_sequences_l1784_178479


namespace NUMINAMATH_CALUDE_inequality_proof_l1784_178438

theorem inequality_proof (m n : ℕ+) : 
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1784_178438


namespace NUMINAMATH_CALUDE_solve_for_y_l1784_178451

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1784_178451


namespace NUMINAMATH_CALUDE_candy_distribution_l1784_178428

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : friends = 6)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1784_178428


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1784_178418

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for A ∩ (ℝ \ B)
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1784_178418


namespace NUMINAMATH_CALUDE_angle_measure_l1784_178474

theorem angle_measure (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 20) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1784_178474


namespace NUMINAMATH_CALUDE_pottery_rim_diameter_l1784_178495

theorem pottery_rim_diameter 
  (chord_length : ℝ) 
  (segment_height : ℝ) 
  (h1 : chord_length = 16) 
  (h2 : segment_height = 2) : 
  ∃ (diameter : ℝ), diameter = 34 ∧ 
  (∃ (radius : ℝ), 
    radius * 2 = diameter ∧
    radius^2 = (radius - segment_height)^2 + (chord_length / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_pottery_rim_diameter_l1784_178495


namespace NUMINAMATH_CALUDE_equation_rearrangement_l1784_178463

theorem equation_rearrangement (x : ℝ) : (x - 5 = 3*x + 7) ↔ (x - 3*x = 7 + 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_rearrangement_l1784_178463


namespace NUMINAMATH_CALUDE_unique_triple_divisibility_l1784_178482

theorem unique_triple_divisibility (a b c : ℕ) : 
  (∃ k : ℕ, (a * b + 1) = k * (2 * c)) ∧
  (∃ m : ℕ, (b * c + 1) = m * (2 * a)) ∧
  (∃ n : ℕ, (c * a + 1) = n * (2 * b)) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_divisibility_l1784_178482


namespace NUMINAMATH_CALUDE_three_random_events_l1784_178496

/-- Represents an event that can occur in a probability space. -/
structure Event where
  description : String
  is_random : Bool

/-- The set of events we're considering. -/
def events : List Event := [
  ⟨"Selecting 3 out of 10 glass cups (8 good quality, 2 defective), all 3 selected are good quality", true⟩,
  ⟨"Randomly pressing a digit and it happens to be correct when forgetting the last digit of a phone number", true⟩,
  ⟨"Opposite electric charges attract each other", false⟩,
  ⟨"A person wins the first prize in a sports lottery", true⟩
]

/-- Counts the number of random events in a list of events. -/
def countRandomEvents (events : List Event) : Nat :=
  events.filter (·.is_random) |>.length

/-- The main theorem stating that exactly three of the given events are random. -/
theorem three_random_events : countRandomEvents events = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_random_events_l1784_178496


namespace NUMINAMATH_CALUDE_infinite_sum_equals_three_l1784_178430

open BigOperators

theorem infinite_sum_equals_three :
  ∑' k, (5^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_three_l1784_178430


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_153_l1784_178423

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the y-intercept of a line given its slope and a point it passes through -/
def calculateYIntercept (slope : ℝ) (p : Point) : ℝ :=
  p.y - slope * p.x

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Main theorem statement -/
theorem quadrilateral_area_is_153 (line1 : Line) (line2 : Line) (O E C : Point) : 
  line1.slope = -3 ∧ 
  E.x = 6 ∧ E.y = 6 ∧ 
  C.x = 10 ∧ C.y = 0 ∧ 
  O.x = 0 ∧ O.y = 0 ∧
  E.y = line1.slope * E.x + line1.intercept ∧
  E.y = line2.slope * E.x + line2.intercept ∧
  C.y = line2.slope * C.x + line2.intercept →
  let B : Point := { x := 0, y := calculateYIntercept line1.slope E }
  let areaOBE := triangleArea O B E
  let areaOEC := triangleArea O E C
  let areaEBC := triangleArea E B C
  areaOBE + areaOEC - areaEBC = 153 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_153_l1784_178423


namespace NUMINAMATH_CALUDE_approximate_number_properties_l1784_178436

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Determines if a number is accurate to a specific place value -/
def is_accurate_to (n : ScientificNotation) (place : Int) : Prop :=
  sorry

/-- Counts the number of significant figures in a number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- The hundreds place value -/
def hundreds : Int :=
  2

theorem approximate_number_properties (n : ScientificNotation) 
  (h1 : n.coefficient = 8.8)
  (h2 : n.exponent = 3) :
  is_accurate_to n hundreds ∧ count_significant_figures n = 2 := by
  sorry

end NUMINAMATH_CALUDE_approximate_number_properties_l1784_178436


namespace NUMINAMATH_CALUDE_sector_forms_cone_l1784_178461

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Given a circular sector, returns the cone formed by aligning its straight sides -/
def sectorToCone (sector : CircularSector) : Cone :=
  sorry

theorem sector_forms_cone :
  let sector : CircularSector := ⟨12, 270 * π / 180⟩
  let cone : Cone := sectorToCone sector
  cone.baseRadius = 9 ∧ cone.slantHeight = 12 := by
  sorry

end NUMINAMATH_CALUDE_sector_forms_cone_l1784_178461


namespace NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l1784_178453

theorem tan_alpha_sqrt_three (α : Real) (h : ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) : 
  Real.tan α = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l1784_178453


namespace NUMINAMATH_CALUDE_expression_simplification_l1784_178458

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1784_178458


namespace NUMINAMATH_CALUDE_simplify_fraction_l1784_178465

theorem simplify_fraction : (24 : ℚ) / 32 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1784_178465


namespace NUMINAMATH_CALUDE_cow_price_problem_l1784_178499

/-- Given the total cost of cows and goats, the number of cows and goats, and the average price of a goat,
    calculate the average price of a cow. -/
def average_cow_price (total_cost : ℕ) (num_cows num_goats : ℕ) (avg_goat_price : ℕ) : ℕ :=
  (total_cost - num_goats * avg_goat_price) / num_cows

/-- Theorem: Given 2 cows and 10 goats with a total cost of 1500 rupees, 
    and an average price of 70 rupees per goat, the average price of a cow is 400 rupees. -/
theorem cow_price_problem : average_cow_price 1500 2 10 70 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cow_price_problem_l1784_178499


namespace NUMINAMATH_CALUDE_unique_nested_sqrt_integer_l1784_178450

theorem unique_nested_sqrt_integer : ∃! (n : ℕ+), ∃ (x : ℤ), x^2 = n + Real.sqrt (n + Real.sqrt (n + Real.sqrt n)) := by
  sorry

end NUMINAMATH_CALUDE_unique_nested_sqrt_integer_l1784_178450


namespace NUMINAMATH_CALUDE_partition_16_into_8_pairs_eq_2027025_l1784_178444

/-- The number of ways to partition 16 distinct elements into 8 unordered pairs -/
def partition_16_into_8_pairs : ℕ :=
  (Nat.factorial 16) / (Nat.pow 2 8 * Nat.factorial 8)

/-- Theorem stating that the number of ways to partition 16 distinct elements
    into 8 unordered pairs is equal to 2027025 -/
theorem partition_16_into_8_pairs_eq_2027025 :
  partition_16_into_8_pairs = 2027025 := by
  sorry

end NUMINAMATH_CALUDE_partition_16_into_8_pairs_eq_2027025_l1784_178444


namespace NUMINAMATH_CALUDE_simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l1784_178445

-- 1. Simplify fraction with square root
theorem simplify_fraction : (2 : ℝ) / (Real.sqrt 3 - 1) = Real.sqrt 3 + 1 := by sorry

-- 2. Simplify harmonic quadratic root (case 1)
theorem simplify_harmonic_root1 : Real.sqrt (4 + 2 * Real.sqrt 3) = Real.sqrt 3 + 1 := by sorry

-- 3. Simplify harmonic quadratic root (case 2)
theorem simplify_harmonic_root2 : Real.sqrt (6 - 2 * Real.sqrt 5) = Real.sqrt 5 - 1 := by sorry

-- 4. Calculate expression with harmonic quadratic roots
theorem calculate_expression (m n : ℝ) 
  (hm : m = 1 / Real.sqrt (5 + 2 * Real.sqrt 6))
  (hn : n = 1 / Real.sqrt (5 - 2 * Real.sqrt 6)) :
  (m - n) / (m + n) = -(Real.sqrt 6) / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l1784_178445


namespace NUMINAMATH_CALUDE_flag_designs_count_l1784_178486

/-- The number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27 -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l1784_178486


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l1784_178410

def candidate1_votes : ℕ := 6136
def candidate2_votes : ℕ := 7636
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |winning_percentage - 45.78| < ε :=
sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l1784_178410


namespace NUMINAMATH_CALUDE_correct_propositions_l1784_178434

theorem correct_propositions (a b c d : ℝ) : 
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧ 
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a - c > b - d)) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∀ (a b c : ℝ), a > b → c > 0 → a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l1784_178434


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l1784_178415

-- Define the total number of ladybugs
def total_ladybugs : ℕ := 67082

-- Define the number of ladybugs with spots
def ladybugs_with_spots : ℕ := 12170

-- Theorem to prove
theorem ladybugs_without_spots : 
  total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l1784_178415


namespace NUMINAMATH_CALUDE_amy_spending_at_fair_l1784_178404

/-- Amy's spending at the fair --/
theorem amy_spending_at_fair (initial_amount final_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : final_amount = 11) :
  initial_amount - final_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_spending_at_fair_l1784_178404


namespace NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l1784_178485

theorem cubic_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l1784_178485


namespace NUMINAMATH_CALUDE_max_value_of_h_l1784_178473

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the function h as the sum of f and g
def h (x : ℝ) : ℝ := f x + g x

-- State the theorem
theorem max_value_of_h :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, h x = 6) ∧ (∀ x, h x ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_h_l1784_178473


namespace NUMINAMATH_CALUDE_number_ratio_problem_l1784_178441

theorem number_ratio_problem (x : ℚ) : 
  (x / 6 = 16 / 480) → x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l1784_178441


namespace NUMINAMATH_CALUDE_product_comparison_l1784_178466

theorem product_comparison (a b c d : ℝ) (h1 : a ≥ b) (h2 : c ≥ d) :
  (∃ (p : ℕ), p ≥ 3 ∧ (a > 0 ∨ b > 0) ∧ (a > 0 ∨ c > 0) ∧ (a > 0 ∨ d > 0) ∧
               (b > 0 ∨ c > 0) ∧ (b > 0 ∨ d > 0) ∧ (c > 0 ∨ d > 0)) →
    a * c ≥ b * d ∧
  (∃ (n : ℕ), n ≥ 3 ∧ (a < 0 ∨ b < 0) ∧ (a < 0 ∨ c < 0) ∧ (a < 0 ∨ d < 0) ∧
               (b < 0 ∨ c < 0) ∧ (b < 0 ∨ d < 0) ∧ (c < 0 ∨ d < 0)) →
    a * c ≤ b * d ∧
  (((a > 0 ∧ b > 0) ∨ (a > 0 ∧ c > 0) ∨ (a > 0 ∧ d > 0) ∨ (b > 0 ∧ c > 0) ∨
    (b > 0 ∧ d > 0) ∨ (c > 0 ∧ d > 0)) ∧
   ((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (a < 0 ∧ d < 0) ∨ (b < 0 ∧ c < 0) ∨
    (b < 0 ∧ d < 0) ∨ (c < 0 ∧ d < 0))) →
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x = y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x < y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x > y) :=
by sorry

end NUMINAMATH_CALUDE_product_comparison_l1784_178466


namespace NUMINAMATH_CALUDE_unique_prime_sum_l1784_178497

/-- Given seven distinct positive integers not exceeding 7, prove that 179 is the only prime expressible as abcd + efg -/
theorem unique_prime_sum (a b c d e f g : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  0 < a ∧ a ≤ 7 ∧
  0 < b ∧ b ≤ 7 ∧
  0 < c ∧ c ≤ 7 ∧
  0 < d ∧ d ≤ 7 ∧
  0 < e ∧ e ≤ 7 ∧
  0 < f ∧ f ≤ 7 ∧
  0 < g ∧ g ≤ 7 →
  (∃ p : ℕ, Nat.Prime p ∧ p = a * b * c * d + e * f * g) ↔ (a * b * c * d + e * f * g = 179) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_l1784_178497


namespace NUMINAMATH_CALUDE_license_plate_count_l1784_178403

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 4

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : total_license_plates = 878800000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1784_178403


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1784_178447

theorem polynomial_simplification (x : ℝ) :
  (12 * x^10 + 5 * x^9 + 3 * x^8) + (2 * x^12 + 9 * x^10 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 10) =
  2 * x^12 + 21 * x^10 + 5 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 10 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1784_178447


namespace NUMINAMATH_CALUDE_one_bedroom_apartment_fraction_l1784_178467

theorem one_bedroom_apartment_fraction :
  let two_bedroom_fraction : ℝ := 0.33
  let total_fraction : ℝ := 0.5
  let one_bedroom_fraction : ℝ := total_fraction - two_bedroom_fraction
  one_bedroom_fraction = 0.17 := by
sorry

end NUMINAMATH_CALUDE_one_bedroom_apartment_fraction_l1784_178467


namespace NUMINAMATH_CALUDE_solution_difference_squared_l1784_178487

theorem solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_solution_difference_squared_l1784_178487


namespace NUMINAMATH_CALUDE_sector_area_l1784_178459

/-- Given a sector with central angle θ and arc length L, 
    the area A of the sector can be calculated. -/
theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = 2 → L = 4 → A = 4 → A = (1/2) * (L/θ)^2 * θ :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1784_178459


namespace NUMINAMATH_CALUDE_phase_shift_of_sine_function_l1784_178493

/-- The phase shift of the function y = 2 sin(2x + π/3) is -π/6 -/
theorem phase_shift_of_sine_function :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x + π / 3)
  ∃ (A B C D : ℝ), A ≠ 0 ∧ B ≠ 0 ∧
    (∀ x, f x = A * Real.sin (B * (x - C)) + D) ∧
    C = -π / 6 :=
by sorry

end NUMINAMATH_CALUDE_phase_shift_of_sine_function_l1784_178493


namespace NUMINAMATH_CALUDE_investment_time_calculation_l1784_178411

/-- Represents the investment scenario of two partners A and B --/
structure Investment where
  a_capital : ℝ
  a_time : ℝ
  b_capital : ℝ
  b_time : ℝ
  profit_ratio : ℝ

/-- Theorem stating the time B's investment was effective --/
theorem investment_time_calculation (i : Investment) 
  (h1 : i.a_capital = 27000)
  (h2 : i.b_capital = 36000)
  (h3 : i.a_time = 12)
  (h4 : i.profit_ratio = 2/1) :
  i.b_time = 4.5 := by
  sorry

#check investment_time_calculation

end NUMINAMATH_CALUDE_investment_time_calculation_l1784_178411


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1784_178498

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  (a 1 + a 1 * q + a 1 * q^2 + a 1 * q^3 = 10 * (a 1 + a 1 * q)) →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1784_178498


namespace NUMINAMATH_CALUDE_company_workers_l1784_178402

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = total / 3 →  -- One-third of workers don't have a retirement plan
  (1 / 5 : ℚ) * (total / 3 : ℚ) = total / 15 →  -- 20% of workers without a retirement plan are women
  (2 / 5 : ℚ) * ((2 * total) / 3 : ℚ) = (4 * total) / 15 →  -- 40% of workers with a retirement plan are men
  men = 144 →  -- There are 144 men
  total - men = 126  -- The number of women workers is 126
  := by sorry

end NUMINAMATH_CALUDE_company_workers_l1784_178402


namespace NUMINAMATH_CALUDE_equal_distribution_of_treats_l1784_178427

theorem equal_distribution_of_treats (cookies cupcakes brownies students : ℕ) 
  (h1 : cookies = 20)
  (h2 : cupcakes = 25)
  (h3 : brownies = 35)
  (h4 : students = 20) :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_treats_l1784_178427


namespace NUMINAMATH_CALUDE_min_dot_product_ep_qp_l1784_178492

/-- The ellipse defined by x^2/36 + y^2/9 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

/-- The fixed point E -/
def E : ℝ × ℝ := (3, 0)

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The minimum value of EP · QP is 6 -/
theorem min_dot_product_ep_qp :
  ∃ (min : ℝ),
    (∀ (P Q : ℝ × ℝ),
      is_on_ellipse P.1 P.2 →
      is_on_ellipse Q.1 Q.2 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) = 0 →
      dot_product (P.1 - E.1, P.2 - E.2) (Q.1 - P.1, Q.2 - P.2) ≥ min) ∧
    min = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_ep_qp_l1784_178492


namespace NUMINAMATH_CALUDE_restaurant_menu_theorem_l1784_178422

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem restaurant_menu_theorem (v : ℕ) : 
  (choose 5 2 * choose v 2 > 200) → v ≥ 7 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_theorem_l1784_178422


namespace NUMINAMATH_CALUDE_season_games_l1784_178475

/-- The number of teams in the league -/
def num_teams : ℕ := 20

/-- The number of times each team faces another team -/
def games_per_matchup : ℕ := 10

/-- Calculate the number of unique matchups in the league -/
def unique_matchups (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of games in the season -/
def total_games (n : ℕ) (g : ℕ) : ℕ := unique_matchups n * g

theorem season_games :
  total_games num_teams games_per_matchup = 1900 := by sorry

end NUMINAMATH_CALUDE_season_games_l1784_178475


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_l1784_178455

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line :
  ∀ x y : ℝ, 
  (y = f x) ∧ 
  (f' x = 4) → 
  ((x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_l1784_178455


namespace NUMINAMATH_CALUDE_min_sides_for_80_intersections_l1784_178437

/-- The number of intersection points between two n-sided polygons -/
def intersection_points (n : ℕ) : ℕ := 80

/-- Proposition: The minimum value of n for which two n-sided polygons can have exactly 80 intersection points is 10 -/
theorem min_sides_for_80_intersections :
  ∀ n : ℕ, intersection_points n = 80 → n ≥ 10 ∧ 
  ∃ (m : ℕ), m = 10 ∧ intersection_points m = 80 :=
sorry

end NUMINAMATH_CALUDE_min_sides_for_80_intersections_l1784_178437


namespace NUMINAMATH_CALUDE_transformed_roots_l1784_178494

-- Define the polynomial and its roots
def P (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x^2 - 6

-- Define the roots of P
def roots (b : ℝ) : Set ℝ := {x | P b x = 0}

-- Define the transformed equation
def Q (b : ℝ) (y : ℝ) : ℝ := 6*y^2 + b*y + 1

-- Theorem statement
theorem transformed_roots (b : ℝ) (a c d : ℝ) (ha : a ∈ roots b) (hc : c ∈ roots b) (hd : d ∈ roots b) :
  Q b ((a + c) / b^3) = 0 ∧ Q b ((a + b) / c^3) = 0 ∧ Q b ((b + c) / a^3) = 0 ∧ Q b ((a + b + c) / d^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_transformed_roots_l1784_178494


namespace NUMINAMATH_CALUDE_rectangle_length_l1784_178416

theorem rectangle_length (width perimeter : ℝ) (h1 : width = 15) (h2 : perimeter = 70) :
  let length := (perimeter - 2 * width) / 2
  length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1784_178416


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1784_178484

theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) :
  current_speed = 8 →
  downstream_distance = 6.283333333333333 →
  downstream_time = 13 / 60 →
  ∃ (boat_speed : ℝ), 
    boat_speed = 21 ∧ 
    (boat_speed + current_speed) * downstream_time = downstream_distance :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1784_178484


namespace NUMINAMATH_CALUDE_x_value_in_sequence_l1784_178431

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem x_value_in_sequence (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 3 = 10 →
  a 4 = 5 →
  a 5 = 15 →
  a 6 = 20 →
  a 7 = 35 →
  a 8 = 55 →
  a 9 = 90 →
  a 0 = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_in_sequence_l1784_178431


namespace NUMINAMATH_CALUDE_fraction_addition_l1784_178420

theorem fraction_addition (d : ℝ) : (6 + 5*d) / 9 + 3 = (33 + 5*d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1784_178420


namespace NUMINAMATH_CALUDE_workshop_workers_l1784_178424

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 22

/-- The number of technicians in the workshop -/
def technicians : ℕ := 7

/-- The average salary of all workers -/
def avg_salary_all : ℚ := 850

/-- The average salary of technicians -/
def avg_salary_tech : ℚ := 1000

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 780

/-- Theorem stating that given the conditions, the total number of workers is 22 -/
theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) =
  (avg_salary_tech * technicians : ℚ) +
  (avg_salary_rest * (total_workers - technicians) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_workshop_workers_l1784_178424


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1784_178439

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (2*a + Complex.I) * (1 - 2*Complex.I) = b * Complex.I ∧ b ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1784_178439


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1784_178426

/-- Given a rhombus with area 200 square units and diagonal ratio 4:3, 
    prove that the length of the longest diagonal is 40√3/3 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℚ) (d1 d2 : ℝ) :
  area = 200 →
  ratio = 4 / 3 →
  d1 / d2 = ratio →
  area = (d1 * d2) / 2 →
  d1 > d2 →
  d1 = 40 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l1784_178426


namespace NUMINAMATH_CALUDE_adam_laundry_l1784_178471

/-- Given a total number of loads and a number of washed loads, calculate the remaining loads to wash. -/
def remaining_loads (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem stating that given 14 total loads and 8 washed loads, the remaining loads is 6. -/
theorem adam_laundry : remaining_loads 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_adam_laundry_l1784_178471


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l1784_178480

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (s : Scores) : Prop :=
  s.math + s.physics = 30 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 25

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) :
  satisfies_conditions s → s.chemistry - s.physics = 20 := by
  sorry


end NUMINAMATH_CALUDE_chemistry_physics_difference_l1784_178480


namespace NUMINAMATH_CALUDE_function_value_at_symmetric_point_l1784_178457

/-- Given a function f(x) = a * sin³(x) + b * tan(x) + 1 where f(2) = 3,
    prove that f(2π - 2) = -1 -/
theorem function_value_at_symmetric_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * (Real.sin x)^3 + b * Real.tan x + 1)
  (h2 : f 2 = 3) :
  f (2 * Real.pi - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_symmetric_point_l1784_178457


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l1784_178462

theorem quadratic_equations_common_root (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 12 = 0 ∧ 3*x^2 - 8*x - 3*k = 0) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l1784_178462


namespace NUMINAMATH_CALUDE_sqrt_175_range_l1784_178409

theorem sqrt_175_range : 13 < Real.sqrt 175 ∧ Real.sqrt 175 < 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_175_range_l1784_178409


namespace NUMINAMATH_CALUDE_sarah_age_l1784_178488

/-- Given the ages of Billy, Joe, and Sarah, prove that Sarah is 10 years old -/
theorem sarah_age (B J S : ℕ) 
  (h1 : B = 2 * J)           -- Billy's age is twice Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60 years
  (h3 : S = J - 10)          -- Sarah's age is 10 years less than Joe's age
  : S = 10 := by             -- Prove that Sarah is 10 years old
  sorry

end NUMINAMATH_CALUDE_sarah_age_l1784_178488


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l1784_178476

theorem walnut_trees_before_planting (trees_to_plant : ℕ) (final_trees : ℕ) 
  (h1 : trees_to_plant = 104)
  (h2 : final_trees = 211) :
  final_trees - trees_to_plant = 107 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l1784_178476


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1784_178456

-- Define the quadratic polynomials
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution sets
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_poly a b c x > 0}

-- Define the condition for equal ratios
def equal_ratios (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

-- State the theorem
theorem not_necessary_not_sufficient
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ¬(equal_ratios a₁ b₁ c₁ a₂ b₂ c₂ ↔ solution_set a₁ b₁ c₁ = solution_set a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l1784_178456


namespace NUMINAMATH_CALUDE_frustum_smaller_radius_l1784_178464

/-- A circular frustum with the given properties -/
structure CircularFrustum where
  r : ℝ  -- radius of the smaller base
  slant_height : ℝ
  lateral_area : ℝ

/-- The theorem statement -/
theorem frustum_smaller_radius (f : CircularFrustum) 
  (h1 : f.slant_height = 3)
  (h2 : f.lateral_area = 84 * Real.pi)
  (h3 : 2 * Real.pi * (3 * f.r) = 3 * (2 * Real.pi * f.r)) :
  f.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_radius_l1784_178464


namespace NUMINAMATH_CALUDE_inequality_proof_l1784_178483

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1784_178483


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l1784_178481

theorem max_value_abc_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l1784_178481


namespace NUMINAMATH_CALUDE_turkey_cost_l1784_178477

/-- The cost of turkeys given their weights and price per kilogram -/
theorem turkey_cost (w1 w2 w3 w4 : ℝ) (price_per_kg : ℝ) : 
  w1 = 6 →
  w2 = 9 →
  w3 = 2 * w2 →
  w4 = (w1 + w2 + w3) / 2 →
  price_per_kg = 2 →
  (w1 + w2 + w3 + w4) * price_per_kg = 99 :=
by
  sorry

#check turkey_cost

end NUMINAMATH_CALUDE_turkey_cost_l1784_178477


namespace NUMINAMATH_CALUDE_g_in_M_l1784_178406

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂ : ℝ, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end NUMINAMATH_CALUDE_g_in_M_l1784_178406


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1784_178400

/-- Calculate the distance traveled with constant acceleration -/
def distance_traveled (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_velocity * time + 0.5 * acceleration * time^2

/-- Convert kilometers to meters -/
def km_to_meters (km : ℝ) : ℝ := km * 1000

theorem bridge_length_proof (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) 
  (h1 : initial_velocity = 3)
  (h2 : acceleration = 0.2)
  (h3 : time = 0.25) :
  km_to_meters (distance_traveled initial_velocity acceleration time) = 756.25 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1784_178400


namespace NUMINAMATH_CALUDE_factorial_units_digit_zero_sum_factorials_units_digit_l1784_178452

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorialsUnitsDigit (n : ℕ) : ℕ :=
  unitsDigit ((List.range n).map factorial).sum

theorem factorial_units_digit_zero (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 := by sorry

theorem sum_factorials_units_digit :
  sumFactorialsUnitsDigit 2010 = 3 := by sorry

end NUMINAMATH_CALUDE_factorial_units_digit_zero_sum_factorials_units_digit_l1784_178452


namespace NUMINAMATH_CALUDE_relationship_between_m_and_a_l1784_178472

theorem relationship_between_m_and_a (m : ℕ) (a : ℝ) 
  (h1 : m > 0) (h2 : a > 0) :
  ((∀ n : ℕ, n > m → (1 : ℝ) / n < a) ∧ 
   (∀ n : ℕ, 0 < n ∧ n ≤ m → (1 : ℝ) / n ≥ a)) ↔ 
  m = ⌊(1 : ℝ) / a⌋ := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_m_and_a_l1784_178472


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1784_178432

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), m < n → ¬(15 ∣ (427398 - m))) ∧ 
  (15 ∣ (427398 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1784_178432


namespace NUMINAMATH_CALUDE_sons_present_age_l1784_178478

/-- Proves that given the conditions about a father and son's ages, the son's present age is 22 years -/
theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

#check sons_present_age

end NUMINAMATH_CALUDE_sons_present_age_l1784_178478


namespace NUMINAMATH_CALUDE_constant_expression_inequality_solution_l1784_178469

-- Part 1: Prove that the expression simplifies to -9 for all real x
theorem constant_expression (x : ℝ) : x * (x - 6) - (3 - x)^2 = -9 := by
  sorry

-- Part 2: Prove that the solution to the inequality is x < 5
theorem inequality_solution : 
  {x : ℝ | x - 2*(x - 3) > 1} = {x : ℝ | x < 5} := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_inequality_solution_l1784_178469


namespace NUMINAMATH_CALUDE_popping_corn_probability_l1784_178468

theorem popping_corn_probability (white yellow blue : ℝ)
  (white_pop yellow_pop blue_pop : ℝ) :
  white = 1/2 →
  yellow = 1/3 →
  blue = 1/6 →
  white_pop = 3/4 →
  yellow_pop = 1/2 →
  blue_pop = 1/3 →
  (white * white_pop) / (white * white_pop + yellow * yellow_pop + blue * blue_pop) = 27/43 := by
  sorry

end NUMINAMATH_CALUDE_popping_corn_probability_l1784_178468


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1784_178425

theorem angle_in_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  |Real.cos (α / 3)| = -Real.cos (α / 3) →  -- |cos(α/3)| = -cos(α/3)
  (π / 2 < α / 3) ∧ (α / 3 < π)  -- α/3 is in the second quadrant
:= by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1784_178425


namespace NUMINAMATH_CALUDE_train_overtake_time_l1784_178440

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed : ℝ) (motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 180.0144 →
  (train_length / ((train_speed - motorbike_speed) / 3.6)) = 18.00144 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l1784_178440


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_l1784_178443

/-- A line with equation y = kx + 2, where k is a real number. -/
structure Line where
  k : ℝ

/-- An ellipse with equation x² + y²/m = 1, where m is a positive real number. -/
structure Ellipse where
  m : ℝ
  h_pos : 0 < m

/-- 
Given a line y = kx + 2 and an ellipse x² + y²/m = 1,
if the line always intersects the ellipse for all real k,
then m is greater than or equal to 4.
-/
theorem line_always_intersects_ellipse (e : Ellipse) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 / e.m = 1) →
  4 ≤ e.m :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_l1784_178443


namespace NUMINAMATH_CALUDE_volumetric_contraction_of_mixed_liquids_l1784_178412

/-- Proves that the volumetric contraction when mixing two liquids with given properties is 21 cm³ -/
theorem volumetric_contraction_of_mixed_liquids :
  let density1 : ℝ := 1.7
  let mass1 : ℝ := 400
  let density2 : ℝ := 1.2
  let mass2 : ℝ := 600
  let total_mass : ℝ := mass1 + mass2
  let mixed_density : ℝ := 1.4
  let volume1 : ℝ := mass1 / density1
  let volume2 : ℝ := mass2 / density2
  let total_volume : ℝ := volume1 + volume2
  let actual_volume : ℝ := total_mass / mixed_density
  let contraction : ℝ := total_volume - actual_volume
  contraction = 21 := by sorry

end NUMINAMATH_CALUDE_volumetric_contraction_of_mixed_liquids_l1784_178412


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_fifteen_l1784_178401

theorem last_digit_of_one_over_two_to_fifteen (n : ℕ) :
  n = 15 →
  (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_fifteen_l1784_178401


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1784_178460

def vector1 : Fin 2 → ℝ := ![5, -3]
def vector2 : Fin 2 → ℝ := ![-4, 6]
def vector3 : Fin 2 → ℝ := ![2, -8]

theorem vector_sum_proof :
  vector1 + vector2 + vector3 = ![3, -5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1784_178460


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1784_178446

-- Define the set of cards
def cards : Finset ℕ := {3, 4, 5, 6, 7}

-- Define the sample space (all possible pairs of cards)
def sample_space : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (fun p => p.1 < p.2)

-- Define event A: sum of selected cards is even
def event_A : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => (p.1 + p.2) % 2 = 0)

-- Define event B: both selected cards are odd
def event_B : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.1 % 2 = 1 ∧ p.2 % 2 = 1)

-- Define the probability measure
def P (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / sample_space.card

-- Theorem to prove
theorem conditional_probability_B_given_A :
  P (event_A ∩ event_B) / P event_A = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l1784_178446


namespace NUMINAMATH_CALUDE_unique_multiple_of_6_l1784_178489

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_multiple_of_6 :
  ∀ n : ℕ, 63470 ≤ n ∧ n ≤ 63479 →
    (is_multiple_of_6 n ↔ n = 63474) :=
by sorry

end NUMINAMATH_CALUDE_unique_multiple_of_6_l1784_178489


namespace NUMINAMATH_CALUDE_expected_sides_is_four_l1784_178413

/-- The number of cuts made in one hour -/
def k : ℕ := 3600

/-- The initial number of sides of the rectangular sheet -/
def initial_sides : ℕ := 4

/-- The total number of sides after k cuts -/
def total_sides (k : ℕ) : ℕ := initial_sides + 4 * k

/-- The total number of polygons after k cuts -/
def total_polygons (k : ℕ) : ℕ := k + 1

/-- The expected number of sides of a randomly picked polygon after k cuts -/
def expected_sides (k : ℕ) : ℚ :=
  (total_sides k : ℚ) / (total_polygons k : ℚ)

theorem expected_sides_is_four :
  expected_sides k = 4 := by sorry

end NUMINAMATH_CALUDE_expected_sides_is_four_l1784_178413


namespace NUMINAMATH_CALUDE_unique_solution_l1784_178491

-- Define the circles
variable (A B C D E F : ℕ)

-- Define the conditions
def valid_arrangement (A B C D E F : ℕ) : Prop :=
  -- All numbers are between 1 and 6
  (A ∈ Finset.range 6) ∧ (B ∈ Finset.range 6) ∧ (C ∈ Finset.range 6) ∧
  (D ∈ Finset.range 6) ∧ (E ∈ Finset.range 6) ∧ (F ∈ Finset.range 6) ∧
  -- All numbers are distinct
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  -- Sums on each line are equal
  A + C + D = A + B ∧
  A + C + D = B + D + F ∧
  A + C + D = E + F ∧
  A + C + D = E + B + C

-- Theorem statement
theorem unique_solution :
  ∀ A B C D E F : ℕ, valid_arrangement A B C D E F → A = 6 ∧ B = 3 :=
sorry


end NUMINAMATH_CALUDE_unique_solution_l1784_178491


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1784_178435

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1784_178435


namespace NUMINAMATH_CALUDE_min_value_theorem_l1784_178449

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n = -1) → 
  (∀ k l, k > 0 → l > 0 → k / m + l / n = -1 → 3*m + n ≤ 3*k + l) →
  3*m + n ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1784_178449


namespace NUMINAMATH_CALUDE_ellipse_range_and_logical_conditions_l1784_178407

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (m + 1) + y^2 / (3 - m) = 1) → 
  (∃ a b : ℝ, a > b ∧ a^2 - b^2 = 3 - m - (m + 1) ∧ 
  ∀ t : ℝ, x^2 / (m + 1) + y^2 / (3 - m) = 1 → 
  (x = 0 → y^2 ≤ a^2) ∧ (y = 0 → x^2 ≤ b^2))

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

theorem ellipse_range_and_logical_conditions (m : ℝ) :
  (p m ↔ -1 < m ∧ m < 1) ∧
  ((¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ 1 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_and_logical_conditions_l1784_178407


namespace NUMINAMATH_CALUDE_apple_count_theorem_l1784_178419

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l1784_178419


namespace NUMINAMATH_CALUDE_sets_theorem_l1784_178421

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Define the theorem
theorem sets_theorem :
  -- Part 1
  (A (1/2) ∩ (Set.univ \ B (1/2)) = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  -- Part 2
  (∀ a : ℝ, Set.Subset (A a) (B a) ↔ -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l1784_178421


namespace NUMINAMATH_CALUDE_sqrt_comparison_l1784_178470

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 6 < Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l1784_178470


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_15_l1784_178414

theorem binomial_coefficient_21_15 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 14 = 38760) →
  (Nat.choose 22 15 = 170544) →
  (Nat.choose 21 15 = 54264) :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_15_l1784_178414


namespace NUMINAMATH_CALUDE_amp_calculation_l1784_178442

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_calculation : amp (amp 10 4) 2 = 7052 := by
  sorry

end NUMINAMATH_CALUDE_amp_calculation_l1784_178442
