import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1653_165365

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = -121 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1653_165365


namespace NUMINAMATH_CALUDE_solve_for_a_l1653_165348

theorem solve_for_a (y : ℝ) (h1 : y > 0) (h2 : (a * y) / 20 + (3 * y) / 10 = 0.7 * y) : a = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1653_165348


namespace NUMINAMATH_CALUDE_count_three_painted_faces_4x4x4_l1653_165361

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (painted_faces : Fin 6 → Bool)

/-- Counts the number of subcubes with at least three painted faces -/
def count_subcubes_with_three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Theorem: In a 4x4x4 cube with all outer faces painted, 
    the number of 1x1x1 subcubes with at least three painted faces is 8 -/
theorem count_three_painted_faces_4x4x4 : 
  ∀ (cube : PaintedCube), 
  cube.size = 4 → 
  (∀ (f : Fin 6), cube.painted_faces f = true) →
  count_subcubes_with_three_painted_faces cube = 8 := by sorry

end NUMINAMATH_CALUDE_count_three_painted_faces_4x4x4_l1653_165361


namespace NUMINAMATH_CALUDE_chessboard_determinability_l1653_165313

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat

/-- Represents the chessboard and the game state -/
structure Chessboard (n : Nat) where
  selected : Set Square
  adjacent_counts : Square → Nat

/-- Defines when a number is "beautiful" (remainder 1 when divided by 3) -/
def is_beautiful (k : Nat) : Bool :=
  k % 3 = 1

/-- Defines when a square is beautiful -/
def beautiful_square (s : Square) : Bool :=
  is_beautiful s.row ∧ is_beautiful s.col

/-- Defines when Bianka can uniquely determine Aranka's selection -/
def can_determine (n : Nat) (board : Chessboard n) : Prop :=
  ∀ (alt_board : Chessboard n),
    (∀ (s : Square), board.adjacent_counts s = alt_board.adjacent_counts s) →
    board.selected = alt_board.selected

/-- The main theorem to be proved -/
theorem chessboard_determinability (n : Nat) :
  (∃ (k : Nat), n = 3 * k + 1) → (∀ (board : Chessboard n), can_determine n board) ∧
  (∃ (k : Nat), n = 3 * k + 2) → (∃ (board : Chessboard n), ¬can_determine n board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_determinability_l1653_165313


namespace NUMINAMATH_CALUDE_no_covering_compact_rationals_l1653_165327

theorem no_covering_compact_rationals :
  ¬ (∃ (A : ℕ → Set ℝ),
    (∀ n, IsCompact (A n)) ∧
    (∀ n, A n ⊆ Set.range (Rat.cast : ℚ → ℝ)) ∧
    (∀ K : Set ℝ, IsCompact K → K ⊆ Set.range (Rat.cast : ℚ → ℝ) →
      ∃ m, K ⊆ A m)) :=
by sorry

end NUMINAMATH_CALUDE_no_covering_compact_rationals_l1653_165327


namespace NUMINAMATH_CALUDE_inequality_solution_l1653_165387

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ 
  (x ∈ Set.Ioc (-4) (-2) ∪ Set.Ioc (-2) (Real.sqrt 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1653_165387


namespace NUMINAMATH_CALUDE_riding_ratio_is_half_l1653_165389

/-- Represents the number of horses and men -/
def total_count : ℕ := 14

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 70

/-- Represents the number of legs a horse has -/
def horse_legs : ℕ := 4

/-- Represents the number of legs a man has -/
def man_legs : ℕ := 2

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := (total_count * horse_legs - legs_on_ground) / (horse_legs - man_legs)

/-- Represents the ratio of riding owners to total owners -/
def riding_ratio : ℚ := riding_owners / total_count

theorem riding_ratio_is_half : riding_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_riding_ratio_is_half_l1653_165389


namespace NUMINAMATH_CALUDE_females_together_count_females_apart_count_l1653_165369

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements where female students must stand together -/
def arrangements_females_together : ℕ :=
  (Nat.factorial num_female) * (Nat.factorial (num_male + 1))

/-- Calculates the number of arrangements where no two female students can stand next to each other -/
def arrangements_females_apart : ℕ :=
  (Nat.factorial num_male) * (Nat.choose (num_male + 1) num_female)

/-- Theorem stating the number of arrangements where female students must stand together -/
theorem females_together_count : arrangements_females_together = 720 := by
  sorry

/-- Theorem stating the number of arrangements where no two female students can stand next to each other -/
theorem females_apart_count : arrangements_females_apart = 1440 := by
  sorry

end NUMINAMATH_CALUDE_females_together_count_females_apart_count_l1653_165369


namespace NUMINAMATH_CALUDE_correct_product_l1653_165340

theorem correct_product (a b c : ℚ) : 
  a = 0.125 → b = 3.2 → c = 4.0 → 
  (125 : ℚ) * 320 = 40000 → a * b = c := by
sorry

end NUMINAMATH_CALUDE_correct_product_l1653_165340


namespace NUMINAMATH_CALUDE_seashell_count_l1653_165300

theorem seashell_count (joan_shells jessica_shells : ℕ) 
  (h1 : joan_shells = 6) 
  (h2 : jessica_shells = 8) : 
  joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l1653_165300


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1653_165316

theorem hyperbola_vertices_distance (x y : ℝ) :
  x^2 / 144 - y^2 / 64 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (64 : ℝ) = 1 ∧ 2 * a = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1653_165316


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1653_165383

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the original point A -/
def A (m : ℝ) : Point :=
  { x := -5 * m, y := 2 * m - 1 }

/-- Moves a point up by a given amount -/
def moveUp (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y + amount }

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Main theorem -/
theorem point_in_fourth_quadrant (m : ℝ) :
  (moveUp (A m) 3).y = 0 → isInFourthQuadrant (A m) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1653_165383


namespace NUMINAMATH_CALUDE_simplify_fraction_l1653_165395

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1653_165395


namespace NUMINAMATH_CALUDE_average_tree_height_l1653_165301

def tree_heights : List ℝ := [1000, 500, 500, 1200]

theorem average_tree_height : (tree_heights.sum / tree_heights.length : ℝ) = 800 := by
  sorry

end NUMINAMATH_CALUDE_average_tree_height_l1653_165301


namespace NUMINAMATH_CALUDE_triangles_5_4_l1653_165353

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of triangles formed by points on two parallel lines -/
def triangles_on_parallel_lines (points_on_line_a points_on_line_b : ℕ) : ℕ :=
  choose points_on_line_a 2 * choose points_on_line_b 1 +
  choose points_on_line_a 1 * choose points_on_line_b 2

/-- Theorem: The number of triangles formed by 5 points on one line and 4 points on a parallel line -/
theorem triangles_5_4 : triangles_on_parallel_lines 5 4 = choose 5 2 * choose 4 1 + choose 5 1 * choose 4 2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_5_4_l1653_165353


namespace NUMINAMATH_CALUDE_minimum_value_and_range_l1653_165338

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem minimum_value_and_range (a : ℝ) :
  (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) →
  ((a = 1 ∨ a = 7) ∧
   (a = 1 → ∀ x, f x a ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5) ∧
   (a = 7 → ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_l1653_165338


namespace NUMINAMATH_CALUDE_oliver_money_result_l1653_165351

/-- Calculates the remaining money after Oliver's transactions -/
def oliver_money (initial : ℝ) (feb_spend_percent : ℝ) (march_add : ℝ) (final_spend_percent : ℝ) : ℝ :=
  let after_feb := initial * (1 - feb_spend_percent)
  let after_march := after_feb + march_add
  after_march * (1 - final_spend_percent)

/-- Theorem stating that Oliver's remaining money is $54.04 -/
theorem oliver_money_result :
  oliver_money 33 0.15 32 0.10 = 54.04 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_result_l1653_165351


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1653_165391

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- State the theorem
theorem unique_quadratic_function :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x^2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1653_165391


namespace NUMINAMATH_CALUDE_complex_plane_properties_l1653_165342

/-- Given complex numbers and their corresponding points in the complex plane, prove various geometric properties. -/
theorem complex_plane_properties (z₁ z₂ z₃ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) 
  (h₂ : z₂ = 1 + 7*I) 
  (h₃ : z₃ = 3 - 4*I) : 
  (z₂.re > 0 ∧ z₂.im > 0) ∧ 
  (z₁ = -z₃) ∧ 
  (z₁.re * z₂.re + z₁.im * z₂.im > 0) ∧
  (z₁.re * (z₂.re + z₃.re) + z₁.im * (z₂.im + z₃.im) = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_plane_properties_l1653_165342


namespace NUMINAMATH_CALUDE_eight_digit_factorization_comparison_l1653_165386

theorem eight_digit_factorization_comparison :
  let total_eight_digit_numbers := 99999999 - 10000000 + 1
  let four_digit_numbers := 9999 - 1000 + 1
  let products_of_four_digit_numbers := four_digit_numbers.choose 2 + four_digit_numbers
  total_eight_digit_numbers - products_of_four_digit_numbers > products_of_four_digit_numbers := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_factorization_comparison_l1653_165386


namespace NUMINAMATH_CALUDE_M_intersect_N_l1653_165382

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1653_165382


namespace NUMINAMATH_CALUDE_circle_properties_l1653_165399

/-- Given a circle with circumference 36 cm, prove its diameter and area. -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  let d := 2 * r
  let A := Real.pi * r^2
  d = 36 / Real.pi ∧ A = 324 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l1653_165399


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l1653_165368

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: In a block with 20 houses and 640 pieces of junk mail, each house receives 32 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 20 640 = 32 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l1653_165368


namespace NUMINAMATH_CALUDE_third_sibling_age_difference_l1653_165304

/-- Represents the ages of four siblings --/
structure SiblingAges where
  youngest : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the sibling age problem --/
def siblingAgeProblem (ages : SiblingAges) : Prop :=
  ages.youngest = 25.75 ∧
  ages.second = ages.youngest + 3 ∧
  ages.third = ages.youngest + 6 ∧
  (ages.youngest + ages.second + ages.third + ages.fourth) / 4 = 30

/-- The theorem stating that the third sibling is 6 years older than the youngest --/
theorem third_sibling_age_difference (ages : SiblingAges) :
  siblingAgeProblem ages → ages.third - ages.youngest = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_sibling_age_difference_l1653_165304


namespace NUMINAMATH_CALUDE_gcd_g_x_l1653_165331

def g (x : ℤ) : ℤ := (3*x+8)*(5*x+1)*(11*x+6)*(2*x+3)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 12096 * k) : 
  Int.gcd (g x) x = 144 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l1653_165331


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l1653_165375

theorem tennis_ball_ratio (total_ordered : ℕ) (extra_yellow : ℕ) : 
  total_ordered = 288 →
  extra_yellow = 90 →
  let white := total_ordered / 2
  let yellow := total_ordered / 2 + extra_yellow
  (white : ℚ) / yellow = 8 / 13 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l1653_165375


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_l1653_165346

/-- The fixed point through which all lines in the family pass -/
def fixed_point : ℝ × ℝ := (2, 2)

/-- The equation of the line for a given k -/
def line_equation (k x y : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + (2 - 14*k) = 0

/-- Theorem stating that the fixed point satisfies the line equation for all k -/
theorem fixed_point_on_all_lines :
  ∀ k : ℝ, line_equation k fixed_point.1 fixed_point.2 :=
by
  sorry


end NUMINAMATH_CALUDE_fixed_point_on_all_lines_l1653_165346


namespace NUMINAMATH_CALUDE_rent_increase_problem_l1653_165303

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 880) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l1653_165303


namespace NUMINAMATH_CALUDE_gcf_of_lcms_eq_210_l1653_165335

theorem gcf_of_lcms_eq_210 : Nat.gcd (Nat.lcm 10 21) (Nat.lcm 14 15) = 210 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_eq_210_l1653_165335


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1653_165350

def A : Set ℤ := {-1, 1}

def B : Set ℤ := {x | |x + 1/2| < 3/2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1653_165350


namespace NUMINAMATH_CALUDE_lucy_age_is_12_l1653_165321

def sisters_ages : List Nat := [2, 4, 6, 10, 12, 14]

def movie_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b = 20

def basketball_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a ≤ 10 ∧ b ≤ 10

def staying_home (lucy_age : Nat) (ages : List Nat) : Prop :=
  lucy_age ∈ ages ∧ ∃ a, a ∈ ages ∧ a ≠ lucy_age

theorem lucy_age_is_12 :
  movie_pair sisters_ages →
  basketball_pair sisters_ages →
  ∃ lucy_age, staying_home lucy_age sisters_ages ∧ lucy_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_lucy_age_is_12_l1653_165321


namespace NUMINAMATH_CALUDE_min_value_f_l1653_165396

def f (c d x : ℝ) : ℝ := x^3 + c*x + d

theorem min_value_f (c d : ℝ) (h : c = 0) : 
  ∀ x, f c d x ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l1653_165396


namespace NUMINAMATH_CALUDE_parallel_condition_l1653_165378

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2-a)x+y=0 -/
def slope1 (a : ℝ) : ℝ := a^2 - a

/-- The slope of the line 2x+y+1=0 -/
def slope2 : ℝ := 2

theorem parallel_condition (a : ℝ) :
  (a = 2 → parallel (slope1 a) slope2) ∧
  (∃ b : ℝ, b ≠ 2 ∧ parallel (slope1 b) slope2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1653_165378


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_of_digits_divisible_l1653_165315

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (start : ℕ) (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → k ≤ N → (sum_of_digits (start + k - 1)) % k = 0

theorem largest_consecutive_sum_of_digits_divisible :
  ∃ start : ℕ, satisfies_condition start 21 ∧
  ∀ N : ℕ, N > 21 → ¬∃ start : ℕ, satisfies_condition start N :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_of_digits_divisible_l1653_165315


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l1653_165380

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 7

/-- The maximum number of times each element can be used -/
def max_repetitions : ℕ := 4

/-- The number of possible sequences -/
def num_sequences : ℕ := num_elements ^ sequence_length

theorem acme_vowel_soup_sequences :
  num_sequences = 78125 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l1653_165380


namespace NUMINAMATH_CALUDE_keith_cards_theorem_l1653_165323

/-- Represents the number of cards Keith has after his dog eats half of his collection -/
def cards_left (initial_cards : ℕ) : ℕ := initial_cards / 2

/-- Theorem stating that if Keith has 8 cards and his dog eats half, he has 4 cards left -/
theorem keith_cards_theorem :
  cards_left 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_cards_theorem_l1653_165323


namespace NUMINAMATH_CALUDE_marias_car_trip_l1653_165320

theorem marias_car_trip (D : ℝ) : 
  D / 2 + (D - D / 2) / 4 + 210 = D → D = 560 := by sorry

end NUMINAMATH_CALUDE_marias_car_trip_l1653_165320


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1653_165308

theorem quadratic_distinct_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1653_165308


namespace NUMINAMATH_CALUDE_smoothie_cost_l1653_165319

/-- The cost of Morgan's smoothie given the prices of other items and the transaction details. -/
theorem smoothie_cost (hamburger_cost onion_rings_cost amount_paid change_received : ℕ) : 
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  amount_paid = 20 →
  change_received = 11 →
  amount_paid - change_received - (hamburger_cost + onion_rings_cost) = 3 := by
  sorry

#check smoothie_cost

end NUMINAMATH_CALUDE_smoothie_cost_l1653_165319


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1653_165366

theorem lattice_points_on_hyperbola :
  let equation := fun (x y : ℤ) => x^2 - y^2 = 1500^2
  (∑' p : ℤ × ℤ, if equation p.1 p.2 then 1 else 0) = 90 :=
sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1653_165366


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1653_165317

/-- Given two perpendicular lines with direction vectors (2, 5) and (b, -3), prove that b = 15/2 -/
theorem perpendicular_lines_b_value (b : ℝ) : 
  let v₁ : Fin 2 → ℝ := ![2, 5]
  let v₂ : Fin 2 → ℝ := ![b, -3]
  (∀ i : Fin 2, v₁ i * v₂ i = 0) → b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1653_165317


namespace NUMINAMATH_CALUDE_left_handed_rock_fans_under_25_l1653_165306

/-- Represents the number of people with specific characteristics in a workshop. -/
structure WorkshopPeople where
  total : ℕ
  leftHanded : ℕ
  rockMusicFans : ℕ
  rightHandedNotRockFans : ℕ
  under25 : ℕ
  rightHandedUnder25RockFans : ℕ

/-- Theorem stating the number of left-handed, rock music fans under 25 in the workshop. -/
theorem left_handed_rock_fans_under_25 (w : WorkshopPeople) 
  (h1 : w.total = 30)
  (h2 : w.leftHanded = 12)
  (h3 : w.rockMusicFans = 18)
  (h4 : w.rightHandedNotRockFans = 5)
  (h5 : w.under25 = 9)
  (h6 : w.rightHandedUnder25RockFans = 3)
  (h7 : w.leftHanded + (w.total - w.leftHanded) = w.total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (w.leftHanded - x) + (w.rockMusicFans - x) + w.rightHandedNotRockFans + 
    w.rightHandedUnder25RockFans + (w.total - w.leftHanded - w.rightHandedNotRockFans - 
    w.rightHandedUnder25RockFans - x) = w.total :=
  sorry


end NUMINAMATH_CALUDE_left_handed_rock_fans_under_25_l1653_165306


namespace NUMINAMATH_CALUDE_equal_shaded_unshaded_probability_l1653_165345

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents a circle -/
structure Circle :=
  (diameter : ℝ)

/-- Represents a position on the grid -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of favorable positions -/
def count_favorable_positions (g : Grid) (c : Circle) : ℕ := sorry

/-- Counts the total number of possible positions -/
def count_total_positions (g : Grid) : ℕ := sorry

/-- Calculates the probability of placing the circle in a favorable position -/
def probability_favorable_position (g : Grid) (c : Circle) : ℚ :=
  (count_favorable_positions g c : ℚ) / (count_total_positions g : ℚ)

theorem equal_shaded_unshaded_probability 
  (g : Grid) 
  (c : Circle) 
  (h1 : g.square_size = 2)
  (h2 : c.diameter = 3)
  (h3 : g.size = 5) :
  probability_favorable_position g c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_equal_shaded_unshaded_probability_l1653_165345


namespace NUMINAMATH_CALUDE_bee_legs_count_l1653_165357

theorem bee_legs_count (legs_per_bee : ℕ) (num_bees : ℕ) (h : legs_per_bee = 6) :
  legs_per_bee * num_bees = 48 ↔ num_bees = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_legs_count_l1653_165357


namespace NUMINAMATH_CALUDE_advance_tickets_sold_l1653_165325

/-- Proves that the number of advance tickets sold is 20 given the ticket prices, total tickets sold, and total receipts. -/
theorem advance_tickets_sold (advance_price same_day_price total_tickets total_receipts : ℕ) 
  (h1 : advance_price = 20)
  (h2 : same_day_price = 30)
  (h3 : total_tickets = 60)
  (h4 : total_receipts = 1600) : 
  ∃ (advance_tickets : ℕ), 
    advance_tickets * advance_price + (total_tickets - advance_tickets) * same_day_price = total_receipts ∧ 
    advance_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_advance_tickets_sold_l1653_165325


namespace NUMINAMATH_CALUDE_gcd_problem_l1653_165356

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = k * 7771) : 
  Nat.gcd (Int.natAbs (8 * a^2 + 57 * a + 132)) (Int.natAbs (2 * a + 9)) = 9 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l1653_165356


namespace NUMINAMATH_CALUDE_log_problem_l1653_165397

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (log10 x)^3 - log10 (x^3) = 125) :
  (log10 x)^4 - log10 (x^4) = 645 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1653_165397


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1653_165334

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2*a*(-1) - b*2 + 2 = 0) → (1/a + 1/b ≥ 4) ∧ ∃ a b, (1/a + 1/b = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1653_165334


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l1653_165343

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number -/
def binary_num : ℕ := 11011001

/-- The base 4 representation of the number -/
def base4_num : ℕ := 3121

theorem binary_to_base4_conversion :
  binary_to_base4 binary_num = base4_num := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l1653_165343


namespace NUMINAMATH_CALUDE_geometric_sum_456_l1653_165355

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_456 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 9 →
  (a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_456_l1653_165355


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1653_165371

def parallelogram (A B C D : ℂ) : Prop :=
  D - A = C - B

theorem parallelogram_fourth_vertex 
  (A B C D : ℂ) 
  (h1 : A = 1 + 3*I) 
  (h2 : B = -I) 
  (h3 : C = 2 + I) 
  (h4 : parallelogram A B C D) : 
  D = 3 + 5*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1653_165371


namespace NUMINAMATH_CALUDE_circle_center_proof_l1653_165392

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 10x + y² - 8y = 16, 
    prove that its center is (5, 4) -/
theorem circle_center_proof (eq : CircleEquation) 
  (h1 : eq.a = 1)
  (h2 : eq.b = -10)
  (h3 : eq.c = 1)
  (h4 : eq.d = -8)
  (h5 : eq.e = -16) :
  CircleCenter.mk 5 4 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_proof_l1653_165392


namespace NUMINAMATH_CALUDE_probability_of_eight_in_three_elevenths_l1653_165394

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a list of digits -/
def digit_probability (d : ℕ) (l : List ℕ) : ℚ := sorry

theorem probability_of_eight_in_three_elevenths :
  digit_probability 8 (decimal_representation (3/11)) = 0 := by sorry

end NUMINAMATH_CALUDE_probability_of_eight_in_three_elevenths_l1653_165394


namespace NUMINAMATH_CALUDE_lesser_fraction_l1653_165393

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1653_165393


namespace NUMINAMATH_CALUDE_expression_simplification_l1653_165312

theorem expression_simplification :
  (((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 9)^2 / 9)) = 268 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1653_165312


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1653_165367

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∉ (A ∩ B) ↔ (x ≤ 4 ∨ x > 5) := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (A ∪ B) ⊆ C a → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1653_165367


namespace NUMINAMATH_CALUDE_solve_for_y_l1653_165373

theorem solve_for_y (x y : ℝ) 
  (eq1 : 9823 + x = 13200) 
  (eq2 : x = y / 3 + 37.5) : 
  y = 10018.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1653_165373


namespace NUMINAMATH_CALUDE_root_difference_l1653_165318

-- Define the equations
def equation1 (x : ℝ) : Prop := 2002^2 * x^2 - 2003 * 2001 * x - 1 = 0
def equation2 (x : ℝ) : Prop := 2001 * x^2 - 2002 * x + 1 = 0

-- Define r and s
def r : ℝ := sorry
def s : ℝ := sorry

-- State the theorem
theorem root_difference : 
  (equation1 r ∧ ∀ x, equation1 x → x ≤ r) ∧ 
  (equation2 s ∧ ∀ x, equation2 x → x ≥ s) → 
  r - s = 2000 / 2001 := by sorry

end NUMINAMATH_CALUDE_root_difference_l1653_165318


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1653_165376

-- Define the universal set U
def U : Set Nat := {1, 3, 5}

-- Define the set A
def A : Set Nat := {1, 5}

-- State the theorem
theorem complement_of_A_wrt_U :
  {x ∈ U | x ∉ A} = {3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1653_165376


namespace NUMINAMATH_CALUDE_librarian_took_books_l1653_165332

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  total_books = 34 →
  books_per_shelf = 3 →
  shelves_needed = 9 →
  total_books - (books_per_shelf * shelves_needed) = 7 := by
sorry

end NUMINAMATH_CALUDE_librarian_took_books_l1653_165332


namespace NUMINAMATH_CALUDE_flagpole_break_height_l1653_165309

/-- Given a flagpole of height 8 meters that breaks and touches the ground 3 meters from its base,
    the height of the break point is √3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 8 →  -- Original height of flagpole
  x^2 + 3^2 = (h - x)^2 →  -- Pythagorean theorem application
  x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l1653_165309


namespace NUMINAMATH_CALUDE_shorter_base_length_for_specific_trapezoid_l1653_165398

/-- Represents a trapezoid with a median line divided by a diagonal -/
structure TrapezoidWithDiagonal where
  median_length : ℝ
  segment_difference : ℝ

/-- Calculates the length of the shorter base of the trapezoid -/
def shorter_base_length (t : TrapezoidWithDiagonal) : ℝ :=
  t.median_length - t.segment_difference

/-- Theorem stating the length of the shorter base given specific measurements -/
theorem shorter_base_length_for_specific_trapezoid :
  let t : TrapezoidWithDiagonal := { median_length := 16, segment_difference := 4 }
  shorter_base_length t = 12 := by
  sorry

end NUMINAMATH_CALUDE_shorter_base_length_for_specific_trapezoid_l1653_165398


namespace NUMINAMATH_CALUDE_unique_recovery_l1653_165362

theorem unique_recovery (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_operations : ∃ (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0),
    ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y}) :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y} :=
sorry

end NUMINAMATH_CALUDE_unique_recovery_l1653_165362


namespace NUMINAMATH_CALUDE_system_solution_l1653_165339

theorem system_solution (x y : ℚ) : 
  (3 * x - 2 * y = 8) ∧ (x + 3 * y = 7) → x = 38 / 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1653_165339


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1653_165379

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - 21*x^3 + 8*x^2 - 17*x + 12 = (x - 3)*(x^4 + 3*x^3 - 12*x^2 - 28*x - 101) + (-201) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1653_165379


namespace NUMINAMATH_CALUDE_sixty_second_pair_l1653_165322

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- The main theorem stating that the 62nd pair is (7,5) -/
theorem sixty_second_pair :
  nthPair 62 = IntPair.mk 7 5 := by
  sorry

end NUMINAMATH_CALUDE_sixty_second_pair_l1653_165322


namespace NUMINAMATH_CALUDE_monster_family_eyes_total_l1653_165374

/-- The number of eyes in the extended monster family -/
def monster_family_eyes : ℕ :=
  let mom_eyes := 1
  let dad_eyes := 3
  let mom_dad_kids_eyes := 3 * 4
  let mom_previous_child_eyes := 5
  let dad_previous_children_eyes := 6 + 2
  let dad_ex_wife_eyes := 1
  let dad_ex_wife_partner_eyes := 7
  let dad_ex_wife_child_eyes := 8
  mom_eyes + dad_eyes + mom_dad_kids_eyes + mom_previous_child_eyes +
  dad_previous_children_eyes + dad_ex_wife_eyes + dad_ex_wife_partner_eyes +
  dad_ex_wife_child_eyes

/-- The total number of eyes in the extended monster family is 45 -/
theorem monster_family_eyes_total :
  monster_family_eyes = 45 := by sorry

end NUMINAMATH_CALUDE_monster_family_eyes_total_l1653_165374


namespace NUMINAMATH_CALUDE_min_fraction_value_l1653_165370

theorem min_fraction_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 → -3 ≤ y' ∧ y' ≤ 1 → (x' + y') / x' ≥ (x + y) / x) →
  (x + y) / x = 0.8 := by
sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1653_165370


namespace NUMINAMATH_CALUDE_system_solution_exists_l1653_165384

theorem system_solution_exists : ∃ (x y : ℝ), 
  (x * Real.sqrt (x * y) + y * Real.sqrt (x * y) = 10) ∧ 
  (x^2 + y^2 = 17) := by
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1653_165384


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l1653_165341

/-- Given a rectangle with one side length b, diagonal length d, and the difference between
    the diagonal and the other side (d-a), prove that the side lengths of the rectangle
    are a = d - √(d² - b²) and b. -/
theorem rectangle_side_lengths
  (b d : ℝ) (h : b > 0) (h' : d > b) :
  let a := d - Real.sqrt (d^2 - b^2)
  (a > 0 ∧ a < d) ∧ 
  (a^2 + b^2 = d^2) ∧
  (d - a = Real.sqrt (d^2 - b^2)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l1653_165341


namespace NUMINAMATH_CALUDE_smallest_angle_tan_equation_l1653_165329

open Real

theorem smallest_angle_tan_equation (x : ℝ) : 
  (0 < x) ∧ 
  (x < 2 * π) ∧
  (tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) →
  x = 5.625 * (π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tan_equation_l1653_165329


namespace NUMINAMATH_CALUDE_Q_subset_P_l1653_165324

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x < 2}
def Q : Set ℝ := {x : ℝ | x^2 < 1}

-- Theorem statement
theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l1653_165324


namespace NUMINAMATH_CALUDE_second_serving_is_ten_l1653_165352

/-- Represents the number of maggots in various scenarios --/
structure MaggotCounts where
  total : ℕ
  firstServing : ℕ
  firstEaten : ℕ
  secondEaten : ℕ

/-- Calculates the number of maggots in the second serving --/
def secondServing (counts : MaggotCounts) : ℕ :=
  counts.total - counts.firstServing

/-- Theorem stating that the second serving contains 10 maggots --/
theorem second_serving_is_ten (counts : MaggotCounts)
  (h1 : counts.total = 20)
  (h2 : counts.firstServing = 10)
  (h3 : counts.firstEaten = 1)
  (h4 : counts.secondEaten = 3) :
  secondServing counts = 10 := by
  sorry

#eval secondServing { total := 20, firstServing := 10, firstEaten := 1, secondEaten := 3 }

end NUMINAMATH_CALUDE_second_serving_is_ten_l1653_165352


namespace NUMINAMATH_CALUDE_transistor_growth_1992_to_2004_l1653_165314

/-- Moore's Law: Number of transistors doubles every 2 years -/
def moores_law (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

/-- Theorem: A CPU with 2,000,000 transistors in 1992 would have 128,000,000 transistors in 2004 -/
theorem transistor_growth_1992_to_2004 :
  moores_law 2000000 (2004 - 1992) = 128000000 := by
  sorry

#eval moores_law 2000000 (2004 - 1992)

end NUMINAMATH_CALUDE_transistor_growth_1992_to_2004_l1653_165314


namespace NUMINAMATH_CALUDE_total_book_pairs_l1653_165347

-- Define the number of books in each genre
def mystery_books : Nat := 4
def fantasy_books : Nat := 4
def biography_books : Nat := 3

-- Define the function to calculate the number of book pairs
def book_pairs : Nat :=
  mystery_books * fantasy_books +
  mystery_books * biography_books +
  fantasy_books * biography_books

-- Theorem statement
theorem total_book_pairs : book_pairs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_book_pairs_l1653_165347


namespace NUMINAMATH_CALUDE_sin_cos_graph_shift_l1653_165360

theorem sin_cos_graph_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x - Real.pi / 4)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  ∃ (shift : ℝ), shift = 3 * Real.pi / 8 ∧
    f x = g (x - shift) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_graph_shift_l1653_165360


namespace NUMINAMATH_CALUDE_student_subject_distribution_l1653_165372

theorem student_subject_distribution (total : ℕ) 
  (math_eng : ℕ) (all_three : ℕ) (math_hist : ℕ) :
  total = 228 →
  math_eng ≥ 0 →
  all_three > 0 →
  all_three % 2 = 0 →
  math_hist = 6 →
  2 * math_eng + 6 * all_three + math_hist = total →
  math_eng = 5 :=
by sorry

end NUMINAMATH_CALUDE_student_subject_distribution_l1653_165372


namespace NUMINAMATH_CALUDE_kyles_speed_l1653_165358

theorem kyles_speed (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (distance_difference : ℝ) :
  joseph_speed = 50 →
  joseph_time = 2.5 →
  kyle_time = 2 →
  distance_difference = 1 →
  joseph_speed * joseph_time = kyle_time * (joseph_speed * joseph_time - distance_difference) / kyle_time →
  (joseph_speed * joseph_time - distance_difference) / kyle_time = 62 :=
by
  sorry

#check kyles_speed

end NUMINAMATH_CALUDE_kyles_speed_l1653_165358


namespace NUMINAMATH_CALUDE_plate_cup_cost_l1653_165363

/-- Given that 100 paper plates and 200 paper cups cost $6.00 in total,
    prove that 20 paper plates and 40 paper cups cost $1.20. -/
theorem plate_cup_cost (plate_cost cup_cost : ℝ) 
  (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_plate_cup_cost_l1653_165363


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_l1653_165310

theorem quadratic_equation_single_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 15 * x + 6 = 0) :
  ∃ x, b * x^2 + 15 * x + 6 = 0 ∧ x = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_l1653_165310


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1653_165388

theorem arithmetic_geometric_mean_inequality 
  (a b c d x y : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_arithmetic : b - a = c - b ∧ c - b = d - c) 
  (h_x : x = (a + d) / 2) 
  (h_y : y = Real.sqrt (b * c)) : 
  x ≥ y := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1653_165388


namespace NUMINAMATH_CALUDE_inequality_proof_l1653_165385

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1653_165385


namespace NUMINAMATH_CALUDE_range_of_a_l1653_165333

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def proposition_p (a : ℝ) : Prop :=
  is_monotonically_increasing (fun x => a^x)

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ¬(¬proposition_p a ∧ ¬proposition_q a))
  (h3 : proposition_p a ∨ proposition_q a) :
  a ∈ Set.Ioo 0 1 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1653_165333


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1653_165311

theorem quadratic_equation_roots (k : ℝ) (h : k > 1) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (2 * x₁^2 - (4*k + 1) * x₁ + 2*k^2 - 1 = 0) ∧
  (2 * x₂^2 - (4*k + 1) * x₂ + 2*k^2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1653_165311


namespace NUMINAMATH_CALUDE_tree_purchase_solution_l1653_165305

/-- Represents the unit prices and purchasing schemes for tree seedlings -/
structure TreePurchase where
  osmanthus_price : ℕ
  camphor_price : ℕ
  schemes : List (ℕ × ℕ)

/-- Defines the conditions of the tree purchasing problem -/
def tree_purchase_problem (p : TreePurchase) : Prop :=
  -- First purchase condition
  10 * p.osmanthus_price + 20 * p.camphor_price = 3000 ∧
  -- Second purchase condition
  8 * p.osmanthus_price + 24 * p.camphor_price = 2800 ∧
  -- Next purchase conditions
  (∀ (o c : ℕ), (o, c) ∈ p.schemes →
    o + c = 40 ∧
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 ∧
    c ≤ 3 * o) ∧
  -- All possible schemes are included
  (∀ (o c : ℕ), o + c = 40 →
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 →
    c ≤ 3 * o →
    (o, c) ∈ p.schemes)

/-- Theorem stating the solution to the tree purchasing problem -/
theorem tree_purchase_solution :
  ∃ (p : TreePurchase),
    tree_purchase_problem p ∧
    p.osmanthus_price = 200 ∧
    p.camphor_price = 50 ∧
    p.schemes = [(10, 30), (11, 29), (12, 28)] :=
  sorry

end NUMINAMATH_CALUDE_tree_purchase_solution_l1653_165305


namespace NUMINAMATH_CALUDE_nectar_water_content_l1653_165326

/-- The percentage of water in honey -/
def honey_water_percentage : ℝ := 25

/-- The weight of flower-nectar needed to produce 1 kg of honey -/
def nectar_weight : ℝ := 1.5

/-- The weight of honey produced from nectar_weight of flower-nectar -/
def honey_weight : ℝ := 1

/-- The percentage of water in flower-nectar -/
def nectar_water_percentage : ℝ := 50

theorem nectar_water_content :
  nectar_water_percentage = 50 :=
sorry

end NUMINAMATH_CALUDE_nectar_water_content_l1653_165326


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1653_165344

/-- The equation (a-2)x^|a-1| + 3 = 9 is linear in x if and only if a = 0 -/
theorem linear_equation_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a - 2) * x^(|a - 1|) + 3 = b * x + c) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1653_165344


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l1653_165359

theorem trig_identity_simplification (θ : Real) :
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) =
  4 * (Real.sin (2 * θ)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l1653_165359


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1653_165302

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 - 24 * x + 10 = a * (x - h)^2 + k) ∧ (a + h + k = -6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1653_165302


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1653_165354

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l1653_165354


namespace NUMINAMATH_CALUDE_incorrect_expression_l1653_165337

theorem incorrect_expression (x y : ℝ) (h : x / y = 3 / 4) : 
  (x - y) / y = -1 / 4 ∧ (x - y) / y ≠ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1653_165337


namespace NUMINAMATH_CALUDE_power_function_increasing_m_eq_3_l1653_165381

/-- A function f(x) = cx^p where c and p are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ c p : ℝ, ∀ x > 0, f x = c * x^p

/-- A function f is increasing on (0, +∞) if for all x, y > 0, x < y implies f(x) < f(y) -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

/-- The main theorem stating that m = 3 is the only value satisfying the conditions -/
theorem power_function_increasing_m_eq_3 :
  ∃! m : ℝ, 
    isPowerFunction (fun x => (m^2 - m - 5) * x^(m-1)) ∧ 
    isIncreasing (fun x => (m^2 - m - 5) * x^(m-1)) :=
sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_eq_3_l1653_165381


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l1653_165330

theorem fraction_inequality_counterexample :
  ∃ (a b c d A B C D : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
    a/b > A/B ∧ 
    c/d > C/D ∧ 
    (a+c)/(b+d) ≤ (A+C)/(B+D) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l1653_165330


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l1653_165364

theorem remainder_of_large_number (M : ℕ) (d : ℕ) (h : M = 123456789012 ∧ d = 252) :
  M % d = 228 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l1653_165364


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1653_165390

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1653_165390


namespace NUMINAMATH_CALUDE_guppy_angelfish_ratio_l1653_165307

/-- Proves that the ratio of guppies to angelfish is 2:1 given the conditions -/
theorem guppy_angelfish_ratio :
  let goldfish : ℕ := 8
  let angelfish : ℕ := goldfish + 4
  let total_fish : ℕ := 44
  let guppies : ℕ := total_fish - (goldfish + angelfish)
  (guppies : ℚ) / angelfish = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_guppy_angelfish_ratio_l1653_165307


namespace NUMINAMATH_CALUDE_pages_left_to_read_total_annotated_pages_l1653_165377

-- Define the book and reading parameters
def total_pages : ℕ := 567
def pages_read_week1 : ℕ := 279
def pages_read_week2 : ℕ := 124
def pages_annotated_week1 : ℕ := 35
def pages_annotated_week2 : ℕ := 15

-- Theorem for pages left to read
theorem pages_left_to_read : 
  total_pages - (pages_read_week1 + pages_read_week2) = 164 := by sorry

-- Theorem for total annotated pages
theorem total_annotated_pages :
  pages_annotated_week1 + pages_annotated_week2 = 50 := by sorry

end NUMINAMATH_CALUDE_pages_left_to_read_total_annotated_pages_l1653_165377


namespace NUMINAMATH_CALUDE_area_difference_of_square_fields_l1653_165349

/-- Given two square fields where the second field's side length is 1% longer than the first,
    and the area of the first field is 1 hectare (10,000 square meters),
    prove that the difference in area between the two fields is 201 square meters. -/
theorem area_difference_of_square_fields (a : ℝ) : 
  a^2 = 10000 → (1.01 * a)^2 - a^2 = 201 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_of_square_fields_l1653_165349


namespace NUMINAMATH_CALUDE_apple_count_l1653_165336

theorem apple_count (initial_oranges : ℕ) (removed_oranges : ℕ) (apples : ℕ) : 
  initial_oranges = 23 →
  removed_oranges = 13 →
  apples = (initial_oranges - removed_oranges) →
  apples = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_count_l1653_165336


namespace NUMINAMATH_CALUDE_probability_one_defective_l1653_165328

/-- The probability of selecting exactly one defective product from a batch -/
theorem probability_one_defective (total : ℕ) (defective : ℕ) : 
  total = 40 →
  defective = 12 →
  (Nat.choose (total - defective) 1 * Nat.choose defective 1) / Nat.choose total 2 = 28 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_l1653_165328
