import Mathlib

namespace NUMINAMATH_CALUDE_infinite_sequence_exists_l2280_228019

-- Define the Ω function
def Omega (n : ℕ+) : ℕ := sorry

-- Define the f function
def f (n : ℕ+) : Int := (-1) ^ (Omega n)

-- State the theorem
theorem infinite_sequence_exists : 
  ∃ (seq : ℕ → ℕ+), (∀ i : ℕ, 
    f (seq i - 1) = 1 ∧ 
    f (seq i) = 1 ∧ 
    f (seq i + 1) = 1) ∧ 
  (∀ i j : ℕ, i ≠ j → seq i ≠ seq j) :=
sorry

end NUMINAMATH_CALUDE_infinite_sequence_exists_l2280_228019


namespace NUMINAMATH_CALUDE_smallest_divisor_after_429_l2280_228096

theorem smallest_divisor_after_429 (n : ℕ) : 
  10000 ≤ n ∧ n < 100000 →  -- n is a five-digit number
  429 ∣ n →                 -- 429 is a divisor of n
  ∃ d : ℕ, d ∣ n ∧ 429 < d ∧ d ≤ 858 ∧ 
    ∀ d' : ℕ, d' ∣ n → 429 < d' → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_429_l2280_228096


namespace NUMINAMATH_CALUDE_faith_works_five_days_l2280_228031

/-- Faith's work schedule and earnings --/
def faith_work_schedule (hourly_rate : ℚ) (regular_hours : ℕ) (overtime_hours : ℕ) (weekly_earnings : ℚ) : Prop :=
  ∃ (days_worked : ℕ),
    (hourly_rate * regular_hours + hourly_rate * 1.5 * overtime_hours) * days_worked = weekly_earnings ∧
    days_worked ≤ 7

theorem faith_works_five_days :
  faith_work_schedule 13.5 8 2 675 →
  ∃ (days_worked : ℕ), days_worked = 5 := by
  sorry

end NUMINAMATH_CALUDE_faith_works_five_days_l2280_228031


namespace NUMINAMATH_CALUDE_lisa_candies_l2280_228065

/-- The number of candies Lisa eats on Mondays and Wednesdays combined per week -/
def candies_mon_wed : ℕ := 4

/-- The number of candies Lisa eats on other days combined per week -/
def candies_other_days : ℕ := 5

/-- The number of weeks it takes Lisa to eat all her candies -/
def weeks_to_eat_all : ℕ := 4

/-- The total number of candies Lisa has -/
def total_candies : ℕ := (candies_mon_wed + candies_other_days) * weeks_to_eat_all

theorem lisa_candies : total_candies = 36 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candies_l2280_228065


namespace NUMINAMATH_CALUDE_min_p_value_l2280_228010

/-- The probability that Alex and Dylan are on the same team, given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  let total_combinations := (52 - 2).choose 2
  let lower_team_combinations := (44 - a).choose 2
  let higher_team_combinations := (a - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 8

theorem min_p_value :
  p min_a = 73 / 137 ∧ 
  p min_a ≥ 1 / 2 ∧
  ∀ a : ℕ, a < min_a → p a < 1 / 2 := by sorry

end NUMINAMATH_CALUDE_min_p_value_l2280_228010


namespace NUMINAMATH_CALUDE_compare_data_fluctuation_l2280_228039

def group_mean (g : String) : ℝ :=
  match g with
  | "A" => 80
  | "B" => 90
  | _ => 0

def group_variance (g : String) : ℝ :=
  match g with
  | "A" => 10
  | "B" => 5
  | _ => 0

def less_fluctuation (g1 g2 : String) : Prop :=
  group_variance g1 < group_variance g2

theorem compare_data_fluctuation (g1 g2 : String) :
  less_fluctuation g1 g2 → group_variance g1 < group_variance g2 :=
by sorry

end NUMINAMATH_CALUDE_compare_data_fluctuation_l2280_228039


namespace NUMINAMATH_CALUDE_floor_width_calculation_l2280_228025

/-- Given a rectangular floor of length 10 m, covered by a square carpet of side 4 m,
    with 64 square meters uncovered, the width of the floor is 8 m. -/
theorem floor_width_calculation (floor_length : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_length = 10 →
  carpet_side = 4 →
  uncovered_area = 64 →
  ∃ (width : ℝ), width = 8 ∧ floor_length * width = carpet_side^2 + uncovered_area :=
by sorry

end NUMINAMATH_CALUDE_floor_width_calculation_l2280_228025


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_a_gt_one_implies_unique_zero_characterization_of_a_l2280_228006

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

/-- The statement that f has exactly one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f a x = 0

/-- The main theorem: if f has exactly one zero in (0, 1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 := by
  sorry

/-- The converse: if a > 1, then f has exactly one zero in (0, 1) -/
theorem a_gt_one_implies_unique_zero :
  ∀ a : ℝ, a > 1 → has_unique_zero_in_interval a := by
  sorry

/-- The final theorem: f has exactly one zero in (0, 1) if and only if a > 1 -/
theorem characterization_of_a :
  ∀ a : ℝ, has_unique_zero_in_interval a ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_a_gt_one_implies_unique_zero_characterization_of_a_l2280_228006


namespace NUMINAMATH_CALUDE_stamp_problem_l2280_228012

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

/-- The minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

theorem stamp_problem :
  let stamps := [5, 7]
  minCoins 50 stamps = 8 :=
by sorry

end NUMINAMATH_CALUDE_stamp_problem_l2280_228012


namespace NUMINAMATH_CALUDE_replaced_student_weight_l2280_228051

theorem replaced_student_weight
  (n : ℕ)
  (initial_total_weight : ℝ)
  (new_student_weight : ℝ)
  (average_decrease : ℝ)
  (h1 : n = 10)
  (h2 : new_student_weight = 60)
  (h3 : average_decrease = 6)
  (h4 : initial_total_weight - (initial_total_weight / n) + (new_student_weight / n) = initial_total_weight - n * average_decrease) :
  initial_total_weight / n - new_student_weight + n * average_decrease = 120 := by
sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l2280_228051


namespace NUMINAMATH_CALUDE_factor_x10_minus_1024_l2280_228055

theorem factor_x10_minus_1024 (x : ℝ) : x^10 - 1024 = (x^5 + 32) * (x^5 - 32) := by
  sorry

end NUMINAMATH_CALUDE_factor_x10_minus_1024_l2280_228055


namespace NUMINAMATH_CALUDE_mike_initial_nickels_l2280_228064

/-- The number of nickels Mike's dad borrowed -/
def borrowed_nickels : ℕ := 75

/-- The number of nickels Mike has now -/
def current_nickels : ℕ := 12

/-- The number of nickels Mike had initially -/
def initial_nickels : ℕ := borrowed_nickels + current_nickels

theorem mike_initial_nickels :
  initial_nickels = 87 :=
by sorry

end NUMINAMATH_CALUDE_mike_initial_nickels_l2280_228064


namespace NUMINAMATH_CALUDE_equidistant_point_sum_of_distances_equidistant_times_l2280_228024

-- Define the points A and B
def A : ℝ := -2
def B : ℝ := 4

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the distances from P to A and B
def distPA (x : ℝ) : ℝ := |x - A|
def distPB (x : ℝ) : ℝ := |x - B|

-- Define the positions of M and N after t seconds
def M (t : ℝ) : ℝ := A - t
def N (t : ℝ) : ℝ := B - 3*t

-- Define the origin O
def O : ℝ := 0

-- Theorem 1: The point equidistant from A and B
theorem equidistant_point : ∃ x : ℝ, distPA x = distPB x ∧ x = 1 := by sorry

-- Theorem 2: Points where sum of distances from A and B is 8
theorem sum_of_distances : ∃ x₁ x₂ : ℝ, 
  distPA x₁ + distPB x₁ = 8 ∧ 
  distPA x₂ + distPB x₂ = 8 ∧ 
  x₁ = -3 ∧ x₂ = 5 := by sorry

-- Theorem 3: Times when one point is equidistant from the other two
theorem equidistant_times : ∃ t₁ t₂ t₃ t₄ t₅ : ℝ,
  (|M t₁| = |N t₁| ∧ t₁ = 1/2) ∧
  (N t₂ = O ∧ t₂ = 4/3) ∧
  (|N t₃ - O| = |N t₃ - M t₃| ∧ t₃ = 2) ∧
  (M t₄ = N t₄ ∧ t₄ = 3) ∧
  (|M t₅ - O| = |M t₅ - N t₅| ∧ t₅ = 8) := by sorry

end NUMINAMATH_CALUDE_equidistant_point_sum_of_distances_equidistant_times_l2280_228024


namespace NUMINAMATH_CALUDE_M_is_real_l2280_228071

-- Define the set M
def M : Set ℂ := {z : ℂ | (z - 1)^2 = Complex.abs (z - 1)^2}

-- Theorem stating that M is equal to the set of real numbers
theorem M_is_real : M = {z : ℂ | z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_is_real_l2280_228071


namespace NUMINAMATH_CALUDE_triangle_properties_l2280_228037

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a + b + c = Real.sqrt 2 + 1 →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  (1/2) * b * c * Real.sin A = (1/6) * Real.sin C →
  a = 1 ∧ A = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2280_228037


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2280_228082

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  collinear a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2280_228082


namespace NUMINAMATH_CALUDE_billie_bakes_three_pies_l2280_228033

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := sorry

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_bakes_three_pies : 
  pies_per_day * baking_days = pies_eaten + cans_needed / cans_per_pie ∧ 
  pies_per_day = 3 := by sorry

end NUMINAMATH_CALUDE_billie_bakes_three_pies_l2280_228033


namespace NUMINAMATH_CALUDE_points_five_units_away_l2280_228054

theorem points_five_units_away (x : ℝ) : 
  (|x - 2| = 5) ↔ (x = 7 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_points_five_units_away_l2280_228054


namespace NUMINAMATH_CALUDE_bolzano_weierstrass_unit_interval_l2280_228061

/-- Bolzano-Weierstrass theorem for sequences in [0, 1) -/
theorem bolzano_weierstrass_unit_interval (s : ℕ → ℝ) (h : ∀ n, 0 ≤ s n ∧ s n < 1) :
  (∃ (a : Set ℕ), Set.Infinite a ∧ (∀ n ∈ a, s n < 1/2)) ∨
  (∃ (b : Set ℕ), Set.Infinite b ∧ (∀ n ∈ b, 1/2 ≤ s n)) ∧
  ∀ ε > 0, ε < 1/2 → ∃ α : ℝ, 0 ≤ α ∧ α ≤ 1 ∧
    ∃ (c : Set ℕ), Set.Infinite c ∧ ∀ n ∈ c, |s n - α| < ε :=
by sorry

end NUMINAMATH_CALUDE_bolzano_weierstrass_unit_interval_l2280_228061


namespace NUMINAMATH_CALUDE_alice_winning_condition_l2280_228011

/-- Game state representing the numbers on the board -/
structure GameState where
  numbers : List ℚ
  deriving Repr

/-- Player type -/
inductive Player
| Alice
| Bob
deriving Repr

/-- Result of the game -/
inductive GameResult
| AliceWins
| BobWins
deriving Repr

/-- Perform a move in the game -/
def makeMove (state : GameState) : GameState :=
  sorry

/-- Play the game with given parameters -/
def playGame (n : ℕ) (c : ℚ) (initialNumbers : List ℕ) : GameResult :=
  sorry

/-- Alice's winning condition -/
def aliceWins (c : ℚ) : Prop :=
  ∀ n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ ∀ initialNumbers : List ℕ,
    initialNumbers.length = n → (∃ x y : ℕ, x ∈ initialNumbers ∧ y ∈ initialNumbers ∧ x ≠ y) →
      playGame n c initialNumbers = GameResult.AliceWins

theorem alice_winning_condition (c : ℚ) :
  aliceWins c ↔ c ≥ (1/2 : ℚ) :=
  sorry

end NUMINAMATH_CALUDE_alice_winning_condition_l2280_228011


namespace NUMINAMATH_CALUDE_three_common_points_l2280_228022

def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0

def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

def is_common_point (x y : ℝ) : Prop := equation1 x y ∧ equation2 x y

def distinct_points (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem three_common_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_common_point p1.1 p1.2 ∧
    is_common_point p2.1 p2.2 ∧
    is_common_point p3.1 p3.2 ∧
    distinct_points p1 p2 ∧
    distinct_points p1 p3 ∧
    distinct_points p2 p3 ∧
    (∀ (p : ℝ × ℝ), is_common_point p.1 p.2 → p = p1 ∨ p = p2 ∨ p = p3) :=
by sorry

end NUMINAMATH_CALUDE_three_common_points_l2280_228022


namespace NUMINAMATH_CALUDE_distance_to_point_l2280_228002

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 2*y + 6

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The distance between two points in 2D space -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance between the center of the circle and (10, 3) is √68 -/
theorem distance_to_point : distance circle_center (10, 3) = Real.sqrt 68 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l2280_228002


namespace NUMINAMATH_CALUDE_part_one_part_two_l2280_228056

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f (-1) x ≥ 3} = {x : ℝ | x ≤ -1.5 ∨ x ≥ 1.5} := by sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, f a x ≥ 2) ↔ (a = 3 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2280_228056


namespace NUMINAMATH_CALUDE_chord_diameter_ratio_l2280_228047

/-- Given two concentric circles with radii R/2 and R, prove that if a chord of the larger circle
    is divided into three equal parts by the smaller circle, then the ratio of this chord to
    the diameter of the larger circle is 3√6/8. -/
theorem chord_diameter_ratio (R : ℝ) (h : R > 0) :
  ∃ (chord : ℝ), 
    (∃ (a : ℝ), chord = 3 * a ∧ 
      (∃ (x : ℝ), x^2 = 2 * a^2 ∧ x = R/2)) →
    chord / (2 * R) = 3 * Real.sqrt 6 / 8 := by
  sorry

end NUMINAMATH_CALUDE_chord_diameter_ratio_l2280_228047


namespace NUMINAMATH_CALUDE_curve_is_two_semicircles_l2280_228023

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  |x| - 1 = Real.sqrt (1 - (y - 1)^2)

-- Define a semicircle
def is_semicircle (center_x center_y radius : ℝ) (x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ x ≥ center_x

-- Theorem statement
theorem curve_is_two_semicircles :
  ∀ x y : ℝ, curve_equation x y ↔
    (is_semicircle 1 1 1 x y ∨ is_semicircle (-1) 1 1 x y) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_semicircles_l2280_228023


namespace NUMINAMATH_CALUDE_complement_union_original_equals_universe_l2280_228091

-- Define the universe set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set C as the complement of M in U
def C : Finset Nat := U \ M

-- Theorem statement
theorem complement_union_original_equals_universe :
  C ∪ M = U := by sorry

end NUMINAMATH_CALUDE_complement_union_original_equals_universe_l2280_228091


namespace NUMINAMATH_CALUDE_total_chapters_read_l2280_228049

def number_of_books : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : number_of_books * chapters_per_book = 384 := by
  sorry

end NUMINAMATH_CALUDE_total_chapters_read_l2280_228049


namespace NUMINAMATH_CALUDE_line_AB_equation_l2280_228069

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define points A and B
def A : ℝ × ℝ → Prop := λ p => C₁ p.1 p.2
def B : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the relation between OA and OB
def OB_eq_2OA (a b : ℝ × ℝ) : Prop := b.1 = 2 * a.1 ∧ b.2 = 2 * a.2

-- Theorem statement
theorem line_AB_equation (a b : ℝ × ℝ) (ha : A a) (hb : B b) (hab : OB_eq_2OA a b) :
  (b.2 - a.2) / (b.1 - a.1) = 1 ∨ (b.2 - a.2) / (b.1 - a.1) = -1 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l2280_228069


namespace NUMINAMATH_CALUDE_g_g_2_equals_263_l2280_228027

def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 1

theorem g_g_2_equals_263 : g (g 2) = 263 := by
  sorry

end NUMINAMATH_CALUDE_g_g_2_equals_263_l2280_228027


namespace NUMINAMATH_CALUDE_liquid_film_radius_l2280_228093

/-- Given a box with dimensions and a liquid that partially fills it, 
    calculate the radius of the circular film formed when poured on water. -/
theorem liquid_film_radius 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (fill_percentage : ℝ) 
  (film_thickness : ℝ) : 
  box_length = 5 → 
  box_width = 4 → 
  box_height = 10 → 
  fill_percentage = 0.8 → 
  film_thickness = 0.05 → 
  ∃ (r : ℝ), r = Real.sqrt (3200 / Real.pi) ∧ 
  r^2 * Real.pi * film_thickness = box_length * box_width * box_height * fill_percentage :=
by sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l2280_228093


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2280_228034

theorem arithmetic_expression_evaluation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2280_228034


namespace NUMINAMATH_CALUDE_macy_tokens_macy_tokens_exist_unique_l2280_228038

/-- Proves that Macy used 11 tokens given the conditions of the problem -/
theorem macy_tokens : ℕ → Prop := fun m =>
  let pitches_per_token : ℕ := 15
  let piper_tokens : ℕ := 17
  let macy_hits : ℕ := 50
  let piper_hits : ℕ := 55
  let total_misses : ℕ := 315
  m * pitches_per_token + piper_tokens * pitches_per_token = macy_hits + piper_hits + total_misses →
  m = 11

/-- Main theorem stating that there exists a unique number of tokens Macy used -/
theorem macy_tokens_exist_unique : ∃! m : ℕ, macy_tokens m := by
  sorry

end NUMINAMATH_CALUDE_macy_tokens_macy_tokens_exist_unique_l2280_228038


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2280_228005

theorem fractional_equation_solution :
  ∃! x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2280_228005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2280_228097

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 - sum_arithmetic_sequence a₁ d 10 / 10 = 2) →
  (sum_arithmetic_sequence (-2008) d 2008 = -2008) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2280_228097


namespace NUMINAMATH_CALUDE_grain_depot_analysis_l2280_228003

def grain_movements : List Int := [25, -31, -16, 33, -36, -20]
def fee_per_ton : ℕ := 5

theorem grain_depot_analysis :
  (List.sum grain_movements = -45) ∧
  (List.sum (List.map (λ x => fee_per_ton * x.natAbs) grain_movements) = 805) := by
  sorry

end NUMINAMATH_CALUDE_grain_depot_analysis_l2280_228003


namespace NUMINAMATH_CALUDE_digit_equation_sum_l2280_228035

/-- Represents a base-10 digit -/
def Digit := Fin 10

/-- Checks if all digits in a natural number are the same -/
def allDigitsSame (n : ℕ) : Prop :=
  ∃ d : Digit, n = d.val * 100 + d.val * 10 + d.val

/-- The main theorem -/
theorem digit_equation_sum :
  ∀ (Y E M L : Digit),
    Y ≠ E → Y ≠ M → Y ≠ L → E ≠ M → E ≠ L → M ≠ L →
    (Y.val * 10 + E.val) * (M.val * 10 + E.val) = L.val * 100 + L.val * 10 + L.val →
    E.val + M.val + L.val + Y.val = 15 := by
  sorry


end NUMINAMATH_CALUDE_digit_equation_sum_l2280_228035


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l2280_228076

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l2280_228076


namespace NUMINAMATH_CALUDE_contestant_paths_count_l2280_228087

/-- Represents the diamond-shaped grid for the word "CONTESTANT" -/
def ContestantGrid : Type := Unit  -- placeholder for the actual grid structure

/-- Represents a valid path in the grid -/
def ValidPath (grid : ContestantGrid) : Type := Unit  -- placeholder for the actual path structure

/-- The number of valid paths in the grid -/
def numValidPaths (grid : ContestantGrid) : ℕ := sorry

/-- The theorem stating that the number of valid paths is 256 -/
theorem contestant_paths_count (grid : ContestantGrid) : numValidPaths grid = 256 := by
  sorry

end NUMINAMATH_CALUDE_contestant_paths_count_l2280_228087


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2280_228089

/-- The perimeter of pentagon ABCDE with given side lengths -/
theorem pentagon_perimeter (AB BC CD DE AE : ℝ) : 
  AB = 1 → BC = Real.sqrt 3 → CD = 2 → DE = Real.sqrt 5 → AE = Real.sqrt 13 →
  AB + BC + CD + DE + AE = 3 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l2280_228089


namespace NUMINAMATH_CALUDE_product_ab_l2280_228068

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l2280_228068


namespace NUMINAMATH_CALUDE_weekly_calorie_allowance_l2280_228001

/-- The number of calories to reduce from the average daily allowance to hypothetically live to 100 years old -/
def calorie_reduction : ℕ := 500

/-- The average daily calorie allowance for a person in their 60's -/
def average_daily_allowance : ℕ := 2000

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance for a person in their 60's to hypothetically live to 100 years old -/
theorem weekly_calorie_allowance :
  (average_daily_allowance - calorie_reduction) * days_in_week = 10500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_calorie_allowance_l2280_228001


namespace NUMINAMATH_CALUDE_mall_a_better_deal_l2280_228060

/-- Calculates the discount for a given spent amount and promotion rule -/
def calculate_discount (spent : ℕ) (promotion_threshold : ℕ) (promotion_discount : ℕ) : ℕ :=
  (spent / promotion_threshold) * promotion_discount

/-- Calculates the final cost after applying the discount -/
def calculate_final_cost (total : ℕ) (discount : ℕ) : ℕ :=
  total - discount

theorem mall_a_better_deal (shoes_price : ℕ) (sweater_price : ℕ)
    (h_shoes : shoes_price = 699)
    (h_sweater : sweater_price = 910)
    (mall_a_threshold : ℕ) (mall_a_discount : ℕ)
    (mall_b_threshold : ℕ) (mall_b_discount : ℕ)
    (h_mall_a : mall_a_threshold = 200 ∧ mall_a_discount = 101)
    (h_mall_b : mall_b_threshold = 101 ∧ mall_b_discount = 50) :
    let total := shoes_price + sweater_price
    let discount_a := calculate_discount total mall_a_threshold mall_a_discount
    let discount_b := calculate_discount total mall_b_threshold mall_b_discount
    let final_cost_a := calculate_final_cost total discount_a
    let final_cost_b := calculate_final_cost total discount_b
    final_cost_a < final_cost_b ∧ final_cost_a = 801 := by
  sorry

end NUMINAMATH_CALUDE_mall_a_better_deal_l2280_228060


namespace NUMINAMATH_CALUDE_distribute_4_2_l2280_228028

/-- The number of ways to distribute n indistinguishable objects into k indistinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_4_2 : distribute 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_2_l2280_228028


namespace NUMINAMATH_CALUDE_initial_plus_bought_equals_total_l2280_228044

/-- The number of bottle caps William had initially -/
def initial_caps : ℕ := 2

/-- The number of bottle caps William bought -/
def bought_caps : ℕ := 41

/-- The total number of bottle caps William has after buying more -/
def total_caps : ℕ := 43

/-- Theorem stating that the initial number of bottle caps plus the bought ones equals the total -/
theorem initial_plus_bought_equals_total : 
  initial_caps + bought_caps = total_caps := by sorry

end NUMINAMATH_CALUDE_initial_plus_bought_equals_total_l2280_228044


namespace NUMINAMATH_CALUDE_bowling_balls_count_l2280_228009

theorem bowling_balls_count (red : ℕ) (green : ℕ) : 
  green = red + 6 →
  red + green = 66 →
  red = 30 := by
sorry

end NUMINAMATH_CALUDE_bowling_balls_count_l2280_228009


namespace NUMINAMATH_CALUDE_math_team_combinations_l2280_228000

theorem math_team_combinations : 
  let total_girls : ℕ := 4
  let total_boys : ℕ := 6
  let girls_on_team : ℕ := 3
  let boys_on_team : ℕ := 2
  (total_girls.choose girls_on_team) * (total_boys.choose boys_on_team) = 60 := by
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l2280_228000


namespace NUMINAMATH_CALUDE_polygon_exterior_angle_pairs_l2280_228090

def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 360 / m = n ∧ 360 / n = m

theorem polygon_exterior_angle_pairs :
  ∃! (S : Finset (ℕ × ℕ)), S.card = 20 ∧ ∀ p : ℕ × ℕ, p ∈ S ↔ is_valid_pair p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angle_pairs_l2280_228090


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_three_composite_reciprocals_l2280_228080

/-- The arithmetic mean of the reciprocals of the first three composite numbers is 13/72. -/
theorem arithmetic_mean_of_first_three_composite_reciprocals :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = 13 / 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_three_composite_reciprocals_l2280_228080


namespace NUMINAMATH_CALUDE_work_completion_l2280_228048

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 > 0)
  (h2 : days2 > 0)
  (h3 : men2 > 0)
  (h4 : days1 = 80)
  (h5 : days2 = 48)
  (h6 : men2 = 20)
  (h7 : ∃ men1 : ℕ, men1 * days1 = men2 * days2) :
  ∃ men1 : ℕ, men1 = 12 ∧ men1 * days1 = men2 * days2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l2280_228048


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2280_228004

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (x₁^2 + 3*x₁ + m = 0) →
  (x₂^2 + 3*x₂ + m = 0) →
  (1/x₁ + 1/x₂ = 1) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2280_228004


namespace NUMINAMATH_CALUDE_sets_problem_l2280_228077

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≤ 2, y = 3^x - 2*a}

-- State the theorem
theorem sets_problem (a : ℝ) :
  (a = 3 → A a ∪ B a = Set.Ioo (-6) 5) ∧
  (A a ∩ B a = A a ↔ a = -1 ∨ (0 ≤ a ∧ a ≤ 7/3)) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l2280_228077


namespace NUMINAMATH_CALUDE_min_value_problem_l2280_228050

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / (x + 2)) + (1 / (y + 1)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2280_228050


namespace NUMINAMATH_CALUDE_cube_edge_length_l2280_228073

/-- Given three cubes with edge lengths 6, 10, and x, when melted together to form a new cube
    with edge length 12, prove that x = 8 -/
theorem cube_edge_length (x : ℝ) : x > 0 → 6^3 + 10^3 + x^3 = 12^3 → x = 8 := by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2280_228073


namespace NUMINAMATH_CALUDE_diamond_commutative_eq_four_lines_l2280_228094

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^3 * b - a * b^3

/-- The set of points (x, y) where x ◇ y = y ◇ x -/
def diamond_commutative_set : Set (ℝ × ℝ) :=
  {p | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of four lines: x = 0, y = 0, y = x, and y = -x -/
def four_lines : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_commutative_eq_four_lines :
  diamond_commutative_set = four_lines := by sorry

end NUMINAMATH_CALUDE_diamond_commutative_eq_four_lines_l2280_228094


namespace NUMINAMATH_CALUDE_days_missed_by_mike_and_sarah_l2280_228074

/-- Given the number of days missed by Vanessa, Mike, and Sarah, prove that Mike and Sarah missed 12 days together. -/
theorem days_missed_by_mike_and_sarah
  (total_days : ℕ)
  (vanessa_mike_days : ℕ)
  (vanessa_days : ℕ)
  (h1 : total_days = 17)
  (h2 : vanessa_mike_days = 14)
  (h3 : vanessa_days = 5)
  : ∃ (mike_days sarah_days : ℕ),
    mike_days + sarah_days = 12 ∧
    vanessa_days + mike_days + sarah_days = total_days ∧
    vanessa_days + mike_days = vanessa_mike_days :=
by
  sorry


end NUMINAMATH_CALUDE_days_missed_by_mike_and_sarah_l2280_228074


namespace NUMINAMATH_CALUDE_final_amount_after_bets_l2280_228020

theorem final_amount_after_bets (initial_amount : ℝ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let win_factor := (3/2 : ℝ)
  let loss_factor := (1/2 : ℝ)
  let final_factor := win_factor ^ num_wins * loss_factor ^ num_losses
  initial_amount * final_factor = 27 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_bets_l2280_228020


namespace NUMINAMATH_CALUDE_increase_by_fraction_l2280_228036

theorem increase_by_fraction (initial : ℝ) (increase : ℚ) (result : ℝ) :
  initial = 120 →
  increase = 5/6 →
  result = initial * (1 + increase) →
  result = 220 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_fraction_l2280_228036


namespace NUMINAMATH_CALUDE_concrete_mixture_theorem_l2280_228099

/-- The amount of 80% cement mixture used in tons -/
def amount_80_percent : ℝ := 7.0

/-- The percentage of cement in the final mixture -/
def final_cement_percentage : ℝ := 0.62

/-- The percentage of cement in the first mixture -/
def first_mixture_percentage : ℝ := 0.20

/-- The percentage of cement in the second mixture -/
def second_mixture_percentage : ℝ := 0.80

/-- The total amount of concrete made in tons -/
def total_concrete : ℝ := 10.0

theorem concrete_mixture_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x * first_mixture_percentage + amount_80_percent * second_mixture_percentage =
      final_cement_percentage * (x + amount_80_percent) ∧
    x + amount_80_percent = total_concrete :=
by sorry

end NUMINAMATH_CALUDE_concrete_mixture_theorem_l2280_228099


namespace NUMINAMATH_CALUDE_triangle_properties_l2280_228085

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the maximum area of the triangle and the ratio of tangents -/
theorem triangle_properties (t : Triangle) (h1 : t.c = 3) :
  (t.C = 2 * Real.pi / 3 → ∃ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4 ∧ 
    ∀ (other_area : ℝ), other_area ≤ area) ∧
  (Real.cos t.B = 1 / t.a → Real.tan t.B / Real.tan t.A = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2280_228085


namespace NUMINAMATH_CALUDE_quaternary_to_decimal_10231_l2280_228063

/-- Converts a quaternary (base 4) number to its decimal equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, digit) acc => acc + digit * (4 ^ i)) 0

/-- The quaternary representation of the number -/
def quaternary_num : List Nat := [1, 3, 2, 0, 1]

theorem quaternary_to_decimal_10231 :
  quaternary_to_decimal quaternary_num = 301 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_to_decimal_10231_l2280_228063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2280_228007

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  arithmetic_sequence a d →
  a 1 = 0 →
  d ≠ 0 →
  a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) →
  m = 37 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2280_228007


namespace NUMINAMATH_CALUDE_scout_sunday_deliveries_l2280_228043

def base_pay : ℝ := 10
def tip_per_customer : ℝ := 5
def saturday_hours : ℝ := 4
def saturday_customers : ℝ := 5
def sunday_hours : ℝ := 5
def total_earnings : ℝ := 155

theorem scout_sunday_deliveries :
  ∃ (sunday_customers : ℝ),
    base_pay * (saturday_hours + sunday_hours) +
    tip_per_customer * (saturday_customers + sunday_customers) = total_earnings ∧
    sunday_customers = 8 := by
  sorry

end NUMINAMATH_CALUDE_scout_sunday_deliveries_l2280_228043


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2280_228081

/-- An arithmetic sequence with first term 6 and the sum of the 3rd and 5th terms equal to 0 -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 6 ∧ 
  a 3 + a 5 = 0 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 8 - 2 * n

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) : GeneralTermFormula a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2280_228081


namespace NUMINAMATH_CALUDE_thabos_books_l2280_228095

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 160 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 25

theorem thabos_books (books : BookCollection) (h : validCollection books) :
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabos_books_l2280_228095


namespace NUMINAMATH_CALUDE_carnation_count_l2280_228058

theorem carnation_count (vase_capacity : ℕ) (rose_count : ℕ) (vase_count : ℕ) :
  vase_capacity = 6 →
  rose_count = 47 →
  vase_count = 9 →
  vase_count * vase_capacity - rose_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_carnation_count_l2280_228058


namespace NUMINAMATH_CALUDE_barn_paint_area_l2280_228041

/-- Calculates the total area to be painted for a barn with given dimensions and conditions -/
def total_area_to_paint (length width height : ℝ) (door_width door_height : ℝ) : ℝ :=
  let wall_area := 2 * (width * height) * 2 + (length * height) * 2
  let roof_area := length * width
  let ceiling_area := length * width
  wall_area + roof_area + ceiling_area

/-- Theorem stating that the total area to be painted for the given barn is 860 square yards -/
theorem barn_paint_area :
  total_area_to_paint 15 10 8 2 3 = 860 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l2280_228041


namespace NUMINAMATH_CALUDE_well_digging_payment_l2280_228015

/-- The total amount paid to two workers for digging a well --/
def total_amount_paid (hours_day1 hours_day2 hours_day3 : ℕ) (hourly_rate : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := 2 * total_hours
  total_man_hours * hourly_rate

/-- Theorem stating that the total amount paid is $660 --/
theorem well_digging_payment :
  total_amount_paid 10 8 15 10 = 660 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l2280_228015


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l2280_228075

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) :
  total_students = 65 →
  not_picked = 17 →
  num_groups = 8 →
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l2280_228075


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l2280_228066

/-- Represents a rectangular garden with length and breadth -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 1800)
  (h2 : garden.length = 500) :
  garden.breadth = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l2280_228066


namespace NUMINAMATH_CALUDE_original_price_after_discount_l2280_228053

/-- Given a product with an unknown original price that becomes 50 yuan cheaper after a 20% discount, prove that its original price is 250 yuan. -/
theorem original_price_after_discount (price : ℝ) : 
  price * (1 - 0.2) = price - 50 → price = 250 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l2280_228053


namespace NUMINAMATH_CALUDE_population_growth_l2280_228079

theorem population_growth (p q : ℕ) (h1 : p^2 + 180 = q^2 + 16) 
  (h2 : ∃ r : ℕ, p^2 + 360 = r^2) : 
  abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 21) < 
  min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 18))
      (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 24))
           (min (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 27))
                (abs (((p^2 + 360 : ℝ) / p^2 - 1) * 100 - 30)))) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l2280_228079


namespace NUMINAMATH_CALUDE_matrix_determinant_l2280_228078

/-- The determinant of the matrix [[x + 2, x, x+1], [x, x + 3, x], [x+1, x, x + 4]] is 6x^2 + 36x + 48 -/
theorem matrix_determinant (x : ℝ) : 
  Matrix.det !![x + 2, x, x + 1; x, x + 3, x; x + 1, x, x + 4] = 6*x^2 + 36*x + 48 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2280_228078


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2280_228026

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 5^(5^n) + 6

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 : units_digit (G 1000) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2280_228026


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l2280_228032

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 7)) ↔ x ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l2280_228032


namespace NUMINAMATH_CALUDE_z_purely_imaginary_z_in_second_quadrant_l2280_228013

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 3*m - 2) (m^2 - 3*m + 2)

/-- Theorem: z is purely imaginary if and only if m = -1/2 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = -1/2 := by sorry

/-- Theorem: z is in the second quadrant if and only if -1/2 < m < 1 -/
theorem z_in_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_z_in_second_quadrant_l2280_228013


namespace NUMINAMATH_CALUDE_carrie_shirts_l2280_228062

/-- The number of shirts Carrie bought -/
def num_shirts : ℕ := sorry

/-- The cost of each shirt -/
def shirt_cost : ℕ := 8

/-- The number of pairs of pants Carrie bought -/
def num_pants : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 18

/-- The number of jackets Carrie bought -/
def num_jackets : ℕ := 2

/-- The cost of each jacket -/
def jacket_cost : ℕ := 60

/-- The amount Carrie paid for her share of the clothes -/
def carrie_payment : ℕ := 94

theorem carrie_shirts : 
  num_shirts * shirt_cost + num_pants * pants_cost + num_jackets * jacket_cost = 2 * carrie_payment ∧
  num_shirts = 4 := by sorry

end NUMINAMATH_CALUDE_carrie_shirts_l2280_228062


namespace NUMINAMATH_CALUDE_common_factor_of_polynomials_l2280_228052

theorem common_factor_of_polynomials (m : ℝ) : 
  ∃ (k₁ k₂ k₃ : ℝ → ℝ), 
    (m * (m - 3) + 2 * (3 - m) = (m - 2) * k₁ m) ∧
    (m^2 - 4*m + 4 = (m - 2) * k₂ m) ∧
    (m^4 - 16 = (m - 2) * k₃ m) := by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomials_l2280_228052


namespace NUMINAMATH_CALUDE_fishmonger_salmon_sales_l2280_228084

/-- Given a fishmonger who sells salmon over two weeks, this theorem proves
    that if the amount sold in the second week is three times the first week,
    and the total amount sold is 200 kg, then the amount sold in the first week is 50 kg. -/
theorem fishmonger_salmon_sales (x : ℝ) 
  (h1 : x ≥ 0)  -- Non-negative sales
  (h2 : x + 3*x = 200) :  -- Total sales over two weeks
  x = 50 := by
  sorry

end NUMINAMATH_CALUDE_fishmonger_salmon_sales_l2280_228084


namespace NUMINAMATH_CALUDE_third_side_length_l2280_228016

theorem third_side_length (a b : ℝ) (θ : ℝ) (ha : a = 11) (hb : b = 15) (hθ : θ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos θ ∧ c = Real.sqrt (346 + 165 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_third_side_length_l2280_228016


namespace NUMINAMATH_CALUDE_two_digit_sum_problem_l2280_228046

/-- Given two-digit numbers ab and cd, and a three-digit number jjj,
    where a, b, c, and d are distinct positive integers,
    c = 9, and ab + cd = jjj, prove that cd = 98. -/
theorem two_digit_sum_problem (ab cd jjj : ℕ) (a b c d : ℕ) : 
  (10 ≤ ab) ∧ (ab < 100) →  -- ab is a two-digit number
  (10 ≤ cd) ∧ (cd < 100) →  -- cd is a two-digit number
  (100 ≤ jjj) ∧ (jjj < 1000) →  -- jjj is a three-digit number
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- a, b, c, d are distinct
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- a, b, c, d are positive
  c = 9 →  -- given condition
  ab + cd = jjj →  -- sum equation
  cd = 98 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_problem_l2280_228046


namespace NUMINAMATH_CALUDE_special_triangle_property_l2280_228088

/-- Triangle with given side, inscribed circle radius, and excircle radius -/
structure SpecialTriangle where
  -- Side length
  a : ℝ
  -- Inscribed circle radius
  r : ℝ
  -- Excircle radius
  r_b : ℝ
  -- Assumption that all values are positive
  a_pos : 0 < a
  r_pos : 0 < r
  r_b_pos : 0 < r_b

/-- Theorem stating the relationship between side length, semiperimeter, and tangent length -/
theorem special_triangle_property (t : SpecialTriangle) :
  ∃ (p : ℝ) (tangent_length : ℝ),
    -- Semiperimeter is positive
    0 < p ∧
    -- Tangent length is positive and less than semiperimeter
    0 < tangent_length ∧ tangent_length < p ∧
    -- The given side length equals semiperimeter minus tangent length
    t.a = p - tangent_length :=
  sorry

end NUMINAMATH_CALUDE_special_triangle_property_l2280_228088


namespace NUMINAMATH_CALUDE_initial_tomatoes_correct_l2280_228029

/-- Represents the initial number of tomatoes in the garden -/
def initial_tomatoes : ℕ := 175

/-- Represents the initial number of potatoes in the garden -/
def initial_potatoes : ℕ := 77

/-- Represents the number of potatoes picked -/
def picked_potatoes : ℕ := 172

/-- Represents the total number of tomatoes and potatoes left after picking -/
def remaining_total : ℕ := 80

/-- Theorem stating that the initial number of tomatoes is correct given the conditions -/
theorem initial_tomatoes_correct : 
  initial_tomatoes + initial_potatoes - picked_potatoes = remaining_total :=
by sorry


end NUMINAMATH_CALUDE_initial_tomatoes_correct_l2280_228029


namespace NUMINAMATH_CALUDE_initial_fish_count_l2280_228014

def fish_eaten_per_day : ℕ := 2
def days_before_adding : ℕ := 14
def fish_added : ℕ := 8
def days_after_adding : ℕ := 7
def final_fish_count : ℕ := 26

theorem initial_fish_count (initial_count : ℕ) : 
  initial_count - (fish_eaten_per_day * days_before_adding) + fish_added - 
  (fish_eaten_per_day * days_after_adding) = final_fish_count → 
  initial_count = 60 := by
sorry

end NUMINAMATH_CALUDE_initial_fish_count_l2280_228014


namespace NUMINAMATH_CALUDE_additive_function_negative_on_positive_properties_l2280_228092

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y and f(x) < 0 for x > 0 -/
def AdditiveFunctionNegativeOnPositive (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧ (∀ x, x > 0 → f x < 0)

/-- Theorem stating that such a function is odd and monotonically decreasing -/
theorem additive_function_negative_on_positive_properties
    (f : ℝ → ℝ) (h : AdditiveFunctionNegativeOnPositive f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ > x₂ → f x₁ < f x₂) := by
  sorry


end NUMINAMATH_CALUDE_additive_function_negative_on_positive_properties_l2280_228092


namespace NUMINAMATH_CALUDE_max_a_fourth_quadrant_l2280_228008

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) → a ≤ 3 ∧ ∃ (a : ℤ), a = 3 ∧ 
    let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
    (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_max_a_fourth_quadrant_l2280_228008


namespace NUMINAMATH_CALUDE_a_share_is_240_l2280_228086

/-- Calculates the share of profit for partner A given the initial investments,
    changes in investment, and total profit. -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) 
                      (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

/-- Theorem stating that given the problem conditions, A's share of the profit is 240. -/
theorem a_share_is_240 : 
  calculate_share_a 3000 4000 1000 1000 12 8 630 = 240 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_240_l2280_228086


namespace NUMINAMATH_CALUDE_greatest_power_less_than_500_l2280_228067

theorem greatest_power_less_than_500 (c d : ℕ+) (h1 : d > 1) 
  (h2 : c^(d:ℕ) < 500) 
  (h3 : ∀ (x y : ℕ+), y > 1 → x^(y:ℕ) < 500 → x^(y:ℕ) ≤ c^(d:ℕ)) : 
  c + d = 24 := by sorry

end NUMINAMATH_CALUDE_greatest_power_less_than_500_l2280_228067


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2280_228018

theorem least_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, 4 * y^2 + 8 * y + 3 = 1 → y ≥ -1) ∧ 
  (4 * (-1)^2 + 8 * (-1) + 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2280_228018


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l2280_228017

def total_marbles : ℕ := 15
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 1
def other_marbles : ℕ := 11
def marbles_to_choose : ℕ := 5

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_theorem :
  (choose_marbles 3 1 * choose_marbles 11 4) +
  (choose_marbles 3 2 * choose_marbles 11 3) +
  (choose_marbles 3 3 * choose_marbles 11 2) = 1540 := by
  sorry

#check marble_selection_theorem

end NUMINAMATH_CALUDE_marble_selection_theorem_l2280_228017


namespace NUMINAMATH_CALUDE_milk_left_over_calculation_l2280_228059

/-- The amount of milk left over given the following conditions:
  - Total milk production is 24 cups per day
  - 80% of milk is consumed by Daisy's kids
  - 60% of remaining milk is used for cooking
  - 25% of remaining milk is given to neighbor
  - 6% of remaining milk is drunk by Daisy's husband
-/
def milk_left_over (total_milk : ℝ) (kids_consumption : ℝ) (cooking_usage : ℝ)
  (neighbor_share : ℝ) (husband_consumption : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption)
  let remaining_after_cooking := remaining_after_kids * (1 - cooking_usage)
  let remaining_after_neighbor := remaining_after_cooking * (1 - neighbor_share)
  remaining_after_neighbor * (1 - husband_consumption)

theorem milk_left_over_calculation :
  milk_left_over 24 0.8 0.6 0.25 0.06 = 1.3536 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_over_calculation_l2280_228059


namespace NUMINAMATH_CALUDE_genetic_material_not_equal_l2280_228072

/-- Represents a cell involved in fertilization -/
structure Cell where
  nucleus : Bool
  cytoplasm : Nat

/-- Represents the process of fertilization -/
def fertilization (sperm : Cell) (egg : Cell) : Prop :=
  sperm.nucleus ∧ egg.nucleus ∧ sperm.cytoplasm < egg.cytoplasm

/-- Represents the zygote formed after fertilization -/
def zygote (sperm : Cell) (egg : Cell) : Prop :=
  fertilization sperm egg

/-- Theorem stating that genetic material in the zygote does not come equally from both parents -/
theorem genetic_material_not_equal (sperm egg : Cell) 
  (h_sperm : sperm.nucleus ∧ sperm.cytoplasm = 0)
  (h_egg : egg.nucleus ∧ egg.cytoplasm > 0)
  (h_zygote : zygote sperm egg) :
  ¬(∃ (x : Nat), x > 0 ∧ x = sperm.cytoplasm ∧ x = egg.cytoplasm) := by
  sorry


end NUMINAMATH_CALUDE_genetic_material_not_equal_l2280_228072


namespace NUMINAMATH_CALUDE_least_negative_b_for_integer_solutions_l2280_228057

theorem least_negative_b_for_integer_solutions (x b : ℤ) : 
  (∃ x : ℤ, x^2 + b*x = 22) → 
  b < 0 → 
  (∀ b' : ℤ, b' < b → ¬∃ x : ℤ, x^2 + b'*x = 22) →
  b = -21 :=
by sorry

end NUMINAMATH_CALUDE_least_negative_b_for_integer_solutions_l2280_228057


namespace NUMINAMATH_CALUDE_acute_angles_sum_l2280_228098

theorem acute_angles_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l2280_228098


namespace NUMINAMATH_CALUDE_product_expansion_l2280_228045

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x) - 5 * x^3) = 3 / x - (15 / 7) * x^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2280_228045


namespace NUMINAMATH_CALUDE_certain_number_equation_l2280_228021

theorem certain_number_equation : ∃ x : ℚ, (55 / 100) * 40 = (4 / 5) * x + 2 :=
by
  -- Proof goes here
  sorry

#check certain_number_equation

end NUMINAMATH_CALUDE_certain_number_equation_l2280_228021


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l2280_228083

/-- Given 20 carrots, where 16 of them have an average weight of 180 grams
    and 4 of them have an average weight of 190 grams, 
    the total weight of all 20 carrots is 3.64 kg. -/
theorem carrot_weight_problem :
  let total_carrots : ℕ := 20
  let remaining_carrots : ℕ := 16
  let removed_carrots : ℕ := 4
  let avg_weight_remaining : ℝ := 180 -- in grams
  let avg_weight_removed : ℝ := 190 -- in grams
  let total_weight_kg : ℝ := 3.64 -- in kg
  total_carrots = remaining_carrots + removed_carrots →
  (remaining_carrots : ℝ) * avg_weight_remaining + (removed_carrots : ℝ) * avg_weight_removed = total_weight_kg * 1000 := by
  sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l2280_228083


namespace NUMINAMATH_CALUDE_community_age_is_35_l2280_228030

/-- Represents the average age of a community given specific demographic information. -/
def community_average_age (women_ratio men_ratio : ℚ) (women_avg_age men_avg_age children_avg_age : ℚ) (children_ratio : ℚ) : ℚ :=
  let total_population := women_ratio + men_ratio + children_ratio * men_ratio
  let total_age := women_ratio * women_avg_age + men_ratio * men_avg_age + children_ratio * men_ratio * children_avg_age
  total_age / total_population

/-- Theorem stating that the average age of the community is 35 years given the specified conditions. -/
theorem community_age_is_35 :
  community_average_age 3 2 40 36 10 (1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_community_age_is_35_l2280_228030


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2280_228040

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expansion function
def expansion_term (x : ℚ) (r k : ℕ) : ℚ :=
  (-1)^k * binomial r k * x^(r - 2*k)

-- Define the constant term of the expansion
def constant_term : ℚ :=
  1 - binomial 2 1 * binomial 4 2 + binomial 4 2 * binomial 0 0

-- Theorem statement
theorem constant_term_expansion :
  constant_term = -5 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2280_228040


namespace NUMINAMATH_CALUDE_group_size_proof_l2280_228042

/-- The number of people in a group where:
    1. The total weight increase is 2.5 kg times the number of people.
    2. The weight difference between the new person and the replaced person is 20 kg. -/
def number_of_people : ℕ := 8

theorem group_size_proof :
  ∃ (n : ℕ), n = number_of_people ∧ 
  (2.5 : ℝ) * n = (20 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2280_228042


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l2280_228070

theorem meaningful_sqrt_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l2280_228070
