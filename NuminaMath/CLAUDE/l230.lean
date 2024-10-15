import Mathlib

namespace NUMINAMATH_CALUDE_stating_max_s_value_l230_23013

/-- Represents the dimensions of the large rectangle to be tiled -/
def large_rectangle : ℕ × ℕ := (1993, 2000)

/-- Represents the area of a 2 × 2 square -/
def square_area : ℕ := 4

/-- Represents the area of a P-rectangle -/
def p_rectangle_area : ℕ := 5

/-- Represents the area of an S-rectangle -/
def s_rectangle_area : ℕ := 4

/-- Represents the total area of the large rectangle -/
def total_area : ℕ := large_rectangle.1 * large_rectangle.2

/-- 
Theorem stating that the maximum value of s (sum of 2 × 2 squares and S-rectangles) 
used to tile the large rectangle is 996500
-/
theorem max_s_value : 
  ∀ a b c : ℕ, 
  a * square_area + b * p_rectangle_area + c * s_rectangle_area = total_area →
  a + c ≤ 996500 :=
sorry

end NUMINAMATH_CALUDE_stating_max_s_value_l230_23013


namespace NUMINAMATH_CALUDE_tenth_element_is_6785_l230_23002

/-- A list of all four-digit integers using digits 5, 6, 7, and 8 exactly once, ordered from least to greatest -/
def fourDigitList : List Nat := sorry

/-- The 10th element in the fourDigitList -/
def tenthElement : Nat := sorry

/-- Theorem stating that the 10th element in the fourDigitList is 6785 -/
theorem tenth_element_is_6785 : tenthElement = 6785 := by sorry

end NUMINAMATH_CALUDE_tenth_element_is_6785_l230_23002


namespace NUMINAMATH_CALUDE_chromatic_number_bound_l230_23041

/-- A graph G is represented by its vertex set and edge set. -/
structure Graph (V : Type) where
  edges : Set (V × V)

/-- The chromatic number of a graph. -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- The number of edges in a graph. -/
def numEdges {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a graph is bounded by a function of its edge count. -/
theorem chromatic_number_bound {V : Type} (G : Graph V) :
  (chromaticNumber G : ℝ) ≤ 1/2 + Real.sqrt (2 * (numEdges G : ℝ) + 1/4) :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_bound_l230_23041


namespace NUMINAMATH_CALUDE_gcd_455_299_l230_23044

theorem gcd_455_299 : Nat.gcd 455 299 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_455_299_l230_23044


namespace NUMINAMATH_CALUDE_annie_crayons_l230_23078

theorem annie_crayons (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 36 → final = 40 → initial = 4 := by sorry

end NUMINAMATH_CALUDE_annie_crayons_l230_23078


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l230_23043

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l230_23043


namespace NUMINAMATH_CALUDE_geometric_series_sum_l230_23034

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 2

/-- Common ratio of the geometric series -/
def r : ℚ := 2/5

/-- Number of terms in the series -/
def n : ℕ := 5

theorem geometric_series_sum :
  geometric_sum a r n = 2062/375 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l230_23034


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l230_23088

/-- Represents the Mountain Loop Trail hike --/
structure MountainLoopTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike --/
def validHike (hike : MountainLoopTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 14 ∧
  hike.day4 + hike.day5 = 36 ∧
  hike.day1 + hike.day3 = 30

/-- The theorem stating the total length of the trail --/
theorem mountain_loop_trail_length (hike : MountainLoopTrail) 
  (h : validHike hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end NUMINAMATH_CALUDE_mountain_loop_trail_length_l230_23088


namespace NUMINAMATH_CALUDE_factorial_sum_equals_35906_l230_23090

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_35906 : 
  7 * factorial 7 + 5 * factorial 5 + 3 * factorial 3 + 2 * (factorial 2)^2 = 35906 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_35906_l230_23090


namespace NUMINAMATH_CALUDE_vegetable_production_equation_l230_23023

def vegetable_growth_rate (initial_production final_production : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_production * (1 + x) ^ years = final_production

theorem vegetable_production_equation :
  ∃ x : ℝ, vegetable_growth_rate 800 968 2 x :=
sorry

end NUMINAMATH_CALUDE_vegetable_production_equation_l230_23023


namespace NUMINAMATH_CALUDE_largest_base4_3digit_decimal_l230_23004

/-- The largest three-digit number in base-4 -/
def largest_base4_3digit : ℕ := 3 * 4^2 + 3 * 4 + 3

/-- Conversion from base-4 to base-10 -/
def base4_to_decimal (n : ℕ) : ℕ := n

theorem largest_base4_3digit_decimal :
  base4_to_decimal largest_base4_3digit = 63 := by sorry

end NUMINAMATH_CALUDE_largest_base4_3digit_decimal_l230_23004


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l230_23039

theorem ordered_pair_solution :
  ∀ x y : ℝ,
  (x + y = (6 - x) + (6 - y)) →
  (x - y = (x - 2) + (y - 2)) →
  (x = 2 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l230_23039


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l230_23009

/-- For any real number a, the inequality x^2 - 2(a-2)x + a > 0 holds for all x ∈ (-∞, 1) ∪ (5, +∞) if and only if a ∈ (1, 5]. -/
theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ (1 < a ∧ a ≤ 5) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l230_23009


namespace NUMINAMATH_CALUDE_triangle_inequality_l230_23069

/-- Given two triangles ABC and A₁B₁C₁, where b₁ and c₁ have areas S and S₁ respectively,
    prove the inequality and its equality condition. -/
theorem triangle_inequality (a b c a₁ b₁ c₁ S S₁ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧ S > 0 ∧ S₁ > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ →
  a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) ≥ 16 * S * S₁ ∧
  (a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) = 16 * S * S₁ ↔
   ∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l230_23069


namespace NUMINAMATH_CALUDE_sixteen_triangles_l230_23091

/-- A right triangle with integer leg lengths and hypotenuse b + 3 --/
structure RightTriangle where
  a : ℕ
  b : ℕ
  hyp_eq : a^2 + b^2 = (b + 3)^2
  b_bound : b < 200

/-- The count of right triangles satisfying the given conditions --/
def count_triangles : ℕ := sorry

/-- Theorem stating that there are exactly 16 right triangles satisfying the conditions --/
theorem sixteen_triangles : count_triangles = 16 := by sorry

end NUMINAMATH_CALUDE_sixteen_triangles_l230_23091


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_101st_term_l230_23022

/-- An arithmetic sequence where the square of each term equals the sum of the first 2n-1 terms. -/
def SpecialArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ n, (a n)^2 = (2 * n - 1) * (a 1 + a n) / 2)

theorem special_arithmetic_sequence_101st_term
  (a : ℕ → ℝ) (h : SpecialArithmeticSequence a) : a 101 = 201 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_101st_term_l230_23022


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l230_23018

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_two_l230_23018


namespace NUMINAMATH_CALUDE_castle_entry_exit_ways_l230_23005

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the castle through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: Given a castle with 8 windows, the number of ways to enter through
    one window and exit through a different window is 56 -/
theorem castle_entry_exit_ways :
  num_windows = 8 → num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_castle_entry_exit_ways_l230_23005


namespace NUMINAMATH_CALUDE_only_pyramid_volume_unconditional_l230_23007

/-- Represents an algorithm --/
inductive Algorithm
  | triangleArea
  | lineSlope
  | commonLogarithm
  | pyramidVolume

/-- Predicate to check if an algorithm requires conditional statements --/
def requiresConditionalStatements (a : Algorithm) : Prop :=
  match a with
  | .triangleArea => true
  | .lineSlope => true
  | .commonLogarithm => true
  | .pyramidVolume => false

/-- Theorem stating that only the pyramid volume algorithm doesn't require conditional statements --/
theorem only_pyramid_volume_unconditional :
    ∀ (a : Algorithm), ¬(requiresConditionalStatements a) ↔ a = Algorithm.pyramidVolume := by
  sorry


end NUMINAMATH_CALUDE_only_pyramid_volume_unconditional_l230_23007


namespace NUMINAMATH_CALUDE_gas_needed_is_eighteen_l230_23032

/-- Calculates the total amount of gas needed to fill both a truck and car tank completely. -/
def total_gas_needed (truck_capacity car_capacity : ℚ) (truck_fullness car_fullness : ℚ) : ℚ :=
  (truck_capacity - truck_capacity * truck_fullness) + (car_capacity - car_capacity * car_fullness)

/-- Proves that the total amount of gas needed to fill both tanks is 18 gallons. -/
theorem gas_needed_is_eighteen :
  total_gas_needed 20 12 (1/2) (1/3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gas_needed_is_eighteen_l230_23032


namespace NUMINAMATH_CALUDE_no_three_integers_divisibility_l230_23061

theorem no_three_integers_divisibility : ¬∃ (x y z : ℤ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (y ∣ (x^2 - 1)) ∧ (z ∣ (x^2 - 1)) ∧
  (x ∣ (y^2 - 1)) ∧ (z ∣ (y^2 - 1)) ∧
  (x ∣ (z^2 - 1)) ∧ (y ∣ (z^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_integers_divisibility_l230_23061


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l230_23000

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
  (h_q_pos : q > 0)
  (h_T : ∀ n : ℕ, T n = (T 1) * q^(n * (n - 1) / 2))
  (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
  (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l230_23000


namespace NUMINAMATH_CALUDE_angle_300_shares_terminal_side_with_neg_60_l230_23082

-- Define the concept of angles sharing the same terminal side
def shares_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - α

-- Theorem statement
theorem angle_300_shares_terminal_side_with_neg_60 :
  shares_terminal_side (-60) 300 := by
  sorry

end NUMINAMATH_CALUDE_angle_300_shares_terminal_side_with_neg_60_l230_23082


namespace NUMINAMATH_CALUDE_monica_reading_plan_l230_23057

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 25

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 3 * books_this_year + 7

theorem monica_reading_plan : books_next_year = 232 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_plan_l230_23057


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l230_23030

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.25 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 143.75 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l230_23030


namespace NUMINAMATH_CALUDE_prob_reach_opposite_after_six_moves_l230_23035

/-- Represents a cube with its vertices and edges. -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- Represents a bug's movement on the cube. -/
structure BugMovement (cube : Cube) where
  start_vertex : Fin 8
  num_moves : Nat
  prob_each_edge : ℝ

/-- The probability of the bug reaching the opposite vertex after a specific number of moves. -/
def prob_reach_opposite (cube : Cube) (movement : BugMovement cube) : ℝ :=
  sorry

/-- Theorem stating that the probability of reaching the opposite vertex after six moves is 1/8. -/
theorem prob_reach_opposite_after_six_moves (cube : Cube) (movement : BugMovement cube) :
  movement.num_moves = 6 →
  movement.prob_each_edge = 1/3 →
  prob_reach_opposite cube movement = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_reach_opposite_after_six_moves_l230_23035


namespace NUMINAMATH_CALUDE_evaluate_expression_l230_23087

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : y * (y - 3 * x) = -5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l230_23087


namespace NUMINAMATH_CALUDE_largest_power_of_five_factor_l230_23094

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 102 + factorial 103 + factorial 104 + factorial 105

theorem largest_power_of_five_factor : 
  (∀ m : ℕ, 5^(24 + 1) ∣ sum_of_factorials → 5^m ∣ sum_of_factorials) ∧ 
  5^24 ∣ sum_of_factorials :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_factor_l230_23094


namespace NUMINAMATH_CALUDE_distance_traveled_l230_23019

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 2 * t + 3

-- Define the theorem
theorem distance_traveled (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  ∫ x in a..b, velocity x = 22 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l230_23019


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l230_23066

/-- The amount of flour required for Mary's cake recipe --/
def flour_required (sugar : ℕ) (flour_sugar_diff : ℕ) (flour_added : ℕ) : ℕ :=
  sugar + flour_sugar_diff

theorem cake_recipe_flour :
  let sugar := 3
  let flour_sugar_diff := 5
  let flour_added := 2
  flour_required sugar flour_sugar_diff flour_added = 8 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l230_23066


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l230_23064

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 6 = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - m*y - 6 = 0 ∧ y = -2) ∧ m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l230_23064


namespace NUMINAMATH_CALUDE_sewn_fabric_theorem_l230_23006

/-- The length of a sewn fabric piece given the number of fabric pieces, 
    length of each piece, and length of each joint. -/
def sewn_fabric_length (num_pieces : ℕ) (piece_length : ℝ) (joint_length : ℝ) : ℝ :=
  num_pieces * piece_length - (num_pieces - 1) * joint_length

/-- Theorem stating that 20 pieces of 10 cm fabric sewn with 0.5 cm joints 
    result in a 190.5 cm long piece. -/
theorem sewn_fabric_theorem :
  sewn_fabric_length 20 10 0.5 = 190.5 := by
  sorry

#eval sewn_fabric_length 20 10 0.5

end NUMINAMATH_CALUDE_sewn_fabric_theorem_l230_23006


namespace NUMINAMATH_CALUDE_mike_sold_45_books_l230_23055

/-- The number of books Mike sold at the garage sale -/
def books_sold (initial_books current_books : ℕ) : ℕ :=
  initial_books - current_books

/-- Proof that Mike sold 45 books -/
theorem mike_sold_45_books (h1 : books_sold 51 6 = 45) : books_sold 51 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_mike_sold_45_books_l230_23055


namespace NUMINAMATH_CALUDE_rectangle_EF_length_l230_23065

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  DF : ℝ
  EF : ℝ
  h_AB : AB = 4
  h_BC : BC = 10
  h_DE_DF : DE = DF
  h_area : DE * DF / 2 = AB * BC / 4

/-- The length of EF in the given rectangle -/
theorem rectangle_EF_length (r : Rectangle) : r.EF = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_EF_length_l230_23065


namespace NUMINAMATH_CALUDE_base_10_729_to_base_7_l230_23073

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

/-- The base-10 number 729 is equal to 2061 in base 7 --/
theorem base_10_729_to_base_7 : base7ToBase10 2 0 6 1 = 729 := by
  sorry

end NUMINAMATH_CALUDE_base_10_729_to_base_7_l230_23073


namespace NUMINAMATH_CALUDE_squares_below_specific_line_l230_23072

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line 10x + 210y = 2100 --/
def specificLine : Line := { a := 10, b := 210, c := 2100 }

theorem squares_below_specific_line :
  countSquaresBelowLine specificLine = 941 :=
sorry

end NUMINAMATH_CALUDE_squares_below_specific_line_l230_23072


namespace NUMINAMATH_CALUDE_complex_product_ab_l230_23029

theorem complex_product_ab (z : ℂ) (a b : ℝ) : 
  z = a + b * Complex.I → 
  z = (4 + 3 * Complex.I) * Complex.I → 
  a * b = -12 := by sorry

end NUMINAMATH_CALUDE_complex_product_ab_l230_23029


namespace NUMINAMATH_CALUDE_min_participants_l230_23098

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race -/
structure Race where
  participants : List Participant
  /-- No two participants finished simultaneously -/
  no_ties : ∀ p1 p2 : Participant, p1 ∈ participants → p2 ∈ participants → p1 ≠ p2 → p1.position ≠ p2.position

/-- The number of people who finished before a given participant -/
def finished_before (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position < p.position)).length

/-- The number of people who finished after a given participant -/
def finished_after (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position > p.position)).length

/-- The theorem stating the minimum number of participants in the race -/
theorem min_participants (race : Race) 
  (andrei dima lenya : Participant)
  (andrei_in : andrei ∈ race.participants)
  (dima_in : dima ∈ race.participants)
  (lenya_in : lenya ∈ race.participants)
  (andrei_cond : finished_before race andrei = (finished_after race andrei) / 2)
  (dima_cond : finished_before race dima = (finished_after race dima) / 3)
  (lenya_cond : finished_before race lenya = (finished_after race lenya) / 4) :
  race.participants.length ≥ 61 := by
  sorry

end NUMINAMATH_CALUDE_min_participants_l230_23098


namespace NUMINAMATH_CALUDE_red_peaches_count_l230_23099

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 6

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 16

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- The total number of red peaches in all baskets -/
def total_red_peaches : ℕ := num_baskets * red_peaches_per_basket

theorem red_peaches_count : total_red_peaches = 96 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l230_23099


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l230_23077

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (4 - 5 * Complex.I) = -52/41 + (99/41) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l230_23077


namespace NUMINAMATH_CALUDE_diameter_circumference_relation_l230_23081

theorem diameter_circumference_relation (c : ℝ) (d : ℝ) (π : ℝ) : c > 0 → d > 0 → π > 0 → c = π * d → d = (1 / π) * c := by
  sorry

end NUMINAMATH_CALUDE_diameter_circumference_relation_l230_23081


namespace NUMINAMATH_CALUDE_wooden_block_stacks_height_difference_l230_23079

/-- The height of wooden block stacks problem -/
theorem wooden_block_stacks_height_difference :
  let first_stack : ℕ := 7
  let second_stack : ℕ := first_stack + 3
  let third_stack : ℕ := second_stack - 6
  let fifth_stack : ℕ := 2 * second_stack
  let total_blocks : ℕ := 55
  let other_stacks_total : ℕ := first_stack + second_stack + third_stack + fifth_stack
  let fourth_stack : ℕ := total_blocks - other_stacks_total
  fourth_stack - third_stack = 10 :=
by sorry

end NUMINAMATH_CALUDE_wooden_block_stacks_height_difference_l230_23079


namespace NUMINAMATH_CALUDE_chloe_boxes_l230_23024

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Chloe has -/
def total_pieces : ℕ := 32

/-- The number of boxes Chloe found -/
def boxes : ℕ := total_pieces / (scarves_per_box + mittens_per_box)

theorem chloe_boxes : boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloe_boxes_l230_23024


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l230_23067

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l230_23067


namespace NUMINAMATH_CALUDE_forty_is_twenty_percent_of_two_hundred_l230_23046

theorem forty_is_twenty_percent_of_two_hundred (x : ℝ) : 40 = (20 / 100) * x → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_forty_is_twenty_percent_of_two_hundred_l230_23046


namespace NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l230_23025

/-- Given a mixture of alcohol and water, this theorem proves that if the initial ratio
    of alcohol to water is 4:3, and adding 4 liters of water changes the ratio to 4:5,
    then the initial quantity of alcohol in the mixture is 8 liters. -/
theorem alcohol_quantity_in_mixture
  (initial_alcohol : ℝ) (initial_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol / (initial_water + 4) = 4 / 5) :
  initial_alcohol = 8 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l230_23025


namespace NUMINAMATH_CALUDE_units_digit_of_product_l230_23027

theorem units_digit_of_product (n : ℕ) : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l230_23027


namespace NUMINAMATH_CALUDE_simplify_absolute_value_sum_l230_23010

theorem simplify_absolute_value_sum (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2*b + 5| + |-3*a + 2*b - 2| = 4*a - 4*b + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_sum_l230_23010


namespace NUMINAMATH_CALUDE_line_through_P_parallel_to_tangent_at_M_l230_23012

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- Point P -/
def P : ℝ × ℝ := (-1, 2)

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at point M -/
def k : ℝ := f' M.1

/-- The equation of the line passing through P and parallel to the tangent line at M -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem line_through_P_parallel_to_tangent_at_M :
  line_equation P.1 P.2 ∧
  ∀ x y, line_equation x y → (y - P.2) = k * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_parallel_to_tangent_at_M_l230_23012


namespace NUMINAMATH_CALUDE_equation_solution_l230_23036

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 19))) = 58 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l230_23036


namespace NUMINAMATH_CALUDE_group_size_problem_l230_23053

theorem group_size_problem (total_cents : ℕ) (h1 : total_cents = 64736) : ∃ n : ℕ, n * n = total_cents ∧ n = 254 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l230_23053


namespace NUMINAMATH_CALUDE_carrot_cost_theorem_l230_23096

/-- Calculates the total cost of carrots for a year given the daily consumption, carrots per bag, and cost per bag. -/
theorem carrot_cost_theorem (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) :
  carrots_per_day = 1 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  (365 * carrots_per_day / carrots_per_bag : ℚ).ceil * cost_per_bag = 146 := by
  sorry

#eval (365 * 1 / 5 : ℚ).ceil * 2

end NUMINAMATH_CALUDE_carrot_cost_theorem_l230_23096


namespace NUMINAMATH_CALUDE_translation_proof_l230_23062

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- Translate a point by a vector -/
def translate (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem translation_proof (A C D : Point)
    (h1 : A = ⟨-1, 4⟩)
    (h2 : C = ⟨4, 7⟩)
    (h3 : D = ⟨-4, 1⟩)
    (h4 : ∃ B : Point, vector A B = vector C D) :
    ∃ B : Point, B = ⟨-9, -2⟩ ∧ vector A B = vector C D := by
  sorry

#check translation_proof

end NUMINAMATH_CALUDE_translation_proof_l230_23062


namespace NUMINAMATH_CALUDE_infinite_solutions_l230_23031

/-- Standard prime factorization of a positive integer -/
def prime_factorization (n : ℕ+) : List (ℕ × ℕ) := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := 
  let factors := prime_factorization n
  (factors.map (fun (p, α) => α)).prod * 
  (factors.map (fun (p, α) => p^(α - 1))).prod

/-- The set of positive integers n satisfying f(n+1) = f(n) + 1 -/
def S : Set ℕ+ := {n | f (n + 1) = f n + 1}

/-- The main theorem to be proved -/
theorem infinite_solutions : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l230_23031


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l230_23001

/-- Represents a color of a tile -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 19)

/-- Represents the coloring of the grid -/
def Coloring := Position → Color

/-- Represents a rectangle in the grid -/
structure Rectangle :=
  (topLeft : Position)
  (bottomRight : Position)

/-- Checks if all vertices of a rectangle have the same color -/
def sameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  let tl := c r.topLeft
  let tr := c ⟨r.topLeft.row, r.bottomRight.col⟩
  let bl := c ⟨r.bottomRight.row, r.topLeft.col⟩
  let br := c r.bottomRight
  tl = tr ∧ tl = bl ∧ tl = br

theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), sameColorVertices r c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l230_23001


namespace NUMINAMATH_CALUDE_intersection_A_C_R_B_range_of_a_l230_23052

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the relative complement of B with respect to ℝ
def C_R_B : Set ℝ := {x | x ∉ B}

-- Theorem 1: A ∩ (C_R B) = {x | -3 < x ≤ 2}
theorem intersection_A_C_R_B : A ∩ C_R_B = {x : ℝ | -3 < x ∧ x ≤ 2} := by
  sorry

-- Theorem 2: If C ⊇ (A ∩ B), then 4/3 ≤ a ≤ 2
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  (C a ⊇ (A ∩ B)) → (4/3 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_C_R_B_range_of_a_l230_23052


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l230_23059

theorem relay_team_arrangements (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l230_23059


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l230_23095

/-- Given that Bryan has 34 books distributed equally in 2 bookshelves,
    prove that there are 17 books in each bookshelf. -/
theorem books_per_bookshelf :
  ∀ (total_books : ℕ) (num_bookshelves : ℕ) (books_per_shelf : ℕ),
    total_books = 34 →
    num_bookshelves = 2 →
    total_books = num_bookshelves * books_per_shelf →
    books_per_shelf = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l230_23095


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l230_23060

theorem divide_by_repeating_decimal : 
  let x : ℚ := 142857 / 999999
  7 / x = 49 := by sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l230_23060


namespace NUMINAMATH_CALUDE_circular_path_time_increase_l230_23020

/-- 
Prove that if a person can go round a circular path 8 times in 40 minutes, 
and the diameter of the circle is increased to 10 times the original diameter, 
then the time required to go round the new path once, traveling at the same speed as before, 
is 50 minutes.
-/
theorem circular_path_time_increase 
  (original_rounds : ℕ) 
  (original_time : ℕ) 
  (diameter_increase : ℕ) 
  (h1 : original_rounds = 8) 
  (h2 : original_time = 40) 
  (h3 : diameter_increase = 10) : 
  (original_time / original_rounds) * diameter_increase = 50 := by
  sorry

#check circular_path_time_increase

end NUMINAMATH_CALUDE_circular_path_time_increase_l230_23020


namespace NUMINAMATH_CALUDE_age_ratio_seven_years_ago_l230_23097

-- Define the present ages of Henry and Jill
def henry_present_age : ℕ := 25
def jill_present_age : ℕ := 16

-- Define the sum of their present ages
def sum_present_ages : ℕ := henry_present_age + jill_present_age

-- Define their ages 7 years ago
def henry_past_age : ℕ := henry_present_age - 7
def jill_past_age : ℕ := jill_present_age - 7

-- Define the theorem
theorem age_ratio_seven_years_ago :
  sum_present_ages = 41 →
  ∃ k : ℕ, henry_past_age = k * jill_past_age →
  henry_past_age / jill_past_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_seven_years_ago_l230_23097


namespace NUMINAMATH_CALUDE_ellipse_and_tangent_line_l230_23085

/-- Given an ellipse and a line passing through its vertex and focus, 
    prove the standard equation of the ellipse and its tangent line equation. -/
theorem ellipse_and_tangent_line 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (ellipse : ℝ → ℝ → Prop) 
  (line : ℝ → ℝ → Prop) 
  (h_ellipse : ellipse = λ x y => x^2/a^2 + y^2/b^2 = 1)
  (h_line : line = λ x y => Real.sqrt 6 * x + 2 * y - 2 * Real.sqrt 6 = 0)
  (h_vertex_focus : ∃ (E F : ℝ × ℝ), 
    ellipse E.1 E.2 ∧ 
    ellipse F.1 F.2 ∧ 
    line E.1 E.2 ∧ 
    line F.1 F.2 ∧ 
    (E.1 = 0 ∧ E.2 = Real.sqrt 6) ∧ 
    (F.1 = 2 ∧ F.2 = 0)) :
  (∀ x y, ellipse x y ↔ x^2/10 + y^2/6 = 1) ∧
  (∀ x y, (Real.sqrt 5 / 10) * x + (Real.sqrt 3 / 6) * y = 1 →
    (x = Real.sqrt 5 ∧ y = Real.sqrt 3) ∨
    ¬(ellipse x y)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_tangent_line_l230_23085


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l230_23058

theorem round_trip_ticket_percentage (total_passengers : ℝ) :
  let round_trip_with_car := 0.20 * total_passengers
  let round_trip_without_car_ratio := 0.40
  let round_trip_passengers := round_trip_with_car / (1 - round_trip_without_car_ratio)
  round_trip_passengers / total_passengers = 1/3 := by
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l230_23058


namespace NUMINAMATH_CALUDE_ice_cream_cost_l230_23049

theorem ice_cream_cost (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) 
  (family_members : ℕ) (remaining_cents : ℕ) :
  pennies = 123 →
  nickels = 85 →
  dimes = 35 →
  quarters = 26 →
  family_members = 5 →
  remaining_cents = 48 →
  let total_cents := pennies + nickels * 5 + dimes * 10 + quarters * 25
  let spent_cents := total_cents - remaining_cents
  let cost_per_scoop := spent_cents / family_members
  cost_per_scoop = 300 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l230_23049


namespace NUMINAMATH_CALUDE_f_increasing_for_x_gt_1_l230_23014

-- Define the function f(x) = (x-1)^2 + 1
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem f_increasing_for_x_gt_1 : ∀ x > 1, deriv f x > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_for_x_gt_1_l230_23014


namespace NUMINAMATH_CALUDE_total_marks_is_660_l230_23089

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ
  history : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science + scores.history

/-- Eva's scores for the second semester -/
def secondSemester : SemesterScores :=
  { maths := 80, arts := 90, science := 90, history := 85 }

/-- Eva's scores for the first semester -/
def firstSemester : SemesterScores :=
  { maths := secondSemester.maths + 10,
    arts := secondSemester.arts - 15,
    science := secondSemester.science - (secondSemester.science / 3),
    history := secondSemester.history + 5 }

/-- Theorem: The total number of marks in all semesters is 660 -/
theorem total_marks_is_660 :
  totalScore firstSemester + totalScore secondSemester = 660 := by
  sorry


end NUMINAMATH_CALUDE_total_marks_is_660_l230_23089


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l230_23026

-- Define an isosceles triangle with an exterior angle of 140°
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  isIsosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  sumOfAngles : angles 0 + angles 1 + angles 2 = 180
  exteriorAngle : ℝ
  exteriorAngleValue : exteriorAngle = 140

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angles 0 = 40 ∧ t.angles 1 = 40 ∧ t.angles 2 = 100) ∨
  (t.angles 0 = 70 ∧ t.angles 1 = 70 ∧ t.angles 2 = 40) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l230_23026


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_squared_l230_23076

/-- Given real numbers x and y satisfying 3(x^3 + y^3) = x + y^2,
    the maximum value of x + y^2 is 1/3. -/
theorem max_value_x_plus_y_squared (x y : ℝ) 
  (h : 3 * (x^3 + y^3) = x + y^2) : 
  ∃ (M : ℝ), M = 1/3 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b^2 → a + b^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_squared_l230_23076


namespace NUMINAMATH_CALUDE_sum_of_products_formula_l230_23003

/-- The sum of products resulting from repeatedly dividing n balls into two groups -/
def sumOfProducts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the sum of products for n balls is n * (n-1) / 2 -/
theorem sum_of_products_formula (n : ℕ) : 
  sumOfProducts n = n * (n - 1) / 2 := by
  sorry

#check sum_of_products_formula

end NUMINAMATH_CALUDE_sum_of_products_formula_l230_23003


namespace NUMINAMATH_CALUDE_constant_sum_zero_l230_23056

theorem constant_sum_zero (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x + 1)) →
  (2 = a + b / (1 + 1)) →
  (3 = a + b / (3 + 1)) →
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_constant_sum_zero_l230_23056


namespace NUMINAMATH_CALUDE_solve_equation_l230_23080

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l230_23080


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l230_23037

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m^2 * x) / (x^2 - m)

theorem tangent_line_at_origin (h : m = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = f m x → (x = 0 ∧ y = 0) → x + y = 0 :=
sorry

theorem monotonically_decreasing_when_m_positive (h : m > 0) :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f m x₁ > f m x₂ :=
sorry

theorem extremum_values_iff_m_negative :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ 
   (∀ (x : ℝ), f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)) ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l230_23037


namespace NUMINAMATH_CALUDE_intersection_equals_B_range_l230_23038

/-- The set A -/
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

/-- The set B parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

/-- The theorem stating the range of m when A ∩ B = B -/
theorem intersection_equals_B_range (m : ℝ) : 
  (A ∩ B m = B m) ↔ (1 ≤ m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_B_range_l230_23038


namespace NUMINAMATH_CALUDE_debate_team_arrangements_l230_23092

/-- Represents the debate team composition -/
structure DebateTeam :=
  (male_count : Nat)
  (female_count : Nat)

/-- The number of arrangements where no two male members are adjacent -/
def non_adjacent_male_arrangements (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to divide the team into four groups of two and assign them to four classes -/
def seminar_groupings (team : DebateTeam) : Nat :=
  sorry

/-- The number of ways to select 4 members (with at least one male) and assign them to four speaker roles -/
def speaker_selections (team : DebateTeam) : Nat :=
  sorry

theorem debate_team_arrangements (team : DebateTeam) 
  (h1 : team.male_count = 3) 
  (h2 : team.female_count = 5) : 
  non_adjacent_male_arrangements team = 14400 ∧ 
  seminar_groupings team = 2520 ∧ 
  speaker_selections team = 1560 :=
sorry

end NUMINAMATH_CALUDE_debate_team_arrangements_l230_23092


namespace NUMINAMATH_CALUDE_digit_sum_proof_l230_23054

/-- Represents the number of '1's in the original number -/
def num_ones : ℕ := 2018

/-- Represents the number of '5's in the original number -/
def num_fives : ℕ := 2017

/-- Represents the original number under the square root -/
def original_number : ℕ :=
  (10^num_ones - 1) / 9 * 10^(num_fives + 1) + 
  5 * (10^num_fives - 1) / 9 * 10^num_ones + 
  6

/-- The sum of digits in the decimal representation of the integer part 
    of the square root of the original number -/
def digit_sum : ℕ := num_ones * 3 + 4

theorem digit_sum_proof : digit_sum = 6055 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l230_23054


namespace NUMINAMATH_CALUDE_veranda_area_l230_23017

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) 
  (h1 : room_length = 20)
  (h2 : room_width = 12)
  (h3 : veranda_width = 2) : 
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
  room_length * room_width = 144 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l230_23017


namespace NUMINAMATH_CALUDE_e_opposite_x_l230_23071

/-- Represents the faces of a cube --/
inductive Face : Type
  | X | A | B | C | D | E

/-- Represents the net of a cube --/
structure CubeNet where
  central : Face
  left : Face
  right : Face
  bottom : Face
  connected_to_right : Face
  connected_to_left : Face

/-- Defines the specific cube net given in the problem --/
def given_net : CubeNet :=
  { central := Face.X
  , left := Face.A
  , right := Face.B
  , bottom := Face.D
  , connected_to_right := Face.C
  , connected_to_left := Face.E
  }

/-- Defines the concept of opposite faces in a cube --/
def opposite (f1 f2 : Face) : Prop := sorry

/-- Theorem stating that in the given net, E is opposite to X --/
theorem e_opposite_x (net : CubeNet) : 
  net = given_net → opposite Face.E Face.X :=
sorry

end NUMINAMATH_CALUDE_e_opposite_x_l230_23071


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l230_23045

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l230_23045


namespace NUMINAMATH_CALUDE_rectangle_area_comparison_l230_23050

theorem rectangle_area_comparison 
  (A B C D A' B' C' D' : ℝ) 
  (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) (hD : 0 ≤ D)
  (hA' : 0 ≤ A') (hB' : 0 ≤ B') (hC' : 0 ≤ C') (hD' : 0 ≤ D')
  (hAA' : A ≤ A') (hBB' : B ≤ B') (hCC' : C ≤ C') (hDB' : D ≤ B') :
  A + B + C + D ≤ A' + B' + C' + D' := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_comparison_l230_23050


namespace NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l230_23008

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 :=
by sorry

end NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l230_23008


namespace NUMINAMATH_CALUDE_hacky_sack_jumping_rope_problem_l230_23093

theorem hacky_sack_jumping_rope_problem : 
  ∀ (hacky_sack_players jump_rope_players : ℕ),
    hacky_sack_players = 6 →
    jump_rope_players = 6 * hacky_sack_players →
    jump_rope_players ≠ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_hacky_sack_jumping_rope_problem_l230_23093


namespace NUMINAMATH_CALUDE_jellybean_probability_l230_23016

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def green_jellybeans : ℕ := 3
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 1 * Nat.choose (total_jellybeans - red_jellybeans - green_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 2 / 13 :=
sorry

end NUMINAMATH_CALUDE_jellybean_probability_l230_23016


namespace NUMINAMATH_CALUDE_square_difference_equality_l230_23021

theorem square_difference_equality : 1010^2 - 994^2 - 1008^2 + 996^2 = 8016 := by sorry

end NUMINAMATH_CALUDE_square_difference_equality_l230_23021


namespace NUMINAMATH_CALUDE_tangent_circle_position_l230_23042

/-- Represents a trapezoid EFGH with a circle tangent to two sides --/
structure TrapezoidWithTangentCircle where
  -- Lengths of the trapezoid sides
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  -- Q is the center of the circle on EF
  EQ : ℝ
  -- Assumption that EF is parallel to GH is implicit in the structure

/-- The main theorem about the tangent circle in a specific trapezoid --/
theorem tangent_circle_position 
  (t : TrapezoidWithTangentCircle)
  (h1 : t.EF = 86)
  (h2 : t.FG = 60)
  (h3 : t.GH = 26)
  (h4 : t.HE = 80)
  (h5 : t.EQ > 0)
  (h6 : t.EQ < t.EF) :
  t.EQ = 160 / 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_position_l230_23042


namespace NUMINAMATH_CALUDE_population_growth_s_curve_l230_23086

/-- Represents the population size at a given time -/
def PopulationSize := ℝ

/-- Represents time -/
def Time := ℝ

/-- Represents the carrying capacity of the environment -/
def CarryingCapacity := ℝ

/-- Represents the growth rate of the population -/
def GrowthRate := ℝ

/-- A function that models population growth over time -/
def populationGrowthModel (t : Time) (K : CarryingCapacity) (r : GrowthRate) : PopulationSize :=
  sorry

/-- Predicate that checks if a function exhibits an S-curve pattern -/
def isSCurve (f : Time → PopulationSize) : Prop :=
  sorry

/-- Theorem stating that population growth often exhibits an S-curve in nature -/
theorem population_growth_s_curve 
  (limitedEnvironment : CarryingCapacity → Prop)
  (environmentalFactors : (Time → PopulationSize) → Prop) :
  ∃ (K : CarryingCapacity) (r : GrowthRate),
    limitedEnvironment K ∧ 
    environmentalFactors (populationGrowthModel · K r) ∧
    isSCurve (populationGrowthModel · K r) :=
  sorry

end NUMINAMATH_CALUDE_population_growth_s_curve_l230_23086


namespace NUMINAMATH_CALUDE_charles_pictures_l230_23063

theorem charles_pictures (total_papers : ℕ) (drawn_today : ℕ) (drawn_yesterday_before : ℕ) (papers_left : ℕ) : 
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before = 6 →
  papers_left = 2 →
  total_papers - (drawn_today + drawn_yesterday_before) - papers_left = 6 :=
by sorry

end NUMINAMATH_CALUDE_charles_pictures_l230_23063


namespace NUMINAMATH_CALUDE_smallest_difference_l230_23070

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = a - b ∧ (∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_l230_23070


namespace NUMINAMATH_CALUDE_sector_central_angle_l230_23068

/-- Given a sector with radius R and a perimeter equal to half the circumference of its circle,
    the central angle of the sector is (π - 2) radians. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 2 * π ∧ 
  (2 * R + R * θ = π * R) → θ = π - 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l230_23068


namespace NUMINAMATH_CALUDE_solution_problem_l230_23074

theorem solution_problem (x y : ℕ) 
  (h1 : 0 < x ∧ x < 30) 
  (h2 : 0 < y ∧ y < 30) 
  (h3 : x + y + x * y = 104) : 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_problem_l230_23074


namespace NUMINAMATH_CALUDE_particle_max_height_l230_23051

/-- The height function of the particle -/
def h (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- The maximum height reached by the particle -/
def max_height : ℝ := 450

/-- Theorem stating that the maximum height reached by the particle is 450 meters -/
theorem particle_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_particle_max_height_l230_23051


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l230_23048

theorem fraction_product_simplification :
  (240 : ℚ) / 20 * 6 / 180 * 10 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l230_23048


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l230_23084

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = - 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l230_23084


namespace NUMINAMATH_CALUDE_scott_smoothie_sales_l230_23015

/-- Proves that Scott sold 40 cups of smoothies given the conditions of the problem -/
theorem scott_smoothie_sales :
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let cakes_sold : ℕ := 18
  let total_money : ℕ := 156
  let smoothies_sold : ℕ := (total_money - cake_price * cakes_sold) / smoothie_price
  smoothies_sold = 40 := by sorry

end NUMINAMATH_CALUDE_scott_smoothie_sales_l230_23015


namespace NUMINAMATH_CALUDE_system_solution_l230_23011

theorem system_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 27) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 31 / 17 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l230_23011


namespace NUMINAMATH_CALUDE_camel_inheritance_theorem_l230_23075

theorem camel_inheritance_theorem :
  let total_camels : ℕ := 17
  let eldest_share : ℚ := 1/2
  let middle_share : ℚ := 1/3
  let youngest_share : ℚ := 1/9
  eldest_share + middle_share + youngest_share = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_camel_inheritance_theorem_l230_23075


namespace NUMINAMATH_CALUDE_arithmetic_sequence_convex_condition_l230_23033

/-- A sequence a is convex if a(n+1) + a(n-1) ≤ 2*a(n) for all n ≥ 2 -/
def IsConvexSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) + a (n - 1) ≤ 2 * a n

/-- The nth term of an arithmetic sequence with first term b₁ and common difference d -/
def ArithmeticSequence (b₁ d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem arithmetic_sequence_convex_condition (d : ℝ) :
  let b := ArithmeticSequence 2 (Real.log d)
  IsConvexSequence (fun n => b n / n) → d ≥ Real.exp 2 := by
  sorry

#check arithmetic_sequence_convex_condition

end NUMINAMATH_CALUDE_arithmetic_sequence_convex_condition_l230_23033


namespace NUMINAMATH_CALUDE_triangle_existence_l230_23083

/-- A triangle with sides x, 10 + x, and 24 can exist if and only if x is a positive integer and x ≥ 34. -/
theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = x ∧ b = x + 10 ∧ c = 24 ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ 
  x ≥ 34 := by
  sorry

#check triangle_existence

end NUMINAMATH_CALUDE_triangle_existence_l230_23083


namespace NUMINAMATH_CALUDE_minimal_ratio_is_two_thirds_l230_23047

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point

/-- The total area of dark spots on the tablecloth -/
def total_spot_area (t : Tablecloth) : ℝ := sorry

/-- The visible area of spots when folded along a specified line -/
def visible_area_when_folded (t : Tablecloth) (fold_type : Nat) : ℝ := sorry

theorem minimal_ratio_is_two_thirds (t : Tablecloth) :
  let S := total_spot_area t
  let S₁ := visible_area_when_folded t 1  -- Folding along first median or diagonal
  (∀ (i : Nat), i ≤ 3 → visible_area_when_folded t i = S₁) ∧  -- First three folds result in S₁
  (visible_area_when_folded t 4 = S) →  -- Fourth fold (other diagonal) results in S
  ∃ (ratio : ℝ), (∀ (r : ℝ), S₁ / S ≥ r → r ≤ ratio) ∧ ratio = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_minimal_ratio_is_two_thirds_l230_23047


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l230_23028

theorem fraction_sum_simplification (a b : ℝ) (h : a ≠ b) :
  a^2 / (a - b) + (2*a*b - b^2) / (b - a) = a - b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l230_23028


namespace NUMINAMATH_CALUDE_triangle_problem_l230_23040

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 1 + (Real.tan t.A / Real.tan t.B) = 2 * t.c / t.b)
  (h2 : t.a = Real.sqrt 3) :
  t.A = π/3 ∧ 
  (∀ (t' : Triangle), t'.a = Real.sqrt 3 → t'.b * t'.c ≤ t.b * t.c → 
    t.b = t.c ∧ t.b = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l230_23040
