import Mathlib

namespace NUMINAMATH_CALUDE_power_calculation_l3037_303727

theorem power_calculation : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3037_303727


namespace NUMINAMATH_CALUDE_true_proposition_l3037_303789

-- Define proposition p
def p : Prop := ∀ x : ℝ, (3 : ℝ) ^ x > 0

-- Define proposition q
def q : Prop := (∀ x : ℝ, x > 0 → x > 1) ∧ ¬(∀ x : ℝ, x > 1 → x > 0)

-- Theorem statement
theorem true_proposition : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_true_proposition_l3037_303789


namespace NUMINAMATH_CALUDE_smallest_with_six_odd_twelve_even_divisors_l3037_303701

/-- Count the number of positive odd integer divisors of a natural number -/
def countOddDivisors (n : ℕ) : ℕ := sorry

/-- Count the number of positive even integer divisors of a natural number -/
def countEvenDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly six positive odd integer divisors and twelve positive even integer divisors -/
def hasSixOddTwelveEvenDivisors (n : ℕ) : Prop :=
  countOddDivisors n = 6 ∧ countEvenDivisors n = 12

theorem smallest_with_six_odd_twelve_even_divisors :
  ∃ (n : ℕ), n > 0 ∧ hasSixOddTwelveEvenDivisors n ∧
  ∀ (m : ℕ), m > 0 → hasSixOddTwelveEvenDivisors m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_smallest_with_six_odd_twelve_even_divisors_l3037_303701


namespace NUMINAMATH_CALUDE_difference_of_numbers_l3037_303781

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) :
  |x - y| = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l3037_303781


namespace NUMINAMATH_CALUDE_total_paint_is_47_l3037_303724

/-- Calculates the total amount of paint used for all canvases --/
def total_paint_used (extra_large_count : ℕ) (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) 
  (extra_large_paint : ℕ) (large_paint : ℕ) (medium_paint : ℕ) (small_paint : ℕ) : ℕ :=
  extra_large_count * extra_large_paint + 
  large_count * large_paint + 
  medium_count * medium_paint + 
  small_count * small_paint

/-- Theorem stating that the total paint used is 47 ounces --/
theorem total_paint_is_47 : 
  total_paint_used 3 5 6 8 4 3 2 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_is_47_l3037_303724


namespace NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l3037_303796

theorem dice_sum_divisibility_probability (n : ℕ) (a b c : ℕ) 
  (h1 : a + b + c = n) 
  (h2 : 0 ≤ a ∧ a ≤ n) 
  (h3 : 0 ≤ b ∧ b ≤ n) 
  (h4 : 0 ≤ c ∧ c ≤ n) :
  (a^3 + b^3 + c^3 + 6*a*b*c : ℚ) / (n^3 : ℚ) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l3037_303796


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3037_303795

theorem book_arrangement_theorem :
  let n : ℕ := 7  -- number of books
  let k : ℕ := 3  -- number of shelves
  let arrangements := (n - 1).choose (k - 1) * n.factorial
  arrangements = 75600 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3037_303795


namespace NUMINAMATH_CALUDE_passengers_from_other_continents_l3037_303705

theorem passengers_from_other_continents 
  (total : ℕ) 
  (north_america : ℚ)
  (europe : ℚ)
  (africa : ℚ)
  (asia : ℚ)
  (h1 : total = 108)
  (h2 : north_america = 1 / 12)
  (h3 : europe = 1 / 4)
  (h4 : africa = 1 / 9)
  (h5 : asia = 1 / 6)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_passengers_from_other_continents_l3037_303705


namespace NUMINAMATH_CALUDE_white_ball_from_first_urn_l3037_303709

/-- Represents an urn with black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing an urn -/
def urn_prob : ℚ := 1/2

/-- Calculate the probability of drawing a white ball from an urn -/
def white_ball_prob (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem to prove -/
theorem white_ball_from_first_urn 
  (urn1 : Urn)
  (urn2 : Urn)
  (h1 : urn1 = ⟨3, 7⟩)
  (h2 : urn2 = ⟨4, 6⟩)
  : (urn_prob * white_ball_prob urn1) / 
    (urn_prob * white_ball_prob urn1 + urn_prob * white_ball_prob urn2) = 7/13 :=
sorry

end NUMINAMATH_CALUDE_white_ball_from_first_urn_l3037_303709


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3037_303728

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that its 8th term is -10. -/
theorem arithmetic_sequence_eighth_term :
  let a : ℕ → ℤ := fun n => 11 - 3 * (n - 1)
  a 8 = -10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3037_303728


namespace NUMINAMATH_CALUDE_binomial_coefficient_condition_l3037_303715

theorem binomial_coefficient_condition (a : ℚ) : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * a^(7-k) * 1^k) = (a + 1)^7 ∧ 
  (Nat.choose 7 6) * a * 1^6 = 1 → 
  a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_condition_l3037_303715


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3037_303708

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 10}
def N : Set ℝ := {x | x < -4/3 ∨ x > 3}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3037_303708


namespace NUMINAMATH_CALUDE_fibonacci_period_correct_l3037_303731

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the Fibonacci sequence modulo 127 -/
def fibonacci_period : ℕ := 256

theorem fibonacci_period_correct :
  fibonacci_period = 256 ∧
  (∀ m : ℕ, m > 0 → m < 256 → ¬(fib m % 127 = 0 ∧ fib (m + 1) % 127 = 1)) ∧
  fib 256 % 127 = 0 ∧
  fib 257 % 127 = 1 := by
  sorry

#check fibonacci_period_correct

end NUMINAMATH_CALUDE_fibonacci_period_correct_l3037_303731


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l3037_303777

/-- The minimum distance between a point on the circle x² + y² = 4 
    and the line √3y + x + 4√3 = 0 is 2√3 - 2 -/
theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | Real.sqrt 3 * p.2 + p.1 + 4 * Real.sqrt 3 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 3 - 2 ∧ 
    (∀ p ∈ circle, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ circle, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l3037_303777


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3037_303718

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_sum : a + b + c + d = 100)
  (h_frac_sum : a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 95) :
  1 / (b + c + d) + 1 / (a + c + d) + 1 / (a + b + d) + 1 / (a + b + c) = 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3037_303718


namespace NUMINAMATH_CALUDE_CD_length_theorem_l3037_303768

-- Define the line segment CD
def CD : Set (ℝ × ℝ × ℝ) := sorry

-- Define the region within 4 units of CD
def region (CD : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) := sorry

-- Define the volume of a set in 3D space
def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the length of a line segment
def length (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem CD_length_theorem (CD : Set (ℝ × ℝ × ℝ)) :
  volume (region CD) = 448 * Real.pi → length CD = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_CD_length_theorem_l3037_303768


namespace NUMINAMATH_CALUDE_abc_perfect_cube_l3037_303759

theorem abc_perfect_cube (a b c : ℤ) (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) :
  ∃ (n : ℤ), a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l3037_303759


namespace NUMINAMATH_CALUDE_opposite_boys_implies_total_l3037_303772

/-- Represents a circular arrangement of boys -/
structure CircularArrangement where
  num_boys : ℕ
  is_opposite : (a b : ℕ) → Prop

/-- The property that the 5th boy is opposite to the 20th boy -/
def fifth_opposite_twentieth (c : CircularArrangement) : Prop :=
  c.is_opposite 5 20

/-- Theorem stating that if the 5th boy is opposite to the 20th boy,
    then the total number of boys is 33 -/
theorem opposite_boys_implies_total (c : CircularArrangement) :
  fifth_opposite_twentieth c → c.num_boys = 33 := by
  sorry

end NUMINAMATH_CALUDE_opposite_boys_implies_total_l3037_303772


namespace NUMINAMATH_CALUDE_regular_polygon_135_degrees_has_8_sides_l3037_303769

/-- A regular polygon with interior angles of 135 degrees has 8 sides -/
theorem regular_polygon_135_degrees_has_8_sides :
  ∀ n : ℕ, 
  n > 2 →
  (180 * (n - 2) : ℝ) = 135 * n →
  n = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_135_degrees_has_8_sides_l3037_303769


namespace NUMINAMATH_CALUDE_sqrt_expression_sum_l3037_303747

theorem sqrt_expression_sum (a b c : ℤ) : 
  (64 + 24 * Real.sqrt 3 : ℝ) = (a + b * Real.sqrt c)^2 →
  c > 0 →
  (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, c = n^2 * m)) →
  a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_sum_l3037_303747


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3037_303785

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3037_303785


namespace NUMINAMATH_CALUDE_george_and_hannah_win_l3037_303799

-- Define the set of students
inductive Student : Type
  | Elaine : Student
  | Frank : Student
  | George : Student
  | Hannah : Student

-- Define a function to represent winning a prize
def wins_prize (s : Student) : Prop := sorry

-- Define the conditions
axiom only_two_winners :
  ∃ (a b : Student), a ≠ b ∧
    (∀ s : Student, wins_prize s ↔ (s = a ∨ s = b))

axiom elaine_implies_frank :
  wins_prize Student.Elaine → wins_prize Student.Frank

axiom frank_implies_george :
  wins_prize Student.Frank → wins_prize Student.George

axiom george_implies_hannah :
  wins_prize Student.George → wins_prize Student.Hannah

-- Theorem to prove
theorem george_and_hannah_win :
  wins_prize Student.George ∧ wins_prize Student.Hannah ∧
  ¬wins_prize Student.Elaine ∧ ¬wins_prize Student.Frank :=
sorry

end NUMINAMATH_CALUDE_george_and_hannah_win_l3037_303799


namespace NUMINAMATH_CALUDE_product_of_odd_numbers_not_always_composite_l3037_303713

theorem product_of_odd_numbers_not_always_composite :
  ∃ (a b : ℕ), 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    ¬(∃ (x : ℕ), 1 < x ∧ x < a * b ∧ (a * b) % x = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_odd_numbers_not_always_composite_l3037_303713


namespace NUMINAMATH_CALUDE_triangle_ratio_l3037_303720

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (area : ℝ)   -- Area

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.area = 8 ∧ t.a = 5 ∧ Real.tan t.B = -4/3

-- Define the theorem
theorem triangle_ratio (t : Triangle) (h : triangle_properties t) :
  (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 5 * Real.sqrt 65 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3037_303720


namespace NUMINAMATH_CALUDE_not_necessarily_true_inequality_l3037_303726

theorem not_necessarily_true_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬(∀ a b c, c < b ∧ b < a ∧ a * c < 0 → b^2 / c > a^2 / c) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_true_inequality_l3037_303726


namespace NUMINAMATH_CALUDE_geometric_sequence_between_9_and_243_l3037_303735

theorem geometric_sequence_between_9_and_243 :
  ∃ (a b : ℝ), 9 < a ∧ a < b ∧ b < 243 ∧
  (9 / a = a / b) ∧ (a / b = b / 243) ∧
  a = 27 ∧ b = 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_between_9_and_243_l3037_303735


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l3037_303707

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating that the number of lattice points on the given line segment is 3 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 47 297 = 3 := by sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l3037_303707


namespace NUMINAMATH_CALUDE_remaining_distance_to_nyc_l3037_303737

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (total_distance first_day second_day third_day : ℕ) : Prop :=
  let distance_walked := first_day + second_day + third_day
  total_distance - distance_walked = 36

theorem remaining_distance_to_nyc :
  richards_journey 70 20 4 10 := by sorry

end NUMINAMATH_CALUDE_remaining_distance_to_nyc_l3037_303737


namespace NUMINAMATH_CALUDE_minimum_race_distance_l3037_303761

/-- The minimum distance a runner must travel in a race with given conditions -/
theorem minimum_race_distance (wall_length : ℝ) (dist_A_to_wall : ℝ) (dist_wall_to_B : ℝ) :
  wall_length = 1600 →
  dist_A_to_wall = 600 →
  dist_wall_to_B = 800 →
  round (Real.sqrt ((wall_length ^ 2) + (dist_A_to_wall + dist_wall_to_B) ^ 2)) = 2127 :=
by sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l3037_303761


namespace NUMINAMATH_CALUDE_cricket_run_rate_l3037_303780

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate : required_run_rate 50 10 (32/10) 282 = 25/4 := by
  sorry

#eval required_run_rate 50 10 (32/10) 282

end NUMINAMATH_CALUDE_cricket_run_rate_l3037_303780


namespace NUMINAMATH_CALUDE_adult_meal_cost_l3037_303790

def restaurant_problem (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) : Prop :=
  let num_adults := total_people - num_kids
  let cost_per_adult := total_cost / num_adults
  cost_per_adult = 7

theorem adult_meal_cost :
  restaurant_problem 13 9 28 := by sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l3037_303790


namespace NUMINAMATH_CALUDE_ratio_product_l3037_303723

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_product_l3037_303723


namespace NUMINAMATH_CALUDE_inequality_proof_l3037_303703

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1 / a + 1 / b ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3037_303703


namespace NUMINAMATH_CALUDE_sandy_total_earnings_l3037_303706

def monday_earnings : ℚ := 12 * 0.5 + 5 * 0.25 + 10 * 0.1
def tuesday_earnings : ℚ := 8 * 0.5 + 15 * 0.25 + 5 * 0.1
def wednesday_earnings : ℚ := 3 * 1 + 4 * 0.5 + 10 * 0.25 + 7 * 0.05
def thursday_earnings : ℚ := 5 * 1 + 6 * 0.5 + 8 * 0.25 + 5 * 0.1 + 12 * 0.05
def friday_earnings : ℚ := 2 * 1 + 7 * 0.5 + 20 * 0.05 + 25 * 0.1

theorem sandy_total_earnings :
  monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings = 44.45 := by
  sorry

end NUMINAMATH_CALUDE_sandy_total_earnings_l3037_303706


namespace NUMINAMATH_CALUDE_max_value_of_function_l3037_303762

theorem max_value_of_function :
  (∀ x : ℝ, x > 1 → (2*x^2 + 7*x - 1) / (x^2 + 3*x) ≤ 19/9) ∧
  (∃ x : ℝ, x > 1 ∧ (2*x^2 + 7*x - 1) / (x^2 + 3*x) = 19/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3037_303762


namespace NUMINAMATH_CALUDE_angela_beth_ages_l3037_303743

/-- Angela and Beth's ages problem -/
theorem angela_beth_ages (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela + beth = 55) : 
  angela + 5 = 49 := by sorry

end NUMINAMATH_CALUDE_angela_beth_ages_l3037_303743


namespace NUMINAMATH_CALUDE_missing_chess_pieces_l3037_303744

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 24

/-- The number of missing chess pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces :
  missing_pieces = 8 := by sorry

end NUMINAMATH_CALUDE_missing_chess_pieces_l3037_303744


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3037_303739

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m = 0 ∧ y^2 + 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3037_303739


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3037_303717

def is_valid_divisor (d : ℕ) : Prop :=
  d > 0 ∧ 150 % d = 50 ∧ 55 % d = 5 ∧ 175 % d = 25

def is_greatest_divisor (d : ℕ) : Prop :=
  is_valid_divisor d ∧ ∀ k > d, ¬is_valid_divisor k

theorem smallest_valid_number : ∃ n : ℕ, n > 0 ∧ is_valid_divisor n ∧ 
  ∃ d : ℕ, is_greatest_divisor d ∧ n % d = 5 ∧ 
  ∀ m < n, ¬(is_valid_divisor m ∧ ∃ k : ℕ, is_greatest_divisor k ∧ m % k = 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3037_303717


namespace NUMINAMATH_CALUDE_pentagram_impossible_l3037_303711

/-- Represents a pentagram arrangement of numbers -/
structure PentagramArrangement :=
  (numbers : Fin 10 → ℕ)
  (is_permutation : Function.Injective numbers)
  (valid_range : ∀ i, numbers i ∈ Finset.range 11 \ {0})

/-- Represents a line in the pentagram -/
inductive PentagramLine
  | Line1 | Line2 | Line3 | Line4 | Line5

/-- Get the four positions on a given line -/
def line_positions (l : PentagramLine) : Fin 4 → Fin 10 :=
  sorry  -- Implementation details omitted

/-- The sum of numbers on a given line -/
def line_sum (arr : PentagramArrangement) (l : PentagramLine) : ℕ :=
  (Finset.range 4).sum (λ i => arr.numbers (line_positions l i))

/-- Statement: It's impossible to arrange numbers 1 to 10 in a pentagram
    such that all line sums are equal -/
theorem pentagram_impossible : ¬ ∃ (arr : PentagramArrangement),
  ∀ (l1 l2 : PentagramLine), line_sum arr l1 = line_sum arr l2 :=
sorry

end NUMINAMATH_CALUDE_pentagram_impossible_l3037_303711


namespace NUMINAMATH_CALUDE_book_selection_combinations_l3037_303742

theorem book_selection_combinations :
  let mystery_books : ℕ := 3
  let fantasy_books : ℕ := 4
  let biography_books : ℕ := 3
  let total_combinations := mystery_books * fantasy_books * biography_books
  total_combinations = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l3037_303742


namespace NUMINAMATH_CALUDE_rowing_downstream_speed_l3037_303756

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem rowing_downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 27 →
  still_water_speed = 31 →
  still_water_speed + (still_water_speed - upstream_speed) = 35 := by
sorry

end NUMINAMATH_CALUDE_rowing_downstream_speed_l3037_303756


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3037_303791

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 2*x - 1 = 0) ↔ ((x - 1)^2 = 2) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3037_303791


namespace NUMINAMATH_CALUDE_product_nonzero_l3037_303766

theorem product_nonzero (n : ℤ) : n ≠ 5 → n ≠ 17 → n ≠ 257 → (n - 5) * (n - 17) * (n - 257) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_product_nonzero_l3037_303766


namespace NUMINAMATH_CALUDE_max_trees_bucked_l3037_303741

/-- Represents the energy and tree-bucking strategy over time -/
structure BuckingStrategy where
  restTime : ℕ
  initialEnergy : ℕ
  timePeriod : ℕ

/-- Calculates the total number of trees bucked given a strategy -/
def totalTreesBucked (s : BuckingStrategy) : ℕ :=
  let buckingTime := s.timePeriod - s.restTime
  let finalEnergy := s.initialEnergy + s.restTime - buckingTime + 1
  (buckingTime * (s.initialEnergy + s.restTime + finalEnergy)) / 2

/-- The main theorem to prove -/
theorem max_trees_bucked :
  ∃ (s : BuckingStrategy),
    s.initialEnergy = 100 ∧
    s.timePeriod = 60 ∧
    (∀ (t : BuckingStrategy),
      t.initialEnergy = 100 ∧
      t.timePeriod = 60 →
      totalTreesBucked t ≤ totalTreesBucked s) ∧
    totalTreesBucked s = 4293 := by
  sorry


end NUMINAMATH_CALUDE_max_trees_bucked_l3037_303741


namespace NUMINAMATH_CALUDE_min_value_theorem_l3037_303788

theorem min_value_theorem (x y : ℝ) :
  3 * |x - y| + |2 * x - 5| = x + 1 →
  2 * x + y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3037_303788


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l3037_303792

/-- The number of terms with integer exponents in the expansion of (√x + 1/(2∛x))^n -/
def integer_exponent_terms (n : ℕ) : ℕ :=
  (Finset.filter (fun r => (2 * n - 3 * r) % 3 = 0) (Finset.range (n + 1))).card

/-- The coefficients of the first three terms in the expansion -/
def first_three_coeffs (n : ℕ) : Fin 3 → ℚ
  | 0 => 1
  | 1 => n / 2
  | 2 => n * (n - 1) / 8

/-- The condition that the first three coefficients form an arithmetic sequence -/
def arithmetic_sequence_condition (n : ℕ) : Prop :=
  2 * (first_three_coeffs n 1) = (first_three_coeffs n 0) + (first_three_coeffs n 2)

theorem binomial_expansion_theorem (n : ℕ) :
  arithmetic_sequence_condition n → integer_exponent_terms n = 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l3037_303792


namespace NUMINAMATH_CALUDE_special_function_at_zero_l3037_303746

/-- A function satisfying f(x + y) = f(x) + f(xy) for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f (x * y)

/-- Theorem: If f is a special function, then f(0) = 0 -/
theorem special_function_at_zero (f : ℝ → ℝ) (h : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_zero_l3037_303746


namespace NUMINAMATH_CALUDE_total_seashells_is_142_l3037_303773

/-- The number of seashells Joan found initially -/
def joans_initial_seashells : ℕ := 79

/-- The number of seashells Mike gave to Joan -/
def mikes_seashells : ℕ := 63

/-- The total number of seashells Joan has -/
def total_seashells : ℕ := joans_initial_seashells + mikes_seashells

/-- Theorem stating that the total number of seashells Joan has is 142 -/
theorem total_seashells_is_142 : total_seashells = 142 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_142_l3037_303773


namespace NUMINAMATH_CALUDE_fraction_equation_sum_l3037_303702

theorem fraction_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 17) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 9/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_sum_l3037_303702


namespace NUMINAMATH_CALUDE_happiness_difference_test_l3037_303784

-- Define the data from the problem
def total_observations : ℕ := 1184
def boys_happy : ℕ := 638
def boys_unhappy : ℕ := 128
def girls_happy : ℕ := 372
def girls_unhappy : ℕ := 46
def total_happy : ℕ := 1010
def total_unhappy : ℕ := 174
def total_boys : ℕ := 766
def total_girls : ℕ := 418

-- Define the χ² calculation function
def chi_square : ℚ :=
  (total_observations : ℚ) * (boys_happy * girls_unhappy - boys_unhappy * girls_happy)^2 /
  (total_happy * total_unhappy * total_boys * total_girls)

-- Define the critical values
def critical_value_001 : ℚ := 6635 / 1000
def critical_value_0005 : ℚ := 7879 / 1000

-- Theorem statement
theorem happiness_difference_test :
  (chi_square > critical_value_001) ∧ (chi_square < critical_value_0005) :=
by sorry

end NUMINAMATH_CALUDE_happiness_difference_test_l3037_303784


namespace NUMINAMATH_CALUDE_max_3k_value_l3037_303749

theorem max_3k_value (k : ℝ) : 
  (∃ x : ℝ, Real.sqrt (x^2 - k) + 2 * Real.sqrt (x^3 - 1) = x) →
  k ≥ 0 →
  k < 2 →
  ∃ m : ℝ, m = 4 ∧ ∀ k' : ℝ, 
    (∃ x : ℝ, Real.sqrt (x'^2 - k') + 2 * Real.sqrt (x'^3 - 1) = x') →
    k' ≥ 0 →
    k' < 2 →
    3 * k' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_3k_value_l3037_303749


namespace NUMINAMATH_CALUDE_max_distance_complex_l3037_303730

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 1 - Complex.I) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w + 1 - Complex.I) = 1 →
    Complex.abs (w - 1 - Complex.I) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l3037_303730


namespace NUMINAMATH_CALUDE_jakes_weight_l3037_303700

/-- Represents the weights of Jake, his sister, and Mark -/
structure SiblingWeights where
  jake : ℝ
  sister : ℝ
  mark : ℝ

/-- The conditions of the problem -/
def weightConditions (w : SiblingWeights) : Prop :=
  w.jake - 12 = 2 * (w.sister + 4) ∧
  w.mark = w.jake + w.sister + 50 ∧
  w.jake + w.sister + w.mark = 385

/-- The theorem stating Jake's current weight -/
theorem jakes_weight (w : SiblingWeights) :
  weightConditions w → w.jake = 118 := by
  sorry

#check jakes_weight

end NUMINAMATH_CALUDE_jakes_weight_l3037_303700


namespace NUMINAMATH_CALUDE_circle_polar_to_cartesian_and_area_l3037_303797

/-- Given a circle C with polar equation p = 2cosθ, this theorem proves that
    its Cartesian equation is x² - 2x + y² = 0 and its area is π. -/
theorem circle_polar_to_cartesian_and_area :
  ∀ (p θ x y : ℝ),
  (p = 2 * Real.cos θ) →                  -- Polar equation
  (x = p * Real.cos θ ∧ y = p * Real.sin θ) →  -- Polar to Cartesian conversion
  (x^2 - 2*x + y^2 = 0) ∧                 -- Cartesian equation
  (Real.pi = (Real.pi : ℝ)) :=            -- Area (π)
by sorry

end NUMINAMATH_CALUDE_circle_polar_to_cartesian_and_area_l3037_303797


namespace NUMINAMATH_CALUDE_no_natural_pairs_satisfying_divisibility_l3037_303758

theorem no_natural_pairs_satisfying_divisibility : 
  ¬∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b^a ∣ a^b - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_pairs_satisfying_divisibility_l3037_303758


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3037_303755

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3037_303755


namespace NUMINAMATH_CALUDE_algebra_textbooks_count_l3037_303783

theorem algebra_textbooks_count : ∃ (x y n : ℕ), 
  x * n + y = 2015 ∧ 
  y * n + x = 1580 ∧ 
  n > 0 ∧ 
  y = 287 := by
  sorry

end NUMINAMATH_CALUDE_algebra_textbooks_count_l3037_303783


namespace NUMINAMATH_CALUDE_alice_exceeds_quota_by_655_l3037_303712

/-- Represents the sales information for a shoe brand -/
structure ShoeBrand where
  name : String
  cost : Nat
  maxSales : Nat
  actualSales : Nat

/-- Calculates the total sales for a given shoe brand -/
def calculateSales (brand : ShoeBrand) : Nat :=
  brand.cost * brand.actualSales

/-- Calculates the total sales across all shoe brands -/
def totalSales (brands : List ShoeBrand) : Nat :=
  brands.foldl (fun acc brand => acc + calculateSales brand) 0

/-- The main theorem stating that Alice exceeds her quota by $655 -/
theorem alice_exceeds_quota_by_655 (brands : List ShoeBrand) (quota : Nat) : 
  brands = [
    { name := "Adidas", cost := 45, maxSales := 15, actualSales := 10 },
    { name := "Nike", cost := 60, maxSales := 12, actualSales := 12 },
    { name := "Reeboks", cost := 35, maxSales := 20, actualSales := 15 },
    { name := "Puma", cost := 50, maxSales := 10, actualSales := 8 },
    { name := "Converse", cost := 40, maxSales := 18, actualSales := 14 }
  ] ∧ quota = 2000 →
  totalSales brands - quota = 655 := by
  sorry

end NUMINAMATH_CALUDE_alice_exceeds_quota_by_655_l3037_303712


namespace NUMINAMATH_CALUDE_cubic_difference_l3037_303763

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y - x * y = 155) 
  (h2 : x^2 + y^2 = 325) : 
  |x^3 - y^3| = 4375 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l3037_303763


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3037_303757

/-- For a parabola with equation y² = 8x, the distance from its focus to its directrix is 4. -/
theorem parabola_focus_directrix_distance (x y : ℝ) : 
  y^2 = 8*x → (distance_focus_to_directrix : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3037_303757


namespace NUMINAMATH_CALUDE_min_value_theorem_l3037_303787

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ 
    t' > 0 ∧ u' > 0 ∧ v' > 0 ∧ w' > 0 ∧
    p' * q' * r' * s' = 16 ∧ t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3037_303787


namespace NUMINAMATH_CALUDE_composite_ratio_l3037_303714

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 264 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l3037_303714


namespace NUMINAMATH_CALUDE_zoo_bird_difference_l3037_303760

/-- Proves that in a zoo with 450 birds and where the number of birds is 5 times
    the number of all other animals, there are 360 more birds than non-bird animals. -/
theorem zoo_bird_difference (total_birds : ℕ) (bird_ratio : ℕ) 
    (h1 : total_birds = 450)
    (h2 : bird_ratio = 5)
    (h3 : total_birds = bird_ratio * (total_birds / bird_ratio)) :
  total_birds - (total_birds / bird_ratio) = 360 := by
  sorry

#eval 450 - (450 / 5)  -- This should evaluate to 360

end NUMINAMATH_CALUDE_zoo_bird_difference_l3037_303760


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l3037_303710

def cupcake_price : ℚ := 2
def cupcake_quantity : ℕ := 5
def doughnut_price : ℚ := 1
def doughnut_quantity : ℕ := 6
def pie_slice_price : ℚ := 2
def pie_slice_quantity : ℕ := 4
def cookie_quantity : ℕ := 15
def total_spent : ℚ := 33

theorem cookie_price_calculation :
  ∃ (cookie_price : ℚ),
    cookie_price * cookie_quantity +
    cupcake_price * cupcake_quantity +
    doughnut_price * doughnut_quantity +
    pie_slice_price * pie_slice_quantity = total_spent ∧
    cookie_price = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l3037_303710


namespace NUMINAMATH_CALUDE_problem_solution_l3037_303738

theorem problem_solution (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) (h4 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (a * b < a^2) ∧ (a - 1/b < b - 1/a) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3037_303738


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3037_303793

theorem trigonometric_equation_solution (x : ℝ) :
  (∃ k : ℤ, x = -π/28 + π*k/7 ∨ x = π/12 + 2*π*k/3 ∨ x = 5*π/44 + 2*π*k/11) ↔
  (Real.cos (11*x) - Real.cos (3*x) - Real.sin (11*x) + Real.sin (3*x) = Real.sqrt 2 * Real.cos (14*x)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3037_303793


namespace NUMINAMATH_CALUDE_exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l3037_303722

/-- Represents the state of a lamp (ON or OFF) -/
inductive LampState
| ON
| OFF

/-- Represents the configuration of n lamps -/
def LampConfig (n : ℕ) := Fin n → LampState

/-- Performs a single step of the lamp changing process -/
def step (n : ℕ) (config : LampConfig n) : LampConfig n :=
  sorry

/-- Checks if all lamps in the configuration are ON -/
def allOn (n : ℕ) (config : LampConfig n) : Prop :=
  sorry

/-- The initial configuration with all lamps ON -/
def initialConfig (n : ℕ) : LampConfig n :=
  sorry

/-- Theorem stating the existence of M(n) for any n > 1 -/
theorem exists_return_steps (n : ℕ) (h : n > 1) :
  ∃ M : ℕ, M > 0 ∧ allOn n ((step n)^[M] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is a power of 2 -/
theorem power_of_two_case (k : ℕ) :
  let n := 2^k
  allOn n ((step n)^[n^2 - 1] (initialConfig n)) :=
  sorry

/-- Theorem for the case when n is one more than a power of 2 -/
theorem power_of_two_plus_one_case (k : ℕ) :
  let n := 2^k + 1
  allOn n ((step n)^[n^2 - n + 1] (initialConfig n)) :=
  sorry

end NUMINAMATH_CALUDE_exists_return_steps_power_of_two_case_power_of_two_plus_one_case_l3037_303722


namespace NUMINAMATH_CALUDE_valid_positions_count_l3037_303764

/-- Represents a 6x6 chess board -/
def Board := Fin 6 → Fin 6 → Bool

/-- Represents a position of 4 chips on the board -/
def ChipPosition := Fin 4 → Fin 6 × Fin 6

/-- Checks if four points are collinear -/
def areCollinear (p₁ p₂ p₃ p₄ : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Checks if a square is attacked by at least one chip -/
def isAttacked (board : Board) (pos : ChipPosition) (x y : Fin 6) : Bool :=
  sorry

/-- Checks if all squares are attacked by at least one chip -/
def allSquaresAttacked (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Checks if a chip position is valid (chips are collinear and all squares are attacked) -/
def isValidPosition (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Counts the number of valid chip positions, including rotations and reflections -/
def countValidPositions (board : Board) : Nat :=
  sorry

/-- The main theorem: there are exactly 48 valid chip positions -/
theorem valid_positions_count :
  ∀ (board : Board), countValidPositions board = 48 :=
sorry

end NUMINAMATH_CALUDE_valid_positions_count_l3037_303764


namespace NUMINAMATH_CALUDE_race_distance_difference_l3037_303740

theorem race_distance_difference (lingling_distance mingming_distance : ℝ) 
  (h1 : lingling_distance = 380.5)
  (h2 : mingming_distance = 405.9) : 
  mingming_distance - lingling_distance = 25.4 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_difference_l3037_303740


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1105_l3037_303770

theorem modular_inverse_11_mod_1105 :
  let m : ℕ := 1105
  let a : ℕ := 11
  let b : ℕ := 201
  m = 5 * 13 * 17 →
  (a * b) % m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1105_l3037_303770


namespace NUMINAMATH_CALUDE_last_locker_opened_l3037_303753

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's movement -/
inductive Direction
| Forward
| Backward

/-- Defines the locker opening process -/
def openLockers (n : Nat) : Nat :=
  sorry

/-- Theorem stating that the last locker opened is number 86 -/
theorem last_locker_opened (n : Nat) (h : n = 512) : openLockers n = 86 := by
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l3037_303753


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3037_303719

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, n - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 2*Real.sqrt 2 + 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 2*Real.sqrt 2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3037_303719


namespace NUMINAMATH_CALUDE_custom_op_difference_l3037_303750

-- Define the custom operator @
def customOp (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem custom_op_difference : (customOp 7 2) - (customOp 2 7) = -20 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l3037_303750


namespace NUMINAMATH_CALUDE_base3_to_decimal_21201_l3037_303725

/-- Converts a list of digits in base 3 to a decimal number -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 0, 2, 1, 2]

/-- Theorem stating that the conversion of 21201 in base 3 to decimal is 208 -/
theorem base3_to_decimal_21201 :
  base3ToDecimal base3Number = 208 := by
  sorry

#eval base3ToDecimal base3Number

end NUMINAMATH_CALUDE_base3_to_decimal_21201_l3037_303725


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l3037_303704

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 7 or less with two dice -/
def waysToRollSevenOrLess : ℕ := 21

/-- The probability of rolling a sum greater than 7 with two dice -/
def probSumGreaterThanSeven : ℚ := 5 / 12

/-- Theorem stating that the probability of rolling a sum greater than 7 with two fair six-sided dice is 5/12 -/
theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 1 - (waysToRollSevenOrLess : ℚ) / totalOutcomes := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_seven_l3037_303704


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3037_303751

/-- Given a rectangle with vertices (-2, y), (8, y), (-2, 3), and (8, 3),
    if the area is 90 square units and y is positive, then y = 12. -/
theorem rectangle_y_value (y : ℝ) : y > 0 → (8 - (-2)) * (y - 3) = 90 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3037_303751


namespace NUMINAMATH_CALUDE_cauchy_schwarz_on_unit_circle_l3037_303786

theorem cauchy_schwarz_on_unit_circle (a b x y : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : x^2 + y^2 = 1) : a*x + b*y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_on_unit_circle_l3037_303786


namespace NUMINAMATH_CALUDE_floor_sqrt_30_squared_l3037_303733

theorem floor_sqrt_30_squared : ⌊Real.sqrt 30⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_30_squared_l3037_303733


namespace NUMINAMATH_CALUDE_village_population_equality_l3037_303767

/-- The rate at which Village X's population is decreasing per year -/
def decrease_rate : ℕ := sorry

/-- The initial population of Village X -/
def village_x_initial : ℕ := 74000

/-- The initial population of Village Y -/
def village_y_initial : ℕ := 42000

/-- The rate at which Village Y's population is increasing per year -/
def village_y_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

theorem village_population_equality :
  village_x_initial - years_until_equal * decrease_rate =
  village_y_initial + years_until_equal * village_y_increase →
  decrease_rate = 1200 := by
sorry

end NUMINAMATH_CALUDE_village_population_equality_l3037_303767


namespace NUMINAMATH_CALUDE_abs_value_equivalence_l3037_303752

theorem abs_value_equivalence (x : ℝ) : -1 < x ∧ x < 1 ↔ |x| < 1 := by sorry

end NUMINAMATH_CALUDE_abs_value_equivalence_l3037_303752


namespace NUMINAMATH_CALUDE_exam_results_l3037_303734

theorem exam_results (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 0.25 * total)
  (h2 : failed_both = 0.4 * total)
  (h3 : passed_both = 0.8 * total) :
  ∃ failed_english : ℝ, failed_english = 0.35 * total :=
by
  sorry

end NUMINAMATH_CALUDE_exam_results_l3037_303734


namespace NUMINAMATH_CALUDE_max_small_squares_in_large_square_l3037_303782

/-- The side length of the large square -/
def large_square_side : ℕ := 8

/-- The side length of the small squares -/
def small_square_side : ℕ := 2

/-- The maximum number of non-overlapping small squares that can fit inside the large square -/
def max_small_squares : ℕ := (large_square_side / small_square_side) ^ 2

theorem max_small_squares_in_large_square :
  max_small_squares = 16 :=
sorry

end NUMINAMATH_CALUDE_max_small_squares_in_large_square_l3037_303782


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l3037_303736

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 272) (h2 : num_friends = 5) :
  total_balloons % num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l3037_303736


namespace NUMINAMATH_CALUDE_identify_tricksters_l3037_303774

/-- Represents an inhabitant of the village -/
inductive Inhabitant
| Knight
| Trickster

/-- The village with its inhabitants -/
structure Village where
  inhabitants : Fin 65 → Inhabitant
  trickster_count : Nat
  knight_count : Nat
  trickster_count_eq : trickster_count = 2
  knight_count_eq : knight_count = 63
  total_count_eq : trickster_count + knight_count = 65

/-- A question asked to an inhabitant about a group of inhabitants -/
def Question := List (Fin 65) → Bool

/-- The result of asking questions to identify tricksters -/
structure IdentificationResult where
  questions_asked : Nat
  tricksters_found : List (Fin 65)
  all_tricksters_found : tricksters_found.length = 2

/-- The main theorem stating that tricksters can be identified with no more than 30 questions -/
theorem identify_tricksters (v : Village) : 
  ∃ (strategy : List Question), 
    ∃ (result : IdentificationResult), 
      result.questions_asked ≤ 30 ∧ 
      (∀ i : Fin 65, v.inhabitants i = Inhabitant.Trickster ↔ i ∈ result.tricksters_found) :=
sorry

end NUMINAMATH_CALUDE_identify_tricksters_l3037_303774


namespace NUMINAMATH_CALUDE_max_value_theorem_l3037_303776

theorem max_value_theorem (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3037_303776


namespace NUMINAMATH_CALUDE_four_numbers_with_avg_six_l3037_303748

theorem four_numbers_with_avg_six (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 6 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 6 →
    max a (max b (max c d)) - min a (min b (min c d)) ≥ max w (max x (max y z)) - min w (min x (min y z)) →
  (((max a (max b (max c d)) + min a (min b (min c d))) - (a + b + c + d)) / 2 : ℚ) = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_avg_six_l3037_303748


namespace NUMINAMATH_CALUDE_marbles_found_vs_lost_l3037_303745

theorem marbles_found_vs_lost (initial : ℕ) (lost : ℕ) (found : ℕ) :
  initial = 7 → lost = 8 → found = 10 → found - lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_found_vs_lost_l3037_303745


namespace NUMINAMATH_CALUDE_expression_evaluation_l3037_303754

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a = 2 / b) :
  (a - 2 / a) * (b + 2 / b) = a^2 - 4 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3037_303754


namespace NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l3037_303794

/-- Given an ellipse with parametric equations x = 2cos(t) and y = 4sin(t),
    prove that the slope of the line OM, where M is the point on the ellipse
    corresponding to t = π/3 and O is the origin, is 2√3. -/
theorem ellipse_slope_at_pi_third :
  let x : ℝ → ℝ := λ t ↦ 2 * Real.cos t
  let y : ℝ → ℝ := λ t ↦ 4 * Real.sin t
  let M : ℝ × ℝ := (x (π/3), y (π/3))
  let O : ℝ × ℝ := (0, 0)
  let slope := (M.2 - O.2) / (M.1 - O.1)
  slope = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l3037_303794


namespace NUMINAMATH_CALUDE_hyperbola_y_relationship_l3037_303771

theorem hyperbola_y_relationship (k : ℝ) (y₁ y₂ : ℝ) (h_k_pos : k > 0) 
  (h_A : y₁ = k / 2) (h_B : y₂ = k / 3) : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_relationship_l3037_303771


namespace NUMINAMATH_CALUDE_inequality_range_l3037_303775

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ 
  (-2 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3037_303775


namespace NUMINAMATH_CALUDE_f_odd_implies_a_zero_necessary_not_sufficient_l3037_303729

noncomputable def f (a x : ℝ) : ℝ := 1 / (x - 1) + a / (x + a - 1) + 1 / (x + 1)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem f_odd_implies_a_zero_necessary_not_sufficient :
  (∃ a : ℝ, is_odd_function (f a)) ∧
  (∀ a : ℝ, is_odd_function (f a) → a = 0 ∨ a = 1) ∧
  (∃ a : ℝ, a ≠ 0 ∧ is_odd_function (f a)) :=
sorry

end NUMINAMATH_CALUDE_f_odd_implies_a_zero_necessary_not_sufficient_l3037_303729


namespace NUMINAMATH_CALUDE_quadratic_function_ordering_l3037_303716

theorem quadratic_function_ordering (m y₁ y₂ y₃ : ℝ) : 
  m < -2 →
  y₁ = (m - 1)^2 - 2*(m - 1) →
  y₂ = m^2 - 2*m →
  y₃ = (m + 1)^2 - 2*(m + 1) →
  y₃ < y₂ ∧ y₂ < y₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_ordering_l3037_303716


namespace NUMINAMATH_CALUDE_a_equals_one_l3037_303779

theorem a_equals_one (a : ℝ) : 
  ((a - Complex.I) ^ 2 * Complex.I).re > 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_l3037_303779


namespace NUMINAMATH_CALUDE_z_purely_imaginary_iff_a_eq_neg_three_l3037_303732

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + 2*a - 3) (a - 1)

/-- Theorem stating that z is purely imaginary if and only if a = -3 -/
theorem z_purely_imaginary_iff_a_eq_neg_three (a : ℝ) :
  isPurelyImaginary (z a) ↔ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_iff_a_eq_neg_three_l3037_303732


namespace NUMINAMATH_CALUDE_smallest_board_is_7x7_l3037_303721

/-- Represents a ship in the Battleship game -/
structure Ship :=
  (length : Nat)

/-- The complete set of ships for the Battleship game -/
def battleshipSet : List Ship := [
  ⟨4⟩,  -- One 1x4 ship
  ⟨3⟩, ⟨3⟩,  -- Two 1x3 ships
  ⟨2⟩, ⟨2⟩, ⟨2⟩,  -- Three 1x2 ships
  ⟨1⟩, ⟨1⟩, ⟨1⟩, ⟨1⟩  -- Four 1x1 ships
]

/-- Represents a square board -/
structure Board :=
  (size : Nat)

/-- Checks if a given board can fit all ships without touching -/
def canFitShips (board : Board) (ships : List Ship) : Prop :=
  sorry

/-- Theorem stating that 7x7 is the smallest square board that can fit all ships -/
theorem smallest_board_is_7x7 :
  (∀ b : Board, b.size < 7 → ¬(canFitShips b battleshipSet)) ∧
  (canFitShips ⟨7⟩ battleshipSet) :=
sorry

end NUMINAMATH_CALUDE_smallest_board_is_7x7_l3037_303721


namespace NUMINAMATH_CALUDE_stating_cubic_factorization_condition_l3037_303798

/-- Represents a cubic equation of the form x^3 + ax^2 + bx + c = 0 -/
structure CubicEquation (α : Type) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the factored form (x^2 + m)(x + n) = 0 -/
structure FactoredForm (α : Type) [Field α] where
  m : α
  n : α

/-- 
Theorem stating the necessary and sufficient condition for a cubic equation 
to be factored into the given form
-/
theorem cubic_factorization_condition {α : Type} [Field α] (eq : CubicEquation α) :
  (∃ (ff : FactoredForm α), 
    ∀ (x : α), x^3 + eq.a * x^2 + eq.b * x + eq.c = 0 ↔ 
    (x^2 + ff.m) * (x + ff.n) = 0) ↔ 
  eq.c = eq.a * eq.b :=
sorry

end NUMINAMATH_CALUDE_stating_cubic_factorization_condition_l3037_303798


namespace NUMINAMATH_CALUDE_train_distance_l3037_303778

/-- Calculates the distance traveled by a train given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a train traveling at 7 m/s for 6 seconds covers 42 meters -/
theorem train_distance : distance_traveled 7 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l3037_303778


namespace NUMINAMATH_CALUDE_shoes_cost_theorem_l3037_303765

theorem shoes_cost_theorem (cost_first_pair : ℝ) (percentage_increase : ℝ) : 
  cost_first_pair = 22 →
  percentage_increase = 50 →
  let cost_second_pair := cost_first_pair * (1 + percentage_increase / 100)
  let total_cost := cost_first_pair + cost_second_pair
  total_cost = 55 := by
sorry

end NUMINAMATH_CALUDE_shoes_cost_theorem_l3037_303765
