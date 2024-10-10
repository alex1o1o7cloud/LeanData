import Mathlib

namespace prob_same_tails_value_l2550_255058

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting tails on a single penny toss -/
def prob_tails : ℚ := 1/2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes (same number of tails) -/
def favorable_outcomes : ℕ := 3

/-- The probability of Ephraim getting the same number of tails as Keiko -/
def prob_same_tails : ℚ := favorable_outcomes / total_outcomes

theorem prob_same_tails_value : prob_same_tails = 3/32 := by sorry

end prob_same_tails_value_l2550_255058


namespace factors_of_60_l2550_255014

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end factors_of_60_l2550_255014


namespace exists_100_same_polygons_l2550_255062

/-- Represents a convex polygon --/
structure ConvexPolygon where
  vertices : ℕ

/-- Represents the state of the paper after some cuts --/
structure PaperState where
  polygons : List ConvexPolygon

/-- A function that performs a single cut --/
def cut (state : PaperState) : PaperState :=
  sorry

/-- A function that checks if there are 100 polygons with the same number of vertices --/
def has_100_same_polygons (state : PaperState) : Bool :=
  sorry

/-- The main theorem --/
theorem exists_100_same_polygons :
  ∃ (n : ℕ), ∀ (initial : PaperState),
    has_100_same_polygons (n.iterate cut initial) = true :=
  sorry

end exists_100_same_polygons_l2550_255062


namespace paint_remaining_l2550_255075

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  let remaining_after_day1 := initial_paint - (1/4 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/2 * remaining_after_day1)
  remaining_after_day2 = 3/8 * initial_paint := by
  sorry

end paint_remaining_l2550_255075


namespace construct_triangle_from_equilateral_vertices_l2550_255065

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop :=
  sorry

/-- Main theorem: Given an acute-angled triangle A₁B₁C₁, there exists a unique triangle ABC
    such that A₁, B₁, and C₁ are the vertices of equilateral triangles drawn outward
    on the sides BC, CA, and AB respectively -/
theorem construct_triangle_from_equilateral_vertices
  (A₁ B₁ C₁ : Point) (h : isAcute (Triangle.mk A₁ B₁ C₁)) :
  ∃! (ABC : Triangle),
    isEquilateral ABC.B ABC.C A₁ ∧
    isEquilateral ABC.C ABC.A B₁ ∧
    isEquilateral ABC.A ABC.B C₁ :=
  sorry

end construct_triangle_from_equilateral_vertices_l2550_255065


namespace infinitely_many_m_minus_f_eq_1989_l2550_255052

/-- The number of factors of 2 in m! -/
def f (m : ℕ) : ℕ := sorry

/-- Condition that 11 · 15m is a positive integer -/
def is_valid (m : ℕ) : Prop := 0 < 11 * 15 * m

/-- The main theorem -/
theorem infinitely_many_m_minus_f_eq_1989 :
  ∀ n : ℕ, ∃ m > n, is_valid m ∧ m - f m = 1989 := by sorry

end infinitely_many_m_minus_f_eq_1989_l2550_255052


namespace triangle_angle_measure_l2550_255038

theorem triangle_angle_measure (A B C : Real) (a b c : Real) (S : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- A = 2B
  (A = 2 * B) →
  -- Area S = a²/4
  (S = a^2 / 4) →
  -- Area formula
  (S = (1/2) * b * c * Real.sin A) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion: A is either π/2 or π/4
  (A = π/2 ∨ A = π/4) :=
by sorry

end triangle_angle_measure_l2550_255038


namespace equation_solution_l2550_255002

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (12 + 3*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0 := by
  sorry

end equation_solution_l2550_255002


namespace board_covering_l2550_255068

-- Define a function to check if a board can be covered by dominoes
def can_cover_board (n m k : ℕ+) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Bool), ∀ (i j : ℕ), i < n.val ∧ j < m.val →
    (arrangement i j = true ∧ arrangement (i + 1) j = true ∧ i + 1 < n.val) ∨
    (arrangement i j = true ∧ arrangement i (j + 1) = true ∧ j + 1 < m.val)

-- State the theorem
theorem board_covering (n m k : ℕ+) :
  can_cover_board n m k ↔ (k.val ∣ n.val ∨ k.val ∣ m.val) :=
sorry

end board_covering_l2550_255068


namespace max_jogs_is_six_l2550_255003

/-- Represents the quantity of each item Bill can buy --/
structure Purchase where
  jags : Nat
  jigs : Nat
  jogs : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Nat :=
  p.jags * 1 + p.jigs * 2 + p.jogs * 7

/-- Checks if a purchase satisfies all conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ totalCost p = 50

/-- Theorem stating that the maximum number of jogs Bill can buy is 6 --/
theorem max_jogs_is_six :
  (∃ p : Purchase, isValidPurchase p ∧ p.jogs = 6) ∧
  (∀ p : Purchase, isValidPurchase p → p.jogs ≤ 6) :=
sorry

end max_jogs_is_six_l2550_255003


namespace weight_order_l2550_255087

-- Define the weights as real numbers
variable (B S C K : ℝ)

-- State the given conditions
axiom suitcase_heavier : S > B
axiom satchel_backpack_heavier : C + B > S + K
axiom basket_satchel_equal_suitcase_backpack : K + C = S + B

-- Theorem to prove
theorem weight_order : C > S ∧ S > B ∧ B > K := by sorry

end weight_order_l2550_255087


namespace f_increasing_and_odd_l2550_255082

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - 5^(-x) else 5^x - 1

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
by sorry

end f_increasing_and_odd_l2550_255082


namespace farmer_land_ownership_l2550_255004

/-- Proves that a farmer owns 5000 acres of land given the described land usage -/
theorem farmer_land_ownership : ∀ (total_land : ℝ),
  (0.9 * total_land * 0.1 + 0.9 * total_land * 0.8 + 450 = 0.9 * total_land) →
  total_land = 5000 := by
  sorry

end farmer_land_ownership_l2550_255004


namespace zeros_before_first_nonzero_eq_five_l2550_255066

/-- The number of zeroes to the right of the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  let denominator := 2^3 * 5^6
  let decimal_places := 6  -- log_10(denominator)
  decimal_places - 1

theorem zeros_before_first_nonzero_eq_five :
  zeros_before_first_nonzero = 5 := by
  sorry

end zeros_before_first_nonzero_eq_five_l2550_255066


namespace high_school_math_club_payment_l2550_255057

theorem high_school_math_club_payment (A : Nat) : 
  A < 10 → (2 * 100 + A * 10 + 3) % 3 = 0 → A = 1 ∨ A = 4 := by
  sorry

end high_school_math_club_payment_l2550_255057


namespace emilee_earnings_l2550_255079

/-- Given the earnings of three people with specific conditions, prove Emilee's earnings. -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  ∃ emilee_earnings : ℕ,
    emilee_earnings = total - (terrence_earnings + (terrence_earnings + jermaine_extra)) ∧
    emilee_earnings = 25 :=
by sorry

end emilee_earnings_l2550_255079


namespace choose_four_from_nine_l2550_255009

theorem choose_four_from_nine :
  Nat.choose 9 4 = 126 := by
  sorry

end choose_four_from_nine_l2550_255009


namespace sum_and_count_theorem_l2550_255055

def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sumIntegers 30 40
  let y := countEvenIntegers 30 40
  x + y = 391 := by sorry

end sum_and_count_theorem_l2550_255055


namespace score_three_points_count_l2550_255041

/-- Represents the number of items to be matched -/
def n : ℕ := 4

/-- Represents the number of points awarded for a correct match -/
def correct_points : ℕ := 3

/-- Represents the number of points awarded for an incorrect match -/
def incorrect_points : ℕ := 0

/-- The total number of ways to match exactly one item correctly and the rest incorrectly -/
def ways_to_score_three_points : ℕ := n

theorem score_three_points_count :
  ways_to_score_three_points = n := by
  sorry

end score_three_points_count_l2550_255041


namespace difference_largest_smallest_l2550_255070

def digits : List Nat := [9, 3, 1, 2, 6, 4]

def max_occurrences : Nat := 2

def largest_number : Nat := 99664332211

def smallest_number : Nat := 1122334699

theorem difference_largest_smallest :
  largest_number - smallest_number = 98541997512 :=
by sorry

end difference_largest_smallest_l2550_255070


namespace regular_octagon_diagonal_length_l2550_255021

/-- The length of a diagonal in a regular octagon -/
theorem regular_octagon_diagonal_length :
  ∀ (side_length : ℝ),
  side_length > 0 →
  ∃ (diagonal_length : ℝ),
  diagonal_length = side_length * Real.sqrt (2 + Real.sqrt 2) ∧
  diagonal_length^2 = 2 * side_length^2 + side_length^2 * Real.sqrt 2 :=
by
  sorry

end regular_octagon_diagonal_length_l2550_255021


namespace hen_count_l2550_255098

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48)
  (h2 : total_feet = 136)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) : 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 28 := by
  sorry

end hen_count_l2550_255098


namespace sum_of_first_n_natural_numbers_l2550_255008

theorem sum_of_first_n_natural_numbers (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ k < 10 ∧ n * (n + 1) / 2 = 111 * k) ↔ n = 36 := by
  sorry

end sum_of_first_n_natural_numbers_l2550_255008


namespace finite_moves_l2550_255059

/-- Represents the position of a number after m minutes -/
def position (initial_pos : ℕ) (m : ℕ) : ℕ :=
  if m ∣ initial_pos then initial_pos + m - 1 else initial_pos - 1

/-- Represents whether a number at initial_pos has moved after m minutes -/
def has_moved (initial_pos : ℕ) (m : ℕ) : Prop :=
  position initial_pos m ≠ initial_pos

/-- The main theorem stating that each natural number moves only finitely many times -/
theorem finite_moves (n : ℕ) : ∃ (M : ℕ), ∀ (m : ℕ), m ≥ M → ¬(has_moved n m) := by
  sorry


end finite_moves_l2550_255059


namespace min_sum_distances_to_lines_l2550_255023

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : 
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line1 := {P : ℝ × ℝ | 4 * P.1 - 3 * P.2 + 6 = 0}
  let line2 := {P : ℝ × ℝ | P.1 = -1}
  let dist_to_line1 (P : ℝ × ℝ) := |4 * P.1 - 3 * P.2 + 6| / Real.sqrt (4^2 + (-3)^2)
  let dist_to_line2 (P : ℝ × ℝ) := |P.1 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ parabola → 
      dist_to_line1 P + dist_to_line2 P ≥ min_dist :=
by sorry

end min_sum_distances_to_lines_l2550_255023


namespace shopping_mall_goods_problem_l2550_255074

/-- Shopping mall goods problem -/
theorem shopping_mall_goods_problem 
  (total_cost_A : ℝ) 
  (total_cost_B : ℝ) 
  (cost_diff : ℝ) 
  (selling_price_A : ℝ) 
  (selling_price_B : ℝ) 
  (discount_rate : ℝ) 
  (min_profit : ℝ)
  (h1 : total_cost_A = 2000)
  (h2 : total_cost_B = 2400)
  (h3 : cost_diff = 8)
  (h4 : selling_price_A = 60)
  (h5 : selling_price_B = 88)
  (h6 : discount_rate = 0.3)
  (h7 : min_profit = 2460)
  : ∃ (cost_price_A cost_price_B : ℝ) (min_units_A : ℕ),
    cost_price_A = 40 ∧ 
    cost_price_B = 48 ∧ 
    min_units_A = 20 ∧
    (total_cost_A / cost_price_A = total_cost_B / cost_price_B) ∧
    (selling_price_A - cost_price_A) * min_units_A + 
    (selling_price_A * (1 - discount_rate) - cost_price_A) * (total_cost_A / cost_price_A - min_units_A) + 
    (selling_price_B - cost_price_B) * (total_cost_B / cost_price_B) ≥ min_profit :=
by sorry

end shopping_mall_goods_problem_l2550_255074


namespace find_number_l2550_255005

theorem find_number : ∃ x : ℝ, 4.75 + 0.303 + x = 5.485 ∧ x = 0.432 := by
  sorry

end find_number_l2550_255005


namespace equation_solution_l2550_255073

theorem equation_solution : ∃! r : ℚ, (r^2 - 5*r + 4)/(r^2 - 8*r + 7) = (r^2 - 2*r - 15)/(r^2 - r - 20) ∧ r = -5/4 := by
  sorry

end equation_solution_l2550_255073


namespace equation_with_two_variables_degree_one_is_linear_l2550_255071

/-- Definition of a linear equation in two variables -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ) (c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- Theorem stating that an equation with two variables and terms of degree 1 is a linear equation in two variables -/
theorem equation_with_two_variables_degree_one_is_linear 
  (f : ℝ → ℝ → ℝ) 
  (h1 : ∃ (x y : ℝ), f x y ≠ f 0 0) -- Condition: contains two variables
  (h2 : ∀ (x y : ℝ), ∃ (a b : ℝ) (c : ℝ), f x y = a * x + b * y + c) -- Condition: terms with variables are of degree 1
  : is_linear_equation_in_two_variables f :=
sorry

end equation_with_two_variables_degree_one_is_linear_l2550_255071


namespace sqrt_of_neg_seven_squared_l2550_255042

theorem sqrt_of_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end sqrt_of_neg_seven_squared_l2550_255042


namespace vacant_seats_l2550_255085

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 60 / 100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end vacant_seats_l2550_255085


namespace min_moves_to_black_l2550_255034

/-- Represents a chessboard with alternating colors -/
structure Chessboard :=
  (size : Nat)
  (alternating : Bool)

/-- Represents a move on the chessboard -/
structure Move :=
  (top_left : Nat × Nat)
  (bottom_right : Nat × Nat)

/-- Function to apply a move to a chessboard -/
def apply_move (board : Chessboard) (move : Move) : Chessboard := sorry

/-- Function to check if all squares are black -/
def all_black (board : Chessboard) : Bool := sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_black (board : Chessboard) :
  board.size = 98 ∧ board.alternating →
  (∃ (moves : List Move), all_black (moves.foldl apply_move board) ∧ moves.length = 98) ∧
  (∀ (moves : List Move), all_black (moves.foldl apply_move board) → moves.length ≥ 98) :=
sorry

end min_moves_to_black_l2550_255034


namespace range_of_fraction_l2550_255081

theorem range_of_fraction (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 ∧ ∃ (a' b' : ℝ), b' > a' ∧ a' > 0 ∧ a' * b' = 2 ∧ (a'^2 + b'^2) / (a' - b') = -4 :=
sorry

end range_of_fraction_l2550_255081


namespace tangent_line_proof_l2550_255015

noncomputable def f (x : ℝ) : ℝ := -(1/2) * x + Real.log x

theorem tangent_line_proof :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  f x₀ = (1/2) * x₀ - 1 ∧
  deriv f x₀ = 1/2 :=
sorry

end tangent_line_proof_l2550_255015


namespace equation_satisfied_l2550_255094

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) :
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end equation_satisfied_l2550_255094


namespace no_solution_equations_l2550_255048

theorem no_solution_equations :
  (∀ x : ℝ, (|2*x| + 7 ≠ 0)) ∧
  (∀ x : ℝ, (Real.sqrt (3*x) + 2 ≠ 0)) ∧
  (∃ x : ℝ, ((x - 5)^2 = 0)) ∧
  (∃ x : ℝ, (Real.cos x - 1 = 0)) ∧
  (∃ x : ℝ, (|x| - 3 = 0)) := by
  sorry

end no_solution_equations_l2550_255048


namespace pigeonhole_principle_on_sequence_l2550_255028

theorem pigeonhole_principle_on_sequence (n : ℕ) : 
  ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 2*n ∧ (i + i) % (2*n) = (j + j) % (2*n) := by
  sorry

end pigeonhole_principle_on_sequence_l2550_255028


namespace conic_section_eccentricity_l2550_255025

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 5 * (16/5)) → 
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
   ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
   ((x : ℝ) → (y : ℝ) → x^2 / m + y^2 = 1 → 
    (e = c / a ∧ ((m > 0 → a^2 - b^2 = c^2) ∧ (m < 0 → b^2 - a^2 = c^2))))) :=
by sorry

end conic_section_eccentricity_l2550_255025


namespace fraction_inequality_l2550_255012

theorem fraction_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x^2 + 3*x + 2) / (x^2 + x - 6) ≠ (x + 2) / (x - 2) :=
by sorry

end fraction_inequality_l2550_255012


namespace jamal_shelving_problem_l2550_255080

/-- The number of books Jamal still has to shelve after working through different sections of the library. -/
def books_to_shelve (initial : ℕ) (history : ℕ) (fiction : ℕ) (children : ℕ) (misplaced : ℕ) : ℕ :=
  initial - history - fiction - children + misplaced

/-- Theorem stating that Jamal has 16 books left to shelve given the specific numbers from the problem. -/
theorem jamal_shelving_problem :
  books_to_shelve 51 12 19 8 4 = 16 := by
  sorry

end jamal_shelving_problem_l2550_255080


namespace expected_rounds_is_correct_l2550_255084

/-- Represents the number of rounds in the ball-drawing experiment -/
inductive Round : Type
  | one : Round
  | two : Round
  | three : Round

/-- The probability distribution of the number of rounds -/
def prob (r : Round) : ℚ :=
  match r with
  | Round.one => 1/4
  | Round.two => 1/12
  | Round.three => 2/3

/-- The expected number of rounds -/
def expected_rounds : ℚ := 29/12

/-- Theorem stating that the expected number of rounds is 29/12 -/
theorem expected_rounds_is_correct :
  (prob Round.one * 1 + prob Round.two * 2 + prob Round.three * 3 : ℚ) = expected_rounds := by
  sorry


end expected_rounds_is_correct_l2550_255084


namespace smallest_area_is_40_l2550_255016

/-- A rectangle with even side lengths that can be divided into squares and dominoes -/
structure CheckeredRectangle where
  width : Nat
  height : Nat
  has_square : Bool
  has_domino : Bool
  width_even : Even width
  height_even : Even height
  both_types : has_square ∧ has_domino

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : Nat :=
  r.width * r.height

/-- Theorem stating the smallest possible area of a valid CheckeredRectangle is 40 -/
theorem smallest_area_is_40 :
  ∀ r : CheckeredRectangle, area r ≥ 40 ∧ ∃ r' : CheckeredRectangle, area r' = 40 := by
  sorry

end smallest_area_is_40_l2550_255016


namespace planes_perpendicular_l2550_255072

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m l : Line) (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : parallel m l) 
  (h4 : perpendicular l β) 
  (h5 : contains α m) : 
  perpendicularPlanes α β :=
sorry

end planes_perpendicular_l2550_255072


namespace walkway_time_when_stopped_l2550_255054

/-- The time it takes to walk a moving walkway when it's stopped -/
theorem walkway_time_when_stopped 
  (length : ℝ) 
  (time_with : ℝ) 
  (time_against : ℝ) 
  (h1 : length = 60) 
  (h2 : time_with = 30) 
  (h3 : time_against = 120) : 
  (2 * length) / (length / time_with + length / time_against) = 48 := by
  sorry

end walkway_time_when_stopped_l2550_255054


namespace inequality_proof_l2550_255091

theorem inequality_proof (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / Real.sqrt (x^4 + x^2 + 1) + y / Real.sqrt (y^4 + y^2 + 1) + z / Real.sqrt (z^4 + z^2 + 1) ≥ -1 / Real.sqrt 3 := by
  sorry

end inequality_proof_l2550_255091


namespace polynomial_inequality_l2550_255006

/-- A polynomial with positive real coefficients -/
structure PositivePolynomial where
  coeffs : List ℝ
  all_positive : ∀ c ∈ coeffs, c > 0

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : PositivePolynomial) (x : ℝ) : ℝ :=
  p.coeffs.enum.foldl (λ acc (i, a) => acc + a * x ^ i) 0

/-- The main theorem -/
theorem polynomial_inequality (p : PositivePolynomial) :
  (evalPoly p 1 ≥ 1 / evalPoly p 1) →
  (∀ x : ℝ, x > 0 → evalPoly p (1/x) ≥ 1 / evalPoly p x) :=
sorry

end polynomial_inequality_l2550_255006


namespace smallest_n_for_shared_vertex_triangles_l2550_255024

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for a monochromatic triangle in a two-colored complete graph -/
def MonochromaticTriangle (n : ℕ) (c : TwoColoring n) (v₁ v₂ v₃ : Fin n) : Prop :=
  v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧
  c v₁ v₂ = c v₂ v₃ ∧ c v₂ v₃ = c v₁ v₃

/-- Predicate for two monochromatic triangles sharing exactly one vertex -/
def SharedVertexTriangles (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ (v₁ v₂ v₃ v₄ v₅ : Fin n),
    MonochromaticTriangle n c v₁ v₂ v₃ ∧
    MonochromaticTriangle n c v₁ v₄ v₅ ∧
    v₂ ≠ v₄ ∧ v₂ ≠ v₅ ∧ v₃ ≠ v₄ ∧ v₃ ≠ v₅

/-- The main theorem: 9 is the smallest n such that any two-coloring of K_n contains two monochromatic triangles sharing exactly one vertex -/
theorem smallest_n_for_shared_vertex_triangles :
  (∀ c : TwoColoring 9, SharedVertexTriangles 9 c) ∧
  (∀ m : ℕ, m < 9 → ∃ c : TwoColoring m, ¬SharedVertexTriangles m c) :=
sorry

end smallest_n_for_shared_vertex_triangles_l2550_255024


namespace sine_cosine_identity_l2550_255032

theorem sine_cosine_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end sine_cosine_identity_l2550_255032


namespace least_number_with_special_property_l2550_255095

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop := sorry

/-- The least positive integer whose digits add to a multiple of 27 yet the number itself is not a multiple of 27 -/
theorem least_number_with_special_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), sum_of_digits n = 27 * k) ∧ 
  ¬(is_divisible_by n 27) ∧
  (∀ (m : ℕ), m < n → 
    ((∃ (k : ℕ), sum_of_digits m = 27 * k) → (is_divisible_by m 27))) :=
by sorry

end least_number_with_special_property_l2550_255095


namespace algebraic_expression_value_l2550_255060

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) :
  a^3 - 2*a^2*b + a*b^2 - 4*a = 0 := by sorry

end algebraic_expression_value_l2550_255060


namespace sum_of_roots_for_f_l2550_255088

theorem sum_of_roots_for_f (f : ℝ → ℝ) : 
  (∀ x, f (x / 4) = x^2 + 3*x + 2) →
  (∃ z₁ z₂, f (4*z₁) = 8 ∧ f (4*z₂) = 8 ∧ z₁ ≠ z₂ ∧ 
    (∀ z, f (4*z) = 8 → z = z₁ ∨ z = z₂) ∧
    z₁ + z₂ = -3/16) := by
  sorry

end sum_of_roots_for_f_l2550_255088


namespace basketball_weight_l2550_255020

theorem basketball_weight (basketball_weight bicycle_weight : ℝ) 
  (h1 : 9 * basketball_weight = 6 * bicycle_weight)
  (h2 : 4 * bicycle_weight = 120) : 
  basketball_weight = 20 := by
sorry

end basketball_weight_l2550_255020


namespace complex_fraction_equality_l2550_255037

theorem complex_fraction_equality : 
  let numerator := ((5 / 2) ^ 2 / (1 / 2) ^ 3) * (5 / 2) ^ 2
  let denominator := ((5 / 3) ^ 4 * (1 / 2) ^ 2) / (2 / 3) ^ 3
  numerator / denominator = 48 := by sorry

end complex_fraction_equality_l2550_255037


namespace sum_of_factors_l2550_255010

theorem sum_of_factors (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -120 →
  p + q + r + s + t = 27 := by
sorry

end sum_of_factors_l2550_255010


namespace quadratic_inequality_range_l2550_255096

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → a ∈ Set.Icc (-8) 0 := by
  sorry

end quadratic_inequality_range_l2550_255096


namespace max_missed_problems_l2550_255086

theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : pass_percentage = 85 / 100) : 
  ∃ (max_missed : ℕ), 
    (max_missed ≤ total_problems) ∧ 
    ((total_problems - max_missed : ℚ) / total_problems ≥ pass_percentage) ∧
    ∀ (n : ℕ), n > max_missed → 
      ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by
  sorry

end max_missed_problems_l2550_255086


namespace inverse_of_A_squared_l2550_255033

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 2, 1]

theorem inverse_of_A_squared :
  A⁻¹ = !![3, -1; 2, 1] →
  (A^2)⁻¹ = !![7, -4; 8, -1] := by sorry

end inverse_of_A_squared_l2550_255033


namespace peters_class_size_l2550_255097

/-- Represents the number of students with a specific number of hands -/
structure HandDistribution :=
  (hands : ℕ)
  (students : ℕ)

/-- Represents the class data -/
structure ClassData :=
  (total_hands : ℕ)
  (distribution : List HandDistribution)
  (unspecified_students : ℕ)

/-- Calculates the total number of students in Peter's class -/
def total_students (data : ClassData) : ℕ :=
  (data.distribution.map (λ d => d.students)).sum + data.unspecified_students + 1

/-- Theorem stating that the total number of students in Peter's class is 17 -/
theorem peters_class_size (data : ClassData) 
  (h1 : data.total_hands = 20)
  (h2 : data.distribution = [
    ⟨2, 7⟩, 
    ⟨1, 3⟩, 
    ⟨3, 1⟩, 
    ⟨0, 2⟩
  ])
  (h3 : data.unspecified_students = 3) :
  total_students data = 17 := by
  sorry

end peters_class_size_l2550_255097


namespace sue_votes_l2550_255031

/-- Given 1000 total votes and Sue receiving 35% of the votes, prove that Sue received 350 votes. -/
theorem sue_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  ↑total_votes * sue_percentage = 350 := by
  sorry

end sue_votes_l2550_255031


namespace election_combinations_l2550_255051

def number_of_students : ℕ := 6
def number_of_positions : ℕ := 3

theorem election_combinations :
  (number_of_students * (number_of_students - 1) * (number_of_students - 2) = 120) :=
by sorry

end election_combinations_l2550_255051


namespace reflection_of_A_across_x_axis_l2550_255017

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point A
def A : Point := (1, -2)

-- Define reflection across x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem to prove
theorem reflection_of_A_across_x_axis :
  reflect_x A = (1, 2) := by sorry

end reflection_of_A_across_x_axis_l2550_255017


namespace saree_price_l2550_255044

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 331.2)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.08) : 
  ∃ original_price : ℝ, 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
sorry

end saree_price_l2550_255044


namespace garden_perimeter_l2550_255093

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  24 * garden_length = 16 * 12 →
  2 * garden_length + 2 * 24 = 64 :=
by
  sorry

end garden_perimeter_l2550_255093


namespace arithmetic_sequence_squares_l2550_255069

theorem arithmetic_sequence_squares (k : ℤ) : k = 1612 →
  ∃ (a d : ℤ), 
    (25 + k = (a - d)^2) ∧ 
    (289 + k = a^2) ∧ 
    (529 + k = (a + d)^2) := by
  sorry

end arithmetic_sequence_squares_l2550_255069


namespace sin_15_plus_cos_15_l2550_255035

theorem sin_15_plus_cos_15 : Real.sin (15 * π / 180) + Real.cos (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end sin_15_plus_cos_15_l2550_255035


namespace absolute_value_expression_l2550_255029

theorem absolute_value_expression (x : ℝ) (h : x < 0) : 
  |x - 3 * Real.sqrt ((x - 2)^2)| = 6 - 4*x := by
  sorry

end absolute_value_expression_l2550_255029


namespace trigonometric_inequality_l2550_255045

theorem trigonometric_inequality (θ : Real) (h : 0 < θ ∧ θ < π/4) :
  3 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 3 * Real.tan θ + 2 * Real.sin θ ≥ 4 * (3 * Real.sqrt 3) ^ (1/4) := by
  sorry

end trigonometric_inequality_l2550_255045


namespace inscribed_quadrilateral_fourth_side_l2550_255022

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (a b c d : ℝ) 
  (θ : ℝ) 
  (h1 : r = 150 * Real.sqrt 2)
  (h2 : a = 150 ∧ b = 150 ∧ c = 150)
  (h3 : θ = 120 * π / 180) : 
  d = 375 * Real.sqrt 2 := by
  sorry

end inscribed_quadrilateral_fourth_side_l2550_255022


namespace line_properties_l2550_255001

-- Define the lines
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0
def l2 (a x y : ℝ) : Prop := a * x + y = 1

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop := Real.sqrt 3 * a + 1 = 0

-- Define angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := θ = 2 * Real.pi / 3

-- Define distance from origin to line
def distance_to_origin (a : ℝ) (d : ℝ) : Prop := 
  d = 1 / Real.sqrt (a^2 + 1)

theorem line_properties (a : ℝ) :
  perpendicular a →
  angle_of_inclination (Real.arctan (-Real.sqrt 3)) ∧
  distance_to_origin a (Real.sqrt 3 / 2) :=
by sorry

end line_properties_l2550_255001


namespace train_optimization_l2550_255027

/-- Represents the relationship between carriages and round trips -/
def round_trips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total number of passengers transported per day -/
def passengers (x : ℝ) : ℝ := 110 * x * round_trips x

/-- The optimal number of carriages -/
def optimal_carriages : ℝ := 6

/-- The optimal number of round trips -/
def optimal_trips : ℝ := round_trips optimal_carriages

/-- The maximum number of passengers per day -/
def max_passengers : ℝ := passengers optimal_carriages

theorem train_optimization :
  (round_trips 4 = 16) →
  (round_trips 7 = 10) →
  (∀ x, round_trips x = -2 * x + 24) →
  (optimal_carriages = 6) →
  (optimal_trips = 12) →
  (max_passengers = 7920) →
  (∀ x, passengers x ≤ max_passengers) :=
by sorry

end train_optimization_l2550_255027


namespace only_cooking_count_l2550_255064

/-- Given information about curriculum participation --/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ
  cooking_and_weaving : ℕ

/-- Theorem stating the number of people who study only cooking --/
theorem only_cooking_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_and_yoga = 7)
  (h5 : cp.all_curriculums = 3)
  (h6 : cp.cooking_and_weaving = 3) :
  cp.cooking - (cp.cooking_and_yoga - cp.all_curriculums) - (cp.cooking_and_weaving - cp.all_curriculums) - cp.all_curriculums = 8 :=
by sorry

end only_cooking_count_l2550_255064


namespace grandfather_money_calculation_l2550_255040

def birthday_money_problem (aunt_money grandfather_money total_money bank_money : ℕ) : Prop :=
  aunt_money = 75 ∧
  bank_money = 45 ∧
  bank_money = total_money / 5 ∧
  total_money = aunt_money + grandfather_money

theorem grandfather_money_calculation :
  ∀ aunt_money grandfather_money total_money bank_money,
  birthday_money_problem aunt_money grandfather_money total_money bank_money →
  grandfather_money = 150 :=
by
  sorry

end grandfather_money_calculation_l2550_255040


namespace middle_integer_of_pairwise_sums_l2550_255007

theorem middle_integer_of_pairwise_sums (x y z : ℤ) 
  (h1 : x < y) (h2 : y < z)
  (sum_xy : x + y = 22)
  (sum_xz : x + z = 24)
  (sum_yz : y + z = 16) :
  y = 7 := by
sorry

end middle_integer_of_pairwise_sums_l2550_255007


namespace history_paper_pages_l2550_255077

/-- Calculates the total number of pages in a paper given the number of days and pages per day -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days with 21 pages written per day has 63 pages in total -/
theorem history_paper_pages : total_pages 3 21 = 63 := by
  sorry

end history_paper_pages_l2550_255077


namespace max_edges_no_triangle_max_edges_no_K4_l2550_255099

/-- The Turán number T(n, r) is the maximum number of edges in a graph with n vertices that does not contain a complete subgraph of r+1 vertices. -/
def turan_number (n : ℕ) (r : ℕ) : ℕ := sorry

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  edges : Finset (Fin n × Fin n)

/-- The number of edges in a graph -/
def num_edges {n : ℕ} (G : Graph n) : ℕ := G.edges.card

/-- A graph contains a triangle if it has a complete subgraph of 3 vertices -/
def has_triangle {n : ℕ} (G : Graph n) : Prop := sorry

/-- A graph contains a K4 if it has a complete subgraph of 4 vertices -/
def has_K4 {n : ℕ} (G : Graph n) : Prop := sorry

theorem max_edges_no_triangle (G : Graph 30) :
  ¬has_triangle G → num_edges G ≤ 225 :=
sorry

theorem max_edges_no_K4 (G : Graph 30) :
  ¬has_K4 G → num_edges G ≤ 200 :=
sorry

end max_edges_no_triangle_max_edges_no_K4_l2550_255099


namespace regular_polygon_perimeter_l2550_255053

/-- A regular polygon with side length 7 and exterior angle 72 degrees has a perimeter of 35 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 72 ∧
  n * exterior_angle = 360 →
  n * side_length = 35 := by
sorry

end regular_polygon_perimeter_l2550_255053


namespace democrats_to_participants_ratio_l2550_255046

/-- Proof of the ratio of democrats to total participants in a meeting --/
theorem democrats_to_participants_ratio :
  ∀ (total_participants : ℕ) 
    (female_democrats : ℕ) 
    (female_ratio : ℚ) 
    (male_ratio : ℚ),
  total_participants = 870 →
  female_democrats = 145 →
  female_ratio = 1/2 →
  male_ratio = 1/4 →
  (female_democrats * 2 + (total_participants - female_democrats * 2) * male_ratio) / total_participants = 1/3 :=
by
  sorry

#check democrats_to_participants_ratio

end democrats_to_participants_ratio_l2550_255046


namespace rectangle_enclosure_count_l2550_255018

theorem rectangle_enclosure_count :
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 5
  let choose_horizontal : ℕ := 2
  let choose_vertical : ℕ := 2
  (Nat.choose horizontal_lines choose_horizontal) * (Nat.choose vertical_lines choose_vertical) = 100 :=
by sorry

end rectangle_enclosure_count_l2550_255018


namespace toothpicks_for_ten_base_triangles_l2550_255030

/-- The number of toothpicks needed to construct a large equilateral triangle -/
def toothpicks_needed (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let shared_sides := (total_sides - 3 * base_triangles) / 2
  let boundary_sides := 3 * base_triangles
  shared_sides + boundary_sides

/-- Theorem stating that 98 toothpicks are needed for a large equilateral triangle with 10 small triangles on its base -/
theorem toothpicks_for_ten_base_triangles :
  toothpicks_needed 10 = 98 := by
  sorry


end toothpicks_for_ten_base_triangles_l2550_255030


namespace schur_inequality_special_case_l2550_255076

theorem schur_inequality_special_case (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := by
  sorry

end schur_inequality_special_case_l2550_255076


namespace parabola_point_ordering_l2550_255036

/-- Represents a parabola of the form y = -x^2 + 2x + c -/
def Parabola (c : ℝ) := {p : ℝ × ℝ | p.2 = -p.1^2 + 2*p.1 + c}

theorem parabola_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : (0, y₁) ∈ Parabola c)
  (h₂ : (1, y₂) ∈ Parabola c)
  (h₃ : (3, y₃) ∈ Parabola c) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end parabola_point_ordering_l2550_255036


namespace trigonometric_equation_solution_l2550_255011

theorem trigonometric_equation_solution (x : Real) :
  (2 * Real.sin (17 * x) + Real.sqrt 3 * Real.cos (5 * x) + Real.sin (5 * x) = 0) ↔
  (∃ k : Int, x = π / 66 * (6 * k - 1) ∨ x = π / 18 * (3 * k + 2)) :=
by sorry

end trigonometric_equation_solution_l2550_255011


namespace inscribed_circle_radius_l2550_255090

theorem inscribed_circle_radius 
  (r : ℝ) 
  (α γ : ℝ) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.sin α * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = ((2 * Real.sqrt 10 - 5) / 5) * r :=
by sorry

end inscribed_circle_radius_l2550_255090


namespace triangle_side_calculation_l2550_255039

theorem triangle_side_calculation (A B C : Real) (a b : Real) : 
  B = π / 6 → -- 30° in radians
  C = 7 * π / 12 → -- 105° in radians
  A = π / 4 → -- 45° in radians (derived from B + C + A = π)
  a = 4 →
  b = 2 * Real.sqrt 2 :=
by sorry

end triangle_side_calculation_l2550_255039


namespace percentage_reduction_l2550_255043

theorem percentage_reduction (P : ℝ) : (85 * P / 100) - 11 = 23 → P = 40 := by
  sorry

end percentage_reduction_l2550_255043


namespace square_side_length_l2550_255083

theorem square_side_length 
  (total_width : ℕ) 
  (total_height : ℕ) 
  (r : ℕ) 
  (s : ℕ) :
  total_width = 3300 →
  total_height = 2000 →
  2 * r + s = total_height →
  2 * r + 3 * s = total_width →
  s = 650 := by
  sorry

end square_side_length_l2550_255083


namespace pure_imaginary_square_root_l2550_255092

theorem pure_imaginary_square_root (a : ℝ) : 
  let z : ℂ := (a - Complex.I) ^ 2
  (∃ (b : ℝ), z = Complex.I * b) → (a = 1 ∨ a = -1) := by
sorry

end pure_imaginary_square_root_l2550_255092


namespace prob_sum_18_three_dice_l2550_255000

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the target sum
def target_sum : ℕ := 18

-- Define the probability of rolling a specific number on a single die
def single_die_prob : ℚ := 1 / die_faces

-- Statement to prove
theorem prob_sum_18_three_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 216 := by sorry

end prob_sum_18_three_dice_l2550_255000


namespace always_quadratic_in_x_l2550_255056

theorem always_quadratic_in_x (k : ℝ) :
  ∃ a b c : ℝ, a ≠ 0 ∧
  ∀ x : ℝ, (k^2 + 1) * x^2 - (k * x - 8) - 1 = a * x^2 + b * x + c :=
by sorry

end always_quadratic_in_x_l2550_255056


namespace sum_of_real_solutions_l2550_255078

theorem sum_of_real_solutions (a : ℝ) (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 := by
  sorry

end sum_of_real_solutions_l2550_255078


namespace shaded_to_white_ratio_is_five_thirds_l2550_255050

/-- A square subdivided into smaller squares where vertices of inner squares 
    are at midpoints of sides of the next larger square -/
structure SubdividedSquare :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The ratio of shaded to white area in a subdivided square -/
def shaded_to_white_ratio (s : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5:3 -/
theorem shaded_to_white_ratio_is_five_thirds (s : SubdividedSquare) :
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end shaded_to_white_ratio_is_five_thirds_l2550_255050


namespace total_weight_is_600_l2550_255026

/-- Proves that the total weight of Verna, Sherry, Jake, and Laura is 600 pounds given the specified conditions. -/
theorem total_weight_is_600 (haley_weight : ℝ) (verna_weight : ℝ) (sherry_weight : ℝ) (jake_weight : ℝ) (laura_weight : ℝ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight = sherry_weight / 2 →
  jake_weight = 3/5 * (haley_weight + verna_weight) →
  laura_weight = sherry_weight - jake_weight →
  verna_weight + sherry_weight + jake_weight + laura_weight = 600 := by
  sorry

#check total_weight_is_600

end total_weight_is_600_l2550_255026


namespace remainder_1234567890_mod_99_l2550_255013

theorem remainder_1234567890_mod_99 : 1234567890 % 99 = 72 := by
  sorry

end remainder_1234567890_mod_99_l2550_255013


namespace small_room_four_painters_l2550_255049

/-- Represents the number of work-days required for a given number of painters to complete a room -/
def work_days (painters : ℕ) (room_size : ℝ) : ℝ := sorry

theorem small_room_four_painters 
  (large_room_size small_room_size : ℝ)
  (h1 : work_days 5 large_room_size = 2)
  (h2 : small_room_size = large_room_size / 2)
  : work_days 4 small_room_size = 1.25 := by sorry

end small_room_four_painters_l2550_255049


namespace product_of_sum_of_squares_l2550_255047

theorem product_of_sum_of_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end product_of_sum_of_squares_l2550_255047


namespace total_pets_l2550_255089

def num_dogs : ℕ := 2
def num_cats : ℕ := 3
def num_fish : ℕ := 2 * (num_dogs + num_cats)

theorem total_pets : num_dogs + num_cats + num_fish = 15 := by
  sorry

end total_pets_l2550_255089


namespace max_term_a_l2550_255019

def a (n : ℕ) : ℚ := n / (n^2 + 2020)

theorem max_term_a :
  ∀ k : ℕ, k ≠ 45 → a k ≤ a 45 := by sorry

end max_term_a_l2550_255019


namespace rectangle_area_is_six_l2550_255067

/-- Represents a square within the rectangle ABCD -/
structure Square where
  side_length : ℝ
  area : ℝ
  area_eq : area = side_length ^ 2

/-- The rectangle ABCD containing three squares -/
structure Rectangle where
  squares : Fin 3 → Square
  non_overlapping : ∀ i j, i ≠ j → (squares i).area + (squares j).area ≤ area
  shaded_square_area : (squares 0).area = 1
  area : ℝ

/-- The theorem stating that the area of rectangle ABCD is 6 square inches -/
theorem rectangle_area_is_six (rect : Rectangle) : rect.area = 6 := by
  sorry

end rectangle_area_is_six_l2550_255067


namespace equal_students_after_transfer_total_students_after_transfer_l2550_255063

/-- Represents a section in Grade 4 -/
inductive Section
| Diligence
| Industry

/-- The number of students in a section before the transfer -/
def students_before (s : Section) : ℕ :=
  match s with
  | Section.Diligence => 23
  | Section.Industry => sorry  -- We don't know this value

/-- The number of students transferred from Industry to Diligence -/
def transferred_students : ℕ := 2

/-- The number of students in a section after the transfer -/
def students_after (s : Section) : ℕ :=
  match s with
  | Section.Diligence => students_before Section.Diligence + transferred_students
  | Section.Industry => students_before Section.Industry - transferred_students

/-- Theorem stating that the sections have equal students after transfer -/
theorem equal_students_after_transfer :
  students_after Section.Diligence = students_after Section.Industry := by sorry

/-- The main theorem to prove -/
theorem total_students_after_transfer :
  students_after Section.Diligence + students_after Section.Industry = 50 := by sorry

end equal_students_after_transfer_total_students_after_transfer_l2550_255063


namespace correct_households_using_both_l2550_255061

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : Nat
  neither : Nat
  onlyA : Nat
  bothRatio : Nat
  /-- Proves that the number of households using both brands is 30 -/
  householdsUsingBoth : Nat

/-- The actual survey data -/
def actualSurvey : SoapSurvey := {
  total := 260
  neither := 80
  onlyA := 60
  bothRatio := 3
  householdsUsingBoth := 30
}

/-- Theorem stating that the number of households using both brands is correct -/
theorem correct_households_using_both (s : SoapSurvey) : 
  s.householdsUsingBoth = 30 ∧ 
  s.total = s.neither + s.onlyA + s.householdsUsingBoth + s.bothRatio * s.householdsUsingBoth :=
by sorry

end correct_households_using_both_l2550_255061
