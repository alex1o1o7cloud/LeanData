import Mathlib

namespace NUMINAMATH_CALUDE_linda_spent_680_l1900_190056

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℚ :=
  let notebook_price : ℚ := 1.20
  let notebook_quantity : ℕ := 3
  let pencil_box_price : ℚ := 1.50
  let pen_box_price : ℚ := 1.70
  notebook_price * notebook_quantity + pencil_box_price + pen_box_price

theorem linda_spent_680 : linda_total_spent = 6.80 := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_680_l1900_190056


namespace NUMINAMATH_CALUDE_equidistant_line_equation_l1900_190033

/-- A line passing through point (3,4) and equidistant from points (-2,2) and (4,-2) -/
structure EquidistantLine where
  -- The line passes through point (3,4)
  passes_through : ℝ → ℝ → Prop
  -- The line is equidistant from points (-2,2) and (4,-2)
  equidistant : ℝ → ℝ → Prop

/-- The equation of the line is either 2x-y-2=0 or 2x+3y-18=0 -/
def line_equation (l : EquidistantLine) : Prop :=
  (∀ x y, l.passes_through x y ∧ l.equidistant x y → 2*x - y - 2 = 0) ∨
  (∀ x y, l.passes_through x y ∧ l.equidistant x y → 2*x + 3*y - 18 = 0)

theorem equidistant_line_equation (l : EquidistantLine) : line_equation l := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_equation_l1900_190033


namespace NUMINAMATH_CALUDE_bowen_spent_twenty_dollars_l1900_190099

/-- The price of a pencil in dollars -/
def pencil_price : ℚ := 25 / 100

/-- The price of a pen in dollars -/
def pen_price : ℚ := 15 / 100

/-- The number of pens Bowen buys -/
def num_pens : ℕ := 40

/-- The number of pencils Bowen buys -/
def num_pencils : ℕ := num_pens + (2 * num_pens) / 5

/-- The total amount Bowen spends in dollars -/
def total_spent : ℚ := num_pens * pen_price + num_pencils * pencil_price

theorem bowen_spent_twenty_dollars : total_spent = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowen_spent_twenty_dollars_l1900_190099


namespace NUMINAMATH_CALUDE_symmetry_axis_l1900_190061

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) f :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l1900_190061


namespace NUMINAMATH_CALUDE_karl_savings_l1900_190006

def folder_cost : ℝ := 3.50
def num_folders : ℕ := 10
def base_discount : ℝ := 0.15
def bulk_discount : ℝ := 0.05
def total_discount : ℝ := base_discount + bulk_discount

theorem karl_savings : 
  num_folders * folder_cost - num_folders * folder_cost * (1 - total_discount) = 7 :=
by sorry

end NUMINAMATH_CALUDE_karl_savings_l1900_190006


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l1900_190026

theorem complex_roots_quadratic (a b : ℝ) : 
  (Complex.mk a 3) ^ 2 - (Complex.mk 12 9) * (Complex.mk a 3) + (Complex.mk 15 65) = 0 ∧
  (Complex.mk b 6) ^ 2 - (Complex.mk 12 9) * (Complex.mk b 6) + (Complex.mk 15 65) = 0 →
  a = 7 / 3 ∧ b = 29 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l1900_190026


namespace NUMINAMATH_CALUDE_right_trapezoid_area_l1900_190030

/-- The area of a right trapezoid with specific dimensions -/
theorem right_trapezoid_area (upper_base lower_base height : ℝ) 
  (h1 : upper_base = 25)
  (h2 : lower_base - 15 = height)
  (h3 : height > 0) : 
  (upper_base + lower_base) * height / 2 = 175 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_area_l1900_190030


namespace NUMINAMATH_CALUDE_proposition_and_variants_l1900_190049

theorem proposition_and_variants (a b : ℝ) :
  (a > b → a * |a| > b * |b|) ∧
  (a * |a| > b * |b| → a > b) ∧
  (a ≤ b → a * |a| ≤ b * |b|) ∧
  (a * |a| ≤ b * |b| → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_proposition_and_variants_l1900_190049


namespace NUMINAMATH_CALUDE_problem_statement_l1900_190045

theorem problem_statement :
  -- Statement 1
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) ∧
  -- Statement 2
  (∀ p q : Prop, (p ∧ q) → (p ∨ q)) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(p ∧ q)) ∧
  -- Statement 4 (negation)
  ¬(∀ A B C D : Set α, (A ∪ B = A ∧ C ∩ D = C) → (A ⊆ B ∧ C ⊆ D)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1900_190045


namespace NUMINAMATH_CALUDE_puppies_theorem_l1900_190091

/-- The number of puppies Alyssa gave away -/
def alyssa_gave : ℕ := 20

/-- The number of puppies Alyssa kept -/
def alyssa_kept : ℕ := 8

/-- The number of puppies Bella gave away -/
def bella_gave : ℕ := 10

/-- The number of puppies Bella kept -/
def bella_kept : ℕ := 6

/-- The total number of puppies Alyssa and Bella had to start with -/
def total_puppies : ℕ := alyssa_gave + alyssa_kept + bella_gave + bella_kept

theorem puppies_theorem : total_puppies = 44 := by
  sorry

end NUMINAMATH_CALUDE_puppies_theorem_l1900_190091


namespace NUMINAMATH_CALUDE_tan_sum_special_l1900_190037

theorem tan_sum_special : Real.tan (17 * π / 180) + Real.tan (28 * π / 180) + Real.tan (17 * π / 180) * Real.tan (28 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l1900_190037


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_360_l1900_190005

/-- The number of perfect square factors of 360 -/
def perfect_square_factors_360 : ℕ :=
  let prime_factorization := (2, 3, 3, 2, 5, 1)
  4

theorem count_perfect_square_factors_360 :
  perfect_square_factors_360 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_360_l1900_190005


namespace NUMINAMATH_CALUDE_max_queens_on_chessboard_l1900_190095

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def on_same_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens attack each other -/
def queens_attack (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ on_same_diagonal p1 p2

/-- A valid placement of queens on the chessboard -/
structure QueenPlacement :=
  (black : List Position)
  (white : List Position)
  (black_valid : black.length ≤ 8)
  (white_valid : white.length ≤ 8)
  (no_attack : ∀ b ∈ black, ∀ w ∈ white, ¬queens_attack b w)

/-- The theorem to be proved -/
theorem max_queens_on_chessboard :
  ∃ (placement : QueenPlacement), placement.black.length = 8 ∧ placement.white.length = 8 ∧
  ∀ (other : QueenPlacement), other.black.length ≤ 8 ∧ other.white.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_queens_on_chessboard_l1900_190095


namespace NUMINAMATH_CALUDE_tournament_ranking_sequences_l1900_190036

/-- Represents a team in the tournament -/
inductive Team
| E | F | G | H | I | J | K

/-- Represents a match between two teams -/
structure Match where
  team1 : Team
  team2 : Team

/-- Represents the tournament structure -/
structure Tournament where
  preliminary_matches : List Match
  semifinal_matches : List Match
  final_match : Match
  third_place_match : Match

/-- Represents a possible ranking sequence of four teams -/
structure RankingSequence where
  first : Team
  second : Team
  third : Team
  fourth : Team

/-- The main theorem to prove -/
theorem tournament_ranking_sequences (t : Tournament) :
  (t.preliminary_matches.length = 3) →
  (t.semifinal_matches.length = 2) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.E ∧ m.team2 = Team.F) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.G ∧ m.team2 = Team.H) →
  (∃ m ∈ t.preliminary_matches, m.team1 = Team.I ∧ m.team2 = Team.J) →
  (∃ m ∈ t.semifinal_matches, m.team2 = Team.K) →
  (∃ ranking_sequences : List RankingSequence,
    ranking_sequences.length = 16 ∧
    ∀ rs ∈ ranking_sequences,
      (rs.first ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.second ∈ [t.final_match.team1, t.final_match.team2]) ∧
      (rs.third ∈ [t.third_place_match.team1, t.third_place_match.team2]) ∧
      (rs.fourth ∈ [t.third_place_match.team1, t.third_place_match.team2])) :=
by
  sorry

end NUMINAMATH_CALUDE_tournament_ranking_sequences_l1900_190036


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1900_190010

/-- Given a hyperbola and a parabola with specific properties, prove that n = 12 -/
theorem hyperbola_parabola_intersection (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), x^2/m - y^2/n = 1) →  -- hyperbola equation
  (∃ (e : ℝ), e = 2) →  -- eccentricity is 2
  (∃ (x y : ℝ), y^2 = 4*m*x) →  -- parabola equation
  (∃ (c : ℝ), c = m) →  -- focus of hyperbola coincides with focus of parabola
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1900_190010


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l1900_190058

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h_surface : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h_circumference : 2 * π * (r₁ + r₂) = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l1900_190058


namespace NUMINAMATH_CALUDE_rectangles_on_specific_grid_l1900_190089

/-- Represents a grid with specified dimensions and properties. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (unit_distance : ℕ)
  (allow_diagonals : Bool)

/-- Counts the number of rectangles that can be formed on the grid. -/
def count_rectangles (g : Grid) : ℕ := sorry

/-- The specific 3x3 grid with 2-unit spacing and allowed diagonals. -/
def specific_grid : Grid :=
  { rows := 3
  , cols := 3
  , unit_distance := 2
  , allow_diagonals := true }

/-- Theorem stating that the number of rectangles on the specific grid is 60. -/
theorem rectangles_on_specific_grid :
  count_rectangles specific_grid = 60 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_specific_grid_l1900_190089


namespace NUMINAMATH_CALUDE_first_knife_cost_is_five_l1900_190046

/-- The cost structure for knife sharpening -/
structure KnifeSharpening where
  first_knife_cost : ℝ
  next_three_cost : ℝ
  remaining_cost : ℝ
  total_knives : ℕ
  total_cost : ℝ

/-- The theorem stating the cost of sharpening the first knife -/
theorem first_knife_cost_is_five (ks : KnifeSharpening)
  (h1 : ks.next_three_cost = 4)
  (h2 : ks.remaining_cost = 3)
  (h3 : ks.total_knives = 9)
  (h4 : ks.total_cost = 32)
  (h5 : ks.total_cost = ks.first_knife_cost + 3 * ks.next_three_cost + 5 * ks.remaining_cost) :
  ks.first_knife_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_knife_cost_is_five_l1900_190046


namespace NUMINAMATH_CALUDE_workbook_problems_l1900_190060

theorem workbook_problems (P : ℚ) 
  (h1 : (1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P) : 
  P = 240 :=
by sorry

end NUMINAMATH_CALUDE_workbook_problems_l1900_190060


namespace NUMINAMATH_CALUDE_min_value_of_z_l1900_190011

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2*x + Real.sqrt 3 * y → z ≥ z_min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1900_190011


namespace NUMINAMATH_CALUDE_expand_expression_l1900_190040

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1900_190040


namespace NUMINAMATH_CALUDE_coin_loading_impossibility_l1900_190012

theorem coin_loading_impossibility 
  (p q : ℝ) 
  (h1 : 0 < p ∧ p < 1) 
  (h2 : 0 < q ∧ q < 1) 
  (h3 : p ≠ 1 - p) 
  (h4 : q ≠ 1 - q) 
  (h5 : p * q = (1 : ℝ) / 4) 
  (h6 : p * (1 - q) = (1 : ℝ) / 4) 
  (h7 : (1 - p) * q = (1 : ℝ) / 4) 
  (h8 : (1 - p) * (1 - q) = (1 : ℝ) / 4) :
  False :=
sorry

end NUMINAMATH_CALUDE_coin_loading_impossibility_l1900_190012


namespace NUMINAMATH_CALUDE_min_c_value_l1900_190007

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2033 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1017 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l1900_190007


namespace NUMINAMATH_CALUDE_three_fifths_equivalence_l1900_190096

/-- Proves the equivalence of various representations of 3/5 -/
theorem three_fifths_equivalence :
  (3 : ℚ) / 5 = 12 / 20 ∧
  (3 : ℚ) / 5 = (10 : ℚ) / (50 / 3) ∧
  (3 : ℚ) / 5 = 60 / 100 ∧
  (3 : ℚ) / 5 = 0.60 ∧
  (3 : ℚ) / 5 = 60 / 100 := by
  sorry

#check three_fifths_equivalence

end NUMINAMATH_CALUDE_three_fifths_equivalence_l1900_190096


namespace NUMINAMATH_CALUDE_convention_handshakes_correct_l1900_190076

/-- Represents the number of handshakes at a twins and triplets convention -/
def convention_handshakes (twin_sets : ℕ) (triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 2) * 2
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

theorem convention_handshakes_correct :
  convention_handshakes 9 6 = 441 := by sorry

end NUMINAMATH_CALUDE_convention_handshakes_correct_l1900_190076


namespace NUMINAMATH_CALUDE_alice_bob_meet_l1900_190098

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 13

/-- The relative clockwise movement between Alice and Bob per turn -/
def relative_move : ℤ := alice_move - (n - bob_move)

/-- The theorem stating that Alice and Bob meet after 9 turns -/
theorem alice_bob_meet :
  (∃ k : ℕ, k > 0 ∧ k * relative_move % n = 0) ∧
  (∀ j : ℕ, j > 0 ∧ j * relative_move % n = 0 → j ≥ 9) ∧
  9 * relative_move % n = 0 := by
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l1900_190098


namespace NUMINAMATH_CALUDE_xy_equation_l1900_190008

theorem xy_equation (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.888888888888889)
  (h2 : x + 2 * y = 10) :
  x + y = 5.666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_xy_equation_l1900_190008


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l1900_190063

theorem quadratic_function_determination (a b c : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, a * x + b ≤ 2) →
  (∃ x : ℝ, a * x + b = 2) →
  (a * x^2 + b * x + c = 2 * x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l1900_190063


namespace NUMINAMATH_CALUDE_trig_identity_l1900_190016

theorem trig_identity (α : ℝ) : 
  (Real.sin α)^2 + (Real.cos (30 * π / 180 - α))^2 - 
  (Real.sin α) * (Real.cos (30 * π / 180 - α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1900_190016


namespace NUMINAMATH_CALUDE_pascal_triangle_41st_number_l1900_190084

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (row : ℕ) : ℕ := row + 1

/-- The row number (0-indexed) of Pascal's triangle with 43 numbers -/
def target_row : ℕ := 42

/-- The index (0-indexed) of the target number in the row -/
def target_index : ℕ := 40

theorem pascal_triangle_41st_number : 
  binomial target_row target_index = 861 ∧ 
  pascal_row_length target_row = 43 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_41st_number_l1900_190084


namespace NUMINAMATH_CALUDE_distance_to_asymptote_l1900_190071

def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

def point : ℝ × ℝ := (3, 0)

theorem distance_to_asymptote :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), hyperbola x y → (a * x + b * y + c = 0 ∨ a * x + b * y - c = 0)) ∧
    (|a * point.1 + b * point.2 + c| / Real.sqrt (a^2 + b^2) = 9/5) :=
sorry

end NUMINAMATH_CALUDE_distance_to_asymptote_l1900_190071


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l1900_190073

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x - a/x)^9
def coeff_x3 (a : ℝ) : ℝ := -binomial 9 3 * a^3

-- Theorem statement
theorem expansion_coefficient_implies_a_value (a : ℝ) : 
  coeff_x3 a = -84 → a = 1 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l1900_190073


namespace NUMINAMATH_CALUDE_prob_three_red_marbles_l1900_190028

def total_marbles : ℕ := 5 + 6 + 7

def prob_all_red (red white blue : ℕ) : ℚ :=
  (red : ℚ) / total_marbles *
  ((red - 1) : ℚ) / (total_marbles - 1) *
  ((red - 2) : ℚ) / (total_marbles - 2)

theorem prob_three_red_marbles :
  prob_all_red 5 6 7 = 5 / 408 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_marbles_l1900_190028


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1900_190014

theorem solve_linear_equation : ∃ x : ℝ, 2 * x - 5 = 15 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1900_190014


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1900_190017

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 < a*x + b ↔ 1 < x ∧ x < 3) → b^a = 81 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1900_190017


namespace NUMINAMATH_CALUDE_small_cubes_to_large_cube_l1900_190070

theorem small_cubes_to_large_cube (large_volume small_volume : ℕ) 
  (h : large_volume = 1000 ∧ small_volume = 8) : 
  (large_volume / small_volume : ℕ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_to_large_cube_l1900_190070


namespace NUMINAMATH_CALUDE_number_of_ones_l1900_190079

theorem number_of_ones (n : ℕ) (hn : n = 999999999) : 
  ∃ x : ℤ, (n : ℤ) * x = (10^81 - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_ones_l1900_190079


namespace NUMINAMATH_CALUDE_school_commute_analysis_l1900_190048

/-- Represents the distribution of students in four commute time groups -/
structure SchoolCommuteData where
  less_than_20: Nat
  between_20_and_40: Nat
  between_41_and_60: Nat
  more_than_60: Nat

/-- The given distribution of students -/
def school_data : SchoolCommuteData :=
  { less_than_20 := 90
  , between_20_and_40 := 60
  , between_41_and_60 := 10
  , more_than_60 := 20 }

theorem school_commute_analysis (data : SchoolCommuteData := school_data) :
  (data.less_than_20 = 90) ∧
  (data.less_than_20 + data.between_20_and_40 + data.between_41_and_60 + data.more_than_60 = 180) ∧
  (data.between_41_and_60 + data.more_than_60 = 30) ∧
  ¬(data.between_20_and_40 + data.between_41_and_60 + data.more_than_60 > data.less_than_20) := by
  sorry

#check school_commute_analysis

end NUMINAMATH_CALUDE_school_commute_analysis_l1900_190048


namespace NUMINAMATH_CALUDE_units_digit_of_65_plus_37_in_octal_l1900_190023

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNumber :=
  sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Gets the units digit of an octal number --/
def unitsDigit (n : OctalNumber) : ℕ :=
  sorry

/-- Theorem: The units digit of 65₈ + 37₈ in base 8 is 4 --/
theorem units_digit_of_65_plus_37_in_octal :
  unitsDigit (octalAdd (toOctal 65) (toOctal 37)) = 4 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_65_plus_37_in_octal_l1900_190023


namespace NUMINAMATH_CALUDE_dog_grouping_ways_l1900_190062

def total_dogs : ℕ := 15
def group_1_size : ℕ := 4
def group_2_size : ℕ := 6
def group_3_size : ℕ := 5

def duke_in_group_1 : Prop := True
def bella_in_group_2 : Prop := True

theorem dog_grouping_ways : 
  total_dogs = group_1_size + group_2_size + group_3_size →
  duke_in_group_1 →
  bella_in_group_2 →
  (Nat.choose (total_dogs - 2) (group_1_size - 1)) * 
  (Nat.choose (total_dogs - group_1_size - 1) (group_2_size - 1)) = 72072 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_ways_l1900_190062


namespace NUMINAMATH_CALUDE_g_composition_of_3_l1900_190027

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_composition_of_3 : g (g (g (g 3))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l1900_190027


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_l1900_190075

theorem right_triangle_inscribed_circle 
  (a b T C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ T > 0 ∧ C > 0) 
  (h_right_triangle : ∃ (c h : ℝ), c = a + b ∧ T = (1/2) * c * h ∧ h^2 = a * b) 
  (h_inscribed : ∃ (R : ℝ), C = π * R^2 ∧ 2 * R = a + b) : 
  a * b = π * T^2 / C := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_l1900_190075


namespace NUMINAMATH_CALUDE_dice_rotation_probability_l1900_190059

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 3

/-- The total number of ways to paint a single die -/
def ways_to_paint_one_die : ℕ := num_colors ^ num_faces

/-- The total number of ways to paint two dice -/
def total_paint_combinations : ℕ := ways_to_paint_one_die ^ 2

/-- The number of ways two dice can appear identical after rotation -/
def identical_after_rotation : ℕ := 1119

/-- The probability that two independently painted dice appear identical after rotation -/
theorem dice_rotation_probability :
  (identical_after_rotation : ℚ) / total_paint_combinations = 1119 / 531441 := by
  sorry

end NUMINAMATH_CALUDE_dice_rotation_probability_l1900_190059


namespace NUMINAMATH_CALUDE_flower_shop_expenses_l1900_190064

/-- Calculates the weekly expenses for running a flower shop --/
theorem flower_shop_expenses 
  (rent : ℝ) 
  (utility_rate : ℝ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (employees_per_shift : ℕ) 
  (hourly_wage : ℝ) 
  (h_rent : rent = 1200) 
  (h_utility : utility_rate = 0.2) 
  (h_hours : hours_per_day = 16) 
  (h_days : days_per_week = 5) 
  (h_employees : employees_per_shift = 2) 
  (h_wage : hourly_wage = 12.5) : 
  rent + rent * utility_rate + 
  (↑hours_per_day * ↑days_per_week * ↑employees_per_shift * hourly_wage) = 3440 := by
  sorry

#check flower_shop_expenses

end NUMINAMATH_CALUDE_flower_shop_expenses_l1900_190064


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1900_190083

theorem quadratic_minimum (b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - 18 * x + b
  ∃ (min_value : ℝ), (∀ x, f x ≥ min_value) ∧ (f 3 = min_value) ∧ (min_value = b - 27) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1900_190083


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1900_190072

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ∧
    P = 7 ∧ Q = -9 ∧ R = 5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1900_190072


namespace NUMINAMATH_CALUDE_h_max_value_f_leq_g_condition_l1900_190001

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
def g (a : ℝ) (x : ℝ) : ℝ := x / (a * x + 1)
def h (x : ℝ) : ℝ := x * Real.exp (-x)

-- Theorem for the maximum value of h(x)
theorem h_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), h y ≤ h x ∧ h x = 1 / Real.exp 1 :=
sorry

-- Theorem for the range of a
theorem f_leq_g_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_h_max_value_f_leq_g_condition_l1900_190001


namespace NUMINAMATH_CALUDE_quadratic_two_positive_roots_l1900_190002

theorem quadratic_two_positive_roots (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (1 - a) * x^2 + (a + 2) * x - 4 = 0 ∧
    (1 - a) * y^2 + (a + 2) * y - 4 = 0) ↔
  (1 < a ∧ a ≤ 2) ∨ a ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_positive_roots_l1900_190002


namespace NUMINAMATH_CALUDE_mod_inverse_two_mod_221_l1900_190085

theorem mod_inverse_two_mod_221 : ∃ x : ℕ, x < 221 ∧ (2 * x) % 221 = 1 :=
by
  use 111
  sorry

end NUMINAMATH_CALUDE_mod_inverse_two_mod_221_l1900_190085


namespace NUMINAMATH_CALUDE_value_of_a_l1900_190018

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a (a : ℝ) : A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1900_190018


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1900_190090

theorem sqrt_equation_solution : 
  ∃ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (2 * (x - 2)) = 3) ∧ (x = 18/5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1900_190090


namespace NUMINAMATH_CALUDE_min_distance_proof_l1900_190093

/-- The distance between the graphs of y = 2x and y = -x^2 - 2x - 1 at a given x -/
def distance (x : ℝ) : ℝ := 2*x - (-x^2 - 2*x - 1)

/-- The minimum non-negative distance between the graphs -/
def min_distance : ℝ := 1

theorem min_distance_proof : 
  ∀ x : ℝ, distance x ≥ 0 → distance x ≥ min_distance :=
sorry

end NUMINAMATH_CALUDE_min_distance_proof_l1900_190093


namespace NUMINAMATH_CALUDE_function_values_l1900_190078

noncomputable section

def f (x : ℝ) : ℝ := -1/x

theorem function_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (f a = -1/3 → a = 3) ∧
  (f (a * b) = 1/6 → b = -2) ∧
  (f c = Real.sin c / Real.cos c → Real.tan c = -1/c) :=
by sorry

end NUMINAMATH_CALUDE_function_values_l1900_190078


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1900_190020

theorem rectangle_to_square (original_length : ℝ) (original_width : ℝ) (square_side : ℝ) : 
  original_width = 24 →
  square_side = 12 →
  original_length * original_width = square_side * square_side →
  original_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1900_190020


namespace NUMINAMATH_CALUDE_equal_money_distribution_l1900_190029

theorem equal_money_distribution (younger_money : ℝ) (h : younger_money > 0) :
  let elder_money := 1.25 * younger_money
  let transfer_amount := 0.1 * elder_money
  elder_money - transfer_amount = younger_money + transfer_amount := by
  sorry

end NUMINAMATH_CALUDE_equal_money_distribution_l1900_190029


namespace NUMINAMATH_CALUDE_harry_pet_feeding_cost_l1900_190055

/-- Represents the annual cost of feeding Harry's pets -/
def annual_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
  (gecko_meals_per_month iguana_meals_per_month : ℕ) 
  (snake_meals_per_year : ℕ)
  (gecko_meal_cost iguana_meal_cost snake_meal_cost : ℕ) : ℕ :=
  (num_geckos * gecko_meals_per_month * 12 * gecko_meal_cost) +
  (num_iguanas * iguana_meals_per_month * 12 * iguana_meal_cost) +
  (num_snakes * snake_meals_per_year * snake_meal_cost)

/-- Theorem stating the annual cost of feeding Harry's pets -/
theorem harry_pet_feeding_cost :
  annual_feeding_cost 3 2 4 2 3 6 8 12 20 = 1920 := by
  sorry

#eval annual_feeding_cost 3 2 4 2 3 6 8 12 20

end NUMINAMATH_CALUDE_harry_pet_feeding_cost_l1900_190055


namespace NUMINAMATH_CALUDE_john_total_cost_l1900_190042

/-- The total cost for John to raise a child and pay for university tuition -/
def total_cost_for_john : ℕ := 
  let cost_per_year_first_8 := 10000
  let years_first_period := 8
  let years_second_period := 10
  let university_tuition := 250000
  let first_period_cost := cost_per_year_first_8 * years_first_period
  let second_period_cost := 2 * cost_per_year_first_8 * years_second_period
  let total_cost := first_period_cost + second_period_cost + university_tuition
  total_cost / 2

/-- Theorem stating that the total cost for John is $265,000 -/
theorem john_total_cost : total_cost_for_john = 265000 := by
  sorry

end NUMINAMATH_CALUDE_john_total_cost_l1900_190042


namespace NUMINAMATH_CALUDE_first_ring_at_start_of_day_l1900_190069

-- Define the clock's properties
def ring_interval : ℕ := 3
def rings_per_day : ℕ := 8
def hours_per_day : ℕ := 24

-- Theorem to prove
theorem first_ring_at_start_of_day :
  ring_interval * rings_per_day = hours_per_day →
  ring_interval ∣ hours_per_day →
  (0 : ℕ) = hours_per_day % ring_interval :=
by
  sorry

#check first_ring_at_start_of_day

end NUMINAMATH_CALUDE_first_ring_at_start_of_day_l1900_190069


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l1900_190025

theorem technician_round_trip_completion (D : ℝ) (h : D > 0) : 
  let total_distance : ℝ := 2 * D
  let completed_distance : ℝ := D + 0.3 * D
  (completed_distance / total_distance) * 100 = 65 := by sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l1900_190025


namespace NUMINAMATH_CALUDE_shopping_money_left_l1900_190024

theorem shopping_money_left (initial_amount : ℝ) (final_amount : ℝ) 
  (spent_percentage : ℝ) (h1 : initial_amount = 4000) 
  (h2 : final_amount = 2800) (h3 : spent_percentage = 0.3) : 
  initial_amount * (1 - spent_percentage) = final_amount := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_left_l1900_190024


namespace NUMINAMATH_CALUDE_find_x_l1900_190065

theorem find_x : ∃ x : ℚ, (1/2 * x) = (1/3 * x + 110) ∧ x = 660 := by sorry

end NUMINAMATH_CALUDE_find_x_l1900_190065


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1900_190019

/-- Given that N(5,8) is the midpoint of line segment CD and C(7,4) is one endpoint,
    the product of coordinates of point D is 36. -/
theorem midpoint_coordinate_product (C D N : ℝ × ℝ) : 
  C = (7, 4) → N = (5, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 * D.2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1900_190019


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l1900_190032

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := 5 * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_220 :
  rectangle_area 16 11 = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l1900_190032


namespace NUMINAMATH_CALUDE_prism_faces_count_l1900_190074

/-- A prism with n sides, where n is at least 3 --/
structure Prism (n : ℕ) where
  sides : n ≥ 3

/-- The number of faces in a prism --/
def num_faces (n : ℕ) (p : Prism n) : ℕ := n + 2

theorem prism_faces_count (n : ℕ) (p : Prism n) : 
  num_faces n p ≠ n :=
sorry

end NUMINAMATH_CALUDE_prism_faces_count_l1900_190074


namespace NUMINAMATH_CALUDE_duty_arrangements_count_l1900_190097

def staff_count : ℕ := 7
def days_count : ℕ := 7
def restricted_days : ℕ := 2
def restricted_staff : ℕ := 2

theorem duty_arrangements_count : 
  (staff_count.factorial) / ((staff_count - days_count).factorial) *
  ((days_count - restricted_days).factorial) / 
  ((days_count - restricted_days - restricted_staff).factorial) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_count_l1900_190097


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_cups_l1900_190080

/-- Represents Mo's drinking habits and last week's statistics -/
structure MoDrinkingHabits where
  rainyDayHotChocolate : ℕ  -- Number of hot chocolate cups on rainy days
  nonRainyDayTea : ℕ        -- Number of tea cups on non-rainy days
  totalCups : ℕ             -- Total cups drunk last week
  teaMoreThanHotChocolate : ℕ  -- Difference between tea and hot chocolate cups
  rainyDays : ℕ             -- Number of rainy days last week

/-- Theorem stating that Mo drinks 11 cups of hot chocolate on rainy mornings -/
theorem mo_hot_chocolate_cups (mo : MoDrinkingHabits)
    (h1 : mo.nonRainyDayTea = 5)
    (h2 : mo.totalCups = 36)
    (h3 : mo.teaMoreThanHotChocolate = 14)
    (h4 : mo.rainyDays = 2) :
    mo.rainyDayHotChocolate = 11 := by
  sorry

end NUMINAMATH_CALUDE_mo_hot_chocolate_cups_l1900_190080


namespace NUMINAMATH_CALUDE_total_people_in_line_l1900_190044

/-- The number of people in a ticket line -/
def ticket_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 people in the ticket line -/
theorem total_people_in_line :
  ticket_line 6 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_line_l1900_190044


namespace NUMINAMATH_CALUDE_library_wall_arrangement_l1900_190035

/-- Proves that the maximum number of desk-bookcase pairs on a 15m wall leaves 3m of space --/
theorem library_wall_arrangement (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) 
  (space_between : ℝ) (h1 : wall_length = 15) (h2 : desk_length = 2) 
  (h3 : bookcase_length = 1.5) (h4 : space_between = 0.5) : 
  ∃ (n : ℕ) (leftover : ℝ), 
    n * (desk_length + bookcase_length + space_between) + leftover = wall_length ∧ 
    leftover = 3 ∧ 
    ∀ m : ℕ, m > n → m * (desk_length + bookcase_length + space_between) > wall_length := by
  sorry

end NUMINAMATH_CALUDE_library_wall_arrangement_l1900_190035


namespace NUMINAMATH_CALUDE_triangle_area_l1900_190053

/-- Given a triangle ABC with side a = 2, angle A = 30°, and angle C = 45°, 
    prove that its area S is equal to √3 + 1 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 2 → 
  A = π / 6 → 
  C = π / 4 → 
  A + B + C = π → 
  a / Real.sin A = c / Real.sin C → 
  S = (1 / 2) * a * c * Real.sin B → 
  S = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l1900_190053


namespace NUMINAMATH_CALUDE_parabola_vertex_l1900_190088

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 2)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 1)

/-- Theorem: The vertex of the parabola y = (x-2)^2 + 1 is (2, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1900_190088


namespace NUMINAMATH_CALUDE_alexandras_magazines_l1900_190004

theorem alexandras_magazines (friday_magazines : ℕ) (saturday_magazines : ℕ) 
  (sunday_multiplier : ℕ) (monday_multiplier : ℕ) (chewed_magazines : ℕ) : 
  friday_magazines = 18 →
  saturday_magazines = 25 →
  sunday_multiplier = 5 →
  monday_multiplier = 3 →
  chewed_magazines = 10 →
  friday_magazines + saturday_magazines + 
  (sunday_multiplier * friday_magazines) + 
  (monday_multiplier * saturday_magazines) - 
  chewed_magazines = 198 := by
sorry

end NUMINAMATH_CALUDE_alexandras_magazines_l1900_190004


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1900_190050

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | |x - 4| + |x - 3| ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
by sorry

-- Part 2
theorem range_of_a :
  ∀ x : ℝ, f x a ≥ 4 → a ≤ -1 ∨ a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1900_190050


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1900_190082

def set_A : Set ℝ := {x | |x - 2| ≤ 1}
def set_B : Set ℝ := {x | (x - 5) / (2 - x) > 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 2 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1900_190082


namespace NUMINAMATH_CALUDE_function_difference_bound_l1900_190054

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - x * Real.log a

theorem function_difference_bound
  (a : ℝ) (ha : 1 < a ∧ a ≤ Real.exp 1) :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
  |f a x₁ - f a x₂| ≤ Real.exp 1 - 2 :=
sorry

end NUMINAMATH_CALUDE_function_difference_bound_l1900_190054


namespace NUMINAMATH_CALUDE_fewer_correct_answers_l1900_190052

-- Define the number of correct answers for each person
def cherry_correct : ℕ := 17
def nicole_correct : ℕ := 22
def kim_correct : ℕ := cherry_correct + 8

-- State the theorem
theorem fewer_correct_answers :
  kim_correct - nicole_correct = 3 ∧
  nicole_correct < kim_correct :=
by sorry

end NUMINAMATH_CALUDE_fewer_correct_answers_l1900_190052


namespace NUMINAMATH_CALUDE_divisibility_problem_l1900_190039

theorem divisibility_problem : ∃ (a b : ℕ), 
  (7^3 ∣ a^2 + a*b + b^2) ∧ 
  ¬(7 ∣ a) ∧ 
  ¬(7 ∣ b) ∧
  a = 1 ∧ 
  b = 18 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1900_190039


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1900_190022

theorem polynomial_simplification (x : ℝ) :
  5 - 7*x - 13*x^2 + 10 + 15*x - 25*x^2 - 20 + 21*x + 33*x^2 - 15*x^3 =
  -15*x^3 - 5*x^2 + 29*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1900_190022


namespace NUMINAMATH_CALUDE_sons_age_l1900_190015

/-- Given a woman and her son, where the woman's age is three years more than twice her son's age,
    and the sum of their ages is 84, prove that the son's age is 27. -/
theorem sons_age (S W : ℕ) (h1 : W = 2 * S + 3) (h2 : W + S = 84) : S = 27 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1900_190015


namespace NUMINAMATH_CALUDE_greatest_three_digit_base7_divisible_by_7_l1900_190038

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 7 number --/
def isThreeDigitBase7 (n : ℕ) : Prop := sorry

/-- The greatest 3-digit base 7 number --/
def greatestThreeDigitBase7 : ℕ := 666

theorem greatest_three_digit_base7_divisible_by_7 :
  isThreeDigitBase7 greatestThreeDigitBase7 ∧
  base7ToDecimal greatestThreeDigitBase7 % 7 = 0 ∧
  ∀ n : ℕ, isThreeDigitBase7 n ∧ base7ToDecimal n % 7 = 0 →
    base7ToDecimal n ≤ base7ToDecimal greatestThreeDigitBase7 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_base7_divisible_by_7_l1900_190038


namespace NUMINAMATH_CALUDE_existence_of_m_n_l1900_190092

theorem existence_of_m_n (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l1900_190092


namespace NUMINAMATH_CALUDE_complex_magnitude_l1900_190043

theorem complex_magnitude (z : ℂ) (h : (3 + Complex.I) / z = 1 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1900_190043


namespace NUMINAMATH_CALUDE_matthews_cows_l1900_190034

/-- Proves that Matthews has 60 cows given the problem conditions -/
theorem matthews_cows :
  ∀ (matthews aaron marovich : ℕ),
  aaron = 4 * matthews →
  aaron + matthews = marovich + 30 →
  matthews + aaron + marovich = 570 →
  matthews = 60 := by
sorry

end NUMINAMATH_CALUDE_matthews_cows_l1900_190034


namespace NUMINAMATH_CALUDE_max_area_2014_l1900_190086

/-- A polygon drawn on a grid with sides following grid lines -/
structure GridPolygon where
  perimeter : ℕ
  sides_follow_grid : Bool

/-- The maximum area of a grid polygon given its perimeter -/
def max_area (p : GridPolygon) : ℕ :=
  (p.perimeter / 4)^2 - if p.perimeter % 4 == 2 then 1/4 else 0

/-- Theorem stating the maximum area of a grid polygon with perimeter 2014 -/
theorem max_area_2014 :
  ∀ (p : GridPolygon), p.perimeter = 2014 → p.sides_follow_grid → max_area p = 253512 := by
  sorry


end NUMINAMATH_CALUDE_max_area_2014_l1900_190086


namespace NUMINAMATH_CALUDE_sequence_existence_iff_k_in_range_l1900_190031

theorem sequence_existence_iff_k_in_range (n : ℕ) :
  (∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ n → j ≤ n → x i < x j)) ↔
  (∀ k : ℕ, k ≤ n → ∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ k → j ≤ k → x i < x j)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_iff_k_in_range_l1900_190031


namespace NUMINAMATH_CALUDE_odot_count_53_l1900_190051

/-- Represents a sequence of four symbols -/
structure SymbolSequence :=
  (symbols : Fin 4 → Char)
  (odot_count : (symbols 2 = '⊙' ∧ symbols 3 = '⊙') ∨ (symbols 1 = '⊙' ∧ symbols 2 = '⊙') ∨ (symbols 0 = '⊙' ∧ symbols 3 = '⊙'))

/-- Counts the occurrences of a symbol in the repeated pattern up to a given position -/
def count_symbol (seq : SymbolSequence) (symbol : Char) (n : Nat) : Nat :=
  (n / 4) * 2 + if n % 4 ≥ 3 then 2 else if n % 4 ≥ 2 then 1 else 0

/-- The main theorem stating that the count of ⊙ in the first 53 positions is 26 -/
theorem odot_count_53 (seq : SymbolSequence) : count_symbol seq '⊙' 53 = 26 := by
  sorry


end NUMINAMATH_CALUDE_odot_count_53_l1900_190051


namespace NUMINAMATH_CALUDE_initial_count_of_numbers_l1900_190013

/-- Given a set of numbers with average 27, prove that if removing 35 results in average 25, then there were initially 5 numbers -/
theorem initial_count_of_numbers (n : ℕ) (S : ℝ) : 
  S / n = 27 →
  (S - 35) / (n - 1) = 25 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_count_of_numbers_l1900_190013


namespace NUMINAMATH_CALUDE_solution_set_eq_expected_set_l1900_190094

/-- The solution set of the inequality x|x-1| > 0 -/
def solution_set : Set ℝ := {x : ℝ | x * |x - 1| > 0}

/-- The expected result set (0,1) ∪ (1,+∞) -/
def expected_set : Set ℝ := Set.Ioo 0 1 ∪ Set.Ioi 1

theorem solution_set_eq_expected_set : solution_set = expected_set := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_expected_set_l1900_190094


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l1900_190087

theorem complex_exp_conversion :
  (Complex.exp (13 * Real.pi * Complex.I / 4)) * (Complex.ofReal (Real.sqrt 2)) = -1 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l1900_190087


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_l1900_190021

theorem quadratic_roots_sum_inverse (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0) :
  ∃ x1 x2 x3 x4 : ℝ, 
    x1 ≠ x2 ∧ x3 ≠ x4 ∧
    x1^2 + p*x1 + q = 0 ∧ x2^2 + p*x2 + q = 0 ∧
    x3^2 + q*x3 + p = 0 ∧ x4^2 + q*x4 + p = 0 ∧
    1/(x1*x3) + 1/(x1*x4) + 1/(x2*x3) + 1/(x2*x4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_l1900_190021


namespace NUMINAMATH_CALUDE_letter_digit_problem_l1900_190047

/-- Represents a mapping from letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a LetterDigitMap is valid according to the problem conditions -/
def is_valid_map (f : LetterDigitMap) : Prop :=
  (f 'E' ≠ f 'H') ∧ (f 'E' ≠ f 'M') ∧ (f 'E' ≠ f 'O') ∧ (f 'E' ≠ f 'P') ∧
  (f 'H' ≠ f 'M') ∧ (f 'H' ≠ f 'O') ∧ (f 'H' ≠ f 'P') ∧
  (f 'M' ≠ f 'O') ∧ (f 'M' ≠ f 'P') ∧
  (f 'O' ≠ f 'P') ∧
  (∀ c, c ∈ ['E', 'H', 'M', 'O', 'P'] → f c ∈ [1, 2, 3, 4, 6, 8, 9])

theorem letter_digit_problem (f : LetterDigitMap) 
  (h1 : is_valid_map f)
  (h2 : f 'E' * f 'H' = f 'M' * f 'O' * f 'P' * f 'O' * 3)
  (h3 : f 'E' + f 'H' = f 'M' + f 'O' + f 'P' + f 'O' + 3) :
  f 'E' * f 'H' + f 'M' * f 'O' * f 'P' * f 'O' * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_letter_digit_problem_l1900_190047


namespace NUMINAMATH_CALUDE_daniel_paid_six_more_l1900_190003

/-- A pizza sharing scenario between Carl and Daniel -/
structure PizzaScenario where
  total_slices : ℕ
  plain_cost : ℚ
  truffle_cost : ℚ
  daniel_truffle_slices : ℕ
  daniel_plain_slices : ℕ
  carl_plain_slices : ℕ

/-- Calculate the payment difference between Daniel and Carl -/
def payment_difference (scenario : PizzaScenario) : ℚ :=
  let total_cost := scenario.plain_cost + scenario.truffle_cost
  let cost_per_slice := total_cost / scenario.total_slices
  let daniel_payment := (scenario.daniel_truffle_slices + scenario.daniel_plain_slices) * cost_per_slice
  let carl_payment := scenario.carl_plain_slices * cost_per_slice
  daniel_payment - carl_payment

/-- The specific pizza scenario described in the problem -/
def pizza : PizzaScenario :=
  { total_slices := 10
  , plain_cost := 10
  , truffle_cost := 5
  , daniel_truffle_slices := 5
  , daniel_plain_slices := 2
  , carl_plain_slices := 3 }

/-- Theorem stating that Daniel paid $6 more than Carl -/
theorem daniel_paid_six_more : payment_difference pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_daniel_paid_six_more_l1900_190003


namespace NUMINAMATH_CALUDE_ylona_initial_count_l1900_190000

/-- The number of rubber bands each person has initially and after Bailey gives some away. -/
structure RubberBands :=
  (bailey_initial : ℕ)
  (justine_initial : ℕ)
  (ylona_initial : ℕ)
  (bailey_final : ℕ)

/-- The conditions of the rubber band problem. -/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.ylona_initial = rb.justine_initial + 2 ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8

/-- Theorem stating that Ylona initially had 24 rubber bands. -/
theorem ylona_initial_count (rb : RubberBands) 
  (h : rubber_band_problem rb) : rb.ylona_initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_ylona_initial_count_l1900_190000


namespace NUMINAMATH_CALUDE_range_of_m_l1900_190057

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < m^2}

def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)

theorem range_of_m : 
  {m : ℝ | necessary_not_sufficient m} = {m : ℝ | -1/2 ≤ m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1900_190057


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l1900_190009

theorem greatest_integer_radius_for_circle (r : ℕ) : r * r ≤ 49 → r ≤ 7 ∧ ∃ (s : ℕ), s = 7 ∧ s * s ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l1900_190009


namespace NUMINAMATH_CALUDE_product_evaluation_l1900_190066

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1900_190066


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_12_l1900_190077

theorem binomial_coefficient_19_12 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 18 11 = 31824) : 
  Nat.choose 19 12 = 77520 - 31824 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_12_l1900_190077


namespace NUMINAMATH_CALUDE_pigeonhole_principle_interns_l1900_190081

theorem pigeonhole_principle_interns (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_interns_l1900_190081


namespace NUMINAMATH_CALUDE_odd_decreasing_properties_l1900_190067

/-- An odd and decreasing function on ℝ -/
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x ≤ y → f y ≤ f x)

theorem odd_decreasing_properties
  (f : ℝ → ℝ) (hf : odd_decreasing_function f) (m n : ℝ) (h : m + n ≥ 0) :
  (f m * f (-m) ≤ 0) ∧ (f m + f n ≤ f (-m) + f (-n)) :=
by sorry

end NUMINAMATH_CALUDE_odd_decreasing_properties_l1900_190067


namespace NUMINAMATH_CALUDE_log_relation_l1900_190068

theorem log_relation (a b : ℝ) : 
  a = Real.log 1024 / Real.log 16 → b = Real.log 32 / Real.log 2 → a = (1/2) * b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l1900_190068


namespace NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l1900_190041

/-- Given a fractional equation (x - 3) / (x - 1) = m / (x - 1),
    if x = 1 is an extraneous root, then m = -2 -/
theorem extraneous_root_implies_m_value :
  ∀ (x m : ℝ), 
    (x - 3) / (x - 1) = m / (x - 1) →
    (1 : ℝ) ≠ 1 →  -- This represents that x = 1 is an extraneous root
    m = -2 := by
  sorry


end NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l1900_190041
