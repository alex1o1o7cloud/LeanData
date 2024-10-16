import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_divisors_1184_l512_51241

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_1184 : sum_of_divisors 1184 = 2394 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_1184_l512_51241


namespace NUMINAMATH_CALUDE_conditional_probability_equal_marginal_l512_51203

-- Define the sample space and events
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)
variable (A B : Set Ω)

-- Define the probabilities and independence
variable (hA : P A = 1/6)
variable (hB : P B = 1/2)
variable (hInd : P.Independent A B)

-- State the theorem
theorem conditional_probability_equal_marginal
  (h_prob_B_pos : P B > 0) :
  P.condProb A B = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_equal_marginal_l512_51203


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l512_51296

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 48 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l512_51296


namespace NUMINAMATH_CALUDE_triangle_angle_constraint_l512_51204

/-- 
Given a triangle ABC with the conditions:
1) 5 * sin(A) + 2 * cos(B) = 5
2) 2 * sin(B) + 5 * cos(A) = 2

This theorem states that either:
a) The triangle is degenerate with angle C = 180°, or
b) There is no solution for a non-degenerate triangle.
-/
theorem triangle_angle_constraint (A B C : ℝ) : 
  (5 * Real.sin A + 2 * Real.cos B = 5) →
  (2 * Real.sin B + 5 * Real.cos A = 2) →
  (A + B + C = Real.pi) →
  ((C = Real.pi ∧ (A = 0 ∨ B = 0)) ∨ 
   ∀ A B C, ¬(A > 0 ∧ B > 0 ∧ C > 0)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_constraint_l512_51204


namespace NUMINAMATH_CALUDE_sports_club_overlap_l512_51244

theorem sports_club_overlap (N B T X : ℕ) (h1 : N = 40) (h2 : B = 20) (h3 : T = 18) (h4 : X = 5) :
  B + T - (N - X) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l512_51244


namespace NUMINAMATH_CALUDE_M_lower_bound_l512_51251

theorem M_lower_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_M_lower_bound_l512_51251


namespace NUMINAMATH_CALUDE_ratio_product_theorem_l512_51210

theorem ratio_product_theorem (a b c : ℝ) : 
  a / b = 3 / 4 ∧ b / c = 4 / 6 ∧ c = 18 → a * b * c = 1944 := by
  sorry

end NUMINAMATH_CALUDE_ratio_product_theorem_l512_51210


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l512_51250

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_gt_perimeter : a * b > 2 * (a + b)) : 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l512_51250


namespace NUMINAMATH_CALUDE_return_probability_limit_l512_51258

/-- Represents a player in the money exchange game --/
inductive Player : Type
| Alan : Player
| Beth : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by a function from Player to ℕ (amount of money) --/
def GameState : Type := Player → ℕ

/-- The initial state of the game where each player has $1 --/
def initialState : GameState :=
  fun p => 1

/-- A single round of the game where players randomly exchange money --/
def playRound (state : GameState) : GameState :=
  sorry

/-- The probability of returning to the initial state after many rounds --/
def returnProbability (numRounds : ℕ) : ℚ :=
  sorry

/-- The main theorem stating that the probability approaches 1/9 as the number of rounds increases --/
theorem return_probability_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |returnProbability n - 1/9| < ε :=
sorry

end NUMINAMATH_CALUDE_return_probability_limit_l512_51258


namespace NUMINAMATH_CALUDE_length_of_PQ_l512_51297

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the locus E
def E : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2/4 = 1 ∧ p.1 ≠ 1 ∧ p.1 ≠ -1}

-- Define the slope product condition
def slope_product (M : ℝ × ℝ) : Prop :=
  (M.2 / (M.1 - 1)) * (M.2 / (M.1 + 1)) = 4

-- Define line l
def l : Set (ℝ × ℝ) := {p | ∃ (k : ℝ), p.2 = k * p.1 - 2}

-- Define the midpoint condition
def midpoint_condition (P Q : ℝ × ℝ) : Prop :=
  let D := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  D.1 > 0 ∧ D.2 = 2

-- Main theorem
theorem length_of_PQ :
  ∀ (P Q : ℝ × ℝ),
  P ∈ E ∧ Q ∈ E ∧ P ∈ l ∧ Q ∈ l ∧
  midpoint_condition P Q ∧
  (∀ M ∈ E, slope_product M) →
  ‖P - Q‖ = 2 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_length_of_PQ_l512_51297


namespace NUMINAMATH_CALUDE_range_of_a_l512_51266

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_domain : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-1) 1

-- Define the inequality condition
axiom inequality : ∀ a, f (2*a - 3) < f (a - 2)

-- Theorem statement
theorem range_of_a : ∀ a, (f (2*a - 3) < f (a - 2)) → a ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l512_51266


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l512_51289

/-- Given a line with equation y - 4 = 4(x - 8), prove that the sum of its x-intercept and y-intercept is -21 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 4 = 4 * (x - 8)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 4 = 4 * (x_int - 8)) ∧ 
    (0 - 4 = 4 * (x_int - 8)) ∧ 
    (y_int - 4 = 4 * (0 - 8)) ∧ 
    (x_int + y_int = -21)) := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l512_51289


namespace NUMINAMATH_CALUDE_tan_triangle_identity_l512_51273

theorem tan_triangle_identity (A B C : Real) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = Real.pi) : 
  (Real.tan A * Real.tan B * Real.tan C) / (Real.tan A + Real.tan B + Real.tan C) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_triangle_identity_l512_51273


namespace NUMINAMATH_CALUDE_c_percentage_less_than_d_l512_51202

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def a_marks : ℕ := 360

def b_marks : ℕ := a_marks * 100 / 90
def c_marks : ℕ := b_marks * 100 / 125

theorem c_percentage_less_than_d :
  (d_marks - c_marks) * 100 / d_marks = 20 := by sorry

end NUMINAMATH_CALUDE_c_percentage_less_than_d_l512_51202


namespace NUMINAMATH_CALUDE_power_of_two_multiplication_l512_51292

theorem power_of_two_multiplication : 2^4 * 2^4 * 2^4 = 2^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_multiplication_l512_51292


namespace NUMINAMATH_CALUDE_square_split_into_pentagons_or_hexagons_l512_51211

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- The number of sides of a polygon -/
def Polygon.sides (p : Polygon) : ℕ := p.vertices.length

/-- A concave polygon -/
def ConcavePolygon (p : Polygon) : Prop := sorry

/-- The area of a polygon -/
def Polygon.area (p : Polygon) : ℝ := sorry

/-- A square with side length 1 -/
def UnitSquare : Polygon := sorry

/-- Two polygons are equal in area -/
def EqualArea (p1 p2 : Polygon) : Prop := p1.area = p2.area

/-- A polygon is contained within another polygon -/
def ContainedIn (p1 p2 : Polygon) : Prop := sorry

/-- The union of two polygons -/
def PolygonUnion (p1 p2 : Polygon) : Polygon := sorry

theorem square_split_into_pentagons_or_hexagons :
  ∃ (p1 p2 : Polygon),
    (p1.sides = 5 ∧ p2.sides = 5 ∨ p1.sides = 6 ∧ p2.sides = 6) ∧
    ConcavePolygon p1 ∧
    ConcavePolygon p2 ∧
    EqualArea p1 p2 ∧
    ContainedIn p1 UnitSquare ∧
    ContainedIn p2 UnitSquare ∧
    PolygonUnion p1 p2 = UnitSquare :=
sorry

end NUMINAMATH_CALUDE_square_split_into_pentagons_or_hexagons_l512_51211


namespace NUMINAMATH_CALUDE_pairball_playing_time_l512_51242

theorem pairball_playing_time (num_children : ℕ) (total_time : ℕ) (pair_size : ℕ) : 
  num_children = 6 →
  pair_size = 2 →
  total_time = 120 →
  (total_time * pair_size) / num_children = 40 := by
sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l512_51242


namespace NUMINAMATH_CALUDE_smith_family_buffet_cost_l512_51247

/-- Represents the cost calculation for a family at a seafood buffet. -/
def buffet_cost (adult_price : ℚ) (child_price : ℚ) (senior_discount : ℚ) 
  (num_adults num_seniors num_children : ℕ) : ℚ :=
  (num_adults * adult_price) + 
  (num_seniors * (adult_price * (1 - senior_discount))) + 
  (num_children * child_price)

/-- Theorem stating that the total cost for Mr. Smith's family at the seafood buffet is $159. -/
theorem smith_family_buffet_cost : 
  buffet_cost 30 15 (1/10) 2 2 3 = 159 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_buffet_cost_l512_51247


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l512_51288

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 39) 
  (h_sum : a 1 + a 3 = 74) : 
  ∀ n : ℕ, a n = -2 * n + 41 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l512_51288


namespace NUMINAMATH_CALUDE_joeys_lawn_mowing_l512_51291

theorem joeys_lawn_mowing (
  sneaker_cost : ℕ)
  (lawn_earnings : ℕ)
  (figure_price : ℕ)
  (figure_count : ℕ)
  (job_hours : ℕ)
  (hourly_rate : ℕ)
  (h1 : sneaker_cost = 92)
  (h2 : lawn_earnings = 8)
  (h3 : figure_price = 9)
  (h4 : figure_count = 2)
  (h5 : job_hours = 10)
  (h6 : hourly_rate = 5)
  : (sneaker_cost - (figure_price * figure_count + job_hours * hourly_rate)) / lawn_earnings = 3 := by
  sorry

end NUMINAMATH_CALUDE_joeys_lawn_mowing_l512_51291


namespace NUMINAMATH_CALUDE_system_equation_ratio_l512_51270

theorem system_equation_ratio (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 4 * y - 10 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 96/13 := by sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l512_51270


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l512_51252

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l512_51252


namespace NUMINAMATH_CALUDE_equation_rewrite_l512_51238

/-- Given an equation with roots α and β, prove that a related equation can be rewritten in terms of α, β, and a constant k. -/
theorem equation_rewrite (a b c d α β : ℝ) (hα : α = (a * α + b) / (c * α + d)) (hβ : β = (a * β + b) / (c * β + d)) :
  ∃ k : ℝ, ∀ y z : ℝ, y = (a * z + b) / (c * z + d) →
    (y - α) / (y - β) = k * (z - α) / (z - β) ∧ k = (c * β + d) / (c * α + d) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l512_51238


namespace NUMINAMATH_CALUDE_sqrt_four_minus_2023_power_zero_equals_one_l512_51234

theorem sqrt_four_minus_2023_power_zero_equals_one :
  Real.sqrt 4 - (2023 : ℝ) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_2023_power_zero_equals_one_l512_51234


namespace NUMINAMATH_CALUDE_lunks_needed_correct_l512_51268

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 1/2

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5/3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- The number of lunks needed to purchase the given number of apples -/
def lunks_needed : ℕ := 24

theorem lunks_needed_correct : 
  ↑lunks_needed = ↑apples_to_buy / (kunk_to_apple_rate * lunk_to_kunk_rate) := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_correct_l512_51268


namespace NUMINAMATH_CALUDE_sum_of_specific_sequences_l512_51212

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_of_sequences (seq1 seq2 : List ℕ) : ℕ :=
  (seq1 ++ seq2).sum

theorem sum_of_specific_sequences :
  let seq1 := arithmetic_sequence 3 10 5
  let seq2 := arithmetic_sequence 7 10 5
  sum_of_sequences seq1 seq2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_sequences_l512_51212


namespace NUMINAMATH_CALUDE_circle_properties_l512_51236

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- State the theorem
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The center is on the negative half of the x-axis
  ∃ (a : ℝ), a < 0 ∧ circle_equation a 0 ∧
  -- The radius is 2
  ∀ (x y : ℝ), circle_equation x y → (x + 2)^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l512_51236


namespace NUMINAMATH_CALUDE_handshakes_in_exhibition_l512_51295

/-- Represents a mixed-doubles tennis exhibition -/
structure MixedDoublesExhibition where
  num_teams : Nat
  players_per_team : Nat

/-- Calculates the total number of handshakes in a mixed-doubles tennis exhibition -/
def total_handshakes (exhibition : MixedDoublesExhibition) : Nat :=
  let total_players := exhibition.num_teams * exhibition.players_per_team
  let handshakes_per_player := total_players - 2  -- Exclude self and partner
  (total_players * handshakes_per_player) / 2

/-- Theorem stating that the total number of handshakes in the given exhibition is 24 -/
theorem handshakes_in_exhibition :
  ∃ (exhibition : MixedDoublesExhibition),
    exhibition.num_teams = 4 ∧
    exhibition.players_per_team = 2 ∧
    total_handshakes exhibition = 24 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_exhibition_l512_51295


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l512_51264

theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l512_51264


namespace NUMINAMATH_CALUDE_election_theorem_l512_51257

theorem election_theorem (winner_percentage : ℝ) (winner_margin : ℕ) (winner_votes : ℕ) :
  winner_percentage = 0.62 →
  winner_votes = 992 →
  winner_margin = 384 →
  ∃ (total_votes : ℕ) (runner_up_votes : ℕ),
    total_votes = 1600 ∧
    runner_up_votes = 608 ∧
    winner_votes = winner_percentage * total_votes ∧
    winner_votes = runner_up_votes + winner_margin :=
by
  sorry

#check election_theorem

end NUMINAMATH_CALUDE_election_theorem_l512_51257


namespace NUMINAMATH_CALUDE_find_N_l512_51299

theorem find_N : ∀ N : ℕ, (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N → N = 5967 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l512_51299


namespace NUMINAMATH_CALUDE_sum_equals_zero_l512_51259

theorem sum_equals_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l512_51259


namespace NUMINAMATH_CALUDE_prob_king_is_one_thirteenth_l512_51222

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (kings_per_suit : ℕ)
  (h_total : total_cards = num_ranks * num_suits)
  (h_kings : kings_per_suit = 1)

/-- The probability of drawing a King from a standard deck -/
def prob_draw_king (d : Deck) : ℚ :=
  (d.num_suits * d.kings_per_suit : ℚ) / d.total_cards

/-- Theorem: The probability of drawing a King from a standard deck is 1/13 -/
theorem prob_king_is_one_thirteenth (d : Deck) 
  (h_standard : d.total_cards = 52 ∧ d.num_ranks = 13 ∧ d.num_suits = 4) : 
  prob_draw_king d = 1 / 13 := by
  sorry

#check prob_king_is_one_thirteenth

end NUMINAMATH_CALUDE_prob_king_is_one_thirteenth_l512_51222


namespace NUMINAMATH_CALUDE_combined_teaching_years_l512_51201

/-- The combined teaching years of Mr. Spencer and Mrs. Randall -/
theorem combined_teaching_years : 
  let spencer_fourth_grade : ℕ := 12
  let spencer_first_grade : ℕ := 5
  let randall_third_grade : ℕ := 18
  let randall_second_grade : ℕ := 8
  (spencer_fourth_grade + spencer_first_grade + randall_third_grade + randall_second_grade) = 43 := by
  sorry

end NUMINAMATH_CALUDE_combined_teaching_years_l512_51201


namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l512_51214

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236.137 := by
  sorry

theorem equality_condition (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 = 236.137 ↔ 
  (Real.cos α = 10 / Real.sqrt 500 ∧ Real.sin α = 20 / Real.sqrt 500 ∧ β = Real.pi/2 - α) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l512_51214


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l512_51208

def canoes_in_month (n : ℕ) : ℕ :=
  2 * (3 ^ (n - 1))

theorem total_canoes_by_april : 
  canoes_in_month 1 + canoes_in_month 2 + canoes_in_month 3 + canoes_in_month 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l512_51208


namespace NUMINAMATH_CALUDE_smallest_binary_palindrome_l512_51253

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number 33 in decimal. -/
def target_number : ℕ := 33

theorem smallest_binary_palindrome :
  (is_palindrome target_number 2) ∧
  (∃ (b : ℕ), b > 2 ∧ is_palindrome target_number b) ∧
  (∀ (m : ℕ), m < target_number →
    ¬(is_palindrome m 2 ∧ (∃ (b : ℕ), b > 2 ∧ is_palindrome m b))) ∧
  (to_base target_number 2 = [1, 0, 0, 0, 0, 1]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_binary_palindrome_l512_51253


namespace NUMINAMATH_CALUDE_irreducible_fraction_to_mersenne_form_l512_51215

theorem irreducible_fraction_to_mersenne_form 
  (p q : ℕ+) 
  (h_q_odd : q.val % 2 = 1) : 
  ∃ (n k : ℕ+), (p : ℚ) / q = (n : ℚ) / (2^k.val - 1) :=
sorry

end NUMINAMATH_CALUDE_irreducible_fraction_to_mersenne_form_l512_51215


namespace NUMINAMATH_CALUDE_triangle_properties_l512_51217

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def altitude_BC (x y : ℝ) : Prop := x - 2*y - 1 = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0
def point_B : ℝ × ℝ := (2, 1)

-- Define the theorem
theorem triangle_properties (ABC : Triangle) :
  altitude_BC ABC.A.1 ABC.A.2 ∧
  altitude_BC ABC.C.1 ABC.C.2 ∧
  angle_bisector_A ABC.A.2 ∧
  ABC.B = point_B →
  ABC.A = (1, 0) ∧
  ABC.C = (4, -3) ∧
  ∀ (x y : ℝ), y = x - 1 ↔ (x = ABC.A.1 ∧ y = ABC.A.2) ∨ (x = ABC.C.1 ∧ y = ABC.C.2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l512_51217


namespace NUMINAMATH_CALUDE_expansion_terms_l512_51216

-- Define the exponent
def n : ℕ := 2016

-- Define the function that represents the number of terms
def num_terms (n : ℕ) : ℕ :=
  4 * n + 1

-- Theorem statement
theorem expansion_terms : num_terms n = 4033 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_l512_51216


namespace NUMINAMATH_CALUDE_ball_box_difference_l512_51267

theorem ball_box_difference : 
  let balls_per_box : ℕ := 6
  let white_balls : ℕ := 30
  let red_balls : ℕ := 18
  let white_boxes := white_balls / balls_per_box
  let red_boxes := red_balls / balls_per_box
  white_boxes - red_boxes = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_box_difference_l512_51267


namespace NUMINAMATH_CALUDE_units_digit_period_four_units_digit_2_power_2012_l512_51227

/-- The units digit of 2^n -/
def unitsDigit (n : ℕ) : ℕ := 2^n % 10

/-- The pattern of units digits for powers of 2 repeats every 4 steps -/
theorem units_digit_period_four (n : ℕ) : 
  unitsDigit n = unitsDigit (n + 4) :=
sorry

/-- The units digit of 2^2012 is 6 -/
theorem units_digit_2_power_2012 : unitsDigit 2012 = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_period_four_units_digit_2_power_2012_l512_51227


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l512_51272

/-- A triangle with integer side lengths and perimeter 24 has maximum side length 11 -/
theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ -- Three different side lengths
  a + b + c = 24 ∧ -- Perimeter is 24
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  c ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l512_51272


namespace NUMINAMATH_CALUDE_born_day_300_years_before_l512_51218

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week 300 years before a given Monday -/
def dayOfWeek300YearsBefore (endDay : DayOfWeek) : DayOfWeek :=
  match endDay with
  | DayOfWeek.Monday => DayOfWeek.Wednesday
  | _ => DayOfWeek.Monday  -- This case should never occur in our problem

/-- Theorem stating that 300 years before a Monday is a Wednesday -/
theorem born_day_300_years_before (endDay : DayOfWeek) 
  (h : endDay = DayOfWeek.Monday) : 
  dayOfWeek300YearsBefore endDay = DayOfWeek.Wednesday :=
by sorry

#check born_day_300_years_before

end NUMINAMATH_CALUDE_born_day_300_years_before_l512_51218


namespace NUMINAMATH_CALUDE_f_2019_is_zero_l512_51231

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) - f x = 2 * f 3

theorem f_2019_is_zero (f : ℝ → ℝ) (h1 : is_even f) (h2 : satisfies_equation f) : f 2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_is_zero_l512_51231


namespace NUMINAMATH_CALUDE_johnson_farm_wheat_acreage_l512_51245

/-- Proves that given the conditions of Johnson Farm, the number of acres of wheat planted is 200 -/
theorem johnson_farm_wheat_acreage :
  let total_land : ℝ := 500
  let corn_cost : ℝ := 42
  let wheat_cost : ℝ := 30
  let total_budget : ℝ := 18600
  let wheat_acres : ℝ := (total_land * corn_cost - total_budget) / (corn_cost - wheat_cost)
  wheat_acres = 200 ∧
  wheat_acres > 0 ∧
  wheat_acres < total_land ∧
  wheat_acres * wheat_cost + (total_land - wheat_acres) * corn_cost = total_budget :=
by sorry

end NUMINAMATH_CALUDE_johnson_farm_wheat_acreage_l512_51245


namespace NUMINAMATH_CALUDE_rectangle_max_area_l512_51246

theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 22 →
  l * w ≤ 121 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l512_51246


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l512_51294

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero :
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l512_51294


namespace NUMINAMATH_CALUDE_no_savings_l512_51248

-- Define the prices and fees
def in_store_price : ℚ := 129.99
def online_payment : ℚ := 29.99
def shipping_fee : ℚ := 11.99

-- Define the number of online payments
def num_payments : ℕ := 4

-- Define the function to calculate savings in cents
def savings_in_cents : ℚ :=
  (in_store_price - (num_payments * online_payment + shipping_fee)) * 100

-- Theorem statement
theorem no_savings : savings_in_cents = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_savings_l512_51248


namespace NUMINAMATH_CALUDE_simplify_expression_l512_51239

theorem simplify_expression (x y : ℝ) : 
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l512_51239


namespace NUMINAMATH_CALUDE_boat_downstream_time_l512_51237

/-- Proves that a boat traveling downstream takes 1 hour to cover 45 km,
    given its speed in still water and the stream's speed. -/
theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 40)
  (h2 : stream_speed = 5)
  (h3 : distance = 45) :
  distance / (boat_speed + stream_speed) = 1 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l512_51237


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l512_51254

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 6 = 0 ∧ x = 1) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l512_51254


namespace NUMINAMATH_CALUDE_largest_number_l512_51205

theorem largest_number (a b c d e : ℝ) : 
  a = 17231 + 1 / 3251 →
  b = 17231 - 1 / 3251 →
  c = 17231 * (1 / 3251) →
  d = 17231 / (1 / 3251) →
  e = 17231.3251 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l512_51205


namespace NUMINAMATH_CALUDE_hedge_sections_count_l512_51255

def section_blocks : ℕ := 30
def block_cost : ℚ := 2
def total_cost : ℚ := 480

theorem hedge_sections_count :
  (total_cost / (section_blocks * block_cost) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_hedge_sections_count_l512_51255


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l512_51260

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

theorem parallel_perpendicular_implication 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β := sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l512_51260


namespace NUMINAMATH_CALUDE_remainder_of_least_number_l512_51256

theorem remainder_of_least_number (n : ℕ) (h1 : n = 261) (h2 : ∀ m < n, m % 37 ≠ n % 37 ∨ m % 7 ≠ n % 7) : n % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_least_number_l512_51256


namespace NUMINAMATH_CALUDE_factorize_x_squared_plus_2x_l512_51284

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x_squared_plus_2x_l512_51284


namespace NUMINAMATH_CALUDE_trajectory_equation_l512_51283

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The point P -/
def P : ℝ × ℝ := (2, 2)

/-- A point is on the trajectory if it's the midpoint of a line segment AB,
    where A and B are intersection points of a line through P and the ellipse -/
def on_trajectory (M : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    (∃ (t : ℝ), A = P + t • (B - P)) ∧
    M = (A + B) / 2

/-- The theorem stating the equation of the trajectory -/
theorem trajectory_equation (x y : ℝ) :
  on_trajectory (x, y) → (x - 1)^2 + 2*(y - 1)^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l512_51283


namespace NUMINAMATH_CALUDE_tan_675_degrees_l512_51235

theorem tan_675_degrees (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (↑m * π / 180) = Real.tan (675 * π / 180) → m = 135 :=
by sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l512_51235


namespace NUMINAMATH_CALUDE_square_sum_reciprocals_l512_51278

theorem square_sum_reciprocals (x y : ℝ) 
  (h : 1 / x - 1 / (2 * y) = 1 / (2 * x + y)) : 
  y^2 / x^2 + x^2 / y^2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocals_l512_51278


namespace NUMINAMATH_CALUDE_angle_is_15_degrees_l512_51290

-- Define the triangle MIT
structure Triangle :=
  (M I T : ℝ × ℝ)

-- Define the points X, Y, O, P
structure Points :=
  (X Y O P : ℝ × ℝ)

def angle_MOP (t : Triangle) (p : Points) : ℝ :=
  sorry  -- The actual calculation of the angle

-- Main theorem
theorem angle_is_15_degrees 
  (t : Triangle) 
  (p : Points) 
  (h1 : t.M.1 = 0 ∧ t.M.2 = 0)  -- Assume M is at (0,0)
  (h2 : t.I.1 = 12 ∧ t.I.2 = 0)  -- MI = 12
  (h3 : (t.M.1 - p.X.1)^2 + (t.M.2 - p.X.2)^2 = 4)  -- MX = 2
  (h4 : (t.I.1 - p.Y.1)^2 + (t.I.2 - p.Y.2)^2 = 4)  -- YI = 2
  (h5 : p.O = ((t.M.1 + t.I.1)/2, (t.M.2 + t.I.2)/2))  -- O is midpoint of MI
  (h6 : p.P = ((p.X.1 + p.Y.1)/2, (p.X.2 + p.Y.2)/2))  -- P is midpoint of XY
  : angle_MOP t p = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_is_15_degrees_l512_51290


namespace NUMINAMATH_CALUDE_expression_evaluation_l512_51277

theorem expression_evaluation :
  (3^2 - 3*2) - (4^2 - 4*2) + (5^2 - 5*2) - (6^2 - 6*2) = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l512_51277


namespace NUMINAMATH_CALUDE_tourist_attraction_temperature_difference_l512_51271

/-- The temperature difference between the highest and lowest temperatures -/
def temperature_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Proof that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem tourist_attraction_temperature_difference :
  temperature_difference 8 (-2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tourist_attraction_temperature_difference_l512_51271


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l512_51293

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l512_51293


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l512_51225

/-- Represents the partnership profit calculation problem --/
theorem partnership_profit_calculation
  (p q r : ℕ) -- Initial capitals
  (h_ratio : p / q = 3 / 2 ∧ q / r = 4 / 3) -- Initial capital ratio
  (h_p_withdraw : ℕ) -- Amount p withdraws after 2 months
  (h_q_share : ℕ) -- q's share of profit in rupees
  (h_duration : ℕ) -- Total duration of partnership in months
  (h_p_withdraw_time : ℕ) -- Time after which p withdraws half capital
  (h_p_withdraw_half : h_p_withdraw = p / 2) -- p withdraws half of initial capital
  (h_duration_val : h_duration = 12) -- Total duration is 12 months
  (h_p_withdraw_time_val : h_p_withdraw_time = 2) -- p withdraws after 2 months
  (h_q_share_val : h_q_share = 144) -- q's share is Rs 144
  : ∃ (total_profit : ℕ), total_profit = 486 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l512_51225


namespace NUMINAMATH_CALUDE_remainder_puzzle_l512_51207

theorem remainder_puzzle : (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_puzzle_l512_51207


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l512_51206

theorem problem_1 : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |-(1/2)| = 2 := by sorry

theorem problem_2 (a : ℝ) (h : a = 2) : 
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l512_51206


namespace NUMINAMATH_CALUDE_linear_equation_condition_l512_51269

/-- The equation (m-1)x^|m|+4=0 is linear if and only if m = -1 -/
theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, (m - 1 : ℝ) * |x|^|m| + 4 = a * x + b) ↔ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l512_51269


namespace NUMINAMATH_CALUDE_airplane_rows_l512_51298

/-- 
Given an airplane with the following conditions:
- Each row has 8 seats
- Only 3/4 of the seats in each row can be occupied
- There are 24 unoccupied seats on the plane

This theorem proves that the number of rows on the airplane is 12.
-/
theorem airplane_rows : 
  ∀ (rows : ℕ), 
  (8 : ℚ) * rows - (3 / 4 : ℚ) * 8 * rows = 24 → 
  rows = 12 := by
sorry

end NUMINAMATH_CALUDE_airplane_rows_l512_51298


namespace NUMINAMATH_CALUDE_seven_number_sequence_average_l512_51286

theorem seven_number_sequence_average (a b c d e f g : ℝ) :
  (a + b + c + d) / 4 = 4 →
  (d + e + f + g) / 4 = 4 →
  d = 11 →
  (a + b + c + d + e + f + g) / 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_seven_number_sequence_average_l512_51286


namespace NUMINAMATH_CALUDE_min_value_and_integral_bound_l512_51263

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x * Real.log x

-- Define the integral G
def G (a b : ℝ) : ℝ := ∫ x in a..b, |Real.log x - Real.log ((a + b) / 2)|

-- State the theorem
theorem min_value_and_integral_bound 
  (k : ℝ) (a b : ℝ) (h1 : k ≠ 0) (h2 : 0 < a) (h3 : a < b) :
  (∃ (x : ℝ), f k x = -1 / Real.exp 1 ∧ ∀ (y : ℝ), f k y ≥ -1 / Real.exp 1) →
  (k = 1 ∧ 
   G a b = a * Real.log a + b * Real.log b - (a + b) * Real.log ((a + b) / 2) ∧
   G a b / (b - a) < Real.log 2) := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_and_integral_bound_l512_51263


namespace NUMINAMATH_CALUDE_distance_from_T_to_S_l512_51219

theorem distance_from_T_to_S (P Q : ℝ) : 
  let S := P + (3/4) * (Q - P)
  let T := P + (1/3) * (Q - P)
  S - T = 25 := by
sorry

end NUMINAMATH_CALUDE_distance_from_T_to_S_l512_51219


namespace NUMINAMATH_CALUDE_existence_of_special_number_l512_51240

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l512_51240


namespace NUMINAMATH_CALUDE_xia_initial_stickers_l512_51220

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Theorem: Xia had 150 stickers at the beginning -/
theorem xia_initial_stickers :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 := by
  sorry

end NUMINAMATH_CALUDE_xia_initial_stickers_l512_51220


namespace NUMINAMATH_CALUDE_common_roots_product_l512_51229

/-- Given two polynomials that share exactly two roots, prove that the product of these common roots is 1/3 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s t u : ℝ),
    (∀ x : ℝ, x^4 - 3*x^3 + C*x + 24 = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    (∀ x : ℝ, x^4 - D*x^3 + 4*x^2 + 72 = (x - p)*(x - q)*(x - t)*(x - u)) ∧
    p ≠ q ∧ 
    p * q = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l512_51229


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l512_51200

theorem complex_fraction_equals_neg_i : (1 - 2*I) / (2 + I) = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l512_51200


namespace NUMINAMATH_CALUDE_radical_axes_theorem_l512_51280

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The radical axis of two circles --/
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 - c1.radius^2 = 
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 - c2.radius^2}

/-- Three lines are either coincident, parallel, or concurrent --/
def linesCoincidentParallelOrConcurrent (l1 l2 l3 : Set (ℝ × ℝ)) : Prop :=
  (l1 = l2 ∧ l2 = l3) ∨ 
  (∀ p1 ∈ l1, ∀ p2 ∈ l2, ∀ p3 ∈ l3, 
    (p1.1 - p2.1) * (p3.2 - p2.2) = (p3.1 - p2.1) * (p1.2 - p2.2)) ∨
  (∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)

/-- The Theorem of Radical Axes --/
theorem radical_axes_theorem (c1 c2 c3 : Circle) :
  linesCoincidentParallelOrConcurrent 
    (radicalAxis c1 c2) 
    (radicalAxis c2 c3) 
    (radicalAxis c3 c1) :=
sorry

end NUMINAMATH_CALUDE_radical_axes_theorem_l512_51280


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l512_51261

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  let givenLine : Line2D := { a := 2, b := 1, c := -3 }
  let pointA : Point2D := { x := 0, y := 4 }
  let resultLine : Line2D := { a := 1, b := -2, c := 8 }
  perpendicularLines givenLine resultLine ∧
  pointOnLine pointA resultLine := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l512_51261


namespace NUMINAMATH_CALUDE_binomial_20_19_l512_51285

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l512_51285


namespace NUMINAMATH_CALUDE_p_at_5_l512_51282

/-- A monic quartic polynomial with specific values at x = 1, 2, 3, and 4 -/
def p : ℝ → ℝ :=
  fun x => x^4 + a*x^3 + b*x^2 + c*x + d
  where
    a : ℝ := sorry
    b : ℝ := sorry
    c : ℝ := sorry
    d : ℝ := sorry

/-- The polynomial p satisfies the given conditions -/
axiom p_cond1 : p 1 = 2
axiom p_cond2 : p 2 = 3
axiom p_cond3 : p 3 = 6
axiom p_cond4 : p 4 = 11

/-- The theorem to be proved -/
theorem p_at_5 : p 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_p_at_5_l512_51282


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l512_51230

theorem gcd_from_lcm_and_ratio (C D : ℕ+) : 
  C.lcm D = 180 → C.val * 6 = D.val * 5 → C.gcd D = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l512_51230


namespace NUMINAMATH_CALUDE_game_installation_time_ratio_l512_51265

theorem game_installation_time_ratio :
  ∀ (install_time : ℝ),
    install_time > 0 →
    10 + install_time + 3 * (10 + install_time) = 60 →
    install_time / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_game_installation_time_ratio_l512_51265


namespace NUMINAMATH_CALUDE_special_number_satisfies_conditions_special_number_unique_l512_51233

/-- A two-digit number that satisfies the given conditions -/
def special_number : ℕ := 50

/-- The property that defines our special number -/
def is_special_number (a : ℕ) : Prop :=
  (a ≥ 10 ∧ a ≤ 99) ∧  -- Two-digit number
  (∃ (q r : ℚ), 
    (101 * a - a^2) / (0.04 * a^2) = q + r ∧
    q = a / 2 ∧
    r = a / (0.04 * a^2))

theorem special_number_satisfies_conditions : 
  is_special_number special_number :=
sorry

theorem special_number_unique : 
  ∀ (n : ℕ), is_special_number n → n = special_number :=
sorry

end NUMINAMATH_CALUDE_special_number_satisfies_conditions_special_number_unique_l512_51233


namespace NUMINAMATH_CALUDE_second_number_calculation_l512_51287

theorem second_number_calculation (A B : ℝ) (h1 : A = 456) (h2 : 0.5 * A = 0.4 * B + 180) : B = 120 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l512_51287


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_four_digit_numbers_exist_l512_51226

def A (n m : ℕ) := n.factorial / (n - m).factorial

theorem four_digit_numbers_count : ℕ → Prop :=
  fun count => (count = A 5 4 - A 4 3) ∧ 
               (count = A 4 1 * A 4 3) ∧ 
               (count = A 4 4 + 3 * A 4 3) ∧ 
               (count ≠ A 5 4 - A 4 4)

theorem four_digit_numbers_exist : ∃ count : ℕ, four_digit_numbers_count count := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_four_digit_numbers_exist_l512_51226


namespace NUMINAMATH_CALUDE_distance_between_anastasia_and_bananastasia_l512_51232

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 343

/-- The time difference in seconds between hearing Anastasia and Bananastasia when they yell simultaneously -/
def simultaneous_time_diff : ℝ := 5

/-- The time difference in seconds between hearing Bananastasia and Anastasia when Bananastasia yells first -/
def sequential_time_diff : ℝ := 5

/-- The distance between Anastasia and Bananastasia in meters -/
def distance : ℝ := 1715

theorem distance_between_anastasia_and_bananastasia :
  ∀ (d : ℝ),
  (d / speed_of_sound = simultaneous_time_diff) ∧
  (2 * d / speed_of_sound - d / speed_of_sound = sequential_time_diff) →
  d = distance := by
  sorry

end NUMINAMATH_CALUDE_distance_between_anastasia_and_bananastasia_l512_51232


namespace NUMINAMATH_CALUDE_paint_remaining_rooms_l512_51228

def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

theorem paint_remaining_rooms 
  (total_rooms : ℕ) 
  (time_per_room : ℕ) 
  (painted_rooms : ℕ) 
  (h1 : total_rooms = 11) 
  (h2 : time_per_room = 7) 
  (h3 : painted_rooms = 2) : 
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 63 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_rooms_l512_51228


namespace NUMINAMATH_CALUDE_unchanged_total_plates_l512_51224

/-- Represents the number of elements in each set of letters for license plates --/
structure LicensePlateSets :=
  (first : Nat)
  (second : Nat)
  (third : Nat)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LicensePlateSets) : Nat :=
  sets.first * sets.second * sets.third

/-- The original configuration of letter sets --/
def originalSets : LicensePlateSets :=
  { first := 5, second := 3, third := 4 }

/-- The new configuration after moving one letter from the first to the third set --/
def newSets : LicensePlateSets :=
  { first := 4, second := 3, third := 5 }

/-- Theorem stating that the total number of license plates remains unchanged --/
theorem unchanged_total_plates :
  totalPlates originalSets = totalPlates newSets :=
by sorry

end NUMINAMATH_CALUDE_unchanged_total_plates_l512_51224


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l512_51275

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 57 / 99 ∧ (∀ n : ℕ, (x * 10^(2*n+2) - ⌊x * 10^(2*n+2)⌋ : ℚ) = 57 / 100) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l512_51275


namespace NUMINAMATH_CALUDE_benny_eggs_dozens_l512_51281

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Benny bought -/
def total_eggs : ℕ := 84

/-- The number of dozens of eggs Benny bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem benny_eggs_dozens : dozens_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_benny_eggs_dozens_l512_51281


namespace NUMINAMATH_CALUDE_winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l512_51262

-- Define a temperature range
structure TemperatureRange where
  min : ℤ
  max : ℤ
  h : min ≤ max

-- Define a predicate for a scenario using negative numbers
def usesNegativeNumbers (range : TemperatureRange) : Prop :=
  range.min < 0

-- Theorem: The given temperature range uses negative numbers
theorem winter_temperature_uses_negative_numbers :
  ∃ (range : TemperatureRange), usesNegativeNumbers range :=
by
  -- The proof would go here
  sorry

-- Example of the temperature range mentioned in the solution
def winter_day_range : TemperatureRange :=
  { min := -2
  , max := 5
  , h := by norm_num }

-- Theorem: The specific winter day range uses negative numbers
theorem specific_winter_day_uses_negative_numbers :
  usesNegativeNumbers winter_day_range :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_winter_temperature_uses_negative_numbers_specific_winter_day_uses_negative_numbers_l512_51262


namespace NUMINAMATH_CALUDE_product_11_cubed_sum_l512_51274

theorem product_11_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 11^3 → a + b + c = 133 := by sorry

end NUMINAMATH_CALUDE_product_11_cubed_sum_l512_51274


namespace NUMINAMATH_CALUDE_spade_calculation_l512_51243

-- Define the ⬥ operation
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 6 5) = -112 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l512_51243


namespace NUMINAMATH_CALUDE_probability_B3_l512_51221

structure Box where
  number : Nat
  balls : List Nat

def initial_boxes : List Box := [
  ⟨1, [1, 1, 2, 3]⟩,
  ⟨2, [1, 1, 3]⟩,
  ⟨3, [1, 1, 1, 2, 2]⟩
]

def draw_and_transfer (boxes : List Box) : List Box := sorry

def second_draw (boxes : List Box) : ℝ := sorry

theorem probability_B3 (boxes : List Box) :
  boxes = initial_boxes →
  second_draw (draw_and_transfer boxes) = 13/48 := by sorry

end NUMINAMATH_CALUDE_probability_B3_l512_51221


namespace NUMINAMATH_CALUDE_smallest_cube_root_plus_small_fraction_l512_51279

theorem smallest_cube_root_plus_small_fraction (m n : ℕ) (r : ℝ) : 
  (m > 0) →
  (n > 0) →
  (r > 0) →
  (r < 1/500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ m' n' r', m' > 0 → n' > 0 → r' > 0 → r' < 1/500 → (m' : ℝ)^(1/3) = n' + r' → m' ≥ m) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_plus_small_fraction_l512_51279


namespace NUMINAMATH_CALUDE_book_arrangement_l512_51249

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  (n! / k!) = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l512_51249


namespace NUMINAMATH_CALUDE_expand_polynomial_l512_51276

theorem expand_polynomial (x : ℝ) : (1 + x^3) * (1 - x^4 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l512_51276


namespace NUMINAMATH_CALUDE_abs_cube_plus_cube_equals_two_cube_l512_51209

theorem abs_cube_plus_cube_equals_two_cube (x : ℝ) : |x^3| + x^3 = 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_abs_cube_plus_cube_equals_two_cube_l512_51209


namespace NUMINAMATH_CALUDE_marble_probability_l512_51213

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 5

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 7

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := blue_marbles + white_marbles + red_marbles

/-- The number of marbles to be drawn -/
def marbles_drawn : ℕ := total_marbles - 2

/-- The probability of having one white and one blue marble remaining after randomly drawing marbles until only two are left -/
theorem marble_probability : 
  (Nat.choose blue_marbles blue_marbles * Nat.choose white_marbles (white_marbles - 1) * Nat.choose red_marbles red_marbles) / 
  Nat.choose total_marbles marbles_drawn = 7 / 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l512_51213


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l512_51223

theorem smallest_k_for_inequality : ∃ (k : ℕ), k = 4 ∧ 
  (∀ (a : ℝ) (n : ℕ), a ∈ Set.Icc 0 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (a : ℝ) (n : ℕ), a ∈ Set.Icc 0 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l512_51223
