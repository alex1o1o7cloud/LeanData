import Mathlib

namespace NUMINAMATH_CALUDE_josh_remaining_money_l2852_285262

def initial_amount : ℝ := 100

def transactions : List ℝ := [12.67, 25.39, 14.25, 4.32, 27.50]

def remaining_money : ℝ := initial_amount - transactions.sum

theorem josh_remaining_money :
  remaining_money = 15.87 := by sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l2852_285262


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2852_285277

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - (k - 1) * x - 2 * k + 8 > 0) ↔ -9 < k ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l2852_285277


namespace NUMINAMATH_CALUDE_points_on_line_implies_a_equals_two_l2852_285257

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 1)
def C (a : ℝ) : ℝ × ℝ := (-4, 2*a)

-- Define the condition for points being on the same line
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem points_on_line_implies_a_equals_two :
  collinear A B (C a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_implies_a_equals_two_l2852_285257


namespace NUMINAMATH_CALUDE_green_notebook_cost_l2852_285242

theorem green_notebook_cost (total_cost black_cost pink_cost : ℕ) 
  (h1 : total_cost = 45)
  (h2 : black_cost = 15)
  (h3 : pink_cost = 10) :
  (total_cost - (black_cost + pink_cost)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_green_notebook_cost_l2852_285242


namespace NUMINAMATH_CALUDE_factorization_equality_l2852_285295

theorem factorization_equality (x y : ℝ) : -2*x^2 + 4*x*y - 2*y^2 = -2*(x-y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2852_285295


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2852_285222

/-- Given a parabola with equation x = (1/(4*m)) * y^2, prove that its focus has coordinates (m, 0) -/
theorem parabola_focus_coordinates (m : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  let focus := (m, 0)
  focus ∈ parabola ∧ ∀ p ∈ parabola, ∃ d : ℝ, (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2852_285222


namespace NUMINAMATH_CALUDE_amy_money_left_l2852_285204

/-- Calculates the amount of money Amy had when she left the fair. -/
def money_left (initial_amount spent : ℕ) : ℕ :=
  initial_amount - spent

/-- Proves that Amy had $11 when she left the fair. -/
theorem amy_money_left :
  money_left 15 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_amy_money_left_l2852_285204


namespace NUMINAMATH_CALUDE_coin_value_theorem_l2852_285241

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins in cents given the number of quarters and dimes -/
def total_value (quarters dimes : ℕ) : ℕ := quarter_value * quarters + dime_value * dimes

/-- Calculates the total value of coins in cents if quarters and dimes were swapped -/
def swapped_value (quarters dimes : ℕ) : ℕ := dime_value * quarters + quarter_value * dimes

theorem coin_value_theorem (quarters dimes : ℕ) :
  quarters + dimes = 30 →
  swapped_value quarters dimes = total_value quarters dimes + 150 →
  total_value quarters dimes = 450 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_theorem_l2852_285241


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l2852_285219

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ring arrangements on four fingers -/
def ring_arrangements : ℕ :=
  let total_rings := 10
  let chosen_rings := 7
  let fingers := 4
  binomial total_rings chosen_rings * 
  factorial chosen_rings * 
  chosen_rings * 
  binomial (chosen_rings + fingers - 2) (fingers - 1)

theorem ring_arrangement_count : ring_arrangements = 264537600 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l2852_285219


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2852_285261

theorem geometric_sequence_product (a b : ℝ) : 
  (5 < a) → (a < b) → (b < 40) → 
  (b / a = a / 5) → (40 / b = b / a) → 
  a * b = 200 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2852_285261


namespace NUMINAMATH_CALUDE_min_value_expression_l2852_285299

theorem min_value_expression (x y z : ℝ) (h1 : x * y ≠ 0) (h2 : x + y ≠ 0) :
  ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 ≥ 5 ∧
  ∃ (x y z : ℝ), x * y ≠ 0 ∧ x + y ≠ 0 ∧
    ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2852_285299


namespace NUMINAMATH_CALUDE_handshake_count_l2852_285268

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of teams -/
def num_teams : ℕ := 2

/-- The number of referees -/
def num_referees : ℕ := 2

/-- The total number of handshakes -/
def total_handshakes : ℕ :=
  -- Handshakes between teams
  team_size * team_size +
  -- Handshakes within each team
  num_teams * (team_size.choose 2) +
  -- Handshakes with referees
  (num_teams * team_size) * num_referees

theorem handshake_count : total_handshakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2852_285268


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l2852_285291

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the four vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ :=
  2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l2852_285291


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2852_285217

theorem opposite_sides_line_range (a : ℝ) : 
  (0 + 0 < a ∧ a < 1 + 1) ∨ (0 + 0 > a ∧ a > 1 + 1) ↔ a < 0 ∨ a > 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2852_285217


namespace NUMINAMATH_CALUDE_least_three_digit_9_heavy_l2852_285240

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem least_three_digit_9_heavy : 
  (∀ n : ℕ, is_three_digit n → is_9_heavy n → 105 ≤ n) ∧ 
  is_three_digit 105 ∧ 
  is_9_heavy 105 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_9_heavy_l2852_285240


namespace NUMINAMATH_CALUDE_card_distribution_count_l2852_285205

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def card_distribution : ℕ :=
  let n_cards : ℕ := 6
  let n_envelopes : ℕ := 3
  let cards_per_envelope : ℕ := 2
  let n_free_cards : ℕ := n_cards - 2  -- A and B are treated as one unit
  let ways_to_distribute_remaining : ℕ := Nat.choose n_free_cards cards_per_envelope
  let envelope_choices_for_ab : ℕ := n_envelopes
  ways_to_distribute_remaining * envelope_choices_for_ab

/-- Theorem stating that the number of card distributions is 18 -/
theorem card_distribution_count : card_distribution = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_count_l2852_285205


namespace NUMINAMATH_CALUDE_minimum_students_l2852_285259

theorem minimum_students (n : ℕ) (h1 : n > 1000) 
  (h2 : n % 10 = 0) (h3 : n % 14 = 0) (h4 : n % 18 = 0) :
  n ≥ 1260 := by
  sorry

end NUMINAMATH_CALUDE_minimum_students_l2852_285259


namespace NUMINAMATH_CALUDE_calculation_proof_l2852_285225

theorem calculation_proof :
  (1 / (Real.sqrt 5 + 2) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5) = 2) ∧
  (2 * Real.sqrt 3 * 612 * (3 + 3/2) = 5508 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2852_285225


namespace NUMINAMATH_CALUDE_steve_socks_l2852_285298

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 48 → matching_pairs = 11 → mismatching_socks = total_socks - 2 * matching_pairs → mismatching_socks = 26 := by
  sorry

end NUMINAMATH_CALUDE_steve_socks_l2852_285298


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2852_285203

theorem P_sufficient_not_necessary_for_Q :
  (∀ a : ℝ, a > 1 → (a - 1) * (a + 1) > 0) ∧
  (∃ a : ℝ, (a - 1) * (a + 1) > 0 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2852_285203


namespace NUMINAMATH_CALUDE_abc_product_l2852_285246

theorem abc_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 432 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2852_285246


namespace NUMINAMATH_CALUDE_jane_average_score_l2852_285294

def jane_scores : List ℝ := [89, 95, 88, 92, 94, 87]

theorem jane_average_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.8333 := by
  sorry

end NUMINAMATH_CALUDE_jane_average_score_l2852_285294


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2852_285236

/-- Given a triangle with sides 6, 10, and 11, prove that an equilateral triangle
    with the same perimeter has side length 9. -/
theorem equilateral_triangle_side_length : 
  ∀ (a b c s : ℝ), 
    a = 6 → b = 10 → c = 11 →  -- Given triangle side lengths
    3 * s = a + b + c →        -- Equilateral triangle has same perimeter
    s = 9 :=                   -- Side length of equilateral triangle is 9
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2852_285236


namespace NUMINAMATH_CALUDE_joan_seashells_l2852_285297

theorem joan_seashells (seashells_left seashells_given_to_sam : ℕ) 
  (h1 : seashells_left = 27) 
  (h2 : seashells_given_to_sam = 43) : 
  seashells_left + seashells_given_to_sam = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2852_285297


namespace NUMINAMATH_CALUDE_three_fourths_of_forty_l2852_285209

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_forty_l2852_285209


namespace NUMINAMATH_CALUDE_sample_average_l2852_285229

theorem sample_average (x : ℝ) : 
  (1 + 3 + 2 + 5 + x) / 5 = 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sample_average_l2852_285229


namespace NUMINAMATH_CALUDE_total_amount_theorem_l2852_285234

def calculate_selling_price (purchase_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  purchase_price * (1 - loss_percentage / 100)

def total_amount_received (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  calculate_selling_price price1 loss1 +
  calculate_selling_price price2 loss2 +
  calculate_selling_price price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  total_amount_received price1 price2 price3 loss1 loss2 loss3 = 1780 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l2852_285234


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_minus_z_squared_l2852_285290

theorem x_squared_minus_y_squared_minus_z_squared (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (diff_eq : x - y = 4)
  (yz_sum : y + z = 7) :
  x^2 - y^2 - z^2 = -12 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_minus_z_squared_l2852_285290


namespace NUMINAMATH_CALUDE_three_prime_divisors_theorem_l2852_285285

theorem three_prime_divisors_theorem (x n : ℕ) :
  x = 2^n - 32 ∧
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
  sorry

end NUMINAMATH_CALUDE_three_prime_divisors_theorem_l2852_285285


namespace NUMINAMATH_CALUDE_T_is_perfect_square_T_equals_fib_squared_l2852_285280

/-- A tetromino tile is formed by gluing together four unit square tiles, edge to edge. -/
def TetrominoTile : Type := Unit

/-- Tₙ is the number of ways to tile a 2×2n rectangular bathroom floor with tetromino tiles. -/
def T (n : ℕ) : ℕ := sorry

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The main theorem: Tₙ is always a perfect square, specifically Fₙ₊₁² -/
theorem T_is_perfect_square (n : ℕ) : ∃ k : ℕ, T n = k ^ 2 :=
  sorry

/-- The specific form of Tₙ in terms of Fibonacci numbers -/
theorem T_equals_fib_squared (n : ℕ) : T n = (fib (n + 1)) ^ 2 :=
  sorry

end NUMINAMATH_CALUDE_T_is_perfect_square_T_equals_fib_squared_l2852_285280


namespace NUMINAMATH_CALUDE_school_distance_proof_l2852_285252

/-- The time in hours it takes to drive to school during rush hour -/
def rush_hour_time : ℚ := 18 / 60

/-- The time in hours it takes to drive to school with no traffic -/
def no_traffic_time : ℚ := 12 / 60

/-- The speed increase in mph when there's no traffic -/
def speed_increase : ℚ := 20

/-- The distance to school in miles -/
def distance_to_school : ℚ := 12

theorem school_distance_proof :
  ∃ (rush_hour_speed : ℚ),
    rush_hour_speed * rush_hour_time = distance_to_school ∧
    (rush_hour_speed + speed_increase) * no_traffic_time = distance_to_school := by
  sorry

#check school_distance_proof

end NUMINAMATH_CALUDE_school_distance_proof_l2852_285252


namespace NUMINAMATH_CALUDE_algebraic_grid_difference_l2852_285201

/-- Represents a 3x3 grid of algebraic expressions -/
structure AlgebraicGrid (α : Type) [Ring α] where
  grid : Matrix (Fin 3) (Fin 3) α

/-- Checks if all rows, columns, and diagonals have the same sum -/
def isValidGrid {α : Type} [Ring α] (g : AlgebraicGrid α) : Prop :=
  let rowSum (i : Fin 3) := g.grid i 0 + g.grid i 1 + g.grid i 2
  let colSum (j : Fin 3) := g.grid 0 j + g.grid 1 j + g.grid 2 j
  let diag1Sum := g.grid 0 0 + g.grid 1 1 + g.grid 2 2
  let diag2Sum := g.grid 0 2 + g.grid 1 1 + g.grid 2 0
  ∀ i j : Fin 3, rowSum i = colSum j ∧ rowSum i = diag1Sum ∧ rowSum i = diag2Sum

theorem algebraic_grid_difference {α : Type} [CommRing α] (x : α) (M N : α) :
  let g : AlgebraicGrid α := {
    grid := λ i j =>
      if i = 0 ∧ j = 0 then M
      else if i = 0 ∧ j = 2 then x^2 - x - 1
      else if i = 1 ∧ j = 2 then x
      else if i = 2 ∧ j = 0 then x^2 - x
      else if i = 2 ∧ j = 1 then x - 1
      else if i = 2 ∧ j = 2 then N
      else 0  -- Other entries are not specified
  }
  isValidGrid g →
  M - N = -2*x^2 + 4*x :=
by
  sorry

end NUMINAMATH_CALUDE_algebraic_grid_difference_l2852_285201


namespace NUMINAMATH_CALUDE_square_root_problem_l2852_285276

theorem square_root_problem (a b : ℝ) 
  (h1 : (2 * a + 1) = 9)
  (h2 : (5 * a + 2 * b - 2) = 16) :
  (3 * a - 4 * b) = 16 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2852_285276


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l2852_285215

def scores : List ℕ := [97, 88, 85, 93, 85]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem scores_mode_and_median :
  mode scores = 85 ∧ median scores = 88 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l2852_285215


namespace NUMINAMATH_CALUDE_range_of_a_l2852_285238

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 4 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a)) ∧ (p a ∨ q a) → (e ≤ a ∧ a < 4) ∨ (a ≤ -4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2852_285238


namespace NUMINAMATH_CALUDE_pencil_cost_l2852_285213

theorem pencil_cost 
  (total_spent : ℚ)
  (notebook_price : ℚ)
  (ruler_pack_price : ℚ)
  (eraser_price : ℚ)
  (notebook_count : ℕ)
  (ruler_pack_count : ℕ)
  (eraser_count : ℕ)
  (pencil_count : ℕ)
  (notebook_discount : ℚ)
  (sales_tax : ℚ)
  (h1 : total_spent = 740/100)
  (h2 : notebook_price = 85/100)
  (h3 : ruler_pack_price = 60/100)
  (h4 : eraser_price = 20/100)
  (h5 : notebook_count = 2)
  (h6 : ruler_pack_count = 1)
  (h7 : eraser_count = 5)
  (h8 : pencil_count = 4)
  (h9 : notebook_discount = 15/100)
  (h10 : sales_tax = 10/100) :
  ∃ (pencil_price : ℚ), pencil_price = 99/100 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_l2852_285213


namespace NUMINAMATH_CALUDE_orange_juice_problem_l2852_285260

/-- Calculates the number of servings of orange juice prepared from concentrate -/
def orange_juice_servings (concentrate_cans : ℕ) (concentrate_oz_per_can : ℕ) 
  (water_cans_per_concentrate : ℕ) (oz_per_serving : ℕ) : ℕ :=
  let total_oz := concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)
  total_oz / oz_per_serving

/-- Theorem stating that 60 cans of 5-oz concentrate mixed with 3 cans of water per
    1 can of concentrate yields 200 servings of 6-oz orange juice -/
theorem orange_juice_problem : 
  orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_problem_l2852_285260


namespace NUMINAMATH_CALUDE_reflection_after_translation_l2852_285235

/-- Given a point A with coordinates (-3, -2), prove that translating it 5 units
    to the right and then reflecting across the y-axis results in a point with
    coordinates (-2, -2). -/
theorem reflection_after_translation :
  let A : ℝ × ℝ := (-3, -2)
  let B : ℝ × ℝ := (A.1 + 5, A.2)
  let B' : ℝ × ℝ := (-B.1, B.2)
  B' = (-2, -2) := by
sorry

end NUMINAMATH_CALUDE_reflection_after_translation_l2852_285235


namespace NUMINAMATH_CALUDE_tilly_bag_cost_l2852_285200

/-- Calculates the cost per bag for Tilly's business --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (total_profit : ℚ) : ℚ :=
  (num_bags * selling_price - total_profit) / num_bags

/-- Proves that the cost per bag is $7 given the problem conditions --/
theorem tilly_bag_cost :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tilly_bag_cost_l2852_285200


namespace NUMINAMATH_CALUDE_voting_theorem_l2852_285228

/-- Represents the number of students voting for each issue and against all issues -/
structure VotingData where
  total : ℕ
  issueA : ℕ
  issueB : ℕ
  issueC : ℕ
  againstAll : ℕ

/-- Calculates the number of students voting for all three issues -/
def studentsVotingForAll (data : VotingData) : ℕ :=
  data.issueA + data.issueB + data.issueC - data.total + data.againstAll

/-- Theorem stating the number of students voting for all three issues -/
theorem voting_theorem (data : VotingData) 
    (h1 : data.total = 300)
    (h2 : data.issueA = 210)
    (h3 : data.issueB = 190)
    (h4 : data.issueC = 160)
    (h5 : data.againstAll = 40) :
  studentsVotingForAll data = 80 := by
  sorry

#eval studentsVotingForAll { total := 300, issueA := 210, issueB := 190, issueC := 160, againstAll := 40 }

end NUMINAMATH_CALUDE_voting_theorem_l2852_285228


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2852_285212

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 18 * x^2 - m * x + 252 = 0) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 18 * y^2 - k * y + 252 = 0) ∧ 
  m = 162 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l2852_285212


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2852_285207

theorem expand_and_simplify (x : ℝ) : (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2852_285207


namespace NUMINAMATH_CALUDE_seventy_million_scientific_notation_l2852_285281

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seventy_million_scientific_notation :
  toScientificNotation 70000000 = ScientificNotation.mk 7.0 7 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_seventy_million_scientific_notation_l2852_285281


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2852_285274

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2852_285274


namespace NUMINAMATH_CALUDE_circle_parameters_l2852_285286

/-- Definition of the circle C -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- Definition of a point being on the circle -/
def point_on_circle (x y a b : ℝ) : Prop :=
  circle_equation x y a b

/-- Definition of the line x + y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Definition of symmetric point with respect to the line x + y - 1 = 0 -/
def symmetric_point (x y x' y' : ℝ) : Prop :=
  x' + y' = x + y ∧ line_equation ((x + x')/2) ((y + y')/2)

/-- Main theorem -/
theorem circle_parameters :
  ∀ (a b : ℝ),
    (point_on_circle 2 1 a b) →
    (∃ (x' y' : ℝ), symmetric_point 2 1 x' y' ∧ point_on_circle x' y' a b) →
    (a = 0 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_circle_parameters_l2852_285286


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2852_285254

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = Real.sqrt 6 →
  A = 2 * π / 3 →
  (a / Real.sin A = b / Real.sin B) →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2852_285254


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l2852_285282

def sum_and_reciprocal_sum_equal (s : Finset ℝ) (n : ℕ) (v : ℝ) : Prop :=
  s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum id = v) ∧ (s.sum (λ x => 1 / x) = v)

theorem max_value_x_plus_reciprocal 
  (s : Finset ℝ) (h : sum_and_reciprocal_sum_equal s 1001 1002) :
  ∀ x ∈ s, x + 1/x ≤ 4007/1002 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l2852_285282


namespace NUMINAMATH_CALUDE_g_equiv_l2852_285250

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Define the function g using f
def g (x : ℝ) : ℝ := 2 * (f x) - 19

-- Theorem stating that g(x) is equivalent to 6x - 29
theorem g_equiv : ∀ x : ℝ, g x = 6 * x - 29 := by
  sorry

end NUMINAMATH_CALUDE_g_equiv_l2852_285250


namespace NUMINAMATH_CALUDE_expand_product_l2852_285267

theorem expand_product (x y : ℝ) : (x + 3) * (x + 2*y + 4) = x^2 + 7*x + 2*x*y + 6*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2852_285267


namespace NUMINAMATH_CALUDE_solve_group_size_l2852_285296

def group_size_problem (n : ℕ) : Prop :=
  let weight_increase_per_person : ℚ := 5/2
  let weight_difference : ℕ := 20
  (weight_difference : ℚ) = n * weight_increase_per_person

theorem solve_group_size : ∃ n : ℕ, group_size_problem n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_group_size_l2852_285296


namespace NUMINAMATH_CALUDE_line_through_point_l2852_285269

/-- Given a line ax + (a+1)y = a + 4 passing through the point (3, -7), prove that a = -11/5 --/
theorem line_through_point (a : ℚ) : 
  (a * 3 + (a + 1) * (-7) = a + 4) → a = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2852_285269


namespace NUMINAMATH_CALUDE_remainder_of_power_divided_by_polynomial_l2852_285256

theorem remainder_of_power_divided_by_polynomial (x : ℤ) :
  (x + 1)^2010 ≡ 1 [ZMOD (x^2 + x + 1)] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_divided_by_polynomial_l2852_285256


namespace NUMINAMATH_CALUDE_added_amount_after_doubling_l2852_285272

theorem added_amount_after_doubling (x y : ℕ) : 
  x = 17 → 3 * (2 * x + y) = 117 → y = 5 := by sorry

end NUMINAMATH_CALUDE_added_amount_after_doubling_l2852_285272


namespace NUMINAMATH_CALUDE_pi_estimation_l2852_285292

theorem pi_estimation (m n : ℕ) (h : m > 0) : 
  ∃ (ε : ℝ), ε > 0 ∧ |4 * (n : ℝ) / (m : ℝ) - π| < ε :=
sorry

end NUMINAMATH_CALUDE_pi_estimation_l2852_285292


namespace NUMINAMATH_CALUDE_larger_number_value_l2852_285239

theorem larger_number_value (x y : ℝ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : y - x = 10) : 
  y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_value_l2852_285239


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l2852_285233

/-- The polar coordinate equation of the line segment y = 1 - x (0 ≤ x ≤ 1) -/
theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (0 ≤ x) ∧ (x ≤ 1) ∧ (y = 1 - x) →
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (ρ = 1 / (Real.cos θ + Real.sin θ)) ∧ (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l2852_285233


namespace NUMINAMATH_CALUDE_little_twelve_conference_games_l2852_285245

/-- Calculates the number of games in a football conference with specified rules -/
def conference_games (num_divisions : ℕ) (teams_per_division : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) : ℕ :=
  let intra_division_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_div_games
  let inter_division_games := (num_divisions * teams_per_division) * (teams_per_division * (num_divisions - 1)) * inter_div_games / 2
  intra_division_games + inter_division_games

/-- The Little Twelve Football Conference scheduling theorem -/
theorem little_twelve_conference_games :
  conference_games 2 6 3 2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_conference_games_l2852_285245


namespace NUMINAMATH_CALUDE_easter_egg_hunt_problem_l2852_285221

/-- Represents the number of eggs of each size found by a child -/
structure EggCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total points for a given EggCount -/
def totalPoints (eggs : EggCount) : ℕ :=
  eggs.small + 3 * eggs.medium + 5 * eggs.large

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt_problem :
  let kevin := EggCount.mk 5 0 3
  let bonnie := EggCount.mk 13 7 2
  let george := EggCount.mk 9 6 1
  let cheryl := EggCount.mk 56 30 15
  totalPoints cheryl - (totalPoints kevin + totalPoints bonnie + totalPoints george) = 125 := by
  sorry


end NUMINAMATH_CALUDE_easter_egg_hunt_problem_l2852_285221


namespace NUMINAMATH_CALUDE_loan_problem_l2852_285216

/-- Proves that given the conditions of the loan problem, the second part is lent for 3 years -/
theorem loan_problem (total : ℝ) (second_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (n : ℝ) : 
  total = 2717 →
  second_part = 1672 →
  rate1 = 0.03 →
  rate2 = 0.05 →
  time1 = 8 →
  (total - second_part) * rate1 * time1 = second_part * rate2 * n →
  n = 3 := by
sorry


end NUMINAMATH_CALUDE_loan_problem_l2852_285216


namespace NUMINAMATH_CALUDE_coconuts_per_crab_calculation_l2852_285226

/-- The number of coconuts Max has -/
def total_coconuts : ℕ := 342

/-- The number of goats Max will have after conversion -/
def total_goats : ℕ := 19

/-- The number of crabs that can be traded for a goat -/
def crabs_per_goat : ℕ := 6

/-- The number of coconuts needed to trade for a crab -/
def coconuts_per_crab : ℕ := 3

theorem coconuts_per_crab_calculation :
  coconuts_per_crab * crabs_per_goat * total_goats = total_coconuts :=
sorry

end NUMINAMATH_CALUDE_coconuts_per_crab_calculation_l2852_285226


namespace NUMINAMATH_CALUDE_product_w_z_is_24_l2852_285266

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  ef : ℝ
  fg : ℝ → ℝ
  gh : ℝ → ℝ
  he : ℝ
  is_parallelogram : ef = gh 0 ∧ fg 0 = he

/-- The product of w and z in the given parallelogram is 24 -/
theorem product_w_z_is_24 (p : Parallelogram)
    (h_ef : p.ef = 42)
    (h_fg : p.fg = fun z => 4 * z^3)
    (h_gh : p.gh = fun w => 3 * w + 6)
    (h_he : p.he = 32) :
    ∃ w z, p.gh w = p.ef ∧ p.fg z = p.he ∧ w * z = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_w_z_is_24_l2852_285266


namespace NUMINAMATH_CALUDE_trigonometric_values_l2852_285211

theorem trigonometric_values (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x - Real.cos x = -17/13) ∧ 
  (4 * Real.sin x * Real.cos x - Real.cos x^2 = -384/169) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_values_l2852_285211


namespace NUMINAMATH_CALUDE_other_bill_value_l2852_285255

theorem other_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) :
  total_bills = 12 →
  total_value = 100 →
  five_dollar_bills = 4 →
  ∃ other_value : ℕ, 
    other_value * (total_bills - five_dollar_bills) + 5 * five_dollar_bills = total_value ∧
    other_value = 10 :=
by sorry

end NUMINAMATH_CALUDE_other_bill_value_l2852_285255


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l2852_285275

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 - 4*x + 2*m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem statement
theorem two_distinct_roots_condition (m : ℝ) :
  has_two_distinct_real_roots m ↔ m < 2 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l2852_285275


namespace NUMINAMATH_CALUDE_triangle_theorem_l2852_285265

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.a = Real.sqrt 3) : 
  (t.A = π/3) ∧ (Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2852_285265


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2852_285258

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2852_285258


namespace NUMINAMATH_CALUDE_boys_percentage_in_class_l2852_285210

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * 100) = 42857 / 100000 :=
by sorry

end NUMINAMATH_CALUDE_boys_percentage_in_class_l2852_285210


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l2852_285293

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + breadth * height + length * breadth)

/-- Theorem: The surface area of a cuboid with length 4 cm, breadth 6 cm, and height 5 cm is 148 cm². -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 6 5 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l2852_285293


namespace NUMINAMATH_CALUDE_sum_of_digits_l2852_285231

theorem sum_of_digits (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  b + c = 10 →
  c + d = 1 →
  a + d = 2 →
  a + b + c + d = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2852_285231


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2852_285224

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2852_285224


namespace NUMINAMATH_CALUDE_car_speed_proof_l2852_285243

/-- Proves that a car traveling for two hours with an average speed of 60 km/h
    and a speed of 30 km/h in the second hour must have a speed of 90 km/h in the first hour. -/
theorem car_speed_proof (x : ℝ) :
  (x + 30) / 2 = 60 →
  x = 90 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2852_285243


namespace NUMINAMATH_CALUDE_binomial_60_3_l2852_285227

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2852_285227


namespace NUMINAMATH_CALUDE_molecular_weight_BaBr2_is_297_l2852_285218

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 297

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaBr2 in grams -/
def total_weight : ℝ := 2376

/-- Theorem: The molecular weight of BaBr2 is 297 grams/mole -/
theorem molecular_weight_BaBr2_is_297 :
  molecular_weight_BaBr2 = total_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_BaBr2_is_297_l2852_285218


namespace NUMINAMATH_CALUDE_same_color_probability_l2852_285208

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (pink : Nat)
  (green : Nat)
  (blue : Nat)
  (total : Nat)
  (h_total : pink + green + blue = total)

/-- The probability of two dice showing the same color -/
def samColorProbability (d : ColoredDie) : Rat :=
  (d.pink^2 + d.green^2 + d.blue^2) / d.total^2

/-- Two 12-sided dice with 3 pink, 4 green, and 5 blue sides each -/
def twelveSidedDie : ColoredDie :=
  { pink := 3
  , green := 4
  , blue := 5
  , total := 12
  , h_total := by rfl }

theorem same_color_probability :
  samColorProbability twelveSidedDie = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2852_285208


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l2852_285278

theorem cubic_sum_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 + a*b*c + b*c*d + c*d*a + d*a*b ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l2852_285278


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_condition_l2852_285223

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ q = a / b ∧ ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

theorem finite_decimal_fraction_condition (n : ℕ) :
  n > 0 → (is_finite_decimal (1 / (n * (n + 1))) ↔ n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_condition_l2852_285223


namespace NUMINAMATH_CALUDE_book_price_increase_l2852_285230

theorem book_price_increase (initial_price decreased_price final_price : ℝ) 
  (h1 : initial_price = 400)
  (h2 : decreased_price = initial_price * (1 - 0.15))
  (h3 : final_price = 476) :
  (final_price - decreased_price) / decreased_price = 0.4 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2852_285230


namespace NUMINAMATH_CALUDE_tournament_team_b_matches_l2852_285249

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  type : Bool -- false for A, true for B

/-- The tournament setup -/
structure Tournament where
  teams : Fin 32 → Team
  matches_played : Fin 32 → Nat
  different_matches : ∀ i j, i ≠ j → matches_played i ≠ matches_played j ∨ (teams i).city = 0 ∧ (teams i).type = false
  no_self_city_matches : ∀ i j, (teams i).city = (teams j).city → matches_played i + matches_played j ≤ 30
  max_one_match : ∀ i, matches_played i ≤ 30

theorem tournament_team_b_matches (t : Tournament) : 
  ∃ i, (t.teams i).city = 0 ∧ (t.teams i).type = true ∧ t.matches_played i = 15 :=
sorry

end NUMINAMATH_CALUDE_tournament_team_b_matches_l2852_285249


namespace NUMINAMATH_CALUDE_base9_246_to_base10_l2852_285251

/-- Converts a three-digit number from base 9 to base 10 -/
def base9ToBase10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 9^2 + d1 * 9^1 + d0 * 9^0

/-- The base 10 representation of 246 in base 9 is 204 -/
theorem base9_246_to_base10 : base9ToBase10 2 4 6 = 204 := by
  sorry

end NUMINAMATH_CALUDE_base9_246_to_base10_l2852_285251


namespace NUMINAMATH_CALUDE_max_min_product_l2852_285248

theorem max_min_product (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b + c = 12 →
  a * b + b * c + c * a = 20 →
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧
             m ≤ 12 ∧
             ∀ (k : ℝ), (∃ (x y z : ℝ), 
               0 < x ∧ 0 < y ∧ 0 < z ∧
               x + y + z = 12 ∧
               x * y + y * z + z * x = 20 ∧
               k = min (x * y) (min (y * z) (z * x))) →
             k ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l2852_285248


namespace NUMINAMATH_CALUDE_power_inequality_l2852_285247

theorem power_inequality (a b t x : ℝ) (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a^x = a + t) : b^x > b + t := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2852_285247


namespace NUMINAMATH_CALUDE_joan_grilled_cheese_sandwiches_l2852_285220

/-- Represents the number of cheese slices required for one ham sandwich. -/
def ham_cheese_slices : ℕ := 2

/-- Represents the number of cheese slices required for one grilled cheese sandwich. -/
def grilled_cheese_slices : ℕ := 3

/-- Represents the total number of cheese slices Joan uses. -/
def total_cheese_slices : ℕ := 50

/-- Represents the number of ham sandwiches Joan makes. -/
def ham_sandwiches : ℕ := 10

/-- Proves that Joan makes 10 grilled cheese sandwiches. -/
theorem joan_grilled_cheese_sandwiches : 
  (total_cheese_slices - ham_cheese_slices * ham_sandwiches) / grilled_cheese_slices = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_grilled_cheese_sandwiches_l2852_285220


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2852_285283

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2852_285283


namespace NUMINAMATH_CALUDE_pure_imaginary_quadratic_l2852_285270

theorem pure_imaginary_quadratic (a : ℝ) : 
  (Complex.mk (a^2 - 4*a + 3) (a - 1)).im ≠ 0 ∧ (Complex.mk (a^2 - 4*a + 3) (a - 1)).re = 0 → 
  a = 1 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_quadratic_l2852_285270


namespace NUMINAMATH_CALUDE_probability_red_ball_l2852_285271

/-- The probability of drawing a red ball from a box with red and black balls -/
theorem probability_red_ball (red_balls black_balls : ℕ) :
  red_balls > 0 →
  black_balls ≥ 0 →
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = 7 / 10 ↔
  red_balls = 7 ∧ black_balls = 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_red_ball_l2852_285271


namespace NUMINAMATH_CALUDE_hall_length_is_36_meters_l2852_285264

-- Define the hall dimensions
def hall_width : ℝ := 15

-- Define the stone dimensions in meters
def stone_length : ℝ := 0.6  -- 6 dm = 0.6 m
def stone_width : ℝ := 0.5   -- 5 dm = 0.5 m

-- Define the number of stones
def num_stones : ℕ := 1800

-- Theorem stating the length of the hall
theorem hall_length_is_36_meters :
  let total_area := (↑num_stones : ℝ) * stone_length * stone_width
  let hall_length := total_area / hall_width
  hall_length = 36 := by sorry

end NUMINAMATH_CALUDE_hall_length_is_36_meters_l2852_285264


namespace NUMINAMATH_CALUDE_range_of_sin4_plus_cos4_l2852_285214

theorem range_of_sin4_plus_cos4 :
  ∀ x : ℝ, (1/2 : ℝ) ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin4_plus_cos4_l2852_285214


namespace NUMINAMATH_CALUDE_function_composition_equality_l2852_285284

theorem function_composition_equality (A B : ℝ) (h : B ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^3
  let g : ℝ → ℝ := λ x ↦ 2 * B * x + B^2
  f (g 2) = 0 → A = 3 / (16 / B + 8 * B + B^3) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2852_285284


namespace NUMINAMATH_CALUDE_negative_a_cubed_div_negative_a_squared_l2852_285288

theorem negative_a_cubed_div_negative_a_squared (a : ℝ) : (-a)^3 / (-a)^2 = -a := by
  sorry

end NUMINAMATH_CALUDE_negative_a_cubed_div_negative_a_squared_l2852_285288


namespace NUMINAMATH_CALUDE_g_f_neg_3_l2852_285273

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 7

-- Define g(f(3)) = 15 as a hypothesis
axiom g_f_3 : ∃ g : ℝ → ℝ, g (f 3) = 15

-- Theorem to prove
theorem g_f_neg_3 : ∃ g : ℝ → ℝ, g (f (-3)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_f_neg_3_l2852_285273


namespace NUMINAMATH_CALUDE_friend_score_l2852_285206

theorem friend_score (edward_score : ℕ) (total_score : ℕ) (friend_score : ℕ) : 
  edward_score = 7 → 
  total_score = 13 → 
  total_score = edward_score + friend_score →
  friend_score = 6 := by
sorry

end NUMINAMATH_CALUDE_friend_score_l2852_285206


namespace NUMINAMATH_CALUDE_triangle_inequality_l2852_285263

theorem triangle_inequality (A B C : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : C = π - A - B) 
  (h : 1 / Real.sin A + 2 / Real.sin B = 3 * (1 / Real.tan A + 1 / Real.tan B)) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2852_285263


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2852_285287

theorem right_triangle_perimeter (a b : ℝ) (h_right : a^2 + b^2 = 5^2) 
  (h_area : (1/2) * a * b = 5) : a + b + 5 = 5 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2852_285287


namespace NUMINAMATH_CALUDE_proportionality_problem_l2852_285237

/-- Given that x is directly proportional to y² and inversely proportional to z,
    prove that x = 24 when y = 2 and z = 3, given that x = 6 when y = 1 and z = 3. -/
theorem proportionality_problem (k : ℝ) :
  (∀ y z, ∃ x, x = k * y^2 / z) →
  (6 = k * 1^2 / 3) →
  (∃ x, x = k * 2^2 / 3 ∧ x = 24) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_problem_l2852_285237


namespace NUMINAMATH_CALUDE_largest_n_with_prime_differences_l2852_285244

theorem largest_n_with_prime_differences : ∃ n : ℕ, 
  (n = 10) ∧ 
  (∀ m : ℕ, m > 10 → 
    ∃ p : ℕ, Prime p ∧ 2 < p ∧ p < m ∧ ¬(Prime (m - p))) ∧
  (∀ p : ℕ, Prime p → 2 < p → p < 10 → Prime (10 - p)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_prime_differences_l2852_285244


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2852_285279

/-- Given that x = 1 is a root of the quadratic equation x^2 + bx - 2 = 0,
    prove that the other root is -2 -/
theorem other_root_of_quadratic (b : ℝ) : 
  (1^2 + b*1 - 2 = 0) → ∃ x : ℝ, x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2852_285279


namespace NUMINAMATH_CALUDE_parabola_fixed_y_coordinate_l2852_285232

/-- A parabola that intersects the x-axis at only one point and passes through two specific points has a fixed y-coordinate for those points. -/
theorem parabola_fixed_y_coordinate (b c m n : ℝ) : 
  (∃ x, x^2 + b*x + c = 0 ∧ ∀ y, y ≠ x → y^2 + b*y + c ≠ 0) →  -- Parabola intersects x-axis at only one point
  (m^2 + b*m + c = n) →                                       -- Point (m, n) is on the parabola
  ((m-8)^2 + b*(m-8) + c = n) →                               -- Point (m-8, n) is on the parabola
  n = 16 := by
sorry

end NUMINAMATH_CALUDE_parabola_fixed_y_coordinate_l2852_285232


namespace NUMINAMATH_CALUDE_xiao_ming_run_distance_l2852_285253

/-- The distance between two adjacent trees in meters -/
def tree_spacing : ℕ := 6

/-- The number of the last tree Xiao Ming runs to -/
def last_tree : ℕ := 200

/-- The total distance Xiao Ming runs in meters -/
def total_distance : ℕ := (last_tree - 1) * tree_spacing

theorem xiao_ming_run_distance :
  total_distance = 1194 :=
sorry

end NUMINAMATH_CALUDE_xiao_ming_run_distance_l2852_285253


namespace NUMINAMATH_CALUDE_books_gotten_rid_of_l2852_285289

def initial_stock : ℕ := 27
def shelves_used : ℕ := 3
def books_per_shelf : ℕ := 7

theorem books_gotten_rid_of : 
  initial_stock - (shelves_used * books_per_shelf) = 6 := by
sorry

end NUMINAMATH_CALUDE_books_gotten_rid_of_l2852_285289


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2852_285202

/-- A right triangle with perimeter 60 and altitude to hypotenuse 12 has sides 15, 20, and 25. -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  h = 12 →
  a^2 + b^2 = c^2 →
  a * b = 2 * h * c →
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2852_285202
