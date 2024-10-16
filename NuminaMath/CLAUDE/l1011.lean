import Mathlib

namespace NUMINAMATH_CALUDE_exponential_grows_faster_than_quadratic_l1011_101181

theorem exponential_grows_faster_than_quadratic : 
  ∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, (2:ℝ)^x > x^2 + ε := by
  sorry

end NUMINAMATH_CALUDE_exponential_grows_faster_than_quadratic_l1011_101181


namespace NUMINAMATH_CALUDE_total_production_proof_l1011_101183

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300
def increase_percentage : ℚ := 20 / 100

theorem total_production_proof :
  let average := (week1_production + week2_production + week3_production) / 3
  let week4_production := average + (average * increase_percentage).floor
  week1_production + week2_production + week3_production + week4_production = 1428 := by
  sorry

end NUMINAMATH_CALUDE_total_production_proof_l1011_101183


namespace NUMINAMATH_CALUDE_equation_solution_range_l1011_101169

-- Define the set of real numbers greater than 0 and not equal to 1
def A : Set ℝ := {a | a > 0 ∧ a ≠ 1}

-- Define the function representing the equation
def f (a : ℝ) (k : ℝ) (x : ℝ) : Prop :=
  Real.log (x - a * k) / Real.log (Real.sqrt a) = Real.log (x^2 - a^2) / Real.log a

-- Define the set of k values that satisfy the equation for some x
def K (a : ℝ) : Set ℝ := {k | ∃ x, f a k x}

-- Theorem statement
theorem equation_solution_range (a : A) :
  K a = {k | k < -1 ∨ (k > 0 ∧ k < 1)} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1011_101169


namespace NUMINAMATH_CALUDE_investment_growth_l1011_101192

/-- Calculates the final amount of an investment after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $1500 invested at 3% interest for 21 years results in approximately $2709.17 -/
theorem investment_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.03
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 2709.17| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l1011_101192


namespace NUMINAMATH_CALUDE_students_on_sports_teams_l1011_101139

theorem students_on_sports_teams 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (both_activities : ℕ) 
  (either_activity : ℕ) 
  (h1 : total_students = 320)
  (h2 : band_students = 85)
  (h3 : both_activities = 60)
  (h4 : either_activity = 225)
  (h5 : either_activity = band_students + sports_students - both_activities) :
  sports_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_students_on_sports_teams_l1011_101139


namespace NUMINAMATH_CALUDE_pablo_puzzle_speed_l1011_101197

/-- Represents the number of pieces Pablo can put together per hour -/
def pieces_per_hour : ℕ := sorry

/-- The number of puzzles with 300 pieces -/
def puzzles_300 : ℕ := 8

/-- The number of puzzles with 500 pieces -/
def puzzles_500 : ℕ := 5

/-- The number of pieces in a 300-piece puzzle -/
def pieces_300 : ℕ := 300

/-- The number of pieces in a 500-piece puzzle -/
def pieces_500 : ℕ := 500

/-- The maximum number of hours Pablo works each day -/
def hours_per_day : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def days_to_complete : ℕ := 7

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := puzzles_300 * pieces_300 + puzzles_500 * pieces_500

/-- The total number of hours Pablo spends on puzzles -/
def total_hours : ℕ := hours_per_day * days_to_complete

theorem pablo_puzzle_speed : pieces_per_hour = 100 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_speed_l1011_101197


namespace NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_6_l1011_101113

/-- Proves that the cost of a child's ticket is 6 dollars given the specified conditions -/
theorem child_ticket_cost (adult_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_attending : ℕ) : ℕ :=
  let child_ticket_cost := (total_revenue - adult_ticket_cost * (total_tickets - children_attending)) / children_attending
  have h1 : adult_ticket_cost = 9 := by sorry
  have h2 : total_tickets = 225 := by sorry
  have h3 : total_revenue = 1875 := by sorry
  have h4 : children_attending = 50 := by sorry
  have h5 : child_ticket_cost * children_attending + adult_ticket_cost * (total_tickets - children_attending) = total_revenue := by sorry
  6

theorem child_ticket_cost_is_6 : child_ticket_cost 9 225 1875 50 = 6 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_child_ticket_cost_is_6_l1011_101113


namespace NUMINAMATH_CALUDE_tangent_fifteen_degrees_ratio_l1011_101136

theorem tangent_fifteen_degrees_ratio (π : Real) :
  let tan15 := Real.tan (15 * π / 180)
  (1 + tan15) / (1 - tan15) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_fifteen_degrees_ratio_l1011_101136


namespace NUMINAMATH_CALUDE_two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l1011_101118

/-- The probability that two people randomly checking into two rooms will each occupy one room -/
theorem two_people_two_rooms_probability : ℝ :=
  1/2

/-- Proof that the probability of two people randomly checking into two rooms 
    and each occupying one room is 1/2 -/
theorem two_people_two_rooms_probability_proof : 
  two_people_two_rooms_probability = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l1011_101118


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1011_101106

theorem complex_equation_solution (i z : ℂ) (hi : i^2 = -1) (hz : i * z = 2 * z + 1) :
  z = -2/5 - 1/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1011_101106


namespace NUMINAMATH_CALUDE_system_solution_l1011_101119

theorem system_solution (a b : ℝ) :
  ∃ (x y : ℝ), 
    (x + y = a ∧ Real.tan x * Real.tan y = b) ∧
    ((b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2) ∨
     (b ≠ 1 → 
      ∃ (k : ℤ), 
        x = (a + 2 * k * Real.pi - Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2 ∧
        y = (a - 2 * k * Real.pi + Real.arccos ((1 + b) / (1 - b) * Real.cos a)) / 2)) ∧
    (b = 1 ∧ ∃ (k : ℤ), a = Real.pi / 2 + k * Real.pi → y = Real.pi / 2 + k * Real.pi - x) :=
by
  sorry


end NUMINAMATH_CALUDE_system_solution_l1011_101119


namespace NUMINAMATH_CALUDE_mary_money_left_l1011_101180

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 3 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

/-- Theorem stating that the amount of money Mary has left is 30 - 8p -/
theorem mary_money_left (p : ℝ) : money_left p = 30 - 8 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l1011_101180


namespace NUMINAMATH_CALUDE_trig_identity_l1011_101130

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1011_101130


namespace NUMINAMATH_CALUDE_min_sum_of_chord_lengths_l1011_101126

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - Focus.1)}

/-- Length of the chord formed by a line with slope k on the parabola -/
noncomputable def ChordLength (k : ℝ) : ℝ :=
  4 + 4 / k^2

/-- Sum of chord lengths for two lines with slopes k₁ and k₂ -/
noncomputable def SumOfChordLengths (k₁ k₂ : ℝ) : ℝ :=
  ChordLength k₁ + ChordLength k₂

/-- Theorem stating the minimum value of the sum of chord lengths -/
theorem min_sum_of_chord_lengths :
  ∀ k₁ k₂ : ℝ, k₁^2 + k₂^2 = 1 →
  24 ≤ SumOfChordLengths k₁ k₂ ∧
  (∃ k₁' k₂' : ℝ, k₁'^2 + k₂'^2 = 1 ∧ SumOfChordLengths k₁' k₂' = 24) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_chord_lengths_l1011_101126


namespace NUMINAMATH_CALUDE_tower_blocks_l1011_101112

theorem tower_blocks (A R : ℕ) (h : A - R = 30) : 35 + A - R = 65 := by
  sorry

end NUMINAMATH_CALUDE_tower_blocks_l1011_101112


namespace NUMINAMATH_CALUDE_congruence_problem_l1011_101160

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 % N = 90 % N) (h3 : 90 % N = 125 % N) : 81 % N = 4 % N := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1011_101160


namespace NUMINAMATH_CALUDE_candy_seller_problem_l1011_101127

theorem candy_seller_problem (num_clowns num_children initial_candies candies_per_person : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : candies_per_person = 20) :
  initial_candies - (num_clowns + num_children) * candies_per_person = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_seller_problem_l1011_101127


namespace NUMINAMATH_CALUDE_divisibility_by_thirty_l1011_101157

theorem divisibility_by_thirty (p : ℕ) (h_prime : Nat.Prime p) (h_ge_seven : p ≥ 7) :
  ∃ k : ℕ, p^2 - 1 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirty_l1011_101157


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1011_101150

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (height_division : r + s = c) 
  (similarity_relations : a^2 = r * c ∧ b^2 = s * c) 
  (leg_ratio : a / b = 1 / 3) :
  r / s = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1011_101150


namespace NUMINAMATH_CALUDE_acme_horseshoe_problem_l1011_101194

/-- Acme's horseshoe manufacturing problem -/
theorem acme_horseshoe_problem (initial_outlay : ℝ) : 
  let cost_per_set : ℝ := 20.75
  let selling_price : ℝ := 50
  let num_sets : ℕ := 950
  let profit : ℝ := 15337.5
  let revenue : ℝ := selling_price * num_sets
  let total_cost : ℝ := initial_outlay + cost_per_set * num_sets
  profit = revenue - total_cost →
  initial_outlay = 12450 := by
  sorry

#check acme_horseshoe_problem

end NUMINAMATH_CALUDE_acme_horseshoe_problem_l1011_101194


namespace NUMINAMATH_CALUDE_game_sale_problem_l1011_101141

theorem game_sale_problem (initial_games : ℕ) (sold_games : ℕ) 
  (sold_at_15 : ℕ) (sold_at_10 : ℕ) (sold_at_8 : ℕ) (games_per_box : ℕ) :
  initial_games = 76 →
  sold_games = 46 →
  sold_at_15 = 20 →
  sold_at_10 = 15 →
  sold_at_8 = 11 →
  games_per_box = 5 →
  sold_games = sold_at_15 + sold_at_10 + sold_at_8 →
  (initial_games - sold_games) % games_per_box = 0 →
  ((initial_games - sold_games) / games_per_box = 6 ∧ 
   sold_at_15 * 15 + sold_at_10 * 10 + sold_at_8 * 8 = 538) := by
  sorry


end NUMINAMATH_CALUDE_game_sale_problem_l1011_101141


namespace NUMINAMATH_CALUDE_circle_ratio_l1011_101107

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l1011_101107


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l1011_101148

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 65 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l1011_101148


namespace NUMINAMATH_CALUDE_correct_negation_l1011_101173

theorem correct_negation :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_correct_negation_l1011_101173


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l1011_101161

def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_seq_problem (a : ℕ → ℚ) :
  is_arithmetic_seq a →
  a 4 + a 5 + a 6 + a 7 = 56 →
  a 4 * a 7 = 187 →
  ((∃ a₁ d, ∀ n, a n = a₁ + (n - 1) * d) ∧ 
   ((a 1 = 5 ∧ ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 2) ∨
    (a 1 = 23 ∧ ∃ d, ∀ n, a n = 23 + (n - 1) * d ∧ d = -2))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_problem_l1011_101161


namespace NUMINAMATH_CALUDE_picture_book_shelves_l1011_101133

theorem picture_book_shelves 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (h1 : total_books = 72)
  (h2 : books_per_shelf = 9)
  (h3 : mystery_shelves = 3) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 5 := by
sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l1011_101133


namespace NUMINAMATH_CALUDE_jack_baseball_cards_l1011_101100

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
  total_cards = 125 →
  baseball_cards = 3 * football_cards + 5 →
  total_cards = baseball_cards + football_cards →
  baseball_cards = 95 := by
sorry

end NUMINAMATH_CALUDE_jack_baseball_cards_l1011_101100


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l1011_101193

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_count : ℕ := 4

theorem dice_probability_theorem :
  (Nat.choose num_dice target_count * 5^(num_dice - target_count)) / 6^num_dice =
  (1365 * 5^11) / 6^15 := by sorry

end NUMINAMATH_CALUDE_dice_probability_theorem_l1011_101193


namespace NUMINAMATH_CALUDE_ping_pong_tournament_participants_ping_pong_tournament_solution_l1011_101108

/-- Represents a ping-pong tournament with elimination rules -/
structure PingPongTournament where
  participants : ℕ
  games_played : ℕ
  remaining_players : ℕ

/-- Conditions for our specific tournament -/
def tournament_conditions (t : PingPongTournament) : Prop :=
  t.games_played = 29 ∧ t.remaining_players = 2

/-- Theorem stating the number of participants in the tournament -/
theorem ping_pong_tournament_participants 
  (t : PingPongTournament) 
  (h : tournament_conditions t) : 
  t.participants = 16 := by
  sorry

/-- Main theorem combining the structure and the proof -/
theorem ping_pong_tournament_solution : 
  ∃ t : PingPongTournament, tournament_conditions t ∧ t.participants = 16 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_tournament_participants_ping_pong_tournament_solution_l1011_101108


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1011_101102

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
theorem ellipse_major_axis_length :
  let f1 : ℝ × ℝ := (15, 10)
  let f2 : ℝ × ℝ := (35, 40)
  let ellipse := {p : ℝ × ℝ | ∃ k, dist p f1 + dist p f2 = k}
  let tangent_to_y_axis := ∃ y, (0, y) ∈ ellipse
  let major_axis_length := Real.sqrt 3400
  tangent_to_y_axis →
  ∃ a b : ℝ × ℝ, a ∈ ellipse ∧ b ∈ ellipse ∧ dist a b = major_axis_length :=
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1011_101102


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1011_101129

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def is_on_segment (P M N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N

def vector_eq (P M N : ℝ × ℝ) : Prop :=
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2))

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, is_on_segment P M N ∧ vector_eq P M N → P = (2, 4) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1011_101129


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1011_101195

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -2) :
  (∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 2 ∧ 
   x₀ ≥ -1 ∧ y₀ ≥ -3/2 ∧ z₀ ≥ -2 ∧
   Real.sqrt (5 * x₀ + 5) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (6 * z₀ + 10) = Real.sqrt 93) ∧
  (∀ (a b c : ℝ), a + b + c = 2 → 
   a ≥ -1 → b ≥ -3/2 → c ≥ -2 →
   Real.sqrt (5 * a + 5) + Real.sqrt (4 * b + 6) + Real.sqrt (6 * c + 10) ≤ Real.sqrt 93) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1011_101195


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_sevenths_l1011_101103

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MovingObjects where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B

/-- The conditions of the problem -/
def satisfies_conditions (obj : MovingObjects) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    (obj.vA * t₁)^2 = (750 - obj.vB * t₁)^2 ∧
    (obj.vA * t₂)^2 = (750 - obj.vB * t₂)^2 ∧
    t₂ - t₁ = 6 ∧
    t₁ = 3

/-- The theorem statement -/
theorem speed_ratio_is_two_sevenths (obj : MovingObjects) 
  (h : satisfies_conditions obj) : 
  obj.vA / obj.vB = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_sevenths_l1011_101103


namespace NUMINAMATH_CALUDE_magnitude_relationship_l1011_101131

theorem magnitude_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b ∧ a * b < a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l1011_101131


namespace NUMINAMATH_CALUDE_circle_center_l1011_101124

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y - 15 = 0

-- Theorem stating that the center of the circle is (3, 1)
theorem circle_center :
  ∃ (x y : ℝ), circle_equation x y ∧ (x, y) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1011_101124


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1011_101116

/-- Proves that if a salary is first increased by 10% and then decreased by 5%, 
    resulting in Rs. 4180, then the original salary must be Rs. 4000. -/
theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 4180) → x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1011_101116


namespace NUMINAMATH_CALUDE_solve_system_l1011_101154

theorem solve_system (x y : ℚ) (h1 : x/2 - 2*y = 2) (h2 : x/2 + 2*y = 12) : y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1011_101154


namespace NUMINAMATH_CALUDE_extremum_and_nonnegative_conditions_l1011_101191

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) - Real.log x

theorem extremum_and_nonnegative_conditions (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ∈ Set.Ioo (1 - ε) (1 + ε) → f a x ≥ f a 1) →
  a = 1 ∧
  (∀ (x : ℝ), x ≥ 1 → f a x ≥ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_nonnegative_conditions_l1011_101191


namespace NUMINAMATH_CALUDE_city_birth_rate_l1011_101142

/-- Represents the birth rate problem in a city --/
theorem city_birth_rate 
  (death_rate : ℕ) 
  (net_increase : ℕ) 
  (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 129600)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 6 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
by sorry

end NUMINAMATH_CALUDE_city_birth_rate_l1011_101142


namespace NUMINAMATH_CALUDE_janinas_pancakes_l1011_101166

/-- Calculates the number of pancakes Janina must sell daily to cover her expenses -/
theorem janinas_pancakes (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) :
  daily_rent = 30 →
  daily_supplies = 12 →
  price_per_pancake = 2 →
  (daily_rent + daily_supplies) / price_per_pancake = 21 := by
  sorry

#check janinas_pancakes

end NUMINAMATH_CALUDE_janinas_pancakes_l1011_101166


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1011_101137

theorem unique_solution_for_equation (b : ℝ) (hb : b ≠ 0) :
  ∃! a : ℝ, a ≠ 0 ∧ (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1011_101137


namespace NUMINAMATH_CALUDE_imaginary_part_of_4_minus_3i_l1011_101135

theorem imaginary_part_of_4_minus_3i :
  Complex.im (4 - 3 * Complex.I) = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_4_minus_3i_l1011_101135


namespace NUMINAMATH_CALUDE_total_tax_percentage_l1011_101176

/-- Calculate the total tax percentage given spending percentages, discounts, and tax rates --/
theorem total_tax_percentage
  (total_amount : ℝ)
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (electronics_percent : ℝ)
  (other_percent : ℝ)
  (clothing_discount : ℝ)
  (electronics_discount : ℝ)
  (clothing_tax : ℝ)
  (food_tax : ℝ)
  (electronics_tax : ℝ)
  (other_tax : ℝ)
  (h1 : clothing_percent = 0.4)
  (h2 : food_percent = 0.15)
  (h3 : electronics_percent = 0.25)
  (h4 : other_percent = 0.2)
  (h5 : clothing_discount = 0.1)
  (h6 : electronics_discount = 0.05)
  (h7 : clothing_tax = 0.04)
  (h8 : food_tax = 0)
  (h9 : electronics_tax = 0.06)
  (h10 : other_tax = 0.08)
  (h11 : total_amount > 0) :
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let electronics_amount := electronics_percent * total_amount
  let other_amount := other_percent * total_amount
  let discounted_clothing := clothing_amount * (1 - clothing_discount)
  let discounted_electronics := electronics_amount * (1 - electronics_discount)
  let total_tax := clothing_tax * discounted_clothing +
                   food_tax * food_amount +
                   electronics_tax * discounted_electronics +
                   other_tax * other_amount
  ∃ ε > 0, |total_tax / total_amount - 0.04465| < ε :=
by sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l1011_101176


namespace NUMINAMATH_CALUDE_playful_not_brown_l1011_101162

structure Dog where
  playful : Prop
  brown : Prop
  knowsTricks : Prop
  canSwim : Prop

axiom all_playful_know_tricks : ∀ (d : Dog), d.playful → d.knowsTricks
axiom no_brown_can_swim : ∀ (d : Dog), d.brown → ¬d.canSwim
axiom cant_swim_dont_know_tricks : ∀ (d : Dog), ¬d.canSwim → ¬d.knowsTricks

theorem playful_not_brown : ∀ (d : Dog), d.playful → ¬d.brown := by
  sorry

end NUMINAMATH_CALUDE_playful_not_brown_l1011_101162


namespace NUMINAMATH_CALUDE_randy_initial_amount_l1011_101155

/-- Proves that Randy's initial amount is $6166.67 given the problem conditions --/
theorem randy_initial_amount :
  ∀ (initial : ℝ),
  (3/4 : ℝ) * (initial + 2900) - 1300 = 5500 →
  initial = 6166.67 := by
sorry

end NUMINAMATH_CALUDE_randy_initial_amount_l1011_101155


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1011_101132

/-- Given a book with 400 pages, prove that if a person has read 20% of it,
    they need to read 320 pages to finish the book. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) :
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1011_101132


namespace NUMINAMATH_CALUDE_prime_conditions_theorem_l1011_101143

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (A : ℕ) : Prop :=
  is_prime A ∧
  A < 100 ∧
  is_prime (A + 10) ∧
  is_prime (A - 20) ∧
  is_prime (A + 30) ∧
  is_prime (A + 60) ∧
  is_prime (A + 70)

theorem prime_conditions_theorem :
  ∀ A : ℕ, satisfies_conditions A ↔ (A = 37 ∨ A = 43 ∨ A = 79) :=
by sorry

end NUMINAMATH_CALUDE_prime_conditions_theorem_l1011_101143


namespace NUMINAMATH_CALUDE_fraction_equality_l1011_101179

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (12/5) / (2 * q + p) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1011_101179


namespace NUMINAMATH_CALUDE_number_equation_l1011_101175

theorem number_equation (x : ℤ) : x - (28 - (37 - (15 - 19))) = 58 ↔ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1011_101175


namespace NUMINAMATH_CALUDE_wage_difference_l1011_101189

/-- Proves the wage difference between a manager and a chef given specific wage relationships -/
theorem wage_difference (manager_wage : ℝ) 
  (h1 : manager_wage = 6.50)
  (h2 : ∃ dishwasher_wage : ℝ, dishwasher_wage = manager_wage / 2)
  (h3 : ∃ chef_wage : ℝ, chef_wage = (manager_wage / 2) * 1.20) :
  manager_wage - ((manager_wage / 2) * 1.20) = 2.60 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l1011_101189


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l1011_101117

/-- The standard form of a hyperbola with foci on the x-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The relationship between a, b, and c in a hyperbola -/
def hyperbola_relation (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem hyperbola_standard_form :
  ∀ (a b : ℝ),
    a > 0 →
    b > 0 →
    hyperbola_relation a b (Real.sqrt 6) →
    hyperbola_equation a b (-5) 2 →
    hyperbola_equation (Real.sqrt 5) 1 = hyperbola_equation a b :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l1011_101117


namespace NUMINAMATH_CALUDE_cake_distribution_l1011_101163

theorem cake_distribution (total_pieces : ℕ) (num_friends : ℕ) (pieces_per_friend : ℕ) :
  total_pieces = 150 →
  num_friends = 50 →
  pieces_per_friend * num_friends = total_pieces →
  pieces_per_friend = 3 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l1011_101163


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1011_101156

/-- Represents the age of a person -/
structure Age :=
  (years : ℕ)

/-- Represents the ratio between two numbers -/
structure Ratio :=
  (numerator : ℕ)
  (denominator : ℕ)

/-- Given two people p and q, their ages 6 years ago, and their current total age,
    proves that the ratio of their current ages is 3:4 -/
theorem age_ratio_proof 
  (p q : Age) 
  (h1 : p.years + 6 = (q.years + 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : (p.years + 6) + (q.years + 6) = 21)  -- The total of their present ages is 21
  : Ratio.mk 3 4 = Ratio.mk (p.years + 6) (q.years + 6) :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_proof_l1011_101156


namespace NUMINAMATH_CALUDE_strawberry_harvest_l1011_101158

/-- Calculates the expected strawberry harvest for a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft avg_yield : ℕ) : 
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  avg_yield = 10 →
  length * width * plants_per_sqft * avg_yield = 6000 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l1011_101158


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1011_101128

noncomputable def z : ℂ := (Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))

theorem imaginary_part_of_z : Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1011_101128


namespace NUMINAMATH_CALUDE_no_roots_below_x0_l1011_101164

theorem no_roots_below_x0 (a b c d x₀ : ℝ) 
  (h1 : ∀ x ≥ x₀, x^2 + a*x + b > 0)
  (h2 : ∀ x ≥ x₀, x^2 + c*x + d > 0) :
  ∀ x > x₀, x^2 + (a+c)/2 * x + (b+d)/2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_below_x0_l1011_101164


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1011_101185

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 →
  inheritance = 41379 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1011_101185


namespace NUMINAMATH_CALUDE_expression_equals_one_l1011_101168

theorem expression_equals_one : 
  (2009 * 2029 + 100) * (1999 * 2039 + 400) / (2019^4 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1011_101168


namespace NUMINAMATH_CALUDE_oh_squared_value_l1011_101109

/-- Given a triangle ABC with circumcenter O, orthocenter H, and circumradius R -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states that for a triangle with R = 10 and a^2 + b^2 + c^2 = 50, OH^2 = 850 -/
theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 10) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  (t.O.1 - t.H.1)^2 + (t.O.2 - t.H.2)^2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_value_l1011_101109


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slopes_l1011_101125

/-- Given a parabola y^2 = 2px and a line intersecting it at points A and B, 
    if the slope of OA is 2 and the slope of AB is 6, then the slope of OB is -3. -/
theorem parabola_line_intersection_slopes (p : ℝ) (y₁ y₂ : ℝ) : 
  let A := (y₁^2 / (2*p), y₁)
  let B := (y₂^2 / (2*p), y₂)
  let k_OA := y₁ / (y₁^2 / (2*p))
  let k_AB := (y₂ - y₁) / (y₂^2 / (2*p) - y₁^2 / (2*p))
  let k_OB := y₂ / (y₂^2 / (2*p))
  k_OA = 2 ∧ k_AB = 6 → k_OB = -3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slopes_l1011_101125


namespace NUMINAMATH_CALUDE_prob_one_one_ten_dice_l1011_101111

/-- The probability of rolling exactly one 1 out of 10 standard 6-sided dice -/
def prob_one_one (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of rolling exactly one 1 out of 10 standard 6-sided dice
    is equal to (10 * 5^9) / 6^10 -/
theorem prob_one_one_ten_dice :
  prob_one_one 10 1 (1/6) = (10 * 5^9) / 6^10 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_one_ten_dice_l1011_101111


namespace NUMINAMATH_CALUDE_maggie_candy_count_l1011_101171

/-- Given the Halloween candy collection scenario, prove that Maggie collected 50 pieces of candy. -/
theorem maggie_candy_count :
  -- Harper collected 30% more candy than Maggie
  ∀ (maggie harper : ℕ), harper = (13 * maggie) / 10 →
  -- Neil collected 40% more candy than Harper
  ∀ (neil : ℕ), neil = (14 * harper) / 10 →
  -- Neil got 91 pieces of candy
  neil = 91 →
  -- Maggie collected 50 pieces of candy
  maggie = 50 := by
sorry

end NUMINAMATH_CALUDE_maggie_candy_count_l1011_101171


namespace NUMINAMATH_CALUDE_f_even_l1011_101188

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem to be proved
theorem f_even : ∀ x, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_even_l1011_101188


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1011_101177

def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c : ℚ) / 999

theorem repeating_decimal_sum :
  repeating_decimal 2 3 4 - repeating_decimal 5 6 7 + repeating_decimal 8 9 1 = 186 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1011_101177


namespace NUMINAMATH_CALUDE_ninth_root_of_unity_sum_l1011_101121

theorem ninth_root_of_unity_sum (ω : ℂ) (h1 : ω ^ 9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ninth_root_of_unity_sum_l1011_101121


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1011_101186

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 24*x + 125

-- Define the lower and upper bounds of the solution interval
def lower_bound : ℝ := 6.71
def upper_bound : ℝ := 17.29

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≤ 9 ↔ lower_bound ≤ x ∧ x ≤ upper_bound := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1011_101186


namespace NUMINAMATH_CALUDE_cyclic_iff_concurrent_l1011_101146

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are cyclic -/
def are_cyclic (A B C D : Point) : Prop :=
  sorry

/-- Check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Get the line passing through two points -/
def line_through_points (A B : Point) : Line :=
  sorry

theorem cyclic_iff_concurrent (A B C D E F : Point) :
  are_cyclic A B C D → are_cyclic C D E F →
  (are_cyclic A B E F ↔ 
    are_concurrent 
      (line_through_points A B) 
      (line_through_points C D) 
      (line_through_points E F)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_iff_concurrent_l1011_101146


namespace NUMINAMATH_CALUDE_min_c_value_l1011_101110

/-- Given positive integers a, b, c satisfying a < b < c, and a system of equations
    with exactly one solution, prove that the minimum value of c is 2002. -/
theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hab : a < b) (hbc : b < c)
    (h_unique : ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 2002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 2002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a'| + |x - b'| + |x - 2002| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1011_101110


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1011_101120

theorem min_value_of_expression (x y : ℝ) :
  (x + y + x * y)^2 + (x - y - x * y)^2 ≥ 0 ∧
  ∃ a b : ℝ, (a + b + a * b)^2 + (a - b - a * b)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1011_101120


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_76_l1011_101174

/-- A rectangle with the given properties --/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 360
  new_area_eq : (length + 10) * (width - 6) = 360

/-- The perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem stating that a rectangle with the given properties has a perimeter of 76 --/
theorem rectangle_perimeter_is_76 (r : Rectangle) : perimeter r = 76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_76_l1011_101174


namespace NUMINAMATH_CALUDE_speed_to_arrive_on_time_l1011_101123

/-- The speed required to arrive on time given early and late arrival conditions -/
theorem speed_to_arrive_on_time (S : ℝ) (t : ℝ) (h1 : S = 90 * (t - 1)) (h2 : S = 60 * (t + 1)) :
  S / t = 72 := by
  sorry

end NUMINAMATH_CALUDE_speed_to_arrive_on_time_l1011_101123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1011_101134

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (h1 : a₁ = -33) (h2 : aₙ = 57) (h3 : d = 5) :
  (aₙ - a₁) / d + 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1011_101134


namespace NUMINAMATH_CALUDE_abs_f_properties_l1011_101172

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the absolute value function of f
def abs_f (x : ℝ) : ℝ := |f x|

-- Theorem stating the properties of |f(x)|
theorem abs_f_properties :
  (∀ x, abs_f f x ≥ 0) ∧ 
  (∀ x, f x ≥ 0 → abs_f f x = f x) ∧
  (∀ x, f x < 0 → abs_f f x = -f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_f_properties_l1011_101172


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1011_101140

theorem quadratic_inequality_solution (m n : ℝ) :
  (∀ x : ℝ, x^2 + m*x + n < 0 ↔ -1 < x ∧ x < 3) →
  m + n = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1011_101140


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1011_101114

/-- Given a cone with base radius 2 and lateral surface forming a semicircle,
    prove that its lateral surface area is 8π. -/
theorem cone_lateral_surface_area (r : ℝ) (h : r = 2) :
  let l := 2 * r  -- slant height is twice the base radius for a semicircle lateral surface
  π * r * l = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1011_101114


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1011_101145

theorem smallest_number_divisible (n : ℕ) : n = 84 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 5 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 10 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 15 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 20 * k)) ∧ 
  (∃ k1 k2 k3 k4 : ℕ, 
    n - 24 = 5 * k1 ∧ 
    n - 24 = 10 * k2 ∧ 
    n - 24 = 15 * k3 ∧ 
    n - 24 = 20 * k4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1011_101145


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1011_101159

theorem imaginary_part_of_z (z : ℂ) : z = (2 - Complex.I) * Complex.I → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1011_101159


namespace NUMINAMATH_CALUDE_xyz_values_l1011_101198

theorem xyz_values (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x * y = 30)
  (eq2 : x * z = 60)
  (eq3 : x + y + z = 27) :
  x = (27 + Real.sqrt 369) / 2 ∧
  y = 60 / ((27 + Real.sqrt 369) / 2) ∧
  z = 30 / ((27 + Real.sqrt 369) / 2) := by
sorry

end NUMINAMATH_CALUDE_xyz_values_l1011_101198


namespace NUMINAMATH_CALUDE_square_state_after_2010_transforms_l1011_101178

/-- Represents the four possible states of the square labeling -/
inductive SquareState
  | BADC
  | DCBA
  | ABCD
  | CDAB

/-- Applies one transformation (reflection then rotation) to the square -/
def transform (s : SquareState) : SquareState :=
  match s with
  | SquareState.BADC => SquareState.DCBA
  | SquareState.DCBA => SquareState.ABCD
  | SquareState.ABCD => SquareState.DCBA
  | SquareState.CDAB => SquareState.BADC

/-- Applies n transformations to the initial square state -/
def applyNTransforms (n : Nat) : SquareState :=
  match n with
  | 0 => SquareState.BADC
  | n + 1 => transform (applyNTransforms n)

theorem square_state_after_2010_transforms :
  applyNTransforms 2010 = SquareState.DCBA := by
  sorry

end NUMINAMATH_CALUDE_square_state_after_2010_transforms_l1011_101178


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1011_101115

/-- The speed of a boat in still water, given the speed of the current and the upstream speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 20) 
  (h2 : upstream_speed = 30) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 50 ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1011_101115


namespace NUMINAMATH_CALUDE_jerome_toy_cars_l1011_101184

/-- The number of toy cars Jerome has now -/
def total_cars (original : ℕ) (last_month : ℕ) (this_month : ℕ) : ℕ :=
  original + last_month + this_month

/-- Theorem: Jerome has 40 toy cars now -/
theorem jerome_toy_cars :
  let original := 25
  let last_month := 5
  let this_month := 2 * last_month
  total_cars original last_month this_month = 40 := by
sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l1011_101184


namespace NUMINAMATH_CALUDE_inequality_proof_l1011_101149

theorem inequality_proof (a b : ℝ) : (6*a - 3*b - 3) * (a^2 + a^2*b - 2*a^3) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1011_101149


namespace NUMINAMATH_CALUDE_gravel_pile_volume_l1011_101147

/-- The volume of a conical pile of gravel -/
theorem gravel_pile_volume (diameter : Real) (height_ratio : Real) : 
  diameter = 10 →
  height_ratio = 0.6 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_gravel_pile_volume_l1011_101147


namespace NUMINAMATH_CALUDE_jacks_initial_dollars_l1011_101170

theorem jacks_initial_dollars (x : ℕ) : 
  x + 36 * 2 = 117 → x = 45 := by sorry

end NUMINAMATH_CALUDE_jacks_initial_dollars_l1011_101170


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1011_101153

theorem sin_cos_sixth_power_sum (α : Real) (h : Real.cos (2 * α) = 1 / 5) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l1011_101153


namespace NUMINAMATH_CALUDE_polynomial_equality_l1011_101138

-- Define the polynomials P and Q
def P (x y z w : ℝ) : ℝ := x * y + x^2 - z + w
def Q (x y z w : ℝ) : ℝ := x + y

-- State the theorem
theorem polynomial_equality (x y z w : ℝ) :
  (x * y + z + w)^2 - (x^2 - 2*z)*(y^2 - 2*w) = 
  (P x y z w)^2 - (x^2 - 2*z)*(Q x y z w)^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1011_101138


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_area_l1011_101122

/-- Given a line passing through (a, 0) and intersecting the y-axis in the first quadrant,
    forming a triangular region with area T, prove that the equation of this line is
    2Tx + a²y - 2aT = 0 -/
theorem line_equation_from_triangle_area (a T : ℝ) (h_a : a ≠ 0) (h_T : T > 0) :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, y = m * x + b → (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0)) ∧
    (1/2 * a * b = T) ∧
    (∀ x y : ℝ, 2 * T * x + a^2 * y - 2 * a * T = 0 ↔ y = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_area_l1011_101122


namespace NUMINAMATH_CALUDE_product_range_check_l1011_101101

theorem product_range_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  (32 * 84 > 2400) ∧ 
  (63 * 54 > 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) := by
  sorry

end NUMINAMATH_CALUDE_product_range_check_l1011_101101


namespace NUMINAMATH_CALUDE_brownies_remaining_l1011_101144

/-- Calculates the number of brownies left after consumption --/
def brownies_left (total : Nat) (tina_daily : Nat) (husband_daily : Nat) (days : Nat) (shared : Nat) : Nat :=
  total - (tina_daily * days) - (husband_daily * days) - shared

/-- Proves that 5 brownies are left under given conditions --/
theorem brownies_remaining : brownies_left 24 2 1 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_remaining_l1011_101144


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1011_101152

/-- Quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_theorem
  (a b c : ℝ) (m n : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_m_ne_n : m ≠ n)
  (h_m_plus_n : m + n = 3)
  (h_b_neg : b < 0) :
  (a = -1 ∧ 
   quadratic a b c 1 = 4 ∧ 
   -b / (2 * a) = 2 → 
   ∀ x, quadratic a b c x = -x^2 + 4*x + 1) ∧
  (a = -1 ∧ 
   (∃! x, quadratic a b c x = 0) → 
   b + 4*c ≤ 1/4) ∧
  (quadratic a b c m = m ∧ 
   quadratic a b c n = n → 
   a > 1/3) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1011_101152


namespace NUMINAMATH_CALUDE_garden_border_material_l1011_101151

/-- The amount of material needed for a decorative border around a circular garden -/
theorem garden_border_material (garden_area : Real) (pi_estimate : Real) (extra_material : Real) : 
  garden_area = 50.24 → pi_estimate = 3.14 → extra_material = 4 →
  2 * pi_estimate * (garden_area / pi_estimate).sqrt + extra_material = 29.12 := by
sorry

end NUMINAMATH_CALUDE_garden_border_material_l1011_101151


namespace NUMINAMATH_CALUDE_no_equilateral_with_100_degree_angle_l1011_101187

-- Define what an equilateral triangle is
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c

-- Define the sum of angles in a triangle
axiom triangle_angle_sum (a b c : ℝ) : a + b + c = 180

-- Theorem: An equilateral triangle cannot have an angle of 100 degrees
theorem no_equilateral_with_100_degree_angle (a b c : ℝ) :
  is_equilateral a b c → ¬(a = 100 ∨ b = 100 ∨ c = 100) :=
by sorry

end NUMINAMATH_CALUDE_no_equilateral_with_100_degree_angle_l1011_101187


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_distances_l1011_101196

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Curve in polar form -/
structure PolarCurve where
  equation : ℝ → ℝ

/-- Function to calculate the minimum sum of reciprocal distances -/
noncomputable def minSumReciprocalDistances (l : Line) (c : PolarCurve) : ℝ :=
  sorry

/-- Theorem stating the minimum value of the sum of reciprocal distances -/
theorem min_sum_reciprocal_distances :
  let p := Point.mk 1 2
  let l := Line.mk p α
  let c := PolarCurve.mk (fun θ ↦ 6 * Real.sin θ)
  minSumReciprocalDistances l c = 2 * Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_distances_l1011_101196


namespace NUMINAMATH_CALUDE_gregs_dog_walking_rate_l1011_101190

/-- Greg's dog walking earnings problem -/
theorem gregs_dog_walking_rate :
  ∀ (rate : ℚ),
  (20 + 10 * rate) +   -- One dog for 10 minutes
  2 * (20 + 7 * rate) +  -- Two dogs for 7 minutes each
  3 * (20 + 9 * rate) = 171  -- Three dogs for 9 minutes each
  →
  rate = 1 := by
sorry

end NUMINAMATH_CALUDE_gregs_dog_walking_rate_l1011_101190


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1011_101104

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1011_101104


namespace NUMINAMATH_CALUDE_tangent_cubic_to_line_l1011_101182

/-- Given that the graph of y = ax³ + 1 is tangent to the line y = x, prove that a = 4/27 -/
theorem tangent_cubic_to_line (a : ℝ) : 
  (∃ x : ℝ, x = a * x^3 + 1 ∧ 3 * a * x^2 = 1) → a = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_tangent_cubic_to_line_l1011_101182


namespace NUMINAMATH_CALUDE_m_range_l1011_101199

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : B m ⊆ A ∧ B m ≠ A → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1011_101199


namespace NUMINAMATH_CALUDE_tan_value_from_log_equation_l1011_101167

theorem tan_value_from_log_equation (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/2))
  (h2 : Real.log (Real.sin (2*x)) - Real.log (Real.sin x) = Real.log (1/2)) :
  Real.tan x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_log_equation_l1011_101167


namespace NUMINAMATH_CALUDE_id_tag_problem_l1011_101105

/-- The set of characters available for creating ID tags -/
def tag_chars : Finset Char := {'M', 'A', 'T', 'H', '2', '0', '3'}

/-- The number of times '2' can appear in a tag -/
def max_twos : Nat := 2

/-- The length of each ID tag -/
def tag_length : Nat := 5

/-- The total number of unique ID tags -/
def total_tags : Nat := 3720

/-- Theorem stating the result of the ID tag problem -/
theorem id_tag_problem :
  (total_tags : ℚ) / 10 = 372 := by sorry

end NUMINAMATH_CALUDE_id_tag_problem_l1011_101105


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l1011_101165

/-- The area of a shaded region within a square, where two congruent right triangles
    are removed from opposite corners. -/
theorem shaded_area_in_square (side : ℝ) (triangle_side : ℝ)
    (h_side : side = 30)
    (h_triangle : triangle_side = 20) :
    side * side - 2 * (1/2 * triangle_side * triangle_side) = 500 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_l1011_101165
