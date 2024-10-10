import Mathlib

namespace square_area_200m_l3434_343430

/-- The area of a square with side length 200 meters is 40000 square meters. -/
theorem square_area_200m : 
  let side_length : ℝ := 200
  let area : ℝ := side_length * side_length
  area = 40000 := by sorry

end square_area_200m_l3434_343430


namespace pablo_puzzle_days_l3434_343474

def puzzles_400 : ℕ := 15
def pieces_per_400 : ℕ := 400
def puzzles_700 : ℕ := 10
def pieces_per_700 : ℕ := 700
def pieces_per_hour : ℕ := 100
def hours_per_day : ℕ := 6

def total_pieces : ℕ := puzzles_400 * pieces_per_400 + puzzles_700 * pieces_per_700

def total_hours : ℕ := (total_pieces + pieces_per_hour - 1) / pieces_per_hour

def days_required : ℕ := (total_hours + hours_per_day - 1) / hours_per_day

theorem pablo_puzzle_days : days_required = 22 := by
  sorry

end pablo_puzzle_days_l3434_343474


namespace no_separable_representation_l3434_343456

theorem no_separable_representation :
  ¬ ∃ (f g : ℝ → ℝ), ∀ x y : ℝ, 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end no_separable_representation_l3434_343456


namespace square_roots_problem_l3434_343496

theorem square_roots_problem (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) :
  (∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) →
  (m - 3)^2 = 4 ∧ (m^2 + 2)^(1/3) = 3 := by
sorry

end square_roots_problem_l3434_343496


namespace first_train_speed_is_40_l3434_343461

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := sorry

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 50

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 200

/-- Theorem stating that given the conditions, the speed of the first train is 40 km/h -/
theorem first_train_speed_is_40 : first_train_speed = 40 := by sorry

end first_train_speed_is_40_l3434_343461


namespace matthew_sharing_l3434_343427

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 14

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 21

/-- The number of crackers each friend received -/
def crackers_per_friend : ℕ := 5

/-- The number of cakes each friend received -/
def cakes_per_friend : ℕ := 5

/-- The maximum number of friends Matthew could share with -/
def max_friends : ℕ := 3

theorem matthew_sharing :
  max_friends = min (initial_crackers / crackers_per_friend) (initial_cakes / cakes_per_friend) :=
by sorry

end matthew_sharing_l3434_343427


namespace regular_octagon_interior_angle_l3434_343458

/-- The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle : ℝ :=
  135

#check regular_octagon_interior_angle

end regular_octagon_interior_angle_l3434_343458


namespace largest_three_digit_special_divisible_l3434_343468

theorem largest_three_digit_special_divisible : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) → m ≤ n) ∧
  (n % 6 = 0) ∧
  (∀ d : ℕ, d > 0 ∧ d ≤ 9 ∧ (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d) → n % d = 0) ∧
  n = 843 := by
sorry

end largest_three_digit_special_divisible_l3434_343468


namespace unique_obtainable_pair_l3434_343405

-- Define the calculator operations
def calc_op1 (p : ℕ × ℕ) : ℕ × ℕ := (p.1 + p.2, p.1)
def calc_op2 (p : ℕ × ℕ) : ℕ × ℕ := (2 * p.1 + p.2 + 1, p.1 + p.2 + 1)

-- Define a predicate for pairs obtainable by the calculator
inductive Obtainable : ℕ × ℕ → Prop where
  | initial : Obtainable (1, 1)
  | op1 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op1 p)
  | op2 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op2 p)

-- State the theorem
theorem unique_obtainable_pair :
  ∀ n : ℕ, ∃! k : ℕ, Obtainable (n, k) :=
sorry

end unique_obtainable_pair_l3434_343405


namespace billy_tickets_left_l3434_343416

theorem billy_tickets_left (tickets_won : ℕ) (difference : ℕ) (tickets_left : ℕ) : 
  tickets_won = 48 → 
  difference = 16 → 
  tickets_won - tickets_left = difference → 
  tickets_left = 32 := by
sorry

end billy_tickets_left_l3434_343416


namespace square_sum_xy_l3434_343472

theorem square_sum_xy (x y : ℝ) 
  (h1 : 2 * x * (x + y) = 72) 
  (h2 : 3 * y * (x + y) = 108) : 
  (x + y)^2 = 72 := by
  sorry

end square_sum_xy_l3434_343472


namespace geometric_sequence_solution_l3434_343419

theorem geometric_sequence_solution (x : ℝ) :
  (1 : ℝ) * x = x * 9 → x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_solution_l3434_343419


namespace dishwasher_manager_wage_ratio_l3434_343466

/-- Proves that the ratio of a dishwasher's hourly wage to a manager's hourly wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (manager_wage chef_wage dishwasher_wage : ℝ),
    manager_wage = 8.5 →
    chef_wage = manager_wage - 3.4 →
    chef_wage = dishwasher_wage * 1.2 →
    dishwasher_wage / manager_wage = 0.5 := by
  sorry

end dishwasher_manager_wage_ratio_l3434_343466


namespace min_value_reciprocal_sum_l3434_343445

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 3 = Real.sqrt (3^x * 3^y) → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 := by sorry

end min_value_reciprocal_sum_l3434_343445


namespace line_through_points_l3434_343444

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  eq_at_point1 : (a * point1.1 + b) = point1.2
  eq_at_point2 : (a * point2.1 + b) = point2.2

/-- Theorem stating that for a line y = ax + b passing through (2, 3) and (6, 19), a - b = 9 -/
theorem line_through_points (l : Line) 
    (h1 : l.point1 = (2, 3))
    (h2 : l.point2 = (6, 19)) : 
  l.a - l.b = 9 := by
  sorry

end line_through_points_l3434_343444


namespace committee_formation_l3434_343424

theorem committee_formation (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 20 := by
  sorry

end committee_formation_l3434_343424


namespace total_money_l3434_343411

/-- The amount of money Beth has -/
def beth_money : ℕ := 70

/-- The amount of money Jan has -/
def jan_money : ℕ := 80

/-- The condition that if Beth had $35 more, she would have $105 -/
axiom beth_condition : beth_money + 35 = 105

/-- The condition that if Jan had $10 less, he would have the same money as Beth -/
axiom jan_condition : jan_money - 10 = beth_money

/-- The theorem stating that Beth and Jan have $150 altogether -/
theorem total_money : beth_money + jan_money = 150 := by
  sorry

end total_money_l3434_343411


namespace difference_of_squares_l3434_343455

theorem difference_of_squares : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_l3434_343455


namespace identity_equals_one_l3434_343412

theorem identity_equals_one (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (x - b) * (x - c) / ((a - b) * (a - c)) +
  (x - c) * (x - a) / ((b - c) * (b - a)) +
  (x - a) * (x - b) / ((c - a) * (c - b)) = 1 :=
by sorry

end identity_equals_one_l3434_343412


namespace factorial_sum_ratio_l3434_343441

theorem factorial_sum_ratio : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10) / 
  (1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10) = 19120 := by
  sorry

end factorial_sum_ratio_l3434_343441


namespace exam_candidates_count_l3434_343433

theorem exam_candidates_count :
  ∀ (x : ℕ),
  (x : ℝ) * 0.07 = (x : ℝ) * 0.06 + 82 →
  x = 8200 :=
by
  sorry

end exam_candidates_count_l3434_343433


namespace arithmetic_geometric_sequence_log_sum_l3434_343475

theorem arithmetic_geometric_sequence_log_sum (a b c x y z : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z →
  (∃ d : ℝ, b - a = d ∧ c - b = d) →
  (∃ q : ℝ, y / x = q ∧ z / y = q) →
  (b - c) * Real.log x + (c - a) * Real.log y + (a - b) * Real.log z = 0 := by
  sorry

end arithmetic_geometric_sequence_log_sum_l3434_343475


namespace lcm_from_product_and_hcf_l3434_343429

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 84942 →
  Nat.gcd A B = 33 →
  Nat.lcm A B = 2574 := by
sorry

end lcm_from_product_and_hcf_l3434_343429


namespace line_inclination_angle_l3434_343410

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

def inclination_angle (f : ℝ → ℝ → Prop) : ℝ := sorry

theorem line_inclination_angle :
  inclination_angle line_equation = π * (5/6) := by sorry

end line_inclination_angle_l3434_343410


namespace unique_two_digit_number_l3434_343418

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, n = q * (10 * (n % 10) + n / 10) + r ∧ q = 4 ∧ r = 3) ∧
  (∃ q r : ℕ, n = q * (n / 10 + n % 10) + r ∧ q = 8 ∧ r = 7) ∧
  n = 71 := by sorry

end unique_two_digit_number_l3434_343418


namespace chess_club_girls_l3434_343464

theorem chess_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end chess_club_girls_l3434_343464


namespace side_significant_digits_equal_area_significant_digits_l3434_343417

-- Define the area of the square
def square_area : ℝ := 2.3406

-- Define the precision of the area measurement (to the nearest ten-thousandth)
def area_precision : ℝ := 0.0001

-- Define the function to count significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits_equal_area_significant_digits :
  count_significant_digits (Real.sqrt square_area) = count_significant_digits square_area :=
sorry

end side_significant_digits_equal_area_significant_digits_l3434_343417


namespace pencil_cost_l3434_343400

/-- The cost of an item when paying with a dollar and receiving change -/
def item_cost (payment : ℚ) (change : ℚ) : ℚ :=
  payment - change

/-- Theorem: Given a purchase where the buyer pays with a one-dollar bill
    and receives 65 cents in change, the cost of the item is 35 cents. -/
theorem pencil_cost :
  let payment : ℚ := 1
  let change : ℚ := 65/100
  item_cost payment change = 35/100 := by
  sorry

end pencil_cost_l3434_343400


namespace polynomial_identity_sum_l3434_343469

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 1 := by
sorry

end polynomial_identity_sum_l3434_343469


namespace simplify_radicals_l3434_343484

theorem simplify_radicals : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 160 = 6 * Real.sqrt 10 := by
  sorry

end simplify_radicals_l3434_343484


namespace intersecting_sets_implies_a_equals_one_l3434_343443

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | a * x^2 - 1 = 0 ∧ a > 0}
def N : Set ℝ := {-1/2, 1/2, 1}

-- Define the "intersect" property
def intersect (A B : Set ℝ) : Prop :=
  (∃ x, x ∈ A ∧ x ∈ B) ∧ (¬(A ⊆ B) ∧ ¬(B ⊆ A))

-- State the theorem
theorem intersecting_sets_implies_a_equals_one :
  ∀ a : ℝ, intersect (M a) N → a = 1 :=
by sorry

end intersecting_sets_implies_a_equals_one_l3434_343443


namespace point_meeting_time_l3434_343434

theorem point_meeting_time (b_initial c_initial b_speed c_speed : ℚ) (h1 : b_initial = -8)
  (h2 : c_initial = 16) (h3 : b_speed = 6) (h4 : c_speed = 2) :
  ∃ t : ℚ, t = 2 ∧ c_initial - b_initial - (b_speed + c_speed) * t = 8 :=
by sorry

end point_meeting_time_l3434_343434


namespace freshman_percentage_l3434_343442

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0)
  (h2 : total_students > 0)
  (h3 : (0.2 * 0.4 * freshman) / total_students = 0.04) :
  freshman / total_students = 0.5 := by
sorry

end freshman_percentage_l3434_343442


namespace race_distance_is_140_l3434_343471

/-- The distance of a race, given the times of two runners and the difference in their finishing positions. -/
def race_distance (time_A time_B : ℕ) (difference : ℕ) : ℕ :=
  let speed_A := 140 / time_A
  let speed_B := 140 / time_B
  140

/-- Theorem stating that the race distance is 140 meters under the given conditions. -/
theorem race_distance_is_140 :
  race_distance 36 45 28 = 140 := by
  sorry

end race_distance_is_140_l3434_343471


namespace chess_tournament_director_games_l3434_343420

theorem chess_tournament_director_games (total_games : ℕ) (h : total_games = 325) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ 
  ∀ (k : ℕ), n * (n - 1) / 2 + k = total_games → k ≥ 0 :=
by sorry

end chess_tournament_director_games_l3434_343420


namespace infinite_series_sum_l3434_343407

theorem infinite_series_sum : 
  let a : ℕ → ℚ := λ n => (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)
  ∑' n, a n = 1 / 800 := by sorry

end infinite_series_sum_l3434_343407


namespace scientific_notation_of_billion_l3434_343483

theorem scientific_notation_of_billion (x : ℝ) (h : x = 61345.05) :
  x * (10 : ℝ)^9 = 6.134505 * (10 : ℝ)^12 := by
  sorry

end scientific_notation_of_billion_l3434_343483


namespace triangle_side_ratio_l3434_343437

theorem triangle_side_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) = b / (a + c) + c / (a + b) := by
  sorry

end triangle_side_ratio_l3434_343437


namespace network_connections_l3434_343409

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end network_connections_l3434_343409


namespace fraction_product_proof_l3434_343436

theorem fraction_product_proof : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_product_proof_l3434_343436


namespace imaginary_part_of_z_l3434_343489

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = -4/5 := by
  sorry

end imaginary_part_of_z_l3434_343489


namespace production_rate_equation_correct_l3434_343452

/-- Represents the production rate of the master and apprentice -/
structure ProductionRate where
  master : ℝ
  apprentice : ℝ
  total : ℝ
  master_total : ℝ
  apprentice_total : ℝ

/-- The production rate equation is correct given the conditions -/
theorem production_rate_equation_correct (p : ProductionRate)
  (h1 : p.master + p.apprentice = p.total)
  (h2 : p.total = 40)
  (h3 : p.master_total = 300)
  (h4 : p.apprentice_total = 100) :
  300 / p.master = 100 / (40 - p.master) :=
sorry

end production_rate_equation_correct_l3434_343452


namespace expectation_linear_transform_binomial_probability_normal_probability_l3434_343421

/-- The expectation of a random variable -/
noncomputable def expectation (X : Real → Real) : Real := sorry

/-- The variance of a random variable -/
noncomputable def variance (X : Real → Real) : Real := sorry

/-- The probability mass function for a binomial distribution -/
noncomputable def binomial_pmf (n : Nat) (p : Real) (k : Nat) : Real := sorry

/-- The cumulative distribution function for a normal distribution -/
noncomputable def normal_cdf (μ σ : Real) (x : Real) : Real := sorry

theorem expectation_linear_transform (X : Real → Real) :
  expectation (fun x => 2 * x + 3) = 2 * expectation X + 3 := by sorry

theorem binomial_probability (X : Real → Real) :
  binomial_pmf 6 (1/2) 3 = 5/16 := by sorry

theorem normal_probability (X : Real → Real) (σ : Real) :
  normal_cdf 2 σ 4 = 0.9 →
  normal_cdf 2 σ 2 - normal_cdf 2 σ 0 = 0.4 := by sorry

end expectation_linear_transform_binomial_probability_normal_probability_l3434_343421


namespace point_in_fourth_quadrant_l3434_343457

-- Define the Cartesian coordinate system
def Cartesian := ℝ × ℝ

-- Define a point in the Cartesian coordinate system
def point : Cartesian := (1, -2)

-- Define the fourth quadrant
def fourth_quadrant (p : Cartesian) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant :
  fourth_quadrant point := by sorry

end point_in_fourth_quadrant_l3434_343457


namespace alice_win_probability_l3434_343451

/-- Represents a player in the tournament -/
inductive Player
| Alice
| Bob
| Other

/-- Represents a move in rock-paper-scissors -/
inductive Move
| Rock
| Paper
| Scissors

/-- The number of players in the tournament -/
def numPlayers : Nat := 8

/-- The number of rounds in the tournament -/
def numRounds : Nat := 3

/-- Returns the move of a given player -/
def playerMove (p : Player) : Move :=
  match p with
  | Player.Alice => Move.Rock
  | Player.Bob => Move.Paper
  | Player.Other => Move.Scissors

/-- Determines the winner of a match between two players -/
def matchWinner (p1 p2 : Player) : Player :=
  match playerMove p1, playerMove p2 with
  | Move.Rock, Move.Scissors => p1
  | Move.Scissors, Move.Paper => p1
  | Move.Paper, Move.Rock => p1
  | Move.Scissors, Move.Rock => p2
  | Move.Paper, Move.Scissors => p2
  | Move.Rock, Move.Paper => p2
  | _, _ => p1  -- In case of a tie, p1 wins (representing a coin flip)

/-- The probability of Alice winning the tournament -/
def aliceWinProbability : Rat := 6/7

theorem alice_win_probability :
  aliceWinProbability = 6/7 := by sorry


end alice_win_probability_l3434_343451


namespace digit_equation_solution_l3434_343449

theorem digit_equation_solution :
  ∀ x y z : ℕ,
    x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →
    (10 * x + 5) * (300 + 10 * y + z) = 7850 →
    x = 2 ∧ y = 1 ∧ z = 4 := by
  sorry

end digit_equation_solution_l3434_343449


namespace right_triangle_area_perimeter_relation_l3434_343463

theorem right_triangle_area_perimeter_relation (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a * b = 3 * (a + b + c) →
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨
   (a = 8 ∧ b = 15 ∧ c = 17) ∨
   (a = 9 ∧ b = 12 ∧ c = 15) ∨
   (b = 7 ∧ a = 24 ∧ c = 25) ∨
   (b = 8 ∧ a = 15 ∧ c = 17) ∨
   (b = 9 ∧ a = 12 ∧ c = 15)) :=
by sorry

end right_triangle_area_perimeter_relation_l3434_343463


namespace circles_tangent_implies_a_eq_plus_minus_one_l3434_343473

/-- Circle E with equation x^2 + y^2 = 4 -/
def circle_E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- Circle F with equation x^2 + (y-a)^2 = 1, parameterized by a -/
def circle_F (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 = 1}

/-- Two circles are internally tangent if they have exactly one point in common -/
def internally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ C1 ∧ p ∈ C2

/-- Main theorem: If circles E and F are internally tangent, then a = ±1 -/
theorem circles_tangent_implies_a_eq_plus_minus_one (a : ℝ) :
  internally_tangent (circle_E) (circle_F a) → a = 1 ∨ a = -1 := by
  sorry

end circles_tangent_implies_a_eq_plus_minus_one_l3434_343473


namespace museum_ticket_cost_l3434_343477

theorem museum_ticket_cost (num_students num_teachers : ℕ) 
  (student_ticket_price teacher_ticket_price : ℚ) : 
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_price = 1 →
  teacher_ticket_price = 3 →
  (num_students : ℚ) * student_ticket_price + (num_teachers : ℚ) * teacher_ticket_price = 24 :=
by sorry

end museum_ticket_cost_l3434_343477


namespace A_nonempty_A_subset_B_l3434_343422

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 16}

/-- Theorem for the non-emptiness of A -/
theorem A_nonempty (a : ℝ) : (A a).Nonempty ↔ a ≥ 6 := by sorry

/-- Theorem for A being a subset of B -/
theorem A_subset_B (a : ℝ) : A a ⊆ B ↔ a < 6 ∨ a > 15/2 := by sorry

end A_nonempty_A_subset_B_l3434_343422


namespace equal_savings_l3434_343494

/-- Represents a person's financial data -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  -- Income ratio condition
  p1.income * 4 = p2.income * 5 ∧
  -- Expenditure ratio condition
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  -- P1's income is 5000
  p1.income = 5000 ∧
  -- Savings is income minus expenditure
  p1.savings = p1.income - p1.expenditure ∧
  p2.savings = p2.income - p2.expenditure ∧
  -- Both persons save the same amount
  p1.savings = p2.savings

/-- The theorem to prove -/
theorem equal_savings (p1 p2 : Person) :
  financialProblem p1 p2 → p1.savings = 2000 ∧ p2.savings = 2000 := by
  sorry

end equal_savings_l3434_343494


namespace sine_even_function_phi_l3434_343482

theorem sine_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by sorry

end sine_even_function_phi_l3434_343482


namespace pens_per_student_after_split_l3434_343462

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of red pens each student initially received --/
def red_pens_per_student : ℕ := 62

/-- The number of black pens each student initially received --/
def black_pens_per_student : ℕ := 43

/-- The total number of pens taken after the first month --/
def pens_taken_first_month : ℕ := 37

/-- The total number of pens taken after the second month --/
def pens_taken_second_month : ℕ := 41

/-- Theorem stating that each student will receive 79 pens when the remaining pens are split equally --/
theorem pens_per_student_after_split : 
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let remaining_pens := total_pens - pens_taken_first_month - pens_taken_second_month
  remaining_pens / num_students = 79 := by
  sorry


end pens_per_student_after_split_l3434_343462


namespace largest_product_of_three_l3434_343423

def S : Finset Int := {-5, -4, -1, 2, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 120 :=
sorry

end largest_product_of_three_l3434_343423


namespace max_value_implies_a_value_l3434_343460

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x

-- Define the theorem
theorem max_value_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 3, f a x ≤ M) →
  (a = 1 ∨ a = -3) := by
  sorry

end max_value_implies_a_value_l3434_343460


namespace even_function_inequality_l3434_343467

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0] -/
def IsIncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

/-- Main theorem -/
theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_even : IsEven f)
  (h_incr : IsIncreasingOnNegative f)
  (h_ineq : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end even_function_inequality_l3434_343467


namespace no_perfect_squares_l3434_343408

theorem no_perfect_squares (a b : ℕ) : 
  ¬(∃ (m n : ℕ), a^2 + b = m^2 ∧ b^2 + a = n^2) := by
  sorry

end no_perfect_squares_l3434_343408


namespace integer_representation_l3434_343428

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 := by
  sorry

end integer_representation_l3434_343428


namespace harkamal_payment_l3434_343465

/-- The total amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 65 = 1145 := by
  sorry

end harkamal_payment_l3434_343465


namespace modular_inverse_13_mod_997_l3434_343440

theorem modular_inverse_13_mod_997 :
  ∃ x : ℕ, x < 997 ∧ (13 * x) % 997 = 1 :=
by
  use 767
  sorry

end modular_inverse_13_mod_997_l3434_343440


namespace group_size_l3434_343431

theorem group_size (num_children : ℕ) (num_women : ℕ) (num_men : ℕ) : 
  num_children = 30 →
  num_women = 3 * num_children →
  num_men = 2 * num_women →
  num_children + num_women + num_men = 300 := by
  sorry

#check group_size

end group_size_l3434_343431


namespace product_of_three_consecutive_integers_divisibility_l3434_343486

theorem product_of_three_consecutive_integers_divisibility
  (k : ℤ)
  (n : ℤ)
  (h1 : n = k * (k + 1) * (k + 2))
  (h2 : 5 ∣ n) :
  (6 ∣ n) ∧
  (10 ∣ n) ∧
  (15 ∣ n) ∧
  (30 ∣ n) ∧
  ∃ m : ℤ, n = m ∧ ¬(20 ∣ m) := by
sorry

end product_of_three_consecutive_integers_divisibility_l3434_343486


namespace flag_actions_total_time_l3434_343402

/-- Calculates the total time spent on flag actions throughout the day -/
theorem flag_actions_total_time 
  (pole_height : ℝ) 
  (half_mast : ℝ) 
  (speed_raise : ℝ) 
  (speed_lower_half : ℝ) 
  (speed_raise_half : ℝ) 
  (speed_lower_full : ℝ) 
  (h1 : pole_height = 60) 
  (h2 : half_mast = 30) 
  (h3 : speed_raise = 2) 
  (h4 : speed_lower_half = 3) 
  (h5 : speed_raise_half = 1.5) 
  (h6 : speed_lower_full = 2.5) :
  pole_height / speed_raise + 
  half_mast / speed_lower_half + 
  half_mast / speed_raise_half + 
  pole_height / speed_lower_full = 84 :=
by sorry


end flag_actions_total_time_l3434_343402


namespace expand_and_simplify_l3434_343404

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^2) - 5 * x^3) = 3 / x^2 - 15 * x^3 / 7 := by
  sorry

end expand_and_simplify_l3434_343404


namespace w_squared_value_l3434_343415

theorem w_squared_value (w : ℚ) (h : (w + 16)^2 = (4*w + 9)*(3*w + 6)) : 
  w^2 = 5929 / 484 := by
  sorry

end w_squared_value_l3434_343415


namespace optimal_profit_l3434_343490

/-- Profit function for n plants per pot -/
def P (n : ℕ) : ℝ := n * (5 - 0.5 * (n - 3))

/-- The optimal number of plants per pot -/
def optimal_plants : ℕ := 5

theorem optimal_profit :
  (P optimal_plants = 20) ∧ 
  (∀ n : ℕ, 3 ≤ n ∧ n ≤ 6 → P n ≤ 20) ∧
  (∀ n : ℕ, 3 ≤ n ∧ n < optimal_plants → P n < 20) ∧
  (∀ n : ℕ, optimal_plants < n ∧ n ≤ 6 → P n < 20) := by
  sorry

#eval P optimal_plants  -- Should output 20

end optimal_profit_l3434_343490


namespace price_decrease_approx_l3434_343448

/-- Original price in dollars for 6 cups -/
def original_price : ℚ := 8

/-- Number of cups in original offer -/
def original_cups : ℕ := 6

/-- Promotional price in dollars for 8 cups -/
def promo_price : ℚ := 6

/-- Number of cups in promotional offer -/
def promo_cups : ℕ := 8

/-- Calculate the percent decrease in price per cup -/
def percent_decrease : ℚ :=
  (original_price / original_cups - promo_price / promo_cups) / (original_price / original_cups) * 100

/-- Theorem stating that the percent decrease is approximately 43.6% -/
theorem price_decrease_approx :
  abs (percent_decrease - 43.6) < 0.1 := by sorry

end price_decrease_approx_l3434_343448


namespace artemon_distance_l3434_343487

-- Define the rectangle
def rectangle_length : ℝ := 6
def rectangle_width : ℝ := 2.5

-- Define speeds
def malvina_speed : ℝ := 4
def buratino_speed : ℝ := 6
def artemon_speed : ℝ := 12

-- Theorem statement
theorem artemon_distance :
  let diagonal : ℝ := Real.sqrt (rectangle_length^2 + rectangle_width^2)
  let meeting_time : ℝ := diagonal / (malvina_speed + buratino_speed)
  let artemon_distance : ℝ := artemon_speed * meeting_time
  artemon_distance = 7.8 := by sorry

end artemon_distance_l3434_343487


namespace expression_evaluation_l3434_343446

theorem expression_evaluation :
  let a : ℚ := -1/2
  let b : ℚ := 3
  3 * a^2 - b^2 - (a^2 - 6*a) - 2*(-b^2 + 3*a) = 19/2 := by sorry

end expression_evaluation_l3434_343446


namespace root_equation_value_l3434_343438

theorem root_equation_value (m : ℝ) : 
  (2 * m^2 + 3 * m - 1 = 0) → (4 * m^2 + 6 * m - 2019 = -2017) := by
  sorry

end root_equation_value_l3434_343438


namespace parallel_segments_k_value_l3434_343497

/-- Given four points on a Cartesian plane where segment AB is parallel to segment XY, 
    prove that k = -8. -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 0))
  (hB : B = (0, -6))
  (hX : X = (0, 10))
  (hY : Y = (18, k))
  (h_parallel : (B.2 - A.2) * (Y.1 - X.1) = (Y.2 - X.2) * (B.1 - A.1)) :
  k = -8 :=
by sorry

end parallel_segments_k_value_l3434_343497


namespace spade_then_ace_probability_l3434_343499

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Theorem: The probability of drawing a spade first and an Ace second from a standard 52-card deck is 1/52 -/
theorem spade_then_ace_probability :
  (NumSpades / StandardDeck) * (NumAces / (StandardDeck - 1)) = 1 / StandardDeck :=
sorry

end spade_then_ace_probability_l3434_343499


namespace parabola_line_intersection_l3434_343488

/-- Parabola defined by x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point (x₀, y₀) is inside the parabola if x₀² < 4y₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := x₀^2 < 4*y₀

/-- Line defined by x₀x = 2(y + y₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := x₀*x = 2*(y + y₀)

/-- No common points between the line and the parabola -/
def no_common_points (x₀ y₀ : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → line x₀ y₀ x y → False

theorem parabola_line_intersection (x₀ y₀ : ℝ) 
  (h : inside_parabola x₀ y₀) : no_common_points x₀ y₀ := by
  sorry

end parabola_line_intersection_l3434_343488


namespace factorization_equality_l3434_343491

theorem factorization_equality (m n : ℝ) : m^2*n - 2*m*n + n = n*(m-1)^2 := by
  sorry

end factorization_equality_l3434_343491


namespace factorization_equality_l3434_343492

theorem factorization_equality (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) := by
sorry

end factorization_equality_l3434_343492


namespace gcd_1908_4187_l3434_343498

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end gcd_1908_4187_l3434_343498


namespace max_stone_value_l3434_343450

/-- Represents the types of stones --/
inductive StoneType
| FivePound
| FourPound
| OnePound

/-- Returns the weight of a stone type in pounds --/
def weight (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 5
  | StoneType.FourPound => 4
  | StoneType.OnePound => 1

/-- Returns the value of a stone type in dollars --/
def value (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 14
  | StoneType.FourPound => 11
  | StoneType.OnePound => 2

/-- Represents a combination of stones --/
structure StoneCombination where
  fivePound : ℕ
  fourPound : ℕ
  onePound : ℕ

/-- Calculates the total weight of a stone combination --/
def totalWeight (c : StoneCombination) : ℕ :=
  c.fivePound * weight StoneType.FivePound +
  c.fourPound * weight StoneType.FourPound +
  c.onePound * weight StoneType.OnePound

/-- Calculates the total value of a stone combination --/
def totalValue (c : StoneCombination) : ℕ :=
  c.fivePound * value StoneType.FivePound +
  c.fourPound * value StoneType.FourPound +
  c.onePound * value StoneType.OnePound

/-- Defines a valid stone combination --/
def isValidCombination (c : StoneCombination) : Prop :=
  totalWeight c ≤ 18 ∧ c.fivePound ≤ 20 ∧ c.fourPound ≤ 20 ∧ c.onePound ≤ 20

theorem max_stone_value :
  ∃ (c : StoneCombination), isValidCombination c ∧
    totalValue c = 50 ∧
    ∀ (c' : StoneCombination), isValidCombination c' → totalValue c' ≤ 50 :=
by sorry

end max_stone_value_l3434_343450


namespace hyperbola_cosine_theorem_l3434_343447

/-- A hyperbola with equation x^2 - y^2 = 2 -/
def Hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_cosine_theorem :
  Hyperbola P.1 P.2 →
  distance P F₁ = 2 * distance P F₂ →
  let cosine_angle := (distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2
                    / (2 * distance P F₁ * distance P F₂)
  cosine_angle = 3/4 := by sorry

end hyperbola_cosine_theorem_l3434_343447


namespace weight_replacement_l3434_343470

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 93 →
  avg_increase = 3.5 →
  new_weight - n * avg_increase = 65 := by
  sorry

end weight_replacement_l3434_343470


namespace octal_734_equals_decimal_476_l3434_343476

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

/-- The octal number 734 is equal to 476 in decimal --/
theorem octal_734_equals_decimal_476 : octal_to_decimal 734 = 476 := by
  sorry

end octal_734_equals_decimal_476_l3434_343476


namespace smallBase_altitude_ratio_l3434_343432

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  smallBase : ℝ
  /-- Length of the larger base -/
  largeBase : ℝ
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- The larger base is twice the smaller base -/
  largeBase_eq : largeBase = 2 * smallBase
  /-- The diagonal is 1.5 times the larger base -/
  diagonal_eq : diagonal = 1.5 * largeBase
  /-- The altitude equals the smaller base -/
  altitude_eq : altitude = smallBase

/-- Theorem: The ratio of the smaller base to the altitude is 1:1 -/
theorem smallBase_altitude_ratio (t : IsoscelesTrapezoid) : t.smallBase / t.altitude = 1 := by
  sorry

end smallBase_altitude_ratio_l3434_343432


namespace square_sum_given_linear_equations_l3434_343454

theorem square_sum_given_linear_equations (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end square_sum_given_linear_equations_l3434_343454


namespace art_museum_cost_l3434_343481

def total_cost (initial_fee : ℕ) (initial_visits_per_year : ℕ) (new_fee : ℕ) (new_visits_per_year : ℕ) (total_years : ℕ) : ℕ :=
  (initial_fee * initial_visits_per_year) + (new_fee * new_visits_per_year * (total_years - 1))

theorem art_museum_cost : 
  total_cost 5 12 7 4 3 = 116 := by sorry

end art_museum_cost_l3434_343481


namespace sqrt_two_thirds_same_type_as_sqrt6_l3434_343493

-- Define what it means for a real number to be of the same type as √6
def same_type_as_sqrt6 (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * Real.sqrt 2 * b * Real.sqrt 3

-- State the theorem
theorem sqrt_two_thirds_same_type_as_sqrt6 :
  same_type_as_sqrt6 (Real.sqrt (2/3)) :=
sorry

end sqrt_two_thirds_same_type_as_sqrt6_l3434_343493


namespace toms_weekly_fluid_intake_l3434_343435

/-- Calculates the total fluid intake in ounces for a week given daily soda and water consumption --/
def weekly_fluid_intake (soda_cans : ℕ) (oz_per_can : ℕ) (water_oz : ℕ) : ℕ :=
  7 * (soda_cans * oz_per_can + water_oz)

/-- Theorem stating Tom's weekly fluid intake --/
theorem toms_weekly_fluid_intake :
  weekly_fluid_intake 5 12 64 = 868 := by
  sorry

end toms_weekly_fluid_intake_l3434_343435


namespace product_probability_l3434_343478

/-- Claire's spinner has 7 equally probable outcomes -/
def claire_spinner : ℕ := 7

/-- Jamie's spinner has 12 equally probable outcomes -/
def jamie_spinner : ℕ := 12

/-- The threshold for the product of spins -/
def threshold : ℕ := 42

/-- The probability that the product of Claire's and Jamie's spins is less than the threshold -/
theorem product_probability : 
  (Finset.filter (λ (pair : ℕ × ℕ) => pair.1 * pair.2 < threshold) 
    (Finset.product (Finset.range claire_spinner) (Finset.range jamie_spinner))).card / 
  (claire_spinner * jamie_spinner : ℚ) = 31 / 42 := by sorry

end product_probability_l3434_343478


namespace circle_circumference_increase_l3434_343406

theorem circle_circumference_increase (r : ℝ) : 
  2 * Real.pi * (r + 2) - 2 * Real.pi * r = 12.56 := by
  sorry

end circle_circumference_increase_l3434_343406


namespace dataset_transformation_l3434_343401

theorem dataset_transformation (initial_points : ℕ) : 
  initial_points = 200 →
  let increased_points := initial_points + initial_points / 5
  let final_points := increased_points - increased_points / 4
  final_points = 180 := by
sorry

end dataset_transformation_l3434_343401


namespace aang_fish_count_l3434_343414

theorem aang_fish_count :
  ∀ (aang_fish : ℕ),
  let sokka_fish : ℕ := 5
  let toph_fish : ℕ := 12
  let total_people : ℕ := 3
  let average_fish : ℕ := 8
  (aang_fish + sokka_fish + toph_fish) / total_people = average_fish →
  aang_fish = 7 := by
sorry

end aang_fish_count_l3434_343414


namespace bookcase_max_weight_bookcase_weight_proof_l3434_343453

/-- The maximum weight a bookcase can hold given the weights of various items and the excess weight -/
theorem bookcase_max_weight 
  (hardcover_weight : ℝ) 
  (textbook_weight : ℝ) 
  (knickknack_weight : ℝ) 
  (excess_weight : ℝ) : ℝ :=
  let total_weight := hardcover_weight + textbook_weight + knickknack_weight
  total_weight - excess_weight

/-- Proves that the bookcase can hold 80 pounds given the specified conditions -/
theorem bookcase_weight_proof 
  (hardcover_weight : ℝ) 
  (textbook_weight : ℝ) 
  (knickknack_weight : ℝ) 
  (excess_weight : ℝ) :
  hardcover_weight = 70 * 0.5 →
  textbook_weight = 30 * 2 →
  knickknack_weight = 3 * 6 →
  excess_weight = 33 →
  bookcase_max_weight hardcover_weight textbook_weight knickknack_weight excess_weight = 80 := by
  sorry

end bookcase_max_weight_bookcase_weight_proof_l3434_343453


namespace required_weekly_hours_l3434_343495

/-- Calculates the required weekly work hours to meet a financial goal given previous work data and future plans. -/
theorem required_weekly_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_total_earnings : ℚ) 
  (future_weeks : ℕ) 
  (future_earnings_goal : ℚ) : 
  summer_weeks > 0 ∧ 
  summer_hours_per_week > 0 ∧ 
  summer_total_earnings > 0 ∧ 
  future_weeks > 0 ∧ 
  future_earnings_goal > 0 →
  (future_earnings_goal / (summer_total_earnings / (summer_weeks * summer_hours_per_week))) / future_weeks = 45 / 16 := by
  sorry

#eval (4500 : ℚ) / ((3600 : ℚ) / (8 * 45)) / 40

end required_weekly_hours_l3434_343495


namespace circle_equation_l3434_343403

theorem circle_equation (x y : ℝ) : 
  (∃ h k r : ℝ, (5*h - 3*k = 8) ∧ 
    ((x - h)^2 + (y - k)^2 = r^2) ∧ 
    (h = r ∨ k = r) ∧ 
    (h = r ∨ k = -r)) →
  ((x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1) :=
by sorry

end circle_equation_l3434_343403


namespace sum_of_x_and_y_l3434_343479

theorem sum_of_x_and_y (x y : ℚ) 
  (eq1 : 5 * x - 7 * y = 17) 
  (eq2 : 3 * x + 5 * y = 11) : 
  x + y = 83 / 23 := by
sorry

end sum_of_x_and_y_l3434_343479


namespace combined_diving_depths_l3434_343485

theorem combined_diving_depths (ron_height : ℝ) (water_depth : ℝ) : 
  ron_height = 12 →
  water_depth = 5 * ron_height →
  let dean_height := ron_height - 11
  let sam_height := dean_height + 2
  let ron_dive := ron_height / 2
  let sam_dive := sam_height
  let dean_dive := dean_height + 3
  ron_dive + sam_dive + dean_dive = 13 := by sorry

end combined_diving_depths_l3434_343485


namespace specific_box_volume_l3434_343480

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def box_volume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

/-- Theorem: The volume of the specific box described in the problem -/
theorem specific_box_volume (x : ℝ) :
  box_volume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end specific_box_volume_l3434_343480


namespace sum_of_digits_of_large_number_l3434_343426

theorem sum_of_digits_of_large_number : ∃ S : ℕ, 
  S = 10^2021 - 2021 ∧ 
  (∃ digits : List ℕ, 
    digits.sum = 18185 ∧ 
    digits.all (λ d => d < 10) ∧
    S = digits.foldr (λ d acc => d + 10 * acc) 0) :=
by sorry

end sum_of_digits_of_large_number_l3434_343426


namespace volume_rotated_square_l3434_343439

/-- The volume of a solid formed by rotating a square around its diagonal -/
theorem volume_rotated_square (area : ℝ) (volume : ℝ) : 
  area = 4 → volume = (4 * Real.sqrt 2 * Real.pi) / 3 := by
  sorry

end volume_rotated_square_l3434_343439


namespace snickers_bars_proof_l3434_343459

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- Calculates the number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ :=
  (total_points_needed - chocolate_bunnies_sold * points_per_bunny) / points_per_snickers

theorem snickers_bars_proof :
  snickers_bars_needed = 48 := by
  sorry

end snickers_bars_proof_l3434_343459


namespace distribute_five_to_three_l3434_343425

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is 150 -/
theorem distribute_five_to_three : distribute 5 3 = 150 := by
  sorry

end distribute_five_to_three_l3434_343425


namespace no_solution_exists_l3434_343413

theorem no_solution_exists : ¬∃ (a b c d : ℤ),
  (a * b * c * d - a = 1961) ∧
  (a * b * c * d - b = 961) ∧
  (a * b * c * d - c = 61) ∧
  (a * b * c * d - d = 1) :=
by sorry

end no_solution_exists_l3434_343413
