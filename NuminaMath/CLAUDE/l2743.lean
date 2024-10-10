import Mathlib

namespace exists_n_no_rational_solution_l2743_274376

-- Define a quadratic polynomial with real coefficients
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem exists_n_no_rational_solution (a b c : ℝ) :
  ∃ n : ℕ, ∀ x : ℚ, QuadraticPolynomial a b c x ≠ (1 : ℝ) / n := by
  sorry

end exists_n_no_rational_solution_l2743_274376


namespace concert_ticket_cost_l2743_274390

/-- Calculates the total cost of concert tickets --/
def concertTicketCost (generalAdmissionPrice : ℚ) (vipPrice : ℚ) (premiumPrice : ℚ)
                      (generalAdmissionQuantity : ℕ) (vipQuantity : ℕ) (premiumQuantity : ℕ)
                      (generalAdmissionDiscount : ℚ) (vipDiscount : ℚ) : ℚ :=
  let generalAdmissionCost := generalAdmissionPrice * generalAdmissionQuantity * (1 - generalAdmissionDiscount)
  let vipCost := vipPrice * vipQuantity * (1 - vipDiscount)
  let premiumCost := premiumPrice * premiumQuantity
  generalAdmissionCost + vipCost + premiumCost

theorem concert_ticket_cost :
  concertTicketCost 6 10 15 6 2 1 (1/10) (3/20) = 644/10 := by
  sorry

end concert_ticket_cost_l2743_274390


namespace geometric_sequence_a6_l2743_274391

/-- Given a geometric sequence {a_n} with common ratio q and a_2 = 8, prove that a_6 = 128 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 2 = 8) 
  (h3 : ∀ n : ℕ, a (n + 1) = q * a n) : a 6 = 128 := by
  sorry

end geometric_sequence_a6_l2743_274391


namespace rationalize_denominator_l2743_274370

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = A + B * Real.sqrt C ∧ A = -2 ∧ B = -1 ∧ C = 3 := by
  sorry

end rationalize_denominator_l2743_274370


namespace cube_edge_multiple_l2743_274363

/-- The ratio of the volumes of two cubes -/
def volume_ratio : ℝ := 0.03703703703703703

/-- Theorem: If the ratio of the volume of cube Q to the volume of cube P is 0.03703703703703703,
    and the length of an edge of cube P is some multiple k of the length of an edge of cube Q,
    then k = 3. -/
theorem cube_edge_multiple (q p k : ℝ) (hq : q > 0) (hp : p > 0) (hk : k > 0)
  (h_edge : p = k * q) (h_volume : q^3 / p^3 = volume_ratio) : k = 3 := by
  sorry

end cube_edge_multiple_l2743_274363


namespace grocery_store_costs_l2743_274313

/-- Grocery store daily operation costs problem -/
theorem grocery_store_costs (total_costs : ℝ) (employees_salary_ratio : ℝ) (delivery_costs_ratio : ℝ)
  (h1 : total_costs = 4000)
  (h2 : employees_salary_ratio = 2 / 5)
  (h3 : delivery_costs_ratio = 1 / 4) :
  total_costs - (employees_salary_ratio * total_costs + delivery_costs_ratio * (total_costs - employees_salary_ratio * total_costs)) = 1800 := by
  sorry

end grocery_store_costs_l2743_274313


namespace consecutive_integer_averages_l2743_274382

theorem consecutive_integer_averages (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (7 * c + 21) / 7) →
  ((7 * d + 21) / 7 = c + 6) :=
by sorry

end consecutive_integer_averages_l2743_274382


namespace non_attacking_knights_count_l2743_274358

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Checks if two positions are distinct --/
def are_distinct (p1 p2 : Position) : Prop :=
  p1 ≠ p2

/-- Calculates the square of the distance between two positions --/
def distance_squared (p1 p2 : Position) : Nat :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if two knights attack each other --/
def knights_attack (p1 p2 : Position) : Prop :=
  distance_squared p1 p2 = 5

/-- Counts the number of ways to place two knights that do not attack each other --/
def count_non_attacking_placements (board : Chessboard) : Nat :=
  sorry

theorem non_attacking_knights_count :
  ∀ (board : Chessboard),
    board.size = 8 →
    count_non_attacking_placements board = 1848 :=
by sorry

end non_attacking_knights_count_l2743_274358


namespace inequality_proof_l2743_274375

theorem inequality_proof (a b : ℝ) : (6*a - 3*b - 3) * (a^2 + a^2*b - 2*a^3) ≤ 0 := by
  sorry

end inequality_proof_l2743_274375


namespace total_oranges_picked_l2743_274302

/-- Represents the number of oranges picked by Jeremy on Monday -/
def monday_pick : ℕ := 100

/-- Represents the number of oranges picked by Jeremy and his brother on Tuesday -/
def tuesday_pick : ℕ := 3 * monday_pick

/-- Represents the number of oranges picked by all three on Wednesday -/
def wednesday_pick : ℕ := 2 * tuesday_pick

/-- Represents the number of oranges picked by the cousin on Wednesday -/
def cousin_wednesday_pick : ℕ := tuesday_pick - (tuesday_pick / 5)

/-- Represents the number of oranges picked by Jeremy on Thursday -/
def jeremy_thursday_pick : ℕ := (7 * monday_pick) / 10

/-- Represents the number of oranges picked by the brother on Thursday -/
def brother_thursday_pick : ℕ := tuesday_pick - monday_pick

/-- Represents the number of oranges picked by the cousin on Thursday -/
def cousin_thursday_pick : ℕ := cousin_wednesday_pick + (3 * cousin_wednesday_pick) / 10

/-- Represents the total number of oranges picked over the four days -/
def total_picked : ℕ := monday_pick + tuesday_pick + wednesday_pick + 
  (jeremy_thursday_pick + brother_thursday_pick + cousin_thursday_pick)

theorem total_oranges_picked : total_picked = 1642 := by sorry

end total_oranges_picked_l2743_274302


namespace red_crayons_per_person_l2743_274364

def initial_rulers : ℕ := 11
def initial_crayons : ℕ := 34
def tim_added_rulers : ℕ := 14
def jane_removed_crayons : ℕ := 20
def jane_added_blue_crayons : ℕ := 8
def number_of_people : ℕ := 3

def total_red_crayons : ℕ := 2 * jane_added_blue_crayons

theorem red_crayons_per_person :
  total_red_crayons / number_of_people = 5 := by sorry

end red_crayons_per_person_l2743_274364


namespace area_of_triangle_is_5_l2743_274385

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Define the triangle formed by the line and coordinate axes
def triangle_area : ℝ := 5

-- Theorem statement
theorem area_of_triangle_is_5 :
  triangle_area = 5 :=
by sorry

end area_of_triangle_is_5_l2743_274385


namespace tuna_sales_problem_l2743_274392

/-- The number of packs of tuna fish sold per hour during the peak season -/
def peak_packs_per_hour : ℕ := 6

/-- The price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- The number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- The additional revenue made during the high season compared to the low season, in dollars -/
def additional_revenue : ℕ := 1800

/-- The number of packs of tuna fish sold per hour during the low season -/
def low_season_packs : ℕ := 4

theorem tuna_sales_problem :
  peak_packs_per_hour * price_per_pack * hours_per_day =
  low_season_packs * price_per_pack * hours_per_day + additional_revenue := by
  sorry

end tuna_sales_problem_l2743_274392


namespace bus_speed_problem_l2743_274393

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := 30

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when both are driving west, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if they drove towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_problem :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by sorry

end bus_speed_problem_l2743_274393


namespace expression_simplification_l2743_274300

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 3) :
  (a - 3) / (a^2 + 6*a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l2743_274300


namespace book_collection_ratio_l2743_274389

theorem book_collection_ratio (first_week : ℕ) (total : ℕ) : 
  first_week = 9 → total = 99 → 
  (total - first_week) / first_week = 10 := by
sorry

end book_collection_ratio_l2743_274389


namespace largest_divisor_of_five_consecutive_integers_l2743_274318

theorem largest_divisor_of_five_consecutive_integers (n : ℤ) :
  ∃ (k : ℤ), k * 60 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ (m : ℤ), m > 60 → ¬∃ (j : ℤ), j * m = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end largest_divisor_of_five_consecutive_integers_l2743_274318


namespace product_17_reciprocal_squares_sum_l2743_274303

theorem product_17_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 17 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 290 / 289 := by
  sorry

end product_17_reciprocal_squares_sum_l2743_274303


namespace max_cash_prize_value_l2743_274365

/-- Represents the promotion setup in a shopping mall -/
structure PromotionSetup where
  total_items : Nat
  daily_necessities : Nat
  chosen_items : Nat
  price_increase : ℝ
  lottery_chances : Nat
  win_probability : ℝ

/-- Calculates the expected value of the total cash prize -/
def expected_cash_prize (m : ℝ) (setup : PromotionSetup) : ℝ :=
  setup.lottery_chances * setup.win_probability * m

/-- Theorem stating the maximum value of m for an advantageous promotion -/
theorem max_cash_prize_value (setup : PromotionSetup) :
  setup.total_items = 7 →
  setup.daily_necessities = 3 →
  setup.chosen_items = 3 →
  setup.price_increase = 150 →
  setup.lottery_chances = 3 →
  setup.win_probability = 1/2 →
  ∃ (m : ℝ), m = 100 ∧ 
    ∀ (x : ℝ), expected_cash_prize x setup ≤ setup.price_increase → x ≤ m :=
by sorry

end max_cash_prize_value_l2743_274365


namespace race_earnings_theorem_l2743_274357

/-- Represents the race parameters and results -/
structure RaceData where
  duration : ℕ         -- Race duration in minutes
  lap_distance : ℕ     -- Distance of one lap in meters
  certificate_rate : ℚ -- Gift certificate rate in dollars per 100 meters
  winner_laps : ℕ      -- Number of laps run by the winner

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (data : RaceData) : ℚ :=
  (data.winner_laps * data.lap_distance * data.certificate_rate) / (100 * data.duration)

/-- Theorem stating that for the given race conditions, the average earnings per minute is $7 -/
theorem race_earnings_theorem (data : RaceData) 
  (h1 : data.duration = 12)
  (h2 : data.lap_distance = 100)
  (h3 : data.certificate_rate = 7/2)
  (h4 : data.winner_laps = 24) :
  average_earnings_per_minute data = 7 := by
  sorry

end race_earnings_theorem_l2743_274357


namespace number_satisfying_equation_l2743_274395

theorem number_satisfying_equation : ∃! x : ℝ, 3 * (x + 2) = 24 + x := by
  sorry

end number_satisfying_equation_l2743_274395


namespace minimum_buses_required_l2743_274310

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 535) (h2 : bus_capacity = 45) :
  ∃ (num_buses : ℕ), num_buses * bus_capacity ≥ total_students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ num_buses ∧
  num_buses = 12 :=
sorry

end minimum_buses_required_l2743_274310


namespace biased_coin_probability_l2743_274306

theorem biased_coin_probability : ∃ (h : ℝ),
  (0 < h ∧ h < 1) ∧
  (Nat.choose 6 2 * h^2 * (1-h)^4 = Nat.choose 6 3 * h^3 * (1-h)^3) ∧
  (Nat.choose 6 4 * h^4 * (1-h)^2 = 19440 / 117649) :=
by sorry

end biased_coin_probability_l2743_274306


namespace f_increasing_and_range_l2743_274324

-- Define the function f and its properties
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0) ∧
  (f (-1) = -2)

-- Theorem statement
theorem f_increasing_and_range (f : ℝ → ℝ) (hf : f_properties f) :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc (-2) 1 = Set.Icc (-4) 2) :=
by sorry

end f_increasing_and_range_l2743_274324


namespace smallest_impossible_score_l2743_274372

def dart_scores : Set ℕ := {0, 1, 3, 7, 8, 12}

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_impossible_score :
  (∀ m : ℕ, m < 22 → is_valid_sum m) ∧ ¬is_valid_sum 22 :=
sorry

end smallest_impossible_score_l2743_274372


namespace x_squared_minus_y_squared_equals_five_l2743_274367

theorem x_squared_minus_y_squared_equals_five
  (a : ℝ) (x y : ℝ) (h1 : a^x * a^y = a^5) (h2 : a^x / a^y = a) :
  x^2 - y^2 = 5 :=
by sorry

end x_squared_minus_y_squared_equals_five_l2743_274367


namespace fraction_squared_times_32_equals_8_l2743_274379

theorem fraction_squared_times_32_equals_8 : ∃ f : ℚ, f^2 * 32 = 2^3 ∧ f = 1/2 := by
  sorry

end fraction_squared_times_32_equals_8_l2743_274379


namespace percentage_comparisons_l2743_274353

theorem percentage_comparisons (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  (x / y) * 100 = 80 ∧
  ((y - x) / x) * 100 = 25 ∧
  ((y - x) / y) * 100 = 20 := by
  sorry

end percentage_comparisons_l2743_274353


namespace min_value_problem_l2743_274344

theorem min_value_problem (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 75) : 
  y₁^2 + 2 * y₂^2 + 3 * y₃^2 ≥ 5625 / 29 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 2 * y₂'^2 + 3 * y₃'^2 = 5625 / 29 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 2 * y₁' + 3 * y₂' + 4 * y₃' = 75 :=
by sorry

end min_value_problem_l2743_274344


namespace smallest_four_digit_mod_5_l2743_274305

theorem smallest_four_digit_mod_5 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 5 = 4 → n ≥ 1004 := by
  sorry

end smallest_four_digit_mod_5_l2743_274305


namespace max_dot_product_in_triangle_l2743_274388

theorem max_dot_product_in_triangle (A B C P : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BAC := Real.arccos ((AB^2 + AC^2 - (B.1 - C.1)^2 - (B.2 - C.2)^2) / (2 * AB * AC))
  let AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  let dot_product := PB.1 * PC.1 + PB.2 * PC.2
  AB = 3 ∧ AC = 4 ∧ BAC = π/3 ∧ AP = 2 →
  ∃ P_max : ℝ × ℝ, dot_product ≤ 10 + 2 * Real.sqrt 37 ∧
            ∃ P_actual : ℝ × ℝ, dot_product = 10 + 2 * Real.sqrt 37 :=
by sorry

end max_dot_product_in_triangle_l2743_274388


namespace polynomial_expansion_l2743_274355

theorem polynomial_expansion (x : ℝ) :
  (7 * x + 5) * (3 * x^2 - 2 * x + 4) = 21 * x^3 + x^2 + 18 * x + 20 := by
  sorry

end polynomial_expansion_l2743_274355


namespace gcd_78_143_l2743_274346

theorem gcd_78_143 : Nat.gcd 78 143 = 13 := by
  sorry

end gcd_78_143_l2743_274346


namespace sin_2alpha_value_l2743_274312

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/4) π) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (π/4 - α)) : 
  Real.sin (2 * α) = -1/9 := by
  sorry

end sin_2alpha_value_l2743_274312


namespace inequality_proof_l2743_274371

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (1 + Real.sqrt x)^2) + (1 / (1 + Real.sqrt y)^2) ≥ 2 / (x + y + 2) := by
  sorry

end inequality_proof_l2743_274371


namespace smallest_number_divisible_l2743_274352

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

end smallest_number_divisible_l2743_274352


namespace race_track_width_proof_l2743_274362

def inner_circumference : Real := 440
def outer_radius : Real := 84.02817496043394

theorem race_track_width_proof :
  let inner_radius := inner_circumference / (2 * Real.pi)
  let width := outer_radius - inner_radius
  ∃ ε > 0, abs (width - 14.021) < ε :=
by sorry

end race_track_width_proof_l2743_274362


namespace arithmetic_sequence_proof_l2743_274369

/-- An arithmetic sequence with given second and eighth terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = -6)
  (h_a8 : a 8 = -18) :
  (∃ d : ℤ, d = -2 ∧ ∀ n : ℕ, a n = -2 * n - 2) :=
sorry

end arithmetic_sequence_proof_l2743_274369


namespace unique_prime_perfect_square_l2743_274331

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 5 * p * (2^(p+1) - 1) = k^2 :=
sorry

end unique_prime_perfect_square_l2743_274331


namespace lateral_surface_area_rotated_unit_square_l2743_274307

/-- The lateral surface area of a cylinder formed by rotating a square with area 1 around one of its sides. -/
theorem lateral_surface_area_rotated_unit_square : 
  ∀ (square_area : ℝ) (cylinder_height : ℝ) (cylinder_base_circumference : ℝ),
    square_area = 1 →
    cylinder_height = Real.sqrt square_area →
    cylinder_base_circumference = Real.sqrt square_area →
    cylinder_height * cylinder_base_circumference = 1 := by
  sorry

#check lateral_surface_area_rotated_unit_square

end lateral_surface_area_rotated_unit_square_l2743_274307


namespace stones_per_bracelet_l2743_274351

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0) 
  (h2 : num_bracelets = 8.0) : 
  total_stones / num_bracelets = 11.0 := by
  sorry

end stones_per_bracelet_l2743_274351


namespace max_puzzle_sets_l2743_274341

/-- Represents the number of puzzles in a set -/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a PuzzleSet is valid according to the given conditions -/
def isValidSet (s : PuzzleSet) : Prop :=
  s.logic + s.visual + s.word ≥ 5 ∧ 2 * s.visual = s.logic

/-- The theorem to be proved -/
theorem max_puzzle_sets :
  ∀ (n : ℕ),
    (∃ (s : PuzzleSet),
      isValidSet s ∧
      n * s.logic ≤ 30 ∧
      n * s.visual ≤ 18 ∧
      n * s.word ≤ 12 ∧
      n * s.logic + n * s.visual + n * s.word = 30 + 18 + 12) →
    n ≤ 3 :=
by sorry

end max_puzzle_sets_l2743_274341


namespace train_distance_theorem_l2743_274301

/-- Calculates the distance traveled by a train given its average speed and travel time. -/
def train_distance (average_speed : ℝ) (travel_time : ℝ) : ℝ :=
  average_speed * travel_time

/-- Represents the train journey details -/
structure TrainJourney where
  average_speed : ℝ
  start_time : ℝ
  end_time : ℝ
  halt_time : ℝ

/-- Theorem stating the distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.average_speed = 87)
  (h2 : journey.start_time = 9)
  (h3 : journey.end_time = 13.75)
  (h4 : journey.halt_time = 0.75) :
  train_distance journey.average_speed (journey.end_time - journey.start_time - journey.halt_time) = 348 := by
  sorry


end train_distance_theorem_l2743_274301


namespace stairs_distance_l2743_274380

theorem stairs_distance (total_time speed_up speed_down : ℝ) 
  (h_total_time : total_time = 4)
  (h_speed_up : speed_up = 2)
  (h_speed_down : speed_down = 3)
  (h_distance_diff : ∀ d : ℝ, d / speed_up + (d + 2) / speed_down = total_time) :
  ∃ d : ℝ, d + 2 = 6 := by
sorry

end stairs_distance_l2743_274380


namespace exists_valid_arrangement_l2743_274333

/-- Represents a circle placement arrangement in a square -/
structure CircleArrangement where
  n : ℕ  -- Side length of the square
  num_circles : ℕ  -- Number of circles placed

/-- Checks if a circle arrangement is valid -/
def is_valid_arrangement (arr : CircleArrangement) : Prop :=
  arr.n ≥ 8 ∧ arr.num_circles > arr.n^2

/-- Theorem stating the existence of a valid circle arrangement -/
theorem exists_valid_arrangement : 
  ∃ (arr : CircleArrangement), is_valid_arrangement arr :=
sorry

end exists_valid_arrangement_l2743_274333


namespace segment_sum_after_n_halvings_sum_after_million_halvings_l2743_274345

/-- The sum of numbers on a segment after n halvings -/
def segmentSum (n : ℕ) : ℕ :=
  3^n + 1

/-- Theorem: The sum of numbers on a segment after n halvings is 3^n + 1 -/
theorem segment_sum_after_n_halvings (n : ℕ) :
  segmentSum n = 3^n + 1 := by
  sorry

/-- Corollary: The sum after one million halvings -/
theorem sum_after_million_halvings :
  segmentSum 1000000 = 3^1000000 + 1 := by
  sorry

end segment_sum_after_n_halvings_sum_after_million_halvings_l2743_274345


namespace multiplication_fraction_simplification_l2743_274368

theorem multiplication_fraction_simplification :
  8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end multiplication_fraction_simplification_l2743_274368


namespace concert_ticket_revenue_l2743_274334

/-- Calculates the total revenue from concert ticket sales given specific discount conditions --/
theorem concert_ticket_revenue :
  let regular_price : ℚ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let total_customers : ℕ := 50
  let first_discount : ℚ := 0.4
  let second_discount : ℚ := 0.15
  
  let first_group_revenue := first_group_size * (regular_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (regular_price * (1 - second_discount))
  let remaining_customers := total_customers - first_group_size - second_group_size
  let full_price_revenue := remaining_customers * regular_price
  
  let total_revenue := first_group_revenue + second_group_revenue + full_price_revenue
  
  total_revenue = 860 := by sorry

end concert_ticket_revenue_l2743_274334


namespace inequality_solution_set_l2743_274356

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end inequality_solution_set_l2743_274356


namespace sum_of_numbers_leq_threshold_l2743_274325

theorem sum_of_numbers_leq_threshold : 
  let numbers : List ℚ := [8/10, 1/2, 9/10]
  let threshold : ℚ := 4/10
  (numbers.filter (λ x => x ≤ threshold)).sum = 0 := by
sorry

end sum_of_numbers_leq_threshold_l2743_274325


namespace point_P_coordinates_l2743_274321

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def is_on_segment (P M N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N

def vector_eq (P M N : ℝ × ℝ) : Prop :=
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2))

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, is_on_segment P M N ∧ vector_eq P M N → P = (2, 4) :=
by sorry

end point_P_coordinates_l2743_274321


namespace average_of_XYZ_l2743_274340

theorem average_of_XYZ (X Y Z : ℝ) 
  (eq1 : 2001 * Z - 4002 * X = 8008)
  (eq2 : 2001 * Y + 5005 * X = 10010) : 
  (X + Y + Z) / 3 = 0.1667 * X + 3 := by
sorry

end average_of_XYZ_l2743_274340


namespace equality_multiplication_l2743_274383

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end equality_multiplication_l2743_274383


namespace quiz_correct_answers_l2743_274329

theorem quiz_correct_answers (cherry kim nicole : ℕ) 
  (h1 : nicole + 3 = kim)
  (h2 : kim = cherry + 8)
  (h3 : cherry = 17) : 
  nicole = 22 := by
sorry

end quiz_correct_answers_l2743_274329


namespace sum_of_cubes_and_cube_of_sum_l2743_274317

theorem sum_of_cubes_and_cube_of_sum : (5 + 7)^3 + (5^3 + 7^3) = 2196 := by sorry

end sum_of_cubes_and_cube_of_sum_l2743_274317


namespace system_solution_l2743_274319

theorem system_solution (x y : ℝ) (h1 : 2*x + 3*y = 5) (h2 : 3*x + 2*y = 10) : x + y = 3 := by
  sorry

end system_solution_l2743_274319


namespace painted_cells_theorem_l2743_274338

/-- Represents a rectangular grid with alternating painted columns and rows -/
structure PaintedGrid where
  rows : Nat
  cols : Nat
  unpaintedCells : Nat

/-- Checks if the grid dimensions are valid (odd number of rows and columns) -/
def PaintedGrid.isValid (grid : PaintedGrid) : Prop :=
  grid.rows % 2 = 1 ∧ grid.cols % 2 = 1

/-- Calculates the number of painted cells in the grid -/
def PaintedGrid.paintedCells (grid : PaintedGrid) : Nat :=
  grid.rows * grid.cols - grid.unpaintedCells

/-- Theorem: If a valid painted grid has 74 unpainted cells, 
    then the number of painted cells is either 301 or 373 -/
theorem painted_cells_theorem (grid : PaintedGrid) :
  grid.isValid ∧ grid.unpaintedCells = 74 →
  grid.paintedCells = 301 ∨ grid.paintedCells = 373 := by
  sorry

end painted_cells_theorem_l2743_274338


namespace binomial_floor_divisibility_l2743_274381

theorem binomial_floor_divisibility (p n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_p : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) :=
sorry

end binomial_floor_divisibility_l2743_274381


namespace equation_rearrangement_l2743_274309

theorem equation_rearrangement (s P k c n : ℝ) 
  (h : P = s / ((1 + k)^n + c)) 
  (h_pos : s > 0) 
  (h_k_pos : k > -1) 
  (h_P_pos : P > 0) 
  (h_denom_pos : (s/P) - c > 0) : 
  n = (Real.log ((s/P) - c)) / (Real.log (1 + k)) := by
sorry

end equation_rearrangement_l2743_274309


namespace rectangle_area_l2743_274335

theorem rectangle_area (perimeter : ℝ) (length_width_ratio : ℝ) : 
  perimeter = 60 → length_width_ratio = 1.5 → 
  let width := perimeter / (2 * (1 + length_width_ratio))
  let length := length_width_ratio * width
  let area := length * width
  area = 216 := by
sorry

end rectangle_area_l2743_274335


namespace binary_to_decimal_11110_l2743_274386

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * 2^position

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 1, 1, 0]

/-- The decimal representation of the binary number -/
def decimalRepresentation : Nat :=
  (List.enumFrom 0 binaryNumber).map (fun (pos, digit) => binaryToDecimal digit pos) |>.sum

/-- Theorem stating that the decimal representation of "11110" is 30 -/
theorem binary_to_decimal_11110 :
  decimalRepresentation = 30 := by sorry

end binary_to_decimal_11110_l2743_274386


namespace omelet_preparation_time_l2743_274361

/-- Calculates the total time spent preparing and cooking omelets -/
def total_omelet_time (pepper_time onion_time mushroom_time tomato_time cheese_time cook_time : ℕ)
                      (num_peppers num_onions num_mushrooms num_tomatoes num_omelets : ℕ) : ℕ :=
  pepper_time * num_peppers +
  onion_time * num_onions +
  mushroom_time * num_mushrooms +
  tomato_time * num_tomatoes +
  cheese_time * num_omelets +
  cook_time * num_omelets

/-- Proves that the total time spent preparing and cooking 10 omelets is 140 minutes -/
theorem omelet_preparation_time :
  total_omelet_time 3 4 2 3 1 6 8 4 6 6 10 = 140 := by
  sorry

end omelet_preparation_time_l2743_274361


namespace hyperbola_eccentricity_l2743_274350

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 4 / 3) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end hyperbola_eccentricity_l2743_274350


namespace largest_z_value_l2743_274314

theorem largest_z_value (x y z : ℝ) : 
  x + y + z = 5 → 
  x * y + y * z + x * z = 3 → 
  z ≤ 13/3 :=
by sorry

end largest_z_value_l2743_274314


namespace expression_simplification_l2743_274374

theorem expression_simplification (x : ℝ) (h : x = 4) :
  (x^2 - 4*x + 4) / (x^2 - 1) / (1 - 3 / (x + 1)) = 2/3 := by
  sorry

end expression_simplification_l2743_274374


namespace f_of_2_eq_6_l2743_274328

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- Theorem: f(2) = 6 -/
theorem f_of_2_eq_6 : f 2 = 6 := by
  sorry

end f_of_2_eq_6_l2743_274328


namespace candle_box_cost_l2743_274360

/-- The cost of a box of candles --/
def box_cost : ℕ := 5

/-- Kerry's age --/
def kerry_age : ℕ := 8

/-- Number of cakes Kerry wants --/
def num_cakes : ℕ := 3

/-- Number of candles in a box --/
def candles_per_box : ℕ := 12

/-- Total number of candles needed --/
def total_candles : ℕ := num_cakes * kerry_age

theorem candle_box_cost : box_cost = 5 := by
  sorry

end candle_box_cost_l2743_274360


namespace spade_or_club_probability_l2743_274308

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- The probability of drawing a card of a specific type from a deck -/
def draw_probability (deck : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / deck.total_cards

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Theorem: The probability of drawing either a ♠ or a ♣ from a standard 52-card deck is 1/2 -/
theorem spade_or_club_probability :
  draw_probability standard_deck (2 * standard_deck.ranks) = 1 / 2 := by
  sorry

end spade_or_club_probability_l2743_274308


namespace trig_identity_l2743_274322

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l2743_274322


namespace positive_solution_x_l2743_274347

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y + 3 * x + 2 * y = 12)
  (eq2 : y * z + 5 * y + 3 * z = 15)
  (eq3 : x * z + 5 * x + 4 * z = 40)
  (x_pos : x > 0) : x = 4 := by
sorry

end positive_solution_x_l2743_274347


namespace dinas_crayons_l2743_274373

theorem dinas_crayons (wanda_crayons : ℕ) (total_crayons : ℕ) (dina_crayons : ℕ) :
  wanda_crayons = 62 →
  total_crayons = 116 →
  total_crayons = wanda_crayons + dina_crayons + (dina_crayons - 2) →
  dina_crayons = 28 := by
  sorry

end dinas_crayons_l2743_274373


namespace cubic_function_extrema_difference_l2743_274387

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 - 3*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 3

theorem cubic_function_extrema_difference (a b : ℝ) :
  f' a (-1) = 0 →
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧ 
    f a b x_max - f a b x_min = 4 := by
  sorry

end cubic_function_extrema_difference_l2743_274387


namespace workshop_average_age_l2743_274330

theorem workshop_average_age 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (num_elderly : ℕ) (avg_age_elderly : ℝ)
  (h1 : num_females = 8)
  (h2 : avg_age_females = 34)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 32)
  (h5 : num_elderly = 5)
  (h6 : avg_age_elderly = 60) :
  let total_people := num_females + num_males + num_elderly
  let total_age := num_females * avg_age_females + num_males * avg_age_males + num_elderly * avg_age_elderly
  total_age / total_people = 38.24 := by
sorry

end workshop_average_age_l2743_274330


namespace equation_not_equivalent_to_expression_with_unknown_l2743_274332

-- Define what an expression is
def Expression : Type := Unit

-- Define what an unknown is
def Unknown : Type := Unit

-- Define what an equation is
def Equation : Type := Unit

-- Define a property for expressions that contain unknowns
def contains_unknown (e : Expression) : Prop := sorry

-- Define the property that an equation contains unknowns
axiom equation_contains_unknown : ∀ (eq : Equation), ∃ (u : Unknown), contains_unknown eq

-- Theorem to prove
theorem equation_not_equivalent_to_expression_with_unknown : 
  ¬(∀ (e : Expression), contains_unknown e → ∃ (eq : Equation), e = eq) :=
sorry

end equation_not_equivalent_to_expression_with_unknown_l2743_274332


namespace corn_planting_bags_used_l2743_274326

/-- Represents the corn planting scenario with given conditions -/
structure CornPlanting where
  kids : ℕ
  earsPerRow : ℕ
  seedsPerEar : ℕ
  seedsPerBag : ℕ
  payPerRow : ℚ
  dinnerCost : ℚ

/-- Calculates the number of bags of corn seeds used by each kid -/
def bagsUsedPerKid (cp : CornPlanting) : ℚ :=
  let totalEarned := 2 * cp.dinnerCost
  let rowsPlanted := totalEarned / cp.payPerRow
  let seedsPerRow := cp.earsPerRow * cp.seedsPerEar
  let totalSeeds := rowsPlanted * seedsPerRow
  totalSeeds / cp.seedsPerBag

/-- Theorem stating that each kid used 140 bags of corn seeds -/
theorem corn_planting_bags_used
  (cp : CornPlanting)
  (h1 : cp.kids = 4)
  (h2 : cp.earsPerRow = 70)
  (h3 : cp.seedsPerEar = 2)
  (h4 : cp.seedsPerBag = 48)
  (h5 : cp.payPerRow = 3/2)
  (h6 : cp.dinnerCost = 36) :
  bagsUsedPerKid cp = 140 := by
  sorry

end corn_planting_bags_used_l2743_274326


namespace point_on_decreasing_linear_function_l2743_274354

/-- A linear function that decreases as x increases -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 2) + 4

/-- The slope of the linear function is negative -/
def isDecreasing (k : ℝ) : Prop :=
  k < 0

/-- The point (3, -1) lies on the graph of the function -/
def pointOnGraph (k : ℝ) : Prop :=
  decreasingLinearFunction k 3 = -1

/-- Theorem: If the linear function y = k(x-2) + 4 is decreasing,
    then the point (3, -1) lies on its graph -/
theorem point_on_decreasing_linear_function :
  ∀ k : ℝ, isDecreasing k → pointOnGraph k :=
by
  sorry

end point_on_decreasing_linear_function_l2743_274354


namespace symmetry_implies_constant_l2743_274399

/-- A bivariate real-coefficient polynomial -/
structure BivariatePolynomial where
  (p : ℝ → ℝ → ℝ)

/-- The property that P(X, Y) = P(X+Y, X-Y) for all real X and Y -/
def has_symmetry (P : BivariatePolynomial) : Prop :=
  ∀ (X Y : ℝ), P.p X Y = P.p (X + Y) (X - Y)

/-- Main theorem: If P has the symmetry property, then it is constant -/
theorem symmetry_implies_constant (P : BivariatePolynomial) 
  (h : has_symmetry P) : 
  ∃ (c : ℝ), ∀ (X Y : ℝ), P.p X Y = c := by
  sorry

end symmetry_implies_constant_l2743_274399


namespace impossible_arrangement_l2743_274348

theorem impossible_arrangement :
  ¬ ∃ (grid : Matrix (Fin 6) (Fin 7) ℕ),
    (∀ i j, grid i j ∈ Set.range (fun n => n + 1) ∩ Set.Icc 1 42) ∧
    (∀ i j, ∃! p, grid p j = grid i j) ∧
    (∀ i j, Even (grid i j + grid (i + 1) j)) :=
by sorry

end impossible_arrangement_l2743_274348


namespace infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l2743_274384

theorem infinitely_many_n_factorial_divisible_by_n_cubed_minus_one :
  {n : ℕ+ | (n.val.factorial : ℤ) % (n.val ^ 3 - 1) = 0}.Infinite :=
sorry

end infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l2743_274384


namespace arithmetic_progression_with_special_properties_l2743_274396

/-- A perfect power is a number of the form n^k where n and k are both natural numbers ≥ 2 -/
def is_perfect_power (x : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ k ≥ 2 ∧ x = n^k

/-- An arithmetic progression is a sequence where the difference between successive terms is constant -/
def is_arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i, s i = a + i * d

theorem arithmetic_progression_with_special_properties :
  ∃ (s : ℕ → ℕ),
    is_arithmetic_progression s ∧
    (∀ i ∈ Finset.range 2016, ¬is_perfect_power (s i)) ∧
    is_perfect_power (Finset.prod (Finset.range 2016) s) :=
sorry

end arithmetic_progression_with_special_properties_l2743_274396


namespace marbles_per_customer_l2743_274320

theorem marbles_per_customer 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 400) 
  (h2 : num_customers = 20) 
  (h3 : remaining_marbles = 100) :
  (initial_marbles - remaining_marbles) / num_customers = 15 :=
by sorry

end marbles_per_customer_l2743_274320


namespace line_slope_problem_l2743_274327

/-- Given a line passing through points (-1, -4) and (5, k) with slope k, prove that k = 4/5 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end line_slope_problem_l2743_274327


namespace solution_volume_l2743_274304

/-- Proves that the total volume of a solution is 10 liters, given that it contains 2.5 liters of pure acid and has a 25% concentration. -/
theorem solution_volume (acid_volume : ℝ) (concentration : ℝ) :
  acid_volume = 2.5 →
  concentration = 0.25 →
  acid_volume / concentration = 10 :=
by
  sorry

end solution_volume_l2743_274304


namespace flour_amount_proof_l2743_274394

/-- The amount of flour in the first combination -/
def flour_amount : ℝ := 17.78

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

theorem flour_amount_proof :
  (40 * cost_per_pound + flour_amount * cost_per_pound = total_cost) ∧
  (30 * cost_per_pound + 25 * cost_per_pound = total_cost) →
  flour_amount = 17.78 := by sorry

end flour_amount_proof_l2743_274394


namespace solution_to_system_l2743_274311

theorem solution_to_system (x y : ℝ) : 
  x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2 →
  (1 / x - 1 / (2 * y) = 2 * y^4 - 2 * x^4) ∧
  (1 / x + 1 / (2 * y) = (3 * x^2 + y^2) * (x^2 + 3 * y^2)) := by
sorry

end solution_to_system_l2743_274311


namespace derivative_even_implies_a_zero_l2743_274378

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem derivative_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f_derivative a x = f_derivative a (-x)) →
  a = 0 := by sorry

end derivative_even_implies_a_zero_l2743_274378


namespace line_equation_from_triangle_area_l2743_274349

/-- Given a line passing through (a, 0) and intersecting the y-axis in the first quadrant,
    forming a triangular region with area T, prove that the equation of this line is
    2Tx + a²y - 2aT = 0 -/
theorem line_equation_from_triangle_area (a T : ℝ) (h_a : a ≠ 0) (h_T : T > 0) :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, y = m * x + b → (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0)) ∧
    (1/2 * a * b = T) ∧
    (∀ x y : ℝ, 2 * T * x + a^2 * y - 2 * a * T = 0 ↔ y = m * x + b) :=
by sorry

end line_equation_from_triangle_area_l2743_274349


namespace contrapositive_equivalence_l2743_274398

theorem contrapositive_equivalence (a b : ℝ) :
  (¬((a - b) * (a + b) = 0) → ¬(a - b = 0)) ↔
  ((a - b = 0) → ((a - b) * (a + b) = 0)) :=
by sorry

end contrapositive_equivalence_l2743_274398


namespace pells_equation_unique_solution_l2743_274359

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℕ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

-- The main theorem
theorem pells_equation_unique_solution (x₀ y₀ x y : ℕ) :
  fundamental_solution x₀ y₀ →
  x^2 - 2003 * y^2 = 1 →
  x > 0 →
  y > 0 →
  all_prime_factors_divide x x₀ →
  x = x₀ ∧ y = y₀ :=
sorry

end pells_equation_unique_solution_l2743_274359


namespace xavier_probability_l2743_274337

theorem xavier_probability (p_x p_y p_z : ℝ) 
  (h1 : p_y = 1/2)
  (h2 : p_z = 5/8)
  (h3 : p_x * p_y * (1 - p_z) = 0.0375) :
  p_x = 0.2 := by
sorry

end xavier_probability_l2743_274337


namespace angle4_value_l2743_274323

-- Define the angles
def angle1 : ℝ := sorry
def angle2 : ℝ := sorry
def angle3 : ℝ := sorry
def angle4 : ℝ := sorry
def angleA : ℝ := 80
def angleB : ℝ := 50

-- State the theorem
theorem angle4_value :
  (angle1 + angle2 = 180) →
  (angle3 = angle4) →
  (angle1 + angleA + angleB = 180) →
  (angle2 + angle3 + angle4 = 180) →
  angle4 = 25 := by
  sorry

end angle4_value_l2743_274323


namespace min_trig_expression_l2743_274339

theorem min_trig_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.sin x + 1 / Real.sin x)^3 + (Real.cos x + 1 / Real.cos x)^3 ≥ 729 * Real.sqrt 2 / 16 := by
  sorry

end min_trig_expression_l2743_274339


namespace min_green_beads_l2743_274343

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  total_sum : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads
    satisfying the given conditions is 27. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) :
  n.green ≥ 27 := by sorry

end min_green_beads_l2743_274343


namespace intersection_of_A_and_B_l2743_274336

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l2743_274336


namespace two_digit_reverse_sum_l2743_274315

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

theorem two_digit_reverse_sum (n : ℕ) :
  is_two_digit n →
  (n : ℤ) - (reverse_digits n : ℤ) = 7 * ((n / 10 : ℤ) + (n % 10 : ℤ)) →
  (n : ℕ) + reverse_digits n = 99 := by
  sorry

end two_digit_reverse_sum_l2743_274315


namespace round_trip_average_speed_l2743_274377

/-- The average speed of a round trip where:
    - The total distance is 2m meters (m meters each way)
    - The northward journey is at 3 minutes per mile
    - The southward journey is at 3 miles per minute
    - 1 mile = 1609.34 meters
-/
theorem round_trip_average_speed (m : ℝ) :
  let meters_per_mile : ℝ := 1609.34
  let north_speed : ℝ := 1 / 3 -- miles per minute
  let south_speed : ℝ := 3 -- miles per minute
  let total_distance : ℝ := 2 * m / meters_per_mile -- in miles
  let north_time : ℝ := m / (meters_per_mile * north_speed) -- in minutes
  let south_time : ℝ := m / (meters_per_mile * south_speed) -- in minutes
  let total_time : ℝ := north_time + south_time -- in minutes
  let average_speed : ℝ := total_distance / (total_time / 60) -- in miles per hour
  average_speed = 60 := by
sorry

end round_trip_average_speed_l2743_274377


namespace elise_savings_elise_savings_proof_l2743_274366

/-- Proves that Elise saved $13 from her allowance -/
theorem elise_savings : ℕ → Prop :=
  fun (saved : ℕ) =>
    let initial : ℕ := 8
    let comic_cost : ℕ := 2
    let puzzle_cost : ℕ := 18
    let final : ℕ := 1
    initial + saved - (comic_cost + puzzle_cost) = final →
    saved = 13

/-- The proof of the theorem -/
theorem elise_savings_proof : elise_savings 13 := by
  sorry

end elise_savings_elise_savings_proof_l2743_274366


namespace kyle_corn_purchase_l2743_274342

-- Define the problem parameters
def total_pounds : ℝ := 30
def total_cost : ℝ := 22.50
def corn_price : ℝ := 1.05
def beans_price : ℝ := 0.55

-- Define the theorem
theorem kyle_corn_purchase :
  ∃ (corn beans : ℝ),
    corn + beans = total_pounds ∧
    corn_price * corn + beans_price * beans = total_cost ∧
    corn = 12 := by
  sorry

end kyle_corn_purchase_l2743_274342


namespace ursula_change_l2743_274397

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change : 
  let hot_dog_price : ℚ := 3/2  -- $1.50 as a rational number
  let salad_price : ℚ := 5/2    -- $2.50 as a rational number
  let hot_dog_count : ℕ := 5
  let salad_count : ℕ := 3
  let bill_value : ℕ := 10
  let bill_count : ℕ := 2
  
  let total_cost : ℚ := hot_dog_price * hot_dog_count + salad_price * salad_count
  let total_paid : ℕ := bill_value * bill_count
  
  (total_paid : ℚ) - total_cost = 5
  := by sorry

end ursula_change_l2743_274397


namespace arithmetic_equalities_l2743_274316

theorem arithmetic_equalities : 
  (-20 + (-14) - (-18) - 13 = -29) ∧ 
  ((-2) * 3 + (-5) - 4 / (-1/2) = -3) ∧ 
  ((-3/8 - 1/6 + 3/4) * (-24) = -5) ∧ 
  (-81 / (9/4) * |(-4/9)| - (-3)^3 / 27 = -15) := by sorry

end arithmetic_equalities_l2743_274316
