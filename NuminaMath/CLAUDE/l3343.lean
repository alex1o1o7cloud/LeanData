import Mathlib

namespace exam_students_count_l3343_334392

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T = 80 * N →
    (T - 350) / (N - 5 : ℝ) = 90 →
    N = 10 :=
by
  sorry

end exam_students_count_l3343_334392


namespace circumference_difference_l3343_334379

/-- Given two circles A and B with areas and π as specified, 
    prove that the difference between their circumferences is 6.2 cm -/
theorem circumference_difference (π : ℝ) (area_A area_B : ℝ) :
  π = 3.1 →
  area_A = 198.4 →
  area_B = 251.1 →
  let radius_A := Real.sqrt (area_A / π)
  let radius_B := Real.sqrt (area_B / π)
  let circumference_A := 2 * π * radius_A
  let circumference_B := 2 * π * radius_B
  circumference_B - circumference_A = 6.2 := by
  sorry

end circumference_difference_l3343_334379


namespace mollys_brothers_children_l3343_334390

/-- The number of children each of Molly's brothers has -/
def children_per_brother : ℕ := 2

theorem mollys_brothers_children :
  let cost_per_package : ℕ := 5
  let num_parents : ℕ := 2
  let num_brothers : ℕ := 3
  let total_cost : ℕ := 70
  let immediate_family : ℕ := num_parents + num_brothers + num_brothers -- includes spouses
  (cost_per_package * (immediate_family + num_brothers * children_per_brother) = total_cost) ∧
  (children_per_brother > 0) :=
by sorry

end mollys_brothers_children_l3343_334390


namespace square_perimeter_from_rectangle_perimeter_l3343_334377

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 28 inches, then the perimeter of the square is 44.8 inches. -/
theorem square_perimeter_from_rectangle_perimeter : 
  ∀ (s : ℝ), 
  s > 0 → -- side length of the square is positive
  (5 * s / 2 = 28) → -- perimeter of each rectangle is 28 inches
  (4 * s = 44.8) -- perimeter of the square is 44.8 inches
:= by sorry

end square_perimeter_from_rectangle_perimeter_l3343_334377


namespace hyperbola_equation_l3343_334307

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.is_asymptote (h : Hyperbola) (m : ℝ) : Prop :=
  m = h.b / h.a ∨ m = -h.b / h.a

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) :
  h.contains 3 (Real.sqrt 2) ∧
  h.is_asymptote (1/3) ∧
  h.is_asymptote (-1/3) →
  h.a^2 = 153 ∧ h.b^2 = 17 :=
sorry

end hyperbola_equation_l3343_334307


namespace student_sample_size_l3343_334370

theorem student_sample_size :
  ∀ (T : ℝ) (freshmen sophomores juniors seniors : ℝ),
    -- All students are either freshmen, sophomores, juniors, or seniors
    T = freshmen + sophomores + juniors + seniors →
    -- 27% are juniors
    juniors = 0.27 * T →
    -- 75% are not sophomores (which means 25% are sophomores)
    sophomores = 0.25 * T →
    -- There are 160 seniors
    seniors = 160 →
    -- There are 24 more freshmen than sophomores
    freshmen = sophomores + 24 →
    -- Prove that the total number of students is 800
    T = 800 := by
  sorry

end student_sample_size_l3343_334370


namespace set_operation_equality_l3343_334331

theorem set_operation_equality (M N P : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → P = {3, 5} → 
  (M ∩ N) ∪ P = {2, 3, 5} := by
  sorry

end set_operation_equality_l3343_334331


namespace henry_tournament_points_l3343_334395

/-- A structure representing a tic-tac-toe tournament result -/
structure TournamentResult where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given tournament result -/
def calculate_points (result : TournamentResult) : ℕ :=
  result.win_points * result.wins +
  result.loss_points * result.losses +
  result.draw_points * result.draws

/-- Theorem: Henry's tournament result yields 44 points -/
theorem henry_tournament_points :
  let henry_result : TournamentResult := {
    win_points := 5,
    loss_points := 2,
    draw_points := 3,
    wins := 2,
    losses := 2,
    draws := 10
  }
  calculate_points henry_result = 44 := by sorry

end henry_tournament_points_l3343_334395


namespace arithmetic_sequence_common_difference_l3343_334335

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  ∃ d : ℚ, d = 3/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l3343_334335


namespace mary_biking_time_l3343_334373

def total_away_time : ℕ := 570 -- 9.5 hours in minutes
def class_time : ℕ := 45
def num_classes : ℕ := 7
def lunch_time : ℕ := 40
def additional_time : ℕ := 105 -- 1 hour 45 minutes in minutes

def total_school_time : ℕ := class_time * num_classes + lunch_time + additional_time

theorem mary_biking_time :
  total_away_time - total_school_time = 110 :=
sorry

end mary_biking_time_l3343_334373


namespace platform_length_platform_length_proof_l3343_334340

/-- Calculates the length of a platform given the length of a train, its speed, and the time it takes to cross the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 160) 
  (h2 : train_speed_kmph = 72) 
  (h3 : crossing_time = 25) : ℝ :=
let train_speed_mps := train_speed_kmph * (1000 / 3600)
let total_distance := train_speed_mps * crossing_time
let platform_length := total_distance - train_length
340

theorem platform_length_proof : platform_length 160 72 25 rfl rfl rfl = 340 := by
  sorry

end platform_length_platform_length_proof_l3343_334340


namespace rectangular_prism_base_area_l3343_334304

theorem rectangular_prism_base_area :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a % 5 = 0 ∧ b % 5 = 0 ∧ a * b = 450 := by
  sorry

end rectangular_prism_base_area_l3343_334304


namespace books_remaining_pauls_remaining_books_l3343_334396

theorem books_remaining (initial books_given books_sold : ℕ) :
  initial ≥ books_given + books_sold →
  initial - (books_given + books_sold) = initial - books_given - books_sold :=
by
  sorry

theorem pauls_remaining_books :
  134 - (39 + 27) = 68 :=
by
  sorry

end books_remaining_pauls_remaining_books_l3343_334396


namespace smallest_n_sum_all_digits_same_l3343_334389

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number has all digits the same -/
def all_digits_same (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 111

/-- The smallest n such that sum_first_n(n) is a three-digit number with all digits the same -/
theorem smallest_n_sum_all_digits_same :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(all_digits_same (sum_first_n m))) ∧
    (all_digits_same (sum_first_n n)) ∧
    n = 36 := by sorry

end smallest_n_sum_all_digits_same_l3343_334389


namespace coprime_27x_plus_4_and_18x_plus_3_l3343_334371

theorem coprime_27x_plus_4_and_18x_plus_3 (x : ℕ) : Nat.gcd (27 * x + 4) (18 * x + 3) = 1 := by
  sorry

end coprime_27x_plus_4_and_18x_plus_3_l3343_334371


namespace quadrilateral_and_triangle_theorem_l3343_334343

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Intersection point of two lines -/
def intersect (l₁ l₂ : Line) : Point := sorry

/-- Line passing through two points -/
def line_through (p q : Point) : Line := sorry

/-- Point where a line parallel to a given direction through a point intersects another line -/
def parallel_intersect (p : Point) (l : Line) (dir : ℝ) : Point := sorry

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Check if two triangles are perspective -/
def perspective (t₁ t₂ : Point × Point × Point) : Prop := sorry

/-- The main theorem -/
theorem quadrilateral_and_triangle_theorem 
  (A B C D E F : Point) 
  (dir₁ dir₂ : ℝ) :
  let EF := line_through E F
  let A₁ := parallel_intersect A EF dir₁
  let B₁ := parallel_intersect B EF dir₁
  let C₁ := parallel_intersect C EF dir₁
  let D₁ := parallel_intersect D EF dir₁
  let desargues_line := sorry -- Definition of Desargues line
  let A' := parallel_intersect A desargues_line dir₂
  let B' := parallel_intersect B desargues_line dir₂
  let C' := parallel_intersect C desargues_line dir₂
  let A₁' := parallel_intersect A₁ desargues_line dir₂
  let B₁' := parallel_intersect B₁ desargues_line dir₂
  let C₁' := parallel_intersect C₁ desargues_line dir₂
  collinear E F (intersect (line_through A C) (line_through B D)) ∧
  perspective (A, B, C) (A₁, B₁, C₁) →
  (1 / distance A A₁ + 1 / distance C C₁ = 1 / distance B B₁ + 1 / distance D D₁) ∧
  (1 / distance A A' + 1 / distance B B' + 1 / distance C C' = 
   1 / distance A₁ A₁' + 1 / distance B₁ B₁' + 1 / distance C₁ C₁') := by
  sorry

end quadrilateral_and_triangle_theorem_l3343_334343


namespace chinese_english_total_score_l3343_334320

theorem chinese_english_total_score 
  (average_score : ℝ) 
  (math_score : ℝ) 
  (num_subjects : ℕ) 
  (h1 : average_score = 97) 
  (h2 : math_score = 100) 
  (h3 : num_subjects = 3) :
  average_score * num_subjects - math_score = 191 :=
by sorry

end chinese_english_total_score_l3343_334320


namespace pie_slices_l3343_334330

/-- Proves that if 3/4 of a pie is given away and 2 slices are left, then the pie was sliced into 8 pieces. -/
theorem pie_slices (total_slices : ℕ) : 
  (3 : ℚ) / 4 * total_slices + 2 = total_slices → total_slices = 8 := by
  sorry

#check pie_slices

end pie_slices_l3343_334330


namespace ceiling_floor_sum_l3343_334354

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end ceiling_floor_sum_l3343_334354


namespace town_population_growth_l3343_334315

theorem town_population_growth (r : ℕ) (h1 : r^3 + 200 = (r + 1)^3 + 27) 
  (h2 : (r + 1)^3 + 300 = (r + 1)^3) : 
  (((r + 1)^3 - r^3) * 100 : ℚ) / r^3 = 72 := by
  sorry

end town_population_growth_l3343_334315


namespace sqrt_21_minus_1_bounds_l3343_334391

theorem sqrt_21_minus_1_bounds : 3 < Real.sqrt 21 - 1 ∧ Real.sqrt 21 - 1 < 4 := by
  sorry

end sqrt_21_minus_1_bounds_l3343_334391


namespace five_graduates_three_companies_l3343_334314

/-- The number of ways to assign n graduates to k companies, with each company hiring at least one person -/
def assignGraduates (n k : ℕ) : ℕ :=
  sorry

theorem five_graduates_three_companies : 
  assignGraduates 5 3 = 150 := by sorry

end five_graduates_three_companies_l3343_334314


namespace garden_fencing_cost_l3343_334372

/-- The cost of fencing a rectangular garden -/
theorem garden_fencing_cost
  (garden_width : ℝ)
  (playground_length playground_width : ℝ)
  (fencing_price : ℝ)
  (h1 : garden_width = 12)
  (h2 : playground_length = 16)
  (h3 : playground_width = 12)
  (h4 : fencing_price = 15)
  (h5 : garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width) :
  2 * (garden_width + (playground_length * playground_width / garden_width)) * fencing_price = 840 :=
by sorry

end garden_fencing_cost_l3343_334372


namespace zekes_estimate_l3343_334317

theorem zekes_estimate (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) (h : x > 2*y) :
  (x + k) - 2*(y + k) < x - 2*y := by
sorry

end zekes_estimate_l3343_334317


namespace black_cows_exceeding_half_l3343_334382

theorem black_cows_exceeding_half (total_cows : ℕ) (non_black_cows : ℕ) : 
  total_cows = 18 → non_black_cows = 4 → 
  (total_cows - non_black_cows) - (total_cows / 2) = 5 := by
  sorry

end black_cows_exceeding_half_l3343_334382


namespace new_speed_calculation_l3343_334322

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 252)
  (h2 : original_time = 6)
  (h3 : time_factor = 3/2) :
  let new_time := original_time * time_factor
  let new_speed := distance / new_time
  new_speed = 28 := by sorry

end new_speed_calculation_l3343_334322


namespace total_rainfall_2012_l3343_334376

-- Define the average monthly rainfall for each year
def rainfall_2010 : ℝ := 37.2
def rainfall_2011 : ℝ := rainfall_2010 + 3.5
def rainfall_2012 : ℝ := rainfall_2011 - 1.2

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem total_rainfall_2012 : 
  rainfall_2012 * months_in_year = 474 := by sorry

end total_rainfall_2012_l3343_334376


namespace hyperbola_C_different_asymptote_l3343_334336

-- Define the hyperbolas
def hyperbola_A (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1
def hyperbola_B (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote
def is_asymptote (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y → (2 * x = 3 * y ∨ 2 * x = -3 * y)

-- Theorem statement
theorem hyperbola_C_different_asymptote :
  ¬(is_asymptote hyperbola_C) ∧
  (is_asymptote hyperbola_A) ∧
  (is_asymptote hyperbola_B) ∧
  (is_asymptote hyperbola_D) := by
  sorry


end hyperbola_C_different_asymptote_l3343_334336


namespace arithmetic_sequence_constant_ratio_l3343_334374

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- Theorem: If the ratio S_{4n}/S_n is constant for all positive n,
    then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_constant_ratio
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (4*n) / S a n = c) :
  a = 5/2 := by
  sorry

end arithmetic_sequence_constant_ratio_l3343_334374


namespace solve_necklace_cost_l3343_334394

def necklace_cost_problem (necklace_cost book_cost total_cost spending_limit overspend : ℚ) : Prop :=
  book_cost = necklace_cost + 5 ∧
  spending_limit = 70 ∧
  overspend = 3 ∧
  total_cost = necklace_cost + book_cost ∧
  total_cost = spending_limit + overspend ∧
  necklace_cost = 34

theorem solve_necklace_cost :
  ∃ (necklace_cost book_cost total_cost spending_limit overspend : ℚ),
    necklace_cost_problem necklace_cost book_cost total_cost spending_limit overspend :=
by sorry

end solve_necklace_cost_l3343_334394


namespace binomial_coefficient_equality_l3343_334387

theorem binomial_coefficient_equality (n : ℕ) : 
  Nat.choose 18 n = Nat.choose 18 2 → n = 2 ∨ n = 16 := by
  sorry

end binomial_coefficient_equality_l3343_334387


namespace determine_fourth_player_wins_l3343_334384

/-- Represents a player in the chess tournament -/
structure Player where
  wins : Nat
  losses : Nat

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Fin 4 → Player
  total_games : Nat

/-- The theorem states that given the wins and losses of three players in a four-player
    round-robin chess tournament, we can determine the number of wins for the fourth player. -/
theorem determine_fourth_player_wins (t : ChessTournament) 
  (h1 : t.players 0 = { wins := 5, losses := 3 })
  (h2 : t.players 1 = { wins := 4, losses := 4 })
  (h3 : t.players 2 = { wins := 2, losses := 6 })
  (h_total : t.total_games = 16)
  (h_balance : ∀ i, (t.players i).wins + (t.players i).losses = 8) :
  (t.players 3).wins = 5 := by
  sorry

end determine_fourth_player_wins_l3343_334384


namespace white_balls_count_l3343_334383

/-- Given a bag of balls with the following properties:
  * The total number of balls is 40
  * The probability of drawing a red ball is 0.15
  * The probability of drawing a black ball is 0.45
  * The remaining balls are white
  
  This theorem proves that the number of white balls in the bag is 16. -/
theorem white_balls_count (total : ℕ) (p_red p_black : ℝ) :
  total = 40 →
  p_red = 0.15 →
  p_black = 0.45 →
  (total : ℝ) * (1 - p_red - p_black) = 16 := by
  sorry

end white_balls_count_l3343_334383


namespace rectangle_area_problem_l3343_334339

theorem rectangle_area_problem (A : ℝ) : 
  let square_side : ℝ := 12
  let new_horizontal : ℝ := square_side + 3
  let new_vertical : ℝ := square_side - A
  let new_area : ℝ := 120
  new_horizontal * new_vertical = new_area → A = 4 :=
by
  sorry

end rectangle_area_problem_l3343_334339


namespace complex_absolute_value_l3343_334368

theorem complex_absolute_value (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_absolute_value_l3343_334368


namespace min_value_expression_l3343_334303

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a^2 * b^2) / (a * b) ≥ 2 := by
  sorry

end min_value_expression_l3343_334303


namespace power_of_two_equality_l3343_334324

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^x → x = 33 := by
  sorry

end power_of_two_equality_l3343_334324


namespace problem_statement_l3343_334380

theorem problem_statement (X Y Z : ℕ+) 
  (h_coprime : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_equation : X.val * Real.log 3 / Real.log 100 + Y.val * Real.log 4 / Real.log 100 = (Z.val : ℝ)^2) :
  X.val + Y.val + Z.val = 4 := by
sorry

end problem_statement_l3343_334380


namespace maxwells_walking_speed_l3343_334347

/-- Proves that Maxwell's walking speed is 24 km/h given the problem conditions -/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (brads_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 72) 
  (h2 : brads_speed = 12) 
  (h3 : maxwell_distance = 24) : 
  maxwell_distance / (maxwell_distance / brads_speed) = 24 := by
  sorry

end maxwells_walking_speed_l3343_334347


namespace smallest_sum_of_primes_with_digit_conditions_l3343_334356

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def digit_count (n : ℕ) (d : ℕ) : ℕ := 
  (n.digits 10).count d

def satisfies_conditions (primes : List ℕ) : Prop :=
  primes.length = 5 ∧
  (∀ p ∈ primes, is_prime p) ∧
  (primes.map (digit_count · 3)).sum = 2 ∧
  (primes.map (digit_count · 7)).sum = 2 ∧
  (primes.map (digit_count · 8)).sum = 2 ∧
  (∀ d ∈ [1, 2, 4, 5, 6, 9], (primes.map (digit_count · d)).sum = 1)

theorem smallest_sum_of_primes_with_digit_conditions :
  ∃ (primes : List ℕ),
    satisfies_conditions primes ∧
    primes.sum = 2063 ∧
    (∀ other_primes : List ℕ, satisfies_conditions other_primes → other_primes.sum ≥ 2063) :=
by sorry

end smallest_sum_of_primes_with_digit_conditions_l3343_334356


namespace sandy_comic_books_l3343_334386

theorem sandy_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end sandy_comic_books_l3343_334386


namespace subset_sum_implies_total_sum_l3343_334332

theorem subset_sum_implies_total_sum (a₁ a₂ a₃ : ℝ) :
  (a₁ + a₂ + a₁ + a₃ + a₂ + a₃ + (a₁ + a₂) + (a₁ + a₃) + (a₂ + a₃) = 12) →
  (a₁ + a₂ + a₃ = 4) := by
  sorry

end subset_sum_implies_total_sum_l3343_334332


namespace handshake_count_l3343_334301

theorem handshake_count (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end handshake_count_l3343_334301


namespace coordinates_of_P_wrt_x_axis_l3343_334365

/-- Given a point P in the Cartesian coordinate system, this function
    returns its coordinates with respect to the x-axis. -/
def coordinates_wrt_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem stating that the coordinates of P(-2, 3) with respect to the x-axis are (-2, -3). -/
theorem coordinates_of_P_wrt_x_axis :
  coordinates_wrt_x_axis (-2, 3) = (-2, -3) := by
  sorry

end coordinates_of_P_wrt_x_axis_l3343_334365


namespace solution_difference_l3343_334381

theorem solution_difference (r s : ℝ) : 
  (((5 * r - 15) / (r^2 + 3*r - 18) = r + 3) ∧
   ((5 * s - 15) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧ (r > s)) →
  r - s = 13 := by
sorry

end solution_difference_l3343_334381


namespace quadratic_vertex_form_l3343_334328

theorem quadratic_vertex_form (a b c : ℝ) (h k : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  (a = 3 ∧ b = 9 ∧ c = 20) →
  h = -1.5 := by sorry

end quadratic_vertex_form_l3343_334328


namespace march_text_messages_l3343_334366

/-- Represents the number of text messages sent in the nth month -/
def T (n : ℕ) : ℕ := n^3 - n^2 + n

/-- Theorem stating that the number of text messages in the 5th month (March) is 105 -/
theorem march_text_messages : T 5 = 105 := by
  sorry

end march_text_messages_l3343_334366


namespace nested_fraction_evaluation_l3343_334360

theorem nested_fraction_evaluation :
  (2 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 15 := by
  sorry

end nested_fraction_evaluation_l3343_334360


namespace train_length_calculation_l3343_334375

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 5.999520038396929 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end train_length_calculation_l3343_334375


namespace binomial_coefficient_seven_two_l3343_334363

theorem binomial_coefficient_seven_two : 
  Nat.choose 7 2 = 21 := by sorry

end binomial_coefficient_seven_two_l3343_334363


namespace simple_interest_problem_l3343_334364

/-- Given a principal amount and an interest rate, if increasing the rate by 4% for 2 years
    yields Rs. 60 more in interest, then the principal amount is Rs. 750. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end simple_interest_problem_l3343_334364


namespace sqrt_five_minus_one_gt_one_l3343_334333

theorem sqrt_five_minus_one_gt_one : Real.sqrt 5 - 1 > 1 := by
  sorry

end sqrt_five_minus_one_gt_one_l3343_334333


namespace cookies_per_bag_l3343_334350

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l3343_334350


namespace m_range_l3343_334327

theorem m_range (x : ℝ) :
  (∀ x, 1/3 < x ∧ x < 1/2 → m - 1 < x ∧ x < m + 1) ∧
  (∃ x, m - 1 < x ∧ x < m + 1 ∧ (x ≤ 1/3 ∨ 1/2 ≤ x)) →
  -1/2 ≤ m ∧ m ≤ 4/3 ∧ m ≠ -1/2 ∧ m ≠ 4/3 :=
by sorry

end m_range_l3343_334327


namespace hundred_from_twos_l3343_334313

theorem hundred_from_twos : (222 / 2) - (22 / 2) = 100 := by
  sorry

end hundred_from_twos_l3343_334313


namespace probability_three_white_balls_l3343_334398

/-- The probability of drawing 3 white balls from a box containing 8 white balls and 7 black balls -/
theorem probability_three_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 8)
  (h3 : black_balls = 7)
  (h4 : total_balls ≥ 3) :
  (Nat.choose white_balls 3 : ℚ) / (Nat.choose total_balls 3) = 8 / 65 := by
sorry

end probability_three_white_balls_l3343_334398


namespace abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l3343_334321

theorem abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  ¬(∀ x : ℝ, |x| ≤ 2 → 0 ≤ x ∧ x ≤ 2) :=
by sorry

end abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l3343_334321


namespace sum_of_four_cubes_equals_1812_l3343_334302

theorem sum_of_four_cubes_equals_1812 :
  (303 : ℤ)^3 + (301 : ℤ)^3 + (-302 : ℤ)^3 + (-302 : ℤ)^3 = 1812 := by
  sorry

end sum_of_four_cubes_equals_1812_l3343_334302


namespace number_ratio_l3343_334305

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l3343_334305


namespace a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3343_334337

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∃ a : ℝ, a = 1 → abs a = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ abs a = 1) := by
  sorry

end a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3343_334337


namespace quadratic_roots_real_l3343_334345

theorem quadratic_roots_real (a b c : ℝ) : 
  let discriminant := 4 * (b^2 + c^2)
  discriminant ≥ 0 := by sorry

end quadratic_roots_real_l3343_334345


namespace count_valid_insertions_l3343_334306

/-- The number of different three-digit numbers that can be inserted into 689???20312 to make it approximately 69 billion when rounded -/
def valid_insertions : ℕ :=
  let ten_million_digits := {5, 6, 7, 8, 9}
  let other_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card ten_million_digits) * (Finset.card other_digits) * (Finset.card other_digits)

theorem count_valid_insertions : valid_insertions = 500 := by
  sorry

end count_valid_insertions_l3343_334306


namespace order_of_expressions_l3343_334308

theorem order_of_expressions :
  let a : ℝ := 3^(3/2)
  let b : ℝ := 3^(5/2)
  let c : ℝ := Real.log 3 / Real.log 0.5
  c < a ∧ a < b := by
  sorry

end order_of_expressions_l3343_334308


namespace problem_i4_1_l3343_334385

theorem problem_i4_1 (f : ℝ → ℝ) :
  (∀ x, f x = (x^2 + x - 2)^2002 + 3) →
  f ((Real.sqrt 5 / 2) - 1/2) = 4 := by
  sorry

end problem_i4_1_l3343_334385


namespace ab_value_l3343_334367

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 405 / 16 := by
  sorry

end ab_value_l3343_334367


namespace four_distinct_solutions_l3343_334346

theorem four_distinct_solutions (p q : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (x₁^2 + p * |x₁| = q * x₁ - 1) ∧
    (x₂^2 + p * |x₂| = q * x₂ - 1) ∧
    (x₃^2 + p * |x₃| = q * x₃ - 1) ∧
    (x₄^2 + p * |x₄| = q * x₄ - 1)) ↔
  (p + |q| + 2 < 0) :=
by sorry

end four_distinct_solutions_l3343_334346


namespace number_equation_l3343_334369

theorem number_equation (x : ℝ) : x - (105 / 21) = 5995 ↔ x = 6000 := by
  sorry

end number_equation_l3343_334369


namespace store_turnover_equation_l3343_334326

/-- Represents the equation for the total turnover in the first quarter of a store,
    given an initial turnover and a monthly growth rate. -/
theorem store_turnover_equation (initial_turnover : ℝ) (growth_rate : ℝ) :
  initial_turnover = 50 →
  initial_turnover * (1 + (1 + growth_rate) + (1 + growth_rate)^2) = 600 :=
by sorry

end store_turnover_equation_l3343_334326


namespace lassis_from_ten_mangoes_l3343_334349

/-- A recipe for making lassis from mangoes -/
structure Recipe where
  mangoes : ℕ
  lassis : ℕ

/-- Given a recipe and a number of mangoes, calculate the number of lassis that can be made -/
def makeLassis (recipe : Recipe) (numMangoes : ℕ) : ℕ :=
  (recipe.lassis * numMangoes) / recipe.mangoes

theorem lassis_from_ten_mangoes (recipe : Recipe) 
  (h1 : recipe.mangoes = 3) 
  (h2 : recipe.lassis = 15) : 
  makeLassis recipe 10 = 50 := by
  sorry

end lassis_from_ten_mangoes_l3343_334349


namespace single_element_condition_intersection_condition_l3343_334393

-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 3 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 = 0}

-- Theorem for the first part of the problem
theorem single_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1/3) := by sorry

-- Theorem for the second part of the problem
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ (a > 1/3 ∨ a = -1) := by sorry

end single_element_condition_intersection_condition_l3343_334393


namespace turtle_ratio_l3343_334348

theorem turtle_ratio (total : ℕ) (green : ℕ) (h1 : total = 3200) (h2 : green = 800) :
  (total - green) / green = 3 := by
  sorry

end turtle_ratio_l3343_334348


namespace rectangular_prism_diagonals_l3343_334352

/-- A rectangular prism. -/
structure RectangularPrism :=
  (vertices : Nat)
  (edges : Nat)

/-- The number of diagonals in a rectangular prism. -/
def num_diagonals (rp : RectangularPrism) : Nat :=
  sorry

/-- Theorem: A rectangular prism with 12 vertices and 18 edges has 24 diagonals. -/
theorem rectangular_prism_diagonals :
  ∀ (rp : RectangularPrism), rp.vertices = 12 → rp.edges = 18 → num_diagonals rp = 24 :=
by
  sorry

end rectangular_prism_diagonals_l3343_334352


namespace f_inequality_solution_set_f_inequality_a_range_l3343_334397

def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

theorem f_inequality_a_range :
  {a : ℝ | ∃ x, f x ≤ 3/2 * a^2 - a} = {a : ℝ | a ≥ 5/3 ∨ a ≤ -1} :=
sorry

end f_inequality_solution_set_f_inequality_a_range_l3343_334397


namespace product_digit_sum_l3343_334338

theorem product_digit_sum : 
  let product := 2 * 3 * 5 * 7 * 11 * 13 * 17
  ∃ (digits : List Nat), 
    (∀ d ∈ digits, d < 10) ∧ 
    (product.repr.toList.map (λ c => c.toNat - '0'.toNat) = digits) ∧
    (digits.sum = 12) := by
  sorry

end product_digit_sum_l3343_334338


namespace f_4_1981_equals_tower_exp_l3343_334344

/-- A function f : ℕ → ℕ → ℕ satisfying the given recursive conditions -/
noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Helper function to represent towering exponentiation -/
def tower_exp : ℕ → ℕ → ℕ
| 0, n => n
| m + 1, n => 2^(tower_exp m n)

/-- The main theorem stating that f(4, 1981) is equal to a specific towering exponentiation -/
theorem f_4_1981_equals_tower_exp : 
  f 4 1981 = tower_exp 12 (2^2) :=
sorry

end f_4_1981_equals_tower_exp_l3343_334344


namespace inequality_equivalence_l3343_334334

theorem inequality_equivalence (x : ℝ) : -1/2 * x + 3 < 0 ↔ x > 6 := by
  sorry

end inequality_equivalence_l3343_334334


namespace triangle_classification_l3343_334318

theorem triangle_classification (a b c : ℝ) (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) :
  a = b ∨ a^2 + b^2 = c^2 := by
  sorry

end triangle_classification_l3343_334318


namespace c_minus_a_positive_l3343_334312

/-- A quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The graph of the quadratic function is a downward-opening parabola -/
def is_downward_opening (f : QuadraticFunction) : Prop :=
  f.a < 0

/-- The y-intercept of the quadratic function is positive -/
def has_positive_y_intercept (f : QuadraticFunction) : Prop :=
  f.c > 0

/-- Theorem stating that if a quadratic function's graph is a downward-opening parabola
    with a positive y-intercept, then c - a > 0 -/
theorem c_minus_a_positive (f : QuadraticFunction)
  (h1 : is_downward_opening f)
  (h2 : has_positive_y_intercept f) :
  f.c - f.a > 0 := by
  sorry

end c_minus_a_positive_l3343_334312


namespace altitude_segment_length_l3343_334357

/-- Represents an acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- Length of one segment created by an altitude -/
  segment1 : ℝ
  /-- Length of another segment created by an altitude -/
  segment2 : ℝ
  /-- Length of a third segment created by an altitude -/
  segment3 : ℝ
  /-- Length of the fourth segment created by an altitude -/
  segment4 : ℝ
  /-- The triangle is acute -/
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0

/-- The theorem stating that for the given acute triangle with altitudes, the fourth segment length is 4.5 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) 
  (h1 : t.segment1 = 4) 
  (h2 : t.segment2 = 6) 
  (h3 : t.segment3 = 3) : 
  t.segment4 = 4.5 := by
  sorry

end altitude_segment_length_l3343_334357


namespace stairs_height_l3343_334316

theorem stairs_height (h : ℝ) 
  (total_height : 3 * h + h / 2 + (h / 2 + 10) = 70) : h = 15 := by
  sorry

end stairs_height_l3343_334316


namespace inequality_solution_l3343_334329

theorem inequality_solution (x : ℝ) : 
  (-1 ≤ (x^2 + 3*x - 1) / (4 - x^2) ∧ (x^2 + 3*x - 1) / (4 - x^2) < 1) ↔ 
  (x < -5/2 ∨ (-1 ≤ x ∧ x < 1)) :=
by sorry

end inequality_solution_l3343_334329


namespace function_properties_l3343_334359

/-- The function f(x) defined on the real line -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x + b

theorem function_properties (a b : ℝ) (h_a : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 0 1, f a b x ≤ 1) 
  (h_max_exists : ∃ x ∈ Set.Icc 0 1, f a b x = 1)
  (h_min : ∀ x ∈ Set.Icc 0 1, f a b x ≥ -2) 
  (h_min_exists : ∃ x ∈ Set.Icc 0 1, f a b x = -2) :
  (a = 1 ∧ b = 1) ∧ 
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a b x > -x + m) ↔ m < -1) :=
sorry

end function_properties_l3343_334359


namespace ellipse_equation_l3343_334309

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  majorAxisLength : ℝ

-- Define the standard form of an ellipse equation
def StandardEllipseEquation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) : 
  e.foci = ((-2, 0), (2, 0)) ∧ e.majorAxisLength = 10 →
  ∀ x y : ℝ, StandardEllipseEquation 25 21 x y :=
by
  sorry

end ellipse_equation_l3343_334309


namespace double_inequality_solution_l3343_334342

theorem double_inequality_solution (x : ℝ) : 
  (4 * x - 3 < (x - 2)^2 ∧ (x - 2)^2 < 6 * x - 5) ↔ (7 < x ∧ x < 9) :=
sorry

end double_inequality_solution_l3343_334342


namespace student_sums_correct_l3343_334361

theorem student_sums_correct (wrong_sums correct_sums total_sums : ℕ) : 
  wrong_sums = 2 * correct_sums →
  total_sums = 36 →
  wrong_sums + correct_sums = total_sums →
  correct_sums = 12 := by
  sorry

end student_sums_correct_l3343_334361


namespace largest_digit_divisible_by_6_l3343_334388

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 : 
  ∀ N : ℕ, N ≤ 9 → 
    (is_divisible_by_6 (71820 + N) → N ≤ 6) ∧ 
    (is_divisible_by_6 (71826)) := by
  sorry

end largest_digit_divisible_by_6_l3343_334388


namespace weight_loss_calculation_l3343_334358

theorem weight_loss_calculation (W : ℝ) (x : ℝ) : 
  W * (1 - x / 100 + 2 / 100) = W * (100 - 10.24) / 100 → x = 12.24 := by
  sorry

end weight_loss_calculation_l3343_334358


namespace bridge_length_proof_l3343_334353

/-- Calculate the distance traveled with constant acceleration -/
def distance_traveled (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_velocity * time + 0.5 * acceleration * time^2

/-- Convert kilometers to meters -/
def km_to_meters (km : ℝ) : ℝ := km * 1000

theorem bridge_length_proof (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) 
  (h1 : initial_velocity = 3)
  (h2 : acceleration = 0.2)
  (h3 : time = 0.25) :
  km_to_meters (distance_traveled initial_velocity acceleration time) = 756.25 := by
  sorry

end bridge_length_proof_l3343_334353


namespace pentagon_fencing_cost_l3343_334310

/-- Calculates the total cost of fencing a pentagon park -/
def fencing_cost (sides : Fin 5 → ℝ) (costs : Fin 5 → ℝ) : ℝ :=
  (sides 0 * costs 0) + (sides 1 * costs 1) + (sides 2 * costs 2) + 
  (sides 3 * costs 3) + (sides 4 * costs 4)

theorem pentagon_fencing_cost :
  let sides : Fin 5 → ℝ := ![50, 75, 60, 80, 65]
  let costs : Fin 5 → ℝ := ![2, 3, 4, 3.5, 5]
  fencing_cost sides costs = 1170 := by sorry

end pentagon_fencing_cost_l3343_334310


namespace power_of_fraction_three_fourths_cubed_l3343_334351

theorem power_of_fraction_three_fourths_cubed :
  (3 / 4 : ℚ) ^ 3 = 27 / 64 := by
  sorry

end power_of_fraction_three_fourths_cubed_l3343_334351


namespace smallest_n_doughnuts_l3343_334341

theorem smallest_n_doughnuts : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m.val - 1) % 11 = 0 → n ≤ m) ∧
  (15 * n.val - 1) % 11 = 0 ∧
  n.val = 3 := by
  sorry

end smallest_n_doughnuts_l3343_334341


namespace odd_number_multiple_square_differences_l3343_334323

theorem odd_number_multiple_square_differences : ∃ (n : ℕ), 
  Odd n ∧ (∃ (a b c d : ℕ), a ≠ c ∧ b ≠ d ∧ n = a^2 - b^2 ∧ n = c^2 - d^2) := by
  sorry

end odd_number_multiple_square_differences_l3343_334323


namespace min_max_inequality_l3343_334325

theorem min_max_inequality (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (p q x m n y : ℕ),
    p = min a b ∧
    q = min c d ∧
    x = max p q ∧
    m = max a b ∧
    n = max c d ∧
    y = min m n ∧
    ((x > y) ∨ (x < y)) :=
by sorry

end min_max_inequality_l3343_334325


namespace women_married_fraction_l3343_334300

theorem women_married_fraction (total : ℕ) (women : ℕ) (married : ℕ) (men : ℕ) :
  women = (61 * total) / 100 →
  married = (60 * total) / 100 →
  men = total - women →
  (men - (men / 3)) * 3 = 2 * men →
  (married - (men / 3) : ℚ) / women = 47 / 61 :=
by
  sorry

end women_married_fraction_l3343_334300


namespace sum_of_roots_equals_six_l3343_334355

theorem sum_of_roots_equals_six : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 9*x) / (x + 3)
  ∃ (a b : ℝ), (f a = 9 ∧ f b = 9 ∧ a ≠ b) ∧ a + b = 6 := by
sorry

end sum_of_roots_equals_six_l3343_334355


namespace circle_center_polar_coordinates_l3343_334378

-- Define the polar equation of the circle
def circle_equation (ρ θ : Real) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center in polar coordinates
def is_center (ρ θ : Real) : Prop := 
  (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- Theorem statement
theorem circle_center_polar_coordinates :
  ∀ ρ θ : Real, circle_equation ρ θ → 
  ∃ ρ_c θ_c : Real, is_center ρ_c θ_c ∧ 
  (ρ - ρ_c * Real.cos (θ - θ_c))^2 + (ρ * Real.sin θ - ρ_c * Real.sin θ_c)^2 = ρ_c^2 :=
sorry

end circle_center_polar_coordinates_l3343_334378


namespace pascal_39th_number_40th_row_l3343_334362

-- Define Pascal's triangle coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem pascal_39th_number_40th_row : pascal 40 38 = 780 := by
  sorry

end pascal_39th_number_40th_row_l3343_334362


namespace zacks_marbles_l3343_334399

/-- Zack's marble distribution problem -/
theorem zacks_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : initial_marbles = 65)
  (h2 : friends = 3)
  (h3 : marbles_per_friend = 20)
  (h4 : initial_marbles % friends ≠ 0) : 
  initial_marbles - friends * marbles_per_friend = 5 :=
by sorry

end zacks_marbles_l3343_334399


namespace least_frood_drop_beats_eat_l3343_334311

theorem least_frood_drop_beats_eat : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (k * (k + 1)) / 2 ≤ 15 * k) ∧ (n * (n + 1)) / 2 > 15 * n :=
by sorry

end least_frood_drop_beats_eat_l3343_334311


namespace sum_of_squares_of_roots_l3343_334319

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 8 * x₁ - 7 = 0) → 
  (5 * x₂^2 + 8 * x₂ - 7 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 134/25) := by
  sorry

end sum_of_squares_of_roots_l3343_334319
