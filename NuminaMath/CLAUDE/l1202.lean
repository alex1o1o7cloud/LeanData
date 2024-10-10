import Mathlib

namespace birgit_travel_time_l1202_120256

def hiking_time : ℝ := 3.5
def distance_traveled : ℝ := 21
def birgit_speed_difference : ℝ := 4
def birgit_travel_distance : ℝ := 8

theorem birgit_travel_time : 
  let total_minutes := hiking_time * 60
  let average_speed := total_minutes / distance_traveled
  let birgit_speed := average_speed - birgit_speed_difference
  birgit_speed * birgit_travel_distance = 48 := by sorry

end birgit_travel_time_l1202_120256


namespace simplify_fraction_l1202_120240

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 0) :
  (1 - 1 / (x - 3)) / ((x^2 - 4*x) / (x^2 - 9)) = (x + 3) / x := by
  sorry

end simplify_fraction_l1202_120240


namespace scientific_notation_equivalence_l1202_120221

theorem scientific_notation_equivalence : 
  1400000000 = (1.4 : ℝ) * (10 : ℝ) ^ 9 := by sorry

end scientific_notation_equivalence_l1202_120221


namespace unique_delivery_exists_l1202_120272

/-- Represents the amount of cargo delivered to each warehouse -/
structure Delivery where
  first : Int
  second : Int
  third : Int

/-- Checks if a delivery satisfies the given conditions -/
def satisfiesConditions (d : Delivery) : Prop :=
  d.first + d.second = 400 ∧
  d.second + d.third = -300 ∧
  d.first + d.third = -440

/-- The theorem stating that there is a unique delivery satisfying the conditions -/
theorem unique_delivery_exists : ∃! d : Delivery, satisfiesConditions d ∧ 
  d.first = -130 ∧ d.second = -270 ∧ d.third = 230 := by
  sorry

end unique_delivery_exists_l1202_120272


namespace divisor_property_solutions_l1202_120286

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ+ :=
  sorry

/-- The property that the fourth power of the number of divisors equals the number itself -/
def has_divisor_property (m : ℕ+) : Prop :=
  (num_divisors m) ^ 4 = m

/-- Theorem stating that only 625, 6561, and 4100625 satisfy the divisor property -/
theorem divisor_property_solutions :
  ∀ m : ℕ+, has_divisor_property m ↔ m ∈ ({625, 6561, 4100625} : Set ℕ+) :=
sorry

end divisor_property_solutions_l1202_120286


namespace rice_and_flour_consumption_l1202_120233

theorem rice_and_flour_consumption (initial_rice initial_flour consumed : ℕ) : 
  initial_rice = 500 →
  initial_flour = 200 →
  initial_rice - consumed = 7 * (initial_flour - consumed) →
  consumed = 150 := by
sorry

end rice_and_flour_consumption_l1202_120233


namespace exists_divisible_by_four_l1202_120239

def collatz_sequence (a₁ : ℕ+) : ℕ → ℕ
  | 0 => a₁.val
  | n + 1 => 
    let prev := collatz_sequence a₁ n
    if prev % 2 = 0 then prev / 2 else 3 * prev + 1

theorem exists_divisible_by_four (a₁ : ℕ+) : 
  ∃ n : ℕ, (collatz_sequence a₁ n) % 4 = 0 := by
  sorry

end exists_divisible_by_four_l1202_120239


namespace age_ratio_theorem_l1202_120269

def age_difference : ℕ := 20
def younger_present_age : ℕ := 35
def years_ago : ℕ := 15

def elder_present_age : ℕ := younger_present_age + age_difference

def younger_past_age : ℕ := younger_present_age - years_ago
def elder_past_age : ℕ := elder_present_age - years_ago

theorem age_ratio_theorem :
  (elder_past_age % younger_past_age = 0) →
  (elder_past_age / younger_past_age = 2) :=
by sorry

end age_ratio_theorem_l1202_120269


namespace circle_center_coordinates_l1202_120212

/-- The circle C is defined by the equation x^2 + y^2 + ax - 2y + b = 0 -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- The line of symmetry is defined by the equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Point P has coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

/-- The symmetric point of P with respect to the line x + y - 1 = 0 -/
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.2 - 1, P.1 - 1)

theorem circle_center_coordinates (a b : ℝ) :
  (circle_equation 2 1 a b) ∧ 
  (circle_equation (symmetric_point point_P).1 (symmetric_point point_P).2 a b) →
  ∃ (h k : ℝ), h = 0 ∧ k = 1 ∧ 
    ∀ (x y : ℝ), circle_equation x y a b ↔ (x - h)^2 + (y - k)^2 = h^2 + k^2 - b :=
by sorry

end circle_center_coordinates_l1202_120212


namespace complex_number_quadrant_l1202_120215

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 - I) / I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l1202_120215


namespace wire_ratio_proof_l1202_120296

theorem wire_ratio_proof (total_length : ℝ) (short_length : ℝ) :
  total_length = 70 →
  short_length = 20 →
  short_length / (total_length - short_length) = 2 / 5 := by
  sorry

end wire_ratio_proof_l1202_120296


namespace log_equation_solution_l1202_120247

theorem log_equation_solution (k c p : ℝ) (h : k > 0) (hp : p > 0) :
  Real.log k^2 / Real.log 10 = c - 2 * Real.log p / Real.log 10 →
  k = 10^c / p := by
sorry

end log_equation_solution_l1202_120247


namespace min_covering_size_l1202_120288

def X : Finset Nat := {1, 2, 3, 4, 5}

def is_covering (F : Finset (Finset Nat)) : Prop :=
  ∀ B ∈ Finset.powerset X, B.card = 3 → ∃ A ∈ F, A ⊆ B

theorem min_covering_size :
  ∃ F : Finset (Finset Nat),
    (∀ A ∈ F, A ⊆ X ∧ A.card = 2) ∧
    is_covering F ∧
    F.card = 10 ∧
    (∀ G : Finset (Finset Nat),
      (∀ A ∈ G, A ⊆ X ∧ A.card = 2) →
      is_covering G →
      G.card ≥ 10) :=
sorry

end min_covering_size_l1202_120288


namespace ivy_cupcakes_l1202_120228

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := 20

/-- The additional number of cupcakes Ivy baked in the afternoon compared to the morning -/
def afternoon_extra : ℕ := 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_extra)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end ivy_cupcakes_l1202_120228


namespace school_sections_theorem_l1202_120214

/-- The number of sections formed when dividing students into equal groups -/
def number_of_sections (boys girls : Nat) : Nat :=
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls)

/-- Theorem stating that the total number of sections for 408 boys and 216 girls is 26 -/
theorem school_sections_theorem :
  number_of_sections 408 216 = 26 := by
  sorry

end school_sections_theorem_l1202_120214


namespace base8_of_215_l1202_120250

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base8_of_215 : toBase8 215 = 327 := by sorry

end base8_of_215_l1202_120250


namespace largest_multiple_of_8_less_than_100_l1202_120209

theorem largest_multiple_of_8_less_than_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l1202_120209


namespace marks_departure_time_correct_l1202_120251

-- Define the problem parameters
def robs_normal_time : ℝ := 1
def robs_additional_time : ℝ := 0.5
def marks_normal_time_factor : ℝ := 3
def marks_time_reduction : ℝ := 0.2
def time_zone_difference : ℝ := 2
def robs_departure_time : ℝ := 11

-- Define the function to calculate Mark's departure time
def calculate_marks_departure_time : ℝ :=
  let robs_travel_time := robs_normal_time + robs_additional_time
  let marks_normal_time := marks_normal_time_factor * robs_normal_time
  let marks_travel_time := marks_normal_time * (1 - marks_time_reduction)
  let robs_arrival_time := robs_departure_time + robs_travel_time
  let marks_arrival_time := robs_arrival_time + time_zone_difference
  marks_arrival_time - marks_travel_time

-- Theorem statement
theorem marks_departure_time_correct :
  calculate_marks_departure_time = 11 + 36 / 60 :=
sorry

end marks_departure_time_correct_l1202_120251


namespace mnp_value_l1202_120265

theorem mnp_value (a b x z : ℝ) (m n p : ℤ) 
  (h : a^12 * x * z - a^10 * z - a^9 * x = a^8 * (b^6 - 1)) 
  (h_equiv : (a^m * x - a^n) * (a^p * z - a^3) = a^8 * b^6) : 
  m * n * p = 4 := by
sorry

end mnp_value_l1202_120265


namespace aquarium_water_volume_l1202_120237

/-- The initial volume of water in the aquarium -/
def initial_volume : ℝ := 36

/-- The volume of water after the cat spills half and Nancy triples the remainder -/
def final_volume : ℝ := 54

theorem aquarium_water_volume : 
  (3 * (initial_volume / 2)) = final_volume :=
by sorry

end aquarium_water_volume_l1202_120237


namespace condition_p_sufficient_not_necessary_for_q_l1202_120219

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

def condition_q (x : ℝ) : Prop := |x - 2| < 1

-- Theorem statement
theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, condition_p x → condition_q x) ∧
  ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by sorry


end condition_p_sufficient_not_necessary_for_q_l1202_120219


namespace solution_set_l1202_120284

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := x^(lg x) = x^3 / 100

-- Theorem statement
theorem solution_set : 
  {x : ℝ | equation x} = {10, 100} :=
sorry

end solution_set_l1202_120284


namespace rectangle_count_is_297_l1202_120216

/-- Represents a grid with a hole in the middle -/
structure Grid :=
  (size : ℕ)
  (hole_x : ℕ)
  (hole_y : ℕ)

/-- Counts the number of non-degenerate rectangles in a grid with a hole -/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- The specific 7x7 grid with a hole at (4,4) -/
def specific_grid : Grid :=
  { size := 7, hole_x := 4, hole_y := 4 }

/-- Theorem stating that the number of non-degenerate rectangles in the specific grid is 297 -/
theorem rectangle_count_is_297 : count_rectangles specific_grid = 297 :=
  sorry

end rectangle_count_is_297_l1202_120216


namespace range_of_a_l1202_120202

-- Define the universal set U
def U : Set ℝ := {x : ℝ | 0 < x ∧ x < 9}

-- Define set A parameterized by a
def A (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∃ x, x ∈ A a) ∧ ¬(A a ⊆ U) ↔ 1 < a ∧ a ≤ 9 := by sorry

end range_of_a_l1202_120202


namespace bus_stoppage_time_l1202_120271

/-- Proves that a bus with given speeds stops for 30 minutes per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 32)
  (h2 : speed_with_stops = 16) : 
  (1 - speed_with_stops / speed_without_stops) * 60 = 30 := by
  sorry

end bus_stoppage_time_l1202_120271


namespace parallel_line_length_l1202_120264

/-- A triangle with a base of 20 inches and height of 10 inches, 
    divided into four equal areas by two parallel lines -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  baseParallel : ℝ
  base_eq : base = 20
  height_eq : height = 10
  equal_areas : baseParallel > 0 ∧ baseParallel < base

theorem parallel_line_length (t : DividedTriangle) : t.baseParallel = 10 := by
  sorry

end parallel_line_length_l1202_120264


namespace a_equals_negative_one_l1202_120262

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (a - 1) + Complex.I^4 * (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a equals -1 -/
theorem a_equals_negative_one : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -1 :=
sorry

end a_equals_negative_one_l1202_120262


namespace adjacent_supplementary_angles_l1202_120244

/-- Given two adjacent supplementary angles, if one is 60°, then the other is 120°. -/
theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  angle1 = 60 → 
  angle1 + angle2 = 180 → 
  angle2 = 120 := by
sorry

end adjacent_supplementary_angles_l1202_120244


namespace optimal_tic_tac_toe_draw_l1202_120295

/-- Represents a player in Tic-Tac-Toe -/
inductive Player : Type
| X : Player
| O : Player

/-- Represents a position on the Tic-Tac-Toe board -/
inductive Position : Type
| one | two | three | four | five | six | seven | eight | nine

/-- Represents the state of a Tic-Tac-Toe game -/
structure GameState :=
  (board : Position → Option Player)
  (currentPlayer : Player)

/-- Represents an optimal move in Tic-Tac-Toe -/
def OptimalMove : GameState → Position → Prop := sorry

/-- Represents the outcome of a Tic-Tac-Toe game -/
inductive GameOutcome : Type
| Draw : GameOutcome
| Win : Player → GameOutcome

/-- Plays a full game of Tic-Tac-Toe with optimal moves -/
def playOptimalGame : GameState → GameOutcome := sorry

/-- Theorem: Every game of Tic-Tac-Toe between optimal players ends in a draw -/
theorem optimal_tic_tac_toe_draw :
  ∀ (initialState : GameState),
  (∀ (state : GameState) (move : Position), OptimalMove state move → 
    playOptimalGame (sorry : GameState) = playOptimalGame state) →
  playOptimalGame initialState = GameOutcome.Draw :=
sorry

end optimal_tic_tac_toe_draw_l1202_120295


namespace range_of_x_for_sqrt_4_minus_x_l1202_120259

theorem range_of_x_for_sqrt_4_minus_x : 
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 4 - x) ↔ x ≤ 4 := by sorry

end range_of_x_for_sqrt_4_minus_x_l1202_120259


namespace sphere_surface_area_with_inscribed_cube_l1202_120263

theorem sphere_surface_area_with_inscribed_cube (edge_length : ℝ) (radius : ℝ) : 
  edge_length = 2 → 
  radius^2 = 3 →
  4 * π * radius^2 = 12 * π := by
sorry

end sphere_surface_area_with_inscribed_cube_l1202_120263


namespace football_victory_points_l1202_120213

/-- Represents the points system in a football competition -/
structure FootballPoints where
  victory : ℕ
  draw : ℕ := 1
  defeat : ℕ := 0

/-- Represents the state of a team in the competition -/
structure TeamState where
  totalMatches : ℕ := 20
  playedMatches : ℕ := 5
  currentPoints : ℕ := 8
  targetPoints : ℕ := 40
  minRemainingWins : ℕ := 9

/-- The minimum number of points for a victory that satisfies the given conditions -/
def minVictoryPoints (points : FootballPoints) (state : TeamState) : Prop :=
  points.victory = 3 ∧
  points.victory * state.minRemainingWins + 
    (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw ≥ 
    state.targetPoints - state.currentPoints ∧
  ∀ v : ℕ, v < points.victory → 
    v * state.minRemainingWins + 
      (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw < 
      state.targetPoints - state.currentPoints

theorem football_victory_points :
  ∃ (points : FootballPoints) (state : TeamState), minVictoryPoints points state := by
  sorry

end football_victory_points_l1202_120213


namespace total_clothes_donated_l1202_120268

/-- Proves that the total number of clothes donated is 87 given the specified conditions --/
theorem total_clothes_donated (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 12 →
  pants = 5 * shirts →
  shorts = pants / 4 →
  shirts + pants + shorts = 87 := by
  sorry

end total_clothes_donated_l1202_120268


namespace sixth_grade_homework_forgetfulness_l1202_120205

theorem sixth_grade_homework_forgetfulness (students_A : ℕ) (students_B : ℕ) 
  (forgot_A_percent : ℚ) (forgot_B_percent : ℚ) (total_forgot_percent : ℚ) :
  students_A = 20 →
  forgot_A_percent = 20 / 100 →
  forgot_B_percent = 15 / 100 →
  total_forgot_percent = 16 / 100 →
  (students_A : ℚ) * forgot_A_percent + (students_B : ℚ) * forgot_B_percent = 
    total_forgot_percent * ((students_A : ℚ) + (students_B : ℚ)) →
  students_B = 80 := by
sorry

end sixth_grade_homework_forgetfulness_l1202_120205


namespace toonie_is_two_dollar_coin_l1202_120279

/-- Represents the types of coins in Antonella's purse -/
inductive Coin
  | Loonie
  | Toonie

/-- The value of a coin in dollars -/
def coin_value (c : Coin) : ℕ :=
  match c with
  | Coin.Loonie => 1
  | Coin.Toonie => 2

/-- Antonella's coin situation -/
structure AntoniellaPurse where
  coins : List Coin
  initial_toonies : ℕ
  spent : ℕ
  remaining : ℕ

/-- The conditions of Antonella's coins -/
def antonellas_coins : AntoniellaPurse :=
  { coins := List.replicate 10 Coin.Toonie,  -- placeholder, actual distribution doesn't matter
    initial_toonies := 4,
    spent := 3,
    remaining := 11 }

theorem toonie_is_two_dollar_coin (purse : AntoniellaPurse := antonellas_coins) :
  ∃ (c : Coin), coin_value c = 2 ∧ c = Coin.Toonie :=
sorry

end toonie_is_two_dollar_coin_l1202_120279


namespace geometric_sequence_sum_l1202_120289

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 2 * a 9 = -8) →
  (a 1 + a 13 = 17 ∨ a 1 + a 13 = -17/2) :=
by sorry

end geometric_sequence_sum_l1202_120289


namespace chocolate_division_l1202_120201

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (friend_fraction : ℚ) :
  total_chocolate = 72 / 7 →
  num_piles = 8 →
  friend_fraction = 1 / 3 →
  friend_fraction * (total_chocolate / num_piles) = 3 / 7 := by
  sorry

end chocolate_division_l1202_120201


namespace more_valid_placements_diff_intersections_l1202_120282

/-- Represents the number of radial streets in city N -/
def radial_streets : ℕ := 7

/-- Represents the number of parallel streets in city N -/
def parallel_streets : ℕ := 7

/-- Total number of intersections in the city -/
def total_intersections : ℕ := radial_streets * parallel_streets

/-- Calculates the number of valid store placements when stores must not be at the same intersection -/
def valid_placements_diff_intersections : ℕ := total_intersections * (total_intersections - 1)

/-- Calculates the number of valid store placements when stores must not be on the same street -/
def valid_placements_diff_streets : ℕ := 
  valid_placements_diff_intersections - 2 * (radial_streets * (total_intersections - radial_streets))

/-- Theorem stating that the condition of different intersections allows more valid placements -/
theorem more_valid_placements_diff_intersections : 
  valid_placements_diff_intersections > valid_placements_diff_streets :=
sorry

end more_valid_placements_diff_intersections_l1202_120282


namespace jimmy_lodging_cost_l1202_120203

/-- Calculates the total lodging cost for Jimmy's vacation --/
def total_lodging_cost (hostel_nights : ℕ) (hostel_cost_per_night : ℕ) 
  (cabin_nights : ℕ) (cabin_total_cost_per_night : ℕ) (cabin_friends : ℕ) : ℕ :=
  hostel_nights * hostel_cost_per_night + 
  cabin_nights * cabin_total_cost_per_night / (cabin_friends + 1)

/-- Theorem stating that Jimmy's total lodging cost is $75 --/
theorem jimmy_lodging_cost : 
  total_lodging_cost 3 15 2 45 2 = 75 := by
  sorry

end jimmy_lodging_cost_l1202_120203


namespace only_A_and_B_excellent_l1202_120238

/-- Represents a student's scores in three components -/
structure StudentScores where
  written : ℝ
  practical : ℝ
  growth : ℝ

/-- Calculates the total evaluation score for a student -/
def totalScore (s : StudentScores) : ℝ :=
  0.5 * s.written + 0.2 * s.practical + 0.3 * s.growth

/-- Determines if a score is excellent (over 90) -/
def isExcellent (score : ℝ) : Prop :=
  score > 90

/-- The scores of student A -/
def studentA : StudentScores :=
  { written := 90, practical := 83, growth := 95 }

/-- The scores of student B -/
def studentB : StudentScores :=
  { written := 98, practical := 90, growth := 95 }

/-- The scores of student C -/
def studentC : StudentScores :=
  { written := 80, practical := 88, growth := 90 }

/-- Theorem stating that only students A and B have excellent scores -/
theorem only_A_and_B_excellent :
  isExcellent (totalScore studentA) ∧
  isExcellent (totalScore studentB) ∧
  ¬isExcellent (totalScore studentC) := by
  sorry


end only_A_and_B_excellent_l1202_120238


namespace inequality_implication_l1202_120234

theorem inequality_implication (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end inequality_implication_l1202_120234


namespace problem_solution_l1202_120266

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end problem_solution_l1202_120266


namespace plot_width_l1202_120267

/-- Proves that a rectangular plot with given conditions has a width of 47.5 meters -/
theorem plot_width (length : ℝ) (poles : ℕ) (pole_distance : ℝ) (width : ℝ) :
  length = 90 →
  poles = 56 →
  pole_distance = 5 →
  (poles - 1 : ℝ) * pole_distance = 2 * (length + width) →
  width = 47.5 := by
  sorry

end plot_width_l1202_120267


namespace only_paintable_integer_l1202_120232

/-- Represents a painting configuration for the fence. -/
structure PaintingConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval
  v : ℕ+  -- Victor's interval

/-- Checks if a picket is painted by Harold. -/
def paintedByHarold (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.h.val = 1

/-- Checks if a picket is painted by Tanya. -/
def paintedByTanya (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.t.val = 2

/-- Checks if a picket is painted by Ulysses. -/
def paintedByUlysses (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.u.val = 3

/-- Checks if a picket is painted by Victor. -/
def paintedByVictor (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.v.val = 4

/-- Checks if a picket is painted by exactly one person. -/
def paintedOnce (config : PaintingConfig) (picket : ℕ) : Prop :=
  (paintedByHarold config picket ∨ paintedByTanya config picket ∨
   paintedByUlysses config picket ∨ paintedByVictor config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByTanya config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByUlysses config picket ∧ paintedByVictor config picket)

/-- Checks if a configuration is paintable. -/
def isPaintable (config : PaintingConfig) : Prop :=
  ∀ picket : ℕ, picket > 0 → paintedOnce config picket

/-- Calculates the paintable integer for a configuration. -/
def paintableInteger (config : PaintingConfig) : ℕ :=
  1000 * config.h.val + 100 * config.t.val + 10 * config.u.val + config.v.val

/-- The main theorem stating that 4812 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ config : PaintingConfig, isPaintable config → paintableInteger config = 4812 := by
  sorry


end only_paintable_integer_l1202_120232


namespace tangent_lines_not_always_same_l1202_120283

-- Define a curve as a function from ℝ to ℝ
def Curve := ℝ → ℝ

-- Define a point in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in ℝ² as a pair of slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a tangent line to a curve at a point
def tangentLineToCurve (f : Curve) (p : Point) : Line := sorry

-- Define a tangent line passing through a point
def tangentLineThroughPoint (p : Point) : Line := sorry

-- The theorem to prove
theorem tangent_lines_not_always_same (f : Curve) (p : Point) : 
  ¬ ∀ (f : Curve) (p : Point), tangentLineToCurve f p = tangentLineThroughPoint p :=
sorry

end tangent_lines_not_always_same_l1202_120283


namespace smallest_ending_9_div_13_proof_l1202_120223

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def smallest_ending_9_div_13 : ℕ := 129

theorem smallest_ending_9_div_13_proof :
  (ends_in_9 smallest_ending_9_div_13) ∧
  (smallest_ending_9_div_13 % 13 = 0) ∧
  (∀ m : ℕ, m < smallest_ending_9_div_13 → ¬(ends_in_9 m ∧ m % 13 = 0)) :=
by sorry

end smallest_ending_9_div_13_proof_l1202_120223


namespace red_bottles_count_l1202_120298

/-- The number of red water bottles in the fridge -/
def red_bottles : ℕ := 2

/-- The number of black water bottles in the fridge -/
def black_bottles : ℕ := 3

/-- The number of blue water bottles in the fridge -/
def blue_bottles : ℕ := 4

/-- The total number of water bottles initially in the fridge -/
def total_bottles : ℕ := 9

theorem red_bottles_count : red_bottles + black_bottles + blue_bottles = total_bottles := by
  sorry

end red_bottles_count_l1202_120298


namespace smallest_number_divisible_l1202_120236

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 21 * k)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 21 * k₄) := by
  sorry

#check smallest_number_divisible

end smallest_number_divisible_l1202_120236


namespace aron_dusting_days_l1202_120220

/-- Represents the cleaning schedule and durations for Aron -/
structure CleaningSchedule where
  vacuumingTimePerDay : ℕ
  vacuumingDaysPerWeek : ℕ
  dustingTimePerDay : ℕ
  totalCleaningTimePerWeek : ℕ

/-- Calculates the number of days Aron spends dusting per week -/
def dustingDaysPerWeek (schedule : CleaningSchedule) : ℕ :=
  let totalVacuumingTime := schedule.vacuumingTimePerDay * schedule.vacuumingDaysPerWeek
  let totalDustingTime := schedule.totalCleaningTimePerWeek - totalVacuumingTime
  totalDustingTime / schedule.dustingTimePerDay

/-- Theorem stating that Aron spends 2 days a week dusting -/
theorem aron_dusting_days (schedule : CleaningSchedule)
    (h1 : schedule.vacuumingTimePerDay = 30)
    (h2 : schedule.vacuumingDaysPerWeek = 3)
    (h3 : schedule.dustingTimePerDay = 20)
    (h4 : schedule.totalCleaningTimePerWeek = 130) :
    dustingDaysPerWeek schedule = 2 := by
  sorry

#eval dustingDaysPerWeek {
  vacuumingTimePerDay := 30,
  vacuumingDaysPerWeek := 3,
  dustingTimePerDay := 20,
  totalCleaningTimePerWeek := 130
}

end aron_dusting_days_l1202_120220


namespace correct_observation_value_l1202_120246

/-- Proves that the correct value of an incorrectly recorded observation is 58,
    given the initial and corrected means of a set of observations. -/
theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℚ)
  (incorrect_value : ℚ)
  (corrected_mean : ℚ)
  (h_n : n = 40)
  (h_initial : initial_mean = 36)
  (h_incorrect : incorrect_value = 20)
  (h_corrected : corrected_mean = 36.45) :
  (n : ℚ) * corrected_mean - ((n : ℚ) * initial_mean - incorrect_value) = 58 :=
sorry

end correct_observation_value_l1202_120246


namespace quadratic_roots_min_value_l1202_120261

theorem quadratic_roots_min_value (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*m*x₁ + m + 6 = 0 →
  x₂^2 - 2*m*x₂ + m + 6 = 0 →
  x₁ ≠ x₂ →
  ∀ y : ℝ, y = (x₁ - 1)^2 + (x₂ - 1)^2 → y ≥ 8 :=
by sorry

end quadratic_roots_min_value_l1202_120261


namespace no_natural_solution_equation_l1202_120257

theorem no_natural_solution_equation : ∀ x y : ℕ, 2^x + 21^x ≠ y^3 := by
  sorry

end no_natural_solution_equation_l1202_120257


namespace odd_divisors_implies_perfect_square_l1202_120293

theorem odd_divisors_implies_perfect_square (n : ℕ) : 
  (Odd (Nat.card {d : ℕ | d ∣ n})) → ∃ m : ℕ, n = m^2 := by
  sorry

end odd_divisors_implies_perfect_square_l1202_120293


namespace tower_painting_ways_level_painting_ways_l1202_120248

/-- Represents the number of ways to paint a single level of the tower -/
def paint_level : ℕ := 96

/-- Represents the number of colors available for painting -/
def num_colors : ℕ := 3

/-- Represents the number of levels in the tower -/
def num_levels : ℕ := 3

/-- Theorem stating the number of ways to paint the entire tower -/
theorem tower_painting_ways :
  num_colors * paint_level * paint_level = 27648 :=
by sorry

/-- Theorem stating the number of ways to paint a single level -/
theorem level_painting_ways :
  paint_level = 96 :=
by sorry

end tower_painting_ways_level_painting_ways_l1202_120248


namespace arithmetic_sequence_sum_example_l1202_120253

/-- The sum of an arithmetic sequence with given parameters. -/
def arithmetic_sequence_sum (n : ℕ) (a l : ℤ) : ℤ :=
  n * (a + l) / 2

/-- Theorem stating that the sum of the given arithmetic sequence is 175. -/
theorem arithmetic_sequence_sum_example :
  arithmetic_sequence_sum 10 (-5) 40 = 175 := by
  sorry

end arithmetic_sequence_sum_example_l1202_120253


namespace fruit_seller_apples_l1202_120260

/-- Theorem: If a fruit seller sells 40% of his apples and has 420 apples remaining, 
    then he originally had 700 apples. -/
theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℚ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end fruit_seller_apples_l1202_120260


namespace sqrt_fraction_equivalence_l1202_120208

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x - 1) / x)) = -x := by sorry

end sqrt_fraction_equivalence_l1202_120208


namespace perpendicular_parallel_implies_perpendicular_l1202_120210

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpendicular l α)
  (h3 : parallel l m) :
  perpendicular m α :=
sorry

end perpendicular_parallel_implies_perpendicular_l1202_120210


namespace difference_of_squares_2006_l1202_120294

def is_difference_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

theorem difference_of_squares_2006 :
  ¬(is_difference_of_squares 2006) ∧
  (is_difference_of_squares 2004) ∧
  (is_difference_of_squares 2005) ∧
  (is_difference_of_squares 2007) :=
sorry

end difference_of_squares_2006_l1202_120294


namespace max_value_and_constraint_optimization_l1202_120241

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2*|x + 1|

-- State the theorem
theorem max_value_and_constraint_optimization :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧ 
  m = 2 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + 2*b^2 + c^2 = m → 
    ab + bc ≤ 1 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀^2 + 2*b₀^2 + c₀^2 = m ∧ a₀*b₀ + b₀*c₀ = 1) :=
by
  sorry


end max_value_and_constraint_optimization_l1202_120241


namespace vector_representation_l1202_120229

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, -2)
def a : ℝ × ℝ := (-4, 0)

theorem vector_representation :
  a = (-1 : ℝ) • e₁ + (-1 : ℝ) • e₂ := by sorry

end vector_representation_l1202_120229


namespace lucas_1364_units_digit_l1202_120277

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Property that Lucas numbers' units digits repeat every 12 terms -/
axiom lucas_units_period (n : ℕ) : lucas n % 10 = lucas (n % 12) % 10

/-- L₁₅ equals 1364 -/
axiom L_15 : lucas 15 = 1364

/-- Theorem: The units digit of L₁₃₆₄ is 7 -/
theorem lucas_1364_units_digit : lucas 1364 % 10 = 7 := by
  sorry

end lucas_1364_units_digit_l1202_120277


namespace ice_cube_melting_l1202_120278

theorem ice_cube_melting (V : ℝ) : 
  V > 0 →
  (1/5) * (3/4) * (2/3) * (1/2) * V = 30 →
  V = 150 := by
sorry

end ice_cube_melting_l1202_120278


namespace gcf_of_36_and_60_l1202_120280

theorem gcf_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcf_of_36_and_60_l1202_120280


namespace r_fraction_of_total_l1202_120276

/-- Given a total amount of 6000 and r having 2400, prove that the fraction of the total amount that r has is 2/5 -/
theorem r_fraction_of_total (total : ℕ) (r_amount : ℕ) 
  (h_total : total = 6000) (h_r : r_amount = 2400) : 
  (r_amount : ℚ) / total = 2 / 5 := by
  sorry

end r_fraction_of_total_l1202_120276


namespace arithmetic_progression_implies_equal_numbers_l1202_120231

theorem arithmetic_progression_implies_equal_numbers 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_arithmetic : (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2 = (a + b) / 2) : 
  a = b := by
sorry

end arithmetic_progression_implies_equal_numbers_l1202_120231


namespace final_time_sum_l1202_120207

def initial_time : Nat := 15 * 60 * 60  -- 3:00:00 PM in seconds
def elapsed_time : Nat := 137 * 60 * 60 + 58 * 60 + 59  -- 137h 58m 59s in seconds

def final_time : Nat := (initial_time + elapsed_time) % (24 * 60 * 60)

def hours (t : Nat) : Nat := (t / 3600) % 12
def minutes (t : Nat) : Nat := (t / 60) % 60
def seconds (t : Nat) : Nat := t % 60

theorem final_time_sum :
  hours final_time + minutes final_time + seconds final_time = 125 := by
sorry

end final_time_sum_l1202_120207


namespace problem_statement_l1202_120217

theorem problem_statement (x Q : ℝ) (h : 2 * (5 * x + 3 * Real.pi) = Q) :
  4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 := by
  sorry

end problem_statement_l1202_120217


namespace find_divisor_l1202_120242

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 100)
  (h2 : quotient = 9)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 11 := by
sorry

end find_divisor_l1202_120242


namespace smartphone_savings_plan_l1202_120274

theorem smartphone_savings_plan (smartphone_cost initial_savings : ℕ) 
  (saving_months weeks_per_month : ℕ) : 
  smartphone_cost = 160 →
  initial_savings = 40 →
  saving_months = 2 →
  weeks_per_month = 4 →
  (smartphone_cost - initial_savings) / (saving_months * weeks_per_month) = 15 := by
sorry

end smartphone_savings_plan_l1202_120274


namespace max_eccentricity_ellipse_l1202_120281

/-- The maximum eccentricity of an ellipse with given properties -/
theorem max_eccentricity_ellipse :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let P : ℝ → ℝ × ℝ := λ x => (x, x + 3)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let c : ℝ := dist A B / 2
  let a (x : ℝ) : ℝ := (dist (P x) A + dist (P x) B) / 2
  let e (x : ℝ) : ℝ := c / a x
  ∃ (x : ℝ), ∀ (y : ℝ), e y ≤ e x ∧ e x = 2 * Real.sqrt 26 / 13 :=
sorry

end max_eccentricity_ellipse_l1202_120281


namespace arithmetic_equality_l1202_120270

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) - 100 = 0 := by
  sorry

end arithmetic_equality_l1202_120270


namespace petya_wins_l1202_120287

/-- Represents the game state -/
structure GameState where
  total_players : Nat
  vasya_turn : Bool

/-- Represents the result of the game -/
inductive GameResult
  | VasyaWins
  | PetyaWins

/-- Optimal play function -/
def optimal_play (state : GameState) : GameResult :=
  sorry

/-- The main theorem -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.total_players = 2022 →
    initial_state.vasya_turn = true →
    optimal_play initial_state = GameResult.PetyaWins :=
  sorry

end petya_wins_l1202_120287


namespace intersection_empty_condition_l1202_120200

theorem intersection_empty_condition (a : ℝ) : 
  let A : Set ℝ := Set.Iio (2 * a)
  let B : Set ℝ := Set.Ioi (3 - a^2)
  A ∩ B = ∅ → 2 * a ≤ 3 - a^2 :=
by sorry

end intersection_empty_condition_l1202_120200


namespace hezekiahs_age_l1202_120227

theorem hezekiahs_age (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  (ryanne_age = hezekiah_age + 7) → 
  (hezekiah_age + ryanne_age = 15) → 
  (hezekiah_age = 4) := by
sorry

end hezekiahs_age_l1202_120227


namespace orange_juice_profit_l1202_120204

-- Define the tree types and their properties
structure TreeType where
  name : String
  trees : ℕ
  orangesPerTree : ℕ
  pricePerCup : ℚ

-- Define the additional costs
def additionalCosts : ℚ := 180

-- Define the number of oranges needed to make one cup of juice
def orangesPerCup : ℕ := 3

-- Define the tree types
def valencia : TreeType := ⟨"Valencia", 150, 400, 4⟩
def navel : TreeType := ⟨"Navel", 120, 650, 9/2⟩
def bloodOrange : TreeType := ⟨"Blood Orange", 160, 500, 5⟩

-- Calculate profit for a single tree type
def calculateProfit (t : TreeType) : ℚ :=
  let totalOranges := t.trees * t.orangesPerTree
  let totalCups := totalOranges / orangesPerCup
  totalCups * t.pricePerCup - additionalCosts

-- Calculate total profit
def totalProfit : ℚ :=
  calculateProfit valencia + calculateProfit navel + calculateProfit bloodOrange

-- Theorem statement
theorem orange_juice_profit : totalProfit = 329795 := by
  sorry

end orange_juice_profit_l1202_120204


namespace dot_product_range_l1202_120275

-- Define the points O and A
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)

-- Define the set of points P on the right branch of the hyperbola
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 = 1 ∧ p.1 > 0}

-- Define the dot product of OA and OP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 + p.2

-- Theorem statement
theorem dot_product_range :
  ∀ p ∈ P, dot_product p > 0 ∧ ∀ M : ℝ, ∃ q ∈ P, dot_product q > M :=
by sorry

end dot_product_range_l1202_120275


namespace conference_games_count_l1202_120206

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def intra_conference_games : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ :=
  (num_teams.choose 2 * intra_conference_games) + (num_teams * non_conference_games)

theorem conference_games_count : total_games = 150 := by
  sorry

end conference_games_count_l1202_120206


namespace inequality_solution_set_l1202_120235

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ 
  x ∈ Set.Ici 5 ∪ Set.Iic (-6) ∪ Set.Icc (-2) 5 :=
sorry

end inequality_solution_set_l1202_120235


namespace total_oranges_picked_l1202_120230

/-- The total number of oranges picked by Joan and Sara -/
def total_oranges (joan_oranges sara_oranges : ℕ) : ℕ :=
  joan_oranges + sara_oranges

/-- Theorem: Given that Joan picked 37 oranges and Sara picked 10 oranges,
    the total number of oranges picked is 47 -/
theorem total_oranges_picked :
  total_oranges 37 10 = 47 := by
  sorry

end total_oranges_picked_l1202_120230


namespace spinner_area_l1202_120224

theorem spinner_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) 
  (h_r : r = 15)
  (h_p_win : p_win = 1/3)
  (h_p_bonus : p_bonus = 1/6) :
  p_win * π * r^2 + p_bonus * π * r^2 = 112.5 * π := by
  sorry

end spinner_area_l1202_120224


namespace shaded_area_percentage_l1202_120255

theorem shaded_area_percentage (square_side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * square_side_length - rectangle_length) * square_side_length) / (rectangle_width * rectangle_length)) * 100 = 14.29 := by
  sorry

end shaded_area_percentage_l1202_120255


namespace father_son_ages_l1202_120225

/-- Proves that given the conditions about the ages of a father and son, their present ages are 36 and 12 years respectively. -/
theorem father_son_ages (father_age son_age : ℕ) : 
  father_age = 3 * son_age ∧ 
  father_age + 12 = 2 * (son_age + 12) →
  father_age = 36 ∧ son_age = 12 := by
  sorry

end father_son_ages_l1202_120225


namespace smallest_values_for_equation_l1202_120258

theorem smallest_values_for_equation (a b c : ℤ) 
  (ha : a > 2) (hb : b < 10) (hc : c ≥ 0) 
  (heq : 32 = a + 2*b + 3*c) : 
  a = 4 ∧ b = 8 ∧ c = 4 := by
sorry

end smallest_values_for_equation_l1202_120258


namespace tenth_group_sample_l1202_120218

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- The number drawn from a specific group in systematic sampling -/
def group_sample (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + s.first_sample

/-- Theorem stating the relationship between samples from different groups -/
theorem tenth_group_sample (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : group_sample s 5 = 22) :
  group_sample s 10 = 47 := by
  sorry

#check tenth_group_sample

end tenth_group_sample_l1202_120218


namespace tan_theta_in_terms_of_x_l1202_120273

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_x : x > 1)
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end tan_theta_in_terms_of_x_l1202_120273


namespace evaluate_y_l1202_120211

theorem evaluate_y (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) - 2 = |x - 2| + |x + 3| - 2 :=
by sorry

end evaluate_y_l1202_120211


namespace line_circle_intersection_l1202_120291

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) :=
by sorry

end line_circle_intersection_l1202_120291


namespace kittens_sold_count_l1202_120285

/-- Represents the pet store's sales scenario -/
structure PetStoreSales where
  kitten_price : ℕ
  puppy_price : ℕ
  total_revenue : ℕ
  puppy_count : ℕ

/-- Calculates the number of kittens sold -/
def kittens_sold (s : PetStoreSales) : ℕ :=
  (s.total_revenue - s.puppy_price * s.puppy_count) / s.kitten_price

/-- Theorem stating the number of kittens sold -/
theorem kittens_sold_count (s : PetStoreSales) 
  (h1 : s.kitten_price = 6)
  (h2 : s.puppy_price = 5)
  (h3 : s.total_revenue = 17)
  (h4 : s.puppy_count = 1) :
  kittens_sold s = 2 := by
  sorry

end kittens_sold_count_l1202_120285


namespace books_read_during_travel_l1202_120290

theorem books_read_during_travel (total_distance : ℝ) (distance_per_book : ℝ) : 
  total_distance = 6987.5 → 
  distance_per_book = 482.3 → 
  ⌊total_distance / distance_per_book⌋ = 14 := by
sorry

end books_read_during_travel_l1202_120290


namespace triangle_area_theorem_l1202_120245

theorem triangle_area_theorem (A B C : Real) (a b c : Real) :
  c = 2 →
  C = π / 3 →
  let m : Real × Real := (Real.sin C + Real.sin (B - A), 4)
  let n : Real × Real := (Real.sin (2 * A), 1)
  (∃ (k : Real), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  let S := (1 / 2) * a * c * Real.sin B
  (S = 4 * Real.sqrt 13 / 13 ∨ S = 2 * Real.sqrt 3 / 3) :=
by sorry

end triangle_area_theorem_l1202_120245


namespace beef_stew_duration_l1202_120299

/-- The number of days the beef stew lasts for 2 people -/
def days_for_two : ℝ := 7

/-- The number of days the beef stew lasts for 5 people -/
def days_for_five : ℝ := 2.8

/-- The number of people in the original scenario -/
def original_people : ℕ := 2

/-- The number of people in the new scenario -/
def new_people : ℕ := 5

theorem beef_stew_duration :
  days_for_two * original_people = days_for_five * new_people :=
by sorry

end beef_stew_duration_l1202_120299


namespace union_equals_S_l1202_120292

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem union_equals_S : S ∪ T = S := by sorry

end union_equals_S_l1202_120292


namespace point_on_line_for_all_k_l1202_120222

/-- The point (-2, -1) lies on the line kx + y + 2k + 1 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

end point_on_line_for_all_k_l1202_120222


namespace initial_rows_count_l1202_120226

theorem initial_rows_count (chairs_per_row : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) 
  (h1 : chairs_per_row = 12)
  (h2 : extra_chairs = 11)
  (h3 : total_chairs = 95) :
  ∃ (initial_rows : ℕ), initial_rows * chairs_per_row + extra_chairs = total_chairs ∧ initial_rows = 7 := by
  sorry

end initial_rows_count_l1202_120226


namespace matthew_hotdogs_l1202_120297

/-- The number of hotdogs Matthew needs to cook for his children -/
def total_hotdogs : ℕ :=
  let ella_emma_hotdogs := 2 + 2
  let luke_hotdogs := 2 * ella_emma_hotdogs
  let hunter_hotdogs := (3 * ella_emma_hotdogs) / 2
  ella_emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end matthew_hotdogs_l1202_120297


namespace mod_twelve_five_eleven_l1202_120249

theorem mod_twelve_five_eleven (m : ℕ) : 
  12^5 ≡ m [ZMOD 11] → 0 ≤ m → m < 11 → m = 1 := by
  sorry

end mod_twelve_five_eleven_l1202_120249


namespace field_trip_buses_l1202_120254

/-- The number of students in the school -/
def total_students : ℕ := 11210

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of school buses needed for the field trip -/
def buses_needed : ℕ := total_students / seats_per_bus

theorem field_trip_buses : buses_needed = 95 := by
  sorry

end field_trip_buses_l1202_120254


namespace parabola_focus_distance_l1202_120252

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h : A.y = Real.sqrt 2) :
  (3 * A.x = A.x + para.p / 2) → para.p = 2 := by
  sorry

end parabola_focus_distance_l1202_120252


namespace greatest_four_digit_multiple_l1202_120243

theorem greatest_four_digit_multiple : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (15 ∣ m) ∧ (25 ∣ m) ∧ (40 ∣ m) ∧ (75 ∣ m) → m ≤ n) ∧
  n = 9600 :=
by sorry

end greatest_four_digit_multiple_l1202_120243
