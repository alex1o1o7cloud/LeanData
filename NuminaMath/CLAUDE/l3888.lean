import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_x_l3888_388880

theorem solve_for_x (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 21) : x = 56 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3888_388880


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l3888_388874

theorem multiplication_value_proof (x : ℝ) : (7.5 / 6) * x = 15 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l3888_388874


namespace NUMINAMATH_CALUDE_base_equation_solution_l3888_388818

/-- Represents a number in a given base -/
def to_base (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Consecutive even positive integers -/
def consecutive_even (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ Even x ∧ Even y ∧ y = x + 2

theorem base_equation_solution (X Y : ℕ) :
  consecutive_even X Y →
  to_base 241 X + to_base 36 Y = to_base 94 (X + Y) →
  X + Y = 22 := by sorry

end NUMINAMATH_CALUDE_base_equation_solution_l3888_388818


namespace NUMINAMATH_CALUDE_triangle_properties_l3888_388893

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) 
  (h4 : t.B = Real.pi / 3) : 
  (t.c = Real.sqrt 13) ∧ 
  (-Real.sqrt 3 < 2 * t.c - t.a) ∧ 
  (2 * t.c - t.a < 2 * Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3888_388893


namespace NUMINAMATH_CALUDE_john_number_theorem_l3888_388811

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_theorem :
  ∃! x : ℕ, is_two_digit x ∧
    84 ≤ switch_digits (5 * x - 7) ∧
    switch_digits (5 * x - 7) ≤ 90 ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_number_theorem_l3888_388811


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3888_388868

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 > 1 ∧ -2 * x ≤ 4) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3888_388868


namespace NUMINAMATH_CALUDE_sum_of_digits_of_number_l3888_388888

/-- The sum of the digits of 10^100 - 57 -/
def sum_of_digits : ℕ := 889

/-- The number we're considering -/
def number : ℕ := 10^100 - 57

/-- Theorem stating that the sum of the digits of our number is equal to sum_of_digits -/
theorem sum_of_digits_of_number : 
  (number.digits 10).sum = sum_of_digits := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_number_l3888_388888


namespace NUMINAMATH_CALUDE_intersection_S_T_l3888_388829

def S : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}
def T : Set ℝ := {x : ℝ | x + 2 ≤ 3}

theorem intersection_S_T : S ∩ T = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3888_388829


namespace NUMINAMATH_CALUDE_line_equation_l3888_388876

/-- A line with slope -2 and sum of x and y intercepts equal to 12 has the general equation 2x + y - 8 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (intercept_sum : ℝ) : 
  slope = -2 →
  intercept_sum = 12 →
  ∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -8 ∧
  l = {(x, y) | a * x + b * y + c = 0} :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3888_388876


namespace NUMINAMATH_CALUDE_three_player_cooperation_strategy_l3888_388853

/-- Represents the dimensions of the game board -/
def boardSize : Nat := 1000

/-- Represents the possible rectangle shapes that can be painted -/
inductive Rectangle
  | twoByOne
  | oneByTwo
  | oneByThree
  | threeByOne

/-- Represents a player in the game -/
inductive Player
  | Andy
  | Bess
  | Charley
  | Dick

/-- Represents a position on the board -/
structure Position where
  x : Fin boardSize
  y : Fin boardSize

/-- Represents a move in the game -/
structure Move where
  player : Player
  rectangle : Rectangle
  position : Position

/-- The game state -/
structure GameState where
  board : Fin boardSize → Fin boardSize → Bool
  currentPlayer : Player

/-- Function to check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Bool := sorry

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Function to check if a player has a valid move -/
def hasValidMove (state : GameState) (player : Player) : Bool := sorry

/-- Theorem: There exists a strategy for three players to make the fourth player lose -/
theorem three_player_cooperation_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      ∃ (losingPlayer : Player),
        ¬(hasValidMove (applyMove initialState (strategy initialState)) losingPlayer) :=
sorry

end NUMINAMATH_CALUDE_three_player_cooperation_strategy_l3888_388853


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3888_388828

/-- Represents the state of a dandelion -/
inductive DandelionState
  | Yellow
  | White
  | Dispersed

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle (openDay : Day) (currentDay : Day) : DandelionState :=
  sorry

/-- Counts the number of dandelions in a specific state on a given day -/
def countDandelions (day : Day) (state : DandelionState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem white_dandelions_on_saturday :
  (countDandelions Day.Monday DandelionState.Yellow = 20) →
  (countDandelions Day.Monday DandelionState.White = 14) →
  (countDandelions Day.Wednesday DandelionState.Yellow = 15) →
  (countDandelions Day.Wednesday DandelionState.White = 11) →
  (countDandelions Day.Saturday DandelionState.White = 6) :=
by sorry

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l3888_388828


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_condition_l3888_388836

theorem quadratic_two_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_condition_l3888_388836


namespace NUMINAMATH_CALUDE_students_with_no_books_l3888_388810

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : ℕ
  one : ℕ
  two : ℕ
  threeOrMore : ℕ

/-- The total number of students in the class -/
def totalStudents : ℕ := 40

/-- The average number of books borrowed per student -/
def averageBooks : ℚ := 2

/-- Calculates the total number of books borrowed -/
def totalBooksBorrowed (b : BookBorrowers) : ℕ :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating the number of students who did not borrow books -/
theorem students_with_no_books (b : BookBorrowers) : 
  b.zero = 1 ∧ 
  b.one = 12 ∧ 
  b.two = 13 ∧ 
  b.zero + b.one + b.two + b.threeOrMore = totalStudents ∧
  (totalBooksBorrowed b : ℚ) / totalStudents = averageBooks :=
by
  sorry


end NUMINAMATH_CALUDE_students_with_no_books_l3888_388810


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3888_388835

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ,
  x - y = 5 →
  x + 2*y = 2 →
  (x^2 - 4*x*y + 4*y^2) / (x^2 - x*y) / (x + y - 3*y^2 / (x - y)) + 1/x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3888_388835


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l3888_388820

/-- Given a right-angled triangle with legs of 5 and 12 steps, 
    the probability that a randomly selected point within the triangle 
    lies within its inscribed circle is 2π/15 -/
theorem inscribed_circle_probability (a b : ℝ) (h1 : a = 5) (h2 : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let triangle_area := a * b / 2
  let circle_area := π * r^2
  circle_area / triangle_area = 2 * π / 15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l3888_388820


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l3888_388897

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)}

theorem integer_fraction_pairs :
  {p : ℕ+ × ℕ+ | is_integer_fraction p.1 p.2} = solution_set :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l3888_388897


namespace NUMINAMATH_CALUDE_games_from_friend_l3888_388845

theorem games_from_friend (games_from_garage_sale : ℕ) 
  (non_working_games : ℕ) (good_games : ℕ) : ℕ :=
  by
  have h1 : games_from_garage_sale = 8 := by sorry
  have h2 : non_working_games = 23 := by sorry
  have h3 : good_games = 6 := by sorry
  
  let total_games := non_working_games + good_games
  
  have h4 : total_games = 29 := by sorry
  
  let games_from_friend := total_games - games_from_garage_sale
  
  have h5 : games_from_friend = 21 := by sorry
  
  exact games_from_friend

end NUMINAMATH_CALUDE_games_from_friend_l3888_388845


namespace NUMINAMATH_CALUDE_floor_area_from_partial_coverage_l3888_388809

/-- The total area of a floor given a carpet covering a known percentage -/
theorem floor_area_from_partial_coverage (carpet_area : ℝ) (coverage_percentage : ℝ) 
  (h1 : carpet_area = 36) 
  (h2 : coverage_percentage = 0.45) : 
  carpet_area / coverage_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_floor_area_from_partial_coverage_l3888_388809


namespace NUMINAMATH_CALUDE_division_remainder_l3888_388859

theorem division_remainder : ∃ (q r : ℕ), 1620 = (1620 - 1365) * q + r ∧ r < (1620 - 1365) ∧ r = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3888_388859


namespace NUMINAMATH_CALUDE_dress_price_difference_l3888_388863

theorem dress_price_difference (P : ℝ) (h : P - 0.15 * P = 68) :
  P - (68 + 0.25 * 68) = -5 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_difference_l3888_388863


namespace NUMINAMATH_CALUDE_total_trophies_in_three_years_l3888_388846

theorem total_trophies_in_three_years :
  let michael_current_trophies : ℕ := 30
  let michael_trophy_increase : ℕ := 100
  let jack_trophy_multiplier : ℕ := 10
  let michael_future_trophies : ℕ := michael_current_trophies + michael_trophy_increase
  let jack_future_trophies : ℕ := jack_trophy_multiplier * michael_current_trophies
  michael_future_trophies + jack_future_trophies = 430 := by
sorry

end NUMINAMATH_CALUDE_total_trophies_in_three_years_l3888_388846


namespace NUMINAMATH_CALUDE_power_multiplication_l3888_388884

theorem power_multiplication (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3888_388884


namespace NUMINAMATH_CALUDE_veridux_male_associates_l3888_388813

/-- Proves the number of male associates at Veridux Corporation --/
theorem veridux_male_associates :
  let total_employees : ℕ := 250
  let female_employees : ℕ := 90
  let total_managers : ℕ := 40
  let female_managers : ℕ := 40
  let male_employees : ℕ := total_employees - female_employees
  let male_associates : ℕ := male_employees
  male_associates = 160 := by
  sorry

end NUMINAMATH_CALUDE_veridux_male_associates_l3888_388813


namespace NUMINAMATH_CALUDE_panda_babies_born_l3888_388865

/-- The number of panda babies born in a zoo with given conditions -/
theorem panda_babies_born (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 →
  pregnancy_rate = 1/4 →
  (total_pandas / 2 : ℚ) * pregnancy_rate * 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_panda_babies_born_l3888_388865


namespace NUMINAMATH_CALUDE_remainder_theorem_l3888_388851

theorem remainder_theorem (k : ℤ) : (1125 * 1127 * (12 * k + 1)) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3888_388851


namespace NUMINAMATH_CALUDE_coin_division_problem_l3888_388885

theorem coin_division_problem (n : ℕ) : 
  (n > 0 ∧ 
   n % 8 = 7 ∧ 
   n % 7 = 5 ∧ 
   ∀ m : ℕ, (m > 0 ∧ m % 8 = 7 ∧ m % 7 = 5) → n ≤ m) →
  (n = 47 ∧ n % 9 = 2) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3888_388885


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3888_388830

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3888_388830


namespace NUMINAMATH_CALUDE_all_terms_even_l3888_388870

theorem all_terms_even (m n : ℤ) (hm : Even m) (hn : Even n) :
  ∀ k : Fin 9, Even ((Finset.range 9).sum (λ i => (Nat.choose 8 i : ℤ) * m^(8 - i) * n^i)) := by
  sorry

end NUMINAMATH_CALUDE_all_terms_even_l3888_388870


namespace NUMINAMATH_CALUDE_first_day_over_200_paperclips_l3888_388861

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_over_200_paperclips :
  (∀ j : ℕ, j < 8 → paperclips j ≤ 200) ∧ paperclips 8 > 200 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_200_paperclips_l3888_388861


namespace NUMINAMATH_CALUDE_smallest_c_for_cosine_zero_l3888_388826

theorem smallest_c_for_cosine_zero (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (∀ x : ℝ, x < 0 → a * Real.cos (b * x + c) ≠ 0) →
  a * Real.cos c = 0 →
  c ≥ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_cosine_zero_l3888_388826


namespace NUMINAMATH_CALUDE_expression_simplification_l3888_388808

variable (a b : ℝ)

theorem expression_simplification :
  (2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a) ∧
  (2/3*(2*a - b) + 2*(b - 2*a) - 3*(2*a - b) - 4/3*(b - 2*a) = -6*a + 3*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3888_388808


namespace NUMINAMATH_CALUDE_geometric_progression_sum_relation_l3888_388875

theorem geometric_progression_sum_relation 
  (a : ℝ) (p q : ℝ) (S S₁ : ℝ) 
  (ha : 0 < a ∧ a < 1) (hp : p > 0) (hq : q > 0)
  (hS : S = (1 - a^p)⁻¹) (hS₁ : S₁ = (1 - a^q)⁻¹) :
  S^q * (S₁ - 1)^p = S₁^p * (S - 1)^q := by sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_relation_l3888_388875


namespace NUMINAMATH_CALUDE_sandbox_length_l3888_388817

theorem sandbox_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 146 → area = 45552 → length * width = area → length = 312 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_length_l3888_388817


namespace NUMINAMATH_CALUDE_joey_study_time_l3888_388858

/-- Calculates the total study time for Joey's SAT exam -/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) 
    * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam -/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_joey_study_time_l3888_388858


namespace NUMINAMATH_CALUDE_mouse_breeding_problem_l3888_388894

theorem mouse_breeding_problem (initial_mice : ℕ) (first_round_pups : ℕ) (eaten_pups : ℕ) (final_mice : ℕ) :
  initial_mice = 8 →
  first_round_pups = 6 →
  eaten_pups = 2 →
  final_mice = 280 →
  ∃ (second_round_pups : ℕ),
    final_mice = initial_mice + initial_mice * first_round_pups +
      (initial_mice + initial_mice * first_round_pups) * second_round_pups -
      (initial_mice + initial_mice * first_round_pups) * eaten_pups ∧
    second_round_pups = 6 :=
by sorry

end NUMINAMATH_CALUDE_mouse_breeding_problem_l3888_388894


namespace NUMINAMATH_CALUDE_expression_evaluation_l3888_388821

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3888_388821


namespace NUMINAMATH_CALUDE_compare_a_b_fraction_inequality_l3888_388838

-- Problem 1
theorem compare_a_b (m n : ℝ) :
  (m^2 + 1) * (n^2 + 4) ≥ (m * n + 2)^2 := by sorry

-- Problem 2
theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) (h5 : e > 0) :
  e / (a - c) < e / (b - d) := by sorry

end NUMINAMATH_CALUDE_compare_a_b_fraction_inequality_l3888_388838


namespace NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l3888_388805

theorem convex_ngon_angle_theorem (n : ℕ) : 
  (n ≥ 3) →  -- n-gon must have at least 3 sides
  (∃ (x : ℝ), x > 0 ∧ x < 150 ∧ 150 * (n - 1) + x = 180 * (n - 2)) →
  (n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11) := by
sorry

end NUMINAMATH_CALUDE_convex_ngon_angle_theorem_l3888_388805


namespace NUMINAMATH_CALUDE_trapezoid_AB_length_l3888_388896

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  AB : ℝ
  -- Length of side CD
  CD : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Condition: The ratio of areas is 5:2
  area_ratio_condition : area_ratio = 5 / 2
  -- Condition: The sum of AB and CD is 280
  sum_sides : AB + CD = 280

/-- Theorem stating that under given conditions, AB = 200 -/
theorem trapezoid_AB_length (t : Trapezoid) : t.AB = 200 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_AB_length_l3888_388896


namespace NUMINAMATH_CALUDE_optimal_time_correct_l3888_388839

/-- The optimal time for Vasya and Petya to cover the distance -/
def optimal_time : ℝ := 0.5

/-- The total distance to be covered -/
def total_distance : ℝ := 3

/-- Vasya's running speed -/
def vasya_run_speed : ℝ := 4

/-- Vasya's skating speed -/
def vasya_skate_speed : ℝ := 8

/-- Petya's running speed -/
def petya_run_speed : ℝ := 5

/-- Petya's skating speed -/
def petya_skate_speed : ℝ := 10

/-- Theorem stating that the optimal time is correct -/
theorem optimal_time_correct :
  ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ total_distance ∧
    (x / vasya_skate_speed + (total_distance - x) / vasya_run_speed = optimal_time) ∧
    ((total_distance - x) / petya_skate_speed + x / petya_run_speed = optimal_time) ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance →
      max (y / vasya_skate_speed + (total_distance - y) / vasya_run_speed)
          ((total_distance - y) / petya_skate_speed + y / petya_run_speed) ≥ optimal_time :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_time_correct_l3888_388839


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l3888_388852

theorem sin_cos_sum_equals_sqrt_three_half : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (130 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_three_half_l3888_388852


namespace NUMINAMATH_CALUDE_sequence_range_l3888_388871

-- Define the sequence a_n
def a (n : ℕ+) (p : ℝ) : ℝ := 2 * (n : ℝ)^2 + p * (n : ℝ)

-- State the theorem
theorem sequence_range (p : ℝ) :
  (∀ n : ℕ+, a n p < a (n + 1) p) ↔ p > -6 :=
sorry

end NUMINAMATH_CALUDE_sequence_range_l3888_388871


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3888_388841

/-- Given an elevator with 6 people and an average weight of 152 lbs, 
    prove that when a new person weighing 145 lbs enters, 
    the new average weight of all 7 people is 151 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) (new_avg_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 152 →
  new_person_weight = 145 →
  new_avg_weight = (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) →
  new_avg_weight = 151 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3888_388841


namespace NUMINAMATH_CALUDE_q_is_false_l3888_388827

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end NUMINAMATH_CALUDE_q_is_false_l3888_388827


namespace NUMINAMATH_CALUDE_difference_of_squares_l3888_388860

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 75) (h2 : a - b = 15) :
  a^2 - b^2 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3888_388860


namespace NUMINAMATH_CALUDE_integer_quadruple_solution_l3888_388886

theorem integer_quadruple_solution :
  ∃! (S : Set (ℕ × ℕ × ℕ × ℕ)),
    S.Nonempty ∧
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔
      (1 < a ∧ a < b ∧ b < c ∧ c < d) ∧
      (∃ k : ℕ, a * b * c * d - 1 = k * ((a - 1) * (b - 1) * (c - 1) * (d - 1)))) ∧
    S = {(3, 5, 17, 255), (2, 4, 10, 80)} :=
by sorry

end NUMINAMATH_CALUDE_integer_quadruple_solution_l3888_388886


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3888_388843

theorem product_from_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → Nat.gcd a b = 6 → a * b = 432 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3888_388843


namespace NUMINAMATH_CALUDE_boys_in_school_l3888_388848

/-- The number of boys in a school, given the initial number of girls, 
    the number of new girls who joined, and the total number of pupils after new girls joined. -/
def number_of_boys (initial_girls new_girls total_pupils : ℕ) : ℕ :=
  total_pupils - (initial_girls + new_girls)

/-- Theorem stating that the number of boys in the school is 222 -/
theorem boys_in_school : number_of_boys 706 418 1346 = 222 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_school_l3888_388848


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3888_388803

-- Define the universe set U
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Finset ℕ := {2, 4, 6}

-- Define set B
def B : Finset ℕ := {1, 3, 5, 7}

-- Theorem statement
theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3888_388803


namespace NUMINAMATH_CALUDE_complex_point_on_line_l3888_388899

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I)⁻¹
  (z.im = 2 * z.re) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l3888_388899


namespace NUMINAMATH_CALUDE_clara_climbs_96_blocks_l3888_388877

/-- The number of stone blocks Clara climbs past in the historical tower -/
def total_blocks (levels : ℕ) (steps_per_level : ℕ) (blocks_per_step : ℕ) : ℕ :=
  levels * steps_per_level * blocks_per_step

/-- Theorem stating that Clara climbs past 96 blocks of stone -/
theorem clara_climbs_96_blocks :
  total_blocks 4 8 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_clara_climbs_96_blocks_l3888_388877


namespace NUMINAMATH_CALUDE_grocery_store_lite_soda_l3888_388866

/-- Given a grocery store with soda bottles, proves that the number of lite soda bottles is 60 -/
theorem grocery_store_lite_soda (regular : ℕ) (diet : ℕ) (lite : ℕ) 
  (h1 : regular = 81)
  (h2 : diet = 60)
  (h3 : diet = lite) : 
  lite = 60 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_lite_soda_l3888_388866


namespace NUMINAMATH_CALUDE_gardener_work_theorem_l3888_388802

/-- Represents the outcome of the gardener's work. -/
structure GardenerOutcome where
  diligentDays : ℕ
  shirkingDays : ℕ

/-- Calculates the pretzel balance based on the gardener's work outcome. -/
def pretzelBalance (outcome : GardenerOutcome) : ℤ :=
  (3 * outcome.diligentDays) - outcome.shirkingDays

theorem gardener_work_theorem :
  ∃ (outcome : GardenerOutcome),
    outcome.diligentDays + outcome.shirkingDays = 26 ∧
    pretzelBalance outcome = 62 ∧
    outcome.diligentDays = 22 ∧
    outcome.shirkingDays = 4 := by
  sorry

#check gardener_work_theorem

end NUMINAMATH_CALUDE_gardener_work_theorem_l3888_388802


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l3888_388879

theorem power_fraction_evaluation :
  ((5^2014)^2 - (5^2012)^2) / ((5^2013)^2 - (5^2011)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l3888_388879


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3888_388878

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3888_388878


namespace NUMINAMATH_CALUDE_functional_equation_identity_l3888_388840

open Function Real

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) →
  (∀ y : ℝ, f y = y) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l3888_388840


namespace NUMINAMATH_CALUDE_chris_age_is_17_l3888_388882

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℕ
  ben : ℕ
  chris : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Six years ago, Chris was the same age as Amy is now
  ages.chris - 6 = ages.amy ∧
  -- In 3 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 3 = (3 * (ages.amy + 3)) / 4

/-- The theorem stating that Chris's age is 17 -/
theorem chris_age_is_17 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.chris = 17 := by
  sorry


end NUMINAMATH_CALUDE_chris_age_is_17_l3888_388882


namespace NUMINAMATH_CALUDE_peanut_seed_sprouting_probability_l3888_388862

/-- The probability of exactly k successes in n independent trials,
    where p is the probability of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem peanut_seed_sprouting_probability :
  let n : ℕ := 3  -- total number of seeds
  let k : ℕ := 2  -- number of seeds we want to sprout
  let p : ℝ := 3/5  -- probability of each seed sprouting
  binomial_probability n k p = 54/125 := by
sorry

end NUMINAMATH_CALUDE_peanut_seed_sprouting_probability_l3888_388862


namespace NUMINAMATH_CALUDE_smallest_age_difference_l3888_388800

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 0 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- Calculates the value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Reverses the digits of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  is_valid := by
    simp [n.is_valid]

/-- The difference between two natural numbers -/
def diff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

theorem smallest_age_difference :
  ∀ (mrs_age : TwoDigitNumber),
    diff (TwoDigitNumber.value mrs_age) (TwoDigitNumber.value (TwoDigitNumber.reverse mrs_age)) ≥ 9 ∧
    ∃ (age : TwoDigitNumber),
      diff (TwoDigitNumber.value age) (TwoDigitNumber.value (TwoDigitNumber.reverse age)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_age_difference_l3888_388800


namespace NUMINAMATH_CALUDE_worker_wage_increase_l3888_388857

theorem worker_wage_increase (original_wage : ℝ) : 
  (original_wage * 1.5 = 42) → original_wage = 28 := by
  sorry

end NUMINAMATH_CALUDE_worker_wage_increase_l3888_388857


namespace NUMINAMATH_CALUDE_symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l3888_388855

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Statement 1
theorem symmetry_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

-- Statement 2
theorem symmetry_f_and_f_swap :
  ∀ x y, y = f x ↔ x = f y :=
sorry

-- Statement 4
theorem same_curve_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

end NUMINAMATH_CALUDE_symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l3888_388855


namespace NUMINAMATH_CALUDE_min_phase_shift_l3888_388842

theorem min_phase_shift (x φ : ℝ) : 
  (∀ x, 2 * Real.sin (x + π/6 - φ) = 2 * Real.sin (x - π/3)) →
  (φ > 0 → φ ≥ π/2) ∧ 
  (∃ φ₀ > 0, ∀ x, 2 * Real.sin (x + π/6 - φ₀) = 2 * Real.sin (x - π/3) ∧ φ₀ = π/2) := by
  sorry

#check min_phase_shift

end NUMINAMATH_CALUDE_min_phase_shift_l3888_388842


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3888_388869

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3888_388869


namespace NUMINAMATH_CALUDE_y_derivative_l3888_388814

noncomputable def y (x : ℝ) : ℝ := 4 * Real.arcsin (4 / (2 * x + 3)) + Real.sqrt (4 * x^2 + 12 * x - 7)

theorem y_derivative (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (2 * Real.sqrt (4 * x^2 + 12 * x - 7)) / (2 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3888_388814


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3888_388831

theorem complex_product_magnitude : Complex.abs ((20 - 15 * Complex.I) * (12 + 25 * Complex.I)) = 25 * Real.sqrt 769 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3888_388831


namespace NUMINAMATH_CALUDE_percentage_difference_l3888_388873

theorem percentage_difference (x : ℝ) : x = 35 → (0.8 * 170) - (x / 100 * 300) = 31 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3888_388873


namespace NUMINAMATH_CALUDE_distance_between_points_l3888_388812

def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (2, 5)

theorem distance_between_points : Real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3888_388812


namespace NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l3888_388832

/-- Given a circle with polar equation ρ² - 2ρcosθ + 4ρsinθ + 4 = 0, its radius is 1 -/
theorem circle_radius_from_polar_equation :
  ∀ ρ θ : ℝ,
  ρ^2 - 2*ρ*(Real.cos θ) + 4*ρ*(Real.sin θ) + 4 = 0 →
  ∃ x y : ℝ,
  (x - 1)^2 + (y + 2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l3888_388832


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3888_388887

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3888_388887


namespace NUMINAMATH_CALUDE_sum_of_y_values_l3888_388806

/-- Given 5 experiments with x and y values, prove the sum of y values -/
theorem sum_of_y_values
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)
  (sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150)
  (regression_eq : ∀ x y, y = 0.67 * x + 54.9) :
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l3888_388806


namespace NUMINAMATH_CALUDE_ripe_oranges_calculation_l3888_388807

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 52

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

theorem ripe_oranges_calculation :
  ripe_oranges_per_day * harvest_days + unripe_oranges_per_day * harvest_days = total_oranges :=
by sorry

end NUMINAMATH_CALUDE_ripe_oranges_calculation_l3888_388807


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3888_388864

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (3 + 2 * Real.sqrt 2)) :
  ((1 - x)^2 / (x - 1)) + (Real.sqrt (x^2 + 4 - 4*x) / (x - 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3888_388864


namespace NUMINAMATH_CALUDE_book_cost_price_l3888_388833

theorem book_cost_price (cost : ℝ) : 
  (cost * 1.18 - cost * 1.12 = 18) → cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3888_388833


namespace NUMINAMATH_CALUDE_first_day_over_200_acorns_l3888_388881

/-- Represents the number of acorns Mark has on a given day -/
def acorns (k : ℕ) : ℕ := 5 * 5^k - 2 * k

/-- Represents the day of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Converts a natural number to a day of the week -/
def toDay (n : ℕ) : Day :=
  match n % 7 with
  | 0 => Day.Monday
  | 1 => Day.Tuesday
  | 2 => Day.Wednesday
  | 3 => Day.Thursday
  | 4 => Day.Friday
  | 5 => Day.Saturday
  | _ => Day.Sunday

theorem first_day_over_200_acorns :
  ∀ k : ℕ, k < 3 → acorns k ≤ 200 ∧
  acorns 3 > 200 ∧
  toDay 3 = Day.Thursday :=
sorry

end NUMINAMATH_CALUDE_first_day_over_200_acorns_l3888_388881


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l3888_388801

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l3888_388801


namespace NUMINAMATH_CALUDE_angle_CDE_is_right_angle_l3888_388825

theorem angle_CDE_is_right_angle 
  (angle_A angle_B angle_C : Real)
  (angle_AEB angle_BED angle_BDE : Real)
  (h1 : angle_A = 90)
  (h2 : angle_B = 90)
  (h3 : angle_C = 90)
  (h4 : angle_AEB = 50)
  (h5 : angle_BED = 40)
  (h6 : angle_BDE = 50)
  : ∃ (angle_CDE : Real), angle_CDE = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_CDE_is_right_angle_l3888_388825


namespace NUMINAMATH_CALUDE_set_inclusion_iff_range_l3888_388822

/-- Given sets A and B, prove that (ℝ \ B) ⊆ A if and only if a ≤ -2 or 1/2 ≤ a < 1 -/
theorem set_inclusion_iff_range (a : ℝ) : 
  let A : Set ℝ := {x | x < -1 ∨ x ≥ 1}
  let B : Set ℝ := {x | x ≤ 2*a ∨ x ≥ a+1}
  (Set.univ \ B) ⊆ A ↔ a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_range_l3888_388822


namespace NUMINAMATH_CALUDE_gcd_2025_2070_l3888_388816

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2025_2070_l3888_388816


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_less_than_one_l3888_388854

theorem negation_of_existence_squared_less_than_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_less_than_one_l3888_388854


namespace NUMINAMATH_CALUDE_salt_merchant_problem_l3888_388892

/-- The salt merchant problem -/
theorem salt_merchant_problem (x y : ℝ) (a : ℝ) 
  (h1 : a * (y - x) = 100)  -- Profit from first transaction
  (h2 : a * y * (y / x - 1) = 120)  -- Profit from second transaction
  (h3 : x > 0)  -- Price in Tver is positive
  (h4 : y > x)  -- Price in Moscow is higher than in Tver
  : a * x = 500 := by
  sorry

end NUMINAMATH_CALUDE_salt_merchant_problem_l3888_388892


namespace NUMINAMATH_CALUDE_car_distance_formula_l3888_388856

/-- The distance traveled by a car after time t -/
def distance (t : ℝ) : ℝ :=
  10 + 60 * t

/-- The initial distance traveled by the car -/
def initial_distance : ℝ := 10

/-- The constant speed of the car after the initial distance -/
def speed : ℝ := 60

theorem car_distance_formula (t : ℝ) :
  distance t = initial_distance + speed * t :=
by sorry

end NUMINAMATH_CALUDE_car_distance_formula_l3888_388856


namespace NUMINAMATH_CALUDE_cross_placements_count_l3888_388895

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Represents a rectangle --/
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a cross shape --/
structure Cross :=
  (size : ℕ)

/-- Function to calculate the number of ways to place a cross in a grid with a rectangle removed --/
def count_cross_placements (g : Grid) (r : Rectangle) (c : Cross) : ℕ :=
  sorry

/-- Theorem stating the number of ways to place a 5-cell cross in a 40x40 grid with a 36x37 rectangle removed --/
theorem cross_placements_count :
  let g := Grid.mk 40
  let r := Rectangle.mk 36 37
  let c := Cross.mk 5
  count_cross_placements g r c = 113 := by
  sorry

end NUMINAMATH_CALUDE_cross_placements_count_l3888_388895


namespace NUMINAMATH_CALUDE_expression_equality_l3888_388823

theorem expression_equality : 49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1 = 254804368 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3888_388823


namespace NUMINAMATH_CALUDE_distance_between_stations_l3888_388891

/-- The distance between two stations given three cars with different speeds --/
theorem distance_between_stations (speed_A speed_B speed_C : ℝ) (time_diff : ℝ) : 
  speed_A = 90 →
  speed_B = 80 →
  speed_C = 60 →
  time_diff = 1/3 →
  (speed_A + speed_B) * ((speed_A + speed_C) * time_diff / (speed_B - speed_C)) = 425 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_stations_l3888_388891


namespace NUMINAMATH_CALUDE_floor_difference_l3888_388847

theorem floor_difference : ⌊(-2.7 : ℝ)⌋ - ⌊(4.5 : ℝ)⌋ = -7 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_l3888_388847


namespace NUMINAMATH_CALUDE_well_digging_rate_l3888_388834

/-- The hourly rate paid to workers for digging a well --/
def hourly_rate (total_payment : ℚ) (num_workers : ℕ) (hours_day1 hours_day2 hours_day3 : ℕ) : ℚ :=
  total_payment / (num_workers * (hours_day1 + hours_day2 + hours_day3))

/-- Theorem stating that under the given conditions, the hourly rate is $10 --/
theorem well_digging_rate : 
  hourly_rate 660 2 10 8 15 = 10 := by
  sorry


end NUMINAMATH_CALUDE_well_digging_rate_l3888_388834


namespace NUMINAMATH_CALUDE_tv_price_change_l3888_388849

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  (P - (x / 100) * P) * (1 + 30 / 100) = P * (1 + 4 / 100) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l3888_388849


namespace NUMINAMATH_CALUDE_square_value_l3888_388804

theorem square_value (x y z : ℝ) 
  (eq1 : 2*x + y + z = 17)
  (eq2 : x + 2*y + z = 14)
  (eq3 : x + y + 2*z = 13) :
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_square_value_l3888_388804


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_removal_l3888_388883

theorem smallest_n_for_candy_removal : ∃ n : ℕ, 
  (∀ k : ℕ, k > 0 → k * (k + 1) / 2 ≥ 64 → n ≤ k) ∧ 
  n * (n + 1) / 2 ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_removal_l3888_388883


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_sum_of_radii_l3888_388815

/-- For a right-angled triangle, the perimeter equals the sum of radii of inscribed and excircles -/
theorem right_triangle_perimeter_equals_sum_of_radii 
  (a b c ρ ρ_a ρ_b ρ_c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem for right-angled triangle
  (h_ρ : ρ = (a + b - c) / 2)  -- Formula for inscribed circle radius
  (h_ρ_a : ρ_a = (a + b + c) / 2 - a)  -- Formula for excircle radius opposite to side a
  (h_ρ_b : ρ_b = (a + b + c) / 2 - b)  -- Formula for excircle radius opposite to side b
  (h_ρ_c : ρ_c = (a + b + c) / 2)  -- Formula for excircle radius opposite to side c
  : a + b + c = ρ + ρ_a + ρ_b + ρ_c := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_sum_of_radii_l3888_388815


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3888_388850

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 4 + a 7 + a 10 = 30 →
  a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3888_388850


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l3888_388867

theorem absolute_value_equation_solution_product : ∃ (x₁ x₂ : ℝ),
  (|2 * x₁ - 1| + 4 = 24) ∧
  (|2 * x₂ - 1| + 4 = 24) ∧
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -99.75) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_product_l3888_388867


namespace NUMINAMATH_CALUDE_community_service_selection_schemes_l3888_388824

theorem community_service_selection_schemes :
  let total_boys : ℕ := 4
  let total_girls : ℕ := 2
  let group_size : ℕ := 4
  let min_girls : ℕ := 1

  let selection_schemes : ℕ := 
    Nat.choose total_girls 1 * Nat.choose total_boys 3 +
    Nat.choose total_girls 2 * Nat.choose total_boys 2

  selection_schemes = 14 :=
by sorry

end NUMINAMATH_CALUDE_community_service_selection_schemes_l3888_388824


namespace NUMINAMATH_CALUDE_pablo_candy_cost_l3888_388890

/-- The cost of candy given Pablo's reading and spending habits -/
def candy_cost (pages_per_book : ℕ) (books_read : ℕ) (earnings_per_page : ℚ) (money_left : ℚ) : ℚ :=
  (pages_per_book * books_read : ℕ) * earnings_per_page - money_left

/-- Theorem stating the cost of candy given Pablo's specific situation -/
theorem pablo_candy_cost :
  candy_cost 150 12 (1 / 100) 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pablo_candy_cost_l3888_388890


namespace NUMINAMATH_CALUDE_sum_floor_equals_217_l3888_388898

theorem sum_floor_equals_217 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_squares : x^2 + y^2 = 4050 ∧ z^2 + w^2 = 4050)
  (products : x*z = 2040 ∧ y*w = 2040) : 
  ⌊x + y + z + w⌋ = 217 := by
sorry

end NUMINAMATH_CALUDE_sum_floor_equals_217_l3888_388898


namespace NUMINAMATH_CALUDE_replaced_tomatoes_cost_is_2_20_l3888_388889

/-- Represents the grocery order with item prices and total costs -/
structure GroceryOrder where
  original_total : ℝ
  original_tomatoes : ℝ
  original_lettuce : ℝ
  original_celery : ℝ
  new_lettuce : ℝ
  new_celery : ℝ
  delivery_tip : ℝ
  new_total : ℝ

/-- Calculates the cost of the replaced can of tomatoes -/
def replaced_tomatoes_cost (order : GroceryOrder) : ℝ :=
  order.new_total - order.original_total - order.delivery_tip -
  (order.new_lettuce - order.original_lettuce) -
  (order.new_celery - order.original_celery) +
  order.original_tomatoes

/-- Theorem stating that the cost of the replaced can of tomatoes is $2.20 -/
theorem replaced_tomatoes_cost_is_2_20 (order : GroceryOrder)
  (h1 : order.original_total = 25)
  (h2 : order.original_tomatoes = 0.99)
  (h3 : order.original_lettuce = 1)
  (h4 : order.original_celery = 1.96)
  (h5 : order.new_lettuce = 1.75)
  (h6 : order.new_celery = 2)
  (h7 : order.delivery_tip = 8)
  (h8 : order.new_total = 35) :
  replaced_tomatoes_cost order = 2.20 := by
  sorry


end NUMINAMATH_CALUDE_replaced_tomatoes_cost_is_2_20_l3888_388889


namespace NUMINAMATH_CALUDE_marie_erasers_l3888_388872

/-- Given that Marie starts with 95.0 erasers and buys 42.0 erasers, 
    prove that she ends up with 137.0 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℝ := 95.0
  let bought_erasers : ℝ := 42.0
  let final_erasers : ℝ := initial_erasers + bought_erasers
  final_erasers = 137.0 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l3888_388872


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l3888_388819

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_proof (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 60 →
  sampleSize = 6 →
  start = 3 →
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l3888_388819


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3888_388844

/-- Atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.33

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Number of Barium atoms in the compound -/
def Ba_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ := Ba_count * Ba_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is 171.35 g/mol -/
theorem compound_molecular_weight : molecular_weight = 171.35 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3888_388844


namespace NUMINAMATH_CALUDE_point_on_circle_l3888_388837

theorem point_on_circle (t : ℝ) :
  let x := (2 - t^2) / (2 + t^2)
  let y := 3*t / (2 + t^2)
  x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_circle_l3888_388837
