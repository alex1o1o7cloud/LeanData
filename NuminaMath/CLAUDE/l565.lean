import Mathlib

namespace blue_pill_cost_l565_56528

/-- Represents the cost of pills for a 21-day regimen --/
structure PillCost where
  blue : ℝ
  yellow : ℝ
  total : ℝ
  h1 : blue = yellow + 3
  h2 : 21 * (blue + yellow) = total

/-- The theorem stating the cost of a blue pill given the conditions --/
theorem blue_pill_cost (pc : PillCost) (h : pc.total = 882) : pc.blue = 22.5 := by
  sorry

end blue_pill_cost_l565_56528


namespace curtain_length_for_given_room_l565_56552

/-- Calculates the required curtain length in inches given the room height in feet and additional material in inches. -/
def curtain_length (room_height_feet : ℕ) (additional_inches : ℕ) : ℕ :=
  room_height_feet * 12 + additional_inches

/-- Theorem stating that for a room height of 8 feet and 5 inches of additional material, the required curtain length is 101 inches. -/
theorem curtain_length_for_given_room : curtain_length 8 5 = 101 := by
  sorry

end curtain_length_for_given_room_l565_56552


namespace negative_reals_sup_and_max_l565_56518

-- Define the set of negative real numbers
def NegativeReals : Set ℝ := {x | x < 0}

-- Theorem statement
theorem negative_reals_sup_and_max :
  (∃ s : ℝ, IsLUB NegativeReals s) ∧
  (¬∃ m : ℝ, m ∈ NegativeReals ∧ ∀ x ∈ NegativeReals, x ≤ m) :=
by sorry

end negative_reals_sup_and_max_l565_56518


namespace smallest_addition_for_divisibility_l565_56505

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 2374) ∧ (1275890 + x) % 2375 = 0 ∧ 
  ∀ y : ℕ, y < x → (1275890 + y) % 2375 ≠ 0 :=
by sorry

end smallest_addition_for_divisibility_l565_56505


namespace fixed_point_of_logarithmic_function_l565_56571

theorem fixed_point_of_logarithmic_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.log x / Real.log a
  f 1 = 1 := by
  sorry

end fixed_point_of_logarithmic_function_l565_56571


namespace f_is_generalized_distance_l565_56517

def generalizedDistance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

def f (x y : ℝ) : ℝ := x^2 + y^2

theorem f_is_generalized_distance : generalizedDistance f := by sorry

end f_is_generalized_distance_l565_56517


namespace zoo_field_trip_buses_l565_56596

theorem zoo_field_trip_buses (fifth_graders sixth_graders seventh_graders : ℕ)
  (teachers_per_grade parents_per_grade : ℕ) (seats_per_bus : ℕ)
  (h1 : fifth_graders = 109)
  (h2 : sixth_graders = 115)
  (h3 : seventh_graders = 118)
  (h4 : teachers_per_grade = 4)
  (h5 : parents_per_grade = 2)
  (h6 : seats_per_bus = 72) :
  (fifth_graders + sixth_graders + seventh_graders +
   3 * (teachers_per_grade + parents_per_grade) + seats_per_bus - 1) / seats_per_bus = 5 :=
by sorry

end zoo_field_trip_buses_l565_56596


namespace set_intersection_problem_l565_56543

theorem set_intersection_problem :
  let M : Set ℝ := {x | x^2 - x = 0}
  let N : Set ℝ := {-1, 0}
  M ∩ N = {0} := by
sorry

end set_intersection_problem_l565_56543


namespace probability_less_than_20_l565_56567

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 2 / 5 := by
  sorry

end probability_less_than_20_l565_56567


namespace game_ends_after_33_rounds_l565_56520

/-- Represents a player in the token redistribution game -/
inductive Player
| P
| Q
| R

/-- State of the game, tracking token counts for each player and the number of rounds played -/
structure GameState where
  tokens : Player → ℕ
  rounds : ℕ

/-- Determines if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.P => 12
    | Player.Q => 10
    | Player.R => 8,
    rounds := 0 }

/-- The main theorem: the game ends after 33 rounds -/
theorem game_ends_after_33_rounds :
  ∃ finalState : GameState,
    finalState.rounds = 33 ∧
    gameEnded finalState ∧
    (∀ n : ℕ, n < 33 → ¬gameEnded ((playRound^[n]) initialState)) :=
  sorry

end game_ends_after_33_rounds_l565_56520


namespace train_length_calculation_l565_56594

/-- The length of a train given jogger and train speeds, initial distance, and time to pass. -/
theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (time_to_pass : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 240 →
  time_to_pass = 39 →
  (train_speed - jogger_speed) * time_to_pass - initial_distance = 150 := by
sorry

end train_length_calculation_l565_56594


namespace two_students_same_type_l565_56591

-- Define the types of books
inductive BookType
  | History
  | Literature
  | Science

-- Define a type for a pair of books
def BookPair := BookType × BookType

-- Define the set of all possible book pairs
def allBookPairs : Finset BookPair :=
  sorry

-- Define the number of students
def numStudents : Nat := 7

-- Theorem statement
theorem two_students_same_type :
  ∃ (s₁ s₂ : Fin numStudents) (bp : BookPair),
    s₁ ≠ s₂ ∧ 
    (∀ (s : Fin numStudents), ∃ (bp : BookPair), bp ∈ allBookPairs) ∧
    (∃ (f : Fin numStudents → BookPair), f s₁ = bp ∧ f s₂ = bp) :=
  sorry

end two_students_same_type_l565_56591


namespace unique_exam_scores_l565_56548

def is_valid_score_set (scores : List Nat) : Prop :=
  scores.length = 5 ∧
  scores.all (λ x => x % 2 = 1 ∧ x < 100) ∧
  scores.Nodup ∧
  scores.sum / scores.length = 80 ∧
  [95, 85, 75, 65].all (λ x => x ∈ scores)

theorem unique_exam_scores :
  ∃! scores : List Nat, is_valid_score_set scores ∧ scores = [95, 85, 79, 75, 65] := by
  sorry

end unique_exam_scores_l565_56548


namespace boy_escapes_l565_56536

/-- Represents the square pool -/
structure Pool :=
  (side_length : ℝ)
  (boy_position : ℝ × ℝ)
  (teacher_position : ℝ × ℝ)

/-- Represents the speeds of the boy and teacher -/
structure Speeds :=
  (boy_swim : ℝ)
  (boy_run : ℝ)
  (teacher_run : ℝ)

/-- Checks if the boy can escape given the pool configuration and speeds -/
def can_escape (p : Pool) (s : Speeds) : Prop :=
  p.side_length = 2 ∧
  p.boy_position = (0, 0) ∧
  p.teacher_position = (1, 1) ∧
  s.boy_swim = s.teacher_run / 3 ∧
  s.boy_run > s.teacher_run

theorem boy_escapes (p : Pool) (s : Speeds) :
  can_escape p s → true :=
sorry

end boy_escapes_l565_56536


namespace expression_simplification_l565_56512

theorem expression_simplification (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (3 * a^2 * b - a * b^2) - (1/2) * (a^2 * b - (2 * a * b^2 - 4)) + 1 = 9 := by
  sorry

end expression_simplification_l565_56512


namespace restaurant_menu_fraction_l565_56515

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (h1 : 6 = (1 / 3 : ℚ) * total_dishes)
  (h2 : 4 ≤ 6) : 
  (2 : ℚ) / total_dishes = 1 / 9 := by sorry

end restaurant_menu_fraction_l565_56515


namespace ali_seashells_left_l565_56540

/-- The number of seashells Ali has left after giving some away and selling half --/
def seashells_left (initial : ℕ) (given_to_friends : ℕ) (given_to_brothers : ℕ) : ℕ :=
  let remaining_after_giving := initial - given_to_friends - given_to_brothers
  remaining_after_giving - remaining_after_giving / 2

/-- Theorem stating that Ali has 55 seashells left --/
theorem ali_seashells_left : seashells_left 180 40 30 = 55 := by
  sorry

end ali_seashells_left_l565_56540


namespace circle_equation_proof_l565_56573

/-- Given a parabola and a hyperbola, prove the equation of a circle with specific properties -/
theorem circle_equation_proof (x y : ℝ) : 
  (∃ (p : ℝ × ℝ), y^2 = 20*x ∧ p = (5, 0)) → -- Parabola equation and its focus
  (x^2/9 - y^2/16 = 1) →                    -- Hyperbola equation
  (∃ (c : ℝ × ℝ) (r : ℝ),                   -- Circle properties
    c = (5, 0) ∧                            -- Circle center at parabola focus
    r = 4 ∧                                 -- Circle radius
    (∀ (x' y' : ℝ), (y' = 4*x'/3 ∨ y' = -4*x'/3) →  -- Asymptotes of hyperbola
      (x' - c.1)^2 + (y' - c.2)^2 = r^2)) →  -- Circle tangent to asymptotes
  (x - 5)^2 + y^2 = 16                       -- Equation of the circle
  := by sorry

end circle_equation_proof_l565_56573


namespace train_distance_proof_l565_56576

/-- The initial distance between two trains -/
def initial_distance : ℝ := 13

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 37

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 43

/-- The time it takes for Train B to overtake and be ahead of Train A, in hours -/
def overtake_time : ℝ := 5

/-- The distance Train B is ahead of Train A after overtaking, in miles -/
def ahead_distance : ℝ := 17

theorem train_distance_proof :
  initial_distance = (speed_B - speed_A) * overtake_time - ahead_distance :=
by sorry

end train_distance_proof_l565_56576


namespace min_value_theorem_l565_56570

-- Define the condition for a, b, c
def satisfies_condition (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3

-- State the theorem
theorem min_value_theorem :
  ∃ (a b c : ℝ), satisfies_condition a b c ∧
  (∀ (a' b' c' : ℝ), satisfies_condition a' b' c' → a + 2*b - 3*c ≤ a' + 2*b' - 3*c') ∧
  a + 2*b - 3*c = -2 :=
sorry

end min_value_theorem_l565_56570


namespace inscribed_circle_chord_length_l565_56562

theorem inscribed_circle_chord_length (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1) :
  let r := (a + b - 1) / 2
  let chord_length := Real.sqrt (1 - 2 * r^2)
  chord_length = Real.sqrt 2 / 2 := by sorry

end inscribed_circle_chord_length_l565_56562


namespace implicit_function_derivative_l565_56569

/-- Given an implicitly defined function y²(x) + x² - 1 = 0,
    prove that the derivative of y with respect to x is -x / y(x) -/
theorem implicit_function_derivative 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x ^ 2 + x ^ 2 - 1 = 0) :
  ∀ x, HasDerivAt y (-(x / y x)) x :=
sorry

end implicit_function_derivative_l565_56569


namespace overtime_hours_example_l565_56565

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  ordinaryRate : ℚ  -- Rate for ordinary hours in dollars
  overtimeRate : ℚ  -- Rate for overtime hours in dollars
  totalHours : ℕ    -- Total hours worked
  totalPay : ℚ      -- Total pay received in dollars

/-- Calculates the number of overtime hours worked -/
def overtimeHours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 -/
theorem overtime_hours_example :
  let w : WorkerPay := {
    ordinaryRate := 60/100,  -- 60 cents
    overtimeRate := 90/100,  -- 90 cents
    totalHours := 50,
    totalPay := 3240/100     -- $32.40
  }
  overtimeHours w = 8 := by sorry

end overtime_hours_example_l565_56565


namespace power_of_three_mod_ten_l565_56581

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end power_of_three_mod_ten_l565_56581


namespace train_passenger_problem_l565_56550

theorem train_passenger_problem (P : ℚ) : 
  (((P * (2/3) + 280) * (1/2) + 12) = 242) → P = 270 := by
  sorry

end train_passenger_problem_l565_56550


namespace geometric_sequence_b_value_l565_56549

theorem geometric_sequence_b_value (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 180) (h₂ : a₃ = 64/25) (h₃ : a₂ > 0) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 21.6 := by
  sorry

end geometric_sequence_b_value_l565_56549


namespace circle_radius_proof_l565_56582

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = 16 * Real.pi) →
  (2 * A₁ = 16 * Real.pi - A₁) →
  (∃ (r : ℝ), r > 0 ∧ A₁ = Real.pi * r^2 ∧ r = 4 * Real.sqrt 3 / 3) := by
  sorry

end circle_radius_proof_l565_56582


namespace unique_k_solution_l565_56500

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem unique_k_solution (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 := by
  sorry

end unique_k_solution_l565_56500


namespace emily_calculation_l565_56563

theorem emily_calculation (a b c : ℝ) 
  (h1 : a - (2*b - c) = 15) 
  (h2 : a - 2*b - c = 5) : 
  a - 2*b = 10 := by sorry

end emily_calculation_l565_56563


namespace B_power_15_minus_3_power_14_l565_56514

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end B_power_15_minus_3_power_14_l565_56514


namespace angle_between_vectors_l565_56566

def a : ℝ × ℝ := (2, -1)

theorem angle_between_vectors (b : ℝ × ℝ) (θ : ℝ) 
  (h1 : ‖b‖ = 2 * Real.sqrt 5)
  (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 10)
  (h3 : θ = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)))
  : θ = π / 3 := by
  sorry

end angle_between_vectors_l565_56566


namespace final_black_goats_count_l565_56544

theorem final_black_goats_count (total : ℕ) (initial_black : ℕ) (new_black : ℕ) :
  total = 93 →
  initial_black = 66 →
  new_black = 21 →
  initial_black ≤ total →
  let initial_white := total - initial_black
  let new_total_black := initial_black + new_black
  let deaths := min initial_white new_total_black
  new_total_black - deaths = 60 :=
by
  sorry

end final_black_goats_count_l565_56544


namespace geometric_sequence_formula_l565_56558

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_diff : |a 2 - a 3| = 14)
  (h_product : a 1 * a 2 * a 3 = 343)
  (h_geometric : geometric_sequence a) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a n = 7 * q^(n - 2) :=
sorry

end geometric_sequence_formula_l565_56558


namespace bug_total_distance_l565_56568

def bug_journey (start end1 end2 end3 : ℝ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_total_distance :
  bug_journey 0 4 (-3) 7 = 21 := by
  sorry

end bug_total_distance_l565_56568


namespace minimize_y_l565_56501

variable (a b c x : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*c*x

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x ≥ y x_min ∧ x_min = (a + b - c) / 2 :=
sorry

end minimize_y_l565_56501


namespace equal_roots_quadratic_l565_56593

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 3 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 3 = 0 → y = x) → 
  a = 4/3 := by
sorry

end equal_roots_quadratic_l565_56593


namespace specific_arrangement_double_coverage_l565_56533

/-- Represents a rectangle on a grid -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the arrangement of rectangles on the grid -/
structure Arrangement where
  rectangles : List Rectangle
  -- Additional properties to describe the specific arrangement could be added here

/-- Counts the number of cells covered by exactly two rectangles in the given arrangement -/
def countDoublyCoveredCells (arr : Arrangement) : ℕ :=
  sorry -- Implementation details would go here

/-- The main theorem stating that for the specific arrangement of three 4x6 rectangles,
    the number of cells covered by exactly two rectangles is 14 -/
theorem specific_arrangement_double_coverage :
  ∃ (arr : Arrangement),
    (arr.rectangles.length = 3) ∧
    (∀ r ∈ arr.rectangles, r.width = 4 ∧ r.height = 6) ∧
    (countDoublyCoveredCells arr = 14) := by
  sorry


end specific_arrangement_double_coverage_l565_56533


namespace x_one_value_l565_56579

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/3) : 
  x₁ = 2/3 := by
  sorry

end x_one_value_l565_56579


namespace glenburgh_parade_squad_l565_56554

theorem glenburgh_parade_squad (m : ℕ) : 
  (∃ k : ℕ, 20 * m = 28 * k + 6) → 
  20 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 20 * n = 28 * j + 6) → 20 * n < 1200 → 20 * n ≤ 20 * m) →
  20 * m = 1160 := by
sorry

end glenburgh_parade_squad_l565_56554


namespace children_still_hiding_l565_56560

theorem children_still_hiding (total : ℕ) (found : ℕ) (seeker : ℕ) : 
  total = 16 → found = 6 → seeker = 1 → total - found - seeker = 9 := by
sorry

end children_still_hiding_l565_56560


namespace jakes_snake_length_l565_56530

/-- Given two snakes where one is 12 inches longer than the other,
    and their combined length is 70 inches, prove that the longer snake is 41 inches long. -/
theorem jakes_snake_length (penny_snake : ℕ) (jake_snake : ℕ)
  (h1 : jake_snake = penny_snake + 12)
  (h2 : penny_snake + jake_snake = 70) :
  jake_snake = 41 :=
by sorry

end jakes_snake_length_l565_56530


namespace toenail_size_ratio_l565_56535

/-- Represents the capacity of the jar in terms of regular toenails -/
def jar_capacity : ℕ := 100

/-- Represents the number of big toenails in the jar -/
def big_toenails : ℕ := 20

/-- Represents the number of regular toenails initially in the jar -/
def regular_toenails : ℕ := 40

/-- Represents the additional regular toenails that can fit in the jar -/
def additional_regular_toenails : ℕ := 20

/-- Represents the ratio of the size of a big toenail to a regular toenail -/
def big_to_regular_ratio : ℚ := 2

theorem toenail_size_ratio :
  (jar_capacity - regular_toenails - additional_regular_toenails) / big_toenails = big_to_regular_ratio :=
sorry

end toenail_size_ratio_l565_56535


namespace sum_of_digits_next_l565_56534

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For a positive integer n where S(n) = 1274, S(n+1) = 1239 -/
theorem sum_of_digits_next (n : ℕ) (h : S n = 1274) : S (n + 1) = 1239 := by
  sorry

end sum_of_digits_next_l565_56534


namespace investment_rate_proof_l565_56586

def total_investment : ℝ := 3000
def high_interest_amount : ℝ := 800
def high_interest_rate : ℝ := 0.1
def total_interest : ℝ := 256

def remaining_investment : ℝ := total_investment - high_interest_amount
def high_interest : ℝ := high_interest_amount * high_interest_rate
def remaining_interest : ℝ := total_interest - high_interest

theorem investment_rate_proof :
  remaining_interest / remaining_investment = 0.08 :=
sorry

end investment_rate_proof_l565_56586


namespace inequality_and_minimum_value_l565_56587

variables (a b x : ℝ)

def f (a b x : ℝ) : ℝ := |2*x - a^4 + (1 - 6*a^2*b^2 - b^4)| + 2*|x - (2*a^3*b + 2*a*b^3 - 1)|

theorem inequality_and_minimum_value :
  (a^4 + 6*a^2*b^2 + b^4 ≥ 4*a*b*(a^2 + b^2)) ∧
  (∀ x, f a b x ≥ 1) := by
  sorry

end inequality_and_minimum_value_l565_56587


namespace ellipse_slope_product_l565_56510

/-- Given an ellipse with equation x^2/25 + y^2/9 = 1, 
    this theorem states that for any point P on the ellipse 
    (distinct from the endpoints of the major axis), 
    the product of the slopes of the lines connecting P 
    to the endpoints of the major axis is -9/25. -/
theorem ellipse_slope_product : 
  ∀ (x y : ℝ), 
  x^2/25 + y^2/9 = 1 →  -- P is on the ellipse
  x ≠ 5 →              -- P is not the right endpoint
  x ≠ -5 →             -- P is not the left endpoint
  ∃ (m₁ m₂ : ℝ),       -- slopes exist
  (m₁ = y / (x - 5) ∧ m₂ = y / (x + 5)) ∧  -- definition of slopes
  m₁ * m₂ = -9/25 :=   -- product of slopes
by sorry


end ellipse_slope_product_l565_56510


namespace gcd_of_powers_of_47_l565_56516

theorem gcd_of_powers_of_47 :
  Nat.Prime 47 →
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 47 + 1) = 1 := by
sorry

end gcd_of_powers_of_47_l565_56516


namespace shortest_altitude_of_triangle_l565_56526

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h * c = 2 * (a * b / 2) →
  h = 7.2 := by sorry

end shortest_altitude_of_triangle_l565_56526


namespace farmer_earnings_proof_l565_56556

/-- Calculates the farmer's earnings after the market fee -/
def farmer_earnings (potatoes carrots tomatoes : ℕ) 
  (potato_bundle_size potato_bundle_price : ℚ)
  (carrot_bundle_size carrot_bundle_price : ℚ)
  (tomato_price canned_tomato_set_size canned_tomato_set_price : ℚ)
  (market_fee_rate : ℚ) : ℚ :=
  let potato_sales := (potatoes / potato_bundle_size) * potato_bundle_price
  let carrot_sales := (carrots / carrot_bundle_size) * carrot_bundle_price
  let fresh_tomato_sales := (tomatoes / 2) * tomato_price
  let canned_tomato_sales := ((tomatoes / 2) / canned_tomato_set_size) * canned_tomato_set_price
  let total_sales := potato_sales + carrot_sales + fresh_tomato_sales + canned_tomato_sales
  let market_fee := total_sales * market_fee_rate
  total_sales - market_fee

/-- The farmer's earnings after the market fee is $618.45 -/
theorem farmer_earnings_proof :
  farmer_earnings 250 320 480 25 1.9 20 2 1 10 15 0.05 = 618.45 := by
  sorry

end farmer_earnings_proof_l565_56556


namespace company_picnic_attendance_l565_56559

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.30000000000000004 →
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.5 := by
  sorry

end company_picnic_attendance_l565_56559


namespace min_value_xyz_plus_2sum_l565_56557

theorem min_value_xyz_plus_2sum (x y z : ℝ) 
  (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hz : |z| ≥ 2) : 
  |x * y * z + 2 * (x + y + z)| ≥ 4 := by
  sorry

end min_value_xyz_plus_2sum_l565_56557


namespace min_distance_sum_l565_56529

/-- A line in 2D space passing through (1,4) and intersecting positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through_point : 1 / a + 4 / b = 1

/-- The sum of distances from origin to intersection points is at least 9 -/
theorem min_distance_sum (l : IntersectingLine) :
  l.a + l.b ≥ 9 := by
  sorry

#check min_distance_sum

end min_distance_sum_l565_56529


namespace ice_block_volume_l565_56555

theorem ice_block_volume (V : ℝ) : 
  V > 0 →
  (8/35 : ℝ) * V = 0.15 →
  V = 0.65625 := by sorry

end ice_block_volume_l565_56555


namespace x_less_than_2_necessary_not_sufficient_l565_56564

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x^2 < 4 ∧ ¬(x < 2)) ∧
  (∀ x : ℝ, x^2 < 4 → x < 2) :=
by sorry

end x_less_than_2_necessary_not_sufficient_l565_56564


namespace min_sum_with_constraint_l565_56551

theorem min_sum_with_constraint (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧ ∃ (x₀ y₀ z₀ : ℝ), (4 / x₀) + (2 / y₀) + (1 / z₀) = 1 ∧ x₀ + 8 * y₀ + 4 * z₀ = 64 := by
  sorry

end min_sum_with_constraint_l565_56551


namespace product_remainder_l565_56539

theorem product_remainder (x : ℕ) :
  (1274 * x * 1277 * 1285) % 12 = 6 → x % 12 = 1 := by
  sorry

end product_remainder_l565_56539


namespace simplify_and_rationalize_l565_56532

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l565_56532


namespace monomial_combination_l565_56599

theorem monomial_combination (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), ∃ (m n : ℤ), 3 * a^(7*x) * b^(y+7) = k * a^m * b^n ∧ 
                           -7 * a^(2-4*y) * b^(2*x) = k * a^m * b^n) → 
  x + y = -1 := by sorry

end monomial_combination_l565_56599


namespace distance_sum_on_unit_circle_l565_56521

theorem distance_sum_on_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  a^4 + b^4 + ((a - b)^4 / 4) + ((a + b)^4 / 4) = 3/2 := by sorry

end distance_sum_on_unit_circle_l565_56521


namespace sqrt_sum_inequality_l565_56595

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end sqrt_sum_inequality_l565_56595


namespace tangent_line_to_circle_l565_56503

/-- A line is tangent to a circle if and only if the distance from the center of the circle
    to the line is equal to the radius of the circle. -/
axiom tangent_line_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ a*x + b*y + c = 0) ↔ 
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) (h_pos : m > 0) 
  (h_tangent : ∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m = 0 → 
    (x - 3)^2 + (y - 4)^2 = 4 ∧ 3*x - 4*y - m = 0) : 
  m = 3 := by
sorry

end tangent_line_to_circle_l565_56503


namespace intersection_nonempty_iff_p_less_than_one_l565_56542

theorem intersection_nonempty_iff_p_less_than_one (p : ℝ) :
  let M : Set ℝ := {x | x ≤ 1}
  let N : Set ℝ := {x | x > p}
  (M ∩ N).Nonempty ↔ p < 1 := by
  sorry

end intersection_nonempty_iff_p_less_than_one_l565_56542


namespace hyperbola_theorem_l565_56506

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector from a point to another
def vec_from_to (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define the condition for point M
def M_condition (M A B : ℝ × ℝ) : Prop :=
  vec_from_to F1 M = vec_add (vec_add (vec_from_to F1 A) (vec_from_to F1 B)) (vec_from_to F1 O)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem hyperbola_theorem (A B M : ℝ × ℝ) 
  (hA : hyperbola A.1 A.2) 
  (hB : hyperbola B.1 B.2) 
  (hM : M_condition M A B) :
  -- 1. The locus of M is (x-6)^2 - y^2 = 4
  ((M.1 - 6)^2 - M.2^2 = 4) ∧
  -- 2. There exists a fixed point C(1, 0) such that CA · CB is constant
  (∃ (C : ℝ × ℝ), C = (1, 0) ∧ 
    dot_product (vec_from_to C A) (vec_from_to C B) = -1) :=
sorry

end hyperbola_theorem_l565_56506


namespace jason_earnings_l565_56592

/-- Calculates Jason's total earnings for the week given his work hours and rates --/
theorem jason_earnings (after_school_rate : ℝ) (saturday_rate : ℝ) (total_hours : ℝ) (saturday_hours : ℝ) :
  after_school_rate = 4 ∧ 
  saturday_rate = 6 ∧ 
  total_hours = 18 ∧ 
  saturday_hours = 8 →
  (total_hours - saturday_hours) * after_school_rate + saturday_hours * saturday_rate = 88 := by
  sorry

end jason_earnings_l565_56592


namespace abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l565_56584

theorem abs_gt_one_necessary_not_sufficient_for_lt_neg_two (x : ℝ) :
  (∀ x, x < -2 → |x| > 1) ∧ 
  (∃ x, |x| > 1 ∧ ¬(x < -2)) :=
by sorry

end abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l565_56584


namespace inequality_solution_set_l565_56598

theorem inequality_solution_set (x : ℝ) : -2 * x + 3 < 0 ↔ x > 3/2 := by
  sorry

end inequality_solution_set_l565_56598


namespace integral_inequality_l565_56538

theorem integral_inequality (m : ℕ+) : 
  0 ≤ ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ∧
  ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ≤ 1 :=
by sorry

end integral_inequality_l565_56538


namespace trainers_average_age_l565_56574

/-- The average age of trainers in a sports club --/
theorem trainers_average_age
  (total_members : ℕ)
  (overall_average : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_trainers : ℕ)
  (women_average : ℚ)
  (men_average : ℚ)
  (h_total : total_members = 70)
  (h_overall : overall_average = 23)
  (h_women : num_women = 30)
  (h_men : num_men = 25)
  (h_trainers : num_trainers = 15)
  (h_women_avg : women_average = 20)
  (h_men_avg : men_average = 25)
  (h_sum : total_members = num_women + num_men + num_trainers) :
  (total_members * overall_average - num_women * women_average - num_men * men_average) / num_trainers = 25 + 2/3 :=
by sorry

end trainers_average_age_l565_56574


namespace fifteenth_student_age_l565_56546

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) :
  total_students = 15 →
  group1_students = 5 →
  group2_students = 9 →
  total_average_age = 15 →
  group1_average_age = 14 →
  group2_average_age = 16 →
  (total_students * total_average_age) - 
    (group1_students * group1_average_age + group2_students * group2_average_age) = 11 := by
  sorry

end fifteenth_student_age_l565_56546


namespace inscribed_circle_ratio_l565_56547

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the inscribed circle P
variable (P : Point)

-- Define that P is the center of the inscribed circle
def is_inscribed_center (P : Point) (A B C D : Point) : Prop := sorry

-- Define the distance function
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem inscribed_circle_ratio 
  (h : is_inscribed_center P A B C D) :
  (distance P A)^2 / (distance P C)^2 = 
  (distance A B * distance A D) / (distance B C * distance C D) := by sorry

end inscribed_circle_ratio_l565_56547


namespace triangle_isosceles_or_right_angled_l565_56589

-- Define a triangle with angles α, β, and γ
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi

-- Define the theorem
theorem triangle_isosceles_or_right_angled (t : Triangle) :
  Real.tan t.β * Real.sin t.γ * Real.sin t.γ = Real.tan t.γ * Real.sin t.β * Real.sin t.β →
  (t.β = t.γ ∨ t.β + t.γ = Real.pi / 2) :=
by
  sorry

end triangle_isosceles_or_right_angled_l565_56589


namespace sqrt_ab_equals_18_l565_56527

theorem sqrt_ab_equals_18 (a b : ℝ) : 
  a = Real.log 9 / Real.log 4 → 
  b = 108 * (Real.log 8 / Real.log 3) → 
  Real.sqrt (a * b) = 18 := by
  sorry

end sqrt_ab_equals_18_l565_56527


namespace unique_integer_solution_l565_56522

theorem unique_integer_solution (x y : ℤ) : 
  ({2 * x, x + y} : Set ℤ) = {7, 4} → x = 2 ∧ y = 5 := by
  sorry

end unique_integer_solution_l565_56522


namespace value_of_P_l565_56561

theorem value_of_P : ∃ P : ℚ, (3/4 : ℚ) * (1/9 : ℚ) * P = (1/4 : ℚ) * (1/8 : ℚ) * 160 ∧ P = 60 := by
  sorry

end value_of_P_l565_56561


namespace team_selection_ways_l565_56577

def total_boys : ℕ := 10
def total_girls : ℕ := 12
def team_size : ℕ := 8
def required_boys : ℕ := 5
def required_girls : ℕ := 3

theorem team_selection_ways :
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 55440 :=
by sorry

end team_selection_ways_l565_56577


namespace min_difference_is_one_l565_56545

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  xz : ℕ
  yz : ℕ
  xy : ℕ

/-- Checks if the given side lengths form a valid triangle --/
def isValidTriangle (t : TriangleSides) : Prop :=
  t.xz + t.yz > t.xy ∧ t.xz + t.xy > t.yz ∧ t.yz + t.xy > t.xz

/-- Checks if the given side lengths satisfy the problem conditions --/
def satisfiesConditions (t : TriangleSides) : Prop :=
  t.xz + t.yz + t.xy = 3001 ∧ t.xz < t.yz ∧ t.yz ≤ t.xy

theorem min_difference_is_one :
  ∀ t : TriangleSides,
    isValidTriangle t →
    satisfiesConditions t →
    ∀ u : TriangleSides,
      isValidTriangle u →
      satisfiesConditions u →
      t.yz - t.xz ≤ u.yz - u.xz →
      t.yz - t.xz = 1 :=
sorry

end min_difference_is_one_l565_56545


namespace order_of_roots_l565_56531

theorem order_of_roots : 5^(2/3) > 16^(1/3) ∧ 16^(1/3) > 2^(4/5) := by
  sorry

end order_of_roots_l565_56531


namespace sqrt_50_plus_sqrt_32_l565_56585

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end sqrt_50_plus_sqrt_32_l565_56585


namespace area_of_triangle_formed_by_tangent_points_l565_56513

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- The main theorem -/
theorem area_of_triangle_formed_by_tangent_points : 
  ∀ (c1 c2 c3 : Circle),
  c1.radius = 1 ∧ c2.radius = 3 ∧ c3.radius = 5 →
  are_externally_tangent c1 c2 ∧ 
  are_externally_tangent c2 c3 ∧ 
  are_externally_tangent c1 c3 →
  is_tangent_to_x_axis c1 ∧ 
  is_tangent_to_x_axis c2 ∧ 
  is_tangent_to_x_axis c3 →
  let (x1, _) := c1.center
  let (x2, _) := c2.center
  let (x3, _) := c3.center
  (1/2) * (|x2 - x1| + |x3 - x2| + |x3 - x1|) * (c3.radius - c1.radius) = 6 :=
by sorry

end area_of_triangle_formed_by_tangent_points_l565_56513


namespace no_solution_equation_one_solutions_equation_two_l565_56504

-- Problem 1
theorem no_solution_equation_one : 
  ¬ ∃ x : ℝ, (1 / (x - 2) + 2 = (1 - x) / (2 - x)) ∧ (x ≠ 2) :=
sorry

-- Problem 2
theorem solutions_equation_two :
  ∀ x : ℝ, (x - 4)^2 = 4*(2*x + 1)^2 ↔ x = 2/5 ∨ x = -2 :=
sorry

end no_solution_equation_one_solutions_equation_two_l565_56504


namespace find_divisor_l565_56509

theorem find_divisor : ∃ d : ℕ, d > 1 ∧ (1077 + 4) % d = 0 ∧ d = 1081 := by
  sorry

end find_divisor_l565_56509


namespace sun_rise_set_differences_l565_56588

/-- Represents a geographical location with latitude and longitude -/
structure Location where
  latitude : Real
  longitude : Real

/-- Calculates the time difference of sunrise between two locations given a solar declination -/
def sunriseTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

/-- Calculates the time difference of sunset between two locations given a solar declination -/
def sunsetTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

def szeged : Location := { latitude := 46.25, longitude := 20.1667 }
def nyiregyhaza : Location := { latitude := 47.9667, longitude := 21.75 }
def winterSolsticeDeclination : Real := -23.5

theorem sun_rise_set_differences (ε : Real) :
  (ε > 0) →
  (∃ d : Real, abs (d - winterSolsticeDeclination) < ε ∧
    sunriseTimeDifference szeged nyiregyhaza d > 0) ∧
  (∃ d : Real, sunsetTimeDifference szeged nyiregyhaza d < 0) :=
sorry

end sun_rise_set_differences_l565_56588


namespace imaginary_part_sum_of_fractions_l565_56524

theorem imaginary_part_sum_of_fractions :
  Complex.im (1 / (Complex.ofReal (-2) + Complex.I) + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1 / 5 := by
  sorry

end imaginary_part_sum_of_fractions_l565_56524


namespace distance_to_origin_l565_56590

/-- The distance between point P(3,1) and the origin (0,0) in the Cartesian coordinate system is √10. -/
theorem distance_to_origin : Real.sqrt ((3 : ℝ) ^ 2 + (1 : ℝ) ^ 2) = Real.sqrt 10 := by
  sorry

end distance_to_origin_l565_56590


namespace arc_length_45_degrees_l565_56575

theorem arc_length_45_degrees (circle_circumference : Real) (central_angle : Real) (arc_length : Real) : 
  circle_circumference = 72 →
  central_angle = 45 →
  arc_length = circle_circumference * (central_angle / 360) →
  arc_length = 9 :=
by sorry

end arc_length_45_degrees_l565_56575


namespace M_equals_N_l565_56519

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end M_equals_N_l565_56519


namespace right_triangle_sin_cos_l565_56523

/-- In a right triangle XYZ with ∠Y = 90°, hypotenuse XZ = 15, and leg XY = 9, sin X = 4/5 and cos X = 3/5 -/
theorem right_triangle_sin_cos (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) (h2 : Z = 15) (h3 : X = 9) :
  Real.sin (Real.arccos (X / Z)) = 4/5 ∧ Real.cos (Real.arccos (X / Z)) = 3/5 := by
  sorry

end right_triangle_sin_cos_l565_56523


namespace sum_of_pyramid_edges_l565_56508

/-- Represents a pyramid structure -/
structure Pyramid where
  vertices : ℕ

/-- The number of edges in a pyramid -/
def Pyramid.edges (p : Pyramid) : ℕ := 2 * p.vertices - 2

/-- Theorem: For three pyramids with a total of 40 vertices, the sum of their edges is 74 -/
theorem sum_of_pyramid_edges (a b c : Pyramid) 
  (h : a.vertices + b.vertices + c.vertices = 40) : 
  a.edges + b.edges + c.edges = 74 := by
  sorry


end sum_of_pyramid_edges_l565_56508


namespace integral_cos_plus_exp_l565_56541

theorem integral_cos_plus_exp (π : Real) : ∫ x in -π..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp π := by
  sorry

end integral_cos_plus_exp_l565_56541


namespace original_number_problem_l565_56553

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 9) = 69 → x = 7 := by
  sorry

end original_number_problem_l565_56553


namespace thirtieth_triangular_number_l565_56511

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l565_56511


namespace sqrt_seven_simplification_l565_56578

theorem sqrt_seven_simplification : 3 * Real.sqrt 7 - Real.sqrt 7 = 2 * Real.sqrt 7 := by
  sorry

end sqrt_seven_simplification_l565_56578


namespace mila_trip_distance_l565_56580

/-- Represents the details of Mila's trip -/
structure MilaTrip where
  /-- Miles per gallon of Mila's car -/
  mpg : ℝ
  /-- Capacity of Mila's gas tank in gallons -/
  tankCapacity : ℝ
  /-- Miles driven in the first leg of the trip -/
  firstLegMiles : ℝ
  /-- Gallons of gas refueled -/
  refueledGallons : ℝ
  /-- Fraction of tank full upon arrival -/
  finalTankFraction : ℝ

/-- Calculates the total distance of Mila's trip -/
def totalDistance (trip : MilaTrip) : ℝ :=
  trip.firstLegMiles + (trip.tankCapacity - trip.finalTankFraction * trip.tankCapacity) * trip.mpg

/-- Theorem stating that Mila's total trip distance is 826 miles -/
theorem mila_trip_distance :
  ∀ (trip : MilaTrip),
    trip.mpg = 40 ∧
    trip.tankCapacity = 16 ∧
    trip.firstLegMiles = 400 ∧
    trip.refueledGallons = 10 ∧
    trip.finalTankFraction = 1/3 →
    totalDistance trip = 826 := by
  sorry

end mila_trip_distance_l565_56580


namespace tip_percentage_calculation_l565_56507

theorem tip_percentage_calculation (total_bill : ℝ) (sales_tax_rate : ℝ) (food_price : ℝ) : 
  total_bill = 211.20 ∧ 
  sales_tax_rate = 0.10 ∧ 
  food_price = 160 → 
  (total_bill - food_price * (1 + sales_tax_rate)) / (food_price * (1 + sales_tax_rate)) = 0.20 := by
  sorry

end tip_percentage_calculation_l565_56507


namespace remaining_money_l565_56525

/-- Calculates the remaining money after purchases and discount --/
theorem remaining_money (initial_amount purchases discount_rate : ℚ) : 
  initial_amount = 10 ∧ 
  purchases = 3 + 2 + 1.5 + 0.75 ∧ 
  discount_rate = 0.05 → 
  initial_amount - (purchases - purchases * discount_rate) = 311/100 := by
  sorry

end remaining_money_l565_56525


namespace sum_of_cubes_mod_6_l565_56597

theorem sum_of_cubes_mod_6 (h : ∀ n : ℕ, n^3 % 6 = n % 6) :
  (Finset.sum (Finset.range 150) (fun i => (i + 1)^3)) % 6 = 5 := by
  sorry

end sum_of_cubes_mod_6_l565_56597


namespace words_with_consonant_count_l565_56502

/-- The set of all letters available --/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants --/
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels --/
def vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering --/
def word_length : Nat := 5

/-- A function that returns the number of words with at least one consonant --/
def words_with_consonant : Nat :=
  letters.card ^ word_length - vowels.card ^ word_length

theorem words_with_consonant_count :
  words_with_consonant = 7744 := by sorry

end words_with_consonant_count_l565_56502


namespace smallest_a_value_l565_56537

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (3/4)^2 + p.b * (3/4) + p.c = -25/16) 
  (vertex_y : -p.b / (2 * p.a) = 3/4)
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  9 ≤ p.a ∧ ∀ a' : ℝ, 0 < a' ∧ a' < 9 → 
    ¬∃ (b' c' : ℝ) (n : ℤ), 
      a' * (3/4)^2 + b' * (3/4) + c' = -25/16 ∧
      -b' / (2 * a') = 3/4 ∧
      a' + b' + c' = n := by
  sorry

end smallest_a_value_l565_56537


namespace x_squared_mod_25_l565_56572

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 21 [ZMOD 25]) : 
  x^2 ≡ 21 [ZMOD 25] := by
  sorry

end x_squared_mod_25_l565_56572


namespace fraction_conversions_l565_56583

theorem fraction_conversions :
  (7 / 9 : ℚ) = 7 / 9 ∧
  (12 / 7 : ℚ) = 12 / 7 ∧
  (3 + 5 / 8 : ℚ) = 29 / 8 ∧
  (6 : ℚ) = 66 / 11 := by
sorry

end fraction_conversions_l565_56583
