import Mathlib

namespace NUMINAMATH_CALUDE_jenga_initial_blocks_jenga_game_proof_l3505_350564

theorem jenga_initial_blocks (players : ℕ) (complete_rounds : ℕ) (blocks_removed_last_round : ℕ) (blocks_remaining : ℕ) : ℕ :=
  let blocks_removed_complete_rounds := players * complete_rounds
  let total_blocks_removed := blocks_removed_complete_rounds + blocks_removed_last_round
  let initial_blocks := total_blocks_removed + blocks_remaining
  initial_blocks

theorem jenga_game_proof :
  jenga_initial_blocks 5 5 1 28 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jenga_initial_blocks_jenga_game_proof_l3505_350564


namespace NUMINAMATH_CALUDE_work_completion_time_l3505_350520

/-- Given a work that can be completed by person A in 20 days, and 0.375 of the work
    can be completed by A and B together in 5 days, prove that person B can complete
    the work alone in 40 days. -/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 20 →
  5 * (work_rate_A + work_rate_B) = 0.375 →
  1 / work_rate_B = 40 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3505_350520


namespace NUMINAMATH_CALUDE_system_solution_l3505_350536

theorem system_solution (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 2*b)
  (eq3 : a + b = 9 - 5*c) :
  3*a + 3*b + 3*c = 18/17 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3505_350536


namespace NUMINAMATH_CALUDE_total_spend_l3505_350570

-- Define the given conditions
def num_tshirts : ℕ := 3
def cost_per_tshirt : ℕ := 20
def cost_pants : ℕ := 50

-- State the theorem
theorem total_spend : 
  num_tshirts * cost_per_tshirt + cost_pants = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_spend_l3505_350570


namespace NUMINAMATH_CALUDE_expression_factorization_l3505_350523

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6) = 5 * x * (3 * x^2 + 17) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3505_350523


namespace NUMINAMATH_CALUDE_nth_equation_l3505_350559

theorem nth_equation (n : ℕ) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3505_350559


namespace NUMINAMATH_CALUDE_cake_division_theorem_l3505_350528

/-- Represents a piece of cake -/
structure CakePiece where
  cookies : ℕ
  roses : ℕ

/-- Represents the whole cake -/
structure Cake where
  totalCookies : ℕ
  totalRoses : ℕ
  pieces : ℕ

/-- Checks if a cake can be evenly divided -/
def isEvenlyDivisible (c : Cake) : Prop :=
  c.totalCookies % c.pieces = 0 ∧ c.totalRoses % c.pieces = 0

/-- Calculates the content of each piece when the cake is evenly divided -/
def pieceContent (c : Cake) (h : isEvenlyDivisible c) : CakePiece :=
  { cookies := c.totalCookies / c.pieces
  , roses := c.totalRoses / c.pieces }

/-- Theorem: If a cake with 48 cookies and 4 roses is cut into 4 equal pieces,
    each piece will have 12 cookies and 1 rose -/
theorem cake_division_theorem (c : Cake)
    (h1 : c.totalCookies = 48)
    (h2 : c.totalRoses = 4)
    (h3 : c.pieces = 4)
    (h4 : isEvenlyDivisible c) :
    pieceContent c h4 = { cookies := 12, roses := 1 } := by
  sorry


end NUMINAMATH_CALUDE_cake_division_theorem_l3505_350528


namespace NUMINAMATH_CALUDE_distance_from_origin_l3505_350572

theorem distance_from_origin (x y : ℝ) (h1 : |y| = 15) (h2 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3505_350572


namespace NUMINAMATH_CALUDE_calculation_proof_l3505_350580

theorem calculation_proof : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3505_350580


namespace NUMINAMATH_CALUDE_total_kittens_l3505_350501

/-- Given an initial number of kittens and additional kittens, 
    prove that the total number of kittens is their sum. -/
theorem total_kittens (initial additional : ℕ) :
  initial + additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_total_kittens_l3505_350501


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3505_350589

theorem interest_rate_difference (principal : ℝ) (original_rate higher_rate : ℝ) 
  (h1 : principal = 500)
  (h2 : principal * higher_rate / 100 - principal * original_rate / 100 = 30) :
  higher_rate - original_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3505_350589


namespace NUMINAMATH_CALUDE_finite_steps_33_disks_infinite_steps_32_disks_l3505_350586

/-- Represents a board with disks -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (disks : Nat)

/-- Represents a move on the board -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the state of the game after some number of steps -/
structure GameState :=
  (board : Board)
  (step : Nat)

/-- Predicate to check if a game state is valid -/
def isValid (state : GameState) : Prop :=
  state.board.disks ≤ state.board.rows * state.board.cols

/-- Predicate to check if a move is valid given the previous move -/
def isValidMove (prevMove : Option Move) (currMove : Move) : Prop :=
  match prevMove with
  | none => true
  | some Move.Up => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Down => currMove = Move.Left ∨ currMove = Move.Right
  | some Move.Left => currMove = Move.Up ∨ currMove = Move.Down
  | some Move.Right => currMove = Move.Up ∨ currMove = Move.Down

/-- Theorem: With 33 disks on a 5x9 board, only finitely many steps are possible -/
theorem finite_steps_33_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 33) :
  ∃ n : Nat, ∀ state : GameState, state.board = board → state.step > n → ¬isValid state :=
sorry

/-- Theorem: With 32 disks on a 5x9 board, infinitely many steps are possible -/
theorem infinite_steps_32_disks (board : Board) (h : board.rows = 5 ∧ board.cols = 9 ∧ board.disks = 32) :
  ∀ n : Nat, ∃ state : GameState, state.board = board ∧ state.step = n ∧ isValid state :=
sorry

end NUMINAMATH_CALUDE_finite_steps_33_disks_infinite_steps_32_disks_l3505_350586


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l3505_350539

open Real

theorem trigonometric_calculations :
  (sin (-60 * π / 180) = -Real.sqrt 3 / 2) ∧
  (cos (-45 * π / 180) = Real.sqrt 2 / 2) ∧
  (tan (-945 * π / 180) = -1) := by
  sorry

-- Definitions and properties used in the proof
axiom sine_odd (x : ℝ) : sin (-x) = -sin x
axiom cosine_even (x : ℝ) : cos (-x) = cos x
axiom tangent_odd (x : ℝ) : tan (-x) = -tan x
axiom sin_60 : sin (60 * π / 180) = Real.sqrt 3 / 2
axiom cos_45 : cos (45 * π / 180) = Real.sqrt 2 / 2
axiom tan_45 : tan (45 * π / 180) = 1
axiom tan_period (x : ℝ) (k : ℤ) : tan (x + k * π) = tan x

end NUMINAMATH_CALUDE_trigonometric_calculations_l3505_350539


namespace NUMINAMATH_CALUDE_robie_has_five_boxes_l3505_350511

/-- Calculates the number of boxes Robie has left after giving some away -/
def robies_boxes (total_cards : ℕ) (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) : ℕ :=
  ((total_cards - unboxed_cards) / cards_per_box) - boxes_given_away

/-- Theorem stating that Robie has 5 boxes left given the initial conditions -/
theorem robie_has_five_boxes :
  robies_boxes 75 10 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robie_has_five_boxes_l3505_350511


namespace NUMINAMATH_CALUDE_validSquaresCount_l3505_350585

/-- Represents a square on the checkerboard -/
structure Square :=
  (x : Nat) -- x-coordinate of the top-left corner
  (y : Nat) -- y-coordinate of the top-left corner
  (size : Nat) -- side length of the square

/-- Defines the 10x10 checkerboard -/
def checkerboard : Nat := 10

/-- Checks if a square contains at least 8 black squares -/
def hasAtLeast8BlackSquares (s : Square) : Bool :=
  -- Implementation details omitted
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that there are exactly 115 valid squares -/
theorem validSquaresCount : countValidSquares = 115 := by
  sorry

end NUMINAMATH_CALUDE_validSquaresCount_l3505_350585


namespace NUMINAMATH_CALUDE_adele_age_fraction_l3505_350526

/-- Given the ages of Jackson, Mandy, and Adele, prove that Adele's age is 3/4 of Jackson's age. -/
theorem adele_age_fraction (jackson_age mandy_age adele_age : ℕ) : 
  jackson_age = 20 →
  mandy_age = jackson_age + 10 →
  (jackson_age + 10) + (mandy_age + 10) + (adele_age + 10) = 95 →
  ∃ f : ℚ, adele_age = f * jackson_age ∧ f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_adele_age_fraction_l3505_350526


namespace NUMINAMATH_CALUDE_solar_usage_exponential_growth_l3505_350542

/-- Represents the percentage of households using solar energy -/
def SolarUsage : ℕ → ℝ
  | 2000 => 6
  | 2010 => 12
  | 2015 => 24
  | 2020 => 48
  | _ => 0  -- For years not specified, we return 0

/-- Checks if the growth is exponential between two time points -/
def IsExponentialGrowth (t₁ t₂ : ℕ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ SolarUsage t₂ = SolarUsage t₁ * r^(t₂ - t₁)

/-- Theorem stating that the solar usage growth is exponential -/
theorem solar_usage_exponential_growth :
  IsExponentialGrowth 2000 2010 ∧
  IsExponentialGrowth 2010 2015 ∧
  IsExponentialGrowth 2015 2020 :=
sorry

end NUMINAMATH_CALUDE_solar_usage_exponential_growth_l3505_350542


namespace NUMINAMATH_CALUDE_cookies_left_l3505_350566

def dozen : ℕ := 12

theorem cookies_left (total : ℕ) (eaten_percent : ℚ) (h1 : total = 2 * dozen) (h2 : eaten_percent = 1/4) :
  total - (eaten_percent * total).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3505_350566


namespace NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3505_350576

/-- Calculates the number of weeks needed to complete bridesmaid dresses -/
def weeks_to_complete_dresses (hours_per_dress : ℕ) (num_bridesmaids : ℕ) (hours_per_week : ℕ) : ℕ :=
  (hours_per_dress * num_bridesmaids) / hours_per_week

/-- Proves that it takes 15 weeks to complete the bridesmaid dresses under given conditions -/
theorem bridesmaid_dresses_completion_time :
  weeks_to_complete_dresses 12 5 4 = 15 := by
  sorry

#eval weeks_to_complete_dresses 12 5 4

end NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3505_350576


namespace NUMINAMATH_CALUDE_system_solution_proof_l3505_350569

theorem system_solution_proof :
  let x₁ : ℚ := 3/2
  let x₂ : ℚ := 1/2
  (3 * x₁ - 5 * x₂ = 2) ∧ (2 * x₁ + 4 * x₂ = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l3505_350569


namespace NUMINAMATH_CALUDE_even_function_coefficient_l3505_350534

theorem even_function_coefficient (a : ℝ) :
  (∀ x : ℝ, (fun x => x^2 + a*x + 1) x = (fun x => x^2 + a*x + 1) (-x)) →
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_coefficient_l3505_350534


namespace NUMINAMATH_CALUDE_share_distribution_l3505_350553

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 378 →
  total = a + b + c →
  12 * a = 8 * b →
  12 * a = 6 * c →
  a = 84 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l3505_350553


namespace NUMINAMATH_CALUDE_last_number_is_odd_l3505_350568

/-- The operation of choosing two numbers and replacing them with their absolute difference -/
def boardOperation (numbers : List Int) : List Int :=
  sorry

/-- The process of repeatedly applying the operation until only one number remains -/
def boardProcess (initialNumbers : List Int) : Int :=
  sorry

/-- The list of integers from 1 to 2018 -/
def initialBoard : List Int :=
  List.range 2018

theorem last_number_is_odd :
  Odd (boardProcess initialBoard) :=
by sorry

end NUMINAMATH_CALUDE_last_number_is_odd_l3505_350568


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3505_350525

/-- Given two vectors a and b in R², prove that if 2a is parallel to b, then the x-coordinate of b is -4. -/
theorem parallel_vectors_imply_x_value (a b : ℝ × ℝ) : 
  a = (2, 3) → 
  b.2 = -6 → 
  ∃ (k : ℝ), (2 * a.1, 2 * a.2) = (k * b.1, k * b.2) → 
  b.1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3505_350525


namespace NUMINAMATH_CALUDE_jorge_total_goals_l3505_350562

theorem jorge_total_goals (last_season_goals this_season_goals : ℕ) 
  (h1 : last_season_goals = 156)
  (h2 : this_season_goals = 187) :
  last_season_goals + this_season_goals = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l3505_350562


namespace NUMINAMATH_CALUDE_intersection_theorem_l3505_350533

def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℝ := {x | x + 1 < 0}

theorem intersection_theorem :
  A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3505_350533


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l3505_350532

theorem nested_square_root_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l3505_350532


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3505_350537

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 4 * x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 / 2 ∧
              x₂ = -1 - Real.sqrt 6 / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3505_350537


namespace NUMINAMATH_CALUDE_card_arrangement_count_l3505_350579

/-- The number of ways to arrange 6 cards into 3 envelopes -/
def arrangement_count : ℕ := 18

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Cards 1 and 2 are in the same envelope -/
def cards_1_2_together : Prop := True

theorem card_arrangement_count :
  arrangement_count = num_envelopes * (num_cards - cards_per_envelope).choose cards_per_envelope :=
sorry

end NUMINAMATH_CALUDE_card_arrangement_count_l3505_350579


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l3505_350531

/-- Proves that given an escalator moving at 12 ft/sec with a length of 160 feet,
    if a person covers the entire length in 8 seconds,
    then the person's walking speed on the escalator is 8 ft/sec. -/
theorem escalator_walking_speed
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 160)
  (h3 : time_taken = 8)
  (h4 : escalator_length = (person_speed + escalator_speed) * time_taken) :
  person_speed = 8 := by
  sorry

#check escalator_walking_speed

end NUMINAMATH_CALUDE_escalator_walking_speed_l3505_350531


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3505_350556

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) ∧ 1/a + 9/b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3505_350556


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3505_350554

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.2 * L
  let new_surface_area := 6 * new_edge_length^2
  let percentage_increase := (new_surface_area - original_surface_area) / original_surface_area * 100
  percentage_increase = 44 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3505_350554


namespace NUMINAMATH_CALUDE_abc_inequality_l3505_350535

theorem abc_inequality (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) > 
  a^(α + 2) * (b + c - a) + b^(α + 2) * (a - b + c) + c^(α + 2) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3505_350535


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l3505_350551

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l3505_350551


namespace NUMINAMATH_CALUDE_science_team_selection_l3505_350547

def number_of_boys : ℕ := 7
def number_of_girls : ℕ := 9
def boys_in_team : ℕ := 2
def girls_in_team : ℕ := 3

theorem science_team_selection :
  (number_of_boys.choose boys_in_team) * (number_of_girls.choose girls_in_team) = 1764 := by
  sorry

end NUMINAMATH_CALUDE_science_team_selection_l3505_350547


namespace NUMINAMATH_CALUDE_kayak_production_sum_l3505_350558

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 6  -- First term (February production)
  let r := 3  -- Common ratio
  let n := 5  -- Number of months (February to June)
  geometric_sum a r n = 726 := by
sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l3505_350558


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3505_350519

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3505_350519


namespace NUMINAMATH_CALUDE_division_chain_l3505_350587

theorem division_chain : (132 / 6) / 2 = 11 := by sorry

end NUMINAMATH_CALUDE_division_chain_l3505_350587


namespace NUMINAMATH_CALUDE_cutlery_theorem_l3505_350573

def cutlery_count (initial_knives : ℕ) : ℕ :=
  let initial_teaspoons := 2 * initial_knives
  let additional_knives := initial_knives / 3
  let additional_teaspoons := (2 * initial_teaspoons) / 3
  let total_knives := initial_knives + additional_knives
  let total_teaspoons := initial_teaspoons + additional_teaspoons
  total_knives + total_teaspoons

theorem cutlery_theorem : cutlery_count 24 = 112 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_theorem_l3505_350573


namespace NUMINAMATH_CALUDE_project_completion_time_l3505_350522

/-- Given a project that A can complete in 20 days, and A and B together can complete in 15 days
    with A quitting 5 days before completion, prove that B can complete the project alone in 30 days. -/
theorem project_completion_time (a_rate b_rate : ℚ) : 
  a_rate = (1 : ℚ) / 20 →                          -- A's work rate
  a_rate + b_rate = (1 : ℚ) / 15 →                 -- A and B's combined work rate
  10 * (a_rate + b_rate) + 5 * b_rate = 1 →        -- Total work done
  b_rate = (1 : ℚ) / 30                            -- B's work rate (reciprocal of completion time)
  := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l3505_350522


namespace NUMINAMATH_CALUDE_walter_hushpuppies_per_guest_l3505_350581

/-- Calculates the number of hushpuppies per guest given the number of guests,
    cooking rate, and total cooking time. -/
def hushpuppies_per_guest (guests : ℕ) (hushpuppies_per_batch : ℕ) 
    (minutes_per_batch : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_batch * hushpuppies_per_batch) / guests

/-- Proves that given the specified conditions, each guest will eat 5 hushpuppies. -/
theorem walter_hushpuppies_per_guest : 
  hushpuppies_per_guest 20 10 8 80 = 5 := by
  sorry

end NUMINAMATH_CALUDE_walter_hushpuppies_per_guest_l3505_350581


namespace NUMINAMATH_CALUDE_remainder_theorem_l3505_350516

theorem remainder_theorem : 
  (86592 : ℤ) % 8 = 0 ∧ (8741 : ℤ) % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3505_350516


namespace NUMINAMATH_CALUDE_consecutive_sum_equality_l3505_350505

theorem consecutive_sum_equality :
  ∃ (a b : ℕ), 
    a ≥ 1 ∧ 
    5 * (a + 2) = 2 * b + 1 ∧ 
    ∀ (x : ℕ), x < a → ¬∃ (y : ℕ), 5 * (x + 2) = 2 * y + 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_equality_l3505_350505


namespace NUMINAMATH_CALUDE_line_arrangements_with_restriction_l3505_350599

def number_of_students : ℕ := 5

def number_of_restricted_students : ℕ := 2

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_restricted_together (n : ℕ) (r : ℕ) : ℕ :=
  (Nat.factorial (n - r + 1)) * (Nat.factorial r)

theorem line_arrangements_with_restriction :
  total_arrangements number_of_students - 
  arrangements_with_restricted_together number_of_students number_of_restricted_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangements_with_restriction_l3505_350599


namespace NUMINAMATH_CALUDE_jays_change_l3505_350548

def book_cost : ℕ := 25
def pen_cost : ℕ := 4
def ruler_cost : ℕ := 1
def amount_paid : ℕ := 50

def total_cost : ℕ := book_cost + pen_cost + ruler_cost

theorem jays_change (change : ℕ) : change = amount_paid - total_cost → change = 20 := by
  sorry

end NUMINAMATH_CALUDE_jays_change_l3505_350548


namespace NUMINAMATH_CALUDE_mean_value_theorem_for_f_l3505_350574

-- Define the function f(x) = x² + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

theorem mean_value_theorem_for_f :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 2,
    f 2 - f (-1) = f' c * (2 - (-1)) ∧
    c = 1 / 2 := by
  sorry

#check mean_value_theorem_for_f

end NUMINAMATH_CALUDE_mean_value_theorem_for_f_l3505_350574


namespace NUMINAMATH_CALUDE_second_square_width_is_seven_l3505_350524

/-- Represents the dimensions of a rectangular piece of fabric -/
structure Fabric where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular piece of fabric -/
def area (f : Fabric) : ℝ := f.length * f.width

/-- Represents the three pieces of fabric and the desired flag dimensions -/
structure FlagProblem where
  fabric1 : Fabric
  fabric2 : Fabric
  fabric3 : Fabric
  flagLength : ℝ
  flagHeight : ℝ

/-- The flag problem with given dimensions -/
def bobbysProblem : FlagProblem :=
  { fabric1 := { length := 8, width := 5 }
  , fabric2 := { length := 10, width := 7 }  -- We'll prove this width
  , fabric3 := { length := 5, width := 5 }
  , flagLength := 15
  , flagHeight := 9
  }

theorem second_square_width_is_seven :
  let p := bobbysProblem
  area p.fabric1 + area p.fabric2 + area p.fabric3 = p.flagLength * p.flagHeight ∧
  p.fabric2.width = 7 := by
  sorry


end NUMINAMATH_CALUDE_second_square_width_is_seven_l3505_350524


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3505_350538

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 1 = f 3 ∧ f 1 > f 4) → (a < 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3505_350538


namespace NUMINAMATH_CALUDE_female_students_count_l3505_350521

theorem female_students_count (total : ℕ) (sample : ℕ) (female_diff : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_female_diff : female_diff = 20)
  (h_ratio : (sample - female_diff) / (sample + female_diff) = 9 / 11) :
  ∃ F : ℕ, F = 720 ∧ F + (total - F) = total := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l3505_350521


namespace NUMINAMATH_CALUDE_cos_2theta_minus_7pi_over_2_l3505_350543

theorem cos_2theta_minus_7pi_over_2 (θ : ℝ) (h : Real.sin θ + Real.cos θ = -Real.sqrt 5 / 3) :
  Real.cos (2 * θ - 7 * Real.pi / 2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_minus_7pi_over_2_l3505_350543


namespace NUMINAMATH_CALUDE_statement_equivalence_l3505_350500

theorem statement_equivalence (x y : ℝ) : 
  ((abs y < abs x) ↔ (x^2 > y^2)) ∧ 
  ((x^3 - y^3 = 0) ↔ (x - y = 0)) ∧ 
  ((x^3 - y^3 ≠ 0) ↔ (x - y ≠ 0)) ∧ 
  ¬((x^2 - y^2 ≠ 0 ∧ x^3 - y^3 ≠ 0) ↔ (x^2 - y^2 ≠ 0 ∨ x^3 - y^3 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3505_350500


namespace NUMINAMATH_CALUDE_ballpoint_pen_price_relation_l3505_350540

/-- Proves the relationship between price and number of pens for a specific box of ballpoint pens -/
theorem ballpoint_pen_price_relation :
  let box_pens : ℕ := 16
  let box_price : ℚ := 24
  let unit_price : ℚ := box_price / box_pens
  ∀ (x : ℚ) (y : ℚ), y = unit_price * x → y = (3/2 : ℚ) * x := by
  sorry

end NUMINAMATH_CALUDE_ballpoint_pen_price_relation_l3505_350540


namespace NUMINAMATH_CALUDE_tan_period_l3505_350513

/-- The period of tan(3x/4) is 4π/3 -/
theorem tan_period (x : ℝ) : 
  (fun x => Real.tan ((3 : ℝ) * x / 4)) = (fun x => Real.tan ((3 : ℝ) * (x + 4 * Real.pi / 3) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_tan_period_l3505_350513


namespace NUMINAMATH_CALUDE_sameGradePercentage_is_32_percent_l3505_350592

/-- Represents the number of students who received the same grade on both tests for each grade. -/
structure SameGradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the percentage of students who received the same grade on both tests. -/
def sameGradePercentage (totalStudents : ℕ) (sameGrades : SameGradeCount) : ℚ :=
  let sameGradeTotal := sameGrades.a + sameGrades.b + sameGrades.c + sameGrades.d + sameGrades.e
  (sameGradeTotal : ℚ) / (totalStudents : ℚ) * 100

/-- Theorem stating that the percentage of students who received the same grade on both tests is 32%. -/
theorem sameGradePercentage_is_32_percent :
  let totalStudents := 50
  let sameGrades := SameGradeCount.mk 4 6 3 2 1
  sameGradePercentage totalStudents sameGrades = 32 := by
  sorry

end NUMINAMATH_CALUDE_sameGradePercentage_is_32_percent_l3505_350592


namespace NUMINAMATH_CALUDE_vector_angle_solution_l3505_350546

/-- Given two plane vectors a and b with unit length and 60° angle between them,
    prove that t = 0 is a valid solution when the angle between a+b and ta-b is obtuse. -/
theorem vector_angle_solution (a b : ℝ × ℝ) (t : ℝ) :
  (norm a = 1) →
  (norm b = 1) →
  (a • b = 1 / 2) →  -- cos 60° = 1/2
  ((a + b) • (t • a - b) < 0) →
  (t = 0) →
  True := by sorry

end NUMINAMATH_CALUDE_vector_angle_solution_l3505_350546


namespace NUMINAMATH_CALUDE_stack_height_three_pipes_l3505_350514

/-- The height of a stack of identical cylindrical pipes -/
def stack_height (num_pipes : ℕ) (pipe_diameter : ℝ) : ℝ :=
  (num_pipes : ℝ) * pipe_diameter

/-- Theorem: The height of a stack of three identical cylindrical pipes with a diameter of 12 cm is 36 cm -/
theorem stack_height_three_pipes :
  stack_height 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_three_pipes_l3505_350514


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3505_350515

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 129.6 := by
  sorry


end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3505_350515


namespace NUMINAMATH_CALUDE_derivative_at_zero_l3505_350594

/-- Given a function f where f(x) = x^2 + 2x * f'(1), prove that f'(0) = -4 -/
theorem derivative_at_zero (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2*x*(deriv f 1)) :
  deriv f 0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l3505_350594


namespace NUMINAMATH_CALUDE_number_of_payment_ways_l3505_350517

/-- Represents the number of ways to pay 16 rubles using 10-ruble, 2-ruble, and 1-ruble coins. -/
def payment_ways : ℕ := 13

/-- Represents the total amount to be paid in rubles. -/
def total_amount : ℕ := 16

/-- Represents the value of a 10-ruble coin. -/
def ten_ruble : ℕ := 10

/-- Represents the value of a 2-ruble coin. -/
def two_ruble : ℕ := 2

/-- Represents the value of a 1-ruble coin. -/
def one_ruble : ℕ := 1

/-- Represents the minimum number of coins of each type available. -/
def min_coins : ℕ := 21

/-- Theorem stating that the number of ways to pay 16 rubles is 13. -/
theorem number_of_payment_ways :
  payment_ways = (Finset.filter
    (fun n : ℕ × ℕ × ℕ => n.1 * ten_ruble + n.2.1 * two_ruble + n.2.2 * one_ruble = total_amount)
    (Finset.product (Finset.range (min_coins + 1))
      (Finset.product (Finset.range (min_coins + 1)) (Finset.range (min_coins + 1))))).card :=
by sorry

end NUMINAMATH_CALUDE_number_of_payment_ways_l3505_350517


namespace NUMINAMATH_CALUDE_park_to_grocery_distance_l3505_350557

/-- The distance from Talia's house to the park, in miles -/
def distance_house_to_park : ℝ := 5

/-- The distance from Talia's house to the grocery store, in miles -/
def distance_house_to_grocery : ℝ := 8

/-- The total distance Talia drives, in miles -/
def total_distance : ℝ := 16

/-- The distance from the park to the grocery store, in miles -/
def distance_park_to_grocery : ℝ := total_distance - distance_house_to_park - distance_house_to_grocery

theorem park_to_grocery_distance :
  distance_park_to_grocery = 3 := by sorry

end NUMINAMATH_CALUDE_park_to_grocery_distance_l3505_350557


namespace NUMINAMATH_CALUDE_divisibility_of_2_pow_55_plus_1_l3505_350582

theorem divisibility_of_2_pow_55_plus_1 : 
  ∃ k : ℤ, 2^55 + 1 = 33 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_2_pow_55_plus_1_l3505_350582


namespace NUMINAMATH_CALUDE_curve_not_parabola_l3505_350503

-- Define the curve equation
def curve_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x^2 + y^2 = 1

-- Define what it means for a curve to be a parabola
-- (This is a simplified definition for the purpose of this statement)
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem curve_not_parabola :
  ∀ k : ℝ, ¬(is_parabola (curve_equation k)) :=
sorry

end NUMINAMATH_CALUDE_curve_not_parabola_l3505_350503


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l3505_350578

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a * x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- Theorem for part (1)
theorem solution_set_when_a_eq_one :
  ∀ x : ℝ, (g x ≥ f 1 x) ↔ (x ≥ 1 ∨ x ≤ -3) :=
sorry

-- Theorem for part (2)
theorem range_of_a_given_f_geq_three_halves :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ f a x ≥ 3/2) →
  (∀ a : ℝ, a > 0 → f a x ≥ 3/2 → a ≥ Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l3505_350578


namespace NUMINAMATH_CALUDE_range_of_m_for_exponential_equation_l3505_350597

/-- The range of m for which the equation 9^(-x^x) = 4 * 3^(-x^x) + m has a real solution for x -/
theorem range_of_m_for_exponential_equation :
  ∀ m : ℝ, (∃ x : ℝ, (9 : ℝ)^(-x^x) = 4 * (3 : ℝ)^(-x^x) + m) ↔ -3 ≤ m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_exponential_equation_l3505_350597


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l3505_350596

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l3505_350596


namespace NUMINAMATH_CALUDE_unique_solution_is_seven_l3505_350577

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_is_seven :
  ∃! n : ℕ, n > 0 ∧ n^2 * factorial n + factorial n = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_seven_l3505_350577


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3505_350561

theorem opposite_of_negative_two :
  ∃ x : ℝ, (x + (-2) = 0 ∧ x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3505_350561


namespace NUMINAMATH_CALUDE_negative_square_cubed_l3505_350510

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l3505_350510


namespace NUMINAMATH_CALUDE_range_of_a_l3505_350595

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) → 
  a ≥ 1 ∨ a ≤ -3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3505_350595


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l3505_350508

/-- Given a function f(x) = ax^3 - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem cubic_function_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l3505_350508


namespace NUMINAMATH_CALUDE_banana_permutations_count_l3505_350544

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_count :
  banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_count_l3505_350544


namespace NUMINAMATH_CALUDE_log_5_18_l3505_350584

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : Real.log 2 / Real.log 10 = a)
variable (h2 : Real.log 3 / Real.log 10 = b)

-- State the theorem to be proved
theorem log_5_18 : Real.log 18 / Real.log 5 = (a + 2*b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_5_18_l3505_350584


namespace NUMINAMATH_CALUDE_kitten_weight_l3505_350590

theorem kitten_weight (kitten smaller_dog larger_dog : ℝ) 
  (total_weight : kitten + smaller_dog + larger_dog = 36)
  (larger_comparison : kitten + larger_dog = 2 * smaller_dog)
  (smaller_comparison : kitten + smaller_dog = larger_dog) :
  kitten = 9 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l3505_350590


namespace NUMINAMATH_CALUDE_tape_length_calculation_l3505_350509

/-- Calculates the length of tape wrapped around a cylindrical core -/
theorem tape_length_calculation 
  (initial_diameter : ℝ) 
  (tape_width : ℝ) 
  (num_wraps : ℕ) 
  (final_diameter : ℝ) 
  (h1 : initial_diameter = 4)
  (h2 : tape_width = 4)
  (h3 : num_wraps = 800)
  (h4 : final_diameter = 16) :
  (π / 2) * (initial_diameter + final_diameter) * num_wraps = 80 * π := by
  sorry

#check tape_length_calculation

end NUMINAMATH_CALUDE_tape_length_calculation_l3505_350509


namespace NUMINAMATH_CALUDE_division_problem_l3505_350507

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5/2) :
  z / x = 2/15 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3505_350507


namespace NUMINAMATH_CALUDE_rod_cutting_l3505_350591

theorem rod_cutting (rod_length : ℝ) (piece_length : ℚ) : 
  rod_length = 58.75 →
  piece_length = 137 + 2/3 →
  ⌊(rod_length * 100) / (piece_length : ℝ)⌋ = 14 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l3505_350591


namespace NUMINAMATH_CALUDE_ascending_order_proof_l3505_350504

theorem ascending_order_proof :
  222^2 < 2^(2^(2^2)) ∧
  2^(2^(2^2)) < 22^(2^2) ∧
  22^(2^2) < 22^22 ∧
  22^22 < 2^222 ∧
  2^222 < 2^(22^2) ∧
  2^(22^2) < 2^(2^22) := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l3505_350504


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3505_350583

theorem bowling_ball_weight (b c k : ℝ) 
  (h1 : 9 * b = 6 * c)
  (h2 : c + k = 42)
  (h3 : 3 * k = 2 * c) :
  b = 16.8 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3505_350583


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3505_350588

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetric_x (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

theorem symmetry_coordinates :
  let A : Point := ⟨1, 2⟩
  let A' : Point := ⟨1, -2⟩
  symmetric_x A A' :=
by sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3505_350588


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_6_l3505_350502

theorem gcf_lcm_sum_4_6 : Nat.gcd 4 6 + Nat.lcm 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_6_l3505_350502


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l3505_350506

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) (chosen : ℕ) 
  (h1 : total = 20) (h2 : defective = 4) (h3 : chosen = 2) :
  (1 : ℚ) - (Nat.choose (total - defective) chosen : ℚ) / (Nat.choose total chosen : ℚ) = 7/19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l3505_350506


namespace NUMINAMATH_CALUDE_vector_at_zero_l3505_350565

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 3 → ℝ

/-- The vector at a given parameter value -/
def vectorAt (line : ParameterizedLine) (t : ℝ) : Fin 3 → ℝ := line.vector t

theorem vector_at_zero (line : ParameterizedLine) 
  (h1 : vectorAt line 1 = ![2, 4, 9])
  (h2 : vectorAt line (-1) = ![-1, 1, 2]) :
  vectorAt line 0 = ![1/2, 5/2, 11/2] := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_l3505_350565


namespace NUMINAMATH_CALUDE_approximate_root_exists_l3505_350575

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_exists :
  ∃ (r : ℝ), r ∈ Set.Icc 1.375 1.4375 ∧ f r = 0 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 1.375 1.4375 → |x - 1.42| ≤ 0.05 := by
  sorry

#check approximate_root_exists

end NUMINAMATH_CALUDE_approximate_root_exists_l3505_350575


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l3505_350512

theorem power_function_not_through_origin (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (m^2 - 3*m + 3) * x^(m^2 - m - 2)
  (∀ x ≠ 0, f x ≠ 0) → (m = 1 ∨ m = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l3505_350512


namespace NUMINAMATH_CALUDE_always_less_than_log_sum_implies_less_than_one_l3505_350598

theorem always_less_than_log_sum_implies_less_than_one (a : ℝ) : 
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_always_less_than_log_sum_implies_less_than_one_l3505_350598


namespace NUMINAMATH_CALUDE_smallest_excluded_number_l3505_350567

theorem smallest_excluded_number : ∃ n : ℕ, 
  (∀ k ∈ Finset.range 200, k + 1 ≠ 128 ∧ k + 1 ≠ 129 → n % (k + 1) = 0) ∧
  (∀ m : ℕ, m < 128 → 
    ¬∃ n : ℕ, (∀ k ∈ Finset.range 200, k + 1 ≠ m ∧ k + 1 ≠ m + 1 → n % (k + 1) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_excluded_number_l3505_350567


namespace NUMINAMATH_CALUDE_zach_monday_miles_l3505_350555

/-- Calculates the number of miles driven on Monday given the rental conditions and total cost --/
def miles_driven_monday (flat_fee : ℚ) (cost_per_mile : ℚ) (thursday_miles : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - flat_fee - cost_per_mile * thursday_miles) / cost_per_mile

/-- Proves that Zach drove 620 miles on Monday given the rental conditions and total cost --/
theorem zach_monday_miles :
  let flat_fee : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let thursday_miles : ℚ := 744
  let total_cost : ℚ := 832
  miles_driven_monday flat_fee cost_per_mile thursday_miles total_cost = 620 := by
  sorry

end NUMINAMATH_CALUDE_zach_monday_miles_l3505_350555


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l3505_350527

/-- Represents a mapping of letters to digits -/
def LetterMapping := Char → Fin 10

/-- Check if a mapping is valid for the cryptarithm -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m 'Г' ≠ 0 ∧
  m 'О' ≠ 0 ∧
  m 'В' ≠ 0 ∧
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧
  (m 'Г' * 1000 + m 'О' * 100 + m 'Р' * 10 + m 'А') +
  (m 'О' * 10000 + m 'Г' * 1000 + m 'О' * 100 + m 'Н' * 10 + m 'Ь') =
  (m 'В' * 100000 + m 'У' * 10000 + m 'Л' * 1000 + m 'К' * 100 + m 'А' * 10 + m 'Н')

theorem cryptarithm_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧
    m 'Г' = 6 ∧ m 'О' = 9 ∧ m 'Р' = 4 ∧ m 'А' = 7 ∧
    m 'Н' = 2 ∧ m 'Ь' = 5 ∧
    m 'В' = 1 ∧ m 'У' = 0 ∧ m 'Л' = 3 ∧ m 'К' = 8 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l3505_350527


namespace NUMINAMATH_CALUDE_function_property_l3505_350529

def is_positive_integer (x : ℝ) : Prop := ∃ n : ℕ, x = n ∧ n > 0

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y → f x < f y)  -- monotonically increasing
  (h2 : ∀ n : ℕ, n > 0 → is_positive_integer (f n))  -- f(n) is a positive integer for positive integer n
  (h3 : ∀ n : ℕ, n > 0 → f (f n) = 2 * n + 1)  -- f(f(n)) = 2n + 1 for positive integer n
  : f 1 = 2 ∧ f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3505_350529


namespace NUMINAMATH_CALUDE_no_real_roots_l3505_350563

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 5) - Real.sqrt (x - 2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3505_350563


namespace NUMINAMATH_CALUDE_base_7_to_10_23456_l3505_350571

def base_7_to_10 (d₁ d₂ d₃ d₄ d₅ : ℕ) : ℕ :=
  d₁ * 7^4 + d₂ * 7^3 + d₃ * 7^2 + d₄ * 7^1 + d₅ * 7^0

theorem base_7_to_10_23456 :
  base_7_to_10 2 3 4 5 6 = 6068 := by sorry

end NUMINAMATH_CALUDE_base_7_to_10_23456_l3505_350571


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3505_350518

theorem sum_of_distinct_prime_factors : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → p ∣ (7^7 - 7^4) ↔ p ∈ s) ∧
  (s.sum id = 24) := by
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l3505_350518


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l3505_350550

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 13*x - 30 = 0) → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y - 30 = 0) → 
  (x = -2 ∨ x = 15) ∧ 
  (∀ y : ℝ, y^2 - 13*y - 30 = 0 → y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l3505_350550


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3505_350552

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ - 12 = 0 ∧ x₂^2 - 4*x₂ - 12 = 0 ∧ x₁ = 6 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3505_350552


namespace NUMINAMATH_CALUDE_solution_set_and_inequality_l3505_350560

-- Define the set T
def T : Set ℝ := {t | t > 1}

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

theorem solution_set_and_inequality :
  -- Part 1: T is the set of all t for which |x-2|+|x-3| < t has a non-empty solution set
  (∀ t : ℝ, (∃ x : ℝ, f x < t) ↔ t ∈ T) ∧
  -- Part 2: For all a, b ∈ T, ab + 1 > a + b
  (∀ a b : ℝ, a ∈ T → b ∈ T → a * b + 1 > a + b) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_inequality_l3505_350560


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l3505_350593

/-- Given a circle with circumference 80 feet, proves that an arc corresponding
    to a central angle of 45° has a length of 10 feet. -/
theorem arc_length_45_degrees (circle : Real) (arc : Real) : 
  circle = 80 → -- The circumference of the circle is 80 feet
  arc = circle * (45 / 360) → -- The arc length is proportional to its central angle (45°)
  arc = 10 := by -- The arc length is 10 feet
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l3505_350593


namespace NUMINAMATH_CALUDE_school_play_scenes_l3505_350545

theorem school_play_scenes (Tom Ben Sam Nick Chris : ℕ) : 
  Tom = 8 ∧ Chris = 5 ∧ 
  Ben > Chris ∧ Ben < Tom ∧
  Sam > Chris ∧ Sam < Tom ∧
  Nick > Chris ∧ Nick < Tom ∧
  (∀ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) ∧
  (∀ pair : ℕ × ℕ, pair.1 ≠ pair.2 → pair.1 ≤ 5 ∧ pair.2 ≤ 5 → 
    ∃ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) →
  (Tom + Ben + Sam + Nick + Chris) / 2 = 16 := by
sorry

end NUMINAMATH_CALUDE_school_play_scenes_l3505_350545


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3505_350530

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, |x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x < 0 ∧ |x - 1| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3505_350530


namespace NUMINAMATH_CALUDE_min_projection_value_l3505_350541

theorem min_projection_value (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  let cos_theta := dot_product / (norm_a * norm_b)
  norm_a = Real.sqrt 6 ∧ 
  ((a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2) = ((3 * a.1 - 4 * b.1) ^ 2 + (3 * a.2 - 4 * b.2) ^ 2) →
  ∃ (min_value : ℝ), min_value = 12 / 7 ∧ ∀ θ : ℝ, norm_a * |cos_theta| ≥ min_value := by
sorry

end NUMINAMATH_CALUDE_min_projection_value_l3505_350541


namespace NUMINAMATH_CALUDE_games_per_box_l3505_350549

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) :
  initial_games = 76 →
  sold_games = 46 →
  num_boxes = 6 →
  (initial_games - sold_games) / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_per_box_l3505_350549
